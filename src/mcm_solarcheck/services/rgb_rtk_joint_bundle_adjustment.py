"""Private metric-candidate RTK-constrained windowed RGB bundle adjustment.

Uses the existing SfM tracks, prior windowed BA, robust GNSS similarity,
and the Schur/window optimizer with fitted ENU camera-position priors.
Independent RTK hold-outs are not independent survey/GCP measurements.
No customer GPS, images, paths or numerical coordinates are published.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from mcm_solarcheck.services.rgb_global_camera_network import RGBUnscaledNetwork
from mcm_solarcheck.services.rgb_windowed_bundle_adjustment import (
    RGBWindowedBAResult, adjust_windowed_unscaled_rgb_network,
)
from mcm_solarcheck.services.rgb_rtk_camera_priors import (
    PrivateRGBRTKPriors, anchor_windowed_rgb_network_to_rtk,
)
from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_per_camera_reprojection_gate import (
    RGBPerCameraReprojectionAudit, audit_rgb_per_camera_reprojection,
)
from mcm_solarcheck.services.rgb_multiview_parallax_gate import (
    RGBMultiviewParallaxAudit, audit_rgb_multiview_parallax,
)
from mcm_solarcheck.services.rgb_private_pitched_plane_candidates import (
    PrivateRGBPitchedPlaneCandidates, propose_private_rgb_pitched_planes,
)


@dataclass(frozen=True)
class PrivateRGBRTKWindowedBA:
    """In-memory ENU candidate only: not a metric product or roof orthophoto."""

    frame_ids: tuple[str, ...]
    world_enu_to_camera_private: tuple[np.ndarray, ...]
    landmarks_enu_private_m: np.ndarray
    before_reprojection_rmse_px: float
    after_reprojection_rmse_px: float
    before_rtk_holdout_p95_m: float
    after_rtk_holdout_p95_m: float
    after_rtk_holdout_max_m: float
    fitted_rtk_camera_count: int
    withheld_rtk_camera_count: int
    total_cameras: int
    completed_windows: int
    joint_objective_before: float
    joint_objective_after: float
    windows_with_updates: int = 0
    windows_without_improvement: int = 0
    state: str = "rtk_regularized_multiview_windowed_ba_unverified"
    rtk_prior_windowed_ba_completed: bool = True
    full_global_joint_ba_completed: bool = False
    independent_survey_verified: bool = False
    roof_geometry_verified: bool = False
    orthophoto_verified: bool = False
    camera_reprojection_audit: RGBPerCameraReprojectionAudit | None = None
    multiview_parallax_audit: RGBMultiviewParallaxAudit | None = None
    pitched_plane_candidates: PrivateRGBPitchedPlaneCandidates | None = None
    refined_intrinsics: PinholeIntrinsics | None = None
    focal_refined: bool = False
    focal_reference_intrinsics: PinholeIntrinsics | None = None

    def render_private_pitched_plane_preview(
        self, candidate_index: int,
        undistorted_rgb_images: tuple[np.ndarray, ...],
        camera_indices: tuple[int, ...],
        intrinsics: PinholeIntrinsics,
        *, meters_per_pixel: float = .05, max_side: int = 1024,
        should_cancel=None,
    ):
        """Vertical BA->plane->RGB raster preview, PRIVATE memory only.

        Supply RGBs sampled under the final camera model and undistorted when
        that model includes distortion. An external calibration file is not
        required for this unverified preview. No roof/GCP validation is implied.
        """
        from mcm_solarcheck.services.rgb_private_pitched_plane_orthorectification import (
            render_private_pitched_rgb_orthopatch,
        )
        intrinsics = self.refined_intrinsics or intrinsics
        if (self.pitched_plane_candidates is None
                or type(candidate_index) is not int
                or not 0 <= candidate_index < len(
                    self.pitched_plane_candidates.planes_private
                )
                or type(camera_indices) is not tuple
                or not 2 <= len(camera_indices) <= 8
                or len(set(camera_indices)) != len(camera_indices)
                or any(type(i) is not int or i < 0
                       or i >= self.total_cameras for i in camera_indices)
                or self.camera_reprojection_audit is None
                or self.camera_reprojection_audit.cameras_above_p95_limit
                or self.multiview_parallax_audit is None
                or not self.multiview_parallax_audit.privacy_safe_summary()[
                    "parallax_quality_passed"
                ]):
            raise ValueError("valid fitted RGB plane and verified stereo quality gates required")
        # Selected camera indices refer to the same frame order as the full
        # immutable BA result, no EXIF/gimbal pose recomputation.
        return render_private_pitched_rgb_orthopatch(
            self.pitched_plane_candidates.planes_private[candidate_index],
            undistorted_rgb_images,
            tuple(self.world_enu_to_camera_private[i] for i in camera_indices),
            intrinsics, meters_per_pixel=meters_per_pixel,
            max_side=max_side, should_cancel=should_cancel,
        )

    def render_private_multi_plane_preview(
        self,
        undistorted_rgb_by_camera: dict[int, np.ndarray],
        intrinsics: PinholeIntrinsics,
        *,
        meters_per_pixel: float = .05,
        max_side: int = 1024,
        max_cameras_per_plane: int = 4,
        should_cancel=None,
    ):
        """Auto-pick fitted cameras, render each pitched facet, align in ENU.

        The mapping contains ONLY already decoded+undistorted private RGB
        images; missing cameras are not invented. No persistence or roof
        verification. For ZIP read-only auto-decode use the companion helper
        in rgb_zip_whole_flight_reconstruction.
        """
        from mcm_solarcheck.services.rgb_private_multi_plane_preview import (
            choose_private_pitched_plane_cameras,
            compose_private_pitched_rgb_orthopatches,
        )
        intrinsics = self.refined_intrinsics or intrinsics
        if (self.pitched_plane_candidates is None
                or not 2 <= len(self.pitched_plane_candidates.planes_private) <= 6
                or self.camera_reprojection_audit is None
                or self.camera_reprojection_audit.cameras_above_p95_limit
                or self.multiview_parallax_audit is None
                or not self.multiview_parallax_audit.privacy_safe_summary()[
                    "parallax_quality_passed"
                ]
                or type(undistorted_rgb_by_camera) is not dict
                or not 2 <= len(undistorted_rgb_by_camera) <= self.total_cameras
                or any(type(i) is not int or not 0 <= i < self.total_cameras
                       for i in undistorted_rgb_by_camera)
                or not isinstance(intrinsics, PinholeIntrinsics)):
            raise ValueError("complete private RTK-BA stereo and pitched planes required")
        available = tuple(sorted(undistorted_rgb_by_camera))
        images = tuple(undistorted_rgb_by_camera[i] for i in available)
        if (any(not isinstance(im, np.ndarray) or im.ndim != 3
                or im.dtype != np.uint8 or im.shape[2] != 3
                for im in images)
                or len({im.shape for im in images}) != 1):
            raise ValueError("aligned calibrated undistorted RGB images required")
        height, width = images[0].shape[:2]
        poses = tuple(self.world_enu_to_camera_private[i] for i in available)
        planes = self.pitched_plane_candidates.planes_private
        patches = []
        for plane_id, plane in enumerate(planes):
            selected_local = choose_private_pitched_plane_cameras(
                plane, poses, intrinsics, width, height,
                max_cameras=max_cameras_per_plane,
                should_cancel=should_cancel,
            )
            global_indices = tuple(available[i] for i in selected_local)
            patches.append(self.render_private_pitched_plane_preview(
                plane_id, tuple(images[i] for i in selected_local),
                global_indices, intrinsics,
                meters_per_pixel=meters_per_pixel,
                max_side=max_side, should_cancel=should_cancel,
            ))
        return compose_private_pitched_rgb_orthopatches(
            planes, tuple(patches), max_side=max_side,
            should_cancel=should_cancel,
        )

    def privacy_safe_summary(self):
        return {
            "schema": "mcm.rgb_rtk_windowed_ba.v1",
            "status": self.state,
            "camera_count": self.total_cameras,
            "windows": self.completed_windows,
            "windows_with_updates": self.windows_with_updates,
            "windows_without_improvement": self.windows_without_improvement,
            "gnss_fit_camera_count": self.fitted_rtk_camera_count,
            "gnss_withheld_camera_count": self.withheld_rtk_camera_count,
            "before_reprojection_rmse_px": self.before_reprojection_rmse_px,
            "after_reprojection_rmse_px": self.after_reprojection_rmse_px,
            "per_camera_reprojection": (
                self.camera_reprojection_audit.privacy_safe_summary()
                if self.camera_reprojection_audit is not None else None
            ),
            "multiview_parallax": (
                self.multiview_parallax_audit.privacy_safe_summary()
                if self.multiview_parallax_audit is not None else None
            ),
            "pitched_plane_hypotheses": (
                self.pitched_plane_candidates.privacy_safe_summary()
                if self.pitched_plane_candidates is not None else None
            ),
            "before_rtk_holdout_p95_m": self.before_rtk_holdout_p95_m,
            "after_rtk_holdout_p95_m": self.after_rtk_holdout_p95_m,
            "after_rtk_holdout_max_m": self.after_rtk_holdout_max_m,
            "joint_objective_before": self.joint_objective_before,
            "joint_objective_after": self.joint_objective_after,
            "rtk_prior_windowed_ba_completed": True,
            "full_global_joint_ba_completed": False,
            "independent_survey_verified": False,
            "roof_geometry_verified": False,
            "orthophoto_verified": False,
            "focal_refined": self.focal_refined,
            "focal_refinement_is_independent_calibration": False,
        }


def adjust_rgb_windowed_bundle_with_private_rtk(
    network: RGBUnscaledNetwork,
    unscaled_windowed: RGBWindowedBAResult,
    priors: PrivateRGBRTKPriors,
    intrinsics: PinholeIntrinsics,
    *,
    max_rtk_holdout_m: float = 2.,
    windowed_options: dict | None = None,
    enforce_camera_p95: bool = True,
    enforce_parallax: bool = True,
    should_cancel=None,
    refine_focal: bool = False,
) -> PrivateRGBRTKWindowedBA:
    """One complete existing SfM flight -> aligned ENU -> robust GNSS window BA.

    The 20%-withheld RTK cameras NEVER contribute to similarity fitting OR
    optimization priors; they still participate in the common pixel tracks.
    Changes to camera zero and one are allowed, unlike unscaled legacy BA.
    """
    if type(enforce_camera_p95) is not bool or type(enforce_parallax) is not bool:
        raise ValueError("explicit final RGB camera/3D parallax policy required")
    if type(refine_focal) is not bool:
        raise ValueError("boolean focal refinement setting required")
    if should_cancel is not None and should_cancel():
        raise RuntimeError("RTK-weighted RGB flight cancelled")
    if (not isinstance(network, RGBUnscaledNetwork)
            or not isinstance(unscaled_windowed, RGBWindowedBAResult)
            or not isinstance(priors, PrivateRGBRTKPriors)
            or not isinstance(intrinsics, PinholeIntrinsics)):
        raise TypeError("existing full SfM/windowed BA and DJI RTK priors required")
    n = len(network.frame_ids)
    if (n < 12 or n > 1000 or priors.frame_ids != network.frame_ids
            or unscaled_windowed.camera_count != n
            or len(unscaled_windowed.world_to_camera) != n):
        raise ValueError("full RGB flight and matched RTK camera identities required")
    focal_reference = (unscaled_windowed.focal_reference_intrinsics or intrinsics)
    if refine_focal and unscaled_windowed.refined_intrinsics is not None:
        intrinsics = unscaled_windowed.refined_intrinsics
    aligned = anchor_windowed_rgb_network_to_rtk(
        network, unscaled_windowed, priors,
        max_withheld_error_m=max_rtk_holdout_m,
        should_cancel=should_cancel,
    )
    if not 0.05 <= max_rtk_holdout_m <= 20:
        raise ValueError("explicit RTK consistency tolerance required")
    scale, rotation, origin = (
        aligned.scale_m_per_sfm_unit, aligned.rotation_sfm_to_enu,
        aligned.translation_enu_private_m,
    )
    # X_ENU = scale * rotation * X_SfM + origin. The equivalent
    # world-ENU->camera pose must preserve EVERY original reprojection:
    # R_ENU = R_SfM * rotation.T, t_ENU = scale*t_SfM - R_ENU*origin.
    poses = []
    for pose in unscaled_windowed.world_to_camera:
        r = pose[:, :3] @ rotation.T
        t = scale * pose[:, 3] - r @ origin
        poses.append(np.column_stack((r, t)))
    metric_candidate = replace(
        network,
        world_to_camera=tuple(poses),
        landmark_xyz_unit_baseline=aligned.landmarks_enu_candidate_private_m,
    )
    fit = np.arange(n) % 5 != 4
    # No GNSS target is supplied for a held-out camera to the BA.
    options = dict(windowed_options or {})
    restricted = {
        "rtk_camera_centers_m", "rtk_camera_sigmas_m", "rtk_fit_mask",
        "gnss_huber_sigma", "should_cancel", "refine_focal", "focal_reference_intrinsics",
    }
    if restricted & options.keys():
        raise ValueError("RTK uncertainty, fit/holdout mask and cancellation are internal")
    if refine_focal:
        options["refine_focal"] = True
        options["focal_reference_intrinsics"] = focal_reference
    fitted = adjust_windowed_unscaled_rgb_network(
        metric_candidate, intrinsics,
        rtk_camera_centers_m=priors.camera_enu_m,
        rtk_camera_sigmas_m=priors.sigma_enu_m,
        rtk_fit_mask=fit,
        gnss_huber_sigma=3.,
        should_cancel=should_cancel,
        **options,
    )
    if not fitted.rtk_camera_priors_optimized:
        raise ValueError("joint RTK-prior camera BA was not executed")
    out_poses = np.asarray(fitted.world_to_camera)
    center = -np.einsum("nji,nj->ni", out_poses[:, :, :3], out_poses[:, :, 3])
    error = np.linalg.norm(center[~fit] - priors.camera_enu_m[~fit], axis=1)
    if (not np.isfinite(error).all()
            or float(error.max()) > max_rtk_holdout_m
            or fitted.weighted_objective_before is None
            or fitted.weighted_objective_after is None
            or fitted.weighted_objective_after > fitted.weighted_objective_before+1e-8):
        raise ValueError("independent RTK holdout consistency or joint BA objective failed")
    result_poses = tuple(np.array(p,copy=True) for p in fitted.world_to_camera)
    xyz = np.array(fitted.landmark_xyz_unit_baseline,copy=True)
    for array in (*result_poses,xyz):
        array.setflags(write=False)
    final_intrinsics = fitted.refined_intrinsics if refine_focal else intrinsics
    if final_intrinsics is None:
        raise ValueError("focal-refining RTK bundle did not return final camera intrinsics")
    pixel_audit = audit_rgb_per_camera_reprojection(
        result_poses, xyz,
        network.observation_frame_indices,
        network.observation_landmark_indices,
        network.observation_pixels_xy,
        final_intrinsics, enforce_camera_p95=enforce_camera_p95,
        should_cancel=should_cancel,
    )
    parallax_audit = audit_rgb_multiview_parallax(
        result_poses, xyz,
        network.observation_frame_indices,
        network.observation_landmark_indices,
        enforce_parallax=enforce_parallax, should_cancel=should_cancel,
    )
    # The next vertical step derives *actual private geometric hypotheses*
    # from the fitted ENU SfM landmarks. No roof semantics or survey truth
    # is inferred, and an empty set is NOT an orthophoto-quality pass.
    pitched = propose_private_rgb_pitched_planes(
        xyz, network.observation_frame_indices,
        network.observation_landmark_indices,
        should_cancel=should_cancel,
    )
    return PrivateRGBRTKWindowedBA(
        network.frame_ids, result_poses, xyz, fitted.before_rmse_px,
        fitted.after_rmse_px, aligned.withheld_p95_m,
        float(np.percentile(error,95)), float(error.max()),
        int(np.sum(fit)), int(np.sum(~fit)), n, fitted.completed_windows,
        float(fitted.weighted_objective_before),
        float(fitted.weighted_objective_after),
        fitted.rtk_windows_with_updates,
        fitted.rtk_windows_without_improvement,
        state=("rtk_camera_reprojection_quality_failed_unverified"
               if pixel_audit.cameras_above_p95_limit else
               "rtk_multiview_parallax_quality_failed_unverified"
               if not parallax_audit.privacy_safe_summary()["parallax_quality_passed"] else
               "rtk_regularized_windows_with_rejections_unverified"
               if fitted.rtk_windows_without_improvement else
               "rtk_regularized_multiview_windowed_ba_unverified"),
        camera_reprojection_audit=pixel_audit,
        multiview_parallax_audit=parallax_audit,
        pitched_plane_candidates=pitched,
        refined_intrinsics=final_intrinsics if refine_focal else None,
        focal_refined=bool(refine_focal and (fitted.focal_refined or unscaled_windowed.focal_refined)),
        focal_reference_intrinsics=focal_reference if refine_focal else None,
    )
