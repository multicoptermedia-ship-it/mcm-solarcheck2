"""Bounded overlapping camera-window BA over the COMPLETE unscaled RGB network.

Reuse DEV-RGB-BA-001 Schur-based BA. Every camera enters at least one window.
Global reprojection is checked before and AFTER each accepted update: neither
windows nor tracks may silently turn into a pseudo-metric roof orthophoto.

The reconstruction's baseline is arbitrary; windows are an optimization
strategy, NOT survey/metric/GCP evidence or true full-global joint BA.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

from mcm_solarcheck.services.rgb_global_camera_network import RGBUnscaledNetwork
from mcm_solarcheck.services.rgb_global_bundle_adjustment import (
    adjust_unscaled_rgb_bundle, _projections, _robust_gnss_cost,
)
from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics


@dataclass(frozen=True)
class RGBWindowedBAResult:
    world_to_camera: tuple[np.ndarray, ...]
    landmark_xyz_unit_baseline: np.ndarray
    before_rmse_px: float
    after_rmse_px: float
    before_p95_px: float
    after_p95_px: float
    camera_count: int
    landmark_count: int
    observation_count: int
    completed_windows: int
    camera_window_size: int
    status: str = "windowed_rgb_ba_unscaled_unverified"
    metric_scale_verified: bool = False
    georeferencing_verified: bool = False
    global_joint_ba_completed: bool = False
    orthophoto_verified: bool = False
    rtk_camera_priors_optimized: bool = False
    weighted_objective_before: float | None = None
    weighted_objective_after: float | None = None
    rtk_windows_with_updates: int = 0
    rtk_windows_without_improvement: int = 0
    refined_intrinsics: PinholeIntrinsics | None = None
    focal_refined: bool = False
    focal_reference_intrinsics: PinholeIntrinsics | None = None


def _camera_windows(count, size, overlap):
    # One local solve for a small flight; overlapping anchor cameras fix the
    # arbitrary coordinate system for subsequent windows.
    start = 0
    while True:
        end = min(start + size, count)
        yield np.arange(start, end, dtype=np.int32)
        if end == count:
            break
        start = end - overlap


def _all_network_errors(poses, points, fid, lid, pixels, intrinsics):
    k = np.asarray([
        intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    ], dtype=np.float64)
    _, projected = _projections(poses, points, fid, lid, k)
    errors = np.linalg.norm(projected - pixels, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("nonfinite whole-flight RGB reprojection")
    return errors


def _whole_flight_robust_objective(errors, poses, centers, sigma, fit, huber):
    """Full-track AND private fitted-RTK objective, never just local window pixels."""
    px_huber = 2.0
    total = float(np.sum(np.where(errors <= px_huber, .5*errors**2,
                                  px_huber*(errors-.5*px_huber))))
    if centers is not None:
        total += _robust_gnss_cost(poses, centers, sigma, fit, huber)[0]
    return total


def adjust_windowed_unscaled_rgb_network(
    network: RGBUnscaledNetwork,
    intrinsics: PinholeIntrinsics,
    *,
    window_cameras: int = 6,
    overlap_cameras: int = 2,
    max_window_landmarks: int = 400,
    max_window_observations: int = 4000,
    max_iterations: int = 8,
    max_cameras_total: int = 1000,
    max_landmarks_total: int = 40000,
    max_observations_total: int = 300000,
    should_cancel=None,
    on_progress=None,
    rtk_camera_centers_m=None,
    rtk_camera_sigmas_m=None,
    rtk_fit_mask=None,
    gnss_huber_sigma=3.,
    refine_focal: bool = False,
    focal_reference_intrinsics: PinholeIntrinsics | None = None,
) -> RGBWindowedBAResult:
    """Refine every image's pose using existing *unscaled* multi-view tracks.

    A window uses all available frames within its extent, and a deterministic,
    bounded subset of shared 3D landmarks; rejected windows fail the whole
    workflow. In RTK mode, global *joint robust objective* (pixels plus GNSS)
    may NEVER increase, but pure pixel RMSE can rise slightly under constraints.
    No absolute accuracy or independent GCP validity is certified.
    """
    def cancel():
        if should_cancel is not None and should_cancel():
            raise RuntimeError("whole-flight RGB bundle adjustment cancelled")
    cancel()
    if not isinstance(network, RGBUnscaledNetwork) or not isinstance(intrinsics, PinholeIntrinsics):
        raise TypeError("existing SfM multi-view network and pinhole camera required")
    if (type(refine_focal) is not bool
            or type(window_cameras) is not int or not 3 <= window_cameras <= 24
            or type(overlap_cameras) is not int
            or not 2 <= overlap_cameras < window_cameras - 1
            or type(max_window_landmarks) is not int
            or not 8 <= max_window_landmarks <= 2000
            or type(max_window_observations) is not int
            or not 24 <= max_window_observations <= 20000
            or type(max_iterations) is not int or not 1 <= max_iterations <= 40
            or type(max_cameras_total) is not int or not 3 <= max_cameras_total <= 1000
            or type(max_landmarks_total) is not int or not 8 <= max_landmarks_total <= 200000
            or type(max_observations_total) is not int or not 24 <= max_observations_total <= 800000):
        raise ValueError("invalid bounded whole-flight BA budget")
    cameras = np.asarray(network.world_to_camera, dtype=np.float64).copy()
    landmarks = np.asarray(network.landmark_xyz_unit_baseline, dtype=np.float64).copy()
    fid = np.asarray(network.observation_frame_indices)
    lid = np.asarray(network.observation_landmark_indices)
    xy = np.asarray(network.observation_pixels_xy, dtype=np.float64)
    camera_count = len(network.frame_ids)
    if camera_count > max_cameras_total:
        raise ValueError("whole-flight RGB camera budget exceeded")
    if (not 3 <= camera_count <= max_cameras_total
            or cameras.shape != (camera_count,3,4)
            or landmarks.ndim != 2 or landmarks.shape[1] != 3
            or not 8 <= len(landmarks) <= max_landmarks_total
            or fid.ndim != 1 or lid.ndim != 1
            or xy.shape != (len(fid),2)
            or not 24 <= len(fid) <= max_observations_total
            or len(lid) != len(fid)
            or not np.issubdtype(fid.dtype,np.integer)
            or not np.issubdtype(lid.dtype,np.integer)
            or np.any(fid < 0) or np.any(fid >= camera_count)
            or np.any(lid < 0) or np.any(lid >= len(landmarks))
            or any(not np.isfinite(x).all() for x in (cameras,landmarks,xy))
            or len(np.unique(fid)) != camera_count
            or len(np.unique(lid)) != len(landmarks)
            or not np.all(np.isfinite([
                intrinsics.fx,intrinsics.fy,intrinsics.cx,intrinsics.cy
            ])) or min(intrinsics.fx,intrinsics.fy) <= 0):
        raise ValueError("invalid complete bounded whole-flight RGB BA data")
    if any(np.count_nonzero(fid == i) < 8 for i in range(camera_count)):
        raise ValueError("insufficient 3D observations in some RGB cameras")
    use_gnss = (rtk_camera_centers_m is not None
                or rtk_camera_sigmas_m is not None
                or rtk_fit_mask is not None)
    centers = sigmas = fit = None
    if use_gnss:
        if rtk_camera_centers_m is None or rtk_camera_sigmas_m is None:
            raise ValueError("both private RTK centers and sigmas required")
        centers = np.asarray(rtk_camera_centers_m, dtype=np.float64)
        sigmas = np.asarray(rtk_camera_sigmas_m, dtype=np.float64)
        fit = (np.ones(camera_count, dtype=bool) if rtk_fit_mask is None
               else np.asarray(rtk_fit_mask))
        if (camera_count < 12 or window_cameras < 5
                or centers.shape != (camera_count,3)
                or sigmas.shape != (camera_count,3)
                or fit.shape != (camera_count,) or fit.dtype != np.bool_
                or not np.isfinite(centers).all()
                or not np.isfinite(sigmas).all()
                or np.any(sigmas[:, :2] < .10) or np.any(sigmas[:, 2] < .20)
                or np.count_nonzero(fit) < 8
                or type(gnss_huber_sigma) not in (int,float)
                or not np.isfinite(gnss_huber_sigma)
                or not .5 <= gnss_huber_sigma <= 10.):
            raise ValueError("complete conservative RTK 12+ camera priors required")
    # The existing BA validates proper rotations and anchor-gauge consistency.
    initial=_all_network_errors(cameras,landmarks,fid,lid,xy,intrinsics)
    before=float(np.sqrt(np.mean(initial**2)))
    before_p95=float(np.percentile(initial,95))
    global_before = _whole_flight_robust_objective(
        initial, cameras, centers, sigmas, fit, gnss_huber_sigma,
    )
    global_current = global_before
    current_intrinsics = intrinsics
    focal_reference = intrinsics if focal_reference_intrinsics is None else focal_reference_intrinsics
    if refine_focal and (not isinstance(focal_reference, PinholeIntrinsics)
            or not np.isfinite([focal_reference.fx,focal_reference.fy,
                                focal_reference.cx,focal_reference.cy]).all()
            or min(focal_reference.fx,focal_reference.fy) <= 0
            or not .5-1.e-12 <= intrinsics.fx/focal_reference.fx <= 2.+1.e-12
            or not np.isclose(intrinsics.fx/intrinsics.fy,
                              focal_reference.fx/focal_reference.fy, rtol=1.e-12)
            or (intrinsics.cx,intrinsics.cy) != (focal_reference.cx,focal_reference.cy)):
        raise ValueError("consistent bounded original focal reference required")
    windows=list(_camera_windows(camera_count,min(window_cameras,camera_count),overlap_cameras))
    if len(windows) > 512:
        raise ValueError("whole-flight RGB BA exceeds window budget")
    # Every image needs actual support: no invisible frames lost through slicing.
    touched=set()
    completed=0
    updates=0
    without_improvement=0
    for numbers in windows:
        cancel()
        local_ids = np.searchsorted(numbers, fid)
        in_window = (local_ids < len(numbers))
        # Guard indexing with clip to handle positions at end.
        in_window &= numbers[np.minimum(local_ids,len(numbers)-1)] == fid
        observation_indices = np.flatnonzero(in_window)
        original_lids=lid[observation_indices]
        available, inverse, counts = np.unique(
            original_lids,return_inverse=True,return_counts=True
        )
        # Only truly multi-view tracks can constrain an in-window pose.
        viable=available[counts>=2]
        if len(viable) < 8:
            raise ValueError("insufficient overlapping 3D tracks for RGB camera window")
        # Deterministic, spatially distributed landmark sampling in track-ID space.
        if len(viable)>max_window_landmarks:
            sampled=np.linspace(0,len(viable)-1,max_window_landmarks).astype(np.int64)
            viable=viable[sampled]
        selected_indices=observation_indices[np.isin(original_lids,viable)]
        if len(selected_indices)>max_window_observations:
            # Prefer points with fewer total observations only if explicitly
            # verified; refuse silent truncation of camera evidence.
            raise ValueError("RGB window observation budget exceeded")
        original_points=np.unique(lid[selected_indices])
        mapped=np.searchsorted(original_points,lid[selected_indices]).astype(np.int32)
        local_camera=np.searchsorted(numbers,fid[selected_indices]).astype(np.int32)
        if (len(original_points)<8 or len(selected_indices)<24
                or any(np.count_nonzero(local_camera == i)<8 for i in range(len(numbers)))):
            raise ValueError("insufficient shared points in some RGB camera window")
        touched.update(int(x) for x in numbers)
        # Reuse Schur BA: legacy anchors two cameras; GNSS mode lets every
        # camera move against its weighted metric RTK prior.
        sub=RGBUnscaledNetwork(
            tuple(network.frame_ids[int(i)] for i in numbers),
            tuple(cameras[int(i)].copy() for i in numbers),
            len(original_points),len(original_points),0,0.,0.,
            landmark_xyz_unit_baseline=landmarks[original_points].copy(),
            observation_landmark_indices=mapped,
            observation_frame_indices=local_camera,
            observation_pixels_xy=xy[selected_indices].copy(),
        )
        fit_options = {}
        if refine_focal:
            # Keep one common camera model within the ORIGINAL flight-wide
            # interval; a new window must not multiply the allowed range.
            fit_options.update(
                refine_focal=True,
                focal_scale_bounds=(
                    float(min(1., .5*focal_reference.fx/current_intrinsics.fx)),
                    float(max(1., 2.*focal_reference.fx/current_intrinsics.fx)),
                ),
            )
        if use_gnss:
            if np.count_nonzero(fit[numbers]) < 3:
                raise ValueError("insufficient independent GNSS priors for RGB window")
            fit_options.update(
                camera_center_priors_m=centers[numbers],
                camera_center_sigma_m=sigmas[numbers],
                camera_prior_fit_mask=fit[numbers],
                gnss_huber_sigma=gnss_huber_sigma,
            )
        fitted=adjust_unscaled_rgb_bundle(
            sub,current_intrinsics,max_iterations=max_iterations,
            max_cameras=len(numbers),max_landmarks=max_window_landmarks,
            max_observations=max_window_observations,
            should_cancel=should_cancel,**fit_options,
        )
        proposal_cameras=cameras.copy()
        proposal_points=landmarks.copy()
        proposal_cameras[numbers] = np.stack(fitted.world_to_camera)
        proposal_points[original_points]=fitted.landmark_xyz_unit_baseline
        proposal_intrinsics = (fitted.refined_intrinsics if refine_focal
                               else current_intrinsics)
        if proposal_intrinsics is None:
            raise ValueError("focal-refining RGB window did not return camera intrinsics")
        old_errors=_all_network_errors(cameras,landmarks,fid,lid,xy,current_intrinsics)
        try:
            new_errors=_all_network_errors(
                proposal_cameras,proposal_points,fid,lid,xy,proposal_intrinsics,
            )
        except ValueError:
            if not refine_focal:
                raise
            new_errors = np.full(len(fid), np.inf)
        # Every candidate is audited against ALL tracks and ALL non-withheld
        # GNSS camera centers. Pixel-only regression is preserved for old mode.
        new_global = _whole_flight_robust_objective(
            new_errors, proposal_cameras, centers, sigmas, fit, gnss_huber_sigma,
        )
        if use_gnss or refine_focal:
            if (new_global > global_current + 1e-8 or
                    (not use_gnss and np.mean(new_errors**2) > np.mean(old_errors**2) + 1.e-9)):
                # Shared landmarks occur OUTSIDE this window. A local Schur
                # minimizer is not necessarily a descent direction for the
                # full-flight objective; backtrack both pose and landmark
                # increments, and fall back to pose-only/point-only steps.
                # Never accept a local improvement that hurts other frames.
                accepted = False
                variants = (("both", "geometry", "poses", "points", "focal")
                            if refine_focal else ("both", "poses", "points"))
                for variant in variants:
                    if accepted:
                        break
                    for alpha in (.5, .25, .125, .0625, .03125, .015625):
                        cancel()
                        trial_cameras = cameras.copy()
                        trial_points = landmarks.copy()
                        trial_intrinsics = current_intrinsics
                        if refine_focal and variant in ("both", "focal"):
                            focal_ratio = float(np.exp(alpha*np.log(
                                proposal_intrinsics.fx/current_intrinsics.fx
                            )))
                            trial_intrinsics = PinholeIntrinsics(
                                current_intrinsics.fx*focal_ratio,
                                current_intrinsics.fy*focal_ratio,
                                current_intrinsics.cx,current_intrinsics.cy,
                            )
                        if variant not in ("points", "focal"):
                            for cam_id in numbers:
                                old_pose = cameras[cam_id]
                                target = proposal_cameras[cam_id]
                                diff_rotation = target[:, :3] @ old_pose[:, :3].T
                                log_rotation, _ = cv2.Rodrigues(diff_rotation)
                                small_rotation, _ = cv2.Rodrigues(alpha * log_rotation)
                                trial_cameras[cam_id, :, :3] = (
                                    small_rotation @ old_pose[:, :3]
                                )
                                trial_cameras[cam_id, :, 3] = (
                                    old_pose[:, 3] +
                                    alpha * (target[:, 3] - old_pose[:, 3])
                                )
                        if variant not in ("poses", "focal"):
                            trial_points[original_points] = (
                                landmarks[original_points] +
                                alpha * (proposal_points[original_points] -
                                         landmarks[original_points])
                            )
                        try:
                            trial_errors = _all_network_errors(
                                trial_cameras, trial_points, fid, lid, xy,
                                trial_intrinsics,
                            )
                        except ValueError:
                            continue
                        trial_global = _whole_flight_robust_objective(
                            trial_errors, trial_cameras,
                            centers, sigmas, fit, gnss_huber_sigma,
                        )
                        if (trial_global < global_current - 1.e-10 and
                                (use_gnss or np.mean(trial_errors**2) <= np.mean(old_errors**2) + 1.e-9)):
                            proposal_cameras, proposal_points = (
                                trial_cameras, trial_points,
                            )
                            new_errors, new_global = trial_errors, trial_global
                            proposal_intrinsics = trial_intrinsics
                            accepted = True
                            break
                if not accepted:
                    # A non-descent local direction can occur for shared tracks.
                    # Refuse the unsafe mutation, record an explicit degraded
                    # window and continue. No improvements anywhere = failure.
                    without_improvement += 1
                    completed += 1
                    if on_progress is not None:
                        on_progress(completed,len(windows),len(numbers))
                    continue
        elif float(np.mean(new_errors**2)) > float(np.mean(old_errors**2)) + 1e-9:
            raise ValueError("RGB window update worsens full-flight camera network")
        if use_gnss and new_global < global_current - 1e-10:
            updates += 1
        global_current = new_global
        cameras,landmarks=proposal_cameras,proposal_points
        current_intrinsics = proposal_intrinsics
        completed+=1
        if on_progress is not None:
            on_progress(completed,len(windows),len(numbers))
    cancel()
    if touched!=set(range(camera_count)):
        raise ValueError("whole-flight RGB BA left camera frames unprocessed")
    final=_all_network_errors(cameras,landmarks,fid,lid,xy,current_intrinsics)
    after=float(np.sqrt(np.mean(final**2)))
    after_p95=float(np.percentile(final,95))
    if not use_gnss and after > before + 1e-8:
        raise ValueError("whole-flight RGB BA did not validate global improvement")
    if use_gnss and (global_current > global_before + 1.e-8 or updates == 0):
        raise ValueError("whole-flight RTK BA lacks a globally improving safe step")
    result_poses=tuple(p.copy() for p in cameras)
    result_points=landmarks.copy()
    for array in (*result_poses,result_points):
        array.setflags(write=False)
    return RGBWindowedBAResult(
        result_poses,result_points,before,after,before_p95,after_p95,
        camera_count,len(result_points),len(fid),completed,
        min(window_cameras,camera_count),
        status=("windowed_rgb_ba_with_rejections_unscaled_unverified"
                if refine_focal and without_improvement and not use_gnss else
                "rtk_regularized_windows_with_rejections_unverified"
                if use_gnss and without_improvement else
                "rtk_regularized_windowed_rgb_ba_candidate_unverified"
                if use_gnss else "windowed_rgb_ba_unscaled_unverified"),
        rtk_camera_priors_optimized=use_gnss,
        weighted_objective_before=global_before if use_gnss else None,
        weighted_objective_after=global_current if use_gnss else None,
        rtk_windows_with_updates=updates if use_gnss else 0,
        rtk_windows_without_improvement=without_improvement if use_gnss else 0,
        refined_intrinsics=(PinholeIntrinsics(*map(float, (
            current_intrinsics.fx,current_intrinsics.fy,
            current_intrinsics.cx,current_intrinsics.cy,
        ))) if refine_focal else None),
        focal_refined=bool(refine_focal and abs(np.log(current_intrinsics.fx/intrinsics.fx)) > 1.e-10),
        focal_reference_intrinsics=focal_reference if refine_focal else None,
    )
