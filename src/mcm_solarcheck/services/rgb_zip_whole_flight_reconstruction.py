"""Private RGB ZIP flight -> SIFT/ORB -> SfM -> RTK-weighted windowed BA.

No import database, ZIP extraction, customer image copies, external executable,
ODM, GPS export or roof/orthophoto certification. A real reconstruction MUST
use all supplied source cameras or fail closed. All geometry stays in RAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_flight_zip_series import (
    PrivateRGBFlightSeries, read_private_rgb_zip_series,
)
from mcm_solarcheck.services.rgb_camera_initialization import (
    initialize_private_rgb_camera,
)
from mcm_solarcheck.services.rgb_rtk_camera_priors import (
    build_private_rtk_camera_priors,
)
from mcm_solarcheck.services.orthomosaic_overlap_candidates import (
    select_overlap_candidates,
)
from mcm_solarcheck.services.rgb_global_camera_network import (
    RGBNetworkFrame, RGBNetworkLink, solve_global_rgb_camera_network,
)
from mcm_solarcheck.services.rgb_sfm_link_graph_gate import (
    audit_complete_rgb_sfm_link_graph,
)
from mcm_solarcheck.services.rgb_sfm_track_support_gate import (
    audit_complete_rgb_sfm_track_support,
)
from mcm_solarcheck.services.rgb_per_camera_reprojection_gate import (
    RGBPerCameraReprojectionQualityError,
)
from mcm_solarcheck.services.rgb_multiview_parallax_gate import (
    RGBMultiviewParallaxQualityError,
)
from mcm_solarcheck.services.rgb_windowed_bundle_adjustment import (
    adjust_windowed_unscaled_rgb_network,
)
from mcm_solarcheck.services.rgb_rtk_joint_bundle_adjustment import (
    adjust_rgb_windowed_bundle_with_private_rtk,
)


class PrivateRGBFlightStageError(ValueError):
    """Public-safe message, with explicitly private in-memory diagnostics.

    Only ``stage`` may be included in a public/aggregate report. ``private_cause``
    and ``private_context`` are for a local caller debugging its own flight;
    they must never be serialized into a customer/GitHub report.
    """

    def __init__(self, stage: str, *, private_cause=None, private_context=None,
                 public_quality_summary=None):
        if stage not in ("intake", "calibration", "camera_initialization", "feature_extraction",
                         "overlap_matching", "global_sfm", "unscaled_ba",
                         "rtk_weighted_ba", "surface_candidate"):
            raise ValueError("unrecognized private RGB processing stage")
        self.stage = stage
        self.private_cause = private_cause
        self.private_context = private_context
        self.public_quality_summary = dict(public_quality_summary or {})
        super().__init__(f"whole-flight RGB {stage} gate not passed")


@dataclass(frozen=True)
class PrivateRGBFlightReconstruction:
    """Private RAM result; export only ``privacy_safe_summary()`` publicly."""
    frame_count: int
    candidate_pair_count: int
    feature_matched_pair_count: int
    cross_strip_matched_pair_count: int
    sfm_accepted_nonadjacent_pair_count: int
    network_landmark_count: int
    sfm_reprojection_p95_px: float
    unscaled_ba_reprojection_rmse_px: float
    rtk_windowed_ba: object
    geometric_link_graph_audit: object
    camera_initialization_summary: dict | None = field(default=None, repr=False, compare=False)
    focal_refined: bool = False
    private_series: PrivateRGBFlightSeries | None = field(default=None, repr=False, compare=False)
    private_network: object = field(default=None, repr=False, compare=False)
    private_intrinsics: PinholeIntrinsics | None = field(default=None, repr=False, compare=False)
    private_distortion: np.ndarray | None = field(default=None, repr=False, compare=False)
    sfm_3d_track_support_audit: object = field(default=None, repr=False, compare=False)
    feature_method: str = "orb"

    def privacy_safe_summary(self) -> dict:
        ba = self.rtk_windowed_ba.privacy_safe_summary()
        return {
            "schema": "mcm.private_complete_rgb_flight_reconstruction.v1",
            "status": "whole_rgb_rtk_windowed_ba_candidate_not_an_orthophoto",
            "frame_count": self.frame_count,
            "feature_method": self.feature_method,
            "descriptor_metric": "l2" if self.feature_method == "sift" else "hamming",
            "candidate_pairs": self.candidate_pair_count,
            "feature_matched_pairs": self.feature_matched_pair_count,
            "cross_strip_matched_pairs": self.cross_strip_matched_pair_count,
            "cross_strip_matches_are_descriptor_only": True,
            "sfm_accepted_spatial_nonadjacent_pairs": (
                self.sfm_accepted_nonadjacent_pair_count
            ),
            "physical_cross_strip_identity_verified": False,
            **self.geometric_link_graph_audit.privacy_safe_summary(),
            **(self.sfm_3d_track_support_audit.privacy_safe_summary()
               if self.sfm_3d_track_support_audit is not None else {}),
            "sfm_track_conflicting_merges_rejected": getattr(
                self.private_network, "rejected_conflicting_track_merges", 0),
            "sfm_accepted_geometric_feature_correspondences": getattr(
                self.private_network, "accepted_geometric_feature_correspondences", 0),
            "sfm_landmarks": self.network_landmark_count,
            "sfm_reprojection_p95_px": self.sfm_reprojection_p95_px,
            "unscaled_ba_reprojection_rmse_px": self.unscaled_ba_reprojection_rmse_px,
            "rtk_windowed_ba": ba,
            "camera_initialization": dict(self.camera_initialization_summary or {}),
            "lens_calibration_verified": bool(
                (self.camera_initialization_summary or {}).get("lens_calibration_verified", False)
            ),
            "focal_length_refined_from_multiview": self.focal_refined,
            "intrinsics_refinement_scope": (
                "common_focal_length_only" if self.camera_initialization_summary
                and self.camera_initialization_summary.get("lens_calibration_verified") is False
                else "external_calibration_fixed"
            ),
            "full_camera_self_calibration_completed": False,
            "all_input_rgb_frames_processed": True,
            "required_flight_coverage_verified": False,
            "independent_survey_verified": False,
            "roof_geometry_verified": False,
            "rgb_orthophoto_verified": False,
            "module_analysis_verified": False,
            "thermal_registration_verified": False,
        }


def _calibrated_camera(calibration: dict, series: PrivateRGBFlightSeries):
    """Same external calibration contract as existing project RGB SfM."""
    if (not isinstance(calibration, dict)
            or calibration.get("schema") != "mcm.rgb_camera_intrinsics_external.v1"
            or calibration.get("lens_calibration_verified") is not True
            or not isinstance(calibration.get("calibration_provenance"), str)
            or not calibration["calibration_provenance"].strip()):
        raise ValueError("independent calibrated lens and provenance required")
    try:
        values = calibration["intrinsics"]
        intrinsics = PinholeIntrinsics(
            *[float(values[name]) for name in ("fx", "fy", "cx", "cy")]
        )
        distortion = np.asarray(
            calibration["distortion_coefficients"], dtype=np.float64
        )
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ValueError("complete external camera calibration required") from exc
    dimensions = {(f.width, f.height) for f in series.frames}
    if (len(dimensions) != 1
            or distortion.shape not in ((4,), (5,), (8,), (12,), (14,))
            or not np.isfinite(distortion).all()
            or not np.isfinite((
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            )).all()
            or min(intrinsics.fx, intrinsics.fy) <= 0):
        raise ValueError("single consistent RGB sensor calibration required")
    width, height = next(iter(dimensions))
    if not 0 <= intrinsics.cx < width or not 0 <= intrinsics.cy < height:
        raise ValueError("calibrated RGB principal point outside sensor")
    return intrinsics, distortion


def _private_rgb_overlap_pairs(frame_ids, camera_enu, *,
                               max_neighbours=10, radius_m=18.):
    """Consecutive flight pairs plus metric-nearest, nonadjacent cross-strip.

    GPS is only a candidate search heuristic, NEVER accepted match geometry.
    Avoid all-to-all 160-frame matching; every image participates in the
    consecutive graph, and spatial cross-strip connections are attempted.
    """
    count = len(frame_ids)
    if (not 12 <= count <= 1000
            or type(max_neighbours) is not int
            or not 2 <= max_neighbours <= 16
            or not np.isfinite(radius_m) or not 2 <= radius_m <= 100):
        raise ValueError("bounded complete RGB spatial-pair search required")
    xyz = np.asarray(camera_enu, dtype=np.float64)
    if xyz.shape != (count, 3) or not np.isfinite(xyz).all():
        raise ValueError("finite private camera positions required")
    cross = set()
    for i in range(count):
        d = np.linalg.norm(xyz[:, :2] - xyz[i, :2], axis=1)
        nearest = sorted(
            (j for j in range(count)
             if abs(i-j) >= 3 and .25 <= d[j] <= radius_m),
            key=lambda j: (float(d[j]), j),
        )
        for j in nearest[:max_neighbours]:
            cross.add((min(i, j), max(i, j)))
    pair_ids = tuple((frame_ids[i], frame_ids[j]) for i, j in sorted(cross))
    return select_overlap_candidates(
        frame_ids, cross_strip_pairs=pair_ids,
        max_candidates=(count-1) + count * max_neighbours,
    )


def _features_from_private_zip(series, intrinsics, distortion, *,
                               feature_method, max_side, max_features,
                               should_cancel, on_progress):
    """Decode one original at a time; detect undistorted SIFT or ORB features."""
    if feature_method not in ("sift", "orb"):
        raise ValueError("supported RGB feature method required")
    width, height = series.frames[0].width, series.frames[0].height
    matrix = np.array((
        (intrinsics.fx, 0., intrinsics.cx),
        (0., intrinsics.fy, intrinsics.cy), (0., 0., 1.),
    ), dtype=np.float64)
    detector = (cv2.SIFT_create(nfeatures=max_features) if feature_method == "sift"
                else cv2.ORB_create(nfeatures=max_features))
    observations, descriptors = [], []
    for i, frame in enumerate(series.frames):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("whole-flight RGB processing cancelled")
        with ZipFile(frame.source_archive) as archive:
            entry = archive.getinfo(frame.member_name)
            if not 128 <= entry.file_size <= 40_000_000:
                raise ValueError("original RGB member exceeds image budget")
            with archive.open(entry) as src:
                data = src.read(40_000_001)
            if len(data) != entry.file_size:
                raise ValueError("original RGB member changed since validated intake")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        del data
        if image is None or image.shape != (height, width):
            raise ValueError("unreadable primary JPEG in RGB flight")
        factor = min(1., max_side / max(width, height))
        if factor < 1.:
            image = cv2.resize(image, (
                max(32, round(width * factor)), max(32, round(height * factor))
            ), interpolation=cv2.INTER_AREA)
        sx, sy = image.shape[1] / width, image.shape[0] / height
        scaled = matrix.copy()
        scaled[0, :] *= sx
        scaled[1, :] *= sy
        image = cv2.undistort(image, scaled, distortion, None, scaled)
        points, desc = detector.detectAndCompute(image, None)
        del image
        if desc is None or len(points) < 16:
            raise ValueError("insufficient RGB feature evidence in whole flight")
        xy = np.asarray([(p.pt[0]/sx, p.pt[1]/sy) for p in points],
                        dtype=np.float64)
        observations.append(RGBNetworkFrame(frame.frame_id, xy))
        descriptors.append(desc)
        if on_progress is not None:
            on_progress("features", i+1, len(series.frames))
    return tuple(observations), descriptors


def _match_distinct_rgb_descriptors(first, second, *, ratio=.75, maximum_distance=64.):
    """Mutual SIFT/ORB matches that beat an alternative in BOTH images.

    Repeated PV cells can produce a nearest neighbour within the Hamming
    budget even when several alternatives are equally good. SIFT uses L2 and
    the same ambiguity test, without applying a Hamming-distance threshold to
    its floating descriptors. Geometry still validates every accepted pair.
    """
    if (not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray)
            or first.ndim != 2 or second.ndim != 2
            or not 2 <= len(first) <= 10000 or not 2 <= len(second) <= 10000
            or not .5 <= ratio < 1. or not 1. <= maximum_distance <= 128.):
        raise ValueError("bounded consistent RGB descriptors and ambiguity policy required")
    orb = (first.dtype == second.dtype == np.uint8
           and first.shape[1:] == second.shape[1:] == (32,))
    sift = (first.dtype == second.dtype == np.float32
            and first.shape[1:] == second.shape[1:] == (128,)
            and np.isfinite(first).all() and np.isfinite(second).all()
            and np.max(np.abs(first)) <= 1.e6 and np.max(np.abs(second)) <= 1.e6)
    if not (orb or sift):
        raise ValueError("bounded ORB uint8x32 or finite SIFT float32x128 descriptors required")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if orb else cv2.NORM_L2,
                            crossCheck=False)
    def distinct(a, b):
        return {row[0].queryIdx: row[0]
                for row in matcher.knnMatch(a, b, k=2)
                if len(row) == 2 and (not orb or row[0].distance <= maximum_distance)
                and row[0].distance < ratio * row[1].distance}
    forward, reverse = distinct(first, second), distinct(second, first)
    accepted = [match for match in forward.values()
                if match.trainIdx in reverse
                and reverse[match.trainIdx].trainIdx == match.queryIdx]
    accepted.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))
    return np.asarray([(match.queryIdx, match.trainIdx) for match in accepted],
                      dtype=np.int32).reshape(-1, 2)


def reconstruct_private_complete_rgb_zip_flight(
    archives: tuple[str | Path, ...],
    calibration: dict | None = None,
    *,
    max_side: int = 1200,
    max_features: int = 1600,
    max_neighbours: int = 10,
    feature_method: str = "sift",
    should_cancel=None,
    on_progress=None,
) -> PrivateRGBFlightReconstruction:
    """Complete ZIP->SfM->window BA->GNSS BA, no file writes at any stage.

    Without a separate calibration file, real EXIF supplies explicitly estimated
    initial camera parameters and the existing BA refines a shared focal scale.
    Passing an external calibration retains its strict verification contract;
    an invalid explicit calibration never silently falls back to estimates.

    Failed geometry is reported as a stage failure; never silently select a
    'good' flight strip or export an optical 2D stitch as an orthophoto.
    """
    stage = "intake"
    private_context = {"feature_method": feature_method}
    automatic_camera = calibration is None
    try:
        if on_progress is not None:
            on_progress(stage, 0, 1)
        series = read_private_rgb_zip_series(
            archives, should_cancel=should_cancel,
        )
        private_context["series"] = series
        ids = tuple(frame.frame_id for frame in series.frames)
        if len(ids) < 12:
            raise ValueError("12+ cameras required for whole-flight RTK BA")
        if automatic_camera:
            stage = "camera_initialization"
            initial_camera = initialize_private_rgb_camera(series)
            intrinsics, distortion = initial_camera.intrinsics, initial_camera.distortion
            camera_summary = initial_camera.privacy_safe_summary()
        else:
            stage = "calibration"
            intrinsics, distortion = _calibrated_camera(calibration, series)
            camera_summary = {
                "source": "external_verified_calibration",
                "status": "external_camera_parameters_supplied",
                "lens_calibration_verified": True,
                "intrinsics_optimized": False,
            }
        private_context.update(intrinsics=intrinsics, distortion=distortion,
                               camera_initialization_summary=camera_summary)
        if on_progress is not None:
            on_progress(stage, 1, 1)
        priors = build_private_rtk_camera_priors(
            series, ids, should_cancel=should_cancel,
        )
        private_context["priors"] = priors
        stage = "feature_extraction"
        if (feature_method not in ("sift", "orb")
                or type(max_side) is not int or not 256 <= max_side <= 2400
                or type(max_features) is not int or not 300 <= max_features <= 4000):
            raise ValueError("supported RGB feature method and bounded budgets required")
        observations, descriptors = _features_from_private_zip(
            series, intrinsics, distortion,
            feature_method=feature_method,
            max_side=max_side, max_features=max_features,
            should_cancel=should_cancel, on_progress=on_progress,
        )
        private_context.update(observations=observations, descriptors=descriptors)
        stage = "overlap_matching"
        candidates = _private_rgb_overlap_pairs(
            ids, priors.camera_enu_m, max_neighbours=max_neighbours,
        )
        private_context["candidates"] = candidates
        links, across = [], 0
        ids_to_indices = {fid: i for i, fid in enumerate(ids)}
        for pair_index, pair in enumerate(candidates):
            if should_cancel is not None and should_cancel():
                raise RuntimeError("whole-flight RGB processing cancelled")
            a = ids_to_indices[pair.first_id]
            b = ids_to_indices[pair.second_id]
            chosen = _match_distinct_rgb_descriptors(descriptors[a], descriptors[b])
            if len(chosen) >= 12:
                links.append(RGBNetworkLink(
                    pair.first_id, pair.second_id,
                    chosen,
                ))
                if pair.reason == "cross_strip":
                    across += 1
            if on_progress is not None:
                on_progress("matches", pair_index+1, len(candidates))
        private_context["links"] = tuple(links)
        if across == 0:
            raise ValueError("no cross-strip matching evidence for RGB flight")
        stage = "global_sfm"
        if on_progress is not None:
            on_progress(stage, 0, 1)
        network = solve_global_rgb_camera_network(
            observations, tuple(links), intrinsics,
            max_track_count=60000, should_cancel=should_cancel,
            on_progress=on_progress,
        )
        if network.frame_ids != ids or len(network.world_to_camera) != len(ids):
            raise ValueError("SfM did not reconstruct every RGB image")
        private_context["network"] = network
        # Audit ALL calibrated RANSAC/pose links, not only descriptor matches.
        # GNSS-near nonadjacent pairs remain flight-strip CANDIDATES.
        audit = audit_complete_rgb_sfm_link_graph(
            ids,
            tuple((p.first_id, p.second_id) for p in links),
            tuple(network.accepted_link_pairs),
            tuple((p.first_id, p.second_id) for p in candidates
                  if p.reason == "cross_strip"),
        )
        actual_count = audit.spatial_nonadjacent_geometric_pairs
        private_context["geometric_link_graph_audit"] = audit
        track_audit = audit_complete_rgb_sfm_track_support(
            ids, tuple(network.accepted_link_pairs),
            tuple((p.first_id, p.second_id) for p in candidates
                  if p.reason == "cross_strip"),
            network.observation_frame_indices, network.observation_landmark_indices,
            landmark_count=len(network.landmark_xyz_unit_baseline),
            should_cancel=should_cancel,
        )
        private_context["sfm_3d_track_support_audit"] = track_audit
        if on_progress is not None:
            on_progress(stage, 1, 1)
        stage = "unscaled_ba"
        if on_progress is not None:
            on_progress(stage, 0, 1)
        refinement_options = {"refine_focal": True} if automatic_camera else {}
        unscaled = adjust_windowed_unscaled_rgb_network(
            network, intrinsics, max_landmarks_total=200000,
            max_observations_total=800000, should_cancel=should_cancel,
            **refinement_options,
        )
        private_context["unscaled"] = unscaled
        if automatic_camera and getattr(unscaled, "refined_intrinsics", None) is not None:
            intrinsics = unscaled.refined_intrinsics
        private_context["intrinsics"] = intrinsics
        if on_progress is not None:
            on_progress(stage, 1, 1)
        stage = "rtk_weighted_ba"
        if on_progress is not None:
            on_progress(stage, 0, 1)
        metric = adjust_rgb_windowed_bundle_with_private_rtk(
            network, unscaled, priors, intrinsics,
            enforce_camera_p95=True, enforce_parallax=True,
            windowed_options={
                "max_landmarks_total": 200000,
                "max_observations_total": 800000,
            }, should_cancel=should_cancel, **refinement_options,
        )
        if metric.total_cameras != len(ids):
            raise ValueError("RTK BA omitted at least one flight image")
        if automatic_camera and getattr(metric, "refined_intrinsics", None) is not None:
            intrinsics = metric.refined_intrinsics
        focal_refined = bool(getattr(unscaled, "focal_refined", False)
                             or getattr(metric, "focal_refined", False))
        if on_progress is not None:
            on_progress(stage, 1, 1)
        return PrivateRGBFlightReconstruction(
            len(ids), len(candidates), len(links), across, actual_count,
            network.reconstructed_landmarks, network.reprojection_p95_px,
            unscaled.after_rmse_px, metric, audit,
            camera_initialization_summary=camera_summary,
            focal_refined=focal_refined,
            private_series=series,
            private_network=network,
            private_intrinsics=intrinsics,
            private_distortion=np.array(distortion, copy=True),
            sfm_3d_track_support_audit=track_audit,
            feature_method=feature_method,
        )
    except (RGBPerCameraReprojectionQualityError, RGBMultiviewParallaxQualityError) as exc:
        raise PrivateRGBFlightStageError(
            stage, private_context=private_context,
            public_quality_summary=exc.audit.privacy_safe_summary(),
        ) from None
    except (ValueError, TypeError, KeyError, IndexError, cv2.error) as exc:
        raise PrivateRGBFlightStageError(
            stage, private_cause=f"{type(exc).__name__}: {exc}",
            private_context=private_context,
        ) from None
