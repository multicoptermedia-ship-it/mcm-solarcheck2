"""Private per-camera pixel evidence audit AFTER RTK-weighted RGB BA.

Whole-flight RMSE can hide single bad cameras. This checks every observed
camera and every retained 3D landmark against the final fitted private geometry.
Only aggregate counts/error percentiles are reportable. No survey/roof claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics


@dataclass(frozen=True)
class RGBPerCameraReprojectionAudit:
    camera_count: int
    landmark_count: int
    observation_count: int
    minimum_observations_per_camera: int
    minimum_cameras_per_landmark: int
    worst_camera_p95_px: float
    flight_p95_px: float
    flight_max_px: float
    cameras_above_p95_limit: int = 0
    camera_p95_limit_px: float = 6.
    state: str = "per_camera_reprojection_checked_not_an_orthophoto"

    def privacy_safe_summary(self) -> dict:
        return {
            "camera_reprojection_audit_completed": True,
            "camera_count": self.camera_count,
            "landmark_count": self.landmark_count,
            "observation_count": self.observation_count,
            "min_observations_per_camera": self.minimum_observations_per_camera,
            "min_distinct_cameras_per_landmark": self.minimum_cameras_per_landmark,
            "worst_camera_reprojection_p95_px": self.worst_camera_p95_px,
            "all_camera_reprojection_p95_px": self.flight_p95_px,
            "all_camera_reprojection_max_px": self.flight_max_px,
            "cameras_above_p95_limit": self.cameras_above_p95_limit,
            "camera_p95_limit_px": self.camera_p95_limit_px,
            "camera_reprojection_quality_passed": (
                self.cameras_above_p95_limit == 0
            ),
            "independent_survey_verified": False,
            "roof_geometry_verified": False,
            "orthophoto_verified": False,
        }


class RGBPerCameraReprojectionQualityError(ValueError):
    """Strict pixel-quality rejection with an aggregate-only private audit.

    The message never contains camera identifiers, GPS, paths or point data.
    Callers may publish ONLY audit.privacy_safe_summary(), not the geometry.
    """

    def __init__(self, audit: RGBPerCameraReprojectionAudit):
        if not isinstance(audit, RGBPerCameraReprojectionAudit):
            raise TypeError("camera pixel quality diagnostics required")
        if audit.cameras_above_p95_limit < 1:
            raise ValueError("cannot reject a passing camera pixel audit")
        self.audit = audit
        super().__init__("individual RGB camera reprojection p95 exceeds tolerance")


def audit_rgb_per_camera_reprojection(
    poses_world_to_camera: tuple[np.ndarray, ...],
    landmarks_xyz: np.ndarray,
    frame_indices: np.ndarray,
    landmark_indices: np.ndarray,
    observed_pixels_xy: np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    max_camera_p95_px: float = 6.,
    min_observations_per_camera: int = 8,
    enforce_camera_p95: bool = True,
    should_cancel=None,
) -> RGBPerCameraReprojectionAudit:
    """Reject invalid/out-of-view geometry and sparse single-view landmarks.

    Errors are computed for *all* retained observations, not an inlier subset
    selected after optimizing. All cameras and all landmarks must contribute.
    """
    if should_cancel is not None and should_cancel():
        raise RuntimeError("RGB camera reprojection audit cancelled")
    count = len(poses_world_to_camera) if type(poses_world_to_camera) is tuple else 0
    if (not 3 <= count <= 1000 or
            type(min_observations_per_camera) is not int or
            not 8 <= min_observations_per_camera <= 1000 or
            type(enforce_camera_p95) is not bool or
            type(max_camera_p95_px) not in (int, float) or
            not np.isfinite(max_camera_p95_px) or
            not .25 <= max_camera_p95_px <= 20 or
            not isinstance(intrinsics, PinholeIntrinsics)):
        raise ValueError("bounded complete RGB camera audit input required")
    poses = np.asarray(poses_world_to_camera, dtype=np.float64)
    xyz = np.asarray(landmarks_xyz, dtype=np.float64)
    fid = np.asarray(frame_indices)
    lid = np.asarray(landmark_indices)
    pixels = np.asarray(observed_pixels_xy, dtype=np.float64)
    k = np.asarray(
        (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy),
        dtype=np.float64,
    )
    if (poses.shape != (count, 3, 4) or
            xyz.ndim != 2 or xyz.shape[1] != 3 or not 8 <= len(xyz) <= 200000
            or fid.ndim != 1 or lid.ndim != 1
            or not np.issubdtype(fid.dtype, np.integer)
            or not np.issubdtype(lid.dtype, np.integer)
            or not 24 <= len(fid) <= 800000 or len(lid) != len(fid)
            or pixels.shape != (len(fid), 2)
            or not np.isfinite(poses).all() or not np.isfinite(xyz).all()
            or not np.isfinite(pixels).all() or not np.isfinite(k).all()
            or min(k[:2]) <= 0 or np.any(fid < 0) or np.any(fid >= count)
            or np.any(lid < 0) or np.any(lid >= len(xyz))):
        raise ValueError("invalid finite full-flight RGB camera geometry")
    for rotation in poses[:, :, :3]:
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4)):
            raise ValueError("invalid RGB camera rotation during pixel audit")
    # Avoid expensive per-frame Python loops when checking the whole 160 flight.
    cameraxyz = np.einsum(
        "nij,nj->ni", poses[fid, :, :3], xyz[lid]
    ) + poses[fid, :, 3]
    if (not np.isfinite(cameraxyz).all()
            or np.any(cameraxyz[:, 2] <= 1e-6)):
        raise ValueError("RGB landmark behind or too close to a fitted camera")
    projected = np.column_stack((
        k[0] * cameraxyz[:, 0] / cameraxyz[:, 2] + k[2],
        k[1] * cameraxyz[:, 1] / cameraxyz[:, 2] + k[3],
    ))
    error = np.linalg.norm(projected - pixels, axis=1)
    if not np.isfinite(error).all():
        raise ValueError("nonfinite fitted RGB reprojection residual")
    per_camera = np.bincount(fid, minlength=count)
    if np.min(per_camera) < min_observations_per_camera:
        raise ValueError("insufficient final fitted 3D pixel support in an RGB camera")
    # Uniqueness matters: dozens of duplicate pixels in one camera do not
    # make an actual multi-view 3D point.
    distinct = np.unique(
        lid.astype(np.int64) * count + fid.astype(np.int64)
    )
    per_landmark = np.bincount(distinct // count, minlength=len(xyz))
    if np.min(per_landmark) < 2:
        raise ValueError("retained RGB landmark lacks distinct multi-view support")
    per_camera_p95 = [
        float(np.percentile(error[fid == i], 95)) for i in range(count)
    ]
    worst = max(per_camera_p95)
    above = sum(value > max_camera_p95_px for value in per_camera_p95)
    audit = RGBPerCameraReprojectionAudit(
        count, len(xyz), len(fid), int(np.min(per_camera)),
        int(np.min(per_landmark)), worst,
        float(np.percentile(error, 95)), float(np.max(error)),
        above, float(max_camera_p95_px),
    )
    if above and enforce_camera_p95:
        raise RGBPerCameraReprojectionQualityError(audit)
    return audit
