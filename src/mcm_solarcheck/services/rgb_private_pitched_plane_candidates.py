"""Private ENU pitched-plane HYPOTHESES from already optimized RGB 3D points.

This is the first roof-geometry *candidate* stage after SfM and RTK-weighted
BA, NOT a detected/verified roof, surveyed surface, polygon or orthophoto.
All 3D geometry and landmark IDs remain in private memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PrivateRGBPitchedPlaneCandidate:
    """Private metric hypothesis; never serialize its plane or point indices."""

    normal_enu_private: np.ndarray
    signed_offset_enu_private_m: float
    landmark_indices_private: np.ndarray
    footprint_xy_private_m: np.ndarray
    slope_deg: float
    supporting_cameras: int
    residual_p95_m: float
    footprint_area_m2: float


@dataclass(frozen=True)
class PrivateRGBPitchedPlaneCandidates:
    planes_private: tuple[PrivateRGBPitchedPlaneCandidate, ...]
    landmark_count: int
    state: str = "unverified_pitched_plane_hypotheses_not_roof_surfaces"

    def privacy_safe_summary(self) -> dict:
        return {
            "schema": "mcm.private_rgb_pitched_plane_hypotheses.v1",
            "status": self.state,
            "plane_candidate_count": len(self.planes_private),
            "reconstructed_3d_landmark_count": self.landmark_count,
            "total_supported_landmarks": sum(
                len(p.landmark_indices_private) for p in self.planes_private
            ),
            "candidate_slope_min_deg": (
                min(p.slope_deg for p in self.planes_private)
                if self.planes_private else None
            ),
            "candidate_slope_max_deg": (
                max(p.slope_deg for p in self.planes_private)
                if self.planes_private else None
            ),
            "worst_candidate_plane_residual_p95_m": (
                max(p.residual_p95_m for p in self.planes_private)
                if self.planes_private else None
            ),
            "minimum_candidate_supporting_cameras": (
                min(p.supporting_cameras for p in self.planes_private)
                if self.planes_private else None
            ),
            "independent_survey_verified": False,
            "roof_surface_identity_verified": False,
            "roof_geometry_verified": False,
            "orthophoto_verified": False,
        }


def _xy_hull(points: np.ndarray) -> tuple[np.ndarray, float]:
    hull = cv2.convexHull(
        np.ascontiguousarray(points[:, :2], dtype=np.float32)
    ).reshape(-1, 2)
    area = float(cv2.contourArea(hull))
    return np.asarray(hull, dtype=np.float64), area


def _fit_plane_svd(points: np.ndarray):
    center = np.mean(points, axis=0)
    _, singular, right = np.linalg.svd(points - center, full_matrices=False)
    if len(singular) != 3 or singular[1] <= 1e-6:
        return None
    normal = right[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    return normal, -float(normal @ center)


def propose_private_rgb_pitched_planes(
    landmarks_enu_private_m: np.ndarray,
    observation_frame_indices: np.ndarray,
    observation_landmark_indices: np.ndarray,
    *,
    min_points: int = 30,
    min_footprint_area_m2: float = 4.,
    max_residual_m: float = .12,
    min_slope_deg: float = 5.,
    max_slope_deg: float = 70.,
    max_planes: int = 4,
    max_trials: int = 240,
    should_cancel=None,
) -> PrivateRGBPitchedPlaneCandidates:
    """RANSAC+SVD split of planar 3D patches, still PRIVATE/UNVERIFIED.

    A patch needs real 3D points, footprint area and >=3 cameras that observed
    its points. The default rejects flat ground. Explicit min_slope_deg=0
    also permits flat roof/ground hypotheses for a general RGB surface preview;
    it never identifies a plane semantically as a roof.
    """
    if should_cancel is not None and should_cancel():
        raise RuntimeError("private RGB pitched-plane proposal cancelled")
    if (type(min_points) is not int or not 24 <= min_points <= 5000
            or type(max_planes) is not int or not 1 <= max_planes <= 6
            or type(max_trials) is not int or not 32 <= max_trials <= 600
            or type(min_footprint_area_m2) not in (int, float)
            or not np.isfinite(min_footprint_area_m2)
            or not 1. <= min_footprint_area_m2 <= 200.
            or type(max_residual_m) not in (int, float)
            or not np.isfinite(max_residual_m)
            or not .025 <= max_residual_m <= .5
            or not 0. <= min_slope_deg < max_slope_deg <= 80.):
        raise ValueError("bounded private 3D pitched-plane proposal policy required")
    xyz = np.asarray(landmarks_enu_private_m, dtype=np.float64)
    fid = np.asarray(observation_frame_indices)
    lid = np.asarray(observation_landmark_indices)
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,)
            or not 24 <= len(xyz) <= 200000 or not np.isfinite(xyz).all()
            or fid.ndim != 1 or lid.ndim != 1 or len(fid) != len(lid)
            or not 24 <= len(fid) <= 800000
            or not np.issubdtype(fid.dtype, np.integer)
            or not np.issubdtype(lid.dtype, np.integer)
            or np.any(fid < 0) or np.any(fid >= 1000)
            or np.any(lid < 0) or np.any(lid >= len(xyz))):
        raise ValueError("finite bounded retained multiview ENU 3D points required")

    # Stable order irrespective of incoming landmark ordering, for reproducible
    # hypotheses. Deterministic RANSAC does NOT certify a physical roof.
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    remaining = order.copy()
    rng = np.random.default_rng(20261029)
    planes = []
    for _ in range(max_planes):
        if len(remaining) < min_points:
            break
        if should_cancel is not None and should_cancel():
            raise RuntimeError("private RGB pitched-plane proposal cancelled")
        # Candidate generation runs on a bounded subset. Fit and validate on
        # ALL remaining real retained 3D points, never on a sampled report.
        if len(remaining) > 2048:
            subset = np.sort(rng.choice(remaining, 2048, replace=False))
        else:
            subset = remaining
        sample_xyz = xyz[subset]
        best = None
        best_count = 0
        for trial in range(max_trials):
            if should_cancel is not None and trial % 32 == 0 and should_cancel():
                raise RuntimeError("private RGB pitched-plane proposal cancelled")
            tri = sample_xyz[rng.choice(len(subset), 3, replace=False)]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            size = float(np.linalg.norm(normal))
            if size < 1e-8:
                continue
            normal /= size
            if normal[2] < 0:
                normal = -normal
            slope = float(np.degrees(np.arccos(np.clip(normal[2], -1., 1.))))
            if slope < min_slope_deg or slope > max_slope_deg:
                continue
            offset = -float(normal @ tri[0])
            hits = int(np.count_nonzero(
                np.abs(sample_xyz @ normal + offset) <= max_residual_m
            ))
            if hits > best_count:
                best = (normal.copy(), offset)
                best_count = hits
        if best is None or best_count < min_points:
            break
        normal, offset = best
        distances = np.abs(xyz[remaining] @ normal + offset)
        inliers = remaining[distances <= max_residual_m]
        if len(inliers) < min_points:
            break
        refined = _fit_plane_svd(xyz[inliers])
        if refined is None:
            break
        normal, offset = refined
        slope = float(np.degrees(np.arccos(np.clip(normal[2], -1., 1.))))
        if slope < min_slope_deg or slope > max_slope_deg:
            break
        distances = np.abs(xyz[remaining] @ normal + offset)
        inliers = remaining[distances <= max_residual_m]
        if len(inliers) < min_points:
            break
        hull, area = _xy_hull(xyz[inliers])
        if len(hull) < 3 or area < min_footprint_area_m2:
            break
        cameras = np.unique(fid[np.isin(lid, inliers)])
        if len(cameras) < 3:
            break
        residual = float(np.percentile(
            np.abs(xyz[inliers] @ normal + offset), 95
        ))
        for arr in (normal, inliers, hull):
            arr.setflags(write=False)
        planes.append(PrivateRGBPitchedPlaneCandidate(
            normal, offset, inliers, hull, slope, len(cameras),
            residual, area,
        ))
        remaining = remaining[~np.isin(remaining, inliers)]
    return PrivateRGBPitchedPlaneCandidates(tuple(planes), len(xyz))
