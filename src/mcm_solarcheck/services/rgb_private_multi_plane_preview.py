"""Private multi-plane RGB preview: camera selection and common metric grid.

Vertical development after the BA pitched-plane renderer. All camera indices,
candidate polygons, ENU positions and RGB pixels are STRICTLY RAM ONLY.
A projected sample is not physical visibility or a real roof label. Overlap
is treated conservatively: near-equal-height planes remain ambiguous, never
silently averaged into invented roof detail.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_private_pitched_plane_candidates import (
    PrivateRGBPitchedPlaneCandidate,
)
from mcm_solarcheck.services.rgb_private_pitched_plane_orthorectification import (
    PrivateRGBPitchedPlaneOrthopatch,
)


def choose_private_pitched_plane_cameras(
    plane: PrivateRGBPitchedPlaneCandidate,
    poses_enu_to_camera: tuple[np.ndarray, ...],
    intrinsics: PinholeIntrinsics,
    image_width: int,
    image_height: int,
    *,
    max_cameras: int = 4,
    min_sample_coverage: float = .20,
    should_cancel=None,
) -> tuple[int, ...]:
    """Select 2..8 strongest visible cameras from fitted BA geometry alone.

    Uses an XY 9x9 sample INSIDE the private candidate convex hull, then
    projects its actual plane heights into each fitted camera. Select the
    first by footprint coverage and subsequent by new coverage + baseline.
    This selects likely source frames, not verified unoccluded photographs.
    """
    if should_cancel is not None and should_cancel():
        raise RuntimeError("private multi-plane camera selection cancelled")
    if (not isinstance(plane, PrivateRGBPitchedPlaneCandidate)
            or not isinstance(intrinsics, PinholeIntrinsics)
            or type(poses_enu_to_camera) is not tuple
            or not 2 <= len(poses_enu_to_camera) <= 1000
            or type(image_width) is not int or not 32 <= image_width <= 8000
            or type(image_height) is not int or not 32 <= image_height <= 6000
            or type(max_cameras) is not int or not 2 <= max_cameras <= 8
            or type(min_sample_coverage) not in (int, float)
            or not np.isfinite(min_sample_coverage)
            or not .05 <= min_sample_coverage <= 1.):
        raise ValueError("bounded private plane camera selection required")
    k = np.array((intrinsics.fx, intrinsics.fy,
                  intrinsics.cx, intrinsics.cy), dtype=np.float64)
    n = np.asarray(plane.normal_enu_private, dtype=np.float64)
    hull = np.asarray(plane.footprint_xy_private_m, dtype=np.float64)
    offset = float(plane.signed_offset_enu_private_m)
    if (not np.isfinite(k).all() or np.any(k[:2] <= 0)
            or n.shape != (3,) or not np.isfinite(n).all()
            or not np.isclose(np.linalg.norm(n), 1., atol=1.e-4)
            or n[2] <= .1 or hull.ndim != 2 or hull.shape[1] != 2
            or len(hull) < 3 or not np.isfinite(hull).all()
            or not np.isfinite(offset)):
        raise ValueError("invalid private pitch camera geometry")
    minimum, maximum = hull.min(axis=0), hull.max(axis=0)
    if np.any(maximum - minimum <= 1.e-5):
        raise ValueError("degenerate pitched plane footprint")
    xs = minimum[0] + (np.arange(9)+.5)/9. * (maximum[0]-minimum[0])
    ys = minimum[1] + (np.arange(9)+.5)/9. * (maximum[1]-minimum[1])
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    # pointPolygonTest is strictly geometric; not a roof semantic classifier.
    interior = np.asarray([
        cv2.pointPolygonTest(np.asarray(hull, np.float32), tuple(map(float, p)),
                             False) >= 0 for p in xy
    ])
    xy = xy[interior]
    if len(xy) < 4:
        raise ValueError("insufficient pitched plane area for camera selection")
    z = (-offset - n[0]*xy[:, 0] - n[1]*xy[:, 1]) / n[2]
    points = np.column_stack((xy, z))
    masks = []
    centers = []
    for i, pose in enumerate(poses_enu_to_camera):
        if should_cancel is not None and i % 16 == 0 and should_cancel():
            raise RuntimeError("private multi-plane camera selection cancelled")
        p = np.asarray(pose, dtype=np.float64)
        if p.shape != (3, 4) or not np.isfinite(p).all():
            raise ValueError("invalid final RGB BA camera pose")
        r, t = p[:, :3], p[:, 3]
        if (not np.allclose(r.T @ r, np.eye(3), atol=1.e-4)
                or not np.isclose(np.linalg.det(r), 1., atol=1.e-4)):
            raise ValueError("invalid final RGB BA camera rotation")
        centers.append(-r.T @ t)
        local = points @ r.T + t
        depth = local[:, 2]
        safe = np.where(depth > 1.e-8, depth, 1.)
        u = k[0]*local[:, 0]/safe + k[2]
        v = k[1]*local[:, 1]/safe + k[3]
        masks.append((depth > 1.e-8) & np.isfinite(u) & np.isfinite(v)
                     & (u >= 0) & (v >= 0)
                     & (u < image_width-1) & (v < image_height-1))
    valid = [i for i, mask in enumerate(masks)
             if float(np.mean(mask)) >= min_sample_coverage]
    if len(valid) < 2:
        raise ValueError("fewer than two fitted cameras cover pitched RGB plane")
    centers = np.asarray(centers, dtype=np.float64)
    selected: list[int] = []
    covered = np.zeros(len(xy), dtype=bool)
    radius = max(float(np.linalg.norm(maximum-minimum)), 1.)
    # At each step prefer coverage gain, then distinct viewing centers.
    # Always add two cameras to avoid an accidental single-view preview.
    while len(selected) < min(max_cameras, len(valid)):
        def quality(i):
            additional = np.count_nonzero(masks[i] & ~covered)/len(xy)
            base = float(np.mean(masks[i]))
            separation = (max(
                (float(np.linalg.norm(centers[i]-centers[j])) for j in selected),
                default=0.
            ))
            return (additional, min(separation/radius, 1.), base, -i)
        winner = max((i for i in valid if i not in selected), key=quality)
        selected.append(winner)
        covered |= masks[winner]
        if len(selected) >= 2 and np.all(covered):
            break
    return tuple(selected)


@dataclass(frozen=True)
class PrivateRGBMultiPlaneOrthopreview:
    rgb_private: np.ndarray
    valid_mask_private: np.ndarray
    ambiguous_overlap_mask_private: np.ndarray
    selected_plane_index_private: np.ndarray
    top_left_xy_enu_private_m: tuple[float, float]
    meters_per_pixel: float
    plane_count: int
    source_camera_counts: tuple[int, ...]
    ambiguous_overlap_pixels: int
    valid_pixel_count: int
    state: str = "private_multi_pitched_3d_rgb_preview_unverified"

    def privacy_safe_summary(self) -> dict:
        total = int(self.valid_mask_private.size)
        return {
            "status": self.state,
            "plane_candidate_count": self.plane_count,
            "width_px": int(self.rgb_private.shape[1]),
            "height_px": int(self.rgb_private.shape[0]),
            "source_cameras_min": min(self.source_camera_counts),
            "source_cameras_max": max(self.source_camera_counts),
            "valid_pixel_count": self.valid_pixel_count,
            "valid_fraction_of_union_bbox": self.valid_pixel_count/total,
            "ambiguous_plane_overlap_pixels": self.ambiguous_overlap_pixels,
            "ambiguous_fraction_of_union_bbox": self.ambiguous_overlap_pixels/total,
            "roof_surface_identity_verified": False,
            "roof_geometry_verified": False,
            "independent_survey_verified": False,
            "orthophoto_verified": False,
        }


def compose_private_pitched_rgb_orthopatches(
    planes: tuple[PrivateRGBPitchedPlaneCandidate, ...],
    patches: tuple[PrivateRGBPitchedPlaneOrthopatch, ...],
    *,
    overlap_height_tolerance_m: float = .15,
    max_side: int = 2048,
    should_cancel=None,
) -> PrivateRGBMultiPlaneOrthopreview:
    """Place independent pitched patches on one north-up ENU grid.

    Pixel priority is highest candidate plane Z only if it is clearly above
    other candidates. Similar-height overlap stays INVALID/ambiguous to avoid
    manufacturing roof ridges or false seam-free orthophotos.
    """
    if should_cancel is not None and should_cancel():
        raise RuntimeError("private multi-plane RGB composition cancelled")
    if (type(planes) is not tuple or type(patches) is not tuple
            or not 2 <= len(planes) <= 6 or len(planes) != len(patches)
            or any(not isinstance(p, PrivateRGBPitchedPlaneCandidate) for p in planes)
            or any(not isinstance(p, PrivateRGBPitchedPlaneOrthopatch) for p in patches)
            or type(overlap_height_tolerance_m) not in (int, float)
            or not np.isfinite(overlap_height_tolerance_m)
            or not .025 <= overlap_height_tolerance_m <= .5
            or type(max_side) is not int or not 64 <= max_side <= 2048):
        raise ValueError("bounded multiple private pitched orthopatches required")
    resolution = float(patches[0].meters_per_pixel)
    if (not np.isfinite(resolution) or not .01 <= resolution <= .5
            or any(not np.isclose(p.meters_per_pixel, resolution, atol=1.e-10)
                   for p in patches)):
        raise ValueError("pitch patches must share a common metric resolution")
    hulls = tuple(np.asarray(p.footprint_xy_private_m, dtype=np.float64)
                  for p in planes)
    if any(h.ndim != 2 or h.shape[1] != 2 or len(h) < 3
           or not np.isfinite(h).all() for h in hulls):
        raise ValueError("valid private candidate hulls required")
    min_x = min(float(h[:, 0].min()) for h in hulls)
    max_x = max(float(h[:, 0].max()) for h in hulls)
    min_y = min(float(h[:, 1].min()) for h in hulls)
    max_y = max(float(h[:, 1].max()) for h in hulls)
    width = int(np.ceil((max_x-min_x)/resolution))
    height = int(np.ceil((max_y-min_y)/resolution))
    if (width < 2 or height < 2 or width > max_side or height > max_side
            or width*height > 2_000_000):
        raise ValueError("private multi-plane RGB canvas exceeds resource budget")
    xx, yy = np.meshgrid(
        min_x + (np.arange(width)+.5)*resolution,
        max_y - (np.arange(height)+.5)*resolution,
    )
    rgb = np.zeros((height,width,3), dtype=np.uint8)
    winner = np.full((height,width), -1, dtype=np.int16)
    highest = np.full((height,width), -np.inf)
    second = np.full((height,width), -np.inf)
    coverage = np.zeros((height,width), dtype=np.uint8)
    for index, (plane, patch, hull) in enumerate(zip(planes, patches, hulls)):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("private multi-plane RGB composition cancelled")
        if (not isinstance(patch.rgb_private, np.ndarray)
                or not isinstance(patch.valid_mask_private, np.ndarray)
                or patch.rgb_private.ndim != 3
                or patch.rgb_private.shape[2] != 3
                or patch.rgb_private.dtype != np.uint8
                or patch.valid_mask_private.dtype != bool
                or patch.valid_mask_private.shape != patch.rgb_private.shape[:2]
                or not np.isfinite(patch.top_left_xy_enu_private_m).all()
                or patch.source_camera_count < 1
                or not np.isclose(patch.top_left_xy_enu_private_m[0],
                                  float(hull[:, 0].min()), atol=1.e-6)
                or not np.isclose(patch.top_left_xy_enu_private_m[1],
                                  float(hull[:, 1].max()), atol=1.e-6)):
            raise ValueError("misaligned or malformed private pitched RGB patch")
        n = np.asarray(plane.normal_enu_private, dtype=np.float64)
        off = float(plane.signed_offset_enu_private_m)
        if (n.shape != (3,) or not np.isfinite(n).all()
                or not np.isclose(np.linalg.norm(n), 1., atol=1.e-4)
                or n[2] <= .1 or not np.isfinite(off)):
            raise ValueError("invalid private pitched plane for common canvas")
        local_x = ((xx - patch.top_left_xy_enu_private_m[0])/resolution -.5
                   ).astype(np.float32)
        local_y = ((patch.top_left_xy_enu_private_m[1] - yy)/resolution -.5
                   ).astype(np.float32)
        # Nearest-neighbour is deliberately used for both RGB and support
        # mask to prevent interpolation across a missing-data roof edge.
        sampled = cv2.remap(
            patch.rgb_private, local_x, local_y,
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0),
        )
        supported = cv2.remap(
            patch.valid_mask_private.astype(np.uint8),
            local_x, local_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        z = (-off - n[0]*xx - n[1]*yy)/n[2]
        coverage += supported.astype(np.uint8)
        above = supported & (z > highest)
        second = np.where(above, highest, np.where(supported & (z > second),
                                                  z, second))
        highest = np.where(above, z, highest)
        rgb[above] = sampled[above]
        winner[above] = index
    # Adjacent matched roof facets may meet at the same ridge height.
    # Such overlaps cannot be disambiguated without true roof boundaries.
    ambiguous = (coverage >= 2) & (
        highest-second <= overlap_height_tolerance_m
    )
    valid = (coverage > 0) & ~ambiguous
    rgb[~valid] = 0
    winner[~valid] = -1
    for array in (rgb, valid, ambiguous, winner):
        array.setflags(write=False)
    return PrivateRGBMultiPlaneOrthopreview(
        rgb, valid, ambiguous, winner,
        (min_x,max_y), resolution, len(planes),
        tuple(p.source_camera_count for p in patches),
        int(np.count_nonzero(ambiguous)), int(np.count_nonzero(valid)),
    )
