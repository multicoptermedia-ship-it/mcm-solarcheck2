"""In-memory inverse projection of RGB onto a PRIVATE pitched 3D plane.

World ENU XY output grid -> fitted plane height -> final calibrated camera
pose -> bilinear UNDISTORTED source RGB pixels. Unlike a 2D stitch, every
output sample is anchored on a different 3D plane point. Not a surveyed roof,
visibility/occlusion solver, radiometric seam correction or final orthophoto.
No image, location, plane, source IDs or coordinates are serialized.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_private_pitched_plane_candidates import (
    PrivateRGBPitchedPlaneCandidate,
)


@dataclass(frozen=True)
class PrivateRGBPitchedPlaneOrthopatch:
    rgb_private: np.ndarray
    valid_mask_private: np.ndarray
    top_left_xy_enu_private_m: tuple[float, float]
    meters_per_pixel: float
    source_camera_count: int
    planar_pixel_count: int
    valid_pixel_count: int
    multi_camera_pixel_count: int
    state: str = "private_pitched_rgb_orthopatch_unverified"

    def privacy_safe_summary(self) -> dict:
        return {
            "status": self.state,
            "width_px": int(self.rgb_private.shape[1]),
            "height_px": int(self.rgb_private.shape[0]),
            "source_camera_count": self.source_camera_count,
            "planar_pixel_count": self.planar_pixel_count,
            "valid_pixel_count": self.valid_pixel_count,
            "multi_camera_pixel_count": self.multi_camera_pixel_count,
            "footprint_coverage_fraction": (
                self.valid_pixel_count / self.planar_pixel_count
            ),
            "multi_camera_coverage_fraction": (
                self.multi_camera_pixel_count / self.planar_pixel_count
            ),
            "independent_survey_verified": False,
            "roof_geometry_verified": False,
            "roof_surface_identity_verified": False,
            "orthophoto_verified": False,
        }


def render_private_pitched_rgb_orthopatch(
    plane: PrivateRGBPitchedPlaneCandidate,
    undistorted_rgb_images: tuple[np.ndarray, ...],
    world_enu_to_camera_poses: tuple[np.ndarray, ...],
    intrinsics: PinholeIntrinsics,
    *,
    meters_per_pixel: float = .05,
    max_side: int = 1024,
    should_cancel=None,
) -> PrivateRGBPitchedPlaneOrthopatch:
    """Render an ENU-north-up, pitched-plane preview; no disk writes.

    INPUT RGB IMAGES MUST ALREADY BE UNDISTORTED at the exact intrinsics
    resolution. No Gimbal/EXIF-derived camera orientation, no 2D homography
    over the entire flight, no blending of unrelated physical surfaces.
    """
    if should_cancel is not None and should_cancel():
        raise RuntimeError("private pitched RGB projection cancelled")
    if (not isinstance(plane, PrivateRGBPitchedPlaneCandidate)
            or not isinstance(intrinsics, PinholeIntrinsics)
            or type(undistorted_rgb_images) is not tuple
            or type(world_enu_to_camera_poses) is not tuple
            or not 1 <= len(undistorted_rgb_images) <= 8
            or len(undistorted_rgb_images) != len(world_enu_to_camera_poses)
            or type(meters_per_pixel) not in (int, float)
            or not np.isfinite(meters_per_pixel)
            or not .01 <= meters_per_pixel <= .50
            or type(max_side) is not int or not 64 <= max_side <= 2048):
        raise ValueError("bounded private pitched-plane projection inputs required")

    n = np.asarray(plane.normal_enu_private, dtype=np.float64)
    hull = np.asarray(plane.footprint_xy_private_m, dtype=np.float64)
    offset = plane.signed_offset_enu_private_m
    k = np.asarray(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
        dtype=np.float64,
    )
    if (n.shape != (3,) or hull.ndim != 2 or hull.shape[1:] != (2,)
            or not 3 <= len(hull) <= 200000 or not np.isfinite(n).all()
            or not np.isfinite(hull).all() or not np.isfinite(offset)
            or not np.isfinite(k).all() or np.any(k[:2] <= 0)
            or n[2] <= .1 or not np.isclose(np.linalg.norm(n), 1., atol=1e-4)
            or not np.isfinite(plane.footprint_area_m2)
            or plane.footprint_area_m2 <= 0):
        raise ValueError("invalid private pitched plane/intrinsics geometry")
    minimum = np.min(hull, axis=0)
    maximum = np.max(hull, axis=0)
    dx, dy = maximum - minimum
    width = int(np.ceil(dx / meters_per_pixel))
    height = int(np.ceil(dy / meters_per_pixel))
    if (width < 2 or height < 2 or width > max_side or height > max_side
            or width * height > 2_000_000):
        raise ValueError("private orthopatch resolution exceeds resource budget")

    # north-up output: column j corresponds to east+, row i to north-.
    xs = minimum[0] + (np.arange(width) + .5) * meters_per_pixel
    ys = maximum[1] - (np.arange(height) + .5) * meters_per_pixel
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((
        xx.ravel(), yy.ravel(),
        (-offset - n[0]*xx.ravel() - n[1]*yy.ravel()) / n[2],
    ))
    # The convex hull is NOT a verified real roof boundary; keep ALL samples
    # outside this provisional inlier footprint strictly transparent/invalid.
    poly = np.column_stack((
        (hull[:, 0] - minimum[0]) / meters_per_pixel - .5,
        (maximum[1] - hull[:, 1]) / meters_per_pixel - .5,
    ))
    footprint_mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(
        footprint_mask, np.round(poly).astype(np.int32), 1,
    )
    footprint = footprint_mask.ravel().astype(bool)
    planar_count = int(np.count_nonzero(footprint))
    if planar_count < 4:
        raise ValueError("private pitched plane has no usable raster footprint")
    rgb_sum = np.zeros((height*width, 3), dtype=np.float64)
    weights = np.zeros(height*width, dtype=np.float64)
    views = np.zeros(height*width, dtype=np.uint8)
    for image, pose in zip(undistorted_rgb_images, world_enu_to_camera_poses):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("private pitched RGB projection cancelled")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or not 32 <= image.shape[0] <= 6000
                or not 32 <= image.shape[1] <= 8000):
            raise ValueError("full-size undistorted RGB camera image required")
        camera = np.asarray(pose, dtype=np.float64)
        if camera.shape != (3,4) or not np.isfinite(camera).all():
            raise ValueError("finite fitted ENU-to-camera extrinsics required")
        rot, tr = camera[:, :3], camera[:, 3]
        if (not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rot), 1., atol=1e-4)):
            raise ValueError("proper final fitted RGB camera rotation required")
        local = points @ rot.T + tr
        depth = local[:, 2]
        z = np.where(depth > 1e-8, depth, 1.)
        u = k[0] * local[:, 0] / z + k[2]
        v = k[1] * local[:, 1] / z + k[3]
        in_image = (
            footprint & np.isfinite(u) & np.isfinite(v)
            & (depth > 1e-8)
            & (u >= 0) & (v >= 0)
            & (u < image.shape[1]-1) & (v < image.shape[0]-1)
        )
        if not np.any(in_image):
            continue
        # Invalid remap entries point to a harmless constant; they NEVER
        # contribute via the separate in_image mask.
        mx = np.where(in_image, u, 0).astype(np.float32).reshape(height,width)
        my = np.where(in_image, v, 0).astype(np.float32).reshape(height,width)
        sampled = cv2.remap(
            image, mx, my, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0),
        ).reshape(-1,3)
        # Keep a deterministic arithmetic mean from all physically visible
        # sources; no claim of occlusion handling or seam-quality validation.
        rgb_sum[in_image] += sampled[in_image]
        weights[in_image] += 1.
        views[in_image] += 1
    covered = weights > 0
    rgb = np.zeros((height*width,3), dtype=np.uint8)
    if np.any(covered):
        rgb[covered] = np.rint(
            rgb_sum[covered] / weights[covered, None]
        ).clip(0,255).astype(np.uint8)
    rgb = rgb.reshape(height,width,3)
    mask = covered.reshape(height,width)
    for array in (rgb,mask):
        array.setflags(write=False)
    return PrivateRGBPitchedPlaneOrthopatch(
        rgb, mask, (float(minimum[0]),float(maximum[1])),
        float(meters_per_pixel), len(undistorted_rgb_images),
        planar_count, int(np.count_nonzero(covered)),
        int(np.count_nonzero(views >= 2)),
    )
