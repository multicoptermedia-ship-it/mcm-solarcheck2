"""Private SfM landmarks -> bounded roof patches -> UNVERIFIED RGB XY raster.

The geometry comes from reconstruction, never fabricated survey controls.  The
verified camera/roof renderers are deliberately untouched.  A surface is only
interpolated inside Delaunay triangles whose edges satisfy an explicit support
gap budget.  Missing support, missing photographs and occlusion remain NoData.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Callable

import cv2
import numpy as np

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.module_plane_rectification import ModuleSurfaceGrid
from mcm_solarcheck.services.rgb_surface_tiles import plan_rgb_surface_tiles
from mcm_solarcheck.services.rgb_private_pitched_plane_candidates import (
    propose_private_rgb_pitched_planes,
)


def _cancel(callback):
    if callback is not None and callback():
        raise RuntimeError("reconstructed RGB surface candidate cancelled")


def _private_array(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RGBReconstructedPlaneCandidate:
    point_enu_private: np.ndarray = field(repr=False)
    normal_enu_private: np.ndarray = field(repr=False)
    triangles_xy_private: np.ndarray = field(repr=False)
    landmark_indices_private: np.ndarray = field(repr=False)
    support_point_count: int
    triangulation_vertex_count: int
    residual_p95_m: float
    residual_max_m: float


@dataclass(frozen=True)
class RGBReconstructedSurfaceCandidates:
    patches: tuple[RGBReconstructedPlaneCandidate, ...]
    input_point_count: int
    eligible_point_count: int
    max_support_gap_m: float
    multiview_support_checked: bool
    state: str = "unverified_sfm_reconstructed_surface_candidates"

    def privacy_safe_summary(self):
        return {
            "status": self.state,
            "surface_patch_count": len(self.patches),
            "input_landmarks": self.input_point_count,
            "eligible_landmarks": self.eligible_point_count,
            "plane_support_landmarks": sum(p.support_point_count for p in self.patches),
            "bounded_support_triangles": sum(len(p.triangles_xy_private) for p in self.patches),
            "maximum_support_gap_m": self.max_support_gap_m,
            "largest_plane_residual_p95_m": max(p.residual_p95_m for p in self.patches),
            "multiview_support_checked": self.multiview_support_checked,
            "roof_geometry_verified": False,
            "independent_survey_verified": False,
            "rgb_orthophoto_verified": False,
        }


def _support_triangles(xy, max_gap, max_vertices):
    """Use actual support vertices; never bridge a gap with a convex hull."""
    offset = xy.min(axis=0)
    local = xy - offset
    cell = max_gap / 8.
    # A deterministic spatial subset bounds Subdiv2D, retaining actual points.
    for _ in range(64):
        _, chosen = np.unique(np.floor(local / cell).astype(np.int64),
                              axis=0, return_index=True)
        if len(chosen) <= max_vertices:
            break
        cell *= 1.3
    vertices = np.unique(local[np.sort(chosen)].astype(np.float32), axis=0)
    if len(vertices) < 3:
        return np.empty((0, 3, 2)), len(vertices)
    span = vertices.max(axis=0)
    if np.max(span) > 100_000:
        raise ValueError("reconstructed surface span exceeds local ENU budget")
    # Positive one-metre margin avoids Subdiv's rectangle boundary convention.
    vertices += np.float32(1.)
    subdiv = cv2.Subdiv2D((0, 0, max(3, ceil(float(span[0])) + 3),
                          max(3, ceil(float(span[1])) + 3)))
    known = {tuple(map(float, p)) for p in vertices}
    for p in vertices:
        subdiv.insert(tuple(map(float, p)))
    accepted = []
    for tri in np.asarray(subdiv.getTriangleList()).reshape(-1, 3, 2):
        if any(tuple(map(float, p)) not in known for p in tri):
            continue
        edges = tri - np.roll(tri, 1, axis=0)
        if np.max(np.linalg.norm(edges, axis=1)) > max_gap:
            continue
        a, b = tri[1] - tri[0], tri[2] - tri[0]
        if abs(float(a[0]*b[1] - a[1]*b[0])) < 1e-9:
            continue
        accepted.append(tri.astype(np.float64) - 1. + offset)
    return np.asarray(accepted, dtype=np.float64).reshape(-1, 3, 2), len(vertices)


def fit_rgb_surface_candidates(
    points_enu,
    observation_frame_indices=None,
    observation_landmark_indices=None,
    *,
    distance_threshold_m: float = .12,
    min_plane_points: int = 30,
    min_plane_span_m: float = .5,
    min_views_per_point: int = 3,
    max_planes: int = 6,
    max_trials: int = 256,
    max_support_gap_m: float = 2.,
    max_vertices_per_plane: int = 4096,
    maximum_tilt_deg: float = 80.,
    should_cancel=None,
) -> RGBReconstructedSurfaceCandidates:
    """Existing RANSAC/SVD fit plus bounded support, without certification.

    Required observation arrays must refer to the original point array;
    distinct observing cameras, rather than repeated observations, are counted.
    This is an estimated piecewise-planar surface, not semantic roof detection.
    """
    _cancel(should_cancel)
    xyz = np.asarray(points_enu, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not 6 <= len(xyz) <= 200_000
            or not np.isfinite(xyz).all()):
        raise ValueError("bounded finite Nx3 reconstructed landmarks required")
    if (type(min_plane_points) is not int or not 24 <= min_plane_points <= min(len(xyz), 5000)
            or type(min_views_per_point) is not int or min_views_per_point < 2
            or type(max_planes) is not int or not 1 <= max_planes <= 6
            or type(max_trials) is not int or not 32 <= max_trials <= 600
            or type(max_vertices_per_plane) is not int or not 16 <= max_vertices_per_plane <= 20_000
            or not np.isfinite([distance_threshold_m, min_plane_span_m,
                                max_support_gap_m, maximum_tilt_deg]).all()
            or not .025 <= distance_threshold_m <= .5
            or not 0 < min_plane_span_m <= 100
            or not 0 < max_support_gap_m <= 50
            or not 0 < maximum_tilt_deg <= 80):
        raise ValueError("explicit bounded reconstructed plane-fit budgets required")
    fid = np.asarray(observation_frame_indices)
    lid = np.asarray(observation_landmark_indices)
    if (fid.ndim != 1 or lid.shape != fid.shape or not 24 <= len(fid) <= 800_000
            or fid.dtype.kind not in "iu" or lid.dtype.kind not in "iu"
            or np.any(fid < 0) or np.any(fid >= 1000)
            or np.any(lid < 0) or np.any(lid >= len(xyz))):
        raise ValueError("real matched multiview camera/landmark observation arrays required")
    pairs = np.unique(np.column_stack((lid, fid)), axis=0)
    views = np.bincount(pairs[:, 0], minlength=len(xyz))
    eligible = views >= min_views_per_point
    eligible_ids = np.flatnonzero(eligible)
    if len(eligible_ids) < min_plane_points:
        raise ValueError("no nondegenerate spatially supported reconstructed roof plane")
    mapping = np.full(len(xyz), -1, dtype=np.int32)
    mapping[eligible_ids] = np.arange(len(eligible_ids), dtype=np.int32)
    local_lid = mapping[lid]
    retained = local_lid >= 0
    # Reuse the already tested source fitter. Its convex hull is only a fit
    # diagnostic: the new renderer accepts bounded support triangles instead.
    proposals = propose_private_rgb_pitched_planes(
        xyz[eligible_ids], fid[retained], local_lid[retained],
        min_points=min_plane_points, min_footprint_area_m2=1.,
        max_residual_m=distance_threshold_m, min_slope_deg=0.,
        max_slope_deg=maximum_tilt_deg, max_planes=max_planes,
        max_trials=max_trials, should_cancel=should_cancel,
    )
    patches = []
    for plane in proposals.planes_private:
        _cancel(should_cancel)
        indices = eligible_ids[plane.landmark_indices_private]
        normal = plane.normal_enu_private
        origin = -plane.signed_offset_enu_private_m*normal
        count = len(indices)
        support = xyz[indices]
        span_singular = np.linalg.svd(support[:, :2]-support[:, :2].mean(axis=0),
                                     compute_uv=False)
        if span_singular[1] / np.sqrt(count) < min_plane_span_m / 4:
            continue
        triangles, vertex_count = _support_triangles(
            support[:, :2], max_support_gap_m, max_vertices_per_plane)
        if not len(triangles):
            continue
        errors = np.abs((support-origin) @ normal)
        patches.append(RGBReconstructedPlaneCandidate(
            _private_array(origin), _private_array(normal),
            _private_array(triangles), _private_array(indices, np.int64),
            count, vertex_count, float(np.percentile(errors, 95)),
            float(errors.max()),
        ))
    if not patches:
        raise ValueError("no nondegenerate spatially supported reconstructed roof plane")
    return RGBReconstructedSurfaceCandidates(
        tuple(patches), len(xyz), int(np.count_nonzero(eligible)),
        float(max_support_gap_m), True,
    )


@dataclass(frozen=True)
class RGBReconstructedSurfaceXY:
    origin_xy_private: tuple[float, float] = field(repr=False)
    height_enu_private: np.ndarray = field(repr=False)
    patch_indices_private: np.ndarray = field(repr=False)
    support_masks_private: tuple[np.ndarray, ...] = field(repr=False)
    metres_per_pixel: float
    width: int
    height: int


def _triangle_mask(triangles, width, height, x_min, y_max, step):
    mask = np.zeros((height, width), dtype=bool)
    for tri in triangles:
        px = (tri[:, 0]-x_min)/step - .5
        py = (y_max-tri[:, 1])/step - .5
        left, right = max(0, int(np.floor(px.min()))), min(width, int(np.ceil(px.max()))+1)
        top, bottom = max(0, int(np.floor(py.min()))), min(height, int(np.ceil(py.max()))+1)
        if left >= right or top >= bottom:
            continue
        xx = x_min+(np.arange(left, right)+.5)*step
        yy = y_max-(np.arange(top, bottom)+.5)*step
        crosses = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            crosses.append((b[0]-a[0])*(yy[:, None]-a[1])
                           - (b[1]-a[1])*(xx[None, :]-a[0]))
        inside = ((crosses[0] >= -1e-9) & (crosses[1] >= -1e-9) & (crosses[2] >= -1e-9)
                  | (crosses[0] <= 1e-9) & (crosses[1] <= 1e-9) & (crosses[2] <= 1e-9))
        mask[top:bottom, left:right] |= inside
    return mask


def sample_rgb_surface_xy(
    surfaces: RGBReconstructedSurfaceCandidates, *, metres_per_pixel=None,
    max_side: int = 4096, max_pixels: int = 4096*4096, should_cancel=None,
) -> RGBReconstructedSurfaceXY:
    """Topmost supported reconstructed surface in ONE common north-up XY grid."""
    _cancel(should_cancel)
    if (not isinstance(surfaces, RGBReconstructedSurfaceCandidates)
            or not surfaces.patches or type(max_side) is not int
            or not 16 <= max_side <= 4096 or type(max_pixels) is not int
            or not 1 <= max_pixels <= 4096*4096):
        raise ValueError("bounded common reconstructed XY surface grid required")
    vertices = np.concatenate([p.triangles_xy_private.reshape(-1, 2)
                               for p in surfaces.patches])
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    span = upper-lower
    if metres_per_pixel is not None and (not np.isfinite(metres_per_pixel)
                                        or metres_per_pixel <= 0):
        raise ValueError("positive candidate raster sampling distance required")
    step = max(float(metres_per_pixel or 0), float(span.max())/max_side)
    width, height = max(1, ceil(span[0]/step)), max(1, ceil(span[1]/step))
    if width*height > max_pixels:
        raise ValueError("candidate XY raster exceeds pixel budget")
    z = np.full((height, width), np.nan, dtype=np.float64)
    owner = np.full((height, width), -1, dtype=np.int16)
    masks = []
    for index, patch in enumerate(surfaces.patches):
        _cancel(should_cancel)
        mask = _triangle_mask(patch.triangles_xy_private, width, height,
                              lower[0], upper[1], step)
        masks.append(mask)
        n, o = patch.normal_enu_private, patch.point_enu_private
        xs = lower[0]+(np.arange(width)+.5)*step
        for row in range(0, height, 512):
            ys = upper[1]-(np.arange(row, min(height, row+512))+.5)*step
            plane_z = o[2]-(n[0]*(xs[None, :]-o[0])
                            + n[1]*(ys[:, None]-o[1]))/n[2]
            old = z[row:row+len(ys)]
            choose = mask[row:row+len(ys)] & (~np.isfinite(old) | (plane_z > old))
            old[choose] = plane_z[choose]
            owner[row:row+len(ys)][choose] = index
    for array in (z, owner, *masks):
        array.setflags(write=False)
    return RGBReconstructedSurfaceXY(
        (float(lower[0]), float(upper[1])), z, owner, tuple(masks),
        step, width, height,
    )


@dataclass(frozen=True)
class RGBReconstructedOrthophotoCandidate:
    bgr: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)
    grid_private: RGBReconstructedSurfaceXY = field(repr=False)
    surface_summary: dict
    camera_count: int
    decoded_source_count: int
    occluded_observations: int
    assumes_zero_distortion: bool
    max_sources_per_tile: int

    def privacy_safe_summary(self):
        support = int(np.count_nonzero(self.grid_private.patch_indices_private >= 0))
        covered = int(np.count_nonzero(self.valid_mask))
        return {
            **self.surface_summary,
            "schema": "mcm.reconstructed_rgb_orthophoto_candidate.v1",
            "status": "unverified_multiplane_rgb_orthophoto_candidate",
            "width": self.grid_private.width, "height": self.grid_private.height,
            "candidate_metres_per_pixel": self.grid_private.metres_per_pixel,
            "supported_surface_pixels": support, "covered_pixels": covered,
            "nodata_pixels": self.valid_mask.size-covered,
            "covered_fraction": covered/self.valid_mask.size,
            "rendered_supported_fraction": covered/support if support else 0.,
            "camera_count": self.camera_count,
            "decoded_source_count": self.decoded_source_count,
            "occluded_observations": self.occluded_observations,
            "maximum_sources_per_tile": self.max_sources_per_tile,
            "source_selection_is_bounded": True,
            "assumes_zero_distortion": self.assumes_zero_distortion,
            "camera_calibration_independently_verified": False,
            "georeferencing_verified": False,
            "full_flight_roof_coverage_verified": False,
            "module_analysis_verified": False,
            "thermal_registration_verified": False,
        }


def _world_pixels(grid, row, column, height, width):
    z = grid.height_enu_private[row:row+height, column:column+width]
    xx = grid.origin_xy_private[0]+(np.arange(column, column+width)+.5)*grid.metres_per_pixel
    yy = grid.origin_xy_private[1]-(np.arange(row, row+height)+.5)*grid.metres_per_pixel
    x, y = np.meshgrid(xx, yy)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _unoccluded(points, center, surfaces, grid):
    visible = np.ones(len(points), dtype=bool)
    rays = points-center
    for patch, mask in zip(surfaces.patches, grid.support_masks_private):
        n, o = patch.normal_enu_private, patch.point_enu_private
        denominator = rays @ n
        usable = np.abs(denominator) > 1e-10
        fraction = np.divide(float((o-center) @ n), denominator,
                             out=np.full(len(points), np.inf), where=usable)
        between = usable & (fraction > 1e-6) & (fraction < 1.-1e-6)
        ids = np.flatnonzero(between)
        if not len(ids):
            continue
        hit = center + fraction[ids, None]*rays[ids]
        cols = np.floor((hit[:, 0]-grid.origin_xy_private[0])/grid.metres_per_pixel).astype(np.int64)
        rows = np.floor((grid.origin_xy_private[1]-hit[:, 1])/grid.metres_per_pixel).astype(np.int64)
        inside = (cols >= 0) & (cols < grid.width) & (rows >= 0) & (rows < grid.height)
        at = np.flatnonzero(inside)
        visible[ids[at[mask[rows[at], cols[at]]]]] = False
    return visible


def render_rgb_reconstructed_surface_candidate(
    surfaces: RGBReconstructedSurfaceCandidates,
    world_to_camera,
    intrinsics: PinholeIntrinsics,
    image_provider: Callable[[int], np.ndarray],
    *, image_size: tuple[int, int], distortion_coefficients=None,
    metres_per_pixel=None, max_side: int = 4096, tile_side_px: int = 512,
    max_sources_per_tile: int = 8, should_cancel=None, on_progress=None,
) -> RGBReconstructedOrthophotoCandidate:
    """Inverse project common-XY surface pixels, select views, retain NoData.

    Provider returns one ORIGINAL BGR uint8 image at a time; it must not rewrite
    originals.  None distortion is an explicit unverified zero-distortion
    hypothesis.  No camera, survey or roof verification object is fabricated.
    """
    _cancel(should_cancel)
    poses = np.asarray(world_to_camera, dtype=np.float64)
    if (poses.ndim != 3 or poses.shape[1:] != (3, 4) or not 1 <= len(poses) <= 1000
            or not np.isfinite(poses).all()
            or not isinstance(intrinsics, PinholeIntrinsics) or not callable(image_provider)
            or type(max_sources_per_tile) is not int or not 1 <= max_sources_per_tile <= 64):
        raise ValueError("bounded reconstructed camera poses and image provider required")
    width, height = image_size
    k = np.array([[intrinsics.fx, 0., intrinsics.cx],
                  [0., intrinsics.fy, intrinsics.cy], [0., 0., 1.]], np.float64)
    if (type(width) is not int or type(height) is not int or not 2 <= width <= 10000
            or not 2 <= height <= 10000 or not np.isfinite(k).all()
            or min(intrinsics.fx, intrinsics.fy) <= 0
            or not 0 <= intrinsics.cx < width or not 0 <= intrinsics.cy < height):
        raise ValueError("finite candidate intrinsics and original image dimensions required")
    for p in poses:
        if (not np.allclose(p[:, :3].T @ p[:, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(p[:, :3]), 1., atol=1e-5)):
            raise ValueError("proper reconstructed camera rotations required")
    distortion = None
    if distortion_coefficients is not None:
        distortion = np.asarray(distortion_coefficients, dtype=np.float64)
        if distortion.shape not in ((4,), (5,), (8,), (12,), (14,)) or not np.isfinite(distortion).all():
            raise ValueError("finite OpenCV candidate distortion coefficients required")
    grid = sample_rgb_surface_xy(surfaces, metres_per_pixel=metres_per_pixel,
                                 max_side=max_side, should_cancel=should_cancel)
    plan = plan_rgb_surface_tiles(ModuleSurfaceGrid(0., 0., grid.metres_per_pixel,
                                                    grid.width, grid.height),
                                  tile_side_px=tile_side_px, max_total_pixels=4096*4096)
    centers = -np.einsum("nji,nj->ni", poses[:, :, :3], poses[:, :, 3])
    rotation_vectors = tuple(cv2.Rodrigues(p[:, :3])[0] for p in poses)
    tiles = []
    assignments = [[] for _ in poses]
    for tile_index, tile in enumerate(plan.tiles):
        _cancel(should_cancel)
        row, col = tile.row*tile_side_px, tile.column*tile_side_px
        h, w = tile.grid.height, tile.grid.width
        world = _world_pixels(grid, row, col, h, w)
        good = np.flatnonzero(np.isfinite(world[:, 2]))
        tiles.append((row, col, h, w))
        if not len(good):
            continue
        probe = world[good[np.linspace(0, len(good)-1, min(16, len(good))).astype(int)]]
        transformed = np.einsum("nij,pj->npi", poses[:, :, :3], probe) + poses[:, None, :, 3]
        front = transformed[:, :, 2] > 1e-8
        # Candidate ranking must use the same distortion model as sampling;
        # an undistorted bounding box can reject a genuinely visible camera.
        projected = np.asarray([
            cv2.projectPoints(probe, rv, p[:, 3], k, distortion)[0].reshape(-1, 2)
            for p, rv in zip(poses, rotation_vectors)
        ])
        front &= np.isfinite(projected).all(axis=2)
        px, py = projected[:, :, 0], projected[:, :, 1]
        intersects = (np.any(front, axis=1)
                      & (np.min(np.where(front, px, np.inf), axis=1) < width-1)
                      & (np.max(np.where(front, px, -np.inf), axis=1) >= 0)
                      & (np.min(np.where(front, py, np.inf), axis=1) < height-1)
                      & (np.max(np.where(front, py, -np.inf), axis=1) >= 0))
        distance = np.linalg.norm(centers-probe.mean(axis=0), axis=1)
        ranked = np.flatnonzero(intersects)
        ranked = sorted(ranked, key=lambda i: float(distance[i]))[:max_sources_per_tile]
        for i in ranked:
            assignments[i].append(tile_index)
    image_out = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    valid_out = np.zeros((grid.height, grid.width), dtype=bool)
    best_score = np.full((grid.height, grid.width), -np.inf, dtype=np.float32)
    decoded = occluded = 0
    normals = np.asarray([p.normal_enu_private for p in surfaces.patches])
    for index, tile_ids in enumerate(assignments):
        _cancel(should_cancel)
        if not tile_ids:
            continue
        source = np.asarray(image_provider(index))
        if source.shape != (height, width, 3) or source.dtype != np.uint8:
            raise ValueError("provider must return original BGR uint8 image with expected dimensions")
        decoded += 1
        rotation = rotation_vectors[index]
        for tile_id in tile_ids:
            _cancel(should_cancel)
            row, col, h, w = tiles[tile_id]
            world = _world_pixels(grid, row, col, h, w)
            ids = np.flatnonzero(np.isfinite(world[:, 2]))
            if not len(ids):
                continue
            xyz = world[ids]
            camera = xyz @ poses[index, :, :3].T + poses[index, :, 3]
            projected, _ = cv2.projectPoints(xyz, rotation, poses[index, :, 3], k, distortion)
            pixels = projected.reshape(-1, 2)
            inside = ((camera[:, 2] > 1e-8) & np.isfinite(pixels).all(axis=1)
                      & (pixels[:, 0] >= 0) & (pixels[:, 0] < width-1)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] < height-1))
            surface_index = grid.patch_indices_private[row:row+h, col:col+w].ravel()[ids]
            toward_camera = centers[index]-xyz
            distance2 = np.sum(toward_camera**2, axis=1)
            incidence = np.einsum("ij,ij->i", toward_camera, normals[surface_index]) / np.sqrt(np.maximum(distance2, 1e-12))
            inside &= incidence > 0
            visible = _unoccluded(xyz, centers[index], surfaces, grid)
            occluded += int(np.count_nonzero(inside & ~visible))
            inside &= visible
            score = incidence**2/np.maximum(distance2, 1e-12)
            old_score = best_score[row:row+h, col:col+w]
            choose = inside & (score > old_score.ravel()[ids])
            if not np.any(choose):
                continue
            map_x = np.full(h*w, -1., dtype=np.float32)
            map_y = np.full(h*w, -1., dtype=np.float32)
            selected = ids[choose]
            map_x[selected], map_y[selected] = pixels[choose].T.astype(np.float32)
            sampled = cv2.remap(source, map_x.reshape(h, w), map_y.reshape(h, w),
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            rr, cc = np.unravel_index(selected, (h, w))
            image_out[row+rr, col+cc] = sampled[rr, cc]
            valid_out[row+rr, col+cc] = True
            old_score[rr, cc] = score[choose]
        del source
        if on_progress is not None:
            on_progress(index+1, len(poses))
    if not np.any(valid_out):
        raise ValueError("reconstructed surface candidate has no visible RGB coverage")
    image_out.setflags(write=False)
    valid_out.setflags(write=False)
    return RGBReconstructedOrthophotoCandidate(
        image_out, valid_out, grid, surfaces.privacy_safe_summary(), len(poses),
        decoded, occluded, distortion is None or not np.any(distortion),
        max_sources_per_tile,
    )
