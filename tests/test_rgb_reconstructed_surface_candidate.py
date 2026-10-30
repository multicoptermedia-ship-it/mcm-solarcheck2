"""Actual synthetic 3D planes and pixel projection; no customer photograph."""
import json

import numpy as np
import pytest

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_reconstructed_surface_candidate import (
    fit_rgb_surface_candidates, sample_rgb_surface_xy,
    render_rgb_reconstructed_surface_candidate,
)


def _patch(x0, x1, y0=-1., y1=1., *, offset=0., tilt_x=0., tilt_y=0.):
    x, y = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 7))
    return np.column_stack((x.ravel(), y.ravel(),
                            (offset+tilt_x*x+tilt_y*y).ravel()))


def _pose(center, target):
    center, target = np.asarray(center, float), np.asarray(target, float)
    forward = target-center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return np.column_stack((rotation, -rotation @ center))


def _pixel(grid, x, y):
    col = int((x-grid.origin_xy_private[0])/grid.metres_per_pixel)
    row = int((grid.origin_xy_private[1]-y)/grid.metres_per_pixel)
    return row, col


def _fit(cloud, **kwargs):
    # Three synthetic overhead views observe the known scene points; the
    # indices are actual fixture provenance, never invented in production.
    return fit_rgb_surface_candidates(
        cloud, np.tile(np.arange(3), len(cloud)),
        np.repeat(np.arange(len(cloud)), 3), **kwargs,
    )


def test_two_genuinely_tilted_planes_keep_gap_nodata_and_private_geometry():
    cloud = np.concatenate((_patch(-4., -2., tilt_x=.3),
                            _patch(2., 4., offset=2., tilt_y=-.25)))
    original = cloud.copy()
    fid = np.tile(np.arange(3), len(cloud))
    lid = np.repeat(np.arange(len(cloud)), 3)
    surfaces = fit_rgb_surface_candidates(
        cloud, fid, lid, distance_threshold_m=.025, max_support_gap_m=.8,
    )
    assert len(surfaces.patches) == 2
    assert all(p.residual_p95_m < 1e-5 for p in surfaces.patches)
    grid = sample_rgb_surface_xy(surfaces, metres_per_pixel=.05, max_side=256)
    left = _pixel(grid, -3., 0.)
    right = _pixel(grid, 3., 0.)
    gap = _pixel(grid, 0., 0.)
    assert grid.height_enu_private[left] == pytest.approx(-.9, abs=.025)
    assert grid.height_enu_private[right] == pytest.approx(2., abs=.025)
    assert grid.patch_indices_private[gap] == -1
    assert np.isnan(grid.height_enu_private[gap])
    assert np.array_equal(cloud, original)
    assert not surfaces.patches[0].point_enu_private.flags.writeable
    assert "normal_enu_private" not in repr(surfaces)
    report = surfaces.privacy_safe_summary()
    assert report["multiview_support_checked"] is True
    assert all(value is False for key, value in report.items() if key.endswith("verified"))
    assert "point_enu" not in json.dumps(report)


def test_bounded_delaunay_does_not_bridge_disconnected_coplanar_roofs():
    cloud = np.concatenate((_patch(-4., -2.), _patch(2., 4.)))
    surfaces = _fit(cloud, max_support_gap_m=.8)
    assert len(surfaces.patches) == 1
    grid = sample_rgb_surface_xy(surfaces, metres_per_pixel=.1, max_side=128)
    assert np.isfinite(grid.height_enu_private[_pixel(grid, -3., 0.)])
    assert np.isnan(grid.height_enu_private[_pixel(grid, 0., 0.)])


def test_xy_render_uses_original_pixels_and_rejects_hidden_lower_surface():
    cloud = np.concatenate((_patch(-2., 4.),
                            _patch(-.7, .7, offset=2.)))
    surfaces = _fit(
        cloud, distance_threshold_m=.025, max_support_gap_m=1.1,
    )
    image = np.full((160, 160, 3), (30, 100, 190), np.uint8)
    before = image.copy()
    result = render_rgb_reconstructed_surface_candidate(
        surfaces, (_pose((-4., 0., 4.), (0., 0., 0.)),),
        PinholeIntrinsics(50., 50., 80., 80.), lambda _: image,
        image_size=(160, 160), metres_per_pixel=.04, max_side=256,
        tile_side_px=64,
    )
    grid = result.grid_private
    foreground = _pixel(grid, -1.4, 0.)
    hidden = _pixel(grid, 3.5, 0.)
    upper = _pixel(grid, 0., 0.)
    assert result.valid_mask[foreground]
    assert result.valid_mask[upper]
    assert not result.valid_mask[hidden]
    assert np.array_equal(result.bgr[foreground], (30, 100, 190))
    assert np.array_equal(result.bgr[hidden], (0, 0, 0))
    assert result.occluded_observations > 0
    assert np.array_equal(image, before)
    assert not result.bgr.flags.writeable
    assert not result.valid_mask.flags.writeable
    summary = result.privacy_safe_summary()
    assert summary["assumes_zero_distortion"] is True
    assert summary["decoded_source_count"] == 1
    assert summary["full_flight_roof_coverage_verified"] is False
    assert all(v is False for k, v in summary.items() if k.endswith("verified"))


def test_geometry_selects_better_view_without_averaging_source_colors():
    surfaces = _fit(_patch(-1., 1.), max_support_gap_m=.6)
    red = np.full((120, 120, 3), (0, 0, 255), np.uint8)
    green = np.full((120, 120, 3), (0, 255, 0), np.uint8)
    poses = (_pose((-2., 0., 5.), (0., 0., 0.)),
             _pose((0., 0., 4.), (0., 0., 0.)))
    calls = []
    def provider(i):
        calls.append(i)
        return (red, green)[i]
    result = render_rgb_reconstructed_surface_candidate(
        surfaces, poses, PinholeIntrinsics(60., 60., 60., 60.), provider,
        image_size=(120, 120), metres_per_pixel=.04, max_side=128,
        tile_side_px=32,
    )
    assert calls == [0, 1]
    at = _pixel(result.grid_private, 0., 0.)
    assert np.array_equal(result.bgr[at], (0, 255, 0))
    assert set(np.unique(result.bgr[result.valid_mask], axis=0).sum(axis=1)) == {255}


def test_tilted_planes_project_gradient_pixels_using_each_reconstructed_height():
    cloud = np.concatenate((_patch(-4., -2., tilt_x=.3),
                            _patch(2., 4., offset=2., tilt_y=-.25)))
    surfaces = _fit(
        cloud, distance_threshold_m=.025, max_support_gap_m=.8,
    )
    yy, xx = np.indices((160, 160))
    image = np.stack((xx, yy, np.full_like(xx, 20)), axis=2).astype(np.uint8)
    result = render_rgb_reconstructed_surface_candidate(
        surfaces, (_pose((0., 0., 8.), (0., 0., 0.)),),
        PinholeIntrinsics(50., 50., 80., 80.), lambda _: image,
        image_size=(160, 160), metres_per_pixel=.05, max_side=256,
        tile_side_px=32, distortion_coefficients=np.zeros(5),
    )
    grid = result.grid_private
    rows, cols = np.nonzero(result.valid_mask)
    x = grid.origin_xy_private[0] + (cols+.5)*grid.metres_per_pixel
    y = grid.origin_xy_private[1] - (rows+.5)*grid.metres_per_pixel
    # Independent analytic nadir-camera projection; using a flat surface for
    # these two roof heights would move the ramp by several source pixels.
    z = np.where(x < 0, .3*x, 2.-.25*y)
    expected = np.column_stack((50*x/(8-z)+80, -50*y/(8-z)+80,
                                np.full(len(x), 20.)))
    assert len(x) > 1000
    assert np.max(np.abs(result.bgr[rows, cols].astype(float)-expected)) < 1.1
    assert not result.valid_mask[_pixel(grid, 0., 0.)]
    assert result.privacy_safe_summary()["assumes_zero_distortion"] is True


def test_distorted_visible_view_is_not_discarded_by_undistorted_tile_bounds():
    surfaces = _fit(
        _patch(0., 1., -.5, .5), max_support_gap_m=.6,
    )
    image = np.full((100, 100, 3), (15, 75, 120), np.uint8)
    result = render_rgb_reconstructed_surface_candidate(
        surfaces, (_pose((-8., 0., 4.), (-8., 0., 0.)),),
        PinholeIntrinsics(30., 30., 50., 50.), lambda _: image,
        image_size=(100, 100), distortion_coefficients=np.array([-.1, 0., 0., 0., 0.]),
        metres_per_pixel=.025, max_side=128, tile_side_px=32,
    )
    assert result.valid_mask.all()
    assert np.all(result.bgr[result.valid_mask] == (15, 75, 120))
    assert result.privacy_safe_summary()["assumes_zero_distortion"] is False


def test_nonfinite_collinear_missing_multiview_and_cancellation_fail():
    cloud = _patch(-1., 1.)
    with pytest.raises(ValueError, match="finite"):
        fit_rgb_surface_candidates(np.full((20, 3), np.nan))
    with pytest.raises(ValueError, match="nondegenerate"):
        _fit(np.column_stack((np.arange(40), np.zeros((40, 2)))))
    with pytest.raises(ValueError, match="nondegenerate"):
        fit_rgb_surface_candidates(cloud, np.zeros(len(cloud), dtype=int),
                                    np.arange(len(cloud)))
    with pytest.raises(RuntimeError, match="cancelled"):
        fit_rgb_surface_candidates(cloud, should_cancel=lambda: True)
    with pytest.raises(ValueError, match="real matched multiview"):
        fit_rgb_surface_candidates(cloud)
    surfaces = _fit(cloud, max_support_gap_m=.6)
    with pytest.raises(RuntimeError, match="cancelled"):
        sample_rgb_surface_xy(surfaces, should_cancel=lambda: True)
    with pytest.raises(ValueError, match="rotations"):
        render_rgb_reconstructed_surface_candidate(
            surfaces, (np.zeros((3, 4)),), PinholeIntrinsics(60, 60, 60, 60),
            lambda _: None, image_size=(120, 120))
