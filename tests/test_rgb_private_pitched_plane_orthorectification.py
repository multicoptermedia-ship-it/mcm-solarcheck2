"""Synthetic pitched RGB orthorectification on actual private 3D plane points.

No original DJI photos, real lens calibration, survey control or roof
certification. Both cameras image the same analytically generated slope.
"""
from dataclasses import replace
import json

import numpy as np
import pytest

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_private_pitched_plane_candidates import (
    PrivateRGBPitchedPlaneCandidate, PrivateRGBPitchedPlaneCandidates,
)
from mcm_solarcheck.services.rgb_private_pitched_plane_orthorectification import (
    render_private_pitched_rgb_orthopatch,
)
from mcm_solarcheck.services.rgb_rtk_joint_bundle_adjustment import (
    PrivateRGBRTKWindowedBA,
)
from mcm_solarcheck.services.rgb_per_camera_reprojection_gate import (
    RGBPerCameraReprojectionAudit,
)
from mcm_solarcheck.services.rgb_multiview_parallax_gate import (
    RGBMultiviewParallaxAudit,
)


def _plane():
    n = np.array((-.2, 0., 1.), dtype=np.float64)
    n /= np.linalg.norm(n)
    # z(x,y) = 1 + .2x, i.e. -0.2x + z -1 == 0.
    return PrivateRGBPitchedPlaneCandidate(
        n, -1./np.sqrt(1.04),
        np.arange(32, dtype=np.int32),
        np.array([[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]]),
        float(np.degrees(np.arctan(.2))), 4, .01, 16.,
    )


def _source(camera_x, k):
    # Proper R flips north/up: looking DOWN from ENU altitude 5 m.
    r = np.diag((1.,-1.,-1.))
    center = np.array((camera_x,0.,5.))
    pose = np.column_stack((r, -r @ center))
    yy, xx = np.indices((320,320), dtype=np.float64)
    rx = (xx-k.cx)/k.fx
    ry = -(yy-k.cy)/k.fy
    # Intersect each camera ray with z = 1 + .2*x.
    t = (4. - .2*camera_x) / (1. + .2*rx)
    world_x = camera_x + rx*t
    world_y = ry*t
    rgb = np.stack((
        np.clip(128 + 20*world_x, 0, 255),
        np.clip(128 + 20*world_y, 0, 255),
        np.full_like(world_x, 80),
    ), axis=-1).round().astype(np.uint8)
    return rgb, pose


def test_two_real_camera_rays_inverse_project_to_same_pitched_world_rgb():
    k = PinholeIntrinsics(120.,120.,160.,160.)
    sources = [_source(x,k) for x in (0.,.4)]
    original = [image.copy() for image,_ in sources]
    result = render_private_pitched_rgb_orthopatch(
        _plane(), tuple(i for i,_ in sources),
        tuple(p for _,p in sources), k, meters_per_pixel=.05,
    )
    assert result.rgb_private.shape == (80,80,3)
    assert result.planar_pixel_count >= 5900
    assert result.valid_pixel_count == result.planar_pixel_count
    assert result.multi_camera_pixel_count == result.planar_pixel_count
    rr, cc = np.indices(result.valid_mask_private.shape)
    east = -2. + (cc + .5)*.05
    north = 2. - (rr + .5)*.05
    expected = np.stack((
        128 + 20*east,
        128 + 20*north,
        np.full_like(east,80.),
    ), axis=-1)
    error = np.abs(
        result.rgb_private.astype(np.float64) - expected
    )
    assert float(np.max(error[result.valid_mask_private])) <= 2.5
    assert not result.rgb_private.flags.writeable
    assert not result.valid_mask_private.flags.writeable
    assert all(np.array_equal(original[i], sources[i][0]) for i in range(2))
    safe = result.privacy_safe_summary()
    assert safe["multi_camera_coverage_fraction"] == 1.
    assert safe["roof_geometry_verified"] is False
    assert safe["orthophoto_verified"] is False
    assert safe["independent_survey_verified"] is False
    for forbidden in ("rgb_private","top_left_xy", "normal_enu",
                      "offset_enu", "landmark_indices", "DJI_", "GPS"):
        assert forbidden not in json.dumps(safe)


def test_below_plane_or_out_of_view_cannot_fabricate_pixel_coverage():
    k = PinholeIntrinsics(120.,120.,160.,160.)
    source, pose = _source(0., k)
    # The camera looks downward from altitude 0 and the plane is ABOVE it.
    behind = pose.copy()
    behind[:,3] = np.array((0.,0.,0.))
    result = render_private_pitched_rgb_orthopatch(
        _plane(), (source,), (behind,), k,
    )
    assert result.valid_pixel_count == 0
    assert result.multi_camera_pixel_count == 0
    assert not np.any(result.rgb_private)
    assert not np.any(result.valid_mask_private)
    assert result.privacy_safe_summary()["orthophoto_verified"] is False


def test_resource_and_pose_validation_are_fail_closed():
    k = PinholeIntrinsics(120.,120.,160.,160.)
    source, pose = _source(0., k)
    with pytest.raises(ValueError, match="resolution exceeds"):
        render_private_pitched_rgb_orthopatch(
            _plane(), (source,), (pose,), k, meters_per_pixel=.01,
            max_side=64,
        )
    bad = pose.copy()
    bad[0,0] = 1.4
    with pytest.raises(ValueError, match="proper final fitted"):
        render_private_pitched_rgb_orthopatch(
            _plane(), (source,), (bad,), k,
        )
    with pytest.raises(ValueError, match="undistorted RGB"):
        render_private_pitched_rgb_orthopatch(
            _plane(), (source[:,:,0],), (pose,), k,
        )
    with pytest.raises(RuntimeError, match="cancelled"):
        render_private_pitched_rgb_orthopatch(
            _plane(), (source,), (pose,), k,
            should_cancel=lambda: True,
        )


def test_existing_rtk_ba_candidate_can_render_in_memory_with_indexed_cameras():
    k = PinholeIntrinsics(120.,120.,160.,160.)
    sources = [_source(x,k) for x in (0., .4)]
    good_pixel = RGBPerCameraReprojectionAudit(
        12,32,384,32,12,1.,1.,1.,
    )
    good_parallax = RGBMultiviewParallaxAudit(
        12,32,384,.25,.75,32,0,2.,3.,
    )
    poses = tuple(
        sources[i][1] if i < 2 else sources[1][1]
        for i in range(12)
    )
    candidate = PrivateRGBRTKWindowedBA(
        tuple(f"synthetic-{i}" for i in range(12)),
        poses, np.zeros((32,3)), 1.,.5,.1,.1,.15,
        10,2,12,3,20.,10.,
        camera_reprojection_audit=good_pixel,
        multiview_parallax_audit=good_parallax,
        pitched_plane_candidates=PrivateRGBPitchedPlaneCandidates(
            (_plane(),),32,
        ),
    )
    patch = candidate.render_private_pitched_plane_preview(
        0, tuple(im for im,_ in sources), (0,1), k,
    )
    assert patch.valid_pixel_count == patch.planar_pixel_count
    assert patch.privacy_safe_summary()["orthophoto_verified"] is False
    with pytest.raises(ValueError, match="valid fitted"):
        candidate.render_private_pitched_plane_preview(
            1, tuple(im for im,_ in sources), (0,1), k,
        )
    with pytest.raises(ValueError, match="valid fitted"):
        replace(candidate, camera_reprojection_audit=None
                ).render_private_pitched_plane_preview(
            0, tuple(im for im,_ in sources), (0,1), k,
        )
