"""Vertical private multi-pitched-facet RGB composition, synthetic only."""
import json
from dataclasses import replace

import numpy as np
import pytest

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_private_pitched_plane_candidates import (
    PrivateRGBPitchedPlaneCandidate, PrivateRGBPitchedPlaneCandidates,
)
from mcm_solarcheck.services.rgb_private_pitched_plane_orthorectification import (
    PrivateRGBPitchedPlaneOrthopatch,
)
from mcm_solarcheck.services.rgb_private_multi_plane_preview import (
    choose_private_pitched_plane_cameras,
    compose_private_pitched_rgb_orthopatches,
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


def _facet(x0, x1, *, slope=0.2, rgb=(0, 0, 0)):
    n = np.array((-slope, 0., 1.), dtype=np.float64)
    n /= np.linalg.norm(n)
    hull = np.array(((x0,-1.),(x1,-1.),(x1,1.),(x0,1.)))
    return PrivateRGBPitchedPlaneCandidate(
        n, -1./np.sqrt(1+slope*slope), np.arange(32,dtype=np.int32),
        hull, float(np.degrees(np.arctan(abs(slope)))),
        4, .01, (x1-x0)*2.,
    )


def _patch(plane, value, meters_per_pixel=.1):
    hull = plane.footprint_xy_private_m
    x0 = hull[:,0].min()
    north = hull[:,1].max()
    w = int(round(np.ptp(hull[:,0])/meters_per_pixel))
    h = int(round(np.ptp(hull[:,1])/meters_per_pixel))
    rgb = np.empty((h,w,3),dtype=np.uint8)
    rgb[:,:,:] = value
    valid = np.ones((h,w),dtype=bool)
    return PrivateRGBPitchedPlaneOrthopatch(
        rgb, valid, (float(x0),float(north)), meters_per_pixel,
        3, h*w, h*w, h*w,
    )


def _poses():
    r = np.diag((1.,-1.,-1.))
    centers = ((-.6,0.,5.), (.0,0.,5.), (.6,0.,5.))
    return tuple(np.column_stack((r,-r @ np.array(c))) for c in centers)


def _metric(planes):
    poses = _poses()
    return PrivateRGBRTKWindowedBA(
        ("synthetic-0","synthetic-1","synthetic-2"),
        poses, np.zeros((32,3)), 1., .5, .1,.1,.2,
        2,1,3,3,20.,10.,
        camera_reprojection_audit=RGBPerCameraReprojectionAudit(
            3,32,96,32,3,1.,1.,1.,
        ),
        multiview_parallax_audit=RGBMultiviewParallaxAudit(
            3,32,96,.25,.75,32,0,2.,3.,
        ),
        pitched_plane_candidates=PrivateRGBPitchedPlaneCandidates(
            tuple(planes),32,
        ),
    )


def test_geometric_camera_selection_uses_real_3d_poses_and_full_image_bounds():
    k = PinholeIntrinsics(60.,60.,64.,64.)
    selected = choose_private_pitched_plane_cameras(
        _facet(-2.,0.), _poses(), k, 128,128,
    )
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert all(0 <= i < 3 for i in selected)
    bad = tuple(np.column_stack(
        (np.diag((1.,-1.,-1.)), np.array((0.,0.,-5.)))
    ) for _ in range(3))
    with pytest.raises(ValueError, match="fewer than two"):
        choose_private_pitched_plane_cameras(
            _facet(-2.,0.), bad, k, 128,128,
        )


def test_two_distinct_pitched_faces_use_one_north_up_enu_rgb_grid():
    left, right = _facet(-2.,0.), _facet(0.,2.,slope=-.25)
    fused = compose_private_pitched_rgb_orthopatches(
        (left,right),
        (_patch(left,(220,40,20)),_patch(right,(20,40,220))),
    )
    assert fused.rgb_private.shape == (20,40,3)
    assert fused.valid_pixel_count == 800
    assert fused.ambiguous_overlap_pixels == 0
    np.testing.assert_array_equal(fused.rgb_private[10,10],(220,40,20))
    np.testing.assert_array_equal(fused.rgb_private[10,30],(20,40,220))
    assert set(np.unique(fused.selected_plane_index_private)) == {0,1}
    assert fused.top_left_xy_enu_private_m == (-2.,1.)
    assert not fused.rgb_private.flags.writeable
    assert not fused.valid_mask_private.flags.writeable
    assert not fused.selected_plane_index_private.flags.writeable
    summary = fused.privacy_safe_summary()
    assert summary["plane_candidate_count"] == 2
    assert summary["orthophoto_verified"] is False
    assert summary["roof_surface_identity_verified"] is False
    for secret in ("rgb_private", "top_left_xy", "selected_plane_index",
                   "footprint", "ENU", "DJI_", "GPS"):
        assert secret not in json.dumps(summary)


def test_same_height_overlapping_3d_facets_stay_ambiguous_not_mixed():
    first, second = _facet(-1.,1.), _facet(-1.,1.)
    fused = compose_private_pitched_rgb_orthopatches(
        (first,second),
        (_patch(first,(255,0,0)),_patch(second,(0,0,255))),
    )
    assert fused.ambiguous_overlap_pixels == 400
    assert fused.valid_pixel_count == 0
    assert np.all(fused.selected_plane_index_private == -1)
    assert not np.any(fused.rgb_private)
    assert not fused.privacy_safe_summary()["orthophoto_verified"]


def test_bad_alignment_or_overbudget_grid_fails_without_writing():
    a,b = _facet(-2.,0.), _facet(0.,2.)
    p,q = _patch(a,(100,0,0)),_patch(b,(0,100,0))
    with pytest.raises(ValueError, match="share a common"):
        compose_private_pitched_rgb_orthopatches(
            (a,b),(p,replace(q,meters_per_pixel=.2)),
        )
    with pytest.raises(ValueError,match="misaligned"):
        compose_private_pitched_rgb_orthopatches(
            (a,b),(p,replace(q,top_left_xy_enu_private_m=(22.,1.))),
        )
    far = _facet(0.,10.)
    large = _patch(far,(0,100,0))
    with pytest.raises(ValueError,match="resource budget"):
        compose_private_pitched_rgb_orthopatches(
            (a,far),(p,large),max_side=64,
        )


def test_metric_ba_auto_selects_and_renders_two_facets_from_undistorted_cameras():
    k = PinholeIntrinsics(60.,60.,64.,64.)
    left,right = _facet(-1.,0.),_facet(0.,1.,slope=-.2)
    metric = _metric((left,right))
    frames = {
        0:np.full((128,128,3),(120,80,50),dtype=np.uint8),
        1:np.full((128,128,3),(120,80,50),dtype=np.uint8),
        2:np.full((128,128,3),(120,80,50),dtype=np.uint8),
    }
    original = {i:img.copy() for i,img in frames.items()}
    fused = metric.render_private_multi_plane_preview(
        frames,k,meters_per_pixel=.05,
    )
    assert fused.plane_count == 2
    assert fused.valid_pixel_count > 600
    assert np.all(fused.rgb_private[fused.valid_mask_private] == [120,80,50])
    assert fused.privacy_safe_summary()["orthophoto_verified"] is False
    assert all(np.array_equal(frames[i], original[i]) for i in frames)
    with pytest.raises(ValueError,match="complete private"):
        replace(metric,camera_reprojection_audit=None
                ).render_private_multi_plane_preview(frames,k)
