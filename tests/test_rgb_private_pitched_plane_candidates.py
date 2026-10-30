"""Vertical RGB photogrammetry slice: fitted private pitched 3D planes.

All points are artificial local ENU metric scenes, not DJI calibration, roof
survey, orthophotos, customer files or measured reconstruction evidence.
"""
import json

import numpy as np
import pytest

from mcm_solarcheck.services.rgb_private_pitched_plane_candidates import (
    propose_private_rgb_pitched_planes,
)


def _two_pitched_facets():
    rng = np.random.default_rng(20261029)
    left_x = rng.uniform(-8., -1., 115)
    right_x = rng.uniform(1., 8., 115)
    left_y = rng.uniform(-5., 5., 115)
    right_y = rng.uniform(-5., 5., 115)
    left = np.column_stack((
        left_x, left_y,
        9. + .22 * left_x + rng.normal(0., .012, 115),
    ))
    right = np.column_stack((
        right_x, right_y,
        9. - .25 * right_x + rng.normal(0., .012, 115),
    ))
    noise = np.column_stack((
        rng.uniform(-12., 12., 28),
        rng.uniform(-7., 7., 28),
        rng.uniform(0., 15., 28),
    ))
    xyz = np.vstack((left, right, noise))
    lid = np.repeat(np.arange(len(xyz), dtype=np.int32), 4)
    fid = np.tile(np.array((0, 3, 6, 9), dtype=np.int32), len(xyz))
    return xyz, fid, lid


def test_two_distinct_real_3d_pitched_hypotheses_from_common_enu_cloud():
    xyz, fid, lid = _two_pitched_facets()
    answer = propose_private_rgb_pitched_planes(xyz, fid, lid, max_planes=2)
    assert len(answer.planes_private) == 2
    assert all(p.slope_deg > 5. for p in answer.planes_private)
    assert all(p.residual_p95_m < .10 for p in answer.planes_private)
    assert all(p.footprint_area_m2 > 4. for p in answer.planes_private)
    assert all(p.supporting_cameras == 4 for p in answer.planes_private)
    assert all(len(p.landmark_indices_private) >= 85
               for p in answer.planes_private)
    assert (len(set(answer.planes_private[0].landmark_indices_private)
                & set(answer.planes_private[1].landmark_indices_private)) == 0)
    # These hypotheses are actual *private* normal/offset/footprint arrays,
    # not just counters. The public summary MUST NOT disclose any of them.
    for plane in answer.planes_private:
        assert plane.normal_enu_private.shape == (3,)
        assert plane.footprint_xy_private_m.shape[1] == 2
        assert not plane.normal_enu_private.flags.writeable
        assert not plane.landmark_indices_private.flags.writeable
        assert not plane.footprint_xy_private_m.flags.writeable
    summary = answer.privacy_safe_summary()
    assert summary["plane_candidate_count"] == 2
    assert summary["roof_surface_identity_verified"] is False
    assert summary["roof_geometry_verified"] is False
    assert summary["independent_survey_verified"] is False
    assert summary["orthophoto_verified"] is False
    text = json.dumps(summary)
    for private in ("normal_enu_private", "footprint_xy_private",
                    "signed_offset_enu_private_m", "landmark_indices",
                    "x_enu", "y_enu", "camera_ids", "V-0001", "GPS"):
        assert private not in text


def test_flat_ground_is_not_misreported_as_pitched_roof_candidate():
    rng = np.random.default_rng(44)
    xyz = np.column_stack((
        rng.uniform(-7., 7., 140),
        rng.uniform(-8., 8., 140),
        np.full(140, 3.),
    ))
    lid = np.repeat(np.arange(len(xyz), dtype=np.int32), 3)
    fid = np.tile(np.array((0, 1, 2), dtype=np.int32), len(xyz))
    answer = propose_private_rgb_pitched_planes(xyz, fid, lid)
    assert answer.planes_private == ()
    assert answer.privacy_safe_summary()["plane_candidate_count"] == 0
    assert answer.privacy_safe_summary()["roof_geometry_verified"] is False

    # An explicit general-surface request may keep the same horizontal points
    # as a geometric hypothesis; it does not label ground as a real roof.
    general = propose_private_rgb_pitched_planes(xyz, fid, lid, min_slope_deg=0.)
    assert len(general.planes_private) == 1
    plane = general.planes_private[0]
    assert plane.slope_deg == pytest.approx(0.)
    assert plane.supporting_cameras == 3
    np.testing.assert_allclose(plane.normal_enu_private, (0., 0., 1.))
    assert plane.signed_offset_enu_private_m == pytest.approx(-3.)
    assert general.privacy_safe_summary()["roof_geometry_verified"] is False
    assert general.privacy_safe_summary()["roof_surface_identity_verified"] is False


def test_candidate_needs_real_multicamera_observations():
    xyz, _, lid = _two_pitched_facets()
    fid = np.zeros(len(lid), dtype=np.int32)
    result = propose_private_rgb_pitched_planes(xyz, fid, lid, max_planes=2)
    assert result.planes_private == ()


def test_unsupported_or_nonfinite_geometry_and_cancellation_fail():
    xyz, fid, lid = _two_pitched_facets()
    wrong = xyz.copy()
    wrong[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite bounded"):
        propose_private_rgb_pitched_planes(wrong, fid, lid)
    wrong_ids = lid.copy()
    wrong_ids[0] = len(xyz)
    with pytest.raises(ValueError, match="finite bounded"):
        propose_private_rgb_pitched_planes(xyz, fid, wrong_ids)
    with pytest.raises(ValueError, match="bounded private"):
        propose_private_rgb_pitched_planes(xyz, fid, lid, min_points=2)
    with pytest.raises(RuntimeError, match="cancelled"):
        propose_private_rgb_pitched_planes(
            xyz, fid, lid, should_cancel=lambda: True,
        )


def test_deterministic_private_hypothesis_geometry_on_same_input():
    xyz, fid, lid = _two_pitched_facets()
    a = propose_private_rgb_pitched_planes(xyz, fid, lid, max_planes=2)
    b = propose_private_rgb_pitched_planes(xyz, fid, lid, max_planes=2)
    assert a.privacy_safe_summary() == b.privacy_safe_summary()
    for x, y in zip(a.planes_private, b.planes_private):
        np.testing.assert_allclose(x.normal_enu_private, y.normal_enu_private)
        np.testing.assert_allclose(
            x.signed_offset_enu_private_m, y.signed_offset_enu_private_m
        )
