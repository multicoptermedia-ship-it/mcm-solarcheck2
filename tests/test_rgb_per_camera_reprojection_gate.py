"""Post-BA private RGB camera-wise reprojection gate, entirely synthetic."""
import json

import numpy as np
import pytest

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_per_camera_reprojection_gate import (
    RGBPerCameraReprojectionQualityError, audit_rgb_per_camera_reprojection,
)


def _flight(n=12, points=25):
    rng = np.random.default_rng(29102026)
    k = PinholeIntrinsics(730., 735., 400., 300.)
    landmarks = np.column_stack((
        rng.uniform(-2., 3., points),
        rng.uniform(-2., 2., points),
        rng.uniform(10., 16., points),
    ))
    pose = tuple(
        np.column_stack((np.eye(3), np.array((-.2*i, .01*i, 0.))))
        for i in range(n)
    )
    fid = np.repeat(np.arange(n, dtype=np.int32), points)
    lid = np.tile(np.arange(points, dtype=np.int32), n)
    camera_xyz = landmarks[lid] + np.asarray(pose)[fid, :, 3]
    pixels = np.column_stack((
        k.fx * camera_xyz[:, 0] / camera_xyz[:, 2] + k.cx,
        k.fy * camera_xyz[:, 1] / camera_xyz[:, 2] + k.cy,
    ))
    pixels += rng.normal(0., .025, pixels.shape)
    return pose, landmarks, fid, lid, pixels, k


def test_all_cameras_and_landmarks_checked_after_bundle_adjustment():
    pose, pts, fid, lid, xy, k = _flight()
    audit = audit_rgb_per_camera_reprojection(pose, pts, fid, lid, xy, k)
    summary = audit.privacy_safe_summary()
    assert audit.camera_count == 12
    assert audit.observation_count == 300
    assert audit.minimum_observations_per_camera == 25
    assert audit.minimum_cameras_per_landmark == 12
    assert audit.worst_camera_p95_px < .2
    assert summary["camera_reprojection_audit_completed"] is True
    assert summary["independent_survey_verified"] is False
    assert summary["orthophoto_verified"] is False
    for secret in ("V-0001", "DJI", "lat", "longitude", "GPS", "private/"):
        assert secret not in json.dumps(summary)


def test_single_bad_camera_cannot_hide_behind_low_whole_flight_reprojection():
    pose, pts, fid, lid, xy, k = _flight(n=160)
    corrupted = xy.copy()
    corrupted[fid == 83, 0] += 25.
    # Only 1/160 cameras is bad. Global p95 remains low but this must fail.
    with pytest.raises(ValueError, match="individual RGB camera"):
        audit_rgb_per_camera_reprojection(
            pose, pts, fid, lid, corrupted, k,
        )
    healthy = audit_rgb_per_camera_reprojection(
        pose, pts, fid, lid, xy, k,
    )
    assert healthy.camera_count == 160


def test_behind_camera_even_with_valid_pixel_array_fails_closed():
    pose, pts, fid, lid, xy, k = _flight()
    behind = pts.copy()
    behind[7, 2] = -1.
    with pytest.raises(ValueError, match="behind"):
        audit_rgb_per_camera_reprojection(pose, behind, fid, lid, xy, k)


def test_sparse_camera_or_single_view_landmark_fails_closed():
    pose, pts, fid, lid, xy, k = _flight()
    # Only seven observations remain for the last camera.
    keep = (fid != 11) | (lid < 7)
    with pytest.raises(ValueError, match="insufficient final"):
        audit_rgb_per_camera_reprojection(
            pose, pts, fid[keep], lid[keep], xy[keep], k,
        )
    # One last landmark survives only in the first camera.
    keep = (lid != 24) | (fid == 0)
    with pytest.raises(ValueError, match="distinct multi-view"):
        audit_rgb_per_camera_reprojection(
            pose, pts, fid[keep], lid[keep], xy[keep], k,
        )


def test_invalid_index_and_cancel_are_private_fail_closed():
    pose, pts, fid, lid, xy, k = _flight()
    invalid = fid.copy()
    invalid[2] = len(pose)
    with pytest.raises(ValueError, match="invalid finite"):
        audit_rgb_per_camera_reprojection(pose, pts, invalid, lid, xy, k)
    with pytest.raises(RuntimeError, match="cancelled"):
        audit_rgb_per_camera_reprojection(
            pose, pts, fid, lid, xy, k, should_cancel=lambda: True,
        )


def test_three_d_point_is_not_multiview_when_same_camera_repeated():
    pose, pts, fid, lid, xy, k = _flight()
    # Repeat a landmark in one camera many times; never treat that as
    # two-camera triangulation evidence.
    keep = (lid != 2) | (fid == 0)
    repeated = 25
    repeat_fid = np.repeat(np.int32(0), repeated)
    repeat_lid = np.repeat(np.int32(2), repeated)
    repeat_xy = np.repeat(xy[(fid == 0) & (lid == 2)], repeated, axis=0)
    with pytest.raises(ValueError, match="distinct multi-view"):
        audit_rgb_per_camera_reprojection(
            pose, pts,
            np.r_[fid[keep], repeat_fid],
            np.r_[lid[keep], repeat_lid],
            np.vstack((xy[keep], repeat_xy)),
            k,
        )


def test_strict_camera_failure_contains_only_aggregate_quality_diagnostics():
    pose, pts, fid, lid, xy, k = _flight(n=160)
    corrupted = xy.copy()
    corrupted[fid == 83, 0] += 25.
    with pytest.raises(RGBPerCameraReprojectionQualityError) as error:
        audit_rgb_per_camera_reprojection(pose, pts, fid, lid, corrupted, k)
    audit = error.value.audit
    summary = audit.privacy_safe_summary()
    assert audit.camera_count == 160
    assert summary["cameras_above_p95_limit"] == 1
    assert summary["camera_p95_limit_px"] == 6.
    assert summary["worst_camera_reprojection_p95_px"] > 20.
    assert summary["camera_reprojection_quality_passed"] is False
    assert summary["independent_survey_verified"] is False
    assert summary["roof_geometry_verified"] is False
    assert summary["orthophoto_verified"] is False
    assert "83" not in str(error.value)
    for private in ("V-0084", "frame_ids", "latitude", "longitude", "GPS",
                    "positions", "camera_to_enu", "private/"):
        assert private not in json.dumps(summary)
    # Explicit diagnostic mode has exactly the same failed quantitative audit,
    # but is never acceptable to the real 160-JPEG CLI.
    diagnostic = audit_rgb_per_camera_reprojection(
        pose, pts, fid, lid, corrupted, k, enforce_camera_p95=False,
    )
    assert diagnostic.privacy_safe_summary() == summary
