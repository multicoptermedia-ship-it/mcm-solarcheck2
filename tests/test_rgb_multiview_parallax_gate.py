"""Synthetic private RGB viewing-ray parallax gates (no source photos/GPS)."""
import json

import numpy as np
import pytest

from mcm_solarcheck.services.rgb_multiview_parallax_gate import (
    RGBMultiviewParallaxQualityError,
    audit_rgb_multiview_parallax,
)


def _scene(n=12, baseline=.70, count=32):
    rng = np.random.default_rng(29102026)
    # World->camera: R=I and translation=-camera_center.
    poses = tuple(
        np.column_stack((
            np.eye(3),
            np.array((-baseline * i, -.05*(i % 2), 0.), dtype=float),
        ))
        for i in range(n)
    )
    # Tracks follow their camera neighborhood along the full synthetic
    # 160-image strip. Fixed world-origin points would be extremely far
    # off-axis near the far end and wrongly simulate near-zero parallax.
    track_start = np.arange(count) % (n - 2)
    xyz = np.column_stack((
        baseline * track_start + rng.uniform(-1., 1., count),
        rng.uniform(-1., 1., count),
        rng.uniform(8., 14., count),
    ))
    # Each 3D point is triangulated from 3 genuine, distinct cameras.
    fid = np.concatenate([
        np.array((i, i+1, i+2), dtype=np.int32)
        for i in track_start
    ])
    lid = np.repeat(np.arange(count, dtype=np.int32), 3)
    return poses, xyz, fid, lid


def test_strong_multiview_parallax_is_not_survey_or_roof_evidence():
    poses, xyz, fid, lid = _scene()
    result = audit_rgb_multiview_parallax(poses, xyz, fid, lid)
    summary = result.privacy_safe_summary()
    assert result.supported_landmark_count == 32
    assert result.weak_landmark_count == 0
    assert summary["parallax_quality_passed"] is True
    assert summary["cameras_below_supported_observation_fraction"] == 0
    assert summary["worst_camera_supported_observation_fraction"] == 1.
    assert summary["landmark_parallax_p10_deg"] > .25
    assert summary["independent_survey_verified"] is False
    assert summary["roof_geometry_verified"] is False
    assert summary["orthophoto_verified"] is False


def test_same_pixel_projection_with_negligible_baseline_cannot_prove_depth():
    # With zero/noisy stereo baseline even perfect 0-px reprojection is
    # compatible with widely different 3D depths; reject weak rays.
    poses, xyz, fid, lid = _scene(baseline=.00001)
    poses = tuple(np.column_stack((np.eye(3), np.array((-1e-5*i, 0., 0.))))
                  for i in range(12))
    with pytest.raises(RGBMultiviewParallaxQualityError) as error:
        audit_rgb_multiview_parallax(poses, xyz, fid, lid)
    diag = error.value.audit.privacy_safe_summary()
    assert diag["supported_landmark_count"] == 0
    assert diag["weak_landmark_count"] == len(xyz)
    assert diag["parallax_quality_passed"] is False
    assert diag["landmark_parallax_median_deg"] < .05
    assert "frame" not in json.dumps(diag)
    assert "DJI" not in json.dumps(diag)
    assert "latitude" not in json.dumps(diag)
    assert "private/" not in str(error.value)
    assert audit_rgb_multiview_parallax(
        poses, xyz, fid, lid, enforce_parallax=False,
    ).privacy_safe_summary() == diag


def test_single_weak_stereo_patch_below_75_percent_gate_fails():
    poses, xyz, fid, lid = _scene()
    # 32 landmarks, all true 3D, but 9 are observed from 3 nearly identical
    # baseline centers rather than the spatially diverse full flight.
    all_poses = poses + tuple(
        np.column_stack((
            np.eye(3), np.array((-1e-5*(i+1), 0., 0.))
        ))
        for i in range(3)
    )
    altered_fid = fid.copy()
    for landmark in range(9):
        altered_fid[lid == landmark] = np.array((12, 13, 14))
    with pytest.raises(RGBMultiviewParallaxQualityError) as error:
        audit_rgb_multiview_parallax(
            all_poses, xyz, altered_fid, lid,
        )
    summary = error.value.audit.privacy_safe_summary()
    assert summary["weak_landmark_count"] >= 9
    assert summary["supported_landmark_fraction"] < .75


def test_multiview_parallax_of_160_images_is_bounded_and_deterministic():
    poses, xyz, fid, lid = _scene(n=160, count=320)
    result = audit_rgb_multiview_parallax(poses, xyz, fid, lid)
    assert result.camera_count == 160
    assert result.landmark_count == 320
    assert result.supported_landmark_count >= 240
    shuffled = np.random.default_rng(4).permutation(len(fid))
    shuffled_result = audit_rgb_multiview_parallax(
        poses, xyz, fid[shuffled], lid[shuffled],
    )
    # Observations may arrive in any order: selecting the lowest-index
    # observing camera makes all angles and the quality decision stable.
    np.testing.assert_allclose(
        shuffled_result.observed_landmark_p10_parallax_deg,
        result.observed_landmark_p10_parallax_deg, atol=1e-8,
    )
    np.testing.assert_allclose(
        shuffled_result.observed_landmark_median_parallax_deg,
        result.observed_landmark_median_parallax_deg, atol=1e-8,
    )
    assert shuffled_result.supported_landmark_count == result.supported_landmark_count
    assert shuffled_result.privacy_safe_summary()["parallax_quality_passed"] is True


def test_multiview_requires_distinct_cameras_and_finite_geometry():
    poses, xyz, fid, lid = _scene()
    altered = fid.copy()
    altered[lid == 0] = 0
    with pytest.raises(ValueError, match="distinct"):
        audit_rgb_multiview_parallax(poses, xyz, altered, lid)
    with pytest.raises(ValueError, match="invalid"):
        audit_rgb_multiview_parallax(
            poses, xyz, fid, lid, min_parallax_deg=0.,
        )
    with pytest.raises(RuntimeError, match="cancelled"):
        audit_rgb_multiview_parallax(
            poses, xyz, fid, lid, should_cancel=lambda: True,
        )


def test_one_weak_flight_region_fails_despite_global_80_percent_parallax():
    """8 weak tracks among 40 still leave a healthy global fraction (80 %).

    The weak 3-camera patch has 8 observations per camera, so a simple
    8-point per-camera support gate or global parallax fraction would pass.
    Its points have essentially no camera baseline and must be rejected.
    """
    poses, xyz, fid, lid = _scene(n=12, count=32)
    extra_poses = tuple(
        np.column_stack((
            np.eye(3), np.array((-1e-6*(j+1), 0., 0.))
        ))
        for j in range(3)
    )
    weak_xyz = np.column_stack((
        np.linspace(-.5, .5, 8), np.zeros(8), np.full(8, 10.)
    ))
    extra_fid = np.tile(np.array((12, 13, 14), dtype=np.int32), 8)
    extra_lid = np.repeat(np.arange(32, 40, dtype=np.int32), 3)
    with pytest.raises(RGBMultiviewParallaxQualityError) as error:
        audit_rgb_multiview_parallax(
            poses + extra_poses, np.vstack((xyz, weak_xyz)),
            np.concatenate((fid, extra_fid)),
            np.concatenate((lid, extra_lid)),
        )
    summary = error.value.audit.privacy_safe_summary()
    assert summary["supported_landmark_fraction"] == .8
    assert summary["cameras_below_supported_observation_fraction"] == 3
    assert summary["worst_camera_supported_observation_fraction"] == 0.
    assert summary["parallax_quality_passed"] is False
    assert "camera_indices" not in json.dumps(summary)
    assert "V-0013" not in json.dumps(summary)


def test_missing_camera_3d_support_and_invalid_fraction_fail_closed():
    poses, xyz, fid, lid = _scene()
    # A pose can exist without any accepted 3D track: never claim this is
    # a fully supported RGB camera network.
    with pytest.raises(ValueError, match="lack any 3D track"):
        audit_rgb_multiview_parallax(
            poses, xyz, fid[fid != 11], lid[fid != 11],
        )
    with pytest.raises(ValueError, match="invalid bounded"):
        audit_rgb_multiview_parallax(
            poses, xyz, fid, lid, min_supported_camera_fraction=0.,
        )
