"""Common-focal BA on a nonplanar multiview scene, without customer data."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_global_camera_network import RGBUnscaledNetwork
from mcm_solarcheck.services.rgb_global_bundle_adjustment import (
    adjust_unscaled_rgb_bundle,
)
from mcm_solarcheck.services.rgb_windowed_bundle_adjustment import (
    RGBWindowedBAResult, adjust_windowed_unscaled_rgb_network,
)
from mcm_solarcheck.services.rgb_rtk_camera_priors import PrivateRGBRTKPriors
from mcm_solarcheck.services.rgb_rtk_joint_bundle_adjustment import (
    adjust_rgb_windowed_bundle_with_private_rtk,
)


def _nonplanar_scene(*, focal_scale=.82, count=6):
    rng = np.random.default_rng(20261030)
    truth = PinholeIntrinsics(1000., 950., 500., 400.)
    points = np.column_stack((
        rng.uniform(-2.5, 2.5, 48),
        rng.uniform(-1.8, 1.8, 48),
        rng.uniform(5., 13., 48),
    ))
    centers = np.array([
        (0., 0., 0.), (.65, .10, .25), (1.1, -.2, -.10),
        (1.5, .3, .2), (.8, .7, -.2), (-.2, .6, .3),
    ])
    rotations = np.array([
        (0., 0., 0.), (.01, .14, .02), (-.03, -.11, .01),
        (.06, .08, -.03), (-.07, .18, .04), (.04, -.09, -.02),
    ])
    if count > len(centers):
        extra = np.arange(len(centers), count, dtype=float)
        centers = np.vstack((centers, np.column_stack((
            .2 * (extra - 4), .5 * np.sin(.7 * extra), .1 * np.cos(.9 * extra),
        ))))
        rotations = np.vstack((rotations, np.column_stack((
            .06 * np.sin(extra), .1 * np.cos(.5 * extra), .03 * np.sin(.2 * extra),
        ))))
    poses, observed, fid, lid = [], [], [], []
    for i in range(count):
        rotation, _ = cv2.Rodrigues(rotations[i])
        translation = -rotation @ centers[i]
        camera = points @ rotation.T + translation
        assert np.all(camera[:, 2] > 0)
        pixels = np.column_stack((
            truth.fx * camera[:, 0] / camera[:, 2] + truth.cx,
            truth.fy * camera[:, 1] / camera[:, 2] + truth.cy,
        ))
        pixels += rng.normal(0., .01, pixels.shape)
        observed.extend(pixels)
        fid.extend([i] * len(points))
        lid.extend(range(len(points)))
        # The existing arbitrary-gauge anchors remain exact. Other poses and
        # every point have small errors, so focal cannot improve in isolation
        # by merely reporting the supplied nominal value as its own result.
        pose = np.column_stack((rotation, translation))
        if i >= 2:
            pose[:, 3] += rng.normal(0., .005, 3)
        poses.append(pose)
    network = RGBUnscaledNetwork(
        tuple(f"synthetic-{i}" for i in range(count)), tuple(poses),
        len(points), len(points), count - 1, 1., 2.,
        landmark_xyz_unit_baseline=points + rng.normal(0., .006, points.shape),
        observation_landmark_indices=np.asarray(lid, dtype=np.int32),
        observation_frame_indices=np.asarray(fid, dtype=np.int32),
        observation_pixels_xy=np.asarray(observed, dtype=np.float64),
    )
    initial = PinholeIntrinsics(
        truth.fx * focal_scale, truth.fy * focal_scale, truth.cx, truth.cy,
    )
    return network, initial, truth


def _actual_rmse(poses, points, fid, lid, observed, camera):
    projected = []
    for frame, landmark in zip(fid, lid):
        xyz = poses[frame][:, :3] @ points[landmark] + poses[frame][:, 3]
        projected.append((
            camera.fx * xyz[0] / xyz[2] + camera.cx,
            camera.fy * xyz[1] / xyz[2] + camera.cy,
        ))
    return float(np.sqrt(np.mean(np.sum((np.asarray(projected) - observed) ** 2, axis=1))))


def test_common_focal_is_optimized_from_actual_multiview_observations():
    network, initial, truth = _nonplanar_scene()
    original_points = network.landmark_xyz_unit_baseline.copy()
    original_poses = tuple(x.copy() for x in network.world_to_camera)
    result = adjust_unscaled_rgb_bundle(
        network, initial, refine_focal=True, max_iterations=35,
    )
    fitted = result.refined_intrinsics
    assert fitted is not None and result.focal_refined
    assert result.focal_observable
    assert result.focal_information_fraction > 1.e-7
    assert abs(fitted.fx - truth.fx) < .03 * truth.fx
    assert abs(fitted.fx - truth.fx) < .2 * abs(initial.fx - truth.fx)
    assert result.after_rmse_px < .1 * result.before_rmse_px
    assert result.iterations > 0
    assert fitted.cx == initial.cx and fitted.cy == initial.cy
    assert fitted.fx / fitted.fy == pytest.approx(initial.fx / initial.fy)
    np.testing.assert_array_equal(result.world_to_camera[0], original_poses[0])
    np.testing.assert_array_equal(result.world_to_camera[1], original_poses[1])
    np.testing.assert_array_equal(network.landmark_xyz_unit_baseline, original_points)
    for actual, expected in zip(network.world_to_camera, original_poses):
        np.testing.assert_array_equal(actual, expected)
    assert not result.metric_scale_verified
    assert not result.roof_geometry_verified
    assert not result.orthophoto_verified


def test_default_and_explicit_fixed_focal_keep_the_same_legacy_solution():
    network, initial, _ = _nonplanar_scene(focal_scale=1.)
    implicit = adjust_unscaled_rgb_bundle(network, initial, max_iterations=5)
    explicit = adjust_unscaled_rgb_bundle(
        network, initial, refine_focal=False, max_iterations=5,
    )
    assert not implicit.focal_refined and not explicit.focal_refined
    assert implicit.before_rmse_px == explicit.before_rmse_px
    assert implicit.after_rmse_px == explicit.after_rmse_px
    assert implicit.iterations == explicit.iterations
    for actual, expected in zip(implicit.world_to_camera, explicit.world_to_camera):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        implicit.landmark_xyz_unit_baseline, explicit.landmark_xyz_unit_baseline,
    )


def test_joint_update_respects_explicit_focal_bounds_when_truth_is_outside():
    network, initial, truth = _nonplanar_scene()
    assert truth.fx > initial.fx * 1.05
    result = adjust_unscaled_rgb_bundle(
        network, initial, refine_focal=True, max_iterations=35,
        focal_scale_bounds=(.90, 1.05),
    )
    fitted = result.refined_intrinsics
    assert fitted is not None
    assert .90 * initial.fx <= fitted.fx <= 1.05 * initial.fx
    assert fitted.fx / fitted.fy == pytest.approx(initial.fx / initial.fy)
    assert result.after_rmse_px < result.before_rmse_px
    assert not result.metric_scale_verified
    assert not result.roof_geometry_verified
    assert not result.orthophoto_verified


def test_refined_focal_is_carried_across_overlapping_windows_and_global_audit():
    network, initial, truth = _nonplanar_scene(count=12)
    result = adjust_windowed_unscaled_rgb_network(
        network, initial, refine_focal=True, max_iterations=25,
    )
    assert result.completed_windows == 3 and result.camera_count == 12
    assert result.focal_refined and result.refined_intrinsics is not None
    fitted = result.refined_intrinsics
    assert abs(fitted.fx - truth.fx) < .05 * truth.fx
    assert result.after_rmse_px < .1 * result.before_rmse_px
    # Reporting residuals under the old nominal K would be inconsistent even
    # if each small solver itself happened to improve its own focal value.
    independent = _actual_rmse(
        result.world_to_camera, result.landmark_xyz_unit_baseline,
        network.observation_frame_indices, network.observation_landmark_indices,
        network.observation_pixels_xy, fitted,
    )
    assert result.after_rmse_px == pytest.approx(independent, abs=1.e-10)
    assert not result.metric_scale_verified
    assert not result.georeferencing_verified
    assert not result.global_joint_ba_completed
    assert not result.orthophoto_verified


def test_later_refinement_stays_inside_the_original_flight_focal_range():
    network, current, truth = _nonplanar_scene(count=12)
    # A previous pass already moved focal to 1.8 times the original estimate.
    # The next pass must not receive a fresh factor-of-two range around that
    # updated camera, even though the image geometry prefers a larger value.
    reference = PinholeIntrinsics(
        current.fx / 1.8, current.fy / 1.8, current.cx, current.cy,
    )
    assert truth.fx > 2. * reference.fx
    fitted = adjust_windowed_unscaled_rgb_network(
        network, current, refine_focal=True, max_iterations=25,
        focal_reference_intrinsics=reference,
    )
    assert fitted.refined_intrinsics is not None
    assert .5 * reference.fx - 1.e-9 <= fitted.refined_intrinsics.fx <= 2. * reference.fx + 1.e-9
    assert fitted.focal_reference_intrinsics == reference
    assert fitted.after_rmse_px < fitted.before_rmse_px


def test_rtk_wrapper_returns_the_final_common_camera_without_releasing_gates():
    network, initial, truth = _nonplanar_scene(count=12)
    centers = np.asarray([-p[:, :3].T @ p[:, 3] for p in network.world_to_camera])
    sigma = np.tile((.10, .10, .20), (12, 1))
    priors = PrivateRGBRTKPriors(
        network.frame_ids, (50., 7., 100.), centers, sigma,
        np.zeros((12, 3)), np.zeros((12, 3)), tuple('50' for _ in range(12)),
        'RtkAlt',
    )
    previous = RGBWindowedBAResult(
        network.world_to_camera, network.landmark_xyz_unit_baseline,
        10., 9., 20., 18., 12, 48, len(network.observation_frame_indices), 3, 6,
    )
    result = adjust_rgb_windowed_bundle_with_private_rtk(
        network, previous, priors, initial,
        refine_focal=True, windowed_options={'max_iterations': 25},
    )
    assert result.focal_refined and result.refined_intrinsics is not None
    assert abs(result.refined_intrinsics.fx - truth.fx) < .05 * truth.fx
    assert result.total_cameras == 12
    assert result.fitted_rtk_camera_count == 10
    assert result.withheld_rtk_camera_count == 2
    assert result.joint_objective_after < result.joint_objective_before
    assert not result.independent_survey_verified
    assert not result.roof_geometry_verified
    assert not result.orthophoto_verified


@pytest.mark.parametrize("flag", [None, 1, "yes", np.nan])
def test_refinement_mode_requires_an_explicit_boolean(flag):
    network, initial, _ = _nonplanar_scene()
    with pytest.raises((TypeError, ValueError), match="focal|refine|invalid|settings"):
        adjust_unscaled_rgb_bundle(network, initial, refine_focal=flag)


def test_nonfinite_or_missing_multiview_support_never_yields_focal_evidence():
    network, initial, _ = _nonplanar_scene()
    broken_pixels = network.observation_pixels_xy.copy()
    broken_pixels[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite|invalid"):
        adjust_unscaled_rgb_bundle(
            replace(network, observation_pixels_xy=broken_pixels),
            initial, refine_focal=True,
        )
    one_camera = replace(
        network,
        observation_frame_indices=np.zeros_like(network.observation_frame_indices),
    )
    with pytest.raises(ValueError, match="support|insufficient|invalid"):
        adjust_unscaled_rgb_bundle(one_camera, initial, refine_focal=True)


@pytest.mark.parametrize("use_gnss", [False, True])
def test_parallel_cameras_expose_focal_depth_gauge_instead_of_calibration(use_gnss):
    """Even perfect pixels cannot distinguish focal from depth in this motion."""
    rng = np.random.default_rng(20261031)
    truth = PinholeIntrinsics(1000., 950., 500., 400.)
    scale = .82
    nominal = PinholeIntrinsics(truth.fx * scale, truth.fy * scale, 500., 400.)
    true_points = np.column_stack((
        rng.uniform(-2., 2., 36), rng.uniform(-1.2, 1.2, 36),
        rng.uniform(5., 13., 36),
    ))
    initial_points = true_points.copy()
    initial_points[:, 2] *= scale
    poses, pixels, fid, lid = [], [], [], []
    for camera in range(6):
        # Parallel optical axes and equal camera altitude permit focal and
        # *all* scene depths to scale together, despite nonplanar scene points.
        center = np.array((.25 * camera, .1 * np.sin(camera), 0.))
        pose = np.column_stack((np.eye(3), -center))
        xyz = true_points - center
        image = np.column_stack((
            truth.fx * xyz[:, 0] / xyz[:, 2] + truth.cx,
            truth.fy * xyz[:, 1] / xyz[:, 2] + truth.cy,
        ))
        poses.append(pose)
        pixels.extend(image)
        fid.extend([camera] * len(true_points))
        lid.extend(range(len(true_points)))
    network = RGBUnscaledNetwork(
        tuple(f"parallel-{i}" for i in range(6)), tuple(poses), 36, 36, 5, 0., 0.,
        landmark_xyz_unit_baseline=initial_points,
        observation_landmark_indices=np.asarray(lid, dtype=np.int32),
        observation_frame_indices=np.asarray(fid, dtype=np.int32),
        observation_pixels_xy=np.asarray(pixels, dtype=np.float64),
    )
    assert _actual_rmse(
        poses, initial_points, network.observation_frame_indices,
        network.observation_landmark_indices, network.observation_pixels_xy,
        nominal,
    ) < 1.e-10
    options = {}
    if use_gnss:
        # Exact GNSS centers constrain pose, but must not be counted as
        # pixel-only focal information or hide the focal/depth ambiguity.
        options = {
            "camera_center_priors_m": np.asarray([-p[:, 3] for p in poses]),
            "camera_center_sigma_m": np.tile((.1, .1, .2), (6, 1)),
        }
    result = adjust_unscaled_rgb_bundle(
        network, nominal, refine_focal=True, **options,
    )
    assert result.focal_information_fraction is not None
    assert result.focal_information_fraction <= 1.e-7
    assert not result.focal_observable
    assert not result.metric_scale_verified
    assert not result.roof_geometry_verified
    assert not result.orthophoto_verified
