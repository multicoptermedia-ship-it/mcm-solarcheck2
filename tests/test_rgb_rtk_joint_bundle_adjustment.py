"""RTK regularized full-flight Schur/windowed BA; all inputs synthetic and private."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from mcm_solarcheck.domain.models import Pose, Position, RTKQuality
from mcm_solarcheck.services.rgb_flight_zip_series import (
    PrivateRGBFlightFrame, PrivateRGBFlightSeries,
)
from mcm_solarcheck.services.rgb_global_camera_network import RGBUnscaledNetwork
from mcm_solarcheck.services.rgb_windowed_bundle_adjustment import RGBWindowedBAResult
from mcm_solarcheck.services.rgb_global_bundle_adjustment import adjust_unscaled_rgb_bundle
from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_rtk_camera_priors import build_private_rtk_camera_priors
from mcm_solarcheck.services.rgb_rtk_joint_bundle_adjustment import (
    adjust_rgb_windowed_bundle_with_private_rtk,
)


def _synthetic_full_rgb_flight(n=12, landmarks_count=None):
    # Realistic LOCAL cross-frame tracks, not every landmark in every image.
    if landmarks_count is None:
        landmarks_count = 8 * n
    rng = np.random.default_rng(20261028)
    k = PinholeIntrinsics(760., 760., 400., 300.)
    points_true = np.column_stack((
        rng.uniform(-2.5, 4.5, landmarks_count),
        rng.uniform(-2.5, 2.5, landmarks_count),
        rng.uniform(9., 15., landmarks_count),
    ))
    i = np.arange(n, dtype=float)
    centers = np.column_stack((.24*i, .55*np.sin(.33*i), .07*np.sin(.51*i)))
    # Simulated RTK frame differs from unscaled SfM frame in scale/rotation.
    heading = np.deg2rad(17.)
    turn = np.array([[np.cos(heading),-np.sin(heading),0.],
                     [np.sin(heading),np.cos(heading),0.],
                     [0.,0.,1.]])
    enu = 1.35 * centers @ turn.T + np.array((9.,-5.,1.))
    ref_lat,ref_lon,ref_alt = 50.,7.,100.
    phi = np.deg2rad(ref_lat)
    e2 = 6.6943799901413165e-3
    a = 6378137.
    radius_n = a / np.sqrt(1-e2*np.sin(phi)**2)
    radius_m = a*(1-e2)/(1-e2*np.sin(phi)**2)**1.5
    east_per_degree = radius_n*np.cos(phi)*np.pi/180.
    north_per_degree = radius_m*np.pi/180.
    frame_ids,poses,observed_pixels,obs_fid,obs_lid,frames = [],[],[],[],[],[]
    for t,(center,cenu) in enumerate(zip(centers,enu)):
        fid = f"V-{t+1:04d}"
        frame_ids.append(fid)
        camera = points_true - center
        pixels = np.column_stack((
            k.fx * camera[:,0] / camera[:,2] + k.cx,
            k.fy * camera[:,1] / camera[:,2] + k.cy,
        ))
        pixels += rng.normal(0.,.05,pixels.shape)
        # Every synthetic landmark is in at least two and at most four
        # adjacent frames, spanning several overlapping camera windows.
        groups = np.minimum(np.arange(landmarks_count)//8, n-2)
        visible = np.flatnonzero((groups <= t) & (t < groups+4))
        observed_pixels.extend(pixels[visible])
        obs_fid.extend([t]*len(visible))
        obs_lid.extend(visible.tolist())
        biased_center = center + rng.normal(0.,.02,3)
        if t in (0,1):
            biased_center += np.array((.08,-.07,.04))
        poses.append(np.column_stack((np.eye(3),-biased_center)))
        frames.append(PrivateRGBFlightFrame(
            Path("internal-only.zip"),f"private/{fid}.jpg",fid,
            "20260825120000",4000,3000,
            Position(ref_lat+cenu[1]/north_per_degree,
                     ref_lon+cenu[0]/east_per_degree,
                     ref_alt+cenu[2]),
            Pose(1.,-90.,0.), Pose(1.,0.,0.),
            RTKQuality("50",.015,.015,.03,altitude_type="RtkAlt"),
            ("DewarpData",),12345,
        ))
    seed_points = points_true + rng.normal(0.,.02,points_true.shape)
    unscaled = RGBUnscaledNetwork(
        tuple(frame_ids),tuple(poses),landmarks_count,landmarks_count,n-1,1.,2.,
        landmark_xyz_unit_baseline=seed_points,
        observation_frame_indices=np.asarray(obs_fid,dtype=np.int32),
        observation_landmark_indices=np.asarray(obs_lid,dtype=np.int32),
        observation_pixels_xy=np.asarray(observed_pixels,dtype=np.float64),
    )
    windows = RGBWindowedBAResult(
        tuple(poses),seed_points,2.0,1.0,3.,2.,n,landmarks_count,
        len(obs_fid),3,6,
    )
    return PrivateRGBFlightSeries(tuple(frames),6,0),unscaled,windows,k


def test_rtk_weighted_window_ba_reuses_full_multiview_network_and_withholds_rtk():
    flight,network,windowed,k = _synthetic_full_rgb_flight()
    priors = build_private_rtk_camera_priors(flight,network.frame_ids)
    prior_original = priors.camera_enu_m.copy()
    answer = adjust_rgb_windowed_bundle_with_private_rtk(
        network,windowed,priors,k,windowed_options={"max_iterations": 5},
        enforce_camera_p95=False,  # diagnostic candidate is NOT an accepted flight
        enforce_parallax=False,
    )
    report = answer.privacy_safe_summary()
    assert answer.total_cameras == 12
    assert answer.fitted_rtk_camera_count == 10
    assert answer.withheld_rtk_camera_count == 2
    assert answer.completed_windows == 3
    assert answer.windows_with_updates >= 1
    assert (answer.windows_with_updates + answer.windows_without_improvement
            == answer.completed_windows)
    assert ("windows_with_updates" in answer.privacy_safe_summary()
            and "windows_without_improvement" in answer.privacy_safe_summary())
    assert answer.joint_objective_after <= answer.joint_objective_before + 1e-8
    assert answer.after_rtk_holdout_max_m < 2.
    assert np.isfinite(answer.after_reprojection_rmse_px)
    assert answer.camera_reprojection_audit is not None
    assert answer.camera_reprojection_audit.camera_count == 12
    assert answer.camera_reprojection_audit.minimum_observations_per_camera >= 8
    assert answer.camera_reprojection_audit.minimum_cameras_per_landmark >= 2
    assert answer.camera_reprojection_audit.worst_camera_p95_px >= 0.
    assert (report["per_camera_reprojection"]["camera_reprojection_quality_passed"]
            == (answer.camera_reprojection_audit.cameras_above_p95_limit == 0))
    if answer.camera_reprojection_audit.cameras_above_p95_limit:
        assert answer.state == "rtk_camera_reprojection_quality_failed_unverified"
    # The real synthetic Schur/RTK BA must reach the next vertical geometry
    # stage and retain PRIVATE 3D hypotheses, without claiming roof success.
    assert answer.pitched_plane_candidates is not None
    assert answer.pitched_plane_candidates.landmark_count == len(
        answer.landmarks_enu_private_m
    )
    assert report["pitched_plane_hypotheses"]["roof_geometry_verified"] is False
    assert report["pitched_plane_hypotheses"]["orthophoto_verified"] is False
    assert report["pitched_plane_hypotheses"]["independent_survey_verified"] is False
    assert report["multiview_parallax"]["parallax_audit_completed"] is True
    assert report["multiview_parallax"]["independent_survey_verified"] is False
    assert report["multiview_parallax"]["roof_geometry_verified"] is False
    assert report["per_camera_reprojection"]["camera_reprojection_audit_completed"]
    assert report["per_camera_reprojection"]["orthophoto_verified"] is False
    assert answer.rtk_prior_windowed_ba_completed
    assert not answer.full_global_joint_ba_completed
    assert not answer.independent_survey_verified
    assert not answer.roof_geometry_verified
    assert not answer.orthophoto_verified
    assert answer.world_enu_to_camera_private[0].shape == (3,4)
    assert not answer.world_enu_to_camera_private[0].flags.writeable
    assert np.array_equal(priors.camera_enu_m,prior_original)
    text = json.dumps(report)
    for private in ("private/", "50.0", "V-0001", "RtkAlt", "latitude",
                    "longitude", "image_path"):
        assert private not in text
    assert report["full_global_joint_ba_completed"] is False


def test_small_schur_gnss_mode_unfixes_first_two_cameras_and_validates_weights():
    flight,network,windowed,k = _synthetic_full_rgb_flight(6)
    # Target positions live in the same coordinate space as this 6-camera scene.
    initial = np.array([-p[:,:3].T @ p[:,3] for p in windowed.world_to_camera])
    desired = initial.copy()
    desired[0,:] += np.array((.05,-.04,.02))
    desired[1,:] += np.array((.04,.01,-.03))
    sigma = np.full((6,3),.30)
    sigma[:,2] = .40
    result = adjust_unscaled_rgb_bundle(
        network,k,max_cameras=6,max_landmarks=100,
        camera_center_priors_m=desired,camera_center_sigma_m=sigma,
        max_iterations=4,
    )
    assert result.rtk_camera_priors_optimized
    assert result.state.startswith("rtk_regularized")
    # Existing legacy path continues fixing cameras zero and one.
    legacy = adjust_unscaled_rgb_bundle(
        network,k,max_cameras=6,max_landmarks=100,max_iterations=4,
    )
    np.testing.assert_allclose(legacy.world_to_camera[0],network.world_to_camera[0])
    np.testing.assert_allclose(legacy.world_to_camera[1],network.world_to_camera[1])
    assert (not np.allclose(result.world_to_camera[0],network.world_to_camera[0])
            or not np.allclose(result.world_to_camera[1],network.world_to_camera[1]))
    with pytest.raises(ValueError,match="conservative RTK"):
        adjust_unscaled_rgb_bundle(
            network,k,max_cameras=6,max_landmarks=100,
            camera_center_priors_m=desired,
            camera_center_sigma_m=np.full((6,3),.0001),
        )


def test_private_window_ba_rejects_identity_mismatch_and_user_overridden_holdouts():
    flight,network,windowed,k = _synthetic_full_rgb_flight()
    priors = build_private_rtk_camera_priors(flight,network.frame_ids)
    with pytest.raises(ValueError,match="matched RTK"):
        adjust_rgb_windowed_bundle_with_private_rtk(
            network,windowed,replace(priors,frame_ids=priors.frame_ids[::-1]),k,
        )
    with pytest.raises(ValueError,match="internal"):
        adjust_rgb_windowed_bundle_with_private_rtk(
            network,windowed,priors,k,
            windowed_options={"rtk_fit_mask":np.ones(12,dtype=bool)},
        )
    with pytest.raises(RuntimeError,match="cancelled"):
        adjust_rgb_windowed_bundle_with_private_rtk(
            network,windowed,priors,k,should_cancel=lambda:True,
        )


def test_strict_rtk_camera_reprojection_gate_rejects_bad_synthetic_window():
    """A robust pixel+GNSS objective pass is NOT a per-camera quality pass."""
    flight,network,windowed,k = _synthetic_full_rgb_flight()
    priors = build_private_rtk_camera_priors(flight,network.frame_ids)
    diagnostic = adjust_rgb_windowed_bundle_with_private_rtk(
        network,windowed,priors,k,
        windowed_options={"max_iterations": 5},
        enforce_camera_p95=False,
        enforce_parallax=False,
    )
    # The same synthetic fixture has known per-camera residual outliers:
    # never silently accept this candidate in the full-flight/strict path.
    assert diagnostic.camera_reprojection_audit.cameras_above_p95_limit >= 1
    with pytest.raises(ValueError,match="individual RGB camera"):
        adjust_rgb_windowed_bundle_with_private_rtk(
            network,windowed,priors,k,
            windowed_options={"max_iterations": 5},
            enforce_camera_p95=True,
        )


def test_final_camera_audit_uses_fitted_focal_instead_of_nominal_camera(monkeypatch):
    """The final strict pixel gate must evaluate the geometry with its fitted K."""
    from mcm_solarcheck.services import rgb_rtk_joint_bundle_adjustment as service
    from mcm_solarcheck.services.rgb_per_camera_reprojection_gate import (
        audit_rgb_per_camera_reprojection,
    )
    flight, network, windowed, truth = _synthetic_full_rgb_flight()
    priors = build_private_rtk_camera_priors(flight, network.frame_ids)
    fid = network.observation_frame_indices
    lid = network.observation_landmark_indices
    poses = np.asarray(network.world_to_camera)
    xyz = np.einsum("nij,nj->ni", poses[fid, :, :3],
                    network.landmark_xyz_unit_baseline[lid]) + poses[fid, :, 3]
    observed = np.column_stack((truth.fx * xyz[:, 0] / xyz[:, 2] + truth.cx,
                                truth.fy * xyz[:, 1] / xyz[:, 2] + truth.cy))
    network = replace(network, observation_pixels_xy=observed)
    nominal = PinholeIntrinsics(.7 * truth.fx, .7 * truth.fy, truth.cx, truth.cy)

    def fitted_window(metric_candidate, starting_camera, **options):
        assert starting_camera == nominal
        assert options["refine_focal"] is True
        # Keep the real Sim(3)-aligned geometry and the real downstream pixel
        # audit. Isolate the handoff of the fitted common camera from BA.
        return replace(
            windowed, world_to_camera=metric_candidate.world_to_camera,
            landmark_xyz_unit_baseline=metric_candidate.landmark_xyz_unit_baseline,
            rtk_camera_priors_optimized=True,
            weighted_objective_before=1., weighted_objective_after=.5,
            refined_intrinsics=truth, focal_refined=True,
            focal_reference_intrinsics=nominal,
        )

    monkeypatch.setattr(service, "adjust_windowed_unscaled_rgb_network", fitted_window)
    answer = adjust_rgb_windowed_bundle_with_private_rtk(
        network, windowed, priors, nominal, refine_focal=True,
        enforce_parallax=False,
    )
    assert answer.refined_intrinsics == truth
    assert answer.focal_refined
    assert answer.camera_reprojection_audit.cameras_above_p95_limit == 0
    assert answer.camera_reprojection_audit.flight_max_px < 1.e-8
    with pytest.raises(ValueError, match="individual RGB camera"):
        audit_rgb_per_camera_reprojection(
            answer.world_enu_to_camera_private, answer.landmarks_enu_private_m,
            fid, lid, observed, nominal,
        )
    assert not answer.roof_geometry_verified
    assert not answer.orthophoto_verified
