"""Multi-frame synthetic 3D survey-free camera network tests.

Pixel correspondences are generated mathematically from 3D points. This
tests the spatial SfM initializer, not just a pair homography.
"""
import numpy as np
import pytest
import cv2
from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_global_camera_network import (
    RGBNetworkFrame,RGBNetworkLink,solve_global_rgb_camera_network,
)


def synthetic_observations(count=4):
    rng=np.random.default_rng(20261028)
    xyz=np.column_stack((
        rng.uniform(-2.2,2.2,90),
        rng.uniform(-1.5,1.5,90),
        rng.uniform(6.0,12.0,90),
    ))
    k=PinholeIntrinsics(740.,740.,400.,300.)
    frames=[]
    for i in range(count):
        translation=np.array((-.65*i,0.03*i,0.))
        camera=xyz+translation
        xy=np.column_stack((k.fx*camera[:,0]/camera[:,2]+k.cx,
                            k.fy*camera[:,1]/camera[:,2]+k.cy))
        xy+=rng.normal(0,.10,xy.shape)
        frames.append(RGBNetworkFrame(f"frame-{i}",xy.astype(np.float64)))
    edges=[]
    for i in range(count):
        for j in range(i+1,min(count,i+3)):
            edges.append(RGBNetworkLink(
                f"frame-{i}",f"frame-{j}",
                np.array([(index,index) for index in range(len(xyz))],np.int32),
            ))
    return tuple(frames),tuple(edges),k


def test_four_real_3d_views_form_one_global_camera_and_landmark_network():
    frames,links,k=synthetic_observations()
    result=solve_global_rgb_camera_network(frames,links,k)
    assert result.frame_ids==tuple(f.frame_id for f in frames)
    assert len(result.world_to_camera)==4
    assert result.reconstructed_landmarks>=12
    assert result.multiview_tracks>=12
    assert result.linked_image_pairs>=3
    assert len(result.accepted_link_pairs) == result.linked_image_pairs
    proposed = {(link.first_id, link.second_id) for link in links}
    assert set(result.accepted_link_pairs).issubset(proposed)
    assert len(set(result.accepted_link_pairs)) == len(result.accepted_link_pairs)
    assert result.reprojection_p95_px < 2.
    assert all(np.isfinite(pose).all() and pose.shape==(3,4)
               for pose in result.world_to_camera)
    assert not result.metric_scale_verified
    assert not result.roof_geometry_verified
    assert not result.orthophoto_verified
    assert not result.bundle_adjustment_completed
    assert result.landmark_xyz_unit_baseline.shape == (result.reconstructed_landmarks,3)
    assert result.observation_pixels_xy.shape[1] == 2
    assert len(result.observation_landmark_indices) == len(result.observation_frame_indices) == len(result.observation_pixels_xy)
    assert set(result.observation_frame_indices.tolist()) == {0,1,2,3}
    assert not result.landmark_xyz_unit_baseline.flags.writeable


def test_disconnected_fourth_image_must_not_be_accepted_as_a_full_network():
    frames,links,k=synthetic_observations()
    edges=tuple(l for l in links if l.second_id!="frame-3")
    with pytest.raises(ValueError,match="disconnected|underconstrained"):
        solve_global_rgb_camera_network(frames,edges,k)


def test_failed_pnp_candidate_is_retried_after_another_camera_adds_support(monkeypatch):
    frames, links, k = synthetic_observations(count=5)
    original = cv2.solvePnPRansac
    calls = []
    def first_candidate_fails(*args, **kwargs):
        calls.append(len(args[0]))
        if len(calls) == 1:
            return False, None, None, None
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, "solvePnPRansac", first_candidate_fails)
    result = solve_global_rgb_camera_network(frames, links, k)
    assert len(result.world_to_camera) == 5
    assert len(calls) >= 4
    assert result.reprojection_p95_px < 3.
    assert not result.orthophoto_verified


def test_all_pnp_candidates_failing_still_rejects_complete_network(monkeypatch):
    frames, links, k = synthetic_observations(count=5)
    calls = []
    def no_pose(*args, **kwargs):
        calls.append(len(args[0]))
        return False, None, None, None
    monkeypatch.setattr(cv2, "solvePnPRansac", no_pose)
    with pytest.raises(ValueError, match="PnP.*insufficient 3D support"):
        solve_global_rgb_camera_network(frames, links, k)
    assert len(calls) >= 2


def test_false_missing_intrinsics_and_insufficient_shared_tracks_fail():
    frames,links,k=synthetic_observations()
    with pytest.raises(ValueError,match="calibrated camera intrinsics"):
        solve_global_rgb_camera_network(frames,links,PinholeIntrinsics(0,740,400,300))
    with pytest.raises(ValueError,match="network|tracks|calibrated"):
        solve_global_rgb_camera_network(frames[:3],(),k)


def test_duplicate_frame_and_oversized_feature_matches_fail_closed():
    frames,links,k=synthetic_observations()
    invalid=(frames[0],frames[0],frames[2])
    with pytest.raises(ValueError,match="unique"):
        solve_global_rgb_camera_network(invalid,links,k)
    with pytest.raises(ValueError,match="correspondence"):
        bad=RGBNetworkLink(links[0].first_id,links[0].second_id,np.full((90,2),9999,np.int32))
        solve_global_rgb_camera_network(frames,(bad,)+links[1:],k)


def test_project_database_to_global_network_uses_all_four_rgb_sources(tmp_path, monkeypatch):
    """Real synthetic SQLite/JPEG inventory, controlled 3D feature observations.

    ORB is replaced ONLY for reproducible feature localization. Real project
    reader, decoding, candidate graph, descriptor matching and pose solver run.
    """
    import sqlite3
    from mcm_solarcheck.services import rgb_project_global_camera_network as service

    synthetic_frames, _, k = synthetic_observations()
    dbpath = tmp_path / "input.sqlite"
    originals = tmp_path / "source"
    originals.mkdir()
    with sqlite3.connect(dbpath) as db:
        db.execute("CREATE TABLE projects(project_id TEXT)")
        db.execute("CREATE TABLE image_frames(project_id TEXT, frame_id TEXT, source_file TEXT, width INTEGER, height INTEGER)")
        db.execute("INSERT INTO projects VALUES (?)", ("P-CAMERAS",))
        for i in range(4):
            path = originals / f"DJI_20260820112430_{120+i:04d}_V.JPG"
            assert cv2.imwrite(str(path), np.full((600,800,3), 30*(i+1), np.uint8))
            db.execute(
                "INSERT INTO image_frames VALUES (?, ?, ?, ?, ?)",
                ("P-CAMERAS", f"frame-{i}", str(path), 800, 600),
            )
    # Unique ORB signatures per 3D landmark, exactly shared across views.
    rng = np.random.default_rng(3491)
    descriptors = rng.integers(0, 256, (90,32), dtype=np.uint8)
    class Feature:
        def __init__(self, pt):
            self.pt = tuple(pt)
    class SyntheticDetector:
        def detectAndCompute(self, image, mask):
            i = round(float(image.mean())/30) - 1
            assert 0 <= i < 4
            return [Feature(pt) for pt in synthetic_frames[i].points_xy], descriptors
    monkeypatch.setattr(service.cv2, "ORB_create", lambda **kwargs: SyntheticDetector())
    progress = []
    calibration = {
        "schema": "mcm.rgb_camera_intrinsics_external.v1",
        "calibration_provenance": "independently generated synthetic K",
        "lens_calibration_verified": True,
        "intrinsics": {"fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy},
        "distortion_coefficients": [0,0,0,0,0],
    }
    result = service.reconstruct_project_rgb_camera_network(
        dbpath, "P-CAMERAS", calibration, on_progress=lambda *tick: progress.append(tick),
    )
    assert len(result.frame_ids) == 4
    assert result.linked_image_pairs >= 3
    assert result.reconstructed_landmarks >= 12
    assert [x[0] for x in progress[:4]] == ["features"]*4
    assert progress[3] == ("features",4,4)
    assert not result.metric_scale_verified
    assert not result.bundle_adjustment_completed


def test_cli_saves_reusable_unscaled_3d_track_bundle(tmp_path,monkeypatch,capsys):
    from mcm_solarcheck.services import rgb_global_camera_network_cli as cli
    import json

    frames,links,k=synthetic_observations()
    network=solve_global_rgb_camera_network(frames,links,k)
    monkeypatch.setattr(cli,"reconstruct_project_rgb_camera_network",lambda *a,**kw:network)
    calibration=tmp_path/"camera.json"
    calibration.write_text(json.dumps({
        "intrinsics":{"fx":k.fx,"fy":k.fy,"cx":k.cx,"cy":k.cy},
    }),encoding="utf-8")
    out=tmp_path/"output"
    out.mkdir()
    assert cli.main([
        "--database","synthetic.sqlite","--project-id","P-TEST",
        "--calibration-json",str(calibration),
        "--output-directory",str(out),
    ])==0
    summary=json.loads((out/"rgb-global-camera-network-unscaled.json").read_text(encoding="utf-8"))
    assert summary["bundle_adjustment_completed"] is False
    assert summary["metric_scale_verified"] is False
    assert summary["unscaled_3d_coordinates_not_metric"] is True
    assert len(summary["camera_frame_ids"])==4
    package=out/"rgb-global-camera-network-unscaled.npz"
    with np.load(package,allow_pickle=False) as arrays:
        assert arrays["poses_world_to_camera_arbitrary"].shape==(4,3,4)
        assert arrays["landmarks_xyz_arbitrary"].shape[1]==3
        assert arrays["observation_pixels_xy"].shape[1]==2
        assert len(arrays["observation_landmark_indices"])==len(arrays["observation_frame_indices"])
    assert "orthophoto_verified" in capsys.readouterr().out


def test_one_to_many_orb_feature_indices_are_rejected_before_essential_ransac():
    frames, links, k = synthetic_observations()
    bad_indices = links[0].feature_indices.copy()
    # Each pair row is different, but two rows claim the same keypoint
    # in the first frame. This cannot be a one-to-one ORB cross-check.
    bad_indices[1, 0] = bad_indices[0, 0]
    assert len(set(map(tuple, bad_indices.tolist()))) == len(bad_indices)
    bad_link = RGBNetworkLink(
        links[0].first_id, links[0].second_id, bad_indices
    )
    with pytest.raises(ValueError, match="one-to-many"):
        solve_global_rgb_camera_network(
            frames, (bad_link,) + links[1:], k,
        )


def test_geometrically_marked_cycles_cannot_silently_discard_contradictory_tracks(
    monkeypatch,
):
    frames, links, k = synthetic_observations()
    corrupt = []
    for link in links:
        indices = link.feature_indices.copy()
        if (link.first_id, link.second_id) == ("frame-0", "frame-2"):
            # Invalid cycle mapping: the same 90 features appear in different
            # camera-2 tracks compared with camera-1/2 correspondences.
            indices[:, 1] = np.roll(indices[:, 1], 45)
        corrupt.append(RGBNetworkLink(
            link.first_id, link.second_id, indices,
        ))
    # Force artificial geometric acceptance to isolate the UNION/TRACK gate;
    # this is NOT proof that these index mappings satisfy a real E-matrix.
    monkeypatch.setattr(cv2, "findEssentialMat", lambda pi, pj, *args, **kwargs: (
        np.eye(3), np.ones((len(pi), 1), dtype=np.uint8),
    ))
    monkeypatch.setattr(cv2, "recoverPose", lambda e, pi, pj, *args, **kwargs: (
        len(pi), np.eye(3), np.array([[1.], [0.], [0.]]),
        np.ones((len(pi), 1), dtype=np.uint8),
    ))
    with pytest.raises(ValueError, match="contradictory multiview"):
        solve_global_rgb_camera_network(frames, tuple(corrupt), k)


def test_global_triplet_bootstrap_can_pose_camera_zero_after_other_cameras():
    """Actual projection/Essential/PnP with changing visibility, no pose doubles.

    The strongest edge touching camera zero sees only two-view points. A
    different three-camera seed reconstructs points that camera zero can use
    later; input order must not make a connected valid reconstruction fail.
    """
    rng = np.random.default_rng(20261030)
    sizes = (80, 30, 160)
    xyz = np.column_stack((
        rng.uniform(-2., 2., sum(sizes)),
        rng.uniform(-1.5, 1.5, sum(sizes)),
        rng.uniform(6., 12., sum(sizes)),
    ))
    a = set(range(80))
    b = set(range(80, 110))
    c = set(range(110, 270))
    visible = (a | b, a | c, b | c, b | c)
    k = PinholeIntrinsics(740., 740., 400., 300.)
    centers = np.array(((0., 0., 0.), (.6, -.1, .05),
                        (1.2, .1, -.05), (1.8, -.1, .1)))
    frames = []
    local_indices = []
    for i, world_ids in enumerate(visible):
        ordered = sorted(world_ids)
        local_indices.append({world: local for local, world in enumerate(ordered)})
        camera = xyz[ordered] - centers[i]
        pixels = np.column_stack((k.fx*camera[:, 0]/camera[:, 2]+k.cx,
                                  k.fy*camera[:, 1]/camera[:, 2]+k.cy))
        frames.append(RGBNetworkFrame(f"visibility-{i}", pixels))
    links = []
    for i in range(4):
        for j in range(i+1, 4):
            shared = sorted(visible[i] & visible[j])
            links.append(RGBNetworkLink(
                frames[i].frame_id, frames[j].frame_id,
                np.array([(local_indices[i][world], local_indices[j][world])
                          for world in shared], dtype=np.int32),
            ))
    original_pixels = tuple(frame.points_xy.copy() for frame in frames)
    ticks = []
    cv2.setRNGSeed(20261030)
    result = solve_global_rgb_camera_network(
        tuple(frames), tuple(links), k, on_progress=lambda *tick: ticks.append(tick),
    )
    assert result.frame_ids == tuple(frame.frame_id for frame in frames)
    assert len(result.world_to_camera) == 4
    assert result.reconstructed_landmarks >= 200
    np.testing.assert_array_equal(
        result.world_to_camera[0], np.column_stack((np.eye(3), np.zeros(3))),
    )
    # The actual initial accepted triplet lacks camera zero: only the 30
    # group-B points can subsequently pose it, despite the 80-point dead end.
    assert any(tick[0] == "bootstrap" for tick in ticks)
    assert [tick for tick in ticks if tick[0] == "poses"][0][1] == 3
    fid, lid = result.observation_frame_indices, result.observation_landmark_indices
    poses = np.asarray(result.world_to_camera)
    camera = np.einsum("nij,nj->ni", poses[fid, :, :3],
                       result.landmark_xyz_unit_baseline[lid]) + poses[fid, :, 3]
    assert np.all(camera[:, 2] > 0.)
    projected = np.column_stack((k.fx*camera[:, 0]/camera[:, 2]+k.cx,
                                 k.fy*camera[:, 1]/camera[:, 2]+k.cy))
    error = np.linalg.norm(projected-result.observation_pixels_xy, axis=1)
    assert error.max() < 3.
    assert set(fid) == {0, 1, 2, 3}
    for frame, before in zip(frames, original_pixels):
        np.testing.assert_array_equal(frame.points_xy, before)
    assert not result.metric_scale_verified and not result.orthophoto_verified


def test_camera_zero_reframe_preserves_all_camera_space_points_and_projections():
    from mcm_solarcheck.services.rgb_global_camera_network import _reframe_camera_zero

    rotations = ((.1, -.2, .05), (-.05, .12, -.1), (.03, -.04, .2))
    translations = ((.7, -.2, .1), (-.3, .15, .2), (.2, .3, -.1))
    poses = {i: np.column_stack((cv2.Rodrigues(np.array(r))[0], t))
             for i, (r, t) in enumerate(zip(rotations, translations))}
    points = {0: np.array((.2, -.1, 8.)), 1: np.array((-.3, .5, 10.))}
    before_poses = {i: value.copy() for i, value in poses.items()}
    before_points = {i: value.copy() for i, value in points.items()}
    reframed, transformed = _reframe_camera_zero(poses, points)
    np.testing.assert_array_equal(
        reframed[0], np.column_stack((np.eye(3), np.zeros(3))),
    )
    for i, pose in poses.items():
        for j, point in points.items():
            before = pose[:, :3] @ point + pose[:, 3]
            after = reframed[i][:, :3] @ transformed[j] + reframed[i][:, 3]
            np.testing.assert_allclose(after, before, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(after[:2]/after[2], before[:2]/before[2],
                                       rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(pose, before_poses[i])
    for i, point in points.items():
        np.testing.assert_array_equal(point, before_points[i])
