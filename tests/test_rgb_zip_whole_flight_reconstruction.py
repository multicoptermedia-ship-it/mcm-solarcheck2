"""Private RGB ZIP -> full 12-frame feature/RTK/BA handoff; synthetic only."""
from io import BytesIO
from dataclasses import replace
import json
import math
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
from PIL import Image

from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_global_camera_network import RGBUnscaledNetwork
from mcm_solarcheck.services.rgb_windowed_bundle_adjustment import RGBWindowedBAResult
from mcm_solarcheck.services import rgb_zip_whole_flight_reconstruction as process
from mcm_solarcheck.services.rgb_zip_whole_flight_cli import main


def _jpg(number, *, focal35=None, pixels=None):
    out = BytesIO()
    exif = Image.Exif()
    if focal35 is not None:
        exif[271] = "DJI"
        exif[272] = "M3T"
        exif[34665] = {41989: focal35}
    image = (Image.new("RGB", (400, 300), (60, 80, 100))
             if pixels is None else Image.fromarray(pixels))
    image.save(
        out, format="JPEG", exif=exif,
    )
    xmp = (f'''<?xpacket begin=" "/><x:xmpmeta xmlns:x="adobe:ns:meta/"
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        drone-dji:GpsLatitude="{50+number*.00001:.8f}"
        drone-dji:GpsLongitude="{7+(number%4)*.00002:.8f}"
        drone-dji:AbsoluteAltitude="+111.22"
        drone-dji:GimbalYawDegree="2.0"
        drone-dji:GimbalPitchDegree="-90.0"
        drone-dji:GimbalRollDegree="0.0"
        drone-dji:FlightYawDegree="3.0"
        drone-dji:FlightPitchDegree="0.0"
        drone-dji:FlightRollDegree="0.0"
        drone-dji:GpsStatus="50" drone-dji:AltitudeType="RtkAlt"
        drone-dji:RtkStdLat="0.012" drone-dji:RtkStdLon="0.013"
        drone-dji:RtkStdHgt="0.023"
        drone-dji:DewarpData="0,0,0"/></x:xmpmeta>''').encode("utf-8")
    segment = bytes((0xff, 0xe1)) + (len(xmp)+2).to_bytes(2, "big") + xmp
    jpeg = out.getvalue()
    return jpeg[:2] + segment + jpeg[2:]


def _fixture(tmp_path, *, focal35=None, pixels=None):
    archives = [tmp_path / "RGB7.zip", tmp_path / "RGB2.tip"]
    for archive_index, archive in enumerate(archives):
        with ZipFile(archive, "w") as z:
            for n in range(1+6*archive_index, 7+6*archive_index):
                z.writestr(f"flight/DJI_20260825120000_{n:04d}_V.JPG",
                           _jpg(n, focal35=focal35, pixels=pixels))
    camera = {
        "schema": "mcm.rgb_camera_intrinsics_external.v1",
        "lens_calibration_verified": True,
        "calibration_provenance": "independent synthetic test calibration",
        "intrinsics": {"fx": 740, "fy": 740, "cx": 200, "cy": 150},
        "distortion_coefficients": [0, 0, 0, 0, 0],
    }
    return tuple(archives), camera


def _stub_pipeline(monkeypatch, traced, *, accepted_nonadjacent=True, refinement_log=None):
    from mcm_solarcheck.services.rgb_windowed_bundle_adjustment import RGBWindowedBAResult

    rng = np.random.default_rng(20261028)
    descriptors = rng.integers(0, 255, size=(32,32), dtype=np.uint8)
    class Point:
        def __init__(self, x, y):
            self.pt = (x, y)
    class ORB:
        def detectAndCompute(self, image, mask):
            traced.append(("decoded", image.shape))
            return ([Point(float(30 + j*8), float(40 + j*4))
                     for j in range(32)], descriptors.copy())
    monkeypatch.setattr(process.cv2, "ORB_create", lambda **k: ORB())

    def fake_global(frames, links, intrinsics, **kwargs):
        traced.append(("sfm", len(frames), len(links)))
        assert len(frames) == 12
        assert len(links) >= 11
        assert tuple(f.frame_id for f in frames) == tuple(
            f"V-{i:04d}" for i in range(1, 13)
        )
        if refinement_log is not None:
            refinement_log["sfm_intrinsics"] = intrinsics
            refinement_log["sfm_progress_forwarded"] = kwargs.get("on_progress")
        poses = tuple(
            np.column_stack((np.eye(3), np.array([-.1*i, 0., 0.])))
            for i in range(12)
        )
        return RGBUnscaledNetwork(
            tuple(f.frame_id for f in frames), poses,
            16, 16, len(links), .7, 1.1,
            landmark_xyz_unit_baseline=np.ones((16,3), np.float64),
            observation_frame_indices=np.repeat(np.arange(12, dtype=np.int32),16),
            observation_landmark_indices=np.tile(
                np.arange(16, dtype=np.int32), 12,
            ),
            observation_pixels_xy=np.ones((192,2), np.float64),
            accepted_link_pairs=tuple(
                (link.first_id, link.second_id) for link in links
                if accepted_nonadjacent or abs(
                    int(link.first_id[2:])-int(link.second_id[2:])
                ) == 1
            ),
        )
    monkeypatch.setattr(process, "solve_global_rgb_camera_network", fake_global)

    def fake_unscaled(network, intrinsics, **kwargs):
        traced.append(("windowed", len(network.frame_ids)))
        fitted = RGBWindowedBAResult(
            network.world_to_camera, network.landmark_xyz_unit_baseline,
            1.2, .7, 2., 1., 12, 16, 192, 3, 6,
        )
        if refinement_log is not None:
            refinement_log["unscaled_refine_focal"] = kwargs.get("refine_focal")
            refined = PinholeIntrinsics(intrinsics.fx * 1.02, intrinsics.fy * 1.02,
                                       intrinsics.cx, intrinsics.cy)
            fitted = replace(fitted, refined_intrinsics=refined, focal_refined=True)
        return fitted
    monkeypatch.setattr(
        process, "adjust_windowed_unscaled_rgb_network", fake_unscaled,
    )

    class MetricCandidate:
        total_cameras = 12
        refined_intrinsics = None
        focal_refined = False
        def privacy_safe_summary(self):
            return {
                "camera_count": 12, "rtk_holdout_p95_m": .08,
                "roof_geometry_verified": False,
                "orthophoto_verified": False,
            }
    def fake_rtk(network, unscaled, priors, intrinsics, **kwargs):
        traced.append(("rtk", len(priors.frame_ids)))
        assert priors.frame_ids == network.frame_ids
        assert priors.camera_enu_m.shape == (12, 3)
        assert np.all(priors.sigma_enu_m[:, :2] >= .10)
        result = MetricCandidate()
        if refinement_log is not None:
            refinement_log["rtk_refine_focal"] = kwargs.get("refine_focal")
            refinement_log["rtk_intrinsics"] = intrinsics
            assert intrinsics is unscaled.refined_intrinsics
            result.refined_intrinsics = PinholeIntrinsics(
                intrinsics.fx * .99, intrinsics.fy * .99,
                intrinsics.cx, intrinsics.cy,
            )
            result.focal_refined = True
            refinement_log["final_intrinsics"] = result.refined_intrinsics
        return result
    monkeypatch.setattr(
        process, "adjust_rgb_windowed_bundle_with_private_rtk", fake_rtk,
    )


def test_private_zip_rgb_pipeline_uses_all_frames_without_extracting(tmp_path, monkeypatch):
    archives, camera = _fixture(tmp_path)
    before = [a.read_bytes() for a in archives]
    traced = []
    _stub_pipeline(monkeypatch, traced)
    result = process.reconstruct_private_complete_rgb_zip_flight(
        archives, camera, max_side=400, max_features=500, feature_method="orb",
    )
    report = result.privacy_safe_summary()
    assert report["frame_count"] == 12
    assert report["feature_method"] == "orb" and report["descriptor_metric"] == "hamming"
    assert report["feature_matched_pairs"] >= 11
    assert report["cross_strip_matched_pairs"] > 0
    assert report["cross_strip_matches_are_descriptor_only"] is True
    assert report["sfm_accepted_spatial_nonadjacent_pairs"] >= 2
    assert report["sfm_geometric_nonadjacent_supported_cameras"] >= 3
    assert report["sfm_graph_connected"] is True
    assert report["sfm_3d_track_graph_connected"] is True
    assert report["sfm_graph_min_camera_degree"] >= 1
    assert report["physical_flight_strip_identity_verified"] is False
    assert report["physical_cross_strip_identity_verified"] is False
    assert report["all_input_rgb_frames_processed"] is True
    assert report["rgb_orthophoto_verified"] is False
    assert len([x for x in traced if x[0] == "decoded"]) == 12
    assert traced[-3:] == [
        ("sfm", 12, result.feature_matched_pair_count),
        ("windowed", 12), ("rtk", 12),
    ]
    output = json.dumps(report)
    for private in ("50.00001", "7.00002", "DJI_", "RGB7.zip",
                    "flight/", "RtkAlt", "private/"):
        assert private not in output
    assert [a.read_bytes() for a in archives] == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["RGB2.tip", "RGB7.zip"]


def test_cli_writes_only_aggregate_and_never_changes_private_jpegs(tmp_path, monkeypatch, capsys):
    archives, camera = _fixture(tmp_path)
    original = [a.read_bytes() for a in archives]
    traced = []
    _stub_pipeline(monkeypatch, traced)
    calibration = tmp_path / "lens.json"
    calibration.write_text(json.dumps(camera), encoding="utf-8")
    report_dir = tmp_path / "local_output"
    report_dir.mkdir()
    report = report_dir / "flight-summary.json"
    args = [
        "--rgb-archive", str(archives[0]),
        "--rgb-archive", str(archives[1]),
        "--calibration-json", str(calibration),
        "--report", str(report), "--max-side", "400",
        "--max-features", "500",
        "--feature-method", "orb",
    ]
    assert main(args) == 0
    output = report.read_text(encoding="utf-8")
    assert json.loads(output)["frame_count"] == 12
    assert "RtkAlt" not in output
    assert "50.00001" not in output
    assert "7.00002" not in output
    assert "flight/" not in capsys.readouterr().out
    assert [a.read_bytes() for a in archives] == original
    with pytest.raises(ValueError, match="new local report"):
        main(args)


def test_explicit_unverified_calibration_blocks_real_rgb_sfm(tmp_path, capsys, monkeypatch):
    archives, camera = _fixture(tmp_path)
    camera["lens_calibration_verified"] = False
    evidence = tmp_path / "invalid.json"
    evidence.write_text(json.dumps(camera), encoding="utf-8")
    def unexpected_automatic_fallback(*_args, **_kwargs):
        pytest.fail("an invalid explicit calibration must not fall back to EXIF estimates")
    monkeypatch.setattr(process, "initialize_private_rgb_camera", unexpected_automatic_fallback)
    status = main([
        "--rgb-archive", str(archives[0]),
        "--rgb-archive", str(archives[1]),
        "--calibration-json", str(evidence),
    ])
    assert status == 2
    result = json.loads(capsys.readouterr().out)
    assert result["failed_stage"] == "calibration"
    assert result["rgb_orthophoto_verified"] is False


def test_real_exif_initializes_same_whole_flight_core_and_propagates_refined_camera(tmp_path, monkeypatch):
    archives, _camera = _fixture(tmp_path, focal35=24)
    originals = [x.read_bytes() for x in archives]
    traced, refinement_log, progress = [], {}, []
    _stub_pipeline(monkeypatch, traced, refinement_log=refinement_log)
    callback = lambda *event: progress.append(event)
    result = process.reconstruct_private_complete_rgb_zip_flight(
        archives, max_side=400, max_features=500, on_progress=callback,
        feature_method="orb",
    )
    initial_focal = 24 * math.hypot(400, 300) / math.hypot(36, 24)
    assert refinement_log["sfm_intrinsics"].fx == pytest.approx(initial_focal)
    assert refinement_log["unscaled_refine_focal"] is True
    assert refinement_log["rtk_refine_focal"] is True
    assert refinement_log["rtk_intrinsics"].fx == pytest.approx(initial_focal * 1.02)
    assert refinement_log["sfm_progress_forwarded"] is callback
    assert result.private_intrinsics is refinement_log["final_intrinsics"]
    assert result.private_intrinsics.fx == pytest.approx(initial_focal * 1.02 * .99)
    assert len(result.private_series.frames) == 12
    assert len(result.private_network.frame_ids) == 12
    assert np.array_equal(result.private_distortion, np.zeros(5))
    report = result.privacy_safe_summary()
    assert report["frame_count"] == 12 and report["all_input_rgb_frames_processed"] is True
    assert report["camera_initialization"]["source"] == "exif_35mm_equivalent"
    assert report["lens_calibration_verified"] is False
    assert report["focal_length_refined_from_multiview"] is True
    assert report["full_camera_self_calibration_completed"] is False
    assert report["rgb_orthophoto_verified"] is False
    for private in ("50.00001", "7.00002", "DJI_", "V-0001", "flight/", "RGB7.zip"):
        assert private not in json.dumps(report)
    assert any(x[0] == "camera_initialization" for x in progress)
    assert any(x[0] == "rtk_weighted_ba" and x[1] == 1 for x in progress)
    assert [x.read_bytes() for x in archives] == originals


def test_cli_accepts_archives_without_external_calibration_file(tmp_path, monkeypatch, capsys):
    archives, _camera = _fixture(tmp_path, focal35=24)
    refinement_log = {}
    _stub_pipeline(monkeypatch, [], refinement_log=refinement_log)
    outdir = tmp_path / "private_output"
    outdir.mkdir()
    report = outdir / "without-calibration.json"
    assert main([
        "--rgb-archive", str(archives[0]), "--rgb-archive", str(archives[1]),
        "--report", str(report), "--max-side", "400", "--max-features", "500",
        "--feature-method", "orb",
    ]) == 0
    data = json.loads(report.read_text())
    assert data["lens_calibration_verified"] is False
    assert data["frame_count"] == 12
    assert data["focal_length_refined_from_multiview"] is True
    assert "private_output" not in capsys.readouterr().out


def test_cli_surface_request_uses_final_camera_and_keeps_failed_geometry_private(tmp_path, monkeypatch, capsys):
    from mcm_solarcheck.services import rgb_zip_whole_flight_cli as cli
    archives, _camera = _fixture(tmp_path, focal35=24)
    log = {}
    _stub_pipeline(monkeypatch, [], refinement_log=log)
    output = tmp_path.parent / (tmp_path.name + "-candidate")
    def rejected_surface(outcome, directory, **kwargs):
        assert outcome.private_intrinsics is log["final_intrinsics"]
        assert directory == output
        raise ValueError("private coordinate or original filename must stay private")
    monkeypatch.setattr(cli, "write_private_rgb_flight_surface_candidate", rejected_surface)
    status = cli.main([
        "--rgb-archive", str(archives[0]), "--rgb-archive", str(archives[1]),
        "--candidate-output-dir", str(output),
        "--max-side", "400", "--max-features", "500",
        "--feature-method", "orb",
    ])
    stdout = capsys.readouterr().out
    data = json.loads(stdout)
    assert status == 2 and data["failed_stage"] == "surface_candidate"
    assert data["all_input_rgb_frames_processed"] is True
    assert data["rgb_orthophoto_verified"] is False
    assert "private coordinate" not in stdout and str(output) not in stdout
    assert not output.exists()


def test_absent_camera_metadata_has_safe_stage_and_private_diagnostics(tmp_path, capsys):
    archives, _camera = _fixture(tmp_path)
    with pytest.raises(process.PrivateRGBFlightStageError) as error:
        process.reconstruct_private_complete_rgb_zip_flight(archives)
    assert error.value.stage == "camera_initialization"
    assert error.value.private_cause
    assert len(error.value.private_context["series"].frames) == 12
    assert "flight/" not in str(error.value)
    status = main(["--rgb-archive", str(archives[0]), "--rgb-archive", str(archives[1])])
    assert status == 2
    output = capsys.readouterr().out
    data = json.loads(output)
    assert data["failed_stage"] == "camera_initialization"
    assert data["all_input_rgb_frames_processed"] is False
    assert "private_context" not in output and "flight/" not in output


@pytest.mark.parametrize("feature_method", ["orb", "sift"])
def test_automatic_camera_keeps_real_blank_image_feature_gate(tmp_path, feature_method):
    archives, _camera = _fixture(tmp_path, focal35=24)
    originals = [archive.read_bytes() for archive in archives]
    with pytest.raises(process.PrivateRGBFlightStageError) as error:
        process.reconstruct_private_complete_rgb_zip_flight(
            archives, max_side=400, max_features=500, feature_method=feature_method,
        )
    assert error.value.stage == "feature_extraction"
    assert error.value.private_context["feature_method"] == feature_method
    assert error.value.private_context["camera_initialization_summary"]["lens_calibration_verified"] is False
    assert [archive.read_bytes() for archive in archives] == originals
    assert sorted(x.name for x in tmp_path.iterdir()) == ["RGB2.tip", "RGB7.zip"]


@pytest.mark.parametrize("feature_method", ["orb", "sift"])
def test_blank_real_jpeg_fails_closed_without_any_partial_report(tmp_path, feature_method):
    archives, camera = _fixture(tmp_path)
    with pytest.raises(process.PrivateRGBFlightStageError) as info:
        process.reconstruct_private_complete_rgb_zip_flight(
            archives, camera, max_side=400, max_features=500, feature_method=feature_method,
        )
    assert info.value.stage == "feature_extraction"
    assert len(list(tmp_path.iterdir())) == 2


def test_nonadjacent_rtk_cross_strip_candidates_are_proposed():
    ids = tuple(f"V-{i:04d}" for i in range(1, 13))
    xyz = np.array([[i*.6, (i%4)*.4, 0.] for i in range(12)])
    pairs = process._private_rgb_overlap_pairs(
        ids, xyz, max_neighbours=5,
    )
    assert len([p for p in pairs if p.reason == "consecutive"]) == 11
    assert len([p for p in pairs if p.reason == "cross_strip"]) >= 5
    assert len({frozenset((p.first_id, p.second_id)) for p in pairs}) == len(pairs)


def test_mutual_ratio_matching_rejects_repeated_descriptors_in_either_image():
    rng = np.random.default_rng(4031)
    unique = rng.integers(0, 256, (24, 32), dtype=np.uint8)
    first, second = unique.copy(), unique[::-1].copy()
    # A perfect match is nevertheless ambiguous if a second cell is identical.
    second = np.vstack((second, second[3:4]))
    first = np.vstack((first, first[5:6]))
    matches = process._match_distinct_rgb_descriptors(first, second)
    assert len(matches) == 22
    assert set(matches[:, 0]) == set(range(24)) - {5, 20}
    assert all(np.array_equal(first[a], second[b]) for a, b in matches)
    assert len(set(matches[:, 0])) == len(set(matches[:, 1])) == 22


def test_sift_l2_matching_rejects_duplicate_alternatives_in_both_images():
    rng = np.random.default_rng(197)
    unique = rng.uniform(0., 220., (24, 128)).astype(np.float32)
    first = np.vstack((unique, unique[5:6]))
    reversed_unique = unique[::-1].copy()
    second = np.vstack((reversed_unique, reversed_unique[3:4]))
    matches = process._match_distinct_rgb_descriptors(first, second)
    assert len(matches) == 22
    assert set(matches[:, 0]) == set(range(24)) - {5, 20}
    assert all(np.array_equal(first[a], second[b]) for a, b in matches)

    # A good SIFT match may have L2 distance >64, since 64 only applies to
    # Hamming bits. Each alternative here is much farther away than its mate.
    first = unique.copy()
    second = unique[::-1].copy() + np.float32(10.)
    matches = process._match_distinct_rgb_descriptors(first, second)
    assert len(matches) == 24
    assert all(b == 23-a for a, b in matches)
    assert np.linalg.norm(first[0]-second[-1]) > 64.
    invalid = first.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite SIFT"):
        process._match_distinct_rgb_descriptors(invalid, second)
    with pytest.raises(ValueError, match="finite SIFT"):
        process._match_distinct_rgb_descriptors(first, np.zeros((24, 32), np.uint8))


def _textured_scene():
    rng = np.random.default_rng(37013)
    grey = rng.integers(30, 225, (300, 400), dtype=np.uint8)
    grey = cv2.GaussianBlur(grey, (3, 3), .8)
    for _ in range(95):
        x, y = int(rng.integers(12, 388)), int(rng.integers(12, 288))
        cv2.circle(grey, (x, y), int(rng.integers(2, 7)),
                   int(rng.choice((10, 245))), -1, cv2.LINE_AA)
    return np.repeat(grey[:, :, None], 3, axis=2)


def test_real_sift_tracks_rotated_cropped_zip_images_in_original_sensor_pixels(tmp_path):
    first = _textured_scene()
    affine = cv2.getRotationMatrix2D((200., 150.), 8., .92)
    affine[:, 2] += (9., -6.)
    second = cv2.warpAffine(first, affine, (400, 300), borderMode=cv2.BORDER_CONSTANT)
    archive = tmp_path / "synthetic-views.zip"
    with ZipFile(archive, "w") as z:
        for number, pixels in enumerate((first, second), 1):
            z.writestr(f"DJI_20260825120000_{number:04d}_V.JPG", _jpg(number, pixels=pixels))
    original = archive.read_bytes()
    series = process.read_private_rgb_zip_series((archive,))
    views, descriptors = process._features_from_private_zip(
        series, PinholeIntrinsics(740., 740., 200., 150.), np.zeros(5),
        feature_method="sift", max_side=300, max_features=700,
        should_cancel=None, on_progress=None,
    )
    assert all(desc.dtype == np.float32 and desc.shape[1] == 128 for desc in descriptors)
    matched = process._match_distinct_rgb_descriptors(*descriptors)
    assert len(matched) >= 45
    original_xy = views[0].points_xy[matched[:, 0]]
    transformed_xy = original_xy @ affine[:, :2].T + affine[:, 2]
    errors = np.linalg.norm(transformed_xy - views[1].points_xy[matched[:, 1]], axis=1)
    assert np.median(errors) < .5
    assert np.mean(errors < 2.) > .95
    assert archive.read_bytes() == original
    assert list(tmp_path.iterdir()) == [archive]


def test_default_real_sift_uses_existing_whole_flight_core_and_reports_provenance(tmp_path, monkeypatch):
    archives, _ = _fixture(tmp_path, focal35=24, pixels=_textured_scene())
    before = [path.read_bytes() for path in archives]
    trace = []
    _stub_pipeline(monkeypatch, trace)
    outcome = process.reconstruct_private_complete_rgb_zip_flight(
        archives, max_side=400, max_features=500,
    )
    report = outcome.privacy_safe_summary()
    assert report["feature_method"] == "sift"
    assert report["descriptor_metric"] == "l2"
    assert report["frame_count"] == 12
    assert report["all_input_rgb_frames_processed"] is True
    assert report["rgb_orthophoto_verified"] is False
    assert not any(stage[0] == "decoded" for stage in trace)  # ORB stub unused.
    assert [stage[0] for stage in trace] == ["sfm", "windowed", "rtk"]
    assert [path.read_bytes() for path in archives] == before


def test_unsupported_feature_method_fails_with_safe_stage_and_private_context(tmp_path):
    archives, _ = _fixture(tmp_path, focal35=24)
    with pytest.raises(process.PrivateRGBFlightStageError) as error:
        process.reconstruct_private_complete_rgb_zip_flight(archives, feature_method="invalid")
    assert error.value.stage == "feature_extraction"
    assert error.value.private_context["feature_method"] == "invalid"
    assert "DJI_" not in str(error.value)


def test_descriptor_only_spatial_links_are_not_geometry_evidence(tmp_path, monkeypatch):
    archives, camera = _fixture(tmp_path)
    seen = []
    # BFMatcher matches all of the synthetic descriptor arrays, but the fake
    # SfM solver accepts ONLY sequential pairs geometrically. The complete
    # flight gate must FAIL CLOSED before any RTK BA.
    _stub_pipeline(monkeypatch, seen, accepted_nonadjacent=False)
    with pytest.raises(process.PrivateRGBFlightStageError) as error:
        process.reconstruct_private_complete_rgb_zip_flight(
            archives, camera, max_side=400, max_features=500, feature_method="orb",
        )
    assert error.value.stage == "global_sfm"
    assert any(stage[0] == "sfm" for stage in seen)
    assert not any(stage[0] == "rtk" for stage in seen)
    assert "flight/" not in str(error.value)


def test_geometry_only_spatial_links_cannot_pass_without_shared_3d_points(
    tmp_path, monkeypatch,
):
    archives, camera = _fixture(tmp_path)
    seen = []
    _stub_pipeline(monkeypatch, seen)
    original_solver = process.solve_global_rgb_camera_network

    def geometrically_accepted_but_disconnected_tracks(*args, **kwargs):
        network = original_solver(*args, **kwargs)
        # Each adjacent camera pair sees 8 genuine common 3D points.
        # Their landmark IDs are private, unique to that adjacent pair:
        # only sequential links have shared 3D support. All essential
        # spatial links still claim geometry acceptance in this stub.
        fid, lid = [], []
        for i in range(11):
            for p in range(8):
                for camera_id in (i, i+1):
                    fid.append(camera_id)
                    lid.append(8*i+p)
        return replace(
            network,
            reconstructed_landmarks=88,
            landmark_xyz_unit_baseline=np.ones((88, 3), dtype=np.float64),
            observation_frame_indices=np.asarray(fid, dtype=np.int32),
            observation_landmark_indices=np.asarray(lid, dtype=np.int32),
            observation_pixels_xy=np.ones((len(fid), 2), dtype=np.float64),
        )

    monkeypatch.setattr(
        process, "solve_global_rgb_camera_network",
        geometrically_accepted_but_disconnected_tracks,
    )
    with pytest.raises(process.PrivateRGBFlightStageError) as error:
        process.reconstruct_private_complete_rgb_zip_flight(
            archives, camera, max_side=400, max_features=500, feature_method="orb",
        )
    assert error.value.stage == "global_sfm"
    assert any(stage[0] == "sfm" for stage in seen)
    assert not any(stage[0] in ("windowed", "rtk") for stage in seen)
    assert "V-0001" not in str(error.value)
