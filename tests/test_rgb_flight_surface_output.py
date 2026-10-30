"""Private candidate artifacts from actual synthetic ZIP JPEGs and 3D planes."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from mcm_solarcheck.domain.models import Pose, RTKQuality
from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_flight_surface_output import (
    validate_private_rgb_surface_output_directory,
    write_private_rgb_flight_surface_candidate,
)
from mcm_solarcheck.services.rgb_flight_zip_series import (
    PrivateRGBFlightFrame, PrivateRGBFlightSeries,
)
from mcm_solarcheck.services.rgb_global_camera_network import RGBUnscaledNetwork
from mcm_solarcheck.services.rgb_rtk_joint_bundle_adjustment import PrivateRGBRTKWindowedBA
from mcm_solarcheck.services.rgb_zip_whole_flight_reconstruction import (
    PrivateRGBFlightReconstruction,
)


def _scene(tmp_path):
    inputs, outputs = tmp_path / "originals", tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    archive = inputs / "private-originals.zip"
    x, y = np.meshgrid(np.linspace(-4., -2., 9), np.linspace(-1., 1., 7))
    left = np.column_stack((x.ravel(), y.ravel(), (.3*x).ravel()))
    x, y = np.meshgrid(np.linspace(2., 4., 9), np.linspace(-1., 1., 7))
    right = np.column_stack((x.ravel(), y.ravel(), (2.-.25*y).ravel()))
    points = np.vstack((left, right))
    frames, poses, observed = [], [], []
    ids = tuple(f"V-{i+1:04d}" for i in range(3))
    camera = PinholeIntrinsics(200., 200., 200., 150.)
    yy, xx = np.indices((300, 400))
    image = np.stack((xx//2, yy//2, np.full_like(xx, 150)), axis=2).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 97])
    assert ok
    payload = jpeg.tobytes()
    with ZipFile(archive, "w") as z:
        for i, frame_id in enumerate(ids):
            member = f"flight/DJI_20261030120000_{i+1:04d}_V.JPG"
            z.writestr(member, payload)
            frames.append(PrivateRGBFlightFrame(
                archive, member, frame_id, "20261030120000", 400, 300,
                None, Pose(), Pose(), RTKQuality(), (), len(payload),
            ))
            rotation = np.diag((1., -1., -1.))
            center = np.array((-.5+.5*i, 0., 8.))
            poses.append(np.column_stack((rotation, -rotation @ center)))
            local = (points-center) @ rotation.T
            observed.append(np.column_stack((
                200.*local[:, 0]/local[:, 2]+200.,
                200.*local[:, 1]/local[:, 2]+150.,
            )))
    fid = np.repeat(np.arange(3, dtype=np.int32), len(points))
    lid = np.tile(np.arange(len(points), dtype=np.int32), 3)
    network = RGBUnscaledNetwork(
        ids, tuple(poses), len(points), len(points), 3, .1, .2,
        landmark_xyz_unit_baseline=points.copy(),
        observation_frame_indices=fid, observation_landmark_indices=lid,
        observation_pixels_xy=np.concatenate(observed),
    )
    ba = PrivateRGBRTKWindowedBA(
        ids, tuple(poses), points.copy(), .2, .1, .2, .1, .2, 2, 1, 3, 1, 2., 1.,
        refined_intrinsics=camera,
    )
    reconstruction = PrivateRGBFlightReconstruction(
        3, 3, 3, 2, 2, len(points), .2, .1, ba, None,
        private_series=PrivateRGBFlightSeries(tuple(frames), 1, 0),
        private_network=network, private_intrinsics=camera,
        private_distortion=np.zeros(5),
    )
    return reconstruction, archive, outputs


def test_real_zip_pixels_form_private_bgra_candidate_and_gap_stays_nodata(tmp_path):
    reconstruction, archive, outputs = _scene(tmp_path)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = outputs / "candidate"
    progress = []
    report = write_private_rgb_flight_surface_candidate(
        reconstruction, destination, metres_per_pixel=.05, max_side=256,
        on_progress=lambda *step: progress.append(step),
    )
    assert set(p.name for p in destination.iterdir()) == {
        "rgb-orthophoto-candidate.png", "valid-mask.png", "summary.json",
    }
    image = cv2.imread(str(destination / "rgb-orthophoto-candidate.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(destination / "valid-mask.png"), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (40, 160, 4)
    np.testing.assert_array_equal(image[:, :, 3], mask)
    assert np.count_nonzero(mask) > 1000
    assert np.all(mask[:, 60:100] == 0)
    assert np.all(image[mask == 0, 3] == 0)
    # One point on the left tilted plane; independent camera projection and
    # the encoded gradient establish that source pixels reached the PNG.
    row, col = 20, 20
    assert mask[row, col] == 255
    x, y = -4.+(col+.5)*.05, 1.-(row+.5)*.05
    z = .3*x
    # Nearest source is the overhead camera at x=-.5 for this left patch.
    u, v = 200.*(x+.5)/(8.-z)+200., -200.*y/(8.-z)+150.
    np.testing.assert_allclose(image[row, col, :3], (u/2., v/2., 150.), atol=2.)
    assert json.loads((destination / "summary.json").read_text()) == report
    assert report["candidate_created"] is True
    assert report["input_frame_count"] == 3
    assert report["output_artifact_count"] == 3
    assert report["camera_count"] == 3
    assert report["assumes_zero_distortion"] is True
    assert all(value is False for key, value in report.items() if key.endswith("verified"))
    public = json.dumps(report)
    for forbidden in (str(tmp_path), "V-0001", "DJI_", "origin_xy", "landmark_indices"):
        assert forbidden not in public
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before
    assert progress[0] == ("surface_fit", 0, 1)
    assert progress[-1] == ("surface_output", 3, 3)
    if os.name != "nt":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in destination.iterdir())


def test_output_validation_is_read_only_and_protects_originals_and_checkouts(tmp_path):
    _, archive, outputs = _scene(tmp_path)
    valid = outputs / "new"
    assert validate_private_rgb_surface_output_directory(valid, (archive,)) == valid
    assert not valid.exists()
    with pytest.raises(ValueError, match="absolute"):
        validate_private_rgb_surface_output_directory(Path("relative-output"), (archive,))
    with pytest.raises(ValueError, match="archive folders"):
        validate_private_rgb_surface_output_directory(archive.parent / "candidate", (archive,))
    with pytest.raises(ValueError, match="already exist"):
        validate_private_rgb_surface_output_directory(outputs, (archive,))
    with pytest.raises(ValueError, match="existing.*parent"):
        validate_private_rgb_surface_output_directory(outputs / "missing" / "candidate", (archive,))
    with pytest.raises(ValueError, match="sequence"):
        validate_private_rgb_surface_output_directory(valid, ())
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /tmp/private-test-metadata\n")
    with pytest.raises(ValueError, match="repositories"):
        validate_private_rgb_surface_output_directory(checkout / "candidate", (archive,))
    assert list(outputs.iterdir()) == []


def test_symlink_ancestor_and_lexical_parent_traversal_are_rejected(tmp_path):
    _, archive, outputs = _scene(tmp_path)
    link = tmp_path / "output-link"
    try:
        link.symlink_to(outputs, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        validate_private_rgb_surface_output_directory(link / "candidate", (archive,))
    with pytest.raises(ValueError, match="absolute"):
        validate_private_rgb_surface_output_directory(outputs / ".." / "candidate", (archive,))
    assert list(outputs.iterdir()) == []


def test_invalid_geometry_cancellation_and_existing_destination_never_write(tmp_path):
    reconstruction, archive, outputs = _scene(tmp_path)
    bad_network = replace(reconstruction.private_network,
                          observation_frame_indices=np.zeros(378, dtype=np.int32))
    with pytest.raises(ValueError, match="nondegenerate"):
        write_private_rgb_flight_surface_candidate(
            replace(reconstruction, private_network=bad_network), outputs / "missing-views",
            max_side=256,
        )
    with pytest.raises(RuntimeError, match="cancelled"):
        write_private_rgb_flight_surface_candidate(
            reconstruction, outputs / "cancelled", should_cancel=lambda: True,
        )
    existing = outputs / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve user data")
    with pytest.raises(ValueError, match="already exist"):
        write_private_rgb_flight_surface_candidate(reconstruction, existing)
    assert sentinel.read_text() == "preserve user data"
    assert set(p.name for p in outputs.iterdir()) == {"existing"}
    assert archive.is_file()


def test_changed_original_member_is_rejected_before_creating_output(tmp_path):
    reconstruction, archive, outputs = _scene(tmp_path)
    # Simulate changed archive input after reconstruction; never render a
    # different-sized replacement as though it were the validated source.
    frame = reconstruction.private_series.frames[0]
    with ZipFile(archive, "r") as z:
        members = {name: z.read(name) for name in z.namelist()}
    members[frame.member_name] += b"changed"
    with ZipFile(archive, "w") as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    with pytest.raises(ValueError, match="member changed"):
        write_private_rgb_flight_surface_candidate(
            reconstruction, outputs / "changed", max_side=256,
        )
    assert list(outputs.iterdir()) == []
