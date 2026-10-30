"""Real JPEG/EXIF -> in-place camera initialization; no estimated truth flags."""
from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from mcm_solarcheck.services.rgb_camera_initialization import initialize_private_rgb_camera
from mcm_solarcheck.services.rgb_flight_zip_series import read_private_rgb_zip_series


def _jpeg(*, focal=24, physical=4.4, model="M3T", size=(400, 300),
          zoom=1., orientation=1, serial="private-device-unit", flat_focal=False):
    exif = Image.Exif()
    exif[271], exif[272], exif[274] = "DJI", model, orientation
    exif[270] = "private-customer-name GPS=private-value"
    specific = {37386: physical, 41988: zoom, 40962: size[0],
                40963: size[1], 42033: serial}
    if focal is not None:
        if flat_focal:
            exif[41989] = focal
        else:
            specific[41989] = focal
    exif[34665] = specific
    output = BytesIO()
    Image.new("RGB", size, (40, 70, 100)).save(output, format="JPEG", exif=exif)
    jpeg = output.getvalue()
    xmp = (b'<?xpacket begin=" "/><x:xmpmeta xmlns:x="adobe:ns:meta/" '
           b'xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
           b'drone-dji:Make="DJI" drone-dji:Model="M3T"></x:xmpmeta>')
    segment = b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp
    return jpeg[:2] + segment + jpeg[2:]


def _series(tmp_path, options=({}, {})):
    archive = tmp_path / "private-originals.zip"
    with ZipFile(archive, "w") as output:
        for number, values in enumerate(options, 1):
            output.writestr(f"private/DJI_20250825121338_{number:04d}_V.JPG", _jpeg(**values))
    return read_private_rgb_zip_series((archive,)), archive


def test_exif_35mm_equivalence_uses_diagonal_and_original_pixel_coordinates(tmp_path):
    series, archive = _series(tmp_path)
    before = archive.read_bytes()
    model = initialize_private_rgb_camera(series)
    # Correct 4:3/35-mm diagonal equivalence; using 400*24/36 would yield 266.67.
    assert model.intrinsics.fx == pytest.approx(277.35009811261455)
    assert model.intrinsics.fy == model.intrinsics.fx
    assert (model.intrinsics.cx, model.intrinsics.cy) == (200., 150.)
    assert model.source == "exif_35mm_equivalent"
    assert np.array_equal(model.distortion, np.zeros(5))
    assert archive.read_bytes() == before
    assert not list(tmp_path.rglob("*.JPG"))


def test_flat_exif_focal_equivalence_is_supported(tmp_path):
    series, _ = _series(tmp_path, ({"flat_focal": True},))
    assert initialize_private_rgb_camera(series).intrinsics.fx == pytest.approx(277.35009811261455)


@pytest.mark.parametrize("focal", [None, 0, 65535])
def test_physical_millimetres_do_not_replace_missing_or_invalid_equivalent_focal(tmp_path, focal):
    series, _ = _series(tmp_path, ({"focal": focal, "physical": 4.4},))
    with pytest.raises(ValueError, match="35 mm-equivalent EXIF focal"):
        initialize_private_rgb_camera(series)


@pytest.mark.parametrize("last", [
    {"focal": 35}, {"physical": 8.8}, {"model": "other-private-camera"},
    {"serial": "different-private-unit"},
])
def test_all_images_must_share_camera_lens_and_focal_mode(tmp_path, last):
    series, _ = _series(tmp_path, ({}, {}, last))
    with pytest.raises(ValueError, match="shared RGB camera") as error:
        initialize_private_rgb_camera(series)
    assert "private" not in str(error.value)


def test_missing_focal_in_late_frame_is_not_hidden_by_first_frame(tmp_path):
    series, _ = _series(tmp_path, ({}, {}, {"focal": None}))
    with pytest.raises(ValueError, match="35 mm-equivalent"):
        initialize_private_rgb_camera(series)


@pytest.mark.parametrize("setting", [{"orientation": 6}, {"zoom": 2.}])
def test_unsupported_rotated_or_cropped_image_modes_fail(tmp_path, setting):
    series, _ = _series(tmp_path, (setting,))
    with pytest.raises(ValueError, match="original RGB"):
        initialize_private_rgb_camera(series)


def test_image_dimensions_are_consistent_over_complete_series(tmp_path):
    series, _ = _series(tmp_path, ({}, {"size": (800, 600)}))
    with pytest.raises(ValueError, match="consistent original RGB image mode"):
        initialize_private_rgb_camera(series)


def test_stale_member_size_after_intake_is_rejected(tmp_path):
    series, _ = _series(tmp_path)
    changed = replace(series.frames[0], image_size_bytes=series.frames[0].image_size_bytes + 1)
    series = replace(series, frames=(changed,) + series.frames[1:])
    with pytest.raises(ValueError, match="unchanged regular original RGB"):
        initialize_private_rgb_camera(series)


def test_summary_is_explicitly_nominal_and_contains_no_private_identifiers(tmp_path):
    series, _ = _series(tmp_path)
    summary = initialize_private_rgb_camera(series).privacy_safe_summary()
    assert summary["lens_calibration_verified"] is False
    assert summary["intrinsics_optimized"] is False
    assert summary["rgb_orthophoto_verified"] is False
    assert "zero_distortion_start_only" in summary["assumptions"]
    assert summary["camera_count"] == 2
    encoded = json.dumps(summary)
    for private in ("private", "DJI_", "M3T", "20250825", "GPS"):
        assert private not in encoded


def test_metadata_header_budget_is_bounded(tmp_path):
    original = _jpeg()
    segment = b"\xff\xee" + (65535).to_bytes(2, "big") + b"x" * 65533
    hostile = original[:2] + segment * 17 + original[2:]
    archive = tmp_path / "bounded-metadata.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("DJI_20250825121338_0001_V.JPG", hostile)
    series = read_private_rgb_zip_series((archive,))
    with pytest.raises(ValueError, match="bounded original RGB JPEG metadata"):
        initialize_private_rgb_camera(series)
