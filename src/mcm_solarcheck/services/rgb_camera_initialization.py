"""Private EXIF-derived camera START VALUES for the existing RGB SfM core.

The EXIF 35-mm-equivalent focal length specifies a diagonal field-of-view
equivalence. It is converted to original-image pixels using both diagonals;
the physical millimetre focal length alone cannot provide that conversion.
These nominal values are neither a measured calibration nor self-calibration.
JPEG/MPO headers are read in place; original image pixels are never decoded,
extracted, rewritten, or included in the public-safe initialization summary.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from math import hypot, isclose, isfinite
from pathlib import Path, PurePosixPath
import stat
import struct
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image

from mcm_solarcheck.importers.m3t_rgb import _jpeg_dimensions
from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
from mcm_solarcheck.services.rgb_flight_zip_series import (
    PrivateRGBFlightFrame, PrivateRGBFlightSeries,
)

_HEADER_LIMIT = 1_048_576
_EXIF_IFD = 34665
_FOCAL_35MM = 41989


@dataclass(frozen=True)
class RGBInitialCameraModel:
    intrinsics: PinholeIntrinsics
    distortion: np.ndarray
    source: str
    frame_count: int
    width: int
    height: int
    focal_35mm_equivalent_mm: float

    def privacy_safe_summary(self) -> dict:
        """Aggregate camera start values only; no file/device/GPS identifiers."""
        return {
            "schema": "mcm.rgb_initial_camera_model.v1",
            "status": "estimated_camera_start_values_not_calibrated",
            "source": self.source,
            "camera_count": self.frame_count,
            "original_width": self.width,
            "original_height": self.height,
            "focal_35mm_equivalent_mm": self.focal_35mm_equivalent_mm,
            "initial_fx_px": self.intrinsics.fx,
            "initial_fy_px": self.intrinsics.fy,
            "initial_cx_px": self.intrinsics.cx,
            "initial_cy_px": self.intrinsics.cy,
            "assumptions": [
                "square_pixels", "principal_point_at_image_center",
                "zero_distortion_start_only", "nominal_exif_diagonal_field_of_view",
            ],
            "shared_camera_metadata_consistent": True,
            "lens_calibration_verified": False,
            "intrinsics_optimized": False,
            "independent_survey_verified": False,
            "rgb_orthophoto_verified": False,
        }


def _read_jpeg_header(stream) -> tuple[bytes, bytes | None]:
    """Read bounded marker segments through SOS, never the entropy-coded image."""
    header = bytearray()

    def read(size):
        if len(header) + size > _HEADER_LIMIT:
            raise ValueError("bounded original RGB JPEG metadata required")
        value = stream.read(size)
        if len(value) != size:
            raise ValueError("complete original RGB JPEG metadata required")
        header.extend(value)
        return value

    if read(2) != b"\xff\xd8":
        raise ValueError("original RGB JPEG metadata required")
    exif_payload = None
    while True:
        if read(1) != b"\xff":
            raise ValueError("valid original RGB JPEG markers required")
        marker = read(1)[0]
        while marker == 0xFF:
            marker = read(1)[0]
        if marker in (0x00, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            raise ValueError("complete original RGB JPEG header required")
        if marker == 0x01:
            continue
        size = int.from_bytes(read(2), "big")
        if size < 2:
            raise ValueError("valid original RGB JPEG segment length required")
        payload = read(size - 2)
        if marker == 0xE1 and payload.startswith(b"Exif\0\0"):
            if exif_payload is not None and payload != exif_payload:
                raise ValueError("unambiguous original RGB EXIF required")
            exif_payload = payload
        if marker == 0xDA:
            return bytes(header), exif_payload


def _number(value, description, *, minimum=0., maximum=1.e6):
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(description)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        raise ValueError(description) from None
    if not isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(description)
    return result


def _metadata(exif_payload):
    if exif_payload is None:
        raise ValueError("35 mm-equivalent EXIF focal length required for automatic RGB initialization")
    try:
        exif = Image.Exif()
        exif.load(exif_payload)
        specific = exif.get_ifd(_EXIF_IFD) if _EXIF_IFD in exif else {}
    except (ValueError, TypeError, IndexError, KeyError, OSError, SyntaxError, struct.error):
        raise ValueError("readable original RGB EXIF required") from None

    def tag(number):
        primary, nested = exif.get(number), specific.get(number)
        if primary is not None and nested is not None and primary != nested:
            raise ValueError("unambiguous original RGB camera metadata required")
        return nested if nested is not None else primary

    make, model = tag(271), tag(272)
    if (not isinstance(make, str) or not make.strip("\0 ")
            or not isinstance(model, str) or not model.strip("\0 ")):
        raise ValueError("consistent identifiable RGB camera metadata required")
    focal = _number(
        tag(_FOCAL_35MM), "35 mm-equivalent EXIF focal length required for automatic RGB initialization",
        minimum=1., maximum=2000.,
    )
    orientation = tag(274)
    if orientation is not None and orientation != 1:
        raise ValueError("unrotated original RGB pixel coordinates required")
    zoom = tag(41988)
    if zoom is not None:
        zoom = _number(zoom, "supported original RGB zoom mode required", minimum=0.)
        if not (isclose(zoom, 0., abs_tol=1.e-8) or isclose(zoom, 1., abs_tol=1.e-8)):
            raise ValueError("uncropped original RGB unit-zoom images required for automatic initialization")
    physical = tag(37386)
    if physical is not None:
        physical = _number(physical, "finite consistent RGB lens metadata required", minimum=1.e-6)
    # These values are only compared internally; serial numbers never leave
    # this function or appear in an initialization/error report.
    identity = tuple(
        value.strip("\0 ") if isinstance(value, str) else value
        for value in (make, model, tag(42036), tag(42033), tag(42037))
    )
    dimensions = (tag(40962), tag(40963))
    return focal, physical, identity, dimensions


def initialize_private_rgb_camera(series: PrivateRGBFlightSeries) -> RGBInitialCameraModel:
    """Initialize one consistent camera from all original EXIF headers.

    Missing/ambiguous focal evidence fails clearly; neither 4.4-mm EXIF nor
    nominal sensor-format names are silently treated as a pixel focal length.
    Matching metadata is not a proof of independently calibrated hardware.
    """
    if (not isinstance(series, PrivateRGBFlightSeries)
            or not isinstance(series.frames, tuple) or not 1 <= len(series.frames) <= 1000
            or any(not isinstance(frame, PrivateRGBFlightFrame) for frame in series.frames)):
        raise TypeError("bounded existing private RGB flight series required")
    dimensions = {(frame.width, frame.height) for frame in series.frames}
    if len(dimensions) != 1:
        raise ValueError("one consistent original RGB image mode required")
    width, height = next(iter(dimensions))
    if (type(width) is not int or type(height) is not int
            or not 256 <= width <= 10000 or not 256 <= height <= 10000):
        raise ValueError("bounded original RGB sensor dimensions required")
    if len({Path(frame.source_archive) for frame in series.frames}) > 64:
        raise ValueError("bounded original RGB archive count required")
    common = None
    with ExitStack() as stack:
        archives = {}
        for frame in series.frames:
            path = Path(frame.source_archive)
            member = PurePosixPath(frame.member_name)
            if (not path.is_file() or path.is_symlink() or member.is_absolute()
                    or ".." in member.parts or "\\" in frame.member_name
                    or any(":" in part for part in member.parts)):
                raise ValueError("existing safe original RGB archive required")
            try:
                archive = archives.get(path)
                if archive is None:
                    archive = stack.enter_context(ZipFile(path, "r"))
                    archives[path] = archive
                entry = archive.getinfo(frame.member_name)
                if (entry.is_dir() or not 128 <= entry.file_size <= 40_000_000
                        or entry.file_size != frame.image_size_bytes
                        or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG)):
                    raise ValueError("unchanged regular original RGB image required")
                with archive.open(entry) as image:
                    header, payload = _read_jpeg_header(image)
            except (BadZipFile, KeyError, OSError, RuntimeError):
                raise ValueError("readable original RGB archive metadata required") from None
            if _jpeg_dimensions(header) != (width, height):
                raise ValueError("unchanged consistent original RGB dimensions required")
            focal, physical, identity, exif_dimensions = _metadata(payload)
            if any(value is not None and value != wanted
                   for value, wanted in zip(exif_dimensions, (width, height))):
                raise ValueError("EXIF dimensions must match original RGB pixel coordinates")
            if common is None:
                common = (focal, physical, identity)
            elif (not isclose(focal, common[0], rel_tol=1.e-8, abs_tol=1.e-8)
                  or physical != common[1] or identity != common[2]):
                raise ValueError("one shared RGB camera, lens and focal mode required")
    focal_pixels = common[0] * hypot(width, height) / hypot(36., 24.)
    distortion = np.zeros(5, dtype=np.float64)
    distortion.setflags(write=False)
    return RGBInitialCameraModel(
        PinholeIntrinsics(focal_pixels, focal_pixels, width / 2., height / 2.),
        distortion, "exif_35mm_equivalent", len(series.frames), width, height, common[0],
    )
