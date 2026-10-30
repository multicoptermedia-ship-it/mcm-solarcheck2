"""Private whole-flight surface preview, with separate explicitly requested output.

Original archive members are read, never extracted or rewritten. The resulting
PNG is a diagnostic candidate with a real NoData alpha channel, not a certified
orthophoto, survey or module-inspection product. Only aggregate JSON is written;
camera identities, source paths and ENU origins stay private in memory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from zipfile import ZipFile

import cv2
import numpy as np

from mcm_solarcheck.services.rgb_flight_zip_series import PrivateRGBFlightSeries
from mcm_solarcheck.services.rgb_reconstructed_surface_candidate import (
    fit_rgb_surface_candidates,
    render_rgb_reconstructed_surface_candidate,
)
from mcm_solarcheck.services.rgb_zip_whole_flight_reconstruction import (
    PrivateRGBFlightReconstruction,
)


def validate_private_rgb_surface_output_directory(output_directory, archives) -> Path:
    """Check a NEW absolute output directory, without creating any filesystem item.

    The immediate parent must already exist. All source archive directories and
    actual Git checkouts are protected. Symlink ancestors and lexical parent
    traversals are rejected before resolution, including dangling symlinks.
    """
    output = Path(output_directory)
    if not output.is_absolute() or ".." in output.parts:
        raise ValueError("new absolute private surface output directory required")
    if isinstance(archives, (str, bytes, Path)):
        raise ValueError("source archive sequence required for output protection")
    sources = tuple(Path(source) for source in archives)
    if not sources:
        raise ValueError("source archive sequence required for output protection")
    for ancestor in (output, *output.parents):
        if ancestor.is_symlink():
            raise ValueError("symlink surface output directory or ancestor disallowed")
        marker = ancestor / ".git"
        # /workspace can contain an empty platform .git marker; a real checkout
        # has a HEAD/config or a .git file referring to worktree metadata.
        if (marker.is_file() or (marker / "HEAD").exists()
                or (marker / "config").exists()):
            raise ValueError("private surface output must remain outside repositories")
    if output.exists():
        raise ValueError("private surface output directory must not already exist")
    if not output.parent.is_dir():
        raise ValueError("existing private surface output parent directory required")
    output = output.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    protected = (repository_root, *(source.resolve().parent for source in sources))
    if any(output.is_relative_to(folder) for folder in protected):
        raise ValueError("private surface output must remain outside original/archive folders")
    return output


def _cancel(callback):
    if callback is not None and callback():
        raise RuntimeError("whole-flight RGB surface output cancelled")


def _read_original_bgr(frame):
    """Same bounded, primary-JPEG ZIP reading contract as feature extraction."""
    member = PurePosixPath(frame.member_name)
    if (member.is_absolute() or ".." in member.parts
            or "\\" in frame.member_name or re.match(r"^[A-Za-z]:", frame.member_name)):
        raise ValueError("unsafe original RGB archive member")
    with ZipFile(frame.source_archive) as archive:
        entry = archive.getinfo(frame.member_name)
        kind = stat.S_IFMT(entry.external_attr >> 16)
        if (entry.is_dir() or kind not in (0, stat.S_IFREG)
                or not 128 <= entry.file_size <= 40_000_000
                or entry.file_size != frame.image_size_bytes):
            raise ValueError("original RGB member changed or exceeds image budget")
        with archive.open(entry) as source:
            data = source.read(40_000_001)
        if len(data) != entry.file_size:
            raise ValueError("original RGB member changed since validated intake")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (frame.height, frame.width, 3):
        raise ValueError("unreadable original primary RGB JPEG")
    return image


def _write_exclusive(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)


def write_private_rgb_flight_surface_candidate(
    reconstruction,
    output_directory,
    *,
    metres_per_pixel=None,
    max_side: int = 4096,
    should_cancel=None,
    on_progress=None,
) -> dict:
    """Fit/render final ENU BA geometry; create three private diagnostic files.

    The new directory contains ``rgb-orthophoto-candidate.png`` (BGRA with
    alpha=0 for NoData), ``valid-mask.png`` and aggregate ``summary.json``.
    No output directory is created unless geometry and rendering succeed.
    An interrupted write leaves its newly created diagnostic directory intact.
    """
    _cancel(should_cancel)
    if not isinstance(reconstruction, PrivateRGBFlightReconstruction):
        raise TypeError("existing private whole-flight RGB reconstruction required")
    series, network = reconstruction.private_series, reconstruction.private_network
    ba = reconstruction.rtk_windowed_ba
    if (not isinstance(series, PrivateRGBFlightSeries) or not series.frames
            or network is None or reconstruction.private_intrinsics is None):
        raise ValueError("private whole-flight camera, track and archive payload required")
    frame_ids = tuple(frame.frame_id for frame in series.frames)
    if (frame_ids != network.frame_ids or frame_ids != ba.frame_ids
            or len(frame_ids) != reconstruction.frame_count
            or len(ba.world_enu_to_camera_private) != len(frame_ids)
            or len(ba.landmarks_enu_private_m) != network.reconstructed_landmarks
            or len(ba.landmarks_enu_private_m) != len(network.landmark_xyz_unit_baseline)):
        raise ValueError("final ENU geometry must preserve whole-flight track identities")
    dimensions = {(frame.width, frame.height) for frame in series.frames}
    if len(dimensions) != 1:
        raise ValueError("single original RGB sensor dimensions required")
    archives = tuple({frame.source_archive for frame in series.frames})
    output = validate_private_rgb_surface_output_directory(output_directory, archives)
    if on_progress is not None:
        on_progress("surface_fit", 0, 1)
    surfaces = fit_rgb_surface_candidates(
        ba.landmarks_enu_private_m,
        network.observation_frame_indices, network.observation_landmark_indices,
        should_cancel=should_cancel,
    )
    if on_progress is not None:
        on_progress("surface_fit", 1, 1)
    candidate = render_rgb_reconstructed_surface_candidate(
        surfaces, ba.world_enu_to_camera_private,
        reconstruction.private_intrinsics,
        lambda index: _read_original_bgr(series.frames[index]),
        image_size=next(iter(dimensions)),
        distortion_coefficients=reconstruction.private_distortion,
        metres_per_pixel=metres_per_pixel, max_side=max_side,
        should_cancel=should_cancel,
        on_progress=(None if on_progress is None else
                     lambda current, total: on_progress("surface_render", current, total)),
    )
    summary = {
        **candidate.privacy_safe_summary(),
        "candidate_created": True,
        "output_artifact_count": 3,
        "input_frame_count": reconstruction.frame_count,
        "original_images_modified": False,
    }
    mask = candidate.valid_mask.astype(np.uint8) * np.uint8(255)
    bgra = np.dstack((candidate.bgr, mask))
    encoded = []
    for name, image in (("rgb-orthophoto-candidate.png", bgra), ("valid-mask.png", mask)):
        _cancel(should_cancel)
        ok, payload = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("private RGB surface candidate PNG encoding failed")
        encoded.append((name, payload.tobytes()))
    encoded.append(("summary.json", (json.dumps(summary, indent=2, allow_nan=False)+"\n").encode("utf-8")))
    _cancel(should_cancel)
    validate_private_rgb_surface_output_directory(output, archives)
    output.mkdir(mode=0o700)
    for index, (name, payload) in enumerate(encoded):
        _cancel(should_cancel)
        _write_exclusive(output / name, payload)
        if on_progress is not None:
            on_progress("surface_output", index+1, len(encoded))
    return summary
