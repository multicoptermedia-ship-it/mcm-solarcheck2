"""Synthetic actual-3D-track support evidence across complete RGB SfM flight."""
import json

import numpy as np
import pytest

from mcm_solarcheck.services.rgb_sfm_track_support_gate import (
    audit_complete_rgb_sfm_track_support,
)


def _flight(n=12, landmarks=16):
    ids = tuple(f"V-{i+1:04d}" for i in range(n))
    accepted = tuple((ids[i], ids[i+1]) for i in range(n-1))
    spatial = tuple((ids[i], ids[i+3]) for i in range(n-3))
    fid = np.repeat(np.arange(n, dtype=np.int32), landmarks)
    lid = np.tile(np.arange(landmarks, dtype=np.int32), n)
    return ids, accepted, spatial, fid, lid


def test_all_cameras_joined_by_real_shared_3d_tracks_with_spatial_support():
    ids, edges, spatial, fid, lid = _flight()
    audit = audit_complete_rgb_sfm_track_support(
        ids, edges + spatial, spatial, fid, lid,
        landmark_count=16,
    )
    report = audit.privacy_safe_summary()
    assert report["sfm_3d_track_graph_connected"] is True
    assert report["sfm_3d_min_shared_landmarks_supported_edge"] == 16
    assert report["sfm_3d_supported_spatial_nonadjacent_pairs"] == 9
    assert report["sfm_3d_weak_geometric_pair_count"] == 0
    assert report["physical_flight_strip_identity_verified"] is False
    assert report["independent_survey_verified"] is False
    assert report["roof_geometry_verified"] is False
    assert report["orthophoto_verified"] is False
    for secret in ("V-0001", "DJI_", "GPS", "landmark_ids",
                   "camera_indices", "private/"):
        assert secret not in json.dumps(report)


def test_essential_spatial_pairs_without_common_landmarks_fail():
    ids, sequential, spatial, _, _ = _flight()
    cameras, points = [], []
    for camera in range(11):
        for index in range(8):
            for fid in (camera, camera+1):
                cameras.append(fid)
                points.append(camera * 8 + index)
    with pytest.raises(ValueError, match="insufficient shared 3D tracks"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential + spatial, spatial,
            np.asarray(cameras, np.int32), np.asarray(points, np.int32),
            landmark_count=88,
        )


def test_retained_3d_network_is_disconnected_even_when_essential_graph_connects():
    ids, sequential, spatial, _, _ = _flight()
    # All frames have at least 8 observations, but 3D landmark identities
    # are distinct in the two camera halves. The E-graph remains connected.
    fid = np.repeat(np.arange(12, dtype=np.int32), 16)
    lid = np.concatenate([
        np.arange(16, dtype=np.int32) if i < 6
        else np.arange(16, 32, dtype=np.int32)
        for i in range(12)
    ])
    with pytest.raises(ValueError, match="disconnected shared-3D"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential + spatial, spatial, fid, lid,
            landmark_count=32,
        )


def test_eight_distinct_shared_points_not_many_duplicate_observations():
    ids, sequential, spatial, fid, lid = _flight()
    # 16 observations of the SAME landmark within each camera produce
    # only 1 distinct 3D point for any camera pair, never 16 tracks.
    with pytest.raises(ValueError, match="insufficient retained 3D"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential+spatial, spatial,
            fid, np.zeros_like(lid),
            landmark_count=16,
        )


def test_invalid_3d_indices_and_reversed_duplicate_links_fail_closed():
    ids, sequential, spatial, fid, lid = _flight()
    with pytest.raises(ValueError, match="duplicate"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential + (sequential[0][::-1],) + spatial,
            spatial, fid, lid,
            landmark_count=16,
        )
    invalid = fid.copy()
    invalid[0] = -1
    with pytest.raises(ValueError, match="bounded"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential + spatial, spatial, invalid, lid,
            landmark_count=16,
        )
    with pytest.raises(RuntimeError, match="cancelled"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential + spatial, spatial, fid, lid,
            landmark_count=16, should_cancel=lambda: True,
        )


def test_1000_frame_iterative_shared_3d_graph_is_bounded():
    ids, sequential, spatial, fid, lid = _flight(n=1000, landmarks=8)
    audit = audit_complete_rgb_sfm_track_support(
        ids, sequential + spatial[:2], spatial,
        fid, lid, landmark_count=8,
    )
    assert audit.camera_count == 1000
    assert audit.supported_spatial_nonadjacent_pairs == 2
    assert audit.spatial_supported_camera_count >= 3


def test_shared_3d_identity_must_reference_an_existing_retained_landmark():
    ids, sequential, spatial, fid, lid = _flight()
    invalid = lid.copy()
    invalid[0] = 180000  # Within generic resource cap, outside actual 16.
    with pytest.raises(ValueError, match="bounded retained"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential + spatial, spatial, fid, invalid,
            landmark_count=16,
        )
    with pytest.raises(ValueError, match="bounded unique"):
        audit_complete_rgb_sfm_track_support(
            ids, sequential + spatial, spatial, fid, lid,
            landmark_count=0,
        )
