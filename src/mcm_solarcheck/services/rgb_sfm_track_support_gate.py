"""Require actual shared retained SfM 3D landmarks across the complete RGB flight.

Calibrated essential-matrix links are candidate geometry, not proof that the
assembled global 3D track bundle joins the same cameras. This read-only gate
verifies a connected graph using ONLY actual final SfM observation identities.
No identifiers, feature locations or metric data are ever exported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBSfMTrackSupportAudit:
    camera_count: int
    supported_link_pair_count: int
    weak_geometrically_accepted_pair_count: int
    supported_spatial_nonadjacent_pairs: int
    spatial_supported_camera_count: int
    minimum_observations_per_camera: int
    minimum_shared_landmarks_per_supported_edge: int
    min_required_shared_landmarks: int

    def privacy_safe_summary(self) -> dict:
        return {
            "sfm_3d_track_graph_connected": True,
            "sfm_3d_supported_pair_count": self.supported_link_pair_count,
            "sfm_3d_weak_geometric_pair_count": (
                self.weak_geometrically_accepted_pair_count
            ),
            "sfm_3d_supported_spatial_nonadjacent_pairs": (
                self.supported_spatial_nonadjacent_pairs
            ),
            "sfm_3d_spatial_supported_camera_count": (
                self.spatial_supported_camera_count
            ),
            "sfm_3d_min_observations_per_camera": (
                self.minimum_observations_per_camera
            ),
            "sfm_3d_min_shared_landmarks_supported_edge": (
                self.minimum_shared_landmarks_per_supported_edge
            ),
            "sfm_3d_required_shared_landmarks_per_edge": (
                self.min_required_shared_landmarks
            ),
            "physical_flight_strip_identity_verified": False,
            "independent_survey_verified": False,
            "roof_geometry_verified": False,
            "orthophoto_verified": False,
        }


def audit_complete_rgb_sfm_track_support(
    frame_ids: tuple[str, ...],
    accepted_geometric_pairs: tuple[tuple[str, str], ...],
    proposed_spatial_nonadjacent_pairs: tuple[tuple[str, str], ...],
    observation_frame_indices: np.ndarray,
    observation_landmark_indices: np.ndarray,
    *,
    landmark_count: int,
    min_shared_landmarks: int = 8,
    should_cancel=None,
) -> RGBSfMTrackSupportAudit:
    """Require whole-flight connectivity from distinct shared 3D point IDs.

    A geometrically accepted pair is retained for this graph ONLY if it shares
    >= min_shared_landmarks distinct reconstructed 3D points. Requiring two
    spatial-nonadjacent supported links with three distinct cameras prevents a
    GPS-proposed/Essential-only cross-region link from masquerading as 3D tracks.
    """
    if should_cancel is not None and should_cancel():
        raise RuntimeError("RGB 3D track-support audit cancelled")
    if (type(frame_ids) is not tuple or not 12 <= len(frame_ids) <= 1000
            or any(type(fid) is not str or not fid for fid in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
            or type(min_shared_landmarks) is not int
            or not 4 <= min_shared_landmarks <= 64
            or type(landmark_count) is not int
            or not 8 <= landmark_count <= 200000):
        raise ValueError("bounded unique RGB track-support frames required")
    names = {name: index for index, name in enumerate(frame_ids)}

    def normalized_pairs(values, title):
        if type(values) is not tuple or len(values) > 16000:
            raise ValueError(f"invalid {title} 3D camera pair evidence")
        seen = set()
        for pair in values:
            if (type(pair) is not tuple or len(pair) != 2
                    or pair[0] not in names or pair[1] not in names
                    or pair[0] == pair[1]):
                raise ValueError(f"invalid {title} 3D camera pair evidence")
            i, j = names[pair[0]], names[pair[1]]
            edge = (min(i, j), max(i, j))
            if edge in seen:
                raise ValueError(f"duplicate {title} 3D camera pair evidence")
            seen.add(edge)
        return seen

    accepted = normalized_pairs(accepted_geometric_pairs, "accepted")
    spatial = normalized_pairs(proposed_spatial_nonadjacent_pairs, "spatial")
    fid, lid = observation_frame_indices, observation_landmark_indices
    if (not accepted or not isinstance(fid, np.ndarray)
            or not isinstance(lid, np.ndarray)
            or fid.ndim != 1 or lid.ndim != 1
            or len(fid) != len(lid) or not 24 <= len(fid) <= 800000
            or not np.issubdtype(fid.dtype, np.integer)
            or not np.issubdtype(lid.dtype, np.integer)
            or np.any(fid < 0) or np.any(fid >= len(frame_ids))
            or np.any(lid < 0) or np.any(lid >= landmark_count)):
        raise ValueError("complete bounded retained SfM 3D observations required")
    # Real integer 3D landmark IDs, never a reprojected descriptor count.
    # Set intersection counts each shared landmark once even if the original
    # track contains duplicated observations within the same camera.
    observed = [set() for _ in frame_ids]
    for camera, point in zip(fid, lid):
        observed[int(camera)].add(int(point))
    sizes = [len(points) for points in observed]
    if min(sizes) < min_shared_landmarks:
        raise ValueError("RGB camera has insufficient retained 3D point support")
    supported = {}
    adjacency = [[] for _ in frame_ids]
    for idx, (a, b) in enumerate(sorted(accepted)):
        if should_cancel is not None and idx % 64 == 0 and should_cancel():
            raise RuntimeError("RGB 3D track-support audit cancelled")
        shared = len(observed[a] & observed[b])
        if shared >= min_shared_landmarks:
            supported[(a, b)] = shared
            adjacency[a].append(b)
            adjacency[b].append(a)
    # A source may have many E-matrix pairs but no assembled 3D point
    # linking its tracks to the other flight cameras.
    seen = {0}
    queue = [0]
    for camera in queue:
        for adjacent in adjacency[camera]:
            if adjacent not in seen:
                seen.add(adjacent)
                queue.append(adjacent)
    if len(seen) != len(frame_ids):
        raise ValueError("disconnected shared-3D-track RGB flight graph")
    nonadjacent = set(supported) & spatial
    spatial_cameras = {name for pair in nonadjacent for name in pair}
    if len(nonadjacent) < 2 or len(spatial_cameras) < 3:
        raise ValueError("insufficient shared 3D tracks across nonadjacent RGB cameras")
    return RGBSfMTrackSupportAudit(
        len(frame_ids), len(supported), len(accepted)-len(supported),
        len(nonadjacent), len(spatial_cameras), min(sizes),
        min(supported.values()), min_shared_landmarks,
    )
