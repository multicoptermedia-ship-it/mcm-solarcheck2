"""Bounded GLOBAL reprojection bundle adjustment on an existing SfM track bundle.

Levenberg-Marquardt with per-landmark Schur elimination. Camera 0 and camera 1
are FIXED to preserve the arbitrary initial baseline (remove similarity gauge).
This is NOT metric, surveyed, georeferenced, or orthophoto-verified geometry.
Uses NumPy and OpenCV already required by SolarCheck. No SciPy/ODM dependency.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics

@dataclass(frozen=True)
class RGBUnscaledBundleAdjustment:
    world_to_camera: tuple[np.ndarray, ...]
    landmark_xyz_unit_baseline: np.ndarray
    before_rmse_px: float
    after_rmse_px: float
    observation_count: int
    iterations: int
    state: str = "global_bundle_adjustment_unscaled_unverified"
    metric_scale_verified: bool = False
    roof_geometry_verified: bool = False
    orthophoto_verified: bool = False
    rtk_camera_priors_optimized: bool = False
    refined_intrinsics: PinholeIntrinsics | None = None
    focal_refined: bool = False
    focal_information_fraction: float | None = None
    focal_observable: bool = False


def _projections(poses, points, camera_ids, landmark_ids, k):
    transformed = np.einsum(
        "nij,nj->ni", poses[camera_ids, :, :3], points[landmark_ids],
    ) + poses[camera_ids, :, 3]
    depth = transformed[:, 2]
    if not np.isfinite(transformed).all() or np.any(depth <= 1.e-6):
        raise ValueError("RGB BA observation behind camera or nonfinite")
    projected = np.column_stack((
        k[0] * transformed[:, 0] / depth + k[2],
        k[1] * transformed[:, 1] / depth + k[3],
    ))
    return transformed, projected


def _small_rotation(delta):
    rotation, _ = cv2.Rodrigues(np.asarray(delta, dtype=np.float64))
    return rotation


def _robust_gnss_cost(poses, centers, sigmas, fitted, huber):
    """Robust GNSS objective in normalized camera-center metres; no GPS output."""
    center = -np.einsum("nji,nj->ni", poses[:, :, :3], poses[:, :, 3])
    whitened = (center[fitted] - centers[fitted]) / sigmas[fitted]
    norm = np.linalg.norm(whitened, axis=1)
    cost = np.sum(np.where(norm <= huber, .5 * norm**2,
                           huber * (norm - .5 * huber)))
    return float(cost), whitened, norm


def _focal_information_fraction(poses, points, fid, lid, pixels, k, first_variable_camera, huber):
    """Pixel-only undamped focal information after eliminating pose/point DOFs.

    This is a local degeneracy diagnostic, not independent calibration evidence.
    GNSS penalties and LM damping deliberately do not make focal information.
    """
    transformed, projected = _projections(poses, points, fid, lid, k)
    errors = np.linalg.norm(projected - pixels, axis=1)
    pose_dof = 6 * (len(poses) - first_variable_camera)
    reduced = np.zeros((pose_dof + 1, pose_dof + 1))
    raw_focal_information = 0.
    for landmark in range(len(points)):
        hll = np.zeros((3, 3))
        cross = np.zeros((pose_dof + 1, 3))
        for ix in np.flatnonzero(lid == landmark):
            i = int(fid[ix]); xc = transformed[ix]; z = xc[2]
            jp = np.array([[k[0]/z, 0., -k[0]*xc[0]/z**2],
                           [0., k[1]/z, -k[1]*xc[1]/z**2]])
            jl = jp @ poses[i, :, :3]
            jc = np.zeros((2, pose_dof + 1))
            jc[:, -1] = (k[0]*xc[0]/z, k[1]*xc[1]/z)
            if i >= first_variable_camera:
                skew = np.array([[0., -xc[2], xc[1]],
                                 [xc[2], 0., -xc[0]], [-xc[1], xc[0], 0.]])
                sl = slice(6*(i-first_variable_camera), 6*(i-first_variable_camera+1))
                jc[:, sl] = jp @ np.column_stack((-skew, np.eye(3)))
            weight = min(1., huber / max(1.e-12, errors[ix]))
            raw_focal_information += weight * float(jc[:, -1] @ jc[:, -1])
            hll += weight * (jl.T @ jl)
            cross += weight * (jc.T @ jl)
            reduced += weight * (jc.T @ jc)
        reduced -= cross @ np.linalg.pinv(hll, rcond=1.e-10) @ cross.T
    reduced = .5 * (reduced + reduced.T)
    diagonal = float(reduced[-1, -1])
    # Normalize against information BEFORE point elimination. In a focal/depth
    # gauge the eliminated diagonal is numerical cancellation around zero;
    # dividing its marginal by that same tiny diagonal would spuriously report
    # almost perfect observability, especially with all GNSS-mode poses free.
    if (not np.isfinite(reduced).all() or raw_focal_information <= 1.e-12
            or diagonal <= 1.e-12 * raw_focal_information):
        return 0.
    coupling = reduced[:-1, -1]
    marginal = diagonal - float(coupling @ np.linalg.pinv(
        reduced[:-1, :-1], rcond=1.e-10) @ coupling)
    return float(np.clip(marginal / raw_focal_information, 0., 1.))


def adjust_unscaled_rgb_bundle(
    network, intrinsics, *, max_iterations=12,
    max_cameras=12, max_landmarks=400, max_observations=4000,
    huber_threshold_px=2., should_cancel=None,
    camera_center_priors_m=None, camera_center_sigma_m=None,
    camera_prior_fit_mask=None, gnss_huber_sigma=3.,
    refine_focal: bool = False,
    focal_scale_bounds: tuple[float, float] = (.5, 2.),
):
    """Jointly refine *all selected* 3D landmarks and non-anchor camera poses.

    Requires the complete previously verified multi-camera observation bundle.
    The small all-network limit is intentional; oversized projects FAIL CLOSED.
    With complete metric RTK priors, ALL camera centers are adjustable and the
    reference frame is constrained by the GNSS penalty, NOT first-two anchors.
    A false fit mask explicitly withholds RTK values from the optimization.
    Neither version verifies surveying, 3D roofs, or an orthophoto.
    Optional focal refinement jointly fits one log-scale for fx/fy; principal
    point, aspect ratio and distortion remain fixed. This is not a complete
    self-calibration model or proof that focal length is observable.
    """
    from mcm_solarcheck.services.rgb_global_camera_network import RGBUnscaledNetwork
    from mcm_solarcheck.services.module_plane_projection import PinholeIntrinsics
    if not isinstance(network, RGBUnscaledNetwork):
        raise TypeError("existing unscaled multi-view RGB network required")
    if not isinstance(intrinsics, PinholeIntrinsics):
        raise TypeError("RGB pinhole camera intrinsics required")
    if (type(refine_focal) is not bool
            or type(max_iterations) is not int or not 1 <= max_iterations <= 40
            or type(max_cameras) is not int or not 3 <= max_cameras <= 40
            or type(max_landmarks) is not int or not 8 <= max_landmarks <= 2000
            or type(max_observations) is not int or not 24 <= max_observations <= 20000
            or type(huber_threshold_px) not in (float, int)
            or not np.isfinite(huber_threshold_px) or not .1 <= huber_threshold_px <= 10):
        raise ValueError("invalid bounded global BA resource and Huber settings")
    if (not isinstance(focal_scale_bounds, (tuple, list)) or len(focal_scale_bounds) != 2
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in focal_scale_bounds)
            or not 0 < focal_scale_bounds[0] <= 1 <= focal_scale_bounds[1]
            or focal_scale_bounds[0] >= focal_scale_bounds[1]):
        raise ValueError("invalid bounded focal refinement scale")
    count = len(network.frame_ids)
    if not 3 <= count <= max_cameras or len(network.world_to_camera) != count:
        raise ValueError("global RGB camera network exceeds BA camera budget")
    points = np.asarray(network.landmark_xyz_unit_baseline, dtype=np.float64)
    cameras = np.asarray(network.world_to_camera, dtype=np.float64)
    lid = np.asarray(network.observation_landmark_indices)
    fid = np.asarray(network.observation_frame_indices)
    pixels = np.asarray(network.observation_pixels_xy, dtype=np.float64)
    k = np.asarray([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy], dtype=np.float64)
    initial_k = k.copy()
    focal_log_scale = 0.
    if (points.ndim != 2 or points.shape[1] != 3
            or not 8 <= len(points) <= max_landmarks
            or cameras.shape != (count, 3, 4)
            or fid.ndim != 1 or lid.ndim != 1 or pixels.shape != (len(fid), 2)
            or not 24 <= len(fid) <= max_observations or len(lid) != len(fid)
            or not np.issubdtype(fid.dtype, np.integer)
            or not np.issubdtype(lid.dtype, np.integer)
            or np.any(fid < 0) or np.any(fid >= count)
            or np.any(lid < 0) or np.any(lid >= len(points))
            or any(not np.isfinite(v).all() for v in (points, cameras, pixels, k))
            or np.min(k[:2]) <= 0):
        raise ValueError("invalid complete finite SfM BA observations")
    if (len(np.unique(fid)) != count
            or len(np.unique(lid)) != len(points)
            or any(np.count_nonzero(fid == j) < 8 for j in range(count))):
        raise ValueError("insufficient all-camera landmark support")
    if any(not np.allclose(c[:,:3].T@c[:,:3], np.eye(3), atol=1e-4)
           or not np.isclose(np.linalg.det(c[:,:3]), 1., atol=1e-4) for c in cameras):
        raise ValueError("invalid initial camera rotation")
    use_gnss = (camera_center_priors_m is not None
                or camera_center_sigma_m is not None
                or camera_prior_fit_mask is not None)
    centers = sigma = fit = None
    if use_gnss:
        if camera_center_priors_m is None or camera_center_sigma_m is None:
            raise ValueError("RTK prior positions and sigma must be provided together")
        centers = np.asarray(camera_center_priors_m, dtype=np.float64)
        sigma = np.asarray(camera_center_sigma_m, dtype=np.float64)
        fit = (np.ones(count, dtype=bool) if camera_prior_fit_mask is None
               else np.asarray(camera_prior_fit_mask))
        if (centers.shape != (count, 3) or sigma.shape != (count, 3)
                or fit.shape != (count,) or fit.dtype != np.bool_
                or not np.isfinite(centers).all()
                or not np.isfinite(sigma).all()
                or np.any(sigma[:, :2] < .10) or np.any(sigma[:, 2] < .20)
                or np.count_nonzero(fit) < 3
                or type(gnss_huber_sigma) not in (float, int)
                or not np.isfinite(gnss_huber_sigma)
                or not .5 <= gnss_huber_sigma <= 10):
            raise ValueError("complete conservative RTK camera priors required")
    # Legacy unscaled SfM fixes the first two cameras. Metric-prior mode instead
    # optimizes ALL poses, including the first two, with a weighted ENU datum.
    first_variable_camera = 0 if use_gnss else 2
    pose_dof = 6 * (count - first_variable_camera)
    dof = pose_dof + int(refine_focal)
    I3 = np.eye(3)
    by_landmark = [np.flatnonzero(lid == j) for j in range(len(points))]
    def evaluate(pose_array, xyz, camera_k=None):
        transformed, projected = _projections(pose_array, xyz, fid, lid,
                                               k if camera_k is None else camera_k)
        residual = projected - pixels
        norms = np.linalg.norm(residual, axis=1)
        h = float(huber_threshold_px)
        # Robust loss is comparable across trials and resistive to outliers.
        cost = float(np.sum(np.where(norms <= h, .5*norms**2, h*(norms-.5*h))))
        if use_gnss:
            gnss_cost, _, _ = _robust_gnss_cost(
                pose_array, centers, sigma, fit, gnss_huber_sigma,
            )
            cost += gnss_cost
        return transformed, residual, norms, cost
    transformed, residual, norms, cost = evaluate(cameras, points)
    before = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    damping = 1.e-3
    accepted = 0
    for _ in range(max_iterations):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("RGB global bundle adjustment cancelled")
        hcc = np.zeros((dof,dof))
        gcc = np.zeros(dof)
        inverse_blocks = []
        cross_blocks = []
        point_gradients = []
        # The 3D landmarks are independent conditional on the camera steps.
        for j, obs_ids in enumerate(by_landmark):
            hll = damping*I3.copy()
            gl = np.zeros(3)
            cross = np.zeros((dof,3))
            for ix in obs_ids:
                i = int(fid[ix])
                xc = transformed[ix]
                z = xc[2]
                derivative = np.array([
                    [k[0]/z, 0., -k[0]*xc[0]/z**2],
                    [0., k[1]/z, -k[1]*xc[1]/z**2],
                ])
                jl = derivative @ cameras[i,:,:3]
                r = residual[ix]
                weight = min(1., float(huber_threshold_px)/max(1e-12,norms[ix]))
                hll += weight*(jl.T@jl)
                gl += weight*(jl.T@r)
                if refine_focal:
                    jf = np.array((k[0]*xc[0]/z, k[1]*xc[1]/z))
                    hcc[-1, -1] += weight * float(jf @ jf)
                    gcc[-1] += weight * float(jf @ r)
                    cross[-1] += weight * (jf @ jl)
                if i >= first_variable_camera:
                    skew = np.array([
                        [0.,-xc[2],xc[1]],
                        [xc[2],0.,-xc[0]],
                        [-xc[1],xc[0],0.],
                    ])
                    jc = derivative @ np.column_stack((-skew,I3))
                    sl = slice(6*(i-first_variable_camera),6*(i-first_variable_camera+1))
                    hcc[sl,sl] += weight*(jc.T@jc)
                    gcc[sl] += weight*(jc.T@r)
                    cross[sl] += weight*(jc.T@jl)
                    if refine_focal:
                        coupling = weight * (jc.T @ jf)
                        hcc[sl, -1] += coupling
                        hcc[-1, sl] += coupling
            inv = np.linalg.inv(hll)
            hcc -= cross@inv@cross.T
            gcc -= cross@inv@gl
            inverse_blocks.append(inv)
            cross_blocks.append(cross)
            point_gradients.append(gl)
        if use_gnss:
            _, whitening, prior_norms = _robust_gnss_cost(
                cameras, centers, sigma, fit, gnss_huber_sigma,
            )
            for index, i in enumerate(np.flatnonzero(fit)):
                # C = -R.T @ t, R' = exp(dw)@R, t' = exp(dw)@t + dt.
                # Thus dC/d(dt) = -R.T and dC/d(dw) = 0 at first order.
                jac = np.zeros((3, 6))
                jac[:, 3:] = -cameras[i, :, :3].T / sigma[i, :, None]
                sl = slice(6*i, 6*(i+1))
                weight = min(1., gnss_huber_sigma / max(1.e-12, prior_norms[index]))
                hcc[sl, sl] += weight * (jac.T @ jac)
                gcc[sl] += weight * (jac.T @ whitening[index])
        if dof:
            hcc += damping*np.eye(dof)
            try:
                camera_delta = np.linalg.solve(hcc,-gcc)
            except np.linalg.LinAlgError as error:
                raise ValueError("RGB global BA singular camera system") from error
        else:
            camera_delta = np.empty(0)
        candidate_k = k.copy()
        candidate_log_scale = focal_log_scale
        if refine_focal:
            candidate_log_scale = float(np.clip(
                focal_log_scale + np.clip(camera_delta[-1], -.25, .25),
                np.log(focal_scale_bounds[0]), np.log(focal_scale_bounds[1]),
            ))
            camera_delta[-1] = candidate_log_scale - focal_log_scale
            candidate_k[:2] = initial_k[:2] * np.exp(candidate_log_scale)
        candidate_points = points.copy()
        for j in range(len(points)):
            candidate_points[j] -= inverse_blocks[j]@(
                point_gradients[j]+cross_blocks[j].T@camera_delta
            )
        candidate_cameras = cameras.copy()
        for i in range(first_variable_camera,count):
            delta = camera_delta[6*(i-first_variable_camera):6*(i-first_variable_camera+1)]
            rotation = _small_rotation(delta[:3])
            candidate_cameras[i,:,:3] = rotation@cameras[i,:,:3]
            candidate_cameras[i,:,3] = rotation@cameras[i,:,3] + delta[3:]
        try:
            next_transformed,next_residual,next_norms,next_cost = evaluate(
                candidate_cameras,candidate_points,candidate_k,
            )
        except ValueError:
            damping *= 10.
            continue
        if next_cost < cost:
            improvement = cost-next_cost
            points,cameras = candidate_points,candidate_cameras
            k, focal_log_scale = candidate_k, candidate_log_scale
            transformed,residual,norms,cost = (
                next_transformed,next_residual,next_norms,next_cost
            )
            damping = max(1.e-9,damping*.5)
            accepted += 1
            if improvement < 1.e-8:
                break
        else:
            damping = min(1.e10,damping*10.)
    if accepted == 0:
        # Exact or already well-converged inputs remain usable if residuals
        # were explicitly evaluated; never claim an improvement that did not occur.
        if not use_gnss and before > huber_threshold_px:
            raise ValueError("RGB global BA failed to improve high reprojection error")
    after = float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
    if not np.isfinite(after) or (not use_gnss and after > before + 1.e-8):
        raise ValueError("RGB global BA could not validate reprojection improvement")
    for p in cameras:
        p.setflags(write=False)
    points.setflags(write=False)
    focal_information = (_focal_information_fraction(
        cameras, points, fid, lid, pixels, k, first_variable_camera,
        float(huber_threshold_px),
    ) if refine_focal else None)
    return RGBUnscaledBundleAdjustment(
        tuple(p.copy() for p in cameras), points,
        before, after, len(fid), accepted,
        state=("rtk_regularized_window_schur_ba_candidate_unverified" if use_gnss
               else "global_bundle_adjustment_unscaled_unverified"),
        rtk_camera_priors_optimized=use_gnss,
        refined_intrinsics=PinholeIntrinsics(*map(float, k)) if refine_focal else None,
        focal_refined=bool(refine_focal and abs(focal_log_scale) > 1.e-10),
        focal_information_fraction=focal_information,
        focal_observable=bool(focal_information is not None and focal_information > 1.e-7),
    )
