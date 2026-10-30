"""Local read-only complete RGB flight run; never export private coordinates.

Example (user-local paths):
python -m mcm_solarcheck.services.rgb_zip_whole_flight_cli \\
  --rgb-archive flight.zip --report absolute-local-report.json

The default camera initialization uses image metadata as unverified start values
and refines common focal length in the existing multi-view pipeline. A separate
verified --calibration-json remains optional.

The JPEG originals are opened from ZIP members in memory, never extracted.
Only aggregated diagnostic numbers go to JSON/stdout. An explicitly selected
private candidate directory can receive an unverified RGB raster and NoData mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2

from mcm_solarcheck.services.rgb_zip_whole_flight_reconstruction import (
    PrivateRGBFlightStageError, reconstruct_private_complete_rgb_zip_flight,
)
from mcm_solarcheck.services.rgb_flight_surface_output import (
    validate_private_rgb_surface_output_directory,
    write_private_rgb_flight_surface_candidate,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PRIVATE whole-flight RGB SfM + RTK windowed BA (not an orthophoto)",
    )
    parser.add_argument("--rgb-archive", action="append", required=True,
                        help="local RGB ZIP/.tip path; repeat for all flight archives")
    parser.add_argument("--calibration-json", type=Path,
                        help="optional independently verified RGB calibration; otherwise estimate initial camera from EXIF")
    parser.add_argument("--report", type=Path,
                        help="optional NEW absolute local JSON (no GPS/images)")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--max-features", type=int, default=1600)
    parser.add_argument("--feature-method", choices=("sift", "orb"), default="sift",
                        help="multiview feature method (default: sift; legacy orb available)")
    parser.add_argument("--candidate-output-dir", type=Path,
                        help="optional NEW absolute private directory for an unverified surface RGB PNG and NoData mask")
    parser.add_argument("--candidate-max-side", type=int, default=4096)
    parser.add_argument("--candidate-metres-per-pixel", type=float)
    args = parser.parse_args(argv)
    archives = tuple(Path(value) for value in args.rgb_archive)
    if args.candidate_output_dir is not None:
        validate_private_rgb_surface_output_directory(args.candidate_output_dir, archives)
    calibration_file = args.calibration_json
    if (calibration_file is not None
            and (not calibration_file.is_file() or calibration_file.is_symlink()
                 or calibration_file.stat().st_size > 1_000_000)):
        raise ValueError("existing bounded external calibration file required")
    report_file = args.report
    if report_file is not None:
        if (not report_file.is_absolute() or report_file.is_symlink()
                or report_file.exists() or not report_file.parent.is_dir()
                or report_file.parent.is_symlink()
                or report_file.parent.resolve() in {
                    path.resolve().parent for path in archives
                }):
            raise ValueError("new local report outside every RGB archive folder required")
    try:
        calibration = (None if calibration_file is None else
                       json.loads(calibration_file.read_text(encoding="utf-8")))
        outcome = reconstruct_private_complete_rgb_zip_flight(
            archives, calibration,
            max_side=args.max_side, max_features=args.max_features,
            feature_method=args.feature_method,
        )
        result = outcome.privacy_safe_summary()
        if args.candidate_output_dir is not None:
            try:
                result["surface_candidate"] = write_private_rgb_flight_surface_candidate(
                    outcome, args.candidate_output_dir,
                    max_side=args.candidate_max_side,
                    metres_per_pixel=args.candidate_metres_per_pixel,
                )
            except (ValueError, TypeError, OSError, cv2.error) as exc:
                raise PrivateRGBFlightStageError("surface_candidate") from exc
        status = 0
    except PrivateRGBFlightStageError as failed:
        result = {
            "schema": "mcm.private_complete_rgb_flight_reconstruction.v1",
            "status": "whole_flight_rgb_quality_gate_failed",
            "failed_stage": failed.stage,
            "all_input_rgb_frames_processed": failed.stage == "surface_candidate",
            "roof_geometry_verified": False,
            "rgb_orthophoto_verified": False,
            "module_analysis_verified": False,
            "thermal_registration_verified": False,
            **failed.public_quality_summary,
        }
        status = 2
    except (ValueError, TypeError, json.JSONDecodeError):
        result = {
            "schema": "mcm.private_complete_rgb_flight_reconstruction.v1",
            "status": "whole_flight_rgb_calibration_gate_failed",
            "failed_stage": "calibration",
            "all_input_rgb_frames_processed": False,
            "roof_geometry_verified": False,
            "rgb_orthophoto_verified": False,
            "module_analysis_verified": False,
            "thermal_registration_verified": False,
        }
        status = 2
    if report_file is not None:
        with report_file.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
    print(json.dumps(result, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
