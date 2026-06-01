"""CLI entrypoint for ok-plan: prepare subject folder + HTML QC report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create out/sub-XX/nifti_files/ with inputs aligned to the pressure "
            "map grid, then build report.html (ROI, segmentation, derived masks). "
            "Pass --config project.json for paths and acoustic settings (CLI flags "
            "override JSON defaults)."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Project JSON (subject, output_root, inputs.*, acoustic.center_frequency_mhz). "
            "Relative paths in JSON resolve against the config file directory."
        ),
    )
    p.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject label for output layout (e.g. sub-02 → out/sub-02/...)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for per-subject outputs (default: ./out)",
    )
    p.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="HTML report filename inside the subject folder (default: report.html)",
    )
    p.add_argument(
        "--charm-dir",
        type=Path,
        default=None,
        help="Path to SimNIBS CHARM / m2m output folder (contains final_tissues.nii.gz)",
    )
    p.add_argument("--t1", type=Path, default=None, help="Subject T1w NIfTI")
    p.add_argument("--pressure", type=Path, default=None, help="Pressure map NIfTI")
    p.add_argument(
        "--temperature",
        type=Path,
        default=None,
        help="Temperature map NIfTI (absolute °C)",
    )
    p.add_argument("--roi", type=Path, default=None, help="Target ROI NIfTI")
    p.add_argument(
        "--center-frequency-mhz",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Center / fundamental frequency in MHz for MI = (|p|_max in Pa / 10⁶) / √f "
            "(default: 0.286 = 286 kHz). Pressure is assumed to be in Pascal."
        ),
    )
    p.add_argument(
        "--report-notes",
        type=str,
        default=None,
        help="Optional text row in the report overview (also settable in JSON).",
    )
    p.add_argument(
        "--methodology-reference-url",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "Shown in the report for MI/thermal context. Default: Brain Stimulation 2025 DOI. "
            "Pass an empty string to omit links."
        ),
    )
    p.add_argument(
        "--baseline-body-temp-c",
        type=float,
        default=None,
        metavar="T",
        help="Body baseline temperature in °C for ΔT = T − T_body (default: 37).",
    )
    p.add_argument(
        "--exposure-duration-min",
        type=float,
        default=None,
        metavar="MIN",
        help=(
            "Uniform exposure duration in minutes for CEM43 max (single-map steady assumption; "
            "default: 1)."
        ),
    )
    p.add_argument(
        "--transforms-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory with ANTs transforms (sub-XX_kplan2std_0GenericAffine.mat, "
            "*_1InverseWarp.nii.gz). Required for atlas region tables."
        ),
    )
    p.add_argument(
        "--atlas-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Julich Brain Atlas directory with *_lh/rh_MNI152.nii.gz and .xml. "
            "Required for atlas region tables."
        ),
    )
    p.add_argument(
        "--include-6db",
        action="store_true",
        default=False,
        help=(
            "Include \u22126 dB focus analysis in the report (tables, figures, atlas). "
            "By default only the \u22123 dB focus is reported."
        ),
    )
    return p


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv_list)

    parser = _build_parser()
    if pre_args.config is not None:
        from ok_plan.plan_config import load_plan_config

        pc = load_plan_config(pre_args.config)
        parser.set_defaults(**pc.as_arg_defaults())

    defaults_for_cli = {
        "output_root": Path("out"),
        "report_name": "report.html",
        "center_frequency_mhz": 0.286,
        "baseline_body_temp_c": 37.0,
        "exposure_duration_min": 1.0,
    }
    ns = parser.parse_args(argv_list)
    for key, val in defaults_for_cli.items():
        if getattr(ns, key, None) is None:
            setattr(ns, key, val)
    return ns


def _is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _validate_charm_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"Charm directory does not exist or is not a directory: {path}")


def _validate_nifti_file(label: str, path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if not _is_nifti_path(path):
        raise SystemExit(f"{label} must be a .nii or .nii.gz file: {path}")
    try:
        nib.load(path)
    except Exception as e:
        raise SystemExit(f"{label} could not be loaded as NIfTI ({path}): {e}") from e


def _validate_args(args: argparse.Namespace) -> None:
    missing: list[str] = []
    if not args.subject:
        missing.append("--subject")
    if args.charm_dir is None:
        missing.append("--charm-dir")
    if args.t1 is None:
        missing.append("--t1")
    if args.pressure is None:
        missing.append("--pressure")
    if args.temperature is None:
        missing.append("--temperature")
    if args.roi is None:
        missing.append("--roi")
    if missing:
        raise SystemExit(
            "Missing required inputs: "
            + ", ".join(missing)
            + ". Provide them on the command line or in --config JSON."
        )
    _validate_charm_dir(args.charm_dir)
    _validate_nifti_file("T1w (--t1)", args.t1)
    _validate_nifti_file("Pressure map (--pressure)", args.pressure)
    _validate_nifti_file("Temperature map (--temperature)", args.temperature)
    _validate_nifti_file("ROI (--roi)", args.roi)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")

    from ok_plan.prepare import prepare_subject_workspace
    from ok_plan.report import write_html_report

    _validate_args(args)
    try:
        ws = prepare_subject_workspace(
            output_root=args.output_root,
            subject=args.subject,
            charm_dir=args.charm_dir,
            t1_path=args.t1,
            roi_path=args.roi,
            pressure_path=args.pressure,
            temperature_path=args.temperature,
            report_name=args.report_name,
        )
        meth_url = args.methodology_reference_url
        write_html_report(
            report_path=ws.report_path,
            charm_dir=ws.charm_dir_effective,
            t1_path=ws.t1_path,
            roi_path=ws.roi_path,
            pressure_path=ws.pressure_path,
            temperature_path=ws.temperature_path,
            source_charm_dir=args.charm_dir,
            center_frequency_mhz=args.center_frequency_mhz,
            report_notes=args.report_notes,
            methodology_reference_url=meth_url,
            baseline_body_temp_c=args.baseline_body_temp_c,
            exposure_duration_min=args.exposure_duration_min,
            transforms_dir=getattr(args, "transforms_dir", None),
            atlas_dir=getattr(args, "atlas_dir", None),
            subject_prefix=args.subject,
            include_6db=getattr(args, "include_6db", False),
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    except ValueError as e:
        raise SystemExit(str(e)) from e

    print(f"Prepared aligned NIfTIs: {ws.nifti_dir}")
    print(f"Tissue masks (native labels, nearest to sim grid): {ws.derived_masks_dir}")
    print(f"Wrote report: {ws.report_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
