#!/usr/bin/env python3
"""
run_transducer_qc.py
====================
Command-line entry point for TUS transducer-position quality control.

Two modes
---------
1. analyse  (default)  — run the full QC pipeline for one subject
2. inspect-xml         — print a human-readable summary of one XML file

Typical usage
-------------
# Full analysis via CLI flags:
python run_transducer_qc.py \\
  --subject sub-05 \\
  --planned-exp     sub-05_planned_exp.xml \\
  --planned-control sub-05_planned_control.xml \\
  --actual-exp      sub-05_actual_exp.xml \\
  --actual-control  sub-05_actual_control.xml \\
  --planned-exp-left-index      2 \\
  --planned-exp-right-index     9 \\
  --planned-control-left-index  2 \\
  --planned-control-right-index 9 \\
  --actual-exp-left-range       6 88 \\
  --actual-exp-right-range      91 172 \\
  --actual-control-left-range   6 88 \\
  --actual-control-right-range  91 172 \\
  --dbscan-eps-mm auto \\
  --dbscan-min-samples 3 \\
  --outdir reports

# Via YAML config:
python run_transducer_qc.py --config sub-05_config.yaml

# Inspect an XML file:
python run_transducer_qc.py inspect-xml sub-05_actual_exp.xml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ── relative import: works whether run as script or from package ──────────────
sys.path.insert(0, str(Path(__file__).parent))
from transducer_qc.transducer_qc import (
    inspect_xml,
    parse_gummarker_xml,
    analyse_condition,
    fig_positions_over_time,
    fig_spatial_clusters,
    fig_cluster_size_summary,
    fig_displacement_summary,
    build_html_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_transducer_qc.py",
        description="TUS transducer-position quality control pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── sub-commands ──────────────────────────────────────────────────────────
    sub = p.add_subparsers(dest="command")

    # inspect-xml sub-command
    insp = sub.add_parser("inspect-xml",
                          help="Print summary of a GUMMarker XML file")
    insp.add_argument("xml_file", help="Path to the XML file to inspect")

    # ── main analysis arguments (on the root parser so they work with --config) ─
    p.add_argument("--config", metavar="YAML",
                   help="YAML config file (all arguments can be set here)")

    p.add_argument("--subject", metavar="ID",
                   help="Subject identifier, e.g. sub-05")

    # XML file paths
    p.add_argument("--planned-exp",     metavar="XML",
                   help="Planned-experimental XML file")
    p.add_argument("--planned-control", metavar="XML",
                   help="Planned-control XML file")
    p.add_argument("--actual-exp",      metavar="XML",
                   help="Actual-experimental XML file")
    p.add_argument("--actual-control",  metavar="XML",
                   help="Actual-control XML file")

    # Planned reference indices
    p.add_argument("--planned-exp-left-index",     type=int, metavar="N",
                   help="Planned exp XML element index for LEFT hemisphere")
    p.add_argument("--planned-exp-right-index",    type=int, metavar="N",
                   help="Planned exp XML element index for RIGHT hemisphere")
    p.add_argument("--planned-control-left-index", type=int, metavar="N",
                   help="Planned control XML element index for LEFT")
    p.add_argument("--planned-control-right-index",type=int, metavar="N",
                   help="Planned control XML element index for RIGHT")

    # Actual indices — range
    p.add_argument("--actual-exp-left-range",      nargs=2, type=int,
                   metavar=("START", "END"),
                   help="Actual exp LEFT frame range [start end] inclusive")
    p.add_argument("--actual-exp-right-range",     nargs=2, type=int,
                   metavar=("START", "END"),
                   help="Actual exp RIGHT frame range")
    p.add_argument("--actual-control-left-range",  nargs=2, type=int,
                   metavar=("START", "END"),
                   help="Actual control LEFT frame range")
    p.add_argument("--actual-control-right-range", nargs=2, type=int,
                   metavar=("START", "END"),
                   help="Actual control RIGHT frame range")

    # Actual indices — explicit lists
    p.add_argument("--actual-exp-left-indices",     nargs="+", type=int,
                   metavar="I", help="Actual exp LEFT explicit index list")
    p.add_argument("--actual-exp-right-indices",    nargs="+", type=int,
                   metavar="I", help="Actual exp RIGHT explicit index list")
    p.add_argument("--actual-control-left-indices", nargs="+", type=int,
                   metavar="I", help="Actual control LEFT explicit index list")
    p.add_argument("--actual-control-right-indices",nargs="+", type=int,
                   metavar="I", help="Actual control RIGHT explicit index list")

    # DBSCAN
    p.add_argument("--dbscan-eps-mm", default="auto",
                   help="DBSCAN eps in mm, or 'auto' (default: auto)")
    p.add_argument("--dbscan-min-samples", type=int, default=3,
                   help="DBSCAN min_samples (default: 3)")

    # Output
    p.add_argument("--outdir", default="reports",
                   help="Output directory (default: reports/)")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# YAML CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_yaml_config(config_path: str) -> Dict:
    """Load YAML config and return as dict with CLI-style keys (using _)."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for --config. pip install pyyaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # normalise: yaml uses underscores, argparse uses hyphens internally
    return cfg


def merge_config_into_args(args: argparse.Namespace,
                            cfg: Dict) -> argparse.Namespace:
    """
    Fill in missing argparse values from YAML config.
    CLI flags take priority over YAML values.
    YAML keys use underscores; argparse attributes also use underscores.
    """
    # mapping: yaml key → argparse dest
    key_map = {
        "subject":                          "subject",
        "planned_exp":                      "planned_exp",
        "planned_control":                  "planned_control",
        "actual_exp":                       "actual_exp",
        "actual_control":                   "actual_control",
        "planned_exp_left_index":           "planned_exp_left_index",
        "planned_exp_right_index":          "planned_exp_right_index",
        "planned_control_left_index":       "planned_control_left_index",
        "planned_control_right_index":      "planned_control_right_index",
        "actual_exp_left_range":            "actual_exp_left_range",
        "actual_exp_right_range":           "actual_exp_right_range",
        "actual_control_left_range":        "actual_control_left_range",
        "actual_control_right_range":       "actual_control_right_range",
        "actual_exp_left_indices":          "actual_exp_left_indices",
        "actual_exp_right_indices":         "actual_exp_right_indices",
        "actual_control_left_indices":      "actual_control_left_indices",
        "actual_control_right_indices":     "actual_control_right_indices",
        "dbscan_eps_mm":                    "dbscan_eps_mm",
        "dbscan_min_samples":               "dbscan_min_samples",
        "outdir":                           "outdir",
    }
    for yaml_key, arg_dest in key_map.items():
        if yaml_key in cfg and getattr(args, arg_dest, None) is None:
            val = cfg[yaml_key]
            # convert range lists to plain Python lists if needed
            if isinstance(val, list) and "range" in yaml_key:
                val = [int(v) for v in val]
            setattr(args, arg_dest, val)
    return args


# ══════════════════════════════════════════════════════════════════════════════
# INDEX RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_indices(range_arg: Optional[List[int]],
                    explicit_arg: Optional[List[int]],
                    label: str) -> List[int]:
    """
    Return a list of XML element indices to include.

    Raises ValueError if both range and explicit are provided.
    """
    if range_arg is not None and explicit_arg is not None:
        raise ValueError(
            f"[{label}] Both --*-range and --*-indices provided. "
            "Use one or the other."
        )
    if explicit_arg is not None:
        return sorted(explicit_arg)
    if range_arg is not None:
        lo, hi = range_arg
        return list(range(lo, hi + 1))
    raise ValueError(
        f"[{label}] No index selection provided. "
        "Use either --*-range START END  or  --*-indices i1 i2 ..."
    )


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_args(args: argparse.Namespace) -> None:
    """Check that all required arguments are present and files exist."""
    required_fields = [
        "subject",
        "actual_exp", "actual_control",
        "planned_exp_left_index", "planned_exp_right_index",
        "planned_control_left_index", "planned_control_right_index",
    ]
    # planned xml files default to actual if not provided
    for field in required_fields:
        if getattr(args, field, None) is None:
            raise ValueError(
                f"Missing required argument: --{field.replace('_','-')}"
            )

    for xml_attr in ["planned_exp", "planned_control",
                     "actual_exp", "actual_control"]:
        path = getattr(args, xml_attr, None)
        if path and not Path(path).exists():
            raise FileNotFoundError(f"XML file not found: {path}")

    # warn if planned XMLs not given (will reuse actual)
    if not args.planned_exp:
        log.warning("--planned-exp not given; using --actual-exp for planned refs too.")
        args.planned_exp = args.actual_exp
    if not args.planned_control:
        log.warning("--planned-control not given; using --actual-control.")
        args.planned_control = args.actual_control


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(args: argparse.Namespace) -> None:
    subject = args.subject
    eps     = (args.dbscan_eps_mm if isinstance(args.dbscan_eps_mm, str)
               else float(args.dbscan_eps_mm))
    if isinstance(eps, str) and eps.lower() != "auto":
        eps = float(eps)
    min_s = int(args.dbscan_min_samples)

    # ── output directories ────────────────────────────────────────────────────
    outdir  = Path(args.outdir)
    fig_dir = outdir / "figures" / subject
    outdir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", outdir)

    # ── parse XMLs ────────────────────────────────────────────────────────────
    log.info("Parsing planned-exp XML …")
    planned_exp_df = parse_gummarker_xml(args.planned_exp)
    log.info("Parsing planned-control XML …")
    planned_ctrl_df = parse_gummarker_xml(args.planned_control)
    log.info("Parsing actual-exp XML …")
    actual_exp_df = parse_gummarker_xml(args.actual_exp)
    log.info("Parsing actual-control XML …")
    actual_ctrl_df = parse_gummarker_xml(args.actual_control)

    # ── reference rows ────────────────────────────────────────────────────────
    def get_ref(df: pd.DataFrame, idx: int, label: str) -> pd.Series:
        rows = df[df["frame"] == idx]
        if rows.empty:
            raise ValueError(
                f"[{label}] Planned index {idx} not found in DataFrame. "
                f"Available frames: {sorted(df['frame'].tolist())}"
            )
        return rows.iloc[0]

    ref = {
        "exp-left":     get_ref(planned_exp_df,  args.planned_exp_left_index,     "exp-left"),
        "exp-right":    get_ref(planned_exp_df,  args.planned_exp_right_index,    "exp-right"),
        "control-left": get_ref(planned_ctrl_df, args.planned_control_left_index, "control-left"),
        "control-right":get_ref(planned_ctrl_df, args.planned_control_right_index,"control-right"),
    }

    # ── resolve actual indices ────────────────────────────────────────────────
    indices = {
        "exp-left":     resolve_indices(args.actual_exp_left_range,
                                        args.actual_exp_left_indices,
                                        "exp-left"),
        "exp-right":    resolve_indices(args.actual_exp_right_range,
                                        args.actual_exp_right_indices,
                                        "exp-right"),
        "control-left": resolve_indices(args.actual_control_left_range,
                                        args.actual_control_left_indices,
                                        "control-left"),
        "control-right":resolve_indices(args.actual_control_right_range,
                                        args.actual_control_right_indices,
                                        "control-right"),
    }

    actual_dfs = {
        "exp-left":     actual_exp_df,
        "exp-right":    actual_exp_df,
        "control-left": actual_ctrl_df,
        "control-right":actual_ctrl_df,
    }

    # ── run analysis per condition ────────────────────────────────────────────
    results: Dict = {}
    for lbl in ["exp-left", "exp-right", "control-left", "control-right"]:
        log.info("Analysing %s …", lbl)
        results[lbl] = analyse_condition(
            actual_df  = actual_dfs[lbl],
            ref_row    = ref[lbl],
            indices    = indices[lbl],
            eps        = eps,
            min_samples= min_s,
            label      = lbl,
        )

    # ── figures ───────────────────────────────────────────────────────────────
    figs_time    = {}
    figs_spatial = {}

    for lbl in results:
        slug = lbl.replace("-", "_")

        tp = str(fig_dir / f"positions_over_time_{slug}.png")
        fig_positions_over_time(results[lbl], lbl, tp)
        figs_time[lbl] = os.path.relpath(tp, outdir)

        sp = str(fig_dir / f"spatial_clusters_{slug}.png")
        fig_spatial_clusters(results[lbl], lbl, ref[lbl], sp)
        figs_spatial[lbl] = os.path.relpath(sp, outdir)

    fig_cls_path  = str(fig_dir / "cluster_size_summary.png")
    fig_disp_path = str(fig_dir / "displacement_summary.png")
    fig_cluster_size_summary(results, fig_cls_path)
    fig_displacement_summary(results, fig_disp_path)

    # ── CSV outputs ───────────────────────────────────────────────────────────
    # all positions with displacement
    combined = []
    for lbl, res in results.items():
        df_out = res["df"].copy()
        df_out["condition"] = lbl
        df_out["cluster"]   = res["labels"]
        df_out["is_medoid"] = False
        df_out.loc[res["df"].index[res["medoid_idx"]], "is_medoid"] = True
        combined.append(df_out)
    all_positions_df = pd.concat(combined, ignore_index=True)
    pos_csv = outdir / f"{subject}_actual_positions_with_displacement.csv"
    all_positions_df.to_csv(pos_csv, index=False)
    log.info("Saved positions CSV: %s", pos_csv)

    # summary CSV
    summary_rows = []
    for lbl, res in results.items():
        s = res["stats"]
        summary_rows.append({
            "subject":          subject,
            "condition":        lbl,
            "n_total":          s["n_total"],
            "n_clusters":       s["n_clusters"],
            "n_noise":          s["n_noise"],
            "best_cluster_id":  s["best_cluster"],
            "best_cluster_n":   s["best_cluster_n"],
            "eps_used_mm":      round(s["eps_used"], 4),
            "medoid_frame":     s["medoid_frame"],
            "medoid_x_mm":      round(s["medoid_x"], 4),
            "medoid_y_mm":      round(s["medoid_y"], 4),
            "medoid_z_mm":      round(s["medoid_z"], 4),
            "medoid_disp_mm":   round(s["medoid_disp_mm"], 3),
            "medoid_ang_deg":   round(s["medoid_ang_deg"], 3),
            "disp_mean_all_mm": round(s["disp_mean_all"], 3),
            "disp_sd_all_mm":   round(s["disp_sd_all"],   3),
            "disp_max_all_mm":  round(s["disp_max_all"],  3),
            "disp_mean_cls_mm": round(s["disp_mean_cls"], 3),
            "disp_sd_cls_mm":   round(s["disp_sd_cls"],   3),
            "ang_mean_all_deg": round(s["ang_mean_all"],  3),
            "ang_sd_all_deg":   round(s["ang_sd_all"],    3),
            "ang_max_all_deg":  round(s["ang_max_all"],   3),
            "ang_mean_cls_deg": round(s["ang_mean_cls"],  3),
            "ang_sd_cls_deg":   round(s["ang_sd_cls"],    3),
            "used_fallback":    s["used_fallback"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summ_csv = outdir / f"{subject}_transducer_position_summary.csv"
    summary_df.to_csv(summ_csv, index=False)
    log.info("Saved summary CSV: %s", summ_csv)

    # ── HTML report ───────────────────────────────────────────────────────────
    overview = {
        "Planned exp XML":          args.planned_exp,
        "Planned control XML":      args.planned_control,
        "Actual exp XML":           args.actual_exp,
        "Actual control XML":       args.actual_control,
        "Planned exp LEFT index":   args.planned_exp_left_index,
        "Planned exp RIGHT index":  args.planned_exp_right_index,
        "Planned ctrl LEFT index":  args.planned_control_left_index,
        "Planned ctrl RIGHT index": args.planned_control_right_index,
        "Actual exp LEFT selection":    _fmt_indices(args.actual_exp_left_range,
                                                     args.actual_exp_left_indices),
        "Actual exp RIGHT selection":   _fmt_indices(args.actual_exp_right_range,
                                                     args.actual_exp_right_indices),
        "Actual ctrl LEFT selection":   _fmt_indices(args.actual_control_left_range,
                                                     args.actual_control_left_indices),
        "Actual ctrl RIGHT selection":  _fmt_indices(args.actual_control_right_range,
                                                     args.actual_control_right_indices),
        "DBSCAN eps":               str(eps),
        "DBSCAN min_samples":       str(min_s),
    }

    html_path = str(outdir / f"{subject}_transducer_position_QC_report.html")
    build_html_report(
        subject=subject,
        overview=overview,
        all_stats={lbl: res["stats"] for lbl, res in results.items()},
        results=results,
        figs_time=figs_time,
        figs_spatial=figs_spatial,
        fig_cluster_summary=os.path.relpath(fig_cls_path,  outdir),
        fig_displacement_summary=os.path.relpath(fig_disp_path, outdir),
        out_path=html_path,
    )

    # ── terminal summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  QC REPORT — {subject}")
    print("=" * 64)
    print(f"  {'Condition':<16} {'Medoid frame':>12}  "
          f"{'Trans (mm)':>12}  {'Angle (°)':>10}")
    print("  " + "─" * 58)
    for lbl, res in results.items():
        s = res["stats"]
        flag = " ⚠ FALLBACK" if s["used_fallback"] else ""
        print(f"  {lbl:<16} {s['medoid_frame']:>12d}  "
              f"{s['medoid_disp_mm']:>11.3f}  "
              f"{s['medoid_ang_deg']:>9.3f}°{flag}")
    print("=" * 64)
    print(f"\n  HTML report : {html_path}")
    print(f"  Summary CSV : {summ_csv}")
    print(f"  Positions   : {pos_csv}")
    print(f"  Figures     : {fig_dir}/\n")


def _fmt_indices(range_arg, explicit_arg) -> str:
    if explicit_arg is not None:
        n = len(explicit_arg)
        return f"explicit list ({n} indices)"
    if range_arg is not None:
        lo, hi = range_arg
        return f"range [{lo}, {hi}] ({hi-lo+1} frames)"
    return "not set"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── inspect-xml sub-command ───────────────────────────────────────────────
    if args.command == "inspect-xml":
        print(inspect_xml(args.xml_file))
        return

    # ── load YAML config if given ─────────────────────────────────────────────
    if args.config:
        cfg  = load_yaml_config(args.config)
        args = merge_config_into_args(args, cfg)

    # ── validate ──────────────────────────────────────────────────────────────
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    # ── run ───────────────────────────────────────────────────────────────────
    try:
        run_analysis(args)
    except Exception as e:
        log.error("Analysis failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
