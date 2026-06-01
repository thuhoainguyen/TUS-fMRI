"""Self-contained HTML report with sticky TOC and scroll-through sections."""

from __future__ import annotations

import base64
import html
import math
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from ok_plan import viz
from ok_plan.atlas import (
    AtlasRegionStats,
    atlas_label_for_roi,
    atlas_region_overlap,
    load_and_warp_julich_atlas,
    region_mask_img,
)
from ok_plan.focus import build_focus_masks_amplitude_db
from ok_plan.geometry import assert_same_space
from ok_plan.focus_roi import FocusRoiPressureStats, focus_roi_pressure_stats
from ok_plan.metrics import MaskSafetyMetrics, safety_metrics_for_mask
from ok_plan.nii_utils import (
    max_delta_t_coords_mm,
    pa_to_isppa_w_per_cm2,
    pa_to_mpa,
    pressure_img_mpa_from_pa,
    roi_centroid_mm,
)
from ok_plan.tissues import find_final_tissues

DEFAULT_METHODOLOGY_REFERENCE_URL = "https://doi.org/10.1016/j.brs.2025.10.007"

# ITRUSST non-significant risk thresholds (Aubry et al., Brain Stimulation 2025)
ITRUSST_MI_NSR = 1.9
ITRUSST_MI_EYE_NSR = 0.23  # FDA ophthalmic limit (MI ≤ 1.9 for all except the eye)
ITRUSST_DELTA_T_NSR_C = 2.0
ITRUSST_T_ABS_NSR_C = 39.0
ITRUSST_CEM43_BRAIN_NSR = 2.0
ITRUSST_CEM43_BONE_NSR = 16.0
ITRUSST_CEM43_SKIN_NSR = 21.0

_MASK_KEY_TO_CEM43_THRESHOLD: dict[str, float] = {
    "scalp": ITRUSST_CEM43_SKIN_NSR,
    "skull": ITRUSST_CEM43_BONE_NSR,
    "inside_skull": ITRUSST_CEM43_BRAIN_NSR,
    "eyes": ITRUSST_CEM43_BRAIN_NSR,
}

_MASK_KEY_TO_MI_THRESHOLD: dict[str, float | None] = {
    "scalp": ITRUSST_MI_NSR,
    "skull": None,  # MI not applicable for skull
    "inside_skull": ITRUSST_MI_NSR,
    "eyes": ITRUSST_MI_EYE_NSR,
}

_MASK_KEY_TO_DISPLAY: dict[str, str] = {
    "scalp": "Scalp",
    "skull": "Skull",
    "inside_skull": "Brain+",
    "eyes": "Eyes",
}

_MASK_ORDER = ["scalp", "skull", "inside_skull", "eyes"]


def _resolved_methodology_url(url: str | None) -> str | None:
    """``None`` -> built-in DOI; empty string -> disable links."""
    if url is None:
        return DEFAULT_METHODOLOGY_REFERENCE_URL
    if url == "":
        return None
    return url


def _b64_png(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Low-level HTML helpers
# ---------------------------------------------------------------------------

def _sub_panel(section_id: str, title: str, inner_html: str) -> str:
    """A sub-panel rendered as an h3 block inside a group."""
    sid = html.escape(section_id, quote=True)
    return (
        f'<div id="{sid}" class="sub-panel">'
        f"<h3>{title}</h3>"
        f"{inner_html}"
        "</div>"
    )


def _sub_panel_img(section_id: str, title: str, png: bytes, caption: str = "") -> str:
    cap = (
        f'<p class="caption">{caption}</p>'
        if caption
        else ""
    )
    sid = html.escape(section_id, quote=True)
    return (
        f'<div id="{sid}" class="sub-panel">'
        f"<h3>{html.escape(title)}</h3>"
        f'<figure><img alt="" src="data:image/png;base64,{_b64_png(png)}">{cap}</figure>'
        "</div>"
    )


def _group_section(section_id: str, title: str, inner_html: str) -> str:
    sid = html.escape(section_id, quote=True)
    return (
        f'<section id="{sid}" class="panel group-panel">'
        f"<h2>{html.escape(title)}</h2>"
        f"{inner_html}"
        "</section>"
    )


def _panel_text(section_id: str, title: str, inner_html: str) -> str:
    sid = html.escape(section_id, quote=True)
    return (
        f'<section id="{sid}" class="panel">'
        f"<h2>{html.escape(title)}</h2>"
        f"{inner_html}"
        "</section>"
    )


def _meta_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(k), html.escape(v, quote=True)
        )
        for k, v in rows
    )
    return f'<table class="meta"><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_num(x: float, *, prec: int = 4) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{prec}g}"


def _fmt_mi(r: MaskSafetyMetrics) -> str:
    if _MASK_KEY_TO_MI_THRESHOLD.get(r.mask_key) is None:
        return "NA"
    return _fmt_num(r.mi)


def _fmt_scientific_html(x: float, *, nd: int = 4) -> str:
    """Scientific notation as HTML ``M &times; 10<sup>e</sup>`` (unicode minus).

    Plain ``1.2345e-06`` causes column widths to jump in narrow summary/table
    layouts; the superscript form renders compactly and consistently.
    """
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "—"
    v = float(x)
    if v == 0.0:
        return "0"
    mantissa, exp = f"{v:.{nd}e}".split("e")
    exp_int = int(exp)
    sign = "\u2212" if exp_int < 0 else ""
    return f"{mantissa}&nbsp;&times;&nbsp;10<sup>{sign}{abs(exp_int)}</sup>"


def _fmt_cem43(x: float) -> str:
    """CEM43 columns: scientific (× 10^e) so small doses are visible."""
    return _fmt_scientific_html(x, nd=4)


def _fmt_mpa(pa: float) -> str:
    """Pressure in MPa from an internal Pa value (4 significant digits)."""
    return _fmt_num(pa_to_mpa(pa))


def _fmt_isppa(pa: float) -> str:
    """Isppa in W/cm² from an internal Pa value (3 decimal places)."""
    v = pa_to_isppa_w_per_cm2(pa)
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return "—"
    return f"{float(v):.3f}"


def _status_badge(value: float, threshold: float, *, na: bool = False) -> str:
    """Green/red badge comparing a metric against its ITRUSST NSR threshold."""
    if na:
        return '<span class="badge badge-na">NA</span>'
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return '<span class="badge badge-na">—</span>'
    if float(value) <= threshold:
        return '<span class="badge badge-ok">OK</span>'
    return '<span class="badge badge-warn">CHECK</span>'


# ---------------------------------------------------------------------------
# TOC (nested)
# ---------------------------------------------------------------------------

def _toc_nav(groups: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
    """Build a nested TOC.

    Each item in *groups* is ``(group_id, group_label, [(child_id, child_label), ...])``.
    If ``group_id`` is empty the children are rendered as top-level items.
    """
    parts: list[str] = []
    for gid, glabel, children in groups:
        if not gid:
            for cid, clabel in children:
                parts.append(
                    f'<li><a href="#{html.escape(cid, quote=True)}">'
                    f"{html.escape(clabel)}</a></li>"
                )
        else:
            sub = "".join(
                f'<li><a href="#{html.escape(cid, quote=True)}">'
                f"{html.escape(clabel)}</a></li>"
                for cid, clabel in children
            )
            parts.append(
                f'<li class="toc-group"><a href="#{html.escape(gid, quote=True)}">'
                f"{html.escape(glabel)}</a>"
                f"<ul>{sub}</ul></li>"
            )
    return (
        '<nav class="toc" aria-label="Report sections">'
        '<p class="toc-title">Contents</p>'
        f'<ul>{"".join(parts)}</ul>'
        "</nav>"
    )


# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------

def _summary_card_section(
    safety_rows: list[MaskSafetyMetrics],
    stats3: FocusRoiPressureStats,
    stats6: FocusRoiPressureStats | None = None,
    *,
    include_6db: bool = False,
) -> str:
    """At-a-glance summary card: mechanical, thermal, and targeting metrics."""

    scalp = next((r for r in safety_rows if r.mask_key == "scalp"), None)
    inside = next((r for r in safety_rows if r.mask_key == "inside_skull"), None)
    skull = next((r for r in safety_rows if r.mask_key == "skull"), None)
    eyes = next((r for r in safety_rows if r.mask_key == "eyes"), None)

    mi_scalp = scalp.mi if scalp else float("nan")
    mi_scalp_badge = _status_badge(mi_scalp, ITRUSST_MI_NSR)

    mi_val = inside.mi if inside else float("nan")
    mi_badge = _status_badge(mi_val, ITRUSST_MI_NSR)

    mi_eye = eyes.mi if eyes else float("nan")
    mi_eye_badge = _status_badge(mi_eye, ITRUSST_MI_EYE_NSR)

    dt_max = max((r.temp_rise_max for r in safety_rows), default=float("nan"))
    dt_badge = _status_badge(dt_max, ITRUSST_DELTA_T_NSR_C)

    t_abs_max = max((r.temp_abs_max for r in safety_rows), default=float("nan"))
    t_abs_badge = _status_badge(t_abs_max, ITRUSST_T_ABS_NSR_C)

    cem_brain = inside.cem43_max if inside else float("nan")
    cem_brain_badge = _status_badge(cem_brain, ITRUSST_CEM43_BRAIN_NSR)

    cem_bone = skull.cem43_max if skull else float("nan")
    cem_bone_badge = _status_badge(cem_bone, ITRUSST_CEM43_BONE_NSR)

    mech_html = (
        '<div class="card">'
        "<h3>Mechanical safety</h3>"
        '<table class="summary-tbl">'
        f"<tr><td>MI (Scalp)</td><td><strong>{_fmt_num(mi_scalp)}</strong></td>"
        f"<td>&le; {ITRUSST_MI_NSR}</td><td>{mi_scalp_badge}</td></tr>"
        f"<tr><td>MI (Brain+)</td><td><strong>{_fmt_num(mi_val)}</strong></td>"
        f"<td>&le; {ITRUSST_MI_NSR}</td><td>{mi_badge}</td></tr>"
        f"<tr><td>MI (Eyes)</td><td><strong>{_fmt_num(mi_eye)}</strong></td>"
        f"<td>&le; {ITRUSST_MI_EYE_NSR}</td><td>{mi_eye_badge}</td></tr>"
        "</table></div>"
    )

    thermal_html = (
        '<div class="card">'
        "<h3>Thermal safety</h3>"
        '<table class="summary-tbl">'
        f"<tr><td>&Delta;T max</td><td><strong>{_fmt_num(dt_max)} &deg;C</strong></td>"
        f"<td>&le; {ITRUSST_DELTA_T_NSR_C} &deg;C</td><td>{dt_badge}</td></tr>"
        f"<tr><td><em>T</em> abs max</td><td><strong>{_fmt_num(t_abs_max)} &deg;C</strong></td>"
        f"<td>&le; {ITRUSST_T_ABS_NSR_C} &deg;C</td><td>{t_abs_badge}</td></tr>"
        f"<tr><td>CEM43 max (brain)</td><td><strong>{_fmt_cem43(cem_brain)}</strong></td>"
        f"<td>&le; {ITRUSST_CEM43_BRAIN_NSR}</td><td>{cem_brain_badge}</td></tr>"
        f"<tr><td>CEM43 max (bone)</td><td><strong>{_fmt_cem43(cem_bone)}</strong></td>"
        f"<td>&le; {ITRUSST_CEM43_BONE_NSR}</td><td>{cem_bone_badge}</td></tr>"
        "</table></div>"
    )

    target_rows = (
        f"<tr><td>&minus;3 dB target coverage</td>"
        f"<td><strong>{stats3.overlap_pct_of_roi:.1f}%</strong> of ROI</td>"
        f"<td colspan='2'>On-target {stats3.on_target_pct:.1f}%</td></tr>"
    )
    if include_6db and stats6 is not None:
        target_rows += (
            f"<tr><td>&minus;6 dB target coverage</td>"
            f"<td><strong>{stats6.overlap_pct_of_roi:.1f}%</strong> of ROI</td>"
            f"<td colspan='2'>On-target {stats6.on_target_pct:.1f}%</td></tr>"
        )
    target_rows += (
        f"<tr><td><em>P</em> max at target</td>"
        f"<td><strong>{_fmt_mpa(stats3.p_max_in_roi_pa)} MPa</strong></td>"
        f"<td colspan='2'>I<sub>sppa</sub> {_fmt_isppa(stats3.p_max_in_roi_pa)} W/cm&sup2;</td></tr>"
    )

    target_html = (
        '<div class="card">'
        "<h3>Targeting efficacy</h3>"
        f'<table class="summary-tbl">{target_rows}</table></div>'
    )

    return (
        '<section id="sec-summary" class="panel summary-panel">'
        "<h2>Summary</h2>"
        "<p>Key metrics versus ITRUSST non-significant risk (NSR) thresholds "
        "(Aubry et al., <em>Brain Stimulation</em> 2025).</p>"
        '<div class="card-row">' + mech_html + thermal_html + "</div>"
        '<div class="card-row">' + target_html + "</div>"
        "</section>"
    )


# ---------------------------------------------------------------------------
# Safety metrics section (with ITRUSST threshold row)
# ---------------------------------------------------------------------------

def _mi_threshold_for_display(mask_key: str) -> str:
    thr = _MASK_KEY_TO_MI_THRESHOLD.get(mask_key)
    if thr is None:
        return "NA"
    return f"&le; {thr}"


def _safety_metrics_section(
    rows: list[MaskSafetyMetrics],
    *,
    center_frequency_mhz: float,
    methodology_reference_url: str | None,
    baseline_body_temp_c: float,
    exposure_duration_min: float,
) -> str:
    """HTML sub-panel: MI, CEM43 max, ΔT max, T abs max per mask with ITRUSST thresholds."""
    body: list[str] = []
    for r in rows:
        cem_thr = _MASK_KEY_TO_CEM43_THRESHOLD.get(r.mask_key, ITRUSST_CEM43_BRAIN_NSR)
        mi_thr = _MASK_KEY_TO_MI_THRESHOLD.get(r.mask_key)
        mi_na = mi_thr is None
        display_name = _MASK_KEY_TO_DISPLAY.get(r.mask_key, r.mask_name)
        body.append(
            "<tr>"
            f"<td>{html.escape(display_name)}</td>"
            f"<td>{_fmt_mi(r)}</td>"
            f"<td>{_mi_threshold_for_display(r.mask_key)}</td>"
            f"<td>{_status_badge(r.mi, mi_thr or 0.0, na=mi_na)}</td>"
            f"<td>{_fmt_cem43(r.cem43_max)}</td>"
            f"<td>&le; {cem_thr}</td>"
            f"<td>{_status_badge(r.cem43_max, cem_thr)}</td>"
            f"<td>{_fmt_num(r.temp_rise_max)}</td>"
            f"<td>&le; {ITRUSST_DELTA_T_NSR_C}</td>"
            f"<td>{_status_badge(r.temp_rise_max, ITRUSST_DELTA_T_NSR_C)}</td>"
            f"<td>{_fmt_num(r.temp_abs_max)}</td>"
            f"<td>&le; {ITRUSST_T_ABS_NSR_C}</td>"
            f"<td>{_status_badge(r.temp_abs_max, ITRUSST_T_ABS_NSR_C)}</td>"
            "</tr>"
        )
    table = (
        '<table class="stats safety">'
        "<thead><tr>"
        "<th>Mask</th>"
        "<th>MI</th><th>NSR</th><th></th>"
        "<th>CEM43 max<br><small>(equiv. min)</small></th><th>NSR</th><th></th>"
        "<th>&Delta;T max<br><small>(&deg;C)</small></th><th>NSR</th><th></th>"
        "<th><em>T</em> abs max<br><small>(&deg;C)</small></th><th>NSR</th><th></th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )
    lit = ""
    if methodology_reference_url:
        esc = html.escape(methodology_reference_url, quote=True)
        lit = (
            '<p class="caption">Cross-check against your protocol and '
            f'<a href="{esc}">ITRUSST guidelines (Brain Stimulation 2025)</a>.</p>'
        )
    foot = (
        f'<p class="caption"><strong>MI</strong> = |p|_max (MPa) / &radic;f, '
        f"f = {html.escape(str(center_frequency_mhz))} MHz. "
        "Skull: MI not applicable. Eyes: FDA ophthalmic limit MI &le; 0.23. "
        "<strong>CEM43</strong> = cumulative equivalent minutes at 43 &deg;C "
        "(Sapareto&ndash;Dewey thermal-dose model, same formulation as k-Wave / k-Plan). "
        "For each voxel, "
        "CEM43 = &int; R<sup>(43&minus;T)</sup> dt with R = 0.25 for T &ge; 43 &deg;C "
        "and R = 0.5 for T &lt; 43 &deg;C. "
        "Under a single-time-point steady-state assumption, each voxel is held at its "
        "mapped absolute temperature for the full exposure "
        f"({html.escape(str(exposure_duration_min))} min), so "
        "CEM43<sub>max</sub> = exposure &times; R<sup>(43&minus;T<sub>max</sub>)</sup>. "
        "NSR &le; 2 (brain), &le; 16 (bone), &le; 21 (skin). "
        f"<strong>&Delta;T</strong> = T &minus; {html.escape(str(baseline_body_temp_c))} &deg;C.</p>"
        + lit
    )
    return (
        '<div id="sec-safety-metrics" class="sub-panel">'
        "<h3>MI, CEM43, and &Delta;T by tissue mask</h3>"
        + table + foot
        + "</div>"
    )


# ---------------------------------------------------------------------------
# ROI / focus overlap section (split tables)
# ---------------------------------------------------------------------------

def _roi_focus_geometry_table(stats_list: list[FocusRoiPressureStats]) -> str:
    """Table A: overlap geometry."""
    def row(s: FocusRoiPressureStats) -> str:
        return (
            "<tr>"
            f"<td>{html.escape(s.label)}</td>"
            f"<td>{s.overlap_pct_of_roi:.1f}</td>"
            f"<td>{s.on_target_pct:.1f}</td>"
            f"<td>{s.off_target_pct:.1f}</td>"
            f"<td>{s.overlap_vol_mm3:.1f}</td>"
            "</tr>"
        )
    return (
        '<table class="stats">'
        "<thead><tr>"
        "<th>Focus</th>"
        "<th>Target coverage<br><small>(% ROI in focus)</small></th>"
        "<th>On-target<br><small>(% focus on ROI)</small></th>"
        "<th>Off-target<br><small>(% focus off ROI)</small></th>"
        "<th>Overlap volume<br><small>(mm&sup3;)</small></th>"
        "</tr></thead><tbody>"
        + "".join(row(s) for s in stats_list)
        + "</tbody></table>"
    )


def _roi_focus_pressure_table(
    stats_list: list[FocusRoiPressureStats],
    *,
    title: str,
    p_max_attr: str,
    p_mean_attr: str,
) -> str:
    """Table B or C: pressure stats for a given zone."""
    def row(s: FocusRoiPressureStats) -> str:
        pmax = getattr(s, p_max_attr)
        pmean = getattr(s, p_mean_attr)
        return (
            "<tr>"
            f"<td>{html.escape(s.label)}</td>"
            f"<td>{_fmt_mpa(pmax)}</td>"
            f"<td>{_fmt_isppa(pmax)}</td>"
            f"<td>{_fmt_mpa(pmean)}</td>"
            f"<td>{_fmt_isppa(pmean)}</td>"
            "</tr>"
        )
    return (
        f"<h4>{html.escape(title)}</h4>"
        '<table class="stats">'
        "<thead><tr>"
        "<th>Focus</th>"
        "<th><em>P</em> max<br><small>(MPa)</small></th>"
        "<th>I<sub>sppa</sub> max<br><small>(W/cm&sup2;)</small></th>"
        "<th><em>P</em> mean<br><small>(MPa)</small></th>"
        "<th>I<sub>sppa</sub> mean<br><small>(W/cm&sup2;)</small></th>"
        "</tr></thead><tbody>"
        + "".join(row(s) for s in stats_list)
        + "</tbody></table>"
    )


def _roi_focus_overlap_tables_html(
    stats3: FocusRoiPressureStats,
    stats6: FocusRoiPressureStats | None = None,
    *,
    include_6db: bool = False,
) -> str:
    """Target-coverage geometry + pressure-in-overlap tables for \u22123/\u22126 dB focus."""
    stats_list = [stats3]
    if include_6db and stats6 is not None:
        stats_list.append(stats6)

    geom = _roi_focus_geometry_table(stats_list)
    p_overlap = _roi_focus_pressure_table(
        stats_list,
        title="Pressure in overlap (focus \u2229 ROI)",
        p_max_attr="p_max_in_overlap_pa",
        p_mean_attr="p_mean_in_overlap_pa",
    )

    return (
        '<div id="sec-roi-tables" class="sub-panel">'
        + "<h3>Overlap and pressure statistics</h3>"
        + '<p class="caption">'
        "<strong>Target coverage</strong> = overlap / ROI voxels. "
        "<strong>On-target</strong> = overlap / focus voxels. "
        "I<sub>sppa</sub> = p&sup2;/(2&rho;c), "
        "&rho; = 1000 kg/m&sup3;, c = 1500 m/s.</p>"
        + geom + p_overlap
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Atlas section
# ---------------------------------------------------------------------------

def _atlas_region_table(
    rows: list[AtlasRegionStats],
    *,
    focus_label: str,
    roi_stats: FocusRoiPressureStats | None = None,
    min_pct: float = 1.0,
    target_region_name: str | None = None,
) -> str:
    """fMRI-style table of atlas regions overlapping a focus mask (outside ROI).

    Sorted by ``pct_region_in_focus`` descending.  A highlighted target ROI row
    is inserted at the top when *roi_stats* is provided. When
    ``target_region_name`` is set, that label (with a ``*`` marker) is shown in
    place of the generic ``Target ROI`` text.
    """
    filtered = sorted(
        [r for r in rows if r.pct_region_in_focus > min_pct],
        key=lambda r: r.pct_region_in_focus,
        reverse=True,
    )

    roi_row = ""
    if roi_stats is not None:
        if target_region_name:
            label_html = f"<strong>{html.escape(target_region_name)}</strong>*"
        else:
            label_html = "<strong>Target ROI</strong>"
        roi_row = (
            '<tr class="roi-highlight">'
            f"<td>{label_html}</td>"
            f"<td>{roi_stats.overlap_pct_of_roi:.1f}</td>"
            f"<td>{roi_stats.n_overlap_voxels:,}</td>"
            f"<td>{_fmt_mpa(roi_stats.p_max_in_roi_pa)}</td>"
            f"<td>{_fmt_isppa(roi_stats.p_max_in_roi_pa)}</td>"
            f"<td>{_fmt_mpa(roi_stats.p_min_in_roi_pa)}</td>"
            f"<td>{_fmt_isppa(roi_stats.p_min_in_roi_pa)}</td>"
            f"<td>{_fmt_mpa(roi_stats.p_mean_in_roi_pa)}</td>"
            f"<td>{_fmt_isppa(roi_stats.p_mean_in_roi_pa)}</td>"
            "</tr>"
        )

    if not filtered and not roi_row:
        return (
            f"<p>No Julich atlas regions have &gt;{min_pct:.0f}% overlap with the "
            f"{html.escape(focus_label)} focus outside the target ROI.</p>"
        )
    body = "".join(
        "<tr>"
        f"<td>{html.escape(r.region_name)}</td>"
        f"<td>{r.pct_region_in_focus:.1f}</td>"
        f"<td>{r.n_overlap_voxels:,}</td>"
        f"<td>{_fmt_mpa(r.p_max_pa)}</td>"
        f"<td>{_fmt_isppa(r.p_max_pa)}</td>"
        f"<td>{_fmt_mpa(r.p_min_pa)}</td>"
        f"<td>{_fmt_isppa(r.p_min_pa)}</td>"
        f"<td>{_fmt_mpa(r.p_mean_pa)}</td>"
        f"<td>{_fmt_isppa(r.p_mean_pa)}</td>"
        "</tr>"
        for r in filtered
    )
    return (
        '<table class="stats">'
        "<thead><tr>"
        "<th>Region</th>"
        "<th>% in focus</th>"
        "<th>Overlap voxels</th>"
        "<th><em>P</em> max<br><small>(MPa)</small></th>"
        "<th>I<sub>sppa</sub> max<br><small>(W/cm&sup2;)</small></th>"
        "<th><em>P</em> min<br><small>(MPa)</small></th>"
        "<th>I<sub>sppa</sub> min<br><small>(W/cm&sup2;)</small></th>"
        "<th><em>P</em> mean<br><small>(MPa)</small></th>"
        "<th>I<sub>sppa</sub> mean<br><small>(W/cm&sup2;)</small></th>"
        "</tr></thead><tbody>"
        + roi_row + body
        + "</tbody></table>"
    )


def _legend_swatches_html(
    legend: list[tuple[str, tuple[float, float, float]]],
    *,
    focus_label: str,
    focus_rgb: tuple[float, float, float],
) -> str:
    """Inline HTML swatch legend for the combined focus+regions overlay."""
    def _swatch(rgb: tuple[float, float, float], label: str) -> str:
        r, g, b = [int(round(255 * v)) for v in rgb[:3]]
        return (
            "<span style='display:inline-flex;align-items:center;"
            "margin-right:1rem;margin-bottom:0.25rem;'>"
            "<span style='display:inline-block;width:0.9rem;height:0.9rem;"
            f"background:rgb({r},{g},{b});border:1px solid #888;"
            "margin-right:0.35rem;'></span>"
            f"{html.escape(label)}</span>"
        )
    parts = [_swatch(focus_rgb, focus_label)]
    for name, rgb in legend:
        parts.append(_swatch(rgb, name))
    return (
        '<p class="caption" style="margin-top:0.25rem;">'
        + "".join(parts)
        + "</p>"
    )


def _atlas_focus_html(
    *,
    rows3: list[AtlasRegionStats],
    rows6: list[AtlasRegionStats] | None = None,
    stats3: FocusRoiPressureStats | None = None,
    stats6: FocusRoiPressureStats | None = None,
    include_6db: bool = False,
    target_region_name: str | None = None,
    combined_overlay_3db: tuple[bytes, list[tuple[str, tuple[float, float, float]]]] | None = None,
    combined_overlay_6db: tuple[bytes, list[tuple[str, tuple[float, float, float]]]] | None = None,
    fg_focus_3db: tuple[float, float, float] = (0.4, 1.0, 0.95),
    fg_focus_6db: tuple[float, float, float] = (0.85, 0.45, 1.0),
) -> str:
    """HTML for atlas region tables (Julich 3.1) + a single combined overlay figure."""
    footnote = ""
    if target_region_name:
        footnote = (
            '<p class="caption">'
            f"* <strong>{html.escape(target_region_name)}</strong> is the Julich atlas "
            "region with the largest voxel overlap with the provided target ROI.</p>"
        )
    result = (
        '<p class="caption">'
        "Julich Brain Atlas 3.1 &mdash; regions in the focus zone "
        "<em>outside</em> the target ROI. "
        "Only regions with &gt;1% overlap are shown. "
        "Sorted by % region in focus (descending). "
        "Target ROI row highlighted at top.</p>"
    )
    table3 = _atlas_region_table(
        rows3,
        focus_label="\u22123 dB",
        roi_stats=stats3,
        target_region_name=target_region_name,
    )
    result += "<h3>\u22123 dB focus</h3>" + table3 + footnote
    if combined_overlay_3db is not None:
        png, legend = combined_overlay_3db
        result += (
            '<div id="sec-atlas-focus-overlay-3db" class="sub-panel">'
            "<h4>\u22123 dB focus \u2229 overlapping atlas regions</h4>"
            '<p class="caption">Warped Julich regions (each a distinct color, '
            "50% transparent) together with the \u22123 dB focus "
            "(50% transparent) on T1w. Overlaps blend the two colors. "
            "Cuts at the focus centroid.</p>"
            "<figure style='margin-bottom:1rem;'>"
            f'<img alt="" src="data:image/png;base64,{_b64_png(png)}">'
            "</figure>"
            + _legend_swatches_html(
                legend, focus_label="\u22123 dB focus", focus_rgb=fg_focus_3db,
            )
            + "</div>"
        )
    if include_6db and rows6 is not None:
        table6 = _atlas_region_table(
            rows6,
            focus_label="\u22126 dB",
            roi_stats=stats6,
            target_region_name=target_region_name,
        )
        result += "<h3>\u22126 dB focus</h3>" + table6 + footnote
        if combined_overlay_6db is not None:
            png6, legend6 = combined_overlay_6db
            result += (
                '<div id="sec-atlas-focus-overlay-6db" class="sub-panel">'
                "<h4>\u22126 dB focus \u2229 overlapping atlas regions</h4>"
                '<p class="caption">Warped Julich regions (each a distinct color, '
                "50% transparent) together with the \u22126 dB focus "
                "(50% transparent) on T1w.</p>"
                "<figure style='margin-bottom:1rem;'>"
                f'<img alt="" src="data:image/png;base64,{_b64_png(png6)}">'
                "</figure>"
                + _legend_swatches_html(
                    legend6, focus_label="\u22126 dB focus", focus_rgb=fg_focus_6db,
                )
                + "</div>"
            )
    return result


# ---------------------------------------------------------------------------
# Atlas region overlay helpers
# ---------------------------------------------------------------------------

def _focus_centroid_mm(
    focus_img: "nib.Nifti1Image",
) -> tuple[float, float, float] | None:
    """World-space centroid of non-zero voxels in ``focus_img``."""
    import numpy as np
    f_img = focus_img
    data = np.asanyarray(f_img.dataobj, dtype=np.float64) != 0
    idx = np.argwhere(data)
    if idx.size == 0:
        return None
    mean_vox = idx.mean(axis=0)
    homog = np.array([mean_vox[0], mean_vox[1], mean_vox[2], 1.0])
    world = f_img.affine @ homog
    return (float(world[0]), float(world[1]), float(world[2]))


def _render_atlas_focus_combined_overlay(
    *,
    t1_img: "nib.Nifti1Image",
    atlas_img: "nib.Nifti1Image",
    hemi_img: "nib.Nifti1Image",
    focus_img: "nib.Nifti1Image",
    rows: list[AtlasRegionStats] | None,
    fg_focus: tuple[float, float, float],
    focus_tag: str,
    min_pct: float = 1.0,
    max_regions: int = 10,
) -> tuple[bytes, list[tuple[str, tuple[float, float, float]]]] | None:
    """Render a single ortho combining the focus and all overlapping atlas regions.

    Returns ``(png_bytes, [(region_name, rgb), ...])`` for legend rendering,
    or ``None`` if there are no qualifying regions. Matches the table
    threshold (``pct_region_in_focus > min_pct``) and caps at ``max_regions``.
    """
    if not rows:
        return None
    picked = sorted(
        (r for r in rows if r.pct_region_in_focus > min_pct),
        key=lambda r: r.pct_region_in_focus,
        reverse=True,
    )[:max_regions]
    if not picked:
        return None

    region_masks: list[tuple[str, "nib.Nifti1Image"]] = [
        (r.region_name, region_mask_img(atlas_img, hemi_img, r.region_label, r.hemi))
        for r in picked
    ]
    cuts = _focus_centroid_mm(focus_img)
    if cuts is None:
        return None
    png, legend = viz.render_focus_atlas_overlays_transparent_ortho(
        t1_img,
        focus_img,
        region_masks,
        title=f"{focus_tag} + overlapping atlas regions (50% transparent)",
        fg_focus=fg_focus,
        cut_coords=cuts,
    )
    return png, legend


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
html { scroll-behavior: smooth; }
body.report-layout {
  display: flex; align-items: flex-start;
  margin: 0; padding: 0; min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  line-height: 1.45; color: #1a1a1a; background: #eaeaea;
}
/* ---- TOC ---- */
.toc {
  position: sticky; top: 0; align-self: flex-start;
  width: 14rem; max-height: 100vh; overflow-y: auto; flex-shrink: 0;
  padding: 1rem 1.1rem 2rem; border-right: 1px solid #ccc;
  background: #f4f4f4; box-sizing: border-box;
}
.toc-title {
  font-weight: 600; font-size: 0.85rem; text-transform: uppercase;
  letter-spacing: 0.04em; color: #555; margin: 0 0 0.5rem;
}
.toc ul { list-style: none; margin: 0; padding: 0; }
.toc li { margin: 0.3rem 0; }
.toc a { color: #0b5fff; text-decoration: none; font-size: 0.92rem; }
.toc a:hover { text-decoration: underline; }
.toc .toc-group > a { font-weight: 600; font-size: 0.94rem; }
.toc .toc-group > ul { padding-left: 0.9rem; margin-top: 0.15rem; }
.toc .toc-group > ul a { font-weight: normal; font-size: 0.88rem; color: #336; }
/* ---- Main ---- */
.report-main {
  flex: 1; min-width: 0; padding: 1.25rem 1.75rem 5rem;
  scroll-snap-type: y proximity;
}
.panel, .group-panel {
  scroll-snap-align: start; scroll-margin-top: 0.75rem;
  margin-bottom: 2.5rem; padding: 1.35rem 1.5rem 2rem;
  background: #fff; border-radius: 6px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.07); border: 1px solid #ddd;
}
.panel h2, .group-panel h2 {
  font-size: 1.2rem; margin: 0 0 0.75rem;
  padding-bottom: 0.35rem; border-bottom: 1px solid #e0e0e0;
}
h3 { font-size: 1.05rem; margin: 1.25rem 0 0.5rem; }
h4 { font-size: 0.95rem; margin: 1rem 0 0.4rem; color: #333; }
figure { margin: 0; }
img { max-width: 100%; height: auto; display: block; }
.caption { font-size: 0.9rem; color: #555; margin: 0.5rem 0 0; }
.sub-panel { margin-top: 1.5rem; padding-top: 0.5rem; border-top: 1px solid #f0f0f0; }
.sub-panel:first-child { margin-top: 0; padding-top: 0; border-top: none; }
/* ---- Tables ---- */
table.meta, table.stats, table.lut, table.summary-tbl {
  width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 0.75rem 0;
}
table.meta th, table.stats th, table.lut th, table.summary-tbl th {
  text-align: left; vertical-align: top;
  border-bottom: 1px solid #e8e8e8; padding: 0.35rem 0.5rem;
}
table.meta th { width: 11rem; }
table.stats th { width: auto; }
table.meta td, table.stats td, table.lut td, table.summary-tbl td {
  border-bottom: 1px solid #e8e8e8; padding: 0.35rem 0.5rem; word-break: break-word;
}
table.lut th { width: auto; }
table.safety th small { font-weight: normal; color: #555; }
.threshold-row td { background: #f8f6f0; font-size: 0.82rem; color: #665; border-top: 2px solid #ddd; }
.roi-highlight td { background: #fff9e6; font-weight: 500; border-top: 2px solid #e0c860; border-bottom: 2px solid #e0c860; }
/* ---- Summary cards ---- */
.summary-panel { background: #fafcff; border-color: #c0d0e8; }
.card-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.card {
  flex: 1 1 280px; background: #fff; border: 1px solid #e0e0e0;
  border-radius: 5px; padding: 0.8rem 1rem;
}
.card h3 { margin: 0 0 0.4rem; font-size: 0.95rem; }
.summary-tbl td { font-size: 0.86rem; padding: 0.25rem 0.4rem; }
/* ---- Badges ---- */
.badge {
  display: inline-block; font-size: 0.72rem; font-weight: 700;
  padding: 0.12em 0.5em; border-radius: 3px; text-transform: uppercase;
  letter-spacing: 0.04em;
}
.badge-ok { background: #d4edda; color: #155724; }
.badge-warn { background: #f8d7da; color: #721c24; }
.badge-na { background: #e9ecef; color: #6c757d; }
/* ---- Collapsible ---- */
details.meta-details { margin: 0.75rem 0; }
details.meta-details summary {
  cursor: pointer; font-weight: 600; font-size: 0.92rem; color: #444;
  padding: 0.3rem 0; user-select: none;
}
details.meta-details summary:hover { color: #0b5fff; }
code {
  font-size: 0.88em; background: #f0f0f0;
  padding: 0.05em 0.35em; border-radius: 3px;
}
/* ---- Responsive ---- */
@media (max-width: 820px) {
  body.report-layout { flex-direction: column; }
  .toc { position: relative; max-height: none; width: 100%;
         border-right: 0; border-bottom: 1px solid #ccc; }
  .report-main { padding: 1rem; }
  .card-row { flex-direction: column; }
}
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_html_report(
    *,
    report_path: Path,
    charm_dir: Path,
    t1_path: Path,
    roi_path: Path,
    pressure_path: Path,
    temperature_path: Path,
    source_charm_dir: Path | None = None,
    center_frequency_mhz: float = 0.286,
    report_notes: str | None = None,
    methodology_reference_url: str | None = None,
    baseline_body_temp_c: float = 37.0,
    exposure_duration_min: float = 1.0,
    transforms_dir: Path | None = None,
    atlas_dir: Path | None = None,
    subject_prefix: str | None = None,
    include_6db: bool = False,
) -> None:
    """Build figures, write derived mask NIfTIs, and a scrollable HTML report.

    ``charm_dir`` is the directory that contains aligned ``final_tissues.nii.gz``
    (typically ``.../sub-XX/nifti_files``).

    Temperature NIfTI is **absolute deg C**. delta-T in tables uses ``T - baseline_body_temp_c``.
    ``methodology_reference_url``: ``None`` uses the Brain Stimulation 2025 DOI;
    pass ``""`` to omit methodology links from the HTML.
    """
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    meth_ref = _resolved_methodology_url(methodology_reference_url)

    # ------------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------------
    t1_img = nib.load(t1_path)
    roi_img = nib.load(roi_path)
    seg_path = find_final_tissues(charm_dir)
    seg_img = nib.load(seg_path)
    pressure_img = nib.load(pressure_path)
    temperature_img = nib.load(temperature_path)
    pressure_mpa_img = pressure_img_mpa_from_pa(pressure_img)

    assert_same_space(
        (t1_img, "T1w"),
        (roi_img, "ROI"),
        (seg_img, "final_tissues"),
        (pressure_img, "pressure"),
        (temperature_img, "temperature"),
    )

    mask_dir = report_path.parent / "derived_masks"
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {mask_dir}. Run ok-plan prepare step first "
            "(native tissue masks must exist before the report)."
        )

    roi_cuts_mm = roi_centroid_mm(roi_img)
    temp_max_cuts_mm = max_delta_t_coords_mm(temperature_img, baseline_body_temp_c)

    # ------------------------------------------------------------------
    # Render figures
    # ------------------------------------------------------------------
    png_pressure = viz.render_scalar_on_t1(
        t1_img,
        pressure_mpa_img,
        title=f"Pressure / MPa ({Path(pressure_path).name})",
        cmap=viz.PRESSURE_CMAP,
        overlay_alpha=viz.FIELD_OVERLAY_ALPHA,
        cut_coords=roi_cuts_mm,
    )
    png_temperature = viz.render_scalar_on_t1(
        t1_img,
        temperature_img,
        title=f"Temperature / \u00b0C ({Path(temperature_path).name})",
        cmap=viz.TEMPERATURE_CMAP,
        overlay_alpha=viz.FIELD_OVERLAY_ALPHA,
        cut_coords=temp_max_cuts_mm,
        vmin=float(baseline_body_temp_c),
        vmax=None,
    )

    inside_img = nib.load(mask_dir / "inside_skull.nii.gz")
    assert_same_space((pressure_img, "pressure"), (inside_img, "inside_skull"))
    focus_minus3, focus_minus6 = build_focus_masks_amplitude_db(
        pressure_img, inside_img
    )
    nib.save(focus_minus3, mask_dir / "focus_minus3db_amplitude.nii.gz")
    nib.save(focus_minus6, mask_dir / "focus_minus6db_amplitude.nii.gz")

    p_mpa_data = np.asanyarray(pressure_mpa_img.dataobj, dtype=np.float64)
    p_mpa_global_min = float(np.nanmin(p_mpa_data[p_mpa_data != 0])) if np.any(p_mpa_data != 0) else 0.0
    p_mpa_global_max = float(np.nanmax(p_mpa_data))

    # Standalone -3/-6 dB amplitude focus maps are no longer rendered here;
    # the focus + ROI (50% transparent) composite below replaces them.
    _ = (p_mpa_global_min, p_mpa_global_max)

    # ------------------------------------------------------------------
    # Tissue masks & safety metrics
    # ------------------------------------------------------------------
    tissue_mask_specs = [
        ("scalp", "Scalp", viz.MASK_COLORS["scalp"]),
        ("skull", "Skull", viz.MASK_COLORS["skull"]),
        ("inside_skull", "Brain+", viz.MASK_COLORS["inside_skull"]),
        ("eyes", "Eyes", viz.MASK_COLORS["eyes"]),
    ]

    mask_by_key: dict[str, nib.Nifti1Image] = {}
    for key, _desc, _rgb in tissue_mask_specs:
        m_img = nib.load(mask_dir / f"{key}.nii.gz")
        assert_same_space((pressure_img, "pressure"), (m_img, f"mask:{key}"))
        mask_by_key[key] = m_img
    png_masks_combined = viz.render_combined_derived_masks_mosaic(
        t1_img,
        mask_by_key,
        title="Derived masks: Scalp (1), Skull (2), Brain+ (3), Eyes (4)",
    )

    safety_rows: list[MaskSafetyMetrics] = []
    for key, desc, _rgb in tissue_mask_specs:
        m_img = mask_by_key[key]
        assert_same_space((pressure_img, "pressure"), (m_img, f"mask:{key}"))
        safety_rows.append(
            safety_metrics_for_mask(
                key,
                desc,
                pressure_img,
                temperature_img,
                m_img,
                center_frequency_mhz=center_frequency_mhz,
                baseline_body_temp_c=baseline_body_temp_c,
                exposure_duration_min=exposure_duration_min,
            )
        )

    # ------------------------------------------------------------------
    # ROI / focus overlap
    # ------------------------------------------------------------------
    stats_focus3 = focus_roi_pressure_stats(
        pressure_img, roi_img, focus_minus3, label="\u22123 dB focus",
    )
    stats_focus6: FocusRoiPressureStats | None = None
    if include_6db:
        stats_focus6 = focus_roi_pressure_stats(
            pressure_img, roi_img, focus_minus6, label="\u22126 dB focus",
        )
    png_roi = viz.render_roi_on_t1(
        t1_img, roi_img,
        title=f"Target ROI ({Path(roi_path).name})",
        cut_coords=roi_cuts_mm,
    )
    png_roi_focus3_transparent = viz.render_roi_focus_transparent_ortho(
        t1_img, roi_img, focus_minus3,
        title="\u22123 dB focus + ROI (50% transparent)",
        fg_focus=viz.MASK_COLORS["focus_minus3db"],
        cut_coords=roi_cuts_mm,
    )
    png_roi_focus6_transparent: bytes | None = None
    if include_6db:
        png_roi_focus6_transparent = viz.render_roi_focus_transparent_ortho(
            t1_img, roi_img, focus_minus6,
            title="\u22126 dB focus + ROI (50% transparent)",
            fg_focus=viz.MASK_COLORS["focus_minus6db"],
            cut_coords=roi_cuts_mm,
        )

    # ------------------------------------------------------------------
    # Atlas (conditional) — Julich 3.1 only
    # ------------------------------------------------------------------
    atlas_rows3: list[AtlasRegionStats] | None = None
    atlas_rows6: list[AtlasRegionStats] | None = None
    atlas_target_name: str | None = None
    combined_overlay_3db: tuple[bytes, list[tuple[str, tuple[float, float, float]]]] | None = None
    combined_overlay_6db: tuple[bytes, list[tuple[str, tuple[float, float, float]]]] | None = None
    if transforms_dir is not None and atlas_dir is not None and subject_prefix is not None:
        atlas_out = report_path.parent / "atlas"
        atlas_img, hemi_img, atlas_lut = load_and_warp_julich_atlas(
            atlas_dir=atlas_dir,
            transforms_dir=transforms_dir,
            reference_img_path=t1_path,
            subject_prefix=subject_prefix,
            output_dir=atlas_out,
        )
        atlas_rows3 = atlas_region_overlap(
            atlas_img, hemi_img, atlas_lut, focus_minus3, roi_img, pressure_img,
        )
        if include_6db:
            atlas_rows6 = atlas_region_overlap(
                atlas_img, hemi_img, atlas_lut, focus_minus6, roi_img, pressure_img,
            )
        _, _, atlas_target_name = atlas_label_for_roi(
            atlas_img, hemi_img, atlas_lut, roi_img,
        )
        combined_overlay_3db = _render_atlas_focus_combined_overlay(
            t1_img=t1_img,
            atlas_img=atlas_img,
            hemi_img=hemi_img,
            focus_img=focus_minus3,
            rows=atlas_rows3,
            fg_focus=viz.MASK_COLORS["focus_minus3db"],
            focus_tag="\u22123 dB focus",
        )
        if include_6db and atlas_rows6 is not None:
            combined_overlay_6db = _render_atlas_focus_combined_overlay(
                t1_img=t1_img,
                atlas_img=atlas_img,
                hemi_img=hemi_img,
                focus_img=focus_minus6,
                rows=atlas_rows6,
                fg_focus=viz.MASK_COLORS["focus_minus6db"],
                focus_tag="\u22126 dB focus",
            )

    # ------------------------------------------------------------------
    # Build HTML structure
    # ------------------------------------------------------------------
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Overview ---
    meta_rows = [
        ("Generated (UTC)", generated),
        ("Aligned NIfTI pack", str(charm_dir)),
        ("final_tissues", str(seg_path)),
        ("T1w", str(t1_path)),
        ("ROI", str(roi_path)),
        ("Pressure map", str(pressure_path)),
        ("Temperature map", str(temperature_path)),
        (
            "Temperature interpretation",
            f"Absolute \u00b0C on disk; \u0394T = T \u2212 {baseline_body_temp_c} \u00b0C.",
        ),
        ("Derived masks directory", str(mask_dir)),
        (
            "Tissue masks",
            "Binarized on native CHARM final_tissues, then nearest-neighbor "
            "resample to the simulation grid. "
            "Skull = labels 7+8 (petra2density-style bone).",
        ),
        (
            "Shared grid",
            f"{tuple(int(x) for x in t1_img.shape[:3])} voxels; same affine for all volumes.",
        ),
        (
            "MI center frequency",
            f"{center_frequency_mhz} MHz.",
        ),
        (
            "CEM43 exposure",
            f"{exposure_duration_min} min uniform (steady-state assumption).",
        ),
    ]
    if atlas_rows3 is not None:
        meta_rows.append((
            "Atlas",
            f"Julich Brain Atlas 3.1 ({atlas_dir}), warped MNI\u2192subject via "
            f"ANTs ({transforms_dir}), NearestNeighbor.",
        ))
    if report_notes:
        meta_rows.append(("Project notes", report_notes))
    if source_charm_dir is not None:
        meta_rows.insert(2, ("Source CHARM / m2m", str(Path(source_charm_dir).resolve())))

    meth_html = ""
    if meth_ref:
        esc = html.escape(meth_ref, quote=True)
        meth_html = (
            f'<p>Methodology reference: <a href="{esc}">{html.escape(meth_ref)}</a>.</p>'
        )
    overview_inner = (
        "<p>All figures share the simulation domain grid. "
        "Pressure and focus cuts pass through the ROI centroid; "
        "temperature cuts pass through the location of maximum &Delta;T. "
        "Pressure and temperature use a semi-transparent overlay on the T1w underlay.</p>"
        + meth_html
        + '<details class="meta-details"><summary>Input files and parameters</summary>'
        + _meta_table(meta_rows)
        + "</details>"
    )
    overview_section = _panel_text("sec-overview", "Overview", overview_inner)

    # --- Summary card ---
    summary_section = _summary_card_section(
        safety_rows, stats_focus3, stats_focus6, include_6db=include_6db,
    )

    # --- Section A: Targeting and Exposure ---
    cap_t = "ROI (red, 50% transparent) + focus (blue, 50% transparent); overlap blended."
    targeting_inner = (
        _sub_panel_img("sec-roi", "Target ROI", png_roi,
                       "ROI in red on T1w; cuts at ROI centroid.")
        + _sub_panel_img("sec-pressure", "Pressure map", png_pressure,
                         "In situ pressure (MPa); cuts at ROI centroid.")
        + _sub_panel_img("sec-roi-f3-transparent",
                         "\u22123 dB focus + ROI (50% transparent)",
                         png_roi_focus3_transparent, cap_t)
    )
    if include_6db and png_roi_focus6_transparent is not None:
        targeting_inner += _sub_panel_img(
            "sec-roi-f6-transparent",
            "\u22126 dB focus + ROI (50% transparent)",
            png_roi_focus6_transparent, cap_t,
        )
    targeting_inner += _roi_focus_overlap_tables_html(
        stats_focus3, stats_focus6, include_6db=include_6db,
    )
    targeting_section = _group_section("sec-targeting", "Targeting and exposure", targeting_inner)

    # --- Section B: Biophysical Safety ---
    safety_inner = (
        _sub_panel_img("sec-temperature", "Temperature map", png_temperature,
                       f"Absolute temperature (\u00b0C); cuts at max \u0394T location; "
                       f"colorbar starts at {baseline_body_temp_c} \u00b0C.")
        + _sub_panel_img("sec-masks", "Derived tissue masks", png_masks_combined,
                         "Scalp (1), Skull (2), Brain+ (3), Eyes (4). "
                         "Native final_tissues binarized then nearest-neighbor to sim grid.")
        + _safety_metrics_section(
            safety_rows,
            center_frequency_mhz=center_frequency_mhz,
            methodology_reference_url=meth_ref,
            baseline_body_temp_c=baseline_body_temp_c,
            exposure_duration_min=exposure_duration_min,
        )
    )
    safety_section = _group_section("sec-safety", "Biophysical safety", safety_inner)

    # --- Section C: Anatomical Context (conditional) ---
    atlas_section_html = ""
    if atlas_rows3 is not None:
        atlas_inner = _atlas_focus_html(
            rows3=atlas_rows3,
            rows6=atlas_rows6,
            stats3=stats_focus3,
            stats6=stats_focus6,
            include_6db=include_6db,
            target_region_name=atlas_target_name,
            combined_overlay_3db=combined_overlay_3db,
            combined_overlay_6db=combined_overlay_6db,
            fg_focus_3db=viz.MASK_COLORS["focus_minus3db"],
            fg_focus_6db=viz.MASK_COLORS["focus_minus6db"],
        )
        atlas_section_html = _group_section(
            "sec-atlas-focus", "Anatomical context (Julich 3.1)", atlas_inner
        )

    # --- Assemble ---
    main_html = (
        overview_section
        + summary_section
        + targeting_section
        + safety_section
        + atlas_section_html
    )

    # --- TOC ---
    targeting_toc: list[tuple[str, str]] = [
        ("sec-roi", "Target ROI"),
        ("sec-pressure", "Pressure map"),
        ("sec-roi-f3-transparent", "\u22123 dB + ROI (transparent)"),
    ]
    if include_6db:
        targeting_toc.append(
            ("sec-roi-f6-transparent", "\u22126 dB + ROI (transparent)")
        )
    targeting_toc.append(("sec-roi-tables", "Overlap stats"))

    toc_groups: list[tuple[str, str, list[tuple[str, str]]]] = [
        ("", "", [("sec-overview", "Overview"), ("sec-summary", "Summary")]),
        ("sec-targeting", "Targeting & Exposure", targeting_toc),
        ("sec-safety", "Biophysical Safety", [
            ("sec-temperature", "Temperature"),
            ("sec-masks", "Tissue masks"),
            ("sec-safety-metrics", "MI / CEM43 / \u0394T"),
        ]),
    ]
    if atlas_rows3 is not None:
        toc_groups.append(("sec-atlas-focus", "Anatomical Context", []))

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ok-plan report</title>
  <style>
{_CSS}
  </style>
</head>
<body class="report-layout">
{_toc_nav(toc_groups)}
<main class="report-main">
<h1 style="margin-top:0;font-size:1.5rem;">ok-plan &mdash; QC report</h1>
{main_html}
</main>
</body>
</html>
"""

    report_path.write_text(doc, encoding="utf-8")
