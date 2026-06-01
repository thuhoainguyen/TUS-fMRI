"""Load per-project JSON settings (paths, output layout, acoustic parameters).

Paths in the JSON may be absolute or relative to the config file's directory,
similar to keeping calibration maps beside a tool (see petra2density-style
layout: project folder holds ``config.json`` + inputs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _resolve(p: str | Path, base: Path) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _req(d: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise ValueError(f"Missing required key {key!r} in {where}")
    return d[key]


@dataclass(frozen=True)
class PlanConfig:
    """Fields consumed by ``ok-plan`` when ``--config`` is used."""

    subject: str
    output_root: Path
    charm_dir: Path
    t1: Path
    pressure: Path
    temperature: Path
    roi: Path
    center_frequency_mhz: float
    report_name: str
    baseline_body_temp_c: float
    exposure_duration_min: float
    transforms_dir: Path | None = None
    atlas_dir: Path | None = None
    report_notes: str | None = None
    methodology_reference_url: str | None = None

    def as_arg_defaults(self) -> dict[str, Any]:
        """Keyword args suitable for ``argparse.ArgumentParser.set_defaults``."""
        return {
            "subject": self.subject,
            "output_root": self.output_root,
            "charm_dir": self.charm_dir,
            "t1": self.t1,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "roi": self.roi,
            "center_frequency_mhz": self.center_frequency_mhz,
            "report_name": self.report_name,
            "baseline_body_temp_c": self.baseline_body_temp_c,
            "exposure_duration_min": self.exposure_duration_min,
            "transforms_dir": self.transforms_dir,
            "atlas_dir": self.atlas_dir,
            "report_notes": self.report_notes,
            "methodology_reference_url": self.methodology_reference_url,
        }


def load_plan_config(path: str | Path) -> PlanConfig:
    """Parse project JSON. Unknown top-level keys are ignored."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    base = cfg_path.parent
    with cfg_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a JSON object: {cfg_path}")

    where = str(cfg_path)
    subject = str(_req(raw, "subject", where=where))
    output_root = _resolve(_req(raw, "output_root", where=where), base)

    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f'Config must contain an "inputs" object: {cfg_path}')
    charm_dir = _resolve(_req(inputs, "charm_dir", where=where), base)
    t1 = _resolve(_req(inputs, "t1", where=where), base)
    pressure = _resolve(_req(inputs, "pressure", where=where), base)
    temperature = _resolve(_req(inputs, "temperature", where=where), base)
    roi = _resolve(_req(inputs, "roi", where=where), base)

    acoustic = raw.get("acoustic", {})
    if acoustic is None:
        acoustic = {}
    if not isinstance(acoustic, dict):
        raise ValueError(f'"acoustic" must be an object if present: {cfg_path}')
    center_frequency_mhz = float(
        acoustic.get("center_frequency_mhz", raw.get("center_frequency_mhz", 0.286))
    )

    thermal = raw.get("thermal", {})
    if thermal is None:
        thermal = {}
    if not isinstance(thermal, dict):
        raise ValueError(f'"thermal" must be an object if present: {cfg_path}')
    baseline_body_temp_c = float(
        thermal.get("baseline_body_temp_c", raw.get("baseline_body_temp_c", 37.0))
    )
    exposure_duration_min = float(
        thermal.get(
            "exposure_duration_min", raw.get("exposure_duration_min", 1.0)
        )
    )

    transforms_dir: Path | None = None
    atlas_dir: Path | None = None
    if isinstance(inputs, dict):
        td = inputs.get("transforms_dir")
        if td is not None:
            transforms_dir = _resolve(td, base)
        ad = inputs.get("atlas_dir")
        if ad is not None:
            atlas_dir = _resolve(ad, base)

    report_name = str(raw.get("report_name", "report.html"))

    report_notes = raw.get("report_notes")
    if report_notes is not None and not isinstance(report_notes, str):
        raise ValueError(f'"report_notes" must be a string if present: {cfg_path}')
    methodology_reference_url: str | None = None
    lit = raw.get("literature")
    if isinstance(lit, dict) and "methodology_review" in lit:
        u = lit.get("methodology_review")
        if u is not None and not isinstance(u, str):
            raise ValueError(
                f'"literature.methodology_review" must be a string or null: {cfg_path}'
            )
        methodology_reference_url = u  # None → report default DOI; "" → omit links

    return PlanConfig(
        subject=subject,
        output_root=output_root,
        charm_dir=charm_dir,
        t1=t1,
        pressure=pressure,
        temperature=temperature,
        roi=roi,
        center_frequency_mhz=center_frequency_mhz,
        report_name=report_name,
        baseline_body_temp_c=baseline_body_temp_c,
        exposure_duration_min=exposure_duration_min,
        transforms_dir=transforms_dir,
        atlas_dir=atlas_dir,
        report_notes=report_notes,
        methodology_reference_url=methodology_reference_url,
    )
