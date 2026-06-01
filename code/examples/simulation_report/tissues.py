"""SimNIBS CHARM / m2m final_tissues labels and derived safety masks."""

from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np

# Default label IDs from SimNIBS m2m final_tissues (see final_tissues_LUT.txt).
LABEL_WM = 1
LABEL_GM = 2
LABEL_CSF = 3
LABEL_BONE = 4
LABEL_SCALP = 5
LABEL_EYE = 6
LABEL_COMPACT_BONE = 7
LABEL_SPONGY_BONE = 8


def bone_from_label(seg: np.ndarray) -> np.ndarray:
    """Compact + spongy bone only (labels 7 and 8), matching petra2density ``bone_from_label``.

    Cortical bone label 4 is excluded from this skull/bone mask so it stays aligned
    with that pipeline's bone definition for PETRA / density workflows.
    """
    d = np.asanyarray(seg)
    return (d == LABEL_COMPACT_BONE) | (d == LABEL_SPONGY_BONE)


def find_final_tissues(charm_dir: Path) -> Path:
    """Return path to final_tissues.nii.gz inside a CHARM / m2m folder."""
    charm_dir = Path(charm_dir)
    candidate = charm_dir / "final_tissues.nii.gz"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Could not find final_tissues.nii.gz under {charm_dir}"
    )


def find_tissue_lut(charm_dir: Path) -> Path | None:
    """Return final_tissues_LUT.txt if present."""
    p = Path(charm_dir) / "final_tissues_LUT.txt"
    return p if p.is_file() else None


def parse_tissue_lut(lut_path: Path) -> dict[int, tuple[str, tuple[float, float, float]]]:
    """Parse SimNIBS-style LUT: label, name, R G B A (0–255)."""
    out: dict[int, tuple[str, tuple[float, float, float]]] = {}
    for raw in lut_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(
            r"^(\d+)\s+(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
            line,
        )
        if not m:
            continue
        lab = int(m.group(1))
        name = m.group(2).strip()
        r, g, b = (int(m.group(i)) / 255.0 for i in (3, 4, 5))
        out[lab] = (name, (r, g, b))
    return out


def build_derived_masks(seg_data: np.ndarray) -> dict[str, np.ndarray]:
    """Binary masks (same shape as seg) for reporting.

    - scalp: label 5
    - skull: compact + spongy bone (labels 7, 8) only — same as petra2density
      ``bone_from_label`` (excludes cortical bone label 4)
    - inside_skull: WM + GM + CSF (1–3), i.e. intracranial soft tissue
    - eyes: label 6
    """
    d = np.asanyarray(seg_data)
    masks = {
        "scalp": (d == LABEL_SCALP).astype(np.uint8),
        "skull": bone_from_label(d).astype(np.uint8),
        "inside_skull": np.isin(d, [LABEL_WM, LABEL_GM, LABEL_CSF]).astype(np.uint8),
        "eyes": (d == LABEL_EYE).astype(np.uint8),
    }
    return masks


def mask_to_nifti(mask: np.ndarray, ref_img: nib.nifti1.Nifti1Image) -> nib.Nifti1Image:
    return nib.Nifti1Image(mask.astype(np.float32), ref_img.affine, ref_img.header)
