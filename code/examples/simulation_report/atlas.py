"""Julich Brain Atlas 3.1 in subject space: parse XML, warp MNI→sim grid, region stats."""

from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from ok_plan.nii_utils import squeeze_to_3d


def parse_julich_xml(xml_path: str | Path) -> dict[int, str]:
    """Return ``{grayvalue: structure_name}`` from a Julich atlas XML.

    Handles encoding issues and mismatched closing tags that appear in
    some Julich XML distributions.
    """
    raw = Path(xml_path).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\ufffd", "")
    if "<JulichBrain-Atlas" in text and "</JulichBrainAtlas" in text:
        text = re.sub(
            r"</JulichBrainAtlas\s*>",
            "</JulichBrain-Atlas>",
            text,
        )
    root = ET.fromstring(text)
    lut: dict[int, str] = {}
    for struct in root.iter("Structure"):
        gv = int(struct.attrib["grayvalue"])
        name = (struct.text or "").strip()
        if gv > 0 and name:
            lut[gv] = name
    return lut


HEMI_LEFT = 1
HEMI_RIGHT = 2
_HEMI_VAL_TO_TAG = {HEMI_LEFT: "L", HEMI_RIGHT: "R"}
_HEMI_TAG_TO_VAL = {"L": HEMI_LEFT, "R": HEMI_RIGHT}


def merge_hemispheres_same_labels(
    lh_img: nib.Nifti1Image,
    rh_img: nib.Nifti1Image,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Merge LH + RH atlases preserving the shared 1..N grayvalue space.

    The two Julich hemisphere files use the same grayvalue for the same
    bilateral region (e.g. grayvalue 1 = "Area 3b (PostCG)" in both LH and RH
    files). This merge keeps that 1:1 mapping: the merged label image still has
    only N distinct non-zero labels. Hemisphere identity is encoded in a
    separate image (``HEMI_LEFT`` / ``HEMI_RIGHT``) so downstream code can
    split region stats by side without inflating the label space.

    Returns ``(merged_labels_img, hemi_img)`` on the same affine/shape as
    ``lh_img``.
    """
    lh = np.asanyarray(squeeze_to_3d(lh_img).dataobj, dtype=np.int32)
    rh = np.asanyarray(squeeze_to_3d(rh_img).dataobj, dtype=np.int32)
    if lh.shape != rh.shape:
        raise ValueError(f"LH/RH shape mismatch: {lh.shape} vs {rh.shape}")

    merged = np.zeros(lh.shape, dtype=np.int32)
    hemi = np.zeros(lh.shape, dtype=np.uint8)

    lh_mask = lh > 0
    merged[lh_mask] = lh[lh_mask]
    hemi[lh_mask] = HEMI_LEFT

    rh_only = (rh > 0) & ~lh_mask
    merged[rh_only] = rh[rh_only]
    hemi[rh_only] = HEMI_RIGHT

    ref = squeeze_to_3d(lh_img)
    merged_img = nib.Nifti1Image(merged, ref.affine, ref.header)
    hemi_img = nib.Nifti1Image(hemi, ref.affine, ref.header)
    return merged_img, hemi_img


def warp_atlas_to_subject(
    atlas_mni_path: str | Path,
    reference_img_path: str | Path,
    affine_mat: str | Path,
    inverse_warp: str | Path,
    *,
    output_path: str | Path | None = None,
) -> nib.Nifti1Image:
    """Use ANTs ``antsApplyTransforms`` to warp a label atlas MNI→subject.

    Applies inverse transforms (std→kplan) using NearestNeighbor interpolation
    for discrete labels. The reference image defines the output grid.
    """
    atlas_mni_path = str(Path(atlas_mni_path).resolve())
    reference_img_path = str(Path(reference_img_path).resolve())
    affine_mat = str(Path(affine_mat).resolve())
    inverse_warp = str(Path(inverse_warp).resolve())

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        output_path = tmp.name
        tmp.close()
    output_path = str(Path(output_path).resolve())

    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", atlas_mni_path,
        "-r", reference_img_path,
        "-o", output_path,
        "-t", f"[{affine_mat},1]",
        "-t", inverse_warp,
        "-n", "NearestNeighbor",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return nib.load(output_path)


def load_and_warp_julich_atlas(
    atlas_dir: str | Path,
    transforms_dir: str | Path,
    reference_img_path: str | Path,
    subject_prefix: str,
    *,
    output_dir: str | Path | None = None,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, dict[int, str]]:
    """Load Julich Brain Atlas 3.1, merge hemispheres, warp MNI→sim grid.

    Returns ``(atlas_subject_img, hemi_subject_img, lut)`` where:

    * ``atlas_subject_img`` holds the merged 1..N grayvalues (shared between
      hemispheres; ~207 labels for Julich 3.1).
    * ``hemi_subject_img`` holds ``HEMI_LEFT`` / ``HEMI_RIGHT`` where the
      warped voxels originated from the LH / RH source file (0 elsewhere).
    * ``lut`` is the bilateral ``{grayvalue: region_name}`` map parsed from
      the atlas XML (names do **not** include L/R suffixes — add those per
      voxel from the hemisphere mask).

    Both warped images are saved into ``output_dir`` when provided.
    """
    atlas_dir = Path(atlas_dir)
    transforms_dir = Path(transforms_dir)

    lh_nii = sorted(atlas_dir.glob("*_lh_MNI152.nii.gz"))
    rh_nii = sorted(atlas_dir.glob("*_rh_MNI152.nii.gz"))
    lh_xml = sorted(atlas_dir.glob("*_lh_MNI152.xml"))
    rh_xml = sorted(atlas_dir.glob("*_rh_MNI152.xml"))
    if not lh_nii or not rh_nii or not lh_xml or not rh_xml:
        raise FileNotFoundError(
            f"Could not find Julich 3.1 atlas files (*_lh/rh_MNI152.*) in {atlas_dir}"
        )

    # LH and RH XML files describe the same 207 bilateral regions with
    # identical grayvalues; either file suffices as the bilateral LUT.
    atlas_lut = parse_julich_xml(lh_xml[0])

    lh_img = nib.load(lh_nii[0])
    rh_img = nib.load(rh_nii[0])
    merged_img, hemi_img = merge_hemispheres_same_labels(lh_img, rh_img)

    if output_dir is not None:
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        merged_mni_path: Path = od / "julich_bilateral_MNI152.nii.gz"
        hemi_mni_path: Path = od / "julich_bilateral_hemi_MNI152.nii.gz"
    else:
        tmp_m = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        tmp_h = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        merged_mni_path = Path(tmp_m.name)
        hemi_mni_path = Path(tmp_h.name)
        tmp_m.close()
        tmp_h.close()
    nib.save(merged_img, merged_mni_path)
    nib.save(hemi_img, hemi_mni_path)

    affine_mat = transforms_dir / f"{subject_prefix}_kplan2std_0GenericAffine.mat"
    inverse_warp = transforms_dir / f"{subject_prefix}_kplan2std_1InverseWarp.nii.gz"
    if not affine_mat.is_file():
        raise FileNotFoundError(f"Affine transform not found: {affine_mat}")
    if not inverse_warp.is_file():
        raise FileNotFoundError(f"Inverse warp not found: {inverse_warp}")

    atlas_warped_path: str | Path | None = None
    hemi_warped_path: str | Path | None = None
    if output_dir is not None:
        atlas_warped_path = Path(output_dir) / "julich_bilateral_subject.nii.gz"
        hemi_warped_path = Path(output_dir) / "julich_bilateral_hemi_subject.nii.gz"

    atlas_subject = warp_atlas_to_subject(
        merged_mni_path,
        reference_img_path,
        affine_mat,
        inverse_warp,
        output_path=atlas_warped_path,
    )
    hemi_subject = warp_atlas_to_subject(
        hemi_mni_path,
        reference_img_path,
        affine_mat,
        inverse_warp,
        output_path=hemi_warped_path,
    )
    return atlas_subject, hemi_subject, atlas_lut


# ---------------------------------------------------------------------------
# Region overlap statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtlasRegionStats:
    """Per-region overlap with a focus mask and pressure summaries.

    ``region_label`` is the atlas grayvalue (shared across hemispheres);
    ``hemi`` is ``"L"`` or ``"R"``. ``region_name`` already includes the
    hemisphere suffix.
    """

    region_label: int
    hemi: str
    region_name: str
    n_region_voxels: int
    n_overlap_voxels: int
    pct_region_in_focus: float
    p_max_pa: float
    p_min_pa: float
    p_mean_pa: float


def _label_array(img: nib.Nifti1Image) -> np.ndarray:
    return np.asanyarray(squeeze_to_3d(img).dataobj, dtype=np.int32)


def atlas_region_overlap(
    atlas_img: nib.Nifti1Image,
    hemi_img: nib.Nifti1Image,
    atlas_lut: dict[int, str],
    focus_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    pressure_img: nib.Nifti1Image,
) -> list[AtlasRegionStats]:
    """Which atlas regions does the focus overlap, *excluding* the target ROI?

    Regions are split per hemisphere using ``hemi_img`` (``HEMI_LEFT`` /
    ``HEMI_RIGHT``). For each ``(grayvalue, side)`` with nonzero overlap in
    ``focus ∩ region ∖ ROI`` we report the region name with ``" L"`` / ``" R"``
    suffix, ``% region in focus``, and pressure stats inside the overlap.

    Sorted by descending ``pct_region_in_focus``.
    """
    a = _label_array(atlas_img)
    h = _label_array(hemi_img)
    f = np.asanyarray(squeeze_to_3d(focus_img).dataobj, dtype=np.float64) != 0
    r = np.asanyarray(squeeze_to_3d(roi_img).dataobj, dtype=np.float64) != 0
    p = np.squeeze(
        np.asanyarray(squeeze_to_3d(pressure_img).dataobj, dtype=np.float64)
    )
    if not (a.shape == h.shape == f.shape == r.shape == p.shape):
        raise ValueError(
            f"Shape mismatch: atlas {a.shape}, hemi {h.shape}, focus {f.shape}, "
            f"roi {r.shape}, pressure {p.shape}"
        )

    focus_excl_roi = f & ~r

    labels_in_zone = np.unique(a[focus_excl_roi])
    labels_in_zone = labels_in_zone[labels_in_zone > 0]

    results: list[AtlasRegionStats] = []
    for lab in labels_in_zone:
        for hemi_val, hemi_tag in _HEMI_VAL_TO_TAG.items():
            region_mask = (a == lab) & (h == hemi_val)
            n_region = int(np.count_nonzero(region_mask))
            if n_region == 0:
                continue
            overlap = region_mask & focus_excl_roi
            n_ov = int(np.count_nonzero(overlap))
            if n_ov == 0:
                continue
            pct = 100.0 * n_ov / max(n_region, 1)
            vals = p[overlap & np.isfinite(p)]
            if vals.size == 0:
                pmax = pmin = pmean = float("nan")
            else:
                pmax = float(np.max(vals))
                pmin = float(np.min(vals))
                pmean = float(np.mean(vals))
            base = atlas_lut.get(int(lab), f"Unknown ({lab})")
            results.append(
                AtlasRegionStats(
                    region_label=int(lab),
                    hemi=hemi_tag,
                    region_name=f"{base} {hemi_tag}",
                    n_region_voxels=n_region,
                    n_overlap_voxels=n_ov,
                    pct_region_in_focus=pct,
                    p_max_pa=pmax,
                    p_min_pa=pmin,
                    p_mean_pa=pmean,
                )
            )
    results.sort(key=lambda s: s.pct_region_in_focus, reverse=True)
    return results


def atlas_label_for_roi(
    atlas_img: nib.Nifti1Image,
    hemi_img: nib.Nifti1Image,
    atlas_lut: dict[int, str],
    roi_img: nib.Nifti1Image,
) -> tuple[int | None, str | None, str | None]:
    """Return ``(grayvalue, hemisphere_tag, suffixed_name)`` for the atlas
    region with the largest voxel overlap with the ROI.

    ``hemisphere_tag`` is ``"L"`` or ``"R"``; ``suffixed_name`` already
    includes it (e.g. ``"Area p24ab (pACC) L"``). Returns all-``None`` when
    the ROI does not overlap any labelled voxel.
    """
    a = _label_array(atlas_img)
    h = _label_array(hemi_img)
    r = np.asanyarray(squeeze_to_3d(roi_img).dataobj, dtype=np.float64) != 0
    if not (a.shape == h.shape == r.shape):
        raise ValueError(
            f"Shape mismatch: atlas {a.shape}, hemi {h.shape}, roi {r.shape}"
        )
    best: tuple[int, str, int] | None = None  # (grayvalue, hemi_tag, count)
    for hemi_val, hemi_tag in _HEMI_VAL_TO_TAG.items():
        side_mask = r & (h == hemi_val)
        if not np.any(side_mask):
            continue
        labs = a[side_mask]
        labs = labs[labs > 0]
        if labs.size == 0:
            continue
        vals, counts = np.unique(labs, return_counts=True)
        top_idx = int(np.argmax(counts))
        count = int(counts[top_idx])
        if best is None or count > best[2]:
            best = (int(vals[top_idx]), hemi_tag, count)
    if best is None:
        return None, None, None
    gv, hemi_tag, _ = best
    base = atlas_lut.get(gv, f"Unknown ({gv})")
    return gv, hemi_tag, f"{base} {hemi_tag}"


def region_mask_img(
    atlas_img: nib.Nifti1Image,
    hemi_img: nib.Nifti1Image,
    grayvalue: int,
    hemi: str,
) -> nib.Nifti1Image:
    """Return a binary NIfTI image for ``(grayvalue, hemisphere)``.

    ``hemi`` must be ``"L"`` or ``"R"``.
    """
    if hemi not in _HEMI_TAG_TO_VAL:
        raise ValueError(f"hemi must be 'L' or 'R' (got {hemi!r})")
    a = _label_array(atlas_img)
    h = _label_array(hemi_img)
    if a.shape != h.shape:
        raise ValueError(f"Atlas shape {a.shape} vs hemi {h.shape}")
    mask = ((a == int(grayvalue)) & (h == _HEMI_TAG_TO_VAL[hemi])).astype(np.uint8)
    ref = squeeze_to_3d(atlas_img)
    return nib.Nifti1Image(mask, ref.affine, ref.header)
