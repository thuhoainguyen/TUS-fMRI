"""Orthogonal (and mosaic) NIfTI figures with T1w underlay.

Nilearn's ``transparency`` argument is passed to matplotlib ``imshow`` as ``alpha``
(opacity): 0 = invisible overlay, 1 = fully opaque. We expose ``overlay_alpha`` in
this module and forward it as ``transparency=...`` to Nilearn.
"""

from __future__ import annotations

import io
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn.plotting import plot_img, plot_stat_map

from ok_plan.nii_utils import squeeze_to_3d


def _close_all() -> None:
    plt.close("all")


def figure_to_png_bytes(
    fig: plt.Figure, *, dpi: int = 120, facecolor: str | None = "white"
) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=facecolor if facecolor is not None else "white",
    )
    plt.close(fig)
    return buf.getvalue()


def _mask_overlay_cmap(
    fg: tuple[float, float, float], *, fg_alpha: float = 1.0
) -> ListedColormap:
    """Label 0 transparent; mask voxels use fg RGBA (alpha combined with overlay_alpha)."""
    return ListedColormap([(0.0, 0.0, 0.0, 0.0), (*fg[:3], float(fg_alpha))])


def render_binary_mask_on_t1(
    t1_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    title: str,
    fg_rgb: tuple[float, float, float],
    overlay_alpha: float = 1.0,
    cut_coords: tuple[float, float, float] | None = None,
) -> bytes:
    """Binary mask over T1w (ortho). ``overlay_alpha``: overlay opacity (1 = opaque)."""
    _close_all()
    mask_img = squeeze_to_3d(mask_img)
    data = np.asanyarray(mask_img.dataobj, dtype=np.float64)
    if not np.any(data != 0):
        plot_img(
            t1_img,
            cut_coords=cut_coords,
            display_mode="ortho",
            cmap="gray",
            dim=False,
            black_bg=False,
            title=f"{title} (empty mask)",
            draw_cross=False,
            annotate=False,
            colorbar=False,
        )
        return figure_to_png_bytes(plt.gcf(), facecolor="white")

    m = (data != 0).astype(np.float32)
    stat = nib.Nifti1Image(m, mask_img.affine, mask_img.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=None,
        cmap=_mask_overlay_cmap(fg_rgb, fg_alpha=1.0),
        vmin=0.0,
        vmax=1.0,
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=False,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def render_binary_mask_mosaic_on_t1(
    t1_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    title: str,
    fg_rgb: tuple[float, float, float],
    overlay_alpha: float = 1.0,
) -> bytes:
    """Binary mask over T1w (mosaic of cuts)."""
    _close_all()
    mask_img = squeeze_to_3d(mask_img)
    data = np.asanyarray(mask_img.dataobj, dtype=np.float64)
    if not np.any(data != 0):
        plot_img(
            t1_img,
            display_mode="mosaic",
            cmap="gray",
            dim=False,
            black_bg=False,
            title=f"{title} (empty mask)",
            draw_cross=False,
            annotate=False,
            colorbar=False,
        )
        return figure_to_png_bytes(plt.gcf(), facecolor="white")

    m = (data != 0).astype(np.float32)
    stat = nib.Nifti1Image(m, mask_img.affine, mask_img.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        threshold=None,
        cmap=_mask_overlay_cmap(fg_rgb, fg_alpha=1.0),
        vmin=0.0,
        vmax=1.0,
        symmetric_cbar=False,
        dim=False,
        display_mode="mosaic",
        draw_cross=False,
        annotate=False,
        colorbar=False,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def render_scalar_on_t1(
    t1_img: nib.Nifti1Image,
    field_img: nib.Nifti1Image,
    *,
    title: str,
    cmap: str,
    overlay_alpha: float | None = None,
    cut_coords: tuple[float, float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> bytes:
    """Scalar field over T1w. ``overlay_alpha`` is Nilearn overlay opacity (0–1).

    Default overlay alpha is tuned so the T1 underlay stays readable through the field.
    """
    if overlay_alpha is None:
        overlay_alpha = FIELD_OVERLAY_ALPHA
    _close_all()
    field_img = squeeze_to_3d(field_img)
    plot_stat_map(
        field_img,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=True,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def render_masked_pressure_on_t1(
    t1_img: nib.Nifti1Image,
    pressure_mpa_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    title: str,
    cmap: str | None = None,
    overlay_alpha: float | None = None,
    cut_coords: tuple[float, float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> bytes:
    """Pressure (MPa) restricted to a binary mask, with full-range colorbar."""
    if cmap is None:
        cmap = PRESSURE_CMAP
    if overlay_alpha is None:
        overlay_alpha = FIELD_OVERLAY_ALPHA
    _close_all()
    pressure_mpa_img = squeeze_to_3d(pressure_mpa_img)
    mask_img = squeeze_to_3d(mask_img)
    p = np.asanyarray(pressure_mpa_img.dataobj, dtype=np.float64)
    m = np.asanyarray(mask_img.dataobj, dtype=np.float64) != 0
    masked = np.where(m, p, 0.0).astype(np.float32)
    if not np.any(m):
        plot_img(
            t1_img,
            cut_coords=cut_coords,
            display_mode="ortho",
            cmap="gray",
            dim=False,
            black_bg=False,
            title=f"{title} (empty mask)",
            draw_cross=False,
            annotate=False,
            colorbar=False,
        )
        return figure_to_png_bytes(plt.gcf(), facecolor="white")
    stat = nib.Nifti1Image(masked, pressure_mpa_img.affine, pressure_mpa_img.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=True,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def render_roi_on_t1(
    t1_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    *,
    title: str = "Target ROI",
    overlay_alpha: float | None = None,
    cut_coords: tuple[float, float, float] | None = None,
) -> bytes:
    if overlay_alpha is None:
        overlay_alpha = MASK_OVERLAY_ALPHA
    roi_img = squeeze_to_3d(roi_img)
    data = np.asanyarray(roi_img.dataobj, dtype=np.float64)
    if not np.any(data != 0):
        raise ValueError("ROI image has no non-zero voxels.")
    return render_binary_mask_on_t1(
        t1_img,
        roi_img,
        title=title,
        fg_rgb=(0.95, 0.15, 0.12),
        overlay_alpha=overlay_alpha,
        cut_coords=cut_coords,
    )


def render_segmentation_on_t1(
    t1_img: nib.Nifti1Image,
    seg_img: nib.Nifti1Image,
    *,
    title: str = "SimNIBS final_tissues",
    overlay_alpha: float | None = None,
    cut_coords: tuple[float, float, float] | None = None,
) -> bytes:
    if overlay_alpha is None:
        overlay_alpha = MASK_OVERLAY_ALPHA
    _close_all()
    seg_img = squeeze_to_3d(seg_img)
    s = np.asanyarray(seg_img.dataobj, dtype=np.float64)
    vmax = max(float(np.nanmax(s)), 1.0)
    plot_stat_map(
        seg_img,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=None,
        cmap="nipy_spectral",
        vmin=0.0,
        vmax=vmax,
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=True,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def render_segmentation_mosaic_on_t1(
    t1_img: nib.Nifti1Image,
    seg_img: nib.Nifti1Image,
    *,
    title: str = "SimNIBS final_tissues (mosaic)",
    overlay_alpha: float | None = None,
) -> bytes:
    if overlay_alpha is None:
        overlay_alpha = MASK_OVERLAY_ALPHA
    _close_all()
    seg_img = squeeze_to_3d(seg_img)
    s = np.asanyarray(seg_img.dataobj, dtype=np.float64)
    vmax = max(float(np.nanmax(s)), 1.0)
    plot_stat_map(
        seg_img,
        bg_img=t1_img,
        threshold=None,
        cmap="nipy_spectral",
        vmin=0.0,
        vmax=vmax,
        symmetric_cbar=False,
        dim=False,
        display_mode="mosaic",
        draw_cross=False,
        annotate=False,
        colorbar=True,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def plot_roi_on_t1(
    t1_path: str | Path,
    roi_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> None:
    """Save ROI-on-T1 PNG, or show interactively if output_path is None."""
    from ok_plan.geometry import assert_same_space

    t1_img = nib.load(Path(t1_path))
    roi_img = nib.load(Path(roi_path))
    assert_same_space((t1_img, "T1w"), (roi_img, "ROI"))
    png = render_roi_on_t1(t1_img, roi_img, title="Target ROI")
    if output_path is not None:
        Path(output_path).write_bytes(png)
        return
    _close_all()
    buf = io.BytesIO(png)
    im = plt.imread(buf)
    _ = plt.figure(figsize=(10, 10))
    ax = plt.axes((0, 0, 1, 1))
    ax.imshow(im)
    ax.axis("off")
    plt.show()


# Full-opacity overlays for ROI / segmentations / binary masks (Nilearn alpha 1).
MASK_OVERLAY_ALPHA = 1.0

# Pressure / temperature: balance field visibility with readable T1 underlay
# (lower alpha = more visible anatomy through the overlay).
FIELD_OVERLAY_ALPHA = 0.52

# Back-compat names (deprecated): map to opacity semantics
ROI_SEG_OVERLAY_TRANSPARENCY = MASK_OVERLAY_ALPHA
FIELD_OVERLAY_TRANSPARENCY = FIELD_OVERLAY_ALPHA

MASK_COLORS = {
    "scalp": (0.25, 0.45, 1.0),
    "skull": (0.95, 0.2, 0.15),
    "inside_skull": (0.2, 0.85, 0.35),
    "eyes": (1.0, 0.85, 0.15),
    "focus_minus3db": (0.4, 1.0, 0.95),
    "focus_minus6db": (0.85, 0.45, 1.0),
}

# Palette for the combined tissue-mask mosaic.
# Ordered to match DERIVED_MASK_MOSAIC_ORDER: scalp, skull, inside_skull, eyes.
CALMING_MASK_PALETTE: list[tuple[float, float, float]] = [
    (0x08 / 255, 0x4c / 255, 0x61 / 255),  # scalp        — #084c61 deep teal
    (0xdb / 255, 0x3a / 255, 0x34 / 255),  # skull        — #db3a34 red
    (0x17 / 255, 0x7e / 255, 0x89 / 255),  # inside_skull — #177e89 teal
    (0xff / 255, 0xc8 / 255, 0x57 / 255),  # eyes         — #ffc857 amber
]

PRESSURE_CMAP = "viridis"
TEMPERATURE_CMAP = "hot"

# Order for single mosaic of four derived masks (label value -> color index in ListedColormap).
DERIVED_MASK_MOSAIC_ORDER: list[tuple[str, int]] = [
    ("scalp", 1),
    ("skull", 2),
    ("inside_skull", 3),
    ("eyes", 4),
]


def render_combined_derived_masks_mosaic(
    t1_img: nib.Nifti1Image,
    mask_by_key: dict[str, nib.Nifti1Image],
    *,
    title: str = "Derived tissue masks (mosaic)",
    overlay_alpha: float = 1.0,
) -> bytes:
    """One mosaic: discrete labels 1–4 for scalp, skull, inside_skull, eyes on T1w.

    Rendered on a black background for high-contrast display of the muted mask
    palette over the anatomical underlay.
    """
    _close_all()
    first_key = DERIVED_MASK_MOSAIC_ORDER[0][0]
    ref = squeeze_to_3d(mask_by_key[first_key])
    combined = np.zeros(ref.shape, dtype=np.float32)
    for key, lab in DERIVED_MASK_MOSAIC_ORDER:
        m = squeeze_to_3d(mask_by_key[key])
        d = np.asanyarray(m.dataobj, dtype=np.float64) != 0
        if d.shape != combined.shape:
            raise ValueError(f"Mask {key} shape {d.shape} != {combined.shape}")
        combined[d] = float(lab)
    n_masks = len(DERIVED_MASK_MOSAIC_ORDER)
    mask_colors = CALMING_MASK_PALETTE[:n_masks]
    cmap = ListedColormap(
        [(0.0, 0.0, 0.0, 0.0)]
        + [(*c[:3], 1.0) for c in mask_colors]
    )
    stat = nib.Nifti1Image(combined, ref.affine, ref.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        threshold=0.5,
        cmap=cmap,
        vmin=0.0,
        vmax=float(n_masks),
        symmetric_cbar=False,
        dim=False,
        display_mode="mosaic",
        draw_cross=False,
        annotate=False,
        colorbar=True,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
        black_bg=True,
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="black")


def _binary_contour_3d(mask: np.ndarray) -> np.ndarray:
    """Return a boolean array that is True only on the surface of ``mask``."""
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(mask, iterations=1)
    return mask & ~eroded


def render_roi_focus_composite_ortho(
    t1_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    focus_img: nib.Nifti1Image,
    *,
    title: str,
    fg_focus: tuple[float, float, float],
    cut_coords: tuple[float, float, float],
    overlay_alpha: float = 1.0,
) -> bytes:
    """Ortho at ``cut_coords``: focus fill (color), ROI white outline, overlap white outline."""
    _close_all()
    roi_img = squeeze_to_3d(roi_img)
    focus_img = squeeze_to_3d(focus_img)
    r = np.asanyarray(roi_img.dataobj, dtype=np.float64) != 0
    f = np.asanyarray(focus_img.dataobj, dtype=np.float64) != 0
    if r.shape != f.shape:
        raise ValueError(f"ROI shape {r.shape} vs focus {f.shape}")
    roi_edge = _binary_contour_3d(r)
    comp = np.zeros(r.shape, dtype=np.float32)
    comp[f & ~r] = 1.0  # focus only → focus color fill
    comp[f & r] = 1.0   # overlap interior → same focus fill
    comp[roi_edge] = 2.0 # ROI outline always on top → white
    roi_outline_rgb = (1.0, 1.0, 1.0)
    cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),   # 0: background
            (*fg_focus, 1.0),         # 1: focus fill
            (*roi_outline_rgb, 1.0),  # 2: ROI white outline
        ]
    )
    stat = nib.Nifti1Image(comp, roi_img.affine, roi_img.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=0.5,
        cmap=cmap,
        vmin=0.0,
        vmax=2.0,
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=False,
        title=title,
        transparency=float(overlay_alpha),
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def render_roi_focus_outlines_ortho(
    t1_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    focus_img: nib.Nifti1Image,
    *,
    title: str,
    fg_roi: tuple[float, float, float] = (0.95, 0.15, 0.12),
    fg_focus: tuple[float, float, float] = (0.2, 0.7, 1.0),
    cut_coords: tuple[float, float, float],
) -> bytes:
    """Ortho showing only outlines of ROI and focus mask on T1w."""
    _close_all()
    roi_img = squeeze_to_3d(roi_img)
    focus_img = squeeze_to_3d(focus_img)
    r = np.asanyarray(roi_img.dataobj, dtype=np.float64) != 0
    f = np.asanyarray(focus_img.dataobj, dtype=np.float64) != 0
    if r.shape != f.shape:
        raise ValueError(f"ROI shape {r.shape} vs focus {f.shape}")
    roi_edge = _binary_contour_3d(r)
    focus_edge = _binary_contour_3d(f)
    comp = np.zeros(r.shape, dtype=np.float32)
    comp[focus_edge] = 1.0
    comp[roi_edge] = 2.0   # ROI on top
    cmap = ListedColormap([
        (0.0, 0.0, 0.0, 0.0),
        (*fg_focus, 1.0),
        (*fg_roi, 1.0),
    ])
    stat = nib.Nifti1Image(comp, roi_img.affine, roi_img.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=0.5,
        cmap=cmap,
        vmin=0.0,
        vmax=2.0,
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=False,
        title=title,
        transparency=1.0,
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")


def render_focus_atlas_overlays_transparent_ortho(
    t1_img: nib.Nifti1Image,
    focus_img: nib.Nifti1Image,
    region_masks: list[tuple[str, nib.Nifti1Image]],
    *,
    title: str,
    fg_focus: tuple[float, float, float] = (0.2, 0.7, 1.0),
    cut_coords: tuple[float, float, float],
    alpha: float = 0.5,
    palette_name: str = "tab10",
) -> tuple[bytes, list[tuple[str, tuple[float, float, float]]]]:
    """Ortho of the focus plus every overlapping atlas region (50 % transparent each).

    Each region gets a distinct color from ``palette_name``. Voxels in
    region ∩ focus display as a midpoint blend of the two colors. Voxels
    in region but not focus show the region color; voxels in focus but
    not any region show the focus color.

    Returns ``(png_bytes, [(region_name, rgb), ...])`` so callers can render
    a legend.
    """
    _close_all()
    focus_img = squeeze_to_3d(focus_img)
    f = np.asanyarray(focus_img.dataobj, dtype=np.float64) != 0

    regions: list[tuple[str, np.ndarray]] = []
    for name, r_img in region_masks:
        r_img = squeeze_to_3d(r_img)
        r = np.asanyarray(r_img.dataobj, dtype=np.float64) != 0
        if r.shape != f.shape:
            raise ValueError(
                f"Region {name} shape {r.shape} != focus {f.shape}"
            )
        regions.append((name, r))

    n = len(regions)
    if n == 0:
        raise ValueError("No regions provided for combined overlay.")

    src_cmap = plt.colormaps[palette_name]
    colors_rgb: list[tuple[float, float, float]] = [
        tuple(float(v) for v in src_cmap(i)[:3]) for i in range(n)
    ]

    # Label encoding:
    #   0: background
    #   1       : focus only (not in any region)
    #   2..n+1  : region_i (outside focus)
    #   n+2..2n+1 : region_i ∩ focus (blend)
    comp = np.zeros(f.shape, dtype=np.float32)
    union_regions = np.zeros(f.shape, dtype=bool)
    for i, (_nm, r) in enumerate(regions, start=1):
        region_only = r & ~f & ~union_regions
        comp[region_only] = float(1 + i)
        union_regions |= r
    focus_only = f & ~union_regions
    comp[focus_only] = 1.0
    overlap_counter = 0
    covered = np.zeros(f.shape, dtype=bool)
    for i, (_nm, r) in enumerate(regions, start=1):
        overlap = r & f & ~covered
        comp[overlap] = float(1 + n + i)
        covered |= overlap
        overlap_counter += int(np.count_nonzero(overlap))

    entries: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)]
    entries.append((*fg_focus, alpha))
    for rgb in colors_rgb:
        entries.append((*rgb, alpha))
    blend_alpha = min(1.0, alpha + 0.15)
    for rgb in colors_rgb:
        br = (fg_focus[0] + rgb[0]) / 2
        bg = (fg_focus[1] + rgb[1]) / 2
        bb = (fg_focus[2] + rgb[2]) / 2
        entries.append((br, bg, bb, blend_alpha))
    cmap = ListedColormap(entries)

    stat = nib.Nifti1Image(comp, focus_img.affine, focus_img.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=0.5,
        cmap=cmap,
        vmin=0.0,
        vmax=float(2 * n + 1),
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=False,
        title=title,
        transparency=1.0,
        resampling_interpolation="nearest",
    )
    png = figure_to_png_bytes(plt.gcf(), facecolor="white")
    legend = [(name, colors_rgb[i]) for i, (name, _r) in enumerate(regions)]
    return png, legend


def render_roi_focus_transparent_ortho(
    t1_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    focus_img: nib.Nifti1Image,
    *,
    title: str,
    fg_roi: tuple[float, float, float] = (0.95, 0.15, 0.12),
    fg_focus: tuple[float, float, float] = (0.2, 0.7, 1.0),
    cut_coords: tuple[float, float, float],
) -> bytes:
    """Ortho showing ROI and focus as 50 % transparent fills on T1w."""
    _close_all()
    roi_img = squeeze_to_3d(roi_img)
    focus_img = squeeze_to_3d(focus_img)
    r = np.asanyarray(roi_img.dataobj, dtype=np.float64) != 0
    f = np.asanyarray(focus_img.dataobj, dtype=np.float64) != 0
    if r.shape != f.shape:
        raise ValueError(f"ROI shape {r.shape} vs focus {f.shape}")
    comp = np.zeros(r.shape, dtype=np.float32)
    comp[f & ~r] = 1.0       # focus only
    comp[r & ~f] = 2.0       # ROI only
    comp[r & f] = 3.0         # overlap
    cmap = ListedColormap([
        (0.0, 0.0, 0.0, 0.0),
        (*fg_focus, 0.5),
        (*fg_roi, 0.5),
        (
            (fg_roi[0] + fg_focus[0]) / 2,
            (fg_roi[1] + fg_focus[1]) / 2,
            (fg_roi[2] + fg_focus[2]) / 2,
            0.65,
        ),
    ])
    stat = nib.Nifti1Image(comp, roi_img.affine, roi_img.header)
    plot_stat_map(
        stat,
        bg_img=t1_img,
        cut_coords=cut_coords,
        threshold=0.5,
        cmap=cmap,
        vmin=0.0,
        vmax=3.0,
        symmetric_cbar=False,
        dim=False,
        display_mode="ortho",
        draw_cross=False,
        annotate=False,
        colorbar=False,
        title=title,
        transparency=1.0,
        resampling_interpolation="nearest",
    )
    return figure_to_png_bytes(plt.gcf(), facecolor="white")
