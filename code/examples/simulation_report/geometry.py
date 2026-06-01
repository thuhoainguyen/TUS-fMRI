"""Require aligned NIfTI grids (same shape and affine) — no resampling."""

from __future__ import annotations

import numpy as np


def assert_same_space(*named_images: tuple[object, str], atol: float = 1e-5) -> None:
    """Raise ValueError if 3D shapes or affines differ."""
    if not named_images:
        return
    ref_img, ref_name = named_images[0]
    ref_shape = tuple(int(x) for x in np.asarray(ref_img.shape)[:3])
    ref_affine = np.asarray(ref_img.affine, dtype=float)
    for img, name in named_images[1:]:
        sh = tuple(int(x) for x in np.asarray(img.shape)[:3])
        if sh != ref_shape:
            raise ValueError(
                f"Shape mismatch: {name} has shape {sh}, "
                f"{ref_name} has {ref_shape}. "
                "All inputs must share the same voxel grid (no resampling). "
                "Resample volumes externally if CHARM and k-plan grids differ."
            )
        aff = np.asarray(img.affine, dtype=float)
        if not np.allclose(aff, ref_affine, atol=atol, rtol=0.0):
            raise ValueError(
                f"Affine mismatch between {name} and {ref_name}. "
                "All inputs must be in the same world space (no resampling)."
            )
