"""Verify the 3D engine surface mesh built from the smooth level set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rocketcfd.mask import load_mask
from rocketcfd.gui.viewer3d import _radial_profile, build_engine_mesh

for name in ("examples/nozzle_small.png", "examples/aerospike.png"):
    mask = load_mask(name, 0.001)
    lam = mask.lam[2:-2, 2:-2]
    axis_row = mask.ny / 2 - 0.5
    solid_half = _radial_profile(1.0 - lam, axis_row)
    verts, faces = build_engine_mesh(solid_half, 0.001)
    assert verts is not None, name
    assert np.isfinite(verts).all()
    assert faces.max() < len(verts)
    print(f"{name:32s} verts={len(verts):7d} faces={len(faces):7d} "
          f"x=[{verts[:,0].min():.3f},{verts[:,0].max():.3f}] m "
          f"r_max={np.hypot(verts[:,1], verts[:,2]).max():.3f} m")
    # cutaway half: only z <= 0 vertices
    vh, fh = build_engine_mesh(solid_half, 0.001, half=True)
    assert vh is not None and vh[:, 2].max() < 1e-6, vh[:, 2].max()
print("engine mesh OK (full + half)")
