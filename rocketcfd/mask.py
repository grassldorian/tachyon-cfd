"""PNG/SVG -> computational mesh.

Pixel classification:
  black  -> wall            (type 1)
  white  -> flow space      (type 0)
  blue   -> pressure inlet  (type 2)
  red    -> pressure outlet (type 4) — absorbs waves at farfield pressure

The solver grid carries a 2-cell halo on every side (type 3 = farfield),
so array shape is (ny + 4, nx + 4). Row 0 of the PNG maps to row 2 of the
grid (image convention: y down).

Smooth embedded boundary (cut-cell mesh)
----------------------------------------
With smooth=True the binary wall mask is converted into a sub-pixel smooth
surface: the solid indicator is Gaussian-filtered and its 0.5 level set
defines the wall. From the level set sampled at cell corners we compute

  ax[j,i]  aperture (open fraction 0..1) of the x-face between cells i-1, i
  ay[j,i]  aperture of the y-face between cells j-1, j
  lam[j,i] fluid volume fraction of cell (i, j)

Cells are reclassified: solid only if lam < 2 %; partially-cut cells become
fluid cells whose fluxes are aperture-weighted. The solver then sees the
smooth surface (wall pressure acts along the true local normal) instead of
a pixel staircase.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

FLUID, WALL, INLET, FARFIELD, OUTLET = 0, 1, 2, 3, 4
LAM_SOLID = 0.02          # below this volume fraction a cell is solid


class DomainMask:
    def __init__(self, cell_type: np.ndarray, wall_dist: np.ndarray,
                 meters_per_pixel: float, rgb: np.ndarray,
                 ax: np.ndarray, ay: np.ndarray, lam: np.ndarray):
        self.cell_type = cell_type            # (ny+4, nx+4) uint8, includes halo
        self.wall_dist = wall_dist            # (ny+4, nx+4) float32 [m]
        self.dx = float(meters_per_pixel)     # uniform cell size [m]
        self.rgb = rgb                        # original image (ny, nx, 3) uint8
        self.ax = ax                          # (ny+4, nx+4) f32 x-face apertures
        self.ay = ay                          # (ny+4, nx+4) f32 y-face apertures
        self.lam = lam                        # (ny+4, nx+4) f32 volume fractions
        self.smooth = False                   # set by load_mask
        self.ny = cell_type.shape[0] - 4
        self.nx = cell_type.shape[1] - 4

    @property
    def n_fluid(self) -> int:
        return int(np.count_nonzero(self.cell_type == FLUID))

    @property
    def n_inlet(self) -> int:
        return int(np.count_nonzero(self.cell_type == INLET))


def classify_pixels(rgb: np.ndarray) -> np.ndarray:
    """Classify each pixel as FLUID / WALL / INLET from RGB values.

    Tolerant of anti-aliasing: blue wins if clearly blue-dominant,
    otherwise dark pixels are walls and the rest is fluid.
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    lum = (r + g + b) // 3

    is_blue = (b > 100) & (b - np.maximum(r, g) > 40)
    is_red = (r > 100) & (r - np.maximum(g, b) > 40) & ~is_blue
    is_wall = (lum < 128) & ~is_blue & ~is_red

    ctype = np.full(rgb.shape[:2], FLUID, dtype=np.uint8)
    ctype[is_wall] = WALL
    ctype[is_blue] = INLET
    ctype[is_red] = OUTLET
    return ctype


def rasterize_svg(path: str, target_px: int = 1000) -> np.ndarray:
    """Render an SVG to an RGB array (white background), long side = target_px."""
    import fitz                                # PyMuPDF
    doc = fitz.open(path)
    page = doc[0]
    long_side = max(page.rect.width, page.rect.height)
    if long_side <= 0:
        raise ValueError("SVG has no drawable area")
    scale = target_px / long_side
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8)
    rgb = rgb.reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    doc.close()
    return rgb


def load_image(path: str, svg_raster_px: int = 1000) -> np.ndarray:
    if str(path).lower().endswith(".svg"):
        return rasterize_svg(path, svg_raster_px)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resample_rgb(rgb: np.ndarray, mesh_scale: float) -> np.ndarray:
    """Resample a classification image by ``mesh_scale`` (nearest-neighbour).

    Nearest-neighbour keeps the wall/inlet/outlet colours crisp so
    ``classify_pixels`` stays deterministic (no blended intermediate colours).
    The physical size of the engine is unchanged because the caller scales
    ``meters_per_pixel`` by the same factor.
    """
    ny, nx = rgb.shape[:2]
    new_w = max(8, int(round(nx * mesh_scale)))
    new_h = max(8, int(round(ny * mesh_scale)))
    if (new_w, new_h) == (nx, ny):
        return rgb
    img = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def _face_aperture(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Open fraction of a face whose endpoint level-set values are pa, pb
    (positive = fluid). Linear zero crossing."""
    both_open = (pa > 0) & (pb > 0)
    cross = (pa > 0) ^ (pb > 0)
    denom = pa - pb
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t = pa / denom                       # crossing position measured from a
    frac = np.where(pa > 0, t, 1.0 - t)  # open part is on the positive side
    out = np.where(both_open, 1.0, np.where(cross, frac, 0.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _cut_cell_geometry(ctype_img: np.ndarray, sigma: float):
    """Smooth level set -> face apertures and cell volume fractions.

    Returns (ax, ay, lam) on the interior (ny, nx) grid:
      ax[j, i] = aperture of the west face of cell (j, i), shape (ny, nx+1)
      ay[j, i] = aperture of the north face of cell (j, i), shape (ny+1, nx)
      lam      = fluid volume fraction, shape (ny, nx)
    """
    ny, nx = ctype_img.shape
    solid = (ctype_img == WALL).astype(np.float32)
    s = ndimage.gaussian_filter(solid, sigma=sigma, mode="nearest")

    # level set at cell corners (nodes), phi > 0 in fluid
    sp = np.pad(s, 1, mode="edge")
    node = 0.25 * (sp[:-1, :-1] + sp[:-1, 1:] + sp[1:, :-1] + sp[1:, 1:])
    phi = (0.5 - node).astype(np.float32)            # (ny+1, nx+1)

    # x-faces: endpoints are nodes (j, i) and (j+1, i)   -> (ny, nx+1)
    ax = _face_aperture(phi[:-1, :], phi[1:, :])
    # y-faces: endpoints are nodes (j, i) and (j, i+1)   -> (ny+1, nx)
    ay = _face_aperture(phi[:, :-1], phi[:, 1:])

    # volume fraction: average of a 4x4 bilinear subsampling of phi > 0
    p00 = phi[:-1, :-1]; p10 = phi[:-1, 1:]
    p01 = phi[1:, :-1];  p11 = phi[1:, 1:]
    lam = np.zeros((ny, nx), dtype=np.float32)
    qp = (np.arange(4) + 0.5) / 4.0
    for v in qp:
        for u in qp:
            val = (p00 * (1 - u) * (1 - v) + p10 * u * (1 - v)
                   + p01 * (1 - u) * v + p11 * u * v)
            lam += (val > 0)
    lam /= 16.0
    return ax, ay, lam


def load_mask(path: str, meters_per_pixel: float, svg_raster_px: int = 1000,
              smooth: bool = True, sigma: float = 1.2,
              mesh_scale: float = 1.0) -> DomainMask:
    rgb = load_image(path, svg_raster_px)
    if mesh_scale and abs(mesh_scale - 1.0) > 1e-6:
        rgb = resample_rgb(rgb, mesh_scale)
        # keep the physical size fixed: finer mesh -> smaller cells
        meters_per_pixel = meters_per_pixel / mesh_scale
    ny, nx = rgb.shape[:2]

    ctype_img = classify_pixels(rgb)

    # ---- cut-cell geometry (smooth sub-pixel walls) ----
    ax_pad = np.ones((ny + 4, nx + 4), dtype=np.float32)
    ay_pad = np.ones((ny + 4, nx + 4), dtype=np.float32)
    lam_pad = np.ones((ny + 4, nx + 4), dtype=np.float32)
    is_smooth = bool(smooth and (ctype_img == WALL).any())
    if is_smooth:
        ax_i, ay_i, lam_i = _cut_cell_geometry(ctype_img, sigma)
        # inlets/outlets are boundary conditions, not geometry: fully open
        inl = ctype_img == INLET
        outl = ctype_img == OUTLET
        lam_i[inl | outl] = 1.0
        # reclassify: solid only where almost no fluid volume remains
        new_type = np.where(lam_i < LAM_SOLID, WALL,
                            np.where(inl, INLET,
                                     np.where(outl, OUTLET, FLUID))).astype(np.uint8)
        ctype_img = new_type
        # faces touching a solid cell are fully closed, so the whole wall
        # segment belongs to the adjacent fluid cell (otherwise a surface
        # lying exactly on a pixel face loses half its area to the solid)
        sol = ctype_img == WALL
        solx = np.pad(sol, ((0, 0), (1, 1)), constant_values=False)
        ax_i = np.where(solx[:, :-1] | solx[:, 1:], 0.0, ax_i)
        soly = np.pad(sol, ((1, 1), (0, 0)), constant_values=False)
        ay_i = np.where(soly[:-1, :] | soly[1:, :], 0.0, ay_i)
        # faces of the padded grid: x-face stored at right cell (i in 2..nx+2)
        ax_pad[2:-2, 2:nx + 3] = ax_i
        ay_pad[2:ny + 3, 2:-2] = ay_i
        lam_pad[2:-2, 2:-2] = np.maximum(lam_i, 0.0)

    cell_type = np.full((ny + 4, nx + 4), FARFIELD, dtype=np.uint8)
    cell_type[2:-2, 2:-2] = ctype_img

    # Wall distance for the SST model: Euclidean distance from every cell
    # to the nearest wall pixel, in meters. If the image contains no walls
    # at all, fall back to a large constant.
    interior = cell_type[2:-2, 2:-2]
    if (interior == WALL).any():
        dist_px = ndimage.distance_transform_edt(interior != WALL)
    else:
        dist_px = np.full((ny, nx), 1e6, dtype=np.float64)
    wall_dist = np.full((ny + 4, nx + 4), 1e6, dtype=np.float32)
    wall_dist[2:-2, 2:-2] = np.maximum(dist_px * meters_per_pixel,
                                       0.5 * meters_per_pixel).astype(np.float32)

    dm = DomainMask(cell_type, wall_dist, meters_per_pixel, rgb,
                    ax_pad, ay_pad, lam_pad)
    dm.smooth = is_smooth
    return dm
