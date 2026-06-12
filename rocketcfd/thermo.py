"""Thermally perfect gas: frozen-composition cp(T) for combustion mixtures.

The solver's "thermally perfect" gas model keeps the chamber-equilibrium
composition frozen through the nozzle (no recombination) but lets cp — and
therefore gamma — vary with temperature. Species molar heat capacities are
JANAF table values; mixtures are mole-fraction-weighted and fitted with a
single cubic in z = T/1000 over 250–4200 K (fit error < 1 %).

Nondimensionalization: everything here works with cp/R (identical for molar
cp/Ru and mass cp/R_specific), so the polynomial pairs with whatever R_gas
the config carries — composition only shapes the curve. h(0) = 0 reference.

Used by cuda_kernels.build_source (compile-time coefficients) and by the
solver/GUI for CPU-side Mach numbers and initial states.
"""
from __future__ import annotations

import numpy as np

# molar cp/Ru at the anchor temperatures below (JANAF thermochemical tables)
T_ANCHORS = np.array([300., 600., 1000., 1500., 2000., 2500., 3000., 3500., 4000.])
SPECIES_CPR = {
    "H2O": [4.04, 4.37, 4.97, 5.56, 6.16, 6.47, 6.72, 6.89, 7.02],
    "CO2": [4.46, 5.69, 6.53, 7.01, 7.27, 7.40, 7.48, 7.55, 7.62],
    "CO":  [3.50, 3.66, 3.99, 4.19, 4.35, 4.42, 4.47, 4.51, 4.54],
    "H2":  [3.47, 3.51, 3.63, 3.85, 4.13, 4.30, 4.46, 4.60, 4.72],
    "N2":  [3.50, 3.61, 3.93, 4.15, 4.32, 4.39, 4.45, 4.50, 4.54],
    "O2":  [3.53, 3.87, 4.20, 4.36, 4.53, 4.64, 4.73, 4.81, 4.89],
    "OH":  [3.60, 3.57, 3.69, 3.95, 4.19, 4.31, 4.40, 4.47, 4.52],
    "H":   [2.50] * 9,
}
SPECIES_M = {"H2O": 18.015, "CO2": 44.01, "CO": 28.01, "H2": 2.016,
             "N2": 28.013, "O2": 31.999, "OH": 17.007, "H": 1.008}

# Chamber-equilibrium mole fractions at ~70 bar (approximate NASA CEA values;
# minor species folded into their nearest major). Each mixture's molar mass
# is kept consistent with the R_gas of the matching preset in config.py.
COMPOSITIONS = {
    "LOX/RP-1 (kerosene)":    {"CO": 0.31, "CO2": 0.13, "H2O": 0.32,
                               "H2": 0.14, "OH": 0.04, "H": 0.03, "O2": 0.03},
    "LOX/LH2":                {"H2O": 0.62, "H2": 0.25, "OH": 0.07,
                               "H": 0.05, "O2": 0.01},
    "LOX/Ethanol (75%)":      {"H2O": 0.40, "CO2": 0.18, "CO": 0.28,
                               "H2": 0.10, "OH": 0.02, "H": 0.01, "O2": 0.01},
    "UDMH/N2O4 (hypergolic)": {"N2": 0.27, "H2O": 0.33, "CO": 0.12,
                               "CO2": 0.12, "H2": 0.08, "OH": 0.04,
                               "O2": 0.02, "H": 0.02},
    "H2O (steam)":            {"H2O": 1.0},
    "Air (cold gas)":         {"N2": 0.79, "O2": 0.21},
}


def mixture_cpr_anchors(comp: dict[str, float]) -> np.ndarray:
    """Mole-weighted cp/R of a mixture at the anchor temperatures."""
    tot = sum(comp.values())
    out = np.zeros_like(T_ANCHORS)
    for sp, x in comp.items():
        out += (x / tot) * np.asarray(SPECIES_CPR[sp])
    return out


def mixture_molar_mass(comp: dict[str, float]) -> float:
    tot = sum(comp.values())
    return sum(x * SPECIES_M[sp] for sp, x in comp.items()) / tot


def cpr_coeffs(propellant: str, fallback_gamma: float = 1.4) -> tuple:
    """Cubic-fit coefficients (c0..c3) of cp/R in z = T/1000.

    Unknown propellant (or "Custom") degrades to constant cp matching
    fallback_gamma, i.e. the calorically perfect gas — so the thermally
    perfect mode is always well defined.
    """
    comp = COMPOSITIONS.get(propellant)
    if not comp:
        return (fallback_gamma / (fallback_gamma - 1.0), 0.0, 0.0, 0.0)
    z = T_ANCHORS / 1000.0
    c = np.polyfit(z, mixture_cpr_anchors(comp), 3)[::-1]   # ascending order
    return tuple(float(v) for v in c)


# ------------------------------------------------------------ evaluation
def cpr(c, T):
    """cp/R at temperature T (scalar or array)."""
    z = np.asarray(T, dtype=np.float64) / 1000.0
    return c[0] + z * (c[1] + z * (c[2] + z * c[3]))


def hr(c, T):
    """h/R with h(0) = 0: integral of cp/R dT."""
    z = np.asarray(T, dtype=np.float64) / 1000.0
    return 1000.0 * z * (c[0] + z * (c[1] / 2 + z * (c[2] / 3 + z * c[3] / 4)))


def er(c, T):
    """e/R = h/R - T (ideal gas, constant R)."""
    return hr(c, T) - np.asarray(T, dtype=np.float64)


def gamma_of_T(c, T):
    cp_r = cpr(c, T)
    return cp_r / (cp_r - 1.0)


def T_from_e(c, e_over_R, T_guess=1000.0, iters=6):
    """Invert e(T) = e_over_R via Newton (vectorized)."""
    T = np.clip(np.asarray(T_guess, dtype=np.float64), 60.0, 5900.0)
    T = np.broadcast_to(T, np.shape(e_over_R)).copy() \
        if np.shape(e_over_R) else float(T)
    for _ in range(iters):
        f = er(c, T) - e_over_R
        cv = np.maximum(cpr(c, T) - 1.0, 0.5)
        T = np.clip(T - f / cv, 50.0, 6000.0)
    return T


# ------------------------------------------------------------- self-test
if __name__ == "__main__":
    R_UNIV = 8314.46
    for name, comp in COMPOSITIONS.items():
        c = cpr_coeffs(name)
        M = mixture_molar_mass(comp)
        fit_err = np.max(np.abs(cpr(c, T_ANCHORS) - mixture_cpr_anchors(comp))
                         / mixture_cpr_anchors(comp))
        g300, g3500 = gamma_of_T(c, 300.0), gamma_of_T(c, 3500.0)
        # round-trip e -> T
        Ts = np.array([300.0, 1500.0, 3500.0])
        Tr = T_from_e(c, er(c, Ts), 1000.0)
        rt = np.max(np.abs(Tr - Ts))
        print(f"{name:26s} M={M:6.2f} R={R_UNIV/M:6.1f}  "
              f"gam(300K)={g300:.3f} gam(3500K)={g3500:.3f}  "
              f"fit_err={fit_err*100:.2f}%  newton_err={rt:.2e} K")
        assert fit_err < 0.02, name
        assert rt < 0.05, name
    # CP fallback must reproduce gamma exactly
    c = cpr_coeffs("Custom", fallback_gamma=1.22)
    assert abs(gamma_of_T(c, 2000.0) - 1.22) < 1e-12
    print("thermo self-test OK")
