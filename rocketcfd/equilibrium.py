"""Shifting-equilibrium combustion thermodynamics (a mini-CEA).

Computes chemical equilibrium of C-H-O(-N) combustion products by Gibbs
minimization, the adiabatic flame temperature, and isentropes from the
chamber — the ingredients for the solver's "equilibrium" gas model, where
the composition shifts (recombines) as the gas expands instead of staying
frozen. Recombination releases heat: equilibrium Isp exceeds frozen by
1-4 % and the effective gamma stays near the chamber value, keeping exit
pressures higher (no spurious sea-level separation of big bells).

Method: base species CO2 / H2O / O2 (+ inert N2) carry the element
balances; minor species (CO, H2, OH, H, O) follow from equilibrium
relations u_i = -dG_i/RT + sum(nu_ib u_b) + (1 - sum(nu)) (ln N - ln p~),
so Newton only iterates on the base log-moles and ln N (<= 5 unknowns).
Species data: JANAF cp anchors (shared with thermo.py), standard formation
enthalpies and entropies at 298.15 K. Condensed species and CH4 are not
included — fine above ~700 K, which is all a nozzle interior sees.
"""
from __future__ import annotations

import numpy as np

from .thermo import T_ANCHORS, SPECIES_CPR, SPECIES_M

RU = 8.31446          # J/(mol K)
P_REF = 1.0e5         # standard pressure for mu0 [Pa]
T298 = 298.15

# formation enthalpy [J/mol] and standard entropy [J/(mol K)] at 298.15 K
DHF298 = {"H2O": -241830.0, "CO2": -393520.0, "CO": -110530.0,
          "H2": 0.0, "O2": 0.0, "N2": 0.0,
          "OH": 37300.0, "H": 217990.0, "O": 249170.0}
S298 = {"H2O": 188.84, "CO2": 213.79, "CO": 197.66,
        "H2": 130.68, "O2": 205.15, "N2": 191.61,
        "OH": 183.71, "H": 114.72, "O": 161.06}
ATOMS = {"H2O": {"H": 2, "O": 1}, "CO2": {"C": 1, "O": 2},
         "CO": {"C": 1, "O": 1}, "H2": {"H": 2}, "O2": {"O": 2},
         "N2": {"N": 2}, "OH": {"H": 1, "O": 1}, "H": {"H": 1}, "O": {"O": 1}}

# JANAF tables only run 300-4000 K here; O is ~H (monatomic, cp/R = 2.5)
SPECIES_CPR = dict(SPECIES_CPR)
SPECIES_CPR["O"] = [2.5] * len(T_ANCHORS)
SPECIES_M = dict(SPECIES_M)
SPECIES_M["O"] = 16.0

# cubic fits of cp/R in z = T/1000 per species (ascending coefficients)
_CPFIT = {sp: np.polyfit(T_ANCHORS / 1000.0, np.asarray(v, float), 3)[::-1]
          for sp, v in SPECIES_CPR.items()}


def _cp_R(sp, T):
    c = _CPFIT[sp]
    z = T / 1000.0
    return c[0] + z * (c[1] + z * (c[2] + z * c[3]))


def _h(sp, T):
    """Molar enthalpy [J/mol] incl. formation enthalpy."""
    c = _CPFIT[sp]
    def H(t):
        z = t / 1000.0
        return 1000.0 * z * (c[0] + z * (c[1] / 2 + z * (c[2] / 3 + z * c[3] / 4)))
    return DHF298[sp] + RU * (H(T) - H(T298))


def _s0(sp, T):
    """Standard molar entropy [J/(mol K)] at p = 1 bar."""
    c = _CPFIT[sp]
    def S(t):
        z = t / 1000.0
        return c[0] * np.log(z) + z * (c[1] + z * (c[2] / 2 + z * c[3] / 3))
    return S298[sp] + RU * (S(T) - S(T298))


def _mu0_RT(sp, T):
    return (_h(sp, T) - T * _s0(sp, T)) / (RU * T)


# ---------------------------------------------------------------- reactants
# fuel/oxidizer: components as (atoms-dict, molar mass g/mol, dHf J/mol)
REACTANTS = {
    "LOX/RP-1 (kerosene)": dict(
        fuel=[({"C": 1, "H": 1.95}, 13.97, -25000.0)],
        ox=[({"O": 2}, 32.0, -12980.0)], OF=2.27),
    "LOX/LH2": dict(
        fuel=[({"H": 2}, 2.016, -9010.0)],
        ox=[({"O": 2}, 32.0, -12980.0)], OF=5.5),
    "LOX/Ethanol (75%)": dict(
        fuel=[({"C": 2, "H": 6, "O": 1}, 46.07, -277600.0),
              ({"H": 2, "O": 1}, 18.015, -285830.0)],
        fuel_w=[0.75, 0.25],
        ox=[({"O": 2}, 32.0, -12980.0)], OF=1.4),
    "UDMH/N2O4 (hypergolic)": dict(
        fuel=[({"C": 2, "H": 8, "N": 2}, 60.10, 48300.0)],
        ox=[({"N": 2, "O": 4}, 92.01, -19560.0)], OF=2.6),
    "LOX/CH4 (methalox)": dict(
        fuel=[({"C": 1, "H": 4}, 16.04, -89000.0)],     # liquid methane
        ox=[({"O": 2}, 32.0, -12980.0)], OF=3.6),
    "MMH/NTO (hypergolic)": dict(
        fuel=[({"C": 1, "H": 6, "N": 2}, 46.07, 54200.0)],   # MMH (l)
        ox=[({"N": 2, "O": 4}, 92.01, -19560.0)], OF=2.16),
    "N2O/HTPB (hybrid)": dict(
        fuel=[({"C": 4, "H": 6}, 54.09, -50000.0)],     # HTPB (C4H6 unit)
        ox=[({"N": 2, "O": 1}, 44.01, 82050.0)], OF=7.0),    # N2O (endothermic)
    "H2O2/RP-1": dict(
        fuel=[({"C": 1, "H": 1.95}, 13.97, -25000.0)],
        ox=[({"H": 2, "O": 2}, 34.01, -187800.0)], OF=7.0),  # H2O2 (l)
}


def reactant_state(propellant):
    """Atom vector b [mol/kg propellant] and reactant enthalpy [J/kg]."""
    r = REACTANTS[propellant]
    of = r["OF"]
    wf, wo = 1.0 / (1.0 + of), of / (1.0 + of)
    fw = r.get("fuel_w", [1.0] * len(r["fuel"]))
    b = {}
    h = 0.0
    for (atoms, M, dhf), w in zip(r["fuel"], fw):
        nmol = wf * w * 1000.0 / M                  # mol/kg propellant
        h += nmol * dhf
        for el, na in atoms.items():
            b[el] = b.get(el, 0.0) + nmol * na
    for atoms, M, dhf in r["ox"]:
        nmol = wo * 1000.0 / M
        h += nmol * dhf
        for el, na in atoms.items():
            b[el] = b.get(el, 0.0) + nmol * na
    return b, h


# ---------------------------------------------------------------- equilibrium
# minor species in terms of bases (stoichiometric coefficients nu)
_MINOR_NU = {
    "CO": {"CO2": 1.0, "O2": -0.5},
    "H2": {"H2O": 1.0, "O2": -0.5},
    "OH": {"H2O": 0.5, "O2": 0.25},
    "H":  {"H2O": 0.5, "O2": -0.25},
    "O":  {"O2": 0.5},
}


def equilibrium(p, T, b, guess=None):
    """Equilibrium composition at (p [Pa], T [K]) for atom vector b [mol/kg].

    Returns dict with n (mol/kg per species), N (total), M (kg/mol mixture),
    h, e, s (per kg), rho, R_eff. `guess` is the u-vector from a previous
    solve (continuation).
    """
    has_C = b.get("C", 0.0) > 0.0
    has_N = b.get("N", 0.0) > 0.0
    bases = (["CO2"] if has_C else []) + ["H2O", "O2"] + (["N2"] if has_N else [])
    minors = ([] if not has_C else ["CO"]) + ["H2", "OH", "H", "O"]
    species = bases + minors
    elems = (["C"] if has_C else []) + ["H", "O"] + (["N"] if has_N else [])
    nb = len(bases)

    mu = {sp: _mu0_RT(sp, T) for sp in species}
    # dG/RT of each minor's formation-from-bases reaction
    dg = {sp: mu[sp] - sum(nu * mu[bs] for bs, nu in _MINOR_NU[sp].items()
                           if bs in mu)
          for sp in minors}
    lnp = np.log(p / P_REF)

    # initial guess: stoichiometric-ish majors
    if guess is None:
        u = {}
        if has_C:
            u["CO2"] = np.log(max(0.4 * b["C"], 1e-8))
        u["H2O"] = np.log(max(0.4 * b["H"] / 2.0, 1e-8))
        u["O2"] = np.log(max(0.02 * b["O"] / 2.0, 1e-8))
        if has_N:
            u["N2"] = np.log(max(b["N"] / 2.0, 1e-12))
        x = np.array([u[bs] for bs in bases] + [np.log(sum(b.values()))])
    else:
        x = guess.copy()

    bvec = np.array([b[e] for e in elems])
    for it in range(80):
        ub = dict(zip(bases, x[:nb]))
        lnN = x[nb]
        un = {}
        for sp in minors:
            s = sum(nu * ub[bs] for bs, nu in _MINOR_NU[sp].items() if bs in ub)
            nsum = sum(nu for bs, nu in _MINOR_NU[sp].items() if bs in ub)
            un[sp] = -dg[sp] + s + (1.0 - nsum) * (lnN - lnp)
        u_all = {**ub, **un}
        n = {sp: np.exp(min(u_all[sp], 60.0)) for sp in species}
        # derivative of each species' u wrt unknowns
        dudx = {}
        for k, bs in enumerate(bases):
            dudx[bs] = np.eye(nb + 1)[k]
        for sp in minors:
            d = np.zeros(nb + 1)
            nsum = 0.0
            for bs, nu in _MINOR_NU[sp].items():
                if bs in ub:
                    d[bases.index(bs)] += nu
                    nsum += nu
            d[nb] += 1.0 - nsum
            dudx[sp] = d
        Ntot = sum(n.values())
        F = np.zeros(nb + 1)
        J = np.zeros((nb + 1, nb + 1))
        for ie, el in enumerate(elems):
            for sp in species:
                a = ATOMS[sp].get(el, 0.0)
                if a:
                    F[ie] += a * n[sp]
                    J[ie] += a * n[sp] * dudx[sp]
            F[ie] -= bvec[ie]
        F[nb] = np.log(Ntot) - lnN
        for sp in species:
            J[nb] += (n[sp] / Ntot) * dudx[sp]
        J[nb, nb] -= 1.0
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -F, rcond=None)[0]
        dx = np.clip(dx, -2.0, 2.0)
        x = x + dx
        if np.max(np.abs(dx)) < 1e-11:
            converged = True
            break
    else:
        converged = False

    ub = dict(zip(bases, x[:nb]))
    lnN = x[nb]
    un = {}
    for sp in minors:
        s = sum(nu * ub[bs] for bs, nu in _MINOR_NU[sp].items() if bs in ub)
        nsum = sum(nu for bs, nu in _MINOR_NU[sp].items() if bs in ub)
        un[sp] = -dg[sp] + s + (1.0 - nsum) * (lnN - lnp)
    u_all = {**ub, **un}
    n = {sp: float(np.exp(min(u_all[sp], 60.0))) for sp in species}
    N = sum(n.values())
    M = 1.0 / N                                     # kg/mol (n is mol/kg)
    h = sum(n[sp] * _h(sp, T) for sp in species)    # J/kg
    s = sum(n[sp] * (_s0(sp, T)
                     - RU * np.log(max(n[sp] / N, 1e-300) * p / P_REF))
            for sp in species)                      # J/(kg K)
    rho = p / (N * RU * T)
    return dict(n=n, N=N, M=M, h=h, e=h - p / rho, s=s, rho=rho,
                R_eff=N * RU, u=x, T=T, p=p, ok=converged)


def flame_T(p, b, h_react, T_lo=1500.0, T_hi=4300.0):
    """Adiabatic flame temperature: h_products(T, p) = h_react (bisection)."""
    guess = None
    for _ in range(60):
        T = 0.5 * (T_lo + T_hi)
        eq = equilibrium(p, T, b, guess)
        guess = eq["u"]
        if eq["h"] > h_react:
            T_hi = T
        else:
            T_lo = T
        if T_hi - T_lo < 0.01:
            break
    return 0.5 * (T_lo + T_hi)


def T_at_sp(p, s_target, b, T_guess, guess=None):
    """Temperature on an isentrope: s(p, T) = s_target (Newton on T)."""
    T = T_guess
    eq = None
    for _ in range(40):
        eq = equilibrium(p, T, b, guess)
        guess = eq["u"]
        # ds/dT at constant p ~ cp_eq/T: estimate with small perturbation
        eq2 = equilibrium(p, T * 1.002, b, guess)
        dsdT = (eq2["s"] - eq["s"]) / (0.002 * T)
        dT = (s_target - eq["s"]) / max(dsdT, 1e-9)
        dT = np.clip(dT, -0.15 * T, 0.15 * T)
        T += dT
        if abs(dT) < 0.005:
            break
    return T, equilibrium(p, T, b, guess)


# ---------------------------------------------------------------- GPU tables
# Regular-grid property tables on (log10 rho, log10 T) — smooth and
# well-conditioned everywhere (unlike (rho,e), which degenerates where e
# barely varies with T). The GPU inverts e -> T with a warm-started Newton
# using the CV table, mirroring the thermally-perfect mode.
#   E  (J/kg)      internal energy
#   RE (J/(kg K))  effective gas constant: p = rho * RE * T
#   A  (m/s)       equilibrium (shifting) sound speed
#   CV (J/(kg K))  (de/dT)|_rho, for the Newton inversion
# Tables depend only on the propellant atom vector -> cached on disk.
NLR, NLT = 64, 96
LR_MIN, LR_MAX = -4.0, 1.7          # log10 rho [kg/m^3]
# down to 110 K: cold hydrolox plumes reach ~120-200 K, and clamping at the
# table floor makes EOS-inconsistent cells that show up as pixel noise
LT_MIN, LT_MAX = np.log10(110.0), np.log10(4400.0)


def _cache_path(propellant):
    import hashlib
    import tempfile
    from pathlib import Path
    b, _hr = reactant_state(propellant)
    key = hashlib.md5(("v3" + repr(sorted(b.items()))).encode()).hexdigest()[:12]
    d = Path(tempfile.gettempdir()) / "tachyon_eqcache"
    d.mkdir(exist_ok=True)
    return d / f"eq_{key}.npz"


T_FREEZE = 900.0    # below this, real exhaust chemistry is kinetically
                    # frozen anyway — and low-T equilibrium Newton is fragile


def _frozen_eval(n, p, T):
    """Properties of a frozen composition n [mol/kg] at (p, T)."""
    N = sum(n.values())
    h = sum(n[sp] * _h(sp, T) for sp in n)
    s = sum(n[sp] * (_s0(sp, T)
                     - RU * np.log(max(n[sp] / N, 1e-300) * p / P_REF))
            for sp in n)
    return dict(n=n, N=N, h=h, s=s, rho=p / (N * RU * T), R_eff=N * RU)


def _sweep(propellant):
    """Equilibrium solves over a (p, T) grid. Below T_FREEZE (or on solver
    failure) the composition is frozen at the last good equilibrium of the
    same pressure line."""
    b, _ = reactant_state(propellant)
    pg = np.geomspace(8.0, 4.0e7, 80)
    Tg = np.linspace(4400.0, 105.0, 100)
    H = np.zeros((len(pg), len(Tg)))
    S = np.zeros_like(H)
    RHO = np.zeros_like(H)
    REF = np.zeros_like(H)
    for ip, p in enumerate(pg):
        guess = None
        frozen_n = None     # composition pinned at exactly T_FREEZE
        prev = None
        for it, T in enumerate(Tg):
            eq = None
            if T >= T_FREEZE:
                try:
                    cand = equilibrium(p, T, b, guess)
                    if cand["ok"] and np.isfinite(cand["h"]) and cand["N"] > 0:
                        eq = cand
                        guess = cand["u"]
                except Exception:
                    eq = None
                if eq is None and prev is not None:    # rare mid-line failure
                    eq = {**_frozen_eval(prev["n"], p, T), "n": prev["n"]}
            if eq is None:                              # below the freeze point
                if frozen_n is None:
                    cand = equilibrium(p, T_FREEZE, b, guess)
                    frozen_n = cand["n"]
                    guess = cand["u"]
                eq = _frozen_eval(frozen_n, p, T)
            prev = eq
            H[ip, it] = eq["h"]
            S[ip, it] = eq["s"]
            RHO[ip, it] = eq["rho"]
            REF[ip, it] = eq["R_eff"]
    return pg, Tg, H, S, RHO, REF, b


def build_tables(propellant):
    """Build (or load cached) regular-grid float32 tables for the GPU."""
    path = _cache_path(propellant)
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}

    pg, Tg, H, S, RHO, REF, b = _sweep(propellant)
    E_pt = H - pg[:, None] / RHO                       # internal energy J/kg
    lr_ax = np.linspace(LR_MIN, LR_MAX, NLR)
    lt_ax = np.linspace(LT_MIN, LT_MAX, NLT)

    # separable resampling: at each sweep T (column), data lie on a smooth
    # p-parameterized line -> 1D interp in log rho; then interp T columns
    def to_grid(Q):
        mid = np.zeros((NLR, len(Tg)))
        for it in range(len(Tg)):
            lr_line = np.log10(RHO[:, it])
            order = np.argsort(lr_line)
            mid[:, it] = np.interp(lr_ax, lr_line[order], Q[:, it][order])
        out = np.zeros((NLR, NLT))
        ltg = np.log10(Tg)[::-1]
        for ir in range(NLR):
            out[ir] = np.interp(lt_ax, ltg, mid[ir, ::-1])
        return out

    Eg = to_grid(E_pt)
    REg = to_grid(REF)
    Sg = to_grid(S)
    Tlin = 10.0 ** lt_ax
    rlin = 10.0 ** lr_ax
    Pg = rlin[:, None] * REg * Tlin[None, :]

    # CV = (de/dT)|_rho, kept >= 0.4*RE for a safe Newton
    CVg = np.gradient(Eg, axis=1) / np.gradient(Tlin)[None, :]
    CVg = np.maximum(CVg, 0.4 * REg)

    # equilibrium sound speed: a^2 = (dp/drho)_s = p_rho - p_T s_rho / s_T
    dr = np.gradient(rlin)
    dT = np.gradient(Tlin)
    p_r = np.gradient(Pg, axis=0) / dr[:, None]
    p_T = np.gradient(Pg, axis=1) / dT[None, :]
    s_r = np.gradient(Sg, axis=0) / dr[:, None]
    s_T = np.gradient(Sg, axis=1) / dT[None, :]
    A2 = p_r - p_T * s_r / np.where(np.abs(s_T) > 1e-12, s_T, 1e-12)
    A2 = np.clip(A2, 1.02 * Pg / rlin[:, None], 2.0 * Pg / rlin[:, None])
    Ag = np.sqrt(A2)

    out = dict(lr_ax=lr_ax.astype(np.float32), lt_ax=lt_ax.astype(np.float32),
               E=Eg.astype(np.float32), RE=REg.astype(np.float32),
               A=Ag.astype(np.float32), CV=CVg.astype(np.float32))
    np.savez_compressed(path, **out)
    return out


def table_interp(tab, name, rho, T):
    """Vectorized bilinear lookup of table `name` at (rho, T) — the numpy
    twin of the kernel's eq_lerp."""
    lr_ax, lt_ax = tab["lr_ax"], tab["lt_ax"]
    A = tab[name]
    t_lo, t_hi = 10.0 ** lt_ax[0] * 1.01, 10.0 ** lt_ax[-1] * 0.99
    lr = np.log10(np.maximum(np.asarray(rho, np.float64), 1e-6))
    lt = np.log10(np.clip(np.asarray(T, np.float64), t_lo, t_hi))
    x = np.clip((lr - lr_ax[0]) / (lr_ax[1] - lr_ax[0]), 0, len(lr_ax) - 1.001)
    y = np.clip((lt - lt_ax[0]) / (lt_ax[1] - lt_ax[0]), 0, len(lt_ax) - 1.001)
    i = x.astype(int)
    j = y.astype(int)
    fx, fy = x - i, y - j
    return (A[i, j] * (1 - fx) * (1 - fy) + A[i + 1, j] * fx * (1 - fy)
            + A[i, j + 1] * (1 - fx) * fy + A[i + 1, j + 1] * fx * fy)


def ambient_state(tab, p_far, T_far):
    """(rho, e, R_eff) of the ambient, self-consistent with the tables."""
    rho = p_far / (400.0 * T_far)
    for _ in range(6):
        re = float(table_interp(tab, "RE", rho, T_far))
        rho = p_far / (re * T_far)
    e = float(table_interp(tab, "E", rho, T_far))
    return rho, e, re


def chamber_isentrope(propellant, p0, T0, npts=40):
    """Inlet table: pressure ratio pr = p/p0 from ~0.35 to 1 along the
    chamber isentrope. Returns (pr, T, V, R_eff, i_choke) where V is from
    exact enthalpy conservation and i_choke is the sonic index."""
    b, _ = reactant_state(propellant)
    eq0 = equilibrium(p0, T0, b)
    s0, h0 = eq0["s"], eq0["h"]
    prs = np.linspace(1.0, 0.35, npts)
    T = np.zeros(npts)
    V = np.zeros(npts)
    Reff = np.zeros(npts)
    rho = np.zeros(npts)
    guess, Tg = eq0["u"], T0
    for i, pr in enumerate(prs):
        Tg, eq = T_at_sp(pr * p0, s0, b, Tg, guess)
        guess = eq["u"]
        T[i] = Tg
        V[i] = np.sqrt(max(2.0 * (h0 - eq["h"]), 0.0))
        Reff[i] = eq["R_eff"]
        rho[i] = eq["rho"]
    a = np.sqrt(np.maximum(np.gradient(prs * p0, rho), 1.0))
    mflux = rho * V
    i_choke = int(np.argmax(mflux))
    return prs, T, V, Reff, i_choke


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    for name, ref_T, ref_M in (("LOX/RP-1 (kerosene)", 3700.0, 21.9),
                               ("LOX/LH2", 3500.0, 13.3),
                               ("LOX/Ethanol (75%)", 3200.0, 23.9),
                               ("UDMH/N2O4 (hypergolic)", 3400.0, 23.6)):
        b, hr = reactant_state(name)
        T0 = flame_T(70.0e5, b, hr)
        eq = equilibrium(70.0e5, T0, b)
        # equilibrium gamma_s = (dlnp/dlnrho)_s via two isentrope points
        _, eq2 = T_at_sp(0.9 * 70e5, eq["s"], b, 0.97 * T0, eq["u"])
        gs = np.log(0.9) / np.log(eq2["rho"] / eq["rho"])
        xs = {sp: eq["n"][sp] / eq["N"] for sp in eq["n"]}
        top = ", ".join(f"{sp} {x:.3f}" for sp, x in
                        sorted(xs.items(), key=lambda kv: -kv[1])[:4])
        print(f"{name:26s} Tc={T0:6.0f} K  M={eq['M']*1000:5.2f}  "
              f"gamma_s={gs:.3f}  [{top}]")
        assert abs(T0 - ref_T) < 220, (name, T0)
        assert abs(eq["M"] * 1000 - ref_M) < 1.2, (name, eq["M"])
        assert 1.10 < gs < 1.26, (name, gs)
    print("equilibrium self-test OK")
