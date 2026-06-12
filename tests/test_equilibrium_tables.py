import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rocketcfd import equilibrium as eqm

t0 = time.perf_counter()
tab = eqm.build_tables("LOX/RP-1 (kerosene)")
print(f"tables built in {time.perf_counter()-t0:.1f} s")
for k, v in tab.items():
    print(f"  {k:6s} {v.shape}  finite: {np.isfinite(v).all()}  "
          f"range [{v.min():.4g}, {v.max():.4g}]")

b, _ = eqm.reactant_state("LOX/RP-1 (kerosene)")
lr_ax, lt_ax = tab["lr_ax"], tab["lt_ax"]


def bilerp(A, lr, lt):
    i = int(np.clip(np.searchsorted(lr_ax, lr) - 1, 0, len(lr_ax) - 2))
    j = int(np.clip(np.searchsorted(lt_ax, lt) - 1, 0, len(lt_ax) - 2))
    fx = np.clip((lr - lr_ax[i]) / (lr_ax[i + 1] - lr_ax[i]), 0, 1)
    fy = np.clip((lt - lt_ax[j]) / (lt_ax[j + 1] - lt_ax[j]), 0, 1)
    return (A[i, j] * (1 - fx) * (1 - fy) + A[i + 1, j] * fx * (1 - fy)
            + A[i, j + 1] * (1 - fx) * fy + A[i + 1, j + 1] * fx * fy)


def T_from_e(rho, e, Tg=2000.0):
    lr = np.log10(rho)
    T = Tg
    for _ in range(6):
        f = bilerp(tab["E"], lr, np.log10(T)) - e
        cv = bilerp(tab["CV"], lr, np.log10(T))
        T = np.clip(T - f / cv, 245.0, 4350.0)
    return T


print("round-trip (p,T) -> equilibrium(rho,e) -> table -> (p,T,a):")
worst = 0.0
for p, T in ((7e6, 3676.0), (3.95e6, 3460.0), (5e5, 2600.0),
             (5e4, 1850.0), (1.01e5, 300.0), (3e4, 1500.0), (2e3, 700.0)):
    if T >= eqm.T_FREEZE:
        eq = eqm.equilibrium(p, T, b)
    else:    # below the freeze temperature the gas model is frozen at 900 K
        hot = eqm.equilibrium(p, eqm.T_FREEZE, b)
        fr = eqm._frozen_eval(hot["n"], p, T)
        fr["e"] = fr["h"] - p / fr["rho"]
        eq = fr
    rho, e = eq["rho"], eq["e"]
    Tt = T_from_e(rho, e, 0.8 * T)
    lr, lt = np.log10(rho), np.log10(Tt)
    pt = rho * bilerp(tab["RE"], lr, lt) * Tt
    at = bilerp(tab["A"], lr, lt)
    err = abs(pt / p - 1)
    worst = max(worst, err)
    print(f"  p={p:9.3g} T={T:6.0f} -> p={pt:9.3g} ({(pt/p-1)*100:+5.2f}%)  "
          f"T={Tt:6.0f} ({Tt-T:+4.0f} K)  a={at:6.0f} m/s")
assert worst < 0.02, f"worst p error {worst*100:.1f}%"
print("table round-trip OK (<2%)")
