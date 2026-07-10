"""CUDA kernels for the GPU solver (compiled at runtime via CuPy RawModule).

Physics/config constants are baked in at compile time, so changing the
configuration requires re-creating the solver (fast, ~1 s).

Array layout: all fields are float32 blocks of shape (NFIELDS, SY, SX) where
SX = nx + 4, SY = ny + 4 (2-cell halo). Cell (i, j) lives at j*SX + i.

  cons  U[6]:  rho, rho*u, rho*v, E, rho*k, rho*omega
  prim  P[10]: rho, u, v, p, T, k, omega, mu_lam, mu_t, F1
  grad  G[10]: ux, uy, vx, vy, Tx, Ty, kx, ky, wx, wy
  flux  F[6]:  face fluxes (x-faces stored at right cell index, y likewise)

Cell types: 0 fluid, 1 wall, 2 pressure inlet, 3 farfield halo.
"""
from __future__ import annotations

from string import Template

import numpy as np

_CUDA_SRC = Template(r"""
#define SX        $SX
#define SY        $SY
#define NX        $NX
#define NY        $NY
#define NC        (SX*SY)
#define DX        $DX
#define GAM       $GAM
#define GM1       $GM1
#define RGAS      $RGAS
#define CPG       $CPG
#define PRL       $PRL
#define PRT       $PRT
#define MUREF     $MUREF
#define TREFS     $TREFS
#define SSUTH     $SSUTH
#define P0IN      $P0IN
#define T0IN      $T0IN
#define PCHOKE    $PCHOKE
#define THERMO    $THERMO
#define CPR0      $CPR0
#define CPR1      $CPR1
#define CPR2      $CPR2
#define CPR3      $CPR3
#define GAMIN     $GAMIN
#define GM1IN     $GM1IN
#define KINFAC    $KINFAC
#define MUTRIN    $MUTRIN
#define MUTMAX    $MUTMAX
#define OUTRELAX  $OUTRELAX
#define OUTRELAXONE $OUTRELAXONE
#define PFAR      $PFAR
#define TFAR      $TFAR
#define RGFAR     $RGFAR
#define UFAR      $UFAR
#define VFAR      $VFAR
#define KFAR      $KFAR
#define WFAR      $WFAR
#define CFL       $CFL
#define SCHEME    $SCHEME
#define ORDER2    $ORDER2
#define LIM       $LIM
#define WENO      $WENO
#define VISC      $VISC
#define TURB      $TURB
#define NOSLIP    $NOSLIP
#define WALLTH    $WALLTH
#define WALL_TW   $WALL_TW
#define RADWALL   $RADWALL
#define WEMISS    $WEMISS
#define STRETCH   $STRETCH
#define CARBFIX    $CARBFIX
#define COMPCORR   $COMPCORR
#define AXI       $AXI
#define AXISJ     $AXISJ
#define AXSYM_TOP $AXSYM_TOP
#define AXSYM_BOT $AXSYM_BOT

// signed radial coordinate of cell row j / of face j (face between j-1 and j);
// AXISJ is half-integer so cell centers never sit exactly on the axis
#define RCELL(j)  (((float)(j) - AXISJ) * DX)
#define RFACE(j)  (((float)(j) - 0.5f - AXISJ) * DX)

// sanitization bounds (SI)
#define RHOMIN 1.0e-6f
#define RHOMAX 1.0e4f
#define PMIN   1.0f
#define PMAX   1.0e9f
#define VMAX   2.0e4f
#define KMINV  1.0e-10f
#define KMAXV  1.0e8f
#define WMINV  1.0e-3f
#define WMAXV  1.0e12f

// SST constants
#define BSTAR 0.09f
#define SIGK1 0.85f
#define SIGK2 1.0f
#define SIGW1 0.5f
#define SIGW2 0.856f
#define BET1  0.075f
#define BET2  0.0828f
#define GAMC1 0.5532f
#define GAMC2 0.4404f
#define A1SST 0.31f

#define IDX(i,j) ((j)*SX + (i))
#define PF(f,c)  P[(f)*NC + (c)]
#define GF(f,c)  G[(f)*NC + (c)]
#define UF(f,c)  U[(f)*NC + (c)]

typedef struct { float rho,u,v,p,T,k,w,mul,mut,F1; } St;

#if STRETCH
// per-column x-cell-width multiplier (physical dx of column i = DX*SXW[i]).
// 1.0 in the uniform nozzle region, geometrically growing downstream in the
// wall-free plume so diamonds reach further. Uploaded after compile.
__device__ float SXW[SX];
#endif

__device__ __forceinline__ float suth(float T){
    return MUREF * powf(T / TREFS, 1.5f) * (TREFS + SSUTH) / (T + SSUTH);
}

#if NOSLIP
// ---- wall functions -------------------------------------------------------
// Reichardt's law of the wall: smooth u+(y+) from the viscous sublayer
// through the log layer (u+ ~ y+ below y+~5, log law above y+~30).
#define KVK 0.41f
__device__ __forceinline__ float uplus_reichardt(float yp){
    return 2.4390f * logf(1.0f + KVK * yp)
         + 7.8f * (1.0f - expf(-yp / 11.0f)
                   - (yp / 11.0f) * expf(-0.33f * yp));
}
// friction velocity from the wall-parallel speed up at distance d:
// fixed point on u_tau = up / u+(y+(u_tau)); converges in a few iterations
// because u+ varies slowly with u_tau.
__device__ __forceinline__ float utau_wf(float up, float d, float rho,
                                         float mu){
    float nu = mu / rho;
    float ut = sqrtf(nu * fmaxf(up, 1.0e-6f) / d);   // laminar first guess
    #pragma unroll
    for (int it = 0; it < 6; it++){
        float yp = fmaxf(d * ut / nu, 1.0e-6f);
        ut = up / fmaxf(uplus_reichardt(yp), 1.0e-3f);
    }
    return fmaxf(ut, 0.0f);
}
// Kader's thermal law of the wall T+(y+, Pr): wall heat flux
// q_w = rho cp u_tau (T_p - T_wall) / T+
__device__ __forceinline__ float tplus_kader(float yp, float pr){
    float pry = pr * yp;
    float G = 0.01f * pry * pry * pry * pry
            / (1.0f + 5.0f * pr * pr * pr * yp);
    float pr13 = cbrtf(pr);
    float beta = (3.85f * pr13 - 1.3f) * (3.85f * pr13 - 1.3f)
               + 2.12f * logf(pr);
    return pr * yp * expf(-G)
         + (2.12f * logf(1.0f + yp) + beta) * expf(-1.0f / fmaxf(G, 1e-10f));
}
#endif

#if THERMO == 2
// Shifting-equilibrium gas: properties from (log10 rho, log10 T) tables
// uploaded after compile (see KernelSet). p = rho*RE*T; e, a, cv tabulated.
#define TNLR   $TNLR
#define TNLT   $TNLT
#define TLR0   $TLR0
#define TDLR   $TDLR
#define TLT0   $TLT0
#define TDLT   $TDLT
#define TMINT  $TMINT
#define TMAXT  $TMAXT
#define NISEN  $NISEN
__device__ float EQ_E[TNLR * TNLT];
__device__ float EQ_RE[TNLR * TNLT];
__device__ float EQ_A[TNLR * TNLT];
__device__ float EQ_CV[TNLR * TNLT];
$ISEN_DEF

__device__ __forceinline__ float eq_lerp(const float* tab, float lr, float lt){
    float x = (lr - TLR0) / TDLR, y = (lt - TLT0) / TDLT;
    x = fminf(fmaxf(x, 0.0f), (float)(TNLR - 1) - 1.0e-3f);
    y = fminf(fmaxf(y, 0.0f), (float)(TNLT - 1) - 1.0e-3f);
    int i = (int)x, j = (int)y;
    float fx = x - (float)i, fy = y - (float)j;
    const float* t0 = tab + i * TNLT + j;
    return t0[0] * (1.0f-fx) * (1.0f-fy) + t0[TNLT] * fx * (1.0f-fy)
         + t0[1] * (1.0f-fx) * fy + t0[TNLT+1] * fx * fy;
}
// T from internal energy: Newton on the E table, warm-started
__device__ __forceinline__ float eq_T_from_e(float lr, float e, float Tg){
    float T = fminf(fmaxf(Tg, TMINT), TMAXT);
    #pragma unroll
    for (int it = 0; it < 5; it++){
        float lt = log10f(T);
        float f  = eq_lerp(EQ_E, lr, lt) - e;
        float cv = eq_lerp(EQ_CV, lr, lt);
        T -= f / cv;
        T = fminf(fmaxf(T, TMINT), TMAXT);
    }
    return T;
}
// T from pressure at given rho: fixed point on p = rho*RE(T)*T
__device__ __forceinline__ float eq_T_from_p(float lr, float rho, float p){
    float T = p / (rho * 430.0f);
    #pragma unroll
    for (int it = 0; it < 3; it++){
        float re = eq_lerp(EQ_RE, lr, log10f(fminf(fmaxf(T, TMINT), TMAXT)));
        T = p / (rho * re);
    }
    return fminf(fmaxf(T, TMINT), TMAXT);
}
#endif

#if THERMO == 1
// Thermally perfect gas (frozen mixture): cp/R cubic in z = T/1000.
__device__ __forceinline__ float cpr_T(float T){
    float z = T * 1.0e-3f;
    return CPR0 + z * (CPR1 + z * (CPR2 + z * CPR3));
}
__device__ __forceinline__ float gam_T(float T){
    float c = cpr_T(T);
    return c / (c - 1.0f);
}
// e/R = h/R - T with h(0) = 0 (integral of the cp/R cubic)
__device__ __forceinline__ float eR_T(float T){
    float z = T * 1.0e-3f;
    float hR = 1.0e3f * z * (CPR0 + z * (0.5f * CPR1
             + z * (0.33333333f * CPR2 + z * 0.25f * CPR3)));
    return hR - T;
}
// invert e(T) by Newton; Tg = initial guess (previous T of the cell)
__device__ __forceinline__ float T_from_e(float eR, float Tg){
    float T = fminf(fmaxf(Tg, 60.0f), 5900.0f);
    #pragma unroll
    for (int it = 0; it < 4; it++){
        float cv = fmaxf(cpr_T(T) - 1.0f, 0.5f);
        T -= (eR_T(T) - eR) / cv;
        T = fminf(fmaxf(T, 50.0f), 6000.0f);
    }
    return T;
}
#endif

// MUSCL slope limiter on two consecutive differences a (backward) and
// b (forward). LIM selects: 0 minmod, 1 van Albada, 2 van Leer, 3 superbee.
// All return 0 at extrema (a*b <= 0) for TVD monotonicity.
__device__ __forceinline__ float limslope(float a, float b){
#if LIM == 1
    float ab = a * b;                       // van Albada (smooth)
    if (ab <= 0.0f) return 0.0f;
    return ab * (a + b) / (a*a + b*b + 1.0e-30f);
#elif LIM == 2
    float ab = a * b;                       // van Leer (harmonic mean)
    if (ab <= 0.0f) return 0.0f;
    return 2.0f * ab / (a + b);
#elif LIM == 3
    if (a * b <= 0.0f) return 0.0f;         // superbee (most compressive)
    float s = (a > 0.0f) ? 1.0f : -1.0f;
    float fa = fabsf(a), fb = fabsf(b);
    return s * fmaxf(fminf(2.0f * fa, fb), fminf(fa, 2.0f * fb));
#else
    if (a * b <= 0.0f) return 0.0f;         // minmod (most dissipative)
    return fabsf(a) < fabsf(b) ? a : b;
#endif
}

#if WENO
// 5th-order WENO (Jiang-Shu): reconstruct the value at the RIGHT face of the
// center cell c0 from the 5-point stencil (m2, m1, c0, p1, p2). Far lower
// numerical dissipation than MUSCL, so shock-cell trains and shear layers
// survive much further downstream. Used only where the whole stencil is
// fluid; cells touching walls/edges fall back to MUSCL.
__device__ __forceinline__ float weno5(float m2, float m1, float c0,
                                       float p1, float p2){
    const float eps = 1.0e-6f;
    float q0 = ( 2.0f*m2 - 7.0f*m1 + 11.0f*c0) * 0.16666667f;
    float q1 = (-1.0f*m1 + 5.0f*c0 +  2.0f*p1) * 0.16666667f;
    float q2 = ( 2.0f*c0 + 5.0f*p1 -  1.0f*p2) * 0.16666667f;
    float b0 = 1.0833333f*(m2 - 2.0f*m1 + c0)*(m2 - 2.0f*m1 + c0)
             + 0.25f*(m2 - 4.0f*m1 + 3.0f*c0)*(m2 - 4.0f*m1 + 3.0f*c0);
    float b1 = 1.0833333f*(m1 - 2.0f*c0 + p1)*(m1 - 2.0f*c0 + p1)
             + 0.25f*(m1 - p1)*(m1 - p1);
    float b2 = 1.0833333f*(c0 - 2.0f*p1 + p2)*(c0 - 2.0f*p1 + p2)
             + 0.25f*(3.0f*c0 - 4.0f*p1 + p2)*(3.0f*c0 - 4.0f*p1 + p2);
    float w0 = 0.1f / ((eps + b0)*(eps + b0));
    float w1 = 0.6f / ((eps + b1)*(eps + b1));
    float w2 = 0.3f / ((eps + b2)*(eps + b2));
    float wi = 1.0f / (w0 + w1 + w2);
    return (w0*q0 + w1*q1 + w2*q2) * wi;
}

#endif

#if CARBFIX
// Ducros-gated shock sensor per cell: ~1 inside a strong compression shock,
// ~0 in expansions, shear layers and turbulence (so it does NOT add
// dissipation to the plume mixing layer or boundary layers). Drives the
// HLLC->HLL blend that cures the carbuncle at the Mach disk.
__device__ __forceinline__ float shock_theta(const float* P, const float* G, int c){
    float divu = GF(0,c) + GF(3,c);          // du/dx + dv/dy
    if (divu >= 0.0f) return 0.0f;           // expansion: never a shock
    float curl = GF(2,c) - GF(1,c);          // dv/dx - du/dy
    float duc = divu*divu / (divu*divu + curl*curl + 1.0e-12f);   // Ducros
    float a = sqrtf(GAM * fmaxf(PF(3,c), PMIN) / fmaxf(PF(0,c), RHOMIN));
    float s = -divu * DX / fmaxf(a, 1.0e-3f);    // compression over a cell / a
    // only strong (near-normal, ~sonic-over-a-cell) shocks — where the
    // carbuncle lives — so weak oblique plume shocks are left to HLLC
    float ramp = fminf(fmaxf(s - 1.0f, 0.0f), 1.0f);            // 0<s1  1>s2
    return duc * ramp;
}
#endif

// State of stencil cell (ic,jc) as seen from owner fluid cell (io,jo).
// Walls and inlets are replaced by ghost states; dir = face direction (0=x,1=y).
// p0eff: ramped inlet total pressure (soft start), <= P0IN.
__device__ St fetch(const float* P, const unsigned char* ct, const float* wd,
                    int ic, int jc, int io, int jo, int dir, float p0eff)
{
    St q;
    int c = IDX(ic, jc);
    int t = ct[c];
    if (t == 0 || t == 3) {
        q.rho = PF(0,c); q.u = PF(1,c); q.v = PF(2,c); q.p = PF(3,c);
        q.T  = PF(4,c);  q.k = PF(5,c); q.w = PF(6,c);
        q.mul = PF(7,c); q.mut = PF(8,c); q.F1 = PF(9,c);
        return q;
    }
    int o = IDX(io, jo);
    float ro = PF(0,o), uo = PF(1,o), vo = PF(2,o), po = PF(3,o);
    if (t == 4) {                       // ---- pressure outlet (red) ----
        float sxd = (float)(io - ic), syd = (float)(jo - jc);  // into-domain
        float into = uo * sxd + vo * syd;
        if (into > 0.0f) {
            // backflow: ambient gas enters here -> anchor the thermodynamics
            // (T stays pinned to ambient or the plume heats unboundedly).
            // The pressure is relaxed like the outflow branch: OUTRELAX=1 is
            // the classic hard pin; lower values let the pressure waves of
            // vortices crossing the outlet leave instead of reflecting.
#if OUTRELAXONE
            q.p = PFAR;                       // hard pin (bit-exact default)
#else
            q.p = po + OUTRELAX * (PFAR - po);
#endif
            q.T = TFAR; q.rho = q.p / (RGFAR * TFAR);
            q.u = uo; q.v = vo;
            q.k = KFAR; q.w = WFAR;
            q.mul = suth(TFAR); q.mut = 0.0f; q.F1 = 1.0f;
            return q;
        }
        q.u = uo; q.v = vo;
#if THERMO == 1
        float go = gam_T(po / (ro * RGAS));
#elif THERMO == 2
        float lro = log10f(fmaxf(ro, RHOMIN));
        float ao  = eq_lerp(EQ_A, lro, log10f(PF(4,o)));
        float go  = fminf(fmaxf(ao * ao * ro / po, 1.05f), 1.67f);
#else
        const float go = GAM;
#endif
        if (uo*uo + vo*vo >= go * po / ro) {    // supersonic: extrapolate
            q.rho = ro; q.p = po;
        } else {
            // subsonic: relax the ghost pressure toward ambient. OUTRELAX=1
            // is the classic hard pin (fully anchors back-pressure but
            // reflects acoustic/vortical disturbances back upstream);
            // 0.2-0.5 lets most waves leave while still holding the mean.
#if OUTRELAXONE
            q.p = PFAR;                       // hard pin (bit-exact default)
#else
            q.p = po + OUTRELAX * (PFAR - po);
#endif
            q.rho = ro * powf(q.p / po, 1.0f / go);
        }
#if THERMO == 2
        q.T = eq_T_from_p(log10f(fmaxf(q.rho, RHOMIN)), q.rho, q.p);
#else
        q.T = q.p / (q.rho * RGAS);
#endif
        q.k = PF(5,o); q.w = PF(6,o);
        q.mul = PF(7,o); q.mut = PF(8,o); q.F1 = PF(9,o);
        return q;
    }
    if (t == 1) {                       // ---- wall ghost (mirror) ----
        q.rho = ro; q.p = po;
#if THERMO == 2
        q.T = PF(4,o);                  // mirrored state: owner's T
#else
        q.T = po / (ro * RGAS);
#endif
#if NOSLIP
        q.u = -uo; q.v = -vo;
#else
        if (dir == 0) { q.u = -uo; q.v =  vo; }
        else          { q.u =  uo; q.v = -vo; }
#endif
        float d  = fmaxf(wd[o], 0.25f * DX);
        float nu = PF(7,o) / ro;
#if NOSLIP
        // Menter automatic wall treatment: blend the viscous-sublayer and
        // log-layer omega so the BC is y+-insensitive; k is zero-gradient
        {
            float utan = (dir == 0) ? fabsf(vo) : fabsf(uo);
            float utw = utau_wf(fmaxf(utan, 1.0e-6f), d, ro, PF(7,o));
            float wv = 6.0f * nu / (BET1 * d * d);
            float wl = utw / (0.3f * KVK * d);
            q.w = fminf(sqrtf(wv * wv + wl * wl), WMAXV);
        }
        q.k = PF(5,o);
#else
        q.k = 0.0f;
        q.w   = fminf(60.0f * nu / (BET1 * d * d), WMAXV);   // Menter wall omega
#endif
        q.mul = PF(7,o);
        q.mut = -PF(8,o);               // face average of mu_t -> 0 at wall
        q.F1  = 1.0f;
        return q;
    }
    // ---- pressure inlet ghost: total conditions P0,T0, flow normal to face ----
    float pi  = fminf(po, p0eff);
    float pr  = fmaxf(pi / p0eff, PCHOKE);       // cap at sonic (choked)
#if THERMO == 2
    // chamber isentrope (shifting equilibrium), tabulated on a uniform
    // pressure-ratio axis at build time
    float xt = (pr - ISEN_PR0) / ISEN_DPR;
    xt = fminf(fmaxf(xt, 0.0f), (float)(NISEN - 1) - 1.0e-3f);
    int ii = (int)xt; float fxt = xt - (float)ii;
    float Tt  = ISEN_T[ii]  * (1.0f - fxt) + ISEN_T[ii + 1]  * fxt;
    float V   = ISEN_V[ii]  * (1.0f - fxt) + ISEN_V[ii + 1]  * fxt;
    float Rei = ISEN_RE[ii] * (1.0f - fxt) + ISEN_RE[ii + 1] * fxt;
#else
    // isentropic relations with the chamber gamma (GAMIN = gamma(T0);
    // equals GAM for the calorically perfect gas) — exact where T ~ T0
    float prx = powf(pr, GM1IN / GAMIN);         // = T/T0
    float Tt  = T0IN * prx;
#if THERMO == 1
    // velocity from exact enthalpy conservation h(T0) = h(Tt) + V^2/2
    // (the constant-cp form over-injects total enthalpy when cp falls
    // with T below T0); the T-p relation above keeps the gamma(T0) form
    float dhR = (eR_T(T0IN) + T0IN) - (eR_T(Tt) + Tt);
    float V   = sqrtf(2.0f * RGAS * fmaxf(dhR, 0.0f));
#else
    float M2  = 2.0f / GM1IN * (1.0f / prx - 1.0f);
    float a   = sqrtf(GAMIN * RGAS * Tt);
    float V   = sqrtf(fmaxf(M2, 0.0f)) * a;
#endif
    const float Rei = RGAS;
#endif
    float sx  = (float)(io - ic), sy = (float)(jo - jc);
    q.rho = pi / (Rei * Tt);
    q.p = pi; q.T = Tt;
    q.u = V * sx; q.v = V * sy;
    q.k = KINFAC * V * V + 1.0e-6f;
    q.mul = suth(Tt);
    q.mut = MUTRIN * q.mul;
    q.w = fmaxf(q.rho * q.k / fmaxf(q.mut, 1.0e-12f), WMINV);
    q.F1 = 1.0f;
    return q;
}

// Numerical flux, selected at compile time:
//   SCHEME 0 = HLL, 1 = HLLC, 2 = Roe (Harten entropy fix), 3 = AUSM+.
// Returns (mass, normal-mom, tangential-mom, energy, k, w).
__device__ void riemann(const St L, const St R, int dir, float shock, float* F)
{
    float unL = (dir == 0) ? L.u : L.v;
    float utL = (dir == 0) ? L.v : L.u;
    float unR = (dir == 0) ? R.u : R.v;
    float utR = (dir == 0) ? R.v : R.u;
#if THERMO == 1
    // T derived from the (MUSCL-reconstructed) p and rho — the struct's T is
    // cell-centered and would be thermodynamically inconsistent here
    float TL = L.p / (L.rho * RGAS), TR = R.p / (R.rho * RGAS);
    float gL = gam_T(TL), gR = gam_T(TR);
    float aL = sqrtf(gL * L.p / L.rho), aR = sqrtf(gR * R.p / R.rho);
    float EL = L.rho * RGAS * eR_T(TL) + 0.5f * L.rho * (L.u*L.u + L.v*L.v);
    float ER = R.rho * RGAS * eR_T(TR) + 0.5f * R.rho * (R.u*R.u + R.v*R.v);
#elif THERMO == 2
    // equilibrium tables, with T consistent with the reconstructed (rho, p)
    float lrL = log10f(fmaxf(L.rho, RHOMIN)), lrR = log10f(fmaxf(R.rho, RHOMIN));
    float TL = eq_T_from_p(lrL, L.rho, L.p), TR = eq_T_from_p(lrR, R.rho, R.p);
    float ltL = log10f(TL), ltR = log10f(TR);
    float aL = eq_lerp(EQ_A, lrL, ltL), aR = eq_lerp(EQ_A, lrR, ltR);
    float EL = L.rho * eq_lerp(EQ_E, lrL, ltL)
             + 0.5f * L.rho * (L.u*L.u + L.v*L.v);
    float ER = R.rho * eq_lerp(EQ_E, lrR, ltR)
             + 0.5f * R.rho * (R.u*R.u + R.v*R.v);
#else
    float aL = sqrtf(GAM * L.p / L.rho), aR = sqrtf(GAM * R.p / R.rho);
    float EL = L.p / GM1 + 0.5f * L.rho * (L.u*L.u + L.v*L.v);
    float ER = R.p / GM1 + 0.5f * R.rho * (R.u*R.u + R.v*R.v);
#endif
    float HL = (EL + L.p) / L.rho, HR = (ER + R.p) / R.rho;

#if SCHEME == 3
    // ------------------------------------------------ AUSM+ (Liou 1996)
    float ah = 0.5f * (aL + aR);
    float ML = unL / ah, MR = unR / ah;
    float Mp, Pp, Mm, Pm;
    if (fabsf(ML) >= 1.0f) {
        Mp = 0.5f * (ML + fabsf(ML));
        Pp = (ML >= 0.0f) ? 1.0f : 0.0f;
    } else {
        float s = ML * ML - 1.0f;
        Mp = 0.25f * (ML + 1.0f) * (ML + 1.0f) + 0.125f * s * s;
        Pp = 0.25f * (ML + 1.0f) * (ML + 1.0f) * (2.0f - ML)
           + 0.1875f * ML * s * s;
    }
    if (fabsf(MR) >= 1.0f) {
        Mm = 0.5f * (MR - fabsf(MR));
        Pm = (MR >= 0.0f) ? 0.0f : 1.0f;
    } else {
        float s = MR * MR - 1.0f;
        Mm = -0.25f * (MR - 1.0f) * (MR - 1.0f) - 0.125f * s * s;
        Pm = 0.25f * (MR - 1.0f) * (MR - 1.0f) * (2.0f + MR)
           - 0.1875f * MR * s * s;
    }
    float mh = Mp + Mm;
    float fm = ah * (mh > 0.0f ? mh * L.rho : mh * R.rho);
    float ph = Pp * L.p + Pm * R.p;
    if (fm >= 0.0f) {
        F[0] = fm; F[1] = fm * unL + ph; F[2] = fm * utL;
        F[3] = fm * HL; F[4] = fm * L.k; F[5] = fm * L.w;
    } else {
        F[0] = fm; F[1] = fm * unR + ph; F[2] = fm * utR;
        F[3] = fm * HR; F[4] = fm * R.k; F[5] = fm * R.w;
    }
#else
    // Roe averages (wave speeds for HLL/HLLC, full FDS for Roe)
    float sl = sqrtf(L.rho), sr = sqrtf(R.rho);
    float wa = sl / (sl + sr);
    float unZ = wa * unL + (1.0f - wa) * unR;
    float utZ = wa * utL + (1.0f - wa) * utR;
    float HZ  = wa * HL  + (1.0f - wa) * HR;
#if THERMO == 1
    // Roe-average sound speed with the interface-averaged gamma (frozen-cp
    // approximation; exact CP algebra does not carry over to variable cp)
    float gZ1 = wa * gL + (1.0f - wa) * gR - 1.0f;
    float aZ  = sqrtf(fmaxf(gZ1 * (HZ - 0.5f * (unZ*unZ + utZ*utZ)), 1.0e-8f));
#elif THERMO == 2
    // table EOS: Roe-weighted average of the local equilibrium sound speeds
    float aZ  = fmaxf(wa * aL + (1.0f - wa) * aR, 1.0e-4f);
#else
    float aZ  = sqrtf(fmaxf(GM1 * (HZ - 0.5f * (unZ*unZ + utZ*utZ)), 1.0e-8f));
#endif

    float FL[6] = {L.rho*unL, L.rho*unL*unL + L.p, L.rho*unL*utL,
                   unL*(EL + L.p), L.rho*unL*L.k, L.rho*unL*L.w};
    float FR[6] = {R.rho*unR, R.rho*unR*unR + R.p, R.rho*unR*utR,
                   unR*(ER + R.p), R.rho*unR*R.k, R.rho*unR*R.w};

#if SCHEME == 2
    // -------------------------- Roe flux-difference splitting
    float rt  = sl * sr;                        // Roe density
    float dp  = R.p - L.p, dun = unR - unL, dut = utR - utL;
    float a2i = 1.0f / (aZ * aZ);
    float al1 = 0.5f * (dp - rt * aZ * dun) * a2i;   // un - a acoustic
    float al5 = 0.5f * (dp + rt * aZ * dun) * a2i;   // un + a acoustic
    float al2 = (R.rho - L.rho) - dp * a2i;          // entropy wave
    float al3 = rt * dut;                            // shear wave
    float l1 = fabsf(unZ - aZ), l2 = fabsf(unZ), l5 = fabsf(unZ + aZ);
    float eps = 0.1f * aZ;                           // Harten entropy fix
    if (l1 < eps) l1 = 0.5f * (l1 * l1 / eps + eps);
    if (l5 < eps) l5 = 0.5f * (l5 * l5 / eps + eps);
    float q2h = 0.5f * (unZ*unZ + utZ*utZ);
    float D0 = l1*al1 + l2*al2 + l5*al5;
    float D1 = l1*al1*(unZ - aZ) + l2*al2*unZ + l5*al5*(unZ + aZ);
    float D2 = l1*al1*utZ + l2*(al2*utZ + al3) + l5*al5*utZ;
    float D3 = l1*al1*(HZ - unZ*aZ) + l2*(al2*q2h + al3*utZ)
             + l5*al5*(HZ + unZ*aZ);
    F[0] = 0.5f * (FL[0] + FR[0] - D0);
    F[1] = 0.5f * (FL[1] + FR[1] - D1);
    F[2] = 0.5f * (FL[2] + FR[2] - D2);
    F[3] = 0.5f * (FL[3] + FR[3] - D3);
    F[4] = F[0] * (F[0] >= 0.0f ? L.k : R.k);        // scalars: mass-flux upwind
    F[5] = F[0] * (F[0] >= 0.0f ? L.w : R.w);
#else
    // -------------------------- HLL / HLLC with Einfeldt wave speeds
    float SL = fminf(unL - aL, unZ - aZ);
    float SR = fmaxf(unR + aR, unZ + aZ);
    float UL[6] = {L.rho, L.rho*unL, L.rho*utL, EL, L.rho*L.k, L.rho*L.w};
    float UR[6] = {R.rho, R.rho*unR, R.rho*utR, ER, R.rho*R.k, R.rho*R.w};
    if (SL >= 0.0f)      { for (int m = 0; m < 6; m++) F[m] = FL[m]; }
    else if (SR <= 0.0f) { for (int m = 0; m < 6; m++) F[m] = FR[m]; }
    else {
#if SCHEME == 1
        float dL = L.rho * (SL - unL), dR = R.rho * (SR - unR);
        float S  = (R.p - L.p + unL * dL - unR * dR) / (dL - dR);
        if (S >= 0.0f) {
            float fac = dL / (SL - S);
            float Us[6] = {fac, fac*S, fac*utL,
                           fac*(EL/L.rho + (S - unL)*(S + L.p/dL)),
                           fac*L.k, fac*L.w};
            for (int m = 0; m < 6; m++) F[m] = FL[m] + SL * (Us[m] - UL[m]);
        } else {
            float fac = dR / (SR - S);
            float Us[6] = {fac, fac*S, fac*utR,
                           fac*(ER/R.rho + (S - unR)*(S + R.p/dR)),
                           fac*R.k, fac*R.w};
            for (int m = 0; m < 6; m++) F[m] = FR[m] + SR * (Us[m] - UR[m]);
        }
        // carbuncle cure: blend the contact-resolving HLLC flux toward the
        // dissipative (carbuncle-free) HLL flux only at strong shocks
        // (shock>0 from the Ducros sensor), curing the Mach-disk instability
        // while keeping HLLC's contact/shear resolution everywhere else.
        if (shock > 0.0f) {
            float inv = 1.0f / (SR - SL);
            for (int m = 0; m < 6; m++) {
                float fhll = (SR*FL[m] - SL*FR[m] + SL*SR*(UR[m]-UL[m])) * inv;
                F[m] = (1.0f - shock) * F[m] + shock * fhll;
            }
        }
#else
        float inv = 1.0f / (SR - SL);
        for (int m = 0; m < 6; m++)
            F[m] = (SR * FL[m] - SL * FR[m] + SL * SR * (UR[m] - UL[m])) * inv;
#endif
    }
#endif
#endif
}

extern "C" {

// ---------------------------------------------------------------- cons2prim
__global__ void cons2prim(const float* U, float* P, const unsigned char* ct,
                          float* dtl, const float* VF, int compute_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= SX || j >= SY) return;
    int c = IDX(i, j);
    int t = ct[c];
    if (t == 3) return;                       // halos handled by halo_fill
    if (t != 0) {                             // wall / inlet: benign placeholder
        PF(0,c) = PFAR / (RGFAR * TFAR); PF(1,c) = 0.0f; PF(2,c) = 0.0f;
        PF(3,c) = PFAR; PF(4,c) = TFAR; PF(5,c) = KFAR; PF(6,c) = WFAR;
        PF(7,c) = suth(TFAR); PF(8,c) = 0.0f; PF(9,c) = 1.0f;
        if (compute_dt) dtl[c] = 1.0e30f;
        return;
    }
    float rho = fminf(fmaxf(UF(0,c), RHOMIN), RHOMAX);
    float inv = 1.0f / rho;
    float u = UF(1,c) * inv, v = UF(2,c) * inv;
#if THERMO == 1
    // T from internal energy (Newton, warm-started from the previous T)
    float eR = (UF(3,c) * inv - 0.5f * (u*u + v*v)) / RGAS;
    float T = T_from_e(eR, PF(4,c));
    float p = rho * RGAS * T;
    p = fminf(fmaxf(p, PMIN), PMAX);
    T = p / (rho * RGAS);
#elif THERMO == 2
    float lrc = log10f(rho);
    float ec = UF(3,c) * inv - 0.5f * (u*u + v*v);
    float T = eq_T_from_e(lrc, ec, PF(4,c));
    float p = rho * eq_lerp(EQ_RE, lrc, log10f(T)) * T;
    p = fminf(fmaxf(p, PMIN), PMAX);
#else
    float p = GM1 * (UF(3,c) - 0.5f * rho * (u*u + v*v));
    p = fminf(fmaxf(p, PMIN), PMAX);
    float T = p / (rho * RGAS);
#endif
    float k = fminf(fmaxf(UF(4,c) * inv, KMINV), KMAXV);
    float w = fminf(fmaxf(UF(5,c) * inv, WMINV), WMAXV);
    float mul = suth(T);
    PF(0,c) = rho; PF(1,c) = u; PF(2,c) = v; PF(3,c) = p; PF(4,c) = T;
    PF(5,c) = k; PF(6,c) = w; PF(7,c) = mul;
    if (compute_dt) {
#if THERMO == 1
        float gdt = gam_T(T);
        float a = sqrtf(gdt * RGAS * T);
#elif THERMO == 2
        const float gdt = GAM;
        float a = eq_lerp(EQ_A, lrc, log10f(T));
#else
        const float gdt = GAM;
        float a = sqrtf(gdt * RGAS * T);
#endif
        float nue = fmaxf(1.3333f, gdt / PRL) * (mul + fmaxf(PF(8,c), 0.0f)) * inv;
        float fy = 1.0f;
#if AXI
        // face-area/volume ratio grows near the axis -> stiffer y direction
        float rca = fabsf(RCELL(j));
        fy = fminf((rca + DX) / rca, 3.0f);
#endif
#if STRETCH
        float dxc = DX * SXW[i];             // stretched x cell width
#else
        float dxc = DX;
#endif
        float lam = (fabsf(u) + a) / dxc + fy * (fabsf(v) + a) / DX + 4.0f * nue / (DX * DX);
        // cut cells: smaller fluid volume -> proportionally smaller stable dt
        dtl[c] = CFL * fmaxf(VF[c], 0.25f) / lam;
    }
}

// ---------------------------------------------------------------- halo_fill
// Farfield / pressure-outlet characteristic boundary on the 2-cell halo ring.
__global__ void halo_fill(float* P, const unsigned char* ct)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= SX || j >= SY) return;
    int c = IDX(i, j);
    if (ct[c] != 3) return;

#if AXSYM_TOP
    // axis along the top image edge: top halo is a symmetry plane (v -> -v)
    if (j < 2 && i >= 2 && i < NX + 2) {
        int cm = IDX(i, 3 - j);
        PF(0,c) = PF(0,cm); PF(1,c) = PF(1,cm); PF(2,c) = -PF(2,cm);
        PF(3,c) = PF(3,cm); PF(4,c) = PF(4,cm); PF(5,c) = PF(5,cm);
        PF(6,c) = PF(6,cm); PF(7,c) = PF(7,cm); PF(8,c) = PF(8,cm);
        PF(9,c) = PF(9,cm);
        return;
    }
#endif
#if AXSYM_BOT
    // axis along the bottom image edge
    if (j >= NY + 2 && i >= 2 && i < NX + 2) {
        int cm = IDX(i, 2 * (NY + 2) - 1 - j);
        PF(0,c) = PF(0,cm); PF(1,c) = PF(1,cm); PF(2,c) = -PF(2,cm);
        PF(3,c) = PF(3,cm); PF(4,c) = PF(4,cm); PF(5,c) = PF(5,cm);
        PF(6,c) = PF(6,cm); PF(7,c) = PF(7,cm); PF(8,c) = PF(8,cm);
        PF(9,c) = PF(9,cm);
        return;
    }
#endif

    int ii = min(max(i, 2), NX + 1);
    int jj = min(max(j, 2), NY + 1);
    float nxo = (i < 2) ? -1.0f : ((i > NX + 1) ? 1.0f : 0.0f);
    float nyo = (j < 2) ? -1.0f : ((j > NY + 1) ? 1.0f : 0.0f);
    float nl = sqrtf(nxo*nxo + nyo*nyo);
    nxo /= nl; nyo /= nl;
    int cI = IDX(ii, jj);

    float rho, u, v, p, k, w, mut;
    if (ct[cI] != 0) {                       // edge cell is wall/inlet: quiescent
        p = PFAR; rho = PFAR / (RGFAR * TFAR);
        u = UFAR; v = VFAR; k = KFAR; w = WFAR; mut = 0.0f;
    } else {
        float ri = PF(0,cI), ui = PF(1,cI), vi = PF(2,cI), pi = PF(3,cI);
        float ki = PF(5,cI), wi = PF(6,cI);
        float un = ui * nxo + vi * nyo;
#if THERMO == 1
        float gI = gam_T(PF(4,cI));
#elif THERMO == 2
        float aI = eq_lerp(EQ_A, log10f(fmaxf(ri, RHOMIN)), log10f(PF(4,cI)));
        float gI = fminf(fmaxf(aI * aI * ri / pi, 1.05f), 1.67f);
#else
        const float gI = GAM;
#endif
        float a  = sqrtf(gI * pi / ri);
        mut = PF(8,cI);
        if (un >= 0.0f) {                    // outflow
            if (un >= a) { rho = ri; u = ui; v = vi; p = pi; k = ki; w = wi; }
            else {                            // subsonic: impose farfield pressure
                p = PFAR;
                rho = ri * powf(PFAR / pi, 1.0f / gI);
                u = ui; v = vi; k = ki; w = wi;
            }
        } else {                              // inflow
            if (-un >= a) {                   // supersonic: full farfield state
                p = PFAR; rho = PFAR / (RGFAR * TFAR);
                u = UFAR; v = VFAR; k = KFAR; w = WFAR; mut = 0.0f;
            } else {                          // subsonic: extrapolate p, fix rest
                p = pi; rho = p / (RGFAR * TFAR);
                u = UFAR; v = VFAR; k = KFAR; w = WFAR; mut = 0.0f;
            }
        }
    }
#if THERMO == 2
    float T = eq_T_from_p(log10f(fmaxf(rho, RHOMIN)), rho, p);
#else
    float T = p / (rho * RGAS);
#endif
    PF(0,c) = rho; PF(1,c) = u; PF(2,c) = v; PF(3,c) = p; PF(4,c) = T;
    PF(5,c) = k; PF(6,c) = w; PF(7,c) = suth(T); PF(8,c) = mut; PF(9,c) = 1.0f;
}

// ---------------------------------------------------------------- gradients
__global__ void gradients(const float* P, float* G, const unsigned char* ct,
                          const float* wd, float p0eff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || j < 2 || i >= NX + 2 || j >= NY + 2) return;
    int c = IDX(i, j);
    if (ct[c] != 0) {
        for (int f = 0; f < 10; f++) GF(f,c) = 0.0f;
        return;
    }
    St qW = fetch(P, ct, wd, i-1, j, i, j, 0, p0eff);
    St qE = fetch(P, ct, wd, i+1, j, i, j, 0, p0eff);
    St qS = fetch(P, ct, wd, i, j-1, i, j, 1, p0eff);
    St qN = fetch(P, ct, wd, i, j+1, i, j, 1, p0eff);
    float hy = 0.5f / DX;
#if STRETCH
    // non-uniform x central difference: 1 / (x_{i+1} - x_{i-1})
    float hx = 1.0f / (DX * (0.5f*SXW[i-1] + SXW[i] + 0.5f*SXW[i+1]));
#else
    float hx = 0.5f / DX;
#endif
    GF(0,c) = (qE.u - qW.u) * hx;  GF(1,c) = (qN.u - qS.u) * hy;
    GF(2,c) = (qE.v - qW.v) * hx;  GF(3,c) = (qN.v - qS.v) * hy;
    GF(4,c) = (qE.T - qW.T) * hx;  GF(5,c) = (qN.T - qS.T) * hy;
    GF(6,c) = (qE.k - qW.k) * hx;  GF(7,c) = (qN.k - qS.k) * hy;
    GF(8,c) = (qE.w - qW.w) * hx;  GF(9,c) = (qN.w - qS.w) * hy;
}

// ---------------------------------------------------------------- turb_visc
// SST eddy viscosity + blending function F1; also stores 2*Sij*Sij in s2.
__global__ void turb_visc(float* P, const float* G, const float* wd,
                          const unsigned char* ct, float* s2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || j < 2 || i >= NX + 2 || j >= NY + 2) return;
    int c = IDX(i, j);
    if (ct[c] != 0) { PF(8,c) = 0.0f; PF(9,c) = 1.0f; s2[c] = 0.0f; return; }
    float ux = GF(0,c), uy = GF(1,c), vx = GF(2,c), vy = GF(3,c);
    float S2 = 2.0f*(ux*ux + vy*vy) + (uy + vx)*(uy + vx);
#if AXI
    float vor = PF(2,c) / RCELL(j);          // hoop strain v/r
    S2 += 2.0f * vor * vor;
#endif
    s2[c] = S2;
#if TURB
    float rho = PF(0,c), k = PF(5,c), w = PF(6,c), mul = PF(7,c);
    float d  = fmaxf(wd[c], 0.25f * DX);
    float nu = mul / rho;
    float sqk = sqrtf(k);
    float t1 = sqk / (BSTAR * w * d);
    float t2 = 500.0f * nu / (d * d * w);
    float arg2 = fmaxf(2.0f * t1, t2);
    float F2 = tanhf(arg2 * arg2);
    float Smag = sqrtf(S2);
    float mut = rho * A1SST * k / fmaxf(A1SST * w, Smag * F2);
    // MUTMAX: user-configurable eddy-viscosity cap (default 1e5 = the classic
    // sanity clamp). RANS SST over-mixes supersonic jets, which both shortens
    // the plume and (via the viscous dt limit) makes it develop slowly —
    // capping around 500-2000x restores long shock-diamond plumes.
    mut = fminf(mut, MUTMAX * mul);
    float dkdw = GF(6,c)*GF(8,c) + GF(7,c)*GF(9,c);
    float CDkw = fmaxf(2.0f * rho * SIGW2 / w * dkdw, 1.0e-20f);
    float arg1 = fminf(fmaxf(t1, t2), 4.0f * rho * SIGW2 * k / (CDkw * d * d));
    float F1 = tanhf(arg1*arg1*arg1*arg1);
    PF(8,c) = mut; PF(9,c) = F1;
#else
    PF(8,c) = 0.0f; PF(9,c) = 1.0f;
#endif
}

// ---------------------------------------------------------------- fluxes
// MUSCL + HLL(C) inviscid flux and central viscous flux through faces.
// dir==0: face between (i-1,j) and (i,j); dir==1: between (i,j-1) and (i,j).
// Components written to F[6] at the right/upper cell index.
__global__ void fluxes(const float* P, const float* G, const unsigned char* ct,
                       const float* wd, float* F, int dir, float p0eff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int oi = (dir == 0) ? 1 : 0, oj = (dir == 0) ? 0 : 1;
    if (i < 2 || j < 2) return;
    if (dir == 0) { if (i > NX + 2 || j >= NY + 2) return; }
    else          { if (i >= NX + 2 || j > NY + 2) return; }

    int iL = i - oi, jL = j - oj;
    int cL = IDX(iL, jL), cR = IDX(i, j);
    int tL = ct[cL], tR = ct[cR];
    bool fL = (tL == 0), fR = (tR == 0);
    if (!fL && !fR) {
        for (int m = 0; m < 6; m++) F[m*NC + cR] = 0.0f;
        return;
    }

    St qL1 = fetch(P, ct, wd, iL, jL, i, j, dir, p0eff);   // L seen from R
    St qR1 = fetch(P, ct, wd, i, j, iL, jL, dir, p0eff);   // R seen from L
    St qL = qL1, qR = qR1;
    bool hi_done = false;

#if WENO
    // 5th-order WENO where the full 6-cell stencil along `dir` is interior
    // fluid and in-bounds; otherwise fall through to MUSCL below.
    {
        int xm = iL - 2*oi, ym = jL - 2*oj;        // furthest upwind cell
        int xp = i + 2*oi,  yp = j + 2*oj;         // furthest downwind cell
        if (xm >= 0 && ym >= 0 && xp < SX && yp < SY) {
            int cm2 = IDX(xm, ym), cm1 = IDX(iL - oi, jL - oj);
            int cp2 = IDX(i + oi, j + oj), cp3 = IDX(xp, yp);
            if (ct[cm2]==0 && ct[cm1]==0 && ct[cL]==0 && ct[cR]==0
                && ct[cp2]==0 && ct[cp3]==0) {
                #define WL(f) weno5(PF(f,cm2),PF(f,cm1),PF(f,cL),PF(f,cR),PF(f,cp2))
                #define WR(f) weno5(PF(f,cp3),PF(f,cp2),PF(f,cR),PF(f,cL),PF(f,cm1))
                qL.rho = WL(0); qL.u = WL(1); qL.v = WL(2);
                qL.p = WL(3);   qL.k = WL(5); qL.w = WL(6);
                qR.rho = WR(0); qR.u = WR(1); qR.v = WR(2);
                qR.p = WR(3);   qR.k = WR(5); qR.w = WR(6);
                #undef WL
                #undef WR
                if (qL.rho < RHOMIN || qL.p < PMIN || qL.k < 0.0f || qL.w < WMINV)
                    qL = qL1;
                if (qR.rho < RHOMIN || qR.p < PMIN || qR.k < 0.0f || qR.w < WMINV)
                    qR = qR1;
                hi_done = true;
            }
        }
    }
#endif

#if ORDER2
    if (!hi_done && (tL == 0 || tL == 3)) {
        St qm = fetch(P, ct, wd, iL - oi, jL - oj, iL, jL, dir, p0eff);
        qL.rho = qL1.rho + 0.5f * limslope(qL1.rho - qm.rho, qR1.rho - qL1.rho);
        qL.u   = qL1.u   + 0.5f * limslope(qL1.u   - qm.u,   qR1.u   - qL1.u);
        qL.v   = qL1.v   + 0.5f * limslope(qL1.v   - qm.v,   qR1.v   - qL1.v);
        qL.p   = qL1.p   + 0.5f * limslope(qL1.p   - qm.p,   qR1.p   - qL1.p);
        qL.k   = qL1.k   + 0.5f * limslope(qL1.k   - qm.k,   qR1.k   - qL1.k);
        qL.w   = qL1.w   + 0.5f * limslope(qL1.w   - qm.w,   qR1.w   - qL1.w);
        if (qL.rho < RHOMIN || qL.p < PMIN || qL.k < 0.0f || qL.w < WMINV) qL = qL1;
    }
    if (!hi_done && (tR == 0 || tR == 3)) {
        St qp = fetch(P, ct, wd, i + oi, j + oj, i, j, dir, p0eff);
        qR.rho = qR1.rho - 0.5f * limslope(qR1.rho - qL1.rho, qp.rho - qR1.rho);
        qR.u   = qR1.u   - 0.5f * limslope(qR1.u   - qL1.u,   qp.u   - qR1.u);
        qR.v   = qR1.v   - 0.5f * limslope(qR1.v   - qL1.v,   qp.v   - qR1.v);
        qR.p   = qR1.p   - 0.5f * limslope(qR1.p   - qL1.p,   qp.p   - qR1.p);
        qR.k   = qR1.k   - 0.5f * limslope(qR1.k   - qL1.k,   qp.k   - qR1.k);
        qR.w   = qR1.w   - 0.5f * limslope(qR1.w   - qL1.w,   qp.w   - qR1.w);
        if (qR.rho < RHOMIN || qR.p < PMIN || qR.k < 0.0f || qR.w < WMINV) qR = qR1;
    }
#endif

    float Fi[6];
    float shock = 0.0f;
#if CARBFIX
    shock = fmaxf(shock_theta(P, G, cL), shock_theta(P, G, cR));
#endif
    riemann(qL, qR, dir, shock, Fi);
    float Fmass = Fi[0], Fn = Fi[1], Ft = Fi[2], FE = Fi[3];
    float Fk = Fi[4], Fw = Fi[5];

#if VISC
#if STRETCH
    // face-normal spacing: stretched in x (dir 0), uniform in y (dir 1)
    float dni = (dir == 0)
              ? 2.0f / (DX * (SXW[iL] + SXW[i]))
              : 1.0f / DX;
#else
    float dni = 1.0f / DX;
#endif
    float dudn = (qR1.u - qL1.u) * dni;
    float dvdn = (qR1.v - qL1.v) * dni;
    float dTdn = (qR1.T - qL1.T) * dni;
    float dkdn = (qR1.k - qL1.k) * dni;
    float dwdn = (qR1.w - qL1.w) * dni;
    float wgL = fL ? 1.0f : 0.0f, wgR = fR ? 1.0f : 0.0f;
    float wsi = 1.0f / fmaxf(wgL + wgR, 1.0f);
    float dudx, dudy, dvdx, dvdy;
    if (dir == 0) {
        dudx = dudn; dvdx = dvdn;
        dudy = (wgL * GF(1,cL) + wgR * GF(1,cR)) * wsi;
        dvdy = (wgL * GF(3,cL) + wgR * GF(3,cR)) * wsi;
    } else {
        dudy = dudn; dvdy = dvdn;
        dudx = (wgL * GF(0,cL) + wgR * GF(0,cR)) * wsi;
        dvdx = (wgL * GF(2,cL) + wgR * GF(2,cR)) * wsi;
    }
    float muf  = 0.5f * (qL1.mul + qR1.mul);
    float mutf = fmaxf(0.5f * (qL1.mut + qR1.mut), 0.0f);
    float mue  = muf + mutf;
    float dvg  = dudx + dvdy;
#if AXI
    // div(V) gains v/r; guarded inverse (the axis face itself has zero area)
    float rfc  = (dir == 0) ? RCELL(j) : RFACE(j);
    float rinv = rfc / (rfc * rfc + 0.0625f * DX * DX);
    dvg += 0.5f * (qL1.v + qR1.v) * rinv;
#endif
    float txx = mue * (2.0f * dudx - 0.66666667f * dvg);
    float tyy = mue * (2.0f * dvdy - 0.66666667f * dvg);
    float txy = mue * (dudy + dvdx);
    float uf = 0.5f * (qL1.u + qR1.u), vf = 0.5f * (qL1.v + qR1.v);
#if THERMO == 1
    float kap = RGAS * cpr_T(0.5f * (qL1.T + qR1.T)) * (muf / PRL + mutf / PRT);
#elif THERMO == 2
    // equilibrium cp = cv + R_eff at the face state (the reaction term in
    // CV models reaction-enhanced conduction)
    float lrf = log10f(fmaxf(0.5f * (qL1.rho + qR1.rho), RHOMIN));
    float ltf = log10f(fmaxf(0.5f * (qL1.T + qR1.T), TMINT));
    float kap = (eq_lerp(EQ_CV, lrf, ltf) + eq_lerp(EQ_RE, lrf, ltf))
              * (muf / PRL + mutf / PRT);
#else
    float kap = CPG * (muf / PRL + mutf / PRT);
#endif
    float F1f = 0.5f * (qL1.F1 + qR1.F1);
    float sigk = F1f * SIGK1 + (1.0f - F1f) * SIGK2;
    float sigw = F1f * SIGW1 + (1.0f - F1f) * SIGW2;
    if (dir == 0) {
        Fn -= txx; Ft -= txy;
        FE -= uf * txx + vf * txy + kap * dTdn;
    } else {
        Fn -= tyy; Ft -= txy;
        FE -= uf * txy + vf * tyy + kap * dTdn;
    }
    Fk -= (muf + sigk * mutf) * dkdn;
    Fw -= (muf + sigw * mutf) * dwdn;
#endif

    F[0*NC + cR] = Fmass;
    if (dir == 0) { F[1*NC + cR] = Fn; F[2*NC + cR] = Ft; }
    else          { F[1*NC + cR] = Ft; F[2*NC + cR] = Fn; }
    F[3*NC + cR] = FE;
    F[4*NC + cR] = Fk;
    F[5*NC + cR] = Fw;
}

// ---------------------------------------------------------------- sst_source
// Explicit production (+ cross-diffusion); destruction handled point-implicitly
// in rk_combine.
__global__ void sst_source(const float* P, const float* G, const float* s2,
                           const unsigned char* ct, float* sk, float* sw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || j < 2 || i >= NX + 2 || j >= NY + 2) return;
    int c = IDX(i, j);
    if (ct[c] != 0) { sk[c] = 0.0f; sw[c] = 0.0f; return; }
#if TURB
    float rho = PF(0,c), k = PF(5,c), w = PF(6,c);
    float mut = PF(8,c), F1 = PF(9,c);
    float S2e = fminf(s2[c], 10.0f * BSTAR * rho * k * w / fmaxf(mut, 1.0e-10f));
    float gamc = F1 * GAMC1 + (1.0f - F1) * GAMC2;
    float dkdw = GF(6,c)*GF(8,c) + GF(7,c)*GF(9,c);
    float CD = 2.0f * (1.0f - F1) * rho * SIGW2 / w * dkdw;
    sk[c] = mut * S2e;
    sw[c] = gamc * rho * S2e + CD;
#else
    sk[c] = 0.0f; sw[c] = 0.0f;
#endif
}

// ---------------------------------------------------------------- rk_combine
// U_new = ca*U0 + cb*U + cc*dt*RHS ; turbulence destruction point-implicit.
// Cut-cell embedded boundary: face fluxes are weighted by apertures AXF/AYF,
// the residual is divided by the cell fluid volume fraction LAM, and the
// embedded wall segment contributes pressure + viscous shear with the smooth
// surface normal (from aperture differences).
__global__ void rk_combine(const float* U0, float* U, const float* P,
                           const float* G, const float* FX, const float* FY,
                           const float* sk, const float* sw,
                           const float* dtl, const unsigned char* ct,
                           const float* AXF, const float* AYF, const float* LAM,
                           const float* wd,
                           float* res, float* QW,
                           float ca, float cb, float cc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || j < 2 || i >= NX + 2 || j >= NY + 2) return;
    int c = IDX(i, j);
    if (ct[c] != 0) { res[c] = 0.0f; return; }
    int cE = IDX(i + 1, j), cN = IDX(i, j + 1);
    float dxi = 1.0f / DX;                    // y-direction (never stretched)
#if STRETCH
    float dxix = 1.0f / (DX * SXW[i]);        // stretched x-cell width
#else
    float dxix = dxi;
#endif
    float dt = dtl[c];
    float aw = AXF[c], ae = AXF[cE], asf = AYF[c], an = AYF[cN];
    float lam = LAM[c];
    float lame = fmaxf(lam, 0.25f);          // small-cell stabilization
    float li = 1.0f / lame;
    float pc = PF(3,c);
    float rhs[6];
#if AXI
    // axisymmetric finite volume: cell volume ~ r_c*dx*dr, y-face areas ~ r_f
    float rc  = RCELL(j);
    float rci = 1.0f / rc;
    float rfS = RFACE(j) * rci, rfN = RFACE(j + 1) * rci;
    #pragma unroll
    for (int m = 0; m < 6; m++)
        rhs[m] = (aw * FX[m*NC + c] - ae * FX[m*NC + cE]) * dxix
               + (rfS * asf * FY[m*NC + c] - rfN * an * FY[m*NC + cN]) * dxi;
    // wall closure chosen so a uniform-pressure still state stays exact
    // (together with the volumetric +p/r hoop source added after the
    //  1/lambda scaling of the surface terms)
    float swx = aw - ae;
    float swy = (rfS * asf - rfN * an) + DX * rci * lame;
#else
    #pragma unroll
    for (int m = 0; m < 6; m++)
        rhs[m] = (aw * FX[m*NC + c] - ae * FX[m*NC + cE]) * dxix
               + (asf * FY[m*NC + c] - an * FY[m*NC + cN]) * dxi;
    float swx = aw - ae;
    float swy = asf - an;
#endif
    // ---- embedded wall segment: pressure force along the smooth normal ----
    rhs[1] -= pc * swx * dxix;
    rhs[2] -= pc * swy * dxi;
#if VISC && NOSLIP
    // wall shear on the embedded segment via the Reichardt wall function:
    // tau_w = rho u_tau^2 — valid from the viscous sublayer (where it
    // reduces to mu u_t/d) through the log layer, so the result no longer
    // depends on where y+ happens to land on the cut-cell mesh
    {
        float gx = aw - ae, gy = asf - an;       // geometric normal (planar)
        float area = sqrtf(gx * gx + gy * gy);
        if (area > 1.0e-4f) {
            float nxw = gx / area, nyw = gy / area;
            float uc = PF(1,c), vc2 = PF(2,c);
            float un_ = uc * nxw + vc2 * nyw;
            float utx = uc - un_ * nxw, uty = vc2 - un_ * nyw;
            float umag = sqrtf(utx * utx + uty * uty);
            float dwl = fmaxf(wd[c], 0.3f * DX);
            float rhoc = PF(0,c), mulc = PF(7,c);
            float ut2 = utau_wf(umag, dwl, rhoc, mulc);
            float coef = rhoc * ut2 * ut2 / fmaxf(umag, 1.0e-6f)
                       * area / DX;
            rhs[1] -= coef * utx;
            rhs[2] -= coef * uty;
#if WALLTH
            // isothermal wall: heat flux via Kader's thermal wall law
            float yp = dwl * ut2 * rhoc / fmaxf(mulc, 1.0e-12f);
            float tp = fmaxf(tplus_kader(fmaxf(yp, 1.0e-3f), PRL), 1.0e-3f);
#if THERMO == 1
            float cpw = RGAS * cpr_T(PF(4,c));
#elif THERMO == 2
            float lrw = log10f(fmaxf(rhoc, RHOMIN));
            float ltw = log10f(fmaxf(PF(4,c), TMINT));
            float cpw = eq_lerp(EQ_CV, lrw, ltw) + eq_lerp(EQ_RE, lrw, ltw);
#else
            const float cpw = CPG;
#endif
            float qw = rhoc * cpw * fmaxf(ut2, 1.0e-6f)
                     * (PF(4,c) - WALL_TW) / tp;     // convective (Kader)
#if RADWALL
            // gray-gas radiative load from the near-wall gas to the wall
            float Tg = PF(4,c);
            qw += WEMISS * 5.670374e-8f
                * (Tg*Tg*Tg*Tg - WALL_TW*WALL_TW*WALL_TW*WALL_TW);
#endif
            rhs[3] -= qw * area / DX;
            QW[c] = qw;                       // diagnostic: wall heat flux
#endif
        }
    }
#endif
    // surface terms scale with 1/(fluid volume fraction)
    #pragma unroll
    for (int m = 0; m < 6; m++)
        rhs[m] *= li;
    // volumetric sources (per unit fluid volume, no lambda scaling)
#if AXI
    {
        float vc  = PF(2,c);
        float vor = vc * rci;
        float mue = PF(7,c) + fmaxf(PF(8,c), 0.0f);
        float dvg = GF(0,c) + GF(3,c) + vor;
        float tqq = mue * (2.0f * vor - 0.66666667f * dvg);
        rhs[2] += (pc - tqq) * rci;
    }
#endif
    rhs[4] += sk[c];
    rhs[5] += sw[c];
    res[c] = rhs[0];

    float r  = ca * U0[0*NC+c] + cb * U[0*NC+c] + cc * dt * rhs[0];
    float ru = ca * U0[1*NC+c] + cb * U[1*NC+c] + cc * dt * rhs[1];
    float rv = ca * U0[2*NC+c] + cb * U[2*NC+c] + cc * dt * rhs[2];
    float E  = ca * U0[3*NC+c] + cb * U[3*NC+c] + cc * dt * rhs[3];

    // point-implicit destruction for k and omega
    float w  = PF(6,c), F1 = PF(9,c);
    float bet = F1 * BET1 + (1.0f - F1) * BET2;
    float bstar_c = BSTAR;
#if COMPCORR
    // Wilcox compressibility correction: dilatational dissipation grows with
    // the turbulent Mach number Mt above Mt0=0.5, moving destruction from
    // omega into k (slower high-Mach shear-layer spreading; off at walls
    // where k->0). xi* = 1.5, Mt0^2 = 0.25.
    {
        float a2c = GAM * fmaxf(PF(3,c), PMIN) / fmaxf(PF(0,c), RHOMIN);
        float Mt2 = 2.0f * fmaxf(PF(5,c), 0.0f) / fmaxf(a2c, 1.0e-6f);
        float Fc = fmaxf(Mt2 - 0.25f, 0.0f);
        bstar_c = BSTAR * (1.0f + 1.5f * Fc);
        bet = fmaxf(bet - BSTAR * 1.5f * Fc, 0.0f);
    }
#endif
    float rk = (ca * U0[4*NC+c] + cb * U[4*NC+c] + cc * dt * rhs[4])
             / (1.0f + cc * dt * bstar_c * w);
    float rw = (ca * U0[5*NC+c] + cb * U[5*NC+c] + cc * dt * rhs[5])
             / (1.0f + cc * dt * 2.0f * bet * w);
#if TURB
    // cut cells: anchor omega to Menter's wall value (smooth-wall treatment)
    if (lam < 0.999f) {
        float rs  = fminf(fmaxf(r, RHOMIN), RHOMAX);
        float dwl = fmaxf(wd[c], 0.25f * DX);
        float nuw = PF(7,c) / fmaxf(PF(0,c), RHOMIN);
        float wwall = fminf(60.0f * nuw / (BET1 * dwl * dwl), WMAXV);
        rw = fmaxf(rw, rs * wwall);
    }
#endif

    // ---- sanitize ----
    r = fminf(fmaxf(r, RHOMIN), RHOMAX);
    float inv = 1.0f / r;
    float u = ru * inv, v = rv * inv;
    u = fminf(fmaxf(u, -VMAX), VMAX);
    v = fminf(fmaxf(v, -VMAX), VMAX);
    float ke = 0.5f * r * (u*u + v*v);
#if THERMO == 1
    // Repair E whenever the recovered temperature does not reproduce the
    // internal energy (Newton clamped at its T bounds — e.g. transiently
    // negative E-ke in deep expansions — or the p clamp fired). In the
    // normal converged case the residual is float32 noise and E is left
    // untouched, so energy stays exactly conserved.
    float eRn = (E - ke) / (r * RGAS);
    float Tn = T_from_e(eRn, PF(4,c));
    float p = r * RGAS * Tn;
    p = fminf(fmaxf(p, PMIN), PMAX);
    float Tc2 = p / (r * RGAS);
    if (fabsf(eR_T(Tc2) - eRn) > 1.0e-3f * fabsf(eRn) + 1.0e-2f)
        E = r * RGAS * eR_T(Tc2) + ke;
#elif THERMO == 2
    // same repair logic against the equilibrium tables
    float lrn = log10f(r);
    float en = (E - ke) / r;
    float Tn = eq_T_from_e(lrn, en, PF(4,c));
    float ltn = log10f(Tn);
    float p = r * eq_lerp(EQ_RE, lrn, ltn) * Tn;
    p = fminf(fmaxf(p, PMIN), PMAX);
    float et = eq_lerp(EQ_E, lrn, ltn);
    if (fabsf(et - en) > 1.0e-3f * fabsf(en) + 1.0e2f)
        E = r * et + ke;
#else
    float p = GM1 * (E - ke);
    p = fminf(fmaxf(p, PMIN), PMAX);
    E = p / GM1 + ke;
#endif
    rk = fminf(fmaxf(rk, r * KMINV), r * KMAXV);
    rw = fminf(fmaxf(rw, r * WMINV), r * WMAXV);

    U[0*NC+c] = r;  U[1*NC+c] = r * u; U[2*NC+c] = r * v;
    U[3*NC+c] = E;  U[4*NC+c] = rk;    U[5*NC+c] = rw;
}

// -------------------------------------------------------- scalar_transport
// Two-gamma plume mixing: transport the exhaust mass fraction Z (1 = pure
// exhaust, 0 = ambient air) as a conserved scalar rho*Z. It rides the SAME
// face mass fluxes (FX[0], FY[0]) the density used, so it is conservative and
// consistent with the flow; turbulent diffusion (Sc_t = 0.7) spreads it
// across the shear layer. Double-buffered (UZin -> UZout) to avoid races.
// Fully decoupled: launched only when two-gamma is on, changes nothing else.
__global__ void scalar_transport(const float* U, const float* UZ0,
        const float* UZin, float* UZout, const float* FX, const float* FY,
        const float* P, const float* AXF, const float* AYF, const float* LAM,
        const float* dtl, const unsigned char* ct, float* Zf,
        float ca, float cb, float cc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || j < 2 || i >= NX + 2 || j >= NY + 2) return;
    int c = IDX(i, j);
    if (ct[c] != 0) { UZout[c] = 0.0f; Zf[c] = 0.0f; return; }
    int cE = IDX(i+1,j), cN = IDX(i,j+1), cW = IDX(i-1,j), cS = IDX(i,j-1);
    float dxi = 1.0f / DX;
#if STRETCH
    float dxix = 1.0f / (DX * SXW[i]);
#else
    float dxix = dxi;
#endif
    float dt = dtl[c];
    float aw = AXF[c], ae = AXF[cE], asf = AYF[c], an = AYF[cN];
    float li = 1.0f / fmaxf(LAM[c], 0.25f);
    float rho_c = fmaxf(U[0*NC+c], RHOMIN);
    float Zc = fminf(fmaxf(UZin[c] / rho_c, 0.0f), 1.0f);
    // neighbour mixture fraction with ghost rules: inlet = 1 (exhaust),
    // farfield/outlet/wall = 0 (air; walls carry ~zero aperture flux anyway)
    #define ZG(nb) ( ct[nb]==0 ? \
        fminf(fmaxf(UZin[nb] / fmaxf(U[0*NC+nb], RHOMIN), 0.0f), 1.0f) \
        : (ct[nb]==2 ? 1.0f : 0.0f) )
    float Zw = ZG(cW), Ze = ZG(cE), Zs = ZG(cS), Zn = ZG(cN);
    // advective Z-flux on each face = (mass flux) * (upwind Z)
    float mfw = FX[0*NC+c], mfe = FX[0*NC+cE];
    float mfs = FY[0*NC+c], mfn = FY[0*NC+cN];
    float fzw = mfw * (mfw >= 0.0f ? Zw : Zc);
    float fze = mfe * (mfe >= 0.0f ? Zc : Ze);
    float fzs = mfs * (mfs >= 0.0f ? Zs : Zc);
    float fzn = mfn * (mfn >= 0.0f ? Zc : Zn);
    float rhsZ;
#if AXI
    float rc = RCELL(j), rci = 1.0f / rc;
    float rfS = RFACE(j) * rci, rfN = RFACE(j+1) * rci;
    rhsZ = (aw*fzw - ae*fze) * dxix + (rfS*asf*fzs - rfN*an*fzn) * dxi;
#else
    rhsZ = (aw*fzw - ae*fze) * dxix + (asf*fzs - an*fzn) * dxi;
#endif
    // turbulent diffusion of Z (Sc_t = 0.7), fluid-fluid faces only
    const float SCTI = 1.0f / 0.7f;
    float mutc = fmaxf(P[8*NC+c], 0.0f);
    float diff = 0.0f;
    if (ct[cW]==0) diff += aw*0.5f*(mutc+fmaxf(P[8*NC+cW],0.0f))*SCTI*(Zw-Zc)*dxix*dxix;
    if (ct[cE]==0) diff += ae*0.5f*(mutc+fmaxf(P[8*NC+cE],0.0f))*SCTI*(Ze-Zc)*dxix*dxix;
    if (ct[cS]==0) diff += asf*0.5f*(mutc+fmaxf(P[8*NC+cS],0.0f))*SCTI*(Zs-Zc)*dxi*dxi;
    if (ct[cN]==0) diff += an*0.5f*(mutc+fmaxf(P[8*NC+cN],0.0f))*SCTI*(Zn-Zc)*dxi*dxi;
    rhsZ = (rhsZ + diff) * li;
    float UZn = ca*UZ0[c] + cb*UZin[c] + cc*dt*rhsZ;
    float Z = fminf(fmaxf(UZn / rho_c, 0.0f), 1.0f);
    UZout[c] = Z * rho_c;
    Zf[c] = Z;
    #undef ZG
}

} // extern "C"
""")


def _f(x: float) -> str:
    """Format a python float as a CUDA float literal."""
    return f"{float(x):.8e}f"


def axis_j(cfg, ny: int) -> float:
    """Axis position in padded-grid j coordinates (half-integer: between rows).

    Interior rows occupy j = 2 .. ny+1. 'top'/'bottom' put the axis on the
    image edge (symmetry plane); 'center' puts it mid-image.

    For an even row count the mid-image position is a half-integer (a cell
    face), so no cell centre lands on r=0 and the grid is symmetric about the
    axis. ``load_mask(axisym_center=True)`` guarantees the even count for real
    runs; the half-cell nudge below only fires on the odd-count fallback (e.g.
    raw test masks) to keep the hard 1/r reciprocal in the kernel finite.
    """
    loc = getattr(cfg, "axis_location", "center")
    if loc == "top":
        return 1.5
    if loc == "bottom":
        return ny + 1.5
    a = 2.0 + ny / 2.0 - 0.5
    if abs(a - round(a)) < 0.25:        # odd ny fallback: nudge off cell centres
        a += 0.5
    return a


def _scheme_id(cfg, mode: int) -> int:
    s = {"hll": 0, "hllc": 1, "roe": 2, "ausm": 3}.get(
        cfg.flux_scheme.lower().rstrip("+"), 1)
    if s == 2 and mode == 2:
        # Roe's characteristic decomposition assumes a perfect gas
        # (entropy-wave energy projection e = p/((gamma-1) rho)); with the
        # tabular equilibrium EOS it injects energy at contacts. HLLC is
        # EOS-general, so fall back to it.
        print("note: Roe is not available with the equilibrium gas model "
              "- using HLLC instead")
        return 1
    return s


def gas_mode(cfg) -> int:
    """0 = calorically perfect, 1 = thermally perfect, 2 = equilibrium."""
    gm = getattr(cfg, "gas_model", "calorically perfect").lower()
    if gm.startswith("equilibrium"):
        return 2
    if gm.startswith("thermally"):
        return 1
    return 0


def build_source(cfg, nx: int, ny: int) -> str:
    g = cfg.gamma
    mode = gas_mode(cfg)
    # combustion efficiency: incomplete energy release lowers the effective
    # chamber temperature (c* ~ sqrt(T0)  ->  T0_eff = eta^2 * T0)
    eta = float(getattr(cfg, "eta_cstar", 1.0))
    t0_eff = cfg.inlet_T0 * eta * eta
    cpr_c = (g / (g - 1.0), 0.0, 0.0, 0.0)
    g_in = g
    rgfar = cfg.R_gas
    tab_subs = dict(TNLR=2, TNLT=2, TLR0="0.0f", TDLR="1.0f",
                    TLT0="0.0f", TDLT="1.0f", TMINT="1.0f", TMAXT="1.0f",
                    NISEN=2, ISEN_DEF="")
    if mode == 1:
        from . import thermo
        cpr_c = thermo.cpr_coeffs(cfg.propellant, fallback_gamma=g)
        g_in = float(thermo.gamma_of_T(cpr_c, t0_eff))
    pchoke = (2.0 / (g_in + 1.0)) ** (g_in / (g_in - 1.0))
    if mode == 2:
        from . import equilibrium as eqm
        if cfg.propellant not in eqm.REACTANTS:
            raise ValueError(
                "Equilibrium mode needs a propellant selection "
                "(e.g. LOX/RP-1, LOX/LH2, LOX/Ethanol or UDMH/N2O4).")
        tab = eqm.build_tables(cfg.propellant)
        prs, Ti, Vi, Rei, ich = eqm.chamber_isentrope(
            cfg.propellant, cfg.inlet_p0, t0_eff)
        pchoke = float(prs[ich])
        order = np.argsort(prs)                       # ascending pr axis
        pr_a, T_a, V_a, Re_a = (prs[order], Ti[order], Vi[order], Rei[order])

        def carr(name, vals):
            body = ",".join(f"{float(v):.7e}f" for v in vals)
            return f"__device__ const float {name}[{len(vals)}] = {{{body}}};"
        isen_def = "\n".join([
            f"#define ISEN_PR0 {pr_a[0]:.7e}f",
            f"#define ISEN_DPR {pr_a[1] - pr_a[0]:.7e}f",
            carr("ISEN_T", T_a), carr("ISEN_V", V_a), carr("ISEN_RE", Re_a)])
        lr, lt = tab["lr_ax"], tab["lt_ax"]
        tab_subs = dict(
            TNLR=len(lr), TNLT=len(lt),
            TLR0=_f(lr[0]), TDLR=_f(lr[1] - lr[0]),
            TLT0=_f(lt[0]), TDLT=_f(lt[1] - lt[0]),
            TMINT=_f(10.0 ** lt[0] * 1.02), TMAXT=_f(10.0 ** lt[-1] * 0.99),
            NISEN=len(pr_a), ISEN_DEF=isen_def)
        _, _, rgfar = eqm.ambient_state(tab, cfg.farfield_p, cfg.farfield_T)
    # effective cell size: mesh_scale resamples the geometry to a finer grid
    # (more cells, smaller cells) keeping the physical size fixed, so the
    # kernel's cell width must be meters_per_pixel / mesh_scale, matching
    # mask.dx. (Using meters_per_pixel alone mis-scaled every run with
    # mesh_scale != 1 — the bug this fixes.)
    dx_eff = cfg.meters_per_pixel / max(getattr(cfg, "mesh_scale", 1.0), 1e-9)
    subs = dict(
        SX=nx + 4, SY=ny + 4, NX=nx, NY=ny,
        DX=_f(dx_eff),
        GAM=_f(g), GM1=_f(g - 1.0), RGAS=_f(cfg.R_gas), CPG=_f(cfg.cp),
        RGFAR=_f(rgfar),
        THERMO=mode,
        CPR0=_f(cpr_c[0]), CPR1=_f(cpr_c[1]),
        CPR2=_f(cpr_c[2]), CPR3=_f(cpr_c[3]),
        GAMIN=_f(g_in), GM1IN=_f(g_in - 1.0),
        **tab_subs,
        PRL=_f(cfg.Pr), PRT=_f(cfg.Pr_t),
        MUREF=_f(cfg.mu_ref), TREFS=_f(cfg.T_ref_sutherland), SSUTH=_f(cfg.S_sutherland),
        P0IN=_f(cfg.inlet_p0), T0IN=_f(t0_eff), PCHOKE=_f(pchoke),
        KINFAC=_f(1.5 * cfg.inlet_turb_intensity ** 2), MUTRIN=_f(cfg.inlet_mut_ratio),
        MUTMAX=_f(max(getattr(cfg, "mut_max_ratio", 1.0e5), 1.0)),
        OUTRELAX=_f(min(max(getattr(cfg, "outlet_relax", 1.0), 0.01), 1.0)),
        OUTRELAXONE=1 if getattr(cfg, "outlet_relax", 1.0) >= 0.9999 else 0,
        PFAR=_f(cfg.farfield_p), TFAR=_f(cfg.farfield_T),
        UFAR=_f(cfg.farfield_u), VFAR=_f(cfg.farfield_v),
        KFAR=_f(1.0e-6), WFAR=_f(10.0),
        CFL=_f(cfg.cfl),
        SCHEME=_scheme_id(cfg, mode),
        ORDER2=1 if cfg.muscl_order >= 2 else 0,
        WENO=1 if cfg.muscl_order >= 5 else 0,
        LIM={"minmod": 0, "vanalbada": 1, "vanleer": 2,
             "superbee": 3}.get(cfg.limiter.lower().replace(" ", ""), 0),
        VISC=1 if cfg.viscous else 0,
        TURB=1 if (cfg.turbulence and cfg.viscous) else 0,
        NOSLIP=1 if cfg.wall_type == "noslip" else 0,
        WALLTH=1 if (getattr(cfg, "wall_T", 0.0) > 0.0
                     and cfg.wall_type == "noslip" and cfg.viscous) else 0,
        WALL_TW=_f(max(getattr(cfg, "wall_T", 0.0), 1.0)),
        RADWALL=1 if (getattr(cfg, "wall_emissivity", 0.0) > 0.0
                      and getattr(cfg, "wall_T", 0.0) > 0.0
                      and cfg.wall_type == "noslip" and cfg.viscous) else 0,
        WEMISS=_f(getattr(cfg, "wall_emissivity", 0.0)),
        STRETCH=1 if getattr(cfg, "plume_stretch", 1.0) > 1.0 + 1e-6 else 0,
        CARBFIX=1 if getattr(cfg, "carbuncle_fix", True) else 0,
        COMPCORR=1 if getattr(cfg, "compressibility_correction", False) else 0,
        AXI=1 if getattr(cfg, "axisymmetric", False) else 0,
        AXISJ=_f(axis_j(cfg, ny)),
        AXSYM_TOP=1 if (getattr(cfg, "axisymmetric", False)
                        and cfg.axis_location == "top") else 0,
        AXSYM_BOT=1 if (getattr(cfg, "axisymmetric", False)
                        and cfg.axis_location == "bottom") else 0,
    )
    return _CUDA_SRC.substitute(subs)


def compute_stretch(mask, cfg):
    """Per-column x-width multiplier (padded, length nx+4) for downstream
    plume stretching, or None when disabled. Columns stay uniform (1.0)
    through the nozzle and start growing geometrically a few cells past the
    last wall, so all cut-cell walls remain on the uniform grid."""
    ratio = float(getattr(cfg, "plume_stretch", 1.0))
    if ratio <= 1.0 + 1e-6:
        return None
    from .mask import WALL
    nx = mask.nx
    ct = mask.cell_type                       # padded (ny+4, nx+4)
    wall_cols = np.where((ct[2:-2, 2:-2] == WALL).any(axis=0))[0]
    start = (int(wall_cols.max()) + 5) if wall_cols.size else int(0.4 * nx)
    start = min(max(start, 1), nx - 1)
    cap = 8.0                                 # cells grow up to 8x the base
    sx = np.ones(nx, dtype=np.float64)
    for k in range(start, nx):
        sx[k] = min(sx[k - 1] * ratio, cap)
    # pad: halo columns copy the nearest interior column
    return np.concatenate([[sx[0], sx[0]], sx, [sx[-1], sx[-1]]])


class KernelSet:
    """Compiles the CUDA module and exposes ready-to-launch kernels."""

    BLOCK = (32, 8)

    def __init__(self, cfg, nx: int, ny: int, stretch_sx=None):
        import cupy as cp
        self.cp = cp
        src = build_source(cfg, nx, ny)
        self.module = cp.RawModule(code=src, options=("-use_fast_math", "--std=c++11"))
        if gas_mode(cfg) == 2:
            from . import equilibrium as eqm
            tab = eqm.build_tables(cfg.propellant)      # disk-cached
            for gname, key in (("EQ_E", "E"), ("EQ_RE", "RE"),
                               ("EQ_A", "A"), ("EQ_CV", "CV")):
                ptr = self.module.get_global(gname)
                dst = cp.ndarray(tab[key].size, dtype=cp.float32, memptr=ptr)
                dst[...] = cp.asarray(
                    np.ascontiguousarray(tab[key], dtype=np.float32).ravel())
        if stretch_sx is not None and getattr(cfg, "plume_stretch", 1.0) > 1.0 + 1e-6:
            sxw = np.ascontiguousarray(stretch_sx, dtype=np.float32)
            assert sxw.size == nx + 4, (sxw.size, nx + 4)
            ptr = self.module.get_global("SXW")
            dst = cp.ndarray(sxw.size, dtype=cp.float32, memptr=ptr)
            dst[...] = cp.asarray(sxw)
        self.sx, self.sy = nx + 4, ny + 4
        bx, by = self.BLOCK
        self.grid = ((self.sx + bx - 1) // bx, (self.sy + by - 1) // by)
        for name in ("cons2prim", "halo_fill", "gradients", "turb_visc",
                     "fluxes", "sst_source", "rk_combine", "scalar_transport"):
            setattr(self, name, self.module.get_function(name))

    def launch(self, kernel, *args):
        kernel(self.grid, self.BLOCK, args)
