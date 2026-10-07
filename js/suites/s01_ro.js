// Suite 1 — Reverse osmosis and membrane design.
// Element-by-element, segment-resolved model of spiral-wound arrays: solution–diffusion,
// Spiegler–Kedem, Kedem–Katchalsky or Donnan–steric pore (extended Nernst–Planck) transport coupled
// with film-theory or two-dimensional boundary-layer concentration polarisation, spacer-channel
// hydraulics, Hagen–Poiseuille permeate-side losses, multi-ion permeate quality with electroneutrality,
// staging, inter-stage boosting, concentrate recycle, an optional second pass and energy recovery,
// plus a start-up transient, a cleaning assessment and a physics-informed surrogate of the array.
import { brent, clamp, linspace, sum, rng, fmt, solveLinear, tridiag, rk4, lhs, lstsq, interp1 } from '../core/num.js';
import { density, viscosity, diffusivityNaCl, tcf, salinityFromTDS, R as RGAS } from '../core/props.js';
import { analyzeWater } from './s02_chem.js';
import { IONS, ION_IDS, WATERS, cloneIons, tds, scaleIons, mixIons, osmoticPressureIons, vantHoff, chargeBalance, conductivity, molar } from '../core/water.js';

/** Generic 8-inch element classes derived from standard-test performance (not brand specific). */
export const MEMBRANES = {
  swhr: { name: 'Seawater · high rejection (37 m²)', area: 37.2, A: 1.25, B: 0.058, sigma: 0.998, pmax: 83, spacerMil: 28, fBoron: 38, fDiv: 0.35, test: '32 g/L NaCl, 55 bar, 8 % recovery → 28 m³/d, 99.80 %' },
  swle: { name: 'Seawater · low energy (41 m²)', area: 40.9, A: 1.7, B: 0.1, sigma: 0.997, pmax: 83, spacerMil: 28, fBoron: 32, fDiv: 0.35, test: '32 g/L NaCl, 55 bar, 8 % recovery → 34 m³/d, 99.75 %' },
  swule: { name: 'Seawater · ultra-low energy (41 m²)', area: 40.9, A: 2.25, B: 0.17, sigma: 0.996, pmax: 83, spacerMil: 28, fBoron: 28, fDiv: 0.35, test: '32 g/L NaCl, 55 bar, 8 % recovery → 47 m³/d, 99.70 %' },
  bwhr: { name: 'Brackish · high rejection (37 m²)', area: 37.2, A: 3.3, B: 0.19, sigma: 0.995, pmax: 41, spacerMil: 34, fBoron: 90, fDiv: 0.3, test: '2 g/L NaCl, 15.5 bar, 15 % recovery → 40 m³/d, 99.5 %' },
  bwle: { name: 'Brackish · low energy (41 m²)', area: 40.9, A: 5.7, B: 0.3, sigma: 0.993, pmax: 41, spacerMil: 34, fBoron: 110, fDiv: 0.3, test: '2 g/L NaCl, 10.3 bar, 15 % recovery → 42 m³/d, 99.3 %' },
  nf: { name: 'Nanofiltration · softening (37 m²)', area: 37.2, A: 8.5, B: 28, sigma: 0.85, pmax: 41, spacerMil: 34, fBoron: 4, fDiv: 0.03, test: '2 g/L MgSO₄, 4.8 bar, 15 % recovery → 97 % MgSO₄ rejection' },
};
// Salt-permeability of each constituent relative to NaCl (boron handled separately through pH speciation).
const REL_B = { Na: 1, K: 1.25, NH4: 1.9, Cl: 1, NO3: 2.6, F: 1.4, HCO3: 1.15, SiO2: 1.1, B: 1 };
const DIVALENT = ['Ca', 'Mg', 'Ba', 'Sr', 'Fe', 'Mn', 'SO4', 'CO3', 'PO4'];
const EL_LEN = 1.016, POROSITY = 0.89;

function relB(id, M, pH, T) {
  if (id === 'B') {
    const pKa = 9.24 - 0.012 * (T - 25) - 0.25; // apparent pKa in saline water
    const borate = 1 / (1 + 10 ** (pKa - pH));
    return (1 - borate) * M.fBoron + borate * 1.2;
  }
  if (DIVALENT.includes(id)) return M.fDiv;
  return REL_B[id] ?? 1;
}

// ---- constants and small physical helpers used by the optional transport and channel models ---------------
const KB = 1.380649e-23, QE = 1.602176634e-19, EPS0 = 8.8541878128e-12, EPS_W = 78.4, V_W = 1.8e-5, MU25 = viscosity(25, 0);
/** Stokes–Einstein radius (m) from the infinite-dilution diffusivity at 25 °C. */
export const stokesRadius = (D) => (KB * 298.15) / (6 * Math.PI * MU25 * D);
/** Hagen–Poiseuille pore-flow permeability A = ε·r²/(8·μ·τ·δ), m/(s·Pa), and the equivalent pore radius for a given A. */
export const poreFlowA = (r, eps, tau, delta, mu = MU25) => (eps * r * r) / (8 * mu * tau * delta);
export const poreRadiusFromA = (A, eps, tau, delta, mu = MU25) => Math.sqrt((8 * mu * tau * delta * A) / eps);
/** Solution–diffusion identity A = D_w·φ_w·V_w/(R·T·δ): water diffusivity in the active layer (m²/s) for a given A (m/s/Pa). */
export const waterDiffusivitySD = (A, phiW, delta, T = 25) => (A * RGAS * (T + 273.15) * delta) / (phiW * V_W);
/** Free-volume (Yasuda) effect of the membrane water content on water (b = 0.45) and salt (b = 1.2) permeability. h = fraction of full hydration. */
export const hydrationFactors = (h, phiW) => { const x = clamp(h, 0.7, 1), g = 1 / (phiW * x) - 1 / phiW; return { A: x * Math.exp(-0.45 * g), B: x * Math.exp(-1.2 * g) }; };
/** Pressure loss of a round tube, bar (Q in m³/h): Hagen–Poiseuille 32·μ·L·v/D² when laminar, Blasius friction when turbulent. */
export function tubeDP(Q, D, L, rho, mu) {
  const v = Math.abs(Q) / 3600 / ((Math.PI * D * D) / 4), Re = (rho * v * D) / mu;
  if (!(v > 0)) return 0;
  return (Re < 2300 ? (32 * mu * L * v) / (D * D) : (0.3164 * Re ** -0.25 * (L / D) * rho * v * v) / 2) / 1e5;
}
/** Area-mean back-pressure of a permeate leaf: slit Hagen–Poiseuille flow with uniform wall inflow, bar per L/m²·h of flux. */
export const leafCoeff = (mu, Lleaf, hp, kPerm = 1) => (kPerm * 8 * mu * Lleaf * Lleaf) / hp ** 3 / 3.6e6 / 1e5;
/** Back-pressure (bar) at each element of one vessel caused by flow in the central permeate tube. qp: m³/h per element. */
export function tubeLoss(qp, D, rho, mu, port = 'conc') {
  const n = qp.length, q = port === 'feed' ? [...qp].reverse() : qp, dp = new Array(n), out = new Array(n);
  let cum = 0;
  for (let j = 0; j < n; j++) { dp[j] = tubeDP(cum + q[j] / 2, D, EL_LEN, rho, mu); cum += q[j]; }
  let down = 0;
  for (let j = n - 1; j >= 0; j--) { out[j] = down + dp[j] / 2; down += dp[j]; }
  return port === 'feed' ? out.reverse() : out;
}

/** Kedem–Katchalsky solute passage cp/cm: Js = ω·Δπ + (1 − σ)·c̄·Jv with the logarithmic-mean concentration c̄ (B = ω·R·T). */
export function kkPassage(Jw, B, sig) {
  if (!(Jw > 0)) return 1;
  if (sig >= 1) return B / (Jw + B);
  if (!(sig > 0)) return 1;
  const g = (x) => Jw * x - B * (1 - x) - ((1 - sig) * Jw * (1 - x)) / Math.log(1 / x);
  // safeguarded Newton from the Spiegler–Kedem passage; bisection-type fallback if it leaves (0, 1) or stalls
  const F = Math.exp((-Jw * (1 - sig)) / Math.max(B, 1e-9));
  let x = clamp((1 - sig) / (1 - sig * F), 1e-12, 1 - 1e-9);
  for (let i = 0; i < 30; i++) {
    const L = -Math.log(x), gx = Jw * x - B * (1 - x) - ((1 - sig) * Jw * (1 - x)) / L, dg = Jw + B - ((1 - sig) * Jw * (-L + (1 - x) / x)) / (L * L), xn = x - gx / dg;
    if (!(xn > 0 && xn < 1) || !Number.isFinite(xn)) break;
    if (Math.abs(xn - x) <= 1e-13 * x) return xn;
    x = xn;
  }
  return brent(g, 1e-14, 1 - 1e-12, 1e-13);
}

/**
 * Two-dimensional steady convection–diffusion of solute in a slit feed channel between two membranes.
 * Velocity: Berman profile u = 1.5·ū(x)·(1 − η²), v = v_w·η(3 − η²)/2 — no slip at the membrane (η = 1), symmetry at the mid-plane (η = 0).
 * Solute: ∂(uc)/∂x + ∂(vc)/∂y = D ∂²c/∂y², ∂c/∂y = 0 at the mid-plane, D ∂c/∂y = v_w·R·c_w at the membrane (R = 0 with v_w = 0: impermeable wall).
 * Finite volumes clustered at the wall, implicit marching in x. Inlet concentration = 1. Lengths in m, velocities in m/s.
 */
export function channelBL({ ub, h, L, D, vw, R = 1, ny = 24, nx = 16, field = false }) {
  const b = h / 2, f = (e) => 0.5 * e * (3 - e * e), ef = new Float64Array(ny + 1), yc = new Float64Array(ny), df = new Float64Array(ny), vf = new Float64Array(ny + 1);
  for (let j = 0; j <= ny; j++) { ef[j] = 1 - ((ny - j) / ny) ** 2; vf[j] = vw * f(ef[j]); }
  for (let j = 0; j < ny; j++) { yc[j] = (b * (ef[j] + ef[j + 1])) / 2; df[j] = f(ef[j + 1]) - f(ef[j]); }
  const dW = (b * (1 - ef[ny - 1])) / 2, wallF = 1 / (1 - (vw * R * dW) / D);
  let c = new Array(ny).fill(1), xp = 0, cpInt = 0, perm = 0, prev = 1;
  const xs = [0], cw = [1], cb = [1], rows = field ? [c.slice()] : null, lo = new Array(ny), di = new Array(ny), up = new Array(ny), rh = new Array(ny);
  for (let n = 1; n <= nx; n++) {
    const x = L * (n / nx) ** 2, dx = x - xp, u0 = ub - (vw * xp) / b, u1 = Math.max(ub - (vw * x) / b, 1e-9 * ub);
    for (let j = 0; j < ny; j++) {
      let d = (u1 * b * df[j]) / dx, l = 0, u = 0;
      if (j < ny - 1) { const dn = yc[j + 1] - yc[j], w = (yc[j + 1] - b * ef[j + 1]) / dn, Dd = D / dn; d += vf[j + 1] * w + Dd; u = vf[j + 1] * (1 - w) - Dd; }
      else d += vw * (1 - R) * wallF;
      if (j > 0) { const dn = yc[j] - yc[j - 1], w = (yc[j] - b * ef[j]) / dn, Dd = D / dn; l = -vf[j] * w - Dd; d += -vf[j] * (1 - w) + Dd; }
      lo[j] = l; di[j] = d; up[j] = u; rh[j] = ((u0 * b * df[j]) / dx) * c[j];
    }
    c = tridiag(lo, di, up, rh);
    let mix = 0; for (let j = 0; j < ny; j++) mix += df[j] * c[j];
    const cwn = c[ny - 1] * wallF, ratio = cwn / mix;
    cpInt += 0.5 * (prev + ratio) * dx; perm += vw * (1 - R) * cwn * dx; prev = ratio; xp = x;
    xs.push(x); cw.push(cwn); cb.push(mix); if (rows) rows.push(c.slice());
  }
  const uEnd = Math.max(ub - (vw * L) / b, 1e-9 * ub);
  return { x: xs, cw, cb, cpMean: cpInt / L, yc: Array.from(yc), b, rows, shear: (3 * ub) / b, balance: (ub * b - uEnd * b * cb[nx] - perm) / (ub * b) };
}
/** Effective mass-transfer coefficient (m/s) of a re-developing boundary layer of length L from the two-dimensional solution. */
const blK = (ub, h, L, D, vw) => vw / Math.log(Math.max(channelBL({ ub, h, L, D, vw }).cpMean, 1 + 1e-12));
/** Lévêque local mass-transfer coefficient for a uniform wall flux, m/s (wall shear rate γ = 6ū/h). */
export const levequeK = (ub, h, x, D) => 0.651 * D ** (2 / 3) * ((6 * ub) / h / x) ** (1 / 3);

/** Damped Newton with a finite-difference Jacobian that gives up as soon as the residual cannot be reduced (used for continuation). */
function newtonFail(F, x0, tol, maxIter) {
  let x = x0.slice(), fx = F(x), nf = Math.hypot(...fx);
  const n = x.length;
  for (let it = 0; it < maxIter && Number.isFinite(nf); it++) {
    if (nf < tol) return { x, converged: true, residual: nf };
    const J = Array.from({ length: n }, () => new Array(n));
    for (let j = 0; j < n; j++) { const xj = x.slice(), dx = 1e-7 * Math.max(1, Math.abs(x[j])); xj[j] += dx; const fj = F(xj); for (let i = 0; i < n; i++) J[i][j] = (fj[i] - fx[i]) / dx; }
    let st;
    try { st = solveLinear(J, fx.map((q) => -q)); } catch { break; }
    let lam = 1, xn = null, fn = null, nn = Infinity;
    for (; lam > 0.01; lam *= 0.5) { xn = x.map((q, i) => q + lam * st[i]); fn = F(xn); nn = Math.hypot(...fn); if (Number.isFinite(nn) && nn < nf) break; }
    if (!(nn < nf)) break;
    x = xn; fx = fn; nf = nn;
  }
  return { x, converged: nf < tol, residual: nf };
}

/** Steric partitioning and hindrance factors of a sphere in a cylindrical pore, λ = r_ion/r_pore. */
export function hindrance(lam) {
  const l = Math.min(Math.max(lam, 0), 0.98), phi = (1 - l) ** 2;
  return { phi, Kc: (2 - phi) * (1 + 0.054 * l - 0.988 * l * l + 0.441 * l ** 3), Kd: 1 - 2.3 * l + 1.154 * l * l + 0.224 * l ** 3 };
}
/**
 * Donnan–steric pore model with dielectric (Born) exclusion: extended Nernst–Planck transport of every ion through a charged pore.
 * species: [{ id, z, c (mol/m³ at the feed-side membrane wall), r (m), D (m²/s at 25 °C) }]; Jv in m/s;
 * m = { rp (m), dx (effective thickness δ·τ/ε, m), X (mol/m³), epsP, de (bool), T (°C), profile (bool) }.
 * Steric + Born + Donnan partitioning at both faces, electroneutrality in the pore and in the permeate (zero current);
 * the pore profile is integrated with RK4 and the permeate composition found by damped Newton iteration with flux continuation.
 */
export function dspmSolve(species, Jv, m) {
  const TK = m.T + 273.15, fD = (TK / 298.15) * (MU25 / viscosity(m.T, 0)), born0 = ((QE * QE) / (8 * Math.PI * EPS0 * KB * TK)) * (1 / m.epsP - 1 / EPS_W);
  const S = species.filter((s) => s.c > 0).map((s) => {
    const h = hindrance(s.r / m.rp), born = m.de && s.z ? Math.exp((-s.z * s.z * born0) / s.r) : 1, Dp = h.Kd * s.D * fD;
    return { id: s.id, z: s.z, c: s.c, r: s.r, D: s.D, lam: s.r / m.rp, phi: h.phi, Kc: h.Kc, Kd: h.Kd, born, part: h.phi * born, Dp, Pe: (h.Kc * Jv * m.dx) / Dp, major: false, T: 1, c0: 0 };
  });
  const T = {};
  for (const s of S) { const a = s.part * s.Kc; s.T = T[s.id] = s.Pe > 1e-12 ? Math.min(1, a / (1 - (1 - a) * Math.exp(-s.Pe))) : 1; s.c0 = s.c * s.part; }
  const all = S.filter((s) => s.z !== 0), out = { T, ions: S, converged: true, xi0: 0, xip: 0, residual: 0, profile: null, ids: [], u: null };
  if (all.length < 2 || !all.some((s) => s.z > 0) || !all.some((s) => s.z < 0) || !(Jv > 0)) { if (!(Jv > 0)) for (const s of S) s.T = T[s.id] = 1; return out; }
  // ions carrying at least 1 % of the charge of their sign are solved together; the others move as tracers in the resulting potential field
  for (const sgn of [1, -1]) {
    const grp = all.filter((s) => s.z * sgn > 0), tot = sum(grp.map((s) => Math.abs(s.z) * s.c)), top = grp.reduce((q, s) => (Math.abs(s.z) * s.c > Math.abs(q.z) * q.c ? s : q));
    for (const s of grp) s.major = s === top || Math.abs(s.z) * s.c >= 0.01 * tot;
  }
  const ch = all.filter((s) => s.major), n = ch.length;
  const z = ch.map((s) => s.z), part = ch.map((s) => s.part), kc = ch.map((s) => s.Kc), pe = ch.map((s) => s.Pe), cm = ch.map((s) => s.c);
  const donnan = (a) => brent((xi) => { let g = m.X; for (let i = 0; i < n; i++) g += z[i] * a[i] * Math.exp(-z[i] * xi); return g; }, -60, 60, 1e-13);
  const xi0 = donnan(cm.map((c, i) => c * part[i])), c0 = cm.map((c, i) => c * part[i] * Math.exp(-z[i] * xi0));
  ch.forEach((s, i) => { s.c0 = c0[i]; });
  const peMax = Math.max(...all.map((s) => s.Pe)), g = new Float64Array(n);
  const rhs = (c, cpv, fr, o) => {
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { g[i] = fr * pe[i] * (c[i] - cpv[i] / kc[i]); num += z[i] * g[i]; den += z[i] * z[i] * c[i]; }
    const E = num / (Math.abs(den) > 1e-300 ? den : 1e-300);
    for (let i = 0; i < n; i++) o[i] = g[i] - z[i] * c[i] * E;
    return E;
  };
  const k1 = new Float64Array(n), k2 = new Float64Array(n), k3 = new Float64Array(n), k4 = new Float64Array(n), tmp = new Float64Array(n);
  const steps = (fr) => 2 * clamp(Math.ceil(0.75 * fr * peMax), 8, 60);
  const integrate = (cpv, fr, keep) => {
    const N = steps(fr), h = 1 / N, c = Float64Array.from(c0);
    let psi = 0;
    if (keep) { keep.y = [0]; keep.c = [Array.from(c)]; keep.psi = [0]; keep.En = []; keep.Em = []; }
    for (let s = 0; s < N; s++) {
      const E1 = rhs(c, cpv, fr, k1);
      for (let i = 0; i < n; i++) tmp[i] = c[i] + 0.5 * h * k1[i];
      const E2 = rhs(tmp, cpv, fr, k2);
      for (let i = 0; i < n; i++) tmp[i] = c[i] + 0.5 * h * k2[i];
      const E3 = rhs(tmp, cpv, fr, k3);
      for (let i = 0; i < n; i++) tmp[i] = c[i] + h * k3[i];
      const E4 = rhs(tmp, cpv, fr, k4);
      for (let i = 0; i < n; i++) c[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
      psi += (h / 6) * (E1 + 2 * E2 + 2 * E3 + E4);
      if (keep) { keep.y.push((s + 1) * h); keep.c.push(Array.from(c)); keep.psi.push(psi); keep.En.push(E1); keep.Em.push(0.5 * (E2 + E3)); if (s === N - 1) keep.En.push(E4); }
    }
    return c;
  };
  const resid = (fr) => (u) => {
    const cpv = new Array(n), res = new Array(n + 1);
    let sz = 0, sa = 0;
    for (let i = 0; i < n; i++) { cpv[i] = cm[i] * Math.exp(Math.min(u[i], 30)); sz += z[i] * cpv[i]; sa += Math.abs(z[i]) * cpv[i]; }
    const ce = integrate(cpv, fr, null);
    for (let i = 0; i < n; i++) res[i] = (ce[i] - cpv[i] * part[i] * Math.exp(-z[i] * u[n])) / c0[i];
    res[n] = sz / sa;
    return res;
  };
  // Newton with flux continuation: from a previous solution for the same ions when there is one, otherwise a direct attempt from the
  // uncharged-solute transmissions (made electroneutral) and, failing that, continuation from the equilibrium state Jv → 0
  const march = (fr0, u0) => {
    let u = u0, done = fr0, step = 1 - fr0, last = null;
    for (let guard = 0; guard < 60; guard++) {
      const fr = Math.abs(1 - done) <= Math.abs(step) * (1 + 1e-12) ? 1 : done + step, s2 = newtonFail(resid(fr), u, 1e-9, 14);
      if (s2.converged) { u = s2.x; done = fr; last = s2; if (fr === 1) return s2; step *= 1.7; } else { step /= 2; if (Math.abs(step) < 1e-4) break; }
    }
    return { x: u, converged: false, residual: last ? last.residual : NaN };
  };
  let sol = null;
  const gs = m.guess;
  if (gs && gs.u && gs.Jv > 0 && gs.ids.join() === ch.map((s) => s.id).join()) sol = march(clamp(gs.Jv / Jv, 0.05, 20), gs.u);
  if (!sol || !sol.converged) {
    let cat = 0, an = 0;
    ch.forEach((s) => { const e = s.z * s.c * s.T; if (e > 0) cat += e; else an -= e; });
    const gu = ch.map((s) => Math.log(s.T * (s.z > 0 ? Math.sqrt(an / cat) : Math.sqrt(cat / an))));
    sol = newtonFail(resid(1), [...gu, donnan(ch.map((s, i) => s.c * Math.exp(gu[i]) * part[i]))], 1e-9, 14);
    if (!sol.converged) sol = march(0, [...new Array(n).fill(0), xi0]);
  }
  const ok = sol.converged;
  out.Jv = Jv;
  out.converged = ok; out.residual = Number.isFinite(sol.residual) ? sol.residual : 1; out.xi0 = xi0; out.xip = sol.x[n]; out.ids = ch.map((s) => s.id); out.u = ok ? sol.x : null;
  if (!ok) return out;
  ch.forEach((s, i) => { s.T = T[s.id] = Math.exp(Math.min(sol.x[i], 30)); });
  const keep = {};
  integrate(ch.map((s) => s.c * s.T), 1, keep);
  // tracer ions: linear transport c(1) = a·c(0) + b·cp in the potential field of the major ions, closed by the exit partitioning
  const N = steps(1), h = 1 / N;
  for (const s of all) {
    if (s.major) continue;
    let a = 1, b = 0;
    const c0t = s.c * s.part * Math.exp(-s.z * xi0), src = s.Pe / s.Kc, pe1 = s.Pe, zz = s.z;
    for (let q = 0; q < N; q++) { // RK4 of da/dy = (Pe − zE)·a and db/dy = (Pe − zE)·b − Pe/Kc
      const r0 = pe1 - zz * keep.En[q], rm = pe1 - zz * keep.Em[q], r1 = pe1 - zz * keep.En[q + 1];
      const a1 = r0 * a, b1 = r0 * b - src, a2 = rm * (a + 0.5 * h * a1), b2 = rm * (b + 0.5 * h * b1) - src, a3 = rm * (a + 0.5 * h * a2), b3 = rm * (b + 0.5 * h * b2) - src, a4 = r1 * (a + h * a3), b4 = r1 * (b + h * b3) - src;
      a += (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4); b += (h / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
    }
    const cpT = (a * c0t) / (s.part * Math.exp(-s.z * out.xip) - b);
    s.c0 = c0t; s.T = T[s.id] = cpT > 0 && Number.isFinite(cpT) ? cpT / s.c : s.T;
  }
  if (m.profile) out.profile = { y: keep.y, psi: keep.psi, ids: out.ids, c: keep.c };
  return out;
}

/** Tanks-in-series transient of the feed channel with a mixed permeate hold-up. Volumes m³, flows m³/h; perm(k, c) → { Qp, S = Qp·cp }. */
export function tanksTransient({ V, Vp, Qf, cf, c0, cp0, perm, tEnd, n = 300 }) {
  const K = V.length, flows = (y) => { let Qin = Qf, q = 0, s = 0; for (let k = 0; k < K; k++) { const p = perm(k, y[k]), qp = Math.min(p.Qp, 0.9 * Qin); q += qp; s += p.S; Qin -= qp; } return { q, s, Qc: Qin }; };
  const rhs = (_, y) => {
    const d = new Array(K + 1);
    let Qin = Qf, prev = cf, q = 0, s = 0;
    for (let k = 0; k < K; k++) { const p = perm(k, y[k]), qp = Math.min(p.Qp, 0.9 * Qin), Qout = Qin - qp; d[k] = (Qin * prev - Qout * y[k] - p.S) / V[k]; Qin = Qout; prev = y[k]; q += qp; s += p.S; }
    d[K] = (s - q * y[K]) / Vp;
    return d;
  };
  const sol = rk4(rhs, [...c0, cp0], 0, tEnd, n);
  return { t: sol.t, y: sol.y, conc: sol.y.map((y) => y[K - 1]), perm: sol.y.map((y) => y[K]), Qp: sol.y.map((y) => flows(y).q) };
}

/** One axial segment of one pressure vessel. Mutates nothing; returns new state and local values. */
function segment(Q, c, P, M, o) {
  const tdsB = tds(c), S = salinityFromTDS(tdsB, o.T), rho = density(o.T, S), mu = viscosity(o.T, S), D = diffusivityNaCl(o.T, S);
  const hs = M.spacerMil * 25.4e-6, W = M.area / (2 * EL_LEN), Ac = W * hs * POROSITY;
  const dh = (4 * POROSITY) / (2 / hs + ((1 - POROSITY) * 8) / hs), u = Q / 3600 / Ac;
  const Re = (rho * u * dh) / mu, Sc = mu / (rho * D);
  const k = Math.max(1e-7, (o.kcp * 0.065 * Re ** 0.875 * Sc ** 0.25 * D) / dh); // m/s
  const fr = 6.23 * Math.max(Re, 1) ** -0.3;
  const dP = (o.kdp * fr * (o.dL / dh) * rho * u * u) / 2 / 1e5;
  // membrane permeability with hydration state and the deposited-scale layer as a resistance in series
  const A0 = M.A * o.tcf * o.ff * (o.hydA ?? 1), A = o.mScale > 0 ? 1 / (1 / A0 + (o.muW * o.alpha * o.mScale) / 3.6e11) : A0, piB = osmoticPressureIons(c, o.T) / 1e5;
  const Pm = P - dP / 2, cL = o.cLeaf || 0, den = 1 + A * cL, bl = o.cpModel === 'bl2d', kk = o.model === 'kk', tab = o.model === 'dspm' ? o.pass : null;
  let Jw = Math.max(0, (A * (Pm - o.Pp - piB)) / den), cp = {}, piP = 0, CP = 1, kEff = k;
  const noDrive = Pm - o.Pp <= 0.02 * piB; // applied pressure far below osmotic pressure: no forward permeation
  for (let it = 0; it < 14; it++) {
    if (bl && it < 2) kEff = blK(u, hs, o.meshLen, D, Math.max(Jw, 0.5) / 3.6e6);
    CP = Math.exp(Math.min(3, Jw / 3.6e6 / kEff));
    const tw = tab ? clamp((Jw - tab.j0) / (tab.j1 - tab.j0), 0, 1) : 0;
    let cat = 0, an = 0;
    for (const id of ION_IDS) {
      const cm = c[id] * CP, Bi = o.Bi[id];
      let ci;
      if (o.model === 'sk') {
        // Spiegler–Kedem: R = σ(1−F)/(1−σF), F = exp(−Jw(1−σ)/Ps)
        const sig = DIVALENT.includes(id) ? Math.min(0.9995, M.sigma + (1 - M.sigma) * 0.6) : M.sigma;
        const Fk = Math.exp((-Jw * (1 - sig)) / Math.max(Bi, 1e-9)), Rj = Jw > 0 ? (sig * (1 - Fk)) / (1 - sig * Fk) : 0;
        ci = cm * (1 - Rj);
      } else if (kk) ci = cm > 0 ? cm * kkPassage(Jw, Bi, DIVALENT.includes(id) ? Math.min(0.9995, M.sigma + (1 - M.sigma) * 0.6) : M.sigma) : 0;
      else if (tab && tab.T0[id] !== undefined) ci = cm * ((1 - tw) * tab.T0[id] + tw * tab.T1[id]);
      else ci = Jw + Bi > 0 ? (Bi * cm) / (Jw + Bi) : cm;
      cp[id] = Math.min(ci, cm);
      const eq = (cp[id] / IONS[id].mw) * IONS[id].z;
      if (eq > 0) cat += eq; else an -= eq;
    }
    if (cat > 0 && an > 0) { // electroneutral permeate: co-transport forces equal equivalents
      const fc = Math.sqrt(an / cat), fa = Math.sqrt(cat / an);
      for (const id of ION_IDS) cp[id] *= IONS[id].z > 0 ? fc : IONS[id].z < 0 ? fa : 1;
    }
    piP = osmoticPressureIons(cp, o.T) / 1e5;
    const sig = o.model === 'sk' || kk ? M.sigma : 1;
    const Jn = noDrive ? 0 : Math.max(0, (A * (Pm - o.Pp - sig * (piB * CP - piP))) / den);
    if (Math.abs(Jn - Jw) < 1e-7 * (1 + Jw)) { Jw = Jn; break; }
    Jw = 0.5 * Jw + 0.5 * Jn;
  }
  const Qp = Math.min((Jw * o.dA) / 1000, Q * 0.95), Qn = Q - Qp, cn = {};
  for (const id of ION_IDS) { if (cp[id] * Qp > Q * c[id]) cp[id] = (Q * c[id]) / Qp; cn[id] = Math.max(0, (Q * c[id] - Qp * cp[id]) / Qn); } // a segment cannot pass more salt than it receives
  // dissipation function of irreversible thermodynamics, W/m²: Φ = Jv·(ΔP − σ·Δπ) + R·T·Σ ln(cm/cp)·(Js − (1 − σ)·c̄·Jv) ≥ 0
  let phi = 0;
  if (Jw > 0) {
    const Jv = Jw / 3.6e6, skk = o.model === 'sk' || kk;
    let s = 0;
    for (const id of ION_IDS) {
      const a = (c[id] * CP) / IONS[id].mw, b = cp[id] / IONS[id].mw;
      if (!(b > 0) || !(a > b * (1 + 1e-12))) continue;
      const L = Math.log(a / b), sg = skk ? (DIVALENT.includes(id) ? Math.min(0.9995, M.sigma + (1 - M.sigma) * 0.6) : M.sigma) : 1;
      s += L * (b - ((1 - sg) * (a - b)) / L);
    }
    phi = (Jv * Jv) / (A / 3.6e11) + Jv * RGAS * (o.T + 273.15) * s;
  }
  return { Q: Qn, c: cn, P: P - dP, Qp, cp, Jw, CP, dP, piB, piP, ndp: Pm - o.Pp - cL * Jw - (piB * CP - piP), Re, k: kEff, u, phi };
}

/** March through one stage (nV parallel vessels of nE elements). */
function stage(feed, nV, nE, M, o, pos = { x0: 0, dx: 1 }, tubeP = null) {
  let Q = feed.Q / nV, c = feed.ions, P = feed.P;
  const els = [], perm = [], so = { ...o, dA: M.area / o.nSeg, dL: EL_LEN / o.nSeg }, wall = o.model === 'dspm' ? Object.fromEntries(ION_IDS.map((k) => [k, 0])) : null;
  let jmin = Infinity, jmax = 0, diss = 0;
  for (let e = 0; e < nE; e++) {
    const Pin = P, Qin = Q, cin = tds(c);
    let qp = 0, jw = 0, cpMax = 1, ndp = 0, ph = 0, kS = 0;
    const pc = Object.fromEntries(ION_IDS.map((k) => [k, 0]));
    so.Pp = o.Pp + (tubeP ? tubeP[e] : 0);
    for (let s = 0; s < o.nSeg; s++) {
      so.mScale = o.scaleMean > 0 ? 2 * o.scaleMean * (pos.x0 + (pos.dx * (e + (s + 0.5) / o.nSeg)) / nE) : 0;
      const r = segment(Q, c, P, M, so);
      for (const id of ION_IDS) pc[id] += r.cp[id] * r.Qp;
      if (wall) for (const id of ION_IDS) wall[id] += (c[id] * r.CP) / (nE * o.nSeg);
      qp += r.Qp; jw += r.Jw / o.nSeg; ndp += r.ndp / o.nSeg; cpMax = Math.max(cpMax, r.CP); ph += r.phi * so.dA; kS += r.k / o.nSeg;
      if (r.Jw < jmin) jmin = r.Jw; if (r.Jw > jmax) jmax = r.Jw;
      Q = r.Q; c = r.c; P = r.P;
    }
    for (const id of ION_IDS) pc[id] = qp > 0 ? pc[id] / qp : 0;
    perm.push({ Q: qp * nV, ions: pc });
    diss += ph * nV;
    els.push({ el: e + 1, Qin, Qout: Q, Pin, dP: Pin - P, flux: jw, rec: qp / Qin, CP: cpMax, ndp, tdsFeed: cin, tdsPerm: tds(pc), Qp: qp, PpTube: tubeP ? tubeP[e] : 0, PpLeaf: (o.cLeaf || 0) * jw, diss: ph, kMT: kS, mScale: o.scaleMean > 0 ? 2 * o.scaleMean * (pos.x0 + (pos.dx * (e + 0.5)) / nE) : 0 });
  }
  const Qp = sum(perm.map((p) => p.Q));
  return { els, conc: { Q: Q * nV, ions: c, P }, perm: { Q: Qp, ions: Qp > 0 ? mixIons(perm.map((p) => ({ Q: p.Q, ions: p.ions }))) : cloneIons({}) }, wall, jmin, jmax, diss };
}

/** Ion transmissions of the pore model at the stage-mean wall composition for the lowest and highest segment flux of the stage. */
function dspmTable(wall, jmin, jmax, cfg, prev = null, profile = false) {
  const sp = ION_IDS.filter((id) => wall[id] > 0).map((id) => ({ id, z: IONS[id].z, c: wall[id] / IONS[id].mw, r: stokesRadius(IONS[id].D), D: IONS[id].D }));
  const j0 = Math.max(Number.isFinite(jmin) ? jmin : 0, 0.2), j1 = Math.max(jmax, j0 * 1.05), m = { ...cfg.dspm, T: cfg.T };
  const a = dspmSolve(sp, j0 / 3.6e6, { ...m, guess: prev?.lo }), b = dspmSolve(sp, j1 / 3.6e6, { ...m, profile, guess: prev?.hi || a });
  return { j0, j1, T0: a.T, T1: b.T, lo: a, hi: b, converged: a.converged && b.converged, jr0: jmin, jr1: jmax, wt: tds(wall), wall };
}
const DSPM_GUESS = []; // last converged pore solutions per stage: starting points for Newton only, never results

/** One pass at a given feed pressure (bar). */
function passAt(Pf, feed, cfg) {
  const M = cfg.M, pH = feed.pH;
  const Bi = Object.fromEntries(ION_IDS.map((id) => [id, M.B * cfg.tcf * cfg.sp * relB(id, M, pH, cfg.T)]));
  const o = { T: cfg.T, tcf: cfg.tcf, ff: cfg.ff, kcp: cfg.kcp, kdp: cfg.kdp, Pp: cfg.Pp, nSeg: cfg.nSeg, model: cfg.model, Bi,
    hydA: cfg.hydA, scaleMean: cfg.scaleMean, alpha: cfg.alpha, muW: cfg.muW, cpModel: cfg.cpModel, meshLen: cfg.meshLen, cLeaf: cfg.permSide === 'hp' ? cfg.cLeaf : 0 };
  const nSt = cfg.vessels.length, warm = (cfg._pass ||= []);
  let rec = cfg.recycle > 0 ? { Q: 0, ions: feed.ions } : null, out;
  for (let it = 0; it < (cfg.recycle > 0 ? 25 : 1); it++) {
    const blended = rec && rec.Q > 0 ? { Q: feed.Q + rec.Q, ions: mixIons([feed, rec]) } : { Q: feed.Q, ions: feed.ions };
    let cur = { ...blended, P: Pf };
    const stages = [];
    for (let s = 0; s < cfg.vessels.length; s++) {
      if (s > 0) cur = { ...cur, P: cur.P - cfg.interLoss + (cfg.boost[s] || 0) };
      const pos = { x0: s / nSt, dx: 1 / nSt }, run1 = (tp) => stage(cur, cfg.vessels[s], cfg.elements, M, o, pos, tp);
      let tp = null, dsp = null;
      o.pass = warm[s] || null;
      let r = run1(null);
      if (cfg.permSide === 'hp') { tp = tubeLoss(r.els.map((e) => e.Qp), cfg.tubeD, cfg.rhoW, cfg.muW, cfg.permPort); r = run1(tp); }
      if (cfg.model === 'dspm') { // pore model solved at the stage-mean wall composition, then the stage is repeated
        const near = (a, b) => Math.abs(a - b) <= 0.02 * Math.abs(b), w = warm[s];
        dsp = w;
        if (!(w && near(r.jmin, w.jr0) && near(r.jmax, w.jr1) && near(tds(r.wall), w.wt))) for (let k = 0, nk = w ? 1 : 3; k < nk; k++) {
          dsp = dspmTable(r.wall, r.jmin, r.jmax, cfg, warm[s] || DSPM_GUESS[s]); o.pass = warm[s] = dsp;
          if (dsp.converged) DSPM_GUESS[s] = { lo: dsp.lo, hi: dsp.hi };
          const r2 = run1(tp), ch = Math.abs(r2.perm.Q - r.perm.Q) / Math.max(r.perm.Q, 1e-9);
          r = r2;
          if (ch < 2e-3) break;
        }
      }
      stages.push({ ...r, feed: cur, nV: cfg.vessels[s], dspm: dsp });
      cur = r.conc;
    }
    const Qp = sum(stages.map((s) => s.perm.Q)), perm = { Q: Qp, ions: mixIons(stages.map((s) => s.perm)) };
    const recQ = cur.Q * cfg.recycle, prev = rec ? rec.Q : 0;
    out = { stages, perm, conc: { Q: cur.Q - recQ, ions: cur.ions, P: cur.P }, blended, recycleQ: recQ };
    if (!rec) break;
    rec = { Q: recQ, ions: cur.ions };
    if (Math.abs(recQ - prev) < 1e-6 * feed.Q) break;
  }
  out.recovery = out.perm.Q / feed.Q;
  return out;
}

function solvePass(feed, cfg) {
  if (cfg.mode === 'pressure') return { Pf: cfg.Pfeed, ...passAt(cfg.Pfeed, feed, cfg) };
  const piF = osmoticPressureIons(feed.ions, cfg.T) / 1e5, target = cfg.recovery;
  const f = (P) => passAt(P, feed, cfg).recovery - target;
  let lo = Math.max(0.5, cfg.Pp + 0.2), hi = Math.max(piF * 1.2 + 10, 12);
  let fhi = f(hi), guard = 0;
  while (fhi < 0 && guard++ < 14) { hi *= 1.4; fhi = f(hi); }
  if (fhi < 0) throw new Error('Target recovery cannot be reached with this array, even at very high pressure. Add membrane area or lower the recovery.');
  const Pf = f(lo) > 0 ? lo : brent(f, lo, hi, 1e-5);
  return { Pf, ...passAt(Pf, feed, cfg) };
}

/** Estimate permeate pH from the carbonate system: CO₂ passes freely, bicarbonate is rejected. */
function permeatePH(feedIons, pHf, permIons) {
  const co2 = molar(feedIons, 'HCO3') * 10 ** (6.35 - pHf), hco3 = Math.max(molar(permIons, 'HCO3'), 1e-9);
  return clamp(co2 > 1e-9 ? 6.35 + Math.log10(hco3 / co2) : pHf, 4.5, 10.5);
}

/** Propose an array (stages, vessels) for a target permeate flow and average flux. */
export function autoSize({ Qp, flux, area, elements, recovery }) {
  const nEl = Math.max(1, Math.ceil((Qp * 1000) / (flux * area))), nV = Math.max(1, Math.ceil(nEl / elements));
  const nSt = recovery > 0.82 ? 3 : recovery > 0.58 ? 2 : 1;
  if (nSt === 1 || nV < nSt) return [nV];
  const ratio = nSt === 2 ? [2, 1] : [4, 2, 1], tot = sum(ratio);
  const v = ratio.map((r) => Math.max(1, Math.round((nV * r) / tot)));
  v[0] += nV - sum(v);
  return v.map((x) => Math.max(1, x));
}

function config(v, passNo = 1) {
  const key = passNo === 1 ? v.membrane : v.membrane2, base = MEMBRANES[key] || MEMBRANES.swhr;
  const M = { ...base };
  if (passNo === 1) { M.A = v.A; M.B = v.B; M.area = v.area; M.spacerMil = v.spacerMil; }
  const T = v.T, age = Math.max(0, v.age);
  const ff = v.ff * (1 - v.fluxDecline / 100) ** age, sp = (1 + v.spIncrease / 100) ** age;
  // membrane water content (free-volume theory), deposited scale, permeate-side Hagen–Poiseuille losses and optional models
  const first = passNo === 1, phiW = clamp(v.phiW ?? 0.25, 0.1, 0.6), hyd = first ? hydrationFactors((v.hydration ?? 100) / 100, phiW) : { A: 1, B: 1 }, muW = viscosity(T, 0), rhoW = density(T, 0);
  const nLeaves = Math.max(1, Math.round(v.nLeaves ?? 16)), Lleaf = M.area / (2 * EL_LEN * nLeaves), hp = (v.permSpacer ?? 0.25) / 1000, model = !first && v.model === 'dspm' ? 'sd' : v.model;
  const rp = (v.rPore ?? 0.45) * 1e-9, Asi = M.A / 3.6e11;
  return { M, T, tcf: tcf(T, T >= 25 ? 2640 : 3020), ff, sp: sp * hyd.B, kcp: v.kcp, kdp: v.kdp, Pp: v.Pp, nSeg: Math.max(1, Math.round(v.nSeg)), model, elements: Math.round(v.elements), interLoss: v.interLoss,
    hydA: hyd.A, hydB: hyd.B, phiW, scaleMean: first ? Math.max(0, v.scaleMass ?? 0) / 1000 : 0, alpha: (v.scaleAlpha ?? 5) * 1e14, muW, rhoW,
    cpModel: v.cpModel ?? 'film', meshLen: (v.meshLen ?? 4) / 1000, permSide: v.permSide ?? 'uniform', permPort: v.permPort ?? 'conc', tubeD: (v.permTubeD ?? 28) / 1000, cLeaf: leafCoeff(muW, Lleaf, hp, v.kPerm ?? 4), Lleaf, hp,
    dspm: { rp, dx: (rp * rp) / (8 * MU25 * Asi), X: v.chargeX ?? -40, epsP: clamp(v.epsPore ?? 55, 2, EPS_W), de: v.dspmDE ?? true } };
}

/** Full RO simulation. Returns stage/element detail, streams and energy. */
export function simulateRO(v) {
  const feedIons = scaleIons(cloneIons(v.ions), v.salinityFactor ?? 1);
  const feed = { Q: v.Qf, ions: feedIons, pH: v.pH, T: v.T };
  const cfg = config(v, 1);
  cfg.mode = v.mode; cfg.Pfeed = v.Pfeed; cfg.recovery = v.recovery / 100; cfg.recycle = clamp(v.recycle / 100, 0, 0.9);
  let vessels = [v.v1, v.v2, v.v3].slice(0, Math.round(v.nStages)).map((x) => Math.max(1, Math.round(x)));
  if (v.design === 'auto') vessels = autoSize({ Qp: v.Qf * (v.recovery / 100), flux: v.targetFlux, area: cfg.M.area, elements: cfg.elements, recovery: v.recovery / 100 });
  cfg.vessels = vessels; cfg.boost = [0, v.boost2, v.boost3];
  const p1 = solvePass(feed, cfg);
  let product = { Q: p1.perm.Q, ions: p1.perm.ions }, p2 = null, pass2 = null;
  if (v.pass2) {
    const frac = clamp(v.pass2Frac / 100, 0.05, 1), f2 = { Q: p1.perm.Q * frac, ions: p1.perm.ions, pH: v.pass2pH, T: v.T };
    const c2 = config(v, 2);
    c2.mode = 'recovery'; c2.recovery = clamp(v.pass2Rec / 100, 0.5, 0.95); c2.recycle = 0; c2.boost = [0, 0, 0];
    c2.vessels = autoSize({ Qp: f2.Q * c2.recovery, flux: v.pass2Flux, area: c2.M.area, elements: c2.elements, recovery: c2.recovery });
    p2 = solvePass(f2, c2); pass2 = { cfg: c2, feed: f2 };
    const bypass = { Q: p1.perm.Q * (1 - frac), ions: p1.perm.ions };
    product = { Q: p2.perm.Q + bypass.Q, ions: mixIons([p2.perm, bypass]) };
  }
  const conc = p1.conc, nEl = sum(vessels) * cfg.elements, area = nEl * cfg.M.area;
  // energy
  const etaP = v.etaPump / 100, etaM = v.etaMotor / 100, Pin = v.Psuction;
  const hydraulic = (Q, dP) => (Q * dP * 1e5) / 3600 / 1000; // kW
  let hpFlow = feed.Q + p1.recycleQ, erdRecovered = 0, boosterKW = 0, hpKW;
  if (v.erd === 'px') {
    const leak = v.erdLeak / 100, QcHP = conc.Q, Qpx = QcHP * (1 - leak);
    hpFlow = feed.Q + p1.recycleQ - Qpx;
    const Ppx = Pin + (v.erdEff / 100) * (conc.P - v.Pbrine);
    boosterKW = hydraulic(Qpx, Math.max(0, p1.Pf - Ppx)) / (etaP * etaM);
    erdRecovered = hydraulic(Qpx, Ppx - Pin);
    hpKW = hydraulic(hpFlow, p1.Pf - Pin) / (etaP * etaM);
  } else if (v.erd === 'turbine') {
    erdRecovered = hydraulic(conc.Q, conc.P - v.Pbrine) * (v.erdEff / 100);
    hpKW = Math.max(0, hydraulic(hpFlow, p1.Pf - Pin) / etaP - erdRecovered) / etaM;
  } else hpKW = hydraulic(hpFlow, p1.Pf - Pin) / (etaP * etaM);
  const boostKW = sum(cfg.boost.map((b, s) => (b > 0 && p1.stages[s] ? hydraulic(p1.stages[s].feed.Q, b) / (etaP * etaM) : 0)));
  const p2KW = p2 ? hydraulic(pass2.feed.Q, p2.Pf - v.Pp) / (etaP * etaM) : 0;
  const power = hpKW + boosterKW + boostKW + p2KW, sec = power / product.Q;
  const overallRec = product.Q / feed.Q;
  const piF = osmoticPressureIons(feedIons, v.T) / 1e5;
  const minSEC = overallRec > 0 && overallRec < 1 ? ((piF * 1e5) / 3.6e6) * (-Math.log(1 - overallRec) / overallRec) : NaN; // reversible limit, kWh/m³
  return { feed, cfg, vessels, p1, p2, pass2, product, conc, nEl, area, hpKW, boosterKW, boostKW, p2KW, power, sec, erdRecovered, hpFlow, overallRec, piF, minSEC,
    permPH: permeatePH(feedIons, v.pH, product.ions) };
}

/** Start-up transient from prescribed initial concentrate-side and permeate-side concentrations (lumped salinity, tanks in series). */
export function startupTransient(r, v, steadyStart = false) {
  const cfg = r.cfg, M = cfg.M, A0 = M.A * cfg.tcf * cfg.ff * cfg.hydA, hs = M.spacerMil * 25.4e-6;
  const vF = (M.area / 2) * hs * POROSITY, vP = (M.area / 2) * cfg.hp + (Math.PI / 4) * cfg.tubeD ** 2 * EL_LEN, K = [], feedT = tds(r.p1.blended.ions);
  r.p1.stages.forEach((st) => st.els.forEach((e, i) => {
    const A = e.mScale > 0 ? 1 / (1 / A0 + (cfg.muW * cfg.alpha * e.mScale) / 3.6e11) : A0, css = i + 1 < st.els.length ? st.els[i + 1].tdsFeed : tds(st.conc.ions);
    K.push({ V: vF * st.nV, area: M.area * st.nV, J: e.flux, css, A, pi: Math.max(0, e.Pin - e.dP / 2 - cfg.Pp - e.PpTube - e.PpLeaf - e.flux / A), S: e.Qp * st.nV * e.tdsPerm });
  }));
  const perm = (k, c) => { const q = K[k]; return q.css > 0 ? { Qp: (Math.max(0, q.J + q.A * q.pi * (1 - c / q.css)) * q.area) / 1000, S: (q.S * c) / q.css } : { Qp: (q.J * q.area) / 1000, S: 0 }; };
  const Qf = r.p1.blended.Q, Vtot = sum(K.map((q) => q.V)), tau = Vtot / Qf, QpSS = r.p1.perm.Q, ss = QpSS > 0 ? sum(K.map((q) => q.S)) / QpSS : 0;
  const c0 = steadyStart ? K.map((q) => q.css) : K.map(() => Math.max(1, (feedT * (v.c0Conc ?? 100)) / 100)), cp0 = steadyStart ? ss : Math.max(0, v.c0Perm ?? 1000);
  const tr = tanksTransient({ V: K.map((q) => q.V), Vp: vP * sum(r.p1.stages.map((s) => s.nV * s.els.length)), Qf, cf: feedT, c0, cp0, perm, tEnd: 6 * tau, n: 300 });
  const t = tr.t.map((x) => x * 3600), lim = v.limTDS ?? 500;
  let iSet = 0, divert = 0;
  for (let i = 0; i < t.length; i++) if (Math.abs(tr.perm[i] - ss) > 0.05 * Math.max(ss, 1e-9)) iSet = Math.min(i + 1, t.length - 1);
  for (let i = 1; i < t.length; i++) if (tr.perm[i] > lim) divert += (0.5 * (tr.Qp[i] + tr.Qp[i - 1]) * (t[i] - t[i - 1])) / 3600;
  return { t, perm: tr.perm, conc: tr.conc, Qp: tr.Qp, tau: tau * 3600, ss, concSS: K[K.length - 1].css, tSettle: t[iSet], settled: iSet < t.length - 1, divert, c0: c0[0], cp0, y: tr.y, css: K.map((q) => q.css) };
}

/** Cleaning assessment: performance normalised to a clean reference at the same pressure, temperature and feed, with a clean-in-place estimate. */
export function cleaningAssessment(v, r) {
  const arr = { design: 'manual', nStages: r.vessels.length, v1: r.vessels[0], v2: r.vessels[1] || 1, v3: r.vessels[2] || 1, mode: 'pressure', Pfeed: r.p1.Pf, pass2: false, nSeg: Math.min(v.nSeg, 2) };
  const act = simulateRO({ ...v, ...arr }), ref = simulateRO({ ...v, ...arr, ff: Math.max(v.ff, 1), scaleMass: 0, kdp: Math.min(v.kdp, 1), hydration: 100 });
  // normalisation in the manner of ASTM D4516: flow per unit net driving pressure, salt passage at equal flux, pressure drop at equal mean flow
  const ndp = (q) => sum(q.p1.stages.map((st) => st.nV * sum(st.els.map((e) => e.ndp)))) / sum(q.p1.stages.map((st) => st.nV * st.els.length)), qm = (q) => q.feed.Q + q.p1.recycleQ - q.p1.perm.Q / 2;
  const Qa = act.p1.perm.Q, Qr = Math.max(ref.p1.perm.Q, 1e-12), nA = Math.max(ndp(act), 1e-9), nR = Math.max(ndp(ref), 1e-9);
  const sp = (q) => tds(q.p1.perm.ions) / tds(q.feed.ions), dp = (q) => q.p1.Pf - q.conc.P, spN = sp(act) * (Qa / Qr), dpN = dp(act) * (qm(ref) / qm(act)) ** 1.7;
  const npf = 100 * (1 - Qa / nA / (Qr / nR)), dpRise = 100 * (dpN / Math.max(dp(ref), 1e-12) - 1), spRise = 100 * (spN / Math.max(sp(ref), 1e-30) - 1);
  const limN = v.cipNPF ?? 10, limD = v.cipDP ?? 15, reasons = [];
  if (![npf, dpRise, spRise].every(Number.isFinite) || ref.p1.recovery > 0.92) { // the clean array would exceed its hydraulic limits at this pressure: compare permeabilities directly
    const cfg = r.cfg, A0 = cfg.M.A * cfg.tcf * cfg.hydA, Aact = 1 / (1 / (A0 * cfg.ff) + (cfg.muW * cfg.alpha * cfg.scaleMean) / 3.6e11), Aref = (A0 / cfg.hydA) * (cfg.ff / v.ff) * Math.max(v.ff, 1);
    const n2 = 100 * (1 - Aact / Aref), d2 = 100 * (v.kdp / Math.min(v.kdp, 1) - 1), why = [];
    if (n2 > limN) why.push(`membrane permeability is ${fmt(n2, 3)} % below clean (trigger ${limN} %)`);
    if (d2 > limD) why.push(`pressure-drop multiplier is ${fmt(d2, 3)} % above nominal (trigger ${limD} %)`);
    const kg = (Math.max(0, v.scaleMass ?? 0) / 1000) * r.area, Mm = cfg.M, hold0 = (Mm.area / 2) * Mm.spacerMil * 25.4e-6 * POROSITY * cfg.elements, nV0 = Math.max(...r.vessels);
    return { npf: n2, dpRise: d2, spRise: null, need: why.length > 0, reasons: why, chem: kg > 0 ? 'Acid clean (citric acid 2 % or HCl to pH 2) for carbonate scale; chelant (Na₄EDTA, pH 10–11) for sulphate scale' : 'Alkaline clean (NaOH pH 11–12), followed by an acid rinse', scaleKg: kg, hclKg: (kg * 2 * 36.461) / 100.087, citricKg: (kg * (2 / 3) * 192.12) / 100.087, edtaKg: (kg * 380.17) / 136.14,
      volume: 1.3 * nV0 * hold0 + 0.5 + 0.02 * nV0, flowPerVessel: Math.min(v.limFeed ?? 17, 9 * (Mm.spacerMil / 28)), QpAct: null, QpRef: null, ndpAct: null, ndpRef: null, dpAct: null, dpRef: null, spAct: null, spRef: null, lumped: true };
  }
  if (npf > limN) reasons.push(`normalised permeate flow is ${fmt(npf, 3)} % below clean (trigger ${limN} %)`);
  if (dpRise > limD) reasons.push(`normalised pressure drop is ${fmt(dpRise, 3)} % above clean (trigger ${limD} %)`);
  if (spRise > 10 && npf > 0.5 * limN) reasons.push(`normalised salt passage is ${fmt(spRise, 3)} % above clean (trigger 10 %)`);
  const M = r.cfg.M, scaleKg = (Math.max(0, v.scaleMass ?? 0) / 1000) * r.area, nVmax = Math.max(...r.vessels), hold = (M.area / 2) * M.spacerMil * 25.4e-6 * POROSITY * r.cfg.elements;
  const chem = scaleKg > 0 ? 'Acid clean (citric acid 2 % or HCl to pH 2) for carbonate scale; chelant (Na₄EDTA, pH 10–11) for sulphate scale' : dpRise > limD ? 'Alkaline clean (NaOH pH 11–12 with surfactant) for biofilm and particulates in the feed channel' : 'Alkaline clean (NaOH pH 11–12) for organic and colloidal fouling, followed by an acid rinse';
  return { npf, dpRise, spRise, need: reasons.length > 0, reasons, chem, scaleKg, hclKg: (scaleKg * 2 * 36.461) / 100.087, citricKg: (scaleKg * (2 / 3) * 192.12) / 100.087, edtaKg: (scaleKg * 380.17) / 136.14,
    volume: 1.3 * nVmax * hold + 0.5 + 0.02 * nVmax, flowPerVessel: Math.min(v.limFeed ?? 17, 9 * (M.spacerMil / 28)), QpAct: Qa, QpRef: Qr, ndpAct: nA, ndpRef: nR, dpAct: dpN, dpRef: dp(ref), spAct: 100 * spN, spRef: 100 * sp(ref) };
}

/**
 * Physics-informed surrogate (grey box): a closed-form lumped solution–diffusion baseline multiplied by a correction exp(θ·φ) whose
 * coefficients are fitted by least squares to runs of the element-resolved model; hold-out runs give the parity statistics.
 */
export function buildSurrogate(v, r, nTrain = 12, nTest = 4) {
  const cfg = r.cfg, M = cfg.M, P0 = r.p1.Pf, T0 = v.T, s0 = v.salinityFactor ?? 1, Qf = r.feed.Q, area = r.area, dp0 = r.p1.Pf - r.conc.P, tdsF = tds(r.feed.ions);
  const base = { ...v, design: 'manual', nStages: r.vessels.length, v1: r.vessels[0], v2: r.vessels[1] || 1, v3: r.vessels[2] || 1, mode: 'pressure', pass2: false, nSeg: 1 };
  const lo = [0.88 * P0, Math.max(3, T0 - 10), 0.9 * s0], hi = [1.15 * P0, Math.min(44, T0 + 10), 1.1 * s0];
  const mech = ([P, T, s]) => { try { const q = simulateRO({ ...base, Pfeed: P, T, salinityFactor: s }); return [q.p1.recovery, tds(q.p1.perm.ions)]; } catch { return [NaN, NaN]; } };
  const phys = ([P, T, s]) => {
    const tc = tcf(T, T >= 25 ? 2640 : 3020), A = M.A * tc * cfg.ff * cfg.hydA, piF = osmoticPressureIons(scaleIons(r.feed.ions, s / s0), T) / 1e5;
    const f = (x) => (area * A * Math.max(0, P - dp0 / 2 - cfg.Pp - (piF * -Math.log(1 - x)) / x)) / 1000 / Qf - x, rec = f(1e-6) <= 0 ? 0 : f(0.98) >= 0 ? 0.98 : brent(f, 1e-6, 0.98, 1e-10);
    const Jw = (rec * Qf * 1000) / area, Bs = M.B * tc * cfg.sp, cbar = rec > 0 ? (tdsF * (s / s0) * -Math.log(1 - rec)) / rec : tdsF;
    return [rec, (Bs * cbar) / (Jw + Bs)];
  };
  const feat = ([P, T, s]) => { const a = P / P0 - 1, b = (T - T0) / 10, c = s / s0 - 1; return [1, a, b, c, a * a, a * b]; };
  const pts = [[P0, T0, s0], ...lhs(nTrain + nTest - 1, 3, 11).map((u) => u.map((x, j) => lo[j] + x * (hi[j] - lo[j])))];
  const rows = pts.map((x) => ({ x, m: mech(x), p: phys(x) })).map((q) => ({ ...q, ok: q.m.every((y) => y > 0 && Number.isFinite(y)) && q.p.every((y) => y > 0 && Number.isFinite(y)) }));
  const train = rows.slice(0, nTrain).filter((q) => q.ok), test = rows.slice(nTrain).filter((q) => q.ok);
  if (train.length < 8 || test.length < 3) return null;
  const theta = [0, 1].map((k) => lstsq(train.map((q) => feat(q.x)), train.map((q) => Math.log(q.m[k] / q.p[k]))));
  const predict = (x) => { const p = phys(x), f = feat(x); return p.map((y, k) => (y > 0 ? y * Math.exp(sum(f.map((a, j) => a * theta[k][j]))) : 0)); };
  const stat = (k, fn) => { const y = test.map((q) => q.m[k]), yh = test.map((q) => fn(q)[k]), mu = sum(y) / y.length, ssT = sum(y.map((a) => (a - mu) ** 2)), ssR = sum(y.map((a, i) => (a - yh[i]) ** 2)); return { r2: ssT > 0 ? 1 - ssR / ssT : 1, maxErr: 100 * Math.max(...y.map((a, i) => Math.abs(yh[i] / a - 1))) }; };
  const Ps = linspace(lo[0], hi[0], 15), Ts = linspace(lo[1], hi[1], 11);
  return { theta, predict, phys, nTrain: train.length, nTest: test.length, rec: stat(0, (q) => predict(q.x)), tdsP: stat(1, (q) => predict(q.x)), recBase: stat(0, (q) => q.p), tdsBase: stat(1, (q) => q.p),
    parity: { mechRec: test.map((q) => 100 * q.m[0]), surRec: test.map((q) => 100 * predict(q.x)[0]), mechTds: test.map((q) => q.m[1]), surTds: test.map((q) => predict(q.x)[1]) },
    design: { mech: rows[0].m, sur: predict(rows[0].x) }, map: { P: Ps, T: Ts, rec: Ts.map((T) => Ps.map((P) => 100 * predict([P, T, s0])[0])), tds: Ts.map((T) => Ps.map((P) => predict([P, T, s0])[1])) } };
}

/** Lead-element feed channel: two-dimensional boundary-layer solution with its wall and symmetry conditions, compared with film theory. */
function channelDiagnostics(r, v) {
  const cfg = r.cfg, M = cfg.M, e = r.p1.stages[0].els[0], hs = M.spacerMil * 25.4e-6, S = salinityFromTDS(e.tdsFeed, v.T), D = diffusivityNaCl(v.T, S);
  const ub = e.Qin / 3600 / ((M.area / (2 * EL_LEN)) * hs * POROSITY), vw = Math.max(e.flux, 0.05) / 3.6e6, L = cfg.meshLen;
  const bl = channelBL({ ub, h: hs, L, D, vw, ny: 40, nx: 40, field: true }), nxU = 36, nyU = 28, shear = bl.shear, dMax = Math.min(bl.b, 5 * ((D * L) / shear) ** (1 / 3));
  const xU = linspace(0, L, nxU), yU = linspace(0, dMax, nyU), z = yU.map(() => new Array(nxU));
  for (let i = 0; i < nxU; i++) {
    let n = 1; while (n < bl.x.length - 1 && bl.x[n] < xU[i]) n++;
    const w = (xU[i] - bl.x[n - 1]) / (bl.x[n] - bl.x[n - 1]);
    for (let j = 0; j < nyU; j++) { const y = bl.b - yU[j], a = interp1(bl.yc, bl.rows[n - 1], y), b = interp1(bl.yc, bl.rows[n], y); z[j][i] = (1 - w) * a + w * b; }
  }
  const film = bl.x.map((x) => (x > 0 ? Math.exp(vw / levequeK(ub, hs, x, D)) : 1));
  let fm = 0; for (let i = 1; i < bl.x.length; i++) fm += 0.5 * (film[i] + film[i - 1]) * (bl.x[i] - bl.x[i - 1]);
  const last = bl.rows[bl.rows.length - 1];
  return { bl, ub, vw, D, L, hs, shear, xU, yU, z, film, filmMean: fm / L, k2d: vw / Math.log(Math.max(bl.cpMean, 1 + 1e-12)), kCorr: e.kMT, betaCorr: e.CP, centreGrad: (last[1] - last[0]) / (bl.yc[1] - bl.yc[0]) / (last[last.length - 1] / bl.b) };
}

const streamOut = (s, T, P, pH) => ({ Q: s.Q, T, P, pH, tds: tds(s.ions), ions: Object.fromEntries(ION_IDS.map((k) => [k, +s.ions[k].toPrecision(6)])) });

const fields = {
  feed: [
    { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.seawater.ions, help: 'Full ionic analysis. Load a reference water, import a laboratory sheet, or use the case feed water.' },
    { key: 'Qf', label: 'Feed flow', unit: 'm³/h', value: 1000, min: 0.1, max: 2e5, help: 'Raw feed flow entering the first pass (before any concentrate recycle).' },
    { key: 'T', label: 'Feed temperature', unit: '°C', value: 25, min: 1, max: 45, typical: [10, 38], help: 'Warmer water lowers pressure but raises salt passage.' },
    { key: 'pH', label: 'Feed pH', unit: '', value: 8.1, min: 2, max: 12, help: 'Controls boron speciation and the carbonate system.' },
    { key: 'salinityFactor', label: 'Salinity multiplier', unit: '×', value: 1, min: 0.05, max: 3, help: 'Scales the whole analysis — convenient for seasonal salinity cases and sensitivity runs.' },
  ],
};

const suite = {
  id: 'ro', num: 1, title: 'Reverse Osmosis & Membrane Design', short: 'RO design', icon: '💧',
  tagline: 'Element-by-element design and rating of RO/NF arrays with full permeate quality, hydraulics and energy.',
  description: 'Solves every membrane element of every stage in axial segments. Water and solute transport follow the solution–diffusion, Spiegler–Kedem or Kedem–Katchalsky model, or an ion-by-ion Donnan–steric pore model for nanofiltration; film theory with a spacer Sherwood correlation gives concentration polarisation; a spacer friction correlation gives pressure loss. Each ion is tracked individually with an electroneutral permeate, so boron, nitrate and divalent rejection, concentration factors and design-limit violations are reported element by element.',
  guide: [
    'Enter the feed analysis, flow and temperature (or pull them from the Case page).',
    'Pick an element class and either let the tool size the array or enter your own stages and vessels.',
    'Choose whether you fix the recovery (pressure is solved) or fix the pressure (recovery is solved).',
    'Run. Check the design-limit warnings, the element profile and the permeate quality; the concentrate is automatically offered to the chemistry, ZLD and sea-discharge suites.',
  ],
  implemented: ['solution-diffusion', 'spiegler-kedem', 'kedem-katchalsky', 'vant hoff', 'extended vant hoff', 'water-flux', 'solute-flux', 'salt-rejection', 'observed-rejection', 'intrinsic-rejection', 'recovery equation', 'concentration-factor', 'concentration-polarization', 'sherwood', 'reynolds', 'schmidt', 'mass-transfer-coefficient', 'pressure-drop', 'darcy-weisbach', 'friction-factor', 'mass balance', 'component material', 'electroneutrality', 'temperature-correction', 'darcy-type', 'activity-coefficient',
    'film theory', 'fouling resistance', 'resistance-in-series', 'osmotic-pressure coupling', 'spacer hydrodynamic', 'energy-recovery coupling', 'mechanistic-empirical', 'scale-formation coupling', 'antiscalant assessment',
    'initial feed concentration', 'initial pressure', 'temperature', 'membrane resistance', 'fouling resistance', 'prescribed inlet', 'outlet-pressure', 'membrane-interface flux', 'solute-partition', 'interface condition', 'permeate-side',
    'feed-water characterisation', 'membrane and element selection', 'array configuration', 'membrane transport', 'salt rejection', 'permeate-quality', 'recovery calculation', 'concentration-polarisation', 'pressure-drop calculation', 'osmotic-pressure assessment', 'hydraulic balancing', 'staging and recirculation', 'energy-consumption', 'membrane ageing', 'sensitivity analysis', 'design optimisation',
    'irreversible-thermodynamics', 'hagen-poiseuille', 'donnan-steric pore model', 'extended nernst-planck', 'physics-informed/data-driven membrane', 'initial permeate and concentrate concentration', 'membrane water content', 'initial deposited-scale mass', 'no-slip wall', 'symmetry condition', 'cleaning assessment'],
  equationsNote: 'Valid for spiral-wound RO/NF elements at 1–45 °C and up to about 120 g/kg bulk salinity. Stage-by-stage scaling indices come from the electrolyte model of suite 2; dosing design is done there and optimisation in suite 11. Reduced-order forms used: the Donnan–steric pore model (with dielectric exclusion, extended Nernst–Planck) is one-dimensional across the active layer, uses ideal activities in the pore and is solved per stage at the mean wall composition for the lowest and highest flux, then interpolated in flux for each segment; the no-slip and symmetry conditions belong to a two-dimensional laminar open-slit boundary-layer solution restarted at each spacer filament (not a resolved spacer geometry — use the CFD suite for that); permeate-side losses are slit and tube Hagen–Poiseuille flow with a spacer resistance factor; the start-up transient is a lumped-salinity tanks-in-series model; membrane water content acts through a free-volume relation and deposited scale through a series resistance, both as prescribed initial states (their growth in time is modelled in suite 10); the physics-informed surrogate is a least-squares correction of a closed-form solution–diffusion baseline, valid only inside its training window.',

  inputs: [
    { group: 'Feed water', help: 'What enters the membrane system.', fields: fields.feed },
    { group: 'Membrane element', help: 'Element class and its transport properties. Selecting a class fills in typical values, which you can then override or calibrate.', fields: [
      { key: 'membrane', label: 'Element class', type: 'select', value: 'swhr', options: Object.entries(MEMBRANES).map(([k, m]) => ({ value: k, label: m.name })), help: 'Generic 8-inch element classes. After changing the class press “Apply class properties”.' },
      { key: 'A', label: 'Water permeability A (25 °C)', unit: 'L/m²·h·bar', value: 1.25, min: 0.05, max: 30, help: 'Pure-water permeability of the active layer.' },
      { key: 'B', label: 'Salt permeability B, NaCl (25 °C)', unit: 'L/m²·h', value: 0.058, min: 0.001, max: 100, help: 'Solute permeability for NaCl; other ions are scaled from it.' },
      { key: 'area', label: 'Active area per element', unit: 'm²', value: 37.2, min: 1, max: 60 },
      { key: 'spacerMil', label: 'Feed-spacer thickness', unit: 'mil', value: 28, min: 17, max: 65, help: 'Thicker spacers lower pressure drop and fouling tendency but reduce packing density.' },
      { key: 'elements', label: 'Elements per pressure vessel', unit: '', value: 7, min: 1, max: 8, step: 1 },
    ] },
    { group: 'Array and operation', fields: [
      { key: 'design', label: 'Array definition', type: 'select', value: 'auto', options: [{ value: 'auto', label: 'Size the array for me (from target flux)' }, { value: 'manual', label: 'I will enter stages and vessels' }] },
      { key: 'targetFlux', label: 'Target average flux', unit: 'L/m²·h', value: 14, min: 3, max: 45, typical: [11, 30], help: 'Seawater open intake 12–15, beach well 14–18, brackish well 23–30.', showIf: (v) => v.design === 'auto' },
      { key: 'nStages', label: 'Number of stages', unit: '', value: 1, min: 1, max: 3, step: 1, showIf: (v) => v.design === 'manual' },
      { key: 'v1', label: 'Vessels in stage 1', unit: '', value: 130, min: 1, max: 5000, step: 1, showIf: (v) => v.design === 'manual' },
      { key: 'v2', label: 'Vessels in stage 2', unit: '', value: 60, min: 1, max: 5000, step: 1, showIf: (v) => v.design === 'manual' && v.nStages >= 2 },
      { key: 'v3', label: 'Vessels in stage 3', unit: '', value: 30, min: 1, max: 5000, step: 1, showIf: (v) => v.design === 'manual' && v.nStages >= 3 },
      { key: 'mode', label: 'Operating specification', type: 'select', value: 'recovery', options: [{ value: 'recovery', label: 'Fix recovery → solve feed pressure' }, { value: 'pressure', label: 'Fix feed pressure → solve recovery' }] },
      { key: 'recovery', label: 'Target recovery', unit: '%', value: 45, min: 5, max: 95, typical: [35, 85], showIf: (v) => v.mode === 'recovery' || v.design === 'auto' },
      { key: 'Pfeed', label: 'Feed pressure', unit: 'bar', value: 60, min: 1, max: 120, showIf: (v) => v.mode === 'pressure' },
      { key: 'Pp', label: 'Permeate back-pressure', unit: 'bar', value: 1, min: 0, max: 20 },
      { key: 'boost2', label: 'Inter-stage boost before stage 2', unit: 'bar', value: 0, min: 0, max: 40 },
      { key: 'boost3', label: 'Inter-stage boost before stage 3', unit: 'bar', value: 0, min: 0, max: 40 },
      { key: 'recycle', label: 'Concentrate recycle to feed', unit: '%', value: 0, min: 0, max: 80, help: 'Share of the final concentrate returned to the feed to keep cross-flow at high recovery.' },
    ] },
    { group: 'Second pass (optional)', fields: [
      { key: 'pass2', label: 'Add a permeate second pass', type: 'bool', value: false, help: 'For tight boron, chloride or TDS limits.' },
      { key: 'membrane2', label: 'Second-pass element class', type: 'select', value: 'bwhr', options: Object.entries(MEMBRANES).map(([k, m]) => ({ value: k, label: m.name })), showIf: (v) => v.pass2 },
      { key: 'pass2Frac', label: 'Share of first-pass permeate treated', unit: '%', value: 100, min: 5, max: 100, showIf: (v) => v.pass2 },
      { key: 'pass2Rec', label: 'Second-pass recovery', unit: '%', value: 90, min: 50, max: 95, showIf: (v) => v.pass2 },
      { key: 'pass2Flux', label: 'Second-pass average flux', unit: 'L/m²·h', value: 32, min: 10, max: 45, showIf: (v) => v.pass2 },
      { key: 'pass2pH', label: 'Second-pass feed pH (caustic dosing)', unit: '', value: 10, min: 5, max: 11, help: 'Raising pH converts boric acid to borate, which is rejected far better.', showIf: (v) => v.pass2 },
    ] },
    { group: 'Pumps and energy recovery', fields: [
      { key: 'erd', label: 'Energy-recovery device', type: 'select', value: 'px', options: [{ value: 'px', label: 'Isobaric pressure exchanger' }, { value: 'turbine', label: 'Pelton turbine / turbocharger' }, { value: 'none', label: 'None (throttle valve)' }] },
      { key: 'erdEff', label: 'Energy-recovery efficiency', unit: '%', value: 96, min: 30, max: 99, showIf: (v) => v.erd !== 'none' },
      { key: 'erdLeak', label: 'Pressure-exchanger lubrication leakage', unit: '%', value: 1.5, min: 0, max: 8, showIf: (v) => v.erd === 'px' },
      { key: 'etaPump', label: 'High-pressure pump efficiency', unit: '%', value: 86, min: 30, max: 93 },
      { key: 'etaMotor', label: 'Motor + drive efficiency', unit: '%', value: 95, min: 60, max: 99 },
      { key: 'Psuction', label: 'Pump suction pressure', unit: 'bar', value: 2.5, min: 0, max: 20 },
      { key: 'Pbrine', label: 'Brine discharge pressure', unit: 'bar', value: 1.5, min: 0, max: 20 },
    ] },
    { group: 'Transport model', tab: 'setup', help: 'How water and solutes cross the membrane.', fields: [
      { key: 'model', label: 'Membrane transport model', type: 'select', value: 'sd', options: [{ value: 'sd', label: 'Solution–diffusion + film theory' }, { value: 'sk', label: 'Spiegler–Kedem (reflection coefficient) + film theory' }, { value: 'kk', label: 'Kedem–Katchalsky (irreversible thermodynamics: Lp, σ, ω)' }, { value: 'dspm', label: 'Donnan–steric pore model, extended Nernst–Planck (nanofiltration)' }],
        help: 'Spiegler–Kedem adds solvent–solute coupling and is preferred for nanofiltration and loose RO. Kedem–Katchalsky is the phenomenological form Jv = Lp(ΔP − σΔπ), Js = ωΔπ + (1 − σ)c̄Jv with the logarithmic-mean concentration. The pore model carries every ion through a charged pore (steric, Donnan and dielectric partitioning, hindered diffusion and convection, electromigration); it is solved per stage and is slower.' },
      { key: 'cpModel', label: 'Concentration-polarisation model', type: 'select', value: 'film', options: [{ value: 'film', label: 'Film theory with the spacer Sherwood correlation' }, { value: 'bl2d', label: '2-D boundary layer (no-slip walls, symmetry plane)' }],
        help: 'The 2-D option solves convection–diffusion across the channel half-height in every segment: parabolic no-slip velocity, zero gradient at the mid-plane, membrane flux condition at the wall, with the boundary layer restarting at each spacer filament. It is slower and does not use the mass-transfer multiplier.' },
      { key: 'meshLen', label: 'Boundary-layer re-development (spacer mesh) length', unit: 'mm', value: 4, min: 1, max: 1000, help: 'Distance over which the concentration boundary layer grows before a spacer filament mixes it. Use the element length (1016 mm) for an open, spacer-free channel.' },
      { key: 'permSide', label: 'Permeate-side pressure', type: 'select', value: 'uniform', options: [{ value: 'uniform', label: 'Uniform back-pressure (losses reported only)' }, { value: 'hp', label: 'Hagen–Poiseuille losses in leaf and central tube (coupled)' }],
        help: 'Coupled: laminar slit flow in the permeate leaf and tube friction along the vessel raise the local permeate pressure, so the back-pressure varies from element to element.' },
      { key: 'permPort', label: 'Permeate outlet end of the vessel', type: 'select', value: 'conc', options: [{ value: 'conc', label: 'Concentrate end' }, { value: 'feed', label: 'Feed end' }], showIf: (v) => v.permSide === 'hp' },
      { key: 'surrogate', label: 'Train the grey-box surrogate and draw the operating map', type: 'bool', value: true, help: 'Fits a physics-informed correction to a closed-form solution–diffusion baseline using a few runs of the element model, checks it on hold-out runs and maps recovery against pressure and temperature.' },
      { key: 'kcp', label: 'Mass-transfer multiplier', unit: '×', value: 1, min: 0.2, max: 5, help: 'Scales the spacer Sherwood correlation Sh = 0.065 Re^0.875 Sc^0.25. Calibrate it or take it from the CFD suite.' },
      { key: 'kdp', label: 'Pressure-drop multiplier', unit: '×', value: 1, min: 0.2, max: 10, help: 'Scales the spacer friction factor f = 6.23 Re^−0.3. Rises as elements foul.' },
      { key: 'interLoss', label: 'Inter-stage piping loss', unit: 'bar', value: 0.3, min: 0, max: 5 },
    ] },
    { group: 'Pore model (Donnan–steric, dielectric exclusion)', tab: 'setup', help: 'Parameters of the charged-pore model. The effective layer thickness δ·τ/ε follows from the water permeability A by the Hagen–Poiseuille relation.', showIf: (v) => v.model === 'dspm', fields: [
      { key: 'rPore', label: 'Pore radius', unit: 'nm', value: 0.45, min: 0.2, max: 2, help: 'Nanofiltration membranes: about 0.35–0.6 nm. Ion radii are Stokes radii from their diffusivities.' },
      { key: 'chargeX', label: 'Membrane charge density', unit: 'mol/m³', value: -40, min: -500, max: 500, help: 'Fixed charge per pore volume; negative for polyamide membranes at neutral pH.' },
      { key: 'dspmDE', label: 'Dielectric (Born) exclusion', type: 'bool', value: true, help: 'Solvation-energy barrier caused by the lower dielectric constant of water confined in the pore.' },
      { key: 'epsPore', label: 'Pore dielectric constant', unit: '–', value: 55, min: 30, max: 78.4, showIf: (v) => v.dspmDE, help: 'Bulk water is 78.4; confined water is typically 35–65.' },
    ] },
    { group: 'Membrane structure and permeate channel', tab: 'setup', help: 'Used for the pore-flow and solution–diffusion structure relations, the hydration state and the permeate-side Hagen–Poiseuille losses.', fields: [
      { key: 'poreEps', label: 'Active-layer porosity (free-volume fraction)', unit: '–', value: 0.05, min: 0.005, max: 0.5, help: 'For the equivalent pore radius r = √(8·μ·τ·δ·A/ε).' },
      { key: 'poreTau', label: 'Pore tortuosity', unit: '–', value: 2.5, min: 1, max: 10 },
      { key: 'activeThk', label: 'Active-layer thickness', unit: 'nm', value: 150, min: 10, max: 2000 },
      { key: 'phiW', label: 'Equilibrium water volume fraction of the active layer', unit: '–', value: 0.25, min: 0.1, max: 0.6, help: 'Water sorbed by the fully hydrated polymer; gives the water diffusivity from A = D·φ·V̄/(R·T·δ).' },
      { key: 'nLeaves', label: 'Membrane leaves per element', unit: '', value: 16, min: 4, max: 60, step: 1, help: 'Sets the permeate path length along each leaf: area ÷ (2 × element length × leaves).' },
      { key: 'permSpacer', label: 'Permeate-channel height', unit: 'mm', value: 0.25, min: 0.15, max: 1 },
      { key: 'kPerm', label: 'Permeate-spacer resistance factor', unit: '×', value: 4, min: 1, max: 20, help: 'Pressure loss of the tricot-filled channel relative to an open slit of the same height (Hagen–Poiseuille).' },
      { key: 'permTubeD', label: 'Central permeate-tube inner diameter', unit: 'mm', value: 28, min: 10, max: 60 },
    ] },
    { group: 'Initial membrane state', tab: 'setup', help: 'Initial conditions: condition of the membranes at the time simulated.', fields: [
      { key: 'hydration', label: 'Membrane water content', unit: '% of full hydration', value: 100, min: 70, max: 100, help: 'Water sorbed in the active layer relative to a fully wetted membrane. Dried-out or compacted membranes lose water and salt permeability (free-volume theory: P ∝ φ·exp(−b/φ), b = 0.45 for water and 1.2 for salt).' },
      { key: 'scaleMass', label: 'Initial deposited-scale mass (array mean)', unit: 'g/m²', value: 0, min: 0, max: 500, help: 'Mineral scale already on the membranes. It is distributed linearly from zero at the feed inlet to twice the mean at the concentrate outlet and acts as a hydraulic resistance in series.' },
      { key: 'scaleAlpha', label: 'Specific resistance of the scale layer', unit: '10¹⁴ m/kg', value: 5, min: 0.01, max: 100, showIf: (v) => v.scaleMass > 0, help: 'Layer resistance = specific resistance × deposited mass per area.' },
      { key: 'c0Conc', label: 'Initial concentrate-side salinity', unit: '% of feed TDS', value: 100, min: 0, max: 300, help: 'Liquid in the feed channels when pressure is applied: 100 % = filled with feed water, about 1 % after a permeate flush.' },
      { key: 'c0Perm', label: 'Initial permeate-side TDS', unit: 'mg/L', value: 1000, min: 0, max: 50000, help: 'Liquid in the permeate channels and tubes at start-up; salt diffuses across the membrane during standstill.' },
      { key: 'ff', label: 'Flow factor (fouling allowance)', unit: '–', value: 0.95, min: 0.3, max: 1.2, help: '1.0 = new, clean membrane. 0.85 is a common three-year design allowance.' },
      { key: 'age', label: 'Membrane age', unit: 'years', value: 0, min: 0, max: 15 },
      { key: 'fluxDecline', label: 'Permeability decline per year', unit: '%/y', value: 7, min: 0, max: 30 },
      { key: 'spIncrease', label: 'Salt-passage increase per year', unit: '%/y', value: 10, min: 0, max: 50 },
    ] },
    { group: 'Design limits', tab: 'setup', help: 'Boundaries checked for every element. Defaults follow common manufacturer guidelines.', fields: [
      { key: 'limFlux', label: 'Maximum lead-element flux', unit: 'L/m²·h', value: 34, min: 5, max: 60 },
      { key: 'limRec', label: 'Maximum element recovery', unit: '%', value: 15, min: 5, max: 30 },
      { key: 'limCP', label: 'Maximum polarisation factor β', unit: '–', value: 1.2, min: 1.05, max: 2 },
      { key: 'limFeed', label: 'Maximum feed flow per vessel', unit: 'm³/h', value: 17, min: 2, max: 40 },
      { key: 'limConc', label: 'Minimum concentrate flow per vessel', unit: 'm³/h', value: 3, min: 0.3, max: 10 },
      { key: 'limDP', label: 'Maximum pressure drop per vessel', unit: 'bar', value: 3.5, min: 0.5, max: 6 },
      { key: 'limTDS', label: 'Product TDS limit', unit: 'mg/L', value: 500, min: 1, max: 5000 },
      { key: 'limBoron', label: 'Product boron limit', unit: 'mg/L', value: 2.4, min: 0.1, max: 10 },
      { key: 'cipNPF', label: 'Cleaning trigger: normalised permeate-flow decline', unit: '%', value: 10, min: 2, max: 30 },
      { key: 'cipDP', label: 'Cleaning trigger: normalised pressure-drop rise', unit: '%', value: 15, min: 5, max: 60 },
    ] },
    { group: 'Axial discretisation', tab: 'mesh', help: 'Each element is divided into equal segments along the flow direction.', fields: [
      { key: 'nSeg', label: 'Segments per element', unit: '', value: 4, min: 1, max: 40, step: 1, help: 'More segments resolve the concentration and flux profile better; use the sensitivity study below to see the effect.' },
    ] },
  ],

  presets: [
    { name: 'Seawater, 45 % recovery, pressure exchanger', values: {} },
    { name: 'Arabian Gulf seawater, 40 % recovery, 32 °C', values: { ions: WATERS.gulf.ions, T: 32, pH: 8.2, recovery: 40, targetFlux: 13 } },
    { name: 'Brackish well, 2 stages, 78 % recovery', values: { ions: WATERS.brackish.ions, Qf: 250, T: 24, pH: 7.6, membrane: 'bwhr', A: 3.3, B: 0.19, spacerMil: 34, recovery: 78, targetFlux: 26, erd: 'none', elements: 6, limFlux: 41, limRec: 19, limTDS: 300, limBoron: 2.4 } },
    { name: 'Nanofiltration softening, 80 % recovery', values: { ions: WATERS.lowbrackish.ions, Qf: 300, T: 18, pH: 7.8, membrane: 'nf', A: 8.5, B: 28, spacerMil: 34, model: 'sk', recovery: 80, targetFlux: 27, erd: 'none', elements: 6, limFlux: 44, limRec: 19 } },
    { name: 'Seawater two-pass for boron < 0.5 mg/L', values: { pass2: true, pass2Frac: 70, limBoron: 0.5, limTDS: 200 } },
    { name: 'Nanofiltration, Donnan–steric pore model (ion by ion)', values: { ions: WATERS.lowbrackish.ions, Qf: 300, T: 18, pH: 7.8, membrane: 'nf', A: 8.5, B: 28, spacerMil: 34, model: 'dspm', recovery: 80, targetFlux: 27, erd: 'none', elements: 6, limFlux: 44, limRec: 19 } },
    { name: 'Scaled and aged seawater array (cleaning assessment)', values: { age: 3, ff: 0.88, scaleMass: 60, kdp: 1.35, permSide: 'hp', c0Conc: 1, c0Perm: 3000 } },
  ],

  pull: ({ feed, outputs }) => [
    { key: 'ions', value: feed.ions, from: 'Case feed water' }, { key: 'Qf', value: feed.Q, from: 'Case feed water' },
    { key: 'T', value: feed.T, from: 'Case feed water' }, { key: 'pH', value: feed.pH, from: 'Case feed water' },
    outputs.cfd?.kMultiplier ? { key: 'kcp', value: outputs.cfd.kMultiplier, from: 'CFD mass-transfer result' } : null,
    outputs.fouling?.normPermeability ? { key: 'ff', value: clamp(outputs.fouling.normPermeability, 0.4, 1.1), from: 'Fouling monitor: normalised permeability' } : null,
    outputs.opt?.best?.recovery ? { key: 'recovery', value: outputs.opt.best.recovery, from: 'Optimiser recommendation' } : null,
  ],
  site: () => [],

  run(v) {
    const r = simulateRO(v), W = [], limits = v;
    const els = r.p1.stages.flatMap((s, i) => s.els.map((e) => ({ ...e, stage: i + 1 })));
    const lead = els[0], maxFlux = Math.max(...els.map((e) => e.flux)), maxRec = Math.max(...els.map((e) => e.rec)), maxCP = Math.max(...els.map((e) => e.CP));
    const avgFlux = (r.p1.perm.Q * 1000) / r.area;
    r.p1.stages.forEach((s, i) => {
      const qf = s.feed.Q / s.nV, qc = s.conc.Q / s.nV, dp = sum(s.els.map((e) => e.dP));
      if (qf > limits.limFeed) W.push({ level: 'bad', msg: `Stage ${i + 1}: feed flow ${fmt(qf, 3)} m³/h per vessel exceeds the ${limits.limFeed} m³/h limit — add vessels.` });
      if (qc < limits.limConc) W.push({ level: 'bad', msg: `Stage ${i + 1}: concentrate flow ${fmt(qc, 3)} m³/h per vessel is below the ${limits.limConc} m³/h minimum — fewer vessels in this stage or add recycle.` });
      if (dp > limits.limDP) W.push({ level: 'bad', msg: `Stage ${i + 1}: pressure drop ${fmt(dp, 3)} bar per vessel exceeds ${limits.limDP} bar.` });
    });
    if (maxFlux > limits.limFlux) W.push({ level: 'bad', msg: `Highest element flux ${fmt(maxFlux, 3)} L/m²·h exceeds the ${limits.limFlux} limit — raise permeate back-pressure on stage 1, add an inter-stage boost, or add area.` });
    if (maxRec * 100 > limits.limRec) W.push({ level: 'warn', msg: `Highest element recovery ${fmt(maxRec * 100, 3)} % exceeds ${limits.limRec} %.` });
    if (maxCP > limits.limCP) W.push({ level: 'warn', msg: `Polarisation factor β reaches ${fmt(maxCP, 3)} (limit ${limits.limCP}) — scaling and salt passage rise at the membrane wall.` });
    if (r.p1.Pf > r.cfg.M.pmax) W.push({ level: 'bad', msg: `Feed pressure ${fmt(r.p1.Pf, 3)} bar exceeds the element rating of ${r.cfg.M.pmax} bar.` });
    const pT = tds(r.product.ions);
    if (pT > limits.limTDS) W.push({ level: 'bad', msg: `Product TDS ${fmt(pT, 3)} mg/L exceeds the ${limits.limTDS} mg/L limit.` });
    if (r.product.ions.B > limits.limBoron) W.push({ level: 'bad', msg: `Product boron ${fmt(r.product.ions.B, 3)} mg/L exceeds ${limits.limBoron} mg/L — add a second pass or raise its pH.` });
    const cb = chargeBalance(r.feed.ions);
    if (Math.abs(cb.errorPct) > 5) W.push({ level: 'warn', msg: `Feed analysis charge imbalance is ${fmt(cb.errorPct, 2)} % — check the laboratory analysis.` });
    if (v.design === 'auto') W.push({ level: 'info', msg: `Array sized automatically: ${r.vessels.join(' : ')} vessels × ${r.cfg.elements} elements (${r.nEl} elements, ${fmt(r.area, 4)} m²).` });
    if (!W.some((w) => w.level === 'bad')) W.unshift({ level: 'info', msg: 'All element-level design limits are satisfied.' });

    // Scaling tendency of each stage's concentrate (bulk and at the membrane wall) from the electrolyte model of suite 2.
    const SCALES = [['calcite', 'Calcite CaCO₃'], ['gypsum', 'Gypsum CaSO₄·2H₂O'], ['barite', 'Barite BaSO₄'], ['celestite', 'Celestite SrSO₄'], ['silica', 'Amorphous silica'], ['fluorite', 'Fluorite CaF₂']];
    const scaleRows = [];
    try {
      r.p1.stages.forEach((st, i) => {
        const cfS = tds(st.conc.ions) / tds(r.feed.ions), pHs = Math.min(9, v.pH + 0.3 * Math.log10(Math.max(cfS, 1))), beta = Math.max(...st.els.map((e) => e.CP));
        const bulk = analyzeWater({ ions: st.conc.ions, T: v.T, pH: pHs }), wall = analyzeWater({ ions: scaleIons(st.conc.ions, beta), T: v.T, pH: pHs });
        const si = (a, k) => (Number.isFinite(a.SI?.[k]) && a.SI[k] > -50 ? a.SI[k] : null);
        scaleRows.push([`Stage ${i + 1} concentrate`, tds(st.conc.ions), pHs, ...SCALES.flatMap(([k]) => [si(bulk, k), si(wall, k)])]);
        const worst = SCALES.map(([k, n]) => [n, si(wall, k)]).filter(([n, x]) => x !== null && x > 0 && !n.startsWith('Calcite')).sort((a, b) => b[1] - a[1])[0];
        if (worst) W.push({ level: worst[1] > 0.6 ? 'bad' : 'warn', msg: `Stage ${i + 1}: ${worst[0]} is supersaturated at the membrane wall (saturation index ${fmt(worst[1], 2)}) — antiscalant is required${worst[1] > 0.6 ? ' and may not be sufficient; lower the recovery' : ''}. Confirm in suite 2.` });
        const sc = si(wall, 'calcite'); if (sc !== null && sc > 1.8) W.push({ level: 'warn', msg: `Stage ${i + 1}: calcite saturation index at the wall is ${fmt(sc, 2)} — dose acid or antiscalant.` });
      });
    } catch { /* chemistry model unavailable for this composition: the table is simply omitted */ }

    const rej = (id) => (r.feed.ions[id] > 0 ? 100 * (1 - r.product.ions[id] / r.feed.ions[id]) : null);
    const cf = tds(r.conc.ions) / tds(r.feed.ions);
    const xs = els.map((_, i) => i + 1);
    // operating-condition sweeps (array fixed at the solved design)
    const fixed = { ...v, design: 'manual', nStages: r.vessels.length, v1: r.vessels[0], v2: r.vessels[1] || 1, v3: r.vessels[2] || 1, mode: 'recovery', pass2: false, nSeg: Math.min(v.nSeg, 2) };
    const nSw = v.model === 'dspm' ? 5 : 7;
    const Ts = linspace(Math.max(5, v.T - 12), Math.min(42, v.T + 12), nSw), sweepT = Ts.map((T) => { try { const q = simulateRO({ ...fixed, T }); return [q.p1.Pf, tds(q.p1.perm.ions), q.sec]; } catch { return [NaN, NaN, NaN]; } });
    const R0 = v.recovery, Rs = linspace(Math.max(10, R0 - 15), Math.min(92, R0 + 12), nSw), sweepR = Rs.map((rc) => { try { const q = simulateRO({ ...fixed, recovery: rc }); return [q.p1.Pf, tds(q.p1.perm.ions), q.sec]; } catch { return [NaN, NaN, NaN]; } });

    // ---- permeate-side Hagen–Poiseuille losses (reported always; coupled to the flux when selected)
    const cfg = r.cfg, M = cfg.M, hpOn = cfg.permSide === 'hp';
    r.p1.stages.forEach((st, i) => { const tp = hpOn ? null : tubeLoss(st.els.map((e) => e.Qp), cfg.tubeD, cfg.rhoW, cfg.muW, cfg.permPort); st.els.forEach((e, j) => { const q = els.find((x) => x.stage === i + 1 && x.el === e.el); q.permLoss = (hpOn ? e.PpTube : tp[j]) + cfg.cLeaf * e.flux; }); });
    const maxPermLoss = Math.max(...els.map((e) => e.permLoss));
    // ---- irreversible thermodynamics: dissipation and entropy production of the membrane
    const dissKW = sum(r.p1.stages.map((st) => st.diss)) / 1000, entropy = dissKW / (v.T + 273.15), dissSEC = dissKW / Math.max(r.p1.perm.Q, 1e-12), minPhi = Math.min(...els.map((e) => e.diss));
    const Asi = (M.A * cfg.tcf * cfg.ff * cfg.hydA) / 3.6e11, A25 = M.A / 3.6e11, eps = v.poreEps ?? 0.05, tau = v.poreTau ?? 2.5, dAct = (v.activeThk ?? 150) * 1e-9;
    const rEq = poreRadiusFromA(A25, eps, tau, dAct), Dwm = waterDiffusivitySD(A25, cfg.phiW, dAct, 25), Rm = 1 / (cfg.muW * Asi), RsMax = cfg.alpha * 2 * cfg.scaleMean;
    // ---- feed-channel boundary layer (no-slip membranes, symmetry plane) and start-up transient from the initial concentrations
    const chd = channelDiagnostics(r, v), su = startupTransient(r, v), cip = cleaningAssessment(v, r);
    if (cip.need) W.push({ level: 'warn', msg: `Cleaning is due: ${cip.reasons.join('; ')}. ${cip.chem}.` });
    if (cfg.scaleMean > 0) W.push({ level: 'info', msg: `Deposited scale (${fmt(v.scaleMass, 3)} g/m² array mean, rising linearly to ${fmt(2 * v.scaleMass, 3)} g/m² at the tail) adds up to ${fmt((100 * RsMax) / Rm, 3)} % to the membrane resistance.` });
    if ((v.hydration ?? 100) < 99.5) W.push({ level: 'info', msg: `Membrane water content at ${fmt(v.hydration, 3)} % of full hydration: water permeability × ${fmt(cfg.hydA, 3)}, salt permeability × ${fmt(cfg.hydB, 3)} (free-volume theory).` });
    if (hpOn && maxPermLoss > 0.5) W.push({ level: 'warn', msg: `Permeate-side pressure loss reaches ${fmt(maxPermLoss, 2)} bar — check the permeate spacer, leaf length and tube diameter.` });
    if (su.ss > 0 && su.cp0 > v.limTDS) W.push({ level: 'info', msg: `Start-up: with ${fmt(su.cp0, 3)} mg/L initially on the permeate side, about ${fmt(su.divert, 2)} m³ of permeate should be diverted before the product limit is met${su.ss > v.limTDS ? ' (the steady permeate itself is above the limit)' : ''}.` });
    const dsp = cfg.model === 'dspm' ? r.p1.stages.map((st) => st.dspm).filter(Boolean) : [];
    let dspProf = null;
    if (dsp.length) {
      if (dsp.some((q) => !q.converged)) W.push({ level: 'warn', msg: 'The pore-model equations did not converge for at least one stage; uncharged-solute (steric) transmissions were used there. Check pore radius, charge density and pore dielectric constant.' });
      if (M.A < 3) W.push({ level: 'warn', msg: 'The Donnan–steric pore model describes nanofiltration membranes. With a tight reverse-osmosis permeability it gives a thick equivalent pore layer and only nanofiltration-level rejection — use solution–diffusion or Spiegler–Kedem for RO elements.' });
      const q = dsp[0]; dspProf = dspmTable(q.wall, q.jr0, q.jr1, cfg, q, true).hi;
    }
    if (cfg.model === 'kk' && minPhi < 0) W.push({ level: 'bad', msg: 'Negative dissipation function: the Kedem–Katchalsky coefficients violate the second law (check σ and B).' });
    let sur = null;
    if (v.surrogate ?? true) { try { sur = buildSurrogate(v, r); } catch { sur = null; } if (!sur) W.push({ level: 'info', msg: 'The grey-box surrogate could not be trained around this operating point (too few valid training runs).' }); }

    const out = {
      streams: { feed: streamOut(r.feed, v.T, r.p1.Pf, v.pH), permeate: streamOut(r.product, v.T, v.Pp, r.permPH), concentrate: streamOut(r.conc, v.T, r.conc.P, Math.min(9, v.pH + 0.3 * Math.log10(Math.max(cf, 1e-3)))) },
      feedPressureBar: r.p1.Pf, concentratePressureBar: r.conc.P, recovery: r.overallRec, permeateFlow: r.product.Q, fluxLMH: avgFlux, membraneArea: r.area, nElements: r.nEl, nVessels: sum(r.vessels),
      sec: r.sec, pumpPower: r.power, dpBar: r.p1.Pf - r.conc.P, cpFactor: maxCP, vessels: r.vessels, concentrationFactor: cf, feedFlow: r.feed.Q, erdType: v.erd,
      membraneDissipation: dissKW, entropyProduction: entropy, permeateSideLoss: maxPermLoss, equivalentPoreRadius: rEq, startupSettleTime: su.tSettle, normPermeateFlowDecline: cip.npf, cleaningDue: cip.need,
    };
    if (sur) { out.surrogateR2 = sur.rec.r2; }
    return {
      summary: `${fmt(r.product.Q, 4)} m³/h of product at ${fmt(pT, 3)} mg/L TDS from ${fmt(r.feed.Q, 4)} m³/h feed (${fmt(100 * r.overallRec, 3)} % recovery) at ${fmt(r.p1.Pf, 3)} bar, using ${fmt(r.sec, 3)} kWh/m³.`,
      warnings: W,
      kpis: [
        { label: 'Product flow', value: r.product.Q, unit: 'm³/h' }, { label: 'Overall recovery', value: 100 * r.overallRec, unit: '%' },
        { label: 'Feed pressure', value: r.p1.Pf, unit: 'bar', status: r.p1.Pf > r.cfg.M.pmax ? 'bad' : 'ok' }, { label: 'Product TDS', value: pT, unit: 'mg/L', status: pT > v.limTDS ? 'bad' : 'ok' },
        { label: 'Salt rejection', value: 100 * (1 - pT / tds(r.feed.ions)), unit: '%', sig: 5 }, { label: 'Product boron', value: r.product.ions.B, unit: 'mg/L', status: r.product.ions.B > v.limBoron ? 'bad' : 'ok' },
        { label: 'Average flux', value: avgFlux, unit: 'L/m²·h' }, { label: 'Lead-element flux', value: lead.flux, unit: 'L/m²·h', status: maxFlux > v.limFlux ? 'bad' : 'ok' },
        { label: 'Concentrate TDS', value: tds(r.conc.ions), unit: 'mg/L' }, { label: 'Concentration factor', value: cf, unit: '×' },
        { label: 'Specific energy', value: r.sec, unit: 'kWh/m³', help: 'Electrical energy of the membrane system per m³ of product' }, { label: 'Thermodynamic minimum', value: r.minSEC, unit: 'kWh/m³', help: 'Reversible work of separation at this recovery' },
        { label: 'Total power', value: r.power, unit: 'kW' }, { label: 'Elements / vessels', value: `${r.nEl} / ${sum(r.vessels)}` },
        { label: 'Membrane area', value: r.area, unit: 'm²' }, { label: 'Feed osmotic pressure', value: r.piF, unit: 'bar' },
        { label: 'Membrane dissipation', value: dissSEC, unit: 'kWh/m³', status: minPhi < 0 ? 'bad' : 'ok', help: 'Dissipation function Φ = Jv·(ΔP − σΔπ) + Σ Js·Δμs integrated over the membrane area, per m³ of permeate; Φ ≥ 0 is the second-law requirement of irreversible thermodynamics' },
        { label: 'Entropy production', value: entropy, unit: 'kW/K', help: 'Membrane dissipation ÷ absolute temperature' },
        { label: 'Permeate-side loss (max)', value: maxPermLoss, unit: 'bar', help: `Hagen–Poiseuille flow in the permeate leaf and central tube; ${hpOn ? 'coupled to the flux' : 'reported only — select the Hagen–Poiseuille permeate-side model to couple it'}` },
        { label: 'β, 2-D boundary layer (lead)', value: chd.bl.cpMean, unit: '–', help: `No-slip membranes and symmetry plane, boundary layer re-developing over ${fmt(1000 * chd.L, 2)} mm; film correlation gives ${fmt(chd.betaCorr, 4)}` },
        { label: 'Start-up settling time', value: su.tSettle, unit: 's', help: 'Time for the permeate TDS to stay within 5 % of its steady value, from the initial concentrate- and permeate-side concentrations' },
        { label: 'Normalised flow vs clean', value: -cip.npf, unit: '%', status: cip.need ? 'warn' : 'ok', help: 'Change of permeate flow per unit net driving pressure against a clean reference at the same pressure, temperature and feed (cleaning assessment)' },
        ...(sur ? [{ label: 'Surrogate hold-out R²', value: sur.rec.r2, unit: '–', sig: 5, status: sur.rec.r2 > 0.98 ? 'ok' : 'warn', help: 'Physics-informed grey-box surrogate of recovery against element-resolved runs it was not trained on' }] : []),
      ],
      recommendations: [
        maxFlux > v.limFlux ? 'Flux is unbalanced toward the lead elements: apply 1–3 bar permeate back-pressure on stage 1 or add an inter-stage booster.' : null,
        r.sec > 1.6 * r.minSEC && v.erd === 'none' && r.conc.P > 20 ? 'Add an isobaric energy-recovery device — the concentrate still carries most of the pumping energy.' : null,
        'Send the concentrate to suite 2 (Brine chemistry) to confirm the scaling margin and antiscalant dose at this recovery.',
        'Use suite 11 (Optimisation) to trade recovery against energy, or suite 13 (Economics) for the cost of water.',
      ].filter(Boolean),
      plots: [
        { type: 'line', title: 'Flux and net driving pressure along the array', xlabel: 'Element position (feed → concentrate)', ylabel: 'Flux (L/m²·h) · NDP (bar)', series: [{ name: 'Water flux', x: xs, y: els.map((e) => e.flux), mode: 'both' }, { name: 'Net driving pressure', x: xs, y: els.map((e) => e.ndp), mode: 'both' }], hlines: [{ y: v.limFlux, label: 'flux limit' }] },
        { type: 'line', title: 'Pressure and osmotic pressure along the array', xlabel: 'Element position', ylabel: 'bar', series: [{ name: 'Feed-side pressure', x: xs, y: els.map((e) => e.Pin), mode: 'both' }, { name: 'Bulk osmotic pressure', x: xs, y: els.map((e) => osmoticPressureIons(scaleIons(r.feed.ions, e.tdsFeed / tds(r.feed.ions)), v.T) / 1e5), mode: 'both' }] },
        { type: 'line', title: 'Salinity and concentration polarisation', xlabel: 'Element position', ylabel: 'Feed-side TDS (g/L) · β (–)', series: [{ name: 'Bulk TDS (g/L)', x: xs, y: els.map((e) => e.tdsFeed / 1000), mode: 'both' }, { name: 'β × 10', x: xs, y: els.map((e) => e.CP * 10), mode: 'both' }] },
        { type: 'line', title: 'Permeate TDS by element', xlabel: 'Element position', ylabel: 'mg/L', series: [{ name: 'Element permeate TDS', x: xs, y: els.map((e) => e.tdsPerm), mode: 'both' }], hlines: [{ y: v.limTDS, label: 'product limit' }] },
        { type: 'bar', title: 'Ion rejection', ylabel: '% rejected', categories: ION_IDS.filter((k) => r.feed.ions[k] > 0).map((k) => IONS[k].label), series: [{ name: 'Rejection', values: ION_IDS.filter((k) => r.feed.ions[k] > 0).map((k) => rej(k)) }] },
        { type: 'bar', title: 'Power breakdown', ylabel: 'kW', categories: ['HP pump', 'ERD booster', 'Inter-stage', 'Second pass', 'Recovered by ERD'], series: [{ name: 'kW', values: [r.hpKW, r.boosterKW, r.boostKW, r.p2KW, -r.erdRecovered] }] },
        { type: 'line', title: 'Effect of feed temperature (array fixed)', xlabel: 'Temperature (°C)', ylabel: 'Pressure (bar) · permeate TDS/10 (mg/L)', series: [{ name: 'Feed pressure', x: Ts, y: sweepT.map((q) => q[0]), mode: 'both' }, { name: 'Permeate TDS ÷ 10', x: Ts, y: sweepT.map((q) => q[1] / 10), mode: 'both' }, { name: 'SEC × 10 (kWh/m³)', x: Ts, y: sweepT.map((q) => q[2] * 10), mode: 'both' }] },
        { type: 'field', title: 'Concentration boundary layer at the lead-element membrane (c / c inlet)', xlabel: 'Distance along one spacer cell (mm)', ylabel: 'Distance from the membrane (µm)', zlabel: 'c/c₀', zunit: '–', x: chd.xU.map((x) => 1000 * x), y: chd.yU.map((y) => 1e6 * y), z: chd.z, cmap: 'salinity', contours: 8,
          note: 'Two-dimensional convection–diffusion with a no-slip parabolic velocity profile, wall suction, zero gradient at the channel mid-plane (symmetry) and the membrane flux condition D·∂c/∂y = Jv·c at the wall.' },
        { type: 'line', title: 'Polarisation along one spacer cell: 2-D boundary layer versus film theory', xlabel: 'Distance (mm)', ylabel: 'c wall / c bulk (–)', series: [{ name: '2-D boundary layer', x: chd.bl.x.map((x) => 1000 * x), y: chd.bl.cw.map((c, i) => c / chd.bl.cb[i]) }, { name: 'Film theory with Lévêque k(x)', x: chd.bl.x.map((x) => 1000 * x), y: chd.film, dash: true }], hlines: [{ y: chd.betaCorr, label: 'spacer correlation β' }] },
        { type: 'line', title: 'Start-up transient from the initial concentrations', xlabel: 'Time after pressurisation (s)', ylabel: 'Permeate TDS (mg/L) · concentrate TDS ÷ 100 (mg/L)', series: [{ name: 'Permeate TDS', x: su.t, y: su.perm }, { name: 'Concentrate TDS ÷ 100', x: su.t, y: su.conc.map((c) => c / 100) }, { name: 'Permeate flow (% of steady)', x: su.t, y: su.Qp.map((q) => (100 * q) / Math.max(r.p1.perm.Q, 1e-12)) }], hlines: [{ y: v.limTDS, label: 'product limit' }],
          note: 'Tanks-in-series salinity balance of the feed channels with a mixed permeate hold-up; each element follows its steady solution–diffusion characteristic. Feed pressure is applied as a step.' },
        ...(sur ? [{ type: 'field', title: 'Operating map from the grey-box surrogate: recovery (%)', xlabel: 'Feed pressure (bar)', ylabel: 'Temperature (°C)', zlabel: 'Recovery', zunit: '%', x: sur.map.P, y: sur.map.T, z: sur.map.rec, cmap: 'viridis', contours: 8, markers: [{ x: r.p1.Pf, y: v.T, label: 'design' }], note: 'Array fixed at the solved design. Closed-form solution–diffusion baseline × fitted correction, trained on element-resolved runs.' },
          { type: 'line', title: 'Surrogate parity on hold-out runs', xlabel: 'Element-resolved model', ylabel: 'Surrogate', series: [{ name: 'Recovery (%)', x: sur.parity.mechRec, y: sur.parity.surRec, mode: 'points' }, { name: 'Permeate TDS ÷ 10 (mg/L)', x: sur.parity.mechTds.map((x) => x / 10), y: sur.parity.surTds.map((x) => x / 10), mode: 'points' }, { name: '1 : 1', x: [0, Math.max(...sur.parity.mechRec, ...sur.parity.mechTds.map((x) => x / 10))], y: [0, Math.max(...sur.parity.mechRec, ...sur.parity.mechTds.map((x) => x / 10))], dash: true }] }] : []),
        ...(dspProf && dspProf.profile ? [{ type: 'line', title: 'Ion concentrations across the active layer (extended Nernst–Planck, stage 1)', xlabel: 'Position in the pore (0 = feed side, 1 = permeate side)', ylabel: 'Pore concentration (mol/m³) · potential × 10 (F·ψ/RT)', series: [...dspProf.profile.ids.map((id, i) => ({ name: IONS[id].label, x: dspProf.profile.y, y: dspProf.profile.c.map((c) => c[i]) })), { name: 'Potential × 10', x: dspProf.profile.y, y: dspProf.profile.psi.map((q) => 10 * q), dash: true }] }] : []),
        { type: 'line', title: 'Effect of recovery (array fixed)', xlabel: 'Recovery (%)', ylabel: 'Pressure (bar) · permeate TDS/10 (mg/L)', series: [{ name: 'Feed pressure', x: Rs, y: sweepR.map((q) => q[0]), mode: 'both' }, { name: 'Permeate TDS ÷ 10', x: Rs, y: sweepR.map((q) => q[1] / 10), mode: 'both' }, { name: 'SEC × 10 (kWh/m³)', x: Rs, y: sweepR.map((q) => q[2] * 10), mode: 'both' }] },
      ],
      tables: [
        { title: 'Stage summary', columns: ['Stage', 'Vessels', 'Feed (m³/h)', 'Permeate (m³/h)', 'Concentrate (m³/h)', 'Recovery (%)', 'Feed P (bar)', 'Conc. P (bar)', 'Avg flux (L/m²·h)', 'Permeate TDS (mg/L)', 'Feed/vessel (m³/h)', 'Conc./vessel (m³/h)'],
          rows: [...r.p1.stages.map((s, i) => [`Pass 1 · ${i + 1}`, s.nV, s.feed.Q, s.perm.Q, s.conc.Q, (100 * s.perm.Q) / s.feed.Q, s.feed.P, s.conc.P, (s.perm.Q * 1000) / (s.nV * r.cfg.elements * r.cfg.M.area), tds(s.perm.ions), s.feed.Q / s.nV, s.conc.Q / s.nV]),
            ...(r.p2 ? r.p2.stages.map((s, i) => [`Pass 2 · ${i + 1}`, s.nV, s.feed.Q, s.perm.Q, s.conc.Q, (100 * s.perm.Q) / s.feed.Q, s.feed.P, s.conc.P, (s.perm.Q * 1000) / (s.nV * r.pass2.cfg.elements * r.pass2.cfg.M.area), tds(s.perm.ions), s.feed.Q / s.nV, s.conc.Q / s.nV]) : [])] },
        { title: 'Element-by-element profile (one vessel per stage)', columns: ['Stage', 'Element', 'Feed (m³/h)', 'Inlet P (bar)', 'ΔP (bar)', 'Flux (L/m²·h)', 'Recovery (%)', 'β', 'NDP (bar)', 'Feed TDS (mg/L)', 'Permeate TDS (mg/L)', 'Permeate (m³/h)', 'Permeate-side loss (bar)', 'Dissipation (W/m²)', 'Scale (g/m²)'],
          rows: els.map((e) => [e.stage, e.el, e.Qin, e.Pin, e.dP, e.flux, 100 * e.rec, e.CP, e.ndp, e.tdsFeed, e.tdsPerm, e.Qp, e.permLoss, e.diss / M.area, 1000 * e.mScale]) },
        { title: 'Membrane structure, state and irreversible thermodynamics', columns: ['Quantity', 'Value', 'Unit'], rows: [
          ['Hydraulic permeability Lp (operating)', Asi, 'm/(s·Pa)'], ['Reflection coefficient σ', cfg.model === 'sk' || cfg.model === 'kk' ? M.sigma : 1, '–'], ['Solute permeability ω = B/(R·T), NaCl', (M.B * cfg.tcf * cfg.sp) / 3.6e6 / (RGAS * (v.T + 273.15)), 'mol/(m²·s·Pa)'],
          ['Membrane dissipation (Σ Φ·dA)', dissKW, 'kW'], ['Entropy production', entropy, 'kW/K'], ['Lowest local dissipation function Φ', minPhi / M.area, 'W/m²'], ['Membrane dissipation per m³ of permeate', dissSEC, 'kWh/m³'],
          ['Equivalent pore radius (Hagen–Poiseuille pore flow)', 1e9 * rEq, 'nm'], ['Surface porosity ε / tortuosity τ / active-layer thickness', `${eps} / ${tau} / ${fmt(1e9 * dAct, 3)} nm`, ''],
          ['Water diffusivity in the active layer (solution–diffusion)', Dwm, 'm²/s'], ['Equilibrium water volume fraction', cfg.phiW, '–'], ['Membrane water content', v.hydration ?? 100, '% of full'], ['Hydration factor on A / on B', `${fmt(cfg.hydA, 4)} / ${fmt(cfg.hydB, 4)}`, '×'],
          ['Membrane resistance Rm', Rm, '1/m'], ['Scale-layer resistance at the tail', RsMax, '1/m'], ['Permeate leaf length / channel height', `${fmt(cfg.Lleaf, 3)} m / ${fmt(1000 * cfg.hp, 3)} mm`, ''], ['Leaf back-pressure coefficient', cfg.cLeaf, 'bar per L/m²·h'], ['Largest permeate-side loss', maxPermLoss, 'bar']],
          note: 'Φ = Jv·(ΔP − σ·Δπ) + R·T·Σ ln(cm/cp)·(Js − (1 − σ)·c̄·Jv) per unit area. The pore radius is the capillary radius that gives the same permeability by A = ε·r²/(8·μ·τ·δ); the water diffusivity follows from A = D·φ·V̄/(R·T·δ).' },
        { title: 'Feed-channel boundary layer, lead element (wall and symmetry conditions)', columns: ['Quantity', 'Value', 'Unit'], rows: [
          ['Mean axial velocity', chd.ub, 'm/s'], ['Velocity at the membrane (no slip)', 0, 'm/s'], ['Wall shear rate 6ū/h', chd.shear, '1/s'], ['Wall (permeation) velocity', chd.vw, 'm/s'], ['Concentration gradient at the mid-plane (symmetry), relative', chd.centreGrad, '–'],
          ['Solute balance residual of the 2-D solution', chd.bl.balance, '–'], ['Mean c wall / c bulk, 2-D boundary layer', chd.bl.cpMean, '–'], ['Mean c wall / c bulk, film theory with Lévêque k(x)', chd.filmMean, '–'], ['Polarisation factor β, spacer correlation', chd.betaCorr, '–'],
          ['Mass-transfer coefficient, 2-D boundary layer', chd.k2d, 'm/s'], ['Mass-transfer coefficient, spacer correlation', chd.kCorr, 'm/s'], ['Re-development (spacer mesh) length', 1000 * chd.L, 'mm']],
          note: `Open-slit solution restarted at every spacer filament. ${cfg.cpModel === 'bl2d' ? 'This model supplies the mass-transfer coefficient of every segment.' : 'Shown for comparison; the spacer Sherwood correlation supplies the mass-transfer coefficient (select the 2-D boundary-layer option to use this solution instead).'}` },
        { title: 'Start-up transient (initial conditions)', columns: ['Quantity', 'Value', 'Unit'], rows: [
          ['Initial concentrate-side TDS', su.c0, 'mg/L'], ['Initial permeate-side TDS', su.cp0, 'mg/L'], ['Feed-channel residence time', su.tau, 's'], ['Steady permeate TDS', su.ss, 'mg/L'], ['Steady concentrate TDS', su.concSS, 'mg/L'],
          ['Permeate TDS after one residence time', interp1(su.t, su.perm, su.tau), 'mg/L'], ['Time to within 5 % of steady permeate TDS', su.settled ? su.tSettle : null, 's'], ['Permeate to divert before the product limit is met', su.ss > v.limTDS ? null : su.divert, 'm³']] },
        { title: 'Cleaning assessment', columns: ['Indicator', 'Now', 'Clean reference', 'Change (%)', 'Trigger (%)'], rows: [
          ['Permeate flow at the solved pressure (m³/h)', cip.QpAct, cip.QpRef, cip.lumped ? null : 100 * (cip.QpAct / cip.QpRef - 1), null], ['Mean net driving pressure (bar)', cip.ndpAct, cip.ndpRef, cip.lumped ? null : 100 * (cip.ndpAct / cip.ndpRef - 1), null],
          ['Normalised permeate flow (m³/h per bar)', cip.lumped ? null : cip.QpAct / cip.ndpAct, cip.lumped ? null : cip.QpRef / cip.ndpRef, -cip.npf, -(v.cipNPF ?? 10)], ['Normalised pressure drop (bar)', cip.dpAct, cip.dpRef, cip.dpRise, v.cipDP ?? 15], ['Normalised salt passage (%)', cip.spAct, cip.spRef, cip.spRise, 10],
          ['Cleaning required', cip.need ? 'yes' : 'no', null, null, null], ['Cleaning-solution volume per stage (m³)', cip.volume, null, null, null], ['Recirculation flow per vessel (m³/h)', cip.flowPerVessel, null, null, null],
          ['Deposited scale to dissolve (kg)', cip.scaleKg, null, null, null], ['Acid demand as HCl / citric acid (kg, carbonate scale)', `${fmt(cip.hclKg, 3)} / ${fmt(cip.citricKg, 3)}`, null, null, null], ['Chelant demand as Na₄EDTA (kg, sulphate scale)', cip.edtaKg, null, null, null]],
          note: `Both columns are simulated at the solved feed pressure with the array fixed and normalised in the manner of ASTM D4516 (flow per unit net driving pressure, salt passage at equal flux, pressure drop at equal mean flow); the reference has flow factor ≥ 1, no deposited scale, full hydration and the nominal pressure-drop multiplier (age effects are kept: they are not reversed by cleaning). Recommended chemistry: ${cip.chem}.${cip.lumped ? ' The clean array would exceed its hydraulic limits at this feed pressure, so the changes shown are those of the membrane permeability and of the pressure-drop multiplier.' : ''}` },
        ...(sur ? [{ title: 'Physics-informed surrogate (grey box)', columns: ['Output', 'Hold-out R²', 'Largest hold-out error (%)', 'Physics baseline alone: largest error (%)', 'Design point: model', 'Design point: surrogate'], rows: [
          ['Recovery (%)', sur.rec.r2, sur.rec.maxErr, sur.recBase.maxErr, 100 * sur.design.mech[0], 100 * sur.design.sur[0]], ['Permeate TDS (mg/L)', sur.tdsP.r2, sur.tdsP.maxErr, sur.tdsBase.maxErr, sur.design.mech[1], sur.design.sur[1]]],
          note: `${sur.nTrain} training and ${sur.nTest} hold-out runs of the element-resolved model (one segment per element) over ±15 % pressure, ±10 °C and ±10 % salinity. Baseline: lumped solution–diffusion with logarithmic-mean concentration; correction: exp(θ·[1, ΔP, ΔT, Δs, ΔP², ΔP·ΔT]).` }] : []),
        ...(dspProf ? [{ title: 'Donnan–steric pore model with dielectric exclusion (stage 1, highest flux)', columns: ['Ion', 'λ = r/rp', 'Steric Φ', 'Born factor', 'Kd', 'Kc', 'Péclet', 'Wall (mol/m³)', 'Pore entrance (mol/m³)', 'Transmission cp/cm', 'Real rejection (%)', 'Solved as'],
          rows: dspProf.ions.map((q) => [IONS[q.id].label, q.lam, q.phi, q.born, q.Kd, q.Kc, q.Pe, q.c, q.c0, q.T, 100 * (1 - q.T), q.z === 0 ? 'neutral (steric)' : q.major ? 'coupled' : 'tracer']),
          note: `Pore radius ${fmt(1e9 * cfg.dspm.rp, 3)} nm, effective thickness δ·τ/ε = ${fmt(1e6 * cfg.dspm.dx, 3)} µm (from A by Hagen–Poiseuille), charge density ${cfg.dspm.X} mol/m³, pore dielectric constant ${cfg.dspm.de ? cfg.dspm.epsP : 'bulk (no dielectric exclusion)'}. Feed-side Donnan potential ${fmt(dspProf.xi0, 3)}, permeate-side ${fmt(dspProf.xip, 3)} (F·ψ/RT). Solved at the stage-mean wall composition for the lowest and highest segment flux and interpolated in flux for each segment; ions with less than 1 % of the charge move as tracers in the potential of the others; ideal solution in the pore.` }] : []),
        { title: 'Stream compositions (mg/L)', columns: ['Constituent', 'Feed', 'Product', 'Concentrate', 'Rejection (%)', 'Concentration factor'],
          rows: [...ION_IDS.map((k) => [`${IONS[k].name} ${IONS[k].label}`, r.feed.ions[k], r.product.ions[k], r.conc.ions[k], rej(k), r.feed.ions[k] > 0 ? r.conc.ions[k] / r.feed.ions[k] : null]),
            ['TDS', tds(r.feed.ions), pT, tds(r.conc.ions), 100 * (1 - pT / tds(r.feed.ions)), cf], ['pH (estimated)', v.pH, r.permPH, out.streams.concentrate.pH, null, null],
            ['Conductivity (µS/cm)', conductivity(r.feed.ions, v.T), conductivity(r.product.ions, v.T), conductivity(r.conc.ions, v.T), null, null], ['Flow (m³/h)', r.feed.Q, r.product.Q, r.conc.Q, null, null]],
          note: r.p2 ? 'Product = second-pass permeate blended with bypassed first-pass permeate. Second-pass concentrate is returned to the first-pass feed; its small dilution effect on the feed is neglected.' : '' },        ...(scaleRows.length ? [{ title: 'Scaling tendency by stage (saturation index, log₁₀ scale; > 0 = supersaturated)', columns: ['Stream', 'TDS (mg/L)', 'pH', ...SCALES.flatMap(([, n]) => [n + ' · bulk', n + ' · wall'])], rows: scaleRows, note: 'Pitzer electrolyte model of suite 2. “Wall” applies the highest polarisation factor of the stage. Antiscalants typically hold sulphate scales up to SI ≈ 0.4–0.6 and calcite up to about 1.8–2.' }] : []),
      ],
      balances: [
        { name: 'Water (m³/h), pass 1', in: r.feed.Q, out: r.p1.perm.Q + r.conc.Q },
        { name: 'Salt (kg/h), pass 1', in: (r.feed.Q * tds(r.feed.ions)) / 1000, out: (r.p1.perm.Q * tds(r.p1.perm.ions) + r.conc.Q * tds(r.conc.ions)) / 1000 },
        { name: 'Chloride (kg/h), pass 1', in: (r.feed.Q * r.feed.ions.Cl) / 1000, out: (r.p1.perm.Q * r.p1.perm.ions.Cl + r.conc.Q * r.conc.ions.Cl) / 1000 },
      ],
      outputs: out,
    };
  },

  mesh: { name: 'Axial discretisation (segments per element)', keys: ['nSeg'], min: 1, note: 'The array, pressure mode and all other inputs are held constant while the number of segments per element is refined.',
    metrics: [{ label: 'Feed pressure', unit: 'bar', get: (r) => r.outputs.feedPressureBar }, { label: 'Product TDS', unit: 'mg/L', get: (r) => r.outputs.streams.permeate.tds }, { label: 'Specific energy', unit: 'kWh/m³', get: (r) => r.outputs.sec }] },

  calibration: {
    note: 'Fit the membrane parameters to plant or pilot data. Each row is one steady operating point: feed pressure, temperature, feed flow and salinity multiplier set the condition; permeate flow, permeate TDS and array pressure drop are the measurements. Use rows spanning pressure, temperature and salinity so that A, B and the multipliers are identifiable. Enter the real array first (Array definition → “I will enter stages and vessels”).',
    params: [{ key: 'A', label: 'Water permeability A', lo: 0.2, hi: 12 }, { key: 'B', label: 'Salt permeability B', lo: 0.005, hi: 60 }, { key: 'kdp', label: 'Pressure-drop multiplier', lo: 0.3, hi: 6 }],
    columns: [{ key: 'Pfeed', label: 'Feed pressure', unit: 'bar' }, { key: 'T', label: 'Temperature', unit: '°C' }, { key: 'Qf', label: 'Feed flow', unit: 'm³/h' }, { key: 'salinityFactor', label: 'Salinity ×', unit: '–' }, { key: 'Qp', label: 'Permeate flow', unit: 'm³/h' }, { key: 'tdsP', label: 'Permeate TDS', unit: 'mg/L' }, { key: 'dP', label: 'Array ΔP', unit: 'bar' }],
    targets: [{ key: 'Qp', label: 'Permeate flow', unit: 'm³/h' }, { key: 'tdsP', label: 'Permeate TDS', unit: 'mg/L' }, { key: 'dP', label: 'Array pressure drop', unit: 'bar' }],
    model(v) {
      const base = { ...v, mode: 'pressure', pass2: false, nSeg: Math.min(v.nSeg, 2) };
      if (v.design === 'auto') { base.design = 'manual'; const vs = autoSize({ Qp: 1000 * 0.45, flux: v.targetFlux, area: v.area, elements: v.elements, recovery: 0.45 }); base.nStages = vs.length; [base.v1, base.v2, base.v3] = [vs[0], vs[1] || 1, vs[2] || 1]; }
      const r = simulateRO(base);
      return { Qp: r.p1.perm.Q, tdsP: tds(r.p1.perm.ions), dP: r.p1.Pf - r.conc.P };
    },
    get sample() { return (this._s ||= synth(11, [[56, 22, 1000, 1], [60, 22, 1000, 1], [64, 25, 1000, 1], [58, 28, 1050, 1], [62, 18, 950, 1], [66, 30, 1000, 1.05], [55, 26, 900, 0.95], [61, 24, 1100, 1.02]])); },
    get validationSample() { return (this._v ||= synth(29, [[59, 20, 1000, 1], [63, 27, 980, 1], [57, 31, 1020, 0.97], [65, 23, 1060, 1.04], [60, 16, 940, 1]])); },
  },

  verify() {
    const d = Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    const r = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120 });
    add('Water mass balance closes', 0, (r.feed.Q - r.p1.perm.Q - r.conc.Q) / r.feed.Q, 1e-10, 'Qf = Qp + Qc (relative error)');
    add('Total-salt balance closes', 0, (r.feed.Q * tds(r.feed.ions) - r.p1.perm.Q * tds(r.p1.perm.ions) - r.conc.Q * tds(r.conc.ions)) / (r.feed.Q * tds(r.feed.ions)), 1e-9, 'Σ over all ions');
    add('Permeate is electroneutral', 0, chargeBalance(r.p1.perm.ions).errorPct, 0.01, 'Σ zᵢcᵢ = 0 in the permeate (% of total equivalents)');
    add('Target recovery is met', d.recovery / 100, r.p1.recovery, 1e-4, 'Pressure solved for the specified recovery');
    const z = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, mode: 'pressure', Pfeed: d.Pp + 0.4 });
    add('No flux when applied pressure is below osmotic pressure', 0, z.p1.perm.Q, 1e-9, 'Limiting case ΔP < Δπ');
    const nacl = { ...cloneIons({}), Na: 229.9, Cl: 354.53 }; // 10 mol/m³ NaCl
    add("van't Hoff osmotic pressure, 10 mol/m³ NaCl at 25 °C", 0.4958, vantHoff(nacl, 25) / 1e5, 1e-3, 'π = i·c·R·T = 2 × 10 × 8.314 × 298.15 Pa');
    const pure = simulateRO({ ...d, ions: cloneIons({ Na: 0.0001, Cl: 0.00015 }), design: 'manual', nStages: 1, v1: 1, elements: 1, Qf: 8, mode: 'pressure', Pfeed: 11, Pp: 1, ff: 1, T: 25, kdp: 1e-9, nSeg: 1 });
    add('Pure-water flux equals A·ΔP', d.A * 10, pure.p1.stages[0].els[0].flux, 0.02, 'Zero-solute limit: Jw = A (P − Pp)');
    const c1 = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, nSeg: 4 }), c2 = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, nSeg: 8 });
    add('Result is insensitive to axial refinement (4 → 8 segments)', 0, Math.abs(c1.p1.Pf - c2.p1.Pf) / c2.p1.Pf, 5e-3, 'Relative change of solved feed pressure');
    add('Specific energy is above the thermodynamic minimum', 1, r.sec > r.minSEC ? 1 : 0, 0, 'Second-law check');
    const sk = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, model: 'sk' });
    add('Spiegler–Kedem and solution–diffusion agree for a tight membrane', 0, Math.abs(sk.p1.Pf - r.p1.Pf) / r.p1.Pf, 0.02, 'σ → 1 limit: both models give the same pressure');
    // ---- irreversible thermodynamics (Kedem–Katchalsky) and the dissipation function
    const kk = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, model: 'kk' }), phiMin = (q) => Math.min(...q.p1.stages.flatMap((st) => st.els.map((e) => e.diss)));
    add('Kedem–Katchalsky agrees with solution–diffusion for a tight membrane', 0, Math.abs(kk.p1.Pf - r.p1.Pf) / r.p1.Pf, 0.02, 'σ → 1 limit of Jv = Lp(ΔP − σΔπ), Js = ωΔπ + (1 − σ)c̄Jv');
    add('Dissipation function is non-negative in every element', 1, phiMin(kk) > 0 && phiMin(r) > 0 && phiMin(sk) > 0 ? 1 : 0, 0, 'Second law: Φ = Jv(ΔP − σΔπ) + Σ Js·Δμs ≥ 0 for all three phenomenological models');
    add('Kedem–Katchalsky passage with σ = 1 equals B/(Jw + B)', 0.058 / 14.058, kkPassage(14, 0.058, 1), 1e-12, 'Solution–diffusion limit');
    { const xi = kkPassage(1e9, 1, 0.85); add('Kedem–Katchalsky convective limit', 0, xi * Math.log(1 / xi) - 0.15 * (1 - xi), 1e-6, 'Jv → ∞: cp = (1 − σ)·c̄, i.e. x·ln(1/x) = (1 − σ)(1 − x) with x = cp/cm'); }
    const kx = kkPassage(20, 3, 0.9), cbar = (1 - kx) / Math.log(1 / kx);
    add('Kedem–Katchalsky solute-flux equation is satisfied', 0, 20 * kx - 3 * (1 - kx) - 0.1 * cbar * 20, 1e-9, 'Js = Jv·cp = B·(cm − cp) + (1 − σ)·c̄·Jv with c̄ the logarithmic mean');
    add('Membrane dissipation is below the pumping energy', 1, sum(r.p1.stages.map((st) => st.diss)) / 1000 < (r.feed.Q * r.p1.Pf) / 36 ? 1 : 0, 0, 'Σ Φ·dA < hydraulic power of the feed');
    // ---- Hagen–Poiseuille: tube, permeate leaf and pore flow
    add('Hagen–Poiseuille tube loss, hand value', (128 * 1e-3 * 2 * (0.05 / 3600)) / (Math.PI * 0.02 ** 4) / 1e5, tubeDP(0.05, 0.02, 2, 1000, 1e-3), 1e-12, 'ΔP = 128·μ·L·Q/(π·D⁴) at Re = 884 (bar)');
    { const mu = 9e-4, L = 1.1, hp = 2.5e-4, J = 20 / 3.6e6, n = 4000; let acc = 0; for (let i = 0; i < n; i++) { const x = ((i + 0.5) / n) * L; acc += (12 * mu * J * (L * L - x * x)) / hp ** 3 / n; }
      add('Permeate-leaf back-pressure equals the integrated slit Hagen–Poiseuille profile', acc / 1e5, leafCoeff(mu, L, hp, 1) * 20, 1e-6, 'Mean of p(x) − p(outlet) with dp/dx = −12·μ·q/h³ and q = 2·Jw·x'); }
    const hpw = simulateRO({ ...d, ions: cloneIons({ Na: 0.0001, Cl: 0.00015 }), design: 'manual', nStages: 1, v1: 1, elements: 1, Qf: 8, mode: 'pressure', Pfeed: 11, Pp: 1, ff: 1, T: 25, kdp: 1e-9, nSeg: 1, permSide: 'hp' }), he = hpw.p1.stages[0].els[0];
    add('Coupled permeate-side loss lowers the pure-water flux by A·ΔP(permeate)', d.A * (10 - he.PpTube - he.PpLeaf), he.flux, 0.02, 'Jw = A·(P − Pp − leaf loss − tube loss)');
    add('Permeate back-pressure varies along the vessel when coupled', 1, (() => { const q = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, permSide: 'hp' }).p1.stages[0].els; return q[0].PpTube > q[6].PpTube && q[6].PpTube > 0 ? 1 : 0; })(), 0, 'Outlet at the concentrate end: the lead element sees the whole tube loss');
    add('Pore-flow radius reproduces the permeability', 1.25 / 3.6e11, poreFlowA(poreRadiusFromA(1.25 / 3.6e11, 0.05, 2.5, 1.5e-7), 0.05, 2.5, 1.5e-7), 1e-24, 'A = ε·r²/(8·μ·τ·δ) round trip');
    add('Equivalent pore radius of a seawater membrane, hand value', 0.43, 1e9 * poreRadiusFromA(1.25 / 3.6e11, 0.05, 2.5, 1.5e-7), 0.02, '√(8 × 8.9e-4 × 2.5 × 1.5e-7 × 3.47e-12 / 0.05) ≈ 0.43 nm');
    // ---- pore model: steric, Donnan, dielectric, extended Nernst–Planck
    const symm = [{ id: 'c', z: 1, c: 20, r: 2e-10, D: 1.5e-9 }, { id: 'a', z: -1, c: 20, r: 2e-10, D: 1.5e-9 }], mp = { rp: 5e-10, dx: 2e-6, X: 0, epsP: 78.4, de: false, T: 25 };
    const sy = dspmSolve(symm, 8e-6, mp), hh = hindrance(0.4), pe = (hh.Kc * 8e-6 * 2e-6) / (hh.Kd * 1.5e-9), an = (hh.phi * hh.Kc) / (1 - (1 - hh.phi * hh.Kc) * Math.exp(-pe));
    add('Pore model: uncharged pore and symmetric salt reduce to the steric-hindrance solution', an, sy.T.c, 1e-7, 'cp/cm = Φ·Kc/(1 − (1 − Φ·Kc)·exp(−Pe)); no potential gradient arises');
    const mix = [{ id: 'Na', z: 1, c: 30, r: stokesRadius(IONS.Na.D), D: IONS.Na.D }, { id: 'Mg', z: 2, c: 5, r: stokesRadius(IONS.Mg.D), D: IONS.Mg.D }, { id: 'Cl', z: -1, c: 30, r: stokesRadius(IONS.Cl.D), D: IONS.Cl.D }, { id: 'SO4', z: -2, c: 5, r: stokesRadius(IONS.SO4.D), D: IONS.SO4.D }];
    const mm = { rp: 4.5e-10, dx: 1.2e-6, X: -40, epsP: 55, de: true, T: 25 }, ds = dspmSolve(mix, 7e-6, mm);
    add('Pore model: Newton iteration converges (exit partitioning residual)', 0, ds.converged ? ds.residual : 1, 1e-8, 'Donnan–steric–dielectric equilibrium at the permeate face for every ion');
    add('Pore model: permeate is electroneutral (zero current)', 0, sum(mix.map((q) => q.z * q.c * ds.T[q.id])) / sum(mix.map((q) => Math.abs(q.z) * q.c * ds.T[q.id])), 1e-8, 'Σ zᵢ·Jv·cp,ᵢ = 0');
    add('Pore model: pore entrance is electroneutral with the fixed charge', 0, (sum(ds.ions.map((q) => q.z * q.c0)) + mm.X) / Math.abs(mm.X), 1e-9, 'Σ zᵢ·cᵢ(0) + X = 0 (Donnan partitioning)');
    add('Pore model: sulphate is rejected better than chloride, magnesium better than sodium', 1, ds.T.SO4 < ds.T.Cl && ds.T.Mg < ds.T.Na ? 1 : 0, 0, 'Steric, Donnan and dielectric exclusion all favour the small monovalent ions');
    add('Pore model: rejection vanishes as the flux tends to zero', 1, dspmSolve(mix, 1e-12, mm).T.Na, 1e-3, 'Diffusion equalises the permeate with the wall solution');
    add('Pore model: dielectric exclusion raises the rejection', 1, ds.T.Na < dspmSolve(mix, 7e-6, { ...mm, de: false }).T.Na ? 1 : 0, 0, 'Born solvation barrier, ε(pore) = 55 against 78.4');
    add('Born partition factor of Na⁺, hand value', Math.exp(-((1.602176634e-19 ** 2 / (8 * Math.PI * 8.8541878128e-12 * 1.380649e-23 * 298.15)) * (1 / 55 - 1 / 78.4)) / stokesRadius(IONS.Na.D)), ds.ions[0].born, 1e-12, 'exp(−z²e²/(8πε₀·r·kT)·(1/ε_p − 1/ε_b)) ≈ 0.44');
    { const don = dspmSolve([{ id: 'c', z: 1, c: 10, r: 1e-12, D: 1e-9 }, { id: 'a', z: -1, c: 10, r: 1e-12, D: 1e-9 }], 1e-6, { rp: 1e-8, dx: 1e-6, X: -30, epsP: 78.4, de: false, T: 25 }), ph = don.ions[1].phi;
      add('Donnan equilibrium of a 1:1 salt at the pore entrance', (-30 + Math.sqrt(900 + 4 * (10 * ph) ** 2)) / 2, don.ions[1].c0, 1e-9, 'Co-ion concentration c = (−|X| + √(X² + 4(Φc)²))/2'); }
    add('Stokes–Einstein radius of Na⁺', 0.184, 1e9 * stokesRadius(IONS.Na.D), 0.002, 'kT/(6πμD) with D = 1.334e-9 m²/s');
    const nfv = { ...d, ions: WATERS.lowbrackish.ions, Qf: 300, T: 18, pH: 7.8, membrane: 'nf', A: 8.5, B: 28, spacerMil: 34, design: 'manual', nStages: 1, v1: 40, elements: 6, mode: 'pressure', Pfeed: 6, nSeg: 2, model: 'dspm' }, nfr = simulateRO(nfv);
    add('Pore model in the array: water and salt balances close', 0, Math.abs(nfr.feed.Q - nfr.p1.perm.Q - nfr.conc.Q) / nfr.feed.Q + Math.abs(nfr.feed.Q * tds(nfr.feed.ions) - nfr.p1.perm.Q * tds(nfr.p1.perm.ions) - nfr.conc.Q * tds(nfr.conc.ions)) / (nfr.feed.Q * tds(nfr.feed.ions)), 1e-9, 'Nanofiltration stage with ion-by-ion transmissions');
    add('Pore model in the array: hardness is rejected better than sodium', 1, nfr.p1.perm.ions.Ca / nfr.feed.ions.Ca < nfr.p1.perm.ions.Na / nfr.feed.ions.Na ? 1 : 0, 0, 'Softening selectivity of a negatively charged nanofiltration membrane');
    // ---- feed-channel boundary layer: no slip, symmetry, membrane flux condition
    { const ub = 0.15, h = 7e-4, D = 1.5e-9, L = 0.004, lowF = channelBL({ ub, h, L, D, vw: 1e-7, ny: 60, nx: 80 }), kNum = 1e-7 / (lowF.cw[80] / lowF.cb[80] - 1);
      add('2-D boundary layer reproduces the Lévêque solution at low flux', 1, kNum / levequeK(ub, h, L, D), 0.02, 'k(x) = 0.651·D^(2/3)·(γ/x)^(1/3): requires the linear no-slip velocity profile at the wall');
      const fin = channelBL({ ub, h, L, D, vw: 6e-6, ny: 48, nx: 60 }); let fm = 0; for (let i = 1; i < fin.x.length; i++) fm += 0.5 * (Math.exp(6e-6 / levequeK(ub, h, fin.x[i], D)) + (fin.x[i - 1] > 0 ? Math.exp(6e-6 / levequeK(ub, h, fin.x[i - 1], D)) : 1)) * (fin.x[i] - fin.x[i - 1]);
      add('2-D boundary layer agrees with film theory at design flux', 0, (fin.cpMean - 1) / (fm / L - 1) - 1, 0.05, 'Mean c_wall/c_bulk against exp(Jv/k(x)) integrated over the cell (relative difference of β − 1)');
      add('2-D boundary layer conserves solute (zero flux through the symmetry plane)', 0, fin.balance, 1e-10, 'Inlet solute flow = outlet solute flow for a fully rejecting membrane');
      const imp = channelBL({ ub, h, L, D, vw: 0, R: 0, ny: 24, nx: 16 });
      add('Impermeable wall leaves the concentration uniform', 1, imp.cw[16], 1e-12, 'No permeation, no solute flux at either boundary'); }
    const b2 = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, cpModel: 'bl2d', meshLen: 4 });
    add('2-D boundary-layer option gives a polarisation factor between 1 and the design limit', 1, (() => { const q = Math.max(...b2.p1.stages[0].els.map((e) => e.CP)); return q > 1.01 && q < 1.4 ? 1 : 0; })(), 0, 'Segment mass-transfer coefficients from the convection–diffusion solution');
    // ---- initial states: start-up concentrations, membrane water content, deposited scale; cleaning assessment
    const st0 = startupTransient(r, d, true), st1 = startupTransient(r, { ...d, c0Conc: 1, c0Perm: 5000 });
    add('Start-up transient: the steady state is a fixed point', 0, Math.max(...st0.perm.map((x) => Math.abs(x / st0.ss - 1))), 1e-9, 'Initial concentrations equal to the steady solution stay there');
    add('Start-up transient converges to the steady permeate TDS', 1, st1.perm[st1.perm.length - 1] / st1.ss, 0.01, 'From a permeate-flushed feed side and 5 000 mg/L on the permeate side');
    add('Start-up transient converges to the steady concentrate TDS', 1, st1.conc[st1.conc.length - 1] / st1.concSS, 0.01, 'Tanks in series reach the element-by-element steady profile');
    { const one = tanksTransient({ V: [2], Vp: 1, Qf: 10, cf: 100, c0: [900], cp0: 0, perm: () => ({ Qp: 0, S: 0 }), tEnd: 0.5, n: 200 });
      add('Single mixed tank without permeation follows the exponential flush', 100 + 800 * Math.exp(-2.5), one.conc[200], 1e-6, 'c(t) = c_f + (c₀ − c_f)·exp(−Q·t/V)'); }
    add('Fully hydrated membrane: no change of permeability', 1, hydrationFactors(1, 0.25).A * hydrationFactors(1, 0.25).B, 1e-14, 'Hydration factors are 1 at 100 % water content');
    add('Free-volume hydration factor, hand value', 0.8 * Math.exp(-0.45 * (1 / 0.2 - 1 / 0.25)), hydrationFactors(0.8, 0.25).A, 1e-14, 'P/P₀ = h·exp(−b·(1/(h·φ) − 1/φ)) at 80 % hydration');
    add('Water diffusivity in the active layer, hand value', 2.87e-10, waterDiffusivitySD(1.25 / 3.6e11, 0.25, 1.5e-7, 25), 5e-12, 'D = A·R·T·δ/(φ·V̄) for A = 1.25 L/m²·h·bar, δ = 150 nm');
    const dry = simulateRO({ ...d, design: 'manual', nStages: 1, v1: 120, hydration: 85 });
    add('Lower membrane water content raises the pressure and the salt rejection', 1, dry.p1.Pf > r.p1.Pf && tds(dry.p1.perm.ions) < tds(r.p1.perm.ions) ? 1 : 0, 0, 'Salt permeability falls faster than water permeability as free volume is lost');
    const scw = simulateRO({ ...d, ions: cloneIons({ Na: 0.0001, Cl: 0.00015 }), design: 'manual', nStages: 1, v1: 1, elements: 1, Qf: 8, mode: 'pressure', Pfeed: 11, Pp: 1, ff: 1, T: 25, kdp: 1e-9, nSeg: 1, scaleMass: 100, scaleAlpha: 5 });
    add('Deposited scale acts as a resistance in series', (d.A * 10) / (1 + (viscosity(25, 0) * 5e14 * 0.1 * d.A) / 3.6e11), scw.p1.stages[0].els[0].flux, 0.02, 'Jw = ΔP/(μ·(Rm + α·m)) with 100 g/m² at mid-element (pure water)');
    const cl0 = cleaningAssessment({ ...d, ff: 1, age: 0, kdp: 1, scaleMass: 0, hydration: 100 }, simulateRO({ ...d, ff: 1, design: 'manual', nStages: 1, v1: 120 })), cl1 = cleaningAssessment({ ...d, ff: 0.8, scaleMass: 150 }, simulateRO({ ...d, ff: 0.8, scaleMass: 150, design: 'manual', nStages: 1, v1: 120 }));
    add('Cleaning assessment: a clean array shows no normalised decline', 0, Math.abs(cl0.npf) + Math.abs(cl0.dpRise) + Math.abs(cl0.spRise), 1e-9, 'State equal to the clean reference');
    add('Cleaning assessment: a fouled and scaled array triggers cleaning', 1, cl1.need && cl1.npf > 10 ? 1 : 0, 0, 'Flow factor 0.8 and 150 g/m² of scale: normalised flow decline above the 10 % trigger');
    add('Acid demand follows the carbonate stoichiometry', (2 * 36.461) / 100.087, cl1.hclKg / cl1.scaleKg, 1e-12, 'CaCO₃ + 2 HCl → CaCl₂ + H₂O + CO₂');
    // ---- physics-informed surrogate
    const sg = buildSurrogate({ ...d, design: 'manual', nStages: 1, v1: 120 }, r);
    add('Grey-box surrogate reproduces hold-out runs of the element model', 1, sg ? sg.rec.r2 : 0, 0.02, 'Recovery, R² on runs not used for training');
    add('Grey-box surrogate: hold-out permeate-TDS error is small', 0, sg ? sg.tdsP.maxErr : 100, 5, 'Largest relative error in % on hold-out runs');
    add('Grey-box correction improves on the physics baseline alone', 1, sg && sg.rec.maxErr < sg.recBase.maxErr ? 1 : 0, 0, 'The fitted correction accounts for polarisation, pressure loss and axial profiles');
    return C;
  },
};

/** Synthetic "measured" data: the model with slightly different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value])), g = rng(seed);
  return pts.map(([Pfeed, T, Qf, salinityFactor]) => {
    const m = suite.calibration.model({ ...d, A: 1.12, B: 0.071, kdp: 1.35, Pfeed, T, Qf, salinityFactor });
    return { Pfeed, T, Qf, salinityFactor, Qp: +(m.Qp * (1 + g.normal(0, 0.006))).toFixed(1), tdsP: +(m.tdsP * (1 + g.normal(0, 0.02))).toFixed(1), dP: +(m.dP * (1 + g.normal(0, 0.03))).toFixed(2) };
  });
}

export default suite;
