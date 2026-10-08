// Suite 4 — Flow in membranes, channels and equipment (CFD).
// Two-dimensional finite-volume solver on a staggered (MAC) Cartesian grid, uniform in x and optionally
// clustered toward the walls in y: SIMPLE/SIMPLEC pressure–velocity coupling with a conjugate-gradient pressure
// solver preconditioned by a re-used banded Cholesky factor, immersed solids by cell blocking, salt transport with
// solution–diffusion membrane walls (concentration polarisation), an energy equation, an algebraic
// mixing-length eddy viscosity, a Darcy–Forchheimer porous zone and Lagrangian particle tracking.
// Extensions: two-equation RANS closures (k–ε, k–ω, k–ω SST, explicit algebraic Reynolds stress), Smagorinsky LES,
// Navier-slip / symmetry walls, mass-flow and pressure inlets, conjugate heat transfer, wall crystallisation with a
// moment population balance, a growing fouling layer, user-defined scalar sources, a D2Q9 lattice-Boltzmann start
// field, interface (volume-fraction / level-set) and dispersed-phase transport, a Maxwell–Stefan film, a regression closure and
// one-dimensional compressible (Euler / Navier–Stokes) studies.
import { clamp, linspace, fmt, gci, rng, mean, solveLinear } from '../core/num.js';
import { density, viscosity, diffusivityNaCl, osmoticPressure, salinityFromTDS, cp as cpWater, conductivityThermal } from '../core/props.js';
import { sliceMesh, polylinesToSegments, rasterize } from '../core/io.js';

const LMH = 1 / 3.6e6; // L/m²·h → m/s
const KAPPA = 0.41, E_WALL = 9.0;

/**
 * Preconditioned conjugate gradients for a symmetric 5-point system on an n1 × n2 lattice:
 * dg·x_P − aE·x_E − aE(W)·x_W − aN·x_N − aN(S)·x_S = rhs. Modified incomplete Cholesky MIC(0).
 * Rows with dg = 0 (solid cells) are skipped. An additive coarse correction with one unknown per column
 * (block correction along the first index) removes the slow modes of strongly anisotropic grids.
 */
export function pcg5(n1, n2, aE, aN, dg, rhs, x, tol = 1e-6, maxIter = 200, W = null) {
  const n = n1 * n2;
  W = W || { r: new Float64Array(n), z: new Float64Array(n), s: new Float64Array(n), q: new Float64Array(n), pc: new Float64Array(n) };
  const { r, z, s, q, pc } = W;
  const cd = new Float64Array(n1), ce = new Float64Array(n1), cr = new Float64Array(n1), cw = new Float64Array(n1);
  for (let j = 0, P = 0; j < n2; j++) for (let i = 0; i < n1; i++, P++) {
    if (!(dg[P] > 0)) continue;
    cd[i] += dg[P] - 2 * (j < n2 - 1 ? aN[P] : 0); ce[i] += aE[P];
  }
  for (let i = 0; i < n1; i++) { const m = cd[i] - (i > 0 ? ce[i - 1] * cw[i - 1] : 0); cd[i] = m > 1e-300 ? m : 1e300; cw[i] = ce[i] / cd[i]; } // Thomas factors
  for (let P = 0; P < n; P++) {
    if (!(dg[P] > 0)) { pc[P] = 0; continue; }
    let e = dg[P];
    if (P > 0) { const t = aE[P - 1] * pc[P - 1]; e -= t * t + 0.97 * t * aN[P - 1] * pc[P - 1]; }
    if (P >= n1) { const t = aN[P - n1] * pc[P - n1]; e -= t * t + 0.97 * t * aE[P - n1] * pc[P - n1]; }
    if (e < 1e-7 * dg[P]) e = 1e-7 * dg[P];
    pc[P] = 1 / Math.sqrt(e);
  }
  const prec = () => {
    for (let P = 0; P < n; P++) {
      if (!pc[P]) { q[P] = 0; continue; }
      let t = r[P];
      if (P > 0) t += aE[P - 1] * pc[P - 1] * q[P - 1];
      if (P >= n1) t += aN[P - n1] * pc[P - n1] * q[P - n1];
      q[P] = t * pc[P];
    }
    for (let P = n - 1; P >= 0; P--) {
      if (!pc[P]) { z[P] = 0; continue; }
      let t = q[P];
      if (P + 1 < n) t += aE[P] * pc[P] * z[P + 1];
      if (P + n1 < n) t += aN[P] * pc[P] * z[P + n1];
      z[P] = t * pc[P];
    }
    // coarse (column-sum) correction: tridiagonal solve along the first index
    cr.fill(0);
    for (let j = 0, P = 0; j < n2; j++) for (let i = 0; i < n1; i++, P++) if (pc[P]) cr[i] += r[P];
    for (let i = 0; i < n1; i++) cr[i] = (cr[i] + (i > 0 ? ce[i - 1] * cr[i - 1] : 0)) / cd[i];
    for (let i = n1 - 2; i >= 0; i--) cr[i] += cw[i] * cr[i + 1];
    for (let j = 0, P = 0; j < n2; j++) for (let i = 0; i < n1; i++, P++) if (pc[P]) z[P] += cr[i];
  };
  let r0 = 0;
  for (let P = 0; P < n; P++) {
    if (!(dg[P] > 0)) { r[P] = 0; x[P] = 0; continue; }
    let ax = dg[P] * x[P];
    if (P + 1 < n) ax -= aE[P] * x[P + 1];
    if (P > 0) ax -= aE[P - 1] * x[P - 1];
    if (P + n1 < n) ax -= aN[P] * x[P + n1];
    if (P >= n1) ax -= aN[P - n1] * x[P - n1];
    r[P] = rhs[P] - ax; r0 += r[P] * r[P];
  }
  r0 = Math.sqrt(r0);
  if (!(r0 > 1e-300)) return { iters: 0, res: 0 };
  prec();
  let rz = 0;
  for (let P = 0; P < n; P++) { s[P] = z[P]; rz += r[P] * z[P]; }
  let it = 0, rn = r0;
  for (; it < maxIter; it++) {
    let sq = 0;
    for (let P = 0; P < n; P++) {
      if (!pc[P]) { q[P] = 0; continue; }
      let ax = dg[P] * s[P];
      if (P + 1 < n) ax -= aE[P] * s[P + 1];
      if (P > 0) ax -= aE[P - 1] * s[P - 1];
      if (P + n1 < n) ax -= aN[P] * s[P + n1];
      if (P >= n1) ax -= aN[P - n1] * s[P - n1];
      q[P] = ax; sq += s[P] * ax;
    }
    if (!(Math.abs(sq) > 1e-300)) break;
    const al = rz / sq;
    rn = 0;
    for (let P = 0; P < n; P++) { x[P] += al * s[P]; r[P] -= al * q[P]; rn += r[P] * r[P]; }
    rn = Math.sqrt(rn);
    if (rn <= tol * r0) { it++; break; }
    prec();
    let rz2 = 0;
    for (let P = 0; P < n; P++) rz2 += r[P] * z[P];
    const be = rz2 / rz; rz = rz2;
    for (let P = 0; P < n; P++) s[P] = z[P] + be * s[P];
  }
  return { iters: it, res: rn / r0 };
}

/**
 * Conjugate gradients preconditioned with a banded Cholesky factorisation for the same 5-point system as pcg5
 * (dg·x_P − aE·x_E − aE(W)·x_W − aN·x_N − aN(S)·x_S = rhs; rows with dg = 0 are skipped).
 * The factor is kept between calls: while the coefficients drift slowly (successive SIMPLE iterations or time
 * steps) the old factor is an excellent preconditioner and the solve takes one to three iterations; it is renewed
 * when the iteration count shows that it has aged (rentFactor > 1 keeps it longer, for systems that drift every
 * step). The unknowns are ordered along the shorter grid direction, so the band width is min(n1, n2). Returns null
 * from the constructor when the band storage would be too large.
 */
export function bandSolver(n1, n2, maxStore = 9e6, rentFactor = 1) {
  const n = n1 * n2, swap = n2 < n1, b = swap ? n2 : n1, w = b + 1;
  if (n * w > maxStore) return null;
  const Lf = new Float64Array(n * w), perm = swap ? new Int32Array(n) : null, r = new Float64Array(n), z = new Float64Array(n), s = new Float64Array(n), q = new Float64Array(n), t = new Float64Array(n);
  if (swap) for (let j = 0, P = 0; j < n2; j++) for (let i = 0; i < n1; i++, P++) perm[P] = i * n2 + j;
  const rent = rentFactor * Math.max(4, b / 3); // a factorisation costs about b/3 preconditioned iterations: renew once that many extra iterations were spent
  let have = false, age = 0, extra = 0;
  const S = { factors: 0, solves: 0, iters: 0 };
  const factor = (aE, aN, dg) => {
    Lf.fill(0);
    for (let j = 0, P = 0; j < n2; j++) for (let i = 0; i < n1; i++, P++) { // lower band of A in the solver ordering
      const m = (swap ? perm[P] : P) * w;
      if (!(dg[P] > 0)) { Lf[m + b] = 1; continue; }
      Lf[m + b] = dg[P];
      if (swap) { if (j > 0 && dg[P - n1] > 0) Lf[m + b - 1] = -aN[P - n1]; if (i > 0 && dg[P - 1] > 0) Lf[m] = -aE[P - 1]; }
      else { if (i > 0 && dg[P - 1] > 0) Lf[m + b - 1] = -aE[P - 1]; if (j > 0 && dg[P - n1] > 0) Lf[m] = -aN[P - n1]; }
    }
    for (let k = 0; k < n; k++) {
      const rk = k * w + b - k, c0 = k > b ? k - b : 0; // entry (k, c) sits at rk + c
      for (let c = c0; c < k; c++) {
        const rc = c * w + b - c;
        let a = Lf[rk + c], a1 = 0, m = c0;
        for (; m + 1 < c; m += 2) { a -= Lf[rk + m] * Lf[rc + m]; a1 -= Lf[rk + m + 1] * Lf[rc + m + 1]; }
        if (m < c) a -= Lf[rk + m] * Lf[rc + m];
        Lf[rk + c] = (a + a1) * Lf[rc + c]; // the diagonal holds the reciprocal pivot
      }
      let d = Lf[rk + k]; const d0 = d;
      for (let m = c0; m < k; m++) { const a = Lf[rk + m]; d -= a * a; }
      Lf[rk + k] = 1 / Math.sqrt(d > 1e-12 * d0 ? d : d0 > 0 ? d0 : 1);
    }
    have = true; age = 0; extra = 0; S.factors++;
  };
  const prec = (src, dst) => { // dst = (L Lᵀ)⁻¹ src
    if (swap) for (let P = 0; P < n; P++) t[perm[P]] = src[P]; else t.set(src);
    for (let k = 0; k < n; k++) {
      const rk = k * w + b - k;
      let a0 = t[k], a1 = 0, a2 = 0, a3 = 0, m = k > b ? k - b : 0;
      for (; m + 3 < k; m += 4) { const g = rk + m; a0 -= Lf[g] * t[m]; a1 -= Lf[g + 1] * t[m + 1]; a2 -= Lf[g + 2] * t[m + 2]; a3 -= Lf[g + 3] * t[m + 3]; }
      for (; m < k; m++) a0 -= Lf[rk + m] * t[m];
      t[k] = (a0 + a1 + a2 + a3) * Lf[rk + k];
    }
    for (let k = n - 1; k >= 0; k--) {
      const rk = k * w + b - k, a = t[k] * Lf[rk + k];
      t[k] = a;
      if (a !== 0) for (let m = k > b ? k - b : 0; m < k; m++) t[m] -= Lf[rk + m] * a;
    }
    if (swap) for (let P = 0; P < n; P++) dst[P] = t[perm[P]]; else dst.set(t);
  };
  const mul = (aE, aN, dg, x, y) => { // y = A x, returns x·y
    let dot = 0;
    for (let P = 0; P < n; P++) {
      if (!(dg[P] > 0)) { y[P] = 0; continue; }
      let ax = dg[P] * x[P];
      if (P + 1 < n) ax -= aE[P] * x[P + 1];
      if (P > 0) ax -= aE[P - 1] * x[P - 1];
      if (P + n1 < n) ax -= aN[P] * x[P + n1];
      if (P >= n1) ax -= aN[P - n1] * x[P - n1];
      y[P] = ax; dot += x[P] * ax;
    }
    return dot;
  };
  /** Solve to a relative residual tol (or to the absolute residual norm absRes, whichever is reached first). zero = true: x holds no starting guess and is overwritten. */
  S.solve = (aE, aN, dg, rhs, x, tol = 1e-6, maxIter = 60, zero = false, absRes = 0) => {
    S.solves++;
    let total = 0;
    for (let pass = 0; pass < 2; pass++) {
      if (!have || age < 0) factor(aE, aN, dg);
      let r0 = 0;
      if (zero && pass === 0) for (let P = 0; P < n; P++) { x[P] = 0; if (dg[P] > 0) { r[P] = rhs[P]; r0 += r[P] * r[P]; } else r[P] = 0; }
      else { mul(aE, aN, dg, x, q); for (let P = 0; P < n; P++) { if (!(dg[P] > 0)) { r[P] = 0; x[P] = 0; continue; } r[P] = rhs[P] - q[P]; r0 += r[P] * r[P]; } }
      r0 = Math.sqrt(r0);
      if (!(r0 > 1e-300) || r0 <= absRes) return { iters: total, res: 0 };
      const goal = Math.max(tol * r0, absRes);
      prec(r, z);
      let rz = 0, it = 0, rn = r0;
      for (let P = 0; P < n; P++) { s[P] = z[P]; rz += r[P] * z[P]; }
      const cap = age === 0 ? 4 : maxIter;
      for (; it < cap; it++) {
        const sq = mul(aE, aN, dg, s, q);
        if (!(Math.abs(sq) > 1e-300)) break;
        const al = rz / sq;
        rn = 0;
        for (let P = 0; P < n; P++) { x[P] += al * s[P]; r[P] -= al * q[P]; rn += r[P] * r[P]; }
        rn = Math.sqrt(rn);
        if (rn <= goal) { it++; break; }
        prec(r, z);
        let rz2 = 0;
        for (let P = 0; P < n; P++) rz2 += r[P] * z[P];
        const be = rz2 / rz; rz = rz2;
        for (let P = 0; P < n; P++) s[P] = z[P] + be * s[P];
      }
      total += it; S.iters += it;
      if (rn <= goal || age === 0 || !Number.isFinite(rn)) { extra += it > 1 ? it - 1 : 0; age = extra >= rent ? -1 : age + 1; return { iters: total, res: rn / r0 }; }
      age = -1; // the aged factor did not converge: renew it and continue from the current iterate
    }
    return { iters: total, res: 1 };
  };
  S.reset = () => { have = false; };
  /** Euclidean norm of rhs − A x over the active rows. */
  S.residual = (aE, aN, dg, rhs, x) => { mul(aE, aN, dg, x, q); let a = 0; for (let P = 0; P < n; P++) if (dg[P] > 0) { const e = rhs[P] - q[P]; a += e * e; } return Math.sqrt(a); };
  return S;
}

/**
 * Source terms of the differential Reynolds-stress transport model (Launder–Reece–Rodi, isotropisation of
 * production, with the Gibson–Launder wall-reflection terms) for a planar mean flow. Stresses are kinematic:
 *   D R_ij/Dt = d_ij + P_ij + φ_ij − ⅔ ε δ_ij,   P_ij = −(R_ik ∂U_j/∂x_k + R_jk ∂U_i/∂x_k)   (exact),
 *   φ_ij = −C1 (ε/k)(R_ij − ⅔ k δ_ij) − C2 (P_ij − ⅔ P δ_ij) + φ_ij,w,
 *   φ_ij,w = [C1′ (ε/k)(R_km n_k n_m δ_ij − 1.5 R_ik n_k n_j − 1.5 R_jk n_k n_i)
 *             + C2′ (φ_km,2 n_k n_m δ_ij − 1.5 φ_ik,2 n_k n_j − 1.5 φ_jk,2 n_k n_i)] f,  f = k^1.5/(2.5 ε d).
 * Writes into out[0..3] the sources of (u′u′, v′v′, w′w′, u′v′) without the linear return term −C1 (ε/k) R_ij,
 * which the callers treat implicitly, and into out[4] the production of k. n = unit wall normal, f = 0 away from walls.
 */
export const RSM = { C1: 1.8, C2: 0.6, C1w: 0.5, C2w: 0.3, Ce1: 1.44, Ce2: 1.92, Cs: 0.22, Ce: 0.18, uv: 0.255, tt: 1.098, nn: 0.247, ww: 0.655 }; // Cs, Ce: Daly–Harlow diffusion; uv…ww: log-layer stress ÷ k (Gibson & Launder 1978)
export function rsmSource(uu, vv, ww, uv, ux, uy, vx, vy, e, k, n1, n2, f, out) {
  const { C1, C2, C1w, C2w } = RSM;
  const Puu = -2 * (uu * ux + uv * uy), Pvv = -2 * (uv * vx + vv * vy), Puv = -(uu * vx + vv * uy + uv * (ux + vy)), Pk = 0.5 * (Puu + Pvv);
  const fuu = -C2 * (Puu - (2 / 3) * Pk), fvv = -C2 * (Pvv - (2 / 3) * Pk), fww = C2 * (2 / 3) * Pk, fuv = -C2 * Puv, iso = (2 / 3) * (C1 - 1) * e;
  let s0 = Puu + fuu + iso, s1 = Pvv + fvv + iso, s2 = fww + iso, s3 = Puv + fuv;
  if (f > 0) {
    const ek = (C1w * e * f) / k, c2 = C2w * f;
    const Rn1 = uu * n1 + uv * n2, Rn2 = uv * n1 + vv * n2, Rnn = Rn1 * n1 + Rn2 * n2, Fn1 = fuu * n1 + fuv * n2, Fn2 = fuv * n1 + fvv * n2, Fnn = Fn1 * n1 + Fn2 * n2;
    s0 += ek * (Rnn - 3 * Rn1 * n1) + c2 * (Fnn - 3 * Fn1 * n1);
    s1 += ek * (Rnn - 3 * Rn2 * n2) + c2 * (Fnn - 3 * Fn2 * n2);
    s2 += ek * Rnn + c2 * Fnn;
    s3 += -1.5 * ek * (Rn1 * n2 + Rn2 * n1) - 1.5 * c2 * (Fn1 * n2 + Fn2 * n1);
  }
  out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3; out[4] = Pk;
  return out;
}

/**
 * Homogeneous shear flow dU/dy = S integrated with the same stress-transport model (no walls, no diffusion):
 * dR_ij/dt = P_ij + φ_ij − ⅔ ε δ_ij, dε/dt = (ε/k)(C_ε1 P − C_ε2 ε). Returns the anisotropy a_ij = R_ij/k − ⅔ δ_ij
 * reached after the dimensionless time S·t = St, with the analytical fixed point of the model for comparison.
 */
export function rsmHomogeneousShear({ S = 1, k0 = 1, eps0 = 0.3, St = 60, steps = 6000 } = {}) {
  const { C1, C2, Ce1, Ce2 } = RSM, o = new Float64Array(5);
  const rhsF = (y) => { const k = 0.5 * (y[0] + y[1] + y[2]), e = y[4]; rsmSource(y[0], y[1], y[2], y[3], 0, S, 0, 0, e, k, 0, 0, 0, o); const c = (C1 * e) / k; return [o[0] - c * y[0], o[1] - c * y[1], o[2] - c * y[2], o[3] - c * y[3], (e / k) * (Ce1 * o[4] - Ce2 * e)]; };
  let y = [(2 / 3) * k0, (2 / 3) * k0, (2 / 3) * k0, 0, eps0];
  const dt = St / S / steps, ax = (a, b, c) => a.map((x, i) => x + c * b[i]);
  for (let q = 0; q < steps; q++) { const k1 = rhsF(y), k2 = rhsF(ax(y, k1, dt / 2)), k3 = rhsF(ax(y, k2, dt / 2)), k4 = rhsF(ax(y, k3, dt)); y = y.map((x, i) => x + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])); }
  const k = 0.5 * (y[0] + y[1] + y[2]), Pe = (-y[3] * S) / y[4], PeT = (Ce2 - 1) / (Ce1 - 1), g = (1 - C2) / (C1 - 1 + PeT);
  return { a11: y[0] / k - 2 / 3, a22: y[1] / k - 2 / 3, a33: y[2] / k - 2 / 3, a12: y[3] / k, PoverEps: Pe, k, eps: y[4],
    theory: { PoverEps: PeT, a11: (4 / 3) * g * PeT, a22: -(2 / 3) * g * PeT, a33: -(2 / 3) * g * PeT, a12: -Math.sqrt(g * PeT * (2 / 3 - (2 / 3) * g * PeT)) } };
}

/** Wall-normal grid: tanh clustering toward both walls. ratio = centre-cell height ÷ wall-cell height. */
export function yGrid(H, ny, ratio = 1) {
  const yf = new Float64Array(ny + 1), yc = new Float64Array(ny), dy = new Float64Array(ny), dyc = new Float64Array(ny + 1);
  const be = ratio > 1.001 ? Math.acosh(Math.sqrt(ratio)) : 0;
  for (let j = 0; j <= ny; j++) { const s = j / ny; yf[j] = be ? 0.5 * H * (1 + Math.tanh(be * (2 * s - 1)) / Math.tanh(be)) : H * s; }
  for (let j = 0; j < ny; j++) { dy[j] = yf[j + 1] - yf[j]; yc[j] = 0.5 * (yf[j] + yf[j + 1]); }
  dyc[0] = 0.5 * dy[0]; dyc[ny] = 0.5 * dy[ny - 1];
  for (let j = 1; j < ny; j++) dyc[j] = yc[j] - yc[j - 1];
  return { yf, yc, dy, dyc };
}

const stencil = (n1, n2) => {
  const f = (m) => new Float64Array(m), n = n1 * n2;
  return { n1, n2, Fx: f((n1 + 1) * n2), Fy: f(n1 * (n2 + 1)), Dx: f((n1 + 1) * n2), Dy: f(n1 * (n2 + 1)), aW: f(n), aE: f(n), aS: f(n), aN: f(n), aP: f(n), b: f(n), bW: f(n2), bE: f(n2), bS: f(n1), bN: f(n1), t1: f(Math.max(n1, n2)), t2: f(Math.max(n1, n2)) };
};

/**
 * Convection–diffusion coefficients for a lattice of control volumes. Fx/Fy are face mass (or volume)
 * fluxes, Dx/Dy face conductances; outer faces use the boundary values bW/bE/bS/bN.
 * scheme: 'upwind' | 'hybrid' | 'quick' (bounded QUICK as a deferred correction on first-order upwind).
 */
function assemble(S, phi, scheme, excl) {
  const { n1, n2, Fx, Fy, Dx, Dy, aW, aE, aS, aN, aP, b, bW, bE, bS, bN } = S, hyb = scheme === 'hybrid';
  for (let j = 0, P = 0; j < n2; j++) for (let i = 0; i < n1; i++, P++) {
    const kx = j * (n1 + 1) + i, fw = Fx[kx], fe = -Fx[kx + 1], fs = Fy[P], fn = -Fy[P + n1];
    let cw, ce, cs, cn;
    if (hyb) { cw = Math.max(fw, Dx[kx] + 0.5 * fw, 0); ce = Math.max(fe, Dx[kx + 1] + 0.5 * fe, 0); cs = Math.max(fs, Dy[P] + 0.5 * fs, 0); cn = Math.max(fn, Dy[P + n1] + 0.5 * fn, 0); }
    else { cw = Dx[kx] + (fw > 0 ? fw : 0); ce = Dx[kx + 1] + (fe > 0 ? fe : 0); cs = Dy[P] + (fs > 0 ? fs : 0); cn = Dy[P + n1] + (fn > 0 ? fn : 0); }
    let bb = 0;
    if (i > 0) aW[P] = cw; else { aW[P] = 0; bb += cw * bW[j]; }
    if (i < n1 - 1) aE[P] = ce; else { aE[P] = 0; bb += ce * bE[j]; }
    if (j > 0) aS[P] = cs; else { aS[P] = 0; bb += cs * bS[i]; }
    if (j < n2 - 1) aN[P] = cn; else { aN[P] = 0; bb += cn * bN[i]; }
    aP[P] = cw + ce + cs + cn; b[P] = bb;
  }
  if (scheme !== 'quick') return;
  // bounded QUICK: φ_f = φ_C + ½ψ(r)(φ_D − φ_C), ψ = max(0, min(2r, (3 + r)/4, 2)); correction is explicit
  const lim = (d1, d2) => { if (d1 * d2 <= 0) return 0; const a1 = Math.abs(d1), a2 = Math.abs(d2); return 0.5 * Math.sign(d2) * Math.min(2 * a1, 0.75 * a2 + 0.25 * a1, 2 * a2); };
  const face = (F, C, D, U) => { if (excl && (excl[C] || excl[D] || excl[U])) return; const c = F * lim(phi[C] - phi[U], phi[D] - phi[C]); b[C] -= c; b[D] += c; };
  for (let j = 0; j < n2; j++) for (let k = 1; k < n1; k++) {
    const F = Fx[j * (n1 + 1) + k], E = j * n1 + k, Wn = E - 1;
    if (F > 0) { if (k >= 2) face(F, Wn, E, Wn - 1); } else if (F < 0 && k + 1 < n1) face(-F, E, Wn, E + 1);
  }
  for (let jj = 1; jj < n2; jj++) for (let i = 0; i < n1; i++) {
    const F = Fy[jj * n1 + i], N = jj * n1 + i, Sn = N - n1;
    if (F > 0) { if (jj >= 2) face(F, Sn, N, Sn - n1); } else if (F < 0 && jj + 1 < n2) face(-F, N, Sn, N + n1);
  }
}

/** Line-by-line TDMA: columns (lines along the second index) swept forward and back, then rows once. */
function lineSolve(S, phi, sweeps = 1) {
  const { n1, n2, aW, aE, aS, aN, aP, b, t1: cp, t2: dp } = S, top = (n2 - 1) * n1, e1 = n1 - 1;
  for (let s = 0; s < sweeps; s++) {
    for (let pass = 0; pass < 3; pass++) {
      if (pass !== 1) { // columns, forward (pass 0) or backward (pass 2); aS is zero in the first row and aN in the last
        for (let ii = 0; ii < n1; ii++) {
          const i = pass ? e1 - ii : ii, hasW = i > 0, hasE = i < e1;
          let c = 0, g = 0;
          for (let j = 0, P = i; j < n2; j++, P += n1) {
            let d = b[P];
            if (hasW) d += aW[P] * phi[P - 1];
            if (hasE) d += aE[P] * phi[P + 1];
            const inv = 1 / (aP[P] - aS[P] * c);
            c = aN[P] * inv; g = (d + aS[P] * g) * inv; cp[j] = c; dp[j] = g;
          }
          let x = dp[n2 - 1]; phi[top + i] = x;
          for (let j = n2 - 2, P = top + i - n1; j >= 0; j--, P -= n1) { x = dp[j] + cp[j] * x; phi[P] = x; }
        }
      } else {
        for (let j = 0; j < n2; j++) {
          const r0 = j * n1, hasS = j > 0, hasN = j < n2 - 1;
          let c = 0, g = 0;
          for (let i = 0, P = r0; i < n1; i++, P++) {
            let d = b[P];
            if (hasS) d += aS[P] * phi[P - n1];
            if (hasN) d += aN[P] * phi[P + n1];
            const inv = 1 / (aP[P] - aW[P] * c);
            c = aE[P] * inv; g = (d + aW[P] * g) * inv; cp[i] = c; dp[i] = g;
          }
          let x = dp[e1]; phi[r0 + e1] = x;
          for (let i = e1 - 1, P = r0 + e1 - 1; i >= 0; i--, P--) { x = dp[i] + cp[i] * x; phi[P] = x; }
        }
      }
    }
  }
}

/**
 * Engine. o = { L, H, nx, ny, stretch, solid, rho, mu, Uin, inlet, scheme, steady, maxIter, tol, alphaU, alphaP,
 * cfl, tEnd, turb, porous, species, energy }. Returns the converged fields and wall data (SI units).
 */
export async function solveChannel(o, ctx) {
  const { nx, ny, L, H, rho, mu } = o, dx = L / nx, n = nx * ny, nu1 = nx + 1, nu = mu / rho;
  const { yf, yc, dy, dyc } = yGrid(H, ny, o.stretch || 1);
  const solid = o.solid || new Uint8Array(n), scheme = o.scheme || 'hybrid';
  // closures and wall types: tm = 'ml' | 'ke' | 'kw' | 'sst' | 'earsm' | 'rsm' | 'les'; walls 'noslip' | 'slip' (Navier) | 'sym'
  const tm = o.turb === true ? 'ml' : o.turb || null, rsm = tm === 'rsm', epsM = tm === 'ke' || rsm, twoEq = epsM || tm === 'kw' || tm === 'sst' || tm === 'earsm';
  const wallB = o.wallB || 'noslip', wallT = o.wallT || 'noslip', bSlip = Math.max(0, o.slipLen || 0), cv = o.creeping ? 0 : 1;
  const u = new Float64Array(nu1 * ny), v = new Float64Array(nx * (ny + 1)), p = new Float64Array(n), pp = new Float64Array(n);
  const ublk = new Uint8Array(nu1 * ny), vblk = new Uint8Array(nx * (ny + 1));
  for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) ublk[j * nu1 + i] = i === 0 || i === nx || solid[j * nx + i - 1] || solid[j * nx + i] ? 1 : 0;
  for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) vblk[j * nx + i] = j === 0 || j === ny || solid[(j - 1) * nx + i] || solid[j * nx + i] ? 1 : 0;

  // inlet profile over the open part of the first column; Uin = mean velocity of the open inlet
  const uin = new Float64Array(ny);
  let ja = ny, jb = -1, openIn = 0;
  for (let j = 0; j < ny; j++) if (!solid[j * nx]) { ja = Math.min(ja, j); jb = Math.max(jb, j); openIn += dy[j]; }
  if (jb < 0) throw new Error('The inlet plane is completely blocked by the geometry.');
  const y0 = yf[ja], hIn = yf[jb + 1] - y0;
  let Qin = 0;
  for (let j = ja; j <= jb; j++) if (!solid[j * nx]) { const s = (yc[j] - y0) / hIn; uin[j] = o.inlet === 'uniform' ? 1 : 6 * s * (1 - s); Qin += uin[j] * dy[j]; }
  const Qt = o.Uin * openIn;
  for (let j = 0; j < ny; j++) uin[j] *= Qin > 0 ? Qt / Qin : 0;
  Qin = Qt;
  let Uref = Math.abs(Qin) / H || 1e-12;
  for (let i = 1; i <= nx; i++) { // initial guess: plug flow through the open height of every column
    const isOpen = (j) => (i === nx ? !solid[j * nx + nx - 1] : !ublk[j * nu1 + i]);
    let open = 0;
    for (let j = 0; j < ny; j++) if (isOpen(j)) open += dy[j];
    for (let j = 0; j < ny; j++) u[j * nu1 + i] = isOpen(j) && open > 0 ? Qin / open : 0;
  }
  if (o.init) { // start from a field supplied by another engine (lattice-Boltzmann hybrid)
    for (let k = 0; k < u.length; k++) if (!ublk[k] && Number.isFinite(o.init.u[k])) u[k] = o.init.u[k];
    for (let k = 0; k < v.length; k++) if (!vblk[k] && Number.isFinite(o.init.v[k])) v[k] = o.init.v[k];
    if (o.init.p) for (let P = 0; P < n; P++) if (!solid[P] && Number.isFinite(o.init.p[P])) p[P] = o.init.p[P];
  }
  for (let j = 0; j < ny; j++) u[j * nu1] = uin[j];

  // membrane / wall permeation velocities (m/s, positive out of the channel)
  const Jb = new Float64Array(nx), Jt = new Float64Array(nx);
  // The walls cannot take out more water than the inlet supplies: while the permeation estimate exceeds 90 % of the feed the wall
  // velocities are scaled down to that share (permCap), and a solution that still needs the cap at the end is rejected with a message.
  let permCap = false, permQ = 0;
  const setWallV = () => {
    let Qp = 0;
    for (let i = 0; i < nx; i++) { v[i] = solid[i] ? 0 : -Jb[i]; v[ny * nx + i] = solid[(ny - 1) * nx + i] ? 0 : Jt[i]; Qp += (v[ny * nx + i] - v[i]) * dx; }
    if (!Number.isFinite(Qp)) { permCap = true; return; }
    permQ = Qp; permCap = Qin > 0 && Qp > 0.9 * Qin;
    if (permCap) { const sc = (0.9 * Qin) / Qp; for (let i = 0; i < nx; i++) { v[i] *= sc; v[ny * nx + i] *= sc; } }
  };

  const permErr = () => new Error(`The membrane walls would take more water out of this section than the cross-flow can supply: the permeate flow of ${fmt(permQ, 3)} m²/s is ${fmt((100 * permQ) / Qin, 3)} % of the feed flow of ${fmt(Qin, 3)} m²/s (per metre of width), i.e. a recovery above 90 % within the simulated length. Raise the cross-flow velocity, shorten the domain, or lower the trans-membrane pressure or the water permeability.`);
  // effective viscosity (cells) and wall distance for the mixing-length closure
  const mue = new Float64Array(n).fill(mu), dist = new Float64Array(n);
  if (o.turb) {
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) dist[j * nx + i] = solid[j * nx + i] ? 0 : Math.min(wallB === 'sym' ? 4 * H : yc[j], wallT === 'sym' ? 4 * H : H - yc[j]);
    for (let pass = 0; pass < 3; pass++) {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const P = j * nx + i; if (solid[P]) continue; if (i > 0) dist[P] = Math.min(dist[P], dist[P - 1] + (solid[P - 1] ? 0.5 * dx : dx)); if (j > 0) dist[P] = Math.min(dist[P], dist[P - nx] + (solid[P - nx] ? 0.5 * dy[j] : dyc[j])); }
      for (let j = ny - 1; j >= 0; j--) for (let i = nx - 1; i >= 0; i--) { const P = j * nx + i; if (solid[P]) continue; if (i < nx - 1) dist[P] = Math.min(dist[P], dist[P + 1] + (solid[P + 1] ? 0.5 * dx : dx)); if (j < ny - 1) dist[P] = Math.min(dist[P], dist[P + nx] + (solid[P + nx] ? 0.5 * dy[j] : dyc[j + 1])); }
    }
  }
  const muCorner = (i, jf) => { // corner (i, jf) shared by cells (i−1..i, jf−1..jf)
    const i0 = i > 0 ? i - 1 : 0, i1 = i < nx ? i : nx - 1, j0 = jf > 0 ? jf - 1 : 0, j1 = jf < ny ? jf : ny - 1;
    return 0.25 * (mue[j0 * nx + i0] + mue[j0 * nx + i1] + mue[j1 * nx + i0] + mue[j1 * nx + i1]);
  };
  let utauG = Math.sqrt((nu * 6 * Uref) / H);
  const tauB = new Float64Array(nx), tauT = new Float64Array(nx), wfB = new Uint8Array(nx), wfT = new Uint8Array(nx);
  const wallShear = () => { // wall shear stress on the channel walls from the wall-adjacent cell
    for (let i = 0; i < nx; i++) {
      for (const top of [0, 1]) {
        const j = top ? ny - 1 : 0, P = j * nx + i, uc = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]), y = 0.5 * dy[j];
        const wt = top ? wallT : wallB;
        let tau = solid[P] || wt === 'sym' ? 0 : (mu * uc) / (y + (wt === 'slip' ? bSlip : 0)), wf = 0;
        if (o.turb && !solid[P] && wt !== 'sym') { const ut = uTau(Math.abs(uc), y); if ((y * ut) / nu > 11.6) { tau = rho * ut * ut * Math.sign(uc); wf = 1; } }
        (top ? wfT : wfB)[i] = wf;
        (top ? tauT : tauB)[i] = tau;
      }
    }
  };
  function uTau(ua, y) { // log-law friction velocity; falls back to the viscous sub-layer for y⁺ < 11.6
    let ut = Math.sqrt((nu * ua) / y);
    if ((y * ut) / nu <= 11.6) return ut;
    for (let k = 0; k < 6; k++) ut = (KAPPA * ua) / Math.log(Math.max(1.5, (E_WALL * y * ut) / nu));
    return ut;
  }
  const updateTurb = () => {
    wallShear();
    let s = 0, m = 0;
    for (let i = 0; i < nx; i++) { if (!solid[i] && wallB !== 'sym') { s += Math.abs(tauB[i]); m++; } if (!solid[(ny - 1) * nx + i] && wallT !== 'sym') { s += Math.abs(tauT[i]); m++; } }
    utauG = Math.sqrt(Math.max(s / Math.max(m, 1), 1e-30) / rho);
    if (twoEq) { turbStep(); return; }
    const uc = (i, j) => 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]), vc = (i, j) => 0.5 * (v[j * nx + i] + v[(j + 1) * nx + i]);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const P = j * nx + i;
      if (solid[P]) { mue[P] = mu; continue; }
      const uy = ((j < ny - 1 ? uc(i, j + 1) : 0) - (j > 0 ? uc(i, j - 1) : 0)) / ((j < ny - 1 ? yc[j + 1] : H) - (j > 0 ? yc[j - 1] : 0));
      const vx = (vc(Math.min(i + 1, nx - 1), j) - vc(Math.max(i - 1, 0), j)) / (2 * dx);
      const ux = (u[j * nu1 + i + 1] - u[j * nu1 + i]) / dx, vy = (v[(j + 1) * nx + i] - v[j * nx + i]) / dy[j];
      const S = Math.sqrt(2 * (ux * ux + vy * vy) + (uy + vx) * (uy + vx)), d = dist[P];
      const lm = Math.min(KAPPA * d, tm === 'les' ? (o.cSmag ?? 0.17) * Math.sqrt(dx * dy[j]) : 0.09 * H) * (1 - Math.exp((-d * utauG) / nu / 26));
      if (tm === 'les') mue[P] = mu + rho * Math.min(lm * lm * S, 2e4 * nu); // Smagorinsky sub-grid viscosity (C_s Δ)²|S|, Δ = √(ΔxΔy)
      else mue[P] = 0.5 * mue[P] + 0.5 * (mu + rho * Math.min(lm * lm * S, 2e4 * nu));
    }
  };

  // ---- two-equation RANS closures: standard k–ε, Wilcox k–ω, Menter k–ω SST and an explicit algebraic
  // Reynolds-stress closure (Wallin–Johansson, 2-D mean flow) on k–ω. Cell-centred k and ε/ω with production,
  // destruction and diffusion; log-law wall functions where y⁺ > 11.6, k = 0 and the viscous ω/ε value otherwise.
  const CMU = 0.09, C25 = CMU ** 0.25, C75 = CMU ** 0.75, K_MIN = 1e-14;
  const lT = o.lTurb || 0.07 * 2 * H;
  let kIn = Math.max(1.5 * ((o.tuIn ?? 0.05) * Uref) ** 2, K_MIN), eIn = (C75 * kIn ** 1.5) / lT, wIn = Math.sqrt(kIn) / (C25 * lT);
  let tk = null, te = null, St = null, S2 = null, PkA = null, dK = null, dE = null, nearW = null, cmuE = null, exx = null, exy = null, F1 = null, crs = null, kOld = null, eOld = null, tRdt = 0, rs4 = null, rsS = null, rsOld = null, wn1 = null, wn2 = null, dKy = null, dEy = null, rxx = null, ryy = null, rxyU = null, rxyV = null, utw = null, gUy = null, gVx = null;
  const rsO = new Float64Array(5);
  const gr = new Float64Array(4); // ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y at a cell centre
  const symB = wallB === 'sym', symT = wallT === 'sym';
  const grads = (i, j) => {
    const k = j * nu1 + i, P = j * nx + i, ie = i < nx - 1 ? 1 : 0, iw = i > 0 ? 1 : 0, uP = 0.5 * (u[k] + u[k + 1]);
    gr[0] = (u[k + 1] - u[k]) / dx; gr[3] = (v[P + nx] - v[P]) / dy[j];
    const uN = j < ny - 1 ? 0.5 * (u[k + nu1] + u[k + nu1 + 1]) : symT ? uP : 0, uS = j > 0 ? 0.5 * (u[k - nu1] + u[k - nu1 + 1]) : symB ? uP : 0;
    gr[1] = (uN - uS) / ((j < ny - 1 ? yc[j + 1] : symT ? yc[j] : H) - (j > 0 ? yc[j - 1] : symB ? yc[j] : 0));
    gr[2] = (0.5 * (v[P + ie] + v[P + ie + nx]) - 0.5 * (v[P - iw] + v[P - iw + nx])) / (2 * dx);
  };
  const gradC = (a, P, i, j, ax) => { // central gradient of a cell field, one-sided next to solids and boundaries
    if (ax === 0) { const e = i < nx - 1 && !solid[P + 1] ? 1 : 0, w = i > 0 && !solid[P - 1] ? 1 : 0; return e + w ? (a[P + e] - a[P - w]) / ((e + w) * dx) : 0; }
    const nn = j < ny - 1 && !solid[P + nx] ? 1 : 0, ss = j > 0 && !solid[P - nx] ? 1 : 0;
    return nn + ss ? (a[P + nn * nx] - a[P - ss * nx]) / (yc[j + nn] - yc[j - ss]) : 0;
  };
  if (twoEq) {
    const f = () => new Float64Array(n);
    tk = f().fill(kIn); te = f().fill(epsM ? eIn : wIn); St = stencil(nx, ny); S2 = f(); PkA = f(); dK = f(); dE = f(); nearW = new Uint8Array(n); kOld = f(); eOld = f();
    if (tm === 'sst') { F1 = f().fill(1); crs = f(); }
    if (tm === 'earsm') { cmuE = f().fill(CMU); exx = f(); exy = f(); }
    if (rsm) { // transported stresses (u′u′, v′v′, w′w′, u′v′), their sources, and the unit normal of the nearest wall
      rs4 = [f().fill((2 / 3) * kIn), f().fill((2 / 3) * kIn), f().fill((2 / 3) * kIn), f()]; rsS = [f(), f(), f(), f()]; rsOld = [f(), f(), f(), f()]; wn1 = f(); wn2 = f(); dKy = f(); dEy = f(); rxx = f(); ryy = f(); rxyU = new Float64Array(nu1 * (ny + 1)); rxyV = new Float64Array(nu1 * (ny + 1)); utw = f(); gUy = f(); gVx = f();
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        if (solid[P]) continue;
        const dE1 = i < nx - 1 ? (dist[P + 1] - dist[P]) / dx : 0, dW1 = i > 0 ? (dist[P] - dist[P - 1]) / dx : 0, dN1 = j < ny - 1 ? (dist[P + nx] - dist[P]) / dyc[j + 1] : -1, dS1 = j > 0 ? (dist[P] - dist[P - nx]) / dyc[j] : 1;
        const g1 = Math.abs(dE1) > Math.abs(dW1) ? dE1 : dW1, g2 = Math.abs(dN1) > Math.abs(dS1) ? dN1 : dS1, gm = Math.hypot(g1, g2);
        if (gm > 1e-12) { wn1[P] = g1 / gm; wn2[P] = g2 / gm; } else wn2[P] = 1;
      }
    }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      nearW[P] = !solid[P] && ((j === 0 && wallB !== 'sym') || (j === ny - 1 && wallT !== 'sym') || (i > 0 && solid[P - 1]) || (i < nx - 1 && solid[P + 1]) || (j > 0 && solid[P - nx]) || (j < ny - 1 && solid[P + nx])) ? 1 : 0;
      if (!solid[P]) mue[P] = mu + rho * Math.min(epsM ? (CMU * kIn * kIn) / eIn : kIn / wIn, 1e5 * nu);
    }
  }
  const exC = (i, jf) => { if (jf <= 0 || jf >= ny) return 0; const i0 = i > 0 ? i - 1 : 0, i1 = i < nx ? i : nx - 1; return 0.25 * (exy[(jf - 1) * nx + i0] + exy[(jf - 1) * nx + i1] + exy[jf * nx + i0] + exy[jf * nx + i1]); };
  const asmT = (phi, dc, inVal, kWall, sch, dcy = dc) => { // dcy: separate diffusivity across the gap (anisotropic gradient diffusion of the stress model)
    for (let j = 0; j < ny; j++) for (let k = 0; k <= nx; k++) {
      const q = j * (nx + 1) + k, R = j * nx + k, Lc = R - 1;
      St.Fx[q] = rho * u[j * nu1 + k] * dy[j];
      St.Dx[q] = k === nx ? 0 : k === 0 ? (solid[R] || !(uin[j] > 0) ? 0 : (dc[R] * dy[j]) / (0.5 * dx)) : solid[Lc] || solid[R] ? 0 : (0.5 * (dc[Lc] + dc[R]) * dy[j]) / dx;
    }
    for (let jf = 0; jf <= ny; jf++) for (let i = 0; i < nx; i++) {
      const q = jf * nx + i;
      St.Fy[q] = rho * v[q] * dx;
      St.Dy[q] = jf === 0 || jf === ny || solid[q - nx] || solid[q] ? 0 : (0.5 * (dcy[q - nx] + dcy[q]) * dx) / dyc[jf];
    }
    St.bS.fill(0); St.bN.fill(0);
    if (kWall) for (let i = 0; i < nx; i++) {
      if (wallB !== 'sym' && !wfB[i] && !solid[i]) St.Dy[i] = (mu * dx) / (0.5 * dy[0]);
      if (wallT !== 'sym' && !wfT[i] && !solid[(ny - 1) * nx + i]) St.Dy[ny * nx + i] = (mu * dx) / (0.5 * dy[ny - 1]);
    }
    for (let j = 0; j < ny; j++) { St.bW[j] = o.inlet === 'periodic' ? phi[j * nx + nx - 1] : inVal; St.bE[j] = phi[j * nx + nx - 1]; }
    assemble(St, phi, sch);
  };
  const turbStep = () => {
    const isE = epsM, sst = tm === 'sst', al = tRdt ? 1 : 0.7, schT = scheme === 'upwind' ? 'upwind' : 'hybrid';
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) { S2[P] = 0; dK[P] = dE[P] = mu; continue; }
      grads(i, j);
      const ux = gr[0], uy = gr[1], vx = gr[2], vy = gr[3], mut = mue[P] - mu, k = Math.max(tk[P], K_MIN), w = te[P];
      S2[P] = 2 * (ux * ux + vy * vy) + (uy + vx) * (uy + vx);
      let sk = isE ? 1 : 0.5, se = isE ? 1 / 1.3 : 0.5;
      if (rsm) { // sources of the four stress equations from the current stresses, mean-velocity gradients and wall distance
        const kk = Math.max(0.5 * (rs4[0][P] + rs4[1][P] + rs4[2][P]), K_MIN), d = Math.max(dist[P], 1e-12), fw = Math.min(kk ** 1.5 / (2.5 * w * d), 1.5);
        rsmSource(rs4[0][P], rs4[1][P], rs4[2][P], rs4[3][P], ux, uy, vx, vy, w, kk, wn1[P], wn2[P], fw, rsO);
        rsS[0][P] = rsO[0]; rsS[1][P] = rsO[1]; rsS[2][P] = rsO[2]; rsS[3][P] = rsO[3]; PkA[P] = rho * rsO[4]; gUy[P] = uy; gVx[P] = vx;
        // generalised gradient diffusion (Daly & Harlow 1970), diagonal part: D_x = μ + ρ C (k/ε) u′u′, D_y = μ + ρ C (k/ε) v′v′
        const tke = Math.min((rho * kk) / w, (1e5 * mu) / Math.max(rs4[0][P], rs4[1][P], K_MIN));
        dK[P] = mu + RSM.Cs * tke * rs4[0][P]; dKy[P] = mu + RSM.Cs * tke * rs4[1][P]; dE[P] = mu + RSM.Ce * tke * rs4[0][P]; dEy[P] = mu + RSM.Ce * tke * rs4[1][P];
        continue;
      }
      if (sst) {
        const d = Math.max(dist[P], 1e-12), cdv = ((2 * rho * 0.856) / w) * (gradC(tk, P, i, j, 0) * gradC(te, P, i, j, 0) + gradC(tk, P, i, j, 1) * gradC(te, P, i, j, 1));
        const arg1 = Math.min(Math.max(Math.sqrt(k) / (CMU * w * d), (500 * nu) / (d * d * w)), (4 * rho * 0.856 * k) / (Math.max(cdv, 1e-10) * d * d)), f1 = Math.tanh(arg1 ** 4);
        F1[P] = f1; crs[P] = (1 - f1) * cdv; sk = 0.85 * f1 + 1 - f1; se = 0.5 * f1 + 0.856 * (1 - f1);
      }
      if (cmuE) { // explicit algebraic Reynolds stress: a = β1 S* + β4 (S*Ω* − Ω*S*), N from the cubic consistency condition
        const tau = Math.max(1 / (CMU * w), 6 * Math.sqrt(nu / (CMU * k * w))), s11 = 0.5 * (ux - vy) * tau, s12 = 0.5 * (uy + vx) * tau, w12 = 0.5 * (uy - vx) * tau;
        const IIs = 2 * (s11 * s11 + s12 * s12), IIo = -2 * w12 * w12, c1 = 1.8, P1 = ((c1 * c1) / 27 + 0.45 * IIs - (2 / 3) * IIo) * c1, P2 = P1 * P1 - ((c1 * c1) / 9 + 0.9 * IIs + (2 / 3) * IIo) ** 3;
        let N;
        if (P2 >= 0) { const sq = Math.sqrt(P2); N = c1 / 3 + Math.cbrt(P1 + sq) + Math.cbrt(P1 - sq); }
        else N = c1 / 3 + 2 * (P1 * P1 - P2) ** (1 / 6) * Math.cos(Math.acos(clamp(P1 / Math.sqrt(P1 * P1 - P2), -1, 1)) / 3);
        const den = N * N - 2 * IIo, b4 = -1.2 / den;
        cmuE[P] = 0.9 * cmuE[P] + 0.1 * clamp((0.6 * N) / den, 0.045, 0.18); // C_μ,eff = −β1/2, bounded and under-relaxed for robustness
        exx[P] = 0.5 * exx[P] + 0.5 * rho * k * (-2 * b4 * s12 * w12); exy[P] = 0.5 * exy[P] + 0.5 * rho * k * (2 * b4 * s11 * w12);
      }
      dK[P] = mu + sk * mut; dE[P] = mu + se * mut;
    }
    // k equation
    asmT(tk, dK, kIn, true, schT, rsm ? dKy : dK);
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) { St.aW[P] = St.aE[P] = St.aS[P] = St.aN[P] = 0; St.aP[P] = 1; St.b[P] = K_MIN; PkA[P] = 0; continue; }
      const vol = dx * dy[j], k = Math.max(tk[P], K_MIN), e = te[P], dest = isE ? (rho * e) / k : rho * CMU * e;
      let Pk = rsm ? clamp(PkA[P], 0, 10 * dest * k) : Math.min((mue[P] - mu) * S2[P], 10 * dest * k);
      if (j === 0 ? wfB[i] : j === ny - 1 ? wfT[i] : 0) { const tw = Math.abs(j === 0 ? tauB[i] : tauT[i]); Pk = (tw * tw) / (KAPPA * rho * C25 * Math.sqrt(k) * 0.5 * dy[j]); }
      PkA[P] = Pk;
      let ap = St.aP[P] + dest * vol;
      St.b[P] += Pk * vol;
      if (tRdt) { ap += tRdt * vol; St.b[P] += tRdt * vol * kOld[P]; }
      ap /= al; St.b[P] += (1 - al) * ap * tk[P]; St.aP[P] = ap;
    }
    lineSolve(St, tk, 1);
    for (let P = 0; P < n; P++) if (!(tk[P] > K_MIN)) tk[P] = K_MIN;
    if (rsm) {
      // wall-adjacent cells on a log-law wall: the friction velocity of the wall function fixes the stress level,
      // u_τ² = |τ_w|/ρ = −u′v′ (constant-stress layer) and k = u_τ²/0.255, or the transported k solved above where that is
      // larger (separation and reattachment, where τ_w passes through zero); elsewhere next to a surface the solved k is used
      for (let P = 0; P < n; P++) utw[P] = 0;
      for (let i = 0; i < nx; i++) { if (wfB[i] && !solid[i]) { utw[i] = Math.abs(tauB[i]) / rho; tk[i] = Math.max(utw[i] / RSM.uv, tk[i]); } const T = (ny - 1) * nx + i; if (wfT[i] && !solid[T] && ny > 1) { utw[T] = Math.abs(tauT[i]) / rho; tk[T] = Math.max(utw[T] / RSM.uv, tk[T]); } }
      // Reynolds-stress transport: convection and gradient diffusion are implicit, the linear return term
      // −C1 (ε/k) R_ij goes to the diagonal and negative normal-stress sources are linearised so the stresses stay positive.
      // Wall-adjacent cells take the log-layer stress levels (Gibson & Launder: 1.098 k, 0.247 k, 0.655 k, ∓0.255 k) in wall axes.
      for (let c = 0; c < 4; c++) {
        const R = rs4[c], Sc4 = rsS[c];
        asmT(R, dK, c < 3 ? (2 / 3) * kIn : 0, false, schT, dKy);
        for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
          if (solid[P] || nearW[P]) {
            let val = R[P];
            if (!solid[P]) {
              const n1 = wn1[P], n2 = wn2[P], t1 = n2, t2 = -n1, kw2 = tk[P], Rtt = RSM.tt * kw2, Rnn = RSM.nn * kw2;
              const utan = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]) * t1 + 0.5 * (v[P] + v[P + nx]) * t2, Rtn = utw[P] > 0 ? -utw[P] * (utan >= 0 ? 1 : -1) : (-RSM.uv * kw2 * utan) / (Math.abs(utan) + 0.2 * Math.sqrt(kw2) + 1e-300); // −u_τ² on a log-law wall; smooth in the sign of the tangential velocity elsewhere
              val = c === 0 ? Rtt * t1 * t1 + Rnn * n1 * n1 + 2 * Rtn * t1 * n1 : c === 1 ? Rtt * t2 * t2 + Rnn * n2 * n2 + 2 * Rtn * t2 * n2 : c === 2 ? RSM.ww * kw2 : Rtt * t1 * t2 + Rnn * n1 * n2 + Rtn * (t1 * n2 + t2 * n1);
            }
            St.aW[P] = St.aE[P] = St.aS[P] = St.aN[P] = 0; St.aP[P] = 1; St.b[P] = val;
            continue;
          }
          const vol = dx * dy[j], kk = Math.max(0.5 * (rs4[0][P] + rs4[1][P] + rs4[2][P]), K_MIN), src = rho * Sc4[P] * vol;
          let ap = St.aP[P] + ((RSM.C1 * rho * te[P]) / kk) * vol;
          if (c < 3 && src < 0) ap -= src / Math.max(R[P], K_MIN); else St.b[P] += src;
          if (tRdt) { ap += tRdt * vol; St.b[P] += tRdt * vol * rsOld[c][P]; }
          ap /= al; St.b[P] += (1 - al) * ap * R[P]; St.aP[P] = ap;
        }
        lineSolve(St, R, 1);
      }
      for (let P = 0; P < n; P++) { // realisability: positive normal stresses and the Schwarz inequality for the shear stress
        if (solid[P]) continue;
        for (let c = 0; c < 3; c++) if (!(rs4[c][P] > (2 / 3) * K_MIN)) rs4[c][P] = (2 / 3) * K_MIN;
        const lim2 = 0.98 * Math.sqrt(rs4[0][P] * rs4[1][P]);
        if (!(Math.abs(rs4[3][P]) <= lim2)) rs4[3][P] = rs4[3][P] > 0 ? lim2 : -lim2;
        if (!nearW[P]) tk[P] = 0.5 * (rs4[0][P] + rs4[1][P] + rs4[2][P]);
      }
    }
    // ε or ω equation
    asmT(te, dE, isE ? eIn : wIn, false, schT, rsm ? dEy : dE);
    const eMin = 1e-9 * (isE ? eIn : wIn);
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      const vol = dx * dy[j], k = Math.max(tk[P], K_MIN), e = te[P];
      if (solid[P] || nearW[P]) { // fixed value: solids keep theirs, wall-adjacent cells take the viscous / log-layer value
        let val = e;
        if (!solid[P]) {
          const d = Math.max(dist[P], 1e-12);
          if (rsm && utw[P] > 0) val = Math.max(utw[P], RSM.uv * k) ** 1.5 / (KAPPA * d); // ε = u_τ³/(κ y) in the log layer
          else if (isE) val = Math.max((C75 * k ** 1.5) / (KAPPA * d), (2 * nu * k) / (d * d));
          else { const wv = (6 * nu) / (0.075 * d * d), wl = Math.sqrt(k) / (C25 * KAPPA * d); val = Math.sqrt(wv * wv + wl * wl); }
        }
        St.aW[P] = St.aE[P] = St.aS[P] = St.aN[P] = 0; St.aP[P] = 1; St.b[P] = val;
        continue;
      }
      let ap = St.aP[P];
      if (isE) { ap += ((1.92 * rho * e) / k) * vol; St.b[P] += ((1.44 * PkA[P] * e) / k) * vol; }
      else {
        const f1 = sst ? F1[P] : 1, beta = 0.075 * f1 + 0.0828 * (1 - f1), gam = (5 / 9) * f1 + 0.44 * (1 - f1), mut = Math.max(mue[P] - mu, 1e-6 * mu);
        ap += beta * rho * e * vol;
        St.b[P] += (sst ? gam * rho * Math.min(S2[P], (10 * CMU * rho * k * e) / mut) : (gam * PkA[P] * e) / k) * vol;
        if (sst) { if (crs[P] > 0) St.b[P] += crs[P] * vol; else ap -= (crs[P] / e) * vol; }
      }
      if (tRdt) { ap += tRdt * vol; St.b[P] += tRdt * vol * eOld[P]; }
      ap /= al; St.b[P] += (1 - al) * ap * te[P]; St.aP[P] = ap;
    }
    lineSolve(St, te, 1);
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (!(te[P] > eMin)) te[P] = eMin;
      if (solid[P]) { mue[P] = mu; continue; }
      const k = tk[P], e = te[P];
      let nut;
      if (isE) nut = (CMU * k * k) / e;
      else if (sst) { const d = Math.max(dist[P], 1e-12), a2 = Math.max((2 * Math.sqrt(k)) / (CMU * e * d), (500 * nu) / (d * d * e)); nut = (0.31 * k) / Math.max(0.31 * e, Math.sqrt(S2[P]) * Math.tanh(a2 * a2)); }
      else if (cmuE) nut = cmuE[P] * k * Math.max(1 / (CMU * e), 6 * Math.sqrt(nu / (CMU * k * e)));
      else nut = k / e;
      mue[P] = 0.5 * mue[P] + 0.5 * (mu + rho * Math.min(nut, 1e5 * nu));
    }
    if (rsm) {
      // Coupling to the mean flow: the momentum equations keep ∇·(μ_t ∇u) implicitly for stability, and the transported
      // stresses enter as −∇·(ρR) + ∇·(−μ_t ∇u) evaluated with the same face viscosities and differences, so that the
      // eddy-viscosity part cancels identically and the mean flow is driven by the transported stresses alone.
      // Normal stresses live at the cell centres (east/west faces of u, north/south faces of v), the shear stress at the
      // cell corners. Faces on walls and next to blocked cells keep the wall-function / effective-viscosity shear.
      // The cell-centred shear stress responds to the velocity difference over two cells, which leaves a cell-to-cell
      // (odd–even) mode undamped; as in the apparent-viscosity interpolation of Lien & Leschziner (1994) the corner stress
      // therefore carries μ_t × (compact face gradient − interpolated cell gradient), a term of second order in the cell size.
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        if (solid[P]) { rxx[P] = ryy[P] = 0; continue; }
        const mt = mue[P] - mu, k = tk[P];
        rxx[P] = 0.5 * rxx[P] + 0.5 * (rho * (rs4[0][P] - (2 / 3) * k) + (mt * (u[j * nu1 + i + 1] - u[j * nu1 + i])) / dx);
        ryy[P] = 0.5 * ryy[P] + 0.5 * (rho * (rs4[1][P] - (2 / 3) * k) + (mt * (v[P + nx] - v[P])) / dy[j]);
      }
      const uvR = rs4[3];
      for (let jf = 1; jf < ny; jf++) {
        const wS = (0.5 * dy[jf]) / dyc[jf], wN = 1 - wS; // linear interpolation across the gap to the face between rows jf − 1 and jf
        for (let i = 1; i < nx; i++) {
          const q = jf * nu1 + i, a = (jf - 1) * nx + i, b2 = jf * nx + i;
          if (solid[a - 1] || solid[a] || solid[b2 - 1] || solid[b2]) { rxyU[q] = rxyV[q] = 0; continue; }
          const ruv = rho * 0.5 * (wS * (uvR[a - 1] + uvR[a]) + wN * (uvR[b2 - 1] + uvR[b2])), mt = muCorner(i, jf) - mu;
          let eu = (mt * (u[jf * nu1 + i] - u[(jf - 1) * nu1 + i])) / dyc[jf], ev = (mt * (v[b2] - v[b2 - 1])) / dx;
          if (!(nearW[a - 1] || nearW[a] || nearW[b2 - 1] || nearW[b2])) { eu = 0.5 * mt * (wS * (gUy[a - 1] + gUy[a]) + wN * (gUy[b2 - 1] + gUy[b2])); ev = 0.5 * mt * (wS * (gVx[a - 1] + gVx[a]) + wN * (gVx[b2 - 1] + gVx[b2])); }
          rxyU[q] = 0.5 * rxyU[q] + 0.5 * (ruv + eu);
          rxyV[q] = 0.5 * rxyV[q] + 0.5 * (ruv + ev);
        }
        rxyV[jf * nu1] = rxyV[jf * nu1 + 1]; rxyV[jf * nu1 + nx] = rxyV[jf * nu1 + nx - 1]; // inlet and outlet planes: zero streamwise gradient
      }
    }
  };
  /** Resolved Reynolds stresses (kinematic) of the two-equation closures: Boussinesq part plus the explicit anisotropy. */
  const reynolds = () => {
    if (rsm) return { uu: Float64Array.from(rs4[0]), vv: Float64Array.from(rs4[1]), ww: Float64Array.from(rs4[2]), uv: Float64Array.from(rs4[3]) };
    const uu = new Float64Array(n), vv = new Float64Array(n), uv = new Float64Array(n);
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) continue;
      grads(i, j);
      const nut = (mue[P] - mu) / rho, k = tk[P], ax = exx ? exx[P] / rho : 0, ay = exy ? exy[P] / rho : 0;
      uu[P] = (2 / 3) * k - 2 * nut * gr[0] + ax; vv[P] = (2 / 3) * k - 2 * nut * gr[3] - ax; uv[P] = -nut * (gr[1] + gr[2]) + ay;
    }
    return { uu, vv, uv };
  };

  // geometric conductances (multiplied by the local viscosity at assembly)
  const Su = stencil(nu1, ny), Sv = stencil(nx, ny + 1);
  const gxu = new Float64Array(Su.Dx.length), gyu = new Float64Array(Su.Dy.length), gxv = new Float64Array(Sv.Dx.length), gyv = new Float64Array(Sv.Dy.length);
  for (let j = 0; j < ny; j++) for (let k = 1; k <= nx; k++) gxu[j * (nu1 + 1) + k] = dy[j] / dx;
  for (let jf = 0; jf <= ny; jf++) for (let i = 1; i < nx; i++) {
    const lo = jf > 0 ? ublk[(jf - 1) * nu1 + i] : 1, hi = jf < ny ? ublk[jf * nu1 + i] : 1;
    let g = lo && hi ? 0 : lo ? dx / (0.5 * dy[jf]) : hi ? dx / (0.5 * dy[jf - 1]) : dx / dyc[jf];
    if (jf === 0 && !hi) g = wallB === 'sym' ? 0 : dx / (0.5 * dy[0] + (wallB === 'slip' ? bSlip : 0));
    if (jf === ny && !lo) g = wallT === 'sym' ? 0 : dx / (0.5 * dy[ny - 1] + (wallT === 'slip' ? bSlip : 0));
    gyu[jf * nu1 + i] = g;
  }
  for (let jf = 1; jf < ny; jf++) for (let k = 0; k <= nx; k++) {
    const le = k > 0 ? vblk[jf * nx + k - 1] : 2, ri = k < nx ? vblk[jf * nx + k] : 2;
    gxv[jf * (nx + 1) + k] = k === nx ? 0 : k === 0 || (le ? 1 : 0) !== (ri ? 1 : 0) ? dyc[jf] / (0.5 * dx) : le && ri ? 0 : dyc[jf] / dx;
  }
  for (let jj = 1; jj <= ny; jj++) for (let i = 0; i < nx; i++) gyv[jj * nx + i] = dx / dy[jj - 1];
  for (let q = 0; q < gxu.length; q++) Su.Dx[q] = mu * gxu[q];
  for (let q = 0; q < gyu.length; q++) Su.Dy[q] = mu * gyu[q];
  for (let q = 0; q < gxv.length; q++) Sv.Dx[q] = mu * gxv[q];
  for (let q = 0; q < gyv.length; q++) Sv.Dy[q] = mu * gyv[q];
  const du = new Float64Array(nu1 * ny), dv = new Float64Array(nx * (ny + 1));
  const pE = new Float64Array(n), pN = new Float64Array(n), pD = new Float64Array(n), rhs = new Float64Array(n);
  const Bs = bandSolver(nx, ny), Wcg = Bs ? null : { r: new Float64Array(n), z: new Float64Array(n), s: new Float64Array(n), q: new Float64Array(n), pc: new Float64Array(n) };
  const por = o.porous ? new Uint8Array(n) : null;
  if (por) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const x = (i + 0.5) * dx; por[j * nx + i] = x >= o.porous.x0 && x <= o.porous.x1 && !solid[j * nx + i] ? 1 : 0; }
  const simplec = o.simplec !== false, aU = o.alphaU ?? 0.7, aPr = o.alphaP ?? (simplec ? 1 : 0.3);
  const uPrev = new Float64Array(u.length);
  let un = null, vn = null;

  /** One SIMPLE(C) iteration. rdt = ρ/Δt for time-accurate steps (0 for steady relaxation). */
  const iterate = (rdt) => {
    const al = rdt ? 1 : aU, turb = !!o.turb, cr = cv * rho * 0.5;
    // Time-accurate steps are stabilised by the inertia term ρ/Δt alone. Where the step is so long that this term adds less than 10 % to
    // the diagonal (creeping flow, cells far finer than the time step resolves) the iteration is under-relaxed by that amount instead.
    const aT = 1.1;
    // ---- u momentum (the laminar conductances are constant and were set once)
    for (let j = 0; j < ny; j++) { const cj = cr * dy[j]; for (let k = 1, q = j * (nu1 + 1) + 1, a = j * nu1; k <= nx; k++, q++, a++) Su.Fx[q] = cj * (u[a] + u[a + 1]); }
    for (let jf = 0; jf <= ny; jf++) { const cx = cr * dx; for (let i = 1, q = jf * nu1 + 1, a = jf * nx; i < nx; i++, q++, a++) Su.Fy[q] = cx * (v[a] + v[a + 1]); }
    if (turb) {
      for (let j = 0; j < ny; j++) for (let k = 1; k <= nx; k++) { const q = j * (nu1 + 1) + k; Su.Dx[q] = mue[j * nx + k - 1] * gxu[q]; }
      for (let jf = 0; jf <= ny; jf++) for (let i = 1; i < nx; i++) { const q = jf * nu1 + i; Su.Dy[q] = muCorner(i, jf) * gyu[q]; }
    }
    if (turb) for (let i = 1; i < nx; i++) for (const top of [0, 1]) { // log-law wall function on the channel walls
      const j = top ? ny - 1 : 0, k = j * nu1 + i;
      if (ublk[k] || (top ? wallT : wallB) === 'sym') continue;
      const ua = Math.abs(u[k]), y = 0.5 * dy[j], ut = uTau(ua, y);
      if ((y * ut) / nu > 11.6 && ua > 0) Su.Dy[(top ? ny : 0) * nu1 + i] = (rho * ut * ut * dx) / ua;
    }
    assemble(Su, u, scheme);
    let dUmax = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) {
      const k = j * nu1 + i;
      if (ublk[k]) { Su.aW[k] = Su.aE[k] = Su.aS[k] = Su.aN[k] = 0; Su.aP[k] = 1; Su.b[k] = u[k]; du[k] = 0; continue; }
      const vol = dx * dy[j], nb = Su.aW[k] + Su.aE[k] + Su.aS[k] + Su.aN[k];
      let ap = Su.aP[k];
      if (por && (por[j * nx + i - 1] || por[j * nx + i])) {
        const vm = 0.25 * (v[j * nx + i - 1] + v[j * nx + i] + v[(j + 1) * nx + i - 1] + v[(j + 1) * nx + i]);
        ap += (mu / o.porous.K + (rho * o.porous.cF * Math.hypot(u[k], vm)) / Math.sqrt(o.porous.K)) * vol;
      }
      if (rdt) { const a0 = ap; ap += rdt * vol; Su.b[k] += rdt * vol * un[k]; if (ap < aT * a0) { Su.b[k] += (aT * a0 - ap) * u[k]; ap = aT * a0; } }
      ap /= al;
      if (rsm) Su.b[k] -= (rxx[j * nx + i] - rxx[j * nx + i - 1]) * dy[j] + (rxyU[(j + 1) * nu1 + i] - rxyU[j * nu1 + i]) * dx;
      else if (exx) Su.b[k] -= (exx[j * nx + i] - exx[j * nx + i - 1]) * dy[j] + (exC(i, j + 1) - exC(i, j)) * dx;
      Su.b[k] += (p[j * nx + i - 1] - p[j * nx + i]) * dy[j] + (1 - al) * ap * u[k];
      Su.aP[k] = ap; du[k] = dy[j] / (simplec ? Math.max(ap - nb, 0.05 * ap) : ap);
    }
    uPrev.set(u); // previous iterate, for the change norm
    lineSolve(Su, u, 1);
    for (let k = 0; k < u.length; k++) { const d = Math.abs(u[k] - uPrev[k]); if (d > dUmax) dUmax = d; }
    // ---- v momentum
    for (let jf = 1; jf < ny; jf++) { const ca = cr * dy[jf - 1], cb = cr * dy[jf]; for (let k = 0, q = jf * (nx + 1), a = (jf - 1) * nu1; k <= nx; k++, q++, a++) Sv.Fx[q] = ca * u[a] + cb * u[a + nu1]; }
    for (let jj = 1; jj <= ny; jj++) { const cx = cr * dx; for (let i = 0, q = jj * nx; i < nx; i++, q++) Sv.Fy[q] = cx * (v[q - nx] + v[q]); }
    if (turb) {
      for (let jf = 1; jf < ny; jf++) for (let k = 0; k <= nx; k++) { const q = jf * (nx + 1) + k; Sv.Dx[q] = muCorner(k, jf) * gxv[q]; }
      for (let jj = 1; jj <= ny; jj++) for (let i = 0; i < nx; i++) { const q = jj * nx + i; Sv.Dy[q] = mue[(jj - 1) * nx + i] * gyv[q]; }
    }
    for (let jf = 0; jf <= ny; jf++) { Sv.bW[jf] = o.inlet === 'periodic' ? v[jf * nx + nx - 1] : 0; Sv.bE[jf] = v[jf * nx + nx - 1]; }
    assemble(Sv, v, scheme);
    for (let jf = 0; jf <= ny; jf++) for (let i = 0; i < nx; i++) {
      const k = jf * nx + i;
      if (vblk[k]) { Sv.aW[k] = Sv.aE[k] = Sv.aS[k] = Sv.aN[k] = 0; Sv.aP[k] = 1; Sv.b[k] = v[k]; dv[k] = 0; continue; }
      const vol = dx * dyc[jf], nb = Sv.aW[k] + Sv.aE[k] + Sv.aS[k] + Sv.aN[k];
      let ap = Sv.aP[k];
      if (por && (por[(jf - 1) * nx + i] || por[jf * nx + i])) {
        const um = 0.25 * (u[(jf - 1) * nu1 + i] + u[(jf - 1) * nu1 + i + 1] + u[jf * nu1 + i] + u[jf * nu1 + i + 1]);
        ap += (mu / o.porous.K + (rho * o.porous.cF * Math.hypot(um, v[k])) / Math.sqrt(o.porous.K)) * vol;
      }
      if (rdt) { const a0 = ap; ap += rdt * vol; Sv.b[k] += rdt * vol * vn[k]; if (ap < aT * a0) { Sv.b[k] += (aT * a0 - ap) * v[k]; ap = aT * a0; } }
      ap /= al;
      if (rsm) Sv.b[k] -= (rxyV[jf * nu1 + i + 1] - rxyV[jf * nu1 + i]) * dyc[jf] + (ryy[jf * nx + i] - ryy[(jf - 1) * nx + i]) * dx;
      else if (exx) Sv.b[k] -= (exC(i + 1, jf) - exC(i, jf)) * dyc[jf] - (exx[jf * nx + i] - exx[(jf - 1) * nx + i]) * dx;
      Sv.b[k] += (p[(jf - 1) * nx + i] - p[jf * nx + i]) * dx + (1 - al) * ap * v[k];
      Sv.aP[k] = ap; dv[k] = dx / (simplec ? Math.max(ap - nb, 0.05 * ap) : ap);
    }
    lineSolve(Sv, v, 1);
    // ---- outlet: zero streamwise gradient, scaled to global continuity
    let Qperm = 0, Qo = 0;
    for (let i = 0; i < nx; i++) Qperm += (v[ny * nx + i] - v[i]) * dx;
    for (let j = 0; j < ny; j++) { const k = j * nu1 + nx; u[k] = solid[j * nx + nx - 1] ? 0 : Math.max(0, u[k - 1]); Qo += u[k] * dy[j]; }
    const tgt = Qin - Qperm;
    if (Qo > 1e-30) for (let j = 0; j < ny; j++) u[j * nu1 + nx] *= tgt / Qo;
    else { let open = 0; for (let j = 0; j < ny; j++) if (!solid[j * nx + nx - 1]) open += dy[j]; for (let j = 0; j < ny; j++) u[j * nu1 + nx] = solid[j * nx + nx - 1] ? 0 : tgt / open; }
    // ---- pressure correction
    let res = 0;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) { pE[P] = pN[P] = pD[P] = rhs[P] = 0; continue; }
      const ke = j * nu1 + i + 1, kn = (j + 1) * nx + i;
      pE[P] = i < nx - 1 && !ublk[ke] ? rho * du[ke] * dy[j] : 0;
      pN[P] = j < ny - 1 && !vblk[kn] ? rho * dv[kn] * dx : 0;
      rhs[P] = rho * ((u[ke - 1] - u[ke]) * dy[j] + (v[kn - nx] - v[kn]) * dx);
      res += Math.abs(rhs[P]);
    }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) pD[P] = solid[P] ? 0 : pE[P] + pN[P] + (i > 0 ? pE[P - 1] : 0) + (j > 0 ? pN[P - nx] : 0);
    // pressure outlet: p' = 0 just outside the last column, so the outlet faces are corrected as well
    let dOut = 0, mOut = 0;
    for (let j = 0; j < ny; j++) if (du[j * nu1 + nx - 1] > 0) { dOut += du[j * nu1 + nx - 1] / dy[j]; mOut++; }
    dOut = mOut ? dOut / mOut : 1 / (rho * Uref);
    for (let j = 0; j < ny; j++) { const P = j * nx + nx - 1; du[j * nu1 + nx] = solid[P] ? 0 : dOut * dy[j]; pD[P] += rho * du[j * nu1 + nx] * dy[j]; }
    if (Bs) Bs.solve(pE, pN, pD, rhs, pp, o.pTol ?? 0.02, o.pIter ?? 60, true); else { pp.fill(0); pcg5(nx, ny, pE, pN, pD, rhs, pp, o.pTol ?? 0.02, o.pIter ?? 60, Wcg); }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) continue;
      p[P] += aPr * pp[P];
      if (i < nx - 1 && !ublk[P + j + 1]) u[j * nu1 + i + 1] += du[j * nu1 + i + 1] * (pp[P] - pp[P + 1]);
      if (j < ny - 1 && !vblk[P + nx]) v[P + nx] += dv[P + nx] * (pp[P] - pp[P + nx]);
      if (i === nx - 1) u[j * nu1 + nx] += du[j * nu1 + nx] * pp[P];
    }
    if (o.inlet === 'periodic') { // recycle the outlet-plane profile to the inlet (fully developed / periodic flow)
      let q = 0;
      for (let j = 0; j < ny; j++) q += (solid[j * nx] ? 0 : Math.max(0, u[j * nu1 + nx - 1])) * dy[j];
      if (q > 1e-30) for (let j = 0; j < ny; j++) u[j * nu1] = solid[j * nx] ? 0 : 0.5 * u[j * nu1] + (0.5 * Math.max(0, u[j * nu1 + nx - 1]) * Qin) / q;
    }
    return { mass: res / (rho * Math.abs(Qin) || 1e-30), dU: dUmax / Uref };
  };

  // ---- scalar transport (salt concentration c in kg/m³, temperature in °C)
  const Sc = stencil(nx, ny), fixS = solid;
  const afB = new Float64Array(nx).fill(1), afT = new Float64Array(nx).fill(1); // membrane permeability factor 1/(1 + A μ R_f) of a fouling layer
  if (o.species?.permB) afB.set(o.species.permB); if (o.species?.permT) afT.set(o.species.permT); // optional local permeability factor (0–1), e.g. membrane covered by a filament footprint
  const makeScalar = (sp, init) => {
    const phi = new Float64Array(n).fill(init), wB = new Float64Array(nx).fill(init), wT = new Float64Array(nx).fill(init), pB = new Float64Array(nx), pT = new Float64Array(nx), nB = new Float64Array(nx), nT = new Float64Array(nx), fB = new Float64Array(nx), fT = new Float64Array(nx);
    const cht = sp.solidD > 0; // conjugate transfer: the scalar is also solved inside the solids (conduction only)
    // Membrane walls: permeation stops where the osmotic pressure of the wall solution balances the driving pressure, so no cell can
    // be more concentrated than that equilibrium (taken at 1.2 × the trans-membrane pressure to leave room for the pressure field).
    // The bound is inactive in a resolved solution; it stops the lagged flux iteration from overshooting in stagnant wall cells.
    let cCap = Infinity;
    if ((sp.bot?.type === 'membrane' || sp.top?.type === 'membrane') && sp.pi && sp.dP > 0) {
      const tgt = 1.2 * sp.dP + 1e3; let lo = Math.max(init, 1e-6), hi = lo;
      for (let q = 0; q < 60 && sp.pi(hi) < tgt && hi < 5e3; q++) hi *= 2;
      if (sp.pi(hi) >= tgt) { for (let q = 0; q < 50; q++) { const m = 0.5 * (lo + hi); if (sp.pi(m) < tgt) lo = m; else hi = m; } }
      cCap = Math.max(hi, 2 * init);
    }
    const Dc = (P) => (solid[P] ? sp.solidD || 0 : sp.D + (o.turb ? (mue[P] - mu) / rho / (sp.sct || 0.85) : 0));
    const hm = (a, b) => (a > 0 && b > 0 ? (2 * a * b) / (a + b) : 0);
    // wall-adjacent diffusivity: molecular in laminar flow and in the viscous sub-layer; in a wall-function cell the
    // scalar law of the wall φ⁺ = σ_t (u⁺ + P), P = 9.24 [(σ/σ_t)^¾ − 1][1 + 0.28 exp(−0.007 σ/σ_t)] (Jayatilleke)
    const wallD = (P, top, i, dl) => {
      if (!o.turb || solid[P]) return Dc(P);
      if (!(top ? wfT : wfB)[i]) return sp.D;
      const ut = Math.sqrt(Math.abs((top ? tauT : tauB)[i]) / rho), st = sp.sct || 0.85, rs = nu / sp.D / st, Pf = 9.24 * (rs ** 0.75 - 1) * (1 + 0.28 * Math.exp(-0.007 * rs));
      return (ut * dl) / Math.max(st * (Math.log(Math.max(1.5, (E_WALL * dl * ut) / nu)) / KAPPA + Pf), 1e-9);
    };
    /** One assembly + line solve. rdt = 1/Δt for transient steps; old = previous time level. */
    const step = (rdt, old, freeze) => {
      for (let j = 0; j < ny; j++) for (let k = 0; k <= nx; k++) {
        const q = j * (nx + 1) + k;
        Sc.Fx[q] = u[j * nu1 + k] * dy[j];
        Sc.Dx[q] = k === nx ? 0 : k === 0 ? (solid[j * nx] || !(uin[j] > 0) ? 0 : (Dc(j * nx) * dy[j]) / (0.5 * dx)) : solid[j * nx + k - 1] || solid[j * nx + k] ? (cht ? (hm(Dc(j * nx + k - 1), Dc(j * nx + k)) * dy[j]) / dx : 0) : (0.5 * (Dc(j * nx + k - 1) + Dc(j * nx + k)) * dy[j]) / dx;
      }
      for (let jf = 0; jf <= ny; jf++) for (let i = 0; i < nx; i++) {
        const q = jf * nx + i;
        Sc.Fy[q] = v[q] * dx;
        Sc.Dy[q] = jf === 0 || jf === ny ? 0 : solid[q - nx] || solid[q] ? (cht ? (hm(Dc(q - nx), Dc(q)) * dx) / dyc[jf] : 0) : (0.5 * (Dc(q - nx) + Dc(q)) * dx) / dyc[jf];
      }
      for (let j = 0; j < ny; j++) { Sc.bW[j] = sp.inVal; Sc.bE[j] = phi[j * nx + nx - 1]; }
      for (const top of [0, 1]) { // fixed-value walls enter through the boundary conductance
        const w = top ? sp.top : sp.bot, j = top ? ny - 1 : 0;
        if (w.type === 'fixed') for (let i = 0; i < nx; i++) { if (solid[j * nx + i] && !cht) continue; Sc.Dy[(top ? ny : 0) * nx + i] = (wallD(j * nx + i, top, i, 0.5 * dy[j]) * dx) / (0.5 * dy[j]); (top ? Sc.bN : Sc.bS)[i] = w.val; }
      }
      assemble(Sc, phi, scheme, cht ? null : fixS);
      let salt = 0;
      for (const top of [0, 1]) {
        const w = top ? sp.top : sp.bot, j = top ? ny - 1 : 0, J = top ? Jt : Jb, wall = top ? wT : wB, perm = top ? pT : pB;
        for (let i = 0; i < nx; i++) {
          const P = j * nx + i, dl = 0.5 * dy[j], D = wallD(P, top, i, dl);
          if (solid[P] && !cht) { wall[i] = sp.inVal; perm[i] = 0; continue; }
          if (solid[P] && (w.type === 'membrane' || w.type === 'react')) { wall[i] = phi[P]; perm[i] = 0; continue; }
          const Aw = w.type === 'membrane' ? sp.A * (top ? afT : afB)[i] : 0;
          if (w.type === 'react') { // complete rejection with first-order surface crystallisation N = k_r (c_w − c_sat)⁺ (film solution across the half cell)
            const Jw = J[i], g = Jw > 1e-14 ? Math.expm1(Math.min(6, (Jw * dl) / D)) / Jw : dl / D, e = 1 + Jw * g;
            let a = 0, b0 = 0, cw = phi[P] * e;
            if (cw > sp.csat && sp.kr > 0) { a = (sp.kr * e) / (1 + sp.kr * g); b0 = (sp.kr * sp.csat) / (1 + sp.kr * g); cw = (phi[P] * e + sp.kr * sp.csat * g) / (1 + sp.kr * g); }
            const S1 = dx * (Jw - a);
            if (S1 < 0) Sc.aP[P] -= S1; else Sc.b[P] += S1 * phi[P];
            Sc.b[P] += dx * b0; wall[i] = cw; (top ? nT : nB)[i] = Math.max(0, a * phi[P] - b0);
          } else if (w.type === 'membrane') {
            // film solution across the wall half-cell: c_w = c_P e^Pe / (1 + (1 − R)(e^Pe − 1)), R = J/(J + B) (or a constant intrinsic rejection sp.R when given)
            let dJ = 0;
            if (!freeze) {
              const R0 = sp.R ?? (J[i] > 0 ? J[i] / (J[i] + sp.B) : 0), e0 = Math.exp(Math.min(6, (J[i] * dl) / D)), E0 = e0 / (1 + (1 - R0) * (e0 - 1)), cw = phi[P] * E0;
              const dP = sp.dP + (p[P] - sp.pRef()), pw = sp.pi(cw), Jn = Math.max(0, Aw * (dP - (pw - sp.pi((1 - R0) * cw))));
              J[i] = 0.5 * J[i] + 0.5 * Jn;
              if (Jn > 0 && cw > 0) dJ = (-Aw * E0 * (sp.pi(1.01 * cw) - pw)) / (0.01 * cw); // ∂J/∂c_P
            }
            // water leaves through the wall at J, salt only at J(1 − R)c_w: the difference concentrates the wall cell.
            // Source S = Δx·J·β·c_P, linearised (Newton) only where that makes it a stabilising sink.
            const R = sp.R ?? (J[i] > 0 ? J[i] / (J[i] + sp.B) : 0), e = Math.exp(Math.min(6, (J[i] * dl) / D)), Ew = e / (1 + (1 - R) * (e - 1));
            const beta = 1 - (1 - R) * Ew, S0 = dx * J[i] * beta * phi[P], S1 = dx * beta * (J[i] + phi[P] * dJ);
            if (S1 < 0) { Sc.aP[P] -= S1; Sc.b[P] += S0 - S1 * phi[P]; } else Sc.b[P] += S0;
            wall[i] = phi[P] * Ew; perm[i] = (1 - R) * wall[i]; salt += J[i] * perm[i] * dx;
          } else if (w.type === 'flux') { Sc.b[P] += w.val * dx; wall[i] = phi[P] + (w.val * dl) / D; }
          else if (w.type === 'conv') { const U = 1 / (1 / w.h + dl / D); Sc.aP[P] += U * dx; Sc.b[P] += U * dx * w.ext; wall[i] = phi[P] + (U * (w.ext - phi[P]) * dl) / D; }
          else if (w.type === 'fixed') wall[i] = w.val;
          else wall[i] = phi[P];
          (top ? fT : fB)[i] = (D * (wall[i] - phi[P])) / dl; // diffusive wall flux into the fluid with the diffusivity the solver uses
        }
      }
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        if (solid[P] && !cht) { Sc.aW[P] = Sc.aE[P] = Sc.aS[P] = Sc.aN[P] = 0; Sc.aP[P] = 1; Sc.b[P] = sp.inVal; continue; }
        if (sp.S) Sc.b[P] += sp.S[P] * dx * dy[j];
        if (sp.Sp) Sc.aP[P] += sp.Sp[P] * dx * dy[j];
        if (rdt) { const m = rdt * dx * dy[j]; Sc.aP[P] += m; Sc.b[P] += m * old[P]; }
        if (!(Sc.aP[P] > 1e-300)) { Sc.aP[P] = 1; Sc.b[P] = phi[P]; }
      }
      pp.set(phi);
      lineSolve(Sc, phi, 1);
      if (cCap < Infinity) for (let P = 0; P < n; P++) if (phi[P] > cCap) phi[P] = cCap;
      let d = 0;
      for (let P = 0; P < n; P++) { const e = Math.abs(phi[P] - pp[P]); if (e > d) d = e; }
      return { change: d, salt };
    };
    return { phi, wB, wT, pB, pT, nB, nT, fB, fT, step, sp };
  };
  const pIn = () => { let s = 0, m = 0; for (let j = 0; j < ny; j++) if (!solid[j * nx]) { s += p[j * nx] * dy[j]; m += dy[j]; } return m ? s / m : 0; };
  let pRefVal = 0;
  const spc = o.species ? makeScalar({ ...o.species, pRef: () => pRefVal, inVal: o.species.c0, bot: { type: o.species.bot || 'none', val: o.species.cwBot ?? o.species.cw }, top: { type: o.species.top || 'none', val: o.species.cwTop ?? o.species.cw } }, o.species.c0) : null;
  const eng = o.energy ? makeScalar({ D: o.energy.alpha, sct: 0.9, inVal: o.energy.Tin, bot: o.energy.bot, top: o.energy.top, solidD: o.energy.solidD || 0, S: o.energy.solidD > 0 && o.energy.qS ? Float64Array.from(solid, (b) => (b ? o.energy.qS : 0)) : null }, o.energy.Tin) : null;
  if (spc) for (const top of [0, 1]) if ((top ? o.species.top : o.species.bot) === 'membrane') { const J = top ? Jt : Jb, J0 = Math.max(0, o.species.A * (o.species.dP - o.species.pi(o.species.c0))); for (let i = 0; i < nx; i++) J[i] = solid[(top ? ny - 1 : 0) * nx + i] ? 0 : J0 * (top ? afT : afB)[i]; }
  setWallV();

  const hist = { it: [], mass: [], dU: [], scal: [] }, tick = async (f, msg) => { if (ctx?.progress) ctx.progress(f, msg); if (ctx?.tick) await ctx.tick(); };
  const tol = o.tol ?? 1e-5, maxIter = o.maxIter ?? 600;
  let iters = 0, converged = false, probe = null, time = 0;
  const rescale = (f) => { // pressure inlet: the flow rate is adjusted until the inlet gauge pressure matches the target
    for (let k = 0; k < u.length; k++) u[k] *= f;
    for (let k = 0; k < v.length; k++) v[k] *= f;
    // the pressure scales with the flow rate in laminar flow and with its square in turbulent flow, where the
    // turbulence fields are carried along as well (k ∝ U², ε ∝ U³, ω and ν_t ∝ U) so that the rescaled state stays a solution
    const fp = o.turb ? f * f : f;
    for (let P = 0; P < n; P++) p[P] *= fp;
    for (let j = 0; j < ny; j++) uin[j] *= f;
    Qin *= f; Uref *= f; setWallV();
    if (!o.turb) return;
    for (let P = 0; P < n; P++) mue[P] = mu + (mue[P] - mu) * f;
    if (!twoEq) return;
    const f2 = f * f, fe = epsM ? f2 * f : f;
    kIn = Math.max(kIn * f2, K_MIN); eIn *= f2 * f; wIn *= f;
    for (let P = 0; P < n; P++) { tk[P] = Math.max(tk[P] * f2, K_MIN); te[P] *= fe; }
    for (const a of [exx, exy, rxx, ryy, rxyU, rxyV, ...(rs4 || [])]) if (a) for (let q = 0; q < a.length; q++) a[q] *= f2;
  };
  let pFac = 1, pBest = -Infinity, uBest = 0, pLast = 0;
  const flowSteady = async (maxIt, f0, f1, tolF = tol) => {
    let ok = false;
    for (let k = 0; k < maxIt; k++) {
      if (o.turb && (twoEq || k % 3 === 0)) updateTurb();
      const r = iterate(0);
      iters++; hist.it.push(iters); hist.mass.push(Math.max(r.mass, 1e-16)); hist.dU.push(Math.max(r.dU, 1e-16));
      if (!Number.isFinite(r.mass) || !Number.isFinite(r.dU) || r.dU > 1e6) throw permCap ? permErr() : Object.assign(new Error('The flow solution diverged. Lower the velocity-relaxation factor, use the hybrid scheme or refine the grid.'), { diverged: true });
      if (k % 12 === 0) await tick(f0 + ((f1 - f0) * k) / maxIt, `Flow iteration ${iters}: continuity residual ${r.mass.toExponential(1)}`);
      if (o.pInlet > 0 && k >= 8 && k % 4 === 0) {
        // a negative inlet pressure means that the passage recovers more static pressure than friction consumes at this flow rate
        // (expansion, flattening velocity profile): the flow is then lowered toward the friction-dominated branch, never raised
        const dpn = pIn(); pLast = dpn; if (dpn > pBest) { pBest = dpn; uBest = Uref; }
        pFac = dpn > 0 ? clamp((o.pInlet / dpn) ** 0.5, 0.7, 1.4) : 0.7;
        if (Uref * pFac < 1e-9) pFac = 1;
        if (Math.abs(pFac - 1) > 1e-9) rescale(pFac);
      }
      if (k > 3 && r.mass < tolF && r.dU < tolF && (!(o.pInlet > 0) || (k > 12 && Math.abs(pFac - 1) < 20 * tol))) { ok = true; break; }
    }
    return ok;
  };
  const scalarSteady = async (sc, maxIt, freeze, scale, quiet) => {
    let last = Infinity;
    for (let k = 0; k < maxIt; k++) {
      if (sc === spc) pRefVal = pIn();
      if (sc.pre) sc.pre();
      const r = sc.step(0, null, freeze);
      if (sc === spc && !freeze) setWallV();
      let sc0 = scale;
      if (!(sc0 > 0)) { sc0 = 1e-300; for (let P = 0; P < n; P++) { const a = Math.abs(sc.phi[P]); if (a > sc0) sc0 = a; } }
      last = r.change / sc0; if (!quiet) hist.scal.push(Math.max(last, 1e-16));
      if (!Number.isFinite(last)) throw permCap ? permErr() : new Error('The scalar transport solution diverged.');
      if (k % 20 === 0 && ctx?.tick) await ctx.tick();
      if (k > 2 && last < Math.max(tol, 1e-7)) break;
    }
    return last;
  };
  let scalRes = 0;
  const pReached = (band) => { // pressure inlet: the controller must have brought the inlet pressure to the target
    if (!(o.pInlet > 0)) return;
    const dpn = pIn();
    if (Math.abs(dpn / o.pInlet - 1) <= band) return;
    const bound = pBest < 0.8 * o.pInlet && iters > 60;
    throw new Error(`The inlet gauge pressure of ${fmt(o.pInlet, 4)} Pa was not reached: after ${iters} iterations the flow-rate controller stands at ${fmt(dpn, 3)} Pa and a mean velocity of ${fmt(Uref, 3)} m/s` + (bound ? `, and the highest inlet pressure it found is ${fmt(pBest, 3)} Pa (at ${fmt(uBest, 3)} m/s). The passage recovers static pressure downstream (an expansion, or a velocity profile that flattens), so the static pressure at the inlet has an upper bound and falls again when the flow is raised. Lower the target pressure, or use a velocity or mass-flow inlet.` : '. Raise the maximum number of iterations or lower the velocity-relaxation factor.'));
  };
  if (o.steady !== false) {
    // with membrane walls the permeation flux is coupled back twice below, so the first pass only needs to come
    // within reach of the tolerance: the last coupling pass decides convergence
    const memb = !!spc && (o.species.bot === 'membrane' || o.species.top === 'membrane'), tolC = Math.max(tol, Math.min(30 * tol, 1e-3));
    converged = await flowSteady(maxIter, 0, spc || eng ? 0.6 : 0.95, memb ? tolC : tol);
    if (!converged) pReached(0.2);
    if (spc) {
      const nS = o.scalIter ?? 400;
      if (memb) {
        await scalarSteady(spc, nS, false, o.species.c0);
        await tick(0.7, 'Coupling permeation flux and flow');
        await flowSteady(Math.min(80, maxIter), 0.7, 0.8, tolC);
        await scalarSteady(spc, Math.round(nS / 2), false, o.species.c0);
        const ok2 = await flowSteady(Math.min(120, maxIter), 0.8, 0.9);
        converged = converged && ok2;
        scalRes = await scalarSteady(spc, nS, true, o.species.c0);
      } else scalRes = await scalarSteady(spc, nS, true, o.species.c0);
    }
    if (eng) scalRes = Math.max(scalRes, await scalarSteady(eng, o.scalIter ?? 400, true, Math.max(1, Math.abs(o.energy.Tin))));
  } else {
    // time-accurate implicit Euler with SIMPLE inner iterations; CFL based on the local velocity
    un = new Float64Array(u.length); vn = new Float64Array(v.length);
    // pressure inlet: the flow rate is found by steady iterations first and then held during the time-accurate run
    if (o.pInlet > 0) { await flowSteady(Math.min(maxIter, 400), 0, 0.2, Math.max(tol, 1e-4)); pReached(0.5); }
    const cOld = spc ? new Float64Array(n) : null, tOld = eng ? new Float64Array(n) : null, tEnd = o.pInlet > 0 && o.tFlowN > 0 ? (o.tFlowN * L) / Math.max(Uref, 1e-9) : o.tEnd, inner = o.inner ?? 2;
    probe = { t: [], v: [], dp: [], i: o.probe?.i ?? Math.round(0.6 * nx), j: o.probe?.j ?? Math.round(ny / 2) };
    const stat = { n: 0, tauB: new Float64Array(nx), tauT: new Float64Array(nx), cB: spc ? new Float64Array(nx) : null, cT: spc ? new Float64Array(nx) : null, JB: new Float64Array(nx), JT: new Float64Array(nx), dp: 0 };
    // small antisymmetric disturbance so that wake instabilities can develop from a symmetric start
    for (let jf = 1; jf < ny; jf++) for (let i = 0; i < nx; i++) if (!vblk[jf * nx + i]) v[jf * nx + i] += 0.03 * Uref * Math.sin((6 * Math.PI * (i + 0.5)) / nx) * Math.sin((Math.PI * yf[jf]) / H);
    let step = 0, hold = 0;
    while (time < tEnd && step < maxIter) {
      let um = 1e-12;
      for (let k = 0; k < u.length; k++) um = Math.max(um, Math.abs(u[k]));
      let vm = 0;
      for (let jf = 1; jf < ny; jf++) for (let i = 0; i < nx; i++) vm = Math.max(vm, Math.abs(v[jf * nx + i]) / dyc[jf]);
      const dt = Math.min((o.cfl ?? 1) / (um / dx + vm), tEnd - time + 1e-12);
      if (hold > 0) hold--; // the flow has stopped changing: it is carried over while only the scalars advance
      else {
        un.set(u); vn.set(v);
        if (twoEq) { tRdt = rho / dt; kOld.set(tk); eOld.set(te); if (rsm) for (let c = 0; c < 4; c++) rsOld[c].set(rs4[c]); }
        if (o.turb) updateTurb();
        let r = null;
        for (let k = 0; k < inner; k++) { r = iterate(rho / dt); iters++; if (r.mass < tol && r.dU < tol) break; } // a time step that is already converged needs no second pass
        if (!Number.isFinite(r.mass) || !Number.isFinite(r.dU) || r.dU > 1e6 || um > 1e6 * Uref) throw permCap ? permErr() : Object.assign(new Error('The transient flow solution diverged. Lower the CFL number.'), { diverged: true });
        hist.it.push(iters); hist.mass.push(Math.max(r.mass, 1e-16)); hist.dU.push(Math.max(r.dU, 1e-16));
        if (r.mass < tol && r.dU < tol) { // steady flow: re-solve it only often enough to keep the accumulated drift below the tolerance
          let dS = 0;
          for (let k = 0; k < u.length; k++) { const e = Math.abs(u[k] - un[k]); if (e > dS) dS = e; }
          if (dS < 0.1 * tol * Uref) hold = Math.min(20, Math.floor((0.5 * tol * Uref) / Math.max(dS, 1e-300)));
        }
      }
      if (spc) { cOld.set(spc.phi); pRefVal = pIn(); spc.step(1 / dt, cOld, false); setWallV(); }
      if (eng) { tOld.set(eng.phi); eng.step(1 / dt, tOld, true); }
      time += dt; step++;
      let po = 0, mo = 0;
      for (let j = 0; j < ny; j++) if (!solid[j * nx + nx - 1]) { po += p[j * nx + nx - 1] * dy[j]; mo += dy[j]; }
      probe.t.push(time); probe.v.push(v[probe.j * nx + probe.i]); probe.dp.push(pIn() - (mo ? po / mo : 0));
      if (time > 0.5 * tEnd) { // time-averaged wall quantities over the second half
        wallShear(); stat.n++; stat.dp += probe.dp[probe.dp.length - 1];
        for (let i = 0; i < nx; i++) { stat.tauB[i] += tauB[i]; stat.tauT[i] += tauT[i]; stat.JB[i] += Jb[i]; stat.JT[i] += Jt[i]; if (spc) { stat.cB[i] += spc.wB[i]; stat.cT[i] += spc.wT[i]; } }
      }
      if (step % 8 === 0) await tick(time / tEnd, `Time ${time.toExponential(2)} s of ${tEnd.toExponential(2)} s`);
    }
    converged = time >= tEnd * 0.999;
    probe.stat = stat; probe.tEnd = tEnd;
  }
  // ---- optional models evaluated on the solved velocity field
  const memB = !!spc && o.species.bot === 'membrane', memT = !!spc && o.species.top === 'membrane', nX = o.scalIter ?? 400, pr = o.precip;
  let scal = null, mom = null, usr = null, foul = null;
  if (pr) { // sparingly soluble salt: transport, wall crystallisation and the moments m0..m3 of the crystal size distribution
    scal = makeScalar({ D: pr.D, sct: 0.85, inVal: pr.c0, kr: pr.kr, csat: pr.csat, bot: { type: pr.bot ? 'react' : 'none' }, top: { type: pr.top ? 'react' : 'none' }, S: pr.pbm ? new Float64Array(n) : null, Sp: pr.pbm ? new Float64Array(n) : null }, pr.c0);
    if (pr.pbm) mom = [0, 1, 2, 3].map((k) => makeScalar({ D: pr.Dp, sct: 0.85, inVal: pr.n0 * pr.d0 ** k, bot: { type: 'none' }, top: { type: 'none' }, S: new Float64Array(n) }, pr.n0 * pr.d0 ** k));
  }
  const solvePrecip = async (its) => {
    for (let pass = 0; pass < (mom ? 2 : 1); pass++) {
      await scalarSteady(scal, its, true, pr.c0, true);
      if (!mom) break;
      for (let k = 0; k < 4; k++) { // dm_k/dt = 0^k B + k G m_(k−1): nucleation B = k_n (S − 1)^n, linear growth G = k_g (S − 1)
        for (let P = 0; P < n; P++) { const ss = solid[P] ? 0 : Math.max(0, scal.phi[P] / pr.csat - 1); mom[k].sp.S[P] = (k ? k * pr.kg * ss * mom[k - 1].phi[P] : 0) + pr.kn * ss ** pr.nn * pr.dNuc ** k; }
        await scalarSteady(mom[k], its, true, 0, true);
      }
      for (let P = 0; P < n; P++) { const on = !solid[P] && scal.phi[P] > pr.csat; scal.sp.Sp[P] = on ? (pr.rhoC * (Math.PI / 2) * pr.kg * mom[2].phi[P]) / pr.csat : 0; scal.sp.S[P] = scal.sp.Sp[P] * pr.csat; }
    }
  };
  if (pr) { await tick(0.92, 'Precipitation and crystal population'); await solvePrecip(nX); }
  if (o.foul && o.steady !== false && (memB || memT)) { // growing deposit: cake build-up minus shear back-transport plus wall scale → hydraulic resistance
    const fo = o.foul, sA = o.species.A * mu * fo.alpha, mB = new Float64Array(nx).fill(memB ? fo.m0 || 0 : 0), mT = new Float64Array(nx).fill(memT ? fo.m0 || 0 : 0);
    const setAf = () => { for (let i = 0; i < nx; i++) { afB[i] = (o.species.permB ? o.species.permB[i] : 1) / (1 + sA * mB[i]); afT[i] = (o.species.permT ? o.species.permT[i] : 1) / (1 + sA * mT[i]); } };
    const Jm = () => { let a = 0, m = 0; for (let i = 0; i < nx; i++) { if (memB && !solid[i]) { a += Jb[i]; m++; } if (memT && !solid[(ny - 1) * nx + i]) { a += Jt[i]; m++; } } return m ? a / m : 0; };
    foul = { t: [0], J: [], mB, mT, J0: Jm() };
    if (fo.m0 > 0) { setAf(); await scalarSteady(spc, 80, false, o.species.c0, true); await flowSteady(15, 0.93, 0.93); }
    foul.J.push(Jm());
    const dtF = fo.time / fo.steps;
    for (let st = 0; st < fo.steps; st++) {
      wallShear();
      for (const top of [0, 1]) {
        if (!(top ? memT : memB)) continue;
        const m = top ? mT : mB, J = top ? Jt : Jb, tau = top ? tauT : tauB, ns = scal ? (top ? scal.nT : scal.nB) : null, row = top ? (ny - 1) * nx : 0;
        for (let i = 0; i < nx; i++) {
          if (solid[row + i]) continue;
          const drive = J[i] * (1 + sA * m[i]), back = fo.kBack * Math.abs(tau[i]), sc = ns ? ns[i] : 0, h = dtF / 16, rate = (mm) => fo.cp * Math.max(0, drive / (1 + sA * mm) - back) + sc;
          let mm = m[i];
          for (let q = 0; q < 16; q++) mm += h * rate(mm + 0.5 * h * rate(mm)); // midpoint rule at constant net driving pressure
          m[i] = mm;
        }
      }
      setAf();
      await scalarSteady(spc, 60, false, o.species.c0, true);
      await flowSteady(10, 0.93, 0.97);
      if (scal) await solvePrecip(60);
      foul.t.push((st + 1) * dtF); foul.J.push(Jm());
      await tick(0.93 + (0.04 * (st + 1)) / fo.steps, `Fouling step ${st + 1} of ${fo.steps}`);
    }
    scalRes = await scalarSteady(spc, nX >> 1, true, o.species.c0, true);
  }
  if (o.user) { // user-defined scalar with a source expression S(phi, c, T, x, y, u, v), linearised where it acts as a sink
    const us = o.user, env = { c: 0, T: us.T0 ?? 0, x: 0, y: 0, u: 0, v: 0 };
    usr = makeScalar({ D: us.D, sct: 0.85, inVal: us.in, bot: us.bot, top: us.top, S: new Float64Array(n), Sp: new Float64Array(n) }, us.in);
    usr.pre = () => {
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        if (solid[P]) { usr.sp.S[P] = usr.sp.Sp[P] = 0; continue; }
        env.c = spc ? spc.phi[P] : 0; if (eng) env.T = eng.phi[P]; env.x = (i + 0.5) * dx; env.y = yc[j]; env.u = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); env.v = 0.5 * (v[P] + v[P + nx]);
        const f0 = usr.phi[P], h = 1e-6 * (Math.abs(f0) + us.scale), s0 = us.src(f0, env), ds = (us.src(f0 + h, env) - s0) / h;
        if (!Number.isFinite(s0) || !Number.isFinite(ds)) throw new Error('The user-defined source expression returned a non-finite value — check for division by zero or logarithms of negative numbers.');
        if (ds < 0) { usr.sp.Sp[P] = -ds; usr.sp.S[P] = s0 - ds * f0; } else { usr.sp.Sp[P] = 0; usr.sp.S[P] = s0; }
      }
    };
    await scalarSteady(usr, nX, true, 0, true);
  }
  if (o.turb) updateTurb(); else wallShear();
  if (permCap) throw permErr();
  return { scal, mom, usr, foul, afB, afT, pFac, nx, ny, dx, L, H, yf, yc, dy, dyc, solid, u, v, p, mue, Jb, Jt, tauB, tauT, Qin, Uref, uin, hist, iters, converged, scalRes, spc, eng, probe, time, utau: utauG, nu1, tm, tk, te, rs: twoEq ? reynolds() : null, wallB, wallT, slipLen: bSlip };
}

// ---------------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------------
/** Solid mask (Uint8Array, row-major, j from the bottom) for the parametric and imported geometries. */
export function buildMask(g, nx, ny, yc) {
  const { L, H } = g, dx = L / nx, m = new Uint8Array(nx * ny), shapes = [];
  const circle = (xc, y0, r) => {
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (((i + 0.5) * dx - xc) ** 2 + (yc[j] - y0) ** 2 <= r * r) m[j * nx + i] = 1;
    const a = linspace(0, 2 * Math.PI, 33);
    shapes.push({ x: a.map((t) => xc + r * Math.cos(t)), y: a.map((t) => y0 + r * Math.sin(t)), closed: true });
  };
  const box = (x0, x1, ya, yb) => {
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const x = (i + 0.5) * dx; if (x >= x0 && x <= x1 && yc[j] >= ya && yc[j] <= yb) m[j * nx + i] = 1; }
    shapes.push({ x: [x0, x1, x1, x0], y: [ya, ya, yb, yb], closed: true });
  };
  let note = '';
  if (g.type === 'spacer' && g.arr !== 'none') {
    const r = Math.min(g.df, 0.9 * H) / 2;
    for (let k = 0; k < g.nFil; k++) {
      const xc = (k + 0.3) * g.lm;
      circle(xc, g.arr === 'submerged' ? H / 2 : g.arr === 'zigzag' && k % 2 ? H - r : r, r);
    }
  } else if (g.type === 'step') box(-dx, g.stepL * L, -H, g.stepH * H);
  else if (g.type === 'baffle') {
    const nb = Math.max(1, Math.round(g.nBaffle)), w = Math.max(1.5 * dx, 0.012 * L);
    for (let k = 0; k < nb; k++) { const xc = ((k + 1) * L) / (nb + 1); if (k % 2) box(xc - w / 2, xc + w / 2, H * (1 - g.baffleH), 2 * H); else box(xc - w / 2, xc + w / 2, -H, g.baffleH * H); }
  } else if (g.type === 'import') {
    const cad = g.cad;
    if (!cad || (cad.kind !== 'mesh' && cad.kind !== 'polylines')) note = 'No CAD geometry is loaded — an empty channel was solved. Import an STL/OBJ/DXF/GeoJSON file on the Inputs tab.';
    else {
      const ax = clamp(Math.round(g.axis ?? 2), 0, 2);
      let segs = cad.kind === 'mesh' ? sliceMesh(cad, ax) : polylinesToSegments(cad.polylines || []);
      segs = segs.filter((s) => s.every(Number.isFinite));
      if (!segs.length) note = 'The imported geometry gave no outline in the chosen section plane — an empty channel was solved.';
      else {
        let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
        for (const s of segs) for (const q of [0, 2]) { lo[0] = Math.min(lo[0], s[q]); hi[0] = Math.max(hi[0], s[q]); lo[1] = Math.min(lo[1], s[q + 1]); hi[1] = Math.max(hi[1], s[q + 1]); }
        const w = Math.max(hi[0] - lo[0], 1e-300), h = Math.max(hi[1] - lo[1], 1e-300);
        // 'fit': scale uniformly so the outline occupies the requested share of the channel, centred at (cx, cy)
        const sc = g.fit === 'absolute' ? g.scale : g.scale * Math.min((g.size * H) / h, (0.95 * L) / w);
        const ox = g.fit === 'absolute' ? g.cx * L - lo[0] * sc : g.cx * L - 0.5 * (lo[0] + hi[0]) * sc, oy = g.fit === 'absolute' ? g.cy * H - lo[1] * sc : g.cy * H - 0.5 * (lo[1] + hi[1]) * sc;
        const tr = segs.map((s) => [s[0] * sc + ox, s[1] * sc + oy, s[2] * sc + ox, s[3] * sc + oy]);
        const nf = 6 * ny, fine = rasterize(tr, 0, L, 0, H, nx, nf);
        for (let j = 0; j < ny; j++) { const jf = clamp(Math.floor((yc[j] / H) * nf), 0, nf - 1); for (let i = 0; i < nx; i++) m[j * nx + i] = fine[jf][i] !== !!g.invert ? 1 : 0; }
        const step = Math.max(1, Math.ceil(tr.length / 400));
        for (let k = 0; k < tr.length; k += step) shapes.push({ x: [tr[k][0], tr[k][2]], y: [tr[k][1], tr[k][3]], closed: false });
        note = `Imported ${cad.name || 'geometry'}: ${segs.length} outline segments, scale factor ${fmt(sc, 3)}.`;
      }
    }
  }
  // keep the inlet and outlet planes open, then drop fluid pockets that the inlet cannot reach
  let openIn = 0, openOut = 0;
  for (let j = 0; j < ny; j++) { if (!m[j * nx]) openIn++; if (!m[j * nx + nx - 1]) openOut++; }
  if (!openIn || !openOut) { for (let j = 0; j < ny; j++) for (const i of [0, 1, nx - 2, nx - 1]) m[j * nx + i] = 0; note += ' The inlet/outlet planes were blocked and have been cleared.'; }
  const seen = new Uint8Array(nx * ny), st = [];
  for (let j = 0; j < ny; j++) if (!m[j * nx]) { seen[j * nx] = 1; st.push(j * nx); }
  while (st.length) {
    const P = st.pop(), i = P % nx, j = (P - i) / nx;
    for (const Q of [i > 0 ? P - 1 : -1, i < nx - 1 ? P + 1 : -1, j > 0 ? P - nx : -1, j < ny - 1 ? P + nx : -1]) if (Q >= 0 && !m[Q] && !seen[Q]) { seen[Q] = 1; st.push(Q); }
  }
  let reach = false, nSolid = 0;
  for (let j = 0; j < ny; j++) if (seen[j * nx + nx - 1]) reach = true;
  if (!reach) throw new Error('The geometry blocks the channel completely: no flow path connects the inlet to the outlet. Reduce the obstacle size or the import scale.');
  // … and pockets on the inlet plane that have no way to the outlet (an inflow into them could not leave): keep what the outlet reaches too
  const back = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) { const P = j * nx + nx - 1; if (seen[P]) { back[P] = 1; st.push(P); } }
  while (st.length) {
    const P = st.pop(), i = P % nx, j = (P - i) / nx;
    for (const Q of [i > 0 ? P - 1 : -1, i < nx - 1 ? P + 1 : -1, j > 0 ? P - nx : -1, j < ny - 1 ? P + nx : -1]) if (Q >= 0 && seen[Q] && !back[Q]) { back[Q] = 1; st.push(Q); }
  }
  for (let P = 0; P < nx * ny; P++) { if (!back[P]) m[P] = 1; nSolid += m[P]; }
  return { solid: m, shapes, note: note.trim(), solidFraction: nSolid / (nx * ny) };
}

// ---------------------------------------------------------------------------------------------------
// Case set-up, reference correlations, particles and post-processing
// ---------------------------------------------------------------------------------------------------
const SIDES = { both: [1, 1], bottom: [1, 0], top: [0, 1], none: [0, 0] };

function fluid(v) {
  const T = v.T, S = salinityFromTDS(Math.max(0, v.c0) * 1000, T);
  if (v.propMode === 'custom') return { T, S, rho: v.rho, mu: v.mu * 1e-3, D: v.Dsalt * 1e-9, cp: cpWater(T, 0), k: conductivityThermal(T, 0), pi: (c) => v.piCoef * 1e5 * c };
  const rho0 = density(T, 0);
  return { T, S, rho: density(T, S), mu: viscosity(T, S), D: diffusivityNaCl(T, S), cp: cpWater(T, S), k: conductivityThermal(T, S),
    pi: (c) => osmoticPressure(T, clamp((1000 * c) / (rho0 + 0.72 * c), 0, 260)) }; // c in kg/m³ → salinity in g/kg
}

function caseConfig(v) {
  const fl = fluid(v), H = v.H * 1e-3, lm = v.lm * 1e-3, nFil = clamp(Math.round(v.nFil), 1, 40);
  const L = v.geom === 'spacer' ? nFil * lm : v.L * 1e-3;
  const nx = clamp(Math.round(v.nx), 12, 480), ny = clamp(Math.round(v.ny), 6, 200), stretch = clamp(v.stretch, 1, 80), g = yGrid(H, ny, stretch);
  const geo = { type: v.geom, arr: v.arr, L, H, df: v.df * 1e-3, lm, nFil, stepH: v.stepH / 100, stepL: v.stepL / 100, baffleH: v.baffleH / 100, nBaffle: v.nBaffle,
    cad: v.cad, axis: +v.cadAxis, fit: v.cadFit, scale: v.cadScale, size: v.cadSize / 100, cx: v.cadX / 100, cy: v.cadY / 100, invert: !!v.cadInvert };
  const mk = buildMask(geo, nx, ny, g.yc);
  const [sb, st] = SIDES[v.sides] || SIDES.both, [eb, et] = SIDES[v.thSides] || SIDES.both, rc = fl.rho * fl.cp;
  const species = v.species === 'off' ? null : { c0: v.c0, D: fl.D, A: (v.A * LMH) / 1e5, B: v.B * LMH, dP: v.dPtm * 1e5, pi: fl.pi, cw: v.species === 'flux' ? v.jwSalt * 1e-6 : v.cwFixed, bot: sb ? v.species : 'none', top: st ? v.species : 'none' };
  const wall = (on) => (!on || v.thWall === 'none' ? { type: 'none' } : v.thWall === 'fixed' ? { type: 'fixed', val: v.Tw } : v.thWall === 'flux' ? { type: 'flux', val: v.qw / rc } : { type: 'conv', h: v.Uw / rc, ext: v.Text });
  const energy = v.energy ? { alpha: fl.k / rc, Tin: v.T, bot: wall(eb), top: wall(et), solidD: v.cht ? Math.max(v.kSolid, 1e-6) / rc : 0, qS: v.cht ? v.qSolid / rc : 0 } : null;
  // Ergun packed bed: K = d_p² ε³ / (150 (1 − ε)²), c_F = 1.75 / √(150 ε³)
  const erg = v.porModel === 'ergun', eps = clamp(v.porEps ?? 0.4, 0.05, 0.95), dpB = (v.porDp ?? 0.5) * 1e-3;
  const porous = v.porous ? { x0: (Math.min(v.porX0, v.porX1) / 100) * L, x1: (Math.max(v.porX0, v.porX1) / 100) * L, K: erg ? (dpB * dpB * eps ** 3) / (150 * (1 - eps) ** 2) : v.porK, cF: erg ? 1.75 / Math.sqrt(150 * eps ** 3) : v.porCF, ergun: erg, eps, dp: dpB } : null;
  // inlet condition: mean velocity, mass flow per metre width, or gauge pressure (flow rate found by the solver)
  let openIn = 0;
  for (let j = 0; j < ny; j++) if (!mk.solid[j * nx]) openIn += g.dy[j];
  const dpIn = Math.max(v.pInlet ?? 0, 1e-9), les = v.turb === 'les';
  // pressure inlet: starting estimate of the flow rate from laminar and turbulent duct friction and, with a porous zone, its Darcy–Forchheimer resistance
  const uPor = porous ? (() => { const a = (fl.rho * porous.cF) / Math.sqrt(porous.K), b = fl.mu / porous.K, c = dpIn / Math.max(porous.x1 - porous.x0, 1e-9 * L); return a > 0 ? (2 * c) / (b + Math.sqrt(b * b + 4 * a * c)) : c / b; })() : Infinity;
  const Uin = v.inletBC === 'massflow' ? Math.max(v.mdot, 1e-12) / (fl.rho * Math.max(openIn, 1e-12)) : v.inletBC === 'pressure' ? clamp(Math.min((dpIn * H * H) / (12 * fl.mu * L), Math.sqrt((4 * dpIn * H) / (0.03 * fl.rho * L)), uPor), 1e-6, 20) : v.Uin;
  // A laminar parabolic profile is not a state a turbulent flow can enter with: it flattens downstream and recovers more static
  // pressure than friction consumes, so no flow rate gives a positive inlet gauge pressure. A turbulent pressure inlet therefore
  // takes the fully developed profile of the flow itself (recycled from the outlet) in a plain channel and a uniform one elsewhere.
  const rans = !!v.turb && v.turb !== 'laminar', pInSub = v.inletBC === 'pressure' && rans && (v.inlet ?? 'parabolic') === 'parabolic' ? (v.geom === 'plain' || (v.geom === 'spacer' && v.arr === 'none') ? 'periodic' : 'uniform') : null;
  const wt = v.wallType || 'noslip', wallB = wt === 'slip' ? 'slip' : wt === 'symboth' ? 'sym' : 'noslip', wallT = wt === 'slip' ? 'slip' : wt === 'symboth' || wt === 'symtop' ? 'sym' : 'noslip';
  const precip = v.precip ? { c0: v.scC0, csat: Math.max(v.scSat, 1e-9), D: v.scD * 1e-9, kr: v.scKr * 1e-6, bot: !!sb, top: !!st, pbm: !!v.pbm, kn: v.pbKn, nn: clamp(v.pbN, 0.5, 6), kg: v.pbKg * 1e-6, Dp: 1e-11, n0: v.pbSeedN, d0: v.pbSeedD * 1e-6, dNuc: 1e-8, rhoC: v.scRho } : null;
  const foul = v.foul && v.species === 'membrane' && v.mode !== 'transient' && !les ? { cp: v.foulC * 1e-3, alpha: v.foulAlpha, kBack: v.foulBack * 1e-6, time: v.foulTime * 3600, steps: clamp(Math.round(v.foulSteps), 1, 40), m0: v.foulM0 * 1e-3 } : null;
  let user = null;
  if (v.usr) {
    const f = compileExpr(v.usrSrc, ['phi', 'c', 'T', 'x', 'y', 'u', 'v']), uw = v.usrWall === 'fixed' ? { type: 'fixed', val: v.usrWallVal } : { type: 'none' };
    user = { D: v.usrD * 1e-9, in: v.usrIn, T0: v.T, scale: Math.max(Math.abs(v.usrIn), Math.abs(v.usrWallVal ?? 0), 1e-9), bot: sb || v.species === 'off' ? uw : { type: 'none' }, top: st || v.species === 'off' ? uw : { type: 'none' }, src: (phi, env) => { env.phi = phi; return f(env); } };
  }
  const o = { L, H, nx, ny, stretch, solid: mk.solid, rho: fl.rho, mu: fl.mu, Uin, inlet: pInSub || v.inlet, tFlowN: v.tFlow, scheme: v.scheme, steady: v.mode !== 'transient' && !les, maxIter: clamp(Math.round(v.maxIter), 5, 6000), tol: clamp(v.tol, 1e-9, 1e-2),
    alphaU: clamp(v.alphaU, 0.2, 0.95), cfl: clamp(v.cfl, 0.2, 10), tEnd: (v.tFlow * L) / Math.max(Uin, 1e-9), turb: v.turb === 'ml' ? true : v.turb && v.turb !== 'laminar' ? v.turb : false, porous, species, energy,
    wallB, wallT, slipLen: (v.slipLen ?? 0) * 1e-6, creeping: !!v.creeping, tuIn: (v.tuIn ?? 5) / 100, lTurb: ((v.lTurb ?? 7) / 100) * 2 * H, cSmag: v.cSmag ?? 0.17, pInlet: v.inletBC === 'pressure' ? dpIn : 0, precip, foul, user };
  return { fl, geo, mk, L, H, nx, ny, g, o, openIn, Uin, pInSub };
}

/** One-dimensional channel reference: friction and Sherwood correlations with film-theory polarisation. */
export function channel1D(v) {
  const fl = fluid(v), H = v.H * 1e-3, dh = 2 * H, U = Math.max(v.Uin, 1e-9), Re = (fl.rho * U * dh) / fl.mu, Sc = fl.mu / (fl.rho * fl.D);
  const spacer = v.geom === 'spacer' && v.arr !== 'none', L = v.geom === 'spacer' ? Math.max(1, Math.round(v.nFil)) * v.lm * 1e-3 : v.L * 1e-3;
  const xs = L / (dh * Re * Sc);
  const f0 = spacer ? 6.23 * Re ** -0.3 : Re < 2300 ? 96 / Re : 0.316 * Re ** -0.25;
  const Sh0 = spacer ? 0.065 * Re ** 0.875 * Sc ** 0.25 : Re < 2300 ? Math.max(8.235, 2.236 * xs ** (-1 / 3)) : 0.023 * Re ** 0.8 * Sc ** (1 / 3);
  const f = f0 * (v.kdp ?? 1), Sh = Sh0 * (v.ksh ?? 1), k = (Sh * fl.D) / dh;
  const A = (v.A * LMH) / 1e5, B = v.B * LMH, dP = v.dPtm * 1e5, c0 = v.c0;
  let J = Math.max(0, A * (dP - fl.pi(c0))), cw = c0, R = 1;
  for (let it = 0; it < 80; it++) {
    const e = Math.exp(Math.min(8, J / k));
    R = J > 0 ? J / (J + B) : 0; cw = (c0 * e) / (1 + (1 - R) * (e - 1));
    const Jn = Math.max(0, A * (dP - (fl.pi(cw) - fl.pi((1 - R) * cw))));
    if (Math.abs(Jn - J) < 1e-12 + 1e-9 * J) { J = Jn; break; }
    J = 0.5 * J + 0.5 * Jn;
  }
  return { Re, Sc, f, Sh, k, dpPerM: (f * fl.rho * U * U) / (2 * dh), flux: J / LMH, cw, cpSalt: (1 - R) * cw, CP: c0 > 0 ? cw / c0 : 1, fl, dh, L, spacer, xs };
}

/** Lagrangian particles: Stokes drag (exact exponential relaxation), settling, Brownian motion, wall and obstacle capture. */
function trackParticles(r, fl, pt) {
  const { nx, ny, dx, dy, yf, yc, solid, u, v, nu1, L, H, uin, Qin, Uref } = r, g = rng(4711), N = clamp(Math.round(pt.n), 1, 3000);
  const d = pt.d, rp = Math.min(d / 2, 0.2 * H), tau = (pt.rho * d * d) / (18 * fl.mu), vs = -tau * 9.80665 * (1 - fl.rho / pt.rho);
  const DB = (1.380649e-23 * (fl.T + 273.15)) / (3 * Math.PI * fl.mu * d);
  const jOf = (y) => { let lo = 0, hi = ny - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (yf[m] <= y) lo = m; else hi = m - 1; } return lo; };
  const wallU = (w, hc) => (w === 'sym' ? 1 : w === 'slip' ? (r.slipLen || 0) / (hc + (r.slipLen || 0)) : 0), wB = wallU(r.wallB, 0.5 * dy[0]), wT = wallU(r.wallT, 0.5 * dy[ny - 1]);
  let jj = 0;
  const vel = (x, y) => {
    const i = clamp(Math.floor(x / dx), 0, nx - 1), j = (jj = jOf(y)), fx = clamp(x / dx - i, 0, 1), j0 = y < yc[j] ? j - 1 : j, j1 = j0 + 1;
    const y0 = j0 >= 0 ? yc[j0] : 0, y1 = j1 < ny ? yc[j1] : H;
    let ua = j0 >= 0 ? u[j0 * nu1 + i] * (1 - fx) + u[j0 * nu1 + i + 1] * fx : 0, ub = j1 < ny ? u[j1 * nu1 + i] * (1 - fx) + u[j1 * nu1 + i + 1] * fx : 0;
    if (j0 < 0) ua = wB * ub; else if (j1 >= ny) ub = wT * ua; // wall value: zero (no-slip), the slip velocity, or the cell value on a symmetry plane
    const xs = x / dx - 0.5, i0 = clamp(Math.floor(xs), 0, nx - 2), gx = clamp(xs - i0, 0, 1), fy = clamp((y - yf[j]) / dy[j], 0, 1);
    const va = v[j * nx + i0] * (1 - gx) + v[j * nx + i0 + 1] * gx, vb = v[(j + 1) * nx + i0] * (1 - gx) + v[(j + 1) * nx + i0 + 1] * gx;
    return [ua + ((ub - ua) * (y - y0)) / (y1 - y0), va + (vb - va) * fy];
  };
  const out = { n: N, bottom: [], top: [], obstacle: [], tRes: [], suspended: 0, traj: [], tau, vs, DB };
  let cum = 0, jIn = 0;
  for (let k = 0; k < N; k++) {
    const tq = ((k + 0.5) / N) * Qin; // flux-weighted release across the inlet
    while (jIn < ny - 1 && cum + uin[jIn] * dy[jIn] < tq) { cum += uin[jIn] * dy[jIn]; jIn++; }
    let x = 1e-3 * dx, y = clamp(yf[jIn] + (uin[jIn] > 0 ? (tq - cum) / uin[jIn] : 0.5 * dy[jIn]), 1.001 * rp, H - 1.001 * rp), t = 0, done = false;
    let [px, py] = vel(x, y);
    const tr = k % Math.max(1, Math.floor(N / 14)) === 0 && out.traj.length < 14 ? { x: [x], y: [y] } : null;
    for (let s = 0; s < 6000 && !done; s++) {
      const [uf, vf] = vel(x, y), dt = Math.min((0.5 * dx) / (Math.abs(uf) + 1e-12), (0.5 * dy[jj]) / (Math.abs(vf) + Math.abs(vs) + 1e-12), L / Uref / 20);
      const e = Math.exp(-dt / tau), sb = Math.sqrt(2 * DB * dt);
      px = uf + (px - uf) * e; py = vf + vs + (py - vf - vs) * e;
      let xn = x + px * dt + sb * g.normal(), yn = y + py * dt + sb * g.normal();
      t += dt;
      if (xn >= L) { out.tRes.push(t); done = true; }
      else if (yn <= rp) { if (g.uniform() < pt.stick) { out.bottom.push(x); done = true; } else { yn = 2 * rp - yn; py = Math.abs(py); } }
      else if (yn >= H - rp) { if (g.uniform() < pt.stick) { out.top.push(x); done = true; } else { yn = 2 * (H - rp) - yn; py = -Math.abs(py); } }
      if (!done) {
        xn = Math.max(xn, 1e-6 * dx); yn = clamp(yn, rp, H - rp);
        if (solid[jOf(yn) * nx + clamp(Math.floor(xn / dx), 0, nx - 1)]) { if (g.uniform() < pt.stick) { out.obstacle.push(x); done = true; } else { xn = x; yn = y; px = 0; py = 0; } }
      }
      x = xn; y = yn;
      if (tr && (s % 6 === 0 || done) && tr.x.length < 400) { tr.x.push(Math.min(x, L)); tr.y.push(y); }
    }
    if (!done) out.suspended++;
    if (tr) out.traj.push(tr);
  }
  return out;
}

/**
 * Mixing analysis of a transported scalar: coefficient of variation of the scalar over every cross-section,
 * CoV(x) = √(⟨(φ − ⟨φ⟩)²⟩) / |⟨φ⟩| with ⟨·⟩ the area average over the fluid part of the section
 * (absolute = true returns the standard deviation itself).
 */
export function sectionCoV(r, phi, absolute = false) {
  const { nx, ny, dy, solid } = r, out = new Array(nx);
  for (let i = 0; i < nx; i++) {
    let a = 0, m = 0, q = 0;
    for (let j = 0; j < ny; j++) { const P = j * nx + i; if (solid[P]) continue; a += phi[P] * dy[j]; m += dy[j]; }
    const mean = m > 0 ? a / m : 0;
    for (let j = 0; j < ny; j++) { const P = j * nx + i; if (solid[P]) continue; q += (phi[P] - mean) ** 2 * dy[j]; }
    out[i] = !(m > 0) ? 0 : absolute ? Math.sqrt(q / m) : Math.abs(mean) > 1e-300 ? Math.sqrt(q / m) / Math.abs(mean) : 0;
  }
  return out;
}

/** Derived wall, bulk and integral quantities from a solved case. */
function post(c, r) {
  const { nx, ny, dx, dy, yc, solid, u, v, p, nu1, L, H, Uref, Qin } = r, { fl } = c, n = nx * ny, dh = 2 * H;
  const xc = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx), uc = new Float64Array(n), vc = new Float64Array(n);
  let aF = 0, aRec = 0, aStag = 0, umax = 0, aV = 0;
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    if (solid[P]) continue;
    uc[P] = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); vc[P] = 0.5 * (v[P] + v[P + nx]);
    const a = dx * dy[j], sp = Math.hypot(uc[P], vc[P]);
    aF += a; aV += a * Math.abs(vc[P]); if (uc[P] < -1e-3 * Uref) aRec += a; if (sp < 0.1 * Uref) aStag += a; if (sp > umax) umax = sp;
  }
  const st = r.probe?.stat?.n ? r.probe.stat : null, avg = (a, b) => (st ? Array.from(b, (x) => x / st.n) : Array.from(a));
  const tauB = avg(r.tauB, st?.tauB), tauT = avg(r.tauT, st?.tauT), JB = avg(r.Jb, st?.JB), JT = avg(r.Jt, st?.JT);
  const colMean = (f) => xc.map((_, i) => { let s = 0, m = 0, q = 0, qf = 0; for (let j = 0; j < ny; j++) { const P = j * nx + i; if (solid[P]) continue; s += f[P] * dy[j]; m += dy[j]; q += uc[P] * dy[j]; qf += uc[P] * f[P] * dy[j]; } return { area: m ? s / m : 0, cup: Math.abs(q) > 1e-6 * Math.abs(Qin) ? qf / q : m ? s / m : 0 }; });
  const pm = colMean(p).map((q) => q.area), ptot = new Float64Array(n);
  for (let P = 0; P < n; P++) ptot[P] = p[P] + 0.5 * fl.rho * (uc[P] * uc[P] + vc[P] * vc[P]);
  const pt = colMean(ptot).map((q) => q.cup); // mass-flow-averaged total pressure
  // pressure-gradient planes: whole spacer pitches where possible
  let i1 = Math.round(0.1 * nx), i2 = Math.round(0.9 * nx) - 1;
  if (c.geo.type === 'spacer' && c.geo.arr !== 'none' && c.geo.nFil >= 3) { i1 = Math.round(c.geo.lm / dx); i2 = Math.round((L - c.geo.lm) / dx); }
  else if (c.geo.type === 'plain' || (c.geo.type === 'spacer' && c.geo.arr === 'none')) { i1 = Math.round(0.5 * nx); i2 = nx - 2; }
  i1 = clamp(i1, 0, nx - 2); i2 = clamp(i2, i1 + 1, nx - 1);
  const dpdx = (pt[i1] - pt[i2]) / ((i2 - i1) * dx);
  const Re = (fl.rho * Uref * dh) / fl.mu, Sc = fl.mu / (fl.rho * fl.D), f = (dpdx * dh) / (0.5 * fl.rho * Uref * Uref);
  const o = { xc, uc, vc, pm, pt, dpdx, Re, Sc, f, dh, umax, recirc: aRec / aF, stagnant: aStag / aF, tauB, tauT, JB, JT, i1, i2, dpTot: pm[0] - pm[nx - 1], crossFlow: aF > 0 && Uref > 0 ? aV / aF / Uref : 0 };
  if (r.spc) o.covC = sectionCoV(r, r.spc.phi);
  if (r.eng) o.sdT = sectionCoV(r, r.eng.phi, true);
  const absMean = (a, skip) => { let s = 0, m = 0; a.forEach((x, i) => { if (!solid[skip * nx + i]) { s += Math.abs(x); m++; } }); return m ? s / m : 0; };
  o.tauMean = 0.5 * (absMean(tauB, 0) + absMean(tauT, ny - 1));
  o.tauMin = Math.min(...tauB.filter((_, i) => !solid[i]).map(Math.abs), ...tauT.filter((_, i) => !solid[(ny - 1) * nx + i]).map(Math.abs));
  o.tauMax = Math.max(...tauB.map(Math.abs), ...tauT.map(Math.abs));
  // wall transfer coefficients for a scalar: k = wall diffusive flux ÷ (wall − mixing-cup bulk)
  const useF = !!c.o.turb; // turbulent runs: wall flux with the effective diffusivity used by the solver
  const transfer = (sc, D, scale, flux) => {
    const bulk = colMean(sc.phi).map((q) => q.cup), wB = st && sc === r.spc ? avg(null, st.cB) : Array.from(sc.wB), wT = st && sc === r.spc ? avg(null, st.cT) : Array.from(sc.wT);
    const side = (top) => {
      const w = top ? wT : wB, j = top ? ny - 1 : 0, k = [], fl2 = [], dr = [];
      for (let i = 0; i < nx; i++) {
        const P = j * nx + i, drive = w[i] - bulk[i], q = solid[P] ? NaN : flux ? flux(top, i, w[i]) : useF ? (top ? sc.fT : sc.fB)[i] : (D * (w[i] - sc.phi[P])) / (0.5 * dy[j]);
        const ok = !solid[P] && Math.abs(drive) > 1e-7 * scale && Number.isFinite(q) && q / drive > 0;
        k.push(ok ? q / drive : null); fl2.push(ok ? q : 0); dr.push(ok ? drive : 0);
      }
      return { w, k, flux: fl2, drive: dr };
    };
    const B = side(0), T = side(1), mean = (ia, ib) => { let a = 0, b = 0; for (const s of [B, T]) for (let i = ia; i <= ib; i++) { a += s.flux[i]; b += s.drive[i]; } return Math.abs(b) > 0 && a / b > 0 ? a / b : null; };
    return { bulk, B, T, kAll: mean(1, nx - 2), kDev: mean(Math.round(0.5 * nx), nx - 2) };
  };
  if (r.spc) {
    const memb = (top) => (top ? c.o.species.top : c.o.species.bot) === 'membrane';
    o.sp = transfer(r.spc, fl.D, c.o.species.c0 || 1, (top, i, cw) => (memb(top) ? (top ? JT : JB)[i] * (cw - (top ? r.spc.pT : r.spc.pB)[i]) : useF ? (top ? r.spc.fT : r.spc.fB)[i] : (fl.D * (cw - r.spc.phi[(top ? ny - 1 : 0) * nx + i])) / (0.5 * dy[top ? ny - 1 : 0])));
    const act = [0, 1].filter((t) => (t ? c.o.species.top : c.o.species.bot) !== 'none');
    let cps = 0, m = 0, cwMax = 0, cwPeak = 0, Js = 0, Jc = 0, mJ = 0;
    for (const t of act) for (let i = 0; i < nx; i++) {
      const row = (t ? ny - 1 : 0) * nx;
      if (solid[row + i]) continue;
      const w = (t ? o.sp.T : o.sp.B).w[i], J = (t ? JT : JB)[i], contact = [-2, -1, 1, 2].some((k) => i + k >= 0 && i + k < nx && solid[row + i + k]);
      cps += o.sp.bulk[i] > 0 ? w / o.sp.bulk[i] : 1; m++; cwPeak = Math.max(cwPeak, w); if (!contact) cwMax = Math.max(cwMax, w); Js += J; Jc += J * (t ? r.spc.pT : r.spc.pB)[i]; mJ++;
    }
    Object.assign(o, { cpMean: m ? cps / m : 1, cwMax: cwMax || cwPeak || c.o.species.c0, cwPeak: cwPeak || c.o.species.c0, Jmean: mJ ? Js / mJ : 0, cPerm: Js > 0 ? Jc / Js : 0, Qperm: Js * dx, nSides: act.length });
  }
  if (r.eng) o.th = transfer(r.eng, c.o.energy.alpha, 1, null);
  // plotting grid: uniform in y (the solver rows are interpolated)
  const yp = Array.from({ length: ny }, (_, j) => ((j + 0.5) * H) / ny), jn = yp.map((y) => { let b = 0; for (let j = 1; j < ny; j++) if (Math.abs(yc[j] - y) < Math.abs(yc[b] - y)) b = j; return b; });
  const j0 = yp.map((y) => { let j = 0; while (j < ny - 2 && yc[j + 1] < y) j++; return j; });
  o.yp = yp; o.jn = jn;
  o.mask = yp.map((_, q) => xc.map((__, i) => !!solid[jn[q] * nx + i]));
  o.field = (a, scale = 1, hole = NaN) => yp.map((y, q) => xc.map((_, i) => {
    if (solid[jn[q] * nx + i]) return hole;
    const A = j0[q] * nx + i, Bq = A + nx, w = clamp((y - yc[j0[q]]) / (yc[j0[q] + 1] - yc[j0[q]]), 0, 1);
    return scale * (solid[A] ? a[Bq] : solid[Bq] ? a[A] : a[A] * (1 - w) + a[Bq] * w);
  }));
  return o;
}

// ---------------------------------------------------------------------------------------------------
// Additional engines: expression parser, lattice-Boltzmann, interface/phase advection, compressible flow,
// Maxwell–Stefan film and regression closure
// ---------------------------------------------------------------------------------------------------
const EXPR_FN = { exp: Math.exp, log: Math.log, ln: Math.log, sqrt: Math.sqrt, abs: Math.abs, min: Math.min, max: Math.max, pow: Math.pow, tanh: Math.tanh, sin: Math.sin, cos: Math.cos, step: (x) => (x > 0 ? 1 : 0) };
/** Compile an arithmetic expression (+ − * / ^, parentheses, exp/log/sqrt/abs/min/max/pow/tanh/sin/cos/step) into a function of an environment object. */
export function compileExpr(src, names) {
  const str = String(src ?? '').trim(), re = /\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)|([A-Za-z_]\w*)|(\*\*|[-+*/^(),]))\s*/y, tok = [];
  if (!str) throw new Error('The source expression is empty — enter for example "-0.5*phi".');
  for (let pos = 0; pos < str.length; pos = re.lastIndex) {
    re.lastIndex = pos;
    const m = re.exec(str);
    if (!m) throw new Error(`Cannot read the expression near "${str.slice(pos, pos + 12)}".`);
    tok.push(m[1] !== undefined ? { n: +m[1] } : m[2] ? { id: m[2] } : { op: m[3] === '**' ? '^' : m[3] });
  }
  let k = 0;
  const peek = (op) => k < tok.length && tok[k].op === op, need = (op) => { if (!peek(op)) throw new Error(`Expected "${op}" in the expression.`); k++; };
  const atom = () => {
    const t = tok[k++];
    if (!t) throw new Error('The expression ends unexpectedly.');
    if (t.n !== undefined) { const c = t.n; return () => c; }
    if (t.op === '(') { const a = expr(); need(')'); return a; }
    if (t.id) {
      if (peek('(')) {
        k++;
        const fn = EXPR_FN[t.id], args = [];
        if (!fn) throw new Error(`Unknown function "${t.id}" in the expression. Available: ${Object.keys(EXPR_FN).join(', ')}.`);
        if (!peek(')')) for (;;) { args.push(expr()); if (peek(',')) k++; else break; }
        need(')');
        if (!args.length) throw new Error(`Function "${t.id}" needs an argument.`);
        return args.length === 1 ? (e) => fn(args[0](e)) : (e) => fn(...args.map((a) => a(e)));
      }
      if (t.id === 'pi') return () => Math.PI;
      if (!names.includes(t.id)) throw new Error(`Unknown symbol "${t.id}" in the expression. Available: ${names.join(', ')}.`);
      const id = t.id;
      return (e) => e[id];
    }
    throw new Error(`Unexpected "${t.op}" in the expression.`);
  };
  const unary = () => {
    if (peek('-')) { k++; const a = unary(); return (e) => -a(e); }
    if (peek('+')) { k++; return unary(); }
    const a = atom();
    if (peek('^')) { k++; const b = unary(); return (e) => a(e) ** b(e); }
    return a;
  };
  const term = () => { let a = unary(); for (;;) { if (peek('*')) { k++; const l = a, r = unary(); a = (e) => l(e) * r(e); } else if (peek('/')) { k++; const l = a, r = unary(); a = (e) => l(e) / r(e); } else return a; } };
  function expr() { let a = term(); for (;;) { if (peek('+')) { k++; const l = a, r = term(); a = (e) => l(e) + r(e); } else if (peek('-')) { k++; const l = a, r = term(); a = (e) => l(e) - r(e); } else return a; } }
  const f = expr();
  if (k < tok.length) throw new Error('Unexpected trailing text in the expression.');
  return f;
}

/**
 * D2Q9 lattice-Boltzmann (BGK, incompressible He–Luo equilibrium) channel flow in lattice units: cell-centred nodes, halfway bounce-back on the
 * walls and on blocked nodes, non-equilibrium-extrapolation velocity inlet and unit-density outlet.
 * uin[j] = inlet velocity profile; tau = relaxation time (ν = (τ − ½)/3).
 */
export function lbmD2Q9({ nx, ny, solid, uin, tau, maxSteps = 20000, tol = 1e-5, check = 100 }) {
  const N = nx * ny, cx = Int8Array.of(0, 1, 0, -1, 0, 1, -1, -1, 1), cy = Int8Array.of(0, 0, 1, 0, -1, 1, 1, -1, -1), w = Float64Array.of(4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36), opp = Int8Array.of(0, 3, 4, 1, 2, 7, 8, 5, 6);
  let f = new Float64Array(9 * N), g = new Float64Array(9 * N);
  const rho = new Float64Array(N).fill(1), ux = new Float64Array(N), uy = new Float64Array(N), uPrev = new Float64Array(N), om = 1 / tau, inner = new Uint8Array(N);
  const feq = (k, r, a, b) => { const cu = cx[k] * a + cy[k] * b; return w[k] * (r + 3 * cu + 4.5 * cu * cu - 1.5 * (a * a + b * b)); };
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    if (solid[P]) continue;
    ux[P] = uin[j];
    for (let k = 0; k < 9; k++) f[9 * P + k] = feq(k, 1, uin[j], 0);
    let ok = i > 0 && i < nx - 1 && j > 0 && j < ny - 1;
    for (let k = 1; k < 9 && ok; k++) if (solid[P + cx[k] + cy[k] * nx]) ok = false;
    inner[P] = ok ? 1 : 0;
  }
  const E = 9, Nn = 9 * nx, w1 = om / 9, w2 = om / 36, w0 = (om * 4) / 9, km = 1 - om;
  let steps = 0, change = 1;
  while (steps < maxSteps) {
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { // collide and stream (push)
      if (solid[P]) continue;
      const q = 9 * P, f0 = f[q], f1 = f[q + 1], f2 = f[q + 2], f3 = f[q + 3], f4 = f[q + 4], f5 = f[q + 5], f6 = f[q + 6], f7 = f[q + 7], f8 = f[q + 8];
      const r = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8, a = f1 - f3 + f5 - f6 - f7 + f8, b = f2 - f4 + f5 + f6 - f7 - f8, base = r - 1.5 * (a * a + b * b);
      rho[P] = r; ux[P] = a; uy[P] = b;
      const o0 = km * f0 + w0 * base, o1 = km * f1 + w1 * (base + 3 * a + 4.5 * a * a), o3 = km * f3 + w1 * (base - 3 * a + 4.5 * a * a), o2 = km * f2 + w1 * (base + 3 * b + 4.5 * b * b), o4 = km * f4 + w1 * (base - 3 * b + 4.5 * b * b);
      const c5 = a + b, c6 = b - a, o5 = km * f5 + w2 * (base + 3 * c5 + 4.5 * c5 * c5), o7 = km * f7 + w2 * (base - 3 * c5 + 4.5 * c5 * c5), o6 = km * f6 + w2 * (base + 3 * c6 + 4.5 * c6 * c6), o8 = km * f8 + w2 * (base - 3 * c6 + 4.5 * c6 * c6);
      g[q] = o0;
      if (inner[P]) { g[q + E + 1] = o1; g[q + Nn + 2] = o2; g[q - E + 3] = o3; g[q - Nn + 4] = o4; g[q + Nn + E + 5] = o5; g[q + Nn - E + 6] = o6; g[q - Nn - E + 7] = o7; g[q - Nn + E + 8] = o8; continue; }
      for (let k = 1; k < 9; k++) {
        const fo = k === 1 ? o1 : k === 2 ? o2 : k === 3 ? o3 : k === 4 ? o4 : k === 5 ? o5 : k === 6 ? o6 : k === 7 ? o7 : o8, i2 = i + cx[k], j2 = j + cy[k];
        if (j2 < 0 || j2 >= ny) g[q + opp[k]] = fo; // halfway bounce-back on the channel walls
        else if (i2 < 0 || i2 >= nx) continue; // open boundary columns are rebuilt below
        else if (solid[j2 * nx + i2]) g[q + opp[k]] = fo;
        else g[9 * (j2 * nx + i2) + k] = fo;
      }
    }
    for (let j = 0; j < ny; j++) for (let side = 0; side < 2; side++) { // open boundaries: non-equilibrium extrapolation
      const P = j * nx + (side ? nx - 1 : 0), Q = side ? P - 1 : P + 1;
      if (solid[P]) continue;
      let r = 1, a = side ? 0 : uin[j], b = 0;
      const live = !solid[Q];
      if (live) { r = 0; a = 0; for (let k = 0; k < 9; k++) { const fk = g[9 * Q + k]; r += fk; a += fk * cx[k]; b += fk * cy[k]; } }
      const rb = side ? 1 : r, ab = side ? a : uin[j], bb = side ? b : 0;
      for (let k = 0; k < 9; k++) g[9 * P + k] = feq(k, rb, ab, bb) + (live ? g[9 * Q + k] - feq(k, r, a, b) : 0);
    }
    const t = f; f = g; g = t; steps++;
    if (steps % check === 0) {
      let d = 0, m = 1e-30;
      for (let P = 0; P < N; P++) { const e = Math.abs(ux[P] - uPrev[P]); if (e > d) d = e; if (Math.abs(ux[P]) > m) m = Math.abs(ux[P]); }
      uPrev.set(ux); change = d / m;
      if (!Number.isFinite(change)) throw new Error('The lattice-Boltzmann run became unstable — raise the lattice resolution.');
      if (change < tol) break;
    }
  }
  for (let P = 0; P < N; P++) { if (solid[P]) continue; let r = 0, a = 0, b = 0; for (let k = 0; k < 9; k++) { const fk = f[9 * P + k]; r += fk; a += fk * cx[k]; b += fk * cy[k]; } rho[P] = r; ux[P] = a; uy[P] = b; }
  return { ux, uy, rho, steps, change, converged: change < tol };
}

/**
 * Transport of a second phase on the staggered grid g = { nx, ny, dx, dy[], dyc[], u, v, solid }.
 * method 'vof': volume fraction with the algebraic THINC/WLIC interface-capturing scheme, direction-split;
 * 'ls': level-set function (third-order upwind-biased differences, Heun steps, periodic redistancing), α = smoothed Heaviside;
 * 'ee': dispersed-phase continuity with an algebraic slip (drift-flux model) — relative velocity vs
 * (hindered by (1 − α)^nRZ) across the gap, dispersion Dd, deposition on the wall the slip points to.
 */
export function advectPhase(g, o) {
  const { nx, ny, dx, dy, dyc, u, v, solid } = g, n = nx * ny, nu1 = nx + 1, ee = o.method === 'ee', ls = o.method === 'ls', cfl = o.cfl ?? 0.4;
  const a = Float64Array.from(o.a0), an = new Float64Array(n), vs = ee ? o.vs || 0 : 0, nRZ = o.nRZ ?? 4.65, Dd = ee ? o.Dd || 0 : 0, aIn = o.aIn || 0;
  const depB = new Float64Array(nx), depT = new Float64Array(nx), hist = { t: [], vol: [], xc: [], yc: [], inn: [], out: [], dep: [] }, yc = new Float64Array(ny);
  for (let j = 0, y = 0; j < ny; j++) { yc[j] = y + 0.5 * dy[j]; y += dy[j]; }
  let rate = 1e-300, dyMin = Infinity;
  for (let j = 0; j < ny; j++) { dyMin = Math.min(dyMin, dy[j]); for (let i = 0; i < nx; i++) { const r = Math.max(Math.abs(u[j * nu1 + i]), Math.abs(u[j * nu1 + i + 1])) / dx + (Math.max(Math.abs(v[j * nx + i]), Math.abs(v[(j + 1) * nx + i])) + Math.abs(vs)) / dy[j]; if (r > rate) rate = r; } }
  let dt = cfl / rate;
  if (Dd > 0) dt = Math.min(dt, 0.2 / (Dd * (1 / (dx * dx) + 1 / (dyMin * dyMin))));
  const nSteps = Math.max(1, Math.min(o.maxSteps ?? 20000, Math.ceil(o.tEnd / dt)));
  dt = Math.min(dt, o.tEnd / nSteps);
  const lim = (d1, d2) => (d1 * d2 <= 0 ? 0 : (2 * d1 * d2) / (d1 + d2)); // van Leer (dispersed phase)
  // THINC/WLIC (Xiao et al. 2005; Yokoi 2007): the volume fraction inside the upwind cell is a tanh profile whose
  // integral over the swept slab gives the flux; weighted by the interface-normal component along the sweep.
  const BETA = 3.5, CB = Math.cosh(BETA), SB = Math.sinh(BETA);
  const thinc = (aC, aM, aPl, c, wgt) => { // returns the swept fraction of the upwind cell filled with the phase, signed like c
    const ac = Math.abs(c), up = ac * aC;
    if (!(wgt > 0) || aC < 1e-8 || aC > 1 - 1e-8 || (aPl - aC) * (aC - aM) <= 0) return c >= 0 ? up : -up;
    const gm = aPl > aM ? 1 : -1, xt = Math.atanh(clamp((CB - Math.exp((BETA * (2 * aC - 1)) / gm)) / SB, -0.999999999999, 0.999999999999)) / BETA;
    const I = c > 0 ? 0.5 * (ac + (gm / BETA) * Math.log(Math.cosh(BETA * (1 - xt)) / Math.cosh(BETA * (1 - ac - xt)))) : 0.5 * (ac + (gm / BETA) * Math.log(Math.cosh(BETA * (ac - xt)) / Math.cosh(BETA * xt)));
    const F = wgt * clamp(I, 0, ac) + (1 - wgt) * up;
    return c >= 0 ? F : -F;
  };
  const wX = new Float64Array(n);
  let bi0 = 0, bi1 = nx - 1, bj0 = 0, bj1 = ny - 1; // active box: cells that hold the phase, plus a margin of three cells
  const box = () => {
    if (ee || aIn > 0) return;
    let i0 = nx, i1 = -1, j0 = ny, j1 = -1;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) if (a[P] > 0) { if (i < i0) i0 = i; if (i > i1) i1 = i; if (j < j0) j0 = j; if (j > j1) j1 = j; }
    bi0 = Math.max(0, i0 - 3); bi1 = Math.min(nx - 1, i1 + 3); bj0 = Math.max(0, j0 - 3); bj1 = Math.min(ny - 1, j1 + 3);
  };
  const normals = () => { // |n_x| / (|n_x| + |n_y|) from central differences of α
    for (let j = bj0; j <= bj1; j++) for (let i = bi0; i <= bi1; i++) {
      const P = j * nx + i;
      const gx = Math.abs(a[i < nx - 1 ? P + 1 : P] - a[i > 0 ? P - 1 : P]) / (2 * dx), gy = Math.abs(a[j < ny - 1 ? P + nx : P] - a[j > 0 ? P - nx : P]) / (2 * dy[j]);
      wX[P] = gx + gy > 1e-12 ? gx / (gx + gy) : 0.5;
    }
  };
  let inn = 0, out = 0;
  const sweepX = (implicit) => {
    if (!ee) normals();
    for (let j = bj0; j <= bj1; j++) {
      const r0 = j * nx;
      let fw = bi0 > 0 ? 0 : u[j * nu1] * dt * (u[j * nu1] > 0 ? aIn : a[r0]); // inlet face, phase volume per unit height
      inn += fw * dy[j];
      for (let i = bi0; i <= bi1; i++) {
        const P = r0 + i, ue = u[j * nu1 + i + 1];
        let fe;
        if (i === nx - 1) { fe = ue * dt * (ue > 0 ? a[P] : 0); out += fe * dy[j]; }
        else if (ee) {
          if (ue >= 0) { const d2 = a[P + 1] - a[P], d1 = i > 0 ? a[P] - a[P - 1] : 0; fe = ue * dt * (a[P] + 0.5 * (1 - (ue * dt) / dx) * lim(d1, d2)); }
          else { const d2 = a[P] - a[P + 1], d1 = i < nx - 2 ? a[P + 1] - a[P + 2] : 0; fe = ue * dt * (a[P + 1] + 0.5 * (1 + (ue * dt) / dx) * lim(d1, d2)); }
        } else if (ue >= 0) fe = dx * thinc(a[P], i > 0 ? a[P - 1] : a[P], a[P + 1], (ue * dt) / dx, wX[P]);
        else fe = dx * thinc(a[P + 1], a[P], i < nx - 2 ? a[P + 2] : a[P + 1], (ue * dt) / dx, wX[P + 1]);
        const dil = ee ? 0 : (dt / dx) * (ue - u[j * nu1 + i]);
        an[P] = solid[P] ? a[P] : implicit ? (a[P] - (fe - fw) / dx) / (1 - dil) : a[P] * (1 + dil) - (fe - fw) / dx;
        fw = fe;
      }
    }
    for (let j = bj0; j <= bj1; j++) for (let i = bi0, P = j * nx + bi0; i <= bi1; i++, P++) a[P] = an[P];
  };
  const sweepY = (implicit) => {
    if (!ee) normals();
    for (let i = bi0; i <= bi1; i++) {
      let fs = 0;
      if (bj0 === 0) { const vw = v[i] + (vs < 0 && o.absorb !== false && !solid[i] ? vs * (1 - a[i]) ** nRZ : 0); fs = vw < 0 ? vw * dt * a[i] : 0; depB[i] -= fs; }
      for (let j = bj0; j <= bj1; j++) {
        const P = j * nx + i;
        let fn;
        if (j === ny - 1) { const vw = v[ny * nx + i] + (vs > 0 && o.absorb !== false && !solid[P] ? vs * (1 - a[P]) ** nRZ : 0); fn = vw > 0 ? vw * dt * a[P] : 0; depT[i] += fn; }
        else if (ee) {
          const blocked = solid[P] || solid[P + nx];
          let vf = v[P + nx];
          if (vs && !blocked) vf += vs * (1 - (vs > 0 ? a[P] : a[P + nx])) ** nRZ;
          if (vf >= 0) { const d2 = a[P + nx] - a[P], d1 = j > 0 ? a[P] - a[P - nx] : 0; fn = vf * dt * (a[P] + (blocked ? 0 : 0.5 * (1 - (vf * dt) / dyc[j + 1]) * lim(d1, d2))); }
          else { const d2 = a[P] - a[P + nx], d1 = j < ny - 2 ? a[P + nx] - a[P + 2 * nx] : 0; fn = vf * dt * (a[P + nx] + (blocked ? 0 : 0.5 * (1 + (vf * dt) / dyc[j + 1]) * lim(d1, d2))); }
          if (Dd > 0 && !blocked) fn -= (dt * Dd * (a[P + nx] - a[P])) / dyc[j + 1];
        } else { const vf = v[P + nx]; fn = vf >= 0 ? dy[j] * thinc(a[P], j > 0 ? a[P - nx] : a[P], a[P + nx], (vf * dt) / dy[j], 1 - wX[P]) : dy[j + 1] * thinc(a[P + nx], a[P], j < ny - 2 ? a[P + 2 * nx] : a[P + nx], (vf * dt) / dy[j + 1], 1 - wX[P + nx]); }
        const dil = ee ? 0 : (dt / dy[j]) * (v[P + nx] - v[P]);
        an[P] = solid[P] ? a[P] : implicit ? (a[P] - (fn - fs) / dy[j]) / (1 - dil) : a[P] * (1 + dil) - (fn - fs) / dy[j];
        fs = fn;
      }
    }
    for (let j = bj0; j <= bj1; j++) for (let i = bi0, P = j * nx + bi0; i <= bi1; i++, P++) a[P] = an[P];
  };
  // level set
  const phi = ls ? Float64Array.from(o.phi0) : null, p1 = ls ? new Float64Array(n) : null, p2 = ls ? new Float64Array(n) : null, uc = ls ? new Float64Array(n) : null, vc = ls ? new Float64Array(n) : null;
  if (ls) for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { uc[P] = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); vc[P] = 0.5 * (v[P] + v[P + nx]); }
  const rhsLS = (f, r) => { // −u·∇φ with third-order upwind-biased differences (index space across the gap)
    for (let j = 0, P = 0; j < ny; j++) {
      const jm = j > 0 ? nx : 0, jp = j < ny - 1 ? nx : 0, jmm = j > 1 ? 2 * nx : jm, jpp = j < ny - 2 ? 2 * nx : jp;
      for (let i = 0; i < nx; i++, P++) {
        const im = i > 0 ? 1 : 0, ip = i < nx - 1 ? 1 : 0, imm = i > 1 ? 2 : im, ipp = i < nx - 2 ? 2 : ip, c = f[P];
        const gx = uc[P] >= 0 ? (2 * f[P + ip] + 3 * c - 6 * f[P - im] + f[P - imm]) / (6 * dx) : -(2 * f[P - im] + 3 * c - 6 * f[P + ip] + f[P + ipp]) / (6 * dx);
        const gy = vc[P] >= 0 ? (2 * f[P + jp] + 3 * c - 6 * f[P - jm] + f[P - jmm]) / (6 * dy[j]) : -(2 * f[P - jm] + 3 * c - 6 * f[P + jp] + f[P + jpp]) / (6 * dy[j]);
        r[P] = -(uc[P] * gx + vc[P] * gy);
      }
    }
  };
  const redistance = (its) => { // ∂φ/∂τ = sign(φ0)(1 − |∇φ|), Godunov upwinding
    const s0 = p2; for (let P = 0; P < n; P++) s0[P] = phi[P] / Math.sqrt(phi[P] * phi[P] + dx * dx);
    for (let it = 0; it < its; it++) {
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        const hm = dyc[j] || dy[0], hp = dyc[j + 1] || dy[ny - 1], c = phi[P];
        const a1 = i > 0 ? (c - phi[P - 1]) / dx : 0, b1 = i < nx - 1 ? (phi[P + 1] - c) / dx : 0, c1 = j > 0 ? (c - phi[P - nx]) / hm : 0, d1 = j < ny - 1 ? (phi[P + nx] - c) / hp : 0, s = s0[P];
        const gx2 = s > 0 ? Math.max(Math.max(a1, 0) ** 2, Math.min(b1, 0) ** 2) : Math.max(Math.min(a1, 0) ** 2, Math.max(b1, 0) ** 2), gy2 = s > 0 ? Math.max(Math.max(c1, 0) ** 2, Math.min(d1, 0) ** 2) : Math.max(Math.min(c1, 0) ** 2, Math.max(d1, 0) ** 2);
        p1[P] = c - 0.3 * Math.min(dx, dy[j]) * s * (Math.sqrt(gx2 + gy2) - 1);
      }
      phi.set(p1);
    }
  };
  // φ inside solids: constant extrapolation from the fluid, three layers deep, so that the upwind stencils see no artificial interface
  const ext = [];
  if (ls) {
    const lay = new Int8Array(n);
    let any = false;
    for (let P = 0; P < n; P++) if (solid[P]) { lay[P] = -1; any = true; }
    for (let layer = 1; any && layer <= 3; layer++) {
      const cur = [];
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        if (lay[P] !== -1) continue;
        const nb = [i > 0 ? P - 1 : -1, i < nx - 1 ? P + 1 : -1, j > 0 ? P - nx : -1, j < ny - 1 ? P + nx : -1].filter((Q) => Q >= 0 && lay[Q] >= 0 && lay[Q] < layer);
        if (nb.length) cur.push([P, nb]);
      }
      for (const [P] of cur) lay[P] = layer;
      ext.push(...cur);
    }
  }
  const fill = (f) => { for (const [P, nb] of ext) { let a = 0; for (const Q of nb) a += f[Q]; f[P] = a / nb.length; } };
  // volume fraction from φ: smoothed Heaviside over 1.5 cell widths h_n = (|φ_x|Δx + |φ_y|Δy)/(|φ_x| + |φ_y|) measured along the interface normal
  const hv = (P, i, j) => {
    if (solid[P]) return 0;
    const gx = Math.abs(phi[i < nx - 1 ? P + 1 : P] - phi[i > 0 ? P - 1 : P]) / ((i > 0 && i < nx - 1 ? 2 : 1) * dx), jn = j < ny - 1 ? j + 1 : j, js = j > 0 ? j - 1 : j, gy = jn > js ? Math.abs(phi[jn * nx + i] - phi[js * nx + i]) / (yc[jn] - yc[js]) : 0;
    const q = (phi[P] * (gx + gy + 1e-300)) / (1.5 * (gx * dx + gy * dy[j]) + 1e-300);
    return q <= -1 ? 0 : q >= 1 ? 1 : 0.5 * (1 + q + Math.sin(Math.PI * q) / Math.PI);
  };
  const lsOut = (w) => { for (let j = 0; j < ny; j++) { const P = j * nx + nx - 1, ue = u[j * nu1 + nx]; if (ue > 0) out += w * ue * dt * dy[j] * hv(P, nx - 1, j); } }; // phase volume leaving through the outlet
  const heav = () => { for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) a[P] = hv(P, i, j); };
  const stats = (t) => { let vol = 0, sx = 0, sy = 0; for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (solid[P]) continue; const w = a[P] * dx * dy[j]; vol += w; sx += w * (i + 0.5) * dx; sy += w * yc[j]; } let dp = 0; for (let i = 0; i < nx; i++) dp += (depB[i] + depT[i]) * dx; hist.inn.push(inn); hist.out.push(out); hist.dep.push(dp); hist.t.push(t); hist.vol.push(vol); hist.xc.push(vol > 0 ? sx / vol : 0); hist.yc.push(vol > 0 ? sy / vol : 0); return vol; };
  if (ls) { fill(phi); heav(); }
  const vol0 = stats(0), every = Math.max(1, Math.floor(nSteps / 60));
  let amin = 0, amax = 1;
  for (let s = 0; s < nSteps; s++) {
    if (ls) {
      lsOut(0.5);
      rhsLS(phi, p1); for (let P = 0; P < n; P++) p2[P] = phi[P] + dt * p1[P];
      fill(p2);
      rhsLS(p2, an); for (let P = 0; P < n; P++) phi[P] += 0.5 * dt * (p1[P] + an[P]);
      fill(phi);
      if ((s + 1) % (o.reinit ?? 40) === 0) { redistance(2); fill(phi); }
      lsOut(0.5);
      if ((s + 1) % every === 0 || s === nSteps - 1) heav();
    } else {
      box();
      if (s % 2) { sweepY(true); sweepX(false); } else { sweepX(true); sweepY(false); } // implicit–explicit dilatation: exactly conservative for ∇·u = 0
      for (let P = 0; P < n; P++) { const q = a[P]; if (q < amin) amin = q; if (q > amax) amax = q; if (!ee) a[P] = q < 0 ? 0 : q > 1 ? 1 : q; else if (q < 0) a[P] = 0; }
    }
    if ((s + 1) % every === 0 || s === nSteps - 1) stats((s + 1) * dt);
  }
  const vol = hist.vol[hist.vol.length - 1];
  return { a, phi, vol0, vol, hist, steps: nSteps, dt, depB, depT, inn, out, overshoot: Math.max(-amin, amax - 1) };
}

/** Third-order upwind-biased derivative along one index (first-order where the stencil leaves the grid). */
const up3 = (c, m1, m2, p1v, p2v, has2m, has2p, vel, d) => (vel >= 0 ? (has2m ? (2 * p1v + 3 * c - 6 * m1 + m2) / (6 * d) : (c - m1) / d) : has2p ? -(2 * m1 + 3 * c - 6 * p1v + p2v) / (6 * d) : (p1v - c) / d);

/**
 * Coupled two-phase flow of two immiscible incompressible fluids on a uniform staggered grid (one-fluid formulation):
 *   ∇·u = 0,  ρ(α)(∂u/∂t + u·∇u) = −∇p + ∇·[μ(α)(∇u + ∇uᵀ)] + ρ(α) g + σ κ ∇α,
 * α = volume fraction of phase A (ρA, μA), 1 − α that of phase B. Explicit projection: the pressure equation is the
 * variable-density Poisson problem ∇·((1/ρ)∇p) = (∇·ũ)/Δt for the provisional velocity ũ; gravity and the continuum-surface-force term are evaluated
 * at the faces with the same difference as the pressure gradient (balanced-force form).
 * method 'vof': α advected by direction-split THINC/WLIC, curvature κ = −∇·(∇α̃/|∇α̃|) from the smoothed fraction α̃;
 * method 'ls' : level-set function φ (third-order upwind, Heun), redistanced every step and shifted by a constant so
 *               that the phase volume is restored (mass correction); α = smoothed Heaviside, κ from φ.
 * o = { nx, ny, W, Hh, rhoA, rhoB, muA, muB, sigma, gx, gy, sd(x, y) (> 0 inside phase A), tEnd, slipSide, slipTB,
 *       cfl, muMix: 'arith' | 'harm', maxSteps, u0(x, y), v0(x, y), snaps: [times at which α is stored],
 *       solid: Uint8Array(nx·ny) blocked cells, flow: { uin: number | [ny], alphaIn(t, y) }, rhoRef }.
 * Without `flow` the box is closed, with free-slip or no-slip side and top/bottom walls; with it x = 0 is a velocity inlet and x = W a
 * pressure outlet (channel flow carrying the second phase). Blocked cells are no-slip solids (spacer filaments, imported sections).
 */
export async function twoPhase2D(o, ctx) {
  const nx = o.nx, ny = o.ny, W = o.W, Hh = o.Hh, dx = W / nx, dy = Hh / ny, n = nx * ny, nu1 = nx + 1, ls = o.method === 'ls', h = Math.min(dx, dy);
  const rA = o.rhoA, rB = o.rhoB, mA = o.muA, mB = o.muB, sig = Math.max(0, o.sigma || 0), gx = o.gx || 0, gy = o.gy ?? -9.80665, harm = o.muMix === 'harm';
  const sS = o.slipSide !== false ? 1 : -1, sT = o.slipTB ? 1 : -1; // ghost tangential velocity = ± interior value (free-slip / no-slip)
  const f = (m) => new Float64Array(m), a = f(n), an = f(n), at = f(n), rho = f(n), mu = f(n), kap = f(n), wk = f(n), p = f(n), u = f(nu1 * ny), v = f(nx * (ny + 1)), us = f(nu1 * ny), vs = f(nx * (ny + 1));
  const phi = ls ? f(n) : null, p1 = ls ? f(n) : null, p2 = ls ? f(n) : null, p3 = ls ? f(n) : null, ucc = f(n), vcc = f(n), wX = f(n);
  const cnx = f((nx + 1) * (ny + 1)), cny = f((nx + 1) * (ny + 1)), pE = f(n), pN = f(n), pD = f(n), rhs = f(n);
  const eps = 1.5 * h, heavi = (q) => (q <= -eps ? 0 : q >= eps ? 1 : 0.5 * (1 + q / eps + Math.sin((Math.PI * q) / eps) / Math.PI));
  // Optional solid mask (blocked cells, as rasterised by buildMask on the same uniform grid) and through-flow: x = 0 is a velocity inlet
  // with the profile flow.uin[j] (m/s) carrying the phase-A fraction flow.alphaIn(t, y), x = W a pressure outlet (p = 0, zero streamwise
  // gradient). With through-flow the solved pressure is the excess over the hydrostatic pressure of the carrier of density o.rhoRef, so
  // that a uniform outlet pressure is the right condition; solids are no-slip (mirror ghost values), the phase fraction and the level
  // set are extended into them from the neighbouring fluid, which is a contact angle of 90° as on the tank walls.
  const solid = o.solid && o.solid.some((q) => q) ? o.solid : null, flow = o.flow || null, msk = !!(solid || flow), rr = flow ? o.rhoRef ?? rB : 0;
  const bu = solid ? new Uint8Array(nu1 * ny) : null, bv = solid ? new Uint8Array(nx * (ny + 1)) : null, uinF = flow ? Float64Array.from({ length: ny }, (_, j) => (solid && solid[j * nx] ? 0 : typeof flow.uin === 'number' ? flow.uin : flow.uin[j])) : null, aInF = flow && typeof flow.alphaIn === 'function' ? flow.alphaIn : null;
  if (solid) {
    for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) bu[j * nu1 + i] = solid[j * nx + i - 1] || solid[j * nx + i] ? 1 : 0;
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) bv[j * nx + i] = solid[(j - 1) * nx + i] || solid[j * nx + i] ? 1 : 0;
  }
  const ext = []; // solid cells within three layers of the fluid, with the fluid (or already filled) neighbours they are extrapolated from
  if (solid) {
    const lay = new Int8Array(n); for (let P = 0; P < n; P++) if (solid[P]) lay[P] = -1;
    for (let layer = 1; layer <= 3; layer++) {
      const cur = [];
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (lay[P] !== -1) continue; const nb = [i > 0 ? P - 1 : -1, i < nx - 1 ? P + 1 : -1, j > 0 ? P - nx : -1, j < ny - 1 ? P + nx : -1].filter((Q) => Q >= 0 && lay[Q] >= 0 && lay[Q] < layer); if (nb.length) cur.push([P, nb]); }
      for (const [P] of cur) lay[P] = layer; ext.push(...cur);
    }
  }
  const fillS = (ff) => { for (const [P, nb] of ext) { let q = 0; for (const Q of nb) q += ff[Q]; ff[P] = q / nb.length; } };
  const cellW = solid ? Float64Array.from(solid, (q) => (q ? 0 : 1)) : null; // 1 in fluid cells
  // initial interface: 4 × 4 sub-sampled volume fraction and the signed distance
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    let m = 0;
    for (let q = 0; q < 4; q++) for (let r = 0; r < 4; r++) if (o.sd((i + (q + 0.5) / 4) * dx, (j + (r + 0.5) / 4) * dy) > 0) m++;
    a[P] = m / 16;
    if (ls) phi[P] = o.sd((i + 0.5) * dx, (j + 0.5) * dy);
  }
  if (solid) { for (let P = 0; P < n; P++) if (solid[P]) { a[P] = 0; if (ls) phi[P] = -h; } fillS(a); if (ls) fillS(phi); }
  if (flow) for (let j = 0; j < ny; j++) { // start from plug flow through the open height of every column; the first projection makes it divergence-free
    let q = 0; for (let jj = 0; jj < ny; jj++) q += uinF[jj];
    for (let i = 0; i <= nx; i++) { let open = 0; for (let jj = 0; jj < ny; jj++) if (i === 0 ? !(solid && solid[jj * nx]) : i === nx ? !(solid && solid[jj * nx + nx - 1]) : !(bu && bu[jj * nu1 + i])) open++; const on = i === 0 ? !(solid && solid[j * nx]) : i === nx ? !(solid && solid[j * nx + nx - 1]) : !(bu && bu[j * nu1 + i]); u[j * nu1 + i] = i === 0 ? uinF[j] : on && open ? q / open : 0; }
  }
  if (o.u0) for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) u[j * nu1 + i] = o.u0(i * dx, (j + 0.5) * dy);
  if (o.v0) for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) v[j * nx + i] = o.v0((i + 0.5) * dx, j * dy);
  const volOf = () => { let s = 0; if (solid) { for (let P = 0; P < n; P++) if (!solid[P]) s += a[P]; } else for (let P = 0; P < n; P++) s += a[P]; return s * dx * dy; };
  const lsVol = (c) => { let s = 0; if (solid) { for (let P = 0; P < n; P++) if (!solid[P]) s += heavi(phi[P] + c); } else for (let P = 0; P < n; P++) s += heavi(phi[P] + c); return s * dx * dy; };
  if (ls) for (let P = 0; P < n; P++) a[P] = heavi(phi[P]);
  const vol0 = volOf();
  const Bs = o.pSolver === 'cg' ? null : bandSolver(nx, ny, 9e6, o.rent ?? 1), Wcg = Bs ? null : { r: f(n), z: f(n), s: f(n), q: f(n), pc: f(n) };

  const props = () => {
    for (let P = 0; P < n; P++) { const c = a[P] < 0 ? 0 : a[P] > 1 ? 1 : a[P]; rho[P] = rB + (rA - rB) * c; mu[P] = harm ? 1 / (c / mA + (1 - c) / mB) : mB + (mA - mB) * c; }
  };
  const curvature = () => {
    if (!(sig > 0)) return;
    if (ls) { // κ = −∇·(∇φ/|∇φ|) by central differences (mirror boundaries)
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        const im = i > 0 ? 1 : 0, ip = i < nx - 1 ? 1 : 0, jm = j > 0 ? nx : 0, jp = j < ny - 1 ? nx : 0, c = phi[P];
        const fx = (phi[P + ip] - phi[P - im]) / (2 * dx), fy = (phi[P + jp] - phi[P - jm]) / (2 * dy), fxx = (phi[P + ip] - 2 * c + phi[P - im]) / (dx * dx), fyy = (phi[P + jp] - 2 * c + phi[P - jm]) / (dy * dy);
        const fxy = (phi[P + ip + jp] - phi[P + ip - jm] - phi[P - im + jp] + phi[P - im - jm]) / (4 * dx * dy), g2 = fx * fx + fy * fy;
        const k = g2 > 1e-24 ? -(fxx * fy * fy - 2 * fx * fy * fxy + fyy * fx * fx) / (g2 * Math.sqrt(g2)) : 0;
        kap[P] = clamp(k, -1 / h, 1 / h); wk[P] = Math.abs(c) < 2 * eps ? 1 : 0;
      }
      return;
    }
    at.set(a);
    for (let pass = 0; pass < (o.smooth ?? 2); pass++) { // α̃: repeated 1-2-1 ⊗ 1-2-1 filter (mirror boundaries)
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { const im = i > 0 ? 1 : 0, ip = i < nx - 1 ? 1 : 0; an[P] = 0.25 * (at[P - im] + 2 * at[P] + at[P + ip]); }
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { const jm = j > 0 ? nx : 0, jp = j < ny - 1 ? nx : 0; at[P] = 0.25 * (an[P - jm] + 2 * an[P] + an[P + jp]); }
    }
    for (let jf = 0; jf <= ny; jf++) for (let i = 0; i <= nx; i++) { // unit normals at the cell corners
      const i0 = i > 0 ? i - 1 : 0, i1 = i < nx ? i : nx - 1, j0 = jf > 0 ? jf - 1 : 0, j1 = jf < ny ? jf : ny - 1;
      const a00 = at[j0 * nx + i0], a10 = at[j0 * nx + i1], a01 = at[j1 * nx + i0], a11 = at[j1 * nx + i1];
      const ex = (a10 + a11 - a00 - a01) / (2 * dx), ey = (a01 + a11 - a00 - a10) / (2 * dy), g = Math.hypot(ex, ey), q = jf * (nx + 1) + i;
      if (g > 1e-6 / h) { cnx[q] = ex / g; cny[q] = ey / g; } else { cnx[q] = 0; cny[q] = 0; }
    }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      const q = j * (nx + 1) + i, r = q + nx + 1;
      const k = -((cnx[q + 1] + cnx[r + 1] - cnx[q] - cnx[r]) / (2 * dx) + (cny[r] + cny[r + 1] - cny[q] - cny[q + 1]) / (2 * dy));
      kap[P] = clamp(k, -1 / h, 1 / h); wk[P] = at[P] * (1 - at[P]);
    }
  };
  const kFace = (P, Q) => { const w1 = wk[P], w2 = wk[Q], s = w1 + w2; return s > 1e-12 ? (w1 * kap[P] + w2 * kap[Q]) / s : 0; };
  const muC = (i, jf) => 0.25 * (mu[(jf - 1) * nx + i - 1] + mu[(jf - 1) * nx + i] + mu[jf * nx + i - 1] + mu[jf * nx + i]); // interior corners only
  const predictor = (dt) => {
    for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) {
      const k = j * nu1 + i, P = j * nx + i, c = u[k], rf = 0.5 * (rho[P - 1] + rho[P]);
      if (bu !== null && bu[k]) { us[k] = 0; continue; }
      const uS = j > 0 ? (bu !== null && bu[k - nu1] ? -c : u[k - nu1]) : sT * c, uN = j < ny - 1 ? (bu !== null && bu[k + nu1] ? -c : u[k + nu1]) : sT * c;
      const vb = 0.25 * (v[P - 1] + v[P] + v[P - 1 + nx] + v[P + nx]);
      const dudx = up3(c, u[k - 1], i >= 2 ? u[k - 2] : 0, u[k + 1], i <= nx - 2 ? u[k + 2] : 0, i >= 2, i <= nx - 2, c, dx);
      const dudy = bu === null ? up3(c, uS, j >= 2 ? u[k - 2 * nu1] : 0, uN, j <= ny - 3 ? u[k + 2 * nu1] : 0, j >= 2, j <= ny - 3, vb, dy) : up3(c, uS, j >= 2 ? u[k - 2 * nu1] : 0, uN, j <= ny - 3 ? u[k + 2 * nu1] : 0, j >= 2 && !bu[k - nu1] && !bu[k - 2 * nu1], j <= ny - 3 && !bu[k + nu1] && !bu[k + 2 * nu1], vb, dy);
      let vis = (2 * mu[P] * (u[k + 1] - c) - 2 * mu[P - 1] * (c - u[k - 1])) / (dx * dx);
      const tn = j < ny - 1 ? muC(i, j + 1) * ((uN - c) / dy + (v[P + nx] - v[P + nx - 1]) / dx) : 0.5 * (mu[P - 1] + mu[P]) * ((uN - c) / dy);
      const ts = j > 0 ? muC(i, j) * ((c - uS) / dy + (v[P] - v[P - 1]) / dx) : 0.5 * (mu[P - 1] + mu[P]) * ((c - uS) / dy);
      vis += (tn - ts) / dy;
      us[k] = c + dt * (-(c * dudx + vb * dudy) + vis / rf + (rr ? gx * (1 - rr / rf) : gx) + (sig > 0 ? (sig * kFace(P - 1, P) * (a[P] - a[P - 1])) / (dx * rf) : 0));
    }
    if (flow) for (let j = 0; j < ny; j++) { const k = j * nu1; us[k] = uinF[j]; us[k + nx] = solid && solid[j * nx + nx - 1] ? 0 : us[k + nx - 1] > 0 || !(bu !== null && bu[k + nx - 1]) ? us[k + nx - 1] : 0; } // inlet profile; zero streamwise gradient at the outlet
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i, c = v[k], rf = 0.5 * (rho[k - nx] + rho[k]);
      if (bv !== null && bv[k]) { vs[k] = 0; continue; }
      const vW = i > 0 ? (bv !== null && bv[k - 1] ? -c : v[k - 1]) : flow ? -c : sS * c, vE = i < nx - 1 ? (bv !== null && bv[k + 1] ? -c : v[k + 1]) : flow ? c : sS * c;
      const ub = 0.25 * (u[(j - 1) * nu1 + i] + u[(j - 1) * nu1 + i + 1] + u[j * nu1 + i] + u[j * nu1 + i + 1]);
      const dvdx = bv === null ? up3(c, vW, i >= 2 ? v[k - 2] : 0, vE, i <= nx - 3 ? v[k + 2] : 0, i >= 2, i <= nx - 3, ub, dx) : up3(c, vW, i >= 2 ? v[k - 2] : 0, vE, i <= nx - 3 ? v[k + 2] : 0, i >= 2 && !bv[k - 1] && !bv[k - 2], i <= nx - 3 && !bv[k + 1] && !bv[k + 2], ub, dx);
      const dvdy = up3(c, v[k - nx], j >= 2 ? v[k - 2 * nx] : 0, v[k + nx], j <= ny - 2 ? v[k + 2 * nx] : 0, j >= 2, j <= ny - 2, c, dy);
      let vis = (2 * mu[k] * (v[k + nx] - c) - 2 * mu[k - nx] * (c - v[k - nx])) / (dy * dy);
      const te = i < nx - 1 ? muC(i + 1, j) * ((vE - c) / dx + (u[j * nu1 + i + 1] - u[(j - 1) * nu1 + i + 1]) / dy) : 0.5 * (mu[k - nx] + mu[k]) * ((vE - c) / dx);
      const tw = i > 0 ? muC(i, j) * ((c - vW) / dx + (u[j * nu1 + i] - u[(j - 1) * nu1 + i]) / dy) : 0.5 * (mu[k - nx] + mu[k]) * ((c - vW) / dx);
      vis += (te - tw) / dx;
      vs[k] = c + dt * (-(ub * dvdx + c * dvdy) + vis / rf + (rr ? gy * (1 - rr / rf) : gy) + (sig > 0 ? (sig * kFace(k - nx, k) * (a[k] - a[k - nx])) / (dy * rf) : 0));
    }
  };
  let pIters = 0;
  const project = (dt) => { // ∇·((1/ρ)∇p) = ∇·u*/Δt, zero normal velocity on the box
    if (!msk) for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      pE[P] = i < nx - 1 ? (2 * dy) / (dx * (rho[P] + rho[P + 1])) : 0;
      pN[P] = j < ny - 1 ? (2 * dx) / (dy * (rho[P] + rho[P + nx])) : 0;
      rhs[P] = -((us[j * nu1 + i + 1] - us[j * nu1 + i]) * dy + (vs[P + nx] - vs[P]) * dx) / dt;
    }
    else for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { // blocked faces carry no pressure coupling; solid cells drop out (zero diagonal)
      if (solid && solid[P]) { pE[P] = pN[P] = rhs[P] = 0; continue; }
      pE[P] = i < nx - 1 && !(solid && solid[P + 1]) ? (2 * dy) / (dx * (rho[P] + rho[P + 1])) : 0;
      pN[P] = j < ny - 1 && !(solid && solid[P + nx]) ? (2 * dx) / (dy * (rho[P] + rho[P + nx])) : 0;
      rhs[P] = -((us[j * nu1 + i + 1] - us[j * nu1 + i]) * dy + (vs[P + nx] - vs[P]) * dx) / dt;
    }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) pD[P] = pE[P] + pN[P] + (i > 0 ? pE[P - 1] : 0) + (j > 0 ? pN[P - nx] : 0);
    if (flow) { for (let j = 0; j < ny; j++) { const P = j * nx + nx - 1; if (!(solid && solid[P])) pD[P] += dy / (dx * rho[P]); } } // pressure outlet: p = 0 one cell beyond the last column
    else if (solid) { let P0 = 0; while (P0 < n - 1 && solid[P0]) P0++; pD[P0] += pE[P0] + pN[P0] + 1e-30; }
    else pD[0] += pE[0] + pN[0]; // reference pressure: removes the null space of the all-Neumann problem
    const r = Bs ? Bs.solve(pE, pN, pD, rhs, p, o.pTol ?? 1e-9, 200) : pcg5(nx, ny, pE, pN, pD, rhs, p, o.pTol ?? 1e-9, 2000, Wcg);
    pIters += r.iters;
    if (!msk) {
      for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) { const P = j * nx + i; u[j * nu1 + i] = us[j * nu1 + i] - (dt * 2 * (p[P] - p[P - 1])) / (dx * (rho[P - 1] + rho[P])); }
      for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) { const k = j * nx + i; v[k] = vs[k] - (dt * 2 * (p[k] - p[k - nx])) / (dy * (rho[k - nx] + rho[k])); }
      return;
    }
    for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) { const P = j * nx + i, k = j * nu1 + i; u[k] = bu !== null && bu[k] ? 0 : us[k] - (dt * 2 * (p[P] - p[P - 1])) / (dx * (rho[P - 1] + rho[P])); }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) { const k = j * nx + i; v[k] = bv !== null && bv[k] ? 0 : vs[k] - (dt * 2 * (p[k] - p[k - nx])) / (dy * (rho[k - nx] + rho[k])); }
    if (flow) for (let j = 0; j < ny; j++) { const k = j * nu1, P = j * nx + nx - 1; u[k] = us[k]; u[k + nx] = solid && solid[P] ? 0 : us[k + nx] + (dt * p[P]) / (dx * rho[P]); }
  };
  let volIn = 0, volOut = 0, tNow = 0; // phase-A volume carried in through the inlet and out through the outlet
  const aInNow = (j) => (aInF ? clamp(aInF(tNow, (j + 0.5) * dy), 0, 1) : 0);
  // THINC/WLIC volume-fraction fluxes (Xiao et al. 2005; Yokoi 2007)
  const BETA = 3.5, CB = Math.cosh(BETA), SB = Math.sinh(BETA);
  const thinc = (aC, aM, aPl, c, wgt) => {
    const ac = Math.abs(c), upw = ac * aC;
    if (!(wgt > 0) || aC < 1e-8 || aC > 1 - 1e-8 || (aPl - aC) * (aC - aM) <= 0) return c >= 0 ? upw : -upw;
    const gm = aPl > aM ? 1 : -1, xt = Math.atanh(clamp((CB - Math.exp((BETA * (2 * aC - 1)) / gm)) / SB, -0.999999999999, 0.999999999999)) / BETA;
    const I = c > 0 ? 0.5 * (ac + (gm / BETA) * Math.log(Math.cosh(BETA * (1 - xt)) / Math.cosh(BETA * (1 - ac - xt)))) : 0.5 * (ac + (gm / BETA) * Math.log(Math.cosh(BETA * (ac - xt)) / Math.cosh(BETA * xt)));
    const F = wgt * clamp(I, 0, ac) + (1 - wgt) * upw;
    return c >= 0 ? F : -F;
  };
  const normals = () => {
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      const ex = Math.abs(a[i < nx - 1 ? P + 1 : P] - a[i > 0 ? P - 1 : P]) / dx, ey = Math.abs(a[j < ny - 1 ? P + nx : P] - a[j > 0 ? P - nx : P]) / dy;
      wX[P] = ex + ey > 1e-12 ? ex / (ex + ey) : 0.5;
    }
  };
  const sweepX = (dt, implicit) => {
    normals();
    for (let j = 0; j < ny; j++) {
      let fw = 0;
      if (flow) { const ui = u[j * nu1]; fw = ((ui * dt) / dx) * (ui > 0 ? aInNow(j) : a[j * nx]); volIn += fw * dx * dy; }
      for (let i = 0; i < nx; i++) {
        const P = j * nx + i, ue = u[j * nu1 + i + 1];
        let fe = 0;
        if (i < nx - 1) fe = ue >= 0 ? thinc(a[P], i > 0 ? a[P - 1] : a[P], a[P + 1], (ue * dt) / dx, wX[P]) : thinc(a[P + 1], a[P], i < nx - 2 ? a[P + 2] : a[P + 1], (ue * dt) / dx, wX[P + 1]);
        else if (flow) { fe = ((ue * dt) / dx) * a[P]; volOut += fe * dx * dy; }
        const dil = (dt / dx) * (ue - u[j * nu1 + i]);
        an[P] = solid !== null && solid[P] ? a[P] : implicit ? (a[P] - (fe - fw)) / (1 - dil) : a[P] * (1 + dil) - (fe - fw);
        fw = fe;
      }
    }
    a.set(an);
  };
  const sweepY = (dt, implicit) => {
    normals();
    for (let i = 0; i < nx; i++) {
      let fs = 0;
      for (let j = 0; j < ny; j++) {
        const P = j * nx + i, vn = v[P + nx];
        let fn = 0;
        if (j < ny - 1) fn = vn >= 0 ? thinc(a[P], j > 0 ? a[P - nx] : a[P], a[P + nx], (vn * dt) / dy, 1 - wX[P]) : thinc(a[P + nx], a[P], j < ny - 2 ? a[P + 2 * nx] : a[P + nx], (vn * dt) / dy, 1 - wX[P + nx]);
        const dil = (dt / dy) * (vn - v[P]);
        an[P] = solid !== null && solid[P] ? a[P] : implicit ? (a[P] - (fn - fs)) / (1 - dil) : a[P] * (1 + dil) - (fn - fs);
        fs = fn;
      }
    }
    a.set(an);
  };
  // level set: transport, redistancing and the volume (mass) correction
  const rhsLS = (ff, r) => {
    for (let j = 0, P = 0; j < ny; j++) {
      const jm = j > 0 ? nx : 0, jp = j < ny - 1 ? nx : 0, jmm = j > 1 ? 2 * nx : jm, jpp = j < ny - 2 ? 2 * nx : jp;
      for (let i = 0; i < nx; i++, P++) {
        const im = i > 0 ? 1 : 0, ip = i < nx - 1 ? 1 : 0, imm = i > 1 ? 2 : im, ipp = i < nx - 2 ? 2 : ip, c = ff[P];
        const ex = ucc[P] >= 0 ? (2 * ff[P + ip] + 3 * c - 6 * ff[P - im] + ff[P - imm]) / (6 * dx) : -(2 * ff[P - im] + 3 * c - 6 * ff[P + ip] + ff[P + ipp]) / (6 * dx);
        const ey = vcc[P] >= 0 ? (2 * ff[P + jp] + 3 * c - 6 * ff[P - jm] + ff[P - jmm]) / (6 * dy) : -(2 * ff[P - jm] + 3 * c - 6 * ff[P + jp] + ff[P + jpp]) / (6 * dy);
        r[P] = -(ucc[P] * ex + vcc[P] * ey);
      }
    }
  };
  const redistance = (its) => { // ∂φ/∂τ = S(φ0)(1 − |∇φ|), Godunov upwinding
    for (let P = 0; P < n; P++) p3[P] = phi[P] / Math.sqrt(phi[P] * phi[P] + h * h);
    for (let it = 0; it < its; it++) {
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        const c = phi[P], a1 = i > 0 ? (c - phi[P - 1]) / dx : 0, b1 = i < nx - 1 ? (phi[P + 1] - c) / dx : 0, c1 = j > 0 ? (c - phi[P - nx]) / dy : 0, d1 = j < ny - 1 ? (phi[P + nx] - c) / dy : 0, s = p3[P];
        const g2 = s > 0 ? Math.max(Math.max(a1, 0) ** 2, Math.min(b1, 0) ** 2) + Math.max(Math.max(c1, 0) ** 2, Math.min(d1, 0) ** 2) : Math.max(Math.min(a1, 0) ** 2, Math.max(b1, 0) ** 2) + Math.max(Math.min(c1, 0) ** 2, Math.max(d1, 0) ** 2);
        p1[P] = c - 0.3 * h * s * (Math.sqrt(g2) - 1);
      }
      phi.set(p1);
    }
  };
  let lsShift = 0;
  const massFix = () => { // constant shift c with ∫H(φ + c) dA = initial volume (secant iteration)
    const tgt = flow ? vol0 + volIn - volOut : vol0; // with through-flow the target follows the volume carried in and out
    if (flow && !(tgt > 1e-9 * W * Hh)) return;
    let c0 = 0, f0 = lsVol(0) - tgt;
    if (Math.abs(f0) < 1e-13 * W * Hh) return;
    let c1 = -f0 / Math.max(1e-30, (lsVol(0.1 * h) - lsVol(-0.1 * h)) / (0.2 * h)), f1 = lsVol(c1) - tgt;
    for (let k = 0; k < 6 && Math.abs(f1) > 1e-12 * W * Hh && f1 !== f0; k++) { const c2 = c1 - (f1 * (c1 - c0)) / (f1 - f0); c0 = c1; f0 = f1; c1 = c2; f1 = lsVol(c1) - tgt; }
    if (!Number.isFinite(c1) || Math.abs(c1) > 2 * h) return;
    for (let P = 0; P < n; P++) phi[P] += c1;
    lsShift += Math.abs(c1);
  };
  const advect = (dt, s) => {
    if (!ls) { if (s % 2) { sweepY(dt, true); sweepX(dt, false); } else { sweepX(dt, true); sweepY(dt, false); } for (let P = 0; P < n; P++) a[P] = a[P] < 0 ? 0 : a[P] > 1 ? 1 : a[P]; if (solid !== null) fillS(a); return; }
    if (flow) for (let j = 0; j < ny; j++) { const ui = u[j * nu1], ue = u[j * nu1 + nx]; if (ui > 0) volIn += ui * dt * dy * aInNow(j); if (ue > 0 && !(solid !== null && solid[j * nx + nx - 1])) volOut += ue * dt * dy * a[j * nx + nx - 1]; }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { ucc[P] = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); vcc[P] = 0.5 * (v[P] + v[P + nx]); }
    rhsLS(phi, p1); for (let P = 0; P < n; P++) p2[P] = phi[P] + dt * p1[P];
    rhsLS(p2, an); for (let P = 0; P < n; P++) phi[P] += 0.5 * dt * (p1[P] + an[P]);
    if (solid !== null) fillS(phi);
    redistance(o.reinit ?? 2); massFix();
    if (solid !== null) fillS(phi);
    for (let P = 0; P < n; P++) a[P] = heavi(phi[P]);
  };
  // time step: convective, capillary (Brackbill), viscous and gravity-wave limits
  const nuMax = Math.max(mA / rA, mB / rB), dtCap = sig > 0 ? Math.sqrt(((rA + rB) * h ** 3) / (4 * Math.PI * sig)) : Infinity, gm = Math.hypot(gx, gy);
  const dtFix = Math.min(0.5 * dtCap, (0.2 * h * h) / nuMax, gm > 0 ? 0.25 * Math.sqrt(h / gm) : Infinity, o.dtMax ?? Infinity), cfl = o.cfl ?? 0.25;
  const hist = { t: [], vol: [], xc: [], yc: [], vr: [], umax: [], hL: [], hR: [], xf: [], circ: [], ke: [], inn: [], out: [], dp: [], xLead: [], xTrail: [], tauMean: [], tauMax: [] };
  const diag = (t) => {
    let s = 0, sx = 0, sy = 0, sv = 0, um = 0, per = 0, ke = 0, hL = 0, hR = 0, xf = 0;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid !== null && solid[P]) continue;
      const c = a[P], uc = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]), vc = 0.5 * (v[P] + v[P + nx]), sp = uc * uc + vc * vc;
      s += c; sx += c * (i + 0.5) * dx; sy += c * (j + 0.5) * dy; sv += c * vc; if (sp > um) um = sp; ke += 0.5 * (rB + (rA - rB) * c) * sp;
      const ex = (a[i < nx - 1 ? P + 1 : P] - a[i > 0 ? P - 1 : P]) / ((i > 0 && i < nx - 1 ? 2 : 1) * dx), ey = (a[j < ny - 1 ? P + nx : P] - a[j > 0 ? P - nx : P]) / ((j > 0 && j < ny - 1 ? 2 : 1) * dy);
      per += Math.hypot(ex, ey);
      if (i === 0) hL += (1 - c) * dy; if (i === nx - 1) hR += (1 - c) * dy; if (j === 0) xf += (1 - c) * dx;
    }
    const V = s * dx * dy;
    hist.t.push(t); hist.vol.push(V); hist.xc.push(s > 0 ? sx / s : 0); hist.yc.push(s > 0 ? sy / s : 0); hist.vr.push(s > 0 ? sv / s : 0); hist.umax.push(Math.sqrt(um)); hist.hL.push(hL); hist.hR.push(hR); hist.xf.push(xf);
    hist.circ.push(per > 0 ? (2 * Math.sqrt(Math.PI * V)) / (per * dx * dy) : 0); hist.ke.push(ke * dx * dy);
    if (flow) { // volumes through the open boundaries, pressure drop inlet − outlet column, and the leading and trailing edge of phase A along x
      let pa2 = 0, na2 = 0, pb2 = 0, nb2 = 0, xl = 0, xt = -1;
      for (let j = 0; j < ny; j++) { const A0 = j * nx, B0 = A0 + nx - 1; if (!(solid !== null && solid[A0])) { pa2 += p[A0]; na2++; } if (!(solid !== null && solid[B0])) { pb2 += p[B0]; nb2++; } }
      for (let i = 0; i < nx; i++) { let q = 0; for (let j = 0; j < ny; j++) { const P = j * nx + i; if (!(solid !== null && solid[P]) && a[P] > q) q = a[P]; } if (q > 0.5) { xl = (i + 1) * dx; if (xt < 0) xt = i * dx; } }
      let ts = 0, tm = 0, tn = 0; // shear stress on the channel walls from the wall-adjacent cells
      for (let i = 0; i < nx; i++) for (const j of [0, ny - 1]) { const P = j * nx + i; if (solid !== null && solid[P]) continue; const tw = (mu[P] * Math.abs(0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]))) / (0.5 * dy); ts += tw; tn++; if (tw > tm) tm = tw; }
      hist.tauMean.push(tn ? ts / tn : 0); hist.tauMax.push(tm);
      hist.inn.push(volIn); hist.out.push(volOut); hist.dp.push((na2 ? pa2 / na2 : 0) - (nb2 ? pb2 / nb2 : 0)); hist.xLead.push(xl); hist.xTrail.push(xt < 0 ? 0 : xt);
    }
    return Math.sqrt(um);
  };
  let t = 0, steps = 0, um = diag(0), amin = 0, amax = 1;
  const maxSteps = o.maxSteps ?? 20000, tEnd = o.tEnd, snapT = (o.snaps || []).filter((x) => x > 0 && x < tEnd).sort((x, y) => x - y), snaps = [];
  let snapK = 0;
  while (t < tEnd * (1 - 1e-12) && steps < maxSteps) {
    let vmax = 1e-300;
    for (let k = 0; k < u.length; k++) { const q = Math.abs(u[k]) / dx; if (q > vmax) vmax = q; }
    for (let k = 0; k < v.length; k++) { const q = Math.abs(v[k]) / dy; if (q > vmax) vmax = q; }
    const dt = Math.min(dtFix, cfl / vmax, tEnd - t);
    tNow = t;
    props(); curvature(); predictor(dt); project(dt); advect(dt, steps);
    t += dt; steps++;
    um = diag(t);
    if (!Number.isFinite(um)) throw new Error('The two-phase solution diverged. Refine the grid, lower the CFL number or reduce the density and viscosity ratio.');
    if (!ls) for (let P = 0; P < n; P++) { if (a[P] < amin) amin = a[P]; if (a[P] > amax) amax = a[P]; }
    while (snapK < snapT.length && t >= snapT[snapK] * (1 - 1e-12)) { snaps.push({ t, a: Float64Array.from(a) }); snapK++; }
    if (steps % 20 === 0) { ctx?.progress?.(Math.min(0.98, t / tEnd), `Two-phase flow: t = ${t.toExponential(2)} s of ${tEnd.toExponential(2)} s (step ${steps})`); if (ctx?.tick) await ctx.tick(); }
  }
  props();
  // pressure jump across the interface: mean pressure inside phase A minus mean pressure in phase B (hydrostatic part removed)
  let pa = 0, na = 0, pb = 0, nb = 0;
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (solid !== null && solid[P]) continue; const q = p[P] - rB * (gx * (i + 0.5) * dx + gy * (j + 0.5) * dy); if (a[P] > 0.999) { pa += q; na++; } else if (a[P] < 0.001) { pb += q; nb++; } }
  let div = 0;
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (solid !== null && solid[P]) continue; div = Math.max(div, Math.abs((u[j * nu1 + i + 1] - u[j * nu1 + i]) / dx + (v[P + nx] - v[P]) / dy)); }
  return { nx, ny, dx, dy, W, Hh, a, phi, u, v, p, rho, kap, hist, snaps, t, steps, vol0, vol: hist.vol[hist.vol.length - 1], dpJump: na && nb ? pa / na - pb / nb : 0, umax: um, div, pIters, pFactors: Bs ? Bs.factors : 0, lsShift, done: t >= tEnd * (1 - 1e-9), method: ls ? 'ls' : 'vof', dtFix, solid, volIn, volOut, through: !!flow };
}

/** Schiller–Naumann drag coefficient times the particle Reynolds number, C_D·Re = 24 (1 + 0.15 Re^0.687) (0.44 Re above Re = 1000). */
export const schillerNaumann = (Re) => (Re < 1000 ? 24 * (1 + 0.15 * Re ** 0.687) : 0.44 * Re);

/** Terminal slip velocity of one sphere from the Schiller–Naumann drag–buoyancy balance (positive number). */
export function terminalSN(d, rhoC, rhoD, muC, g = 9.80665) {
  const drv = (Math.abs(rhoD - rhoC) * g * d * d) / (18 * muC); // Stokes value
  let ut = drv;
  for (let k = 0; k < 200; k++) { const Re = (rhoC * ut * d) / muC, un = drv / (schillerNaumann(Re) / 24); if (Math.abs(un - ut) < 1e-14 * drv) { ut = un; break; } ut = 0.5 * (ut + un); }
  return ut;
}

/**
 * Eulerian–Eulerian two-fluid model on a uniform staggered grid: a continuous phase c and a dispersed phase d
 * (particles, droplets or bubbles of diameter dP) as interpenetrating continua that share one pressure field.
 *   ∂α_d/∂t + ∇·(α_d u_d) = 0,  α_c = 1 − α_d,  ∇·(α_c u_c + α_d u_d) = 0,
 *   α_k ρ_k (∂u_k/∂t + u_k·∇u_k) = −α_k ∇p + α_k ρ_k g + ∇·(α_k μ_k ∇u_k) ± K (u_d − u_c) ± F_vm ± F_td,
 * K = ¾ C_D α_d α_c ρ_c |u_r| α_c^−2.65 / dP with the Schiller–Naumann C_D(α_c Re) (Wen–Yu swarm correction, which
 * reduces to the single-sphere law as α_d → 0), virtual mass F_vm = C_vm α_d ρ_c (du_c/dt − du_d/dt) and turbulent
 * dispersion F_td = −K D_td ∇α_d/(α_d α_c). Drag and virtual mass are implicit: the two momentum equations are
 * solved together face by face (2 × 2 system, the partial-elimination idea), so the stiff relaxation of small
 * particles does not limit the time step. The shared pressure follows from the mixture volume balance, a Poisson
 * equation whose coefficient is the α-weighted phase mobility. α_d is advanced in flux form (van Leer) and kept
 * below the packing limit by a flux limiter; the inter-phase friction rises steeply just below packing.
 * o = { nx, ny, L, H, rhoC, muC, rhoD, dP, gx, gy, alpha0 (number or f(x, y)), flow: { U, alphaIn } | null, cvm, Dtd,
 *       alphaMax, tEnd, cfl, slipC, maxSteps, deposit }. Without o.flow the box is closed; with it x = 0 is a velocity inlet and
 * x = L a pressure outlet. Walls: no-slip (or free-slip) for the continuous phase, free-slip for the dispersed phase.
 * deposit = true lets the dispersed phase leave through the wall it settles (or rises) onto, at its wall-normal velocity one
 * face inside; the same volume of continuous phase takes its place, so the mixture volume flux through the wall stays zero.
 */
export async function twoFluid2D(o, ctx) {
  const nx = o.nx, ny = o.ny, L = o.L, H = o.H, dx = L / nx, dy = H / ny, n = nx * ny, nu1 = nx + 1, h = Math.min(dx, dy);
  const rc = o.rhoC, rd = o.rhoD, muc = o.muC, nuc = muc / rc, dP = o.dP, gx = o.gx || 0, gy = o.gy ?? -9.80665, aMax = o.alphaMax ?? 0.6, Vm = (o.cvm ?? 0) * rc, Dtd = Math.max(0, o.Dtd || 0);
  const flow = o.flow || null, Uin = flow ? flow.U : 0, aIn = flow ? clamp(flow.alphaIn, 0, 0.95 * aMax) : 0, sC = o.slipC ? 1 : -1;
  // The solved pressure is the excess over the hydrostatic pressure of the continuous phase, p′ = p − ρ_c g·x: gravity then drops out of the
  // continuous-phase equation, acts on the dispersed phase as g (1 − ρ_c/ρ_d), and a uniform p′ is the correct condition on the outlet plane.
  const gdx = gx * (1 - rc / rd), gdy = gy * (1 - rc / rd);
  const f = (m) => new Float64Array(m), al = f(n), p = f(n), uc = f(nu1 * ny), vc = f(nx * (ny + 1)), ud = f(nu1 * ny), vd = f(nx * (ny + 1));
  const tuc = f(nu1 * ny), tvc = f(nx * (ny + 1)), tud = f(nu1 * ny), tvd = f(nx * (ny + 1)), afx = f(nu1 * ny), afy = f(nx * (ny + 1)), bcx = f(nu1 * ny), bdx = f(nu1 * ny), bcy = f(nx * (ny + 1)), bdy = f(nx * (ny + 1));
  const sgx = new Int8Array(nu1 * ny), sgy = new Int8Array(nx * (ny + 1)), Fx = f(nu1 * ny), Fy = f(nx * (ny + 1)), inn = f(n), out = f(n), frc = f(n), frd = f(n), pE = f(n), pN = f(n), pD = f(n), rhs = f(n);
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) al[P] = clamp(typeof o.alpha0 === 'function' ? o.alpha0((i + 0.5) * dx, (j + 0.5) * dy) : o.alpha0 || 0, 0, aMax);
  // Optional solid mask (blocked cells): no-slip for the continuous phase, free-slip and impermeable for the dispersed phase, which is
  // captured on the solid faces it settles (or rises) onto when o.deposit is set. flow.U is the mean velocity over the open inlet height.
  const solid = o.solid && o.solid.some((q) => q) ? o.solid : null, bu = solid ? new Uint8Array(nu1 * ny) : null, bv = solid ? new Uint8Array(nx * (ny + 1)) : null, capS = [];
  if (solid) {
    for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) bu[j * nu1 + i] = (i > 0 && solid[j * nx + i - 1]) || (i < nx && solid[j * nx + i]) ? 1 : 0;
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) bv[j * nx + i] = solid[(j - 1) * nx + i] || solid[j * nx + i] ? 1 : 0;
    for (let P = 0; P < n; P++) if (solid[P]) al[P] = 0;
    if (gdy !== 0) for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { // fluid cells resting on (settling) or hanging under (rising) a solid, with the face the phase arrives through
      if (solid[P]) continue;
      if (gdy < 0 ? j > 0 && solid[P - nx] && j < ny - 1 && !solid[P + nx] : j < ny - 1 && solid[P + nx] && j > 0 && !solid[P - nx]) capS.push([P, gdy < 0 ? P + nx : P]);
    }
  }
  if (flow) for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) {
    if (!solid) { uc[j * nu1 + i] = Uin; ud[j * nu1 + i] = Uin; continue; }
    let open = 0, inl = 0; for (let jj = 0; jj < ny; jj++) { if (!bu[jj * nu1 + i]) open++; if (!bu[jj * nu1]) inl++; }
    const q = bu[j * nu1 + i] || !open ? 0 : (Uin * inl) / open; uc[j * nu1 + i] = q; ud[j * nu1 + i] = q; // plug flow through the open height of every column
  }
  const Bs = bandSolver(nx, ny, 9e6, 2), Wcg = Bs ? null : { r: f(n), z: f(n), s: f(n), q: f(n), pc: f(n) };
  const lim = (d1, d2) => (d1 * d2 <= 0 ? 0 : (2 * d1 * d2) / (d1 + d2));
  // drag function k′ = K/α_d (kg/m³·s) from the local slip speed and the continuous-phase fraction
  const kDrag = (ur, ac, am) => {
    const Re = (ac * rc * ur * dP) / muc, k = (0.75 * schillerNaumann(Re) * muc * ac ** -2.65) / (dP * dP), q = (am - 0.9 * aMax) / (0.1 * aMax);
    return q > 0 ? k * (1 + 1e3 * q * q) : k;
  };
  const faceAlpha = (dt) => { // van Leer value of α_d on every face, upwind with respect to the dispersed-phase velocity
    for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) {
      const k = j * nu1 + i, P = j * nx + i, w = ud[k];
      sgx[k] = w >= 0 ? 1 : -1;
      if (i === 0) { afx[k] = flow && w >= 0 ? aIn : al[P]; continue; }
      if (i === nx) { afx[k] = al[P - 1]; continue; }
      const c = Math.abs(w) * dt / dx;
      if (w >= 0) afx[k] = al[P - 1] + 0.5 * (1 - c) * lim(i >= 2 ? al[P - 1] - al[P - 2] : 0, al[P] - al[P - 1]);
      else afx[k] = al[P] + 0.5 * (1 - c) * lim(i <= nx - 2 ? al[P] - al[P + 1] : 0, al[P - 1] - al[P]);
    }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i, w = vd[k], c = Math.abs(w) * dt / dy;
      sgy[k] = w >= 0 ? 1 : -1;
      if (w >= 0) afy[k] = al[k - nx] + 0.5 * (1 - c) * lim(j >= 2 ? al[k - nx] - al[k - 2 * nx] : 0, al[k] - al[k - nx]);
      else afy[k] = al[k] + 0.5 * (1 - c) * lim(j <= ny - 2 ? al[k] - al[k + nx] : 0, al[k - nx] - al[k]);
    }
  };
  // couple the two provisional face velocities through implicit drag and virtual mass; returns the pressure mobilities
  let Bc = 0, Bd = 0, Hc = 0, Hd = 0;
  const couple = (am, tc, td, kp, dt, ga, gd) => { // gd: reduced gravity on the dispersed phase along this face normal
    const ac = 1 - am, m = Vm + kp * dt, a11 = ac * rc + am * m, det = ac * rc * rd + ac * rc * m + am * m * rd;
    let r1 = ac * rc * tc + am * Vm * (tc - td), r2 = rd * (td + dt * gd) - Vm * (tc - td);
    if (Dtd > 0 && ga !== 0) { const q = (dt * kp * Dtd * ga) / ac; r1 += q; r2 -= q / Math.max(am, 1e-6); }
    Hc = ((rd + m) * r1 + am * m * r2) / det; Hd = (m * r1 + a11 * r2) / det;
    Bc = ((rd + m) * ac + am * m) / det; Bd = (m + ac * rc) / det;
  };
  const step = (dt) => {
    faceAlpha(dt);
    for (let P = 0; P < n; P++) { frc[P] = 1 - al[P]; frd[P] = al[P] > 1e-6 ? al[P] : 1e-6; }
    // ---- x faces
    for (let j = 0; j < ny; j++) for (let i = 1; i <= nx; i++) {
      const k = j * nu1 + i, P = j * nx + i;
      if (bu !== null && bu[k]) { tuc[k] = 0; tud[k] = 0; bcx[k] = 0; bdx[k] = 0; continue; }
      if (i === nx) { // pressure outlet: zero-gradient provisional velocity, mobility of the last cell
        if (!flow) continue;
        const am = al[P - 1]; couple(am, tuc[k - 1], tud[k - 1], kDrag(Math.abs(ud[k] - uc[k]), 1 - am, am), dt, 0, 0);
        tuc[k] = Hc; tud[k] = Hd; bcx[k] = Bc; bdx[k] = Bd; continue;
      }
      const am = 0.5 * (al[P - 1] + al[P]), ac = 1 - am;
      let pc = 0, pd = 0;
      for (let ph = 0; ph < 2; ph++) {
        const U = ph ? ud : uc, V = ph ? vd : vc, sw = ph ? 1 : sC, c = U[k], fr = ph ? frd : frc;
        const uS = j > 0 ? (bu !== null && bu[k - nu1] ? (ph ? c : -c) : U[k - nu1]) : sw * c, uN = j < ny - 1 ? (bu !== null && bu[k + nu1] ? (ph ? c : -c) : U[k + nu1]) : sw * c, vb = 0.25 * (V[P - 1] + V[P] + V[P - 1 + nx] + V[P + nx]);
        const adv = c * up3(c, U[k - 1], i >= 2 ? U[k - 2] : 0, U[k + 1], i <= nx - 2 ? U[k + 2] : 0, i >= 2, i <= nx - 2, c, dx) + vb * up3(c, uS, j >= 2 ? U[k - 2 * nu1] : 0, uN, j <= ny - 3 ? U[k + 2 * nu1] : 0, j >= 2, j <= ny - 3, vb, dy);
        // ∇·(α ν ∇u)/α with the phase fraction of the neighbouring cells / corners
        const fm = 0.5 * (fr[P - 1] + fr[P]), aN = j < ny - 1 ? 0.5 * fm + 0.25 * (fr[P - 1 + nx] + fr[P + nx]) : fm, aS = j > 0 ? 0.5 * fm + 0.25 * (fr[P - 1 - nx] + fr[P - nx]) : fm;
        const vis = (nuc * ((fr[P] * (U[k + 1] - c) - fr[P - 1] * (c - U[k - 1])) / (dx * dx) + (aN * (uN - c) - aS * (c - uS)) / (dy * dy))) / fm;
        const t = c + dt * (-adv + vis);
        if (ph) pd = t; else pc = t;
      }
      const vr = 0.25 * (vd[P - 1] + vd[P] + vd[P - 1 + nx] + vd[P + nx] - vc[P - 1] - vc[P] - vc[P - 1 + nx] - vc[P + nx]);
      couple(am, pc, pd, kDrag(Math.hypot(ud[k] - uc[k], vr), ac, am), dt, Dtd > 0 ? (al[P] - al[P - 1]) / dx : 0, gdx);
      tuc[k] = Hc; tud[k] = Hd; bcx[k] = Bc; bdx[k] = Bd;
    }
    if (flow) for (let j = 0; j < ny; j++) { const q = bu !== null && bu[j * nu1] ? 0 : Uin; tuc[j * nu1] = q; tud[j * nu1] = q; }
    // ---- y faces
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i, am = 0.5 * (al[k - nx] + al[k]), ac = 1 - am;
      if (bv !== null && bv[k]) { tvc[k] = 0; tvd[k] = 0; bcy[k] = 0; bdy[k] = 0; continue; }
      let pc = 0, pd = 0;
      for (let ph = 0; ph < 2; ph++) {
        const U = ph ? ud : uc, V = ph ? vd : vc, sw = ph || flow ? 1 : sC, c = V[k], fr = ph ? frd : frc;
        const inW = flow && i === 0 ? 0 : null; // inlet: zero cross-flow velocity on the boundary
        const vW = i > 0 ? (bv !== null && bv[k - 1] ? (ph ? c : -c) : V[k - 1]) : inW !== null ? -c : sw * c, vE = i < nx - 1 ? (bv !== null && bv[k + 1] ? (ph ? c : -c) : V[k + 1]) : flow ? c : sw * c;
        const ub = 0.25 * (U[(j - 1) * nu1 + i] + U[(j - 1) * nu1 + i + 1] + U[j * nu1 + i] + U[j * nu1 + i + 1]);
        const adv = ub * up3(c, vW, i >= 2 ? V[k - 2] : 0, vE, i <= nx - 3 ? V[k + 2] : 0, i >= 2, i <= nx - 3, ub, dx) + c * up3(c, V[k - nx], j >= 2 ? V[k - 2 * nx] : 0, V[k + nx], j <= ny - 2 ? V[k + 2 * nx] : 0, j >= 2, j <= ny - 2, c, dy);
        const fm = 0.5 * (fr[k] + fr[k - nx]), aE = i < nx - 1 ? 0.5 * fm + 0.25 * (fr[k + 1] + fr[k - nx + 1]) : fm, aW = i > 0 ? 0.5 * fm + 0.25 * (fr[k - 1] + fr[k - nx - 1]) : fm;
        const vis = (nuc * ((aE * (vE - c) - aW * (c - vW)) / (dx * dx) + (fr[k] * (V[k + nx] - c) - fr[k - nx] * (c - V[k - nx])) / (dy * dy))) / fm;
        const t = c + dt * (-adv + vis);
        if (ph) pd = t; else pc = t;
      }
      const ur = 0.25 * (ud[(j - 1) * nu1 + i] + ud[(j - 1) * nu1 + i + 1] + ud[j * nu1 + i] + ud[j * nu1 + i + 1] - uc[(j - 1) * nu1 + i] - uc[(j - 1) * nu1 + i + 1] - uc[j * nu1 + i] - uc[j * nu1 + i + 1]);
      couple(am, pc, pd, kDrag(Math.hypot(ur, vd[k] - vc[k]), ac, am), dt, Dtd > 0 ? (al[k] - al[k - nx]) / dy : 0, gdy);
      tvc[k] = Hc; tvd[k] = Hd; bcy[k] = Bc; bdy[k] = Bd;
    }
    // ---- shared pressure from the mixture volume balance ∇·(α_c u_c + α_d u_d) = 0
    const jx = (k) => (1 - afx[k]) * tuc[k] + afx[k] * tud[k], jy = (k) => (1 - afy[k]) * tvc[k] + afy[k] * tvd[k];
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      const ke = j * nu1 + i + 1, kn = P + nx;
      pE[P] = i < nx - 1 ? (dt * ((1 - afx[ke]) * bcx[ke] + afx[ke] * bdx[ke]) * dy) / dx : 0;
      pN[P] = j < ny - 1 ? (dt * ((1 - afy[kn]) * bcy[kn] + afy[kn] * bdy[kn]) * dx) / dy : 0;
      rhs[P] = -((jx(ke) - jx(ke - 1)) * dy + ((j < ny - 1 ? jy(kn) : 0) - (j > 0 ? jy(P) : 0)) * dx);
    }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) pD[P] = pE[P] + pN[P] + (i > 0 ? pE[P - 1] : 0) + (j > 0 ? pN[P - nx] : 0);
    if (flow) for (let j = 0; j < ny; j++) { const k = j * nu1 + nx; pD[j * nx + nx - 1] += (dt * ((1 - afx[k]) * bcx[k] + afx[k] * bdx[k]) * dy) / dx; }
    else if (solid) { let P0 = 0; while (P0 < n - 1 && solid[P0]) P0++; pD[P0] += pE[P0] + pN[P0] + 1e-30; }
    else pD[0] += pE[0] + pN[0];
    if (Bs) Bs.solve(pE, pN, pD, rhs, p, 1e-10, 200); else pcg5(nx, ny, pE, pN, pD, rhs, p, 1e-10, 2000, Wcg);
    for (let j = 0; j < ny; j++) for (let i = 1; i <= nx; i++) {
      const k = j * nu1 + i, P = j * nx + i;
      if (i === nx) { if (flow) { uc[k] = tuc[k] + (dt * bcx[k] * p[P - 1]) / dx; ud[k] = tud[k] + (dt * bdx[k] * p[P - 1]) / dx; } continue; }
      const gp = (p[P] - p[P - 1]) / dx; uc[k] = tuc[k] - dt * bcx[k] * gp; ud[k] = tud[k] - dt * bdx[k] * gp;
    }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) { const k = j * nx + i, gp = (p[k] - p[k - nx]) / dy; vc[k] = tvc[k] - dt * bcy[k] * gp; vd[k] = tvd[k] - dt * bdy[k] * gp; }
    // ---- dispersed-phase continuity in flux form, limited so that no cell exceeds the packing fraction
    inn.fill(0); out.fill(0);
    for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) {
      const k = j * nu1 + i, P = j * nx + i, w = ud[k];
      let af = afx[k];
      if (i > 0 && i < nx && (w >= 0 ? 1 : -1) !== sgx[k]) af = w >= 0 ? al[P - 1] : al[P];
      if (i === nx && w < 0) af = 0;
      const F = af * w * dt * dy; Fx[k] = F;
      if (F > 0) { if (i < nx) inn[P] += F; if (i > 0) out[P - 1] += F; } else if (F < 0) { if (i > 0) inn[P - 1] -= F; if (i < nx) out[P] -= F; }
    }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i, w = vd[k], F = ((w >= 0 ? 1 : -1) !== sgy[k] ? (w >= 0 ? al[k - nx] : al[k]) : afy[k]) * w * dt * dx; Fy[k] = F;
      if (F > 0) { inn[k] += F; out[k - nx] += F; } else if (F < 0) { inn[k - nx] -= F; out[k] -= F; }
    }
    // a face flux is scaled by the room left in the receiving cell and by the content of the donor cell (positivity)
    const room = (P) => (inn[P] > 0 ? Math.min(1, (Math.max(0, aMax - al[P]) * dx * dy) / inn[P]) : 1), have = (P) => (out[P] > al[P] * dx * dy ? (al[P] * dx * dy) / out[P] : 1);
    for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) { const k = j * nu1 + i, P = j * nx + i; Fx[k] *= Fx[k] > 0 ? Math.min(room(P), have(P - 1)) : Math.min(room(P - 1), have(P)); }
    if (flow) for (let j = 0; j < ny; j++) { if (Fx[j * nu1] > 0) Fx[j * nu1] *= room(j * nx); else Fx[j * nu1] *= have(j * nx); if (Fx[j * nu1 + nx] > 0) Fx[j * nu1 + nx] *= have(j * nx + nx - 1); }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) { const k = j * nx + i; Fy[k] *= Fy[k] > 0 ? Math.min(room(k), have(k - nx)) : Math.min(room(k - nx), have(k)); }
    let qIn = 0, qOut = 0, cIn = 0, cOut = 0, clip = 0;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      const q = al[P] - (Fx[j * nu1 + i + 1] - Fx[j * nu1 + i] + (j < ny - 1 ? Fy[P + nx] : 0) - (j > 0 ? Fy[P] : 0)) / (dx * dy);
      if (q < 0) { clip -= q; al[P] = 0; } else al[P] = q;
    }
    if (flow) for (let j = 0; j < ny; j++) { qIn += Fx[j * nu1]; qOut += Fx[j * nu1 + nx]; const ki = j * nu1, ko = ki + nx; cIn += ((1 - afx[ki]) * uc[ki] + afx[ki] * ud[ki]) * dt * dy - Fx[ki]; cOut += ((1 - afx[ko]) * uc[ko] + afx[ko] * ud[ko]) * dt * dy - Fx[ko]; } // continuous phase = mixture volume flux of the pressure equation − dispersed flux actually transported
    let dep = 0;
    if (o.deposit && gdy !== 0 && ny > 1) { // capture on the wall the dispersed phase moves toward
      const jw = gdy < 0 ? 0 : ny - 1, jf = gdy < 0 ? 1 : ny - 1;
      // with obstacles the liquid is deflected toward and along the walls and carries the dispersed phase with it without depositing it:
      // there the capture uses the velocity of the dispersed phase relative to the liquid (its settling or rise velocity) instead of its own
      for (let i = 0; i < nx; i++) { const P = jw * nx + i, w = solid !== null ? vd[jf * nx + i] - vc[jf * nx + i] : vd[jf * nx + i]; if (solid !== null && solid[P]) continue; if (gdy < 0 ? w < 0 : w > 0) { const d = al[P] * Math.min(1, (Math.abs(w) * dt) / dy); al[P] -= d; dep += d; } }
    }
    let depS = 0;
    if (o.deposit) for (const [P, kf] of capS) { const w = vd[kf] - vc[kf]; if (gdy < 0 ? w < 0 : w > 0) { const d = al[P] * Math.min(1, (Math.abs(w) * dt) / dy); al[P] -= d; depS += d; } } // capture on the solid surfaces that face the settling (or rising) phase
    return { qIn, qOut, cIn, cOut, clip: clip * dx * dy, dep: (dep + depS) * dx * dy, depS: depS * dx * dy };
  };
  const hist = { t: [], vol: [], front: [], slip: [], amax: [], dIn: [], dOut: [], cIn: [], cOut: [], umax: [], dep: [], depS: [], dp: [] };
  const a0m = typeof o.alpha0 === 'number' ? o.alpha0 : 0, gdir = Math.abs(gy) >= Math.abs(gx) ? 1 : 0;
  let dIn = 0, dOut = 0, cInT = 0, cOutT = 0, clipT = 0, depT = 0, depST = 0;
  const diag = (t) => {
    let s = 0, am = 0, um = 0, ws = 0, sl = 0;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      const a = al[P]; s += a; if (a > am) am = a;
      const ur = 0.5 * (ud[j * nu1 + i] + ud[j * nu1 + i + 1] - uc[j * nu1 + i] - uc[j * nu1 + i + 1]), vr = 0.5 * (vd[P] + vd[P + nx] - vc[P] - vc[P + nx]);
      um = Math.max(um, Math.abs(uc[j * nu1 + i]), Math.abs(vc[P]), Math.abs(ud[j * nu1 + i]), Math.abs(vd[P]));
      if (a0m > 0 ? Math.abs(a - a0m) < 0.02 * a0m : a > 1e-9) { ws += a; sl += a * (gdir ? vr : ur); }
    }
    // suspension front along gravity: outermost position where the section-mean fraction crosses half the initial value
    let front = 0;
    if (a0m > 0 && !flow && gdir) {
      const prof = new Float64Array(ny); for (let j = 0; j < ny; j++) { let q = 0; for (let i = 0; i < nx; i++) q += al[j * nx + i]; prof[j] = q / nx; }
      const down = rd > rc ? gy < 0 : gy > 0; // the dispersed phase moves toward −y
      if (down) { front = 0; for (let j = ny - 1; j >= 0; j--) if (prof[j] >= 0.5 * a0m) { const up = j < ny - 1 ? prof[j + 1] : 0; front = (j + 0.5) * dy + (dy * (prof[j] - 0.5 * a0m)) / Math.max(prof[j] - up, 1e-300); break; } }
      else { front = H; for (let j = 0; j < ny; j++) if (prof[j] >= 0.5 * a0m) { const lo = j > 0 ? prof[j - 1] : 0; front = (j + 0.5) * dy - (dy * (prof[j] - 0.5 * a0m)) / Math.max(prof[j] - lo, 1e-300); break; } }
    }
    hist.t.push(t); hist.vol.push(s * dx * dy); hist.front.push(front); hist.slip.push(ws > 0 ? sl / ws : 0); hist.amax.push(am); hist.dIn.push(dIn); hist.dOut.push(dOut); hist.cIn.push(cInT); hist.cOut.push(cOutT); hist.umax.push(um); hist.dep.push(depT); hist.depS.push(depST);
    if (flow) { let pa = 0, na = 0, pb = 0, nb = 0; for (let j = 0; j < ny; j++) { const A0 = j * nx, B0 = A0 + nx - 1; if (!(solid !== null && solid[A0])) { pa += p[A0]; na++; } if (!(solid !== null && solid[B0])) { pb += p[B0]; nb++; } } hist.dp.push((na ? pa / na : 0) - (nb ? pb / nb : 0)); }
    return um;
  };
  const vol0 = (() => { let s = 0; for (let P = 0; P < n; P++) s += al[P]; return s * dx * dy; })();
  const ut = terminalSN(dP, rc, rd, muc, Math.hypot(gx, gy)), cfl = o.cfl ?? 0.3, dtV = Math.min((0.2 * h * h) / nuc, Dtd > 0 ? (0.2 * h * h) / Dtd : Infinity, o.dtMax ?? Infinity);
  let t = 0, steps = 0, um = diag(0);
  const tEnd = o.tEnd, maxSteps = o.maxSteps ?? 20000, every = Math.max(1, o.sample ?? 1);
  while (t < tEnd * (1 - 1e-12) && steps < maxSteps) {
    const dt = Math.min(dtV, (cfl * h) / Math.max(um, ut, Math.abs(Uin), 1e-12), tEnd - t);
    const r = step(dt);
    dIn += r.qIn; dOut += r.qOut; cInT += r.cIn; cOutT += r.cOut; clipT += r.clip; depT += r.dep; depST += r.depS;
    t += dt; steps++;
    if (steps % every === 0 || t >= tEnd * (1 - 1e-12)) um = diag(t);
    else { um = 0; for (let k = 0; k < uc.length; k++) um = Math.max(um, Math.abs(uc[k]), Math.abs(ud[k])); for (let k = 0; k < vc.length; k++) um = Math.max(um, Math.abs(vc[k]), Math.abs(vd[k])); }
    if (!Number.isFinite(um)) throw new Error('The two-fluid solution diverged. Refine the grid or lower the CFL number.');
    if (steps % 25 === 0) { ctx?.progress?.(Math.min(0.98, t / tEnd), `Two-fluid model: t = ${t.toExponential(2)} s of ${tEnd.toExponential(2)} s (step ${steps})`); if (ctx?.tick) await ctx.tick(); }
  }
  let div = 0; // residual of the mixture volume balance with the face fractions of the last step
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    const ke = j * nu1 + i + 1, kn = P + nx, q = (k, a, b) => (1 - afx[k]) * a[k] + afx[k] * b[k], r = (k, a, b) => (k < nx || k >= ny * nx ? 0 : (1 - afy[k]) * a[k] + afy[k] * b[k]);
    div = Math.max(div, Math.abs((q(ke, uc, ud) - q(ke - 1, uc, ud)) / dx + (r(kn, vc, vd) - r(P, vc, vd)) / dy));
  }
  const pT = Float64Array.from(p, (q, P) => q + rc * (gx * ((P % nx) + 0.5) * dx + gy * (((P - (P % nx)) / nx + 0.5) * dy - H))); // total pressure, zero reference at the top
  return { nx, ny, dx, dy, L, H, al, p: pT, pExcess: p, uc, vc, ud, vd, hist, t, steps, vol0, vol: hist.vol[hist.vol.length - 1], dIn, dOut, cIn: cInT, cOut: cOutT, clip: clipT, dep: depT, depSolid: depST, solid, div, ut, done: t >= tEnd * (1 - 1e-9) };
}

/** Exact Riemann solution of the 1-D Euler equations for an ideal gas (Toro): state [ρ, u, p] at ξ = x/t. */
export function riemannExact(L, R, gam, xi) {
  const g1 = (gam - 1) / (2 * gam), g2 = (gam + 1) / (2 * gam), g3 = (2 * gam) / (gam - 1), g4 = 2 / (gam - 1), g5 = 2 / (gam + 1), g6 = (gam - 1) / (gam + 1), g7 = (gam - 1) / 2;
  const cL = Math.sqrt((gam * L.p) / L.rho), cR = Math.sqrt((gam * R.p) / R.rho);
  const fk = (p, s, c) => { if (p <= s.p) { const pr = p / s.p; return [g4 * c * (pr ** g1 - 1), (1 / (s.rho * c)) * pr ** -g2]; } const A = g5 / s.rho, B = g6 * s.p, q = Math.sqrt(A / (p + B)); return [(p - s.p) * q, (1 - (0.5 * (p - s.p)) / (B + p)) * q]; };
  let p = Math.max(1e-8 * Math.min(L.p, R.p), ((cL + cR - g7 * (R.u - L.u)) / (cL / L.p ** g1 + cR / R.p ** g1)) ** g3);
  for (let it = 0; it < 60; it++) { const [fl, dl] = fk(p, L, cL), [fr, dr] = fk(p, R, cR), pn = Math.max(1e-10 * p, p - (fl + fr + R.u - L.u) / (dl + dr)); if (Math.abs(pn - p) < 1e-13 * (pn + p)) { p = pn; break; } p = pn; }
  const us = 0.5 * (L.u + R.u) + 0.5 * (fk(p, R, cR)[0] - fk(p, L, cL)[0]);
  if (xi <= us) {
    if (p <= L.p) { const sh = L.u - cL, cm = cL * (p / L.p) ** g1, st = us - cm; if (xi <= sh) return [L.rho, L.u, L.p, p, us]; if (xi > st) return [L.rho * (p / L.p) ** (1 / gam), us, p, p, us]; const c = g5 * (cL + g7 * (L.u - xi)); return [L.rho * (c / cL) ** g4, g5 * (cL + g7 * L.u + xi), L.p * (c / cL) ** g3, p, us]; }
    const pr = p / L.p, sl = L.u - cL * Math.sqrt(g2 * pr + g1);
    return xi <= sl ? [L.rho, L.u, L.p, p, us] : [(L.rho * (pr + g6)) / (pr * g6 + 1), us, p, p, us];
  }
  if (p > R.p) { const pr = p / R.p, sr = R.u + cR * Math.sqrt(g2 * pr + g1); return xi >= sr ? [R.rho, R.u, R.p, p, us] : [(R.rho * (pr + g6)) / (pr * g6 + 1), us, p, p, us]; }
  const sh = R.u + cR, cm = cR * (p / R.p) ** g1, st = us + cm;
  if (xi >= sh) return [R.rho, R.u, R.p, p, us];
  if (xi <= st) return [R.rho * (p / R.p) ** (1 / gam), us, p, p, us];
  const c = g5 * (cR - g7 * (R.u - xi));
  return [R.rho * (c / cR) ** g4, g5 * (-cR + g7 * R.u + xi), R.p * (c / cR) ** g3, p, us];
}

/**
 * One-dimensional / quasi-one-dimensional compressible flow of an ideal gas: finite volumes, MUSCL (minmod)
 * reconstruction, HLLC Riemann fluxes, two-stage SSP Runge–Kutta. Optional area variation A(x), viscous stress
 * and heat conduction (Navier–Stokes terms) and Darcy wall friction. Boundaries: 'open', 'wall',
 * { p0, T0 } (stagnation inlet) or { pb } (back-pressure outlet).
 */
export function euler1D(o) {
  const { n, L, gam = 1.4, Rg = 287, cfl = 0.7 } = o, dx = L / n, cpg = (gam * Rg) / (gam - 1), g1 = gam - 1;
  const xc = Array.from({ length: n }, (_, i) => (i + 0.5) * dx), A = xc.map((x) => (o.area ? o.area(x) : 1)), Af = Array.from({ length: n + 1 }, (_, i) => (o.area ? o.area(i * dx) : 1));
  const W = [0, 1, 2].map(() => new Float64Array(n + 4)), U = [0, 1, 2].map(() => new Float64Array(n)), U0 = [0, 1, 2].map(() => new Float64Array(n)), Fl = [0, 1, 2].map(() => new Float64Array(n + 1)), mu = o.mu || 0, kth = o.kth || 0, fr = o.fric ? o.fric.f / (2 * o.fric.D) : 0;
  for (let i = 0; i < n; i++) { const s = o.init(xc[i]); U[0][i] = s.rho; U[1][i] = s.rho * s.u; U[2][i] = s.p / g1 + 0.5 * s.rho * s.u * s.u; }
  const bcL = o.left || 'open', bcR = o.right || 'open';
  const prim = () => {
    for (let i = 0; i < n; i++) { const r = U[0][i], uu = U[1][i] / r; W[0][i + 2] = r; W[1][i + 2] = uu; W[2][i + 2] = g1 * (U[2][i] - 0.5 * r * uu * uu); }
    for (let side = 0; side < 2; side++) {
      const bc = side ? bcR : bcL, I = side ? n + 1 : 2, r = W[0][I], uu = W[1][I], p = W[2][I];
      let g = [r, uu, p];
      if (bc === 'wall') g = [r, -uu, p];
      else if (bc.p0) { const un = Math.max(side ? -uu : uu, 0), T = Math.max(bc.T0 - (un * un) / (2 * cpg), 0.3 * bc.T0), pg = bc.p0 * (T / bc.T0) ** (gam / g1); g = [pg / (Rg * T), side ? -un : un, pg]; }
      else if (bc.pb !== undefined) { const c = Math.sqrt((gam * p) / r), un = side ? uu : -uu; if (un < c) g = [r * (bc.pb / p) ** (1 / gam), uu, bc.pb]; }
      for (const q of side ? [n + 2, n + 3] : [1, 0]) for (let k = 0; k < 3; k++) W[k][q] = bc === 'wall' && k === 1 ? -W[1][side ? 2 * n + 3 - q : 3 - q] : g[k];
    }
  };
  const mmod = (a, b) => (a * b <= 0 ? 0 : Math.abs(a) < Math.abs(b) ? a : b);
  const rhs = (dU) => {
    prim();
    let smax = 0;
    for (let f = 0; f <= n; f++) { // face f lies between padded cells f+1 and f+2
      const iL = f + 1, iR = f + 2, W0 = W[0], W1 = W[1], W2 = W[2];
      let rl = W0[iL] + 0.5 * mmod(W0[iL] - W0[iL - 1], W0[iR] - W0[iL]), rr = W0[iR] - 0.5 * mmod(W0[iR] - W0[iL], W0[iR + 1] - W0[iR]);
      let ul = W1[iL] + 0.5 * mmod(W1[iL] - W1[iL - 1], W1[iR] - W1[iL]), ur = W1[iR] - 0.5 * mmod(W1[iR] - W1[iL], W1[iR + 1] - W1[iR]);
      let pl = W2[iL] + 0.5 * mmod(W2[iL] - W2[iL - 1], W2[iR] - W2[iL]), prr = W2[iR] - 0.5 * mmod(W2[iR] - W2[iL], W2[iR + 1] - W2[iR]);
      if (!(rl > 0 && pl > 0)) { rl = W[0][iL]; ul = W[1][iL]; pl = W[2][iL]; }
      if (!(rr > 0 && prr > 0)) { rr = W[0][iR]; ur = W[1][iR]; prr = W[2][iR]; }
      const cl = Math.sqrt((gam * pl) / rl), cr = Math.sqrt((gam * prr) / rr), El = pl / g1 + 0.5 * rl * ul * ul, Er = prr / g1 + 0.5 * rr * ur * ur;
      const sl = Math.min(ul - cl, ur - cr), sr = Math.max(ul + cl, ur + cr), sm = (prr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / (rl * (sl - ul) - rr * (sr - ur));
      smax = Math.max(smax, Math.abs(sl), Math.abs(sr));
      let F0, F1, F2;
      if (sl >= 0) { F0 = rl * ul; F1 = rl * ul * ul + pl; F2 = ul * (El + pl); }
      else if (sr <= 0) { F0 = rr * ur; F1 = rr * ur * ur + prr; F2 = ur * (Er + prr); }
      else if (sm >= 0) { const c = (rl * (sl - ul)) / (sl - sm), es = c * (El / rl + (sm - ul) * (sm + pl / (rl * (sl - ul)))); F0 = rl * ul + sl * (c - rl); F1 = rl * ul * ul + pl + sl * (c * sm - rl * ul); F2 = ul * (El + pl) + sl * (es - El); }
      else { const c = (rr * (sr - ur)) / (sr - sm), es = c * (Er / rr + (sm - ur) * (sm + prr / (rr * (sr - ur)))); F0 = rr * ur + sr * (c - rr); F1 = rr * ur * ur + prr + sr * (c * sm - rr * ur); F2 = ur * (Er + prr) + sr * (es - Er); }
      if (mu || kth) { const du = (W[1][iR] - W[1][iL]) / dx, dT = (W[2][iR] / W[0][iR] - W[2][iL] / W[0][iL]) / (Rg * dx), um = 0.5 * (W[1][iL] + W[1][iR]); F1 -= (4 / 3) * mu * du; F2 -= (4 / 3) * mu * um * du + kth * dT; }
      Fl[0][f] = F0 * Af[f]; Fl[1][f] = F1 * Af[f]; Fl[2][f] = F2 * Af[f];
    }
    for (let i = 0; i < n; i++) {
      const p = W[2][i + 2], r = W[0][i + 2], uu = W[1][i + 2];
      dU[0][i] = -(Fl[0][i + 1] - Fl[0][i]) / (dx * A[i]);
      dU[1][i] = (-(Fl[1][i + 1] - Fl[1][i]) + p * (Af[i + 1] - Af[i])) / (dx * A[i]) - fr * r * uu * Math.abs(uu);
      dU[2][i] = -(Fl[2][i + 1] - Fl[2][i]) / (dx * A[i]);
    }
    return smax;
  };
  const d1 = [0, 1, 2].map(() => new Float64Array(n)), d2 = [0, 1, 2].map(() => new Float64Array(n));
  let t = 0, steps = 0, resid = 1, mIn = 0, mOut = 0;
  const maxSteps = o.maxSteps ?? 40000, tEnd = o.tEnd ?? Infinity;
  while (t < tEnd * (1 - 1e-12) && steps < maxSteps) {
    for (let k = 0; k < 3; k++) U0[k].set(U[k]);
    const smax = rhs(d1), dt = Math.min((cfl * dx) / Math.max(smax, 1e-30), tEnd - t), fi = Fl[0][0], fo = Fl[0][n];
    for (let k = 0; k < 3; k++) for (let i = 0; i < n; i++) U[k][i] = U0[k][i] + dt * d1[k][i];
    rhs(d2);
    mIn += 0.5 * dt * (fi + Fl[0][0]); mOut += 0.5 * dt * (fo + Fl[0][n]);
    let dr = 0;
    for (let k = 0; k < 3; k++) for (let i = 0; i < n; i++) { const un = U0[k][i] + 0.5 * dt * (d1[k][i] + d2[k][i]); if (k === 0) dr = Math.max(dr, Math.abs(un - U0[0][i]) / U0[0][i]); U[k][i] = un; }
    if (!(U[0].every((r) => r > 0)) || !Number.isFinite(dr)) throw new Error('The compressible solution became unphysical (negative density) — lower the CFL number or the pressure ratio.');
    t += dt; steps++; resid = dr;
    if (o.steadyTol && steps > 50 && dr < o.steadyTol) break;
    if (o.steadyFlux && steps % 100 === 0 && steps >= 400) { let lo = Infinity, hi = -Infinity; for (let f = 0; f <= n; f++) { const q = Fl[0][f]; if (q < lo) lo = q; if (q > hi) hi = q; } if (hi - lo < o.steadyFlux * Math.max(Math.abs(hi), Math.abs(lo))) break; } // uniform mass flux = steady
  }
  prim();
  const rho = Array.from(W[0].subarray(2, n + 2)), u = Array.from(W[1].subarray(2, n + 2)), p = Array.from(W[2].subarray(2, n + 2)), T = p.map((q, i) => q / (Rg * rho[i])), M = u.map((q, i) => q / Math.sqrt(gam * Rg * T[i]));
  let mass = 0; for (let i = 0; i < n; i++) mass += rho[i] * A[i] * dx;
  return { x: xc, A, rho, u, p, T, M, t, steps, resid, mass, mIn, mOut, mdot: rho.map((r, i) => r * u[i] * A[i]), mdotIn: Fl[0][0], mdotOut: Fl[0][n] };
}

/** Isentropic quasi-1-D nozzle with a normal shock where the back pressure requires it. Returns M(x), p(x)/p0 and the regime. */
export function nozzleExact(areaRatio /* A/A* per station */, iThroat, pbRatio, gam) {
  const g1 = gam - 1, ar = (M) => (1 / M) * ((2 / (gam + 1)) * (1 + 0.5 * g1 * M * M)) ** ((gam + 1) / (2 * g1)), pr = (M) => (1 + 0.5 * g1 * M * M) ** (-gam / g1);
  const inv = (a, sup) => { let lo = sup ? 1 : 1e-6, hi = sup ? 50 : 1; for (let k = 0; k < 80; k++) { const m = 0.5 * (lo + hi); if ((ar(m) > a) === sup) hi = m; else lo = m; } return 0.5 * (lo + hi); };
  const ne = areaRatio.length - 1, ae = areaRatio[ne], peSub = pr(inv(ae, false)), peSup = pr(inv(ae, true));
  const shockAt = (as) => { const M1 = inv(as, true), M2 = Math.sqrt((1 + 0.5 * g1 * M1 * M1) / (gam * M1 * M1 - 0.5 * g1)), p02 = ((((gam + 1) * M1 * M1) / (g1 * M1 * M1 + 2)) ** (gam / g1)) * ((gam + 1) / (2 * gam * M1 * M1 - g1)) ** (1 / g1), a2 = ar(M2) / as; return { M1, M2, p02, a2 }; };
  const peShockExit = (() => { const s = shockAt(ae); return pr(s.M2) * s.p02; })();
  let regime, As = null, sh = null, Mth = 1;
  if (pbRatio >= peSub) { regime = 'subsonic'; const Me = Math.sqrt((2 / g1) * (pbRatio ** (-g1 / gam) - 1)), aStar = ae / ar(Math.max(Me, 1e-6)); Mth = inv(Math.max(1, 1 / aStar), false); return { regime, M: areaRatio.map((a) => inv(Math.max(1, a / aStar), false)), p: areaRatio.map((a) => pr(inv(Math.max(1, a / aStar), false))), choked: false, mFactor: ar(1) / ar(Math.max(Mth, 1e-6)) * 0 + 1 / aStar, peSub, peSup }; }
  if (pbRatio > peShockExit) { // normal shock inside the diverging part: bisection on the shock area ratio
    let lo = 1, hi = ae;
    for (let k = 0; k < 70; k++) { const m = 0.5 * (lo + hi), s = shockAt(m), pe = pr(inv(ae * s.a2, false)) * s.p02; if (pe > pbRatio) lo = m; else hi = m; }
    As = 0.5 * (lo + hi); sh = shockAt(As); regime = 'shock';
  } else regime = pbRatio > peSup ? 'overexpanded' : 'supersonic';
  const M = [], p = [];
  areaRatio.forEach((a, i) => { if (i <= iThroat) { const m = inv(a, false); M.push(m); p.push(pr(m)); } else if (As !== null && a > As) { const m = inv(a * sh.a2, false); M.push(m); p.push(pr(m) * sh.p02); } else { const m = inv(a, true); M.push(m); p.push(pr(m)); } });
  return { regime, M, p, choked: true, mFactor: 1, As, peSub, peSup, peShockExit };
}

/**
 * Maxwell–Stefan film model for two solutes (1, 2) in a solvent (3) across a stagnant film of thickness delta:
 * −c_t dx_i/dz = Σ_j (x_j N_i − x_i N_j)/Đ_ij with the total molar flux N_t = Jv·c_t toward the wall and solute
 * fluxes N_i = (1 − R_i) N_t x_i,wall. Integrated with RK4 from the bulk (z = 0) to the wall (z = delta).
 */
export function msFilm({ xb, D13, D23, D12, ct, Jv, delta, rej = [1, 1], steps = 80 }) {
  const Nt = Jv * ct, rhs = (x, N) => { const x3 = 1 - x[0] - x[1], N3 = Nt - N[0] - N[1]; return [-((x[1] * N[0] - x[0] * N[1]) / D12 + (x3 * N[0] - x[0] * N3) / D13) / ct, -((x[0] * N[1] - x[1] * N[0]) / D12 + (x3 * N[1] - x[1] * N3) / D23) / ct]; };
  const march = (N) => { const h = delta / steps, prof = [[0, xb[0], xb[1]]]; let x = [xb[0], xb[1]]; for (let s = 0; s < steps; s++) { const k1 = rhs(x, N), k2 = rhs([x[0] + 0.5 * h * k1[0], x[1] + 0.5 * h * k1[1]], N), k3 = rhs([x[0] + 0.5 * h * k2[0], x[1] + 0.5 * h * k2[1]], N), k4 = rhs([x[0] + h * k3[0], x[1] + h * k3[1]], N); x = [x[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), x[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]; prof.push([(s + 1) * h, x[0], x[1]]); } return { xw: x, prof }; };
  let N = [(1 - rej[0]) * Nt * xb[0], (1 - rej[1]) * Nt * xb[1]], r = march(N), it = 0;
  for (; it < 60; it++) { const Nn = [(1 - rej[0]) * Nt * r.xw[0], (1 - rej[1]) * Nt * r.xw[1]], d = Math.abs(Nn[0] - N[0]) + Math.abs(Nn[1] - N[1]); N = [0.5 * (N[0] + Nn[0]), 0.5 * (N[1] + Nn[1])]; r = march(N); if (d <= 1e-13 * Math.abs(Nt)) break; }
  return { xw: r.xw, prof: r.prof, N, Nt, iters: it };
}

/** Power-law regression closure y = a Π x_k^b_k by ridge least squares in log space, with leave-one-out cross-validation. */
export function fitClosure(X, y, lambda = 1e-9) {
  const m = X.length, nf = X[0].length + 1, rows = X.map((r) => [1, ...r.map(Math.log)]), ly = y.map(Math.log);
  const fit = (skip) => { const A = Array.from({ length: nf }, () => new Array(nf).fill(0)), g = new Array(nf).fill(0); for (let i = 0; i < m; i++) { if (i === skip) continue; for (let a = 0; a < nf; a++) { g[a] += rows[i][a] * ly[i]; for (let b = 0; b < nf; b++) A[a][b] += rows[i][a] * rows[i][b]; } } for (let a = 1; a < nf; a++) A[a][a] += lambda * m; A[0][0] += 1e-14; return solveLinear(A, g); };
  const ev = (c, r) => Math.exp(r.reduce((s, x, k) => s + c[k] * x, 0)), coef = fit(-1), pred = rows.map((r) => ev(coef, r)), loo = rows.map((r, i) => ev(fit(i), r));
  const mly = ly.reduce((s, q) => s + q, 0) / m, sst = ly.reduce((s, q) => s + (q - mly) ** 2, 0) || 1e-300, r2 = (p) => 1 - p.reduce((s, q, i) => s + (Math.log(q) - ly[i]) ** 2, 0) / sst;
  return { coef, a: Math.exp(coef[0]), exps: coef.slice(1), pred, loo, r2: r2(pred), r2loo: r2(loo), maxErr: Math.max(...pred.map((q, i) => Math.abs(q / y[i] - 1))), maxErrLoo: Math.max(...loo.map((q, i) => Math.abs(q / y[i] - 1))), predict: (x) => ev(coef, [1, ...x.map(Math.log)]) };
}

// ---------------------------------------------------------------------------------------------------
// Study helpers: lattice-Boltzmann start field, phase transport, regression closure, compressible studies
// ---------------------------------------------------------------------------------------------------
/** Lattice-Boltzmann solution of the case on a uniform lattice, mapped to the finite-volume faces as the starting field. */
function lbmStart(c, v) {
  const { L, H, fl, geo, nx, ny, g } = c, U = c.Uin, ReH = (fl.rho * U * H) / fl.mu;
  let nyL = clamp(Math.round(v.lbmNy ?? 20), 8, 96), nxL = Math.max(8, Math.round((nyL * L) / H));
  if (nxL * nyL > 60000) { nyL = Math.max(8, Math.floor(Math.sqrt((60000 * H) / L))); nxL = Math.max(8, Math.round((nyL * L) / H)); }
  if (nxL * nyL > 90000) return { note: 'The domain is too long for the lattice-Boltzmann lattice (more than 90 000 nodes at 8 nodes across the gap); the finite-volume solver was used alone.' };
  let uL = 0.1, nuL = (uL * nyL) / Math.max(ReH, 1e-9);
  if (nuL > 1 / 3) { nuL = 1 / 3; uL = (nuL * ReH) / nyL; }
  if (nuL < 0.01) { uL = 0.15; nuL = (uL * nyL) / ReH; }
  if (nuL < 0.004) return { note: `Reynolds number ${fmt(2 * ReH, 3)} is too high for a stable BGK lattice with ${nyL} nodes across the gap (relaxation time below 0.512). Raise the lattice resolution; the finite-volume solver was used alone.` };
  let mk;
  try { mk = buildMask(geo, nxL, nyL, Float64Array.from({ length: nyL }, (_, j) => ((j + 0.5) * H) / nyL)); } catch (e) { return { note: 'The geometry closes the passage on the lattice-Boltzmann lattice — raise the lattice resolution. The finite-volume solver was used alone.' }; }
  const sol = mk.solid, uin = new Float64Array(nyL), tau = 3 * nuL + 0.5;
  let ja = nyL, jb = -1, qs = 0, nOpen = 0;
  for (let j = 0; j < nyL; j++) if (!sol[j * nxL]) { ja = Math.min(ja, j); jb = Math.max(jb, j); nOpen++; }
  for (let j = ja; j <= jb; j++) if (!sol[j * nxL]) { const s = (j + 0.5 - ja) / (jb + 1 - ja); uin[j] = v.inlet === 'uniform' ? 1 : 6 * s * (1 - s); qs += uin[j]; }
  for (let j = 0; j < nyL; j++) uin[j] *= (uL * nOpen) / qs;
  const lb = lbmD2Q9({ nx: nxL, ny: nyL, solid: sol, uin, tau, maxSteps: Math.min(40000, Math.ceil(7e7 / (nxL * nyL))), tol: 2e-5, check: 200 });
  const vs = U / uL, ps = (fl.rho * vs * vs) / 3, N = nxL * nyL, pL = new Float64Array(N);
  let rOut = 0, mOut = 0;
  for (let j = 0; j < nyL; j++) if (!sol[j * nxL + nxL - 1]) { rOut += lb.rho[j * nxL + nxL - 1]; mOut++; }
  rOut = mOut ? rOut / mOut : 1;
  for (let P = 0; P < N; P++) pL[P] = ps * (lb.rho[P] - rOut);
  for (let j = 0, P = 0; j < nyL; j++) for (let i = 0; i < nxL; i++, P++) { // pressure inside solids: mean of the fluid neighbours (for interpolation only)
    if (!sol[P]) continue;
    let a = 0, m = 0;
    for (const Q of [i > 0 ? P - 1 : -1, i < nxL - 1 ? P + 1 : -1, j > 0 ? P - nxL : -1, j < nyL - 1 ? P + nxL : -1]) if (Q >= 0 && !sol[Q]) { a += ps * (lb.rho[Q] - rOut); m++; }
    pL[P] = m ? a / m : 0;
  }
  const smp = (a, x, y) => { const fi = clamp((x / L) * nxL - 0.5, 0, nxL - 1), fj = clamp((y / H) * nyL - 0.5, 0, nyL - 1), i = Math.min(Math.floor(fi), nxL - 2), j = Math.min(Math.floor(fj), nyL - 2), tx = fi - i, ty = fj - j; return (a[j * nxL + i] * (1 - tx) + a[j * nxL + i + 1] * tx) * (1 - ty) + (a[(j + 1) * nxL + i] * (1 - tx) + a[(j + 1) * nxL + i + 1] * tx) * ty; };
  const nu1 = nx + 1, dx = L / nx, init = { u: new Float64Array(nu1 * ny), v: new Float64Array(nx * (ny + 1)), p: new Float64Array(nx * ny) };
  for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) init.u[j * nu1 + i] = vs * smp(lb.ux, i * dx, g.yc[j]);
  for (let jf = 1; jf < ny; jf++) for (let i = 0; i < nx; i++) init.v[jf * nx + i] = vs * smp(lb.uy, (i + 0.5) * dx, g.yf[jf]);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) init.p[j * nx + i] = smp(pL, (i + 0.5) * dx, g.yc[j]);
  // flow-averaged total pressure per lattice column and the largest speed
  const pt = new Float64Array(nxL);
  let umax = 0;
  for (let i = 0; i < nxL; i++) { let a = 0, m = 0; for (let j = 0; j < nyL; j++) { const P = j * nxL + i; if (sol[P]) continue; const ux = vs * lb.ux[P], uy = vs * lb.uy[P], sp = Math.hypot(ux, uy); if (sp > umax) umax = sp; a += ux * (pL[P] + 0.5 * fl.rho * sp * sp); m += ux; } pt[i] = Math.abs(m) > 1e-300 ? a / m : 0; }
  const ptAt = (x) => { const fi = clamp((x / L) * nxL - 0.5, 0, nxL - 1), i = Math.min(Math.floor(fi), nxL - 2); return pt[i] + (pt[i + 1] - pt[i]) * (fi - i); };
  return { init, lb, nxL, nyL, tau, uL, vs, sol, smp, ptAt, umax, steps: lb.steps, converged: lb.converged };
}

/** Initial phase distribution and transport of the second phase on the solved velocity field. */
function runPhase(r, v, fl) {
  const { nx, ny, dx, dy, dyc, yf, yc, solid, L, H } = r, n = nx * ny, method = v.mp, a0 = new Float64Array(n), phi0 = new Float64Array(n);
  const R = 0.5 * (v.mpD / 100) * H, xb = (v.mpX / 100) * L, yb = clamp((v.mpY / 100) * H, 0, H), x1 = xb + (v.mpLen / 100) * L;
  const sd = v.mpInit === 'slug' ? (x) => Math.min(x - xb, x1 - x) : (x, y) => R - Math.hypot(x - xb, y - yb);
  if (method !== 'ee') for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    if (solid[P]) { phi0[P] = -Math.max(dx, R); continue; }
    let sIn = 0;
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (sd((i + (a + 0.5) / 4) * dx, yf[j] + ((b + 0.5) / 4) * dy[j]) > 0) sIn++;
    a0[P] = sIn / 16; phi0[P] = sd((i + 0.5) * dx, yc[j]);
  }
  else for (let P = 0; P < n; P++) a0[P] = solid[P] ? 0 : v.mpAlphaIn / 100; // dispersed phase: the domain starts at the inlet fraction
  const d = v.mpDp * 1e-6, vsl = method === 'ee' ? (-(v.mpRho - fl.rho) * 9.80665 * d * d) / (18 * fl.mu) : 0, tEnd = (v.mpTime * L) / Math.max(r.Uref, 1e-12);
  const res = advectPhase({ nx, ny, dx, dy, dyc, u: r.u, v: r.v, solid }, { method, a0, phi0, tEnd, aIn: method === 'ee' ? v.mpAlphaIn / 100 : 0, vs: vsl, Dd: Math.max(0, v.mpDisp), cfl: 0.4, maxSteps: 12000 });
  return { ...res, a0, tEnd, tSim: res.steps * res.dt, vsl, R, xb, yb, x1, method };
}

/**
 * Martin & Moyce (1952), Phil. Trans. R. Soc. Lond. A 244, 312–324 (Part IV), Table 2: collapse of a rectangular liquid column,
 * n² = 2 (height = 2 × base a), a = 2¼ in, column of means — surge-front position Z = x/a at T = n t √(g/a) = t √(2g/a).
 * Read from a scan of the paper (page 317). The paper did not measure the instant of release for this table: the runs are aligned
 * at Z = 1.44 (T = 1.19 in every run), so T carries an unknown offset and only differences of T, i.e. front speeds, are comparable.
 * The table continues to Z = 14; the values kept here cover a tank four column widths long.
 */
export const MARTIN_MOYCE = { T: [0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98, 2.2, 2.32, 2.51, 2.65, 2.83, 2.97, 3.11, 3.33], Z: [1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.0, 3.22, 3.44, 3.67, 3.89, 4.11] };
/**
 * Hysing et al., rising-bubble benchmark, test case 1 (ρ 1000/100, μ 10/1, g 0.98, σ 24.5; Re = 35, Eo = 10): finest-grid results of
 * the three groups (TU Dortmund TP2D, EPFL FreeLIFE, Magdeburg MooNMD) from Table 12 of the benchmark proposal (TU Dortmund
 * Ergebnisberichte Angewandte Mathematik Nr. 351, 2007) and the reference data files published with it by TU Dortmund (featflow.de).
 */
export const HYSING1 = { yc3: [1.0813, 1.0799, 1.0817], vMax: [0.2417, 0.2421, 0.2417], tVmax: [0.9213, 0.9313, 0.9239], cMin: [0.9013, 0.9011, 0.9013], tCmin: [1.9041, 1.875, 1.9] };
/**
 * Equilibrium of homogeneous turbulent shear flow as tabulated by Speziale, Sarkar & Gatski (ICASE Report 90-5 / NASA CR-181979,
 * 1990), Table 1, with b_ij = (u_i′u_j′ − ⅔ k δ_ij)/2k: the experiments of Tavoularis & Corrsin (1981) and the Launder–Reece–Rodi model.
 */
export const SHEAR_EQ = { exp: { b11: 0.201, b22: -0.147, b12: -0.15, SKe: 6.08 }, lrr: { b11: 0.193, b22: -0.096, b12: -0.185, SKe: 5.65 } };

/** Linear interpolation in a recorded history. */
const histAt = (ts, ys, t) => { let k = 1; while (k < ts.length - 1 && ts[k] < t) k++; if (ts.length < 2) return ys[0] ?? 0; const w = clamp((t - ts[k - 1]) / Math.max(ts[k] - ts[k - 1], 1e-300), 0, 1); return ys[k - 1] + w * (ys[k] - ys[k - 1]); };
/** Least-squares slope of ys(ts) between t0 and t1. */
const histSlope = (ts, ys, t0, t1) => { let st = 0, sy = 0, stt = 0, sty = 0, m = 0; for (let k = 0; k < ts.length; k++) if (ts[k] >= t0 && ts[k] <= t1) { st += ts[k]; sy += ys[k]; stt += ts[k] * ts[k]; sty += ts[k] * ys[k]; m++; } const d = m * stt - st * st; return m > 1 && d > 0 ? (m * sty - st * sy) / d : 0; };
/** Period of an oscillating record from its zero crossings about `mid` (mean spacing of crossings × 2); 0 when fewer than two crossings. */
const histPeriod = (ts, ys, mid = 0) => { const cr = []; for (let k = 1; k < ts.length; k++) { const a = ys[k - 1] - mid, b = ys[k] - mid; if (a * b < 0) cr.push(ts[k - 1] + ((ts[k] - ts[k - 1]) * a) / (a - b)); } return cr.length >= 2 ? (2 * (cr[cr.length - 1] - cr[0])) / (cr.length - 1) : 0; };
/** Angular frequency of the interfacial gravity–capillary wave of wavenumber k between two layers (heavy below, depth h1; light above, depth h2) in a closed tank. */
export const sloshOmega = (k, h1, h2, rho1, rho2, g, sigma = 0) => Math.sqrt(Math.max(0, ((rho1 - rho2) * g * k + sigma * k ** 3) / (rho1 / Math.tanh(k * h1) + rho2 / Math.tanh(k * h2))));

const TP_CASES = { dam: 'collapse of a liquid column (dam break)', bubble: 'rising bubble or drop', slosh: 'sloshing of a free surface', rt: 'Rayleigh–Taylor overturning (heavy over light)' };
/** Set-up of the coupled two-phase study from the inputs: geometry, fluids, the signed distance of the initial interface and the reference scales. */
function twoPhaseCase(v) {
  const Wt = clamp(v.tpW, 1, 1e5) * 1e-3, Ht = clamp(v.tpH, 1, 1e5) * 1e-3, nx = clamp(Math.round(v.tpNx), 16, 200), ny = clamp(Math.round(v.tpNy), 16, 200), g = clamp(v.tpG, 0.01, 100), kind = TP_CASES[v.tpCase] ? v.tpCase : 'dam';
  const r1 = Math.max(v.tpRho1, 1e-3), r2 = Math.max(v.tpRho2, 1e-3), m1 = Math.max(v.tpMu1, 1e-6) * 1e-3, m2 = Math.max(v.tpMu2, 1e-6) * 1e-3, sigma = Math.max(0, v.tpSigma) * 1e-3;
  let sd, len, light = false, outline, a = 0, hc = 0, D = 0, yb = 0, h0 = 0, amp = 0;
  if (kind === 'dam') { a = (clamp(v.tpColW, 5, 90) / 100) * Wt; hc = (clamp(v.tpColH, 5, 100) / 100) * Ht; sd = (x, y) => Math.min(a - x, hc - y); len = a; outline = { x: [0, a, a, 0], y: [hc, hc, 0, 0] }; }
  else if (kind === 'bubble') { D = (clamp(v.tpBubD, 5, 90) / 100) * Wt; yb = (clamp(v.tpBubY, 5, 95) / 100) * Ht; sd = (x, y) => 0.5 * D - Math.hypot(x - 0.5 * Wt, y - yb); len = D; light = true; const th = linspace(0, 2 * Math.PI, 49); outline = { x: th.map((t) => 0.5 * Wt + 0.5 * D * Math.cos(t)), y: th.map((t) => yb + 0.5 * D * Math.sin(t)) }; }
  else { h0 = (clamp(v.tpFill, 5, 95) / 100) * Ht; amp = (clamp(v.tpAmp, 0, 40) / 100) * Ht; amp = Math.min(amp, 0.9 * Math.min(h0, Ht - h0)); const kx = ((kind === 'rt' ? 2 : 1) * Math.PI) / Wt, eta = (x) => h0 + amp * Math.cos(kx * x); sd = kind === 'rt' ? (x, y) => y - eta(x) : (x, y) => eta(x) - y; len = Wt; const xs = linspace(0, Wt, 41); outline = { x: xs, y: xs.map(eta) }; }
  // phase A of the solver is the one inside the initial region: the liquid, except for the bubble (light phase inside)
  const rhoA = light ? r2 : r1, rhoB = light ? r1 : r2, muA = light ? m2 : m1, muB = light ? m1 : m2, tc = Math.sqrt(len / g), tEnd = clamp(v.tpTime, 0.01, 200) * tc;
  return { Wt, Ht, nx, ny, g, kind, r1, r2, m1, m2, sigma, sd, len, light, outline, a, hc, D, yb, h0, amp, rhoA, rhoB, muA, muB, tc, tEnd, method: v.tpMethod === 'ls' ? 'ls' : 'vof', slip: v.tpSlip !== false };
}

/** Channel geometry of the multiphase studies: the spacer-filled channel or the imported section of the Geometry inputs, rasterised on a uniform grid of ny cells across the gap (square cells along the flow, at most nxMax). */
function mfChannel(v, dom, nyIn, nxMax = 240) {
  const H = clamp(v.H, 0.05, 2000) * 1e-3, lm = v.lm * 1e-3, nFil = clamp(Math.round(v.nFil), 1, 40), imp = dom === 'import', L = imp ? clamp(v.L, 0.2, 50000) * 1e-3 : nFil * lm;
  const ny = clamp(Math.round(nyIn), 6, 64), nx = clamp(Math.round((ny * L) / H), 16, nxMax), yc = Float64Array.from({ length: ny }, (_, j) => ((j + 0.5) * H) / ny);
  const geo = { type: imp ? 'import' : 'spacer', arr: v.arr, L, H, df: v.df * 1e-3, lm, nFil, cad: v.cad, axis: +v.cadAxis, fit: v.cadFit, scale: v.cadScale, size: v.cadSize / 100, cx: v.cadX / 100, cy: v.cadY / 100, invert: !!v.cadInvert };
  const mk = buildMask(geo, nx, ny, yc);
  let open = 0; for (let j = 0; j < ny; j++) if (!mk.solid[j * nx]) open++;
  return { H, L, nx, ny, dx: L / nx, dy: H / ny, mk, geo, open, name: imp ? 'imported section' : v.arr === 'none' ? 'empty channel' : `spacer-filled channel (${v.arr} filaments)` };
}

/** Free-interface study in a channel geometry: an air slug or a train of bubbles carried through the spacer-filled channel or an imported section by the liquid cross-flow (two-way coupled). */
async function runTwoPhaseChannel(v, ctx) {
  const dom = v.tpDomain === 'import' ? 'import' : 'spacer', c = mfChannel(v, dom, v.tpChNy ?? 12), { H, L, nx, ny, dx, dy, mk } = c, n = nx * ny, nu1 = nx + 1, W = [], warn = (msg) => W.push({ level: 'warn', msg });
  const r1 = Math.max(v.tpRho1, 1e-3), r2 = Math.max(v.tpRho2, 1e-3), m1 = Math.max(v.tpMu1, 1e-6) * 1e-3, m2 = Math.max(v.tpMu2, 1e-6) * 1e-3, sigma = Math.max(0, v.tpSigma) * 1e-3, g = clamp(v.tpG, 0, 100), U = clamp(v.tpUin ?? 0.3, 1e-4, 20), method = v.tpMethod === 'ls' ? 'ls' : 'vof';
  const x0 = (clamp(v.tpChX ?? 8, 0, 90) / 100) * L, kind = v.tpChInit === 'bubbles' ? 'bubbles' : 'slug', nB = clamp(Math.round(v.tpChN ?? 3), 1, 12), D = (clamp(v.tpChD ?? 60, 10, 95) / 100) * H, len = (clamp(v.tpChLen ?? 20, 2, 80) / 100) * L;
  const cen = Array.from({ length: nB }, (_, k) => [x0 + 0.5 * D + k * 1.6 * D, H * (0.5 + (k % 2 ? 0.12 : -0.12))]).filter((q) => q[0] + 0.5 * D < 0.98 * L);
  const sd = kind === 'slug' ? (x) => Math.min(x - x0, x0 + len - x) : (x, y) => Math.max(...cen.map((q) => 0.5 * D - Math.hypot(x - q[0], y - q[1])));
  // inlet: parabolic over the open height of the first column (liquid only), mean velocity U over that height
  const uin = new Float64Array(ny); { let ja = ny, jb = -1; for (let j = 0; j < ny; j++) if (!mk.solid[j * nx]) { ja = Math.min(ja, j); jb = Math.max(jb, j); } let q = 0; for (let j = ja; j <= jb; j++) if (!mk.solid[j * nx]) { const sq = (j + 0.5 - ja) / (jb + 1 - ja); uin[j] = 6 * sq * (1 - sq); q += uin[j]; } for (let j = 0; j < ny; j++) uin[j] *= (U * c.open) / q; }
  const tFlow = L / U, tEnd = clamp(v.tpChTime ?? 0.8, 0.02, 20) * tFlow;
  ctx?.progress?.(0.02, 'Two-phase channel flow: start');
  const r = await twoPhase2D({ method, nx, ny, W: L, Hh: H, rhoA: r2, rhoB: r1, muA: m2, muB: m1, sigma, gy: -g, sd, tEnd, slipSide: true, slipTB: false, solid: mk.solid, flow: { uin }, rhoRef: r1, snaps: [tEnd / 3, (2 * tEnd) / 3], maxSteps: 20000, cfl: clamp(v.tpCfl ?? 0.25, 0.05, 0.5) }, ctx);
  ctx?.progress?.(0.985, 'Post-processing');
  const h = r.hist, last = h.t.length - 1, mm = (a) => Array.from(a, (x) => x * 1e3), xs = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx * 1e3), ys = Array.from({ length: ny }, (_, j) => (j + 0.5) * dy * 1e3);
  const rows = (f, hole = NaN) => Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => (mk.solid[j * nx + i] ? hole : f(j * nx + i, i, j)))), mask = Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => !!mk.solid[j * nx + i]));
  const Uc = rows((P, i, j) => 0.5 * (r.u[j * nu1 + i] + r.u[j * nu1 + i + 1]), 0), Vc = rows((P) => 0.5 * (r.v[P] + r.v[P + nx]), 0);
  const shapes = mk.shapes.map((q) => ({ x: mm(q.x), y: mm(q.y), closed: q.closed, color: '#0f172a' })), base = { type: 'field', xlabel: 'x (mm)', ylabel: 'y (mm)', x: xs, y: ys, mask, equal: L / H <= 8, shapes }, fld = { ...base, zlabel: 'Gas (second-phase) volume fraction', zunit: '–', zmin: 0, zmax: 1, cmap: 'viridis' };
  const name = method === 'ls' ? 'level set (third-order upwind, redistancing, volume correction)' : 'volume of fluid (THINC/WLIC)', bal = (r.vol0 + r.volIn - r.volOut - r.vol) / Math.max(r.vol0, 1e-300), Aflu = (n - mk.solid.reduce((a, b) => a + b, 0)) * dx * dy;
  // slug transport: velocity of the centroid while the whole phase is inside, arrival of the leading edge at the outlet, pressure drop with and without the gas
  const kIn = h.out.findIndex((x) => x > 1e-3 * r.vol0), kA = kIn < 0 ? last : Math.max(1, kIn - 1), uC = kA > 2 ? histSlope(h.t.slice(0, kA + 1), h.xc.slice(0, kA + 1), 0.15 * h.t[kA], h.t[kA]) : 0, tArr = kIn < 0 ? null : h.t[kIn];
  const dpMax = Math.max(...h.dp.slice(Math.min(5, last))), dpEnd = h.dp[last], gone = r.vol < 0.02 * r.vol0, tauPk = Math.max(...h.tauMax), tauMn = Math.max(...h.tauMean), tauEnd = h.tauMean[last];
  const Re = (r1 * U * 2 * H) / m1, We = sigma > 0 ? (r1 * U * U * H) / sigma : Infinity, Ca = sigma > 0 ? (m1 * U) / sigma : Infinity, Eo = sigma > 0 ? (Math.abs(r1 - r2) * g * H * H) / sigma : Infinity;
  const plots = [{ ...fld, title: `Gas fraction and velocity at t = ${fmt(r.t * 1e3, 3)} ms (${method === 'ls' ? 'level set' : 'volume of fluid'})`, z: rows((P) => clamp(r.a[P], 0, 1)), u: Uc, v: Vc, vectors: true, note: `Dark outlines: ${c.name === 'imported section' ? 'imported solids' : 'spacer filaments'} (blocked cells of the ${nx} × ${ny} grid). The gas moves with the computed flow and acts back on it through the density, the viscosity and the surface tension.` },
    ...(r.snaps || []).map((q) => ({ ...fld, title: `Gas fraction at t = ${fmt(q.t * 1e3, 3)} ms`, z: rows((P) => clamp(q.a[P], 0, 1)) })),
    { ...base, title: 'Pressure in excess of the liquid hydrostatic pressure (zero at the outlet)', zlabel: 'Pressure', zunit: 'Pa', z: rows((P) => r.p[P]), cmap: 'coolwarm', contours: 10 },
    { ...base, title: 'Velocity magnitude and streamlines', zlabel: 'Speed', zunit: 'm/s', z: rows((P, i, j) => Math.hypot(Uc[j][i], Vc[j][i])), cmap: 'turbo', u: Uc, v: Vc, stream: true },
    { type: 'line', title: 'Gas volume: hold-up, inflow and outflow', xlabel: 'Time (ms)', ylabel: 'Volume (mm³ per mm depth)', series: [{ name: 'In the channel', x: mm(h.t), y: h.vol.map((x) => x * 1e6) }, { name: 'Left through the outlet (cumulative)', x: mm(h.t), y: h.out.map((x) => x * 1e6) }, { name: 'Entered through the inlet (cumulative)', x: mm(h.t), y: h.inn.map((x) => x * 1e6), dash: true }, { name: 'Hold-up + outflow − inflow', x: mm(h.t), y: h.vol.map((x, k) => (x + h.out[k] - h.inn[k]) * 1e6), dash: true }], note: 'The last curve stays at the initial gas volume: the phase balance closes at every step.' },
    { type: 'line', title: 'Pressure drop and wall shear while the gas passes', xlabel: 'Time (ms)', ylabel: 'Pressure drop (Pa) · wall shear (Pa × 100)', series: [{ name: 'Pressure drop, inlet − outlet column (Pa)', x: mm(h.t.slice(3)), y: h.dp.slice(3) }, { name: 'Mean wall shear × 100 (Pa)', x: mm(h.t.slice(3)), y: h.tauMean.slice(3).map((x) => 100 * x) }, { name: 'Largest wall shear × 100 (Pa)', x: mm(h.t.slice(3)), y: h.tauMax.slice(3).map((x) => 100 * x), dash: true }], note: 'Two-way coupling: the interface raises the pressure drop (capillary pressure at the filament gaps and the liquid displaced around the gas) and the shear on the membrane walls, which is why air sparging is used against fouling.' },
    { type: 'line', title: 'Position of the gas along the channel', xlabel: 'Time (ms)', ylabel: 'x (mm)', series: [{ name: 'Leading edge', x: mm(h.t), y: mm(h.xLead) }, { name: 'Centroid', x: mm(h.t), y: mm(h.xc) }, { name: 'Trailing edge', x: mm(h.t), y: mm(h.xTrail) }, { name: 'Mean liquid velocity × time (from the initial centroid)', x: mm(h.t), y: h.t.map((t) => Math.min(L, h.xc[0] + U * t) * 1e3), dash: true, color: '#94a3b8' }] }];
  const kpis = [{ label: 'Gas velocity (centroid) ÷ mean liquid velocity', value: uC / U, unit: '', help: 'While all the gas is inside the channel; a slug moves faster than the mean flow when it rides the fast core, slower when it is held at the filaments' }, { label: 'Gas velocity (centroid)', value: uC, unit: 'm/s' },
    { label: 'Leading edge reaches the outlet at', value: tArr === null ? '–' : tArr * 1e3, unit: tArr === null ? '' : 'ms', status: tArr === null ? 'warn' : 'ok', help: tArr === null ? 'The gas has not reached the outlet in the simulated time' : `Plug-flow estimate ${fmt(((L - (kind === 'slug' ? x0 + len : Math.max(...cen.map((q) => q[0])) + 0.5 * D)) / U) * 1e3, 3)} ms` },
    { label: 'Gas still in the channel at the end', value: (100 * r.vol) / Math.max(r.vol0, 1e-300), unit: '% of initial', help: 'Gas held in the wakes of the filaments or not yet arrived' },
    { label: 'Largest pressure drop during the passage', value: dpMax, unit: 'Pa' }, { label: gone ? 'Pressure drop, liquid alone (end of run)' : 'Pressure drop at the end of the run', value: dpEnd, unit: 'Pa' },
    { label: 'Largest wall shear (space and time)', value: tauPk, unit: 'Pa' }, { label: 'Mean wall shear: peak ÷ end of run', value: tauEnd > 0 ? tauMn / tauEnd : 0, unit: '', help: 'Shear enhancement by the passing gas when the gas has left by the end of the run' },
    { label: 'Gas volume balance', value: 100 * bal, unit: '% of initial', status: Math.abs(bal) > 1e-3 ? 'warn' : 'ok', help: '(initial + inflow − outflow − hold-up) ÷ initial' }, { label: 'Largest velocity', value: Math.max(...h.umax), unit: 'm/s' }, { label: 'Time steps', value: r.steps, unit: '' }, { label: 'Largest velocity divergence', value: r.div, unit: '1/s' }];
  const outputs = { phaseVolumeError: bal, maxVelocity: Math.max(...h.umax), slugVelocity: uC, pressureDropMax: dpMax, wallShearMax: tauPk, gasHoldupEnd: r.vol / Math.max(r.vol0, 1e-300), timeSteps: r.steps };
  if (c.mk.note) W.push({ level: 'info', msg: c.mk.note });
  if (!r.done) warn(`The run stopped after ${r.steps} steps at t = ${fmt(r.t * 1e3, 3)} ms of ${fmt(tEnd * 1e3, 3)} ms: the time step is limited to ${fmt(r.dtFix * 1e6, 3)} µs by surface tension or viscosity on this grid — shorten the simulated time, raise the velocity or coarsen the grid.`);
  if (Math.abs(bal) > 1e-3) warn(`The gas volume balance is off by ${fmt(100 * bal, 3)} % of the initial volume${method === 'ls' ? ' (the level-set volume correction cannot follow the gas while it leaves through the outlet)' : ''}.`);
  const cellsD = (kind === 'slug' ? Math.min(len, H) : D) / Math.min(dx, dy);
  if (cellsD < 8) warn(`The ${kind === 'slug' ? 'slug' : 'bubbles'} span${kind === 'slug' ? 's' : ''} ${fmt(cellsD, 3)} cells: curvature and surface tension need about 12 cells — raise the cells across the gap.`);
  if (dom === 'spacer' && v.arr !== 'none' && (v.df * 1e-3) / dy < 4) warn(`A filament is ${fmt((v.df * 1e-3) / dy, 2)} cells across; use at least 5–6 cells per filament diameter.`);
  W.unshift({ level: 'info', msg: `Coupled two-phase flow through the ${c.name}: one-fluid Navier–Stokes equations with ${name} on ${nx} × ${ny} cells (${fmt(100 * mk.solidFraction, 3)} % blocked), velocity inlet with a parabolic liquid profile, pressure outlet, no-slip walls and solids, 90° contact angle; ${r.steps} steps. Re = ${fmt(Re, 3)}, We = ${Number.isFinite(We) ? fmt(We, 3) : '∞'}, Ca = ${Number.isFinite(Ca) ? fmt(Ca, 3) : '∞'}.` });
  const stp = Math.max(1, Math.ceil(h.t.length / 40)), hr = [];
  for (let k = 0; k < h.t.length; k += stp) hr.push([h.t[k] * 1e3, h.vol[k] * 1e6, h.inn[k] * 1e6, h.out[k] * 1e6, h.xc[k] * 1e3, h.xLead[k] * 1e3, h.dp[k], h.tauMean[k], h.tauMax[k], h.umax[k]]);
  const tables = [{ title: 'Two-phase channel study: set-up and numerics', columns: ['Item', 'Value'], rows: [['Domain', c.name], ['Initial gas', kind === 'slug' ? `slug filling the gap from ${fmt(x0 * 1e3, 3)} to ${fmt((x0 + len) * 1e3, 3)} mm` : `${cen.length} bubble${cen.length > 1 ? 's' : ''} of ${fmt(D * 1e3, 3)} mm`], ['Interface method', name], ['Channel (mm)', `${fmt(L * 1e3, 4)} × ${fmt(H * 1e3, 4)}`], ['Grid', `${nx} × ${ny} (${fmt(dx * 1e6, 3)} × ${fmt(dy * 1e6, 3)} µm)`], ['Fluid area (mm²)', Aflu * 1e6], ['Mean inlet velocity (m/s)', U], ['Flow-through time (ms)', tFlow * 1e3],
      ['Liquid: density (kg/m³) · viscosity (mPa·s)', `${fmt(r1, 4)} · ${fmt(m1 * 1e3, 4)}`], ['Gas: density (kg/m³) · viscosity (mPa·s)', `${fmt(r2, 4)} · ${fmt(m2 * 1e3, 4)}`], ['Surface tension (mN/m)', sigma * 1e3], ['Gravity, along −y (m/s²)', g], ['Reynolds number ρ U 2H/μ', Re], ['Weber number ρ U² H/σ', Number.isFinite(We) ? We : 'no surface tension'], ['Capillary number μ U/σ', Number.isFinite(Ca) ? Ca : 'no surface tension'], ['Eötvös number Δρ g H²/σ', Number.isFinite(Eo) ? Eo : 'no surface tension'],
      ['Initial gas volume (mm³ per mm depth)', r.vol0 * 1e6], ['Time steps', r.steps], ['Stability limit of the time step without flow (µs)', r.dtFix * 1e6], ['Pressure-solver iterations per step', r.steps ? r.pIters / r.steps : 0], ['Largest |∇·u| (1/s)', r.div]] },
    { title: 'History', columns: ['t (ms)', 'Gas in the channel (mm³/mm)', 'Gas in (mm³/mm)', 'Gas out (mm³/mm)', 'Centroid x (mm)', 'Leading edge (mm)', 'Pressure drop (Pa)', 'Mean wall shear (Pa)', 'Largest wall shear (Pa)', 'Largest speed (m/s)'], rows: hr }];
  for (const k of Object.keys(outputs)) if (!Number.isFinite(outputs[k])) delete outputs[k];
  return { summary: `${kind === 'slug' ? 'Air slug' : `${cen.length} bubbles`} carried through the ${c.name} at ${fmt(U, 3)} m/s (${method === 'ls' ? 'level set' : 'volume of fluid'}): gas velocity ${fmt(uC / U, 3)} × the mean liquid velocity, pressure drop up to ${fmt(dpMax, 3)} Pa${gone ? ` against ${fmt(dpEnd, 3)} Pa for the liquid alone` : ''}, wall shear up to ${fmt(tauPk, 3)} Pa; ${fmt((100 * r.vol) / Math.max(r.vol0, 1e-300), 3)} % of the gas is still in the channel; gas volume balance ${fmt(100 * Math.abs(bal), 2)} %.`,
    kpis, warnings: W, recommendations: ['Check the gas volume balance and the divergence first, then refine the cells across the gap: interface positions converge at about first order.', 'Compare the pressure drop and the wall shear during the passage with their values once the gas has left (lengthen the simulated time until the hold-up is zero) to quantify the sparging effect.', 'Surface tension limits the time step as (cell size)^1.5: for long simulated times use a coarser grid or a higher velocity.'], plots, tables,
    balances: [{ name: 'Gas volume (m³ per metre depth): initial + inflow vs hold-up + outflow', in: r.vol0 + r.volIn, out: r.vol + r.volOut }], outputs };
}

/** Study "two-phase flow with a free interface": the coupled one-fluid Navier–Stokes solver with volume-of-fluid or level-set interface capturing. */
async function runTwoPhase(v, ctx) {
  if (v.tpDomain === 'spacer' || v.tpDomain === 'import') return runTwoPhaseChannel(v, ctx);
  const c = twoPhaseCase(v), { Wt, Ht, nx, ny, g, kind, sigma, len, tc, tEnd } = c, W = [], warn = (msg) => W.push({ level: 'warn', msg }), n = nx * ny, nu1 = nx + 1, dx = Wt / nx, dy = Ht / ny;
  ctx?.progress?.(0.02, 'Two-phase flow: start');
  const r = await twoPhase2D({ method: c.method, nx, ny, W: Wt, Hh: Ht, rhoA: c.rhoA, rhoB: c.rhoB, muA: c.muA, muB: c.muB, sigma, gy: -g, sd: c.sd, tEnd, slipSide: c.slip, slipTB: c.slip, snaps: [tEnd / 3, (2 * tEnd) / 3], maxSteps: 8000, cfl: clamp(v.tpCfl ?? 0.25, 0.05, 0.5) }, ctx);
  ctx?.progress?.(0.985, 'Post-processing');
  const h = r.hist, last = h.t.length - 1, liq = (a) => (c.light ? 1 - a : a), mm = (a) => Array.from(a, (x) => x * 1e3);
  const xs = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx * 1e3), ys = Array.from({ length: ny }, (_, j) => (j + 0.5) * dy * 1e3);
  const rows = (f) => Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => f(j * nx + i, i, j)));
  const U = rows((P, i, j) => 0.5 * (r.u[j * nu1 + i] + r.u[j * nu1 + i + 1])), V = rows((P) => 0.5 * (r.v[P] + r.v[P + nx]));
  const shp = [{ x: mm(c.outline.x), y: mm(c.outline.y), closed: kind === 'bubble', color: '#ffffff', dash: true }];
  const base = { type: 'field', xlabel: 'x (mm)', ylabel: 'y (mm)', x: xs, y: ys, equal: true }, fld = { ...base, zlabel: 'Liquid volume fraction', zunit: '–', zmin: 0, zmax: 1, cmap: 'viridis', shapes: shp };
  const name = c.method === 'ls' ? 'level set (third-order upwind, redistancing, volume correction)' : 'volume of fluid (THINC/WLIC)', A2 = Wt * Ht, vLiq0 = c.light ? A2 - r.vol0 : r.vol0, vLiq = c.light ? A2 - r.vol : r.vol, vErr = (r.vol - r.vol0) / Math.max(r.vol0, 1e-300);
  const Eo = sigma > 0 ? (Math.abs(c.r1 - c.r2) * g * len * len) / sigma : Infinity, Ug = Math.sqrt(g * len), Re = (c.r1 * Ug * len) / c.m1, keMax = Math.max(...h.ke), umaxAll = Math.max(...h.umax);
  const plots = [{ ...fld, title: `Liquid fraction and velocity at t = ${fmt(r.t, 3)} s (${c.method === 'ls' ? 'level set' : 'volume of fluid'})`, z: rows((P) => clamp(liq(r.a[P]), 0, 1)), u: U, v: V, vectors: true, note: 'Dashed white line: initial interface. The velocity field is that of the single set of momentum equations shared by both phases.' },
    ...(r.snaps || []).map((s) => ({ ...fld, title: `Liquid fraction at t = ${fmt(s.t, 3)} s`, z: rows((P) => clamp(liq(s.a[P]), 0, 1)) })),
    { ...base, title: 'Pressure', zlabel: 'Pressure (zero at the reference cell)', zunit: 'Pa', z: rows((P) => r.p[P] - r.p[(ny - 1) * nx]), cmap: 'coolwarm', contours: 10 },
    { ...base, title: 'Velocity magnitude', zlabel: 'Speed', zunit: 'm/s', z: rows((P, i, j) => Math.hypot(U[j][i], V[j][i])), cmap: 'turbo', u: U, v: V, stream: true }];
  const kpis = [{ label: 'Change of the phase volume', value: 100 * vErr, unit: '%', status: Math.abs(vErr) > 1e-3 ? 'warn' : 'ok', help: c.method === 'ls' ? 'The level set is shifted by a constant after every step so that the phase volume is restored' : 'Flux-form transport of the volume fraction conserves the phase volume to round-off' },
    { label: 'Largest velocity during the run', value: umaxAll, unit: 'm/s' }, { label: 'Peak kinetic energy', value: keMax, unit: 'J per m depth' }, { label: 'Simulated time', value: r.t, unit: 's', status: r.done ? 'ok' : 'warn' }, { label: 'Time steps', value: r.steps, unit: '' }, { label: 'Largest velocity divergence', value: r.div, unit: '1/s', help: 'max |∇·u| after the last pressure projection' }];
  const outputs = { phaseVolumeError: vErr, maxVelocity: umaxAll, kineticEnergyMax: keMax, timeSteps: r.steps }, sumRows = [], recs = [];
  let summary;
  if (kind === 'dam') {
    const sT = Math.sqrt((2 * g) / c.a), Tn = h.t.map((t) => t * sT), Z = h.xf.map((x) => (Wt - x) / c.a), col = h.hL.map((x) => (Ht - x) / c.hc), n2 = c.hc / c.a, mmk = Math.abs(n2 - 2) < 0.1, Zend = Z[last], hit = Tn[Z.findIndex((z) => z >= 0.98 * (Wt / c.a))] ?? null;
    const T1 = hit ?? Tn[last], sl = histSlope(Tn, Z, 0.4 * T1, 0.95 * T1), slE = histSlope(MARTIN_MOYCE.T, MARTIN_MOYCE.Z, 1.1, 3.4);
    plots.push({ type: 'line', title: 'Surge front along the floor and height of the column', xlabel: 'T = t √(2g/a)', ylabel: 'Z = x_front / a  ·  column height / initial height', series: [{ name: 'Front position Z (computed)', x: Tn, y: Z }, ...(mmk ? [{ name: 'Martin & Moyce (1952), n² = 2', x: MARTIN_MOYCE.T, y: MARTIN_MOYCE.Z, mode: 'points' }] : []), { name: 'Column height at the left wall ÷ initial height', x: Tn, y: col, dash: true }], note: mmk ? 'Points: surge-front positions of Martin & Moyce (1952), Table 2 (n² = 2, a = 2¼ in, mean of the runs). The instant of release was not measured for that table — its runs are aligned at Z = 1.44, T = 1.19 — so the points may be shifted along T as a whole: compare the slope (front speed), not the absolute time.' : 'Martin & Moyce data are shown when the column is twice as high as wide (n² = 2).' });
    kpis.unshift({ label: 'Front position at the end, x/a', value: Zend, unit: '' }, { label: 'Front speed dZ/dT', value: sl, unit: '', help: `Slope of the front position over the second half of the travel; Martin & Moyce measured ${fmt(slE, 3)} for n² = 2, and the frictionless shallow-water limit (Ritter) is 2 √(n²/2) = ${fmt(2 * Math.sqrt(n2 / 2), 3)}` }, { label: 'Front speed', value: (sl * c.a * sT), unit: 'm/s' });
    if (hit !== null) kpis.unshift({ label: 'Front reaches the far wall at T', value: hit, unit: '' });
    Object.assign(outputs, { frontPosition: Zend, frontSpeedDimensionless: sl });
    if (mmk) { const k2 = MARTIN_MOYCE.T.map((T, k) => (T <= Tn[last] && (hit === null || T <= hit) ? [T, histAt(Tn, Z, T), MARTIN_MOYCE.Z[k]] : null)).filter(Boolean); if (k2.length) sumRows.push(...k2.map((q) => [`Front position Z at T = ${q[0]}: computed · Martin & Moyce`, `${fmt(q[1], 3)} · ${fmt(q[2], 3)}`])); }
    summary = `Dam break of a ${fmt(c.a * 1e3, 3)} × ${fmt(c.hc * 1e3, 3)} mm liquid column (${c.method === 'ls' ? 'level set' : 'volume of fluid'}): front at x/a = ${fmt(Zend, 3)} after T = ${fmt(Tn[last], 3)}, front speed dZ/dT = ${fmt(sl, 3)}, phase volume conserved to ${fmt(100 * Math.abs(vErr), 2)} %.`;
    recs.push('Compare the front position with the Martin & Moyce points (column twice as high as wide) and refine the grid until it stops changing.');
  } else if (kind === 'bubble') {
    const up = c.r2 < c.r1 ? 1 : -1, vr = h.vr.map((x) => up * x); let vm = 0, tm = 0; vr.forEach((x, k) => { if (x > vm) { vm = x; tm = h.t[k]; } });
    const cmin = Math.min(...h.circ.slice(1)), R0 = 0.5 * c.D, cells = c.D / Math.min(dx, dy);
    plots.push({ type: 'line', title: 'Bubble centroid and rise velocity', xlabel: 'Time (s)', ylabel: 'Centroid height (mm) · rise velocity (mm/s)', series: [{ name: 'Centroid height (mm)', x: h.t, y: mm(h.yc) }, { name: 'Rise velocity of the phase (mm/s)', x: h.t, y: mm(vr) }] },
      { type: 'line', title: 'Bubble shape: circularity', xlabel: 'Time (s)', ylabel: 'Perimeter of the equal-area circle ÷ perimeter', ymax: 1.05, series: [{ name: 'Circularity', x: h.t.slice(1), y: h.circ.slice(1) }], note: '1 for a circle; values below 1 measure the deformation. The perimeter is integrated from |∇α|, so the first value reflects the grid resolution of the initial circle.' });
    kpis.unshift({ label: 'Rise velocity at the end', value: vr[last], unit: 'm/s' }, { label: 'Largest rise velocity', value: vm, unit: 'm/s', help: `Reached at t = ${fmt(tm, 3)} s` }, { label: 'Centroid displacement', value: up * (h.yc[last] - h.yc[0]), unit: 'm' }, { label: 'Smallest circularity', value: cmin, unit: '' });
    if (sigma > 0) kpis.push({ label: 'Pressure jump across the interface ÷ σ/R', value: r.dpJump / (sigma / R0), unit: '', help: 'Laplace law for a two-dimensional (cylindrical) interface: Δp = σ/R; departs from 1 while the bubble deforms and accelerates' });
    Object.assign(outputs, { riseVelocity: vr[last], riseVelocityMax: vm, circularityMin: cmin });
    if (cells < 12) warn(`The bubble spans ${fmt(cells, 3)} cells; at least 12–16 cells per diameter are needed for the curvature (surface tension) to be resolved — raise the cell counts or the bubble size.`);
    summary = `Bubble of ${fmt(c.D * 1e3, 3)} mm (${c.method === 'ls' ? 'level set' : 'volume of fluid'}): rise velocity ${fmt(vr[last], 3)} m/s at the end (largest ${fmt(vm, 3)} m/s), centroid moved ${fmt(up * (h.yc[last] - h.yc[0]) * 1e3, 3)} mm, circularity down to ${fmt(cmin, 3)}; phase volume conserved to ${fmt(100 * Math.abs(vErr), 2)} %.`;
    recs.push('The Eötvös and Reynolds numbers in the set-up table place the bubble on the shape-regime map (spherical, ellipsoidal, skirted).');
  } else {
    const kx = ((kind === 'rt' ? 2 : 1) * Math.PI) / Wt, etaL = h.hL.map((x) => (kind === 'rt' ? x : Ht - x) - c.h0);
    if (kind === 'slosh') {
      const om = sloshOmega(kx, c.h0, Ht - c.h0, c.r1, c.r2, g, sigma), Tth = om > 0 ? (2 * Math.PI) / om : 0, Tnum = histPeriod(h.t, etaL, 0);
      plots.push({ type: 'line', title: 'Free-surface elevation at the left wall', xlabel: 'Time (s)', ylabel: 'Elevation above the still level (mm)', series: [{ name: 'Computed', x: h.t, y: mm(etaL) }, { name: 'Linear wave theory (inviscid, small amplitude)', x: h.t, y: h.t.map((t) => c.amp * Math.cos(om * t) * 1e3), dash: true }], note: 'Theory: ω² = [(ρ₁ − ρ₂) g k + σ k³] / [ρ₁ coth(k h₁) + ρ₂ coth(k h₂)] with k = π/W (first sloshing mode).' });
      kpis.unshift({ label: 'Sloshing period, computed', value: Tnum, unit: 's', status: Tnum > 0 ? 'ok' : 'warn', help: 'From the zero crossings of the elevation at the left wall; needs at least half a period of simulated time' }, { label: 'Sloshing period, linear theory', value: Tth, unit: 's' });
      Object.assign(outputs, { sloshPeriod: Tnum, sloshPeriodTheory: Tth });
      if (c.amp > 0.1 * c.h0) W.push({ level: 'info', msg: 'The initial amplitude exceeds 10 % of the liquid depth: the motion is nonlinear and the period departs from the linear-wave value.' });
      summary = `Sloshing in a ${fmt(Wt * 1e3, 3)} mm tank filled to ${fmt(c.h0 * 1e3, 3)} mm (${c.method === 'ls' ? 'level set' : 'volume of fluid'}): period ${Tnum > 0 ? fmt(Tnum, 3) + ' s' : 'not yet measurable'} against ${fmt(Tth, 3)} s from linear wave theory; phase volume conserved to ${fmt(100 * Math.abs(vErr), 2)} %.`;
    } else {
      const At = (c.r1 - c.r2) / (c.r1 + c.r2), gr = Math.sqrt(Math.max(0, At * g * kx - (sigma * kx ** 3) / (c.r1 + c.r2)));
      plots.push({ type: 'line', title: 'Height of the heavy-phase centroid and kinetic energy', xlabel: 'Time (s)', ylabel: 'Centroid height (mm) · kinetic energy (mJ per m depth)', series: [{ name: 'Centroid height of the heavy phase (mm)', x: h.t, y: mm(h.yc) }, { name: 'Kinetic energy (mJ/m)', x: h.t, y: mm(h.ke) }] });
      kpis.unshift({ label: 'Fall of the heavy-phase centroid', value: h.yc[0] - h.yc[last], unit: 'm' }, { label: 'Linear growth rate √(A g k)', value: gr, unit: '1/s', help: `Inviscid Rayleigh–Taylor growth rate of the seeded mode, Atwood number ${fmt(At, 3)}` });
      Object.assign(outputs, { centroidFall: h.yc[0] - h.yc[last], growthRateTheory: gr });
      summary = `Rayleigh–Taylor overturning (${c.method === 'ls' ? 'level set' : 'volume of fluid'}): the heavy-phase centroid fell ${fmt((h.yc[0] - h.yc[last]) * 1e3, 3)} mm in ${fmt(r.t, 3)} s (linear growth rate ${fmt(gr, 3)} 1/s); phase volume conserved to ${fmt(100 * Math.abs(vErr), 2)} %.`;
    }
    recs.push('For the comparison with linear wave theory keep the amplitude below a few per cent of the depth and the viscosity low.');
  }
  plots.push({ type: 'line', title: 'Conservation and kinetic energy', xlabel: 'Time (s)', ylabel: 'Relative volume change (10⁻⁶) · kinetic energy ÷ peak', series: [{ name: 'Phase-volume change (parts per million)', x: h.t, y: h.vol.map((x) => (1e6 * (x - r.vol0)) / Math.max(r.vol0, 1e-300)) }, { name: 'Kinetic energy ÷ peak', x: h.t, y: h.ke.map((x) => x / Math.max(keMax, 1e-300)) }, { name: 'Largest speed ÷ peak', x: h.t, y: h.umax.map((x) => x / Math.max(umaxAll, 1e-300)), dash: true }] });
  if (!r.done) warn(`The run stopped after ${r.steps} steps at t = ${fmt(r.t, 3)} s of ${fmt(tEnd, 3)} s: the time step is limited to ${fmt(r.dtFix, 3)} s by surface tension, viscosity or gravity waves on this grid — shorten the simulated time or coarsen the grid.`);
  if (Math.abs(vErr) > 1e-3) warn(`The phase volume changed by ${fmt(100 * vErr, 3)} % — refine the grid or lower the CFL number.`);
  if (Math.max(c.r1, c.r2) / Math.min(c.r1, c.r2) > 2000) W.push({ level: 'info', msg: 'Density ratios above about 2000 make the variable-density pressure equation stiff; the run stays stable but the pressure solver needs more iterations.' });
  W.unshift({ level: 'info', msg: `Coupled two-phase flow, ${TP_CASES[kind]}: one-fluid Navier–Stokes equations with ${name} on ${nx} × ${ny} cells, ${r.steps} steps; density ratio ${fmt(Math.max(c.r1, c.r2) / Math.min(c.r1, c.r2), 3)}, viscosity ratio ${fmt(Math.max(c.m1, c.m2) / Math.min(c.m1, c.m2), 3)}${sigma > 0 ? `, Eötvös (Bond) number ${fmt(Eo, 3)}` : ', no surface tension'}.` });
  const stp = Math.max(1, Math.ceil(h.t.length / 40)), hr = [];
  for (let k = 0; k < h.t.length; k += stp) hr.push([h.t[k], (h.vol[k] - r.vol0) / Math.max(r.vol0, 1e-300), h.umax[k], h.ke[k], h.xc[k] * 1e3, h.yc[k] * 1e3, h.vr[k]]);
  const tables = [{ title: 'Two-phase study: set-up and numerics', columns: ['Item', 'Value'], rows: [['Case', TP_CASES[kind]], ['Interface method', name], ['Equations', 'One-fluid incompressible Navier–Stokes, variable density and viscosity, continuum surface force, balanced-force projection'], ['Grid', `${nx} × ${ny} (${fmt(dx * 1e3, 3)} × ${fmt(dy * 1e3, 3)} mm)`], ['Tank (mm)', `${fmt(Wt * 1e3, 4)} × ${fmt(Ht * 1e3, 4)}`], ['Liquid: density (kg/m³) · viscosity (mPa·s)', `${fmt(c.r1, 4)} · ${fmt(c.m1 * 1e3, 4)}`], ['Second phase: density (kg/m³) · viscosity (mPa·s)', `${fmt(c.r2, 4)} · ${fmt(c.m2 * 1e3, 4)}`], ['Surface tension (mN/m)', sigma * 1e3], ['Gravity (m/s²)', g],
      ['Reference length ℓ (mm)', len * 1e3], ['Reference time √(ℓ/g) (s)', tc], ['Reynolds number ρ₁ √(gℓ) ℓ/μ₁', Re], ['Eötvös (Bond) number Δρ g ℓ²/σ', Number.isFinite(Eo) ? Eo : 'no surface tension'], ['Walls', c.slip ? 'free-slip' : 'no-slip'], ['Time steps', r.steps], ['Stability limit of the time step without flow (s)', r.dtFix], ['Pressure-solver iterations per step', r.steps ? r.pIters / r.steps : 0], ['Largest |∇·u| (1/s)', r.div], ...(c.method === 'ls' ? [['Sum of the level-set volume-correction shifts ÷ cell size', r.lsShift / Math.min(dx, dy)]] : []), ...sumRows] },
    { title: 'History', columns: ['t (s)', 'Relative volume change', 'Largest speed (m/s)', 'Kinetic energy (J/m)', 'Centroid x (mm)', 'Centroid y (mm)', 'Vertical velocity of the phase (m/s)'], rows: hr, note: 'Centroid and velocity refer to the phase inside the initial region (the liquid column, the bubble, or the lower/upper layer).' }];
  recs.push('Check the phase-volume change and the divergence first; then repeat on a finer grid (Mesh tab) — interface problems converge at first order in the cell size.', 'Switch between volume of fluid and level set: agreement of the two methods is a strong check of the interface dynamics.');
  for (const k of Object.keys(outputs)) if (!Number.isFinite(outputs[k])) delete outputs[k];
  return { summary, kpis, warnings: W, recommendations: recs, plots, tables, balances: [{ name: 'Liquid volume (m³ per metre depth): initial vs final', in: vLiq0, out: vLiq }, { name: 'Mass of both phases (kg per metre depth): initial vs final', in: c.rhoA * r.vol0 + c.rhoB * (A2 - r.vol0), out: c.rhoA * r.vol + c.rhoB * (A2 - r.vol) }], outputs };
}

/** Hindered slip velocity of the two-fluid drag law at dispersed fraction a: k′(u_r) u_r = α_c |ρ_d − ρ_c| g (drag–buoyancy balance in the mixture). */
export function hinderedSlip(a, d, rhoC, rhoD, muC, g = 9.80665) {
  const ac = 1 - a, drv = ac * Math.abs(rhoD - rhoC) * g;
  let ur = terminalSN(d, rhoC, rhoD, muC, g) * ac ** 3.65;
  for (let k = 0; k < 300; k++) { const un = drv / ((0.75 * schillerNaumann((ac * rhoC * ur * d) / muC) * muC * ac ** -2.65) / (d * d)); if (Math.abs(un - ur) < 1e-14 * un) { ur = un; break; } ur = 0.5 * (ur + un); }
  return ur;
}

/** Study "dispersed two-phase flow": the Eulerian–Eulerian two-fluid model in a closed column (batch settling or flotation) or a flow-through channel. */
async function runTwoFluid(v, ctx) {
  // domain: the rectangular column / channel of this group, or the spacer-filled channel / imported section of the Geometry inputs (through-flow)
  const chan = v.tfDomain === 'spacer' || v.tfDomain === 'import' ? mfChannel(v, v.tfDomain, v.tfChNy ?? 12, 200) : null;
  const L = chan ? chan.L : clamp(v.tfL, 1, 1e5) * 1e-3, H = chan ? chan.H : clamp(v.tfH, 1, 1e5) * 1e-3, nx = chan ? chan.nx : clamp(Math.round(v.tfNx), 3, 200), ny = chan ? chan.ny : clamp(Math.round(v.tfNy), 8, 240), g = clamp(v.tfG, 0.01, 100), flowC = chan ? true : v.tfCase === 'flow';
  const rc = Math.max(v.tfRhoC, 1e-3), muc = Math.max(v.tfMuC, 1e-6) * 1e-3, rd = Math.max(v.tfRhoD, 1e-3), dP = clamp(v.tfDp, 0.01, 2e4) * 1e-6, aMax = clamp(v.tfAmax, 0.3, 0.74), a0 = clamp(v.tfAlpha / 100, 1e-6, 0.9 * aMax), W = [], warn = (msg) => W.push({ level: 'warn', msg });
  const ut = terminalSN(dP, rc, rd, muc, g), ur0 = hinderedSlip(a0, dP, rc, rd, muc, g), uh = (1 - a0) * ur0, heavy = rd > rc, ReP = (rc * ut * dP) / muc, tauP = ((rd + 0.5 * rc) * dP * dP) / (18 * muc), U = flowC ? clamp(v.tfU, 1e-5, 20) : 0;
  const tEnd = flowC ? (clamp(v.tfTimeF, 0.05, 50) * L) / U : (clamp(v.tfTimeB, 0.02, 5) * H) / Math.max(uh, 1e-12), dep = flowC && v.tfDeposit !== false;
  ctx?.progress?.(0.02, 'Two-fluid model: start');
  const r = await twoFluid2D({ nx, ny, L, H, rhoC: rc, muC: muc, rhoD: rd, dP, gy: -g, alpha0: a0, flow: flowC ? { U, alphaIn: a0 } : null, cvm: clamp(v.tfCvm, 0, 2), Dtd: Math.max(0, v.tfDtd), alphaMax: aMax, tEnd, slipC: !!v.tfSlip, deposit: dep, maxSteps: chan ? 20000 : 12000, solid: chan ? chan.mk.solid : null }, ctx);
  ctx?.progress?.(0.985, 'Post-processing');
  const h = r.hist, last = h.t.length - 1, nu1 = nx + 1, dx = L / nx, dy = H / ny, mm = (a) => Array.from(a, (x) => x * 1e3), xs = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx * 1e3), ys = Array.from({ length: ny }, (_, j) => (j + 0.5) * dy * 1e3);
  const rows = (f) => Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => f(j * nx + i, i, j)));
  const cen = (uu, vv) => [rows((P, i, j) => 0.5 * (uu[j * nu1 + i] + uu[j * nu1 + i + 1])), rows((P) => 0.5 * (vv[P] + vv[P + nx]))], [Ud, Vd] = cen(r.ud, r.vd), [Uc, Vc] = cen(r.uc, r.vc);
  const base = { type: 'field', xlabel: 'x (mm)', ylabel: 'y (mm)', x: xs, y: ys, equal: L / H <= 8 && H / L <= 8, ...(chan ? { mask: Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => !!chan.mk.solid[j * nx + i])), shapes: chan.mk.shapes.map((q) => ({ x: mm(q.x), y: mm(q.y), closed: q.closed, color: '#0f172a' })) } : {}) }, prof = (a) => Array.from({ length: ny }, (_, j) => { let s = 0; for (let i = 0; i < nx; i++) s += a[j * nx + i]; return (100 * s) / nx; });
  const bal = (r.vol0 + r.dIn - r.dOut - r.dep - r.vol) / Math.max(r.vol0 + r.dIn, 1e-300), amaxEnd = h.amax[last], phase = heavy ? 'particles' : 'bubbles or drops', dir = heavy ? 'settling' : 'rise';
  const plots = [{ ...base, title: `Dispersed-phase volume fraction and velocity at t = ${fmt(r.t, 3)} s`, zlabel: 'Dispersed-phase fraction', zunit: '%', z: rows((P) => 100 * r.al[P]), zmin: 0, cmap: 'viridis', u: Ud, v: Vd, vectors: true, note: 'Arrows: velocity of the dispersed phase (its own momentum equation).' },
    { ...base, title: 'Continuous-phase speed and streamlines', zlabel: 'Speed of the continuous phase', zunit: 'mm/s', z: rows((P, i, j) => 1e3 * Math.hypot(Uc[j][i], Vc[j][i])), cmap: 'turbo', u: Uc, v: Vc, stream: true },
    { ...base, title: 'Slip velocity between the phases', zlabel: '|u_d − u_c|', zunit: 'mm/s', z: rows((P, i, j) => 1e3 * Math.hypot(Ud[j][i] - Uc[j][i], Vd[j][i] - Vc[j][i])), cmap: 'viridis' },
    { ...base, title: 'Shared pressure in excess of the continuous-phase hydrostatic pressure', zlabel: 'p − ρ_c g (H − y)', zunit: 'Pa', z: rows((P) => r.pExcess[P]), cmap: 'coolwarm', contours: 8 }];
  const kpis = [{ label: `Terminal ${dir} velocity of one particle (Schiller–Naumann)`, value: ut, unit: 'm/s' }, { label: 'Particle Reynolds number at terminal velocity', value: ReP, unit: '' }, { label: `Hindered ${dir} velocity at the feed fraction (theory)`, value: uh, unit: 'm/s', help: 'α_c × slip velocity from the drag–buoyancy balance with the Wen–Yu swarm correction; equals u_t (1 − α)^4.65 (Richardson–Zaki) in the Stokes regime' }];
  const outputs = { terminalVelocity: ut, hinderedVelocity: uh, particleReynolds: ReP, dispersedVolumeError: bal, maxFraction: amaxEnd, timeSteps: r.steps }, sumRows = [], recs = [];
  let summary;
  const balances = [{ name: 'Dispersed-phase volume (m³ per metre depth): initial + inflow vs hold-up + outflow + deposit', in: r.vol0 + r.dIn, out: r.vol + r.dOut + r.dep }];
  if (!flowC) {
    const zBed = (H * a0) / aMax, tMeet = (H - zBed) / Math.max(uh, 1e-12), zF = h.front, t1 = Math.min(r.t, 0.85 * tMeet), t0 = 0.25 * t1, vf = Math.abs(histSlope(h.t, zF, t0, t1)), slipEnd = Math.abs(histAt(h.t, h.slip, 0.3 * t1)), zTh = h.t.map((t) => (heavy ? Math.max(zBed, H - uh * t) : Math.min(H - zBed, uh * t))); // the front meets the packed layer of height H α₀/α_max at t = tMeet
    const yy = ys, pb = (() => { let b2 = 0, tp = 0; for (let i = 0; i < nx; i++) { b2 += r.p[i] / nx; tp += r.p[(ny - 1) * nx + i] / nx; } return b2 - tp; })(), rhoM = r.vol / (L * H) * rd + (1 - r.vol / (L * H)) * rc;
    plots.push({ type: 'line', title: 'Section-mean dispersed fraction over the height', xlabel: 'Dispersed-phase fraction (%)', ylabel: 'y (mm)', series: [{ name: `At t = ${fmt(r.t, 3)} s`, x: prof(r.al), y: yy }, { name: 'Initial', x: yy.map(() => 100 * a0), y: yy, dash: true }], hlines: [{ y: zTh[last] * 1e3, label: 'suspension front, kinematic (Kynch) theory' }] },
      { type: 'line', title: 'Suspension front', xlabel: 'Time (s)', ylabel: 'Front height (mm)', series: [{ name: 'Two-fluid model (half the initial fraction)', x: h.t, y: mm(zF) }, { name: 'Kynch theory with the hindered velocity', x: h.t, y: mm(zTh), dash: true }] });
    kpis.push({ label: 'Front velocity, two-fluid model', value: vf, unit: 'm/s', status: Math.abs(vf / Math.max(uh, 1e-300) - 1) > 0.1 ? 'warn' : 'ok', help: 'Slope of the suspension-front position; kinematic (Kynch) theory gives the hindered velocity' }, { label: 'Slip velocity in the suspension', value: slipEnd, unit: 'm/s', help: `Theory ${fmt(ur0, 3)} m/s` }, { label: 'Largest fraction (packed layer)', value: 100 * amaxEnd, unit: '%' }, { label: 'Pressure difference bottom − top ÷ mixture weight', value: pb / (rhoM * g * (H - dy)), unit: '', help: 'The shared pressure carries the weight of both phases: Δp = [α ρ_d + (1 − α) ρ_c] g H between the lowest and highest cell centres' });
    Object.assign(outputs, { frontVelocity: vf, slipVelocity: slipEnd });
    const conv = Math.max(...h.umax) / Math.max(uh, 1e-300);
    if (conv > 20) W.push({ level: 'info', msg: `Buoyancy-driven convection developed in the column: the largest velocity is ${fmt(conv, 3)} times the hindered ${dir} velocity. The two-fluid equations carry no particle-contact stress below the packing limit, so small lateral differences of the fraction in the dense layer set the mixture in motion; the section-mean front still follows the kinematic theory. A turbulent-dispersion diffusivity of about 10⁻⁵ m²/s keeps the column laterally uniform.` });
    sumRows.push(['Largest velocity ÷ hindered velocity', conv], ['Time for the front to reach the packed layer (s)', tMeet], ['Time for the front to cross the column, H ÷ hindered velocity (s)', H / Math.max(uh, 1e-300)], ['Richardson–Zaki estimate u_t (1 − α)^4.65 (m/s)', ut * (1 - a0) ** 4.65]);
    summary = `Batch ${heavy ? 'settling' : 'flotation'} of ${fmt(dP * 1e6, 3)} µm ${phase} at ${fmt(100 * a0, 3)} % (two-fluid model): front velocity ${fmt(vf * 1e3, 3)} mm/s against ${fmt(uh * 1e3, 3)} mm/s from kinematic theory; terminal velocity ${fmt(ut * 1e3, 3)} mm/s; dispersed volume conserved to ${fmt(100 * Math.abs(bal), 2)} %.`;
    recs.push('Raise the initial fraction to see hindered settling slow the front as (1 − α)^4.65, and the packed layer grow from the wall.');
  } else {
    const k0 = Math.floor(0.75 * last), dIn = h.dIn[last] - h.dIn[k0], dOut = h.dOut[last] - h.dOut[k0], dDep = h.dep[last] - h.dep[k0], capt = dIn > 0 ? 1 - dOut / dIn : 0, haz = Math.min(1, (ut * L) / (U * H)), jo = Array.from({ length: ny }, (_, j) => 100 * r.al[j * nx + nx - 1]);
    plots.push({ type: 'line', title: 'Dispersed fraction across the gap at the outlet', xlabel: 'Dispersed-phase fraction (%)', ylabel: 'y (mm)', series: [{ name: 'Outlet', x: jo, y: ys }, { name: 'Mid-length', x: Array.from({ length: ny }, (_, j) => 100 * r.al[j * nx + (nx >> 1)]), y: ys }, { name: 'Inlet', x: ys.map(() => 100 * a0), y: ys, dash: true }] },
      { type: 'line', title: 'Cumulative dispersed-phase volumes', xlabel: 'Time (s)', ylabel: 'Volume (mm³ per mm depth)', series: [{ name: 'Inflow', x: h.t, y: h.dIn.map((x) => x * 1e6) }, { name: 'Outflow', x: h.t, y: h.dOut.map((x) => x * 1e6) }, { name: 'Captured on the wall', x: h.t, y: h.dep.map((x) => x * 1e6) }, { name: 'Hold-up', x: h.t, y: h.vol.map((x) => x * 1e6), dash: true }] });
    kpis.push({ label: 'Capture efficiency (last quarter of the run)', value: 100 * capt, unit: '%', help: dep ? 'Share of the entering dispersed phase that does not leave with the flow' : 'Wall capture is switched off: the dispersed phase accumulates on the wall and slides to the outlet' }, { label: 'Ideal-settler capture v_t L/(U H) (Hazen)', value: 100 * haz, unit: '%', help: 'Plug flow of a dilute suspension' }, { label: 'Largest fraction', value: 100 * amaxEnd, unit: '%' }, { label: 'Mixture volume balance, outflow ÷ inflow', value: (r.cOut + r.dOut) / Math.max(r.cIn + r.dIn, 1e-300), unit: '' });
    Object.assign(outputs, { captureEfficiency: capt, hazenCapture: haz });
    if (chan) { // deposition on the obstacles and the pressure drop of the suspension
      const dS = h.depS[last] - h.depS[k0], dpE = h.dp[last];
      kpis.push({ label: 'Captured on the filaments / imported solids', value: dIn > 0 ? (100 * dS) / dIn : 0, unit: '% of inflow', help: `Dispersed phase arriving on the solid faces that look ${heavy ? 'upward' : 'downward'} (settling or rising onto them); the rest of the capture is on the channel wall` }, { label: 'Captured on the channel wall', value: dIn > 0 ? (100 * (dDep - dS)) / dIn : 0, unit: '% of inflow' }, { label: 'Pressure drop, inlet − outlet column', value: dpE, unit: 'Pa' });
      Object.assign(outputs, { captureOnSolids: dIn > 0 ? dS / dIn : 0, pressureDrop: dpE });
      sumRows.push(['Domain', chan.name], ['Blocked share of the grid (%)', 100 * chan.mk.solidFraction], ['Dispersed volume captured on the solids ÷ entering, last quarter', dIn > 0 ? dS / dIn : 0]);
      if (chan.mk.note) W.push({ level: 'info', msg: chan.mk.note });
      W.push({ level: 'info', msg: `Deposition model in the channel geometry: the dispersed phase is captured where it ${heavy ? 'settles' : 'rises'} onto the channel wall or onto a solid face (gravitational deposition with the computed phase velocity). Inertial impaction and interception on the upstream faces of the filaments are not modelled: there the dispersed phase slips around the solid with the liquid.` });
    }
    balances.push({ name: 'Mixture volume (m³ per metre depth): inflow vs outflow', in: r.cIn + r.dIn, out: r.cOut + r.dOut });
    sumRows.push(['Dispersed volume captured ÷ entering, last quarter', dIn > 0 ? dDep / dIn : 0], ['Surface loading U H / L (m/s)', (U * H) / L]);
    const gpm = (g * Math.abs(rd - rc) * a0) / rc, uDen = Math.sqrt(gpm * H);
    if (uDen > 0.5 * U) W.push({ level: 'info', msg: `The suspension is ${fmt((100 * Math.abs(rd - rc) * a0) / rc, 3)} % ${heavy ? 'denser' : 'lighter'} than the liquid: its density-current speed √(g′H) = ${fmt(uDen, 3)} m/s is not small against the inlet velocity, so the two-way coupling drives a current along the ${heavy ? 'floor' : 'ceiling'} and the capture departs from the ideal-settler value, which assumes plug flow.` });
    if (r.t * U < 2.5 * L) warn(`Only ${fmt((r.t * U) / L, 3)} flow-through times were simulated; the capture efficiency needs about three to settle.`);
    summary = `Flow-through ${heavy ? 'settler' : 'flotation channel'} (two-fluid model): ${fmt(100 * capt, 3)} % of the ${fmt(dP * 1e6, 3)} µm ${phase} are captured against ${fmt(100 * haz, 3)} % for the ideal settler; terminal velocity ${fmt(ut * 1e3, 3)} mm/s; dispersed volume conserved to ${fmt(100 * Math.abs(bal), 2)} %.`;
    recs.push('Lower the velocity or lengthen the channel until the surface loading U·H/L falls below the terminal velocity for complete capture.');
  }
  plots.push({ type: 'line', title: 'Largest fraction, slip velocity and volume balance', xlabel: 'Time (s)', ylabel: 'Fraction ÷ packing limit · slip ÷ terminal velocity', series: [{ name: 'Largest fraction ÷ packing limit', x: h.t, y: h.amax.map((x) => x / aMax) }, { name: 'Mean slip velocity ÷ single-particle terminal velocity', x: h.t, y: h.slip.map((x) => Math.abs(x) / Math.max(ut, 1e-300)) }] });
  if (!r.done) warn(`The run stopped after ${r.steps} steps at t = ${fmt(r.t, 3)} s of ${fmt(tEnd, 3)} s — shorten the simulated time or coarsen the grid.`);
  if (Math.abs(bal) > 1e-6) warn(`The dispersed-phase volume balance is off by ${fmt(100 * bal, 3)} %.`);
  if (r.clip > 1e-9 * Math.max(r.vol0, 1e-300)) W.push({ level: 'info', msg: `Negative fractions totalling ${fmt(r.clip, 3)} m² were clipped by the positivity limiter.` });
  if (dP / Math.min(dx, dy) > 0.5) warn(`The particles (${fmt(dP * 1e6, 3)} µm) are not small against the cells (${fmt(Math.min(dx, dy) * 1e6, 3)} µm): the interpenetrating-continua assumption of the two-fluid model does not hold — use the interface-capturing study for resolved bubbles and drops.`);
  if (ReP > 1000) W.push({ level: 'info', msg: `Particle Reynolds number ${fmt(ReP, 3)}: the drag is on the Newton plateau C_D = 0.44; large bubbles deform and rise more slowly than rigid spheres.` });
  W.unshift({ level: 'info', msg: `Eulerian–Eulerian two-fluid model on ${nx} × ${ny} cells, ${r.steps} steps: continuity and momentum equations for each phase, one shared pressure, Schiller–Naumann drag with the Wen–Yu swarm correction${v.tfCvm > 0 ? `, virtual mass (C_vm = ${fmt(v.tfCvm, 2)})` : ''}${v.tfDtd > 0 ? ', turbulent dispersion' : ''}; relaxation time of a particle ${fmt(tauP, 3)} s.` });
  const tables = [{ title: 'Two-fluid study: set-up and numerics', columns: ['Item', 'Value'], rows: [['Case', chan ? `${chan.name}: velocity inlet, pressure outlet` : flowC ? 'flow-through channel (velocity inlet, pressure outlet)' : 'closed column (batch)'], ['Grid', `${nx} × ${ny} (${fmt(dx * 1e3, 3)} × ${fmt(dy * 1e3, 3)} mm)`], ['Domain (mm)', `${fmt(L * 1e3, 4)} × ${fmt(H * 1e3, 4)}`], ['Continuous phase: density (kg/m³) · viscosity (mPa·s)', `${fmt(rc, 4)} · ${fmt(muc * 1e3, 4)}`], ['Dispersed phase: density (kg/m³) · diameter (µm)', `${fmt(rd, 4)} · ${fmt(dP * 1e6, 4)}`], ['Feed fraction (%)', 100 * a0], ['Packing limit (%)', 100 * aMax],
      ['Stokes velocity (m/s)', (Math.abs(rd - rc) * g * dP * dP) / (18 * muc)], ['Terminal velocity, Schiller–Naumann (m/s)', ut], ['Hindered slip velocity at the feed fraction (m/s)', ur0], ['Particle relaxation time (s)', tauP], ['Time steps', r.steps], ['Residual of the mixture volume balance, max |∇·(α_c u_c + α_d u_d)| (1/s)', r.div], ['Dispersed-phase volume balance error (relative)', bal], ...sumRows] }];
  recs.push('Check the dispersed-phase volume balance and the mixture divergence first; then refine the grid across the direction of settling (Mesh tab).');
  for (const k of Object.keys(outputs)) if (!Number.isFinite(outputs[k])) delete outputs[k];
  return { summary, kpis, warnings: W, recommendations: recs, plots, tables, balances, outputs };
}

/** Compressible studies: shock tube (Riemann problem) and quasi-1-D converging–diverging nozzle for vapour lines and ejector nozzles. */
function runCompressible(v) {
  const gam = clamp(v.gasGamma, 1.05, 1.67), Rg = 8314.462618 / Math.max(v.gasM, 1), n = clamp(Math.round(v.cmpN), 40, 1000), W = [], cpg = (gam * Rg) / (gam - 1);
  const visc = v.cmpVisc ? { mu: v.cmpMu * 1e-6, kth: (v.cmpMu * 1e-6 * cpg) / 0.95 } : { mu: 0, kth: 0 };
  if (v.study === 'shock') {
    const Lx = v.cmpL, x0 = (clamp(v.cmpX0, 5, 95) / 100) * Lx, st = (p, T) => ({ rho: (p * 1e5) / (Rg * (T + 273.15)), u: 0, p: p * 1e5 }), Ls = st(v.cmpPL, v.cmpTL), Rs = st(v.cmpPR, v.cmpTR);
    const cL = Math.sqrt((gam * Ls.p) / Ls.rho), cR = Math.sqrt((gam * Rs.p) / Rs.rho), star = riemannExact(Ls, Rs, gam, 0), ps = star[3], us = star[4];
    const hi = ps > Rs.p ? Rs : Ls, lo = ps > Rs.p ? 1 : -1, cQ = ps > Rs.p ? cR : cL, Ms = Math.sqrt(((gam + 1) / (2 * gam)) * (Math.max(ps / hi.p, 1) - 1) + 1), Ssh = lo * Ms * cQ;
    const vmax = Math.max(cL, cR, Math.abs(Ssh), Math.abs(us) + Math.max(cL, cR)), tEnd = v.cmpT > 0 ? v.cmpT * 1e-3 : (0.8 * Math.min(x0, Lx - x0)) / vmax;
    const r = euler1D({ n, L: Lx, gam, Rg, init: (x) => (x < x0 ? Ls : Rs), tEnd, ...visc });
    const ex = r.x.map((x) => riemannExact(Ls, Rs, gam, (x - x0) / tEnd)), exT = ex.map((e) => e[2] / (Rg * e[0]));
    let e1 = 0, m0 = 0;
    for (let i = 0; i < n; i++) { e1 += Math.abs(r.rho[i] - ex[i][0]); m0 += ex[i][0]; }
    e1 /= m0;
    const mass0 = Ls.rho * x0 + Rs.rho * (Lx - x0), Tsh = (() => { const side = ps > Rs.p ? Rs : Ls, pr = ps / side.p, g6 = (gam - 1) / (gam + 1), rs = (side.rho * (pr + g6)) / (pr * g6 + 1); return ps / (Rg * rs) - 273.15; })();
    if (tEnd * vmax > Math.min(x0, Lx - x0)) W.push({ level: 'warn', msg: 'The waves reach the ends of the tube within the simulated time; the open-end condition lets them leave and the comparison with the exact Riemann solution no longer applies there.' });
    if (v.cmpVisc) W.push({ level: 'info', msg: 'Viscous stress and heat conduction are included (1-D compressible Navier–Stokes); at this scale they only thicken the contact surface slightly.' });
    W.unshift({ level: 'info', msg: `Riemann problem solved on ${n} cells in ${r.steps} steps (MUSCL–HLLC, SSP-RK2): density differs from the exact solution by ${fmt(100 * e1, 3)} % in the L1 norm.` });
    const line = (title, ylabel, num, exa) => ({ type: 'line', title, xlabel: 'x (m)', ylabel, series: [{ name: 'Finite-volume solution', x: r.x, y: num }, { name: 'Exact Riemann solution', x: r.x, y: exa, dash: true }], vlines: [{ x: x0, label: 'diaphragm' }] });
    const stp = Math.max(1, Math.ceil(n / 40)), rows = [];
    for (let i = 0; i < n; i += stp) rows.push([r.x[i], r.p[i] / 1e5, r.rho[i], r.u[i], r.T[i] - 273.15, r.M[i], ex[i][2] / 1e5]);
    return {
      summary: `Shock tube ${fmt(v.cmpPL, 3)} → ${fmt(v.cmpPR, 3)} bar: shock Mach number ${fmt(Ms, 3)} (${fmt(Math.abs(Ssh), 3)} m/s), contact velocity ${fmt(us, 3)} m/s, intermediate pressure ${fmt(ps / 1e5, 3)} bar.`,
      kpis: [{ label: 'Shock Mach number', value: Ms, unit: '' }, { label: 'Shock speed', value: Math.abs(Ssh), unit: 'm/s' }, { label: 'Intermediate (star) pressure', value: ps / 1e5, unit: 'bar', help: 'Pressure between the shock and the expansion fan, from the exact Riemann solution' }, { label: 'Contact (gas) velocity', value: us, unit: 'm/s' }, { label: 'Temperature behind the shock', value: Tsh, unit: '°C' },
        { label: 'Peak numerical pressure', value: Math.max(...r.p) / 1e5, unit: 'bar' }, { label: 'Density error vs exact solution (L1)', value: 100 * e1, unit: '%', status: e1 > 0.03 ? 'warn' : 'ok' }, { label: 'Simulated time', value: tEnd * 1e3, unit: 'ms' }, { label: 'Time steps', value: r.steps, unit: '' }],
      warnings: W, recommendations: ['Refine the cell count until the L1 error against the exact solution stops falling in proportion.', 'Use the nozzle study to size the motive nozzle of a steam ejector or to check choking in a vapour line.'],
      plots: [line('Pressure', 'p (bar)', r.p.map((p) => p / 1e5), ex.map((e) => e[2] / 1e5)), line('Density', 'ρ (kg/m³)', r.rho, ex.map((e) => e[0])), line('Velocity', 'u (m/s)', r.u, ex.map((e) => e[1])), line('Temperature', 'T (°C)', r.T.map((t) => t - 273.15), exT.map((t) => t - 273.15))],
      tables: [{ title: 'Profiles at the end of the run', columns: ['x (m)', 'p (bar)', 'ρ (kg/m³)', 'u (m/s)', 'T (°C)', 'Mach', 'p exact (bar)'], rows }, { title: 'Gas and wave summary', columns: ['Item', 'Value'], rows: [['Ratio of specific heats γ', gam], ['Gas constant (J/kg·K)', Rg], ['Sound speed, left (m/s)', cL], ['Sound speed, right (m/s)', cR], ['Shock speed (m/s)', Ssh], ['Contact speed (m/s)', us], ['Cells', n], ['Scheme', 'MUSCL (minmod) + HLLC, SSP-RK2']] }],
      balances: [{ name: 'Gas mass (kg per m² of cross-section)', in: mass0 + r.mIn, out: r.mass + r.mOut }],
      outputs: { shockMach: Ms, starPressureBar: ps / 1e5, contactVelocity: us, l1Error: e1 },
    };
  }
  // converging–diverging nozzle
  const Lx = v.cmpL, At = (Math.PI * (v.nzDt * 1e-3) ** 2) / 4, arIn = Math.max(v.nzAin, 1.01), ae = Math.max(v.nzAex, 1.0), xt = 0.35 * Lx, p0 = v.nzP0 * 1e5, T0 = v.nzT0 + 273.15, pb = clamp(v.nzPb * 1e5, 1e-4 * p0, 0.9999 * p0);
  const ratio = (x) => (x < xt ? 1 + (arIn - 1) * 0.5 * (1 + Math.cos((Math.PI * x) / xt)) : 1 + (ae - 1) * 0.5 * (1 - Math.cos((Math.PI * (x - xt)) / (Lx - xt)))), area = (x) => At * ratio(clamp(x, 0, Lx));
  const fric = v.nzFric > 0 ? { f: v.nzFric, D: v.nzDt * 1e-3 } : null;
  const r = euler1D({ n, L: Lx, gam, Rg, area, init: (x) => { const p = p0 + ((pb - p0) * x) / Lx; return { rho: p / (Rg * T0), u: 20, p }; }, left: { p0, T0 }, right: { pb }, steadyTol: 2e-8, steadyFlux: 1e-3, maxSteps: clamp(40 * n, 4000, 40000), cfl: 0.8, fric, ...visc });
  const it = Math.round((xt / Lx) * n - 0.5), ex = nozzleExact(r.x.map(ratio), it, pb / p0, gam), mCh = At * p0 * Math.sqrt(gam / (Rg * T0)) * (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1)));
  const md = mean(r.mdot.slice(Math.round(0.2 * n), Math.round(0.8 * n))), Mmax = Math.max(...r.M), choked = Mmax > 0.98;
  let ish = -1, jump = 0;
  for (let i = it + 1; i < n - 1; i++) { const dpp = (r.p[i + 1] - r.p[i - 1]) / r.p[i]; if (dpp > jump) { jump = dpp; ish = i; } }
  const shock = choked && jump > 0.15 && r.M[Math.max(ish - 2, 0)] > 1.02 ? r.x[ish] : null;
  const thrust = r.rho[n - 1] * r.u[n - 1] ** 2 * area(Lx) + (r.p[n - 1] - pb) * area(Lx);
  const names = { subsonic: 'subsonic throughout (venturi)', shock: 'choked with a normal shock in the diverging part', overexpanded: 'choked, supersonic exit, over-expanded (oblique shocks outside)', supersonic: 'choked, supersonic exit, under-expanded' };
  W.push({ level: 'info', msg: `Quasi-1-D nozzle on ${n} cells: ${r.steps} pseudo-time steps, last density change ${r.resid.toExponential(1)} per step. Regime from isentropic theory: ${names[ex.regime]}.` });
  if (r.resid > 1e-3) W.push({ level: 'warn', msg: 'The time march has not settled to a steady state (the shock may be oscillating at the exit) — change the back pressure slightly or refine the grid.' });
  if (fric) W.push({ level: 'info', msg: 'Wall friction is included (Fanno-type momentum sink), so the flow departs from the isentropic theory shown for comparison.' });
  if (Math.abs(r.mdotIn - r.mdotOut) > 0.01 * Math.abs(md)) W.push({ level: 'warn', msg: 'Inlet and outlet mass flows differ by more than 1 % — the solution is not yet steady.' });
  const stp = Math.max(1, Math.ceil(n / 40)), rows = [];
  for (let i = 0; i < n; i += stp) rows.push([r.x[i], ratio(r.x[i]), r.M[i], ex.M[i], r.p[i] / 1e5, (ex.p[i] * p0) / 1e5, r.T[i] - 273.15, r.u[i]]);
  return {
    summary: `Nozzle at ${fmt(v.nzP0, 3)} bar / ${fmt(v.nzT0, 3)} °C against ${fmt(pb / 1e5, 3)} bar: mass flow ${fmt(md * 3600, 3)} kg/h (${choked ? 'choked' : 'not choked'}), exit Mach number ${fmt(r.M[n - 1], 3)}.`,
    kpis: [{ label: 'Mass flow', value: md * 3600, unit: 'kg/h' }, { label: 'Mass flow ÷ choked (isentropic) value', value: md / mCh, unit: '', help: 'ṁ* = A_t p0 √(γ/(R T0)) (2/(γ+1))^((γ+1)/(2(γ−1)))' }, { label: 'Throat Mach number', value: r.M[it], unit: '', status: choked ? 'ok' : 'warn' }, { label: 'Exit Mach number', value: r.M[n - 1], unit: '' }, { label: 'Exit Mach number, theory', value: ex.M[n - 1], unit: '' },
      { label: 'Exit static pressure', value: r.p[n - 1] / 1e5, unit: 'bar' }, { label: 'Exit velocity', value: r.u[n - 1], unit: 'm/s' }, { label: 'Normal-shock position', value: shock ?? '–', unit: 'm' }, { label: 'Nozzle thrust', value: thrust, unit: 'N' }, { label: 'Pseudo-time steps', value: r.steps, unit: '' }],
    warnings: W, recommendations: [choked ? 'The nozzle is choked: the mass flow depends only on the upstream pressure and temperature and the throat area.' : 'Lower the back pressure or the throat area to choke the nozzle if a fixed motive flow is wanted.', ex.regime === 'shock' ? 'A normal shock stands inside the diverging part — lower the back pressure or shorten the divergent section to recover a supersonic exit.' : null].filter(Boolean),
    plots: [{ type: 'line', title: 'Mach number along the nozzle', xlabel: 'x (m)', ylabel: 'Mach', series: [{ name: 'Finite-volume solution', x: r.x, y: r.M }, { name: 'Isentropic / normal-shock theory', x: r.x, y: ex.M, dash: true }], hlines: [{ y: 1, label: 'sonic' }], vlines: [{ x: xt, label: 'throat' }] },
      { type: 'line', title: 'Static pressure', xlabel: 'x (m)', ylabel: 'p (bar)', series: [{ name: 'Finite-volume solution', x: r.x, y: r.p.map((p) => p / 1e5) }, { name: 'Theory', x: r.x, y: ex.p.map((p) => (p * p0) / 1e5), dash: true }], hlines: [{ y: pb / 1e5, label: 'back pressure' }] },
      { type: 'line', title: 'Temperature and velocity', xlabel: 'x (m)', ylabel: 'T (°C) · u (m/s)', series: [{ name: 'Temperature (°C)', x: r.x, y: r.T.map((t) => t - 273.15) }, { name: 'Velocity (m/s)', x: r.x, y: r.u }] },
      { type: 'line', title: 'Area ratio and mass flow', xlabel: 'x (m)', ylabel: 'A/A_t · ṁ/ṁ*', series: [{ name: 'A / A_throat', x: r.x, y: r.x.map(ratio) }, { name: 'Local mass flow ÷ choked value', x: r.x, y: r.mdot.map((m) => m / mCh) }] }],
    tables: [{ title: 'Profiles along the nozzle', columns: ['x (m)', 'A/A_t', 'Mach', 'Mach (theory)', 'p (bar)', 'p theory (bar)', 'T (°C)', 'u (m/s)'], rows }, { title: 'Gas and nozzle summary', columns: ['Item', 'Value'], rows: [['Ratio of specific heats γ', gam], ['Gas constant (J/kg·K)', Rg], ['Throat area (mm²)', At * 1e6], ['Choked mass flow, isentropic (kg/h)', mCh * 3600], ['Back pressure for fully subsonic flow (bar)', (ex.peSub * p0) / 1e5], ['Design (shock-free supersonic) exit pressure (bar)', (ex.peSup * p0) / 1e5], ['Cells', n], ['Scheme', 'MUSCL (minmod) + HLLC, SSP-RK2 pseudo-time march']] }],
    balances: [{ name: 'Gas mass flow (kg/s)', in: r.mdotIn, out: r.mdotOut }],
    outputs: { massFlowKgH: md * 3600, exitMach: r.M[n - 1], choked: choked ? 1 : 0, throatMach: r.M[it] },
  };
}

/** Design of solver runs (velocity × pitch) on a coarser grid and power-law regression of f and Sh with cross-validation. */
async function trainClosure(v, c, ctx) {
  const geo = v.geom === 'spacer' && v.arr !== 'none', nxc = clamp(Math.round(v.nx / 2), 32, v.nx), nyc = clamp(Math.round(v.ny * 0.625), 12, v.ny), X = [], U = [], f = [], sh = [];
  const pitches = geo ? [1, 1.5] : [1], vel = [0.4, 0.7, 1, 1.5, 2.2];
  let k = 0, prev = null;
  for (const pm of pitches) for (const um of vel) {
    if (um === vel[0]) prev = null;
    ctx?.progress?.(0.97 + (0.03 * k++) / (pitches.length * vel.length), `Regression closure: training run ${k} of ${pitches.length * vel.length}`);
    const vv = { ...v, Uin: c.Uin * um, inletBC: 'velocity', lm: v.lm * pm, nx: nxc, ny: nyc, mode: 'steady', turb: v.turb === 'les' ? 'laminar' : v.turb, maxIter: Math.min(v.maxIter, 500), tol: Math.max(v.tol, 1e-4), energy: false, particles: false, precip: false, pbm: false, foul: false, usr: false, mp: 'off', ml: false, ms: false, engine: 'fv' };
    const cc = caseConfig(vv);
    if (prev) { const f = cc.Uin / prev.U; cc.o.init = { u: prev.r.u.map((x) => x * f), v: prev.r.v.map((x) => x * f), p: prev.r.p.map((x) => x * f) }; } // continuation in velocity on the same grid
    const rr = await solveChannel(cc.o, { tick: ctx?.tick }), qq = post(cc, rr), kd = qq.sp ? qq.sp.kDev ?? qq.sp.kAll : null;
    prev = { r: rr, U: cc.Uin };
    if (!(qq.f > 0) || !Number.isFinite(qq.f)) continue;
    X.push([qq.Re, ...(geo ? [cc.geo.lm / cc.H] : [])]); U.push(rr.Uref); f.push(qq.f); sh.push(kd ? (kd * qq.dh) / cc.fl.D : NaN);
  }
  if (X.length < 4) throw new Error('The regression closure needs at least four usable training runs — the solver did not return a positive friction factor for the sampled velocities.');
  const okS = sh.every((s) => s > 0 && Number.isFinite(s));
  return { X, U, f, sh, geo, nx: nxc, ny: nyc, fitF: fitClosure(X, f), fitS: okS ? fitClosure(X, sh) : null };
}

/** Example outline for the imported-geometry presets: an elliptic strand section (2 : 1), 24 points, arbitrary units (auto-fitted into the channel). */
const STRAND_CAD = { kind: 'polylines', name: 'elliptic strand section (example outline)', polylines: [{ x: Array.from({ length: 24 }, (_, k) => +(2 * Math.cos((2 * Math.PI * k) / 24)).toFixed(4)), y: Array.from({ length: 24 }, (_, k) => +Math.sin((2 * Math.PI * k) / 24).toFixed(4)), closed: true }] };
const TURB_NAMES = { ml: 'algebraic mixing length', ke: 'standard k–ε with wall functions', kw: 'Wilcox k–ω', sst: 'Menter k–ω SST', earsm: 'explicit algebraic Reynolds stress (Wallin–Johansson) on k–ω', rsm: 'differential Reynolds-stress transport (Launder–Reece–Rodi, wall reflection of Gibson–Launder) with ε', les: 'LES, Smagorinsky sub-grid model' };

const F = (key, label, unit, value, min, max, help, extra = {}) => ({ key, label, unit, value, min, max, help, ...extra });
const SEL = (key, label, value, options, help, extra = {}) => ({ key, label, type: 'select', value, options: options.map(([v, l]) => ({ value: v, label: l })), help, ...extra });
const isSpacer = (v) => v.geom === 'spacer', isImport = (v) => v.geom === 'import', hasSpecies = (v) => v.species !== 'off';
const isTP = (v) => v.study === 'twophase', isTF = (v) => v.study === 'twofluid', tpCh = (v) => isTP(v) && (v.tpDomain === 'spacer' || v.tpDomain === 'import'), tpTank = (v) => isTP(v) && !tpCh(v), tpIs = (...k) => (v) => tpTank(v) && k.includes(v.tpCase), tfCh = (v) => isTF(v) && (v.tfDomain === 'spacer' || v.tfDomain === 'import'), tfTank = (v) => isTF(v) && !tfCh(v);
const isCmp = (v) => v.study === 'shock' || v.study === 'nozzle', twoEqSel = (v) => ['ke', 'kw', 'sst', 'earsm', 'rsm'].includes(v.turb), BOOL = (key, label, help, extra = {}) => ({ key, label, type: 'bool', value: false, help, ...extra });

const suite = {
  id: 'cfd', num: 4, title: 'Flow in Membranes, Channels & Equipment (CFD)', short: 'CFD', icon: '🌀',
  tagline: 'Two-dimensional finite-volume CFD of spacer-filled membrane channels, ducts and imported shapes with salt, heat, particle and second-phase transport, turbulence closures up to Reynolds-stress transport, scaling and fouling, plus coupled two-phase flow (free interface and two-fluid model) and 1-D compressible gas studies.',
  description: 'Solves the incompressible Navier–Stokes equations on a staggered Cartesian grid with SIMPLE-type pressure–velocity coupling and a preconditioned conjugate-gradient pressure solver. Spacer filaments, steps, baffles or imported CAD sections are immersed as blocked cells. Salt transport is coupled to solution–diffusion membrane walls, so concentration polarisation, local permeate flux, wall shear, friction factor and Sherwood number come straight from the resolved fields and are compared with the Hagen–Poiseuille, Lévêque/Graetz and Schock–Miquel relations. Optional models add two-equation, algebraic and differential Reynolds-stress turbulence closures, large-eddy simulation, conjugate heat transfer, precipitation with a crystal population balance, a growing fouling layer, one-way transport of a second phase (interface capturing or a dispersed phase with slip), a lattice-Boltzmann start field, a regression closure trained on solver runs, and one-dimensional compressible flow in vapour lines and nozzles. Two multiphase studies solve coupled two-phase flow in a 2-D tank, in the spacer-filled membrane channel or around an imported section: a free interface with density and viscosity jump and surface tension, captured by volume of fluid or by a level set (air slug or bubbles carried through the channel), and a dispersed phase with its own momentum equation (Eulerian–Eulerian two-fluid model; particles or droplets carried through the channel and deposited on walls and filaments).',
  guide: [
    'Choose the geometry: a spacer-filled membrane channel, an empty channel, a sudden expansion, baffles, or an imported STL/OBJ/DXF/GeoJSON section.',
    'Enter the fluid, cross-flow velocity and membrane data (or pull them from the case and the RO suite).',
    'On Model setup pick steady or transient flow, the convection scheme, the wall conditions for salt and heat, and optional turbulence, porous-zone and particle models.',
    'Optional physics is switched on one model at a time on Model setup: inlet and wall types, a second phase, scaling and fouling, multicomponent diffusion, a user-defined scalar, or the compressible gas studies under Study type. Each adds its own results, balance and warnings.',
    'Multiphase flow: under Study type choose the free-interface study (volume of fluid or level set: dam break, rising bubble, sloshing, Rayleigh–Taylor) or the Eulerian–Eulerian two-fluid study (batch settling or flotation, flow-through settler). Their inputs appear on Model setup and their grid on the Mesh tab. Each has a Domain choice: its own rectangular tank, or the spacer-filled channel or imported section of the Geometry inputs with a liquid inlet and a pressure outlet (air slug or bubbles through the spacer; particles or droplets depositing on walls and filaments).',
    'Run. Check the residual history and the conservation closure first, then read the fields, wall profiles and the comparison with correlations.',
    'Use the Mesh tab to quantify numerical uncertainty; the mass-transfer multiplier is offered to the RO design suite.',
  ],
  implemented: ['continuity equation', 'incompressible', 'euler equations', 'stokes-flow', 'reynolds-averaged', 'turbulent kinetic-energy', 'turbulent-dissipation', 'specific-dissipation-rate', 'sst equations', 'large-eddy-simulation', 'species-conservation', 'convection-diffusion', 'the energy equation', 'fourier', 'fick', 'maxwell-stefan', 'darcy equation', 'brinkman', 'forchheimer', 'ergun', 'hagen-poiseuille', 'darcy-weisbach',
    'navier-stokes-species', 'navier-stokes-solution-diffusion', 'cfd-concentration-polarization', 'cfd-porous-media', 'cfd-fouling', 'cfd-particle-deposition', 'cfd-population-balance', 'cfd-precipitation', 'cfd-heat/mass-transfer', 'conjugate heat-transfer', 'eulerian-lagrangian', 'lattice-boltzmann', 'cfd-machine-learning', 'reynolds-stress', 'volume-of-fluid', 'level-set', 'eulerian-eulerian', 'multiphase flow',
    'velocity', 'initial pressure field', 'concentration', 'temperature', 'turbulence quantities', 'phase fractions', 'particle distribution', 'deposited material', 'velocity-inlet', 'mass-flow-inlet', 'pressure-inlet', 'pressure-outlet', 'no-slip wall', 'navier-slip', 'symmetry', 'periodic', 'fully developed', 'wall-function', 'impermeable wall', 'specified species concentration', 'specified species flux', 'membrane permeation flux', 'prescribed temperature', 'prescribed heat flux', 'convective heat-transfer',
    'geometry creation and import', 'computational meshing', 'fluid-property definition', 'laminar-flow', 'turbulent-flow modelling', 'porous-media flow', 'species transport', 'salt transport', 'module heat transfer', 'concentration polarisation', 'membrane-wall transport', 'spacer hydrodynamics', 'particle transport and deposition', 'wall shear stress', 'pressure-drop prediction', 'mixing analysis', 'fouling-layer', 'crystallisation and particle formation', 'transient simulation', 'user-defined physical model', 'mesh-independence', 'numerical convergence monitoring', 'scientific visualisation'],
  equationsNote: 'Scope of the channel study: two-dimensional, incompressible, constant-property flow on a Cartesian grid (uniform in x, wall-clustered in y) with solids represented by blocked cells (stair-step surfaces). A 2-D section represents filaments transverse to the flow; diamond or woven three-dimensional spacer meshes need a 3-D solver, so treat friction and Sherwood numbers as section values and calibrate the 1-D multipliers against element data. Steady runs are valid while the flow is steady (roughly channel Reynolds number below 300–400 with filaments); above that use the transient mode. Turbulence: algebraic mixing length, standard k–ε with log-law wall functions (first cell at y⁺ > 11.6), Wilcox k–ω and Menter k–ω SST (wall functions or integration to the wall), each with transport equations for k and ε or ω. Two Reynolds-stress options: the explicit algebraic (Wallin–Johansson) solution of the stress equations on k–ω with a bounded effective C_μ, and the differential model, which solves transport equations for u′u′, v′v′, w′w′ and u′v′ (exact production, Launder–Reece–Rodi isotropisation-of-production pressure–strain with the Gibson–Launder wall reflection, Daly–Harlow gradient diffusion without its cross-diffusion terms) and for ε; their divergence drives the momentum equations. The differential model is a high-Reynolds-number closure: wall-adjacent cells hold the log-layer stress levels of the wall function (first cell at y⁺ > 11.6), the mean flow is two-dimensional (u′w′ = v′w′ = 0), and steady runs of separated flows may stall at a continuity residual of about 10⁻⁴ — use the upwind scheme or the transient mode there. LES uses the Smagorinsky sub-grid model on the 2-D grid: without vortex stretching it is indicative only. Compressible flow (Euler equations, optionally with viscous stress, heat conduction and wall friction) is solved in one dimension for an ideal gas — shock tube and quasi-1-D nozzle — not in the 2-D channel. Maxwell–Stefan diffusion is solved as a ternary film across the polarisation layer whose thickness comes from the CFD mass-transfer coefficient, not as a coupled 2-D multicomponent field. The second-phase option of the channel study transports a phase on the solved single-phase velocity field: a sharp interface by THINC/WLIC volume-fraction advection or by a level-set function (not volume-conserving), or a dilute dispersed phase by a drift-flux continuity equation with an algebraic (Stokes, hindered) slip velocity and wall deposition. That option is interface kinematics and one-way drift-flux transport: the phases do not act back on the channel flow. Coupled multiphase flow is solved by the two multiphase studies (Study type). Free interface: one-fluid incompressible Navier–Stokes equations with variable density and viscosity and the continuum surface force on a uniform staggered grid, explicit balanced-force projection with a variable-density pressure equation, in a closed rectangular tank or in a channel with a velocity inlet and a pressure outlet in which the spacer filaments or an imported section are blocked cells (the same staircase mask as the channel study, rasterised on the uniform grid; no-slip solids; the gas starts inside the channel and the inlet delivers liquid), interface by THINC/WLIC volume of fluid or by a level set with redistancing and a constant-shift volume correction; two-dimensional, laminar, no phase change, no contact-angle model (90° at the walls), explicit time step limited by the capillary, viscous and convective criteria, and first-order convergence of interface positions with the cell size. Two-fluid model: continuity and momentum equations for a continuous and a mono-sized dispersed phase with one shared pressure, Schiller–Naumann drag with the Wen–Yu swarm correction, virtual mass and turbulent dispersion, a packing limit, closed box or a channel with velocity inlet and pressure outlet, optionally with the spacer filaments or an imported section as blocked cells (no-slip for the liquid, free-slip and impermeable for the dispersed phase, which is captured where it settles or rises onto a wall or a solid face — inertial impaction and interception on the filaments are not modelled); laminar, no particle-contact (solids) pressure below the packing limit, so dense columns develop buoyancy-driven convection that a real suspension damps, no lift force, no coalescence or break-up. Precipitation transports one sparingly soluble salt with first-order wall crystallisation and a four-moment population balance (primary nucleation, linear growth, no aggregation or breakage). The fouling layer feeds back through its hydraulic resistance, not by narrowing the passage. Conjugate heat transfer conducts through the blocked cells. The lattice-Boltzmann option (D2Q9, BGK, laminar, no-slip) supplies the starting field and a comparison; the reported results are those of the finite-volume solver. The regression closure is a cross-validated power law fitted to 5–10 extra solver runs and is valid only inside the sampled range. The pressure inlet is a flow-rate controller on the static pressure of the inlet plane (found by steady iterations, also ahead of a transient run); with a turbulence closure it takes the fully developed (recycled) profile in a plain channel and a uniform one elsewhere, and a target that the passage cannot reach is reported. The periodic option recycles the outlet-plane profile (and turbulence quantities) to the inlet; permeation is retained at the walls. Bounded QUICK is formulated for uniform spacing and is applied unchanged on the clustered y-grid.',

  inputs: [
    { group: 'Geometry', help: 'A 2-D section through the flow passage: x along the flow, y across the gap.', fields: [
      SEL('geom', 'Geometry type', 'spacer', [['spacer', 'Spacer-filled membrane channel'], ['plain', 'Empty channel / duct'], ['step', 'Sudden expansion (backward-facing step)'], ['baffle', 'Baffled duct (bend-like obstacles)'], ['import', 'Imported CAD section']], 'Parametric shapes or an imported outline immersed in the channel.'),
      F('H', 'Channel height (gap)', 'mm', 0.71, 0.05, 2000, '28 mil feed spacer = 0.71 mm; 34 mil = 0.86 mm. For pipes and ducts enter the full gap.', { typical: [0.4, 1.2] }),
      F('L', 'Domain length', 'mm', 12, 0.2, 50000, 'Length of the simulated section.', { showIf: (v) => !isSpacer(v) }),
      SEL('arr', 'Filament arrangement', 'zigzag', [['zigzag', 'Zigzag (alternating walls)'], ['cavity', 'Cavity (all on the bottom wall)'], ['submerged', 'Submerged (mid-channel)'], ['none', 'No filaments']], 'Position of the transverse filaments.', { showIf: isSpacer }),
      F('df', 'Filament diameter', 'mm', 0.36, 0.02, 1000, 'About half the channel height for a two-layer net.', { showIf: (v) => isSpacer(v) && v.arr !== 'none' }),
      F('lm', 'Filament spacing (pitch)', 'mm', 3, 0.2, 5000, 'Centre-to-centre distance between successive filaments.', { showIf: isSpacer, typical: [2, 6] }),
      F('nFil', 'Number of pitches simulated', '', 4, 1, 40, 'Domain length = pitches × spacing. Use at least three so that the middle pitches are free of end effects.', { showIf: isSpacer, step: 1 }),
      F('stepH', 'Step height', '% of gap', 50, 5, 90, 'Solid step under the inlet; the inlet occupies the remaining height.', { showIf: (v) => v.geom === 'step' }),
      F('stepL', 'Step length', '% of length', 15, 2, 60, '', { showIf: (v) => v.geom === 'step' }),
      F('nBaffle', 'Number of baffles', '', 3, 1, 12, 'Thin plates alternating from the bottom and top walls.', { showIf: (v) => v.geom === 'baffle', step: 1 }),
      F('baffleH', 'Baffle height', '% of gap', 55, 5, 85, '', { showIf: (v) => v.geom === 'baffle' }),
      { key: 'cad', label: 'CAD geometry (STL, OBJ, DXF, GeoJSON or x,y points)', type: 'file', value: null, showIf: isImport, help: 'Closed outlines become solid obstacles. Triangle meshes are cut at the mid-plane normal to the chosen axis.' },
      SEL('cadAxis', 'Section plane normal (meshes)', '2', [['2', 'z (x–y section)'], ['1', 'y (x–z section)'], ['0', 'x (y–z section)']], '', { showIf: isImport }),
      SEL('cadFit', 'Placement', 'fit', [['fit', 'Auto-fit into the channel'], ['absolute', 'Use file coordinates × scale']], 'Auto-fit scales the outline uniformly so that it occupies the chosen share of the gap.', { showIf: isImport }),
      F('cadScale', 'Scale factor', '×', 1, 1e-6, 1e6, 'Multiplies the auto-fit size, or converts file units to metres in absolute mode (0.001 for mm).', { showIf: isImport }),
      F('cadSize', 'Occupied share of the gap (auto-fit)', '%', 55, 5, 100, '', { showIf: (v) => isImport(v) && v.cadFit === 'fit' }),
      F('cadX', 'Position along the channel', '% of length', 40, 0, 100, 'Centre (auto-fit) or lower-left corner (absolute).', { showIf: isImport }),
      F('cadY', 'Position across the gap', '% of gap', 50, 0, 100, '', { showIf: isImport }),
      { key: 'cadInvert', label: 'Outline describes the fluid passage (invert)', type: 'bool', value: false, showIf: isImport, help: 'Tick when the imported outline is the flow passage itself rather than an obstacle.' },
    ] },
    { group: 'Fluid and operation', help: 'Properties are evaluated from temperature and salinity unless you enter them yourself.', fields: [
      SEL('propMode', 'Fluid properties', 'seawater', [['seawater', 'Seawater / brine from T and salinity'], ['custom', 'Custom fluid']], ''),
      F('T', 'Temperature', '°C', 25, 1, 95, 'Bulk (inlet) temperature.'),
      F('c0', 'Inlet salt concentration', 'g/L', 35, 0, 250, 'Total dissolved solids of the feed.'),
      F('rho', 'Density', 'kg/m³', 1000, 0.5, 3000, '', { showIf: (v) => v.propMode === 'custom' }),
      F('mu', 'Dynamic viscosity', 'mPa·s', 1, 0.005, 5000, '', { showIf: (v) => v.propMode === 'custom' }),
      F('Dsalt', 'Solute diffusivity', '10⁻⁹ m²/s', 1.5, 0.001, 100, '', { showIf: (v) => v.propMode === 'custom' }),
      F('piCoef', 'Osmotic-pressure coefficient', 'bar per g/L', 0.76, 0, 5, 'Linear osmotic pressure π = coefficient × concentration.', { showIf: (v) => v.propMode === 'custom' }),
      F('Uin', 'Mean inlet (cross-flow) velocity', 'm/s', 0.1, 0.0005, 20, 'RO feed channels run at 0.05–0.3 m/s.', { typical: [0.05, 0.3] }),
    ] },
    { group: 'Membrane walls', help: 'Solution–diffusion walls: Jw = A(ΔP − Δπ(c_wall)), Js = B(c_wall − c_perm).', showIf: (v) => v.species === 'membrane', fields: [
      F('A', 'Water permeability A', 'L/m²·h·bar', 1.25, 0, 50, 'Set to zero for impermeable walls.'),
      F('B', 'Salt permeability B', 'L/m²·h', 0.058, 0, 100, ''),
      F('dPtm', 'Trans-membrane pressure at the inlet', 'bar', 55, 0, 150, 'Feed pressure minus permeate pressure.'),
    ] },
    { group: 'Study type', tab: 'setup', help: 'The channel study is the 2-D incompressible solver. The two gas studies solve the compressible Euler / Navier–Stokes equations in one dimension for vapour lines, vents and ejector nozzles; they ignore the channel inputs.', fields: [
      SEL('study', 'Study', 'channel', [['channel', 'Channel / equipment flow (2-D incompressible)'], ['shock', 'Compressible gas: shock tube / pressure-wave in a vapour line (1-D)'], ['nozzle', 'Compressible gas: converging–diverging nozzle (quasi-1-D)'], ['twophase', 'Multiphase: two-phase flow with a free interface (coupled volume of fluid / level set, 2-D tank)'], ['twofluid', 'Multiphase: dispersed flow, Eulerian–Eulerian two-fluid model (2-D)']], 'Shock tube: sudden opening of a valve or rupture disc between two gas states. Nozzle: motive nozzle of a steam ejector or a choked vent. The two multiphase studies solve the coupled flow of two phases (their inputs appear in the groups below) and ignore the channel inputs.'),
      F('gasGamma', 'Ratio of specific heats γ', '–', 1.33, 1.05, 1.67, 'Steam 1.33 (superheated) to 1.135 (wet), air 1.4.', { showIf: isCmp }),
      F('gasM', 'Molar mass of the gas', 'g/mol', 18.02, 2, 200, 'Steam 18.02, air 28.96. Ideal-gas behaviour is assumed.', { showIf: isCmp }),
      F('cmpL', 'Length of the tube / nozzle', 'm', 1, 0.01, 1000, 'Length of the 1-D domain.', { showIf: isCmp }),
      F('cmpN', 'Number of cells', '', 200, 40, 1000, 'Finite volumes along the axis.', { showIf: isCmp, step: 1 }),
      F('cmpPL', 'Left (high-pressure) state: pressure', 'bar', 2, 0.001, 300, 'Absolute pressure upstream of the diaphragm.', { showIf: (v) => v.study === 'shock' }),
      F('cmpTL', 'Left state: temperature', '°C', 125, -50, 1000, 'Gas temperature upstream of the diaphragm.', { showIf: (v) => v.study === 'shock' }),
      F('cmpPR', 'Right (low-pressure) state: pressure', 'bar', 0.2, 0.001, 300, 'Absolute pressure downstream of the diaphragm.', { showIf: (v) => v.study === 'shock' }),
      F('cmpTR', 'Right state: temperature', '°C', 65, -50, 1000, 'Gas temperature downstream of the diaphragm.', { showIf: (v) => v.study === 'shock' }),
      F('cmpX0', 'Diaphragm position', '% of length', 50, 5, 95, 'Location of the initial discontinuity.', { showIf: (v) => v.study === 'shock' }),
      F('cmpT', 'Simulated time (0 = automatic)', 'ms', 0, 0, 1e6, 'Automatic: until the fastest wave has covered 80 % of the way to the nearer end.', { showIf: (v) => v.study === 'shock' }),
      F('nzP0', 'Stagnation (motive) pressure', 'bar', 3, 0.01, 300, 'Absolute total pressure at the nozzle inlet.', { showIf: (v) => v.study === 'nozzle' }),
      F('nzT0', 'Stagnation temperature', '°C', 134, -50, 1000, 'Total temperature at the nozzle inlet.', { showIf: (v) => v.study === 'nozzle' }),
      F('nzPb', 'Back (suction-chamber) pressure', 'bar', 0.1, 0.0001, 300, 'Static pressure downstream of the nozzle exit.', { showIf: (v) => v.study === 'nozzle' }),
      F('nzDt', 'Throat diameter', 'mm', 10, 0.1, 2000, 'Sets the throat area and the choked mass flow.', { showIf: (v) => v.study === 'nozzle' }),
      F('nzAin', 'Inlet ÷ throat area', '–', 4, 1.01, 100, 'Contraction ratio of the converging part.', { showIf: (v) => v.study === 'nozzle' }),
      F('nzAex', 'Exit ÷ throat area', '–', 4, 1, 100, 'Expansion ratio of the diverging part (1 = converging nozzle).', { showIf: (v) => v.study === 'nozzle' }),
      F('nzFric', 'Darcy friction factor of the wall', '–', 0, 0, 0.2, '0 = frictionless (Euler). A positive value adds the wall-friction momentum sink −f ρu|u|/(2D).', { showIf: (v) => v.study === 'nozzle' }),
      BOOL('cmpVisc', 'Include viscous stress and heat conduction (Navier–Stokes terms)', 'Adds (4/3)μ ∂u/∂x and k ∂T/∂x to the 1-D fluxes.', { showIf: isCmp }),
      F('cmpMu', 'Gas viscosity', 'µPa·s', 13, 1, 200, 'Steam at 130 °C ≈ 13 µPa·s; the conductivity follows from a Prandtl number of 0.95.', { showIf: (v) => isCmp(v) && v.cmpVisc }),
    ] },
    { group: 'Models', tab: 'setup', help: 'Governing equations and closures.', fields: [
      SEL('mode', 'Time treatment', 'steady', [['steady', 'Steady state (SIMPLEC iterations)'], ['transient', 'Transient (implicit Euler, vortex shedding)']], 'Use transient when the steady solution will not converge because the wake sheds vortices.'),
      SEL('scheme', 'Convection scheme', 'hybrid', [['hybrid', 'Hybrid central/upwind (robust)'], ['upwind', 'First-order upwind'], ['quick', 'Bounded QUICK (higher order)']], 'Applied to momentum and scalars.'),
      SEL('turb', 'Turbulence', 'laminar', [['laminar', 'Laminar'], ['ml', 'RANS: mixing-length eddy viscosity + wall function'], ['ke', 'RANS: standard k–ε with wall functions'], ['kw', 'RANS: k–ω (Wilcox)'], ['sst', 'RANS: k–ω SST (Menter)'], ['earsm', 'RANS: explicit algebraic Reynolds stress on k–ω'], ['rsm', 'RANS: Reynolds-stress transport equations (differential model, with ε)'], ['les', 'LES: Smagorinsky sub-grid model (transient, 2-D)']], 'Spacer channels are laminar or transitional; use a RANS closure for ducts above Re ≈ 3000. k–ε and the Reynolds-stress transport model need the first cell at y⁺ > 11.6 (wall refinement 1–3); k–ω and SST also integrate to the wall. The explicit algebraic option solves the stress equations algebraically on k–ω; the Reynolds-stress transport option solves a transport equation for each of u′u′, v′v′, w′w′ and u′v′ (Launder–Reece–Rodi pressure–strain with wall reflection) and for ε, and the momentum equations are driven by those stresses. LES in 2-D lacks vortex stretching and is indicative only.'),
      F('tuIn', 'Inlet turbulence intensity', '%', 5, 0.1, 30, 'Sets the inlet and initial turbulent kinetic energy k = 1.5 (Tu·U)².', { showIf: twoEqSel }),
      F('lTurb', 'Inlet turbulence length scale', '% of d_h', 7, 0.5, 50, 'Sets the inlet and initial ε = C_μ^¾ k^1.5/ℓ and ω = √k/(C_μ^¼ ℓ).', { showIf: twoEqSel }),
      F('cSmag', 'Smagorinsky constant C_s', '–', 0.17, 0.05, 0.3, 'Sub-grid viscosity ν_sgs = (C_s Δ)²|S|; 0.1 for shear flows, 0.17 for isotropic turbulence.', { showIf: (v) => v.turb === 'les' }),
      BOOL('creeping', 'Creeping flow (Stokes limit: drop the convective terms)', 'Valid for Reynolds numbers well below 1, e.g. flow through fine porous structures and micro-channels.'),
      SEL('engine', 'Flow engine', 'fv', [['fv', 'Finite volume (SIMPLE)'], ['lbm', 'Hybrid: lattice-Boltzmann (D2Q9) start + finite-volume solve']], 'The hybrid first solves the flow with a lattice-Boltzmann method on a uniform lattice, hands that field to the finite-volume solver and compares the two. Laminar, no-slip cases only.'),
      F('lbmNy', 'Lattice nodes across the gap', '', 20, 8, 96, 'The lattice is square, so the node count along the channel follows from the aspect ratio (capped at 60 000 nodes).', { showIf: (v) => v.engine === 'lbm', step: 1 }),
      SEL('species', 'Salt transport and wall condition', 'membrane', [['membrane', 'Membrane walls (permeation, polarisation)'], ['fixed', 'Fixed wall concentration (mass-transfer analogue)'], ['flux', 'Specified wall salt flux'], ['off', 'No species equation']], ''),
      F('jwSalt', 'Wall salt flux into the fluid', 'mg/m²·s', 5, -1e4, 1e4, 'Positive values add salt at the wall (dissolution, ion-exchange release); negative values remove it.', { showIf: (v) => v.species === 'flux' }),
      F('cwFixed', 'Wall concentration', 'g/L', 70, 0, 400, '', { showIf: (v) => v.species === 'fixed' }),
      SEL('sides', 'Walls carrying the membrane / fixed concentration', 'both', [['both', 'Both walls'], ['bottom', 'Bottom wall'], ['top', 'Top wall']], 'Spiral-wound feed channels have membrane on both sides; flat-sheet test cells on one.', { showIf: hasSpecies }),
    ] },
    { group: 'Boundary and initial conditions', tab: 'setup', help: 'Walls and filaments are no-slip. The outlet is a zero-gradient pressure outlet. The initial field is plug flow at the inlet concentration and temperature.', fields: [
      SEL('inlet', 'Inlet velocity profile', 'parabolic', [['parabolic', 'Parabolic (fully developed laminar)'], ['uniform', 'Uniform (developing flow)'], ['periodic', 'Periodic: profile recycled from the outlet']], 'The periodic option reproduces the repeating flow deep inside a spacer-filled channel.'),
      SEL('inletBC', 'Inlet condition', 'velocity', [['velocity', 'Velocity inlet (mean velocity above)'], ['massflow', 'Mass-flow inlet'], ['pressure', 'Pressure inlet (flow rate is a result)']], 'With a pressure inlet the solver adjusts the flow rate until the inlet gauge pressure equals the target (steady runs).'),
      F('mdot', 'Mass flow per metre of channel width', 'kg/s·m', 0.073, 1e-6, 1e4, 'The 2-D section is one metre wide: ṁ = ρ·U·(open inlet height).', { showIf: (v) => v.inletBC === 'massflow' }),
      F('pInlet', 'Inlet gauge pressure (relative to the outlet)', 'Pa', 60, 0.001, 1e7, 'Static pressure at the inlet plane above the outlet pressure.', { showIf: (v) => v.inletBC === 'pressure' }),
      SEL('wallType', 'Channel-wall momentum condition', 'noslip', [['noslip', 'No-slip walls'], ['slip', 'Navier-slip walls (slip length)'], ['symtop', 'Symmetry plane at the top (half channel)'], ['symboth', 'Symmetry planes top and bottom (free-slip)']], 'Navier slip: u_wall = b ∂u/∂y, for hydrophobic or structured surfaces. A symmetry plane halves the domain of a symmetric channel or removes wall friction altogether. Obstacles stay no-slip.'),
      F('slipLen', 'Slip length b', 'µm', 1, 0, 1e5, 'Typically 0.01–1 µm for hydrophobic walls, tens of µm for super-hydrophobic textures.', { showIf: (v) => v.wallType === 'slip' }),
    ] },
    { group: 'Energy equation', tab: 'setup', help: 'Temperature field with prescribed wall temperature, heat flux or a convective (membrane-distillation style) wall.', fields: [
      { key: 'energy', label: 'Solve the energy equation', type: 'bool', value: false },
      SEL('thWall', 'Thermal wall condition', 'conv', [['conv', 'Convective: q = U (T_ext − T_wall)'], ['fixed', 'Prescribed wall temperature'], ['flux', 'Prescribed heat flux'], ['none', 'Adiabatic']], '', { showIf: (v) => v.energy }),
      SEL('thSides', 'Thermally active walls', 'both', [['both', 'Both walls'], ['bottom', 'Bottom wall'], ['top', 'Top wall']], '', { showIf: (v) => v.energy }),
      F('Tw', 'Wall temperature', '°C', 20, 0, 150, '', { showIf: (v) => v.energy && v.thWall === 'fixed' }),
      F('qw', 'Wall heat flux into the fluid', 'W/m²', -15000, -1e6, 1e6, 'Negative values cool the fluid (heat leaving through a distillation membrane).', { showIf: (v) => v.energy && v.thWall === 'flux' }),
      F('Uw', 'Wall heat-transfer coefficient U', 'W/m²·K', 800, 1, 1e5, 'Membrane plus permeate-side resistance.', { showIf: (v) => v.energy && v.thWall === 'conv' }),
      F('Text', 'External (permeate-side) temperature', '°C', 20, 0, 150, '', { showIf: (v) => v.energy && v.thWall === 'conv' }),
      BOOL('cht', 'Conjugate heat transfer: conduct heat through the solids', 'Filaments, baffles and imported solids take part in the energy equation with their own conductivity instead of being adiabatic.', { showIf: (v) => v.energy }),
      F('kSolid', 'Solid thermal conductivity', 'W/m·K', 0.22, 0.01, 500, 'Polypropylene 0.22, PVDF 0.19, stainless steel 16, titanium 22, copper 390.', { showIf: (v) => v.energy && v.cht }),
      F('qSolid', 'Heat release inside the solids', 'W/m³', 0, -1e10, 1e10, 'For electrically heated spacers or heating tubes; 0 for passive solids.', { showIf: (v) => v.energy && v.cht }),
    ] },
    { group: 'Porous zone', tab: 'setup', help: 'Darcy–Forchheimer momentum sink −(μ/K + ρ c_F |u|/√K) u inside a slab of the channel; viscous (Brinkman) stresses are retained.', fields: [
      { key: 'porous', label: 'Add a porous zone', type: 'bool', value: false },
      SEL('porModel', 'Resistance law', 'kcf', [['kcf', 'Darcy–Forchheimer: enter K and c_F'], ['ergun', 'Ergun equation: packed bed of particles']], 'Ergun: K = d_p² ε³/(150(1 − ε)²) and c_F = 1.75/√(150 ε³), i.e. −dp/dx = 150 μ(1−ε)²U/(d_p²ε³) + 1.75 ρ(1−ε)U²/(d_p ε³).', { showIf: (v) => v.porous }),
      F('porDp', 'Bed particle diameter', 'mm', 0.5, 0.001, 100, 'Sauter mean diameter of the packing (media filter grain, resin bead).', { showIf: (v) => v.porous && v.porModel === 'ergun' }),
      F('porEps', 'Bed porosity ε', '–', 0.4, 0.05, 0.95, 'Void fraction; 0.36–0.42 for randomly packed spheres.', { showIf: (v) => v.porous && v.porModel === 'ergun' }),
      F('porX0', 'Zone start', '% of length', 40, 0, 100, '', { showIf: (v) => v.porous }),
      F('porX1', 'Zone end', '% of length', 70, 0, 100, '', { showIf: (v) => v.porous }),
      F('porK', 'Permeability K', 'm²', 1e-9, 1e-16, 1e-2, 'Packed 1 mm beads ≈ 1e-9 m²; fine filter media 1e-12 m².', { showIf: (v) => v.porous && v.porModel !== 'ergun' }),
      F('porCF', 'Forchheimer coefficient c_F', '–', 0.55, 0, 10, 'Ergun-type inertial coefficient (≈ 0.55 for packed beds).', { showIf: (v) => v.porous && v.porModel !== 'ergun' }),
    ] },
    { group: 'Particle tracking', tab: 'setup', help: 'Lagrangian particles released across the inlet: Stokes drag, gravity settling (−y), Brownian motion and capture on walls and obstacles.', fields: [
      { key: 'particles', label: 'Track particles', type: 'bool', value: false },
      F('nPart', 'Number of particles', '', 300, 10, 3000, '', { showIf: (v) => v.particles, step: 1 }),
      F('dPart', 'Particle diameter', 'µm', 5, 0.01, 2000, '', { showIf: (v) => v.particles }),
      F('rhoPart', 'Particle density', 'kg/m³', 1500, 500, 8000, '', { showIf: (v) => v.particles }),
      F('stick', 'Attachment efficiency', '–', 1, 0, 1, 'Probability that a particle touching a surface stays attached.', { showIf: (v) => v.particles }),
    ] },
    { group: 'Multiphase flow: free interface (volume of fluid / level set)', tab: 'setup', showIf: isTP, help: 'Two immiscible incompressible fluids in a closed two-dimensional tank, solved as one fluid with variable density and viscosity: ∇·u = 0, ρ(α) Du/Dt = −∇p + ∇·[μ(α)(∇u + ∇uᵀ)] + ρ(α) g + σ κ ∇α. The interface moves with the computed flow and acts back on it through the density, the viscosity and the surface tension (two-way coupling). Volume of fluid transports the volume fraction with the THINC/WLIC scheme and conserves the phase volume to round-off; the level set transports a signed distance, is redistanced every step and corrected so that the phase volume is restored.', fields: [
      SEL('tpDomain', 'Domain', 'tank', [['tank', 'Closed rectangular tank'], ['spacer', 'Spacer-filled membrane channel (Geometry inputs), through-flow'], ['import', 'Imported CAD section in a channel (Geometry inputs), through-flow']], 'The channel domains take the gap, the filaments (or the imported outline) and the length from the Geometry group on the Inputs tab — the same solid mask as the channel study — with a liquid inlet at x = 0 and a pressure outlet at x = L: an air slug or bubbles are carried through by the cross-flow (air sparging against fouling).'),
      SEL('tpMethod', 'Interface method', 'vof', [['vof', 'Volume of fluid (THINC/WLIC, conservative)'], ['ls', 'Level set (redistancing + volume correction)']], 'Both methods drive the same momentum and pressure equations.'),
      SEL('tpChInit', 'Gas at the start', 'slug', [['slug', 'Slug filling the gap over a length'], ['bubbles', 'Train of bubbles']], 'The gas (second phase) starts inside the channel near the inlet; the inlet itself delivers liquid.', { showIf: tpCh }),
      F('tpUin', 'Mean inlet velocity of the liquid', 'm/s', 0.3, 1e-4, 20, 'Cross-flow velocity over the open inlet height (parabolic profile). The capillary time step does not depend on it, so faster flows need fewer steps per flow-through time.', { showIf: tpCh }),
      F('tpChX', 'Upstream end of the gas', '% of length', 8, 0, 90, 'Upstream face of the slug, or of the first bubble.', { showIf: tpCh }),
      F('tpChLen', 'Slug length', '% of length', 20, 2, 80, '', { showIf: (v) => tpCh(v) && v.tpChInit !== 'bubbles' }),
      F('tpChD', 'Bubble diameter', '% of gap', 60, 10, 95, '', { showIf: (v) => tpCh(v) && v.tpChInit === 'bubbles' }),
      F('tpChN', 'Number of bubbles', '', 3, 1, 12, 'Spaced 1.6 diameters apart, alternately above and below mid-gap.', { showIf: (v) => tpCh(v) && v.tpChInit === 'bubbles', step: 1 }),
      F('tpChTime', 'Simulated time (channel)', 'flow-through times', 0.8, 0.02, 20, 'One flow-through time = channel length ÷ mean velocity.', { showIf: tpCh }),
      SEL('tpCase', 'Problem', 'dam', [['dam', 'Collapse of a liquid column (dam break)'], ['bubble', 'Rising bubble or drop'], ['slosh', 'Sloshing of a free surface'], ['rt', 'Rayleigh–Taylor overturning (heavy over light)']], 'Initial arrangement of the two phases in the tank.', { showIf: tpTank }),
      F('tpW', 'Tank width', 'mm', 400, 1, 1e5, 'Closed box; x is horizontal.', { showIf: tpTank }),
      F('tpH', 'Tank height', 'mm', 300, 1, 1e5, 'Gravity acts along −y.', { showIf: tpTank }),
      F('tpColW', 'Liquid column: width a', '% of tank width', 25, 5, 90, 'The column stands against the left wall.', { showIf: tpIs('dam') }),
      F('tpColH', 'Liquid column: height', '% of tank height', 66.6667, 5, 100, 'A column twice as high as wide is the Martin & Moyce experiment (n² = 2).', { showIf: tpIs('dam') }),
      F('tpBubD', 'Bubble diameter', '% of tank width', 50, 5, 90, 'Initial circle of the lighter phase, centred across the width.', { showIf: tpIs('bubble') }),
      F('tpBubY', 'Bubble centre height', '% of tank height', 25, 5, 95, '', { showIf: tpIs('bubble') }),
      F('tpFill', 'Still level of the interface', '% of tank height', 50, 5, 95, 'Liquid depth (sloshing) or height of the light layer under the heavy one (Rayleigh–Taylor).', { showIf: tpIs('slosh', 'rt') }),
      F('tpAmp', 'Initial amplitude of the interface', '% of tank height', 2, 0, 40, 'Cosine displacement: first sloshing mode (half a wave across the tank), or one full wave for Rayleigh–Taylor.', { showIf: tpIs('slosh', 'rt') }),
      F('tpRho1', 'Liquid (heavier phase): density', 'kg/m³', 1000, 0.01, 30000, ''),
      F('tpMu1', 'Liquid (heavier phase): viscosity', 'mPa·s', 1, 1e-4, 1e7, ''),
      F('tpRho2', 'Gas or lighter liquid: density', 'kg/m³', 1.2, 0.01, 30000, 'Air 1.2, steam at 1 bar 0.6, oil 850.'),
      F('tpMu2', 'Gas or lighter liquid: viscosity', 'mPa·s', 0.018, 1e-4, 1e7, 'Air 0.018, steam 0.012.'),
      F('tpSigma', 'Surface tension', 'mN/m', 72, 0, 1e5, 'Water–air 72, seawater–air 74, oil–water 20–50. 0 switches the capillary force off.'),
      F('tpG', 'Gravity', 'm/s²', 9.80665, 0.01, 100, ''),
      F('tpTime', 'Simulated time', '× √(ℓ/g)', 2.2, 0.01, 200, 'ℓ = column width (dam break), bubble diameter, or tank width (sloshing, Rayleigh–Taylor). The first sloshing period of a half-filled tank is about 2.6 √(W/g).', { showIf: tpTank }),
      { key: 'tpSlip', label: 'Free-slip walls (off: no-slip)', type: 'bool', value: true, showIf: tpTank, help: 'Free slip is the usual choice for inertia-dominated tank problems on grids that do not resolve the wall boundary layers.' },
    ] },
    { group: 'Multiphase flow: Eulerian–Eulerian two-fluid model', tab: 'setup', showIf: isTF, help: 'A continuous liquid and a dispersed phase (particles, drops or bubbles much smaller than a cell) as interpenetrating continua: each phase has its own continuity and momentum equation, both share one pressure, and they exchange momentum through drag (Schiller–Naumann with the Wen–Yu swarm correction), virtual mass and turbulent dispersion. The dispersed fraction is limited by the packing fraction, so a sediment or foam layer builds up on the wall the phase moves toward.', fields: [
      SEL('tfDomain', 'Domain', 'tank', [['tank', 'Rectangular column or channel (dimensions below)'], ['spacer', 'Spacer-filled membrane channel (Geometry inputs), through-flow'], ['import', 'Imported CAD section in a channel (Geometry inputs), through-flow']], 'The channel domains take the gap, the filaments (or the imported outline) and the length from the Geometry group on the Inputs tab — the same solid mask as the channel study — and feed the suspension at x = 0: particles or droplets are carried through the spacer-filled channel and deposit on the walls and on the filaments.'),
      SEL('tfCase', 'Problem', 'batch', [['batch', 'Closed column: batch settling or flotation'], ['flow', 'Flow-through channel: settler or flotation channel with an inlet']], 'The closed column starts with a uniform suspension; the channel is fed with it at x = 0 and has a pressure outlet at x = L.', { showIf: tfTank }),
      F('tfL', 'Domain width (along x)', 'mm', 50, 1, 1e5, 'Channel length for the flow-through case.', { showIf: tfTank }),
      F('tfH', 'Domain height', 'mm', 200, 1, 1e5, 'Gravity acts along −y.', { showIf: tfTank }),
      F('tfRhoC', 'Continuous phase: density', 'kg/m³', 1000, 0.01, 30000, ''),
      F('tfMuC', 'Continuous phase: viscosity', 'mPa·s', 1, 1e-4, 1e7, ''),
      F('tfRhoD', 'Dispersed phase: density', 'kg/m³', 2500, 0.01, 30000, 'Heavier than the liquid settles, lighter (bubbles, oil drops) rises. Sand 2650, gypsum crystals 2320, air 1.2.'),
      F('tfDp', 'Dispersed phase: particle / bubble diameter', 'µm', 100, 0.01, 2e4, 'Must be small against the cell size.'),
      F('tfAlpha', 'Dispersed-phase volume fraction (initial and inlet)', '%', 10, 1e-4, 60, ''),
      F('tfU', 'Inlet velocity', 'm/s', 0.02, 1e-5, 20, 'Both phases enter at this velocity (mean over the open inlet height in a channel geometry).', { showIf: (v) => tfCh(v) || (isTF(v) && v.tfCase === 'flow') }),
      { key: 'tfDeposit', label: 'Capture the dispersed phase on the wall it reaches', type: 'bool', value: true, showIf: (v) => tfCh(v) || (isTF(v) && v.tfCase === 'flow'), help: 'In a channel geometry: capture on the wall and on the solid faces the phase settles (or rises) onto. Sludge hopper or foam skimmer: the phase leaves through the wall and the same volume of liquid takes its place. Off: it accumulates as a moving layer.' },
      F('tfTimeB', 'Simulated time (closed column)', '× H ÷ hindered velocity', 0.5, 0.02, 5, '1 = the time the suspension front needs to cross the column.', { showIf: (v) => tfTank(v) && v.tfCase !== 'flow' }),
      F('tfTimeF', 'Simulated time (flow-through)', 'flow-through times', 3, 0.05, 50, 'One flow-through time = length ÷ inlet velocity; about three are needed for a steady capture efficiency.', { showIf: (v) => tfCh(v) || (isTF(v) && v.tfCase === 'flow') }),
      F('tfCvm', 'Virtual-mass coefficient C_vm', '–', 0.5, 0, 2, '0.5 for spheres; matters for bubbles, negligible for heavy particles.'),
      F('tfDtd', 'Turbulent-dispersion diffusivity', 'm²/s', 0, 0, 1e-2, 'Spreads the dispersed phase down its concentration gradient; 0 for laminar suspensions.'),
      F('tfAmax', 'Packing limit of the dispersed phase', '–', 0.6, 0.3, 0.74, 'Random loose packing of spheres ≈ 0.6.'),
      F('tfG', 'Gravity', 'm/s²', 9.80665, 0.01, 100, ''),
      { key: 'tfSlip', label: 'Free-slip walls for the continuous phase', type: 'bool', value: false, help: 'The dispersed phase always slips along the walls.' },
    ] },
    { group: 'Second phase (one-way transport)', tab: 'setup', help: 'A second phase transported on the solved velocity field: a sharp interface (gas bubble, air slug or displacing liquid) by volume-of-fluid or level-set, or a dilute dispersed phase (particles, droplets, micro-bubbles) by a drift-flux continuity equation with an algebraic slip velocity. One-way coupling: the second phase does not alter the flow and surface tension is not included — for the coupled problem (interface with density jump and surface tension, or a dispersed phase with its own momentum equation) choose one of the two multiphase studies under Study type.', fields: [
      SEL('mp', 'Second-phase model', 'off', [['off', 'None'], ['vof', 'Volume of fluid (sharp interface, conservative)'], ['ls', 'Level set (signed-distance interface)'], ['ee', 'Dispersed phase: drift-flux transport with slip velocity']], 'Volume of fluid conserves the phase volume to round-off; the level set gives smooth interface geometry but loses or gains a little area.'),
      SEL('mpInit', 'Initial phase distribution', 'slug', [['bubble', 'Circular bubble / drop'], ['slug', 'Slug filling the gap over a length']], 'Initial phase fraction field: 1 inside the region, 0 elsewhere.', { showIf: (v) => v.mp === 'vof' || v.mp === 'ls' }),
      F('mpD', 'Bubble diameter', '% of gap', 45, 5, 95, 'Diameter of the initial circular region.', { showIf: (v) => (v.mp === 'vof' || v.mp === 'ls') && v.mpInit === 'bubble' }),
      F('mpX', 'Initial position along the channel', '% of length', 12, 0, 95, 'Bubble centre or upstream face of the slug.', { showIf: (v) => v.mp === 'vof' || v.mp === 'ls' }),
      F('mpY', 'Bubble position across the gap', '% of gap', 50, 0, 100, 'Centre of the bubble.', { showIf: (v) => (v.mp === 'vof' || v.mp === 'ls') && v.mpInit === 'bubble' }),
      F('mpLen', 'Slug length', '% of length', 10, 1, 90, 'Streamwise extent of the slug.', { showIf: (v) => (v.mp === 'vof' || v.mp === 'ls') && v.mpInit === 'slug' }),
      F('mpAlphaIn', 'Dispersed-phase volume fraction at the inlet', '%', 0.5, 0, 30, 'Inlet phase fraction of the dispersed phase; the domain starts clean.', { showIf: (v) => v.mp === 'ee' }),
      F('mpDp', 'Dispersed particle / bubble diameter', 'µm', 20, 0.01, 5000, 'Sets the Stokes slip velocity (ρ_d − ρ) g d²/(18 μ), hindered by (1 − α)^4.65.', { showIf: (v) => v.mp === 'ee' }),
      F('mpRho', 'Dispersed-phase density', 'kg/m³', 2650, 0.1, 20000, 'Heavier than the liquid settles to the bottom wall, lighter (bubbles, oil) rises to the top wall.', { showIf: (v) => v.mp === 'ee' }),
      F('mpDisp', 'Dispersion coefficient of the dispersed phase', 'm²/s', 0, 0, 1e-2, 'Shear-induced or turbulent dispersion across the gap; 0 for pure advection and slip.', { showIf: (v) => v.mp === 'ee' }),
      F('mpTime', 'Simulated time', 'flow-through times', 0.6, 0.01, 50, 'One flow-through time = length ÷ mean velocity. Use 3 or more for the dispersed phase to reach a steady distribution.', { showIf: (v) => v.mp && v.mp !== 'off' }),
    ] },
    { group: 'Precipitation, crystallisation and fouling', tab: 'setup', help: 'A sparingly soluble salt (gypsum, calcium carbonate, silica …) transported with the flow, crystallising on the walls and in the bulk; and a deposit layer that grows in time and adds hydraulic resistance to the membrane.', fields: [
      BOOL('precip', 'Transport a scaling salt with wall crystallisation', 'Solves a second species that the membrane rejects completely; where its wall concentration exceeds the solubility it deposits at N = k_r (c_wall − c_sat).'),
      F('scC0', 'Scaling-salt concentration in the feed', 'g/L', 2.0, 0, 200, 'For example calcium sulphate expressed as CaSO₄.', { showIf: (v) => v.precip }),
      F('scSat', 'Solubility of the scaling salt', 'g/L', 2.1, 1e-6, 500, 'Saturation concentration at the operating temperature and ionic strength (from the chemistry suite).', { showIf: (v) => v.precip }),
      F('scD', 'Diffusivity of the scaling salt', '10⁻⁹ m²/s', 0.9, 0.001, 100, 'Mutual diffusivity of the salt in water.', { showIf: (v) => v.precip }),
      F('scKr', 'Surface crystallisation rate constant k_r', 'µm/s', 0.5, 0, 1e5, 'First-order surface-integration constant; large values make the deposition diffusion-controlled.', { showIf: (v) => v.precip }),
      F('scRho', 'Crystal density', 'kg/m³', 2320, 500, 8000, 'Gypsum 2320, calcite 2710, amorphous silica 2200.', { showIf: (v) => v.precip }),
      BOOL('pbm', 'Bulk crystals: population balance (method of moments)', 'Transports the moments m₀…m₃ of the crystal size distribution with nucleation B = k_n (S − 1)^n and growth G = k_g (S − 1).', { showIf: (v) => v.precip }),
      F('pbKn', 'Nucleation rate constant k_n', '1/m³·s', 1e10, 0, 1e30, 'Primary nucleation rate at S − 1 = 1.', { showIf: (v) => v.precip && v.pbm }),
      F('pbN', 'Nucleation order n', '–', 2, 0.5, 6, 'Exponent on the relative supersaturation.', { showIf: (v) => v.precip && v.pbm }),
      F('pbKg', 'Growth rate constant k_g', 'µm/s', 0.05, 0, 1e4, 'Linear crystal growth rate at S − 1 = 1.', { showIf: (v) => v.precip && v.pbm }),
      F('pbSeedN', 'Seed crystals in the feed: number density', '1/m³', 0, 0, 1e30, 'Initial and inlet particle distribution of the crystal phase (mono-sized seeds).', { showIf: (v) => v.precip && v.pbm }),
      F('pbSeedD', 'Seed crystal size', 'µm', 1, 0.001, 1000, 'Diameter of the seed crystals.', { showIf: (v) => v.precip && v.pbm }),
      BOOL('foul', 'Grow a fouling layer on the membrane in time', 'Quasi-steady march: after each time step the deposit resistance is updated and flow, polarisation and precipitation are solved again.', { showIf: (v) => v.species === 'membrane' }),
      F('foulC', 'Foulant (colloid) concentration in the feed', 'mg/L', 5, 0, 1e5, 'Particulate matter carried to the membrane by the permeate flow.', { showIf: (v) => v.foul && v.species === 'membrane' }),
      F('foulAlpha', 'Specific resistance of the deposit α', 'm/kg', 1e15, 1e10, 1e18, 'Colloidal cakes 1e14–1e16 m/kg; biofilm up to 1e17.', { showIf: (v) => v.foul && v.species === 'membrane' }),
      F('foulBack', 'Shear back-transport coefficient k_b', 'µm/s per Pa', 2, 0, 1e4, 'Deposition occurs only where the local flux exceeds k_b·|τ_w| (critical-flux concept).', { showIf: (v) => v.foul && v.species === 'membrane' }),
      F('foulTime', 'Fouling period', 'h', 48, 0.01, 1e5, 'Total operating time simulated.', { showIf: (v) => v.foul && v.species === 'membrane' }),
      F('foulSteps', 'Number of fouling time steps', '', 6, 1, 40, 'Each step re-solves the coupled flow and salt fields.', { showIf: (v) => v.foul && v.species === 'membrane', step: 1 }),
      F('foulM0', 'Initial deposit on the membrane', 'g/m²', 0, 0, 1e4, 'Deposited material present at the start (uniform).', { showIf: (v) => v.foul && v.species === 'membrane' }),
      F('foulRho', 'Deposit bulk density', 'kg/m³', 1200, 50, 8000, 'Converts deposit mass per area into layer thickness.', { showIf: (v) => v.foul && v.species === 'membrane' }),
    ] },
    { group: 'Multicomponent diffusion and user-defined model', tab: 'setup', help: 'Optional post-models evaluated with the solved fields.', fields: [
      BOOL('ms', 'Maxwell–Stefan film: second solute across the polarisation layer', 'Integrates the ternary Maxwell–Stefan equations (NaCl, a second solute, water) across the boundary-layer film whose thickness comes from the CFD mass-transfer coefficient, and compares with independent Fick films.'),
      F('msC2', 'Second solute: bulk concentration', 'g/L', 2.5, 1e-6, 300, 'For example MgSO₄ in seawater ≈ 2–3 g/L.', { showIf: (v) => v.ms }),
      F('msM2', 'Second solute: molar mass', 'g/mol', 120.4, 1, 5000, 'MgSO₄ 120.4, CaCl₂ 111, boric acid 61.8.', { showIf: (v) => v.ms }),
      F('msD23', 'Maxwell–Stefan diffusivity solute 2 – water', '10⁻⁹ m²/s', 0.85, 0.001, 100, 'Equals the binary Fick diffusivity at infinite dilution.', { showIf: (v) => v.ms }),
      F('msD12', 'Maxwell–Stefan diffusivity NaCl – solute 2', '10⁻⁹ m²/s', 0.4, 0.001, 1000, 'Solute–solute friction; a very large value recovers two independent Fick films.', { showIf: (v) => v.ms }),
      F('msR2', 'Membrane rejection of solute 2', '%', 99.8, 0, 100, 'Intrinsic rejection at the membrane surface.', { showIf: (v) => v.ms }),
      BOOL('usr', 'User-defined scalar with a source expression', 'Transports an extra scalar φ with your own volumetric source term, e.g. a reacting biocide, a decaying tracer or residence-time age (source = 1).'),
      { key: 'usrSrc', label: 'Source term S(phi, c, T, x, y, u, v)', type: 'text', value: '-0.5*phi', showIf: (v) => v.usr, help: 'Units of φ per second. Operators + − * / ^ and exp, log, sqrt, abs, min, max, pow, tanh, sin, cos, step. phi = the scalar, c = salt (g/L), T = temperature (°C), x, y in m, u, v in m/s. Examples: "-0.5*phi" (first-order decay), "1" (fluid age), "-2e-3*phi*c".' },
      F('usrIn', 'Inlet (and initial) value of the scalar', 'user units', 1, -1e12, 1e12, 'Value carried in by the feed.', { showIf: (v) => v.usr }),
      F('usrD', 'Diffusivity of the scalar', '10⁻⁹ m²/s', 1, 0, 1e6, 'Molecular diffusivity; the turbulent part is added automatically.', { showIf: (v) => v.usr }),
      SEL('usrWall', 'Wall condition of the scalar', 'none', [['none', 'Zero flux'], ['fixed', 'Fixed wall value']], 'Applied on the walls that carry the salt condition.', { showIf: (v) => v.usr }),
      F('usrWallVal', 'Wall value of the scalar', 'user units', 0, -1e12, 1e12, 'Used with the fixed-value wall.', { showIf: (v) => v.usr && v.usrWall === 'fixed' }),
      BOOL('ml', 'Train a regression closure for f and Sh on extra solver runs', 'Runs the solver at five velocities (and two pitches for spacers) on a coarser grid, fits power laws f(Re, pitch/H) and Sh(Re, pitch/H) by ridge regression with leave-one-out cross-validation, and applies them in the 1-D model. Adds 5–10 solver runs.'),
    ] },
    { group: '1-D reference model and limits', tab: 'setup', help: 'Multipliers on the friction and Sherwood correlations of the 1-D channel model used for comparison and calibration, and the limits checked in the results.', fields: [
      F('kdp', 'Friction multiplier (1-D model)', '×', 1, 0.1, 20, 'Fitted on the Calibrate tab against measured pressure drop.'),
      F('ksh', 'Mass-transfer multiplier (1-D model)', '×', 1, 0.1, 10, 'Fitted on the Calibrate tab against measured flux.'),
      F('limCP', 'Maximum polarisation factor', '–', 1.2, 1.02, 3, 'Common membrane-manufacturer guideline.'),
      F('limTau', 'Minimum wall shear for fouling control', 'Pa', 0.2, 0, 100, 'Wall regions below this shear are reported as deposition-prone.'),
      F('limDP', 'Maximum pressure gradient', 'bar/m', 0.6, 0.01, 100, 'About 0.6 bar/m corresponds to 4 bar over a seven-element vessel.'),
    ] },
    { group: 'Grid', tab: 'mesh', help: 'Structured Cartesian grid: uniform in x, clustered toward both walls in y so that the thin concentration boundary layer (a few micrometres at Sc ≈ 600) is resolved.', fields: [
      F('nx', 'Cells along the flow, nx', '', 128, 12, 480, 'At least 6 cells across a filament diameter.', { step: 1 }),
      F('ny', 'Cells across the gap, ny', '', 32, 6, 200, '', { step: 1 }),
      F('stretch', 'Wall refinement (centre ÷ wall cell height)', '×', 10, 1, 80, '1 = uniform grid. 8–20 is needed to resolve concentration polarisation; the results report the wall-cell height against the boundary-layer thickness.'),
    ] },
    { group: 'Grid of the multiphase studies', tab: 'mesh', showIf: (v) => isTP(v) || isTF(v), help: 'Uniform staggered grid. Free interface: use square cells and at least 12–16 cells across a bubble or the liquid column. Two-fluid model: refine along the direction of settling.', fields: [
      F('tpNx', 'Free interface: cells along x', '', 64, 16, 200, '', { step: 1, showIf: tpTank }),
      F('tpNy', 'Free interface: cells along y', '', 48, 16, 200, '', { step: 1, showIf: tpTank }),
      F('tpChNy', 'Free interface in a channel: cells across the gap', '', 12, 6, 64, 'The cells are square: the number along the flow follows from the channel length (at most 240). Surface tension limits the time step as (cell size)^1.5, so the cost rises steeply with this number.', { step: 1, showIf: tpCh }),
      F('tpCfl', 'Free interface: CFL number', '–', 0.25, 0.05, 0.5, 'Convective limit of the explicit time step; the capillary, viscous and gravity-wave limits are applied as well.', { showIf: isTP }),
      F('tfNx', 'Two-fluid model: cells along x', '', 10, 3, 200, '', { step: 1, showIf: tfTank }),
      F('tfNy', 'Two-fluid model: cells along y', '', 80, 8, 240, '', { step: 1, showIf: tfTank }),
      F('tfChNy', 'Two-fluid model in a channel: cells across the gap', '', 12, 6, 64, 'The cells are square: the number along the flow follows from the channel length (at most 200).', { step: 1, showIf: tfCh }),
    ] },
    { group: 'Solver controls', tab: 'mesh', fields: [
      F('maxIter', 'Maximum iterations (or time steps)', '', 600, 5, 6000, '', { step: 1 }),
      F('tol', 'Convergence tolerance', '–', 1e-5, 1e-9, 1e-2, 'On the normalised continuity residual and the velocity change per iteration.'),
      F('alphaU', 'Velocity under-relaxation', '–', 0.85, 0.2, 0.95, 'Lower it (0.5–0.7) if a steady run oscillates.'),
      F('cfl', 'CFL number (transient)', '–', 2, 0.2, 10, 'Time step = CFL × cell size ÷ local velocity.', { showIf: (v) => v.mode === 'transient' }),
      F('tFlow', 'Simulated time (transient)', 'flow-through times', 4, 0.2, 100, 'One flow-through time = length ÷ mean velocity. Statistics are averaged over the second half.', { showIf: (v) => v.mode === 'transient' }),
    ] },
  ],

  presets: [
    { name: 'Seawater RO feed channel, zigzag spacer, 0.1 m/s', values: {} },
    { name: 'Empty membrane channel — Lévêque / Hagen–Poiseuille benchmark', values: { geom: 'plain', L: 20, nx: 100, ny: 32, stretch: 14 } },
    { name: 'Vortex shedding behind submerged filaments (transient)', values: { arr: 'submerged', H: 2, df: 0.5, lm: 8, nFil: 1, Uin: 0.45, mode: 'transient', scheme: 'quick', species: 'off', nx: 96, ny: 32, stretch: 1, tFlow: 6, maxIter: 1000, cfl: 2.5 } },
    { name: 'Membrane-distillation feed channel with heat transfer', values: { arr: 'zigzag', H: 2, df: 1, lm: 12, nFil: 4, T: 60, Uin: 0.05, limTau: 0.05, species: 'off', energy: true, thWall: 'conv', Uw: 900, Text: 22, stretch: 6 } },
    { name: 'Turbulent brine duct with a heated wall, k–ω SST', values: { geom: 'plain', H: 50, L: 600, Uin: 1, turb: 'sst', species: 'off', inlet: 'uniform', stretch: 2, nx: 80, ny: 30, energy: true, thWall: 'fixed', Tw: 60, alphaU: 0.7, limTau: 0.05 } },
    { name: 'Steam-ejector motive nozzle (compressible, quasi-1-D)', values: { study: 'nozzle' } },
    { name: 'Gypsum scaling: wall crystallisation, crystal population and fouling layer', values: { precip: true, pbm: true, foul: true } },
    { name: 'Air slug displaced through the spacer channel (volume of fluid)', values: { mp: 'vof', species: 'off' } },
    { name: 'Turbulent brine duct, Reynolds-stress transport model', values: { geom: 'plain', H: 50, L: 600, Uin: 1, turb: 'rsm', species: 'off', inlet: 'uniform', stretch: 2, nx: 60, ny: 24, alphaU: 0.7, limTau: 0.05 } },
    { name: 'Multiphase: dam break of a water column (volume of fluid, Martin & Moyce)', values: { study: 'twophase', tpCase: 'dam', tpMethod: 'vof' } },
    { name: 'Multiphase: rising bubble (level set, fluid properties of the Hysing et al. benchmark, case 1)', values: { study: 'twophase', tpCase: 'bubble', tpMethod: 'ls', tpW: 1000, tpH: 2000, tpBubD: 50, tpBubY: 25, tpRho1: 1000, tpMu1: 10000, tpRho2: 100, tpMu2: 1000, tpSigma: 24500, tpG: 0.98, tpTime: 4.2, tpSlip: true, tpNx: 32, tpNy: 64 } },
    { name: 'Multiphase: air slug through the spacer-filled channel (free interface, sparging)', values: { study: 'twophase', tpDomain: 'spacer', nFil: 2, tpChInit: 'slug' } },
    { name: 'Multiphase: bubbles past an imported strand section (free interface, level set)', values: { study: 'twophase', tpDomain: 'import', tpMethod: 'ls', tpChInit: 'bubbles', tpChN: 2, tpChD: 45, H: 1, L: 6, cad: STRAND_CAD, cadSize: 45, cadX: 55, tpUin: 0.4, tpChTime: 0.7 } },
    { name: 'Multiphase: batch settling of a suspension (two-fluid model, Kynch front)', values: { study: 'twofluid', tfCase: 'batch' } },
    { name: 'Multiphase: particles carried through the spacer-filled channel (two-fluid model, deposition)', values: { study: 'twofluid', tfDomain: 'spacer', nFil: 2, tfDp: 20, tfAlpha: 0.5, tfU: 0.1, tfRhoD: 2650 } },
    { name: 'Multiphase: oil droplets past an imported strand section (two-fluid model)', values: { study: 'twofluid', tfDomain: 'import', H: 1, L: 6, cad: STRAND_CAD, cadSize: 45, cadX: 45, tfDp: 30, tfAlpha: 1, tfU: 0.05, tfRhoD: 850 } },
    { name: 'Multiphase: settling channel with wall capture (two-fluid model, Hazen)', values: { study: 'twofluid', tfCase: 'flow', tfL: 120, tfH: 20, tfDp: 50, tfAlpha: 0.01, tfU: 0.03, tfSlip: true, tfNx: 48, tfNy: 16 } },
    { name: 'Sudden expansion in a brine duct, turbulent, with particles', values: { geom: 'step', H: 100, L: 1400, Uin: 1.2, c0: 65, turb: 'ml', species: 'off', stretch: 3, nx: 120, ny: 30, particles: true, dPart: 60, rhoPart: 2650, alphaU: 0.7, maxIter: 900 } },
  ],

  pull: ({ feed, outputs } = {}) => [
    Number.isFinite(feed?.T) ? { key: 'T', value: feed.T, from: 'Case feed water' } : null,
    outputs?.ro?.streams?.feed?.tds ? { key: 'c0', value: outputs.ro.streams.feed.tds / 1000, from: 'RO design: feed TDS' } : null,
    outputs?.ro?.feedPressureBar ? { key: 'dPtm', value: outputs.ro.feedPressureBar - 1, from: 'RO design: feed pressure (1 bar permeate side)' } : null,
  ].filter(Boolean),
  site: (site) => (Number.isFinite(site?.data?.sst) ? [{ key: 'T', value: site.data.sst, from: 'Sea-surface temperature at site' }] : []),

  async run(v, ctx) {
    if (v.study === 'shock' || v.study === 'nozzle') return runCompressible(v);
    if (v.study === 'twophase') return runTwoPhase(v, ctx);
    if (v.study === 'twofluid') return runTwoFluid(v, ctx);
    if (!(v.Uin > 0) && (v.inletBC ?? 'velocity') === 'velocity') throw new Error('Enter a mean inlet velocity greater than zero — a membrane channel without cross-flow has no steady state.');
    const c = caseConfig(v);
    if (v.inletBC === 'massflow' && c.Uin > 100) throw new Error(`The mass flow of ${fmt(v.mdot, 4)} kg/s per metre of width through an open inlet height of ${fmt(c.openIn * 1e3, 3)} mm is a mean velocity of ${fmt(c.Uin, 3)} m/s — far outside the range of this incompressible-liquid model (the velocity inlet is limited to 20 m/s). Check the mass flow, the channel height and the fluid density.`);
    let lbm = null;
    if (v.engine === 'lbm') {
      if (c.o.turb || c.o.wallB !== 'noslip' || c.o.wallT !== 'noslip' || c.o.creeping || c.o.pInlet) lbm = { note: 'The lattice-Boltzmann start is available for laminar flow with no-slip walls and a velocity or mass-flow inlet; the finite-volume solver was used alone.' };
      else { ctx?.progress?.(0.01, 'Lattice-Boltzmann flow'); lbm = lbmStart(c, v); if (lbm.init) c.o.init = lbm.init; }
    }
    // a diverged solution is repeated once with the most robust numerics before the run is given up
    let r, robust = false;
    try { r = await solveChannel(c.o, ctx); }
    catch (e) {
      if (!e.diverged) throw e;
      robust = true; Object.assign(c.o, { scheme: 'upwind', alphaU: Math.min(c.o.alphaU, 0.4), cfl: 0.5 * Math.min(c.o.cfl, 1), inner: 4, init: undefined });
      if (c.o.pInlet > 0) { c.o.Uin = Math.max(0.01 * c.o.Uin, 1e-6); c.o.maxIter = Math.max(c.o.maxIter, 400); } // pressure inlet: approach the target from a low flow rate
      ctx?.progress?.(0.02, 'The solution diverged: repeating with first-order upwind and stronger under-relaxation');
      r = await solveChannel(c.o, ctx);
    }
    const q = post(c, r), W = [], { fl, L, H, nx, ny } = c, o = c.o, one = channel1D(v.inletBC && v.inletBC !== 'velocity' ? { ...v, Uin: r.Uref } : v);
    ctx?.progress?.(0.96, 'Post-processing');
    const mm = (a) => a.map((x) => x * 1e3), xm = mm(q.xc), ym = mm(q.yp), dy0 = r.dy[0];
    const speed = q.field(q.uc.map((u, P) => Math.hypot(u, q.vc[P])), 1), U = q.field(q.uc, 1, 0), V = q.field(q.vc, 1, 0);
    const shapes = c.mk.shapes.map((s) => ({ x: mm(s.x), y: mm(s.y), closed: s.closed, color: '#0f172a' }));
    const base = { type: 'field', xlabel: 'x (mm)', ylabel: 'y (mm)', x: xm, y: ym, mask: q.mask, equal: L / H <= 8, shapes };
    const Sh = (k) => (k === null ? null : (k * q.dh) / fl.D), turbulent = o.turb;
    const fHP = 96 / q.Re, fSM = 6.23 * q.Re ** -0.3, fBl = 0.316 * q.Re ** -0.25, shSM = 0.065 * q.Re ** 0.875 * q.Sc ** 0.25, kSM = (shSM * fl.D) / q.dh;
    const sp = q.sp, kDev = sp?.kDev ?? sp?.kAll ?? null, shDev = Sh(kDev), shAll = Sh(sp?.kAll ?? null);
    const kMult = kDev ? kDev / kSM : null, memb = v.species === 'membrane';
    const massErr = (r.Qin - (() => { let s = 0; for (let j = 0; j < ny; j++) s += r.u[j * r.nu1 + nx] * r.dy[j]; for (let i = 0; i < nx; i++) s += (r.v[ny * nx + i] - r.v[i]) * r.dx; return s; })()) / (r.Qin || 1);
    // --- warnings
    if (c.mk.note) W.push({ level: 'info', msg: c.mk.note });
    if (robust) W.push({ level: 'warn', msg: `The solution diverged with the chosen numerics and was repeated with first-order upwind convection, a velocity relaxation of ${fmt(o.alphaU, 2)}${o.steady ? '' : ` and a CFL number of ${fmt(o.cfl, 2)}`}${o.pInlet > 0 ? ', the inlet pressure being approached from a hundredth of the first flow-rate estimate' : ''}: the result is more diffusive than the scheme you selected would give — refine the grid to confirm it.` });
    if (!r.converged) W.push({ level: 'warn', msg: o.steady ? `The steady solution did not reach the tolerance in ${r.iters} iterations (continuity residual ${fmt(r.hist.mass.at(-1), 2)}). The flow is probably unsteady: use the transient mode, the hybrid scheme, a lower relaxation factor or more iterations.` : `The transient run stopped at ${fmt(r.time, 3)} s before the requested ${fmt(r.probe?.tEnd ?? o.tEnd, 3)} s — raise the maximum number of time steps.` });
    if (!turbulent && q.Re > 2500) W.push({ level: 'warn', msg: `Reynolds number ${fmt(q.Re, 3)} is above the laminar range; the laminar solution under-predicts friction and mixing. Switch on a RANS turbulence closure.` });
    if (turbulent) W.push({ level: 'info', msg: `Turbulence closure — ${TURB_NAMES[r.tm]}: friction velocity ${fmt(r.utau, 3)} m/s, first-cell y⁺ ≈ ${fmt((0.5 * dy0 * r.utau * fl.rho) / fl.mu, 3)}. The log-law wall function is used where y⁺ > 11.6, together with the scalar law of the wall (Jayatilleke) for salt and heat. Separated regions are only approximately represented by this closure.` });
    if (o.steady && q.recirc > 0.02 && q.uc.some((u, P) => P % nx === nx - 1 && u < -0.02 * r.Uref)) W.push({ level: 'warn', msg: 'Reverse flow reaches the outlet plane — lengthen the domain so that the recirculation closes inside it.' });
    if (sp && kDev && !(turbulent && (0.5 * dy0 * r.utau * fl.rho) / fl.mu > 11.6)) { const dc = fl.D / kDev; if (0.5 * dy0 > 0.35 * dc) W.push({ level: 'warn', msg: `The wall cell (${fmt(dy0 * 1e6, 3)} µm) is coarse relative to the concentration boundary layer (≈ ${fmt(dc * 1e6, 3)} µm): polarisation and Sherwood number are under-resolved. Increase the wall refinement or ny.` }); }
    if (memb && q.cpMean > v.limCP) W.push({ level: 'warn', msg: `Mean polarisation factor ${fmt(q.cpMean, 3)} exceeds the limit of ${v.limCP}; the highest wall concentration is ${fmt(q.cwMax, 3)} g/L.` });
    if (Math.abs(q.dpdx) / 1e5 > v.limDP) W.push({ level: 'bad', msg: `Pressure gradient ${fmt(q.dpdx / 1e5, 3)} bar/m exceeds the limit of ${v.limDP} bar/m.` });
    const lowShear = [...q.tauB, ...q.tauT].filter((t) => Math.abs(t) < v.limTau).length / (2 * nx);
    if (lowShear > 0.25) W.push({ level: 'info', msg: `${fmt(100 * lowShear, 3)} % of the wall length sees a shear stress below ${v.limTau} Pa — these wake and stagnation zones are the likely fouling sites.` });
    if (v.inlet === 'periodic' && memb) W.push({ level: 'info', msg: 'Periodic inlet: the velocity profile is recycled; the concentration field still develops from the inlet value.' });
    if (!W.some((w) => w.level !== 'info')) W.unshift({ level: 'info', msg: o.steady ? `Converged in ${r.iters} iterations; mass closes to ${fmt(Math.abs(massErr) * 100, 2)} %.` : `Transient run completed: ${fmt(r.time, 3)} s simulated in ${r.hist.it.length} time steps; wall quantities are averaged over the second half.` });
    // --- particles
    const pt = v.particles ? trackParticles(r, fl, { n: v.nPart, d: v.dPart * 1e-6, rho: v.rhoPart, stick: clamp(v.stick, 0, 1) }) : null;
    // --- thermal
    const th = q.th, nuDev = th?.kDev ? (th.kDev * q.dh) / o.energy.alpha : null, Pr = (fl.mu * fl.cp) / fl.k;
    const meanOf = (a, row) => { let s = 0, m = 0; a.forEach((x, i) => { if (!r.solid[row * nx + i]) { s += x; m++; } }); return m ? s / m : 0; };
    const tpc = th && v.thWall === 'conv' ? (() => { const tw = [], tb = mean(th.bulk); if (o.energy.bot.type === 'conv') tw.push(meanOf(th.B.w, 0)); if (o.energy.top.type === 'conv') tw.push(meanOf(th.T.w, ny - 1)); return tw.length && Math.abs(tb - v.Text) > 1e-9 ? (mean(tw) - v.Text) / (tb - v.Text) : null; })() : null;
    // --- plots
    const plots = [
      { ...base, title: 'Velocity magnitude and streamlines', zlabel: 'Speed', zunit: 'm/s', z: speed, u: U, v: V, stream: true, cmap: 'viridis', note: `Maximum speed ${fmt(q.umax, 3)} m/s (${fmt(q.umax / r.Uref, 3)} × mean).` },
      { ...base, title: 'Pressure', zlabel: 'Pressure relative to outlet', zunit: 'Pa', z: q.field(r.p), cmap: 'coolwarm', contours: 10 },
    ];
    if (sp) plots.push({ ...base, title: 'Salt concentration', zlabel: 'Concentration', zunit: 'g/L', z: q.field(r.spc.phi), cmap: 'salinity', note: 'The polarisation layer is only a few micrometres thick; see the wall-concentration profile for the membrane-surface values.' });
    if (th) plots.push({ ...base, title: 'Temperature', zlabel: 'Temperature', zunit: '°C', z: q.field(r.eng.phi), cmap: 'thermal', contours: 8 });
    if (turbulent) plots.push({ ...base, title: 'Eddy-viscosity ratio ν_t / ν', zlabel: 'ν_t/ν', zunit: '–', z: q.field(r.mue.map((m) => m / fl.mu - 1)), cmap: 'turbo' });
    plots.push({ type: 'line', title: 'Wall shear stress', xlabel: 'x (mm)', ylabel: 'τ_w (Pa)', zeroY: true, series: [{ name: 'Bottom wall', x: xm, y: q.tauB }, { name: 'Top wall', x: xm, y: q.tauT }], hlines: [{ y: (6 * fl.mu * r.Uref) / H, label: 'plane Poiseuille 6μU/H' }], note: 'Negative values mark reversed near-wall flow (recirculation).' });
    if (sp) {
      const xs = q.xc.map((x) => x / (q.dh * q.Re * q.Sc)), lev = xs.map((s) => Math.max(memb ? 8.235 : 7.541, (memb ? 1.4904 : 1.2326) * s ** (-1 / 3)));
      plots.push({ type: 'line', title: 'Local Sherwood number', xlabel: 'x (mm)', ylabel: 'Sh = k·d_h / D', logy: true, series: [{ name: 'Bottom wall', x: xm, y: sp.B.k.map(Sh) }, { name: 'Top wall', x: xm, y: sp.T.k.map(Sh) }, { name: `Lévêque / Graetz, empty channel (${memb ? 'flux' : 'concentration'} wall)`, x: xm, y: lev, dash: true }], hlines: [{ y: shSM, label: 'Schock–Miquel spacer correlation' }] });
      plots.push({ type: 'line', title: memb ? 'Membrane-wall concentration and local permeate flux' : 'Wall and bulk concentration', xlabel: 'x (mm)', ylabel: memb ? 'Concentration (g/L) · flux (L/m²·h)' : 'Concentration (g/L)', series: [{ name: 'Wall concentration, bottom', x: xm, y: sp.B.w }, { name: 'Wall concentration, top', x: xm, y: sp.T.w }, { name: 'Mixing-cup bulk concentration', x: xm, y: sp.bulk, dash: true },
        ...(memb ? [{ name: 'Permeate flux, bottom (L/m²·h)', x: xm, y: q.JB.map((j) => j / LMH) }, { name: 'Permeate flux, top (L/m²·h)', x: xm, y: q.JT.map((j) => j / LMH) }] : [])] });
    }
    if (th) plots.push({ type: 'line', title: 'Local Nusselt number and wall temperature', xlabel: 'x (mm)', ylabel: 'Nu (–) · T (°C)', series: [{ name: 'Nu, bottom', x: xm, y: th.B.k.map((k) => (k === null ? null : (k * q.dh) / o.energy.alpha)) }, { name: 'Nu, top', x: xm, y: th.T.k.map((k) => (k === null ? null : (k * q.dh) / o.energy.alpha)) }, { name: 'Wall temperature, bottom', x: xm, y: th.B.w }, { name: 'Bulk temperature', x: xm, y: th.bulk, dash: true }] });
    plots.push({ type: 'line', title: 'Mean pressure along the channel', xlabel: 'x (mm)', ylabel: 'Pa', series: [{ name: 'Section-mean static pressure', x: xm, y: q.pm }, { name: 'Flow-averaged total pressure', x: xm, y: q.pt, dash: true }], vlines: [{ x: xm[q.i1], label: 'Δp from' }, { x: xm[q.i2], label: 'Δp to' }] });
    if (q.covC || q.sdT) plots.push({ type: 'line', title: 'Mixing analysis: non-uniformity over the cross-section', xlabel: 'x (mm)', ylabel: 'CoV (%) · standard deviation (K)', zeroY: true, series: [...(q.covC ? [{ name: 'Salt concentration: coefficient of variation (%)', x: xm, y: q.covC.map((a) => 100 * a) }] : []), ...(q.sdT ? [{ name: 'Temperature: standard deviation (K)', x: xm, y: q.sdT }] : [])],
      note: 'Area-weighted standard deviation of the scalar over each section (÷ section mean for the salt): 0 = perfectly mixed. Polarisation layers and thermal boundary layers raise it; filaments that sweep the walls lower it.' });
    const stations = [0.25, 0.5, 0.9].map((s) => clamp(Math.round(s * nx), 0, nx - 1)), ycm = mm(Array.from(r.yc));
    plots.push({ type: 'line', title: 'Velocity profiles', xlabel: 'u (m/s)', ylabel: 'y (mm)', series: [...stations.map((i) => ({ name: `x = ${fmt(xm[i], 3)} mm`, x: Array.from(r.yc, (_, j) => q.uc[j * nx + i]), y: ycm })), { name: 'Plane Poiseuille', x: Array.from(r.yc, (y) => 6 * r.Uref * (y / H) * (1 - y / H)), y: ycm, dash: true }] });
    plots.push({ type: 'line', title: o.steady ? 'Convergence history' : 'Residual history (per time step)', xlabel: 'Iteration', ylabel: 'Normalised residual', logy: true, series: [{ name: 'Continuity', x: r.hist.it, y: r.hist.mass }, { name: 'Velocity change', x: r.hist.it, y: r.hist.dU }, ...(r.hist.scal.length ? [{ name: 'Scalar change (own iteration count)', x: r.hist.scal.map((_, i) => i + 1), y: r.hist.scal, dash: true }] : [])], hlines: [{ y: o.tol, label: 'tolerance' }] });
    let strouhal = null;
    if (r.probe?.t.length > 8) {
      const pv = r.probe.v, t = r.probe.t, half = Math.floor(t.length / 2), m = mean(pv.slice(half));
      let zc = 0, t0 = null, t1 = null;
      for (let k = half + 1; k < t.length; k++) if ((pv[k - 1] - m) * (pv[k] - m) < 0) { zc++; if (t0 === null) t0 = t[k]; t1 = t[k]; }
      const amp = Math.max(...pv.slice(half)) - Math.min(...pv.slice(half));
      if (zc >= 4 && amp > 0.01 * r.Uref) strouhal = ((zc - 1) / 2 / (t1 - t0)) * ((c.geo.type === 'spacer' ? c.geo.df : H) / r.Uref);
      plots.push({ type: 'line', title: 'Probe signal: transverse velocity and pressure drop', xlabel: 'Time (s)', ylabel: 'v (m/s) · Δp (kPa)', series: [{ name: 'Transverse velocity at probe', x: t, y: pv }, { name: 'Inlet–outlet Δp (kPa)', x: t, y: r.probe.dp.map((d) => d / 1000) }], note: strouhal ? `Periodic shedding detected: Strouhal number ≈ ${fmt(strouhal, 3)} (filament diameter, mean velocity).` : 'No sustained oscillation detected at the probe.' });
    }
    if (pt) {
      const bins = 20, hist = (a) => { const h = new Array(bins).fill(0); a.forEach((x) => h[clamp(Math.floor((x / L) * bins), 0, bins - 1)]++); return h; };
      plots.push({ ...base, title: 'Particle trajectories', zlabel: 'Speed', zunit: 'm/s', z: speed, cmap: 'viridis', shapes: [...shapes, ...pt.traj.map((t) => ({ x: mm(t.x), y: mm(t.y), closed: false, color: '#ffffff' }))] });
      plots.push({ type: 'bar', title: 'Deposition map along the channel', ylabel: 'Particles deposited', categories: Array.from({ length: bins }, (_, k) => fmt(((k + 0.5) / bins) * L * 1e3, 3)), stacked: true, series: [{ name: 'Bottom wall', values: hist(pt.bottom) }, { name: 'Top wall', values: hist(pt.top) }, { name: 'Obstacles', values: hist(pt.obstacle) }] });
    }
    // --- tables
    const step = Math.max(1, Math.ceil(nx / 32)), rows = [];
    for (let i = 0; i < nx; i += step) rows.push([xm[i], q.pm[i], q.tauB[i], q.tauT[i], ...(sp ? [sp.bulk[i], sp.B.w[i], sp.T.w[i], q.JB[i] / LMH, q.JT[i] / LMH, Sh(sp.B.k[i]), Sh(sp.T.k[i])] : []), ...(th ? [th.bulk[i], th.B.w[i], th.T.w[i]] : [])]);
    const cmp = [
      ['Darcy friction factor f', q.f, fHP, fSM, turbulent ? fBl : null, one.f],
      ['f·Re (d_h basis; 96 for plane Poiseuille)', q.f * q.Re, 96, fSM * q.Re, turbulent ? fBl * q.Re : null, one.f * q.Re],
      ['Fanning f·Re (24 for plane Poiseuille)', (q.f * q.Re) / 4, 24, (fSM * q.Re) / 4, null, (one.f * q.Re) / 4],
      ['Pressure gradient (Pa/m)', q.dpdx, (fHP * fl.rho * r.Uref ** 2) / (2 * q.dh), (fSM * fl.rho * r.Uref ** 2) / (2 * q.dh), turbulent ? (fBl * fl.rho * r.Uref ** 2) / (2 * q.dh) : null, one.dpPerM],
    ];
    if (sp) {
      const lev = (memb ? 2.236 : 1.849) * one.xs ** (-1 / 3);
      cmp.push(['Mean Sherwood number, whole wall', shAll, Math.max(memb ? 8.235 : 7.541, lev), shSM, null, one.Sh], ['Mean Sherwood number, downstream half', shDev, null, shSM, null, one.Sh], ['Mass-transfer coefficient (µm/s)', kDev ? kDev * 1e6 : null, null, kSM * 1e6, null, one.k * 1e6]);
      if (memb) cmp.push(['Mean permeate flux (L/m²·h)', q.Jmean / LMH, null, null, null, one.flux], ['Polarisation factor c_wall/c_bulk', q.cpMean, null, null, null, one.CP], ['Peak wall concentration at obstacle contact lines (g/L)', q.cwPeak, null, null, null, null]);
    }
    if (th) cmp.push(['Nusselt number, downstream half', nuDev, 7.541, null, turbulent ? 0.023 * q.Re ** 0.8 * Pr ** 0.4 : null, null]);
    const tables = [
      { title: 'Comparison with analytical solutions and correlations', columns: ['Quantity', 'CFD (this run)', 'Empty channel, laminar (Hagen–Poiseuille / Lévêque–Graetz)', 'Spacer correlation (Schock–Miquel)', 'Turbulent duct (Blasius / Dittus–Boelter)', 'Calibrated 1-D model'], rows: cmp,
        note: `Re = ρ·U·d_h/μ with d_h = 2H and U the superficial mean velocity. Schock–Miquel: f = 6.23 Re^−0.3, Sh = 0.065 Re^0.875 Sc^0.25 (commercial 3-D net spacers; a 2-D transverse-filament section is expected to differ). Pressure gradient = loss of flow-averaged total pressure between x = ${fmt(xm[q.i1], 3)} and ${fmt(xm[q.i2], 3)} mm.` },
      { title: 'Streamwise profiles', columns: ['x (mm)', 'Mean p (Pa)', 'τ_w bottom (Pa)', 'τ_w top (Pa)', ...(sp ? ['Bulk c (g/L)', 'Wall c bottom (g/L)', 'Wall c top (g/L)', 'Flux bottom (L/m²·h)', 'Flux top (L/m²·h)', 'Sh bottom', 'Sh top'] : []), ...(th ? ['Bulk T (°C)', 'Wall T bottom (°C)', 'Wall T top (°C)'] : [])], rows },
      { title: 'Grid, fluid and solver summary', columns: ['Item', 'Value'], rows: [
        ['Grid nx × ny (cells)', `${nx} × ${ny} (${nx * ny})`], ['Cell size Δx (µm)', r.dx * 1e6], ['Wall-cell height (µm)', dy0 * 1e6], ['Centre-cell height (µm)', r.dy[ny >> 1] * 1e6], ['Blocked (solid) cells (%)', 100 * c.mk.solidFraction],
        ['Density (kg/m³)', fl.rho], ['Viscosity (mPa·s)', fl.mu * 1e3], ['Salt diffusivity (m²/s)', fl.D], ['Schmidt number', q.Sc], ['Prandtl number', Pr],
        ['Iterations', r.iters], ['Final continuity residual', r.hist.mass.at(-1) ?? 0], ['Final velocity change', r.hist.dU.at(-1) ?? 0], ['Final scalar change', r.scalRes], ['Scheme', v.scheme], ['Converged', r.converged ? 'yes' : 'no']] },
    ];
    if (pt) tables.push({ title: 'Particle fate', columns: ['Fate', 'Particles', 'Share (%)'], rows: [['Deposited on bottom wall', pt.bottom.length], ['Deposited on top wall', pt.top.length], ['Captured on obstacles', pt.obstacle.length], ['Left through the outlet', pt.tRes.length], ['Still suspended (recirculating)', pt.suspended]].map(([a, b]) => [a, b, (100 * b) / pt.n]),
      note: `Relaxation time ${fmt(pt.tau, 3)} s, settling velocity ${fmt(-pt.vs * 1e6, 3)} µm/s, Brownian diffusivity ${fmt(pt.DB, 3)} m²/s. Mean residence time of escaped particles ${pt.tRes.length ? fmt(mean(pt.tRes), 3) : '–'} s (plug-flow time ${fmt(L / r.Uref, 3)} s).` });
    // --- balances and outputs
    const balances = [{ name: 'Water volume (m²/s per metre width)', in: r.Qin, out: r.Qin * (1 - massErr) }];
    if (sp) { let cin = 0, cout = 0; for (let j = 0; j < ny; j++) { cin += r.uin[j] * o.species.c0 * r.dy[j]; cout += r.u[j * r.nu1 + nx] * r.spc.phi[j * nx + nx - 1] * r.dy[j]; } let per = 0, fixIn = 0; for (let i = 0; i < nx; i++) { per += (r.Jb[i] * r.spc.pB[i] + r.Jt[i] * r.spc.pT[i]) * r.dx; fixIn += (sp.B.flux[i] + sp.T.flux[i]) * r.dx; } balances.push({ name: 'Salt (g/s per metre width)', in: cin * 1000 + (memb ? 0 : fixIn * 1000), out: (cout + per) * 1000 }); }
    if (th) { let hin = 0, hout = 0, qw = 0; for (let j = 0; j < ny; j++) { hin += r.uin[j] * v.T * r.dy[j]; hout += r.u[j * r.nu1 + nx] * r.eng.phi[j * nx + nx - 1] * r.dy[j]; } for (let i = 0; i < nx; i++) { qw += (th.B.flux[i] + th.T.flux[i]) * r.dx; hout += (r.Jb[i] * r.eng.phi[i] + r.Jt[i] * r.eng.phi[(ny - 1) * nx + i]) * r.dx; } balances.push({ name: 'Heat (kW per metre width, relative to 0 °C)', in: ((hin + qw) * fl.rho * fl.cp) / 1000, out: (hout * fl.rho * fl.cp) / 1000 }); }
    const pumpW = (q.dpdx * r.Uref * H) / Math.max(1, q.nSides ?? 2);
    const outputs = { dpPerM: q.dpdx, frictionFactor: q.f, sherwood: shDev, kMass: kDev, cpFactor: sp ? q.cpMean : null, wallShear: q.tauMean, kMultiplier: kMult ? clamp(kMult, 0.2, 5) : null, reynolds: q.Re, schmidt: q.Sc, maxWallConc: sp ? q.cwMax : null, fluxLMH: memb ? q.Jmean / LMH : null, recirculationFraction: q.recirc, crossFlowIntensity: q.crossFlow, nusselt: nuDev, converged: r.converged ? 1 : 0, iterations: r.iters };
    for (const k of Object.keys(outputs)) if (!Number.isFinite(outputs[k])) delete outputs[k];
    const kpis = [
      { label: 'Reynolds number (d_h = 2H)', value: q.Re, unit: '', status: !turbulent && q.Re > 2500 ? 'warn' : 'ok', help: 'ρ·U·d_h/μ with the superficial mean velocity' },
      { label: 'Pressure gradient', value: q.dpdx / 100, unit: 'mbar/m', status: Math.abs(q.dpdx) / 1e5 > v.limDP ? 'bad' : 'ok', help: 'Loss of flow-averaged total pressure per metre between the two marked planes' },
      { label: 'Darcy friction factor', value: q.f, unit: '', help: 'f = (Δp/L)·d_h / (½ρU²)' },
      { label: 'f·Re', value: q.f * q.Re, unit: '', help: '96 for fully developed laminar flow between plates' },
      { label: 'Mean wall shear stress', value: q.tauMean, unit: 'Pa' },
      { label: 'Minimum wall shear', value: q.tauMin, unit: 'Pa', status: q.tauMin < v.limTau ? 'warn' : 'ok' },
      { label: 'Recirculation (reverse-flow) area', value: 100 * q.recirc, unit: '%', help: 'Share of the fluid area with upstream-directed velocity' },
      { label: 'Stagnant area (< 10 % of mean speed)', value: 100 * q.stagnant, unit: '%' },
      { label: 'Cross-flow (mixing) intensity ⟨|v|⟩/U', value: q.crossFlow, unit: '', help: 'Area-mean transverse speed ÷ mean axial velocity: 0 in a straight empty channel, larger where filaments or baffles stir the flow across the gap' },
      { label: 'Specific pumping power', value: pumpW, unit: 'W/m² of wall', help: 'Hydraulic power dissipated per unit active wall area' },
    ];
    if (q.covC) kpis.push({ label: 'Salt non-uniformity at the outlet (CoV)', value: 100 * q.covC[nx - 1], unit: '%', help: 'Coefficient of variation of the concentration over the outlet section; see the mixing-analysis plot' });
    if (q.sdT) kpis.push({ label: 'Temperature non-uniformity at the outlet', value: q.sdT[nx - 1], unit: 'K', help: 'Standard deviation of the temperature over the outlet section' });
    if (sp) kpis.push({ label: 'Sherwood number (downstream half)', value: shDev ?? '–', unit: '' }, { label: 'Mass-transfer coefficient', value: kDev ? kDev * 1e6 : '–', unit: 'µm/s' }, { label: 'k ÷ Schock–Miquel correlation', value: kMult ?? '–', unit: '×', help: 'Offered to the RO design suite as its mass-transfer multiplier' });
    if (memb) kpis.push({ label: 'Polarisation factor (mean)', value: q.cpMean, unit: '', status: q.cpMean > v.limCP ? 'warn' : 'ok' }, { label: 'Maximum wall concentration', value: q.cwMax, unit: 'g/L', help: 'Highest membrane-surface concentration away from obstacle contact lines; the contact-line peak is listed in the comparison table' }, { label: 'Mean permeate flux', value: q.Jmean / LMH, unit: 'L/m²·h' }, { label: 'Local salt rejection', value: sp.bulk[0] > 0 ? 100 * (1 - q.cPerm / v.c0) : 0, unit: '%', sig: 5 });
    if (th) kpis.push({ label: 'Nusselt number (downstream half)', value: nuDev ?? '–', unit: '' }, { label: 'Outlet bulk temperature', value: th.bulk[nx - 1], unit: '°C' }, ...(tpc !== null ? [{ label: 'Temperature-polarisation coefficient', value: tpc, unit: '', help: '(T_wall − T_ext)/(T_bulk − T_ext); 1 = no polarisation' }] : []));
    if (pt) kpis.push({ label: 'Particles deposited', value: (100 * (pt.bottom.length + pt.top.length + pt.obstacle.length)) / pt.n, unit: '%' });
    if (strouhal) kpis.push({ label: 'Strouhal number', value: strouhal, unit: '' });
    kpis.push({ label: 'Iterations', value: r.iters, unit: '', status: r.converged ? 'ok' : 'warn' }, { label: 'Mass-balance closure', value: Math.abs(massErr) * 100, unit: '%', status: Math.abs(massErr) > 1e-3 ? 'warn' : 'ok' });
    // ================= extended models =================
    const info = (msg) => W.push({ level: 'info', msg }), warn = (msg) => W.push({ level: 'warn', msg }), n = nx * ny, sumRows = tables[2].rows;
    const cellArea = (P) => r.dx * r.dy[(P / nx) | 0];
    kpis.push({ label: 'Mass flow per metre width', value: fl.rho * r.Qin, unit: 'kg/s·m', help: 'ρ × volume flow through the 2-D section' });
    outputs.massFlow = fl.rho * r.Qin; outputs.meanVelocity = r.Uref;
    if (v.inletBC === 'pressure') { kpis.push({ label: 'Inlet gauge pressure (target)', value: v.pInlet, unit: 'Pa' }, { label: 'Resulting mean velocity', value: r.Uref, unit: 'm/s', help: 'Found by the flow-rate controller so that the inlet pressure equals the target' }); sumRows.push(['Inlet condition', 'pressure inlet'], ['Inlet pressure reached, first cell column (Pa)', q.pm[0]]); if (!o.steady) W.push({ level: 'info', msg: 'Pressure inlet in a transient run: the flow rate was found by steady iterations before the time-accurate run and then held, so the instantaneous inlet pressure fluctuates about the target.' }); if (c.pInSub) W.push({ level: 'info', msg: `Pressure inlet with a turbulence closure: the laminar parabolic inlet profile was replaced by ${c.pInSub === 'periodic' ? 'the fully developed profile of the flow itself (recycled from the outlet plane)' : 'a uniform profile'}. A parabolic profile flattens in turbulent flow and recovers more static pressure than friction consumes, so no flow rate would give a positive inlet gauge pressure.` }); }
    else if (v.inletBC === 'massflow') sumRows.push(['Inlet condition', `mass-flow inlet, ${fmt(v.mdot, 4)} kg/s per metre width`]);
    if (v.creeping) { info('Creeping-flow (Stokes) limit: the convective terms of the momentum equations are dropped, so the pressure drop is exactly proportional to the flow rate and the flow is reversible.'); sumRows.push(['Momentum convection', 'off (Stokes flow)']); }
    if (r.wallB !== 'noslip' || r.wallT !== 'noslip') {
      const us = (top) => { const j = top ? ny - 1 : 0; let a = 0, m = 0; for (let i = Math.round(nx / 2); i < nx; i++) if (!r.solid[j * nx + i]) { const wt = top ? r.wallT : r.wallB; a += wt === 'sym' ? q.uc[j * nx + i] : wt === 'slip' ? ((top ? q.tauT : q.tauB)[i] * o.slipLen) / fl.mu : 0; m++; } return m ? a / m : 0; };
      kpis.push({ label: r.wallB === 'slip' ? 'Wall slip velocity (downstream half)' : 'Velocity on the symmetry plane', value: Math.max(us(0), us(1)), unit: 'm/s', help: r.wallB === 'slip' ? 'Navier slip: u_wall = b·∂u/∂y with slip length b' : 'Zero shear and zero normal velocity on the symmetry plane' });
      sumRows.push(['Bottom wall', r.wallB === 'slip' ? `Navier slip, b = ${fmt(v.slipLen, 3)} µm` : r.wallB === 'sym' ? 'symmetry plane' : 'no-slip'], ['Top wall', r.wallT === 'slip' ? `Navier slip, b = ${fmt(v.slipLen, 3)} µm` : r.wallT === 'sym' ? 'symmetry plane' : 'no-slip']);
    }
    // --- two-equation closures and LES
    if (r.tk) {
      const isR = r.tm === 'rsm', isE = r.tm === 'ke' || isR, yPlus = (0.5 * dy0 * r.utau) / (fl.mu / fl.rho), ist = clamp(Math.round(0.85 * nx), 0, nx - 1);
      let kmax = 0, ntMax = 0;
      for (let P = 0; P < n; P++) if (!r.solid[P]) { kmax = Math.max(kmax, r.tk[P]); ntMax = Math.max(ntMax, r.mue[P] / fl.mu - 1); }
      plots.push({ ...base, title: 'Turbulent kinetic energy', zlabel: 'k', zunit: 'm²/s²', z: q.field(r.tk), cmap: 'turbo' });
      plots.push({ ...base, title: isE ? 'Turbulent dissipation rate ε' : 'Specific dissipation rate ω (log₁₀)', zlabel: isE ? 'ε' : 'log₁₀ ω', zunit: isE ? 'm²/s³' : 'log₁₀(1/s)', z: q.field(isE ? r.te : r.te.map((w) => Math.log10(Math.max(w, 1e-30)))), cmap: 'viridis' });
      const ut = Math.max(Math.sqrt(Math.abs(q.tauB[ist]) / fl.rho), 1e-12), half = Math.ceil(ny / 2), yp = [], up = [], kp = [], uvp = [], uup = [], vvp = [], wwp = [];
      for (let j = 0; j < half; j++) { const P = j * nx + ist; if (r.solid[P]) continue; yp.push(Math.max((r.yc[j] * ut) / (fl.mu / fl.rho), 1e-3)); up.push(q.uc[P] / ut); kp.push(r.tk[P] / (ut * ut)); uvp.push(-r.rs.uv[P] / (ut * ut)); uup.push(r.rs.uu[P] / (ut * ut)); vvp.push(r.rs.vv[P] / (ut * ut)); if (isR) wwp.push(r.rs.ww[P] / (ut * ut)); }
      if (yp.length > 2 && r.wallB === 'noslip') {
        plots.push({ type: 'line', title: 'Velocity profile in wall units (bottom wall, x = 85 % of length)', xlabel: 'y⁺', ylabel: 'u⁺', logx: true, series: [{ name: 'Computed', x: yp, y: up, mode: 'both' }, { name: 'Log law u⁺ = ln(y⁺)/0.41 + 5.2', x: yp, y: yp.map((y) => (y > 11.6 ? Math.log(y) / KAPPA + 5.2 : y)), dash: true }] });
        plots.push({ type: 'line', title: 'Reynolds stresses in wall units', xlabel: 'y⁺', ylabel: 'stress / u_τ²', logx: true, series: [{ name: 'k⁺', x: yp, y: kp }, { name: '−u′v′⁺', x: yp, y: uvp }, { name: 'u′u′⁺', x: yp, y: uup }, { name: 'v′v′⁺', x: yp, y: vvp }, ...(isR ? [{ name: 'w′w′⁺', x: yp, y: wwp }, { name: 'Total-stress balance 1 − 2y/H', x: yp, y: yp.map((_, k2) => Math.max(0, 1 - (2 * r.yc[k2]) / H)), dash: true, color: '#94a3b8' }] : [])], note: isR ? 'Each stress is the solution of its own transport equation; in fully developed channel flow −u′v′⁺ follows the total-stress line. The cell next to the wall holds the log-layer levels of the wall function.' : r.tm === 'earsm' ? 'Normal-stress anisotropy comes from the explicit algebraic Reynolds-stress closure.' : 'Boussinesq closures give nearly isotropic normal stresses in simple shear (u′u′ ≈ v′v′ ≈ ⅔k).' });
      }
      const jm = Math.min(Math.round(0.2 * ny), half - 1), Pm = jm * nx + ist, aXX = r.tk[Pm] > 0 ? (r.rs.uu[Pm] - r.rs.vv[Pm]) / (2 * r.tk[Pm]) : 0, a12 = r.tk[Pm] > 0 ? r.rs.uv[Pm] / r.tk[Pm] : 0;
      kpis.push({ label: 'Friction velocity u_τ', value: r.utau, unit: 'm/s' }, { label: 'First-cell y⁺', value: yPlus, unit: '', status: isE && yPlus < 11.6 ? 'warn' : 'ok' }, { label: 'Peak turbulence intensity √(⅔k)/U', value: Math.sqrt((2 / 3) * kmax) / r.Uref, unit: '' }, { label: 'Maximum eddy-viscosity ratio ν_t/ν', value: ntMax, unit: '' });
      if (r.tm === 'earsm' || isR) kpis.push({ label: 'Normal-stress anisotropy (u′u′ − v′v′)/2k', value: aXX, unit: '', help: 'Evaluated at 20 % of the gap, x = 85 % of the length; 0 for a Boussinesq closure' }, { label: 'Shear-stress ratio u′v′/k', value: a12, unit: '', help: '≈ −0.30 in an equilibrium boundary layer' });
      if (isR) {
        plots.push({ ...base, title: 'Reynolds shear stress −u′v′ (transported)', zlabel: '−u′v′', zunit: 'm²/s²', z: q.field(r.rs.uv.map((x) => -x)), cmap: 'coolwarm', contours: 8 }, { ...base, title: 'Normal-stress anisotropy (u′u′ − v′v′)/2k (transported stresses)', zlabel: '(u′u′ − v′v′)/2k', zunit: '–', z: q.field(r.rs.uu.map((x, P) => (r.tk[P] > 1e-12 ? (x - r.rs.vv[P]) / (2 * r.tk[P]) : 0))), cmap: 'viridis' });
        let real = 0, cnt = 0; for (let P = 0; P < n; P++) if (!r.solid[P]) { cnt++; if (Math.abs(r.rs.uv[P]) > 0.979 * Math.sqrt(r.rs.uu[P] * r.rs.vv[P])) real++; }
        if (real > 0.02 * cnt) warn(`Reynolds-stress model: the shear stress sits on the realisability bound |u′v′| ≤ √(u′u′·v′v′) in ${fmt((100 * real) / cnt, 2)} % of the cells — the solution there is limited rather than resolved; refine the grid or check the convergence.`);
      }
      if (isR && yPlus < 11.6) warn(`The Reynolds-stress transport model is a high-Reynolds-number closure with wall functions and needs the first cell in the log layer (y⁺ > 11.6, ideally 30–300); it is at y⁺ = ${fmt(yPlus, 3)}. Lower the wall refinement toward 1–3 or use the k–ω SST model, which integrates to the wall.`);
      else if (isE && yPlus < 11.6) warn(`Standard k–ε with wall functions needs the first cell in the log layer (y⁺ > 11.6, ideally 30–300); it is at y⁺ = ${fmt(yPlus, 3)}. Lower the wall refinement toward 1–3 or use the k–ω SST model, which integrates to the wall.`);
      tables.push({ title: 'Turbulence closure summary', columns: ['Item', 'Value'], rows: [['Closure', TURB_NAMES[r.tm]], ['Inlet turbulence intensity (%)', v.tuIn], ['Inlet length scale (mm)', o.lTurb * 1e3], ['Friction velocity (m/s)', r.utau], ['First-cell y⁺', yPlus], ['Peak k (m²/s²)', kmax], ['Peak ν_t/ν', ntMax], ['Skin-friction coefficient C_f = τ_w/(½ρU²)', q.tauMean / (0.5 * fl.rho * r.Uref ** 2)], ['Dean correlation 0.073 Re_H^−0.25 (plane channel)', 0.073 * (q.Re / 2) ** -0.25], ['Shear-stress ratio u′v′/k at 20 % of the gap', a12], ['Anisotropy (u′u′ − v′v′)/2k at 20 % of the gap', aXX], ...(isR ? [['w′w′/k at 20 % of the gap', r.tk[Pm] > 0 ? r.rs.ww[Pm] / r.tk[Pm] : 0], ['Pressure–strain constants C1, C2, C1′, C2′', `${RSM.C1}, ${RSM.C2}, ${RSM.C1w}, ${RSM.C2w}`], ['Diffusion constants C_s, C_ε (Daly–Harlow)', `${RSM.Cs}, ${RSM.Ce}`]] : [])] });
      outputs.turbulentKineticEnergy = kmax; outputs.frictionVelocity = r.utau;
    }
    if (r.tm === 'les') {
      info('Large-eddy simulation in two dimensions: the filtered Navier–Stokes equations are advanced in time with the Smagorinsky sub-grid viscosity (C_s Δ)²|S̄|, Δ = √(ΔxΔy), van Driest damping and a wall model where y⁺ > 11.6. Without the third dimension there is no vortex stretching, so the resolved eddies follow the inverse cascade of 2-D turbulence — treat the statistics as indicative only.');
      if (v.mode !== 'transient') info('LES is time-dependent by definition: the run was switched to the transient mode.');
      let nt = 0, m = 0;
      for (let P = 0; P < n; P++) if (!r.solid[P]) { nt += r.mue[P] / fl.mu - 1; m++; }
      const pv = r.probe?.v || [], h2 = pv.slice(Math.floor(pv.length / 2)), mv = h2.length ? mean(h2) : 0, rms = h2.length ? Math.sqrt(mean(h2.map((x) => (x - mv) ** 2))) : 0;
      kpis.push({ label: 'Mean sub-grid viscosity ratio ν_sgs/ν', value: m ? nt / m : 0, unit: '' }, { label: 'Resolved fluctuation v′_rms / U at the probe', value: rms / r.Uref, unit: '' });
      sumRows.push(['Sub-grid model', `Smagorinsky, C_s = ${fmt(o.cSmag, 3)}`]);
    }
    // --- porous zone: Ergun / Darcy–Forchheimer check
    if (o.porous) {
      const pz = o.porous, ia = clamp(Math.ceil(pz.x0 / r.dx + 0.5), 0, nx - 1), ib = clamp(Math.floor(pz.x1 / r.dx - 0.5), 0, nx - 1);
      if (ib > ia + 1) {
        const g1 = (q.pm[ia] - q.pm[ib]) / ((ib - ia) * r.dx), ref = (fl.mu * r.Uref) / pz.K + (fl.rho * pz.cF * r.Uref ** 2) / Math.sqrt(pz.K);
        kpis.push({ label: 'Pressure gradient inside the porous zone', value: g1 / 1e5, unit: 'bar/m' }, { label: pz.ergun ? 'Ergun equation (1-D packed bed)' : 'Darcy–Forchheimer (1-D)', value: ref / 1e5, unit: 'bar/m', help: pz.ergun ? '150 μ(1−ε)²U/(d_p²ε³) + 1.75 ρ(1−ε)U²/(d_p ε³) at the superficial velocity' : 'μU/K + ρ c_F U²/√K at the superficial velocity' });
        outputs.porousGradient = g1;
      }
      if (pz.ergun) { sumRows.push(['Packed-bed particle diameter (mm)', pz.dp * 1e3], ['Bed porosity', pz.eps], ['Ergun permeability K (m²)', pz.K], ['Ergun inertial coefficient c_F', pz.cF], ['Particle Reynolds number ρ U d_p / (μ(1−ε))', (fl.rho * r.Uref * pz.dp) / (fl.mu * (1 - pz.eps))]); }
    }
    // --- conjugate heat transfer
    if (th && o.energy.solidD > 0) {
      let ts = 0, m = 0, tmax = -Infinity, qgen = 0, qwS = 0;
      for (let P = 0; P < n; P++) if (r.solid[P]) { const T = r.eng.phi[P]; ts += T; m++; if (T > tmax) tmax = T; qgen += o.energy.qS * cellArea(P); }
      for (const top of [0, 1]) { const w = top ? o.energy.top : o.energy.bot, j = top ? ny - 1 : 0, tw = top ? r.eng.wT : r.eng.wB; if (w.type === 'none') continue; for (let i = 0; i < nx; i++) if (r.solid[j * nx + i]) qwS += ((o.energy.solidD * (tw[i] - r.eng.phi[j * nx + i])) / (0.5 * r.dy[j])) * r.dx; }
      const hb = balances.find((b) => b.name.startsWith('Heat'));
      if (hb) hb.in += ((qgen + qwS) * fl.rho * fl.cp) / 1000;
      const Tz = q.yp.map((_, a) => q.xc.map((__, i) => r.eng.phi[q.jn[a] * nx + i]));
      plots.push({ type: 'field', title: 'Temperature in fluid and solids (conjugate heat transfer)', xlabel: 'x (mm)', ylabel: 'y (mm)', x: xm, y: ym, z: Tz, zlabel: 'Temperature', zunit: '°C', cmap: 'thermal', contours: 8, equal: L / H <= 8, shapes });
      if (m) kpis.push({ label: 'Mean solid temperature', value: ts / m, unit: '°C' }, { label: 'Hottest solid temperature', value: tmax, unit: '°C' });
      kpis.push({ label: 'Heat conducted through solids to the walls', value: (qwS * fl.rho * fl.cp) / 1000, unit: 'kW/m' });
      sumRows.push(['Solid thermal conductivity (W/m·K)', v.kSolid], ['Heat release in solids (W/m³)', v.qSolid], ['Fluid thermal conductivity (W/m·K)', fl.k]);
      if (!m) info('Conjugate heat transfer is switched on but the geometry contains no solid cells.');
    } else if (v.cht && !v.energy) info('Conjugate heat transfer needs the energy equation — switch it on.');
    // --- lattice-Boltzmann hybrid
    if (lbm) {
      if (lbm.note) warn(lbm.note);
      else {
        const xa = xm[q.i1] / 1e3, xb = xm[q.i2] / 1e3, gL = (lbm.ptAt(xa) - lbm.ptAt(xb)) / (xb - xa), ist = clamp(Math.round(0.75 * nx), 0, nx - 1), ycm2 = mm(Array.from(r.yc)), stepX = Math.ceil(lbm.nxL / 200), nxP = Math.floor(lbm.nxL / stepX), stepY = Math.ceil(lbm.nyL / 100), nyP = Math.floor(lbm.nyL / stepY);
        const zl = [], ml = [];
        for (let a = 0; a < nyP; a++) { const row = [], mrow = []; for (let b = 0; b < nxP; b++) { const P = a * stepY * lbm.nxL + b * stepX; mrow.push(!!lbm.sol[P]); row.push(lbm.sol[P] ? NaN : lbm.vs * Math.hypot(lbm.lb.ux[P], lbm.lb.uy[P])); } zl.push(row); ml.push(mrow); }
        plots.push({ type: 'field', title: 'Lattice-Boltzmann velocity magnitude (D2Q9 lattice)', xlabel: 'x (mm)', ylabel: 'y (mm)', x: Array.from({ length: nxP }, (_, b) => ((b * stepX + 0.5) * L * 1e3) / lbm.nxL), y: Array.from({ length: nyP }, (_, a) => ((a * stepY + 0.5) * H * 1e3) / lbm.nyL), z: zl, mask: ml, zlabel: 'Speed', zunit: 'm/s', cmap: 'viridis', equal: L / H <= 8, shapes });
        plots.push({ type: 'line', title: `Velocity profile at x = ${fmt(xm[ist], 3)} mm: lattice-Boltzmann vs finite volume`, xlabel: 'u (m/s)', ylabel: 'y (mm)', series: [{ name: 'Finite volume (final)', x: Array.from(r.yc, (_, j) => q.uc[j * nx + ist]), y: ycm2 }, { name: 'Lattice-Boltzmann', x: Array.from(r.yc, (y) => lbm.vs * lbm.smp(lbm.lb.ux, q.xc[ist], y)), y: ycm2, dash: true }] });
        const dev = Math.abs(q.dpdx) > 0 ? gL / q.dpdx - 1 : 0;
        kpis.push({ label: 'Lattice-Boltzmann pressure gradient', value: gL / 100, unit: 'mbar/m' }, { label: 'Lattice-Boltzmann ÷ finite-volume pressure gradient', value: 1 + dev, unit: '', status: Math.abs(dev) > 0.15 ? 'warn' : 'ok' });
        tables.push({ title: 'Lattice-Boltzmann / finite-volume hybrid', columns: ['Quantity', 'Lattice-Boltzmann', 'Finite volume'], rows: [['Grid', `${lbm.nxL} × ${lbm.nyL} lattice`, `${nx} × ${ny} cells`], ['Pressure gradient (Pa/m)', gL, q.dpdx], ['Maximum speed (m/s)', lbm.umax, q.umax], ['Time steps / iterations', lbm.steps, r.iters], ['Relaxation time τ', lbm.tau, null], ['Lattice Mach number u/c_s', lbm.uL * Math.sqrt(3), null], ['Converged', lbm.converged ? 'yes' : 'no', r.converged ? 'yes' : 'no']],
          note: 'The lattice-Boltzmann field (BGK collisions, halfway bounce-back walls) is interpolated to the staggered faces as the starting field; the finite-volume solver then enforces discrete continuity and solves the salt and heat transport on it. Wall permeation and turbulence closures act only in the finite-volume stage.' });
        info(`Hybrid run: D2Q9 lattice-Boltzmann flow on ${lbm.nxL} × ${lbm.nyL} nodes (${lbm.steps} steps, τ = ${fmt(lbm.tau, 3)}) started the finite-volume solution, which then needed ${r.iters} iterations.`);
        if (!lbm.converged) info('The lattice-Boltzmann stage stopped at its step limit before reaching a steady state; it still serves as the starting field.');
        outputs.lbmGradient = gL;
      }
    }
    // --- species flux wall
    if (v.species === 'flux') sumRows.push(['Specified wall salt flux (mg/m²·s)', v.jwSalt]);
    // --- precipitation, crystal population and fouling
    if (r.scal) {
      const pr = o.precip, S = new Float64Array(n), sw = (a) => Array.from(a, (c) => c / pr.csat), depR = (a) => Array.from(a, (x) => x * 3.6e6); // kg/m²/s → g/m²/h
      for (let P = 0; P < n; P++) S[P] = r.scal.phi[P] / pr.csat;
      const act = [pr.bot ? ['bottom', r.scal.wB, r.scal.nB, 0] : null, pr.top ? ['top', r.scal.wT, r.scal.nT, ny - 1] : null].filter(Boolean);
      let swMax = 0, dep = 0, depMax = 0, wet = 0;
      for (const [, w, nn, row] of act) for (let i = 0; i < nx; i++) { if (r.solid[row * nx + i]) continue; swMax = Math.max(swMax, w[i] / pr.csat); dep += nn[i] * r.dx; depMax = Math.max(depMax, nn[i]); wet += r.dx; }
      plots.push({ ...base, title: 'Supersaturation ratio of the scaling salt', zlabel: 'S = c/c_sat', zunit: '–', z: q.field(S), cmap: 'turbo' });
      plots.push({ type: 'line', title: 'Wall supersaturation and scale deposition flux', xlabel: 'x (mm)', ylabel: 'S_wall (–) · flux (g/m²·h)', series: act.flatMap(([nm, w, nn]) => [{ name: `Wall supersaturation, ${nm}`, x: xm, y: sw(w) }, { name: `Deposition flux, ${nm} (g/m²·h)`, x: xm, y: depR(nn) }]), hlines: [{ y: 1, label: 'saturation' }] });
      let cin = 0, cout = 0, sink = 0;
      for (let j = 0; j < ny; j++) { cin += r.uin[j] * pr.c0 * r.dy[j]; cout += r.u[j * r.nu1 + nx] * r.scal.phi[j * nx + nx - 1] * r.dy[j]; }
      if (r.scal.sp.Sp) for (let P = 0; P < n; P++) sink += (r.scal.sp.Sp[P] * r.scal.phi[P] - r.scal.sp.S[P]) * cellArea(P);
      balances.push({ name: 'Scaling salt (g/s per metre width)', in: cin * 1000, out: (cout + dep + sink) * 1000 });
      kpis.push({ label: 'Maximum wall supersaturation', value: swMax, unit: '', status: swMax > 1 ? 'warn' : 'ok', help: 'c_wall / c_sat of the sparingly soluble salt; above 1 scale forms on the surface' }, { label: 'Mean scale deposition flux', value: wet ? (dep / wet) * 3.6e6 : 0, unit: 'g/m²·h' }, { label: 'Peak scale deposition flux', value: depMax * 3.6e6, unit: 'g/m²·h' }, { label: 'Scaling salt removed at the walls', value: cin > 0 ? (100 * dep) / cin : 0, unit: '% of feed' });
      outputs.maxSupersaturation = swMax; outputs.scaleFlux = wet ? dep / wet : 0;
      const rowsP = [['Feed concentration of the scaling salt (g/L)', pr.c0], ['Solubility (g/L)', pr.csat], ['Bulk supersaturation', pr.c0 / pr.csat], ['Maximum wall supersaturation', swMax], ['Surface rate constant (µm/s)', pr.kr * 1e6], ['Wall deposition (g/s per m width)', dep * 1000], ['Bulk crystallisation (g/s per m width)', sink * 1000]];
      if (r.mom) {
        const [m0, m1, m2, m3] = r.mom.map((m) => m.phi), cup = (a) => q.xc.map((_, i) => { let s = 0, qq = 0; for (let j = 0; j < ny; j++) { const P = j * nx + i; if (r.solid[P]) continue; s += q.uc[P] * a[P] * r.dy[j]; qq += q.uc[P] * r.dy[j]; } return Math.abs(qq) > 1e-300 ? s / qq : 0; });
        const c0m = cup(m0), c1m = cup(m1), c2m = cup(m2), c3m = cup(m3), d10 = c0m.map((a, i) => (a > 1e-30 ? (c1m[i] / a) * 1e6 : 0)), d32 = c2m.map((a, i) => (a > 1e-300 ? (c3m[i] / a) * 1e6 : 0)), e = nx - 1;
        plots.push({ ...base, title: 'Crystal number density (zeroth moment, log₁₀)', zlabel: 'log₁₀ m₀', zunit: 'log₁₀(1/m³)', z: q.field(m0.map((x) => Math.log10(Math.max(x, 1)))), cmap: 'viridis' });
        plots.push({ type: 'line', title: 'Crystal population along the channel (flow-averaged moments)', xlabel: 'x (mm)', ylabel: 'size (µm) · log₁₀ number (1/m³)', series: [{ name: 'Number-mean size d₁₀ (µm)', x: xm, y: d10 }, { name: 'Sauter mean size d₃₂ (µm)', x: xm, y: d32 }, { name: 'log₁₀ number density', x: xm, y: c0m.map((a) => Math.log10(Math.max(a, 1))) }] });
        kpis.push({ label: 'Crystal number density at the outlet', value: c0m[e], unit: '1/m³' }, { label: 'Sauter mean crystal size at the outlet', value: d32[e], unit: 'µm' }, { label: 'Suspended crystal mass at the outlet', value: pr.rhoC * (Math.PI / 6) * c3m[e] * 1e3, unit: 'mg/L' });
        rowsP.push(['Outlet number density (1/m³)', c0m[e]], ['Outlet d₁₀ (µm)', d10[e]], ['Outlet d₃₂ (µm)', d32[e]], ['Outlet crystal volume fraction', (Math.PI / 6) * c3m[e]], ['Nucleation law', `B = ${fmt(pr.kn, 3)} (S − 1)^${fmt(pr.nn, 3)} 1/m³·s`], ['Growth law', `G = ${fmt(pr.kg * 1e6, 3)} (S − 1) µm/s`]);
        outputs.crystalD32 = d32[e];
      }
      tables.push({ title: 'Precipitation and crystal population', columns: ['Item', 'Value'], rows: rowsP, note: 'The scaling salt is transported as its own species, fully rejected by the membrane, and crystallises on the active walls at N = k_r (c_wall − c_sat). The population balance is solved by the method of moments (m₀…m₃) with primary nucleation and size-independent linear growth; crystal growth consumes the dissolved salt.' });
    } else if (v.pbm && !v.precip) info('The population balance needs the precipitation model — switch it on.');
    if (r.foul) {
      const fo = r.foul, th2 = (a) => Array.from(a, (m) => (m / v.foulRho) * 1e6), th = [...(o.species.bot === 'membrane' ? th2(fo.mB) : []), ...(o.species.top === 'membrane' ? th2(fo.mT) : [])], J0 = fo.J[0] || 1e-300, Je = fo.J[fo.J.length - 1];
      plots.push({ type: 'line', title: 'Permeate flux decline during fouling', xlabel: 'Time (h)', ylabel: 'Mean flux (L/m²·h)', zeroY: true, series: [{ name: 'Mean permeate flux', x: fo.t.map((t) => t / 3600), y: fo.J.map((j) => j / LMH), mode: 'both' }] });
      plots.push({ type: 'line', title: 'Fouling layer along the membrane at the end of the period', xlabel: 'x (mm)', ylabel: 'thickness (µm) · permeability factor (–)', series: [...(o.species.bot === 'membrane' ? [{ name: 'Deposit thickness, bottom (µm)', x: xm, y: th2(fo.mB) }, { name: 'Permeability factor, bottom', x: xm, y: Array.from(r.afB) }] : []), ...(o.species.top === 'membrane' ? [{ name: 'Deposit thickness, top (µm)', x: xm, y: th2(fo.mT) }, { name: 'Permeability factor, top', x: xm, y: Array.from(r.afT) }] : [])] });
      const mMean = mean([...(o.species.bot === 'membrane' ? fo.mB : []), ...(o.species.top === 'membrane' ? fo.mT : [])]);
      kpis.push({ label: 'Flux decline over the fouling period', value: 100 * (1 - Je / J0), unit: '%', status: 1 - Je / J0 > 0.15 ? 'warn' : 'ok' }, { label: 'Mean deposit load', value: mMean * 1e3, unit: 'g/m²' }, { label: 'Mean / maximum deposit thickness', value: `${fmt(mean(th), 3)} / ${fmt(Math.max(...th), 3)}`, unit: 'µm' }, { label: 'Fouling ÷ membrane resistance', value: o.species.A * fl.mu * o.foul.alpha * mMean, unit: '' });
      tables.push({ title: 'Fouling history', columns: ['Time (h)', 'Mean flux (L/m²·h)', 'Relative flux'], rows: fo.t.map((t, k) => [t / 3600, fo.J[k] / LMH, fo.J[k] / J0]), note: 'Deposit growth per unit area: dm/dt = c_f·max(0, J − k_b|τ_w|) + N_scale. The layer adds the hydraulic resistance R_f = α·m in series with the membrane; flow, polarisation and (if enabled) precipitation are re-solved after every step, so the fields shown belong to the fouled state.' });
      outputs.fluxDecline = 1 - Je / J0; outputs.depositThickness = mean(th) * 1e-6;
      if (Math.max(...th) * 1e-6 > 0.1 * H) warn('The deposit is thicker than 10 % of the channel gap: the blockage of the flow passage, which this model does not include, is no longer negligible.');
    } else if (v.foul) info(v.species !== 'membrane' ? 'The fouling model needs membrane walls (salt transport = membrane).' : 'The fouling model runs on steady solutions only.');
    // --- Maxwell–Stefan film (ternary)
    if (v.ms) {
      if (!(sp && kDev)) info('The Maxwell–Stefan film model needs a salt-transport solution with a resolved mass-transfer coefficient.');
      else {
        const M1 = 58.44, M2 = Math.max(v.msM2, 1), c1 = sp.bulk[Math.round(nx / 2)], c2 = v.msC2, ct = ((fl.rho - c1 - c2) / 18.015 + c1 / M1 + c2 / M2) * 1e3; // mol/m³
        const xb = [(c1 / M1) * 1e3 / ct, (c2 / M2) * 1e3 / ct], J = memb ? q.Jmean : kDev * 0.5, delta = fl.D / kDev, R1 = memb && q.cwMax > 0 ? clamp(1 - q.cPerm / (q.cpMean * v.c0), 0, 1) : 1, R2 = clamp(v.msR2 / 100, 0, 1), D23 = v.msD23 * 1e-9, D12 = v.msD12 * 1e-9;
        const f = msFilm({ xb, D13: fl.D, D23, D12, ct, Jv: J, delta, rej: [R1, R2] }), fick = (xbk, D, R) => { const e = Math.exp((J * delta) / D); return (xbk * e) / (1 + (1 - R) * (e - 1)); }, w1 = fick(xb[0], fl.D, R1), w2 = fick(xb[1], D23, R2);
        const zs = f.prof.map((pt) => (delta - pt[0]) * 1e6);
        plots.push({ type: 'line', title: 'Maxwell–Stefan concentration profiles across the boundary-layer film', xlabel: 'Distance from the membrane (µm)', ylabel: 'c / c_bulk', series: [{ name: 'NaCl (Maxwell–Stefan)', x: zs, y: f.prof.map((pt) => pt[1] / xb[0]) }, { name: 'Second solute (Maxwell–Stefan)', x: zs, y: f.prof.map((pt) => pt[2] / xb[1]) }, { name: 'Second solute (independent Fick film)', x: zs, y: f.prof.map((pt) => { const e = Math.exp((J * pt[0]) / D23), ed = Math.exp((J * delta) / D23); return e / (1 + (1 - R2) * (ed - 1)); }), dash: true }] });
        tables.push({ title: 'Multicomponent (Maxwell–Stefan) film versus independent Fick films', columns: ['Solute', 'Bulk (g/L)', 'Wall, Maxwell–Stefan (g/L)', 'Wall, Fick film (g/L)', 'Polarisation factor MS', 'Polarisation factor Fick'], rows: [['NaCl', c1, (f.xw[0] / xb[0]) * c1, (w1 / xb[0]) * c1, f.xw[0] / xb[0], w1 / xb[0]], ['Second solute', c2, (f.xw[1] / xb[1]) * c2, (w2 / xb[1]) * c2, f.xw[1] / xb[1], w2 / xb[1]]],
          note: `Film thickness δ = D/k = ${fmt(delta * 1e6, 3)} µm from the CFD mass-transfer coefficient, volume flux ${fmt(J / LMH, 3)} L/m²·h, total molar concentration ${fmt(ct / 1000, 4)} kmol/m³ (taken constant). Salts are treated as neutral electrolytes; Đ(NaCl–water) = ${fmt(fl.D, 3)}, Đ(2–water) = ${fmt(D23, 3)}, Đ(1–2) = ${fmt(D12, 3)} m²/s. The solute–solute friction term is what separates the two columns.` });
        kpis.push({ label: 'Maxwell–Stefan polarisation factor, second solute', value: f.xw[1] / xb[1], unit: '', help: 'Wall ÷ bulk mole fraction from the ternary Maxwell–Stefan film' }, { label: 'Multicomponent correction to the Fick film', value: 100 * (f.xw[1] / w2 - 1), unit: '%' });
        outputs.msPolarisation = f.xw[1] / xb[1];
      }
    }
    // --- user-defined scalar
    if (r.usr) {
      const a = r.usr.phi;
      let cup = 0, qq = 0, lo = Infinity, hi = -Infinity, sIn = 0, sOut = 0, src = 0;
      for (let j = 0; j < ny; j++) { const P = j * nx + nx - 1, ue = r.u[j * r.nu1 + nx]; cup += ue * a[P] * r.dy[j]; qq += ue * r.dy[j]; sIn += r.uin[j] * o.user.in * r.dy[j]; }
      sOut = cup;
      for (let P = 0; P < n; P++) if (!r.solid[P]) { lo = Math.min(lo, a[P]); hi = Math.max(hi, a[P]); src += (r.usr.sp.S[P] - r.usr.sp.Sp[P] * a[P]) * cellArea(P); }
      plots.push({ ...base, title: 'User-defined scalar', zlabel: 'φ', zunit: 'user units', z: q.field(a), cmap: 'viridis', note: `Source expression: S = ${String(v.usrSrc).slice(0, 120)}` });
      kpis.push({ label: 'User scalar at the outlet (flow-averaged)', value: Math.abs(qq) > 0 ? cup / qq : 0, unit: 'user units' }, { label: 'User scalar range', value: `${fmt(lo, 3)} … ${fmt(hi, 3)}`, unit: 'user units' }, { label: 'Integrated user source', value: src, unit: 'units·m²/s' });
      if (o.user.bot.type === 'none' && o.user.top.type === 'none') balances.push({ name: 'User scalar (units·m²/s)', in: sIn + src, out: sOut });
      outputs.userScalarOut = Math.abs(qq) > 0 ? cup / qq : 0;
    }
    // --- second phase
    if (v.mp && v.mp !== 'off') {
      const ph = runPhase(r, v, fl), ee = ph.method === 'ee', names = { vof: 'Volume of fluid (THINC/WLIC)', ls: 'Level set', ee: 'Dispersed phase, drift-flux model (algebraic slip)' };
      const outline = ee ? [] : [v.mpInit === 'slug' ? { x: mm([ph.xb, ph.x1, ph.x1, ph.xb]), y: mm([0, 0, H, H]), closed: true, color: '#ffffff', dash: true } : { x: linspace(0, 2 * Math.PI, 41).map((t) => (ph.xb + ph.R * Math.cos(t)) * 1e3), y: linspace(0, 2 * Math.PI, 41).map((t) => (ph.yb + ph.R * Math.sin(t)) * 1e3), closed: true, color: '#ffffff', dash: true }];
      plots.push({ ...base, shapes: [...shapes, ...outline], title: ee ? 'Dispersed-phase volume fraction' : `Second-phase volume fraction after ${fmt(ph.tSim, 3)} s`, zlabel: 'α', zunit: '–', z: q.field(ph.a), zmin: 0, ...(ee ? {} : { zmax: 1 }), cmap: 'salinity', note: ee ? '' : 'The dashed outline is the initial position.' });
      plots.push({ type: 'line', title: ee ? 'Dispersed-phase hold-up in the domain' : 'Phase volume and centroid', xlabel: 'Time (s)', ylabel: ee ? 'Volume (mm² per m width)' : 'relative volume · position ÷ length', series: ee ? [{ name: 'Dispersed volume', x: ph.hist.t, y: ph.hist.vol.map((x) => x * 1e6) }] : [{ name: 'Volume ÷ initial volume', x: ph.hist.t, y: ph.hist.vol.map((x) => x / (ph.vol0 || 1e-300)) }, { name: 'Centroid x / L', x: ph.hist.t, y: ph.hist.xc.map((x) => x / L) }, { name: 'Centroid y / H', x: ph.hist.t, y: ph.hist.yc.map((y) => y / H) }] });
      if (ee) {
        const dB = ph.depB.reduce((s, x) => s + x * r.dx, 0), dT = ph.depT.reduce((s, x) => s + x * r.dx, 0), hN = ph.hist.t.length - 1, hM = Math.max(0, Math.floor(0.75 * hN)), dIn = ph.hist.inn[hN] - ph.hist.inn[hM], eta = dIn > 0 ? (ph.hist.dep[hN] - ph.hist.dep[hM]) / dIn : 0, hazen = Math.min(1, (Math.abs(ph.vsl) * L) / (r.Uref * H));
        plots.push({ type: 'line', title: 'Dispersed-phase deposition along the walls', xlabel: 'x (mm)', ylabel: 'Deposited volume per area (µm)', series: [{ name: 'Bottom wall', x: xm, y: Array.from(ph.depB, (x) => x * 1e6) }, { name: 'Top wall', x: xm, y: Array.from(ph.depT, (x) => x * 1e6) }] });
        kpis.push({ label: 'Slip (settling) velocity', value: ph.vsl * 1e6, unit: 'µm/s', help: 'Drag–buoyancy balance of the dispersed phase (Stokes), negative = toward the bottom wall; hindered by (1 − α)^4.65' }, { label: 'Dispersed phase captured on the walls', value: 100 * eta, unit: '% of inflow', help: 'Deposition rate ÷ inflow rate over the last quarter of the simulated time' }, { label: 'Ideal settler (Hazen) capture v_s L / (U H)', value: 100 * hazen, unit: '%' }, { label: 'Peak dispersed volume fraction', value: Math.max(...ph.a) * 100, unit: '%' });
        balances.push({ name: 'Dispersed phase volume (mm² per m width)', in: (ph.vol0 + ph.inn) * 1e6, out: (ph.vol + ph.out + dB + dT) * 1e6 });
        outputs.dispersedCapture = eta;
      } else {
        const k0 = 0, k1 = ph.hist.t.length - 1, vx = k1 > 0 ? (ph.hist.xc[k1] - ph.hist.xc[k0]) / (ph.hist.t[k1] - ph.hist.t[k0]) : 0, left = ph.out + ph.hist.dep[k1];
        let sharpV = 0, allV = 0;
        for (let P = 0; P < n; P++) if (!r.solid[P]) { const w = ph.a[P] * cellArea(P); allV += w; if (ph.a[P] > 0.95) sharpV += w; }
        const cellsX = v.mpInit === 'slug' ? (ph.x1 - ph.xb) / r.dx : (2 * ph.R) / r.dx;
        if (cellsX < 6) warn(`The initial ${v.mpInit === 'slug' ? 'slug' : 'bubble'} spans only ${fmt(cellsX, 2)} cells along the flow; an interface needs at least 6–8 cells to stay sharp — increase nx or the size.`);
        kpis.push({ label: 'Second-phase volume (initial → final)', value: `${fmt(ph.vol0 * 1e6, 3)} → ${fmt(ph.vol * 1e6, 3)}`, unit: 'mm²/m' }, { label: 'Phase-volume conservation error', value: ph.vol0 > 0 ? (100 * (ph.vol + left - ph.vol0)) / ph.vol0 : 0, unit: '%', status: ph.vol0 > 0 && Math.abs(ph.vol + left - ph.vol0) / ph.vol0 > 0.02 ? 'warn' : 'ok', help: ph.method === 'ls' ? 'Level-set transport is not conservative; the error measures the numerical area change' : 'Includes the volume that left through the outlet and the permeable walls' }, { label: 'Mean interface travel speed ÷ mean velocity', value: vx / r.Uref, unit: '' }, { label: 'Interface sharpness: phase volume in cells with α > 0.95', value: allV > 0 ? (100 * sharpV) / allV : 0, unit: '%', help: 'A resolved, undeformed region scores 80–95 %; shear in the channel stretches the interface and lowers it' });
        outputs.phaseVolumeError = ph.vol0 > 0 ? (ph.vol + left - ph.vol0) / ph.vol0 : 0;
        if (ph.method === 'ls' && ph.vol0 > 0 && Math.abs(ph.vol + left - ph.vol0) / ph.vol0 > 0.05) warn(`The level set changed the phase volume by ${fmt((100 * (ph.vol + left - ph.vol0)) / ph.vol0, 3)} %: sheets and threads thinner than about three cells are lost when the interface is stretched around obstacles. Refine the grid or use the volume-of-fluid model, which conserves the volume.`);
        if (!(ph.vol0 > 0)) warn('The initial second-phase region lies outside the fluid — move or enlarge it.');
      }
      if (ph.tSim < 0.999 * ph.tEnd) warn(`The phase transport stopped at ${fmt(ph.tSim, 3)} s of ${fmt(ph.tEnd, 3)} s (step limit of 12 000) — coarsen the wall refinement or shorten the simulated time.`);
      info(`${names[ph.method]}: ${ph.steps} explicit steps of ${ph.dt.toExponential(2)} s on the solved velocity field (one-way coupling: the second phase is carried by the flow and does not change it; no surface tension).`);
      sumRows.push(['Second-phase model', names[ph.method]], ['Phase-transport steps', ph.steps]);
    }
    // --- regression closure trained on solver runs
    if (v.ml) {
      const mlr = await trainClosure(v, c, ctx);
      const fx = mlr.fitF, fs = mlr.fitS, names = ['Re', ...(mlr.geo ? ['pitch/H'] : [])], law = (ft) => `${fmt(ft.a, 4)} · ${names.map((nm, k) => `${nm}^${fmt(ft.exps[k], 3)}`).join(' · ')}`;
      plots.push({ type: 'line', title: 'Regression closure: parity of learned correlations against the solver runs', xlabel: 'CFD value', ylabel: 'Regression value', logx: true, logy: true, series: [{ name: 'Friction factor: fit', x: mlr.f, y: fx.pred, mode: 'points' }, { name: 'Friction factor: leave-one-out', x: mlr.f, y: fx.loo, mode: 'points' }, ...(fs ? [{ name: 'Sherwood: fit', x: mlr.sh, y: fs.pred, mode: 'points' }, { name: 'Sherwood: leave-one-out', x: mlr.sh, y: fs.loo, mode: 'points' }] : []), { name: 'Parity', x: [Math.min(...mlr.f), Math.max(...(fs ? mlr.sh : mlr.f))], y: [Math.min(...mlr.f), Math.max(...(fs ? mlr.sh : mlr.f))], dash: true }] });
      tables.push({ title: 'Training runs for the regression closure', columns: ['Run', 'Mean velocity (m/s)', 'Re', ...(mlr.geo ? ['pitch/H'] : []), 'f (CFD)', 'f (fit)', 'f (left out)', ...(fs ? ['Sh (CFD)', 'Sh (fit)', 'Sh (left out)'] : [])], rows: mlr.X.map((x, k) => [k + 1, mlr.U[k], ...x, mlr.f[k], fx.pred[k], fx.loo[k], ...(fs ? [mlr.sh[k], fs.pred[k], fs.loo[k]] : [])]),
        note: `Learned closures on a ${mlr.nx} × ${mlr.ny} grid: f = ${law(fx)}${fs ? `; Sh = ${law(fs)}` : ''}. Ridge least squares in log space; the leave-one-out columns are predictions for a run that was withheld from the fit.` });
      kpis.push({ label: 'Learned friction law: Re exponent', value: fx.exps[0], unit: '', help: '−1 for laminar developed flow, about −0.25 to −0.3 for turbulent or spacer-filled channels' }, { label: 'Friction closure: worst leave-one-out error', value: 100 * fx.maxErrLoo, unit: '%', status: fx.maxErrLoo > 0.15 ? 'warn' : 'ok' });
      if (fs) {
        const sML = fs.predict([q.Re, ...(mlr.geo ? [c.geo.lm / H] : [])]), fML = fx.predict([q.Re, ...(mlr.geo ? [c.geo.lm / H] : [])]), one2 = channel1D({ ...v, Uin: r.Uref, ksh: (v.ksh ?? 1) * (sML / (one.Sh / (v.ksh ?? 1))), kdp: (v.kdp ?? 1) * (fML / (one.f / (v.kdp ?? 1))) });
        kpis.push({ label: 'Learned Sherwood law: Re exponent', value: fs.exps[0], unit: '' }, { label: 'Sherwood closure: worst leave-one-out error', value: 100 * fs.maxErrLoo, unit: '%', status: fs.maxErrLoo > 0.15 ? 'warn' : 'ok' }, { label: '1-D model with the learned closure: flux', value: one2.flux, unit: 'L/m²·h', help: 'Film-theory channel model with the regression Sh and f in place of the literature correlations' }, { label: '1-D model with the learned closure: polarisation', value: one2.CP, unit: '' });
        outputs.mlSherwood = sML; outputs.mlShExponent = fs.exps[0];
      }
      outputs.mlFrictionExponent = fx.exps[0];
      info(`Regression closure trained on ${mlr.X.length} additional solver runs (${mlr.nx} × ${mlr.ny} cells each); cross-validated R² = ${fmt(fx.r2loo, 4)} for friction${fs ? ` and ${fmt(fs.r2loo, 4)} for the Sherwood number` : ''}. It is valid only inside the sampled range of Reynolds number${mlr.geo ? ' and pitch' : ''}.`);
    }
    for (const k of Object.keys(outputs)) if (!Number.isFinite(outputs[k])) delete outputs[k];
    return {
      summary: `Re = ${fmt(q.Re, 3)}: pressure gradient ${fmt(q.dpdx / 100, 3)} mbar/m (f = ${fmt(q.f, 3)}), mean wall shear ${fmt(q.tauMean, 3)} Pa, ${fmt(100 * q.recirc, 3)} % recirculating area` + (sp && shDev ? `, Sherwood number ${fmt(shDev, 3)}` : '') + (memb ? `, polarisation factor ${fmt(q.cpMean, 3)} at ${fmt(q.Jmean / LMH, 3)} L/m²·h.` : '.'),
      kpis, warnings: W,
      recommendations: [
        !r.converged && o.steady ? 'Switch the time treatment to transient: unconverged steady residuals behind filaments usually mean vortex shedding.' : null,
        memb && q.cpMean > v.limCP ? 'Reduce polarisation by raising the cross-flow velocity, lowering the flux (pressure) or using a zigzag arrangement with a shorter pitch.' : null,
        q.recirc > 0.2 ? 'Large recirculation zones sit behind the obstacles: they trap foulants and salt. Try a larger pitch-to-height ratio or submerged filaments and compare the wall-shear profiles.' : null,
        Math.abs(q.dpdx) / 1e5 > v.limDP ? 'Lower the velocity or use a thicker spacer: the pressure gradient exceeds the vessel limit.' : null,
        kMult ? `Use the mass-transfer multiplier ${fmt(clamp(kMult, 0.2, 5), 3)} in suite 1 (RO design) — it is offered there automatically.` : null,
        'Run the three-level grid study on the Mesh tab and quote the grid-convergence index with the results.',
        'Calibrate the friction and mass-transfer multipliers of the 1-D model against channel or element test data on the Calibrate tab.',
      ].filter(Boolean),
      plots, tables, balances, outputs,
    };
  },

  mesh: [{ name: 'Spatial grid (nx × ny)', keys: ['nx', 'ny'], min: 12, note: 'Both directions are refined together with the wall-clustering ratio held constant. All other inputs are unchanged.',
    metrics: [{ label: 'Pressure gradient', unit: 'Pa/m', get: (r) => r.outputs.dpPerM ?? 0 }, { label: 'Mean Sherwood number', unit: '–', get: (r) => r.outputs.sherwood ?? 0 }, { label: 'Maximum wall concentration', unit: 'g/L', get: (r) => r.outputs.maxWallConc ?? 0 }] },
    { name: 'Multiphase studies: free-interface grid (study = two-phase flow)', keys: ['tpNx', 'tpNy'], min: 16, note: 'Applies when the study type is the free-interface two-phase flow; interface problems converge at about first order.', metrics: [{ label: 'Largest velocity', unit: 'm/s', get: (r) => r.outputs.maxVelocity ?? 0 }, { label: 'Front position x/a (dam break)', unit: '–', get: (r) => r.outputs.frontPosition ?? 0 }, { label: 'Rise velocity (bubble)', unit: 'm/s', get: (r) => r.outputs.riseVelocity ?? 0 }] },
    { name: 'Multiphase studies in a channel geometry: cells across the gap, free interface (domain = spacer channel or imported section)', keys: ['tpChNy'], min: 6, note: 'Applies to the free-interface study in a channel domain; the cells stay square.', metrics: [{ label: 'Gas velocity ÷ mean liquid velocity', unit: '–', get: (r) => (r.outputs.slugVelocity ?? 0) }, { label: 'Largest pressure drop', unit: 'Pa', get: (r) => r.outputs.pressureDropMax ?? 0 }] },
    { name: 'Multiphase studies in a channel geometry: cells across the gap, two-fluid model (domain = spacer channel or imported section)', keys: ['tfChNy'], min: 6, note: 'Applies to the two-fluid study in a channel domain; the cells stay square.', metrics: [{ label: 'Capture efficiency', unit: '–', get: (r) => r.outputs.captureEfficiency ?? 0 }, { label: 'Pressure drop', unit: 'Pa', get: (r) => r.outputs.pressureDrop ?? 0 }] },
    { name: 'Multiphase studies: two-fluid grid (study = two-fluid model)', keys: ['tfNx', 'tfNy'], min: 3, note: 'Applies when the study type is the Eulerian–Eulerian two-fluid model.', metrics: [{ label: 'Front velocity (closed column)', unit: 'm/s', get: (r) => r.outputs.frontVelocity ?? 0 }, { label: 'Capture efficiency (flow-through)', unit: '–', get: (r) => r.outputs.captureEfficiency ?? 0 }] }],

  calibration: {
    note: 'Fit the 1-D channel model (friction and Sherwood correlations with film-theory polarisation) to flat-sheet cell or element data: each row is one operating point with the cross-flow velocity, trans-membrane pressure and feed concentration; the measurements are the pressure gradient and the permeate flux. The fitted multipliers express how the real 3-D spacer departs from the correlations; compare them with the CFD ratios in the results table.',
    params: [{ key: 'kdp', label: 'Friction multiplier', lo: 0.2, hi: 8 }, { key: 'ksh', label: 'Mass-transfer multiplier', lo: 0.2, hi: 5 }, { key: 'A', label: 'Water permeability A', lo: 0.2, hi: 12 }],
    columns: [{ key: 'Uin', label: 'Cross-flow velocity', unit: 'm/s' }, { key: 'dPtm', label: 'Trans-membrane pressure', unit: 'bar' }, { key: 'c0', label: 'Feed concentration', unit: 'g/L' }, { key: 'dpPerM', label: 'Pressure gradient', unit: 'mbar/m' }, { key: 'flux', label: 'Permeate flux', unit: 'L/m²·h' }],
    targets: [{ key: 'dpPerM', label: 'Pressure gradient', unit: 'mbar/m' }, { key: 'flux', label: 'Permeate flux', unit: 'L/m²·h' }],
    model(v) { const m = channel1D(v); return { dpPerM: m.dpPerM / 100, flux: m.flux }; },
    get sample() { return (this._s ||= synth(5, [[0.06, 55, 35], [0.1, 55, 35], [0.15, 55, 35], [0.2, 60, 35], [0.25, 60, 38], [0.1, 65, 40], [0.15, 50, 32], [0.3, 62, 35]])); },
    get validationSample() { return (this._v ||= synth(23, [[0.08, 58, 35], [0.12, 52, 33], [0.18, 64, 37], [0.22, 56, 35], [0.28, 60, 39], [0.14, 68, 42]])); },
  },

  async verify() {
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Number.isFinite(got) && Math.abs(got - expected) <= tol, note });
    const rho = 1000, mu = 1e-3, H = 1e-3, U = 0.05, base = { H, rho, mu, Uin: U, scheme: 'hybrid', tol: 1e-8, maxIter: 1500 };
    const colP = (r, i) => { let s = 0, m = 0; for (let j = 0; j < r.ny; j++) if (!r.solid[j * r.nx + i]) { s += r.p[j * r.nx + i] * r.dy[j]; m += r.dy[j]; } return s / m; };
    const grad = (r) => (colP(r, 2) - colP(r, r.nx - 3)) / ((r.nx - 5) * r.dx), exact = (12 * mu * U) / H ** 2;
    // 1–2: plane Poiseuille flow
    const p1 = await solveChannel({ ...base, L: 6e-3, nx: 24, ny: 24, inlet: 'parabolic' });
    add('Poiseuille pressure gradient', 1, grad(p1) / exact, 0.006, '−dp/dx = 12 μ U / H² for fully developed plane-channel flow (ratio)');
    const p2 = await solveChannel({ ...base, L: 12e-3, nx: 40, ny: 24, inlet: 'uniform' });
    let um = 0; for (let j = 0; j < 24; j++) um = Math.max(um, p2.u[j * p2.nu1 + 36]);
    add('Developed centre-line velocity from a uniform inlet', 1.5, um / U, 0.015, 'u_max = 1.5 U after the hydrodynamic entrance length (Re = 100)');
    // 3: grid-convergence order with Richardson extrapolation
    const gs = [];
    for (const ny of [32, 16, 8]) gs.push(grad(await solveChannel({ ...base, L: 2e-3, nx: 10, ny, inlet: 'parabolic' })));
    const g = gci([1 / 32, 1 / 16, 1 / 8], gs);
    add('Observed order of accuracy (Poiseuille, 8/16/32 cells)', 2, g.p, 0.25, 'Second-order central diffusion; Richardson extrapolation with the grid-convergence index');
    add('Richardson-extrapolated pressure gradient', 1, g.fExact / exact, 2e-3, 'Extrapolated value against the analytical solution (ratio)');
    // 4: zero-flow limit
    const z = await solveChannel({ ...base, L: 4e-3, nx: 16, ny: 10, Uin: 0, inlet: 'uniform', maxIter: 20 });
    add('Zero-flow limit', 0, Math.max(...z.u.map(Math.abs), ...z.v.map(Math.abs), ...z.p.map(Math.abs)), 1e-12, 'No inflow → fluid at rest with uniform pressure');
    // 5–6: obstacle case — mass conservation and symmetry
    const nx = 48, ny = 20, gy = yGrid(H, ny, 1), mk = buildMask({ type: 'spacer', arr: 'submerged', L: 6e-3, H, df: 0.4e-3, lm: 3e-3, nFil: 2 }, nx, ny, gy.yc);
    const s = await solveChannel({ ...base, L: 6e-3, nx, ny, solid: mk.solid, inlet: 'parabolic', tol: 1e-7 });
    let qo = 0; for (let j = 0; j < ny; j++) qo += s.u[j * s.nu1 + nx] * s.dy[j];
    add('Global mass conservation with immersed filaments', 0, (s.Qin - qo) / s.Qin, 1e-6, '(inflow − outflow)/inflow');
    let asym = 0; for (let j = 0; j < ny / 2; j++) for (let i = 0; i <= nx; i++) asym = Math.max(asym, Math.abs(s.u[j * s.nu1 + i] - s.u[(ny - 1 - j) * s.nu1 + i]));
    add('Symmetric solution for a symmetric geometry', 0, asym / U, 2e-3, 'max |u(y) − u(H − y)| / U for mid-channel filaments');
    // 7: pure diffusion between two walls at different concentration
    const d = await solveChannel({ ...base, L: 2e-3, nx: 8, ny: 16, stretch: 6, Uin: 0, inlet: 'uniform', maxIter: 5, species: { c0: 1.5, D: 1.5e-9, bot: 'fixed', top: 'fixed', cwBot: 2, cwTop: 1 }, scalIter: 60 });
    let dev = 0; for (let j = 0; j < 16; j++) dev = Math.max(dev, Math.abs(d.spc.phi[j * 8 + 4] - (2 - d.yc[j] / H)));
    add('Pure-diffusion limit: linear concentration profile', 0, dev, 1e-6, 'c(y) = c_bottom + (c_top − c_bottom) y/H with no flow');
    // 8: Lévêque mass transfer (fixed wall concentration, developed flow, Sc = 667)
    const D = 1.5e-9, lv = await solveChannel({ ...base, L: 10e-3, nx: 50, ny: 40, stretch: 40, inlet: 'parabolic', tol: 1e-7, species: { c0: 1, D, bot: 'fixed', top: 'none', cw: 2 }, scalIter: 200 });
    const il = 40, xl = (il + 0.5) * lv.dx, P = il, cwl = 2; let qf = 0, qq = 0;
    for (let j = 0; j < 40; j++) { const uc = 0.5 * (lv.u[j * lv.nu1 + il] + lv.u[j * lv.nu1 + il + 1]); qf += uc * lv.spc.phi[j * 50 + il] * lv.dy[j]; qq += uc * lv.dy[j]; }
    const kl = (D * (cwl - lv.spc.phi[P])) / (0.5 * lv.dy[0]) / (cwl - qf / qq), shEx = 1.2326 * (xl / (2 * H * ((rho * U * 2 * H) / mu) * (mu / (rho * D)))) ** (-1 / 3);
    add('Lévêque local Sherwood number', 1, (kl * 2 * H) / D / shEx, 0.03, 'Sh_x = 1.233 (x / d_h Re Sc)^(−1/3), constant wall concentration (ratio)');
    // 9: membrane wall — salt balance and film-theory consistency
    const pi = (c) => 0.76e5 * c, m = await solveChannel({ ...base, L: 8e-3, nx: 40, ny: 32, stretch: 30, Uin: 0.1, inlet: 'parabolic', tol: 1e-7, species: { c0: 35, D, A: (1.2 * LMH) / 1e5, B: 0.06 * LMH, dP: 55e5, pi, bot: 'membrane', top: 'membrane' }, scalIter: 300 });
    let sin = 0, sout = 0; for (let j = 0; j < 32; j++) { sin += m.uin[j] * 35 * m.dy[j]; sout += m.u[j * m.nu1 + 40] * m.spc.phi[j * 40 + 39] * m.dy[j]; }
    for (let i = 0; i < 40; i++) sout += (m.Jb[i] * m.spc.pB[i] + m.Jt[i] * m.spc.pT[i]) * m.dx;
    add('Salt balance with permeating membrane walls', 0, (sin - sout) / sin, 2e-4, '(salt in − salt out − salt in permeate)/salt in');
    const i9 = 30, Jx = m.Jb[i9], cwx = m.spc.wB[i9], cpx = m.spc.pB[i9];
    add('Local flux obeys Jw = A(ΔP − Δπ(c_wall))', 1, Jx / (((1.2 * LMH) / 1e5) * (55e5 + m.p[i9] - colP(m, 0) - (pi(cwx) - pi(cpx)))), 1e-3, 'Solution–diffusion wall condition recovered from the converged field (ratio)');
    // 10: Darcy–Brinkman porous channel
    const K = 1e-10, pz = await solveChannel({ ...base, L: 4e-3, nx: 16, ny: 40, stretch: 30, Uin: 0.01, inlet: 'uniform', porous: { x0: -1, x1: 1, K, cF: 0 } });
    const a = H / (2 * Math.sqrt(K)), brink = ((mu * 0.01) / K) / (1 - Math.tanh(a) / a);
    add('Darcy–Brinkman porous channel pressure gradient', 1, ((colP(pz, 6) - colP(pz, 12)) / (6 * pz.dx)) / brink, 0.005, '−dp/dx = μU/K ÷ (1 − tanh(a)/a), a = H/(2√K) (ratio)');
    // 11: developed Nusselt number, constant wall temperature
    const al = 1.43e-7, ht = await solveChannel({ ...base, L: 30e-3, nx: 60, ny: 24, Uin: 0.01, inlet: 'parabolic', energy: { alpha: al, Tin: 60, bot: { type: 'fixed', val: 20 }, top: { type: 'fixed', val: 20 } }, scalIter: 200 });
    let hf = 0, hq = 0; const ih = 50;
    for (let j = 0; j < 24; j++) { const uc = 0.5 * (ht.u[j * ht.nu1 + ih] + ht.u[j * ht.nu1 + ih + 1]); hf += uc * ht.eng.phi[j * 60 + ih] * ht.dy[j]; hq += uc * ht.dy[j]; }
    add('Fully developed Nusselt number, isothermal walls', 7.541, ((20 - ht.eng.phi[ih]) / (0.5 * ht.dy[0]) / (20 - hf / hq)) * 2 * H, 0.05, 'Nu = 7.541 for plane Poiseuille flow (d_h = 2H)');
    // 12: 1-D reference model reduces to pure-water flux without salt
    const dflt = Object.fromEntries(suite.inputs.flatMap((gq) => gq.fields).map((f) => [f.key, f.value]));
    add('1-D model: pure-water flux equals A·ΔP', dflt.A * 20, channel1D({ ...dflt, c0: 0, dPtm: 20 }).flux, 1e-6, 'Zero-solute limit of the solution–diffusion/film model');
    // ---- extended models
    const gradX = (r, ia, ib) => (colP(r, ia) - colP(r, ib)) / ((ib - ia) * r.dx);
    { // 13–17: two-equation closures on fully developed turbulent plane-channel flow (Re_H = 50 000), Dean's correlation and the log law
      const Ht = 0.05, Ut = 1, ReH = (rho * Ut * Ht) / mu, cfD = 0.073 * ReH ** -0.25, nxt = 8, nyt = 20;
      for (const [tm, nm, tolC] of [['ke', 'standard k–ε', 0.08], ['kw', 'k–ω', 0.08], ['sst', 'k–ω SST', 0.08], ['earsm', 'algebraic Reynolds stress', 0.08]]) {
        const t = await solveChannel({ H: Ht, L: 10 * Ht, nx: nxt, ny: nyt, rho, mu, Uin: Ut, inlet: 'periodic', scheme: 'hybrid', tol: 3e-6, maxIter: 1500, turb: tm, alphaU: 0.7, ...(tm === 'sst' ? { energy: { alpha: 1.43e-7, Tin: 20, bot: { type: 'fixed', val: 60 }, top: { type: 'fixed', val: 60 } }, scalIter: 300 } : {}) });
        const tw = (gradX(t, 2, nxt - 2) * Ht) / 2, ut = Math.sqrt(tw / rho), ic = nxt - 2, jl = 3, Pl = jl * nxt + ic;
        add(`Turbulent channel, ${nm}: skin friction vs Dean`, 1, tw / (0.5 * rho * Ut * Ut) / cfD, tolC, 'C_f = 0.073 Re_H^−0.25 (Dean 1978), periodic channel with 20 cells across, first cell at y⁺ ≈ 60 (ratio)');
        if (tm === 'sst') add('Turbulent channel, k–ω SST: log-law velocity at y⁺ ≈ 180', 1, (0.5 * (t.u[jl * t.nu1 + ic] + t.u[jl * t.nu1 + ic + 1])) / ut / (Math.log((t.yc[jl] * ut * rho) / mu) / KAPPA + 5.2), 0.06, 'u⁺ = ln(y⁺)/0.41 + 5.2 (ratio)');
        if (tm === 'sst') {
          let hq = 0, hf = 0; for (let j = 0; j < nyt; j++) { const uc = 0.5 * (t.u[j * t.nu1 + ic] + t.u[j * t.nu1 + ic + 1]); hq += uc * t.dy[j]; hf += uc * t.eng.phi[j * nxt + ic] * t.dy[j]; }
          add('Turbulent heat transfer with the scalar law of the wall: Nusselt number', 1, ((t.eng.fB[ic] / (60 - hf / hq)) * 2 * Ht) / 1.43e-7 / (0.023 * (2 * ReH) ** 0.8 * 7 ** 0.4), 0.25, 'Local Nu at 4 hydraulic diameters against Dittus–Boelter 0.023 Re^0.8 Pr^0.4 (Pr = 7); the thermal entrance raises it by 10–20 % (ratio)');
        }
        if (tm === 'kw') { // periodic (recycled) inlet: the inlet plane carries the developed profile and the pressure gradient is uniform along the channel
          let dp = 0; for (let j = 0; j < nyt; j++) dp = Math.max(dp, Math.abs(t.u[j * t.nu1] - t.u[j * t.nu1 + nxt - 1]));
          add('Periodic inlet: inlet-plane profile equals the developed profile downstream', 0, dp / Ut, 2e-3, 'max |u(inlet, y) − u(last cell face, y)| / U after the recycling has converged (turbulent channel, k–ω)');
          add('Periodic inlet: pressure gradient is uniform along the channel', 1, gradX(t, 1, 4) / gradX(t, 4, nxt - 1), 0.01, 'Upstream ÷ downstream gradient: no entrance region remains (ratio)');
        }
        if (tm === 'ke') add('Turbulent channel, k–ε: structure parameter −u′v′/k in the log layer', 0.3, -t.rs.uv[Pl] / t.tk[Pl], 0.03, '√C_μ = 0.30 where production balances dissipation');
        if (tm === 'earsm') add('Algebraic Reynolds stress: normal-stress anisotropy (u′u′ − v′v′)/2k', 0.26, (t.rs.uu[Pl] - t.rs.vv[Pl]) / (2 * t.tk[Pl]), 0.08, 'Log-layer value of the Wallin–Johansson closure (0 for an eddy-viscosity model; experiments ≈ 0.25–0.30)');
      }
    }
    { // LES: the Smagorinsky term must vanish on a resolved laminar flow
      const l = await solveChannel({ ...base, L: 2e-3, nx: 10, ny: 16, inlet: 'parabolic', turb: 'les', steady: false, tEnd: (2 * 2e-3) / U, cfl: 2, maxIter: 400 });
      let nt = 0; for (let P = 0; P < l.mue.length; P++) nt = Math.max(nt, l.mue[P] / mu - 1);
      add('LES (Smagorinsky): laminar limit keeps the Poiseuille pressure gradient', 1, gradX(l, 2, 7) / exact, 0.02, `Transient filtered equations at Re = 100; largest sub-grid viscosity ratio ${nt.toExponential(1)} (ratio)`);
      const jS = 4, yS = l.yc[jS], dS = Math.min(yS, H - yS), lmS = Math.min(KAPPA * dS, 0.17 * Math.sqrt(l.dx * l.dy[jS])) * (1 - Math.exp((-dS * l.utau) / (mu / rho) / 26)), SS = ((6 * U) / H) * Math.abs(1 - (2 * yS) / H);
      add('LES (Smagorinsky): sub-grid viscosity equals (C_s Δ f_vD)² |S|', 1, (l.mue[jS * 10 + 6] / mu - 1) / ((rho * lmS * lmS * SS) / mu), 0.02, 'Poiseuille shear |S| = (6U/H)|1 − 2y/H|, Δ = √(ΔxΔy), van Driest damping with the computed friction velocity (ratio)');
    }
    { // creeping flow: pressure drop proportional to flow rate around an obstacle
      const gy2 = yGrid(H, 12, 1), mk2 = buildMask({ type: 'spacer', arr: 'submerged', L: 2e-3, H, df: 0.4e-3, lm: 2e-3, nFil: 1 }, 16, 12, gy2.yc), oc = { ...base, L: 2e-3, nx: 16, ny: 12, solid: mk2.solid, inlet: 'parabolic', creeping: true, tol: 1e-7 };
      const c1 = await solveChannel({ ...oc, Uin: 0.2 }), c2 = await solveChannel({ ...oc, Uin: 0.4 });
      add('Stokes limit: pressure drop doubles with the flow rate', 2, (colP(c2, 1) - colP(c2, 14)) / (colP(c1, 1) - colP(c1, 14)), 2e-3, 'Creeping-flow option around filaments at a nominal Re of 400–800, where the full equations give a ratio well above 2');
    }
    { // wall and inlet conditions
      const bs = 1e-4, sl = await solveChannel({ ...base, tol: 1e-7, Uin: 0.005, L: 4e-3, nx: 16, ny: 16, inlet: 'parabolic', wallB: 'slip', wallT: 'slip', slipLen: bs });
      add('Navier-slip walls: Poiseuille pressure gradient', 1, gradX(sl, 9, 13) / (exact / 10 / (1 + (6 * bs) / H)), 0.006, '−dp/dx = 12 μU / (H² (1 + 6b/H)) with slip length b = H/10 (ratio)');
      const sy = await solveChannel({ ...base, tol: 1e-7, Uin: 0.002, L: 8e-3, nx: 20, ny: 20, inlet: 'parabolic', wallT: 'sym' });
      add('Symmetry plane: half-channel pressure gradient', 1, gradX(sy, 13, 17) / ((3 * mu * 0.002) / H ** 2), 0.006, '−dp/dx = 3 μU/h² for a half channel of height h (ratio)');
      let usym = 0; for (let i = 13; i < 17; i++) usym += 0.5 * (sy.u[19 * sy.nu1 + i] + sy.u[19 * sy.nu1 + i + 1]) / 4;
      add('Symmetry plane: centre-line velocity', 1.5, usym / 0.002 / (1 - 1 / (3 * 40 * 40)), 0.01, 'u = 1.5 U on the symmetry plane (cell-centre value corrected to the plane)');
      const dpT = 5, pi2 = await solveChannel({ ...base, L: 6e-3, nx: 24, ny: 16, inlet: 'parabolic', pInlet: dpT, Uin: 0.02, tol: 1e-7 });
      add('Pressure inlet: the imposed inlet pressure is reached', 1, colP(pi2, 0) / dpT, 1e-3, 'Section-mean static pressure of the first cell column ÷ target after the flow-rate controller has converged');
      add('Pressure inlet: resulting flow rate', 1, pi2.Uref / ((dpT * H * H) / (12 * mu * 6e-3)), 0.06, 'U = Δp H² / (12 μ L); the pressure reference sits within one cell of the outlet plane (ratio)');
      const cm = caseConfig({ ...dflt, geom: 'plain', inletBC: 'massflow', mdot: 0.05 });
      add('Mass-flow inlet: ρ·U·H equals the specified mass flow', 0.05, cm.fl.rho * cm.Uin * cm.H, 1e-12, 'kg/s per metre of channel width');
      const ce = caseConfig({ ...dflt, geom: 'plain', porous: true, porModel: 'ergun', porDp: 0.5, porEps: 0.4 }).o.porous, Ue = 0.05;
      const eg = await solveChannel({ ...base, Uin: Ue, L: 4e-3, nx: 16, ny: 12, inlet: 'uniform', wallB: 'sym', wallT: 'sym', porous: { x0: -1, x1: 1, K: ce.K, cF: ce.cF } });
      add('Ergun equation: packed-bed pressure gradient', 1, gradX(eg, 4, 12) / ((150 * mu * 0.36 * Ue) / (2.5e-7 * 0.064) + (1.75 * rho * 0.6 * Ue * Ue) / (5e-4 * 0.064)), 2e-3, '150 μ(1−ε)²U/(d_p²ε³) + 1.75 ρ(1−ε)U²/(d_p ε³), d_p = 0.5 mm, ε = 0.4, free-slip walls (ratio)');
      const jw = 2e-6, fx = await solveChannel({ ...base, L: 6e-3, nx: 24, ny: 20, stretch: 4, inlet: 'parabolic', species: { c0: 1, D: 1.5e-9, bot: 'flux', top: 'none', cw: jw }, scalIter: 300 });
      let si = 0, so = 0; for (let j = 0; j < 20; j++) { si += fx.uin[j] * fx.dy[j]; so += fx.u[j * fx.nu1 + 24] * fx.spc.phi[j * 24 + 23] * fx.dy[j]; }
      add('Specified wall species flux: salt gained equals flux × wall length', 1, (so - si) / (jw * 6e-3), 2e-3, '(outflow − inflow) ÷ (J_w L) (ratio)');
    }
    { // pressure inlet on a turbulent duct of 50 mm × 600 mm (60 Pa): every closure must find the flow rate, and the friction must be Dean's
      const Hd = 0.05, Ld = 0.6, dpD = 60, nxd = 16, nyd = 24;
      for (const [tm, nm] of [['ke', 'k–ε'], ['sst', 'k–ω SST'], ['rsm', 'Reynolds-stress transport']]) {
        const cd = caseConfig({ ...dflt, geom: 'plain', H: 50, L: 600, species: 'off', propMode: 'custom', rho, mu: 1, turb: tm, inletBC: 'pressure', pInlet: dpD, nx: nxd, ny: nyd, stretch: 2, alphaU: 0.7 }), t = await solveChannel(cd.o);
        const ReH = (rho * t.Uref * Hd) / mu, gD = (0.073 * ReH ** -0.25 * rho * t.Uref ** 2) / Hd;
        add(`Pressure inlet, turbulent duct 50 mm × 600 mm, ${nm}: inlet pressure reached`, 1, t.converged ? colP(t, 0) / dpD : NaN, 2e-3, `Section-mean static pressure of the first cell column ÷ the 60 Pa target after the flow-rate controller has converged (${t.iters} iterations, U = ${fmt(t.Uref, 4)} m/s, Re_H = ${fmt(ReH, 3)}); inlet profile ${cd.o.inlet === 'periodic' ? 'recycled from the outlet (fully developed)' : cd.o.inlet}`);
        add(`Pressure inlet, turbulent duct 50 mm × 600 mm, ${nm}: pressure gradient vs Dean`, 1, gradX(t, 2, nxd - 3) / gD, 0.08, '−dp/dx = 2 τ_w/H with C_f = 0.073 Re_H^−0.25 (Dean 1978) at the flow rate the controller found; 16 × 24 cells, wall-function grid (ratio)');
      }
      let msg = ''; try { await solveChannel({ H: Hd, L: Ld, nx: nxd, ny: nyd, stretch: 2, rho, mu, Uin: 0.5, pInlet: dpD, inlet: 'parabolic', scheme: 'hybrid', tol: 1e-5, maxIter: 300, turb: 'ke', alphaU: 0.7 }); } catch (e) { msg = e.message; }
      add('Pressure inlet: an unreachable target is reported, not run away with', 1, /was not reached/.test(msg) && /upper bound/.test(msg) ? 1 : 0, 0, 'A laminar parabolic profile forced on the same turbulent duct flattens downstream and recovers more static pressure than friction consumes, so the inlet static pressure is bounded (about 40 Pa here); the controller lowers the flow when the inlet pressure turns negative and the run ends with a message naming the bound (before the fix it multiplied the flow by 1.4 without limit)');
    }
    { // conjugate heat transfer: conduction through a solid layer and a stagnant fluid layer in series
      const nxc = 8, nyc = 20, sol = new Uint8Array(nxc * nyc); for (let j = 0; j < nyc / 2; j++) for (let i = 0; i < nxc; i++) sol[j * nxc + i] = 1;
      const al2 = 1.5e-7, ch = await solveChannel({ ...base, Uin: 0, L: 2e-3, nx: nxc, ny: nyc, solid: sol, inlet: 'uniform', maxIter: 5, energy: { alpha: al2, Tin: 50, solidD: 4 * al2, bot: { type: 'fixed', val: 80 }, top: { type: 'fixed', val: 20 } }, scalIter: 400 });
      let devT = 0; for (let j = 0; j < nyc; j++) { const y = ch.yc[j] / H, Tex = y < 0.5 ? 80 - 24 * y : 68 - 96 * (y - 0.5); devT = Math.max(devT, Math.abs(ch.eng.phi[j * nxc + 4] - Tex)); }
      add('Conjugate heat transfer: two-layer conduction profile', 0, devT, 1e-4, 'Solid (k_s = 4 k_f) and fluid layers of equal thickness between 80 and 20 °C: interface at 68 °C (largest deviation, K)');
    }
    { // lattice-Boltzmann
      const nxl = 32, nyl = 12, uL = 0.05, tauL = 0.8, lb = lbmD2Q9({ nx: nxl, ny: nyl, solid: new Uint8Array(nxl * nyl), uin: Float64Array.from({ length: nyl }, (_, j) => { const q = (j + 0.5) / nyl; return 6 * uL * q * (1 - q); }), tau: tauL, maxSteps: 6000, tol: 1e-7 });
      const pmL = (i) => { let a = 0; for (let j = 0; j < nyl; j++) a += (lb.rho[j * nxl + i] - 1) / 3; return a / nyl; };
      add('Lattice-Boltzmann (D2Q9 BGK): Poiseuille pressure gradient', 1, (pmL(8) - pmL(24)) / 16 / ((12 * ((tauL - 0.5) / 3) * uL) / nyl ** 2), 0.01, '−dp/dx = 12 ν U / H² in lattice units, 12 nodes across the gap, τ = 0.8 (ratio)');
      const fvP = p1, dev = Math.abs((pmL(8) - pmL(24)) / 16 / ((12 * ((tauL - 0.5) / 3) * uL) / nyl ** 2) - grad(fvP) / exact);
      add('Lattice-Boltzmann vs finite volume on plane Poiseuille flow', 0, dev, 0.01, 'Difference of the two normalised pressure gradients');
    }
    { // compressible flow
      const Ls = { rho: 1, u: 0, p: 1 }, Rs = { rho: 0.125, u: 0, p: 0.1 }, ns = 200, sod = euler1D({ n: ns, L: 1, gam: 1.4, Rg: 1, init: (x) => (x < 0.5 ? Ls : Rs), tEnd: 0.2 });
      let e1 = 0; for (let i = 0; i < ns; i++) e1 += Math.abs(sod.rho[i] - riemannExact(Ls, Rs, 1.4, (sod.x[i] - 0.5) / 0.2)[0]) / ns;
      add('Euler equations: Sod shock tube, L1 density error', 0, e1, 0.006, 'MUSCL–HLLC on 200 cells at t = 0.2 against the exact Riemann solution');
      add('Exact Riemann solver: Sod star pressure', 0.30313, riemannExact(Ls, Rs, 1.4, 0)[3], 2e-5, 'Reference value from Toro (2009)');
      add('Euler equations: mass conservation in the shock tube', 0, sod.mass + sod.mOut - sod.mIn - 0.5625, 1e-12, 'Total mass before the waves reach the ends');
      const gm = 1.4, Rg = 287, p0 = 3e5, T0 = 400, ar = (x) => (x < 0.4 ? 1 + 1.5 * 0.5 * (1 + Math.cos((Math.PI * x) / 0.4)) : 1 + 0.5 * (1 - Math.cos((Math.PI * (x - 0.4)) / 0.6))), nn = 80;
      const nz = euler1D({ n: nn, L: 1, gam: gm, Rg, area: ar, init: (x) => ({ rho: p0 / (Rg * T0), u: 10, p: p0 * (1 - 0.9 * x) }), left: { p0, T0 }, right: { pb: 0.1 * p0 }, steadyFlux: 1e-3, maxSteps: 4000, cfl: 0.8 });
      const exn = nozzleExact(nz.x.map(ar), 31, 0.1, gm), mch = p0 * Math.sqrt(gm / (Rg * T0)) * (2 / (gm + 1)) ** ((gm + 1) / (2 * (gm - 1)));
      add('Quasi-1-D nozzle: choked mass flow', 1, nz.mdot[40] / mch, 0.005, 'ṁ* = A_t p0 √(γ/RT0) (2/(γ+1))^((γ+1)/(2(γ−1))) (ratio)');
      add('Quasi-1-D nozzle: supersonic exit Mach number', exn.M[nn - 1], nz.M[nn - 1], 0.02, 'Area–Mach relation for A_e/A_t = 2');
    }
    { // Maxwell–Stefan film, user expressions, regression closure
      const Jm = 1e-5, Dm = 1.5e-9, dl = 5e-5, f1 = msFilm({ xb: [0.01, 1e-9], D13: Dm, D23: 1e-9, D12: 1e-9, ct: 55500, Jv: Jm, delta: dl, rej: [1, 1] });
      add('Maxwell–Stefan film: binary Fick limit', 1, f1.xw[0] / 0.01 / Math.exp((Jm * dl) / Dm), 1e-8, 'Trace second solute: x_w/x_b = exp(J δ / D) of film theory (ratio)');
      const f2 = msFilm({ xb: [0.01, 0.004], D13: Dm, D23: 1e-9, D12: 1e-3, ct: 55500, Jv: Jm, delta: dl, rej: [1, 1] });
      add('Maxwell–Stefan film: uncoupled limit for large Đ₁₂', 1, f2.xw[1] / 0.004 / Math.exp((Jm * dl) / 1e-9), 0.012, 'Without solute–solute friction each solute follows its own Fick film up to the finite-concentration (x₃ < 1) correction (ratio)');
      add('Expression parser', 14, compileExpr('2*exp(-x)+3^2 - -1 + max(1, 2, phi)/2', ['x', 'phi'])({ x: 0, phi: 4 }), 1e-12, '2·e⁰ + 9 + 1 + 4/2');
      const kd = 5, Lu = 10e-3, us = await solveChannel({ ...base, scheme: 'quick', L: Lu, nx: 80, ny: 6, inlet: 'uniform', wallB: 'sym', wallT: 'sym', user: { D: 1e-12, in: 1, scale: 1, bot: { type: 'none' }, top: { type: 'none' }, src: (phi) => -kd * phi }, scalIter: 400 });
      add('User-defined scalar: first-order decay in plug flow', Math.exp((-kd * (Lu - 0.5 * us.dx)) / U), us.usr.phi[3 * 80 + 79], 0.004, 'φ = exp(−k x/U) with the source expression −k·φ');
      const Xs = [], fs2 = [];
      for (const uq of [0.01, 0.02, 0.04, 0.08]) { const rq = await solveChannel({ ...base, Uin: uq, L: 2e-3, nx: 10, ny: 12, inlet: 'parabolic' }); Xs.push([(rho * uq * 2 * H) / mu]); fs2.push((grad(rq) * 2 * H) / (0.5 * rho * uq * uq)); }
      const fc = fitClosure(Xs, fs2);
      add('Regression closure trained on solver runs: friction exponent', -1, fc.exps[0], 0.01, 'Power-law fit f = a Re^b to four laminar solver runs recovers b = −1');
      add('Regression closure: learned f·Re', 96, fc.a, 1.5, 'Coefficient of the fitted law against Hagen–Poiseuille (12 cells across)');
    }
    { // second phase: Zalesak disc (VOF), translating circle (level set), ideal settler (drift-flux dispersed phase)
      const mkG = (N) => { const d = 1 / N, dyv = new Float64Array(N).fill(d), dyc2 = new Float64Array(N + 1).fill(d); dyc2[0] = dyc2[N] = d / 2; return { nx: N, ny: N, dx: d, dy: dyv, dyc: dyc2, u: new Float64Array((N + 1) * N), v: new Float64Array(N * (N + 1)), solid: new Uint8Array(N * N) }; };
      const fillA = (G, sd) => { const N = G.nx, a0 = new Float64Array(N * N), ph = new Float64Array(N * N); for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let q = 0; for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (sd((i + (a + 0.5) / 4) / N, (j + (b + 0.5) / 4) / N) > 0) q++; a0[j * N + i] = q / 16; ph[j * N + i] = sd((i + 0.5) / N, (j + 0.5) / N); } return [a0, ph]; };
      const Nz = 64, Gz = mkG(Nz);
      for (let j = 0; j < Nz; j++) for (let i = 0; i <= Nz; i++) Gz.u[j * (Nz + 1) + i] = -2 * Math.PI * ((j + 0.5) / Nz - 0.5);
      for (let j = 0; j <= Nz; j++) for (let i = 0; i < Nz; i++) Gz.v[j * Nz + i] = 2 * Math.PI * ((i + 0.5) / Nz - 0.5);
      const [az] = fillA(Gz, (x, y) => -Math.max(Math.hypot(x - 0.5, y - 0.75) - 0.15, -Math.max(Math.abs(x - 0.5) - 0.025, y - 0.85))), zr = advectPhase(Gz, { method: 'vof', a0: az, tEnd: 0.5, cfl: 0.5 });
      let l1 = 0, s0 = 0; for (let P = 0; P < Nz * Nz; P++) { l1 += Math.abs(zr.a[P] - az[Nz * Nz - 1 - P]); s0 += az[P]; } // half a turn maps cell P onto its point reflection
      add('Volume of fluid: mass conservation on the Zalesak disc', 0, (zr.vol - zr.vol0) / zr.vol0, 1e-6, 'Relative change of the phase volume after half a revolution of the slotted disc (64 × 64 cells)');
      add('Volume of fluid: Zalesak disc shape error', 0, l1 / s0, 0.12, 'Σ|α − α_exact|/Σα after half a revolution against the exactly rotated disc, THINC/WLIC on 64 × 64 cells (a full revolution gives 0.11, and 0.06 on 100 × 100)');
      const Nl = 48, Gl = mkG(Nl); Gl.u.fill(1); Gl.v.fill(0.5);
      const [al0, pl0] = fillA(Gl, (x, y) => 0.15 - Math.hypot(x - 0.3, y - 0.3)), lr = advectPhase(Gl, { method: 'ls', a0: al0, phi0: pl0, tEnd: 0.4, cfl: 0.4 });
      const kk = lr.hist.t.length - 1;
      add('Level set: area of a translated circle', 1, lr.vol / lr.vol0, 0.01, 'Circle of radius 0.15 carried 0.45 domain lengths diagonally with redistancing (ratio of areas)');
      add('Level set: translation distance of the centroid', 0.4, lr.hist.xc[kk] - lr.hist.xc[0], 0.004, 'Uniform velocity (1, 0.5) for 0.4 time units');
      const nxs = 60, nys = 20, Ls2 = 0.06, Hs = 0.01, Us = 0.05, vset = -0.002, gs = yGrid(Hs, nys, 1), Gs = { nx: nxs, ny: nys, dx: Ls2 / nxs, dy: gs.dy, dyc: gs.dyc, u: new Float64Array((nxs + 1) * nys).fill(Us), v: new Float64Array(nxs * (nys + 1)), solid: new Uint8Array(nxs * nys) };
      const er = advectPhase(Gs, { method: 'ee', a0: new Float64Array(nxs * nys).fill(1e-4), aIn: 1e-4, vs: vset, nRZ: 0, tEnd: (4 * Ls2) / Us, cfl: 0.4 }), hn = er.hist.t.length - 1, hm = Math.floor(0.75 * hn);
      add('Dispersed phase (drift flux): ideal-settler capture', (Math.abs(vset) * Ls2) / (Us * Hs), (er.hist.dep[hn] - er.hist.dep[hm]) / (er.hist.inn[hn] - er.hist.inn[hm]), 0.005, 'Hazen: captured fraction = v_s L/(U H) in plug flow with settling velocity v_s');
      add('Dispersed phase (drift flux): volume balance', 0, (er.vol0 + er.inn - er.vol - er.out - er.hist.dep[hn]) / er.inn, 1e-9, '(initial + inflow − hold-up − outflow − deposit) ÷ inflow');
    }
    { // Reynolds-stress transport model: homogeneous shear (no walls, no diffusion) and fully developed channel flow
      const hs = rsmHomogeneousShear(), th = hs.theory;
      add('Reynolds-stress transport: production ÷ dissipation in homogeneous shear', th.PoverEps, hs.PoverEps, 1e-3, 'Fixed point of the ε equation, P/ε = (C_ε2 − 1)/(C_ε1 − 1) = 2.09, reached by integrating the stress and ε equations to S·t = 60');
      add('Reynolds-stress transport: equilibrium anisotropy of the pressure–strain model', 0, Math.max(Math.abs(hs.a11 - th.a11), Math.abs(hs.a22 - th.a22), Math.abs(hs.a33 - th.a33), Math.abs(hs.a12 - th.a12)), 1e-4, `Largest deviation of a_ij = u_i′u_j′/k − ⅔δ_ij from the analytical fixed point of the Launder–Reece–Rodi model (a11 = ${fmt(th.a11, 4)}, a22 = a33 = ${fmt(th.a22, 4)}, a12 = ${fmt(th.a12, 4)})`);
      const srcS = 'Speziale, Sarkar & Gatski, ICASE Report 90-5 / NASA CR-181979 (1990), Table 1, read from the report on the NASA technical reports server', eq = SHEAR_EQ;
      add('Reynolds-stress transport: homogeneous-shear equilibrium equals the tabulated Launder–Reece–Rodi values', 0, Math.max(Math.abs(hs.a11 / 2 - eq.lrr.b11), Math.abs(hs.a22 / 2 - eq.lrr.b22), Math.abs(hs.a12 / 2 - eq.lrr.b12), Math.abs(hs.PoverEps / -hs.a12 - eq.lrr.SKe) / 10), 6e-4, `Largest deviation from the model column of the table (b11 = 0.193, b22 = −0.096, b12 = −0.185, S k/ε = 5.65, the last one ÷ 10); b_ij = (u_i′u_j′ − ⅔kδ_ij)/2k = a_ij/2. Computed ${fmt(hs.a11 / 2, 4)}, ${fmt(hs.a22 / 2, 4)}, ${fmt(hs.a12 / 2, 4)}, ${fmt(hs.PoverEps / -hs.a12, 4)}. Source: ${srcS}`);
      add('Reynolds-stress transport: streamwise anisotropy b11 against the homogeneous-shear experiment', eq.exp.b11, hs.a11 / 2, 0.02, `Experiments of Tavoularis & Corrsin (1981) as tabulated in the same table (b11 = 0.201, b22 = −0.147, b12 = −0.150, S k/ε = 6.08); the primary paper was not consulted. Source: ${srcS}`);
      add('Reynolds-stress transport: shear anisotropy b12 against the homogeneous-shear experiment', eq.exp.b12, hs.a12 / 2, 0.04, `Tabulated experiment −0.150; the model gives ${fmt(hs.a12 / 2, 3)} — it over-predicts the shear anisotropy by about a quarter and under-predicts −b22 (${fmt(hs.a22 / 2, 3)} against −0.147), the known weakness of this pressure–strain model that the table documents. The tolerance is set to that documented model error, not to the experimental scatter`);
      const Ht = 0.05, Ut = 1, ReH = (rho * Ut * Ht) / mu, cfD = 0.073 * ReH ** -0.25, nxt = 8, nyt = 20, t = await solveChannel({ H: Ht, L: 10 * Ht, nx: nxt, ny: nyt, rho, mu, Uin: Ut, inlet: 'periodic', scheme: 'hybrid', tol: 3e-6, maxIter: 1500, turb: 'rsm', alphaU: 0.7 });
      const tw = (gradX(t, 2, nxt - 2) * Ht) / 2, ut = Math.sqrt(tw / rho), ic = nxt - 2, jl = 3, jm = 6, Pl = jl * nxt + ic, Pm = jm * nxt + ic;
      add('Turbulent channel, Reynolds-stress transport: skin friction vs Dean', 1, tw / (0.5 * rho * Ut * Ut) / cfD, 0.08, 'C_f = 0.073 Re_H^−0.25 (Dean 1978); the mean flow is driven by the transported u′v′ (ratio)');
      add('Turbulent channel, Reynolds-stress transport: log-law velocity', 1, (0.5 * (t.u[jl * t.nu1 + ic] + t.u[jl * t.nu1 + ic + 1])) / ut / (Math.log((t.yc[jl] * ut * rho) / mu) / KAPPA + 5.2), 0.05, `u⁺ = ln(y⁺)/0.41 + 5.2 at y⁺ ≈ ${fmt((t.yc[jl] * ut * rho) / mu, 3)} (ratio)`);
      add('Turbulent channel, Reynolds-stress transport: total-stress balance', 1 - (2 * t.yc[jm]) / Ht, -t.rs.uv[Pm] / (ut * ut), 0.03, 'Fully developed flow: −u′v′/u_τ² = 1 − 2y/H outside the viscous layer; u_τ from the pressure gradient, u′v′ from its own transport equation');
      add('Turbulent channel, Reynolds-stress transport: wall-normal stress v′v′/k in the log layer', 0.247, t.rs.vv[Pl] / t.tk[Pl], 0.06, `Gibson & Launder (1978) wall-reflection equilibrium (u′u′/k = 1.098, v′v′/k = 0.247, w′w′/k = 0.655); computed u′u′/k = ${fmt(t.rs.uu[Pl] / t.tk[Pl], 3)}, w′w′/k = ${fmt(t.rs.ww[Pl] / t.tk[Pl], 3)}`);
      add('Turbulent channel, Reynolds-stress transport: stresses are realisable', 1, t.rs.uu.every((x, P) => x > 0 && t.rs.vv[P] > 0 && t.rs.ww[P] > 0 && t.rs.uv[P] ** 2 <= x * t.rs.vv[P] && Math.abs(0.5 * (x + t.rs.vv[P] + t.rs.ww[P]) - t.tk[P]) <= 1e-9 * t.tk[P]) ? 1 : 0, 0, 'Positive normal stresses, Schwarz inequality for u′v′, and k equal to half the trace in every cell');
    }
    { // coupled two-phase flow: volume of fluid and level set driving the same momentum equations
      const gE = 9.80665, sl = (ts, ys, t0, t1) => histSlope(ts, ys, t0, t1), mmS = sl(MARTIN_MOYCE.T, MARTIN_MOYCE.Z, 1.15, 3.05);
      for (const [method, nm] of [['vof', 'Volume of fluid'], ['ls', 'Level set']]) {
        const R = 0.25, sig = 0.07, dr = await twoPhase2D({ method, nx: 40, ny: 40, W: 1, Hh: 1, rhoA: 1000, rhoB: 1, muA: 1e-3, muB: 1e-5, sigma: sig, gy: 0, sd: (x, y) => R - Math.hypot(x - 0.5, y - 0.5), tEnd: 0.5 });
        add(`${nm}, coupled flow: Laplace pressure jump of a static drop`, 1, dr.dpJump / (sig / R), 0.03, `Δp = σ/R for a cylindrical interface, density ratio 1000, 10 cells per radius (ratio); spurious capillary number μU/σ = ${fmt((dr.umax * 1e-3) / sig, 2)}`);
        const rs = await twoPhase2D({ method, nx: 32, ny: 32, W: 1, Hh: 1, rhoA: 1000, rhoB: 1.2, muA: 1e-3, muB: 1.8e-5, sigma: 0.07, sd: (x, y) => 0.5 - y, tEnd: 0.5 });
        add(`${nm}, coupled flow: flat interface under gravity stays at rest`, 0, rs.umax, 1e-8, 'Balanced-force discretisation: gravity and the pressure gradient cancel at the faces for a hydrostatic water–air layer (largest velocity, m/s)');
        const h0 = 0.5, om = sloshOmega(Math.PI, h0, 1 - h0, 1000, 1, gE, 0), so = await twoPhase2D({ method, nx: 32, ny: 32, W: 1, Hh: 1, rhoA: 1000, rhoB: 1, muA: 1e-3, muB: 1e-5, sigma: 0, sd: (x, y) => h0 + 0.01 * Math.cos(Math.PI * x) - y, tEnd: (1.3 * 2 * Math.PI) / om });
        add(`${nm}, coupled flow: period of the first sloshing mode`, 1, histPeriod(so.hist.t, so.hist.hL.map((x) => 1 - x - h0)) / ((2 * Math.PI) / om), 0.03, 'Linear wave theory ω² = (ρ₁ − ρ₂) g k / (ρ₁ coth kh₁ + ρ₂ coth kh₂), k = π/W, amplitude 1 % of the depth, 32 × 32 cells (ratio)');
        const a = 0.05715, sT = Math.sqrt((2 * gE) / a), db = await twoPhase2D({ method, nx: 60, ny: 30, W: 5 * a, Hh: 2.5 * a, rhoA: 1000, rhoB: 1.2, muA: 1e-3, muB: 1.8e-5, sigma: 0, sd: (x, y) => Math.min(a - x, 2 * a - y), tEnd: 3 / sT });
        add(`${nm}, coupled flow: dam-break front speed against Martin & Moyce`, mmS, sl(db.hist.t.map((x) => x * sT), db.hist.xf.map((x) => (5 * a - x) / a), 1.2, 3), 0.1 * mmS, `Slope dZ/dT of the surge front (Z = x/a, T = t√(2g/a)) between T = 1.2 and 3 for the 2¼ in water column twice as high as wide, 12 cells per column width, against the least-squares slope ${fmt(mmS, 4)} of the eleven measured points with 1.19 ≤ T ≤ 2.97 (Z = 1.44 … 3.67) of Table 2 of Martin & Moyce (1952), Phil. Trans. R. Soc. A 244, 312–324 (n² = 2, a = 2¼ in, mean column; read by eye from a scan of page 317 of the paper). Only the slope is compared because the table has no measured time origin; tolerance 10 %`);
        add(`${nm}, coupled flow: phase volume conserved through the dam break`, 0, db.vol / db.vol0 - 1, 1e-9, method === 'ls' ? 'Level set with the constant-shift volume correction after every redistancing' : 'Flux-form THINC/WLIC transport with the divergence-free projected velocity');
        const hb = await twoPhase2D({ method, nx: 24, ny: 48, W: 1, Hh: 2, rhoA: 100, rhoB: 1000, muA: 1, muB: 10, sigma: 24.5, gy: -0.98, sd: (x, y) => 0.25 - Math.hypot(x - 0.5, y - 0.5), tEnd: 3, slipSide: true, slipTB: false });
        const srcH = 'benchmark proposal of Hysing, Turek, Kuzmin, Parolini, Burman, Ganesan & Tobiska (TU Dortmund, Ergebnisberichte Angewandte Mathematik Nr. 351, 2007), Table 12 and text, and the reference data files of the three codes published with it (featflow.de); the journal version (Int. J. Numer. Meth. Fluids 60, 2009) was not consulted';
        let vmB = 0, tmB = 0; hb.hist.vr.forEach((x, k) => { if (x > vmB) { vmB = x; tmB = hb.hist.t[k]; } });
        add(`${nm}, coupled flow: rising-bubble centroid at t = 3`, 1.081, hb.hist.yc[hb.hist.yc.length - 1], 0.02, `Test case 1 (Re = 35, Eo = 10, density and viscosity ratio 10): finest grids of the three codes ${HYSING1.yc3.join(', ')}; stated reference 1.081 ± 0.001. Computed on 24 × 48 cells (on 40 × 80: 1.084 volume of fluid, 1.088 level set). Source: ${srcH}`);
        add(`${nm}, coupled flow: largest rise velocity of the bubble`, 0.2419, vmB, 0.012, `Test case 1: finest grids ${HYSING1.vMax.join(', ')}; stated reference 0.2419 ± 0.0002. 24 × 48 cells (the value converges from below: 0.240 on 40 × 80)`);
        add(`${nm}, coupled flow: time of the largest rise velocity`, 0.927, tmB, 0.03, `Test case 1: finest grids ${HYSING1.tVmax.join(', ')}; the reference maximum occurs between t = 0.921 and 0.932. 24 × 48 cells`);
        add(`${nm}, coupled flow: projected velocity is divergence-free`, 0, hb.div, 1e-6, 'max |∇·u| after the variable-density pressure projection (1/s)');
      }
    }
    { // Eulerian–Eulerian two-fluid model
      const gE = 9.80665, rc = 1000, rd = 2500, muc = 1e-3, d1 = 300e-6, u1 = terminalSN(d1, rc, rd, muc), Re1 = (rc * u1 * d1) / muc;
      add('Two-fluid model: terminal velocity satisfies the Schiller–Naumann drag balance', (4 / 3) * ((rc * (rd - rc) * gE * d1 ** 3) / (muc * muc)), schillerNaumann(Re1) * Re1, 1e-6 * schillerNaumann(Re1) * Re1, `C_D Re² = (4/3) Ar for a 300 µm sphere of density 2500 kg/m³ in water (Re = ${fmt(Re1, 3)}, u_t = ${fmt(u1 * 1e3, 3)} mm/s)`);
      add('Two-fluid model: Stokes limit of the terminal velocity', 1, terminalSN(5e-6, rc, rd, muc) / (((rd - rc) * gE * 25e-12) / (18 * muc)), 1e-3, '5 µm particle, Re ≈ 10⁻⁴ (ratio to (ρ_d − ρ_c) g d²/18μ)');
      const dS = 20e-6, aS = 0.15, uS = terminalSN(dS, rc, rd, muc), Hc = 0.02, tS = (0.5 * Hc) / uS, bs = await twoFluid2D({ nx: 4, ny: 80, L: 0.004, H: Hc, rhoC: rc, muC: muc, rhoD: rd, dP: dS, alpha0: aS, tEnd: tS, slipC: true });
      add('Two-fluid model: batch-settling front against Kynch theory with Richardson–Zaki hindrance', uS * (1 - aS) ** 4.65, -histSlope(bs.hist.t, bs.hist.front, 0.3 * tS, tS), 0.02 * uS * (1 - aS) ** 4.65, 'Suspension front of 20 µm particles at 15 % by volume falls at u_t (1 − α)^4.65 (Stokes regime): drag, buoyancy and the return flow of the liquid from the coupled phase momentum and continuity equations (m/s)');
      const dB = 100e-6, aB = 0.2, uB = terminalSN(dB, rc, rd, muc), HB = 0.1, bb = await twoFluid2D({ nx: 4, ny: 60, L: 0.02, H: HB, rhoC: rc, muC: muc, rhoD: rd, dP: dB, alpha0: aB, tEnd: (0.4 * HB) / uB, slipC: true }), urB = hinderedSlip(aB, dB, rc, rd, muc);
      add('Two-fluid model: slip velocity in the suspension at finite Reynolds number', urB, -bb.hist.slip[bb.hist.slip.length - 1], 1e-3 * urB, 'k′(u_r) u_r = α_c (ρ_d − ρ_c) g with the Schiller–Naumann / Wen–Yu drag, 100 µm particles at 20 % (m/s)');
      let pb = 0, pt = 0; for (let i = 0; i < 4; i++) { pb += bb.p[i] / 4; pt += bb.p[59 * 4 + i] / 4; }
      add('Two-fluid model: shared pressure carries the weight of both phases', (aB * rd + (1 - aB) * rc) * gE * HB * (59 / 60), pb - pt, 0.005 * (aB * rd + (1 - aB) * rc) * gE * HB, 'Pressure difference between the lowest and highest cell centres = mean mixture density × g × height (hold-up balance, Pa), while the suspension is settling');
      add('Two-fluid model: dispersed-phase volume is conserved', 0, bb.vol / bb.vol0 - 1, 1e-10, 'Flux-form continuity equation with the packing limiter (relative change)');
      add('Two-fluid model: mixture volume balance', 0, bb.div * (HB / uB), 1e-6, 'max |∇·(α_c u_c + α_d u_d)| × H/u_t after the shared-pressure solve');
      const dF = 50e-6, uF = terminalSN(dF, rc, rd, muc), Lf = 0.06, Hf = 0.01, Uf = 0.05, fl = await twoFluid2D({ nx: 40, ny: 12, L: Lf, H: Hf, rhoC: rc, muC: muc, rhoD: rd, dP: dF, alpha0: 1e-3, flow: { U: Uf, alphaIn: 1e-3 }, tEnd: (4 * Lf) / Uf, slipC: true, deposit: true }), hn = fl.hist.t.length - 1, hm = Math.floor(0.75 * hn);
      add('Two-fluid model: ideal-settler capture (Hazen)', (uF * Lf) / (Uf * Hf), (fl.hist.dep[hn] - fl.hist.dep[hm]) / (fl.hist.dIn[hn] - fl.hist.dIn[hm]), 0.02 * ((uF * Lf) / (Uf * Hf)), 'Captured fraction = u_t L/(U H) for a dilute suspension in plug flow through a channel with a velocity inlet, pressure outlet and capture on the floor');
      add('Two-fluid model: flow-through volume balance of the dispersed phase', 0, (fl.vol0 + fl.dIn - fl.dOut - fl.dep - fl.vol) / fl.dIn, 1e-10, '(initial + inflow − outflow − captured − hold-up) ÷ inflow');
      add('Two-fluid model: mixture inflow equals mixture outflow', 1, (fl.cOut + fl.dOut) / (fl.cIn + fl.dIn), 1e-8, 'Incompressible mixture: Σ α_k u_k through the outlet ÷ through the inlet');
    }
    { // multiphase studies in channel geometries: solid mask (the rasteriser of the channel study) and through-flow boundaries
      const Lc = 6e-3, Hc = 1e-3, Uc = 0.1, x0 = 0.75e-3, x1 = 1.75e-3;
      for (const [method, nm] of [['vof', 'Volume of fluid'], ['ls', 'Level set']]) { // a slug carried through an empty through-flow channel in plug flow
        const r = await twoPhase2D({ method, nx: 48, ny: 8, W: Lc, Hh: Hc, rhoA: 1.2, rhoB: rho, muA: 1.8e-5, muB: mu, sigma: 0, gy: 0, sd: (x) => Math.min(x - x0, x1 - x), tEnd: 0.07, flow: { uin: Uc }, slipTB: true }), h = r.hist;
        const kh = h.out.findIndex((q) => q >= 0.5 * r.vol0), tHalf = kh > 0 ? h.t[kh - 1] + ((0.5 * r.vol0 - h.out[kh - 1]) / (h.out[kh] - h.out[kh - 1])) * (h.t[kh] - h.t[kh - 1]) : NaN, k3 = h.t.findIndex((t) => t >= 0.03);
        add(`${nm}, through-flow channel: an air slug travels at the flow velocity`, 1, (h.xc[k3] - h.xc[0]) / (Uc * h.t[k3]), method === 'ls' ? 3e-3 : 1e-6, 'Centroid displacement ÷ U·t after 30 ms in plug flow (uniform inlet, free-slip walls, pressure outlet), air slug in water, 48 × 8 cells (ratio)');
        add(`${nm}, through-flow channel: the slug arrives at the outlet on time`, 1, tHalf / ((Lc - 0.5 * (x0 + x1)) / Uc), method === 'ls' ? 0.015 : 2e-3, 'Time at which half of the gas has left through the outlet ÷ (distance of the slug centre from the outlet ÷ U) (ratio)');
        add(`${nm}, through-flow channel: gas volume balance through the outlet`, 0, (r.vol0 + r.volIn - r.volOut - r.vol) / r.vol0, method === 'ls' ? 2e-3 : 1e-12, method === 'ls' ? '(initial + inflow − outflow − hold-up) ÷ initial after the slug has left; the level set is corrected to the volume the boundary fluxes leave in the channel, which is first-order accurate while the interface crosses the outlet' : `(initial + inflow − outflow − hold-up) ÷ initial after the slug has left: flux-form transport; ${fmt((100 * r.volOut) / r.vol0, 6)} % of the gas left through the outlet`);
      }
      const nxs = 36, nys = 12, Ls2 = 4.5e-3, gys = yGrid(Hc, nys, 1), mks = buildMask({ type: 'spacer', arr: 'zigzag', L: Ls2, H: Hc, df: 0.4e-3, lm: 2.25e-3, nFil: 2 }, nxs, nys, gys.yc);
      const colS = (pq, i) => { let a2 = 0, m2 = 0; for (let j = 0; j < nys; j++) if (!mks.solid[j * nxs + i]) { a2 += pq[j * nxs + i]; m2++; } return a2 / m2; }, dpS = (pq) => colS(pq, 1) - colS(pq, nxs - 2);
      const prof = (Uq) => Float64Array.from({ length: nys }, (_, j) => { const q = (j + 0.5) / nys; return 6 * Uq * q * (1 - q); });
      for (const [method, nm] of [['vof', 'Volume of fluid'], ['ls', 'Level set']]) { // slug with surface tension and gravity between the filaments, before it reaches the outlet
        const r = await twoPhase2D({ method, nx: nxs, ny: nys, W: Ls2, Hh: Hc, rhoA: 1.2, rhoB: rho, muA: 1.8e-5, muB: mu, sigma: 0.072, gy: -9.80665, rhoRef: rho, sd: (x) => Math.min(x - 0.25e-3, 0.9e-3 - x), tEnd: 0.012, flow: { uin: prof(Uc) }, solid: mks.solid });
        add(`${nm}, spacer-filled channel: gas volume conserved around the filaments`, 0, (r.vol0 + r.volIn - r.volOut - r.vol) / r.vol0, 1e-10, `Air slug with surface tension (72 mN/m) and gravity carried ${fmt((r.hist.xc.at(-1) - r.hist.xc[0]) * 1e3, 3)} mm between zigzag filaments (${mks.solid.reduce((a2, b2) => a2 + b2, 0)} blocked cells of ${nxs} × ${nys}), ${r.steps} steps; (initial + inflow − outflow − hold-up) ÷ initial`);
        if (method === 'vof') add('Free interface in a spacer-filled channel: velocity is divergence-free in the fluid cells', 0, r.div * (Hc / Uc), 1e-8, 'max |∇·u| × H/U after the pressure projection with blocked faces and the pressure outlet');
      }
      const Ul = 0.02, scP = await solveChannel({ H: Hc, L: Ls2, nx: nxs, ny: nys, rho, mu, Uin: Ul, inlet: 'parabolic', scheme: 'quick', tol: 1e-8, maxIter: 3000, solid: mks.solid }), scU = await solveChannel({ H: Hc, L: Ls2, nx: nxs, ny: nys, rho, mu, Uin: Ul, inlet: 'uniform', scheme: 'quick', tol: 1e-8, maxIter: 3000, solid: mks.solid });
      const sp = await twoPhase2D({ method: 'vof', nx: nxs, ny: nys, W: Ls2, Hh: Hc, rhoA: rho, rhoB: rho, muA: mu, muB: mu, sigma: 0, gy: 0, sd: () => -1, tEnd: (3 * Ls2) / Ul, flow: { uin: prof(Ul) }, solid: mks.solid });
      add('Free-interface solver, single-phase limit: pressure drop over the spacer equals that of the channel study', 1, dpS(sp.p) / dpS(scP.p), 0.05, `Same solid mask (2 zigzag filaments, ${nxs} × ${nys} cells), parabolic inlet, Re = 40: explicit projection solver run to the steady state ÷ SIMPLE solver with bounded QUICK (${fmt(dpS(sp.p), 4)} against ${fmt(dpS(scP.p), 4)} Pa). The two discretisations converge to each other: 3.1 %, 1.0 % and 0.07 % apart on 36 × 12, 48 × 16 and 72 × 24 cells; at Re = 200 they are 29 %, 13 % and 7 % apart on the same grids, where the convection schemes differ (tolerance 5 %)`);
      const tfd = await twoFluid2D({ nx: nxs, ny: nys, L: Ls2, H: Hc, rhoC: rho, muC: mu, rhoD: 2500, dP: 5e-6, gy: -9.80665, alpha0: 1e-6, flow: { U: Ul, alphaIn: 1e-6 }, tEnd: (3 * Ls2) / Ul, solid: mks.solid, deposit: true });
      add('Two-fluid model, dilute limit: pressure drop over the spacer equals that of the channel study', 1, dpS(tfd.pExcess) / dpS(scU.p), 0.05, `Dispersed fraction 10⁻⁶ of 5 µm particles: the continuous phase is single-phase flow; same solid mask, uniform inlet, Re = 40 (${fmt(dpS(tfd.pExcess), 4)} against ${fmt(dpS(scU.p), 4)} Pa, tolerance 5 %)`);
      const tfs = await twoFluid2D({ nx: nxs, ny: nys, L: Ls2, H: Hc, rhoC: rho, muC: mu, rhoD: 2500, dP: 30e-6, gy: -9.80665, alpha0: 0.01, flow: { U: 0.1, alphaIn: 0.01 }, tEnd: (3 * Ls2) / 0.1, solid: mks.solid, deposit: true });
      add('Two-fluid model, spacer-filled channel: dispersed-phase balance with deposition on walls and filaments', 0, (tfs.vol0 + tfs.dIn - tfs.dOut - tfs.dep - tfs.vol) / tfs.dIn, 1e-10, `(initial + inflow − outflow − captured − hold-up) ÷ inflow, 30 µm particles at 1 % through zigzag filaments; ${fmt((100 * tfs.dep) / tfs.dIn, 3)} % of the inflow is captured, ${fmt((100 * tfs.depSolid) / Math.max(tfs.dep, 1e-300), 3)} % of that on the filaments`);
      add('Two-fluid model, spacer-filled channel: mixture inflow equals mixture outflow', 1, (tfs.cOut + tfs.dOut) / (tfs.cIn + tfs.dIn), 1e-8, 'Σ α_k u_k through the outlet ÷ through the inlet with blocked cells in the shared-pressure equation');
      const nxf = 40, nyf = 15, Lf2 = 0.06, Hf2 = 0.0125, hs2 = 0.0025, solF = new Uint8Array(nxf * nyf); for (let j = 0; j < 3; j++) for (let i = 0; i < nxf; i++) solF[j * nxf + i] = 1;
      const dF2 = 50e-6, uF2 = terminalSN(dF2, rho, 2500, mu), hz = await twoFluid2D({ nx: nxf, ny: nyf, L: Lf2, H: Hf2, rhoC: rho, muC: mu, rhoD: 2500, dP: dF2, alpha0: 1e-3, flow: { U: 0.05, alphaIn: 1e-3 }, tEnd: (4 * Lf2) / 0.05, slipC: false, deposit: true, solid: solF }), hn2 = hz.hist.t.length - 1, hm2 = Math.floor(0.75 * hn2);
      add('Two-fluid model with a solid mask: settling onto a solid surface (Hazen)', (uF2 * Lf2) / (0.05 * (Hf2 - hs2)), (hz.hist.dep[hn2] - hz.hist.dep[hm2]) / (hz.hist.dIn[hn2] - hz.hist.dIn[hm2]), 0.02 * ((uF2 * Lf2) / (0.05 * (Hf2 - hs2))), `Channel whose floor is a slab of blocked cells (no-slip): captured fraction = u_t L/(U h) with h the open height, for any unidirectional velocity profile; all of the capture is on the solid (${fmt((100 * hz.depSolid) / Math.max(hz.dep, 1e-300), 4)} %)`);
    }
    { // precipitation, population balance and fouling
      const Dp2 = 1e-9, kr = 2e-6, cs = 1, ct = 3, rw = await solveChannel({ ...base, Uin: 0, L: 2e-3, nx: 8, ny: 16, inlet: 'uniform', maxIter: 5, species: { c0: 2, D: Dp2, bot: 'react', top: 'fixed', cwTop: ct, kr, csat: cs }, scalIter: 80 });
      add('Wall crystallisation: reaction–diffusion wall concentration', (Dp2 * ct / H + kr * cs) / (Dp2 / H + kr), rw.spc.wB[4], 1e-6, 'k_r (c_w − c_sat) = D (c_top − c_w)/H between a reacting and a fixed wall, no flow');
      add('Wall crystallisation: deposition flux', kr * ((Dp2 * ct / H + kr * cs) / (Dp2 / H + kr) - cs), rw.spc.nB[4], 1e-12, 'N = k_r (c_w − c_sat), kg/m²·s');
      const kn = 1e9, kg2 = 1e-4, Lp = 5e-3, pb = await solveChannel({ ...base, scheme: 'quick', L: Lp, nx: 40, ny: 4, inlet: 'uniform', wallB: 'sym', wallT: 'sym', scalIter: 200, precip: { c0: 1.5, csat: 1, D: 1e-9, kr: 0, bot: false, top: false, pbm: true, kn, nn: 2, kg: kg2, Dp: 1e-12, n0: 0, d0: 0, dNuc: 0, rhoC: 0 } });
      const tR = (Lp - 0.5 * pb.dx) / U, Pe = 2 * 40 + 39;
      add('Population balance: nucleated number density in plug flow', 1, pb.mom[0].phi[Pe] / (kn * 0.25 * tR), 0.015, 'm₀ = B t with B = k_n (S − 1)² at S = 1.5 (ratio)');
      add('Population balance: Sauter mean size m₃/m₂', 1, pb.mom[3].phi[Pe] / pb.mom[2].phi[Pe] / (0.75 * kg2 * 0.5 * tR), 0.03, 'Moment equations with constant growth G give d₃₂ = ¾ G t (ratio)');
      const Af = (1 * LMH) / 1e5, alp = 1e15, cpf = 0.05, tF = 3600 * 20, fo = await solveChannel({ ...base, Uin: 0.1, L: 4e-3, nx: 12, ny: 12, stretch: 4, inlet: 'parabolic', tol: 1e-7, species: { c0: 1, D: 1.5e-9, A: Af, B: 0, dP: 20e5, pi: () => 0, bot: 'membrane', top: 'membrane' }, scalIter: 100, foul: { cp: cpf, alpha: alp, kBack: 0, time: tF, steps: 6, m0: 0 } });
      const J0 = fo.foul.J[0], sA = Af * mu * alp;
      add('Fouling layer: constant-pressure cake-filtration flux decline', 1 / Math.sqrt(1 + 2 * sA * cpf * J0 * tF), fo.foul.J[6] / J0, 0.01, 'J/J₀ = (1 + 2 α μ A c_f J₀ t)^−½ without back-transport or osmotic pressure');
    }
    { // compressible Navier–Stokes terms of the 1-D solver: heat conduction and viscous stress
      const gm = 1.4, Rg = 287, p0 = 1e5, T0 = 300, r0 = p0 / (Rg * T0), cpg = (gm * Rg) / (gm - 1), c0 = Math.sqrt(gm * Rg * T0), nc = 32, kk = Math.PI, tE = 0.02, chi = 2.5, eps = 1e-3;
      const th = euler1D({ n: nc, L: 1, gam: gm, Rg, left: 'wall', right: 'wall', kth: chi * r0 * cpg, tEnd: tE, init: (x) => { const T = T0 * (1 + eps * Math.cos(kk * x)); return { rho: p0 / (Rg * T), u: 0, p: p0 }; } });
      add('Compressible Navier–Stokes: decay of a temperature wave by conduction', Math.exp(-chi * kk * kk * tE), (th.T[0] - th.T[nc - 1]) / (2 * eps * T0 * Math.cos((kk * 0.5) / nc)), 0.01, 'Isobaric entropy mode between walls: amplitude ∝ exp(−κ k² t / ρ c_p), 32 cells');
      const tA = (4 * 2) / c0, nuV = 0.5 / ((4 / 3) * kk * kk * tA), ac = (muV) => { const e = euler1D({ n: nc, L: 1, gam: gm, Rg, left: 'wall', right: 'wall', mu: muV, tEnd: tA, init: (x) => ({ rho: r0, u: 1e-3 * c0 * Math.sin(kk * x), p: p0 }) }); let a = 0; for (let i = 0; i < nc; i++) a += 0.5 * r0 * e.u[i] ** 2 + (e.p[i] - p0) ** 2 / (2 * r0 * c0 * c0); return a; };
      add('Compressible Navier–Stokes: viscous damping of a standing sound wave', Math.exp(-0.5), ac(nuV * r0) / ac(0), 0.02, 'Acoustic energy with ÷ without viscosity after four periods: exp(−(4/3) ν k² t), which cancels the numerical dissipation of the scheme');
    }
    { // Lagrangian particles: ideal settler
      const Hs = 0.01, Ls = 0.06, Us = 0.05, ps = await solveChannel({ H: Hs, L: Ls, nx: 30, ny: 10, rho, mu, Uin: Us, inlet: 'uniform', scheme: 'hybrid', tol: 1e-8, maxIter: 400, wallB: 'sym', wallT: 'sym' });
      const dp = 68e-6, rhoP = 2650, vsP = ((rhoP - rho) * 9.80665 * dp * dp) / (18 * mu), pt = trackParticles(ps, { mu, rho, T: 25 }, { n: 400, d: dp, rho: rhoP, stick: 1 });
      add('Lagrangian particles: capture in an ideal settler', (vsP * Ls) / (Us * Hs) + dp / 2 / Hs, pt.bottom.length / pt.n, 0.01, 'Hazen: deposited fraction = v_s L/(U H) (+ interception by the particle radius) for Stokes particles released uniformly in plug flow');
      add('Lagrangian particles: every particle is accounted for', 400, pt.bottom.length + pt.top.length + pt.obstacle.length + pt.tRes.length + pt.suspended, 0, 'Deposited + escaped + suspended = released');
    }
    { // inlet, thermal walls, mixing metric, time accuracy
      const cm2 = caseConfig({ ...dflt, geom: 'plain', L: 6, nx: 24, ny: 12, stretch: 1, species: 'off', inletBC: 'massflow', mdot: 0.05 }), rm = await solveChannel(cm2.o);
      let mo = 0; for (let j = 0; j < 12; j++) mo += cm2.fl.rho * rm.u[j * rm.nu1 + 24] * rm.dy[j];
      add('Mass-flow inlet: solved outflow carries the specified mass flow', 0.05, mo, 1e-7, 'ρ ∫u dy at the outlet of the converged solution, kg/s per metre width');
      const alc = 1.5e-7, hc = 3e-4, cw2 = await solveChannel({ ...base, Uin: 0, L: 2e-3, nx: 8, ny: 16, inlet: 'uniform', maxIter: 5, energy: { alpha: alc, Tin: 50, bot: { type: 'fixed', val: 80 }, top: { type: 'conv', h: hc, ext: 20 } }, scalIter: 400 });
      add('Convective wall: surface temperature between conduction and the external film', 20 + 60 / (H / alc + 1 / hc) / hc, cw2.eng.wT[4], 1e-4, 'q = (T_hot − T_ext)/(H/k + 1/h); T_wall = T_ext + q/h for a stagnant layer between a hot wall and a convective wall (°C)');
      const qk = 2e-5, hq2 = await solveChannel({ ...base, L: 30e-3, nx: 60, ny: 24, Uin: 0.01, inlet: 'parabolic', energy: { alpha: 1.43e-7, Tin: 20, bot: { type: 'flux', val: qk }, top: { type: 'flux', val: qk } }, scalIter: 300 });
      let qf2 = 0, qq2 = 0; for (let j = 0; j < 24; j++) { const uc = 0.5 * (hq2.u[j * hq2.nu1 + 50] + hq2.u[j * hq2.nu1 + 51]); qf2 += uc * hq2.eng.phi[j * 60 + 50] * hq2.dy[j]; qq2 += uc * hq2.dy[j]; }
      add('Fully developed Nusselt number, uniform wall heat flux', 8.235, (qk * 2 * H) / (1.43e-7 * (hq2.eng.wB[50] - qf2 / qq2)), 0.06, 'Nu = 8.235 for plane Poiseuille flow heated at both walls (d_h = 2H)');
      add('Mixing analysis: coefficient of variation of a linear profile', 1 / Math.sqrt(12) / 1.5, sectionCoV(d, d.spc.phi)[4], 2e-3, 'c from 2 to 1 across the gap: CoV = (Δc/√12)/c̄');
      const Dt = 1.6e-6, tT = 0.05, tr = await solveChannel({ H, L: 4e-3, nx: 40, ny: 20, rho, mu: 1, Uin: U, inlet: 'uniform', scheme: 'hybrid', wallB: 'sym', wallT: 'sym', steady: false, tEnd: tT, cfl: 0.25, maxIter: 400, species: { c0: 1, D: Dt, bot: 'fixed', top: 'fixed', cw: 2 } });
      const yM = tr.yc[9]; let ser = 0; for (let m = 1; m < 40; m += 2) ser += (4 / (m * Math.PI)) * Math.sin((m * Math.PI * yM) / H) * Math.exp((-m * m * Math.PI * Math.PI * Dt * tT) / (H * H));
      add('Transient solver: diffusion from suddenly salted walls', 2 - ser, tr.spc.phi[9 * 40 + 36], 0.005, 'c(y, t) = c_w − (c_w − c₀) Σ (4/mπ) sin(mπy/H) exp(−m²π²Dt/H²) in plug flow ahead of the inlet front; implicit Euler, 100 steps');
    }
    return C;
  },
};

const HELP = {
  stepL: 'Length of the step measured from the inlet.', baffleH: 'How far each baffle reaches across the gap.', cadAxis: 'Axis normal to the cutting plane through the middle of a 3-D mesh.', cadSize: 'Height of the fitted outline as a share of the channel gap.', cadY: 'Centre (auto-fit) or lower edge (absolute) of the outline.',
  propMode: 'Seawater correlations cover 0–180 °C and 0–160 g/kg.', rho: 'Fluid density.', mu: 'Dynamic viscosity of the fluid.', Dsalt: 'Molecular diffusivity of the solute.', B: 'Solute permeability; sets the local salt passage Js = B (c_wall − c_perm).',
  species: 'Membrane walls couple permeation to the wall concentration; the fixed-concentration wall is the classical mass-transfer benchmark.', cwFixed: 'Concentration imposed on the active walls.', energy: 'Adds a temperature field with constant properties.', thWall: 'Condition applied on the thermally active walls; obstacles are adiabatic.', thSides: 'Walls on which the thermal condition is applied.',
  Tw: 'Temperature imposed on the active walls.', Text: 'Temperature on the far side of the wall or membrane.', porous: 'Useful for packed inserts, filters and fouling layers.', porX0: 'Upstream face of the porous slab.', porX1: 'Downstream face of the porous slab.',
  particles: 'Run after the flow has converged (one-way coupling).', nPart: 'Released flux-weighted across the inlet.', dPart: 'Colloids 0.1–1 µm, silt 2–60 µm, sand above 60 µm.', rhoPart: 'Silica 2650, organic matter 1050–1200, calcium carbonate 2700 kg/m³.',
  ny: 'At least 6 cells across a filament and 4–5 cells inside the concentration boundary layer.', maxIter: 'Steady: outer SIMPLEC iterations. Transient: time steps.',
};
for (const f of suite.inputs.flatMap((g) => g.fields)) if (!f.help && HELP[f.key]) f.help = HELP[f.key];

/** Synthetic channel-test data: the 1-D model with different "true" parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value])), g = rng(seed);
  return pts.map(([Uin, dPtm, c0]) => {
    const m = suite.calibration.model({ ...d, kdp: 1.45, ksh: 0.82, A: 1.12, Uin, dPtm, c0 });
    return { Uin, dPtm, c0, dpPerM: +(m.dpPerM * (1 + g.normal(0, 0.025))).toFixed(2), flux: +(m.flux * (1 + g.normal(0, 0.012))).toFixed(2) };
  });
}

export default suite;
