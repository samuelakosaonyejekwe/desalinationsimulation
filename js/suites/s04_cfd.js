// Suite 4 — Flow in membranes, channels and equipment (CFD).
// Two-dimensional finite-volume solver on a staggered (MAC) Cartesian grid, uniform in x and optionally
// clustered toward the walls in y: SIMPLE/SIMPLEC pressure–velocity coupling with an incomplete-Cholesky
// preconditioned conjugate-gradient pressure solver, immersed solids by cell blocking, salt transport with
// solution–diffusion membrane walls (concentration polarisation), an energy equation, an algebraic
// mixing-length eddy viscosity, a Darcy–Forchheimer porous zone and Lagrangian particle tracking.
// Extensions: two-equation RANS closures (k–ε, k–ω, k–ω SST, explicit algebraic Reynolds stress), Smagorinsky LES,
// Navier-slip / symmetry walls, mass-flow and pressure inlets, conjugate heat transfer, wall crystallisation with a
// moment population balance, a growing fouling layer, user-defined scalar sources, a D2Q9 lattice-Boltzmann start
// field, volume-of-fluid / level-set / dispersed-phase transport, a Maxwell–Stefan film, a regression closure and
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
  const { n1, n2, aW, aE, aS, aN, aP, b, t1: cp, t2: dp } = S;
  const col = (i) => {
    for (let j = 0; j < n2; j++) {
      const P = j * n1 + i;
      let d = b[P];
      if (i > 0) d += aW[P] * phi[P - 1];
      if (i < n1 - 1) d += aE[P] * phi[P + 1];
      const m = aP[P] - (j > 0 ? aS[P] * cp[j - 1] : 0);
      cp[j] = aN[P] / m; dp[j] = (d + (j > 0 ? aS[P] * dp[j - 1] : 0)) / m;
    }
    phi[(n2 - 1) * n1 + i] = dp[n2 - 1];
    for (let j = n2 - 2; j >= 0; j--) phi[j * n1 + i] = dp[j] + cp[j] * phi[(j + 1) * n1 + i];
  };
  const row = (j) => {
    for (let i = 0; i < n1; i++) {
      const P = j * n1 + i;
      let d = b[P];
      if (j > 0) d += aS[P] * phi[P - n1];
      if (j < n2 - 1) d += aN[P] * phi[P + n1];
      const m = aP[P] - (i > 0 ? aW[P] * cp[i - 1] : 0);
      cp[i] = aE[P] / m; dp[i] = (d + (i > 0 ? aW[P] * dp[i - 1] : 0)) / m;
    }
    phi[j * n1 + n1 - 1] = dp[n1 - 1];
    for (let i = n1 - 2; i >= 0; i--) phi[j * n1 + i] = dp[i] + cp[i] * phi[j * n1 + i + 1];
  };
  for (let s = 0; s < sweeps; s++) {
    for (let i = 0; i < n1; i++) col(i);
    for (let j = 0; j < n2; j++) row(j);
    for (let i = n1 - 1; i >= 0; i--) col(i);
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
  // closures and wall types: tm = 'ml' | 'ke' | 'kw' | 'sst' | 'earsm' | 'les'; walls 'noslip' | 'slip' (Navier) | 'sym'
  const tm = o.turb === true ? 'ml' : o.turb || null, twoEq = tm === 'ke' || tm === 'kw' || tm === 'sst' || tm === 'earsm';
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
  const setWallV = () => { for (let i = 0; i < nx; i++) { v[i] = solid[i] ? 0 : -Jb[i]; v[ny * nx + i] = solid[(ny - 1) * nx + i] ? 0 : Jt[i]; } };

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
  const lT = o.lTurb || 0.07 * 2 * H, kIn = Math.max(1.5 * ((o.tuIn ?? 0.05) * Uref) ** 2, K_MIN), eIn = (C75 * kIn ** 1.5) / lT, wIn = Math.sqrt(kIn) / (C25 * lT);
  let tk = null, te = null, St = null, S2 = null, PkA = null, dK = null, dE = null, nearW = null, cmuE = null, exx = null, exy = null, F1 = null, crs = null, kOld = null, eOld = null, tRdt = 0;
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
    tk = f().fill(kIn); te = f().fill(tm === 'ke' ? eIn : wIn); St = stencil(nx, ny); S2 = f(); PkA = f(); dK = f(); dE = f(); nearW = new Uint8Array(n); kOld = f(); eOld = f();
    if (tm === 'sst') { F1 = f().fill(1); crs = f(); }
    if (tm === 'earsm') { cmuE = f().fill(CMU); exx = f(); exy = f(); }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      nearW[P] = !solid[P] && ((j === 0 && wallB !== 'sym') || (j === ny - 1 && wallT !== 'sym') || (i > 0 && solid[P - 1]) || (i < nx - 1 && solid[P + 1]) || (j > 0 && solid[P - nx]) || (j < ny - 1 && solid[P + nx])) ? 1 : 0;
      if (!solid[P]) mue[P] = mu + rho * Math.min(tm === 'ke' ? (CMU * kIn * kIn) / eIn : kIn / wIn, 1e5 * nu);
    }
  }
  const exC = (i, jf) => { if (jf <= 0 || jf >= ny) return 0; const i0 = i > 0 ? i - 1 : 0, i1 = i < nx ? i : nx - 1; return 0.25 * (exy[(jf - 1) * nx + i0] + exy[(jf - 1) * nx + i1] + exy[jf * nx + i0] + exy[jf * nx + i1]); };
  const asmT = (phi, dc, inVal, kWall, sch) => {
    for (let j = 0; j < ny; j++) for (let k = 0; k <= nx; k++) {
      const q = j * (nx + 1) + k, R = j * nx + k, Lc = R - 1;
      St.Fx[q] = rho * u[j * nu1 + k] * dy[j];
      St.Dx[q] = k === nx ? 0 : k === 0 ? (solid[R] || !(uin[j] > 0) ? 0 : (dc[R] * dy[j]) / (0.5 * dx)) : solid[Lc] || solid[R] ? 0 : (0.5 * (dc[Lc] + dc[R]) * dy[j]) / dx;
    }
    for (let jf = 0; jf <= ny; jf++) for (let i = 0; i < nx; i++) {
      const q = jf * nx + i;
      St.Fy[q] = rho * v[q] * dx;
      St.Dy[q] = jf === 0 || jf === ny || solid[q - nx] || solid[q] ? 0 : (0.5 * (dc[q - nx] + dc[q]) * dx) / dyc[jf];
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
    const isE = tm === 'ke', sst = tm === 'sst', al = tRdt ? 1 : 0.7, schT = scheme === 'upwind' ? 'upwind' : 'hybrid';
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) { S2[P] = 0; dK[P] = dE[P] = mu; continue; }
      grads(i, j);
      const ux = gr[0], uy = gr[1], vx = gr[2], vy = gr[3], mut = mue[P] - mu, k = Math.max(tk[P], K_MIN), w = te[P];
      S2[P] = 2 * (ux * ux + vy * vy) + (uy + vx) * (uy + vx);
      let sk = isE ? 1 : 0.5, se = isE ? 1 / 1.3 : 0.5;
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
    asmT(tk, dK, kIn, true, schT);
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) { St.aW[P] = St.aE[P] = St.aS[P] = St.aN[P] = 0; St.aP[P] = 1; St.b[P] = K_MIN; PkA[P] = 0; continue; }
      const vol = dx * dy[j], k = Math.max(tk[P], K_MIN), e = te[P], dest = isE ? (rho * e) / k : rho * CMU * e;
      let Pk = Math.min((mue[P] - mu) * S2[P], 10 * dest * k);
      if (j === 0 ? wfB[i] : j === ny - 1 ? wfT[i] : 0) { const tw = Math.abs(j === 0 ? tauB[i] : tauT[i]); Pk = (tw * tw) / (KAPPA * rho * C25 * Math.sqrt(k) * 0.5 * dy[j]); }
      PkA[P] = Pk;
      let ap = St.aP[P] + dest * vol;
      St.b[P] += Pk * vol;
      if (tRdt) { ap += tRdt * vol; St.b[P] += tRdt * vol * kOld[P]; }
      ap /= al; St.b[P] += (1 - al) * ap * tk[P]; St.aP[P] = ap;
    }
    lineSolve(St, tk, 1);
    for (let P = 0; P < n; P++) if (!(tk[P] > K_MIN)) tk[P] = K_MIN;
    // ε or ω equation
    asmT(te, dE, isE ? eIn : wIn, false, schT);
    const eMin = 1e-9 * (isE ? eIn : wIn);
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      const vol = dx * dy[j], k = Math.max(tk[P], K_MIN), e = te[P];
      if (solid[P] || nearW[P]) { // fixed value: solids keep theirs, wall-adjacent cells take the viscous / log-layer value
        let val = e;
        if (!solid[P]) {
          const d = Math.max(dist[P], 1e-12);
          if (isE) val = Math.max((C75 * k ** 1.5) / (KAPPA * d), (2 * nu * k) / (d * d));
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
  };
  /** Resolved Reynolds stresses (kinematic) of the two-equation closures: Boussinesq part plus the explicit anisotropy. */
  const reynolds = () => {
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
  const du = new Float64Array(nu1 * ny), dv = new Float64Array(nx * (ny + 1));
  const pE = new Float64Array(n), pN = new Float64Array(n), pD = new Float64Array(n), rhs = new Float64Array(n);
  const Wcg = { r: new Float64Array(n), z: new Float64Array(n), s: new Float64Array(n), q: new Float64Array(n), pc: new Float64Array(n) };
  const por = o.porous ? new Uint8Array(n) : null;
  if (por) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const x = (i + 0.5) * dx; por[j * nx + i] = x >= o.porous.x0 && x <= o.porous.x1 && !solid[j * nx + i] ? 1 : 0; }
  const simplec = o.simplec !== false, aU = o.alphaU ?? 0.7, aPr = o.alphaP ?? (simplec ? 1 : 0.3);
  const uPrev = new Float64Array(u.length);
  let un = null, vn = null;

  /** One SIMPLE(C) iteration. rdt = ρ/Δt for time-accurate steps (0 for steady relaxation). */
  const iterate = (rdt) => {
    const al = rdt ? 1 : aU, turb = !!o.turb;
    // ---- u momentum
    for (let j = 0; j < ny; j++) for (let k = 1; k <= nx; k++) {
      const q = j * (nu1 + 1) + k;
      Su.Fx[q] = cv * rho * 0.5 * (u[j * nu1 + k - 1] + u[j * nu1 + k]) * dy[j];
      Su.Dx[q] = (turb ? mue[j * nx + k - 1] : mu) * gxu[q];
    }
    for (let jf = 0; jf <= ny; jf++) for (let i = 1; i < nx; i++) {
      const q = jf * nu1 + i;
      Su.Fy[q] = cv * rho * 0.5 * (v[jf * nx + i - 1] + v[jf * nx + i]) * dx;
      Su.Dy[q] = (turb ? muCorner(i, jf) : mu) * gyu[q];
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
      if (rdt) { ap += rdt * vol; Su.b[k] += rdt * vol * un[k]; }
      ap /= al;
      if (exx) Su.b[k] -= (exx[j * nx + i] - exx[j * nx + i - 1]) * dy[j] + (exC(i, j + 1) - exC(i, j)) * dx;
      Su.b[k] += (p[j * nx + i - 1] - p[j * nx + i]) * dy[j] + (1 - al) * ap * u[k];
      Su.aP[k] = ap; du[k] = dy[j] / (simplec ? Math.max(ap - nb, 0.05 * ap) : ap);
    }
    uPrev.set(u); // previous iterate, for the change norm
    lineSolve(Su, u, 1);
    for (let k = 0; k < u.length; k++) { const d = Math.abs(u[k] - uPrev[k]); if (d > dUmax) dUmax = d; }
    // ---- v momentum
    for (let jf = 1; jf < ny; jf++) for (let k = 0; k <= nx; k++) {
      const q = jf * (nx + 1) + k;
      Sv.Fx[q] = cv * rho * 0.5 * (u[(jf - 1) * nu1 + k] * dy[jf - 1] + u[jf * nu1 + k] * dy[jf]);
      Sv.Dx[q] = (turb ? muCorner(k, jf) : mu) * gxv[q];
    }
    for (let jj = 1; jj <= ny; jj++) for (let i = 0; i < nx; i++) {
      const q = jj * nx + i;
      Sv.Fy[q] = cv * rho * 0.5 * (v[(jj - 1) * nx + i] + v[jj * nx + i]) * dx;
      Sv.Dy[q] = (turb ? mue[(jj - 1) * nx + i] : mu) * gyv[q];
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
      if (rdt) { ap += rdt * vol; Sv.b[k] += rdt * vol * vn[k]; }
      ap /= al;
      if (exx) Sv.b[k] -= (exC(i + 1, jf) - exC(i, jf)) * dyc[jf] - (exx[jf * nx + i] - exx[(jf - 1) * nx + i]) * dx;
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
    pp.fill(0);
    pcg5(nx, ny, pE, pN, pD, rhs, pp, o.pTol ?? 0.02, o.pIter ?? 60, Wcg);
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
    for (let P = 0; P < n; P++) p[P] *= f;
    for (let j = 0; j < ny; j++) uin[j] *= f;
    Qin *= f; Uref *= f; setWallV();
  };
  let pFac = 1;
  const flowSteady = async (maxIt, f0, f1) => {
    let ok = false;
    for (let k = 0; k < maxIt; k++) {
      if (o.turb && (twoEq || k % 3 === 0)) updateTurb();
      const r = iterate(0);
      iters++; hist.it.push(iters); hist.mass.push(Math.max(r.mass, 1e-16)); hist.dU.push(Math.max(r.dU, 1e-16));
      if (!Number.isFinite(r.mass) || !Number.isFinite(r.dU)) throw new Error('The flow solution diverged. Lower the velocity-relaxation factor, use the hybrid scheme or refine the grid.');
      if (k % 12 === 0) await tick(f0 + ((f1 - f0) * k) / maxIt, `Flow iteration ${iters}: continuity residual ${r.mass.toExponential(1)}`);
      if (o.pInlet > 0 && k >= 8 && k % 4 === 0) { const dpn = pIn(); pFac = dpn > 0 ? clamp((o.pInlet / dpn) ** 0.5, 0.7, 1.4) : 1.4; if (Math.abs(pFac - 1) > 1e-9) rescale(pFac); }
      if (k > 3 && r.mass < tol && r.dU < tol && (!(o.pInlet > 0) || (k > 12 && Math.abs(pFac - 1) < 20 * tol))) { ok = true; break; }
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
      if (!Number.isFinite(last)) throw new Error('The scalar transport solution diverged.');
      if (k % 20 === 0 && ctx?.tick) await ctx.tick();
      if (k > 2 && last < Math.max(tol, 1e-7)) break;
    }
    return last;
  };
  let scalRes = 0;
  if (o.steady !== false) {
    converged = await flowSteady(maxIter, 0, spc || eng ? 0.6 : 0.95);
    if (spc) {
      const memb = o.species.bot === 'membrane' || o.species.top === 'membrane', nS = o.scalIter ?? 400;
      if (memb) {
        await scalarSteady(spc, nS, false, o.species.c0);
        await tick(0.7, 'Coupling permeation flux and flow');
        await flowSteady(Math.min(80, maxIter), 0.7, 0.8);
        await scalarSteady(spc, Math.round(nS / 2), false, o.species.c0);
        const ok2 = await flowSteady(Math.min(80, maxIter), 0.8, 0.9);
        converged = converged && ok2;
        scalRes = await scalarSteady(spc, nS, true, o.species.c0);
      } else scalRes = await scalarSteady(spc, nS, true, o.species.c0);
    }
    if (eng) scalRes = Math.max(scalRes, await scalarSteady(eng, o.scalIter ?? 400, true, Math.max(1, Math.abs(o.energy.Tin))));
  } else {
    // time-accurate implicit Euler with SIMPLE inner iterations; CFL based on the local velocity
    un = new Float64Array(u.length); vn = new Float64Array(v.length);
    const cOld = spc ? new Float64Array(n) : null, tOld = eng ? new Float64Array(n) : null, tEnd = o.tEnd, inner = o.inner ?? 2;
    probe = { t: [], v: [], dp: [], i: o.probe?.i ?? Math.round(0.6 * nx), j: o.probe?.j ?? Math.round(ny / 2) };
    const stat = { n: 0, tauB: new Float64Array(nx), tauT: new Float64Array(nx), cB: spc ? new Float64Array(nx) : null, cT: spc ? new Float64Array(nx) : null, JB: new Float64Array(nx), JT: new Float64Array(nx), dp: 0 };
    // small antisymmetric disturbance so that wake instabilities can develop from a symmetric start
    for (let jf = 1; jf < ny; jf++) for (let i = 0; i < nx; i++) if (!vblk[jf * nx + i]) v[jf * nx + i] += 0.03 * Uref * Math.sin((6 * Math.PI * (i + 0.5)) / nx) * Math.sin((Math.PI * yf[jf]) / H);
    let step = 0;
    while (time < tEnd && iters < maxIter * inner) {
      let um = 1e-12;
      for (let k = 0; k < u.length; k++) um = Math.max(um, Math.abs(u[k]));
      let vm = 0;
      for (let jf = 1; jf < ny; jf++) for (let i = 0; i < nx; i++) vm = Math.max(vm, Math.abs(v[jf * nx + i]) / dyc[jf]);
      const dt = Math.min((o.cfl ?? 1) / (um / dx + vm), tEnd - time + 1e-12);
      un.set(u); vn.set(v);
      if (twoEq) { tRdt = rho / dt; kOld.set(tk); eOld.set(te); }
      if (o.turb) updateTurb();
      let r = null;
      for (let k = 0; k < inner; k++) { r = iterate(rho / dt); iters++; }
      if (!Number.isFinite(r.mass)) throw new Error('The transient flow solution diverged. Lower the CFL number.');
      hist.it.push(iters); hist.mass.push(Math.max(r.mass, 1e-16)); hist.dU.push(Math.max(r.dU, 1e-16));
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
    probe.stat = stat;
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
  return { scal, mom, usr, foul, afB, afT, pFac, nx, ny, dx, L, H, yf, yc, dy, dyc, solid, u, v, p, mue, Jb, Jt, tauB, tauT, Qin, Uref, uin, hist, iters, converged, scalRes, spc, eng, probe, time, utau: utauG, nu1, tm, tk, te, rs: twoEq ? reynolds() : null, wallB, wallT };
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
  for (let P = 0; P < nx * ny; P++) { if (!seen[P]) m[P] = 1; nSolid += m[P]; }
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
  const Uin = v.inletBC === 'massflow' ? Math.max(v.mdot, 1e-12) / (fl.rho * Math.max(openIn, 1e-12)) : v.inletBC === 'pressure' ? clamp(Math.min((dpIn * H * H) / (12 * fl.mu * L), Math.sqrt((4 * dpIn * H) / (0.03 * fl.rho * L))), 1e-6, 20) : v.Uin;
  const wt = v.wallType || 'noslip', wallB = wt === 'slip' ? 'slip' : wt === 'symboth' ? 'sym' : 'noslip', wallT = wt === 'slip' ? 'slip' : wt === 'symboth' || wt === 'symtop' ? 'sym' : 'noslip';
  const precip = v.precip ? { c0: v.scC0, csat: Math.max(v.scSat, 1e-9), D: v.scD * 1e-9, kr: v.scKr * 1e-6, bot: !!sb, top: !!st, pbm: !!v.pbm, kn: v.pbKn, nn: clamp(v.pbN, 0.5, 6), kg: v.pbKg * 1e-6, Dp: 1e-11, n0: v.pbSeedN, d0: v.pbSeedD * 1e-6, dNuc: 1e-8, rhoC: v.scRho } : null;
  const foul = v.foul && v.species === 'membrane' && v.mode !== 'transient' && !les ? { cp: v.foulC * 1e-3, alpha: v.foulAlpha, kBack: v.foulBack * 1e-6, time: v.foulTime * 3600, steps: clamp(Math.round(v.foulSteps), 1, 40), m0: v.foulM0 * 1e-3 } : null;
  let user = null;
  if (v.usr) {
    const f = compileExpr(v.usrSrc, ['phi', 'c', 'T', 'x', 'y', 'u', 'v']), uw = v.usrWall === 'fixed' ? { type: 'fixed', val: v.usrWallVal } : { type: 'none' };
    user = { D: v.usrD * 1e-9, in: v.usrIn, T0: v.T, scale: Math.max(Math.abs(v.usrIn), Math.abs(v.usrWallVal ?? 0), 1e-9), bot: sb || v.species === 'off' ? uw : { type: 'none' }, top: st || v.species === 'off' ? uw : { type: 'none' }, src: (phi, env) => { env.phi = phi; return f(env); } };
  }
  const o = { L, H, nx, ny, stretch, solid: mk.solid, rho: fl.rho, mu: fl.mu, Uin, inlet: v.inlet, scheme: v.scheme, steady: v.mode !== 'transient' && !les, maxIter: clamp(Math.round(v.maxIter), 5, 6000), tol: clamp(v.tol, 1e-9, 1e-2),
    alphaU: clamp(v.alphaU, 0.2, 0.95), cfl: clamp(v.cfl, 0.2, 10), tEnd: (v.tFlow * L) / Math.max(Uin, 1e-9), turb: v.turb === 'ml' ? true : v.turb && v.turb !== 'laminar' ? v.turb : false, porous, species, energy,
    wallB, wallT, slipLen: (v.slipLen ?? 0) * 1e-6, creeping: !!v.creeping, tuIn: (v.tuIn ?? 5) / 100, lTurb: ((v.lTurb ?? 7) / 100) * 2 * H, cSmag: v.cSmag ?? 0.17, pInlet: v.inletBC === 'pressure' ? dpIn : 0, precip, foul, user };
  return { fl, geo, mk, L, H, nx, ny, g, o, openIn, Uin };
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
  let jj = 0;
  const vel = (x, y) => {
    const i = clamp(Math.floor(x / dx), 0, nx - 1), j = (jj = jOf(y)), fx = clamp(x / dx - i, 0, 1), j0 = y < yc[j] ? j - 1 : j, j1 = j0 + 1;
    const y0 = j0 >= 0 ? yc[j0] : 0, y1 = j1 < ny ? yc[j1] : H;
    const ua = j0 >= 0 ? u[j0 * nu1 + i] * (1 - fx) + u[j0 * nu1 + i + 1] * fx : 0, ub = j1 < ny ? u[j1 * nu1 + i] * (1 - fx) + u[j1 * nu1 + i + 1] * fx : 0;
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

/** Derived wall, bulk and integral quantities from a solved case. */
function post(c, r) {
  const { nx, ny, dx, dy, yc, solid, u, v, p, nu1, L, H, Uref, Qin } = r, { fl } = c, n = nx * ny, dh = 2 * H;
  const xc = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx), uc = new Float64Array(n), vc = new Float64Array(n);
  let aF = 0, aRec = 0, aStag = 0, umax = 0;
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    if (solid[P]) continue;
    uc[P] = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); vc[P] = 0.5 * (v[P] + v[P + nx]);
    const a = dx * dy[j], sp = Math.hypot(uc[P], vc[P]);
    aF += a; if (uc[P] < -1e-3 * Uref) aRec += a; if (sp < 0.1 * Uref) aStag += a; if (sp > umax) umax = sp;
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
  const o = { xc, uc, vc, pm, pt, dpdx, Re, Sc, f, dh, umax, recirc: aRec / aF, stagnant: aStag / aF, tauB, tauT, JB, JT, i1, i2, dpTot: pm[0] - pm[nx - 1] };
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
 * 'ee': dispersed-phase continuity of the two-fluid model in the algebraic-slip limit — relative velocity vs
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
  let k = 0;
  for (const pm of pitches) for (const um of vel) {
    ctx?.progress?.(0.97 + (0.03 * k++) / (pitches.length * vel.length), `Regression closure: training run ${k} of ${pitches.length * vel.length}`);
    const vv = { ...v, Uin: c.Uin * um, inletBC: 'velocity', lm: v.lm * pm, nx: nxc, ny: nyc, mode: 'steady', turb: v.turb === 'les' ? 'laminar' : v.turb, maxIter: Math.min(v.maxIter, 500), energy: false, particles: false, precip: false, pbm: false, foul: false, usr: false, mp: 'off', ml: false, ms: false, engine: 'fv' };
    const cc = caseConfig(vv), rr = await solveChannel(cc.o, { tick: ctx?.tick }), qq = post(cc, rr), kd = qq.sp ? qq.sp.kDev ?? qq.sp.kAll : null;
    if (!(qq.f > 0) || !Number.isFinite(qq.f)) continue;
    X.push([qq.Re, ...(geo ? [cc.geo.lm / cc.H] : [])]); U.push(rr.Uref); f.push(qq.f); sh.push(kd ? (kd * qq.dh) / cc.fl.D : NaN);
  }
  if (X.length < 4) throw new Error('The regression closure needs at least four usable training runs — the solver did not return a positive friction factor for the sampled velocities.');
  const okS = sh.every((s) => s > 0 && Number.isFinite(s));
  return { X, U, f, sh, geo, nx: nxc, ny: nyc, fitF: fitClosure(X, f), fitS: okS ? fitClosure(X, sh) : null };
}

const TURB_NAMES = { ml: 'algebraic mixing length', ke: 'standard k–ε with wall functions', kw: 'Wilcox k–ω', sst: 'Menter k–ω SST', earsm: 'explicit algebraic Reynolds stress (Wallin–Johansson) on k–ω', les: 'LES, Smagorinsky sub-grid model' };

const F = (key, label, unit, value, min, max, help, extra = {}) => ({ key, label, unit, value, min, max, help, ...extra });
const SEL = (key, label, value, options, help, extra = {}) => ({ key, label, type: 'select', value, options: options.map(([v, l]) => ({ value: v, label: l })), help, ...extra });
const isSpacer = (v) => v.geom === 'spacer', isImport = (v) => v.geom === 'import', hasSpecies = (v) => v.species !== 'off';
const isCmp = (v) => v.study === 'shock' || v.study === 'nozzle', twoEqSel = (v) => ['ke', 'kw', 'sst', 'earsm'].includes(v.turb), BOOL = (key, label, help, extra = {}) => ({ key, label, type: 'bool', value: false, help, ...extra });

const suite = {
  id: 'cfd', num: 4, title: 'Flow in Membranes, Channels & Equipment (CFD)', short: 'CFD', icon: '🌀',
  tagline: 'Two-dimensional finite-volume CFD of spacer-filled membrane channels, ducts and imported shapes with salt, heat, particle and second-phase transport, turbulence closures, scaling and fouling, plus 1-D compressible gas studies.',
  description: 'Solves the incompressible Navier–Stokes equations on a staggered Cartesian grid with SIMPLE-type pressure–velocity coupling and a preconditioned conjugate-gradient pressure solver. Spacer filaments, steps, baffles or imported CAD sections are immersed as blocked cells. Salt transport is coupled to solution–diffusion membrane walls, so concentration polarisation, local permeate flux, wall shear, friction factor and Sherwood number come straight from the resolved fields and are compared with the Hagen–Poiseuille, Lévêque/Graetz and Schock–Miquel relations. Optional models add two-equation and algebraic Reynolds-stress turbulence closures, large-eddy simulation, conjugate heat transfer, precipitation with a crystal population balance, a growing fouling layer, a second phase (volume of fluid, level set or dispersed), a lattice-Boltzmann start field, a regression closure trained on solver runs, and one-dimensional compressible flow in vapour lines and nozzles.',
  guide: [
    'Choose the geometry: a spacer-filled membrane channel, an empty channel, a sudden expansion, baffles, or an imported STL/OBJ/DXF/GeoJSON section.',
    'Enter the fluid, cross-flow velocity and membrane data (or pull them from the case and the RO suite).',
    'On Model setup pick steady or transient flow, the convection scheme, the wall conditions for salt and heat, and optional turbulence, porous-zone and particle models.',
    'Optional physics is switched on one model at a time on Model setup: inlet and wall types, a second phase, scaling and fouling, multicomponent diffusion, a user-defined scalar, or the compressible gas studies under Study type. Each adds its own results, balance and warnings.',
    'Run. Check the residual history and the conservation closure first, then read the fields, wall profiles and the comparison with correlations.',
    'Use the Mesh tab to quantify numerical uncertainty; the mass-transfer multiplier is offered to the RO design suite.',
  ],
  implemented: ['continuity equation', 'incompressible', 'euler equations', 'stokes-flow', 'reynolds-averaged', 'reynolds-stress', 'turbulent kinetic-energy', 'turbulent-dissipation', 'specific-dissipation-rate', 'sst equations', 'large-eddy-simulation', 'species-conservation', 'convection-diffusion', 'the energy equation', 'fourier', 'fick', 'maxwell-stefan', 'darcy equation', 'brinkman', 'forchheimer', 'ergun', 'hagen-poiseuille', 'darcy-weisbach',
    'navier-stokes-species', 'navier-stokes-solution-diffusion', 'cfd-concentration-polarization', 'cfd-porous-media', 'cfd-fouling', 'cfd-particle-deposition', 'cfd-population-balance', 'cfd-precipitation', 'cfd-heat/mass-transfer', 'conjugate heat-transfer', 'eulerian-lagrangian', 'eulerian-eulerian', 'volume-of-fluid', 'level-set', 'lattice-boltzmann', 'cfd-machine-learning',
    'velocity', 'initial pressure field', 'concentration', 'temperature', 'turbulence quantities', 'phase fractions', 'particle distribution', 'deposited material', 'velocity-inlet', 'mass-flow-inlet', 'pressure-inlet', 'pressure-outlet', 'no-slip wall', 'navier-slip', 'symmetry', 'periodic', 'fully developed', 'wall-function', 'impermeable wall', 'specified species concentration', 'specified species flux', 'membrane permeation flux', 'prescribed temperature', 'prescribed heat flux', 'convective heat-transfer',
    'geometry creation and import', 'computational meshing', 'fluid-property definition', 'laminar-flow', 'turbulent-flow modelling', 'porous-media flow', 'species transport', 'salt transport', 'module heat transfer', 'concentration polarisation', 'membrane-wall transport', 'spacer hydrodynamics', 'multiphase flow', 'particle transport and deposition', 'wall shear stress', 'pressure-drop prediction', 'mixing analysis', 'fouling-layer', 'crystallisation and particle formation', 'transient simulation', 'user-defined physical model', 'mesh-independence', 'numerical convergence monitoring', 'scientific visualisation'],
  equationsNote: 'Scope of the channel study: two-dimensional, incompressible, constant-property flow on a Cartesian grid (uniform in x, wall-clustered in y) with solids represented by blocked cells (stair-step surfaces). A 2-D section represents filaments transverse to the flow; diamond or woven three-dimensional spacer meshes need a 3-D solver, so treat friction and Sherwood numbers as section values and calibrate the 1-D multipliers against element data. Steady runs are valid while the flow is steady (roughly channel Reynolds number below 300–400 with filaments); above that use the transient mode. Turbulence: algebraic mixing length, standard k–ε with log-law wall functions (first cell at y⁺ > 11.6), Wilcox k–ω and Menter k–ω SST (wall functions or integration to the wall), each with transport equations for k and ε or ω. The Reynolds-stress option is the explicit algebraic (Wallin–Johansson) solution of the stress-transport equations on k–ω with a bounded effective C_μ — the differential Reynolds-stress transport model itself is not solved. LES uses the Smagorinsky sub-grid model on the 2-D grid: without vortex stretching it is indicative only. Compressible flow (Euler equations, optionally with viscous stress, heat conduction and wall friction) is solved in one dimension for an ideal gas — shock tube and quasi-1-D nozzle — not in the 2-D channel. Maxwell–Stefan diffusion is solved as a ternary film across the polarisation layer whose thickness comes from the CFD mass-transfer coefficient, not as a coupled 2-D multicomponent field. The second phase (volume of fluid with THINC/WLIC, level set, or the Eulerian–Eulerian dispersed phase in its algebraic-slip limit) is transported on the solved velocity field: one-way coupling, no surface tension, and the level set is not volume-conserving. Precipitation transports one sparingly soluble salt with first-order wall crystallisation and a four-moment population balance (primary nucleation, linear growth, no aggregation or breakage). The fouling layer feeds back through its hydraulic resistance, not by narrowing the passage. Conjugate heat transfer conducts through the blocked cells. The lattice-Boltzmann option (D2Q9, BGK, laminar, no-slip) supplies the starting field and a comparison; the reported results are those of the finite-volume solver. The regression closure is a cross-validated power law fitted to 5–10 extra solver runs and is valid only inside the sampled range. The pressure inlet is a flow-rate controller for steady runs. The periodic option recycles the outlet-plane profile (and turbulence quantities) to the inlet; permeation is retained at the walls. Bounded QUICK is formulated for uniform spacing and is applied unchanged on the clustered y-grid.',

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
      SEL('study', 'Study', 'channel', [['channel', 'Channel / equipment flow (2-D incompressible)'], ['shock', 'Compressible gas: shock tube / pressure-wave in a vapour line (1-D)'], ['nozzle', 'Compressible gas: converging–diverging nozzle (quasi-1-D)']], 'Shock tube: sudden opening of a valve or rupture disc between two gas states. Nozzle: motive nozzle of a steam ejector or a choked vent.'),
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
      SEL('turb', 'Turbulence', 'laminar', [['laminar', 'Laminar'], ['ml', 'RANS: mixing-length eddy viscosity + wall function'], ['ke', 'RANS: standard k–ε with wall functions'], ['kw', 'RANS: k–ω (Wilcox)'], ['sst', 'RANS: k–ω SST (Menter)'], ['earsm', 'RANS: explicit algebraic Reynolds stress on k–ω'], ['les', 'LES: Smagorinsky sub-grid model (transient, 2-D)']], 'Spacer channels are laminar or transitional; use a RANS closure for ducts above Re ≈ 3000. k–ε needs the first cell at y⁺ > 11.6 (wall refinement 1–3); k–ω and SST also integrate to the wall. The Reynolds-stress option is the explicit algebraic solution of the stress-transport equations (not the full differential model). LES in 2-D lacks vortex stretching and is indicative only.'),
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
    { group: 'Second phase (multiphase)', tab: 'setup', help: 'A second phase transported on the solved velocity field: a sharp interface (gas bubble, air slug or displacing liquid) by volume-of-fluid or level-set, or a dilute dispersed phase (particles, droplets, micro-bubbles) by the Eulerian–Eulerian model in its algebraic-slip limit. One-way coupling: the second phase does not alter the flow and surface tension is not included.', fields: [
      SEL('mp', 'Second-phase model', 'off', [['off', 'None'], ['vof', 'Volume of fluid (sharp interface, conservative)'], ['ls', 'Level set (signed-distance interface)'], ['ee', 'Eulerian–Eulerian dispersed phase with slip velocity']], 'Volume of fluid conserves the phase volume to round-off; the level set gives smooth interface geometry but loses or gains a little area.'),
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
    if (!(v.Uin > 0) && (v.inletBC ?? 'velocity') === 'velocity') throw new Error('Enter a mean inlet velocity greater than zero — a membrane channel without cross-flow has no steady state.');
    const c = caseConfig(v);
    let lbm = null;
    if (v.engine === 'lbm') {
      if (c.o.turb || c.o.wallB !== 'noslip' || c.o.wallT !== 'noslip' || c.o.creeping || c.o.pInlet) lbm = { note: 'The lattice-Boltzmann start is available for laminar flow with no-slip walls and a velocity or mass-flow inlet; the finite-volume solver was used alone.' };
      else { ctx?.progress?.(0.01, 'Lattice-Boltzmann flow'); lbm = lbmStart(c, v); if (lbm.init) c.o.init = lbm.init; }
    }
    const r = await solveChannel(c.o, ctx), q = post(c, r), W = [], { fl, L, H, nx, ny } = c, o = c.o, one = channel1D(v.inletBC && v.inletBC !== 'velocity' ? { ...v, Uin: r.Uref } : v);
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
    if (!r.converged) W.push({ level: 'warn', msg: o.steady ? `The steady solution did not reach the tolerance in ${r.iters} iterations (continuity residual ${fmt(r.hist.mass.at(-1), 2)}). The flow is probably unsteady: use the transient mode, the hybrid scheme, a lower relaxation factor or more iterations.` : `The transient run stopped at ${fmt(r.time, 3)} s before the requested ${fmt(o.tEnd, 3)} s — raise the maximum number of time steps.` });
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
    const outputs = { dpPerM: q.dpdx, frictionFactor: q.f, sherwood: shDev, kMass: kDev, cpFactor: sp ? q.cpMean : null, wallShear: q.tauMean, kMultiplier: kMult ? clamp(kMult, 0.2, 5) : null, reynolds: q.Re, schmidt: q.Sc, maxWallConc: sp ? q.cwMax : null, fluxLMH: memb ? q.Jmean / LMH : null, recirculationFraction: q.recirc, nusselt: nuDev, converged: r.converged ? 1 : 0, iterations: r.iters };
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
      { label: 'Specific pumping power', value: pumpW, unit: 'W/m² of wall', help: 'Hydraulic power dissipated per unit active wall area' },
    ];
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
    if (v.inletBC === 'pressure') { kpis.push({ label: 'Inlet gauge pressure (target)', value: v.pInlet, unit: 'Pa' }, { label: 'Resulting mean velocity', value: r.Uref, unit: 'm/s', help: 'Found by the flow-rate controller so that the inlet pressure equals the target' }); sumRows.push(['Inlet condition', 'pressure inlet'], ['Inlet pressure reached, first cell column (Pa)', q.pm[0]]); if (!o.steady) warn('The pressure inlet is a flow-rate controller for steady runs; this transient run kept the initial flow-rate estimate.'); }
    else if (v.inletBC === 'massflow') sumRows.push(['Inlet condition', `mass-flow inlet, ${fmt(v.mdot, 4)} kg/s per metre width`]);
    if (v.creeping) { info('Creeping-flow (Stokes) limit: the convective terms of the momentum equations are dropped, so the pressure drop is exactly proportional to the flow rate and the flow is reversible.'); sumRows.push(['Momentum convection', 'off (Stokes flow)']); }
    if (r.wallB !== 'noslip' || r.wallT !== 'noslip') {
      const us = (top) => { const j = top ? ny - 1 : 0; let a = 0, m = 0; for (let i = Math.round(nx / 2); i < nx; i++) if (!r.solid[j * nx + i]) { const wt = top ? r.wallT : r.wallB; a += wt === 'sym' ? q.uc[j * nx + i] : wt === 'slip' ? ((top ? q.tauT : q.tauB)[i] * o.slipLen) / fl.mu : 0; m++; } return m ? a / m : 0; };
      kpis.push({ label: r.wallB === 'slip' ? 'Wall slip velocity (downstream half)' : 'Velocity on the symmetry plane', value: Math.max(us(0), us(1)), unit: 'm/s', help: r.wallB === 'slip' ? 'Navier slip: u_wall = b·∂u/∂y with slip length b' : 'Zero shear and zero normal velocity on the symmetry plane' });
      sumRows.push(['Bottom wall', r.wallB === 'slip' ? `Navier slip, b = ${fmt(v.slipLen, 3)} µm` : r.wallB === 'sym' ? 'symmetry plane' : 'no-slip'], ['Top wall', r.wallT === 'slip' ? `Navier slip, b = ${fmt(v.slipLen, 3)} µm` : r.wallT === 'sym' ? 'symmetry plane' : 'no-slip']);
    }
    // --- two-equation closures and LES
    if (r.tk) {
      const isE = r.tm === 'ke', yPlus = (0.5 * dy0 * r.utau) / (fl.mu / fl.rho), ist = clamp(Math.round(0.85 * nx), 0, nx - 1);
      let kmax = 0, ntMax = 0;
      for (let P = 0; P < n; P++) if (!r.solid[P]) { kmax = Math.max(kmax, r.tk[P]); ntMax = Math.max(ntMax, r.mue[P] / fl.mu - 1); }
      plots.push({ ...base, title: 'Turbulent kinetic energy', zlabel: 'k', zunit: 'm²/s²', z: q.field(r.tk), cmap: 'turbo' });
      plots.push({ ...base, title: isE ? 'Turbulent dissipation rate ε' : 'Specific dissipation rate ω (log₁₀)', zlabel: isE ? 'ε' : 'log₁₀ ω', zunit: isE ? 'm²/s³' : 'log₁₀(1/s)', z: q.field(isE ? r.te : r.te.map((w) => Math.log10(Math.max(w, 1e-30)))), cmap: 'viridis' });
      const ut = Math.max(Math.sqrt(Math.abs(q.tauB[ist]) / fl.rho), 1e-12), half = Math.ceil(ny / 2), yp = [], up = [], kp = [], uvp = [], uup = [], vvp = [];
      for (let j = 0; j < half; j++) { const P = j * nx + ist; if (r.solid[P]) continue; yp.push(Math.max((r.yc[j] * ut) / (fl.mu / fl.rho), 1e-3)); up.push(q.uc[P] / ut); kp.push(r.tk[P] / (ut * ut)); uvp.push(-r.rs.uv[P] / (ut * ut)); uup.push(r.rs.uu[P] / (ut * ut)); vvp.push(r.rs.vv[P] / (ut * ut)); }
      if (yp.length > 2 && r.wallB === 'noslip') {
        plots.push({ type: 'line', title: 'Velocity profile in wall units (bottom wall, x = 85 % of length)', xlabel: 'y⁺', ylabel: 'u⁺', logx: true, series: [{ name: 'Computed', x: yp, y: up, mode: 'both' }, { name: 'Log law u⁺ = ln(y⁺)/0.41 + 5.2', x: yp, y: yp.map((y) => (y > 11.6 ? Math.log(y) / KAPPA + 5.2 : y)), dash: true }] });
        plots.push({ type: 'line', title: 'Reynolds stresses in wall units', xlabel: 'y⁺', ylabel: 'stress / u_τ²', logx: true, series: [{ name: 'k⁺', x: yp, y: kp }, { name: '−u′v′⁺', x: yp, y: uvp }, { name: 'u′u′⁺', x: yp, y: uup }, { name: 'v′v′⁺', x: yp, y: vvp }], note: r.tm === 'earsm' ? 'Normal-stress anisotropy comes from the explicit algebraic Reynolds-stress closure.' : 'Boussinesq closures give nearly isotropic normal stresses in simple shear (u′u′ ≈ v′v′ ≈ ⅔k).' });
      }
      const jm = Math.min(Math.round(0.2 * ny), half - 1), Pm = jm * nx + ist, aXX = r.tk[Pm] > 0 ? (r.rs.uu[Pm] - r.rs.vv[Pm]) / (2 * r.tk[Pm]) : 0, a12 = r.tk[Pm] > 0 ? r.rs.uv[Pm] / r.tk[Pm] : 0;
      kpis.push({ label: 'Friction velocity u_τ', value: r.utau, unit: 'm/s' }, { label: 'First-cell y⁺', value: yPlus, unit: '', status: isE && yPlus < 11.6 ? 'warn' : 'ok' }, { label: 'Peak turbulence intensity √(⅔k)/U', value: Math.sqrt((2 / 3) * kmax) / r.Uref, unit: '' }, { label: 'Maximum eddy-viscosity ratio ν_t/ν', value: ntMax, unit: '' });
      if (r.tm === 'earsm') kpis.push({ label: 'Normal-stress anisotropy (u′u′ − v′v′)/2k', value: aXX, unit: '', help: 'Evaluated at 20 % of the gap, x = 85 % of the length; 0 for a Boussinesq closure' }, { label: 'Shear-stress ratio u′v′/k', value: a12, unit: '', help: '≈ −0.30 in an equilibrium boundary layer' });
      if (isE && yPlus < 11.6) warn(`Standard k–ε with wall functions needs the first cell in the log layer (y⁺ > 11.6, ideally 30–300); it is at y⁺ = ${fmt(yPlus, 3)}. Lower the wall refinement toward 1–3 or use the k–ω SST model, which integrates to the wall.`);
      tables.push({ title: 'Turbulence closure summary', columns: ['Item', 'Value'], rows: [['Closure', TURB_NAMES[r.tm]], ['Inlet turbulence intensity (%)', v.tuIn], ['Inlet length scale (mm)', o.lTurb * 1e3], ['Friction velocity (m/s)', r.utau], ['First-cell y⁺', yPlus], ['Peak k (m²/s²)', kmax], ['Peak ν_t/ν', ntMax], ['Skin-friction coefficient C_f = τ_w/(½ρU²)', q.tauMean / (0.5 * fl.rho * r.Uref ** 2)], ['Dean correlation 0.073 Re_H^−0.25 (plane channel)', 0.073 * (q.Re / 2) ** -0.25], ['Shear-stress ratio u′v′/k at 20 % of the gap', a12], ['Anisotropy (u′u′ − v′v′)/2k at 20 % of the gap', aXX]] });
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
      const ph = runPhase(r, v, fl), ee = ph.method === 'ee', names = { vof: 'Volume of fluid (THINC/WLIC)', ls: 'Level set', ee: 'Eulerian–Eulerian dispersed phase (algebraic slip)' };
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

  mesh: { name: 'Spatial grid (nx × ny)', keys: ['nx', 'ny'], min: 12, note: 'Both directions are refined together with the wall-clustering ratio held constant. All other inputs are unchanged.',
    metrics: [{ label: 'Pressure gradient', unit: 'Pa/m', get: (r) => r.outputs.dpPerM ?? 0 }, { label: 'Mean Sherwood number', unit: '–', get: (r) => r.outputs.sherwood ?? 0 }, { label: 'Maximum wall concentration', unit: 'g/L', get: (r) => r.outputs.maxWallConc ?? 0 }] },

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
        if (tm === 'ke') add('Turbulent channel, k–ε: structure parameter −u′v′/k in the log layer', 0.3, -t.rs.uv[Pl] / t.tk[Pl], 0.03, '√C_μ = 0.30 where production balances dissipation');
        if (tm === 'earsm') add('Algebraic Reynolds stress: normal-stress anisotropy (u′u′ − v′v′)/2k', 0.26, (t.rs.uu[Pl] - t.rs.vv[Pl]) / (2 * t.tk[Pl]), 0.08, 'Log-layer value of the Wallin–Johansson closure (0 for an eddy-viscosity model; experiments ≈ 0.25–0.30)');
      }
    }
    { // LES: the Smagorinsky term must vanish on a resolved laminar flow
      const l = await solveChannel({ ...base, L: 2e-3, nx: 10, ny: 16, inlet: 'parabolic', turb: 'les', steady: false, tEnd: (2 * 2e-3) / U, cfl: 2, maxIter: 400 });
      let nt = 0; for (let P = 0; P < l.mue.length; P++) nt = Math.max(nt, l.mue[P] / mu - 1);
      add('LES (Smagorinsky): laminar limit keeps the Poiseuille pressure gradient', 1, gradX(l, 2, 7) / exact, 0.02, `Transient filtered equations at Re = 100; largest sub-grid viscosity ratio ${nt.toExponential(1)} (ratio)`);
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
    { // second phase: Zalesak disc (VOF), translating circle (level set), ideal settler (Eulerian–Eulerian)
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
      add('Eulerian–Eulerian dispersed phase: ideal-settler capture', (Math.abs(vset) * Ls2) / (Us * Hs), (er.hist.dep[hn] - er.hist.dep[hm]) / (er.hist.inn[hn] - er.hist.inn[hm]), 0.005, 'Hazen: captured fraction = v_s L/(U H) in plug flow with settling velocity v_s');
      add('Eulerian–Eulerian dispersed phase: volume balance', 0, (er.vol0 + er.inn - er.vol - er.out - er.hist.dep[hn]) / er.inn, 1e-9, '(initial + inflow − hold-up − outflow − deposit) ÷ inflow');
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
