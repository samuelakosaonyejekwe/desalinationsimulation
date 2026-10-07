// Suite 4 — Flow in membranes, channels and equipment (CFD).
// Two-dimensional finite-volume solver on a staggered (MAC) Cartesian grid, uniform in x and optionally
// clustered toward the walls in y: SIMPLE/SIMPLEC pressure–velocity coupling with an incomplete-Cholesky
// preconditioned conjugate-gradient pressure solver, immersed solids by cell blocking, salt transport with
// solution–diffusion membrane walls (concentration polarisation), an energy equation, an algebraic
// mixing-length eddy viscosity, a Darcy–Forchheimer porous zone and Lagrangian particle tracking.
import { clamp, linspace, fmt, gci, rng, mean } from '../core/num.js';
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
  const Uref = Math.abs(Qin) / H || 1e-12;
  for (let i = 1; i <= nx; i++) { // initial guess: plug flow through the open height of every column
    const isOpen = (j) => (i === nx ? !solid[j * nx + nx - 1] : !ublk[j * nu1 + i]);
    let open = 0;
    for (let j = 0; j < ny; j++) if (isOpen(j)) open += dy[j];
    for (let j = 0; j < ny; j++) u[j * nu1 + i] = isOpen(j) && open > 0 ? Qin / open : 0;
  }
  for (let j = 0; j < ny; j++) u[j * nu1] = uin[j];

  // membrane / wall permeation velocities (m/s, positive out of the channel)
  const Jb = new Float64Array(nx), Jt = new Float64Array(nx);
  const setWallV = () => { for (let i = 0; i < nx; i++) { v[i] = solid[i] ? 0 : -Jb[i]; v[ny * nx + i] = solid[(ny - 1) * nx + i] ? 0 : Jt[i]; } };

  // effective viscosity (cells) and wall distance for the mixing-length closure
  const mue = new Float64Array(n).fill(mu), dist = new Float64Array(n);
  if (o.turb) {
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) dist[j * nx + i] = solid[j * nx + i] ? 0 : Math.min(yc[j], H - yc[j]);
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
  const tauB = new Float64Array(nx), tauT = new Float64Array(nx);
  const wallShear = () => { // wall shear stress on the channel walls from the wall-adjacent cell
    for (let i = 0; i < nx; i++) {
      for (const top of [0, 1]) {
        const j = top ? ny - 1 : 0, P = j * nx + i, uc = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]), y = 0.5 * dy[j];
        let tau = solid[P] ? 0 : (mu * uc) / y;
        if (o.turb && !solid[P]) { const ut = uTau(Math.abs(uc), y); if ((y * ut) / nu > 11.6) tau = rho * ut * ut * Math.sign(uc); }
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
    for (let i = 0; i < nx; i++) { if (!solid[i]) { s += Math.abs(tauB[i]); m++; } if (!solid[(ny - 1) * nx + i]) { s += Math.abs(tauT[i]); m++; } }
    utauG = Math.sqrt(Math.max(s / Math.max(m, 1), 1e-30) / rho);
    const uc = (i, j) => 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]), vc = (i, j) => 0.5 * (v[j * nx + i] + v[(j + 1) * nx + i]);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const P = j * nx + i;
      if (solid[P]) { mue[P] = mu; continue; }
      const uy = ((j < ny - 1 ? uc(i, j + 1) : 0) - (j > 0 ? uc(i, j - 1) : 0)) / ((j < ny - 1 ? yc[j + 1] : H) - (j > 0 ? yc[j - 1] : 0));
      const vx = (vc(Math.min(i + 1, nx - 1), j) - vc(Math.max(i - 1, 0), j)) / (2 * dx);
      const ux = (u[j * nu1 + i + 1] - u[j * nu1 + i]) / dx, vy = (v[(j + 1) * nx + i] - v[j * nx + i]) / dy[j];
      const S = Math.sqrt(2 * (ux * ux + vy * vy) + (uy + vx) * (uy + vx)), d = dist[P];
      const lm = Math.min(KAPPA * d, 0.09 * H) * (1 - Math.exp((-d * utauG) / nu / 26));
      mue[P] = 0.5 * mue[P] + 0.5 * (mu + rho * Math.min(lm * lm * S, 2e4 * nu));
    }
  };

  // geometric conductances (multiplied by the local viscosity at assembly)
  const Su = stencil(nu1, ny), Sv = stencil(nx, ny + 1);
  const gxu = new Float64Array(Su.Dx.length), gyu = new Float64Array(Su.Dy.length), gxv = new Float64Array(Sv.Dx.length), gyv = new Float64Array(Sv.Dy.length);
  for (let j = 0; j < ny; j++) for (let k = 1; k <= nx; k++) gxu[j * (nu1 + 1) + k] = dy[j] / dx;
  for (let jf = 0; jf <= ny; jf++) for (let i = 1; i < nx; i++) {
    const lo = jf > 0 ? ublk[(jf - 1) * nu1 + i] : 1, hi = jf < ny ? ublk[jf * nu1 + i] : 1;
    gyu[jf * nu1 + i] = lo && hi ? 0 : lo ? dx / (0.5 * dy[jf]) : hi ? dx / (0.5 * dy[jf - 1]) : dx / dyc[jf];
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
      Su.Fx[q] = rho * 0.5 * (u[j * nu1 + k - 1] + u[j * nu1 + k]) * dy[j];
      Su.Dx[q] = (turb ? mue[j * nx + k - 1] : mu) * gxu[q];
    }
    for (let jf = 0; jf <= ny; jf++) for (let i = 1; i < nx; i++) {
      const q = jf * nu1 + i;
      Su.Fy[q] = rho * 0.5 * (v[jf * nx + i - 1] + v[jf * nx + i]) * dx;
      Su.Dy[q] = (turb ? muCorner(i, jf) : mu) * gyu[q];
    }
    if (turb) for (let i = 1; i < nx; i++) for (const top of [0, 1]) { // log-law wall function on the channel walls
      const j = top ? ny - 1 : 0, k = j * nu1 + i;
      if (ublk[k]) continue;
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
      Su.b[k] += (p[j * nx + i - 1] - p[j * nx + i]) * dy[j] + (1 - al) * ap * u[k];
      Su.aP[k] = ap; du[k] = dy[j] / (simplec ? Math.max(ap - nb, 0.05 * ap) : ap);
    }
    uPrev.set(u); // previous iterate, for the change norm
    lineSolve(Su, u, 1);
    for (let k = 0; k < u.length; k++) { const d = Math.abs(u[k] - uPrev[k]); if (d > dUmax) dUmax = d; }
    // ---- v momentum
    for (let jf = 1; jf < ny; jf++) for (let k = 0; k <= nx; k++) {
      const q = jf * (nx + 1) + k;
      Sv.Fx[q] = rho * 0.5 * (u[(jf - 1) * nu1 + k] * dy[jf - 1] + u[jf * nu1 + k] * dy[jf]);
      Sv.Dx[q] = (turb ? muCorner(k, jf) : mu) * gxv[q];
    }
    for (let jj = 1; jj <= ny; jj++) for (let i = 0; i < nx; i++) {
      const q = jj * nx + i;
      Sv.Fy[q] = rho * 0.5 * (v[(jj - 1) * nx + i] + v[jj * nx + i]) * dx;
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
  const makeScalar = (sp, init) => {
    const phi = new Float64Array(n).fill(init), wB = new Float64Array(nx).fill(init), wT = new Float64Array(nx).fill(init), pB = new Float64Array(nx), pT = new Float64Array(nx);
    const Dc = (P) => sp.D + (o.turb ? (mue[P] - mu) / rho / (sp.sct || 0.85) : 0);
    /** One assembly + line solve. rdt = 1/Δt for transient steps; old = previous time level. */
    const step = (rdt, old, freeze) => {
      for (let j = 0; j < ny; j++) for (let k = 0; k <= nx; k++) {
        const q = j * (nx + 1) + k;
        Sc.Fx[q] = u[j * nu1 + k] * dy[j];
        Sc.Dx[q] = k === nx ? 0 : k === 0 ? (solid[j * nx] || !(uin[j] > 0) ? 0 : (Dc(j * nx) * dy[j]) / (0.5 * dx)) : solid[j * nx + k - 1] || solid[j * nx + k] ? 0 : (0.5 * (Dc(j * nx + k - 1) + Dc(j * nx + k)) * dy[j]) / dx;
      }
      for (let jf = 0; jf <= ny; jf++) for (let i = 0; i < nx; i++) {
        const q = jf * nx + i;
        Sc.Fy[q] = v[q] * dx;
        Sc.Dy[q] = jf === 0 || jf === ny || solid[q - nx] || solid[q] ? 0 : (0.5 * (Dc(q - nx) + Dc(q)) * dx) / dyc[jf];
      }
      for (let j = 0; j < ny; j++) { Sc.bW[j] = sp.inVal; Sc.bE[j] = phi[j * nx + nx - 1]; }
      for (const top of [0, 1]) { // fixed-value walls enter through the boundary conductance
        const w = top ? sp.top : sp.bot, j = top ? ny - 1 : 0;
        if (w.type === 'fixed') for (let i = 0; i < nx; i++) { if (solid[j * nx + i]) continue; Sc.Dy[(top ? ny : 0) * nx + i] = (Dc(j * nx + i) * dx) / (0.5 * dy[j]); (top ? Sc.bN : Sc.bS)[i] = w.val; }
      }
      assemble(Sc, phi, scheme, fixS);
      let salt = 0;
      for (const top of [0, 1]) {
        const w = top ? sp.top : sp.bot, j = top ? ny - 1 : 0, J = top ? Jt : Jb, wall = top ? wT : wB, perm = top ? pT : pB;
        for (let i = 0; i < nx; i++) {
          const P = j * nx + i, dl = 0.5 * dy[j], D = Dc(P);
          if (solid[P]) { wall[i] = sp.inVal; perm[i] = 0; continue; }
          if (w.type === 'membrane') {
            // film solution across the wall half-cell: c_w = c_P e^Pe / (1 + (1 − R)(e^Pe − 1)), R = J/(J + B)
            let dJ = 0;
            if (!freeze) {
              const R0 = J[i] > 0 ? J[i] / (J[i] + sp.B) : 0, e0 = Math.exp(Math.min(6, (J[i] * dl) / D)), E0 = e0 / (1 + (1 - R0) * (e0 - 1)), cw = phi[P] * E0;
              const dP = sp.dP + (p[P] - sp.pRef()), pw = sp.pi(cw), Jn = Math.max(0, sp.A * (dP - (pw - sp.pi((1 - R0) * cw))));
              J[i] = 0.5 * J[i] + 0.5 * Jn;
              if (Jn > 0 && cw > 0) dJ = (-sp.A * E0 * (sp.pi(1.01 * cw) - pw)) / (0.01 * cw); // ∂J/∂c_P
            }
            // water leaves through the wall at J, salt only at J(1 − R)c_w: the difference concentrates the wall cell.
            // Source S = Δx·J·β·c_P, linearised (Newton) only where that makes it a stabilising sink.
            const R = J[i] > 0 ? J[i] / (J[i] + sp.B) : 0, e = Math.exp(Math.min(6, (J[i] * dl) / D)), Ew = e / (1 + (1 - R) * (e - 1));
            const beta = 1 - (1 - R) * Ew, S0 = dx * J[i] * beta * phi[P], S1 = dx * beta * (J[i] + phi[P] * dJ);
            if (S1 < 0) { Sc.aP[P] -= S1; Sc.b[P] += S0 - S1 * phi[P]; } else Sc.b[P] += S0;
            wall[i] = phi[P] * Ew; perm[i] = (1 - R) * wall[i]; salt += J[i] * perm[i] * dx;
          } else if (w.type === 'flux') { Sc.b[P] += w.val * dx; wall[i] = phi[P] + (w.val * dl) / D; }
          else if (w.type === 'conv') { const U = 1 / (1 / w.h + dl / D); Sc.aP[P] += U * dx; Sc.b[P] += U * dx * w.ext; wall[i] = phi[P] + (U * (w.ext - phi[P]) * dl) / D; }
          else if (w.type === 'fixed') wall[i] = w.val;
          else wall[i] = phi[P];
        }
      }
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
        if (solid[P]) { Sc.aW[P] = Sc.aE[P] = Sc.aS[P] = Sc.aN[P] = 0; Sc.aP[P] = 1; Sc.b[P] = sp.inVal; continue; }
        if (rdt) { const m = rdt * dx * dy[j]; Sc.aP[P] += m; Sc.b[P] += m * old[P]; }
        if (!(Sc.aP[P] > 1e-300)) { Sc.aP[P] = 1; Sc.b[P] = phi[P]; }
      }
      pp.set(phi);
      lineSolve(Sc, phi, 1);
      let d = 0;
      for (let P = 0; P < n; P++) { const e = Math.abs(phi[P] - pp[P]); if (e > d) d = e; }
      return { change: d, salt };
    };
    return { phi, wB, wT, pB, pT, step };
  };
  const pIn = () => { let s = 0, m = 0; for (let j = 0; j < ny; j++) if (!solid[j * nx]) { s += p[j * nx] * dy[j]; m += dy[j]; } return m ? s / m : 0; };
  let pRefVal = 0;
  const spc = o.species ? makeScalar({ ...o.species, pRef: () => pRefVal, inVal: o.species.c0, bot: { type: o.species.bot || 'none', val: o.species.cwBot ?? o.species.cw }, top: { type: o.species.top || 'none', val: o.species.cwTop ?? o.species.cw } }, o.species.c0) : null;
  const eng = o.energy ? makeScalar({ D: o.energy.alpha, sct: 0.9, inVal: o.energy.Tin, bot: o.energy.bot, top: o.energy.top }, o.energy.Tin) : null;
  if (spc) for (const top of [0, 1]) if ((top ? o.species.top : o.species.bot) === 'membrane') { const J = top ? Jt : Jb, J0 = Math.max(0, o.species.A * (o.species.dP - o.species.pi(o.species.c0))); for (let i = 0; i < nx; i++) J[i] = solid[(top ? ny - 1 : 0) * nx + i] ? 0 : J0; }
  setWallV();

  const hist = { it: [], mass: [], dU: [], scal: [] }, tick = async (f, msg) => { if (ctx?.progress) ctx.progress(f, msg); if (ctx?.tick) await ctx.tick(); };
  const tol = o.tol ?? 1e-5, maxIter = o.maxIter ?? 600;
  let iters = 0, converged = false, probe = null, time = 0;
  const flowSteady = async (maxIt, f0, f1) => {
    let ok = false;
    for (let k = 0; k < maxIt; k++) {
      if (o.turb && k % 3 === 0) updateTurb();
      const r = iterate(0);
      iters++; hist.it.push(iters); hist.mass.push(Math.max(r.mass, 1e-16)); hist.dU.push(Math.max(r.dU, 1e-16));
      if (!Number.isFinite(r.mass) || !Number.isFinite(r.dU)) throw new Error('The flow solution diverged. Lower the velocity-relaxation factor, use the hybrid scheme or refine the grid.');
      if (k % 12 === 0) await tick(f0 + ((f1 - f0) * k) / maxIt, `Flow iteration ${iters}: continuity residual ${r.mass.toExponential(1)}`);
      if (k > 3 && r.mass < tol && r.dU < tol) { ok = true; break; }
    }
    return ok;
  };
  const scalarSteady = async (sc, maxIt, freeze, scale) => {
    let last = Infinity;
    for (let k = 0; k < maxIt; k++) {
      if (sc === spc) pRefVal = pIn();
      const r = sc.step(0, null, freeze);
      if (sc === spc && !freeze) setWallV();
      last = r.change / scale; hist.scal.push(Math.max(last, 1e-16));
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
  if (o.turb) updateTurb(); else wallShear();
  return { nx, ny, dx, L, H, yf, yc, dy, dyc, solid, u, v, p, mue, Jb, Jt, tauB, tauT, Qin, Uref, uin, hist, iters, converged, scalRes, spc, eng, probe, time, utau: utauG, nu1 };
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
  const species = v.species === 'off' ? null : { c0: v.c0, D: fl.D, A: (v.A * LMH) / 1e5, B: v.B * LMH, dP: v.dPtm * 1e5, pi: fl.pi, cw: v.cwFixed, bot: sb ? v.species : 'none', top: st ? v.species : 'none' };
  const wall = (on) => (!on || v.thWall === 'none' ? { type: 'none' } : v.thWall === 'fixed' ? { type: 'fixed', val: v.Tw } : v.thWall === 'flux' ? { type: 'flux', val: v.qw / rc } : { type: 'conv', h: v.Uw / rc, ext: v.Text });
  const energy = v.energy ? { alpha: fl.k / rc, Tin: v.T, bot: wall(eb), top: wall(et) } : null;
  const porous = v.porous ? { x0: (Math.min(v.porX0, v.porX1) / 100) * L, x1: (Math.max(v.porX0, v.porX1) / 100) * L, K: v.porK, cF: v.porCF } : null;
  const o = { L, H, nx, ny, stretch, solid: mk.solid, rho: fl.rho, mu: fl.mu, Uin: v.Uin, inlet: v.inlet, scheme: v.scheme, steady: v.mode !== 'transient', maxIter: clamp(Math.round(v.maxIter), 5, 6000), tol: clamp(v.tol, 1e-9, 1e-2),
    alphaU: clamp(v.alphaU, 0.2, 0.95), cfl: clamp(v.cfl, 0.2, 10), tEnd: (v.tFlow * L) / Math.max(v.Uin, 1e-9), turb: v.turb === 'ml', porous, species, energy };
  return { fl, geo, mk, L, H, nx, ny, g, o };
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
  const transfer = (sc, D, scale, flux) => {
    const bulk = colMean(sc.phi).map((q) => q.cup), wB = st && sc === r.spc ? avg(null, st.cB) : Array.from(sc.wB), wT = st && sc === r.spc ? avg(null, st.cT) : Array.from(sc.wT);
    const side = (top) => {
      const w = top ? wT : wB, j = top ? ny - 1 : 0, k = [], fl2 = [], dr = [];
      for (let i = 0; i < nx; i++) {
        const P = j * nx + i, drive = w[i] - bulk[i], q = solid[P] ? NaN : flux ? flux(top, i, w[i]) : (D * (w[i] - sc.phi[P])) / (0.5 * dy[j]);
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
    o.sp = transfer(r.spc, fl.D, c.o.species.c0 || 1, (top, i, cw) => (memb(top) ? (top ? JT : JB)[i] * (cw - (top ? r.spc.pT : r.spc.pB)[i]) : (fl.D * (cw - r.spc.phi[(top ? ny - 1 : 0) * nx + i])) / (0.5 * dy[top ? ny - 1 : 0])));
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
  o.yp = yp;
  o.mask = yp.map((_, q) => xc.map((__, i) => !!solid[jn[q] * nx + i]));
  o.field = (a, scale = 1, hole = NaN) => yp.map((y, q) => xc.map((_, i) => {
    if (solid[jn[q] * nx + i]) return hole;
    const A = j0[q] * nx + i, Bq = A + nx, w = clamp((y - yc[j0[q]]) / (yc[j0[q] + 1] - yc[j0[q]]), 0, 1);
    return scale * (solid[A] ? a[Bq] : solid[Bq] ? a[A] : a[A] * (1 - w) + a[Bq] * w);
  }));
  return o;
}

const F = (key, label, unit, value, min, max, help, extra = {}) => ({ key, label, unit, value, min, max, help, ...extra });
const SEL = (key, label, value, options, help, extra = {}) => ({ key, label, type: 'select', value, options: options.map(([v, l]) => ({ value: v, label: l })), help, ...extra });
const isSpacer = (v) => v.geom === 'spacer', isImport = (v) => v.geom === 'import', hasSpecies = (v) => v.species !== 'off';

const suite = {
  id: 'cfd', num: 4, title: 'Flow in Membranes, Channels & Equipment (CFD)', short: 'CFD', icon: '🌀',
  tagline: 'Two-dimensional finite-volume CFD of spacer-filled membrane channels, ducts and imported shapes with salt, heat and particle transport.',
  description: 'Solves the incompressible Navier–Stokes equations on a staggered Cartesian grid with SIMPLE-type pressure–velocity coupling and a preconditioned conjugate-gradient pressure solver. Spacer filaments, steps, baffles or imported CAD sections are immersed as blocked cells. Salt transport is coupled to solution–diffusion membrane walls, so concentration polarisation, local permeate flux, wall shear, friction factor and Sherwood number come straight from the resolved fields and are compared with the Hagen–Poiseuille, Lévêque/Graetz and Schock–Miquel relations.',
  guide: [
    'Choose the geometry: a spacer-filled membrane channel, an empty channel, a sudden expansion, baffles, or an imported STL/OBJ/DXF/GeoJSON section.',
    'Enter the fluid, cross-flow velocity and membrane data (or pull them from the case and the RO suite).',
    'On Model setup pick steady or transient flow, the convection scheme, the wall conditions for salt and heat, and optional turbulence, porous-zone and particle models.',
    'Run. Check the residual history and the conservation closure first, then read the fields, wall profiles and the comparison with correlations.',
    'Use the Mesh tab to quantify numerical uncertainty; the mass-transfer multiplier is offered to the RO design suite.',
  ],
  implemented: ['continuity equation', 'incompressible', 'stokes-flow', 'reynolds-averaged', 'species-conservation', 'convection-diffusion', 'the energy equation', 'fourier', 'fick', 'darcy equation', 'brinkman', 'forchheimer', 'hagen-poiseuille', 'darcy-weisbach',
    'navier-stokes-species', 'navier-stokes-solution-diffusion', 'cfd-concentration-polarization', 'cfd-porous-media', 'cfd-particle-deposition', 'cfd-heat/mass-transfer', 'eulerian-lagrangian',
    'velocity', 'initial pressure field', 'concentration', 'temperature', 'particle distribution', 'deposited material', 'velocity-inlet', 'pressure-outlet', 'no-slip wall', 'periodic', 'fully developed', 'wall-function', 'impermeable wall', 'specified species concentration', 'membrane permeation flux', 'prescribed temperature', 'prescribed heat flux', 'convective heat-transfer',
    'geometry creation and import', 'computational meshing', 'fluid-property definition', 'laminar-flow', 'turbulent-flow modelling', 'porous-media flow', 'species transport', 'salt transport', 'module heat transfer', 'concentration polarisation', 'membrane-wall transport', 'spacer hydrodynamics', 'particle transport and deposition', 'wall shear stress', 'pressure-drop prediction', 'mixing analysis', 'transient simulation', 'mesh-independence', 'numerical convergence monitoring', 'scientific visualisation'],
  equationsNote: 'Scope: two-dimensional, incompressible, constant-property flow on a Cartesian grid (uniform in x, wall-clustered in y) with solids represented by blocked cells (stair-step surfaces). A 2-D section represents filaments transverse to the flow; diamond or woven three-dimensional spacer meshes need a 3-D solver, so treat friction and Sherwood numbers as section values and calibrate the 1-D multipliers against element data. Steady runs are valid while the flow is steady (roughly channel Reynolds number below 300–400 with filaments); above that use the transient mode. Turbulence is limited to an algebraic mixing-length eddy viscosity with van Driest damping and a log-law wall function — k–ε, k–ω SST, Reynolds-stress and LES closures, compressible flow, multiphase (VOF/level-set), population-balance, precipitation and conjugate heat transfer are listed for reference only and are not solved. The periodic option recycles the outlet-plane profile to the inlet; permeation is retained at the walls. Bounded QUICK is formulated for uniform spacing and is applied unchanged on the clustered y-grid.',

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
    { group: 'Models', tab: 'setup', help: 'Governing equations and closures.', fields: [
      SEL('mode', 'Time treatment', 'steady', [['steady', 'Steady state (SIMPLEC iterations)'], ['transient', 'Transient (implicit Euler, vortex shedding)']], 'Use transient when the steady solution will not converge because the wake sheds vortices.'),
      SEL('scheme', 'Convection scheme', 'hybrid', [['hybrid', 'Hybrid central/upwind (robust)'], ['upwind', 'First-order upwind'], ['quick', 'Bounded QUICK (higher order)']], 'Applied to momentum and scalars.'),
      SEL('turb', 'Turbulence', 'laminar', [['laminar', 'Laminar'], ['ml', 'RANS: mixing-length eddy viscosity + wall function']], 'Spacer channels are laminar or transitional; use the RANS option for ducts above Re ≈ 3000.'),
      SEL('species', 'Salt transport and wall condition', 'membrane', [['membrane', 'Membrane walls (permeation, polarisation)'], ['fixed', 'Fixed wall concentration (mass-transfer analogue)'], ['off', 'No species equation']], ''),
      F('cwFixed', 'Wall concentration', 'g/L', 70, 0, 400, '', { showIf: (v) => v.species === 'fixed' }),
      SEL('sides', 'Walls carrying the membrane / fixed concentration', 'both', [['both', 'Both walls'], ['bottom', 'Bottom wall'], ['top', 'Top wall']], 'Spiral-wound feed channels have membrane on both sides; flat-sheet test cells on one.', { showIf: hasSpecies }),
    ] },
    { group: 'Boundary and initial conditions', tab: 'setup', help: 'Walls and filaments are no-slip. The outlet is a zero-gradient pressure outlet. The initial field is plug flow at the inlet concentration and temperature.', fields: [
      SEL('inlet', 'Inlet velocity profile', 'parabolic', [['parabolic', 'Parabolic (fully developed laminar)'], ['uniform', 'Uniform (developing flow)'], ['periodic', 'Periodic: profile recycled from the outlet']], 'The periodic option reproduces the repeating flow deep inside a spacer-filled channel.'),
    ] },
    { group: 'Energy equation', tab: 'setup', help: 'Temperature field with prescribed wall temperature, heat flux or a convective (membrane-distillation style) wall.', fields: [
      { key: 'energy', label: 'Solve the energy equation', type: 'bool', value: false },
      SEL('thWall', 'Thermal wall condition', 'conv', [['conv', 'Convective: q = U (T_ext − T_wall)'], ['fixed', 'Prescribed wall temperature'], ['flux', 'Prescribed heat flux'], ['none', 'Adiabatic']], '', { showIf: (v) => v.energy }),
      SEL('thSides', 'Thermally active walls', 'both', [['both', 'Both walls'], ['bottom', 'Bottom wall'], ['top', 'Top wall']], '', { showIf: (v) => v.energy }),
      F('Tw', 'Wall temperature', '°C', 20, 0, 150, '', { showIf: (v) => v.energy && v.thWall === 'fixed' }),
      F('qw', 'Wall heat flux into the fluid', 'W/m²', -15000, -1e6, 1e6, 'Negative values cool the fluid (heat leaving through a distillation membrane).', { showIf: (v) => v.energy && v.thWall === 'flux' }),
      F('Uw', 'Wall heat-transfer coefficient U', 'W/m²·K', 800, 1, 1e5, 'Membrane plus permeate-side resistance.', { showIf: (v) => v.energy && v.thWall === 'conv' }),
      F('Text', 'External (permeate-side) temperature', '°C', 20, 0, 150, '', { showIf: (v) => v.energy && v.thWall === 'conv' }),
    ] },
    { group: 'Porous zone', tab: 'setup', help: 'Darcy–Forchheimer momentum sink −(μ/K + ρ c_F |u|/√K) u inside a slab of the channel; viscous (Brinkman) stresses are retained.', fields: [
      { key: 'porous', label: 'Add a porous zone', type: 'bool', value: false },
      F('porX0', 'Zone start', '% of length', 40, 0, 100, '', { showIf: (v) => v.porous }),
      F('porX1', 'Zone end', '% of length', 70, 0, 100, '', { showIf: (v) => v.porous }),
      F('porK', 'Permeability K', 'm²', 1e-9, 1e-16, 1e-2, 'Packed 1 mm beads ≈ 1e-9 m²; fine filter media 1e-12 m².', { showIf: (v) => v.porous }),
      F('porCF', 'Forchheimer coefficient c_F', '–', 0.55, 0, 10, 'Ergun-type inertial coefficient (≈ 0.55 for packed beds).', { showIf: (v) => v.porous }),
    ] },
    { group: 'Particle tracking', tab: 'setup', help: 'Lagrangian particles released across the inlet: Stokes drag, gravity settling (−y), Brownian motion and capture on walls and obstacles.', fields: [
      { key: 'particles', label: 'Track particles', type: 'bool', value: false },
      F('nPart', 'Number of particles', '', 300, 10, 3000, '', { showIf: (v) => v.particles, step: 1 }),
      F('dPart', 'Particle diameter', 'µm', 5, 0.01, 2000, '', { showIf: (v) => v.particles }),
      F('rhoPart', 'Particle density', 'kg/m³', 1500, 500, 8000, '', { showIf: (v) => v.particles }),
      F('stick', 'Attachment efficiency', '–', 1, 0, 1, 'Probability that a particle touching a surface stays attached.', { showIf: (v) => v.particles }),
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
    { name: 'Sudden expansion in a brine duct, turbulent, with particles', values: { geom: 'step', H: 100, L: 1400, Uin: 1.2, c0: 65, turb: 'ml', species: 'off', stretch: 3, nx: 120, ny: 30, particles: true, dPart: 60, rhoPart: 2650, alphaU: 0.7, maxIter: 900 } },
  ],

  pull: ({ feed, outputs } = {}) => [
    Number.isFinite(feed?.T) ? { key: 'T', value: feed.T, from: 'Case feed water' } : null,
    outputs?.ro?.streams?.feed?.tds ? { key: 'c0', value: outputs.ro.streams.feed.tds / 1000, from: 'RO design: feed TDS' } : null,
    outputs?.ro?.feedPressureBar ? { key: 'dPtm', value: outputs.ro.feedPressureBar - 1, from: 'RO design: feed pressure (1 bar permeate side)' } : null,
  ].filter(Boolean),
  site: (site) => (Number.isFinite(site?.data?.sst) ? [{ key: 'T', value: site.data.sst, from: 'Sea-surface temperature at site' }] : []),

  async run(v, ctx) {
    if (!(v.Uin > 0)) throw new Error('Enter a mean inlet velocity greater than zero — a membrane channel without cross-flow has no steady state.');
    const c = caseConfig(v), r = await solveChannel(c.o, ctx), q = post(c, r), W = [], { fl, L, H, nx, ny } = c, o = c.o, one = channel1D(v);
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
    if (!turbulent && q.Re > 2500) W.push({ level: 'warn', msg: `Reynolds number ${fmt(q.Re, 3)} is above the laminar range; the laminar solution under-predicts friction and mixing. Switch on the mixing-length RANS closure.` });
    if (turbulent) W.push({ level: 'info', msg: `Algebraic mixing-length closure: friction velocity ${fmt(r.utau, 3)} m/s, first-cell y⁺ ≈ ${fmt((0.5 * dy0 * r.utau * fl.rho) / fl.mu, 3)}. The log-law wall function is used where y⁺ > 11.6. Separated regions are only approximately represented by this closure.` });
    if (o.steady && q.recirc > 0.02 && q.uc.some((u, P) => P % nx === nx - 1 && u < -0.02 * r.Uref)) W.push({ level: 'warn', msg: 'Reverse flow reaches the outlet plane — lengthen the domain so that the recirculation closes inside it.' });
    if (sp && kDev) { const dc = fl.D / kDev; if (0.5 * dy0 > 0.35 * dc) W.push({ level: 'warn', msg: `The wall cell (${fmt(dy0 * 1e6, 3)} µm) is coarse relative to the concentration boundary layer (≈ ${fmt(dc * 1e6, 3)} µm): polarisation and Sherwood number are under-resolved. Increase the wall refinement or ny.` }); }
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
    if (th) { let hin = 0, hout = 0, qw = 0; for (let j = 0; j < ny; j++) { hin += r.uin[j] * v.T * r.dy[j]; hout += r.u[j * r.nu1 + nx] * r.eng.phi[j * nx + nx - 1] * r.dy[j]; } for (let i = 0; i < nx; i++) qw += (th.B.flux[i] + th.T.flux[i]) * r.dx; balances.push({ name: 'Heat (kW per metre width, relative to 0 °C)', in: ((hin + qw) * fl.rho * fl.cp) / 1000, out: (hout * fl.rho * fl.cp) / 1000 }); }
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
