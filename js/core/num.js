// Numerical toolbox shared by every suite: root finding, linear algebra, ODE integration,
// optimisation, statistics and random sampling. Pure functions, no DOM.

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)));
export const logspace = (a, b, n) => linspace(Math.log10(a), Math.log10(b), n).map((e) => 10 ** e);
export const sum = (a) => a.reduce((s, x) => s + x, 0);
export const mean = (a) => (a.length ? sum(a) / a.length : NaN);
export const variance = (a) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return sum(a.map((x) => (x - m) ** 2)) / (a.length - 1);
};
export const std = (a) => Math.sqrt(variance(a));
export const quantile = (a, q) => {
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const p = (s.length - 1) * q, i = Math.floor(p), f = p - i;
  return i + 1 < s.length ? s[i] * (1 - f) + s[i + 1] * f : s[i];
};
export const isNum = (x) => typeof x === 'number' && Number.isFinite(x);

/** Linear interpolation on a monotone-increasing table (clamped at the ends). */
export function interp1(xs, ys, x) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

/** Brent's method. Returns the root of f on a bracketing interval [a,b]. */
export function brent(f, a, b, tol = 1e-10, maxIter = 200) {
  let fa = f(a), fb = f(b);
  if (fa === 0) return a;
  if (fb === 0) return b;
  if (fa * fb > 0) throw new Error('brent: root is not bracketed');
  let c = a, fc = fa, d = b - a, e = d;
  for (let i = 0; i < maxIter; i++) {
    if (fb * fc > 0) { c = a; fc = fa; d = e = b - a; }
    if (Math.abs(fc) < Math.abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
    const tol1 = 2 * Number.EPSILON * Math.abs(b) + 0.5 * tol, xm = 0.5 * (c - b);
    if (Math.abs(xm) <= tol1 || fb === 0) return b;
    if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
      const s = fb / fa;
      let p, q;
      if (a === c) { p = 2 * xm * s; q = 1 - s; }
      else {
        const qq = fa / fc, r = fb / fc;
        p = s * (2 * xm * qq * (qq - r) - (b - a) * (r - 1));
        q = (qq - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q;
      p = Math.abs(p);
      if (2 * p < Math.min(3 * xm * q - Math.abs(tol1 * q), Math.abs(e * q))) { e = d; d = p / q; }
      else { d = xm; e = d; }
    } else { d = xm; e = d; }
    a = b; fa = fb;
    b += Math.abs(d) > tol1 ? d : (xm > 0 ? tol1 : -tol1);
    fb = f(b);
  }
  return b;
}

/** Bisection that tolerates a non-bracketed start by returning the better end. */
export function solve1(f, lo, hi, tol = 1e-9) {
  const flo = f(lo), fhi = f(hi);
  if (!(flo * fhi <= 0)) return Math.abs(flo) < Math.abs(fhi) ? lo : hi;
  return brent(f, lo, hi, tol);
}

/** Scalar Newton with numerical derivative and step damping. */
export function newton1(f, x0, { tol = 1e-10, maxIter = 60, h = 1e-6 } = {}) {
  let x = x0;
  for (let i = 0; i < maxIter; i++) {
    const fx = f(x);
    if (Math.abs(fx) < tol) return x;
    const dx = h * Math.max(1, Math.abs(x));
    const d = (f(x + dx) - fx) / dx;
    if (!Number.isFinite(d) || d === 0) break;
    let step = fx / d, lam = 1;
    while (lam > 1e-4 && !(Math.abs(f(x - lam * step)) < Math.abs(fx))) lam *= 0.5;
    x -= lam * step;
  }
  return x;
}

/** Solve A x = b by Gaussian elimination with partial pivoting (A: array of rows). */
export function solveLinear(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (Math.abs(M[p][k]) < 1e-300) throw new Error('solveLinear: singular matrix');
    [M[k], M[p]] = [M[p], M[k]];
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      if (f === 0) continue;
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/** Thomas algorithm: a = sub-diagonal, b = diagonal, c = super-diagonal, d = right-hand side. */
export function tridiag(a, b, c, d) {
  const n = d.length, cp = new Array(n), dp = new Array(n), x = new Array(n);
  cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0];
  for (let i = 1; i < n; i++) {
    const m = b[i] - a[i] * cp[i - 1];
    cp[i] = c[i] / m;
    dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
  }
  x[n - 1] = dp[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
  return x;
}

/** Multi-dimensional damped Newton with finite-difference Jacobian. F: R^n -> R^n. */
export function newtonN(F, x0, { tol = 1e-9, maxIter = 80, h = 1e-7 } = {}) {
  let x = [...x0], fx = F(x), n = x.length;
  const norm = (v) => Math.sqrt(sum(v.map((q) => q * q)));
  let converged = false, it = 0;
  for (; it < maxIter; it++) {
    if (norm(fx) < tol) { converged = true; break; }
    const J = Array.from({ length: n }, () => new Array(n));
    for (let j = 0; j < n; j++) {
      const xj = [...x], dx = h * Math.max(1, Math.abs(x[j]));
      xj[j] += dx;
      const fj = F(xj);
      for (let i = 0; i < n; i++) J[i][j] = (fj[i] - fx[i]) / dx;
    }
    let step;
    try { step = solveLinear(J, fx.map((v) => -v)); } catch { break; }
    let lam = 1, xn, fn;
    for (;;) {
      xn = x.map((v, i) => v + lam * step[i]);
      fn = F(xn);
      if (fn.every(Number.isFinite) && norm(fn) < norm(fx)) break;
      lam *= 0.5;
      if (lam < 1e-6) break;
    }
    x = xn; fx = fn;
  }
  return { x, residual: norm(fx), converged: converged || norm(fx) < tol, iterations: it };
}

/** Classical RK4 integrator. f(t, y) -> dy/dt (arrays). Returns { t:[], y:[[]] }. */
export function rk4(f, y0, t0, t1, n) {
  const h = (t1 - t0) / n, ts = [t0], ys = [[...y0]];
  let y = [...y0], t = t0;
  const ax = (a, b, s) => a.map((v, i) => v + s * b[i]);
  for (let i = 0; i < n; i++) {
    const k1 = f(t, y), k2 = f(t + h / 2, ax(y, k1, h / 2)), k3 = f(t + h / 2, ax(y, k2, h / 2)), k4 = f(t + h, ax(y, k3, h));
    y = y.map((v, j) => v + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    t += h;
    ts.push(t); ys.push([...y]);
  }
  return { t: ts, y: ys };
}

/** Adaptive Dormand–Prince RK45. Returns { t:[], y:[[]] }. Optional stop(t,y) terminates early. */
export function rk45(f, y0, t0, t1, { rtol = 1e-6, atol = 1e-9, hInit, maxSteps = 20000, stop } = {}) {
  const c = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
  const a = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
  const b5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
  const b4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40];
  let t = t0, y = [...y0], h = hInit ?? (t1 - t0) / 100;
  const ts = [t], ys = [[...y]], n = y.length;
  for (let s = 0; s < maxSteps && t < t1; s++) {
    if (t + h > t1) h = t1 - t;
    const k = [];
    for (let i = 0; i < 7; i++) {
      const yi = y.map((v, j) => { let acc = v; for (let m = 0; m < i; m++) acc += h * a[i][m] * k[m][j]; return acc; });
      k.push(f(t + c[i] * h, yi));
    }
    const y5 = y.map((v, j) => v + h * sum(b5.map((bb, i) => bb * k[i][j])));
    const y4 = y.map((v, j) => v + h * sum(b4.map((bb, i) => bb * k[i][j])));
    let err = 0;
    for (let j = 0; j < n; j++) err = Math.max(err, Math.abs(y5[j] - y4[j]) / (atol + rtol * Math.max(Math.abs(y[j]), Math.abs(y5[j]))));
    if (err <= 1 || h < 1e-14 * Math.max(1, Math.abs(t))) {
      t += h; y = y5; ts.push(t); ys.push([...y]);
      if (stop && stop(t, y)) break;
    }
    h *= clamp(0.9 * Math.pow(Math.max(err, 1e-10), -0.2), 0.2, 5);
  }
  return { t: ts, y: ys };
}

/** Nelder–Mead simplex minimiser with box bounds (projection). */
export function nelderMead(f, x0, { lo, hi, tol = 1e-8, maxIter = 2000, scale = 0.1 } = {}) {
  const n = x0.length;
  const proj = (x) => x.map((v, i) => clamp(v, lo ? lo[i] : -Infinity, hi ? hi[i] : Infinity));
  const fe = (x) => { const v = f(x); return Number.isFinite(v) ? v : 1e300; };
  let S = [proj(x0)];
  for (let i = 0; i < n; i++) {
    const p = [...x0];
    const span = lo && hi ? hi[i] - lo[i] : Math.max(1, Math.abs(p[i]));
    p[i] += scale * span * (hi && p[i] + scale * span > hi[i] ? -1 : 1);
    S.push(proj(p));
  }
  let F = S.map(fe), evals = n + 1, it = 0;
  for (; it < maxIter; it++) {
    const idx = F.map((_, i) => i).sort((p, q) => F[p] - F[q]);
    S = idx.map((i) => S[i]); F = idx.map((i) => F[i]);
    if (Math.abs(F[n] - F[0]) <= tol * (Math.abs(F[0]) + tol)) break;
    const cen = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cen[j] += S[i][j] / n;
    const at = (s) => proj(cen.map((c0, j) => c0 + s * (S[n][j] - c0)));
    const xr = at(-1), fr = fe(xr); evals++;
    if (fr < F[0]) {
      const xe = at(-2), fx = fe(xe); evals++;
      if (fx < fr) { S[n] = xe; F[n] = fx; } else { S[n] = xr; F[n] = fr; }
    } else if (fr < F[n - 1]) { S[n] = xr; F[n] = fr; }
    else {
      const xc = at(fr < F[n] ? -0.5 : 0.5), fc = fe(xc); evals++;
      if (fc < Math.min(fr, F[n])) { S[n] = xc; F[n] = fc; }
      else for (let i = 1; i <= n; i++) { S[i] = proj(S[i].map((v, j) => S[0][j] + 0.5 * (v - S[0][j]))); F[i] = fe(S[i]); evals++; }
    }
  }
  const best = F.indexOf(Math.min(...F));
  return { x: S[best], f: F[best], iterations: it, evals };
}

/** Seedable RNG (mulberry32) with normal / lognormal / triangular helpers. */
export function rng(seed = 12345) {
  let s = seed >>> 0;
  const u = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = (mu = 0, sd = 1) => mu + sd * Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
  return {
    uniform: (a = 0, b = 1) => a + (b - a) * u(),
    normal,
    lognormal: (mu, sd) => Math.exp(normal(mu, sd)),
    triangular: (a, c, b) => { const r = u(), fc = (c - a) / (b - a); return r < fc ? a + Math.sqrt(r * (b - a) * (c - a)) : b - Math.sqrt((1 - r) * (b - a) * (b - c)); },
    int: (n) => Math.floor(u() * n),
  };
}

/** Differential evolution: robust global minimiser inside box bounds. */
export function diffEvolution(f, lo, hi, { pop = 30, gens = 120, F = 0.7, CR = 0.9, seed = 1, onGen } = {}) {
  const r = rng(seed), n = lo.length;
  let P = Array.from({ length: pop }, () => lo.map((l, j) => r.uniform(l, hi[j])));
  let val = P.map(f), hist = [];
  for (let g = 0; g < gens; g++) {
    for (let i = 0; i < pop; i++) {
      let a, b, c;
      do a = r.int(pop); while (a === i);
      do b = r.int(pop); while (b === i || b === a);
      do c = r.int(pop); while (c === i || c === a || c === b);
      const jr = r.int(n);
      const trial = P[i].map((v, j) => (r.uniform() < CR || j === jr ? clamp(P[a][j] + F * (P[b][j] - P[c][j]), lo[j], hi[j]) : v));
      const ft = f(trial);
      if (ft <= val[i]) { P[i] = trial; val[i] = ft; }
    }
    const best = Math.min(...val);
    hist.push(best);
    if (onGen) onGen(g, best);
  }
  const bi = val.indexOf(Math.min(...val));
  return { x: P[bi], f: val[bi], history: hist };
}

/** Levenberg–Marquardt least squares. resid(p) -> residual array. Returns covariance and standard errors. */
export function levenbergMarquardt(resid, p0, { lo, hi, maxIter = 100, tol = 1e-10 } = {}) {
  let p = [...p0], r = resid(p), S = sum(r.map((v) => v * v)), lam = 1e-3;
  const n = p.length, m = r.length;
  const jac = (pp, rr) => {
    const J = Array.from({ length: m }, () => new Array(n));
    for (let j = 0; j < n; j++) {
      const q = [...pp], d = 1e-6 * Math.max(1e-8, Math.abs(q[j]) || 1);
      q[j] += d;
      const rj = resid(q);
      for (let i = 0; i < m; i++) J[i][j] = (rj[i] - rr[i]) / d;
    }
    return J;
  };
  let J = jac(p, r), it = 0;
  for (; it < maxIter; it++) {
    const A = Array.from({ length: n }, () => new Array(n).fill(0)), g = new Array(n).fill(0);
    for (let i = 0; i < m; i++) for (let a = 0; a < n; a++) {
      g[a] += J[i][a] * r[i];
      for (let b = 0; b < n; b++) A[a][b] += J[i][a] * J[i][b];
    }
    let improved = false;
    for (let tries = 0; tries < 12; tries++) {
      const Ad = A.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lam) + 1e-30 : v)));
      let dp;
      try { dp = solveLinear(Ad, g.map((v) => -v)); } catch { lam *= 10; continue; }
      const pn = p.map((v, j) => clamp(v + dp[j], lo ? lo[j] : -Infinity, hi ? hi[j] : Infinity));
      const rn = resid(pn), Sn = sum(rn.map((v) => v * v));
      if (Number.isFinite(Sn) && Sn < S) {
        const rel = (S - Sn) / Math.max(S, 1e-300);
        p = pn; r = rn; S = Sn; lam = Math.max(lam / 5, 1e-12); improved = true;
        J = jac(p, r);
        if (rel < tol) tries = 99;
        break;
      }
      lam *= 5;
    }
    if (!improved) break;
  }
  // parameter covariance = s^2 (J^T J)^-1
  let se = new Array(n).fill(NaN), cov = null;
  try {
    const A = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) A[a][b] += J[i][a] * J[i][b];
    const s2 = S / Math.max(1, m - n);
    cov = Array.from({ length: n }, (_, j) => solveLinear(A, Array.from({ length: n }, (_, i) => (i === j ? 1 : 0))).map((v) => v * s2));
    se = cov.map((row, i) => Math.sqrt(Math.abs(row[i])));
  } catch { /* unidentifiable parameters: leave standard errors undefined */ }
  return { p, sse: S, residuals: r, iterations: it, se, cov };
}

/** Ordinary least squares y = X b. Returns coefficients. */
export function lstsq(X, y) {
  const n = X[0].length, A = Array.from({ length: n }, () => new Array(n).fill(0)), g = new Array(n).fill(0);
  for (let i = 0; i < X.length; i++) for (let a = 0; a < n; a++) {
    g[a] += X[i][a] * y[i];
    for (let b = 0; b < n; b++) A[a][b] += X[i][a] * X[i][b];
  }
  for (let a = 0; a < n; a++) A[a][a] += 1e-12;
  return solveLinear(A, g);
}

/** Straight-line fit. Returns { slope, intercept, r2 }. */
export function linfit(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  const slope = sxx ? sxy / sxx : 0;
  return { slope, intercept: my - slope * mx, r2: sxx && syy ? (sxy * sxy) / (sxx * syy) : 0 };
}

/** Trapezoidal integral of tabulated y(x). */
export const trapz = (x, y) => { let s = 0; for (let i = 1; i < x.length; i++) s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]); return s; };

/** Agreement metrics between measured and predicted arrays. */
export function metrics(meas, pred) {
  const n = Math.min(meas.length, pred.length);
  const m = meas.slice(0, n), p = pred.slice(0, n), res = p.map((v, i) => v - m[i]);
  const bias = mean(res), mae = mean(res.map(Math.abs)), rmse = Math.sqrt(mean(res.map((v) => v * v)));
  const range = Math.max(...m) - Math.min(...m), mm = mean(m);
  const ssTot = sum(m.map((v) => (v - mm) ** 2)), ssRes = sum(res.map((v) => v * v));
  const nz = m.filter((v) => v !== 0);
  const mape = nz.length === n ? 100 * mean(res.map((v, i) => Math.abs(v / m[i]))) : NaN;
  const sd = std(res), half = n > 1 ? (1.96 * sd) / Math.sqrt(n) : NaN;
  return { n, bias, mae, rmse, nrmse: range ? rmse / range : NaN, mape, r2: ssTot ? 1 - ssRes / ssTot : NaN, ci95: [bias - half, bias + half], residuals: res };
}

/**
 * Grid-convergence index by Richardson extrapolation on three systematically refined grids.
 * h: representative cell sizes (fine -> coarse), f: solution values on those grids.
 */
export function gci(h, f, Fs = 1.25) {
  const [h1, h2, h3] = h, [f1, f2, f3] = f;
  const r21 = h2 / h1, r32 = h3 / h2, e21 = f2 - f1, e32 = f3 - f2;
  if (Math.abs(e21) < 1e-14 * Math.max(1, Math.abs(f1)) || Math.abs(e32) < 1e-300) {
    return { p: NaN, fExact: f1, gciFine: 0, gciCoarse: 0, asymptotic: 1, type: 'grid-insensitive', e21, e32 };
  }
  const s = Math.sign(e32 / e21) || 1;
  let p = Math.abs(Math.log(Math.abs(e32 / e21))) / Math.log(r21);
  for (let i = 0; i < 60; i++) {
    const q = Math.log((r21 ** p - s) / (r32 ** p - s));
    const pn = Math.abs(Math.log(Math.abs(e32 / e21)) + q) / Math.log(r21);
    if (!Number.isFinite(pn) || Math.abs(pn - p) < 1e-10) break;
    p = pn;
  }
  p = clamp(p, 0.3, 8);
  const fExact = f1 + (f1 - f2) / (r21 ** p - 1);
  const ea21 = Math.abs((f1 - f2) / (f1 || 1e-300)), ea32 = Math.abs((f2 - f3) / (f2 || 1e-300));
  const gciFine = (Fs * ea21) / (r21 ** p - 1), gciCoarse = (Fs * ea32) / (r32 ** p - 1);
  return { p, fExact, gciFine, gciCoarse, asymptotic: gciCoarse / (r21 ** p * gciFine), type: e32 / e21 > 0 ? 'monotonic' : 'oscillatory', e21, e32 };
}

/** Latin-hypercube sample of n points in the unit cube of dimension d. */
export function lhs(n, d, seed = 7) {
  const r = rng(seed), cols = [];
  for (let j = 0; j < d; j++) {
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const k = r.int(i + 1); [perm[i], perm[k]] = [perm[k], perm[i]]; }
    cols.push(perm.map((v) => (v + r.uniform()) / n));
  }
  return Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
}

/** Histogram with nb bins -> { centers, counts }. */
export function histogram(a, nb = 30) {
  let lo = Infinity, hi = -Infinity;
  for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const w = (hi - lo) / nb || 1;
  const counts = new Array(nb).fill(0);
  for (const v of a) counts[Math.min(nb - 1, Math.floor((v - lo) / w))]++;
  return { centers: counts.map((_, i) => lo + (i + 0.5) * w), counts, width: w };
}

/** Number formatting used across tables and KPI tiles. */
export function fmt(x, sig = 4) {
  if (x === null || x === undefined || x === '') return '–';
  if (typeof x !== 'number') return String(x);
  if (!Number.isFinite(x)) return Number.isNaN(x) ? '–' : (x > 0 ? '∞' : '−∞');
  if (x === 0) return '0';
  const ax = Math.abs(x);
  if (ax >= 1e7 || ax < 1e-4) return x.toExponential(Math.max(1, sig - 1)).replace('e+', 'e');
  const digits = Math.max(0, sig - 1 - Math.floor(Math.log10(ax)));
  return x.toLocaleString('en-US', { maximumFractionDigits: Math.min(8, digits), minimumFractionDigits: 0 });
}
