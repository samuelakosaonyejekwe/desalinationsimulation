// Suite 11 — Optimisation, machine learning and custom numerical modelling.
// Ten self-contained studies around the RO plant model of suite 1 and around user data:
// constrained and multi-objective optimisation, sensitivity and uncertainty analysis, surrogate
// models (response surface, Gaussian process, neural network), a physics-informed network,
// forecasting and state estimation, predictive control and reinforcement learning, a custom
// equation workbench driven by a safe expression evaluator, and parameter estimation with
// identifiability and Bayesian inference. Every algorithm is implemented here or in core/num.js.
import { clamp, linspace, sum, mean, std, variance, quantile, solveLinear, tridiag, rk4, rk45, newtonN, nelderMead, diffEvolution,
  levenbergMarquardt, lstsq, metrics, gci, lhs, rng, histogram, fmt, isNum } from '../core/num.js';
import { WATERS, cloneIons, tds, scaleIons, osmoticPressureIons } from '../core/water.js';
import roSuite, { simulateRO, autoSize, MEMBRANES } from './s01_ro.js';
import { analyzeWater, makeSolution, saturation } from './s02_chem.js';

// ======================================================================================================
// 1 · Small linear-algebra and statistics helpers
// ======================================================================================================
const HAS = Object.prototype.hasOwnProperty;
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const zeros = (n, m) => (m === undefined ? new Array(n).fill(0) : Array.from({ length: n }, () => new Array(m).fill(0)));
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const matVec = (A, x) => A.map((r) => dot(r, x));
const matMul = (A, B) => A.map((r) => B[0].map((_, j) => { let s = 0; for (let k = 0; k < B.length; k++) s += r[k] * B[k][j]; return s; }));
const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const maxAbs = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const invert = (A) => transpose(A.map((_, j) => solveLinear(A, A.map((_, i) => (i === j ? 1 : 0)))));
const finite = (x, d = 0) => (Number.isFinite(x) ? x : d);
const range = (n) => Array.from({ length: n }, (_, i) => i);
const lcFirst = (s) => String(s).replace(/^[A-Z](?=[a-z])/, (c) => c.toLowerCase()); // lower-case a label inside a sentence, keeping acronyms
const rmse = (a, b) => Math.sqrt(mean(a.map((v, i) => (v - b[i]) ** 2)));
const numRows = (rows, keys) => (Array.isArray(rows) ? rows : []).filter((r) => r && keys.every((k) => isNum(r[k])));
function shuffled(n, g) { const p = range(n); for (let i = n - 1; i > 0; i--) { const k = g.int(i + 1); [p[i], p[k]] = [p[k], p[i]]; } return p; }

/** Error function (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
export function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
const normPdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
/** Inverse standard-normal CDF (Acklam's rational approximation, relative error < 1.2e-9). */
export function normInv(p) {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const q0 = clamp(p, 1e-12, 1 - 1e-12), tail = (q) => (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  if (q0 < 0.02425) return tail(Math.sqrt(-2 * Math.log(q0)));
  if (q0 > 1 - 0.02425) return -tail(Math.sqrt(-2 * Math.log(1 - q0)));
  const q = q0 - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Cholesky factor L (A = L·Lᵀ) of a symmetric positive-definite matrix, or null when it is not. */
export function cholesky(A) {
  const n = A.length, L = new Array(n);
  for (let i = 0; i < n; i++) {
    const Li = (L[i] = new Array(n).fill(0)), Ai = A[i];
    for (let j = 0; j <= i; j++) {
      const Lj = L[j];
      let s = Ai[j];
      for (let k = 0; k < j; k++) s -= Li[k] * Lj[k];
      if (i === j) { if (!(s > 0)) return null; Li[i] = Math.sqrt(s); } else Li[j] = s / Lj[j];
    }
  }
  return L;
}
function fwdSub(L, b) { const n = b.length, y = new Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; } return y; }
function backSubT(L, y) { const n = y.length, x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; }
const cholSolve = (L, b) => backSubT(L, fwdSub(L, b));
function standardizer(X) {
  const d = X[0].length, mu = [], sd = [];
  for (let j = 0; j < d; j++) { const c = X.map((r) => r[j]); mu.push(mean(c)); sd.push(std(c) || 1); }
  return { mu, sd, f: (x) => x.map((v, j) => (v - mu[j]) / sd[j]) };
}

// ======================================================================================================
// 2 · Safe expression evaluator (no code execution: tokeniser → recursive descent → closures)
// ======================================================================================================
const FUNCS = new Map([
  ['sin', [Math.sin, 1, 1]], ['cos', [Math.cos, 1, 1]], ['tan', [Math.tan, 1, 1]], ['exp', [Math.exp, 1, 1]], ['ln', [Math.log, 1, 1]],
  ['log10', [Math.log10, 1, 1]], ['sqrt', [Math.sqrt, 1, 1]], ['abs', [Math.abs, 1, 1]], ['tanh', [Math.tanh, 1, 1]], ['erf', [erf, 1, 1]],
  ['step', [(x) => (x >= 0 ? 1 : 0), 1, 1]], ['min', [Math.min, 1, 8]], ['max', [Math.max, 1, 8]], ['pow', [Math.pow, 2, 2]],
]);
const CONSTS = new Map([['pi', Math.PI], ['e', Math.E]]);
const CMP = new Map([['<', (a, b) => a < b], ['>', (a, b) => a > b], ['<=', (a, b) => a <= b], ['>=', (a, b) => a >= b], ['==', (a, b) => a === b], ['!=', (a, b) => a !== b]]);
const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

function tokenize(src) {
  const s = String(src ?? '').replace(/[−–]/g, '-').replace(/×/g, '*').replace(/÷/g, '/');
  if (s.length > 400) throw new Error('Expression is too long (400 characters maximum).');
  const out = [];
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') { i++; continue; }
    if ((ch >= '0' && ch <= '9') || (ch === '.' && s[i + 1] >= '0' && s[i + 1] <= '9')) {
      const m = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(s.slice(i));
      out.push({ t: 'num', v: parseFloat(m[0]) }); i += m[0].length; continue;
    }
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_') {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(s.slice(i));
      out.push({ t: 'id', v: m[0] }); i += m[0].length; continue;
    }
    const two = s.slice(i, i + 2);
    if (CMP.has(two)) { out.push({ t: 'op', v: two }); i += 2; continue; }
    if ('+-*/^(),<>'.includes(ch)) { out.push({ t: 'op', v: ch }); i++; continue; }
    throw new Error(`Character “${ch}” is not allowed in an expression.`);
  }
  return out;
}

/**
 * Parse an arithmetic expression into an evaluator. Grammar: comparison > sum > product > unary > power > primary.
 * Only numbers, the operators + − * / ^, comparisons, parentheses, named variables and the whitelisted
 * functions exist — there is no property access, indexing, string or assignment syntax at all.
 * `allowed` (optional) lists the variable names that may appear. Returns { text, vars, evaluate(scope) }.
 */
export function parseExpr(text, allowed) {
  const tk = tokenize(text), vars = new Set(), allow = allowed ? new Set(allowed) : null;
  let p = 0, depth = 0;
  const isOp = (v) => p < tk.length && tk[p].t === 'op' && tk[p].v === v;
  const fail = (msg) => { throw new Error(`${msg} (in “${String(text).slice(0, 60)}”)`); };
  function expr() {
    if (++depth > 60) fail('Expression is nested too deeply');
    let a = add();
    if (p < tk.length && tk[p].t === 'op' && CMP.has(tk[p].v)) { const f = CMP.get(tk[p++].v), l = a, r = add(); a = (s) => (f(l(s), r(s)) ? 1 : 0); }
    depth--;
    return a;
  }
  function add() {
    let a = mul();
    while (isOp('+') || isOp('-')) { const op = tk[p++].v, l = a, r = mul(); a = op === '+' ? (s) => l(s) + r(s) : (s) => l(s) - r(s); }
    return a;
  }
  function mul() {
    let a = unary();
    while (isOp('*') || isOp('/')) { const op = tk[p++].v, l = a, r = unary(); a = op === '*' ? (s) => l(s) * r(s) : (s) => l(s) / r(s); }
    return a;
  }
  function unary() {
    let neg = false;
    while (isOp('-') || isOp('+')) { if (tk[p++].v === '-') neg = !neg; }
    const u = power();
    return neg ? (s) => -u(s) : u;
  }
  function power() {
    const b = primary();
    if (!isOp('^')) return b;
    p++;
    const e = unary();
    return (s) => Math.pow(b(s), e(s));
  }
  function primary() {
    if (p >= tk.length) fail('Expression ends unexpectedly');
    const t = tk[p++];
    if (t.t === 'num') { const c = t.v; return () => c; }
    if (t.t === 'id') {
      const name = t.v;
      if (isOp('(')) {
        const F = FUNCS.get(name);
        if (!F) fail(`Function “${name}” is not available; allowed functions are ${[...FUNCS.keys()].join(', ')}`);
        p++;
        const args = [];
        if (!isOp(')')) { args.push(expr()); while (isOp(',')) { p++; args.push(expr()); } }
        if (!isOp(')')) fail('Missing “)”');
        p++;
        if (args.length < F[1] || args.length > F[2]) fail(`Function “${name}” takes ${F[1] === F[2] ? F[1] : `${F[1]}–${F[2]}`} argument(s)`);
        const fn = F[0], [a0, a1] = args;
        if (args.length === 1) return (s) => fn(a0(s));
        if (args.length === 2) return (s) => fn(a0(s), a1(s));
        return (s) => fn(...args.map((a) => a(s)));
      }
      if (FUNCS.has(name)) fail(`“${name}” is a function and needs parentheses`);
      if (allow && !allow.has(name) && !CONSTS.has(name)) fail(`Unknown variable “${name}”`);
      vars.add(name);
      return (s) => {
        if (HAS.call(s, name)) return +s[name];
        if (CONSTS.has(name)) return CONSTS.get(name);
        throw new Error(`Unknown variable “${name}”.`);
      };
    }
    if (t.v === '(') { const e = expr(); if (!isOp(')')) fail('Missing “)”'); p++; return e; }
    return fail(`Unexpected “${t.v}”`);
  }
  const fn = expr();
  if (p < tk.length) fail(`Unexpected “${tk[p].v}”`);
  return { text: String(text), vars: [...vars], evaluate: fn };
}
/** One-off evaluation of an expression with a scope object of numeric variables. */
export const evalExpr = (text, scope = {}) => parseExpr(text).evaluate(scope);

/** Read a parameter table [{ name, value }] into a clean scope; names must be plain identifiers. */
function paramScope(rows) {
  const sc = Object.create(null);
  for (const r of Array.isArray(rows) ? rows : []) {
    const name = String(r?.name ?? '').trim();
    if (!name) continue;
    if (!IDENT.test(name) || FUNCS.has(name)) throw new Error(`Parameter name “${name.slice(0, 30)}” is not valid: use letters, digits and underscores, and avoid function names.`);
    sc[name] = finite(+r.value);
  }
  return sc;
}
const blank = (s) => s === null || s === undefined || String(s).trim() === '';

// ======================================================================================================
// 3 · Optimisation: finite-difference gradients, dual QP, SQP, KKT analysis, NSGA-II
// ======================================================================================================
/** Central finite-difference gradient of a scalar function. */
export function fdGradient(f, x, h = 1e-6) {
  return x.map((_, j) => { const d = h * Math.max(1, Math.abs(x[j])), a = [...x], b = [...x]; a[j] += d; b[j] -= d; return (f(a) - f(b)) / (2 * d); });
}

/** Pre-factorise the dual of  min ½xᵀHx + cᵀx  s.t.  A x ≤ b  (Hildreth's method). */
export function qpPrepare(H, A) {
  const Hinv = invert(H), AH = A.map((r) => matVec(Hinv, r));
  return { Hinv, A, AH, P: AH.map((ah) => A.map((aj) => dot(ah, aj))) };
}
/** Solve the convex QP for given c, b by Gauss–Seidel ascent on the dual. Returns { x, lambda, iterations }. */
export function qpSolve(prep, c, b, { maxIter = 300, tol = 1e-10, lam0 } = {}) {
  const { Hinv, A, AH, P } = prep, m = A.length, x = matVec(Hinv, c).map((v) => -v);
  if (!m) return { x, lambda: [], iterations: 0 };
  const d = b.map((bi, i) => bi - dot(A[i], x)); // slack of every constraint at the unconstrained minimiser
  if (d.every((v) => v >= 0)) return { x, lambda: zeros(m), iterations: 0 };
  const lam = lam0 && lam0.length === m ? [...lam0] : zeros(m);
  let it = 0;
  for (; it < maxIter; it++) {
    let change = 0, scale = 0;
    for (let i = 0; i < m; i++) {
      if (P[i][i] < 1e-14) continue;
      let w = d[i];
      for (let j = 0; j < m; j++) if (j !== i && lam[j] !== 0) w += P[i][j] * lam[j];
      const li = Math.min(1e12, Math.max(0, -w / P[i][i]));
      change = Math.max(change, Math.abs(li - lam[i])); scale = Math.max(scale, li); lam[i] = li;
    }
    if (change <= tol * (1 + scale)) break;
  }
  for (let i = 0; i < m; i++) if (lam[i] > 0) for (let k = 0; k < x.length; k++) x[k] -= AH[i][k] * lam[i];
  return { x, lambda: lam, iterations: it };
}

/**
 * SQP-lite: damped-BFGS sequential quadratic programming with forward-difference gradients, an L1 merit
 * line search and bounds handled as linear constraints of the QP. cons(x) returns g(x) ≤ 0.
 * Returns { x, f, g, lambda, lamLo, lamHi, kkt, iterations, evals, converged, history }.
 */
export function sqpLite(f, cons, x0, { lo, hi, maxIter = 40, tol = 1e-6, h = 1e-7 } = {}) {
  const n = x0.length, L = lo || new Array(n).fill(-Infinity), U = hi || new Array(n).fill(Infinity);
  let evals = 0;
  const F = (x) => { evals++; const v = f(x); return Number.isFinite(v) ? v : 1e30; };
  const G = (x) => (cons ? cons(x).map((v) => (Number.isFinite(v) ? v : 1e3)) : []);
  const proj = (x) => x.map((v, j) => clamp(v, L[j], U[j]));
  const viol = (g) => sum(g.map((v) => Math.max(0, v)));
  const grads = (x, fx, gx) => {
    const gf = zeros(n), J = gx.map(() => zeros(n));
    for (let j = 0; j < n; j++) {
      const span = U[j] - L[j];
      let dx = h * (Number.isFinite(span) ? span : Math.max(1, Math.abs(x[j])));
      if (x[j] + dx > U[j]) dx = -dx;
      const xp = [...x]; xp[j] += dx;
      const fp = F(xp), gp = G(xp);
      gf[j] = (fp - fx) / dx;
      for (let i = 0; i < gx.length; i++) J[i][j] = (gp[i] - gx[i]) / dx;
    }
    return { gf, J };
  };
  let x = proj(x0), fx = F(x), gx = G(x), B = eye(n), mu = 1, it = 0, converged = false, kkt = Infinity;
  let { gf, J } = grads(x, fx, gx), lam = zeros(gx.length), lamLo = zeros(n), lamHi = zeros(n);
  const m = gx.length, history = [];
  const subproblem = () => {
    const A = [], b = [], tag = [];
    for (let i = 0; i < m; i++) { A.push(J[i]); b.push(-gx[i]); tag.push(['g', i]); }
    for (let j = 0; j < n; j++) {
      if (Number.isFinite(U[j])) { A.push(range(n).map((k) => (k === j ? 1 : 0))); b.push(U[j] - x[j]); tag.push(['hi', j]); }
      if (Number.isFinite(L[j])) { A.push(range(n).map((k) => (k === j ? -1 : 0))); b.push(x[j] - L[j]); tag.push(['lo', j]); }
    }
    const qp = qpSolve(qpPrepare(B, A), gf, b, { maxIter: 500 });
    lam = zeros(m); lamLo = zeros(n); lamHi = zeros(n);
    qp.lambda.forEach((l, r) => { const [t, i] = tag[r]; if (t === 'g') lam[i] = l; else if (t === 'hi') lamHi[i] = l; else lamLo[i] = l; });
    const st = gf.map((v, k) => v + sum(qp.lambda.map((l, r) => l * A[r][k])));
    kkt = Math.max(maxAbs(st), ...gx.map((v) => Math.max(0, v)), 0);
    return qp.x;
  };
  for (; it < maxIter; it++) {
    const d = subproblem();
    history.push({ it, f: fx, viol: viol(gx), kkt, evals });
    if (maxAbs(d) < tol * (1 + maxAbs(x))) { converged = true; break; }
    mu = Math.max(mu, 1.5 * Math.max(0, ...lam) + 1e-3);
    const phi0 = fx + mu * viol(gx), D = Math.min(0, dot(gf, d) - mu * viol(gx));
    let a = 1, xn, fn, gn, ok = false;
    for (let ls = 0; ls < 16; ls++) {
      xn = proj(x.map((v, k) => v + a * d[k])); fn = F(xn); gn = G(xn);
      if (fn + mu * viol(gn) <= phi0 + 1e-4 * a * D) { ok = true; break; }
      a *= 0.5;
    }
    if (!ok) break;
    const s = xn.map((v, k) => v - x[k]), gr = grads(xn, fn, gn);
    const y = range(n).map((k) => gr.gf[k] - gf[k] + sum(lam.map((l, i) => l * (gr.J[i][k] - J[i][k]))));
    const Bs = matVec(B, s), sBs = dot(s, Bs), sy = dot(s, y);
    if (sBs > 1e-18) { // Powell-damped BFGS keeps the Hessian approximation positive definite
      const th = sy >= 0.2 * sBs ? 1 : (0.8 * sBs) / (sBs - sy), r = y.map((v, k) => th * v + (1 - th) * Bs[k]), sr = dot(s, r);
      if (sr > 1e-18) B = B.map((row, i) => row.map((v, j) => v - (Bs[i] * Bs[j]) / sBs + (r[i] * r[j]) / sr));
    }
    x = xn; fx = fn; gx = gn; gf = gr.gf; J = gr.J;
  }
  if (!converged) { subproblem(); history.push({ it, f: fx, viol: viol(gx), kkt, evals }); }
  return { x, f: fx, g: gx, lambda: lam, lamLo, lamHi, kkt, iterations: it, evals, converged, history };
}

/**
 * First-order optimality check at a candidate point: central-difference gradients, active-set detection and
 * non-negative least-squares multipliers minimising ‖∇f + Σλᵢ∇gᵢ + bound terms‖.
 */
export function kktCheck(f, cons, x, { lo, hi, h = 1e-5, actTol = 1e-3 } = {}) {
  const n = x.length, g0 = cons ? cons(x) : [], m = g0.length, gf = zeros(n), J = g0.map(() => zeros(n));
  for (let j = 0; j < n; j++) {
    const span = lo && hi && Number.isFinite(hi[j] - lo[j]) ? hi[j] - lo[j] : Math.max(1, Math.abs(x[j])), d = h * span;
    const up = !hi || x[j] + d <= hi[j], dn = !lo || x[j] - d >= lo[j];
    const a = [...x], b = [...x];
    if (up) a[j] += d;
    if (dn) b[j] -= d;
    const w = a[j] - b[j] || 1, ga = cons ? cons(a) : [], gb = cons ? cons(b) : [];
    gf[j] = (f(a) - f(b)) / w;
    for (let i = 0; i < m; i++) J[i][j] = (ga[i] - gb[i]) / w;
  }
  const cols = [];
  for (let i = 0; i < m; i++) if (g0[i] >= -actTol) cols.push({ t: 'g', i, a: J[i] });
  for (let j = 0; j < n; j++) {
    const span = lo && hi ? hi[j] - lo[j] : 1;
    if (hi && x[j] >= hi[j] - 1e-6 * span) cols.push({ t: 'hi', i: j, a: range(n).map((k) => (k === j ? 1 : 0)) });
    if (lo && x[j] <= lo[j] + 1e-6 * span) cols.push({ t: 'lo', i: j, a: range(n).map((k) => (k === j ? -1 : 0)) });
  }
  const l = zeros(cols.length), r = [...gf];
  for (let sweep = 0; sweep < 400; sweep++) {
    let ch = 0;
    cols.forEach((c, k) => {
      const aa = dot(c.a, c.a);
      if (aa < 1e-30) return;
      const ln = Math.max(0, l[k] - dot(c.a, r) / aa), dl = ln - l[k];
      if (dl !== 0) { for (let q = 0; q < n; q++) r[q] += dl * c.a[q]; l[k] = ln; ch = Math.max(ch, Math.abs(dl)); }
    });
    if (ch < 1e-13) break;
  }
  const lambda = zeros(m), lamLo = zeros(n), lamHi = zeros(n);
  cols.forEach((c, k) => { if (c.t === 'g') lambda[c.i] = l[k]; else if (c.t === 'hi') lamHi[c.i] = l[k]; else lamLo[c.i] = l[k]; });
  return { gf, J, g: g0, lambda, lamLo, lamHi, residual: maxAbs(r), relResidual: maxAbs(r) / Math.max(maxAbs(gf), 1e-12), active: cols.filter((c) => c.t === 'g').map((c) => c.i) };
}

// ---- NSGA-II ---------------------------------------------------------------------------------------------
const dominates = (p, q) => {
  if (p.cv > 0 || q.cv > 0) return p.cv < q.cv; // constrained domination (Deb): feasibility first
  let better = false;
  for (let i = 0; i < p.f.length; i++) { if (p.f[i] > q.f[i]) return false; if (p.f[i] < q.f[i]) better = true; }
  return better;
};
/** Fast non-dominated sort: assigns .rank and returns the fronts as arrays of individuals. */
export function nonDominatedSort(P) {
  const S = P.map(() => []), nDom = new Array(P.length).fill(0), fronts = [[]];
  for (let i = 0; i < P.length; i++) {
    for (let j = 0; j < P.length; j++) {
      if (i === j) continue;
      if (dominates(P[i], P[j])) S[i].push(j); else if (dominates(P[j], P[i])) nDom[i]++;
    }
    if (nDom[i] === 0) { P[i].rank = 0; fronts[0].push(i); }
  }
  for (let k = 0; fronts[k].length; k++) {
    const next = [];
    for (const i of fronts[k]) for (const j of S[i]) if (--nDom[j] === 0) { P[j].rank = k + 1; next.push(j); }
    fronts.push(next);
  }
  return fronts.filter((f) => f.length).map((f) => f.map((i) => P[i]));
}
/** Crowding distance of one front (assigns .crowd). */
export function crowdingDistance(front) {
  for (const p of front) p.crowd = 0;
  const M = front[0].f.length;
  for (let k = 0; k < M; k++) {
    const s = [...front].sort((a, b) => a.f[k] - b.f[k]), span = s[s.length - 1].f[k] - s[0].f[k];
    s[0].crowd = s[s.length - 1].crowd = Infinity;
    if (span > 0) for (let i = 1; i < s.length - 1; i++) s[i].crowd += (s[i + 1].f[k] - s[i - 1].f[k]) / span;
  }
}
/** Create the initial NSGA-II state. evalFn(x) → { f: [objectives to minimise], cv: total constraint violation ≥ 0, … }. */
export function nsga2Init(evalFn, lo, hi, { pop = 40, seed = 1, etaC = 15, etaM = 20, pc = 0.9, pm } = {}) {
  const n = lo.length, N = Math.max(4, 2 * Math.round(pop / 2)), g = rng(seed);
  const make = (x) => { const e = evalFn(x); return { ...e, x, cv: Math.max(0, finite(e.cv, 1e9)), f: e.f.map((v) => finite(v, 1e30)) }; };
  const P = lhs(N, n, seed).map((u) => make(u.map((q, j) => lo[j] + q * (hi[j] - lo[j]))));
  nonDominatedSort(P).forEach(crowdingDistance);
  return { P, N, n, lo, hi, g, make, etaC, etaM, pc, pm: pm ?? 1 / n, gen: 0, evals: N };
}
/** Advance NSGA-II by one generation: tournament selection, SBX crossover, polynomial mutation, elitist truncation. */
export function nsga2Step(S) {
  const { P, N, n, lo, hi, g, etaC, etaM } = S;
  const pick = () => { const a = P[g.int(N)], b = P[g.int(N)]; return a.rank < b.rank || (a.rank === b.rank && a.crowd > b.crowd) ? a : b; };
  const kids = [];
  while (kids.length < N) {
    const p1 = pick().x, p2 = pick().x, c1 = [...p1], c2 = [...p2];
    if (g.uniform() < S.pc) for (let j = 0; j < n; j++) {
      if (g.uniform() > 0.5) continue;
      const u = g.uniform(), beta = u <= 0.5 ? (2 * u) ** (1 / (etaC + 1)) : (1 / (2 * (1 - u))) ** (1 / (etaC + 1));
      c1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j]); c2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j]);
    }
    for (const c of [c1, c2]) {
      for (let j = 0; j < n; j++) {
        if (g.uniform() < S.pm) { const u = g.uniform(), dl = u < 0.5 ? (2 * u) ** (1 / (etaM + 1)) - 1 : 1 - (2 * (1 - u)) ** (1 / (etaM + 1)); c[j] += dl * (hi[j] - lo[j]); }
        c[j] = clamp(c[j], lo[j], hi[j]);
      }
      kids.push(S.make(c));
    }
  }
  S.evals += kids.length;
  const R = [...P, ...kids.slice(0, N)], next = [];
  for (const front of nonDominatedSort(R)) {
    crowdingDistance(front);
    if (next.length + front.length <= N) next.push(...front);
    else { next.push(...front.sort((a, b) => (a.crowd === b.crowd ? 0 : b.crowd > a.crowd ? 1 : -1)).slice(0, N - next.length)); break; }
  }
  S.P = next; S.gen++;
  return S;
}
/** NSGA-II driver. Returns { pop, front (feasible rank-0 individuals), evals }. */
export function nsga2(evalFn, lo, hi, opts = {}) {
  const S = nsga2Init(evalFn, lo, hi, opts);
  for (let k = 0; k < (opts.gens ?? 50); k++) nsga2Step(S);
  return { pop: S.P, front: paretoFront(S.P), evals: S.evals };
}
const paretoFront = (P) => { const f = P.filter((p) => p.rank === 0 && p.cv === 0); return f.length ? f : P.filter((p) => p.rank === 0); };
/** Knee point: the front member closest to the utopia point after normalising every objective to 0–1. */
export function kneePoint(front) {
  const M = front[0].f.length, lo = range(M).map((k) => Math.min(...front.map((p) => p.f[k]))), hi = range(M).map((k) => Math.max(...front.map((p) => p.f[k])));
  let best = front[0], bd = Infinity;
  for (const p of front) { const d = Math.hypot(...p.f.map((v, k) => (hi[k] > lo[k] ? (v - lo[k]) / (hi[k] - lo[k]) : 0))); if (d < bd) { bd = d; best = p; } }
  return best;
}

// ======================================================================================================
// 4 · Global sensitivity analysis (Morris screening, Sobol indices by Saltelli sampling)
// ======================================================================================================
/** Morris one-at-a-time trajectories in the unit cube. Returns { pts, steps } — evaluate pts, then call morrisEstimate. */
export function morrisPlan(k, r, seed = 1, levels = 4) {
  const g = rng(seed), delta = levels / (2 * (levels - 1)), pts = [], steps = [];
  for (let t = 0; t < r; t++) {
    let x = range(k).map(() => g.int(levels) / (levels - 1)), prev = pts.length;
    pts.push(x);
    for (const i of shuffled(k, g)) {
      const d = x[i] + delta <= 1 + 1e-12 ? delta : -delta;
      x = [...x]; x[i] += d; pts.push(x);
      steps.push({ i, from: prev, to: pts.length - 1, d }); prev = pts.length - 1;
    }
  }
  return { pts, steps, k, r };
}
/** Elementary-effect statistics μ, μ* and σ for each factor from the model outputs Y at plan.pts. */
export function morrisEstimate(plan, Y) {
  const ee = range(plan.k).map(() => []);
  for (const s of plan.steps) ee[s.i].push((Y[s.to] - Y[s.from]) / s.d);
  return { mu: ee.map((e) => mean(e)), muStar: ee.map((e) => mean(e.map(Math.abs))), sigma: ee.map((e) => std(e)) };
}
/** Saltelli sample: matrices A, B and the k hybrid matrices AB(i); N·(k+2) points in the unit cube. */
export function saltelliPlan(k, N, seed = 1) {
  const U = lhs(N, 2 * k, seed), A = U.map((u) => u.slice(0, k)), B = U.map((u) => u.slice(k)), pts = [...A, ...B];
  for (let i = 0; i < k; i++) for (let j = 0; j < N; j++) { const x = [...A[j]]; x[i] = B[j][i]; pts.push(x); }
  return { pts, k, N };
}
/** First-order (Saltelli 2010) and total (Jansen) Sobol indices from the outputs at saltelliPlan points. */
export function saltelliEstimate(Y, k, N) {
  const f0 = mean(Y.slice(0, 2 * N)), Yc = Y.map((q) => q - f0); // centring lowers the estimator variance
  const yA = Yc.slice(0, N), yB = Yc.slice(N, 2 * N), V = variance(Yc) || 1e-300, S = [], ST = []; // variance pooled over all N·(k+2) points
  for (let i = 0; i < k; i++) {
    const yAB = Yc.slice((2 + i) * N, (3 + i) * N);
    let s1 = 0, st = 0;
    for (let j = 0; j < N; j++) { s1 += yB[j] * (yAB[j] - yA[j]); st += (yA[j] - yAB[j]) ** 2; }
    S.push(s1 / N / V); ST.push(st / (2 * N) / V);
  }
  return { S, ST, mean: f0, variance: V };
}
/** Convenience wrapper: Sobol indices of f on the unit cube. */
export function sobolIndices(f, k, N, seed = 1) {
  const plan = saltelliPlan(k, N, seed);
  return saltelliEstimate(plan.pts.map(f), k, N);
}

// ======================================================================================================
// 5 · Surrogate models: polynomial response surface, Gaussian process, feed-forward neural network
// ======================================================================================================
/** Quadratic response surface on standardised inputs; falls back to pure-quadratic or linear terms when data are scarce. */
export function polyFit(X, y, order = 2) {
  const sc = standardizer(X), d = X[0].length, n = X.length, nFull = 1 + d + (d * (d + 1)) / 2;
  const mode = order < 2 ? 'linear' : nFull <= 0.7 * n ? 'full quadratic' : 1 + 2 * d <= 0.7 * n ? 'pure quadratic' : 'linear';
  const feat = (x) => {
    const z = sc.f(x), q = [1, ...z];
    if (mode !== 'linear') for (let i = 0; i < d; i++) for (let j = i; j < d; j++) if (mode === 'full quadratic' || i === j) q.push(z[i] * z[j]);
    return q;
  };
  const beta = lstsq(X.map(feat), y);
  return { predict: (x) => dot(feat(x), beta), beta, mode, nTerms: beta.length };
}

/**
 * Gaussian-process regression with an anisotropic squared-exponential kernel. Hyper-parameters
 * (length scales, signal and noise standard deviations, in log space) maximise the log marginal likelihood
 * unless `theta` is supplied. predict(x) → { mean, sd (latent), sdObs (including noise) }.
 */
export function gpFit(X, y, { theta, theta0, maxIter = 200 } = {}) {
  const sx = standardizer(X), ym = mean(y), ys = std(y) || 1, Z = X.map(sx.f), t = y.map((v) => (v - ym) / ys), n = Z.length, d = Z[0].length;
  const D2 = range(d).map((k) => { const a = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) a[i * n + j] = (Z[i][k] - Z[j][k]) ** 2; return a; }); // pairwise squared distances per input
  const unpack = (p) => ({ l: p.slice(0, d).map(Math.exp), sf2: Math.exp(2 * p[d]), sn2: Math.exp(2 * p[d + 1]) });
  const build = (p) => {
    const th = unpack(p), w = th.l.map((l) => 0.5 / (l * l)), K = new Array(n);
    for (let i = 0; i < n; i++) K[i] = new Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) { let s = 0; for (let k = 0; k < d; k++) s += w[k] * D2[k][i * n + j]; K[i][j] = K[j][i] = th.sf2 * Math.exp(-s); }
      K[i][i] = th.sf2 + th.sn2 + 1e-9;
    }
    const L = cholesky(K);
    return L ? { th, w, L, alpha: cholSolve(L, t) } : null;
  };
  const nlml = (p) => {
    const m = build(p);
    if (!m) return 1e10;
    let s = 0.5 * dot(t, m.alpha) + 0.5 * n * Math.log(2 * Math.PI);
    for (let i = 0; i < n; i++) s += Math.log(m.L[i][i]);
    return s;
  };
  let p = theta;
  if (!p) {
    const lo = [...new Array(d).fill(-2.3), -3, -7], hi = [...new Array(d).fill(4), 3, 0.5];
    p = nelderMead(nlml, theta0 || [...new Array(d).fill(Math.log(2)), 0, Math.log(0.05)], { lo, hi, maxIter, scale: theta0 ? 0.05 : 0.12, tol: 1e-7 }).x;
  }
  let m = build(p);
  for (let k = 0; !m && k < 6; k++) { p = [...p]; p[d + 1] += 1; m = build(p); } // add noise until the kernel matrix factorises
  if (!m) throw new Error('Gaussian process: the kernel matrix could not be factorised — check for duplicated rows.');
  const predict = (x) => {
    const z = sx.f(x), ks = new Array(n);
    for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < d; k++) s += m.w[k] * (Z[i][k] - z[k]) ** 2; ks[i] = m.th.sf2 * Math.exp(-s); }
    const v = fwdSub(m.L, ks), va = Math.max(0, m.th.sf2 - dot(v, v));
    return { mean: ym + ys * dot(ks, m.alpha), sd: ys * Math.sqrt(va), sdObs: ys * Math.sqrt(va + m.th.sn2) };
  };
  return { predict, theta: p, lml: -nlml(p), lengthScales: m.th.l, signal: ys * Math.sqrt(m.th.sf2), noise: ys * Math.sqrt(m.th.sn2), n };
}

/** Initialise a fully connected network; sizes = [inputs, hidden…, outputs]. Xavier-scaled seeded weights. */
export function nnInit(sizes, seed = 1) {
  const g = rng(seed);
  return sizes.slice(1).map((nOut, l) => { const nIn = sizes[l], s = Math.sqrt(2 / (nIn + nOut)); return { W: Array.from({ length: nOut }, () => Array.from({ length: nIn }, () => g.normal(0, s))), b: zeros(nOut) }; });
}
/** Forward pass (tanh hidden layers, linear output). Returns the activations of every layer. */
export function nnForward(net, x) {
  const acts = [x];
  let a = x;
  for (let l = 0; l < net.length; l++) {
    const { W, b } = net[l], z = new Array(b.length);
    for (let i = 0; i < b.length; i++) { let s = b[i]; const w = W[i]; for (let j = 0; j < a.length; j++) s += w[j] * a[j]; z[i] = l < net.length - 1 ? Math.tanh(s) : s; }
    acts.push(z); a = z;
  }
  return acts;
}
/** Mean-squared-error loss and its gradient with respect to every weight and bias by back-propagation. */
export function nnLossGrad(net, X, Y) {
  const grads = net.map((l) => ({ W: l.W.map((r) => zeros(r.length)), b: zeros(l.b.length) })), N = X.length;
  let loss = 0;
  for (let s = 0; s < N; s++) {
    const acts = nnForward(net, X[s]), out = acts[acts.length - 1];
    let delta = out.map((o, k) => { const e = o - Y[s][k]; loss += (e * e) / N; return (2 * e) / N; });
    for (let l = net.length - 1; l >= 0; l--) {
      const aPrev = acts[l], g = grads[l], W = net[l].W, nd = l > 0 ? zeros(aPrev.length) : null;
      for (let i = 0; i < delta.length; i++) {
        const di = delta[i], gw = g.W[i], w = W[i];
        g.b[i] += di;
        for (let j = 0; j < aPrev.length; j++) { gw[j] += di * aPrev[j]; if (nd) nd[j] += w[j] * di; }
      }
      if (nd) { for (let j = 0; j < nd.length; j++) nd[j] *= 1 - aPrev[j] * aPrev[j]; delta = nd; }
    }
  }
  return { loss, grads };
}
function adamStep(params, grads, st, lr) { // one Adam update of a flat parameter array
  st.t++;
  const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** st.t, c2 = 1 - b2 ** st.t;
  for (let i = 0; i < params.length; i++) {
    st.m[i] = b1 * st.m[i] + (1 - b1) * grads[i]; st.v[i] = b2 * st.v[i] + (1 - b2) * grads[i] * grads[i];
    params[i] -= (lr * (st.m[i] / c1)) / (Math.sqrt(st.v[i] / c2) + 1e-8);
  }
}
/** Flat-array engine used for training: same network and gradient as nnForward / nnLossGrad without per-sample allocation. */
export function nnFast(sizes) {
  const L = sizes.length - 1, off = [], act = sizes.map((s) => new Float64Array(s)), del = sizes.map((s) => new Float64Array(s));
  let nW = 0;
  for (let l = 0; l < L; l++) { off.push(nW); nW += sizes[l + 1] * (sizes[l] + 1); }
  const fwd = (w, x) => {
    act[0].set(x);
    for (let l = 0; l < L; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1], a = act[l], z = act[l + 1], o = off[l], ob = o + nOut * nIn;
      for (let i = 0; i < nOut; i++) { let s = w[ob + i]; const r = o + i * nIn; for (let j = 0; j < nIn; j++) s += w[r + j] * a[j]; z[i] = l < L - 1 ? Math.tanh(s) : s; }
    }
    return act[L];
  };
  /** Gradient of the mean squared error over samples idx[from..to) into g; returns the summed squared error. */
  const grad = (w, Z, T, idx, from, to, g) => {
    g.fill(0);
    let sse = 0;
    const m = to - from, nO = sizes[L];
    for (let q = from; q < to; q++) {
      const s = idx[q], out = fwd(w, Z[s]);
      for (let k = 0; k < nO; k++) { const e = out[k] - T[s][k]; sse += e * e; del[L][k] = (2 * e) / m; }
      for (let l = L - 1; l >= 0; l--) {
        const nIn = sizes[l], nOut = sizes[l + 1], a = act[l], dl = del[l + 1], dp = del[l], o = off[l], ob = o + nOut * nIn;
        if (l > 0) dp.fill(0);
        for (let i = 0; i < nOut; i++) { const di = dl[i], r = o + i * nIn; g[ob + i] += di; for (let j = 0; j < nIn; j++) { g[r + j] += di * a[j]; if (l > 0) dp[j] += w[r + j] * di; } }
        if (l > 0) for (let j = 0; j < nIn; j++) dp[j] *= 1 - a[j] * a[j];
      }
    }
    return sse;
  };
  const mse = (w, Z, T) => { let s = 0; for (let i = 0; i < Z.length; i++) { const o = fwd(w, Z[i]); for (let k = 0; k < o.length; k++) s += (o[k] - T[i][k]) ** 2; } return s / Z.length; };
  return { nW, fwd, grad, mse, pack: (net) => Float64Array.from(net.flatMap((l) => [...l.W.flat(), ...l.b])),
    unpack: (w) => range(L).map((l) => { const nIn = sizes[l], nOut = sizes[l + 1]; return { W: range(nOut).map((i) => Array.from(w.subarray(off[l] + i * nIn, off[l] + (i + 1) * nIn))), b: Array.from(w.subarray(off[l] + nOut * nIn, off[l] + nOut * (nIn + 1))) }; }) };
}
/**
 * Train a small tanh network with mini-batch Adam on standardised data. A validation set (Xval, yval) enables
 * early stopping: training stops after `patience` epochs without improvement and the best weights are restored.
 */
export function nnTrain(X, y, { hidden = [8], epochs = 600, lr = 0.02, seed = 1, Xval, yval, patience = 80, l2 = 1e-4, batch = 8 } = {}) {
  const sx = standardizer(X), ym = mean(y), ys = std(y) || 1, Z = X.map(sx.f), T = y.map((v) => [(v - ym) / ys]), n = Z.length;
  const hasVal = Xval && Xval.length > 0, Zv = hasVal ? Xval.map(sx.f) : null, Tv = hasVal ? yval.map((v) => [(v - ym) / ys]) : null;
  const sizes = [X[0].length, ...hidden, 1], E = nnFast(sizes), w = E.pack(nnInit(sizes, seed)), gr = new Float64Array(E.nW), st = { t: 0, m: new Float64Array(E.nW), v: new Float64Array(E.nW) };
  const g = rng(seed + 17), bs = clamp(Math.round(batch) || n, 1, n), hist = { epoch: [], train: [], val: [] }, every = Math.max(1, Math.floor(epochs / 150));
  let best = Infinity, bestW = Float64Array.from(w), bestEpoch = 0, ep = 0;
  for (; ep < epochs; ep++) {
    const order = shuffled(n, g), rate = lr * (0.1 + 0.9 * (1 - ep / epochs));
    let sse = 0;
    for (let s = 0; s < n; s += bs) {
      sse += E.grad(w, Z, T, order, s, Math.min(n, s + bs), gr);
      for (let i = 0; i < E.nW; i++) gr[i] += 2 * l2 * w[i];
      adamStep(w, gr, st, rate);
    }
    const vl = hasVal ? E.mse(w, Zv, Tv) : sse / n;
    if (ep % every === 0) { hist.epoch.push(ep); hist.train.push(sse / n); hist.val.push(vl); }
    if (vl < best - 1e-9) { best = vl; bestW.set(w); bestEpoch = ep; } else if (hasVal && ep - bestEpoch > patience) break;
  }
  return { predict: (x) => ym + ys * E.fwd(bestW, sx.f(x))[0], history: hist, epochs: ep, bestEpoch, net: E.unpack(bestW), nWeights: E.nW, stoppedEarly: ep < epochs };
}

// ---- Physics-informed network for the film (concentration-polarisation) equation dθ/dξ = Pe·θ, θ(0) = 1 ---
/** Loss (mean squared ODE residual at the collocation points + weighted boundary error) and analytic gradient. */
export function pinnLossGrad(p, xi, Pe, wBC = 10) {
  const H = p.w.length, M = xi.length, g = { w: zeros(H), b: zeros(H), v: zeros(H), c: 0 };
  let loss = 0;
  for (const x of xi) {
    let N = p.c, dN = 0;
    const t = new Array(H), s = new Array(H);
    for (let j = 0; j < H; j++) { t[j] = Math.tanh(p.w[j] * x + p.b[j]); s[j] = 1 - t[j] * t[j]; N += p.v[j] * t[j]; dN += p.v[j] * p.w[j] * s[j]; }
    const r = dN - Pe * N, k = (2 * r) / M;
    loss += (r * r) / M; g.c += -k * Pe;
    for (let j = 0; j < H; j++) {
      const ds = -2 * t[j] * s[j];
      g.v[j] += k * (p.w[j] * s[j] - Pe * t[j]);
      g.w[j] += k * (p.v[j] * s[j] + p.v[j] * p.w[j] * ds * x - Pe * p.v[j] * s[j] * x);
      g.b[j] += k * (p.v[j] * p.w[j] * ds - Pe * p.v[j] * s[j]);
    }
  }
  let N0 = p.c;
  for (let j = 0; j < H; j++) N0 += p.v[j] * Math.tanh(p.b[j]);
  const e = N0 - 1, kb = 2 * wBC * e;
  loss += wBC * e * e; g.c += kb;
  for (let j = 0; j < H; j++) { const tb = Math.tanh(p.b[j]); g.v[j] += kb * tb; g.b[j] += kb * p.v[j] * (1 - tb * tb); }
  return { loss, g };
}
/** Train the single-hidden-layer network θ̂(ξ) = c + Σ vⱼ·tanh(wⱼξ + bⱼ) on the residual of dθ/dξ = Pe·θ with θ(0) = 1. */
export function pinnTrain(Pe, { neurons = 10, iters = 3000, lr = 0.02, nColl = 24, seed = 1, wBC = 10 } = {}) {
  const g = rng(seed), H = Math.max(2, Math.round(neurons)), xi = linspace(0, 1, Math.max(4, Math.round(nColl)));
  const p = { w: range(H).map(() => g.normal(0, 1)), b: range(H).map(() => g.normal(0, 0.5)), v: range(H).map(() => g.normal(0, 0.5)), c: 1 };
  const pack = (q) => [...q.w, ...q.b, ...q.v, q.c], unpack = (a) => { p.w = a.slice(0, H); p.b = a.slice(H, 2 * H); p.v = a.slice(2 * H, 3 * H); p.c = a[3 * H]; };
  const a = pack(p), st = { t: 0, m: zeros(a.length), v: zeros(a.length) }, history = { it: [], loss: [] };
  let best = Infinity, bestA = [...a];
  for (let it = 0; it < iters; it++) {
    const { loss, g: gr } = pinnLossGrad(p, xi, Pe, wBC);
    if (loss < best) { best = loss; bestA = [...a]; }
    if (it % Math.max(1, Math.floor(iters / 120)) === 0) { history.it.push(it); history.loss.push(loss); }
    adamStep(a, pack(gr), st, lr * 0.02 ** (it / iters)); unpack(a);
  }
  unpack(bestA);
  const predict = (x) => { let N = p.c; for (let j = 0; j < H; j++) N += p.v[j] * Math.tanh(p.w[j] * x + p.b[j]); return N; };
  return { p, predict, history, loss: best, nParams: a.length };
}

// ======================================================================================================
// 6 · Time series: Holt–Winters, autoregression, extended Kalman filter for a fouling state model
// ======================================================================================================
/** Additive Holt–Winters with damped trend; par = { alpha, beta, gamma, phi }, m = season length (< 2 → no season). */
export function holtWinters(y, m, par) {
  const n = y.length, seas = m >= 2 && n >= 2 * m, { alpha, beta, gamma, phi } = par, fit = [];
  let L, T, S = [];
  if (seas) { const m1 = mean(y.slice(0, m)), m2 = mean(y.slice(m, 2 * m)); L = m1; T = (m2 - m1) / m; S = y.slice(0, m).map((v) => v - m1); }
  else { const q = Math.min(n - 1, 5); L = y[0]; T = q > 0 ? (y[q] - y[0]) / q : 0; }
  for (let t = 0; t < n; t++) {
    const s = seas ? S[t % m] : 0;
    fit.push(L + phi * T + s);
    const Ln = alpha * (y[t] - s) + (1 - alpha) * (L + phi * T);
    T = beta * (Ln - L) + (1 - beta) * phi * T;
    if (seas) S[t % m] = gamma * (y[t] - Ln) + (1 - gamma) * s;
    L = Ln;
  }
  const forecast = (h) => { let ph = 0, q = 1; return range(h).map((k) => { q *= phi; ph += q; return L + ph * T + (seas ? S[(n + k) % m] : 0); }); };
  return { fit, forecast, level: L, trend: T, season: S, seasonal: seas };
}
/** Fit the Holt–Winters smoothing constants by minimising the one-step-ahead squared error (Nelder–Mead). */
export function hwFit(y, m) {
  const start = m >= 2 && y.length >= 2 * m ? m : 1;
  const sse = (p) => { const f = holtWinters(y, m, { alpha: p[0], beta: p[1], gamma: p[2], phi: p[3] }).fit; let s = 0; for (let t = start; t < y.length; t++) s += (y[t] - f[t]) ** 2; return s; };
  const r = nelderMead(sse, [0.3, 0.05, 0.1, 0.95], { lo: [0.02, 0.001, 0.001, 0.8], hi: [0.98, 0.5, 0.9, 1], maxIter: 250, scale: 0.15, tol: 1e-9 });
  const par = { alpha: r.x[0], beta: r.x[1], gamma: r.x[2], phi: r.x[3] }, model = holtWinters(y, m, par);
  const resid = y.map((v, t) => v - model.fit[t]), sigma = Math.sqrt(r.f / Math.max(1, y.length - start - 4));
  /** Forecast standard deviation h steps ahead (state-space form of the additive damped model). */
  const sdAhead = (h) => { let v = 1, ph = 0, q = 1; const out = []; for (let j = 1; j <= h; j++) { out.push(sigma * Math.sqrt(v)); q *= par.phi; ph += q; const c = par.alpha * (1 + par.beta * ph) + (model.seasonal && j % m === 0 ? par.gamma * (1 - par.alpha) : 0); v += c * c; } return out; };
  return { par, model, resid, sigma, sdAhead, start };
}
/** AR(p) on the series (d = 0) or on its first difference (d = 1) by least squares, with ψ-weights for prediction intervals. */
export function arFit(y, p, d = 0) {
  const w = d ? y.slice(1).map((v, i) => v - y[i]) : [...y], X = [], t = [];
  for (let k = p; k < w.length; k++) { X.push([1, ...range(p).map((i) => w[k - 1 - i])]); t.push(w[k]); }
  if (X.length < p + 3) throw new Error('Autoregression: not enough data points for the chosen order.');
  const beta = lstsq(X, t), c = beta[0], phi = beta.slice(1), res = X.map((r, i) => t[i] - dot(r, beta));
  const sigma = Math.sqrt(sum(res.map((v) => v * v)) / Math.max(1, X.length - p - 1));
  // equivalent AR polynomial acting on the levels: (1 − B)^d · φ(B)
  const a = d ? range(p + 1).map((i) => (i === 0 ? 1 + phi[0] : i === p ? -phi[p - 1] : phi[i] - phi[i - 1])) : phi;
  const forecast = (hist, h) => { const b = [...hist.slice(-a.length)], out = []; for (let k = 0; k < h; k++) { let v = c; for (let i = 0; i < a.length; i++) v += a[i] * b[b.length - 1 - i]; out.push(v); b.push(v); } return out; };
  const sdAhead = (h) => { const psi = [1], out = []; let v = 0; for (let j = 0; j < h; j++) { if (j > 0) { let s = 0; for (let i = 1; i <= Math.min(j, a.length); i++) s += a[i - 1] * psi[j - i]; psi.push(s); } v += psi[j] * psi[j]; out.push(sigma * Math.sqrt(v)); } return out; };
  return { c, phi, a, sigma, forecast, sdAhead, order: p, d };
}
/**
 * Extended Kalman filter for the fouling state model K' = −r·(K − K∞), r' = 0 with measurement y = K + noise.
 * State x = [K, r]. Innovations beyond `gate` standard deviations are flagged and not assimilated (unless they persist).
 */
export function ekfFouling(t, y, { Kinf = 0.7, r0 = 0.005, K0, sdMeas = 0.01, qK = 1e-7, qr = 1e-9, P0 = [1e-3, 1e-5], gate = 3 } = {}) {
  let K = K0 ?? y[0], r = r0, P = [[P0[0], 0], [0, P0[1]]], run = 0;
  const R = sdMeas * sdMeas, out = { K: [], rate: [], pred: [], z: [], flag: [], sdK: [], sdPred: [] };
  for (let k = 0; k < y.length; k++) {
    const dt = k > 0 ? Math.max(0, t[k] - t[k - 1]) : 0;
    if (dt > 0) {
      const F = [[1 - r * dt, -(K - Kinf) * dt], [0, 1]];
      K -= r * (K - Kinf) * dt;
      const FP = matMul(F, P); P = matMul(FP, transpose(F));
      P[0][0] += qK * dt; P[1][1] += qr * dt;
    }
    const S = P[0][0] + R, nu = y[k] - K, z = nu / Math.sqrt(S), bad = Math.abs(z) > gate;
    out.pred.push(K); out.sdPred.push(Math.sqrt(S)); out.z.push(z); out.flag.push(bad);
    run = bad ? run + 1 : 0;
    if (!bad || run > 5) {
      const g0 = P[0][0] / S, g1 = P[1][0] / S;
      K += g0 * nu; r = Math.max(0, r + g1 * nu);
      P = [[(1 - g0) * P[0][0], (1 - g0) * P[0][1]], [P[1][0] - g1 * P[0][0], P[1][1] - g1 * P[0][1]]];
      P[0][1] = P[1][0] = 0.5 * (P[0][1] + P[1][0]);
    }
    out.K.push(K); out.rate.push(r); out.sdK.push(Math.sqrt(Math.max(0, P[0][0])));
  }
  /** Propagate the final state h steps of size dt without measurements: mean and standard deviation of K. */
  out.forecast = (h, dt) => {
    let k1 = K, PP = P.map((q) => [...q]);
    const m = [], sd = [];
    for (let i = 0; i < h; i++) {
      const F = [[1 - r * dt, -(k1 - Kinf) * dt], [0, 1]];
      k1 -= r * (k1 - Kinf) * dt;
      PP = matMul(matMul(F, PP), transpose(F)); PP[0][0] += qK * dt; PP[1][1] += qr * dt;
      m.push(k1); sd.push(Math.sqrt(Math.max(0, PP[0][0]) + R));
    }
    return { mean: m, sd };
  };
  out.x = [K, r]; out.P = P;
  return out;
}

// ======================================================================================================
// 7 · Control: linear MPC as a constrained QP, PI controller, tabular Q-learning and dynamic programming
// ======================================================================================================
/**
 * Condensed linear MPC for x⁺ = A x + B u + w, y = C x. Minimises Σ (y − r)ᵀQ(y − r) + Σ ΔuᵀRΔu over N moves
 * subject to u, Δu and output bounds. Q, R: diagonal weights (arrays). Returns { solve(x, uPrev, ref, w) }.
 */
export function mpcBuild({ A, B, C }, { N = 15, Q, R, umin, umax, dumax, ymin, ymax } = {}) {
  const nx = A.length, nu = B[0].length, ny = C.length, nU = N * nu;
  const pw = [eye(nx)];
  for (let k = 1; k <= N; k++) pw.push(matMul(pw[k - 1], A));
  const CA = pw.map((M) => matMul(C, M)), CAB = pw.map((M) => matMul(matMul(C, M), B));
  const Su = zeros(N * ny, nU), Sx = zeros(N * ny, nx), Sw = zeros(N * ny, nx);
  for (let k = 1; k <= N; k++) {
    const acc = zeros(ny, nx);
    for (let j = 0; j < k; j++) {
      for (let a = 0; a < ny; a++) { for (let b = 0; b < nu; b++) Su[(k - 1) * ny + a][j * nu + b] = CAB[k - 1 - j][a][b]; for (let b = 0; b < nx; b++) acc[a][b] += CA[j][a][b]; }
    }
    for (let a = 0; a < ny; a++) for (let b = 0; b < nx; b++) { Sx[(k - 1) * ny + a][b] = CA[k][a][b]; Sw[(k - 1) * ny + a][b] = acc[a][b]; }
  }
  const H = zeros(nU, nU);
  for (let r = 0; r < N * ny; r++) { const q = Q[r % ny]; if (q) for (let a = 0; a < nU; a++) { const sa = Su[r][a]; if (sa) for (let b = 0; b < nU; b++) H[a][b] += 2 * q * sa * Su[r][b]; } }
  for (let k = 0; k < N; k++) for (let b = 0; b < nu; b++) { // ΔuᵀRΔu with Δu_k = u_k − u_{k−1}
    const i = k * nu + b, w = 2 * R[b];
    H[i][i] += w;
    if (k > 0) { const j = i - nu; H[j][j] += w; H[i][j] -= w; H[j][i] -= w; }
  }
  for (let a = 0; a < nU; a++) H[a][a] += 1e-9;
  const Ac = [], kind = [];
  const unit = (i, s) => range(nU).map((q) => (q === i ? s : 0));
  for (let k = 0; k < N; k++) for (let b = 0; b < nu; b++) {
    const i = k * nu + b;
    if (umax && Number.isFinite(umax[b])) { Ac.push(unit(i, 1)); kind.push(['umax', k, b]); }
    if (umin && Number.isFinite(umin[b])) { Ac.push(unit(i, -1)); kind.push(['umin', k, b]); }
    if (dumax && Number.isFinite(dumax[b])) {
      const up = unit(i, 1), dn = unit(i, -1);
      if (k > 0) { up[i - nu] = -1; dn[i - nu] = 1; }
      Ac.push(up); kind.push(['dup', k, b]); Ac.push(dn); kind.push(['ddn', k, b]);
    }
  }
  for (let k = 0; k < N; k++) for (let a = 0; a < ny; a++) {
    const r = k * ny + a;
    if (ymax && Number.isFinite(ymax[a])) { Ac.push([...Su[r]]); kind.push(['ymax', r, a]); }
    if (ymin && Number.isFinite(ymin[a])) { Ac.push(Su[r].map((v) => -v)); kind.push(['ymin', r, a]); }
  }
  const prep = qpPrepare(H, Ac);
  let warm = null;
  return {
    H, nConstraints: Ac.length,
    solve(x, uPrev, ref, w = zeros(nx)) {
      const free = range(N * ny).map((r) => dot(Sx[r], x) + dot(Sw[r], w)), c = zeros(nU);
      for (let r = 0; r < N * ny; r++) { const q = Q[r % ny]; if (!q) continue; const e = free[r] - (typeof ref === 'function' ? ref(Math.floor(r / ny) + 1)[r % ny] : ref[r % ny]); for (let a = 0; a < nU; a++) c[a] += 2 * q * Su[r][a] * e; }
      for (let b = 0; b < nu; b++) c[b] -= 2 * R[b] * uPrev[b];
      const bb = kind.map(([t, k, i]) => (t === 'umax' ? umax[i] : t === 'umin' ? -umin[i] : t === 'dup' ? dumax[i] + (k === 0 ? uPrev[i] : 0) : t === 'ddn' ? dumax[i] - (k === 0 ? uPrev[i] : 0) : t === 'ymax' ? ymax[i] - free[k] : free[k] - ymin[i]));
      const qp = qpSolve(prep, c, bb, { maxIter: clamp(Math.round(2.5e6 / (Ac.length * Ac.length + 1)), 30, 250), tol: 1e-8, lam0: warm }); // iteration budget bounded for large horizons
      warm = qp.lambda;
      return { u: qp.x.slice(0, nu), U: qp.x, iterations: qp.iterations, active: qp.lambda.filter((l) => l > 1e-9).length };
    },
  };
}

/**
 * Tabular Q-learning for a finite-horizon problem. env = { T, nS, nA, step(t, s, a) → { s2, reward }, terminal(s) }.
 * Returns the Q table, the per-block learning curve and the greedy policy.
 */
export function qLearn(env, { episodes = 4000, alpha = 0.3, gamma = 1, eps0 = 1, epsMin = 0.05, seed = 1, blocks = 40, s0 } = {}) {
  const g = rng(seed), { T, nS, nA } = env, Q = Array.from({ length: T + 1 }, () => Array.from({ length: nS }, () => zeros(nA)));
  const argmax = (q) => { let b = 0; for (let a = 1; a < q.length; a++) if (q[a] > q[b]) b = a; return b; };
  const curve = { episode: [], explore: [], greedy: [] }, per = Math.max(1, Math.floor(episodes / blocks));
  const greedyReturn = (start) => { let s = start, G = 0; for (let t = 0; t < T; t++) { const r = env.step(t, s, argmax(Q[t][s])); G += r.reward; s = r.s2; } return G + env.terminal(s); };
  let acc = 0;
  for (let ep = 0; ep < episodes; ep++) {
    const eps = Math.max(epsMin, eps0 * (1 - ep / (0.7 * episodes)));
    let s = g.int(nS), G = 0; // exploring starts
    for (let t = 0; t < T; t++) {
      const a = g.uniform() < eps ? g.int(nA) : argmax(Q[t][s]), r = env.step(t, s, a);
      const target = r.reward + (t === T - 1 ? env.terminal(r.s2) : gamma * Math.max(...Q[t + 1][r.s2]));
      Q[t][s][a] += alpha * (target - Q[t][s][a]);
      G += r.reward + (t === T - 1 ? env.terminal(r.s2) : 0); s = r.s2;
    }
    acc += G;
    if ((ep + 1) % per === 0) { curve.episode.push(ep + 1); curve.explore.push(acc / per); curve.greedy.push(greedyReturn(s0 ?? 0)); acc = 0; }
  }
  return { Q, curve, policy: (t, s) => argmax(Q[t][s]), greedyReturn };
}
/** Exact finite-horizon dynamic programme (Bellman backward induction) for the same environment. */
export function dpSolve(env) {
  const { T, nS, nA } = env, V = Array.from({ length: T + 1 }, () => zeros(nS)), pol = Array.from({ length: T }, () => zeros(nS));
  for (let s = 0; s < nS; s++) V[T][s] = env.terminal(s);
  for (let t = T - 1; t >= 0; t--) for (let s = 0; s < nS; s++) {
    let best = -Infinity, ba = 0;
    for (let a = 0; a < nA; a++) { const r = env.step(t, s, a), q = r.reward + V[t + 1][r.s2]; if (q > best) { best = q; ba = a; } }
    V[t][s] = best; pol[t][s] = ba;
  }
  return { V, policy: (t, s) => pol[t][s] };
}

// ======================================================================================================
// 8 · 1-D transient convection–diffusion–reaction solver (finite volumes, θ-scheme, Thomas algorithm)
// ======================================================================================================
/**
 * ∂u/∂t + v ∂u/∂x = D ∂²u/∂x² + R(u, x, t) on 0 ≤ x ≤ L with cell-centred finite volumes.
 * left/right = { type: 'dirichlet' | 'neumann' | 'noflux', val(t) } (neumann value = ∂u/∂x; noflux = zero total flux).
 * theta = 0.5 Crank–Nicolson, 1 backward Euler; the reaction term is iterated (Picard) at the θ-level.
 * Returns { x, t (saved times), U (saved profiles), u (final), mean (final domain average) }.
 */
export function solveCDR({ L = 1, nx = 40, tEnd = 1, nt = 100, v = 0, D = 1, R, ic, left, right, theta = 0.5, scheme = 'central', nSave = 30 }) {
  const n = Math.max(3, Math.round(nx)), M = Math.max(1, Math.round(nt)), dx = L / n, dt = tEnd / M, x = range(n).map((i) => (i + 0.5) * dx);
  const up = scheme === 'upwind';
  const aL = up ? (v >= 0 ? v + D / dx : D / dx) : v / 2 + D / dx, aR = up ? (v >= 0 ? -D / dx : v - D / dx) : v / 2 - D / dx;
  // semi-discrete operator du/dt = Lop·u + s(t) + R
  const lo = zeros(n), di = zeros(n), hi = zeros(n);
  for (let i = 0; i < n; i++) {
    if (i < n - 1) { di[i] -= aL / dx; hi[i] -= aR / dx; }
    if (i > 0) { lo[i] += aL / dx; di[i] += aR / dx; }
  }
  if (left.type === 'dirichlet') di[0] -= (2 * D) / dx / dx; else if (left.type === 'neumann') di[0] += v / dx;
  if (right.type === 'dirichlet') di[n - 1] -= (2 * D) / dx / dx; else if (right.type === 'neumann') di[n - 1] -= v / dx;
  const src = (t) => {
    const s0 = left.type === 'dirichlet' ? ((v + (2 * D) / dx) * left.val(t)) / dx : left.type === 'neumann' ? (-(v * left.val(t) * dx) / 2 - D * left.val(t)) / dx : 0;
    const s1 = right.type === 'dirichlet' ? (-(v - (2 * D) / dx) * right.val(t)) / dx : right.type === 'neumann' ? -((v * right.val(t) * dx) / 2 - D * right.val(t)) / dx : 0;
    return [s0, s1];
  };
  const a = lo.map((q) => -theta * dt * q), b = di.map((q) => 1 - theta * dt * q), c = hi.map((q) => -theta * dt * q);
  let u = x.map((xi) => finite(ic(xi))), t = 0;
  const every = Math.max(1, Math.ceil(M / nSave)), ts = [0], U = [[...u]];
  for (let k = 0; k < M; k++) {
    const [s0a, s1a] = src(t), [s0b, s1b] = src(t + dt), tm = t + theta * dt, rhs0 = new Array(n);
    for (let i = 0; i < n; i++) rhs0[i] = u[i] + (1 - theta) * dt * ((i > 0 ? lo[i] * u[i - 1] : 0) + di[i] * u[i] + (i < n - 1 ? hi[i] * u[i + 1] : 0));
    rhs0[0] += dt * (theta * s0b + (1 - theta) * s0a); rhs0[n - 1] += dt * (theta * s1b + (1 - theta) * s1a);
    let un = u;
    for (let pic = 0; pic < (R ? 3 : 1); pic++) {
      const rhs = R ? rhs0.map((q, i) => q + dt * finite(R(theta * un[i] + (1 - theta) * u[i], x[i], tm))) : rhs0;
      un = tridiag(a, b, c, rhs);
    }
    u = un; t += dt;
    if ((k + 1) % every === 0 || k === M - 1) { ts.push(t); U.push([...u]); }
  }
  return { x, t: ts, U, u, mean: mean(u), dx, dt };
}

// ======================================================================================================
// 9 · RO plant model wrapper (suite 1 engine) shared by the optimisation, sensitivity, UQ and surrogate tasks
// ======================================================================================================
let RO_DEF = null;
const roDefaults = () => (RO_DEF ||= Object.fromEntries(roSuite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value])));
const SW_SET = ['swhr', 'swle', 'swule'], BW_SET = ['bwhr', 'bwle'];
const memProps = (key) => { const M = MEMBRANES[key] || MEMBRANES.swhr; return { membrane: MEMBRANES[key] ? key : 'swhr', A: M.A, B: M.B, area: M.area, spacerMil: M.spacerMil }; };
const memName = (key) => (MEMBRANES[key]?.name || key).replace(/ \(.*\)$/, '');
/** RO input object of the base design from this suite's inputs. */
function roBase(v) {
  return { ...roDefaults(), ions: cloneIons(v.ions), Qf: v.Qf, T: v.T, pH: v.pH, ...memProps(v.membrane), recovery: v.recovery0, targetFlux: v.flux0, elements: Math.round(v.elements0),
    ff: v.ff, etaPump: v.etaPump, erd: v.erd, erdEff: v.erdEff, nSeg: clamp(Math.round(v.roSeg), 1, 6), design: 'auto', mode: 'recovery', pass2: false, recycle: 0, salinityFactor: 1 };
}
/** Solve the RO plant at a fixed recovery with a secant iteration on feed pressure (several times faster than bracketing). */
const warm = { corr: 1, slope: 0.012, calls: 0 }; // pressure-estimate correction and dR/dP carried from the previous solve (reset every run)
const warmReset = () => { warm.corr = 1; warm.slope = 0.012; warm.calls = 0; };
function roSolve(inp) {
  const target = inp.recovery / 100, el = Math.round(inp.elements);
  const vessels = inp.design === 'auto' ? autoSize({ Qp: inp.Qf * target, flux: inp.targetFlux, area: inp.area, elements: el, recovery: target }) : [inp.v1, inp.v2, inp.v3].slice(0, Math.round(inp.nStages));
  const q = { ...inp, elements: el, design: 'manual', nStages: vessels.length, v1: vessels[0], v2: vessels[1] || 1, v3: vessels[2] || 1, mode: 'pressure' };
  const f = (P) => { warm.calls++; const r = simulateRO({ ...q, Pfeed: P }); return [r.p1.recovery - target, r]; };
  const piF = osmoticPressureIons(scaleIons(cloneIons(inp.ions), inp.salinityFactor ?? 1), inp.T) / 1e5;
  const flux = (inp.Qf * target * 1000) / (sum(vessels) * el * inp.area), Aeff = Math.max(0.05, inp.A * inp.ff * (1 + 0.03 * (inp.T - 25)));
  const est = inp.Pp + (1.1 * piF * -Math.log(1 - target)) / target + flux / Aeff + 0.7, Pmin = inp.Pp + 0.3;
  let P0 = clamp(est * warm.corr, Pmin, 200), [f0, r1] = f(P0), P1 = P0, f1 = f0;
  if (Math.abs(f0) > 1e-6) {
    P1 = clamp(P0 - f0 / warm.slope, Math.max(Pmin, 0.88 * P0), 1.12 * P0);
    if (P1 === P0) P1 = P0 * 1.01;
    [f1, r1] = f(P1);
    for (let it = 0; it < 16 && Math.abs(f1) > 1e-6; it++) {
      const slope = (f1 - f0) / (P1 - P0);
      if (!(slope > 1e-9)) break;
      warm.slope = slope;
      const P2 = clamp(P1 - f1 / slope, Math.max(Pmin, P1 - 20), P1 + 20);
      if (P2 === P1) break;
      P0 = P1; f0 = f1; P1 = P2; [f1, r1] = f(P1);
    }
  }
  if (Math.abs(f1) > 1e-6) return simulateRO({ ...q, mode: 'recovery' }); // fall back to the bracketing solver of suite 1
  warm.corr = clamp(P1 / est, 0.6, 1.6);
  return r1;
}
const crf = (i, n) => (i > 1e-9 ? (i * (1 + i) ** n) / ((1 + i) ** n - 1) : 1 / n);
/** Simple cost of water ($/m³ product): energy + membrane and vessel capital recovery + chemicals/pre-treatment + brine disposal. */
function costOfWater(m, v) {
  const annual = Math.max(1e-9, m.Qp * 8760 * (v.avail / 100)), i = v.interest / 100, r = Math.max(1e-6, m.rec / 100);
  const c = { energy: m.sec * v.elecPrice, membrane: (m.nEl * v.elemPrice * crf(i, Math.max(0.5, v.memLife)) + m.nV * v.vesselPrice * crf(i, 20)) / annual, chemicals: v.chemPrice / r, brine: (v.brinePrice * (1 - r)) / r };
  c.total = c.energy + c.membrane + c.chemicals + c.brine;
  return c;
}
const CONS = [
  { id: 'tds', label: 'Product TDS', unit: 'mg/L', val: (m) => m.tds, lim: (v) => v.limTDS },
  { id: 'boron', label: 'Product boron', unit: 'mg/L', val: (m) => m.boron, lim: (v) => v.limBoron },
  { id: 'flux', label: 'Lead-element flux', unit: 'L/m²·h', val: (m) => m.maxFlux, lim: (v) => v.limFlux },
  { id: 'P', label: 'Feed pressure', unit: 'bar', val: (m) => m.Pf, lim: (v, m) => Math.min(v.limP, m.pmax) },
  { id: 'cp', label: 'Polarisation factor β', unit: '–', val: (m) => m.maxCP, lim: (v) => v.limCP },
  { id: 'conc', label: 'Concentrate flow per vessel', unit: 'm³/h', val: (m) => m.qcMin, lim: (v) => v.limConc, min: true },
  { id: 'sec', label: 'Specific energy (cap)', unit: 'kWh/m³', val: (m) => m.sec, lim: (v) => v.limSEC, on: (v) => v.limSEC > 0 },
  { id: 'si', label: 'Concentrate saturation index', unit: 'SI', val: (m) => m.si, lim: (v) => finite(v.limSI), abs: true, on: (v) => !!SCALE_SETS[v.conScale] },
];
const SCALE_SETS = { gypsum: ['gypsum'], calcite: ['calcite'], barite: ['barite'], celestite: ['celestite'], silica: ['silica'], sulfate: ['gypsum', 'barite', 'celestite'] };
/** Constraints in force for the current inputs: the six operating limits plus the optional energy cap and scaling limit. */
const consFor = (v) => CONS.filter((c) => !c.on || c.on(v));
const concentratePH = (ions, pH, feedTds) => Math.min(9, pH + 0.3 * Math.log10(Math.max(1, tds(ions) / Math.max(feedTds, 1e-9))));
/** Largest saturation index of the selected scale minerals in the concentrate (Pitzer speciation of suite 2 at the estimated concentrate pH). */
export function concentrateSI(ions, T, pH, which = 'gypsum') {
  const ids = SCALE_SETS[which] || SCALE_SETS.gypsum, SI = saturation(makeSolution({ ions, T, pH }).eq, 1, {}, ids);
  return Math.max(...ids.map((k) => finite(SI[k], -99)));
}
/** Simulate one design and return performance, cost and normalised constraint values g ≤ 0. */
function evalDesign(inp, v) {
  try {
    const r = roSolve(inp), els = r.p1.stages.flatMap((s) => s.els);
    const m = { sec: r.sec, Pf: r.p1.Pf, tds: tds(r.product.ions), boron: r.product.ions.B, maxFlux: Math.max(...els.map((e) => e.flux)), maxCP: Math.max(...els.map((e) => e.CP)),
      qcMin: Math.min(...r.p1.stages.map((s) => s.conc.Q / s.nV)), rec: 100 * r.overallRec, nEl: r.nEl, nV: sum(r.vessels), area: r.area, Qp: r.product.Q, Qb: r.conc.Q, power: r.power,
      avgFlux: (r.p1.perm.Q * 1000) / r.area, vessels: r.vessels, pmax: r.cfg.M.pmax, minSEC: r.minSEC };
    if (![m.sec, m.Pf, m.tds, m.maxFlux, m.maxCP].every(Number.isFinite)) throw new Error('non-finite result');
    if (SCALE_SETS[v.conScale]) { m.concPH = concentratePH(r.conc.ions, inp.pH, tds(r.feed.ions)); m.si = concentrateSI(r.conc.ions, inp.T, m.concPH, v.conScale); m.concIons = r.conc.ions; if (!Number.isFinite(m.si)) throw new Error('non-finite saturation index'); }
    const cost = costOfWater(m, v), g = consFor(v).map((c) => { const L = c.lim(v, m), x = c.val(m); return c.abs ? x - L : c.min ? (L - x) / Math.max(L, 1e-9) : (x - L) / Math.max(L, 1e-9); });
    return { ok: true, m, cost, g, viol: sum(g.map((q) => Math.max(0, q))), inp };
  } catch (e) { const nc = consFor(v).length; return { ok: false, g: new Array(nc).fill(1), viol: nc, err: e.message, inp }; }
}
const OBJ = {
  sec: { label: 'Specific energy', unit: 'kWh/m³', f: (e) => e.m.sec },
  cost: { label: 'Cost of water', unit: '$/m³', f: (e) => e.cost.total },
  maxrec: { label: 'Recovery', unit: '%', f: (e) => -e.m.rec, show: (x) => -x },
  brine: { label: 'Brine volume', unit: 'm³ per m³ product', f: (e) => (100 - e.m.rec) / e.m.rec },
};
/** Decision variables (continuous, integer and categorical) with their bounds, from the inputs. */
function decisionVars(v) {
  const set = v.memSet === 'bw' ? BW_SET : v.memSet === 'sw' ? SW_SET : [v.membrane], D = [];
  const add = (d) => { if (d.hi > d.lo) D.push(d); };
  add({ key: 'recovery', label: 'Recovery', unit: '%', lo: v.recLo, hi: v.recHi, x0: v.recovery0 });
  add({ key: 'targetFlux', label: 'Average flux', unit: 'L/m²·h', lo: v.fluxLo, hi: v.fluxHi, x0: v.flux0 });
  if (v.optPp) add({ key: 'Pp', label: 'Permeate back-pressure', unit: 'bar', lo: 0.5, hi: v.ppHi, x0: 1 });
  if (v.optBoost) add({ key: 'boost2', label: 'Inter-stage boost', unit: 'bar', lo: 0, hi: v.boostHi, x0: 0 });
  add({ key: 'elements', label: 'Elements per vessel', unit: '', int: true, lo: Math.round(v.elLo), hi: Math.round(v.elHi), x0: Math.round(v.elements0) });
  if (set.length > 1) add({ key: 'membrane', label: 'Element class', unit: '', int: true, cat: set, lo: 0, hi: set.length - 1, x0: Math.max(0, set.indexOf(v.membrane)) });
  return D;
}
/** Cached design evaluator on the unit cube (integers and categories are rounded inside). */
function makeEvaluator(v, base, vars) {
  const cache = new Map();
  let evals = 0;
  const decode = (u) => {
    const o = {};
    vars.forEach((d, j) => { let x = d.lo + clamp(u[j], 0, 1) * (d.hi - d.lo); if (d.int) x = Math.round(x); if (d.cat) Object.assign(o, memProps(d.cat[x])); else o[d.key] = x; });
    return o;
  };
  const at = (u) => {
    const o = decode(u), k = vars.map((d) => (d.cat ? o.membrane : (+o[d.key]).toFixed(5))).join('|');
    let e = cache.get(k);
    if (!e) { evals++; e = evalDesign({ ...base, ...o }, v); e.o = o; cache.set(k, e); }
    return e;
  };
  const encode = (o) => vars.map((d) => clamp(((d.cat ? d.cat.indexOf(o.membrane) : o[d.key]) - d.lo) / (d.hi - d.lo), 0, 1));
  return { at, decode, encode, get evals() { return evals; } };
}
/** Extra KPIs and tables when the energy cap or the scaling constraint is switched on: full speciation of the concentrate by suite 2. */
function designEvidence(v, eBase, eOpt) {
  const kpis = [], tables = [];
  if (v.limSEC > 0) kpis.push({ label: 'Specific-energy cap', value: v.limSEC, unit: 'kWh/m³', status: eOpt.m.sec <= v.limSEC * (1 + FEAS) ? 'ok' : 'bad', help: 'Energy constraint: specific energy of the optimum must not exceed this value.' });
  if (SCALE_SETS[v.conScale] && Number.isFinite(eOpt.m.si)) {
    const full = (e) => (e.ok && e.m.concIons ? analyzeWater({ ions: e.m.concIons, T: e.inp.T, pH: e.m.concPH }) : null), a0 = full(eBase), a1 = full(eOpt), ids = ['gypsum', 'barite', 'celestite', 'calcite', 'silica', 'fluorite'];
    kpis.push({ label: 'Concentrate saturation index', value: eOpt.m.si, unit: 'log₁₀(IAP/K)', status: eOpt.m.si <= finite(v.limSI) + 1e-3 ? 'ok' : 'bad', help: `Largest index of the constrained minerals (${SCALE_SETS[v.conScale].join(', ')}); limit ${fmt(finite(v.limSI))}.` });
    if (a1) tables.push({ title: 'Scaling of the concentrate (electrolyte model of suite 2)', columns: ['Mineral', 'Saturation index, base case', 'Saturation index, optimum', 'Constrained', 'Limit'],
      rows: [...ids.map((k) => [k, a0 ? finite(a0.SI[k], -99) : null, finite(a1.SI[k], -99), SCALE_SETS[v.conScale].includes(k) ? 'yes' : 'no', SCALE_SETS[v.conScale].includes(k) ? finite(v.limSI) : null]), ['Concentrate pH (estimated)', eBase.ok ? eBase.m.concPH ?? null : null, eOpt.m.concPH, '', null], ['Concentrate TDS (mg/L)', a0 ? a0.tds : null, a1.tds, '', null]],
      note: 'The constraint SI ≤ limit is evaluated for every candidate design from a Pitzer speciation of the simulated concentrate; this table repeats the full analysis at the base case and at the optimum. −99 marks a mineral whose constituents are absent.' });
  }
  return { kpis, tables };
}
const designRow = (vars, o) => vars.map((d) => (d.cat ? memName(o.membrane) : o[d.key]));
const designOut = (e) => ({ recovery: e.m.rec, targetFlux: e.inp.targetFlux, elements: e.inp.elements, membrane: e.inp.membrane, Pp: e.inp.Pp, boost2: e.inp.boost2, sec: e.m.sec, cost: e.cost.total, productTDS: e.m.tds, feedPressure: e.m.Pf, nElements: e.m.nEl });
const FEAS = 1e-3; // a design is accepted as feasible when no limit is exceeded by more than 0.1 %
const stepXY = (hist) => ({ x: hist.map((h) => h[0]), y: hist.map((h) => h[1]) });

// ---- Task 1: single-objective constrained optimisation -------------------------------------------------------
async function taskOpt(v, ctx) {
  const base = roBase(v), vars = decisionVars(v), n = vars.length, O = OBJ[v.objective] || OBJ.cost, CL = consFor(v);
  if (!n) throw new Error('No decision variable is free: widen at least one pair of bounds.');
  const E = makeEvaluator(v, base, vars), eBase = evalDesign(base, v);
  if (!eBase.ok) throw new Error('The base design cannot be simulated: ' + (eBase.err || 'check the RO plant inputs.'));
  const fScale = Math.abs(O.f(eBase)) || 1, rho = Math.max(0.1, v.optRho);
  const fObj = (e) => (e.ok ? O.f(e) / fScale : 1e3);
  const merit = (e) => (e.ok ? fObj(e) + rho * e.viol + 10 * rho * sum(e.g.map((q) => Math.max(0, q) ** 2)) : 1e3);
  const cIdx = range(n).filter((j) => !vars[j].int), iIdx = range(n).filter((j) => vars[j].int);
  const snap = (u) => u.map((q, j) => (vars[j].int ? Math.round(q * (vars[j].hi - vars[j].lo)) / (vars[j].hi - vars[j].lo) : clamp(q, 0, 1)));
  const embed = (u, uc) => { const w = [...u]; cIdx.forEach((j, q) => (w[j] = uc[q])); return w; };
  const ones = new Array(cIdx.length).fill(1), zc = new Array(cIdx.length).fill(0);
  const nStarts = clamp(Math.round(v.nStarts), 1, 8), seed = Math.round(v.seed), starts = [], lh = lhs(nStarts, n, seed);
  let cur = null;
  const track = (u) => { const m = merit(E.at(u)); cur.calls++; if (m < cur.best - 1e-12) { cur.best = m; cur.hist.push([cur.calls, m]); } return m; };
  const continuous = (u, algo) => { // optimise the continuous variables with the integers held fixed
    if (!cIdx.length) return { u };
    const uc0 = cIdx.map((j) => u[j]);
    if (algo === 'sqp') {
      const r = sqpLite((uc) => { const w = embed(u, uc); track(w); return fObj(E.at(w)); }, (uc) => E.at(embed(u, uc)).g, uc0, { lo: zc, hi: ones, maxIter: Math.round(v.sqpIter), tol: 1e-4, h: 2e-2 } /* a wide difference step averages over the integer vessel-count steps of the array sizing */);
      return { u: embed(u, r.x), sqp: r };
    }
    const r = nelderMead((uc) => track(embed(u, uc)), uc0, { lo: zc, hi: ones, maxIter: Math.round(v.nmIter), tol: 1e-7, scale: 0.12 });
    return { u: embed(u, r.x) };
  };
  const discrete = (u) => { // neighbourhood search on the integer and categorical variables
    let best = merit(E.at(u)), improved = false;
    for (const j of iIdx) {
      const d = vars[j], span = d.hi - d.lo, x0 = Math.round(u[j] * span);
      for (const x of d.cat ? range(span + 1) : [x0 - 1, x0 + 1]) {
        if (x < 0 || x > span || x === Math.round(u[j] * span)) continue;
        const w = [...u]; w[j] = x / span;
        const m = track(w);
        if (m < best - 1e-7) { best = m; u = w; improved = true; }
      }
    }
    return { u, improved };
  };
  for (let k = 0; k < nStarts; k++) {
    ctx.progress(0.05 + (0.75 * k) / nStarts, `Start ${k + 1} of ${nStarts} (${v.algo === 'de' ? 'differential evolution' : v.algo === 'sqp' ? 'SQP' : 'Nelder–Mead'})`);
    await ctx.tick();
    cur = { calls: 0, best: Infinity, hist: [], seed: seed + k };
    let u = snap(k === 0 ? vars.map((d) => clamp((d.x0 - d.lo) / (d.hi - d.lo), 0, 1)) : lh[k]), sqp = null;
    if (v.algo === 'de') {
      const r = diffEvolution(track, new Array(n).fill(0), new Array(n).fill(1), { pop: Math.max(5, Math.round(v.dePop)), gens: Math.max(1, Math.round(v.deGens)), F: 0.6, CR: 0.9, seed: seed + k });
      u = snap(r.x);
      if (v.optPolish && cIdx.length) u = continuous(u, 'nm').u; // local polish of the global result
    } else {
      for (let round = 0; round < 3; round++) {
        const c = continuous(u, v.algo); u = c.u; sqp = c.sqp || sqp;
        const dsc = discrete(u); u = dsc.u;
        if (!dsc.improved) break;
      }
    }
    const e = E.at(u);
    cur.hist.push([cur.calls, cur.best]);
    starts.push({ ...cur, u, e, merit: merit(e), sqp });
  }
  const bestS = starts.reduce((a, b) => (b.merit < a.merit ? b : a));
  const eOpt = bestS.e;
  if (!eOpt.ok) throw new Error('No design inside the bounds could be simulated. Widen the bounds or check the plant inputs.');
  ctx.progress(0.88, 'Optimality check and trade-off sweep'); await ctx.tick();
  // first-order optimality in the continuous variables (integers fixed at the optimum)
  const kk = cIdx.length ? kktCheck((uc) => fObj(E.at(embed(bestS.u, uc))), (uc) => E.at(embed(bestS.u, uc)).g, cIdx.map((j) => bestS.u[j]), { lo: zc, hi: ones, h: 1e-2, actTol: 4e-3 }) : { lambda: CL.map(() => 0), residual: 0, relResidual: 0, lamLo: [], lamHi: [], active: [] };
  const feasible = Math.max(...eOpt.g) <= FEAS, fOpt = O.f(eOpt), f0 = O.f(eBase), show = O.show || ((x) => x);
  const gain = (100 * (f0 - fOpt)) / Math.abs(f0 || 1);
  // sweep of the objective along recovery through the optimum
  const jR = vars.findIndex((d) => d.key === 'recovery'), sweep = { x: [], f: [], viol: [] };
  if (jR >= 0) for (const q of linspace(0, 1, 9)) { const w = [...bestS.u]; w[jR] = q; const e = E.at(w); if (e.ok) { sweep.x.push(e.m.rec); sweep.f.push(show(O.f(e))); sweep.viol.push(100 * Math.max(...e.g)); } }
  const W = [], atBound = vars.filter((d, j) => !d.int && (bestS.u[j] < 1e-3 || bestS.u[j] > 1 - 1e-3));
  if (!feasible) W.push({ level: 'bad', msg: `No fully feasible design was found: the best point still violates ${CL.filter((c, i) => eOpt.g[i] > FEAS).map((c) => lcFirst(c.label)).join(', ')}. Relax a limit, widen the bounds or allow another element class.` });
  else W.push({ level: 'info', msg: 'The optimum satisfies every operating constraint.' });
  if (Math.max(...eBase.g) > FEAS) W.push({ level: 'warn', msg: `The base design itself violates ${CL.filter((c, i) => eBase.g[i] > FEAS).map((c) => lcFirst(c.label)).join(', ')} — the comparison is against an infeasible reference.` });
  if (atBound.length) W.push({ level: 'info', msg: `${atBound.map((d) => d.label).join(', ')} ended on a bound: the optimum is limited by the search range, not by the plant.` });
  const feasStarts = starts.filter((s) => s.e.ok), spread = feasStarts.length > 1 ? (Math.max(...feasStarts.map((s) => s.merit)) - Math.min(...feasStarts.map((s) => s.merit))) / Math.max(1e-9, Math.abs(bestS.merit)) : 0;
  if (spread > 0.02) W.push({ level: 'warn', msg: `The ${nStarts} starts disagree by ${fmt(100 * spread, 2)} % in the penalised objective — the problem is multi-modal or not fully converged; use more starts, generations or the global algorithm.` });
  if (cIdx.length && feasible && kk.relResidual > 0.15) W.push({ level: 'warn', msg: `The first-order optimality residual is ${fmt(100 * kk.relResidual, 2)} % of the objective gradient: the point is good but not a tight stationary point (finite-difference noise or an integer-limited optimum).` });
  const active = CL.filter((c, i) => eOpt.g[i] > -4e-3), costCats = ['Energy', 'Membranes + vessels', 'Chemicals / pre-treatment', 'Brine disposal'], costOf = (e) => [e.cost.energy, e.cost.membrane, e.cost.chemicals, e.cost.brine];
  const extra = designEvidence(v, eBase, eOpt);
  const perfRows = [['Recovery (%)', (e) => e.m.rec], ['Average flux (L/m²·h)', (e) => e.m.avgFlux], ['Elements per vessel', (e) => e.inp.elements], ['Element class', (e) => memName(e.inp.membrane)], ['Vessels per stage', (e) => e.m.vessels.join(' : ')],
    ['Elements installed', (e) => e.m.nEl], ['Feed pressure (bar)', (e) => e.m.Pf], ['Specific energy (kWh/m³)', (e) => e.m.sec], ['Cost of water ($/m³)', (e) => e.cost.total], ['Product flow (m³/h)', (e) => e.m.Qp], ['Brine flow (m³/h)', (e) => e.m.Qb],
    ['Product TDS (mg/L)', (e) => e.m.tds], ['Product boron (mg/L)', (e) => e.m.boron], ['Lead-element flux (L/m²·h)', (e) => e.m.maxFlux], ['Polarisation factor β', (e) => e.m.maxCP], ['Permeate back-pressure (bar)', (e) => e.inp.Pp], ['Inter-stage boost (bar)', (e) => e.inp.boost2]];
  return {
    summary: `${O.label} ${gain < 0 ? 'changed' : O.show ? 'raised' : 'reduced'} from ${fmt(show(f0), 4)} to ${fmt(show(fOpt), 4)} ${O.unit} (${fmt(Math.abs(gain), 3)} % ${gain >= 0 ? 'better' : 'worse'} than the base case) at ${fmt(eOpt.m.rec, 3)} % recovery, ${fmt(eOpt.m.avgFlux, 3)} L/m²·h and ${eOpt.inp.elements} × ${memName(eOpt.inp.membrane)} elements per vessel; ${feasible ? `all constraints met, ${active.length} active` : 'constraints NOT all met'}.`,
    warnings: W,
    kpis: [
      { label: `Optimal ${lcFirst(O.label)}`, value: show(fOpt), unit: O.unit, status: feasible ? 'ok' : 'bad' }, { label: 'Improvement on base case', value: gain, unit: '%', status: gain >= 0 ? 'ok' : 'warn' },
      { label: 'Optimal recovery', value: eOpt.m.rec, unit: '%' }, { label: 'Optimal average flux', value: eOpt.m.avgFlux, unit: 'L/m²·h' },
      { label: 'Elements per vessel', value: eOpt.inp.elements }, { label: 'Element class', value: memName(eOpt.inp.membrane) },
      { label: 'Specific energy', value: eOpt.m.sec, unit: 'kWh/m³' }, { label: 'Cost of water', value: eOpt.cost.total, unit: '$/m³' },
      { label: 'Feed pressure', value: eOpt.m.Pf, unit: 'bar' }, { label: 'Product TDS', value: eOpt.m.tds, unit: 'mg/L', status: eOpt.g[0] > FEAS ? 'bad' : 'ok' },
      { label: 'Largest constraint violation', value: 100 * Math.max(0, ...eOpt.g), unit: '% of limit', status: feasible ? 'ok' : 'bad' }, { label: 'Active constraints', value: active.length ? active.map((c) => c.label).join(', ') : 'none' },
      { label: 'KKT stationarity residual', value: kk.relResidual, unit: 'relative', status: kk.relResidual < 0.15 ? 'ok' : 'warn', help: '‖∇f + Σλ∇g‖∞ / ‖∇f‖∞ in the continuous variables with non-negative multipliers on the active set' }, { label: 'Model evaluations', value: E.evals },
      ...extra.kpis,
    ],
    recommendations: [
      feasible ? `Apply the optimum to suite 1 (RO design): it is offered there as the recommended recovery of ${fmt(eOpt.m.rec, 3)} %.` : 'Relax the most violated limit or add a second pass in suite 1, then re-run the optimisation.',
      active.length ? `The optimum is held by ${active.map((c) => lcFirst(c.label)).join(' and ')}; the shadow prices in the constraint table show how much the objective would gain per 1 % relaxation of each limit.` : 'No constraint is active: the optimum is an interior trade-off of the objective itself.',
      'Run the multi-objective task to see the whole energy–recovery–quality trade-off instead of a single point.',
      'Confirm the scaling margin of the optimal recovery in suite 2 and the full cost of water in suite 13.',
    ],
    plots: [
      { type: 'line', title: 'Convergence history (best penalised objective)', xlabel: 'Objective evaluations', ylabel: 'Penalised objective relative to base case', series: starts.map((s, k) => ({ name: `Start ${k + 1} (seed ${s.seed})`, ...stepXY(s.hist), mode: 'step' })), hlines: [{ y: 1, label: 'base case' }] },
      { type: 'bar', title: 'Constraint utilisation (100 % = at the limit)', ylabel: '% of limit', categories: CL.map((c) => c.label), series: [{ name: 'Base case', values: eBase.g.map((q) => 100 * (1 + q)) }, { name: 'Optimum', values: eOpt.g.map((q) => 100 * (1 + q)) }], note: 'For the minimum-concentrate-flow limit the bar shows how far the flow has fallen towards the minimum (100 % = at the minimum).' },
      { type: 'bar', title: 'Cost of water breakdown', ylabel: '$/m³', stacked: true, categories: ['Base case', 'Optimum'], series: costCats.map((nm, i) => ({ name: nm, values: [costOf(eBase)[i], costOf(eOpt)[i]] })) },
      ...(sweep.x.length > 2 ? [{ type: 'line', title: `${O.label} versus recovery through the optimum`, xlabel: 'Recovery (%)', ylabel: O.unit, series: [{ name: O.label, x: sweep.x, y: sweep.f, mode: 'both' }], vlines: [{ x: eOpt.m.rec, label: 'optimum' }] },
        { type: 'line', title: 'Most critical constraint versus recovery', xlabel: 'Recovery (%)', ylabel: 'Largest normalised constraint (% over limit)', series: [{ name: 'max g(x)', x: sweep.x, y: sweep.viol, mode: 'both' }], hlines: [{ y: 0, label: 'feasible below this line' }], vlines: [{ x: eOpt.m.rec, label: 'optimum' }] }] : []),
      { type: 'bar', title: 'Multi-start results', ylabel: `${O.label} (${O.unit})`, categories: starts.map((s, k) => `Start ${k + 1}`), series: [{ name: O.label, values: starts.map((s) => (s.e.ok ? show(O.f(s.e)) : 0)) }] },
    ],
    tables: [
      { title: 'Optimum compared with the base case', columns: ['Quantity', 'Base case', 'Optimum'], rows: perfRows.map(([nm, f]) => [nm, f(eBase), f(eOpt)]) },
      { title: 'Constraints at the optimum', columns: ['Constraint', 'Type', 'Value', 'Limit', CL.some((c) => c.abs) ? 'Slack (% of limit; index units for the saturation index)' : 'Slack (% of limit)', 'Status', 'Multiplier λ', `Shadow price (${O.unit} per +1 % of limit)`],
        rows: CL.map((c, i) => [`${c.label} (${c.unit})`, c.min ? '≥' : '≤', c.val(eOpt.m), c.lim(v, eOpt.m), c.abs ? -eOpt.g[i] : -100 * eOpt.g[i], eOpt.g[i] > FEAS ? 'violated' : eOpt.g[i] > -4e-3 ? 'active' : 'inactive', kk.lambda[i], kk.lambda[i] * fScale * 0.01]),
        note: `Constraints are normalised as g = (value − limit)/limit ≤ 0 and enforced with the exact penalty ρ·Σmax(0, g) plus a quadratic term (ρ = ${fmt(rho)}); a limit counts as met when it is exceeded by less than 0.1 %. Multipliers are non-negative least-squares estimates from finite-difference gradients at the optimum with the integer variables held fixed; KKT residual ${fmt(kk.residual, 3)} (relative ${fmt(kk.relResidual, 3)}).` },
      { title: 'Decision variables and bounds', columns: ['Variable', 'Lower bound', 'Upper bound', 'Base', 'Optimum', 'Type', 'Bound multiplier'],
        rows: vars.map((d, j) => { const q = cIdx.indexOf(j); return [d.label + (d.unit ? ` (${d.unit})` : ''), d.cat ? memName(d.cat[0]) : d.lo, d.cat ? memName(d.cat[d.cat.length - 1]) : d.hi, d.cat ? memName(base.membrane) : base[d.key], d.cat ? memName(eOpt.inp.membrane) : eOpt.o[d.key], d.cat ? 'categorical' : d.int ? 'integer' : 'continuous', q >= 0 ? Math.max(kk.lamLo[q], kk.lamHi[q]) : null]; }) },
      { title: 'Multi-start summary', columns: ['Start', 'Seed', ...vars.map((d) => d.label), `${O.label} (${O.unit})`, 'Max violation (%)', 'Objective calls', 'SQP iterations', 'SQP KKT residual'],
        rows: starts.map((s, k) => [k + 1, s.seed, ...designRow(vars, s.e.o), s.e.ok ? show(O.f(s.e)) : null, 100 * Math.max(0, ...s.e.g), s.calls, s.sqp ? s.sqp.iterations : null, s.sqp ? s.sqp.kkt : null]) },
      ...extra.tables,
    ],
    outputs: { task: 'opt', best: designOut(eOpt), objective: show(fOpt), objectiveName: v.objective, feasible, baseObjective: show(f0), study: { a: show(fOpt), b: eOpt.m.rec } },
  };
}

// ---- Task 2: multi-objective optimisation (NSGA-II) ----------------------------------------------------------
async function taskMO(v, ctx) {
  const base = roBase(v), vars = decisionVars(v), n = vars.length;
  if (!n) throw new Error('No decision variable is free: widen at least one pair of bounds.');
  const E = makeEvaluator(v, base, vars), eBase = evalDesign(base, v), third = v.moThird;
  const objs = [{ label: 'Specific energy', lc: 'minimise specific energy', f: (e) => e.m.sec }, { label: 'Recovery', lc: 'maximise recovery', f: (e) => -e.m.rec }];
  if (third === 'tds') objs.push({ label: 'Product TDS', lc: 'minimise product TDS', f: (e) => e.m.tds });
  if (third === 'cost') objs.push({ label: 'Cost of water', lc: 'minimise cost of water', f: (e) => e.cost.total });
  const evalFn = (u) => { const e = E.at(u); return e.ok ? { f: objs.map((o) => o.f(e)), cv: Math.max(...e.g) > FEAS ? e.viol : 0, e } : { f: objs.map(() => 1e6), cv: 100, e }; };
  const gens = Math.max(1, Math.round(v.moGens)), S = nsga2Init(evalFn, new Array(n).fill(0), new Array(n).fill(1), { pop: Math.round(v.moPop), seed: Math.round(v.seed) });
  const conv = { g: [], sec: [], rec: [], nFront: [] };
  const log = () => { const F = paretoFront(S.P).filter((p) => p.e.ok); conv.g.push(S.gen); conv.sec.push(F.length ? Math.min(...F.map((p) => p.e.m.sec)) : 0); conv.rec.push(F.length ? Math.max(...F.map((p) => p.e.m.rec)) : 0); conv.nFront.push(F.length); };
  log();
  for (let g = 0; g < gens; g++) { nsga2Step(S); log(); ctx.progress(0.05 + (0.9 * (g + 1)) / gens, `Generation ${g + 1} of ${gens}`); await ctx.tick(); }
  const seen = new Set(), front = paretoFront(S.P).filter((p) => p.e.ok && !seen.has(p.e) && seen.add(p.e)).sort((a, b) => a.e.m.rec - b.e.m.rec).slice(0, 60);
  if (!front.length) throw new Error('No simulated design survived: widen the bounds or check the plant inputs.');
  const feasible = front[0].cv === 0, knee = kneePoint(front), ek = knee.e, set = vars.find((d) => d.cat)?.cat || [base.membrane];
  const W = [];
  if (!feasible) W.push({ level: 'bad', msg: 'No design satisfies all constraints; the points shown are the least-violating ones. Relax a limit or widen the bounds.' });
  if (front.length < 5) W.push({ level: 'warn', msg: `Only ${front.length} distinct non-dominated designs were found — increase the population or the number of generations.` });
  W.push({ level: 'info', msg: `${S.evals} design evaluations requested, ${E.evals} distinct plant simulations (others reused from cache).` });
  const X = (f) => front.map((p) => f(p.e)), pt = (e, f) => ({ x: [e.m.rec], y: [f(e)] });
  const scatter = (title, ylabel, f) => ({ type: 'line', title, xlabel: 'Recovery (%)', ylabel, series: [{ name: 'Pareto designs', x: X((e) => e.m.rec), y: X(f), mode: objs.length === 2 ? 'both' : 'points' }, { name: 'Knee point (recommended)', ...pt(ek, f), mode: 'points' }, ...(eBase.ok ? [{ name: 'Base case', ...pt(eBase, f), mode: 'points' }] : [])] });
  const extremes = objs.map((o, k) => front.reduce((a, b) => (b.f[k] < a.f[k] ? b : a)));
  return {
    summary: `${front.length} non-dominated designs span ${fmt(Math.min(...X((e) => e.m.sec)), 3)}–${fmt(Math.max(...X((e) => e.m.sec)), 3)} kWh/m³ and ${fmt(Math.min(...X((e) => e.m.rec)), 3)}–${fmt(Math.max(...X((e) => e.m.rec)), 3)} % recovery. Recommended compromise (knee): ${fmt(ek.m.rec, 3)} % recovery, ${fmt(ek.m.sec, 3)} kWh/m³, ${fmt(ek.m.tds, 3)} mg/L product TDS with ${memName(ek.inp.membrane)} elements.`,
    warnings: W,
    kpis: [
      { label: 'Pareto designs', value: front.length, status: front.length >= 5 ? 'ok' : 'warn' }, { label: 'Knee recovery', value: ek.m.rec, unit: '%' }, { label: 'Knee specific energy', value: ek.m.sec, unit: 'kWh/m³' },
      { label: 'Knee product TDS', value: ek.m.tds, unit: 'mg/L' }, { label: 'Knee cost of water', value: ek.cost.total, unit: '$/m³' }, { label: 'Knee average flux', value: ek.m.avgFlux, unit: 'L/m²·h' },
      { label: 'Knee element class', value: memName(ek.inp.membrane) }, { label: 'Lowest energy on front', value: Math.min(...X((e) => e.m.sec)), unit: 'kWh/m³' }, { label: 'Highest recovery on front', value: Math.max(...X((e) => e.m.rec)), unit: '%' },
      { label: 'Lowest product TDS on front', value: Math.min(...X((e) => e.m.tds)), unit: 'mg/L' }, { label: 'Feasible front', value: feasible ? 'yes' : 'no', status: feasible ? 'ok' : 'bad' }, { label: 'Plant simulations', value: E.evals },
    ],
    recommendations: [
      `Use the knee design as a balanced starting point; move along the front toward ${fmt(Math.max(...X((e) => e.m.rec)), 3)} % recovery if intake and brine costs dominate, or toward ${fmt(Math.min(...X((e) => e.m.sec)), 3)} kWh/m³ if electricity dominates.`,
      'Every point on the front is optimal for some weighting of the objectives — pick with the cost of water column or with site-specific priorities, not by the single-objective optimum alone.',
      'Re-run with another seed: a robust front should overlay the present one.',
    ],
    plots: [
      scatter('Pareto front: specific energy versus recovery', 'Specific energy (kWh/m³)', (e) => e.m.sec),
      scatter('Pareto front: product TDS versus recovery', 'Product TDS (mg/L)', (e) => e.m.tds),
      scatter('Cost of water along the front', 'Cost of water ($/m³)', (e) => e.cost.total),
      { type: 'line', title: 'Product quality versus energy', xlabel: 'Specific energy (kWh/m³)', ylabel: 'Product TDS (mg/L)', series: [{ name: 'Pareto designs', x: X((e) => e.m.sec), y: X((e) => e.m.tds), mode: 'points' }, { name: 'Knee point', x: [ek.m.sec], y: [ek.m.tds], mode: 'points' }] },
      { type: 'line', title: 'Convergence of the front by generation', xlabel: 'Generation', ylabel: 'Energy (kWh/m³) · recovery/10 (%) · designs/10', series: [{ name: 'Lowest energy on front', x: conv.g, y: conv.sec, mode: 'both' }, { name: 'Highest recovery ÷ 10', x: conv.g, y: conv.rec.map((q) => q / 10), mode: 'both' }, { name: 'Front size ÷ 10', x: conv.g, y: conv.nFront.map((q) => q / 10), mode: 'step' }] },
    ],
    tables: [
      { title: 'Pareto-optimal designs (sorted by recovery)', columns: ['#', ...vars.map((d) => d.label + (d.unit ? ` (${d.unit})` : '')), 'Recovery achieved (%)', 'SEC (kWh/m³)', 'Product TDS (mg/L)', 'Boron (mg/L)', 'Feed pressure (bar)', 'Cost ($/m³)', 'Elements', 'Knee'],
        rows: front.map((p, i) => [i + 1, ...designRow(vars, p.e.o), p.e.m.rec, p.e.m.sec, p.e.m.tds, p.e.m.boron, p.e.m.Pf, p.e.cost.total, p.e.m.nEl, p === knee ? '◀ recommended' : '']) },
      { title: 'Extreme designs and the recommended compromise', columns: ['Design', 'Recovery (%)', 'SEC (kWh/m³)', 'Product TDS (mg/L)', 'Cost ($/m³)', 'Flux (L/m²·h)', 'Element class'],
        rows: [...extremes.map((p, k) => [`Extreme: ${objs[k].lc}`, p.e]), ['Knee point', ek], ...(eBase.ok ? [['Base case', eBase]] : [])].map(([nm, e]) => [nm, e.m.rec, e.m.sec, e.m.tds, e.cost.total, e.m.avgFlux, memName(e.inp.membrane)]),
        note: `NSGA-II: population ${S.N}, ${gens} generations, simulated binary crossover (η = 15) and polynomial mutation (η = 20), constrained domination. Objectives: ${objs.map((o) => o.lc).join(', ')}. The knee is the design nearest the utopia point after scaling each objective to 0–1.` },
    ],
    outputs: { task: 'mo', best: designOut(ek), objective: ek.m.sec, objectiveName: 'knee SEC', feasible,
      pareto: front.map((p) => ({ recovery: +p.e.m.rec.toFixed(3), flux: +p.e.m.avgFlux.toFixed(3), elements: p.e.inp.elements, membraneIndex: Math.max(0, set.indexOf(p.e.inp.membrane)), sec: +p.e.m.sec.toFixed(4), productTDS: +p.e.m.tds.toFixed(2), cost: +p.e.cost.total.toFixed(4), feedPressure: +p.e.m.Pf.toFixed(2) })),
      study: { a: ek.m.sec, b: ek.m.rec } },
  };
}

// ---- Task 3: sensitivity analysis -----------------------------------------------------------------------------
const SA_OUT = [{ key: 'sec', label: 'Specific energy', unit: 'kWh/m³', f: (e) => e.m.sec }, { key: 'Pf', label: 'Feed pressure', unit: 'bar', f: (e) => e.m.Pf }, { key: 'tds', label: 'Product TDS', unit: 'mg/L', f: (e) => e.m.tds }, { key: 'cost', label: 'Cost of water', unit: '$/m³', f: (e) => e.cost.total }];
function saFactors(v, base) {
  const p = clamp(v.saRange, 0.5, 40) / 100, rel = (key, label, unit, cap = Infinity) => ({ key, label, unit, nom: base[key], lo: base[key] * (1 - p), hi: Math.min(cap, base[key] * (1 + p)) });
  return [{ key: 'salinityFactor', label: 'Feed salinity', unit: '×', nom: 1, lo: 1 - p, hi: 1 + p }, { key: 'T', label: 'Feed temperature', unit: '°C', nom: base.T, lo: Math.max(2, base.T - v.saDT), hi: Math.min(45, base.T + v.saDT) },
    rel('recovery', 'Recovery', '%', 90), rel('targetFlux', 'Average flux', 'L/m²·h'), rel('A', 'Water permeability A', 'L/m²·h·bar'), rel('B', 'Salt permeability B', 'L/m²·h'),
    rel('etaPump', 'Pump efficiency', '%', 93), rel('erdEff', 'Energy-recovery efficiency', '%', 99), rel('kcp', 'Mass-transfer coefficient', '×')];
}
async function taskSA(v, ctx) {
  const base = roBase(v), fac = saFactors(v, base), k = fac.length, e0 = evalDesign(base, v);
  if (!e0.ok) throw new Error('The base design cannot be simulated: ' + (e0.err || 'check the RO plant inputs.'));
  const nom = SA_OUT.map((o) => o.f(e0)), oi = Math.max(0, SA_OUT.findIndex((o) => o.key === v.saOutput)), out = SA_OUT[oi];
  let nEval = 1, nFail = 0;
  const at = (over) => { nEval++; const e = evalDesign({ ...base, ...over }, v); if (!e.ok) { nFail++; return nom; } return SA_OUT.map((o) => o.f(e)); };
  const cube = (u, idx = range(k)) => Object.fromEntries(idx.map((j, q) => [fac[j].key, fac[j].lo + u[q] * (fac[j].hi - fac[j].lo)]));
  const runAll = async (pts, idx, f0, f1, msg) => { const Y = []; for (let i = 0; i < pts.length; i++) { Y.push(at(cube(pts[i], idx))); if (i % 12 === 11) { ctx.progress(f0 + ((f1 - f0) * i) / pts.length, msg); await ctx.tick(); } } return Y; };
  // one-at-a-time tornado
  const oat = fac.map((f) => ({ lo: at({ [f.key]: f.lo }), hi: at({ [f.key]: f.hi }) }));
  ctx.progress(0.12, 'Morris screening'); await ctx.tick();
  // Morris elementary effects
  const mp = morrisPlan(k, clamp(Math.round(v.saR), 2, 50), Math.round(v.seed)), Ym = await runAll(mp.pts, range(k), 0.12, 0.4, 'Morris screening');
  const mor = SA_OUT.map((_, o) => morrisEstimate(mp, Ym.map((y) => y[o])));
  const order = range(k).sort((a, b) => mor[oi].muStar[b] - mor[oi].muStar[a]);
  // Sobol indices on the most influential factors (others held at nominal)
  const kS = clamp(Math.round(v.saTop), 2, k), idx = order.slice(0, kS), N = clamp(Math.round(v.sobolN), 8, 2000), sp = saltelliPlan(kS, N, Math.round(v.seed) + 11);
  const Ys = await runAll(sp.pts, idx, 0.4, 0.97, 'Sobol indices (Saltelli sampling)');
  const sob = SA_OUT.map((_, o) => saltelliEstimate(Ys.map((y) => y[o]), kS, N)), so = sob[oi];
  const ordS = range(kS).sort((a, b) => so.ST[b] - so.ST[a]), sumS = sum(so.S), top = fac[idx[ordS[0]]];
  const swing = fac.map((_, j) => Math.abs(oat[j].hi[oi] - oat[j].lo[oi])), ordT = range(k).sort((a, b) => swing[b] - swing[a]);
  const W = [];
  if (nFail) W.push({ level: 'warn', msg: `${nFail} of ${nEval} sampled designs could not be simulated and were replaced by the nominal result — narrow the ranges.` });
  if (N < 64) W.push({ level: 'info', msg: `Sobol indices use N = ${N} base samples (${sp.pts.length} simulations): rankings are reliable, but individual indices carry roughly ±${fmt(0.6 / Math.sqrt(N), 1)} sampling error. Increase N for publication-quality values.` });
  if (so.S.some((s) => s < -0.1) || sumS > 1.25) W.push({ level: N < 200 ? 'info' : 'warn', msg: 'Some first-order indices fall outside 0–1, a sign of sampling noise: increase the Sobol base sample.' });
  W.push({ level: 'info', msg: `Ranges: ±${fmt(v.saRange)} % on multiplicative factors and ±${fmt(v.saDT)} °C on temperature, uniform. Sobol indices are computed for the ${kS} factors with the largest Morris μ*; the others are held at their nominal value.` });
  const interact = Math.max(0, 1 - sumS), resolved = N >= 200, nonLin = mor[oi].sigma[order[0]] / Math.max(1e-300, mor[oi].muStar[order[0]]);
  return {
    summary: `${top.label} is the dominant driver of ${lcFirst(out.label)} (total Sobol index ${fmt(so.ST[ordS[0]], 3)}), followed by ${lcFirst(fac[idx[ordS[1]]].label)} (${fmt(so.ST[ordS[1]], 3)}). ${resolved ? `First-order effects explain ${fmt(100 * clamp(sumS, 0, 1), 3)} % of the variance; the rest is interaction.` : `The first-order indices sum to ${fmt(sumS, 3)}; with N = ${N} this sum is uncertain by about ±${fmt(1.2 / Math.sqrt(N), 1)}, so the interaction share is not resolved — the Morris σ/μ* ratio of ${fmt(nonLin, 2)} for the top factor indicates ${nonLin > 0.5 ? 'noticeable' : 'weak'} non-linearity or interaction.`}`,
    warnings: W,
    kpis: [
      { label: 'Most influential factor', value: top.label }, { label: 'Its total Sobol index', value: so.ST[ordS[0]], unit: '–' }, { label: 'Its first-order index', value: so.S[ordS[0]], unit: '–' },
      { label: 'Second factor', value: fac[idx[ordS[1]]].label }, { label: 'Sum of first-order indices', value: sumS, unit: '–', status: resolved && (sumS > 1.25 || sumS < 0.5) ? 'warn' : 'ok', help: 'Close to 1 for an additive model; needs a few hundred base samples to be meaningful' }, { label: 'Interaction share', value: resolved ? 100 * interact : 'not resolved at this N', unit: resolved ? '%' : '' }, { label: 'Morris σ/μ* of top factor', value: nonLin, unit: '–', help: 'Above about 0.5: the effect is non-linear or interacts with other factors' },
      { label: `Nominal ${lcFirst(out.label)}`, value: nom[oi], unit: out.unit }, { label: 'Output standard deviation', value: Math.sqrt(so.variance), unit: out.unit }, { label: 'Largest one-at-a-time swing', value: swing[ordT[0]], unit: out.unit },
      { label: 'Factors screened', value: k }, { label: 'Model evaluations', value: nEval },
    ],
    recommendations: [
      `Control or measure ${lcFirst(top.label)} first: it carries the largest share of the variance of ${lcFirst(out.label)}.`,
      `Factors with small μ* and small σ (${order.slice(-2).map((j) => lcFirst(fac[j].label)).join(', ')}) can be fixed at nominal values in calibration and uncertainty studies.`,
      (resolved ? interact > 0.15 : nonLin > 0.5) ? 'Interactions or non-linearity are significant: use the total index (not the tornado) for ranking and vary factors jointly in design studies.' : 'The response is nearly additive: one-at-a-time results are a fair summary here.',
      'Carry the top factors into the uncertainty task with realistic probability distributions.',
    ],
    plots: [
      { type: 'bar', title: `Tornado: one-at-a-time change of ${lcFirst(out.label)}`, ylabel: `Δ ${out.unit}`, categories: ordT.map((j) => fac[j].label), series: [{ name: 'Factor at low value', values: ordT.map((j) => oat[j].lo[oi] - nom[oi]) }, { name: 'Factor at high value', values: ordT.map((j) => oat[j].hi[oi] - nom[oi]) }] },
      { type: 'bar', title: `Morris screening: mean absolute elementary effect μ* (${lcFirst(out.label)})`, ylabel: `μ* (${out.unit} over the full range)`, categories: order.map((j) => fac[j].label), series: [{ name: 'μ* (importance)', values: order.map((j) => mor[oi].muStar[j]) }, { name: 'σ (non-linearity / interaction)', values: order.map((j) => mor[oi].sigma[j]) }] },
      { type: 'line', title: 'Morris μ*–σ plane', xlabel: `μ* (${out.unit})`, ylabel: `σ (${out.unit})`, series: order.slice(0, 8).map((j) => ({ name: fac[j].label, x: [mor[oi].muStar[j]], y: [mor[oi].sigma[j]], mode: 'points' })), note: 'Right = important; high = non-linear or interacting.' },
      { type: 'bar', title: `Sobol indices of ${lcFirst(out.label)}`, ylabel: 'Share of variance (–)', categories: ordS.map((q) => fac[idx[q]].label), series: [{ name: 'First-order S', values: ordS.map((q) => so.S[q]) }, { name: 'Total ST', values: ordS.map((q) => so.ST[q]) }] },
      { type: 'bar', title: 'Total Sobol index for every output', ylabel: 'ST (–)', categories: idx.map((j) => fac[j].label), series: SA_OUT.map((o, q) => ({ name: o.label, values: sob[q].ST })) },
    ],
    tables: [
      { title: 'One-at-a-time (tornado) results', columns: ['Factor', 'Low', 'Nominal', 'High', ...SA_OUT.flatMap((o) => [`${o.label} at low`, `${o.label} at high`])], rows: fac.map((f, j) => [`${f.label} (${f.unit})`, f.lo, f.nom, f.hi, ...SA_OUT.flatMap((_, o) => [oat[j].lo[o], oat[j].hi[o]])]), note: `Nominal outputs: ${SA_OUT.map((o, q) => `${o.label} ${fmt(nom[q], 4)} ${o.unit}`).join('; ')}.` },
      { title: 'Morris elementary effects (per full factor range)', columns: ['Factor', ...SA_OUT.flatMap((o) => [`${o.label} μ`, `${o.label} μ*`, `${o.label} σ`])], rows: order.map((j) => [fac[j].label, ...SA_OUT.flatMap((_, o) => [mor[o].mu[j], mor[o].muStar[j], mor[o].sigma[j]])]), note: `${mp.r} trajectories on a 4-level grid, ${mp.pts.length} simulations. The sign of μ gives the direction of the effect.` },
      { title: 'Sobol variance-based indices', columns: ['Factor', ...SA_OUT.flatMap((o) => [`${o.label} S`, `${o.label} ST`])], rows: ordS.map((q) => [fac[idx[q]].label, ...SA_OUT.flatMap((_, o) => [sob[o].S[q], sob[o].ST[q]])]), note: `Saltelli (2010) first-order and Jansen total estimators, N = ${N}, ${sp.pts.length} simulations, Latin-hypercube base matrices.` },
    ],
    outputs: { task: 'sa', objective: so.ST[ordS[0]], ranking: ordS.map((q) => fac[idx[q]].key), totalIndices: Object.fromEntries(idx.map((j, q) => [fac[j].key, +so.ST[q].toFixed(4)])), study: { a: so.ST[ordS[0]], b: sumS } },
  };
}

// ---- Task 4: uncertainty quantification -------------------------------------------------------------------------
const triInv = (u, a, c, b) => (b <= a ? a : u < (c - a) / (b - a) ? a + Math.sqrt(u * (b - a) * (c - a)) : b - Math.sqrt((1 - u) * (b - a) * (b - c)));
function spearman(a, b) {
  const rank = (x) => { const o = range(x.length).sort((i, j) => x[i] - x[j]), r = new Array(x.length); o.forEach((i, k) => (r[i] = k)); return r; };
  const ra = rank(a), rb = rank(b), ma = mean(ra), mb = mean(rb);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { sab += (ra[i] - ma) * (rb[i] - mb); saa += (ra[i] - ma) ** 2; sbb += (rb[i] - mb) ** 2; }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
}
async function taskUQ(v, ctx) {
  const base = roBase(v), e0 = evalDesign(base, v);
  if (!e0.ok) throw new Error('The base design cannot be simulated: ' + (e0.err || 'check the RO plant inputs.'));
  // the plant is built: the array of the nominal design is kept while the uncertain inputs vary
  const fixed = { ...base, design: 'manual', nStages: e0.m.vessels.length, v1: e0.m.vessels[0], v2: e0.m.vessels[1] || 1, v3: e0.m.vessels[2] || 1 };
  const N = clamp(Math.round(v.uqN), 10, 5000), seed = Math.round(v.seed), g = rng(seed);
  const U = v.uqSampling === 'mc' ? range(N).map(() => range(5).map(() => g.uniform())) : lhs(N, 5, seed);
  const dS = v.uqSalSd / 100, dSP = v.uqSpDrift / 100;
  const IN = [
    { label: 'Feed salinity multiplier', dist: `Normal(1, ${fmt(dS, 3)})`, f: (u) => Math.max(0.3, 1 + dS * normInv(u)), key: 'salinityFactor' },
    { label: 'Feed temperature (°C)', dist: `Normal(${fmt(base.T)}, ${fmt(v.uqTsd)})`, f: (u) => clamp(base.T + v.uqTsd * normInv(u), 2, 45), key: 'T' },
    { label: 'Permeability (flow factor)', dist: `Triangular loss 0–${fmt(v.uqDecl)} %, mode ${fmt(v.uqDecl / 3, 3)} %`, f: (u) => base.ff * (1 - triInv(u, 0, v.uqDecl / 300, v.uqDecl / 100)), key: 'ff' },
    { label: 'Salt permeability multiplier', dist: `Lognormal(median ${fmt(1 + dSP / 2, 4)}, σ_ln ${fmt(dSP / 2, 3)})`, f: (u) => Math.exp(Math.log(1 + dSP / 2) + (dSP / 2) * normInv(u)), key: 'B', mult: true },
    { label: 'Pump efficiency (%)', dist: `Normal(${fmt(base.etaPump)}, ${fmt(v.uqEtaSd)})`, f: (u) => clamp(base.etaPump + v.uqEtaSd * normInv(u), 40, 93), key: 'etaPump' },
  ];
  const OUT = [{ label: 'Specific energy', unit: 'kWh/m³', f: (m) => m.sec }, { label: 'Feed pressure', unit: 'bar', f: (m) => m.Pf, lim: Math.min(v.limP, e0.m.pmax) }, { label: 'Product TDS', unit: 'mg/L', f: (m) => m.tds, lim: v.limTDS },
    { label: 'Product boron', unit: 'mg/L', f: (m) => m.boron, lim: v.limBoron }, { label: 'Lead-element flux', unit: 'L/m²·h', f: (m) => m.maxFlux, lim: v.limFlux }, { label: 'Polarisation factor β', unit: '–', f: (m) => m.maxCP, lim: v.limCP }];
  const xs = [], ys = [];
  let nFail = 0;
  for (let i = 0; i < N; i++) {
    const x = IN.map((d, j) => d.f(U[i][j])), over = Object.fromEntries(IN.map((d, j) => [d.key, d.mult ? base[d.key] * x[j] : x[j]]));
    const e = evalDesign({ ...fixed, ...over }, v);
    if (e.ok) { xs.push(x); ys.push(OUT.map((o) => o.f(e.m))); } else nFail++;
    if (i % 12 === 11) { ctx.progress(0.03 + (0.94 * i) / N, `Sample ${i + 1} of ${N}`); await ctx.tick(); }
  }
  const n = ys.length;
  if (n < 8) throw new Error('Too few samples could be simulated — reduce the input uncertainty or check the plant inputs.');
  const col = (o) => ys.map((y) => y[o]), st = OUT.map((o, q) => { const c = col(q), pv = o.lim !== undefined ? c.filter((x) => x > o.lim).length / n : null; return { c, mean: mean(c), sd: std(c), p5: quantile(c, 0.05), p50: quantile(c, 0.5), p95: quantile(c, 0.95), min: Math.min(...c), max: Math.max(...c), pv }; });
  const anyViol = ys.filter((y) => OUT.some((o, q) => o.lim !== undefined && y[q] > o.lim)).length / n;
  const run = { n: [], m: [], lo: [], hi: [] };
  let s1 = 0, s2 = 0;
  st[0].c.forEach((x, i) => { s1 += x; s2 += x * x; const k = i + 1, m = s1 / k, sd = k > 1 ? Math.sqrt(Math.max(0, (s2 - k * m * m) / (k - 1))) : 0; if (k >= 5 && (k % Math.max(1, Math.floor(n / 80)) === 0 || k === n)) { run.n.push(k); run.m.push(m); run.lo.push(m - (1.96 * sd) / Math.sqrt(k)); run.hi.push(m + (1.96 * sd) / Math.sqrt(k)); } });
  const se = st[0].sd / Math.sqrt(n), hist = (q, nb = 18) => { const h = histogram(st[q].c, nb); return { type: 'bar', title: `Distribution of ${lcFirst(OUT[q].label)}`, ylabel: 'Samples', categories: h.centers.map((x) => fmt(x, 4)), series: [{ name: `${OUT[q].label} (${OUT[q].unit})`, values: h.counts }], note: `P5 ${fmt(st[q].p5, 4)} · P50 ${fmt(st[q].p50, 4)} · P95 ${fmt(st[q].p95, 4)} ${OUT[q].unit}` }; };
  const cdf = (q) => { const s = [...st[q].c].sort((a, b) => a - b); return { type: 'line', title: `Cumulative probability of ${lcFirst(OUT[q].label)}`, xlabel: `${OUT[q].label} (${OUT[q].unit})`, ylabel: 'Probability of not exceeding', ymin: 0, ymax: 1, series: [{ name: 'Empirical CDF', x: s, y: s.map((_, i) => (i + 0.5) / n), mode: 'step' }], vlines: [...(OUT[q].lim !== undefined ? [{ x: OUT[q].lim, label: 'limit' }] : []), { x: st[q].p50, label: 'P50' }] }; };
  const rc = OUT.slice(0, 3).map((_, q) => IN.map((_, j) => spearman(xs.map((x) => x[j]), col(q))));
  const W = [], worst = OUT.map((o, q) => ({ o, pv: st[q].pv })).filter((x) => x.pv !== null).sort((a, b) => b.pv - a.pv)[0];
  if (nFail) W.push({ level: 'warn', msg: `${nFail} of ${N} samples could not reach the target recovery and are excluded; they should be read as additional pressure-limit violations.` });
  if (anyViol > 0.05) W.push({ level: anyViol > 0.25 ? 'bad' : 'warn', msg: `${fmt(100 * anyViol, 3)} % of the sampled operating conditions violate at least one limit (most often ${lcFirst(worst.o.label)}, ${fmt(100 * worst.pv, 3)} %).` });
  else W.push({ level: 'info', msg: anyViol ? `The design is robust: only ${fmt(100 * anyViol, 2)} % of sampled conditions violate a limit.` : 'The design is robust: none of the sampled conditions violates an operating limit.' });
  if (se / Math.abs(st[0].mean) > 0.005) W.push({ level: 'info', msg: `The standard error of the mean specific energy is ${fmt((100 * se) / st[0].mean, 2)} % — increase the sample count for tighter statistics.` });
  return {
    summary: `With ${n} ${v.uqSampling === 'mc' ? 'Monte-Carlo' : 'Latin-hypercube'} samples the specific energy is ${fmt(st[0].p50, 4)} kWh/m³ (P5–P95: ${fmt(st[0].p5, 4)}–${fmt(st[0].p95, 4)}), feed pressure ${fmt(st[1].p50, 3)} bar (P95 ${fmt(st[1].p95, 3)}) and product TDS ${fmt(st[2].p50, 3)} mg/L (P95 ${fmt(st[2].p95, 3)}); ${fmt(100 * anyViol, 3)} % of cases violate an operating limit.`,
    warnings: W,
    kpis: [
      { label: 'SEC, median (P50)', value: st[0].p50, unit: 'kWh/m³' }, { label: 'SEC, P5 – P95', value: `${fmt(st[0].p5, 4)} – ${fmt(st[0].p95, 4)}`, unit: 'kWh/m³' }, { label: 'SEC, mean ± std. error', value: `${fmt(st[0].mean, 4)} ± ${fmt(se, 2)}`, unit: 'kWh/m³' },
      { label: 'Feed pressure P95', value: st[1].p95, unit: 'bar', status: st[1].p95 > OUT[1].lim ? 'bad' : 'ok' }, { label: 'Product TDS P95', value: st[2].p95, unit: 'mg/L', status: st[2].p95 > v.limTDS ? 'bad' : 'ok' }, { label: 'Product boron P95', value: st[3].p95, unit: 'mg/L', status: st[3].p95 > v.limBoron ? 'bad' : 'ok' },
      { label: 'P(pressure limit exceeded)', value: 100 * st[1].pv, unit: '%', status: st[1].pv > 0.05 ? 'warn' : 'ok' }, { label: 'P(TDS limit exceeded)', value: 100 * st[2].pv, unit: '%', status: st[2].pv > 0.05 ? 'warn' : 'ok' }, { label: 'P(boron limit exceeded)', value: 100 * st[3].pv, unit: '%', status: st[3].pv > 0.05 ? 'warn' : 'ok' },
      { label: 'P(flux limit exceeded)', value: 100 * st[4].pv, unit: '%', status: st[4].pv > 0.05 ? 'warn' : 'ok' }, { label: 'P(any limit exceeded)', value: 100 * anyViol, unit: '%', status: anyViol > 0.25 ? 'bad' : anyViol > 0.05 ? 'warn' : 'ok' }, { label: 'Valid samples', value: n },
    ],
    recommendations: [
      st[1].pv > 0.05 ? `Size the high-pressure pump and vessels for the P95 pressure of ${fmt(st[1].p95, 3)} bar, or lower the recovery in the warm/saline season.` : `Rate the high-pressure pump for at least the P95 pressure of ${fmt(st[1].p95, 3)} bar rather than the nominal ${fmt(e0.m.Pf, 3)} bar.`,
      st[2].pv > 0.05 || st[3].pv > 0.05 ? 'Product quality is at risk under ageing and warm water: consider a tighter element class or a partial second pass (suite 1).' : 'Product-quality limits hold across the sampled conditions.',
      'Use P5/P50/P95 energy in the economics suite instead of a single nominal value.',
      'Check convergence on the Mesh tab: the sample-count study re-runs this task at three sample sizes.',
    ],
    plots: [hist(0), hist(1), hist(2), cdf(1), cdf(2),
      { type: 'line', title: 'Convergence of the mean specific energy', xlabel: 'Number of samples', ylabel: 'kWh/m³', series: [{ name: 'Running mean', x: run.n, y: run.m }, { name: '95 % confidence band (low)', x: run.n, y: run.lo, dash: true }, { name: '95 % confidence band (high)', x: run.n, y: run.hi, dash: true }] },
      { type: 'bar', title: 'Rank correlation between uncertain inputs and outputs', ylabel: 'Spearman ρ', categories: IN.map((d) => d.label), series: OUT.slice(0, 3).map((o, q) => ({ name: o.label, values: rc[q] })) }],
    tables: [
      { title: 'Output statistics', columns: ['Output', 'Mean', 'Std. dev.', 'P5', 'P50', 'P95', 'Min', 'Max', 'Nominal', 'Limit', 'P(violation) %', '± 95 % CI (%)'],
        rows: OUT.map((o, q) => [`${o.label} (${o.unit})`, st[q].mean, st[q].sd, st[q].p5, st[q].p50, st[q].p95, st[q].min, st[q].max, o.f(e0.m), o.lim ?? null, st[q].pv === null ? null : 100 * st[q].pv, st[q].pv === null ? null : 196 * Math.sqrt((st[q].pv * (1 - st[q].pv)) / n)]) },
      { title: 'Input distributions sampled', columns: ['Uncertain input', 'Distribution', 'Sample mean', 'Sample std. dev.', 'Min', 'Max'], rows: IN.map((d, j) => { const c = xs.map((x) => x[j]); return [d.label, d.dist, mean(c), std(c), Math.min(...c), Math.max(...c)]; }),
        note: `The membrane array of the nominal design (${e0.m.vessels.join(' : ')} vessels) and the recovery are held fixed; feed pressure adjusts to each sampled condition. Seed ${seed}.` },
    ],
    outputs: { task: 'uq', objective: st[0].mean, secP5: st[0].p5, secP50: st[0].p50, secP95: st[0].p95, pressureP95: st[1].p95, tdsP95: st[2].p95, pViolation: anyViol, study: { a: st[0].mean, b: st[1].p95, c: st[2].mean } },
  };
}

// ---- Task 5: surrogate modelling / machine learning --------------------------------------------------------------
const ML_TARGET = { sec: { label: 'Specific energy', unit: 'kWh/m³', f: (e) => e.m.sec }, Pf: { label: 'Feed pressure', unit: 'bar', f: (e) => e.m.Pf }, tds: { label: 'Product TDS', unit: 'mg/L', f: (e) => e.m.tds }, cost: { label: 'Cost of water', unit: '$/m³', f: (e) => e.cost.total } };
/** Built-in plant-like data set: permeate flow as a function of pressure, temperature and feed salinity with 1.5 % noise. */
function mlSampleTable() {
  const g = rng(2024);
  return lhs(40, 3, 5).map((u) => {
    const P = 45 + 20 * u[0], T = 15 + 17 * u[1], c = 30 + 12 * u[2], y = 1.1 * Math.exp(0.028 * (T - 25)) * Math.max(0, P - 0.92 * c) * (1 + g.normal(0, 0.015));
    return { x1: +P.toFixed(2), x2: +T.toFixed(2), x3: +c.toFixed(2), x4: null, x5: null, x6: null, y: +y.toFixed(3) };
  });
}
const expectedImprovement = (mu, sd, best) => { if (sd < 1e-12) return Math.max(0, best - mu); const z = (best - mu) / sd; return (best - mu) * normCdf(z) + sd * normPdf(z); };
async function taskML(v, ctx) {
  const seed = Math.round(v.seed), fromRO = v.mlSource === 'ro';
  let X, y, names, yName, yUnit, truth = null, lo, hi, nomX = null;
  if (fromRO) {
    const base = roBase(v), tg = ML_TARGET[v.mlTarget] || ML_TARGET.sec;
    const dims = [{ key: 'recovery', label: 'Recovery (%)', lo: v.recLo, hi: v.recHi, nom: base.recovery }, { key: 'targetFlux', label: 'Average flux (L/m²·h)', lo: v.fluxLo, hi: v.fluxHi, nom: base.targetFlux },
      { key: 'T', label: 'Temperature (°C)', lo: Math.max(5, base.T - 8), hi: Math.min(42, base.T + 8), nom: base.T }, { key: 'salinityFactor', label: 'Salinity multiplier (×)', lo: 0.9, hi: 1.15, nom: 1 }].filter((d) => d.hi > d.lo);
    if (dims.length < 2) throw new Error('Widen the recovery and flux bounds: the design-of-experiments needs at least two varying inputs.');
    truth = (x) => { const e = evalDesign({ ...base, ...Object.fromEntries(dims.map((d, j) => [d.key, x[j]])) }, v); return e.ok ? tg.f(e) : NaN; };
    names = dims.map((d) => d.label); yName = tg.label; yUnit = tg.unit; lo = dims.map((d) => d.lo); hi = dims.map((d) => d.hi); nomX = dims.map((d) => clamp(d.nom, d.lo, d.hi));
    const U = lhs(clamp(Math.round(v.mlN), 20, 400), dims.length, seed);
    X = []; y = [];
    for (let i = 0; i < U.length; i++) {
      const x = U[i].map((q, j) => lo[j] + q * (hi[j] - lo[j])), t = truth(x);
      if (Number.isFinite(t)) { X.push(x); y.push(t); }
      if (i % 12 === 11) { ctx.progress(0.02 + (0.3 * i) / U.length, `Design of experiments: plant simulation ${i + 1} of ${U.length}`); await ctx.tick(); }
    }
  } else {
    const rows = numRows(v.mlData, ['y']), cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'].filter((k) => rows.length && rows.every((r) => isNum(r[k])) && new Set(rows.map((r) => r[k])).size > 1);
    if (!cols.length) throw new Error('The data table needs at least one fully numeric, non-constant input column (x1…x6) and a y column.');
    X = rows.map((r) => cols.map((k) => r[k])); y = rows.map((r) => r.y); names = cols; yName = 'y'; yUnit = '';
    lo = cols.map((_, j) => Math.min(...X.map((x) => x[j]))); hi = cols.map((_, j) => Math.max(...X.map((x) => x[j])));
  }
  const n = X.length, d = X[0]?.length || 0;
  if (n < Math.max(12, d + 8)) throw new Error(`Only ${n} usable data rows: at least ${Math.max(12, d + 8)} are needed to train and test the models.`);
  // chronology-free random split: train / validation (model selection, early stopping) / test (never used for fitting)
  const perm = shuffled(n, rng(seed + 1)), nTe = Math.max(3, Math.round((n * clamp(v.mlTest, 5, 40)) / 100)), nVa = Math.max(3, Math.round((n * clamp(v.mlVal, 5, 40)) / 100));
  const iTe = perm.slice(0, nTe), iVa = perm.slice(nTe, nTe + nVa), iTr = perm.slice(nTe + nVa), take = (idx, A) => idx.map((i) => A[i]);
  if (iTr.length < d + 4) throw new Error('Too few training rows after the split — reduce the validation/test shares or add data.');
  const Xtr = take(iTr, X), ytr = take(iTr, y), Xva = take(iVa, X), yva = take(iVa, y), Xte = take(iTe, X), yte = take(iTe, y);
  const hidden = [Math.round(v.nnH1), ...(v.nnLayers >= 2 ? [Math.round(v.nnH2)] : [])].map((h) => clamp(h, 2, 32)), nnOpt = { hidden, epochs: clamp(Math.round(v.nnEpochs), 20, 5000), lr: v.nnLr, seed };
  ctx.progress(0.36, 'Training response surface, Gaussian process and neural network'); await ctx.tick();
  let theta0 = null; // hyper-parameters of the main fit warm-start the cross-validation and learning-curve fits
  const fitAll = (Xa, ya, Xv, yv, fixedEpochs) => {
    const poly = polyFit(Xa, ya), gp = gpFit(Xa, ya, fixedEpochs ? { theta0: theta0, maxIter: 60 } : {}), nn = nnTrain(Xa, ya, fixedEpochs ? { ...nnOpt, epochs: fixedEpochs } : { ...nnOpt, Xval: Xv, yval: yv, patience: Math.round(v.nnPatience) });
    return { poly, gp, nn, pred: [(x) => poly.predict(x), (x) => gp.predict(x).mean, (x) => nn.predict(x)] };
  };
  const M = fitAll(Xtr, ytr, Xva, yva);
  theta0 = M.gp.theta;
  const MN = ['Polynomial response surface', 'Gaussian process', 'Neural network'], base0 = mean(ytr);
  const score = (f, Xs, ys) => { const m = metrics(ys, Xs.map(f)); return { rmse: m.rmse, mae: m.mae, r2: finite(m.r2), bias: m.bias }; };
  const val = M.pred.map((f) => score(f, Xva, yva)), test = M.pred.map((f) => score(f, Xte, yte)), bl = score(() => base0, Xte, yte);
  const sel = range(3).reduce((a, b) => (val[b].rmse < val[a].rmse ? b : a));
  // k-fold cross-validation on train + validation rows (network trained for the early-stopped number of epochs)
  ctx.progress(0.55, 'k-fold cross-validation'); await ctx.tick();
  const iCV = [...iTr, ...iVa], K = clamp(Math.round(v.mlK), 2, 10), cv = [[], [], []], ep = Math.max(40, M.nn.bestEpoch + 1);
  for (let f = 0; f < K; f++) {
    const te = iCV.filter((_, q) => q % K === f), tr = iCV.filter((_, q) => q % K !== f);
    const m = fitAll(take(tr, X), take(tr, y), null, null, ep);
    m.pred.forEach((p, q) => cv[q].push(rmse(take(te, X).map(p), take(te, y))));
    await ctx.tick();
  }
  // learning curve: test error versus number of training rows
  ctx.progress(0.75, 'Learning curve'); await ctx.tick();
  const lc = { n: [], e: [[], [], []] };
  for (const fr of [0.35, 0.55, 0.78, 1]) {
    const k = Math.max(d + 4, Math.round(fr * iTr.length));
    if (lc.n.includes(k)) continue;
    const m = fr === 1 ? M : fitAll(Xtr.slice(0, k), ytr.slice(0, k), null, null, ep);
    lc.n.push(k); m.pred.forEach((p, q) => lc.e[q].push(rmse(Xte.map(p), yte)));
  }
  const gpT = Xte.map((x) => M.gp.predict(x)), cover = gpT.filter((p, i) => Math.abs(yte[i] - p.mean) <= 2 * p.sdObs).length / nTe;
  // one-dimensional slice through the first input with the others at their reference value
  const ref = nomX || range(d).map((j) => mean(Xtr.map((x) => x[j]))), sx = linspace(lo[0], hi[0], 31), sl = sx.map((q) => { const x = [...ref]; x[0] = q; return x; }), gs = sl.map((x) => M.gp.predict(x));
  const sTruth = truth ? linspace(lo[0], hi[0], 7).map((q) => { const x = [...ref]; x[0] = q; return [q, truth(x)]; }).filter((p) => Number.isFinite(p[1])) : [];
  // Bayesian optimisation with expected improvement
  let bo = null;
  if (v.mlBO) {
    ctx.progress(0.85, 'Bayesian optimisation (expected improvement)'); await ctx.tick();
    const sgn = v.boGoal === 'max' ? -1 : 1, nb = truth ? Math.min(2, d) : d, corners = nb <= 6 ? range(2 ** nb).map((c) => range(nb).map((j) => (c >> j) & 1)) : [], cand = [...lhs(400, nb, seed + 5), ...corners].map((u) => { const x = [...ref]; for (let j = 0; j < nb; j++) x[j] = lo[j] + u[j] * (hi[j] - lo[j]); return x; });
    let Xb = [...X], yb = y.map((q) => sgn * q), gpb = gpFit(Xb, yb, { theta: M.gp.theta });
    if (truth) {
      const used = new Set(), trace = [], pts = [];
      let best = Infinity, bx = null;
      for (let it = 0; it <= clamp(Math.round(v.boIter), 1, 40); it++) {
        let pick = -1, bestA = -Infinity;
        cand.forEach((x, i) => { if (used.has(i)) return; const p = gpb.predict(x), a = it === 0 ? -p.mean : expectedImprovement(p.mean, p.sd, best); if (a > bestA) { bestA = a; pick = i; } });
        if (pick < 0) break;
        used.add(pick);
        const x = cand[pick], t = truth(x);
        if (!Number.isFinite(t)) continue;
        if (sgn * t < best) { best = sgn * t; bx = x; }
        Xb.push(x); yb.push(sgn * t); gpb = gpFit(Xb, yb, { theta: M.gp.theta });
        trace.push(sgn * best); pts.push([it, t, it === 0 ? 0 : bestA]);
        await ctx.tick();
      }
      const tRef = truth(ref);
      if (bx) bo = { kind: 'loop', x: bx, y: sgn * best, trace, pts, refY: finite(tRef, sgn * best), nb };
    } else {
      const best = Math.min(...yb);
      let pick = 0, bestA = -Infinity;
      cand.forEach((x, i) => { const p = gpb.predict(x), a = expectedImprovement(p.mean, p.sd, best); if (a > bestA) { bestA = a; pick = i; } });
      const p = gpb.predict(cand[pick]);
      bo = { kind: 'suggest', x: cand[pick], mean: sgn * p.mean, sd: p.sd, ei: bestA, best: sgn * best, nb };
    }
  }
  const skill = bl.rmse > 0 ? 100 * (1 - test[sel].rmse / bl.rmse) : 0, W = [];
  if (test[sel].r2 < 0.9) W.push({ level: 'warn', msg: `The selected model explains only ${fmt(100 * test[sel].r2, 3)} % of the test variance — add data, inputs or noise handling before relying on it.` });
  if (test[sel].rmse > 1.6 * val[sel].rmse && nTe >= 8) W.push({ level: 'warn', msg: 'Test error is clearly above validation error: the model selection may have over-fitted the small validation set.' });
  if (cover < 0.8) W.push({ level: 'warn', msg: `Only ${fmt(100 * cover, 3)} % of test points fall inside the Gaussian-process ±2σ band (≈95 % expected): its uncertainty is over-confident here.` });
  if (M.nn.stoppedEarly) W.push({ level: 'info', msg: `Neural-network training stopped early at epoch ${M.nn.epochs} (best validation loss at epoch ${M.nn.bestEpoch}).` });
  W.push({ level: 'info', msg: `Split of ${n} rows: ${iTr.length} training, ${nVa} validation (early stopping and model selection), ${nTe} test (never used for fitting). ${fromRO ? 'Data come from the mechanistic RO model, so the surrogate reproduces that model — not plant reality.' : 'Data come from the imported table.'}` });
  const lim = [Math.min(...yte, ...Xte.map(M.pred[sel])), Math.max(...yte, ...Xte.map(M.pred[sel]))];
  return {
    summary: `${MN[sel]} is selected on validation data and predicts ${lcFirst(yName)} on the untouched test set with RMSE ${fmt(test[sel].rmse, 3)}${yUnit ? ' ' + yUnit : ''} (R² ${fmt(test[sel].r2, 4)}), ${fmt(skill, 3)} % better than the mean baseline (${fmt(bl.rmse, 3)}).${bo?.kind === 'loop' ? ` Bayesian optimisation ${v.boGoal === 'max' ? 'raised' : 'lowered'} it to ${fmt(bo.y, 4)} ${yUnit} (reference point ${fmt(bo.refY, 4)}) in ${bo.trace.length} plant evaluations.` : ''}`,
    warnings: W,
    kpis: [
      { label: 'Selected model', value: MN[sel] }, { label: 'Test RMSE (selected)', value: test[sel].rmse, unit: yUnit }, { label: 'Test R² (selected)', value: test[sel].r2, unit: '–', status: test[sel].r2 > 0.9 ? 'ok' : 'warn' },
      { label: 'Test MAE (selected)', value: test[sel].mae, unit: yUnit }, { label: 'Mean-baseline RMSE', value: bl.rmse, unit: yUnit }, { label: 'Skill over baseline', value: skill, unit: '%', status: skill > 50 ? 'ok' : 'warn' },
      { label: 'Response-surface test RMSE', value: test[0].rmse, unit: yUnit }, { label: 'Gaussian-process test RMSE', value: test[1].rmse, unit: yUnit }, { label: 'Neural-network test RMSE', value: test[2].rmse, unit: yUnit },
      { label: `${K}-fold CV RMSE (selected)`, value: mean(cv[sel]), unit: yUnit }, { label: 'GP ±2σ coverage on test', value: 100 * cover, unit: '%', status: cover >= 0.8 ? 'ok' : 'warn' }, { label: 'Rows train / val / test', value: `${iTr.length} / ${nVa} / ${nTe}` },
      ...(bo?.kind === 'loop' ? [{ label: `Bayesian-optimised ${lcFirst(yName)}`, value: bo.y, unit: yUnit }, { label: 'Change versus reference point', value: bo.refY ? (100 * (bo.y - bo.refY)) / Math.abs(bo.refY) : 0, unit: '%' }] : []),
    ],
    recommendations: [
      `Use the selected model (${MN[sel]}) for fast what-if studies inside the sampled ranges only; surrogates do not extrapolate.`,
      test[1].rmse < test[0].rmse * 0.8 ? 'The response is clearly non-quadratic: prefer the Gaussian process or the network over the polynomial.' : 'A quadratic response surface is already adequate — the simplest model is the safest choice.',
      bo?.kind === 'suggest' ? `Next experiment suggested by expected improvement: ${names.map((nm, j) => `${nm} = ${fmt(bo.x[j], 4)}`).join(', ')}.` : 'Add points where the ±2σ band is widest to improve the surrogate most efficiently.',
      'Re-train whenever the plant model or its calibration changes, and keep the test rows untouched for an honest error estimate.',
    ],
    plots: [
      { type: 'line', title: 'Parity plot on the test set', xlabel: `Actual ${yName} (${yUnit})`, ylabel: `Predicted (${yUnit})`, series: [...MN.map((nm, q) => ({ name: nm, x: yte, y: Xte.map(M.pred[q]), mode: 'points' })), { name: 'Perfect agreement', x: lim, y: lim, dash: true }] },
      { type: 'line', title: 'Test residuals', xlabel: `Predicted (${yUnit})`, ylabel: 'Predicted − actual', series: MN.map((nm, q) => { const p = Xte.map(M.pred[q]); return { name: nm, x: p, y: p.map((u, i) => u - yte[i]), mode: 'points' }; }), hlines: [{ y: 0, label: 'zero' }] },
      { type: 'line', title: 'Neural-network training history', xlabel: 'Epoch', ylabel: 'Standardised mean-squared error', logy: true, series: [{ name: 'Training loss', x: M.nn.history.epoch, y: M.nn.history.train }, { name: 'Validation loss', x: M.nn.history.epoch, y: M.nn.history.val }], vlines: [{ x: M.nn.bestEpoch, label: 'best' }] },
      { type: 'line', title: 'Learning curve: test error versus training rows', xlabel: 'Training rows', ylabel: `Test RMSE (${yUnit})`, series: [...MN.map((nm, q) => ({ name: nm, x: lc.n, y: lc.e[q], mode: 'both' })), { name: 'Mean baseline', x: [lc.n[0], lc.n[lc.n.length - 1]], y: [bl.rmse, bl.rmse], dash: true }] },
      { type: 'line', title: `Slice along ${names[0]} (other inputs at reference)`, xlabel: names[0], ylabel: `${yName} (${yUnit})`, series: [{ name: 'GP mean', x: sx, y: gs.map((p) => p.mean) }, { name: 'GP mean − 2σ', x: sx, y: gs.map((p) => p.mean - 2 * p.sdObs), dash: true }, { name: 'GP mean + 2σ', x: sx, y: gs.map((p) => p.mean + 2 * p.sdObs), dash: true },
        { name: 'Response surface', x: sx, y: sl.map(M.pred[0]) }, { name: 'Neural network', x: sx, y: sl.map(M.pred[2]) }, ...(sTruth.length ? [{ name: 'Plant model', x: sTruth.map((p) => p[0]), y: sTruth.map((p) => p[1]), mode: 'points' }] : [])] },
      { type: 'bar', title: `${K}-fold cross-validation RMSE by fold`, ylabel: `RMSE (${yUnit})`, categories: range(K).map((f) => `Fold ${f + 1}`), series: MN.map((nm, q) => ({ name: nm, values: cv[q] })) },
      ...(bo?.kind === 'loop' ? [{ type: 'line', title: 'Bayesian optimisation: best value found', xlabel: 'Plant evaluation', ylabel: `${yName} (${yUnit})`, series: [{ name: 'Best so far', x: bo.pts.map((p) => p[0]), y: bo.trace, mode: 'step' }, { name: 'Evaluated point', x: bo.pts.map((p) => p[0]), y: bo.pts.map((p) => p[1]), mode: 'points' }], hlines: [{ y: bo.refY, label: 'reference point' }] }] : []),
    ],
    tables: [
      { title: 'Model comparison', columns: ['Model', 'Validation RMSE', 'Test RMSE', 'Test MAE', 'Test R²', 'Test bias', `${K}-fold CV RMSE (mean)`, 'CV RMSE (std)', 'Selected'],
        rows: [...MN.map((nm, q) => [nm, val[q].rmse, test[q].rmse, test[q].mae, test[q].r2, test[q].bias, mean(cv[q]), std(cv[q]), q === sel ? '◀' : '']), ['Mean of training data (baseline)', null, bl.rmse, bl.mae, bl.r2, bl.bias, null, null, '']],
        note: 'Models are fitted on the training rows only. The validation rows select the model and stop the network; the test rows are used once, for this table. Cross-validation uses the training + validation rows.' },
      { title: 'Model details', columns: ['Item', 'Value'], rows: [['Inputs', names.join(', ')], ['Response-surface terms', `${M.poly.nTerms} (${M.poly.mode})`], ['GP log marginal likelihood', M.gp.lml], ['GP length scales (standardised inputs)', M.gp.lengthScales.map((q) => fmt(q, 3)).join(', ')],
        ['GP signal / noise standard deviation', `${fmt(M.gp.signal, 3)} / ${fmt(M.gp.noise, 3)}${yUnit ? ' ' + yUnit : ''}`], ['Network architecture', `${d} – ${hidden.join(' – ')} – 1 (tanh), ${M.nn.nWeights} weights`], ['Network epochs run / best', `${M.nn.epochs} / ${M.nn.bestEpoch}`], ['Optimiser', `Adam, mini-batches of 8, learning rate ${fmt(v.nnLr)} decaying to 10 %, L2 1e-4, seed ${seed}`]] },
      ...(bo ? [{ title: bo.kind === 'loop' ? 'Bayesian optimisation result' : 'Suggested next experiment (maximum expected improvement)', columns: ['Quantity', 'Value'],
        rows: bo.kind === 'loop' ? [...names.map((nm, j) => [nm + (j >= bo.nb ? ' (held at reference)' : ''), bo.x[j]]), [`${yName} at optimum (${yUnit})`, bo.y], [`${yName} at reference point (${yUnit})`, bo.refY], ['Plant evaluations used', bo.trace.length]]
          : [...names.map((nm, j) => [nm, bo.x[j]]), ['Predicted mean', bo.mean], ['Predicted standard deviation', bo.sd], ['Expected improvement', bo.ei], ['Best observed so far', bo.best]],
        note: `Goal: ${v.boGoal === 'max' ? 'maximise' : 'minimise'} ${lcFirst(yName)}. No operating constraints are applied here — use the constrained optimisation task for a design decision.` }] : []),
    ],
    outputs: { task: 'ml', objective: test[sel].rmse, surrogate: { target: yName, model: MN[sel], testRMSE: test[sel].rmse, testR2: test[sel].r2, baselineRMSE: bl.rmse }, ...(bo?.kind === 'loop' && fromRO ? { best: { recovery: bo.x[0], targetFlux: bo.x[1] } } : {}), study: { a: test[sel].rmse, b: test[sel].r2 } },
  };
}

// ---- Task 6: physics-informed neural network for the polarisation film equation ----------------------------------------
async function taskPINN(v, ctx) {
  const Jw = v.pinnJw / 3.6e6, k = v.pinnK * 1e-6, D = v.pinnD * 1e-9, Pe = Jw / k, delta = (D / k) * 1e6, cb = v.pinnCb, cp = v.pinnCp;
  if (!(Pe > 0)) throw new Error('Flux and mass-transfer coefficient must be positive.');
  const opt = { neurons: v.pinnNeurons, iters: clamp(Math.round(v.pinnIters), 200, 40000), lr: v.pinnLr, nColl: v.pinnColl, wBC: v.pinnWbc }, seed = Math.round(v.seed);
  const xi = linspace(0, 1, 101), exact = xi.map((x) => Math.exp(Pe * x)), runs = [];
  for (let s = 0; s < 3; s++) {
    ctx.progress(0.05 + 0.28 * s, `Training network ${s + 1} of 3 (seed ${seed + s})`); await ctx.tick();
    const r = pinnTrain(Pe, { ...opt, seed: seed + s }), th = xi.map(r.predict), err = th.map((q, i) => Math.abs(q - exact[i]) / exact[i]);
    runs.push({ r, th, err, maxErr: Math.max(...err), rms: rmse(th, exact), wall: th[100], seed: seed + s });
  }
  const best = runs.reduce((a, b) => (b.r.loss < a.r.loss ? b : a)); // chosen by training loss, never by the error against the known solution
  // second-order finite-difference (trapezoidal) solution of the same ODE
  const nF = clamp(Math.round(v.pinnFD), 4, 2000), xf = linspace(0, 1, nF + 1), h = 1 / nF, fd = [1];
  for (let i = 0; i < nF; i++) fd.push((fd[i] * (1 + (Pe * h) / 2)) / (1 - (Pe * h) / 2));
  const fdErr = fd.map((q, i) => Math.abs(q - Math.exp(Pe * xf[i])) / Math.exp(Pe * xf[i])), eps = 1e-4;
  const resid = xi.map((x) => (best.r.predict(Math.min(1, x + eps)) - best.r.predict(Math.max(0, x - eps))) / (Math.min(1, x + eps) - Math.max(0, x - eps)) - Pe * best.r.predict(x));
  const conc = (th) => th.map((q) => cp + (cb - cp) * q), ymu = xi.map((x) => x * delta), cpf = Math.exp(Pe), W = [];
  if (best.maxErr > 0.02) W.push({ level: 'warn', msg: `The network misses the analytical profile by up to ${fmt(100 * best.maxErr, 3)} % — train longer, add neurons or collocation points. A finite-difference solve of this problem is exact to ${fmt(100 * Math.max(...fdErr), 2)} % at negligible cost.` });
  else W.push({ level: 'info', msg: `The network reproduces the analytical solution within ${fmt(100 * best.maxErr, 2)} % everywhere, without ever being shown that solution.` });
  if (Pe > 2.5) W.push({ level: 'warn', msg: `Film Péclet number ${fmt(Pe, 3)} is very high (polarisation factor ${fmt(cpf, 3)}): the profile is steep and small networks train slowly on it.` });
  W.push({ level: 'info', msg: 'Scope: a single-hidden-layer network on a one-dimensional steady problem with a known solution. It demonstrates and verifies the method; it is not a substitute for the mechanistic solvers of the other suites.' });
  return {
    summary: `A physics-informed network with ${Math.round(opt.neurons)} hidden neurons, trained on the film equation alone, gives a polarisation factor of ${fmt(best.wall, 5)} against the analytical exp(Jw/k) = ${fmt(cpf, 5)} (maximum profile error ${fmt(100 * best.maxErr, 2)} %); the ${nF}-interval finite-difference solution gives ${fmt(fd[nF], 5)}.`,
    warnings: W,
    kpis: [
      { label: 'Film Péclet number Jw/k', value: Pe, unit: '–' }, { label: 'Polarisation factor, analytical', value: cpf, unit: '–' }, { label: 'Polarisation factor, network', value: best.wall, unit: '–', status: Math.abs(best.wall - cpf) / cpf < 0.02 ? 'ok' : 'warn' },
      { label: 'Polarisation factor, finite difference', value: fd[nF], unit: '–' }, { label: 'Network maximum error', value: 100 * best.maxErr, unit: '%', status: best.maxErr < 0.02 ? 'ok' : 'warn' }, { label: 'Network RMS error', value: best.rms, unit: '–' },
      { label: 'Finite-difference maximum error', value: 100 * Math.max(...fdErr), unit: '%' }, { label: 'Final training loss', value: best.r.loss, unit: '–' }, { label: 'Wall concentration (network)', value: cp + (cb - cp) * best.wall, unit: 'g/L' },
      { label: 'Wall concentration (analytical)', value: cp + (cb - cp) * cpf, unit: 'g/L' }, { label: 'Boundary-layer thickness D/k', value: delta, unit: 'µm' }, { label: 'Trainable parameters', value: best.r.nParams },
    ],
    recommendations: [
      'Use the seed table to judge robustness: a trustworthy physics-informed solution should not depend on the random initialisation.',
      'For design work keep the analytical film model of suite 1 or the CFD suite; reserve physics-informed networks for inverse problems where data and equations must be combined.',
      best.maxErr > 0.005 ? 'Increase the iterations or the boundary-loss weight to tighten the solution.' : 'Accuracy is already at the level of the finite-difference reference.',
    ],
    plots: [
      { type: 'line', title: 'Concentration profile across the polarisation layer', xlabel: 'Distance from the bulk toward the membrane (µm)', ylabel: 'Concentration (g/L)', series: [{ name: 'Analytical', x: ymu, y: conc(exact) }, { name: 'Physics-informed network', x: ymu, y: conc(best.th), dash: true }, { name: 'Finite difference', x: xf.map((x) => x * delta), y: conc(fd), mode: 'points' }] },
      { type: 'line', title: 'Relative error against the analytical solution', xlabel: 'Dimensionless distance ξ', ylabel: 'Relative error (–)', logy: true, series: [{ name: 'Network', x: xi, y: best.err.map((e) => Math.max(e, 1e-12)) }, { name: 'Finite difference', x: xf, y: fdErr.map((e) => Math.max(e, 1e-12)), mode: 'both' }] },
      { type: 'line', title: 'Training loss (equation residual + boundary term)', xlabel: 'Adam iteration', ylabel: 'Loss', logy: true, series: runs.map((q) => ({ name: `Seed ${q.seed}`, x: q.r.history.it, y: q.r.history.loss.map((l) => Math.max(l, 1e-16)) })) },
      { type: 'line', title: 'Equation residual of the trained network between collocation points', xlabel: 'Dimensionless distance ξ', ylabel: 'dθ/dξ − Pe·θ', series: [{ name: 'Residual', x: xi, y: resid }], hlines: [{ y: 0, label: 'exact' }] },
    ],
    tables: [
      { title: 'Robustness to the random initialisation', columns: ['Seed', 'Final loss', 'Polarisation factor', 'Error of polarisation factor (%)', 'Maximum profile error (%)', 'RMS error', 'Used'], rows: runs.map((q) => [q.seed, q.r.loss, q.wall, (100 * Math.abs(q.wall - cpf)) / cpf, 100 * q.maxErr, q.rms, q === best ? '◀ lowest loss' : '']) },
      { title: 'Method comparison', columns: ['Method', 'Unknowns', 'Polarisation factor', 'Maximum relative error (%)'], rows: [['Analytical θ = exp(Pe·ξ)', 0, cpf, 0], [`Network, ${opt.iters} Adam iterations`, best.r.nParams, best.wall, 100 * best.maxErr], [`Finite difference (trapezoidal), ${nF} intervals`, nF, fd[nF], 100 * Math.max(...fdErr)]],
        note: `Equation: Jw·c − D·dc/dy = Jw·cp, written as dθ/dξ = Pe·θ with θ = (c − cp)/(cb − cp), ξ = y/δ, θ(0) = 1. Loss = mean squared residual at ${Math.round(opt.nColl)} collocation points + ${fmt(opt.wBC)} × boundary error².` },
    ],
    outputs: { task: 'pinn', objective: best.maxErr, cpFactor: best.wall, cpFactorExact: cpf, study: { a: best.wall, b: best.maxErr } },
  };
}

// ---- Task 7: forecasting, anomaly detection and state estimation ----------------------------------------------------
/** Built-in operating record: normalised permeate flow over 180 days with fouling trend, weekly pattern, noise and three faults. */
function tsSampleTable() {
  const g = rng(42);
  return range(180).map((t) => {
    let y = 0.7 + 0.3 * Math.exp(-0.004 * t) + 0.008 * Math.sin((2 * Math.PI * t) / 7) + 0.004 * Math.cos((4 * Math.PI * t) / 7) + g.normal(0, 0.006);
    if (t >= 60 && t <= 62) y -= 0.04; if (t === 110) y += 0.05; if (t >= 150 && t <= 153) y -= 0.035;
    return { t, y: +y.toFixed(4) };
  });
}
async function taskTS(v, ctx) {
  const rows = numRows(v.tsData, ['t', 'y']).sort((a, b) => a.t - b.t).filter((r, i, a) => i === 0 || r.t > a[i - 1].t), n = rows.length;
  if (n < 30) throw new Error(`The operating-data table has ${n} usable rows; at least 30 with increasing time are needed.`);
  const t = rows.map((r) => r.t), y = rows.map((r) => r.y), dts = t.slice(1).map((q, i) => q - t[i]), dt = quantile(dts, 0.5);
  const nTr = clamp(Math.round((n * v.tsTrain) / 100), 20, n - 5), nTe = n - nTr, ytr = y.slice(0, nTr), yte = y.slice(nTr), tte = t.slice(nTr);
  const m = Math.round(v.tsSeason), p = clamp(Math.round(v.tsArP), 1, 12), dd = v.tsArD === 'd1' ? 1 : 0, H = clamp(Math.round(v.tsHorizon), 1, nTe);
  ctx.progress(0.1, 'Fitting Holt–Winters and autoregressive models'); await ctx.tick();
  const hw = hwFit(ytr, m), ar = arFit(ytr, p, dd);
  const ekfOpt = { Kinf: v.ekfKinf, r0: v.ekfRate, sdMeas: v.ekfMeasSd, gate: v.tsGate, qK: (0.15 * v.ekfMeasSd) ** 2, qr: (0.02 * Math.max(v.ekfRate, 1e-4)) ** 2, P0: [(2 * v.ekfMeasSd) ** 2, (0.5 * Math.max(v.ekfRate, 1e-4)) ** 2] };
  const kfTr = ekfFouling(t.slice(0, nTr), ytr, ekfOpt), kf = ekfFouling(t, y, ekfOpt);
  // single-origin forecast over the whole test period
  const fHW = hw.model.forecast(nTe), sHW = hw.sdAhead(nTe), fAR = ar.forecast(ytr, nTe), sAR = ar.sdAhead(nTe), fK = kfTr.forecast(nTe, dt);
  const sc = (f) => { const q = metrics(yte, f); return [q.rmse, q.mae]; }, cov = (f, s) => (100 * yte.filter((q, i) => Math.abs(q - f[i]) <= 1.96 * s[i]).length) / nTe;
  // one-step-ahead predictions through the test period (parameters frozen at the training fit)
  const hwAll = holtWinters(y, m, hw.par).fit, one = { hw: hwAll.slice(nTr), ar: range(nTe).map((i) => ar.forecast(y.slice(0, nTr + i), 1)[0]), kf: kf.pred.slice(nTr), naive: y.slice(nTr - 1, n - 1) };
  // rolling-origin evaluation of the H-step forecast
  ctx.progress(0.45, 'Rolling-origin evaluation'); await ctx.tick();
  const lead = { hw: zeros(H), ar: zeros(H), kf: zeros(H), naive: zeros(H) }, sq = { hw: 0, ar: 0, kf: 0, naive: 0 }, ab = { hw: 0, ar: 0, kf: 0, naive: 0 };
  let nOr = 0;
  const stepO = Math.max(1, Math.floor((n - H - nTr) / 14));
  for (let o = nTr; o + H <= n; o += stepO) {
    const hist = y.slice(0, o), f = { hw: holtWinters(hist, m, hw.par).forecast(H), ar: ar.forecast(hist, H), kf: ekfFouling(t.slice(0, o), hist, ekfOpt).forecast(H, dt).mean, naive: new Array(H).fill(y[o - 1]) };
    for (const key of Object.keys(f)) { for (let j = 0; j < H; j++) lead[key][j] += Math.abs(f[key][j] - y[o + j]); const e = f[key][H - 1] - y[o + H - 1]; sq[key] += e * e; ab[key] += Math.abs(e); }
    nOr++;
  }
  const MOD = [['hw', `Holt–Winters${hw.model.seasonal ? ` (season ${m})` : ' (no season)'}`], ['ar', `AR(${p})${dd ? ' on differences' : ''}`], ['kf', 'Kalman fouling state model'], ['naive', 'Persistence (baseline)']];
  const hRmse = Object.fromEntries(MOD.map(([k]) => [k, nOr ? Math.sqrt(sq[k] / nOr) : 0])), bestK = MOD.slice(0, 3).reduce((a, b) => (hRmse[b[0]] < hRmse[a[0]] ? b : a));
  // anomaly detection: Kalman innovation gate and an EWMA chart of Holt–Winters one-step residuals
  const res = y.map((q, i) => q - hwAll[i]), r0 = hw.resid.slice(hw.start), med = quantile(r0, 0.5), sdR = 1.4826 * quantile(r0.map((q) => Math.abs(q - med)), 0.5) || hw.sigma || 1e-6;
  const lam = clamp(v.tsEwma, 0.05, 1), ucl = 3 * sdR * Math.sqrt(lam / (2 - lam)), ew = [];
  let z = 0;
  res.forEach((q, i) => { z = i < hw.start ? 0 : lam * q + (1 - lam) * z; ew.push(z); });
  const flagK = range(n).filter((i) => i > 2 && kf.flag[i]), flagE = range(n).filter((i) => Math.abs(ew[i]) > ucl), flagged = [...new Set([...flagK, ...flagE])].sort((a, b) => a - b);
  // remaining time to the cleaning threshold from the posterior of the final state
  ctx.progress(0.8, 'Remaining time to threshold'); await ctx.tick();
  const [Kn, rn] = kf.x, thr = v.tsThreshold, CAP = 3650, g = rng(Math.round(v.seed)), L = cholesky([[kf.P[0][0] + 1e-18, kf.P[0][1]], [kf.P[1][0], kf.P[1][1] + 1e-24]]) || [[Math.sqrt(kf.P[0][0]), 0], [0, Math.sqrt(kf.P[1][1])]];
  const ttt = (K, r) => (K <= thr ? 0 : thr <= v.ekfKinf || r <= 1e-9 ? CAP : Math.min(CAP, Math.log((K - v.ekfKinf) / (thr - v.ekfKinf)) / r));
  const rulS = range(400).map(() => { const a = g.normal(), b = g.normal(); return ttt(Kn + L[0][0] * a, rn + L[1][0] * a + L[1][1] * b); }), rul = ttt(Kn, rn), r10 = quantile(rulS, 0.1), r50 = quantile(rulS, 0.5), r90 = quantile(rulS, 0.9);
  const hF = Math.max(H, Math.min(Math.round((1.3 * Math.min(r90, 3 * (t[n - 1] - t[0]))) / dt), 400)), fut = kf.forecast(hF, dt), tf = range(hF).map((i) => t[n - 1] + (i + 1) * dt);
  const W = [];
  if (flagged.length) W.push({ level: 'warn', msg: `${flagged.length} observation${flagged.length > 1 ? 's are' : ' is'} flagged as anomalous (first at t = ${fmt(t[flagged[0]])}, last at t = ${fmt(t[flagged[flagged.length - 1]])}). Check instruments and operating events at those times.` });
  else W.push({ level: 'info', msg: 'No anomaly was flagged by the innovation gate or the EWMA chart.' });
  if (rul >= CAP) W.push({ level: 'info', msg: `With the estimated state the series does not reach the threshold of ${fmt(thr)} (asymptote ${fmt(v.ekfKinf)}); the remaining time is reported at the ${CAP}-unit cap.` });
  else if (rul < 3 * H * dt) W.push({ level: rul <= 0 ? 'bad' : 'warn', msg: rul <= 0 ? `The estimated state ${fmt(Kn, 4)} is already below the threshold ${fmt(thr)} — cleaning is due.` : `The threshold of ${fmt(thr)} is expected in about ${fmt(r50, 3)} time units (P10–P90: ${fmt(r10, 3)}–${fmt(r90, 3)}): schedule the cleaning.` });
  if (hRmse[bestK[0]] > hRmse.naive) W.push({ level: 'warn', msg: `No model beats simple persistence at the ${H}-step horizon; the series is close to a random walk at this horizon or needs more history.` });
  if (!hw.model.seasonal && m >= 2) W.push({ level: 'info', msg: 'The training record is shorter than two seasons, so Holt–Winters ran without a seasonal component.' });
  const band = (f, s, sign) => f.map((q, i) => q + sign * 1.96 * s[i]);
  return {
    summary: `${bestK[1]} gives the best ${H}-step forecast (rolling-origin RMSE ${fmt(hRmse[bestK[0]], 3)} against ${fmt(hRmse.naive, 3)} for persistence). The state estimator puts the current normalised permeability at ${fmt(Kn, 4)} with a fouling rate of ${fmt(100 * rn * Math.max(0, Kn - v.ekfKinf), 3)} %/time unit; ${rul >= CAP ? 'the threshold is not reached' : `the threshold ${fmt(thr)} is reached in about ${fmt(r50, 3)} time units`}. ${flagged.length} anomalies flagged.`,
    warnings: W,
    kpis: [
      { label: `Best ${H}-step model`, value: bestK[1] }, { label: `${H}-step RMSE (best)`, value: hRmse[bestK[0]] }, { label: `${H}-step RMSE (persistence)`, value: hRmse.naive },
      { label: 'Skill over persistence', value: hRmse.naive > 0 ? 100 * (1 - hRmse[bestK[0]] / hRmse.naive) : 0, unit: '%', status: hRmse[bestK[0]] < hRmse.naive ? 'ok' : 'warn' }, { label: 'Estimated permeability now', value: Kn, unit: '–', status: Kn <= thr ? 'bad' : 'ok' }, { label: 'Estimated decline rate', value: 100 * rn * Math.max(0, Kn - v.ekfKinf), unit: '%/time unit' },
      { label: 'Time to threshold (P50)', value: r50, unit: 'time units', status: r50 < 3 * H * dt ? 'warn' : 'ok' }, { label: 'Time to threshold (P10–P90)', value: `${fmt(r10, 3)} – ${fmt(r90, 3)}`, unit: 'time units' }, { label: 'Anomalies flagged', value: flagged.length, status: flagged.length ? 'warn' : 'ok' },
      { label: '95 % interval coverage (Holt–Winters, test)', value: cov(fHW, sHW), unit: '%' }, { label: 'Rows train / test', value: `${nTr} / ${nTe}` }, { label: 'Rolling origins', value: nOr },
    ],
    recommendations: [
      rul < CAP ? `Plan the next cleaning around t = ${fmt(t[n - 1] + r50, 4)} (earliest ${fmt(t[n - 1] + r10, 4)}); send the fouling rate to suite 10 for the cleaning-cost optimum.` : 'Decline is levelling off above the threshold: no cleaning is forecast from permeability alone — watch pressure drop and salt passage too.',
      flagged.length ? 'Review the flagged periods before re-training: confirmed instrument faults should be removed from the record, real process upsets kept.' : 'Keep the innovation gate running online: it reacts within one sample to a step fault.',
      'Re-fit the models whenever the membranes are cleaned or replaced — a cleaning resets the state and invalidates the trend.',
    ],
    plots: [
      { type: 'line', title: 'Forecast over the held-out test period', xlabel: 'Time', ylabel: 'Normalised signal', series: [{ name: 'Training data', x: t.slice(0, nTr), y: ytr, mode: 'points' }, { name: 'Test data', x: tte, y: yte, mode: 'points' }, { name: 'Holt–Winters', x: tte, y: fHW }, { name: 'Holt–Winters 95 % low', x: tte, y: band(fHW, sHW, -1), dash: true }, { name: 'Holt–Winters 95 % high', x: tte, y: band(fHW, sHW, 1), dash: true }, { name: 'Autoregression', x: tte, y: fAR }, { name: 'Kalman state model', x: tte, y: fK.mean }], vlines: [{ x: t[nTr], label: 'forecast origin' }] },
      { type: 'line', title: 'State estimate and projection to the cleaning threshold', xlabel: 'Time', ylabel: 'Normalised permeability', series: [{ name: 'Measurements', x: t, y, mode: 'points' }, { name: 'Filtered state', x: t, y: kf.K }, { name: 'Projection', x: tf, y: fut.mean, dash: true }, { name: 'Projection − 2σ', x: tf, y: fut.mean.map((q, i) => q - 2 * fut.sd[i]), dash: true }, { name: 'Projection + 2σ', x: tf, y: fut.mean.map((q, i) => q + 2 * fut.sd[i]), dash: true }], hlines: [{ y: thr, label: 'cleaning threshold' }] },
      { type: 'line', title: 'Estimated fouling-rate constant', xlabel: 'Time', ylabel: 'r (1/time unit)', series: [{ name: 'Extended Kalman filter estimate', x: t, y: kf.rate }] },
      { type: 'line', title: 'Standardised innovations of the state estimator', xlabel: 'Time', ylabel: 'Innovation / σ', series: [{ name: 'Innovation', x: t, y: kf.z, mode: 'both' }, { name: 'Flagged', x: flagK.map((i) => t[i]), y: flagK.map((i) => kf.z[i]), mode: 'points' }], hlines: [{ y: v.tsGate, label: 'gate' }, { y: -v.tsGate, label: 'gate' }] },
      { type: 'line', title: 'EWMA control chart of one-step residuals', xlabel: 'Time', ylabel: 'EWMA of residual', series: [{ name: 'EWMA', x: t, y: ew }, { name: 'Out of control', x: flagE.map((i) => t[i]), y: flagE.map((i) => ew[i]), mode: 'points' }], hlines: [{ y: ucl, label: 'upper limit' }, { y: -ucl, label: 'lower limit' }] },
      { type: 'line', title: 'Rolling-origin error by lead time', xlabel: 'Lead time (steps)', ylabel: 'Mean absolute error', series: MOD.map(([k, nm]) => ({ name: nm, x: range(H).map((j) => j + 1), y: lead[k].map((q) => (nOr ? q / nOr : 0)), mode: 'both', dash: k === 'naive' })) },
    ],
    tables: [
      { title: 'Forecast accuracy on unseen data', columns: ['Model', 'One-step RMSE', 'One-step MAE', `${H}-step RMSE (rolling origin)`, `${H}-step MAE (rolling origin)`, 'Whole-test RMSE (single origin)', '95 % interval coverage (%)'],
        rows: [['hw', fHW, sHW], ['ar', fAR, sAR], ['kf', fK.mean, fK.sd], ['naive', new Array(nTe).fill(ytr[nTr - 1]), null]].map(([k, f, s]) => [MOD.find((q) => q[0] === k)[1], ...sc(one[k]), hRmse[k], nOr ? ab[k] / nOr : 0, sc(f)[0], s ? cov(f, s) : null]),
        note: `Chronological split: first ${nTr} rows for fitting, last ${nTe} for testing; ${nOr} forecast origins roll through the test period with parameters frozen at the training fit.` },
      { title: 'Flagged anomalies', columns: ['Time', 'Value', 'Innovation / σ', 'EWMA / limit', 'Detector', 'Period'], rows: flagged.slice(0, 80).map((i) => [t[i], y[i], kf.z[i], ew[i] / ucl, [flagK.includes(i) ? 'innovation gate' : null, flagE.includes(i) ? 'EWMA' : null].filter(Boolean).join(' + '), i < nTr ? 'training' : 'test']), note: flagged.length ? `Innovation gate ±${fmt(v.tsGate)}σ; EWMA weight ${fmt(lam)}, limits ±3σ·√(λ/(2−λ)) with a robust σ from the training residuals.` : 'None.' },
      { title: 'Fitted model parameters', columns: ['Model', 'Parameter', 'Value'], rows: [['Holt–Winters', 'level α', hw.par.alpha], ['Holt–Winters', 'trend β', hw.par.beta], ['Holt–Winters', 'season γ', hw.par.gamma], ['Holt–Winters', 'damping φ', hw.par.phi], ['Holt–Winters', 'residual σ', hw.sigma],
        ['Autoregression', 'intercept', ar.c], ...ar.phi.map((q, i) => ['Autoregression', `φ${i + 1}`, q]), ['Autoregression', 'residual σ', ar.sigma], ['Kalman state model', 'permeability K', Kn], ['Kalman state model', 'rate constant r', rn], ['Kalman state model', 'asymptote K∞ (input)', v.ekfKinf], ['Kalman state model', 'σ of K', Math.sqrt(Math.max(0, kf.P[0][0]))], ['Kalman state model', 'σ of r', Math.sqrt(Math.max(0, kf.P[1][1]))]] },
    ],
    outputs: { task: 'ts', objective: hRmse[bestK[0]], foulingRate: 100 * rn * Math.max(0, Kn - v.ekfKinf), normPermeability: Kn, daysToThreshold: r50, anomalies: flagged.length, study: { a: hRmse[bestK[0]], b: Kn } },
  };
}

// ---- Task 8: model-predictive control and reinforcement learning ------------------------------------------------------
/** Discrete linear model of an RO train feeding a product tank: states [flux deviation (L/m²·h), level deviation (m)], input = feed-pressure deviation (bar). */
function roTrainModel(v) {
  const dt = v.ctlDt, tau = Math.max(0.05, v.ctlTau), a = Math.exp(-dt / tau), Kp = v.ctlKp, c = v.ctlArea / 1000 / 60 / v.ctlTank, q = dt - tau * (1 - a);
  return { A: [[a, 0], [c * tau * (1 - a), 1]], B: [[(1 - a) * Kp], [c * q * Kp]], C: [[0, 1], [1, 0]], dt, w: (d1, d2) => [-(1 - a) * Kp * d2, -c * q * Kp * d2 - (dt * d1) / 60 / v.ctlTank] };
}
const DEMAND = [2, 2, 2, 2, 2, 3, 4, 5, 5, 4, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 4, 3, 2, 2]; // hourly demand in tank-level steps of 2.5 %
/** Pump-scheduling environment: 24 hourly decisions, 41 tank levels, 0/1/2 trains running, time-of-use tariff. */
function tankEnv(v) {
  const tariff = (h) => (h >= 17 && h <= 21 ? v.rlPeak : h >= 7 && h <= 22 ? v.rlShoulder : v.rlOff), s0 = clamp(Math.round(v.rlStart / 2.5), 0, 40), E = v.rlPower;
  return { T: 24, nS: 41, nA: 3, s0, tariff,
    step(t, s, a) { const l = s + 3 * a - DEMAND[t], short = Math.max(0, -l), cost = a * E * tariff(t); return { s2: clamp(l, 0, 40), reward: -(cost + v.rlPenalty * 2.5 * short), cost, short, spill: Math.max(0, l - 40) }; },
    terminal: (s) => -Math.max(0, s0 - s) * (E / 3) * v.rlPeak };
}
function rollout(env, policy) {
  let s = env.s0, cost = 0, short = 0, peak = 0, energy = 0;
  const act = [], lev = [2.5 * s];
  for (let t = 0; t < env.T; t++) { const a = policy(t, s), r = env.step(t, s, a); cost += r.cost; short += 2.5 * r.short; energy += a; if (t >= 17 && t <= 21) peak += a; act.push(a); s = r.s2; lev.push(2.5 * s); }
  return { cost, short, act, lev, term: -env.terminal(s), total: cost - env.terminal(s), peakShare: energy ? (100 * peak) / energy : 0, trainHours: energy };
}
async function taskCTL(v, ctx) {
  const mdl = roTrainModel(v), dt = mdl.dt, n = clamp(Math.round(v.ctlTsim / dt), 20, 2000), Tend = n * dt, tS = 0.06 * Tend, tD1 = 0.42 * Tend, tD2 = 0.72 * Tend;
  const ref = (k) => (k * dt >= tS ? v.ctlStep : 0), d1 = (k) => (k * dt >= tD1 ? v.ctlDist : 0), d2 = (k) => (k * dt >= tD2 ? v.ctlOsm : 0);
  const next = (x, u, w) => [mdl.A[0][0] * x[0] + mdl.B[0][0] * u + w[0], mdl.A[1][0] * x[0] + x[1] + mdl.B[1][0] * u + w[1]];
  const N = clamp(Math.round(v.ctlN), 3, 40), ctrl = mpcBuild(mdl, { N, Q: [v.ctlQy, 0], R: [v.ctlR], umin: [-v.ctlUmax], umax: [v.ctlUmax], dumax: [v.ctlDu], ymin: [-Infinity, -v.ctlJmax], ymax: [Infinity, v.ctlJmax] });
  const sim = async (kind) => {
    let x = [0, 0], u = 0, wh = [0, 0], I = 0, act = 0;
    const o = { t: [], h: [], J: [], u: [], r: [] };
    for (let k = 0; k < n; k++) {
      const r = ref(k);
      if (kind === 'mpc') { const s = ctrl.solve(x, [u], [r, 0], wh); u = clamp(s.u[0], Math.max(-v.ctlUmax, u - v.ctlDu), Math.min(v.ctlUmax, u + v.ctlDu)); act += s.active > 0 ? 1 : 0; }
      else { // PI with rate limit, clamping and conditional-integration anti-windup
        const e = r - x[1], raw = v.ctlKc * (e + I / Math.max(0.1, v.ctlTi)), un = clamp(raw, Math.max(-v.ctlUmax, u - v.ctlDu), Math.min(v.ctlUmax, u + v.ctlDu));
        if (un === raw || (raw - un) * e < 0) I += e * dt;
        u = un;
      }
      o.t.push(k * dt); o.h.push(x[1]); o.J.push(x[0]); o.u.push(u); o.r.push(r);
      const xp = next(x, u, wh), xn = next(x, u, mdl.w(d1(k), d2(k)));
      wh = wh.map((q, i) => q + 0.6 * (xn[i] - xp[i])); x = xn;
      if (kind === 'mpc' && k % 40 === 39) { ctx.progress(0.05 + (0.5 * k) / n, 'Model-predictive control simulation'); await ctx.tick(); }
    }
    const iae = sum(o.h.map((q, k) => Math.abs(o.r[k] - q))) * dt, kS = Math.ceil(tS / dt), kD = Math.ceil(tD1 / dt), band = 0.05 * Math.abs(v.ctlStep || 1);
    let settle = 0;
    for (let k = kS; k < kD; k++) if (Math.abs(o.h[k] - v.ctlStep) > band) settle = (k + 1 - kS) * dt;
    return { ...o, iae, settle, over: Math.max(0, Math.max(...o.h.slice(kS, kD)) - v.ctlStep), jExc: Math.max(0, Math.max(...o.J.map(Math.abs)) - v.ctlJmax), tExc: o.J.filter((q) => Math.abs(q) > v.ctlJmax * 1.001).length * dt,
      effort: sum(o.u.map((q, k) => (k ? Math.abs(q - o.u[k - 1]) : Math.abs(q)))), dev2: Math.max(...o.h.slice(kD).map((q, k) => Math.abs(q - o.r[kD + k]))), active: act };
  };
  const mpc = await sim('mpc'), pi = await sim('pi');
  // reinforcement learning: pump scheduling against the tariff
  ctx.progress(0.6, 'Q-learning the pump schedule'); await ctx.tick();
  const env = tankEnv(v), ql = qLearn(env, { episodes: clamp(Math.round(v.rlEpisodes), 200, 200000), alpha: clamp(v.rlAlpha, 0.01, 1), eps0: 1, epsMin: 0.05, seed: Math.round(v.seed), s0: env.s0 });
  const dp = dpSolve(env), rQ = rollout(env, ql.policy), rD = rollout(env, dp.policy), rR = rollout(env, (t, s) => (s < 12 ? 2 : s < 28 ? 1 : 0)), rF = rollout(env, () => 1);
  const gap = rD.total > 0 ? (100 * (rQ.total - rD.total)) / rD.total : 0, save = rR.total > 0 ? (100 * (rR.total - rQ.total)) / rR.total : 0, hrs = range(24), W = [];
  if (mpc.jExc > 1e-3 * Math.max(1, v.ctlJmax)) W.push({ level: 'warn', msg: `The predictive controller exceeded the flux limit by ${fmt(mpc.jExc, 3)} L/m²·h: the constraints cannot all be met for this disturbance — relax the rate limit or lengthen the horizon.` });
  else W.push({ level: 'info', msg: `The predictive controller kept flux and pressure inside their limits throughout (constraints active on ${mpc.active} of ${n} steps).` });
  if (pi.jExc > 1e-3) W.push({ level: 'warn', msg: `The PI loop drove flux ${fmt(pi.jExc, 3)} L/m²·h beyond the ±${fmt(v.ctlJmax)} limit for ${fmt(pi.tExc, 3)} min — it knows only the pressure clamp, not the flux constraint.` });
  if (rQ.short > 0) W.push({ level: 'bad', msg: `The learned schedule leaves ${fmt(rQ.short, 3)} % of tank volume of demand unmet — train longer or raise the shortage penalty.` });
  if (gap > 5) W.push({ level: 'warn', msg: `The learned schedule costs ${fmt(gap, 3)} % more than the dynamic-programming optimum: more episodes or a slower exploration decay are needed.` });
  W.push({ level: 'info', msg: 'Scope: a two-state linear plant model and a tabular agent on a 24-hour, 41-level problem. Deep reinforcement learning and nonlinear MPC are outside this tool.' });
  const stepS = (a) => ({ x: [...hrs, 24], y: [...a, a[23]], mode: 'step' });
  return {
    summary: `Predictive control tracks the level set-point with an integral absolute error of ${fmt(mpc.iae, 3)} m·min against ${fmt(pi.iae, 3)} for PI while holding flux within ±${fmt(v.ctlJmax)} L/m²·h (PI exceeds it by ${fmt(pi.jExc, 3)}). The learned pump schedule costs $${fmt(rQ.total, 4)}/day — ${fmt(save, 3)} % below level-band control and ${gap < 0.05 ? 'equal to' : `${fmt(gap, 3)} % above`} the exact dynamic-programming optimum.`,
    warnings: W,
    kpis: [
      { label: 'MPC integral absolute error', value: mpc.iae, unit: 'm·min' }, { label: 'PI integral absolute error', value: pi.iae, unit: 'm·min' }, { label: 'MPC flux-limit excess', value: mpc.jExc, unit: 'L/m²·h', status: mpc.jExc > 1e-3 ? 'warn' : 'ok' },
      { label: 'PI flux-limit excess', value: pi.jExc, unit: 'L/m²·h', status: pi.jExc > 1e-3 ? 'warn' : 'ok' }, { label: 'MPC settling time (5 %)', value: mpc.settle, unit: 'min' }, { label: 'PI settling time (5 %)', value: pi.settle, unit: 'min' },
      { label: 'MPC overshoot', value: mpc.over, unit: 'm' }, { label: 'PI overshoot', value: pi.over, unit: 'm' }, { label: 'Learned schedule cost', value: rQ.total, unit: '$/day', status: rQ.short > 0 ? 'bad' : 'ok' },
      { label: 'Level-band control cost', value: rR.total, unit: '$/day' }, { label: 'Dynamic-programming optimum', value: rD.total, unit: '$/day' }, { label: 'Saving versus level-band control', value: save, unit: '%', status: save >= 0 ? 'ok' : 'warn' },
      { label: 'Gap to optimum', value: gap, unit: '%', status: gap <= 5 ? 'ok' : 'warn' }, { label: 'Peak-tariff share of running hours', value: rQ.peakShare, unit: '%' },
    ],
    recommendations: [
      pi.jExc > 1e-3 ? 'Where flux or recovery limits matter during transients, use constraint-aware predictive control or add an override/selector to the PI loop.' : 'For this disturbance size a well-tuned PI loop is adequate; MPC pays off when constraints become active.',
      `Shift production to off-peak hours: the learned schedule runs only ${fmt(rQ.peakShare, 3)} % of its train-hours in the peak tariff against ${fmt(rR.peakShare, 3)} % for level-band control.`,
      'Identify the flux gain and time constant from a step test on the real train before using these tunings; send the pump schedule to suite 12 for the energy check.',
    ],
    plots: [
      { type: 'line', title: 'Tank level: set-point tracking and disturbance rejection', xlabel: 'Time (min)', ylabel: 'Level deviation (m)', series: [{ name: 'Set-point', x: mpc.t, y: mpc.r, mode: 'step', dash: true }, { name: 'MPC', x: mpc.t, y: mpc.h }, { name: 'PI', x: pi.t, y: pi.h }], vlines: [{ x: tD1, label: 'demand step' }, { x: tD2, label: 'salinity step' }] },
      { type: 'line', title: 'Permeate flux deviation and its constraint', xlabel: 'Time (min)', ylabel: 'Flux deviation (L/m²·h)', series: [{ name: 'MPC', x: mpc.t, y: mpc.J }, { name: 'PI', x: pi.t, y: pi.J }], hlines: [{ y: v.ctlJmax, label: 'upper limit' }, { y: -v.ctlJmax, label: 'lower limit' }] },
      { type: 'line', title: 'Manipulated variable: feed-pressure deviation', xlabel: 'Time (min)', ylabel: 'Pressure deviation (bar)', series: [{ name: 'MPC', x: mpc.t, y: mpc.u, mode: 'step' }, { name: 'PI', x: pi.t, y: pi.u, mode: 'step' }], hlines: [{ y: v.ctlUmax, label: 'upper bound' }, { y: -v.ctlUmax, label: 'lower bound' }] },
      { type: 'line', title: 'Q-learning curve', xlabel: 'Episode', ylabel: 'Daily cost including penalties ($)', series: [{ name: 'Greedy policy from the start level', x: ql.curve.episode, y: ql.curve.greedy.map((q) => -q) }, { name: 'Exploring episodes (block mean)', x: ql.curve.episode, y: ql.curve.explore.map((q) => -q), dash: true }], hlines: [{ y: rD.total, label: 'exact optimum' }], logy: true },
      { type: 'line', title: 'Pump schedule over the day', xlabel: 'Hour of day', ylabel: 'Trains running · tariff ×10 ($/kWh)', series: [{ name: 'Q-learning', ...stepS(rQ.act) }, { name: 'Dynamic programming (optimum)', ...stepS(rD.act), dash: true }, { name: 'Level-band control', ...stepS(rR.act), dash: true }, { name: 'Tariff × 10', ...stepS(hrs.map((h) => 10 * env.tariff(h))) }] },
      { type: 'line', title: 'Tank level over the day', xlabel: 'Hour of day', ylabel: 'Level (% of volume)', ymin: 0, ymax: 100, series: [{ name: 'Q-learning', x: [...hrs, 24], y: rQ.lev }, { name: 'Dynamic programming', x: [...hrs, 24], y: rD.lev, dash: true }, { name: 'Level-band control', x: [...hrs, 24], y: rR.lev, dash: true }] },
    ],
    tables: [
      { title: 'Controller comparison', columns: ['Controller', 'IAE (m·min)', 'Settling time (min)', 'Overshoot (m)', 'Max deviation after demand step (m)', 'Flux-limit excess (L/m²·h)', 'Time above flux limit (min)', 'Total pressure movement (bar)'],
        rows: [['MPC', mpc], ['PI', pi]].map(([nm, q]) => [nm, q.iae, q.settle, q.over, q.dev2, q.jExc, q.tExc, q.effort]),
        note: `Plant: τ·dJ/dt = −J + Kp·(u − Δπ), A_tank·dh/dt = A_m·J − ΔQ_demand, sampled every ${fmt(dt)} min. MPC: horizon ${N}, weights Q = ${fmt(v.ctlQy)}, R = ${fmt(v.ctlR)}, ${ctrl.nConstraints} linear constraints solved by a dual (Hildreth) QP with a disturbance estimator for offset-free tracking. PI: Kc = ${fmt(v.ctlKc)} bar/m, Ti = ${fmt(v.ctlTi)} min.` },
      { title: 'Pump-scheduling policies', columns: ['Policy', 'Energy cost ($/day)', 'End-of-day refill charge ($)', 'Total ($/day)', 'Train-hours', 'Share in peak tariff (%)', 'Unmet demand (% volume)'],
        rows: [['Q-learning (greedy)', rQ], ['Dynamic programming (exact optimum)', rD], ['Level-band control', rR], ['One train continuously', rF]].map(([nm, q]) => [nm, q.cost, q.term, q.total, q.trainHours, q.peakShare, q.short]),
        note: 'Tank drawn below its starting level at the end of the day is charged at the peak tariff so that policies cannot win by emptying the tank.' },
      { title: 'Hourly schedule', columns: ['Hour', 'Tariff ($/kWh)', 'Demand (% volume/h)', 'Q-learning trains', 'Optimum trains', 'Level-band trains', 'Level with Q-learning (%)'], rows: hrs.map((h) => [h, env.tariff(h), 2.5 * DEMAND[h], rQ.act[h], rD.act[h], rR.act[h], rQ.lev[h]]) },
    ],
    outputs: { task: 'ctl', objective: rQ.total, mpcIAE: mpc.iae, piIAE: pi.iae, scheduleCost: rQ.total, optimalCost: rD.total, schedule: rQ.act, study: { a: mpc.iae, b: rQ.total } },
  };
}

// ---- Task 9: custom numerical modelling workbench ----------------------------------------------------------------------
const compile = (text, names, label) => { if (blank(text)) throw new Error(`${label} is empty — enter an expression.`); return parseExpr(text, names); };
const scopeOf = (ps, extra) => Object.assign(Object.create(null), ps, extra);
/** Method-of-manufactured-solutions case for the PDE solver: u = 1 + e^(−t)·sin(πx/2), v = 0.6, D = 0.05, R = −0.8u + S(x, t). */
export function mmsCase(nx, nt, theta = 0.5, scheme = 'central') {
  const vv = 0.6, D = 0.05, k = 0.8, hp = Math.PI / 2, um = (x, t) => 1 + Math.exp(-t) * Math.sin(hp * x);
  const S = (x, t) => { const e = Math.exp(-t), s = Math.sin(hp * x), c = Math.cos(hp * x); return -e * s + vv * hp * e * c + D * hp * hp * e * s + k * (1 + e * s); };
  const r = solveCDR({ L: 1, nx, tEnd: 0.5, nt, v: vv, D, R: (u, x, t) => -k * u + S(x, t), ic: (x) => um(x, 0), left: { type: 'dirichlet', val: () => 1 }, right: { type: 'neumann', val: () => 0 }, theta, scheme });
  return { err: Math.sqrt(mean(r.u.map((q, i) => (q - um(r.x[i], 0.5)) ** 2))), mean: r.mean, dx: r.dx };
}
async function wbODE(v, ctx, ps) {
  const n = clamp(Math.round(v.odeN), 1, 4), names = ['t', 'y1', 'y2', 'y3', 'y4', ...Object.keys(ps)], fs = range(n).map((i) => compile(v['odeF' + (i + 1)], names, `dy${i + 1}/dt`));
  const sc = scopeOf(ps, { t: 0, y1: 0, y2: 0, y3: 0, y4: 0 }), f = (t, y) => { sc.t = t; for (let i = 0; i < n; i++) sc['y' + (i + 1)] = y[i]; return fs.map((q) => q.evaluate(sc)); };
  const y0 = range(n).map((i) => finite(+v['odeY' + (i + 1)])), tEnd = Math.max(1e-9, v.odeT), nSt = clamp(Math.ceil(tEnd / Math.max(1e-9, v.odeDt) - 1e-9), 2, 50000);
  const ok = (s) => s.y.every((r) => r.every(Number.isFinite));
  const fix = [nSt, 2 * nSt, 4 * nSt].map((k) => rk4(f, y0, 0, tEnd, k)); await ctx.tick();
  const adp = rk45(f, y0, 0, tEnd, { rtol: 1e-9, atol: 1e-12, maxSteps: 40000 });
  if (!ok(adp)) throw new Error('The solution became non-finite (division by zero or blow-up). Check the expressions, initial values and end time.');
  if (!fix.every(ok)) throw new Error(`Fixed-step RK4 diverges with Δt = ${fmt(tEnd / nSt)}: the system is stiff. Reduce the ODE time step on the Mesh tab to about ${fmt(tEnd / Math.max(1, adp.t.length - 1), 2)} or less (the adaptive integrator needed ${adp.t.length - 1} steps).`);
  const end = (s) => s.y[s.y.length - 1], ref = end(adp), reached = Math.abs(adp.t[adp.t.length - 1] - tEnd) < 1e-9 * tEnd;
  const errs = fix.map((s) => Math.max(...end(s).map((q, i) => Math.abs(q - end(fix[2])[i]) / Math.max(1e-12, Math.abs(ref[i])))));
  const e1 = Math.max(...end(fix[0]).map((q, i) => Math.abs(q - end(fix[1])[i]))), e2 = Math.max(...end(fix[1]).map((q, i) => Math.abs(q - end(fix[2])[i]))), order = e1 > 1e-13 && e2 > 1e-15 ? Math.log2(e1 / e2) : 4;
  const sol = v.odeMethod === 'rk4' ? fix[0] : adp, relErr = Math.max(...end(fix[0]).map((q, i) => Math.abs(q - ref[i]) / Math.max(1e-12, Math.abs(ref[i])))), W = [];
  if (!reached) W.push({ level: 'warn', msg: 'The adaptive integrator hit its step limit before the end time — the system is stiff or singular; the fixed-step result is shown for comparison.' });
  if (relErr > 1e-3) W.push({ level: 'warn', msg: `Fixed-step RK4 with Δt = ${fmt(tEnd / nSt)} differs from the adaptive reference by ${fmt(100 * relErr, 2)} % — reduce the time step (see the step-size study on the Mesh tab).` });
  else W.push({ level: 'info', msg: `Fixed-step and adaptive solutions agree to ${fmt(100 * relErr, 2)} % at the end time; observed order of the fixed-step method ≈ ${fmt(order, 3)}.` });
  const thin = (s, k = 40) => { const st = Math.max(1, Math.ceil(s.t.length / k)); return range(s.t.length).filter((i) => i % st === 0 || i === s.t.length - 1); }, id = thin(sol);
  const scl = range(n).map((i) => Math.max(1e-300, ...sol.y.map((r) => Math.abs(r[i]))));
  return {
    summary: `Integrated ${n} user-defined equation${n > 1 ? 's' : ''} to t = ${fmt(tEnd)}: ${range(n).map((i) => `y${i + 1} = ${fmt(end(sol)[i], 5)}`).join(', ')} (${v.odeMethod === 'rk4' ? `RK4, ${nSt} steps` : `adaptive RK45, ${adp.t.length - 1} steps`}).`,
    warnings: W,
    kpis: [...range(n).map((i) => ({ label: `y${i + 1} at end time`, value: end(sol)[i] })), ...range(n).map((i) => ({ label: `y${i + 1} range`, value: `${fmt(Math.min(...sol.y.map((r) => r[i])), 4)} – ${fmt(Math.max(...sol.y.map((r) => r[i])), 4)}` })),
      { label: 'Fixed steps', value: nSt }, { label: 'Adaptive steps', value: adp.t.length - 1 }, { label: 'RK4 vs adaptive difference', value: 100 * relErr, unit: '%', status: relErr > 1e-3 ? 'warn' : 'ok' }, { label: 'Observed order (RK4)', value: order, unit: '–' }],
    recommendations: ['Use the step-size study on the Mesh tab (ODE time step) to quantify the numerical uncertainty of the end values.', 'Stiff systems (very different time scales) need a much smaller step or a reformulation; the adaptive method reports this by taking many steps.', 'Move parameters into the parameter table so they can be changed without editing the equations.'],
    plots: [
      { type: 'line', title: 'Solution', xlabel: 'Time', ylabel: 'State value', series: range(n).map((i) => ({ name: `y${i + 1}`, x: sol.t, y: sol.y.map((r) => r[i]) })) },
      { type: 'line', title: 'Solution scaled by each state’s largest magnitude', xlabel: 'Time', ylabel: 'y / max|y|', series: range(n).map((i) => ({ name: `y${i + 1}`, x: sol.t, y: sol.y.map((r) => r[i] / scl[i]) })) },
      ...(n >= 2 ? [{ type: 'line', title: 'Phase plane', xlabel: 'y1', ylabel: 'y2', series: [{ name: 'Trajectory', x: sol.y.map((r) => r[0]), y: sol.y.map((r) => r[1]) }, { name: 'Start', x: [y0[0]], y: [y0[1]], mode: 'points' }] }] : []),
      { type: 'line', title: 'Fixed-step RK4 versus adaptive RK45', xlabel: 'Time', ylabel: 'y1', series: [{ name: `RK4, Δt = ${fmt(tEnd / nSt)}`, x: fix[0].t, y: fix[0].y.map((r) => r[0]), mode: 'points' }, { name: 'Adaptive RK45', x: adp.t, y: adp.y.map((r) => r[0]) }] },
    ],
    tables: [
      { title: 'Trajectory', columns: ['t', ...range(n).map((i) => `y${i + 1}`), ...range(n).map((i) => `dy${i + 1}/dt`)], rows: id.map((i) => [sol.t[i], ...sol.y[i], ...f(sol.t[i], sol.y[i])]) },
      { title: 'Step-size check (end values)', columns: ['Method', 'Steps', ...range(n).map((i) => `y${i + 1}`), 'Change to finest RK4 (relative)'], rows: [...fix.map((s, k) => [`RK4, Δt = ${fmt(tEnd / (nSt * 2 ** k))}`, nSt * 2 ** k, ...end(s), errs[k]]), ['Adaptive RK45 (rtol 1e-9)', adp.t.length - 1, ...ref, null]] },
      { title: 'Equations as parsed', columns: ['Equation', 'Expression', 'Initial value'], rows: range(n).map((i) => [`dy${i + 1}/dt`, fs[i].text, y0[i]]), note: `Parameters: ${Object.keys(ps).map((k) => `${k} = ${fmt(ps[k])}`).join(', ') || 'none'}.` },
    ],
    outputs: { task: 'wb', mode: 'ode', objective: end(fix[0])[0], final: end(sol), study: { a: end(fix[0])[0], b: end(fix[0])[Math.min(1, n - 1)] } },
  };
}
async function wbALG(v, ctx, ps) {
  const n = clamp(Math.round(v.algN), 1, 4), xn = ['x1', 'x2', 'x3', 'x4'], pk = Object.keys(ps), fs = range(n).map((i) => compile(v['algF' + (i + 1)], [...xn, ...pk], `Equation ${i + 1}`));
  const mk = (pp) => { const sc = scopeOf(pp, { x1: 0, x2: 0, x3: 0, x4: 0 }); return (x) => { for (let i = 0; i < n; i++) sc[xn[i]] = x[i]; return fs.map((q) => finite(q.evaluate(sc), 1e30)); }; };
  const F = mk(ps), x0 = range(n).map((i) => finite(+v['algX' + (i + 1)])), sol = newtonN(F, x0, { tol: 1e-10, maxIter: 120 });
  await ctx.tick();
  // robustness: Newton from scattered starting points
  const roots = [], U = lhs(16, n, Math.round(v.seed));
  let nConv = 0;
  for (const u of U) {
    const s = newtonN(F, x0.map((q, i) => (Math.abs(q) > 1e-12 ? q * (0.4 + 1.2 * u[i]) : 2 * u[i] - 1)), { tol: 1e-9, maxIter: 80 });
    if (!s.converged) continue;
    nConv++;
    const hit = roots.find((r) => r.x.every((q, i) => Math.abs(q - s.x[i]) <= 1e-5 * (1 + Math.abs(q))));
    if (hit) hit.count++; else roots.push({ x: s.x, count: 1 });
  }
  // parameter elasticities of the solution, d ln x / d ln p
  const el = pk.filter((k) => fs.some((q) => q.vars.includes(k)) && ps[k] !== 0).map((k) => { const s = newtonN(mk({ ...ps, [k]: ps[k] * 1.01 }), sol.x, { tol: 1e-10 }); return { k, e: sol.x.map((q, i) => (s.converged && Math.abs(q) > 1e-12 ? (s.x[i] - q) / q / 0.01 : 0)) }; });
  const span = Math.max(Math.abs(sol.x[0]) * 0.5, 0.5), sx = linspace(sol.x[0] - span, sol.x[0] + span, 61), sy = sx.map((q) => { const x = [...sol.x]; x[0] = q; return Math.max(1e-16, Math.hypot(...F(x))); });
  const W = [sol.converged ? { level: 'info', msg: `Newton converged in ${sol.iterations} iterations to a residual of ${fmt(sol.residual, 2)}.` } : { level: 'bad', msg: `Newton did not converge (residual ${fmt(sol.residual, 3)}). Try other initial guesses; ${nConv} of ${U.length} scattered starts did converge.` }];
  if (roots.length > 1) W.push({ level: 'warn', msg: `${roots.length} distinct solutions were found from scattered starting points — check which one is physically meaningful.` });
  return {
    summary: sol.converged ? `Solved ${n} nonlinear equation${n > 1 ? 's' : ''}: ${range(n).map((i) => `x${i + 1} = ${fmt(sol.x[i], 6)}`).join(', ')} (residual ${fmt(sol.residual, 2)}, ${sol.iterations} Newton iterations).` : `The Newton iteration stopped at a residual of ${fmt(sol.residual, 3)} without converging.`,
    warnings: W,
    kpis: [...range(n).map((i) => ({ label: `x${i + 1}`, value: finite(sol.x[i]), status: sol.converged ? 'ok' : 'bad' })), { label: 'Residual norm', value: finite(sol.residual, 1e30), status: sol.converged ? 'ok' : 'bad' }, { label: 'Newton iterations', value: sol.iterations },
      { label: 'Converged', value: sol.converged ? 'yes' : 'no', status: sol.converged ? 'ok' : 'bad' }, { label: 'Scattered starts converging', value: (100 * nConv) / U.length, unit: '%' }, { label: 'Distinct solutions found', value: roots.length, status: roots.length > 1 ? 'warn' : 'ok' }],
    recommendations: ['Scale the equations so that all residuals have similar magnitude; Newton converges faster and the residual norm becomes meaningful.', roots.length > 1 ? 'Several roots exist: constrain the problem (bounds, physical inequalities) or start close to the wanted branch.' : 'The same root is reached from scattered starts — the solution is robust to the initial guess.', 'The elasticity chart shows which parameters the solution depends on most.'],
    plots: [
      { type: 'line', title: 'Residual norm along x1 through the solution', xlabel: 'x1', ylabel: '‖F(x)‖', logy: true, series: [{ name: '‖F‖', x: sx, y: sy }], vlines: [{ x: finite(sol.x[0]), label: 'solution' }] },
      ...(el.length ? [{ type: 'bar', title: 'Parameter elasticities of the solution', ylabel: '% change of unknown per % change of parameter', categories: el.map((q) => q.k), series: range(n).map((i) => ({ name: `x${i + 1}`, values: el.map((q) => finite(q.e[i])) })) }] : []),
      { type: 'bar', title: 'Solutions reached from scattered starting points', ylabel: 'Number of starts', categories: [...roots.map((_, i) => `Solution ${i + 1}`), 'Not converged'], series: [{ name: 'Starts', values: [...roots.map((r) => r.count), U.length - nConv] }] },
    ],
    tables: [
      { title: 'Solution', columns: ['Unknown', 'Initial guess', 'Solution', 'Equation', 'Residual'], rows: range(n).map((i) => [`x${i + 1}`, x0[i], finite(sol.x[i]), fs[i].text + ' = 0', finite(F(sol.x)[i], 1e30)]) },
      { title: 'Solutions found from 16 scattered starts', columns: ['Solution', ...range(n).map((i) => `x${i + 1}`), 'Starts converging to it'], rows: roots.map((r, i) => [i + 1, ...r.x, r.count]), note: `Parameters: ${pk.map((k) => `${k} = ${fmt(ps[k])}`).join(', ') || 'none'}.` },
    ],
    outputs: { task: 'wb', mode: 'alg', objective: finite(sol.residual, 1e30), solution: sol.x.map((q) => finite(q)), study: { a: finite(sol.x[0]), b: finite(sol.residual, 1e30) } },
  };
}
async function wbOPT(v, ctx, ps) {
  const rows = numRows(v.optVars, ['lo', 'hi']).filter((r) => r.hi > r.lo).slice(0, 6), n = rows.length;
  if (!n) throw new Error('The variable table needs at least one row with lower bound < upper bound.');
  const xn = range(n).map((i) => `x${i + 1}`), names = [...xn, ...Object.keys(ps)], fo = compile(v.optF, names, 'Objective'), gs = [v.optG1, v.optG2, v.optG3].filter((q) => !blank(q)).map((q, i) => compile(q, names, `Constraint ${i + 1}`)), he = blank(v.optH) ? null : compile(v.optH, names, 'Equality constraint');
  const sc = scopeOf(ps, {}), set = (x) => { for (let i = 0; i < n; i++) sc[xn[i]] = x[i]; };
  let evals = 0;
  const f = (x) => { evals++; set(x); return finite(fo.evaluate(sc), 1e30); }, g = (x) => { set(x); return gs.map((q) => finite(q.evaluate(sc), 1e30)); }, h = (x) => { set(x); return he ? finite(he.evaluate(sc), 1e30) : 0; };
  const lo = rows.map((r) => r.lo), hi = rows.map((r) => r.hi), x0 = rows.map((r) => clamp(isNum(r.x0) ? r.x0 : 0.5 * (r.lo + r.hi), r.lo, r.hi));
  if (Math.abs(f(x0)) >= 1e29) throw new Error('The objective is not finite at the starting point — check the expression (division by zero, logarithm or root of a negative number) or move the start.');
  // augmented Lagrangian: bound-constrained inner minimisations, multiplier updates outside
  let x = x0, lam = gs.map(() => 0), mu = 0, rho = 10, prevViol = Infinity;
  const hist = [], viol = (q) => Math.max(0, ...g(q), Math.abs(h(q)));
  for (let it = 0; it < 10; it++) {
    const L = (q) => { const gq = g(q), hq = h(q); let s = f(q) + mu * hq + 0.5 * rho * hq * hq; gq.forEach((gi, i) => { s += (Math.max(0, lam[i] + rho * gi) ** 2 - lam[i] ** 2) / (2 * rho); }); return s; };
    if (v.wbAlgo === 'de' && it === 0) x = diffEvolution(L, lo, hi, { pop: clamp(12 * n, 16, 60), gens: 80, seed: Math.round(v.seed) }).x;
    x = nelderMead(L, x, { lo, hi, maxIter: 300 * n, tol: 1e-12, scale: it === 0 ? 0.1 : 0.02 }).x;
    const gq = g(x), vq = viol(x);
    lam = lam.map((l, i) => Math.max(0, l + rho * gq[i])); mu += rho * h(x);
    hist.push({ it: it + 1, f: f(x), viol: vq, rho });
    if (vq < 1e-8 && it >= 2 && Math.abs(hist[it].f - hist[it - 1].f) < 1e-10 * (1 + Math.abs(hist[it].f))) break;
    if (vq > 0.25 * prevViol) rho *= 4;
    prevViol = vq;
    await ctx.tick();
  }
  const cons = (q) => [...g(q), ...(he ? [h(q), -h(q)] : [])], kk = kktCheck(f, cons, x, { lo, hi, h: 1e-6, actTol: 1e-5 }), fx = f(x), gx = g(x), vq = viol(x), feasible = vq < 1e-5, nEv = evals, W = [];
  W.push(feasible ? { level: 'info', msg: `Constraints satisfied (largest violation ${fmt(vq, 2)}); ${gx.filter((q) => q > -1e-5).length} inequality constraint(s) active.` } : { level: 'bad', msg: `The best point violates the constraints by ${fmt(vq, 3)} — the problem may be infeasible inside the bounds.` });
  if (feasible && kk.relResidual > 1e-2) W.push({ level: 'warn', msg: `Stationarity residual is ${fmt(kk.relResidual, 2)} of the gradient norm: the point may be a kink or not fully converged; try the global algorithm or another start.` });
  const plots = [{ type: 'line', title: 'Augmented-Lagrangian iterations', xlabel: 'Outer iteration', ylabel: 'Objective · violation', series: [{ name: 'Objective', x: hist.map((q) => q.it), y: hist.map((q) => q.f), mode: 'both' }, { name: 'Largest constraint violation', x: hist.map((q) => q.it), y: hist.map((q) => q.viol), mode: 'both' }] }];
  if (n >= 2) {
    const gx1 = linspace(lo[0], hi[0], 41), gx2 = linspace(lo[1], hi[1], 41), z = [], mask = [];
    for (const b of gx2) { const zr = [], mr = []; for (const a of gx1) { const q = [...x]; q[0] = a; q[1] = b; const fq = f(q); zr.push(fq); mr.push(viol(q) > 1e-9 || Math.abs(fq) >= 1e29); } z.push(zr); mask.push(mr); }
    const fl = z.flat().filter((q) => Math.abs(q) < 1e29), zl = quantile(fl, 0.02), zh = quantile(fl, 0.9);
    plots.push({ type: 'field', title: 'Objective over x1–x2 (infeasible region masked)', xlabel: 'x1', ylabel: 'x2', zlabel: 'Objective', x: gx1, y: gx2, z: z.map((r) => r.map((q) => clamp(q, zl, zh))), zmin: zl, zmax: zh, mask, cmap: 'viridis', contours: 10, markers: [{ x: x[0], y: x[1], label: 'optimum' }, { x: x0[0], y: x0[1], label: 'start' }] });
  } else { const sx = linspace(lo[0], hi[0], 81); plots.push({ type: 'line', title: 'Objective along x1', xlabel: 'x1', ylabel: 'Objective', series: [{ name: 'f(x1)', x: sx, y: sx.map((q) => f([q])) }], vlines: [{ x: x[0], label: 'optimum' }] }); }
  if (gs.length) plots.push({ type: 'bar', title: 'Constraint values at the optimum (≤ 0 is feasible)', ylabel: 'g(x)', categories: gs.map((_, i) => `g${i + 1}`), series: [{ name: 'At start', values: g(x0) }, { name: 'At optimum', values: gx }] });
  return {
    summary: `${feasible ? 'Constrained minimum' : 'Best (infeasible) point'}: f = ${fmt(fx, 6)} at ${xn.map((nm, i) => `${nm} = ${fmt(x[i], 5)}`).join(', ')}; ${gx.filter((q) => q > -1e-5).length} of ${gs.length} inequality constraints active, KKT residual ${fmt(kk.residual, 2)}.`,
    warnings: W,
    kpis: [{ label: 'Objective at optimum', value: fx, status: feasible ? 'ok' : 'bad' }, { label: 'Objective at start', value: f(x0) }, ...xn.map((nm, i) => ({ label: nm, value: x[i] })), { label: 'Largest constraint violation', value: vq, status: feasible ? 'ok' : 'bad' },
      { label: 'Active inequality constraints', value: gx.filter((q) => q > -1e-5).length }, { label: 'KKT stationarity residual', value: kk.residual, status: kk.relResidual < 1e-2 ? 'ok' : 'warn' }, { label: 'Outer iterations', value: hist.length }, { label: 'Objective evaluations', value: nEv }],
    recommendations: ['A non-zero multiplier is the rate at which the objective would improve if that constraint were relaxed by one unit.', v.wbAlgo === 'de' ? 'Differential evolution started the search globally; repeat with another seed to confirm the optimum.' : 'Nelder–Mead is local: for multi-modal objectives switch to the global algorithm or try several starting points.', 'Write constraints as g(x) ≤ 0 and scale them to order one for well-conditioned multipliers.'],
    plots,
    tables: [
      { title: 'Variables', columns: ['Variable', 'Lower bound', 'Upper bound', 'Start', 'Optimum', 'Bound multiplier'], rows: xn.map((nm, i) => [nm, lo[i], hi[i], x0[i], x[i], Math.max(kk.lamLo[i], kk.lamHi[i])]) },
      { title: 'Constraints', columns: ['Constraint', 'Expression', 'Value at optimum', 'Status', 'Lagrange multiplier'],
        rows: [...gs.map((q, i) => [`g${i + 1} ≤ 0`, q.text, gx[i], gx[i] > 1e-5 ? 'violated' : gx[i] > -1e-5 ? 'active' : 'inactive', kk.lambda[i]]), ...(he ? [['h = 0', he.text, h(x), Math.abs(h(x)) < 1e-5 ? 'satisfied' : 'violated', kk.lambda[gs.length] - kk.lambda[gs.length + 1]]] : [])], note: `Objective: ${fo.text}. Constraints are enforced by an augmented Lagrangian; the multipliers shown are non-negative least-squares estimates from the gradients at the optimum (KKT conditions). Parameters: ${Object.keys(ps).map((k) => `${k} = ${fmt(ps[k])}`).join(', ') || 'none'}.` },
      { title: 'Iteration history', columns: ['Outer iteration', 'Objective', 'Largest violation', 'Penalty ρ'], rows: hist.map((q) => [q.it, q.f, q.viol, q.rho]) },
    ],
    outputs: { task: 'wb', mode: 'opt', objective: fx, solution: x, feasible, study: { a: fx, b: x[0] } },
  };
}
async function wbPDE(v, ctx, ps) {
  const pk = Object.keys(ps), L = Math.max(1e-12, v.pdeL), tEnd = Math.max(1e-12, v.pdeT), nx = clamp(Math.round(v.pdeNx), 4, 2000), nt = clamp(Math.round(v.pdeNt), 2, 20000);
  const Rc = blank(v.pdeR) ? null : compile(v.pdeR, ['u', 'x', 't', ...pk], 'Reaction term'), Re = Rc && !Rc.vars.length && Rc.evaluate({}) === 0 ? null : Rc, ice = compile(v.pdeIC, ['x', ...pk], 'Initial condition');
  const bc = (type, text, label) => { if (type === 'noflux') return { type, val: () => 0 }; const e = compile(text, ['t', ...pk], label), s = scopeOf(ps, { t: 0 }); return { type, val: (t) => { s.t = t; return finite(e.evaluate(s)); } }; };
  const left = bc(v.pdeBCL, v.pdeBCLv, 'Left boundary value'), right = bc(v.pdeBCR, v.pdeBCRv, 'Right boundary value'), sR = scopeOf(ps, { u: 0, x: 0, t: 0 }), sI = scopeOf(ps, { x: 0 });
  const R = Re ? (u, x, t) => { sR.u = u; sR.x = x; sR.t = t; return Re.evaluate(sR); } : null, ic = (x) => { sI.x = x; return ice.evaluate(sI); };
  const cfg = { L, tEnd, v: v.pdeV, D: Math.max(0, v.pdeD), R, ic, left, right, theta: v.pdeTheta === 'be' ? 1 : 0.5, scheme: v.pdeScheme };
  const s = solveCDR({ ...cfg, nx, nt }); await ctx.tick();
  if (!s.u.every(Number.isFinite)) throw new Error('The solution became non-finite. Check the reaction expression and the boundary values, or use more time steps.');
  const fine = solveCDR({ ...cfg, nx: 2 * nx, nt: 2 * nt, nSave: 2 }); await ctx.tick();
  const wall = (b, u, a, c) => (b.type === 'dirichlet' ? b.val(tEnd) : 1.5 * u[a] - 0.5 * u[c]), uL = wall(left, s.u, 0, 1), uR = wall(right, s.u, nx - 1, nx - 2); // boundary values (second-order extrapolation)
  const fineOn = s.x.map((_, i) => 0.5 * (fine.u[2 * i] + fine.u[2 * i + 1])), dNorm = rmse(s.u, fineOn), scale = Math.max(1e-300, maxAbs(s.u)), pe = cfg.D > 0 ? (Math.abs(cfg.v) * s.dx) / cfg.D : cfg.v ? 1e9 : 0, peTxt = pe >= 1e9 ? 'unbounded (no diffusion)' : fmt(pe, 3);
  const m = [16, 32, 64].map((k) => mmsCase(k, k, cfg.theta, cfg.scheme)), pObs = Math.log2(m[1].err / m[2].err), W = [];
  if (cfg.scheme === 'central' && pe > 2) W.push({ level: 'warn', msg: `Cell Péclet number v·Δx/D is ${peTxt}, above 2: the central scheme may oscillate. Use more cells or the upwind scheme.` });
  if (dNorm / scale > 0.01) W.push({ level: 'warn', msg: `Halving Δx and Δt changes the final profile by ${fmt((100 * dNorm) / scale, 2)} % (RMS) — refine the grid (see the grid study on the Mesh tab).` });
  else W.push({ level: 'info', msg: `Halving Δx and Δt changes the final profile by only ${fmt((100 * dNorm) / scale, 2)} % (RMS).` });
  let steady = null;
  if (!R && v.pdeBCL === 'dirichlet' && v.pdeBCR === 'noflux' && cfg.D > 0) { const ub = left.val(tEnd), ex = s.x.map((x) => ub * Math.exp((cfg.v * x) / cfg.D)); steady = { ex, gap: rmse(s.u, ex) / Math.max(1e-300, maxAbs(ex)), wall: ub * Math.exp((cfg.v * L) / cfg.D) }; W.push({ level: 'info', msg: `Analytical steady state for these boundary conditions is u = u₀·exp(v·x/D); the solution at the end time is within ${fmt(100 * steady.gap, 2)} % of it (wall value ${fmt(uR, 5)} against ${fmt(steady.wall, 5)}).` }); }
  const kk = [...new Set(range(6).map((q) => Math.round((q * (s.t.length - 1)) / 5)))], mid = Math.floor(nx / 2);
  return {
    summary: `Solved the convection–diffusion–reaction equation on ${nx} cells and ${nt} ${cfg.theta === 1 ? 'backward-Euler' : 'Crank–Nicolson'} steps: domain mean ${fmt(s.mean, 5)}, left boundary ${fmt(uL, 5)}, right boundary ${fmt(uR, 5)} at t = ${fmt(tEnd)}. The solver reproduces a manufactured solution with observed order ${fmt(pObs, 3)}.`,
    warnings: W,
    kpis: [{ label: 'Domain mean at end time', value: s.mean }, { label: 'Value at left boundary', value: uL }, { label: 'Value at right boundary', value: uR }, { label: 'Maximum over domain', value: Math.max(...s.u) }, { label: 'Minimum over domain', value: Math.min(...s.u) },
      { label: 'Cell Péclet number', value: pe >= 1e9 ? peTxt : pe, unit: pe >= 1e9 ? '' : '–', status: cfg.scheme === 'central' && pe > 2 ? 'warn' : 'ok' }, { label: 'Diffusion number D·Δt/Δx²', value: finite((cfg.D * s.dt) / s.dx ** 2), unit: '–' }, { label: 'Courant number v·Δt/Δx', value: (Math.abs(cfg.v) * s.dt) / s.dx, unit: '–' },
      { label: 'Change on grid halving (RMS)', value: (100 * dNorm) / scale, unit: '%', status: dNorm / scale > 0.01 ? 'warn' : 'ok' }, { label: 'Observed order (manufactured solution)', value: pObs, unit: '–', status: pObs > (cfg.theta === 1 || cfg.scheme === 'upwind' ? 0.8 : 1.8) ? 'ok' : 'warn' }],
    recommendations: ['Run the grid study on the Mesh tab (PDE grid and time step) for a grid-convergence index of the mean and wall values.', cfg.theta === 1 ? 'Backward Euler is first-order in time but damps oscillations; switch to Crank–Nicolson for accuracy once the solution is smooth.' : 'Crank–Nicolson is second-order; if the solution rings after a sharp initial step, use backward Euler or more time steps.', 'Use the reaction expression for sources, sinks and nonlinear kinetics, e.g. -k*u or mu*u*(1-u/K).'],
    plots: [
      { type: 'line', title: 'Profiles at selected times', xlabel: 'x', ylabel: 'u', series: [...kk.map((q) => ({ name: `t = ${fmt(s.t[q], 3)}`, x: s.x, y: s.U[q] })), ...(steady ? [{ name: 'Analytical steady state', x: s.x, y: steady.ex, dash: true }] : [])] },
      { type: 'field', title: 'Solution u(x, t)', xlabel: 'x', ylabel: 't', zlabel: 'u', x: s.x, y: s.t, z: s.U, cmap: 'salinity', contours: 8 },
      { type: 'line', title: 'History at the boundaries and mid-domain', xlabel: 't', ylabel: 'u', series: [{ name: 'Left cell', x: s.t, y: s.U.map((r) => r[0]) }, { name: 'Mid-domain', x: s.t, y: s.U.map((r) => r[mid]) }, { name: 'Right cell', x: s.t, y: s.U.map((r) => r[nx - 1]) }] },
      { type: 'line', title: 'Code verification by a manufactured solution', xlabel: 'Cell size Δx', ylabel: 'RMS error', logx: true, logy: true, series: [{ name: 'Error against manufactured solution', x: m.map((q) => q.dx), y: m.map((q) => q.err), mode: 'both' }, { name: 'Second-order slope', x: m.map((q) => q.dx), y: m.map((q) => m[2].err * (q.dx / m[2].dx) ** 2), dash: true }] },
    ],
    tables: [
      { title: 'Final profile', columns: ['x', 'u (this grid)', 'u (grid and step halved)', 'Difference'], rows: range(nx).filter((i) => i % Math.max(1, Math.ceil(nx / 40)) === 0 || i === nx - 1).map((i) => [s.x[i], s.u[i], fineOn[i], s.u[i] - fineOn[i]]) },
      { title: 'Manufactured-solution verification of the solver', columns: ['Cells = steps', 'Δx', 'RMS error', 'Observed order'], rows: m.map((q, i) => [[16, 32, 64][i], q.dx, q.err, i ? Math.log2(m[i - 1].err / q.err) : null]), note: 'u = 1 + e^(−t)·sin(πx/2) with v = 0.6, D = 0.05, R = −0.8u + S(x, t); Dirichlet left, Neumann right; same scheme and time integrator as selected.' },
      { title: 'Problem as parsed', columns: ['Item', 'Value'], rows: [['Equation', '∂u/∂t + v·∂u/∂x = D·∂²u/∂x² + R(u, x, t)'], ['v, D', `${fmt(cfg.v)}, ${fmt(cfg.D)}`], ['R(u, x, t)', Re ? Re.text : '0'], ['Initial condition u(x, 0)', ice.text], ['Left boundary', v.pdeBCL === 'noflux' ? 'zero total flux' : `${v.pdeBCL}: ${v.pdeBCLv}`], ['Right boundary', v.pdeBCR === 'noflux' ? 'zero total flux' : `${v.pdeBCR}: ${v.pdeBCRv}`], ['Discretisation', `${nx} finite volumes, ${cfg.scheme} convection, ${nt} steps, θ = ${cfg.theta}`]] },
    ],
    outputs: { task: 'wb', mode: 'pde', objective: s.mean, mean: s.mean, left: uL, right: uR, study: { a: s.mean, b: uR } },
  };
}
const taskWB = (v, ctx) => { const ps = paramScope(v.wbParams); return v.wbMode === 'alg' ? wbALG(v, ctx, ps) : v.wbMode === 'opt' ? wbOPT(v, ctx, ps) : v.wbMode === 'pde' ? wbPDE(v, ctx, ps) : wbODE(v, ctx, ps); };

// ---- Task 10: parameter estimation, identifiability and Bayesian inference -----------------------------------------------
const tQuant = (nu) => { const z = 1.959964, n = Math.max(1, nu); return z + (z ** 3 + z) / (4 * n) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * n * n) + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * n ** 3); };
function peSampleTable() { const g = rng(77); return [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 105, 120, 135, 150, 165].map((x) => ({ x, y: +(0.72 + 0.28 * Math.exp(-0.021 * x) + g.normal(0, 0.007)).toFixed(4) })); }
async function taskPE(v, ctx) {
  const data = numRows(v.peData, ['x', 'y']).sort((a, b) => a.x - b.x), n = data.length, prm = (Array.isArray(v.peParams) ? v.peParams : []).filter((r) => r && !blank(r.name) && isNum(r.value)).slice(0, 6), np = prm.length;
  if (!np) throw new Error('Add at least one parameter (name and starting value) to the parameter table.');
  if (n < np + 2) throw new Error(`${n} data rows cannot identify ${np} parameters: at least ${np + 2} rows are needed.`);
  const pn = prm.map((r) => String(r.name).trim());
  for (const nm of pn) if (!IDENT.test(nm) || FUNCS.has(nm) || nm === 'x') throw new Error(`Parameter name “${nm.slice(0, 30)}” is not valid.`);
  const ex = compile(v.peModel, ['x', ...pn], 'Model expression'), sc = scopeOf({}, { x: 0 }), xs = data.map((r) => r.x), ys = data.map((r) => r.y);
  const model = (p, x) => { for (let i = 0; i < np; i++) sc[pn[i]] = p[i]; sc.x = x; return ex.evaluate(sc); };
  const lo = prm.map((r) => (isNum(r.lo) ? r.lo : -Infinity)), hi = prm.map((r, i) => (isNum(r.hi) && r.hi > lo[i] ? r.hi : Infinity)), p0 = prm.map((r, i) => clamp(r.value, lo[i], hi[i]));
  const resid = (p) => xs.map((x, i) => finite(model(p, x) - ys[i], 1e6)), fit = levenbergMarquardt(resid, p0, { lo, hi, maxIter: 200 }), p = fit.p, dof = n - np, s2 = fit.sse / dof, s = Math.sqrt(s2), tq = tQuant(dof);
  const okCov = !!fit.cov && fit.se.every((q) => Number.isFinite(q) && q > 0), se = fit.se.map((q, i) => (Number.isFinite(q) && q > 0 ? q : Math.abs(p[i]) * 0.1 + 1e-6));
  const corr = range(np).map((i) => range(np).map((j) => (okCov ? clamp(fit.cov[i][j] / (fit.se[i] * fit.se[j]), -1, 1) : i === j ? 1 : 0)));
  const jac = (x) => p.map((_, j) => { const d = 1e-6 * Math.max(1e-6, Math.abs(p[j])), q = [...p]; q[j] += d; return (model(q, x) - model(p, x)) / d; });
  const pred = xs.map((x) => model(p, x)), mt = metrics(ys, pred), xg = linspace(xs[0], xs[n - 1], 61), yg = xg.map((x) => model(p, x));
  const sdFit = xg.map((x) => { if (!okCov) return 0; const J = jac(x); return Math.sqrt(Math.max(0, dot(J, matVec(fit.cov, J)))); });
  // collinearity index of the scaled sensitivity matrix (Brun et al.): γ = 1/√λmin
  const Sm = xs.map(jac), nrm = range(np).map((j) => Math.sqrt(sum(Sm.map((r) => r[j] ** 2))) || 1), G = range(np).map((a) => range(np).map((b) => sum(Sm.map((r) => (r[a] / nrm[a]) * (r[b] / nrm[b])))));
  let gamma = 1e6;
  try { let w = new Array(np).fill(1 / Math.sqrt(np)), lamInv = 1; for (let it = 0; it < 60; it++) { const z = solveLinear(G.map((r, i) => r.map((q, j) => q + (i === j ? 1e-12 : 0))), w); lamInv = Math.hypot(...z); w = z.map((q) => q / lamInv); } gamma = Math.min(1e6, Math.sqrt(lamInv)); } catch { /* singular sensitivity matrix: parameters are not jointly identifiable */ }
  // profile likelihood: fix one parameter on a grid, re-fit the others, compare SSE with the 95 % threshold
  ctx.progress(0.3, 'Profile likelihood'); await ctx.tick();
  const thrSSE = fit.sse * (1 + (tq * tq) / dof), prof = range(np).map((i) => {
    const pts = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4].map((k) => {
      const pi = clamp(p[i] + k * se[i], lo[i], hi[i]), free = range(np).filter((j) => j !== i);
      if (!free.length) return [pi, sum(resid(p.map((q, j) => (j === i ? pi : q))).map((q) => q * q))];
      const r = levenbergMarquardt((q) => { const full = [...p]; full[i] = pi; free.forEach((j, a) => (full[j] = q[a])); return resid(full); }, free.map((j) => p[j]), { lo: free.map((j) => lo[j]), hi: free.map((j) => hi[j]), maxIter: 60 });
      return [pi, r.sse];
    });
    const left = pts.slice(0, 6).some((q) => q[1] > thrSSE), right = pts.slice(7).some((q) => q[1] > thrSSE);
    return { pts, left, right, ok: left && right };
  });
  const exact = fit.sse <= 1e-18 * sum(ys.map((q) => q * q)), seTxt = (i) => (okCov ? fmt(fit.se[i], 2) : 'n/a');
  const ident = range(np).map((i) => (exact && okCov ? 'well identified' : !okCov ? 'not identifiable' : !prof[i].ok ? 'practically non-identifiable (flat profile)' : se[i] / Math.max(1e-300, Math.abs(p[i])) < 0.25 ? 'well identified' : 'weakly identified'));
  // Bayesian posterior by random-walk Metropolis–Hastings (uniform priors inside the bounds, Jeffreys prior on σ)
  let mc = null;
  if (v.peBayes) {
    ctx.progress(0.55, 'Metropolis–Hastings sampling'); await ctx.tick();
    const nTot = clamp(Math.round(v.peMcmcN), 400, 200000), per = Math.floor(nTot / 2), burn = Math.floor(0.3 * per), d = np + 1, g = rng(Math.round(v.seed));
    const C = okCov ? cholesky(fit.cov.map((r, i) => r.map((q, j) => q + (i === j ? 1e-12 * Math.abs(q) + 1e-300 : 0)))) : null, Lp = C || range(np).map((i) => range(np).map((j) => (i === j ? se[i] : 0))), sc0 = 2.4 / Math.sqrt(d);
    const logPost = (th) => { for (let i = 0; i < np; i++) if (th[i] < lo[i] || th[i] > hi[i]) return -Infinity; const sg = Math.exp(th[np]), r = resid(th.slice(0, np)); return -(n + 1) * th[np] - sum(r.map((q) => q * q)) / (2 * sg * sg); };
    const chains = [];
    let acc = 0;
    for (let c = 0; c < 2; c++) {
      let th = [...p.map((q, i) => clamp(q + (c ? 1 : -1) * se[i], lo[i], hi[i])), Math.log(Math.max(s, 1e-12))], lp = logPost(th);
      const S = [];
      for (let it = 0; it < per; it++) {
        const zz = range(np).map(() => g.normal()), prop = th.map((q, i) => (i < np ? q + sc0 * sum(Lp[i].map((l, j) => l * zz[j])) : q + (sc0 * g.normal()) / Math.sqrt(2 * Math.max(1, dof)))), lq = logPost(prop);
        if (Math.log(1 - g.uniform()) < lq - lp) { th = prop; lp = lq; acc++; }
        S.push(th);
        if (it % 2000 === 1999) await ctx.tick();
      }
      chains.push(S);
    }
    const post = chains.flatMap((S) => S.slice(burn)), colOf = (S, j) => S.map((q) => (j < np ? q[j] : Math.exp(q[np])));
    const rhat = range(d).map((j) => { const a = colOf(chains[0].slice(burn), j), b = colOf(chains[1].slice(burn), j), Wv = 0.5 * (variance(a) + variance(b)), Bv = a.length * variance([mean(a), mean(b)]); return Wv > 0 ? Math.sqrt(((a.length - 1) / a.length) * 1 + Bv / (a.length * Wv)) : 1; });
    const st = range(d).map((j) => { const c = colOf(post, j); return { c, mean: mean(c), sd: std(c), lo: quantile(c, 0.025), med: quantile(c, 0.5), hi: quantile(c, 0.975) }; });
    const draw = range(200).map((q) => post[Math.floor(((q + 0.5) * post.length) / 200)]), bandLo = [], bandHi = [];
    for (const x of xg) { const c = draw.map((th) => model(th, x)).filter(Number.isFinite); bandLo.push(quantile(c, 0.025)); bandHi.push(quantile(c, 0.975)); }
    mc = { chains, st, rhat, acc: acc / (2 * per), per, burn, bandLo, bandHi };
  }
  const worst = range(np).filter((i) => ident[i] !== 'well identified'), W = [];
  if (!okCov) W.push({ level: 'bad', msg: 'The parameter covariance could not be computed: at least one parameter has no effect on the model or two parameters are fully redundant.' });
  if (worst.length) W.push({ level: 'warn', msg: `${worst.map((i) => pn[i]).join(', ')} ${worst.length > 1 ? 'are' : 'is'} not well identified by these data. Add observations where the model is sensitive to ${worst.length > 1 ? 'them' : 'it'}, or fix ${worst.length > 1 ? 'them' : 'it'} at an independent value.` });
  else W.push({ level: 'info', msg: exact ? 'The model reproduces the data exactly (noise-free data): the parameters are determined without statistical uncertainty.' : 'All parameters are well identified: finite two-sided profile-likelihood intervals and relative standard errors below 25 %.' });
  if (gamma > 15) W.push({ level: 'warn', msg: `Collinearity index ${fmt(gamma, 3)} is above 15: some parameters compensate each other (largest correlation ${fmt(Math.max(...corr.flatMap((r, i) => r.filter((_, j) => j !== i).map(Math.abs)), 0), 3)}).` });
  if (mc && Math.max(...mc.rhat) > 1.1) W.push({ level: 'warn', msg: `The two Markov chains have not mixed (largest R̂ = ${fmt(Math.max(...mc.rhat), 3)}): increase the number of samples.` });
  if (mc && (mc.acc < 0.1 || mc.acc > 0.6)) W.push({ level: 'info', msg: `Metropolis acceptance rate is ${fmt(100 * mc.acc, 3)} % (20–45 % is ideal); the posterior may be strongly non-Gaussian or bounded.` });
  const thin = (a, k = 400) => { const stp = Math.max(1, Math.ceil(a.length / k)); return a.filter((_, i) => i % stp === 0); };
  return {
    summary: `Fitted ${np} parameter${np > 1 ? 's' : ''} to ${n} points: ${pn.map((nm, i) => `${nm} = ${fmt(p[i], 4)} ± ${seTxt(i)}`).join(', ')}; RMSE ${fmt(mt.rmse, 3)}, R² ${fmt(finite(mt.r2), 4)}. ${worst.length ? `${worst.length} parameter${worst.length > 1 ? 's are' : ' is'} not well identified.` : 'All parameters are identifiable.'}`,
    warnings: W,
    kpis: [...pn.slice(0, 4).map((nm, i) => ({ label: `${nm} (± std. error)`, value: `${fmt(p[i], 5)} ± ${seTxt(i)}`, status: ident[i] === 'well identified' ? 'ok' : 'warn' })),
      { label: 'RMSE', value: mt.rmse }, { label: 'R²', value: finite(mt.r2), status: mt.r2 > 0.9 ? 'ok' : 'warn' }, { label: 'Residual standard deviation', value: s }, { label: 'Degrees of freedom', value: dof },
      { label: 'Collinearity index γ', value: gamma, status: gamma > 15 ? 'warn' : 'ok', help: 'Above about 15 the parameters cannot be estimated independently' }, { label: 'Identifiable parameters', value: `${np - worst.length} of ${np}`, status: worst.length ? 'warn' : 'ok' },
      ...(mc ? [{ label: 'MCMC acceptance rate', value: 100 * mc.acc, unit: '%' }, { label: 'Largest R̂ (chain mixing)', value: Math.max(...mc.rhat), status: Math.max(...mc.rhat) > 1.1 ? 'warn' : 'ok' }] : [])],
    recommendations: ['Report parameters with their confidence or credible intervals, never as bare numbers.', worst.length ? 'Design the next experiment where the confidence band is widest, or fix weakly identified parameters from independent measurements.' : 'Validate the fitted model on data that were not used here before using it for prediction.', mc ? 'Where the posterior histogram is skewed or cut by a bound, use the credible interval rather than the symmetric ± standard error.' : 'Enable the Bayesian option to check whether the Gaussian (± standard error) approximation holds.'],
    plots: [
      { type: 'line', title: 'Fit with 95 % confidence and prediction bands', xlabel: 'x', ylabel: 'y', series: [{ name: 'Data', x: xs, y: ys, mode: 'points' }, { name: 'Fitted model', x: xg, y: yg }, { name: 'Confidence band (low)', x: xg, y: yg.map((q, i) => q - tq * sdFit[i]), dash: true }, { name: 'Confidence band (high)', x: xg, y: yg.map((q, i) => q + tq * sdFit[i]), dash: true },
        { name: 'Prediction band (low)', x: xg, y: yg.map((q, i) => q - tq * Math.sqrt(sdFit[i] ** 2 + s2)), dash: true }, { name: 'Prediction band (high)', x: xg, y: yg.map((q, i) => q + tq * Math.sqrt(sdFit[i] ** 2 + s2)), dash: true }, ...(mc ? [{ name: 'Posterior 95 % band (low)', x: xg, y: mc.bandLo }, { name: 'Posterior 95 % band (high)', x: xg, y: mc.bandHi }] : [])] },
      { type: 'line', title: 'Residuals', xlabel: 'x', ylabel: 'Model − data', series: [{ name: 'Residual', x: xs, y: fit.residuals, mode: 'both' }], hlines: [{ y: 0, label: 'zero' }, { y: 2 * s, label: '+2σ' }, { y: -2 * s, label: '−2σ' }] },
      { type: 'line', title: 'Profile likelihood (sum of squares relative to the minimum)', xlabel: 'Parameter value ÷ estimate', ylabel: 'SSE / SSE min', series: range(np).map((i) => ({ name: pn[i], x: prof[i].pts.map((q) => (p[i] ? q[0] / p[i] : q[0])), y: prof[i].pts.map((q) => q[1] / Math.max(fit.sse, 1e-300)), mode: 'both' })), hlines: [{ y: 1 + (tq * tq) / dof, label: '95 % threshold' }] },
      ...(mc ? [{ type: 'line', title: 'Markov-chain traces', xlabel: 'Iteration', ylabel: 'Parameter ÷ least-squares estimate', series: range(Math.min(np, 3)).flatMap((j) => mc.chains.map((S, c) => ({ name: `${pn[j]} · chain ${c + 1}`, x: thin(range(mc.per)), y: thin(S.map((q) => (p[j] ? q[j] / p[j] : q[j]))) }))), vlines: [{ x: mc.burn, label: 'end of burn-in' }] },
        ...range(Math.min(np, 3)).map((j) => { const hgm = histogram(thin(mc.st[j].c, 20000), 24); return { type: 'bar', title: `Posterior of ${pn[j]}`, ylabel: 'Samples', categories: hgm.centers.map((q) => fmt(q, 4)), series: [{ name: pn[j], values: hgm.counts }], note: `Median ${fmt(mc.st[j].med, 5)}, 95 % credible interval ${fmt(mc.st[j].lo, 5)} – ${fmt(mc.st[j].hi, 5)}; least squares ${fmt(p[j], 5)}.` }; })] : []),
    ],
    tables: [
      { title: 'Estimated parameters', columns: ['Parameter', 'Start', 'Estimate', 'Std. error', 'Relative error (%)', '95 % CI low', '95 % CI high', 'Profile bounded below', 'Profile bounded above', 'Identifiability'],
        rows: range(np).map((i) => [pn[i], p0[i], p[i], okCov ? fit.se[i] : null, okCov ? (100 * fit.se[i]) / Math.max(1e-300, Math.abs(p[i])) : null, okCov ? p[i] - tq * fit.se[i] : null, okCov ? p[i] + tq * fit.se[i] : null, prof[i].left ? 'yes' : 'no', prof[i].right ? 'yes' : 'no', ident[i]]),
        note: `Model: y = ${ex.text}. Levenberg–Marquardt, ${fit.iterations} iterations, SSE ${fmt(fit.sse, 4)}, ${n} points. Confidence bands by the delta method, var(ŷ) = JᵀCJ.` },
      { title: 'Parameter correlation matrix', columns: ['', ...pn], rows: range(np).map((i) => [pn[i], ...corr[i]]) },
      ...(mc ? [{ title: 'Posterior summary (Metropolis–Hastings)', columns: ['Quantity', 'Posterior mean', 'Posterior std. dev.', '2.5 %', 'Median', '97.5 %', 'R̂', 'Least-squares value'], rows: [...range(np).map((j) => [pn[j], mc.st[j].mean, mc.st[j].sd, mc.st[j].lo, mc.st[j].med, mc.st[j].hi, mc.rhat[j], p[j]]), ['Noise σ', mc.st[np].mean, mc.st[np].sd, mc.st[np].lo, mc.st[np].med, mc.st[np].hi, mc.rhat[np], s]],
        note: `Two chains of ${mc.per} iterations from dispersed starts, first ${mc.burn} discarded; proposal covariance (2.4²/d)·C from the least-squares fit; acceptance ${fmt(100 * mc.acc, 3)} %.` }] : []),
      { title: 'Data and fit', columns: ['x', 'y measured', 'y model', 'Residual'], rows: xs.map((x, i) => [x, ys[i], pred[i], fit.residuals[i]]) },
    ],
    outputs: { task: 'pe', objective: fit.sse, rmse: mt.rmse, parameters: Object.fromEntries(pn.map((nm, i) => [nm, p[i]])), standardErrors: Object.fromEntries(pn.map((nm, i) => [nm, okCov ? fit.se[i] : 0])), identifiable: !worst.length, study: { a: p[0], b: mt.rmse } },
  };
}

// ======================================================================================================
// 9b · Further numerical engines: simplex and branch-and-bound, optimal control (Euler–Lagrange, Pontryagin),
//      neural ODE / neural PDE adjoints, proper orthogonal decomposition, bagging, co-kriging, worker pool
// ======================================================================================================
/**
 * Two-phase primal simplex on a dense tableau (Dantzig pricing, Bland's rule once the iteration count grows).
 * Minimises cᵀx subject to rows [{ a: [...], op: '<=' | '>=' | '=', b }] and x ≥ 0.
 * Returns { status: 'optimal' | 'infeasible' | 'unbounded' | 'iteration limit', x, f, dual (∂f/∂bᵢ per row), iterations }.
 */
export function lpSolve(c, rows, { maxIter } = {}) {
  const n = c.length, m = rows.length, eps = 1e-9;
  const sg = rows.map((r) => (r.b < 0 ? -1 : 1)), op = rows.map((r, i) => (sg[i] > 0 || r.op === '=' ? r.op : r.op === '<=' ? '>=' : '<='));
  let nc = n;
  const sCol = op.map((o) => (o === '=' ? -1 : nc++)), nReal = nc, aCol = op.map((o) => (o === '<=' ? -1 : nc++));
  const T = rows.map((r, i) => {
    const row = new Float64Array(nc + 1);
    for (let j = 0; j < n; j++) row[j] = sg[i] * (r.a[j] || 0);
    if (sCol[i] >= 0) row[sCol[i]] = op[i] === '<=' ? 1 : -1;
    if (aCol[i] >= 0) row[aCol[i]] = 1;
    row[nc] = sg[i] * r.b;
    return row;
  });
  const basis = op.map((o, i) => (o === '<=' ? sCol[i] : aCol[i])), z = new Float64Array(nc + 1), limit = maxIter ?? 60 * (m + nc) + 200, blandFrom = 8 * (m + nc) + 50;
  let iterations = 0;
  const price = (cost) => { // reduced costs z_j = c_j − c_Bᵀ·B⁻¹·a_j ; z[nc] = −objective
    z.fill(0);
    for (let j = 0; j < nc; j++) z[j] = cost[j];
    for (let i = 0; i < m; i++) { const cb = cost[basis[i]]; if (cb) { const Ti = T[i]; for (let j = 0; j <= nc; j++) z[j] -= cb * Ti[j]; } }
  };
  const pivot = (r, s) => {
    const Tr = T[r], p = Tr[s];
    for (let j = 0; j <= nc; j++) Tr[j] /= p;
    Tr[s] = 1;
    for (let i = 0; i < m; i++) { if (i === r) continue; const Ti = T[i], f = Ti[s]; if (f !== 0) { for (let j = 0; j <= nc; j++) Ti[j] -= f * Tr[j]; Ti[s] = 0; } }
    const f = z[s];
    if (f !== 0) { for (let j = 0; j <= nc; j++) z[j] -= f * Tr[j]; z[s] = 0; }
    basis[r] = s;
  };
  const iterate = (nEnter) => {
    for (;;) {
      if (iterations++ > limit) return 'iteration limit';
      const bland = iterations > blandFrom;
      let s = -1, best = -eps;
      for (let j = 0; j < nEnter; j++) if (z[j] < best) { s = j; if (bland) break; best = z[j]; }
      if (s < 0) return 'optimal';
      let r = -1, ratio = Infinity;
      for (let i = 0; i < m; i++) { const a = T[i][s]; if (a > eps) { const q = Math.max(0, T[i][nc]) / a; if (q < ratio - 1e-12 || (r >= 0 && q < ratio + 1e-12 && basis[i] < basis[r])) { ratio = q; r = i; } } }
      if (r < 0) return 'unbounded';
      pivot(r, s);
    }
  };
  const fail = (status) => ({ status, x: new Array(n).fill(0), f: NaN, dual: new Array(m).fill(0), iterations });
  if (nc > nReal) { // phase 1: minimise the sum of the artificial variables
    const c1 = new Float64Array(nc); for (let j = nReal; j < nc; j++) c1[j] = 1;
    price(c1);
    const st = iterate(nReal);
    if (st === 'iteration limit') return fail(st);
    if (-z[nc] > 1e-7 * (1 + Math.max(...rows.map((r) => Math.abs(r.b))))) return fail('infeasible');
    for (let i = 0; i < m; i++) if (basis[i] >= nReal) { let s = -1, big = 1e-9; for (let j = 0; j < nReal; j++) if (Math.abs(T[i][j]) > big) { big = Math.abs(T[i][j]); s = j; } if (s >= 0) pivot(i, s); } // drive degenerate artificials out
  }
  const c2 = new Float64Array(nc); for (let j = 0; j < n; j++) c2[j] = c[j];
  price(c2);
  const st = iterate(nReal);
  if (st !== 'optimal') return fail(st);
  const x = new Array(n).fill(0);
  for (let i = 0; i < m; i++) if (basis[i] < n) x[basis[i]] = Math.max(0, T[i][nc]);
  const dual = rows.map((_, i) => { const y = -z[aCol[i] >= 0 ? aCol[i] : sCol[i]] * sg[i]; return y === 0 ? 0 : y; });
  return { status: 'optimal', x, f: dot(c, x), dual, iterations };
}
/**
 * Mixed-integer linear programme by branch and bound on the LP relaxation (depth-first, better child first, bound pruning).
 * `ints` lists the indices of the integer variables. A rounding heuristic at the root (integers fixed to the rounded-up, then to
 * the nearest, relaxed values) supplies a first incumbent. The search stops at maxNodes or when the simplex work (pivots × tableau
 * size) exceeds maxWork. Returns { status, x, f, bound, gap, nodes, lpIterations, relaxation }.
 */
export function milpSolve(c, rows, ints, { maxNodes = 2000, maxWork = Infinity, tol = 1e-6 } = {}) {
  const n = c.length, unit = (j) => { const a = new Array(n).fill(0); a[j] = 1; return a; }, isInt = new Set(ints), size = (rows.length + 1) * (n + 2 * rows.length + 1);
  const root = lpSolve(c, rows);
  let nodes = 1, lpIterations = root.iterations, best = null;
  if (root.status !== 'optimal') return { status: root.status, x: root.x, f: NaN, bound: NaN, gap: NaN, nodes, lpIterations, relaxation: root };
  const clean = (x) => x.map((q, j) => (isInt.has(j) ? Math.round(q) : q)), worse = (f) => best && f >= best.f - 1e-9 * (1 + Math.abs(best.f));
  if (ints.some((j) => Math.abs(root.x[j] - Math.round(root.x[j])) > tol)) for (const round of [(q) => Math.ceil(q - tol), Math.round]) { // primal heuristic
    const lp = lpSolve(c, [...rows, ...ints.map((j) => ({ a: unit(j), op: '=', b: round(root.x[j]) }))]);
    nodes++; lpIterations += lp.iterations;
    if (lp.status === 'optimal' && !worse(lp.f)) best = { f: lp.f, x: clean(lp.x) };
  }
  const stack = [{ extra: [], lp: root }];
  while (stack.length && nodes < maxNodes && lpIterations * size < maxWork) {
    const nd = stack.pop();
    if (worse(nd.lp.f)) continue;
    let jb = -1, fr = tol;
    for (const j of ints) { const d = Math.abs(nd.lp.x[j] - Math.round(nd.lp.x[j])); if (d > fr) { fr = d; jb = j; } }
    if (jb < 0) { best = { f: nd.lp.f, x: clean(nd.lp.x) }; continue; }
    const xj = nd.lp.x[jb], kids = [];
    for (const row of [{ a: unit(jb), op: '<=', b: Math.floor(xj) }, { a: unit(jb), op: '>=', b: Math.ceil(xj) }]) {
      const extra = [...nd.extra, row], lp = lpSolve(c, [...rows, ...extra]);
      nodes++; lpIterations += lp.iterations;
      if (lp.status === 'optimal' && !worse(lp.f)) kids.push({ extra, lp });
    }
    kids.sort((a, b) => b.lp.f - a.lp.f);
    stack.push(...kids);
  }
  const open = stack.filter((k) => !worse(k.lp.f)).map((k) => k.lp.f);
  if (!best) return { status: open.length ? 'node limit' : 'infeasible', x: root.x, f: NaN, bound: root.f, gap: NaN, nodes, lpIterations, relaxation: root };
  const bound = open.length ? Math.min(best.f, ...open) : best.f;
  return { status: open.length ? 'node limit' : 'optimal', x: best.x, f: best.f, bound, gap: Math.abs(best.f - bound) / Math.max(1e-12, Math.abs(best.f)), nodes, lpIterations, relaxation: root };
}

/**
 * Pontryagin's principle as a two-point boundary-value problem solved by shooting on the initial costate.
 * prob = { x0, xT (entries null = free end, then λ(T) = 0), T, f(x, u) → ẋ, L(x, u) running cost, dHdx(x, λ, u) → ∂H/∂x with
 * H = L + λ·f, uStar(x, λ) → the control minimising H inside its bounds }. State, costate and cost are integrated with RK4.
 * For one state a bracket [λlo, λhi] may be given (Brent); otherwise a damped Newton iteration on λ(0) is used.
 */
export function pmpShoot(prob, { N = 100, lam0, bracket } = {}) {
  const n = prob.x0.length, h = prob.T / N;
  const rhs = (z) => { const x = z.slice(0, n), lam = z.slice(n, 2 * n), u = prob.uStar(x, lam), dx = prob.f(x, u), dl = prob.dHdx(x, lam, u); return [...dx, ...dl.map((q) => -q), prob.L(x, u)]; };
  const march = (l0, keep) => {
    let z = [...prob.x0, ...l0, 0];
    const path = keep ? [z] : null, ax = (a, b, s) => a.map((q, i) => q + s * b[i]);
    for (let k = 0; k < N; k++) {
      const k1 = rhs(z), k2 = rhs(ax(z, k1, h / 2)), k3 = rhs(ax(z, k2, h / 2)), k4 = rhs(ax(z, k3, h));
      z = z.map((q, i) => q + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
      if (keep) path.push(z);
    }
    return keep ? path : z;
  };
  const resid = (l0) => { const z = march(l0); return range(n).map((i) => (prob.xT[i] === null || prob.xT[i] === undefined ? z[n + i] : z[i] - prob.xT[i])); };
  let l0, converged = true, iterations = 0;
  if (n === 1 && bracket) {
    const fa = resid([bracket[0]])[0], fb = resid([bracket[1]])[0];
    if (!(fa * fb <= 0)) throw new Error('Optimal control: the terminal condition cannot be bracketed — the target is not reachable within the control bounds.');
    l0 = [brentRoot((q) => resid([q])[0], bracket[0], bracket[1], fa, fb)];
  } else {
    const r = newtonN(resid, lam0 || new Array(n).fill(0), { tol: 1e-11, maxIter: 60 });
    l0 = r.x; converged = r.converged; iterations = r.iterations;
  }
  const path = march(l0, true), t = range(N + 1).map((k) => k * h), x = path.map((z) => z.slice(0, n)), lam = path.map((z) => z.slice(n, 2 * n)), u = path.map((z, k) => prob.uStar(x[k], lam[k]));
  const H = path.map((z, k) => prob.L(x[k], u[k]) + dot(lam[k], prob.f(x[k], u[k])));
  return { t, x, lam, u, H, J: path[N][2 * n], lam0: l0, residual: maxAbs(resid(l0)), converged, iterations };
}
/** Root of a bracketed scalar function by bisection with secant acceleration (Illinois regula falsi); used where the function values at the ends are already known. */
function brentRoot(f, a, b, fa, fb, tol = 1e-13) {
  if (fa === 0) return a;
  if (fb === 0) return b;
  for (let it = 0; it < 200; it++) {
    let c = b - (fb * (b - a)) / (fb - fa);
    if (!(c > Math.min(a, b) && c < Math.max(a, b)) || it % 3 === 2) c = 0.5 * (a + b);
    const fc = f(c);
    if (fc === 0 || Math.abs(b - a) <= tol * (1 + Math.abs(c))) return c;
    if (fa * fc < 0) { b = c; fb = fc; } else { a = c; fa = fc; }
  }
  return 0.5 * (a + b);
}
/**
 * Euler–Lagrange boundary-value problem d/dt(∂L/∂ẋ) − ∂L/∂x = 0, x(0) = x0, x(T) = xT for one state, discretised with the
 * midpoint (variational) rule on N intervals and solved by Newton's method on the interior nodes.
 * Lv(x, v) = ∂L/∂ẋ and Lx(x, v) = ∂L/∂x. Returns { t, x, v (interval velocities), residual, converged }.
 */
export function eulerLagrangeBVP(Lv, Lx, x0, xT, T, N = 40) {
  const dt = T / N, full = (y) => [x0, ...y, xT];
  const F = (y) => {
    const X = full(y), lv = new Array(N), lx = new Array(N);
    for (let k = 0; k < N; k++) { const xm = 0.5 * (X[k] + X[k + 1]), v = (X[k + 1] - X[k]) / dt; lv[k] = Lv(xm, v); lx[k] = Lx(xm, v); }
    return range(N - 1).map((j) => (lv[j + 1] - lv[j]) / dt - 0.5 * (lx[j] + lx[j + 1]));
  };
  const r = newtonN(F, range(N - 1).map((j) => x0 + ((xT - x0) * (j + 1)) / N), { tol: 1e-11, maxIter: 60 }), X = full(r.x);
  return { t: range(N + 1).map((k) => k * dt), x: X, v: range(N).map((k) => (X[k + 1] - X[k]) / dt), residual: maxAbs(F(r.x)), converged: r.converged };
}

/** Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations. Returns { values (descending), vectors (one array per eigenvector) }. */
export function symEig(A0) {
  const n = A0.length, A = A0.map((r) => [...r]), V = eye(n);
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0, diag = 0;
    for (let i = 0; i < n; i++) { diag += A[i][i] * A[i][i]; for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j]; }
    if (off <= 1e-30 * (diag + off) || off === 0) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p][q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = range(n).sort((a, b) => A[b][b] - A[a][a]);
  return { values: order.map((i) => A[i][i]), vectors: order.map((i) => V.map((r) => r[i])) };
}
/**
 * Proper orthogonal decomposition by the method of snapshots: eigen-decomposition of the m × m snapshot covariance.
 * S = snapshots (m rows of length n). Returns { mean, modes (orthonormal, most energetic first), sigma (singular values), energy (cumulative fraction) }.
 */
export function podBasis(S, { center = true, rmax } = {}) {
  const m = S.length, n = S[0].length, mu = center ? range(n).map((j) => mean(S.map((r) => r[j]))) : zeros(n), F = S.map((r) => r.map((q, j) => q - mu[j]));
  const C = F.map((a) => F.map((b) => dot(a, b))), eg = symEig(C), tot = sum(eg.values.map((q) => Math.max(0, q))) || 1e-300, modes = [], sigma = [];
  for (let k = 0; k < Math.min(m, n, rmax ?? m); k++) {
    const lam = eg.values[k];
    if (!(lam > 1e-13 * eg.values[0])) break;
    const s = Math.sqrt(lam), phi = zeros(n);
    for (let i = 0; i < m; i++) { const w = eg.vectors[k][i] / s; for (let j = 0; j < n; j++) phi[j] += w * F[i][j]; }
    for (const prev of modes) { const d = dot(phi, prev); for (let j = 0; j < n; j++) phi[j] -= d * prev[j]; } // re-orthogonalise against rounding
    const nrm = Math.hypot(...phi) || 1;
    modes.push(phi.map((q) => q / nrm)); sigma.push(s);
  }
  let acc = 0;
  return { mean: mu, modes, sigma, energy: sigma.map((s) => (acc += (s * s) / tot)), total: tot };
}

/** Semi-discrete finite-volume operator of solveCDR: du/dt = A·u + s (A tri-diagonal: lo, di, hi; s acts on the first and last cell). */
export function cdrOperator({ L = 1, nx = 40, v = 0, D = 1, left, right, scheme = 'central' }) {
  const n = nx, dx = L / n, up = scheme === 'upwind', lo = zeros(n), di = zeros(n), hi = zeros(n);
  const aL = up ? (v >= 0 ? v + D / dx : D / dx) : v / 2 + D / dx, aR = up ? (v >= 0 ? -D / dx : v - D / dx) : v / 2 - D / dx;
  for (let i = 0; i < n; i++) { if (i < n - 1) { di[i] -= aL / dx; hi[i] -= aR / dx; } if (i > 0) { lo[i] += aL / dx; di[i] += aR / dx; } }
  if (left.type === 'dirichlet') di[0] -= (2 * D) / dx / dx; else if (left.type === 'neumann') di[0] += v / dx;
  if (right.type === 'dirichlet') di[n - 1] -= (2 * D) / dx / dx; else if (right.type === 'neumann') di[n - 1] -= v / dx;
  const s0 = left.type === 'dirichlet' ? ((v + (2 * D) / dx) * left.val) / dx : left.type === 'neumann' ? (-(v * left.val * dx) / 2 - D * left.val) / dx : 0;
  const s1 = right.type === 'dirichlet' ? (-(v - (2 * D) / dx) * right.val) / dx : right.type === 'neumann' ? -((v * right.val * dx) / 2 - D * right.val) / dx : 0;
  return { n, dx, lo, di, hi, s0, s1, x: range(n).map((i) => (i + 0.5) * dx), apply: (u) => u.map((q, i) => (i > 0 ? lo[i] * u[i - 1] : 0) + di[i] * q + (i < n - 1 ? hi[i] * u[i + 1] : 0) + (i === 0 ? s0 : 0) + (i === n - 1 ? s1 : 0)) };
}

// ---- Neural ODE: dy/dt = W2·tanh(W1·y + b1) + b2, RK4, discrete adjoint -----------------------------------------------------
const nodeLayout = (d, H) => ({ b1: H * d, W2: H * d + H, b2: H * d + H + d * H, n: H * d + H + d * H + d });
/** Seeded initial parameters of the neural right-hand side (flat array: W1, b1, W2, b2). */
export function nodeInit(d, H, seed = 1, scale = 0.5) {
  const g = rng(seed), o = nodeLayout(d, H), p = new Float64Array(o.n);
  for (let i = 0; i < H * d; i++) p[i] = g.normal(0, 1);
  for (let i = 0; i < H; i++) p[o.b1 + i] = g.normal(0, 0.5);
  for (let i = 0; i < d * H; i++) p[o.W2 + i] = g.normal(0, scale / Math.sqrt(H));
  return p;
}
/** Evaluate the network: writes tanh activations into z (length H) and dy/dt into out (length d). */
export function nodeF(p, d, H, y, z, out) {
  const o = nodeLayout(d, H);
  for (let i = 0; i < H; i++) { let s = p[o.b1 + i]; for (let j = 0; j < d; j++) s += p[i * d + j] * y[j]; z[i] = Math.tanh(s); }
  for (let k = 0; k < d; k++) { let s = p[o.b2 + k]; for (let i = 0; i < H; i++) s += p[o.W2 + k * H + i] * z[i]; out[k] = s; }
  return out;
}
/** Vector–Jacobian product of the network: adds aᵀ·∂f/∂θ to g and returns aᵀ·∂f/∂y (z = activations stored by nodeF at y). */
export function nodeVJP(p, d, H, y, z, a, g) {
  const o = nodeLayout(d, H), gy = new Array(d).fill(0);
  for (let k = 0; k < d; k++) { g[o.b2 + k] += a[k]; for (let i = 0; i < H; i++) g[o.W2 + k * H + i] += a[k] * z[i]; }
  for (let i = 0; i < H; i++) {
    let b = 0;
    for (let k = 0; k < d; k++) b += p[o.W2 + k * H + i] * a[k];
    b *= 1 - z[i] * z[i];
    g[o.b1 + i] += b;
    for (let j = 0; j < d; j++) { g[i * d + j] += b * y[j]; gy[j] += p[i * d + j] * b; }
  }
  return gy;
}
/** Integrate the neural ODE from y0 through the times ts (RK4 with `sub` steps per interval); returns the state at every time. */
export function nodeSolve(p, d, H, y0, ts, sub = 2) {
  const out = [[...y0]], z = new Array(H), k1 = new Array(d), k2 = new Array(d), k3 = new Array(d), k4 = new Array(d), w = new Array(d);
  let y = [...y0];
  for (let q = 1; q < ts.length; q++) {
    const h = (ts[q] - ts[q - 1]) / sub;
    for (let s = 0; s < sub; s++) {
      nodeF(p, d, H, y, z, k1);
      for (let j = 0; j < d; j++) w[j] = y[j] + 0.5 * h * k1[j];
      nodeF(p, d, H, w, z, k2);
      for (let j = 0; j < d; j++) w[j] = y[j] + 0.5 * h * k2[j];
      nodeF(p, d, H, w, z, k3);
      for (let j = 0; j < d; j++) w[j] = y[j] + h * k3[j];
      nodeF(p, d, H, w, z, k4);
      y = y.map((v, j) => v + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    }
    out.push([...y]);
  }
  return out;
}
/**
 * Mean squared trajectory error of the neural ODE and its exact gradient by back-propagation through the RK4 integrator
 * (discrete adjoint). trajs = [{ t: [...], y: [[...], ...] }]; y[0] is the initial condition, the later rows are fitted.
 */
export function nodeLossGrad(p, d, H, trajs, sub = 2) {
  const o = nodeLayout(d, H), g = new Float64Array(p.length);
  let nObs = 0, maxS = 0, loss = 0;
  for (const tr of trajs) { nObs += (tr.t.length - 1) * d; maxS = Math.max(maxS, (tr.t.length - 1) * sub); }
  const Y = new Float64Array(maxS * 4 * d), Z = new Float64Array(maxS * 4 * H), hs = new Float64Array(maxS), E = new Float64Array(maxS * d), has = new Uint8Array(maxS); // tape: stage inputs and activations of every step
  const k = new Float64Array(4 * d), y = new Float64Array(d), a = new Float64Array(d), abar = new Float64Array(d), gy = new Float64Array(d), acc = new Float64Array(d);
  const fwd = (yo, zo, ko) => {
    for (let i = 0; i < H; i++) { let q = p[o.b1 + i]; for (let j = 0; j < d; j++) q += p[i * d + j] * Y[yo + j]; Z[zo + i] = Math.tanh(q); }
    for (let c = 0; c < d; c++) { let q = p[o.b2 + c]; for (let i = 0; i < H; i++) q += p[o.W2 + c * H + i] * Z[zo + i]; k[ko + c] = q; }
  };
  const vjp = (yo, zo) => { // abarᵀ·∂f/∂θ is added to g, abarᵀ·∂f/∂y is written to gy
    gy.fill(0);
    for (let c = 0; c < d; c++) { const ac = abar[c]; g[o.b2 + c] += ac; for (let i = 0; i < H; i++) g[o.W2 + c * H + i] += ac * Z[zo + i]; }
    for (let i = 0; i < H; i++) {
      let b = 0;
      for (let c = 0; c < d; c++) b += p[o.W2 + c * H + i] * abar[c];
      const z = Z[zo + i]; b *= 1 - z * z; g[o.b1 + i] += b;
      for (let j = 0; j < d; j++) { g[i * d + j] += b * Y[yo + j]; gy[j] += p[i * d + j] * b; }
    }
  };
  for (const tr of trajs) {
    let s = 0;
    for (let j = 0; j < d; j++) y[j] = tr.y[0][j];
    has.fill(0);
    for (let q = 1; q < tr.t.length; q++) {
      const h = (tr.t[q] - tr.t[q - 1]) / sub;
      for (let c = 0; c < sub; c++, s++) {
        const y0 = s * 4 * d, z0 = s * 4 * H;
        hs[s] = h;
        for (let j = 0; j < d; j++) Y[y0 + j] = y[j];
        fwd(y0, z0, 0);
        for (let j = 0; j < d; j++) Y[y0 + d + j] = y[j] + 0.5 * h * k[j];
        fwd(y0 + d, z0 + H, d);
        for (let j = 0; j < d; j++) Y[y0 + 2 * d + j] = y[j] + 0.5 * h * k[d + j];
        fwd(y0 + 2 * d, z0 + 2 * H, 2 * d);
        for (let j = 0; j < d; j++) Y[y0 + 3 * d + j] = y[j] + h * k[2 * d + j];
        fwd(y0 + 3 * d, z0 + 3 * H, 3 * d);
        for (let j = 0; j < d; j++) y[j] += (h / 6) * (k[j] + 2 * k[d + j] + 2 * k[2 * d + j] + k[3 * d + j]);
      }
      has[s - 1] = 1;
      for (let j = 0; j < d; j++) { const e = y[j] - tr.y[q][j]; E[(s - 1) * d + j] = e; loss += (e * e) / nObs; }
    }
    a.fill(0);
    for (s -= 1; s >= 0; s--) { // reverse sweep: adjoint of every RK4 stage
      if (has[s]) for (let j = 0; j < d; j++) a[j] += (2 * E[s * d + j]) / nObs;
      const h = hs[s], y0 = s * 4 * d, z0 = s * 4 * H;
      for (let j = 0; j < d; j++) { acc[j] = a[j]; abar[j] = (h / 6) * a[j]; }
      vjp(y0 + 3 * d, z0 + 3 * H);
      for (let j = 0; j < d; j++) { acc[j] += gy[j]; abar[j] = (h / 3) * a[j] + h * gy[j]; }
      vjp(y0 + 2 * d, z0 + 2 * H);
      for (let j = 0; j < d; j++) { acc[j] += gy[j]; abar[j] = (h / 3) * a[j] + 0.5 * h * gy[j]; }
      vjp(y0 + d, z0 + H);
      for (let j = 0; j < d; j++) { acc[j] += gy[j]; abar[j] = (h / 6) * a[j] + 0.5 * h * gy[j]; }
      vjp(y0, z0);
      for (let j = 0; j < d; j++) a[j] = acc[j] + gy[j];
    }
  }
  return { loss, grad: g };
}
/** Train the neural ODE with Adam on the trajectory loss; the best parameters seen are returned. */
export function nodeTrain(d, H, trajs, { epochs = 600, lr = 0.02, seed = 1, sub = 2, p0 } = {}) {
  const p = p0 ? Float64Array.from(p0) : nodeInit(d, H, seed), st = { t: 0, m: new Float64Array(p.length), v: new Float64Array(p.length) }, history = { epoch: [], loss: [] }, every = Math.max(1, Math.floor(epochs / 120));
  let best = Infinity, bestP = Float64Array.from(p);
  for (let ep = 0; ep < epochs; ep++) {
    const { loss, grad } = nodeLossGrad(p, d, H, trajs, sub);
    if (!Number.isFinite(loss)) break;
    if (loss < best) { best = loss; bestP.set(p); }
    if (ep % every === 0) { history.epoch.push(ep); history.loss.push(loss); }
    adamStep(p, grad, st, lr * 0.05 ** (ep / epochs));
  }
  const fin = nodeLossGrad(p, d, H, trajs, sub).loss;
  if (fin < best) { best = fin; bestP.set(p); }
  return { p: bestP, loss: best, history, nParams: p.length };
}

// ---- Neural PDE: convection–diffusion with a neural reaction closure R_θ(u), IMEX θ-scheme, discrete adjoint -----------------
/**
 * March ∂u/∂t + v ∂u/∂x = D ∂²u/∂x² + R(u) with the transport part implicit (θ-scheme on the operator of cdrOperator) and the
 * closure explicit: (I − θΔt·A)·uⁿ⁺¹ = (I + (1 − θ)Δt·A)·uⁿ + Δt·s + Δt·R(uⁿ). clos(u) → R; returns every time level.
 */
export function imexMarch(op, u0, nt, dt, clos, theta = 0.5) {
  const n = op.n, a = op.lo.map((q) => -theta * dt * q), b = op.di.map((q) => 1 - theta * dt * q), c = op.hi.map((q) => -theta * dt * q), U = [[...u0]];
  let u = [...u0];
  for (let k = 0; k < nt; k++) {
    const rhs = new Array(n);
    for (let i = 0; i < n; i++) rhs[i] = u[i] + (1 - theta) * dt * ((i > 0 ? op.lo[i] * u[i - 1] : 0) + op.di[i] * u[i] + (i < n - 1 ? op.hi[i] * u[i + 1] : 0)) + dt * clos(u[i]);
    rhs[0] += dt * op.s0; rhs[n - 1] += dt * op.s1;
    u = tridiag(a, b, c, rhs); U.push(u);
  }
  return U;
}
/**
 * Loss (mean squared mismatch to the observed snapshots) of the PDE with the neural closure R_θ(u) = rRef·net(u/uRef) and its gradient
 * by the discrete adjoint of the IMEX scheme. exps = [{ op, u0, nt, dt, obs: [{ k (time level), u: [...] }] }].
 */
export function npdeLossGrad(p, H, exps, { uRef = 1, rRef = 1, theta = 0.5 } = {}) {
  const g = new Float64Array(p.length), nObs = sum(exps.map((e) => sum(e.obs.map((q) => q.u.length))));
  let loss = 0;
  for (const ex of exps) {
    const { op, nt, dt } = ex, n = op.n, at = new Map(ex.obs.map((q) => [q.k, q.u])), Z = new Float64Array(nt * n * H), U = [[...ex.u0]];
    const a = op.lo.map((q) => -theta * dt * q), b = op.di.map((q) => 1 - theta * dt * q), c = op.hi.map((q) => -theta * dt * q);
    for (let k = 0; k < nt; k++) { // forward march (same scheme as imexMarch) keeping the hidden activations for the reverse sweep
      const u = U[k], rhs = new Array(n);
      for (let i = 0; i < n; i++) {
        const sIn = u[i] / uRef, zo = (k * n + i) * H;
        let f = p[3 * H];
        for (let j = 0; j < H; j++) { const z = Math.tanh(p[j] * sIn + p[H + j]); Z[zo + j] = z; f += p[2 * H + j] * z; }
        rhs[i] = u[i] + (1 - theta) * dt * ((i > 0 ? op.lo[i] * u[i - 1] : 0) + op.di[i] * u[i] + (i < n - 1 ? op.hi[i] * u[i + 1] : 0)) + dt * rRef * f;
      }
      rhs[0] += dt * op.s0; rhs[n - 1] += dt * op.s1;
      U.push(tridiag(a, b, c, rhs));
    }
    const aT = range(n).map((i) => (i > 0 ? c[i - 1] : 0)), cT = range(n).map((i) => (i < n - 1 ? a[i + 1] : 0)); // transposed tri-diagonal
    const dLoss = (k) => { const d = at.get(k); if (!d) return null; return U[k].map((q, i) => { const e = q - d[i]; loss += (e * e) / nObs; return (2 * e) / nObs; }); };
    let ub = dLoss(nt) || zeros(n);
    for (let k = nt - 1; k >= 0; k--) {
      const w = tridiag(aT, b, cT, ub), un = U[k], nb = new Array(n);
      for (let i = 0; i < n; i++) {
        const sIn = un[i] / uRef, ai = dt * rRef * w[i], zo = (k * n + i) * H;
        let gy = 0;
        g[3 * H] += ai;
        for (let j = 0; j < H; j++) { const z = Z[zo + j], bj = p[2 * H + j] * ai * (1 - z * z); g[2 * H + j] += ai * z; g[H + j] += bj; g[j] += bj * sIn; gy += p[j] * bj; }
        nb[i] = w[i] + (1 - theta) * dt * ((i > 0 ? op.hi[i - 1] * w[i - 1] : 0) + op.di[i] * w[i] + (i < n - 1 ? op.lo[i + 1] * w[i + 1] : 0)) + gy / uRef;
      }
      const dl = k > 0 ? dLoss(k) : null;
      if (dl) for (let i = 0; i < n; i++) nb[i] += dl[i];
      ub = nb;
    }
  }
  return { loss, grad: g };
}
/** The scalar closure R_θ(u) = rRef·[b₂ + Σ W₂ⱼ·tanh(W₁ⱼ·u/uRef + b₁ⱼ)] for the parameter layout of nodeInit(1, H). */
export function npdeClosure(p, H, uRef = 1, rRef = 1) {
  return (u) => { const s = u / uRef; let f = p[3 * H]; for (let j = 0; j < H; j++) f += p[2 * H + j] * Math.tanh(p[j] * s + p[H + j]); return rRef * f; };
}

// ---- Ensembles and multi-fidelity surrogates ----------------------------------------------------------------------------------
/**
 * Bootstrap aggregation (bagging). fit(X, y, memberIndex) → predict(x). Every member is trained on a bootstrap resample; the rows it
 * did not see give the out-of-bag prediction of each training point. Returns { members, predict(x) → { mean, sd, all }, oob: { pred, rmse, n } }.
 */
export function bagEnsemble(X, y, fit, { B = 12, seed = 1 } = {}) {
  const n = X.length, g = rng(seed), members = [], inBag = [];
  for (let b = 0; b < B; b++) {
    const idx = range(n).map(() => g.int(n)), seen = new Uint8Array(n);
    for (const i of idx) seen[i] = 1;
    members.push(fit(idx.map((i) => X[i]), idx.map((i) => y[i]), b)); inBag.push(seen);
  }
  const predict = (x, upTo = B) => { const all = members.slice(0, upTo).map((f) => f(x)), mu = mean(all); return { mean: mu, sd: all.length > 1 ? std(all) : 0, all }; };
  const pred = range(n).map((i) => { const out = members.filter((_, b) => !inBag[b][i]).map((f) => f(X[i])); return out.length ? mean(out) : null; }), ok = range(n).filter((i) => pred[i] !== null);
  return { members, predict, B, oob: { pred, n: ok.length, rmse: ok.length ? Math.sqrt(mean(ok.map((i) => (pred[i] - y[i]) ** 2))) : NaN } };
}
/**
 * Two-level recursive co-kriging (Kennedy–O'Hagan autoregressive model): y_high(x) = ρ·y_low(x) + δ(x). A Gaussian process is fitted
 * to the low-fidelity data, ρ follows from least squares at the high-fidelity points and a second process models the discrepancy δ.
 */
export function coKrige(XL, yL, XH, yH) {
  const gpL = gpFit(XL, yL), muL = XH.map((x) => gpL.predict(x).mean), n = XH.length, d = XH[0].length, sx = standardizer(XH);
  const slope = (m, y) => { const a = mean(m), b = mean(y), sxx = sum(m.map((q) => (q - a) ** 2)); return sxx > 1e-300 ? sum(m.map((q, i) => (q - a) * (y[i] - b))) / sxx : 1; };
  const kinds = { // candidate forms of the discrepancy δ(x); the one with the smallest cross-validation error is kept
    constant: (m, X, y) => { const rho = slope(m, y), c = mean(y.map((q, i) => q - rho * m[i])); return { rho, delta: () => ({ mean: c, sd: 0 }) }; },
    'Gaussian process': (m, X, y) => { const rho = slope(m, y), del = y.map((q, i) => q - rho * m[i]); if (std(del) < 1e-12 * (1 + Math.abs(mean(del)))) return kinds.constant(m, X, y); const g = gpFit(X, del, { maxIter: 120 }); return { rho, delta: (x) => g.predict(x) }; },
    'linear trend': (m, X, y) => { const beta = lstsq(X.map((x, i) => [m[i], 1, ...sx.f(x)]), y); return { rho: beta[0], delta: (x) => ({ mean: beta[1] + dot(sx.f(x), beta.slice(2)), sd: 0 }) }; },
  };
  const names = ['constant', 'Gaussian process', ...(n >= d + 4 ? ['linear trend'] : [])], K = Math.min(n, 8), cv = {};
  for (const nm of names) {
    let sq = 0;
    for (let f = 0; f < K; f++) {
      const tr = range(n).filter((i) => i % K !== f), te = range(n).filter((i) => i % K === f);
      try { const M = kinds[nm](tr.map((i) => muL[i]), tr.map((i) => XH[i]), tr.map((i) => yH[i])); for (const i of te) sq += (M.rho * muL[i] + M.delta(XH[i]).mean - yH[i]) ** 2; } catch { sq = Infinity; }
    }
    cv[nm] = Number.isFinite(sq) ? Math.sqrt(sq / n) : Infinity;
  }
  const kind = names.reduce((a, b) => (cv[b] < cv[a] ? b : a)), M = kinds[kind](muL, XH, yH), floor = finite(cv[kind]);
  const predict = (x) => { const a = gpL.predict(x), b = M.delta(x); return { mean: M.rho * a.mean + b.mean, sd: Math.hypot(M.rho * a.sd, Math.max(b.sd, floor)), low: a.mean, delta: b.mean }; };
  return { predict, rho: M.rho, kind, cv, gpL };
}
/**
 * Sample-average form of the joint chance constraint P[g(x, ξ) ≤ 0] ≥ 1 − α: the ⌊α·m⌋ worst of the m scenario values are set aside
 * and the remaining ones must be feasible. α = 0 is the worst-case (robust) constraint. Returns the violation of the kept scenarios.
 */
export function chanceViolation(g, alpha = 0) {
  const kept = [...g].sort((a, b) => b - a).slice(Math.floor(alpha * g.length + 1e-9));
  return { viol: sum(kept.map((q) => Math.max(0, q))), quad: sum(kept.map((q) => Math.max(0, q) ** 2)), dropped: g.length - kept.length };
}

// ---- Worker-thread pool for batches of plant simulations ----------------------------------------------------------------------
const stripEval = (e) => (e.ok ? { ok: true, m: e.m, cost: e.cost, g: e.g, viol: e.viol } : { ok: false, g: e.g, viol: e.viol, err: e.err });
/** Evaluate one chunk of designs (overrides of the base RO inputs); the pressure warm start is reset so a chunk gives the same numbers on any thread. */
function evalJobs(jobs, v, base) { warmReset(); return jobs.map((j) => stripEval(evalDesign({ ...base, ...j }, v))); }
function parHandle(msg) {
  if (!msg || msg.brinelab !== 'par') return null;
  if (msg.type === 'ping') return { brinelab: 'par', type: 'pong', id: msg.id };
  if (msg.type !== 'eval') return null;
  try { return { brinelab: 'par', type: 'done', id: msg.id, out: evalJobs(msg.jobs, msg.v, roBase(msg.v)) }; } catch (e) { return { brinelab: 'par', type: 'fail', id: msg.id, err: String(e?.message || e) }; }
}
const nodeThreads = () => { try { return globalThis.process?.getBuiltinModule?.('node:worker_threads') || null; } catch { return null; } };
/**
 * Start `n` worker threads that load this module (Node worker_threads, or module Web Workers when the page is served over http/https).
 * Resolves to { live, workers, reason, startMs, run(chunks, v) → results per chunk, close() }. When threads cannot be started the pool
 * reports live = false and the caller evaluates the same chunks on the calling thread.
 */
export async function parPool(n, { timeout = 6000 } = {}) {
  const want = clamp(Math.round(n) || 0, 0, 16), t0 = Date.now(), url = import.meta.url, ws = [], pending = new Map();
  const dead = (reason) => { for (const w of ws) { try { w.kill(); } catch { /* already gone */ } } return { live: false, workers: 0, reason, startMs: Date.now() - t0, run: null, close() {} }; };
  if (want < 1) return dead('parallel evaluation switched off');
  let nextId = 1;
  const settle = (m) => { const p = pending.get(m?.id); if (!p) return; pending.delete(m.id); clearTimeout(p.timer); if (m.type === 'fail') p.reject(new Error(m.err)); else p.resolve(m); };
  const failAll = (err) => { for (const [id, p] of pending) { clearTimeout(p.timer); p.reject(err); pending.delete(id); } };
  const call = (w, msg, ms) => new Promise((resolve, reject) => {
    const id = nextId++, timer = setTimeout(() => { pending.delete(id); reject(new Error('worker timed out')); }, ms);
    pending.set(id, { resolve, reject, timer });
    try { w.post({ ...msg, brinelab: 'par', id }); } catch (e) { clearTimeout(timer); pending.delete(id); reject(e); }
  });
  try {
    const wt = nodeThreads();
    for (let i = 0; i < want; i++) {
      if (wt && /^file:/.test(url || '')) {
        const w = new wt.Worker(new URL(url), { workerData: { brinelabPar: 1 } });
        w.on('message', settle); w.on('error', failAll);
        ws.push({ post: (m) => w.postMessage(m), kill: () => w.terminate() });
      } else if (typeof Worker === 'function' && /^https?:/.test(url || '')) {
        const w = new Worker(url, { type: 'module' });
        w.onmessage = (e) => settle(e.data); w.onerror = () => failAll(new Error('worker failed to load'));
        ws.push({ post: (m) => w.postMessage(m), kill: () => w.terminate() });
      } else return dead('worker threads are not available in this environment');
    }
    await Promise.all(ws.map((w) => call(w, { type: 'ping' }, timeout)));
  } catch (e) { failAll(new Error('pool closed')); return dead('worker threads could not be started (' + String(e?.message || e).slice(0, 80) + ')'); }
  return { live: true, workers: ws.length, reason: '', startMs: Date.now() - t0,
    run: (chunks, v) => Promise.all(chunks.map((jobs, i) => (jobs.length ? call(ws[i % ws.length], { type: 'eval', jobs, v }, 120000).then((m) => m.out) : Promise.resolve([])))),
    close() { failAll(new Error('pool closed')); for (const w of ws) { try { w.kill(); } catch { /* already gone */ } } } };
}
/** Batched evaluator: splits each batch into a fixed number of contiguous chunks and runs them on the pool, or on this thread with cooperative yielding. */
async function makeBatcher(v, base, nWorkers, ctx) {
  const chunks = Math.max(1, clamp(Math.round(nWorkers) || 0, 0, 16)), vLite = JSON.parse(JSON.stringify(v)), pool = await parPool(nWorkers);
  const st = { pool, chunks, jobs: 0, wall: 0, seqRate: null, agree: null, fellBack: false };
  const split = (jobs) => range(chunks).map((c) => jobs.slice(Math.floor((c * jobs.length) / chunks), Math.floor(((c + 1) * jobs.length) / chunks)));
  st.run = async (jobs) => {
    const parts = split(jobs), t0 = Date.now();
    let out = null;
    if (pool.live && !st.fellBack) { try { out = await pool.run(parts, vLite); } catch { st.fellBack = true; } }
    if (!out) { out = []; for (const p of parts) { out.push(evalJobs(p, v, base)); await ctx.tick(); } }
    st.wall += Date.now() - t0; st.jobs += jobs.length;
    if (pool.live && !st.fellBack && st.seqRate === null && parts[0].length) { // time the first chunk on this thread: sequential rate and a check that both threads agree
      const t1 = Date.now(), ref = evalJobs(parts[0], v, base);
      st.seqRate = (Date.now() - t1) / parts[0].length;
      st.agree = Math.max(0, ...ref.map((e, i) => (e.ok && out[0][i].ok ? Math.abs(e.m.sec - out[0][i].m.sec) + Math.abs(e.cost.total - out[0][i].cost.total) : e.ok === out[0][i].ok ? 0 : 1)));
    }
    return out.flat();
  };
  st.speedup = () => (st.seqRate !== null && st.wall > 0 ? (st.seqRate * st.jobs) / st.wall : null);
  return st;
}

// ---- Task 11: optimal control of a batch RO cycle (Euler–Lagrange, Hamiltonian system, Pontryagin) -----------------------------
/** Batch RO cycle: tank volume V (m³) falls at the permeate rate q (m³/h); L = P·q/(36·η) is the pumping power (kW). */
function batchModel(v) {
  const V0 = Math.max(1e-6, v.ocV0), Vf = V0 * (1 - clamp(v.ocRec, 1, 95) / 100), T = Math.max(1e-6, v.ocT), Am = Math.max(1e-6, v.ocArea), a = (Am * Math.max(1e-6, v.ocA)) / 1000, qk = (Am * Math.max(0.1, v.ocK) * 3.6) / 1000;
  const eta = clamp(v.ocEta, 5, 100) / 100, pi0 = Math.max(0, v.ocPi0), dp = Math.max(0, v.ocDp), qmax = (Am * Math.max(0.1, v.ocFluxMax)) / 1000, w = 1 / (36 * eta), Vlow = 0.2 * Vf;
  const pi = (V) => (pi0 * V0) / Math.max(V, Vlow);
  const P = (V, q) => q / a + pi(V) * Math.exp(q / qk) + dp;
  const L = (V, q) => w * P(V, q) * q;
  const Lq = (V, q) => w * ((2 * q) / a + pi(V) * Math.exp(q / qk) * (1 + q / qk) + dp);
  const Lqq = (V, q) => w * (2 / a + (pi(V) * Math.exp(q / qk) * (2 + q / qk)) / qk);
  const LV = (V, q) => (V > Vlow ? -w * ((pi0 * V0) / (V * V)) * q * Math.exp(q / qk) : 0);
  /** Control minimising the Hamiltonian H = L − λ·q on [0, qcap]: stationarity ∂L/∂q = λ (Newton from above, L is convex in q), then clipping. */
  const qStar = (V, lam, qcap) => {
    if (Lq(V, 0) >= lam) return 0;
    if (Lq(V, qcap) <= lam) return qcap;
    let q = qcap;
    for (let it = 0; it < 60; it++) { const dq = (Lq(V, q) - lam) / Lqq(V, q); q -= dq; if (Math.abs(dq) < 1e-14 * (1 + q)) break; }
    return clamp(q, 0, qcap);
  };
  /** Permeate rate at a fixed feed pressure (inverse of P(V, q)). */
  const qOfP = (V, Pf) => { if (P(V, 0) >= Pf) return 0; let q = Math.min(qmax, (Pf - dp) * a); for (let it = 0; it < 60; it++) { const dq = (P(V, q) - Pf) / (1 / a + (pi(V) * Math.exp(q / qk)) / qk); q -= dq; if (Math.abs(dq) < 1e-14 * (1 + q)) break; } return clamp(q, 0, qmax); };
  return { V0, Vf, T, Am, a, qk, eta, pi0, dp, qmax, w, pi, P, L, Lq, Lqq, LV, qStar, qOfP, flux: (q) => (1000 * q) / Am };
}
/** Pontryagin solution of the batch cycle with the permeate rate limited to qcap. */
function batchPMP(B, N, qcap) {
  const prob = { x0: [B.V0], xT: [B.Vf], T: B.T, f: (x, u) => [-u[0]], L: (x, u) => B.L(x[0], u[0]), dHdx: (x, lam, u) => [B.LV(x[0], u[0])], uStar: (x, lam) => [B.qStar(x[0], lam[0], qcap)] };
  return pmpShoot(prob, { N, bracket: [B.Lq(B.V0, 0), 1.05 * B.Lq(0.9 * B.Vf, qcap) + 1e-9] });
}
/** Direct transcription of the same problem: piecewise-constant permeate rates as variables of a nonlinear programme solved by SQP. */
function batchDirect(B, Nd, qcap, q0) {
  const dt = B.T / Nd, dV = B.V0 - B.Vf;
  const cost = (u) => { let V = B.V0, J = 0; for (let k = 0; k < Nd; k++) { const q = u[k] * qcap; J += dt * B.L(V - 0.5 * q * dt, q); V -= q * dt; } return J; };
  const J0 = cost(q0.map((q) => q / qcap)) || 1, h = (u) => (sum(u) * qcap * dt - dV) / dV;
  const r = sqpLite((u) => cost(u) / J0, (u) => { const e = h(u); return [e, -e]; }, q0.map((q) => clamp(q / qcap, 0, 1)), { lo: zeros(Nd), hi: new Array(Nd).fill(1), maxIter: 80, tol: 1e-8, h: 1e-6 });
  return { q: r.x.map((u) => u * qcap), J: cost(r.x), dt, eqErr: Math.abs(h(r.x)), iterations: r.iterations, kkt: r.kkt, lambda: (r.lambda[1] - r.lambda[0]) * J0 / dV, evals: r.evals };
}
const lerpAt = (ts, ys, t) => { if (t <= ts[0]) return ys[0]; const n = ts.length; if (t >= ts[n - 1]) return ys[n - 1]; let i = 1; while (ts[i] < t) i++; const f = (t - ts[i - 1]) / (ts[i] - ts[i - 1]); return ys[i - 1] + f * (ys[i] - ys[i - 1]); };
async function taskOC(v, ctx) {
  const B = batchModel(v), N = clamp(Math.round(v.ocN), 10, 2000), dV = B.V0 - B.Vf, qAvg = dV / B.T;
  if (B.qmax * B.T < dV * 1.0005) throw new Error(`The flux limit of ${fmt(v.ocFluxMax)} L/m²·h cannot deliver ${fmt(dV, 4)} m³ of permeate in ${fmt(B.T)} h (average flux needed ${fmt(B.flux(qAvg), 4)} L/m²·h): raise the limit, the area or the cycle time.`);
  ctx.progress(0.1, 'Pontryagin boundary-value problem (shooting on the costate)'); await ctx.tick();
  const qFree = Math.max(20 * qAvg, 2 * B.qmax), pm = batchPMP(B, N, B.qmax), free = batchPMP(B, N, qFree);
  ctx.progress(0.4, 'Euler–Lagrange equation (finite differences, Newton)'); await ctx.tick();
  const Nel = Math.min(N, 80), el = eulerLagrangeBVP((V, vd) => -B.Lq(V, -vd), (V, vd) => B.LV(V, -vd), B.V0, B.Vf, B.T, Nel);
  const elGap = Math.max(...el.t.map((t, k) => Math.abs(el.x[k] - lerpAt(free.t, free.x.map((x) => x[0]), t)))) / dV;
  let elJ = 0; for (let k = 0; k < Nel; k++) elJ += (B.T / Nel) * B.L(0.5 * (el.x[k] + el.x[k + 1]), -el.v[k]);
  ctx.progress(0.6, 'Direct transcription (nonlinear programme, SQP)'); await ctx.tick();
  const Nd = Math.min(N, 24), tq = pm.t, qp = pm.u.map((u) => u[0]), cell = (k) => { const a0 = (k * B.T) / Nd, b0 = ((k + 1) * B.T) / Nd, pts = linspace(a0, b0, 9); return mean(pts.map((t) => lerpAt(tq, qp, t))); };
  const nl = batchDirect(B, Nd, B.qmax, new Array(Nd).fill(qAvg)), pmCell = range(Nd).map(cell), nlGapQ = Math.max(...nl.q.map((q, k) => Math.abs(q - pmCell[k]))) / qAvg;
  // reference policies: constant flux and constant pressure, integrated with the same RK4 rule
  const policy = (qOf) => { const r = rk4((t, y) => { const q = qOf(y[0]); return [-q, B.L(y[0], q)]; }, [B.V0, 0], 0, B.T, N); return { t: r.t, V: r.y.map((y) => y[0]), q: r.y.map((y) => qOf(y[0])), J: r.y[N][1] }; };
  const cf = policy(() => qAvg), endV = (Pf) => policy((V) => B.qOfP(V, Pf)).V[N] - B.Vf;
  let Plo = B.P(B.V0, 0), Phi = B.P(B.Vf, B.qmax) + 5, cp = null;
  if (endV(Phi) < 0) { const Pc = brentRoot(endV, Plo, Phi, endV(Plo), endV(Phi), 1e-12); cp = { ...policy((V) => B.qOfP(V, Pc)), P: Pc }; }
  const Vp = pm.x.map((x) => x[0]), lam = pm.lam.map((l) => l[0]), Pp = Vp.map((V, k) => B.P(V, qp[k])), Hm = mean(pm.H), Hdrift = (Math.max(...pm.H) - Math.min(...pm.H)) / Math.max(1e-12, Math.abs(Hm));
  const HfreeDrift = (Math.max(...free.H) - Math.min(...free.H)) / Math.max(1e-12, Math.abs(mean(free.H))), onLimit = (100 * qp.filter((q) => q >= B.qmax * (1 - 1e-9)).length) / qp.length;
  const Emin = (B.pi0 * B.V0 * Math.log(B.V0 / B.Vf)) / 36, sec = (J) => J / dV, save = (J) => (100 * (J - pm.J)) / J, volErr = Math.abs(Vp[N] - B.Vf) / dV;
  const fluxOf = (qs) => qs.map(B.flux), W = [], qFreeMax = Math.max(...free.u.map((u) => u[0]));
  if (onLimit > 0.5) W.push({ level: 'info', msg: `The flux limit is active for ${fmt(onLimit, 3)} % of the cycle: the unconstrained Euler–Lagrange extremal would start at ${fmt(B.flux(qFreeMax), 3)} L/m²·h, so Pontryagin's principle clips the control at ${fmt(v.ocFluxMax)} L/m²·h and re-distributes the remaining volume.` });
  else W.push({ level: 'info', msg: 'The flux limit is never reached: the constrained and unconstrained extremals coincide.' });
  if (nlGapQ > 0.03) W.push({ level: 'warn', msg: `Direct transcription and the Pontryagin solution differ by up to ${fmt(100 * nlGapQ, 2)} % of the mean permeate rate — the nonlinear programme stopped before full convergence (${nl.iterations} SQP iterations).` });
  if (Hdrift > 1e-4) W.push({ level: 'warn', msg: `The Hamiltonian varies by ${fmt(100 * Hdrift, 2)} % along the path; it should be constant for this autonomous problem — use more time steps.` });
  if (!cp) W.push({ level: 'info', msg: 'No constant feed pressure inside the flux limit reaches the target volume; the constant-pressure reference is omitted.' });
  const rows = [['Pontryagin optimum (flux limit enforced)', pm.J, sec(pm.J), Math.max(...Pp), Math.max(...fluxOf(qp)), 0], ['Direct transcription (NLP, SQP)', nl.J, sec(nl.J), Math.max(...nl.q.map((q, k) => B.P(B.V0 - sum(nl.q.slice(0, k + 1)) * nl.dt, q))), Math.max(...fluxOf(nl.q)), (100 * (nl.J - pm.J)) / pm.J],
    ['Unconstrained extremal (Euler–Lagrange)', elJ, sec(elJ), Math.max(...free.x.map((x, k) => B.P(x[0], free.u[k][0]))), B.flux(qFreeMax), (100 * (elJ - pm.J)) / pm.J], ['Constant flux', cf.J, sec(cf.J), Math.max(...cf.V.map((V) => B.P(V, qAvg))), B.flux(qAvg), (100 * (cf.J - pm.J)) / pm.J],
    ...(cp ? [['Constant feed pressure', cp.J, sec(cp.J), cp.P, Math.max(...fluxOf(cp.q)), (100 * (cp.J - pm.J)) / pm.J]] : []), ['Reversible limit (η = 1, no losses)', Emin, sec(Emin), null, null, (100 * (Emin - pm.J)) / pm.J]];
  const ks = [...new Set(range(21).map((i) => Math.round((i * N) / 20)))];
  return {
    summary: `The energy-optimal batch cycle needs ${fmt(pm.J, 4)} kWh (${fmt(sec(pm.J), 4)} kWh/m³) to produce ${fmt(dV, 4)} m³ in ${fmt(B.T)} h — ${cp ? `${fmt(save(cp.J), 3)} % less than constant pressure and ` : ''}${fmt(save(cf.J), 3)} % less than constant flux. The Pontryagin shooting solution, the Euler–Lagrange boundary-value solution and the direct nonlinear programme agree, and the Hamiltonian is constant to ${fmt(100 * Hdrift, 2)} %.`,
    warnings: W,
    kpis: [
      { label: 'Optimal cycle energy', value: pm.J, unit: 'kWh', status: 'ok' }, { label: 'Optimal specific energy', value: sec(pm.J), unit: 'kWh/m³' },
      { label: 'Saving against constant flux', value: save(cf.J), unit: '%' }, ...(cp ? [{ label: 'Saving against constant pressure', value: save(cp.J), unit: '%' }] : []),
      { label: 'Initial costate λ(0)', value: pm.lam0[0], unit: 'kWh/m³', help: 'Shadow price of permeate volume: the marginal energy of the next m³ at the start of the cycle.' }, { label: 'Final costate λ(T)', value: lam[N], unit: 'kWh/m³' },
      { label: 'Hamiltonian drift along the path', value: Hdrift, unit: 'relative', status: Hdrift < 1e-4 ? 'ok' : 'warn', help: 'H = L − λ·q is conserved for a time-invariant problem; the drift is the integration error.' },
      { label: 'Terminal-volume error', value: volErr, unit: 'of ΔV', status: volErr < 1e-6 ? 'ok' : 'warn', help: 'Residual of the terminal-state constraint V(T) = V_f after shooting.' },
      { label: 'Euler–Lagrange vs Hamiltonian path', value: elGap, unit: 'of ΔV', status: elGap < 2e-3 ? 'ok' : 'warn', help: 'Largest difference in V(t) between the finite-difference Euler–Lagrange solution and the canonical (state–costate) solution of the unconstrained problem.' },
      { label: 'NLP vs Pontryagin energy', value: (100 * (nl.J - pm.J)) / pm.J, unit: '%', status: Math.abs(nl.J - pm.J) / pm.J < 2e-3 ? 'ok' : 'warn', help: `Direct transcription with ${Nd} piecewise-constant intervals solved by SQP.` },
      { label: 'Time on the flux limit', value: onLimit, unit: '%' }, { label: 'Peak feed pressure', value: Math.max(...Pp), unit: 'bar' }, { label: 'Reversible minimum energy', value: Emin, unit: 'kWh' },
    ],
    recommendations: [
      'Program the feed-pump set-point to follow the optimal flux trajectory: nearly constant flux, tapered towards the end of the cycle where concentration polarisation is most expensive.',
      onLimit > 0.5 ? 'Relaxing the flux limit (cleaner feed, better pre-treatment) would save further energy — the costate shows the marginal value.' : 'The flux limit is slack; a shorter cycle or less membrane area could be afforded.',
      'Check the peak pressure against the element rating and the concentrate chemistry in suite 2 at the final concentration factor.',
    ],
    plots: [
      { type: 'line', title: 'Control: permeate flux along the cycle', xlabel: 'Time (h)', ylabel: 'Flux (L/m²·h)', series: [{ name: 'Pontryagin optimum', x: pm.t, y: fluxOf(qp) }, { name: `Direct transcription (${Nd} intervals)`, x: range(Nd + 1).map((k) => k * nl.dt), y: fluxOf([...nl.q, nl.q[Nd - 1]]), mode: 'step' },
        { name: 'Unconstrained extremal (Euler–Lagrange)', x: range(Nel).map((k) => (k + 0.5) * (B.T / Nel)), y: fluxOf(el.v.map((q) => -q)), dash: true }, ...(cp ? [{ name: 'Constant pressure', x: cp.t, y: fluxOf(cp.q), dash: true }] : [])], hlines: [{ y: v.ocFluxMax, label: 'flux limit' }] },
      { type: 'line', title: 'State: tank volume', xlabel: 'Time (h)', ylabel: 'Volume (m³)', series: [{ name: 'Pontryagin optimum', x: pm.t, y: Vp }, { name: 'Euler–Lagrange (unconstrained)', x: el.t, y: el.x, dash: true }, ...(cp ? [{ name: 'Constant pressure', x: cp.t, y: cp.V, dash: true }] : [])], hlines: [{ y: B.Vf, label: 'terminal volume' }] },
      { type: 'line', title: 'Costate (shadow price of volume) and marginal pumping energy', xlabel: 'Time (h)', ylabel: 'kWh/m³', series: [{ name: 'Costate λ(t)', x: pm.t, y: lam }, { name: '∂L/∂q at the applied control', x: pm.t, y: Vp.map((V, k) => B.Lq(V, qp[k])), dash: true }], note: 'Where the two curves coincide the control is interior (∂H/∂q = 0); where λ exceeds ∂L/∂q the control sits on its upper bound.' },
      { type: 'line', title: 'Hamiltonian along the optimal path', xlabel: 'Time (h)', ylabel: 'H = L − λ·q (kW)', series: [{ name: 'Flux limit enforced', x: pm.t, y: pm.H }, { name: 'Unconstrained', x: free.t, y: free.H, dash: true }] },
      { type: 'line', title: 'Feed pressure', xlabel: 'Time (h)', ylabel: 'Pressure (bar)', series: [{ name: 'Pontryagin optimum', x: pm.t, y: Pp }, { name: 'Constant flux', x: cf.t, y: cf.V.map((V) => B.P(V, qAvg)), dash: true }, { name: 'Osmotic pressure in the tank', x: pm.t, y: Vp.map(B.pi), dash: true }] },
    ],
    tables: [
      { title: 'Operating policies compared', columns: ['Policy', 'Energy (kWh)', 'Specific energy (kWh/m³)', 'Peak pressure (bar)', 'Peak flux (L/m²·h)', 'Energy relative to optimum (%)'], rows,
        note: 'The unconstrained extremal ignores the flux limit and is therefore a lower bound; the direct nonlinear programme solves the constrained problem on a coarser control grid.' },
      { title: 'Optimal trajectory', columns: ['Time (h)', 'Volume (m³)', 'Flux (L/m²·h)', 'Feed pressure (bar)', 'Osmotic pressure (bar)', 'Costate λ (kWh/m³)', 'Hamiltonian (kW)'], rows: ks.map((k) => [pm.t[k], Vp[k], B.flux(qp[k]), Pp[k], B.pi(Vp[k]), lam[k], pm.H[k]]) },
      { title: 'Formulation and solution evidence', columns: ['Item', 'Value'], rows: [['State and control', 'V(t) tank volume; q(t) permeate rate, 0 ≤ q ≤ q_max'], ['Running cost', 'L = P·q/(36·η), P = q/(A·A_m) + π₀·(V₀/V)·exp(J/k) + Δp'], ['Hamiltonian', 'H = L(V, q) − λ·q'],
        ['Canonical equations', 'dV/dt = −q*, dλ/dt = −∂L/∂V, ∂L/∂q = λ (interior) or q* on a bound'], ['Boundary conditions', `V(0) = ${fmt(B.V0)}, V(T) = ${fmt(B.Vf, 5)} m³ (terminal-state constraint)`], ['Shooting residual', pm.residual], ['Euler–Lagrange residual', el.residual],
        ['Hamiltonian drift, unconstrained path', HfreeDrift], ['NLP equality residual', nl.eqErr], ['NLP multiplier of the volume constraint (kWh/m³)', nl.lambda], ['SQP iterations / evaluations', `${nl.iterations} / ${nl.evals}`]] },
    ],
    balances: [{ name: 'Permeate volume (m³)', in: dV, out: B.V0 - Vp[N] }],
    outputs: { task: 'oc', best: { energy: pm.J, sec: sec(pm.J), lambda0: pm.lam0[0], peakPressure: Math.max(...Pp), fluxStart: B.flux(qp[0]), fluxEnd: B.flux(qp[N]) }, objective: pm.J, study: { a: pm.J, b: pm.lam0[0] } },
  };
}

// ---- Task 12: linear and mixed-integer programming — production scheduling of RO trains against a tariff --------------------------
const tariffAt = (v, h) => (h >= 17 && h <= 21 ? v.lpPeak : h >= 7 && h <= 22 ? v.lpShoulder : v.lpOff);
function lpTrainTable() { return [{ cap: 400, sec: 2.9, minLoad: 60, aux: 60 }, { cap: 400, sec: 3.1, minLoad: 60, aux: 60 }, { cap: 300, sec: 3.6, minLoad: 50, aux: 45 }]; }
/** Build the scheduling model: load fractions x[i][t], unmet demand u[t] and on/off variables y[i][t]; storage rows are scaled by the tank volume. */
function scheduleModel(v) {
  const trains = numRows(v.lpTrains, ['cap', 'sec']).filter((r) => r.cap > 0).slice(0, 5).map((r) => ({ cap: r.cap, sec: Math.max(0, r.sec), min: clamp(finite(r.minLoad, 0), 0, 100) / 100, aux: Math.max(0, finite(r.aux, 0)) }));
  if (!trains.length) throw new Error('The train table needs at least one row with a capacity and a specific energy.');
  const NP = [4, 6, 8, 12, 24].includes(+v.lpPeriods) ? +v.lpPeriods : 8, h = 24 / NP, nT = trains.length, dm = mean(DEMAND);
  const d = range(NP).map((t) => (Math.max(0, v.lpDemand) * mean(range(h).map((q) => DEMAND[t * h + q]))) / dm), p = range(NP).map((t) => mean(range(h).map((q) => tariffAt(v, t * h + q))));
  const Smax = Math.max(1e-6, v.lpTank), S0 = (Smax * clamp(v.lpS0, 0, 100)) / 100, pen = Math.max(0, v.lpPenalty), n = 2 * nT * NP + NP, X = (i, t) => i * NP + t, U = (t) => nT * NP + t, Y = (i, t) => nT * NP + NP + i * NP + t;
  const cScale = Math.max(1e-9, h * mean(p) * mean(trains.map((r) => r.sec * r.cap))), c = new Array(n).fill(0), rows = [], tag = [];
  for (let t = 0; t < NP; t++) { c[U(t)] = (h * pen * Math.max(...d, 1)) / cScale; for (let i = 0; i < nT; i++) { c[X(i, t)] = (h * p[t] * trains[i].sec * trains[i].cap) / cScale; c[Y(i, t)] = (h * p[t] * trains[i].aux) / cScale; } }
  const dMax = Math.max(...d, 1);
  for (let t = 0; t < NP; t++) { // cumulative storage balance: S_t = S0 + h·Σ(production + unmet − demand)
    const a = new Array(n).fill(0), cum = h * sum(d.slice(0, t + 1));
    for (let q = 0; q <= t; q++) { a[U(q)] = (h * dMax) / Smax; for (let i = 0; i < nT; i++) a[X(i, q)] = (h * trains[i].cap) / Smax; }
    rows.push({ a, op: '>=', b: (cum - S0) / Smax }); tag.push(['lo', t]);
    rows.push({ a: [...a], op: '<=', b: (cum - S0 + Smax) / Smax }); tag.push(['hi', t]);
    if (t === NP - 1) { rows.push({ a: [...a], op: '>=', b: cum / Smax }); tag.push(['end', t]); } // terminal condition: finish at least as full as at the start
  }
  for (let t = 0; t < NP; t++) for (let i = 0; i < nT; i++) {
    const a1 = new Array(n).fill(0), a2 = new Array(n).fill(0), a3 = new Array(n).fill(0);
    a1[X(i, t)] = 1; a1[Y(i, t)] = -1; rows.push({ a: a1, op: '<=', b: 0 }); tag.push(['cap', t, i]);
    if (trains[i].min > 0) { a2[X(i, t)] = 1; a2[Y(i, t)] = -trains[i].min; rows.push({ a: a2, op: '>=', b: 0 }); tag.push(['min', t, i]); }
    a3[Y(i, t)] = 1; rows.push({ a: a3, op: '<=', b: 1 }); tag.push(['on', t, i]);
  }
  const ints = range(nT * NP).map((q) => nT * NP + NP + q);
  const read = (x) => {
    const prod = range(NP).map((t) => trains.map((r, i) => r.cap * x[X(i, t)])), on = range(NP).map((t) => trains.map((_, i) => x[Y(i, t)])), unmet = range(NP).map((t) => dMax * x[U(t)]), level = [S0];
    for (let t = 0; t < NP; t++) level.push(level[t] + h * (sum(prod[t]) + unmet[t] - d[t]));
    const energy = sum(range(NP).map((t) => h * sum(trains.map((r, i) => r.sec * prod[t][i] + r.aux * on[t][i])))), cost = sum(range(NP).map((t) => h * p[t] * sum(trains.map((r, i) => r.sec * prod[t][i] + r.aux * on[t][i]))));
    return { prod, on, unmet, level, energy, cost, penalty: h * pen * sum(unmet), total: cost + h * pen * sum(unmet) };
  };
  return { trains, NP, h, nT, d, p, Smax, S0, pen, c, rows, tag, ints, cScale, read, n };
}
async function taskLP(v, ctx) {
  const M = scheduleModel(v), integer = v.lpMode !== 'lp';
  ctx.progress(0.2, 'Simplex on the linear relaxation'); await ctx.tick();
  const lp = lpSolve(M.c, M.rows);
  if (lp.status !== 'optimal') throw new Error(`The linear programme is ${lp.status}: check the tank size, the demand and the train capacities.`);
  ctx.progress(0.4, integer ? 'Branch and bound on the on/off variables' : 'Reading the solution'); await ctx.tick();
  const mi = integer ? milpSolve(M.c, M.rows, M.ints, { maxNodes: clamp(Math.round(v.lpNodes), 10, 20000), maxWork: 8e8 }) : null;
  if (mi && !Number.isFinite(mi.f)) throw new Error(`Branch and bound found no integer schedule within ${mi.nodes} nodes (${mi.status}); raise the node limit or relax the minimum loads.`);
  const rel = M.read(lp.x), sol = mi ? M.read(mi.x) : rel, fOpt = (mi ? mi.f : lp.f) * M.cScale, fRel = lp.f * M.cScale, gap = mi ? mi.gap : 0, intGap = mi ? (100 * (mi.f - lp.f)) / Math.max(1e-12, Math.abs(mi.f)) : 0;
  // flat reference: every train on all day at the same load fraction
  const fr = Math.min(1, sum(M.d) / M.NP / sum(M.trains.map((r) => r.cap))), flatCost = sum(range(M.NP).map((t) => M.h * M.p[t] * sum(M.trains.map((r) => r.sec * r.cap * fr + r.aux))));
  const water = M.h * sum(sol.prod.map(sum)), peakIdx = range(M.NP).filter((t) => M.p[t] >= Math.max(...M.p) - 1e-12), peakShare = (100 * sum(peakIdx.map((t) => sum(sol.prod[t])))) / Math.max(1e-9, sum(sol.prod.map(sum)));
  const dualOf = (kind) => M.tag.map((g, r) => (g[0] === kind ? (lp.dual[r] * M.cScale) / M.Smax : null)).filter((q) => q !== null), dHi = dualOf('hi'), dLo = dualOf('lo'), dEnd = dualOf('end')[0];
  const tankValue = -sum(dHi), W = [], names = M.trains.map((_, i) => `Train ${i + 1}`), per = range(M.NP).map((t) => `${String(t * M.h).padStart(2, '0')}–${String((t + 1) * M.h).padStart(2, '0')} h`);
  if (sum(sol.unmet) > 1e-6) W.push({ level: 'bad', msg: `Demand cannot be met in full: ${fmt(M.h * sum(sol.unmet), 4)} m³ is left unserved — add capacity or storage.` });
  if (mi && mi.status === 'node limit') W.push({ level: 'warn', msg: `Branch and bound stopped at its search limit (${mi.nodes} nodes, ${mi.lpIterations} simplex pivots) with an optimality gap of ${fmt(100 * gap, 3)} %; the schedule shown is the best integer solution found.` });
  else if (mi) W.push({ level: 'info', msg: `Branch and bound proved optimality after ${mi.nodes} nodes and ${mi.lpIterations} simplex pivots; the linear relaxation is ${fmt(intGap, 3)} % cheaper than the best on/off schedule.` });
  else W.push({ level: 'info', msg: `Linear programme solved in ${lp.iterations} simplex pivots; minimum loads and auxiliary power are relaxed (trains may run at any fraction).` });
  return {
    summary: `${integer ? 'Mixed-integer' : 'Linear'} programme with ${M.n} variables and ${M.rows.length} constraints: the cheapest schedule costs ${fmt(fOpt, 5)} $/d (${fmt(fOpt / Math.max(1e-9, water), 4)} $/m³ energy), ${fmt((100 * (flatCost - sol.cost)) / flatCost, 3)} % below running every train at constant load, with ${fmt(peakShare, 3)} % of production in the peak-tariff periods.`,
    warnings: W,
    kpis: [
      { label: 'Optimal daily energy cost', value: fOpt, unit: '$/d', status: sum(sol.unmet) > 1e-6 ? 'bad' : 'ok' }, { label: 'Saving against constant load', value: (100 * (flatCost - sol.cost)) / flatCost, unit: '%' },
      { label: 'LP relaxation (lower bound)', value: fRel, unit: '$/d' }, { label: 'Integrality gap', value: intGap, unit: '%', help: 'Cost of the on/off and minimum-load restrictions relative to the relaxed linear programme.' },
      { label: 'Optimality gap', value: 100 * gap, unit: '%', status: gap < 1e-6 ? 'ok' : 'warn', help: 'Distance between the best integer solution and the best remaining bound of the branch-and-bound tree.' },
      { label: 'Branch-and-bound nodes', value: mi ? mi.nodes : 0 }, { label: 'Simplex pivots', value: mi ? mi.lpIterations : lp.iterations },
      { label: 'Energy used', value: sol.energy, unit: 'kWh/d' }, { label: 'Water produced', value: water, unit: 'm³/d' }, { label: 'Production in peak periods', value: peakShare, unit: '%' },
      { label: 'Value of one more m³ of storage', value: tankValue, unit: '$/d per m³', help: 'Sum of the dual values of the tank-full constraints of the linear relaxation.' }, { label: 'Marginal cost of the end-of-day level', value: finite(dEnd), unit: '$ per m³' },
    ],
    recommendations: [
      tankValue > 1e-9 ? `Storage is binding: each additional m³ of tank volume is worth ${fmt(tankValue, 3)} $/d in the relaxed problem.` : 'The tank never limits the schedule; storage is not the bottleneck.',
      'Feed the per-period production targets to the plant supervisor; the predictive-control task of this suite can track them.',
      'Re-run with the measured specific energy of each train (suite 12) and the site tariff (Global Site Data).',
    ],
    plots: [
      { type: 'bar', title: 'Optimal production per period', ylabel: 'm³/h', stacked: true, categories: per, series: names.map((nm, i) => ({ name: nm, values: sol.prod.map((r) => r[i]) })) },
      { type: 'line', title: 'Tank level and tariff', xlabel: 'Hour of day', ylabel: 'Tank level (% of volume)', series: [{ name: integer ? 'Level, on/off schedule' : 'Level, LP schedule', x: range(M.NP + 1).map((t) => t * M.h), y: sol.level.map((s) => (100 * s) / M.Smax), mode: 'both' }, ...(integer ? [{ name: 'Level, LP relaxation', x: range(M.NP + 1).map((t) => t * M.h), y: rel.level.map((s) => (100 * s) / M.Smax), dash: true }] : []),
        { name: 'Tariff (¢/kWh)', x: range(M.NP + 1).map((t) => t * M.h), y: [...M.p, M.p[M.NP - 1]].map((q) => 100 * q), mode: 'step', dash: true }], hlines: [{ y: 100, label: 'full' }, { y: 0, label: 'empty' }] },
      { type: 'bar', title: 'Demand and production', ylabel: 'm³/h', categories: per, series: [{ name: 'Demand', values: M.d }, { name: 'Production', values: sol.prod.map(sum) }, ...(integer ? [{ name: 'Production, LP relaxation', values: rel.prod.map(sum) }] : [])] },
      { type: 'bar', title: 'Shadow prices of the storage constraints (LP relaxation)', ylabel: '$/d per m³', categories: per, series: [{ name: 'Tank full', values: dHi.map((q) => -q) }, { name: 'Tank empty', values: dLo }] },
    ],
    tables: [
      { title: 'Schedule', columns: ['Period', 'Tariff ($/kWh)', 'Demand (m³/h)', ...names.map((nm) => `${nm} (m³/h)`), ...(integer ? names.map((nm) => `${nm} on`) : []), 'Unmet (m³/h)', 'Level at end (%)'],
        rows: range(M.NP).map((t) => [per[t], M.p[t], M.d[t], ...sol.prod[t], ...(integer ? sol.on[t].map((q) => (q > 0.5 ? 'on' : 'off')) : []), sol.unmet[t], (100 * sol.level[t + 1]) / M.Smax]) },
      { title: 'Solutions compared', columns: ['Schedule', 'Energy cost ($/d)', 'Energy (kWh/d)', 'Unmet demand (m³/d)'], rows: [['Constant load, all trains on', flatCost, sum(range(M.NP).map((t) => M.h * sum(M.trains.map((r) => r.sec * r.cap * fr + r.aux)))), 0], ['Linear relaxation', rel.cost, rel.energy, M.h * sum(rel.unmet)], ...(integer ? [['On/off schedule (branch and bound)', sol.cost, sol.energy, M.h * sum(sol.unmet)]] : [])] },
      { title: 'Trains', columns: ['Train', 'Capacity (m³/h)', 'Specific energy (kWh/m³)', 'Minimum load (%)', 'Auxiliary power when on (kW)', 'Hours on', 'Mean load when on (%)'], rows: M.trains.map((r, i) => { const onH = M.h * sum(sol.on.map((q) => q[i])); return [names[i], r.cap, r.sec, 100 * r.min, r.aux, onH, onH > 1e-9 ? (100 * M.h * sum(sol.prod.map((q) => q[i]))) / (onH * r.cap) : 0]; }),
        note: 'Model: minimise Σ tariff·(specific energy·production + auxiliary power·on) subject to the cumulative storage balance 0 ≤ S ≤ S_max, S(end) ≥ S(start), capacity·on ≥ production ≥ minimum load·on. The linear programme is solved by a two-phase simplex method; integrality by branch and bound on its relaxation.' },
    ],
    balances: [{ name: 'Water over the day (m³)', in: M.S0 + water + M.h * sum(sol.unmet), out: sol.level[M.NP] + M.h * sum(M.d) }],
    outputs: { task: 'lp', best: { cost: fOpt, energy: sol.energy, peakShare, production: sol.prod.map(sum), level: sol.level }, objective: fOpt, relaxation: fRel, gap, study: { a: fOpt, b: fRel } },
  };
}

// ---- Task 13: neural ordinary differential equation -----------------------------------------------------------------------------
/** Built-in record: four runs between cleanings following dK/dt = −0.09·(K − 0.7)^1.6 (unknown to the model) with measurement noise. */
function nodeSampleTable() {
  const g = rng(404), rows = [];
  [1.0, 0.94, 0.88, 0.97].forEach((K0, r) => {
    const s = rk4((t, y) => [-0.09 * Math.max(0, y[0] - 0.7) ** 1.6], [K0], 0, 120, 120);
    for (let k = 0; k <= 120; k += 8) rows.push({ run: r + 1, t: k, y: +(s.y[k][0] + (k ? g.normal(0, 0.003) : 0)).toFixed(4) });
  });
  return rows;
}
async function taskNODE(v, ctx) {
  const rows = numRows(v.nodeData, ['run', 't', 'y']), ids = [...new Set(rows.map((r) => r.run))].sort((a, b) => a - b);
  const runs = ids.map((id) => { const q = rows.filter((r) => r.run === id).sort((a, b) => a.t - b.t).filter((r, i, a) => !i || r.t > a[i - 1].t); return { id, t: q.map((r) => r.t - q[0].t), y: q.map((r) => r.y) }; }).filter((r) => r.t.length >= 3);
  if (!runs.length) throw new Error('The trajectory table needs at least one run with three or more time-ordered rows (run, t, y).');
  let train, test, mode;
  if (runs.length >= 2) { const hold = runs.find((r) => r.id === Math.round(v.nodeHold)) || runs[runs.length - 1]; train = runs.filter((r) => r !== hold); test = [{ ...hold }]; mode = `run ${hold.id} held out`; }
  else { const r = runs[0], k = Math.max(3, Math.round(0.7 * r.t.length)); if (r.t.length - k < 2) throw new Error('A single run needs at least six rows so that its last part can be held out.'); train = [{ id: r.id, t: r.t.slice(0, k), y: r.y.slice(0, k) }]; test = [{ id: r.id, t: r.t.slice(k - 1).map((q) => q - r.t[k - 1]), y: r.y.slice(k - 1), t0: r.t[k - 1] }]; mode = 'last 30 % of the run held out'; }
  const all = train.flatMap((r) => r.y), ym = mean(all), ys = std(all) || 1, slopes = train.flatMap((r) => range(r.t.length - 1).map((i) => (r.y[i + 1] - r.y[i]) / ys / (r.t[i + 1] - r.t[i]))), tSpan = Math.max(...runs.map((r) => r.t[r.t.length - 1])) || 1, tRef = clamp(1 / (Math.sqrt(mean(slopes.map((q) => q * q))) || 1 / tSpan), tSpan / 50, tSpan), H = clamp(Math.round(v.nodeH), 2, 32), sub = clamp(Math.round(v.nodeSub), 1, 10), seed = Math.round(v.seed);
  const scale = (r) => ({ t: r.t.map((q) => q / tRef), y: r.y.map((q) => [(q - ym) / ys]) }), trS = train.map(scale);
  ctx.progress(0.1, 'Training the neural ODE (back-propagation through RK4)'); await ctx.tick();
  const fit = nodeTrain(1, H, trS, { epochs: clamp(Math.round(v.nodeEpochs), 20, 20000), lr: v.nodeLr, seed, sub });
  ctx.progress(0.8, 'Held-out trajectory and gradient check'); await ctx.tick();
  const predict = (r, ts) => nodeSolve(fit.p, 1, H, [(r.y[0] - ym) / ys], ts.map((q) => q / tRef), sub).map((q) => ym + ys * q[0]);
  const err = (set, f) => Math.sqrt(mean(set.flatMap((r) => { const p = f(r); return r.y.slice(1).map((q, i) => (p[i + 1] - q) ** 2); })));
  // baseline: first-order law y' = −r·(y − y∞) with the closed-form solution, fitted to the same training runs
  const lin = levenbergMarquardt((p) => train.flatMap((r) => r.t.slice(1).map((t, i) => p[1] + (r.y[0] - p[1]) * Math.exp(-p[0] * t) - r.y[i + 1])), [1 / tRef, Math.min(...all) - 0.5 * ys], { lo: [0, -1e9], hi: [1e3 / tRef, 1e9] }).p;
  const linPred = (r) => r.t.map((t) => lin[1] + (r.y[0] - lin[1]) * Math.exp(-lin[0] * t));
  const eTr = err(train, (r) => predict(r, r.t)), eTe = err(test, (r) => predict(r, r.t)), bTr = err(train, linPred), bTe = err(test, linPred);
  // adjoint gradient against central differences on a few parameters
  const lg = nodeLossGrad(fit.p, 1, H, trS, sub), gIdx = [0, H, 2 * H, 3 * H].filter((i) => i < fit.p.length);
  const p0 = nodeInit(1, H, seed + 5), l0 = nodeLossGrad(p0, 1, H, trS, sub);
  const gErr = Math.max(...gIdx.map((i) => { const w = Float64Array.from(p0), hh = 1e-6; w[i] += hh; const a = nodeLossGrad(w, 1, H, trS, sub).loss; w[i] -= 2 * hh; const b = nodeLossGrad(w, 1, H, trS, sub).loss; return Math.abs((a - b) / (2 * hh) - l0.grad[i]) / Math.max(1e-12, maxAbs(Array.from(l0.grad))); }));
  const yLo = Math.min(...runs.flatMap((r) => r.y)), yHi = Math.max(...runs.flatMap((r) => r.y)), yy = linspace(yLo, yHi, 40), z = new Array(H), o = [0];
  const rhs = yy.map((q) => (ys / tRef) * nodeF(fit.p, 1, H, [(q - ym) / ys], z, o)[0]), fd = train.flatMap((r) => range(r.t.length - 1).map((i) => [0.5 * (r.y[i] + r.y[i + 1]), (r.y[i + 1] - r.y[i]) / (r.t[i + 1] - r.t[i])]));
  const W = [], better = eTe < bTe;
  W.push({ level: 'info', msg: `Validation: ${mode}. The network was trained on ${train.length} run${train.length > 1 ? 's' : ''} (${sum(train.map((r) => r.t.length - 1))} fitted points) and never saw the held-out data.` });
  if (eTe > 3 * eTr && eTe > 0.02 * ys) W.push({ level: 'warn', msg: `The held-out error (${fmt(eTe, 3)}) is ${fmt(eTe / eTr, 3)} times the training error — the learned dynamics do not generalise; add runs covering the held-out range or reduce the network.` });
  if (!better) W.push({ level: 'warn', msg: 'The first-order baseline predicts the held-out trajectory at least as well as the neural ODE: the data do not support a more flexible law.' });
  const t0 = (r) => r.t0 || 0, dense = (r) => linspace(0, r.t[r.t.length - 1], 40);
  return {
    summary: `A neural ODE dy/dt = f_θ(y) with ${H} hidden neurons (${fit.nParams} parameters), integrated by RK4 and trained by back-propagation through the integrator, reproduces the training runs to an RMS error of ${fmt(eTr, 3)} and the held-out trajectory to ${fmt(eTe, 3)} (first-order baseline: ${fmt(bTe, 3)}).`,
    warnings: W,
    kpis: [
      { label: 'Held-out trajectory RMSE', value: eTe, status: better ? 'ok' : 'warn', help: 'Error on data never used for training, in the units of y.' }, { label: 'Training RMSE', value: eTr },
      { label: 'Baseline held-out RMSE', value: bTe, help: 'First-order law y′ = −r·(y − y∞) fitted to the same training runs.' }, { label: 'Improvement on baseline', value: (100 * (bTe - eTe)) / Math.max(1e-300, bTe), unit: '%', status: better ? 'ok' : 'warn' },
      { label: 'Adjoint gradient check', value: gErr, unit: 'relative', status: gErr < 1e-5 ? 'ok' : 'warn', help: 'Largest difference between the back-propagated gradient and central finite differences, relative to the largest gradient entry.' },
      { label: 'Network parameters', value: fit.nParams }, { label: 'Final training loss (scaled)', value: fit.loss }, { label: 'Gradient norm at the end', value: Math.hypot(...lg.grad) },
      { label: 'Baseline rate constant', value: lin[0], unit: '1/time' }, { label: 'Baseline plateau y∞', value: lin[1] },
    ],
    recommendations: [
      'Use the learned right-hand side inside the Kalman estimator or a predictive controller only within the range of y covered by the training runs (see the phase plot).',
      better ? 'The neural law beats the first-order model on unseen data — the decline is not a simple exponential.' : 'Keep the first-order model: it is simpler and predicts the held-out data equally well.',
      'Hold out a different run and re-train to check that the conclusion does not depend on the split.',
    ],
    plots: [
      { type: 'line', title: 'Trajectories: data and neural ODE', xlabel: 'Time', ylabel: 'y', series: [...train.flatMap((r) => [{ name: `Run ${r.id} data`, x: r.t, y: r.y, mode: 'points' }, { name: `Run ${r.id} neural ODE`, x: dense(r), y: predict(r, dense(r)) }]),
        ...test.flatMap((r) => [{ name: `Held-out run ${r.id} data`, x: r.t.map((q) => q + t0(r)), y: r.y, mode: 'points' }, { name: 'Held-out prediction', x: dense(r).map((q) => q + t0(r)), y: predict(r, dense(r)), dash: true }, { name: 'First-order baseline', x: r.t.map((q) => q + t0(r)), y: linPred(r), dash: true }])] },
      { type: 'line', title: 'Learned right-hand side (phase plot)', xlabel: 'y', ylabel: 'dy/dt', series: [{ name: 'Neural ODE f_θ(y)', x: yy, y: rhs }, { name: 'First-order baseline', x: yy, y: yy.map((q) => -lin[0] * (q - lin[1])), dash: true }, { name: 'Finite differences of the training data', x: fd.map((q) => q[0]), y: fd.map((q) => q[1]), mode: 'points' }] },
      { type: 'line', title: 'Training loss', xlabel: 'Epoch', ylabel: 'Mean squared error (scaled)', logy: true, series: [{ name: 'Loss', x: fit.history.epoch, y: fit.history.loss.map((q) => Math.max(q, 1e-300)) }] },
    ],
    tables: [
      { title: 'Errors by run', columns: ['Run', 'Role', 'Points', 'Neural ODE RMSE', 'Baseline RMSE'], rows: [...train.map((r) => [r.id, 'training', r.t.length, err([r], (q) => predict(q, q.t)), err([r], linPred)]), ...test.map((r) => [r.id, 'held out', r.t.length, err([r], (q) => predict(q, q.t)), err([r], linPred)])] },
      { title: 'Held-out trajectory', columns: ['Time', 'Measured', 'Neural ODE', 'Baseline'], rows: test.flatMap((r) => { const p = predict(r, r.t), b = linPred(r); return r.t.map((t, i) => [t + t0(r), r.y[i], p[i], b[i]]); }) },
      { title: 'Model', columns: ['Item', 'Value'], rows: [['Right-hand side', `dy/dt = W₂·tanh(W₁·y + b₁) + b₂, ${H} hidden neurons`], ['Integrator', `Classical RK4, ${sub} step(s) per sampling interval`], ['Gradient', 'Discrete adjoint: reverse sweep through every RK4 stage'], ['Scaling', `y standardised (mean ${fmt(ym)}, s.d. ${fmt(ys)}), time divided by ${fmt(tRef)} so that the scaled slopes are of order one`], ['Optimiser', `Adam, ${Math.round(v.nodeEpochs)} epochs, learning rate decaying from ${fmt(v.nodeLr)}`]] },
    ],
    outputs: { task: 'node', best: { heldOutRmse: eTe, trainRmse: eTr, baselineRmse: bTe }, objective: eTe, study: { a: eTe, b: eTr } },
  };
}

// ---- Task 14: neural partial differential equation (learned reaction closure inside the transport solver) ------------------------
async function taskNPDE(v, ctx) {
  const L = Math.max(1e-9, v.npL), vel = v.npV, D = Math.max(0, v.npD), tEnd = Math.max(1e-9, v.npT), nx = clamp(Math.round(v.npNx), 6, 200), nt = clamp(Math.round(v.npNt), 6, 2000), H = clamp(Math.round(v.npH), 2, 24), seed = Math.round(v.seed);
  const kmax = Math.max(0, v.npKmax), Km = Math.max(1e-9, v.npKm), hidden = (u) => (-kmax * Math.max(0, u)) / (Km + Math.max(0, u)), cTrain = [v.npC1, v.npC2].map((q) => Math.max(1e-6, q)), cTest = Math.max(1e-6, v.npCtest), g = rng(seed + 31);
  const uRef = Math.max(...cTrain, cTest), rRef = (uRef * Math.max(Math.abs(vel), D / L, L / tEnd)) / L, dt = tEnd / nt, nObs = 6, kObs = range(nObs).map((q) => Math.round(((q + 1) * nt) / nObs)), theta = 0.5;
  const bc = (cin) => ({ left: { type: 'dirichlet', val: cin }, right: { type: 'neumann', val: 0 } });
  /** "Measured" fields: the finite-volume solver on a twice finer grid with the hidden law, averaged onto the model cells. */
  const truth = (cin) => { const s = solveCDR({ L, nx: 2 * nx, tEnd, nt: 2 * nt, v: vel, D, R: (u) => hidden(u), ic: () => 0, left: { type: 'dirichlet', val: () => cin }, right: { type: 'neumann', val: () => 0 }, theta: 0.5, nSave: 2 * nt }); return kObs.map((k) => range(nx).map((i) => 0.5 * (s.U[2 * k][2 * i] + s.U[2 * k][2 * i + 1]))); };
  const mkExp = (cin, noisy) => { const T = truth(cin); return { cin, op: cdrOperator({ L, nx, v: vel, D, ...bc(cin) }), u0: zeros(nx), nt, dt, clean: T, obs: T.map((u, q) => ({ k: kObs[q], u: u.map((w) => w + (noisy ? g.normal(0, (v.npNoise / 100) * cin) : 0)) })) }; };
  ctx.progress(0.05, 'Generating the experiments with the finite-volume solver'); await ctx.tick();
  const exps = cTrain.map((c) => mkExp(c, true)), exT = mkExp(cTest, false), sc = { uRef, rRef, theta };
  const p = nodeInit(1, H, seed, 0.2), st = { t: 0, m: new Float64Array(p.length), v: new Float64Array(p.length) }, epochs = clamp(Math.round(v.npEpochs), 10, 5000), hist = { ep: [], loss: [] }, every = Math.max(1, Math.floor(epochs / 100));
  const g0 = npdeLossGrad(p, H, exps, sc), gi = [0, H, 2 * H, 3 * H], gErr = Math.max(...gi.map((i) => { const w = Float64Array.from(p), hh = 1e-6; w[i] += hh; const a = npdeLossGrad(w, H, exps, sc).loss; w[i] -= 2 * hh; const b = npdeLossGrad(w, H, exps, sc).loss; return Math.abs((a - b) / (2 * hh) - g0.grad[i]) / Math.max(1e-300, maxAbs(Array.from(g0.grad))); }));
  let best = Infinity, bestP = Float64Array.from(p);
  for (let ep = 0; ep < epochs; ep++) {
    const r = npdeLossGrad(p, H, exps, sc);
    if (!Number.isFinite(r.loss)) break;
    if (r.loss < best) { best = r.loss; bestP.set(p); }
    if (ep % every === 0) { hist.ep.push(ep); hist.loss.push(r.loss); }
    adamStep(p, r.grad, st, v.npLr * 0.05 ** (ep / epochs));
    if (ep % 40 === 39) { ctx.progress(0.1 + (0.7 * ep) / epochs, `Training the closure: epoch ${ep + 1} of ${epochs}`); await ctx.tick(); }
  }
  const net = npdeClosure(bestP, H, uRef, rRef);
  // baseline closure R = −k·u with k fitted by golden-section search on the same loss
  const lossOf = (clos, set) => { let s = 0, m = 0; for (const ex of set) { const U = imexMarch(ex.op, ex.u0, nt, dt, clos, theta); for (const ob of ex.obs) for (let i = 0; i < nx; i++) { s += (U[ob.k][i] - ob.u[i]) ** 2; m++; } } return s / m; };
  let ka = 0, kb = (4 * kmax) / Km + 4 / tEnd; const gr = 0.381966;
  for (let it = 0; it < 40; it++) { const k1 = ka + gr * (kb - ka), k2 = kb - gr * (kb - ka); if (lossOf((u) => -k1 * u, exps) < lossOf((u) => -k2 * u, exps)) kb = k2; else ka = k1; }
  const kLin = 0.5 * (ka + kb), rel = (clos) => Math.sqrt(lossOf(clos, [exT])) / cTest, eNN = rel(net), eLin = rel((u) => -kLin * u), eNone = rel(() => 0), eTrain = Math.sqrt(lossOf(net, exps)) / mean(cTrain);
  // the learned closure inside the Crank–Nicolson/Picard finite-volume solver: consistency of the two discretisations
  const UT = imexMarch(exT.op, exT.u0, nt, dt, net, theta), fv = solveCDR({ L, nx, tEnd, nt, v: vel, D, R: (u) => net(u), ic: () => 0, left: { type: 'dirichlet', val: () => cTest }, right: { type: 'neumann', val: () => 0 }, theta: 0.5, nSave: nt });
  const fvGap = Math.max(...fv.u.map((q, i) => Math.abs(q - UT[nt][i]))) / cTest, uu = linspace(0, uRef, 40), closErr = Math.sqrt(mean(uu.map((u) => (net(u) - hidden(u)) ** 2))) / Math.max(1e-300, Math.abs(hidden(uRef)));
  const x = exT.op.x, last = (ex) => ex.obs[nObs - 1].u, W = [], better = eNN < eLin;
  W.push({ level: 'info', msg: `The data are a synthetic experiment: the finite-volume solver on ${2 * nx} cells with a hidden saturating law, ${fmt(v.npNoise)} % noise, inlet concentrations ${cTrain.map((c) => fmt(c)).join(' and ')}; the test inlet ${fmt(cTest)} was not used for training.` });
  if (!better) W.push({ level: 'warn', msg: 'The fitted first-order closure predicts the unseen experiment at least as well as the network — train longer or widen the range of inlet concentrations.' });
  if (fvGap > 0.02) W.push({ level: 'warn', msg: `The explicit-closure scheme and the Crank–Nicolson/Picard solver differ by ${fmt(100 * fvGap, 2)} % with the learned closure: the time step is too coarse for the reaction rate.` });
  return {
    summary: `A network with ${H} hidden neurons learned the reaction term R(u) of the transport equation from concentration snapshots of ${cTrain.length} experiments, by the discrete adjoint of the finite-volume scheme. On an unseen inlet concentration the predicted fields are within ${fmt(100 * eNN, 3)} % RMS (fitted first-order law: ${fmt(100 * eLin, 3)} %, no reaction: ${fmt(100 * eNone, 3)} %).`,
    warnings: W,
    kpis: [
      { label: 'Unseen-experiment field error', value: 100 * eNN, unit: '% of inlet', status: better ? 'ok' : 'warn' }, { label: 'Training field error', value: 100 * eTrain, unit: '% of inlet' },
      { label: 'First-order closure, unseen error', value: 100 * eLin, unit: '% of inlet' }, { label: 'No-reaction model, unseen error', value: 100 * eNone, unit: '% of inlet' },
      { label: 'Closure error against the hidden law', value: 100 * closErr, unit: '% of R(u_max)', help: 'RMS difference between the learned and the generating reaction term over the concentration range.' },
      { label: 'Adjoint gradient check', value: gErr, unit: 'relative', status: gErr < 1e-5 ? 'ok' : 'warn', help: 'Adjoint gradient against central finite differences at the initial parameters.' },
      { label: 'Closure inside the finite-volume solver', value: 100 * fvGap, unit: '% of inlet', status: fvGap < 0.02 ? 'ok' : 'warn', help: 'Final profile of the Crank–Nicolson/Picard solver with the learned closure against the training scheme.' },
      { label: 'Fitted first-order rate', value: kLin, unit: '1/s' }, { label: 'Outlet concentration, unseen case', value: UT[nt][nx - 1] }, { label: 'Network parameters', value: p.length },
    ],
    recommendations: [
      'Replace the synthetic experiments by measured profiles only after the closure has been recovered on a case like this one: it shows how much data and which concentration range are needed.',
      'A closure is only identified inside the range of concentrations seen in training (see the reaction-term plot).',
      'Use the workbench PDE task for the same equation with a reaction term you type yourself.',
    ],
    plots: [
      { type: 'line', title: 'Reaction term: learned closure and hidden law', xlabel: 'Concentration u', ylabel: 'R(u)', series: [{ name: 'Neural closure', x: uu, y: uu.map(net) }, { name: 'Hidden law (generates the data)', x: uu, y: uu.map(hidden), dash: true }, { name: 'Fitted first-order law', x: uu, y: uu.map((u) => -kLin * u), dash: true }] },
      { type: 'line', title: 'Final profiles: data and model', xlabel: 'x (m)', ylabel: 'u', series: [...exps.flatMap((ex) => [{ name: `Training data, inlet ${fmt(ex.cin)}`, x, y: last(ex), mode: 'points' }, { name: `Model, inlet ${fmt(ex.cin)}`, x, y: imexMarch(ex.op, ex.u0, nt, dt, net, theta)[nt] }]),
        { name: `Unseen data, inlet ${fmt(cTest)}`, x, y: last(exT), mode: 'points' }, { name: 'Prediction, unseen inlet', x, y: UT[nt], dash: true }, { name: 'First-order closure, unseen inlet', x, y: imexMarch(exT.op, exT.u0, nt, dt, (u) => -kLin * u, theta)[nt], dash: true }] },
      { type: 'field', title: 'Predicted field for the unseen inlet concentration', xlabel: 'x (m)', ylabel: 't (s)', zlabel: 'u', x, y: range(nt + 1).map((k) => k * dt), z: UT, cmap: 'salinity', contours: 8 },
      { type: 'line', title: 'Training loss', xlabel: 'Epoch', ylabel: 'Mean squared field error', logy: true, series: [{ name: 'Loss', x: hist.ep, y: hist.loss.map((q) => Math.max(q, 1e-300)) }] },
    ],
    tables: [
      { title: 'Field errors (RMS, relative to the inlet concentration)', columns: ['Model', 'Training experiments (%)', 'Unseen experiment (%)'], rows: [['Neural closure', 100 * eTrain, 100 * eNN], ['First-order closure R = −k·u', (100 * Math.sqrt(lossOf((u) => -kLin * u, exps))) / mean(cTrain), 100 * eLin], ['No reaction', (100 * Math.sqrt(lossOf(() => 0, exps))) / mean(cTrain), 100 * eNone]] },
      { title: 'Unseen experiment: outlet history', columns: ['Time (s)', 'Reference', 'Neural closure', 'First-order closure'], rows: (() => { const Ul = imexMarch(exT.op, exT.u0, nt, dt, (u) => -kLin * u, theta); return kObs.map((k, q) => [k * dt, exT.clean[q][nx - 1], UT[k][nx - 1], Ul[k][nx - 1]]); })() },
      { title: 'Formulation', columns: ['Item', 'Value'], rows: [['Equation', '∂u/∂t + v·∂u/∂x = D·∂²u/∂x² + R_θ(u)'], ['Closure', `R_θ(u) = r_ref·[W₂·tanh(W₁·u/u_ref + b₁) + b₂], ${H} hidden neurons`], ['Boundary conditions', 'inlet: fixed concentration; outlet: zero gradient'], ['Discretisation', `${nx} finite volumes, ${nt} steps, transport implicit (Crank–Nicolson), closure explicit`], ['Gradient', 'Discrete adjoint: transposed tri-diagonal solves backwards in time'], ['Data', `${cTrain.length} experiments × ${nObs} snapshots × ${nx} cells`]] },
    ],
    outputs: { task: 'npde', best: { unseenError: eNN, closureError: closErr, firstOrderRate: kLin }, objective: eNN, study: { a: UT[nt][nx - 1], b: eNN } },
  };
}

// ---- Task 15: reduced-order model of the transient polarisation layer (POD–Galerkin) ------------------------------------------------
/** Galerkin projection of du/dt = A·u + s on u = uRef + Σ aₖ·φₖ, integrated with Crank–Nicolson; returns the reduced coordinates at every step. */
export function galerkinROM(op, modes, uRef, nt, dt) {
  const r = modes.length, n = op.n, hom = { ...op, s0: 0, s1: 0 }, lin = (u) => u.map((q, i) => (i > 0 ? hom.lo[i] * u[i - 1] : 0) + hom.di[i] * q + (i < n - 1 ? hom.hi[i] * u[i + 1] : 0));
  const APhi = modes.map(lin), Ar = modes.map((pi) => APhi.map((ap) => dot(pi, ap))), g = modes.map((pi) => dot(pi, op.apply(uRef)));
  const Ml = range(r).map((i) => range(r).map((j) => (i === j ? 1 : 0) - 0.5 * dt * Ar[i][j])), Mr = range(r).map((i) => range(r).map((j) => (i === j ? 1 : 0) + 0.5 * dt * Ar[i][j])), Minv = invert(Ml), step = matMul(Minv, Mr), drive = matVec(Minv, g.map((q) => dt * q));
  const out = [zeros(r)];
  for (let k = 0; k < nt; k++) { const a = matVec(step, out[k]); out.push(a.map((q, i) => q + drive[i])); }
  return { a: out, Ar, lift: (a) => uRef.map((q, j) => { let s = q; for (let k = 0; k < r; k++) s += a[k] * modes[k][j]; return s; }) };
}
async function taskROM(v, ctx) {
  const delta = Math.max(1e-9, v.romDelta * 1e-6), D = Math.max(1e-15, v.romD * 1e-9), cb = Math.max(1e-9, v.romCb), tEnd = Math.max(1e-9, v.romT), nx = clamp(Math.round(v.romNx), 8, 400), nt = clamp(Math.round(v.romNt), 10, 5000), nSnap = clamp(Math.round(v.romSnap), 4, 60);
  const J = (lmh) => lmh / 3.6e6, jTrain = [v.romJ1, v.romJ2].map((q) => Math.max(0, q)), jTest = Math.max(0, v.romJt), left = { type: 'dirichlet', val: cb }, right = { type: 'noflux', val: 0 }, uRef = new Array(nx).fill(cb), dt = tEnd / nt;
  const fom = (lmh, save) => solveCDR({ L: delta, nx, tEnd, nt, v: J(lmh), D, ic: () => cb, left: { type: 'dirichlet', val: () => cb }, right: { type: 'noflux', val: () => 0 }, theta: 0.5, nSave: save });
  ctx.progress(0.1, 'Full-order snapshots'); await ctx.tick();
  const runs = jTrain.map((j) => fom(j, nSnap)), S = runs.flatMap((r) => r.U.slice(1).map((u) => u.map((q) => q - cb)));
  ctx.progress(0.35, 'Proper orthogonal decomposition'); await ctx.tick();
  const pod = podBasis(S, { center: false, rmax: 12 }), nModes = pod.modes.length;
  if (!nModes) throw new Error('The snapshots do not vary: set a non-zero flux so that a polarisation layer develops.');
  const r = v.romR >= 1 ? Math.min(nModes, Math.round(v.romR)) : Math.max(1, Math.min(nModes, pod.energy.findIndex((e) => e >= 0.999999) + 1 || nModes));
  const ref = fom(jTest, nt), opT = cdrOperator({ L: delta, nx, v: J(jTest), D, left, right }), norm2 = (U) => Math.sqrt(mean(U.flatMap((u) => u.map((q) => (q - cb) ** 2)))) || 1e-300;
  const romErr = (k) => { const R = galerkinROM(opT, pod.modes.slice(0, k), uRef, nt, dt); let s = 0, m = 0; for (let q = 0; q <= nt; q++) { const u = R.lift(R.a[q]); for (let i = 0; i < nx; i++) { s += (u[i] - ref.U[q][i]) ** 2; m++; } } return { rom: R, err: Math.sqrt(s / m) / norm2(ref.U) }; };
  const projErr = (k, snaps) => { const Ph = pod.modes.slice(0, k); let s = 0, m = 0; for (const f of snaps) { const a = Ph.map((ph) => dot(ph, f)); for (let i = 0; i < nx; i++) { let q = f[i]; for (let j = 0; j < k; j++) q -= a[j] * Ph[j][i]; s += q * q; m++; } } return Math.sqrt(s / m) / (Math.sqrt(mean(snaps.flatMap((f) => f.map((q) => q * q)))) || 1e-300); };
  ctx.progress(0.6, 'Galerkin reduced-order model at the unseen flux'); await ctx.tick();
  const ks = range(nModes).map((k) => k + 1), sweep = ks.map(romErr), pe = ks.map((k) => projErr(k, S)), peT = ks.map((k) => projErr(k, ref.U.map((u) => u.map((q) => q - cb)))), R = sweep[r - 1].rom, uEnd = R.lift(R.a[nt]);
  // consistency of the operator used for the projection with the finite-volume solver: full-order Crank–Nicolson march in the original space
  const full = imexMarch(opT, uRef, nt, dt, () => 0, 0.5), opGap = Math.max(...full[nt].map((q, i) => Math.abs(q - ref.u[i]))) / cb;
  const wall = (u) => 1.5 * u[nx - 1] - 0.5 * u[nx - 2], t = range(nt + 1).map((k) => k * dt), wF = ref.U.map(wall), wR = R.a.map((a) => wall(R.lift(a))), Pe = (J(jTest) * delta) / D, cpExact = Math.exp(Pe), tau = (delta * delta) / D;
  const W = [], e = sweep[r - 1].err;
  W.push({ level: 'info', msg: `Snapshots were taken at ${jTrain.map((q) => fmt(q)).join(' and ')} L/m²·h; the reduced model is evaluated at ${fmt(jTest)} L/m²·h, which is ${jTest >= Math.min(...jTrain) && jTest <= Math.max(...jTrain) ? 'inside' : 'OUTSIDE'} the sampled range.` });
  if (e > 0.02) W.push({ level: 'warn', msg: `The reduced model with ${r} mode${r > 1 ? 's' : ''} misses the full solution by ${fmt(100 * e, 3)} % — retain more modes or add snapshots closer to the target flux.` });
  if (tEnd < 2 * tau) W.push({ level: 'info', msg: `The simulated time (${fmt(tEnd)} s) is shorter than two diffusion times δ²/D = ${fmt(tau, 3)} s: the layer has not reached its steady state.` });
  return {
    summary: `Proper orthogonal decomposition of ${S.length} finite-volume snapshots gives ${nModes} modes; ${r} of them capture ${fmt(100 * pod.energy[r - 1], 8)} % of the fluctuation energy. The Galerkin reduced model (${r} unknowns instead of ${nx}) reproduces the full solution at an unseen flux to ${fmt(100 * e, 3)} % in space and time, and the wall concentration factor to ${fmt(wR[nt] / cb, 5)} against ${fmt(wF[nt] / cb, 5)}.`,
    warnings: W,
    kpis: [
      { label: 'Modes retained', value: r }, { label: 'Energy captured', value: 100 * pod.energy[r - 1], unit: '%' }, { label: 'Reduced-model error at the unseen flux', value: 100 * e, unit: '%', status: e < 0.02 ? 'ok' : 'warn', help: 'RMS difference to the full-order solution over all cells and time levels, relative to the RMS concentration rise.' },
      { label: 'Projection error of the training snapshots', value: 100 * pe[r - 1], unit: '%' }, { label: 'Projection error of the unseen solution', value: 100 * peT[r - 1], unit: '%', help: 'Best possible error of this basis for the unseen case; the Galerkin error cannot be lower.' },
      { label: 'Polarisation factor, reduced model', value: wR[nt] / cb, unit: '–' }, { label: 'Polarisation factor, full model', value: wF[nt] / cb, unit: '–' }, { label: 'Steady film-theory value exp(J·δ/D)', value: cpExact, unit: '–' },
      { label: 'Unknowns per time step', value: `${nx} → ${r}` }, { label: 'Operator consistency with the finite-volume solver', value: opGap, unit: 'relative', status: opGap < 1e-8 ? 'ok' : 'warn', help: 'Full-order march with the projected operator against the solver that produced the snapshots.' },
    ],
    recommendations: ['Use the reduced model where the full one is called thousands of times (optimisation, control, uncertainty): it is exact only inside the sampled parameter range.', 'Check the error-versus-modes plot: add modes until the Galerkin error meets the projection error.', 'The full transient solver with user-defined terms is in the workbench PDE task.'],
    plots: [
      { type: 'line', title: 'Energy not captured and error versus number of modes', xlabel: 'Number of modes', ylabel: 'Relative value', logy: true, series: [{ name: 'Energy not captured', x: ks, y: pod.energy.map((q) => Math.max(1 - q, 1e-16)), mode: 'both' }, { name: 'Galerkin error, unseen flux', x: ks, y: sweep.map((q) => Math.max(q.err, 1e-16)), mode: 'both' }, { name: 'Projection error, unseen flux', x: ks, y: peT.map((q) => Math.max(q, 1e-16)), mode: 'both', dash: true }], vlines: [{ x: r, label: 'retained' }] },
      { type: 'line', title: 'POD modes', xlabel: 'Distance from the bulk (µm)', ylabel: 'Mode shape', series: pod.modes.slice(0, Math.min(4, nModes)).map((m, k) => ({ name: `Mode ${k + 1} (σ = ${fmt(pod.sigma[k], 3)})`, x: opT.x.map((q) => q * 1e6), y: m })) },
      { type: 'line', title: 'Wall concentration at the unseen flux', xlabel: 'Time (s)', ylabel: 'Concentration at the membrane (g/L)', series: [{ name: `Full model (${nx} cells)`, x: t, y: wF }, { name: `Reduced model (${r} modes)`, x: t, y: wR, dash: true }], hlines: [{ y: cb * cpExact, label: 'steady film theory' }] },
      { type: 'line', title: 'Final profile at the unseen flux', xlabel: 'Distance from the bulk (µm)', ylabel: 'Concentration (g/L)', series: [{ name: 'Full model', x: opT.x.map((q) => q * 1e6), y: ref.u }, { name: 'Reduced model', x: opT.x.map((q) => q * 1e6), y: uEnd, mode: 'points' }] },
    ],
    tables: [
      { title: 'Modes, energy and errors', columns: ['Modes', 'Singular value', 'Cumulative energy (%)', 'Projection error, training (%)', 'Projection error, unseen (%)', 'Galerkin error, unseen (%)'], rows: ks.map((k, i) => [k, pod.sigma[i], 100 * pod.energy[i], 100 * pe[i], 100 * peT[i], 100 * sweep[i].err]) },
      { title: 'Formulation', columns: ['Item', 'Value'], rows: [['Full model', `∂c/∂t + J·∂c/∂x = D·∂²c/∂x², c = c_b at the bulk edge, zero salt flux at the membrane; ${nx} finite volumes, ${nt} Crank–Nicolson steps`], ['Snapshots', `${S.length} profiles from ${jTrain.length} training fluxes`], ['Decomposition', 'Method of snapshots: Jacobi eigen-decomposition of the snapshot covariance'], ['Reduced model', 'Galerkin projection of the finite-volume operator on the modes, lifted by the bulk concentration'], ['Film Péclet number J·δ/D', Pe], ['Diffusion time δ²/D (s)', tau]] },
    ],
    outputs: { task: 'rom', best: { modes: r, error: e, cpFactor: wR[nt] / cb }, objective: e, study: { a: wR[nt] / cb, b: wF[nt] / cb } },
  };
}

// ---- Shared data builder for the ensemble and multi-fidelity tasks -----------------------------------------------------------------
function doeDims(v, base) {
  return [{ key: 'recovery', label: 'Recovery (%)', lo: v.recLo, hi: v.recHi, nom: base.recovery }, { key: 'targetFlux', label: 'Average flux (L/m²·h)', lo: v.fluxLo, hi: v.fluxHi, nom: base.targetFlux },
    { key: 'T', label: 'Temperature (°C)', lo: Math.max(5, base.T - 8), hi: Math.min(42, base.T + 8), nom: base.T }, { key: 'salinityFactor', label: 'Salinity multiplier (×)', lo: 0.9, hi: 1.15, nom: 1 }].filter((d) => d.hi > d.lo);
}
const cubeTo = (dims, u) => u.map((q, j) => dims[j].lo + q * (dims[j].hi - dims[j].lo));
/** Zero-dimensional RO model (log-mean osmotic pressure, one average flux): the cheapest fidelity level. */
function lumpedRO(inp) {
  const R = clamp(inp.recovery / 100, 0.01, 0.98), piF = osmoticPressureIons(scaleIons(cloneIons(inp.ions), inp.salinityFactor ?? 1), inp.T) / 1e5, tc = 1 + 0.03 * (inp.T - 25), ln = -Math.log(1 - R) / R;
  const Pf = inp.Pp + 1.1 * piF * ln + inp.targetFlux / Math.max(0.05, inp.A * inp.ff * tc) + 0.7, erd = inp.erd === 'none' ? 0 : inp.erdEff / 100, etaM = (inp.etaMotor ?? 95) / 100;
  const sec = (Pf / (36 * (inp.etaPump / 100) * etaM)) * (inp.erd === 'px' ? 1 + ((1 - erd) * (1 - R)) / R : (1 - erd * (inp.etaPump / 100) * (1 - R)) / R), Bs = inp.B * tc;
  return { sec, Pf, tds: (Bs * tds(scaleIons(cloneIons(inp.ions), inp.salinityFactor ?? 1)) * ln * 1.1) / (inp.targetFlux + Bs) };
}
const MF_TARGET = { sec: { label: 'Specific energy', unit: 'kWh/m³' }, Pf: { label: 'Feed pressure', unit: 'bar' }, tds: { label: 'Product TDS', unit: 'mg/L' } };

// ---- Task 16: ensemble models (bagging, out-of-bag error, model averaging) ---------------------------------------------------------
async function taskENS(v, ctx) {
  const seed = Math.round(v.seed), fromRO = v.mlSource !== 'table';
  let X = [], y = [], names, yName, yUnit;
  if (fromRO) {
    const base = roBase(v), tg = ML_TARGET[v.mlTarget] || ML_TARGET.sec, dims = doeDims(v, base), U = lhs(clamp(Math.round(v.mlN), 20, 400), dims.length, seed);
    if (dims.length < 2) throw new Error('Widen the recovery and flux bounds: the design-of-experiments needs at least two varying inputs.');
    for (let i = 0; i < U.length; i++) { const x = cubeTo(dims, U[i]), e = evalDesign({ ...base, ...Object.fromEntries(dims.map((d, j) => [d.key, x[j]])) }, v); if (e.ok && Number.isFinite(tg.f(e))) { X.push(x); y.push(tg.f(e)); } if (i % 12 === 11) { ctx.progress(0.02 + (0.4 * i) / U.length, `Plant simulation ${i + 1} of ${U.length}`); await ctx.tick(); } }
    names = dims.map((d) => d.label); yName = tg.label; yUnit = tg.unit;
  } else {
    const rows = numRows(v.mlData, ['y']), cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'].filter((k) => rows.length && rows.every((r) => isNum(r[k])) && new Set(rows.map((r) => r[k])).size > 1);
    if (!cols.length) throw new Error('The data table needs at least one fully numeric, non-constant input column (x1…x6) and a y column.');
    X = rows.map((r) => cols.map((k) => r[k])); y = rows.map((r) => r.y); names = cols; yName = 'y'; yUnit = '';
  }
  const n = X.length, d = X[0]?.length || 0;
  if (n < Math.max(16, d + 10)) throw new Error(`Only ${n} usable data rows: at least ${Math.max(16, d + 10)} are needed for an ensemble with a test set.`);
  const perm = shuffled(n, rng(seed + 1)), nTe = Math.max(4, Math.round((n * clamp(v.ensTest, 5, 40)) / 100)), iTe = perm.slice(0, nTe), iTr = perm.slice(nTe), take = (idx, A) => idx.map((i) => A[i]);
  const Xtr = take(iTr, X), ytr = take(iTr, y), Xte = take(iTe, X), yte = take(iTe, y), B = clamp(Math.round(v.ensB), 3, 60), hid = clamp(Math.round(v.ensH), 2, 24), ep = clamp(Math.round(v.ensEpochs), 20, 3000);
  const learner = (kind) => (Xa, ya, b) => (kind === 'poly' ? polyFit(Xa, ya).predict : nnTrain(Xa, ya, { hidden: [hid], epochs: ep, lr: 0.02, seed: seed + 100 + b }).predict);
  const fitM = v.ensBase === 'poly' ? learner('poly') : v.ensBase === 'mix' ? (Xa, ya, b) => learner(b % 2 ? 'poly' : 'nn')(Xa, ya, b) : learner('nn');
  ctx.progress(0.45, `Training ${B} bootstrap members`); await ctx.tick();
  const bag = bagEnsemble(Xtr, ytr, fitM, { B, seed: seed + 3 });
  ctx.progress(0.8, 'Reference models and averaging'); await ctx.tick();
  const single = fitM(Xtr, ytr, 0), P = Xte.map((x) => bag.predict(x)), eEns = rmse(P.map((p) => p.mean), yte), eSingle = rmse(Xte.map(single), yte), eMem = range(B).map((b) => rmse(P.map((p) => p.all[b]), yte));
  const mseMem = mean(eMem.map((q) => q * q)), ambiguity = mean(P.map((p) => mean(p.all.map((q) => (q - p.mean) ** 2)))), ident = Math.abs(mseMem - eEns * eEns - ambiguity) / Math.max(1e-300, mseMem);
  const bySize = range(B).map((k) => rmse(Xte.map((x) => bag.predict(x, k + 1).mean), yte)), absErr = P.map((p, i) => Math.abs(p.mean - yte[i])), cover = (100 * P.filter((p, i) => absErr[i] <= 2 * p.sd).length) / nTe, rho = finite(spearman(absErr, P.map((p) => p.sd)));
  // averaging of the three surrogate families trained once on the same rows
  const fam = [['Polynomial response surface', polyFit(Xtr, ytr).predict], ['Gaussian process', ((gp) => (x) => gp.predict(x).mean)(gpFit(Xtr, ytr))], ['Neural network', nnTrain(Xtr, ytr, { hidden: [hid], epochs: ep, lr: 0.02, seed }).predict]], eFam = fam.map(([, f]) => rmse(Xte.map(f), yte)), avg = (x) => mean(fam.map(([, f]) => f(x))), eAvg = rmse(Xte.map(avg), yte);
  const span = Math.max(...y) - Math.min(...y) || 1, bl = rmse(yte.map(() => mean(ytr)), yte), W = [], j0 = 0, xs = linspace(Math.min(...X.map((x) => x[j0])), Math.max(...X.map((x) => x[j0])), 30), mid = range(d).map((j) => mean(X.map((x) => x[j]))), sl = xs.map((q) => bag.predict(mid.map((m, j) => (j === j0 ? q : m))));
  W.push({ level: 'info', msg: `Each member saw a bootstrap resample; ${bag.oob.n} of ${Xtr.length} training rows have an out-of-bag prediction. The out-of-bag error (${fmt(bag.oob.rmse, 3)}) estimates the test error (${fmt(eEns, 3)}) without a test set.` });
  if (bag.oob.rmse > 2 * eEns || eEns > 2 * bag.oob.rmse) W.push({ level: 'warn', msg: `The out-of-bag estimate (${fmt(bag.oob.rmse, 3)}) and the test error (${fmt(eEns, 3)}) differ by more than a factor of two: the test set is small or contains points unlike the training rows — do not rely on either figure alone.` });
  if (cover < 60) W.push({ level: 'warn', msg: `Only ${fmt(cover, 3)} % of the test points fall inside ± 2 member standard deviations: the spread measures disagreement between members, not the full prediction error — do not use it as a confidence interval without calibration.` });
  if (eEns > eSingle * 1.02) W.push({ level: 'info', msg: 'The bagged ensemble is not more accurate than one model trained on all rows: this learner is already stable on these data (bagging helps high-variance learners).' });
  return {
    summary: `A bagged ensemble of ${B} ${v.ensBase === 'poly' ? 'response surfaces' : v.ensBase === 'mix' ? 'networks and response surfaces' : 'neural networks'} predicts ${lcFirst(yName)} with a test RMSE of ${fmt(eEns, 3)} ${yUnit} (single model ${fmt(eSingle, 3)}, mean member ${fmt(Math.sqrt(mseMem), 3)}); the out-of-bag estimate is ${fmt(bag.oob.rmse, 3)} and the mean member spread ${fmt(mean(P.map((p) => p.sd)), 3)} ${yUnit}.`,
    warnings: W,
    kpis: [
      { label: 'Ensemble test RMSE', value: eEns, unit: yUnit, status: eEns <= eSingle * 1.02 ? 'ok' : 'warn' }, { label: 'Single-model test RMSE', value: eSingle, unit: yUnit }, { label: 'Mean member test RMSE', value: Math.sqrt(mseMem), unit: yUnit },
      { label: 'Out-of-bag RMSE', value: bag.oob.rmse, unit: yUnit, help: 'Each training row predicted only by the members that did not see it.' }, { label: 'Mean member spread', value: mean(P.map((p) => p.sd)), unit: yUnit },
      { label: 'Test points inside ± 2 spreads', value: cover, unit: '%', status: cover >= 60 ? 'ok' : 'warn' }, { label: 'Rank correlation of spread and error', value: rho, unit: '–' },
      { label: 'Ambiguity decomposition residual', value: ident, unit: 'relative', status: ident < 1e-9 ? 'ok' : 'warn', help: 'Mean member MSE = ensemble MSE + mean member variance must hold exactly.' },
      { label: 'Averaged surrogate families, test RMSE', value: eAvg, unit: yUnit, help: 'Equal-weight average of response surface, Gaussian process and network.' }, { label: 'Test RMSE of the mean (no model)', value: bl, unit: yUnit }, { label: 'Ensemble error relative to the data range', value: (100 * eEns) / span, unit: '%' },
    ],
    recommendations: ['Use the ensemble mean for prediction and the member spread as a flag for inputs where the models disagree (sparse data, extrapolation).', 'The out-of-bag error is a free validation estimate: compare it with the test error before trusting it on new data.', 'Bagging lowers variance, not bias: if all members share the same systematic error, add data or change the model family.'],
    plots: [
      { type: 'line', title: 'Test error versus ensemble size', xlabel: 'Members averaged', ylabel: `Test RMSE (${yUnit || 'units of y'})`, series: [{ name: 'Bagged ensemble', x: range(B).map((k) => k + 1), y: bySize, mode: 'both' }], hlines: [{ y: eSingle, label: 'single model' }, { y: bag.oob.rmse, label: 'out-of-bag estimate' }] },
      { type: 'line', title: 'Predicted against actual (test set)', xlabel: `Actual ${yName}`, ylabel: `Predicted ${yName}`, series: [{ name: 'Ensemble mean', x: yte, y: P.map((p) => p.mean), mode: 'points' }, { name: 'Single model', x: yte, y: Xte.map(single), mode: 'points' }, { name: 'Perfect', x: [Math.min(...yte), Math.max(...yte)], y: [Math.min(...yte), Math.max(...yte)], dash: true }] },
      { type: 'line', title: 'Member spread against absolute error (test set)', xlabel: 'Member standard deviation', ylabel: 'Absolute error of the ensemble mean', series: [{ name: 'Test points', x: P.map((p) => p.sd), y: absErr, mode: 'points' }] },
      { type: 'line', title: `Ensemble along ${names[j0]} (other inputs at their mean)`, xlabel: names[j0], ylabel: `${yName}${yUnit ? ` (${yUnit})` : ''}`, series: [{ name: 'Ensemble mean', x: xs, y: sl.map((p) => p.mean) }, { name: 'Mean + 2 spreads', x: xs, y: sl.map((p) => p.mean + 2 * p.sd), dash: true }, { name: 'Mean − 2 spreads', x: xs, y: sl.map((p) => p.mean - 2 * p.sd), dash: true }, ...range(Math.min(4, B)).map((b) => ({ name: `Member ${b + 1}`, x: xs, y: sl.map((p) => p.all[b]), dash: true }))] },
      { type: 'bar', title: 'Test RMSE of members and combinations', ylabel: yUnit || 'units of y', categories: ['Best member', 'Mean member', 'Worst member', 'Bagged ensemble', 'Single model', ...fam.map((f) => f[0]), 'Family average'], series: [{ name: 'Test RMSE', values: [Math.min(...eMem), Math.sqrt(mseMem), Math.max(...eMem), eEns, eSingle, ...eFam, eAvg] }] },
    ],
    tables: [
      { title: 'Ensemble evidence', columns: ['Quantity', 'Value'], rows: [['Training rows / test rows', `${Xtr.length} / ${nTe}`], ['Members', B], ['Mean member MSE', mseMem], ['Ensemble MSE', eEns * eEns], ['Mean member variance (ambiguity)', ambiguity], ['Out-of-bag RMSE', bag.oob.rmse], ['Test RMSE of the ensemble', eEns], ['Rows with an out-of-bag prediction', bag.oob.n]], note: 'Ambiguity decomposition (Krogh & Vedelsby): the ensemble error equals the mean member error minus the mean member variance, so an ensemble is never worse than its average member.' },
      { title: 'Surrogate families and their average', columns: ['Model', 'Test RMSE', 'Relative to data range (%)'], rows: [...fam.map((f, i) => [f[0], eFam[i], (100 * eFam[i]) / span]), ['Equal-weight average', eAvg, (100 * eAvg) / span], ['Bagged ensemble', eEns, (100 * eEns) / span]] },
      { title: 'Test predictions', columns: [...names, `Actual ${yName}`, 'Ensemble mean', 'Member spread', 'Single model'], rows: Xte.slice(0, 30).map((x, i) => [...x, yte[i], P[i].mean, P[i].sd, single(x)]) },
    ],
    outputs: { task: 'ens', best: { rmse: eEns, oobRmse: bag.oob.rmse, singleRmse: eSingle, members: B }, objective: eEns, study: { a: eEns, b: bag.oob.rmse } },
  };
}

// ---- Task 17: multi-fidelity surrogate (recursive co-kriging) ------------------------------------------------------------------------
async function taskMF(v, ctx) {
  const seed = Math.round(v.seed), base = roBase(v), dims = doeDims(v, base), d = dims.length, key = MF_TARGET[v.mfTarget] ? v.mfTarget : 'sec', tg = MF_TARGET[key], segH = clamp(Math.round(v.mfSegH), 2, 6), lumped = v.mfLow === 'lumped';
  if (d < 2) throw new Error('Widen the recovery and flux bounds: the design-of-experiments needs at least two varying inputs.');
  const inpOf = (x) => ({ ...base, ...Object.fromEntries(dims.map((q, j) => [q.key, x[j]])) }), pick = (e) => (key === 'sec' ? e.m.sec : key === 'Pf' ? e.m.Pf : e.m.tds);
  const high = (x) => { const e = evalDesign({ ...inpOf(x), nSeg: segH }, v); return e.ok ? pick(e) : NaN; }, low = (x) => { if (lumped) return lumpedRO(inpOf(x))[key]; const e = evalDesign({ ...inpOf(x), nSeg: 1 }, v); return e.ok ? pick(e) : NaN; };
  const nH = clamp(Math.round(v.mfNH), d + 2, 60), nL = Math.max(nH + 2, clamp(Math.round(v.mfNL), 6, 300)), nT = clamp(Math.round(v.mfNT), 5, 100), ratio = lumped ? 0 : 1 / segH, nS = Math.max(nH, Math.round(nH + ratio * nL));
  const sample = async (U, f, msg, p0, p1) => { const X = [], y = []; for (let i = 0; i < U.length; i++) { const x = cubeTo(dims, U[i]), q = f(x); if (Number.isFinite(q)) { X.push(x); y.push(q); } if (i % 6 === 5) { ctx.progress(p0 + ((p1 - p0) * i) / U.length, `${msg} ${i + 1} of ${U.length}`); await ctx.tick(); } } return { X, y }; };
  const UH = lhs(nH, d, seed), UL = [...UH, ...lhs(nL - nH, d, seed + 1)]; // nested design: every high-fidelity point is also a low-fidelity point
  const tL0 = Date.now(), Lo = await sample(UL, low, 'Low-fidelity run', 0.02, 0.2), tL = (Date.now() - tL0) / Math.max(1, UL.length), tH0 = Date.now(), Hi = await sample(UH, high, 'High-fidelity run', 0.2, 0.4), tH = (Date.now() - tH0) / Math.max(1, UH.length);
  const Te = await sample(lhs(nT, d, seed + 2), high, 'Test run', 0.4, 0.65), Sf = nS > nH ? await sample(lhs(nS, d, seed + 3), high, 'Equal-cost single-fidelity run', 0.65, 0.85) : Hi;
  if (Hi.X.length < d + 2 || Te.X.length < 4 || Lo.X.length < 6) throw new Error('Too few simulations converged: widen the bounds or check the plant inputs.');
  ctx.progress(0.88, 'Co-kriging and reference surrogates'); await ctx.tick();
  const ck = coKrige(Lo.X, Lo.y, Hi.X, Hi.y), gpH = gpFit(Hi.X, Hi.y), gpS = Sf === Hi ? gpH : gpFit(Sf.X, Sf.y);
  const pMF = Te.X.map((x) => ck.predict(x)), eMF = rmse(pMF.map((p) => p.mean), Te.y), eH = rmse(Te.X.map((x) => gpH.predict(x).mean), Te.y), eS = rmse(Te.X.map((x) => gpS.predict(x).mean), Te.y), loT = Te.X.map(low), eL = rmse(loT, Te.y), eLgp = rmse(pMF.map((p) => p.low), Te.y);
  const span = Math.max(...Te.y) - Math.min(...Te.y) || 1, cover = (100 * pMF.filter((p, i) => Math.abs(p.mean - Te.y[i]) <= 2 * p.sd + 1e-12).length) / Te.y.length, win = eMF < eS, W = [];
  const j0 = 0, xs = linspace(dims[j0].lo, dims[j0].hi, 25), at = (q) => dims.map((dm, j) => (j === j0 ? q : clamp(dm.nom, dm.lo, dm.hi))), slMF = xs.map((q) => ck.predict(at(q))), xs7 = linspace(dims[j0].lo, dims[j0].hi, 7), hiSl = xs7.map((q) => high(at(q)));
  W.push({ level: 'info', msg: `Cost accounting: one low-fidelity run counts as ${lumped ? '0' : fmt(ratio, 3)} high-fidelity run${lumped ? ' (algebraic model)' : ` (1 of ${segH} segments per element)`}, so ${nH} high + ${Lo.X.length} low runs cost as much as ${nS} high-fidelity runs; measured times were ${fmt(tL, 3)} and ${fmt(tH, 3)} ms per run.` });
  if (!win) W.push({ level: 'warn', msg: `At equal cost the single-fidelity Gaussian process (${nS} runs) is as accurate as the co-kriging model: the low-fidelity model is ${lumped ? 'too different from' : 'too expensive relative to'} the high-fidelity one to pay off here.` });
  if (Math.abs(ck.rho) < 0.3) W.push({ level: 'warn', msg: `The scaling factor ρ = ${fmt(ck.rho, 3)} is small: the two fidelity levels are poorly correlated.` });
  return {
    summary: `Co-kriging of ${Hi.X.length} high-fidelity and ${Lo.X.length} low-fidelity runs predicts ${lcFirst(tg.label)} with a test RMSE of ${fmt(eMF, 3)} ${tg.unit}, against ${fmt(eH, 3)} for a Gaussian process on the same ${Hi.X.length} high-fidelity runs and ${fmt(eS, 3)} for one given the same total cost (${nS} runs); the low-fidelity model alone is off by ${fmt(eL, 3)} ${tg.unit}.`,
    warnings: W,
    kpis: [
      { label: 'Multi-fidelity test RMSE', value: eMF, unit: tg.unit, status: win ? 'ok' : 'warn' }, { label: 'Single fidelity, same high-fidelity runs', value: eH, unit: tg.unit }, { label: 'Single fidelity, equal cost', value: eS, unit: tg.unit },
      { label: 'Low-fidelity model alone', value: eL, unit: tg.unit, help: 'Systematic error of the cheap model against the high-fidelity test runs.' }, { label: 'Error reduction at equal cost', value: (100 * (eS - eMF)) / Math.max(1e-300, eS), unit: '%', status: win ? 'ok' : 'warn' },
      { label: 'Scaling factor ρ', value: ck.rho, unit: '–', help: 'y_high ≈ ρ·y_low + δ(x)' }, { label: 'Discrepancy model', value: ck.kind, help: 'Form of δ(x) chosen by cross-validation on the high-fidelity runs: constant, linear trend or Gaussian process.' }, { label: 'Test points inside ± 2 σ', value: cover, unit: '%' }, { label: 'Equivalent high-fidelity cost', value: nS, unit: 'runs' },
      { label: 'Multi-fidelity error relative to range', value: (100 * eMF) / span, unit: '%' }, { label: 'High / low fidelity runs', value: `${Hi.X.length} / ${Lo.X.length}` },
    ],
    recommendations: [win ? 'Spend the simulation budget on many cheap runs plus a few accurate ones: the discrepancy is smoother than the response itself.' : 'For this response the accurate model is cheap enough to sample directly.', 'Place the next high-fidelity run where the co-kriging standard deviation is largest.', 'Use the surrogate in the optimisation and uncertainty tasks only inside the sampled bounds.'],
    plots: [
      { type: 'line', title: 'Predicted against high-fidelity test runs', xlabel: `High-fidelity ${tg.label} (${tg.unit})`, ylabel: `Predicted (${tg.unit})`, series: [{ name: 'Co-kriging', x: Te.y, y: pMF.map((p) => p.mean), mode: 'points' }, { name: 'Single fidelity, equal cost', x: Te.y, y: Te.X.map((x) => gpS.predict(x).mean), mode: 'points' }, { name: 'Low-fidelity model', x: Te.y, y: loT, mode: 'points' }, { name: 'Perfect', x: [Math.min(...Te.y), Math.max(...Te.y)], y: [Math.min(...Te.y), Math.max(...Te.y)], dash: true }] },
      { type: 'line', title: `Fidelity levels along ${dims[j0].label}`, xlabel: dims[j0].label, ylabel: `${tg.label} (${tg.unit})`, series: [{ name: 'Co-kriging mean', x: xs, y: slMF.map((p) => p.mean) }, { name: 'Co-kriging ± 2 σ (upper)', x: xs, y: slMF.map((p) => p.mean + 2 * p.sd), dash: true }, { name: 'Co-kriging ± 2 σ (lower)', x: xs, y: slMF.map((p) => p.mean - 2 * p.sd), dash: true },
        { name: 'Low-fidelity model', x: xs, y: xs.map((q) => low(at(q))), dash: true }, { name: 'High-fidelity model', x: xs7.filter((_, i) => Number.isFinite(hiSl[i])), y: hiSl.filter(Number.isFinite), mode: 'points' }, { name: 'Single fidelity, equal cost', x: xs, y: xs.map((q) => gpS.predict(at(q)).mean), dash: true }] },
      { type: 'bar', title: 'Test RMSE by model', ylabel: tg.unit, categories: ['Low fidelity alone', `GP on ${Hi.X.length} high runs`, `GP on ${Sf.X.length} high runs (equal cost)`, 'Co-kriging'], series: [{ name: 'Test RMSE', values: [eL, eH, eS, eMF] }] },
      { type: 'line', title: 'Discrepancy δ(x) = y_high − ρ·y_low at the test points', xlabel: `High-fidelity ${tg.label} (${tg.unit})`, ylabel: `δ (${tg.unit})`, series: [{ name: 'Actual', x: Te.y, y: Te.y.map((q, i) => q - ck.rho * pMF[i].low), mode: 'points' }, { name: 'Modelled', x: Te.y, y: pMF.map((p) => p.delta), mode: 'points' }] },
    ],
    tables: [
      { title: 'Models compared on the high-fidelity test set', columns: ['Model', 'High runs', 'Low runs', 'Cost (high-fidelity runs)', `Test RMSE (${tg.unit})`, 'Relative to range (%)'], rows: [['Low-fidelity model alone', 0, 0, 0, eL, (100 * eL) / span], ['Gaussian process on the low-fidelity runs', 0, Lo.X.length, ratio * Lo.X.length, eLgp, (100 * eLgp) / span], ['Gaussian process, high fidelity only', Hi.X.length, 0, Hi.X.length, eH, (100 * eH) / span], ['Gaussian process, equal cost', Sf.X.length, 0, Sf.X.length, eS, (100 * eS) / span], ['Co-kriging (ρ·low + δ)', Hi.X.length, Lo.X.length, Hi.X.length + ratio * Lo.X.length, eMF, (100 * eMF) / span]],
        note: `High fidelity: element model of suite 1 with ${segH} segments per element. Low fidelity: ${lumped ? 'algebraic zero-dimensional model (log-mean osmotic pressure, one average flux)' : 'the same model with one segment per element'}. Recursive co-kriging: a Gaussian process on the low-fidelity runs, ρ by least squares at the nested high-fidelity points and a discrepancy δ(x) whose form (constant, linear trend or Gaussian process) is chosen by cross-validation — here: ${ck.kind}.` },
      { title: 'Test points', columns: [...dims.map((q) => q.label), 'High fidelity', 'Low fidelity', 'Co-kriging', 'σ', 'Single fidelity, equal cost'], rows: Te.X.slice(0, 30).map((x, i) => [...x, Te.y[i], loT[i], pMF[i].mean, pMF[i].sd, gpS.predict(x).mean]) },
    ],
    outputs: { task: 'mf', best: { rmse: eMF, singleRmse: eS, rho: ck.rho, lowRmse: eL }, objective: eMF, study: { a: eMF, b: eS } },
  };
}

// ---- Task 18: robust / stochastic design (sample-average approximation, chance constraints, worst case) with batched parallel evaluation --
async function taskROB(v, ctx) {
  const base = roBase(v), seed = Math.round(v.seed), O = OBJ[v.objective] || OBJ.cost, CL = consFor(v), worst = v.robMode === 'worst', S = clamp(Math.round(v.robS), 2, 64), nTest = clamp(Math.round(v.robTest), 4, 200), alpha = worst ? 0 : clamp(v.robAlpha, 0, 50) / 100;
  const vars = [{ key: 'recovery', label: 'Recovery', unit: '%', lo: v.recLo, hi: v.recHi, x0: v.recovery0 }, { key: 'targetFlux', label: 'Average flux', unit: 'L/m²·h', lo: v.fluxLo, hi: v.fluxHi, x0: v.flux0 }].filter((d) => d.hi > d.lo), n = vars.length;
  if (!n) throw new Error('No decision variable is free: widen the recovery or flux bounds.');
  const scen = (U) => U.map((u) => ({ salinityFactor: 1 + (v.robSal / 100) * clamp(normInv(u[0]), -2.5, 2.5), T: clamp(base.T + v.robT * clamp(normInv(u[1]), -2.5, 2.5), 1, 45), ff: base.ff * (1 - (v.robDecl / 100) * u[2]) }));
  const nominal = [{ salinityFactor: 1, T: base.T, ff: base.ff }], scIn = scen(lhs(S, 3, seed + 11)), scOut = scen(lhs(nTest, 3, seed + 97));
  const design = (u) => Object.fromEntries(vars.map((d, j) => [d.key, d.lo + clamp(u[j], 0, 1) * (d.hi - d.lo)]));
  const bat = await makeBatcher(v, base, v.parWorkers, ctx), rho = 20;
  try {
    const eBase = evalDesign(base, v);
    if (!eBase.ok) throw new Error('The base design cannot be simulated: ' + (eBase.err || 'check the RO plant inputs.'));
    const fScale = Math.abs(O.f(eBase)) || 1, allowed = (m) => Math.floor(alpha * m + 1e-9);
    /** Sample-average statistics of one design over a scenario set. */
    const stats = (es, robust) => {
      const f = es.map((e) => (e.ok ? O.f(e) : 1e3 * fScale)), gmax = es.map((e) => Math.max(...e.g)), cv = chanceViolation(gmax, robust ? alpha : 0);
      const viol = cv.viol, obj = (worst && robust ? Math.max(...f) : mean(f)) / fScale;
      return { f, gmax, mean: mean(f), worst: Math.max(...f), pViol: gmax.filter((q) => q > FEAS).length / es.length, viol, merit: obj + rho * viol + 10 * rho * cv.quad, gWorst: range(CL.length).map((j) => Math.max(...es.map((e) => e.g[j]))) };
    };
    const evalMany = async (us, sc, robust) => { const jobs = us.flatMap((u) => sc.map((s) => ({ ...design(u), ...s }))), out = await bat.run(jobs); return us.map((_, i) => stats(out.slice(i * sc.length, (i + 1) * sc.length), robust)); };
    /** Generation-synchronous differential evolution: every generation is one batch of (candidates × scenarios) plant simulations. */
    const search = async (sc, robust, sd, label, p0, p1) => {
      const g = rng(sd), NP = clamp(Math.round(v.robPop), 4, 40), G = clamp(Math.round(v.robGens), 1, 60), hist = [];
      let P = lhs(NP, n, sd); P[0] = vars.map((d) => clamp((d.x0 - d.lo) / (d.hi - d.lo), 0, 1));
      let F = await evalMany(P, sc, robust);
      hist.push(Math.min(...F.map((q) => q.merit)));
      for (let gen = 0; gen < G; gen++) {
        const trial = P.map((x, i) => { const pick = () => { let k; do k = g.int(NP); while (k === i); return k; }, a = P[pick()], b = P[pick()], c = P[pick()], jr = g.int(n); return x.map((q, j) => (g.uniform() < 0.9 || j === jr ? clamp(a[j] + 0.6 * (b[j] - c[j]), 0, 1) : q)); });
        const Ft = await evalMany(trial, sc, robust);
        for (let i = 0; i < NP; i++) if (Ft[i].merit <= F[i].merit) { P[i] = trial[i]; F[i] = Ft[i]; }
        hist.push(Math.min(...F.map((q) => q.merit)));
        ctx.progress(p0 + ((p1 - p0) * (gen + 1)) / G, `${label}: generation ${gen + 1} of ${G}`); await ctx.tick();
      }
      const ib = range(NP).reduce((a, b) => (F[b].merit < F[a].merit ? b : a)), nPol = clamp(Math.round(v.robPolish), 0, 20);
      let ub = P[ib], Fb = F[ib], step = 0.1;
      for (let it = 0; it < nPol; it++) { // pattern search around the best point: one batch of 2·n neighbours per refinement
        const cand = range(n).flatMap((j) => [-1, 1].map((sg) => ub.map((q, k) => (k === j ? clamp(q + sg * step, 0, 1) : q)))), Fc = await evalMany(cand, sc, robust), kb = range(cand.length).reduce((a, b) => (Fc[b].merit < Fc[a].merit ? b : a));
        if (Fc[kb].merit < Fb.merit - 1e-12) { ub = cand[kb]; Fb = Fc[kb]; } else step *= 0.5;
        hist.push(Fb.merit);
        ctx.progress(p1, `${label}: refinement ${it + 1} of ${nPol}`); await ctx.tick();
      }
      return { u: ub, st: Fb, hist };
    };
    ctx.progress(0.05, 'Nominal (deterministic) optimum'); await ctx.tick();
    const nom = await search(nominal, false, seed, 'Nominal design', 0.05, 0.25), rob = await search(scIn, true, seed + 1, 'Robust design', 0.25, 0.85);
    ctx.progress(0.88, 'Out-of-sample validation'); await ctx.tick();
    const [nomIn, robNom] = [(await evalMany([nom.u], scIn, false))[0], (await evalMany([rob.u], nominal, false))[0]], [nomOut, robOut] = await evalMany([nom.u, rob.u], scOut, false);
    const dN = design(nom.u), dR = design(rob.u), eN = evalDesign({ ...base, ...dN }, v), eR = evalDesign({ ...base, ...dR }, v), show = O.show || ((x) => x);
    if (!eN.ok || !eR.ok) throw new Error('The optimal designs could not be re-simulated: widen the bounds or check the plant inputs.');
    const price = (100 * (robNom.mean - nom.st.mean)) / Math.abs(nom.st.mean || 1), feasR = rob.st.pViol <= alpha + 1e-9, sp = bat.speedup(), W = [];
    if (!feasR) W.push({ level: 'bad', msg: `No design inside the bounds keeps the violation probability below ${fmt(100 * alpha, 3)} % over the ${S} scenarios (best found: ${fmt(100 * rob.st.pViol, 3)} %). Relax a limit, widen the bounds or reduce the uncertainty.` });
    else W.push({ level: 'info', msg: `The robust design violates a limit in ${fmt(100 * rob.st.pViol, 3)} % of the ${S} design scenarios (allowed ${fmt(100 * alpha, 3)} %) and in ${fmt(100 * robOut.pViol, 3)} % of ${nTest} fresh scenarios; the nominal optimum fails in ${fmt(100 * nomOut.pViol, 3)} % of them.` });
    if (robOut.pViol > alpha + 2 * Math.sqrt((alpha * (1 - alpha) + 0.01) / nTest)) W.push({ level: 'warn', msg: 'The out-of-sample violation probability is clearly above the target: the sample-average approximation used too few scenarios — increase the scenario count.' });
    W.push(bat.pool.live && !bat.fellBack ? { level: 'info', msg: `Plant simulations ran on ${bat.pool.workers} worker threads (${bat.jobs} runs in ${bat.wall} ms after ${bat.pool.startMs} ms start-up); the same chunk on the calling thread gave identical numbers (largest difference ${fmt(bat.agree ?? 0, 2)}).` }
      : { level: 'info', msg: v.parWorkers >= 1 ? `Parallel evaluation was requested but ${bat.pool.reason || 'the pool stopped responding'}: the ${bat.jobs} runs were evaluated on the calling thread in ${bat.chunks} chunks.` : `${bat.jobs} plant simulations evaluated on the calling thread in batches (set “Worker threads” above 0 to evaluate them in parallel).` });
    const cRow = (e) => CL.map((c) => c.val(e.m)), scRow = (s, i, a, b) => [i + 1, s.salinityFactor, s.T, s.ff, 100 * a.gmax[i], 100 * b.gmax[i], show(a.f[i]), show(b.f[i])];
    return {
      summary: `${worst ? 'Worst-case' : 'Chance-constrained'} design over ${S} sampled scenarios: recovery ${fmt(dR.recovery ?? base.recovery, 4)} % and flux ${fmt(dR.targetFlux ?? base.targetFlux, 4)} L/m²·h instead of the nominal optimum ${fmt(dN.recovery ?? base.recovery, 4)} % and ${fmt(dN.targetFlux ?? base.targetFlux, 4)} L/m²·h. The price of robustness is ${fmt(price, 3)} % in ${lcFirst(O.label)} at nominal conditions; the out-of-sample violation probability falls from ${fmt(100 * nomOut.pViol, 3)} % to ${fmt(100 * robOut.pViol, 3)} %.`,
      warnings: W,
      kpis: [
        { label: `Robust design: ${lcFirst(O.label)} at nominal conditions`, value: show(robNom.mean), unit: O.unit, status: feasR ? 'ok' : 'bad' }, { label: `Nominal optimum: ${lcFirst(O.label)}`, value: show(nom.st.mean), unit: O.unit },
        { label: 'Price of robustness', value: price, unit: '%', help: 'Loss of the objective at nominal conditions accepted to stay feasible under uncertainty.' }, { label: `Robust design: expected ${lcFirst(O.label)}`, value: show(rob.st.mean), unit: O.unit },
        { label: 'Robust recovery', value: dR.recovery ?? base.recovery, unit: '%' }, { label: 'Robust average flux', value: dR.targetFlux ?? base.targetFlux, unit: 'L/m²·h' }, { label: 'Nominal recovery', value: dN.recovery ?? base.recovery, unit: '%' }, { label: 'Nominal average flux', value: dN.targetFlux ?? base.targetFlux, unit: 'L/m²·h' },
        { label: 'Violation probability, robust (out of sample)', value: 100 * robOut.pViol, unit: '%', status: robOut.pViol <= alpha + 0.1 ? 'ok' : 'warn' }, { label: 'Violation probability, nominal (out of sample)', value: 100 * nomOut.pViol, unit: '%', status: nomOut.pViol <= alpha + 1e-9 ? 'ok' : 'warn' },
        { label: 'Plant simulations', value: bat.jobs }, { label: 'Worker threads used', value: bat.pool.live && !bat.fellBack ? bat.pool.workers : 0 },
        ...(sp !== null ? [{ label: 'Measured parallel speed-up', value: sp, unit: '×', status: sp > 1 ? 'ok' : 'warn', help: 'Time the calling thread needs per simulation (measured on the first chunk) × number of simulations ÷ wall time of the pool. Wall-clock measurement: it varies from run to run and with the load of the machine.' }] : []),
      ],
      recommendations: [
        feasR ? `Design for ${fmt(dR.recovery ?? base.recovery, 3)} % recovery: it keeps every limit in at least ${fmt(100 * (1 - alpha), 3)} % of the sampled feed conditions at a cost of ${fmt(price, 3)} % at nominal conditions.` : 'Reduce the uncertainty (tighter pre-treatment, temperature control) or relax the limiting constraint.',
        'Check the limiting constraint in the scenario table: it shows which limit the uncertainty pushes against.', 'Increase the number of scenarios until the in-sample and out-of-sample violation probabilities agree.',
      ],
      plots: [
        { type: 'bar', title: 'Probability of violating a limit', ylabel: '%', categories: [`Design scenarios (${S})`, `Fresh scenarios (${nTest})`], series: [{ name: 'Nominal optimum', values: [100 * nomIn.pViol, 100 * nomOut.pViol] }, { name: 'Robust design', values: [100 * rob.st.pViol, 100 * robOut.pViol] }] },
        { type: 'line', title: 'Most critical constraint in every design scenario', xlabel: 'Scenario', ylabel: 'Largest normalised constraint (% over limit)', series: [{ name: 'Nominal optimum', x: range(S).map((i) => i + 1), y: nomIn.gmax.map((q) => 100 * q), mode: 'points' }, { name: 'Robust design', x: range(S).map((i) => i + 1), y: rob.st.gmax.map((q) => 100 * q), mode: 'points' }], hlines: [{ y: 0, label: 'feasible below this line' }] },
        { type: 'bar', title: 'Worst constraint utilisation over the design scenarios (100 % = at the limit)', ylabel: '% of limit', categories: CL.map((c) => c.label), series: [{ name: 'Nominal optimum', values: nomIn.gWorst.map((q) => 100 * (1 + q)) }, { name: 'Robust design', values: rob.st.gWorst.map((q) => 100 * (1 + q)) }] },
        { type: 'line', title: 'Convergence of the two searches', xlabel: 'Batch (generations, then pattern-search refinements)', ylabel: 'Best penalised objective (relative to base case)', series: [{ name: 'Nominal problem', x: range(nom.hist.length), y: nom.hist, mode: 'step' }, { name: worst ? 'Worst-case problem' : 'Sample-average problem', x: range(rob.hist.length), y: rob.hist, mode: 'step' }] },
        { type: 'line', title: `${O.label} over the fresh scenarios`, xlabel: 'Scenario (sorted)', ylabel: O.unit, series: [{ name: 'Nominal optimum', x: range(nTest).map((i) => i + 1), y: [...nomOut.f].map(show).sort((a, b) => a - b) }, { name: 'Robust design', x: range(nTest).map((i) => i + 1), y: [...robOut.f].map(show).sort((a, b) => a - b) }] },
      ],
      tables: [
        { title: 'Nominal optimum and robust design', columns: ['Quantity', 'Nominal optimum', 'Robust design'], rows: [...vars.map((d) => [`${d.label} (${d.unit})`, dN[d.key], dR[d.key]]), [`${O.label} at nominal conditions (${O.unit})`, show(nom.st.mean), show(robNom.mean)], [`Expected ${lcFirst(O.label)} over the design scenarios (${O.unit})`, show(nomIn.mean), show(rob.st.mean)], [`Worst ${lcFirst(O.label)} over the design scenarios (${O.unit})`, show(O.show ? Math.min(...nomIn.f) : nomIn.worst), show(O.show ? Math.min(...rob.st.f) : rob.st.worst)],
          ['Violation probability, design scenarios (%)', 100 * nomIn.pViol, 100 * rob.st.pViol], ['Violation probability, fresh scenarios (%)', 100 * nomOut.pViol, 100 * robOut.pViol], ['Feed pressure at nominal conditions (bar)', eN.m.Pf, eR.m.Pf], ['Specific energy at nominal conditions (kWh/m³)', eN.m.sec, eR.m.sec], ['Elements installed', eN.m.nEl, eR.m.nEl]] },
        { title: 'Constraints at nominal conditions and in the worst design scenario', columns: ['Constraint', 'Limit', 'Nominal optimum, nominal', 'Robust design, nominal', 'Nominal optimum, worst scenario (% over limit)', 'Robust design, worst scenario (% over limit)'], rows: CL.map((c, j) => [`${c.label} (${c.unit})`, c.lim(v, eR.m), cRow(eN)[j], cRow(eR)[j], 100 * nomIn.gWorst[j], 100 * rob.st.gWorst[j]]),
          note: `Sample-average approximation: the ${worst ? 'worst objective over' : 'mean objective over'} ${S} Latin-hypercube scenarios of feed salinity (σ ${fmt(v.robSal)} %), temperature (σ ${fmt(v.robT)} °C) and permeability loss (0–${fmt(v.robDecl)} %) is minimised; ${worst ? 'every scenario must satisfy every limit' : `at most ${allowed(S)} of ${S} scenarios may violate a limit (joint chance constraint, ${fmt(100 * alpha, 3)} %)`}. The array is sized by the design and the feed pressure adapts to each scenario. For the saturation index the last two columns give 100 × (index − limit).` },
        { title: 'Design scenarios', columns: ['Scenario', 'Salinity factor', 'Temperature (°C)', 'Flow factor', 'Nominal optimum: worst constraint (% over limit)', 'Robust design: worst constraint (% over limit)', `Nominal optimum: ${O.label} (${O.unit})`, `Robust design: ${O.label} (${O.unit})`], rows: scIn.map((s, i) => scRow(s, i, nomIn, rob.st)) },
        { title: 'Batched evaluation', columns: ['Item', 'Value'], rows: [['Plant simulations', bat.jobs], ['Chunks per batch', bat.chunks], ['Worker threads', bat.pool.live && !bat.fellBack ? bat.pool.workers : 0], ['Execution', bat.pool.live && !bat.fellBack ? 'parallel worker threads' : 'calling thread, cooperative batches'], ['Thread agreement (largest difference)', bat.agree ?? 0]], note: 'Wall-clock times and the measured speed-up are reported in the warnings and KPIs only, because they change from run to run.' },
      ],
      outputs: { task: 'rob', best: { recovery: dR.recovery ?? base.recovery, targetFlux: dR.targetFlux ?? base.targetFlux, sec: eR.m.sec, cost: eR.cost.total, feedPressure: eR.m.Pf, productTDS: eR.m.tds, nElements: eR.m.nEl }, objective: show(robNom.mean), objectiveName: v.objective, feasible: feasR, nominal: { recovery: dN.recovery ?? base.recovery, targetFlux: dN.targetFlux ?? base.targetFlux, objective: show(nom.st.mean) }, priceOfRobustness: price, violationProbability: robOut.pViol, study: { a: show(robNom.mean), b: price } },
    };
  } finally { bat.pool.close(); }
}

// ======================================================================================================
// 10 · Suite declaration
// ======================================================================================================
const TASKS = {
  opt: { label: 'Constrained optimisation of the RO plant', run: taskOpt }, mo: { label: 'Multi-objective optimisation (Pareto front)', run: taskMO },
  sa: { label: 'Sensitivity analysis (tornado, Morris, Sobol)', run: taskSA }, uq: { label: 'Uncertainty quantification (Monte Carlo)', run: taskUQ },
  ml: { label: 'Surrogate modelling and machine learning', run: taskML }, pinn: { label: 'Physics-informed neural network (verified demo)', run: taskPINN },
  ts: { label: 'Forecasting, anomaly detection and state estimation', run: taskTS }, ctl: { label: 'Predictive control and reinforcement learning', run: taskCTL },
  wb: { label: 'Custom numerical modelling workbench', run: taskWB }, pe: { label: 'Parameter estimation and identifiability', run: taskPE },
  oc: { label: 'Optimal control of a batch RO cycle (Euler–Lagrange, Pontryagin)', run: taskOC }, lp: { label: 'Linear and mixed-integer programming (train scheduling)', run: taskLP },
  node: { label: 'Neural ordinary differential equation', run: taskNODE }, npde: { label: 'Neural partial differential equation (learned closure)', run: taskNPDE },
  rom: { label: 'Reduced-order model (proper orthogonal decomposition)', run: taskROM }, ens: { label: 'Ensemble models (bagging, out-of-bag error)', run: taskENS },
  mf: { label: 'Multi-fidelity surrogate (co-kriging)', run: taskMF }, rob: { label: 'Robust and stochastic design under uncertainty', run: taskROB },
};
const is = (...t) => (v) => t.includes(v.task);
const usesRO = (v) => ['opt', 'mo', 'sa', 'uq', 'mf', 'rob'].includes(v.task) || (['ml', 'ens'].includes(v.task) && v.mlSource === 'ro');
const usesDesign = (v) => ['opt', 'mo', 'mf', 'rob'].includes(v.task) || (['ml', 'ens'].includes(v.task) && v.mlSource === 'ro');
const wb = (mode) => (v) => v.task === 'wb' && v.wbMode === mode;
const lazy = (obj, make) => Object.defineProperty(obj, 'value', { enumerable: true, configurable: true, get() { const val = make(); Object.defineProperty(obj, 'value', { value: val, enumerable: true, writable: true }); return val; } });
const studyGet = (task, key, mode) => (r) => {
  const o = r.outputs || {};
  if (o.task !== task || (mode && o.mode !== mode)) throw new Error(`select the task “${TASKS[task].label}”${mode ? ` in ${mode.toUpperCase()} mode` : ''} on the Inputs tab before running this study`);
  return o.study[key];
};
const memOptions = Object.entries(MEMBRANES).map(([k, m]) => ({ value: k, label: m.name }));
const nums = (prefix, n, label, vals, showFn, help) => range(n).map((i) => ({ key: prefix + (i + 1), label: `${label} ${i + 1}`, unit: '', value: vals[i], min: -1e12, max: 1e12, help: i ? undefined : help, showIf: (v) => showFn(v, i) }));
const texts = (prefix, n, label, vals, showFn, help) => range(n).map((i) => ({ key: prefix + (i + 1), label: label(i + 1), type: 'text', value: vals[i], help: i ? undefined : help, showIf: (v) => showFn(v, i) }));

const suite = {
  id: 'opt', num: 11, title: 'Optimization, AI & Custom Numerical Modelling', short: 'Optimise & AI', icon: '🧠',
  tagline: 'Optimise, rank, quantify, learn, forecast, control and build your own models — eighteen verified studies in one workspace.',
  description: 'Wraps the element-by-element RO model of suite 1 in constrained single- and multi-objective optimisers, global sensitivity analysis and Monte-Carlo uncertainty propagation, and trains response-surface, Gaussian-process and neural-network surrogates on it or on imported data. Further tasks cover a physics-informed network checked against an analytical solution, forecasting with anomaly detection and a Kalman state estimator, predictive control with reinforcement learning, a custom equation workbench (ODE, algebraic, optimisation, PDE) driven by a safe expression evaluator, and nonlinear parameter estimation with identifiability and Bayesian inference. Eight more studies add optimal control by Pontryagin\'s principle, linear and mixed-integer programming, a neural ODE, a neural PDE closure, a POD reduced-order model, bagged ensembles, multi-fidelity co-kriging and robust design under uncertainty with optional parallel evaluation.',
  guide: [
    'Pick a study in “Task”. Only the inputs of that study are shown; every study runs with its defaults.',
    'For the plant studies, pull the feed water and the base design from the Case page and suite 1, then set bounds and limits on the Model setup tab.',
    'Run. Read the warnings first, then the KPIs; each task reports its own evidence (convergence, test error, verification against a known solution).',
    'Change the random seed and run again: results that matter should not depend on it.',
    'The optimum recovery is offered back to suite 1; fouling rate and time-to-cleaning feed suite 10.',
  ],
  implemented: ['objective-function', 'equality constraint', 'karush-kuhn-tucker', 'lagrange multiplier', 'nonlinear-programming', 'mixed-integer programming', 'dynamic-programming', 'model-predictive-control', 'least-square', 'maximum-likelihood', 'bayesian equation', 'gaussian-process', 'artificial-neural-network', 'state-space',
    'physics-informed neural-network', 'grey-box', 'mechanistic-machine-learning', 'surrogate-assisted optimization', 'bayesian-mechanistic', 'digital-twin state-estimation', 'reinforcement-learning', 'genetic-algorithm-mechanistic', 'euler-lagrange', 'hamiltonian equation', 'pontryagin maximum-principle', 'linear-programming equation', 'neural ordinary differential', 'neural partial differential', 'reduced-order/cfd', 'ensemble model', 'multi-fidelity', 'robust/stochastic optimization', 'energy constraint', 'scaling constraint', 'parallel computing',
    'initial decision variable', 'state vector', 'model parameter', 'neural-network parameter', 'prior distribution', 'covariance matri', 'controller state', 'physical conservation constraint', 'parameter bound', 'operating envelope', 'water-quality constraint', 'pressure constraint', 'recovery constraint', 'terminal-state constraint', 'pde boundary constraint', 'feasibility constraint',
    'numerical equation solving', 'ordinary- and partial-differential-equation solving', 'nonlinear-system solving', 'parameter estimation', 'model calibration', 'model validation', 'sensitivity analysis', 'uncertainty quantification', 'monte carlo simulation', 'deterministic optimisation', 'nonlinear optimisation', 'mixed-integer optimisation', 'global optimisation', 'multi-objective optimisation',
    'surrogate modelling', 'machine learning', 'deep learning', 'physics-informed machine learning', 'time-series forecasting', 'anomaly detection', 'predictive maintenance', 'reinforcement learning', 'model-predictive control', 'digital-twin modelling', 'automated data processing', 'scenario analysis', 'custom-model development'],
  equationsNote: 'Everything runs in the browser, so problem sizes are deliberately small: networks have one or two hidden layers and a few hundred weights, Gaussian processes a few hundred points, NSGA-II a few hundred plant simulations, the reinforcement-learning agent is tabular and the predictive controller linear. Surrogates trained on the RO model reproduce that model, not plant reality, and must not be extrapolated. The newer studies are genuine but small-scale forms of their methods: optimal control (Euler–Lagrange boundary-value problem, Hamiltonian state–costate system and Pontryagin\'s principle with a bounded control) is solved for one state — a batch RO cycle with a lumped polarisation law; linear programmes use a dense two-phase simplex and integer variables a depth-first branch and bound, suitable for a few hundred variables; the neural ODE has one state and is trained by back-propagation through RK4; the neural PDE is a learned scalar reaction closure R(u) inside the 1-D finite-volume transport solver, trained on synthetic experiments by the discrete adjoint (it is not a neural operator for an arbitrary PDE); the reduced-order model is a POD–Galerkin projection of that 1-D convection–diffusion solver, not of a Navier–Stokes field; ensembles are bagged response surfaces or small networks; the multi-fidelity model is two-level recursive co-kriging; robust design uses a sample-average approximation with a few dozen scenarios, so its violation probabilities carry sampling error (compare the out-of-sample figure); the scaling constraint uses the Pitzer speciation of suite 2 on the simulated concentrate with an estimated concentrate pH and no antiscalant kinetics. Parallel computing means worker threads that evaluate chunks of plant simulations: it needs the multi-file edition served over http(s) or Node; where threads cannot start the same chunks run sequentially and the result says so. Deep reinforcement learning is not included. User expressions are parsed by a built-in arithmetic evaluator; no user text is ever executed as code.',

  inputs: [
    { group: 'Study', help: 'Choose what to compute. Each task has its own inputs, shown below and on the Model setup and Mesh tabs.', fields: [
      { key: 'task', label: 'Task', type: 'select', value: 'opt', options: Object.entries(TASKS).map(([k, t]) => ({ value: k, label: t.label })), help: 'Eighteen independent studies. The presets (“Load example…”) jump straight to each one.' },
      { key: 'seed', label: 'Random seed', unit: '', value: 7, min: 0, max: 1e6, step: 1, help: 'All sampling, initialisation and stochastic searches are seeded: the same seed reproduces the same result.' },
    ] },
    { group: 'RO plant (base case)', help: 'The plant model of suite 1 evaluated by the optimisation, sensitivity, uncertainty and surrogate tasks. Other RO settings keep the defaults of suite 1 (single pass, pressure exchanger).', showIf: usesRO, fields: [
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.seawater.ions, help: 'Full ionic analysis; pull it from the Case page.' },
      { key: 'Qf', label: 'Feed flow', unit: 'm³/h', value: 1000, min: 0.1, max: 2e5, help: 'Raw feed flow, held constant in every study.' },
      { key: 'T', label: 'Feed temperature', unit: '°C', value: 25, min: 1, max: 45, typical: [10, 38] },
      { key: 'pH', label: 'Feed pH', unit: '', value: 8.1, min: 2, max: 12 },
      { key: 'membrane', label: 'Element class (base case)', type: 'select', value: 'swhr', options: memOptions, help: 'Generic 8-inch classes of suite 1; permeabilities follow the class.' },
      { key: 'recovery0', label: 'Base recovery', unit: '%', value: 45, min: 5, max: 95, typical: [35, 85], help: 'Recovery of the reference design that the optimum is compared with.' },
      { key: 'flux0', label: 'Base average flux', unit: 'L/m²·h', value: 14, min: 3, max: 45, typical: [11, 30], help: 'The array is sized automatically from this flux.' },
      { key: 'elements0', label: 'Base elements per vessel', unit: '', value: 7, min: 1, max: 8, step: 1 },
      { key: 'ff', label: 'Flow factor (fouling allowance)', unit: '–', value: 0.95, min: 0.3, max: 1.2, help: '1.0 = clean new membrane.' },
      { key: 'erd', label: 'Energy-recovery device', type: 'select', value: 'px', options: [{ value: 'px', label: 'Isobaric pressure exchanger' }, { value: 'turbine', label: 'Pelton turbine / turbocharger' }, { value: 'none', label: 'None (throttle valve)' }] },
      { key: 'erdEff', label: 'Energy-recovery efficiency', unit: '%', value: 96, min: 30, max: 99 },
      { key: 'etaPump', label: 'High-pressure pump efficiency', unit: '%', value: 86, min: 30, max: 93 },
    ] },
    { group: 'Objective and cost model', help: 'What “best” means. The cost of water is deliberately simple; suite 13 holds the full economics.', showIf: usesRO, fields: [
      { key: 'objective', label: 'Objective', type: 'select', value: 'cost', options: [{ value: 'cost', label: 'Minimise cost of water' }, { value: 'sec', label: 'Minimise specific energy' }, { value: 'maxrec', label: 'Maximise recovery' }, { value: 'brine', label: 'Minimise brine volume per m³ product' }], showIf: is('opt', 'rob') },
      { key: 'elecPrice', label: 'Electricity price', unit: '$/kWh', value: 0.08, min: 0, max: 2, typical: [0.03, 0.25] },
      { key: 'elemPrice', label: 'Membrane element price', unit: '$/element', value: 650, min: 0, max: 5000 },
      { key: 'memLife', label: 'Membrane life', unit: 'years', value: 5, min: 0.5, max: 15 },
      { key: 'vesselPrice', label: 'Pressure-vessel price (installed)', unit: '$/vessel', value: 4000, min: 0, max: 50000, help: 'Recovered over 20 years.' },
      { key: 'interest', label: 'Interest rate', unit: '%/y', value: 6, min: 0, max: 30 },
      { key: 'avail', label: 'Plant availability', unit: '%', value: 95, min: 30, max: 100 },
      { key: 'chemPrice', label: 'Intake, pre-treatment and chemicals', unit: '$/m³ feed', value: 0.06, min: 0, max: 2, help: 'Charged per m³ of feed, so it favours high recovery.' },
      { key: 'brinePrice', label: 'Brine disposal', unit: '$/m³ brine', value: 0.02, min: 0, max: 5 },
    ] },
    { group: 'Operating data', help: 'A time series of a normalised performance indicator (permeate flow, permeability or pressure). Replace the built-in synthetic record with plant data.', showIf: is('ts'), fields: [
      lazy({ key: 'tsData', label: 'Operating record', type: 'table', columns: [{ key: 't', label: 'Time', unit: 'd' }, { key: 'y', label: 'Normalised permeate flow', unit: '–' }], help: 'Equally spaced samples work best. The built-in record contains a fouling trend, a weekly pattern, noise and three injected faults.' }, tsSampleTable),
      { key: 'tsThreshold', label: 'Cleaning threshold', unit: '–', value: 0.8, min: 0, max: 2, help: 'Level of the normalised signal at which a cleaning is triggered (commonly 10–20 % below the clean value).' },
      { key: 'tsSeason', label: 'Season length', unit: 'samples', value: 7, min: 0, max: 400, step: 1, help: 'Number of samples in one repeating cycle (7 for daily data with a weekly pattern). 0 or 1 disables seasonality.' },
      { key: 'tsHorizon', label: 'Forecast horizon', unit: 'samples', value: 14, min: 1, max: 200, step: 1 },
    ] },
    { group: 'Data for surrogate models', showIf: is('ml', 'ens'), fields: [
      { key: 'mlSource', label: 'Training data', type: 'select', value: 'ro', options: [{ value: 'ro', label: 'Design of experiments on the RO model (Latin hypercube)' }, { value: 'table', label: 'Imported data table' }] },
      { key: 'mlTarget', label: 'Quantity to learn', type: 'select', value: 'sec', options: Object.entries(ML_TARGET).map(([k, t]) => ({ value: k, label: `${t.label} (${t.unit})` })), showIf: (v) => v.mlSource === 'ro', help: 'Inputs are recovery, average flux, temperature and feed salinity.' },
      { key: 'mlN', label: 'Number of plant simulations', unit: '', value: 70, min: 20, max: 400, step: 1, showIf: (v) => v.mlSource === 'ro' },
      lazy({ key: 'mlData', label: 'Data table', type: 'table', columns: [...range(6).map((i) => ({ key: 'x' + (i + 1), label: 'x' + (i + 1) })), { key: 'y', label: 'y' }], showIf: (v) => v.mlSource === 'table', help: 'Up to six inputs x1…x6 and one output y. Columns left empty are ignored. The built-in example is permeate flow against pressure (x1), temperature (x2) and feed salinity (x3).' }, mlSampleTable),
    ] },
    { group: 'Film-layer problem', help: 'Steady concentration polarisation at a membrane: Jw·c − D·dc/dy = Jw·cp. Exact solution: (c − cp)/(cb − cp) = exp(Jw·y/D).', showIf: is('pinn'), fields: [
      { key: 'pinnJw', label: 'Water flux', unit: 'L/m²·h', value: 25, min: 0.5, max: 200 },
      { key: 'pinnK', label: 'Mass-transfer coefficient', unit: 'µm/s', value: 25, min: 1, max: 500, help: 'k = D/δ; typical spiral-wound values are 15–60 µm/s.' },
      { key: 'pinnD', label: 'Solute diffusivity', unit: '10⁻⁹ m²/s', value: 1.5, min: 0.1, max: 10 },
      { key: 'pinnCb', label: 'Bulk concentration', unit: 'g/L', value: 35, min: 0.01, max: 300 },
      { key: 'pinnCp', label: 'Permeate concentration', unit: 'g/L', value: 0.3, min: 0, max: 100 },
    ] },
    { group: 'RO train and product tank (linear model)', help: 'Deviation variables around the operating point: flux responds to pressure with a first-order lag, the tank integrates production minus demand.', showIf: is('ctl'), fields: [
      { key: 'ctlKp', label: 'Flux gain', unit: 'L/m²·h per bar', value: 1.2, min: 0.05, max: 20, help: 'Close to the water permeability A of the membranes.' },
      { key: 'ctlTau', label: 'Flux time constant', unit: 'min', value: 3, min: 0.1, max: 120 },
      { key: 'ctlArea', label: 'Membrane area', unit: 'm²', value: 30000, min: 10, max: 5e6 },
      { key: 'ctlTank', label: 'Tank cross-section', unit: 'm²', value: 200, min: 1, max: 1e5 },
      { key: 'ctlStep', label: 'Level set-point step', unit: 'm', value: 0.5, min: -5, max: 5 },
      { key: 'ctlDist', label: 'Demand step disturbance', unit: 'm³/h', value: 50, min: -5000, max: 5000 },
      { key: 'ctlOsm', label: 'Osmotic-pressure step disturbance', unit: 'bar', value: 1.5, min: -20, max: 20, help: 'A salinity or temperature change seen as a loss of driving pressure.' },
      { key: 'ctlJmax', label: 'Flux deviation limit', unit: '± L/m²·h', value: 3, min: 0.1, max: 50, help: 'Hard constraint for the predictive controller.' },
      { key: 'ctlUmax', label: 'Pressure deviation limit', unit: '± bar', value: 6, min: 0.1, max: 60 },
      { key: 'ctlDu', label: 'Pressure rate limit', unit: 'bar/step', value: 1, min: 0.01, max: 20 },
    ] },
    { group: 'Pump scheduling against a tariff', help: 'Two identical trains fill a product tank; each train-hour raises the level by 7.5 % of the volume. Demand follows a fixed daily profile with morning and evening peaks.', showIf: is('ctl'), fields: [
      { key: 'rlPower', label: 'Energy per train-hour', unit: 'kWh', value: 900, min: 1, max: 1e5 },
      { key: 'rlOff', label: 'Off-peak tariff (23–07 h)', unit: '$/kWh', value: 0.06, min: 0, max: 2 },
      { key: 'rlShoulder', label: 'Shoulder tariff (07–17 h, 22 h)', unit: '$/kWh', value: 0.11, min: 0, max: 2 },
      { key: 'rlPeak', label: 'Peak tariff (17–22 h)', unit: '$/kWh', value: 0.18, min: 0, max: 2 },
      { key: 'rlStart', label: 'Tank level at midnight', unit: '%', value: 50, min: 0, max: 100 },
      { key: 'rlPenalty', label: 'Penalty for unmet demand', unit: '$ per % volume', value: 500, min: 0, max: 1e6 },
    ] },
    { group: 'Workbench', help: 'Type your own equations. Allowed: numbers, + − * / ^, parentheses, comparisons (< > <= >= == !=, giving 1 or 0), the constants pi and e, the names in the parameter table, and the functions sin cos tan exp ln log10 sqrt abs min max pow tanh erf step.', showIf: is('wb'), fields: [
      { key: 'wbMode', label: 'Problem type', type: 'select', value: 'ode', options: [{ value: 'ode', label: 'ODE system dy/dt = f(t, y)' }, { value: 'alg', label: 'Nonlinear algebraic system F(x) = 0' }, { value: 'opt', label: 'Constrained minimisation' }, { value: 'pde', label: '1-D transient convection–diffusion–reaction PDE' }] },
      { key: 'wbParams', label: 'Parameters', type: 'table', columns: [{ key: 'name', label: 'Name', type: 'text' }, { key: 'value', label: 'Value' }], help: 'Named constants usable in every expression of the workbench.',
        value: [{ name: 'Am', value: 40 }, { name: 'Lp', value: 0.0012 }, { name: 'Pf', value: 30 }, { name: 'bpi', value: 0.75 }, { name: 'mu', value: 2.5 }, { name: 'Bmax', value: 50 }, { name: 'Aw', value: 1.2 }, { name: 'Ph', value: 58 }, { name: 'cb', value: 35 }, { name: 'kmt', value: 120 }, { name: 'eta', value: 0.8 }, { name: 'cel', value: 0.1 }, { name: 'pi0', value: 26 }, { name: 'cpre', value: 0.25 }, { name: 'kd', value: 0.3 }] },
    ] },
    { group: 'ODE system', help: 'Up to four states y1…y4 as functions of t. Default: batch RO concentration — y1 tank volume (m³), y2 concentration (g/L) — and logistic biofilm growth y3 (µm).', showIf: wb('ode'), fields: [
      { key: 'odeN', label: 'Number of states', unit: '', value: 3, min: 1, max: 4, step: 1 },
      ...texts('odeF', 4, (i) => `dy${i}/dt =`, ['-Am*Lp*max(Pf - bpi*y2, 0)', 'y2*Am*Lp*max(Pf - bpi*y2, 0)/y1', 'mu*y3*(1 - y3/Bmax)', '0'], (v, i) => i < v.odeN, 'Right-hand side in t, y1…y4 and the parameters.'),
      ...nums('odeY', 4, 'Initial value y', [2, 5, 1, 0], (v, i) => i < v.odeN),
      { key: 'odeT', label: 'End time', unit: 'time', value: 1.5, min: 1e-9, max: 1e9 },
    ] },
    { group: 'Algebraic system', help: 'Up to four equations Fᵢ(x1…x4) = 0. Default: flux x1 (L/m²·h) and wall concentration x2 (g/L) of one membrane element with concentration polarisation.', showIf: wb('alg'), fields: [
      { key: 'algN', label: 'Number of equations', unit: '', value: 2, min: 1, max: 4, step: 1 },
      ...texts('algF', 4, (i) => `Equation ${i}: 0 =`, ['x1 - Aw*(Ph - bpi*x2)', 'x2 - cb*exp(x1/kmt)', 'x3', 'x4'], (v, i) => i < v.algN, 'Left-hand side of Fᵢ = 0.'),
      ...nums('algX', 4, 'Initial guess x', [20, 40, 0, 0], (v, i) => i < v.algN),
    ] },
    { group: 'Minimisation problem', help: 'Minimise f(x) subject to g(x) ≤ 0, h(x) = 0 and bounds. Default: recovery x1 and pressure x2 minimising energy plus pre-treatment cost of an ideal single-stage RO.', showIf: wb('opt'), fields: [
      { key: 'optF', label: 'Objective f(x) to minimise', type: 'text', value: 'cel*x2/(36*eta) + cpre/x1' },
      { key: 'optG1', label: 'Constraint g1(x) ≤ 0', type: 'text', value: 'pi0/(1 - x1) + 6 - x2', help: 'Leave empty if not needed.' },
      { key: 'optG2', label: 'Constraint g2(x) ≤ 0', type: 'text', value: 'x2 - 70' },
      { key: 'optG3', label: 'Constraint g3(x) ≤ 0', type: 'text', value: '' },
      { key: 'optH', label: 'Equality h(x) = 0', type: 'text', value: '' },
      { key: 'optVars', label: 'Variables x1, x2, … (one row each)', type: 'table', columns: [{ key: 'lo', label: 'Lower bound' }, { key: 'hi', label: 'Upper bound' }, { key: 'x0', label: 'Start' }], value: [{ lo: 0.2, hi: 0.7, x0: 0.4 }, { lo: 30, hi: 80, x0: 60 }], help: 'Row 1 is x1, row 2 is x2 … (up to six).' },
    ] },
    { group: 'Transport equation', help: '∂u/∂t + v·∂u/∂x = D·∂²u/∂x² + R(u, x, t). Default: build-up of the polarisation layer — salt carried to the membrane at the permeate velocity and diffusing back.', showIf: wb('pde'), fields: [
      { key: 'pdeL', label: 'Domain length', unit: 'm', value: 1e-4, min: 1e-9, max: 1e6 },
      { key: 'pdeT', label: 'End time', unit: 's', value: 30, min: 1e-9, max: 1e9 },
      { key: 'pdeV', label: 'Velocity v', unit: 'm/s', value: 1e-5, min: -1e3, max: 1e3 },
      { key: 'pdeD', label: 'Diffusivity D', unit: 'm²/s', value: 1.5e-9, min: 0, max: 1e3 },
      { key: 'pdeR', label: 'Reaction term R(u, x, t)', type: 'text', value: '0', help: 'For example -kd*u for first-order decay.' },
      { key: 'pdeIC', label: 'Initial condition u(x, 0)', type: 'text', value: '35' },
      { key: 'pdeBCL', label: 'Left boundary (x = 0)', type: 'select', value: 'dirichlet', options: [{ value: 'dirichlet', label: 'Fixed value u' }, { value: 'neumann', label: 'Fixed gradient ∂u/∂x' }, { value: 'noflux', label: 'Wall: zero total flux' }] },
      { key: 'pdeBCLv', label: 'Left boundary value (function of t)', type: 'text', value: '35', showIf: (v) => v.pdeBCL !== 'noflux' },
      { key: 'pdeBCR', label: 'Right boundary (x = L)', type: 'select', value: 'noflux', options: [{ value: 'dirichlet', label: 'Fixed value u' }, { value: 'neumann', label: 'Fixed gradient ∂u/∂x' }, { value: 'noflux', label: 'Wall: zero total flux' }] },
      { key: 'pdeBCRv', label: 'Right boundary value (function of t)', type: 'text', value: '0', showIf: (v) => v.pdeBCR !== 'noflux' },
    ] },
    { group: 'Model and data to fit', help: 'Generic nonlinear regression y = f(x; parameters). Default: normalised permeate flow declining toward a plateau, y = a + (1 − a)·exp(−k·x).', showIf: is('pe'), fields: [
      { key: 'peModel', label: 'Model y =', type: 'text', value: 'a + (1 - a)*exp(-k*x)', help: 'Expression in x and the parameter names below.' },
      { key: 'peParams', label: 'Parameters to estimate', type: 'table', columns: [{ key: 'name', label: 'Name', type: 'text' }, { key: 'value', label: 'Start value' }, { key: 'lo', label: 'Lower bound' }, { key: 'hi', label: 'Upper bound' }], value: [{ name: 'a', value: 0.6, lo: 0, hi: 1 }, { name: 'k', value: 0.05, lo: 0.0001, hi: 1 }], help: 'Bounds also act as the limits of the uniform prior in the Bayesian option.' },
      lazy({ key: 'peData', label: 'Measurements', type: 'table', columns: [{ key: 'x', label: 'x' }, { key: 'y', label: 'y' }] }, peSampleTable),
    ] },

    { group: 'Batch RO cycle', help: 'A closed batch: the tank volume V falls as permeate is withdrawn at the rate q(t) and the retained salt concentrates. Minimise the pumping energy ∫P·q/(36·η)dt for a fixed permeate volume in a fixed time, with P = q/(A·area) + π₀·(V₀/V)·exp(J/k) + Δp.', showIf: is('oc'), fields: [
      { key: 'ocV0', label: 'Initial batch volume', unit: 'm³', value: 10, min: 0.01, max: 1e5, help: 'Feed charged to the tank at the start of the cycle.' },
      { key: 'ocRec', label: 'Batch recovery', unit: '%', value: 60, min: 1, max: 95, help: 'Fixes the terminal volume V(T) = V₀·(1 − recovery): the terminal-state constraint.' },
      { key: 'ocT', label: 'Cycle time', unit: 'h', value: 3, min: 0.01, max: 1000, help: 'Fixed duration in which the permeate volume must be produced.' },
      { key: 'ocArea', label: 'Membrane area', unit: 'm²', value: 80, min: 0.1, max: 1e6, help: 'Active membrane area of the batch unit.' },
      { key: 'ocA', label: 'Water permeability A', unit: 'L/m²·h·bar', value: 3, min: 0.05, max: 50, help: 'Pure-water permeability at operating temperature.' },
      { key: 'ocPi0', label: 'Initial osmotic pressure', unit: 'bar', value: 8, min: 0, max: 80, help: 'Osmotic pressure of the feed; it rises as V₀/V (complete rejection).' },
      { key: 'ocK', label: 'Mass-transfer coefficient', unit: 'µm/s', value: 8, min: 1, max: 1000, help: 'Concentration polarisation exp(J/k) makes high flux expensive late in the cycle.' },
      { key: 'ocDp', label: 'Hydraulic losses', unit: 'bar', value: 1, min: 0, max: 20, help: 'Constant feed-side pressure loss added to the pump pressure.' },
      { key: 'ocFluxMax', label: 'Flux limit', unit: 'L/m²·h', value: 26, min: 1, max: 200, help: 'Upper bound on the control (fouling limit). When it is active the optimum follows from Pontryagin\'s principle rather than from the Euler–Lagrange equation alone.' },
      { key: 'ocEta', label: 'Pump efficiency', unit: '%', value: 80, min: 5, max: 100, help: 'Hydraulic-to-electric efficiency of the feed pump.' },
    ] },
    { group: 'Train scheduling problem', help: 'Choose the production of each RO train in every period of the day so that demand is met from a product tank at the lowest energy cost. Demand follows the built-in daily profile with morning and evening peaks.', showIf: is('lp'), fields: [
      { key: 'lpMode', label: 'Problem class', type: 'select', value: 'milp', options: [{ value: 'milp', label: 'Mixed-integer: trains on/off with minimum load (branch and bound)' }, { value: 'lp', label: 'Linear programme: any load fraction (simplex)' }], help: 'The linear programme is the relaxation of the on/off problem and always gives a lower bound.' },
      lazy({ key: 'lpTrains', label: 'RO trains', type: 'table', columns: [{ key: 'cap', label: 'Capacity', unit: 'm³/h' }, { key: 'sec', label: 'Specific energy', unit: 'kWh/m³' }, { key: 'minLoad', label: 'Minimum load', unit: '%' }, { key: 'aux', label: 'Auxiliary power when on', unit: 'kW' }], help: 'Up to five trains. Minimum load and auxiliary power only matter for the on/off problem.' }, lpTrainTable),
      { key: 'lpDemand', label: 'Mean demand', unit: 'm³/h', value: 650, min: 0, max: 1e6, help: 'Daily average; the hourly profile is scaled to it.' },
      { key: 'lpPeriods', label: 'Periods per day', type: 'select', value: 8, options: [4, 6, 8, 12, 24].map((q) => ({ value: q, label: `${q} (${24 / q} h each)` })), help: 'More periods give a finer schedule and a much larger branch-and-bound tree: with 12 or 24 periods the on/off search usually stops at its limit with a proven gap.' },
      { key: 'lpTank', label: 'Product-tank volume', unit: 'm³', value: 4000, min: 1, max: 1e7, help: 'Storage that decouples production from demand.' },
      { key: 'lpS0', label: 'Tank level at midnight', unit: '%', value: 50, min: 0, max: 100, help: 'The day must end at least as full (terminal condition).' },
      { key: 'lpOff', label: 'Off-peak tariff (23–07 h)', unit: '$/kWh', value: 0.06, min: 0, max: 2, help: 'Night tariff.' },
      { key: 'lpShoulder', label: 'Shoulder tariff (07–17 h, 22 h)', unit: '$/kWh', value: 0.11, min: 0, max: 2, help: 'Day tariff outside the peak.' },
      { key: 'lpPeak', label: 'Peak tariff (17–22 h)', unit: '$/kWh', value: 0.18, min: 0, max: 2, help: 'Evening peak tariff.' },
      { key: 'lpPenalty', label: 'Penalty for unmet demand', unit: '$/m³', value: 5, min: 0, max: 1e4, help: 'Keeps the problem feasible: unserved water is allowed at this price.' },
    ] },
    { group: 'Trajectories to learn', help: 'One state y(t) observed in several runs (for example normalised permeability between cleanings). The network learns dy/dt = f(y); the first row of each run is its initial condition.', showIf: is('node'), fields: [
      lazy({ key: 'nodeData', label: 'Runs', type: 'table', columns: [{ key: 'run', label: 'Run' }, { key: 't', label: 'Time' }, { key: 'y', label: 'y' }], help: 'The built-in record holds four fouling runs that follow a non-exponential decline law with measurement noise.' }, nodeSampleTable),
      { key: 'nodeHold', label: 'Run held out for validation', unit: '', value: 4, min: 1, max: 1000, step: 1, help: 'Never used for training. With a single run its last 30 % is held out instead.' },
    ] },
    { group: 'Transport problem with an unknown reaction', help: 'A reacting species (for example residual chlorine or a nutrient) is carried along a channel: ∂u/∂t + v·∂u/∂x = D·∂²u/∂x² + R(u). Experiments at two inlet concentrations are generated with a hidden saturating law R = −k_max·u/(K_m + u); a network learns R(u) from the concentration fields and is tested on a third inlet concentration.', showIf: is('npde'), fields: [
      { key: 'npL', label: 'Channel length', unit: 'm', value: 1, min: 1e-6, max: 1e5, help: 'Length of the 1-D domain.' }, { key: 'npV', label: 'Velocity', unit: 'm/s', value: 0.1, min: 0, max: 100, help: 'Mean axial velocity (inlet at x = 0).' }, { key: 'npD', label: 'Dispersion coefficient', unit: 'm²/s', value: 0.004, min: 1e-12, max: 100, help: 'Axial dispersion or diffusion coefficient.' },
      { key: 'npT', label: 'Duration of each experiment', unit: 's', value: 12, min: 1e-6, max: 1e7, help: 'Each experiment starts from a clean channel and runs this long.' },
      { key: 'npKmax', label: 'Hidden law: maximum rate', unit: 'conc./s', value: 0.5, min: 0, max: 1e6, help: 'Used only to generate the data; the model never sees it.' }, { key: 'npKm', label: 'Hidden law: half-saturation', unit: 'conc.', value: 1.5, min: 1e-6, max: 1e6, help: 'Concentration at which the hidden rate is half its maximum.' },
      { key: 'npC1', label: 'Inlet concentration, experiment 1', unit: 'conc.', value: 2, min: 1e-6, max: 1e6, help: 'First training experiment.' }, { key: 'npC2', label: 'Inlet concentration, experiment 2', unit: 'conc.', value: 6, min: 1e-6, max: 1e6, help: 'Second training experiment; together they set the concentration range the closure is identified on.' },
      { key: 'npCtest', label: 'Inlet concentration, unseen test', unit: 'conc.', value: 4, min: 1e-6, max: 1e6, help: 'Used only to test the learned closure.' }, { key: 'npNoise', label: 'Measurement noise', unit: '% of inlet', value: 1, min: 0, max: 20, help: 'Standard deviation of the noise added to the training fields.' },
    ] },
    { group: 'Polarisation layer', help: 'Transient build-up of the concentration boundary layer at a membrane: ∂c/∂t + J·∂c/∂x = D·∂²c/∂x² across a film of thickness δ, bulk concentration at one side and a salt-rejecting wall at the other. Snapshots at two fluxes train the reduced model; it is tested at a third flux.', showIf: is('rom'), fields: [
      { key: 'romJ1', label: 'Training flux 1', unit: 'L/m²·h', value: 15, min: 0, max: 300, help: 'First flux at which snapshots are taken.' }, { key: 'romJ2', label: 'Training flux 2', unit: 'L/m²·h', value: 45, min: 0, max: 300, help: 'Second flux at which snapshots are taken.' }, { key: 'romJt', label: 'Unseen test flux', unit: 'L/m²·h', value: 30, min: 0, max: 300, help: 'Flux at which the reduced model is compared with the full one.' },
      { key: 'romDelta', label: 'Film thickness δ', unit: 'µm', value: 60, min: 1, max: 5000, help: 'Thickness of the concentration boundary layer, δ = D/k.' }, { key: 'romD', label: 'Solute diffusivity', unit: '10⁻⁹ m²/s', value: 1.5, min: 0.01, max: 100, help: 'Diffusivity of the solute in water.' },
      { key: 'romCb', label: 'Bulk concentration', unit: 'g/L', value: 35, min: 0.001, max: 400, help: 'Concentration at the bulk edge of the film.' }, { key: 'romT', label: 'Simulated time', unit: 's', value: 10, min: 1e-3, max: 1e5, help: 'A few diffusion times δ²/D reach the steady layer.' },
    ] },
    { group: 'Fidelity levels', help: 'High fidelity: the element model of suite 1 with several segments per element. Low fidelity: the same model with one segment, or an algebraic zero-dimensional model. Inputs are recovery, average flux, temperature and feed salinity.', showIf: is('mf'), fields: [
      { key: 'mfTarget', label: 'Quantity to learn', type: 'select', value: 'sec', options: Object.entries(MF_TARGET).map(([k, t]) => ({ value: k, label: `${t.label} (${t.unit})` })), help: 'Output of the RO model that the surrogate predicts.' },
      { key: 'mfLow', label: 'Low-fidelity model', type: 'select', value: 'lumped', options: [{ value: 'lumped', label: 'Algebraic zero-dimensional model (negligible cost)' }, { value: 'coarse', label: 'Element model with one segment per element' }], help: 'A low-fidelity run is charged as 0 (algebraic) or 1/segments (coarse) of a high-fidelity run.' },
      { key: 'mfSegH', label: 'High fidelity: segments per element', unit: '', value: 2, min: 2, max: 6, step: 1, help: 'Resolution of the accurate model; each run costs about this many coarse runs.' },
      { key: 'mfNH', label: 'High-fidelity runs', unit: '', value: 8, min: 4, max: 60, step: 1, help: 'Expensive runs used for training.' }, { key: 'mfNL', label: 'Low-fidelity runs', unit: '', value: 40, min: 6, max: 300, step: 1, help: 'The high-fidelity points are a subset of the low-fidelity design (nested).' },
      { key: 'mfNT', label: 'High-fidelity test runs', unit: '', value: 12, min: 5, max: 100, step: 1, help: 'Used only to measure the error.' },
    ] },
    { group: 'Uncertainty and risk', help: 'Uncertain feed conditions and membrane state. The design (recovery and flux, hence the array) is fixed before the uncertainty is revealed; the feed pressure adapts to each scenario.', showIf: is('rob'), fields: [
      { key: 'robMode', label: 'Formulation', type: 'select', value: 'chance', options: [{ value: 'chance', label: 'Stochastic: expected objective, chance constraint' }, { value: 'worst', label: 'Robust: worst-case objective, every scenario feasible' }], help: 'Chance constraint: a share of scenarios may violate a limit. Worst case: none may.' },
      { key: 'robS', label: 'Design scenarios', unit: '', value: 6, min: 2, max: 64, step: 1, help: 'Latin-hypercube sample used inside the optimisation (sample-average approximation).' },
      { key: 'robAlpha', label: 'Allowed violation probability', unit: '%', value: 20, min: 0, max: 50, showIf: (v) => v.robMode !== 'worst', help: 'Joint chance constraint: share of scenarios in which any limit may be exceeded.' },
      { key: 'robSal', label: 'Feed salinity, standard deviation', unit: '%', value: 3, min: 0, max: 20, help: 'Normal, truncated at ± 2.5 standard deviations.' }, { key: 'robT', label: 'Feed temperature, standard deviation', unit: '°C', value: 2, min: 0, max: 12, help: 'Normal, truncated at ± 2.5 standard deviations.' },
      { key: 'robDecl', label: 'Permeability loss, worst case', unit: '%', value: 10, min: 0, max: 60, help: 'Uniform between 0 and this loss.' },
      { key: 'robTest', label: 'Fresh scenarios for validation', unit: '', value: 16, min: 4, max: 200, step: 1, help: 'Independent sample: the out-of-sample violation probability is the honest one.' },
    ] },

    // ---- Model setup tab ---------------------------------------------------------------------------------
    { group: 'Decision variables and bounds', tab: 'setup', help: 'The search space. A variable whose bounds coincide is held fixed.', showIf: usesDesign, fields: [
      { key: 'recLo', label: 'Recovery, lower bound', unit: '%', value: 35, min: 5, max: 95 }, { key: 'recHi', label: 'Recovery, upper bound', unit: '%', value: 55, min: 5, max: 95 },
      { key: 'fluxLo', label: 'Average flux, lower bound', unit: 'L/m²·h', value: 10, min: 3, max: 45 }, { key: 'fluxHi', label: 'Average flux, upper bound', unit: 'L/m²·h', value: 20, min: 3, max: 45 },
      { key: 'elLo', label: 'Elements per vessel, minimum', unit: '', value: 6, min: 1, max: 8, step: 1, showIf: is('opt', 'mo'), help: 'Integer variable.' }, { key: 'elHi', label: 'Elements per vessel, maximum', unit: '', value: 8, min: 1, max: 8, step: 1, showIf: is('opt', 'mo') },
      { key: 'memSet', label: 'Element classes to choose from', type: 'select', value: 'sw', showIf: is('opt', 'mo'), options: [{ value: 'sw', label: 'Seawater: high rejection / low energy / ultra-low energy' }, { value: 'bw', label: 'Brackish: high rejection / low energy' }, { value: 'fixed', label: 'Keep the base-case class' }], help: 'Categorical variable of the mixed-integer problem.' },
      { key: 'optPp', label: 'Optimise permeate back-pressure', type: 'bool', value: false, showIf: is('opt', 'mo'), help: 'Throttling the permeate balances flux along the vessel at an energy cost.' },
      { key: 'ppHi', label: 'Back-pressure, upper bound', unit: 'bar', value: 5, min: 0.6, max: 20, showIf: (v) => is('opt', 'mo')(v) && v.optPp },
      { key: 'optBoost', label: 'Optimise inter-stage boost', type: 'bool', value: false, showIf: is('opt', 'mo'), help: 'Only matters when the array has two or more stages (recovery above about 58 %).' },
      { key: 'boostHi', label: 'Inter-stage boost, upper bound', unit: 'bar', value: 10, min: 0.5, max: 40, showIf: (v) => is('opt', 'mo')(v) && v.optBoost },
    ] },
    { group: 'Operating constraints', tab: 'setup', help: 'Inequality constraints of the optimisation and the limits whose violation probability is reported by the uncertainty task.', showIf: usesRO, fields: [
      { key: 'limTDS', label: 'Product TDS limit', unit: 'mg/L', value: 500, min: 1, max: 5000 }, { key: 'limBoron', label: 'Product boron limit', unit: 'mg/L', value: 2.4, min: 0.05, max: 10 },
      { key: 'limFlux', label: 'Maximum lead-element flux', unit: 'L/m²·h', value: 34, min: 5, max: 60 }, { key: 'limP', label: 'Maximum feed pressure', unit: 'bar', value: 70, min: 2, max: 120, help: 'The element rating also applies.' },
      { key: 'limCP', label: 'Maximum polarisation factor β', unit: '–', value: 1.2, min: 1.02, max: 2 }, { key: 'limConc', label: 'Minimum concentrate flow per vessel', unit: 'm³/h', value: 3, min: 0.1, max: 10 },
      { key: 'limSEC', label: 'Specific-energy cap', unit: 'kWh/m³', value: 0, min: 0, max: 20, showIf: is('opt', 'mo', 'rob'), help: 'Energy constraint: specific energy of the membrane system must stay below this value. 0 switches the constraint off.' },
      { key: 'conScale', label: 'Scaling constraint on the concentrate', type: 'select', value: 'off', showIf: is('opt', 'mo', 'rob'), options: [{ value: 'off', label: 'Off' }, { value: 'gypsum', label: 'Gypsum (CaSO₄·2H₂O)' }, { value: 'sulfate', label: 'Sulphate scales: gypsum, barite, celestite' }, { value: 'barite', label: 'Barite (BaSO₄)' }, { value: 'celestite', label: 'Celestite (SrSO₄)' }, { value: 'calcite', label: 'Calcite (CaCO₃)' }, { value: 'silica', label: 'Amorphous silica' }], help: 'Saturation index of the concentrate from the electrolyte model of suite 2 (Pitzer), evaluated for every candidate design.' },
      { key: 'limSI', label: 'Maximum saturation index', unit: 'log₁₀(IAP/K)', value: 0, min: -3, max: 3, showIf: (v) => is('opt', 'mo', 'rob')(v) && v.conScale !== 'off', help: '0 = saturation; a positive value is the supersaturation an antiscalant can hold.' },
    ] },
    { group: 'Optimisation algorithm', tab: 'setup', showIf: is('opt'), fields: [
      { key: 'algo', label: 'Algorithm', type: 'select', value: 'de', options: [{ value: 'de', label: 'Differential evolution (global, integers relaxed)' }, { value: 'nm', label: 'Nelder–Mead simplex + integer neighbourhood search' }, { value: 'sqp', label: 'SQP with finite-difference gradients + integer neighbourhood search' }], help: 'The global method is the most robust; SQP is the fastest near a smooth optimum and reports its own KKT residual.' },
      { key: 'nStarts', label: 'Number of starts', unit: '', value: 2, min: 1, max: 8, step: 1, help: 'Independent runs with seeds seed, seed + 1, …; the first local start is the base case.' },
      { key: 'optRho', label: 'Penalty weight ρ', unit: '–', value: 20, min: 0.1, max: 1e4, help: 'Exact penalty on normalised constraint violations; must exceed the largest multiplier.' },
      { key: 'dePop', label: 'Population size', unit: '', value: 10, min: 5, max: 80, step: 1, showIf: (v) => v.algo === 'de' }, { key: 'deGens', label: 'Generations', unit: '', value: 8, min: 1, max: 200, step: 1, showIf: (v) => v.algo === 'de' },
      { key: 'optPolish', label: 'Polish the best point with Nelder–Mead', type: 'bool', value: true, showIf: (v) => v.algo === 'de' },
      { key: 'nmIter', label: 'Nelder–Mead iterations', unit: '', value: 25, min: 5, max: 500, step: 1, showIf: (v) => v.algo !== 'sqp' }, { key: 'sqpIter', label: 'SQP iterations', unit: '', value: 12, min: 2, max: 100, step: 1, showIf: (v) => v.algo === 'sqp' },
    ] },
    { group: 'NSGA-II settings', tab: 'setup', showIf: is('mo'), fields: [
      { key: 'moThird', label: 'Third objective', type: 'select', value: 'tds', options: [{ value: 'tds', label: 'Minimise product TDS' }, { value: 'cost', label: 'Minimise cost of water' }, { value: 'none', label: 'None (energy versus recovery only)' }], help: 'Energy (minimise) and recovery (maximise) are always objectives.' },
      { key: 'moPop', label: 'Population size', unit: '', value: 24, min: 8, max: 200, step: 2 }, { key: 'moGens', label: 'Generations', unit: '', value: 12, min: 1, max: 300, step: 1 },
    ] },
    { group: 'Sensitivity settings', tab: 'setup', showIf: is('sa'), fields: [
      { key: 'saOutput', label: 'Output to rank', type: 'select', value: 'sec', options: SA_OUT.map((o) => ({ value: o.key, label: `${o.label} (${o.unit})` })), help: 'Tables list all four outputs.' },
      { key: 'saRange', label: 'Factor range', unit: '± %', value: 10, min: 0.5, max: 40, help: 'Applied to salinity, recovery, flux, permeabilities, efficiencies and the mass-transfer coefficient.' }, { key: 'saDT', label: 'Temperature range', unit: '± °C', value: 5, min: 0.5, max: 15 },
      { key: 'saR', label: 'Morris trajectories', unit: '', value: 4, min: 2, max: 50, step: 1, help: 'Each costs (factors + 1) simulations.' }, { key: 'saTop', label: 'Factors carried to Sobol analysis', unit: '', value: 4, min: 2, max: 9, step: 1 },
      { key: 'sobolN', label: 'Sobol base sample N', unit: '', value: 32, min: 8, max: 2000, step: 1, help: 'Costs N × (factors + 2) simulations; 100+ gives stable indices.' },
    ] },
    { group: 'Input distributions', tab: 'setup', help: 'Uncertain operating conditions and membrane state over the life of the plant.', showIf: is('uq'), fields: [
      { key: 'uqSampling', label: 'Sampling', type: 'select', value: 'lhs', options: [{ value: 'lhs', label: 'Latin hypercube (stratified)' }, { value: 'mc', label: 'Plain Monte Carlo' }] },
      { key: 'uqSalSd', label: 'Feed salinity, standard deviation', unit: '%', value: 3, min: 0, max: 20 }, { key: 'uqTsd', label: 'Feed temperature, standard deviation', unit: '°C', value: 3, min: 0, max: 12 },
      { key: 'uqDecl', label: 'Permeability decline, worst case', unit: '%', value: 15, min: 0, max: 60, help: 'Triangular distribution from 0 to this loss, most likely one third of it.' },
      { key: 'uqSpDrift', label: 'Salt-passage drift', unit: '%', value: 20, min: 0, max: 200, help: 'Lognormal multiplier on salt permeability with median 1 + drift/2.' }, { key: 'uqEtaSd', label: 'Pump efficiency, standard deviation', unit: '% points', value: 1.5, min: 0, max: 10 },
    ] },
    { group: 'Training settings', tab: 'setup', showIf: is('ml'), fields: [
      { key: 'mlTest', label: 'Test share', unit: '%', value: 20, min: 5, max: 40, help: 'Held out completely; used once for the reported errors.' }, { key: 'mlVal', label: 'Validation share', unit: '%', value: 20, min: 5, max: 40, help: 'Used for early stopping and to select the model.' },
      { key: 'mlK', label: 'Cross-validation folds', unit: '', value: 5, min: 2, max: 10, step: 1 },
      { key: 'nnLayers', label: 'Hidden layers', type: 'select', value: 1, options: [{ value: 1, label: 'One' }, { value: 2, label: 'Two' }] }, { key: 'nnH1', label: 'Neurons, first hidden layer', unit: '', value: 8, min: 2, max: 32, step: 1 },
      { key: 'nnH2', label: 'Neurons, second hidden layer', unit: '', value: 6, min: 2, max: 32, step: 1, showIf: (v) => +v.nnLayers >= 2 }, { key: 'nnEpochs', label: 'Maximum epochs', unit: '', value: 500, min: 20, max: 5000, step: 10 },
      { key: 'nnLr', label: 'Learning rate (Adam)', unit: '', value: 0.02, min: 1e-4, max: 0.5 }, { key: 'nnPatience', label: 'Early-stopping patience', unit: 'epochs', value: 80, min: 5, max: 2000, step: 1 },
      { key: 'mlBO', label: 'Bayesian optimisation with the Gaussian process', type: 'bool', value: true, help: 'Expected-improvement loop on the plant model, or a suggested next experiment for table data.' },
      { key: 'boIter', label: 'Bayesian-optimisation evaluations', unit: '', value: 6, min: 1, max: 40, step: 1, showIf: (v) => v.mlBO && v.mlSource === 'ro' }, { key: 'boGoal', label: 'Goal', type: 'select', value: 'min', options: [{ value: 'min', label: 'Minimise' }, { value: 'max', label: 'Maximise' }], showIf: (v) => v.mlBO },
    ] },
    { group: 'Network and training', tab: 'setup', showIf: is('pinn'), fields: [
      { key: 'pinnNeurons', label: 'Hidden neurons (tanh)', unit: '', value: 8, min: 2, max: 40, step: 1 }, { key: 'pinnIters', label: 'Adam iterations', unit: '', value: 3000, min: 200, max: 40000, step: 100 },
      { key: 'pinnLr', label: 'Initial learning rate', unit: '', value: 0.02, min: 1e-4, max: 0.2 }, { key: 'pinnColl', label: 'Collocation points', unit: '', value: 24, min: 4, max: 400, step: 1 },
      { key: 'pinnWbc', label: 'Boundary-loss weight', unit: '', value: 10, min: 0.1, max: 1000 }, { key: 'pinnFD', label: 'Finite-difference intervals (reference)', unit: '', value: 24, min: 4, max: 2000, step: 1 },
    ] },
    { group: 'Forecast models and state estimator', tab: 'setup', showIf: is('ts'), fields: [
      { key: 'tsTrain', label: 'Training share (chronological)', unit: '%', value: 75, min: 40, max: 95 }, { key: 'tsArP', label: 'Autoregressive order p', unit: '', value: 3, min: 1, max: 12, step: 1 },
      { key: 'tsArD', label: 'Autoregression on', type: 'select', value: 'd1', options: [{ value: 'd1', label: 'First differences (trending data)' }, { value: 'd0', label: 'Levels (stationary data)' }] },
      { key: 'ekfRate', label: 'Fouling-rate constant, initial', unit: '1/d', value: 0.004, min: 0, max: 1, help: 'r in dK/dt = −r·(K − K∞); the filter updates it. Calibrate it on the Calibration tab.' }, { key: 'ekfKinf', label: 'Plateau permeability K∞', unit: '–', value: 0.7, min: 0, max: 1.5 },
      { key: 'ekfMeasSd', label: 'Measurement standard deviation', unit: '–', value: 0.012, min: 1e-5, max: 1 }, { key: 'tsGate', label: 'Innovation gate', unit: 'σ', value: 3, min: 1.5, max: 8, help: 'Larger innovations are flagged and not assimilated.' },
      { key: 'tsEwma', label: 'EWMA weight λ', unit: '–', value: 0.25, min: 0.05, max: 1 },
    ] },
    { group: 'Controllers', tab: 'setup', showIf: is('ctl'), fields: [
      { key: 'ctlN', label: 'MPC prediction horizon', unit: 'steps', value: 15, min: 3, max: 40, step: 1 }, { key: 'ctlQy', label: 'MPC level-error weight', unit: '', value: 50, min: 0.01, max: 1e5 }, { key: 'ctlR', label: 'MPC move weight', unit: '', value: 0.5, min: 0, max: 1e4 },
      { key: 'ctlKc', label: 'PI gain', unit: 'bar/m', value: 25, min: 0, max: 1000 }, { key: 'ctlTi', label: 'PI integral time', unit: 'min', value: 50, min: 0.5, max: 5000 }, { key: 'ctlTsim', label: 'Simulated time', unit: 'min', value: 360, min: 20, max: 5000 }, { key: 'ctlDt', label: 'Sampling time', unit: 'min', value: 2, min: 0.05, max: 60, help: 'Both controllers act once per sample.' },
      { key: 'rlEpisodes', label: 'Q-learning episodes', unit: '', value: 12000, min: 200, max: 200000, step: 100 }, { key: 'rlAlpha', label: 'Q-learning rate α', unit: '–', value: 0.5, min: 0.01, max: 1 },
    ] },
    { group: 'Solver options', tab: 'setup', showIf: is('wb'), fields: [
      { key: 'odeMethod', label: 'ODE integrator', type: 'select', value: 'rk45', options: [{ value: 'rk45', label: 'Adaptive Dormand–Prince RK45' }, { value: 'rk4', label: 'Fixed-step RK4' }], showIf: wb('ode') },
      { key: 'wbAlgo', label: 'Minimiser', type: 'select', value: 'nm', options: [{ value: 'nm', label: 'Nelder–Mead (local)' }, { value: 'de', label: 'Differential evolution, then Nelder–Mead (global)' }], showIf: wb('opt'), help: 'Constraints are handled by an augmented Lagrangian in both cases.' },
      { key: 'pdeTheta', label: 'Time integration', type: 'select', value: 'cn', options: [{ value: 'cn', label: 'Crank–Nicolson (second order)' }, { value: 'be', label: 'Backward Euler (first order, damped)' }], showIf: wb('pde') },
      { key: 'pdeScheme', label: 'Convection scheme', type: 'select', value: 'central', options: [{ value: 'central', label: 'Central (second order)' }, { value: 'upwind', label: 'Upwind (first order, bounded)' }], showIf: wb('pde') },
    ] },
    { group: 'Neural ODE training', tab: 'setup', showIf: is('node'), fields: [
      { key: 'nodeH', label: 'Hidden neurons (tanh)', unit: '', value: 8, min: 2, max: 32, step: 1, help: 'Width of the single hidden layer of the right-hand side.' }, { key: 'nodeEpochs', label: 'Adam epochs', unit: '', value: 800, min: 20, max: 20000, step: 10, help: 'Full-batch gradient steps.' },
      { key: 'nodeLr', label: 'Initial learning rate', unit: '', value: 0.05, min: 1e-4, max: 0.3, help: 'Adam step size; it decays to 5 % of this value.' }, { key: 'nodeSub', label: 'RK4 steps per sampling interval', unit: '', value: 1, min: 1, max: 10, step: 1, help: 'The gradient is propagated backwards through every stage of every step.' },
    ] },
    { group: 'Closure network and training', tab: 'setup', showIf: is('npde'), fields: [
      { key: 'npH', label: 'Hidden neurons (tanh)', unit: '', value: 6, min: 2, max: 24, step: 1, help: 'Width of the single hidden layer of the closure.' }, { key: 'npEpochs', label: 'Adam epochs', unit: '', value: 200, min: 10, max: 5000, step: 10, help: 'Full-batch gradient steps; each needs one forward and one adjoint solve per experiment.' }, { key: 'npLr', label: 'Initial learning rate', unit: '', value: 0.05, min: 1e-4, max: 0.5, help: 'Adam step size; it decays to 5 % of this value.' },
    ] },
    { group: 'Reduced basis', tab: 'setup', showIf: is('rom'), fields: [
      { key: 'romR', label: 'Modes retained', unit: '', value: 0, min: 0, max: 12, step: 1, help: '0 = as many as needed to capture 99.9999 % of the snapshot energy.' }, { key: 'romSnap', label: 'Snapshots per training run', unit: '', value: 25, min: 4, max: 60, step: 1, help: 'Saved profiles of each full-order run.' },
    ] },
    { group: 'Ensemble settings', tab: 'setup', showIf: is('ens'), fields: [
      { key: 'ensBase', label: 'Base learner', type: 'select', value: 'nn', options: [{ value: 'nn', label: 'Neural network (high variance: bagging helps)' }, { value: 'poly', label: 'Polynomial response surface' }, { value: 'mix', label: 'Alternating network and response surface' }], help: 'Model type trained on every bootstrap resample.' },
      { key: 'ensB', label: 'Bootstrap members', unit: '', value: 12, min: 3, max: 60, step: 1, help: 'Number of bootstrap resamples and models.' }, { key: 'ensH', label: 'Hidden neurons per network', unit: '', value: 6, min: 2, max: 24, step: 1, help: 'Width of the hidden layer of each member.' }, { key: 'ensEpochs', label: 'Epochs per network', unit: '', value: 250, min: 20, max: 3000, step: 10, help: 'Training length of each member (no early stopping).' },
      { key: 'ensTest', label: 'Test share', unit: '%', value: 25, min: 5, max: 40, help: 'Held out completely; the out-of-bag error is computed without it.' },
    ] },
    { group: 'Search and parallel evaluation', tab: 'setup', showIf: is('rob'), fields: [
      { key: 'robPop', label: 'Population size', unit: '', value: 6, min: 4, max: 40, step: 1, help: 'Every generation is one batch of population × scenarios plant simulations.' }, { key: 'robGens', label: 'Generations', unit: '', value: 2, min: 1, max: 60, step: 1, help: 'Differential-evolution generations before the local refinement.' }, { key: 'robPolish', label: 'Pattern-search refinements', unit: '', value: 3, min: 0, max: 20, step: 1, help: 'Local refinement of the best design; each one is a batch of four neighbouring designs.' },
      { key: 'parWorkers', label: 'Worker threads', unit: '', value: 0, min: 0, max: 16, step: 1, help: 'Parallel computing: batches are split into this many chunks and simulated on worker threads; the speed-up is measured. 0 evaluates on the calling thread. If threads cannot be started (for example in the single-file edition) the same chunks run on the calling thread and the result says so.' },
    ] },
    { group: 'Branch and bound', tab: 'setup', showIf: is('lp'), fields: [
      { key: 'lpNodes', label: 'Node limit', unit: '', value: 1500, min: 10, max: 20000, step: 10, help: 'The search stops here, or earlier when its simplex work budget is used up, and reports the remaining optimality gap.' },
    ] },
    { group: 'Bayesian inference', tab: 'setup', showIf: is('pe'), fields: [
      { key: 'peBayes', label: 'Sample the posterior (Metropolis–Hastings)', type: 'bool', value: true }, { key: 'peMcmcN', label: 'Total MCMC iterations (two chains)', unit: '', value: 6000, min: 400, max: 200000, step: 100, showIf: (v) => v.peBayes },
    ] },

    // ---- Mesh tab ------------------------------------------------------------------------------------------
    { group: 'Resolution', tab: 'mesh', help: 'Numerical resolution of the selected task. The studies below re-run the task at three levels.', fields: [
      { key: 'roSeg', label: 'RO model: segments per element', unit: '', value: 1, min: 1, max: 6, step: 1, showIf: usesRO, help: 'One segment keeps the many plant simulations fast; suite 1 quantifies the discretisation error (below 2 % on pressure).' },
      { key: 'uqN', label: 'Monte-Carlo samples', unit: '', value: 160, min: 10, max: 5000, step: 10, showIf: is('uq') },
      { key: 'odeDt', label: 'ODE time step (fixed-step RK4)', unit: 'time', value: 0.05, min: 1e-9, max: 1e9, showIf: wb('ode') },
      { key: 'pdeNx', label: 'PDE cells', unit: '', value: 40, min: 4, max: 2000, step: 1, showIf: wb('pde') }, { key: 'pdeNt', label: 'PDE time steps', unit: '', value: 120, min: 2, max: 20000, step: 1, showIf: wb('pde') },
      { key: 'ocN', label: 'Optimal control: time steps', unit: '', value: 100, min: 10, max: 2000, step: 1, showIf: is('oc'), help: 'RK4 steps of the state–costate integration.' },
      { key: 'npNx', label: 'Neural PDE: cells', unit: '', value: 20, min: 6, max: 200, step: 1, showIf: is('npde'), help: 'Finite volumes of the model the closure is trained in.' }, { key: 'npNt', label: 'Neural PDE: time steps', unit: '', value: 40, min: 6, max: 2000, step: 1, showIf: is('npde'), help: 'Time steps of each experiment.' },
      { key: 'romNx', label: 'Reduced-order model: cells of the full model', unit: '', value: 60, min: 8, max: 400, step: 1, showIf: is('rom'), help: 'Finite volumes of the full-order model.' }, { key: 'romNt', label: 'Reduced-order model: time steps', unit: '', value: 100, min: 10, max: 5000, step: 1, showIf: is('rom'), help: 'Crank–Nicolson steps of the full and the reduced model.' },
    ] },
  ],

  presets: [
    { name: '1 · Optimise cost of water (global search, seawater RO)', values: { task: 'opt' } },
    { name: '1 · Minimise specific energy with SQP and KKT multipliers', values: { task: 'opt', objective: 'sec', algo: 'sqp' } },
    { name: '2 · Pareto front: energy vs recovery vs product quality (NSGA-II)', values: { task: 'mo' } },
    { name: '3 · Sensitivity ranking: tornado, Morris and Sobol', values: { task: 'sa' } },
    { name: '4 · Uncertainty: P5/P50/P95 and probability of limit violations', values: { task: 'uq' } },
    { name: '5 · Surrogates of the RO model + Bayesian optimisation', values: { task: 'ml' } },
    { name: '5 · Surrogates from an imported data table (two-layer network)', values: { task: 'ml', mlSource: 'table', nnLayers: 2, boGoal: 'max' } },
    { name: '6 · Physics-informed network for the polarisation film', values: { task: 'pinn' } },
    { name: '7 · Forecast permeate flow, flag anomalies, time to cleaning', values: { task: 'ts' } },
    { name: '8 · MPC versus PI and Q-learned pump schedule', values: { task: 'ctl' } },
    { name: '9 · Workbench ODE: batch RO concentration + biofilm growth', values: { task: 'wb', wbMode: 'ode' } },
    { name: '9 · Workbench algebraic: flux with concentration polarisation', values: { task: 'wb', wbMode: 'alg' } },
    { name: '9 · Workbench optimisation: recovery–pressure trade-off', values: { task: 'wb', wbMode: 'opt' } },
    { name: '9 · Workbench PDE: transient polarisation layer', values: { task: 'wb', wbMode: 'pde' } },
    { name: '9 · Workbench PDE: decaying tracer pulse in a channel (upwind)', values: { task: 'wb', wbMode: 'pde', pdeL: 1, pdeT: 1.5, pdeV: 0.4, pdeD: 0.002, pdeR: '-kd*u', pdeIC: 'exp(-((x - 0.2)/0.05)^2)', pdeBCL: 'dirichlet', pdeBCLv: '0', pdeBCR: 'neumann', pdeBCRv: '0', pdeScheme: 'upwind', pdeNx: 100, pdeNt: 150 } },
    { name: '10 · Fit a fouling-decline model with identifiability and MCMC', values: { task: 'pe' } },
    { name: '1 · Optimise cost with an energy cap and a gypsum-scaling limit', values: { task: 'opt', limSEC: 1.8, conScale: 'gypsum', limSI: -0.33 } },
    { name: '11 · Optimal control: energy-optimal batch RO cycle (Pontryagin)', values: { task: 'oc' } },
    { name: '12 · Train scheduling against a tariff (simplex + branch and bound)', values: { task: 'lp' } },
    { name: '12 · Train scheduling as a pure linear programme (12 periods)', values: { task: 'lp', lpMode: 'lp', lpPeriods: 12 } },
    { name: '13 · Neural ODE for fouling decline with a held-out run', values: { task: 'node' } },
    { name: '14 · Neural PDE: learn a reaction closure from concentration fields', values: { task: 'npde' } },
    { name: '15 · Reduced-order model of the polarisation layer (POD–Galerkin)', values: { task: 'rom' } },
    { name: '16 · Bagged ensemble of surrogates with out-of-bag error', values: { task: 'ens' } },
    { name: '17 · Multi-fidelity co-kriging of the RO model', values: { task: 'mf' } },
    { name: '18 · Chance-constrained RO design with energy and scaling limits', values: { task: 'rob', limSEC: 2.08, conScale: 'gypsum', limSI: -0.25 } },
    { name: '18 · Worst-case design evaluated on 4 worker threads', values: { task: 'rob', robMode: 'worst', parWorkers: 4 } },
  ],

  pull: ({ feed, outputs } = {}) => [
    { key: 'ions', value: feed?.ions, from: 'Case feed water' }, { key: 'Qf', value: feed?.Q, from: 'Case feed water' }, { key: 'T', value: feed?.T, from: 'Case feed water' }, { key: 'pH', value: feed?.pH, from: 'Case feed water' },
    outputs?.ro?.recovery ? { key: 'recovery0', value: clamp(100 * outputs.ro.recovery, 5, 95), from: 'RO design: recovery of the base case' } : null,
    outputs?.ro?.fluxLMH ? { key: 'flux0', value: clamp(outputs.ro.fluxLMH, 3, 45), from: 'RO design: average flux of the base case' } : null,
    outputs?.econ?.electricityPrice ? { key: 'elecPrice', value: outputs.econ.electricityPrice, from: 'Economics: electricity price' } : null,
    outputs?.econ?.membranePrice ? { key: 'elemPrice', value: outputs.econ.membranePrice, from: 'Economics: element price' } : null,
    outputs?.pump?.pumpEfficiency ? { key: 'etaPump', value: clamp(outputs.pump.pumpEfficiency <= 1 ? 100 * outputs.pump.pumpEfficiency : outputs.pump.pumpEfficiency, 30, 93), from: 'Pump suite: high-pressure pump efficiency' } : null,
    outputs?.fouling?.normPermeability ? { key: 'ff', value: clamp(outputs.fouling.normPermeability, 0.4, 1.1), from: 'Fouling monitor: normalised permeability' } : null,
    outputs?.fouling?.foulingRate ? { key: 'ekfRate', value: clamp(outputs.fouling.foulingRate / 30, 1e-5, 0.5), from: 'Fouling monitor: permeability decline rate' } : null,
    outputs?.cfd?.kMass ? { key: 'pinnK', value: clamp(outputs.cfd.kMass * 1e6, 1, 500), from: 'CFD: mass-transfer coefficient' } : null,
  ],
  site: (site) => [
    { key: 'elecPrice', value: site?.data?.electricityPrice, from: 'Electricity price at site' }, { key: 'interest', value: site?.data?.lendingRate, from: 'Lending rate at site' }, { key: 'T', value: site?.data?.sst, from: 'Sea-surface temperature at site' },
  ],

  async run(v, ctx0) {
    const ctx = { ...ctx0, progress: (f, m) => ctx0?.progress?.(f, m), tick: () => (ctx0?.tick ? ctx0.tick() : Promise.resolve()) };
    const T = TASKS[v.task] || TASKS.opt;
    ctx.progress(0.01, T.label);
    warmReset();
    await ctx.tick();
    const res = await T.run(v, ctx);
    ctx.progress(1, 'Done');
    return res;
  },

  mesh: [
    { name: 'Monte-Carlo sample count (uncertainty task)', keys: ['uqN'], min: 20, note: 'Each study belongs to one task: select “Uncertainty quantification” for the sample-count study, or the workbench in ODE or PDE mode for the step-size and grid studies. Sampling error falls as 1/√N rather than with a fixed order, so read the three levels as a convergence check; the grid-convergence index applies strictly to the ODE and PDE studies.',
      metrics: [{ label: 'Mean specific energy', unit: 'kWh/m³', get: studyGet('uq', 'a') }, { label: 'Feed pressure P95', unit: 'bar', get: studyGet('uq', 'b') }, { label: 'Mean product TDS', unit: 'mg/L', get: studyGet('uq', 'c') }] },
    { name: 'ODE time step (workbench, fixed-step RK4)', keys: ['odeDt'], refine: 'divide', metrics: [{ label: 'y1 at end time', unit: '', get: studyGet('wb', 'a', 'ode') }, { label: 'y2 (or y1) at end time', unit: '', get: studyGet('wb', 'b', 'ode') }] },
    { name: 'PDE grid and time step (workbench)', keys: ['pdeNx', 'pdeNt'], min: 8, metrics: [{ label: 'Domain mean at end time', unit: '', get: studyGet('wb', 'a', 'pde') }, { label: 'Value at right boundary', unit: '', get: studyGet('wb', 'b', 'pde') }] },
    { name: 'Optimal control: time steps of the state–costate integration', keys: ['ocN'], min: 10, metrics: [{ label: 'Optimal cycle energy', unit: 'kWh', get: studyGet('oc', 'a') }, { label: 'Initial costate', unit: 'kWh/m³', get: studyGet('oc', 'b') }] },
    { name: 'Reduced-order model: grid and time step of the full model', keys: ['romNx', 'romNt'], min: 8, metrics: [{ label: 'Polarisation factor, reduced model', unit: '–', get: studyGet('rom', 'a') }, { label: 'Polarisation factor, full model', unit: '–', get: studyGet('rom', 'b') }] },
  ],

  calibration: {
    note: 'Calibrate the fouling state model used by the Kalman estimator of the forecasting task, K(t) = K∞ + (1 − K∞)·exp(−r·t), against normalised permeability measured since the last cleaning. Each row is one day-stamped measurement; use a different run between cleanings for validation.',
    params: [{ key: 'ekfRate', label: 'Fouling-rate constant r (1/d)', lo: 1e-4, hi: 0.1 }, { key: 'ekfKinf', label: 'Plateau permeability K∞', lo: 0.3, hi: 0.98 }],
    columns: [{ key: 't', label: 'Time since cleaning', unit: 'd' }, { key: 'Kn', label: 'Normalised permeability', unit: '–' }],
    targets: [{ key: 'Kn', label: 'Normalised permeability', unit: '–' }],
    model: (v) => ({ Kn: v.ekfKinf + (1 - v.ekfKinf) * Math.exp(-Math.max(0, v.ekfRate) * (isNum(v.t) ? v.t : 0)) }),
    get sample() { return (this._s ||= foulingSynth(31, [0, 10, 20, 30, 45, 60, 75, 90, 110, 130, 150, 180])); },
    get validationSample() { return (this._v ||= foulingSynth(53, [5, 25, 50, 80, 120, 160, 200])); },
  },

  async verify() {
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Number.isFinite(got) && Math.abs(got - expected) <= tol, note });
    const rosen = (x) => (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, himmel = (x) => (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2;
    // optimisers on benchmark functions
    const nm = nelderMead(rosen, [-1.2, 1], { maxIter: 4000, tol: 1e-14 });
    add('Nelder–Mead finds the Rosenbrock minimum', 0, Math.hypot(nm.x[0] - 1, nm.x[1] - 1), 1e-4, 'Distance from (1, 1)');
    const de = diffEvolution(himmel, [-5, -5], [5, 5], { pop: 30, gens: 120, seed: 3 });
    add('Differential evolution finds a Himmelblau minimum', 0, de.f, 1e-8, 'f = 0 at each of the four minima');
    const sq = sqpLite((x) => (x[0] - 0.3) ** 2 + 2 * (x[1] + 0.4) ** 2 + (x[2] - 1) ** 2, null, [2, 2, -2], { lo: [-5, -5, -5], hi: [5, 5, 5] });
    add('SQP finds the minimum of a shifted sphere', 0, Math.hypot(sq.x[0] - 0.3, sq.x[1] + 0.4, sq.x[2] - 1), 1e-4, `Converged in ${sq.iterations} iterations`);
    const sr = sqpLite(rosen, null, [-1.2, 1], { lo: [-3, -3], hi: [3, 3], maxIter: 200, tol: 1e-8, h: 1e-9 });
    add('SQP (BFGS) finds the Rosenbrock minimum', 0, Math.hypot(sr.x[0] - 1, sr.x[1] - 1), 1e-3, `${sr.iterations} iterations, ${sr.evals} function evaluations`);
    // constrained problem with known KKT point: min x² + y² s.t. x + y ≥ 1 → (½, ½), λ = 1
    const cf = (x) => x[0] ** 2 + x[1] ** 2, cg = (x) => [1 - x[0] - x[1]], sc = sqpLite(cf, cg, [2, 0], { lo: [-3, -3], hi: [3, 3] });
    add('SQP reaches the known constrained optimum', 0, Math.hypot(sc.x[0] - 0.5, sc.x[1] - 0.5), 1e-4, 'min x² + y² subject to x + y ≥ 1');
    add('SQP Lagrange multiplier equals the analytical value', 1, sc.lambda[0], 1e-3, 'λ = 1 from ∇f + λ∇g = 0');
    add('SQP KKT residual vanishes at the optimum', 0, sc.kkt, 1e-4, 'Stationarity and feasibility');
    const kk = kktCheck(cf, cg, [0.5, 0.5], { lo: [-3, -3], hi: [3, 3] });
    add('KKT check recovers the multiplier by non-negative least squares', 1, kk.lambda[0], 1e-5, `Stationarity residual ${kk.residual.toExponential(1)}`);
    const pen = nelderMead((x) => cf(x) + 20 * Math.max(0, cg(x)[0]), [2, 0], { maxIter: 3000, tol: 1e-14 });
    add('Exact penalty with Nelder–Mead reaches the same optimum', 0, Math.hypot(pen.x[0] - 0.5, pen.x[1] - 0.5), 2e-3, 'ρ = 20 > λ = 1');
    const qp = qpSolve(qpPrepare([[2, 0], [0, 2]], [[1, 1]]), [-4, -4], [2]);
    add('Dual QP solver: known solution and multiplier', 0, Math.hypot(qp.x[0] - 1, qp.x[1] - 1) + Math.abs(qp.lambda[0] - 2), 1e-8, 'min (x−2)² + (y−2)² s.t. x + y ≤ 2 → (1, 1), λ = 2');
    // gradients
    const xg = [-0.7, 1.3], ga = [-2 * (1 - xg[0]) - 400 * xg[0] * (xg[1] - xg[0] ** 2), 200 * (xg[1] - xg[0] ** 2)], gn = fdGradient(rosen, xg);
    add('Finite-difference gradient matches the analytical gradient', 0, Math.max(Math.abs(gn[0] - ga[0]), Math.abs(gn[1] - ga[1])) / Math.hypot(...ga), 1e-7, 'Rosenbrock at (−0.7, 1.3), central differences');
    const net = nnInit([3, 5, 4, 1], 5), g = rng(9), Xn = range(6).map(() => range(3).map(() => g.normal())), Yn = Xn.map((x) => [Math.sin(x[0]) + x[1] * x[2]]), bp = nnLossGrad(net, Xn, Yn);
    let gErr = 0;
    for (const [l, i, j] of [[0, 1, 2], [0, 4, 0], [1, 2, 3], [1, 0, 0], [2, 0, 1]]) { const w0 = net[l].W[i][j], h = 1e-6; net[l].W[i][j] = w0 + h; const a = nnLossGrad(net, Xn, Yn).loss; net[l].W[i][j] = w0 - h; const b = nnLossGrad(net, Xn, Yn).loss; net[l].W[i][j] = w0; gErr = Math.max(gErr, Math.abs((a - b) / (2 * h) - bp.grads[l].W[i][j])); }
    { const b0 = net[1].b[2], h = 1e-6; net[1].b[2] = b0 + h; const a = nnLossGrad(net, Xn, Yn).loss; net[1].b[2] = b0 - h; const b = nnLossGrad(net, Xn, Yn).loss; net[1].b[2] = b0; gErr = Math.max(gErr, Math.abs((a - b) / (2 * h) - bp.grads[1].b[2])); }
    add('Back-propagation gradient matches the numerical gradient', 0, gErr, 1e-7, 'Two hidden layers, five weights and one bias checked');
    const fe = nnFast([3, 5, 4, 1]), fg = new Float64Array(fe.nW); fe.grad(fe.pack(net), Xn, Yn, range(6), 0, 6, fg);
    add('Training engine (flat arrays) gives the same gradient as back-propagation', 0, Math.max(...fe.pack(bp.grads).map((q, i) => Math.abs(q - fg[i]))), 1e-12, `All ${fe.nW} weights and biases`);
    const pp = { w: [0.8, -1.1, 0.4], b: [0.1, 0.5, -0.3], v: [0.7, -0.2, 0.9], c: 0.6 }, pxi = linspace(0, 1, 7), pg = pinnLossGrad(pp, pxi, 0.6, 5);
    let pErr = 0;
    for (const [k, j] of [['w', 0], ['w', 2], ['b', 1], ['v', 2], ['c', -1]]) { const h = 1e-6, get = () => (j < 0 ? pp.c : pp[k][j]), set = (q) => { if (j < 0) pp.c = q; else pp[k][j] = q; }, q0 = get(); set(q0 + h); const a = pinnLossGrad(pp, pxi, 0.6, 5).loss; set(q0 - h); const b = pinnLossGrad(pp, pxi, 0.6, 5).loss; set(q0); pErr = Math.max(pErr, Math.abs((a - b) / (2 * h) - (j < 0 ? pg.g.c : pg.g[k][j]))); }
    add('Physics-informed loss gradient matches the numerical gradient', 0, pErr, 1e-7, 'Residual and boundary terms, derivative of the network included');
    const pn = pinnTrain(0.5, { neurons: 8, iters: 2500, seed: 3 });
    add('Physics-informed network reproduces exp(Pe·ξ) at the wall', Math.exp(0.5), pn.predict(1), 0.01, 'Trained on the equation residual only');
    // multi-objective
    const zdt = (x) => { const gz = 1 + (9 * sum(x.slice(1))) / (x.length - 1); return { f: [x[0], gz * (1 - Math.sqrt(x[0] / gz))], cv: 0 }; };
    const ns = nsga2(zdt, new Array(6).fill(0), new Array(6).fill(1), { pop: 60, gens: 120, seed: 4 });
    add('NSGA-II converges to the ZDT1 Pareto front', 0, mean(ns.front.map((p) => Math.abs(p.f[1] - (1 - Math.sqrt(p.f[0]))))), 0.02, `Mean distance of ${ns.front.length} front points from f2 = 1 − √f1`);
    add('NSGA-II spreads along the ZDT1 front', 1, Math.max(...ns.front.map((p) => p.f[0])) - Math.min(...ns.front.map((p) => p.f[0])), 0.1, 'Range of f1 covered');
    // Gaussian process
    const gx = linspace(0, 6, 13).map((x) => [x]), gy = gx.map((x) => Math.sin(x[0])), gp = gpFit(gx, gy), gt = linspace(0.25, 5.75, 12);
    add('Gaussian process interpolates its training points', 0, Math.max(...gx.map((x, i) => Math.abs(gp.predict(x).mean - gy[i]))), 1e-3, 'Noise-free samples of sin(x)');
    add('Gaussian process recovers sin(x) between the points', 0, Math.max(...gt.map((x) => Math.abs(gp.predict([x]).mean - Math.sin(x)))), 0.02, 'Maximum error at 12 unseen points');
    // sensitivity and sampling
    const ish = (u) => { const x = u.map((q) => Math.PI * (2 * q - 1)); return Math.sin(x[0]) + 7 * Math.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * Math.sin(x[0]); }, so = sobolIndices(ish, 3, 2048, 5);
    add('Sobol first-order index S1 of the Ishigami function', 0.3139, so.S[0], 0.04, 'Analytical value 0.3139');
    add('Sobol first-order index S2 of the Ishigami function', 0.4424, so.S[1], 0.04, 'Analytical value 0.4424');
    add('Sobol total index ST3 of the Ishigami function', 0.2437, so.ST[2], 0.04, 'Pure interaction effect of x3');
    const lin = morrisEstimate(morrisPlan(3, 6, 2), morrisPlan(3, 6, 2).pts.map((u) => 2 * u[0] - 5 * u[1]));
    add('Morris μ* of a linear function equals its coefficients', 0, Math.abs(lin.muStar[0] - 2) + Math.abs(lin.muStar[1] - 5) + lin.muStar[2] + lin.sigma[1], 1e-9, 'y = 2x1 − 5x2: μ* = (2, 5, 0), σ = 0');
    const ln = lhs(4000, 1, 8).map((u) => Math.exp(0.25 * normInv(u[0])));
    add('Monte-Carlo mean of a lognormal variable', Math.exp(0.03125), mean(ln), 1e-3, 'exp(σ²/2), σ = 0.25, Latin-hypercube sample of 4000');
    add('Monte-Carlo variance of a lognormal variable', (Math.exp(0.0625) - 1) * Math.exp(0.0625), variance(ln), 2e-3, '(exp(σ²) − 1)·exp(σ²)');
    // filters and forecasting
    const gk = rng(21), yk = range(200).map(() => 0.9 + gk.normal(0, 0.02)), kf = ekfFouling(range(200), yk, { Kinf: 0.5, r0: 0, K0: 0.5, sdMeas: 0.02, qK: 0, qr: 0, P0: [1, 0], gate: 1e9 });
    add('Kalman filter converges on a constant signal', 0.9, kf.x[0], 0.005, 'Estimate after 200 noisy samples (σ = 0.02)');
    add('Kalman posterior variance equals the analytical value', 1 / (1 + 200 / 0.0004), kf.P[0][0], 1e-12, '1/(1/P₀ + n/R)');
    const ga1 = rng(13), ya = [0];
    for (let i = 1; i < 4000; i++) ya.push(0.7 * ya[i - 1] + ga1.normal(0, 1));
    add('AR(1) coefficient recovered by least squares', 0.7, arFit(ya, 1, 0).phi[0], 0.04, '4000 simulated samples with φ = 0.7');
    const yh = range(60).map((t) => 10 + 0.5 * t + [1, -2, 0.5, 0.5][t % 4]), hw = hwFit(yh, 4), hf = hw.model.forecast(8);
    add('Holt–Winters forecasts a trend + season series exactly', 0, Math.max(...hf.map((q, k) => Math.abs(q - (10 + 0.5 * (60 + k) + [1, -2, 0.5, 0.5][(60 + k) % 4])))), 0.05, 'Noise-free linear trend with period-4 season, 8 steps ahead');
    // PDE solver: manufactured solution
    const m = [20, 40, 80].map((k) => mmsCase(k, k)), gc = gci([m[2].dx, m[1].dx, m[0].dx], [m[2].mean, m[1].mean, m[0].mean]);
    add('PDE solver: observed order from the manufactured-solution error', 2, Math.log2(m[1].err / m[2].err), 0.25, `RMS errors ${m.map((q) => q.err.toExponential(2)).join(' → ')}`);
    add('PDE solver: observed order from Richardson extrapolation (GCI)', 2, gc.p, 0.4, `Domain mean; fine-grid GCI ${(100 * gc.gciFine).toExponential(1)} %`);
    const st = solveCDR({ L: 1, nx: 80, tEnd: 40, nt: 400, v: 1, D: 0.5, ic: () => 1, left: { type: 'dirichlet', val: () => 1 }, right: { type: 'noflux', val: () => 0 }, theta: 1 });
    add('PDE solver: steady polarisation profile u = exp(v·x/D)', Math.exp(2), st.u[79] * Math.exp((2 * st.dx) / 2), 0.02, 'Wall value of the film solution, extrapolated half a cell');
    // expression evaluator
    const ex = [['1 + 2*3', 7], ['-2^2', -4], ['2^3^2', 512], ['(1 + 2)*3 - 4/8', 8.5], ['max(1, 5, 3) + min(4, 2)', 7], ['sqrt(16) + abs(-3) + ln(e)', 8], ['3 > 2', 1], ['step(-1) + step(2)', 1], ['1e-3*x + pow(y, 2)', 9.002], ['cos(pi) + tanh(0) + log10(1000)', 2], ['2*-3', -6], ['erf(0.5)', 0.5204999]];
    add('Expression evaluator: arithmetic, precedence and functions', 0, Math.max(...ex.map(([s, val]) => Math.abs(evalExpr(s, { x: 2, y: 3 }) - val))), 1e-6, `${ex.length} expressions`);
    const bad = ['constructor.constructor("x")()', 'window', 'a.b', 'x[0]', 'alert(1)', '1; 2', '"s"', 'x = 3', '__proto__', 'this', 'globalThis.process', 'toString()', '(() => 1)()', 'Function("return 1")()', 'require("fs")', 'process.exit()', '`1`', 'x ? 1 : 2', '1 +', 'sin', 'x{}'];
    add('Expression evaluator rejects every unsafe or malformed input', bad.length, bad.filter((s) => { try { parseExpr(s, ['x']).evaluate({ x: 1 }); return false; } catch { return true; } }).length, 0, 'Property access, calls of unknown names, strings, assignment and globals are all refused');
    // control and reinforcement learning
    const dt = 0.2, di = mpcBuild({ A: [[1, dt], [0, 1]], B: [[dt * dt / 2], [dt]], C: [[1, 0]] }, { N: 15, Q: [1], R: [0.05], umin: [-1], umax: [1] });
    let xd = [0, 0], ud = 0, uMax = 0;
    for (let k = 0; k < 120; k++) { ud = di.solve(xd, [ud], [1]).u[0]; uMax = Math.max(uMax, Math.abs(ud)); xd = [xd[0] + dt * xd[1] + (dt * dt / 2) * ud, xd[1] + dt * ud]; }
    add('MPC drives a double integrator to the set-point', 0, Math.abs(xd[0] - 1) + Math.abs(xd[1]), 1e-2, 'Position 1, velocity 0 after 24 time units');
    add('MPC respects the input bound', 1, uMax, 1e-6, '|u| ≤ 1 active during the manoeuvre');
    const env = tankEnv({ rlPeak: 0.18, rlShoulder: 0.11, rlOff: 0.06, rlStart: 50, rlPower: 900, rlPenalty: 500 }), opt = rollout(env, dpSolve(env).policy), ql = rollout(env, qLearn(env, { episodes: 12000, alpha: 0.5, seed: 2, s0: env.s0 }).policy);
    add('Q-learning approaches the dynamic-programming optimum', 0, (ql.total - opt.total) / opt.total, 0.05, `Daily cost ${ql.total.toFixed(0)} against the exact ${opt.total.toFixed(0)}`);
    const L = cholesky([[4, 2, 0.6], [2, 5, 1], [0.6, 1, 3]]), xs = cholSolve(L, [1, 2, 3]);
    add('Cholesky solve satisfies the linear system', 0, Math.abs(4 * xs[0] + 2 * xs[1] + 0.6 * xs[2] - 1) + Math.abs(2 * xs[0] + 5 * xs[1] + xs[2] - 2) + Math.abs(0.6 * xs[0] + xs[1] + 3 * xs[2] - 3), 1e-12, 'Residual of A·x = b');
    add('Inverse normal CDF is consistent with the error function', 0.975, normCdf(normInv(0.975)), 1e-6, 'Φ(Φ⁻¹(0.975))');
    { // ---- checks of the newer studies (own scope)
    // linear and mixed-integer programming
    const l1 = lpSolve([-3, -5], [{ a: [1, 0], op: '<=', b: 4 }, { a: [0, 2], op: '<=', b: 12 }, { a: [3, 2], op: '<=', b: 18 }]);
    add('Simplex solves the textbook linear programme', 36, -l1.f, 1e-9, `max 3x + 5y s.t. x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18 → (${fmt(l1.x[0])}, ${fmt(l1.x[1])}), z = 36`);
    add('Simplex dual values equal the known shadow prices', 0, Math.abs(l1.dual[0]) + Math.abs(l1.dual[1] + 1.5) + Math.abs(l1.dual[2] + 1), 1e-9, '∂z/∂b = (0, 3/2, 1) for the maximisation');
    const l2 = lpSolve([1, 2, 3], [{ a: [1, 1, 1], op: '=', b: 10 }, { a: [1, 0, 0], op: '<=', b: 4 }, { a: [0, 1, 0], op: '<=', b: 5 }, { a: [1, 1, 0], op: '>=', b: 2 }]);
    add('Two-phase simplex with equality and ≥ rows', 17, l2.f, 1e-9, 'min x + 2y + 3z s.t. x + y + z = 10, x ≤ 4, y ≤ 5 → (4, 5, 1)');
    add('Simplex detects infeasible and unbounded programmes', 2, (lpSolve([1], [{ a: [1], op: '<=', b: 1 }, { a: [1], op: '>=', b: 2 }]).status === 'infeasible' ? 1 : 0) + (lpSolve([-1], [{ a: [-1], op: '<=', b: 0 }]).status === 'unbounded' ? 1 : 0), 0, 'x ≤ 1 with x ≥ 2; min −x without an upper limit');
    const kv = [24, 13, 23, 15, 16, 11, 9, 21], kw = [12, 7, 11, 8, 9, 6, 5, 10], unitRow = (j) => range(8).map((q) => (q === j ? 1 : 0));
    let brute = 0; for (let b = 0; b < 256; b++) { let w = 0, val = 0; for (let j = 0; j < 8; j++) if (b >> j & 1) { w += kw[j]; val += kv[j]; } if (w <= 26 && val > brute) brute = val; }
    const mk = milpSolve(kv.map((q) => -q), [{ a: kw, op: '<=', b: 26 }, ...range(8).map((j) => ({ a: unitRow(j), op: '<=', b: 1 }))], range(8));
    add('Branch and bound finds the knapsack optimum', brute, -mk.f, 1e-9, `${mk.nodes} nodes against enumeration of all 256 subsets; LP bound ${fmt(-mk.relaxation.f, 5)}`);
    const asg = [[9, 2, 7], [6, 4, 3], [5, 8, 1]], am = milpSolve(asg.flat(), [...range(3).map((i) => ({ a: range(9).map((q) => (Math.floor(q / 3) === i ? 1 : 0)), op: '=', b: 1 })), ...range(3).map((j) => ({ a: range(9).map((q) => (q % 3 === j ? 1 : 0)), op: '=', b: 1 }))], range(9));
    add('Mixed-integer assignment problem: known minimum cost', 9, am.f, 1e-9, '3 × 3 cost matrix, optimum 2 + 6 + 1');
    // optimal control
    const di2 = pmpShoot({ x0: [0, 0], xT: [1, 0], T: 1, f: (x, u) => [x[1], u[0]], L: (x, u) => 0.5 * u[0] ** 2, dHdx: (x, l) => [0, l[0]], uStar: (x, l) => [-l[1]] }, { N: 20 });
    add('Pontryagin shooting: minimum-energy double integrator, cost', 6, di2.J, 1e-9, 'x(0) = v(0) = 0 → x(1) = 1, v(1) = 0; J = ½∫u² = 6');
    add('Pontryagin shooting: optimal control u*(t) = 6 − 12t', 0, Math.max(...di2.t.map((t, k) => Math.abs(di2.u[k][0] - (6 - 12 * t)))), 1e-8, 'Largest deviation along the path');
    const lq = pmpShoot({ x0: [1], xT: [null], T: 2, f: (x, u) => [u[0]], L: (x, u) => 0.5 * (x[0] ** 2 + u[0] ** 2), dHdx: (x) => [x[0]], uStar: (x, l) => [-l[0]] }, { N: 200 });
    add('Linear-quadratic regulator: optimal cost ½·tanh(T)', 0.5 * Math.tanh(2), lq.J, 1e-8, 'min ½∫(x² + u²)dt, ẋ = u, x(0) = 1, free end; costate λ(0) = Riccati gain tanh(T)');
    add('Linear-quadratic regulator: initial costate equals the Riccati solution', Math.tanh(2), lq.lam0[0], 1e-8, 'P(0) = tanh(T) from −dP/dt = 1 − P², P(T) = 0');
    add('Hamiltonian is constant along the optimal path', 0, (Math.max(...lq.H) - Math.min(...lq.H)) / Math.abs(lq.H[0]), 1e-8, 'Autonomous problem: H = ½x² − ½λ²');
    const elv = eulerLagrangeBVP((x, vd) => vd, (x) => x, 0, 1, 1, 80);
    add('Euler–Lagrange boundary-value solver: x″ = x', 0, Math.max(...elv.t.map((t, k) => Math.abs(elv.x[k] - Math.sinh(t) / Math.sinh(1)))), 5e-5, 'L = ½(ẋ² + x²), x(0) = 0, x(1) = 1 → sinh t / sinh 1');
    const bv = { ocV0: 10, ocRec: 50, ocT: 2, ocArea: 100, ocA: 3, ocPi0: 5, ocK: 1e9, ocDp: 1, ocFluxMax: 1e4, ocEta: 80 }, Bm = batchModel(bv), bp = batchPMP(Bm, 100, 1e3), qc = 5 / 2;
    add('Batch RO without polarisation: the optimal flux is constant', 0, Math.max(...bp.u.map((u) => Math.abs(u[0] - qc))) / qc, 1e-7, 'Analytical result of the Euler–Lagrange equation when the osmotic term is an exact differential');
    add('Batch RO: optimal energy equals the closed form', Bm.w * ((qc * qc * 2) / Bm.a + 5 * 10 * Math.log(2) + 5), bp.J, 1e-7, 'w·(q²T/a + π₀V₀·ln(V₀/V_f) + Δp·ΔV)');
    const bc2 = batchModel({ ...bv, ocK: 8, ocFluxMax: 27, ocRec: 60, ocT: 3, ocArea: 80, ocPi0: 8 }), bq = batchPMP(bc2, 60, bc2.qmax), bd = batchDirect(bc2, 16, bc2.qmax, new Array(16).fill(2));
    add('Bounded control: Pontryagin and direct transcription agree on the energy', 0, (bd.J - bq.J) / bq.J, 1e-3, `Flux limit active; NLP with 16 intervals, ${bd.iterations} SQP iterations`);
    add('Bounded control: the control never exceeds its limit', 1, Math.max(...bq.u.map((u) => u[0])) / bc2.qmax, 1e-12, 'Clipping of the Hamiltonian minimiser');
    // neural ODE and neural PDE
    const gq = rng(3), trj = [0.4, -0.7].map((y0) => ({ t: [0, 0.3, 0.7, 1.2], y: [[y0, 0.2], ...range(3).map(() => [gq.normal(), gq.normal()])] })), pn0 = nodeInit(2, 4, 11), lg0 = nodeLossGrad(pn0, 2, 4, trj, 2);
    let nErr = 0;
    for (const i of [0, 5, 9, 13, 20, pn0.length - 1]) { const w = Float64Array.from(pn0), hh = 1e-6; w[i] += hh; const a = nodeLossGrad(w, 2, 4, trj, 2).loss; w[i] -= 2 * hh; const b = nodeLossGrad(w, 2, 4, trj, 2).loss; nErr = Math.max(nErr, Math.abs((a - b) / (2 * hh) - lg0.grad[i])); }
    add('Neural ODE: adjoint gradient through RK4 matches the numerical gradient', 0, nErr, 1e-7, 'Two states, four hidden neurons, six parameters checked');
    const tt = linspace(0, 2, 11), dec = [1, 0.5, -0.6].map((y0) => ({ t: tt, y: tt.map((t) => [y0 * Math.exp(-t)]) })), nf = nodeTrain(1, 5, dec, { epochs: 400, lr: 0.05, seed: 2, sub: 1 });
    add('Neural ODE learns exponential decay and predicts an unseen start', 0.75 * Math.exp(-2), nodeSolve(nf.p, 1, 5, [0.75], tt, 1)[10][0], 0.01, 'Trained on y(0) = 1, 0.5 and −0.6; tested from y(0) = 0.75 at t = 2');
    const opv = cdrOperator({ L: 1, nx: 12, v: 0.3, D: 0.02, left: { type: 'dirichlet', val: 2 }, right: { type: 'neumann', val: 0 } }), pv = nodeInit(1, 3, 7), exv = [{ op: opv, u0: zeros(12), nt: 10, dt: 0.2, obs: [{ k: 4, u: range(12).map((i) => 1 + 0.1 * i) }, { k: 10, u: range(12).map((i) => 2 - 0.05 * i) }] }], gv = npdeLossGrad(pv, 3, exv, { uRef: 2, rRef: 0.5 });
    let vErr = 0;
    for (const i of [0, 2, 4, 7, 9]) { const w = Float64Array.from(pv), hh = 1e-6; w[i] += hh; const a = npdeLossGrad(w, 3, exv, { uRef: 2, rRef: 0.5 }).loss; w[i] -= 2 * hh; const b = npdeLossGrad(w, 3, exv, { uRef: 2, rRef: 0.5 }).loss; vErr = Math.max(vErr, Math.abs((a - b) / (2 * hh) - gv.grad[i])); }
    add('Neural PDE: discrete-adjoint gradient matches the numerical gradient', 0, vErr, 1e-7, 'Transposed tri-diagonal solves backwards in time, five parameters checked');
    const fvv = solveCDR({ L: 1, nx: 12, tEnd: 2, nt: 10, v: 0.3, D: 0.02, ic: () => 0, left: { type: 'dirichlet', val: () => 2 }, right: { type: 'neumann', val: () => 0 }, theta: 0.5 }), imv = imexMarch(opv, zeros(12), 10, 0.2, () => 0, 0.5)[10];
    add('Neural-PDE time stepping reproduces the finite-volume solver without a closure', 0, Math.max(...imv.map((q, i) => Math.abs(q - fvv.u[i]))), 1e-12, 'Same operator and Crank–Nicolson scheme');
    // proper orthogonal decomposition
    const eg = symEig([[2, 1, 0], [1, 2, 0], [0, 0, 5]]);
    add('Jacobi eigen-solver: known eigenvalues', 0, Math.abs(eg.values[0] - 5) + Math.abs(eg.values[1] - 3) + Math.abs(eg.values[2] - 1), 1e-12, 'Eigenvalues 5, 3, 1');
    const m1 = range(9).map((j) => Math.sin(j / 3)), m2 = range(9).map((j) => Math.cos(j / 2)), snaps = range(7).map((k) => m1.map((q, j) => (k - 3) * q + 0.5 * k * k * m2[j])), pd = podBasis(snaps, { center: false });
    add('POD of rank-two snapshots finds exactly two modes', 2, pd.modes.length, 0, `Singular values ${pd.sigma.map((q) => fmt(q, 4)).join(', ')}`);
    add('POD modes are orthonormal and reconstruct the snapshots', 0, Math.abs(dot(pd.modes[0], pd.modes[1])) + Math.abs(dot(pd.modes[0], pd.modes[0]) - 1) + Math.max(...snaps.map((f) => { const a = pd.modes.map((ph) => dot(ph, f)); return maxAbs(f.map((q, j) => q - a[0] * pd.modes[0][j] - a[1] * pd.modes[1][j])); })), 1e-10, 'Projection residual of all seven snapshots');
    const opr = cdrOperator({ L: 1, nx: 10, v: 0.5, D: 0.1, left: { type: 'dirichlet', val: 1 }, right: { type: 'noflux', val: 0 } }), fomr = imexMarch(opr, new Array(10).fill(1), 40, 0.05, () => 0, 0.5), pr = podBasis(fomr.slice(1).map((u) => u.map((q) => q - 1)), { center: false }), gr2 = galerkinROM(opr, pr.modes, new Array(10).fill(1), 40, 0.05);
    add('Galerkin reduced model with a complete basis reproduces the full model', 0, Math.max(...gr2.lift(gr2.a[40]).map((q, i) => Math.abs(q - fomr[40][i]))), 5e-6, `${pr.modes.length} modes of a 10-cell problem (modes below 10⁻¹³ of the leading eigenvalue are dropped)`);
    // ensembles, multi-fidelity, chance constraints
    const ge = rng(17), Xe = range(80).map(() => [ge.uniform(-1, 1), ge.uniform(-1, 1)]), ye = Xe.map((x) => 2 * x[0] - x[1] + ge.normal(0, 0.1)), bg = bagEnsemble(Xe, ye, (Xa, ya) => polyFit(Xa, ya, 1).predict, { B: 20, seed: 4 }), Xq = range(40).map(() => [ge.uniform(-1, 1), ge.uniform(-1, 1)]), yq = Xq.map((x) => 2 * x[0] - x[1]);
    const Pq = Xq.map((x) => bg.predict(x)), mseE = mean(Pq.map((p, i) => (p.mean - yq[i]) ** 2)), mseM = mean(range(20).map((b) => mean(Pq.map((p, i) => (p.all[b] - yq[i]) ** 2)))), amb = mean(Pq.map((p) => mean(p.all.map((q) => (q - p.mean) ** 2))));
    add('Ensemble: ambiguity decomposition holds exactly', 0, Math.abs(mseM - mseE - amb) / mseM, 1e-10, 'mean member MSE = ensemble MSE + mean member variance');
    add('Ensemble: out-of-bag error recovers the noise level', 0.1, bg.oob.rmse, 0.025, 'Bagged linear fits of y = 2x₁ − x₂ + N(0, 0.1²), 80 rows');
    const fH = (x) => (6 * x - 2) ** 2 * Math.sin(12 * x - 4), fL = (x) => 0.5 * fH(x) + 10 * (x - 0.5) - 5, xL = linspace(0, 1, 11).map((x) => [x]), xH = [0, 0.4, 0.6, 0.8, 1].map((x) => [x]), ckf = coKrige(xL, xL.map((x) => fL(x[0])), xH, xH.map((x) => fH(x[0]))), gHf = gpFit(xH, xH.map((x) => fH(x[0]))), xtf = linspace(0.03, 0.97, 25);
    const eMFf = rmse(xtf.map((x) => ckf.predict([x]).mean), xtf.map(fH)), eSFf = rmse(xtf.map((x) => gHf.predict([x]).mean), xtf.map(fH));
    add('Co-kriging beats the single-fidelity process on the Forrester benchmark', 0, eMFf / eSFf, 0.2, `RMSE ${fmt(eMFf, 3)} against ${fmt(eSFf, 3)} with 5 high- and 11 low-fidelity points; ρ = ${fmt(ckf.rho, 4)} (exact 2), discrepancy: ${ckf.kind}`);
    const xi = lhs(400, 1, 3).map((u) => normInv(u[0])), cc = (al) => nelderMead((x) => x[0] + 50 * chanceViolation(xi.map((q) => q - x[0]), al).viol, [3], { lo: [-5], hi: [5], maxIter: 400, tol: 1e-12, scale: 0.5 }).x[0];
    add('Chance constraint by sample-average approximation: 90 % quantile of a normal variable', 1.2816, cc(0.1), 0.03, 'min x s.t. P[ξ ≤ x] ≥ 0.9, ξ ~ N(0, 1), 400 Latin-hypercube scenarios');
    add('Worst-case constraint equals the largest scenario', Math.max(...xi), cc(0), 1e-4, 'α = 0: every scenario must be feasible');
    // energy and scaling constraints of the RO optimisation, and the worker pool
    const dv = Object.fromEntries(suite.inputs.flatMap((gq2) => gq2.fields).map((f) => [f.key, f.value])), vb = roBase(dv), e0 = evalDesign(vb, dv), vc = { ...dv, limSEC: 0.9 * e0.m.sec, conScale: 'gypsum', limSI: -0.1 };
    warmReset();
    const e1 = evalDesign(vb, vc), aw = analyzeWater({ ions: e1.m.concIons, T: vb.T, pH: e1.m.concPH });
    add('Energy constraint: normalised value of a 10 % tighter cap', 1 / 0.9 - 1, e1.g[6], 1e-6, 'g = (SEC − cap)/cap with cap = 0.9 × SEC');
    add('Scaling constraint: saturation index equals the full speciation of suite 2', aw.SI.gypsum, e1.m.si, 1e-9, `Gypsum in the concentrate at ${fmt(e0.m.rec, 3)} % recovery; g = SI − limit = ${fmt(e1.g[7], 4)}`);
    const e2 = evalDesign({ ...vb, recovery: vb.recovery + 8 }, vc);
    add('Scaling constraint tightens with recovery', 1, e2.m.si > e1.m.si ? 1 : 0, 0, `Gypsum SI ${fmt(e1.m.si, 4)} → ${fmt(e2.m.si, 4)} for +8 points of recovery`);
    const jobs = [40, 44, 48, 52].map((q) => ({ recovery: q })), bt = await makeBatcher(dv, vb, 2, { tick: async () => {} });
    try {
      const got = await bt.run(jobs), want = [evalJobs(jobs.slice(0, 2), dv, vb), evalJobs(jobs.slice(2), dv, vb)].flat(), live = bt.pool.live && !bt.fellBack;
      add(live ? 'Parallel evaluation on worker threads reproduces the calling thread' : 'Chunked evaluation reproduces the direct evaluation (worker threads unavailable here)', 0, Math.max(...got.map((e, i) => Math.abs(e.m.sec - want[i].m.sec) + Math.abs(e.cost.total - want[i].cost.total))), 1e-12, live ? `Four plant simulations on ${bt.pool.workers} threads, started in ${bt.pool.startMs} ms` : `Sequential fallback: ${bt.pool.reason}`);
    } finally { bt.pool.close(); }
    warmReset();
    }
    return C;
  },
};

/** Synthetic "measured" permeability decline: the fouling model with slightly different true parameters plus noise. */
function foulingSynth(seed, days) {
  const g = rng(seed);
  return days.map((t) => ({ t, Kn: +(0.74 + 0.26 * Math.exp(-0.0052 * t) + g.normal(0, 0.004)).toFixed(4) }));
}

// Worker-thread entry: when this module is loaded inside a worker started by parPool it answers batch-evaluation requests.
{
  const wt = nodeThreads();
  if (wt && !wt.isMainThread && wt.workerData?.brinelabPar) wt.parentPort.on('message', (m) => { const r = parHandle(m); if (r) wt.parentPort.postMessage(r); });
  else if (typeof DedicatedWorkerGlobalScope !== 'undefined' && globalThis instanceof DedicatedWorkerGlobalScope) globalThis.addEventListener('message', (e) => { const r = parHandle(e.data); if (r) globalThis.postMessage(r); });
}

export default suite;
