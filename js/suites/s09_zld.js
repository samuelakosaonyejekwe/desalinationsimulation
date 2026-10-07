// Suite 9 — Brine concentration, crystallization and zero liquid discharge.
// A brine is softened (optional selective precipitation), pre-concentrated by high-pressure or
// osmotically assisted RO, evaporated in a falling-film brine concentrator (MVC, MEE or MEE-TVC) and
// taken to solids in a forced-circulation crystallizer with centrifuge, dryer and purge handling.
// The evaporation path is followed stepwise with Pitzer-based mineral equilibria (suite 2 engine), so the
// order, onset and mass of every salt are predicted; the crystallizer is an MSMPR population balance
// solved by the method of moments and by a finite-volume discretisation of the size coordinate.
import { brent, clamp, linspace, sum, rng, fmt, rk45, interp1 } from '../core/num.js';
import { psat, tsat, latentHeat, cp, viscosity, R, KELVIN } from '../core/props.js';
import { IONS, ION_IDS, WATERS, scaleIons } from '../core/water.js';
import { MINERALS, EVAPORITE_MINERALS, REAGENTS, ACTIVITY_MODELS, makeSolution, equilibrate, precipitateSolution, doseSolution, solutionToIons, saturationIndex, componentIndex } from './s02_chem.js';

const MW_W = 0.0180153, RHO_W = 997, KB = 1.380649e-23, NA = 6.02214076e23, LN10 = Math.LN10;
const iOf = Object.fromEntries(['Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'C'].map((k) => [k, componentIndex(k)]));
const TRACK = ['calcite', 'gypsum', 'anhydrite', 'halite', 'glauberite', 'epsomite', 'sylvite', 'carnallite', 'bischofite'];
const SIMPLE_SET = ['calcite', 'gypsum', 'anhydrite', 'silica', 'brucite', 'halite', 'thenardite', 'glauberite', 'epsomite', 'sylvite', 'carnallite', 'bischofite'];

/** Specific heat of NaCl-type brines up to saturation, J/(kg·K) (the seawater polynomial is not valid above 180 g/kg). */
const cpBrine = (T, S) => cp(T, 0) * (1 - 9.2e-4 * S + 3.2e-7 * S * S);
const hL = (T, S) => cpBrine(T / 2, S) * T; // J/kg relative to 0 °C
/** Boiling-point elevation from the water activity: the vapour above the brine is at aw·psat(T). */
export const bpeFromAw = (T, aw) => T - tsat(clamp(aw, 0.05, 1) * psat(T));
const osmoticBar = (T, aw) => (-R * (T + KELVIN) * Math.log(aw)) / 1.807e-5 / 1e5;
/** Shaft work of a vapour compressor per kg of vapour, J/kg: suction at psat(T − BPE), superheated at T; discharge condenses at T + dT. */
export function compressorWork(T, bpe, dT, eta) {
  const k = 1.33, Rv = 461.5, p1 = psat(T - bpe), p2 = psat(T + dT);
  return ((k / (k - 1)) * Rv * (T + KELVIN) * ((p2 / p1) ** ((k - 1) / k) - 1)) / eta;
}

// ---- evaporation path -------------------------------------------------------------------------------
/**
 * Stepwise water removal with fractional crystallisation. segs = [{ unit, wEnd, T, precip, pCO2 }] gives the
 * water inventory (kg, same basis as sol.w) at which each unit ends. Solids formed in a step are removed.
 */
export function evaporationPath(sol0, segs, { nEvap = 36, minerals = EVAPORITE_MINERALS } = {}) {
  const steps = [], cum = Object.fromEntries(minerals.map((k) => [k, 0])), onset = {}, evap = {}, solids = {}, w00 = sol0.w;
  const snap = (s, unit, T, step) => {
    const io = solutionToIons(s), e = s.eq, SI = {};
    for (const id of minerals) SI[id] = MINERALS[id]._st.every(([i]) => e.tot[i] > 0) ? saturationIndex(e, id) : -99;
    return { unit, cf: w00 / s.w, w: s.w, T, S: io.salinity, tds: io.tds, rho: io.density, gPerKgw: io.gPerKgw, aw: e.aw, I: e.I, pH: e.pH, SI, m: Object.fromEntries(Object.entries(iOf).map(([k, i]) => [k, e.tot[i]])), step, cum: { ...cum }, ions: io.ions };
  };
  let sol = sol0.eq ? sol0 : equilibrate(sol0), hint = [], stopped = null;
  steps.push(snap(sol, 'feed', sol.T, {}));
  const act = [];
  let wPrev = sol.w;
  for (const g of segs) { if (g.wEnd < wPrev * (1 - 1e-9)) { act.push({ ...g, w0: wPrev }); wPrev = g.wEnd; } }
  const lnTot = sum(act.map((g) => Math.log(g.w0 / g.wEnd))) || 1;
  outer: for (const g of act) {
    const ns = Math.max(2, Math.round((nEvap * Math.log(g.w0 / g.wEnd)) / lnTot));
    evap[g.unit] = 0; solids[g.unit] = {};
    sol = { ...sol, T: g.T };
    for (let k = 1; k <= ns; k++) {
      const wT = g.w0 * (g.wEnd / g.w0) ** (k / ns), prev = steps[steps.length - 1];
      if (!(wT < sol.w)) continue; // water of hydration already took the inventory below this target
      const dW = sol.w - wT;
      let s, r = null;
      try {
        if (g.precip) { r = precipitateSolution({ ...sol, w: wT }, minerals, { pCO2: g.pCO2, hint }); hint = r.active; s = r.sol; } else s = equilibrate({ ...sol, w: wT });
      } catch (err) { stopped = `equilibrium solver stopped at concentration factor ${fmt(w00 / sol.w, 3)} (${err.message})`; break outer; }
      if (!Number.isFinite(s.eq.aw) || !Number.isFinite(s.pH)) { stopped = `model range exceeded at concentration factor ${fmt(w00 / sol.w, 3)}`; break outer; }
      const step = {};
      if (r) for (const id of minerals) {
        const x = Math.max(0, r.solids[id]);
        if (x > 1e-11 * w00) {
          step[id] = x; cum[id] += x; solids[g.unit][id] = (solids[g.unit][id] || 0) + x;
          if (!onset[id]) { // log-linear interpolation of the saturation index between the previous state and this (unprecipitated) one
            const a = prev.SI[id], b = r.si0[id], cf1 = prev.cf, cf2 = w00 / wT;
            onset[id] = { cf: a < 0 && b > a ? Math.exp(Math.log(cf1) + ((0 - a) / (b - a)) * Math.log(cf2 / cf1)) : cf1, unit: g.unit };
          }
        }
      }
      evap[g.unit] += dW; sol = s;
      steps.push(snap(sol, g.unit, g.T, step));
      if (sol.eq.aw < 0.33 || sol.eq.I > 22) { stopped = `mother liquor reached the end of the model range (water activity ${fmt(sol.eq.aw, 3)}, ionic strength ${fmt(sol.eq.I, 3)} mol/kg) at concentration factor ${fmt(w00 / sol.w, 3)}`; break outer; }
    }
  }
  return { steps, sol, cum, onset, evap, solids, stopped };
}

// ---- MSMPR crystallizer ------------------------------------------------------------------------------
/** Steady state of an MSMPR crystallizer with secondary nucleation B = kb·MT^j·σ^b and growth G = kg·σ^g. */
export function msmprSteady(K, tau, MT) {
  const sigma = (MT ** (1 - K.j) / (6 * K.kv * K.rhoc * K.kb * K.kg ** 3 * tau ** 4)) ** (1 / (K.b + 3 * K.g));
  const G = K.kg * sigma ** K.g, B = K.kb * MT ** K.j * sigma ** K.b;
  return { sigma, G, B, n0: B / G, L50: 3.67206 * G * tau, L43: 4 * G * tau, L32: 3 * G * tau, cv: 0.5, mu0: B * tau };
}
const kinetics = (K, sig, mt) => {
  const s = clamp(sig, 0, 3), G = K.kg * s ** K.g, ln = Math.log1p(s);
  const Bp = s > 1e-6 ? K.primA * Math.exp(-K.primB / (ln * ln)) : 0; // classical primary nucleation
  return { G, Bp, B: Bp + K.kb * Math.max(mt, 0) ** K.j * s ** K.b };
};
/**
 * Start-up dynamics by the method of moments (μ0…μ5 and the supersaturation), in scaled variables:
 * θ = t/τ, moments scaled by the steady-state population so that all unknowns are O(1).
 */
export function msmprDynamic(K, tau, MT, seed, tEnd = 12) {
  const ss = msmprSteady(K, tau, MT), Lr = ss.G * tau, N0 = ss.n0 * Lr, m3 = K.rhoc * K.kv * N0 * Lr ** 3;
  const mu0 = seed.mass / (K.rhoc * K.kv * seed.L ** 3), x = seed.L / Lr;
  const y0 = [0, 1, 2, 3, 4, 5].map((j) => (mu0 / N0) * x ** j).concat([seed.sigma]);
  const f = (th, y) => {
    const r = kinetics(K, y[6], m3 * y[3]), gr = (r.G * tau) / Lr;
    return [(r.B * tau) / N0 - y[0], gr * y[0] - y[1], 2 * gr * y[1] - y[2], 3 * gr * y[2] - y[3], 4 * gr * y[3] - y[4], 5 * gr * y[4] - y[5], (MT - 3 * m3 * gr * y[2]) / K.cstar - y[6]];
  };
  const sol = rk45(f, y0, 0, tEnd, { rtol: 1e-6, atol: 1e-10, hInit: 1e-4, maxSteps: 60000 });
  const out = { th: sol.t, sigma: [], G: [], B: [], Bp: [], mt: [], L43: [], cv: [], mu0: [], ss, Lr, N0, complete: sol.t[sol.t.length - 1] >= tEnd - 1e-9 };
  out.th = [];
  sol.y.forEach((y, i) => {
    if (i > 0 && !(sol.t[i] > out.th[out.th.length - 1])) return; // keep the time axis strictly increasing for interpolation
    const mt = m3 * y[3], r = kinetics(K, y[6], mt);
    out.th.push(sol.t[i]); out.sigma.push(y[6]); out.G.push(r.G); out.B.push(r.B); out.Bp.push(r.Bp); out.mt.push(mt); out.mu0.push(y[0] * N0);
    out.L43.push(y[3] > 0 ? (Lr * y[4]) / y[3] : 0); out.cv.push(y[4] > 0 ? Math.sqrt(Math.max(0, (y[5] * y[3]) / (y[4] * y[4]) - 1)) : 0);
  });
  return out;
}
/**
 * Discretised population balance ∂n/∂t + G ∂n/∂L = −n/τ with the nucleation flux G·n(0) = B as inlet boundary.
 * Finite volumes on a uniform size grid, upwind or van Leer-limited fluxes, Heun (SSP-RK2) time stepping.
 * Growth and nucleation histories come from the moment solution.
 */
export function pbeFV(K, tau, dyn, seed, { nL = 80, Lmax, scheme = 'vanleer', tEnd = 12 } = {}) {
  const N = Math.max(4, Math.round(nL)), dL = Lmax / N, n = new Float64Array(N), a = new Float64Array(N), b = new Float64Array(N), F = new Float64Array(N + 1);
  n[Math.min(N - 1, Math.floor(seed.L / dL))] = seed.mass / (K.rhoc * K.kv * seed.L ** 3) / dL;
  const hi = scheme === 'vanleer';
  const rhs = (u, G, B, out) => {
    F[0] = B;
    for (let i = 0; i < N; i++) {
      let face = u[i];
      if (hi && i > 0 && i < N - 1) { const dm = u[i] - u[i - 1], dp = u[i + 1] - u[i], pr = dm * dp; if (pr > 0) face += pr / (dm + dp); } // van Leer limiter ½ψ(r)·Δ in harmonic-mean form
      F[i + 1] = G * face;
    }
    for (let i = 0; i < N; i++) out[i] = -(F[i + 1] - F[i]) / dL - u[i] / tau;
  };
  let t = 0, nSteps = 0;
  const T = tEnd * tau;
  while (t < T - 1e-9 * T && nSteps < 400000) {
    const G = interp1(dyn.th, dyn.G, t / tau), dt = Math.min((0.4 * dL) / Math.max(G, 1e-30), 0.04 * tau, T - t), t1 = (t + dt) / tau;
    rhs(n, G, interp1(dyn.th, dyn.B, t / tau), a);
    for (let i = 0; i < N; i++) b[i] = n[i] + dt * a[i];
    rhs(b, interp1(dyn.th, dyn.G, t1), interp1(dyn.th, dyn.B, t1), a);
    for (let i = 0; i < N; i++) n[i] = Math.max(0, 0.5 * n[i] + 0.5 * (b[i] + dt * a[i]));
    t += dt; nSteps++;
  }
  const edge = (i) => i * dL, mom = (j) => { let s = 0; for (let i = 0; i < N; i++) s += (n[i] * (edge(i + 1) ** (j + 1) - edge(i) ** (j + 1))) / (j + 1); return s; };
  const mu = [0, 1, 2, 3, 4, 5].map(mom), cm = [0];
  for (let i = 0; i < N; i++) cm.push(cm[i] + (n[i] * (edge(i + 1) ** 4 - edge(i) ** 4)) / 4);
  const q = (p) => { const target = p * cm[N]; for (let i = 0; i < N; i++) if (cm[i + 1] >= target) return edge(i) + (dL * (target - cm[i])) / Math.max(cm[i + 1] - cm[i], 1e-300); return Lmax; };
  return { L: Array.from({ length: N }, (_, i) => (i + 0.5) * dL), n: Array.from(n), dL, mu, L10: q(0.1), L50: q(0.5), L90: q(0.9), L43: mu[3] > 0 ? mu[4] / mu[3] : 0, cv: mu[4] > 0 ? Math.sqrt(Math.max(0, (mu[5] * mu[3]) / (mu[4] * mu[4]) - 1)) : 0, mt: K.rhoc * K.kv * mu[3], nSteps, tail: n[N - 1] * dL / Math.max(mu[0], 1e-300) };
}
const kin = (v, salt, cstar) => {
  const M = MINERALS[salt] || MINERALS.halite, kT = KB * (v.Tcx + KELVIN), vm = M.mw / 1000 / (M.rho * NA);
  return { kg: v.kg, g: v.gExp, kb: v.kb, b: v.bExp, j: v.jMT, kv: v.kv, rhoc: M.rho, cstar, primA: 10 ** v.logA, primB: (v.het * 16 * Math.PI * (M.sigma * 1e-3) ** 3 * vm * vm) / (3 * kT ** 3) };
};

// ---- Gibbs–Thomson, dissolution and Ostwald ripening ---------------------------------------------------
const D_SALT = 1.5e-9; // m²/s, diffusivity of the dissolved salt
/** Capillary (Gibbs–Thomson) length α = 2γVm/(νRT) for the radius: ln(c_r/c∞) = α/r. sigma in J/m², vm in m³/mol. */
export const capillaryLength = (gamma, vm, T, nu = 1) => (2 * gamma * vm) / (nu * R * (T + KELVIN));
/** Relative supersaturation in equilibrium with a crystal of size L (diameter or edge): σ*(L) = exp(2α/L) − 1. */
export const sigmaCrit = (L, alpha) => Math.expm1(Math.min((2 * alpha) / Math.max(L, 1e-12), 50));
/** Lifshitz–Slyozov–Wagner coarsening constant K (m³/s): ⟨r⟩³ = r₀³ + K·t with K = (4/9)·D·c∞·Vm·α. cInf in mol/m³. */
export const lswConstant = (D, cInf, vm, alpha) => (4 / 9) * D * cInf * vm * alpha;
/** Noyes–Whitney dissolution velocity of a crystal face (m/s): dL/dt = −2·k_L·c*·(under-saturation)/ρc, with k_L = Sh·D/L. cstar in kg/m³. */
export const dissolutionVelocity = (L, under, cstar, rhoc, Sh = 2, D = D_SALT) => (2 * Sh * D * cstar * Math.max(under, 0)) / (rhoc * Math.max(L, 1e-12));
/**
 * Ostwald ripening of a crystal population in a closed, saturated slurry (diffusion-controlled, LSW closure):
 * dr/dt = (A/r)·(1/r* − 1/r) with A = D·c∞·Vm·α and the critical radius r* = number-mean radius, which keeps the
 * crystal volume constant. Lagrangian size classes [{ N, r }]; classes that dissolve are removed.
 */
export function ripen(classes, A, tEnd, { maxSteps = 8000, nOut = 40 } = {}) {
  const src = classes.filter((c) => c.N > 0 && c.r > 0), Nn = Float64Array.from(src, (c) => c.N), r = Float64Array.from(src, (c) => c.r), rIni = Float64Array.from(r), hf = new Float64Array(src.length), st = [0, 0, 0];
  let m = src.length;
  const stat = () => { let s0 = 0, s1 = 0, s3 = 0; for (let i = 0; i < m; i++) { const x = r[i], w = Nn[i]; s0 += w; s1 += w * x; s3 += w * x * x * x; } st[0] = s0; st[1] = s1; st[2] = s3; };
  stat();
  const v0 = st[2], hist = { t: [0], rMean: [st[0] > 0 ? st[1] / st[0] : 0], number: [st[0]], volume: [v0] };
  let t = 0, steps = 0, next = tEnd / nOut;
  while (t < tEnd * (1 - 1e-12) && m > 1 && steps < maxSteps) {
    const rs = st[1] / st[0];
    let dt = tEnd - t;
    for (let i = 0; i < m; i++) { const v = Math.abs((A / r[i]) * (1 / rs - 1 / r[i])); if (v > 0) { const lim = (0.12 * r[i]) / v; if (lim < dt) dt = lim; } }
    if (dt < 1e-9 * tEnd) dt = 1e-9 * tEnd;
    // midpoint (RK2) step with the critical radius re-evaluated at the half step
    let s0 = 0, s1 = 0;
    for (let i = 0; i < m; i++) { const h = Math.max(r[i] + 0.5 * dt * (A / r[i]) * (1 / rs - 1 / r[i]), 1e-3 * r[i]); hf[i] = h; s0 += Nn[i]; s1 += Nn[i] * h; }
    const rm = s1 / s0;
    for (let i = 0; i < m; i++) r[i] += dt * (A / hf[i]) * (1 / rm - 1 / hf[i]);
    t += dt; steps++;
    // dissolved classes leave; the crystal volume is restored exactly (closed system)
    stat();
    const lim = 2e-3 * (st[1] / st[0]);
    let k = 0;
    for (let i = 0; i < m; i++) if (r[i] > 0.22 * rIni[i] && r[i] > lim) { r[k] = r[i]; Nn[k] = Nn[i]; rIni[k] = rIni[i]; k++; }
    m = k; stat();
    const f = Math.cbrt(v0 / st[2]);
    for (let i = 0; i < m; i++) r[i] *= f;
    stat();
    if (t >= next || t >= tEnd * (1 - 1e-12)) { hist.t.push(t); hist.rMean.push(st[1] / st[0]); hist.number.push(st[0]); hist.volume.push(st[2]); next = t + tEnd / nOut; }
  }
  return { ...hist, classes: Array.from({ length: m }, (_, i) => ({ N: Nn[i], r: r[i] })), steps, complete: t >= tEnd * (1 - 1e-9), A };
}

// ---- two-zone (compartment) crystallizer with size-dependent growth, dissolution and wall deposition -------
/**
 * Population balance of a forced-circulation crystallizer resolved into two compartments that exchange slurry with
 * the circulation flow: zone 1 = boiling (flash) zone where the supersaturation is generated, zone 2 = body and
 * heater loop, from which the product is withdrawn. Each zone has its own supersaturation and size distribution.
 * Growth is size dependent through the Gibbs–Thomson relation (crystals smaller than the critical size dissolve with
 * a diffusion-limited Noyes–Whitney rate), crystals may deposit on the wall with the first-order rate kw (1/s; 0 = no-flux wall),
 * and zone 2 may be shifted in saturation by the heater temperature rise (dSig2, relative solubility change).
 * Finite volumes on a geometric size grid, upwind fluxes, implicit Euler with a block-tridiagonal solve; the dissolved
 * excess follows from an exact mass balance each step. o.phi1 = volume share of zone 1, o.theta = turnover time (s).
 */
export function zoneCrystallizer(K, tau, MT, seed, o = {}) {
  const N = Math.max(20, Math.round(o.nL ?? 120)), Lmin = o.Lmin ?? 1e-6, Lmax = o.Lmax ?? 5e-3, rat = (Lmax / Lmin) ** (1 / N), phi = [clamp(o.phi1 ?? 0.15, 0.01, 0.99), 0], theta = Math.max(o.theta ?? 60, 1e-6), kw = Math.max(o.kw ?? 0, 0), alpha = o.gt === false ? 0 : Math.max(o.alpha ?? 0, 0), dSig = [0, o.dSig2 ?? 0];
  phi[1] = 1 - phi[0];
  const e = Float64Array.from({ length: N + 1 }, (_, i) => Lmin * rat ** i), Lc = Float64Array.from({ length: N }, (_, i) => 0.5 * (e[i] + e[i + 1])), dL = Lc.map((_, i) => e[i + 1] - e[i]), L3 = Lc.map((x) => x ** 3), rk = K.rhoc * K.kv, cs = K.cstar;
  const scF = e.map((x) => (alpha > 0 ? sigmaCrit(x, alpha) : 0)), kdF = e.map((x) => (2 * 2 * D_SALT * cs) / (K.rhoc * x)), dL3 = Float64Array.from({ length: N }, (_, f) => (f > 0 ? L3[f] - L3[f - 1] : 0)); // critical supersaturation and dissolution coefficient at the faces
  const g1 = K.g === 1, kgc = K.kg, gex = K.g, vel = (sig, f) => { const d = sig - scF[f]; return d >= 0 ? (g1 ? kgc * (d < 3 ? d : 3) : kgc * (d < 3 ? d : 3) ** gex) : kdF[f] * (d > -1 ? d : -1); };
  const n = [new Float64Array(N), new Float64Array(N)], x = [seed.sigma * cs, seed.sigma * cs], is = clamp(Math.floor(Math.log(Math.max(seed.L, Lmin) / Lmin) / Math.log(rat)), 0, N - 1);
  for (const z of [0, 1]) n[z][is] = seed.mass / (rk * L3[is]) / dL[is];
  const mass = (z) => { let s = 0; for (let i = 0; i < N; i++) s += n[z][i] * dL[i] * L3[i]; return rk * s; };
  const birth = (sig) => { if (!(alpha > 0) || !(sig > 0)) return 0; const Ls = (2 * alpha) / Math.log1p(sig); let i = 0; while (i < N - 1 && e[i] < Ls) i++; return i; }; // nuclei appear just above the critical size
  const nuc = (sig, mt) => (sig > 0 ? kinetics(K, sig, mt).B : 0);
  // growth uptake of dissolved salt per unit zone volume (kg/m³/s) for a given population
  const uptake = (z, sig, pop, mt) => {
    let u = 0;
    for (let f = 1; f < N; f++) { const v = vel(sig, f); u += v * (v >= 0 ? pop[f - 1] : pop[f]) * dL3[f]; }
    const v0 = vel(sig, 0);
    if (v0 < 0) u += v0 * pop[0] * L3[0]; // complete dissolution through the lower boundary
    return rk * (u + nuc(sig, mt) * L3[birth(sig)]);
  };
  // time steps: fine during the start-up transient (first third of the run), four times longer afterwards
  const S = MT / tau, dt1 = tau / Math.max(10, Math.round(o.nt ?? 40)), tRun = (o.tEnd ?? 12) * tau, n1 = Math.ceil(tRun / 3 / dt1), n2 = Math.ceil((tRun - n1 * dt1) / (4 * dt1)), nSteps = n1 + n2, dt2 = (tRun - n1 * dt1) / Math.max(n2, 1), ex = [1 / (theta * phi[0]), 1 / (theta * phi[1])], wd = [0, 1 / (tau * phi[1])], src = [S / phi[0], 0];
  const hist = { th: [0], s1: [seed.sigma], s2: [seed.sigma - dSig[1]], mt: [phi[0] * mass(0) + phi[1] * mass(1)], L43: [] };
  const acc = { made: 0, product: 0, dissolvedOut: 0, wall: 0, top: 0 }, m0 = phi[0] * (mass(0) + x[0]) + phi[1] * (mass(1) + x[1]);
  const sigOf = (z, xx) => xx / cs - dSig[z];
  const nw = [new Float64Array(N), new Float64Array(N)], bl = new Float64Array(2 * N), bu = new Float64Array(2 * N), bd = new Float64Array(4 * N), br = new Float64Array(2 * N), sol = new Float64Array(2 * N), cp = new Float64Array(4 * N), dp = new Float64Array(2 * N);
  let time = 0, jac = null, fresh = false; // Jacobian of the two zone balances, reused between iterations and steps while it converges
  for (let k = 0; k < nSteps; k++) {
    const mt = [mass(0), mass(1)], dt = k < n1 ? dt1 : dt2;
    time += dt;
    // 1 — implicit solve of the dissolved excess for a given population: damped Newton on the two zone balances
    const xLo = -0.95 * cs, xHi = 3.2 * cs + Math.max(dSig[1], 0) * cs;
    const solveX = (pop, a0, b0) => {
      const F = (a, b) => { const u0 = uptake(0, sigOf(0, a), pop[0], mt[0]), u1 = uptake(1, sigOf(1, b), pop[1], mt[1]); return [(a - x[0]) / dt - src[0] + u0 - ex[0] * (b - a), (b - x[1]) / dt + u1 - ex[1] * (a - b) + wd[1] * b]; };
      let a = a0, b = b0, f0 = F(a, b), nrm = Math.abs(f0[0]) + Math.abs(f0[1]);
      for (let it = 0; it < 40 && nrm > 1e-8 * (S + cs / dt); it++) {
        if (!jac) { const h = 1e-6 * cs, fa = F(a + h, b), fb = F(a, b + h); jac = [(fa[0] - f0[0]) / h, (fb[0] - f0[0]) / h, (fa[1] - f0[1]) / h, (fb[1] - f0[1]) / h]; fresh = true; } else fresh = false;
        const det = jac[0] * jac[3] - jac[1] * jac[2];
        if (!(Math.abs(det) > 0)) { if (fresh) break; jac = null; continue; }
        const da = (-f0[0] * jac[3] + f0[1] * jac[1]) / det, db = (-f0[1] * jac[0] + f0[0] * jac[2]) / det;
        let lam = 1, a2 = a, b2 = b, f2 = f0, n2 = nrm;
        for (let q = 0; q < (fresh ? 12 : 2); q++) { a2 = clamp(a + lam * da, xLo, xHi); b2 = clamp(b + lam * db, xLo, xHi); f2 = F(a2, b2); n2 = Math.abs(f2[0]) + Math.abs(f2[1]); if (n2 < nrm) break; lam *= 0.5; }
        if (!(n2 < nrm)) { if (fresh) break; jac = null; continue; } // a stale Jacobian is rebuilt once before giving up
        const moved = Math.abs(a2 - a) + Math.abs(b2 - b), slow = n2 > 0.3 * nrm;
        a = a2; b = b2; f0 = f2; nrm = n2;
        if (slow) jac = null; // chord iteration while it converges fast, a new Jacobian otherwise
        if (moved < 1e-11 * cs) break;
      }
      return [a, b];
    };
    // 2 — implicit population step for a given supersaturation: block-tridiagonal system in (zone 1, zone 2) per size cell
    const popStep = (sg, B, ib) => {
      for (let i = 0; i < N; i++) for (const z of [0, 1]) {
        const vl = vel(sg[z], i), vu = vel(sg[z], i + 1), q = 2 * i + z, out = (Math.max(vu, 0) - Math.min(vl, 0)) / dL[i];
        bl[q] = i > 0 ? Math.max(vl, 0) / dL[i] : 0; bu[q] = i < N - 1 ? -Math.min(vu, 0) / dL[i] : 0;
        bd[2 * q + z] = 1 / dt + out + ex[z] + wd[z] + kw; bd[2 * q + (1 - z)] = -ex[z];
        br[q] = n[z][i] / dt + (i === ib[z] ? B[z] / dL[i] : 0);
      }
      // block Thomas algorithm with 2×2 blocks (off-diagonal blocks are diagonal)
      for (let i = 0; i < N; i++) {
        let d00 = bd[4 * i], d01 = bd[4 * i + 1], d10 = bd[4 * i + 2], d11 = bd[4 * i + 3], r0 = br[2 * i], r1 = br[2 * i + 1];
        if (i > 0) { const l0 = bl[2 * i], l1 = bl[2 * i + 1]; d00 -= l0 * cp[4 * (i - 1)]; d01 -= l0 * cp[4 * (i - 1) + 1]; d10 -= l1 * cp[4 * (i - 1) + 2]; d11 -= l1 * cp[4 * (i - 1) + 3]; r0 += l0 * dp[2 * (i - 1)]; r1 += l1 * dp[2 * (i - 1) + 1]; }
        const det = d00 * d11 - d01 * d10, i00 = d11 / det, i01 = -d01 / det, i10 = -d10 / det, i11 = d00 / det, u0 = bu[2 * i], u1 = bu[2 * i + 1];
        cp[4 * i] = i00 * u0; cp[4 * i + 1] = i01 * u1; cp[4 * i + 2] = i10 * u0; cp[4 * i + 3] = i11 * u1; // X_i = dp_i + cp_i·X_{i+1}
        dp[2 * i] = i00 * r0 + i01 * r1; dp[2 * i + 1] = i10 * r0 + i11 * r1;
      }
      sol[2 * N - 2] = dp[2 * N - 2]; sol[2 * N - 1] = dp[2 * N - 1];
      for (let i = N - 2; i >= 0; i--) { sol[2 * i] = dp[2 * i] + cp[4 * i] * sol[2 * i + 2] + cp[4 * i + 1] * sol[2 * i + 3]; sol[2 * i + 1] = dp[2 * i + 1] + cp[4 * i + 2] * sol[2 * i + 2] + cp[4 * i + 3] * sol[2 * i + 3]; }
      for (let i = 0; i < N; i++) { nw[0][i] = sol[2 * i] > 1e-200 ? sol[2 * i] : 0; nw[1][i] = sol[2 * i + 1] > 1e-200 ? sol[2 * i + 1] : 0; } // also keeps denormal numbers out of the solver
    };
    // fixed-point between the two: the supersaturation is re-solved with the new population until both agree
    let sg = [0, 0], B = [0, 0], ib = [0, 0], pop = n, xa = x[0], xb = x[1];
    for (let pass = 0; pass < 6; pass++) {
      [xa, xb] = solveX(pop, xa, xb);
      const s2 = [sigOf(0, xa), sigOf(1, xb)], change = Math.abs(s2[0] - sg[0]) + Math.abs(s2[1] - sg[1]);
      if (pass > 0 && change < 1e-3 * (Math.abs(s2[0]) + Math.abs(s2[1]) + 1e-6)) break;
      sg = s2; B = [nuc(sg[0], mt[0]), nuc(sg[1], mt[1])]; ib = [birth(sg[0]), birth(sg[1])];
      popStep(sg, B, ib); pop = nw;
    }
    n[0].set(nw[0]); n[1].set(nw[1]);
    // 3 — realised uptake with the new population, then the exact dissolved balance (linear 2×2 in the exchange terms)
    const u = [uptake(0, sg[0], n[0], mt[0]), uptake(1, sg[1], n[1], mt[1])];
    const a00 = 1 / dt + ex[0], a01 = -ex[0], a10 = -ex[1], a11 = 1 / dt + ex[1] + wd[1], c0 = x[0] / dt + src[0] - u[0], c1 = x[1] / dt - u[1], dd = a00 * a11 - a01 * a10;
    x[0] = (c0 * a11 - a01 * c1) / dd; x[1] = (a00 * c1 - a10 * c0) / dd;
    const m1 = mass(1), mAll = [mass(0), m1];
    acc.made += S * dt; acc.product += wd[1] * phi[1] * m1 * dt; acc.dissolvedOut += wd[1] * phi[1] * x[1] * dt; acc.wall += kw * (phi[0] * mAll[0] + phi[1] * m1) * dt;
    for (const z of [0, 1]) { const vN = vel(sg[z], N); if (vN > 0) acc.top += phi[z] * rk * vN * n[z][N - 1] * L3[N - 1] * dt; }
    if (k % Math.max(1, Math.round(nSteps / 120)) === 0 || k === nSteps - 1) { hist.th.push(time / tau); hist.s1.push(sigOf(0, x[0])); hist.s2.push(sigOf(1, x[1])); hist.mt.push(phi[0] * mAll[0] + phi[1] * m1); }
  }
  // product statistics (zone 2)
  const stats = (pop) => {
    const mu = [0, 1, 2, 3, 4, 5].map((j) => { let s = 0; for (let i = 0; i < N; i++) s += pop[i] * dL[i] * Lc[i] ** j; return s; }), cm = [0];
    for (let i = 0; i < N; i++) cm.push(cm[i] + pop[i] * dL[i] * L3[i]);
    const qn = (p) => { const tg = p * cm[N]; for (let i = 0; i < N; i++) if (cm[i + 1] >= tg) return e[i] + (dL[i] * (tg - cm[i])) / Math.max(cm[i + 1] - cm[i], 1e-300); return Lmax; };
    return { mu, L10: qn(0.1), L50: qn(0.5), L90: qn(0.9), L43: mu[3] > 0 ? mu[4] / mu[3] : 0, cv: mu[4] > 0 ? Math.sqrt(Math.max(0, (mu[5] * mu[3]) / (mu[4] * mu[4]) - 1)) : 0, mt: rk * mu[3] };
  };
  // ideal reference on the same grid: one perfectly mixed zone, no Gibbs–Thomson effect, no wall loss — the steady upwind solution in closed form
  const refPop = new Float64Array(N), refMass = (sg) => { const G = kgc * sg ** gex, B = kinetics(K, sg, Math.max(MT - sg * cs, 0)).B; let m = 0; for (let i = 0; i < N; i++) { refPop[i] = i === 0 ? B / (G + dL[0] / tau) : (refPop[i - 1] * G) / (G + dL[i] / tau); m += refPop[i] * dL[i] * L3[i]; } return rk * m; };
  const fRef = (sg) => refMass(sg) + sg * cs - MT, sTop = Math.min(3, (0.999 * MT) / cs), sRef = fRef(sTop) > 0 ? brent(fRef, 1e-12, sTop, 1e-14) : sTop; // σ = MT/c* (no crystals) is the trivial root and is excluded
  refMass(sRef);
  const reference = { sigma: sRef, G: kgc * sRef ** gex, pop: Array.from(refPop), ...stats(refPop) };
  const sig = [sigOf(0, x[0]), sigOf(1, x[1])], st = [stats(n[0]), stats(n[1])], ok = sig.every((q) => Number.isFinite(q) && q > -1 && q < 3.5) && Number.isFinite(st[1].mt), mEnd = phi[0] * (mass(0) + x[0]) + phi[1] * (mass(1) + x[1]);
  return { L: Array.from(Lc), dL: Array.from(dL), edges: Array.from(e), n: [Array.from(n[0]), Array.from(n[1])], sigma: sig, G: [K.kg * Math.max(sig[0], 0) ** K.g, K.kg * Math.max(sig[1], 0) ** K.g], Lcrit: sig.map((s) => (alpha > 0 && s > 0 ? (2 * alpha) / Math.log1p(s) : 0)), stats: st, product: st[1], reference, ok, hist, phi, theta, kw, alpha, nSteps, dt: dt1,
    wallRate: kw * (phi[0] * st[0].mt + phi[1] * st[1].mt), balance: { in: m0 + acc.made, out: mEnd + acc.product + acc.dissolvedOut + acc.wall + acc.top }, acc };
}

// ---- membrane distillation and electrodialysis as concentration steps ---------------------------------------
/**
 * Direct-contact membrane distillation over a sequence of liquor states. Flux J = B·(aw·psat(T_fm) − psat(T_pm)) with
 * the membrane-surface temperatures from the temperature-polarisation coefficient; heat = latent + conduction.
 * states = [{ w, aw }] path points (water inventory per basis), Kw converts the basis to kg/h. Returns area (m²) and heat (kW).
 */
export function mdUnit(states, Kw, { Tf, Tp, B, tpc, hm, recovery = 0 }) {
  const dTm = tpc * (Tf - Tp), Tfm = Tf - 0.5 * (1 - tpc) * (Tf - Tp), Tpm = Tp + 0.5 * (1 - tpc) * (Tf - Tp), lam = latentHeat(0.5 * (Tfm + Tpm));
  const flux = (aw) => B * (aw * psat(Tfm) - psat(Tpm)); // kg/m²/s
  let area = 0, heat = 0, mv = 0, jMin = Infinity, jMax = -Infinity, reversed = false;
  for (let i = 1; i < states.length; i++) {
    const d = Kw * (states[i - 1].w - states[i].w) / 3600; // kg/s of distillate in this step
    if (!(d > 0)) continue;
    const j0 = flux(0.5 * (states[i - 1].aw + states[i].aw)), j = Math.max(j0, 0.02 * B * psat(Tfm));
    if (j0 <= 0) reversed = true;
    jMin = Math.min(jMin, j0); jMax = Math.max(jMax, j0);
    area += d / j; heat += d * lam + (d / j) * hm * dTm; mv += d;
  }
  const eta = heat > 0 ? (mv * lam) / heat : 0;
  return { area, kWt: (heat / 1000) * (1 - recovery), eta, flux: mv > 0 ? (mv / area) * 3600 : 0, jMin: Number.isFinite(jMin) ? jMin * 3600 : 0, jMax: Number.isFinite(jMax) ? jMax * 3600 : 0, reversed, Tfm, Tpm, gor: heat > 0 ? (mv * lam) / (heat * (1 - recovery)) : 0, awStop: psat(Tpm) / psat(Tfm) };
}
/**
 * Electrodialysis concentrator: Faraday's law for the salt transferred, cell-pair voltage = ohmic drop + membrane
 * (concentration) potential, and the concentration ceiling set by electro-osmotic water transport.
 * eq = equivalents of salt transferred (mol/s); cd, cc = diluate and concentrate normality (eq/m³).
 */
export function edUnit({ eq, cd, cc, T, i, eta, rcp, tw, perm = 0.92 }) {
  const Umem = 2 * perm * ((R * (T + KELVIN)) / F_CONST) * Math.log(Math.max(cc / Math.max(cd, 1e-9), 1)), U = i * rcp + Umem, I = (F_CONST * eq) / eta;
  return { U, Umem, Uohm: i * rcp, current: I, kW: (I * U) / 1000, area: I / i, kWhPerKmol: (F_CONST * U) / eta / 3.6e6 * 1000, mMax: 1 / (tw * MW_W) };
}
const F_CONST = 96485.33212;

// ---- solids separation by cake filtration ------------------------------------------------------------------
/**
 * Filter area for a constant-pressure cake filtration (Ruth equation): t/V = μαc·V/(2A²ΔP) + μR_m/(AΔP).
 * V = filtrate volume per cycle (m³), c = dry solids per m³ of filtrate (kg/m³), alpha in m/kg, dP in Pa, t in s.
 */
export function ruthArea({ V, c, alpha, Rm = 1e11, dP, t, mu = 1e-3 }) {
  const qa = dP * t, qb = -mu * Rm * V, qc = -(mu * alpha * c * V * V) / 2;
  return (-qb + Math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa);
}
export const ruthTime = ({ V, c, alpha, Rm = 1e11, dP, A, mu = 1e-3 }) => (mu * alpha * c * V * V) / (2 * A * A * dP) + (mu * Rm * V) / (A * dP);
/** Capital-recovery factor for interest rate i (fraction) and n years. */
export const crf = (i, n) => (i > 1e-9 ? (i * (1 + i) ** n) / ((1 + i) ** n - 1) : 1 / n);

// ---- pretreatment: selective precipitation ------------------------------------------------------------
function soften(feed, v, W) {
  const out = { sol: feed, stages: [], reagents: {} };
  if (v.softMode === 'none') return out;
  const use = (r, mol) => { if (mol > 0) out.reagents[r] = (out.reagents[r] || 0) + mol; };
  // stage 1: alkali to the target pH — magnesium hydroxide (with any co-precipitating CaCO3)
  const set1 = ['brucite', 'calcite', 'gypsum', 'portlandite'], rA = REAGENTS[v.alkali];
  const at = (mol) => precipitateSolution(mol > 0 ? doseSolution(feed, v.alkali, mol) : feed, set1);
  const top = (2.6 * feed.n[iOf.Mg] + 2 * feed.n[iOf.C] + Math.abs(feed.alk) + 0.02 * feed.w) / feed.w / rA.alk, g = (mol) => at(mol).sol.pH - v.softPH;
  let mol1 = 0;
  if (g(0) < 0) { if (g(top) < 0) { mol1 = top; W.push({ level: 'warn', msg: `The target pH ${v.softPH} is not reached with a practical ${rA.name} dose.` }); } else mol1 = brent(g, 0, top, 1e-7 * top); }
  const r1 = at(mol1);
  use(v.alkali, mol1 * feed.w);
  out.stages.push({ name: 'Magnesium hydroxide stage', pH: r1.sol.pH, solids: r1.solids });
  let cur = r1.sol;
  if (v.softMode === 'mgca' && v.sodaRatio > 0 && cur.n[iOf.Ca] > 0) { // stage 2: soda ash for the remaining calcium
    const mol2 = (v.sodaRatio * cur.n[iOf.Ca]) / cur.w, r2 = precipitateSolution(doseSolution(cur, 'soda', mol2), ['calcite', 'brucite', 'gypsum']);
    use('soda', mol2 * cur.w); out.stages.push({ name: 'Calcium carbonate stage', pH: r2.sol.pH, solids: r2.solids }); cur = r2.sol;
  }
  if (cur.pH > v.softNeutral) { // neutralise with hydrochloric acid before the membranes and evaporator
    let tot = 0;
    for (let it = 0; it < 4; it++) { const d = (cur.alk - equilibrate(cur, { pH: v.softNeutral }).alk) / cur.w; if (!(d > 1e-12)) break; tot += d; cur = doseSolution(cur, 'hcl', d); }
    use('hcl', tot * cur.w);
  }
  out.sol = cur;
  return out;
}

/** Compact number formatting for notes (significant digits, no locale lookup). */
const fq = (x, sig = 4) => (typeof x !== 'number' ? String(x ?? '–') : !Number.isFinite(x) ? '–' : x === 0 ? '0' : Math.abs(x) >= 1e7 || Math.abs(x) < 1e-4 ? x.toExponential(Math.max(1, sig - 1)) : String(+x.toPrecision(sig)));
/** Mass-based size quantile of Lagrangian classes [{ N, r }] (returns a diameter). */
function classQuantile(cl, p) {
  const s = cl.map((c) => [2 * c.r, c.N * c.r ** 3]).sort((a, b) => a[0] - b[0]), tot = sum(s.map((q) => q[1]));
  let acc = 0;
  for (const [L, m] of s) { acc += m; if (acc >= p * tot) return L; }
  return s.length ? s[s.length - 1][0] : 0;
}
/**
 * Crystallizer physics beyond the ideal MSMPR: loop hydrodynamics and heat balance (turnover time, tube Reynolds
 * number, mass-transfer coefficient, pressure drop), the two-zone population balance with Gibbs–Thomson growth and
 * dissolution and the wall boundary condition, Ostwald ripening in the slurry hold tank, and wall-deposit dissolution.
 */
export function crystallizerPhysics(v, r) {
  const c = r.csd;
  if (!c) return null;
  const M = MINERALS[c.salt] || MINERALS.halite, T = v.Tcx, vm = M.mw / 1000 / M.rho, alpha = capillaryLength(M.sigma * 1e-3, vm, T, M._nu), cInf = c.cstar / (M.mw / 1000), Klsw = lswConstant(D_SALT, cInf, vm, alpha), A = (9 / 4) * Klsw;
  // circulation loop: the heat duty and the temperature rise per pass fix the circulation flow, hence the turnover time
  const lam = latentHeat(T), Qheat = (r.cx.mv / 3600) * lam, phiS = c.solidVol, rhoSl = r.end.rho * (1 - phiS) + c.MT, cpSl = cpBrine(T, Math.min(r.end.S, 300)) * (1 - c.MT / rhoSl) + 850 * (c.MT / rhoSl);
  const dTp = Math.max(v.dTpass ?? 3, 0.2), Qc = Qheat / (rhoSl * cpSl * dTp), theta = Qc > 0 ? c.volume / Qc : 1e9, dTube = Math.max(v.dTube ?? 38, 5) * 1e-3, uT = Math.max(v.uTube ?? 2, 0.1);
  const muL = viscosity(T, Math.min(r.end.S, 300)), muSl = muL * (1 + 2.5 * phiS + 10.05 * phiS * phiS + 0.00273 * Math.exp(16.6 * phiS)), Re = (rhoSl * uT * dTube) / muSl, Sc = muSl / (r.end.rho * D_SALT), Sh = 0.023 * Re ** 0.8 * Sc ** (1 / 3), kL = (Sh * D_SALT) / dTube;
  const nTubes = Math.max(1, Qc / (uT * Math.PI * dTube * dTube / 4)), Ltube = r.cx.md ? 6 : r.cx.area / (nTubes * Math.PI * dTube), fD = 0.316 / Re ** 0.25, dP = (fD * (Ltube / dTube) + 4) * 0.5 * rhoSl * uT * uT, pumpKW = (Qc * dP) / 0.7 / 1000;
  const dlnc = (LN10 / M._nu) * (M.logK(T + 1) - M.logK(T - 1)) / 2, dSig2 = dlnc * dTp / 2, flash = (cpSl * dTp) / lam;
  // wall geometry: vessel (height = 2 × diameter) plus the heater surface
  const Dv = Math.cbrt((2 * c.volume) / Math.PI), aWall = (2.5 * Math.PI * Dv * Dv + (r.cx.md ? 0 : r.cx.area)) / c.volume, deposit = v.wallBC === 'deposit', kw = deposit ? Math.max(v.kDep ?? 0.005, 0) * 1e-6 * aWall : 0;
  const seed = { mass: Math.max(1e-6, v.seedMass), L: v.seedL * 1e-6, sigma: v.sigma0 / 100 }, Lmax = Math.max(v.lmaxMult * c.Gend * c.tau, 2 * seed.L);
  const zone = v.zoneOn === false ? null : zoneCrystallizer(c.K, c.tau, c.MT, seed, { phi1: (v.phi1 ?? 15) / 100, theta, alpha, gt: v.gtOn !== false, kw, dSig2, nL: Math.max(100, Math.round(1.25 * v.nL)), Lmax: 3 * Lmax, tEnd: v.tEnd, nt: 20 });
  // wall deposit: crystals captured by the wall plus growth of the deposit itself at the bulk supersaturation; removed by a boil-out
  const zok = zone && zone.ok, sigB = zok ? zone.sigma[1] : c.sigEnd, rateCap = zok && aWall > 0 ? zone.wallRate / aWall : deposit ? Math.max(v.kDep ?? 0.005, 0) * 1e-6 * c.MT : 0, wallRate = rateCap; // kg/m²/s
  const days = Math.max(v.runDays ?? 30, 0), thick = (wallRate * 86400 * days) / (M.rho * 0.8), tBoil = wallRate > 0 ? (thick * M.rho * 0.8) / (kL * c.cstar) : 0;
  // dissolution of fines: Gibbs–Thomson critical size at the product-zone supersaturation and Noyes–Whitney time in the heated loop
  const Lcrit = sigB > 0 ? (2 * alpha) / Math.log1p(sigB) : Infinity, Lf = Math.max(v.finesL ?? 10, 0.1) * 1e-6, under = Math.max(sigmaCrit(Lf, alpha) - (sigB - dSig2 * 2), 0), vDis = dissolutionVelocity(Lf, under, c.cstar, M.rho), tFines = vDis > 0 ? Lf / vDis : Infinity, tLoop = theta * (1 - (v.phi1 ?? 15) / 100);
  // Ostwald ripening of the product slurry in the hold tank
  const src = zok ? zone.L.map((L, i) => ({ N: zone.n[1][i] * zone.dL[i], r: L / 2 })) : c.fv.L.map((L, i) => ({ N: c.fv.n[i] * c.fv.dL, r: L / 2 })), cl0 = src.filter((q) => q.N > 1e-200 && q.r > 0), tHold = Math.max(v.tHold ?? 2, 0) * 3600;
  const rip = cl0.length > 1 && tHold > 0 ? ripen(cl0, A, tHold, { maxSteps: 4000, nOut: 30 }) : null, n0 = sum(cl0.map((q) => q.N));
  const hold = rip ? { n0, n1: sum(rip.classes.map((q) => q.N)), L10a: classQuantile(cl0, 0.1), L10b: classQuantile(rip.classes, 0.1), L50a: classQuantile(cl0, 0.5), L50b: classQuantile(rip.classes, 0.5), rMean0: rip.rMean[0], rMean1: rip.rMean[rip.rMean.length - 1], complete: rip.complete } : null;
  return { M, alpha, cInf, Klsw, A, theta, Qc, Re, Sc, Sh, kL, nTubes, Ltube, dP, pumpKW, dlnc, dSig2, flash, aWall, Dv, kw, zone, wallRate, rateCap, thick, tBoil, Lcrit, Lf, tFines, tLoop, under, rip, hold, t1pct: Klsw > 0 ? (0.0303 * (0.5 * c.fv.L50) ** 3) / Klsw : Infinity, muSl, rhoSl };
}

/** Filter-press sizing (Ruth cake filtration) for the softening sludge and the separated seed-slurry solids. */
export function solidsFiltration(v, r) {
  const streams = [];
  for (const st of r.softSolids) { const kg = sum(Object.values(st.kgh)); if (kg > 1e-9) streams.push({ name: st.name, kg, rho: sum(Object.entries(st.kgh).map(([id, x]) => x * MINERALS[id].rho)) / kg }); }
  if (v.bcSolids !== 'cake') { const kg = sum(Object.values(r.bcSolids)); if (kg > 1e-9) streams.push({ name: 'Seed-slurry solids of the concentrator', kg, rho: sum(Object.entries(r.bcSolids).map(([id, x]) => x * MINERALS[id].rho)) / kg }); }
  const x = clamp((v.filtFeed ?? 8) / 100, 0.005, 0.5), mc = clamp((v.filtMoist ?? 35) / 100, 0.05, 0.8), tf = Math.max(v.filtTime ?? 30, 1) * 60, td = 1800, dP = Math.max(v.filtDP ?? 6, 0.2) * 1e5, alpha = Math.max(v.filtAlpha ?? 5e11, 1e8), rhoL = 1100;
  return streams.map((q) => {
    const liq = (1 - x) / x - mc / (1 - mc), cF = rhoL / Math.max(liq, 0.05), V = ((q.kg * (tf + td)) / 3600) / cF, area = ruthArea({ V, c: cF, alpha, dP, t: tf }), eps = mc / rhoL / (mc / rhoL + (1 - mc) / q.rho);
    return { ...q, c: cF, V, area, thick: (cF * V) / area / (q.rho * (1 - eps)), tCheck: ruthTime({ V, c: cF, alpha, dP, A: area }), tf, liquor: (q.kg * mc) / (1 - mc), cycles: 86400 / (tf + td) };
  });
}

/** Capital and operating cost of the ZLD train (order-of-magnitude correlations with a capacity exponent). */
export function zldEconomics(v, r, day) {
  const sc = (unit, cap) => (cap > 0 ? unit * 1000 * (cap / 1000) ** clamp(v.capexScale ?? 0.7, 0.3, 1) : 0), m3d = (kgh) => (kgh * 24) / RHO_W, items = [];
  const add = (name, cost, basis) => { if (cost > 0) items.push({ name, cost, basis }); };
  if (v.softMode !== 'none') add('Softening and clarification', sc(v.capexSoft ?? 400, v.Q * 24), `${fq(v.Q * 24, 3)} m³/d of brine`);
  if (r.ro.on) add(v.preconc === 'ed' ? 'Electrodialysis stacks' : 'Membrane pre-concentration', v.preconc === 'ed' ? (v.capexED ?? 450) * r.ro.ed.area + sc(600, r.ro.perm * 24) : sc((v.capexMem ?? 1800) * (v.preconc === 'oaro' ? 1.6 : 1), r.ro.perm * 24), v.preconc === 'ed' ? `${fq(r.ro.ed.area, 3)} m² of cell pairs + polishing RO` : `${fq(r.ro.perm * 24, 3)} m³/d of permeate`);
  if (r.bc.on) add(v.bcType === 'mdc' ? 'Membrane-distillation concentrator' : 'Brine concentrator', v.bcType === 'mdc' ? (v.capexMD ?? 250) * r.bc.area + sc(1500, m3d(r.bc.mv)) : sc(v.capexBC ?? 6000, m3d(r.bc.mv)), `${fq(m3d(r.bc.mv), 3)} m³/d of distillate`);
  if (r.cx.on) add(v.cxDrive === 'md' ? 'Membrane-distillation crystallizer' : 'Crystallizer', v.cxDrive === 'md' ? (v.capexMD ?? 250) * r.cx.area + sc(4000, m3d(r.cx.mv)) : sc(v.capexCX ?? 14000, m3d(r.cx.mv)), `${fq(m3d(r.cx.mv), 3)} m³/d of condensate`);
  add('Solids handling (centrifuge, filter, dryer)', (v.capexSolids ?? 30000) * ((r.solidsTotal * 24) / 1000), `${fq((r.solidsTotal * 24) / 1000, 3)} t/d of solids`);
  if (r.pond.area > 0) add('Evaporation pond', (v.capexPond ?? 60) * r.pond.area, `${fq(r.pond.area / 1e4, 3)} ha lined`);
  const direct = sum(items.map((q) => q.cost)), capex = direct * (1 + (v.capexIndirect ?? 35) / 100), f = crf((v.rate ?? 8) / 100, Math.max(1, v.life ?? 20)), av = clamp((v.avail ?? 92) / 100, 0.1, 1), daysY = 365 * av;
  const memCap = sum(items.filter((q) => /Membrane|Electrodialysis/.test(q.name)).map((q) => q.cost));
  const opex = [['Electricity', 24 * r.kWe * v.elecPrice * daysY], ['Heat', (24 * r.kWt * v.steamPrice * daysY) / 1000], ['Reagents', day.reagCost * daysY], ['Solids disposal less salt revenue', -day.rev * daysY], ['Maintenance', ((v.maint ?? 3) / 100) * capex], ['Membrane replacement', 0.15 * memCap], ['Labour', v.labour ?? 400000]];
  const opexY = sum(opex.map((q) => q[1])), annual = f * capex + opexY, brineY = v.Q * 24 * daysY, waterY = r.recovered * 24 * daysY;
  return { items, direct, capex, crf: f, opex, opexY, annual, capexY: f * capex, perBrine: annual / Math.max(brineY, 1e-9), perWater: waterY > 0 ? annual / waterY : 0, brineY, waterY };
}

const D = () => Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));

/** Complete ZLD train. All flows in kg/h or m³/h; solids in kg/h. */
export function simulateZLD(v) {
  const W = [], model = ACTIVITY_MODELS[v.model] ? v.model : 'pitzer';
  const feed = makeSolution({ ions: v.ions, T: v.T, pH: v.pH, model }), io0 = solutionToIons(feed), Kw = (v.Q * 1000 * io0.kgwPerL) / feed.w; // kg/h of water per unit of basis water
  const minerals = v.salts === 'simple' ? SIMPLE_SET : EVAPORITE_MINERALS;
  const soft = soften(feed, v, W), s1 = soft.sol, g1 = solutionToIons(s1).gPerKgw * s1.w;
  const wFor = (S) => (g1 * (1000 / clamp(S, 1, 900) - 1)) / 1000; // water inventory at which the dissolved salts reach S g/kg (before precipitation)
  const wRO = v.preconc === 'none' ? s1.w : Math.min(s1.w, wFor(v.roTDS)), wBC = Math.min(wRO, wFor(v.bcTDS)), wCX = wBC * clamp(v.purge / 100, 0.002, 1);
  const Tc = (T) => (v.chemT === 'unit' ? T : 25), pCO2 = v.pCO2 * 1e-6, mdc = v.bcType === 'mdc', Tb = mdc ? clamp(v.mdTf ?? 70, 30, 95) : v.Tbc, mdPar = (Tf) => ({ Tf, Tp: Math.min(v.mdTp ?? 25, Tf - 5), B: Math.max(v.mdB ?? 0.7, 1e-3) / 3.6e6, tpc: clamp(v.mdTPC ?? 0.6, 0.1, 1), hm: Math.max(v.mdCond ?? 250, 0), recovery: clamp((v.mdHR ?? 50) / 100, 0, 0.95) });
  const segOf = (u) => path.steps.filter((q, i) => q.unit === u || path.steps[i + 1]?.unit === u);
  const path = evaporationPath(s1, [{ unit: 'ro', wEnd: wRO, T: Tc(v.T), precip: false }, { unit: 'bc', wEnd: wBC, T: Tc(Tb), precip: true, pCO2 }, { unit: 'cx', wEnd: wCX, T: Tc(v.Tcx), precip: true, pCO2 }], { nEvap: Math.max(4, Math.round(v.nEvap)), minerals });
  if (path.stopped) W.push({ level: 'warn', msg: `Evaporation path: ${path.stopped}. The remaining liquor is treated as purge.` });
  const last = (u) => { for (let i = path.steps.length - 1; i >= 0; i--) if (path.steps[i].unit === u) return path.steps[i]; return null; };
  const first = (u) => { const i = path.steps.findIndex((s) => s.unit === u); return i > 0 ? path.steps[i - 1] : null; };
  const kgh = (obj) => Object.fromEntries(Object.entries(obj || {}).map(([id, mol]) => [id, (Math.max(0, mol) * MINERALS[id].mw * Kw) / 1000]));
  const vol = (st) => (Kw * st.w * (1 + st.gPerKgw / 1000)) / st.rho; // m³/h of brine at a path state
  const etaM = v.etaMotor / 100, units = [];
  const st0 = path.steps[0];

  // --- membrane pre-concentration
  const ro = { on: (path.evap.ro || 0) > 0, perm: (Kw * (path.evap.ro || 0)) / RHO_W, kW: 0, P: 0, wLeast: 0, pi: st0 ? osmoticBar(v.T, st0.aw) : 0 };
  if (ro.on) {
    const out = last('ro'), seg = path.steps.filter((s, i) => s.unit === 'ro' || path.steps[i + 1]?.unit === 'ro');
    for (let i = 1; i < seg.length; i++) ro.wLeast += (0.5 * (osmoticBar(v.T, seg[i].aw) + osmoticBar(v.T, seg[i - 1].aw)) * (Kw * (seg[i - 1].w - seg[i].w))) / RHO_W / 36; // kW, reversible
    ro.pi = osmoticBar(v.T, out.aw);
    if (v.preconc === 'hpro') {
      ro.P = 1.08 * ro.pi + 2;
      ro.kW = (vol(st0) * ro.P - (v.erdEff / 100) * vol(out) * (ro.P - 2)) / 36 / (v.etaPump / 100) / etaM;
      if (ro.P > v.pMax) W.push({ level: 'bad', msg: `High-pressure RO would need ${fmt(ro.P, 3)} bar to reach ${v.roTDS} g/kg (osmotic pressure ${fmt(ro.pi, 3)} bar), above the ${v.pMax} bar rating — lower the target or use osmotically assisted RO.` });
    } else if (v.preconc === 'ed') {
      // electrodialysis: all dissolved salt is transferred to the concentrate (the diluate is polished by RO and its reject returns)
      const eqh = Kw * st0.w * (st0.m.Na + st0.m.K + 2 * st0.m.Ca + 2 * st0.m.Mg), ed = edUnit({ eq: eqh / 3600, cd: ((v.edDil ?? 3) / 58.443) * 1000, cc: eqh / vol(out), T: v.T, i: Math.max(v.edI ?? 300, 1), eta: clamp((v.edEta ?? 90) / 100, 0.05, 1), rcp: Math.max(v.edRcp ?? 25, 0.1) * 1e-4, tw: Math.max(v.edTw ?? 12, 0.5) });
      ro.P = 0; ro.ed = ed; ro.edMolality = eqh / (Kw * out.w); ro.kW = ed.kW / 0.95 + ro.perm * (v.edRoSEC ?? 1.5) + 0.15 * vol(st0);
      if (ro.edMolality > ed.mMax) W.push({ level: 'bad', msg: `Electrodialysis cannot reach ${v.roTDS} g/kg: with ${v.edTw} mol of water carried per equivalent of salt the concentrate cannot exceed ${fmt(ed.mMax, 3)} eq/kg (requested ${fmt(ro.edMolality, 3)}) — lower the target or use membranes with less water transport.` });
    } else { ro.P = v.oaroP; ro.kW = ro.wLeast / (v.eta2 / 100); }
    for (const [id, lim] of [['gypsum', 0.36], ['anhydrite', 0.36], ['silica', 0.18], ['calcite', 1.8], ['barite', 1.78]]) if (out.SI[id] > lim) W.push({ level: 'warn', msg: `${MINERALS[id].name} reaches SI ${fmt(out.SI[id], 3)} in the membrane concentrate, beyond what antiscalant normally controls (${lim}) — soften first or lower the membrane target.` });
    units.push({ name: { hpro: 'High-pressure RO', ed: 'Electrodialysis concentrator' }[v.preconc] || 'Osmotically assisted RO', feedW: Kw * st0.w, removed: Kw * path.evap.ro, out, T: v.T, bpe: null, kWe: ro.kW, kWt: 0, area: ro.ed ? ro.ed.area : null, solids: 0 });
  }

  // --- brine concentrator
  const bc = { on: (path.evap.bc || 0) > 0, mv: Kw * (path.evap.bc || 0), kWe: 0, kWt: 0, area: 0, bpe: 0, comp: 0, aux: 0, vent: 0, econ: 0, steam: 0, balance: null };
  const bcSolids = kgh(path.solids.bc);
  if (bc.on) {
    const inn = first('bc'), out = last('bc'), lam = latentHeat(Tb), Qlat = (bc.mv / 3600) * lam; // W
    bc.bpe = bpeFromAw(Tb, out.aw);
    if (out.SI.halite > -0.02 || bcSolids.halite > 0) W.push({ level: 'warn', msg: 'Sodium chloride starts to crystallise inside the brine concentrator — lower its outlet salinity so that halite forms only in the crystallizer.' });
    if (v.bcType === 'mvc') {
      const ws = compressorWork(v.Tbc, bc.bpe, v.dT, v.etaComp / 100);
      bc.comp = (bc.mv * ws) / 3.6e6; bc.kWe = bc.comp / etaM + (v.bcPumps * bc.mv) / 1000; bc.area = Qlat / (v.U * 1000 * v.dT);
      // energy balance with a feed/distillate preheater: auxiliary heat closes the balance (or surplus is vented)
      const mf = Kw * inn.w * (1 + inn.gPerKgw / 1000), mb = Kw * out.w * (1 + out.gPerKgw / 1000), md = bc.mv, ms = sum(Object.values(bcSolids));
      const Tin = ro.on ? v.T : v.T, Hin = (mf * hL(Tin, inn.S)) / 3.6e6, Hd = (md * hL(Math.min(v.Tbc, Tin + v.approach), 0)) / 3.6e6, Hb = ((mb + ms) * hL(v.Tbc, out.S)) / 3.6e6, loss = ((v.heatLoss / 100) * Qlat) / 1000;
      const need = Hd + Hb + loss - Hin - bc.comp;
      bc.aux = Math.max(0, need); bc.vent = Math.max(0, -need); bc.kWt = bc.aux;
      bc.balance = { in: Hin + bc.comp + bc.aux, out: Hd + Hb + loss + bc.vent, hydration: (mf - md - mb - ms) };
    } else if (mdc) {
      // membrane-distillation concentrator: area and heat from the vapour-pressure driving force along the concentration path
      const md = mdUnit(segOf('bc'), Kw, mdPar(Tb));
      bc.md = md; bc.area = md.area; bc.kWt = md.kWt; bc.kWe = ((v.mdPump ?? 2) * bc.mv) / 1000;
      if (md.reversed) W.push({ level: 'bad', msg: `Membrane distillation stalls before ${v.bcTDS} g/kg: the vapour pressure of the brine at ${fmt(md.Tfm, 3)} °C falls below that of the permeate at ${fmt(md.Tpm, 3)} °C when the water activity drops under ${fmt(md.awStop, 3)} — raise the feed temperature or lower the outlet salinity.` });
    } else {
      const N = Math.max(1, Math.round(v.nEff));
      bc.econ = 0.85 * N * (v.bcType === 'tvc' ? 1 + v.tvcRa : 1);
      bc.kWt = Qlat / 1000 / bc.econ; bc.steam = (bc.kWt * 3.6) / (latentHeat(v.steamT) / 1000); // t/h
      const dTe = (v.Tbc - v.Tlast) / N - bc.bpe;
      if (dTe < 1.5) W.push({ level: 'bad', msg: `Only ${fmt(dTe, 2)} K of driving temperature difference is left per effect after the boiling-point elevation (${fmt(bc.bpe, 3)} K) — use fewer effects, a higher top temperature, or MVC.` });
      bc.area = Qlat / (v.U * 1000 * Math.max(dTe, 0.5)); bc.kWe = (v.bcPumps * bc.mv) / 1000;
    }
    units.push({ name: { mvc: 'Brine concentrator (falling-film MVC)', mee: `Brine concentrator (${Math.round(v.nEff)}-effect MEE)`, tvc: `Brine concentrator (${Math.round(v.nEff)}-effect MEE-TVC)`, mdc: 'Membrane-distillation concentrator (direct contact)' }[v.bcType], feedW: Kw * inn.w, removed: bc.mv, out, T: Tb, bpe: bc.bpe, kWe: bc.kWe, kWt: bc.kWt, area: bc.area, solids: sum(Object.values(bcSolids)) });
  }

  // --- crystallizer, centrifuge, dryer and purge
  const end = path.steps[path.steps.length - 1], cxOn = (path.evap.cx || 0) > 0, cxSolids = kgh(path.solids.cx);
  const cakeSolids = { ...cxSolids };
  if (v.bcSolids === 'cake') for (const [id, x] of Object.entries(bcSolids)) cakeSolids[id] = (cakeSolids[id] || 0) + x;
  const Sdry = sum(Object.values(cakeSolids)), main = Object.keys(cakeSolids).reduce((a, k) => (cakeSolids[k] > (cakeSolids[a] || 0) ? k : a), 'halite');
  const mlMass = Kw * end.w * (1 + end.gPerKgw / 1000), xs = end.gPerKgw / (1000 + end.gPerKgw), xm = clamp(v.cakeMoist / 100, 0, 0.6);
  const adher = cxOn ? Math.min(mlMass, (Sdry * xm) / (1 - xm)) : 0, washE = adher > 0 && v.wash > 0 ? 1 - Math.exp((-v.wash * Sdry) / adher) : 0;
  const cakeSalt = adher * xs * (1 - washE), purgeMass = mlMass - adher, purgeSalt = purgeMass * xs + adher * xs * washE, purgeQ = purgeMass / end.rho, washW = cxOn ? v.wash * Sdry : 0;
  const cx = { on: cxOn, mv: Kw * (path.evap.cx || 0) + washW, bpe: 0, kWe: 0, kWt: 0, area: 0 };
  if (cxOn) {
    const lam = latentHeat(v.Tcx), Qlat = (cx.mv / 3600) * lam;
    cx.bpe = bpeFromAw(v.Tcx, end.aw);
    if (v.cxDrive === 'mvr') { cx.kWe = (cx.mv * compressorWork(v.Tcx, cx.bpe, v.dTcx, v.etaComp / 100)) / 3.6e6 / etaM; if (cx.bpe > v.bpeMax) W.push({ level: 'bad', msg: `Crystallizer boiling-point elevation ${fmt(cx.bpe, 3)} K exceeds the ${v.bpeMax} K a vapour compressor can economically overcome — use steam drive or purge more mother liquor.` }); }
    else if (v.cxDrive === 'md') {
      // membrane-distillation crystallizer: the membrane modules remove the water, crystals form in the circulated tank
      const md = mdUnit(segOf('cx'), Kw, mdPar(clamp(v.Tcx, 30, 95))), sc = (path.evap.cx || 0) > 0 ? cx.mv / (Kw * path.evap.cx) : 1;
      cx.md = md; cx.kWt = md.kWt * sc; cx.mdArea = md.area * sc;
      if (v.Tcx > 90) W.push({ level: 'warn', msg: 'Membrane-distillation modules are limited to about 90 °C; the crystallizer temperature is above that.' });
      if (md.reversed) W.push({ level: 'bad', msg: `The membrane-distillation crystallizer stalls: at a water activity below ${fmt(md.awStop, 3)} the mother liquor at ${fmt(md.Tfm, 3)} °C no longer has a higher vapour pressure than the permeate at ${fmt(md.Tpm, 3)} °C — raise the temperature, cool the permeate or purge more.` });
    } else cx.kWt = Qlat / 1000 / 0.92;
    cx.kWe += (v.cxPumps * cx.mv) / 1000 + (3 * Sdry) / 1000; // recirculation pump and centrifuge
    cx.area = cx.md ? cx.mdArea : Qlat / (v.Ufc * 1000 * v.dTcx);
    units.push({ name: v.cxDrive === 'md' ? 'Membrane-distillation crystallizer' : `Forced-circulation crystallizer (${v.cxDrive === 'mvr' ? 'MVR' : 'steam'})`, feedW: Kw * first('cx').w, removed: cx.mv, out: end, T: v.Tcx, bpe: cx.bpe, kWe: cx.kWe, kWt: cx.kWt, area: cx.area, solids: Sdry });
  }
  const dryW = adher * (1 - xs), dryer = { kWt: (dryW * (latentHeat(100) + 4180 * Math.max(0, 100 - v.Tcx))) / 3.6e6 / (v.etaDry / 100) };
  const purgeDry = v.purgeFate === 'dryer' ? { kWt: (purgeMass * (1 - xs) * latentHeat(100)) / 3.6e6 / (v.etaDry / 100) } : { kWt: 0 };
  const purity = Sdry > 0 ? (cakeSolids[main] || 0) / (Sdry + cakeSalt) : 0;
  // evaporation pond for the purge
  const pond = { area: 0, ok: true };
  if (v.purgeFate === 'pond' && purgeQ > 0) {
    const f = clamp((end.aw - v.pondRH / 100) / (1 - v.pondRH / 100), 0, 1), net = (v.pondEvap * f - v.pondRain) / 1000; // m/y
    pond.ok = net > 0.05; pond.area = pond.ok ? (purgeQ * 8760) / net : 0;
    if (!pond.ok) W.push({ level: 'bad', msg: `The purge (water activity ${fmt(end.aw, 3)}) cannot evaporate in a pond at ${v.pondRH} % relative humidity and ${v.pondEvap} mm/y pan evaporation — dry or solidify it instead.` });
  }
  const liquid = v.purgeFate === 'discharge' ? purgeQ : 0;

  // --- crystal-size distribution of the main product
  const prod = cakeSolids[main] || 0;
  let csd = null;
  if (cxOn && cxSolids[main] > 0) {
    const M = MINERALS[main], lim = Math.min(...M._st.map(([i, nu]) => end.m[Object.keys(iOf).find((k) => iOf[k] === i)] != null ? end.m[Object.keys(iOf).find((k) => iOf[k] === i)] / nu : Infinity));
    const cstar = Number.isFinite(lim) && lim > 0 ? Math.max(20, (lim * M.mw * end.rho) / (1000 + end.gPerKgw)) : 300; // kg of the salt per m³ of liquor at saturation
    const K = kin(v, main, cstar), tau = v.tau * 3600, MT = v.MT, ss = msmprSteady(K, tau, MT), seed = { mass: Math.max(1e-6, v.seedMass), L: v.seedL * 1e-6, sigma: v.sigma0 / 100 };
    const dyn = msmprDynamic(K, tau, MT, seed, v.tEnd), Gend = dyn.G[dyn.G.length - 1];
    const fv = pbeFV(K, tau, dyn, seed, { nL: v.nL, Lmax: Math.max(v.lmaxMult * Gend * tau, 2 * seed.L), scheme: v.scheme, tEnd: v.tEnd });
    if (!dyn.complete) W.push({ level: 'warn', msg: 'The crystallizer start-up integration stopped before the requested time (very stiff kinetics); the size distribution may not be at steady state.' });
    csd = { K, tau, MT, ss, dyn, fv, salt: main, cstar, volume: (cxSolids[main] * v.tau) / MT, slurryQ: cxSolids[main] / MT, solidVol: MT / K.rhoc, Gend, Bend: dyn.B[dyn.B.length - 1], sigEnd: dyn.sigma[dyn.sigma.length - 1] };
    if (csd.solidVol > 0.35) W.push({ level: 'warn', msg: `Magma density ${v.MT} kg/m³ is ${fmt(100 * csd.solidVol, 3)} % solids by volume — above about 30 % the slurry becomes difficult to circulate.` });
  }

  // --- overall balances
  const hyd = (obj) => sum(Object.entries(obj || {}).map(([id, mol]) => Math.max(0, mol) * MINERALS[id].nW * MW_W * Kw));
  const softSolids = soft.stages.map((s) => ({ ...s, kgh: kgh(s.solids) }));
  const water = { in: Kw * feed.w, perm: Kw * (path.evap.ro || 0), dist: bc.mv, cond: Kw * (path.evap.cx || 0), hydration: hyd(path.cum) + sum(soft.stages.map((s) => hyd(s.solids))), cake: dryW, purge: purgeMass * (1 - xs) };
  const recovered = (water.perm + water.dist + water.cond) / RHO_W;
  const kWe = ro.kW + bc.kWe + cx.kWe, kWt = bc.kWt + cx.kWt + dryer.kWt + purgeDry.kWt;
  const reagents = Object.fromEntries(Object.entries(soft.reagents).map(([r, mol]) => [r, (mol * REAGENTS[r].mw * Kw) / 1000])); // kg/h
  // salts by destination (kg/h)
  const salts = [];
  const tiny = 2e-5 * ((Kw * feed.w * io0.gPerKgw) / 1000); // traces below 0.002 % of the dissolved feed salts are not listed
  const push = (id, kg, unit, dest, value) => { if (kg > tiny) salts.push({ id, name: MINERALS[id]?.name || id, formula: MINERALS[id]?.formula || '', kg, unit, dest, value, onset: path.onset[id]?.cf ?? null }); };
  softSolids.forEach((s, i) => { const tot = sum(Object.values(s.kgh)) || 1; for (const [id, kg] of Object.entries(s.kgh)) push(id, kg, s.name, i === 0 && id === 'brucite' ? `Mg(OH)₂ product (${fmt((100 * kg) / tot, 3)} % of stage solids)` : id === 'calcite' ? 'CaCO₃ by-product' : 'Softening sludge', id === 'brucite' ? v.priceMg : id === 'calcite' ? 15 : -v.disposal); });
  if (v.bcSolids !== 'cake') for (const [id, kg] of Object.entries(bcSolids)) push(id, kg, 'Brine concentrator', 'Seed-slurry solids, separated', id === 'gypsum' ? 8 : -v.disposal);
  for (const [id, kg] of Object.entries(cakeSolids)) push(id, kg, 'Crystallizer', id === main ? `Product salt (cake purity ${fmt(100 * purity, 4)} %)` : 'Co-crystallised in the product cake', id === main && id === 'halite' ? (purity >= v.purityMin / 100 ? v.priceNaCl : 0.25 * v.priceNaCl) : 0);
  const bittern = cakeSalt + (v.purgeFate === 'discharge' ? 0 : purgeSalt);
  if (cakeSalt > 1e-9) salts.push({ id: 'ml', name: 'Mother-liquor salts in cake moisture', formula: 'mixed', kg: cakeSalt, unit: 'Centrifuge / dryer', dest: 'Impurity in the product cake', value: 0, onset: null });
  if (v.purgeFate !== 'discharge' && purgeSalt > 1e-9) salts.push({ id: 'bittern', name: 'Bittern solids from the purge', formula: 'Mg/K/Ca chlorides and sulphates', kg: purgeSalt, unit: v.purgeFate === 'pond' ? 'Evaporation pond' : 'Purge dryer', dest: 'Mixed solids for disposal', value: -v.disposal, onset: null });
  const solidsTotal = sum(salts.map((s) => s.kg));
  return { v, W, feed, io0, Kw, soft, softSolids, s1, path, units, ro, bc, cx, dryer, purgeDry, end, cakeSolids, bcSolids, cxSolids, Sdry, main, purity, adher, washE, cakeSalt, purgeMass, purgeSalt, purgeQ, liquid, pond, csd, water, recovered, kWe, kWt, reagents, salts, solidsTotal, bittern, prod, minerals, wRO, wBC, wCX, xs, Tb };
}

const suite = {
  id: 'zld', num: 9, title: 'Brine Concentration, Crystallization & ZLD', short: 'ZLD & salts', icon: '🧂',
  tagline: 'From brine to distilled water and dry salts: evaporation path, salt sequence, crystallizer and the final water/solids balance.',
  description: 'Follows a brine through softening, membrane pre-concentration, a falling-film brine concentrator and a forced-circulation crystallizer. At every evaporation step the Pitzer-based mineral equilibria decide which salts crystallise, so the precipitation sequence, salt yields and purity, boiling-point elevation, compressor work, steam demand and heat-transfer areas follow from the actual brine chemistry. The crystallizer size distribution is solved as an MSMPR population balance with primary and secondary nucleation and power-law growth, both by moments and on a discretised size grid, and again as a two-zone model with size-dependent growth, dissolution of fines, Ostwald ripening and a wall boundary condition. Electrodialysis and membrane distillation can replace the pressure- and steam-driven steps; filter sizing and a levelised-cost estimate complete the train. The final balance states how much water is recovered, which solids leave, and whether any liquid discharge remains.',
  guide: [
    'Pull the RO concentrate (or the brine from suite 2) or enter a brine analysis and flow.',
    'Choose the pretreatment, the membrane pre-concentration step and the type of evaporator and crystallizer drive.',
    'Set the brine-concentrator outlet salinity just below sodium-chloride saturation and choose the mother-liquor purge and where it goes.',
    'Run. Read the precipitation-sequence plot and the salt table, then the unit table for energy and area, and the ZLD balance for the remaining liquid.',
  ],
  referenceOnly: ['cfd-population-balance', 'cfd-crystallization'],
  implemented: ['total and component mass balance', 'energy balance', 'phase-equilibrium', 'solubility-product', 'saturation-index', 'supersaturation equation', 'classical nucleation equation', 'primary-nucleation', 'secondary-nucleation', 'crystal-growth', 'population-balance equation', 'moment equation', 'crystal-size-distribution', 'evaporation equation', 'vapour–liquid equilibrium', 'solid–liquid equilibrium', 'heat-transfer equation',
    'electrolyte-equilibrium–crystallization', 'evaporation–precipitation', 'nucleation–growth–population-balance', 'ro–crystallization', 'zld process-integration', 'resource-recovery–selective-precipitation', 'thermodynamic–kinetic crystallization',
    'dissolution equations', 'ostwald-ripening', 'cfd–population-balance', 'cfd–crystallization', 'membrane-distillation–crystallization', 'electrodialysis–crystallization', 'crystallizer wall no-flux', 'crystal-wall deposition', 'dissolution', 'filtration', 'process economics',
    'brine composition', 'supersaturation', 'temperature', 'pressure', 'initial crystal population', 'seed size distribution', 'initial solid fraction', 'brine-feed concentration/flow', 'heat-flux or temperature', 'evaporation/vapour-flux', 'outlet population-flux', 'solid–liquid equilibrium/interface',
    'concentrated-electrolyte thermodynamics', 'brine concentration', 'evaporation', 'mechanical and thermal vapour compression', 'mineral saturation', 'nucleation', 'crystal growth', 'precipitation', 'solid-liquid equilibrium', 'crystalliser modelling', 'solids separation', 'centrifugation', 'drying', 'mother-liquor recycling', 'salt-purity', 'selective mineral recovery', 'chemical dosing', 'scale management', 'heat integration', 'water-recovery calculation', 'waste minimisation', 'zero-liquid-discharge assessment', 'resource recovery', 'energy analysis'],
  equationsNote: 'Mineral equilibria use the Harvie–Møller–Weare Pitzer set (25 °C interaction parameters; solubility products and the Debye–Hückel slope follow temperature), which reproduces the seawater evaporation sequence at 25 °C; at evaporator temperatures the onset points are indicative (± 10–15 % in concentration factor) and double salts of the hot Mg–K–SO₄ system are approximate. The path stops at a water activity of 0.33 or an ionic strength of 22 mol/kg; CaCl₂ hydrates are not included, so calcium-chloride bitterns always leave with the purge. Crystallisation is fractional (solids leave the liquor as they form). The reference MSMPR model assumes a well-mixed crystallizer with size-independent growth. The two-zone model adds what a flow simulation coupled to a population balance would resolve, in reduced (compartment) form: a boiling zone and a body/heater-loop zone with separate supersaturation and size distribution, exchanging slurry with the circulation flow that follows from the heat duty and the temperature rise per pass; it is not a three-dimensional flow field, so dead zones, classification and local short-circuiting are not predicted. Its size grid is first-order upwind (about 5 % in median size), which is why it is always compared with the ideal case on the same grid. Growth below the Gibbs–Thomson critical size turns into diffusion-limited dissolution (Noyes–Whitney with Sh = 2); ripening in the hold tank is diffusion-controlled Lifshitz–Slyozov–Wagner coarsening at constant crystal volume. Wall deposition uses a single first-order capture velocity. Agglomeration and breakage are not modelled. Membrane distillation is a lumped direct-contact model (vapour-pressure driving force with a temperature-polarisation coefficient, no wetting or module pressure drop); electrodialysis uses the Faraday law with a cell-pair voltage from ohmic and membrane potentials and a water-transport ceiling, with the diluate polished by RO. Filter sizing uses the incompressible-cake Ruth equation. Costs are order-of-magnitude correlations (± 40 %) for comparing options, not a budget estimate. Evaporator energy is a lumped single-stage balance (MVC) or a steam-economy correlation (MEE/TVC); use suite 6 for effect-by-effect thermal design.',

  inputs: [
    { group: 'Feed brine', help: 'The brine to be taken to zero liquid discharge.', fields: [
      { key: 'ions', label: 'Brine analysis (mg/L)', type: 'ions', value: WATERS.robrine.ions, help: 'Complete ionic analysis; the salt sequence depends on every major ion.' },
      { key: 'Q', label: 'Brine flow', unit: 'm³/h', value: 100, min: 0.01, max: 1e5 },
      { key: 'T', label: 'Brine temperature', unit: '°C', value: 26, min: 1, max: 60 },
      { key: 'pH', label: 'Brine pH', unit: '', value: 7.9, min: 2, max: 12 },
    ] },
    { group: 'Pretreatment: softening and selective recovery', help: 'Reagent dosing before concentration. Removing magnesium and calcium raises the salt purity and lets the membranes and evaporator run to higher concentration.', fields: [
      { key: 'softMode', label: 'Selective precipitation', type: 'select', value: 'none', options: [{ value: 'none', label: 'None' }, { value: 'mg', label: 'Alkali to target pH → Mg(OH)₂ recovery' }, { value: 'mgca', label: 'Alkali, then soda ash → Mg(OH)₂ and CaCO₃' }] },
      { key: 'alkali', label: 'Alkali', type: 'select', value: 'naoh', options: [{ value: 'naoh', label: 'Caustic soda (pure Mg(OH)₂)' }, { value: 'lime', label: 'Hydrated lime (cheaper, adds calcium)' }], showIf: (v) => v.softMode !== 'none' },
      { key: 'softPH', label: 'Target pH of the magnesium stage', unit: '', value: 10.6, min: 9, max: 12, help: 'Brucite precipitates above about pH 9.5; 10.5–11 removes more than 98 % of the magnesium.', showIf: (v) => v.softMode !== 'none' },
      { key: 'sodaRatio', label: 'Soda ash per remaining calcium', unit: 'mol/mol', value: 1.02, min: 0, max: 1.5, showIf: (v) => v.softMode === 'mgca' },
      { key: 'softNeutral', label: 'pH after neutralisation (HCl)', unit: '', value: 7.5, min: 5, max: 11, showIf: (v) => v.softMode !== 'none' },
    ] },
    { group: 'Membrane pre-concentration', help: 'Removing water with membranes costs far less energy than evaporating it.', fields: [
      { key: 'preconc', label: 'Technology', type: 'select', value: 'hpro', options: [{ value: 'none', label: 'None' }, { value: 'hpro', label: 'High-pressure RO (up to 120 bar)' }, { value: 'oaro', label: 'Osmotically assisted RO (multi-stage, about 70 bar)' }, { value: 'ed', label: 'Electrodialysis concentrator (diluate polished by RO)' }], help: 'Electrodialysis moves the salt instead of the water; its concentrate strength is limited by the water that travels with the ions.' },
      { key: 'roTDS', label: 'Target concentrate salinity', unit: 'g/kg', value: 120, min: 20, max: 250, help: 'High-pressure RO reaches about 120–130 g/kg; osmotically assisted RO 150–230 g/kg.', showIf: (v) => v.preconc !== 'none' },
      { key: 'pMax', label: 'Membrane pressure rating', unit: 'bar', value: 120, min: 40, max: 200, showIf: (v) => v.preconc === 'hpro' },
      { key: 'erdEff', label: 'Energy-recovery efficiency', unit: '%', value: 94, min: 0, max: 99, showIf: (v) => v.preconc === 'hpro' },
      { key: 'etaPump', label: 'High-pressure pump efficiency', unit: '%', value: 84, min: 30, max: 93, showIf: (v) => v.preconc === 'hpro' },
      { key: 'oaroP', label: 'Operating pressure', unit: 'bar', value: 68, min: 30, max: 90, showIf: (v) => v.preconc === 'oaro' },
      { key: 'eta2', label: 'Second-law efficiency', unit: '%', value: 28, min: 5, max: 70, help: 'Reversible work of separation divided by actual electricity use; 20–35 % for multi-stage osmotically assisted RO.', showIf: (v) => v.preconc === 'oaro' },
      { key: 'edI', label: 'Current density', unit: 'A/m²', value: 300, min: 20, max: 1500, help: 'Per cell-pair area; sets the membrane area and the ohmic voltage.', showIf: (v) => v.preconc === 'ed' },
      { key: 'edEta', label: 'Current efficiency', unit: '%', value: 90, min: 30, max: 100, help: 'Share of the current that carries salt into the concentrate (co-ion leakage and back-diffusion lower it).', showIf: (v) => v.preconc === 'ed' },
      { key: 'edRcp', label: 'Cell-pair area resistance', unit: 'Ω·cm²', value: 25, min: 2, max: 200, help: 'Two membranes plus the diluate and concentrate channels.', showIf: (v) => v.preconc === 'ed' },
      { key: 'edTw', label: 'Water transported with the salt', unit: 'mol H₂O/eq', value: 12, min: 2, max: 40, help: 'Electro-osmosis plus osmosis; it caps the concentrate at 1/(t_w·18 g/mol) equivalents per kg of water.', showIf: (v) => v.preconc === 'ed' },
      { key: 'edDil', label: 'Diluate salinity leaving the stack', unit: 'g/kg', value: 3, min: 0.2, max: 30, help: 'The diluate is polished by RO; its reject returns to the stack.', showIf: (v) => v.preconc === 'ed' },
      { key: 'edRoSEC', label: 'Energy of the polishing RO', unit: 'kWh/m³', value: 1.5, min: 0.2, max: 6, help: 'Per m³ of water recovered from the diluate.', showIf: (v) => v.preconc === 'ed' },
    ] },
    { group: 'Brine concentrator', help: 'Seeded-slurry falling-film evaporator.', fields: [
      { key: 'bcType', label: 'Type', type: 'select', value: 'mvc', options: [{ value: 'mvc', label: 'Mechanical vapour compression (MVC)' }, { value: 'mee', label: 'Multi-effect evaporator, steam driven' }, { value: 'tvc', label: 'Multi-effect with thermal vapour compression' }, { value: 'mdc', label: 'Membrane distillation (direct contact, low-grade heat)' }], help: 'Membrane distillation works below the boiling point with heat at 60–90 °C; its flux follows the vapour-pressure difference and falls as the water activity of the brine drops.' },
      { key: 'bcTDS', label: 'Outlet salinity', unit: 'g/kg', value: 240, min: 60, max: 320, typical: [200, 260], help: 'Kept just below sodium-chloride saturation (about 265 g/kg for seawater-type brines).' },
      { key: 'Tbc', label: 'Boiling temperature (top effect for MEE)', unit: '°C', value: 100, min: 40, max: 125, showIf: (v) => v.bcType !== 'mdc' },
      { key: 'mdTf', label: 'Membrane-distillation feed temperature', unit: '°C', value: 70, min: 40, max: 95, help: 'Bulk temperature of the brine in the modules of the concentrator.', showIf: (v) => v.bcType === 'mdc' },
      { key: 'mdTp', label: 'Permeate-side temperature', unit: '°C', value: 25, min: 5, max: 60, help: 'Bulk temperature of the cold distillate loop.', showIf: (v) => v.bcType === 'mdc' || v.cxDrive === 'md' },
      { key: 'mdB', label: 'Membrane vapour permeability', unit: 'kg/m²·h·kPa', value: 0.7, min: 0.05, max: 5, help: 'Flux per unit vapour-pressure difference across the membrane; 0.3–1.5 for commercial hydrophobic membranes.', showIf: (v) => v.bcType === 'mdc' || v.cxDrive === 'md' },
      { key: 'mdTPC', label: 'Temperature-polarisation coefficient', unit: '–', value: 0.6, min: 0.2, max: 1, help: 'Share of the bulk temperature difference that acts across the membrane itself.', showIf: (v) => v.bcType === 'mdc' || v.cxDrive === 'md' },
      { key: 'mdCond', label: 'Membrane conduction coefficient', unit: 'W/m²·K', value: 250, min: 0, max: 3000, help: 'Heat conducted through the membrane without evaporating water (thermal conductivity / thickness).', showIf: (v) => v.bcType === 'mdc' || v.cxDrive === 'md' },
      { key: 'mdHR', label: 'Heat recovered from the permeate loop', unit: '%', value: 50, min: 0, max: 90, help: 'Share of the module heat duty returned to the feed by the recuperator.', showIf: (v) => v.bcType === 'mdc' || v.cxDrive === 'md' },
      { key: 'mdPump', label: 'Circulation pumps of the membrane modules', unit: 'kWh/t', value: 2, min: 0, max: 15, help: 'Electricity per tonne of distillate.', showIf: (v) => v.bcType === 'mdc' },
      { key: 'dT', label: 'Tube temperature difference (MVC)', unit: 'K', value: 3.5, min: 1, max: 12, help: 'Condensing vapour minus boiling brine. Smaller values save compressor energy but need more area.', showIf: (v) => v.bcType === 'mvc' },
      { key: 'U', label: 'Overall heat-transfer coefficient', unit: 'kW/m²·K', value: 2.2, min: 0.3, max: 5 },
      { key: 'etaComp', label: 'Compressor isentropic efficiency', unit: '%', value: 76, min: 40, max: 90 },
      { key: 'nEff', label: 'Number of effects', unit: '', value: 4, min: 1, max: 12, step: 1, showIf: (v) => v.bcType === 'mee' || v.bcType === 'tvc' },
      { key: 'Tlast', label: 'Last-effect temperature', unit: '°C', value: 48, min: 30, max: 90, showIf: (v) => v.bcType === 'mee' || v.bcType === 'tvc' },
      { key: 'tvcRa', label: 'Thermocompressor entrainment ratio', unit: 'kg/kg', value: 1, min: 0.2, max: 3, showIf: (v) => v.bcType === 'tvc' },
      { key: 'steamT', label: 'Heating-steam temperature', unit: '°C', value: 120, min: 60, max: 200, showIf: (v) => v.bcType === 'mee' || v.bcType === 'tvc' },
      { key: 'bcSolids', label: 'Seed-slurry solids (CaSO₄, CaCO₃, silica)', type: 'select', value: 'separate', options: [{ value: 'separate', label: 'Separated before the crystallizer' }, { value: 'cake', label: 'Carried into the salt cake' }] },
    ] },
    { group: 'Crystallizer and solids handling', fields: [
      { key: 'cxDrive', label: 'Crystallizer drive', type: 'select', value: 'steam', options: [{ value: 'steam', label: 'Steam (single effect)' }, { value: 'mvr', label: 'Mechanical vapour recompression' }, { value: 'md', label: 'Membrane-distillation crystallizer' }], help: 'In a membrane-distillation crystallizer the modules remove the water below the boiling point and the crystals form in the circulated tank.' },
      { key: 'Tcx', label: 'Crystallizer temperature', unit: '°C', value: 80, min: 30, max: 125 },
      { key: 'purge', label: 'Mother-liquor purge', unit: '% of crystallizer feed water', value: 12, min: 0.5, max: 100, help: 'Water left in the mother liquor that is not evaporated. The purge removes highly soluble Mg, K and Ca salts; 100 % means no crystallizer.' },
      { key: 'tau', label: 'Crystal residence time', unit: 'h', value: 1.5, min: 0.1, max: 12 },
      { key: 'MT', label: 'Magma (slurry) density', unit: 'kg/m³', value: 250, min: 20, max: 700, help: 'Mass of crystals per m³ of slurry; 150–350 kg/m³ is typical.' },
      { key: 'cakeMoist', label: 'Centrifuge cake moisture', unit: '% liquor', value: 4, min: 0.5, max: 40 },
      { key: 'wash', label: 'Cake wash (condensate)', unit: 'kg/kg salt', value: 0.05, min: 0, max: 1, help: 'Displacement wash that removes mother-liquor impurities; the wash returns to the crystallizer.' },
      { key: 'purgeFate', label: 'Destination of the purge', type: 'select', value: 'dryer', options: [{ value: 'dryer', label: 'Dryer / solidification (true ZLD)' }, { value: 'pond', label: 'Evaporation pond' }, { value: 'discharge', label: 'Liquid disposal (deep well, haulage)' }] },
      { key: 'pondEvap', label: 'Pan evaporation at site', unit: 'mm/y', value: 2000, min: 100, max: 4000, showIf: (v) => v.purgeFate === 'pond' },
      { key: 'pondRain', label: 'Rainfall', unit: 'mm/y', value: 120, min: 0, max: 3000, showIf: (v) => v.purgeFate === 'pond' },
      { key: 'pondRH', label: 'Mean relative humidity', unit: '%', value: 40, min: 5, max: 95, showIf: (v) => v.purgeFate === 'pond' },
    ] },
    { group: 'Process economics', help: 'Capital-cost correlations (installed cost at 1000 m³/d, scaled with a capacity exponent), financing and fixed operating costs. Order-of-magnitude figures for comparing train options.', fields: [
      { key: 'rate', label: 'Discount rate', unit: '%/y', value: 8, min: 0, max: 25, help: 'Used in the capital-recovery factor.' }, { key: 'life', label: 'Plant life', unit: 'years', value: 20, min: 3, max: 40, help: 'Amortisation period.' }, { key: 'avail', label: 'Availability', unit: '%', value: 92, min: 30, max: 100, help: 'Share of the year in operation.' },
      { key: 'capexMem', label: 'Membrane pre-concentration', unit: '$ per m³/d', value: 1800, min: 100, max: 20000, help: 'Installed cost per m³/d of permeate at 1000 m³/d (osmotically assisted RO is taken 1.6 times higher).' },
      { key: 'capexBC', label: 'Brine concentrator', unit: '$ per m³/d', value: 6000, min: 500, max: 50000, help: 'Installed cost per m³/d of distillate at 1000 m³/d.' }, { key: 'capexCX', label: 'Crystallizer', unit: '$ per m³/d', value: 14000, min: 1000, max: 100000, help: 'Installed cost per m³/d of condensate at 1000 m³/d.' },
      { key: 'capexScale', label: 'Capacity exponent', unit: '–', value: 0.7, min: 0.4, max: 1, help: 'Cost ∝ capacity^n; 0.6–0.8 for evaporators.' }, { key: 'capexSolids', label: 'Solids handling', unit: '$ per t/d', value: 30000, min: 0, max: 5e5, help: 'Centrifuge, filter press, dryer and conveying per tonne of solids per day.' },
      { key: 'capexSoft', label: 'Softening and clarification', unit: '$ per m³/d', value: 400, min: 0, max: 5000, help: 'Per m³/d of brine treated, at 1000 m³/d.', showIf: (v) => v.softMode !== 'none' }, { key: 'capexED', label: 'Electrodialysis stacks', unit: '$/m²', value: 450, min: 50, max: 3000, help: 'Per m² of cell-pair area, installed.', showIf: (v) => v.preconc === 'ed' },
      { key: 'capexMD', label: 'Membrane-distillation modules', unit: '$/m²', value: 250, min: 30, max: 2000, help: 'Per m² of membrane, installed.', showIf: (v) => v.bcType === 'mdc' || v.cxDrive === 'md' }, { key: 'capexPond', label: 'Lined evaporation pond', unit: '$/m²', value: 60, min: 5, max: 400, help: 'Earthworks and liner.', showIf: (v) => v.purgeFate === 'pond' },
      { key: 'capexIndirect', label: 'Indirect costs and contingency', unit: '% of direct', value: 35, min: 0, max: 150, help: 'Engineering, commissioning, owner costs and contingency.' }, { key: 'maint', label: 'Maintenance', unit: '% of capital/y', value: 3, min: 0, max: 15, help: 'Annual maintenance and insurance.' }, { key: 'labour', label: 'Labour', unit: '$/y', value: 400000, min: 0, max: 1e7, help: 'Operators and supervision.' },
    ] },
    { group: 'Prices (for the revenue hint)', fields: [
      { key: 'elecPrice', label: 'Electricity', unit: '$/kWh', value: 0.08, min: 0, max: 1 }, { key: 'steamPrice', label: 'Heat (steam)', unit: '$/MWh', value: 25, min: 0, max: 300 },
      { key: 'priceNaCl', label: 'Sodium chloride (industrial grade)', unit: '$/t', value: 45, min: 0, max: 500 }, { key: 'purityMin', label: 'Purity needed to sell the salt', unit: '%', value: 97, min: 80, max: 99.9 },
      { key: 'priceMg', label: 'Magnesium hydroxide', unit: '$/t', value: 280, min: 0, max: 2000 }, { key: 'disposal', label: 'Disposal of mixed solids', unit: '$/t', value: 60, min: 0, max: 1000 },
    ] },
    { group: 'Chemistry model', tab: 'setup', help: 'Thermodynamic basis of the evaporation path.', fields: [
      { key: 'model', label: 'Activity model', type: 'select', value: 'pitzer', options: [{ value: 'pitzer', label: ACTIVITY_MODELS.pitzer }], help: 'Only the Pitzer model is valid at brine-concentrator and crystallizer salinities.' },
      { key: 'salts', label: 'Solid phases considered', type: 'select', value: 'all', options: [{ value: 'all', label: 'Full evaporite set (incl. double salts)' }, { value: 'simple', label: 'Simple salts only' }] },
      { key: 'chemT', label: 'Temperature of the equilibria', type: 'select', value: 'unit', options: [{ value: 'unit', label: 'Operating temperature of each unit' }, { value: '25', label: '25 °C (validated parameter set)' }], help: 'Hot equilibria capture the lower solubility of anhydrite and the higher solubility of halite, with less certain double-salt data.' },
      { key: 'pCO2', label: 'CO₂ partial pressure in evaporator vents', unit: 'µatm', value: 1000, min: 1, max: 1e5, help: 'Boundary condition for the carbonate system while boiling: CO₂ is stripped with the vapour.' },
    ] },
    { group: 'Evaporator details and limits', tab: 'setup', fields: [
      { key: 'approach', label: 'Preheater approach temperature', unit: 'K', value: 4, min: 1, max: 30, help: 'Distillate leaves this much warmer than the entering brine.' },
      { key: 'heatLoss', label: 'Heat loss', unit: '% of latent duty', value: 2, min: 0, max: 15 },
      { key: 'bcPumps', label: 'Recirculation and auxiliaries, concentrator', unit: 'kWh/t', value: 1.8, min: 0, max: 10 },
      { key: 'cxPumps', label: 'Recirculation pump, crystallizer', unit: 'kWh/t', value: 6, min: 0, max: 30 },
      { key: 'Ufc', label: 'Crystallizer heater coefficient', unit: 'kW/m²·K', value: 2.5, min: 0.3, max: 6 },
      { key: 'dTcx', label: 'Crystallizer heater temperature difference', unit: 'K', value: 8, min: 2, max: 25 },
      { key: 'bpeMax', label: 'Largest boiling-point elevation for MVR', unit: 'K', value: 18, min: 5, max: 40 },
      { key: 'etaMotor', label: 'Motor and drive efficiency', unit: '%', value: 95, min: 60, max: 99 },
      { key: 'etaDry', label: 'Dryer thermal efficiency', unit: '%', value: 60, min: 20, max: 95 },
    ] },
    { group: 'Crystallisation kinetics', tab: 'setup', help: 'Growth G = kg·σ^g and nucleation B = A·exp(−B′/ln²S) + kb·MT^j·σ^b with relative supersaturation σ = S − 1.', fields: [
      { key: 'kg', label: 'Growth constant kg', unit: 'm/s', value: 5e-6, min: 1e-9, max: 1e-2, help: 'Linear growth rate at σ = 1. Sodium chloride: 10⁻⁶–10⁻⁵ m/s.' },
      { key: 'gExp', label: 'Growth order g', unit: '–', value: 1, min: 0.5, max: 3 },
      { key: 'kb', label: 'Secondary-nucleation constant kb', unit: '#/(kg·s)', value: 1e8, min: 1e3, max: 1e13 },
      { key: 'bExp', label: 'Nucleation order b', unit: '–', value: 2, min: 0.5, max: 6 },
      { key: 'jMT', label: 'Magma-density exponent j', unit: '–', value: 1, min: 0, max: 2 },
      { key: 'kv', label: 'Volume shape factor', unit: '–', value: 1, min: 0.3, max: 1.2, help: '1 for cubes sized by their edge, π/6 for spheres.' },
      { key: 'logA', label: 'Primary nucleation: log₁₀ A', unit: 'log(m⁻³s⁻¹)', value: 30, min: 15, max: 36 },
      { key: 'het', label: 'Primary nucleation: heterogeneous factor', unit: '–', value: 0.3, min: 0.01, max: 1 },
    ] },
    { group: 'Crystallizer zones, dissolution, ripening and wall', tab: 'setup', help: 'Physics beyond the ideal mixed crystallizer: two compartments exchanging slurry with the circulation flow, size-dependent growth and dissolution (Gibbs–Thomson), the boundary condition at the wall, coarsening in the slurry hold tank and removal of wall deposits.', fields: [
      { key: 'zoneOn', label: 'Solve the two-zone crystallizer model', type: 'bool', value: true, help: 'Boiling zone (where supersaturation is generated) and body + heater loop (where the product is withdrawn), each with its own supersaturation and size distribution; the exchange follows from the heat duty and the temperature rise per pass.' },
      { key: 'phi1', label: 'Volume share of the boiling zone', unit: '%', value: 15, min: 2, max: 60, help: 'Part of the active volume near the liquid surface where the vapour flashes.', showIf: (v) => v.zoneOn },
      { key: 'dTpass', label: 'Temperature rise per pass through the heater', unit: 'K', value: 3, min: 0.5, max: 15, help: 'Sets the circulation flow (duty / (ρ·cp·ΔT)) and with it the turnover time of the slurry. Small values keep the supersaturation at the boiling surface low.' },
      { key: 'gtOn', label: 'Size-dependent solubility (Gibbs–Thomson)', type: 'bool', value: true, help: 'Crystals smaller than the critical size 2α/ln(1+σ) are undersaturated and dissolve at a diffusion-limited rate; larger ones grow more slowly than the bulk law. This is the mechanism of Ostwald ripening.', showIf: (v) => v.zoneOn },
      { key: 'wallBC', label: 'Boundary condition at the wall', type: 'select', value: 'noflux', options: [{ value: 'noflux', label: 'No flux: crystals do not deposit' }, { value: 'deposit', label: 'Deposition: crystals stick and the deposit grows' }], help: 'With deposition the population balance loses crystals to the wall at the rate k_dep × wall area / volume; the deposit is removed by periodic boil-outs.' },
      { key: 'kDep', label: 'Deposition velocity of crystals', unit: 'µm/s', value: 0.005, min: 0, max: 10, help: 'Mass-transfer velocity of crystals towards the wall times their sticking probability; 0.001–0.02 µm/s gives the millimetres per month seen in well-run forced-circulation units.', showIf: (v) => v.wallBC === 'deposit' },
      { key: 'runDays', label: 'Run time between boil-outs', unit: 'd', value: 30, min: 1, max: 365, help: 'The wall deposit accumulates over this period and is then dissolved with condensate.', showIf: (v) => v.wallBC === 'deposit' },
      { key: 'uTube', label: 'Velocity in the heater tubes', unit: 'm/s', value: 2, min: 0.5, max: 4, help: 'Forced-circulation crystallizers run at 1.5–3 m/s to limit scaling.' }, { key: 'dTube', label: 'Heater tube diameter', unit: 'mm', value: 38, min: 15, max: 80, help: 'Inner diameter.' },
      { key: 'finesL', label: 'Fines size for the dissolution check', unit: 'µm', value: 10, min: 0.5, max: 200, help: 'Size of the fine crystals whose dissolution time in the heated loop is reported.' },
      { key: 'tHold', label: 'Residence time in the slurry hold tank', unit: 'h', value: 2, min: 0, max: 72, help: 'Saturated slurry held before the centrifuge: small crystals dissolve and large ones grow (Ostwald ripening).' },
    ] },
    { group: 'Solids filtration', tab: 'setup', help: 'Filter press for the softening sludge and the seed-slurry solids of the concentrator, sized with the cake-filtration (Ruth) equation.', fields: [
      { key: 'filtAlpha', label: 'Specific cake resistance', unit: 'm/kg', value: 5e11, min: 1e9, max: 1e14, help: 'Calcium carbonate and sulphate 10¹⁰–10¹¹, magnesium hydroxide 10¹²–10¹³ m/kg.' }, { key: 'filtDP', label: 'Filtration pressure', unit: 'bar', value: 6, min: 0.5, max: 16, help: 'Pressure difference across cake and cloth.' },
      { key: 'filtTime', label: 'Filtration time per cycle', unit: 'min', value: 30, min: 5, max: 240, help: '30 min are added for opening, discharge and closing.' }, { key: 'filtFeed', label: 'Solids in the thickened feed', unit: '% w/w', value: 8, min: 1, max: 40, help: 'Underflow of the clarifier or hydrocyclone that feeds the press.' },
      { key: 'filtMoist', label: 'Cake moisture', unit: '% w/w', value: 35, min: 10, max: 75, help: 'Liquor left in the filter cake.' },
    ] },
    { group: 'Initial condition of the crystallizer', tab: 'setup', help: 'State at start-up; the size distribution is integrated from here to steady state.', fields: [
      { key: 'seedMass', label: 'Seed loading', unit: 'kg/m³', value: 20, min: 0, max: 400 }, { key: 'seedL', label: 'Seed size', unit: 'µm', value: 100, min: 5, max: 1000 },
      { key: 'sigma0', label: 'Initial supersaturation', unit: '%', value: 0.5, min: 0, max: 50 }, { key: 'tEnd', label: 'Simulated time', unit: 'residence times', value: 12, min: 3, max: 40 },
    ] },
    { group: 'Discretisation', tab: 'mesh', help: 'Number of evaporation steps along the concentration path and the size grid of the population balance.', fields: [
      { key: 'nEvap', label: 'Evaporation steps', unit: '', value: 32, min: 6, max: 300, step: 1, help: 'Geometric steps in remaining water, shared between the units.' },
      { key: 'nL', label: 'Crystal-size cells', unit: '', value: 80, min: 10, max: 1000, step: 1 },
      { key: 'lmaxMult', label: 'Size domain', unit: '× Gτ', value: 16, min: 8, max: 40, help: 'Largest size on the grid as a multiple of the characteristic size Gτ.' },
      { key: 'scheme', label: 'Growth-flux scheme', type: 'select', value: 'vanleer', options: [{ value: 'vanleer', label: 'High resolution (van Leer limiter)' }, { value: 'upwind', label: 'First-order upwind' }] },
    ] },
  ],

  presets: [
    { name: 'SWRO concentrate → HPRO → MVC concentrator → steam crystallizer', values: {} },
    { name: 'Inland brackish concentrate with lime–soda softening, OARO and pond', values: { ions: scaleIons(WATERS.brackish.ions, 4.4), Q: 40, T: 24, pH: 7.4, softMode: 'mgca', alkali: 'lime', softPH: 10.8, preconc: 'oaro', roTDS: 150, bcTDS: 250, purge: 8, purgeFate: 'pond', pondEvap: 2400, pondRH: 35 } },
    { name: 'Seawater mineral recovery: Mg(OH)₂, then MEE-TVC to salt', values: { ions: WATERS.seawater.ions, Q: 200, T: 25, pH: 8.1, softMode: 'mg', alkali: 'naoh', softPH: 10.6, preconc: 'hpro', roTDS: 70, bcType: 'tvc', Tbc: 76, nEff: 3, Tlast: 44, bcTDS: 245, Tcx: 70, purge: 15 } },
    { name: 'Produced water, MVC concentrator, liquid purge to disposal well', values: { ions: WATERS.produced.ions, Q: 60, T: 40, pH: 6.8, preconc: 'none', bcTDS: 250, purge: 30, purgeFate: 'discharge', cxDrive: 'steam' } },
    { name: 'Minimal liquid discharge: concentrator only, brine to pond', values: { preconc: 'oaro', roTDS: 160, bcTDS: 250, purge: 100, purgeFate: 'pond' } },
  ],

  pull: ({ outputs }) => {
    const b = outputs?.chem?.streams?.brine?.ions ? outputs.chem.streams.brine : outputs?.ro?.streams?.concentrate, from = outputs?.chem?.streams?.brine?.ions ? 'Brine chemistry: concentrate' : 'RO design: concentrate';
    if (!b?.ions) return [];
    return [{ key: 'ions', value: b.ions, from }, b.Q ? { key: 'Q', value: b.Q, from } : null, b.T != null ? { key: 'T', value: clamp(b.T, 1, 60), from } : null, b.pH ? { key: 'pH', value: b.pH, from } : null].filter(Boolean);
  },
  site: (site) => [
    site?.data?.humidity != null ? { key: 'pondRH', value: clamp(site.data.humidity, 5, 95), from: 'Relative humidity at site' } : null,
    site?.data?.ghiDaily != null ? { key: 'pondEvap', value: Math.round(clamp(365 * ((0.5 * site.data.ghiDaily * 3.6) / 2.45), 100, 4000)), from: 'Evaporation estimated from solar irradiation (energy balance)' } : null,
    site?.data?.electricityPrice != null ? { key: 'elecPrice', value: site.data.electricityPrice, from: 'Electricity price at site' } : null,
  ].filter(Boolean),

  run(v, ctx) {
    ctx?.progress?.(0.1, 'Evaporation path and mineral equilibria…');
    const r = simulateZLD(v), { path, ro, bc, cx, csd, water, end } = r, W = r.W, steps = path.steps;
    ctx?.progress?.(0.85, 'Balances…');
    const td = (kg) => (kg * 24) / 1000, recPct = (100 * (water.perm + water.dist + water.cond)) / water.in, zld = r.liquid <= 1e-9;
    if (!zld) W.push({ level: 'warn', msg: `${fmt(r.liquid, 3)} m³/h of mother-liquor purge still leaves as liquid — this is minimal, not zero, liquid discharge.` });
    else W.unshift({ level: 'info', msg: 'No liquid discharge remains: all water leaves as product water, vapour or moisture bound in solids.' });
    if (r.cxSolids.halite > 0 && r.main === 'halite' && r.purity < v.purityMin / 100) W.push({ level: 'info', msg: `Salt-cake purity ${fmt(100 * r.purity, 4)} % is below the ${v.purityMin} % sales grade — increase the purge or the wash, separate the seed solids, or soften the brine first.` });
    if (!cx.on && v.purge < 100) W.push({ level: 'info', msg: 'The crystallizer is idle: the brine concentrator already reaches the requested end point.' });
    const rev = sum(r.salts.map((s) => td(s.kg) * s.value)), energyCost = 24 * (r.kWe * v.elecPrice + (r.kWt * v.steamPrice) / 1000);
    const reagCost = 24 * sum(Object.entries(r.reagents).map(([k, kg]) => (kg / 1000) * ({ naoh: 450, lime: 130, soda: 280, hcl: 180 }[k] || 0)));
    const saltsOut = {};
    for (const s of r.salts) saltsOut[s.name] = +(td(s.kg) + (saltsOut[s.name] || 0)).toPrecision(6);
    ctx?.progress?.(0.9, 'Crystallizer zones, filtration and costs…');
    // extended models: crystallizer physics, filtration, economics, membrane distillation and electrodialysis details
    const X = { K: [], PL: [], TB: [], BAL: [], out: {} }, ph = crystallizerPhysics(v, r), filt = solidsFiltration(v, r), eco = zldEconomics(v, r, { rev, reagCost }), inf = (x, dflt = 'none') => (Number.isFinite(x) ? x : dflt);
    if (ph) {
      const c = r.csd, z = ph.zone, um = 1e6, h = ph.hold, mm = (ph.thick * 1000) / Math.max(v.runDays ?? 30, 1) * 30;
      if (z && !z.ok) W.push({ level: 'warn', msg: 'The two-zone crystallizer model did not reach a physical solution for these kinetics; only the ideal mixed crystallizer is reported.' });
      if (z && z.ok) {
        const R0 = z.reference, P1 = z.stats[0], P2 = z.stats[1], md = (pop) => { const tot = Math.max(sum(z.L.map((q, j) => pop[j] * q ** 3 * z.dL[j])), 1e-300); return z.L.map((L, i) => (100 * pop[i] * L ** 3) / tot / 1e6); };
        X.TB.push({ title: 'Two-zone crystallizer model versus the ideal mixed crystallizer', columns: ['Quantity', 'Boiling zone', 'Body and heater loop (product)', 'Ideal single zone, same grid', 'Unit'], rows: [
          ['Relative supersaturation σ', z.sigma[0], z.sigma[1], R0.sigma, '–'], ['Growth rate of large crystals', z.G[0] * 1e9, z.G[1] * 1e9, R0.G * 1e9, 'nm/s'], ['Critical size (Gibbs–Thomson)', inf(z.Lcrit[0] * um, 0), inf(z.Lcrit[1] * um, 0), null, 'µm'],
          ['Median size L50 (mass)', P1.L50 * um, P2.L50 * um, R0.L50 * um, 'µm'], ['Mass-mean size L₄₃', P1.L43 * um, P2.L43 * um, R0.L43 * um, 'µm'], ['L10 / L90', `${fq(P1.L10 * um, 3)} / ${fq(P1.L90 * um, 3)}`, `${fq(P2.L10 * um, 3)} / ${fq(P2.L90 * um, 3)}`, `${fq(R0.L10 * um, 3)} / ${fq(R0.L90 * um, 3)}`, 'µm'],
          ['Coefficient of variation', P1.cv, P2.cv, R0.cv, '–'], ['Magma density', P1.mt, P2.mt, R0.mt, 'kg/m³'], ['Crystal number', P1.mu[0], P2.mu[0], R0.mu[0], '#/m³']],
          note: `Compartment form of a coupled flow / population-balance model: ${fq(100 * z.phi[0], 3)} % of the volume is the boiling zone, the slurry turns over every ${fq(ph.theta, 3)} s (circulation ${fq(ph.Qc * 3600, 3)} m³/h from the heat duty and ${v.dTpass} K rise per pass). ${v.gtOn !== false ? 'Growth is size dependent (Gibbs–Thomson) and crystals below the critical size dissolve.' : 'Size-independent growth.'} Wall: ${v.wallBC === 'deposit' ? `crystals deposit at ${v.kDep} µm/s on ${fq(ph.aWall, 3)} m² of wall per m³` : 'no flux'}. ${z.L.length} size cells, ${z.nSteps} implicit time steps; the last column is the ideal mixed crystallizer on the same size grid, so differences between the columns are physical, not numerical.` });
        X.PL.push({ type: 'line', title: 'Two-zone crystallizer: product size distribution (mass basis)', xlabel: 'Crystal size L (µm)', ylabel: 'Mass density (% per µm)', logx: true, series: [{ name: 'Product zone (body and loop)', x: z.L.map((L) => L * um), y: md(z.n[1]) }, { name: 'Boiling zone', x: z.L.map((L) => L * um), y: md(z.n[0]), dash: true }, { name: 'Ideal single zone on the same grid', x: z.L.map((L) => L * um), y: md(R0.pop), dash: true }], vlines: z.Lcrit[1] > 0 && Number.isFinite(z.Lcrit[1]) ? [{ x: Math.max(z.Lcrit[1] * um, z.L[0] * um), label: 'critical size' }] : [] });
        X.PL.push({ type: 'line', title: 'Two-zone crystallizer: supersaturation in the zones during start-up', xlabel: 'Time (residence times)', ylabel: 'Relative supersaturation σ × 10⁴', series: [{ name: 'Boiling zone', x: z.hist.th, y: z.hist.s1.map((q) => q * 1e4) }, { name: 'Body and heater loop', x: z.hist.th, y: z.hist.s2.map((q) => q * 1e4) }], hlines: [{ y: R0.sigma * 1e4, label: 'ideal mixed' }] });
        X.BAL.push({ name: 'Two-zone crystallizer: salt made by evaporation vs crystals + dissolved excess + product + wall (kg/m³)', in: z.balance.in, out: z.balance.out });
        X.K.push({ label: 'Product L50, two-zone model', value: P2.L50 * um, unit: 'µm', help: `Ideal mixed crystallizer on the same grid: ${fq(R0.L50 * um, 4)} µm` }, { label: 'Supersaturation, boiling zone / body', value: z.sigma[1] > 0 ? z.sigma[0] / z.sigma[1] : 1, unit: '×', status: z.sigma[1] > 0 && z.sigma[0] / z.sigma[1] > 4 ? 'warn' : 'ok', help: 'Ratio of the relative supersaturation in the two compartments; high values favour fines and encrustation at the boiling surface' });
        X.out.zoneL50um = P2.L50 * um; X.out.zoneSigmaRatio = z.sigma[1] > 0 ? z.sigma[0] / z.sigma[1] : 1;
        if (P2.mt < 0.5 * c.MT) W.push({ level: 'warn', msg: `In the two-zone model the crystal content settles at ${fq(P2.mt, 3)} kg/m³ instead of the design magma density of ${fq(c.MT, 3)} kg/m³: with these kinetics the crystals are washed out faster than they form and the salt leaves dissolved — raise the residence time or the seed loading.` });
        if (z.sigma[1] > 0 && z.sigma[0] / z.sigma[1] > 4) W.push({ level: 'warn', msg: `The boiling zone is ${fq(z.sigma[0] / z.sigma[1], 3)} times more supersaturated than the crystallizer body — raise the circulation (smaller temperature rise per pass) to avoid excessive nucleation and scaling at the surface.` });
      }
      X.TB.push({ title: 'Circulation loop, dissolution, ripening and wall', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [
        ['Circulation flow', ph.Qc * 3600, 'm³/h', `Heat duty / (ρ·cp·ΔT) with ${v.dTpass} K per pass`], ['Turnover time of the slurry', ph.theta, 's', 'Active volume / circulation flow'], ['Vapour flashed per pass', 100 * ph.flash, '% of the liquor', 'cp·ΔT / λ'],
        ['Tube Reynolds number', ph.Re, '–', `Slurry viscosity ${fq(ph.muSl * 1000, 3)} mPa·s (Thomas), ${fq(ph.nTubes, 3)} tubes`], ['Mass-transfer coefficient in the tubes', ph.kL * 1e6, 'µm/s', `Sh = 0.023·Re^0.8·Sc^⅓ = ${fq(ph.Sh, 3)}`], ['Loop pressure drop', ph.dP / 1e5, 'bar', `Circulation pump about ${fq(ph.pumpKW, 3)} kW`],
        ['Solubility change with temperature', 100 * ph.dlnc, '%/K', ph.dlnc >= 0 ? 'Normal solubility: the heated loop is less saturated' : 'Inverse solubility: the heater surface tends to scale'], ['Capillary length α = 2γVm/νRT', ph.alpha * 1e9, 'nm', `Interfacial energy ${ph.M.sigma} mJ/m²`], ['Critical size at the product supersaturation', inf(ph.Lcrit * 1e6, 'not supersaturated'), 'µm', 'Smaller crystals dissolve (Gibbs–Thomson)'],
        [`Dissolution time of ${v.finesL} µm fines in the heated loop`, inf(ph.tFines, 'do not dissolve'), 's', `Noyes–Whitney, Sh = 2; time in the loop per pass ${fq(ph.tLoop, 3)} s`], ['Ripening constant K (LSW)', ph.Klsw, 'm³/s', '⟨r⟩³ = r₀³ + K·t, K = 8γ·c∞·Vm²·D / (9νRT)'], ['Time for the mean size to coarsen by 1 %', inf(ph.t1pct / 86400, 'never'), 'd', 'Product crystals are far too large to ripen'],
        ...(h ? [['Hold tank: crystal number after / before', h.n0 > 0 ? h.n1 / h.n0 : 1, '–', `${v.tHold} h of ripening: the finest crystals dissolve`], ['Hold tank: L10 before → after', `${fq(h.L10a * 1e6, 4)} → ${fq(h.L10b * 1e6, 4)}`, 'µm', 'Mass basis'], ['Hold tank: L50 before → after', `${fq(h.L50a * 1e6, 4)} → ${fq(h.L50b * 1e6, 4)}`, 'µm', 'Mass basis'], ['Hold tank: number-mean size before → after', `${fq(2 * h.rMean0 * 1e6, 4)} → ${fq(2 * h.rMean1 * 1e6, 4)}`, 'µm', h.complete ? 'Lagrangian size classes, constant crystal volume' : 'Integration stopped at the step limit']] : []),
        ['Wall boundary condition', v.wallBC === 'deposit' ? 'Deposition' : 'No flux', '', v.wallBC === 'deposit' ? `Crystal flux to the wall ${fq(ph.rateCap * 86400, 3)} kg/m²·d on ${fq(ph.aWall, 3)} m² of wall per m³` : 'Crystals leave only with the product slurry'], ['Wall deposit growth', mm, 'mm per 30 d', `${fq(ph.thick * 1000, 3)} mm after ${v.runDays ?? 30} d`], ['Boil-out time to dissolve the deposit', ph.tBoil / 3600, 'h', 'Noyes–Whitney with condensate at the tube mass-transfer coefficient']] });
      if (ph.rip) X.PL.push({ type: 'line', title: 'Ostwald ripening in the slurry hold tank', xlabel: 'Time (h)', ylabel: 'Crystal number (% of initial) · number-mean size (µm)', series: [{ name: 'Crystal number (% of initial)', x: ph.rip.t.map((t) => t / 3600), y: ph.rip.number.map((n) => (100 * n) / ph.rip.number[0]) }, { name: 'Number-mean size (µm)', x: ph.rip.t.map((t) => t / 3600), y: ph.rip.rMean.map((q) => 2 * q * 1e6) }], note: 'Small crystals are more soluble than large ones, so they dissolve and feed the growth of the rest at constant total crystal volume.' });
      X.K.push({ label: 'Hold tank: crystals surviving ripening', value: h && h.n0 > 0 ? (100 * h.n1) / h.n0 : 100, unit: '% of number', help: `Ostwald ripening over ${v.tHold} h; the mass-median size changes from ${h ? fq(h.L50a * 1e6, 4) : '–'} to ${h ? fq(h.L50b * 1e6, 4) : '–'} µm` }, { label: 'Wall deposit growth', value: mm, unit: 'mm per 30 d', status: mm > 5 ? 'warn' : 'ok', help: v.wallBC === 'deposit' ? 'Crystals captured by the wall (deposition boundary condition)' : 'No-flux wall selected' });
      X.out.wallDepositMmPer30d = mm; X.out.ripeningNumberRatio = h && h.n0 > 0 ? h.n1 / h.n0 : 1; X.out.turnoverTime = ph.theta;
      if (mm > 5) W.push({ level: 'warn', msg: `The wall deposit grows by ${fq(mm, 3)} mm per month — plan boil-outs every ${fq(Math.max(1, (30 * 3) / mm), 2)} days or raise the tube velocity.` });
      if (ph.dlnc < 0) W.push({ level: 'info', msg: `${ph.M.name} becomes less soluble on heating (${fq(100 * ph.dlnc, 2)} %/K): the heater tubes are the most scale-prone surface.` });
    }
    if (r.ro.ed) { const e = r.ro.ed; X.TB.push({ title: 'Electrodialysis concentrator', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [['Cell-pair voltage', e.U, 'V', `Ohmic ${fq(e.Uohm, 3)} V + membrane potential ${fq(e.Umem, 3)} V`], ['Total current through one cell pair equivalent', e.current / 1000, 'kA', 'Faraday law: I = F·(eq/s)/current efficiency'], ['Cell-pair area', e.area, 'm²', `At ${v.edI} A/m²`], ['Stack energy', e.kWhPerKmol, 'kWh per kmol of charge', 'F·U/η'], ['Stack power', e.kW, 'kW', 'Before rectifier losses'], ['Concentrate strength', r.ro.edMolality, 'eq/kg water', `Ceiling from water transport: ${fq(e.mMax, 3)} eq/kg`]], note: 'The diluate is polished by RO and its reject returns to the stack, so the salt of the brine ends up in the concentrate that feeds the evaporator and crystallizer.' }); X.K.push({ label: 'Electrodialysis cell-pair voltage', value: e.U, unit: 'V' }); X.out.edArea = e.area; }
    for (const [u, md] of [['concentrator', r.bc.md], ['crystallizer', r.cx.md]]) if (md) { X.TB.push({ title: `Membrane distillation: ${u}`, columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [['Mean flux', md.flux, 'kg/m²·h', `From ${fq(md.jMax, 3)} at the inlet to ${fq(md.jMin, 3)} at the outlet water activity`], ['Membrane-surface temperatures', `${fq(md.Tfm, 3)} / ${fq(md.Tpm, 3)}`, '°C', 'Feed side / permeate side (temperature polarisation)'], ['Thermal efficiency', 100 * md.eta, '%', 'Latent heat / (latent + conduction)'], ['Gained output ratio', md.gor, '–', `With ${v.mdHR ?? 50} % heat recovery`], ['Water activity at which the flux stops', md.awStop, '–', 'psat(permeate side) / psat(feed side)']] }); X.K.push({ label: `Membrane-distillation flux, ${u}`, value: md.flux, unit: 'kg/m²·h', status: md.reversed ? 'bad' : 'ok' }); }
    if (filt.length) {
      X.TB.push({ title: 'Solids separation: filter press (cake filtration)', columns: ['Solids stream', 'Dry solids (kg/h)', 'Filtrate per cycle (m³)', 'Filter area (m²)', 'Cake thickness (mm)', 'Cycles per day', 'Liquor in the cake (kg/h)'], rows: filt.map((q) => [q.name, q.kg, q.V, q.area, q.thick * 1000, q.cycles, q.liquor]),
        note: `Ruth equation t/V = μαc·V/(2A²ΔP) + μR_m/(AΔP) with α = ${fq(v.filtAlpha, 3)} m/kg, ΔP = ${v.filtDP} bar and ${v.filtTime} min of filtration per cycle on a feed thickened to ${v.filtFeed} % solids. The liquor held in the cake (${v.filtMoist} %) is small and is not deducted from the water balance.` });
      X.K.push({ label: 'Filter-press area', value: sum(filt.map((q) => q.area)), unit: 'm²', help: 'Cake filtration of the softening sludge and seed-slurry solids' });
      X.out.filterArea = sum(filt.map((q) => q.area));
    }
    X.TB.push({ title: 'Process economics', columns: ['Item', 'Capital ($)', 'Annual cost ($/y)', 'Basis'], rows: [...eco.items.map((q) => [q.name, q.cost, null, q.basis]), ['Indirect costs and contingency', eco.capex - eco.direct, null, `${v.capexIndirect ?? 35} % of direct cost`], ['Total capital', eco.capex, eco.capexY, `Capital-recovery factor ${fq(eco.crf, 4)} (${v.rate} %, ${v.life} y)`], ...eco.opex.map(([n, x]) => [n, null, x, '']), ['Total annual cost', null, eco.annual, `${fq(eco.brineY, 4)} m³/y of brine at ${v.avail} % availability`]],
      note: `Levelised cost ${fq(eco.perBrine, 3)} $ per m³ of brine treated, ${fq(eco.perWater, 3)} $ per m³ of water recovered. Installed-cost correlations scale with capacity^${v.capexScale}; treat the result as ± 40 %.` });
    X.PL.push({ type: 'bar', title: 'Annual cost of the train', ylabel: '$/y', categories: ['Capital charge', ...eco.opex.map((q) => q[0])], series: [{ name: '$/y', values: [eco.capexY, ...eco.opex.map((q) => q[1])] }] });
    X.BAL.push({ name: 'Annual cost: capital charge + operating items vs total ($/y, scaled)', in: 1, out: eco.annual !== 0 ? (eco.capexY + sum(eco.opex.map((q) => q[1]))) / eco.annual : 1 });
    X.K.push({ label: 'Levelised cost of brine treatment', value: eco.perBrine, unit: '$/m³ brine', help: `${fq(eco.perWater, 3)} $ per m³ of water recovered` }, { label: 'Capital cost', value: eco.capex / 1e6, unit: 'M$' });
    Object.assign(X.out, { levelisedCostPerM3Brine: eco.perBrine, levelisedCostPerM3Water: eco.perWater, capex: eco.capex, opex: eco.opexY });
    const cf = steps.map((s) => s.cf), on = (id) => path.onset[id]?.cf ?? null, unitLabel = { ro: 'Membrane', bc: 'Brine concentrator', cx: 'Crystallizer' };
    const bounds = ['ro', 'bc'].map((u) => { for (let i = steps.length - 1; i >= 0; i--) if (steps[i].unit === u) return { x: steps[i].cf, label: `${unitLabel[u]} outlet` }; return null; }).filter(Boolean);
    const formed = r.minerals.filter((id) => path.cum[id] > 1e-12), kgm3 = (mol, id) => (mol * MINERALS[id].mw * r.Kw) / 1000 / v.Q;
    const seqOrder = formed.slice().sort((a, b) => (on(a) ?? 1e9) - (on(b) ?? 1e9));
    // MVC energy map: specific compressor energy versus outlet salinity (through BPE) and tube temperature difference
    const fx = linspace(Math.max(40, steps[0].S), Math.max(Math.max(40, steps[0].S) + 20, Math.min(300, Math.max(...steps.map((s) => s.S)))), 12), fy = linspace(1.5, 9, 9), Ss = steps.map((s) => s.S), aws = steps.map((s) => s.aw);
    const mono = Ss.every((x, i) => i === 0 || x > Ss[i - 1]), awAt = (S) => (mono ? interp1(Ss, aws, S) : clamp(1 - 0.00095 * S, 0.3, 1));
    const fz = fy.map((dT) => fx.map((S) => compressorWork(v.Tbc, bpeFromAw(v.Tbc, awAt(S)), dT, v.etaComp / 100) / 3.6e6 * 1000 / (v.etaMotor / 100)));
    const kpis = [
      { label: 'Water recovered', value: r.recovered, unit: 'm³/h' }, { label: 'Overall water recovery', value: recPct, unit: '%', status: recPct > 90 ? 'ok' : 'warn' },
      { label: 'Liquid discharge', value: r.liquid, unit: 'm³/h', status: zld ? 'ok' : 'warn', help: zld ? 'Zero liquid discharge achieved' : 'Mother-liquor purge sent to liquid disposal' },
      { label: 'Total dry solids', value: td(r.solidsTotal), unit: 't/d' }, { label: `${MINERALS[r.main]?.name || 'Salt'} product`, value: td(r.prod), unit: 't/d' }, { label: 'Product-cake purity', value: 100 * r.purity, unit: '%', status: r.prod <= 0 || r.purity >= v.purityMin / 100 ? 'ok' : 'warn', help: 'Main salt divided by all dry solids in the centrifuge cake' },
      { label: 'Electric power', value: r.kWe, unit: 'kW' }, { label: 'Thermal power', value: r.kWt, unit: 'kW' },
      { label: 'Electricity per m³ recovered', value: r.recovered > 0 ? r.kWe / r.recovered : 0, unit: 'kWh/m³' }, { label: 'Heat per m³ recovered', value: r.recovered > 0 ? r.kWt / r.recovered : 0, unit: 'kWh/m³' },
      { label: 'Halite onset', value: on('halite') != null ? on('halite') : 'not reached', unit: on('halite') != null ? '× feed' : '', help: 'Concentration factor (water basis) at which sodium chloride starts to crystallise' },
      { label: 'CaSO₄ onset', value: Math.min(on('gypsum') ?? Infinity, on('anhydrite') ?? Infinity) < Infinity ? Math.min(on('gypsum') ?? Infinity, on('anhydrite') ?? Infinity) : 'not reached', unit: '× feed' },
      { label: 'Concentrator BPE', value: bc.bpe, unit: 'K' }, { label: 'Crystallizer BPE', value: cx.bpe, unit: 'K', status: v.cxDrive === 'mvr' && cx.bpe > v.bpeMax ? 'bad' : 'ok' },
      { label: 'Concentrator area', value: bc.area, unit: 'm²' }, { label: 'Concentrator specific energy', value: bc.mv > 0 ? (bc.kWe * 1000) / bc.mv : 0, unit: 'kWh/t', help: 'Electricity per tonne of distillate' },
      { label: 'Crystal median size L50', value: csd ? csd.fv.L50 * 1e6 : 0, unit: 'µm' }, { label: 'Crystallizer volume', value: csd ? csd.volume : 0, unit: 'm³' },
      { label: 'Salt revenue less disposal', value: rev, unit: '$/d', help: 'Indicative: saleable salts at the entered prices minus disposal of mixed solids' },
      ...X.K,
    ];
    const stream = (label, st) => [label, ...['Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3'].map((k) => st.ions[k]), st.tds / 1000, st.S, st.pH, st.rho, st.aw, bpeFromAw(st.unit === 'cx' ? v.Tcx : st.unit === 'bc' ? r.Tb : 100, st.aw), st.cf];
    const lastOf = (u) => { for (let i = steps.length - 1; i >= 0; i--) if (steps[i].unit === u) return steps[i]; return null; };
    const out = {
      waterRecovered: r.recovered, solids: td(r.solidsTotal), secElec: r.recovered > 0 ? r.kWe / r.recovered : 0, secThermal: r.recovered > 0 ? r.kWt / r.recovered : 0, power: r.kWe, heat: r.kWt, salts: saltsOut, liquidDischarge: r.liquid,
      recovery: recPct / 100, zld, saltPurity: r.purity, haliteOnsetCF: on('halite') ?? 0, productSalt: MINERALS[r.main]?.name || 'none', productSaltTpd: td(r.prod), L50um: csd ? csd.fv.L50 * 1e6 : 0, cv: csd ? csd.fv.cv : 0, crystallizerVolume: csd ? csd.volume : 0, bcArea: bc.area, bpeBC: bc.bpe, bpeCX: cx.bpe, pondArea: r.pond.area,
      revenuePerDay: rev, energyCostPerDay: energyCost, reagents: Object.fromEntries(Object.entries(r.reagents).map(([k, kg]) => [REAGENTS[k].name, +(kg * 24).toPrecision(5)])), chemicals: sum(Object.values(r.reagents)) * 24, pathSolidsTpd: td(sum(formed.map((id) => (path.cum[id] * MINERALS[id].mw * r.Kw) / 1000))),
      streams: { purge: { Q: r.purgeQ, T: v.Tcx, P: 1, pH: +end.pH.toFixed(3), tds: end.tds, ions: Object.fromEntries(ION_IDS.map((k) => [k, +end.ions[k].toPrecision(6)])) } },
      ...X.out,
    };
    return {
      summary: `${fmt(r.recovered, 3)} m³/h of water is recovered from ${fmt(v.Q, 3)} m³/h of brine (${fmt(recPct, 3)} %), leaving ${fmt(td(r.solidsTotal), 3)} t/d of solids${r.prod > 0 ? ` including ${fmt(td(r.prod), 3)} t/d of ${(MINERALS[r.main]?.name || 'salt').toLowerCase()} at ${fmt(100 * r.purity, 3)} % purity` : ''}; ${zld ? 'no liquid discharge remains' : `${fmt(r.liquid, 3)} m³/h of liquid purge remains`}. Energy: ${fmt(r.kWe, 3)} kW electric and ${fmt(r.kWt, 3)} kW heat.`,
      warnings: W, kpis,
      recommendations: [
        ro.on && v.preconc === 'hpro' && ro.P > v.pMax ? 'Switch the pre-concentration to osmotically assisted RO or lower its target salinity.' : null,
        !ro.on && steps[0].S < 100 ? 'Add a membrane pre-concentration step: every tonne of water removed by membranes instead of evaporation saves roughly 15–20 kWh.' : null,
        r.softSolids.length === 0 && (on('gypsum') ?? on('anhydrite') ?? 99) < (lastOf('ro')?.cf ?? 0) ? 'Calcium sulphate saturates inside the membrane step: soften the brine or rely on a seeded-slurry design.' : null,
        cx.on && v.cxDrive === 'steam' && cx.bpe < 0.6 * v.bpeMax ? 'The crystallizer boiling-point elevation is moderate: a vapour-recompression drive would cut the steam demand.' : null,
        r.purity < v.purityMin / 100 && r.softSolids.length === 0 && r.prod > 0 ? 'For saleable salt, remove magnesium and calcium upstream (selective precipitation) so that less bittern contaminates the cake.' : null,
        v.purgeFate === 'discharge' ? 'Route the purge to a dryer or an evaporation pond to reach true zero liquid discharge.' : null,
        'Use suite 13 (Economics) for the full cost of water including this ZLD train, and suite 6 for effect-by-effect evaporator design.',
      ].filter(Boolean),
      plots: [
        { type: 'line', title: 'Precipitation sequence: cumulative solids along the evaporation path', xlabel: 'Concentration factor relative to feed (water basis)', ylabel: 'Solids formed (kg per m³ of feed brine)', logx: true, logy: true, ymin: 1e-3, series: seqOrder.length ? seqOrder.map((id) => ({ name: MINERALS[id].name, x: cf, y: steps.map((s) => Math.max(1e-4, kgm3(s.cum[id], id))) })) : [{ name: 'no solids form', x: cf, y: cf.map(() => 1e-4) }], vlines: bounds, note: 'Fractional crystallisation: each salt appears where its saturation index reaches zero.' },
        { type: 'line', title: 'Saturation indices along the path', xlabel: 'Concentration factor', ylabel: 'Saturation index', logx: true, ymin: -5, ymax: 2.5, series: TRACK.filter((id) => r.minerals.includes(id) && steps.some((s) => s.SI[id] > -90)).map((id) => ({ name: MINERALS[id].name, x: cf, y: steps.map((s) => clamp(s.SI[id], -5, 2.5)) })), hlines: [{ y: 0, label: 'saturation' }], vlines: bounds },
        { type: 'line', title: 'Dissolved major ions in the liquor', xlabel: 'Concentration factor', ylabel: 'Molality (mol/kg water)', logx: true, logy: true, ymin: 1e-4, series: ['Na', 'Cl', 'Mg', 'K', 'SO4', 'Ca'].map((k) => ({ name: IONS[k].label, x: cf, y: steps.map((s) => Math.max(1e-5, s.m[k])) })), vlines: bounds },
        { type: 'line', title: 'Water activity, salinity and boiling-point elevation', xlabel: 'Concentration factor', ylabel: 'aw × 100 (–) · salinity (g/kg ÷ 10) · BPE (K)', logx: true, series: [{ name: 'Water activity × 100', x: cf, y: steps.map((s) => 100 * s.aw) }, { name: 'Salinity ÷ 10 (g/kg)', x: cf, y: steps.map((s) => s.S / 10) }, { name: 'BPE at 100 °C (K)', x: cf, y: steps.map((s) => bpeFromAw(100, s.aw)), mode: 'both' }], vlines: bounds },
        { type: 'bar', title: 'Where the water goes', ylabel: 't/h', categories: ['Membrane permeate', 'Concentrator distillate', 'Crystallizer condensate', 'Water of hydration in salts', 'Cake moisture (dryer vapour)', v.purgeFate === 'discharge' ? 'Liquid purge' : 'Purge water (evaporated)'], series: [{ name: 'Water', values: [water.perm, water.dist, water.cond, water.hydration, water.cake, water.purge].map((x) => x / 1000) }] },
        { type: 'bar', title: 'Energy by unit', ylabel: 'kW', stacked: true, categories: ['Membranes', 'Brine concentrator', 'Crystallizer', 'Dryers'], series: [{ name: 'Electricity', values: [ro.kW, bc.kWe, cx.kWe, 0] }, { name: 'Heat', values: [0, bc.kWt, cx.kWt, r.dryer.kWt + r.purgeDry.kWt] }] },
        ...(csd ? [
          { type: 'line', title: `Crystal-size distribution of ${MINERALS[csd.salt].name.toLowerCase()} (mass basis)`, xlabel: 'Crystal size L (µm)', ylabel: 'Mass density (% per µm)', series: [
            { name: `Finite-volume PBE (${csd.fv.L.length} cells)`, x: csd.fv.L.map((L) => L * 1e6), y: csd.fv.L.map((L, i) => (100 * csd.fv.n[i] * L ** 3) / Math.max(csd.fv.mu[3], 1e-300) / 1e6), mode: 'both' },
            { name: 'Analytical MSMPR at the final growth rate', x: csd.fv.L.map((L) => L * 1e6), y: csd.fv.L.map((L) => { const g = csd.Gend * csd.tau, z = L / g; return (100 * z ** 3 * Math.exp(-z)) / 6 / g / 1e6; }), dash: true }], vlines: [{ x: csd.fv.L50 * 1e6, label: 'L50' }] },
          { type: 'line', title: 'Crystallizer start-up (method of moments)', xlabel: 'Time (residence times)', ylabel: 'L₄₃ (µm) · magma density (kg/m³) · σ × 10⁴', series: [{ name: 'Mass-mean size L₄₃ (µm)', x: csd.dyn.th, y: csd.dyn.L43.map((x) => x * 1e6) }, { name: 'Magma density (kg/m³)', x: csd.dyn.th, y: csd.dyn.mt }, { name: 'Supersaturation σ × 10⁴', x: csd.dyn.th, y: csd.dyn.sigma.map((x) => x * 1e4) }], hlines: [{ y: csd.ss.L43 * 1e6, label: 'steady L₄₃' }] },
        ] : []),
        { type: 'field', title: 'MVC compressor energy versus brine salinity and tube ΔT', xlabel: 'Brine salinity in the evaporator (g/kg)', ylabel: 'Tube temperature difference (K)', zlabel: 'Compressor electricity', zunit: 'kWh/t', x: fx, y: fy, z: fz, cmap: 'thermal', contours: 8, markers: bc.on && v.bcType === 'mvc' && lastOf('bc') ? [{ x: clamp(lastOf('bc').S, fx[0], fx[fx.length - 1]), y: clamp(v.dT, 1.5, 9), label: 'design' }] : [], note: 'Salinity acts through the boiling-point elevation, which the compressor must overcome in addition to the tube temperature difference.' },
        { type: 'bar', title: 'Solids produced', ylabel: 't/d', categories: r.salts.length ? r.salts.map((s) => `${s.name} · ${s.unit}`) : ['none'], series: [{ name: 't/d', values: r.salts.length ? r.salts.map((s) => td(s.kg)) : [0] }] },
        ...X.PL,
      ],
      tables: [
        { title: 'Unit operations', columns: ['Unit', 'Water in (t/h)', 'Water removed (t/h)', 'Outlet liquor (m³/h)', 'Outlet salinity (g/kg)', 'Concentration factor', 'Temperature (°C)', 'BPE (K)', 'Electricity (kW)', 'Heat (kW)', 'Heat-transfer area (m²)', 'Solids formed (t/d)'],
          rows: r.units.length ? r.units.map((u) => [u.name, u.feedW / 1000, u.removed / 1000, (r.Kw * u.out.w * (1 + u.out.gPerKgw / 1000)) / u.out.rho, u.out.S, u.out.cf, u.T, u.bpe, u.kWe, u.kWt, u.area, td(u.solids)]) : [['No concentration unit is active', 0, 0, v.Q, steps[0].S, 1, v.T, null, 0, 0, null, 0]],
          note: `${ro.on ? (ro.ed ? `Electrodialysis: ${fmt(ro.ed.U, 3)} V per cell pair, ${fmt(ro.ed.area, 3)} m² of cell pairs, outlet osmotic pressure ${fmt(ro.pi, 3)} bar. ` : `Membrane step: ${fmt(ro.P, 3)} bar, outlet osmotic pressure ${fmt(ro.pi, 3)} bar, reversible work ${fmt(ro.wLeast, 3)} kW. `) : ''}${bc.on && v.bcType === 'mvc' ? `MVC: compressor shaft power ${fmt(bc.comp, 4)} kW, auxiliary heat ${fmt(bc.aux, 3)} kW, surplus heat ${fmt(bc.vent, 3)} kW. ` : ''}${bc.md ? `Membrane distillation: mean flux ${fmt(bc.md.flux, 3)} kg/m²·h at ${fmt(r.Tb, 3)} °C feed, thermal efficiency ${fmt(100 * bc.md.eta, 3)} %. ` : ''}${bc.on && (v.bcType === 'mee' || v.bcType === 'tvc') ? `Steam economy ${fmt(bc.econ, 3)} kg/kg, heating steam ${fmt(bc.steam, 3)} t/h. ` : ''}Dryer heat: ${fmt(r.dryer.kWt + r.purgeDry.kWt, 3)} kW.` },
        { title: 'Salts and solids', columns: ['Solid', 'Formula', 'Onset (× feed)', 'Formed in', 't/d', 'Destination', 'Value ($/t)', 'Value ($/d)'], rows: r.salts.length ? r.salts.map((s) => [s.name, s.formula, s.onset, s.unit, td(s.kg), s.dest, s.value, td(s.kg) * s.value]) : [['No solids', '', null, '', 0, '', 0, 0]],
          note: `Indicative daily figures: salts ${fmt(rev, 3)} $/d, energy ${fmt(-energyCost, 3)} $/d, reagents ${fmt(-reagCost, 3)} $/d (caustic 450, lime 130, soda ash 280, HCl 180 $/t).${Object.keys(r.reagents).length ? ' Reagent use: ' + Object.entries(r.reagents).map(([k, kg]) => `${REAGENTS[k].name} ${fmt(kg, 3)} kg/h`).join(', ') + '.' : ''}` },
        { title: 'Liquor along the train', columns: ['Stream', 'Na (mg/L)', 'K (mg/L)', 'Ca (mg/L)', 'Mg (mg/L)', 'Cl (mg/L)', 'SO₄ (mg/L)', 'HCO₃ (mg/L)', 'TDS (g/L)', 'Salinity (g/kg)', 'pH', 'Density (kg/m³)', 'Water activity', 'BPE (K)', 'Concentration factor'],
          rows: [stream(r.softSolids.length ? 'Softened feed' : 'Feed brine', steps[0]), ...['ro', 'bc', 'cx'].map((u) => (lastOf(u) ? stream({ ro: 'Membrane concentrate', bc: 'Concentrator blowdown', cx: 'Mother liquor / purge' }[u], lastOf(u)) : null)).filter(Boolean)] },
        { title: 'Crystallizer population balance', columns: ['Quantity', 'Finite-volume PBE', 'Method of moments', 'Analytical MSMPR', 'Unit'], rows: csd ? [
          ['Median size L50 (mass)', csd.fv.L50 * 1e6, null, 3.67206 * csd.Gend * csd.tau * 1e6, 'µm'], ['Mass-mean size L₄₃', csd.fv.L43 * 1e6, csd.dyn.L43[csd.dyn.L43.length - 1] * 1e6, 4 * csd.Gend * csd.tau * 1e6, 'µm'], ['L10 / L90', `${fmt(csd.fv.L10 * 1e6, 3)} / ${fmt(csd.fv.L90 * 1e6, 3)}`, null, `${fmt(1.745 * csd.Gend * csd.tau * 1e6, 3)} / ${fmt(6.681 * csd.Gend * csd.tau * 1e6, 3)}`, 'µm'],
          ['Coefficient of variation', csd.fv.cv, csd.dyn.cv[csd.dyn.cv.length - 1], 0.5, '–'], ['Magma density', csd.fv.mt, csd.dyn.mt[csd.dyn.mt.length - 1], csd.MT, 'kg/m³'], ['Crystal number', csd.fv.mu[0], csd.dyn.mu0[csd.dyn.mu0.length - 1], csd.Bend * csd.tau, '#/m³'],
          ['Growth rate G', null, csd.Gend * 1e9, csd.ss.G * 1e9, 'nm/s'], ['Nucleation rate B', null, csd.Bend, csd.ss.B, '#/m³·s'], ['Relative supersaturation σ', null, csd.sigEnd, csd.ss.sigma, '–'],
          ['Active volume', csd.volume, null, null, 'm³'], ['Slurry draw-off', csd.slurryQ, null, null, 'm³/h'], ['Solids volume fraction', csd.solidVol, null, null, '–'], ['Saturation concentration used', csd.cstar, null, null, 'kg/m³']] : [['Crystallizer not active', null, null, null, '']],
          note: csd ? `Product: ${MINERALS[csd.salt].name}. ${csd.fv.nSteps} time steps on ${csd.fv.L.length} size cells; population leaving through the upper size boundary: ${fmt(100 * csd.fv.tail, 2)} %.` : '' },
        { title: 'Final water and solids balance', columns: ['Item', 't/h', 'Share of feed water (%)', 'Fate'], rows: [
          ['Water in feed brine', water.in / 1000, 100, 'In'], ['Membrane permeate', water.perm / 1000, (100 * water.perm) / water.in, 'Recovered water'], ['Concentrator distillate', water.dist / 1000, (100 * water.dist) / water.in, 'Recovered water'], ['Crystallizer condensate', water.cond / 1000, (100 * water.cond) / water.in, 'Recovered water'],
          ['Water of hydration', water.hydration / 1000, (100 * water.hydration) / water.in, 'Bound in solids'], ['Cake moisture', water.cake / 1000, (100 * water.cake) / water.in, 'Vapour from the salt dryer'], ['Purge water', water.purge / 1000, (100 * water.purge) / water.in, { dryer: 'Vapour from the purge dryer', pond: `Evaporation pond (${fmt(r.pond.area / 1e4, 3)} ha)`, discharge: 'LIQUID DISCHARGE' }[v.purgeFate]],
          ['Dissolved salts in feed', (r.Kw * r.feed.w * r.io0.gPerKgw) / 1e6, null, 'In'], ['Dry solids out', r.solidsTotal / 1000, null, 'Products and waste'], ['Dissolved salts in liquid purge', v.purgeFate === 'discharge' ? r.purgeSalt / 1000 : 0, null, 'Liquid disposal']],
          note: zld ? 'Zero liquid discharge is confirmed: no liquid stream leaves the battery limit.' : 'A liquid stream remains — see the purge row.' },
        ...X.TB,
      ],
      balances: (() => {
        const sol = (obj, k) => sum(Object.entries(obj || {}).map(([id, mol]) => Math.max(0, mol) * (MINERALS[id].stoich[k] || 0)));
        const softCl = (r.soft.reagents.hcl || 0), softNa = (r.soft.reagents.naoh || 0) + 2 * (r.soft.reagents.soda || 0), sNa = sum(r.soft.stages.map((s) => sol(s.solids, 'Na')));
        return [
          { name: 'Water (t/h)', in: water.in / 1000, out: (water.perm + water.dist + water.cond + water.hydration + water.cake + water.purge) / 1000 },
          { name: 'Chloride (kmol/h)', in: ((r.feed.n[iOf.Cl] + softCl) * r.Kw) / 1000, out: ((path.sol.n[iOf.Cl] + sol(path.cum, 'Cl')) * r.Kw) / 1000 },
          { name: 'Sodium (kmol/h)', in: ((r.feed.n[iOf.Na] + softNa) * r.Kw) / 1000, out: ((path.sol.n[iOf.Na] + sol(path.cum, 'Na') + sNa) * r.Kw) / 1000 },
          { name: 'Sulphate (kmol/h)', in: (r.feed.n[iOf.SO4] * r.Kw) / 1000, out: ((path.sol.n[iOf.SO4] + sol(path.cum, 'SO4') + sum(r.soft.stages.map((s) => sol(s.solids, 'SO4')))) * r.Kw) / 1000 },
          { name: 'Magnesium (kmol/h)', in: (r.feed.n[iOf.Mg] * r.Kw) / 1000, out: ((path.sol.n[iOf.Mg] + sol(path.cum, 'Mg') + sum(r.soft.stages.map((s) => sol(s.solids, 'Mg')))) * r.Kw) / 1000 },
          ...(bc.balance ? [{ name: 'Brine-concentrator energy (kW)', in: bc.balance.in, out: bc.balance.out }] : []),
          ...X.BAL,
          ...(csd ? [{ name: 'Crystal + dissolved excess mass in the crystallizer (kg/m³): analytical transient vs population balance', in: csd.MT + (csd.dyn.mt[0] + csd.dyn.sigma[0] * csd.cstar - csd.MT) * Math.exp(-csd.dyn.th[csd.dyn.th.length - 1]), out: csd.dyn.mt[csd.dyn.mt.length - 1] + csd.sigEnd * csd.cstar }] : []),
        ];
      })(),
      outputs: out,
    };
  },

  mesh: [
    { name: 'Crystal-size grid of the population balance', keys: ['nL'], min: 10, note: 'The size grid of the finite-volume population balance is refined at fixed kinetics and domain length.', metrics: [{ label: 'Median crystal size L50', unit: 'µm', get: (r) => r.outputs.L50um }, { label: 'Coefficient of variation', unit: '–', get: (r) => r.outputs.cv }] },
    { name: 'Evaporation steps along the concentration path', keys: ['nEvap'], min: 6, metrics: [{ label: 'Halite onset concentration factor', unit: '×', get: (r) => r.outputs.haliteOnsetCF }, { label: 'Solids crystallised along the path', unit: 't/d', get: (r) => r.outputs.pathSolidsTpd }] },
  ],

  calibration: {
    note: 'Fit the crystallizer kinetics to measured product size. Each row is one steady operating period with its residence time and magma density; the measurement is the mass-median crystal size from sieve or laser-diffraction analysis. Data spanning both residence time and slurry density are needed to separate the nucleation constant from the magma-density exponent.',
    params: [{ key: 'kb', label: 'Secondary-nucleation constant kb', lo: 1e5, hi: 1e11 }, { key: 'jMT', label: 'Magma-density exponent j', lo: 0, hi: 2 }],
    columns: [{ key: 'tauX', label: 'Residence time', unit: 'h' }, { key: 'mtX', label: 'Magma density', unit: 'kg/m³' }, { key: 'L50', label: 'Median crystal size', unit: 'µm' }],
    targets: [{ key: 'L50', label: 'Median crystal size', unit: 'µm' }],
    model: (v) => ({ L50: msmprSteady({ kg: v.kg, g: v.gExp, kb: v.kb, b: v.bExp, j: v.jMT, kv: v.kv, rhoc: MINERALS.halite.rho }, (v.tauX ?? v.tau) * 3600, v.mtX ?? v.MT).L50 * 1e6 }),
    get sample() { return (this._s ||= synth(3, [[0.75, 150], [1, 250], [1.5, 120], [1.5, 250], [1.5, 380], [2, 200], [2.5, 300], [3, 150]])); },
    get validationSample() { return (this._v ||= synth(23, [[0.9, 320], [1.2, 180], [1.8, 340], [2.2, 130], [2.8, 260], [3.5, 220]])); },
  },

  verify() {
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    const d = D(), r = simulateZLD(d), w = r.water;
    add('Water balance of the train closes', 0, (w.in - w.perm - w.dist - w.cond - w.hydration - w.cake - w.purge) / w.in, 1e-9, 'Feed water = recovered + hydration + cake moisture + purge');
    const clS = sum(Object.entries(r.path.cum).map(([id, mol]) => mol * (MINERALS[id].stoich.Cl || 0)));
    add('Chloride balance closes', 0, (r.feed.n[iOf.Cl] - r.path.sol.n[iOf.Cl] - clS) / r.feed.n[iOf.Cl], 1e-9, 'Dissolved in feed = in salts + in mother liquor');
    // seawater at 25 °C: onset of gypsum and halite
    const sw = makeSolution({ ions: WATERS.seawater.ions, T: 25, pH: 8.1 }), p = evaporationPath(sw, [{ unit: 'cx', wEnd: sw.w / 14, T: 25, precip: true, pCO2: 4.2e-4 }], { nEvap: 26, minerals: ['calcite', 'gypsum', 'anhydrite', 'halite'] });
    add('Seawater evaporation: gypsum appears at a concentration factor of about 3.8', 3.8, p.onset.gypsum?.cf ?? 0, 0.4, 'Harvie–Møller–Weare sequence at 25 °C (literature 3.3–3.8)');
    add('Seawater evaporation: halite appears at a concentration factor of about 10.8', 10.8, p.onset.halite?.cf ?? 0, 0.5, 'Literature 10.6–11');
    // pure NaCl solution: analytical halite yield
    const n = new Float64Array(sw.n.length); n[iOf.Na] = 1; n[iOf.Cl] = 1;
    const q = evaporationPath(equilibrate({ T: 25, model: 'pitzer', n, alk: 0, w: 1, pH: 7 }), [{ unit: 'cx', wEnd: 0.05, T: 25, precip: true }], { nEvap: 10, minerals: ['halite'] });
    add('Pure NaCl brine: crystallised mass equals feed minus saturated remainder', 1 - 6.0947 * 0.05, q.cum.halite, 2e-3, '1 mol NaCl per kg water evaporated to 5 % of the water; solubility 6.095 mol/kg');
    const sat = evaporationPath(equilibrate({ T: 25, model: 'pitzer', n, alk: 0, w: 1, pH: 7 }), [{ unit: 'cx', wEnd: 0.15, T: 25, precip: true }], { nEvap: 6, minerals: ['halite'] });
    add('Boiling-point elevation of saturated NaCl brine at 1 atm', 8.7, bpeFromAw(108.5, sat.sol.eq.aw), 1.0, 'Literature: saturated sodium-chloride solution boils at about 108.7 °C');
    // MSMPR benchmarks
    if (r.csd) {
      const c = r.csd, g = c.Gend * c.tau;
      add('Population balance: median size equals 3.67·G·τ', 1, c.fv.L50 / (3.67206 * g), 0.03, 'Analytical MSMPR mass distribution (finite-volume grid)');
      add('Population balance: coefficient of variation equals 0.5', 0.5, c.fv.cv, 0.03, 'Analytical MSMPR result for size-independent growth');
      add('Population balance: crystal number equals B·τ', 1, c.fv.mu[0] / (c.Bend * c.tau), 0.03, 'Number conservation: nuclei born = crystals withdrawn');
      add('Moments reach the analytical steady state', 1, c.dyn.L43[c.dyn.L43.length - 1] / c.ss.L43, 0.01, 'L₄₃ = 4·G·τ with G from the closed-form steady state');
      add('Crystal mass balance of the crystallizer', 1, (c.dyn.mt[c.dyn.mt.length - 1] + c.sigEnd * c.cstar) / (c.MT + (c.dyn.mt[0] + c.dyn.sigma[0] * c.cstar - c.MT) * Math.exp(-c.dyn.th[c.dyn.th.length - 1])), 1e-4, 'Crystals + dissolved excess follow M(t) = MT + (M₀ − MT)·e^(−t/τ) exactly; MT = production × residence time / volume');
    }
    const ws = compressorWork(100, r.bc.bpe, d.dT, 1), wmin = (latentHeat(100) * (r.bc.bpe + d.dT)) / (100 + KELVIN + d.dT);
    add('Compressor work is close to and above the Carnot heat-pump limit', 1, ws > 0.9 * wmin && ws < 1.35 * wmin ? 1 : 0, 0, `Isentropic ${fmt(ws / 1000, 4)} kJ/kg versus λ·ΔT/T = ${fmt(wmin / 1000, 4)} kJ/kg`);
    if (r.bc.balance) add('Brine-concentrator energy balance closes', 0, (r.bc.balance.in - r.bc.balance.out) / r.bc.balance.in, 1e-9, 'Feed enthalpy + compressor work + auxiliary heat = distillate + blowdown + losses');
    const dis = simulateZLD({ ...d, purgeFate: 'discharge', nEvap: 8 });
    add('Purge routed to a dryer leaves no liquid discharge', 0, r.liquid, 0, 'ZLD limiting case');
    add('Purge routed to disposal appears as liquid discharge', 1, dis.liquid > 0 ? 1 : 0, 0, 'Complementary case');
    const coarse = simulateZLD({ ...d, nEvap: 16 });
    // Ostwald ripening: the Lifshitz–Slyozov–Wagner self-similar distribution must coarsen with the LSW constant
    const Ar = 1e-21, Kr = (4 / 9) * Ar, r0 = 1e-7, lswPdf = (q) => (q < 1.5 ? ((q * q) / ((q + 3) ** (7 / 3) * (1.5 - q) ** (11 / 3))) * Math.exp(-q / (1.5 - q)) : 0);
    const rp = ripen(Array.from({ length: 300 }, (_, i) => ({ N: lswPdf((1.5 * (i + 0.5)) / 300), r: (1.5 * (i + 0.5) * r0) / 300 })), Ar, (7 * r0 ** 3) / Kr, { maxSteps: 20000 }), le = rp.t.length - 1;
    add('Ostwald ripening follows the LSW law ⟨r⟩³ = r₀³ + K·t', 1, (rp.rMean[le] ** 3 - rp.rMean[0] ** 3) / rp.t[le] / Kr, 0.05, 'Self-similar LSW size distribution of 300 classes, mean radius doubles; K = (4/9)·D·c∞·Vm·α');
    add('Ripening: crystal number falls as 1/⟨r⟩³ at constant volume', 1, (rp.number[le] / rp.number[0]) * (rp.rMean[le] / rp.rMean[0]) ** 3, 0.03, 'Self-similar coarsening; the dissolved small crystals feed the large ones');
    add('LSW constant (hand calculation)', (8 * 0.038 * 5000 * 2.7e-5 ** 2 * 1.5e-9) / (9 * 2 * R * 353.15), lswConstant(1.5e-9, 5000, 2.7e-5, capillaryLength(0.038, 2.7e-5, 80, 2)), 1e-26, 'K = 8γ·c∞·Vm²·D/(9νRT) for γ = 38 mJ/m², c∞ = 5 kmol/m³, 80 °C');
    add('Gibbs–Thomson: a crystal of the critical size is in equilibrium', 0.004, sigmaCrit((2 * 3.5e-10) / Math.log1p(0.004), 3.5e-10), 1e-12, 'σ*(L) = exp(2α/L) − 1 evaluated at L* = 2α/ln(1 + σ)');
    add('Noyes–Whitney dissolution velocity (hand calculation)', (2 * 2 * 1.5e-9 * 300 * 0.01) / (2165 * 1e-5), dissolutionVelocity(1e-5, 0.01, 300, 2165), 1e-15, 'dL/dt = 2·Sh·D·c*·Δσ/(ρc·L) for a 10 µm crystal, 1 % undersaturation');
    // two-zone population balance
    const Kz = { kg: 5e-6, g: 1, kb: 1e8, b: 2, j: 1, kv: 1, rhoc: 2165, cstar: 300, primA: 1e30, primB: 1e9 }, tz = 5400, sz = { mass: 20, L: 1e-4, sigma: 0.005 }, ssz = msmprSteady(Kz, tz, 250), og = { gt: false, nL: 400, tEnd: 16, Lmax: 16 * ssz.G * tz, Lmin: 5e-6 };
    const zm = zoneCrystallizer(Kz, tz, 250, sz, { ...og, theta: 1e-3 });
    add('Two-zone model, fast circulation and no size effects: analytical MSMPR median size', 1, zm.product.L50 / ssz.L50, 0.03, 'L50 = 3.67·G·τ; geometric grid of 400 cells, first-order upwind');
    add('…and the transient solver reaches the closed-form steady state of the same grid', 1, zm.product.L50 / zm.reference.L50, 0.005, 'Implicit block-tridiagonal time stepping versus the direct steady solution');
    add('No-flux wall: crystals leave only with the product (number balance)', 1, zm.product.mu[0] / (kinetics(Kz, zm.sigma[1], zm.product.mt).B * tz), 0.01, 'Crystal number = nucleation rate × residence time; nothing is lost at the wall');
    add('Two-zone model: salt balance closes', 0, (zm.balance.in - zm.balance.out) / zm.balance.in, 1e-8, 'Salt generated = crystals + dissolved excess + product + wall deposit');
    const zw = zoneCrystallizer(Kz, tz, 250, sz, { ...og, theta: 1e-3, kw: 1 / tz });
    add('Wall deposition shortens the crystal life: L₄₃ ∝ G/(1/τ + k_w)', (zw.G[1] / (2 / tz)) / (zm.G[1] * tz), zw.product.L43 / zm.product.L43, 0.01, 'Deposition rate constant equal to the draw-off rate, compared with the no-flux wall');
    add('…and the wall then takes half of the crystal output', 1, (zw.wallRate * tz) / zw.product.mt, 0.01, 'Deposition flux k_w·M_T equals the product flux M_T/τ');
    const z2 = zoneCrystallizer(Kz, tz, 250, sz, { gt: true, alpha: 3.5e-10, nL: 160, tEnd: 12, Lmax: 16 * ssz.G * tz, theta: 60, phi1: 0.15 });
    add('Finite circulation: the boiling zone is more supersaturated than the body', 1, z2.sigma[0] > 1.5 * z2.sigma[1] && Math.abs(zm.sigma[0] / zm.sigma[1] - 1) < 1e-3 ? 1 : 0, 0, `σ₁/σ₂ = ${fmt(z2.sigma[0] / z2.sigma[1], 3)} at a 60 s turnover; 1.000 in the fast-circulation limit`);
    const zd = zoneCrystallizer(Kz, tz, 1e-9, { mass: 20, L: 2e-5, sigma: -0.05 }, { gt: false, nL: 160, tEnd: 0.2, Lmax: 2e-3, theta: 1e-3, nt: 200 });
    add('Dissolution: seed crystals in an undersaturated liquor dissolve until it is saturated', 1, zd.product.mt < 0.6 * 20 * Math.exp(-0.2) && zd.sigma[1] > -0.02 && zd.sigma[1] <= 1e-6 && Math.abs(zd.balance.in / zd.balance.out - 1) < 1e-8 ? 1 : 0, 0, `20 kg/m³ of 20 µm seed at σ = −5 %: ${fmt(zd.product.mt, 3)} kg/m³ left after 0.2 residence times (washout alone would leave ${fmt(20 * Math.exp(-0.2), 3)}), σ rises to ${fmt(zd.sigma[1], 2)}`);
    // membrane distillation and electrodialysis
    const mdp = { Tf: 70, Tp: 25, B: 0.7 / 3.6e6, tpc: 1, hm: 0 }, md1 = mdUnit([{ w: 2, aw: 1 }, { w: 1, aw: 1 }], 3600, mdp);
    add('Membrane distillation: pure-water flux from the vapour-pressure difference', (0.7 * (psat(70) - psat(25))) / 1000, md1.flux, 1e-9, 'J = B·(psat(70 °C) − psat(25 °C)), kg/m²·h, no polarisation');
    add('Membrane distillation: without conduction all heat is latent', 1, md1.eta, 1e-12, 'Thermal efficiency = 1 when the conduction coefficient is zero');
    const awS = psat(25) / psat(70), md0 = mdUnit([{ w: 2, aw: awS }, { w: 1, aw: awS }], 3600, mdp);
    add('Membrane distillation: the flux vanishes when aw·psat(feed) = psat(permeate)', 0, md0.jMax, 1e-9, `Water activity ${fmt(awS, 3)}: the limit of concentration by membrane distillation`);
    const edh = edUnit({ eq: 1, cd: 100, cc: 100, T: 25, i: 300, eta: 1, rcp: 1 / 300, tw: 10 });
    add('Electrodialysis: Faraday law', 96.48533212, edh.kW, 1e-6, '1 mol/s of charge at 1 V and 100 % current efficiency = 96.485 kW');
    add('Electrodialysis: concentration ceiling from water transport', 1 / (10 * 0.0180153), edh.mMax, 1e-9, '10 mol of water per equivalent → 5.55 eq/kg');
    const mdz = simulateZLD({ ...d, preconc: 'ed', bcType: 'mdc', cxDrive: 'md', Tcx: 70, nEvap: 12 }), wz = mdz.water;
    add('Electrodialysis → membrane-distillation concentrator → membrane-distillation crystallizer: water balance closes', 0, (wz.in - wz.perm - wz.dist - wz.cond - wz.hydration - wz.cake - wz.purge) / wz.in, 1e-9, 'Alternative train with the same evaporation path');
    add('…and its heat demand is at least the latent duty after heat recovery', 1, mdz.bc.kWt >= ((mdz.bc.mv / 3600) * latentHeat(61) * 0.5) / 1000 && mdz.ro.ed.area > 0 ? 1 : 0, 0, 'Conduction through the membrane adds to the latent heat');
    // filtration and economics
    const fa = ruthArea({ V: 2, c: 80, alpha: 5e11, Rm: 0, dP: 6e5, t: 1800 });
    add('Cake filtration (Ruth equation, hand calculation)', 2 * Math.sqrt((1e-3 * 5e11 * 80) / (2 * 6e5 * 1800)), fa, 1e-9, 'A = V·√(μαc/(2ΔP·t)) without cloth resistance');
    add('Filter area reproduces the cycle time including the cloth resistance', 1800, ruthTime({ V: 2, c: 80, alpha: 5e11, dP: 6e5, A: ruthArea({ V: 2, c: 80, alpha: 5e11, dP: 6e5, t: 1800 }) }), 1e-6, 't = μαc·V²/(2A²ΔP) + μR_m·V/(AΔP)');
    add('Capital-recovery factor', 0.101852, crf(0.08, 20), 1e-6, 'i(1+i)ⁿ/((1+i)ⁿ − 1) for 8 % and 20 years');
    const ec = zldEconomics(d, r, { rev: 0, reagCost: 0 });
    add('Levelised cost equals annual cost over brine treated', ec.annual / (d.Q * 8760 * 0.92), ec.perBrine, 1e-9, 'Capital charge + operating cost, 92 % availability');
    add('Salt yield is insensitive to the number of evaporation steps (16 → 32)', 0, Math.abs((coarse.cxSolids.halite || 0) - (r.cxSolids.halite || 0)) / Math.max(1e-9, r.cxSolids.halite || 0), 0.01, 'Relative change of the halite production');
    return C;
  },
};

const HELP = {
  Q: 'Volume flow of the brine entering the train.', T: 'Temperature of the incoming brine.', Tbc: 'Brine temperature in the evaporator; for multi-effect units the temperature of the first effect.', heatLoss: 'Radiation and vent losses of the evaporator.', pH: 'Measured pH of the brine.', softMode: 'Reagent softening ahead of the membranes and evaporator.', alkali: 'Caustic soda gives a pure magnesium hydroxide; lime is cheaper but adds calcium.',
  sodaRatio: 'Slight excess over the calcium left after the magnesium stage.', softNeutral: 'Acid is dosed until this pH before the membranes and evaporator.', pMax: 'Highest feed pressure the elements and vessels allow.', erdEff: 'Pressure-exchanger efficiency on the concentrate.', etaPump: 'Hydraulic efficiency of the high-pressure pump.', oaroP: 'Feed pressure of each osmotically assisted stage.',
  U: 'Condensing vapour to boiling brine, including fouling.', etaComp: 'Used for the brine-concentrator compressor and the crystallizer recompressor.', nEff: 'More effects save steam but need more temperature difference.', Tlast: 'Set by the cooling-water temperature.', tvcRa: 'Entrained vapour per kg of motive steam.', steamT: 'Saturation temperature of the heating steam.',
  bcSolids: 'Whether the calcium sulphate and carbonate seed solids are filtered out or follow the brine into the crystallizer.', Tcx: 'Operating temperature of the crystallizer body.', tau: 'Active volume divided by the slurry draw-off; longer times give larger crystals.', cakeMoist: 'Mother liquor left on the crystals after the centrifuge.',
  purgeFate: 'Where the mother-liquor purge with the highly soluble salts goes.', pondEvap: 'Annual open-water evaporation at the site.', pondRain: 'Annual rainfall falling on the pond.', pondRH: 'Evaporation stops when the water activity of the liquor equals the relative humidity.',
  elecPrice: 'Used for the energy cost.', steamPrice: 'Per MWh of heat delivered.', priceNaCl: 'Ex-works price of saleable salt.', purityMin: 'Below this purity the salt is valued at a quarter of the price.', priceMg: 'Per tonne of Mg(OH)₂.', disposal: 'Landfill or haulage of mixed salts and sludge.',
  salts: 'The simple set skips the double salts of the Mg–K–SO₄ system.', bcPumps: 'Electricity per tonne of distillate.', cxPumps: 'Electricity per tonne of evaporated water.', Ufc: 'Forced-convection heater of the crystallizer.', dTcx: 'Heating medium minus slurry temperature.', bpeMax: 'Above this boiling-point elevation recompression is not economic.',
  etaMotor: 'Applied to compressors and pumps.', etaDry: 'Heat used for evaporation divided by heat supplied.', gExp: '1 for diffusion- or first-order surface-controlled growth.', kb: 'Nuclei formed per kg of crystals and second at unit supersaturation.', bExp: 'Sensitivity of nucleation to supersaturation.', jMT: '1 when crystal–impeller and crystal–crystal contacts dominate.',
  logA: 'Kinetic prefactor of classical nucleation theory.', het: '1 = homogeneous nucleation; lower values for nucleation on foreign surfaces.', seedMass: 'Crystals charged at start-up per m³ of slurry.', seedL: 'Size of the seed crystals.', sigma0: 'Supersaturation of the liquor at start-up.', tEnd: 'About 10 residence times are needed to reach steady state.',
  nL: 'Uniform cells of the reference population balance; the two-zone model uses 1.25 times as many (at least 100) on a geometric grid.', scheme: 'The limiter removes most of the numerical diffusion of the upwind scheme.',
};
for (const g of suite.inputs) for (const f of g.fields) if (!f.help && HELP[f.key]) f.help = HELP[f.key];

/** Synthetic plant data: the steady-state MSMPR model with different true kinetics plus deterministic noise. */
function synth(seed, pts) {
  const d = D(), g = rng(seed);
  return pts.map(([tauX, mtX]) => ({ tauX, mtX, L50: +(suite.calibration.model({ ...d, kb: 2.4e8, jMT: 0.75, tauX, mtX }).L50 * (1 + g.normal(0, 0.02))).toFixed(1) }));
}

export default suite;
