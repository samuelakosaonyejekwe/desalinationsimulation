// Suite 9 — Brine concentration, crystallization and zero liquid discharge.
// A brine is softened (optional selective precipitation), pre-concentrated by high-pressure or
// osmotically assisted RO, evaporated in a falling-film brine concentrator (MVC, MEE or MEE-TVC) and
// taken to solids in a forced-circulation crystallizer with centrifuge, dryer and purge handling.
// The evaporation path is followed stepwise with Pitzer-based mineral equilibria (suite 2 engine), so the
// order, onset and mass of every salt are predicted; the crystallizer is an MSMPR population balance
// solved by the method of moments and by a finite-volume discretisation of the size coordinate.
import { brent, clamp, linspace, sum, rng, fmt, rk45, interp1 } from '../core/num.js';
import { psat, tsat, latentHeat, cp, R, KELVIN } from '../core/props.js';
import { IONS, ION_IDS, WATERS, scaleIons } from '../core/water.js';
import { MINERALS, EVAPORITE_MINERALS, REAGENTS, ACTIVITY_MODELS, makeSolution, equilibrate, precipitateSolution, doseSolution, solutionToIons, saturationIndex, componentIndex } from './s02_chem.js';

const MW_W = 0.0180153, RHO_W = 997, KB = 1.380649e-23, NA = 6.02214076e23;
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

const D = () => Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));

/** Complete ZLD train. All flows in kg/h or m³/h; solids in kg/h. */
export function simulateZLD(v) {
  const W = [], model = ACTIVITY_MODELS[v.model] ? v.model : 'pitzer';
  const feed = makeSolution({ ions: v.ions, T: v.T, pH: v.pH, model }), io0 = solutionToIons(feed), Kw = (v.Q * 1000 * io0.kgwPerL) / feed.w; // kg/h of water per unit of basis water
  const minerals = v.salts === 'simple' ? SIMPLE_SET : EVAPORITE_MINERALS;
  const soft = soften(feed, v, W), s1 = soft.sol, g1 = solutionToIons(s1).gPerKgw * s1.w;
  const wFor = (S) => (g1 * (1000 / clamp(S, 1, 900) - 1)) / 1000; // water inventory at which the dissolved salts reach S g/kg (before precipitation)
  const wRO = v.preconc === 'none' ? s1.w : Math.min(s1.w, wFor(v.roTDS)), wBC = Math.min(wRO, wFor(v.bcTDS)), wCX = wBC * clamp(v.purge / 100, 0.002, 1);
  const Tc = (T) => (v.chemT === 'unit' ? T : 25), pCO2 = v.pCO2 * 1e-6;
  const path = evaporationPath(s1, [{ unit: 'ro', wEnd: wRO, T: Tc(v.T), precip: false }, { unit: 'bc', wEnd: wBC, T: Tc(v.Tbc), precip: true, pCO2 }, { unit: 'cx', wEnd: wCX, T: Tc(v.Tcx), precip: true, pCO2 }], { nEvap: Math.max(4, Math.round(v.nEvap)), minerals });
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
    } else { ro.P = v.oaroP; ro.kW = ro.wLeast / (v.eta2 / 100); }
    for (const [id, lim] of [['gypsum', 0.36], ['anhydrite', 0.36], ['silica', 0.18], ['calcite', 1.8], ['barite', 1.78]]) if (out.SI[id] > lim) W.push({ level: 'warn', msg: `${MINERALS[id].name} reaches SI ${fmt(out.SI[id], 3)} in the membrane concentrate, beyond what antiscalant normally controls (${lim}) — soften first or lower the membrane target.` });
    units.push({ name: v.preconc === 'hpro' ? 'High-pressure RO' : 'Osmotically assisted RO', feedW: Kw * st0.w, removed: Kw * path.evap.ro, out, T: v.T, bpe: null, kWe: ro.kW, kWt: 0, area: null, solids: 0 });
  }

  // --- brine concentrator
  const bc = { on: (path.evap.bc || 0) > 0, mv: Kw * (path.evap.bc || 0), kWe: 0, kWt: 0, area: 0, bpe: 0, comp: 0, aux: 0, vent: 0, econ: 0, steam: 0, balance: null };
  const bcSolids = kgh(path.solids.bc);
  if (bc.on) {
    const inn = first('bc'), out = last('bc'), lam = latentHeat(v.Tbc), Qlat = (bc.mv / 3600) * lam; // W
    bc.bpe = bpeFromAw(v.Tbc, out.aw);
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
    } else {
      const N = Math.max(1, Math.round(v.nEff));
      bc.econ = 0.85 * N * (v.bcType === 'tvc' ? 1 + v.tvcRa : 1);
      bc.kWt = Qlat / 1000 / bc.econ; bc.steam = (bc.kWt * 3.6) / (latentHeat(v.steamT) / 1000); // t/h
      const dTe = (v.Tbc - v.Tlast) / N - bc.bpe;
      if (dTe < 1.5) W.push({ level: 'bad', msg: `Only ${fmt(dTe, 2)} K of driving temperature difference is left per effect after the boiling-point elevation (${fmt(bc.bpe, 3)} K) — use fewer effects, a higher top temperature, or MVC.` });
      bc.area = Qlat / (v.U * 1000 * Math.max(dTe, 0.5)); bc.kWe = (v.bcPumps * bc.mv) / 1000;
    }
    units.push({ name: { mvc: 'Brine concentrator (falling-film MVC)', mee: `Brine concentrator (${Math.round(v.nEff)}-effect MEE)`, tvc: `Brine concentrator (${Math.round(v.nEff)}-effect MEE-TVC)` }[v.bcType], feedW: Kw * inn.w, removed: bc.mv, out, T: v.Tbc, bpe: bc.bpe, kWe: bc.kWe, kWt: bc.kWt, area: bc.area, solids: sum(Object.values(bcSolids)) });
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
    if (v.cxDrive === 'mvr') { cx.kWe = (cx.mv * compressorWork(v.Tcx, cx.bpe, v.dTcx, v.etaComp / 100)) / 3.6e6 / etaM; if (cx.bpe > v.bpeMax) W.push({ level: 'bad', msg: `Crystallizer boiling-point elevation ${fmt(cx.bpe, 3)} K exceeds the ${v.bpeMax} K a vapour compressor can economically overcome — use steam drive or purge more mother liquor.` }); } else cx.kWt = Qlat / 1000 / 0.92;
    cx.kWe += (v.cxPumps * cx.mv) / 1000 + (3 * Sdry) / 1000; // recirculation pump and centrifuge
    cx.area = Qlat / (v.Ufc * 1000 * v.dTcx);
    units.push({ name: `Forced-circulation crystallizer (${v.cxDrive === 'mvr' ? 'MVR' : 'steam'})`, feedW: Kw * first('cx').w, removed: cx.mv, out: end, T: v.Tcx, bpe: cx.bpe, kWe: cx.kWe, kWt: cx.kWt, area: cx.area, solids: Sdry });
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
  return { v, W, feed, io0, Kw, soft, softSolids, s1, path, units, ro, bc, cx, dryer, purgeDry, end, cakeSolids, bcSolids, cxSolids, Sdry, main, purity, adher, washE, cakeSalt, purgeMass, purgeSalt, purgeQ, liquid, pond, csd, water, recovered, kWe, kWt, reagents, salts, solidsTotal, bittern, prod, minerals, wRO, wBC, wCX, xs };
}

const suite = {
  id: 'zld', num: 9, title: 'Brine Concentration, Crystallization & ZLD', short: 'ZLD & salts', icon: '🧂',
  tagline: 'From brine to distilled water and dry salts: evaporation path, salt sequence, crystallizer and the final water/solids balance.',
  description: 'Follows a brine through softening, membrane pre-concentration, a falling-film brine concentrator and a forced-circulation crystallizer. At every evaporation step the Pitzer-based mineral equilibria decide which salts crystallise, so the precipitation sequence, salt yields and purity, boiling-point elevation, compressor work, steam demand and heat-transfer areas follow from the actual brine chemistry. The crystallizer size distribution is solved as an MSMPR population balance with primary and secondary nucleation and power-law growth, both by moments and on a discretised size grid. The final balance states how much water is recovered, which solids leave, and whether any liquid discharge remains.',
  guide: [
    'Pull the RO concentrate (or the brine from suite 2) or enter a brine analysis and flow.',
    'Choose the pretreatment, the membrane pre-concentration step and the type of evaporator and crystallizer drive.',
    'Set the brine-concentrator outlet salinity just below sodium-chloride saturation and choose the mother-liquor purge and where it goes.',
    'Run. Read the precipitation-sequence plot and the salt table, then the unit table for energy and area, and the ZLD balance for the remaining liquid.',
  ],
  implemented: ['total and component mass balance', 'energy balance', 'phase-equilibrium', 'solubility-product', 'saturation-index', 'supersaturation equation', 'classical nucleation equation', 'primary-nucleation', 'secondary-nucleation', 'crystal-growth', 'population-balance equation', 'moment equation', 'crystal-size-distribution', 'evaporation equation', 'vapour–liquid equilibrium', 'solid–liquid equilibrium', 'heat-transfer equation',
    'electrolyte-equilibrium–crystallization', 'evaporation–precipitation', 'nucleation–growth–population-balance', 'ro–crystallization', 'zld process-integration', 'resource-recovery–selective-precipitation', 'thermodynamic–kinetic crystallization',
    'brine composition', 'supersaturation', 'temperature', 'pressure', 'initial crystal population', 'seed size distribution', 'initial solid fraction', 'brine-feed concentration/flow', 'heat-flux or temperature', 'evaporation/vapour-flux', 'outlet population-flux', 'solid–liquid equilibrium/interface',
    'concentrated-electrolyte thermodynamics', 'brine concentration', 'evaporation', 'mechanical and thermal vapour compression', 'mineral saturation', 'nucleation', 'crystal growth', 'precipitation', 'solid-liquid equilibrium', 'crystalliser modelling', 'solids separation', 'centrifugation', 'drying', 'mother-liquor recycling', 'salt-purity', 'selective mineral recovery', 'chemical dosing', 'scale management', 'heat integration', 'water-recovery calculation', 'waste minimisation', 'zero-liquid-discharge assessment', 'resource recovery', 'energy analysis'],
  equationsNote: 'Mineral equilibria use the Harvie–Møller–Weare Pitzer set (25 °C interaction parameters; solubility products and the Debye–Hückel slope follow temperature), which reproduces the seawater evaporation sequence at 25 °C; at evaporator temperatures the onset points are indicative (± 10–15 % in concentration factor) and double salts of the hot Mg–K–SO₄ system are approximate. The path stops at a water activity of 0.33 or an ionic strength of 22 mol/kg; CaCl₂ hydrates are not included, so calcium-chloride bitterns always leave with the purge. Crystallisation is fractional (solids leave the liquor as they form). The MSMPR model assumes a well-mixed, size-independent-growth crystallizer without agglomeration, breakage, classification or Ostwald ripening. Evaporator energy is a lumped single-stage balance (MVC) or a steam-economy correlation (MEE/TVC); use suite 6 for effect-by-effect thermal design.',

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
      { key: 'preconc', label: 'Technology', type: 'select', value: 'hpro', options: [{ value: 'none', label: 'None' }, { value: 'hpro', label: 'High-pressure RO (up to 120 bar)' }, { value: 'oaro', label: 'Osmotically assisted RO (multi-stage, about 70 bar)' }] },
      { key: 'roTDS', label: 'Target concentrate salinity', unit: 'g/kg', value: 120, min: 20, max: 250, help: 'High-pressure RO reaches about 120–130 g/kg; osmotically assisted RO 150–230 g/kg.', showIf: (v) => v.preconc !== 'none' },
      { key: 'pMax', label: 'Membrane pressure rating', unit: 'bar', value: 120, min: 40, max: 200, showIf: (v) => v.preconc === 'hpro' },
      { key: 'erdEff', label: 'Energy-recovery efficiency', unit: '%', value: 94, min: 0, max: 99, showIf: (v) => v.preconc === 'hpro' },
      { key: 'etaPump', label: 'High-pressure pump efficiency', unit: '%', value: 84, min: 30, max: 93, showIf: (v) => v.preconc === 'hpro' },
      { key: 'oaroP', label: 'Operating pressure', unit: 'bar', value: 68, min: 30, max: 90, showIf: (v) => v.preconc === 'oaro' },
      { key: 'eta2', label: 'Second-law efficiency', unit: '%', value: 28, min: 5, max: 70, help: 'Reversible work of separation divided by actual electricity use; 20–35 % for multi-stage osmotically assisted RO.', showIf: (v) => v.preconc === 'oaro' },
    ] },
    { group: 'Brine concentrator', help: 'Seeded-slurry falling-film evaporator.', fields: [
      { key: 'bcType', label: 'Type', type: 'select', value: 'mvc', options: [{ value: 'mvc', label: 'Mechanical vapour compression (MVC)' }, { value: 'mee', label: 'Multi-effect evaporator, steam driven' }, { value: 'tvc', label: 'Multi-effect with thermal vapour compression' }] },
      { key: 'bcTDS', label: 'Outlet salinity', unit: 'g/kg', value: 240, min: 60, max: 320, typical: [200, 260], help: 'Kept just below sodium-chloride saturation (about 265 g/kg for seawater-type brines).' },
      { key: 'Tbc', label: 'Boiling temperature (top effect for MEE)', unit: '°C', value: 100, min: 40, max: 125 },
      { key: 'dT', label: 'Tube temperature difference (MVC)', unit: 'K', value: 3.5, min: 1, max: 12, help: 'Condensing vapour minus boiling brine. Smaller values save compressor energy but need more area.', showIf: (v) => v.bcType === 'mvc' },
      { key: 'U', label: 'Overall heat-transfer coefficient', unit: 'kW/m²·K', value: 2.2, min: 0.3, max: 5 },
      { key: 'etaComp', label: 'Compressor isentropic efficiency', unit: '%', value: 76, min: 40, max: 90 },
      { key: 'nEff', label: 'Number of effects', unit: '', value: 4, min: 1, max: 12, step: 1, showIf: (v) => v.bcType !== 'mvc' },
      { key: 'Tlast', label: 'Last-effect temperature', unit: '°C', value: 48, min: 30, max: 90, showIf: (v) => v.bcType !== 'mvc' },
      { key: 'tvcRa', label: 'Thermocompressor entrainment ratio', unit: 'kg/kg', value: 1, min: 0.2, max: 3, showIf: (v) => v.bcType === 'tvc' },
      { key: 'steamT', label: 'Heating-steam temperature', unit: '°C', value: 120, min: 60, max: 200, showIf: (v) => v.bcType !== 'mvc' },
      { key: 'bcSolids', label: 'Seed-slurry solids (CaSO₄, CaCO₃, silica)', type: 'select', value: 'separate', options: [{ value: 'separate', label: 'Separated before the crystallizer' }, { value: 'cake', label: 'Carried into the salt cake' }] },
    ] },
    { group: 'Crystallizer and solids handling', fields: [
      { key: 'cxDrive', label: 'Crystallizer drive', type: 'select', value: 'steam', options: [{ value: 'steam', label: 'Steam (single effect)' }, { value: 'mvr', label: 'Mechanical vapour recompression' }] },
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
    ];
    const stream = (label, st) => [label, ...['Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3'].map((k) => st.ions[k]), st.tds / 1000, st.S, st.pH, st.rho, st.aw, bpeFromAw(st.unit === 'cx' ? v.Tcx : st.unit === 'bc' ? v.Tbc : 100, st.aw), st.cf];
    const lastOf = (u) => { for (let i = steps.length - 1; i >= 0; i--) if (steps[i].unit === u) return steps[i]; return null; };
    const out = {
      waterRecovered: r.recovered, solids: td(r.solidsTotal), secElec: r.recovered > 0 ? r.kWe / r.recovered : 0, secThermal: r.recovered > 0 ? r.kWt / r.recovered : 0, power: r.kWe, heat: r.kWt, salts: saltsOut, liquidDischarge: r.liquid,
      recovery: recPct / 100, zld, saltPurity: r.purity, haliteOnsetCF: on('halite') ?? 0, productSalt: MINERALS[r.main]?.name || 'none', productSaltTpd: td(r.prod), L50um: csd ? csd.fv.L50 * 1e6 : 0, cv: csd ? csd.fv.cv : 0, crystallizerVolume: csd ? csd.volume : 0, bcArea: bc.area, bpeBC: bc.bpe, bpeCX: cx.bpe, pondArea: r.pond.area,
      revenuePerDay: rev, energyCostPerDay: energyCost, reagents: Object.fromEntries(Object.entries(r.reagents).map(([k, kg]) => [REAGENTS[k].name, +(kg * 24).toPrecision(5)])), chemicals: sum(Object.values(r.reagents)) * 24, pathSolidsTpd: td(sum(formed.map((id) => (path.cum[id] * MINERALS[id].mw * r.Kw) / 1000))),
      streams: { purge: { Q: r.purgeQ, T: v.Tcx, P: 1, pH: +end.pH.toFixed(3), tds: end.tds, ions: Object.fromEntries(ION_IDS.map((k) => [k, +end.ions[k].toPrecision(6)])) } },
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
        { type: 'field', title: 'MVC compressor energy versus brine salinity and tube ΔT', xlabel: 'Brine salinity in the evaporator (g/kg)', ylabel: 'Tube temperature difference (K)', zlabel: 'Compressor electricity', zunit: 'kWh/t', x: fx, y: fy, z: fz, cmap: 'thermal', contours: 8, markers: bc.on && v.bcType === 'mvc' ? [{ x: clamp(lastOf('bc').S, fx[0], fx[fx.length - 1]), y: clamp(v.dT, 1.5, 9), label: 'design' }] : [], note: 'Salinity acts through the boiling-point elevation, which the compressor must overcome in addition to the tube temperature difference.' },
        { type: 'bar', title: 'Solids produced', ylabel: 't/d', categories: r.salts.length ? r.salts.map((s) => `${s.name} · ${s.unit}`) : ['none'], series: [{ name: 't/d', values: r.salts.length ? r.salts.map((s) => td(s.kg)) : [0] }] },
      ],
      tables: [
        { title: 'Unit operations', columns: ['Unit', 'Water in (t/h)', 'Water removed (t/h)', 'Outlet liquor (m³/h)', 'Outlet salinity (g/kg)', 'Concentration factor', 'Temperature (°C)', 'BPE (K)', 'Electricity (kW)', 'Heat (kW)', 'Heat-transfer area (m²)', 'Solids formed (t/d)'],
          rows: r.units.length ? r.units.map((u) => [u.name, u.feedW / 1000, u.removed / 1000, (r.Kw * u.out.w * (1 + u.out.gPerKgw / 1000)) / u.out.rho, u.out.S, u.out.cf, u.T, u.bpe, u.kWe, u.kWt, u.area, td(u.solids)]) : [['No concentration unit is active', 0, 0, v.Q, steps[0].S, 1, v.T, null, 0, 0, null, 0]],
          note: `${ro.on ? `Membrane step: ${fmt(ro.P, 3)} bar, outlet osmotic pressure ${fmt(ro.pi, 3)} bar, reversible work ${fmt(ro.wLeast, 3)} kW. ` : ''}${bc.on && v.bcType === 'mvc' ? `MVC: compressor shaft power ${fmt(bc.comp, 4)} kW, auxiliary heat ${fmt(bc.aux, 3)} kW, surplus heat ${fmt(bc.vent, 3)} kW. ` : ''}${bc.on && v.bcType !== 'mvc' ? `Steam economy ${fmt(bc.econ, 3)} kg/kg, heating steam ${fmt(bc.steam, 3)} t/h. ` : ''}Dryer heat: ${fmt(r.dryer.kWt + r.purgeDry.kWt, 3)} kW.` },
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
    add('Salt yield is insensitive to the number of evaporation steps (16 → 32)', 0, Math.abs((coarse.cxSolids.halite || 0) - (r.cxSolids.halite || 0)) / Math.max(1e-9, r.cxSolids.halite || 0), 0.01, 'Relative change of the halite production');
    return C;
  },
};

/** Synthetic plant data: the steady-state MSMPR model with different true kinetics plus deterministic noise. */
function synth(seed, pts) {
  const d = D(), g = rng(seed);
  return pts.map(([tauX, mtX]) => ({ tauX, mtX, L50: +(suite.calibration.model({ ...d, kb: 2.4e8, jMT: 0.75, tauX, mtX }).L50 * (1 + g.normal(0, 0.02))).toFixed(1) }));
}

export default suite;
