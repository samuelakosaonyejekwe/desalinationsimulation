// Suite 3 — Whole-plant process simulation.
// Sequential-modular steady-state flowsheet solver with tear-stream convergence (Wegstein or damped
// substitution) over a library of unit operations: intake, pumps, dosing, filtration, RO/NF passes,
// isobaric energy recovery, turbines, valves, mixers, splitters, heat exchangers, a reduced MED block,
// a vapour-compression brine concentrator, a crystalliser, remineralisation and storage.
// Streams carry mass flow and the mass flow of every ion, so total and component balances close exactly;
// enthalpy and physical + salinity exergy give energy closure and exergy destruction per unit.
import { clamp, linspace, sum, rng, fmt, rk4, interp1 } from '../core/num.js';
import { density, cp, enthalpyLiquid, latentHeat, psat, bpe, tcf, osmoticCoefficient, salinityFromTDS, R, KELVIN } from '../core/props.js';
import { IONS, ION_IDS, WATERS, tds, scaleIons, osmoticPressureIons, hardness, alkalinity } from '../core/water.js';
import roSuite, { simulateRO, MEMBRANES } from './s01_ro.js';

const P0 = 1.01325, MW_W = 18.015, SMAX = 260, REF_FF = 0.95, NI = ION_IDS.length;
const IX = Object.fromEntries(ION_IDS.map((k, i) => [k, i])), MW = ION_IDS.map((k) => IONS[k].mw);
const zeros = () => new Array(NI).fill(0);
const pK1 = (S) => 6.35 - 0.37 * Math.min(1, S / 35); // apparent first dissociation constant of carbonic acid
const tcfRO = (T) => tcf(T, T >= 25 ? 2640 : 3020);

// ---- streams --------------------------------------------------------------------------------------
/** Build a stream from mass flow m (kg/h), ion mass flows w (kg/h, ION_IDS order), T (°C) and P (bar a). */
export function mkS(m, w, T, P, o = {}) {
  let salt = 0;
  for (let i = 0; i < NI; i++) salt += w[i];
  const solid = o.phase === 'solid', S = m > 0 ? Math.min(1000, (1000 * salt) / m) : 0, rho = solid ? 1900 : density(T, Math.min(S, SMAX)), Q = m / rho, co2 = o.co2 || 0;
  const pH = !solid && co2 > 1e-12 && w[IX.HCO3] > 1e-15 ? clamp(pK1(S) + Math.log10(w[IX.HCO3] / 61.017 / (co2 / 44.01)), 3.5, 10.8) : o.pH ?? 7;
  return { m, w, T, P, pH, co2, tss: o.tss || 0, phase: o.phase || 'liquid', salt, S, rho, Q, tds: Q > 0 ? (1000 * salt) / Q : 0 };
}
const keep = (s) => ({ co2: s.co2, tss: s.tss, phase: s.phase, pH: s.pH });
const zeroS = (T, P) => mkS(0, zeros(), T, P, { pH: 7 });
/** Stream from volumetric flow (m³/h) and a composition in mg/L. */
export function fromVol(Q, ions, T, P, pH = 7, tssMg = 0) {
  const S = salinityFromTDS(tds(ions), T), w = ION_IDS.map((k) => ((+ions[k] || 0) * Q) / 1000);
  return mkS(density(T, Math.min(S, SMAX)) * Q, w, T, P, { pH, co2: w[IX.HCO3] > 0 ? (w[IX.HCO3] / 61.017) * 44.01 * 10 ** (pK1(S) - pH) : 0, tss: (tssMg * Q) / 1000 });
}
export const ionsOf = (s) => Object.fromEntries(ION_IDS.map((k, i) => [k, s.Q > 0 ? (1000 * s.w[i]) / s.Q : 0]));
/** Mass flow that gives volumetric flow Q for a known salt load. */
function byVol(Q, w, T) {
  const salt = sum(w);
  let m = Q * 1000;
  for (let i = 0; i < 5; i++) m = Q * density(T, Math.min(SMAX, m > 0 ? (1000 * salt) / m : 0));
  return m;
}
const cpOf = (s) => (s.phase === 'solid' ? 1200 : cp(s.T, Math.min(s.S, SMAX)));
const hSpec = (s) => (s.phase === 'solid' ? 1200 * s.T : enthalpyLiquid(s.T, Math.min(s.S, SMAX))) + ((s.P - P0) * 1e5) / s.rho; // J/kg, includes flow work
/** Enthalpy flow, kW. */
export const Hk = (s) => (s.m * hSpec(s)) / 3.6e6;
/** Give a set of outlet streams one common temperature that satisfies the energy balance (H in kW). */
function settle(specs, H, Tg) {
  let T = Tg, out = specs.map((q) => mkS(q.m, q.w, T, q.P, q.o));
  for (let i = 0; i < 8; i++) {
    const C = sum(out.map((s) => s.m * cpOf(s))) / 3.6e6;
    if (!(C > 1e-12)) break;
    const dT = (H - sum(out.map(Hk))) / C;
    T = clamp(T + dT, -5, 400);
    out = specs.map((q) => mkS(q.m, q.w, T, q.P, q.o));
    if (Math.abs(dT) < 1e-10) break;
  }
  return out;
}
const settle1 = (s, P, Hadd = 0) => settle([{ m: s.m, w: s.w, P, o: keep(s) }], Hk(s) + Hadd, s.T)[0];
const scaleS = (s, f) => mkS(s.m * f, s.w.map((x) => x * f), s.T, s.P, { co2: s.co2 * f, tss: s.tss * f, phase: s.phase, pH: s.pH });

/** Dead state for exergy: the raw feed at ambient pressure. */
function deadState(raw) {
  const Ni = sum(raw.w.map((x, i) => (1000 * x) / MW[i])), Nw = (1000 * (raw.m - raw.salt)) / MW_W;
  return { T: raw.T, xw: Nw / (Nw + Ni), xi: Ni / (Nw + Ni), phi: Ni > 0 ? osmoticCoefficient(raw.T, Math.min(raw.S, 200)) : 1 };
}
/** Physical and chemical (salinity) exergy flow of a stream, kW. */
export function exergy(s, dead) {
  if (!(s.m > 0)) return [0, 0];
  const T0 = dead.T + KELVIN, Tk = s.T + KELVIN, c = s.phase === 'solid' ? 1200 : cp(0.5 * (s.T + dead.T), Math.min(s.S, SMAX));
  const ph = (s.m * (c * (Tk - T0 - T0 * Math.log(Tk / T0)) + ((s.P - P0) * 1e5) / s.rho)) / 3.6e6;
  if (s.phase === 'solid') return [ph, 0];
  const Ni = sum(s.w.map((x, i) => (1000 * x) / MW[i])), Nw = (1000 * (s.m - s.salt)) / MW_W, Nt = Nw + Ni;
  const ch = (dead.phi * R * T0 * (Nw * Math.log(Nw / Nt / dead.xw) + (Ni > 0 && dead.xi > 0 ? Ni * Math.log(Ni / Nt / dead.xi) : 0))) / 3.6e6;
  return [ph, ch];
}

/** Langelier saturation index of a water (ions in mg/L). */
export function langelier(ions, T, pH) {
  const ca = Math.max(1e-6, 2.497 * (+ions.Ca || 0)), alk = Math.max(1e-6, alkalinity(ions)), t = Math.max(1, tds(ions));
  return pH - (9.3 + (Math.log10(t) - 1) / 10 + (-13.12 * Math.log10(T + KELVIN) + 34.55) - (Math.log10(ca) - 0.4) - Math.log10(alk));
}

// ---- chemicals: ions added, species formed (+) or consumed (−) by reaction, CO₂ released and acidity (kmol H⁺), all per kg of pure product
const CHEM = {
  ferric: { ions: { Fe: 0.3443, Cl: 0.6557 }, gen: { Fe: -0.3443 }, solid: 0.6589, acid: 3 / 162.2 },
  acid: { ions: { SO4: 0.9794 }, acid: 2 / 98.08 },
  antiscalant: {}, co2: { co2: 1 },
  sbs: { ions: { Na: 0.2209, SO4: 0.7791 } },
  hypo: { ions: { Na: 0.3088, Cl: 0.4763 } },
  naoh: { ions: { Na: 0.5748 }, gen: { HCO3: 1.5254 }, co2: -1.1003 },
  lime: { ions: { Ca: 0.5409 }, gen: { HCO3: 1.6471 }, co2: -1.188 },
  calcite: { ions: { Ca: 0.4004, CO3: 0.5996 }, gen: { CO3: -0.5996, HCO3: 1.2192 }, co2: -0.4397 },
};

/** Add chemicals [{ name, kind, kgh }] to a stream; returns the outlet and the bookkeeping for the balances. */
function applyChems(s, list, pHforce) {
  const w = [...s.w], chemIons = zeros(), gen = zeros(), chem = {};
  let co2 = s.co2, add = 0, toTss = 0;
  for (const c of list) {
    const k = c.kgh, sp = CHEM[c.kind] || {};
    if (!(k > 0)) continue;
    chem[c.name] = (chem[c.name] || 0) + k;
    for (const [id, f] of Object.entries(sp.ions || {})) { w[IX[id]] += f * k; chemIons[IX[id]] += f * k; }
    const need = sp.co2 < 0 ? -sp.co2 * k : 0, phi = need > 0 ? Math.min(1, co2 / need) : 1; // reactions that take up CO₂ stop when it runs out
    for (const [id, f] of Object.entries(sp.gen || {})) { const i = IX[id], g = Math.max(-w[i], f * k * phi); w[i] += g; gen[i] += g; }
    co2 = Math.max(0, co2 + (sp.co2 || 0) * k * phi);
    if (sp.acid) { // acidity first turns carbonate into bicarbonate, then bicarbonate into dissolved CO₂
      const H = sp.acid * k, x1 = Math.min(H, w[IX.CO3] / 60.009), x2 = Math.min(H - x1, w[IX.HCO3] / 61.017 + x1);
      w[IX.CO3] -= x1 * 60.009; gen[IX.CO3] -= x1 * 60.009; w[IX.HCO3] += (x1 - x2) * 61.017; gen[IX.HCO3] += (x1 - x2) * 61.017; co2 += x2 * 44.01;
    }
    if (c.kind === 'naoh' && phi < 1) { // hydroxide left after the CO₂ is used converts bicarbonate to carbonate
      const x = Math.min((k * (1 - phi)) / 40.0, w[IX.HCO3] / 61.017);
      w[IX.HCO3] -= x * 61.017; gen[IX.HCO3] -= x * 61.017; w[IX.CO3] += x * 60.009; gen[IX.CO3] += x * 60.009;
    }
    add += k * (1 - (sp.solid || 0)); toTss += k * (sp.solid || 0);
  }
  const Hadd = (add * hSpec(s)) / 3.6e6, o = { co2, tss: s.tss + toTss, phase: s.phase, pH: pHforce ?? s.pH };
  const out = settle([{ m: s.m + add, w, P: s.P, o }], Hk(s) + Hadd, s.T)[0];
  return { outs: [out], rec: { chem, chemMass: add, chemIons, gen, toTss, Hadd } };
}

/** Flash of a saturated liquid from T1 to T2: vapour mass fraction. */
export const flashFraction = (T1, T2, S = 0) => Math.max(0, (enthalpyLiquid(T1, S) - enthalpyLiquid(T2, S)) / latentHeat(T2, 0));
/** Adiabatic vapour compressor: specific work (J/kg) between saturation temperatures, isentropic efficiency eta. */
export function compressorWork(Tin, Tout, eta) {
  const cpv = 1860, k = 1.33, ratio = psat(Tout) / psat(Tin);
  return { w: (cpv * (Tin + KELVIN) * (ratio ** ((k - 1) / k) - 1)) / eta, ratio };
}
/** Counter-flow effectiveness. */
export const effectiveness = (NTU, Cr) => (Math.abs(1 - Cr) < 1e-9 ? NTU / (1 + NTU) : (1 - Math.exp(-NTU * (1 - Cr))) / (1 - Cr * Math.exp(-NTU * (1 - Cr))));

/** Normative salts of a solids stream (kg/h by name) from its ion mass flows. */
function normativeSalts(w) {
  const n = (k) => w[IX[k]] / IONS[k].mw, out = {};
  let Ca = n('Ca'), Mg = n('Mg'), Na = n('Na'), K = n('K'), Cl = n('Cl'), SO4 = n('SO4'), C = n('HCO3') / 2 + n('CO3'), x;
  const take = (name, mw, a) => { if (a > 1e-12) out[name] = (out[name] || 0) + a * mw; };
  x = Math.min(Ca, C); take('Calcium carbonate', 100.09, x); Ca -= x;
  x = Math.min(Ca, SO4); take('Gypsum', 172.17, x); Ca -= x; SO4 -= x;
  x = Math.min(Mg, SO4); take('Magnesium sulphate', 120.37, x); Mg -= x; SO4 -= x;
  x = Math.min(Na / 2, SO4); take('Sodium sulphate', 142.04, x); Na -= 2 * x;
  x = Math.min(Na, Cl); take('Sodium chloride', 58.44, x); Cl -= x;
  x = Math.min(K, Cl); take('Potassium chloride', 74.55, x); Cl -= x;
  x = Math.min(Mg, Cl / 2); take('Magnesium chloride', 95.21, x); Cl -= 2 * x;
  x = Math.min(Ca, Cl / 2); take('Calcium chloride', 110.98, x);
  return out;
}

// ---- unit operations: (inlet streams, parameters, solver state) → { outs, rec } ----------------------
const UNITS = {
  screen: ([s], p) => ({ outs: [settle1(s, s.P - p.dp)], rec: { dP: -p.dp } }),
  tank: ([s], p) => { const Po = Math.min(s.P, P0 + 0.3); return { outs: [settle1(s, Po)], rec: { volume: s.Q * p.hours, dP: Po - s.P } }; },
  valve: ([s], p) => { const Po = Math.min(s.P, p.Pout); return { outs: [settle1(s, Po)], rec: { dP: Po - s.P, throttled: (s.Q * (s.P - Po)) / 36 } }; },
  pump: ([s], p) => {
    const dP = p.Pout - s.P;
    if (!(dP > 0) || !(s.m > 0)) return { outs: [s], rec: { dP: 0, power: 0, shaft: 0, hyd: 0 } };
    const hyd = (s.Q * dP) / 36, shaft = hyd / p.eta;
    return { outs: [settle1(s, p.Pout, shaft)], rec: { dP, hyd, shaft, power: shaft / p.etaM, eff: p.eta } };
  },
  turbine: ([s], p) => {
    const dP = Math.max(0, s.P - p.Pout), shaft = (p.eta * s.Q * dP) / 36;
    return { outs: [settle1(s, s.P - dP, -shaft)], rec: { dP: -dP, shaft: -shaft, power: -shaft / p.etaM, hyd: (s.Q * dP) / 36, eff: p.eta } };
  },
  mix: (ins) => {
    const live = ins.filter((s) => s.m > 0);
    if (!live.length) return { outs: [zeroS(ins[0].T, ins[0].P)], rec: {} };
    const m = sum(live.map((s) => s.m)), w = zeros();
    for (const s of live) for (let i = 0; i < NI; i++) w[i] += s.w[i];
    const o = { co2: sum(live.map((s) => s.co2)), tss: sum(live.map((s) => s.tss)), pH: sum(live.map((s) => s.pH * s.m)) / m };
    return { outs: settle([{ m, w, P: Math.min(...live.map((s) => s.P)), o }], sum(live.map(Hk)), sum(live.map((s) => s.T * s.m)) / m), rec: {} };
  },
  split: ([s], p) => { const f = clamp(p.frac ?? (s.Q > 0 ? p.Q / s.Q : 0), 0, 1); return { outs: [scaleS(s, f), scaleS(s, 1 - f)], rec: { split: f } }; },
  dose: ([s], p) => {
    const list = (p.chems || []).map(([name, kind, dose]) => ({ name, kind, kgh: (dose * s.Q) / 1000 }));
    if (p.pHset && s.Q > 0) { // caustic needed to reach a set pH: CO₂ → HCO₃⁻, part of HCO₃⁻ → CO₃²⁻, plus free hydroxide
      const c = s.co2 / 44.01, b = s.w[IX.HCO3] / 61.017, ratio = 10 ** (p.pHset - pK1(s.S)), f2 = 1 / (1 + 10 ** (10.33 - p.pHset));
      list.push({ name: 'Caustic soda', kind: 'naoh', kgh: 40.0 * (Math.max(0, (ratio * c - b) / (1 + ratio)) + f2 * (b + c) + 10 ** (p.pHset - 14) * s.Q) });
    }
    return applyChems(s, list, p.pHset && s.Q > 0 ? p.pHset : undefined);
  },
  remin: ([s], p) => {
    if (!(s.Q > 0)) return { outs: [s], rec: {} };
    const ions = ionsOf(s), dCa = Math.max(0, (p.hardness - hardness(ions)) * 0.4004), calcite = p.method === 'calcite', kg = (dCa * s.Q) / 1000 / (calcite ? 0.4004 : 0.5409);
    const up = calcite ? 0.4397 * kg : 1.188 * kg, hco3 = s.w[IX.HCO3] / 61.017 + ((calcite ? 1.2192 : 1.6471) * kg) / 61.017; // kmol/h after dissolution
    let co2Dose = 0, naoh = 0;
    for (let i = 0; i < 4; i++) { // residual CO₂ sets the pH; surplus CO₂ is neutralised with caustic
      const hb = hco3 + naoh, after = { ...ions, Ca: ions.Ca + dCa, HCO3: (hb * 61017) / s.Q }, ratio = 10 ** (clamp(p.lsi - langelier(after, s.T, 0), 6.5, 9.5) - pK1(s.S)), free = s.co2 / 44.01 - up / 44.01;
      co2Dose = Math.max(0, (hco3 / ratio - free) * 44.01);
      naoh = co2Dose > 0 ? 0 : Math.max(0, (ratio * free - hco3) / (1 + ratio));
    }
    const list = [{ name: 'Carbon dioxide', kind: 'co2', kgh: co2Dose }, { name: calcite ? 'Calcite' : 'Hydrated lime', kind: calcite ? 'calcite' : 'lime', kgh: kg }, { name: 'Caustic soda', kind: 'naoh', kgh: naoh * 40.0 }];
    const r = applyChems(s, list), q = r.outs[0];
    r.rec.lsi = langelier(ionsOf(q), q.T, q.pH); r.rec.dP = 0;
    return r;
  },
  filter: ([s], p) => {
    const f = clamp(p.recovery, 0, 1), Pf = s.P - p.dp, H = Hk(s);
    if (f >= 1 || !(s.m > 0)) return { outs: [settle([{ m: s.m, w: s.w, P: Pf, o: { ...keep(s), tss: s.tss * (1 - p.tssRem) } }], H, s.T)[0]], rec: { dP: -p.dp, recovery: 1, sludge: s.tss * p.tssRem } };
    const outs = settle([{ m: s.m * f, w: s.w.map((x) => x * f), P: Pf, o: { co2: s.co2 * f, tss: s.tss * (1 - p.tssRem), pH: s.pH } }, { m: s.m * (1 - f), w: s.w.map((x) => x * (1 - f)), P: P0 + 0.2, o: { co2: s.co2 * (1 - f), tss: s.tss * p.tssRem, pH: s.pH } }], H, s.T);
    return { outs, rec: { dP: -p.dp, recovery: f, sludge: s.tss * p.tssRem } };
  },
  // Reduced membrane pass: net driving pressure, pressure drop and ion passage scaled from an element-by-element reference solution.
  ro: ([s], p, st) => {
    const ref = p.ref, r = clamp(p.recovery, 0, 0.985), Qf = s.Q;
    if (!(Qf > 1e-9)) { st.Pn[p.key] = st.P[p.key]; return { outs: [zeroS(s.T, p.Pp), zeroS(s.T, s.P)], rec: { recovery: 0, dP: 0, flux: 0, area: p.area || 0, Preq: st.P[p.key], idle: true } }; }
    const area = p.area > 0 ? p.area : ref.areaPer * Qf, Qp = r * Qf, J = (Qp * 1000) / area, fac = r > 1e-9 ? -Math.log(1 - r) / r : 1, tc = tcfRO(s.T), cf = ionsOf(s);
    const kT = (tc / ref.tc) * (fac / ref.fac) * p.kSP * (J > 1e-9 ? ref.J / J : 1e12), wp = zeros(), cpI = {};
    ION_IDS.forEach((k, i) => { cpI[k] = cf[k] * Math.min(1, ref.SP[i] * kT); wp[i] = (cpI[k] * Qp) / 1000; });
    const dPi = (osmoticPressureIons(scaleIons(cf, fac), s.T) - osmoticPressureIons(cpI, s.T)) / 1e5;
    const load = (Qf * (1 - r / 2)) / ((area / ref.areaPer) * (1 - ref.r / 2)), dP = (ref.dP + ref.boost2) * load ** 1.7 - ref.boost2;
    const Preq = p.Pp + ref.NDP * (J / ref.J) * (ref.tc / tc) * (REF_FF / p.ff) + dPi + dP / 2;
    const mp = Qp > 0 ? byVol(Qp, wp, s.T) : 0, fr = Qp / Qf, hyd = (ref.s2frac * Qf * ref.boost2) / 36, shaft = hyd / p.eta;
    const outs = settle([{ m: mp, w: wp, P: p.Pp, o: { co2: s.co2 * fr, pH: s.pH } }, { m: s.m - mp, w: s.w.map((x, i) => x - wp[i]), P: s.P - dP, o: { co2: s.co2 * (1 - fr), pH: s.pH, tss: s.tss } }], Hk(s) + shaft, s.T);
    st.Pn[p.key] = Preq;
    const nV = (sum(ref.vessels) * area) / (ref.areaPer * ref.Qf);
    return { outs, rec: { recovery: fr, dP: -dP, flux: J, leadFlux: J * ref.leadRatio, area, Preq, Pfeed: s.P, NDP: Preq - p.Pp - dPi - dP / 2, dPi, shaft, power: shaft / p.etaM, hyd, vessels: nV, elements: area / ref.elArea, concPerVessel: (ref.concPerFeed * outs[1].Q) / ((1 - ref.r) * (area / ref.areaPer)), pmax: ref.pmax } };
  },
  // Isobaric pressure exchanger: volumes are exchanged, a lubrication flow leaks to the low-pressure side and a little brine mixes into the feed.
  px: ([b, f], p) => {
    if (!(b.Q > 1e-9) || !(f.Q > 1e-9)) return { outs: [zeroS(f.T, b.P), mixOf(b, f, f.P - p.dpLP)], rec: { eff: 0, mixing: 0 } };
    const Qh = Math.min(b.Q * (1 - p.leak), f.Q * 0.999), of = Math.max(0, f.Q / b.Q - 1), M = p.mix0 * Math.exp(-12 * of);
    const wh = b.w.map((x, i) => ((f.w[i] / f.Q) * (1 - M) + (x / b.Q) * M) * Qh), mh = byVol(Qh, wh, f.T), ch = ((f.co2 / f.Q) * (1 - M) + (b.co2 / b.Q) * M) * Qh;
    const outs = settle([{ m: mh, w: wh, P: b.P - p.dpHP, o: { co2: ch, pH: f.pH } }, { m: b.m + f.m - mh, w: b.w.map((x, i) => x + f.w[i] - wh[i]), P: f.P - p.dpLP, o: { co2: b.co2 + f.co2 - ch, pH: b.pH, tss: b.tss + f.tss } }], Hk(b) + Hk(f), (b.T * b.m + f.T * f.m) / (b.m + f.m));
    return { outs, rec: { eff: (outs[0].Q * outs[0].P + outs[1].Q * outs[1].P) / (b.Q * b.P + f.Q * f.P), mixing: M, leak: b.Q - Qh, recovered: (outs[0].Q * (outs[0].P - f.P)) / 36, dP: -p.dpHP, overflush: of } };
  },
  hx: ([hot, cold], p) => {
    if (!(hot.m > 0 && cold.m > 0) || hot.T <= cold.T) return { outs: [hot, cold], rec: { duty: 0, eps: 0, NTU: 0, lmtd: 0, area: p.UA / p.U } };
    const Ch = (hot.m * cpOf(hot)) / 3.6e6, Cc = (cold.m * cpOf(cold)) / 3.6e6, Cmin = Math.min(Ch, Cc), Cr = Cmin / Math.max(Ch, Cc), NTU = p.UA / Cmin, eps = effectiveness(NTU, Cr), Q = eps * Cmin * (hot.T - cold.T);
    const ho = settle1(hot, hot.P - (p.dp || 0), -Q), co = settle1(cold, cold.P - (p.dp || 0), Q), d1 = hot.T - co.T, d2 = ho.T - cold.T, lmtd = Math.abs(d1 - d2) < 1e-9 ? d1 : (d1 - d2) / Math.log(d1 / d2);
    return { outs: [ho, co], rec: { duty: Q, eps, NTU, lmtd, area: p.UA / p.U, dP: -(p.dp || 0), Cr } };
  },
  // Reduced multi-effect distillation: performance ratio fixes the steam demand, the energy balance fixes the cooling-water outlet.
  med: ([s], p) => {
    if (!(s.m > 0)) return { outs: [zeroS(s.T, 2), zeroS(s.T, P0 + 0.5), s], rec: { heat: 0, recovery: 0 } };
    const F = s.m * p.make, D = F * p.rec, wF = s.w.map((x) => x * p.make), kd = s.salt > 0 ? (D * 5e-6) / (s.salt * p.make) : 0, wd = wF.map((x) => x * kd);
    const steam = D / 3600 / p.GOR, heat = (steam * latentHeat(p.Ts)) / 1000, aux = (p.secElec * D) / 1000, cool = (heat * p.loss);
    const dist = mkS(D, wd, p.Tlast, 2, { pH: 6.5, co2: 0 }), SB = (1000 * (sum(wF) - sum(wd))) / (F - D), brine = mkS(F - D, wF.map((x, i) => x - wd[i]), p.Tlast + bpe(p.Tlast, Math.min(SB, 120)), P0 + 0.5, { pH: s.pH, co2: s.co2 * p.make, tss: s.tss * p.make });
    const cw = settle([{ m: s.m - F, w: s.w.map((x) => x * (1 - p.make)), P: Math.max(P0 + 0.5, s.P - 1), o: { co2: s.co2 * (1 - p.make), tss: s.tss * (1 - p.make), pH: s.pH } }], Hk(s) + heat + aux - cool - Hk(dist) - Hk(brine), s.T + 8)[0];
    const n = Math.max(2, Math.round(p.GOR / 0.85)), dTe = Math.max(0.8, (p.Ts - p.Tlast) / n - bpe(0.5 * (p.Ts + p.Tlast), Math.min(SB, 120)));
    return { outs: [dist, brine, cw], rec: { heat, Ts: p.Ts, cool, power: aux, shaft: aux, recovery: D / s.m, steam, effects: n, dTcw: cw.T - s.T, flash: flashFraction(p.Ts - 3, p.Tlast, Math.min(SB, 120)), area: (n * heat * 0.9) / (2.8 * dTe) + (heat * 0.95) / (2.2 * Math.max(2, p.Tlast - 0.5 * (s.T + cw.T))), condDuty: heat - cool - (Hk(dist) + Hk(brine) - (Hk(s) * F) / s.m) + aux } };
  },
  // Mechanical-vapour-compression evaporator concentrating to a target salinity.
  bc: ([s], p) => {
    if (!(s.m > 0) || s.S >= 0.98 * p.target) return { outs: [zeroS(s.T, P0 + 0.5), s], rec: { recovery: 0, power: 0, shaft: 0 } };
    const mc = (1000 * s.salt) / p.target, D = s.m - mc, wd = s.w.map((x) => (x * D * 1e-5) / Math.max(s.salt, 1e-12)), rise = bpe(100, Math.min(p.target, SMAX)), cw = compressorWork(100, 100 + rise + p.dT, p.etaComp);
    const shaft = ((D / 3600) * cw.w) / 1000 + p.aux * s.Q, o = { pH: s.pH, tss: 0 };
    const outs = settle([{ m: D, w: wd, P: P0 + 0.5, o: { pH: 6.5, co2: 0 } }, { m: mc, w: s.w.map((x, i) => x - wd[i]), P: P0 + 0.5, o: { ...o, co2: 0, tss: s.tss } }], Hk(s) + shaft, s.T + 10);
    return { outs, rec: { recovery: D / s.m, shaft, power: shaft / p.etaM, ratio: cw.ratio, wSpec: cw.w / 3600, bpe: rise, area: ((D / 3600) * latentHeat(100)) / 1000 / (2.5 * p.dT), dP: P0 + 0.5 - s.P } };
  },
  // Forced-circulation crystalliser: evaporates to a moist salt cake with a small purge.
  cryst: ([s], p) => {
    if (!(s.m > 0)) return { outs: [zeroS(s.T, P0), mkS(0, zeros(), s.T, P0, { phase: 'solid' }), zeroS(s.T, P0)], rec: { heat: 0, solids: 0 } };
    const pu = clamp(p.purge, 0, 0.5), ws = s.w.map((x) => x * (1 - pu)), ms = Math.min(s.m * (1 - pu), sum(ws) / (1 - p.moist)), D = s.m * (1 - pu) - ms, wd = ws.map((x) => (x * D * 1e-5) / Math.max(sum(ws), 1e-12));
    const specs = [{ m: D, w: wd, P: P0 + 0.3, o: { pH: 6.5, co2: 0 } }, { m: ms, w: ws.map((x, i) => x - wd[i]), P: P0, o: { phase: 'solid', tss: s.tss } }, { m: s.m * pu, w: s.w.map((x) => x * pu), P: P0 + 0.3, o: { pH: s.pH, co2: 0 } }];
    if (p.drive === 'mvr') {
      const cw = compressorWork(100, 100 + bpe(100, SMAX) + p.dT + 3, p.etaComp), shaft = ((D / 3600) * cw.w) / 1000 + 2.5 * s.Q;
      return { outs: settle(specs, Hk(s) + shaft, s.T + 15), rec: { shaft, power: shaft / p.etaM, solids: ms, recovery: D / s.m, wSpec: cw.w / 3600, dP: P0 - s.P } };
    }
    const heat = ((D / 3600) * latentHeat(75) * 1.08) / 1000, aux = 2.5 * s.Q, outs = specs.map((q) => mkS(q.m, q.w, 45, q.P, q.o));
    return { outs, rec: { heat, Ts: 120, cool: Hk(s) + heat + aux - sum(outs.map(Hk)), shaft: aux, power: aux / p.etaM, solids: ms, recovery: D / s.m, dP: P0 - s.P } };
  },
};
function mixOf(a, b, P) { return UNITS.mix([{ ...a, P }, { ...b, P }]).outs[0]; }

// ---- reference membrane solution (element-by-element, from the RO suite) ------------------------------
const RO_D = Object.fromEntries(roSuite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const refCache = new Map();
function roReference(o) {
  const id = JSON.stringify([o.key, o.T, o.pH, o.recovery, o.flux, o.elements, o.boost2, o.Pp, o.p2, ION_IDS.map((k) => +(+o.ions[k] || 0).toPrecision(6))]);
  if (refCache.has(id)) return refCache.get(id);
  const M = MEMBRANES[o.key] || MEMBRANES.swhr, T = clamp(o.T, 1, 45);
  let r;
  try {
    r = simulateRO({ ...RO_D, ions: o.ions, Qf: 1000, T, pH: clamp(o.pH, 2, 12), salinityFactor: 1, membrane: o.key, A: M.A, B: M.B, area: M.area, spacerMil: M.spacerMil, elements: o.elements, design: 'auto', targetFlux: o.flux, mode: 'recovery', recovery: o.recovery,
      boost2: o.boost2 || 0, boost3: 0, recycle: 0, erd: 'none', model: o.key === 'nf' ? 'sk' : 'sd', ff: REF_FF, age: 0, nSeg: 2, Pp: o.Pp, pass2: !!o.p2, ...(o.p2 ? { membrane2: o.p2.key, pass2Frac: 100, pass2Rec: o.p2.rec, pass2Flux: o.p2.flux, pass2pH: o.p2.pH } : {}) });
  } catch (e) { throw new Error(`The ${M.name} block cannot reach ${o.recovery} % recovery on this feed (${e.message})`); }
  const pack = (p, feed, cfg, boost2) => {
    const Qp = p.perm.Q, rr = Qp / feed.Q, area = sum(cfg.vessels) * cfg.elements * cfg.M.area, fac = rr > 1e-9 ? -Math.log(1 - rr) / rr : 1, dP = p.Pf - p.conc.P, els = p.stages.flatMap((s) => s.els);
    const dPi = (osmoticPressureIons(scaleIons(feed.ions, fac), T) - osmoticPressureIons(p.perm.ions, T)) / 1e5, J = (Qp * 1000) / area;
    return { Qf: feed.Q, r: rr, Pf: p.Pf, dP, boost2: boost2 || 0, J, areaPer: area / feed.Q, T, tc: tcfRO(T), fac, NDP: p.Pf - o.Pp - dPi - dP / 2, SP: ION_IDS.map((k) => (feed.ions[k] > 0 ? Math.min(1, p.perm.ions[k] / feed.ions[k]) : 0)),
      leadRatio: Math.max(...els.map((e) => e.flux)) / J, vessels: cfg.vessels, elements: cfg.elements, elArea: cfg.M.area, pmax: cfg.M.pmax, name: cfg.M.name, concPerFeed: p.conc.Q / p.stages[p.stages.length - 1].nV / feed.Q * (sum(cfg.vessels) / 1), s2frac: p.stages[1] ? p.stages[1].feed.Q / feed.Q : 0, permIons: p.perm.ions };
  };
  const out = { p1: pack(r.p1, r.feed, r.cfg, o.boost2), p2: r.p2 ? pack(r.p2, r.pass2.feed, r.pass2.cfg, 0) : null };
  out.p1.concPerFeed = r.p1.conc.Q / r.p1.stages[r.p1.stages.length - 1].nV; // m³/h per vessel of the last stage at the reference
  if (out.p2) out.p2.concPerFeed = r.p2.conc.Q / r.p2.stages[r.p2.stages.length - 1].nV;
  if (refCache.size > 60) refCache.clear();
  refCache.set(id, out);
  return out;
}

/** Reference solutions needed by the selected template at design conditions. */
function references(v) {
  const base = { T: v.T, pH: v.pH, elements: Math.round(v.elements), Pp: v.Pperm }, ref = {};
  let ions = Object.fromEntries(ION_IDS.map((k) => [k, +v.ions?.[k] || 0]));
  if (v.template === 'nfro') { ref.nf = roReference({ ...base, key: 'nf', ions, recovery: v.nfRec, flux: v.nfFlux, boost2: 0 }).p1; ions = ref.nf.permIons; }
  const two = v.template === 'swro2', a = roReference({ ...base, key: v.membrane, ions, recovery: v.recovery, flux: v.flux, boost2: v.boost2, p2: two ? { key: v.membrane2, rec: v.pass2Rec, flux: v.pass2Flux, pH: v.pass2pH } : null });
  ref.ro1 = a.p1; if (two) ref.ro2 = a.p2;
  return ref;
}

// ---- flowsheet construction ---------------------------------------------------------------------------
const ROWS = ['Intake and pretreatment', 'Desalination', 'Thermal and brine treatment', 'Product water', 'Brine and residuals'];
const PRE = { daf_dmf: 'Dissolved-air flotation and media filters', dmf: 'Dual-media filters', uf: 'Ultrafiltration' };
const NAMES = { raw: 'Raw feed', s1: 'Screened feed', s2: 'Intake pump discharge', medIn: 'Seawater to MED', s2b: 'Seawater to RO line', s2c: 'Preheated seawater', s2d: 'Seawater + warm reject', medD: 'MED distillate', medD2: 'Cooled distillate', medB: 'MED brine', medCW: 'MED cooling-water reject',
  s3: 'Coagulated feed', s4: 'Filtered water', bw: 'Backwash waste', s5: 'Cartridge filtrate', s6: 'Conditioned feed', byp: 'Blend bypass', s6b: 'Feed to membranes', nfF: 'NF feed', nfP: 'NF permeate', nfC: 'NF concentrate', nfC2: 'NF concentrate (let down)', nfT: 'NF permeate (boosted)',
  s7: 'Feed + second-pass return', lpF: 'LP feed to pressure exchanger', hpS: 'HP pump suction', hpD: 'HP pump discharge', pxHP: 'HP feed from pressure exchanger', bst: 'Circulation pump discharge', roF: 'Membrane feed', perm1: 'First-pass permeate', conc: 'RO concentrate', brine1: 'Depressurised concentrate',
  p1d: 'Permeate after caustic', p2in: 'Second-pass feed', p2byp: 'Second-pass bypass', p2F: 'Second-pass pump discharge', perm2: 'Second-pass permeate', p2cH: 'Second-pass concentrate', p2c: 'Second-pass concentrate return', prodA: 'Blended permeate', prodB: 'Blended product', prodC: 'Remineralised water', prodD: 'Disinfected water', prodE: 'Product tank outlet', product: 'Product water',
  bcD: 'Concentrator distillate', bcC: 'Concentrated brine', crC: 'Crystalliser condensate', solids: 'Salt cake', purge: 'Crystalliser purge', brineMix: 'Blended brine and waste', brine: 'Discharge' };

function buildFlowsheet(v, ref, C = {}) {
  const t = v.template, U = [], tears = [], prod = [], brines = [], etaM = v.etaMotor / 100, lp = { eta: v.etaLP / 100, etaM }, hp = { eta: v.etaHP / 100, etaM };
  const add = (id, name, type, row, ins, outs, p = {}) => U.push({ id, name, type, row: ROWS[row], in: ins, out: outs, p: typeof p === 'function' ? p : () => p });
  const ro = (key, refK, rec, area) => ({ ref: refK, recovery: rec / 100, area, key, Pp: v.Pperm, ff: v.roFF, kSP: v.kSP, ...hp });
  let cur = 'raw';
  add('SCR', 'Intake and screens', 'screen', 0, [cur], ['s1'], { dp: v.screenDp }); cur = 's1';
  add('IP', 'Intake pump', 'pump', 0, [cur], ['s2'], { Pout: v.Pint, ...lp }); cur = 's2';
  if (t === 'hybrid') {
    add('MS', 'Seawater split to MED', 'split', 0, [cur], ['medIn', 's2b'], { frac: v.medFrac / 100 }); cur = 's2b';
    add('MED', 'MED evaporator and condenser', 'med', 2, ['medIn'], ['medD', 'medB', 'medCW'], { make: v.medMakeup / 100, rec: v.medRec / 100, GOR: v.medGOR, Ts: v.medTs, Tlast: v.medTlast, secElec: v.medSecElec, loss: v.medLoss / 100 });
    let dk = 'medD';
    if (v.useHX) { add('HX', 'Distillate cooler / feed preheater', 'hx', 2, ['medD', cur], ['medD2', 's2c'], { UA: v.hxUA, U: v.hxU, dp: 0.3 }); dk = 'medD2'; cur = 's2c'; }
    if (v.rejectToRO) { add('MXW', 'Warm-reject mixer', 'mix', 0, [cur, 'medCW'], ['s2d']); cur = 's2d'; } else brines.push('medCW');
    prod.push(dk); brines.push('medB');
  }
  if (v.useCoag) { add('COAG', 'Coagulation and flocculation', 'dose', 0, [cur], ['s3'], { chems: [['Ferric chloride', 'ferric', v.coagDose]] }); cur = 's3'; }
  const hasBW = v.pretreat !== 'none' && v.preRec < 100;
  if (v.pretreat !== 'none') { add('PRE', PRE[v.pretreat] || 'Filtration', 'filter', 0, [cur], hasBW ? ['s4', 'bw'] : ['s4'], { recovery: v.preRec / 100, tssRem: v.tssRem / 100, dp: v.preDp }); cur = 's4'; }
  if (v.useCartridge) { add('CF', 'Cartridge filters', 'filter', 0, [cur], ['s5'], { recovery: 1, tssRem: 0.9, dp: v.cartDp }); cur = 's5'; }
  const ch = [];
  if (v.useAcid) ch.push(['Sulphuric acid', 'acid', v.acidDose]);
  if (v.useAntiscalant) ch.push(['Antiscalant', 'antiscalant', v.asDose]);
  if (v.useSBS) ch.push(['Sodium bisulphite', 'sbs', v.sbsDose]);
  if (ch.length) { add('CHEM', 'Chemical conditioning', 'dose', 0, [cur], ['s6'], { chems: ch }); cur = 's6'; }
  if (t === 'bwro' && v.blendPct > 0) { add('BYP', 'Blend-bypass splitter', 'split', 1, [cur], ['byp', 's6b'], { frac: v.blendPct / 100 }); cur = 's6b'; prod.push('byp'); }
  if (t === 'nfro') {
    add('NFP', 'NF feed pump', 'pump', 1, [cur], ['nfF'], (S, st) => ({ Pout: st.P.nf, ...hp }));
    add('NF', 'Nanofiltration pass', 'ro', 1, ['nfF'], ['nfP', 'nfC'], ro('nf', ref.nf, v.nfRec, C.areaNF));
    add('NFV', 'NF concentrate valve', 'valve', 4, ['nfC'], ['nfC2'], { Pout: P0 + 0.5 }); brines.push('nfC2');
    add('TP', 'NF permeate transfer pump', 'pump', 1, ['nfP'], ['nfT'], { Pout: Math.max(v.Pperm, v.Pint - 1), ...lp }); cur = 'nfT';
  }
  const two = t === 'swro2';
  if (two) { add('RCY', 'Second-pass concentrate return', 'mix', 1, [cur, 'p2c'], ['s7']); cur = 's7'; tears.push('p2c'); }
  if (v.erd === 'px') {
    add('PXS', 'Feed split to pressure exchanger', 'split', 1, [cur], ['lpF', 'hpS'], (S) => ({ Q: S.conc.Q * (1 + v.pxOver / 100) }));
    add('HPP', 'High-pressure pump', 'pump', 1, ['hpS'], ['hpD'], (S, st) => ({ Pout: st.P.ro1, ...hp }));
    add('PX', 'Pressure exchanger', 'px', 1, ['conc', 'lpF'], ['pxHP', 'brine1'], { leak: v.pxLeak / 100, mix0: v.pxMix / 100, dpHP: v.pxDpHP, dpLP: v.pxDpLP });
    add('BST', 'Circulation pump', 'pump', 1, ['pxHP'], ['bst'], (S, st) => ({ Pout: st.P.ro1, eta: v.etaBooster / 100, etaM }));
    add('MXF', 'Membrane feed mixer', 'mix', 1, ['hpD', 'bst'], ['roF']);
    tears.push('conc');
  } else add('HPP', 'High-pressure pump', 'pump', 1, [cur], ['roF'], (S, st) => ({ Pout: st.P.ro1, ...hp }));
  add('RO1', t === 'bwro' ? 'Brackish RO pass' : 'RO pass', 'ro', 1, ['roF'], ['perm1', 'conc'], ro('ro1', ref.ro1, v.recovery, C.area1));
  if (v.erd === 'turbine') add('TRB', 'Pelton turbine', 'turbine', 1, ['conc'], ['brine1'], { Pout: P0 + 0.3, eta: v.etaTurbine / 100, etaM });
  else if (v.erd !== 'px') add('BV', 'Concentrate control valve', 'valve', 1, ['conc'], ['brine1'], { Pout: P0 + 0.5 });
  let pk = 'perm1';
  if (two) {
    add('CAU', 'Caustic dosing', 'dose', 3, ['perm1'], ['p1d'], { pHset: v.pass2pH });
    add('P2S', 'Second-pass splitter', 'split', 3, ['p1d'], ['p2in', 'p2byp'], { frac: v.pass2Frac / 100 });
    add('P2P', 'Second-pass pump', 'pump', 3, ['p2in'], ['p2F'], (S, st) => ({ Pout: st.P.ro2, ...hp }));
    add('RO2', 'Second RO pass', 'ro', 3, ['p2F'], ['perm2', 'p2cH'], ro('ro2', ref.ro2, v.pass2Rec, C.area2));
    add('P2V', 'Second-pass concentrate valve', 'valve', 3, ['p2cH'], ['p2c'], { Pout: Math.max(P0 + 0.5, v.Pint - 0.2) });
    add('MXP', 'Permeate blend', 'mix', 3, ['perm2', 'p2byp'], ['prodA']); pk = 'prodA';
  }
  prod.unshift(pk);
  let bk = 'brine1';
  if (t === 'mld') {
    add('BC', 'Brine concentrator (MVC)', 'bc', 2, [bk], ['bcD', 'bcC'], { target: v.bcTarget, dT: v.bcDT, etaComp: v.etaComp / 100, aux: v.bcAux, etaM }); prod.push('bcD'); bk = 'bcC';
    if (v.useCryst) { add('CR', 'Crystalliser', 'cryst', 2, [bk], ['crC', 'solids', 'purge'], { purge: v.crystPurge / 100, moist: v.crystMoist / 100, drive: v.crystDrive, dT: v.bcDT, etaComp: v.etaComp / 100, etaM }); prod.push('crC'); bk = 'purge'; }
  }
  brines.unshift(bk);
  if (prod.length > 1) { add('MXQ', 'Product blending', 'mix', 3, prod, ['prodB']); pk = 'prodB'; }
  if (v.useRemin) { add('REM', v.reminMethod === 'calcite' ? 'Calcite contactor + CO₂' : 'Lime + CO₂ remineralisation', 'remin', 3, [pk], ['prodC'], { method: v.reminMethod, hardness: v.hardTarget, lsi: v.lsiTarget }); pk = 'prodC'; }
  if (v.useDisinfect) { add('DIS', 'Disinfection', 'dose', 3, [pk], ['prodD'], { chems: [['Sodium hypochlorite', 'hypo', v.clDose]] }); pk = 'prodD'; }
  add('PT', 'Product tank', 'tank', 3, [pk], ['prodE'], { hours: v.tankHours });
  add('PP', 'Product transfer pump', 'pump', 3, ['prodE'], ['product'], { Pout: v.Pdist, ...lp });
  if (hasBW) brines.push('bw');
  add('MXB', 'Brine and waste blending', 'mix', 4, brines, ['brineMix']);
  add('OUT', 'Outfall / discharge', 'valve', 4, ['brineMix'], ['brine'], { Pout: P0 });
  return { units: U, tears, outs: ['product', 'brine', ...(t === 'mld' && v.useCryst ? ['solids'] : [])] };
}

// ---- sequential-modular solver with tear streams --------------------------------------------------
const vecOf = (S, tears, P) => [...tears.flatMap((k) => [S[k].m, S[k].T, S[k].P, S[k].co2, ...S[k].w]), ...Object.values(P)];
export function solveFlowsheet(fs, raw, guess, Pguess, opt = {}) {
  const { method = 'wegstein', tol = 1e-9, maxIter = 80, damping = 0.5 } = opt, S = { raw, ...guess }, st = { P: { ...Pguess }, Pn: {} }, keysP = Object.keys(Pguess);
  let recs = [], xPrev = null, gPrev = null, res = 1, it = 0;
  const history = [];
  for (it = 1; it <= maxIter; it++) {
    const x = vecOf(S, fs.tears, st.P), meta = fs.tears.map((k) => keep(S[k]));
    st.Pn = { ...st.P }; recs = [];
    for (const u of fs.units) {
      const ins = u.in.map((k) => S[k]), r = UNITS[u.type](ins, u.p(S, st), st);
      u.out.forEach((k, i) => (S[k] = r.outs[i]));
      recs.push({ u, inS: ins, outS: r.outs.slice(0, u.out.length), rec: r.rec || {} });
    }
    const g = vecOf(S, fs.tears, st.Pn);
    res = 0;
    for (let i = 0; i < x.length; i++) res = Math.max(res, Math.abs(g[i] - x[i]) / (Math.abs(x[i]) + 1e-9));
    history.push(res);
    if (res < tol) { st.P = { ...st.Pn }; break; }
    let nx;
    if (method === 'direct' || (method === 'wegstein' && it < 3)) nx = g;
    else if (method === 'damped') nx = g.map((gi, i) => x[i] + damping * (gi - x[i]));
    else nx = g.map((gi, i) => { const dx = x[i] - xPrev[i]; if (Math.abs(dx) < 1e-14) return gi; const s = (gi - gPrev[i]) / dx, q = clamp(s / (s - 1), -5, 0); return Number.isFinite(q) ? q * x[i] + (1 - q) * gi : gi; });
    xPrev = x; gPrev = g;
    let o = 0;
    fs.tears.forEach((k, j) => { const m = Math.max(0, nx[o]), T = nx[o + 1], P = nx[o + 2], c = Math.max(0, nx[o + 3]); S[k] = mkS(m, nx.slice(o + 4, o + 4 + NI).map((q) => Math.max(0, q)), T, P, { ...meta[j], co2: c, pH: S[k].pH }); o += 4 + NI; });
    keysP.forEach((k, j) => (st.P[k] = nx[o + j]));
  }
  return { S, recs, P: st.P, residual: res, iterations: Math.min(it, maxIter), converged: res < tol, history };
}

/** Closure of the total-mass, ion and energy balances of one unit record (absolute errors and scales). */
export function unitBalance(r) {
  const k = r.rec, mIn = sum(r.inS.map((s) => s.m)) + (k.chemMass || 0), mOut = sum(r.outS.map((s) => s.m));
  let ion = 0, scale = 1e-12;
  for (let i = 0; i < NI; i++) { const a = sum(r.inS.map((s) => s.w[i])) + (k.chemIons?.[i] || 0) + (k.gen?.[i] || 0), b = sum(r.outS.map((s) => s.w[i])); ion = Math.max(ion, Math.abs(a - b) / Math.max(1e-9, a, sum(r.inS.map((s) => s.salt)) * 1e-6)); scale = Math.max(scale, a); }
  const hIn = sum(r.inS.map(Hk)) + (k.shaft || 0) + (k.heat || 0) + (k.Hadd || 0), hOut = sum(r.outS.map(Hk)) + (k.cool || 0);
  return { mass: mIn > 0 ? Math.abs(mIn - mOut) / mIn : 0, ion, energy: Math.abs(hIn - hOut) / Math.max(1, Math.abs(hIn)), mIn, mOut, hIn, hOut };
}

/** One steady-state case. cond = { T, sf (salinity factor), load (fraction of design feed) }, C = fixed design sizes or {}. */
export function solveCase(v, ref, cond = {}, C = {}) {
  const T = cond.T ?? v.T, sf = cond.sf ?? 1, load = cond.load ?? 1, ions = scaleIons(v.ions || {}, sf), r = clamp(v.recovery / 100, 0.01, 0.98);
  const raw = fromVol(v.Qf * load, ions, T, P0, v.pH, v.tss), fs = buildFlowsheet(v, ref, C), guess = {}, Pg = { ro1: ref.ro1.Pf };
  if (fs.tears.includes('conc')) guess.conc = fromVol(raw.Q * 0.85 * (1 - r), scaleIons(ions, 1 / (1 - r)), T, ref.ro1.Pf - ref.ro1.dP, v.pH);
  if (fs.tears.includes('p2c')) guess.p2c = fromVol(raw.Q * 0.04, scaleIons(ions, 0.02), T, v.Pint, v.pH);
  if (ref.ro2) Pg.ro2 = ref.ro2.Pf;
  if (ref.nf) Pg.nf = ref.nf.Pf;
  const sol = solveFlowsheet(fs, raw, guess, Pg, { method: v.tearMethod, tol: v.tearTol, maxIter: Math.round(v.maxIter), damping: v.damping });
  const dead = deadState(raw), T0 = dead.T + KELVIN, S = sol.S, get = (id) => sol.recs.find((q) => q.u.id === id);
  let power = 0, heat = 0, cool = 0, exIn = 0, sludge = 0, solids = 0;
  const chem = {}, consumers = {};
  for (const q of sol.recs) {
    const k = q.rec, e = (list) => sum(list.map((s) => sum(exergy(s, dead))));
    q.exQ = k.heat ? k.heat * (1 - T0 / ((k.Ts ?? 100) + KELVIN)) : 0;
    q.exAdd = k.chemMass > 0 && q.inS[0].m > 0 ? (k.chemMass * exergy(q.inS[0], dead)[0]) / q.inS[0].m : 0; // dosed mass arrives at stream pressure and temperature
    q.exDest = e(q.inS) + Math.max(0, k.power || 0) + q.exQ + q.exAdd - e(q.outS) - Math.max(0, -(k.power || 0));
    q.bal = unitBalance(q);
    power += k.power || 0; heat += k.heat || 0; cool += k.cool || 0; exIn += Math.max(0, k.power || 0) + q.exQ + q.exAdd; sludge += k.sludge || 0; solids += k.solids || 0;
    if (k.power) consumers[q.u.name] = (consumers[q.u.name] || 0) + k.power;
    for (const [n, x] of Object.entries(k.chem || {})) chem[n] = (chem[n] || 0) + x;
  }
  const product = S.product, brine = S.brine, outs = fs.outs.map((k) => S[k]), chemMass = sum(sol.recs.map((q) => q.rec.chemMass || 0));
  const least = sum(outs.map((s) => exergy(s, dead)[1])) - exergy(raw, dead)[1], ro1 = get('RO1')?.rec || {}, ro2 = get('RO2')?.rec, nf = get('NF')?.rec;
  const plant = {
    mass: { in: raw.m + chemMass, out: sum(outs.map((s) => s.m)) },
    ion: ION_IDS.map((_, i) => ({ in: raw.w[i] + sum(sol.recs.map((q) => (q.rec.chemIons?.[i] || 0) + (q.rec.gen?.[i] || 0))), out: sum(outs.map((s) => s.w[i])) })),
    energy: { in: Hk(raw) + sum(sol.recs.map((q) => (q.rec.shaft || 0) + (q.rec.heat || 0) + (q.rec.Hadd || 0))), out: sum(outs.map(Hk)) + cool },
  };
  return { v, cond: { T, sf, load }, raw, fs, sol, S, dead, get, product, brine, solidsS: S.solids, power, heat, cool, exIn, exDest: sum(sol.recs.map((q) => q.exDest)), least, chem, consumers, sludge, solids, ro1, ro2, nf, plant,
    recovery: raw.Q > 0 ? product.Q / raw.Q : 0, secElec: product.Q > 0 ? power / product.Q : 0, secThermal: product.Q > 0 ? heat / product.Q : 0, etaII: exIn > 0 ? least / exIn : 0, prodIons: ionsOf(product) };
}

/** Design sizes taken from the converged design case. */
function designOf(res) {
  const g = (id) => res.get(id), q = (id) => g(id)?.inS[0]?.Q || 0;
  return { area1: res.ro1.area, area2: res.ro2?.area, areaNF: res.nf?.area, P1: res.ro1.Preq, hpPower: g('HPP')?.rec.power || 0, preQ: q('PRE') || q('CF') || res.raw.Q, pxQ: q('PX'), bcQ: q('BC'), medQ: q('MED'), J1: res.ro1.flux, rawQ: res.raw.Q };
}

/** Utilisation of every equipment limit in a case (value ÷ limit); the largest is the bottleneck. */
function limitsOf(res, D, v) {
  const L = [], add = (name, value, limit, unit) => { if (limit > 0 && Number.isFinite(value)) L.push({ name, value, limit, unit, util: value / limit }); }, g = (id) => res.get(id);
  add('RO feed pressure (pump head / element rating)', res.ro1.Preq, Math.min(res.ro1.pmax || 1e9, D.P1 * (1 + v.pumpMargin / 100)), 'bar');
  add('High-pressure pump motor', g('HPP')?.rec.power || 0, D.hpPower * (1 + v.pumpMargin / 100), 'kW');
  add('Membrane flux', res.ro1.flux, D.J1 * (v.fluxMargin / 100), 'L/m²·h');
  add('Pretreatment hydraulic load', g('PRE')?.inS[0].Q || g('CF')?.inS[0].Q || res.raw.Q, D.preQ * (1 + v.hydMargin / 100), 'm³/h');
  add('Product TDS', res.product.tds, v.limTDS, 'mg/L');
  if (res.ro1.concPerVessel > 0) add('Minimum concentrate flow per vessel', v.limConc, res.ro1.concPerVessel, 'm³/h');
  if (D.pxQ > 0) add('Pressure-exchanger flow', g('PX')?.inS[0].Q || 0, D.pxQ * (1 + v.hydMargin / 100), 'm³/h');
  if (v.maxRecChem > 0) add('Scaling limit on recovery', 100 * res.ro1.recovery, v.maxRecChem, '%');
  if (g('MED')) add('MED condenser temperature rise', g('MED').rec.dTcw, v.medMaxDT, 'K');
  if (D.bcQ > 0) add('Brine-concentrator feed', g('BC')?.inS[0].Q || 0, D.bcQ * (1 + v.hydMargin / 100), 'm³/h');
  if (res.ro2) add('Second-pass feed pressure', res.ro2.Preq, res.ro2.pmax, 'bar');
  L.sort((a, b) => b.util - a.util);
  return L;
}

/** Design case plus the reference solutions and fixed sizes. */
export function plantDesign(v) {
  const ref = references(v), res = solveCase(v, ref), D = designOf(res);
  return { ref, res, D, C: { area1: D.area1, area2: D.area2, areaNF: D.areaNF } };
}

/**
 * Lumped dynamics of the product tank with a PI level controller (variable speed) or train on/off control.
 * States: stored volume, controller integral, cumulative net inflow, energy, energy cost. Integrated with RK4.
 */
export function dynamicSim(o) {
  const n = Math.max(8, Math.round((o.hours * 60) / o.dtMin)), dt = o.hours / n, onoff = o.mode === 'onoff', fixed = o.mode === 'const';
  let y = [o.L0 * o.Vmax, 0, 0, 0, 0], t = 0, running = clamp(Math.ceil(o.uff * o.nTrains - 1e-9), 0, o.nTrains), last = -9, spill = 0, unmet = 0, starts = 0;
  const uOf = (tt, V, I) => { if (onoff) return running / o.nTrains; if (fixed) return o.uff; return clamp(o.uff + o.Kp * (o.Lsp(tt) - V / o.Vmax) + o.Ki * I, o.uMin, 1); };
  const f = (tt, [V, I]) => { const e = o.Lsp(tt) - V / o.Vmax, raw = o.uff + o.Kp * e + o.Ki * I, u = uOf(tt, V, I), sat = (raw > 1 && e > 0) || (raw < o.uMin && e < 0), q = u * o.Qn, pw = q * o.sec(u); return [q - o.demand(tt), onoff || fixed || sat ? 0 : e, q - o.demand(tt), pw, pw * o.price(tt)]; };
  const out = { t: [], L: [], u: [], Qd: [], P: [], sp: [] };
  for (let k = 0; k <= n; k++) {
    const lev = y[0] / o.Vmax;
    if (onoff && t - last >= 0.5 - 1e-9) { if (lev < o.onLow && running < o.nTrains) { running++; starts++; last = t; } else if (lev > o.offHigh && running > 0) { running--; last = t; } }
    const u = uOf(t, y[0], y[1]);
    out.t.push(t); out.L.push(100 * lev); out.u.push(100 * u); out.Qd.push(o.demand(t)); out.P.push(u * o.Qn * o.sec(u)); out.sp.push(100 * o.Lsp(t));
    if (k === n) break;
    y = rk4(f, y, t, t + dt, 1).y[1]; t += dt;
    if (y[0] > o.Vmax) { spill += y[0] - o.Vmax; y[0] = o.Vmax; }
    if (y[0] < 0) { unmet -= y[0]; y[0] = 0; }
  }
  const i0 = out.t.findIndex((x) => x >= o.hours - 24 - 1e-9), last24 = (a) => a.slice(Math.max(0, i0)), e = last24(out.L).map((L, i) => Math.abs(L - last24(out.sp)[i]));
  const W = last24(out.P), tt = last24(out.t), price = tt.map((x) => o.price(x));
  let energy = 0, cost = 0;
  for (let i = 1; i < tt.length; i++) { const h = tt[i] - tt[i - 1]; energy += 0.5 * (W[i] + W[i - 1]) * h; cost += 0.5 * (W[i] * price[i] + W[i - 1] * price[i - 1]) * h; }
  return { ...out, i0: Math.max(0, i0), dt, n, Vend: y[0], net: y[2], spill, unmet, starts, energy, cost, iae: (sum(e) * dt), Lmin: Math.min(...last24(out.L)), Lmax: Math.max(...last24(out.L)), sat: last24(out.u).filter((x) => x >= 99.999 || x <= 100 * o.uMin + 1e-3).length / Math.max(1, last24(out.u).length), totalEnergy: y[3], totalCost: y[4] };
}

const streamOut = (s) => ({ Q: s.Q, T: s.T, P: s.P, pH: s.pH, tds: s.tds, ions: Object.fromEntries(Object.entries(ionsOf(s)).map(([k, x]) => [k, +x.toPrecision(6)])) });
const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.type === 'table' ? f.value.map((r) => ({ ...r })) : f.value]));
const is = (...t) => (v) => t.includes(v.template);
const memOpts = Object.entries(MEMBRANES).map(([value, m]) => ({ value, label: m.name }));

const suite = {
  id: 'plant', num: 3, title: 'Whole-Plant Process Simulation', short: 'Plant flowsheet', icon: '🏭',
  tagline: 'Steady-state flowsheet from intake to outfall with recycle convergence, full stream tables, energy, exergy, off-design cases and tank dynamics.',
  description: 'Solves a complete desalination flowsheet unit by unit in the direction of flow and converges the recycle (tear) streams with Wegstein acceleration. Every stream carries mass flow, temperature, pressure and the mass flow of each ion, so total, component and energy balances close around every unit and around the plant, and exergy destruction is located unit by unit. Membrane passes use a reduced model anchored to the element-by-element solution of the RO suite; thermal desalination, the brine concentrator and the crystalliser are reduced energy-balance blocks. Off-design cases identify the first equipment limit reached, and a tank model with a PI level controller shows operation over a daily demand profile.',
  guide: [
    'Choose a flowsheet template and switch optional units on or off; the Flowsheet tab draws the result.',
    'Enter the feed (or pull it from the Case page) and the key parameter of each unit — recovery, efficiencies, doses, targets.',
    'Run. Check the stream table and the balance closure on the Verify tab, then the power, chemical and exergy breakdowns.',
    'Read the off-design table to see which unit limits the plant at minimum and maximum temperature, salinity and part load, and the daily operation plots for the tank and controller.',
  ],
  implemented: ['total mass-balance', 'component mass-balance', 'steady-flow energy', 'enthalpy balance', 'entropy balance', 'exergy-balance', 'momentum/pressure-drop', 'flash equation', 'pump equation', 'compressor equation', 'valve equation', 'heat-exchanger equation', 'overall heat-transfer', 'logarithmic-mean-temperature-difference', 'effectiveness-ntu', 'reactor balance', 'separator equation', 'evaporator balance', 'crystallizer balance', 'membrane mass-transfer',
    'membrane-process flowsheet', 'ro-med model', 'ro-crystallizer', 'ro-zld', 'process-exergy', 'process-environmental', 'steady-state/dynamic hybrid', 'mechanistic-surrogate',
    'initial inventories', 'vessel liquid level', 'composition', 'temperature', 'pressure', 'equipment state', 'controller state', 'plant feed-flow/composition/temperature/pressure', 'product-water specification', 'discharge-pressure', 'utility steam', 'cooling-water', 'ambient heat-loss', 'electrical-power constraint', 'terminal process specification',
    'process-flowsheet construction', 'feed-water definition', 'material-stream management', 'mass balancing', 'energy balancing', 'thermodynamic-property calculation', 'pumps and compressors', 'valves and piping', 'mixers and splitters', 'separators', 'heat exchangers', 'membrane units', 'reactors', 'evaporators', 'condensers', 'crystallisers', 'chemical-dosing systems', 'utilities', 'recycle streams', 'energy recovery', 'steady-state simulation', 'dynamic simulation', 'process control', 'equipment sizing', 'sensitivity analysis', 'environmental accounting'],
  equationsNote: 'Steady state with lumped unit models. Membrane passes scale net driving pressure, pressure drop and ion passage from one element-by-element reference solution (suite 1) with flux, temperature and salinity, which is accurate near the design point and approximate far from it. MED, the brine concentrator and the crystalliser are reduced blocks (performance ratio, compressor work, single-effect heat demand) — use suites 6 and 9 for their detailed design. Enthalpy neglects heat of mixing; salinity exergy uses an ideal-mixture form scaled by the osmotic coefficient of the feed, and the exergy of dosed chemicals and solid salts is not counted. Seawater property correlations are extrapolated above about 160 g/kg. pH follows a simplified carbonate system. Dosing reactions are stoichiometric.',

  inputs: [
    { group: 'Feed water', help: 'Plant boundary condition: what the intake delivers.', fields: [
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.seawater.ions, help: 'Full ionic analysis carried through every stream.' },
      { key: 'Qf', label: 'Intake flow', unit: 'm³/h', value: 1000, min: 1, max: 2e5 },
      { key: 'T', label: 'Feed temperature (design)', unit: '°C', value: 25, min: 2, max: 44 },
      { key: 'pH', label: 'Feed pH', unit: '', value: 8.1, min: 4, max: 10 },
      { key: 'tss', label: 'Suspended solids', unit: 'mg/L', value: 6, min: 0, max: 500 },
    ] },
    { group: 'Flowsheet', help: 'Template and optional units. The Flowsheet tab shows the resulting process flow diagram after a run.', fields: [
      { key: 'template', label: 'Flowsheet template', type: 'select', value: 'swro', options: [{ value: 'swro', label: 'Seawater RO, single pass' }, { value: 'swro2', label: 'Seawater RO, two pass' }, { value: 'bwro', label: 'Brackish RO with blend bypass' }, { value: 'hybrid', label: 'Hybrid RO + MED' }, { value: 'mld', label: 'RO → brine concentrator → crystalliser (MLD/ZLD)' }, { value: 'nfro', label: 'Nanofiltration → RO' }] },
      { key: 'pretreat', label: 'Pretreatment', type: 'select', value: 'daf_dmf', options: [{ value: 'daf_dmf', label: 'Dissolved-air flotation + media filters' }, { value: 'dmf', label: 'Dual-media filters' }, { value: 'uf', label: 'Ultrafiltration' }, { value: 'none', label: 'None (well water)' }] },
      { key: 'useCoag', label: 'Coagulation and flocculation', type: 'bool', value: true },
      { key: 'useCartridge', label: 'Cartridge filters', type: 'bool', value: true },
      { key: 'useAcid', label: 'Acid dosing', type: 'bool', value: false },
      { key: 'useAntiscalant', label: 'Antiscalant dosing', type: 'bool', value: true },
      { key: 'useSBS', label: 'Bisulphite dechlorination', type: 'bool', value: true },
      { key: 'erd', label: 'Energy recovery', type: 'select', value: 'px', options: [{ value: 'px', label: 'Isobaric pressure exchanger' }, { value: 'turbine', label: 'Pelton turbine' }, { value: 'none', label: 'None (control valve)' }] },
      { key: 'useRemin', label: 'Remineralisation', type: 'bool', value: true },
      { key: 'useDisinfect', label: 'Final disinfection', type: 'bool', value: true },
      { key: 'useHX', label: 'Distillate cooler / feed preheater', type: 'bool', value: true, showIf: is('hybrid') },
      { key: 'rejectToRO', label: 'Send warm MED reject to the RO feed', type: 'bool', value: true, help: 'The classic hybrid synergy: warmer RO feed needs less pressure.', showIf: is('hybrid') },
      { key: 'useCryst', label: 'Crystalliser after the concentrator (zero liquid discharge)', type: 'bool', value: true, showIf: is('mld') },
    ] },
    { group: 'Intake and pretreatment', fields: [
      { key: 'screenDp', label: 'Screen head loss', unit: 'bar', value: 0.05, min: 0, max: 1 },
      { key: 'Pint', label: 'Intake pump discharge pressure', unit: 'bar a', value: 4.5, min: 1.5, max: 12 },
      { key: 'etaLP', label: 'Low-pressure pump efficiency', unit: '%', value: 82, min: 30, max: 92 },
      { key: 'coagDose', label: 'Ferric chloride dose', unit: 'mg/L', value: 5, min: 0, max: 60, showIf: (v) => v.useCoag },
      { key: 'preRec', label: 'Pretreatment recovery', unit: '%', value: 95, min: 60, max: 100, help: 'Share of the flow that leaves as filtrate; the rest is backwash and flotation waste.', showIf: (v) => v.pretreat !== 'none' },
      { key: 'tssRem', label: 'Suspended-solids removal', unit: '%', value: 99, min: 0, max: 100, showIf: (v) => v.pretreat !== 'none' },
      { key: 'preDp', label: 'Pretreatment pressure loss', unit: 'bar', value: 0.8, min: 0, max: 4, showIf: (v) => v.pretreat !== 'none' },
      { key: 'cartDp', label: 'Cartridge-filter pressure loss', unit: 'bar', value: 0.3, min: 0, max: 2, showIf: (v) => v.useCartridge },
      { key: 'acidDose', label: 'Sulphuric acid dose', unit: 'mg/L', value: 15, min: 0, max: 200, showIf: (v) => v.useAcid },
      { key: 'asDose', label: 'Antiscalant dose', unit: 'mg/L', value: 2.5, min: 0, max: 15, showIf: (v) => v.useAntiscalant },
      { key: 'sbsDose', label: 'Sodium bisulphite dose', unit: 'mg/L', value: 3, min: 0, max: 20, showIf: (v) => v.useSBS },
    ] },
    { group: 'Membrane system', fields: [
      { key: 'membrane', label: 'Element class', type: 'select', value: 'swhr', options: memOpts },
      { key: 'recovery', label: 'RO recovery (on membrane feed)', unit: '%', value: 45, min: 5, max: 92 },
      { key: 'flux', label: 'Design average flux', unit: 'L/m²·h', value: 14, min: 4, max: 40 },
      { key: 'elements', label: 'Elements per vessel', unit: '', value: 7, min: 1, max: 8, step: 1 },
      { key: 'boost2', label: 'Inter-stage booster (two-stage arrays)', unit: 'bar', value: 0, min: 0, max: 30 },
      { key: 'Pperm', label: 'Permeate pressure', unit: 'bar a', value: 1.3, min: 1, max: 10 },
      { key: 'etaHP', label: 'High-pressure pump efficiency', unit: '%', value: 86, min: 30, max: 93 },
      { key: 'etaMotor', label: 'Motor and drive efficiency', unit: '%', value: 95.5, min: 60, max: 99 },
      { key: 'pxLeak', label: 'Pressure-exchanger lubrication flow', unit: '% of brine', value: 1.5, min: 0, max: 8, showIf: (v) => v.erd === 'px' },
      { key: 'pxMix', label: 'Pressure-exchanger volumetric mixing', unit: '%', value: 6, min: 0, max: 15, showIf: (v) => v.erd === 'px' },
      { key: 'pxOver', label: 'Pressure-exchanger over-flush', unit: '%', value: 0, min: 0, max: 20, showIf: (v) => v.erd === 'px' },
      { key: 'pxDpHP', label: 'Pressure-exchanger HP differential', unit: 'bar', value: 0.7, min: 0, max: 3, showIf: (v) => v.erd === 'px' },
      { key: 'pxDpLP', label: 'Pressure-exchanger LP differential', unit: 'bar', value: 0.6, min: 0, max: 3, showIf: (v) => v.erd === 'px' },
      { key: 'etaBooster', label: 'Circulation pump efficiency', unit: '%', value: 80, min: 30, max: 92, showIf: (v) => v.erd === 'px' },
      { key: 'etaTurbine', label: 'Turbine efficiency', unit: '%', value: 88, min: 30, max: 93, showIf: (v) => v.erd === 'turbine' },
      { key: 'membrane2', label: 'Second-pass element class', type: 'select', value: 'bwhr', options: memOpts, showIf: is('swro2') },
      { key: 'pass2Frac', label: 'Share of permeate sent to the second pass', unit: '%', value: 70, min: 0, max: 100, showIf: is('swro2') },
      { key: 'pass2Rec', label: 'Second-pass recovery', unit: '%', value: 90, min: 50, max: 95, showIf: is('swro2') },
      { key: 'pass2Flux', label: 'Second-pass flux', unit: 'L/m²·h', value: 32, min: 10, max: 45, showIf: is('swro2') },
      { key: 'pass2pH', label: 'Second-pass feed pH', unit: '', value: 10, min: 7, max: 11, help: 'Caustic is dosed to this pH to improve boron rejection.', showIf: is('swro2') },
      { key: 'blendPct', label: 'Pretreated feed bypassed to the product', unit: '%', value: 8, min: 0, max: 100, help: 'Blending restores minerals in brackish plants; 100 % bypasses the membranes entirely.', showIf: is('bwro') },
      { key: 'nfRec', label: 'Nanofiltration recovery', unit: '%', value: 65, min: 20, max: 85, showIf: is('nfro') },
      { key: 'nfFlux', label: 'Nanofiltration flux', unit: 'L/m²·h', value: 20, min: 5, max: 40, showIf: is('nfro') },
    ] },
    { group: 'Thermal desalination (hybrid)', showIf: is('hybrid'), fields: [
      { key: 'medFrac', label: 'Share of intake sent to MED', unit: '%', value: 40, min: 5, max: 90 },
      { key: 'medMakeup', label: 'Make-up feed share of MED seawater', unit: '%', value: 35, min: 10, max: 80, help: 'The rest is condenser cooling water that leaves warm.' },
      { key: 'medRec', label: 'MED recovery of make-up feed', unit: '%', value: 40, min: 10, max: 60 },
      { key: 'medGOR', label: 'Gained output ratio', unit: 'kg/kg', value: 9.5, min: 2, max: 16, help: 'Distillate per kg of heating steam.' },
      { key: 'medTs', label: 'Heating-steam temperature', unit: '°C', value: 70, min: 55, max: 130 },
      { key: 'medTlast', label: 'Last-effect temperature', unit: '°C', value: 40, min: 30, max: 60 },
      { key: 'medSecElec', label: 'MED auxiliary electricity', unit: 'kWh/m³', value: 1.5, min: 0.3, max: 5 },
      { key: 'medLoss', label: 'Heat loss to ambient', unit: '% of steam heat', value: 2, min: 0, max: 15 },
      { key: 'hxUA', label: 'Distillate cooler UA', unit: 'kW/K', value: 400, min: 1, max: 1e5, showIf: (v) => v.useHX },
      { key: 'hxU', label: 'Overall heat-transfer coefficient', unit: 'kW/m²·K', value: 3, min: 0.2, max: 8, showIf: (v) => v.useHX },
    ] },
    { group: 'Brine concentrator and crystalliser', showIf: is('mld'), fields: [
      { key: 'bcTarget', label: 'Concentrator outlet salinity', unit: 'g/kg', value: 220, min: 80, max: 260 },
      { key: 'bcDT', label: 'Evaporator temperature difference', unit: 'K', value: 5, min: 2, max: 12 },
      { key: 'etaComp', label: 'Vapour-compressor isentropic efficiency', unit: '%', value: 75, min: 40, max: 88 },
      { key: 'bcAux', label: 'Recirculation and auxiliary power', unit: 'kWh/m³ feed', value: 1.2, min: 0, max: 8 },
      { key: 'crystDrive', label: 'Crystalliser drive', type: 'select', value: 'steam', options: [{ value: 'steam', label: 'Steam heated' }, { value: 'mvr', label: 'Mechanical vapour recompression' }], showIf: (v) => v.useCryst },
      { key: 'crystMoist', label: 'Salt-cake moisture', unit: '%', value: 8, min: 1, max: 30, showIf: (v) => v.useCryst },
      { key: 'crystPurge', label: 'Crystalliser purge', unit: '% of feed', value: 2, min: 0, max: 25, help: 'Liquor bled to control impurities; it is the remaining liquid discharge.', showIf: (v) => v.useCryst },
    ] },
    { group: 'Product water', fields: [
      { key: 'reminMethod', label: 'Remineralisation method', type: 'select', value: 'lime', options: [{ value: 'lime', label: 'Hydrated lime + CO₂' }, { value: 'calcite', label: 'Calcite contactor + CO₂' }], showIf: (v) => v.useRemin },
      { key: 'hardTarget', label: 'Hardness target', unit: 'mg/L as CaCO₃', value: 65, min: 0, max: 250, showIf: (v) => v.useRemin },
      { key: 'lsiTarget', label: 'Langelier index target', unit: '–', value: 0.1, min: -1, max: 1, showIf: (v) => v.useRemin },
      { key: 'clDose', label: 'Chlorine dose (as NaOCl)', unit: 'mg/L', value: 1.5, min: 0, max: 10, showIf: (v) => v.useDisinfect },
      { key: 'tankHours', label: 'Product storage', unit: 'h of production', value: 6, min: 0.5, max: 72 },
      { key: 'Pdist', label: 'Delivery pressure', unit: 'bar a', value: 5, min: 1.1, max: 40 },
    ] },
    { group: 'Carbon and tariff', fields: [
      { key: 'gridCarbon', label: 'Grid emission factor', unit: 'kgCO₂/kWh', value: 0.45, min: 0, max: 1.3 },
      { key: 'heatCarbon', label: 'Heat emission factor', unit: 'kgCO₂/kWh', value: 0.07, min: 0, max: 0.5 },
      { key: 'offPrice', label: 'Electricity price (off-peak)', unit: '$/kWh', value: 0.07, min: 0, max: 2 },
      { key: 'peakPrice', label: 'Electricity price (peak)', unit: '$/kWh', value: 0.16, min: 0, max: 2 },
      { key: 'peakStart', label: 'Peak period starts', unit: 'h', value: 17, min: 0, max: 24 },
      { key: 'peakEnd', label: 'Peak period ends', unit: 'h', value: 22, min: 0, max: 24 },
    ] },
    { group: 'Solver and models', tab: 'setup', fields: [
      { key: 'tearMethod', label: 'Recycle convergence', type: 'select', value: 'wegstein', options: [{ value: 'wegstein', label: 'Wegstein (bounded)' }, { value: 'damped', label: 'Damped successive substitution' }, { value: 'direct', label: 'Direct successive substitution' }] },
      { key: 'damping', label: 'Damping factor', unit: '–', value: 0.6, min: 0.05, max: 1, showIf: (v) => v.tearMethod === 'damped' },
      { key: 'tearTol', label: 'Tear-stream tolerance (relative)', unit: '–', value: 1e-9, min: 1e-12, max: 1e-3 },
      { key: 'maxIter', label: 'Maximum iterations', unit: '', value: 120, min: 5, max: 1000, step: 1 },
      { key: 'roFF', label: 'Membrane flow factor', unit: '–', value: 0.95, min: 0.4, max: 1.2, help: 'Permeability relative to new membranes; calibrate it against plant pressure.' },
      { key: 'kSP', label: 'Salt-passage multiplier', unit: '×', value: 1, min: 0.3, max: 5, help: 'Scales ion passage of every membrane pass; rises as membranes age.' },
    ] },
    { group: 'Off-design envelope', tab: 'setup', help: 'Boundary conditions of the minimum / normal / maximum cases; equipment sizes stay at the design values.', fields: [
      { key: 'Tmin', label: 'Minimum feed temperature', unit: '°C', value: 17, min: 1, max: 44 },
      { key: 'Tmax', label: 'Maximum feed temperature', unit: '°C', value: 32, min: 2, max: 45 },
      { key: 'salMin', label: 'Minimum salinity', unit: '% of design', value: 96, min: 50, max: 100 },
      { key: 'salMax', label: 'Maximum salinity', unit: '% of design', value: 105, min: 100, max: 150 },
      { key: 'minLoad', label: 'Minimum part load', unit: '% of design feed', value: 50, min: 20, max: 95 },
    ] },
    { group: 'Equipment limits', tab: 'setup', fields: [
      { key: 'pumpMargin', label: 'Pump head and motor margin', unit: '%', value: 10, min: 0, max: 50 },
      { key: 'fluxMargin', label: 'Maximum flux', unit: '% of design', value: 120, min: 100, max: 200 },
      { key: 'hydMargin', label: 'Hydraulic margin of pretreatment and ERD', unit: '%', value: 10, min: 0, max: 50 },
      { key: 'limTDS', label: 'Product TDS limit', unit: 'mg/L', value: 500, min: 5, max: 5000 },
      { key: 'limConc', label: 'Minimum concentrate flow per vessel', unit: 'm³/h', value: 3, min: 0.3, max: 10 },
      { key: 'maxRecChem', label: 'Scaling-limited recovery (0 = not checked)', unit: '%', value: 0, min: 0, max: 99, help: 'Take it from the brine-chemistry suite.' },
      { key: 'medMaxDT', label: 'Maximum condenser temperature rise', unit: 'K', value: 12, min: 3, max: 25, showIf: is('hybrid') },
    ] },
    { group: 'Daily operation (dynamic)', tab: 'setup', help: 'Initial conditions and controller of the product-tank simulation.', fields: [
      { key: 'dynamic', label: 'Simulate daily operation', type: 'bool', value: true },
      { key: 'ctrlMode', label: 'Production control', type: 'select', value: 'vfd', options: [{ value: 'vfd', label: 'Variable flow, PI level control' }, { value: 'onoff', label: 'Trains on/off on level' }], showIf: (v) => v.dynamic },
      { key: 'demandPct', label: 'Average demand', unit: '% of design product', value: 85, min: 20, max: 100, showIf: (v) => v.dynamic },
      { key: 'demand', label: 'Demand profile', type: 'table', columns: [{ key: 'h', label: 'Hour' }, { key: 'f', label: 'Demand', unit: '% of average' }], value: [{ h: 0, f: 60 }, { h: 4, f: 55 }, { h: 6, f: 110 }, { h: 8, f: 135 }, { h: 12, f: 110 }, { h: 16, f: 105 }, { h: 19, f: 140 }, { h: 22, f: 85 }], showIf: (v) => v.dynamic, help: 'Piecewise-linear over the day and repeated; it is normalised to the average demand.' },
      { key: 'levelSP', label: 'Level set-point', unit: '% full', value: 60, min: 10, max: 95, showIf: (v) => v.dynamic },
      { key: 'initLevel', label: 'Initial level', unit: '% full', value: 60, min: 0, max: 100, showIf: (v) => v.dynamic },
      { key: 'Kp', label: 'Controller gain', unit: 'load fraction per level fraction', value: 2.5, min: 0, max: 50, showIf: (v) => v.dynamic && v.ctrlMode === 'vfd' },
      { key: 'Ki', label: 'Integral gain', unit: '1/h', value: 0.4, min: 0, max: 20, showIf: (v) => v.dynamic && v.ctrlMode === 'vfd' },
      { key: 'nTrains', label: 'Number of trains', unit: '', value: 4, min: 1, max: 24, step: 1, showIf: (v) => v.dynamic && v.ctrlMode === 'onoff' },
      { key: 'onLow', label: 'Start a train below', unit: '% full', value: 40, min: 5, max: 90, showIf: (v) => v.dynamic && v.ctrlMode === 'onoff' },
      { key: 'offHigh', label: 'Stop a train above', unit: '% full', value: 80, min: 10, max: 99, showIf: (v) => v.dynamic && v.ctrlMode === 'onoff' },
      { key: 'tou', label: 'Evaluate time-of-use load shifting', type: 'bool', value: true, showIf: (v) => v.dynamic },
    ] },
    { group: 'Time step', tab: 'mesh', help: 'Fixed step of the fourth-order Runge–Kutta integration of the tank and controller.', fields: [
      { key: 'dtMin', label: 'Time step', unit: 'min', value: 10, min: 0.5, max: 60 },
    ] },
  ],

  presets: [
    { name: 'Seawater RO, single pass, pressure exchanger', values: {} },
    { name: 'Seawater RO, two pass for low boron and TDS', values: { template: 'swro2', limTDS: 200, pass2Frac: 70 } },
    { name: 'Brackish RO, 78 % recovery with blend bypass', values: { template: 'bwro', ions: WATERS.brackish.ions, Qf: 400, T: 24, pH: 7.6, tss: 1, membrane: 'bwhr', recovery: 78, flux: 26, elements: 6, erd: 'none', pretreat: 'none', useCoag: false, useSBS: false, useAcid: true, acidDose: 25, asDose: 3.5, blendPct: 6, Pint: 3.5, hardTarget: 80, Tmin: 20, Tmax: 28, salMin: 95, salMax: 110, limTDS: 500 } },
    { name: 'Hybrid RO + MED on Gulf seawater', values: { template: 'hybrid', ions: WATERS.gulf.ions, Qf: 3000, T: 30, pH: 8.2, recovery: 40, flux: 13, Tmin: 20, Tmax: 35, pretreat: 'dmf' } },
    { name: 'Near-zero liquid discharge: RO → concentrator → crystalliser', values: { template: 'mld', ions: WATERS.brackish.ions, Qf: 200, T: 25, pH: 7.6, tss: 2, membrane: 'bwhr', recovery: 80, flux: 24, elements: 6, erd: 'none', pretreat: 'uf', preRec: 94, useCoag: false, useSBS: false, useAcid: true, acidDose: 30, asDose: 4, Pint: 3.5, limTDS: 300, Tmin: 18, Tmax: 30 } },
    { name: 'Nanofiltration softening ahead of RO', values: { template: 'nfro', recovery: 52, flux: 15 } },
  ],

  pull: ({ feed, outputs }) => {
    const o = outputs || {}, P = [];
    if (feed?.ions) P.push({ key: 'ions', value: feed.ions, from: 'Case feed water' });
    if (feed?.Q > 0) P.push({ key: 'Qf', value: feed.Q, from: 'Case feed water' });
    if (Number.isFinite(feed?.T)) P.push({ key: 'T', value: clamp(feed.T, 2, 44), from: 'Case feed water' });
    if (Number.isFinite(feed?.pH)) P.push({ key: 'pH', value: clamp(feed.pH, 4, 10), from: 'Case feed water' });
    if (o.ro?.recovery > 0) P.push({ key: 'recovery', value: clamp(100 * o.ro.recovery, 5, 92), from: 'RO design: recovery' });
    if (o.ro?.fluxLMH > 0) P.push({ key: 'flux', value: clamp(o.ro.fluxLMH, 4, 40), from: 'RO design: average flux' });
    if (o.pump?.pumpEfficiency > 0) P.push({ key: 'etaHP', value: clamp(o.pump.pumpEfficiency <= 1 ? 100 * o.pump.pumpEfficiency : o.pump.pumpEfficiency, 30, 93), from: 'Pumps and energy recovery: HP pump efficiency' });
    if (o.chem?.antiscalantDose > 0) P.push({ key: 'asDose', value: clamp(o.chem.antiscalantDose, 0, 15), from: 'Brine chemistry: antiscalant dose' });
    if (o.chem?.acidDose > 0) P.push({ key: 'acidDose', value: clamp(o.chem.acidDose, 0, 200), from: 'Brine chemistry: acid dose' });
    if (o.chem?.maxRecovery > 0) P.push({ key: 'maxRecChem', value: clamp(100 * o.chem.maxRecovery, 0, 99), from: 'Brine chemistry: scaling-limited recovery' });
    if (o.thermal?.GOR > 0) P.push({ key: 'medGOR', value: clamp(o.thermal.GOR, 2, 16), from: 'Thermal desalination: gained output ratio' });
    if (o.thermal?.secElec > 0) P.push({ key: 'medSecElec', value: clamp(o.thermal.secElec, 0.3, 5), from: 'Thermal desalination: auxiliary electricity' });
    if (o.zld?.secElec > 0 && o.zld?.waterRecovered > 0) P.push({ key: 'bcAux', value: clamp(0.06 * o.zld.secElec, 0, 8), from: 'ZLD: auxiliary share of specific electricity' });
    if (o.fouling?.normPermeability > 0) P.push({ key: 'roFF', value: clamp(o.fouling.normPermeability, 0.4, 1.2), from: 'Fouling monitor: normalised permeability' });
    if (o.opt?.best?.recovery > 0) P.push({ key: 'recovery', value: clamp(o.opt.best.recovery, 5, 92), from: 'Optimiser recommendation' });
    return P;
  },
  site: (site) => {
    const d = site?.data || {}, P = [];
    if (Number.isFinite(d.sst)) P.push({ key: 'T', value: clamp(d.sst, 2, 44), from: 'Sea-surface temperature at site' });
    if (d.gridCarbon >= 0 && d.gridCarbon !== null && d.gridCarbon !== undefined) P.push({ key: 'gridCarbon', value: d.gridCarbon, from: 'Site grid emission factor' });
    if (d.electricityPrice > 0) P.push({ key: 'offPrice', value: d.electricityPrice, from: 'Site electricity price' });
    return P;
  },

  async run(v, ctx) {
    const { ref, res, D, C } = plantDesign(v), W = [], S = res.S, recs = res.sol.recs, dead = res.dead;
    ctx?.progress?.(0.35, 'Design case converged');
    await ctx?.tick?.();
    // ---- off-design envelope
    const caseDefs = [['Normal', {}], ['Minimum temperature', { T: Math.min(v.Tmin, v.T) }], ['Maximum temperature', { T: Math.max(v.Tmax, v.T) }], ['Minimum salinity', { sf: v.salMin / 100 }], ['Maximum salinity', { sf: v.salMax / 100 }], ['Cold and saline', { T: Math.min(v.Tmin, v.T), sf: v.salMax / 100 }],
      [`Part load ${fmt(v.minLoad, 3)} %`, { load: v.minLoad / 100 }], [`Part load ${fmt(0.5 * (v.minLoad + 100), 3)} %`, { load: 0.5 * (v.minLoad / 100 + 1) }], ['Overload 110 %', { load: 1.1 }]];
    const cases = caseDefs.map(([name, cond], i) => { const q = i === 0 ? res : solveCase(v, ref, cond, C), lim = limitsOf(q, D, v); return { name, q, lim, top: lim[0] }; });
    ctx?.progress?.(0.7, 'Off-design cases solved');
    const lims = cases[0].lim, product = res.product, brine = res.brine, pIons = res.prodIons, Qp = product.Q, m3d = Qp * 24;
    const chemKgD = Object.fromEntries(Object.entries(res.chem).map(([k, x]) => [k, 24 * x])), chemTot = sum(Object.values(chemKgD)), sludgeKgD = 24 * res.sludge, solidsTd = (24 * res.solids) / 1000;
    const co2 = res.power * v.gridCarbon + res.heat * v.heatCarbon, lsi = langelier(pIons, product.T, product.pH), salts = res.solidsS ? normativeSalts(res.solidsS.w) : {};
    const liquidOut = brine.Q, minWork = Qp > 0 ? res.least / Qp : 0;
    // ---- dynamic operation
    let dyn = null, dynAlt = [];
    if (v.dynamic && Qp > 0) {
      const loads = [v.minLoad / 100, 0.5 * (v.minLoad / 100 + 1), 1], secs = [cases[6], cases[7], cases[0]].map((c) => c.q.secElec + (c.q.secThermal > 0 ? 0 : 0)), secOf = (u) => interp1(loads, secs, clamp(u, loads[0], 1));
      const tab = (v.demand || []).filter((r) => Number.isFinite(+r.h) && Number.isFinite(+r.f)).map((r) => [clamp(+r.h, 0, 24), Math.max(0, +r.f)]).sort((a, b) => a[0] - b[0]);
      const hs = tab.length ? [tab[tab.length - 1][0] - 24, ...tab.map((r) => r[0]), tab[0][0] + 24] : [0, 24], fsr = tab.length ? [tab[tab.length - 1][1], ...tab.map((r) => r[1]), tab[0][1]] : [100, 100];
      const grid = linspace(0, 24, 97), meanF = sum(grid.slice(0, 96).map((h) => interp1(hs, fsr, h))) / 96 || 100, uff = v.demandPct / 100;
      const demand = (t) => (uff * Qp * interp1(hs, fsr, ((t % 24) + 24) % 24)) / meanF, inPeak = (t) => { const h = ((t % 24) + 24) % 24; return v.peakStart <= v.peakEnd ? h >= v.peakStart && h < v.peakEnd : h >= v.peakStart || h < v.peakEnd; };
      const base = { Qn: Qp, Vmax: Math.max(1, Qp * v.tankHours), L0: v.initLevel / 100, hours: 72, dtMin: v.dtMin, mode: v.ctrlMode, Kp: v.Kp, Ki: v.Ki, uMin: v.minLoad / 100, uff, nTrains: Math.round(v.nTrains), onLow: v.onLow / 100, offHigh: v.offHigh / 100, demand, sec: secOf, price: (t) => (inPeak(t) ? v.peakPrice : v.offPrice), Lsp: () => v.levelSP / 100 };
      dyn = dynamicSim(base); dyn.inPeak = inPeak; dyn.Vmax = base.Vmax;
      if (v.tou) {
        const pre = (t) => { const h = ((t % 24) + 24) % 24, d = (v.peakStart - h + 24) % 24; return d > 0 && d <= 6; };
        dynAlt = [['Constant production at average demand', dynamicSim({ ...base, mode: 'const' })], [v.ctrlMode === 'vfd' ? 'PI level control, fixed set-point' : 'Train on/off control', dyn],
          ['Level control with time-of-use set-point', dynamicSim({ ...base, mode: 'vfd', Lsp: (t) => (inPeak(t) ? Math.max(0.15, v.levelSP / 100 - 0.35) : pre(t) ? Math.min(0.92, v.levelSP / 100 + 0.28) : v.levelSP / 100) })]];
      }
    }
    // ---- warnings
    if (!res.sol.converged) W.push({ level: 'bad', msg: `Recycle streams did not converge in ${res.sol.iterations} iterations (residual ${fmt(res.sol.residual, 2)}). Use damped substitution or raise the iteration limit.` });
    if (res.ro1.Preq > (res.ro1.pmax || 1e9)) W.push({ level: 'bad', msg: `RO feed pressure ${fmt(res.ro1.Preq, 3)} bar exceeds the element rating of ${res.ro1.pmax} bar — lower the recovery or flux.` });
    if (product.tds > v.limTDS) W.push({ level: 'bad', msg: `Product TDS ${fmt(product.tds, 3)} mg/L exceeds the ${v.limTDS} mg/L limit${v.template === 'bwro' ? ' — reduce the blend bypass' : v.template === 'swro' ? ' — add a second pass or lower the flux multiplier' : ''}.` });
    if (v.useRemin && Math.abs(lsi - v.lsiTarget) > 0.3) W.push({ level: 'warn', msg: `Langelier index of the product is ${fmt(lsi, 2)} (target ${v.lsiTarget}); the water is ${lsi < 0 ? 'corrosive' : 'scale forming'}.` });
    if (!v.useRemin && lsi < -1 && v.template !== 'bwro') W.push({ level: 'warn', msg: `Product water is aggressive (Langelier index ${fmt(lsi, 2)}) — switch on remineralisation.` });
    if (res.ro1.concPerVessel > 0 && res.ro1.concPerVessel < v.limConc) W.push({ level: 'warn', msg: `Concentrate flow per vessel ${fmt(res.ro1.concPerVessel, 3)} m³/h is below the ${v.limConc} m³/h minimum.` });
    if (v.maxRecChem > 0 && v.recovery > v.maxRecChem) W.push({ level: 'bad', msg: `Recovery ${v.recovery} % exceeds the scaling limit of ${fmt(v.maxRecChem, 3)} % from the chemistry suite.` });
    const med = res.get('MED');
    if (med && med.rec.dTcw > v.medMaxDT) W.push({ level: 'warn', msg: `MED cooling water warms by ${fmt(med.rec.dTcw, 3)} K (limit ${v.medMaxDT} K) — send more seawater to the condenser or lower the make-up share.` });
    if (med && S.roF.T > 40) W.push({ level: 'warn', msg: `RO feed reaches ${fmt(S.roF.T, 3)} °C after blending the warm reject — above 40 °C membrane life suffers.` });
    if (v.template === 'bwro' && tds(v.ions || {}) > 15000) W.push({ level: 'info', msg: 'The brackish template is being run on a seawater-strength feed; check the element class and the blend bypass.' });
    const worst = cases.filter((c) => c.top && c.top.util > 1.0005);
    for (const c of worst.slice(0, 4)) W.push({ level: 'warn', msg: `${c.name}: ${c.top.name} reaches ${fmt(100 * c.top.util, 3)} % of its limit (${fmt(c.top.value, 3)} against ${fmt(c.top.limit, 3)} ${c.top.unit}).` });
    const negEx = recs.filter((q) => q.exDest < -1e-5 * Math.max(1, res.exIn));
    if (negEx.length) W.push({ level: 'warn', msg: `Exergy destruction is slightly negative in ${negEx.map((q) => q.u.name).join(', ')} — outside the validity of the ideal-mixture exergy model.` });
    if (dyn && (dyn.unmet > 1e-6 || dyn.spill > 1e-6)) W.push({ level: 'warn', msg: `Daily operation: ${dyn.unmet > 1e-6 ? `demand of ${fmt(dyn.unmet, 3)} m³ could not be met` : ''}${dyn.unmet > 1e-6 && dyn.spill > 1e-6 ? ' and ' : ''}${dyn.spill > 1e-6 ? `${fmt(dyn.spill, 3)} m³ overflowed` : ''} over three days — enlarge the tank or retune the controller.` });
    if (!W.some((w) => w.level === 'bad')) W.unshift({ level: 'info', msg: `Flowsheet converged in ${res.sol.iterations} iterations (tear residual ${fmt(res.sol.residual, 2)}); all balances close.` });

    // ---- tables
    const producer = {}, consumer = {};
    for (const q of recs) { q.u.out.forEach((k) => (producer[k] = q.u.id)); q.u.in.forEach((k) => (consumer[k] = [...(consumer[k] || []), q.u.id])); }
    const keys = [...new Set(['raw', ...recs.flatMap((q) => q.u.out)])].filter((k) => S[k] && S[k].m > 1e-9);
    const ex = Object.fromEntries(keys.map((k) => [k, exergy(S[k], dead)]));
    const streamRows = keys.map((k) => { const s = S[k]; return [NAMES[k] || k, `${producer[k] || 'feed'} → ${(consumer[k] || ['out']).join(', ')}`, s.phase, s.Q, s.m / 1000, s.T, s.P, s.phase === 'solid' ? null : s.tds, s.S, s.phase === 'solid' ? null : s.pH, s.rho, hSpec(s) / 1000, ex[k][0], ex[k][1]]; });
    const compRows = keys.filter((k) => S[k].phase !== 'solid').map((k) => { const c = ionsOf(S[k]); return [NAMES[k] || k, ...ION_IDS.map((id) => c[id]), S[k].tss > 0 && S[k].Q > 0 ? (1000 * S[k].tss) / S[k].Q : 0]; });
    const unitRows = recs.map((q) => { const k = q.rec, a = q.inS[0], b = q.outS[0], chems = Object.entries(k.chem || {}).filter(([, x]) => x > 0).map(([n, x]) => `${n} ${fmt(24 * x, 3)} kg/d`).join('; ');
      return [q.u.id, q.u.name, sum(q.inS.map((s) => s.Q)), b.Q, k.recovery !== undefined ? 100 * k.recovery : null, k.dP ?? null, a.T, b.T, k.power || 0, (k.heat || 0) + (k.duty || 0), k.cool || 0, chems || null, q.exDest, res.exDest > 0 ? (100 * q.exDest) / res.exDest : 0, 100 * q.bal.mass, 100 * q.bal.energy]; });
    const groups = { 'Intake and low-pressure pumping': ['IP', 'TP', 'PP'], 'High-pressure pumping': ['HPP', 'NFP'], 'Circulation and inter-stage boosting': ['BST', 'RO1', 'NF'], 'Second pass': ['P2P', 'RO2'], 'Thermal desalination auxiliaries': ['MED'], 'Brine concentrator and crystalliser': ['BC', 'CR'], 'Recovered by turbine': ['TRB'] };
    const pw = Object.entries(groups).map(([n, ids]) => [n, sum(recs.filter((q) => ids.includes(q.u.id)).map((q) => q.rec.power || 0))]).filter((x) => Math.abs(x[1]) > 1e-9);
    const exRank = recs.filter((q) => q.exDest > 1e-6 * Math.max(1, res.exIn)).sort((a, b) => b.exDest - a.exDest).slice(0, 12);
    const exOut = sum(res.fs.outs.map((k) => sum(exergy(S[k], dead)))) - res.least;
    const sizing = [];
    for (const q of recs) {
      const k = q.rec, a = q.inS[0];
      if (q.u.type === 'pump' && k.power > 0) sizing.push([q.u.name, 'Motor rating (shaft power + 10 %)', 1.1 * k.shaft, 'kW', `${fmt(a.Q, 4)} m³/h × ${fmt(k.dP, 3)} bar, η ${fmt(100 * k.eff, 3)} %`]);
      if (q.u.type === 'ro') sizing.push([q.u.name, 'Membrane area', k.area, 'm²', `${Math.ceil(k.elements)} elements in ${Math.ceil(k.vessels)} vessels at ${fmt(k.flux, 3)} L/m²·h`]);
      if (q.u.type === 'px') sizing.push([q.u.name, 'Units of 68 m³/h', Math.ceil(a.Q / 68), '–', `${fmt(a.Q, 4)} m³/h of brine, efficiency ${fmt(100 * k.eff, 4)} %`]);
      if (q.u.type === 'filter' && q.u.id === 'PRE') sizing.push([q.u.name, v.pretreat === 'uf' ? 'Membrane area at 70 L/m²·h' : 'Filter area at 12 m/h', v.pretreat === 'uf' ? (a.Q * 1000) / 70 : a.Q / 12, 'm²', `${fmt(a.Q, 4)} m³/h`]);
      if (q.u.id === 'CF') sizing.push([q.u.name, 'Cartridges (3.5 m³/h each)', Math.ceil(a.Q / 3.5), '–', `${fmt(a.Q, 4)} m³/h`]);
      if (q.u.type === 'hx') sizing.push([q.u.name, 'Heat-transfer area', k.area, 'm²', `duty ${fmt(k.duty, 4)} kW, ε ${fmt(k.eps, 3)}, NTU ${fmt(k.NTU, 3)}, LMTD ${fmt(k.lmtd, 3)} K`]);
      if (q.u.type === 'med') sizing.push([q.u.name, 'Heat-transfer area (estimate)', k.area, 'm²', `${k.effects} effects, steam ${fmt(k.steam, 3)} kg/s, flashing brine releases ${fmt(100 * k.flash, 2)} % vapour`]);
      if (q.u.type === 'bc') sizing.push([q.u.name, 'Evaporator area (estimate)', k.area || 0, 'm²', k.wSpec ? `compression ratio ${fmt(k.ratio, 3)}, ${fmt(k.wSpec, 3)} kWh per tonne of distillate, boiling-point rise ${fmt(k.bpe, 3)} K` : 'not in service']);
      if (q.u.type === 'tank') sizing.push([q.u.name, 'Volume', k.volume, 'm³', `${v.tankHours} h of production`]);
      for (const [n, x] of Object.entries(k.chem || {})) if (x > 0) sizing.push([`${n} storage (${q.u.name})`, '14-day stock', (14 * 24 * x) / 1000, 't', `${fmt(24 * x, 3)} kg/d`]);
    }
    const quality = [['TDS (mg/L)', res.raw.tds, product.tds, brine.tds], ['pH', res.raw.pH, product.pH, brine.pH], ['Temperature (°C)', res.raw.T, product.T, brine.T], ['Hardness (mg/L as CaCO₃)', hardness(ionsOf(res.raw)), hardness(pIons), hardness(ionsOf(brine))], ['Alkalinity (mg/L as CaCO₃)', alkalinity(ionsOf(res.raw)), alkalinity(pIons), alkalinity(ionsOf(brine))],
      ['Langelier index', langelier(ionsOf(res.raw), res.raw.T, res.raw.pH), lsi, langelier(ionsOf(brine), brine.T, brine.pH)], ['Boron (mg/L)', ionsOf(res.raw).B, pIons.B, ionsOf(brine).B], ['Chloride (mg/L)', ionsOf(res.raw).Cl, pIons.Cl, ionsOf(brine).Cl], ['Flow (m³/h)', res.raw.Q, product.Q, brine.Q]];
    const utilRows = [...pw.map(([n, x]) => [`Electricity — ${n}`, x, 'kW']), ['Electricity — total', res.power, 'kW'], ['Heating steam duty', res.heat, 'kW'], ['Heating steam flow', med ? med.rec.steam * 3.6 : res.heat > 0 ? (res.heat / (latentHeat(120) / 1000)) * 3.6 : 0, 't/h'], ['Cooling and heat-loss duty', res.cool, 'kW'],
      ['Cooling seawater (MED reject)', S.medCW ? S.medCW.Q : 0, 'm³/h'], ...Object.entries(chemKgD).map(([n, x]) => [`Chemical — ${n}`, x, 'kg/d']), ['Chemicals — total', chemTot, 'kg/d'], ['Sludge (dry solids)', sludgeKgD, 'kg/d'], ['Salt cake', solidsTd, 't/d'], ...Object.entries(salts).map(([n, x]) => [`Salt — ${n}`, (24 * x) / 1000, 't/d']),
      ['CO₂ from electricity', (24 * res.power * v.gridCarbon) / 1000, 't/d'], ['CO₂ from heat', (24 * res.heat * v.heatCarbon) / 1000, 't/d']];
    const bn = cases.map((c) => c.top).filter(Boolean).sort((a, b) => b.util - a.util)[0], bnCase = cases.find((c) => c.top === bn);
    // ---- flowsheet diagram data
    const pfd = { caption: `${suite.inputs[1].fields[0].options.find((o) => o.value === v.template)?.label} — ${recs.length} units, ${keys.length} streams, ${res.fs.tears.length} tear stream(s).`, rows: ROWS.map((label) => ({ label, units: recs.filter((q) => q.u.row === label).map((q) => { const k = q.rec, b = q.outS[0], lines = [`${fmt(b.Q, 4)} m³/h · ${fmt(b.P, 3)} bar · ${fmt(b.T, 3)} °C`, `${fmt(b.tds, 4)} mg/L TDS`];
      if (k.power) lines.push(`${fmt(k.power, 4)} kW`);
      if (k.recovery !== undefined && q.u.type !== 'filter') lines.push(`recovery ${fmt(100 * k.recovery, 3)} %`);
      if (k.heat) lines.push(`heat ${fmt(k.heat, 4)} kW`);
      if (k.duty) lines.push(`duty ${fmt(k.duty, 4)} kW`);
      if (k.eff && q.u.type === 'px') lines.push(`efficiency ${fmt(100 * k.eff, 4)} %`);
      if (k.solids) lines.push(`${fmt((24 * k.solids) / 1000, 3)} t/d salt cake`);
      return { id: q.u.id, name: q.u.name, lines, from: q.u.in.map((x) => NAMES[x] || x).join(' + '), status: q.exDest > 0.2 * res.exDest ? 'warn' : '' }; }) })).filter((r) => r.units.length) };

    const out = { productFlow: Qp, brineFlow: liquidOut, recovery: res.recovery, power: res.power, heat: res.heat, secElec: res.secElec, secThermal: res.secThermal, chemicals: chemTot, streams: { product: streamOut(product), brine: streamOut(brine) },
      chemicalsByName: chemKgD, sludge: sludgeKgD, solids: solidsTd, salts: Object.fromEntries(Object.entries(salts).map(([n, x]) => [n, (24 * x) / 1000])), co2PerM3: Qp > 0 ? co2 / Qp : 0, feedPressureBar: res.ro1.Preq, membraneArea: res.ro1.area + (res.ro2?.area || 0) + (res.nf?.area || 0), exergyEfficiency: res.etaII, exergyDestroyed: res.exDest, minWork,
      tearResidual: res.sol.residual, iterations: res.sol.iterations, lsi, productTDS: product.tds, bottleneck: bn ? bn.name : 'none', tankVolume: Qp * v.tankHours, dynLevelMin: dyn ? dyn.Lmin : v.initLevel, dynLevelMax: dyn ? dyn.Lmax : v.initLevel, dynEnergyCost: dyn ? dyn.cost : 0, dynIAE: dyn ? dyn.iae : 0 };
    const last = (a) => (dyn ? a.slice(dyn.i0) : []), th = dyn ? last(dyn.t).map((x) => x - dyn.t[dyn.i0]) : [];
    const cn = cases.map((c) => c.name);
    return {
      summary: `${fmt(Qp, 4)} m³/h (${fmt(m3d, 4)} m³/d) of product at ${fmt(product.tds, 3)} mg/L from ${fmt(res.raw.Q, 4)} m³/h intake — overall recovery ${fmt(100 * res.recovery, 3)} %, ${fmt(res.secElec, 3)} kWh/m³ electricity${res.heat > 0 ? ` and ${fmt(res.secThermal, 3)} kWh/m³ heat` : ''}, ${fmt(res.power, 4)} kW. ${fmt(liquidOut, 4)} m³/h leaves as ${fmt(brine.tds / 1000, 3)} g/L discharge${solidsTd > 0 ? ` and ${fmt(solidsTd, 3)} t/d as salt cake` : ''}. Second-law efficiency ${fmt(100 * res.etaII, 3)} %${bn ? `; first limit reached: ${bn.name.toLowerCase()} (${bnCase.name.toLowerCase()})` : ''}.`,
      warnings: W,
      kpis: [
        { label: 'Product flow', value: Qp, unit: 'm³/h', help: `${fmt(m3d, 5)} m³/d` }, { label: 'Overall recovery', value: 100 * res.recovery, unit: '%', help: 'Product ÷ intake' },
        { label: 'Product TDS', value: product.tds, unit: 'mg/L', status: product.tds > v.limTDS ? 'bad' : 'ok' }, { label: 'Liquid discharge', value: liquidOut, unit: 'm³/h' },
        { label: 'Discharge TDS', value: brine.tds / 1000, unit: 'g/L' }, { label: 'Specific electricity', value: res.secElec, unit: 'kWh/m³' },
        { label: 'Specific heat', value: res.secThermal, unit: 'kWh/m³' }, { label: 'Electrical power', value: res.power, unit: 'kW' },
        { label: 'RO feed pressure', value: res.ro1.Preq, unit: 'bar a', status: res.ro1.Preq > (res.ro1.pmax || 1e9) ? 'bad' : 'ok' }, { label: 'Membrane area', value: out.membraneArea, unit: 'm²' },
        { label: 'Second-law efficiency', value: 100 * res.etaII, unit: '%', help: 'Least work of separation ÷ exergy supplied (electricity + exergy of heat)' }, { label: 'Exergy destroyed', value: res.exDest, unit: 'kW' },
        { label: 'Least work of separation', value: minWork, unit: 'kWh/m³' }, { label: 'Chemicals', value: chemTot, unit: 'kg/d' },
        { label: 'Sludge and salt', value: sludgeKgD / 1000 + solidsTd, unit: 't/d', help: 'Dry pretreatment solids plus crystalliser cake' }, { label: 'Carbon footprint', value: Qp > 0 ? co2 / Qp : 0, unit: 'kgCO₂/m³', help: `${fmt((24 * co2) / 1000, 3)} t/d` },
        { label: 'Product Langelier index', value: lsi, unit: '–', status: v.useRemin && Math.abs(lsi - v.lsiTarget) > 0.3 ? 'warn' : 'ok' }, { label: 'Recycle convergence', value: `${res.sol.iterations} it · ${fmt(res.sol.residual, 2)}`, status: res.sol.converged ? 'ok' : 'bad' },
      ],
      recommendations: [
        exRank[0] ? `${exRank[0].u.name} destroys the most exergy (${fmt(exRank[0].exDest, 3)} kW, ${fmt((100 * exRank[0].exDest) / res.exDest, 2)} %) — the first place to look for energy savings.` : null,
        v.erd === 'none' && res.ro1.Preq > 20 ? `The concentrate valve throttles ${fmt(res.get('BV')?.rec.throttled || 0, 3)} kW — fit an energy-recovery device.` : null,
        bn && bn.util > 1 ? `${bn.name} is the bottleneck in the “${bnCase.name}” case at ${fmt(100 * bn.util, 3)} % of its limit: add margin there before anything else.` : bn ? `Largest utilisation across the envelope: ${bn.name.toLowerCase()} at ${fmt(100 * bn.util, 3)} % (${bnCase.name.toLowerCase()}).` : null,
        dynAlt.length && dynAlt[2][1].cost < dynAlt[1][1].cost * 0.995 && dynAlt[2][1].unmet < 1e-6 ? `Shifting production away from the peak tariff with the tank saves ${fmt(dynAlt[1][1].cost - dynAlt[2][1].cost, 3)} $/d (${fmt((100 * (dynAlt[1][1].cost - dynAlt[2][1].cost)) / dynAlt[1][1].cost, 2)} %).` : null,
        'Send the product, brine, power and chemical totals to suite 13 (Economics); refine the membrane array in suite 1 and the brine route in suites 5 and 9.',
      ].filter(Boolean),
      plots: [
        { type: 'bar', title: 'Electrical power by consumer', ylabel: 'kW', categories: pw.map((x) => x[0]), series: [{ name: 'kW', values: pw.map((x) => x[1]) }] },
        { type: 'bar', title: 'Where the supplied exergy goes', ylabel: 'kW', categories: ['Least work of separation', ...exRank.map((q) => q.u.name), 'Other units', 'Leaves with products and discharge'], series: [{ name: 'Exergy (kW)', values: [res.least, ...exRank.map((q) => q.exDest), Math.max(0, res.exDest - sum(exRank.map((q) => q.exDest))), exOut] }],
          note: `Exergy supplied: ${fmt(res.exIn, 4)} kW (electricity ${fmt(res.exIn - sum(recs.map((q) => q.exQ)), 4)} kW, exergy of heat ${fmt(sum(recs.map((q) => q.exQ)), 4)} kW). Bars after the first are exergy destroyed in each unit.` },
        { type: 'line', title: 'Pressure, salinity and temperature along the main line', xlabel: 'Unit number along the flowsheet', ylabel: 'bar · g/L · °C', series: (() => { const x = recs.map((_, i) => i + 1); return [{ name: 'Outlet pressure (bar a)', x, y: recs.map((q) => q.outS[0].P), mode: 'both' }, { name: 'Outlet TDS (g/L)', x, y: recs.map((q) => Math.min(300, q.outS[0].tds / 1000)), mode: 'both' }, { name: 'Outlet temperature (°C)', x, y: recs.map((q) => q.outS[0].T), mode: 'both' }]; })(), note: recs.map((q, i) => `${i + 1} ${q.u.id}`).join(' · ') },
        { type: 'line', title: 'Recycle convergence', xlabel: 'Iteration', ylabel: 'Largest relative change of the tear variables', logy: true, series: [{ name: v.tearMethod, x: res.sol.history.map((_, i) => i + 1), y: res.sol.history.map((x) => Math.max(x, 1e-16)), mode: 'both' }], hlines: [{ y: v.tearTol, label: 'tolerance' }] },
        { type: 'bar', title: 'Off-design cases: specific energy and feed pressure', ylabel: 'kWh/m³ · bar ÷ 10', categories: cn, series: [{ name: 'Specific electricity (kWh/m³)', values: cases.map((c) => c.q.secElec) }, { name: 'RO feed pressure ÷ 10 (bar)', values: cases.map((c) => c.q.ro1.Preq / 10) }, { name: 'Product TDS ÷ 100 (mg/L)', values: cases.map((c) => c.q.product.tds / 100) }] },
        { type: 'bar', title: 'Off-design cases: utilisation of the limiting equipment', ylabel: '% of limit', categories: cn, series: [{ name: 'Highest utilisation', values: cases.map((c) => (c.top ? 100 * c.top.util : 0)) }], note: cases.map((c) => `${c.name}: ${c.top ? c.top.name : '–'}`).join(' · ') },
        { type: 'bar', title: 'Chemical consumption', ylabel: 'kg/d', categories: Object.keys(chemKgD).length ? Object.keys(chemKgD) : ['none'], series: [{ name: 'kg/d', values: Object.keys(chemKgD).length ? Object.values(chemKgD) : [0] }] },
        ...(dyn ? [
          { type: 'line', title: 'Daily operation: tank level and production', xlabel: 'Hour of day', ylabel: '%', ymin: 0, series: [{ name: 'Tank level (% full)', x: th, y: last(dyn.L) }, { name: 'Production (% of design)', x: th, y: last(dyn.u), mode: v.ctrlMode === 'onoff' ? 'step' : 'line' }, { name: 'Demand (% of design product)', x: th, y: last(dyn.Qd).map((q) => (100 * q) / Qp) }, { name: 'Set-point', x: th, y: last(dyn.sp), dash: true }], vlines: [{ x: v.peakStart, label: 'peak tariff' }, { x: v.peakEnd, label: '' }] },
          { type: 'line', title: 'Daily operation: electrical power', xlabel: 'Hour of day', ylabel: 'kW', zeroY: true, series: [{ name: dynAlt.length ? dynAlt[1][0] : 'Power', x: th, y: last(dyn.P), mode: v.ctrlMode === 'onoff' ? 'step' : 'line' }, ...(dynAlt.length ? [{ name: dynAlt[2][0], x: dynAlt[2][1].t.slice(dynAlt[2][1].i0).map((x) => x - dynAlt[2][1].t[dynAlt[2][1].i0]), y: dynAlt[2][1].P.slice(dynAlt[2][1].i0) }] : [])], vlines: [{ x: v.peakStart, label: 'peak tariff' }, { x: v.peakEnd, label: '' }] },
        ] : []),
      ],
      tables: [
        { title: 'Stream table — conditions', columns: ['Stream', 'From → to', 'Phase', 'Flow (m³/h)', 'Mass flow (t/h)', 'T (°C)', 'P (bar a)', 'TDS (mg/L)', 'Salinity (g/kg)', 'pH', 'Density (kg/m³)', 'Enthalpy (kJ/kg)', 'Physical exergy (kW)', 'Chemical exergy (kW)'], rows: streamRows, note: `Dead state: raw feed at ${fmt(dead.T, 3)} °C and 1.013 bar. Enthalpy is relative to liquid at 0 °C and includes flow work.` },
        { title: 'Stream table — composition (mg/L)', columns: ['Stream', ...ION_IDS.map((k) => IONS[k].label), 'Suspended solids'], rows: compRows },
        { title: 'Unit operations', columns: ['Tag', 'Unit', 'Inlet (m³/h)', 'Main outlet (m³/h)', 'Recovery (%)', 'ΔP (bar)', 'T in (°C)', 'T out (°C)', 'Power (kW)', 'Heat duty (kW)', 'Cooling (kW)', 'Chemicals', 'Exergy destroyed (kW)', 'Share of destruction (%)', 'Mass closure (%)', 'Energy closure (%)'], rows: unitRows, note: 'Heat duty is steam heat for thermal units and exchanged heat for heat exchangers. Negative power is generation. Closure columns are the balance errors of each unit.' },
        { title: 'Plant performance and water quality', columns: ['Quantity', 'Feed', 'Product', 'Discharge'], rows: quality },
        { title: 'Utilities, chemicals and residuals', columns: ['Item', 'Value', 'Unit'], rows: utilRows },
        { title: 'Off-design cases and bottlenecks', columns: ['Case', 'Feed T (°C)', 'Salinity (%)', 'Load (%)', 'Product (m³/h)', 'Product TDS (mg/L)', 'RO feed pressure (bar)', 'Flux (L/m²·h)', 'Power (kW)', 'SEC (kWh/m³)', 'Limiting equipment', 'Utilisation (%)', 'Within limits'],
          rows: cases.map((c) => [c.name, c.q.cond.T, 100 * c.q.cond.sf, 100 * c.q.cond.load, c.q.product.Q, c.q.product.tds, c.q.ro1.Preq, c.q.ro1.flux, c.q.power, c.q.secElec, c.top ? c.top.name : '–', c.top ? 100 * c.top.util : null, c.top && c.top.util > 1.0005 ? 'no' : 'yes']), note: `Equipment sizes are frozen at the design case; recovery is held. Design-case utilisations: ${lims.slice(0, 5).map((l) => `${l.name} ${fmt(100 * l.util, 3)} %`).join('; ')}.` },
        { title: 'Equipment sizing hints', columns: ['Equipment', 'Quantity', 'Value', 'Unit', 'Basis'], rows: sizing },
        ...(dyn ? [{ title: 'Daily operation and controller performance', columns: ['Strategy', 'Energy (MWh/d)', 'Energy cost ($/d)', 'Minimum level (%)', 'Maximum level (%)', 'Integral absolute error (%·h)', 'Starts', 'Time at a limit (%)', 'Unmet demand (m³)', 'Overflow (m³)'],
          rows: (dynAlt.length ? dynAlt : [['Selected control', dyn]]).map(([n, d]) => [n, d.energy / 1000, d.cost, d.Lmin, d.Lmax, d.iae, d.starts, 100 * d.sat, d.unmet, d.spill]), note: `Tank ${fmt(dyn.Vmax, 4)} m³; three days simulated with a ${fmt(dyn.dt * 60, 3)} min step, last day reported. Unmet demand and overflow are totals over the three days.` }] : []),
      ],
      balances: [
        { name: 'Plant total mass (t/h)', in: res.plant.mass.in / 1000, out: res.plant.mass.out / 1000 },
        { name: 'Plant dissolved salts (kg/h)', in: sum(res.plant.ion.map((x) => x.in)), out: sum(res.plant.ion.map((x) => x.out)) },
        { name: 'Plant chloride (kg/h)', in: res.plant.ion[IX.Cl].in, out: res.plant.ion[IX.Cl].out }, { name: 'Plant sodium (kg/h)', in: res.plant.ion[IX.Na].in, out: res.plant.ion[IX.Na].out },
        { name: 'Plant calcium (kg/h)', in: res.plant.ion[IX.Ca].in, out: res.plant.ion[IX.Ca].out }, { name: 'Plant sulphate (kg/h)', in: res.plant.ion[IX.SO4].in, out: res.plant.ion[IX.SO4].out },
        { name: 'Plant energy (kW)', in: res.plant.energy.in, out: res.plant.energy.out },
        (() => { const q = recs.reduce((a, b) => (b.bal.mass > a.bal.mass ? b : a)); return { name: `Largest unit mass error: ${q.u.name} (kg/h)`, in: q.bal.mIn, out: q.bal.mOut }; })(),
        (() => { const q = recs.reduce((a, b) => (b.bal.energy > a.bal.energy ? b : a)); return { name: `Largest unit energy error: ${q.u.name} (kW)`, in: q.bal.hIn, out: q.bal.hOut }; })(),
      ],
      outputs: out, pfd,
    };
  },

  mesh: { name: 'Time step of the daily-operation simulation', keys: ['dtMin'], refine: 'divide', note: 'The Runge–Kutta step of the tank and controller model is refined; the steady-state flowsheet has no grid.',
    metrics: [{ label: 'Minimum tank level', unit: '%', get: (r) => r.outputs.dynLevelMin }, { label: 'Daily energy cost', unit: '$/d', get: (r) => r.outputs.dynEnergyCost }, { label: 'Integral absolute level error', unit: '%·h', get: (r) => r.outputs.dynIAE }] },

  calibration: {
    note: 'Fit the membrane flow factor, the salt-passage multiplier and the high-pressure pump efficiency to plant historian data. Each row is one steady operating day: feed temperature, salinity relative to design and plant load set the case (equipment sizes stay at the design values); RO feed pressure, total electrical power and product TDS are the measurements. Use days that span the seasonal temperature range and at least one part-load day.',
    params: [{ key: 'roFF', label: 'Membrane flow factor', lo: 0.5, hi: 1.2 }, { key: 'kSP', label: 'Salt-passage multiplier', lo: 0.4, hi: 4 }, { key: 'etaHP', label: 'High-pressure pump efficiency (%)', lo: 55, hi: 92 }],
    columns: [{ key: 'Tcase', label: 'Feed temperature', unit: '°C' }, { key: 'salCase', label: 'Salinity', unit: '% of design' }, { key: 'loadCase', label: 'Load', unit: '% of design' }, { key: 'Pfeed', label: 'RO feed pressure', unit: 'bar' }, { key: 'power', label: 'Electrical power', unit: 'kW' }, { key: 'prodTDS', label: 'Product TDS', unit: 'mg/L' }],
    targets: [{ key: 'Pfeed', label: 'RO feed pressure', unit: 'bar' }, { key: 'power', label: 'Electrical power', unit: 'kW' }, { key: 'prodTDS', label: 'Product TDS', unit: 'mg/L' }],
    model(v) {
      const ref = references(v), base = { ...v, roFF: REF_FF, kSP: 1 }, key = JSON.stringify(Object.entries(v).filter(([k, x]) => typeof x !== 'object' && !['roFF', 'kSP', 'etaHP', 'Tcase', 'salCase', 'loadCase'].includes(k)));
      let C = suite._design?.key === key && suite._design.ref === ref.ro1 ? suite._design.C : null; // sizes depend on the design, not on the fitted parameters
      if (!C) { const q = solveCase(base, ref); C = { area1: q.ro1.area, area2: q.ro2?.area, areaNF: q.nf?.area }; suite._design = { key, ref: ref.ro1, C }; }
      const r = solveCase(v, ref, { T: v.Tcase ?? v.T, sf: (v.salCase ?? 100) / 100, load: (v.loadCase ?? 100) / 100 }, C);
      return { Pfeed: r.ro1.Preq, power: r.power, prodTDS: r.product.tds };
    },
    get sample() { return (this._s ||= synth(17, [[25, 100, 100], [20, 100, 100], [17, 101, 100], [29, 99, 100], [32, 98, 100], [24, 103, 100], [26, 100, 80], [22, 100, 65], [30, 102, 90]])); },
    get validationSample() { return (this._v ||= synth(41, [[23, 100, 100], [18, 102, 100], [27, 99, 100], [31, 100, 95], [25, 104, 100], [21, 100, 72], [28, 97, 85]])); },
  },

  verify() {
    const d = defaultsOf(suite), C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    const ref = references(d), res = solveCase(d, ref), recs = res.sol.recs, mx = (f) => Math.max(...recs.map(f));
    add('Total mass balance closes around every unit', 0, mx((q) => q.bal.mass), 1e-10, `Largest relative error over ${recs.length} units`);
    add('Every ion balance closes around every unit', 0, mx((q) => q.bal.ion), 1e-9, 'Largest relative error over all units and all 18 constituents (with reaction terms)');
    add('Energy balance closes around every unit', 0, mx((q) => q.bal.energy), 1e-8, 'Σ ṁh in + shaft work + heat = Σ ṁh out + cooling');
    add('Plant total mass balance', 0, (res.plant.mass.in - res.plant.mass.out) / res.plant.mass.in, 1e-7, 'Feed + chemicals = product + discharge + solids');
    add('Plant ion balances (worst constituent)', 0, Math.max(...res.plant.ion.map((x) => Math.abs(x.in - x.out) / Math.max(x.in, 1e-6 * res.raw.salt))), 1e-6, 'Each of the 18 constituents across the whole plant');
    add('Plant energy balance', 0, (res.plant.energy.in - res.plant.energy.out) / res.plant.energy.in, 1e-7, 'First law around the plant boundary');
    add('Recycle (tear) residual below tolerance', 0, res.sol.residual, d.tearTol, `Converged in ${res.sol.iterations} Wegstein iterations`);
    const dir = solveCase({ ...d, tearMethod: 'direct', maxIter: 400 }, ref);
    add('Wegstein and direct substitution give the same solution', 0, Math.abs(dir.product.Q - res.product.Q) / res.product.Q + Math.abs(dir.ro1.Preq - res.ro1.Preq) / res.ro1.Preq, 1e-7, `Direct substitution needs ${dir.sol.iterations} iterations, Wegstein ${res.sol.iterations}`);
    // mixer and splitter hand calculations
    const a = fromVol(100, WATERS.seawater.ions, 20, 2, 8), b = fromVol(300, WATERS.brackish.ions, 30, 2, 7.5), m = UNITS.mix([a, b]).outs[0];
    add('Mixer: blended chloride, hand calculation', (100 * 19810 + 300 * 1290) / m.Q, ionsOf(m).Cl, 1e-6, 'c = Σ Q·c / Q_out (mg/L)');
    add('Mixer: outlet temperature, hand calculation', (a.m * cpOf(a) * 20 + b.m * cpOf(b) * 30) / (a.m * cpOf(a) + b.m * cpOf(b)), m.T, 0.03, 'Heat-capacity-weighted mean (°C); the solver balances the enthalpy function instead of constant heat capacities');
    add('Mixer: mass is conserved', a.m + b.m, m.m, 1e-9, 'kg/h');
    const sp = UNITS.split([a], { frac: 0.3 }).outs;
    add('Splitter: flow split and unchanged composition', 30, sp[0].Q + Math.abs(sp[0].tds - a.tds) + Math.abs(sp[1].tds - a.tds), 1e-9, '30 % of 100 m³/h with identical TDS in both branches');
    // pump
    const pu = UNITS.pump([a], { Pout: 63, eta: 0.85, etaM: 0.95 });
    add('Pump hydraulic power, hand calculation', (a.rho * 9.80665 * (100 / 3600) * ((61e5) / (a.rho * 9.80665))) / 1000, pu.rec.hyd, 1e-9, 'ρ·g·Q·H with H = Δp/ρg = Q·Δp (kW)');
    add('Pump shaft work appears as enthalpy rise', pu.rec.shaft, Hk(pu.outs[0]) - Hk(a), 1e-6, 'First law on the pump (kW)');
    // heat exchanger
    const hot = fromVol(120, { Na: 2, Cl: 3 }, 40, 2, 6.5), hx = UNITS.hx([hot, a], { UA: 300, U: 3, dp: 0 });
    add('Heat exchanger: ε–NTU duty equals UA × LMTD', hx.rec.duty, 300 * hx.rec.lmtd, 0.01 * hx.rec.duty, 'Same exchanger rated by both methods (kW, within 1 % — property variation with temperature)');
    add('Heat exchanger: energy leaving the hot side enters the cold side', Hk(hot) - Hk(hx.outs[0]), Hk(hx.outs[1]) - Hk(a), 1e-6, 'kW');
    add('Effectiveness of a balanced counter-flow exchanger', 2 / 3, effectiveness(2, 1), 1e-12, 'ε = NTU/(1 + NTU) at C_r = 1');
    add('Effectiveness with C_r = 0', 1 - Math.exp(-1.5), effectiveness(1.5, 0), 1e-12, 'ε = 1 − exp(−NTU), condenser limit');
    // limiting cases
    const zf = fromVol(500, WATERS.seawater.ions, 25, 60, 8), z = UNITS.ro([zf], { ref: ref.ro1, recovery: 0, key: 'ro1', Pp: 1.3, ff: 0.95, kSP: 1, eta: 0.85, etaM: 0.95 }, { P: { ro1: 60 }, Pn: {} });
    add('Zero recovery: no permeate, concentrate equals feed', zf.m, z.outs[1].m + z.outs[0].m * 1e6 + Math.abs(z.outs[1].salt - zf.salt) * 1e6, 1e-6, 'Limiting case of the membrane block (mass and salt flow, kg/h)');
    const by = solveCase({ ...d, template: 'bwro', blendPct: 100, useCoag: false, useAntiscalant: false, useSBS: false, useRemin: false, useDisinfect: false, useAcid: false }, ref);
    add('100 % bypass: product salinity equals the feed', by.raw.S, by.product.S, 1e-9, 'All pretreated feed bypasses the membranes (g/kg); no chemicals dosed');
    add('100 % bypass: no high-pressure pumping', 0, by.get('HPP').rec.power, 1e-9, 'kW');
    add('Reduced membrane model reproduces the reference pressure', ref.ro1.Pf, UNITS.ro([fromVol(ref.ro1.Qf, d.ions, d.T, ref.ro1.Pf, d.pH)], { ref: ref.ro1, recovery: ref.ro1.r, key: 'ro1', Pp: d.Pperm, ff: REF_FF, kSP: 1, eta: 0.85, etaM: 0.95 }, { P: {}, Pn: {} }).rec.Preq, 0.02, 'Same feed as the element-by-element solution (bar)');
    // second law
    add('Exergy destruction is non-negative in every unit', 0, Math.min(0, ...recs.map((q) => q.exDest / Math.max(1, res.exIn))), 1e-6, 'Smallest value relative to the exergy supplied');
    add('Exergy supplied = least work + destruction + exergy leaving', res.exIn, res.least + res.exDest + (sum(res.fs.outs.map((k) => sum(exergy(res.S[k], res.dead)))) - res.least) - sum(exergy(res.raw, res.dead)), 1e-4 * res.exIn, 'Exergy balance around the plant (kW)');
    add('Electricity use exceeds the least work of separation', 1, res.power > res.least && res.etaII < 1 ? 1 : 0, 0, `Second-law efficiency ${fmt(100 * res.etaII, 3)} %`);
    add('Chemical exergy of the dead state is zero', 0, exergy(res.raw, res.dead)[1], 1e-9, 'Raw feed at ambient conditions');
    // reduced thermal blocks
    add('Flash fraction, hand calculation', (cp(65, 60) * 10) / latentHeat(60, 0), flashFraction(70, 60, 60), 2e-4, 'x ≈ c_p·ΔT / λ for brine flashing from 70 to 60 °C');
    add('Compressor work, hand calculation', (1860 * 373.15 * ((psat(110) / psat(100)) ** (0.33 / 1.33) - 1)) / 0.75, compressorWork(100, 110, 0.75).w, 1e-6, 'w = c_p·T₁·(r^((k−1)/k) − 1)/η (J/kg)');
    // dynamic volume balance
    const dy = dynamicSim({ Qn: 400, Vmax: 2400, L0: 0.5, hours: 48, dtMin: 10, mode: 'vfd', Kp: 2.5, Ki: 0.4, uMin: 0.5, uff: 0.85, nTrains: 4, onLow: 0.4, offHigh: 0.8, demand: (t) => 340 * (1 + 0.4 * Math.sin((2 * Math.PI * t) / 24)), sec: () => 3, price: () => 0.1, Lsp: () => 0.6 });
    add('Tank volume balance over the dynamic run', dy.Vend - 1200, dy.net + dy.unmet - dy.spill, 1e-6, 'ΔV = ∫(production − demand) dt, corrected for overflow and unmet demand (m³)');
    return C;
  },

  views: [{ id: 'flowsheet', label: 'Flowsheet', tip: 'Process flow diagram of the selected template with the key numbers of every unit',
    render(el, api) {
      const h = api.h, res = api.result();
      if (!res || !res.pfd) { el.append(h('p', { class: 'note' }, 'Run the simulation to draw the process flow diagram of the selected template with the flow, pressure, salinity and power of every unit.')); return; }
      el.append(h('p', { class: 'note' }, res.pfd.caption + ' Each box shows the main outlet stream of the unit; highlighted boxes destroy more than a fifth of the total exergy.'));
      el.append(h('div', { class: 'pfd' }, res.pfd.rows.map((row) => h('div', { class: 'pfd-row' }, h('div', { class: 'pfd-stream' }, h('strong', null, row.label)),
        row.units.map((u, i) => [i ? h('div', { class: 'pfd-arrow', 'aria-hidden': 'true' }, '→') : null, h('div', { class: 'pfd-unit' + (u.status ? ' ' + u.status : ''), title: 'Inlet: ' + u.from }, h('strong', null, `${u.id} · ${u.name}`), u.lines.map((l) => h('div', { class: 'pfd-stream' }, l)))])))));
      if (api.kpiGrid) el.append(api.kpiGrid(res.kpis.slice(0, 8)));
    } }],
};

/** Synthetic plant-historian data: the model with different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([Tcase, salCase, loadCase]) => {
    const m = suite.calibration.model({ ...d, roFF: 0.86, kSP: 1.3, etaHP: 83, Tcase, salCase, loadCase });
    return { Tcase, salCase, loadCase, Pfeed: +(m.Pfeed * (1 + g.normal(0, 0.004))).toFixed(2), power: +(m.power * (1 + g.normal(0, 0.006))).toFixed(1), prodTDS: +(m.prodTDS * (1 + g.normal(0, 0.02))).toFixed(1) };
  });
}

export default suite;
