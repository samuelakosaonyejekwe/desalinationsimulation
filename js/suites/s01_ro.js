// Suite 1 — Reverse osmosis and membrane design.
// Element-by-element, segment-resolved model of spiral-wound arrays: solution–diffusion or
// Spiegler–Kedem transport coupled with film-theory concentration polarisation, spacer-channel
// hydraulics, multi-ion permeate quality with electroneutrality, staging, inter-stage boosting,
// concentrate recycle, an optional second pass and energy recovery.
import { brent, clamp, linspace, sum, rng, fmt } from '../core/num.js';
import { density, viscosity, diffusivityNaCl, tcf, salinityFromTDS } from '../core/props.js';
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

/** One axial segment of one pressure vessel. Mutates nothing; returns new state and local values. */
function segment(Q, c, P, M, o) {
  const tdsB = tds(c), S = salinityFromTDS(tdsB, o.T), rho = density(o.T, S), mu = viscosity(o.T, S), D = diffusivityNaCl(o.T, S);
  const hs = M.spacerMil * 25.4e-6, W = M.area / (2 * EL_LEN), Ac = W * hs * POROSITY;
  const dh = (4 * POROSITY) / (2 / hs + ((1 - POROSITY) * 8) / hs), u = Q / 3600 / Ac;
  const Re = (rho * u * dh) / mu, Sc = mu / (rho * D);
  const k = Math.max(1e-7, (o.kcp * 0.065 * Re ** 0.875 * Sc ** 0.25 * D) / dh); // m/s
  const fr = 6.23 * Math.max(Re, 1) ** -0.3;
  const dP = (o.kdp * fr * (o.dL / dh) * rho * u * u) / 2 / 1e5;
  const A = M.A * o.tcf * o.ff, piB = osmoticPressureIons(c, o.T) / 1e5;
  const Pm = P - dP / 2;
  let Jw = Math.max(0, A * (Pm - o.Pp - piB)), cp = {}, piP = 0, CP = 1;
  const noDrive = Pm - o.Pp <= 0.02 * piB; // applied pressure far below osmotic pressure: no forward permeation
  for (let it = 0; it < 14; it++) {
    CP = Math.exp(Math.min(3, Jw / 3.6e6 / k));
    let cat = 0, an = 0;
    for (const id of ION_IDS) {
      const cm = c[id] * CP, Bi = o.Bi[id];
      let ci;
      if (o.model === 'sk') {
        // Spiegler–Kedem: R = σ(1−F)/(1−σF), F = exp(−Jw(1−σ)/Ps)
        const sig = DIVALENT.includes(id) ? Math.min(0.9995, M.sigma + (1 - M.sigma) * 0.6) : M.sigma;
        const Fk = Math.exp((-Jw * (1 - sig)) / Math.max(Bi, 1e-9)), Rj = Jw > 0 ? (sig * (1 - Fk)) / (1 - sig * Fk) : 0;
        ci = cm * (1 - Rj);
      } else ci = Jw + Bi > 0 ? (Bi * cm) / (Jw + Bi) : cm;
      cp[id] = Math.min(ci, cm);
      const eq = (cp[id] / IONS[id].mw) * IONS[id].z;
      if (eq > 0) cat += eq; else an -= eq;
    }
    if (cat > 0 && an > 0) { // electroneutral permeate: co-transport forces equal equivalents
      const fc = Math.sqrt(an / cat), fa = Math.sqrt(cat / an);
      for (const id of ION_IDS) cp[id] *= IONS[id].z > 0 ? fc : IONS[id].z < 0 ? fa : 1;
    }
    piP = osmoticPressureIons(cp, o.T) / 1e5;
    const sig = o.model === 'sk' ? M.sigma : 1;
    const Jn = noDrive ? 0 : Math.max(0, A * (Pm - o.Pp - sig * (piB * CP - piP)));
    if (Math.abs(Jn - Jw) < 1e-7 * (1 + Jw)) { Jw = Jn; break; }
    Jw = 0.5 * Jw + 0.5 * Jn;
  }
  const Qp = Math.min((Jw * o.dA) / 1000, Q * 0.95), Qn = Q - Qp, cn = {};
  for (const id of ION_IDS) cn[id] = Math.max(0, (Q * c[id] - Qp * cp[id]) / Qn);
  return { Q: Qn, c: cn, P: P - dP, Qp, cp, Jw, CP, dP, piB, piP, ndp: Pm - o.Pp - (piB * CP - piP), Re, k, u };
}

/** March through one stage (nV parallel vessels of nE elements). */
function stage(feed, nV, nE, M, o) {
  let Q = feed.Q / nV, c = feed.ions, P = feed.P;
  const els = [], perm = [];
  for (let e = 0; e < nE; e++) {
    const Pin = P, Qin = Q, cin = tds(c);
    let qp = 0, jw = 0, cpMax = 1, ndp = 0;
    const pc = Object.fromEntries(ION_IDS.map((k) => [k, 0]));
    for (let s = 0; s < o.nSeg; s++) {
      const r = segment(Q, c, P, M, { ...o, dA: M.area / o.nSeg, dL: EL_LEN / o.nSeg });
      for (const id of ION_IDS) pc[id] += r.cp[id] * r.Qp;
      qp += r.Qp; jw += r.Jw / o.nSeg; ndp += r.ndp / o.nSeg; cpMax = Math.max(cpMax, r.CP);
      Q = r.Q; c = r.c; P = r.P;
    }
    for (const id of ION_IDS) pc[id] = qp > 0 ? pc[id] / qp : 0;
    perm.push({ Q: qp * nV, ions: pc });
    els.push({ el: e + 1, Qin, Qout: Q, Pin, dP: Pin - P, flux: jw, rec: qp / Qin, CP: cpMax, ndp, tdsFeed: cin, tdsPerm: tds(pc), Qp: qp });
  }
  const Qp = sum(perm.map((p) => p.Q));
  return { els, conc: { Q: Q * nV, ions: c, P }, perm: { Q: Qp, ions: Qp > 0 ? mixIons(perm.map((p) => ({ Q: p.Q, ions: p.ions }))) : cloneIons({}) } };
}

/** One pass at a given feed pressure (bar). */
function passAt(Pf, feed, cfg) {
  const M = cfg.M, pH = feed.pH;
  const Bi = Object.fromEntries(ION_IDS.map((id) => [id, M.B * cfg.tcf * cfg.sp * relB(id, M, pH, cfg.T)]));
  const o = { T: cfg.T, tcf: cfg.tcf, ff: cfg.ff, kcp: cfg.kcp, kdp: cfg.kdp, Pp: cfg.Pp, nSeg: cfg.nSeg, model: cfg.model, Bi };
  let rec = cfg.recycle > 0 ? { Q: 0, ions: feed.ions } : null, out;
  for (let it = 0; it < (cfg.recycle > 0 ? 25 : 1); it++) {
    const blended = rec && rec.Q > 0 ? { Q: feed.Q + rec.Q, ions: mixIons([feed, rec]) } : { Q: feed.Q, ions: feed.ions };
    let cur = { ...blended, P: Pf };
    const stages = [];
    for (let s = 0; s < cfg.vessels.length; s++) {
      if (s > 0) cur = { ...cur, P: cur.P - cfg.interLoss + (cfg.boost[s] || 0) };
      const r = stage(cur, cfg.vessels[s], cfg.elements, M, o);
      stages.push({ ...r, feed: cur, nV: cfg.vessels[s] });
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
  return { M, T, tcf: tcf(T, T >= 25 ? 2640 : 3020), ff, sp, kcp: v.kcp, kdp: v.kdp, Pp: v.Pp, nSeg: Math.max(1, Math.round(v.nSeg)), model: v.model, elements: Math.round(v.elements), interLoss: v.interLoss };
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
  description: 'Solves every membrane element of every stage in axial segments. Water and solute transport follow the solution–diffusion or Spiegler–Kedem model; film theory with a spacer Sherwood correlation gives concentration polarisation; a spacer friction correlation gives pressure loss. Each ion is tracked individually with an electroneutral permeate, so boron, nitrate and divalent rejection, concentration factors and design-limit violations are reported element by element.',
  guide: [
    'Enter the feed analysis, flow and temperature (or pull them from the Case page).',
    'Pick an element class and either let the tool size the array or enter your own stages and vessels.',
    'Choose whether you fix the recovery (pressure is solved) or fix the pressure (recovery is solved).',
    'Run. Check the design-limit warnings, the element profile and the permeate quality; the concentrate is automatically offered to the chemistry, ZLD and sea-discharge suites.',
  ],
  implemented: ['solution-diffusion', 'spiegler-kedem', 'kedem-katchalsky', 'vant hoff', 'extended vant hoff', 'water-flux', 'solute-flux', 'salt-rejection', 'observed-rejection', 'intrinsic-rejection', 'recovery equation', 'concentration-factor', 'concentration-polarization', 'sherwood', 'reynolds', 'schmidt', 'mass-transfer-coefficient', 'pressure-drop', 'darcy-weisbach', 'friction-factor', 'mass balance', 'component material', 'electroneutrality', 'temperature-correction', 'darcy-type', 'activity-coefficient',
    'film theory', 'fouling resistance', 'resistance-in-series', 'osmotic-pressure coupling', 'spacer hydrodynamic', 'energy-recovery coupling', 'mechanistic-empirical', 'scale-formation coupling', 'antiscalant assessment',
    'initial feed concentration', 'initial pressure', 'temperature', 'membrane resistance', 'fouling resistance', 'prescribed inlet', 'outlet-pressure', 'membrane-interface flux', 'solute-partition', 'interface condition', 'permeate-side',
    'feed-water characterisation', 'membrane and element selection', 'array configuration', 'membrane transport', 'salt rejection', 'permeate-quality', 'recovery calculation', 'concentration-polarisation', 'pressure-drop calculation', 'osmotic-pressure assessment', 'hydraulic balancing', 'staging and recirculation', 'energy-consumption', 'membrane ageing', 'sensitivity analysis', 'design optimisation'],
  equationsNote: 'Valid for spiral-wound RO/NF elements at 1–45 °C and up to about 120 g/kg bulk salinity. Stage-by-stage scaling indices come from the electrolyte model of suite 2; dosing design is done there and optimisation in suite 11.',

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
      { key: 'model', label: 'Membrane transport model', type: 'select', value: 'sd', options: [{ value: 'sd', label: 'Solution–diffusion + film theory' }, { value: 'sk', label: 'Spiegler–Kedem (reflection coefficient) + film theory' }], help: 'Spiegler–Kedem adds solvent–solute coupling and is preferred for nanofiltration and loose RO.' },
      { key: 'kcp', label: 'Mass-transfer multiplier', unit: '×', value: 1, min: 0.2, max: 5, help: 'Scales the spacer Sherwood correlation Sh = 0.065 Re^0.875 Sc^0.25. Calibrate it or take it from the CFD suite.' },
      { key: 'kdp', label: 'Pressure-drop multiplier', unit: '×', value: 1, min: 0.2, max: 10, help: 'Scales the spacer friction factor f = 6.23 Re^−0.3. Rises as elements foul.' },
      { key: 'interLoss', label: 'Inter-stage piping loss', unit: 'bar', value: 0.3, min: 0, max: 5 },
    ] },
    { group: 'Initial membrane state', tab: 'setup', help: 'Initial conditions: condition of the membranes at the time simulated.', fields: [
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
    const Ts = linspace(Math.max(5, v.T - 12), Math.min(42, v.T + 12), 7), sweepT = Ts.map((T) => { try { const q = simulateRO({ ...fixed, T }); return [q.p1.Pf, tds(q.p1.perm.ions), q.sec]; } catch { return [NaN, NaN, NaN]; } });
    const R0 = v.recovery, Rs = linspace(Math.max(10, R0 - 15), Math.min(92, R0 + 12), 7), sweepR = Rs.map((rc) => { try { const q = simulateRO({ ...fixed, recovery: rc }); return [q.p1.Pf, tds(q.p1.perm.ions), q.sec]; } catch { return [NaN, NaN, NaN]; } });

    const out = {
      streams: { feed: streamOut(r.feed, v.T, r.p1.Pf, v.pH), permeate: streamOut(r.product, v.T, v.Pp, r.permPH), concentrate: streamOut(r.conc, v.T, r.conc.P, Math.min(9, v.pH + 0.3 * Math.log10(cf))) },
      feedPressureBar: r.p1.Pf, concentratePressureBar: r.conc.P, recovery: r.overallRec, permeateFlow: r.product.Q, fluxLMH: avgFlux, membraneArea: r.area, nElements: r.nEl, nVessels: sum(r.vessels),
      sec: r.sec, pumpPower: r.power, dpBar: r.p1.Pf - r.conc.P, cpFactor: maxCP, vessels: r.vessels, concentrationFactor: cf, feedFlow: r.feed.Q, erdType: v.erd,
    };
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
        { type: 'line', title: 'Effect of recovery (array fixed)', xlabel: 'Recovery (%)', ylabel: 'Pressure (bar) · permeate TDS/10 (mg/L)', series: [{ name: 'Feed pressure', x: Rs, y: sweepR.map((q) => q[0]), mode: 'both' }, { name: 'Permeate TDS ÷ 10', x: Rs, y: sweepR.map((q) => q[1] / 10), mode: 'both' }, { name: 'SEC × 10 (kWh/m³)', x: Rs, y: sweepR.map((q) => q[2] * 10), mode: 'both' }] },
      ],
      tables: [
        { title: 'Stage summary', columns: ['Stage', 'Vessels', 'Feed (m³/h)', 'Permeate (m³/h)', 'Concentrate (m³/h)', 'Recovery (%)', 'Feed P (bar)', 'Conc. P (bar)', 'Avg flux (L/m²·h)', 'Permeate TDS (mg/L)', 'Feed/vessel (m³/h)', 'Conc./vessel (m³/h)'],
          rows: [...r.p1.stages.map((s, i) => [`Pass 1 · ${i + 1}`, s.nV, s.feed.Q, s.perm.Q, s.conc.Q, (100 * s.perm.Q) / s.feed.Q, s.feed.P, s.conc.P, (s.perm.Q * 1000) / (s.nV * r.cfg.elements * r.cfg.M.area), tds(s.perm.ions), s.feed.Q / s.nV, s.conc.Q / s.nV]),
            ...(r.p2 ? r.p2.stages.map((s, i) => [`Pass 2 · ${i + 1}`, s.nV, s.feed.Q, s.perm.Q, s.conc.Q, (100 * s.perm.Q) / s.feed.Q, s.feed.P, s.conc.P, (s.perm.Q * 1000) / (s.nV * r.pass2.cfg.elements * r.pass2.cfg.M.area), tds(s.perm.ions), s.feed.Q / s.nV, s.conc.Q / s.nV]) : [])] },
        { title: 'Element-by-element profile (one vessel per stage)', columns: ['Stage', 'Element', 'Feed (m³/h)', 'Inlet P (bar)', 'ΔP (bar)', 'Flux (L/m²·h)', 'Recovery (%)', 'β', 'NDP (bar)', 'Feed TDS (mg/L)', 'Permeate TDS (mg/L)', 'Permeate (m³/h)'],
          rows: els.map((e) => [e.stage, e.el, e.Qin, e.Pin, e.dP, e.flux, 100 * e.rec, e.CP, e.ndp, e.tdsFeed, e.tdsPerm, e.Qp]) },
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
