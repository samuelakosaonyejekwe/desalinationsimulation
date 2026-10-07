// Suite 3 — Whole-plant process simulation.
// Sequential-modular steady-state flowsheet solver with tear-stream convergence (Wegstein or damped
// substitution) over a library of unit operations: intake, pumps, dosing, filtration, RO/NF passes,
// isobaric energy recovery, turbines, valves, mixers, splitters, heat exchangers, a reduced MED block,
// a vapour-compression brine concentrator, a crystalliser, remineralisation and storage.
// Streams carry mass flow and the mass flow of every ion, so total and component balances close exactly;
// enthalpy and physical + salinity exergy give energy closure and exergy destruction per unit.
import { clamp, linspace, sum, rng, fmt, rk4, interp1, brent, lstsq, solveLinear, metrics, trapz } from '../core/num.js';
import { density, cp, enthalpyLiquid, enthalpyVapour, latentHeat, psat, tsat, bpe, tcf, osmoticCoefficient, salinityFromTDS, R, KELVIN, F as FARADAY } from '../core/props.js';
import { IONS, ION_IDS, WATERS, tds, scaleIons, osmoticPressureIons, hardness, alkalinity, conductivity } from '../core/water.js';
import roSuite, { simulateRO, MEMBRANES } from './s01_ro.js';

const P0 = 1.01325, MW_W = 18.015, SMAX = 260, REF_FF = 0.95, NI = ION_IDS.length;
const IX = Object.fromEntries(ION_IDS.map((k, i) => [k, i])), MW = ION_IDS.map((k) => IONS[k].mw), ZI = ION_IDS.map((k) => IONS[k].z);
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
export function compressorWork(Tin, Tout, eta, model = 'isentropic') {
  const cpv = 1860, k = 1.33, ratio = psat(Tout) / psat(Tin);
  if (model === 'polytropic') return { w: cpv * (Tin + KELVIN) * (ratio ** ((k - 1) / (k * eta)) - 1), ratio }; // eta is then the polytropic efficiency: (n − 1)/n = (k − 1)/(k·η)
  return { w: (cpv * (Tin + KELVIN) * (ratio ** ((k - 1) / k) - 1)) / eta, ratio };
}
/**
 * Gas compressor / vacuum pump between P1 and P2 (bar) with inlet at T (°C): shaft work in J per mol of gas.
 * 'isentropic': w = Z·c_p·T₁·(r^((k−1)/k) − 1)/η_s; 'polytropic': w = Z·c_p·T₁·(r^((n−1)/n) − 1) with (n−1)/n = (k−1)/(k·η_p).
 * Equal-ratio stages (ratio ≤ maxRatio each) with intercooling back to the inlet temperature.
 */
export function gasCompressor(P1, P2, T, cpMol, eta, model = 'isentropic', Z = 1, maxRatio = 4) {
  const r = Math.max(1, P2 / P1), N = Math.max(1, Math.ceil(Math.log(r) / Math.log(maxRatio) - 1e-9)), rs = r ** (1 / N), k = cpMol / (cpMol - R), e = (k - 1) / k, Tk = T + KELVIN, poly = model === 'polytropic';
  const rise = poly ? rs ** (e / eta) - 1 : (rs ** e - 1) / eta;
  return { w: N * Z * cpMol * Tk * rise, stages: N, ratio: r, stageRatio: rs, k, n: poly ? 1 / (1 - e / eta) : k, Tout: Tk * (1 + rise) - KELVIN };
}
/** Counter-flow effectiveness. */
export const effectiveness = (NTU, Cr) => (Math.abs(1 - Cr) < 1e-9 ? NTU / (1 + NTU) : (1 - Math.exp(-NTU * (1 - Cr))) / (1 - Cr * Math.exp(-NTU * (1 - Cr))));

// ---- phase equilibrium of water with dissolved gases over a salt solution (γ–φ formulation) -------------
// Critical constants and acentric factor for the Peng–Robinson equation of state; Henry constant in pure water at 25 °C (bar per unit mole
// fraction), its van 't Hoff coefficient (K), Setschenow salting-out constant (kg per mol of ionic strength) and ideal-gas heat capacity (J/mol·K).
const GAS = {
  H2O: { name: 'Water', mw: 18.015, Tc: 647.096, Pc: 220.64, om: 0.3443, cp: 33.6 },
  CO2: { name: 'Carbon dioxide', mw: 44.01, Tc: 304.13, Pc: 73.77, om: 0.2239, H: 1651, C: 2400, ks: 0.11, cp: 37.1 },
  N2: { name: 'Nitrogen', mw: 28.013, Tc: 126.19, Pc: 33.96, om: 0.0372, H: 8.67e4, C: 1300, ks: 0.138, cp: 29.1 },
  O2: { name: 'Oxygen', mw: 31.999, Tc: 154.58, Pc: 50.43, om: 0.0222, H: 4.41e4, C: 1500, ks: 0.125, cp: 29.4 },
};
const VOL = ['H2O', 'CO2', 'N2', 'O2'], AIR = { N2: 0.7808, O2: 0.2095 };
/** Henry constant of a dissolved gas in pure water, bar per unit mole fraction. */
export const henry = (id, T) => GAS[id].H * Math.exp(-GAS[id].C * (1 / (T + KELVIN) - 1 / 298.15));
/** Water activity of a salt solution from the osmotic coefficient: ln a_w = −φ·M_w·Σm (Σm = mol of ions per kg of water). */
export const waterActivity = (T, S, mTot = (31.843 * S) / (1000 - S)) => Math.exp(-osmoticCoefficient(T, Math.min(S, 200)) * mTot * 0.018015);
/** Davies activity coefficient of an ion of charge z at ionic strength I (mol/kg). */
export const davies = (z, I, T = 25) => 10 ** (-(0.4913 + 6.08e-4 * T + 5.95e-6 * T * T) * z * z * (Math.sqrt(I) / (1 + Math.sqrt(I)) - 0.3 * I));

/** Peng–Robinson equation of state for a vapour mixture y (water, CO₂, N₂, O₂) at T (°C) and P (bar): compressibility and fugacity coefficients. */
export function pengRobinson(y, T, P) {
  const Tk = T + KELVIN, RT = R * Tk, p = P * 1e5, a = [], b = [];
  for (const id of VOL) { const g = GAS[id], kap = 0.37464 + 1.54226 * g.om - 0.26992 * g.om * g.om, al = (1 + kap * (1 - Math.sqrt(Tk / g.Tc))) ** 2; a.push((0.45724 * R * R * g.Tc * g.Tc * al) / (g.Pc * 1e5)); b.push((0.0778 * R * g.Tc) / (g.Pc * 1e5)); }
  const sa = sum(y.map((yi, i) => yi * Math.sqrt(a[i]))), am = sa * sa, bm = sum(y.map((yi, i) => yi * b[i])), A = (am * p) / (RT * RT), B = (bm * p) / RT;
  const c2 = B - 1, c1 = A - 3 * B * B - 2 * B, c0 = -(A * B - B * B - B ** 3), pp = c1 - (c2 * c2) / 3, qq = (2 * c2 ** 3) / 27 - (c2 * c1) / 3 + c0, disc = (qq * qq) / 4 + pp ** 3 / 27;
  let Z;
  if (disc > 0) Z = Math.cbrt(-qq / 2 + Math.sqrt(disc)) + Math.cbrt(-qq / 2 - Math.sqrt(disc)) - c2 / 3;
  else { const r = 2 * Math.sqrt(-pp / 3); Z = r * Math.cos(Math.acos(clamp((3 * qq) / (pp * r), -1, 1)) / 3) - c2 / 3; } // largest of three real roots = vapour
  for (let i = 0; i < 3; i++) Z -= (((Z + c2) * Z + c1) * Z + c0) / ((3 * Z + 2 * c2) * Z + c1);
  const L = Math.log((Z + (1 + Math.SQRT2) * B) / (Z + (1 - Math.SQRT2) * B));
  return { Z, B2: bm - am / RT, phi: y.map((_, i) => Math.exp((b[i] / bm) * (Z - 1) - Math.log(Z - B) - (A / (2 * Math.SQRT2 * B)) * ((2 * Math.sqrt(a[i]) * sa) / am - b[i] / bm) * L)) };
}

/** Rachford–Rice equation Σ zᵢ(Kᵢ − 1)/(1 + β(Kᵢ − 1)) = 0 for the vapour fraction β; a non-volatile component has K = 0. */
export function rachfordRice(z, K) {
  const f = (b) => { let s = 0; for (let i = 0; i < z.length; i++) s += (z[i] * (K[i] - 1)) / (1 + b * (K[i] - 1)); return s; };
  if (!(f(0) > 0)) return 0; // at or below the bubble point
  if (K.every((k) => k > 0) && sum(z.map((zi, i) => zi / K[i])) <= 1) return 1; // at or above the dew point
  return brent(f, 0, 1 - 1e-13, 1e-18);
}

/**
 * Isothermal flash of water + dissolved CO₂, N₂ and O₂ over a salt solution at T (°C) and P (bar) by equality of fugacities
 * xᵢ·γᵢ·fᵢ° = yᵢ·φᵢ·P. n = { H2O, CO2, N2, O2, salt } in any molar unit (salt = dissolved ions, non-volatile).
 * Water: fᵢ° = P_sat·φ_sat·Poynting with x_w·γ_w = a_w from the osmotic coefficient (modified Raoult's law); gases: fᵢ° = Henry constant
 * with a Setschenow activity coefficient. model 'ideal' = Raoult + Henry with ideal gas; 'gamma' = activity coefficients, ideal gas;
 * 'pr' = activity coefficients and Peng–Robinson fugacity coefficients. o = { S (g/kg), I (mol/kg) } of the feed liquid.
 */
export function flashVLE(n, T, P, o = {}) {
  const model = o.model || 'pr', ideal = model === 'ideal', eos = model === 'pr', RT = R * (T + KELVIN), S0 = o.S || 0, I0 = o.I || 0;
  const nt = sum(VOL.map((k) => n[k] || 0)) + (n.salt || 0), z = [...VOL.map((k) => (n[k] || 0) / nt), (n.salt || 0) / nt], ps = psat(T) / 1e5, vL = 0.018015 / density(T, 0);
  const phiSat = eos ? pengRobinson([1, 0, 0, 0], T, ps).phi[0] : 1, m0 = z[4] / (z[0] * 0.018015), r0 = S0 / (1000 - S0);
  const liquid = (x) => { // activity of water and activity coefficients of the liquid of composition x
    const m = x[4] / (x[0] * 0.018015), cf = m0 > 0 ? m / m0 : 1, aw = ideal ? x[0] : waterActivity(T, Math.min(SMAX, (1000 * r0 * cf) / (1 + r0 * cf)), m);
    return { aw, gamma: [aw / x[0], ...VOL.slice(1).map((k) => (ideal ? 1 : 10 ** (GAS[k].ks * I0 * cf)))] };
  };
  const f0 = (Pb) => [ps * phiSat * (ideal ? 1 : Math.exp((vL * (Pb - ps) * 1e5) / RT)), ...VOL.slice(1).map((k) => henry(k, T) * (ideal ? 1 : Math.exp((33e-6 * (Pb - ps) * 1e5) / RT)))];
  const vap = (y, Pb) => { const t = sum(y); return eos && t > 0 ? pengRobinson(y.map((q) => q / t), T, Pb) : { Z: 1, phi: [1, 1, 1, 1], B2: 0 }; };
  let liq = liquid(z), Pb = Math.max(P, ps), y = [1, 0, 0, 0], v = vap(y, Pb); // bubble pressure of the feed
  for (let i = 0; i < 60; i++) { const f = f0(Pb); v = vap(y, Pb); const pp = f.map((fi, j) => (z[j] * liq.gamma[j] * fi) / v.phi[j]), Pn = sum(pp), d = Math.abs(Pn - Pb); y = pp.map((q) => q / Pn); Pb = Pn; if (d < 1e-14 * Pn) break; }
  const Pbub = Pb, two = Pbub > P * (1 + 1e-12), Pe = two ? P : Pbub, fP = f0(Pe);
  let x = z, beta = 0, K = fP.map((fi, j) => (liq.gamma[j] * fi) / (v.phi[j] * Pe));
  if (two) for (let it = 0; it < 100; it++) {
    const Ka = [...K, 0]; beta = rachfordRice(z, Ka); x = z.map((zi, j) => zi / (1 + beta * (Ka[j] - 1))); y = K.map((k, j) => k * x[j]);
    liq = liquid(x); v = vap(y, Pe);
    const Kn = fP.map((fi, j) => (liq.gamma[j] * fi) / (v.phi[j] * Pe)), d = Math.max(...Kn.map((k, j) => Math.abs(k / K[j] - 1)));
    K = Kn;
    if (d < 1e-14) break;
  }
  const part = (f) => Object.fromEntries(VOL.map((k, j) => [k, f(j)]));
  return { model, T, P, Peq: Pe, beta, x, y, K, gamma: liq.gamma, phi: v.phi, Z: v.Z, aw: liq.aw, phiSat, Pbub, fL: fP.map((fi, j) => x[j] * liq.gamma[j] * fi), fV: y.map((yi, j) => yi * v.phi[j] * Pe),
    L: part((j) => (1 - beta) * x[j] * nt), V: part((j) => beta * y[j] * nt) };
}
/** Mole fractions of N₂ and O₂ dissolved in a solution in equilibrium with moist air at 1 atm (Henry's law with salting-out); sat = fraction of saturation. */
export function airSaturation(T, S, I = 0, mTot, sat = 1) {
  const pd = P0 - (waterActivity(T, Math.min(S, 200), mTot) * psat(T)) / 1e5;
  return Object.fromEntries(Object.entries(AIR).map(([k, yk]) => [k, (sat * yk * pd) / (henry(k, T) * 10 ** (GAS[k].ks * I))]));
}
/** Flash of a liquid stream, with its dissolved CO₂ and the air it holds at sat × air saturation, at pressure P (bar). Flows in mol/h. */
export function streamFlash(s, P, o = {}) {
  const kgw = Math.max(1e-9, s.m - s.salt), Nw = (1000 * kgw) / MW_W, Ni = sum(s.w.map((x, i) => (1000 * x) / MW[i])), I = (0.5 * sum(s.w.map((x, i) => ((1000 * x) / MW[i]) * ZI[i] ** 2))) / kgw;
  const air = airSaturation(o.Tair ?? s.T, s.S, I, Ni / kgw, o.sat ?? 1), n = { H2O: Nw, salt: Ni, CO2: (1000 * s.co2) / 44.01, N2: air.N2 * (Nw + Ni), O2: air.O2 * (Nw + Ni) }; // the air content is set where the water met the atmosphere (Tair)
  return { ...flashVLE(n, s.T, P, { model: o.model, S: Math.min(s.S, SMAX), I }), n, I };
}
// Ion-size parameter (Å) and linear term of the Truesdell–Jones extended Debye–Hückel equation (WATEQ set).
const TJ = { Ca: [5, 0.165], Mg: [5.5, 0.2], Na: [4, 0.075], K: [3.5, 0.015], Cl: [3.5, 0.015], SO4: [5, -0.04], HCO3: [5.4, 0], CO3: [5.4, 0] };
/** Single-ion activity coefficient: Truesdell–Jones where parameters exist, Davies otherwise. I in mol/kg. */
export function ionGamma(id, I, T = 25) {
  const p = TJ[id], z = IONS[id].z, q = Math.sqrt(I);
  return p ? 10 ** ((-(0.4913 + 6.08e-4 * T + 5.95e-6 * T * T) * z * z * q) / (1 + (0.3248 + 1.4e-4 * T) * p[0] * q) + p[1] * I) : davies(z, Math.min(I, 0.7), T);
}
/**
 * Electrolyte state of a stream: ionic strength (mol/kg), water activity (osmotic coefficient), ion activity coefficients, free fractions of
 * sulphate and bicarbonate after the main ion pairs (MgSO₄°, NaSO₄⁻, CaSO₄°, NaHCO₃°, MgHCO₃⁺, CaHCO₃⁺) and the saturation indices of calcite and gypsum.
 */
export function electrolyte(s) {
  const kgw = s.m - s.salt;
  if (!(kgw > 0) || s.phase === 'solid') return null;
  const mol = (k) => (1000 * s.w[IX[k]]) / IONS[k].mw / kgw, mTot = sum(ION_IDS.map(mol)), I = 0.5 * sum(ION_IDS.map((k) => mol(k) * IONS[k].z ** 2)), Tk = s.T + KELVIN, g = (k) => ionGamma(k, Math.min(I, 3), s.T), gp = g('Na');
  const aw = waterActivity(s.T, Math.min(s.S, 200), mTot), ca = mol('Ca'), hc = mol('HCO3'), so = mol('SO4');
  const fSO4 = 1 / (1 + 10 ** 2.37 * g('Mg') * g('SO4') * mol('Mg') + (10 ** 0.7 * g('Na') * g('SO4') * mol('Na')) / gp + 10 ** 2.3 * g('Ca') * g('SO4') * ca);
  const fHCO3 = 1 / (1 + 10 ** -0.25 * g('Na') * g('HCO3') * mol('Na') + (10 ** 1.07 * g('Mg') * g('HCO3') * mol('Mg')) / gp + (10 ** 1.11 * g('Ca') * g('HCO3') * ca) / gp);
  const K2 = 10 ** -(107.8871 + 0.03252849 * Tk - 5151.79 / Tk - 38.92561 * Math.log10(Tk) + 563713.9 / (Tk * Tk)), Kc = 10 ** (-171.9065 - 0.077993 * Tk + 2839.319 / Tk + 71.595 * Math.log10(Tk));
  return { I, aw, mTot, gNaCl: Math.sqrt(g('Na') * g('Cl')), gCa: g('Ca'), gSO4: g('SO4'), fSO4, fHCO3,
    siCalcite: ca > 0 && hc > 0 ? Math.log10((g('Ca') * ca * K2 * g('HCO3') * hc * fHCO3) / 10 ** -s.pH / Kc) : null, siGypsum: ca > 0 && so > 0 ? Math.log10((g('Ca') * ca * g('SO4') * so * fSO4 * aw * aw) / 10 ** -4.58) : null };
}
/** Entropy flow of a stream relative to the dead state, kW/K: incompressible-liquid thermal part plus the mixing part on the same basis as the salinity exergy. */
export function entropy(s, dead) {
  if (!(s.m > 0)) return 0;
  const T0 = dead.T + KELVIN, c = s.phase === 'solid' ? 1200 : cp(0.5 * (s.T + dead.T), Math.min(s.S, SMAX));
  let x = s.m * c * Math.log((s.T + KELVIN) / T0);
  if (s.phase !== 'solid') { const Ni = sum(s.w.map((q, i) => (1000 * q) / MW[i])), Nw = (1000 * (s.m - s.salt)) / MW_W, Nt = Nw + Ni; x -= dead.phi * R * (Nw * Math.log(Nw / Nt / dead.xw) + (Ni > 0 && dead.xi > 0 ? Ni * Math.log(Ni / Nt / dead.xi) : 0)); }
  return x / 3.6e6;
}
/** Enthalpy flow on the incompressible-liquid basis of the exergy function (relative to the dead-state temperature and pressure), kW. */
const hInc = (s, dead) => (s.m > 0 ? (s.m * ((s.phase === 'solid' ? 1200 : cp(0.5 * (s.T + dead.T), Math.min(s.S, SMAX))) * (s.T - dead.T) + ((s.P - P0) * 1e5) / s.rho)) / 3.6e6 : 0);

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
    if (calcite && kg > 0) { // first-order dissolution towards the CO₂-limited equilibrium: dC/dt = k·(C_eq − C) → contact time = −ln(1 − X)/k
      const X = clamp(up / Math.max(up, s.co2 + co2Dose), 0, 0.995), kd = 0.2 * 1.03 ** (s.T - 20);
      r.rec.approach = X; r.rec.kDiss = kd; r.rec.ebct = -Math.log(1 - X) / kd; r.rec.bed = (s.Q * r.rec.ebct) / 60;
    }
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
    const mc = (1000 * s.salt) / p.target, D = s.m - mc, wd = s.w.map((x) => (x * D * 1e-5) / Math.max(s.salt, 1e-12)), rise = bpe(100, Math.min(p.target, SMAX)), cw = compressorWork(100, 100 + rise + p.dT, p.etaComp, p.compModel);
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
      const cw = compressorWork(100, 100 + bpe(100, SMAX) + p.dT + 3, p.etaComp, p.compModel), shaft = ((D / 3600) * cw.w) / 1000 + 2.5 * s.Q;
      return { outs: settle(specs, Hk(s) + shaft, s.T + 15), rec: { shaft, power: shaft / p.etaM, solids: ms, recovery: D / s.m, wSpec: cw.w / 3600, dP: P0 - s.P } };
    }
    const heat = ((D / 3600) * latentHeat(75) * 1.08) / 1000, aux = 2.5 * s.Q, outs = specs.map((q) => mkS(q.m, q.w, 45, q.P, q.o));
    return { outs, rec: { heat, Ts: 120, cool: Hk(s) + heat + aux - sum(outs.map(Hk)), shaft: aux, power: aux / p.etaM, solids: ms, recovery: D / s.m, dP: P0 - s.P } };
  },
  // Vacuum deaerator / decarbonator: single adiabatic equilibrium flash at the vessel pressure, vent compressed to atmosphere, liquid pumped back to line pressure.
  deaer: ([s], p) => {
    if (!(s.m > 0)) return { outs: [s], rec: { dP: 0, power: 0, shaft: 0 } };
    const Pv = Math.min(p.P, s.P), o2 = { ...keep(s) };
    const at = (T) => { // equilibrium at T and the energy left over: zero at the adiabatic flash temperature
      const f = streamFlash(T === s.T ? s : mkS(s.m, s.w, T, s.P, o2), Pv, { model: p.model, sat: p.sat, Tair: s.T }), V = f.V, mW = (V.H2O * 18.015) / 1000, mC = Math.min(s.co2, (V.CO2 * 44.01) / 1000);
      const hV = (mW * enthalpyVapour(T) + mC * (enthalpyLiquid(T, 0) + (R * GAS.CO2.C) / 0.04401) + R * (GAS.N2.C * V.N2 + GAS.O2.C * V.O2)) / 3.6e6; // vapour enthalpy and heats of desorption (R·C from the Henry constants), kW
      return { f, V, mW, mC, hV, res: Hk(s) - hV - Hk(mkS(s.m - mW - mC, s.w, T, s.P, { ...o2, co2: s.co2 - mC })) };
    };
    let a = at(s.T);
    if (a.res < 0) { const lo = Math.min(s.T, tsat(Pv * 1e5)) - 4; if (at(lo).res > 0) a = at(brent((T) => at(T).res, lo, s.T, 1e-7)); }
    const { f, V, mW, mC, hV } = a, Tf = f.T, nV = sum(VOL.map((k) => V[k])) / 3600;
    const cpMix = nV > 0 ? sum(VOL.map((k, j) => f.y[j] * GAS[k].cp)) : GAS.H2O.cp, comp = gasCompressor(Pv, P0 + 0.05, Tf, cpMix, p.etaV, p.compModel, f.Z), shaftV = (nV * comp.w) / 1000, hyd = (s.Q * (s.P - Pv)) / 36, shaftP = hyd / p.eta;
    const out = settle([{ m: s.m - mW - mC, w: s.w, P: s.P, o: { ...keep(s), co2: s.co2 - mC } }], Hk(s) - hV + shaftP, Tf)[0], rem = (k) => (f.n[k] > 0 ? V[k] / f.n[k] : 0);
    return { outs: [out], rec: { dP: 0, shaft: shaftP + shaftV, power: (shaftP + shaftV) / p.etaM, hyd, ventMass: mW + mC, ventH: hV + shaftV, ventCO2: mC, ventWater: mW, ventMol: nV * 3600, vle: f, comp, shaftV, Pv, Tf, removal: { CO2: rem('CO2'), N2: rem('N2'), O2: rem('O2') } } };
  },
  // Reduced brine-recirculation multi-stage flash: the stage-by-stage flash equation fixes the recirculating brine flow, the brine heater the steam demand.
  msf: ([s], p) => {
    if (!(s.m > 0)) return { outs: [zeroS(s.T, 2), zeroS(s.T, P0 + 0.5), s], rec: { heat: 0, recovery: 0 } };
    const Fm = s.m * p.make, D = Fm * p.rec, wF = s.w.map((x) => x * p.make), kd = s.salt > 0 ? (D * 5e-6) / (s.salt * p.make) : 0, wd = wF.map((x) => x * kd), SB = Math.min(SMAX, (1000 * (sum(wF) - sum(wd))) / (Fm - D));
    const N = Math.max(2, Math.round(p.stages)), dT = (p.TBT - p.Tlast) / N, prof = [];
    let Sr = SB, Mr = Fm, yTot = 0;
    for (let it = 0; it < 4; it++) { // flash of the recirculating brine through the stages; its salinity follows from the make-up and the recycle
      let B = 1; prof.length = 0;
      for (let j = 1; j <= N; j++) { const dj = B * flashFraction(p.TBT - (j - 1) * dT, p.TBT - j * dT, Math.min(Sr, 160)); B -= dj; prof.push({ stage: j, T: p.TBT - j * dT, d: dj, cum: 1 - B }); }
      yTot = 1 - B; Mr = Math.max(Fm, D / yTot); Sr = (Fm * s.S + (Mr - Fm) * SB) / Mr;
    }
    const Tm = 0.5 * (p.TBT + p.Tlast), loss = bpe(Tm, Math.min(Sr, 120)) + 1, t1 = p.TBT - dT - loss - p.TTD, Ts = p.TBT + 8; // recycle brine leaves the recovery tubes at t1
    const heat = (Mr * cp(0.5 * (p.TBT + t1), Math.min(Sr, 160)) * (p.TBT - t1)) / 3.6e6, steam = (heat * 1000) / latentHeat(Ts), aux = (p.secElec * D) / 1000, cool = heat * p.loss;
    const dist = mkS(D, wd, p.Tlast, 2, { pH: 6.5, co2: 0 }), brine = mkS(Fm - D, wF.map((x, i) => x - wd[i]), p.Tlast, P0 + 0.5, { pH: s.pH, co2: s.co2 * p.make, tss: s.tss * p.make });
    const cw = settle([{ m: s.m - Fm, w: s.w.map((x) => x * (1 - p.make)), P: Math.max(P0 + 0.5, s.P - 1), o: { co2: s.co2 * (1 - p.make), tss: s.tss * (1 - p.make), pH: s.pH } }], Hk(s) + heat + aux - cool - Hk(dist) - Hk(brine), s.T + 8)[0];
    const lmtd = dT / Math.log((dT + p.TTD) / p.TTD), area = ((Mr * yTot) / 3600 * latentHeat(Tm)) / 1000 / (2.6 * lmtd) + heat / (2.6 * Math.max(2, (p.TBT - t1) / Math.log((Ts - t1) / (Ts - p.TBT))));
    return { outs: [dist, brine, cw], rec: { heat, Ts, cool, power: aux, shaft: aux, recovery: D / s.m, steam, effects: N, stages: N, dTcw: cw.T - s.T, GOR: steam > 0 ? D / 3600 / steam : 0, Mr, Sr, yTot, t1, dTstage: dT, prof, area } };
  },
  // Electrodialysis stack: Faraday's law fixes the current, ohmic and Donnan potentials the cell-pair voltage; charged species move to the concentrate in proportion to their equivalents.
  ed: ([s], p) => {
    if (!(s.Q > 1e-9)) return { outs: [s, zeroS(s.T, s.P)], rec: { recovery: 0, power: 0, shaft: 0, dP: 0 } };
    const c = ionsOf(s), ceq = sum(ION_IDS.map((k) => (IONS[k].z > 0 ? (IONS[k].z * c[k]) / IONS[k].mw : 0))), rec = clamp(p.rec, 0.3, 0.99), cut = clamp(p.cut, 0, 0.99), Qd = s.Q * rec, Neq = Qd * ceq * cut; // eq/m³ and eq/h
    if (!(Neq > 0)) return { outs: [scaleS(s, rec), scaleS(s, 1 - rec)], rec: { recovery: rec, power: 0, shaft: 0, dP: 0, area: 0, cellPairs: 0 } };
    const lm = (a, b) => (Math.abs(a - b) < 1e-12 * a ? a : (a - b) / Math.log(a / b)), cdo = ceq * (1 - cut), cco = ceq + Neq / (s.Q - Qd), cd = lm(ceq, cdo), cc = lm(cco, ceq), lam = 0.0105 * (1 + 0.02 * (s.T - 25)); // equivalent conductivity, S·m²/eq
    const ilim = (FARADAY * p.k * cdo) / 0.5, i = p.iFrac * ilim, Itot = (FARADAY * Neq) / 3600 / p.xi, area = Itot / i, don = 2 * p.alpha * ((R * (s.T + KELVIN)) / FARADAY) * Math.log(cc / cd), ohm = i * (p.rcp * 1e-4 + p.gap / (lam * cd) + p.gap / (lam * cc)), U = ohm + don;
    const dc = (U * Itot) / 1000, pump = (s.Q * p.dp) / 36 / 0.75, shaft = dc + pump, mwt = (p.tw * Neq * MW_W) / 1000, wd = s.w.map((x, j) => x * rec * (ZI[j] !== 0 ? 1 - cut : 1)), md = s.m * rec - mwt - (rec * s.salt - sum(wd));
    const outs = settle([{ m: md, w: wd, P: s.P, o: { co2: s.co2 * rec, pH: s.pH } }, { m: s.m - md, w: s.w.map((x, j) => x - wd[j]), P: s.P, o: { co2: s.co2 * (1 - rec), pH: s.pH, tss: s.tss } }], Hk(s) + shaft, s.T);
    return { outs, rec: { recovery: outs[0].Q / s.Q, dP: 0, shaft, power: dc / p.etaRect + pump / p.etaM, area, cellPairs: area / p.aCell, i, ilim, U, don, ohm, Itot, dc, Neq, ceq, cut, sec: (dc / p.etaRect + pump / p.etaM) / outs[0].Q, minWork: (R * (s.T + KELVIN) * Neq * Math.log(cc / cd)) / 3.6e6 } };
  },
  // Direct-contact membrane distillation of brine: vapour-pressure driving force with the water activity of the brine, conduction loss through the membrane, heat recovery.
  md: ([s], p) => {
    const idle = { outs: [zeroS(s.T, P0 + 0.3), s], rec: { recovery: 0, power: 0, shaft: 0, heat: 0, area: 0, flux: 0 } };
    if (!(s.m > 0) || s.S >= 0.98 * SMAX) return idle;
    const n = 8, dTm = p.tpc * (p.Th - p.Tc), Tfm = 0.5 * (p.Th + p.Tc) + dTm / 2, Tpm = Tfm - dTm, Nion = sum(s.w.map((x, i) => (1000 * x) / MW[i])), Dreq = Math.min(s.m * p.rec, 0.98 * (s.m - (1000 * s.salt) / SMAX));
    let area = 0, Q = 0, lat = 0, D = 0;
    for (let k = 0; k < n; k++) { // the brine concentrates along the module, which lowers its vapour pressure
      const mk = s.m - (Dreq * (k + 0.5)) / n, aw = waterActivity(Tfm, Math.min((1000 * s.salt) / mk, 200), Nion / (mk - s.salt)), J = p.Bm * (aw * psat(Tfm) - psat(Tpm)); // kg/m²·s
      if (!(J > 1e-9)) break;
      const a = Dreq / n / 3600 / J; area += a; lat += (Dreq / n / 3600) * latentHeat(Tfm); Q += (Dreq / n / 3600) * latentHeat(Tfm) + p.hm * dTm * a; D += Dreq / n;
    }
    if (!(D > 0)) return idle;
    const heat = (Q * (1 - p.eps)) / 1000, aux = p.aux * s.Q, wd = s.w.map((x) => (x * D * 1e-5) / Math.max(s.salt, 1e-12)), specs = [{ m: D, w: wd, P: P0 + 0.3, o: { pH: 6.5, co2: 0 } }, { m: s.m - D, w: s.w.map((x, i) => x - wd[i]), P: P0 + 0.3, o: { pH: s.pH, co2: s.co2, tss: s.tss } }];
    let outs = [mkS(specs[0].m, specs[0].w, p.Tc, specs[0].P, specs[0].o), mkS(specs[1].m, specs[1].w, p.Tc + 5, specs[1].P, specs[1].o)], cool = Hk(s) + heat + aux - sum(outs.map(Hk));
    if (cool < 0) { outs = settle(specs, Hk(s) + heat + aux, p.Tc + 5); cool = 0; }
    return { outs, rec: { heat, Ts: p.Th + 10, cool, shaft: aux, power: aux / p.etaM, recovery: D / s.m, area, flux: D / area, eta: lat / Q, GOR: lat / 1000 / heat, Tfm, Tpm, dTm, dP: P0 + 0.3 - s.P, short: D < Dreq * 0.999 && D < s.m * p.rec * 0.999 } };
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
  msfIn: 'Seawater to MSF', msfD: 'MSF distillate', msfD2: 'Cooled distillate', msfB: 'MSF blowdown', msfCW: 'MSF cooling-water reject', dea: 'Deaerated feed', edD: 'ED diluate', edC: 'ED concentrate', mdD: 'MD distillate', mdC: 'MD concentrate',
  bcD: 'Concentrator distillate', bcC: 'Concentrated brine', crC: 'Crystalliser condensate', solids: 'Salt cake', purge: 'Crystalliser purge', brineMix: 'Blended brine and waste', brine: 'Discharge' };

function buildFlowsheet(v, ref, C = {}) {
  const t = v.template, U = [], tears = [], prod = [], brines = [], etaM = v.etaMotor / 100, lp = { eta: v.etaLP / 100, etaM }, hp = { eta: v.etaHP / 100, etaM };
  const add = (id, name, type, row, ins, outs, p = {}) => U.push({ id, name, type, row: ROWS[row], in: ins, out: outs, p: typeof p === 'function' ? p : () => p });
  const ro = (key, refK, rec, area) => ({ ref: refK, recovery: rec / 100, area, key, Pp: v.Pperm, ff: v.roFF, kSP: v.kSP, ...hp });
  let cur = 'raw';
  add('SCR', 'Intake and screens', 'screen', 0, [cur], ['s1'], { dp: v.screenDp }); cur = 's1';
  add('IP', 'Intake pump', 'pump', 0, [cur], ['s2'], { Pout: v.Pint, ...lp }); cur = 's2';
  if (t === 'hybrid') {
    const msf = v.thermalTech === 'msf', th = msf ? 'msf' : 'med';
    add('MS', `Seawater split to ${msf ? 'MSF' : 'MED'}`, 'split', 0, [cur], [th + 'In', 's2b'], { frac: v.medFrac / 100 }); cur = 's2b';
    if (msf) add('MSF', 'MSF evaporator (brine recirculation)', 'msf', 2, ['msfIn'], ['msfD', 'msfB', 'msfCW'], { make: v.medMakeup / 100, rec: v.medRec / 100, stages: v.msfStages, TBT: Math.max(v.msfTBT, v.medTlast + 20), TTD: v.msfTTD, Tlast: v.medTlast, secElec: v.msfSecElec, loss: v.medLoss / 100 });
    else add('MED', 'MED evaporator and condenser', 'med', 2, ['medIn'], ['medD', 'medB', 'medCW'], { make: v.medMakeup / 100, rec: v.medRec / 100, GOR: v.medGOR, Ts: v.medTs, Tlast: v.medTlast, secElec: v.medSecElec, loss: v.medLoss / 100 });
    let dk = th + 'D';
    if (v.useHX) { add('HX', 'Distillate cooler / feed preheater', 'hx', 2, [dk, cur], [th + 'D2', 's2c'], { UA: v.hxUA, U: v.hxU, dp: 0.3 }); dk = th + 'D2'; cur = 's2c'; }
    if (v.rejectToRO) { add('MXW', 'Warm-reject mixer', 'mix', 0, [cur, th + 'CW'], ['s2d']); cur = 's2d'; } else brines.push(th + 'CW');
    prod.push(dk); brines.push(th + 'B');
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
  if (v.useDeaer) { add('DEA', 'Vacuum deaerator / decarbonator', 'deaer', 0, [cur], ['dea'], { P: v.deaerP, model: v.vleModel, sat: v.airSat / 100, etaV: v.etaVac / 100, compModel: v.compModel, ...lp }); cur = 'dea'; }
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
  if (v.polish === 'ed') { add('ED', 'Electrodialysis polishing', 'ed', 3, [pk], ['edD', 'edC'], { cut: v.edCut / 100, rec: v.edRec / 100, iFrac: v.edIFrac / 100, xi: v.edXi / 100, rcp: v.edRcp, gap: 5e-4, k: 3e-5, alpha: 0.95, tw: 8, dp: 0.8, aCell: 0.6, etaRect: 0.95, etaM }); pk = 'edD'; brines.push('edC'); }
  prod.unshift(pk);
  let bk = 'brine1';
  if (t !== 'mld' && v.brineConc === 'md') { add('MD', 'Membrane distillation of RO brine', 'md', 2, [bk], ['mdD', 'mdC'], { rec: v.mdRec / 100, Th: v.mdTh, Tc: Math.min(v.mdTc, v.mdTh - 10), Bm: v.mdBm / 3.6e6, hm: v.mdHm, tpc: v.mdTPC, eps: v.mdEps / 100, aux: 0.6, etaM }); prod.push('mdD'); bk = 'mdC'; }
  if (t === 'mld') {
    add('BC', 'Brine concentrator (MVC)', 'bc', 2, [bk], ['bcD', 'bcC'], { target: v.bcTarget, dT: v.bcDT, etaComp: v.etaComp / 100, compModel: v.compModel, aux: v.bcAux, etaM }); prod.push('bcD'); bk = 'bcC';
    if (v.useCryst) { add('CR', 'Crystalliser', 'cryst', 2, [bk], ['crC', 'solids', 'purge'], { purge: v.crystPurge / 100, moist: v.crystMoist / 100, drive: v.crystDrive, dT: v.bcDT, etaComp: v.etaComp / 100, compModel: v.compModel, etaM }); prod.push('crC'); bk = 'purge'; }
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
  const k = r.rec, mIn = sum(r.inS.map((s) => s.m)) + (k.chemMass || 0), mOut = sum(r.outS.map((s) => s.m)) + (k.ventMass || 0);
  let ion = 0, scale = 1e-12;
  for (let i = 0; i < NI; i++) { const a = sum(r.inS.map((s) => s.w[i])) + (k.chemIons?.[i] || 0) + (k.gen?.[i] || 0), b = sum(r.outS.map((s) => s.w[i])); ion = Math.max(ion, Math.abs(a - b) / Math.max(1e-9, a, sum(r.inS.map((s) => s.salt)) * 1e-6)); scale = Math.max(scale, a); }
  const hIn = sum(r.inS.map(Hk)) + (k.shaft || 0) + (k.heat || 0) + (k.Hadd || 0), hOut = sum(r.outS.map(Hk)) + (k.cool || 0) + (k.ventH || 0);
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
    // entropy balance: generation = entropy out − entropy in − entropy carried by heat; heat not carried by the streams leaves to the surroundings at T0
    const hi = sum(q.inS.map((s) => hInc(s, dead))), ho = sum(q.outS.map((s) => hInc(s, dead))), a0 = q.inS[0], fa = k.chemMass > 0 && a0.m > 0 ? k.chemMass / a0.m : 0;
    q.sIn = sum(q.inS.map((s) => entropy(s, dead))) + fa * (a0.m > 0 ? (a0.m * cp(0.5 * (a0.T + dead.T), Math.min(a0.S, SMAX)) * Math.log((a0.T + KELVIN) / T0)) / 3.6e6 : 0); q.sOut = sum(q.outS.map((s) => entropy(s, dead)));
    q.sQ = k.heat ? k.heat / ((k.Ts ?? 100) + KELVIN) : 0;
    q.q0 = hi + fa * hInc(a0, dead) + Math.max(0, k.power || 0) + (k.heat || 0) - ho - Math.max(0, -(k.power || 0)); // heat rejected to the surroundings (first law on the same property basis)
    q.sGen = q.sOut - q.sIn - q.sQ + q.q0 / T0;
    power += k.power || 0; heat += k.heat || 0; cool += k.cool || 0; exIn += Math.max(0, k.power || 0) + q.exQ + q.exAdd; sludge += k.sludge || 0; solids += k.solids || 0;
    if (k.power) consumers[q.u.name] = (consumers[q.u.name] || 0) + k.power;
    for (const [n, x] of Object.entries(k.chem || {})) chem[n] = (chem[n] || 0) + x;
  }
  const product = S.product, brine = S.brine, outs = fs.outs.map((k) => S[k]), chemMass = sum(sol.recs.map((q) => q.rec.chemMass || 0)), ventMass = sum(sol.recs.map((q) => q.rec.ventMass || 0)), ventH = sum(sol.recs.map((q) => q.rec.ventH || 0));
  const least = sum(outs.map((s) => exergy(s, dead)[1])) - exergy(raw, dead)[1], ro1 = get('RO1')?.rec || {}, ro2 = get('RO2')?.rec, nf = get('NF')?.rec;
  const plant = {
    mass: { in: raw.m + chemMass, out: sum(outs.map((s) => s.m)) + ventMass },
    ion: ION_IDS.map((_, i) => ({ in: raw.w[i] + sum(sol.recs.map((q) => (q.rec.chemIons?.[i] || 0) + (q.rec.gen?.[i] || 0))), out: sum(outs.map((s) => s.w[i])) })),
    energy: { in: Hk(raw) + sum(sol.recs.map((q) => (q.rec.shaft || 0) + (q.rec.heat || 0) + (q.rec.Hadd || 0))), out: sum(outs.map(Hk)) + cool + ventH },
  };
  return { v, cond: { T, sf, load }, raw, fs, sol, S, dead, get, product, brine, solidsS: S.solids, power, heat, cool, exIn, exDest: sum(sol.recs.map((q) => q.exDest)), sGen: sum(sol.recs.map((q) => q.sGen)), ventMass, least, chem, consumers, sludge, solids, ro1, ro2, nf, plant,
    recovery: raw.Q > 0 ? product.Q / raw.Q : 0, secElec: product.Q > 0 ? power / product.Q : 0, secThermal: product.Q > 0 ? heat / product.Q : 0, etaII: exIn > 0 ? least / exIn : 0, prodIons: ionsOf(product) };
}

/** Design sizes taken from the converged design case. */
function designOf(res) {
  const g = (id) => res.get(id), q = (id) => g(id)?.inS[0]?.Q || 0;
  return { area1: res.ro1.area, area2: res.ro2?.area, areaNF: res.nf?.area, P1: res.ro1.Preq, hpPower: g('HPP')?.rec.power || 0, preQ: q('PRE') || q('CF') || res.raw.Q, pxQ: q('PX'), bcQ: q('BC'), medQ: q('MED') || q('MSF'), J1: res.ro1.flux, rawQ: res.raw.Q };
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
  const thm = g('MED') || g('MSF');
  if (thm) add(`${thm.u.id} condenser temperature rise`, thm.rec.dTcw, v.medMaxDT, 'K');
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
  let y = [o.L0 * o.Vmax, 0, 0, 0, 0], t = 0, running = clamp(Math.ceil(o.uff * o.nTrains - 1e-9), 0, o.nTrains), last = -9, spill = 0, unmet = 0, starts = 0, cap = 1;
  const uFree = (tt, V, I) => { if (onoff) return running / o.nTrains; if (fixed) return o.uff; return clamp(o.uff + o.Kp * (o.Lsp(tt) - V / o.Vmax) + o.Ki * I, o.uMin, 1); };
  const uOf = (tt, V, I) => Math.min(cap, uFree(tt, V, I)); // cap < 1 only while the available electrical power limits production (load shedding)
  const f = (tt, [V, I]) => { const e = o.Lsp(tt) - V / o.Vmax, raw = o.uff + o.Kp * e + o.Ki * I, u = uOf(tt, V, I), sat = (raw > 1 && e > 0) || (raw < o.uMin && e < 0) || (cap < 1 && e > 0), q = u * o.Qn, pw = q * o.sec(u); return [q - o.demand(tt), onoff || fixed || sat ? 0 : e, q - o.demand(tt), pw, pw * o.price(tt)]; };
  const out = { t: [], L: [], u: [], Qd: [], P: [], sp: [] }, sp = o.supply, pwr = (u) => u * o.Qn * o.sec(u);
  const sy = sp ? { E: sp.E0 || 0, ren: [], grid: [], soc: [], curt: [], dis: [], chg: [], shed: 0, shedSteps: 0 } : null;
  for (let k = 0; k <= n; k++) {
    const lev = y[0] / o.Vmax;
    if (onoff && t - last >= 0.5 - 1e-9) { if (lev < o.onLow && running < o.nTrains) { running++; starts++; last = t; } else if (lev > o.offHigh && running > 0) { running--; last = t; } }
    if (sp) { // supply dispatch held over the step: renewables first, then the battery, then grid import up to its limit; production is shed if that is not enough
      const u0 = uFree(t, y[0], y[1]), ren = sp.pv(t) + sp.wind(t), disMax = Math.min(sp.Pb || 0, (sy.E * sp.eta) / dt), avail = ren + disMax + sp.gridMax;
      cap = 1;
      if (pwr(u0) > avail + 1e-9) { let lo = 0, hi = u0; for (let i = 0; i < 40; i++) { const mid = 0.5 * (lo + hi); if (pwr(mid) <= avail) lo = mid; else hi = mid; } cap = onoff ? Math.floor(lo * o.nTrains + 1e-9) / o.nTrains : lo < o.uMin - 1e-9 ? 0 : lo; }
      const ul = Math.min(cap, u0), P = pwr(ul), sur = ren - P, chg = sur > 0 ? Math.min(sur, sp.Pb || 0, ((sp.Emax || 0) - sy.E) / (sp.eta * dt)) : 0, dis = sur < 0 ? Math.min(-sur, disMax) : 0;
      sy.ren.push(ren); sy.chg.push(chg); sy.dis.push(dis); sy.curt.push(Math.max(0, sur - chg)); sy.grid.push(Math.max(0, -sur - dis)); sy.soc.push(sp.Emax > 0 ? (100 * sy.E) / sp.Emax : 0);
      if (k < n) { sy.E += (chg * sp.eta - dis / sp.eta) * dt; if (cap < 1) { sy.shed += (u0 - ul) * o.Qn * dt; sy.shedSteps++; } }
    }
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
  if (sy) { // energy bookkeeping of the supply (rectangle rule on the held values), whole run and last day
    const I = (a, from = 0) => sum(a.slice(from, n)) * dt, j0 = Math.max(0, i0), load = out.P;
    Object.assign(sy, { Eend: sy.E, total: { ren: I(sy.ren), grid: I(sy.grid), curt: I(sy.curt), chg: I(sy.chg), dis: I(sy.dis), load: I(load) }, day: { ren: I(sy.ren, j0), grid: I(sy.grid, j0), curt: I(sy.curt, j0), chg: I(sy.chg, j0), dis: I(sy.dis, j0), load: I(load, j0) }, gridCost: sum(sy.grid.slice(j0, n).map((g, i) => g * o.price(out.t[j0 + i]))) * dt, gridPeak: Math.max(...sy.grid) });
  }
  return { ...out, ...(sy ? { supply: sy } : {}), i0: Math.max(0, i0), dt, n, Vend: y[0], net: y[2], spill, unmet, starts, energy, cost, iae: (sum(e) * dt), Lmin: Math.min(...last24(out.L)), Lmax: Math.max(...last24(out.L)), sat: last24(out.u).filter((x) => x >= 99.999 || x <= 100 * o.uMin + 1e-3).length / Math.max(1, last24(out.u).length), totalEnergy: y[3], totalCost: y[4] };
}

/** Hourly PV and wind power over a day (kW at hours 0…24) from the daily irradiation (kWh/m²·d) and the mean wind speed (m/s). */
export function resourceProfile(o) {
  const hours = linspace(0, 24, 25), shape = hours.map((h) => (h > 6 && h < 18 ? Math.sin((Math.PI * (h - 6)) / 12) ** 1.3 : 0)), A = trapz(hours, shape);
  const G = shape.map((x) => (1000 * o.ghi * x) / A), v = hours.map((h) => o.wind * (1 + 0.2 * Math.cos((2 * Math.PI * (h - 15)) / 24))); // W/m² and m/s
  const curve = (u) => (u < 3 || u > 25 ? 0 : u >= 12 ? 1 : (u ** 3 - 27) / (1728 - 27)), us = linspace(0, 30, 121);
  const cf = (vm) => (vm > 0 ? trapz(us, us.map((u) => curve(u) * ((Math.PI * u) / (2 * vm * vm)) * Math.exp((-Math.PI * u * u) / (4 * vm * vm)))) : 0); // Rayleigh-distributed speed within the hour
  const pv = G.map((g) => (o.pvKW * (o.pr ?? 0.8) * g) / 1000), wind = v.map((vm) => o.windKW * cf(vm)), at = (a) => (t) => interp1(hours, a, ((t % 24) + 24) % 24);
  return { hours, G, v, pv, wind, pvAt: at(pv), windAt: at(wind), pvEnergy: trapz(hours, pv), windEnergy: trapz(hours, wind) };
}

/**
 * Thermal inventory of the product tank and the product-pH trim loop, driven by the flows of the level simulation.
 * Tank: dV/dt = q_in − q_out, d(V·T)/dt = q_in·T_p − q_out·T − UA·(T − T_amb)/(ρc), with the stored energy ρc·V·T as initial condition and a daily ambient cycle.
 * pH loop: stirred contact tank with carbonate equilibrium, first-order sensor lag and a PID controller on the caustic trim
 * (derivative on the measurement, output limits, back-calculation anti-windup). Times in hours, concentrations in mol/m³.
 */
export function auxDynamics(o) {
  const RC = 1.161, n = Math.max(8, Math.round(o.hours / o.dt)), ph = o.ph, eps = 1e-3 * o.Vmax, qIn = (t) => Math.max(0, interp1(o.t, o.qIn, t)), qD = (t) => Math.max(0, interp1(o.t, o.qOut, t)), Ta = (t) => o.Ta + o.swing * Math.cos((2 * Math.PI * (t - 15)) / 24);
  const pHof = (A, Ct) => (ph ? clamp(ph.pK + Math.log10(Math.max(A, 1e-12) / Math.max(Ct - A, 1e-9 * Ct)), 3.5, 10.8) : 7);
  const ctrl = (y, q) => { const pH = pHof(y[5], y[6]), e = ph.sp - y[7], raw = ph.Kc * (e + y[8] / ph.Ti - (ph.Td * (pH - y[7])) / ph.tauM); return { pH, e, raw, s: q > 1e-9 ? clamp(raw, 0, ph.sMax) : 0 }; };
  // states: volume, V·T, energy in, energy out, heat loss, alkalinity, total carbonate, measured pH, integral of the error, integral |error|, caustic used
  const f = (t, y) => {
    const V = Math.max(y[0], eps), T = y[1] / V, q = qIn(t), dem = qD(t), qo = y[0] <= eps ? Math.min(dem, q) : y[0] >= o.Vmax ? Math.max(dem, q) : dem, loss = o.UA * (T - Ta(t)), d = [q - qo, q * o.Tp - qo * T - loss / RC, RC * q * o.Tp, RC * qo * T, loss, 0, 0, 0, 0, 0, 0];
    if (ph) { const c = ctrl(y, q), k = q / ph.Vc; d[5] = k * (ph.A0 + c.s - y[5]); d[6] = k * (ph.A0 + ph.c0 * (1 + ph.dist(t)) - y[6]); d[7] = (c.pH - y[7]) / ph.tauM; d[8] = q > 1e-9 ? c.e + (ph.aw ? (ph.Ti / (ph.Kc * ph.Tt)) * (c.s - c.raw) : 0) : 0; d[9] = Math.abs(ph.sp - c.pH); d[10] = c.s * q; }
    return d;
  };
  const r = rk4(f, [o.V0, o.V0 * o.T0, 0, 0, 0, ph ? ph.A0 + ph.s0 : 0, ph ? ph.A0 + ph.c0 : 0, ph ? ph.sp : 7, ph ? (ph.s0 * ph.Ti) / ph.Kc : 0, 0, 0], 0, o.hours, n), every = Math.max(1, Math.round(n / 432)), out = { t: [], T: [], Ta: [], E: [], pH: [], pHm: [], dose: [] };
  const Tof = (y) => y[1] / Math.max(y[0], eps), k24 = Math.max(0, n - Math.round(24 / (o.hours / n))), ye = r.y[n], y24 = r.y[k24];
  for (let k = 0; k <= n; k += every) { const y = r.y[k], t = r.t[k]; out.t.push(t); out.T.push(Tof(y)); out.Ta.push(Ta(t)); out.E.push(RC * y[0] * (Tof(y) - o.Ta)); if (ph) { const c = ctrl(y, qIn(t)); out.pH.push(c.pH); out.pHm.push(y[7]); out.dose.push(c.s); } }
  const res = { ...out, i0: Math.max(0, out.t.findIndex((x) => x >= o.hours - 24 - 1e-9)), Tend: Tof(ye), Vend: ye[0], stored0: RC * o.V0 * o.T0, storedEnd: RC * ye[1], Ein: ye[2], Eout: ye[3], Eloss: ye[4], lossDay: ye[4] - y24[4], Tmin: Math.min(...out.T), Tmax: Math.max(...out.T) };
  if (ph) { const day = r.y.slice(k24), cs = day.map((y, j) => ctrl(y, qIn(r.t[k24 + j]))), pHs = cs.map((c) => c.pH);
    Object.assign(res, { iae: ye[9] - y24[9], doseMol: ye[10] - y24[10], sat: cs.filter((c) => c.raw <= 0 || c.raw >= ph.sMax).length / cs.length, Imax: Math.max(...r.y.map((y) => Math.abs(y[8]))), pHend: pHs[pHs.length - 1], pHmEnd: ye[7], pHmin: Math.min(...pHs), pHmax: Math.max(...pHs) }); }
  return res;
}

/** Capital-recovery factor. */
export const crf = (i, n) => (Math.abs(i) < 1e-12 ? 1 / n : (i * (1 + i) ** n) / ((1 + i) ** n - 1));
const PRICE = { 'Ferric chloride': 0.5, 'Sulphuric acid': 0.15, Antiscalant: 3, 'Sodium bisulphite': 0.6, 'Sodium hypochlorite': 0.9, 'Caustic soda': 0.55, 'Carbon dioxide': 0.15, 'Hydrated lime': 0.18, Calcite: 0.08 }; // $/kg, indicative
/**
 * Indicative unit-level cost roll-up: purchased-equipment cost of every unit from simple size-based cost functions, an installation factor,
 * annualisation with the capital-recovery factor and operating costs (energy, heat, chemicals, membranes, maintenance, labour, residuals).
 */
export function plantCost(res, v) {
  const rows = [], d = (Q) => 24 * Q, pre = { daf_dmf: 160, dmf: 95, uf: 140 };
  let elements = 0;
  for (const q of res.sol.recs) {
    const k = q.rec, a = q.inS[0], b = q.outS[0], t = q.u.type;
    let c = 0, basis = '';
    if (t === 'screen') { c = 45 * d(a.Q); basis = '45 $ per m³/d of intake'; }
    else if (t === 'pump' && k.shaft > 0) { const hp = b.P > 20; c = 2500 * k.shaft ** 0.75 * (hp ? 1.8 : 1); basis = `2500·kW^0.75${hp ? ' × 1.8 (high pressure)' : ''}, ${fmt(k.shaft, 3)} kW shaft`; }
    else if (t === 'turbine') { c = 450 * Math.abs(k.shaft); basis = '450 $ per kW recovered'; }
    else if (t === 'filter') { const u = q.u.id === 'PRE' ? pre[v.pretreat] || 120 : 10; c = u * d(a.Q); basis = `${u} $ per m³/d of feed`; }
    else if (t === 'dose') { const nC = Object.keys(k.chem || {}).length; c = 15000 * nC + 2 * d(a.Q); basis = `${nC} dosing system(s) at 15 000 $ + 2 $ per m³/d`; }
    else if (t === 'ro') { c = 650 * k.elements + 2800 * k.vessels + 60 * k.area; elements += k.elements || 0; basis = `${Math.ceil(k.elements || 0)} elements × 650 $ + ${Math.ceil(k.vessels || 0)} vessels × 2800 $ + 60 $/m² racks and piping`; }
    else if (t === 'px') { c = 28000 * Math.ceil(a.Q / 68); basis = '28 000 $ per 68 m³/h unit'; }
    else if (t === 'hx') { c = 600 * (k.area || 0); basis = '600 $ per m² of heat-transfer area'; }
    else if (t === 'med') { c = 900 * d(b.Q); basis = '900 $ per m³/d of distillate'; }
    else if (t === 'msf') { c = 1250 * d(b.Q); basis = '1250 $ per m³/d of distillate'; }
    else if (t === 'bc') { c = 4000 * d(b.Q); basis = '4000 $ per m³/d of distillate'; }
    else if (t === 'cryst') { c = 9000 * d(a.Q); basis = '9000 $ per m³/d of feed'; }
    else if (t === 'md') { c = 180 * (k.area || 0); basis = '180 $ per m² of membrane (modules and heat exchangers)'; }
    else if (t === 'ed') { c = 320 * (k.area || 0); basis = '320 $ per m² of cell pair (stack and rectifier)'; }
    else if (t === 'deaer') { c = 12 * d(a.Q) + 4000 * (k.shaftV || 0) ** 0.7; basis = '12 $ per m³/d + vacuum pump 4000·kW^0.7'; }
    else if (t === 'remin') { c = 22 * d(a.Q); basis = '22 $ per m³/d of product'; }
    else if (t === 'tank') { c = 160 * (k.volume || 0); basis = '160 $ per m³ of storage'; }
    if (c > 0) rows.push({ id: q.u.id, name: q.u.name, basis, cost: c });
  }
  const equip = sum(rows.map((r) => r.cost)), capex = equip * v.capexFactor, av = v.availability / 100, hrs = 8760 * av, annual = res.product.Q * hrs, f = crf(v.discount / 100, v.plantLife);
  const pk = (((v.peakEnd - v.peakStart) % 24) + 24) % 24 / 24, price = v.offPrice * (1 - pk) + v.peakPrice * pk;
  const opex = { 'Electricity': res.power * hrs * price, 'Heat': res.heat * hrs * v.heatPrice, 'Chemicals': sum(Object.entries(res.chem).map(([n, x]) => x * (PRICE[n] ?? 0.5))) * hrs, 'Membrane replacement': (650 * elements) / Math.max(1, v.memLife),
    'Maintenance (2 % of capital)': 0.02 * capex, 'Labour and overheads': v.labour * annual, 'Sludge and salt disposal': 0.06 * (res.sludge + res.solids) * hrs };
  const opexTot = sum(Object.values(opex)), annCap = f * capex;
  return { rows, equip, capex, crf: f, annCap, opex, opexTot, annual, price, lcow: annual > 0 ? (annCap + opexTot) / annual : 0, capexPerM3d: res.product.Q > 0 ? capex / d(res.product.Q) : 0 };
}

/**
 * Weighted least-squares data reconciliation: minimise Σ((x − y)/σ)² subject to g(x) = 0 by successive linearisation,
 * x ← y − V·Jᵀ·(J·V·Jᵀ)⁻¹·(g(x) + J·(y − x)). Returns the adjusted values, the global χ² test and the standardised adjustments (measurement test).
 */
export function reconcile(y, sig, g) {
  const n = y.length, V = sig.map((s) => s * s);
  const jac = (x) => { const g0 = g(x), J = g0.map(() => new Array(n).fill(0)); for (let j = 0; j < n; j++) { const h = 1e-6 * Math.max(1e-9, Math.abs(x[j])), xp = [...x]; xp[j] += h; const g1 = g(xp); for (let i = 0; i < g0.length; i++) J[i][j] = (g1[i] - g0[i]) / h; } return [g0, J]; };
  const JVJ = (J) => J.map((a) => J.map((b) => sum(a.map((aj, j) => aj * V[j] * b[j]))));
  let x = [...y], it = 0, J, g0;
  for (it = 1; it <= 40; it++) {
    [g0, J] = jac(x);
    const lam = solveLinear(JVJ(J), g0.map((gi, i) => gi + sum(J[i].map((a, j) => a * (y[j] - x[j]))))), xn = y.map((yj, j) => yj - V[j] * sum(J.map((r, i) => r[j] * lam[i]))), step = Math.max(...xn.map((q, j) => Math.abs(q - x[j]) / Math.max(1e-12, Math.abs(q))));
    x = xn;
    if (step < 1e-11) break;
  }
  it = Math.min(it, 40); [g0, J] = jac(x);
  const M = JVJ(J), m = g0.length, adj = y.map((yj, j) => yj - x[j]), obj = sum(adj.map((a, j) => (a * a) / V[j])), crit = [3.841, 5.991, 7.815, 9.488, 11.07, 12.592, 14.067, 15.507][Math.min(7, m - 1)];
  const z = adj.map((a, j) => { const e = new Array(m).fill(0).map((_, i) => J[i][j]), w = solveLinear(M, e), va = V[j] * V[j] * sum(e.map((ei, i) => ei * w[i])); return va > 1e-300 ? Math.abs(a) / Math.sqrt(va) : 0; });
  return { x, adj, z, obj, dof: m, crit, globalFail: obj > crit, residual: Math.max(...g0.map(Math.abs)), raw: g(y), iterations: it };
}

/** Quadratic response surface in two normalised variables: y = b·[1, a, b, a², b², a·b] fitted by least squares. */
export function fitSurface(X, Y) {
  const feat = ([a, b]) => [1, a, b, a * a, b * b, a * b], coef = lstsq(X.map(feat), Y);
  return { coef, predict: (x) => sum(feat(x).map((q, i) => q * coef[i])) };
}

/** Golden-section search for the minimum of f on [a, b]; returns the best point and every evaluation. */
export function goldenMin(f, a, b, n = 10) {
  const g = (Math.sqrt(5) - 1) / 2, evals = [], F = (x) => { const y = f(x); evals.push([x, y]); return y; };
  let c = b - g * (b - a), d = a + g * (b - a), fc = F(c), fd = F(d);
  for (let i = 0; i < n; i++) { if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = F(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = F(d); } }
  const best = evals.reduce((p, q) => (q[1] < p[1] ? q : p));
  return { x: best[0], f: best[1], evals: evals.sort((p, q) => p[0] - q[0]) };
}

const streamOut = (s) => ({ Q: s.Q, T: s.T, P: s.P, pH: s.pH, tds: s.tds, ions: Object.fromEntries(Object.entries(ionsOf(s)).map(([k, x]) => [k, +x.toPrecision(6)])) });
const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.type === 'table' ? f.value.map((r) => ({ ...r })) : f.value]));
const is = (...t) => (v) => t.includes(v.template);
const memOpts = Object.entries(MEMBRANES).map(([value, m]) => ({ value, label: m.name }));

const suite = {
  id: 'plant', num: 3, title: 'Whole-Plant Process Simulation', short: 'Plant flowsheet', icon: '🏭',
  tagline: 'Steady-state flowsheet from intake to outfall with recycle convergence, full stream tables, energy, exergy, off-design cases and tank dynamics.',
  description: 'Solves a complete desalination flowsheet unit by unit in the direction of flow and converges the recycle (tear) streams with Wegstein acceleration. Every stream carries mass flow, temperature, pressure and the mass flow of each ion, so total, component and energy balances close around every unit and around the plant, and exergy destruction is located unit by unit. Membrane passes use a reduced model anchored to the element-by-element solution of the RO suite; thermal desalination, the brine concentrator and the crystalliser are reduced energy-balance blocks. Off-design cases identify the first equipment limit reached, and a tank model with a PI level controller shows operation over a daily demand profile, optionally with PV, wind, a battery and a grid-import limit. The same run reports the entropy balance, the vapour–liquid equilibrium of the dissolved gases, an indicative cost roll-up, the environmental loads, a response-surface surrogate of the flowsheet and a data reconciliation of plant measurements.',
  guide: [
    'Choose a flowsheet template and switch optional units on or off; the Flowsheet tab draws the result.',
    'Enter the feed (or pull it from the Case page) and the key parameter of each unit — recovery, efficiencies, doses, targets.',
    'Run. Check the stream table and the balance closure on the Verify tab, then the power, chemical and exergy breakdowns.',
    'Read the off-design table to see which unit limits the plant at minimum and maximum temperature, salinity and part load, and the daily operation plots for the tank and controller.',
  ],
  implemented: ['total mass-balance', 'component mass-balance', 'steady-flow energy', 'enthalpy balance', 'entropy balance', 'exergy-balance', 'momentum/pressure-drop', 'flash equation', 'pump equation', 'compressor equation', 'valve equation', 'heat-exchanger equation', 'overall heat-transfer', 'logarithmic-mean-temperature-difference', 'effectiveness-ntu', 'reactor balance', 'separator equation', 'evaporator balance', 'crystallizer balance', 'membrane mass-transfer',
    'phase-equilibrium equation', 'equality-of-fugacity', 'raoult', 'henry', 'equations of state', 'activity-coefficient', 'rachford-rice',
    'membrane-process flowsheet', 'ro-med model', 'ro-crystallizer', 'ro-zld', 'process-exergy', 'process-environmental', 'steady-state/dynamic hybrid', 'mechanistic-surrogate',
    'electrolyte-thermodynamic', 'ro-msf', 'ro-ed model', 'ro-md model', 'desalination-renewable', 'process-economic', 'digital-twin', 'phase fraction', 'stored thermal energy', 'optimisation', 'plant-wide economic',
    'initial inventories', 'vessel liquid level', 'composition', 'temperature', 'pressure', 'equipment state', 'controller state', 'plant feed-flow/composition/temperature/pressure', 'product-water specification', 'discharge-pressure', 'utility steam', 'cooling-water', 'ambient heat-loss', 'electrical-power constraint', 'terminal process specification',
    'process-flowsheet construction', 'feed-water definition', 'material-stream management', 'mass balancing', 'energy balancing', 'thermodynamic-property calculation', 'pumps and compressors', 'valves and piping', 'mixers and splitters', 'separators', 'heat exchangers', 'membrane units', 'reactors', 'evaporators', 'condensers', 'crystallisers', 'chemical-dosing systems', 'utilities', 'recycle streams', 'energy recovery', 'steady-state simulation', 'dynamic simulation', 'process control', 'equipment sizing', 'sensitivity analysis', 'environmental accounting'],
  equationsNote: 'Steady state with lumped unit models. Membrane passes scale net driving pressure, pressure drop and ion passage from one element-by-element reference solution (suite 1) with flux, temperature and salinity, which is accurate near the design point and approximate far from it. MED, the brine concentrator and the crystalliser are reduced blocks (performance ratio, compressor work, single-effect heat demand) — use suites 6 and 9 for their detailed design. Enthalpy neglects heat of mixing; salinity exergy uses an ideal-mixture form scaled by the osmotic coefficient of the feed, and the exergy of dosed chemicals and solid salts is not counted. Seawater property correlations are extrapolated above about 160 g/kg. pH follows a simplified carbonate system. Dosing reactions are stoichiometric; the calcite contactor is sized with first-order dissolution kinetics. Entropy generation is evaluated on the same incompressible-liquid basis as the exergy, with the heat-of-mixing residual of the property model treated as heat exchanged with the surroundings. Phase equilibrium covers water with dissolved CO₂, N₂ and O₂ (γ–φ: water activity from the osmotic coefficient, Henry constants with Setschenow salting-out, Peng–Robinson vapour, zero binary interaction parameters); the deaerator is one adiabatic equilibrium stage and bicarbonate is not re-speciated during the flash. Ion activities use the Truesdell–Jones extended Debye–Hückel equation with the main sulphate and bicarbonate ion pairs — indicative above about 2 mol/kg. MSF (brine recirculation, stage-wise flash), electrodialysis (Faraday’s law, ohmic and Donnan potentials, uniform current density, non-selective transport of charged species) and membrane distillation (vapour-pressure driving force with fixed temperature-polarisation coefficient) are reduced blocks — use suites 6, 7 and 8 for their design. Cost functions are indicative order-of-magnitude correlations. The surrogate is a quadratic response surface valid only inside the off-design envelope. Data reconciliation assumes steady state and takes unmetered side streams from the model. Renewable profiles are a clear-day solar shape and a diurnal wind cycle scaled to the site means, not measured time series. The pH loop uses the first carbonic-acid equilibrium only.',

  inputs: [
    { group: 'Feed water', help: 'Plant boundary condition: what the intake delivers.', fields: [
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.seawater.ions, help: 'Full ionic analysis carried through every stream.' },
      { key: 'Qf', label: 'Intake flow', unit: 'm³/h', value: 1000, min: 1, max: 2e5 },
      { key: 'T', label: 'Feed temperature (design)', unit: '°C', value: 25, min: 2, max: 44 },
      { key: 'pH', label: 'Feed pH', unit: '', value: 8.1, min: 4, max: 10 },
      { key: 'tss', label: 'Suspended solids', unit: 'mg/L', value: 6, min: 0, max: 500 },
      { key: 'airSat', label: 'Dissolved-air saturation of the feed', unit: '% of saturation', value: 100, min: 0, max: 160, help: 'Dissolved nitrogen and oxygen relative to equilibrium with air at 1 atm (Henry’s law). Above about 101 % the feed carries a free-gas phase at atmospheric pressure; the phase fractions are reported with the results.' },
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
      { key: 'thermalTech', label: 'Thermal process of the hybrid', type: 'select', value: 'med', options: [{ value: 'med', label: 'Multi-effect distillation (MED)' }, { value: 'msf', label: 'Multi-stage flash (MSF, brine recirculation)' }], showIf: is('hybrid'), help: 'MSF is solved stage by stage with the flash equation; its gained output ratio is a result, not an input.' },
      { key: 'useDeaer', label: 'Vacuum deaerator / decarbonator on the conditioned feed', type: 'bool', value: false, help: 'Equilibrium flash of the feed under vacuum: strips oxygen, nitrogen and part of the dissolved CO₂ (which raises the pH); adds a vacuum pump and an extraction pump.' },
      { key: 'polish', label: 'Permeate polishing', type: 'select', value: 'none', options: [{ value: 'none', label: 'None' }, { value: 'ed', label: 'Electrodialysis of the permeate (RO–ED)' }], help: 'ED removes charged species only; its concentrate joins the discharge.' },
      { key: 'brineConc', label: 'Brine concentration after RO', type: 'select', value: 'none', options: [{ value: 'none', label: 'None' }, { value: 'md', label: 'Membrane distillation of the RO brine (RO–MD)' }], showIf: (v) => v.template !== 'mld', help: 'MD recovers distillate from the concentrate with low-grade heat; the distillate joins the product.' },
    ] },
    { group: 'Vacuum deaerator', showIf: (v) => v.useDeaer, fields: [
      { key: 'deaerP', label: 'Deaerator pressure', unit: 'bar a', value: 0.08, min: 0.015, max: 1, help: 'Must stay above the vapour pressure of the feed or the water boils off; lower pressure strips more gas.' },
      { key: 'etaVac', label: 'Vacuum-pump efficiency', unit: '%', value: 60, min: 20, max: 85, help: 'Isentropic or polytropic, as chosen on the model-setup tab.' },
    ] },
    { group: 'Electrodialysis polishing', showIf: (v) => v.polish === 'ed', fields: [
      { key: 'edCut', label: 'Salt removal (charged species)', unit: '%', value: 50, min: 5, max: 95 },
      { key: 'edRec', label: 'Water recovery of the stack', unit: '%', value: 92, min: 50, max: 98 },
      { key: 'edIFrac', label: 'Current density', unit: '% of limiting', value: 70, min: 10, max: 95, help: 'The limiting current density follows from the diluate outlet concentration.' },
      { key: 'edXi', label: 'Current efficiency', unit: '%', value: 90, min: 50, max: 99 },
      { key: 'edRcp', label: 'Membrane-pair area resistance', unit: 'Ω·cm²', value: 7, min: 1, max: 40 },
    ] },
    { group: 'Membrane distillation of the RO brine', showIf: (v) => v.brineConc === 'md' && v.template !== 'mld', fields: [
      { key: 'mdRec', label: 'Water recovered from the brine', unit: '%', value: 40, min: 5, max: 80 },
      { key: 'mdTh', label: 'Hot-side bulk temperature', unit: '°C', value: 70, min: 40, max: 90 },
      { key: 'mdTc', label: 'Cold-side bulk temperature', unit: '°C', value: 25, min: 10, max: 50 },
      { key: 'mdBm', label: 'Membrane distillation coefficient', unit: 'kg/m²·h·kPa', value: 1.8, min: 0.2, max: 10, help: 'Flux per unit vapour-pressure difference across the membrane.' },
      { key: 'mdHm', label: 'Membrane conduction coefficient', unit: 'W/m²·K', value: 400, min: 50, max: 2000 },
      { key: 'mdTPC', label: 'Temperature-polarisation coefficient', unit: '–', value: 0.6, min: 0.2, max: 0.95 },
      { key: 'mdEps', label: 'Heat recovered between passes', unit: '%', value: 60, min: 0, max: 90 },
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
      { key: 'medGOR', label: 'Gained output ratio', unit: 'kg/kg', value: 9.5, min: 2, max: 16, help: 'Distillate per kg of heating steam.', showIf: (v) => v.thermalTech !== 'msf' },
      { key: 'medTs', label: 'Heating-steam temperature', unit: '°C', value: 70, min: 55, max: 130, showIf: (v) => v.thermalTech !== 'msf' },
      { key: 'msfStages', label: 'Number of flash stages', unit: '', value: 24, min: 4, max: 40, step: 1, showIf: (v) => v.thermalTech === 'msf' },
      { key: 'msfTBT', label: 'Top brine temperature', unit: '°C', value: 110, min: 70, max: 120, showIf: (v) => v.thermalTech === 'msf' },
      { key: 'msfTTD', label: 'Condenser terminal temperature difference', unit: 'K', value: 3, min: 1, max: 8, showIf: (v) => v.thermalTech === 'msf' },
      { key: 'msfSecElec', label: 'MSF auxiliary electricity', unit: 'kWh/m³', value: 3.5, min: 1, max: 7, showIf: (v) => v.thermalTech === 'msf' },
      { key: 'medTlast', label: 'Last-effect temperature', unit: '°C', value: 40, min: 30, max: 60 },
      { key: 'medSecElec', label: 'MED auxiliary electricity', unit: 'kWh/m³', value: 1.5, min: 0.3, max: 5, showIf: (v) => v.thermalTech !== 'msf' },
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
    { group: 'Indicative economics', help: 'Order-of-magnitude cost functions per unit rolled up to a cost of water. Indicative only — use the Economics suite for a full estimate.', fields: [
      { key: 'discount', label: 'Discount rate', unit: '%/y', value: 6, min: 0, max: 25 },
      { key: 'plantLife', label: 'Plant life', unit: 'y', value: 25, min: 5, max: 50, step: 1 },
      { key: 'availability', label: 'Availability', unit: '%', value: 94, min: 50, max: 100 },
      { key: 'capexFactor', label: 'Installation factor on equipment cost', unit: '×', value: 2.2, min: 1, max: 5, help: 'Piping, electrical, instrumentation, civil works, engineering and contingency.' },
      { key: 'heatPrice', label: 'Heat price', unit: '$/kWh', value: 0.012, min: 0, max: 0.2 },
      { key: 'labour', label: 'Labour and overheads', unit: '$/m³', value: 0.05, min: 0, max: 1 },
      { key: 'memLife', label: 'Membrane life', unit: 'y', value: 5, min: 1, max: 12 },
    ] },
    { group: 'Solver and models', tab: 'setup', fields: [
      { key: 'tearMethod', label: 'Recycle convergence', type: 'select', value: 'wegstein', options: [{ value: 'wegstein', label: 'Wegstein (bounded)' }, { value: 'damped', label: 'Damped successive substitution' }, { value: 'direct', label: 'Direct successive substitution' }] },
      { key: 'damping', label: 'Damping factor', unit: '–', value: 0.6, min: 0.05, max: 1, showIf: (v) => v.tearMethod === 'damped' },
      { key: 'tearTol', label: 'Tear-stream tolerance (relative)', unit: '–', value: 1e-9, min: 1e-12, max: 1e-3 },
      { key: 'maxIter', label: 'Maximum iterations', unit: '', value: 120, min: 5, max: 1000, step: 1 },
      { key: 'roFF', label: 'Membrane flow factor', unit: '–', value: 0.95, min: 0.4, max: 1.2, help: 'Permeability relative to new membranes; calibrate it against plant pressure.' },
      { key: 'kSP', label: 'Salt-passage multiplier', unit: '×', value: 1, min: 0.3, max: 5, help: 'Scales ion passage of every membrane pass; rises as membranes age.' },
      { key: 'vleModel', label: 'Vapour–liquid equilibrium model', type: 'select', value: 'pr', options: [{ value: 'pr', label: 'Activity coefficients + Peng–Robinson equation of state' }, { value: 'gamma', label: 'Modified Raoult’s law (activity coefficients, ideal gas)' }, { value: 'ideal', label: 'Raoult’s law + Henry’s law (ideal)' }], help: 'Used for the dissolved-gas equilibrium of the feed and the vacuum-deaerator flash. All three are tabulated side by side.' },
      { key: 'compModel', label: 'Compressor model', type: 'select', value: 'isentropic', options: [{ value: 'isentropic', label: 'Isentropic efficiency' }, { value: 'polytropic', label: 'Polytropic efficiency' }], help: 'Applies to the vapour compressors of the concentrator and crystalliser and to the deaerator vacuum pump; the efficiency inputs are read accordingly.' },
      { key: 'surrogate', label: 'Train a response-surface surrogate of the flowsheet', type: 'bool', value: true, help: 'Quadratic surface of energy, pressure and product TDS against feed temperature and salinity from 9 flowsheet solutions, checked on held-out cases and used for the fast off-design maps.' },
      { key: 'optimise', label: 'Optimise the RO recovery', type: 'select', value: 'none', options: [{ value: 'none', label: 'No' }, { value: 'sec', label: 'Minimise specific electricity' }, { value: 'cost', label: 'Minimise the indicative cost of water' }], help: 'Golden-section search that re-designs the flowsheet at every trial recovery (about a dozen extra flowsheet designs).' },
    ] },
    { group: 'Digital twin: data reconciliation', tab: 'setup', help: 'Weighted least-squares reconciliation of flow and conductivity measurements with the water and salt balances of the RO pass, the plant, the product line and the waste line; flags instruments whose reading the balances contradict.', fields: [
      { key: 'twin', label: 'Measurements', type: 'select', value: 'synthetic', options: [{ value: 'synthetic', label: 'Synthetic test data (model + noise + one biased meter)' }, { value: 'table', label: 'Measurement table' }, { value: 'off', label: 'Off' }] },
      { key: 'twinSigQ', label: 'Flow-meter standard uncertainty', unit: '% of reading', value: 1, min: 0.1, max: 10, showIf: (v) => v.twin !== 'off' },
      { key: 'twinSigC', label: 'Conductivity standard uncertainty', unit: '% of reading', value: 2, min: 0.1, max: 15, showIf: (v) => v.twin !== 'off' },
      { key: 'twinTable', label: 'Plant measurements', type: 'table', columns: [{ key: 'point', label: 'Measuring point', type: 'text' }, { key: 'Q', label: 'Flow', unit: 'm³/h' }, { key: 'EC', label: 'Conductivity at 25 °C', unit: 'µS/cm' }],
        value: [{ point: 'Raw intake', Q: 1001, EC: 53170 }, { point: 'Membrane feed', Q: 933, EC: 55510 }, { point: 'First-pass permeate', Q: 423.4, EC: 461 }, { point: 'RO concentrate', Q: 551, EC: 91840 }, { point: 'Product', Q: 420.7, EC: 562 }, { point: 'Discharge', Q: 579.7, EC: 87720 }],
        showIf: (v) => v.twin === 'table', help: 'Keep the six rows in this order. Leave a cell empty if the point is not measured: it is then estimated from the balances. The default readings belong to the default seawater plant and contain a faulty concentrate flow meter.' },
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
    { group: 'Power supply and grid limit (dynamic)', tab: 'setup', showIf: (v) => v.dynamic, help: 'Hourly PV and wind generation, a battery and a limit on grid import are dispatched in the daily simulation; production is shed when power is short and the product tank bridges the gap.', fields: [
      { key: 'renew', label: 'Renewable supply', type: 'select', value: 'none', options: [{ value: 'none', label: 'None (grid only)' }, { value: 'pv', label: 'Photovoltaic' }, { value: 'wind', label: 'Wind' }, { value: 'pvwind', label: 'Photovoltaic + wind' }] },
      { key: 'pvKW', label: 'PV capacity', unit: 'kWp', value: 2000, min: 0, max: 1e6, showIf: (v) => v.renew === 'pv' || v.renew === 'pvwind' },
      { key: 'ghi', label: 'Daily global irradiation', unit: 'kWh/m²·d', value: 5.5, min: 0.5, max: 9, showIf: (v) => v.renew === 'pv' || v.renew === 'pvwind' },
      { key: 'windKW', label: 'Wind capacity', unit: 'kW', value: 1500, min: 0, max: 1e6, showIf: (v) => v.renew === 'wind' || v.renew === 'pvwind' },
      { key: 'windMean', label: 'Mean wind speed at hub height', unit: 'm/s', value: 6.5, min: 1, max: 15, showIf: (v) => v.renew === 'wind' || v.renew === 'pvwind' },
      { key: 'battKWh', label: 'Battery capacity', unit: 'kWh', value: 2000, min: 0, max: 1e6, showIf: (v) => v.renew !== 'none', help: 'Starts half full; charges and discharges at up to half its capacity per hour.' },
      { key: 'gridLimit', label: 'Grid import', type: 'select', value: 'none', options: [{ value: 'none', label: 'Unlimited' }, { value: 'cap', label: 'Limited to a maximum power' }, { value: 'island', label: 'None (island operation)' }] },
      { key: 'gridMax', label: 'Maximum grid import', unit: 'kW', value: 800, min: 0, max: 1e6, showIf: (v) => v.gridLimit === 'cap' },
    ] },
    { group: 'Tank heat and product pH loop (dynamic)', tab: 'setup', showIf: (v) => v.dynamic, help: 'Initial stored thermal energy of the product tank with heat loss to ambient, and a second control loop: PID caustic trim on the product pH.', fields: [
      { key: 'initTankT', label: 'Initial tank temperature', unit: '°C', value: 26, min: 1, max: 60, help: 'Sets the initial stored thermal energy ρ·c·V·T of the tank inventory.' },
      { key: 'tankU', label: 'Tank heat-loss coefficient', unit: 'W/m²·K', value: 4, min: 0, max: 30 },
      { key: 'airTemp', label: 'Mean air temperature', unit: '°C', value: 28, min: -20, max: 50 },
      { key: 'airSwing', label: 'Daily air-temperature swing (±)', unit: 'K', value: 6, min: 0, max: 20 },
      { key: 'ctrl2', label: 'Second control loop', type: 'select', value: 'ph', options: [{ value: 'ph', label: 'Product pH by caustic trim (PID)' }, { value: 'none', label: 'None' }] },
      { key: 'phKc', label: 'pH controller gain', unit: '× inverse process gain', value: 0.8, min: 0.05, max: 5, showIf: (v) => v.ctrl2 === 'ph', help: 'Relative to the slope of the titration curve at the design point, so 1 is a unit loop gain.' },
      { key: 'phTi', label: 'Integral time', unit: 'min', value: 8, min: 0.5, max: 120, showIf: (v) => v.ctrl2 === 'ph' },
      { key: 'phTd', label: 'Derivative time', unit: 'min', value: 0.5, min: 0, max: 10, showIf: (v) => v.ctrl2 === 'ph' },
      { key: 'phMaxDose', label: 'Maximum caustic trim', unit: 'mg/L NaOH', value: 1, min: 0.01, max: 50, showIf: (v) => v.ctrl2 === 'ph' },
      { key: 'phDist', label: 'Dissolved-CO₂ disturbance', unit: '± %', value: 40, min: 0, max: 90, showIf: (v) => v.ctrl2 === 'ph' },
      { key: 'phAW', label: 'Anti-windup (back-calculation)', type: 'bool', value: true, showIf: (v) => v.ctrl2 === 'ph' },
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
    if (d.ghiDaily > 0) P.push({ key: 'ghi', value: clamp(d.ghiDaily, 0.5, 9), from: 'Site daily global irradiation' });
    if (d.windSpeed > 0) P.push({ key: 'windMean', value: clamp(d.windSpeed, 1, 15), from: 'Site wind speed' });
    if (Number.isFinite(d.airTemp)) P.push({ key: 'airTemp', value: clamp(d.airTemp, -20, 50), from: 'Site air temperature' });
    if (d.lendingRate > 0) P.push({ key: 'discount', value: clamp(d.lendingRate, 0, 25), from: 'Site lending rate' });
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
    let dyn = null, dynAlt = [], aux = null, prof = null, base0 = null, tankUA = 0, phSP = 7;
    if (v.dynamic && Qp > 0) {
      const loads = [v.minLoad / 100, 0.5 * (v.minLoad / 100 + 1), 1], secs = [cases[6], cases[7], cases[0]].map((c) => c.q.secElec + (c.q.secThermal > 0 ? 0 : 0)), secOf = (u) => interp1(loads, secs, clamp(u, loads[0], 1));
      const tab = (v.demand || []).filter((r) => Number.isFinite(+r.h) && Number.isFinite(+r.f)).map((r) => [clamp(+r.h, 0, 24), Math.max(0, +r.f)]).sort((a, b) => a[0] - b[0]);
      const hs = tab.length ? [tab[tab.length - 1][0] - 24, ...tab.map((r) => r[0]), tab[0][0] + 24] : [0, 24], fsr = tab.length ? [tab[tab.length - 1][1], ...tab.map((r) => r[1]), tab[0][1]] : [100, 100];
      const grid = linspace(0, 24, 97), meanF = sum(grid.slice(0, 96).map((h) => interp1(hs, fsr, h))) / 96 || 100, uff = v.demandPct / 100;
      const demand = (t) => (uff * Qp * interp1(hs, fsr, ((t % 24) + 24) % 24)) / meanF, inPeak = (t) => { const h = ((t % 24) + 24) % 24; return v.peakStart <= v.peakEnd ? h >= v.peakStart && h < v.peakEnd : h >= v.peakStart || h < v.peakEnd; };
      const base = { Qn: Qp, Vmax: Math.max(1, Qp * v.tankHours), L0: v.initLevel / 100, hours: 72, dtMin: v.dtMin, mode: v.ctrlMode, Kp: v.Kp, Ki: v.Ki, uMin: v.minLoad / 100, uff, nTrains: Math.round(v.nTrains), onLow: v.onLow / 100, offHigh: v.offHigh / 100, demand, sec: secOf, price: (t) => (inPeak(t) ? v.peakPrice : v.offPrice), Lsp: () => v.levelSP / 100 };
      if (v.renew !== 'none' || v.gridLimit !== 'none') { // renewable supply, battery and grid-import limit
        const Emax = v.renew !== 'none' ? v.battKWh : 0;
        prof = v.renew !== 'none' ? resourceProfile({ ghi: v.ghi, wind: v.windMean, pvKW: v.renew === 'wind' ? 0 : v.pvKW, windKW: v.renew === 'pv' ? 0 : v.windKW }) : null;
        base.supply = { pv: prof ? prof.pvAt : () => 0, wind: prof ? prof.windAt : () => 0, Emax, E0: 0.5 * Emax, Pb: Emax / 2, eta: Math.sqrt(0.9), gridMax: v.gridLimit === 'cap' ? v.gridMax : v.gridLimit === 'island' ? 0 : 1e12 };
      }
      base0 = base;
      dyn = dynamicSim(base); dyn.inPeak = inPeak; dyn.Vmax = base.Vmax;
      { // thermal inventory of the tank and the pH trim loop follow the flows of the level simulation
        const sub = Math.max(1, Math.ceil(dyn.dt * 60 - 1e-9)), cP = (1000 * product.co2) / 44.01 / Qp, aP = pIons.HCO3 / 61.017, pK = pK1(product.S), Dt = ((4 * base.Vmax) / Math.PI) ** (1 / 3);
        const phOn = v.ctrl2 === 'ph' && cP > 1e-9 && aP > 1e-9, gain = phOn ? (1 / Math.LN10) * (1 / aP + 1.41 / cP) : 1;
        const s0 = phOn ? (aP * cP * (10 ** 0.15 - 1)) / (cP + 10 ** 0.15 * aP) : 0; // caustic that lifts the pH by 0.15 at design: (A + s)/(c − s) = 10^0.15·A/c
        tankUA = (v.tankU * 1.5 * Math.PI * Dt * Dt) / 1000; phSP = phOn ? clamp(pK + Math.log10(aP / cP) + 0.15, 3.5, 10.8) : product.pH;
        aux = auxDynamics({ t: dyn.t, qIn: dyn.u.map((u) => (u / 100) * Qp), qOut: dyn.Qd, Vmax: base.Vmax, V0: base.L0 * base.Vmax, hours: 72, dt: dyn.dt / sub, Tp: product.T, T0: v.initTankT, Ta: v.airTemp, swing: v.airSwing, UA: tankUA,
          ph: phOn ? { pK, sp: phSP, s0, A0: aP, c0: cP, Vc: Qp / 6, tauM: 1.5 / 60, Kc: Math.max(0.01, v.phKc) / gain, Ti: Math.max(0.2, v.phTi) / 60, Td: v.phTd / 60, Tt: Math.max(0.1, v.phTi) / 120, sMax: v.phMaxDose / 40, aw: !!v.phAW, dist: (t) => { const h = ((t % 24) + 24) % 24; return h >= 8 && h < 12 ? v.phDist / 100 : h >= 16 && h < 20 ? -v.phDist / 100 : 0; } } : null });
      }
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
    const med = res.get('MED') || res.get('MSF');
    if (med && med.rec.dTcw > v.medMaxDT) W.push({ level: 'warn', msg: `${med.u.id} cooling water warms by ${fmt(med.rec.dTcw, 3)} K (limit ${v.medMaxDT} K) — send more seawater to the condenser or lower the make-up share.` });
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
    const groups = { 'Intake and low-pressure pumping': ['IP', 'TP', 'PP'], 'High-pressure pumping': ['HPP', 'NFP'], 'Circulation and inter-stage boosting': ['BST', 'RO1', 'NF'], 'Second pass': ['P2P', 'RO2'], 'Thermal desalination auxiliaries': ['MED', 'MSF'], 'Brine concentrator and crystalliser': ['BC', 'CR'], 'Deaerator vacuum and extraction pumps': ['DEA'], 'Electrodialysis polishing': ['ED'], 'Membrane distillation auxiliaries': ['MD'], 'Recovered by turbine': ['TRB'] };
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
      if (q.u.type === 'msf') sizing.push([q.u.name, 'Heat-transfer area (estimate)', k.area, 'm²', `${k.stages} stages of ${fmt(k.dTstage, 3)} K, recirculating brine ${fmt(k.Mr / 1000, 4)} t/h, gained output ratio ${fmt(k.GOR, 3)}, steam ${fmt(k.steam, 3)} kg/s`]);
      if (q.u.type === 'ed') sizing.push([q.u.name, 'Cell-pair area', k.area || 0, 'm²', k.area ? `${Math.ceil(k.cellPairs)} cell pairs of 0.6 m², ${fmt(k.i, 3)} A/m² (${fmt((100 * k.i) / k.ilim, 3)} % of the limiting current density), ${fmt(k.U, 3)} V per pair, ${fmt(k.sec, 3)} kWh/m³` : 'not in service']);
      if (q.u.type === 'md') sizing.push([q.u.name, 'Membrane area', k.area || 0, 'm²', k.area ? `flux ${fmt(k.flux, 3)} kg/m²·h, thermal efficiency ${fmt(100 * k.eta, 3)} %, gained output ratio ${fmt(k.GOR, 3)}, membrane surfaces at ${fmt(k.Tfm, 3)} / ${fmt(k.Tpm, 3)} °C` : 'not in service']);
      if (q.u.type === 'deaer' && k.comp) sizing.push([q.u.name, 'Vacuum-pump shaft power', k.shaftV, 'kW', `${fmt(k.ventMol / 1000, 3)} kmol/h of vent from ${fmt(k.Pv, 3)} bar a in ${k.comp.stages} stage(s) of ratio ${fmt(k.comp.stageRatio, 3)}, ${v.compModel} efficiency ${fmt(v.etaVac, 3)} %, exponent ${fmt(k.comp.n, 4)}; extraction pump ${fmt(k.hyd, 3)} kW hydraulic`]);
      if (q.u.type === 'remin' && k.ebct) sizing.push([q.u.name, 'Calcite bed volume', k.bed, 'm³', `first-order dissolution k = ${fmt(k.kDiss, 3)} 1/min, ${fmt(100 * k.approach, 3)} % approach to the CO₂-limited equilibrium → ${fmt(k.ebct, 3)} min empty-bed contact time`]);
      if (q.u.type === 'tank') sizing.push([q.u.name, 'Volume', k.volume, 'm³', `${v.tankHours} h of production`]);
      for (const [n, x] of Object.entries(k.chem || {})) if (x > 0) sizing.push([`${n} storage (${q.u.name})`, '14-day stock', (14 * 24 * x) / 1000, 't', `${fmt(24 * x, 3)} kg/d`]);
    }
    const quality = [['TDS (mg/L)', res.raw.tds, product.tds, brine.tds], ['pH', res.raw.pH, product.pH, brine.pH], ['Temperature (°C)', res.raw.T, product.T, brine.T], ['Hardness (mg/L as CaCO₃)', hardness(ionsOf(res.raw)), hardness(pIons), hardness(ionsOf(brine))], ['Alkalinity (mg/L as CaCO₃)', alkalinity(ionsOf(res.raw)), alkalinity(pIons), alkalinity(ionsOf(brine))],
      ['Langelier index', langelier(ionsOf(res.raw), res.raw.T, res.raw.pH), lsi, langelier(ionsOf(brine), brine.T, brine.pH)], ['Boron (mg/L)', ionsOf(res.raw).B, pIons.B, ionsOf(brine).B], ['Chloride (mg/L)', ionsOf(res.raw).Cl, pIons.Cl, ionsOf(brine).Cl], ['Flow (m³/h)', res.raw.Q, product.Q, brine.Q]];
    const utilRows = [...pw.map(([n, x]) => [`Electricity — ${n}`, x, 'kW']), ['Electricity — total', res.power, 'kW'], ['Heating steam duty', res.heat, 'kW'], ['Heating steam flow', med ? med.rec.steam * 3.6 : res.heat > 0 ? (res.heat / (latentHeat(120) / 1000)) * 3.6 : 0, 't/h'], ['Cooling and heat-loss duty', res.cool, 'kW'],
      ['Cooling seawater (MED reject)', S.medCW ? S.medCW.Q : S.msfCW ? S.msfCW.Q : 0, 'm³/h'], ...Object.entries(chemKgD).map(([n, x]) => [`Chemical — ${n}`, x, 'kg/d']), ['Chemicals — total', chemTot, 'kg/d'], ['Sludge (dry solids)', sludgeKgD, 'kg/d'], ['Salt cake', solidsTd, 't/d'], ...Object.entries(salts).map(([n, x]) => [`Salt — ${n}`, (24 * x) / 1000, 't/d']),
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
    // ---- second law: entropy balance of every unit
    const xK = [], xP = [], xT = [], xR = [], xO = {}, T0k = dead.T + KELVIN;
    xK.push({ label: 'Entropy generated', value: res.sGen, unit: 'kW/K', help: `T₀·S_gen = ${fmt(T0k * res.sGen, 4)} kW, equal to the exergy destroyed (Gouy–Stodola)` });
    xT.push({ title: 'Entropy balance by unit (second law)', columns: ['Tag', 'Unit', 'Entropy in (kW/K)', 'Entropy out (kW/K)', 'Entropy with heat supplied (kW/K)', 'Heat to surroundings (kW)', 'Entropy generated (kW/K)', 'T₀ × entropy generated (kW)', 'Exergy destroyed (kW)'],
      rows: [...recs.map((q) => [q.u.id, q.u.name, q.sIn, q.sOut, q.sQ, q.q0, q.sGen, T0k * q.sGen, q.exDest]), ['–', 'Plant total', null, null, sum(recs.map((q) => q.sQ)), sum(recs.map((q) => q.q0)), res.sGen, T0k * res.sGen, res.exDest]],
      note: `Stream entropy is relative to the dead state (${fmt(dead.T, 3)} °C): c·ln(T/T₀) for the incompressible liquid plus the mixing term used for the salinity exergy. S_gen = ΣS_out − ΣS_in − Q/T_source + Q₀/T₀, where Q₀ is the heat that leaves to the surroundings (cooling duty, motor and friction losses and the heat-of-mixing residual of the property model). T₀·S_gen must equal the exergy destroyed.` });
    // ---- phase equilibrium of the dissolved gases and electrolyte state
    const vOpt = { model: v.vleModel, sat: v.airSat / 100 }, feedV = streamFlash(res.raw, P0, vOpt), dea = res.get('DEA'), vS = dea ? dea.inS[0] : res.raw, vP = dea ? dea.rec.Pv : P0, vle = dea ? dea.rec.vle : feedV;
    const vNames = { ideal: 'Raoult’s law + Henry’s law, ideal gas', gamma: 'Modified Raoult’s law (activity coefficients), ideal gas', pr: 'Activity coefficients + Peng–Robinson fugacity coefficients' }, vAll = Object.keys(vNames).map((m) => (m === vle.model ? vle : streamFlash(dea ? { ...vS, T: vle.T } : vS, vP, { ...vOpt, model: m, Tair: vS.T })));
    const rem = (f, k) => (f.n[k] > 0 ? (100 * f.V[k]) / f.n[k] : 0), o2mg = res.raw.Q > 0 ? (feedV.n.O2 * GAS.O2.mw) / res.raw.Q : 0;
    xK.push({ label: 'Feed bubble pressure', value: feedV.Pbub, unit: 'bar a', status: feedV.Pbub > 1.02 * P0 ? 'warn' : 'ok', help: `Total pressure of water vapour and dissolved gases; free-gas mole fraction of the feed at 1 atm ${fmt(feedV.beta, 3)}, dissolved oxygen ${fmt(o2mg, 3)} mg/L` });
    if (dea) xK.push({ label: 'Deaerator CO₂ removal', value: 100 * dea.rec.removal.CO2, unit: '%', help: `Single adiabatic equilibrium flash at ${fmt(vP, 3)} bar a and ${fmt(vle.T, 3)} °C; vapour mole fraction ${fmt(vle.beta, 3)}` }, { label: 'Deaerator O₂ removal', value: 100 * dea.rec.removal.O2, unit: '%' });
    if (dea && dea.rec.ventWater > 1e-3 * dea.inS[0].m) W.push({ level: 'warn', msg: `The deaerator pressure (${fmt(vP, 3)} bar a) is below the vapour pressure of the feed: ${fmt(dea.rec.ventWater / 1000, 3)} t/h of water boils off, the feed cools to ${fmt(dea.rec.Tf, 3)} °C and the vacuum pump draws ${fmt(dea.rec.shaftV, 3)} kW. Raise the pressure above ${fmt((1.1 * vle.aw * psat(dea.inS[0].T)) / 1e5, 2)} bar a.` });
    if (feedV.Pbub > 1.02 * P0) W.push({ level: 'warn', msg: `The feed is supersaturated with dissolved gas (bubble pressure ${fmt(feedV.Pbub, 4)} bar a at ${fmt(v.airSat, 3)} % air saturation): a free-gas fraction of ${fmt(feedV.beta, 2)} (mole basis) forms at atmospheric pressure — vent the intake and expect gas binding in filters.` });
    xT.push({ title: `Vapour–liquid equilibrium — ${dea ? `vacuum deaerator at ${fmt(vP, 3)} bar a` : 'raw feed at its bubble point'}`, columns: ['Component', 'Feed mole fraction z', 'Liquid x', 'Vapour y', 'K = y/x', 'Liquid activity coefficient γ', 'Vapour fugacity coefficient φ', 'Liquid fugacity (bar)', 'Vapour fugacity (bar)'],
      rows: [...VOL.map((k, j) => [GAS[k].name, vle.x[j] * (1 + vle.beta * (vle.K[j] - 1)), vle.x[j], vle.y[j], vle.K[j], vle.gamma[j], vle.phi[j], vle.fL[j], vle.fV[j]]), ['Dissolved ions (non-volatile)', vle.x[4] * (1 - vle.beta), vle.x[4], 0, 0, null, null, null, null]],
      note: `${vNames[vle.model]} at ${fmt(vle.T, 4)} °C and ${fmt(vle.Peq, 4)} bar a${dea ? ' (adiabatic flash: the temperature follows from the energy balance)' : ''}. Vapour fraction from the Rachford–Rice equation: ${fmt(vle.beta, 3)} (mole basis); bubble pressure ${fmt(vle.Pbub, 4)} bar a; water activity ${fmt(vle.aw, 5)}; vapour compressibility ${fmt(vle.Z, 5)}. Equilibrium means equal fugacity of every component in both phases: x·γ·f° = y·φ·P, with f° = saturation pressure (water) or Henry constant (gases).${vle.beta > 0 ? '' : ' The stream is single-phase at this pressure, so K, y and the fugacities are those of the first bubble.'} Nitrogen and oxygen are trace species held at ${fmt(v.airSat, 3)} % of air saturation; like suspended solids they are tracked outside the stream mass flow.` });
    xT.push({ title: 'Phase-equilibrium models compared', columns: ['Model', 'Bubble pressure (bar a)', 'Vapour fraction (mol/mol)', 'CO₂ to vapour (%)', 'O₂ to vapour (%)', 'N₂ to vapour (%)', 'K of water', 'γ of water', 'φ of water vapour', 'φ of CO₂', 'Vapour mole fraction of water'],
      rows: vAll.map((f) => [vNames[f.model] + (f.model === vle.model ? ' (selected)' : ''), f.Pbub, f.beta, rem(f, 'CO2'), rem(f, 'O2'), rem(f, 'N2'), f.K[0], f.gamma[0], f.phi[0], f.phi[1], f.y[0]]), note: `Same stream (${dea ? 'deaerator inlet' : 'raw feed'}) at ${fmt(vle.T, 4)} °C and ${fmt(vP, 4)} bar a under the three formulations. Salt lowers the vapour pressure of water (activity ${fmt(vle.aw, 4)}) and the solubility of the gases (Setschenow salting-out).` });
    const el = [res.raw, product, brine].map((s) => (s.m > 0 ? electrolyte(s) : null)), elC = S.conc && S.conc.m > 0 ? electrolyte(S.conc) : null, elRow = (name, key) => [name, ...el.map((e) => (e ? e[key] : null))];
    quality.push(elRow('Ionic strength (mol/kg)', 'I'), elRow('Water activity', 'aw'), elRow('Mean activity coefficient of NaCl', 'gNaCl'), elRow('Free sulphate after ion pairing (fraction)', 'fSO4'), elRow('Calcite saturation index (ion activities)', 'siCalcite'), elRow('Gypsum saturation index (ion activities)', 'siGypsum'));
    if (elC && elC.siGypsum > 0.35) W.push({ level: 'warn', msg: `The RO concentrate is supersaturated in gypsum by the activity model (saturation index ${fmt(elC.siGypsum, 2)}) — check the antiscalant limit in the brine-chemistry suite or lower the recovery.` });
    // ---- indicative economics
    const cost = plantCost(res, v), capShare = cost.annual > 0 ? cost.annCap / cost.annual : 0, costItems = [['Capital recovery', cost.annCap], ...Object.entries(cost.opex)].filter((x) => x[1] > 0);
    xK.push({ label: 'Indicative cost of water', value: cost.lcow, unit: '$/m³', help: `Indicative only: installed capital ${fmt(cost.capex / 1e6, 3)} M$ (${fmt(cost.capexPerM3d, 3)} $ per m³/d), capital share ${fmt(capShare, 3)} $/m³` });
    xT.push({ title: 'Indicative cost roll-up by unit', columns: ['Tag', 'Unit', 'Cost function', 'Equipment (k$)', 'Installed (k$)', 'Share of capital (%)', 'Capital charge ($/m³)'],
      rows: [...cost.rows.map((r) => [r.id, r.name, r.basis, r.cost / 1000, (r.cost * v.capexFactor) / 1000, cost.equip > 0 ? (100 * r.cost) / cost.equip : 0, cost.annual > 0 ? (cost.crf * r.cost * v.capexFactor) / cost.annual : 0]), ['–', 'Total', `installation factor ${fmt(v.capexFactor, 3)}`, cost.equip / 1000, cost.capex / 1000, 100, capShare]],
      note: `Indicative order-of-magnitude cost functions, not a quotation. Capital-recovery factor ${fmt(cost.crf, 4)} (${fmt(v.discount, 3)} % over ${fmt(v.plantLife, 3)} years), ${fmt(v.availability, 3)} % availability, ${fmt(cost.annual / 1e6, 4)} million m³/y. Use the Economics suite for a bankable estimate.` });
    xT.push({ title: 'Indicative cost of water', columns: ['Item', 'Annual cost (k$/y)', 'Cost of water ($/m³)', 'Share (%)'], rows: [...costItems.map(([n, x]) => [n, x / 1000, cost.annual > 0 ? x / cost.annual : 0, (100 * x) / (cost.annCap + cost.opexTot)]), ['Total', (cost.annCap + cost.opexTot) / 1000, cost.lcow, 100]], note: `Average electricity price ${fmt(cost.price, 3)} $/kWh from the tariff; heat ${fmt(v.heatPrice, 3)} $/kWh.` });
    xP.push({ type: 'bar', title: 'Indicative cost of water by item', ylabel: '$/m³', categories: costItems.map((x) => x[0]), series: [{ name: '$/m³', values: costItems.map((x) => (cost.annual > 0 ? x[1] / cost.annual : 0)) }] });
    // ---- environmental accounting and reactions
    const chemUp = 24 * sum(recs.filter((q) => q.u.row !== ROWS[3]).flatMap((q) => Object.values(q.rec.chem || {}))), saltLoad = (24 * brine.salt) / 1000, thermalLoad = (brine.m * cpOf(brine) * (brine.T - res.raw.T)) / 3.6e6;
    xT.push({ title: 'Environmental accounting', columns: ['Item', 'Value', 'Unit', 'Basis'], rows: [
      ['CO₂ from electricity', (24 * res.power * v.gridCarbon) / 1000, 't/d', `${fmt(v.gridCarbon, 3)} kg/kWh`], ['CO₂ from heat', (24 * res.heat * v.heatCarbon) / 1000, 't/d', `${fmt(v.heatCarbon, 3)} kg/kWh`], ['CO₂ per m³ of product', Qp > 0 ? co2 / Qp : 0, 'kg/m³', 'Electricity and heat'],
      ['CO₂ stripped to the deaerator vent', 24 * (dea?.rec.ventCO2 || 0), 'kg/d', 'Dissolved CO₂ released by the flash'], ['Salt load of the discharge', saltLoad, 't/d', 'Dissolved salts in the liquid discharge'], ['Discharge salinity', brine.S, 'g/kg', ''],
      ['Salinity excess over the intake', brine.S - res.raw.S, 'g/kg', `${fmt(res.raw.S > 0 ? (100 * (brine.S - res.raw.S)) / res.raw.S : 0, 3)} % above ambient`], ['Density excess of the discharge', brine.rho - res.raw.rho, 'kg/m³', 'Negative buoyancy of the plume'], ['Temperature excess of the discharge', brine.T - res.raw.T, 'K', ''],
      ['Thermal load of the discharge', thermalLoad, 'kW', 'ṁ·c_p·ΔT'], ['Chemicals dosed upstream of the discharge', chemUp, 'kg/d', 'Pretreatment and membrane conditioning chemicals'], ['Antiscalant discharged', 24 * (res.chem.Antiscalant || 0), 'kg/d', 'Leaves with the concentrate'],
      ['Sludge (dry solids)', sludgeKgD, 'kg/d', 'Backwash and flotation waste'], ['Salt cake', solidsTd, 't/d', 'Crystalliser'], ['Intake per m³ of product', Qp > 0 ? res.raw.Q / Qp : 0, 'm³/m³', '']] });
    const rxn = recs.filter((q) => q.rec.gen && sum(Object.values(q.rec.chem || {})) > 0);
    if (rxn.length) xT.push({ title: 'Reactors: dosing reactions and extents', columns: ['Tag', 'Unit', 'Reagents (kg/h)', 'HCO₃⁻ formed (kmol/h)', 'CO₃²⁻ formed (kmol/h)', 'Dissolved CO₂ change (kmol/h)', 'Iron precipitated (kg/h)', 'pH in', 'pH out'],
      rows: rxn.map((q) => [q.u.id, q.u.name, Object.entries(q.rec.chem).filter(([, x]) => x > 0).map(([n, x]) => `${n} ${fmt(x, 3)}`).join('; '), q.rec.gen[IX.HCO3] / 61.017, q.rec.gen[IX.CO3] / 60.009, (q.outS[0].co2 - q.inS[0].co2) / 44.01, -q.rec.gen[IX.Fe], q.inS[0].pH, q.outS[0].pH]),
      note: 'Stoichiometric reactors: FeCl₃ + 3 HCO₃⁻ → Fe(OH)₃ + 3 CO₂ + 3 Cl⁻; H₂SO₄ + 2 HCO₃⁻ → 2 CO₂ + SO₄²⁻; Ca(OH)₂ + 2 CO₂ → Ca²⁺ + 2 HCO₃⁻; CaCO₃ + CO₂ + H₂O → Ca²⁺ + 2 HCO₃⁻; NaOH + CO₂ → Na⁺ + HCO₃⁻. Every ion balance includes these generation terms.' });
    // ---- response-surface surrogate of the flowsheet over feed temperature and salinity
    let sur = null;
    if (v.surrogate && Qp > 0) {
      const a0 = Math.min(v.Tmin, v.T), a1 = Math.max(v.Tmax, v.T), Tlo = a1 - a0 < 1 ? a0 - 1 : a0, Thi = a1 - a0 < 1 ? a1 + 1 : a1, b0 = Math.min(1, v.salMin / 100), b1 = Math.max(1, v.salMax / 100), slo = b1 - b0 < 0.01 ? b0 - 0.01 : b0, shi = b1 - b0 < 0.01 ? b1 + 0.01 : b1;
      const nrm = (T, sf) => [(2 * (T - Tlo)) / (Thi - Tlo) - 1, (2 * (sf - slo)) / (shi - slo) - 1], den = (a, b) => ({ T: Tlo + ((a + 1) / 2) * (Thi - Tlo), sf: slo + ((b + 1) / 2) * (shi - slo) });
      const tg = [['Specific electricity', 'kWh/m³', (q) => q.secElec], ['RO feed pressure', 'bar', (q) => q.ro1.Preq], ['Product TDS', 'mg/L', (q) => q.product.tds], ['Electrical power', 'kW', (q) => q.power]];
      const train = [-1, 0, 1].flatMap((a) => [-1, 0, 1].map((b) => { const c = den(a, b); return { x: [a, b], q: solveCase(v, ref, c, C) }; }));
      const test = [...cases.slice(1, 6).map((c) => ({ x: nrm(c.q.cond.T, c.q.cond.sf), q: c.q })), ...[[0.5, -0.5]].map((x) => ({ x, q: solveCase(v, ref, den(x[0], x[1]), C) }))];
      const fits = tg.map(([name, unit, get]) => { const f = fitSurface(train.map((p) => p.x), train.map((p) => get(p.q))), meas = test.map((p) => get(p.q)), pred = test.map((p) => f.predict(p.x)), m = metrics(meas, pred), scale = Math.max(1e-12, Math.abs(get(res)));
        return { name, unit, f, meas, pred, rmse: m.rmse, r2: Number.isFinite(m.r2) ? m.r2 : 1, maxErr: 100 * Math.max(...meas.map((y, i) => Math.abs(pred[i] - y) / Math.max(1e-12, Math.abs(y)))), scale }; });
      const gx = linspace(Tlo, Thi, 21), gy = linspace(100 * slo, 100 * shi, 17), map = (f) => gy.map((s) => gx.map((T) => f.predict(nrm(T, s / 100))));
      sur = { fits, maxErr: Math.max(...fits.map((f) => f.maxErr)), Tlo, Thi, slo, shi };
      xK.push({ label: 'Surrogate held-out error', value: sur.maxErr, unit: '%', status: sur.maxErr > 3 ? 'warn' : 'ok', help: `Largest relative error of the response surface on ${test.length} flowsheet cases not used for training` });
      xP.push({ type: 'field', title: 'Fast off-design map (surrogate): specific electricity', xlabel: 'Feed temperature (°C)', ylabel: 'Salinity (% of design)', zlabel: 'Specific electricity', zunit: 'kWh/m³', x: gx, y: gy, z: map(fits[0].f), cmap: 'viridis', contours: 8,
        markers: [...train.map((p) => { const c = den(p.x[0], p.x[1]); return { x: c.T, y: 100 * c.sf, label: '' }; }), { x: v.T, y: 100, label: 'design' }], note: 'Quadratic response surface trained on 9 flowsheet solutions (markers); equipment sizes frozen at the design case.' });
      xP.push({ type: 'field', title: 'Fast off-design map (surrogate): RO feed pressure', xlabel: 'Feed temperature (°C)', ylabel: 'Salinity (% of design)', zlabel: 'RO feed pressure', zunit: 'bar', x: gx, y: gy, z: map(fits[1].f), cmap: 'thermal', contours: 8, markers: [{ x: v.T, y: 100, label: 'design' }] });
      xP.push({ type: 'line', title: 'Surrogate parity on held-out flowsheet cases', xlabel: 'Flowsheet value ÷ design value', ylabel: 'Surrogate value ÷ design value', series: [...fits.map((f) => ({ name: f.name, x: f.meas.map((y) => y / f.scale), y: f.pred.map((y) => y / f.scale), mode: 'points' })), (() => { const all = fits.flatMap((f) => f.meas.map((y) => y / f.scale)), lo = Math.min(...all), hi = Math.max(...all); return { name: '1 : 1', x: [lo, hi], y: [lo, hi], dash: true }; })()] });
      xT.push({ title: 'Mechanistic–surrogate model: response surface of the flowsheet', columns: ['Target', 'Unit', 'b₀', 'b_T', 'b_S', 'b_TT', 'b_SS', 'b_TS', 'Held-out RMSE', 'Held-out largest error (%)', 'Held-out R²'], rows: fits.map((f) => [f.name, f.unit, ...f.f.coef, f.rmse, f.maxErr, f.r2]),
        note: `y = b₀ + b_T·t + b_S·s + b_TT·t² + b_SS·s² + b_TS·t·s with t and s scaled to −1…1 over ${fmt(Tlo, 3)}–${fmt(Thi, 3)} °C and ${fmt(100 * slo, 4)}–${fmt(100 * shi, 4)} % of design salinity. Trained on a 3 × 3 grid of flowsheet solutions, tested on ${test.length} other solutions (the off-design cases and one interior point).` });
      if (sur.maxErr > 3) W.push({ level: 'warn', msg: `The response-surface surrogate misses held-out flowsheet cases by up to ${fmt(sur.maxErr, 3)} % — the plant changes regime inside the off-design envelope (a limit, an idle unit or boiling in the deaerator), so use the off-design table instead of the fast map.` });
      xO.surrogate = { temperature: [Tlo, Thi], salinityPct: [100 * slo, 100 * shi], secCoef: fits[0].f.coef, pressureCoef: fits[1].f.coef, tdsCoef: fits[2].f.coef, powerCoef: fits[3].f.coef, maxErrorPct: sur.maxErr };
    }
    ctx?.progress?.(0.85, 'Surrogate and economics evaluated');
    // ---- digital twin: reconcile plant measurements with the mass and salt balances
    if (v.twin !== 'off' && S.roF?.Q > 1e-6 && S.perm1?.Q > 1e-6 && S.conc?.Q > 1e-6 && Qp > 0) {
      const pts = [['Raw intake', res.raw], ['Membrane feed', S.roF], ['First-pass permeate', S.perm1], ['RO concentrate', S.conc], ['Product', product], ['Discharge', brine]], st = pts.map((p) => p[1]), nP = pts.length;
      const ec = st.map((s) => conductivity(ionsOf(s), 25)), kf = st.map((s, i) => (ec[i] > 0 ? s.tds / ec[i] : 0.6)), rho = st.map((s) => s.rho), xm = [...st.map((s) => s.Q), ...st.map((s) => s.tds)], sQ = v.twinSigQ / 100, sC = v.twinSigC / 100, g7 = rng(7);
      const rowsIn = v.twin === 'table' ? (v.twinTable || []) : st.map((s, i) => ({ Q: s.Q * (1 + g7.normal(0, sQ)) * (i === 3 ? 1.08 : 1), EC: ec[i] * (1 + g7.normal(0, sC)) }));
      const y = [], sig = [], pseudo = [];
      for (let j = 0; j < 2 * nP; j++) { const r = rowsIn[j % nP] || {}, raw = j < nP ? +r.Q : +r.EC * kf[j - nP], okM = Number.isFinite(raw) && raw > 0 && (j < nP ? r.Q : r.EC) !== '' && (j < nP ? r.Q : r.EC) !== null; y.push(okM ? raw : xm[j]); sig.push(okM ? Math.max(1e-9, raw * (j < nP ? sQ : sC)) : 0.3 * Math.max(1e-9, xm[j])); pseudo.push(!okM); }
      const k3 = rho[0] * xm[0] - rho[4] * xm[4] - rho[5] * xm[5], k4 = xm[0] * xm[6] - xm[4] * xm[10] - xm[5] * xm[11], k5 = rho[4] * xm[4] - rho[2] * xm[2], k6 = rho[5] * xm[5] - rho[3] * xm[3];
      const gB = (x) => [(rho[1] * x[1] - rho[2] * x[2] - rho[3] * x[3]) / (rho[1] * xm[1]), (x[1] * x[7] - x[2] * x[8] - x[3] * x[9]) / (xm[1] * xm[7]), (rho[0] * x[0] - rho[4] * x[4] - rho[5] * x[5] - k3) / (rho[0] * xm[0]), (x[0] * x[6] - x[4] * x[10] - x[5] * x[11] - k4) / (xm[0] * xm[6]), (rho[4] * x[4] - rho[2] * x[2] - k5) / (rho[4] * xm[4]), (rho[5] * x[5] - rho[3] * x[3] - k6) / (rho[5] * xm[5])];
      let rc = null;
      try { rc = reconcile(y, sig, gB); } catch (e) { W.push({ level: 'warn', msg: `Data reconciliation failed (${e.message}); check the measurement table.` }); }
      if (rc && rc.x.every((q) => Number.isFinite(q))) {
        const zc = 2.81, names = [...pts.map((p) => `${p[0]} flow`), ...pts.map((p) => `${p[0]} TDS from conductivity`)], iMax = rc.z.indexOf(Math.max(...rc.z)), flags = rc.z.map((z, j) => (pseudo[j] ? 'not measured — estimated' : z > zc ? (j === iMax && rc.globalFail ? 'gross error' : 'suspect') : 'ok')), nGross = flags.filter((f) => f === 'gross error' || f === 'suspect').length;
        const recOf = (x) => x[2] / x[1], rejOf = (x) => 100 * (1 - x[8] / x[7]);
        xK.push({ label: 'Reconciled RO recovery', value: 100 * recOf(rc.x), unit: '%', status: rc.globalFail ? 'warn' : 'ok', help: `Data reconciliation of ${2 * nP - pseudo.filter(Boolean).length} measurements against 6 balances: χ² = ${fmt(rc.obj, 3)} (95 % limit ${fmt(rc.crit, 3)}), ${nGross} flagged; model ${fmt(100 * recOf(xm), 4)} %` });
        xT.push({ title: 'Digital twin: data reconciliation of plant measurements', columns: ['Measurement', 'Unit', 'Measured', 'Standard uncertainty', 'Reconciled', 'Adjustment (%)', 'Standardised adjustment', 'Flag', 'Model value', 'Reconciled − model (%)'],
          rows: names.map((nm, j) => [nm, j < nP ? 'm³/h' : 'mg/L', pseudo[j] ? null : y[j], pseudo[j] ? null : sig[j], rc.x[j], (100 * (rc.x[j] - y[j])) / y[j], rc.z[j], flags[j], xm[j], xm[j] > 0 ? (100 * (rc.x[j] - xm[j])) / xm[j] : 0]),
          note: `${v.twin === 'table' ? 'Measurements from the input table' : `Synthetic measurements: the model values with ${fmt(v.twinSigQ, 2)} % (flow) and ${fmt(v.twinSigC, 2)} % (conductivity) random error and a +8 % bias injected on the concentrate flow meter`}. Weighted least squares subject to the water and salt balances of the RO pass and of the whole plant and the water balances of the product line and of the waste line (dosed chemicals, solids and unmetered side streams such as backwash, bypass or distillate taken from the model); converged in ${rc.iterations} iterations, largest balance residual ${fmt(rc.residual, 2)}. A standardised adjustment above ${zc} flags a suspect instrument. Conductivity at 25 °C is converted to TDS with the factor of the model composition of each stream.` });
        xT.push({ title: 'Digital twin: state estimates', columns: ['Quantity', 'From raw measurements', 'Reconciled', 'Model'], rows: [['RO recovery (%)', 100 * recOf(y), 100 * recOf(rc.x), 100 * recOf(xm)], ['RO salt rejection (%)', rejOf(y), rejOf(rc.x), rejOf(xm)], ['Concentration factor of the RO pass', y[9] / y[7], rc.x[9] / rc.x[7], xm[9] / xm[7]],
          ['RO water-balance error (%)', 100 * rc.raw[0], 100 * gB(rc.x)[0], 0], ['RO salt-balance error (%)', 100 * rc.raw[1], 100 * gB(rc.x)[1], 0], ['Plant water-balance error (%)', 100 * rc.raw[2], 100 * gB(rc.x)[2], 0], ['Plant salt-balance error (%)', 100 * rc.raw[3], 100 * gB(rc.x)[3], 0], ['Product-line water-balance error (%)', 100 * rc.raw[4], 100 * gB(rc.x)[4], 0], ['Waste-line water-balance error (%)', 100 * rc.raw[5], 100 * gB(rc.x)[5], 0], [`Global test statistic χ² (95 % limit ${fmt(rc.crit, 3)})`, rc.obj, null, null]] });
        xP.push({ type: 'bar', title: 'Digital twin: standardised adjustments of the measurements', ylabel: '|adjustment| ÷ its standard deviation', categories: names, series: [{ name: 'Standardised adjustment', values: rc.z }], note: `Values above ${zc} indicate a gross error (95 % family-wise).` });
        if (rc.globalFail) W.push({ level: v.twin === 'table' ? 'warn' : 'info', msg: `Data reconciliation: the ${v.twin === 'table' ? '' : 'synthetic '}measurements do not satisfy the balances within their uncertainty (χ² ${fmt(rc.obj, 3)} > ${fmt(rc.crit, 3)}); most likely faulty instrument: ${names[iMax].toLowerCase()} (standardised adjustment ${fmt(rc.z[iMax], 3)}).` });
        xO.twin = { recovery: recOf(rc.x), rejectionPct: rejOf(rc.x), chi2: rc.obj, chi2Limit: rc.crit, flagged: nGross, suspect: rc.globalFail ? names[iMax] : 'none' };
      }
    }
    // ---- optimisation of the RO recovery on the flowsheet
    if (v.optimise !== 'none' && Qp > 0) {
      const byCost = v.optimise === 'cost', memo = new Map();
      const evalR = (r) => { const key = +r.toFixed(3); if (!memo.has(key)) { let o; try { const q = plantDesign({ ...v, recovery: key }).res, pen = Math.max(0, q.ro1.Preq / (q.ro1.pmax || 1e9) - 1) + Math.max(0, q.product.tds / v.limTDS - 1) + (v.maxRecChem > 0 ? Math.max(0, key / v.maxRecChem - 1) : 0) + (q.sol.converged ? 0 : 1), base = byCost ? plantCost(q, v).lcow : q.secElec; o = { r: key, base, pen, f: base * (1 + 10 * pen), P: q.ro1.Preq, tds: q.product.tds, Q: q.product.Q, ok: true }; } catch { o = { r: key, base: 0, pen: 1, f: 1e6, P: 0, tds: 0, Q: 0, ok: false }; } memo.set(key, o); } return memo.get(key); };
      const lo = Math.max(5, v.recovery - 15), hi = Math.min(92, v.recovery + 15), gs = goldenMin((r) => evalR(r).f, lo, hi, 8), cur = evalR(v.recovery), best = [...memo.values()].reduce((a, b) => (b.f < a.f ? b : a)), pts = [...memo.values()].filter((o) => o.ok).sort((a, b) => a.r - b.r), unit = byCost ? '$/m³' : 'kWh/m³';
      xK.push({ label: 'Optimum RO recovery', value: best.r, unit: '%', help: `Golden-section search on the flowsheet (${gs.evals.length} evaluations): ${byCost ? 'indicative cost of water' : 'specific electricity'} ${fmt(best.base, 4)} ${unit} against ${fmt(cur.base, 4)} ${unit} now` });
      xP.push({ type: 'line', title: `Optimisation: ${byCost ? 'indicative cost of water' : 'specific electricity'} against RO recovery`, xlabel: 'RO recovery (%)', ylabel: unit, series: [{ name: 'Flowsheet evaluations', x: pts.map((o) => o.r), y: pts.map((o) => o.base), mode: 'both' }, { name: 'Infeasible (pressure, TDS or scaling limit)', x: pts.filter((o) => o.pen > 0).map((o) => o.r), y: pts.filter((o) => o.pen > 0).map((o) => o.base), mode: 'points' }], vlines: [{ x: v.recovery, label: 'current' }, { x: best.r, label: 'optimum' }] });
      xT.push({ title: 'Optimisation of the RO recovery', columns: ['RO recovery (%)', byCost ? 'Cost of water ($/m³)' : 'Specific electricity (kWh/m³)', 'RO feed pressure (bar)', 'Product TDS (mg/L)', 'Product (m³/h)', 'Constraint violation', 'Feasible'], rows: pts.map((o) => [o.r, o.base, o.P, o.tds, o.Q, o.pen, o.pen > 0 ? 'no' : 'yes']), note: `Each row is a complete re-design of the flowsheet at that recovery (reference membrane solution, tear convergence, sizing). Constraints: element pressure rating, product TDS limit${v.maxRecChem > 0 ? ', scaling-limited recovery' : ''}; violations are penalised. The search covers ${fmt(lo, 3)}–${fmt(hi, 3)} % (15 points either side of the present recovery); small steps in the curve come from whole pressure vessels.` });
      if (best.pen === 0 && Math.abs(best.r - v.recovery) > 0.5 && best.base < cur.base * 0.998) xR.push(`Optimisation: an RO recovery of ${fmt(best.r, 3)} % lowers the ${byCost ? 'indicative cost of water' : 'specific electricity'} from ${fmt(cur.base, 4)} to ${fmt(best.base, 4)} ${unit}.`);
      xO.optimum = { recovery: best.r, objective: best.base, current: cur.base, feasible: best.pen === 0 };
    }
    // ---- daily operation: power supply, tank heat and pH loop
    if (dyn && dyn.supply) {
      const sy = dyn.supply, dd = sy.day, j0 = dyn.i0, tt = th.slice(0, -1), cut = (a) => a.slice(j0, dyn.n), rf = dd.load > 0 ? clamp(1 - dd.grid / dd.load, 0, 1) : 0, cu = dd.ren > 0 ? dd.curt / dd.ren : 0;
      xK.push({ label: 'Renewable fraction', value: 100 * rf, unit: '%', help: 'Share of the plant load not imported from the grid over the last simulated day' }, { label: 'Curtailed renewable energy', value: 100 * cu, unit: '%', status: cu > 0.25 ? 'warn' : 'ok' }, { label: 'Grid import', value: dd.grid / 1000, unit: 'MWh/d', help: `Peak import ${fmt(sy.gridPeak, 4)} kW` }, { label: 'Production shed', value: sy.shed / 3, unit: 'm³/d', status: sy.shed > 1e-6 ? 'warn' : 'ok', help: 'Product not made because the available electrical power was insufficient (average over three days)' });
      xP.push({ type: 'line', title: 'Daily operation: power supply and load', xlabel: 'Hour of day', ylabel: 'kW', zeroY: true, series: [{ name: 'Plant load', x: tt, y: cut(dyn.P) }, { name: 'PV + wind available', x: tt, y: cut(sy.ren) }, { name: 'Grid import', x: tt, y: cut(sy.grid) }, { name: 'Battery discharge', x: tt, y: cut(sy.dis) }, { name: 'Curtailed', x: tt, y: cut(sy.curt), dash: true }], ...(v.gridLimit === 'cap' ? { hlines: [{ y: v.gridMax, label: 'grid limit' }] } : {}) });
      if (base0.supply.Emax > 0) xP.push({ type: 'line', title: 'Daily operation: battery state of charge', xlabel: 'Hour of day', ylabel: '% of capacity', ymin: 0, ymax: 100, series: [{ name: 'State of charge', x: tt, y: cut(sy.soc) }] });
      xT.push({ title: 'Power supply, storage and grid limit (last simulated day)', columns: ['Item', 'Value', 'Unit'], rows: [['Plant load', dd.load / 1000, 'MWh/d'], ['PV + wind available', dd.ren / 1000, 'MWh/d'], ['Grid import', dd.grid / 1000, 'MWh/d'], ['Battery charged', dd.chg / 1000, 'MWh/d'], ['Battery discharged', dd.dis / 1000, 'MWh/d'], ['Curtailed', dd.curt / 1000, 'MWh/d'], ['Renewable fraction', 100 * rf, '%'], ['Curtailment', 100 * cu, '%'], ['Peak grid import', sy.gridPeak, 'kW'],
        ['Grid energy cost', sy.gridCost, '$/d'], ['Production shed (three days)', sy.shed, 'm³'], ['Time with production limited by power', (100 * sy.shedSteps) / dyn.n, '%'], ['Avoided CO₂', ((dd.load - dd.grid) * v.gridCarbon) / 1000, 't/d']],
        note: `Dispatch every ${fmt(dyn.dt * 60, 3)} min: renewables first, then the battery (${fmt(base0.supply.Emax, 4)} kWh, 90 % round trip, C/2), then grid import${v.gridLimit === 'cap' ? ` up to ${fmt(v.gridMax, 4)} kW` : v.gridLimit === 'island' ? ' — none (island operation)' : ''}. If that is not enough the production is shed and the product tank bridges the gap.` });
      if (prof) xT.push({ title: 'Hourly renewable resource and generation', columns: ['Hour', 'Global irradiance (W/m²)', 'Wind speed (m/s)', 'PV output (kW)', 'Wind output (kW)'], rows: prof.hours.slice(0, 24).map((h, i) => [h, prof.G[i], prof.v[i], prof.pv[i], prof.wind[i]]), note: `Clear-day shape scaled to ${fmt(v.ghi, 3)} kWh/m²·d (${fmt(prof.pvEnergy / 1000, 3)} MWh/d from PV at a performance ratio of 0.8); wind speed with an afternoon maximum around ${fmt(v.windMean, 3)} m/s, Rayleigh-distributed within the hour, cubic power curve 3–12–25 m/s (${fmt(prof.windEnergy / 1000, 3)} MWh/d).` });
      if (sy.shed > 1e-6) W.push({ level: 'warn', msg: `Electrical-power constraint: ${fmt(sy.shed, 3)} m³ of production was shed over three days because renewables, battery and the grid limit could not carry the load${dyn.unmet > 1e-6 ? '; demand was not fully met' : '; the tank covered the demand'}.` });
      Object.assign(xO, { renewableFraction: rf, curtailment: cu, gridImport: dd.grid, productionShed: sy.shed });
    }
    if (aux) {
      const ta = aux.t.slice(Math.max(0, aux.i0)).map((x) => x - aux.t[Math.max(0, aux.i0)]), la = (a) => a.slice(Math.max(0, aux.i0));
      xK.push({ label: 'Tank heat loss to ambient', value: aux.lossDay, unit: 'kWh/d', help: `Negative when the air is warmer than the water. Product-tank temperature ${fmt(aux.Tmin, 3)}–${fmt(aux.Tmax, 3)} °C; stored thermal energy relative to the mean air temperature ${fmt(aux.E[aux.E.length - 1] / 1000, 3)} MWh at the end` });
      xP.push({ type: 'line', title: 'Daily operation: product-tank temperature and stored thermal energy', xlabel: 'Hour of day', ylabel: '°C · MWh', series: [{ name: 'Tank temperature (°C)', x: ta, y: la(aux.T) }, { name: 'Ambient air (°C)', x: ta, y: la(aux.Ta), dash: true }, { name: 'Stored thermal energy above mean ambient (MWh)', x: ta, y: la(aux.E).map((e) => e / 1000) }] });
      const dynRows = [['Initial tank temperature', v.initTankT, '°C'], ['Final tank temperature', aux.Tend, '°C'], ['Initial stored thermal energy (relative to 0 °C)', aux.stored0 / 1000, 'MWh'], ['Final stored thermal energy (relative to 0 °C)', aux.storedEnd / 1000, 'MWh'], ['Energy in with product (three days)', aux.Ein / 1000, 'MWh'], ['Energy out with delivered water (three days)', aux.Eout / 1000, 'MWh'], ['Heat loss to ambient (three days)', aux.Eloss / 1000, 'MWh'], ['Heat loss to ambient (last day)', aux.lossDay, 'kWh/d'], ['Tank heat-loss coefficient UA', tankUA, 'kW/K']];
      if (aux.iae !== undefined) {
        xK.push({ label: 'Product pH control error', value: aux.iae, unit: 'pH·h/d', status: aux.pHmax - aux.pHmin > 0.6 ? 'warn' : 'ok', help: `PID caustic trim: pH ${fmt(aux.pHmin, 3)}–${fmt(aux.pHmax, 3)} around the set-point ${fmt(phSP, 3)}; controller output at a limit ${fmt(100 * aux.sat, 3)} % of the time` });
        xP.push({ type: 'line', title: 'Daily operation: product pH loop (PID with anti-windup)', xlabel: 'Hour of day', ylabel: 'pH · mg/L', series: [{ name: 'pH in the contact tank', x: ta, y: la(aux.pH) }, { name: 'Measured pH (sensor lag)', x: ta, y: la(aux.pHm), dash: true }, { name: 'Caustic trim (mg/L NaOH × 10)', x: ta, y: la(aux.dose).map((s) => 400 * s) }], hlines: [{ y: phSP, label: 'set-point' }], vlines: [{ x: 8, label: 'CO₂ +' }, { x: 16, label: 'CO₂ −' }] });
        dynRows.push(['pH set-point', phSP, '–'], ['Integral absolute pH error (last day)', aux.iae, 'pH·h'], ['pH range (last day)', aux.pHmax - aux.pHmin, '–'], ['Caustic trim used (last day)', (aux.doseMol * 40) / 1000, 'kg/d'], ['Controller output at a limit', 100 * aux.sat, '% of time'], ['Largest integrator state', aux.Imax, 'pH·h']);
      }
      xT.push({ title: 'Product-tank thermal inventory and pH control loop', columns: ['Item', 'Value', 'Unit'], rows: dynRows, note: `Tank energy balance d(ρcVT)/dt = ρc·q_in·T_product − ρc·q_out·T − UA·(T − T_air), air at ${fmt(v.airTemp, 3)} ± ${fmt(v.airSwing, 3)} °C over the day.${aux.iae !== undefined ? ` pH loop: 10-minute contact tank, 1.5-minute sensor lag, PID on a flow-paced caustic trim (set-point 0.15 above the untrimmed product pH) limited to 0–${fmt(v.phMaxDose, 3)} mg/L${v.phAW ? ' with back-calculation anti-windup' : ' without anti-windup'}; disturbances are the production changes of the level loop and ±${fmt(v.phDist, 3)} % steps of dissolved CO₂ at 08:00–12:00 and 16:00–20:00 (only caustic can be dosed, so the negative step saturates the controller).` : ''}` });
      Object.assign(xO, { tankHeatLoss: aux.lossDay, tankTempEnd: aux.Tend, ...(aux.iae !== undefined ? { phIAE: aux.iae } : {}) });
    }
    Object.assign(xO, { entropyGenerated: res.sGen, bubblePressure: feedV.Pbub, feedVapourFraction: feedV.beta, dissolvedOxygen: o2mg, costOfWater: cost.lcow, capexIndicative: cost.capex, opexIndicative: cost.opexTot, saltLoad, dischargeExcessSalinity: brine.S - res.raw.S });
    Object.assign(out, xO);
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
        ...xK,
      ],
      recommendations: [
        exRank[0] ? `${exRank[0].u.name} destroys the most exergy (${fmt(exRank[0].exDest, 3)} kW, ${fmt((100 * exRank[0].exDest) / res.exDest, 2)} %) — the first place to look for energy savings.` : null,
        v.erd === 'none' && res.ro1.Preq > 20 ? `The concentrate valve throttles ${fmt(res.get('BV')?.rec.throttled || 0, 3)} kW — fit an energy-recovery device.` : null,
        bn && bn.util > 1 ? `${bn.name} is the bottleneck in the “${bnCase.name}” case at ${fmt(100 * bn.util, 3)} % of its limit: add margin there before anything else.` : bn ? `Largest utilisation across the envelope: ${bn.name.toLowerCase()} at ${fmt(100 * bn.util, 3)} % (${bnCase.name.toLowerCase()}).` : null,
        dynAlt.length && dynAlt[2][1].cost < dynAlt[1][1].cost * 0.995 && dynAlt[2][1].unmet < 1e-6 ? `Shifting production away from the peak tariff with the tank saves ${fmt(dynAlt[1][1].cost - dynAlt[2][1].cost, 3)} $/d (${fmt((100 * (dynAlt[1][1].cost - dynAlt[2][1].cost)) / dynAlt[1][1].cost, 2)} %).` : null,
        ...xR,
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
        ...xP,
      ],
      tables: [
        { title: 'Stream table — conditions', columns: ['Stream', 'From → to', 'Phase', 'Flow (m³/h)', 'Mass flow (t/h)', 'T (°C)', 'P (bar a)', 'TDS (mg/L)', 'Salinity (g/kg)', 'pH', 'Density (kg/m³)', 'Enthalpy (kJ/kg)', 'Physical exergy (kW)', 'Chemical exergy (kW)'], rows: streamRows, note: `Dead state: raw feed at ${fmt(dead.T, 3)} °C and 1.013 bar. Enthalpy is relative to liquid at 0 °C and includes flow work.` },
        { title: 'Stream table — composition (mg/L)', columns: ['Stream', ...ION_IDS.map((k) => IONS[k].label), 'Suspended solids'], rows: compRows },
        { title: 'Unit operations', columns: ['Tag', 'Unit', 'Inlet (m³/h)', 'Main outlet (m³/h)', 'Recovery (%)', 'ΔP (bar)', 'T in (°C)', 'T out (°C)', 'Power (kW)', 'Heat duty (kW)', 'Cooling (kW)', 'Chemicals', 'Exergy destroyed (kW)', 'Share of destruction (%)', 'Mass closure (%)', 'Energy closure (%)'], rows: unitRows, note: 'Heat duty is steam heat for thermal units and exchanged heat for heat exchangers. Negative power is generation. Closure columns are the balance errors of each unit.' },
        { title: 'Plant performance and water quality', columns: ['Quantity', 'Feed', 'Product', 'Discharge'], rows: quality, note: 'Electrolyte rows: water activity from the osmotic coefficient, ion activity coefficients from the Truesdell–Jones extended Debye–Hückel equation, sulphate and bicarbonate corrected for their main ion pairs; indicative above an ionic strength of about 2 mol/kg.' },
        { title: 'Utilities, chemicals and residuals', columns: ['Item', 'Value', 'Unit'], rows: utilRows },
        { title: 'Off-design cases and bottlenecks', columns: ['Case', 'Feed T (°C)', 'Salinity (%)', 'Load (%)', 'Product (m³/h)', 'Product TDS (mg/L)', 'RO feed pressure (bar)', 'Flux (L/m²·h)', 'Power (kW)', 'SEC (kWh/m³)', 'Limiting equipment', 'Utilisation (%)', 'Within limits'],
          rows: cases.map((c) => [c.name, c.q.cond.T, 100 * c.q.cond.sf, 100 * c.q.cond.load, c.q.product.Q, c.q.product.tds, c.q.ro1.Preq, c.q.ro1.flux, c.q.power, c.q.secElec, c.top ? c.top.name : '–', c.top ? 100 * c.top.util : null, c.top && c.top.util > 1.0005 ? 'no' : 'yes']), note: `Equipment sizes are frozen at the design case; recovery is held. Design-case utilisations: ${lims.slice(0, 5).map((l) => `${l.name} ${fmt(100 * l.util, 3)} %`).join('; ')}.` },
        { title: 'Equipment sizing hints', columns: ['Equipment', 'Quantity', 'Value', 'Unit', 'Basis'], rows: sizing },
        ...(dyn ? [{ title: 'Daily operation and controller performance', columns: ['Strategy', 'Energy (MWh/d)', 'Energy cost ($/d)', 'Minimum level (%)', 'Maximum level (%)', 'Integral absolute error (%·h)', 'Starts', 'Time at a limit (%)', 'Unmet demand (m³)', 'Overflow (m³)'],
          rows: (dynAlt.length ? dynAlt : [['Selected control', dyn]]).map(([n, d]) => [n, d.energy / 1000, d.cost, d.Lmin, d.Lmax, d.iae, d.starts, 100 * d.sat, d.unmet, d.spill]), note: `Tank ${fmt(dyn.Vmax, 4)} m³; three days simulated with a ${fmt(dyn.dt * 60, 3)} min step, last day reported. Unmet demand and overflow are totals over the three days.` }] : []),
        ...xT,
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
    // second law: entropy balance and Gouy–Stodola
    const T0v = res.dead.T + KELVIN;
    add('Gouy–Stodola: T₀·S_gen equals the exergy destroyed in every unit', 0, mx((q) => Math.abs(T0v * q.sGen - q.exDest)) / res.exIn, 1e-10, 'Entropy balance and exergy balance are evaluated separately for each unit (largest difference ÷ exergy supplied)');
    add('Entropy generation is non-negative in every unit', 0, Math.min(0, ...recs.map((q) => q.sGen)) / res.sGen, 1e-9, 'Second law, unit by unit');
    const vd = { T: zf.T, xw: 1, xi: 0, phi: 1 }, vv = UNITS.valve([zf], { Pout: 2 }).outs[0], sgV = entropy(vv, vd) - entropy(zf, vd) + (hInc(zf, vd) - hInc(vv, vd)) / (zf.T + KELVIN);
    add('Entropy generated by throttling, hand calculation', (zf.Q * 58) / 36 / (0.5 * (zf.T + vv.T) + KELVIN), sgV, 0.01 * sgV, 'Adiabatic valve from 60 to 2 bar: S_gen = V̇·Δp/T (kW/K)');
    const m1 = fromVol(100, { Na: 2, Cl: 3 }, 20, 2, 7), m2 = fromVol(100, { Na: 2, Cl: 3 }, 40, 2, 7), mm = UNITS.mix([m1, m2]).outs[0], md0 = { T: 20, xw: 1, xi: 0, phi: 1 }, cm = cp(30, 0) / 3.6e6;
    add('Entropy generated by mixing two temperatures, hand calculation', m1.m * cm * Math.log((mm.T + KELVIN) / 293.15) + m2.m * cm * Math.log((mm.T + KELVIN) / 313.15), entropy(mm, md0) - entropy(m1, md0) - entropy(m2, md0) + (hInc(m1, md0) + hInc(m2, md0) - hInc(mm, md0)) / 293.15, 1.3e-3, 'S_gen = Σ ṁ·c·ln(T_mix/T_i) for equal streams at 20 and 40 °C (kW/K)');
    // phase equilibrium
    add('Rachford–Rice: two-component analytical solution', -(0.4 * 2 + 0.6 * -0.5) / (2 * -0.5), rachfordRice([0.4, 0.6], [3, 0.5]), 1e-12, 'β = −(z₁(K₁−1) + z₂(K₂−1)) / ((K₁−1)(K₂−1)) for z = (0.4, 0.6), K = (3, 0.5)');
    const sw = fromVol(1000, WATERS.seawater.ions, 25, P0, 8.1), fresh = fromVol(1000, { Na: 1, Cl: 1.5 }, 25, P0, 7), fl = streamFlash(sw, 0.08, { model: 'pr' }), ids = ['H2O', 'CO2', 'N2', 'O2'];
    add('Flash: fugacities are equal in liquid and vapour', 0, Math.max(...fl.fL.map((a, j) => Math.abs(a - fl.fV[j]) / a)), 1e-10, `Seawater with dissolved CO₂, N₂, O₂ at 25 °C and 0.08 bar, Peng–Robinson vapour; vapour fraction ${fmt(fl.beta, 3)}`);
    add('Flash: every component is conserved between the phases', 0, Math.max(...ids.map((k) => Math.abs(fl.L[k] + fl.V[k] - fl.n[k]) / fl.n[k])), 1e-12, 'L·x + V·y = F·z for water, CO₂, N₂ and O₂');
    add('Raoult’s law: bubble pressure of pure water is its saturation pressure', psat(60) / 1e5, flashVLE({ H2O: 1 }, 60, 1, { model: 'ideal' }).Pbub, 1e-12, 'bar at 60 °C');
    const noGas = { H2O: fl.n.H2O, salt: fl.n.salt };
    add('Modified Raoult’s law: vapour pressure of seawater', psat(25) / 1e5 / (1 + 0.57357 * (sw.S / (1000 - sw.S))), flashVLE(noGas, 25, 1, { model: 'gamma', S: sw.S, I: fl.I }).Pbub, 1e-4, 'γ·x·P_sat with the water activity from the osmotic coefficient against the seawater vapour-pressure correlation (bar)');
    add('Ideal Raoult’s law overstates the lowering by the salt', 1, flashVLE(noGas, 25, 1, { model: 'ideal' }).Pbub < flashVLE(noGas, 25, 1, { model: 'gamma', S: sw.S, I: fl.I }).Pbub ? 1 : 0, 0, 'x_w < a_w because the osmotic coefficient of seawater is below one');
    const o2 = (s) => (streamFlash(s, P0, {}).n.O2 * 31.999) / s.Q;
    add('Henry’s law: oxygen in air-saturated fresh water at 25 °C', 8.26, o2(fresh), 0.1, 'mg/L, standard dissolved-oxygen table');
    add('Henry’s law with salting-out: oxygen in air-saturated seawater at 25 °C', 6.75, o2(sw), 0.15, 'mg/L at 35 g/kg, standard dissolved-oxygen table');
    add('Phase fractions: an air-supersaturated feed splits into two phases', 1, streamFlash(sw, P0, { sat: 1.3 }).beta > 0 && streamFlash(sw, P0, { sat: 0.9 }).beta === 0 ? 1 : 0, 0, '130 % of air saturation gives a free-gas fraction at 1 atm, 90 % none');
    const prN = pengRobinson([0, 0, 1, 0], 26.85, 0.1);
    add('Equation of state: ideal-gas limit', 1, pengRobinson([0.4, 0.1, 0.3, 0.2], 25, 1e-5).Z, 1e-6, 'Peng–Robinson compressibility → 1 as P → 0');
    add('Equation of state: fugacity coefficient agrees with its second virial coefficient', (prN.B2 * 1e4) / (R * 300), Math.log(prN.phi[2]), 1e-7, 'ln φ = B·P/RT for nitrogen at 300 K and 0.1 bar');
    add('Equation of state: saturated steam at 100 °C', 0.984, pengRobinson([1, 0, 0, 0], 100, 1.01325).Z, 0.01, 'Compressibility from steam tables (P·v/RT)');
    const es = electrolyte(sw);
    add('Activity coefficient of NaCl at 0.1 mol/kg', 0.778, Math.sqrt(ionGamma('Na', 0.1) * ionGamma('Cl', 0.1)), 0.01, 'Mean ionic activity coefficient, literature value at 25 °C');
    add('Water activity of standard seawater', 0.9815, es.aw, 0.001, '35 g/kg at 25 °C');
    add('Gypsum saturation index of standard seawater', -0.64, es.siGypsum, 0.1, 'Ion-association value from speciation codes at 25 °C; seawater is undersaturated');
    // compressors
    add('Polytropic and isentropic compressor work coincide at 100 % efficiency', compressorWork(100, 110, 1).w, compressorWork(100, 110, 1, 'polytropic').w, 1e-6, 'J/kg');
    const gc = gasCompressor(0.5, 1.5, 25, 30, 0.7), gp = gasCompressor(0.5, 1.5, 25, 30, 0.7, 'polytropic');
    add('Gas compressor: isentropic work, hand calculation', (30 * 298.15 * (3 ** (R / 30) - 1)) / 0.7, gc.w, 1e-6, 'w = c_p·T₁·(r^(R/c_p) − 1)/η (J/mol)');
    add('Gas compressor: polytropic work, hand calculation', 30 * 298.15 * (3 ** (R / 30 / 0.7) - 1), gp.w, 1e-6, 'w = c_p·T₁·(r^((n−1)/n) − 1) with (n−1)/n = (k−1)/(k·η_p) (J/mol)');
    // new unit operations inside the flowsheet
    const xv = { ...d, useDeaer: true, polish: 'ed', brineConc: 'md' }, ext = solveCase(xv, ref), xr = ext.sol.recs, dea = ext.get('DEA').rec, edr = ext.get('ED'), mdr = ext.get('MD');
    add('Flowsheet with deaerator, electrodialysis and membrane distillation: unit balances close', 0, Math.max(...xr.map((q) => Math.max(q.bal.mass, q.bal.ion, q.bal.energy))), 1e-8, 'Mass (with the vent), ions and energy around every unit');
    add('Same flowsheet: plant mass and energy balances', 0, Math.abs(ext.plant.mass.in - ext.plant.mass.out) / ext.plant.mass.in + Math.abs(ext.plant.energy.in - ext.plant.energy.out) / ext.plant.energy.in, 1e-7, 'Vent mass and enthalpy included');
    add('Deaerator: CO₂ leaving the liquid equals CO₂ in the vent', dea.ventCO2, ext.get('DEA').inS[0].co2 - ext.get('DEA').outS[0].co2, 1e-9, 'kg/h');
    add('Deaerator: oxygen removal by the vacuum flash', 1, dea.removal.O2 > 0.85 && dea.removal.O2 < 1 && dea.removal.CO2 < dea.removal.O2 ? 1 : 0, 0, `O₂ ${fmt(100 * dea.removal.O2, 3)} %, CO₂ ${fmt(100 * dea.removal.CO2, 3)} % — the more soluble gas is stripped less`);
    const ei = edr.inS[0], ceq = sum(ION_IDS.map((k) => (IONS[k].z > 0 ? (IONS[k].z * ionsOf(ei)[k]) / IONS[k].mw : 0)));
    add('Electrodialysis: current from Faraday’s law, hand calculation', (FARADAY * ei.Q * 0.92 * ceq * 0.5) / 3600 / 0.9, edr.rec.Itot, 1e-6 * edr.rec.Itot, 'I·N = F·Q_d·ΔC/ξ (A × cell pairs)');
    add('Electrodialysis: energy use exceeds the minimum work of the transfer', 1, edr.rec.dc > edr.rec.minWork && edr.rec.i < edr.rec.ilim ? 1 : 0, 0, `DC power ${fmt(edr.rec.dc, 3)} kW against ${fmt(edr.rec.minWork, 3)} kW; below the limiting current density`);
    add('Electrodialysis: neutral boron is not removed', ionsOf(ei).B, ionsOf(edr.outS[0]).B, 0.01 * ionsOf(ei).B, 'mg/L in the diluate equals the feed (water transport aside)');
    const pw = fromVol(100, { Na: 1, Cl: 1.5 }, 25, 2, 7), mp = { rec: 0.01, Th: 70, Tc: 25, Bm: 5e-7, hm: 400, tpc: 0.6, eps: 0, aux: 0, etaM: 0.95 }, mdp = UNITS.md([pw], mp).rec, mdb = UNITS.md([fromVol(100, scaleIons(WATERS.seawater.ions, 3), 25, 2, 7)], mp).rec;
    add('Membrane distillation: pure-water flux, hand calculation', 5e-7 * (psat(61) - psat(34)) * 3600, mdp.flux, 1e-3 * mdp.flux, 'J = B·(P_sat(T_fm) − P_sat(T_pm)), kg/m²·h');
    add('Membrane distillation: brine lowers the flux through its water activity', 1, mdb.flux < mdp.flux && mdr.rec.eta < 1 && mdr.rec.cool >= 0 ? 1 : 0, 0, `Flux ${fmt(mdb.flux, 3)} against ${fmt(mdp.flux, 3)} kg/m²·h; thermal efficiency ${fmt(100 * mdr.rec.eta, 3)} %`);
    const gf = fromVol(1000, WATERS.gulf.ions, 30, 4.5, 8.2), mq = UNITS.msf([gf], { make: 0.35, rec: 0.4, stages: 24, TBT: 110, TTD: 3, Tlast: 40, secElec: 3.5, loss: 0.02 }), mb = unitBalance({ inS: [gf], outS: mq.outs, rec: mq.rec });
    add('Multi-stage flash: mass and energy balances close', 0, mb.mass + mb.energy + mb.ion, 1e-9, 'Distillate, blowdown and cooling-water reject');
    add('Multi-stage flash: stage-wise flash against the continuous limit', 1 - Math.exp((-cp(75, mq.rec.Sr) * 70) / latentHeat(75, 0)), mq.rec.yTot, 0.004, 'Distillate per kg of recirculating brine: 1 − exp(−c_p·ΔT/λ) for a 70 K flashing range');
    add('Multi-stage flash: gained output ratio in the range of operating plants', 8.5, mq.rec.GOR, 1.5, '24 stages, 110 °C top brine temperature (kg distillate per kg steam)');
    const cal = solveCase({ ...d, reminMethod: 'calcite' }, ref).get('REM').rec;
    add('Calcite contactor: first-order dissolution kinetics', 1 - Math.exp(-cal.kDiss * cal.ebct), cal.approach, 1e-12, 'X = 1 − exp(−k·t) at the empty-bed contact time');
    // economics, environment, surrogate, reconciliation, optimisation
    const pc = plantCost(res, d);
    add('Capital-recovery factor, hand value', 0.078227, crf(0.06, 25), 1e-6, 'i(1+i)ⁿ/((1+i)ⁿ − 1) at 6 % and 25 years');
    add('Cost roll-up: unit costs sum to the plant total and to the cost of water', pc.lcow * pc.annual, crf(d.discount / 100, d.plantLife) * d.capexFactor * sum(pc.rows.map((r) => r.cost)) + sum(Object.values(pc.opex)), 1e-6, '$/y');
    add('Indicative cost of water in the range of seawater RO plants', 0.9, pc.lcow, 0.5, '$/m³ for a 10 000 m³/d plant');
    add('Salt load of the discharge from the plant salt balance', res.raw.salt + sum(recs.map((q) => sum(q.rec.chemIons || []) + sum(q.rec.gen || []))) - res.product.salt, res.brine.salt, 1e-6 * res.brine.salt, 'kg/h: feed salts + dosed ions + reaction terms − product salts');
    const pq = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], tf = ([a, b]) => 2 - 0.3 * a + 0.1 * b + 0.05 * a * a - 0.02 * b * b + 0.04 * a * b;
    add('Response surface recovers an exact quadratic', tf([0.3, -0.7]), fitSurface(pq, pq.map(tf)).predict([0.3, -0.7]), 1e-9, 'Least-squares fit on the 3 × 3 training grid');
    const C1 = { area1: res.ro1.area }, cond = ([a, b]) => ({ T: 24.5 + 7.5 * a, sf: 1.005 + 0.045 * b }), sf = fitSurface(pq, pq.map((x) => solveCase(d, ref, cond(x), C1).secElec)), held = [[0.5, -0.5], [-0.6, 0.3]];
    add('Surrogate against held-out flowsheet solutions', 0, Math.max(...held.map((x) => Math.abs(sf.predict(x) / solveCase(d, ref, cond(x), C1).secElec - 1))), 0.01, 'Largest relative error of the specific-electricity surface at two operating points not used for training');
    const r2 = reconcile([100, 104], [1, 2], (x) => [x[0] - x[1]]);
    add('Data reconciliation: two meters on one flow', (100 / 1 + 104 / 4) / (1 + 1 / 4), r2.x[0], 1e-9, 'Inverse-variance weighted mean');
    const yb = [100, 45.2, 58.9, 35000, 300, 61000], rb = reconcile(yb, [1, 0.45, 0.59, 700, 6, 1220], (x) => [(x[0] - x[1] - x[2]) / 100, (x[0] * x[3] - x[1] * x[4] - x[2] * x[5]) / 3.5e6]);
    add('Data reconciliation: adjusted values satisfy the water and salt balances', 0, rb.residual, 1e-10, 'Bilinear salt balance solved by successive linearisation');
    const rg = reconcile([100, 45, 55, 100.5, 45.3, 61], [1, 0.45, 0.55, 1, 0.45, 0.55], (x) => [x[0] - x[1] - x[2], x[3] - x[4] - x[5], x[0] - x[3], x[1] - x[4], x[2] - x[5]]);
    add('Data reconciliation: a biased meter is singled out', 5, rg.z.indexOf(Math.max(...rg.z)) + (rg.globalFail ? 0 : 99), 0, 'Duplicate metering of a splitter with +10 % bias on the sixth meter: global test fails and the largest standardised adjustment is on that meter');
    add('Golden-section search finds an analytical minimum', 2, goldenMin((x) => (x - 2) ** 2 + 1, 0, 5, 30).x, 1e-5, 'Minimum of (x − 2)² + 1');
    // renewable supply, power constraint, tank heat and pH loop
    const pr = resourceProfile({ ghi: 6, wind: 7, pvKW: 1000, windKW: 1000 });
    add('Hourly solar profile integrates to the daily irradiation', 6, trapz(pr.hours, pr.G) / 1000, 1e-9, 'kWh/m²·d');
    add('PV energy equals capacity × performance ratio × irradiation', 1000 * 0.8 * 6, pr.pvEnergy, 1e-6, 'kWh/d');
    const dsup = { Qn: 400, Vmax: 2400, L0: 0.6, hours: 72, dtMin: 10, mode: 'vfd', Kp: 2.5, Ki: 0.4, uMin: 0.5, uff: 0.85, nTrains: 4, onLow: 0.4, offHigh: 0.8, demand: () => 340, sec: () => 3, price: () => 0.1, Lsp: () => 0.6 };
    const ds = dynamicSim({ ...dsup, supply: { pv: pr.pvAt, wind: pr.windAt, Emax: 1500, E0: 750, Pb: 750, eta: Math.sqrt(0.9), gridMax: 500 } }).supply, tt = ds.total;
    add('Power supply: generation + grid + battery discharge = load + charging + curtailment', tt.ren + tt.grid + tt.dis, tt.load + tt.chg + tt.curt, 1e-6 * tt.load, 'kWh over three days');
    add('Battery energy balance', ds.Eend - 750, Math.sqrt(0.9) * tt.chg - tt.dis / Math.sqrt(0.9), 1e-6 * tt.chg + 1e-9, 'ΔE = η·charged − discharged/η (kWh)');
    const dc = dynamicSim({ ...dsup, supply: { pv: () => 0, wind: () => 0, Emax: 0, E0: 0, Pb: 0, eta: 1, gridMax: 700 } });
    add('Electrical-power constraint: grid import never exceeds the limit', 700, Math.max(...dc.supply.grid), 1e-6, 'kW; the plant needs 1020 kW at the demand, so production is shed');
    add('Electrical-power constraint: shed production is the shortfall', 72 * 340 - 72 * (700 / 3) - 0.6 * 2400, dc.unmet, 1, 'm³ of unmet demand = demand − power-limited production − initial stock');
    const cool = auxDynamics({ t: [0, 48], qIn: [0, 0], qOut: [0, 0], Vmax: 1000, V0: 500, hours: 48, dt: 0.25, Tp: 25, T0: 60, Ta: 20, swing: 0, UA: 5, ph: null });
    add('Stored thermal energy: closed tank cools exponentially', 20 + 40 * Math.exp((-5 * 48) / (1.161 * 500)), cool.Tend, 1e-6, 'T = T_a + (T₀ − T_a)·exp(−UA·t/ρcV) (°C)');
    const tk = auxDynamics({ t: [0, 24, 48], qIn: [300, 420, 300], qOut: [340, 340, 340], Vmax: 3000, V0: 1500, hours: 48, dt: 0.1, Tp: 26, T0: 35, Ta: 28, swing: 6, UA: 4, ph: null });
    add('Stored thermal energy: tank energy balance over the run', tk.storedEnd - tk.stored0, tk.Ein - tk.Eout - tk.Eloss, 1e-8 * tk.Ein, 'Δ(ρcVT) = energy in − energy out − heat lost to ambient (kWh)');
    const php = { pK: 6.35, sp: 8.3, s0: 0, A0: 1, c0: 1 / 10 ** (8.3 - 6.35) * 1.3, Vc: 50, tauM: 1.5 / 60, Kc: 0.01, Ti: 8 / 60, Td: 0.5 / 60, Tt: 4 / 60, sMax: 0.05, aw: true, dist: () => 0 }, pha = { t: [0, 24], qIn: [300, 300], qOut: [300, 300], Vmax: 3000, V0: 1500, hours: 24, dt: 1 / 60, Tp: 25, T0: 25, Ta: 25, swing: 0, UA: 0 };
    add('pH loop: integral action removes the offset after a CO₂ step', 8.3, auxDynamics({ ...pha, ph: php }).pHmEnd, 1e-4, 'Dissolved CO₂ 30 % above design from t = 0; PID caustic trim restores the set-point');
    const wOn = auxDynamics({ ...pha, ph: { ...php, sMax: 0.002 } }).Imax, wOff = auxDynamics({ ...pha, ph: { ...php, sMax: 0.002, aw: false } }).Imax;
    add('pH loop: anti-windup bounds the integrator while the actuator is saturated', 1, wOn < 0.2 * wOff ? 1 : 0, 0, `Largest integrator state ${fmt(wOn, 3)} pH·h with back-calculation against ${fmt(wOff, 3)} pH·h without`);
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
