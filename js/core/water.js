// Water-analysis data model shared by all suites: ion table, reference waters and bulk helpers.
// Compositions are objects keyed by ION id holding concentrations in mg/L.
import { R, KELVIN, osmoticCoefficient, density, salinityFromTDS } from './props.js';

/** id -> { label, mw (g/mol), z (charge), D (m²/s at 25 °C, infinite dilution), lambda (S·cm²/eq) } */
export const IONS = {
  Na: { label: 'Na⁺', name: 'Sodium', mw: 22.990, z: 1, D: 1.334e-9, lambda: 50.1 },
  K: { label: 'K⁺', name: 'Potassium', mw: 39.098, z: 1, D: 1.957e-9, lambda: 73.5 },
  Ca: { label: 'Ca²⁺', name: 'Calcium', mw: 40.078, z: 2, D: 0.792e-9, lambda: 59.5 },
  Mg: { label: 'Mg²⁺', name: 'Magnesium', mw: 24.305, z: 2, D: 0.706e-9, lambda: 53.1 },
  Ba: { label: 'Ba²⁺', name: 'Barium', mw: 137.327, z: 2, D: 0.847e-9, lambda: 63.6 },
  Sr: { label: 'Sr²⁺', name: 'Strontium', mw: 87.62, z: 2, D: 0.791e-9, lambda: 59.4 },
  NH4: { label: 'NH₄⁺', name: 'Ammonium', mw: 18.039, z: 1, D: 1.957e-9, lambda: 73.5 },
  Fe: { label: 'Fe²⁺', name: 'Iron', mw: 55.845, z: 2, D: 0.719e-9, lambda: 54.0 },
  Mn: { label: 'Mn²⁺', name: 'Manganese', mw: 54.938, z: 2, D: 0.712e-9, lambda: 53.5 },
  Cl: { label: 'Cl⁻', name: 'Chloride', mw: 35.453, z: -1, D: 2.032e-9, lambda: 76.3 },
  SO4: { label: 'SO₄²⁻', name: 'Sulphate', mw: 96.06, z: -2, D: 1.065e-9, lambda: 80.0 },
  HCO3: { label: 'HCO₃⁻', name: 'Bicarbonate', mw: 61.017, z: -1, D: 1.185e-9, lambda: 44.5 },
  CO3: { label: 'CO₃²⁻', name: 'Carbonate', mw: 60.009, z: -2, D: 0.923e-9, lambda: 69.3 },
  NO3: { label: 'NO₃⁻', name: 'Nitrate', mw: 62.004, z: -1, D: 1.902e-9, lambda: 71.4 },
  F: { label: 'F⁻', name: 'Fluoride', mw: 18.998, z: -1, D: 1.475e-9, lambda: 55.4 },
  PO4: { label: 'PO₄³⁻', name: 'Phosphate', mw: 94.971, z: -3, D: 0.824e-9, lambda: 69.0 },
  SiO2: { label: 'SiO₂', name: 'Silica', mw: 60.084, z: 0, D: 1.1e-9, lambda: 0 },
  B: { label: 'B', name: 'Boron', mw: 10.811, z: 0, D: 1.1e-9, lambda: 0 },
};
export const ION_IDS = Object.keys(IONS);
export const CATIONS = ION_IDS.filter((k) => IONS[k].z > 0);
export const ANIONS = ION_IDS.filter((k) => IONS[k].z < 0);

/** Reference waters (mg/L). Standard seawater follows the Millero reference composition at S ≈ 35. */
export const WATERS = {
  seawater: { name: 'Standard seawater (35 g/kg)', T: 25, pH: 8.1, ions: { Na: 11020, K: 408, Ca: 421, Mg: 1314, Ba: 0.02, Sr: 8.1, NH4: 0, Fe: 0, Mn: 0, Cl: 19810, SO4: 2776, HCO3: 110, CO3: 15, NO3: 0.5, F: 1.3, PO4: 0.1, SiO2: 2, B: 4.6 } },
  gulf: { name: 'Arabian Gulf seawater (45 g/kg)', T: 30, pH: 8.2, ions: { Na: 14170, K: 525, Ca: 541, Mg: 1690, Ba: 0.02, Sr: 10.4, NH4: 0, Fe: 0, Mn: 0, Cl: 25470, SO4: 3570, HCO3: 140, CO3: 19, NO3: 0.5, F: 1.7, PO4: 0.1, SiO2: 2, B: 5.9 } },
  redsea: { name: 'Red Sea seawater (41 g/kg)', T: 28, pH: 8.2, ions: { Na: 12910, K: 478, Ca: 493, Mg: 1539, Ba: 0.02, Sr: 9.5, NH4: 0, Fe: 0, Mn: 0, Cl: 23210, SO4: 3252, HCO3: 128, CO3: 17, NO3: 0.5, F: 1.5, PO4: 0.1, SiO2: 2, B: 5.4 } },
  med: { name: 'Mediterranean seawater (38.5 g/kg)', T: 22, pH: 8.1, ions: { Na: 12120, K: 449, Ca: 463, Mg: 1445, Ba: 0.02, Sr: 8.9, NH4: 0, Fe: 0, Mn: 0, Cl: 21790, SO4: 3054, HCO3: 121, CO3: 16, NO3: 0.5, F: 1.4, PO4: 0.1, SiO2: 2, B: 5.1 } },
  brackish: { name: 'Brackish groundwater (≈3.5 g/L)', T: 25, pH: 7.6, ions: { Na: 880, K: 22, Ca: 180, Mg: 85, Ba: 0.08, Sr: 4.5, NH4: 0, Fe: 0.1, Mn: 0.02, Cl: 1290, SO4: 720, HCO3: 310, CO3: 0.6, NO3: 12, F: 0.9, PO4: 0.1, SiO2: 28, B: 0.6 } },
  lowbrackish: { name: 'Low-salinity brackish / river (≈1 g/L)', T: 20, pH: 7.8, ions: { Na: 190, K: 8, Ca: 85, Mg: 28, Ba: 0.05, Sr: 1.2, NH4: 0.2, Fe: 0.05, Mn: 0.01, Cl: 250, SO4: 190, HCO3: 240, CO3: 0.7, NO3: 9, F: 0.4, PO4: 0.2, SiO2: 14, B: 0.2 } },
  produced: { name: 'Oil-field produced water (≈70 g/L)', T: 40, pH: 6.8, ions: { Na: 22500, K: 380, Ca: 3200, Mg: 520, Ba: 12, Sr: 180, NH4: 60, Fe: 8, Mn: 1, Cl: 42000, SO4: 320, HCO3: 480, CO3: 0.2, NO3: 0, F: 1, PO4: 0, SiO2: 35, B: 30 } },
  robrine: { name: 'SWRO concentrate (≈64 g/L, 45 % recovery)', T: 26, pH: 7.9, ions: { Na: 20020, K: 741, Ca: 765, Mg: 2388, Ba: 0.04, Sr: 14.7, NH4: 0, Fe: 0, Mn: 0, Cl: 35990, SO4: 5045, HCO3: 196, CO3: 26, NO3: 0.9, F: 2.3, PO4: 0.2, SiO2: 3.6, B: 7.2 } },
};

export const cloneIons = (ions) => Object.fromEntries(ION_IDS.map((k) => [k, +(ions?.[k] ?? 0)]));

/** Total dissolved solids, mg/L. */
export const tds = (ions) => ION_IDS.reduce((s, k) => s + (+ions[k] || 0), 0);

/** Molar concentration of one ion, mol/m³ (= mmol/L). */
export const molar = (ions, k) => (+ions[k] || 0) / IONS[k].mw;

/** Sum of molar concentrations of all dissolved species, mol/m³. */
export const totalMolar = (ions) => ION_IDS.reduce((s, k) => s + molar(ions, k), 0);

/** Ionic strength on the molar scale, mol/L. */
export const ionicStrength = (ions) => 0.5 * ION_IDS.reduce((s, k) => s + (molar(ions, k) / 1000) * IONS[k].z ** 2, 0);

/** Charge balance: { cations, anions } in meq/L and error in % of the total. */
export function chargeBalance(ions) {
  let cat = 0, an = 0;
  for (const k of ION_IDS) {
    const eq = molar(ions, k) * IONS[k].z;
    if (eq > 0) cat += eq; else an -= eq;
  }
  return { cations: cat, anions: an, errorPct: cat + an > 0 ? (100 * (cat - an)) / (cat + an) : 0 };
}

/** Enforce electroneutrality by adjusting Na (if anions exceed) or Cl (if cations exceed). */
export function balanceCharge(ions) {
  const out = cloneIons(ions), cb = chargeBalance(out), d = cb.cations - cb.anions;
  if (d > 0) out.Cl += d * IONS.Cl.mw; else out.Na += -d * IONS.Na.mw;
  return out;
}

/** Scale every ion by a factor (concentration factor, dilution, blending weight). */
export const scaleIons = (ions, f) => Object.fromEntries(ION_IDS.map((k) => [k, (+ions[k] || 0) * f]));

/** Flow-weighted blend of several streams: [{ ions, Q }]. */
export function mixIons(streams) {
  const Q = streams.reduce((s, x) => s + x.Q, 0) || 1;
  return Object.fromEntries(ION_IDS.map((k) => [k, streams.reduce((s, x) => s + (+x.ions[k] || 0) * x.Q, 0) / Q]));
}

/** Seawater-type composition scaled to a target TDS (mg/L). */
export const seawaterAtTDS = (target) => scaleIons(WATERS.seawater.ions, target / tds(WATERS.seawater.ions));

/**
 * Osmotic pressure of an arbitrary ionic composition, Pa.
 * Extended van't Hoff form: pi = phi · R · T · Σc, with the osmotic coefficient evaluated
 * at the equivalent seawater salinity (phi -> 1 at infinite dilution).
 */
export function osmoticPressureIons(ions, T = 25) {
  const S = salinityFromTDS(tds(ions), T);
  return osmoticCoefficient(T, S) * R * (T + KELVIN) * totalMolar(ions);
}

/** Ideal van't Hoff osmotic pressure (phi = 1), Pa. Used for verification limits. */
export const vantHoff = (ions, T = 25) => R * (T + KELVIN) * totalMolar(ions);

/** Electrical conductivity at T from limiting ionic conductances with an ionic-strength correction, µS/cm. */
export function conductivity(ions, T = 25) {
  const I = ionicStrength(ions);
  let k = 0;
  for (const id of ION_IDS) k += (molar(ions, id) / 1000) * Math.abs(IONS[id].z) * IONS[id].lambda; // S·cm²/L -> mS/cm ×1
  const corr = 1 / (1 + (0.75 * Math.sqrt(I)) / (1 + 0.6 * Math.sqrt(I)) + 0.06 * I);
  return k * 1000 * corr * (1 + 0.0191 * (T - 25));
}

/** Hardness as mg/L CaCO3 and alkalinity as mg/L CaCO3. */
export const hardness = (ions) => 50.04 * (2 * molar(ions, 'Ca') + 2 * molar(ions, 'Mg') + 2 * molar(ions, 'Sr') + 2 * molar(ions, 'Ba'));
export const alkalinity = (ions) => 50.04 * (molar(ions, 'HCO3') + 2 * molar(ions, 'CO3'));

/** One-call summary of a water analysis used by headers, KPI tiles and exports. */
export function summarize(ions, T = 25) {
  const t = tds(ions), S = salinityFromTDS(t, T), cb = chargeBalance(ions);
  return {
    tds: t, salinity: S, density: density(T, S), ionicStrength: ionicStrength(ions), osmoticBar: osmoticPressureIons(ions, T) / 1e5,
    conductivity: conductivity(ions, T), hardness: hardness(ions), alkalinity: alkalinity(ions), chargeErrorPct: cb.errorPct, cations: cb.cations, anions: cb.anions,
  };
}
