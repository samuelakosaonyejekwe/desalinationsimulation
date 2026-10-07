// Thermophysical properties of water, seawater/brine and steam.
// Conventions: T in °C, S = salinity in g/kg, p in Pa unless stated otherwise.
// Seawater correlations follow Sharqawy, Lienhard & Zubair (2010) and Nayar et al. (2016);
// saturation follows the IAPWS-IF97 region-4 equation.

export const R = 8.314462618; // J/(mol·K)
export const F = 96485.33212; // C/mol
export const G = 9.80665; // m/s²
export const KELVIN = 273.15;

/** Density of seawater, kg/m³ (0–180 °C, 0–160 g/kg). */
export function density(T, S = 0) {
  const s = S / 1000;
  const rw = 9.999e2 + 2.034e-2 * T - 6.162e-3 * T ** 2 + 2.261e-5 * T ** 3 - 4.657e-8 * T ** 4;
  return rw + s * (8.02e2 - 2.001 * T + 1.677e-2 * T ** 2 - 3.06e-5 * T ** 3) - 1.613e-5 * s * s * T * T;
}

/** Dynamic viscosity of seawater, Pa·s (0–180 °C, 0–150 g/kg). */
export function viscosity(T, S = 0) {
  const s = S / 1000;
  const mw = 4.2844e-5 + 1 / (0.157 * (T + 64.993) ** 2 - 91.296);
  const A = 1.541 + 1.998e-2 * T - 9.52e-5 * T * T, B = 7.974 - 7.561e-2 * T + 4.724e-4 * T * T;
  return mw * (1 + A * s + B * s * s);
}

/** Specific heat of seawater, J/(kg·K) (0–180 °C, 0–180 g/kg). */
export function cp(T, S = 0) {
  const K = T + KELVIN;
  const A = 5.328 - 9.76e-2 * S + 4.04e-4 * S * S, B = -6.913e-3 + 7.351e-4 * S - 3.15e-6 * S * S;
  const C = 9.6e-6 - 1.927e-6 * S + 8.23e-9 * S * S, D = 2.5e-9 + 1.666e-9 * S - 7.125e-12 * S * S;
  return 1000 * (A + B * K + C * K * K + D * K ** 3);
}

/** Thermal conductivity of seawater, W/(m·K). */
export function conductivityThermal(T, S = 0) {
  return (0.5706 + 1.756e-3 * T - 6.46e-6 * T * T) * (1 - 2.2e-4 * S);
}

/** Saturation pressure of pure water, Pa (IAPWS-IF97 region 4). */
export function psat(T) {
  const n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.1202082470247e5, -0.32325550322333e7, 0.1491510861353e2, -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3];
  const K = T + KELVIN, th = K + n[8] / (K - n[9]);
  const A = th * th + n[0] * th + n[1], B = n[2] * th * th + n[3] * th + n[4], C = n[5] * th * th + n[6] * th + n[7];
  return 1e6 * ((2 * C) / (-B + Math.sqrt(B * B - 4 * A * C))) ** 4;
}

/** Saturation temperature of pure water, °C, for pressure p in Pa. */
export function tsat(p) {
  const n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.1202082470247e5, -0.32325550322333e7, 0.1491510861353e2, -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3];
  const b = (p / 1e6) ** 0.25;
  const E = b * b + n[2] * b + n[5], Fq = n[0] * b * b + n[3] * b + n[6], Gq = n[1] * b * b + n[4] * b + n[7];
  const D = (2 * Gq) / (-Fq - Math.sqrt(Fq * Fq - 4 * E * Gq));
  return (n[9] + D - Math.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D))) / 2 - KELVIN;
}

/** Vapour pressure above seawater, Pa (Raoult-type salinity depression). */
export function psatSeawater(T, S = 0) {
  return psat(T) / (1 + 0.57357 * (S / (1000 - S)));
}

/** Antoine equation for water (1–100 °C), Pa. Kept for verification against the IAPWS form. */
export function antoine(T) {
  return 133.322368 * 10 ** (8.07131 - 1730.63 / (233.426 + T));
}

/** Latent heat of vaporisation, J/kg. */
export function latentHeat(T, S = 0) {
  return 1000 * (2501.897149 - 2.407064037 * T + 1.192217e-3 * T * T - 1.5863e-5 * T ** 3) * (1 - S / 1000);
}

/** Boiling-point elevation of seawater, K. */
export function bpe(T, S) {
  const s = S / 1000;
  return (-4.584e-4 * T * T + 2.823e-1 * T + 17.95) * s * s + (1.536e-4 * T * T + 5.267e-2 * T + 6.56) * s;
}

/** Specific enthalpy of seawater relative to 0 °C, J/kg (integrated mean cp). */
export function enthalpyLiquid(T, S = 0) {
  return 0.5 * (cp(0, S) + cp(T, S)) * T;
}

/** Specific enthalpy of saturated vapour at T, J/kg. */
export function enthalpyVapour(T) {
  return enthalpyLiquid(T, 0) + latentHeat(T, 0);
}

/** Saturated-vapour density by the ideal-gas law, kg/m³. */
export function vapourDensity(T) {
  return (psat(T) * 0.01801528) / (R * (T + KELVIN));
}

/** Osmotic coefficient of seawater (valid 0–200 °C, 10–120 g/kg; Debye–Hückel-style blend below 10 g/kg). */
export function osmoticCoefficient(T, S) {
  const f = (Sg) => {
    const s = Sg / 1000;
    return 8.9453e-1 + 4.1561e-4 * T - 4.6262e-6 * T * T + 2.2211e-11 * T ** 4 - 1.1445e-1 * s - 1.4783e-3 * s * T - 1.3526e-8 * s * T ** 3 + 7.0132 * s * s + 5.696e-2 * s * s * T - 2.8624e-4 * s * s * T * T;
  };
  if (S >= 10) return f(Math.min(S, 200));
  return 1 - (1 - f(10)) * Math.sqrt(Math.max(S, 0) / 10);
}

/** Osmotic pressure of seawater-type water from salinity, Pa. */
export function osmoticPressure(T, S) {
  const mTot = (31.843 * S) / (1000 - S); // mol of ions per kg of water
  return osmoticCoefficient(T, S) * R * (T + KELVIN) * density(T, 0) * mTot;
}

/** Diffusivity of NaCl in water, m²/s. */
export function diffusivityNaCl(T, S = 35) {
  return 6.725e-6 * Math.exp(0.1546e-3 * density(T, S) * (S / 1000) - 2513 / (T + KELVIN));
}

/** Practical conversion from TDS (mg/L) to electrical conductivity at 25 °C (µS/cm). */
export function conductivityFromTDS(tds) {
  if (tds <= 0) return 0;
  // ratio falls from ~0.5 (fresh) to ~0.7 (seawater and brines)
  const k = 0.5 + 0.2 * Math.min(1, Math.log10(1 + tds / 500) / Math.log10(1 + 35000 / 500));
  return tds / k;
}

/** Salinity in g/kg from TDS in mg/L (iterates on density). */
export function salinityFromTDS(tds, T = 25) {
  let S = tds / 1000;
  for (let i = 0; i < 6; i++) S = tds / density(T, S);
  return S;
}

/** TDS in mg/L from salinity in g/kg. */
export const tdsFromSalinity = (S, T = 25) => S * density(T, S);

/** Temperature-correction factor for membrane permeability relative to 25 °C (Arrhenius form). */
export function tcf(T, e = 2640) {
  return Math.exp(e * (1 / 298.15 - 1 / (T + KELVIN)));
}
