// Suite 8 — Forward osmosis, membrane distillation and emerging desalination processes.
// FO / PRO: solution–diffusion with internal and external concentration polarisation in both membrane
// orientations, reverse solute flux and a 1-D module model. MD (DCMD, AGMD, VMD, SGMD): coupled heat and
// mass transfer with the dusty-gas model, temperature and concentration polarisation, liquid-entry pressure
// and a 1-D module model. Hybrids: FO–RO and RO–MD chains; humidification–dehumidification as a
// reduced-order emerging process and capacitive deionisation with a porous-electrode (transmission-line) model. Further chains:
// FO–MD, electrodialysis–FO, MD–crystalliser and RO + capacitive polishing; optional pore-size distribution, batch dynamics with
// fouling, scaling and wetting, a material library, a Pareto sweep and a grey-box flux correction.
import { brent, solve1, newtonN, clamp, linspace, interp1, sum, rng, fmt, rk4, solveLinear } from '../core/num.js';
import { R, F, KELVIN, density, viscosity, cp, conductivityThermal, psat, tsat, psatSeawater, antoine, latentHeat, enthalpyLiquid as hL, diffusivityNaCl, salinityFromTDS, tdsFromSalinity } from '../core/props.js';
import { IONS, ION_IDS, WATERS, cloneIons, tds, scaleIons, osmoticPressureIons } from '../core/water.js';
import roSuite, { simulateRO } from './s01_ro.js';
import edSuite, { simulateED, gcs, mDonnan, pnpRamp, debyeLength } from './s07_ed.js';
import { nsga2 } from './s11_opt.js';

const MW_W = 0.018015, KB = 1.380649e-23, SIGMA_W = 2.641e-10, CPA = 1006, CPV = 1860, H0V = 2501e3;
const K = (T) => T + KELVIN;
const tFromH = (h, S) => { let T = h / cp(40, S); for (let i = 0; i < 14; i++) { const d = (hL(T, S) - h) / cp(T, S); T -= d; if (Math.abs(d) < 1e-11) break; } return T; };

// ---- draw solutes -------------------------------------------------------------------------------------------
/** Draw-solute library: van't Hoff number ν, osmotic coefficient φ = a₀ + a₁c + a₂c² (c in mol/L, 25 °C), diffusivity, permeability relative to NaCl. */
export const DRAWS = {
  nacl: { name: 'Sodium chloride NaCl', M: 58.44, nu: 2, phi: [0.917, 0.008, 0.0105], D: 1.48e-9, bRel: 1, sol: 5.4, ions: { Na: 22.99, Cl: 35.453 } },
  mgcl2: { name: 'Magnesium chloride MgCl₂', M: 95.21, nu: 3, phi: [0.86, 0.18, 0.07], D: 1.07e-9, bRel: 0.25, sol: 4.9, ions: { Mg: 24.305, Cl: 70.906 } },
  mgso4: { name: 'Magnesium sulphate MgSO₄', M: 120.37, nu: 2, phi: [0.55, -0.04, 0.05], D: 0.85e-9, bRel: 0.1, sol: 2.8, ions: { Mg: 24.305, SO4: 96.06 } },
  na2so4: { name: 'Sodium sulphate Na₂SO₄', M: 142.04, nu: 3, phi: [0.76, -0.14, 0.03], D: 1.23e-9, bRel: 0.15, sol: 1.9, ions: { Na: 45.98, SO4: 96.06 } },
  cacl2: { name: 'Calcium chloride CaCl₂', M: 110.98, nu: 3, phi: [0.86, 0.13, 0.065], D: 1.34e-9, bRel: 0.4, sol: 6.7, ions: { Ca: 40.078, Cl: 70.906 } },
  nh4hco3: { name: 'Ammonium bicarbonate NH₄HCO₃ (thermolytic)', M: 79.06, nu: 2, phi: [0.9, -0.08, 0.01], D: 1.4e-9, bRel: 2, sol: 2.7, ions: { NH4: 18.039, HCO3: 61.017 } },
  glucose: { name: 'Glucose (direct-use / fertigation type)', M: 180.16, nu: 1, phi: [1, 0.03, 0], D: 0.67e-9, bRel: 0.02, sol: 5, ions: null },
};
/** Osmotic pressure of a draw solution, Pa (c in mol/m³). */
export function drawOsmotic(key, c, T = 25) {
  const d = DRAWS[key], cm = Math.max(c, 0) / 1000;
  return d.nu * (d.phi[0] + d.phi[1] * cm + d.phi[2] * cm * cm) * Math.max(c, 0) * R * K(T);
}

/**
 * Ion analysis (mg/L) that represents a draw solution of concentration c (mol/m³) in the ion-based RO, ED and MD models. Ionic draws: their own ions.
 * Non-ionic draws (glucose): the osmotically equivalent NaCl solution — equal osmotic pressure, hence equal water activity, vapour-pressure lowering
 * and RO driving pressure; the solute itself stays on the draw side (it is neither volatile nor charged).
 */
export function drawAsIons(key, c, T = 25) {
  const d = DRAWS[key];
  if (d.ions) return { ions: cloneIons(Object.fromEntries(Object.entries(d.ions).map(([k, m]) => [k, m * c]))), equivalent: false, cEq: c };
  const pi = drawOsmotic(key, c, T), cEq = pi > 0 ? brent((x) => drawOsmotic('nacl', x, T) - pi, 0, 8000, 1e-9, 100) : 0;
  return { ions: cloneIons(Object.fromEntries(Object.entries(DRAWS.nacl.ions).map(([k, m]) => [k, m * cEq]))), equivalent: true, cEq };
}

/** Spacer-channel mass transfer (same correlation family as the RO suite): Sh = 0.065 Re^0.875 Sc^0.25. */
function massTransfer(u, h, T, S, D, mult = 1) {
  const rho = density(T, S), mu = viscosity(T, S), eps = 0.89, dh = (4 * eps) / (2 / h + ((1 - eps) * 8) / h), Re = (rho * u * dh) / mu, Sc = mu / (rho * D);
  return { k: Math.max(1e-7, (mult * 0.065 * Math.max(Re, 1) ** 0.875 * Sc ** 0.25 * D) / dh), Re, dh, dpPerM: (6.23 * Math.max(Re, 1) ** -0.3 * rho * u * u) / (2 * dh) };
}

/**
 * Local FO / PRO water and solute fluxes with ECP and ICP.
 * st = { f (feed concentration factor), cFd (draw solute in feed, mol/m³), cD (draw, mol/m³) };
 * m = { A (m/s·Pa), Bd, Bf (m/s), KFf, KFd, KD (s/m), piF(f), piD(c), dP (Pa on the draw side) }.
 */
export function foFlux(st, m) {
  const ex = (x) => Math.exp(clamp(x, -60, 60));
  const at = (Jw) => {
    const EFf = ex(Jw * m.KFf), EFd = ex(Jw * m.KFd), ED = ex(-Jw * m.KD);
    const dc = (st.cD * ED - st.cFd * EFd) / (1 + (m.Bd / Jw) * (EFd - ED)), Js = m.Bd * dc; // reverse draw-solute flux, mol/m²·s
    const cDm = st.cD * ED - (Js / Jw) * (1 - ED), cFdm = st.cFd * EFd + (Js / Jw) * (EFd - 1), fm = (st.f * EFf) / (1 + (m.Bf / Jw) * (EFf - 1));
    const piDm = m.piD(cDm), piFm = m.piF(fm), piR = m.piD(cFdm);
    return { Jw, Js, cDm, cFdm, fm, piDm, piFm, piR, Jf: m.Bf * fm, res: m.A * (piDm - piFm - piR - m.dP) - Jw };
  };
  const piDb = m.piD(st.cD), piFb = m.piF(st.f) + m.piD(st.cFd), top = m.A * (piDb - piFb - m.dP);
  if (!(top > 1e-13)) return { Jw: 0, Js: m.Bd * Math.max(0, st.cD - st.cFd) / (1 + m.Bd * (m.KD + m.KFd)), cDm: st.cD, cFdm: st.cFd, fm: st.f, piDm: piDb, piFm: m.piF(st.f), piR: m.piD(st.cFd), Jf: m.Bf * st.f, piDb, piFb, res: 0 };
  const lo = top * 1e-9; let hi = top, g = 0;
  if (at(lo).res > 0) while (at(hi).res > 0 && g++ < 40) hi *= 1.25; // the root lies below A·Δπ_bulk except when a feed-side pressure outruns the osmotic terms
  const Jw = at(lo).res <= 0 ? lo : at(hi).res > 0 ? hi : brent((x) => at(x).res, lo, hi, 1e-12 * top, 100);
  return { ...at(Jw), piDb, piFb };
}

const PI_X = [...linspace(0, 0.1, 9), ...linspace(0.125, 1, 36)], piCache = { key: '', pis: [], fs: [] }; // feed osmotic pressure versus concentration factor (memoised table up to about 300 g/L)
function foSetup(v, ov = {}) {
  const p = { ...v, ...ov }, T = p.T, d = DRAWS[p.draw] || DRAWS.nacl, ions = scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1), tdsF = tds(ions), S = salinityFromTDS(tdsF, T);
  const key = `${T}|${ION_IDS.map((k) => ions[k]).join(',')}`;
  if (piCache.key !== key) { const fmax = Math.max(4, 3e5 / Math.max(tdsF, 1e-6)); piCache.key = key; piCache.fs = PI_X.map((x) => x * fmax); piCache.pis = piCache.fs.map((f) => (f > 0 ? osmoticPressureIons(scaleIons(ions, f), T) : 0)); }
  const fs = piCache.fs, pis = piCache.pis, nP = fs.length, slope = (pis[nP - 1] - pis[nP - 2]) / (fs[nP - 1] - fs[nP - 2]);
  const piF = (f) => (f > fs[nP - 1] ? pis[nP - 1] + slope * (f - fs[nP - 1]) : interp1(fs, pis, Math.max(f, 0))); // linear extrapolation beyond the table
  const Df = diffusivityNaCl(T, S), Dd = d.D * (K(T) / 298.15) * (viscosity(25, 0) / viscosity(T, 0)), h = p.hch / 1000, tc = Math.exp(2000 * (1 / 298.15 - 1 / K(T)));
  const mF = massTransfer(p.uF / 100, h, T, S, Df, p.kcp), mD = massTransfer(p.uD / 100, h, T, 40, Dd, p.kcp), kFd = mF.k * (Dd / Df) ** 0.75, Sp = (p.Sfo * 1e-6) / clamp((p.hydration ?? 100) / 100, 0.2, 1), pro = p.process === 'pro', alds = pro || p.orient === 'alds'; // a partly wetted support layer has a proportionally larger structural parameter
  const m = { A: (p.AFO * tc) / 3.6e6 / 1e5, Bf: (p.BFO * tc) / 3.6e6, Bd: (p.BFO * tc * d.bRel) / 3.6e6, piF, piD: (c) => drawOsmotic(p.draw, c, T), dP: pro ? p.dPpro * 1e5 : -(p.dPfeed || 0) * 1e5, // pressure-assisted FO: hydraulic pressure on the feed side adds to the osmotic driving force
    KFf: alds ? Sp / Df + 1 / mF.k : 1 / mF.k, KFd: alds ? Sp / Dd + 1 / kFd : 1 / kFd, KD: alds ? 1 / mD.k : Sp / Dd + 1 / mD.k };
  return { p, T, d, ions, tdsF, S, m, mF, mD, Df, Dd, alds, pro };
}

/** Module-scale FO / PRO: 1-D co- or counter-current model along the membrane (N segments, midpoint rule). */
export function simulateFO(v, ov = {}) {
  const s = foSetup(v, ov), { p, m, d } = s, N = Math.max(2, Math.round(p.nSeg)), area = p.areaFO, dA = area / N, QF0 = p.Qf / 3600, QD0 = p.Qd / 3600, cD0 = p.cDraw * 1000, counter = p.flow === 'counter';
  const sgn = counter ? -1 : 1, state = (f, dd) => ({ f: f.nf / f.Q, cFd: f.nd / f.Q, cD: Math.max(dd.n, 0) / dd.Q });
  const adv = (f, dd, fl, w) => [{ Q: f.Q - w * fl.Jw * dA, nf: f.nf - w * fl.Jf * dA, nd: f.nd + w * fl.Js * dA }, { Q: dd.Q + sgn * w * fl.Jw * dA, n: dd.n - sgn * w * fl.Js * dA, nf: dd.nf + sgn * w * fl.Jf * dA }];
  /** One pass along the feed path; for counter-current flow g holds the guessed totals that fix the draw state at the feed inlet. */
  const sweep = (g) => {
    let F = { Q: QF0, nf: QF0, nd: 0 }, D = counter ? { Q: QD0 + g.Vw, n: QD0 * cD0 - g.Sd, nf: g.Sf } : { Q: QD0, n: QD0 * cD0, nf: 0 }; // feed: flow, feed-solute "factor flow", draw-solute moles
    const segs = [], run = { Vw: 0, Sd: 0, Sf: 0 };
    for (let k = 0; k < N; k++) {
      const f1 = foFlux(state(F, D), m), [Fm, Dm] = adv(F, D, f1, 0.5), stM = state(Fm, Dm), f2 = foFlux(stM, m);
      f2.Jw = Math.min(f2.Jw, (0.5 * F.Q) / dA);
      segs.push({ x: (k + 0.5) / N, ...f2, f: stM.f, cD: stM.cD, cFd: stM.cFd, QF: Fm.Q, QD: Dm.Q });
      [F, D] = adv(F, D, f2, 1);
      run.Vw += f2.Jw * dA; run.Sd += f2.Js * dA; run.Sf += f2.Jf * dA;
    }
    return { F, D, segs, run };
  };
  let sw = sweep({ Vw: 0, Sd: 0, Sf: 0 }), it = 0, conv = !counter;
  if (counter) { // secant on the transferred water; the (weakly coupled) solute totals follow by substitution
    const vmax = 0.98 * QF0;
    let x0 = 0, h0 = sw.run.Vw, x1 = Math.min(sw.run.Vw, vmax), last = sw.run;
    for (; it < 80; it++) {
      sw = sweep({ Vw: x1, Sd: last.Sd, Sf: last.Sf });
      const h1 = sw.run.Vw - x1, err = Math.abs(h1) / (Math.abs(sw.run.Vw) + 1e-30) + Math.abs(sw.run.Sd - last.Sd) / (Math.abs(sw.run.Sd) + 1e-30);
      last = sw.run;
      if (err < 1e-11) { conv = true; break; }
      const x2 = Math.abs(h1 - h0) > 1e-300 ? x1 - (h1 * (x1 - x0)) / (h1 - h0) : sw.run.Vw;
      x0 = x1; h0 = h1; x1 = Number.isFinite(x2) ? clamp(x2, 0, vmax) : Math.min(sw.run.Vw, vmax);
      if (x1 === x0) x1 = Math.min(sw.run.Vw, vmax);
    }
    for (let k = 0; !conv && k < 8; k++) { // fallback when osmotic equilibrium inside the module makes the residual non-smooth: bracketing on a monotone residual
      const fr = last, x = solve1((v) => sweep({ Vw: v, Sd: fr.Sd, Sf: fr.Sf }).run.Vw - v, 0, vmax, 1e-14 * QF0);
      sw = sweep({ Vw: x, Sd: fr.Sd, Sf: fr.Sf }); it++;
      conv = Math.abs(sw.run.Sd - fr.Sd) <= 1e-9 * Math.abs(sw.run.Sd) + 1e-30 && Math.abs(sw.run.Vw - x) <= 1e-8 * QF0;
      last = sw.run;
    }
  }
  const tot = sw.run, segs = sw.segs, F = sw.F, D = sw.D;
  const drawOut = counter ? { Q: QD0 + tot.Vw, n: QD0 * cD0 - tot.Sd, nf: tot.Sf } : D, drawInEnd = counter ? D : { Q: QD0, n: QD0 * cD0, nf: 0 };
  const Jw = tot.Vw / area, Js = tot.Sd / area, cDout = drawOut.n / drawOut.Q, fOut = F.nf / F.Q;
  const dpF = s.mF.dpPerM * p.Lfo, dpD = s.mD.dpPerM * p.Lfo, Ppump = (QF0 * (dpF + (s.pro ? 0 : (p.dPfeed || 0) * 1e5)) + QD0 * dpD) / (p.etaPump / 100);
  return { ...s, N, area, segs, tot, conv, iterations: it, QF0, QD0, cD0, feedOut: F, drawOut, drawInEnd, Jw, Js, JwLMH: Jw * 3.6e6, JsGMH: Js * d.M * 3600, srsf: Jw > 0 ? (Js * d.M) / Jw / 1000 : 0, cDout, fOut,
    recovery: tot.Vw / QF0, dilution: cD0 / cDout, dpF, dpD, Ppump, powerDensity: s.pro ? sum(segs.map((g) => g.Jw)) / N * m.dP : 0,
    closure: counter ? Math.abs(drawInEnd.Q - QD0) / QD0 + Math.abs(drawInEnd.n - QD0 * cD0) / (QD0 * cD0) : 0 };
}

/** Energy to re-concentrate the diluted draw solution, per m³ of product water. */
export function regeneration(r, p) {
  const d = r.d, piC = drawOsmotic(p.draw, r.cD0, r.T) / 1e5, rec = clamp(r.tot.Vw / r.drawOut.Q, 0.01, 0.95);
  if (p.regen === 'ro') {
    const P = piC * 1.08 + 4, e = p.etaPump / 100, erd = p.etaERD / 100; // pressure: concentrate osmotic pressure + net driving and friction allowance
    return { type: 'RO / NF', elec: (P * 1e5 * (1 - erd * (1 - rec))) / (rec * e) / 3.6e6, heat: 0, P, rec, feasible: P <= 83, note: `Draw re-concentrated at ${fmt(P, 3)} bar and ${fmt(100 * rec, 3)} % recovery with isobaric energy recovery.` };
  }
  if (p.regen === 'thermolytic') return { type: 'Thermolytic stripping', elec: 0.3, heat: (r.cDout * p.hStrip * 1000) / 3.6e6, P: 0, rec, feasible: p.draw === 'nh4hco3', note: `${p.hStrip} kJ per mole of draw solute stripped from the diluted draw (${fmt(r.cDout / 1000, 3)} mol/L).` };
  if (p.regen === 'thermal') return { type: 'Distillation / MD', elec: 1.2, heat: (latentHeat(60) * density(25, 0)) / p.gorRegen / 3.6e6, P: 0, rec, feasible: true, note: `Evaporative re-concentration with a gain-output ratio of ${p.gorRegen}.` };
  return { type: 'None (diluted draw used directly)', elec: 0, heat: 0, P: 0, rec, feasible: true, note: 'The diluted draw solution is the product (fertigation, osmotic dilution before RO, emergency drinks).' };
}

// ---- membrane distillation ---------------------------------------------------------------------------------
/** Mean free path of water vapour, m (T in °C, P in Pa). */
export const meanFreePath = (T, P) => (KB * K(T)) / (Math.SQRT2 * Math.PI * SIGMA_W ** 2 * P);
/** Surface tension of water, N/m (IAPWS), reduced by an optional factor for surfactants/organics. */
export const surfaceTension = (T, f = 1) => { const t = 1 - K(T) / 647.096; return f * 0.2358 * t ** 1.256 * (1 - 0.625 * t); };
/** Liquid-entry pressure (Laplace–Cantor), Pa: LEP = −2·B·γ·cosθ / r_max. */
export const liquidEntryPressure = (rmax, thetaDeg, gamma, B = 1) => (-2 * B * gamma * Math.cos((thetaDeg * Math.PI) / 180)) / rmax;

/**
 * Dusty-gas membrane coefficients, kg/(m²·s·Pa). mem = { r (mean pore radius, m), eps, tau, delta (m) };
 * T mean membrane temperature (°C), P total pressure in the pores (Pa), pv mean vapour pressure (Pa).
 */
export function dustyGas(mem, T, P, pv, model = 'auto', vacuum = false) {
  const Tk = K(T), g = mem.eps / (mem.tau * mem.delta), Kn = meanFreePath(T, P) / (2 * mem.r);
  const Bk = ((2 * g * mem.r) / 3) * Math.sqrt((8 * MW_W) / (Math.PI * R * Tk));
  const PD = 1.895e-5 * Tk ** 2.072, pa = Math.max(P - pv, 0.02 * P), Bd = (g * PD * MW_W) / (pa * R * Tk); // P·D of water vapour in air, Pa·m²/s
  const Bv = (g * mem.r * mem.r * MW_W * Math.max(pv, 1)) / (8 * 1.0e-5 * (Tk / 300) * R * Tk);
  const regime = Kn > 1 ? 'Knudsen' : Kn > 0.01 ? 'transition' : vacuum ? 'viscous' : 'molecular';
  let B;
  if (vacuum) B = model === 'knudsen' || (model === 'auto' && Kn > 1) ? Bk : Bk + Bv;
  else if (model === 'knudsen' || (model === 'auto' && Kn > 1)) B = Bk;
  else if (model === 'molecular' || (model === 'auto' && Kn < 0.01)) B = Bd;
  else B = 1 / (1 / Bk + 1 / Bd);
  return { B, Bk, Bd, Bv, Kn, regime };
}

/** Channel heat and mass transfer for MD: spacer-filled (Nu = a·Re^0.5·Pr^⅓) or open laminar/turbulent channel. */
function mdChannel(u, h, T, S, L, c) {
  const rho = density(T, S), mu = viscosity(T, S), k = conductivityThermal(T, S), cpl = cp(T, S), dh = c.spacer ? (4 * 0.8) / (2 / h + (0.2 * 8) / h) : 2 * h, Re = (rho * u * dh) / mu, Pr = (cpl * mu) / k, D = diffusivityNaCl(T, Math.min(S, 150)), Sc = mu / (rho * D);
  const nu = (x) => (c.spacer ? c.nuA * Math.max(Re, 1) ** 0.5 * x ** (1 / 3) : Re < 2100 ? Math.max(4.36, 1.86 * ((Re * x * dh) / L) ** (1 / 3)) : 0.023 * Re ** 0.8 * x ** 0.33);
  return { h: (c.fh * nu(Pr) * k) / dh, kMass: (c.fh * nu(Sc) * D) / dh, Re, Pr, dh, rho, dpPerM: ((c.spacer ? 6.23 * Math.max(Re, 1) ** -0.3 : Re < 2100 ? 96 / Math.max(Re, 1) : 0.316 * Re ** -0.25) * rho * u * u) / (2 * dh) };
}

/**
 * Local MD heat and mass balance at one position. c holds the configuration (see mdConfig); cold = { T } for
 * DCMD/AGMD coolant, { T, pg } for the sweep gas; returns flux N (kg/m²·s), interface temperatures and heat fluxes.
 */
export function mdLocal(c, Tf, S, cold) {
  const type = c.type, P = c.P, kel = c.kelvin ? kelvinFactor(Tf, c.mem.r, c.theta) : 1;
  const pw = c.antoine ? antoine : psat;
  // fz = { N, dg } frozen flux (for concentration polarisation) and dusty-gas coefficients; null → evaluate them here
  const side = (Tfm, fz) => {
    const Sm = Math.min(S * Math.exp(clamp((fz ? fz.N : 0) / (c.rhoF * c.kMass), -3, 3)), 350), pF = (c.antoine ? antoine(Tfm) * (psatSeawater(Tfm, Sm) / psat(Tfm)) : psatSeawater(Tfm, Sm)) * kel; // pure-water curve × activity of the brine
    let Tc, pP, B, qc, dg = fz ? fz.dg : null;
    if (type === 'vmd') { Tc = tsat(c.Pv); pP = c.Pv; dg ||= memCoeff(c.mem, Tfm, 0.5 * (pF + c.Pv), 0.5 * (pF + c.Pv), c.model, true); B = dg.B; qc = 0; }
    else if (type === 'agmd') {
      Tc = cold.T + (c.hf / c.hc) * (Tf - Tfm); pP = pw(Tc);
      const Tm = 0.5 * (Tfm + Tc), pm = 0.5 * (pF + pP); dg ||= memCoeff(c.mem, Tm, P, pm, c.model);
      const Bgap = (1.895e-5 * K(Tm) ** 2.072 * MW_W) / (Math.max(P - pm, 0.02 * P) * R * K(Tm) * c.gap);
      B = 1 / (1 / dg.B + 1 / Bgap); qc = (Tfm - Tc) / (1 / c.hm + c.gap / 0.027);
    } else if (type === 'sgmd') { Tc = (c.hm * Tfm + c.hg * cold.T) / (c.hm + c.hg); pP = cold.pg; dg ||= memCoeff(c.mem, 0.5 * (Tfm + Tc), P, 0.5 * (pF + pP), c.model); B = 1 / (1 / dg.B + 1 / c.Bg); qc = c.hm * (Tfm - Tc); }
    else { Tc = cold.T + (c.hf / c.hp) * (Tf - Tfm); pP = pw(Tc); dg ||= memCoeff(c.mem, 0.5 * (Tfm + Tc), P, 0.5 * (pF + pP), c.model); B = dg.B; qc = c.hm * (Tfm - Tc); }
    const N = B * (pF - pP), q = c.hf * (Tf - Tfm);
    return { N, Tc, pF, pP, B, qc, Sm, dg, Tfm, q, res: q - N * latentHeat(Tfm) - qc };
  };
  const Tother = type === 'vmd' ? Math.min(tsat(c.Pv), Tf) - 5 : cold.T, Tlow = Math.min(Tother, Tf), Thigh = Math.max(Tother, Tf); // the membrane-surface temperature lies between the two bulk temperatures
  let fz = side(0.5 * (Tf + Tlow), null), Tfm = Tf; // first estimate of flux and membrane coefficient
  if (Thigh - Tlow > 1e-9) for (let pass = 0; pass < 3; pass++) {
    const f = (x) => side(x, fz).res;
    Tfm = pass ? solve1(f, Math.max(Tlow, Tfm - 0.5), Math.min(Thigh, Tfm + 0.5), 1e-11) : solve1(f, Tlow, Thigh, 1e-4);
    if (pass < 2) fz = side(Tfm, { N: side(Tfm, fz).N, dg: null }); // refresh the frozen coefficients at the new temperatures
  }
  else fz = side(Tf, { N: side(Tf, fz).N, dg: null });
  const e = side(Tfm, fz), lat = e.N * latentHeat(Tfm);
  return { ...e, lat, eta: e.q > 0 ? clamp(lat / e.q, 0, 1) : 0, tpc: Thigh - Tlow < 1e-9 ? 1 : type === 'vmd' ? (Tfm - e.Tc) / Math.max(Tf - e.Tc, 1e-9) : (Tfm - e.Tc) / Math.max(Tf - cold.T, 1e-9) };
}

function mdConfig(p, Tf, Tp, S) {
  const mem = { r: (p.dPore * 1e-6) / 2, eps: p.epsM, tau: p.tauM, delta: p.deltaM * 1e-6, sg: p.poreDist === 'lognormal' || p.poreDist === 'network' ? Math.max(p.sigmaPore, 1) : 1, net: p.poreDist === 'network' ? { nx: p.pnNx ?? 24, nz: p.pnNz ?? 12, seed: Math.round(p.pnSeed ?? 7), theta: p.theta, gamma: surfaceTension(Tf, p.gammaF ?? 1), dP: Math.max(0, (p.pFeed ?? 0) * 1e5), B: p.lepB ?? 1 } : null }, hF = p.hF / 1000, L = p.Lmd, opt = { spacer: !!p.spacerMD, nuA: p.nuA, fh: p.fh };
  const chF = mdChannel(p.uFm, hF, 0.5 * (Tf + Tp) + 0.25 * (Tf - Tp), S, L, opt), chP = mdChannel(p.uPm, hF, Tp + 0.25 * (Tf - Tp), 0, L, opt), km = mem.eps * 0.027 + (1 - mem.eps) * p.kPoly;
  const P = 101325, hg = (7.54 * 0.027) / (2 * hF) * p.fh; // laminar parallel-plate Nusselt number for the sweep gas
  return { type: p.mdType, mem, P, model: p.mdModel, kelvin: !!p.kelvin, antoine: p.vpModel === 'antoine', theta: p.theta, hf: chF.h, hp: chP.h, hm: km / mem.delta, km, kMass: chF.kMass, rhoF: chF.rho, chF, chP,
    gap: p.gapMD / 1000, hc: 1 / (1 / 5000 + p.plateT / 1000 / p.plateK + 1 / chP.h), Pv: p.Pvac * 1000, hg, Bg: (0.622 * hg) / (CPA * P), hF, L };
}

/** Module-scale MD along the membrane (per metre of width): feed and cold stream in counter- or co-current flow. */
export function mdModule(p, S0, ov = {}) {
  const q = { ...p, ...ov }, Tf0 = q.Tf, Tp0 = q.Tp, c = mdConfig(q, Tf0, Tp0, S0), N = Math.max(2, Math.round(q.nSeg)), dA = c.L / N, type = c.type, counter = q.flowMD === 'counter' && (type === 'dcmd' || type === 'agmd');
  const mf0 = density(Tf0, S0) * q.uFm * c.hF, salt = (mf0 * S0) / 1000, mc0 = type === 'sgmd' ? (101325 / (287.05 * K(Tp0))) * q.uGas * c.hF : density(Tp0, 0) * q.uPm * c.hF;
  const w0 = type === 'sgmd' ? (0.622 * (q.rhGas / 100) * psat(Tp0)) / (101325 - (q.rhGas / 100) * psat(Tp0)) : 0, hGas = (T, w) => CPA * T + w * (H0V + CPV * T), tGas = (h, w) => (h - w * H0V) / (CPA + CPV * w);
  const march = (Tout, Ntot) => {
    const Tc0 = counter ? Tout : Tp0, sg = counter ? -1 : 1, segs = [];
    let s = { mf: mf0, Hf: mf0 * hL(Tf0, S0), mc: type === 'dcmd' && counter ? mc0 + Ntot : mc0, w: counter ? w0 + Ntot / mc0 : w0, Hc: 0 }, Nsum = 0, qsum = 0, lat = 0, Hd = 0;
    s.Hc = type === 'sgmd' ? mc0 * hGas(Tc0, s.w) : s.mc * hL(Tc0, 0);
    const bulkT = (x) => (type === 'sgmd' ? tGas(x.Hc / mc0, x.w) : type === 'vmd' ? tsat(c.Pv) : tFromH(x.Hc / x.mc, 0));
    const local = (x) => {
      const S = Math.min((1000 * salt) / x.mf, 350), Tf = tFromH(x.Hf / x.mf, S), Tb = bulkT(x), pg = type === 'sgmd' ? Math.min((x.w * 101325) / (0.622 + x.w), psat(Tb)) : 0, lo = mdLocal(c, Tf, S, { T: Tb, pg });
      return { S, Tf, Tb, lo, flow: lo.q + lo.N * hL(lo.Tfm, 0) }; // flow = enthalpy leaving the feed per m²
    };
    const adv = (x, e, f) => {
      const o = { ...x, Hf: x.Hf - f * e.flow * dA, mf: x.mf - f * e.lo.N * dA };
      if (type === 'dcmd') { o.Hc += sg * f * e.flow * dA; o.mc += sg * f * e.lo.N * dA; }
      else if (type === 'agmd') o.Hc += sg * f * (e.flow - e.lo.N * hL(e.lo.Tc, 0)) * dA;
      else if (type === 'sgmd') { o.Hc += sg * f * e.flow * dA; o.w += (sg * f * e.lo.N * dA) / mc0; }
      return o;
    };
    for (let k = 0; k < N; k++) { // midpoint rule
      const e = local(adv(s, local(s), 0.5));
      segs.push({ ...e.lo, x: (k + 0.5) * dA, Tf: e.Tf, Tb: e.Tb, S: e.S });
      s = adv(s, e, 1); Nsum += e.lo.N * dA; qsum += e.lo.q * dA; lat += e.lo.N * latentHeat(e.lo.Tfm) * dA; Hd += type === 'agmd' ? e.lo.N * hL(e.lo.Tc, 0) * dA : 0;
      if (!(s.mf > 0.02 * mf0) || !Number.isFinite(s.Hc) || !Number.isFinite(s.Hf)) break;
    }
    const Sout = (1000 * salt) / s.mf;
    return { segs, Nsum, qsum, lat, Hd, mf: s.mf, Hf: s.Hf, Sout, TfOut: tFromH(s.Hf / s.mf, Sout), TcEnd: bulkT(s), mcEnd: s.mc, wEnd: s.w, Hc: s.Hc };
  };
  let res, Tout = Tp0, Ntot = 0, shots = 0;
  if (!counter) res = march(Tp0, 0);
  else { // shooting on the cold-stream outlet temperature (secant, with a bracketing fallback) and on the transferred mass
    const g = (T) => { shots++; res = march(T, Ntot); return res.TcEnd - Tp0; };
    let t0 = Tp0 + 0.5 * (Tf0 - Tp0), g0 = g(t0), t1 = clamp(t0 - g0, Tp0, Tf0), ok = false;
    for (let it = 0; it < 40; it++) {
      const g1 = g(t1);
      if (Math.abs(g1) < 1e-9 && Math.abs(res.Nsum - Ntot) <= 1e-9 * Math.abs(res.Nsum) + 1e-16) { ok = true; break; }
      Ntot = res.Nsum;
      const t2 = Math.abs(g1 - g0) > 1e-14 ? t1 - (g1 * (t1 - t0)) / (g1 - g0) : t1 - g1;
      t0 = t1; g0 = g1; t1 = Number.isFinite(t2) ? clamp(t2, Tp0 - 1, Tf0 + 1) : 0.5 * (Tp0 + Tf0);
    }
    if (!ok) for (let it = 0; it < 12; it++) { t1 = solve1(g, Tp0, Tf0, 1e-10); g(t1); if (Math.abs(res.Nsum - Ntot) <= 1e-10 * Math.abs(res.Nsum) + 1e-16) break; Ntot = res.Nsum; }
    Tout = t1;
  }
  const coldIn = counter ? res.TcEnd : Tp0, coldOut = counter ? Tout : res.TcEnd, mcOut = type === 'dcmd' ? mc0 + res.Nsum : mc0;
  // energy accounting per metre of width (W/m)
  const HfIn = mf0 * hL(Tf0, S0), Hcold = type === 'sgmd' ? mc0 * (hGas(coldOut, counter ? w0 + res.Nsum / mc0 : res.wEnd) - hGas(Tp0, w0)) : type === 'vmd' ? 0 : mcOut * hL(coldOut, 0) - mc0 * hL(Tp0, 0);
  const cpf = cp(Tf0, S0), Qsens = mf0 * cpf * (Tf0 - res.TfOut); // heat needed to bring the returned feed back to the inlet temperature
  let Qin = Qsens, Qrec = 0;
  if (type === 'agmd' && q.recover) Qin = Math.min(Qsens, mf0 * cpf * Math.max(Tf0 - coldOut, 0.1)); // coolant is the feed itself: only the top-up heat is supplied
  else if (type === 'dcmd' && q.recover) { Qrec = (q.effHX / 100) * Math.min(mf0 * cpf, mcOut * cp(coldOut, 0)) * Math.max(0, coldOut - res.TfOut); Qin = Math.max(Qsens - Qrec, 0.02 * Qsens); }
  const flux = res.Nsum / c.L, lam = latentHeat(0.5 * (Tf0 + Tp0)), dpF = c.chF.dpPerM * c.L, dpP = type === 'sgmd' ? 0.3 * c.chP.dpPerM * c.L * 0.02 : c.chP.dpPerM * c.L;
  let Wpump = ((mf0 / c.chF.rho) * dpF + (type === 'vmd' ? 0 : type === 'sgmd' ? (mc0 / 1.15) * 800 : (mc0 / 998) * dpP)) / (q.etaPump / 100);
  if (type === 'vmd') Wpump += (0.005 * res.Nsum * 287.05 * K(Tp0) * Math.log(101325 / c.Pv)) / 0.4 + ((res.lat / (cp(Tp0, 0) * 8)) / 998) * 5e4 / (q.etaPump / 100); // vacuum pump for leaked air + condenser cooling-water pumping
  return { c, q, N, segs: res.segs, flux, fluxLMH: flux * 3600, Nsum: res.Nsum, mf0, mc0, S0, Sout: res.Sout, TfOut: res.TfOut, coldIn, coldOut, eta: res.qsum > 0 ? res.lat / res.qsum : 0, tpc: sum(res.segs.map((s) => s.tpc)) / res.segs.length,
    cpc: Math.max(...res.segs.map((s) => s.Sm / s.S)), Qsens, Qin, Qrec, gor: (res.Nsum * lam) / Qin, stec: Qin / res.Nsum / 3.6e6 * 1000, Wpump, sec: Wpump / res.Nsum / 3.6e6 * 1000, recovery: res.Nsum / mf0, counter, shots, dpF,
    balance: { in: HfIn - res.Hf, out: type === 'vmd' ? res.qsum + sum(res.segs.map((s) => s.N * hL(s.Tfm, 0))) * dA : Hcold + res.Hd }, matchErr: counter ? Math.abs(res.TcEnd - Tp0) : 0, SmMax: Math.max(...res.segs.map((s) => s.Sm)) };
}

/** MD plant sizing for a make-up feed Qf (m³/h) and loop recovery: the modules see the bleed (loop) salinity. */
export function simulateMD(v, ov = {}) {
  const p = { ...v, ...ov }, ions = scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1), tdsF = tds(ions), Sfeed = salinityFromTDS(tdsF, 25), rec = clamp(p.mdRec / 100, 0, 0.95);
  const Sloop = Math.min(Sfeed / (1 - rec), 330), m = mdModule(p, rec > 0 ? Sloop : Sfeed), make = (p.Qf * density(25, Sfeed)) / 3600, prod = rec > 0 ? rec * make : m.recovery * make; // kg/s
  const width = rec > 0 ? prod / Math.max(m.Nsum, 1e-12) : make / m.mf0, area = width * m.c.L, Qheat = m.Qin * width + (rec > 0 ? make * cp(p.T, Sfeed) * Math.max(0, m.TfOut - p.T) * (m.q.recover ? 0.3 : 1) : 0);
  const gamma = surfaceTension(p.Tf, p.gammaF), lep = liquidEntryPressure(p.rMax * 1e-6, p.theta, gamma, p.lepB);
  const cfw = m.SmMax / Math.max(Sfeed, 1e-9), sat = { nacl: m.SmMax / 264, gypsum: (ions.Ca * ions.SO4 * cfw * cfw) / (421 * 2776 * 3.3 * 3.3) }; // screening: standard seawater reaches gypsum saturation near a concentration factor of 3.3
  return { p, ions, tdsF, Sfeed, Sloop, rec, m, make, prod, width, area, Qheat, lep, gamma, sat, sth: Qheat / prod / 3.6e6 * 1000, sel: (m.Wpump * width) / prod / 3.6e6 * 1000, gor: (prod * latentHeat(0.5 * (p.Tf + p.Tp))) / Qheat,
    brine: make - prod, Sbrine: rec > 0 ? Sloop : m.Sout, recOverall: prod / make, recirc: width * m.mf0 };
}

// ---- humidification–dehumidification (reduced order) -----------------------------------------------------------
const wSat = (T, P = 101325) => (0.622 * psat(T)) / (P - psat(T)), hAir = (T) => CPA * T + wSat(T) * (H0V + CPV * T); // saturated moist air, per kg dry air
/** Closed-air, open-water, water-heated HDH cycle with component effectiveness (energy-based definition). */
export function simulateHDH(v) {
  const T0 = v.T, T2 = v.Ttop, S = salinityFromTDS(tds(scaleIons(cloneIons(v.ions), v.salinityFactor ?? 1)), T0), mw = (v.Qf * density(T0, S)) / 3600, ma = mw / v.MR, eD = v.effD / 100, eH = v.effH / 100, hw = (T) => hL(T, S);
  const F = ([Ta1, Ta2]) => {
    const pw = ma * (wSat(Ta2) - wSat(Ta1)), dHa = ma * (hAir(Ta2) - hAir(Ta1));
    const dHd = dHa - pw * hL(Ta1, 0), maxD = Math.min(ma * (hAir(Ta2) - hAir(T0)) - ma * (wSat(Ta2) - wSat(T0)) * hL(T0, 0), mw * (hw(Ta2) - hw(T0))); // dehumidifier: air cooled, seawater preheated
    const maxH = Math.min(ma * (hAir(T2) - hAir(Ta1)), mw * hw(T2) - (mw - pw) * hw(Ta1)); // humidifier: air heated and humidified by the hot seawater
    return [(dHd - eD * maxD) / (mw * 1e4), (dHa - eH * maxH) / (mw * 1e4)];
  };
  const sol = newtonN(F, [T0 + 0.25 * (T2 - T0), T0 + 0.7 * (T2 - T0)], { tol: 1e-11 }), [Ta1, Ta2] = sol.x, pw = ma * (wSat(Ta2) - wSat(Ta1));
  const T1 = tFromH(hw(T0) + (ma * (hAir(Ta2) - hAir(Ta1)) - pw * hL(Ta1, 0)) / mw, S), Qh = mw * (hw(T2) - hw(T1)), T3 = tFromH((mw * hw(T2) - ma * (hAir(Ta2) - hAir(Ta1))) / (mw - pw), S);
  const fan = (ma / 1.1) * v.dpAir, pump = (mw / density(T0, S)) * 1.5e5;
  return { v, S, mw, ma, Ta1, Ta2, T1, T2, T3, T0, pw, Qh, gor: (pw * latentHeat(T0)) / Qh, rr: pw / mw, sth: Qh / pw / 3.6e6 * 1000, sel: (fan + pump) / (v.etaPump / 100) / pw / 3.6e6 * 1000, converged: sol.converged, residual: sol.residual,
    balance: { in: Qh + mw * hw(T0), out: (mw - pw) * hw(T3) + pw * hL(Ta1, 0) }, Sbrine: (S * mw) / (mw - pw) };
}

// ---- pore-scale relations: Kelvin equation, pore-size distribution, wetting ------------------------------------------
const erfc = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x); return x >= 0 ? y : 2 - y; };
/** Kelvin equation: vapour-pressure ratio p/p_sat over the curved liquid meniscus at the mouth of a hydrophobic pore of radius r (m). */
export const kelvinFactor = (T, r, thetaDeg) => Math.exp((2 * surfaceTension(T) * MW_W * Math.abs(Math.cos((thetaDeg * Math.PI) / 180))) / (density(T, 0) * r * R * K(T)));
/**
 * Pore-scale → continuum membrane coefficient: the dusty-gas flux of every pore class of a log-normal size distribution
 * (median radius mem.r, geometric standard deviation mem.sg) is weighted with its open area ∝ f(r)·r², so that each class
 * sits in its own Knudsen / transition / molecular regime. Returns the same fields as dustyGas.
 */
export function dustyGasPSD(mem, T, P, pv, model = 'auto', vacuum = false, n = 21) {
  const one = dustyGas(mem, T, P, pv, model, vacuum);
  if (!(mem.sg > 1.0001)) return one;
  const s = Math.log(mem.sg), acc = { B: 0, Bk: 0, Bd: 0, Bv: 0 }; let wsum = 0, kn = 0; const cls = [];
  for (let q = 0; q < n; q++) {
    const u = -5 + (10 * q) / (n - 1), r = mem.r * Math.exp(s * u), w = (q === 0 || q === n - 1 ? 1 : q % 2 ? 4 : 2) * Math.exp(-0.5 * u * u) * r * r, d = dustyGas({ ...mem, r }, T, P, pv, model, vacuum);
    wsum += w; acc.B += w * d.B; acc.Bk += w * d.Bk; acc.Bd += w * d.Bd; acc.Bv += w * d.Bv; if (d.Kn > 1) kn += w; cls.push({ r, w, B: d.B, Kn: d.Kn });
  }
  return { ...one, B: acc.B / wsum, Bk: acc.Bk / wsum, Bd: acc.Bd / wsum, Bv: acc.Bv / wsum, Bmean: one.B, knudsenShare: kn / wsum, classes: cls.map((q) => ({ r: q.r, area: q.w / wsum, flux: (q.w * q.B) / acc.B, Kn: q.Kn })) };
}
// ---- pore network: 2-D lattice of throats between the two membrane faces ---------------------------------------------------
const netCache = new Map();
/**
 * Build (and cache) a pore network: nz rows of nx pore bodies between the feed face and the permeate face, joined by (nz + 1)·nx through-plane throats and
 * nz·nx in-plane throats (periodic sideways). Throat radii are log-normal (median r, geometric standard deviation sg, truncated at ±3σ) from a seeded generator.
 * Wetting by invasion percolation: liquid at the feed face enters a throat when the pressure excess dP exceeds its liquid-entry pressure −2Bγcosθ/r and the throat
 * touches liquid already; the pore body behind an invaded throat fills. Returns radii, invaded throats, wet bodies, breakthrough flag and invasion depth.
 */
export function poreNetwork({ r, sg = 1, nx = 24, nz = 12, seed = 7, theta = 120, gamma = 0.066, dP = 0, B = 1 }) {
  nx = clamp(Math.round(nx), 3, 80); nz = clamp(Math.round(nz), 2, 60);
  const key = [r, sg, nx, nz, seed, theta, gamma, dP, B].map((x) => (+x).toPrecision(8)).join('|');
  let net = netCache.get(key);
  if (net) return net;
  const g = rng(seed), s = Math.log(Math.max(sg, 1)), draw = () => r * Math.exp(s * clamp(g.normal(), -3, 3));
  const rv = Float64Array.from({ length: (nz + 1) * nx }, draw), rh = Float64Array.from({ length: nz * nx }, draw); // through-plane throat (k, i): body row k−1 → row k; in-plane throat (k, i): body (k, i) → (k, i+1)
  const rc = dP > 0 && theta > 90 ? liquidEntryPressure(1, theta, gamma, B) / dP : Infinity; // throats wider than rc can be invaded
  const invade = (rCrit) => {
    const wv = new Uint8Array(rv.length), wh = new Uint8Array(rh.length), wet = new Uint8Array(nz * nx), stack = [];
    let through = false, depth = 0;
    const fill = (k, i) => { const n = k * nx + i; if (!wet[n]) { wet[n] = 1; stack.push(n); if (k + 1 > depth) depth = k + 1; } };
    for (let i = 0; i < nx; i++) if (rv[i] > rCrit) { wv[i] = 1; fill(0, i); }
    while (stack.length) {
      const n = stack.pop(), i = n % nx, k = (n - i) / nx, il = (i + nx - 1) % nx, ir = (i + 1) % nx;
      if (rv[(k + 1) * nx + i] > rCrit && !wv[(k + 1) * nx + i]) { wv[(k + 1) * nx + i] = 1; if (k + 1 < nz) fill(k + 1, i); else through = true; }
      if (k > 0 && rv[k * nx + i] > rCrit && !wv[k * nx + i]) { wv[k * nx + i] = 1; fill(k - 1, i); }
      if (rh[k * nx + i] > rCrit && !wh[k * nx + i]) { wh[k * nx + i] = 1; fill(k, ir); }
      if (rh[k * nx + il] > rCrit && !wh[k * nx + il]) { wh[k * nx + il] = 1; fill(k, il); }
    }
    let nw = 0; for (let q = 0; q < wv.length; q++) nw += wv[q]; for (let q = 0; q < wh.length; q++) nw += wh[q];
    let nb = 0; for (let q = 0; q < wet.length; q++) nb += wet[q];
    return { wv, wh, wet, through, depth: depth / nz, fraction: nw / (wv.length + wh.length), bodies: nb / wet.length };
  };
  const inv = invade(rc);
  // breakthrough radius of this network: the largest critical radius for which liquid still reaches the permeate face (bisection over the sorted throat radii)
  const all = Float64Array.from([...rv, ...rh]).sort(); let lo = 0, hi = all.length - 1;
  if (!invade(all[0] * 0.999).through) hi = -1; else while (hi - lo > 1) { const m = (lo + hi) >> 1; if (invade(all[m] * 0.999999).through) lo = m; else hi = m; }
  const rBreak = hi < 0 ? Infinity : all[lo], lepNet = theta > 90 && Number.isFinite(rBreak) ? liquidEntryPressure(rBreak, theta, gamma, B) : Infinity; let sv2 = 0; for (let q = 0; q < rv.length; q++) sv2 += rv[q] * rv[q];
  net = { r, sg, nx, nz, seed, rv, rh, rc, ...inv, invade, rBreak, lepNet, sumRv2: sv2, rMaxThroat: all[all.length - 1], rMinThroat: all[0] };
  if (netCache.size > 30) netCache.clear();
  netCache.set(key, net);
  return net;
}
/**
 * Vapour transport through a pore network: every throat carries the dusty-gas (Knudsen ‖ molecular, or viscous under vacuum) conductance of its own radius,
 * g = πr²·b(r)/ℓ; the vapour pressures of the pore bodies follow from the linear network equations Σ g·(p_i − p_j) = 0 (banded Cholesky factorisation) between
 * the feed face (and every wetted body, where the meniscus sits) and the permeate face. gOf(r) → b(r) for unit length. Returns the network conductance K (per
 * unit pressure difference), the flow through every layer of through-plane throats (equal in a dry network: conservation across cut planes) and the body pressures.
 */
export function solvePoreNetwork(net, gOf) {
  const { nx, nz, rv, rh, wv, wet } = net, N = nx * nz, bw = nx, gv = new Float64Array(rv.length), gh = new Float64Array(rh.length);
  for (let q = 0; q < rv.length; q++) gv[q] = Math.PI * rv[q] * rv[q] * gOf(rv[q]);
  for (let q = 0; q < rh.length; q++) gh[q] = Math.PI * rh[q] * rh[q] * gOf(rh[q]);
  for (let i = 0; i < nx; i++) if (wv[nz * nx + i]) gv[nz * nx + i] = 0; // a liquid bridge to the permeate face carries no vapour
  // lower band storage A[n·(bw+1) + (bw − (n − m))] for m ≤ n; unknowns: body vapour pressures with p_feed = 1, p_permeate = 0
  const W = bw + 1, A = new Float64Array(N * W), b = new Float64Array(N), at = (n, m) => n * bw + bw + m; // = n·W + (bw − (n − m))
  const link = (n, m, g) => { // conductance g between bodies n and m (m > n)
    const dn = wet[n], dm = wet[m];
    if (!dn) A[at(n, n)] += g; if (!dm) A[at(m, m)] += g;
    if (!dn && !dm) A[at(m, n)] -= g; else if (dn && !dm) b[m] += g; else if (!dn && dm) b[n] += g;
  };
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
    const n = k * nx + i;
    if (wet[n]) { A[at(n, n)] = 1; b[n] = 1; }
    if (k === 0 && !wet[n]) { A[at(n, n)] += gv[i]; b[n] += gv[i]; }
    if (k === nz - 1 && !wet[n]) A[at(n, n)] += gv[nz * nx + i];
    if (k < nz - 1) link(n, n + nx, gv[(k + 1) * nx + i]);
    if (i < nx - 1) link(n, n + 1, gh[n]); else if (nx > 2) link(k * nx, n, gh[n]); // periodic in-plane throat
  }
  for (let n = 0; n < N; n++) { // banded Cholesky A = L·Lᵀ (row n stored at n·bw + bw + column)
    const m0 = Math.max(0, n - bw), rn = n * bw + bw;
    for (let m = m0; m <= n; m++) { const rm = m * bw + bw; let q = A[rn + m]; for (let j = Math.max(m0, m - bw); j < m; j++) q -= A[rn + j] * A[rm + j]; A[rn + m] = m === n ? Math.sqrt(Math.max(q, 1e-300)) : q / A[rm + m]; }
  }
  const pb = new Float64Array(N);
  for (let n = 0; n < N; n++) { const rn = n * bw + bw; let q = b[n]; for (let j = Math.max(0, n - bw); j < n; j++) q -= A[rn + j] * pb[j]; pb[n] = q / A[rn + n]; }
  for (let n = N - 1; n >= 0; n--) { let q = pb[n]; for (let j = n + 1, je = Math.min(N - 1, n + bw); j <= je; j++) q -= A[j * bw + bw + n] * pb[j]; pb[n] = q / A[n * bw + bw + n]; }
  const layers = new Array(nz + 1).fill(0);
  for (let k = 0; k <= nz; k++) for (let i = 0; i < nx; i++) layers[k] += gv[k * nx + i] * ((k === 0 ? 1 : pb[(k - 1) * nx + i]) - (k === nz ? 0 : pb[k * nx + i]));
  return { K: layers[nz], layers, p: pb, gv, gh };
}
/**
 * Membrane coefficient from the pore network (same fields as dustyGas). The network value is put on the same open-area (porosity) basis as the continuum
 * models: B = B_single·[K_net / K_uniform]·[n·r² / Σr_t²] over the through-plane throats, so that a network of equal throats returns the single-pore coefficient exactly.
 */
export function dustyGasNetwork(mem, T, P, pv, model = 'auto', vacuum = false) {
  const one = dustyGas(mem, T, P, pv, model, vacuum), net = poreNetwork({ r: mem.r, sg: mem.sg, ...mem.net });
  // the network ÷ single-pore ratio depends on the state only through the Knudsen : molecular mix, which changes slowly: it is solved once per 1 K and 2 % of
  // vapour pressure and reused (the single-pore coefficient in front is always evaluated at the exact state)
  const Tq = Math.round(T), pq = +pv.toPrecision(2) || pv, Pq = +P.toPrecision(3), key = `${Tq}|${Pq}|${pq}|${model}|${vacuum ? 1 : 0}`, memo = (net.memo ||= new Map());
  let m = memo.get(key);
  if (!m) {
    const unit = { eps: 1, tau: 1, delta: 1, r: mem.r }, gOf = (r) => { unit.r = r; return dustyGas(unit, Tq, Pq, pq, model, vacuum).B; }, sol = solvePoreNetwork(net, gOf), gMed = Math.PI * mem.r * mem.r * gOf(mem.r);
    m = { ratio: (sol.K * (net.nz + 1) ** 2 * mem.r * mem.r) / (gMed * net.sumRv2), bundle: mem.sg > 1.0001 ? dustyGasPSD(mem, Tq, Pq, pq, model, vacuum).B / dustyGas(mem, Tq, Pq, pq, model, vacuum).B : 1, K: sol.K, layers: sol.layers, p: sol.p, gv: sol.gv };
    if (memo.size > 400) memo.clear();
    memo.set(key, m);
  }
  // a membrane wetted through has no vapour gap left: a residual coefficient (0.1 % of the dry single-pore value) keeps the module model defined while the result reports the breakthrough
  const ratio = Math.max(m.ratio, 1e-3);
  return { ...one, B: one.B * ratio, Bmean: one.B, Bbundle: one.B * m.bundle, network: { ratio, flooded: m.ratio < 1e-3, K: m.K, layers: m.layers, p: m.p, gv: m.gv, net } };
}
const memCoeff = (mem, ...a) => (mem.net ? dustyGasNetwork(mem, ...a) : mem.sg > 1.0001 ? dustyGasPSD(mem, ...a) : dustyGas(mem, ...a));
/** Pores that the liquid can enter at a trans-membrane pressure dP (Pa): critical radius r* = −2Bγcosθ/dP and the number / area fractions of a log-normal distribution above it. */
export function wettedFraction(rMed, sg, thetaDeg, gamma, dP, B = 1) {
  const rc = dP > 0 && thetaDeg > 90 ? liquidEntryPressure(1, thetaDeg, gamma, B) / dP : thetaDeg > 90 ? Infinity : 0;
  if (!(sg > 1.0001)) return { rc, number: rMed > rc ? 1 : 0, area: rMed > rc ? 1 : 0 };
  const s = Math.log(sg), u = rc > 0 ? Math.log(rc / rMed) / s : -Infinity;
  return { rc, number: Number.isFinite(u) ? 0.5 * erfc(u / Math.SQRT2) : u < 0 ? 1 : 0, area: Number.isFinite(u) ? 0.5 * erfc((u - 2 * s) / Math.SQRT2) : u < 0 ? 1 : 0 };
}
/** Geometric standard deviation of the pore sizes: the input for a log-normal membrane, otherwise inferred from the largest pore taken as the 99th percentile. */
const poreSigma = (p) => (p.poreDist === 'lognormal' || p.poreDist === 'network' ? Math.max(p.sigmaPore, 1.0001) : Math.max(1.02, ((p.rMax * 2) / p.dPore) ** (1 / 2.326)));

// ---- membrane-material library -----------------------------------------------------------------------------------------
/** Typical published properties of conventional and novel membrane materials. MD: pore diameter (µm), porosity, tortuosity, thickness (µm), polymer conductivity (W/m·K), contact angle (°). FO: A (L/m²·h·bar), B (L/m²·h), S (µm). */
export const MATERIALS = {
  pvdf: { kind: 'md', name: 'PVDF, phase inversion', dPore: 0.22, epsM: 0.75, tauM: 2.1, deltaM: 125, kPoly: 0.19, theta: 112 },
  ptfe: { kind: 'md', name: 'PTFE, stretched', dPore: 0.2, epsM: 0.85, tauM: 1.6, deltaM: 60, kPoly: 0.26, theta: 135 },
  pp: { kind: 'md', name: 'Polypropylene, thermally induced phase separation', dPore: 0.2, epsM: 0.72, tauM: 2.3, deltaM: 150, kPoly: 0.15, theta: 120 },
  enf: { kind: 'md', name: 'Electrospun PVDF-HFP nanofibre', dPore: 0.4, epsM: 0.88, tauM: 1.4, deltaM: 100, kPoly: 0.19, theta: 145 },
  cnt: { kind: 'md', name: 'Carbon-nanotube coated PVDF', dPore: 0.25, epsM: 0.8, tauM: 1.5, deltaM: 90, kPoly: 0.35, theta: 150 },
  omni: { kind: 'md', name: 'Omniphobic fluorinated nanoparticle coating', dPore: 0.3, epsM: 0.75, tauM: 1.9, deltaM: 110, kPoly: 0.19, theta: 158 },
  cta: { kind: 'fo', name: 'Cellulose triacetate, asymmetric', AFO: 0.55, BFO: 0.48, Sfo: 480 },
  tfc: { kind: 'fo', name: 'Thin-film composite polyamide', AFO: 2.5, BFO: 0.4, Sfo: 450 },
  aqp: { kind: 'fo', name: 'Aquaporin biomimetic', AFO: 1.2, BFO: 0.12, Sfo: 220 },
  tfn: { kind: 'fo', name: 'Thin-film nanocomposite (graphene-oxide interlayer)', AFO: 3.5, BFO: 0.45, Sfo: 250 },
};
/** Inputs with the selected library material applied (the custom entry leaves the membrane fields as entered). */
export function applyMaterial(v) {
  const m = MATERIALS[v.memMat];
  if (!m || (m.kind === 'md' ? !isMD(v) : !isFO(v))) return v;
  const { kind, name, ...props } = m, o = { ...v, ...props };
  if (kind === 'md') o.rMax = Math.max(v.rMax * (m.dPore / v.dPore), m.dPore / 2);
  return o;
}

// ---- capacitive deionisation: porous-electrode (transmission-line) model -------------------------------------------
const EPS_W = 78.4 * 8.8541878128e-12, LAM_NACL = 0.01264; // permittivity of water (F/m), molar conductivity of NaCl at 25 °C (S·m²/mol)
/**
 * Unified double layer of a charged pore, valid from thin to overlapping layers. Per m² of pore wall of one electrode, for the symmetric
 * electrode pair (σ = surface charge, mol/m²; c = salt concentration of the electroneutral liquid in equilibrium with the pore, mol/m³):
 *   thin layers (Gouy–Chapman):        φ_g = 2·asinh(σ / 4λ_D c),        S_g = b + √(σ² + (4λ_D c)²) − 4λ_D c
 *   overlapping layers (mod. Donnan):  φ_d = asinh(σ / b),               S_d = √(σ² + b²),            b = 2h·c·e^μ
 * φ = diffuse/Donnan potential in RT/F, S = salt held by the pore (liquid + double-layer excess) and so removed from the cell, h = pore volume
 * per wall area (half-width of a slit pore), μ = non-electrostatic attraction of the ions into the pore (kT). Co-ion expulsion is part of both:
 * the differential charge efficiency ∂S/∂σ is tanh(φ_g/2) and tanh(φ_d), always below one. Automatic switch on the pore size against the Debye
 * length, r = h/λ_D(c):  θ = smoothstep of ln r between r = 2 (θ = 0, Donnan) and r = 4 (θ = 1, Gouy–Chapman),
 *   φ = θφ_g + (1 − θ)φ_d,   S = θS_g + (1 − θ)S_d.
 * For r ≥ 2 the Gouy–Chapman pore content exceeds the Donnan one at every charge (S_g ≥ S_d, because b ≥ 4λ_D c) and its salt capacity is positive, so
 * ∂S/∂c > 0, 0 ≤ ∂S/∂σ < 1 and ∂φ/∂σ > 0 hold in every regime: the salt inventory is a monotone state function and the cell balance is well posed
 * for any salinity and pore size (the thin-layer formula alone gives a negative co-ion content once the layers overlap).
 * o = { kL: λ_D·√c, h, emu: e^μ, model: 'auto' | 'gcs' | 'md' }.
 */
export function edlState(sigma, c, o) {
  const s = Math.abs(sigma), sg = sigma < 0 ? -1 : 1, B = 4 * o.kL * Math.sqrt(c), Rg = Math.sqrt(s * s + B * B), w = (s * s) / (Rg + B), pg = 2 * Math.asinh(s / B), b = 2 * o.h * c * o.emu, Sd = Math.sqrt(s * s + b * b), pd = Math.asinh(s / b), Sg = b + w;
  let th = 1;
  if (o.model === 'md') th = 0;
  else if (o.model !== 'gcs') { const xi = Math.log(b / (o.emu * B)) / EDL_LN3; th = xi <= 0 ? 0 : xi >= 1 ? 1 : xi * xi * (3 - 2 * xi); } // b/(e^μ·B) = r/2
  return { pd: sg * (th * pg + (1 - th) * pd), S: th * Sg + (1 - th) * Sd, theta: th, ceff: th * c + ((1 - th) * Sd) / (2 * o.h), phiG: sg * pg, phiD: sg * pd, Sg, Sd, eff: th * Math.tanh(pg / 2) + (1 - th) * Math.tanh(pd), r: (2 * b) / (o.emu * B) };
}
const EDL_LN3 = Math.LN2; // ln(r₁/r₀) of the blending zone, r₀ = 2, r₁ = 4
/** Equilibrium of the unified double layer with a Stern layer in series at half-cell voltage Vh (V): Newton on the surface charge. */
export function edlEquilibrium(Vh, c, o, Cst, T = 25) {
  const Vt = (R * K(T)) / F, f = (s) => Vt * edlState(s, c, o).pd + (s * F) / Cst - Vh;
  if (!(Vh > 0)) return { sigma: 0, ...edlState(0, c, o) };
  const hi = (Vh * Cst) / F; let s = brent(f, 0, hi, 1e-12 * hi, 80);
  if (!Number.isFinite(s)) s = 0;
  return { sigma: s, ...edlState(s, c, o), lamD: o.kL / Math.sqrt(c) };
}
/**
 * Porous carbon electrode pair with a flow-by spacer. Along the electrode depth x (0 = spacer side, L_e = current collector):
 *   charge balance      a·F·∂σ/∂t = ∂/∂x(κ_e ∂Δφ/∂x),     Δφ = σF/C_St + (RT/F)·φ(σ, c)      (Stern layer + unified diffuse / Donnan layer, edlState)
 *   boundaries          ∂Δφ/∂x = 0 at the collector (no ion flux);  spacer side: prescribed half-cell voltage through the series resistance, or prescribed current
 *   salt (mixed)        d/dt[V_sp·c + a·∫S(σ, c) dx + Γ_L] = q(c_in − c)         (S holds the pore liquid and the double-layer excess, so the balance closes by construction)
 *   Langmuir            dθ/dt = k[Kc(1 − θ) − θ],  Γ_L = q_m·θ   (non-electrostatic adsorption on the carbon)
 *   pore conductivity   κ_e = Λ·ε^1.5·c_pore, c_pore = c for thin layers and the mean ion concentration of the pore (surface conduction) for overlapping ones
 * Finite volumes in x; fully implicit variable-step BDF2 in time (first step of every half-cycle backward Euler, steps graded from the fastest electrode time
 * constant), Newton iteration on (σ₁ … σ_n, ln c) with a bordered-tridiagonal Jacobian. The integrals (charge, energy, salt taken from the flow) use the same
 * formula, so the charge and salt balances close to the Newton tolerance for any step size, salinity and double-layer area.
 */
export function simulatePorousCDI(v, c0, ov = {}) {
  const p = { ...v, ...ov }, ask = p.cdiEdl === 'gcs' || p.cdiEdl === 'md' ? p.cdiEdl : 'auto';
  let r = porousCDICore(p, Math.max(c0, 1e-9), ask);
  if (!r) { r = porousCDICore(p, Math.max(c0, 1e-9), 'auto'); r.fallback = ask; } // the forced thin-layer model has no solution when its double layers would hold more salt than the pores: the unified model is used instead
  return r;
}
function porousCDICore(p, c0, model) {
  const T = p.T, Vt = (R * K(T)) / F, Le = p.cdiLe * 1e-6, pM = p.cdiPor, n = clamp(Math.round(p.cdiNx), 3, 60), nAsk = n, dx = Le / n, av = p.cdiArea * p.cdiRho * 1e6, Cst = p.cdiCst, hsp = p.cdiHsp * 1e-6, psp = 0.7;
  const Rc = p.cdiRc * 1e-4, qA = p.cdiQ / 1000 / 60, Lam = LAM_NACL * (1 + 0.0191 * (T - 25)), mE = 2 * Le * p.cdiRho * 1e6, qmA = (p.lgQm / 58.44e3) * mE, KL = p.lgK, ka = p.lgKa / 60, Vsp = hsp * psp, Vmix = Vsp + 2 * Le * pM;
  const kL = Math.sqrt((EPS_W * R * K(T)) / (2 * F * F)), lam = (c) => kL / Math.sqrt(c), hP = pM / av, th0 = edlState(0, c0, { kL, h: hP, emu: 1, model }).theta, muEff = clamp(p.cdiMu ?? 0, 0, 4) * (1 - th0), eo = { kL, h: hP, emu: Math.exp(muEff), model } /* the attraction term belongs to overlapped (micro)pores: it fades out as the feed makes the layers thin */, kp = Lam * pM ** 1.5, ksp = Lam * psp ** 1.5;
  const pdOf = (s, c) => edlState(s, c, eo).pd, dphi = (s, c) => Vt * pdOf(s, c) + (s * F) / Cst, wOf = (s, c) => edlState(s, c, eo).S - 2 * hP * c;
  const Imax = p.cdiMode === 'cc' ? p.cdiI : Infinity, eq = edlEquilibrium(p.cdiV / 2, c0, eo, Cst, T), thEq = (KL * c0) / (1 + KL * c0), sScale = Math.max(eq.sigma, (1e-3 * Cst * Vt) / F), mScale = Vmix * c0 + av * Le * sScale;
  // differential capacitance (F/m²) at zero charge and pore conductivity: fastest and slowest electrode time constants
  const cap = (c) => { const e = edlState(0, c, eo), dpd = e.theta / (2 * kL * Math.sqrt(c)) + (1 - e.theta) / (2 * hP * c * eo.emu); return 1 / (1 / Cst + (Vt * dpd) / F); }, kapE = (c) => kp * edlState(0, c, eo).ceff;
  const tauRC = (Le * Le * av * cap(c0)) / kapE(c0), dtFast = Math.min((0.3 * dx * dx * av * cap(c0)) / kapE(c0), 0.2 * (Vmix / qA));
  // work arrays; state vector Y = [σ_1 … σ_n, ln c, θ_L, M (salt inventory), E, Q, salt taken from the flow]
  const NY = n + 6, PH = new Float64Array(n), SS = new Float64Array(n), KP = new Float64Array(n), TH = new Float64Array(n), R0 = new Float64Array(n + 1), R1 = new Float64Array(n + 1), sg = new Float64Array(n), sp = new Float64Array(n), be = new Float64Array(NY);
  const tl = new Float64Array(n), td = new Float64Array(n), tu = new Float64Array(n), bc = new Float64Array(n), gr = new Float64Array(n), S0 = new Float64Array(n), xa = new Float64Array(n), xb = new Float64Array(n), cw = new Float64Array(n);
  let al = 1, dt = 0, Vh = 0, thick = 0, thMin = 1, dtMin = Infinity, nSteps = 0, nNewton = 0, nFail = 0, bad = false, abort = false;
  const resid = (s, u, out) => {
    const c = Math.exp(u);
    for (let j = 0; j < n; j++) { const e = edlState(s[j], c, eo); PH[j] = Vt * e.pd + (s[j] * F) / Cst; SS[j] = e.S; KP[j] = kp * e.ceff; TH[j] = e.theta; }
    const Rh = (0.5 * hsp) / (ksp * c) + 0.5 * Rc + dx / (2 * KP[0]), I0 = clamp((Vh - PH[0]) / Rh, -Imax, Imax);
    let Ip = I0, ms = 0;
    for (let j = 0; j < n; j++) { const In = j < n - 1 ? (((2 * KP[j] * KP[j + 1]) / (KP[j] + KP[j + 1])) * (PH[j] - PH[j + 1])) / dx : 0; out[j] = al * s[j] - be[j] - (dt * (Ip - In)) / (av * F * dx); Ip = In; ms += SS[j]; }
    const th = (be[n + 1] + dt * ka * KL * c) / (al + dt * ka * (KL * c + 1)), M = Vsp * c + av * dx * ms + qmA * th;
    out[n] = al * M - be[n + 2] - dt * qA * (c0 - c);
    return { I0, th, M, c, Vcell: 2 * (PH[0] + I0 * Rh) };
  };
  const norm = (Rv) => { let m = 0; for (let j = 0; j < n; j++) m = Math.max(m, Math.abs(Rv[j]) / sScale); return m + Math.abs(Rv[n]) / mScale; };
  /** One implicit step from Y1 (and Y0 for BDF2, step ratio rho) to a new state; null if Newton fails. */
  const step = (Y1, Y0, h, rho, V) => {
    dt = h; Vh = V; al = Y0 ? (1 + 2 * rho) / (1 + rho) : 1;
    const b1 = Y0 ? 1 + rho : 1, b0 = Y0 ? -(rho * rho) / (1 + rho) : 0;
    for (let q = 0; q < NY; q++) be[q] = b1 * Y1[q] + (Y0 ? b0 * Y0[q] : 0);
    for (let j = 0; j < n; j++) sg[j] = Y1[j];
    let u = Y1[n], st = resid(sg, u, R0), nr = norm(R0), ok = false;
    for (let it = 0; it < 40; it++) {
      nNewton++;
      for (let j = 0; j < n; j++) S0[j] = SS[j];
      for (let k = 0; k < 3; k++) { // tridiagonal block by three coloured perturbations
        for (let j = 0; j < n; j++) sp[j] = sg[j] + (j % 3 === k ? 1e-7 * (Math.abs(sg[j]) + sScale) : 0);
        resid(sp, u, R1);
        for (let j = k; j < n; j += 3) { const e = sp[j] - sg[j]; td[j] = (R1[j] - R0[j]) / e; if (j > 0) tu[j - 1] = (R1[j - 1] - R0[j - 1]) / e; if (j < n - 1) tl[j + 1] = (R1[j + 1] - R0[j + 1]) / e; gr[j] = (al * av * dx * (SS[j] - S0[j])) / e; }
      }
      resid(sg, u + 1e-6, R1);
      for (let j = 0; j < n; j++) bc[j] = (R1[j] - R0[j]) / 1e-6;
      const dd = (R1[n] - R0[n]) / 1e-6;
      // bordered tridiagonal solve: T·xa = −R, T·xb = bc, then the salt row
      cw[0] = tu[0] / td[0]; xa[0] = -R0[0] / td[0]; xb[0] = bc[0] / td[0];
      for (let j = 1; j < n; j++) { const m = td[j] - tl[j] * cw[j - 1]; cw[j] = tu[j] / m; xa[j] = (-R0[j] - tl[j] * xa[j - 1]) / m; xb[j] = (bc[j] - tl[j] * xb[j - 1]) / m; }
      for (let j = n - 2; j >= 0; j--) { xa[j] -= cw[j] * xa[j + 1]; xb[j] -= cw[j] * xb[j + 1]; }
      let ga = 0, gb = 0; for (let j = 0; j < n; j++) { ga += gr[j] * xa[j]; gb += gr[j] * xb[j]; }
      let du = (-R0[n] - ga) / (dd - gb);
      if (!Number.isFinite(du)) break;
      let f = Math.abs(du) > 1.5 ? 1.5 / Math.abs(du) : 1, mx = 0, nrN = 0, stN = null, tiny = Math.abs(du) < 1e-9;
      if (tiny) for (let j = 0; j < n; j++) if (Math.abs(xa[j] - xb[j] * du) > 1e-9 * sScale) { tiny = false; break; }
      for (let ls = 0; ls < (tiny ? 1 : 6); ls++) { // damped update (a Newton step below 1e-9 of the scales is taken in full: the residual is then at its round-off floor)
        for (let j = 0; j < n; j++) sp[j] = sg[j] + f * (xa[j] - xb[j] * du);
        stN = resid(sp, u + f * du, R1); nrN = norm(R1);
        if (Number.isFinite(nrN) && (nrN <= nr * (1 - 0.1 * f) + 1e-14 || ls === 5)) break;
        f *= 0.4;
      }
      if (!Number.isFinite(nrN)) break;
      for (let j = 0; j < n; j++) { const d = sp[j] - sg[j]; mx = Math.max(mx, Math.abs(d) / sScale); sg[j] = sp[j]; R0[j] = R1[j]; }
      R0[n] = R1[n]; u += f * du; st = stN; nr = nrN;
      if (tiny || nr < 1e-13) { ok = true; break; }
    }
    if (!ok) return null;
    const Yn = new Float64Array(NY);
    for (let j = 0; j < n; j++) Yn[j] = sg[j];
    Yn[n] = u; Yn[n + 1] = st.th; Yn[n + 2] = st.M; Yn[n + 3] = (be[n + 3] + dt * st.Vcell * st.I0) / al; Yn[n + 4] = (be[n + 4] + dt * st.I0) / al; Yn[n + 5] = (be[n + 5] + dt * qA * (c0 - st.c)) / al;
    Yn.I0 = st.I0; Yn.Vcell = st.Vcell;
    return Yn;
  };
  const inventory = (s, c, th) => { let ms = 0, q = 0; for (let j = 0; j < n; j++) { ms += edlState(s[j], c, eo).S; q += s[j]; } return { edl: av * dx * ms - 2 * Le * pM * c /* excess over the pore liquid at the spacer concentration */, lang: qmA * th, mix: Vmix * c, charge: F * av * dx * q, M: Vsp * c + av * dx * ms + qmA * th }; };
  const capacity = (Y) => { const c = Math.exp(Y[n]); let d = 0; for (let j = 0; j < n; j++) { d += (edlState(Y[j], c * 1.001, eo).S - edlState(Y[j], c * 0.999, eo).S) / (0.002 * c); thMin = Math.min(thMin, edlState(Y[j], c, eo).theta); } return Vsp + av * dx * d; };
  const tc = p.cdiTc * 60, tdis = p.cdiTd * 60, nOut = clamp(Math.round(p.cdiNt || 120), 20, 600), nCyc = clamp(Math.round(p.cdiCycles), 1, 8), sub = 2;
  let Y = new Float64Array(NY);
  { const s0 = (clamp(p.cdiQ0, 0, 100) / 100) * eq.sigma, th0 = (clamp(p.lgTheta0, 0, 100) / 100) * thEq; for (let j = 0; j < n; j++) Y[j] = s0; Y[n] = Math.log(c0); Y[n + 1] = th0; Y[n + 2] = inventory(Y, c0, th0).M; }
  const legacy = (Yv) => [...Array.from(Yv.subarray(0, n)), Math.exp(Yv[n]), Yv[n + 1], Yv[n + 3], Yv[n + 4], Yv[n + 5]];
  const tt = [], cEff = [], cur = [], volt = [], snaps = [];
  let last = null, tOff = 0;
  for (let cyc = 0; cyc < nCyc; cyc++) {
    const keep = cyc === nCyc - 1, half = (V, dur, tag) => {
      const Dt = dur / nOut, Ya = Y, sA = inventory(Ya, Math.exp(Ya[n]), Ya[n + 1]);
      // advance over [0, h] by one step; on Newton failure restart with backward Euler on halved steps
      let Yp = null, hp = 0;
      const adv = (h, depth) => {
        if (abort) return;
        let Yn = step(Y, Yp, h, Yp ? h / hp : 1, V);
        if (!Yn && Yp) Yn = step(Y, null, h, 1, V);
        if (Yn) { Yp = Y; hp = h; Y = Yn; nSteps++; dtMin = Math.min(dtMin, h); const den = capacity(Y); thick = Math.max(thick, 1 - den / Vmix); if (!(den > 0)) { bad = true; if (model === 'gcs') abort = true; } return; }
        if (model === 'gcs') { bad = true; abort = true; return; } // the forced thin-layer model is abandoned at the first failure (the caller switches to the unified model)
        if (depth >= 6 || nNewton > 60000) { nFail++; if (nFail > 24) abort = true; return; } // keep the last converged state (reported as a warning); balances stay closed
        Yp = null; adv(h / 2, depth + 1); adv(h / 2, depth + 1);
      };
      const rec = (k, t) => { if (!keep) return; const c = Math.exp(Y[n]); tt.push((tOff + t) / 60); cEff.push(c); cur.push(k === 0 ? clamp((V - dphi(Y[0], c)) / ((0.5 * hsp) / (ksp * c) + 0.5 * Rc + dx / (2 * kp * edlState(Y[0], c, eo).ceff)), -Imax, Imax) : Y.I0); volt.push(k === 0 ? 2 * V - (Number.isFinite(Imax) ? 2 * (V - dphi(Y[0], c)) - 2 * cur[cur.length - 1] * ((0.5 * hsp) / (ksp * c) + 0.5 * Rc + dx / (2 * kp * edlState(Y[0], c, eo).ceff)) : 0) : Y.Vcell); if (tag === 'c' && k % Math.ceil(nOut / 5) === 0) snaps.push({ t, sigma: Array.from(Y.subarray(0, n)), phi: Array.from(Y.subarray(0, n), (s) => dphi(s, c)) }); };
      rec(0, 0);
      // first output interval: steps graded geometrically from the fastest time constant
      const g = 1.4, d0 = Math.min(Math.max(dtFast, 1e-7 * Dt), Dt / sub), m = Math.max(sub, Math.ceil(Math.log(1 + ((g - 1) * Dt) / d0) / Math.log(g))), sc = (Dt * (g - 1)) / (g ** m - 1);
      for (let q = 0; q < m; q++) adv(sc * g ** q, 0);
      rec(1, Dt);
      for (let k = 2; k <= nOut; k++) { for (let q = 0; q < sub; q++) adv(Dt / sub, 0); rec(k, k * Dt); }
      tOff += dur;
      const sB = inventory(Y, Math.exp(Y[n]), Y[n + 1]);
      return { flow: Y[n + 5] - Ya[n + 5], y0: legacy(Ya), y1: legacy(Y), s0: sA, s1: sB, E: Y[n + 3] - Ya[n + 3], Q: Y[n + 4] - Ya[n + 4], sub };
    };
    if (keep) tOff = 0;
    const ch = half(p.cdiV / 2, tc, 'c'), dis = half(p.cdiVdis / 2, tdis, 'd');
    last = { ch, dis };
    if (bad && model === 'gcs') return null;
    if (abort) break;
  }
  const { ch, dis } = last, salt = ch.flow, charge = ch.Q, Enet = ch.E - (clamp(p.cdiRecov, 0, 95) / 100) * Math.max(0, -dis.E), vol = qA * tc, e0 = edlState(0, c0, eo), cEnd = ch.y1[n];
  return { p, T, c0, n, nAsk, Le, dx, av, mE, eq, thEq, tt, cEff, cur, volt, snaps, x: Array.from({ length: n }, (_, j) => (j + 0.5) * dx), ch, dis, salt, charge, Enet, vol, lamD: lam(c0), tauRC, dtStab: Number.isFinite(dtMin) ? dtMin : 0,
    cAvg: Math.max(0, c0 - salt / vol), removal: salt / (vol * c0), sac: (salt * 58.44e3) / mE, eff: charge > 0 ? (F * salt) / charge : 0, sec: Enet / 3.6e6 / vol, ePerMol: Enet / Math.max(Math.abs(salt), 1e-30) / 1000, waterRec: tc / (tc + tdis), prod: (qA * 3.6e6 * tc) / (tc + tdis),
    thick, sigmaEnd: ch.y1.slice(0, n), cEnd, thetaEnd: ch.y1[n + 1], storedEDL: ch.s1.edl - ch.s0.edl, storedLang: ch.s1.lang - ch.s0.lang, storedMix: ch.s1.mix - ch.s0.mix, chargeStored: ch.s1.charge - ch.s0.charge, pdOf, wOf, dphi,
    edlModel: model, muEff, hPore: hP, poreRatio: hP / lam(c0), thetaFeed: e0.theta, thetaMin: thMin, thetaEnd2: edlState(ch.y1[0], cEnd, eo).theta, regime: e0.theta >= 0.999 ? 'thin double layers (Gouy–Chapman–Stern)' : e0.theta <= 0.001 ? 'overlapping double layers (modified Donnan)' : 'transition (blended)', steps: nSteps, newton: nNewton, failedSteps: nFail, fallback: null, edl: (s, c) => edlState(s, c, eo) };
}
/** Diffuse layer at the electrode surface resolved with the Poisson–Nernst–Planck solver of suite 7 (insulating wall at the diffuse-layer potential). */
export function edlProfile(psiD, c, T) {
  const lam = debyeLength(2 * c, T), s = pnpRamp({ z: [1, -1], T, ratio: 1.15, res: 12, layers: [{ L: 30 * lam, D: [1.33e-9, 2.03e-9], n: 24 }], left: { type: 'wall', psi: psiD, flux: [0, 0] }, right: { type: 'bulk', c: [c, c], psi: 0 } }, { left: { psi: 0 } });
  const Vt = (R * K(T)) / F;
  return { ...s, lam, gc: s.x.map((x) => 4 * Vt * Math.atanh(Math.tanh(psiD / (4 * Vt)) * Math.exp(-x / lam))), sigmaGC: Math.sqrt(8 * EPS_W * R * K(T) * c) * Math.sinh(psiD / (2 * Vt)) };
}
/** Equivalent 1:1 salt concentration (mol/m³) of an ion analysis: half of the total charge equivalents. */
const eqConc = (ions) => 0.5 * sum(ION_IDS.map((k) => (Math.abs(IONS[k].z) * (+ions[k] || 0)) / IONS[k].mw));

// ---- crystallisation: mixed-suspension mixed-product-removal crystalliser on the MD loop ----------------------------------
/** NaCl-equivalent solubility, g per kg of solution. */
export const saltSolubility = (T) => 264.2 + 0.17 * (T - 25);
/**
 * MSMPR population balance n(L) = n₀·exp(−L/Gτ) with growth G = k_g·σ^g and secondary nucleation B₀ = k_b·M_T·σ^b.
 * The suspension-density balance M_T = 6·k_v·ρ_c·n₀·(Gτ)⁴ fixes the relative supersaturation: σ = [6·k_v·ρ_c·τ⁴·k_g³·k_b]^(−1/(3g+b)).
 */
export function msmpr({ tau, kg, kb, g = 1, b = 2, kv = 0.5, rhoC = 2165, MT }) {
  const sigma = (1 / (6 * kv * rhoC * tau ** 4 * kg ** 3 * kb)) ** (1 / (3 * g + b)), G = kg * sigma ** g, B0 = kb * MT * sigma ** b, n0 = B0 / G, L0 = G * tau;
  return { sigma, G, B0, n0, L0, L43: 4 * L0, LD: 3 * L0, cv: 50, kv, rhoC, MT, massDensity: (L) => kv * rhoC * n0 * L ** 3 * Math.exp(-L / L0) };
}
/** MD–crystalliser: the MD loop runs at the saturation of the crystalliser, the solids leave as crystals and a small bleed purges impurities. */
export function simulateMDC(v, ov = {}) {
  const p = { ...v, ...ov }, ions = scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1), tdsF = tds(ions), Sfeed = salinityFromTDS(tdsF, 25), make = (p.Qf * density(25, Sfeed)) / 3600;
  const cr = msmpr({ tau: p.tauCr * 3600, kg: p.kgCr * 1e-6, kb: p.kbCr * 1e8, MT: p.slurry }), Ssat = saltSolubility(p.Tcr), bleed = (clamp(p.bleed, 0, 50) / 100) * make;
  const salt = (make * Sfeed) / 1000, crystal = salt > (bleed * Ssat * (1 + cr.sigma)) / 1000, Sloop = crystal ? Ssat * (1 + cr.sigma) : (1000 * salt) / bleed; // a large bleed keeps the loop below saturation: no crystals
  const solids = crystal ? salt - (bleed * Sloop) / 1000 : 0, prod = make - bleed - solids, m = mdModule(p, Sloop), width = prod / Math.max(m.Nsum, 1e-12), area = width * m.c.L;
  const Qheat = m.Qin * width + make * cp(p.T, Sfeed) * Math.max(0, m.TfOut - p.T) * (m.q.recover ? 0.3 : 1), Vcr = (Math.max(solids, 0) * p.tauCr * 3600) / p.slurry, gamma = surfaceTension(p.Tf, p.gammaF);
  return { p, ions, tdsF, Sfeed, make, cr, Ssat, Sloop, bleed, solids, prod, m, width, area, Qheat, Vcr, lep: liquidEntryPressure(p.rMax * 1e-6, p.theta, gamma, p.lepB), gamma, wallSat: m.SmMax / saltSolubility(p.Tf), sth: Qheat / prod / 3.6e6 * 1000, sel: (m.Wpump * width) / prod / 3.6e6 * 1000 + 0.5,
    gor: (prod * latentHeat(0.5 * (p.Tf + p.Tp))) / Qheat, recOverall: prod / (make - Math.max(solids, 0)) };
}

// ---- dynamic operation: batch concentration with fouling, scaling, wetting and membrane hydration --------------------------
/**
 * Batch MD of a feed tank through the installed membrane area (time in hours):
 *   dM/dt = −(N + J_leak)·A,  N = f_mod·(1 − φ_s)(1 − x_w)·N_local(T_f, S, h_f′),  1/h_f′ = 1/h_f + δ_d/k_d,
 *   deposit dm_d/dt = c_fou·N/ρ − k_rem·m_d,  scale dm_s/dt = k_sc·(Ω_wall − 1)₊²,  φ_s = m_s/(m_s + m_b),
 *   contact angle θ = θ₀ − Δθ·m_d/(m_d + m_½),  wetting dx_w/dt = (x_eq(θ) − x_w)₊/τ_w with x_eq the area share of pores above the critical radius.
 */
export function dynamicMD(v, r) {
  const p = v, c0 = r.m.c, A = r.area, S0 = r.Sfeed, rho0 = density(25, S0), M0 = p.batchVol * rho0, cold = { T: p.Tp, pg: (p.rhGas / 100) * psat(p.Tp) }, Sm0 = r.rec > 0 ? r.Sloop : r.Sfeed, base = mdLocal(c0, p.Tf, Sm0, cold);
  const fmod = base.N > 0 ? r.m.flux / base.N : 1, sg = poreSigma(p), gamma = surfaceTension(p.Tf, p.gammaF), dP = p.pFeed * 1e5, kd = 0.6, rhoD = 1200, mB = 20, mH = 5, n = clamp(Math.round(p.ntDyn), 20, 2000), tEnd = p.tDyn;
  const gyp0 = r.sat.gypsum / Math.max(r.m.SmMax / Math.max(r.Sfeed, 1e-9), 1e-9) ** 2;
  const at = (y) => {
    const M = Math.max(y[0], 0.02 * M0), salt = Math.max(y[1], 0), S = Math.min((1000 * salt) / M, 350), md = Math.max(y[2], 0), ms = Math.max(y[3], 0), xw = clamp(y[4], 0, 1);
    const theta = p.theta - p.thetaDrop * (md / (md + mH)), cfg = { ...c0, hf: 1 / (1 / c0.hf + md / 1000 / rhoD / kd), theta }, lo = mdLocal(cfg, p.Tf, S, cold), phi = ms / (ms + mB), live = y[0] > 0.05 * M0 ? 1 : 0;
    const N = live * fmod * (1 - phi) * (1 - xw) * Math.max(lo.N, 0), xeq = wettedFraction(c0.mem.r, sg, theta, gamma, dP, p.lepB).area, leak = (live * xw * c0.mem.eps * c0.mem.r ** 2 * dP * density(p.Tf, S)) / (8 * viscosity(p.Tf, S) * c0.mem.tau * c0.mem.delta);
    const omega = Math.max(lo.Sm / saltSolubility(lo.Tfm), gyp0 * (lo.Sm / Math.max(S0, 1e-9)) ** 2);
    return { M, S, md, ms, xw, theta, N, leak, xeq, omega, lo, phi };
  };
  const rhs = (t, y) => { const s = at(y), lk = s.leak * A * 3600; return [-s.N * A * 3600 - lk, (-lk * s.S) / 1000, (p.cFou * s.N * 3600) / density(25, 0) - p.kRem * s.md, p.kScaleMD * Math.max(0, s.omega - 1) ** 2, Math.max(0, s.xeq - s.xw) / Math.max(p.tauWet, 1e-3), s.N * A * 3600, (lk * s.S) / 1000, lk]; };
  const y0 = [M0, (M0 * S0) / 1000, 0, 0, clamp(p.wet0 / 100, 0, 1), 0, 0, 0], sol = rk4(rhs, y0, 0, tEnd, n), rows = sol.y.map((y, k) => ({ t: sol.t[k], ...at(y), prod: y[5] + y[7], y }));
  const tdsP = rows.map((q) => { const w = q.N + q.leak; return w > 0 ? (1e3 * q.leak * q.S) / w : 0; }), first = (f) => { const k = rows.findIndex(f); return k < 0 ? null : rows[k].t; }, e = rows[rows.length - 1];
  return { rows, tdsP, M0, S0, A, fmod, sg, flux0: rows[0].N * 3600, fluxEnd: e.N * 3600, decline: rows[0].N > 0 ? 1 - e.N / rows[0].N : 0, tWet: first((q) => q.xw > 0.01), tSat: first((q) => q.omega >= 1), cf: e.S / S0, recovery: e.prod / M0, product: e.prod,
    tdsMix: e.prod > 0 ? (1e6 * e.y[6]) / e.prod : 0, saltBal: { in: (M0 * S0) / 1000, out: e.y[1] + e.y[6] }, waterBal: { in: M0, out: e.y[0] + e.y[5] + e.y[7] } };
}
/**
 * Batch FO between a feed tank and a draw tank (time in hours): dV_F/dt = −J_w·A = −dV_D/dt, dn_D/dt = −J_s·A, with the flux from the
 * ICP/ECP coupon model at the tank compositions, a cake resistance dR_c/dt = α_c·c_fou·J_w − k_rem·R_c in series with the membrane
 * (1/A′ = 1/A + μR_c) and the support layer wetting out from its initial hydration h₀: dh/dt = (1 − h)/τ_h, S_eff = S/h.
 */
export function dynamicFO(v, ov = {}) {
  const p = { ...v, ...ov }, A = p.areaFO, VF0 = p.batchVol, VD0 = p.drawVol, n = clamp(Math.round(p.ntDyn), 20, 2000), mu = viscosity(p.T, 0), h0 = clamp((p.hydration ?? 100) / 100, 0.2, 1), d = DRAWS[p.draw] || DRAWS.nacl;
  const at = (y) => { const VF = Math.max(y[0], 0.02 * VF0), VD = Math.max(y[1], 1e-9), h = clamp(y[5], 0.2, 1), s = foSetup(p, { hydration: 100 * h }), m = { ...s.m, A: 1 / (1 / s.m.A + mu * Math.max(y[4], 0)) }, st = { f: VF0 / VF, cFd: Math.max(y[3], 0) / VF, cD: Math.max(y[2], 0) / VD }, fl = y[0] > 0.03 * VF0 ? foFlux(st, m) : { Jw: 0, Js: 0 }; return { VF, VD, h, st, fl, s }; };
  const rhs = (t, y) => { const q = at(y), jw = q.fl.Jw * A * 3600, js = q.fl.Js * A * 3600; return [-jw, jw, -js, js, p.alphaCake * 1e13 * (p.cFou / 1000) * q.fl.Jw * 3600 - p.kRem * Math.max(y[4], 0), (1 - q.h) / Math.max(p.tauHyd, 1e-3)]; };
  const y0 = [VF0, VD0, VD0 * p.cDraw * 1000, 0, 0, h0], sol = rk4(rhs, y0, 0, p.tDyn, n), rows = sol.y.map((y, k) => ({ t: sol.t[k], ...at(y), Rc: Math.max(y[4], 0), y })), e = rows[rows.length - 1];
  return { rows, VF0, VD0, A, d, flux0: rows[0].fl.Jw * 3.6e6, fluxEnd: e.fl.Jw * 3.6e6, fluxMax: Math.max(...rows.map((q) => q.fl.Jw)) * 3.6e6, recovery: 1 - e.y[0] / VF0, cDend: e.st.cD / 1000, cf: e.st.f, soluteLoss: (e.y[3] * d.M) / 1000, waterBal: { in: VF0 + VD0, out: e.y[0] + e.y[1] }, soluteBal: { in: VD0 * p.cDraw * 1000, out: e.y[2] + e.y[3] } };
}

// ---- multi-objective sweep and grey-box correction -------------------------------------------------------------------------
/** Non-dominated set of points for (maximise fx, minimise fy), sorted by fx, with the knee (largest distance from the chord of the front). */
export function paretoFront(pts, fx, fy) {
  const front = pts.filter((a) => !pts.some((b) => b !== a && fx(b) >= fx(a) && fy(b) <= fy(a) && (fx(b) > fx(a) || fy(b) < fy(a)))).sort((a, b) => fx(a) - fx(b));
  if (front.length < 3) return { front, knee: front[front.length - 1] || null };
  const a = front[0], b = front[front.length - 1], dx = fx(b) - fx(a) || 1, dy = fy(b) - fy(a) || 1;
  let knee = front[0], best = -Infinity;
  for (const q of front) { const u = (fx(q) - fx(a)) / dx, w = (fy(q) - fy(a)) / dy, dist = u - w; if (dist > best) { best = dist; knee = q; } }
  return { front, knee };
}
/** Hypervolume (area dominated relative to a reference point) of a two-objective set: maximise fx, minimise fy. */
export function hypervolume2(pts, fx, fy, refX, refY) {
  const f = paretoFront(pts.filter((q) => fx(q) > refX && fy(q) < refY), fx, fy).front.sort((a, b) => fx(b) - fx(a));
  let hv = 0, prevY = refY;
  for (const q of f) { if (fy(q) < prevY) { hv += (fx(q) - refX) * (prevY - fy(q)); prevY = fy(q); } }
  return hv;
}
/**
 * Flux-versus-energy designs of an MD module (hot-feed temperature, velocity, channel length) or an FO module (draw concentration, velocity).
 * A coarse grid of full module solutions is kept as backdrop; the Pareto front itself is searched with NSGA-II (suite 11) over the continuous variables
 * and merged with the grid, so it is never worse than the grid front. Deterministic for a given seed.
 */
export function paretoDesigns(v, S, md, { pop = 16, gens = 5, seed = 3 } = {}) {
  const grid = [], ga = [];
  let evalMD = null, evalFO = null, lo, hi, dec;
  if (md) {
    const Sm = v.mdRec > 0 ? Math.min(S / (1 - clamp(v.mdRec / 100, 0, 0.95)), 330) : S, Tlo = Math.min(Math.max(v.Tp + 13, 45), 86);
    evalMD = (Tf, u, L) => { try { const m = mdModule(v, Sm, { Tf, uFm: u, uPm: u, Lmd: L, nSeg: 8 }); if (m.fluxLMH > 0 && Number.isFinite(m.stec) && Number.isFinite(m.sec)) return { a: Tf, b: u, c: L, flux: m.fluxLMH, en: m.stec, aux: m.sec }; } catch { /* infeasible point */ } return null; };
    for (const Tf of [50, 60, 70, 80, 88].filter((t) => t > v.Tp + 12)) for (const u of [0.08, 0.15, 0.3, 0.6]) for (const L of [0.5, 1, 2.5]) { const q = evalMD(Tf, u, L); if (q) grid.push(q); }
    lo = [Tlo, Math.log(0.08), Math.log(0.5)]; hi = [88, Math.log(0.6), Math.log(2.5)]; dec = (x) => evalMD(x[0], Math.exp(x[1]), Math.exp(x[2]));
  } else {
    const d = DRAWS[v.draw] || DRAWS.nacl, cHi = Math.min(d.sol, Math.max(3, 2 * v.cDraw));
    evalFO = (cD, u) => { try { const r = simulateFO(v, { cDraw: cD, uF: u, uD: u, nSeg: 8 }), Vw = r.tot.Vw * 3600; if (!(Vw > 0)) return null; const reg = regeneration(r, { ...v, cDraw: cD }), en = r.Ppump / 1000 / Vw + reg.elec + 0.1 * reg.heat; if (Number.isFinite(en)) return { a: cD, b: u, c: r.dilution, flux: r.JwLMH, en, aux: r.srsf }; } catch { /* infeasible point */ } return null; };
    for (const cD of linspace(0.4, cHi, 6)) for (const u of [5, 10, 20, 35]) { const q = evalFO(cD, u); if (q) grid.push(q); }
    lo = [0.4, Math.log(5)]; hi = [Math.max(cHi, 0.41), Math.log(35)]; dec = (x) => evalFO(x[0], Math.exp(x[1]));
  }
  let evals = grid.length;
  if (grid.length > 2 && hi.every((h, k) => h > lo[k])) {
    const res = nsga2((x) => { const q = dec(x); evals++; if (!q) return { f: [0, 1e30], cv: 1 }; ga.push(q); return { f: [-q.flux, q.en], cv: 0 }; }, lo, hi, { pop, gens, seed });
    void res;
  }
  const all = [...grid, ...ga], fx = (q) => q.flux, fy = (q) => q.en, refX = 0, refY = 1.05 * Math.max(...all.map(fy), 1e-9);
  return { grid, ga, all, evals, seed, gridFront: paretoFront(grid, fx, fy).front.length, hvGrid: hypervolume2(grid, fx, fy, refX, refY), hvAll: hypervolume2(all, fx, fy, refX, refY) };
}
/**
 * Grey-box (physics-informed) correction of the mechanistic flux: ln(J_measured / J_model) = β₀ + β₁x₁ + β₂x₂ fitted by ridge regression,
 * so that the model keeps its structure and the data only shift it; leave-one-out errors show whether the correction generalises.
 */
export function greyBox(rows, mech, feat, lam = 1e-3) {
  const data = rows.map((q) => ({ q, m: mech(q), x: [1, ...feat(q)] })).filter((d) => d.m > 0 && d.q.Jw > 0 && d.x.every(Number.isFinite));
  if (data.length < 4) throw new Error('at least four usable test rows are needed for the grey-box correction');
  const fit = (set) => { const k = set[0].x.length, G = Array.from({ length: k }, () => new Array(k).fill(0)), g = new Array(k).fill(0); for (const d of set) { const y = Math.log(d.q.Jw / d.m); for (let a = 0; a < k; a++) { g[a] += d.x[a] * y; for (let b = 0; b < k; b++) G[a][b] += d.x[a] * d.x[b]; } } for (let a = 0; a < k; a++) G[a][a] += (a ? lam : 1e-9) * set.length; return solveLinear(G, g); };
  const corr = (beta, x) => Math.exp(clamp(sum(beta.map((b, a) => b * x[a])), -0.7, 0.7)), beta = fit(data), rm = (e) => Math.sqrt(sum(e.map((x) => x * x)) / e.length);
  const loo = data.map((d, k) => { const b = fit(data.filter((_, j) => j !== k)); return d.m * corr(b, d.x); });
  return { beta, n: data.length, meas: data.map((d) => d.q.Jw), mech: data.map((d) => d.m), fitted: data.map((d) => d.m * corr(beta, d.x)), loo, rmseMech: rm(data.map((d) => d.m - d.q.Jw)), rmseFit: rm(data.map((d) => d.m * corr(beta, d.x) - d.q.Jw)), rmseLoo: rm(data.map((d, k) => loo[k] - d.q.Jw)), predict: (q) => mech(q) * corr(beta, [1, ...feat(q)]), factor: (q) => corr(beta, [1, ...feat(q)]) };
}

// ---- shared helpers ---------------------------------------------------------------------------------------------
const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const round = (o) => Object.fromEntries(ION_IDS.map((k) => [k, +(+o[k] || 0).toPrecision(6)]));
const stream = (Q, T, pH, ions) => ({ Q, T, P: 1, pH, tds: tds(ions), ions: round(cloneIons(ions)) });
const isFO = (v) => ['fo', 'pro', 'fo_ro', 'fo_md', 'ed_fo'].includes(v.process), isMD = (v) => ['md', 'ro_md', 'fo_md', 'md_cr'].includes(v.process), needsHeat = (v) => isMD(v) || v.process === 'hdh', isCDI = (v) => v.process === 'cdi' || v.process === 'cdi_ro', hasDyn = (v) => v.process === 'fo' || v.process === 'md';
const mdIs = (...t) => (v) => isMD(v) && t.includes(v.mdType);
const tryOr = (f, d = null) => { try { const x = f(); return Number.isFinite(x) ? x : d; } catch { return d; } };
const clean = (plots) => plots.filter(Boolean).map((pl) => { if (pl.type !== 'line') return pl; const series = pl.series.map((s) => { const keep = s.y.map((y, i) => Number.isFinite(y) && Number.isFinite(s.x[i])); return { ...s, x: s.x.filter((_, i) => keep[i]), y: s.y.filter((_, i) => keep[i]) }; }).filter((s) => s.x.length); return { ...pl, series }; }).filter((pl) => pl.type !== 'line' || pl.series.length);

function heatSource(v, QkW, Tsupply, W) {
  if (v.source === 'solar') {
    const eta = clamp(v.etaColl / 100 - (0.004 * Math.max(0, Tsupply + 8 - 28)), 0.1, 0.9), E = QkW * 24 * (v.solarFrac / 100), area = E / (v.ghi * eta);
    return { title: 'Solar-thermal coupling', rows: [['Heat demand (kW)', QkW], ['Collector supply temperature (°C)', Tsupply + 8], ['Collector efficiency at this temperature (%)', 100 * eta], ['Solar heat required (kWh/d)', E], ['Collector area (m²)', area], ['Hot-water storage for 14 h at ΔT 15 K (m³)', (QkW * (v.solarFrac / 100) * 14 * 3.6e6) / (4180 * 15 * 985)]], note: `Daily irradiation ${v.ghi} kWh/m²·d; efficiency falls 0.4 %-points per kelvin above ambient.` };
  }
  if (v.source === 'waste') {
    const ok = v.Twh >= Tsupply + 5, cover = ok ? Math.min(1, v.Qwh / Math.max(QkW, 1e-9)) : 0;
    if (!ok) W.push({ level: 'bad', msg: `The waste-heat stream (${v.Twh} °C) is colder than the ${fmt(Tsupply + 5, 3)} °C needed to heat the feed to ${fmt(Tsupply, 3)} °C.` });
    return { title: 'Waste-heat coupling', rows: [['Heat demand (kW)', QkW], ['Waste heat available (kW)', v.Qwh], ['Minimum stream temperature (°C)', Tsupply + 5], ['Share of demand covered (%)', 100 * cover]], note: 'A 5 K pinch is assumed between the waste-heat stream and the feed heater outlet.' };
  }
  return null;
}

const GB_FO = [{ cDraw:0.5, uF:6, Jw:10.12 },
  { cDraw:0.5, uF:15, Jw:11.61 },
  { cDraw:0.5, uF:30, Jw:12.4 },
  { cDraw:1, uF:6, Jw:13.57 },
  { cDraw:1, uF:15, Jw:15.85 },
  { cDraw:1, uF:30, Jw:16.9 },
  { cDraw:1.5, uF:10, Jw:17.44 },
  { cDraw:1.5, uF:25, Jw:19.32 },
  { cDraw:2, uF:6, Jw:17.28 },
  { cDraw:2, uF:15, Jw:19.75 },
  { cDraw:2, uF:30, Jw:21.24 },
  { cDraw:0.75, uF:20, Jw:14.55 }];
const GB_MD = [{ Tf:45, uFm:0.1, Jw:8.27 },
  { Tf:45, uFm:0.3, Jw:8.52 },
  { Tf:55, uFm:0.1, Jw:16.22 },
  { Tf:55, uFm:0.25, Jw:16.44 },
  { Tf:55, uFm:0.5, Jw:16.31 },
  { Tf:65, uFm:0.1, Jw:28.43 },
  { Tf:65, uFm:0.25, Jw:28.85 },
  { Tf:65, uFm:0.5, Jw:29.72 },
  { Tf:75, uFm:0.15, Jw:48.12 },
  { Tf:75, uFm:0.3, Jw:49.52 },
  { Tf:80, uFm:0.25, Jw:62.64 },
  { Tf:80, uFm:0.5, Jw:64.82 }];

const suite = {
  id: 'fomd', num: 8, title: 'Forward Osmosis, Membrane Distillation & Emerging Desalination', short: 'FO · MD', icon: '🧪',
  tagline: 'Osmotically and thermally driven membrane processes with polarisation, wetting and energy analysis, capacitive deionisation with porous electrodes, and FO–RO, FO–MD, RO–MD, electrodialysis–FO, MD–crystalliser and RO–CDI chains.',
  description: 'Forward osmosis and pressure-retarded osmosis are solved with the solution–diffusion model including internal and external concentration polarisation, reverse solute flux and a co-/counter-current module model with a draw-solute library and regeneration energy. Membrane distillation (direct-contact, air-gap, vacuum and sweeping-gas) couples the dusty-gas vapour transport to the heat balance with temperature and concentration polarisation, conductive loss, liquid-entry pressure and a module model with heat recovery. Chained FO–RO, FO–MD, RO–MD, electrodialysis–FO, MD–crystalliser and RO–capacitive-polishing hybrids report each process contribution; a humidification–dehumidification cycle and a porous-electrode capacitive-deionisation cell (Gouy–Chapman–Stern double layers, transmission-line charging, Langmuir adsorption) are included as emerging processes. Optional modules add a log-normal pore-size distribution, batch dynamics with fouling, scaling and wetting, a membrane-material library, a flux–energy Pareto sweep and a grey-box flux correction.',
  guide: [
    'Choose the process. Forward osmosis needs a feed, a draw solution and a membrane; membrane distillation needs hot-feed and coolant temperatures and a hydrophobic membrane.',
    'Enter the feed analysis and flow (or pull the case feed, or an RO concentrate for brine concentration by MD).',
    'Adjust membrane properties and transport correlations on Model setup; calibrate A, B and S (FO) or tortuosity and heat-transfer multiplier (MD) against test data.',
    'Run. Check the polarisation losses, the wetting margin (MD) and the regeneration or heat demand; for hybrids read the per-process contribution table.',
  ],
  implemented: ['solution-diffusion', 'water-flux', 'solute-flux', 'vant hoff', 'external concentration-polarization', 'internal concentration-polarization', 'structural-parameter', 'mass-transfer film', 'knudsen-diffusion', 'molecular-diffusion', 'knudsen-molecular transition', 'dusty-gas', 'vapour-pressure', 'antoine', 'kelvin', 'heat-conduction', 'convective heat-transfer', 'latent-heat balance', 'temperature-polarization', 'poisson-nernst-planck', 'gouy-chapman', 'stern-layer', 'langmuir adsorption', 'porous-electrode charge-balance', 'donnan', 'pore-scale/continuum models', 'capacitive-deionization–ro',
    'fo-ro', 'ro-md', 'solar-md', 'heat-and-mass-transfer md', 'osmotic-hydraulic coupled', 'fo-md', 'md-crystallization', 'electrodialysis-fo', 'capacitive-deionization-ro', 'electrochemical-membrane', 'pore-scale/continuum', 'physics-informed emerging-process',
    'feed/draw concentration', 'temperatures', 'pressures', 'pore vapour state', 'feed/draw inlet', 'osmotic membrane-interface', 'vapour-liquid equilibrium', 'membrane heat/mass-flux continuity', 'convective thermal boundar', 'insulated boundar', 'membrane hydration', 'electrode charge', 'ion concentration as applicable', 'prescribed voltage/current', 'no-ion-flux',
    'membrane transport', 'draw-solution modelling', 'osmotic-property calculation', 'internal and external concentration polarisation', 'reverse-solute flux', 'porous-membrane transport', 'heat transfer', 'mass transfer', 'vapour transport', 'temperature polarisation', 'membrane wetting', 'scaling', 'module hydrodynamics', 'solar and waste-heat integration', 'hybrid-process configuration', 'energy analysis', 'sensitivity analysis', 'pore-scale transport', 'fouling', 'crystallisation', 'novel membrane-material', 'dynamic operation', 'multi-objective optimisation'],
  equationsNote: 'FO/PRO: steady solution–diffusion with film-theory ECP and a support-layer ICP described by the structural parameter (divided by the support hydration when the support is not fully wetted); draw osmotic coefficients are quadratic fits at 25 °C and the feed solutes are lumped into one species with the permeability of NaCl; a feed-side hydraulic pressure gives pressure-assisted FO. MD: one-dimensional film model with the dusty-gas membrane coefficient (Knudsen, molecular, their series combination, plus Poiseuille flow under vacuum) for the mean pore, optionally integrated over a log-normal pore-size distribution with every pore class in its own regime (bundle of parallel capillaries), or from a pore network: a two-dimensional lattice of throats with seeded log-normal radii between the two membrane faces, each throat with its own Knudsen / molecular (or viscous) conductance, solved as a linear network by banded Cholesky factorisation, with wetting by invasion percolation from the feed face (a throat is invaded when the pressure exceeds its entry pressure and it touches liquid; the breakthrough pressure of the lattice is reported). The network is two-dimensional and isothermal across the membrane; its coefficient is put on the same porosity, tortuosity and thickness basis as the continuum models; the vapour-pressure lowering of the brine follows the seawater correlation of the property library and the Kelvin correction is optional (below 1 % for 0.1–0.5 µm pores). Heat-transfer coefficients come from a spacer or open-channel Nusselt correlation. Capacitive deionisation: 1-D porous-electrode charge balance (transmission line) with a unified double layer of a 1:1 salt — Gouy–Chapman–Stern while the pore half-width exceeds four Debye lengths, modified Donnan (overlapping layers, with the micropore attraction term) below two, smoothly blended in between and re-evaluated as the pore liquid is depleted — so that co-ion expulsion, a charge efficiency below one and a positive salt capacity hold for feeds from a few mg/L to seawater; the pore conductivity includes the counter-ions of overlapped pores; a mixed spacer volume and Langmuir adsorption on the carbon complete the cell. Time integration is fully implicit (variable-step BDF2, Newton with a bordered-tridiagonal Jacobian) in conservative form, so the salt and charge balances close to round-off. The blend is an interpolation between the two limiting theories, not a solution of the Poisson–Boltzmann equation in a slit; the diffuse layer at the carbon surface is additionally resolved with the Poisson–Nernst–Planck solver of suite 7 as a check on the Gouy–Chapman relation; Faradaic side reactions are not modelled. Non-ionic draws (glucose) enter the ion-based RO and MD steps of the hybrids as the osmotically equivalent NaCl solution; electrodialysis cannot regenerate them and the hybrid then reports the FO step alone. Hybrid chains solve each process with its own model and link them by the shared streams; the electrodialysis step uses the stack model of suite 7 and the RO steps the element model of suite 1. The MD–crystalliser treats the total salinity as NaCl-equivalent with an MSMPR population balance (size-independent growth, secondary nucleation). Batch dynamics are lumped (well-mixed tanks, module flux scaled from the inlet coupon) with reduced-order kinetics for cake or deposit growth, surface scaling, contact-angle loss and progressive pore wetting; scaling is otherwise screened by NaCl saturation and a gypsum ratio scaled from seawater — use suite 2 for speciation. The Pareto front is searched with NSGA-II (elitist non-dominated sorting genetic algorithm, suite 11) over the continuous design variables and merged with a grid of full module solutions that is kept as backdrop; the grey-box correction is a three-coefficient ridge regression on the ratio of measured to modelled flux. Humidification–dehumidification is an effectiveness-based cycle model.',

  inputs: [
    { group: 'Process and feed', help: 'Which process is solved and what water it treats.', fields: [
      { key: 'process', label: 'Process', type: 'select', value: 'fo', options: [{ value: 'fo', label: 'Forward osmosis (FO)' }, { value: 'pro', label: 'Pressure-retarded osmosis (PRO)' }, { value: 'md', label: 'Membrane distillation (MD)' }, { value: 'fo_ro', label: 'Hybrid: FO + RO draw regeneration' }, { value: 'ro_md', label: 'Hybrid: RO + MD brine concentration' }, { value: 'fo_md', label: 'Hybrid: FO + MD draw regeneration' }, { value: 'ed_fo', label: 'Hybrid: FO + electrodialysis draw regeneration' }, { value: 'md_cr', label: 'Hybrid: MD + crystalliser (zero liquid discharge)' }, { value: 'hdh', label: 'Humidification–dehumidification (HDH)' }, { value: 'cdi', label: 'Capacitive deionisation — porous-electrode model' }, { value: 'cdi_ro', label: 'Hybrid: RO + capacitive-deionisation polishing' }], help: 'Each process shows its own inputs. The examples load realistic settings.' },
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.brackish.ions, help: 'Feed to the FO membrane, the MD loop or, in the RO–MD hybrid, the RO feed.' },
      { key: 'salinityFactor', label: 'Salinity multiplier', unit: '×', value: 1, min: 0.01, max: 8, help: 'Scales the whole analysis — convenient for sensitivity runs.' },
      { key: 'Qf', label: 'Feed flow', unit: 'm³/h', value: 50, min: 0.01, max: 1e5, help: 'FO: feed entering the module. MD and HDH: make-up feed. RO–MD: feed to the RO.' },
      { key: 'T', label: 'Feed supply temperature', unit: '°C', value: 25, min: 5, max: 45, help: 'FO operating temperature; for MD and HDH the temperature of the cold make-up.' },
      { key: 'pH', label: 'Feed pH', unit: '', value: 7.6, min: 2, max: 12, help: 'Passed on with the product and concentrate streams.' },
    ] },
    { group: 'Forward osmosis', showIf: isFO, help: 'Draw solution, module and flow arrangement.', fields: [
      { key: 'draw', label: 'Draw solute', type: 'select', value: 'nacl', options: Object.entries(DRAWS).map(([k, d]) => ({ value: k, label: d.name })), help: 'Sets osmotic pressure, diffusivity (ICP) and reverse leakage.' },
      { key: 'cDraw', label: 'Draw concentration at inlet', unit: 'mol/L', value: 1, min: 0.05, max: 6, help: 'Must give an osmotic pressure well above that of the concentrated feed.' },
      { key: 'Qd', label: 'Draw flow at inlet', unit: 'm³/h', value: 25, min: 0.01, max: 1e5, help: 'A larger draw flow is diluted less and keeps the driving force.' },
      { key: 'areaFO', label: 'Membrane area', unit: 'm²', value: 2500, min: 0.01, max: 1e6, help: 'Total active area of the FO modules.' },
      { key: 'orient', label: 'Membrane orientation', type: 'select', value: 'alfs', options: [{ value: 'alfs', label: 'Active layer facing feed (FO mode)' }, { value: 'alds', label: 'Active layer facing draw (PRO mode)' }], showIf: (v) => v.process !== 'pro', help: 'FO mode suffers dilutive ICP on the draw side; PRO mode gives higher flux but fouls the support.' },
      { key: 'flow', label: 'Flow arrangement', type: 'select', value: 'counter', options: [{ value: 'counter', label: 'Counter-current' }, { value: 'co', label: 'Co-current' }], help: 'Counter-current keeps a more uniform driving force.' },
      { key: 'dPfeed', label: 'Hydraulic pressure on the feed side (pressure-assisted FO)', unit: 'bar', value: 0, min: 0, max: 20, showIf: (v) => v.process !== 'pro', help: 'Adds to the osmotic driving force: J_w = A·(Δπ_eff + ΔP). 0 = ordinary FO.' },
      { key: 'edTarget', label: 'Product TDS after electrodialysis', unit: 'mg/L', value: 500, min: 20, max: 5000, showIf: (v) => v.process === 'ed_fo', help: 'Target salinity of the electrodialysis diluate, which is the product water.' },
      { key: 'dPpro', label: 'Hydraulic pressure on the draw side', unit: 'bar', value: 12, min: 0, max: 80, showIf: (v) => v.process === 'pro', help: 'Power density peaks near half of the osmotic-pressure difference.' },
      { key: 'etaTurb', label: 'Turbine / pressure-exchanger efficiency', unit: '%', value: 88, min: 30, max: 98, showIf: (v) => v.process === 'pro', help: 'Conversion of the pressurised permeate into electricity.' },
      { key: 'regen', label: 'Draw regeneration', type: 'select', value: 'ro', options: [{ value: 'ro', label: 'Reverse osmosis / nanofiltration' }, { value: 'thermal', label: 'Distillation or MD' }, { value: 'thermolytic', label: 'Thermolytic stripping (NH₄HCO₃)' }, { value: 'none', label: 'None — diluted draw is used directly' }], showIf: (v) => v.process === 'fo', help: 'How the diluted draw is re-concentrated to recover the product water.' },
    ] },
    { group: 'Membrane distillation', showIf: isMD, help: 'Configuration, temperatures and module.', fields: [
      { key: 'mdType', label: 'Configuration', type: 'select', value: 'dcmd', options: [{ value: 'dcmd', label: 'Direct contact (DCMD)' }, { value: 'agmd', label: 'Air gap (AGMD)' }, { value: 'vmd', label: 'Vacuum (VMD)' }, { value: 'sgmd', label: 'Sweeping gas (SGMD)' }], help: 'DCMD gives the highest flux; AGMD the best heat economy; VMD removes conduction losses.' },
      { key: 'Tf', label: 'Hot-feed inlet temperature', unit: '°C', value: 60, min: 30, max: 95, typical: [50, 85], help: 'Vapour pressure — and flux — rise exponentially with this temperature.' },
      { key: 'Tp', label: 'Coolant / permeate inlet temperature', unit: '°C', value: 20, min: 2, max: 60, help: 'Cold-side inlet (sweep-gas inlet for SGMD, condenser coolant for VMD).' },
      { key: 'mdRec', label: 'Water recovery of the MD loop', unit: '%', value: 0, min: 0, max: 95, help: '0 = single pass through the module. Above 0 the feed is recirculated (feed-and-bleed) and the modules see the concentrated loop salinity.' },
      { key: 'uFm', label: 'Feed velocity', unit: 'm/s', value: 0.25, min: 0.01, max: 2, help: 'Cross-flow velocity in the hot channel.' },
      { key: 'uPm', label: 'Coolant / permeate velocity', unit: 'm/s', value: 0.25, min: 0.01, max: 2, showIf: mdIs('dcmd', 'agmd'), help: 'Cross-flow velocity in the cold channel.' },
      { key: 'Lmd', label: 'Channel length', unit: 'm', value: 1, min: 0.05, max: 12, help: 'Longer channels recover more heat but lose driving force.' },
      { key: 'hF', label: 'Channel height', unit: 'mm', value: 2, min: 0.5, max: 10, help: 'Height of the feed and coolant channels.' },
      { key: 'flowMD', label: 'Flow arrangement', type: 'select', value: 'counter', options: [{ value: 'counter', label: 'Counter-current' }, { value: 'co', label: 'Co-current' }], showIf: mdIs('dcmd', 'agmd'), help: 'Counter-current keeps the temperature difference along the module. Sweeping-gas and vacuum modules are solved co-currently.' },
      { key: 'gapMD', label: 'Air-gap width', unit: 'mm', value: 2, min: 0.3, max: 10, showIf: mdIs('agmd'), help: 'Stagnant air between membrane and condensing plate.' },
      { key: 'Pvac', label: 'Permeate-side absolute pressure', unit: 'kPa', value: 6, min: 0.8, max: 60, showIf: mdIs('vmd'), help: 'Must be below the feed vapour pressure and above the condenser saturation pressure.' },
      { key: 'uGas', label: 'Sweep-gas velocity', unit: 'm/s', value: 2, min: 0.1, max: 15, showIf: mdIs('sgmd'), help: 'Air velocity in the permeate channel.' },
      { key: 'rhGas', label: 'Sweep-gas inlet humidity', unit: '%', value: 30, min: 0, max: 100, showIf: mdIs('sgmd'), help: 'Relative humidity of the gas entering at the coolant temperature.' },
      { key: 'recover', label: 'Heat recovery', type: 'bool', value: true, showIf: mdIs('dcmd', 'agmd'), help: 'DCMD: external exchanger between warm permeate and returning feed. AGMD: the coolant is the feed itself.' },
      { key: 'effHX', label: 'Heat-recovery exchanger effectiveness', unit: '%', value: 80, min: 0, max: 97, showIf: (v) => isMD(v) && v.mdType === 'dcmd' && v.recover, help: 'Effectiveness of the external exchanger.' },
    ] },
    { group: 'RO stage of the hybrid', showIf: (v) => v.process === 'ro_md' || v.process === 'fo_ro' || v.process === 'cdi_ro', help: 'The reverse-osmosis step is solved with the element-by-element model of suite 1.', fields: [
      { key: 'roRec', label: 'RO recovery', unit: '%', value: 45, min: 10, max: 85, showIf: (v) => v.process === 'ro_md' || v.process === 'cdi_ro', help: 'Recovery of the seawater or brackish RO ahead of the MD brine concentrator.' },
      { key: 'roFlux', label: 'RO design flux', unit: 'L/m²·h', value: 14, min: 5, max: 35, help: 'Average flux used to size the RO array.' },
    ] },
    { group: 'Crystalliser of the MD loop', showIf: (v) => v.process === 'md_cr', help: 'Mixed-suspension mixed-product-removal crystalliser fed by the saturated MD loop.', fields: [
      { key: 'Tcr', label: 'Crystalliser temperature', unit: '°C', value: 40, min: 10, max: 80, help: 'Sets the solubility of the loop; keep it below the MD feed temperature so that the membrane sees undersaturated brine.' },
      { key: 'tauCr', label: 'Crystal residence time', unit: 'h', value: 2, min: 0.2, max: 12, help: 'Suspension volume ÷ slurry withdrawal; longer times give larger crystals at lower supersaturation.' },
      { key: 'slurry', label: 'Suspension density', unit: 'kg crystals per m³', value: 150, min: 10, max: 500, help: 'Crystal mass held per m³ of suspension.' },
      { key: 'bleed', label: 'Mother-liquor bleed', unit: '% of feed mass', value: 2, min: 0, max: 50, help: 'Purge of impurities; 0 = zero liquid discharge.' },
    ] },
    { group: 'Capacitive deionisation cell', showIf: isCDI, help: 'Pair of porous carbon electrodes with a flow-by spacer; in the RO hybrid it polishes the RO permeate.', fields: [
      { key: 'cdiMode', label: 'Electrical boundary condition', type: 'select', value: 'cv', options: [{ value: 'cv', label: 'Prescribed voltage (constant voltage)' }, { value: 'cc', label: 'Prescribed current (constant current, voltage-limited)' }], help: 'Constant voltage charges fastest; constant current gives a steady effluent concentration.' },
      { key: 'cdiV', label: 'Charging voltage', unit: 'V', value: 1.2, min: 0.2, max: 1.6, help: 'Cell voltage (upper limit in constant-current operation). Keep below about 1.23 V.' },
      { key: 'cdiI', label: 'Charging / discharging current density', unit: 'A/m²', value: 15, min: 0.5, max: 500, showIf: (v) => v.cdiMode === 'cc', help: 'Per m² of cell area.' },
      { key: 'cdiVdis', label: 'Discharge voltage', unit: 'V', value: 0, min: 0, max: 1, help: '0 V = short-circuit regeneration.' },
      { key: 'cdiTc', label: 'Charging (adsorption) time', unit: 'min', value: 10, min: 0.2, max: 120, help: 'Half-cycle producing desalinated water.' },
      { key: 'cdiTd', label: 'Discharge (desorption) time', unit: 'min', value: 10, min: 0.2, max: 120, help: 'Half-cycle producing concentrate.' },
      { key: 'cdiQ', label: 'Flow per cell area', unit: 'L/m²·min', value: 0.3, min: 0.05, max: 20, help: 'Spacer throughput.' },
      { key: 'cdiCycles', label: 'Cycles simulated', unit: '', value: 2, min: 1, max: 8, step: 1, help: 'The last cycle is reported; with short-circuit discharge the second cycle is already close to the periodic state.' },
      { key: 'cdiRecov', label: 'Energy recovered on discharge', unit: '%', value: 0, min: 0, max: 95, help: 'Share of the discharge energy returned by the power electronics.' },
    ] },
    { group: 'Humidification–dehumidification', showIf: (v) => v.process === 'hdh', help: 'Closed-air, open-water cycle with a seawater heater.', fields: [
      { key: 'Ttop', label: 'Top seawater temperature', unit: '°C', value: 75, min: 45, max: 95, help: 'Seawater temperature leaving the heater and entering the humidifier.' },
      { key: 'MR', label: 'Seawater-to-dry-air mass ratio', unit: '–', value: 3, min: 0.5, max: 12, help: 'The gain-output ratio peaks where the heat capacities of the two streams balance.' },
      { key: 'effH', label: 'Humidifier effectiveness', unit: '%', value: 85, min: 40, max: 98, help: 'Energy-based effectiveness.' },
      { key: 'effD', label: 'Dehumidifier effectiveness', unit: '%', value: 85, min: 40, max: 98, help: 'Energy-based effectiveness.' },
      { key: 'dpAir', label: 'Air-loop pressure drop', unit: 'Pa', value: 400, min: 20, max: 5000, help: 'For the fan power.' },
    ] },
    { group: 'Heat source', showIf: needsHeat, help: 'Where the low-grade heat comes from.', fields: [
      { key: 'source', label: 'Heat source', type: 'select', value: 'heat', options: [{ value: 'heat', label: 'Hot water or steam (no sizing)' }, { value: 'solar', label: 'Solar-thermal collectors' }, { value: 'waste', label: 'Waste-heat stream' }], help: 'Adds a sizing table to the results.' },
      { key: 'ghi', label: 'Daily solar irradiation', unit: 'kWh/m²·d', value: 5.8, min: 0.5, max: 10, showIf: (v) => v.source === 'solar', help: 'Annual-mean global horizontal irradiation at the site.' },
      { key: 'etaColl', label: 'Collector efficiency near ambient', unit: '%', value: 65, min: 20, max: 85, showIf: (v) => v.source === 'solar', help: 'Optical efficiency; thermal losses are subtracted for the supply temperature.' },
      { key: 'solarFrac', label: 'Solar fraction', unit: '%', value: 80, min: 5, max: 100, showIf: (v) => v.source === 'solar', help: 'Share of the daily heat demand supplied by the collectors.' },
      { key: 'Twh', label: 'Waste-heat stream temperature', unit: '°C', value: 85, min: 35, max: 300, showIf: (v) => v.source === 'waste', help: 'Supply temperature of the waste-heat carrier.' },
      { key: 'Qwh', label: 'Waste heat available', unit: 'kW', value: 2000, min: 1, max: 1e7, showIf: (v) => v.source === 'waste', help: 'Heat that can be extracted from the stream.' },
    ] },
    { group: 'FO membrane and transport model', tab: 'setup', showIf: isFO, help: 'Solution–diffusion parameters and polarisation correlations.', fields: [
      { key: 'AFO', label: 'Water permeability A (25 °C)', unit: 'L/m²·h·bar', value: 2.2, min: 0.1, max: 12, help: 'Thin-film composite FO membranes: 1–5; cellulose triacetate: 0.4–0.8.' },
      { key: 'BFO', label: 'Salt permeability B, NaCl (25 °C)', unit: 'L/m²·h', value: 0.45, min: 0.005, max: 10, help: 'Other solutes are scaled from it with the library ratio.' },
      { key: 'Sfo', label: 'Structural parameter S', unit: 'µm', value: 400, min: 0, max: 5000, typical: [200, 800], help: 'S = thickness × tortuosity ÷ porosity of the support layer; it controls internal polarisation.' },
      { key: 'hydration', label: 'Support-layer hydration (initial)', unit: '% of pore volume wetted', value: 100, min: 20, max: 100, help: 'A hydrophobic support that is not fully wetted has fewer open pores: S_eff = S ÷ hydration. In the batch simulation this is the initial condition and the support wets out with time.' },
      { key: 'uF', label: 'Feed-channel velocity', unit: 'cm/s', value: 15, min: 0.5, max: 100, help: 'Sets the external mass-transfer coefficient on the feed side.' },
      { key: 'uD', label: 'Draw-channel velocity', unit: 'cm/s', value: 15, min: 0.5, max: 100, help: 'Sets the external mass-transfer coefficient on the draw side.' },
      { key: 'hch', label: 'Channel (spacer) height', unit: 'mm', value: 0.8, min: 0.2, max: 5, help: 'Both channels.' },
      { key: 'Lfo', label: 'Flow-path length', unit: 'm', value: 3, min: 0.05, max: 30, help: 'Used only for the pressure drop and pumping energy.' },
      { key: 'kcp', label: 'Mass-transfer multiplier', unit: '×', value: 1, min: 0.1, max: 10, help: 'Scales Sh = 0.065 Re^0.875 Sc^0.25.' },
      { key: 'etaERD', label: 'Energy-recovery efficiency of the regeneration RO', unit: '%', value: 95, min: 0, max: 99, showIf: (v) => v.process === 'fo' && v.regen === 'ro', help: 'Isobaric pressure exchanger on the draw concentrate.' },
      { key: 'gorRegen', label: 'Gain-output ratio of thermal regeneration', unit: '–', value: 5, min: 0.5, max: 16, showIf: (v) => v.process === 'fo' && v.regen === 'thermal', help: 'kg of water evaporated per kg-equivalent of heating steam.' },
      { key: 'hStrip', label: 'Stripping heat per mole of draw solute', unit: 'kJ/mol', value: 150, min: 40, max: 600, showIf: (v) => v.process === 'fo' && v.regen === 'thermolytic', help: 'Decomposition enthalpy of ammonium bicarbonate plus column reboiler losses.' },
    ] },
    { group: 'MD membrane', tab: 'setup', showIf: isMD, help: 'Hydrophobic microporous membrane.', fields: [
      { key: 'dPore', label: 'Mean pore diameter', unit: 'µm', value: 0.2, min: 0.01, max: 2, typical: [0.1, 0.45], help: 'Controls the Knudsen number and the membrane coefficient.' },
      { key: 'poreDist', label: 'Pore-size model', type: 'select', value: 'mean', options: [{ value: 'mean', label: 'Single mean pore (continuum)' }, { value: 'lognormal', label: 'Log-normal pore-size distribution (bundle of parallel capillaries)' }, { value: 'network', label: 'Pore network (2-D lattice of throats, invasion-percolation wetting)' }], help: 'Bundle: every pore class carries the dusty-gas flux of its own Knudsen number, weighted by its open area (pores in parallel). Network: log-normal throats on a lattice between the two faces, solved as a linear conductance network — narrow throats in series throttle the flux, cross-links let it by-pass them, and liquid invades from the feed side through connected wide throats.' },
      { key: 'sigmaPore', label: 'Geometric standard deviation of the pore sizes', unit: '–', value: 1.3, min: 1, max: 2.5, showIf: (v) => v.poreDist === 'lognormal' || v.poreDist === 'network', help: '1 = uniform pores; commercial MD membranes show 1.1–1.5.' },
      { key: 'rMax', label: 'Largest pore radius', unit: 'µm', value: 0.2, min: 0.01, max: 3, help: 'Sets the liquid-entry pressure (the largest pores wet first).' },
      { key: 'epsM', label: 'Porosity', unit: '–', value: 0.8, min: 0.2, max: 0.95, help: 'Void fraction of the membrane.' },
      { key: 'tauM', label: 'Tortuosity', unit: '–', value: 2, min: 1, max: 6, help: 'Often estimated as (2 − ε)²/ε. Calibrate from a flux test.' },
      { key: 'deltaM', label: 'Thickness', unit: 'µm', value: 100, min: 5, max: 600, help: 'Thin membranes give more flux but more conductive heat loss.' },
      { key: 'kPoly', label: 'Polymer thermal conductivity', unit: 'W/m·K', value: 0.19, min: 0.05, max: 0.6, help: 'PVDF 0.17–0.19, PTFE 0.25–0.27, PP 0.11–0.17.' },
      { key: 'theta', label: 'Contact angle', unit: '°', value: 120, min: 91, max: 175, help: 'Water contact angle of the membrane surface.' },
      { key: 'lepB', label: 'Pore-geometry factor', unit: '–', value: 1, min: 0.3, max: 1, help: '1 for cylindrical pores.' },
      { key: 'gammaF', label: 'Surface-tension factor of the feed', unit: '×', value: 1, min: 0.3, max: 1.1, help: 'Below 1 for feeds with surfactants, oils or alcohols.' },
      { key: 'pFeed', label: 'Feed-side gauge pressure', unit: 'bar', value: 0.5, min: 0, max: 5, help: 'Highest hydraulic pressure at the membrane.' },
    ] },
    { group: 'MD transport model', tab: 'setup', showIf: isMD, help: 'Vapour-transport regime and channel correlations.', fields: [
      { key: 'mdModel', label: 'Vapour transport through the pores', type: 'select', value: 'auto', options: [{ value: 'auto', label: 'Dusty-gas, regime from the Knudsen number' }, { value: 'transition', label: 'Knudsen + molecular in series' }, { value: 'knudsen', label: 'Knudsen diffusion only' }, { value: 'molecular', label: 'Molecular diffusion only' }], help: 'Under vacuum the Knudsen and viscous (Poiseuille) terms act in parallel.' },
      { key: 'vpModel', label: 'Vapour-pressure equation', type: 'select', value: 'iapws', options: [{ value: 'iapws', label: 'IAPWS-IF97 saturation curve' }, { value: 'antoine', label: 'Antoine equation' }], help: 'Both are multiplied by the water activity of the brine; they agree within about 0.5 % between 20 and 90 °C.' },
      { key: 'kelvin', label: 'Kelvin (curved-interface) correction', type: 'bool', value: false, help: 'Raises the vapour pressure at the pore mouth; below 1 % for common pore sizes.' },
      { key: 'spacerMD', label: 'Spacer-filled channels', type: 'bool', value: true, help: 'Spacers roughly double heat transfer at the price of pressure drop.' },
      { key: 'nuA', label: 'Spacer Nusselt coefficient', unit: '–', value: 1.15, min: 0.2, max: 4, showIf: (v) => v.spacerMD, help: 'Nu = a·Re^0.5·Pr^⅓ (Phattaranawik-type correlation with typical spacer geometry).' },
      { key: 'fh', label: 'Heat- and mass-transfer multiplier', unit: '×', value: 1, min: 0.2, max: 5, help: 'Scales both film coefficients. Calibrate from the temperature-polarisation of a test.' },
      { key: 'plateT', label: 'Condensing-plate thickness', unit: 'mm', value: 0.5, min: 0.05, max: 5, showIf: mdIs('agmd'), help: 'Cooling plate of the air-gap module.' },
      { key: 'plateK', label: 'Condensing-plate conductivity', unit: 'W/m·K', value: 15, min: 0.1, max: 400, showIf: mdIs('agmd'), help: 'Stainless steel 15, polymer film 0.2.' },
    ] },
    { group: 'Membrane material', tab: 'setup', showIf: (v) => isFO(v) || isMD(v), help: 'Library of conventional and novel membrane materials; every run also lists how each of them would perform here.', fields: [
      { key: 'memMat', label: 'Membrane material', type: 'select', value: 'custom', options: [{ value: 'custom', label: 'Custom — use the membrane fields as entered' }, ...Object.entries(MATERIALS).map(([k, m]) => ({ value: k, label: `${m.kind === 'md' ? 'MD' : 'FO'} · ${m.name}` }))], help: 'A library entry replaces the membrane properties of its process family (MD: pore size, porosity, tortuosity, thickness, conductivity, contact angle; FO: A, B and S). Entries of the other family are ignored.' },
    ] },
    { group: 'Porous electrode and double layer', tab: 'setup', showIf: isCDI, help: 'Transmission-line model of the carbon electrodes with a unified double layer (Gouy–Chapman–Stern for thin layers, modified Donnan for overlapping ones) and Langmuir adsorption.', fields: [
      { key: 'cdiLe', label: 'Electrode thickness', unit: 'µm', value: 400, min: 20, max: 2000, help: 'Each electrode; the charging time constant grows with its square.' },
      { key: 'cdiPor', label: 'Macroporosity of the electrode', unit: '–', value: 0.4, min: 0.1, max: 0.8, help: 'Electrolyte-filled transport pores; pore conductivity = κ·porosity^1.5.' },
      { key: 'cdiRho', label: 'Electrode density', unit: 'g/cm³', value: 0.5, min: 0.1, max: 1.5, help: 'Carbon mass per electrode volume.' },
      { key: 'cdiArea', label: 'Double-layer area of the meso- and macropores', unit: 'm²/g', value: 80, min: 5, max: 3000, help: 'Wall area of the electrolyte-filled pores. Together with the porosity it sets the pore half-width (pore volume ÷ area); the double-layer model switches from Gouy–Chapman–Stern to modified Donnan when that width falls below a few Debye lengths, so microporous carbons (large area) and dilute feeds are covered too.' },
      { key: 'cdiEdl', label: 'Double-layer model', type: 'select', value: 'auto', options: [{ value: 'auto', label: 'Unified: Gouy–Chapman–Stern ↔ modified Donnan, switched on pore size ÷ Debye length' }, { value: 'gcs', label: 'Gouy–Chapman–Stern only (thin layers)' }, { value: 'md', label: 'Modified Donnan only (overlapping layers)' }], help: 'The unified model uses the thin-layer relations while the pore half-width exceeds four Debye lengths, the modified-Donnan relations below two, and blends smoothly in between — also when the pore liquid is depleted during charging. It has a solution for every feed from a few mg/L to seawater.' },
      { key: 'cdiMu', label: 'Ion attraction into overlapped pores', unit: 'kT', value: 1.5, min: 0, max: 4, help: 'Non-electrostatic attraction term of the modified-Donnan model. It is faded out as the feed makes the double layers thin, so it does not act on wide pores.' },
      { key: 'cdiCst', label: 'Stern-layer capacitance', unit: 'F/m²', value: 0.2, min: 0.02, max: 2, help: 'Compact-layer capacitance in series with the diffuse (Gouy–Chapman) layer.' },
      { key: 'cdiHsp', label: 'Spacer thickness', unit: 'µm', value: 200, min: 30, max: 2000, help: 'Flow channel between the electrodes (porosity 0.7).' },
      { key: 'cdiRc', label: 'Contact and lead resistance', unit: 'Ω·cm²', value: 10, min: 0, max: 500, help: 'Electronic resistances of the cell.' },
      { key: 'cdiQ0', label: 'Initial electrode charge', unit: '% of equilibrium charge', value: 0, min: 0, max: 100, help: 'Initial condition of the first cycle: 0 = fully discharged electrodes.' },
      { key: 'lgQm', label: 'Langmuir adsorption capacity', unit: 'mg NaCl per g', value: 1, min: 0, max: 30, help: 'Non-electrostatic (physical) adsorption capacity of the carbon.' },
      { key: 'lgK', label: 'Langmuir constant K', unit: 'm³/mol', value: 0.05, min: 0.0001, max: 10, help: 'Equilibrium coverage θ = K·c / (1 + K·c).' },
      { key: 'lgKa', label: 'Langmuir rate constant', unit: '1/min', value: 0.5, min: 0.001, max: 60, help: 'dθ/dt = k·[K·c·(1 − θ) − θ].' },
      { key: 'lgTheta0', label: 'Initial Langmuir coverage', unit: '% of equilibrium', value: 100, min: 0, max: 100, help: '100 = electrode pre-equilibrated with the feed; 0 = fresh electrode.' },
    ] },
    { group: 'Crystallisation kinetics', tab: 'setup', showIf: (v) => v.process === 'md_cr', help: 'Growth and secondary nucleation of the MSMPR crystalliser.', fields: [
      { key: 'kgCr', label: 'Growth-rate constant k_g', unit: '10⁻⁶ m/s', value: 3, min: 0.01, max: 100, help: 'G = k_g·σ with σ the relative supersaturation.' },
      { key: 'kbCr', label: 'Nucleation-rate constant k_b', unit: '10⁸ per kg·s', value: 7, min: 0.001, max: 1e5, help: 'B₀ = k_b·M_T·σ² (secondary nucleation, proportional to the suspension density).' },
    ] },
    { group: 'Dynamic batch operation', tab: 'setup', showIf: hasDyn, help: 'Time simulation of a feed batch through the installed membrane area with fouling, scaling, wetting (MD) or support hydration (FO).', fields: [
      { key: 'dynamic', label: 'Simulate batch operation over time', type: 'bool', value: false, help: 'Adds time histories of flux, salinity, deposits and wetting.' },
      { key: 'tDyn', label: 'Simulated time', unit: 'h', value: 24, min: 0.1, max: 2000, showIf: (v) => v.dynamic, help: 'Duration of the batch.' },
      { key: 'batchVol', label: 'Feed batch volume', unit: 'm³', value: 10, min: 0.001, max: 1e5, showIf: (v) => v.dynamic, help: 'Initial content of the feed tank.' },
      { key: 'drawVol', label: 'Draw batch volume', unit: 'm³', value: 5, min: 0.001, max: 1e5, showIf: (v) => v.dynamic && v.process === 'fo', help: 'Initial content of the draw tank.' },
      { key: 'cFou', label: 'Foulant concentration in the feed', unit: 'mg/L', value: 5, min: 0, max: 500, showIf: (v) => v.dynamic, help: 'Colloids and organics carried to the membrane by the water flux.' },
      { key: 'kRem', label: 'Deposit removal rate', unit: '1/h', value: 0.02, min: 0, max: 5, showIf: (v) => v.dynamic, help: 'First-order shear removal of the deposit or cake.' },
      { key: 'alphaCake', label: 'Specific cake resistance', unit: '10¹³ m/kg', value: 5, min: 0, max: 1000, showIf: (v) => v.dynamic && v.process === 'fo', help: 'Hydraulic resistance added per kg/m² of cake.' },
      { key: 'tauHyd', label: 'Support wet-out time constant', unit: 'h', value: 2, min: 0.01, max: 200, showIf: (v) => v.dynamic && v.process === 'fo', help: 'Time over which a partly hydrated support layer wets out.' },
      { key: 'kScaleMD', label: 'Surface scaling coefficient', unit: 'g/m²·h', value: 5, min: 0, max: 500, showIf: (v) => v.dynamic && v.process === 'md', help: 'Scale growth = k·(Ω_wall − 1)² above saturation at the membrane surface.' },
      { key: 'thetaDrop', label: 'Contact-angle loss of a fouled surface', unit: '°', value: 25, min: 0, max: 90, showIf: (v) => v.dynamic && v.process === 'md', help: 'Deposits make the surface less hydrophobic; half of the loss is reached at 5 g/m².' },
      { key: 'tauWet', label: 'Pore-wetting time constant', unit: 'h', value: 6, min: 0.01, max: 500, showIf: (v) => v.dynamic && v.process === 'md', help: 'Lag with which pores above the critical radius fill with liquid.' },
      { key: 'wet0', label: 'Initially wetted pore area', unit: '%', value: 0, min: 0, max: 100, showIf: (v) => v.dynamic && v.process === 'md', help: 'Initial condition of the membrane (0 = dry pores).' },
    ] },
    { group: 'Multi-objective sweep and grey-box model', tab: 'setup', showIf: hasDyn, help: 'Design trade-offs and data-driven correction of the mechanistic flux.', fields: [
      { key: 'pareto', label: 'Pareto sweep: flux versus specific energy', type: 'bool', value: false, help: 'Multi-objective optimisation with NSGA-II over the continuous design variables (MD: temperature, velocity, length; FO: draw strength, velocity); a grid of designs is shown as backdrop.' },
      { key: 'greybox', label: 'Grey-box correction from test data', type: 'bool', value: false, help: 'Fits a three-coefficient correction of the modelled flux to the test rows below and reports whether it generalises (leave-one-out).' },
      { key: 'gbFO', label: 'FO test data', type: 'table', columns: [{ key: 'cDraw', label: 'Draw concentration', unit: 'mol/L' }, { key: 'uF', label: 'Cross-flow velocity', unit: 'cm/s' }, { key: 'Jw', label: 'Measured water flux', unit: 'L/m²·h' }], value: GB_FO, showIf: (v) => v.greybox && v.process === 'fo', help: 'Coupon tests with the present feed and membrane.' },
      { key: 'gbMD', label: 'MD test data', type: 'table', columns: [{ key: 'Tf', label: 'Hot-feed temperature', unit: '°C' }, { key: 'uFm', label: 'Feed velocity', unit: 'm/s' }, { key: 'Jw', label: 'Measured flux', unit: 'L/m²·h' }], value: GB_MD, showIf: (v) => v.greybox && v.process === 'md', help: 'Flat-sheet tests at the present coolant temperature and feed salinity.' },
    ] },
    { group: 'Pumps', tab: 'setup', help: 'For the electrical energy of circulation.', fields: [
      { key: 'etaPump', label: 'Pump + motor efficiency', unit: '%', value: 70, min: 20, max: 92, help: 'Applied to all circulation pumps and fans.' },
    ] },
    { group: 'Discretisation', tab: 'mesh', help: 'Number of segments along the membrane in the module models; cells and time steps of the dynamic models.', showIf: (v) => v.process !== 'hdh', fields: [
      { key: 'nSeg', label: 'Segments along the module', unit: '', value: 24, min: 2, max: 400, step: 1, showIf: (v) => v.process !== 'cdi', help: 'Use the sensitivity study to confirm that the result no longer depends on it.' },
      { key: 'pnNx', label: 'Pore network: throats across the lattice', unit: '', value: 24, min: 6, max: 64, step: 1, showIf: (v) => isMD(v) && v.poreDist === 'network', help: 'In-plane size of the 2-D lattice (periodic). Larger lattices average out the random radii.' },
      { key: 'pnNz', label: 'Pore network: pore bodies through the thickness', unit: '', value: 12, min: 3, max: 40, step: 1, showIf: (v) => isMD(v) && v.poreDist === 'network', help: 'Rows of pore bodies between the feed face and the permeate face.' },
      { key: 'pnSeed', label: 'Pore network: random seed', unit: '', value: 7, min: 1, max: 9999, step: 1, showIf: (v) => isMD(v) && v.poreDist === 'network', help: 'Seed of the log-normal throat radii; change it to see the sample-to-sample scatter.' },
      { key: 'cdiNx', label: 'Cells across the electrode', unit: '', value: 8, min: 3, max: 60, step: 1, showIf: isCDI, help: 'Finite volumes of the porous-electrode charge balance. The time integration is implicit, so the grid is used as entered whatever the salinity or cycle length.' },
      { key: 'cdiNt', label: 'Output steps per half-cycle', unit: '', value: 120, min: 20, max: 600, step: 1, showIf: isCDI, help: 'Each output interval is covered by two implicit (BDF2) steps; the first interval of every half-cycle is graded from the fastest electrode time constant to resolve the current peak.' },
      { key: 'ntDyn', label: 'Time steps of the batch simulation', unit: '', value: 200, min: 20, max: 2000, step: 1, showIf: (v) => hasDyn(v) && v.dynamic, help: 'RK4 steps over the simulated time.' },
    ] },
  ],

  presets: [
    { name: 'FO of brackish water with 1 mol/L NaCl draw, RO regeneration', values: {} },
    { name: 'FO seawater desalination with thermolytic NH₄HCO₃ draw', values: { ions: WATERS.seawater.ions, pH: 8.1, draw: 'nh4hco3', cDraw: 2.5, Qd: 40, Qf: 50, areaFO: 3000, regen: 'thermolytic' } },
    { name: 'PRO: river water against seawater-strength draw', values: { process: 'pro', ions: WATERS.lowbrackish.ions, pH: 7.8, draw: 'nacl', cDraw: 0.6, Qf: 100, Qd: 100, areaFO: 5000, dPpro: 12, Sfo: 300 } },
    { name: 'DCMD of seawater, 60 / 20 °C, 50 % loop recovery', values: { process: 'md', ions: WATERS.seawater.ions, pH: 8.1, Qf: 5, mdRec: 50 } },
    { name: 'AGMD with internal heat recovery, 80 °C, solar heat', values: { process: 'md', mdType: 'agmd', ions: WATERS.seawater.ions, pH: 8.1, Tf: 80, Tp: 25, Lmd: 5, uFm: 0.08, uPm: 0.08, Qf: 2, source: 'solar' } },
    { name: 'Hybrid RO + MD brine concentration', values: { process: 'ro_md', ions: WATERS.seawater.ions, pH: 8.1, Qf: 100, Tf: 70, mdRec: 55 } },
    { name: 'Humidification–dehumidification, 75 °C', values: { process: 'hdh', ions: WATERS.seawater.ions, pH: 8.1, Qf: 5 } },
    { name: 'Hybrid FO + MD draw regeneration', values: { process: 'fo_md', cDraw: 1.5, Qd: 25, Tf: 60, Tp: 20 } },
    { name: 'Hybrid FO + electrodialysis with a dilute draw', values: { process: 'ed_fo', cDraw: 0.3, Qd: 20, areaFO: 2500 } },
    { name: 'MD + crystalliser on RO brine (zero liquid discharge)', values: { process: 'md_cr', ions: WATERS.robrine.ions, pH: 7.9, Qf: 5, Tf: 65, Tp: 20 } },
    { name: 'Capacitive deionisation of 1 g/L brackish water, porous electrodes', values: { process: 'cdi', ions: WATERS.lowbrackish.ions, pH: 7.8, Qf: 5 } },
    { name: 'Hybrid seawater RO + capacitive polishing of the permeate', values: { process: 'cdi_ro', ions: WATERS.seawater.ions, pH: 8.1, Qf: 100, roRec: 45, cdiQ: 1, cdiV: 1.0 } },
  ],

  pull: ({ feed, outputs }) => [
    feed?.ions ? { key: 'ions', value: feed.ions, from: 'Case feed water' } : null, feed?.Q ? { key: 'Qf', value: feed.Q, from: 'Case feed water' } : null,
    feed?.T ? { key: 'T', value: clamp(feed.T, 5, 45), from: 'Case feed water' } : null, feed?.pH ? { key: 'pH', value: feed.pH, from: 'Case feed water' } : null,
    outputs?.ro?.streams?.concentrate?.ions ? { key: 'ions', value: outputs.ro.streams.concentrate.ions, from: 'RO concentrate (brine concentration by MD)' } : null,
    outputs?.ro?.streams?.concentrate?.Q ? { key: 'Qf', value: outputs.ro.streams.concentrate.Q, from: 'RO concentrate flow' } : null,
    outputs?.ro?.recovery ? { key: 'roRec', value: clamp(100 * outputs.ro.recovery, 10, 85), from: 'RO design recovery' } : null,
  ],
  site: (site) => [
    { key: 'T', value: site?.data?.sst !== undefined && site?.data?.sst !== null ? clamp(site.data.sst, 5, 45) : undefined, from: 'Sea-surface temperature at site' },
    { key: 'Tp', value: site?.data?.sst !== undefined && site?.data?.sst !== null ? clamp(site.data.sst, 2, 60) : undefined, from: 'Sea-surface temperature (MD coolant)' },
    { key: 'ghi', value: site?.data?.ghiDaily, from: 'Daily solar irradiation at site' },
  ],

  run(v0, ctx) { // synchronous by default; with the Pareto search or the pore network it yields once first (when the caller can) so the interface shows progress
    const v = applyMaterial(v0), pr = v.process, go = () => {
      const res = pr === 'md' ? runMD(v) : pr === 'ro_md' ? runROMD(v) : pr === 'hdh' ? runHDH(v) : pr === 'cdi' ? runCDI(v) : pr === 'cdi_ro' ? runCDIRO(v) : pr === 'md_cr' ? runMDC(v) : pr === 'fo_md' || pr === 'ed_fo' ? runFOChain(v) : runFO(v);
      return addExtras(res, v);
    };
    if ((v.pareto || (isMD(v) && v.poreDist === 'network')) && ctx?.tick) return (async () => { ctx.progress?.(0.1, v.pareto ? 'Multi-objective search (NSGA-II)' : 'Solving the pore network'); await ctx.tick(); return go(); })();
    return go();
  },

  mesh: { name: 'Segments along the module', keys: ['nSeg'], min: 2, note: 'Membrane area, flows and inlet conditions are held constant.',
    metrics: [{ label: 'Water flux', unit: 'L/m²·h', get: (r) => r.outputs.flux }, { label: 'Specific thermal energy', unit: 'kWh/m³', get: (r) => r.outputs.secThermal }, { label: 'Product flow', unit: 'm³/h', get: (r) => r.outputs.streams.product.Q }] },

  calibration: {
    note: 'Fit membrane parameters to coupon or module tests of the selected process family. Forward osmosis: rows give the draw concentration, the feed salinity multiplier and the feed velocity; the measurements are the water flux and the reverse solute flux — fit A, B and S and untick the MD parameters. Membrane distillation: rows give hot and cold inlet temperatures and the feed velocity; the measurements are the flux and the thermal efficiency (enter it in the second target column) — fit the tortuosity and the heat-transfer multiplier and untick the FO parameters. Leave the columns of the other process empty.',
    params: [{ key: 'AFO', label: 'FO water permeability A', lo: 0.2, hi: 10 }, { key: 'BFO', label: 'FO salt permeability B', lo: 0.01, hi: 5 }, { key: 'Sfo', label: 'FO structural parameter S (µm)', lo: 50, hi: 3000 }, { key: 'tauM', label: 'MD tortuosity', lo: 1, hi: 5 }, { key: 'fh', label: 'MD heat-transfer multiplier', lo: 0.3, hi: 4 }],
    columns: [{ key: 'cDraw', label: 'Draw concentration', unit: 'mol/L' }, { key: 'salinityFactor', label: 'Feed salinity ×', unit: '–' }, { key: 'uF', label: 'FO feed velocity', unit: 'cm/s' }, { key: 'Tf', label: 'MD hot inlet', unit: '°C' }, { key: 'Tp', label: 'MD cold inlet', unit: '°C' }, { key: 'uFm', label: 'MD feed velocity', unit: 'm/s' }, { key: 'Jw', label: 'Water flux', unit: 'L/m²·h' }, { key: 'aux', label: 'Reverse solute flux (g/m²·h) · MD thermal efficiency (%)', unit: '' }],
    targets: [{ key: 'Jw', label: 'Water flux', unit: 'L/m²·h' }, { key: 'aux', label: 'Reverse solute flux (FO) or thermal efficiency (MD)', unit: 'g/m²·h · %' }],
    model(v) {
      if (isMD(v)) { const S = salinityFromTDS(tds(v.ions) * (v.salinityFactor ?? 1), 25), c = mdConfig(v, v.Tf, v.Tp, S), lo = mdLocal(c, v.Tf, S, { T: v.Tp, pg: 0.3 * psat(v.Tp) }); return { Jw: lo.N * 3600, aux: 100 * lo.eta }; }
      const s = foSetup(v), f = foFlux({ f: 1, cFd: 0, cD: v.cDraw * 1000 }, s.m);
      return { Jw: f.Jw * 3.6e6, aux: f.Js * s.d.M * 3600 };
    },
    get sample() { return (this._s ||= synth(9, [[0.5, 1, 15], [1, 1, 15], [1.5, 1, 15], [2, 1, 15], [1, 0.3, 15], [1, 2, 15], [1, 1, 6], [1, 1, 30], [2, 0.3, 25]])); },
    get validationSample() { return (this._v ||= synth(31, [[0.75, 1, 15], [1.25, 1, 15], [1.75, 0.6, 20], [1, 1.5, 10], [0.6, 0.5, 12], [2, 2, 15]])); },
  },

  async verify() {
    const d = defaultsOf(suite), C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    // FO limits
    const s0 = foSetup({ ...d, Sfo: 0, BFO: 1e-12, uF: 1e5, uD: 1e5, kcp: 1e4 }), f0 = foFlux({ f: 1, cFd: 0, cD: 1000 }, s0.m), ideal = s0.m.A * (drawOsmotic('nacl', 1000) - s0.m.piF(1));
    add('FO flux without polarisation equals A·(π_D − π_F)', ideal * 3.6e6, f0.Jw * 3.6e6, 2e-3 * ideal * 3.6e6, 'S → 0, k → ∞, B → 0 (L/m²·h)');
    const sg = foSetup({ ...d, draw: 'glucose', ions: cloneIons({}) }), mg = { ...sg.m, piD: (c) => 1 * 1.0 * c * R * K(25), Bd: 2e-8 }, fg = foFlux({ f: 0, cFd: 0, cD: 1500 }, mg);
    add('Specific reverse solute flux equals B / (A·ν·R·T)', mg.Bd / (mg.A * R * K(25)), fg.Js / fg.Jw, 1e-6 * (mg.Bd / (mg.A * R * K(25))), 'Pure-water feed, ideal draw (mol/m³): independent of S, k and concentration');
    add('FO: no osmotic difference, no flux', 0, foFlux({ f: 1, cFd: 0, cD: 0 }, foSetup(d).m).Jw, 0, 'Limiting case π_D = 0 < π_F');
    const fo = simulateFO(d);
    add('FO module water balance closes', 0, (fo.QF0 + fo.QD0 - fo.feedOut.Q - fo.drawOut.Q) / fo.QF0, 1e-12, 'Feed + draw in = feed + draw out');
    add('FO module draw-solute balance closes', 0, (fo.QD0 * fo.cD0 - fo.drawOut.n - fo.feedOut.nd) / (fo.QD0 * fo.cD0), 1e-12, 'Draw solute in = diluted draw + reverse leakage into the feed');
    add('FO counter-current iteration meets the draw inlet condition', 0, fo.closure, 1e-8, 'Marched draw inlet flow and solute versus the specified inlet (relative)');
    add('ICP reduces the flux', 1, f0.Jw > foFlux({ f: 1, cFd: 0, cD: 1000 }, foSetup(d).m).Jw ? 1 : 0, 0, 'Flux with S = 400 µm is below the ideal value');
    const sp = foSetup({ ...d, process: 'pro', Sfo: 0, BFO: 1e-12, uF: 1e5, uD: 1e5, kcp: 1e4 }), dpi = drawOsmotic('nacl', 1000) - sp.m.piF(1), Ps = linspace(0.05, 0.95, 181).map((x) => x * dpi), Wd = Ps.map((P) => foFlux({ f: 1, cFd: 0, cD: 1000 }, { ...sp.m, dP: P }).Jw * P);
    add('PRO power density peaks at ΔP = Δπ / 2 (ideal membrane)', 0.5, Ps[Wd.indexOf(Math.max(...Wd))] / dpi, 0.006, 'W = A·ΔP·(Δπ − ΔP)');
    // MD
    add('Liquid-entry pressure, 0.2 µm, 120°, 0.072 N/m', 3.6, liquidEntryPressure(0.2e-6, 120, 0.072, 1) / 1e5, 0.005, 'Laplace–Cantor with geometry factor 1 (bar)');
    add('Mean free path of water vapour at 50 °C, 1 atm', 0.142, meanFreePath(50, 101325) * 1e6, 0.002, 'k_B T / (√2 π σ² P) with σ = 2.641 Å (µm)');
    const mem = { r: 0.1e-6, eps: 0.8, tau: 2, delta: 100e-6 }, dg = dustyGas(mem, 40, 101325, 1e4);
    add('Knudsen-number regimes', 3, (dustyGas({ ...mem, r: 0.01e-6 }, 50, 101325, 1e4).regime === 'Knudsen' ? 1 : 0) + (dg.regime === 'transition' ? 1 : 0) + (dustyGas({ ...mem, r: 20e-6 }, 50, 101325, 1e4).regime === 'molecular' ? 1 : 0), 0, '0.02 µm pores: Knudsen; 0.2 µm: transition; 40 µm: molecular');
    add('Knudsen coefficient, hand value', ((2 * 0.8 * 0.1e-6) / (3 * 2 * 100e-6)) * Math.sqrt((8 * 0.018015) / (Math.PI * 8.314462618 * 313.15)), dg.Bk, 1e-15, '(2εr / 3τδ)·√(8M / πRT), kg/m²·s·Pa');
    const md = { ...d, process: 'md', ions: WATERS.seawater.ions }, c1 = mdConfig(md, 60, 20, 35), l1 = mdLocal(c1, 60, 35, { T: 20 });
    add('DCMD flux at 60 / 20 °C, 0.2 µm PVDF', 30, l1.N * 3600, 10, 'Literature 20–40 L/m²·h for 0.2 µm PVDF/PTFE membranes');
    const c0 = mdConfig({ ...md, Tf: 40, Tp: 40 }, 40, 40, 0), l0 = mdLocal(c0, 40, 0, { T: 40 });
    add('No temperature difference, no flux', 0, l0.N * 3600, 1e-9, 'Pure water on both sides at 40 °C');
    add('Brine at equal temperature draws vapour backwards', 1, mdLocal(mdConfig({ ...md, Tf: 40, Tp: 40 }, 40, 40, 100), 40, 100, { T: 40 }).N <= 0 ? 1 : 0, 0, 'Vapour-pressure lowering of the salt (osmotic distillation limit)');
    add('Local heat balance at the membrane closes', 0, l1.res / l1.q, 1e-7, 'h_f(T_f − T_fm) = N·λ + conduction');
    const mm = mdModule(md, 35);
    add('MD module energy balance closes', 0, (mm.balance.in - mm.balance.out) / mm.balance.in, 1e-9, 'Enthalpy lost by the feed = enthalpy gained by the cold stream');
    add('MD counter-current shooting meets the coolant inlet temperature', 0, mm.matchErr, 1e-7, 'K');
    add('Antoine and IAPWS vapour pressures agree at 60 °C', 0, antoine(60) / psat(60) - 1, 5e-3, 'Relative difference of the two saturation curves offered for MD');
    add('Thermal efficiency lies between 0 and 1 and rises with temperature', 1, mm.eta > 0.3 && mm.eta < 1 && mdLocal(mdConfig({ ...md, Tf: 80 }, 80, 20, 35), 80, 35, { T: 20 }).eta > l1.eta ? 1 : 0, 0, `η = ${fmt(100 * mm.eta, 3)} % along the module`);
    const a = mdModule(md, 35, { nSeg: 24 }), b = mdModule(md, 35, { nSeg: 48 });
    add('MD flux is insensitive to refinement (24 → 48 segments)', 0, Math.abs(a.flux - b.flux) / b.flux, 5e-3, 'Relative change');
    const hd = simulateHDH({ ...d, ...suite.presets[6].values });
    add('HDH cycle energy balance closes', 0, (hd.balance.in - hd.balance.out) / hd.balance.in, 1e-8, 'Heater duty + seawater in = brine + condensate out');
    add('HDH gain-output ratio in the literature range', 1.8, hd.gor, 1.0, 'Water-heated closed-air cycles: about 1–3');
    // ---- pore-scale relations
    add('Kelvin equation, hand value', 1.00439, kelvinFactor(60, 1e-7, 120), 2e-5, 'exp(2γM|cosθ| / ρrRT) with γ = 0.0662 N/m, r = 0.1 µm, θ = 120°, 60 °C');
    const lk = mdLocal({ ...c1, kelvin: true }, 60, 35, { T: 20 });
    add('Kelvin correction raises the vapour pressure and the flux slightly', 1, lk.N > l1.N && lk.N < 1.02 * l1.N && Math.abs(lk.pF / l1.pF - kelvinFactor(60, c1.mem.r, md.theta)) < 2e-3 ? 1 : 0, 0, `Flux × ${fmt(lk.N / l1.N, 5)} for 0.2 µm pores`);
    const mk = { ...mem, sg: 1.4 };
    add('Pore-size distribution: Knudsen coefficient of log-normal pores', Math.exp(2.5 * Math.log(1.4) ** 2), dustyGasPSD(mk, 50, 101325, 1e4, 'knudsen').B / dustyGas(mk, 50, 101325, 1e4, 'knudsen').B, 2e-4, 'Area-weighted integral of B_K ∝ r: ⟨r³⟩/(⟨r²⟩·r_median) = exp(2.5·ln²σ_g)');
    add('Pore-size distribution collapses to the mean pore for uniform pores', 1, dustyGasPSD({ ...mem, sg: 1.0002 }, 40, 101325, 1e4).B / dg.B, 1e-5, 'σ_g → 1');
    add('Pore-size distribution: molecular diffusion does not depend on the pore size', 1, dustyGasPSD(mk, 50, 101325, 1e4, 'molecular').B / dustyGas(mk, 50, 101325, 1e4, 'molecular').B, 1e-9, 'Only the Knudsen part gains from large pores');
    {
      const mN = { ...mem, sg: 1.3, net: { nx: 24, nz: 12, seed: 7, theta: 120, gamma: 0.066, dP: 0, B: 1 } }, nU = dustyGasNetwork({ ...mN, sg: 1 }, 50, 101325, 1e4), nR = dustyGasNetwork(mN, 50, 101325, 1e4), nBig = dustyGasNetwork({ ...mN, net: { ...mN.net, nx: 48, nz: 24 } }, 50, 101325, 1e4);
      add('Pore network of equal throats returns the single-pore (analytic) coefficient', 1, nU.B / nU.Bmean, 1e-10, 'Uniform radius: only the through-plane throats carry flow, K = n_x·g/(n_z + 1)');
      add('Pore network conserves the vapour flow across every cut plane', 0, Math.max(...nR.network.layers.map((q) => Math.abs(q - nR.network.K) / nR.network.K)), 1e-10, `Flow through each of the ${nR.network.layers.length} layers of through-plane throats (log-normal radii, σ_g = 1.3)`);
      add('Pore network: series bottlenecks put the coefficient below the parallel-capillary bundle', 1, nR.B < nR.Bbundle && nR.B < nR.Bmean && nR.B > 0.6 * nR.Bmean ? 1 : 0, 0, `Network ${fmt(nR.B / nR.Bmean, 4)}, bundle ${fmt(nR.Bbundle / nR.Bmean, 4)} × the single mean pore`);
      add('Pore network result is insensitive to the lattice size', nR.B / nR.Bmean, nBig.B / nBig.Bmean, 0.03, '24 × 12 against 48 × 24 lattice');
      const lepM = liquidEntryPressure(mem.r, 120, 0.066), nDry = poreNetwork({ r: mem.r, sg: 1.3, nx: 24, nz: 12, seed: 7, theta: 120, gamma: 0.066, dP: 0.5 * nR.network.net.lepNet }), nWet = poreNetwork({ r: mem.r, sg: 1.3, nx: 24, nz: 12, seed: 7, theta: 120, gamma: 0.066, dP: 1.02 * nR.network.net.lepNet });
      add('Invasion percolation: breakthrough near the entry pressure of the median throat', lepM, nR.network.net.lepNet, 0.15 * lepM, 'Bond-percolation threshold of the square lattice is one half: liquid spans the membrane once the wider half of the throats can be entered (Pa)');
      add('Invasion percolation: no liquid path below the breakthrough pressure, one above it', 1, !nDry.through && nWet.through && nDry.fraction < nWet.fraction ? 1 : 0, 0, `Wetted throats ${fmt(100 * nDry.fraction, 3)} % at half the breakthrough pressure, ${fmt(100 * nWet.fraction, 3)} % just above it`);
    }
    add('Half of the pores lie above the median radius', 0.5, wettedFraction(1e-7, 1.3, 120, 0.066, liquidEntryPressure(1e-7, 120, 0.066)).number, 1e-6, 'Feed pressure equal to the liquid-entry pressure of the median pore');
    // ---- pressure-assisted FO, hydration
    const sP = foSetup({ ...d, Sfo: 0, BFO: 1e-12, uF: 1e5, uD: 1e5, kcp: 1e4, dPfeed: 5 }), fP = foFlux({ f: 1, cFd: 0, cD: 1000 }, sP.m);
    add('Pressure-assisted FO: J_w = A·(Δπ + ΔP) without polarisation', (ideal + sP.m.A * 5e5) * 3.6e6, fP.Jw * 3.6e6, 2e-3 * (ideal + sP.m.A * 5e5) * 3.6e6, '5 bar on the feed side (L/m²·h)');
    add('Half-hydrated support doubles the structural parameter', foFlux({ f: 1, cFd: 0, cD: 1000 }, foSetup({ ...d, Sfo: 800 }).m).Jw * 3.6e6, foFlux({ f: 1, cFd: 0, cD: 1000 }, foSetup({ ...d, Sfo: 400, hydration: 50 }).m).Jw * 3.6e6, 1e-9, 'S_eff = S ÷ hydration (L/m²·h)');
    // ---- capacitive deionisation: porous-electrode model
    const cd = { ...d, ...suite.presets[10].values, lgQm: 0, cdiCycles: 1, cdiRc: 0, cdiHsp: 1, cdiVdis: 0 }, tau = simulatePorousCDI(cd, 20, { cdiV: 0.01, cdiTc: 0.05, cdiTd: 0.05, cdiNt: 20 }).tauRC;
    const lin = simulatePorousCDI(cd, 20, { cdiV: 0.01, cdiTc: (0.5 * tau) / 60, cdiTd: 0.01, cdiNx: 24, cdiNt: 40 }), qInf = lin.av * lin.Le * 0.005 / (1 / cd.cdiCst + lin.lamD / (78.4 * 8.8541878128e-12));
    add('Porous electrode: transmission-line charging matches the analytical series', 1 - (8 / Math.PI ** 2) * (Math.exp(-(Math.PI ** 2) / 8) + Math.exp((-9 * (Math.PI ** 2)) / 8) / 9), lin.chargeStored / qInf, 0.01, 'Q(t)/Q∞ = 1 − Σ 8/((2m+1)²π²)·exp(−(2m+1)²π²t/4τ) at t = τ/2, τ = L²aC/κ (5 mV, constant capacitance, no ion flux at the collector)');
    const eqm = simulatePorousCDI(cd, 20, { cdiV: 1.0, cdiTc: (60 * tau) / 60, cdiTd: 0.01, cdiNt: 60 }), gq = gcs(0.5, eqm.cEnd, { T: cd.T, cSternA: cd.cdiCst });
    add('Porous electrode: fully charged state equals the Gouy–Chapman–Stern equilibrium', gq.sigma, eqm.sigmaEnd[eqm.n - 1], 2e-3 * gq.sigma, 'Surface charge at the collector after 60 time constants versus the double-layer relation of suite 7 (mol/m²)');
    add('Gouy–Chapman diffuse potential + Stern potential = half-cell voltage', 0.5, eqm.dphi(eqm.sigmaEnd[0], eqm.cEnd), 2e-3, 'Δφ_d + σF/C_St at the end of charging (V)');
    add('Porous electrode: charge passed equals charge stored', 0, (eqm.charge - eqm.chargeStored) / eqm.charge, 1e-6, 'Charge balance ∫I dt = F·a·∫σ dx');
    const cdN = simulatePorousCDI({ ...d, ...suite.presets[10].values }, 17);
    add('Capacitive deionisation: salt removed from the flow equals salt stored', 0, (cdN.salt - cdN.storedEDL - cdN.storedLang - cdN.storedMix) / cdN.salt, 2e-3, 'Double-layer excess + Langmuir + spacer and macropore inventory');
    add('Local charge efficiency is tanh(Δφ_d/2) below one', 1, cdN.eff > 0.3 && cdN.eff < 1 ? 1 : 0, 0, `Cycle charge efficiency ${fmt(100 * cdN.eff, 3)} %`);
    const lg = simulatePorousCDI({ ...cd, lgQm: 2, lgKa: 5, lgTheta0: 0 }, 20, { cdiV: 0, cdiTc: 30, cdiTd: 0.01, cdiNt: 40 });
    add('Langmuir adsorption reaches θ = K·c/(1 + K·c)', (cd.lgK * lg.cEnd) / (1 + cd.lgK * lg.cEnd), lg.thetaEnd, 1e-4, 'Fresh electrode at zero voltage after 30 min');
    const cc = simulatePorousCDI({ ...cd, cdiMode: 'cc', cdiI: 8 }, 20, { cdiTc: 1, cdiTd: 1, cdiNt: 30 });
    add('Prescribed-current boundary: the cell current equals the set value', 8, cc.cur[1], 1e-9, 'Constant-current charging below the voltage limit (A/m²)');
    // ---- unified double layer (Gouy–Chapman–Stern ↔ modified Donnan)
    {
      const kL = Math.sqrt((EPS_W * R * K(25)) / (2 * F * F)), hP = 1e-8, o = (model, mu = 0) => ({ kL, h: hP, emu: Math.exp(mu), model });
      const cThin = 100, cOver = 0.5, gT = gcs(0.5, cThin, { T: 25, cSternA: 0.2 }), uT = edlEquilibrium(0.5, cThin, o('auto'), 0.2, 25), mO = mDonnan(0.5, cOver, { T: 25, muAtt: 1.5, cStern: 0.2 / hP / 1e6 }), uO = edlEquilibrium(0.5, cOver, o('auto', 1.5), 0.2, 25);
      add('Unified double layer: thin-layer limit equals Gouy–Chapman–Stern (suite 7)', gT.sigma, uT.sigma, 1e-6 * gT.sigma, `Surface charge at 0.5 V, pore half-width ÷ Debye length = ${fmt(hP / (kL / Math.sqrt(cThin)), 3)} (mol/m²)`);
      add('Unified double layer: overlapping limit equals the modified-Donnan model (suite 7)', mO.sigma * hP, uO.sigma, 1e-6 * mO.sigma * hP, `Charge at 0.5 V with 1.5 kT attraction, pore half-width ÷ Debye length = ${fmt(hP / (kL / Math.sqrt(cOver)), 3)} (mol/m²)`);
      let minCap = Infinity, minGcs = Infinity, maxEff = -Infinity, minEff = Infinity, minSlope = Infinity;
      for (const h of [3e-10, 3e-9, 3e-8, 3e-7]) for (const mu of [0, 2]) for (let a = -3; a <= 3; a += 0.25) for (let b = -9; b <= -3.5; b += 0.5) {
        const c = 10 ** a, sg = 10 ** b, q = o('auto', mu); q.h = h;
        const e0 = edlState(sg, c, q), e1 = edlState(sg, c * 1.001, q), e2 = edlState(sg * 1.001, c, q), qg = { ...q, model: 'gcs' };
        minGcs = Math.min(minGcs, (edlState(sg, c * 1.001, qg).S - edlState(sg, c, qg).S) / (0.001 * c) / (2 * h));
        minCap = Math.min(minCap, (e1.S - e0.S) / (0.001 * c) / (2 * h)); const le = (e2.S - e0.S) / (0.001 * sg); maxEff = Math.max(maxEff, le); minEff = Math.min(minEff, le); minSlope = Math.min(minSlope, e2.pd - e0.pd);
      }
      add('Unified double layer: salt capacity ∂S/∂c is never negative in any regime', 1, minCap >= 0 && minGcs < 0 ? 1 : 0, 0, `Lowest value ${fmt(minCap, 3)} × the pore volume over 0.001–1000 mol/m³, 10⁻⁹–3·10⁻⁴ mol/m² and pore half-widths of 0.3–300 nm; the thin-layer formula alone reaches ${fmt(minGcs, 3)} on the same grid, which is why it had no solution for dilute feeds`);
      add('Unified double layer: differential charge efficiency stays between 0 and 1 (co-ion expulsion)', 1, minEff >= -1e-9 && maxEff < 1 && minSlope > 0 ? 1 : 0, 0, `∂S/∂σ from ${fmt(Math.max(minEff, 0), 3)} to ${fmt(maxEff, 4)} on the same grid; the potential rises monotonically with the charge`);
      const cdU = { ...d, ...suite.presets[10].values, cdiMu: 0, cdiCycles: 1, cdiNt: 40 }, cZone = (3 * kL / (cdU.cdiPor / (cdU.cdiArea * cdU.cdiRho * 1e6))) ** 2;
      const zg = simulatePorousCDI({ ...cdU, cdiEdl: 'gcs' }, cZone), zm = simulatePorousCDI({ ...cdU, cdiEdl: 'md' }, cZone), za = simulatePorousCDI(cdU, cZone);
      add('Gouy–Chapman–Stern and modified-Donnan cycles agree in their overlap region', zg.sac, zm.sac, 0.2 * zg.sac, `Salt adsorption per cycle at pore half-width = 3 Debye lengths (${fmt(cZone * 58.44, 3)} mg/L), no attraction term: ${fmt(zg.sac, 4)} and ${fmt(zm.sac, 4)} mg/g; unified model ${fmt(za.sac, 4)} mg/g`);
      add('Unified model lies between the two limiting models in the overlap region', 1, za.sac >= Math.min(zg.sac, zm.sac) - 0.02 * zg.sac && za.sac <= Math.max(zg.sac, zm.sac) + 0.02 * zg.sac && !zg.fallback ? 1 : 0, 0, `Charge efficiency ${fmt(100 * zg.eff, 3)} % (thin), ${fmt(100 * zm.eff, 3)} % (Donnan), ${fmt(100 * za.eff, 3)} % (unified)`);
      let wS = 0, wQ = 0, effOk = true, nSw = 0;
      for (const mgL of [5, 20, 100, 300, 1000, 3500, 12000, 35000]) { const q = simulatePorousCDI({ ...cdU, cdiMu: 1.5 }, mgL / 58.44); nSw++; wS = Math.max(wS, Math.abs(q.salt - q.storedEDL - q.storedLang - q.storedMix) / Math.max(Math.abs(q.salt), 1e-12)); wQ = Math.max(wQ, Math.abs(q.charge - q.chargeStored) / Math.abs(q.charge)); effOk = effOk && q.eff > 0 && q.eff < 1 && q.failedSteps === 0 && Number.isFinite(q.sec); }
      add('Capacitive deionisation: salt balance closes from 5 mg/L to seawater', 0, wS, 1e-7, `Largest relative imbalance over ${nSw} feeds (5 … 35 000 mg/L) — implicit conservative integration`);
      add('Capacitive deionisation: charge balance closes from 5 mg/L to seawater', 0, wQ, 1e-8, 'Charge passed versus charge stored, same sweep');
      add('Capacitive deionisation: charge efficiency below one over the whole salinity sweep', 1, effOk ? 1 : 0, 0, 'Co-ion expulsion in both double-layer regimes; every implicit step converged');
      let nBad = 0, nRun = 0; const lohi = (k) => { const f = suite.inputs.flatMap((g) => g.fields).find((q) => q.key === k); return [f.min, f.max]; };
      const corner = ['cdiArea', 'cdiV', 'cdiLe', 'cdiQ', 'cdiTc', 'cdiCst', 'cdiPor', 'cdiHsp', 'cdiI'];
      for (let m = 0; m < 12; m++) { // corners of the allowed input ranges (Gray-code-like pattern) at 5 mg/L and at seawater strength
        const ov = { cdiCycles: 1, cdiNt: 24, cdiNx: 5, cdiMode: m % 3 === 0 ? 'cc' : 'cv' }; corner.forEach((k, q) => { ov[k] = lohi(k)[((m >> (q % 4)) + q + (m > 5 ? 1 : 0)) % 2]; });
        try { const q = simulatePorousCDI({ ...cdU, cdiMu: 1.5, ...ov }, m % 2 ? 600 : 0.0856); nRun++; if (![q.removal, q.sec, q.eff, q.sac, q.cEnd].every(Number.isFinite) || Math.abs(q.charge - q.chargeStored) > 1e-6 * Math.max(Math.abs(q.charge), 1e-9)) nBad++; } catch { nBad++; }
      }
      add('Capacitive deionisation: no input corner of the allowed ranges fails', 0, nBad, 0, `${nRun} extreme combinations of area, voltage, thickness, flow, time, capacitance, porosity and spacer at 5 mg/L and 35 g/L, constant voltage and constant current`);
    }
    const ep = edlProfile(0.1, 20, 25);
    add('Poisson–Nernst–Planck double layer reproduces the Gouy–Chapman potential', 0, Math.max(...ep.psi.map((q, k) => Math.abs(q - ep.gc[k]))), 2e-4, 'Insulating wall at 100 mV in 20 mol/m³ (largest deviation, V)');
    add('Poisson–Nernst–Planck surface charge equals the Grahame equation', ep.sigmaGC, ep.sigmaLeft, 0.01 * ep.sigmaGC, 'C/m²');
    // ---- crystalliser, hybrid chains
    const cr = msmpr({ tau: 7200, kg: 3e-6, kb: 7e8, MT: 150 }), Lq = linspace(0, 40 * cr.L0, 4001);
    add('MSMPR supersaturation, hand value', (1 / (6 * 0.5 * 2165 * 7200 ** 4 * 3e-6 ** 3 * 7e8)) ** 0.2, cr.sigma, 1e-12, 'σ = [6·k_v·ρ_c·τ⁴·k_g³·k_b]^(−1/5)');
    add('MSMPR population balance returns the suspension density', 150, sum(Lq.slice(1).map((L, k) => 0.5 * (cr.massDensity(L) + cr.massDensity(Lq[k])) * (L - Lq[k]))), 0.05, '∫ k_v·ρ_c·L³·n(L) dL (kg/m³)');
    const mc = simulateMDC({ ...d, ...suite.presets[9].values });
    add('MD–crystalliser salt balance closes', 0, ((mc.make * mc.Sfeed) / 1000 - mc.solids - (mc.bleed * mc.Sloop) / 1000) / ((mc.make * mc.Sfeed) / 1000), 1e-12, 'Feed salt = crystals + bleed');
    add('MD–crystalliser loop is supersaturated by the MSMPR value', mc.cr.sigma, mc.Sloop / saltSolubility(mc.p.Tcr) - 1, 1e-12, 'Loop salinity = solubility × (1 + σ)');
    const nc = simulateMDC({ ...d, ...suite.presets[9].values, bleed: 40 });
    add('MD–crystalliser: a large bleed keeps the loop unsaturated and gives no crystals', 0, nc.solids + Math.max(0, nc.Sloop - saltSolubility(nc.p.Tcr)), 0, `Loop at ${fmt(nc.Sloop, 4)} g/kg with a 40 % bleed`);
    const chain = await Promise.all([7, 8, 11].map((k) => suite.run({ ...d, ...suite.presets[k].values }))), worst = Math.max(...chain.flatMap((q) => q.balances.map((b) => Math.abs(b.in - b.out) / Math.max(Math.abs(b.in), 1e-30))));
    add('Hybrid chains (FO–MD, electrodialysis–FO, RO–CDI) close all their balances', 0, worst, 2e-3, 'Largest relative imbalance over water, solute, energy, salt and charge balances of the three chains');
    add('Electrodialysis reaches the product target on the diluted draw', d.edTarget, chain[1].kpis.find((q) => q.label === 'Product TDS').value, 1, 'mg/L');
    // ---- dynamics, Pareto, grey box
    const rm = simulateMD(md), dq = dynamicMD({ ...md, tDyn: 2, batchVol: 1e6, cFou: 10, kRem: 0, kScaleMD: 0, thetaDrop: 0, wet0: 0, tauWet: 6, ntDyn: 40 }, rm), eR = dq.rows[dq.rows.length - 1];
    add('Batch MD: deposit grows as c_fou·J·t without removal', (10 * eR.N * 3600 * 2) / density(25, 0), eR.md, 0.01 * eR.md, 'Large tank, no scaling or wetting (g/m²)');
    add('Batch MD: tank water and salt are conserved', 0, Math.abs(dq.waterBal.in - dq.waterBal.out) / dq.waterBal.in + Math.abs(dq.saltBal.in - dq.saltBal.out) / dq.saltBal.in, 1e-9, 'Tank = distillate + leak + remaining tank');
    const dw = dynamicMD({ ...md, tDyn: 48, batchVol: 50, theta: 95, pFeed: 2, cFou: 50, thetaDrop: 20, kRem: 0, kScaleMD: 0, wet0: 0, tauWet: 6, ntDyn: 100 }, rm);
    add('Batch MD: a fouled, barely hydrophobic membrane wets progressively', 1, dw.rows[dw.rows.length - 1].xw > 0.05 && dw.rows[5].xw < dw.rows[dw.rows.length - 1].xw && dw.tdsMix > 0 ? 1 : 0, 0, `Wetted pore area ${fmt(100 * dw.rows[dw.rows.length - 1].xw, 3)} % after 48 h; distillate ${fmt(dw.tdsMix, 3)} mg/kg`);
    const df = dynamicFO({ ...d, dynamic: true, tDyn: 48, batchVol: 10, drawVol: 5, cFou: 0, kRem: 0.02, alphaCake: 5, tauHyd: 2, ntDyn: 400 });
    add('Batch FO: water and draw solute are conserved', 0, Math.abs(df.waterBal.in - df.waterBal.out) / df.waterBal.in + Math.abs(df.soluteBal.in - df.soluteBal.out) / df.soluteBal.in, 1e-9, 'Feed tank + draw tank');
    add('Batch FO runs down to osmotic equilibrium', 0, df.fluxEnd / df.fluxMax, 0.02, 'Flux at the end ÷ largest flux of the batch');
    { const dm = { ...d, process: 'md' }, pd = paretoDesigns(dm, 35, true, { pop: 12, gens: 3 }), fr = paretoFront(pd.all, (q) => q.flux, (q) => q.en).front;
      add('NSGA-II front is never worse than the grid front (hypervolume)', 1, pd.hvAll >= pd.hvGrid - 1e-12 && pd.ga.length > 0 ? 1 : 0, 0, `Dominated area ${fmt(pd.hvGrid, 5)} (grid of ${pd.grid.length}) → ${fmt(pd.hvAll, 5)} with ${pd.ga.length} NSGA-II designs`);
      add('Pareto front holds no dominated design', 0, fr.filter((a) => pd.all.some((b) => b.flux >= a.flux && b.en <= a.en && (b.flux > a.flux || b.en < a.en))).length, 0, `${fr.length} non-dominated designs of ${pd.all.length}`);
      const pd2 = paretoDesigns(dm, 35, true, { pop: 12, gens: 3 });
      add('NSGA-II search is deterministic for a given seed', pd.hvAll, pd2.hvAll, 1e-12, 'Two runs, same seed'); }
    const pp = [[1, 5], [2, 4], [3, 6], [4, 8], [2.5, 7], [5, 20], [4.5, 9]].map(([f, e]) => ({ f, e })), pf = paretoFront(pp, (q) => q.f, (q) => q.e);
    add('Pareto front keeps exactly the non-dominated designs', 5, pf.front.length + (pf.front.some((a) => pp.some((b) => b.f >= a.f && b.e <= a.e && (b.f > a.f || b.e < a.e))) ? 100 : 0), 0, 'Seven test points (flux, energy): (2,4), (3,6), (4,8), (4.5,9) and (5,20) survive; (1,5) and (2.5,7) are dominated');
    const gbT = greyBox(GB_FO.map((q) => ({ ...q, Jw: 1.2 * (q.cDraw * 10 + q.uF) })), (q) => q.cDraw * 10 + q.uF, (q) => [Math.log(q.cDraw), Math.log(q.uF / 15)], 1e-9);
    add('Grey-box correction recovers a uniform 20 % bias', Math.log(1.2), gbT.beta[0], 1e-6, 'Data = 1.2 × model: β₀ = ln 1.2, β₁ = β₂ = 0, leave-one-out error ≈ 0');
    add('Grey-box correction leaves a perfect model unchanged', 1, greyBox(GB_FO.map((q) => ({ ...q, Jw: q.cDraw * 10 + q.uF })), (q) => q.cDraw * 10 + q.uF, (q) => [Math.log(q.cDraw), Math.log(q.uF / 15)]).factor({ cDraw: 1.3, uF: 12 }), 1e-9, 'Correction factor at an arbitrary point');
    return C;
  },
};

// ---- result builders ----------------------------------------------------------------------------------------
function runFO(v) {
  const r = simulateFO(v), p = v, W = [], d = r.d, pro = r.pro, hybrid = p.process === 'fo_ro', T = r.T, N = r.N;
  const coupon = foFlux({ f: 1, cFd: 0, cD: r.cD0 }, r.m), ideal = r.m.A * (coupon.piDb - coupon.piFb - r.m.dP);
  if (r.tot.Vw <= 1e-12) W.push({ level: 'bad', msg: `No water flux: the draw osmotic pressure (${fmt(coupon.piDb / 1e5, 3)} bar) does not exceed the feed osmotic pressure (${fmt(coupon.piFb / 1e5, 3)} bar)${pro ? ' plus the applied pressure' : ''}. Raise the draw concentration.` });
  if (p.cDraw > d.sol) W.push({ level: 'bad', msg: `${p.cDraw} mol/L exceeds the solubility of ${d.name} (about ${d.sol} mol/L).` });
  if (r.segs.some((g) => g.Jw <= 1e-9 * (coupon.Jw + 1e-30)) && r.tot.Vw > 0) W.push({ level: 'warn', msg: 'Osmotic equilibrium is reached inside the module: part of the membrane area is idle. Reduce the area or raise the draw flow.' });
  if (r.recovery > 0.85) W.push({ level: 'warn', msg: `Feed recovery of ${fmt(100 * r.recovery, 3)} % concentrates the feed ${fmt(r.fOut, 3)}-fold — check scaling of the feed concentrate in suite 2.` });
  if (r.srsf > 1 && !pro) W.push({ level: 'warn', msg: `Specific reverse solute flux is ${fmt(r.srsf, 3)} g per litre of water — draw-solute make-up and feed contamination will be significant.` });
  if (!r.conv) W.push({ level: 'warn', msg: 'The counter-current iteration did not fully converge.' });
  const reg = pro ? null : hybrid ? null : regeneration(r, p);
  if (reg && !reg.feasible) W.push({ level: 'bad', msg: p.regen === 'ro' ? `Regenerating this draw by RO needs about ${fmt(reg.P, 3)} bar, above the 83 bar rating of seawater elements — use a weaker draw, a thermal method or high-pressure / osmotically assisted RO.` : `Thermolytic stripping only applies to ammonium bicarbonate draws, not to ${d.name}.` });
  // hybrid: RO regeneration with the element-by-element model of suite 1
  let ro = null, roErr = null;
  if (hybrid) {
    {
      const da = drawAsIons(p.draw, r.cDout, T), ionsD = da.ions, rec = clamp(r.tot.Vw / r.drawOut.Q, 0.05, 0.85);
      try { ro = simulateRO({ ...defaultsOf(roSuite), ions: ionsD, Qf: r.drawOut.Q * 3600, T, pH: 7, recovery: 100 * rec, targetFlux: p.roFlux, design: 'auto', mode: 'recovery', nSeg: 2 }); } catch (e) { roErr = e.message; }
      if (da.equivalent && !roErr) W.push({ level: 'info', msg: `${d.name} carries no ions: the regeneration RO is solved for the osmotically equivalent NaCl solution (${fmt((da.cEq * 58.44) / 1000, 3)} g/L, same osmotic pressure). Pressure, area and energy carry over; the permeate salinity shown is that equivalent and is an upper bound, because RO rejects the larger sugar molecule better than NaCl.` });
    }
    if (roErr) W.push({ level: 'bad', msg: `RO regeneration of the diluted draw could not be solved: ${roErr}` });
    else if (ro.p1.Pf > ro.cfg.M.pmax) W.push({ level: 'bad', msg: `The regeneration RO needs ${fmt(ro.p1.Pf, 3)} bar, above the ${ro.cfg.M.pmax} bar element rating — lower the draw concentration.` });
  }
  if (!W.some((w) => w.level !== 'info')) W.unshift({ level: 'info', msg: pro ? 'PRO module solved; water and solute balances close.' : 'FO module solved; water and solute balances close.' });

  const Vw = r.tot.Vw * 3600, eFO = r.Ppump / 1000 / Math.max(Vw, 1e-12), turb = pro ? (r.tot.Vw * r.m.dP * (p.etaTurb / 100)) / 1000 : 0;
  const secE = pro ? 0 : hybrid ? eFO + (ro ? (ro.power / Math.max(ro.product.Q, 1e-9)) : 0) : eFO + reg.elec, secT = reg ? reg.heat : 0;
  const prodQ = hybrid && ro ? ro.product.Q : Vw, prodIons = hybrid && ro ? ro.product.ions : cloneIons(d.ions ? Object.fromEntries(Object.entries(d.ions).map(([k, m]) => [k, m * (p.regen === 'none' && !hybrid ? r.cDout : 0)])) : {});
  const feedOutIons = scaleIons(r.ions, r.fOut); if (d.ions) for (const [k, m] of Object.entries(d.ions)) feedOutIons[k] = (feedOutIons[k] || 0) + m * (r.feedOut.nd / r.feedOut.Q);
  const xs = r.segs.map((g) => g.x), loss = { 'Bulk osmotic difference': (coupon.piDb - coupon.piFb) / 1e5, [r.alds ? 'Dilutive ECP (draw side)' : 'Dilutive ICP (support, draw side)']: (coupon.piDb - coupon.piDm) / 1e5, [r.alds ? 'Concentrative ICP (support, feed side)' : 'Concentrative ECP (feed side)']: (coupon.piFm - r.m.piF(1)) / 1e5, 'Reverse solute at the feed face': coupon.piR / 1e5, ...(pro ? { 'Applied hydraulic pressure': r.m.dP / 1e5 } : {}), 'Effective driving pressure': coupon.Jw / r.m.A / 1e5 };
  // sweeps on the inlet coupon
  const cds = linspace(0.2, Math.min(d.sol, Math.max(3, p.cDraw * 1.5)), 12), Ss = [0, 100, 200, 400, 800, 1500], flux = (ov, cD = r.cD0, dP) => { const s = foSetup(p, ov), m = dP === undefined ? s.m : { ...s.m, dP }; return foFlux({ f: 1, cFd: 0, cD }, m); };
  const sweepC = (o) => cds.map((c) => tryOr(() => flux({ ...o, cDraw: c }, c * 1000).Jw * 3.6e6)), us = [3, 6, 10, 15, 25, 40];
  const fS = [100, 250, 400, 600, 900, 1300], fC = linspace(0.4, Math.min(d.sol, Math.max(2.5, p.cDraw * 1.3)), 6);
  const plots = [
    { type: 'line', title: 'Water flux and reverse solute flux along the module', xlabel: 'Position along the feed path (fraction)', ylabel: 'L/m²·h · g/m²·h', series: [{ name: 'Water flux (L/m²·h)', x: xs, y: r.segs.map((g) => g.Jw * 3.6e6) }, { name: 'Reverse solute flux (g/m²·h)', x: xs, y: r.segs.map((g) => g.Js * d.M * 3600) }], zeroY: true },
    { type: 'line', title: 'Osmotic pressures along the module', xlabel: 'Position along the feed path (fraction)', ylabel: 'bar', series: [{ name: 'Draw, bulk', x: xs, y: r.segs.map((g) => g.piDb / 1e5) }, { name: 'Draw, at the active layer', x: xs, y: r.segs.map((g) => g.piDm / 1e5), dash: true }, { name: 'Feed, bulk', x: xs, y: r.segs.map((g) => g.piFb / 1e5) }, { name: 'Feed, at the active layer', x: xs, y: r.segs.map((g) => (g.piFm + g.piR) / 1e5), dash: true }], note: 'The gap between the dashed curves is the effective driving force; the rest is lost to concentration polarisation.' },
    { type: 'line', title: 'Flows and concentrations along the module', xlabel: 'Position along the feed path (fraction)', ylabel: 'see legend', series: [{ name: 'Feed flow (m³/h)', x: xs, y: r.segs.map((g) => g.QF * 3600) }, { name: 'Draw flow (m³/h)', x: xs, y: r.segs.map((g) => g.QD * 3600) }, { name: 'Draw concentration × 10 (mol/L)', x: xs, y: r.segs.map((g) => g.cD / 100) }, { name: 'Feed concentration factor × 10', x: xs, y: r.segs.map((g) => g.f * 10) }] },
    { type: 'bar', title: 'Where the osmotic driving force goes (module inlet)', ylabel: 'bar', categories: Object.keys(loss), series: [{ name: 'Pressure', values: Object.values(loss) }] },
    { type: 'line', title: 'Flux versus draw concentration and orientation', xlabel: 'Draw concentration (mol/L)', ylabel: 'Water flux (L/m²·h)', series: [{ name: 'Active layer facing feed', x: cds, y: sweepC({ orient: 'alfs', process: 'fo' }), mode: 'both' }, { name: 'Active layer facing draw', x: cds, y: sweepC({ orient: 'alds', process: 'fo' }), mode: 'both' }, { name: 'No polarisation: A·Δπ', x: cds, y: cds.map((c) => Math.max(0, r.m.A * (drawOsmotic(p.draw, c * 1000, T) - r.m.piF(1))) * 3.6e6), dash: true }], vlines: [{ x: p.cDraw, label: 'operating' }], note: 'Inlet conditions, no applied pressure. The flux flattens with concentration because internal polarisation grows exponentially with flux.' },
    { type: 'line', title: 'Effect of the structural parameter and cross-flow velocity', xlabel: 'Structural parameter S (µm) · velocity × 30 (cm/s)', ylabel: 'Water flux (L/m²·h)', series: [{ name: 'Flux versus S', x: Ss, y: Ss.map((S) => tryOr(() => flux({ Sfo: S }).Jw * 3.6e6)), mode: 'both' }, { name: 'Flux versus velocity (x = 30 × cm/s)', x: us.map((u) => u * 30), y: us.map((u) => tryOr(() => flux({ uF: u, uD: u }).Jw * 3.6e6)), mode: 'both' }], vlines: [{ x: p.Sfo, label: 'S' }] },
    { type: 'field', title: 'Flux versus structural parameter and draw concentration', xlabel: 'Structural parameter S (µm)', ylabel: 'Draw concentration (mol/L)', zlabel: 'Water flux', zunit: 'L/m²·h', x: fS, y: fC, z: fC.map((c) => fS.map((S) => tryOr(() => flux({ Sfo: S, cDraw: c }, c * 1000).Jw * 3.6e6, 0))), cmap: 'viridis', contours: 8, markers: [{ x: clamp(p.Sfo, 100, 1300), y: clamp(p.cDraw, fC[0], fC[5]), label: 'operating' }] },
  ];
  if (pro) {
    const dpi = (coupon.piDb - coupon.piFb) / 1e5, Pb = linspace(0, Math.max(dpi, 1), 25), wd = Pb.map((P) => tryOr(() => flux({}, r.cD0, P * 1e5).Jw * P * 1e5, 0));
    plots.push({ type: 'line', title: 'PRO power density versus applied pressure (module inlet)', xlabel: 'Hydraulic pressure difference (bar)', ylabel: 'W/m² · L/m²·h', series: [{ name: 'Power density (W/m²)', x: Pb, y: wd }, { name: 'Water flux (L/m²·h)', x: Pb, y: Pb.map((P) => tryOr(() => flux({}, r.cD0, P * 1e5).Jw * 3.6e6, 0)) }, { name: 'Ideal membrane: A·ΔP·(Δπ − ΔP) (W/m²)', x: Pb, y: Pb.map((P) => r.m.A * P * 1e5 * Math.max(0, dpi - P) * 1e5), dash: true }], vlines: [{ x: p.dPpro, label: 'operating' }, { x: dpi / 2, label: 'Δπ/2' }] });
    if (r.powerDensity < 5) W.push({ level: 'info', msg: `Average power density is ${fmt(r.powerDensity, 3)} W/m²; about 5 W/m² is the usual threshold for economic PRO.` });
  }
  const kpis = [
    { label: 'Average water flux', value: r.JwLMH, unit: 'L/m²·h' }, { label: 'Water transferred', value: Vw, unit: 'm³/h' }, { label: 'Feed recovery', value: 100 * r.recovery, unit: '%', status: r.recovery > 0.85 ? 'warn' : 'ok' },
    { label: 'Draw dilution factor', value: r.dilution, unit: '×' }, { label: 'Draw outlet concentration', value: r.cDout / 1000, unit: 'mol/L' }, { label: 'Reverse solute flux', value: r.JsGMH, unit: 'g/m²·h' },
    { label: 'Specific reverse solute flux', value: r.srsf, unit: 'g/L', status: r.srsf > 1 ? 'warn' : 'ok', help: 'Draw solute lost per litre of water; equals B/(A·ν·R·T) for an ideal draw' }, { label: 'Feed concentration factor', value: r.fOut, unit: '×' },
    { label: 'Flux ÷ ideal flux (inlet)', value: ideal > 0 ? (100 * coupon.Jw) / ideal : 0, unit: '%', help: 'Share of A·Δπ that survives concentration polarisation' }, { label: 'Feed osmotic pressure', value: coupon.piFb / 1e5, unit: 'bar' }, { label: 'Draw osmotic pressure', value: coupon.piDb / 1e5, unit: 'bar' },
    { label: 'Membrane area', value: r.area, unit: 'm²' }, { label: 'Draw-solute loss', value: r.tot.Sd * d.M * 3.6, unit: 'kg/h' },
    ...(pro ? [{ label: 'Power density', value: r.powerDensity, unit: 'W/m²', status: r.powerDensity < 5 ? 'warn' : 'ok' }, { label: 'Gross turbine power', value: turb, unit: 'kW' }, { label: 'Net power after pumping', value: turb - r.Ppump / 1000, unit: 'kW' }, { label: 'Energy per m³ of permeate', value: (turb - r.Ppump / 1000) / Math.max(Vw, 1e-12), unit: 'kWh/m³' }]
      : hybrid ? [{ label: 'Product flow', value: prodQ, unit: 'm³/h' }, { label: 'Product TDS', value: ro ? tds(ro.product.ions) : 0, unit: 'mg/L' }, { label: 'RO pressure', value: ro ? ro.p1.Pf : 0, unit: 'bar', status: ro && ro.p1.Pf > ro.cfg.M.pmax ? 'bad' : 'ok' }, { label: 'Combined specific energy', value: secE, unit: 'kWh/m³' }, { label: 'Overall recovery', value: 100 * (prodQ / p.Qf), unit: '%' }]
        : [{ label: 'Regeneration electricity', value: reg.elec, unit: 'kWh/m³' }, { label: 'Regeneration heat', value: reg.heat, unit: 'kWh/m³' }, { label: 'FO pumping energy', value: eFO, unit: 'kWh/m³' }]),
  ];
  const tables = [
    { title: 'Streams', columns: ['Stream', 'Flow (m³/h)', 'Osmotic pressure (bar)', 'Concentration'], rows: [['Feed in', p.Qf, r.m.piF(1) / 1e5, `${fmt(r.tdsF, 4)} mg/L`], ['Feed out (concentrate)', r.feedOut.Q * 3600, (r.m.piF(r.fOut) + r.m.piD(r.feedOut.nd / r.feedOut.Q)) / 1e5, `${fmt(r.tdsF * r.fOut, 4)} mg/L + ${fmt((r.feedOut.nd / r.feedOut.Q) * d.M, 3)} mg/L draw solute`], ['Draw in', p.Qd, drawOsmotic(p.draw, r.cD0, T) / 1e5, `${fmt(p.cDraw, 4)} mol/L`], ['Draw out (diluted)', r.drawOut.Q * 3600, drawOsmotic(p.draw, r.cDout, T) / 1e5, `${fmt(r.cDout / 1000, 4)} mol/L`]] },
    { title: 'Profile along the module', columns: ['Position', 'Water flux (L/m²·h)', 'Reverse solute flux (g/m²·h)', 'Feed factor', 'Draw (mol/L)', 'Draw at active layer (mol/L)', 'Feed-side polarisation factor', 'π draw bulk (bar)', 'π feed bulk (bar)', 'Effective driving pressure (bar)'],
      rows: r.segs.filter((_, k) => N <= 40 || k % Math.ceil(N / 40) === 0).map((g) => [g.x, g.Jw * 3.6e6, g.Js * d.M * 3600, g.f, g.cD / 1000, g.cDm / 1000, g.f > 0 ? g.fm / g.f : 1, g.piDb / 1e5, g.piFb / 1e5, g.Jw / r.m.A / 1e5 + (pro ? r.m.dP / 1e5 : 0)]) },
    { title: 'Transport parameters', columns: ['Quantity', 'Value', 'Unit'], rows: [['Feed-side mass-transfer coefficient', r.mF.k * 1e6, 'µm/s'], ['Draw-side mass-transfer coefficient', r.mD.k * 1e6, 'µm/s'], ['Draw-solute diffusivity', r.Dd * 1e9, '10⁻⁹ m²/s'], ['Solute resistivity of the support K = S/D', (p.Sfo * 1e-6) / r.Dd / 86400, 'd/m'], ['Feed Reynolds number', r.mF.Re, '–'], ['Draw-solute permeability B', r.m.Bd * 3.6e6, 'L/m²·h'], ['Theoretical B/(A·ν·φ·R·T)', (r.m.Bd / (r.m.A * d.nu * d.phi[0] * R * K(T))) * d.M / 1000, 'g/L'], ['Feed-channel pressure drop', r.dpF / 1e5, 'bar'], ['Draw-channel pressure drop', r.dpD / 1e5, 'bar']] },
  ];
  if (reg) tables.push({ title: 'Draw regeneration', columns: ['Quantity', 'Value'], rows: [['Method', reg.type], ['Water to be removed from the diluted draw (%)', 100 * reg.rec], ['Electricity (kWh per m³ of product)', reg.elec], ['Heat (kWh per m³ of product)', reg.heat], ['Draw-solute make-up (kg per m³ of product)', r.srsf]], note: reg.note });
  if (hybrid && ro) tables.push({ title: 'Contribution of each process', columns: ['Process', 'Water handled (m³/h)', 'Pressure (bar)', 'Electricity (kW)', 'Specific energy (kWh per m³ product)', 'Role'],
    rows: [['Forward osmosis', Vw, 0, r.Ppump / 1000, r.Ppump / 1000 / prodQ, `Extracts ${fmt(100 * r.recovery, 3)} % of the feed into the draw`], ['Reverse osmosis', ro.product.Q, ro.p1.Pf, ro.power, ro.power / prodQ, `Re-concentrates the draw at ${fmt(100 * ro.overallRec, 3)} % recovery (${ro.nEl} elements)`], ['Hybrid total', prodQ, null, r.Ppump / 1000 + ro.power, secE, `Overall recovery ${fmt((100 * prodQ) / p.Qf, 3)} %; brine reduced to ${fmt(r.feedOut.Q * 3600, 3)} m³/h`]],
    note: 'The RO permeate is the product; the RO concentrate returns to the FO as regenerated draw. Draw solute passing the RO membrane leaves with the product.' });
  return {
    summary: pro ? `PRO transfers ${fmt(Vw, 4)} m³/h at ${fmt(r.JwLMH, 3)} L/m²·h against ${p.dPpro} bar, giving ${fmt(r.powerDensity, 3)} W/m² and ${fmt(turb - r.Ppump / 1000, 3)} kW net.`
      : `${fmt(Vw, 4)} m³/h of water is drawn through ${fmt(r.area, 4)} m² at ${fmt(r.JwLMH, 3)} L/m²·h (${fmt(100 * r.recovery, 3)} % of the feed), diluting the ${d.name.split(' ')[0].toLowerCase()} draw ${fmt(r.dilution, 3)}-fold with ${fmt(r.srsf, 3)} g/L reverse solute loss${hybrid && ro ? `; RO regeneration at ${fmt(ro.p1.Pf, 3)} bar brings the combined energy to ${fmt(secE, 3)} kWh/m³.` : reg ? `; regeneration needs ${fmt(reg.elec, 3)} kWh/m³ of electricity${reg.heat > 0 ? ` and ${fmt(reg.heat, 3)} kWh/m³ of heat` : ''}.` : '.'}`,
    warnings: W, kpis,
    recommendations: [
      coupon.Jw < 0.35 * ideal && !r.alds ? 'Internal concentration polarisation consumes most of the driving force: a thinner, more open support (lower S) helps more than a stronger draw.' : null,
      r.srsf > 1 ? 'Choose a draw solute with larger or multivalent ions (MgCl₂, Na₂SO₄) or a membrane with lower B to cut the reverse solute flux.' : null,
      reg && !reg.feasible && p.regen === 'ro' ? 'Dilute the draw less concentrated or regenerate thermally (MD in this suite, MED in suite 6).' : null,
      pro && Math.abs(p.dPpro - (coupon.piDb - coupon.piFb) / 2e5) > 0.2 * ((coupon.piDb - coupon.piFb) / 2e5) ? `Operate near ΔP ≈ Δπ/2 = ${fmt((coupon.piDb - coupon.piFb) / 2e5, 3)} bar for maximum power density.` : null,
      'Send the feed concentrate to suite 2 (Brine chemistry) for scaling, and compare the cost of water with direct RO in suite 13.',
    ].filter(Boolean),
    plots: clean(plots), tables,
    balances: [{ name: 'Water (m³/h)', in: (r.QF0 + r.QD0) * 3600, out: (r.feedOut.Q + r.drawOut.Q) * 3600 }, { name: 'Draw solute (mol/s)', in: r.QD0 * r.cD0, out: r.drawOut.n + r.feedOut.nd }, { name: 'Feed solutes (factor·m³/h)', in: r.QF0 * 3600, out: (r.feedOut.nf + r.drawOut.nf) * 3600 }],
    outputs: { flux: r.JwLMH, secThermal: secT, secElec: pro ? -(turb - r.Ppump / 1000) / Math.max(Vw, 1e-12) : secE, area: r.area + (ro ? ro.area : 0), recovery: hybrid ? prodQ / p.Qf : r.recovery, reverseSoluteFlux: r.JsGMH, powerDensity: r.powerDensity, process: p.process,
      streams: { product: stream(prodQ, T, 7, prodIons), concentrate: stream(r.feedOut.Q * 3600, T, p.pH, feedOutIons) } },
  };
}

function mdWarnings(r, p, W) {
  const m = r.m, c = m.c, dg = m.segs[0].dg, margin = r.lep / 1e5 - p.pFeed;
  if (margin < 0) W.push({ level: 'bad', msg: `Wetting: the feed pressure (${p.pFeed} bar) exceeds the liquid-entry pressure of ${fmt(r.lep / 1e5, 3)} bar — brine will penetrate the pores.` });
  else if (margin < 1) W.push({ level: 'warn', msg: `Wetting margin is only ${fmt(margin, 2)} bar (liquid-entry pressure ${fmt(r.lep / 1e5, 3)} bar) — smaller pores, a more hydrophobic surface or a lower feed pressure are advisable.` });
  if (r.sat.nacl >= 1) W.push({ level: 'bad', msg: `The membrane-surface salinity (${fmt(m.SmMax, 3)} g/kg) reaches NaCl saturation: crystals will form on the membrane (MD-crystalliser regime) and promote wetting.` });
  else if (r.sat.gypsum > 1) W.push({ level: 'warn', msg: `Calcium sulphate is supersaturated at the membrane surface (ratio about ${fmt(r.sat.gypsum, 3)}) — dose antiscalant or limit the loop recovery.` });
  if (m.eta < 0.4 && c.type !== 'vmd') W.push({ level: 'warn', msg: `Thermal efficiency is ${fmt(100 * m.eta, 3)} %: most of the heat crosses the membrane by conduction. Use a thicker or more porous membrane or a higher feed temperature.` });
  if (c.type === 'vmd' && tsat(c.Pv) < p.Tp + 3) W.push({ level: 'warn', msg: `Vapour at ${p.Pvac} kPa condenses at ${fmt(tsat(c.Pv), 3)} °C, which the ${p.Tp} °C coolant cannot provide — raise the permeate pressure or chill the condenser.` });
  if (c.type === 'vmd' && psatSeawater(m.TfOut, m.Sout) < c.Pv) W.push({ level: 'warn', msg: 'Towards the module outlet the feed vapour pressure falls below the vacuum pressure: that part of the membrane is idle.' });
  if (m.fluxLMH <= 0) W.push({ level: 'bad', msg: 'No positive flux: the temperature difference is too small for this feed salinity.' });
  if (dg.Kn > 1 && p.mdModel === 'molecular') W.push({ level: 'info', msg: `Knudsen number is ${fmt(dg.Kn, 3)}: Knudsen diffusion dominates, so the molecular-diffusion-only model overestimates the flux.` });
  if (!W.some((w) => w.level !== 'info')) W.unshift({ level: 'info', msg: `Vapour transport is in the ${dg.regime} regime (Kn = ${fmt(dg.Kn, 3)}); energy and mass balances close.` });
}

function mdPlots(r, p) {
  const m = r.m, xs = m.segs.map((g) => g.x), Sl = r.rec > 0 ? r.Sloop : r.Sfeed, one = (ov, S = Sl) => { const q = { ...p, ...ov }, c = mdConfig(q, q.Tf, q.Tp, S); return mdLocal(c, q.Tf, S, { T: q.Tp, pg: (0.622 * 0 + (q.rhGas / 100) * psat(q.Tp)) }); };
  const Ts = linspace(Math.max(p.Tp + 8, 35), 90, 12), us = [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2], Ss = [0, 35, 70, 120, 180, 240, 300], fT = linspace(Math.max(p.Tp + 10, 40), 90, 6), fU = [0.05, 0.15, 0.3, 0.6, 1];
  const lo = m.segs[0], res = { 'Feed boundary layer': 1 / m.c.hf, 'Membrane (vapour + conduction in parallel)': lo.q > 0 && lo.Tfm > lo.Tc ? (lo.Tfm - lo.Tc) / lo.q : 0, 'Cold side': m.c.type === 'vmd' || !(lo.q > 0) ? 0 : Math.max(0, lo.Tc - lo.Tb) / lo.q };
  return [
    { type: 'line', title: 'Temperature profiles along the module', xlabel: 'Distance from the feed inlet (m)', ylabel: '°C', series: [{ name: 'Feed bulk', x: xs, y: m.segs.map((g) => g.Tf) }, { name: 'Feed-side membrane surface', x: xs, y: m.segs.map((g) => g.Tfm), dash: true }, { name: m.c.type === 'agmd' ? 'Condensing surface' : m.c.type === 'vmd' ? 'Saturation at permeate pressure' : 'Cold-side membrane surface', x: xs, y: m.segs.map((g) => g.Tc), dash: true }, ...(m.c.type === 'vmd' ? [] : [{ name: m.c.type === 'sgmd' ? 'Sweep gas bulk' : 'Coolant / permeate bulk', x: xs, y: m.segs.map((g) => g.Tb) }])], note: 'The gap between bulk and surface curves is temperature polarisation.' },
    { type: 'line', title: 'Flux and thermal efficiency along the module', xlabel: 'Distance from the feed inlet (m)', ylabel: 'L/m²·h · %', series: [{ name: 'Vapour flux (L/m²·h)', x: xs, y: m.segs.map((g) => g.N * 3600) }, { name: 'Thermal efficiency (%)', x: xs, y: m.segs.map((g) => 100 * g.eta) }, { name: 'Temperature-polarisation coefficient × 100', x: xs, y: m.segs.map((g) => 100 * g.tpc) }] },
    { type: 'bar', title: 'Heat-transfer resistances at the feed inlet', ylabel: 'm²·K/kW', categories: Object.keys(res), series: [{ name: 'Resistance', values: Object.values(res).map((x) => x * 1000) }] },
    { type: 'line', title: 'Flux versus feed temperature', xlabel: 'Hot-feed temperature (°C)', ylabel: 'L/m²·h · %', series: [{ name: 'Flux (L/m²·h)', x: Ts, y: Ts.map((T) => tryOr(() => one({ Tf: T }).N * 3600)), mode: 'both' }, { name: 'Thermal efficiency (%)', x: Ts, y: Ts.map((T) => tryOr(() => 100 * one({ Tf: T }).eta)), mode: 'both' }], vlines: [{ x: p.Tf, label: 'operating' }], note: 'Module-inlet conditions at the loop salinity.' },
    { type: 'line', title: 'Flux versus cross-flow velocity', xlabel: 'Feed velocity (m/s)', ylabel: 'L/m²·h · –', series: [{ name: 'Flux (L/m²·h)', x: us, y: us.map((u) => tryOr(() => one({ uFm: u, uPm: u }).N * 3600)), mode: 'both' }, { name: 'Temperature-polarisation coefficient × 50', x: us, y: us.map((u) => tryOr(() => 50 * one({ uFm: u, uPm: u }).tpc)), mode: 'both' }], vlines: [{ x: p.uFm, label: 'operating' }] },
    { type: 'line', title: 'Flux versus feed salinity', xlabel: 'Salinity (g/kg)', ylabel: 'L/m²·h', series: [{ name: 'Flux', x: Ss, y: Ss.map((S) => tryOr(() => one({}, S).N * 3600)), mode: 'both' }], vlines: [{ x: Sl, label: 'loop' }, { x: 264, label: 'NaCl saturation' }], note: 'MD loses little flux with salinity, unlike pressure-driven membranes — its niche is brine concentration.' },
    { type: 'field', title: 'Flux versus feed temperature and velocity', xlabel: 'Hot-feed temperature (°C)', ylabel: 'Feed velocity (m/s)', zlabel: 'Flux', zunit: 'L/m²·h', x: fT, y: fU, z: fU.map((u) => fT.map((T) => tryOr(() => one({ Tf: T, uFm: u, uPm: u }).N * 3600, 0))), cmap: 'thermal', contours: 8, markers: [{ x: clamp(p.Tf, fT[0], 90), y: clamp(p.uFm, 0.05, 1), label: 'operating' }] },
  ];
}

function mdTables(r, p) {
  const m = r.m, c = m.c, lo = m.segs[0], dg = lo.dg;
  return [
    { title: 'Profile along the module', columns: ['Position (m)', 'Feed T (°C)', 'Membrane T, feed side (°C)', 'Cold-side surface T (°C)', 'Cold bulk T (°C)', 'Salinity (g/kg)', 'Wall salinity (g/kg)', 'Flux (L/m²·h)', 'Heat flux (kW/m²)', 'Conduction loss (kW/m²)', 'Thermal efficiency (%)', 'TPC', 'Membrane coefficient (10⁻⁷ kg/m²·s·Pa)'],
      rows: m.segs.filter((_, k) => m.segs.length <= 40 || k % Math.ceil(m.segs.length / 40) === 0).map((g) => [g.x, g.Tf, g.Tfm, g.Tc, g.Tb, g.S, g.Sm, g.N * 3600, g.q / 1000, g.qc / 1000, 100 * g.eta, g.tpc, g.B * 1e7]) },
    { title: 'Membrane and transport data', columns: ['Quantity', 'Value', 'Unit'], rows: [['Knudsen number', dg.Kn, '–'], ['Transport regime', dg.regime, ''], ['Knudsen coefficient', dg.Bk * 1e7, '10⁻⁷ kg/m²·s·Pa'], ['Molecular-diffusion coefficient', dg.Bd * 1e7, '10⁻⁷ kg/m²·s·Pa'], ['Viscous (Poiseuille) coefficient', dg.Bv * 1e7, '10⁻⁷ kg/m²·s·Pa'], ['Effective coefficient used', lo.B * 1e7, '10⁻⁷ kg/m²·s·Pa'],
      ['Membrane thermal conductivity', c.km, 'W/m·K'], ['Membrane conduction coefficient', c.hm, 'W/m²·K'], ['Feed-side heat-transfer coefficient', c.hf, 'W/m²·K'], ['Cold-side heat-transfer coefficient', c.type === 'sgmd' ? c.hg : c.type === 'agmd' ? c.hc : c.hp, 'W/m²·K'], ['Feed Reynolds number', c.chF.Re, '–'], ['Surface tension of the feed', r.gamma * 1000, 'mN/m'], ['Liquid-entry pressure', r.lep / 1e5, 'bar'], ['Wetting margin', r.lep / 1e5 - p.pFeed, 'bar'], ['Concentration-polarisation factor', m.cpc, '–'], ['Feed pressure drop', m.dpF / 1e5, 'bar']] },
  ];
}

function runMD(v) {
  const r = simulateMD(v), p = v, W = [], m = r.m, T = p.T;
  mdWarnings(r, p, W);
  const Qp = (r.prod * 3600) / density(25, 0), Qb = (r.brine * 3600) / density(25, r.Sbrine), hs = heatSource(p, r.Qheat / 1000, p.Tf, W), name = { dcmd: 'DCMD', agmd: 'AGMD', vmd: 'VMD', sgmd: 'SGMD' }[m.c.type];
  const tables = [{ title: 'Plant summary', columns: ['Quantity', 'Value', 'Unit'], rows: [['Make-up feed', p.Qf, 'm³/h'], ['Distillate', Qp, 'm³/h'], ['Brine', Qb, 'm³/h'], ['Brine salinity', r.Sbrine, 'g/kg'], ['Recirculating feed flow', (r.recirc * 3600) / density(p.Tf, r.Sloop), 'm³/h'], ['Membrane area', r.area, 'm²'], ['Total channel width', r.width, 'm'], ['Single-pass recovery of the module', 100 * m.recovery, '%'], ['Feed outlet temperature', m.TfOut, '°C'], ['Cold-stream outlet temperature', m.coldOut, '°C'], ['Heat supplied', r.Qheat / 1000, 'kW'], ['Heat recovered internally', (m.Qrec * r.width) / 1000, 'kW'], ['Pumping power', (m.Wpump * r.width) / 1000, 'kW']] }, ...mdTables(r, p)];
  if (hs) tables.push({ title: hs.title, columns: ['Quantity', 'Value'], rows: hs.rows, note: hs.note });
  return {
    summary: `${name} produces ${fmt(Qp, 4)} m³/h of distillate from ${fmt(p.Qf, 4)} m³/h at ${fmt(m.fluxLMH, 3)} L/m²·h on ${fmt(r.area, 4)} m² (${p.Tf} / ${p.Tp} °C), with ${fmt(100 * m.eta, 3)} % thermal efficiency, a gain-output ratio of ${fmt(r.gor, 3)} and ${fmt(r.sth, 4)} kWh of heat per m³.`,
    warnings: W,
    kpis: [
      { label: 'Average flux', value: m.fluxLMH, unit: 'L/m²·h', status: m.fluxLMH <= 0 ? 'bad' : 'ok' }, { label: 'Distillate', value: Qp, unit: 'm³/h' }, { label: 'Overall recovery', value: 100 * r.recOverall, unit: '%' },
      { label: 'Thermal efficiency', value: 100 * m.eta, unit: '%', status: m.eta < 0.4 && m.c.type !== 'vmd' ? 'warn' : 'ok', help: 'Latent heat ÷ total heat crossing the membrane' }, { label: 'Gain-output ratio', value: r.gor, unit: '–' }, { label: 'Specific heat', value: r.sth, unit: 'kWh/m³' },
      { label: 'Specific electricity', value: r.sel, unit: 'kWh/m³' }, { label: 'Temperature-polarisation coefficient', value: m.tpc, unit: '–', help: 'Trans-membrane ÷ bulk temperature difference' }, { label: 'Concentration-polarisation factor', value: m.cpc, unit: '–' },
      { label: 'Membrane area', value: r.area, unit: 'm²' }, { label: 'Liquid-entry pressure', value: r.lep / 1e5, unit: 'bar', status: r.lep / 1e5 - p.pFeed < 0 ? 'bad' : r.lep / 1e5 - p.pFeed < 1 ? 'warn' : 'ok' }, { label: 'Knudsen number', value: m.segs[0].dg.Kn, unit: '–', help: m.segs[0].dg.regime + ' regime' },
      { label: 'Brine salinity', value: r.Sbrine, unit: 'g/kg', status: r.sat.nacl >= 1 ? 'bad' : 'ok' }, { label: 'Heat demand', value: r.Qheat / 1000, unit: 'kW' }, { label: 'Feed outlet temperature', value: m.TfOut, unit: '°C' },
    ],
    recommendations: [
      m.tpc < 0.5 ? 'Temperature polarisation is strong: raise the cross-flow velocity or use spacers before changing the membrane.' : null,
      r.gor < 1 && m.c.type === 'dcmd' ? 'For better heat economy use air-gap modules with internal heat recovery, longer channels or multi-stage arrangements (gain-output ratios of 3–8 are reported).' : null,
      r.lep / 1e5 - p.pFeed < 1 ? 'Increase the wetting margin (LEP − feed pressure) to at least 1 bar.' : null,
      p.source === 'heat' ? 'MD runs on 50–85 °C heat: try the solar or waste-heat option to size the heat supply.' : null,
      'Send the brine to suite 9 (Brine concentration and ZLD) or suite 2 (Chemistry) for the crystallisation sequence.',
    ].filter(Boolean),
    plots: clean(mdPlots(r, p)), tables,
    balances: [{ name: 'Module energy per metre width (W/m)', in: m.balance.in, out: m.balance.out }, { name: 'Water (kg/s)', in: r.make, out: r.prod + r.brine }, { name: 'Salt (kg/s)', in: (r.make * r.Sfeed) / 1000, out: (r.brine * (r.rec > 0 ? r.Sfeed / (1 - r.rec) : (r.make * r.Sfeed) / r.brine)) / 1000 }],
    outputs: { flux: m.fluxLMH, secThermal: r.sth, secElec: r.sel, area: r.area, recovery: r.recOverall, gor: r.gor, thermalEfficiency: m.eta, heat: r.Qheat / 1000, lepBar: r.lep / 1e5, process: 'md-' + m.c.type,
      streams: { product: stream(Qp, m.coldOut, 6.5, cloneIons({})), concentrate: stream(Qb, m.TfOut, p.pH, scaleIons(r.ions, tdsFromSalinity(r.Sbrine, 25) / Math.max(r.tdsF, 1e-9))) } },
  };
}

function runROMD(v) {
  const p = v, W = [], ionsF = scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1);
  let ro;
  try { ro = simulateRO({ ...defaultsOf(roSuite), ions: ionsF, Qf: p.Qf, T: p.T, pH: p.pH, recovery: p.roRec, targetFlux: p.roFlux, design: 'auto', mode: 'recovery', nSeg: 2 }); }
  catch (e) { throw new Error(`The RO stage could not be solved at ${p.roRec} % recovery: ${e.message}`); }
  if (ro.p1.Pf > ro.cfg.M.pmax) W.push({ level: 'bad', msg: `RO feed pressure ${fmt(ro.p1.Pf, 3)} bar exceeds the ${ro.cfg.M.pmax} bar element rating — lower the RO recovery and let the MD stage do more.` });
  const mdRec = p.mdRec > 0 ? p.mdRec : 50, r = simulateMD({ ...p, ions: ro.conc.ions, salinityFactor: 1, Qf: ro.conc.Q, mdRec }), m = r.m;
  mdWarnings(r, { ...p, mdRec }, W);
  const QpMD = (r.prod * 3600) / density(25, 0), Qb = (r.brine * 3600) / density(25, r.Sbrine), Qp = ro.product.Q + QpMD, hs = heatSource(p, r.Qheat / 1000, p.Tf, W), Pmd = (m.Wpump * r.width) / 1000;
  const prodIons = scaleIons(ro.product.ions, ro.product.Q / Qp), secE = (ro.power + Pmd) / Qp, secT = r.Qheat / 1000 / Qp, rec = Qp / p.Qf;
  const tables = [
    { title: 'Contribution of each process', columns: ['Process', 'Feed (m³/h)', 'Product (m³/h)', 'Recovery of its feed (%)', 'Share of total product (%)', 'Electricity (kW)', 'Heat (kW)', 'Electricity (kWh/m³ of its product)', 'Heat (kWh/m³ of its product)', 'Membrane area (m²)'],
      rows: [['Reverse osmosis', p.Qf, ro.product.Q, 100 * ro.overallRec, (100 * ro.product.Q) / Qp, ro.power, 0, ro.sec, 0, ro.area], ['Membrane distillation', ro.conc.Q, QpMD, 100 * r.recOverall, (100 * QpMD) / Qp, Pmd, r.Qheat / 1000, r.sel, r.sth, r.area], ['Hybrid total', p.Qf, Qp, 100 * rec, 100, ro.power + Pmd, r.Qheat / 1000, secE, secT, ro.area + r.area]],
      note: `Brine volume falls from ${fmt(ro.conc.Q, 4)} to ${fmt(Qb, 4)} m³/h (${fmt(100 * (1 - Qb / ro.conc.Q), 3)} % reduction); the MD modules operate at the loop salinity of ${fmt(r.Sloop, 3)} g/kg.` },
    { title: 'Streams', columns: ['Stream', 'Flow (m³/h)', 'TDS (mg/L)', 'Temperature (°C)'], rows: [['Feed', p.Qf, tds(ionsF), p.T], ['RO permeate', ro.product.Q, tds(ro.product.ions), p.T], ['RO concentrate → MD', ro.conc.Q, tds(ro.conc.ions), p.T], ['MD distillate', QpMD, 0, m.coldOut], ['Blended product', Qp, tds(prodIons), p.T], ['Final brine', Qb, tdsFromSalinity(r.Sbrine, 25), m.TfOut]] },
    ...mdTables(r, p),
  ];
  if (hs) tables.push({ title: hs.title, columns: ['Quantity', 'Value'], rows: hs.rows, note: hs.note });
  const recs = [30, 45, 60, 75].filter((x) => x !== mdRec).concat(mdRec).sort((a, b) => a - b), sw = recs.map((x) => { try { const q = simulateMD({ ...p, ions: ro.conc.ions, salinityFactor: 1, Qf: ro.conc.Q, mdRec: x, nSeg: Math.min(p.nSeg, 8) }); return [x, 100 * (ro.product.Q + (q.prod * 3600) / 997) / p.Qf, q.m.fluxLMH, q.Sbrine]; } catch { return [x, NaN, NaN, NaN]; } });
  return {
    summary: `RO recovers ${fmt(100 * ro.overallRec, 3)} % at ${fmt(ro.p1.Pf, 3)} bar and MD concentrates its brine by a further ${fmt(100 * r.recOverall, 3)} %, for an overall recovery of ${fmt(100 * rec, 3)} % (${fmt(Qp, 4)} m³/h) using ${fmt(secE, 3)} kWh/m³ of electricity and ${fmt(secT, 3)} kWh/m³ of heat; final brine ${fmt(r.Sbrine, 3)} g/kg.`,
    warnings: W,
    kpis: [
      { label: 'Total product', value: Qp, unit: 'm³/h' }, { label: 'Overall recovery', value: 100 * rec, unit: '%' }, { label: 'RO product', value: ro.product.Q, unit: 'm³/h' }, { label: 'MD distillate', value: QpMD, unit: 'm³/h' },
      { label: 'Combined electricity', value: secE, unit: 'kWh/m³' }, { label: 'Combined heat', value: secT, unit: 'kWh/m³' }, { label: 'RO feed pressure', value: ro.p1.Pf, unit: 'bar', status: ro.p1.Pf > ro.cfg.M.pmax ? 'bad' : 'ok' }, { label: 'RO specific energy', value: ro.sec, unit: 'kWh/m³' },
      { label: 'MD flux', value: m.fluxLMH, unit: 'L/m²·h' }, { label: 'MD thermal efficiency', value: 100 * m.eta, unit: '%' }, { label: 'MD gain-output ratio', value: r.gor, unit: '–' }, { label: 'MD membrane area', value: r.area, unit: 'm²' }, { label: 'RO membrane area', value: ro.area, unit: 'm²' },
      { label: 'Final brine flow', value: Qb, unit: 'm³/h' }, { label: 'Final brine salinity', value: r.Sbrine, unit: 'g/kg', status: r.sat.nacl >= 1 ? 'bad' : 'ok' }, { label: 'Brine volume reduction', value: 100 * (1 - Qb / ro.conc.Q), unit: '%' }, { label: 'Product TDS', value: tds(prodIons), unit: 'mg/L' },
    ],
    recommendations: [r.sat.gypsum > 1 ? 'Calcium sulphate limits the MD loop: soften or dose antiscalant ahead of the MD stage (see suite 2).' : null, 'Use waste heat or solar heat for the MD stage; its electricity demand is small.', 'Send the final brine to suite 9 (ZLD) for the crystalliser, or to suite 5 for discharge.'].filter(Boolean),
    plots: clean([
      { type: 'bar', title: 'Product and energy contribution of each process', ylabel: 'see categories', categories: ['Product (m³/h)', 'Electricity (kW)', 'Heat (kW ÷ 10)'], series: [{ name: 'Reverse osmosis', values: [ro.product.Q, ro.power, 0] }, { name: 'Membrane distillation', values: [QpMD, Pmd, r.Qheat / 1e4] }] },
      { type: 'line', title: 'Effect of the MD loop recovery', xlabel: 'MD recovery of the RO brine (%)', ylabel: 'see legend', series: [{ name: 'Overall recovery (%)', x: sw.map((q) => q[0]), y: sw.map((q) => q[1]), mode: 'both' }, { name: 'MD flux (L/m²·h)', x: sw.map((q) => q[0]), y: sw.map((q) => q[2]), mode: 'both' }, { name: 'Final brine salinity ÷ 5 (g/kg)', x: sw.map((q) => q[0]), y: sw.map((q) => q[3] / 5), mode: 'both' }], vlines: [{ x: mdRec, label: 'operating' }] },
      ...mdPlots(r, { ...p, mdRec }),
    ]),
    tables,
    balances: [{ name: 'Water (m³/h)', in: p.Qf, out: ro.product.Q + ro.conc.Q }, { name: 'MD water (kg/s)', in: r.make, out: r.prod + r.brine }, { name: 'MD module energy (W/m)', in: m.balance.in, out: m.balance.out }],
    outputs: { flux: m.fluxLMH, secThermal: secT, secElec: secE, area: ro.area + r.area, recovery: rec, roRecovery: ro.overallRec, mdRecovery: r.recOverall, heat: r.Qheat / 1000, power: ro.power + Pmd, process: 'ro-md',
      streams: { product: stream(Qp, p.T, 6.8, prodIons), concentrate: stream(Qb, m.TfOut, p.pH, scaleIons(ro.conc.ions, tdsFromSalinity(r.Sbrine, 25) / Math.max(tds(ro.conc.ions), 1e-9))) } },
  };
}

function runHDH(v) {
  const r = simulateHDH(v), W = [], p = v, Qp = (r.pw * 3600) / 997, hs = heatSource(p, r.Qh / 1000, p.Ttop, W);
  if (!r.converged) W.push({ level: 'warn', msg: 'The cycle iteration did not fully converge — check the effectiveness values and the mass ratio.' });
  if (r.gor < 1) W.push({ level: 'info', msg: `Gain-output ratio is ${fmt(r.gor, 3)}: the streams are thermally unbalanced. Scan the mass ratio in the plot below.` });
  if (!W.length) W.push({ level: 'info', msg: 'Cycle solved; energy and water balances close.' });
  const MRs = linspace(1, 8, 15), tops = linspace(55, 90, 8), at = (ov) => { try { const q = simulateHDH({ ...p, ...ov }); return q.converged && q.pw > 0 ? q : null; } catch { return null; } };
  const sw = MRs.map((x) => at({ MR: x })), st = tops.map((x) => at({ Ttop: x })), fM = [1.5, 2.5, 3.5, 5, 7], fT = [55, 65, 75, 85];
  const tables = [{ title: 'Cycle state points', columns: ['Point', 'Temperature (°C)', 'Flow (kg/s)', 'Humidity ratio (g/kg dry air)'], rows: [['Seawater in', r.T0, r.mw, null], ['Seawater after dehumidifier (preheated)', r.T1, r.mw, null], ['Seawater after heater', r.T2, r.mw, null], ['Brine leaving humidifier', r.T3, r.mw - r.pw, null], ['Air leaving dehumidifier (cold, saturated)', r.Ta1, r.ma, 1000 * wSat(r.Ta1)], ['Air leaving humidifier (hot, saturated)', r.Ta2, r.ma, 1000 * wSat(r.Ta2)], ['Condensate', r.Ta1, r.pw, null]] },
    { title: 'Performance', columns: ['Quantity', 'Value', 'Unit'], rows: [['Heater duty', r.Qh / 1000, 'kW'], ['Distillate', Qp, 'm³/h'], ['Gain-output ratio', r.gor, '–'], ['Recovery ratio', 100 * r.rr, '%'], ['Specific heat', r.sth, 'kWh/m³'], ['Specific electricity (fan + pump)', r.sel, 'kWh/m³'], ['Brine salinity', r.Sbrine, 'g/kg']] }];
  if (hs) tables.push({ title: hs.title, columns: ['Quantity', 'Value'], rows: hs.rows, note: hs.note });
  return {
    summary: `The humidification–dehumidification cycle produces ${fmt(Qp, 3)} m³/h of distillate from ${fmt(p.Qf, 3)} m³/h of seawater heated to ${p.Ttop} °C, with a gain-output ratio of ${fmt(r.gor, 3)} (${fmt(r.sth, 4)} kWh of heat per m³).`,
    warnings: W,
    kpis: [{ label: 'Distillate', value: Qp, unit: 'm³/h' }, { label: 'Gain-output ratio', value: r.gor, unit: '–', status: r.gor < 1 ? 'warn' : 'ok' }, { label: 'Recovery ratio', value: 100 * r.rr, unit: '%' }, { label: 'Specific heat', value: r.sth, unit: 'kWh/m³' }, { label: 'Specific electricity', value: r.sel, unit: 'kWh/m³' }, { label: 'Heater duty', value: r.Qh / 1000, unit: 'kW' },
      { label: 'Air temperature, humidifier outlet', value: r.Ta2, unit: '°C' }, { label: 'Air temperature, dehumidifier outlet', value: r.Ta1, unit: '°C' }, { label: 'Seawater preheat temperature', value: r.T1, unit: '°C' }, { label: 'Brine temperature', value: r.T3, unit: '°C' }, { label: 'Dry-air flow', value: r.ma, unit: 'kg/s' }, { label: 'Brine salinity', value: r.Sbrine, unit: 'g/kg' }],
    recommendations: ['Tune the mass ratio to the peak of the gain-output curve; air extraction between the columns (not modelled) raises it further.', 'HDH tolerates poor feed quality and uses low-grade heat: it suits small, decentralised units — compare cost with MD and small RO in suite 13.'],
    plots: clean([
      { type: 'line', title: 'Gain-output ratio versus mass ratio', xlabel: 'Seawater-to-dry-air mass ratio', ylabel: 'GOR · %', series: [{ name: 'Gain-output ratio', x: MRs, y: sw.map((q) => (q ? q.gor : NaN)), mode: 'both' }, { name: 'Recovery ratio (%)', x: MRs, y: sw.map((q) => (q ? 100 * q.rr : NaN)), mode: 'both' }], vlines: [{ x: p.MR, label: 'operating' }] },
      { type: 'line', title: 'Effect of the top temperature', xlabel: 'Top seawater temperature (°C)', ylabel: 'GOR · %', series: [{ name: 'Gain-output ratio', x: tops, y: st.map((q) => (q ? q.gor : NaN)), mode: 'both' }, { name: 'Recovery ratio (%)', x: tops, y: st.map((q) => (q ? 100 * q.rr : NaN)), mode: 'both' }], vlines: [{ x: p.Ttop, label: 'operating' }] },
      { type: 'line', title: 'Temperature levels of the cycle', xlabel: 'State point', ylabel: '°C', series: [{ name: 'Seawater / brine', x: [1, 2, 3, 4], y: [r.T0, r.T1, r.T2, r.T3], mode: 'both' }, { name: 'Air', x: [2, 3, 4], y: [r.Ta1, r.Ta2, r.Ta1], mode: 'both' }], note: '1 seawater inlet · 2 after dehumidifier · 3 after heater / humidifier top · 4 humidifier bottom.' },
      { type: 'bar', title: 'Energy flows', ylabel: 'kW', categories: ['Heater duty', 'Heat recovered in dehumidifier', 'Latent heat of the product', 'Rejected with brine'], series: [{ name: 'kW', values: [r.Qh / 1000, (r.mw * (hL(r.T1, r.S) - hL(r.T0, r.S))) / 1000, (r.pw * latentHeat(r.T0)) / 1000, ((r.mw - r.pw) * (hL(r.T3, r.S) - hL(r.T0, r.S))) / 1000] }] },
      { type: 'field', title: 'Gain-output ratio versus mass ratio and top temperature', xlabel: 'Mass ratio', ylabel: 'Top temperature (°C)', zlabel: 'GOR', zunit: '–', x: fM, y: fT, z: fT.map((T) => fM.map((M) => at({ MR: M, Ttop: T })?.gor ?? 0)), cmap: 'viridis', contours: 8, markers: [{ x: clamp(p.MR, 1.5, 7), y: clamp(p.Ttop, 55, 85), label: 'operating' }] },
    ]),
    tables,
    balances: [{ name: 'Energy (kW)', in: r.balance.in / 1000, out: r.balance.out / 1000 }, { name: 'Water (kg/s)', in: r.mw, out: r.mw - r.pw + r.pw }],
    outputs: { flux: 0, secThermal: r.sth, secElec: r.sel, area: 0, recovery: r.rr, gor: r.gor, heat: r.Qh / 1000, process: 'hdh',
      streams: { product: stream(Qp, r.Ta1, 6.5, cloneIons({})), concentrate: stream(((r.mw - r.pw) * 3600) / density(r.T3, r.Sbrine), r.T3, p.pH, scaleIons(cloneIons(p.ions), ((p.salinityFactor ?? 1) * r.Sbrine) / Math.max(r.S, 1e-9))) } },
  };
}

// ---- result builders of the emerging processes, hybrid chains and optional modules -----------------------------------------
function cdiResult(r, p, W, name = 'CDI') {
  const kpis = [], plots = [], tables = [], Vt = (R * K(r.T)) / F, psiD = Vt * r.pdOf(r.sigmaEnd[0], r.cEnd), stern = (r.sigmaEnd[0] * F) / p.cdiCst;
  if (p.cdiV > 1.23) W.push({ level: 'warn', msg: 'Charging above 1.23 V risks water electrolysis and carbon oxidation.' });
  if (r.c0 > 100) W.push({ level: 'warn', msg: `The electrode feed holds ${fmt(r.c0, 3)} eq/m³: capacitive deionisation suits water below about 3–5 g/L.` });
  if (r.removal < 0.1) W.push({ level: 'info', msg: `Only ${fmt(100 * r.removal, 3)} % of the salt is removed per pass — lower the flow per cell area or lengthen the charging step.` });
  if (r.fallback) W.push({ level: 'warn', msg: `The thin-layer (Gouy–Chapman–Stern) model has no solution here — its double layers would hold more co-ions than the pores contain — so the unified double-layer model was used instead.` });
  else if (r.edlModel === 'auto' && r.thetaMin < 0.999) W.push({ level: 'info', msg: `Double layers ${r.thetaFeed < 0.001 ? 'overlap in the pores' : 'approach the pore size'}: pore half-width ${fmt(r.hPore * 1e9, 3)} nm against a Debye length of ${fmt(r.lamD * 1e9, 3)} nm in the feed. The unified model uses ${r.thetaFeed < 0.001 ? 'the modified-Donnan relations' : `a blend (thin-layer weight ${fmt(r.thetaFeed, 2)} in the feed, down to ${fmt(r.thetaMin, 2)} in the depleted pore liquid)`}.` });
  if (r.failedSteps > 0) W.push({ level: 'warn', msg: `${r.failedSteps} implicit time step(s) did not converge and were skipped; the balances still close but the cycle is slightly shortened. Raise the number of output steps.` });
  if (r.tauRC * 3 > p.cdiTc * 60) W.push({ level: 'info', msg: `The charging step (${p.cdiTc} min) is shorter than three electrode time constants (${fmt((3 * r.tauRC) / 60, 3)} min): the inner part of the electrode stays under-used.` });
  let edl = null; try { edl = edlProfile(Math.max(psiD, 1e-4), r.cEnd, r.T); if (!edl.converged) edl = null; } catch { edl = null; }
  kpis.push({ label: `${name} salt removal`, value: 100 * r.removal, unit: '%' }, { label: 'Salt adsorption per cycle', value: r.sac, unit: 'mg/g', help: 'mg NaCl per gram of both electrodes' }, { label: 'Charge efficiency', value: 100 * r.eff, unit: '%', help: 'Salt removed per unit of charge; locally tanh(Δφ_d/2) for thin (Gouy–Chapman–Stern) and tanh(Δφ_D) for overlapping (modified-Donnan) double layers' },
    { label: `${name} specific energy`, value: r.sec, unit: 'kWh/m³' }, { label: 'Energy per mole of salt', value: r.ePerMol, unit: 'kJ/mol' }, { label: 'Electrode time constant L²·a·C/κ', value: r.tauRC, unit: 's' }, { label: 'Diffuse-layer potential at end of charge', value: 1000 * psiD, unit: 'mV', help: `Stern-layer drop ${fmt(1000 * stern, 4)} mV` },
    { label: 'Langmuir coverage at end of charge', value: r.thetaEnd, unit: '–', help: `Equilibrium coverage at the feed concentration ${fmt(r.thEq, 3)}` });
  const prof = r.snaps.filter((_, k) => k > 0);
  plots.push({ type: 'line', title: 'Effluent concentration during the last cycle', xlabel: 'Time (min)', ylabel: 'mg/L as NaCl', series: [{ name: 'Effluent', x: r.tt, y: r.cEff.map((c) => c * 58.44) }], hlines: [{ y: r.c0 * 58.44, label: 'feed' }], vlines: [{ x: p.cdiTc, label: 'discharge starts' }] },
    { type: 'line', title: 'Cell current and voltage', xlabel: 'Time (min)', ylabel: 'A/m² · V', series: [{ name: 'Current density (A/m²)', x: r.tt, y: r.cur }, { name: 'Cell voltage × 10 (V)', x: r.tt, y: r.volt.map((x) => 10 * x) }], note: p.cdiMode === 'cc' ? `Prescribed current ${p.cdiI} A/m² until the cell reaches ${p.cdiV} V, then constant voltage.` : `Prescribed voltage ${p.cdiV} V; the current is limited by the spacer, contact and pore-electrolyte resistances.` },
    { type: 'line', title: 'Charge penetrating the porous electrode (transmission-line charging)', xlabel: 'Depth from the spacer (µm)', ylabel: 'Surface charge (mC/m²)', series: prof.map((q) => ({ name: `t = ${fmt(q.t, 3)} s`, x: r.x.map((x) => x * 1e6), y: q.sigma.map((s) => s * F * 1000) })), note: 'No ionic flux at the current collector (right edge); the charging front diffuses in with the time constant L²·a·C/κ.' });
  if (edl) plots.push({ type: 'line', title: 'Double layer at the carbon surface: Poisson–Nernst–Planck versus Gouy–Chapman', xlabel: 'Distance from the surface (nm)', ylabel: 'mV · c/c_bulk', series: [{ name: 'Potential, Poisson–Nernst–Planck (mV)', x: edl.x.map((x) => x * 1e9), y: edl.psi.map((q) => 1000 * q) }, { name: 'Potential, Gouy–Chapman (mV)', x: edl.x.map((x) => x * 1e9), y: edl.gc.map((q) => 1000 * q), mode: 'points' }, { name: 'Counter-ion c/c_bulk', x: edl.x.map((x) => x * 1e9), y: edl.c[1].map((c) => c / r.cEnd) }, { name: 'Co-ion c/c_bulk × 10', x: edl.x.map((x) => x * 1e9), y: edl.c[0].map((c) => (10 * c) / r.cEnd) }], vlines: [{ x: edl.lam * 1e9, label: 'Debye length' }], note: 'Diffuse part of the double layer at the end of charging, resolved on a Debye-graded mesh; the Stern layer carries the rest of the electrode potential.' });
  tables.push({ title: `${name} cycle`, columns: ['Quantity', 'Value', 'Unit'], rows: [['Feed concentration (1:1 equivalent)', r.c0, 'mol/m³'], ['Average product concentration', r.cAvg, 'mol/m³'], ['Salt removed per cycle', r.salt * 1000, 'mmol/m²'], ['— stored in the double layers', r.storedEDL * 1000, 'mmol/m²'], ['— adsorbed (Langmuir)', r.storedLang * 1000, 'mmol/m²'], ['— change of spacer and macropore inventory', r.storedMix * 1000, 'mmol/m²'], ['Charge passed on charging', r.charge, 'C/m²'], ['Charge stored in the double layers', r.chargeStored, 'C/m²'], ['Charge efficiency', 100 * r.eff, '%'], ['Net energy', r.Enet, 'J/m²'], ['Water recovery', 100 * r.waterRec, '%'], ['Productivity', r.prod, 'L/m²·h']] },
    { title: 'Double layer and electrode', columns: ['Quantity', 'Value', 'Unit'], rows: [['Double-layer model used', r.edlModel === 'auto' ? `unified — ${r.regime}` : r.edlModel === 'gcs' ? 'Gouy–Chapman–Stern (forced)' : 'modified Donnan (forced)', ''], ['Pore half-width (pore volume ÷ wall area)', r.hPore * 1e9, 'nm'], ['Pore half-width ÷ Debye length in the feed', r.poreRatio, '–'], ['Thin-layer weight θ in the feed / lowest during the cycle', `${fmt(r.thetaFeed, 3)} / ${fmt(r.thetaMin, 3)}`, '–'], ['Attraction term applied', r.muEff, 'kT'], ['Equilibrium surface charge at the charging voltage', r.eq.sigma * F * 1000, 'mC/m²'], ['Surface charge at the spacer side, end of charge', r.sigmaEnd[0] * F * 1000, 'mC/m²'], ['Surface charge at the collector side, end of charge', r.sigmaEnd[r.n - 1] * F * 1000, 'mC/m²'], ['Diffuse-layer potential', 1000 * psiD, 'mV'], ['Stern-layer potential', 1000 * stern, 'mV'], ['Debye length in the feed', r.lamD * 1e9, 'nm'], ['Surface charge from the Poisson–Nernst–Planck profile', edl ? edl.sigmaLeft * 1000 : null, 'mC/m²'], ['Diffuse charge, Gouy–Chapman (Grahame)', edl ? edl.sigmaGC * 1000 : null, 'mC/m²'],
      ['Double-layer area per electrode volume', r.av / 1e6, 'm²/cm³'], ['Electrode mass (both)', r.mE, 'g/m²'], ['Electrode time constant', r.tauRC, 's'], ['Smallest implicit time step', r.dtStab, 's'], ['Implicit steps / Newton iterations', `${r.steps} / ${r.newton}`, ''], ['Cells across the electrode (used)', r.n, ''], ['Share of the liquid salt capacity taken up by the double layers (largest)', r.thick, '–'], ['Langmuir equilibrium coverage at the feed concentration', r.thEq, '–'], ['Initial electrode charge', p.cdiQ0, '% of equilibrium']] });
  const terms = [r.storedEDL, r.storedLang, r.storedMix].map((x) => 1000 * x), pos = (a) => sum(a.map((x) => Math.max(x, 0))); // gross balance: inventories that shrink are sources
  return { kpis, plots, tables, bal: [{ name: `${name} salt: taken from the flow and released inventories vs stored (mmol/m²)`, in: Math.max(r.salt * 1000, 0) + pos(terms.map((x) => -x)), out: pos(terms) + Math.max(-r.salt * 1000, 0) }, { name: `${name} charge: passed vs stored in the double layers (C/m²)`, in: r.charge, out: r.chargeStored }] };
}

function runCDI(v) {
  const p = v, W = [], ions = scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1), c0 = eqConc(ions), r = simulatePorousCDI(p, c0), q = cdiResult(r, p, W, 'CDI'), qA = p.cdiQ / 1000 / 60, cell = p.Qf / 3600 / qA / r.waterRec;
  if (!W.length) W.push({ level: 'info', msg: 'Porous-electrode model solved; salt and charge balances close.' });
  const Vs = linspace(0.4, 1.4, 5), sw = Vs.map((V) => { try { const s = simulatePorousCDI(p, c0, { cdiV: V, cdiNx: Math.min(p.cdiNx, 5), cdiCycles: 1, cdiNt: 30 }); return [s.sac, 100 * s.eff, s.sec]; } catch { return [NaN, NaN, NaN]; } });
  const ionsP = scaleIons(ions, r.cAvg / c0), ionsC = scaleIons(ions, 1 + (1 - r.cAvg / c0) * (p.cdiTc / p.cdiTd));
  return {
    summary: `The capacitive-deionisation cell removes ${fmt(100 * r.removal, 3)} % of the salt (${fmt(tds(ions), 4)} → ${fmt(tds(ionsP), 4)} mg/L) with ${fmt(r.sac, 3)} mg/g per cycle at ${fmt(100 * r.eff, 3)} % charge efficiency and ${fmt(r.sec, 3)} kWh/m³; ${fmt(cell, 4)} m² of cell area treat ${fmt(p.Qf, 3)} m³/h.`,
    warnings: W,
    kpis: [{ label: 'Product TDS (cycle average)', value: tds(ionsP), unit: 'mg/L' }, ...q.kpis, { label: 'Cell area required', value: cell, unit: 'm²' }, { label: 'Productivity', value: r.prod, unit: 'L/m²·h' }, { label: 'Water recovery', value: 100 * r.waterRec, unit: '%' }],
    recommendations: [r.eff < 0.6 ? 'Charge efficiency is low: co-ion expulsion dominates at this salinity and voltage — raise the voltage towards 1.2 V or add ion-exchange membranes (MCDI, suite 7).' : null, 'Compare with the membrane (MCDI) model and with electrodialysis in suite 7 for the same water.'].filter(Boolean),
    plots: clean([...q.plots, { type: 'line', title: 'Effect of the charging voltage', xlabel: 'Cell voltage (V)', ylabel: 'mg/g · % · kWh/m³', series: [{ name: 'Salt adsorption per cycle (mg/g)', x: Vs, y: sw.map((s) => s[0]), mode: 'both' }, { name: 'Charge efficiency (% ÷ 10)', x: Vs, y: sw.map((s) => s[1] / 10), mode: 'both' }, { name: 'Specific energy × 10 (kWh/m³)', x: Vs, y: sw.map((s) => 10 * s[2]), mode: 'both' }], vlines: [{ x: p.cdiV, label: 'operating' }], note: 'One cycle from the same initial state on a coarse electrode grid.' }]),
    tables: q.tables, balances: q.bal,
    outputs: { flux: r.prod, secThermal: 0, secElec: r.sec, area: cell, recovery: r.waterRec, saltAdsorption: r.sac, chargeEfficiency: r.eff, process: 'cdi', streams: { product: stream(p.Qf, p.T, p.pH, ionsP), concentrate: stream((p.Qf * p.cdiTd) / p.cdiTc, p.T, p.pH, ionsC) } },
  };
}

function runCDIRO(v) {
  const p = v, W = [], ionsF = scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1);
  let ro;
  try { ro = simulateRO({ ...defaultsOf(roSuite), ions: ionsF, Qf: p.Qf, T: p.T, pH: p.pH, recovery: p.roRec, targetFlux: p.roFlux, design: 'auto', mode: 'recovery', nSeg: 2 }); }
  catch (e) { throw new Error(`The RO stage could not be solved at ${p.roRec} % recovery: ${e.message}`); }
  if (ro.p1.Pf > ro.cfg.M.pmax) W.push({ level: 'bad', msg: `RO feed pressure ${fmt(ro.p1.Pf, 3)} bar exceeds the ${ro.cfg.M.pmax} bar element rating — lower the RO recovery.` });
  const c0 = Math.max(eqConc(ro.product.ions), 1e-3), r = simulatePorousCDI(p, c0), q = cdiResult(r, p, W, 'CDI polishing'), qA = p.cdiQ / 1000 / 60, Qcdi = ro.product.Q * r.waterRec, cell = ro.product.Q / 3600 / qA, Pcdi = r.sec * Qcdi;
  const ionsP = scaleIons(ro.product.ions, r.cAvg / c0), secE = (ro.power + Pcdi) / Qcdi, rec = Qcdi / p.Qf;
  if (!W.length) W.push({ level: 'info', msg: 'RO and the capacitive polishing step solved; balances close.' });
  return {
    summary: `RO recovers ${fmt(100 * ro.overallRec, 3)} % at ${fmt(ro.p1.Pf, 3)} bar and the capacitive cell polishes its permeate from ${fmt(tds(ro.product.ions), 3)} to ${fmt(tds(ionsP), 3)} mg/L, giving ${fmt(Qcdi, 4)} m³/h at ${fmt(secE, 3)} kWh/m³ in total.`,
    warnings: W,
    kpis: [{ label: 'Polished product', value: Qcdi, unit: 'm³/h' }, { label: 'Product TDS', value: tds(ionsP), unit: 'mg/L' }, { label: 'RO permeate TDS', value: tds(ro.product.ions), unit: 'mg/L' }, { label: 'Combined specific energy', value: secE, unit: 'kWh/m³' }, { label: 'Overall recovery', value: 100 * rec, unit: '%' }, { label: 'RO feed pressure', value: ro.p1.Pf, unit: 'bar', status: ro.p1.Pf > ro.cfg.M.pmax ? 'bad' : 'ok' }, ...q.kpis, { label: 'CDI cell area', value: cell, unit: 'm²' }, { label: 'RO membrane area', value: ro.area, unit: 'm²' }],
    recommendations: ['Capacitive polishing replaces a second RO pass when only a few mg/L must be removed; compare both in suite 13.', 'The desorption water can be returned to the RO feed to keep the overall recovery.'],
    plots: clean([{ type: 'bar', title: 'Contribution of each process', ylabel: 'see categories', categories: ['Salt removed (kg/h)', 'Electricity (kW)'], series: [{ name: 'Reverse osmosis', values: [(ro.product.Q * (tds(ionsF) - tds(ro.product.ions))) / 1000, ro.power] }, { name: 'Capacitive deionisation', values: [(Qcdi * (tds(ro.product.ions) - tds(ionsP))) / 1000, Pcdi] }] }, ...q.plots]),
    tables: [{ title: 'Contribution of each process', columns: ['Process', 'Feed (m³/h)', 'Product (m³/h)', 'Product TDS (mg/L)', 'Electricity (kW)', 'Specific energy (kWh/m³ of its product)', 'Area (m²)'], rows: [['Reverse osmosis', p.Qf, ro.product.Q, tds(ro.product.ions), ro.power, ro.sec, ro.area], ['Capacitive deionisation', ro.product.Q, Qcdi, tds(ionsP), Pcdi, r.sec, cell], ['Hybrid total', p.Qf, Qcdi, tds(ionsP), ro.power + Pcdi, secE, ro.area + cell]], note: 'The capacitive cell treats the whole RO permeate; its desorption flush is counted as loss.' }, ...q.tables],
    balances: [{ name: 'RO water (m³/h)', in: p.Qf, out: ro.product.Q + ro.conc.Q }, ...q.bal],
    outputs: { flux: r.prod, secThermal: 0, secElec: secE, area: ro.area + cell, recovery: rec, power: ro.power + Pcdi, process: 'cdi-ro', streams: { product: stream(Qcdi, p.T, 6.8, ionsP), concentrate: stream(ro.conc.Q, p.T, p.pH, ro.conc.ions) } },
  };
}

/** FO followed by a second process that re-concentrates the diluted draw and delivers the product: membrane distillation or electrodialysis. */
function runFOChain(v) {
  const p = v, W = [], r = simulateFO(p), d = r.d, T = r.T, Vw = r.tot.Vw * 3600, Qdil = r.drawOut.Q * 3600, md = p.process === 'fo_md';
  if (!(Vw > 1e-6 * Math.max(p.Qf, 1e-9))) { // nothing for the downstream step to treat: show the FO step on its own instead of failing
    const res = runFO({ ...v, process: 'fo' });
    res.warnings = [{ level: 'bad', msg: `Practically no water crosses the FO membrane (${fmt(Math.max(Vw, 0), 3)} m³/h), so the ${md ? 'membrane-distillation' : 'electrodialysis'} step of the hybrid has nothing to recover. The forward-osmosis step is shown on its own — raise the draw concentration or the membrane area.` }, ...res.warnings.filter((w) => w.level !== 'info' || !/module solved/.test(w.msg))];
    res.outputs = { ...res.outputs, process: 'fo (hybrid step skipped: no water transferred)' };
    return res;
  }
  if (!d.ions && !md) { // electrodialysis moves ions only: show the FO step on its own instead of failing
    const res = runFO({ ...v, process: 'fo' });
    res.warnings = [{ level: 'bad', msg: `Electrodialysis cannot regenerate a ${d.name.split(' ')[0].toLowerCase()} draw: the solute carries no charge, so no current would move it. The forward-osmosis step is shown on its own with the regeneration chosen under "Draw regeneration". Pick an ionic draw solute for this hybrid, or use the FO + MD or FO + RO hybrid, which both work with this draw.` }, ...res.warnings.filter((w) => w.level !== 'info' || !/module solved/.test(w.msg))];
    res.outputs = { ...res.outputs, process: 'fo (electrodialysis not applicable to a non-ionic draw)' };
    return res;
  }
  if (!r.conv) W.push({ level: 'warn', msg: 'The counter-current FO iteration did not fully converge.' });
  const da = drawAsIons(p.draw, r.cDout, T);
  if (da.equivalent) W.push({ level: 'info', msg: `${d.name} carries no ions: membrane distillation is solved for the osmotically equivalent NaCl solution (${fmt((da.cEq * 58.44) / 1000, 3)} g/L), which has the same water activity and therefore the same vapour-pressure lowering. The sugar is non-volatile and stays in the draw loop; its higher viscosity is not included.` });
  const ionsD = da.ions, frac = clamp(Vw / Qdil, 0.02, 0.97), eFO = r.Ppump / 1000, fracM = clamp((Vw * density(25, 0)) / (Qdil * density(25, salinityFromTDS(tds(ionsD), 25))), 0.02, 0.95); // mass fraction of the diluted draw that is the transferred water
  const feedOutIons = scaleIons(r.ions, r.fOut); if (d.ions) for (const [k, m] of Object.entries(d.ions)) feedOutIons[k] = (feedOutIons[k] || 0) + m * (r.feedOut.nd / r.feedOut.Q);
  const common = { xs: r.segs.map((g) => g.x) }, foPlot = { type: 'line', title: 'FO water flux and osmotic pressures along the module', xlabel: 'Position along the feed path (fraction)', ylabel: 'L/m²·h · bar', series: [{ name: 'Water flux (L/m²·h)', x: common.xs, y: r.segs.map((g) => g.Jw * 3.6e6) }, { name: 'Draw osmotic pressure, bulk (bar)', x: common.xs, y: r.segs.map((g) => g.piDb / 1e5) }, { name: 'Feed osmotic pressure, bulk (bar)', x: common.xs, y: r.segs.map((g) => g.piFb / 1e5) }] };
  const foK = [{ label: 'FO water flux', value: r.JwLMH, unit: 'L/m²·h' }, { label: 'Water transferred by FO', value: Vw, unit: 'm³/h' }, { label: 'FO feed recovery', value: 100 * r.recovery, unit: '%' }, { label: 'Draw dilution factor', value: r.dilution, unit: '×' }, { label: 'Reverse solute flux', value: r.JsGMH, unit: 'g/m²·h' }];
  const foBal = [{ name: 'FO water (m³/h)', in: (r.QF0 + r.QD0) * 3600, out: (r.feedOut.Q + r.drawOut.Q) * 3600 }, { name: 'FO draw solute (mol/s)', in: r.QD0 * r.cD0, out: r.drawOut.n + r.feedOut.nd }];
  if (md) {
    const q = simulateMD({ ...p, ions: ionsD, salinityFactor: 1, Qf: Qdil, mdRec: 100 * fracM }), m = q.m, Qp = (q.prod * 3600) / density(25, 0), Pmd = (m.Wpump * q.width) / 1000, hs = heatSource(p, q.Qheat / 1000, p.Tf, W);
    mdWarnings(q, { ...p, mdRec: 100 * fracM }, W);
    const tables = [{ title: 'Contribution of each process', columns: ['Process', 'Role', 'Water handled (m³/h)', 'Membrane area (m²)', 'Flux (L/m²·h)', 'Electricity (kW)', 'Heat (kW)', 'Electricity (kWh per m³ product)', 'Heat (kWh per m³ product)'],
      rows: [['Forward osmosis', `Draws ${fmt(100 * r.recovery, 3)} % of the feed into the draw`, Vw, r.area, r.JwLMH, eFO, 0, eFO / Qp, 0], ['Membrane distillation', `Re-concentrates the draw from ${fmt(r.cDout / 1000, 3)} to ${fmt(p.cDraw, 3)} mol/L`, Qp, q.area, m.fluxLMH, Pmd, q.Qheat / 1000, Pmd / Qp, q.sth], ['Hybrid total', `Overall recovery ${fmt((100 * Qp) / p.Qf, 3)} %`, Qp, r.area + q.area, null, eFO + Pmd, q.Qheat / 1000, (eFO + Pmd) / Qp, q.sth]], note: 'The MD distillate is the product; the MD concentrate returns to the FO as regenerated draw. The FO membrane keeps foulants and scalants away from the MD membrane.' }, ...mdTables(q, p)];
    if (hs) tables.push({ title: hs.title, columns: ['Quantity', 'Value'], rows: hs.rows, note: hs.note });
    return {
      summary: `FO transfers ${fmt(Vw, 4)} m³/h into the ${d.name.split(' ')[0].toLowerCase()} draw at ${fmt(r.JwLMH, 3)} L/m²·h and membrane distillation recovers it as ${fmt(Qp, 4)} m³/h of distillate at ${fmt(m.fluxLMH, 3)} L/m²·h, using ${fmt(q.sth, 4)} kWh of heat and ${fmt((eFO + Pmd) / Qp, 3)} kWh of electricity per m³.`,
      warnings: W.length ? W : [{ level: 'info', msg: 'FO–MD chain solved; balances close.' }],
      kpis: [{ label: 'Product (MD distillate)', value: Qp, unit: 'm³/h' }, { label: 'Overall recovery', value: (100 * Qp) / p.Qf, unit: '%' }, ...foK, { label: 'MD flux', value: m.fluxLMH, unit: 'L/m²·h' }, { label: 'MD membrane area', value: q.area, unit: 'm²' }, { label: 'MD thermal efficiency', value: 100 * m.eta, unit: '%' }, { label: 'Combined heat', value: q.sth, unit: 'kWh/m³' }, { label: 'Combined electricity', value: (eFO + Pmd) / Qp, unit: 'kWh/m³' }, { label: 'Draw salinity in the MD loop', value: q.Sloop, unit: 'g/kg' }],
      recommendations: ['Use waste or solar heat for the MD stage (Heat source group); its electricity demand is small.', 'A stronger draw raises the FO flux but lowers the MD vapour pressure only slightly — MD tolerates concentrated draws well.'],
      plots: clean([{ type: 'bar', title: 'Membrane area and energy of each process', ylabel: 'see categories', categories: ['Membrane area (m² ÷ 10)', 'Electricity (kW)', 'Heat (kW ÷ 100)'], series: [{ name: 'Forward osmosis', values: [r.area / 10, eFO, 0] }, { name: 'Membrane distillation', values: [q.area / 10, Pmd, q.Qheat / 1e5] }] }, foPlot, ...mdPlots(q, { ...p, mdRec: 100 * fracM })]),
      tables, balances: [...foBal, { name: 'Draw loop: water transferred by FO vs MD distillate (kg/s)', in: (Vw * density(25, 0)) / 3600, out: q.prod }, { name: 'MD module energy (W/m)', in: m.balance.in, out: m.balance.out }],
      outputs: { flux: r.JwLMH, secThermal: q.sth, secElec: (eFO + Pmd) / Qp, area: r.area + q.area, recovery: Qp / p.Qf, heat: q.Qheat / 1000, power: eFO + Pmd, process: 'fo-md', streams: { product: stream(Qp, m.coldOut, 6.5, cloneIons({})), concentrate: stream(r.feedOut.Q * 3600, T, p.pH, feedOutIons) } },
    };
  }
  let ed;
  try { ed = simulateED({ ...defaultsOf(edSuite), ions: ionsD, Qp: Vw, T, pH: 7, mode: 'design', targetTDS: p.edTarget, recovery: clamp(100 * frac, 30, 97), maxStages: 16, nSeg: 6, edr: false }, { tol: 1e-5 }); }
  catch (e) { throw new Error(`Electrodialysis of the diluted draw could not be solved: ${e.message}`); }
  if (![ed.sec, ed.tdsP, ed.Qprod, ed.area, ed.Pel, ed.eff].every(Number.isFinite) || !(ed.Qprod > 0)) { // the diluted draw is outside what an ED stack can treat: show the FO step alone
    const res = runFO({ ...v, process: 'fo' });
    res.warnings = [{ level: 'bad', msg: `Electrodialysis has no solution for the diluted draw (${fmt(tds(ionsD) / 1000, 3)} g/L): this hybrid needs a dilute draw, below about 0.3 mol/L. The forward-osmosis step is shown on its own.` }, ...res.warnings.filter((w) => w.level !== 'info' || !/module solved/.test(w.msg))];
    res.outputs = { ...res.outputs, process: 'fo (electrodialysis step has no solution)' };
    return res;
  }
  if (!ed.reached) W.push({ level: 'bad', msg: `Electrodialysis does not desalt the diluted draw (${fmt(tds(ionsD) / 1000, 3)} g/L) to ${p.edTarget} mg/L within ${ed.nSt} stages — use a weaker draw solution.` });
  if (100 * frac < 30) W.push({ level: 'warn', msg: `FO adds only ${fmt(100 * frac, 3)} % of water to the draw; the electrodialysis step was solved at its minimum recovery of 30 %.` });
  if (tds(ionsD) > 15000) W.push({ level: 'info', msg: `The diluted draw holds ${fmt(tds(ionsD) / 1000, 3)} g/L: electrodialysis energy grows in proportion to the salt it must move, so this hybrid suits dilute draws.` });
  const Ped = (ed.Pel + ed.Ppump) / 1000, Qp = ed.Qprod * 3600, secE = (Ped + eFO) / Qp;
  return {
    summary: `FO transfers ${fmt(Vw, 4)} m³/h into the draw and electrodialysis splits the diluted draw into ${fmt(Qp, 4)} m³/h of product at ${fmt(ed.tdsP, 4)} mg/L and regenerated draw, in ${ed.nSt} stage${ed.nSt > 1 ? 's' : ''} of ${ed.Ncp} cell pairs at ${fmt(secE, 3)} kWh/m³.`,
    warnings: W.length ? W : [{ level: 'info', msg: 'Electrodialysis–FO chain solved; balances close.' }],
    kpis: [{ label: 'Product (ED diluate)', value: Qp, unit: 'm³/h' }, { label: 'Product TDS', value: ed.tdsP, unit: 'mg/L', status: ed.reached ? 'ok' : 'bad' }, { label: 'Overall recovery', value: (100 * Qp) / p.Qf, unit: '%' }, ...foK, { label: 'Combined specific energy', value: secE, unit: 'kWh/m³' }, { label: 'ED specific energy', value: ed.sec, unit: 'kWh/m³' }, { label: 'ED stages × cell pairs', value: `${ed.nSt} × ${ed.Ncp}` }, { label: 'ED membrane area', value: ed.area, unit: 'm²' }, { label: 'ED current efficiency', value: 100 * ed.eff, unit: '%' }, { label: 'Regenerated draw strength', value: ed.tdsC / 1000, unit: 'g/L' }],
    recommendations: ['Electrodialysis–FO pays off with dilute draws (below about 0.3 mol/L) and fouling feeds that would foul an ED stack directly.', 'Check the stack design in suite 7 with the diluted draw as feed.'],
    plots: clean([{ type: 'bar', title: 'Membrane area and electricity of each process', ylabel: 'see categories', categories: ['Membrane area (m² ÷ 10)', 'Electricity (kW)'], series: [{ name: 'Forward osmosis', values: [r.area / 10, eFO] }, { name: 'Electrodialysis', values: [ed.area / 10, Ped] }] }, foPlot,
      { type: 'line', title: 'Electrodialysis of the diluted draw: stage by stage', xlabel: 'Stage', ylabel: 'mg/L · A/m²', logy: true, series: [{ name: 'Diluate leaving the stage (mg/L)', x: ed.stages.map((s) => s.n), y: ed.stages.map((s) => Math.max(s.tdsOut, 1e-6)), mode: 'both' }, { name: 'Mean current density (A/m²)', x: ed.stages.map((s) => s.n), y: ed.stages.map((s) => Math.max(s.iAvg, 1e-6)), mode: 'both' }] }]),
    tables: [{ title: 'Contribution of each process', columns: ['Process', 'Role', 'Water handled (m³/h)', 'Membrane area (m²)', 'Electricity (kW)', 'Specific energy (kWh per m³ product)'], rows: [['Forward osmosis', `Draws ${fmt(100 * r.recovery, 3)} % of the feed into the draw at ${fmt(r.JwLMH, 3)} L/m²·h`, Vw, r.area, eFO, eFO / Qp], ['Electrodialysis', `Desalts the diluted draw (${fmt(tds(ionsD) / 1000, 3)} g/L) to ${fmt(ed.tdsP, 3)} mg/L at ${fmt(100 * ed.eff, 3)} % current efficiency`, Qp, ed.area, Ped, Ped / Qp], ['Hybrid total', `Overall recovery ${fmt((100 * Qp) / p.Qf, 3)} %`, Qp, r.area + ed.area, eFO + Ped, secE]], note: 'The ED diluate is the product and the ED concentrate returns to the FO as regenerated draw.' },
      { title: 'Electrodialysis stages', columns: ['Stage', 'Diluate in (mg/L)', 'Diluate out (mg/L)', 'V per cell pair', 'Mean i (A/m²)', 'Max i / i_lim (%)', 'DC power (kW)'], rows: ed.stages.map((s) => [s.n, s.tdsIn, s.tdsOut, s.U, s.iAvg, 100 * s.ratioMax, s.P / 1000]) }],
    balances: [...foBal, { name: 'ED water (m³/h)', in: ed.tr.Qf * 3600, out: (ed.tr.Qp + ed.tr.Qbd) * 3600 }],
    outputs: { flux: r.JwLMH, secThermal: 0, secElec: secE, area: r.area + ed.area, recovery: Qp / p.Qf, power: eFO + Ped, process: 'ed-fo', streams: { product: stream(Qp, T, 7, ed.ionsP), concentrate: stream(r.feedOut.Q * 3600, T, p.pH, feedOutIons) } },
  };
}

function runMDC(v) {
  const p = v, W = [], r = simulateMDC(p), m = r.m, cr = r.cr, Qp = (r.prod * 3600) / density(25, 0), hs = heatSource(p, r.Qheat / 1000, p.Tf, W), margin = r.lep / 1e5 - p.pFeed;
  if (!(r.solids > 0)) W.push({ level: 'bad', msg: `No crystals form: the bleed of ${p.bleed} % carries away all the salt and keeps the loop at ${fmt(r.Sloop, 4)} g/kg, below saturation. Lower the bleed.` });
  if (r.wallSat >= 1) W.push({ level: 'warn', msg: `The membrane surface is ${fmt(100 * (r.wallSat - 1), 3)} % above saturation at the feed temperature: crystals will also grow on the membrane. Raise the cross-flow velocity or the temperature difference between loop and crystalliser.` });
  if (margin < 1) W.push({ level: margin < 0 ? 'bad' : 'warn', msg: `Wetting margin is ${fmt(margin, 2)} bar (liquid-entry pressure ${fmt(r.lep / 1e5, 3)} bar); saturated brine and crystals promote wetting.` });
  if (cr.L43 < 1e-4) W.push({ level: 'info', msg: `Mass-mean crystal size is only ${fmt(cr.L43 * 1e6, 3)} µm — a longer residence time gives coarser, easier-to-dewater crystals.` });
  if (!W.length) W.push({ level: 'info', msg: 'MD–crystalliser solved; water, salt and population balances close.' });
  const Ls = linspace(0, 10 * cr.L0, 41), taus = [0.5, 1, 2, 3, 4, 6], ssum = Math.max(r.solids, 0);
  const tables = [{ title: 'Contribution of each unit', columns: ['Unit', 'Role', 'Throughput', 'Unit of throughput', 'Size', 'Unit of size', 'Heat (kW)', 'Electricity (kW)'], rows: [['Membrane distillation', `Evaporates the water at ${fmt(m.fluxLMH, 3)} L/m²·h from saturated brine`, Qp, 'm³/h distillate', r.area, 'm² membrane', r.Qheat / 1000, (m.Wpump * r.width) / 1000], ['Crystalliser (MSMPR)', `Grows crystals of ${fmt(cr.L43 * 1e6, 3)} µm at ${fmt(100 * cr.sigma, 3)} % supersaturation`, ssum * 86.4, 't/d solids', r.Vcr, 'm³ suspension', 0, 0.5 * Qp], ['Bleed', 'Purges impurities', (r.bleed * 3600) / density(25, r.Sloop), 'm³/h', null, '', 0, 0]], note: 'Total salinity is treated as NaCl-equivalent; 0.5 kWh per m³ of distillate is allowed for slurry circulation and solids separation.' },
    { title: 'Crystalliser', columns: ['Quantity', 'Value', 'Unit'], rows: [['Solubility at the crystalliser temperature', r.Ssat, 'g/kg'], ['Relative supersaturation', 100 * cr.sigma, '%'], ['Loop salinity', r.Sloop, 'g/kg'], ['Growth rate', cr.G * 1e9, 'nm/s'], ['Nucleation rate', cr.B0, '1/m³·s'], ['Nuclei population density', cr.n0, '1/m⁴'], ['Mass-mean size L₄,₃ = 4Gτ', cr.L43 * 1e6, 'µm'], ['Dominant size 3Gτ', cr.LD * 1e6, 'µm'], ['Coefficient of variation (mass)', cr.cv, '%'], ['Suspension density', p.slurry, 'kg/m³'], ['Residence time', p.tauCr, 'h'], ['Crystalliser volume', r.Vcr, 'm³'], ['Solids production', ssum * 86.4, 't/d'], ['Wall saturation ratio in the MD module', r.wallSat, '–']] }, ...mdTables(r, p)];
  if (hs) tables.push({ title: hs.title, columns: ['Quantity', 'Value'], rows: hs.rows, note: hs.note });
  return {
    summary: `The MD–crystalliser turns ${fmt(p.Qf, 3)} m³/h of brine into ${fmt(Qp, 4)} m³/h of distillate and ${fmt(ssum * 86.4, 3)} t/d of crystals (mass-mean size ${fmt(cr.L43 * 1e6, 3)} µm) on ${fmt(r.area, 4)} m² of membrane, using ${fmt(r.sth, 4)} kWh of heat per m³.`,
    warnings: W,
    kpis: [{ label: 'Distillate', value: Qp, unit: 'm³/h' }, { label: 'Solids', value: ssum * 86.4, unit: 't/d', status: r.solids > 0 ? 'ok' : 'bad' }, { label: 'Water recovery', value: 100 * r.recOverall, unit: '%' }, { label: 'MD flux at saturation', value: m.fluxLMH, unit: 'L/m²·h' }, { label: 'Membrane area', value: r.area, unit: 'm²' }, { label: 'Specific heat', value: r.sth, unit: 'kWh/m³' }, { label: 'Specific electricity', value: r.sel, unit: 'kWh/m³' },
      { label: 'Supersaturation in the crystalliser', value: 100 * cr.sigma, unit: '%' }, { label: 'Crystal growth rate', value: cr.G * 1e9, unit: 'nm/s' }, { label: 'Mass-mean crystal size', value: cr.L43 * 1e6, unit: 'µm' }, { label: 'Crystalliser volume', value: r.Vcr, unit: 'm³' }, { label: 'Loop salinity', value: r.Sloop, unit: 'g/kg' }, { label: 'Wall saturation ratio (membrane)', value: r.wallSat, unit: '–', status: r.wallSat >= 1 ? 'warn' : 'ok' }, { label: 'Liquid-entry pressure', value: r.lep / 1e5, unit: 'bar', status: margin < 0 ? 'bad' : margin < 1 ? 'warn' : 'ok' }],
    recommendations: ['Keep the crystalliser a few kelvin colder than the MD feed so that the loop is undersaturated at the membrane.', 'Send the solids and bleed to suite 9 (ZLD) for the salt sequence and dewatering.'],
    plots: clean([{ type: 'line', title: 'Crystal size distribution (MSMPR population balance)', xlabel: 'Crystal size (µm)', ylabel: 'Mass density (kg/m³ per µm)', series: [{ name: 'Mass distribution', x: Ls.map((L) => L * 1e6), y: Ls.map((L) => cr.massDensity(L) * 1e-6) }], vlines: [{ x: cr.L43 * 1e6, label: 'mass mean' }, { x: cr.LD * 1e6, label: 'dominant' }], note: 'n(L) = n₀·exp(−L/Gτ); the area under the curve is the suspension density.' },
      { type: 'line', title: 'Effect of the crystalliser residence time', xlabel: 'Residence time (h)', ylabel: 'µm · %', series: [{ name: 'Mass-mean size (µm)', x: taus, y: taus.map((t) => msmpr({ tau: t * 3600, kg: p.kgCr * 1e-6, kb: p.kbCr * 1e8, MT: p.slurry }).L43 * 1e6), mode: 'both' }, { name: 'Supersaturation × 100 (%)', x: taus, y: taus.map((t) => 1e4 * msmpr({ tau: t * 3600, kg: p.kgCr * 1e-6, kb: p.kbCr * 1e8, MT: p.slurry }).sigma), mode: 'both' }], vlines: [{ x: p.tauCr, label: 'operating' }] }, ...mdPlots({ ...r, rec: 0.5, Sloop: r.Sloop, Sfeed: r.Sfeed }, p)]),
    tables,
    balances: [{ name: 'Water + salt (kg/s)', in: r.make, out: r.prod + r.bleed + ssum }, { name: 'Salt (kg/s)', in: (r.make * r.Sfeed) / 1000, out: ssum + (r.bleed * r.Sloop) / 1000 }, { name: 'Crystal mass: population balance vs suspension density (kg/m³)', in: p.slurry, out: 6 * cr.kv * cr.rhoC * cr.n0 * cr.L0 ** 4 }, { name: 'MD module energy (W/m)', in: m.balance.in, out: m.balance.out }],
    outputs: { flux: m.fluxLMH, secThermal: r.sth, secElec: r.sel, area: r.area, recovery: r.recOverall, solids: ssum * 86.4, crystalSize: cr.L43 * 1e6, heat: r.Qheat / 1000, process: 'md-crystalliser', streams: { product: stream(Qp, m.coldOut, 6.5, cloneIons({})), concentrate: stream((r.bleed * 3600) / density(25, r.Sloop), p.Tcr, p.pH, scaleIons(r.ions, tdsFromSalinity(r.Sloop, 25) / Math.max(r.tdsF, 1e-9))) } },
  };
}

/** Appends the always-on material comparison and the optional pore-scale, dynamic, multi-objective and grey-box modules to a result. */
function addExtras(res, v) {
  const fo = isFO(v), md = isMD(v), W = res.warnings, K = res.kpis, P = [], Tb = res.tables, B = (res.balances ||= []), O = res.outputs, sf = v.salinityFactor ?? 1;
  if (!fo && !md) return res;
  const S = salinityFromTDS(tds(v.ions) * sf, 25), cold = { T: v.Tp, pg: (v.rhGas / 100) * psat(v.Tp) }, coupon = (q) => { const c = mdConfig(q, q.Tf, q.Tp, S); return { c, lo: mdLocal(c, q.Tf, S, cold) }; }, foC = (q) => { const s = foSetup(q); return { s, f: foFlux({ f: 1, cFd: 0, cD: q.cDraw * 1000 }, s.m) }; };
  // ---- membrane-material library
  if (md) {
    const rows = [['Present membrane', v], ...Object.entries(MATERIALS).filter(([, m]) => m.kind === 'md').map(([k, m]) => [m.name, applyMaterial({ ...v, memMat: k })])].map(([name, q]) => { const { c, lo } = coupon(q); return [name, q.dPore, q.epsM, q.tauM, q.deltaM, q.kPoly, q.theta, lo.N * 3600, 100 * lo.eta, liquidEntryPressure(q.rMax * 1e-6, q.theta, surfaceTension(q.Tf, q.gammaF), q.lepB) / 1e5, lo.dg.Kn]; });
    Tb.push({ title: 'Membrane-material library at these operating conditions', columns: ['Material', 'Pore diameter (µm)', 'Porosity', 'Tortuosity', 'Thickness (µm)', 'Polymer conductivity (W/m·K)', 'Contact angle (°)', 'Flux (L/m²·h)', 'Thermal efficiency (%)', 'Liquid-entry pressure (bar)', 'Knudsen number'], rows, note: 'Module-inlet coupon at the feed salinity. Select a material on Model setup to use its properties in the whole calculation; the values are typical published data for conventional (PVDF, PTFE, PP) and novel (nanofibre, carbon-nanotube, omniphobic) membranes.' });
    P.push({ type: 'bar', title: 'Flux and thermal efficiency of the library materials', ylabel: 'L/m²·h · %', categories: rows.map((r) => r[0].split(',')[0]), series: [{ name: 'Flux (L/m²·h)', values: rows.map((r) => r[7]) }, { name: 'Thermal efficiency (%)', values: rows.map((r) => r[8]) }] });
  } else {
    const rows = [['Present membrane', v], ...Object.entries(MATERIALS).filter(([, m]) => m.kind === 'fo').map(([k, m]) => [m.name, applyMaterial({ ...v, memMat: k })])].map(([name, q]) => { const { s, f } = foC(q); return [name, q.AFO, q.BFO, q.Sfo, f.Jw * 3.6e6, f.Js * s.d.M * 3600, f.Jw > 0 ? (f.Js * s.d.M) / f.Jw / 1000 : 0]; });
    Tb.push({ title: 'Membrane-material library at these operating conditions', columns: ['Material', 'A (L/m²·h·bar)', 'B (L/m²·h)', 'S (µm)', 'Water flux (L/m²·h)', 'Reverse solute flux (g/m²·h)', 'Specific reverse solute flux (g/L)'], rows, note: 'Module-inlet coupon. Select a material on Model setup to use its properties in the whole calculation; the values are typical published data for cellulose-triacetate, thin-film-composite, aquaporin and nanocomposite membranes.' });
    P.push({ type: 'bar', title: 'Water flux of the library materials (module inlet)', ylabel: 'L/m²·h', categories: rows.map((r) => r[0].split(',')[0]), series: [{ name: 'Water flux', values: rows.map((r) => r[4]) }] });
    if ((v.hydration ?? 100) < 100 || (v.dPfeed || 0) > 0) { const dry = foC(v), full = foC({ ...v, hydration: 100, dPfeed: 0 }); K.push({ label: 'Flux ÷ flux of the fully hydrated, unpressurised membrane', value: full.f.Jw > 0 ? (100 * dry.f.Jw) / full.f.Jw : 0, unit: '%', help: `Support hydration ${v.hydration ?? 100} % (effective structural parameter ${fmt(v.Sfo / clamp((v.hydration ?? 100) / 100, 0.2, 1), 4)} µm), feed-side pressure ${v.dPfeed || 0} bar` }); }
  }
  // ---- pore-scale relations (MD)
  if (md) {
    const { c, lo } = coupon(v), sg = poreSigma(v), gamma = surfaceTension(v.Tf, v.gammaF), wf = wettedFraction(c.mem.r, sg, v.theta, gamma, v.pFeed * 1e5, v.lepB), kel = kelvinFactor(v.Tf, c.mem.r, v.theta), one = dustyGas({ ...c.mem, sg: 1 }, 0.5 * (lo.Tfm + lo.Tc), c.P, 0.5 * (lo.pF + lo.pP), c.model, c.type === 'vmd');
    const leak = (wf.area * c.mem.eps * c.mem.r ** 2 * v.pFeed * 1e5 * density(v.Tf, S)) / (8 * viscosity(v.Tf, S) * c.mem.tau * c.mem.delta), tdsLeak = lo.N + leak > 0 ? (1e3 * leak * S) / (lo.N + leak) : 0;
    Tb.push({ title: 'Pore-scale relations', columns: ['Quantity', 'Value', 'Unit'], rows: [['Kelvin vapour-pressure factor at the pore mouth', kel, '–'], ['Kelvin correction applied', v.kelvin ? 'yes' : 'no', ''], ['Geometric standard deviation of the pore sizes', sg, '–'], ['Pore-size model', v.poreDist === 'lognormal' ? 'log-normal distribution (parallel capillaries)' : v.poreDist === 'network' ? 'pore network (2-D lattice)' : 'single mean pore', ''], ['Membrane coefficient used', lo.B * 1e7, '10⁻⁷ kg/m²·s·Pa'], ['Membrane coefficient of the single mean pore', one.B * 1e7, '10⁻⁷ kg/m²·s·Pa'], ['Critical wetting radius at the feed pressure', Number.isFinite(wf.rc) ? wf.rc * 1e6 : null, 'µm'], ['Pores above the critical radius (number)', 100 * wf.number, '%'], ['Pores above the critical radius (open area)', 100 * wf.area, '%'], ['Liquid leak through wetted pores', leak * 3600, 'kg/m²·h'], ['Distillate salinity from the leak', tdsLeak, 'g/kg']], note: 'The distribution of pore sizes is log-normal around the mean pore; when only the mean pore is modelled its spread is inferred from the largest pore (taken as the 99th percentile) for the wetting estimate.' });
    if (lo.dg.classes) {
      K.push({ label: 'Membrane coefficient ÷ single-pore value', value: lo.dg.B / lo.dg.Bmean, unit: '–', help: `Log-normal pore sizes, geometric standard deviation ${fmt(sg, 3)}; ${fmt(100 * lo.dg.knudsenShare, 3)} % of the open area is in the Knudsen regime` });
      P.push({ type: 'line', title: 'Pore-size distribution and where the vapour flows', xlabel: 'Pore radius (µm)', ylabel: 'share per class (%)', logx: true, series: [{ name: 'Open area', x: lo.dg.classes.map((q) => q.r * 1e6), y: lo.dg.classes.map((q) => 100 * q.area), mode: 'both' }, { name: 'Vapour flux', x: lo.dg.classes.map((q) => q.r * 1e6), y: lo.dg.classes.map((q) => 100 * q.flux), mode: 'both' }], vlines: [{ x: c.mem.r * 1e6, label: 'mean pore' }, ...(Number.isFinite(wf.rc) && wf.rc < c.mem.r * 60 ? [{ x: wf.rc * 1e6, label: 'wetting' }] : [])], note: 'Each pore class carries the dusty-gas flux of its own Knudsen number; large pores carry more than their share of area.' });
    }
    if (lo.dg.network) {
      const nw = lo.dg.network, net = nw.net, dPf = v.pFeed * 1e5, lepMed = liquidEntryPressure(c.mem.r, v.theta, gamma, v.lepB), ps = linspace(0.3, 1.6, 14).map((f) => f * lepMed), curve = ps.map((q) => net.invade(liquidEntryPressure(1, v.theta, gamma, v.lepB) / q));
      K.push({ label: 'Network membrane coefficient ÷ single-pore value', value: lo.dg.B / lo.dg.Bmean, unit: '–', help: `Bundle of parallel capillaries: ${fmt(lo.dg.Bbundle / lo.dg.Bmean, 4)}; ${net.nx} × ${net.nz} lattice, geometric standard deviation ${fmt(sg, 3)}` },
        { label: 'Wetted throats (invasion percolation)', value: 100 * net.fraction, unit: '%', status: net.through ? 'bad' : net.fraction > 0 ? 'warn' : 'ok', help: `Liquid front reaches ${fmt(100 * net.depth, 3)} % of the thickness at ${v.pFeed} bar` },
        { label: 'Network breakthrough pressure', value: Number.isFinite(net.lepNet) ? net.lepNet / 1e5 : 0, unit: 'bar', status: net.through ? 'bad' : 'ok', help: `Pressure at which a connected path of invaded throats first spans the membrane; the widest throat alone would wet at ${fmt(liquidEntryPressure(net.rMaxThroat, v.theta, gamma, v.lepB) / 1e5, 3)} bar, the median throat at ${fmt(lepMed / 1e5, 3)} bar` });
      if (net.through) W.push({ level: 'bad', msg: `Pore network: at ${v.pFeed} bar the invaded throats form a liquid path through the whole membrane (breakthrough pressure ${fmt(net.lepNet / 1e5, 3)} bar) — the membrane wets and brine reaches the distillate.` });
      else if (net.fraction > 0) W.push({ level: 'warn', msg: `Pore network: liquid has entered ${fmt(100 * net.fraction, 3)} % of the throats from the feed side (front at ${fmt(100 * net.depth, 3)} % of the thickness) without spanning the membrane; the shorter vapour path raises the flux but the margin to breakthrough at ${fmt(net.lepNet / 1e5, 3)} bar is ${fmt((net.lepNet - dPf) / 1e5, 3)} bar.` });
      const pz = Array.from({ length: net.nz }, (_, k) => Array.from({ length: net.nx }, (_, i) => nw.p[k * net.nx + i])), fluxV = Array.from({ length: net.nz + 1 }, (_, k) => Array.from({ length: net.nx }, (_, i) => nw.gv[k * net.nx + i] * ((k === 0 ? 1 : nw.p[(k - 1) * net.nx + i]) - (k === net.nz ? 0 : nw.p[k * net.nx + i])))), fmax = Math.max(...fluxV.flat(), 1e-300), Kn = Math.max(nw.K, 1e-300);
      const order = Array.from(net.rv.keys()).sort((a, b) => net.rv[a] - net.rv[b]), nb = 8, cls = Array.from({ length: nb }, (_, q) => { const idx = order.slice(Math.floor((q * order.length) / nb), Math.floor(((q + 1) * order.length) / nb)); return { r: Math.exp(sum(idx.map((t) => Math.log(net.rv[t]))) / idx.length), area: sum(idx.map((t) => net.rv[t] ** 2)) / net.sumRv2, flux: sum(idx.map((t) => fluxV[Math.floor(t / net.nx)][t % net.nx])) / (Kn * (net.nz + 1)) }; });
      P.push({ type: 'field', title: 'Pore network: vapour pressure of the pore bodies', xlabel: 'Position along the membrane (throat spacings)', ylabel: 'Depth from the feed face (pore bodies)', zlabel: 'Vapour pressure', zunit: '(p − p_permeate)/(p_feed − p_permeate)', x: Array.from({ length: net.nx }, (_, i) => i + 1), y: Array.from({ length: net.nz }, (_, k) => k + 1), z: pz, cmap: 'thermal', contours: 6, note: `Linear conductance network of ${net.rv.length + net.rh.length} throats (seed ${net.seed}); wetted bodies sit at the feed vapour pressure. Flow through every cut plane below the liquid front: ${fmt(Math.min(...nw.layers.slice(Math.min(net.nz, Math.ceil(net.depth * net.nz)))) / Kn, 6)}–${fmt(Math.max(...nw.layers.slice(Math.min(net.nz, Math.ceil(net.depth * net.nz)))) / Kn, 6)} of the total.` },
        { type: 'field', title: 'Pore network: vapour flow through the through-plane throats', xlabel: 'Position along the membrane (throat spacings)', ylabel: 'Throat layer from the feed face', zlabel: 'Flow', zunit: '÷ largest throat flow', x: Array.from({ length: net.nx }, (_, i) => i + 1), y: Array.from({ length: net.nz + 1 }, (_, k) => k + 1), z: fluxV.map((row) => row.map((q) => q / fmax)), cmap: 'viridis', note: 'Preferential paths through connected wide throats; narrow throats in series throttle whole columns.' },
        { type: 'line', title: 'Pore network: where the vapour flows, and wetting by invasion percolation', xlabel: 'Throat radius (µm)  ·  or feed pressure ÷ 10 (bar)', ylabel: 'share per class (%)  ·  wetted throats (%)', series: [{ name: 'Open area per radius class', x: cls.map((q) => q.r * 1e6), y: cls.map((q) => 100 * q.area), mode: 'both' }, { name: 'Vapour flow per radius class', x: cls.map((q) => q.r * 1e6), y: cls.map((q) => 100 * q.flux), mode: 'both' }, { name: 'Wetted throats versus feed pressure (x = bar ÷ 10)', x: ps.map((q) => q / 1e6), y: curve.map((q) => 100 * q.fraction), mode: 'both' }], vlines: Number.isFinite(net.lepNet) ? [{ x: net.lepNet / 1e6, label: 'breakthrough' }] : [], note: 'Unlike the bundle of parallel capillaries, the widest throats do not carry a flux in proportion to their conductance: each is in series with its neighbours.' });
      Tb.push({ title: 'Pore network compared with the continuum models', columns: ['Model', 'Membrane coefficient (10⁻⁷ kg/m²·s·Pa)', '÷ single mean pore'], rows: [['Single mean pore (dusty gas)', lo.dg.Bmean * 1e7, 1], ['Bundle of parallel capillaries (log-normal classes)', lo.dg.Bbundle * 1e7, lo.dg.Bbundle / lo.dg.Bmean], ['Pore network (used)', lo.dg.B * 1e7, lo.dg.B / lo.dg.Bmean]], note: `Same porosity, tortuosity and thickness in all three. Lattice ${net.nx} × ${net.nz}, throat radii ${fmt(net.rMinThroat * 1e6, 3)}–${fmt(net.rMaxThroat * 1e6, 3)} µm (median ${fmt(c.mem.r * 1e6, 3)} µm).` },
        { title: 'Pore network: wetting by invasion percolation', columns: ['Feed pressure (bar)', 'Wetted throats (%)', 'Wetted pore bodies (%)', 'Liquid front (% of thickness)', 'Breakthrough'], rows: ps.map((q, k) => [q / 1e5, 100 * curve[k].fraction, 100 * curve[k].bodies, 100 * curve[k].depth, curve[k].through ? 'yes' : 'no']), note: `Operating feed pressure ${v.pFeed} bar: ${fmt(100 * net.fraction, 3)} % of the throats wetted. A throat is invaded only if it is wider than the critical radius and touches liquid already, so isolated wide pores inside the membrane stay dry — the analytic distribution estimate above counts them as wetted.` });
      O.networkCoefficientRatio = lo.dg.B / lo.dg.Bmean; O.networkWettedFraction = net.fraction; O.networkBreakthroughBar = Number.isFinite(net.lepNet) ? net.lepNet / 1e5 : 0;
    }
    if (wf.area > 0.001) W.push({ level: wf.area > 0.05 ? 'bad' : 'warn', msg: `${fmt(100 * wf.area, 3)} % of the pore area lies above the critical wetting radius of ${fmt(wf.rc * 1e6, 3)} µm at ${v.pFeed} bar: brine leaks through and the distillate reaches about ${fmt(tdsLeak, 3)} g/kg.` });
    O.wettedPoreArea = wf.area; O.kelvinFactor = kel;
  }
  // ---- dynamic operation
  if (v.dynamic && v.process === 'md') try {
    const r = simulateMD(v), dy = dynamicMD(v, r), ts = dy.rows.map((q) => q.t);
    K.push({ label: 'Flux decline over the batch', value: 100 * dy.decline, unit: '%', help: `${fmt(dy.flux0, 3)} → ${fmt(dy.fluxEnd, 3)} L/m²·h in ${v.tDyn} h` }, { label: 'Batch recovery', value: 100 * dy.recovery, unit: '%', help: `Concentration factor ${fmt(dy.cf, 3)}` }, { label: 'Wetted pore area at the end', value: 100 * dy.rows[dy.rows.length - 1].xw, unit: '%', status: dy.rows[dy.rows.length - 1].xw > 0.01 ? 'warn' : 'ok' }, { label: 'Distillate salinity (batch average)', value: dy.tdsMix, unit: 'mg/kg' });
    if (dy.tWet !== null) W.push({ level: 'warn', msg: `Pore wetting passes 1 % of the pore area after ${fmt(dy.tWet, 3)} h of operation as deposits lower the contact angle.` });
    if (dy.tSat !== null) W.push({ level: 'info', msg: `The membrane surface reaches saturation after ${fmt(dy.tSat, 3)} h: scale starts to grow and block pores (crystallisation regime).` });
    P.push({ type: 'line', title: 'Batch operation: flux and tank salinity', xlabel: 'Time (h)', ylabel: 'L/m²·h · g/kg', series: [{ name: 'Vapour flux (L/m²·h)', x: ts, y: dy.rows.map((q) => q.N * 3600) }, { name: 'Tank salinity ÷ 10 (g/kg)', x: ts, y: dy.rows.map((q) => q.S / 10) }, { name: 'Wall saturation ratio × 10', x: ts, y: dy.rows.map((q) => 10 * q.omega) }], hlines: [{ y: 10, label: 'saturation' }] },
      { type: 'line', title: 'Fouling, scaling and wetting during the batch', xlabel: 'Time (h)', ylabel: 'see legend', series: [{ name: 'Organic / colloidal deposit (g/m²)', x: ts, y: dy.rows.map((q) => q.md) }, { name: 'Mineral scale (g/m²)', x: ts, y: dy.rows.map((q) => q.ms) }, { name: 'Contact angle ÷ 10 (°)', x: ts, y: dy.rows.map((q) => q.theta / 10) }, { name: 'Wetted pore area (%)', x: ts, y: dy.rows.map((q) => 100 * q.xw) }, { name: 'Distillate salinity (g/kg)', x: ts, y: dy.tdsP }], note: `${fmt(dy.A, 4)} m² of membrane on a ${v.batchVol} m³ tank; module flux = ${fmt(dy.fmod, 3)} × inlet coupon flux.` });
    Tb.push({ title: 'Dynamic batch operation', columns: ['Time (h)', 'Tank mass (t)', 'Salinity (g/kg)', 'Flux (L/m²·h)', 'Deposit (g/m²)', 'Scale (g/m²)', 'Contact angle (°)', 'Wetted pore area (%)', 'Wall saturation ratio', 'Distillate salinity (g/kg)'], rows: dy.rows.filter((_, k) => k % Math.ceil(dy.rows.length / 25) === 0 || k === dy.rows.length - 1).map((q, k, a) => [q.t, q.M / 1000, q.S, q.N * 3600, q.md, q.ms, q.theta, 100 * q.xw, q.omega, dy.tdsP[dy.rows.indexOf(q)]]) });
    B.push({ name: 'Batch tank water + salt (t)', in: dy.waterBal.in / 1000, out: dy.waterBal.out / 1000 }, { name: 'Batch tank salt (t)', in: dy.saltBal.in / 1000, out: dy.saltBal.out / 1000 });
    O.dynamicFluxDecline = dy.decline; O.batchRecovery = dy.recovery;
  } catch (e) { W.push({ level: 'warn', msg: `The dynamic batch simulation failed: ${e.message}` }); }
  if (v.dynamic && v.process === 'fo') try {
    const dy = dynamicFO(v), cl = dynamicFO(v, { cFou: 0 }), ts = dy.rows.map((q) => q.t);
    K.push({ label: 'Batch flux, start → end', value: dy.fluxEnd, unit: 'L/m²·h', help: `Started at ${fmt(dy.flux0, 3)} L/m²·h; without fouling the batch would end at ${fmt(cl.fluxEnd, 3)} L/m²·h` }, { label: 'Batch feed recovery', value: 100 * dy.recovery, unit: '%', help: `${fmt(100 * cl.recovery, 3)} % without fouling` }, { label: 'Draw concentration at the end', value: dy.cDend, unit: 'mol/L' }, { label: 'Draw solute lost to the feed', value: dy.soluteLoss, unit: 'kg' });
    P.push({ type: 'line', title: 'Batch operation: flux, recovery and draw strength', xlabel: 'Time (h)', ylabel: 'see legend', series: [{ name: 'Water flux (L/m²·h)', x: ts, y: dy.rows.map((q) => q.fl.Jw * 3.6e6) }, { name: 'Water flux without fouling (L/m²·h)', x: cl.rows.map((q) => q.t), y: cl.rows.map((q) => q.fl.Jw * 3.6e6), dash: true }, { name: 'Feed recovery (% ÷ 10)', x: ts, y: dy.rows.map((q) => 10 * (1 - q.y[0] / dy.VF0)) }, { name: 'Draw concentration × 10 (mol/L)', x: ts, y: dy.rows.map((q) => q.st.cD / 100) }] },
      { type: 'line', title: 'Cake resistance and support-layer hydration during the batch', xlabel: 'Time (h)', ylabel: 'see legend', series: [{ name: 'Cake resistance (10¹³ 1/m)', x: ts, y: dy.rows.map((q) => q.Rc / 1e13) }, { name: 'Support hydration (%  ÷ 10)', x: ts, y: dy.rows.map((q) => 10 * q.h) }], note: 'The cake adds a hydraulic resistance in series with the active layer; a partly wetted support has a larger effective structural parameter until it wets out.' });
    Tb.push({ title: 'Dynamic batch operation', columns: ['Time (h)', 'Feed tank (m³)', 'Draw tank (m³)', 'Draw (mol/L)', 'Feed concentration factor', 'Flux (L/m²·h)', 'Cake resistance (10¹³ 1/m)', 'Support hydration (%)'], rows: dy.rows.filter((_, k) => k % Math.ceil(dy.rows.length / 25) === 0 || k === dy.rows.length - 1).map((q) => [q.t, q.y[0], q.y[1], q.st.cD / 1000, q.st.f, q.fl.Jw * 3.6e6, q.Rc / 1e13, 100 * q.h]) });
    B.push({ name: 'Batch water, feed + draw tanks (m³)', in: dy.waterBal.in, out: dy.waterBal.out }, { name: 'Batch draw solute (mol)', in: dy.soluteBal.in, out: dy.soluteBal.out });
    O.dynamicFluxEnd = dy.fluxEnd; O.batchRecovery = dy.recovery;
  } catch (e) { W.push({ level: 'warn', msg: `The dynamic batch simulation failed: ${e.message}` }); }
  // ---- multi-objective sweep
  if (v.pareto && (v.process === 'md' || v.process === 'fo')) {
    const po = paretoDesigns(v, S, md), pts = po.all;
    if (pts.length > 2) {
      const pf = paretoFront(pts, (q) => q.flux, (q) => q.en), kn = pf.knee, unit = md ? 'kWh heat per m³' : 'kWh-equivalent per m³';
      K.push({ label: 'Pareto-optimal designs', value: pf.front.length, unit: `of ${pts.length}`, help: `${po.grid.length} grid designs (backdrop) + ${po.ga.length} designs of the NSGA-II search (${po.evals} model evaluations, seed ${po.seed}); hypervolume ${fmt(po.hvGrid, 4)} → ${fmt(po.hvAll, 4)}` }, { label: 'Knee of the Pareto front: flux', value: kn.flux, unit: 'L/m²·h', help: md ? `Hot feed ${fmt(kn.a, 3)} °C, velocity ${fmt(kn.b, 3)} m/s, channel length ${fmt(kn.c, 3)} m` : `Draw ${fmt(kn.a, 3)} mol/L, velocity ${fmt(kn.b, 3)} cm/s` }, { label: 'Knee of the Pareto front: energy', value: kn.en, unit: unit });
      P.push({ type: 'line', title: 'Multi-objective sweep: flux versus specific energy', xlabel: 'Average flux (L/m²·h)', ylabel: md ? 'Specific heat (kWh/m³)' : 'Specific energy (kWh-equivalent/m³)', logy: true, series: [{ name: 'Grid designs (backdrop)', x: po.grid.map((q) => q.flux), y: po.grid.map((q) => Math.max(q.en, 1e-6)), mode: 'points' }, { name: 'NSGA-II population', x: po.ga.map((q) => q.flux), y: po.ga.map((q) => Math.max(q.en, 1e-6)), mode: 'points' }, { name: 'Pareto front', x: pf.front.map((q) => q.flux), y: pf.front.map((q) => Math.max(q.en, 1e-6)), mode: 'both' }, { name: 'Knee', x: [kn.flux], y: [Math.max(kn.en, 1e-6)], mode: 'points' }], note: (md ? 'Feed temperature × velocity × channel length, each a full module solution; a design is Pareto-optimal when no other design has both more flux and less heat demand.' : 'Draw concentration × cross-flow velocity, each a full module solution with its regeneration energy (heat counted at 10 % of its value as electricity).') + ` Front: elitist non-dominated sorting genetic algorithm (NSGA-II) over the continuous design variables, merged with the grid; the grid alone gives ${po.gridFront} front designs.` });
      Tb.push({ title: 'Pareto-optimal designs', columns: md ? ['Hot-feed temperature (°C)', 'Velocity (m/s)', 'Channel length (m)', 'Flux (L/m²·h)', 'Specific heat (kWh/m³)', 'Specific electricity (kWh/m³)'] : ['Draw concentration (mol/L)', 'Velocity (cm/s)', 'Draw dilution (×)', 'Flux (L/m²·h)', 'Specific energy (kWh-eq/m³)', 'Specific reverse solute flux (g/L)'], rows: pf.front.map((q) => [q.a, q.b, q.c, q.flux, q.en, q.aux]) });
      O.pareto = pf.front.map((q) => ({ flux: q.flux, energy: q.en })); O.paretoKneeFlux = kn.flux;
    } else W.push({ level: 'warn', msg: 'Too few feasible designs for a Pareto front.' });
  }
  // ---- grey-box correction
  if (v.greybox && (v.process === 'md' || v.process === 'fo')) try {
    const num = (r, ks) => ks.every((k) => Number.isFinite(+r?.[k]) && +r[k] > 0);
    const gb = md ? greyBox((v.gbMD || []).filter((r) => num(r, ['Tf', 'uFm', 'Jw'])).map((r) => ({ Tf: +r.Tf, uFm: +r.uFm, Jw: +r.Jw })), (q) => coupon({ ...v, Tf: q.Tf, uFm: q.uFm, uPm: q.uFm }).lo.N * 3600, (q) => [(q.Tf - 60) / 20, Math.log(q.uFm / 0.25)])
      : greyBox((v.gbFO || []).filter((r) => num(r, ['cDraw', 'uF', 'Jw'])).map((r) => ({ cDraw: +r.cDraw, uF: +r.uF, Jw: +r.Jw })), (q) => foC({ ...v, cDraw: q.cDraw, uF: q.uF, uD: q.uF }).f.Jw * 3.6e6, (q) => [Math.log(q.cDraw), Math.log(q.uF / 15)]);
    const now = md ? { Tf: v.Tf, uFm: v.uFm } : { cDraw: v.cDraw, uF: v.uF }, mNow = md ? coupon(v).lo.N * 3600 : foC(v).f.Jw * 3.6e6, fac = gb.factor(now), hi = Math.max(...gb.meas, ...gb.mech);
    K.push({ label: 'Grey-box corrected flux (module inlet)', value: mNow * fac, unit: 'L/m²·h', help: `Mechanistic model ${fmt(mNow, 4)} L/m²·h × data-driven factor ${fmt(fac, 4)}` }, { label: 'Flux error, grey-box (leave-one-out)', value: gb.rmseLoo, unit: 'L/m²·h', status: gb.rmseLoo > gb.rmseMech ? 'warn' : 'ok', help: `Mechanistic model alone: ${fmt(gb.rmseMech, 3)} L/m²·h on ${gb.n} test rows` });
    if (gb.rmseLoo > gb.rmseMech) W.push({ level: 'info', msg: 'The grey-box correction does not generalise better than the mechanistic model on these data (leave-one-out error is larger) — calibrate the physical parameters instead.' });
    P.push({ type: 'line', title: 'Grey-box model: parity with the test data', xlabel: 'Measured flux (L/m²·h)', ylabel: 'Predicted flux (L/m²·h)', series: [{ name: 'Mechanistic model', x: gb.meas, y: gb.mech, mode: 'points' }, { name: 'Grey-box, leave-one-out', x: gb.meas, y: gb.loo, mode: 'points' }, { name: '1 : 1', x: [0, hi], y: [0, hi], dash: true }], note: 'ln(J_measured / J_model) = β₀ + β₁x₁ + β₂x₂ by ridge regression: the physics fixes the structure and the data shift it.' });
    Tb.push({ title: 'Grey-box (physics-informed) correction', columns: ['Quantity', 'Value', 'Unit'], rows: [['Test rows used', gb.n, ''], ['β₀ (overall bias, ln)', gb.beta[0], '–'], [md ? 'β₁ per 20 K of feed temperature' : 'β₁ per ln(draw concentration)', gb.beta[1], '–'], ['β₂ per ln(velocity)', gb.beta[2], '–'], ['RMSE of the mechanistic model', gb.rmseMech, 'L/m²·h'], ['RMSE of the grey-box fit', gb.rmseFit, 'L/m²·h'], ['RMSE of the grey-box, leave-one-out', gb.rmseLoo, 'L/m²·h'], ['Correction factor at the operating point', fac, '–'], ['Corrected inlet flux', mNow * fac, 'L/m²·h']], note: 'The correction is bounded to a factor of two either way. Edit the test data on Model setup.' });
    O.greyBoxFactor = fac;
  } catch (e) { W.push({ level: 'warn', msg: `The grey-box correction could not be fitted: ${e.message}` }); }
  res.plots = [...res.plots, ...clean(P)];
  return res;
}

/** Synthetic FO coupon data: the model with slightly different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([cDraw, salinityFactor, uF]) => {
    const m = suite.calibration.model({ ...d, AFO: 1.85, BFO: 0.56, Sfo: 520, cDraw, salinityFactor, uF });
    return { cDraw, salinityFactor, uF, Jw: +(m.Jw * (1 + g.normal(0, 0.012))).toFixed(2), aux: +(m.aux * (1 + g.normal(0, 0.025))).toFixed(2) };
  });
}

export default suite;
