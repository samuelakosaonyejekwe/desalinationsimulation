// Suite 8 — Forward osmosis, membrane distillation and emerging desalination processes.
// FO / PRO: solution–diffusion with internal and external concentration polarisation in both membrane
// orientations, reverse solute flux and a 1-D module model. MD (DCMD, AGMD, VMD, SGMD): coupled heat and
// mass transfer with the dusty-gas model, temperature and concentration polarisation, liquid-entry pressure
// and a 1-D module model. Hybrids: FO–RO and RO–MD chains; humidification–dehumidification as a
// reduced-order emerging process. (Capacitive deionisation is part of suite 7.)
import { brent, solve1, newtonN, clamp, linspace, interp1, sum, rng, fmt } from '../core/num.js';
import { R, KELVIN, density, viscosity, cp, conductivityThermal, psat, tsat, psatSeawater, antoine, latentHeat, enthalpyLiquid as hL, diffusivityNaCl, salinityFromTDS, tdsFromSalinity } from '../core/props.js';
import { ION_IDS, WATERS, cloneIons, tds, scaleIons, osmoticPressureIons } from '../core/water.js';
import roSuite, { simulateRO } from './s01_ro.js';

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
  const lo = top * 1e-9, Jw = at(lo).res <= 0 ? lo : brent((x) => at(x).res, lo, top, 1e-12 * top, 100);
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
  const mF = massTransfer(p.uF / 100, h, T, S, Df, p.kcp), mD = massTransfer(p.uD / 100, h, T, 40, Dd, p.kcp), kFd = mF.k * (Dd / Df) ** 0.75, Sp = p.Sfo * 1e-6, pro = p.process === 'pro', alds = pro || p.orient === 'alds';
  const m = { A: (p.AFO * tc) / 3.6e6 / 1e5, Bf: (p.BFO * tc) / 3.6e6, Bd: (p.BFO * tc * d.bRel) / 3.6e6, piF, piD: (c) => drawOsmotic(p.draw, c, T), dP: pro ? p.dPpro * 1e5 : 0,
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
  const dpF = s.mF.dpPerM * p.Lfo, dpD = s.mD.dpPerM * p.Lfo, Ppump = (QF0 * dpF + QD0 * dpD) / (p.etaPump / 100);
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
  const type = c.type, P = c.P, kel = c.kelvin ? Math.exp((2 * surfaceTension(Tf) * MW_W * Math.abs(Math.cos((c.theta * Math.PI) / 180))) / (density(Tf, 0) * c.mem.r * R * K(Tf))) : 1;
  const pw = c.antoine ? antoine : psat;
  // fz = { N, dg } frozen flux (for concentration polarisation) and dusty-gas coefficients; null → evaluate them here
  const side = (Tfm, fz) => {
    const Sm = Math.min(S * Math.exp(clamp((fz ? fz.N : 0) / (c.rhoF * c.kMass), -3, 3)), 350), pF = (c.antoine ? antoine(Tfm) * (psatSeawater(Tfm, Sm) / psat(Tfm)) : psatSeawater(Tfm, Sm)) * kel; // pure-water curve × activity of the brine
    let Tc, pP, B, qc, dg = fz ? fz.dg : null;
    if (type === 'vmd') { Tc = tsat(c.Pv); pP = c.Pv; dg ||= dustyGas(c.mem, Tfm, 0.5 * (pF + c.Pv), 0.5 * (pF + c.Pv), c.model, true); B = dg.B; qc = 0; }
    else if (type === 'agmd') {
      Tc = cold.T + (c.hf / c.hc) * (Tf - Tfm); pP = pw(Tc);
      const Tm = 0.5 * (Tfm + Tc), pm = 0.5 * (pF + pP); dg ||= dustyGas(c.mem, Tm, P, pm, c.model);
      const Bgap = (1.895e-5 * K(Tm) ** 2.072 * MW_W) / (Math.max(P - pm, 0.02 * P) * R * K(Tm) * c.gap);
      B = 1 / (1 / dg.B + 1 / Bgap); qc = (Tfm - Tc) / (1 / c.hm + c.gap / 0.027);
    } else if (type === 'sgmd') { Tc = (c.hm * Tfm + c.hg * cold.T) / (c.hm + c.hg); pP = cold.pg; dg ||= dustyGas(c.mem, 0.5 * (Tfm + Tc), P, 0.5 * (pF + pP), c.model); B = 1 / (1 / dg.B + 1 / c.Bg); qc = c.hm * (Tfm - Tc); }
    else { Tc = cold.T + (c.hf / c.hp) * (Tf - Tfm); pP = pw(Tc); dg ||= dustyGas(c.mem, 0.5 * (Tfm + Tc), P, 0.5 * (pF + pP), c.model); B = dg.B; qc = c.hm * (Tfm - Tc); }
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
  const mem = { r: (p.dPore * 1e-6) / 2, eps: p.epsM, tau: p.tauM, delta: p.deltaM * 1e-6 }, hF = p.hF / 1000, L = p.Lmd, opt = { spacer: !!p.spacerMD, nuA: p.nuA, fh: p.fh };
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

// ---- shared helpers ---------------------------------------------------------------------------------------------
const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const round = (o) => Object.fromEntries(ION_IDS.map((k) => [k, +(+o[k] || 0).toPrecision(6)]));
const stream = (Q, T, pH, ions) => ({ Q, T, P: 1, pH, tds: tds(ions), ions: round(cloneIons(ions)) });
const isFO = (v) => v.process === 'fo' || v.process === 'pro' || v.process === 'fo_ro', isMD = (v) => v.process === 'md' || v.process === 'ro_md', needsHeat = (v) => isMD(v) || v.process === 'hdh';
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

const suite = {
  id: 'fomd', num: 8, title: 'Forward Osmosis, Membrane Distillation & Emerging Desalination', short: 'FO · MD', icon: '🧪',
  tagline: 'Osmotically and thermally driven membrane processes with polarisation, wetting and energy analysis, plus FO–RO and RO–MD hybrids.',
  description: 'Forward osmosis and pressure-retarded osmosis are solved with the solution–diffusion model including internal and external concentration polarisation, reverse solute flux and a co-/counter-current module model with a draw-solute library and regeneration energy. Membrane distillation (direct-contact, air-gap, vacuum and sweeping-gas) couples the dusty-gas vapour transport to the heat balance with temperature and concentration polarisation, conductive loss, liquid-entry pressure and a module model with heat recovery. Chained FO–RO and RO–MD hybrids report each process contribution, and a humidification–dehumidification cycle is included as a reduced-order emerging process.',
  guide: [
    'Choose the process. Forward osmosis needs a feed, a draw solution and a membrane; membrane distillation needs hot-feed and coolant temperatures and a hydrophobic membrane.',
    'Enter the feed analysis and flow (or pull the case feed, or an RO concentrate for brine concentration by MD).',
    'Adjust membrane properties and transport correlations on Model setup; calibrate A, B and S (FO) or tortuosity and heat-transfer multiplier (MD) against test data.',
    'Run. Check the polarisation losses, the wetting margin (MD) and the regeneration or heat demand; for hybrids read the per-process contribution table.',
  ],
  implemented: ['solution-diffusion', 'water-flux', 'solute-flux', 'vant hoff', 'external concentration-polarization', 'internal concentration-polarization', 'structural-parameter', 'mass-transfer film', 'knudsen-diffusion', 'molecular-diffusion', 'knudsen-molecular transition', 'dusty-gas', 'vapour-pressure', 'antoine', 'kelvin', 'heat-conduction', 'convective heat-transfer', 'latent-heat balance', 'temperature-polarization',
    'fo-ro', 'ro-md', 'solar-md', 'heat-and-mass-transfer md', 'osmotic-hydraulic coupled',
    'feed/draw concentration', 'temperatures', 'pressures', 'pore vapour state', 'feed/draw inlet', 'osmotic membrane-interface', 'vapour-liquid equilibrium', 'membrane heat/mass-flux continuity', 'convective thermal boundar', 'insulated boundar',
    'membrane transport', 'draw-solution modelling', 'osmotic-property calculation', 'internal and external concentration polarisation', 'reverse-solute flux', 'porous-membrane transport', 'heat transfer', 'mass transfer', 'vapour transport', 'temperature polarisation', 'membrane wetting', 'scaling', 'module hydrodynamics', 'solar and waste-heat integration', 'hybrid-process configuration', 'energy analysis', 'sensitivity analysis'],
  equationsNote: 'FO/PRO: steady solution–diffusion with film-theory ECP and a support-layer ICP described by the structural parameter; draw osmotic coefficients are quadratic fits at 25 °C and the feed solutes are lumped into one species with the permeability of NaCl. MD: one-dimensional film model with the dusty-gas membrane coefficient (Knudsen, molecular, their series combination, plus Poiseuille flow under vacuum), pore-size distribution represented by the mean pore diameter; the vapour-pressure lowering of the brine follows the seawater correlation of the property library and the Kelvin correction is optional (below 1 % for 0.1–0.5 µm pores). Heat-transfer coefficients come from a spacer or open-channel Nusselt correlation. Scaling and crystallisation are screened by NaCl saturation and a gypsum ratio scaled from seawater — use suite 2 for speciation. Humidification–dehumidification is an effectiveness-based cycle model. Dynamic operation, fouling kinetics and pore-scale simulations are not included; capacitive deionisation is solved in suite 7.',

  inputs: [
    { group: 'Process and feed', help: 'Which process is solved and what water it treats.', fields: [
      { key: 'process', label: 'Process', type: 'select', value: 'fo', options: [{ value: 'fo', label: 'Forward osmosis (FO)' }, { value: 'pro', label: 'Pressure-retarded osmosis (PRO)' }, { value: 'md', label: 'Membrane distillation (MD)' }, { value: 'fo_ro', label: 'Hybrid: FO + RO draw regeneration' }, { value: 'ro_md', label: 'Hybrid: RO + MD brine concentration' }, { value: 'hdh', label: 'Humidification–dehumidification (HDH)' }], help: 'Each process shows its own inputs. The examples load realistic settings.' },
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
    { group: 'RO stage of the hybrid', showIf: (v) => v.process === 'ro_md' || v.process === 'fo_ro', help: 'The reverse-osmosis step is solved with the element-by-element model of suite 1.', fields: [
      { key: 'roRec', label: 'RO recovery', unit: '%', value: 45, min: 10, max: 85, showIf: (v) => v.process === 'ro_md', help: 'Recovery of the seawater or brackish RO ahead of the MD brine concentrator.' },
      { key: 'roFlux', label: 'RO design flux', unit: 'L/m²·h', value: 14, min: 5, max: 35, help: 'Average flux used to size the RO array.' },
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
    { group: 'Pumps', tab: 'setup', help: 'For the electrical energy of circulation.', fields: [
      { key: 'etaPump', label: 'Pump + motor efficiency', unit: '%', value: 70, min: 20, max: 92, help: 'Applied to all circulation pumps and fans.' },
    ] },
    { group: 'Discretisation', tab: 'mesh', help: 'Number of segments along the membrane in the module models.', showIf: (v) => v.process !== 'hdh', fields: [
      { key: 'nSeg', label: 'Segments along the module', unit: '', value: 24, min: 2, max: 400, step: 1, help: 'Use the sensitivity study to confirm that the result no longer depends on it.' },
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

  run(v) {
    if (v.process === 'md') return runMD(v);
    if (v.process === 'ro_md') return runROMD(v);
    if (v.process === 'hdh') return runHDH(v);
    return runFO(v);
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

  verify() {
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
    if (!d.ions) roErr = 'An RO regeneration cannot be simulated ion-by-ion for a non-ionic draw solute.';
    else {
      const ionsD = cloneIons(Object.fromEntries(Object.entries(d.ions).map(([k, m]) => [k, m * r.cDout]))), rec = clamp(r.tot.Vw / r.drawOut.Q, 0.05, 0.85);
      try { ro = simulateRO({ ...defaultsOf(roSuite), ions: ionsD, Qf: r.drawOut.Q * 3600, T, pH: 7, recovery: 100 * rec, targetFlux: p.roFlux, design: 'auto', mode: 'recovery', nSeg: 2 }); } catch (e) { roErr = e.message; }
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

/** Synthetic FO coupon data: the model with slightly different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([cDraw, salinityFactor, uF]) => {
    const m = suite.calibration.model({ ...d, AFO: 1.85, BFO: 0.56, Sfo: 520, cDraw, salinityFactor, uF });
    return { cDraw, salinityFactor, uF, Jw: +(m.Jw * (1 + g.normal(0, 0.012))).toFixed(2), aux: +(m.aux * (1 + g.normal(0, 0.025))).toFixed(2) };
  });
}

export default suite;
