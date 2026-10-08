// Suite 7 — Electrodialysis and membrane electrochemical processes.
// ED / EDR stacks resolved along the flow path and across stages, ion by ion: electroneutral Nernst–Planck
// boundary layers (analytical film solution), limiting current density, Donnan potentials, ohmic losses,
// electrode kinetics, Faraday's law with co-ion leakage, back-diffusion, shunt currents and water transport.
// Reduced-order models of bipolar-membrane ED (acid/base) and membrane capacitive deionisation are included,
// together with a 1-D Nernst–Planck–Donnan profile solver across film | membrane | film.
import { brent, clamp, linspace, logspace, sum, rng, fmt, rk4, solveLinear, tridiag, newtonN, lhs, gci, nelderMead } from '../core/num.js';
import { R, F, KELVIN, density, viscosity, cp as cpWater } from '../core/props.js';
import { IONS, ION_IDS, WATERS, cloneIons, tds, scaleIons, chargeBalance, balanceCharge } from '../core/water.js';
import { solveChannel, buildMask, yGrid } from './s04_cfd.js';
import { nnTrain, gpFit } from './s11_opt.js';

const CH = ION_IDS.filter((k) => IONS[k].z !== 0), NI = CH.length; // charged species tracked through the membranes
const Z = CH.map((k) => IONS[k].z), AZ = Z.map(Math.abs), LAM = CH.map((k) => IONS[k].lambda), DI = CH.map((k) => IONS[k].D), MW = CH.map((k) => IONS[k].mw);
const DIV = CH.map((k) => Math.abs(IONS[k].z) > 1);
const vt = (T) => (R * (T + KELVIN)) / F; // thermal voltage RT/F, V
const toMolar = (ions) => CH.map((k) => (+ions[k] || 0) / IONS[k].mw); // mg/L → mol/m³
const toIons = (c, base) => { const o = cloneIons(base || {}); CH.forEach((k, j) => (o[k] = c[j] * MW[j])); return o; };
const tdsOf = (c, neutral = 0) => sum(c.map((x, j) => x * MW[j])) + neutral;

/** Nernst (ideal membrane) potential for a concentration ratio, V. */
export const nernst = (ratio, T = 25, z = 1) => (vt(T) / z) * Math.log(ratio);
/** Donnan potential of a membrane with fixed charge X against a 1:1 solution of concentration c (Teorell–Meyer–Sievers), V. Negative for a cation-exchange membrane. */
export const donnanPotential = (c, X, T = 25) => -vt(T) * Math.asinh(X / (2 * c));

/** Bulk electrolyte properties of a composition c (mol/m³ per charged ion) at T. */
export function electrolyte(c, T) {
  let eqC = 0, eqA = 0, kap = 0, kc = 0, I = 0, dc = 0, da = 0, tot = 0;
  for (let j = 0; j < NI; j++) {
    const e = AZ[j] * c[j], g = e * LAM[j];
    kap += g; I += 0.0005 * c[j] * Z[j] * Z[j]; tot += c[j];
    if (Z[j] > 0) { eqC += e; kc += g; dc += e * DI[j]; } else { eqA += e; da += e * DI[j]; }
  }
  const sI = Math.sqrt(I), corr = 1 / (1 + (0.55 * sI) / (1 + 0.6 * sI) + 0.03 * I), fT = ((T + KELVIN) / 298.15) * (viscosity(25, 0) / viscosity(T, 0));
  const Dc = (eqC > 0 ? dc / eqC : 1.33e-9) * fT, Da = (eqA > 0 ? da / eqA : 2.03e-9) * fT;
  return { ceq: 0.5 * (eqC + eqA), eqC, eqA, kappa: Math.max(kap * 1e-4 * corr * (1 + 0.0191 * (T - 25)), 1e-9), tPlus: kap > 0 ? kc / kap : 0.4, I, Dc, Da, Ds: (2 * Dc * Da) / (Dc + Da), tot };
}

/** Spacer-channel hydraulics and mass transfer: Sh = a·Re^b·Sc^(1/3). */
export function channel(u, h, eps, T, Ds, par) {
  const rho = density(T, 0), mu = viscosity(T, 0);
  if (par.ns) { // Sherwood number and friction of the Navier–Stokes / Nernst–Planck solution at its reference state, rescaled to the local state:
    // open channel Sh ∝ (Re·Sc)^⅓ (Lévêque), Δp ∝ μu; spacer-filled channel (2-D solution) Sh ∝ Re^b·Sc^⅓, Δp ∝ ρu²·Re^−0.3
    const ns = par.ns, nRe = ns.nRe ?? 1 / 3, nDp = ns.nDp ?? 1, dh = 2 * h, Re = (rho * u * dh) / mu, Sc = mu / (rho * Ds), Sh = Math.max(ns.Sh * (Re / ns.Re) ** nRe * (Sc / ns.Sc) ** (1 / 3), ns.ShMin ?? 8.235), k = (Sh * Ds) / dh;
    return { dh, Re, Sc, Sh, k, delta: Ds / k, dpPerM: ns.dpPerM * (u / ns.U) ** nDp * (mu / ns.mu) ** (2 - nDp) };
  }
  const dh = (4 * eps) / (2 / h + ((1 - eps) * 8) / h), Re = (rho * u * dh) / mu, Sc = mu / (rho * Ds);
  const Sh = Math.max(par.shA * Math.max(Re, 1e-6) ** par.shB * Sc ** (1 / 3), 3), k = (Sh * Ds) / dh;
  return { dh, Re, Sc, Sh, k, delta: Ds / k, dpPerM: (par.kdp * 6.23 * Math.max(Re, 1) ** -0.3 * rho * u * u) / (2 * dh) };
}

const ghkWarm = { C: 0, A: 0 }; // last GHK roots (RT/F) of the cation and anion membrane: starting values of the next Newton solve
/**
 * Local cell-pair electrochemistry for given diluate/concentrate compositions.
 * Returns the limiting currents and a function U(i) with its components (V per cell pair).
 */
export function cellPair(cd, cc, u, G, par, T) {
  // a completely desalted stream keeps a trace of salt (10⁻⁹ eq/m³) so that the limiting current and the film potentials stay defined
  const el = (x) => { const e = electrolyte(x, T); return e.ceq > 1e-9 ? e : electrolyte(x.map((q, j) => Math.max(q, 1e-9 / (NI * AZ[j]))), T); };
  const d = el(cd), c = el(cc), hd = channel(u, G.h, G.eps, T, d.Ds, par), hc = channel(u, G.h, G.eps, T, c.Ds, par), V = vt(T);
  const tC = 0.5 * (1 + par.alphaC), tA = 0.5 * (1 + par.alphaA); // counter-ion transport numbers in the membranes
  // film coefficients of the analytical Nernst–Planck solution: K1 sets the slope of c, K2/K1 the potential drop
  const K = (tm, Dct, Dco) => ({ k1: tm / Dct - (1 - tm) / Dco, k2: tm / Dct + (1 - tm) / Dco });
  const kC = K(tC, d.Dc, d.Da), kA = K(tA, d.Da, d.Dc), kCc = K(tC, c.Dc, c.Da), kAc = K(tA, c.Da, c.Dc);
  const ilimC = (2 * F * d.ceq) / (hd.delta * kC.k1), ilimA = (2 * F * d.ceq) / (hd.delta * kA.k1), ilim = Math.min(ilimC, ilimA);
  const bulkD = Math.max(G.h - 2 * hd.delta, 0.2 * G.h), bulkC = Math.max(G.h - 2 * hc.delta, 0.2 * G.h);
  const Rohm = par.Rcem + par.Raem + (bulkD / d.kappa + bulkC / c.kappa) / G.shadow;
  // membrane potential: permselectivity × Nernst (default) or the Goldman–Hodgkin–Katz constant-field potential solved ion by ion for the
  // bulk compositions, with its local slope dE/d ln(ratio) carrying the concentration-polarisation correction
  let ghk = null;
  if (par.membModel === 'ghk') {
    // Newton warm starts: the root of the neighbouring state (previous call) for the bulk potentials, and the bulk root for the perturbed composition
    const pm = membranePermeabilities(cd, par), up = cc.map((x) => x * 1.02), ln = Math.log(1.02), EC = ghkPotential(pm.PC, Z, cd, cc, T, ghkWarm.C), EA = -ghkPotential(pm.PA, Z, cd, cc, T, ghkWarm.A);
    ghkWarm.C = EC / V; ghkWarm.A = -EA / V;
    ghk = { E: EC + EA, aC: (ghkPotential(pm.PC, Z, cd, up, T, ghkWarm.C) - EC) / (V * ln), aA: (-ghkPotential(pm.PA, Z, cd, up, T, ghkWarm.A) - EA) / (V * ln) };
  }
  const parts = (i) => {
    const wdC = d.ceq * (1 - i / ilimC), wdA = d.ceq * (1 - i / ilimA), wcC = c.ceq + (i * hc.delta * kCc.k1) / (2 * F), wcA = c.ceq + (i * hc.delta * kAc.k1) / (2 * F);
    const donnan = ghk ? ghk.E : V * (par.alphaC + par.alphaA) * Math.log(c.ceq / d.ceq);
    const memb = ghk ? ghk.E + V * (ghk.aC * Math.log((wcC * d.ceq) / (wdC * c.ceq)) + ghk.aA * Math.log((wcA * d.ceq) / (wdA * c.ceq))) : V * (par.alphaC * Math.log(wcC / wdC) + par.alphaA * Math.log(wcA / wdA));
    const films = V * ((kC.k2 / kC.k1) * Math.log(d.ceq / wdC) + (kA.k2 / kA.k1) * Math.log(d.ceq / wdA) + (kCc.k2 / kCc.k1) * Math.log(wcC / c.ceq) + (kAc.k2 / kAc.k1) * Math.log(wcA / c.ceq));
    return { donnan, polar: memb - donnan + films, ohmMem: i * (par.Rcem + par.Raem), ohmD: (i * bulkD) / d.kappa / G.shadow, ohmC: (i * bulkC) / c.kappa / G.shadow, wd: Math.min(wdC, wdA), wc: Math.max(wcC, wcA), U: memb + films + i * Rohm };
  };
  return { d, c, hd, hc, tC, tA, ilimC, ilimA, ilim, Rohm, parts, V, ghk, U: (i) => parts(i).U, E0: ghk ? ghk.E : V * (par.alphaC + par.alphaA) * Math.log(c.ceq / d.ceq) };
}

/** Plateau length (V per cell pair) before over-limiting conduction: empirical input, or two depleted layers at the electroconvection threshold. */
const plateauOf = (cp, par) => (par.ol ? 2 * par.ol.Vc * cp.V : par.plateau);
/**
 * Current density (A/m²) for a cell-pair voltage: film branch below the limiting current plus the over-limiting branch — empirical
 * (plateau and conductance ratio) or from the electroconvection model, where i/i_lim − 1 = slope·(V − V_c) in each depleted layer.
 */
export function currentAt(cp, Ucp, par) {
  if (!(Ucp > cp.E0)) return { i: 0, iFilm: 0, iOver: 0 };
  const top = cp.ilim * (1 - 1e-9), hi = Math.min(top, (Ucp - cp.E0) / cp.Rohm), iFilm = cp.U(hi) <= Ucp ? hi : brent((i) => cp.U(i) - Ucp, 0, hi, 1e-13 * cp.ilim, 100); // U ≥ E₀ + i·R_ohm bounds the root
  const Uol = cp.U(0.98 * cp.ilim) + plateauOf(cp, par), iOver = !(Ucp > Uol) ? 0 : par.ol ? (cp.ilim * par.ol.slope * (Ucp - Uol)) / (2 * cp.V + cp.ilim * par.ol.slope * cp.Rohm) : (par.olSlope * (Ucp - Uol)) / cp.Rohm;
  return { i: iFilm + iOver, iFilm, iOver };
}

/** Fluxes (mol/s per cell pair) out of the diluate over an area dA for the given local state. */
function fluxes(st, Ucp, G, par, T, dA) {
  const u = st.qd / (G.W * G.h * G.eps), cp = cellPair(st.cd, st.cc, u, G, par, T), cur = currentAt(cp, Ucp, par);
  const iSalt = cur.iFilm + (1 - par.fws) * cur.iOver, iWs = par.fws * cur.iOver, pr = cp.parts(Math.min(cur.iFilm, cp.ilim * (1 - 1e-9)));
  let wC = 0, wA = 0, gC = 0, gA = 0;
  const w = new Array(NI), g = new Array(NI);
  for (let j = 0; j < NI; j++) {
    const sel = DIV[j] ? (Z[j] > 0 ? par.selDivC : par.selDivA) : 1;
    w[j] = AZ[j] * LAM[j] * st.cd[j] * sel; g[j] = AZ[j] * LAM[j] * st.cc[j];
    if (Z[j] > 0) { wC += w[j]; gC += g[j]; } else { wA += w[j]; gA += g[j]; }
  }
  const jF = iSalt / F, diff = (par.PsC + par.PsA) * Math.max(0, pr.wc - pr.wd); // eq/m²·s
  const dn = new Array(NI);
  for (let j = 0; j < NI; j++) {
    const cat = Z[j] > 0, out = (cat ? cp.tC : cp.tA) * jF * (w[j] / ((cat ? wC : wA) || 1)); // counter-ion migration out of the diluate
    const back = ((1 - (cat ? cp.tA : cp.tC)) * jF + diff) * (g[j] / ((cat ? gC : gA) || 1)); // co-ion leakage and salt back-diffusion into the diluate
    dn[j] = clamp(((out - back) * dA) / AZ[j], -0.5 * st.qc * st.cc[j], 0.5 * st.qd * st.cd[j]);
  }
  const jw = (par.tw * cur.i * 18.015e-6) / F + 2 * par.Lp * R * (T + KELVIN) * 0.93 * (cp.c.tot - cp.d.tot); // m³/m²·s, electro-osmosis + osmosis
  return { dn, dq: Math.min(jw * dA, 0.3 * st.qd), cp, cur, pr, iWs, u, jw };
}

/** March one hydraulic stage (co-current diluate and concentrate) in nSeg segments with a midpoint (RK2) step. */
export function marchStage(st0, Ucp, G, par, T, nSeg) {
  const dA = (G.W * G.L) / nSeg, segs = [];
  let st = { qd: st0.qd, qc: st0.qc, cd: [...st0.cd], cc: [...st0.cc] }, I = 0, Iws = 0, Awet = 0;
  const step = (s, fl, f) => {
    const qd = s.qd - f * fl.dq, qc = s.qc + f * fl.dq;
    return { qd, qc, cd: s.cd.map((c, j) => Math.max(0, (s.qd * c - f * fl.dn[j]) / qd)), cc: s.cc.map((c, j) => Math.max(0, (s.qc * c + f * fl.dn[j]) / qc)) };
  };
  for (let s = 0; s < nSeg; s++) {
    const f1 = fluxes(st, Ucp, G, par, T, dA), mid = step(st, f1, 0.5), f2 = fluxes(mid, Ucp, G, par, T, dA);
    segs.push({ x: ((s + 0.5) * G.L) / nSeg, i: f2.cur.i, iFilm: f2.cur.iFilm, ilim: f2.cp.ilim, ratio: f2.cur.i / f2.cp.ilim, ceqD: f2.cp.d.ceq, ceqC: f2.cp.c.ceq, kD: f2.cp.d.kappa, kC: f2.cp.c.kappa, pr: f2.pr, wd: f2.pr.wd, wc: f2.pr.wc, u: f2.u, k: f2.cp.hd.k, Re: f2.cp.hd.Re, dpPerM: f2.cp.hd.dpPerM, iWs: f2.iWs, jw: f2.jw, cd: mid.cd, cc: mid.cc });
    I += f2.cur.i * dA; Iws += f2.iWs * dA; Awet += dA;
    st = step(st, f2, 1);
  }
  return { st, segs, I, Iws, iAvg: I / Awet, ratioMax: Math.max(...segs.map((s) => s.ratio)), dp: (sum(segs.map((s) => s.dpPerM)) * G.L) / nSeg };
}

/** Electrode pair: reversible water-electrolysis voltage + Butler–Volmer (or Tafel) overpotentials + rinse-compartment ohmic drop, V. */
export function electrodeVoltage(i, par, T) {
  const a = par.aBV ?? 0.5, kin = par.kinetics || 'bv'; // symmetric Butler–Volmer (α = 0.5) has the closed form η = (2RT/F)·asinh(i / 2i₀)
  return 1.229 + overpotential(i, par.i0a, T, a, kin) + overpotential(i, par.i0c, T, a, kin) + i * par.Rrinse;
}

/** Simple saturation ratios of the concentrate (Davies activity coefficients): gypsum and calcite. */
export function scaling(ions, pH, T) {
  const I = 0.5 * sum(CH.map((k) => ((+ions[k] || 0) / IONS[k].mw / 1000) * IONS[k].z ** 2)), sI = Math.sqrt(Math.min(I, 1.2));
  const lg2 = -0.509 * 4 * (sI / (1 + sI) - 0.3 * Math.min(I, 1.2)), g2 = 10 ** lg2, g1 = 10 ** (lg2 / 4), m = (k) => (+ions[k] || 0) / IONS[k].mw / 1000;
  const kGyp = 10 ** (-4.58 - 0.0005 * (T - 25)), kCal = 10 ** (-8.48 - 0.012 * (T - 25)), pK2 = 10.33 - 0.009 * (T - 25);
  const co3 = m('CO3') + (m('HCO3') * g1 * 10 ** (pH - pK2)) / g2;
  return { I, gypsum: (g2 * g2 * m('Ca') * m('SO4')) / kGyp, calcite: (g2 * g2 * m('Ca') * Math.min(co3, m('CO3') + m('HCO3'))) / kCal };
}

/**
 * 1-D steady electroneutral Nernst–Planck profile across diluate film | charged membrane | concentrate film
 * for a 1:1 salt (Teorell–Meyer–Sievers membrane with Donnan equilibrium at both faces). Lengths in m,
 * concentrations in mol/m³, current density in A/m². X > 0 is the fixed-charge concentration of a cation-exchange membrane.
 */
export function npProfile({ cd, cc, i, X, dm, Dp, Dm, DpM, DmM, deltaD, deltaC, T = 25, n = 40 }) {
  const V = vt(T), co = (cw) => 0.5 * (-X + Math.sqrt(X * X + 4 * cw * cw));
  const run = (Jm, keep) => {
    const Jp = i / F + Jm, Ka = Jp / Dp + Jm / Dm, Kb = Jp / Dp - Jm / Dm, cwd = cd - 0.5 * deltaD * Ka;
    if (!(cwd > 0)) return { res: -1e9 * (1 - cwd), ok: false };
    let c = co(cwd), phi = (Math.abs(Ka) > 1e-30 ? (Kb / Ka) * Math.log(cwd / cd) : (-Kb * deltaD) / (2 * cd)) + Math.log(cwd / (c + X));
    const xs = [], cs = [], cplus = [], ps = [], h = dm / n, dphi = (cm) => -(Jp / DpM - Jm / DmM) / (2 * cm + X), dc = (cm) => -Jm / DmM + cm * dphi(cm);
    if (keep) for (let k = 0; k <= 12; k++) { const x = (deltaD * k) / 12, cx = cd - 0.5 * x * Ka; xs.push(x - deltaD); cs.push(cx); cplus.push(cx); ps.push(Math.abs(Ka) > 1e-30 ? (Kb / Ka) * Math.log(cx / cd) : (-Kb * x) / (2 * cd)); }
    const donL = Math.log(cwd / (c + X));
    for (let k = 0; k <= n; k++) {
      if (keep) { xs.push(k * h); cs.push(c); cplus.push(c + X); ps.push(phi); }
      if (k === n) break;
      const k1 = dc(c), p1 = dphi(c), k2 = dc(c + 0.5 * h * k1), p2 = dphi(c + 0.5 * h * k1), k3 = dc(c + 0.5 * h * k2), p3 = dphi(c + 0.5 * h * k2), k4 = dc(c + h * k3), p4 = dphi(c + h * k3);
      c += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4); phi += (h / 6) * (p1 + 2 * p2 + 2 * p3 + p4);
      if (!(c > 0)) return { res: -1e9, ok: false };
    }
    const cwc = Math.sqrt(c * (c + X)), donR = Math.log((c + X) / cwc), cEnd = cwc - 0.5 * deltaC * Ka;
    if (keep) {
      phi += donR;
      for (let k = 0; k <= 12; k++) { const x = (deltaC * k) / 12, cx = cwc - 0.5 * x * Ka; xs.push(dm + x); cs.push(cx); cplus.push(cx); ps.push(phi + (Math.abs(Ka) > 1e-30 && cx > 0 ? (Kb / Ka) * Math.log(cx / cwc) : (-Kb * x) / (2 * cwc))); }
    }
    return { res: cEnd - cc, ok: true, Jp, Jm, cwd, cwc, donL: donL * V, donR: donR * V, x: xs, c: cs, cplus, phi: ps.map((p) => p * V) };
  };
  // bracket the co-ion flux (negative: leaks from concentrate to diluate)
  const scale = i / F + (DmM * Math.max(cc, cd)) / dm + 1e-12;
  let lo = -1e-6 * scale, hi = 1e-6 * scale, flo = run(lo).res, fhi = run(hi).res, guard = 0;
  while (flo * fhi > 0 && guard++ < 60) { if (Math.abs(flo) < Math.abs(fhi)) { lo -= (hi - lo) * 1.6; flo = run(lo).res; } else { hi += (hi - lo) * 1.6; fhi = run(hi).res; } }
  const Jm = flo * fhi <= 0 ? brent((j) => run(j).res, lo, hi, 1e-16 * scale + 1e-22, 200) : Math.abs(flo) < Math.abs(fhi) ? lo : hi, out = run(Jm, true);
  return { ...out, tm: i > 0 ? (F * out.Jp) / i : NaN, potential: out.phi ? -(out.phi[out.phi.length - 1] - out.phi[0]) : NaN, residual: out.res };
}

// ---- stack / train ----------------------------------------------------------------------------------------
function params(v) {
  return { alphaC: v.alphaC, alphaA: v.alphaA, Rcem: v.Rcem * 1e-4, Raem: v.Raem * 1e-4, shA: v.shA, shB: v.shB, kdp: v.kdp, selDivC: v.selDivC, selDivA: v.selDivA, PsC: v.Ps * 1e-8, PsA: v.Ps * 1e-8,
    tw: v.tw, Lp: (v.Lp * 1e-6) / 3600 / 1e5, plateau: v.plateau, olSlope: v.olSlope, fws: v.fws, i0a: v.i0a, i0c: v.i0c, Rrinse: v.Rrinse * 1e-4, shunt: clamp(v.shunt / 100, 0, 0.5),
    membModel: v.membModel || 'tms', kinetics: v.kinetics || 'bv', aBV: clamp(v.alphaBV ?? 0.5, 0.05, 0.95), ol: null, ns: null };
}
const geometry = (v) => ({ W: v.W, L: v.Lpath, h: v.hsp / 1000, eps: v.eps, shadow: v.shadow });

/** Stage voltage for which the highest local i / i_lim equals phi (monotone in the voltage): secant with a brent fallback. */
function voltageForRatio(st, phi, G, par, T, nSeg, guess) {
  const f = (U) => marchStage(st, U, G, par, T, nSeg).ratioMax - phi;
  let u0 = guess > 0 ? guess : 0.4, f0 = f(u0), u1 = u0 * clamp(phi / (f0 + phi), 0.4, 2.5), f1;
  for (let k = 0; k < 10; k++) {
    if (Math.abs(f0) < 1e-7) return u0;
    f1 = f(u1);
    if (Math.abs(f1) < 1e-7) return u1;
    const u2 = u1 - (f1 * (u1 - u0)) / (f1 - f0);
    if (!(u2 > 0) || !Number.isFinite(u2)) break;
    u0 = u1; f0 = f1; u1 = u2;
  }
  let hi = 0.6, guard = 0;
  while (f(hi) < 0 && guard++ < 12) hi *= 1.8;
  return guard > 12 ? hi : brent(f, 1e-4, hi, 1e-7, 60);
}

/**
 * Continuous train: nSt hydraulic stages in series with a feed-and-bleed concentrate loop.
 * spec = { type: 'ratio', phi } | { type: 'voltage', U } | { type: 'current', i }.
 */
export function solveTrain(cf, o, spec, nSt, nSeg) {
  const { G, par, T, Ncp, qd0, rec } = o, Qd = qd0 * Ncp, Qf = Qd / rec, Qmk = Qf - Qd; // concentrate make-up = feed not sent to the diluate
  const warm = o.warm || {}, Ug = warm.U || [];
  let tr = warm.tr || new Array(NI).fill(0), wtr = warm.wtr || 0, out, ccIn = cf, qcIn = qd0;
  for (let it = 0; it < 14; it++) {
    // steady concentrate loop: blow-down carries make-up salt + transferred salt
    const Qbd = Qmk + wtr * Ncp, ccOut = cf.map((c, j) => Math.max(0, (Qmk * c + tr[j] * Ncp) / Math.max(Qbd, 1e-12))), qcOut = qd0 + wtr;
    const Qrec = Math.max(0, qcOut * Ncp - Qbd), Qin = Qrec + Qmk;
    ccIn = cf.map((c, j) => (Qrec * ccOut[j] + Qmk * c) / Qin); qcIn = Qin / Ncp;
    let st = { qd: qd0, qc: qcIn, cd: cf, cc: ccIn };
    const stages = [];
    for (let s = 0; s < nSt; s++) {
      let U;
      if (spec.type === 'voltage') U = spec.U;
      else if (spec.type === 'ratio') U = voltageForRatio(st, spec.phi, G, par, T, nSeg, Ug[s] ? (Ug[s] * spec.phi) / (warm.phi || spec.phi) : s ? Ug[s - 1] : 0);
      else { const f = (x) => marchStage(st, x, G, par, T, nSeg).iAvg - spec.i; let hi = 1, g = 0; while (f(hi) < 0 && g++ < 10) hi *= 2; U = f(hi) < 0 ? hi : brent(f, 1e-4, hi, 1e-6, 50); }
      const m = marchStage(st, U, G, par, T, nSeg);
      stages.push({ ...m, U, in: st }); Ug[s] = U;
      st = m.st;
    }
    const trN = cf.map((c, j) => qd0 * c - st.qd * st.cd[j]), wN = qd0 - st.qd, err = Math.max(...trN.map((x, j) => Math.abs(x - tr[j]))) / (Math.max(...trN.map(Math.abs)) + 1e-30);
    out = { stages, dil: st, ccIn, qcIn, ccOut, Qbd, Qrec, Qmk, Qd, Qf };
    tr = trN; wtr = wN; warm.phi = spec.phi;
    if (err < (o.tol || 1e-7)) break;
  }
  // close the loop exactly with the final transfers
  out.Qbd = Qmk + wtr * Ncp;
  out.ccOut = cf.map((c, j) => (Qmk * c + tr[j] * Ncp) / out.Qbd);
  out.Qp = out.dil.qd * Ncp; out.cp = out.dil.cd; out.tr = tr;
  if (o.warm) { o.warm.tr = tr; o.warm.wtr = wtr; o.warm.U = Ug; }
  return out;
}

/** Complete ED/EDR calculation: design (target product TDS), fixed voltage or fixed current. */
export function simulateED(v, ov = {}) {
  const p = { ...v, ...ov }, T = p.T, par = params(p), G = geometry(p), nSeg = Math.max(2, Math.round(p.nSeg));
  const ionsF = balanceCharge(scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1)), cf = toMolar(ionsF), neutral = tds(ionsF) - tdsOf(cf), rec = clamp(p.recovery / 100, 0.3, 0.97);
  const Qd = p.Qp / 3600; // diluate flow through the stack ≈ product flow, m³/s
  attachModels(p, par, G, T, cf, p.mode === 'design' ? p.uLin / 100 : Qd / Math.max(1, Math.round(p.Ncp)) / (G.W * G.h * G.eps));
  let Ncp, qd0, nSt = Math.max(1, Math.round(p.nStages)), tr, phi = null, reached = true;
  const mk = (n) => ({ G, par, T, Ncp: n, qd0: Qd / n, rec, tol: ov.tol, warm: {} });
  if (p.mode === 'design') {
    if (tds(ionsF) <= p.targetTDS) throw new Error(`The feed (${fmt(tds(ionsF), 4)} mg/L) already meets the ${p.targetTDS} mg/L target — nothing to desalinate.`);
    qd0 = (p.uLin / 100) * G.W * G.h * G.eps; Ncp = Math.max(1, Math.ceil(Qd / qd0)); qd0 = Qd / Ncp;
    const o = mk(Ncp), phiMax = clamp(p.safety / 100, 0.05, 0.99), target = p.targetTDS, tdsAt = (n, ph) => { const t = solveTrain(cf, o, { type: 'ratio', phi: ph }, n, nSeg); return [tdsOf(t.cp, neutral), t]; };
    let found = null;
    for (nSt = 1; nSt <= Math.round(p.maxStages); nSt++) { const [td, t] = tdsAt(nSt, phiMax); tr = t; if (td <= target) { found = td; break; } }
    if (found === null) { nSt = Math.round(p.maxStages); reached = false; phi = phiMax; }
    else if (tdsAt(nSt, 0.02)[0] <= target) { phi = 0.02; tr = tdsAt(nSt, phi)[1]; }
    else { phi = brent((ph) => tdsAt(nSt, ph)[0] - target, 0.02, phiMax, 1e-5, 40); tr = tdsAt(nSt, phi)[1]; }
  } else {
    Ncp = Math.max(1, Math.round(p.Ncp));
    tr = solveTrain(cf, mk(Ncp), p.mode === 'current' ? { type: 'current', i: p.iSet } : { type: 'voltage', U: p.Ucp }, nSt, nSeg);
  }
  // ---- electrical accounting
  const nPar = Math.max(1, Math.ceil(Ncp / Math.max(1, Math.round(p.cpStack))));
  const stages = tr.stages.map((s, k) => {
    const Uel = electrodeVoltage(s.iAvg, par, T), Ustack = (Ncp / nPar) * s.U + Uel, Istack = s.I / (1 - par.shunt); // nPar parallel stacks share the cell pairs
    return { ...s, n: k + 1, Uel, Ustack, Istack, P: nPar * Ustack * Istack, tdsIn: tdsOf(s.in.cd, neutral), tdsOut: tdsOf(s.st.cd, neutral), tdsC: tdsOf(s.st.cc, neutral) };
  });
  const Pdc = sum(stages.map((s) => s.P)), Pel = Pdc / (p.etaRect / 100), dp = sum(stages.map((s) => s.dp)) + nSt * p.dpManifold * 1e5;
  const Ppump = ((tr.Qd + tr.qcIn * Ncp) * dp) / (p.etaPump / 100), edrLoss = p.edr ? clamp(p.offSpec / (p.revInterval * 60), 0, 0.5) : 0;
  const Qprod = tr.Qp * (1 - edrLoss), eqRem = sum(tr.tr.map((x, j) => (Z[j] > 0 ? x * AZ[j] : 0))), Itot = nPar * sum(stages.map((s) => s.Istack));
  const area = 2 * Ncp * nSt * G.W * G.L, tdsP = tdsOf(tr.cp, neutral), tdsF = tds(ionsF), ionsP = toIons(tr.cp, ionsF), ionsC = toIons(tr.ccOut, ionsF);
  // local pH estimates from water splitting (H⁺ to the diluate, OH⁻ to the concentrate), buffered by bicarbonate
  const hAdd = sum(stages.map((s) => s.Iws)) / F / tr.dil.qd, alk = (ionsP.HCO3 || 0) / 61.017, co2 = ((ionsF.HCO3 || 0) / 61.017) * 10 ** (6.35 - p.pH);
  const pHd = hAdd <= 1e-12 ? p.pH : hAdd < alk ? clamp(6.35 + Math.log10((alk - hAdd) / (co2 + hAdd)), 2, p.pH) : clamp(-Math.log10((hAdd - alk) / 1000 + 1e-7), 1.5, p.pH);
  const cfc = tdsOf(tr.ccOut, neutral) / tdsF, pHc = Math.min(11, p.pH + 0.3 * Math.log10(Math.max(cfc, 1)) + (hAdd > 1e-12 ? Math.min(2.5, Math.log10(1 + (hAdd * 1e3) / Math.max(alk, 1e-3))) : 0));
  const wall = Math.max(...stages.flatMap((s) => s.segs.map((g) => g.wc / g.ceqC))), sc = scaling(ionsC, pHc, T), scWall = scaling(scaleIons(ionsC, wall), pHc, T);
  return { p, par, G, T, Ncp, nSt, nPar, nSeg, phi, reached, tr, stages, cf, neutral, ionsF, ionsP, ionsC, tdsF, tdsP, tdsC: tds(ionsC), Pdc, Pel, Ppump, dp, Qprod, Qfeed: tr.Qf, Qconc: tr.Qbd + tr.Qp * edrLoss, edrLoss,
    sec: (Pel + Ppump) / 1000 / (Qprod * 3600), secDC: Pdc / 1000 / (Qprod * 3600), secPump: Ppump / 1000 / (Qprod * 3600), eff: ((F * eqRem) / Math.max(sum(stages.map((s) => s.I)), 1e-30)) * (1 - par.shunt), effMem: (F * eqRem) / Math.max(sum(stages.map((s) => s.I)), 1e-30),
    eqRem, Itot, area, iAvg: sum(stages.map((s) => s.iAvg)) / nSt, ratioMax: Math.max(...stages.map((s) => s.ratioMax)), waterRec: Qprod / tr.Qf, pHd, pHc, sc, scWall, wall, hAdd,
    wMin: minWork(cf, tr.cp, tr.ccOut, tr.Qp, tr.Qbd, T) };
}

/** Reversible work of the separation from the ideal-solution Gibbs energy of the three streams, kWh per m³ of product. */
function minWork(cf, cp, cc, Qp, Qc, T) {
  const g = (c) => sum(c.map((x) => (x > 0 ? x * Math.log(x) : 0)));
  return (R * (T + KELVIN) * (Qp * g(cp) + Qc * g(cc) - (Qp + Qc) * g(cf.map((x, j) => (Qp * cp[j] + Qc * cc[j]) / (Qp + Qc))))) / Qp / 3.6e6;
}

/** Batch (recirculating) operation at constant cell-pair voltage, integrated with RK4. */
export function simulateBatch(v) {
  const p = v, T = p.T, par = params(p), G = geometry(p), nSeg = clamp(Math.round(p.nSeg / 2), 2, 10), Ncp = Math.max(1, Math.round(p.Ncp)), nt = Math.max(4, Math.round(p.nt));
  const ionsF = balanceCharge(scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1)), cf = toMolar(ionsF), neutral = tds(ionsF) - tdsOf(cf), rec = clamp(p.recovery / 100, 0.3, 0.97);
  const Vd0 = p.Vbatch, Vc0 = (Vd0 * (1 - rec)) / rec, qd0 = (p.uLin / 100) * G.W * G.h * G.eps, tEnd = p.tBatch * 60;
  attachModels(p, par, G, T, cf, p.uLin / 100);
  const unpack = (y) => ({ Vd: y[2 * NI], Vc: y[2 * NI + 1], cd: y.slice(0, NI).map((n) => Math.max(n, 0) / y[2 * NI]), cc: y.slice(NI, 2 * NI).map((n) => Math.max(n, 0) / y[2 * NI + 1]) });
  const pass = (y) => { const s = unpack(y); return { s, m: marchStage({ qd: qd0, qc: qd0, cd: s.cd, cc: s.cc }, p.Ucp, G, par, T, nSeg) }; };
  const rhs = (t, y) => {
    const { s, m } = pass(y), d = new Array(2 * NI + 2);
    for (let j = 0; j < NI; j++) { const r = Ncp * (qd0 * s.cd[j] - m.st.qd * m.st.cd[j]); d[j] = -r; d[NI + j] = r; }
    d[2 * NI] = -Ncp * (qd0 - m.st.qd); d[2 * NI + 1] = -d[2 * NI];
    return d;
  };
  if (tdsOf(cf, neutral) <= p.targetTDS) throw new Error(`The feed (${fmt(tdsOf(cf, neutral), 4)} mg/L) already meets the ${p.targetTDS} mg/L target — nothing to desalinate.`);
  const y0 = [...cf.map((c) => c * Vd0), ...cf.map((c) => c * Vc0), Vd0, Vc0], sol = rk4(rhs, y0, 0, tEnd, nt);
  const t = [], tdsD = [], tdsC = [], iAvg = [], P = [], ratio = [];
  let E = 0, charge = 0, last = null, tHit = null;
  sol.y.forEach((y, k) => {
    const { s, m } = pass(y), Uel = electrodeVoltage(m.iAvg, par, T), Pk = ((Ncp * p.Ucp + Uel) * m.I) / (1 - par.shunt) / (p.etaRect / 100);
    t.push(sol.t[k] / 60); tdsD.push(tdsOf(s.cd, neutral)); tdsC.push(tdsOf(s.cc, neutral)); iAvg.push(m.iAvg); P.push(Pk); ratio.push(m.ratioMax);
    if (tHit === null && k > 0 && tdsD[k] <= p.targetTDS) tHit = t[k - 1] + ((t[k] - t[k - 1]) * (tdsD[k - 1] - p.targetTDS)) / (tdsD[k - 1] - tdsD[k]);
    if (k > 0 && tHit === null) { E += 0.5 * (Pk + last.P) * (sol.t[k] - sol.t[k - 1]); charge += 0.5 * (m.I + last.I) * (sol.t[k] - sol.t[k - 1]); }
    else if (k > 0 && E >= 0 && last.done !== true && tHit !== null) { const fr = (tHit - t[k - 1]) / (t[k] - t[k - 1]); E += 0.5 * (Pk + last.P) * (sol.t[k] - sol.t[k - 1]) * fr; charge += 0.5 * (m.I + last.I) * (sol.t[k] - sol.t[k - 1]) * fr; }
    last = { P: Pk, I: m.I, done: tHit !== null && k > 0 && last && (last.done || tdsD[k] <= p.targetTDS) };
  });
  const kEnd = tHit === null ? nt : Math.max(1, t.findIndex((x) => x >= tHit)), fr = tHit === null ? 1 : (tHit - t[kEnd - 1]) / (t[kEnd] - t[kEnd - 1]);
  const lerp = (a) => a[kEnd - 1] + fr * (a[kEnd] - a[kEnd - 1]), yA = sol.y[kEnd - 1], yB = sol.y[kEnd], yE = yA.map((a, j) => a + fr * (yB[j] - a)), sE = unpack(yE);
  const tUse = tHit ?? t[nt], m0 = pass(y0).m, dp = m0.dp + p.dpManifold * 1e5, Ppump = (2 * qd0 * Ncp * dp) / (p.etaPump / 100), eqRem = sum(cf.map((c, j) => (Z[j] > 0 ? AZ[j] * (c * Vd0 - sE.cd[j] * sE.Vd) : 0)));
  return { p, par, G, T, Ncp, nSeg, nt, t, tdsD, tdsC, iAvg, P, ratio, tHit, tUse, sE, cf, neutral, ionsF, ionsP: toIons(sE.cd, ionsF), ionsC: toIons(sE.cc, ionsF), tdsP: lerp(tdsD), tdsCend: lerp(tdsC), tdsF: tds(ionsF),
    E, Ppump, sec: (E + Ppump * tUse * 60) / 3.6e6 / sE.Vd, secDC: E / 3.6e6 / sE.Vd, eff: (F * eqRem) / Math.max(charge * Ncp, 1e-30), area: 2 * Ncp * G.W * G.L, Vd0, Vc0, m0, y0, yE,
    Qprod: sE.Vd / (tUse / 60), waterRec: sE.Vd / (Vd0 + Vc0), ratioMax: Math.max(...ratio.slice(0, kEnd + 1)), iMean: sum(iAvg.slice(0, kEnd + 1)) / (kEnd + 1), power: E / (tUse * 60) };
}

// ---- bipolar-membrane ED (reduced order) -------------------------------------------------------------------
const kappaStrong = (c, lam, T) => { const I = c / 1000, s = Math.sqrt(I); return Math.max(1e-6, (lam * c * 1e-4 * (1 + 0.0191 * (T - 25))) / (1 + (0.55 * s) / (1 + 0.6 * s) + 0.03 * I)); };
/** Voltage of one BPM | AEM | CEM repeating unit at current density i (A/m²) producing acid/base of concentration c (mol/L). */
export function bpmUnit(i, cAcid, cBase, cSalt, v) {
  const T = v.T, Ews = vt(T) * Math.LN10 * (14 + Math.log10(Math.max(cAcid, 1e-7)) + Math.log10(Math.max(cBase, 1e-7))) * ((T + KELVIN) / 298.15) ** 0; // 0.828 V for 1 mol/L products at 25 °C
  const h = v.hsp / 1000, Rsol = h * (1 / kappaStrong(cAcid * 1000, 426.1, T) + 1 / kappaStrong(cBase * 1000, 248.4, T) + 1 / kappaStrong(Math.max(cSalt, 1), 126.4, T)) / v.shadow;
  const Rm = (v.Rbpm + v.Rcem + v.Raem) * 1e-4, eta = clamp(v.bpEff / 100 - (v.bpLoss / 100) * 0.5 * (cAcid + cBase), 0.2, 1);
  return { Ews, U: Ews + v.bpOver + i * (Rm + Rsol), ohmMem: i * Rm, ohmSol: i * Rsol, eta };
}
export function simulateBPMED(v) {
  const T = v.T, ionsF = scaleIons(cloneIons(v.ions), v.salinityFactor ?? 1), cf = toMolar(ionsF), el = electrolyte(cf, T), Qf = v.Qp / 3600, conv = clamp(v.bpConv / 100, 0.05, 0.98);
  const cOut = el.ceq * (1 - conv), cMean = (el.ceq - cOut) / Math.log(el.ceq / cOut), u = bpmUnit(v.iB, v.cProd, v.cProd, cMean, v);
  const nEq = Qf * el.ceq * conv, I = (nEq * F) / u.eta, area = I / v.iB, units = Math.max(1, Math.ceil(area / (v.W * v.Lpath))), stacks = Math.max(1, Math.ceil(units / v.bpUnits));
  const Uel = electrodeVoltage(v.iB, params(v), T), P = (u.U * I + stacks * Uel * v.iB * v.W * v.Lpath) / (v.etaRect / 100), Qprod = nEq / (v.cProd * 1000); // m³/s of each product
  const Ppump = ((Qf + 2 * Qprod * 4) * (v.dpManifold + 0.4) * 1e5) / (v.etaPump / 100), naoh = nEq * 0.039997, hcl = nEq * 0.036461; // kg/s
  return { v, T, ionsF, el, u, conv, cOut, nEq, I, area, units, stacks, P, Ppump, Qprod, naoh, hcl, Uel, ePerKg: (P + Ppump) / 3.6e6 / naoh, ePerM3: (P + Ppump) / 1000 / (Qf * 3600), eTheo: (u.Ews * F) / 0.039997 / 3.6e6 };
}

// ---- membrane capacitive deionisation (reduced order) ------------------------------------------------------
/** Modified-Donnan micropore state at half-cell voltage Vh (V): charge density σ and ion concentration (mol/m³ of micropores). */
export function mDonnan(Vh, c, v) {
  const V = vt(v.T), ce = 2 * c * Math.exp(v.muAtt), Cst = v.cStern * 1e6, f = (s) => V * Math.asinh(s / ce) + (s * F) / Cst - Vh;
  const sig = Vh <= 0 ? 0 : brent(f, 0, (Vh * Cst) / F + 1, 1e-9, 100);
  return { sigma: sig, ions: Math.sqrt(sig * sig + ce * ce), phiD: V * Math.asinh(sig / ce), phiSt: (sig * F) / Cst };
}
/** Gouy–Chapman–Stern planar double layer: charge and salt excess per m² of electrode surface. */
export function gcs(Vh, c, v) {
  const V = vt(v.T), lamD = Math.sqrt((78.4 * 8.8541878e-12 * R * (v.T + KELVIN)) / (2 * F * F * Math.max(c, 1e-6))), Cst = v.cSternA;
  const f = (pd) => pd * V + (4 * F * c * lamD * Math.sinh(pd / 2)) / Cst - Vh, pd = Vh <= 0 ? 0 : brent(f, 0, Vh / V, 1e-10, 100);
  return { phiD: pd * V, lamD, sigma: 4 * lamD * c * Math.sinh(pd / 2), w: 8 * lamD * c * Math.sinh(pd / 4) ** 2, eff: Math.tanh(pd / 4) };
}
/** Equilibrium salt adsorption capacity (mg NaCl per g of both electrodes) and charge (C/g) between Vdis and Vch. */
export function cdiEquilibrium(c, v, Vch = v.Vch, Vdis = v.Vdis) {
  const vmi = v.vmi * 1e-6, a = mDonnan(Vch / 2, c, v), b = mDonnan(Math.max(0, Vdis) / 2, c, v), charge = 0.5 * vmi * (a.sigma - b.sigma) * F;
  const gammaCDI = 0.5 * vmi * (a.ions - b.ions), gamma = v.cdiMem ? (v.alphaC * charge) / F : gammaCDI;
  return { charge, gamma, sac: gamma * 58.44e3, eff: charge > 0 ? (gamma * F) / charge : 0, effCDI: charge > 0 ? (gammaCDI * F) / charge : 0, a, b };
}
export function simulateCDI(v) {
  const T = v.T, ionsF = scaleIons(cloneIons(v.ions), v.salinityFactor ?? 1), el = electrolyte(toMolar(ionsF), T), c0 = el.ceq, eq = cdiEquilibrium(c0, v), M = v.cdiLoad, vmi = v.vmi * 1e-6, nt = Math.max(8, Math.round(v.nt));
  const qA = v.cdiFlow / 1000 / 60, ESR = v.cdiESR * 1e-4, sigOf = (Q) => (2 * Q) / (M * vmi * F), Veq = (Q) => { const s = Math.max(0, sigOf(Q)), ce = 2 * c0 * Math.exp(v.muAtt); return 2 * (vt(T) * Math.asinh(s / ce) + (s * F) / (v.cStern * 1e6)); };
  // charge per m² of cell, RC dynamics against the equilibrium (modified-Donnan) voltage
  const Qdis = 0.5 * M * vmi * eq.b.sigma * F, tA = v.cdiTads * 60, tD = v.cdiTdes * 60;
  const lamOf = (Q) => { if (v.cdiMem) return v.alphaC; const s = Math.max(1e-9, sigOf(Q)), ce = 2 * c0 * Math.exp(v.muAtt); return s / Math.sqrt(s * s + ce * ce); }; // differential charge efficiency dΓ/dΣ
  const curAt = (Vap, Q) => { const I = (Vap - Veq(Q)) / ESR; return I > 0 ? Math.min(I, (0.95 * F * qA * c0) / lamOf(Q)) : I; }; // charging cannot remove more salt than the flow supplies
  const ads = rk4((t, y) => [curAt(v.Vch, y[0])], [Qdis], 0, tA, nt), des = rk4((t, y) => [curAt(v.Vdis, y[0])], [ads.y[nt][0]], 0, tD, nt);
  const tt = [], ceff = [], cur = [], volt = [];
  let Ein = 0, Erec = 0, salt = 0;
  const walk = (sol, Vap, sign, t0) => sol.t.forEach((t, k) => {
    const Q = sol.y[k][0], I = curAt(Vap, Q), cOut = Math.max(0, c0 - (lamOf(Q) * I) / F / qA);
    tt.push((t0 + t) / 60); ceff.push(cOut * 58.44); cur.push(I); volt.push(Veq(Q));
    if (k > 0) { const dt = t - sol.t[k - 1], Im = 0.5 * (I + cur[cur.length - 2]), Qp = sol.y[k - 1][0]; if (sign > 0) { Ein += Vap * Im * dt; salt += (0.5 * (lamOf(Q) + lamOf(Qp)) * (Q - Qp)) / F; } else Erec += -0.5 * (Veq(Q) + volt[volt.length - 2]) * Im * dt; } // salt integrated in charge space: dΓ = Λ·dΣ/F
  });
  walk(ads, v.Vch, 1, 0); const nAds = tt.length; walk(des, v.Vdis, -1, tA);
  const vol = qA * tA, Enet = Ein - (v.cdiRecov / 100) * Math.max(0, Erec), cAvg = Math.max(0, c0 - salt / vol), dQ = ads.y[nt][0] - Qdis;
  const sE = Math.max(0, sigOf(ads.y[nt][0])), ce0 = 2 * c0 * Math.exp(v.muAtt), saltState = v.cdiMem ? (v.alphaC * dQ) / F : 0.5 * M * vmi * (Math.sqrt(sE * sE + ce0 * ce0) - eq.b.ions);
  const Qw = v.Qp / 3600, cellArea = (Qw / qA) * ((tA + tD) / tA);
  return { v, T, ionsF, el, c0, eq, tt, ceff, cur, volt, nAds, Ein, Erec, Enet, salt, saltState, vol, cAvg, dQ, removal: 1 - cAvg / c0, sacDyn: (salt * 58.44e3) / M, effDyn: dQ > 0 ? (salt * F) / dQ : 0, sec: Enet / 3.6e6 / vol, ePerMol: Enet / Math.max(salt, 1e-30) / 1000,
    waterRec: tA / (tA + tD), cellArea, mass: (cellArea * M) / 1000, prod: (qA * 3.6e6 * tA) / (tA + tD), power: (Enet * cellArea) / (tA + tD) / 1000 };
}

const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const isED = (v) => v.process === 'ed', cont = (v) => isED(v) && v.mode !== 'batch';
const round = (o) => Object.fromEntries(ION_IDS.map((k) => [k, +(+o[k] || 0).toPrecision(6)]));
const stream = (Q, T, pH, ions) => ({ Q, T, P: 1, pH, tds: tds(ions), ions: round(ions) });

/** Polarisation curve of one cell pair at fixed compositions: voltage versus current density through all regimes. */
export function polarisation(cd, cc, u, G, par, T, n = 60) {
  const cp = cellPair(cd, cc, u, G, par, T), pl = plateauOf(cp, par), Umax = cp.U(0.98 * cp.ilim) + pl + 0.8, U = linspace(cp.E0, Umax, n);
  return { cp, U, i: U.map((x) => currentAt(cp, x, par).i), Ulim: cp.U(0.98 * cp.ilim), Uover: cp.U(0.98 * cp.ilim) + pl };
}

// ---- Poisson–Nernst–Planck (1-D, steady, Scharfetter–Gummel, Newton) ---------------------------------------------
const EPS0 = 8.8541878128e-12;
/** Bernoulli function B(u) = u / (eᵘ − 1) and its derivative (exponentially fitted fluxes). */
const bern = (u) => (Math.abs(u) < 1e-5 ? 1 - u / 2 + (u * u) / 12 : u > 600 ? 0 : u / Math.expm1(u));
const dbern = (u) => { if (Math.abs(u) < 1e-4) return -0.5 + u / 6; if (u > 600) return 0; if (u < -600) return -1; const e = Math.expm1(u); return (e - u * (e + 1)) / (e * e); };
/** Debye length (m) of a solution holding Σ zᵢ²cᵢ = q (mol/m³). */
export const debyeLength = (q, T = 25, epsr = 78.4) => Math.sqrt((epsr * EPS0 * R * (T + KELVIN)) / (F * F * Math.max(q, 1e-12)));
/** Multi-ion Donnan potential (in units of RT/F) of a phase with signed fixed charge w (mol/m³) against a solution c: Σ zᵢcᵢe^(−zᵢψ) + w = 0. */
export function donnanMulti(c, z, w) {
  const f = (p) => sum(c.map((x, i) => z[i] * x * Math.exp(-z[i] * p))) + w;
  if (Math.abs(w) < 1e-300) return 0;
  let lo = -1, hi = 1, g = 0;
  while (f(lo) < 0 && g++ < 80) lo *= 2;
  g = 0; while (f(hi) > 0 && g++ < 80) hi *= 2;
  return brent(f, lo, hi, 1e-14, 200);
}
function invSmall(A, m, X, off, M) { // Gauss–Jordan inverse of a small dense matrix (flat, row-major, partial pivoting) written to X at offset off; M is a 2m² work array
  const w = 2 * m;
  M.fill(0);
  for (let i = 0; i < m; i++) { for (let j = 0; j < m; j++) M[i * w + j] = A[i * m + j]; M[i * w + m + i] = 1; }
  for (let k = 0; k < m; k++) {
    let p = k; for (let i = k + 1; i < m; i++) if (Math.abs(M[i * w + k]) > Math.abs(M[p * w + k])) p = i;
    if (!(Math.abs(M[p * w + k]) > 1e-300)) throw new Error('singular block');
    if (p !== k) for (let j = 0; j < w; j++) { const t = M[k * w + j]; M[k * w + j] = M[p * w + j]; M[p * w + j] = t; }
    const d = 1 / M[k * w + k]; for (let j = 0; j < w; j++) M[k * w + j] *= d;
    for (let i = 0; i < m; i++) if (i !== k) { const f = M[i * w + k]; if (f !== 0) for (let j = 0; j < w; j++) M[i * w + j] -= f * M[k * w + j]; }
  }
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) X[off + i * m + j] = M[i * w + m + j];
}
/**
 * Steady 1-D Poisson–Nernst–Planck problem on a stack of layers (solution films, charged membranes):
 *   dJᵢ/dx = 0,  Jᵢ = −Dᵢ(dcᵢ/dx + zᵢcᵢ dψ/dx),  d/dx(ε dψ/dx) = −(F²/RT)(Σ zᵢcᵢ + ω),  ψ = Fφ/RT.
 * spec = { z:[…], layers:[{ L (m), D:[…] (m²/s), X (signed fixed charge, mol/m³), epsr }], T, epsr, ratio, res,
 *   left/right: { type:'bulk', c:[…], psi (V) } | { type:'wall', psi (V) or sigma (C/m²), flux:[…] (mol/m²·s, +x), cFix:[… or null] } }.
 * Finite volumes on a geometrically graded mesh that resolves the Debye length at every interface, exponentially fitted
 * (Scharfetter–Gummel) fluxes and a damped Newton iteration on (c₁…c_n, ψ) with a block-tridiagonal Jacobian.
 */
export function solvePNP(spec, warm = null) {
  const T = spec.T ?? 25, z = spec.z, ns = z.length, m = ns + 1, epsr0 = spec.epsr ?? 78.4, Vt = vt(T), K0 = (F * F) / (EPS0 * R * (T + KELVIN)), ratio = clamp(spec.ratio ?? 1.25, 1.02, 3), res = spec.res ?? 5;
  const Lb = spec.left, Rb = spec.right, lay = spec.layers, nl = lay.length, bulk = Lb.type === 'bulk' ? Lb.c : Rb.c, bulkR = Rb.type === 'bulk' ? Rb.c : Lb.c;
  const q2 = (c) => sum(c.map((x, i) => z[i] * z[i] * x)), wallQ = (b) => (b.type !== 'wall' ? 0 : (b.cFix || []).some((x) => x != null) ? Math.max(...b.cFix.map((x) => x || 0)) : q2(bulk) * Math.exp(Math.min(Math.abs((b.psi ?? 0) / Vt), 12))); // counter-ion enrichment at a charged wall shortens the local screening length
  const cref = Math.max(...bulk, ...bulkR, 1e-9);
  let mesh = warm?.mesh;
  if (spec.meshOnly || !mesh) { // two-sided geometric grading of every layer
    const x = [0], fl = [];
    lay.forEach((ly, q) => {
      const hc = ly.L / (ly.n || 16), er = ly.epsr ?? epsr0;
      const end = (nb, wall, c) => (nb ? Math.min(hc, debyeLength(Math.max(q2(c), Math.abs(ly.X || 0), Math.abs(nb.X || 0)), T, er) / res) : wall.type === 'wall' ? Math.min(hc, debyeLength(Math.max(q2(c), wallQ(wall), Math.abs(ly.X || 0)), T, er) / res) : hc);
      let ha = end(lay[q - 1], Lb, bulk), hb = end(lay[q + 1], Rb, bulkR), tot = 0;
      const a = [], b = [];
      while (tot < ly.L && a.length + b.length < 4000) { if (ha <= hb) { a.push(ha); tot += ha; ha = Math.min(ha * ratio, hc); } else { b.push(hb); tot += hb; hb = Math.min(hb * ratio, hc); } }
      const hs = [...a, ...b.reverse()], f = ly.L / tot;
      for (const h of hs) { x.push(x[x.length - 1] + h * f); fl.push(q); }
    });
    mesh = { x, fl, N: x.length };
    if (spec.meshOnly) return { warm: { mesh } };
  }
  const { x, fl, N } = mesh, nf = N - 1, h = new Float64Array(nf), ef = new Float64Array(nf), g = Array.from({ length: ns }, () => new Float64Array(nf)), om = new Float64Array(N);
  for (let k = 0; k < nf; k++) { h[k] = x[k + 1] - x[k]; const ly = lay[fl[k]]; ef[k] = (ly.epsr ?? epsr0) / h[k]; for (let i = 0; i < ns; i++) g[i][k] = ly.D[i] / h[k]; }
  for (let k = 0; k < N; k++) om[k] = 0.5 * ((k > 0 ? (lay[fl[k - 1]].X || 0) * h[k - 1] : 0) + (k < nf ? (lay[fl[k]].X || 0) * h[k] : 0)); // fixed charge of the control volume, mol/m²
  // state: concentrations in units of cref, potential in units of RT/F
  const c = Array.from({ length: ns }, () => new Float64Array(N)), ps = new Float64Array(N);
  const psiL = (Lb.psi ?? 0) / Vt, psiR = (Rb.psi ?? 0) / Vt, Ltot = x[nf];
  if (warm?.c) { for (let i = 0; i < ns; i++) c[i].set(warm.c[i]); ps.set(warm.ps); }
  else for (let k = 0; k < N; k++) { // electroneutral / Donnan starting guess
    const s = x[k] / Ltot, cb = bulk.map((v, i) => v + (bulkR[i] - v) * s), wv = k > 0 && k < nf ? (lay[fl[k - 1]].X || 0) * 0.5 + (lay[fl[k]].X || 0) * 0.5 : lay[fl[Math.min(k, nf - 1)]].X || 0, pd = donnanMulti(cb, z, wv);
    ps[k] = psiL + (psiR - psiL) * s + pd;
    for (let i = 0; i < ns; i++) c[i][k] = (cb[i] * Math.exp(-z[i] * pd)) / cref;
  }
  const A = new Float64Array(N * m * m), B = new Float64Array(N * m * m), C = new Float64Array(N * m * m), r = new Float64Array(N * m), Binv = new Float64Array(N * m * m), wk = new Float64Array(2 * m * m), tmp = new Float64Array(m * m), Bk = new Float64Array(m * m), rr = new Float64Array(N * m), dx = new Float64Array(N * m);
  const flux = (i, k) => { const u = z[i] * (ps[k + 1] - ps[k]); return g[i][k] * (bern(u) * c[i][k] - bern(-u) * c[i][k + 1]); }; // in units of cref
  const bcRow = (k, b, sgn) => { // boundary node k; sgn = +1 left (outward face k), −1 right (face k−1)
    const o = k * m * m, kf = sgn > 0 ? k : k - 1, nb = sgn > 0 ? C : A, k2 = k + sgn;
    for (let i = 0; i < ns; i++) {
      const row = o + i * m;
      if (b.type === 'bulk' || (b.cFix && b.cFix[i] != null)) { B[row + i] = 1; r[k * m + i] = c[i][k] - (b.type === 'bulk' ? b.c[i] : b.cFix[i]) / cref; continue; }
      // prescribed flux through the wall (default: none): J_face − J_spec = 0, with J positive in +x
      const ka = Math.min(k, k2), u = z[i] * (ps[ka + 1] - ps[ka]), gg = g[i][kf], bp = bern(u), bm = bern(-u), dJ = gg * z[i] * (dbern(u) * c[i][ka] + dbern(-u) * c[i][ka + 1]), sc = 1 / gg;
      const dA = gg * bp * sc, dB_ = -gg * bm * sc; // ∂J/∂c at nodes ka and ka+1
      if (sgn > 0) { B[row + i] = dA; nb[row + i] = dB_; B[row + ns] = -dJ * sc; nb[row + ns] = dJ * sc; } else { nb[row + i] = dA; B[row + i] = dB_; nb[row + ns] = -dJ * sc; B[row + ns] = dJ * sc; }
      r[k * m + i] = (gg * (bp * c[i][ka] - bm * c[i][ka + 1]) - ((b.flux && b.flux[i]) || 0) / cref) * sc;
    }
    const row = o + ns * m;
    if (b.type === 'bulk' || b.sigma == null) { B[row + ns] = 1; r[k * m + ns] = ps[k] - (sgn > 0 ? psiL : psiR); return; }
    // surface-charge (Neumann) condition: ε dψ/dn + K0·(½ control-volume charge) + σF/(ε₀RT) = 0
    const e = ef[kf], sc = 1 / e; let rho = om[k] / cref;
    for (let i = 0; i < ns; i++) { rho += 0.5 * h[kf] * z[i] * c[i][k]; B[row + i] = K0 * cref * 0.5 * h[kf] * z[i] * sc; }
    B[row + ns] = -1; nb[row + ns] = 1;
    r[k * m + ns] = (e * (ps[k2] - ps[k]) + K0 * cref * rho + (b.sigma * F) / (EPS0 * R * (T + KELVIN))) * sc;
  };
  let it = 0, conv = false, upd = Infinity;
  const maxIt = spec.maxIt ?? 80;
  for (; it < maxIt; it++) {
    A.fill(0); B.fill(0); C.fill(0);
    for (let k = 1; k < nf; k++) {
      const o = k * m * m, hb = 0.5 * (h[k - 1] + h[k]);
      for (let i = 0; i < ns; i++) {
        const row = o + i * m, uL = z[i] * (ps[k] - ps[k - 1]), uR = z[i] * (ps[k + 1] - ps[k]), gL = g[i][k - 1], gR = g[i][k], sc = 1 / Math.max(gL, gR);
        const dL = gL * z[i] * (dbern(uL) * c[i][k - 1] + dbern(-uL) * c[i][k]), dR = gR * z[i] * (dbern(uR) * c[i][k] + dbern(-uR) * c[i][k + 1]);
        A[row + i] = -gL * bern(uL) * sc; B[row + i] = (gR * bern(uR) + gL * bern(-uL)) * sc; C[row + i] = -gR * bern(-uR) * sc;
        A[row + ns] = dL * sc; B[row + ns] = (-dR - dL) * sc; C[row + ns] = dR * sc;
        r[k * m + i] = (flux(i, k) - flux(i, k - 1)) * sc;
      }
      const row = o + ns * m, sc = 1 / (ef[k - 1] + ef[k]); let rho = om[k] / cref;
      for (let i = 0; i < ns; i++) { rho += hb * z[i] * c[i][k]; B[row + i] = K0 * cref * hb * z[i] * sc; }
      A[row + ns] = ef[k - 1] * sc; B[row + ns] = -1; C[row + ns] = ef[k] * sc;
      r[k * m + ns] = (ef[k] * (ps[k + 1] - ps[k]) - ef[k - 1] * (ps[k] - ps[k - 1]) + K0 * cref * rho) * sc;
    }
    bcRow(0, Lb, 1); bcRow(nf, Rb, -1);
    // block Thomas elimination
    try {
      for (let k = 0; k < N; k++) {
        const o = k * m * m;
        for (let j = 0; j < m * m; j++) Bk[j] = B[o + j];
        for (let j = 0; j < m; j++) rr[k * m + j] = r[k * m + j];
        if (k > 0) {
          const op = (k - 1) * m * m;
          for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) { let s = 0; for (let q = 0; q < m; q++) s += A[o + a * m + q] * Binv[op + q * m + b]; tmp[a * m + b] = s; } // M = A·B'⁻¹
          for (let a = 0; a < m; a++) { let s = 0; for (let b = 0; b < m; b++) { s += tmp[a * m + b] * rr[(k - 1) * m + b]; let t = 0; for (let q = 0; q < m; q++) t += tmp[a * m + q] * C[op + q * m + b]; Bk[a * m + b] -= t; } rr[k * m + a] -= s; }
        }
        invSmall(Bk, m, Binv, o, wk);
      }
    } catch { break; }
    for (let k = N - 1; k >= 0; k--) {
      const o = k * m * m;
      for (let a = 0; a < m; a++) { let s = rr[k * m + a]; if (k < nf) for (let b = 0; b < m; b++) s -= C[o + a * m + b] * dx[(k + 1) * m + b]; tmp[a] = s; }
      for (let a = 0; a < m; a++) { let s = 0; for (let b = 0; b < m; b++) s += Binv[o + a * m + b] * tmp[b]; dx[k * m + a] = s; }
    }
    // damping: limit the potential change to two thermal voltages; large concentration changes are applied as factors (keeps c > 0)
    let lam = 1, mp = 0, mc = 0;
    for (let k = 0; k < N; k++) { mp = Math.max(mp, Math.abs(dx[k * m + ns])); for (let i = 0; i < ns; i++) mc = Math.max(mc, Math.abs(dx[k * m + i]) / (c[i][k] + 1e-9)); }
    if (!Number.isFinite(mp) || !Number.isFinite(mc)) break;
    if (mp > 2) lam = 2 / mp;
    for (let k = 0; k < N; k++) { ps[k] -= lam * dx[k * m + ns]; for (let i = 0; i < ns; i++) { const q = (-lam * dx[k * m + i]) / (c[i][k] + 1e-300); c[i][k] = Math.max(Math.abs(q) > 0.3 ? c[i][k] * Math.exp(clamp(q, -3, 3)) : c[i][k] * (1 + q), 1e-200); } }
    upd = Math.max(mp, mc);
    if (lam === 1 && upd < (spec.tol ?? 1e-10)) { conv = true; it++; break; }
  }
  const J = z.map((_, i) => { let s = 0, lo = Infinity, hi = -Infinity; for (let k = 0; k < nf; k++) { const f = flux(i, k) * cref; s += f; lo = Math.min(lo, f); hi = Math.max(hi, f); } return { mean: s / nf, spread: hi - lo }; });
  const cur = F * sum(J.map((j, i) => z[i] * j.mean)), rho = new Array(N), cOut = c.map((ci) => Array.from(ci, (v) => v * cref));
  for (let k = 0; k < N; k++) { const lyX = k > 0 && k < nf ? 0.5 * ((lay[fl[k - 1]].X || 0) + (lay[fl[k]].X || 0)) : lay[fl[Math.min(k, nf - 1)]].X || 0; rho[k] = F * (sum(z.map((zi, i) => zi * cOut[i][k])) + lyX); }
  const E0 = (-(ps[1] - ps[0]) * Vt) / h[0], En = (-(ps[nf] - ps[nf - 1]) * Vt) / h[nf - 1];
  return { x: Array.from(x), c: cOut, psi: Array.from(ps, (p) => p * Vt), rho, J: J.map((j) => j.mean), fluxSpread: Math.max(...J.map((j) => j.spread)) / (Math.max(...J.map((j) => Math.abs(j.mean))) + 1e-300), current: cur, converged: conv, iterations: it, update: upd, nodes: N, fieldLeft: E0, fieldRight: En,
    sigmaLeft: (lay[0].epsr ?? epsr0) * EPS0 * E0 - 0.5 * rho[0] * h[0], sigmaRight: -(lay[nl - 1].epsr ?? epsr0) * EPS0 * En - 0.5 * rho[nf] * h[nf - 1], lamD: debyeLength(q2(bulk), T, epsr0), warm: { mesh, c, ps } };
}
const pnpCopy = (w) => ({ mesh: w.mesh, c: w.c.map((a) => Float64Array.from(a)), ps: Float64Array.from(w.ps) });
/** Continuation of a converged PNP solution from parameter s0 to s1 of the family specAt(s), with adaptive sub-steps. */
export function pnpContinue(specAt, sol, s0, s1, ds0 = s1 - s0) {
  let s = s0, ds = ds0, guard = 0;
  while (Math.abs(s1 - s) > 1e-12 * (Math.abs(s1) + 1) && guard++ < 300) {
    const sn = Math.abs(s1 - s) <= Math.abs(ds) ? s1 : s + ds, nx = solvePNP({ maxIt: 40, ...specAt(sn) }, pnpCopy(sol.warm));
    if (nx.converged) { sol = nx; s = sn; ds *= 1.7; } else { ds *= 0.35; if (Math.abs(ds) < 1e-6 * Math.abs(s1 - s0)) return { ...sol, converged: false, reached: s }; }
  }
  return { ...sol, reached: s };
}
/** PNP with continuation: ramps the boundary potentials / wall data linearly from the easy state `from` to the target. */
export function pnpRamp(spec, from = {}, steps = 4) {
  const lerp = (a, b, s) => (a == null || b == null ? b : Array.isArray(b) ? b.map((v, i) => (v == null ? v : (a[i] ?? 0) + (v - (a[i] ?? 0)) * s)) : a + (b - a) * s);
  const at = (s) => ({ ...spec, left: { ...spec.left, ...Object.fromEntries(Object.keys(from.left || {}).map((k) => [k, lerp(from.left[k], spec.left[k], s)])) }, right: { ...spec.right, ...Object.fromEntries(Object.keys(from.right || {}).map((k) => [k, lerp(from.right[k], spec.right[k], s)])) } });
  const sol = solvePNP(at(0), solvePNP({ ...spec, meshOnly: true }).warm); // the mesh is graded for the target state
  return sol.converged ? pnpContinue(at, sol, 0, 1, 1 / steps) : sol;
}

// ---- Goldman–Hodgkin–Katz constant-field membrane ---------------------------------------------------------------
/** GHK flux of one ion (mol/m²·s for P in m/s, c in mol/m³): J = P·z·u·(c₁ − c₂e^(−zu)) / (1 − e^(−zu)), u = F(φ₁ − φ₂)/RT. */
export const ghkFlux = (P, z, c1, c2, u) => P * (bern(-z * u) * c1 - bern(z * u) * c2);
/** GHK zero-current membrane potential φ₁ − φ₂ (V) for any mixture of valences: Σ zᵢJᵢ(u) = 0 solved for u. */
export function ghkPotential(P, z, c1, c2, T = 25, u0 = 0) {
  const cur = (u) => { let s = 0; for (let j = 0; j < z.length; j++) if (P[j] > 0 && (c1[j] > 0 || c2[j] > 0)) s += z[j] * ghkFlux(P[j], z[j], c1[j], c2[j], u); return s; };
  let u = Number.isFinite(u0) ? u0 : 0, ok = false; // Newton on the monotone current–voltage relation (optionally warm-started, u0 in RT/F), bracketing fallback
  for (let k = 0; k < 40; k++) {
    let f = 0, df = 0;
    for (let j = 0; j < z.length; j++) if (P[j] > 0 && (c1[j] > 0 || c2[j] > 0)) {
      // one exponential per ion: B(−x) = B(x) + x and B′(−x) = −1 − B′(x)
      const x = z[j] * u, ax = Math.abs(x); let B, dB;
      if (ax < 1e-5) { B = 1 - x / 2 + (x * x) / 12; dB = -0.5 + x / 6; } else if (x > 600) { B = 0; dB = 0; } else if (x < -600) { B = -x; dB = -1; } else { const e = Math.expm1(x); B = x / e; dB = ax < 1e-4 ? -0.5 + x / 6 : (e - x * (e + 1)) / (e * e); }
      f += z[j] * P[j] * ((B + x) * c1[j] - B * c2[j]); df -= z[j] * z[j] * P[j] * ((-1 - dB) * c1[j] + dB * c2[j]);
    }
    if (df === 0 && f === 0) return 0; // no permeating ion on either side
    if (!(df > 0)) break;
    const d = clamp(f / df, -3, 3); u -= d;
    if (Math.abs(d) < 1e-13) { ok = true; break; }
  }
  if (ok) return vt(T) * u;
  let lo = -1, hi = 1, g = 0;
  while (cur(lo) > 0 && g++ < 9) lo *= 2;
  g = 0; while (cur(hi) < 0 && g++ < 9) hi *= 2;
  return cur(lo) * cur(hi) > 0 ? 0 : vt(T) * brent(cur, lo, hi, 1e-13, 100);
}

// ---- electrode kinetics --------------------------------------------------------------------------------------------
/** Activation overpotential (V) at current density i: Butler–Volmer i = i₀[e^(αFη/RT) − e^(−(1−α)Fη/RT)] or its Tafel limit η = (RT/αF)·ln(i/i₀). */
export function overpotential(i, i0, T = 25, a = 0.5, kin = 'bv') {
  const V = vt(T);
  if (!(i > 0)) return 0;
  if (kin === 'tafel') return i > i0 ? (V / a) * Math.log(i / i0) : 0;
  let e = 2 * V * Math.asinh(i / (2 * i0));
  if (Math.abs(a - 0.5) < 1e-9) return e;
  e = Math.min(e, (V / a) * Math.log(1 + i / i0) + 1e-9);
  for (let k = 0; k < 60; k++) { const p = Math.exp((a * e) / V), q = Math.exp((-(1 - a) * e) / V), d = (i0 * (p - q) - i) / ((i0 / V) * (a * p + (1 - a) * q)); e -= d; if (Math.abs(d) < 1e-15) break; }
  return e;
}

// ---- Maxwell–Stefan transport in a charged membrane ------------------------------------------------------------------
/**
 * Steady Maxwell–Stefan (friction) transport of n ions through a membrane with fixed charge X (signed, mol/m³), water and polymer:
 *   −(∇cᵢ + zᵢcᵢ∇ψ) = Σⱼ (xⱼNᵢ − xᵢNⱼ)/Đᵢⱼ + (x_wNᵢ − xᵢN_w)/Đᵢw + x_mNᵢ/Đᵢm,   Σ zᵢcᵢ + X = 0,   F Σ zᵢNᵢ = i,
 * with Donnan equilibrium at both faces and the water flux N_w fixed by a zero hydrostatic pressure difference across the membrane
 * (electro-osmotic drag). Shooting on the n + 1 constant fluxes (RK4 across the membrane, Newton). Đᵢⱼ may be a number or a matrix.
 */
export function msMembrane({ z, cL, cR, X, dm, Diw, Dim, Dij = Infinity, Dwm, cw, i, T = 25, n = 40 }) {
  const ns = z.length, V = vt(T), pL = donnanMulti(cL, z, X), pR = donnanMulti(cR, z, X), mL = cL.map((c, j) => c * Math.exp(-z[j] * pL)), mR = cR.map((c, j) => c * Math.exp(-z[j] * pR)), aX = Math.abs(X);
  const dij = (a, b) => (typeof Dij === 'number' ? Dij : Dij[a][b]), ref = mL.indexOf(Math.max(...mL)), idx = z.map((_, j) => j).filter((j) => j !== ref), h = dm / n;
  const Dbar = sum(Diw) / ns, Nref = Math.abs(i) / F + (Dbar * Math.max(...mL, ...mR)) / dm, cs = Math.max(aX, ...mL, ...mR);
  const grad = (c, N, Nw) => {
    const ct = cw + sum(c) + aX, xw = cw / ct, xm = aX / ct, rhs = new Array(ns);
    let zr = 0, zz = 0, wA = 0, wB = xm / Dwm;
    for (let a = 0; a < ns; a++) {
      let s = (xw * N[a] - (c[a] / ct) * Nw) / Diw[a] + (xm * N[a]) / Dim[a];
      for (let b = 0; b < ns; b++) if (b !== a) { const d = dij(a, b); if (Number.isFinite(d)) s += ((c[b] / ct) * N[a] - (c[a] / ct) * N[b]) / d; }
      rhs[a] = s; zr += z[a] * s; zz += z[a] * z[a] * c[a]; wA += (xw * N[a]) / Diw[a]; wB += c[a] / ct / Diw[a];
    }
    const dpsi = -zr / zz;
    return { dc: rhs.map((s, a) => -z[a] * c[a] * dpsi - s), dpsi, wA, wB };
  };
  const shoot = (u, keep, iq = i) => {
    const N = u.slice(0, ns).map((q) => q * Nref), Nw = u[ns] * Nref;
    let c = [...mL], psi = 0, IA = 0, IB = 0;
    const prof = keep ? { x: [0], c: [[...c]], psi: [0] } : null;
    for (let k = 0; k < n; k++) {
      const k1 = grad(c, N, Nw), c2 = c.map((v, a) => Math.max(v + 0.5 * h * k1.dc[a], 1e-12)), k2 = grad(c2, N, Nw), c3 = c.map((v, a) => Math.max(v + 0.5 * h * k2.dc[a], 1e-12)), k3 = grad(c3, N, Nw), c4 = c.map((v, a) => Math.max(v + h * k3.dc[a], 1e-12)), k4 = grad(c4, N, Nw);
      c = c.map((v, a) => Math.max(v + (h / 6) * (k1.dc[a] + 2 * k2.dc[a] + 2 * k3.dc[a] + k4.dc[a]), 1e-12)); psi += (h / 6) * (k1.dpsi + 2 * k2.dpsi + 2 * k3.dpsi + k4.dpsi);
      IA += (h / 6) * (k1.wA + 2 * k2.wA + 2 * k3.wA + k4.wA); IB += (h / 6) * (k1.wB + 2 * k2.wB + 2 * k3.wB + k4.wB);
      if (keep) { prof.x.push((k + 1) * h); prof.c.push([...c]); prof.psi.push(psi * V); }
    }
    const res = idx.map((j) => (c[j] - mR[j]) / cs);
    res.push((F * sum(N.map((q, a) => z[a] * q)) - iq) / (F * Nref)); res.push((Nw * IB - IA) / (Nref * IB)); // current and zero pressure difference
    return { res, c, psi, N, Nw, prof };
  };
  // start: the friction equations are linear in (N, N_w, Δψ) for linear concentration profiles — solve that system at the mean composition
  const cm = mL.map((v, a) => 0.5 * (v + mR[a])), ctm = cw + sum(cm) + aX, xwm = cw / ctm, xmm = aX / ctm, nn = ns + 2;
  const guess = (iq) => {
    const A = Array.from({ length: nn }, () => new Array(nn).fill(0)), bv = new Array(nn).fill(0);
    for (let a = 0; a < ns; a++) {
      A[a][a] = xwm / Diw[a] + xmm / Dim[a]; A[a][ns] = -(cm[a] / ctm) / Diw[a]; A[a][ns + 1] = (z[a] * cm[a]) / dm; bv[a] = -(mR[a] - mL[a]) / dm;
      for (let b = 0; b < ns; b++) if (b !== a) { const d = dij(a, b); if (Number.isFinite(d)) { A[a][a] += cm[b] / ctm / d; A[a][b] -= cm[a] / ctm / d; } }
      A[ns][a] = z[a]; A[ns + 1][a] = -xwm / Diw[a]; A[ns + 1][ns] += cm[a] / ctm / Diw[a];
    }
    bv[ns] = iq / F; A[ns + 1][ns] += xmm / Dwm;
    for (let a = 0; a < nn; a++) { const sc = Math.max(...A[a].map(Math.abs)) || 1; for (let b = 0; b < nn; b++) A[a][b] /= sc; bv[a] /= sc; }
    try { return solveLinear(A, bv).slice(0, ns + 1).map((q) => q / Nref); } catch { return new Array(ns + 1).fill(0); }
  };
  const u0 = guess(i);
  let sol = newtonN((u) => shoot(u).res, u0, { tol: 1e-10, maxIter: 60, h: 1e-7 }), iEff = i;
  if (!(sol.residual < 1e-7)) { // continuation in the current density from the diffusion-only state
    let us = guess(0), f = 0, df = 0.25, okc = false;
    const at = (fq, start) => newtonN((u) => shoot(u, false, fq * i).res, start, { tol: 1e-10, maxIter: 40, h: 1e-7 });
    let sq = at(0, us);
    if (sq.residual < 1e-6) { us = sq.x; for (let g = 0; g < 80 && f < 1; g++) { const fn = Math.min(1, f + df); sq = at(fn, us); if (sq.residual < 1e-6) { us = sq.x; f = fn; df *= 1.6; if (f >= 1) { sol = sq; okc = true; } } else { df *= 0.4; if (df < 1e-4) break; } } }
    if (!okc && !(sol.residual < 1e-7)) { sol = { x: us, residual: 0, iterations: sol.iterations, converged: false }; iEff = f * i; } // keep the last converged current of the continuation
  }
  const out = shoot(sol.x, true, iEff);
  const ct = cw + sum(mL) + aX;
  return { N: out.N, Nw: out.Nw, iUsed: iEff, t: out.N.map((q, a) => (iEff !== 0 ? (z[a] * F * q) / iEff : 0)), tw: iEff !== 0 ? (F * out.Nw) / iEff : 0, mL, mR, donL: pL * V, donR: pR * V, potential: (pL - pR - out.psi) * V, x: out.prof.x, c: out.prof.c, psi: out.prof.psi,
    Deff: Diw.map((d, a) => 1 / (cw / ct / d + aX / ct / Dim[a])), converged: iEff === i && (sol.converged || sol.residual < 1e-7), residual: sol.residual, iterations: sol.iterations };
}

// ---- electroconvection: Rubinstein–Zaltzman electro-osmotic slip model ---------------------------------------------
/** Stokes mode of wavenumber k in the unit layer: W(0) = W(1) = W′(1) = 0, W′(0) = 1 (biharmonic solution). */
export function stokesMode(k) {
  const sh = Math.sinh(k), ch = Math.cosh(k), [a, b, d] = solveLinear([[k, 0, 1], [sh, sh, ch], [k * ch, sh + k * ch, ch + k * sh]], [1, 0, 0]);
  return { W: (y) => (a + b * y) * Math.sinh(k * y) + d * y * Math.cosh(k * y), dW: (y) => b * Math.sinh(k * y) + (a + b * y) * k * Math.cosh(k * y) + d * Math.cosh(k * y) + d * y * k * Math.sinh(k * y) };
}
/**
 * Marginal voltage (in RT/F) of the quiescent limiting state for wavenumber k: Pe·V²/8 = −1/(k²·g′(0)), g″ − k²g = W.
 * ell > 0 applies the short-wave cut-off of the slip, exp(−(k·ℓ)²) (finite thickness of the extended space charge), which gives the
 * marginal curve a minimum — the critical wavenumber.
 */
export function ecMarginal(k, Pe, ell = 0) {
  const m = stokesMode(k), n = 400; let s = 0;
  for (let q = 0; q <= n; q++) { const y = q / n, w = q === 0 || q === n ? 1 : q % 2 ? 4 : 2; s += w * m.W(y) * (Math.sinh(k * (1 - y)) / Math.sinh(k)); }
  return Math.sqrt(8 / (Pe * k * k * (s / (3 * n)))) * Math.exp(0.5 * (k * ell) ** 2);
}
/** Critical (lowest-threshold) wavenumber and voltage of the marginal curve with the cut-off length ell (> 0): bracketing scan + golden-section refinement. */
export function ecCritical(Pe, ell) {
  const f = (k) => ecMarginal(k, Pe, ell); let kb = 0.8, vb = Infinity;
  for (let k = 0.8; k <= Math.min(14, 2.5 / ell + 1); k += 0.4) { const v = f(k); if (v < vb) { vb = v; kb = k; } }
  let a = Math.max(0.4, kb - 0.4), b = kb + 0.4; const g = 0.6180339887498949; let x1 = b - g * (b - a), x2 = a + g * (b - a), f1 = f(x1), f2 = f(x2);
  for (let it = 0; it < 22; it++) { if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = f(x1); } else { a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = f(x2); } }
  const kc = 0.5 * (a + b);
  return { kc, Vc: f(kc) };
}
/**
 * Non-linear electroconvection in the depleted diffusion layer (lengths in δ, time in δ²/D, c in bulk units, V in RT/F):
 *   c_t + u·∇c = ∇²c,  c(x,0) = 0,  c(x,1) = 1,  Stokes flow driven by the slip u_s = −(Pe·V²/8)·G_ℓ * ∂ₓ ln(∂c/∂y) at the membrane,
 * where G_ℓ is the Gaussian short-wave cut-off exp(−(kℓ)²) of the slip (ℓ = 0 switches it off). Periodic cell of one wavelength 2π/k.
 * Discretisation: second-order central differences on a grid clustered towards the depleted interface (y_j = (e^{βj/n} − 1)/(e^β − 1)),
 * second-order one-sided wall gradient, slip differentiated spectrally, Stokes problem solved exactly per Fourier mode (all modes the
 * cut-off passes), Peaceman–Rachford ADI in time (tridiagonal across the layer, cyclic tridiagonal along it) — the steady state is free
 * of time-step and splitting error. `init` restarts from a coarser solution (nested iteration).
 * Returns the (time-averaged, if unsteady) Sherwood number ⟨∂c/∂y⟩ = i / i_lim.
 */
export function ecSolve({ V, Pe, k, nx = 0, ny = 20, tEnd = 4, modes = 0, keep = false, ell = 0, beta = 2.5, init = null, cfl = 2, dtMax = 0.01, raw = false }) {
  const Lx = (2 * Math.PI) / k, amp = (Pe * V * V) / 8, Mw = modes > 0 ? Math.round(modes) : ell > 0 ? Math.ceil(3 / (k * ell)) : 3;
  nx = nx > 0 ? Math.max(6, 2 * Math.round(nx / 2)) : Math.max(16, 2 * Mw + 2);
  const M = Math.max(1, Math.min(Mw, nx / 2 - 1)), dx = Lx / nx, n1 = ny + 1, N = n1 * nx;
  const y = Float64Array.from({ length: n1 }, (_, j) => (beta > 1e-9 ? Math.expm1((beta * j) / ny) / Math.expm1(beta) : j / ny)); y[ny] = 1;
  const d2m = new Float64Array(n1), d2p = new Float64Array(n1), d1m = new Float64Array(n1), d1p = new Float64Array(n1), d10 = new Float64Array(n1);
  for (let j = 1; j < ny; j++) { const hm = y[j] - y[j - 1], hp = y[j + 1] - y[j]; d2m[j] = 2 / (hm * (hm + hp)); d2p[j] = 2 / (hp * (hm + hp)); d1m[j] = -hp / (hm * (hm + hp)); d1p[j] = hm / (hp * (hm + hp)); d10[j] = (hp - hm) / (hm * hp); }
  const kn = new Float64Array(M), fl = new Float64Array(M), Wn = [], dWn = [], sn = [], cn = [];
  for (let q = 0; q < M; q++) { const kk = (q + 1) * k, m = stokesMode(kk); kn[q] = kk; fl[q] = Math.exp(-((kk * ell) ** 2)); Wn.push(Float64Array.from(y, (yy) => kk * m.W(yy))); dWn.push(Float64Array.from(y, (yy) => m.dW(yy))); sn.push(Float64Array.from({ length: nx }, (_, i) => Math.sin(kk * i * dx))); cn.push(Float64Array.from({ length: nx }, (_, i) => Math.cos(kk * i * dx))); }
  const c = new Float64Array(N), cs = new Float64Array(N), u = new Float64Array(N), w = new Float64Array(N), jw = new Float64Array(nx), lj = new Float64Array(nx), ev = new Float64Array(nx), gv = new Float64Array(nx);
  if (init && init._c && init._y && init._nx) { // bilinear transfer of a coarser solution (periodic along the membrane)
    const ya = init._y, n0 = init._nx, c0 = init._c, m0 = ya.length - 1; let jj = 0;
    for (let j = 0; j <= ny; j++) {
      while (jj < m0 - 1 && ya[jj + 1] < y[j]) jj++;
      const fy = clamp((y[j] - ya[jj]) / (ya[jj + 1] - ya[jj]), 0, 1);
      for (let i = 0; i < nx; i++) { const xs = (i * n0) / nx, i0 = Math.floor(xs) % n0, fx = xs - Math.floor(xs), i1 = (i0 + 1) % n0; c[j * nx + i] = (1 - fy) * ((1 - fx) * c0[jj * n0 + i0] + fx * c0[jj * n0 + i1]) + fy * ((1 - fx) * c0[(jj + 1) * n0 + i0] + fx * c0[(jj + 1) * n0 + i1]); }
    }
  } else for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) c[j * nx + i] = y[j] + Math.sin(Math.PI * y[j]) * (0.02 * Math.cos(k * i * dx) + 0.004 * Math.cos(2 * k * i * dx + 1));
  for (let i = 0; i < nx; i++) { c[i] = 0; c[ny * nx + i] = 1; }
  const h1 = y[1], h2 = y[2], g1 = h2 / (h1 * (h2 - h1)), g2 = -h1 / (h2 * (h2 - h1)); // second-order wall gradient with c(0) = 0
  const flow = () => {
    let nu = 0, umax = 0;
    for (let i = 0; i < nx; i++) { const g = g1 * c[nx + i] + g2 * c[2 * nx + i]; nu += g; jw[i] = g > 1e-6 ? g : 1e-6; lj[i] = Math.log(jw[i]); }
    u.fill(0); w.fill(0);
    for (let q = 0; q < M; q++) {
      let A = 0, B = 0; const s = sn[q], co = cn[q];
      for (let i = 0; i < nx; i++) { A += lj[i] * s[i]; B += lj[i] * co[i]; }
      const f = (2 / nx) * amp * fl[q] * kn[q], a = f * B, b = -f * A; // slip u_s = −amp·∂ₓ ln j, differentiated in Fourier space
      if (Math.abs(a) + Math.abs(b) < 1e-14) continue;
      const dW = dWn[q], Wq = Wn[q];
      for (let i = 0; i < nx; i++) { ev[i] = a * s[i] + b * co[i]; gv[i] = -a * co[i] + b * s[i]; }
      for (let j = 0, o = 0; j <= ny; j++) { const dj = dW[j], wj = Wq[j]; for (let i = 0; i < nx; i++, o++) { u[o] += ev[i] * dj; w[o] += gv[i] * wj; } }
    }
    for (let q = 0; q < N; q++) { const a = Math.abs(u[q]); if (a > umax) umax = a; }
    return { nu: nu / nx, umax };
  };
  const al = new Float64Array(nx), bd = new Float64Array(nx), cu = new Float64Array(nx), rr = new Float64Array(nx), zz = new Float64Array(nx), cq = new Float64Array(nx), ta = new Float64Array(n1), tb = new Float64Array(n1), tc = new Float64Array(n1), td = new Float64Array(n1);
  const idx2 = 1 / (dx * dx), i2dx = 1 / (2 * dx);
  const cyclic = (o) => { // cyclic tridiagonal system (Sherman–Morrison), solution written to cs[o … o + nx)
    const gam = -bd[0], bn = bd[nx - 1] - (al[0] * cu[nx - 1]) / gam;
    let bet = bd[0] - gam; cq[0] = cu[0] / bet; rr[0] /= bet; zz[0] = gam / bet;
    for (let i = 1; i < nx; i++) { bet = (i === nx - 1 ? bn : bd[i]) - al[i] * cq[i - 1]; cq[i] = cu[i] / bet; rr[i] = (rr[i] - al[i] * rr[i - 1]) / bet; zz[i] = ((i === nx - 1 ? cu[nx - 1] : 0) - al[i] * zz[i - 1]) / bet; }
    for (let i = nx - 2; i >= 0; i--) { rr[i] -= cq[i] * rr[i + 1]; zz[i] -= cq[i] * zz[i + 1]; }
    const fact = (rr[0] + (al[0] * rr[nx - 1]) / gam) / (1 + zz[0] + (al[0] * zz[nx - 1]) / gam);
    for (let i = 0; i < nx; i++) cs[o + i] = rr[i] - fact * zz[i];
  };
  let t = 0, steps = 0, f = flow(), nuAvg = 0, tAvg = 0, nuLast = f.nu, tLast = 0, steady = false;
  const hist = [];
  while (t < tEnd && steps < 20000) {
    const dt = Math.min(dtMax, (cfl * dx) / (f.umax + 1e-9), tEnd - t), hd = 0.5 * dt;
    for (let j = 1; j < ny; j++) { // implicit along the membrane
      const o = j * nx, am = d2m[j], ap = d2p[j], bm = d1m[j], b0 = d10[j], bp = d1p[j];
      for (let i = 0; i < nx; i++) { const q = o + i, uu = u[q], ww = w[q]; al[i] = -hd * (idx2 + uu * i2dx); cu[i] = -hd * (idx2 - uu * i2dx); bd[i] = 1 + dt * idx2; rr[i] = c[q] + hd * (am * c[q - nx] + ap * c[q + nx] - (am + ap) * c[q] - ww * (bm * c[q - nx] + b0 * c[q] + bp * c[q + nx])); }
      cyclic(o);
    }
    for (let i = 0; i < nx; i++) { cs[i] = 0; cs[ny * nx + i] = 1; }
    for (let i = 0; i < nx; i++) { // implicit across the layer
      const ip = (i + 1) % nx, im = (i + nx - 1) % nx;
      tb[0] = 1; tc[0] = 0; td[0] = 0;
      for (let j = 1; j < ny; j++) { const o = j * nx, q = o + i, ww = w[q]; ta[j] = -hd * (d2m[j] - ww * d1m[j]); tc[j] = -hd * (d2p[j] - ww * d1p[j]); tb[j] = 1 + hd * (d2m[j] + d2p[j] + ww * d10[j]); td[j] = cs[q] + hd * (idx2 * (cs[o + ip] - 2 * cs[q] + cs[o + im]) - u[q] * i2dx * (cs[o + ip] - cs[o + im])); }
      for (let j = 1; j < ny; j++) { const m = ta[j] / tb[j - 1]; tb[j] -= m * tc[j - 1]; td[j] -= m * td[j - 1]; }
      let prev = 1;
      for (let j = ny - 1; j >= 1; j--) { prev = (td[j] - tc[j] * prev) / tb[j]; c[j * nx + i] = prev; }
    }
    t += dt; steps++; f = flow();
    if (!Number.isFinite(f.nu)) return { nu: NaN, steps, ok: false };
    if (t > 0.6 * tEnd) { nuAvg += f.nu * dt; tAvg += dt; }
    if (t - tLast >= 0.05) { hist.push([t, f.nu]); if (t > 0.3 && Math.abs(f.nu - nuLast) < 3e-7 * (1 + f.nu)) { steady = true; break; } nuLast = f.nu; tLast = t; }
  }
  const out = { nu: steady || tAvg <= 0 ? f.nu : nuAvg / tAvg, steps, t, steady, umax: f.umax, Lx, ok: true, hist, modes: M, nx, ny, h: 1 / Math.sqrt(nx * ny), _c: c, _y: y, _nx: nx };
  if (keep) { // fields on a uniform grid across the layer for plotting (linear interpolation from the clustered grid)
    const yu = Array.from({ length: n1 }, (_, j) => j / ny), row = (a) => { let jj = 0; return yu.map((yy) => { while (jj < ny - 1 && y[jj + 1] < yy) jj++; const fy = clamp((yy - y[jj]) / (y[jj + 1] - y[jj]), 0, 1); return Array.from({ length: nx + 1 }, (_, i) => (1 - fy) * a[jj * nx + (i % nx)] + fy * a[(jj + 1) * nx + (i % nx)]); }); };
    out.x = Array.from({ length: nx + 1 }, (_, i) => i * dx); out.y = yu; out.c = row(c); out.u = row(u); out.w = row(w); out.jw = Array.from(jw); out.yGrid = Array.from(y);
  }
  return out;
}
const ecCache = new Map();
/**
 * Electroconvection characteristics for the stack model, grid-converged by construction. In the scaled problem the Sherwood number depends only
 * on V/V_c, the cut-off length and the cell wavelength (Pe enters through V_c ∝ Pe^-½). Steps: (1) critical wavenumber k_c from the minimum of
 * the marginal curve; (2) scan of cell wavelengths around 2π/k_c on the coarse grid, keeping the one that transports most; (3) solution on
 * three systematically refined grids (ratio 1.5, nested iteration) at 1.3 and 1.7 × threshold, Richardson extrapolation and grid-convergence
 * index. The extrapolated Sherwood number at 1.7 × threshold sets the over-limiting slope used by the stack model.
 */
export function electroconvection(Pe, opt = {}) {
  const ny = clamp(Math.round(opt.ny || 20), 12, 48), ell = clamp(opt.ell ?? 0.2, 0.1, 0.4), key = `${ny}|${ell.toPrecision(4)}`;
  let u = ecCache.get(key);
  if (!u) {
    const cr = ecCritical(1, ell), ns = [Math.max(8, Math.round(ny / 1.5)), ny, Math.round(1.5 * ny)];
    const grid = (k, m) => { const Lx = (2 * Math.PI) / k, M = Math.ceil(3 / (k * ell)), a = Math.max(0.5 * Lx, (2 * M + 2) / ns[0]); return { nx: 2 * Math.ceil((a * m) / 2), ny: m, modes: M }; };
    // the scan grid does not depend on the resolution input, so the selected wavelength does not either
    const scan = [0.63, 0.8, 1, 1.25, 1.6].map((kf) => { const k = kf * cr.kc, M = Math.ceil(3 / (k * ell)), s = ecSolve({ V: 1.7 * cr.Vc, Pe: 1, k, ell, nx: 2 * Math.ceil(Math.max(7 * ((2 * Math.PI) / k), 2 * M + 2) / 2), ny: 14, modes: M, tEnd: 3 }); return { kf, k, nu: Number.isFinite(s.nu) ? s.nu : 1, steady: !!s.steady, sol: s }; });
    const pool = scan.some((q) => q.steady) ? scan.filter((q) => q.steady) : scan, sel = pool.reduce((m, q) => (q.nu > m.nu ? q : m), pool[0]);
    const triplet = (fV, first) => {
      const sols = []; let prev = first;
      for (let l = 0; l < 3; l++) { const s = ecSolve({ V: fV * cr.Vc, Pe: 1, k: sel.k, ell, ...grid(sel.k, ns[l]), init: prev, tEnd: prev ? 3 : 4, keep: l === 2 }); sols.push(s); prev = s; }
      const nu = sols.map((s) => (Number.isFinite(s.nu) ? Math.max(1, s.nu) : 1)), g = gci([sols[2].h, sols[1].h, sols[0].h], [nu[2], nu[1], nu[0]]);
      const ex = Number.isFinite(g.fExact) && g.type === 'monotonic' ? Math.max(1, g.fExact) : nu[2];
      return { f: fV, nu, nuExtrap: ex, gciPct: 100 * g.gciFine, order: Number.isFinite(g.p) ? g.p : 2, monotone: g.type !== 'oscillatory', steady: sols.every((s) => s.steady), grids: sols.map((s) => ({ nx: s.nx, ny: s.ny, h: s.h, steps: s.steps, steady: !!s.steady })), modes: sols[2].modes, fine: sols[2] };
    };
    const hi = triplet(1.7, sel.sol), lo = triplet(1.3, null), fine = hi.fine; delete hi.fine; delete lo.fine;
    u = { ell, kcU: cr.kc, VcU: cr.Vc, kSel: sel.k, scan: scan.map((q) => ({ kf: q.kf, k: q.k, wavelength: (2 * Math.PI) / q.k, nu: q.nu, steady: q.steady })), hi, lo, slopeU: Math.max(0, (hi.nuExtrap - 1) / (0.7 * cr.Vc)),
      field: fine.c ? { x: fine.x, y: fine.y, c: fine.c, u: fine.u, w: fine.w, nu: fine.nu, umax: fine.umax } : null };
    if (ecCache.size > 12) ecCache.clear();
    ecCache.set(key, u);
  }
  const sq = Math.sqrt(Pe), Vc = u.VcU / sq; // V_c ∝ Pe^-½; Sherwood numbers are functions of V/V_c only
  return { Pe, kc: u.kcU, kSel: u.kSel, ell: u.ell, Vc, VcShort: Math.sqrt(32 / Pe), pts: [u.lo, u.hi].map((q) => ({ V: q.f * Vc, nu: q.nuExtrap, steady: q.steady, gciPct: q.gciPct, order: q.order, grids: q.nu })), slope: u.slopeU * sq, scan: u.scan,
    richardson: { ...u.hi, lo: u.lo }, nuExtrap: u.hi.nuExtrap, gciPct: u.hi.gciPct, order: u.hi.order, monotone: u.hi.monotone, steady: u.hi.steady, field: u.field };
}
/** Material electro-osmotic Péclet number Pe = ε(RT/F)² / (ηD). */
export const ecPeclet = (T, D) => (78.4 * (1 - 0.0046 * (T - 25)) * EPS0 * vt(T) ** 2) / (viscosity(T, 0) * D);

// ---- Navier–Stokes / Nernst–Planck in the open channel between two membranes --------------------------------------
/**
 * Developing laminar flow between parallel plates from a uniform inlet profile: boundary-layer (parabolised) Navier–Stokes
 *   u u_x + v u_y = −p′/ρ + ν u_yy,  u_x + v_y = 0,  ∫u dy = U·h,  u = v = 0 at both walls,  p = pOut at the outlet,
 * marched implicitly in x (pressure gradient from the mass constraint), coupled to the electroneutral Nernst–Planck (salt)
 * balance u c_x + v c_y = D c_yy with a prescribed salt flux jw (mol/m²·s) leaving through each wall.
 */
export function channelNS({ U, h, L, T = 25, D, c0 = 1, jw = [0, 0], ny = 48, growth = 1.08, pOut = 0, keep = 12 }) {
  const rho = density(T, 0), mu = viscosity(T, 0), nu = mu / rho, y = Float64Array.from({ length: ny + 1 }, (_, j) => 0.5 * h * (1 - Math.cos((Math.PI * j) / ny)));
  const xs = [0]; let dxq = L * 2e-6; while (xs[xs.length - 1] < L) { xs.push(Math.min(L, xs[xs.length - 1] + dxq)); dxq *= growth; if (L - xs[xs.length - 1] < 0.3 * dxq) xs[xs.length - 1] = L; }
  const n = ny + 1, a = new Array(n).fill(0), b = new Array(n).fill(1), cc = new Array(n).fill(0), wq = (f) => { let s = 0; for (let j = 1; j < n; j++) s += 0.5 * (f[j] + f[j - 1]) * (y[j] - y[j - 1]); return s; };
  let u = Array.from({ length: n }, (_, j) => (j === 0 || j === ny ? 0 : U)), v = new Array(n).fill(0), c = new Array(n).fill(c0), p = 0;
  const out = { x: [], uc: [], dpdx: [], p: [], cw0: [], cw1: [], cb: [], Sh: [], prof: [], y: Array.from(y), rho, mu };
  const kp = new Set(Array.from({ length: keep }, (_, q) => Math.round(((xs.length - 1) * (q + 1)) / keep)));
  for (let s = 1; s < xs.length; s++) {
    const dx = xs[s] - xs[s - 1], r1 = new Array(n).fill(0), r2 = new Array(n).fill(0);
    for (let j = 1; j < ny; j++) {
      const hm = y[j] - y[j - 1], hp = y[j + 1] - y[j], dm = 2 / (hm * (hm + hp)), dp = 2 / (hp * (hm + hp)), am = -hp / (hm * (hm + hp)), ap = hm / (hp * (hm + hp)), a0 = (hp - hm) / (hm * hp);
      a[j] = v[j] * am - nu * dm; cc[j] = v[j] * ap - nu * dp; b[j] = u[j] / dx + v[j] * a0 + nu * (dm + dp); r1[j] = (u[j] * u[j]) / dx; r2[j] = -1;
    }
    a[0] = 0; b[0] = 1; cc[0] = 0; a[ny] = 0; b[ny] = 1; cc[ny] = 0;
    const ua = tridiag(a, b, cc, r1), ub = tridiag(a, b, cc, r2), P = (U * h - wq(ua)) / wq(ub), un = ua.map((q, j) => q + P * ub[j]); // P = p′/ρ
    const vn = new Array(n).fill(0); for (let j = 1; j < n; j++) vn[j] = vn[j - 1] - 0.5 * ((un[j] - u[j]) / dx + (un[j - 1] - u[j - 1]) / dx) * (y[j] - y[j - 1]);
    // salt: same operator with the new velocity field, flux boundary conditions at both walls
    const rc = new Array(n).fill(0);
    for (let j = 1; j < ny; j++) {
      const hm = y[j] - y[j - 1], hp = y[j + 1] - y[j], dm = 2 / (hm * (hm + hp)), dp = 2 / (hp * (hm + hp)), am = -hp / (hm * (hm + hp)), ap = hm / (hp * (hm + hp)), a0 = (hp - hm) / (hm * hp);
      a[j] = vn[j] * am - D * dm; cc[j] = vn[j] * ap - D * dp; b[j] = un[j] / dx + vn[j] * a0 + D * (dm + dp); rc[j] = (un[j] * c[j]) / dx;
    }
    a[0] = 0; b[0] = D / (y[1] - y[0]); cc[0] = -D / (y[1] - y[0]); rc[0] = -jw[0]; a[ny] = -D / (y[ny] - y[ny - 1]); b[ny] = D / (y[ny] - y[ny - 1]); cc[ny] = 0; rc[ny] = -jw[1];
    c = tridiag(a, b, cc, rc); u = un; v = vn; p += rho * P * dx;
    const cb = wq(u.map((q, j) => q * c[j])) / (U * h), jm = 0.5 * (jw[0] + jw[1]), cwm = 0.5 * (c[0] + c[ny]);
    out.x.push(xs[s]); out.uc.push(Math.max(...u) / U); out.dpdx.push(rho * P); out.p.push(p); out.cw0.push(c[0]); out.cw1.push(c[ny]); out.cb.push(cb); out.Sh.push(jm !== 0 && cb - cwm !== 0 ? (jm / (cb - cwm)) * ((2 * h) / D) : NaN);
    if (kp.has(s)) out.prof.push({ x: xs[s], u: u.map((q) => q / U), c: [...c] });
  }
  const dpTot = -p, dpFd = (12 * mu * U * L) / (h * h), iE = out.uc.findIndex((q) => q >= 0.99 * 1.5), Re = (rho * U * 2 * h) / mu;
  let shInt = 0; for (let s = 1; s < out.x.length; s++) if (Number.isFinite(out.Sh[s]) && Number.isFinite(out.Sh[s - 1])) shInt += 0.5 * (out.Sh[s] + out.Sh[s - 1]) * (out.x[s] - out.x[s - 1]);
  return { ...out, p: out.p.map((q) => pOut + (q - p)), pIn: pOut + dpTot, dp: dpTot, dpFd, Kinc: (dpTot - dpFd) / (0.5 * rho * U * U), fRe: (-out.dpdx[out.dpdx.length - 1] * 2 * (2 * h) * (2 * h)) / (mu * U), entrance: iE >= 0 ? out.x[iE] : L, developed: iE >= 0, Re, Sc: nu / D,
    ShMean: shInt / (out.x[out.x.length - 1] - out.x[0]), ShEnd: out.Sh[out.Sh.length - 1], cwMin: Math.min(...out.cw0, ...out.cw1), saltIn: U * h * c0, saltOut: U * h * out.cb[out.cb.length - 1] + (jw[0] + jw[1]) * L };
}

// ---- full 2-D Navier–Stokes + salt transport in the (spacer-filled) channel: finite-volume solver of suite 4 --------------
/**
 * Two-dimensional steady incompressible Navier–Stokes equations (SIMPLE-type pressure–velocity coupling on a staggered finite-volume grid, suite 4)
 * with the convection–diffusion equation of the salt in one ED channel of height h. Geometry: open channel over the whole flow path (arr = 'none'), or
 * nFil pitches of transverse spacer filaments (zigzag, cavity or submerged; diameter df·h, pitch lm·h) as immersed solids. Both membranes take the
 * salt flux jw (mol/m²·s) out of the solution — the wall condition equivalent to the current density, jw = (t̄ − t)·i/F. The salt equation is linear,
 * so the solved field scales with jw: the Sherwood number does not depend on it. U is the superficial velocity (flow ÷ full cross-section).
 * Returns local and mean Sherwood numbers (2h basis), the pressure gradient over whole pitches, friction factor and the fields.
 */
export async function channelCFD({ U, h, L, T = 25, D, c0 = 1, jw = 0, arr = 'none', df = 0.5, lm = 4, nFil = 8, nx = 0, ny = 24, stretch = 4, maxIter = 500, tol = 2e-5, scalIter = 300 }, ctx) {
  const rho = density(T, 0), mu = viscosity(T, 0), spacer = arr !== 'none', Lc = spacer ? nFil * lm * h : L, nxx = nx > 0 ? Math.round(nx) : spacer ? nFil * Math.max(12, Math.round((20 * ny) / 24)) : Math.max(60, 5 * ny), g = yGrid(h, ny, stretch);
  const mk = buildMask({ type: 'spacer', arr, L: Lc, H: h, df: df * h, lm: lm * h, nFil }, nxx, ny, g.yc), jr = jw !== 0 ? jw : (1e-3 * D * c0) / h;
  const r = await solveChannel({ L: Lc, H: h, nx: nxx, ny, stretch, solid: mk.solid, rho, mu, Uin: U, inlet: 'parabolic', scheme: 'hybrid', steady: true, maxIter, tol, alphaU: 0.7, scalIter, species: { c0, D, A: 0, B: 0, dP: 0, pi: () => 0, cw: -jr, bot: 'flux', top: 'flux' } }, ctx);
  const { dx, dy, u, v, p, nu1, solid } = r, phi = r.spc.phi, xc = new Array(nxx), cb = new Array(nxx), pm = new Array(nxx), cwB = Array.from(r.spc.wB), cwT = Array.from(r.spc.wT), ShLoc = new Array(nxx);
  for (let i = 0; i < nxx; i++) {
    let q = 0, qc = 0, ps = 0, m = 0;
    for (let j = 0; j < ny; j++) { const P = j * nxx + i; if (solid[P]) continue; const uc = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); q += uc * dy[j]; qc += uc * phi[P] * dy[j]; ps += p[P] * dy[j]; m += dy[j]; }
    xc[i] = (i + 0.5) * dx; cb[i] = Math.abs(q) > 1e-9 * U * h ? qc / q : c0; pm[i] = m ? ps / m : 0;
  }
  // averaging planes: whole pitches away from the inlet and outlet (spacer) or the whole path (open channel, stage-mean Sherwood number)
  let i1 = 1, i2 = nxx - 2, ip1 = Math.round(0.5 * nxx), ip2 = nxx - 2;
  if (spacer && nFil >= 4) { i1 = ip1 = Math.round((2 * lm * h) / dx); i2 = ip2 = Math.round((Lc - lm * h) / dx) - 1; }
  let fl = 0, dr = 0, cwMin = Infinity;
  for (let i = 0; i < nxx; i++) {
    let sm = 0, m = 0;
    for (const top of [0, 1]) { const P = (top ? ny - 1 : 0) * nxx + i; if (solid[P]) continue; const cw = (top ? cwT : cwB)[i], d = cb[i] - cw; cwMin = Math.min(cwMin, cw); if (i >= i1 && i <= i2) { fl += jr; dr += d; } if (d > 0) { sm += jr / d; m++; } }
    ShLoc[i] = m ? ((sm / m) * 2 * h) / D : null;
  }
  const dpdx = (pm[ip1] - pm[ip2 + (spacer && nFil >= 4 ? 1 : 0)]) / ((ip2 + (spacer && nFil >= 4 ? 1 : 0) - ip1) * dx), dh = 2 * h, Re = (rho * U * dh) / mu, f = (dpdx * dh) / (0.5 * rho * U * U);
  let saltOut = 0, wallOut = 0; for (let j = 0; j < ny; j++) saltOut += u[j * nu1 + nxx] * phi[j * nxx + nxx - 1] * dy[j];
  for (let i = 0; i < nxx; i++) for (const top of [0, 1]) if (!solid[(top ? ny - 1 : 0) * nxx + i]) wallOut += jr * dx;
  // cell-centred fields as rows for plotting
  const rows = (fn) => Array.from({ length: ny }, (_, j) => Array.from({ length: nxx }, (_, i) => fn(j, i))), ucell = (j, i) => 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]), vcell = (j, i) => 0.5 * (v[j * nxx + i] + v[(j + 1) * nxx + i]);
  return { arr, spacer, U, h, Lc, D, c0, jw: jr, rho, mu, Re, Sc: mu / (rho * D), dh, Sh: dr > 0 ? ((fl / dr) * dh) / D : 8.235, ShLoc, dpdx, dpTot: pm[0] - pm[nxx - 1], f, fRe: f * Re, xc, yc: Array.from(g.yc), cb, cwB, cwT, pm, cwMin,
    saltIn: U * h * c0, saltOut: saltOut + wallOut, converged: !!r.converged, iters: r.iters, scalRes: r.scalRes, nx: nxx, ny, solidFraction: mk.solidFraction, shapes: mk.shapes,
    fields: { uu: rows(ucell), vv: rows(vcell), speed: rows((j, i) => (solid[j * nxx + i] ? 0 : Math.hypot(ucell(j, i), vcell(j, i)))), c: rows((j, i) => phi[j * nxx + i]), mask: rows((j, i) => !!solid[j * nxx + i]) } };
}
const cfdCache = new Map();
const cfdGeomKey = (p, G) => `${p.spArr || 'zigzag'}|${(+(p.spDf ?? 0.5)).toPrecision(4)}|${(+(p.spLm ?? 4)).toPrecision(4)}|${G.h.toPrecision(5)}|${G.eps.toPrecision(4)}|${G.L.toPrecision(5)}|${Math.round(p.cfdNy || 24)}`;
/** Solve (or fetch) the 2-D channel solution for the run conditions; the stack model picks it up through attachModels. u0 = mean velocity in the open cross-section of the stack model. */
export async function prepareChannelCFD(p, ctx) {
  const G = geometry(p), T = p.T, cf = toMolar(balanceCharge(scaleIons(cloneIons(p.ions), p.salinityFactor ?? 1))), Ds = electrolyte(cf, T).Ds;
  const u0 = Math.max(p.mode === 'design' || p.mode === 'batch' ? p.uLin / 100 : p.Qp / 3600 / Math.max(1, Math.round(p.Ncp)) / (G.W * G.h * G.eps), 1e-4), gk = cfdGeomKey(p, G), key = `${gk}|${u0.toPrecision(5)}|${T}|${Ds.toPrecision(5)}`;
  let hit = cfdCache.get(key);
  if (!hit) {
    const arr = p.spArr || 'zigzag', ny = clamp(Math.round(p.cfdNy || 24), 12, 48);
    const sol = await channelCFD({ U: u0 * G.eps, h: G.h, L: G.L, T, D: Ds, arr, df: clamp(p.spDf ?? 0.5, 0.2, 0.8), lm: clamp(p.spLm ?? 4, 2, 12), ny }, ctx);
    const rho = density(T, 0);
    hit = { cfd: true, gk, Sh: sol.Sh, Re: (rho * u0 * 2 * G.h) / sol.mu, Sc: sol.Sc, dpPerM: sol.dpdx, U: u0, mu: sol.mu, nRe: sol.spacer ? clamp(p.shB ?? 0.5, 0.3, 0.9) : 1 / 3, nDp: sol.spacer ? 1.7 : 1, ShMin: sol.spacer ? 3 : 8.235, sol };
    if (cfdCache.size > 24) cfdCache.clear();
    cfdCache.set(key, hit);
  }
  cfdCache.set(`last|${gk}`, hit);
  return hit;
}

// ---- fouling and scaling of the stack (reduced-order kinetics) ----------------------------------------------------
/**
 * Membrane fouling and scaling at constant current: organic deposit m_f (g/m², anion membrane) and mineral scale m_s (g/m², concentrate side)
 *   dm_f/dt = k_dep·c_f·i_eff/100 − (k_det·u/u_ref + k_rev)·m_f,   dm_s/dt = k_s·[(S_g − 1)₊² + (√S_c − 1)₊²] − k_rev·m_s,   i_eff = i/(1 − b),
 * with the blocked area fraction b = m_s/(m_s + m_b) and polarity reversal homogenised to k_rev = −ln(1 − η)/τ. Integrated with RK4 (time in days).
 */
export function foulingED({ i, u, cFoul, kDep, kDet, rFoul, Sg, Sc, kScale, rScale, mBlock, edr, revInterval, revEff, days, n = 200 }) {
  const kRev = edr ? (-Math.log(1 - clamp(revEff, 0, 0.999)) / revInterval) * 1440 : 0, sup = Math.max(0, Sg - 1) ** 2 + Math.max(0, Math.sqrt(Math.max(Sc, 0)) - 1) ** 2;
  const rhs = (t, y) => { const b = Math.max(y[1], 0) / (Math.max(y[1], 0) + mBlock); return [kDep * cFoul * (i / (1 - b) / 100) - (kDet * (u / 0.08) + kRev) * y[0], kScale * sup - kRev * y[1]]; };
  const sub = clamp(Math.ceil((2 * days * (kDet * (u / 0.08) + kRev + (kScale * sup) / mBlock)) / n), 1, 400), sol = rk4(rhs, [0, 0], 0, days, n * sub); // sub-steps keep the explicit integration stable for fast removal rates
  const t = [], mf = [], ms = []; for (let k = 0; k <= n; k++) { t.push(sol.t[k * sub]); mf.push(Math.max(sol.y[k * sub][0], 0)); ms.push(Math.max(sol.y[k * sub][1], 0)); }
  const block = ms.map((m) => m / (m + mBlock));
  const dR = mf.map((m, k) => (rFoul * m + rScale * ms[k]) * 1e-4), dU = dR.map((r, k) => (i * r) / (1 - block[k]) + 0); // Ω·m² and V per cell pair
  return { t, mf, ms, block, dR, dU, kRev, sup, mfInf: kDet * (u / 0.08) + kRev > 0 ? (kDep * cFoul * (i / 100)) / (kDet * (u / 0.08) + kRev) : Infinity };
}

// ---- kernel (Gaussian-process type) surrogate of the mechanistic model ------------------------------------------------
/** Kernel ridge regression with a squared-exponential kernel on inputs scaled to the unit cube; length scale by leave-one-out error. */
export function kernelFit(X, y, lam = 1e-6) {
  const n = X.length, mean = sum(y) / n, yc = Float64Array.from(y, (v) => v - mean), D2 = new Float64Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < r; c++) { let q = 0; const a = X[r], b = X[c]; for (let k = 0; k < a.length; k++) q += (a[k] - b[k]) ** 2; D2[r * n + c] = D2[c * n + r] = q; }
  const L = new Float64Array(n * n), Li = new Float64Array(n * n), z = new Float64Array(n);
  let best = null;
  for (const ell of [0.25, 0.4, 0.6, 0.9, 1.4]) {
    // Cholesky factor K = L·Lᵀ, then L⁻¹: α = K⁻¹y and diag(K⁻¹) = column norms of L⁻¹ give the leave-one-out residuals α_q / (K⁻¹)_qq in O(n³)
    const g = -1 / (2 * ell * ell); let ok = true;
    for (let r = 0; r < n && ok; r++) for (let c = 0; c <= r; c++) {
      let q = Math.exp(g * D2[r * n + c]) + (r === c ? lam : 0);
      for (let k = 0; k < c; k++) q -= L[r * n + k] * L[c * n + k];
      if (r === c) { if (!(q > 1e-14)) { ok = false; break; } L[r * n + r] = Math.sqrt(q); } else L[r * n + c] = q / L[c * n + c];
    }
    if (!ok) continue;
    Li.fill(0);
    for (let c = 0; c < n; c++) { Li[c * n + c] = 1 / L[c * n + c]; for (let r = c + 1; r < n; r++) { let q = 0; for (let k = c; k < r; k++) q -= L[r * n + k] * Li[k * n + c]; Li[r * n + c] = q / L[r * n + r]; } }
    for (let r = 0; r < n; r++) { let q = 0; for (let k = 0; k <= r; k++) q += Li[r * n + k] * yc[k]; z[r] = q; } // z = L⁻¹y
    const alpha = new Array(n); let loo = 0;
    for (let c = 0; c < n; c++) { let a = 0, dg = 0; for (let r = c; r < n; r++) { const v = Li[r * n + c]; a += v * z[r]; dg += v * v; } alpha[c] = a; loo += (a / dg) ** 2; }
    if (Number.isFinite(loo) && (!best || loo < best.loo)) best = { ell, alpha, loo };
  }
  if (!best) throw new Error('The surrogate could not be trained (singular kernel matrix).');
  const gb = -1 / (2 * best.ell * best.ell);
  return { ...best, rmseLoo: Math.sqrt(best.loo / n), predict: (x) => { let q = mean; for (let r = 0; r < n; r++) { let d = 0; const a = X[r]; for (let k = 0; k < a.length; k++) d += (a[k] - x[k]) ** 2; q += best.alpha[r] * Math.exp(gb * d); } return q; } };
}

/** GHK permeabilities (m²/s, relative) of the cation and anion membranes for the tracked ions: counter-ions by diffusivity × selectivity; co-ions scaled so that the small-ratio slope for a single salt equals the permselectivity. */
function membranePermeabilities(c, par) {
  let eC = 0, eA = 0, pc = 0, pa = 0, pcs = 0, pas = 0;
  const sel = Z.map((zj, j) => (DIV[j] ? (zj > 0 ? par.selDivC : par.selDivA) : 1));
  for (let j = 0; j < NI; j++) { const e = AZ[j] * c[j]; if (Z[j] > 0) { eC += e; pc += e * DI[j]; pcs += e * DI[j] * sel[j]; } else { eA += e; pa += e * DI[j]; pas += e * DI[j] * sel[j]; } }
  const mc = eC > 0 ? pc / eC : 1.33e-9, ma = eA > 0 ? pa / eA : 2.03e-9, mcs = eC > 0 ? pcs / eC : mc, mas = eA > 0 ? pas / eA : ma;
  const rC = ((1 - par.alphaC) / (1 + par.alphaC)) * (mcs / ma), rA = ((1 - par.alphaA) / (1 + par.alphaA)) * (mas / mc);
  return { PC: Z.map((zj, j) => DI[j] * (zj > 0 ? sel[j] : rC)), PA: Z.map((zj, j) => DI[j] * (zj < 0 ? sel[j] : rA)) };
}

// ---- coupling of the membrane-scale models to the stack ----------------------------------------------------------
/** Reference Navier–Stokes / Nernst–Planck solution of the open channel at velocity U (unit wall flux: the Sherwood number does not depend on it). */
const nsCache = new Map();
function nsReference(U, G, T, Ds, pOut = 0) {
  const key = `${U}|${G.h}|${G.L}|${T}|${Ds}|${pOut}`; let hit = nsCache.get(key); // the same reference state recurs in every re-design and sweep of a run
  if (hit) return hit;
  const s = channelNS({ U, h: G.h, L: G.L, T, D: Ds, c0: 1, jw: [(1e-3 * Ds) / G.h, (1e-3 * Ds) / G.h], pOut });
  hit = { Sh: s.ShMean, Re: s.Re, Sc: s.Sc, dpPerM: s.dp / G.L, U, mu: s.mu, sol: s };
  if (nsCache.size > 60) nsCache.clear();
  nsCache.set(key, hit);
  return hit;
}
/** Attach the optional sub-models (channel Navier–Stokes, electroconvection) to the parameter set of a run. */
function attachModels(p, par, G, T, cf, u0) {
  const Ds = electrolyte(cf, T).Ds;
  if (p.flowModel === 'ns') par.ns = nsReference(Math.max(u0, 1e-4), G, T, Ds, (p.pOut || 0) * 1e5);
  else if (p.flowModel === 'cfd2d') { // 2-D solution prepared by the (asynchronous) run; synchronous callers reuse the latest solution of the same geometry, rescaled in channel()
    par.ns = cfdCache.get(`last|${cfdGeomKey(p, G)}`) || null;
    if (!par.ns) { par.cfdMissing = true; if ((p.spArr || 'zigzag') === 'none') par.ns = nsReference(Math.max(u0, 1e-4), G, T, Ds, (p.pOut || 0) * 1e5); }
  }
  if (p.olModel === 'rz') par.ol = electroconvection(ecPeclet(T, Ds), { ny: p.ecN, ell: p.ecEll });
  return par;
}
/** Electrochemical–thermal coupling: stack energy balance (Joule and polarisation heat, pumping dissipation, heat loss) iterated with the temperature-dependent conductivity, diffusivity and thermal voltage. */
export function simulateEDThermal(v, ov = {}) {
  const Tin = v.T, loss = clamp((v.thLoss || 0) / 100, 0, 1), hist = [];
  let T = Tin, r = null, iso = null, th = null;
  for (let it = 0; it < 8; it++) {
    r = simulateED(v, { ...ov, T });
    if (!iso) iso = { sec: r.sec, U: r.stages.map((s) => s.U), ncp: r.Ncp * r.nSt, kappa: electrolyte(r.cf, T).kappa };
    const Wsep = r.wMin * 3.6e6 * r.Qprod, Qgen = Math.max(0, r.Pdc - Wsep) + r.Ppump * (v.etaPump / 100), mcp = r.tr.Qf * density(T, 0) * cpWater(T, 0), dT = (Qgen * (1 - loss)) / mcp, Tn = Math.min(Tin + 0.5 * dT, 60);
    th = { Tin, Tmean: Tn, Tout: Tin + dT, dT, Qgen, Wsep, mcp, loss, iterations: it + 1, iso, kappa: electrolyte(r.cf, Tn).kappa };
    hist.push([it + 1, T, dT, r.sec]);
    if (Math.abs(Tn - T) < 5e-3) break;
    T = Tn;
  }
  th.hist = hist;
  return { ...r, th };
}
/** Single-pass rating of an nSt-stage stack at the same voltage on every stage (used for maps, the surrogate and the optimiser). */
export function rateStack(cf, cc, U, uCm, nSt, G, par, T, nSeg = 4) {
  const q = (uCm / 100) * G.W * G.h * G.eps; let s = { qd: q, qc: q, cd: cf, cc }, E = 0, I = 0;
  for (let k = 0; k < nSt; k++) { const m = marchStage(s, U, G, par, T, nSeg); E += U * m.I; I += m.I; s = m.st; }
  const t0 = tdsOf(cf), t1 = tdsOf(s.cd);
  return { removal: 1 - t1 / t0, sec: E / (s.qd * 3.6e6), iAvg: I / (nSt * G.W * G.L), cd: s.cd, areaSpec: (2 * nSt * G.W * G.L) / (s.qd * 3600) };
}
/** Names of the surrogate learners compared by stackSurrogate. */
export const SURROGATES = { kernel: 'Kernel ridge regression', gp: 'Gaussian process', nn: 'Neural network' };
/**
 * Train, compare and test three surrogates of the stack rating on the same design points: kernel ridge regression (squared-exponential kernel,
 * leave-one-out length scale), a Gaussian process (anisotropic kernel, hyper-parameters by maximum marginal likelihood) and a feed-forward neural
 * network (two tanh layers, Adam, early stopping). Inputs: feed-salinity multiplier, voltage per cell pair and velocity; outputs ln(SEC) and salt removal.
 * Selection: every learner is first trained on three quarters of the design and scored on the remaining quarter (validation); the learner with the
 * lowest validation error is selected per output and retrained on the whole design. A separate Latin-hypercube set, never seen during training or
 * selection, gives the held-out errors of all three. Deterministic for a given seed.
 */
export function stackSurrogate(r, nTrain = 48, nTest = 16, seed = 11) {
  const rg = { s: [0.4, 2.5], U: [0.25, 1.2], u: [3, 16] }, cc0 = r.tr.ccIn, T = r.T;
  const at = (x) => { const sf = rg.s[0] * (rg.s[1] / rg.s[0]) ** x[0], U = rg.U[0] + (rg.U[1] - rg.U[0]) * x[1], u = rg.u[0] * (rg.u[1] / rg.u[0]) ** x[2], q = rateStack(r.cf.map((c) => c * sf), cc0.map((c) => c * sf), U, u, r.nSt, r.G, r.par, T); return { x, sf, U, u, sec: Math.max(q.sec, 1e-9), removal: q.removal }; };
  const toX = (sf, U, u) => [Math.log(sf / rg.s[0]) / Math.log(rg.s[1] / rg.s[0]), (U - rg.U[0]) / (rg.U[1] - rg.U[0]), Math.log(u / rg.u[0]) / Math.log(rg.u[1] / rg.u[0])];
  const tr = lhs(nTrain, 3, seed).map(at), te = lhs(nTest, 3, seed + 18).map(at), X = tr.map((q) => q.x), Xt = te.map((q) => q.x);
  const stat = (a, b) => { const m = sum(a) / a.length, ss = sum(a.map((v) => (v - m) ** 2)), se = sum(a.map((v, k) => (v - b[k]) ** 2)); return { r2: ss > 0 ? 1 - se / ss : 1, rmse: Math.sqrt(se / a.length) }; };
  const isVal = (k) => k % 4 === 3, Xf = X.filter((_, k) => !isVal(k)), Xv = X.filter((_, k) => isVal(k)), targets = { sec: tr.map((q) => Math.log(q.sec)), removal: tr.map((q) => q.removal) }, back = { sec: Math.exp, removal: (y) => y };
  const fitters = {
    kernel: (A, y) => { const m = kernelFit(A, y); return { predict: m.predict, info: m }; },
    gp: (A, y, o = {}) => { const m = gpFit(A, y, { maxIter: o.theta0 ? 60 : 160, theta0: o.theta0 }); return { predict: (x) => m.predict(x).mean, info: m }; },
    nn: (A, y, o = {}) => { const m = nnTrain(A, y, { hidden: [10, 8], epochs: o.epochs || 700, lr: 0.02, seed, Xval: o.Xval, yval: o.yval, patience: 120, l2: 1e-5, batch: 8 }); return { predict: m.predict, info: m }; },
  };
  const models = {}, best = {}, fin = {};
  for (const key of ['sec', 'removal']) {
    const y = targets[key], yf = y.filter((_, k) => !isVal(k)), yv = y.filter((_, k) => isVal(k)), ytrue = te.map((q) => q[key]);
    for (const name of Object.keys(fitters)) {
      let val = Infinity, test = { r2: 0, rmse: Infinity }, final = null, predT = ytrue.map(() => NaN), ok = false;
      try {
        const m1 = fitters[name](Xf, yf, { Xval: Xv, yval: yv }); val = stat(yv, Xv.map(m1.predict)).rmse; // selection stage: fit on 75 %, score on 25 %
        final = name === 'nn' ? fitters.nn(X, y, { epochs: Math.max(60, Math.round(1.15 * (m1.info.bestEpoch + 1))) }) : name === 'gp' ? fitters.gp(X, y, { theta0: m1.info.theta }) : fitters.kernel(X, y); // final stage: all design points
        predT = Xt.map((x) => back[key](final.predict(x))); test = stat(ytrue, predT); ok = predT.every(Number.isFinite) && Number.isFinite(val);
      } catch { ok = false; }
      (models[name] ||= { name: SURROGATES[name] })[key] = { val: ok ? val : Infinity, ...test, ok, pred: predT, final };
    }
    best[key] = Object.keys(fitters).filter((n) => models[n][key].ok).reduce((m, n) => (m === null || models[n][key].val < models[m][key].val ? n : m), null);
    if (!best[key]) throw new Error('none of the surrogate learners could be trained');
    fin[key] = models[best[key]][key].final;
  }
  const mS = models.kernel.sec.final?.info || kernelFit(X, targets.sec), mR = models.kernel.removal.final?.info || kernelFit(X, targets.removal);
  const pred = (sf, U, u) => { const x = toX(sf, U, u); return { sec: Math.exp(fin.sec.predict(x)), removal: fin.removal.predict(x) }; };
  const pt = te.map((q, k) => ({ ...q, p: { sec: models[best.sec].sec.pred[k], removal: models[best.removal].removal.pred[k] } })), ptrK = tr.map((q) => Math.exp(mS.predict(q.x)));
  for (const n of Object.keys(models)) for (const key of ['sec', 'removal']) delete models[n][key].final;
  return { rg, train: tr, test: pt, pred, mS, mR, models, best, nFit: Xf.length, nVal: Xv.length, seed, sec: stat(pt.map((q) => q.sec), pt.map((q) => q.p.sec)), removal: stat(pt.map((q) => q.removal), pt.map((q) => q.p.removal)), fitSec: stat(tr.map((q) => q.sec), ptrK), at: (sf, U, u) => at(toX(sf, U, u)) };
}
/** Unit cost used by the operating-condition optimiser: electricity plus straight-line membrane replacement, $/m³ of product. */
export const edCost = (r, p) => p.cElecED * r.sec + (p.cMem * r.area) / (Math.max(p.memLife, 0.1) * 8760 * 0.9 * r.Qprod * 3600);
/**
 * Operating fraction of the limiting current and linear velocity for the lowest unit cost (design mode). A coarse grid of complete re-designs gives the
 * cost map (backdrop) and the starting point; a bounded Nelder–Mead simplex then searches the continuous variables. The cost steps where the number of
 * stages changes, so the best design met anywhere (grid or simplex) is returned.
 */
export function optimiseED(v, phis = [40, 55, 70, 85, 95], us = [4, 7, 10, 14], refine = true) {
  const cells = [], z = us.map(() => phis.map(() => null));
  const design = (ph, u) => { try { const q = simulateED({ ...v, mode: 'design' }, { safety: ph, uLin: u, nSeg: Math.min(Math.max(2, Math.round(v.nSeg)), 4), tol: 1e-4 }); if (q.reached) return { phi: ph, u, cost: edCost(q, v), sec: q.sec, area: q.area, nSt: q.nSt, Ncp: q.Ncp }; } catch { /* infeasible design */ } return null; };
  us.forEach((u, a) => phis.forEach((ph, b) => { const c = design(ph, u); if (c) { cells.push(c); z[a][b] = c.cost; } }));
  const gridBest = cells.reduce((m, c) => (!m || c.cost < m.cost ? c : m), null), path = [];
  let best = gridBest, evals = us.length * phis.length, iterations = 0;
  if (refine && gridBest) {
    const lo = [Math.min(...phis), Math.min(...us)], hi = [Math.max(...phis), Math.max(...us)], worst = Math.max(...cells.map((c) => c.cost));
    const f = (x) => { const c = design(x[0], x[1]); evals++; if (!c) return 10 * worst; path.push(c); if (c.cost < best.cost) best = c; return c.cost; };
    iterations = nelderMead(f, [gridBest.phi, gridBest.u], { lo, hi, tol: 2e-4, maxIter: 18, scale: 0.07 }).iterations;
  }
  return { cells, best, gridBest, phis, us, z, evals, iterations, path, refined: !!(refine && gridBest) };
}

/** Results of the optional membrane-scale, thermal, fouling, surrogate and optimisation models for a continuous ED run. */
function advancedED(r, p, W) {
  const kpis = [], plots = [], tables = [], recs = [], out = {}, bal = [], T = r.T, st = r.stages, seg0 = st[0].segs[0], e0 = electrolyte(r.cf, T), eC = electrolyte(r.tr.ccIn, T), V = vt(T), i0 = Math.max(seg0.iFilm, 1e-6), delta = seg0.k > 0 ? e0.Ds / seg0.k : 5e-5;
  const X = p.Xfix * 1000, dm = p.dMem * 1e-6, Dsol = [e0.Dc, e0.Da], Dmem = Dsol.map((d) => 0.05 * d), ratio = clamp(p.pnpRatio || 1.25, 1.05, 2);
  // ---- always shown: GHK membrane potentials and electrode kinetics at the stack inlet
  const pm = membranePermeabilities(r.cf, r.par), gC = ghkPotential(pm.PC, Z, r.cf, r.tr.ccIn, T), gA = -ghkPotential(pm.PA, Z, r.cf, r.tr.ccIn, T), lnr = Math.log(eC.ceq / e0.ceq);
  tables.push({ title: 'Membrane potential and electrode kinetics (stack inlet)', columns: ['Quantity', 'Value', 'Unit'], rows: [
    ['Cation membrane · Goldman–Hodgkin–Katz zero-current potential', 1000 * gC, 'mV'], ['Cation membrane · permselectivity × Nernst potential', 1000 * V * r.par.alphaC * lnr, 'mV'], ['Anion membrane · Goldman–Hodgkin–Katz zero-current potential', 1000 * gA, 'mV'], ['Anion membrane · permselectivity × Nernst potential', 1000 * V * r.par.alphaA * lnr, 'mV'],
    ['Model used in the voltage balance', r.par.membModel === 'ghk' ? 'Goldman–Hodgkin–Katz' : 'Permselectivity × Nernst', ''], ['Anode overpotential', 1000 * overpotential(r.iAvg, r.par.i0a, T, r.par.aBV, r.par.kinetics), 'mV'], ['Cathode overpotential', 1000 * overpotential(r.iAvg, r.par.i0c, T, r.par.aBV, r.par.kinetics), 'mV'],
    ['Tafel slope 2.303·RT/(αF)', (1000 * Math.LN10 * V) / r.par.aBV, 'mV per decade'], ['Kinetic law', r.par.kinetics === 'tafel' ? 'Tafel' : 'Butler–Volmer', '']],
    note: 'The constant-field (GHK) potential is solved ion by ion for the inlet diluate and concentrate; it equals the permselectivity form for a single salt at small concentration ratios and adds the bi-ionic contribution of mixed feeds.' });
  if (r.par.membModel === 'ghk') kpis.push({ label: 'GHK membrane potential (inlet, both membranes)', value: 1000 * (gC + gA), unit: 'mV', help: 'Zero-current potential of the cation plus the anion membrane from the Goldman–Hodgkin–Katz equation' });
  // ---- electroconvection
  if (r.par.ol) {
    const ol = r.par.ol, Vs = linspace(0, 2.2 * ol.Vc, 23), rh = ol.richardson, fld = ol.field, lam = (2 * Math.PI) / ol.kSel, gst = ol.gciPct < 5 && ol.monotone ? 'ok' : 'warn';
    kpis.push({ label: 'Electroconvection threshold', value: ol.Vc * V, unit: 'V per depleted layer', help: `${fmt(ol.Vc, 3)} thermal voltages; electro-osmotic Péclet number ${fmt(ol.Pe, 3)}` }, { label: 'Over-limiting slope d(i/i_lim)/dV', value: ol.slope / V, unit: '1/V', help: 'From the Richardson-extrapolated Sherwood number at 1.7 × threshold' },
      { label: 'Over-limiting current at 1.7 × threshold (grid-extrapolated)', value: ol.nuExtrap, unit: '× i_lim', help: `Richardson extrapolation of three grids: ${rh.nu.map((q) => fmt(q, 5)).join(' → ')}` },
      { label: 'Numerical uncertainty of the over-limiting current (GCI)', value: ol.gciPct, unit: '%', status: gst, help: `Grid-convergence index of the fine grid (safety factor 1.25); observed order ${fmt(ol.order, 3)}` },
      { label: 'Observed order of grid convergence', value: ol.order, unit: '–', help: 'Formal order of the scheme: 2' }, { label: 'Vortex-pair wavelength', value: lam, unit: '× δ', help: `Most-transporting cell of the wavelength scan; critical wavelength ${fmt((2 * Math.PI) / ol.kc, 3)} δ` });
    if (!ol.monotone || ol.gciPct > 5) W.push({ level: 'warn', msg: `Electroconvection: the three-grid sequence is ${ol.monotone ? 'monotone' : 'not monotone'} with a grid-convergence index of ${fmt(ol.gciPct, 3)} % — raise the number of grid cells across the layer.` });
    if (!ol.steady) W.push({ level: 'info', msg: 'Electroconvection: the vortex solution is unsteady on at least one grid; its Sherwood number is a time average.' });
    plots.push({ type: 'line', title: 'Over-limiting current from electroconvection (Rubinstein–Zaltzman slip model)', xlabel: 'Voltage across one depleted diffusion layer (V)', ylabel: 'i / i_lim', series: [{ name: 'Stack model: 1 + slope·(V − V_c)', x: Vs.map((x) => x * V), y: Vs.map((x) => 1 + ol.slope * Math.max(0, x - ol.Vc)) }, { name: 'Non-linear vortex solutions (grid-extrapolated)', x: [ol.Vc, ...ol.pts.map((q) => q.V)].map((x) => x * V), y: [1, ...ol.pts.map((q) => q.nu)], mode: 'points' }], vlines: [{ x: ol.Vc * V, label: 'instability threshold' }], note: `Vortex pair of wavelength ${fmt(lam, 3)} δ; slip cut-off length ${fmt(ol.ell, 3)} δ. Without the cut-off the marginal curve has no minimum (short-wave limit √(32/Pe) = ${fmt(Math.sqrt(32 / ol.Pe), 3)} thermal voltages).` });
    plots.push({ type: 'line', title: 'Electroconvection: grid convergence and wavelength scan', xlabel: 'Representative cell size h = 1/√(n_x·n_y)  ·  or wavelength ÷ 10 δ', ylabel: 'i / i_lim at 1.7 × threshold', series: [{ name: 'Three grids', x: rh.grids.map((g) => g.h), y: rh.nu, mode: 'both' }, { name: 'Richardson extrapolation (h → 0)', x: [0], y: [ol.nuExtrap], mode: 'points' }, { name: 'Wavelength scan on the scan grid (x = wavelength ÷ 10 δ)', x: ol.scan.map((q) => q.wavelength / 10), y: ol.scan.map((q) => q.nu), mode: 'points' }], note: `Observed order ${fmt(ol.order, 3)}, grid-convergence index ${fmt(ol.gciPct, 3)} %. The stack model uses the extrapolated value.` });
    if (fld) plots.push({ type: 'field', title: 'Electroconvective vortices in the depleted layer at 1.7 × threshold', xlabel: 'Along the membrane (x/δ)', ylabel: 'Distance from the membrane (y/δ)', zlabel: 'Concentration', zunit: 'c / c_bulk', x: fld.x, y: fld.y, z: fld.c, u: fld.u, v: fld.w, stream: true, cmap: 'salinity', contours: 8, note: `Finest grid: Sherwood number ${fmt(fld.nu, 4)} (i / i_lim); slip velocity up to ${fmt(fld.umax, 3)} D/δ.` });
    tables.push({ title: 'Electroconvection model', columns: ['Quantity', 'Value', 'Unit'], rows: [['Electro-osmotic Péclet number ε(RT/F)²/(ηD)', ol.Pe, '–'], ['Slip cut-off length ℓ', ol.ell, 'δ'], ['Critical wavenumber k_c·δ (minimum of the marginal curve)', ol.kc, '–'], ['Vortex wavenumber k·δ (most-transporting cell)', ol.kSel, '–'], ['Threshold voltage (linear stability)', ol.Vc, 'RT/F'], ['Threshold voltage', ol.Vc * V, 'V'], ['Plateau length per cell pair (two depleted layers)', 2 * ol.Vc * V, 'V'], ...ol.pts.map((q) => [`Sherwood number at ${fmt(q.V / ol.Vc, 3)} × threshold (extrapolated)`, q.nu, '–']), ['Over-limiting slope d(i/i_lim)/dV', ol.slope, 'per RT/F']], note: 'These values replace the empirical plateau length and over-limiting conductance in the polarisation curve and in the stack solution.' });
    tables.push({ title: 'Electroconvection: grid-convergence study (Richardson extrapolation)', columns: ['Grid', 'Cells along × across', 'Cell size h', 'i/i_lim at 1.7 × threshold', 'i/i_lim at 1.3 × threshold', 'Steady'], rows: [...rh.grids.map((g, q) => [['coarse', 'medium', 'fine'][q], `${g.nx} × ${g.ny}`, g.h, rh.nu[q], rh.lo.nu[q], g.steady ? 'yes' : 'time-averaged']), ['extrapolated (h → 0)', '–', 0, rh.nuExtrap, rh.lo.nuExtrap, '–'], ['observed order p', '–', null, rh.order, rh.lo.order, '–'], ['grid-convergence index (%)', '–', null, rh.gciPct, rh.lo.gciPct, '–']],
      note: `Refinement ratio 1.5 in both directions, ${rh.modes} Fourier modes of the Stokes flow (all that the cut-off passes), grid clustered towards the depleted interface. Wavelength scan (÷ critical): ${ol.scan.map((q) => `${fmt(1 / q.kf, 3)} → ${fmt(q.nu, 4)}${q.steady ? '' : ' (unsteady)'}`).join(', ')}.` });
    out.electroconvectionThresholdV = ol.Vc * V; out.overLimitingRatio = ol.nuExtrap; out.overLimitingGciPct = ol.gciPct;
  }
  // ---- Navier–Stokes / Nernst–Planck channel
  if (r.par.ns && r.par.ns.cfd) {
    const ns = r.par.ns, s = ns.sol, tm = 0.5 * (1 + r.par.alphaC), ts = e0.Dc / (e0.Dc + e0.Da), jw = ((tm - ts) * r.iAvg) / F, sc = jw / s.jw, cOf = (c) => e0.ceq - sc * (s.c0 - c), ilimC = (F * s.Sh * e0.Ds * e0.ceq) / (2 * r.G.h) / (tm - ts);
    const hd0 = channel(ns.U, r.G.h, r.G.eps, T, e0.Ds, { ...r.par, ns: null }), xm = s.xc.map((x) => x * 1000), ym = s.yc.map((y) => y * 1000), pOut = (p.pOut || 0) * 1e5, dpStage = s.dpdx * r.G.L, cwMin = cOf(s.cwMin);
    const base = { type: 'field', xlabel: 'x (mm)', ylabel: 'y (mm)', x: xm, y: ym, mask: s.fields.mask, equal: s.Lc / s.h <= 8, shapes: s.shapes.map((q) => ({ ...q, x: q.x.map((x) => x * 1000), y: q.y.map((y) => y * 1000) })) };
    kpis.push({ label: 'Sherwood number (2-D Navier–Stokes)', value: s.Sh, unit: '–', help: `Flux-weighted over ${s.spacer ? 'whole spacer pitches' : 'the whole flow path'}; the spacer correlation gives ${fmt((hd0.Sh * 2 * r.G.h) / hd0.dh, 3)} on the same 2h basis` }, { label: 'Boundary-layer thickness (2-D Navier–Stokes)', value: ((2 * r.G.h) / s.Sh) * 1e6, unit: 'µm', help: 'δ = D/k = 2h/Sh' },
      { label: 'Channel pressure drop per stage (2-D Navier–Stokes)', value: dpStage / 1e5, unit: 'bar', help: `Pressure gradient ${fmt(s.dpdx, 4)} Pa/m over whole pitches; friction factor × Reynolds number ${fmt(s.fRe, 4)}` }, { label: 'Channel inlet pressure (2-D Navier–Stokes)', value: (pOut + dpStage) / 1e5, unit: 'bar' });
    if (!s.converged) W.push({ level: 'warn', msg: `The 2-D Navier–Stokes solution stopped after ${s.iters} iterations without meeting its tolerance — refine the grid across the channel or lower the velocity.` });
    if (cwMin <= 0) W.push({ level: 'warn', msg: 'The 2-D channel solution reaches zero salt concentration at a membrane wall at the mean current density: locally the current exceeds the limiting value (typically behind a filament).' });
    plots.push({ ...base, title: 'Velocity in the ED channel (2-D Navier–Stokes, finite volumes)', z: s.fields.speed, u: s.fields.uu, v: s.fields.vv, stream: true, zlabel: 'Speed', zunit: 'm/s', cmap: 'viridis', note: `${s.spacer ? `Spacer filaments (${s.arr}), ` : 'Open channel, '}superficial velocity ${fmt(s.U * 100, 3)} cm/s, Re = ${fmt(s.Re, 3)} (2h basis); ${s.nx} × ${s.ny} cells, ${s.iters} iterations.` },
      { ...base, title: 'Salt concentration in the diluate channel at the mean current density', z: s.fields.c.map((row) => row.map(cOf)), zlabel: 'Concentration', zunit: 'eq/m³', cmap: 'salinity', contours: 8, note: `Both membranes remove (t̄ − t)·i/F = ${fmt(jw * 1000, 3)} mmol/m²·s; lowest wall concentration ${fmt(Math.max(cwMin, 0), 3)} eq/m³ of ${fmt(e0.ceq, 3)} in the bulk at the inlet.` },
      { type: 'line', title: 'Local Sherwood number and pressure along the channel (2-D Navier–Stokes)', xlabel: 'x (mm)', ylabel: 'see legend', series: [{ name: 'Local Sherwood number (both walls)', x: xm.filter((_, k) => s.ShLoc[k] !== null), y: s.ShLoc.filter((q) => q !== null) }, { name: 'Section-mean pressure above the outlet (Pa)', x: xm, y: s.pm.map((q) => q - s.pm[s.nx - 1]) }], hlines: [{ y: s.Sh, label: 'mean used by the stack model' }] });
    tables.push({ title: 'Channel hydrodynamics and mass transfer (2-D Navier–Stokes + salt transport)', columns: ['Quantity', 'Value', 'Unit'], rows: [['Spacer arrangement', s.spacer ? s.arr : 'open channel', ''], ['Solved length', s.Lc * 1000, 'mm'], ['Grid (along × across)', `${s.nx} × ${s.ny}`, 'cells'], ['Solid fraction of the section', s.solidFraction, '–'], ['Reynolds number (2h, superficial velocity)', s.Re, '–'], ['Schmidt number', s.Sc, '–'], ['Friction factor (Darcy, 2h)', s.f, '–'], ['Friction factor × Reynolds number', s.fRe, '–'], ['Pressure gradient', s.dpdx, 'Pa/m'], ['Pressure drop of one stage', dpStage / 1e5, 'bar'], ['Mean Sherwood number (2h)', s.Sh, '–'], ['Mass-transfer coefficient', (s.Sh * s.D) / (2 * s.h), 'm/s'], ['Boundary-layer thickness D/k', ((2 * s.h) / s.Sh) * 1e6, 'µm'], ['Limiting current density from this Sherwood number', ilimC, 'A/m²'], ['Sherwood number of the spacer correlation (2h basis)', (hd0.Sh * 2 * r.G.h) / hd0.dh, '–'], ['Pressure gradient of the spacer correlation', hd0.dpPerM, 'Pa/m'], ['Flow iterations', s.iters, ''], ['Converged', s.converged ? 'yes' : 'no', '']],
      note: `The stack model takes its mass-transfer coefficient and pressure gradient from this solution${s.spacer ? `, rescaled along the flow path with Sh ∝ Re^${fmt(ns.nRe, 2)}·Sc^⅓ and Δp ∝ u^1.7` : ', rescaled with Sh ∝ (Re·Sc)^⅓ and Δp ∝ μ·u'}. Two-dimensional section: filaments are transverse cylinders; the 3-D mesh of a woven spacer is not resolved.` });
    bal.push({ name: '2-D channel salt: inflow vs outflow + wall flux (mol/s per m width, solver units)', in: s.saltIn, out: s.saltOut });
    out.cfdSherwood = s.Sh; out.cfdDpPerM = s.dpdx; out.cfdFrictionRe = s.fRe;
  } else if (r.par.ns) {
    const tm = 0.5 * (1 + r.par.alphaC), ts = e0.Dc / (e0.Dc + e0.Da), u0 = r.par.ns.U, jw = ((tm - ts) * r.iAvg) / F;
    const s = channelNS({ U: u0, h: r.G.h, L: r.G.L, T, D: e0.Ds, c0: e0.ceq, jw: [jw, jw], pOut: (p.pOut || 0) * 1e5 }), ilimNS = (F * s.ShMean * e0.Ds * e0.ceq) / (2 * r.G.h) / (tm - ts);
    kpis.push({ label: 'Channel inlet pressure (Navier–Stokes)', value: s.pIn / 1e5, unit: 'bar', help: `Outlet pressure ${p.pOut || 0} bar plus the momentum-equation pressure drop of one stage` }, { label: 'Mean Sherwood number (Navier–Stokes–Nernst–Planck)', value: s.ShMean, unit: '–' }, { label: 'Hydrodynamic entrance length', value: s.entrance * 1000, unit: 'mm' });
    if (s.cwMin <= 0) W.push({ level: 'warn', msg: 'The Navier–Stokes–Nernst–Planck channel solution reaches zero salt concentration at the membrane wall: the mean current exceeds the local limiting current of the open channel.' });
    plots.push({ type: 'line', title: 'Developing velocity profile between the membranes (Navier–Stokes)', xlabel: 'u / U', ylabel: 'y / h', series: s.prof.filter((_, k) => k % 3 === 0 || k === s.prof.length - 1).map((q) => ({ name: `x = ${fmt(q.x * 1000, 3)} mm`, x: q.u, y: s.y.map((yy) => yy / r.G.h) })), note: `Uniform inflow at ${fmt(u0 * 100, 3)} cm/s, no slip at both membranes; Re = ${fmt(s.Re, 3)}, fully developed centre-line velocity 1.5 U, f·Re = ${fmt(s.fRe, 4)}.` },
      { type: 'line', title: 'Pressure, Sherwood number and wall concentration along the channel', xlabel: 'Distance from the inlet (m)', ylabel: 'see legend', logx: true, series: [{ name: 'Pressure above outlet (mbar)', x: s.x, y: s.p.map((q) => (q - (p.pOut || 0) * 1e5) / 100) }, { name: 'Local Sherwood number', x: s.x, y: s.Sh }, { name: 'Wall ÷ bulk concentration × 10', x: s.x, y: s.cw0.map((q, k) => (10 * q) / s.cb[k]) }], note: 'Prescribed salt flux (t̄ − t)·i/F through both walls at the mean current density of the stack.' });
    tables.push({ title: 'Channel hydrodynamics and mass transfer (Navier–Stokes–Nernst–Planck)', columns: ['Quantity', 'Value', 'Unit'], rows: [['Reynolds number (2h)', s.Re, '–'], ['Schmidt number', s.Sc, '–'], ['Entrance length', s.entrance * 1000, 'mm'], ['Pressure drop of one stage', s.dp / 1e5, 'bar'], ['Fully developed (Hagen–Poiseuille) pressure drop', s.dpFd / 1e5, 'bar'], ['Friction factor × Reynolds number at the outlet', s.fRe, '–'], ['Outlet pressure (boundary condition)', p.pOut || 0, 'bar'], ['Inlet pressure', s.pIn / 1e5, 'bar'], ['Mean Sherwood number', s.ShMean, '–'], ['Sherwood number at the outlet', s.ShEnd, '–'], ['Limiting current density from the mean Sherwood number', ilimNS, 'A/m²'], ['Lowest wall concentration', s.cwMin, 'eq/m³']], note: 'Open (spacer-free) channel. With this option the stack model takes its mass-transfer coefficient and pressure gradient from this solution instead of the spacer correlation.' });
    bal.push({ name: 'Channel salt balance, Navier–Stokes–Nernst–Planck (mol/s per m width)', in: s.saltIn, out: s.saltOut });
    out.channelInletPressureBar = s.pIn / 1e5;
  }
  // ---- Poisson–Nernst–Planck
  if (p.pnp && p.pnp !== 'off') try {
    const lam = debyeLength(2 * e0.ceq, T);
    if (p.pnp === 'membrane') {
      const base = { cd: e0.ceq, cc: eC.ceq, X, dm, Dp: e0.Dc, Dm: e0.Da, DpM: Dmem[0], DmM: Dmem[1], deltaD: delta, deltaC: delta, T, n: 80 };
      let pf = npProfile({ ...base, i: i0 }), it = i0; if (!pf.ok) { it = 0.5 * i0; pf = npProfile({ ...base, i: it }); }
      const spec = (Vv) => ({ z: [1, -1], T, ratio, layers: [{ L: delta, D: Dsol }, { L: dm, D: Dmem, X: -X }, { L: delta, D: Dsol }], left: { type: 'bulk', c: [e0.ceq, e0.ceq], psi: Vv }, right: { type: 'bulk', c: [eC.ceq, eC.ceq], psi: 0 } });
      const cm = Math.sqrt(e0.ceq * eC.ceq); // continuation: equal concentrations and no voltage (exact Donnan equilibrium) → target concentrations and voltage
      let Va = pf.ok ? pf.potential : 0.1, s = pnpRamp(spec(Va), { left: { psi: 0, c: [cm, cm] }, right: { c: [cm, cm] } }), Vb = Va, sb = s;
      if (s.converged && pf.ok) for (let q = 0; q < 8 && Math.abs(sb.current - it) > 1e-7 * it; q++) { const Vn = clamp(q === 0 ? Va * (it / (s.current || it)) : Vb - ((sb.current - it) * (Vb - Va)) / (sb.current - s.current || 1e-30), 0.6 * Vb, 1.6 * Vb + 0.02); if (q > 0) { Va = Vb; s = sb; } const nx = pnpContinue(spec, sb, Vb, Vn); if (!nx.converged) break; Vb = Vn; sb = nx; }
      s = sb;
      if (!s.converged) W.push({ level: 'warn', msg: 'The Poisson–Nernst–Planck solver did not converge at the operating current; the last converged state is shown.' });
      const um = s.x.map((x) => (x - delta) * 1e6), kI = s.x.findIndex((x) => x >= delta), win = s.x.map((x, k) => k).filter((k) => Math.abs(s.x[k] - delta) < 12 * lam), rhoMax = Math.max(...s.rho.map(Math.abs));
      kpis.push({ label: 'Debye length in the diluate', value: lam * 1e9, unit: 'nm' }, { label: 'Membrane-system voltage, Poisson–Nernst–Planck', value: 1000 * Vb, unit: 'mV', help: `Electroneutral Nernst–Planck–Donnan model: ${pf.ok ? fmt(1000 * pf.potential, 4) : '–'} mV at the same current` }, { label: 'Counter-ion transport number (PNP)', value: s.current !== 0 ? (F * s.J[0]) / s.current : 0, unit: '–' });
      plots.push({ type: 'line', title: 'Poisson–Nernst–Planck solution: diluate film | cation membrane | concentrate film', xlabel: 'Distance from the diluate-side membrane face (µm)', ylabel: 'mol/m³ · mV', series: [{ name: 'Co-ion, PNP (mol/m³)', x: um, y: s.c[1] }, { name: 'Counter-ion ÷ 20 inside the membrane, PNP', x: um, y: s.c[0].map((c, k) => (s.x[k] > delta && s.x[k] < delta + dm ? c / 20 : c)), dash: true }, { name: 'Potential, PNP (mV)', x: um, y: s.psi.map((q) => 1000 * (q - s.psi[0])) }, ...(pf.ok ? [{ name: 'Co-ion, electroneutral Nernst–Planck–Donnan', x: pf.x.map((x) => x * 1e6), y: pf.c, mode: 'points' }, { name: 'Potential, electroneutral (mV)', x: pf.x.map((x) => x * 1e6), y: pf.phi.map((q) => 1000 * q), mode: 'points' }] : [])], vlines: [{ x: 0, label: 'membrane' }, { x: dm * 1e6, label: '' }], note: `${s.nodes} nodes graded geometrically to ${fmt(Math.min(...s.x.slice(1).map((x, k) => x - s.x[k])) * 1e9, 2)} nm at the interfaces; Scharfetter–Gummel fluxes, Newton iteration (${s.iterations} iterations on the last step). Current ${fmt(s.current, 4)} A/m².` },
        { type: 'line', title: 'Space-charge region at the diluate | membrane interface', xlabel: 'Distance from the interface (nm)', ylabel: 'see legend', series: [{ name: 'Space charge ρ/F in the solution, × 10 (mol/m³)', x: win.map((k) => (s.x[k] - delta) * 1e9), y: win.map((k) => (s.x[k] <= delta ? (10 * s.rho[k]) / F : 0)) }, { name: 'Space charge ρ/F in the membrane ÷ 10 (mol/m³)', x: win.map((k) => (s.x[k] - delta) * 1e9), y: win.map((k) => (s.x[k] > delta ? s.rho[k] / F / 10 : 0)) }, { name: 'Potential relative to the interface (mV)', x: win.map((k) => (s.x[k] - delta) * 1e9), y: win.map((k) => 1000 * (s.psi[k] - s.psi[kI])) }], note: 'The electroneutral model replaces this double layer by a Donnan potential jump.' });
      tables.push({ title: 'Poisson–Nernst–Planck solution', columns: ['Quantity', 'Value', 'Unit'], rows: [['Mesh nodes', s.nodes, ''], ['Debye length, diluate', lam * 1e9, 'nm'], ['Debye length, membrane', debyeLength(X, T) * 1e9, 'nm'], ['Current density', s.current, 'A/m²'], ['Voltage across film | membrane | film', 1000 * Vb, 'mV'], ['Same, electroneutral Nernst–Planck–Donnan', pf.ok ? 1000 * pf.potential : null, 'mV'], ['Counter-ion flux', s.J[0], 'mol/m²·s'], ['Co-ion flux', s.J[1], 'mol/m²·s'], ['Co-ion flux, electroneutral model', pf.ok ? pf.Jm : null, 'mol/m²·s'], ['Flux non-uniformity over the mesh (current continuity)', s.fluxSpread, '–'], ['Largest space-charge density', rhoMax / F, 'mol/m³'], ['Field at the diluate-side interface', (-(s.psi[kI + 1] - s.psi[kI - 1]) / (s.x[kI + 1] - s.x[kI - 1])) / 1e6, 'MV/m']] });
      out.pnpVoltage = Vb;
    } else if (p.pnp === 'wall') {
      const psi0 = (p.pnpPsi0 || 100) / 1000, s = pnpRamp({ z: [1, -1], T, ratio: Math.min(ratio, 1.15), res: 12, layers: [{ L: 40 * lam, D: Dsol, n: 30 }], left: { type: 'wall', psi: psi0, flux: [0, 0] }, right: { type: 'bulk', c: [e0.ceq, e0.ceq], psi: 0 } }, { left: { psi: 0 } }, 4);
      const gc = (x) => 4 * V * Math.atanh(Math.tanh(psi0 / (4 * V)) * Math.exp(-x / lam)), sg = Math.sqrt(8 * 78.4 * EPS0 * R * (T + KELVIN) * e0.ceq) * Math.sinh(psi0 / (2 * V)), kk = s.x.map((x, k) => k).filter((k) => s.x[k] <= 8 * lam);
      if (!s.converged) W.push({ level: 'warn', msg: 'The Poisson–Nernst–Planck solver did not converge for this wall potential.' });
      kpis.push({ label: 'Debye length in the diluate', value: lam * 1e9, unit: 'nm' }, { label: 'Surface charge of the wall (PNP)', value: s.sigmaLeft * 1000, unit: 'mC/m²', help: `Grahame equation: ${fmt(sg * 1000, 4)} mC/m²` }, { label: 'Current through the insulating wall', value: s.current, unit: 'A/m²', help: 'Zero-flux (zero-current) boundary condition' });
      plots.push({ type: 'line', title: 'Diffuse double layer at an insulating charged wall (Poisson–Nernst–Planck)', xlabel: 'Distance from the wall (nm)', ylabel: 'mV · c/c_bulk', series: [{ name: 'Potential, PNP (mV)', x: kk.map((k) => s.x[k] * 1e9), y: kk.map((k) => 1000 * s.psi[k]) }, { name: 'Potential, Gouy–Chapman (mV)', x: kk.map((k) => s.x[k] * 1e9), y: kk.map((k) => 1000 * gc(s.x[k])), mode: 'points' }, { name: 'Counter-ion c/c_bulk', x: kk.map((k) => s.x[k] * 1e9), y: kk.map((k) => s.c[1][k] / e0.ceq) }, { name: 'Co-ion c/c_bulk × 10', x: kk.map((k) => s.x[k] * 1e9), y: kk.map((k) => (10 * s.c[0][k]) / e0.ceq) }], vlines: [{ x: lam * 1e9, label: 'Debye length' }], note: 'No ion crosses the wall (zero-flux, zero-current boundary); the bulk boundary holds the diluate concentration.' });
      tables.push({ title: 'Poisson–Nernst–Planck solution', columns: ['Quantity', 'Value', 'Unit'], rows: [['Mesh nodes', s.nodes, ''], ['Wall potential', 1000 * psi0, 'mV'], ['Debye length', lam * 1e9, 'nm'], ['Surface charge, PNP', s.sigmaLeft * 1000, 'mC/m²'], ['Surface charge, Grahame equation', sg * 1000, 'mC/m²'], ['Counter-ion enrichment at the wall', s.c[1][0] / e0.ceq, '×'], ['Boltzmann factor exp(Fψ₀/RT)', Math.exp(psi0 / V), '×'], ['Current', s.current, 'A/m²']] });
      out.pnpSurfaceCharge = s.sigmaLeft;
    } else {
      const ilim = (2 * F * e0.Dc * e0.ceq) / delta, cw = X, dl = (Vv) => ({ z: [1, -1], T, ratio: Math.min(ratio, 1.15), layers: [{ L: delta, D: Dsol, n: 30 }], left: { type: 'bulk', c: [e0.ceq, e0.ceq], psi: 0 }, right: { type: 'wall', cFix: [cw, null], flux: [0, 0], psi: -(Math.log(cw / e0.ceq) * V + Vv) } });
      let s = pnpRamp(dl(0), { right: { psi: 0, cFix: [e0.ceq, null] } }, 4), Vp = 0; const Vl = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20, 25, 30, 40].map((x) => x * V), cur = [], done = [];
      for (const Vv of Vl) { const nx = s.converged ? pnpContinue(dl, s, Vp, Vv) : s; if (!nx.converged) break; s = nx; Vp = Vv; cur.push(s.current / ilim); done.push(Vv); }
      if (done.length < Vl.length) W.push({ level: 'info', msg: `The Poisson–Nernst–Planck sweep of the depleted layer stopped at ${fmt(Vp, 3)} V.` });
      const xm = s.x.map((x) => (delta - x) * 1e6), esc = s.x.filter((x, k) => Math.abs(s.rho[k]) / F > 0.05 * e0.ceq), escW = esc.length ? delta - Math.min(...esc) : 0, G2 = done.length > 2 ? ((cur[cur.length - 1] - cur[cur.length - 2]) * ilim) / (done[done.length - 1] - done[done.length - 2]) : 0;
      kpis.push({ label: 'Limiting current of the film (PNP plateau)', value: (cur.find((_, k) => done[k] >= 8 * V) ?? cur[cur.length - 1] ?? 0) * ilim, unit: 'A/m²', help: `Classical value 2·F·D₊·c/δ = ${fmt(ilim, 4)} A/m²` }, { label: 'Extended space-charge thickness', value: escW * 1e6, unit: 'µm', help: `At ${fmt(Vp, 3)} V across the layer; the equilibrium Debye length is ${fmt(lam * 1e9, 3)} nm` }, { label: 'Over-limiting conductance from space charge', value: G2, unit: 'S/m²' });
      plots.push({ type: 'line', title: 'Current–voltage curve of the depleted diffusion layer (Poisson–Nernst–Planck)', xlabel: 'Voltage across the layer (V)', ylabel: 'i / i_lim', series: [{ name: 'Poisson–Nernst–Planck', x: done, y: cur, mode: 'both' }, { name: 'Electroneutral: 1 − exp(−FV/2RT)', x: done, y: done.map((q) => 1 - Math.exp(-q / (2 * V))), dash: true }], hlines: [{ y: 1, label: 'i_lim' }], note: 'Ideal cation-exchange surface: fixed counter-ion concentration, no co-ion flux. The slow rise above the plateau is carried by the extended space charge.' },
        { type: 'line', title: `Ion profiles in the depleted layer at ${fmt(Vp, 3)} V`, xlabel: 'Distance from the membrane surface (µm)', ylabel: 'mol/m³', logx: true, logy: true, series: [{ name: 'Counter-ion', x: xm.slice(0, -1), y: s.c[0].slice(0, -1) }, { name: 'Co-ion', x: xm.slice(0, -1), y: s.c[1].slice(0, -1).map((c) => Math.max(c, 1e-12)) }], note: 'Where the two curves separate the solution carries a net space charge (non-equilibrium double layer).' });
      tables.push({ title: 'Poisson–Nernst–Planck solution', columns: ['Voltage across the layer (V)', 'i / i_lim (PNP)', 'i / i_lim (electroneutral)'], rows: done.map((q, k) => [q, cur[k], 1 - Math.exp(-q / (2 * V))]), note: `${s.nodes} nodes; no-flux condition for the co-ion and prescribed concentration for the counter-ion at the membrane surface.` });
      out.pnpLimitingCurrent = ilim;
    }
  } catch (e) { W.push({ level: 'warn', msg: `The Poisson–Nernst–Planck model could not be solved: ${e.message}` }); }
  // ---- Maxwell–Stefan
  if (p.msModel) try {
    const eqT = e0.eqC + e0.eqA, act = CH.map((_, j) => j).filter((j) => AZ[j] * r.cf[j] > 1e-3 * eqT), z = act.map((j) => Z[j]), cL = act.map((j) => r.cf[j]), cR = act.map((j) => r.tr.ccIn[j]), fT = e0.Dc / (sum(act.map((j) => (Z[j] > 0 ? AZ[j] * r.cf[j] * DI[j] : 0))) / (e0.eqC || 1) || 1.33e-9);
    const Diw = act.map((j) => p.msDw * DI[j] * fT), Dim = act.map((j) => p.msDm * DI[j] * fT), cw = p.msWater * 55500, arg = { z, cL, cR, X: -X, dm, Diw, Dim, cw, i: i0, T, n: Math.max(6, Math.round(p.nProf)) };
    const ms = msMembrane({ ...arg, Dij: p.msDij * 1e-11, Dwm: p.msDwm * 1e-10 }), np = msMembrane({ ...arg, Dij: Infinity, Dwm: 1e-30, Diw: ms.Deff, Dim: act.map(() => Infinity), cw: 1e12 });
    if (!ms.converged) W.push({ level: 'warn', msg: `The Maxwell–Stefan membrane model converged only up to ${fmt(ms.iUsed, 3)} A/m² instead of the inlet current density of ${fmt(i0, 3)} A/m²; its results are shown for that lower current.` });
    const tCount = sum(ms.t.filter((_, a) => z[a] > 0)), tNP = sum(np.t.filter((_, a) => z[a] > 0));
    kpis.push({ label: 'Counter-ion transport number (Maxwell–Stefan)', value: tCount, unit: '–', help: `Nernst–Planck with the same effective diffusivities: ${fmt(tNP, 4)}; stack input ${fmt(0.5 * (1 + r.par.alphaC), 4)}` }, { label: 'Electro-osmotic water transport (Maxwell–Stefan)', value: ms.tw, unit: 'mol H₂O per Faraday', help: 'Water dragged through the cation membrane by ion–water friction' });
    plots.push({ type: 'line', title: 'Maxwell–Stefan ion profiles in the cation membrane (stack inlet)', xlabel: 'Position in the membrane (µm)', ylabel: 'mol/m³', logy: true, series: act.map((j, a) => ({ name: IONS[CH[j]].label, x: ms.x.map((x) => x * 1e6), y: ms.c.map((c) => Math.max(c[a], 1e-9)) })), note: `Fixed charge ${p.Xfix} mol/L, ${fmt(ms.iUsed, 3)} A/m²; Donnan equilibrium at both faces.` });
    tables.push({ title: 'Maxwell–Stefan transport in the cation membrane (stack inlet)', columns: ['Ion', 'Diluate face (mol/m³)', 'Concentrate face (mol/m³)', 'Flux (mmol/m²·s)', 'Transport number', 'Transport number, Nernst–Planck', 'Effective diffusivity (10⁻¹¹ m²/s)'], rows: [...act.map((j, a) => [`${IONS[CH[j]].name} ${IONS[CH[j]].label}`, ms.mL[a], ms.mR[a], ms.N[a] * 1000, ms.t[a], np.t[a], ms.Deff[a] * 1e11]), ['Water', cw, cw, ms.Nw * 1000, ms.tw, 0, null]], note: 'Friction of every ion with water, polymer and the other ions; the water flux follows from a zero pressure difference across the membrane. The Nernst–Planck column drops ion–ion friction and water motion. Ions below 0.1 % of the feed equivalents are left out.' });
    out.msCounterIonTransport = tCount; out.msWaterTransport = ms.tw;
  } catch (e) { W.push({ level: 'warn', msg: `The Maxwell–Stefan membrane model could not be solved: ${e.message}` }); }
  // ---- electrochemical–thermal
  if (r.th) {
    const th = r.th;
    if (th.Tout > 45) W.push({ level: 'warn', msg: `The stack outlet reaches ${fmt(th.Tout, 3)} °C; most ion-exchange membranes are limited to about 40–45 °C.` });
    kpis.push({ label: 'Stack temperature rise', value: th.dT, unit: 'K', help: `${fmt(th.Qgen / 1000, 3)} kW of Joule and polarisation heat; mean stack temperature ${fmt(th.Tmean, 4)} °C` }, { label: 'Specific energy, isothermal at inlet temperature', value: th.iso.sec, unit: 'kWh/m³', help: 'Without the conductivity feedback of the warmer stack' });
    tables.push({ title: 'Electrochemical–thermal coupling', columns: ['Quantity', 'Value', 'Unit'], rows: [['Inlet temperature', th.Tin, '°C'], ['Mean stack temperature', th.Tmean, '°C'], ['Outlet temperature', th.Tout, '°C'], ['Heat released in the stack', th.Qgen / 1000, 'kW'], ['Reversible work stored in the streams', th.Wsep / 1000, 'kW'], ['Heat lost to the surroundings', 100 * th.loss, '%'], ['Feed conductivity at inlet temperature', th.iso.kappa, 'S/m'], ['Feed conductivity at stack temperature', th.kappa, 'S/m'], ['Specific energy, isothermal', th.iso.sec, 'kWh/m³'], ['Specific energy, with thermal feedback', r.sec, 'kWh/m³'], ['Iterations', th.iterations, '']], note: 'ρ·c_p·Q·ΔT = (1 − loss)·(DC power − reversible work + pumping dissipation); the stack is then re-solved at the mean temperature.' });
    bal.push({ name: 'Stack energy (kW): heat released vs enthalpy rise + loss', in: th.Qgen / 1000, out: (th.mcp * th.dT + th.loss * th.Qgen) / 1000 });
    out.stackTemperatureRise = th.dT;
  }
  // ---- fouling and scaling
  if (p.foulModel) {
    const f = foulingED({ i: r.iAvg, u: seg0.u, cFoul: p.cFoul, kDep: p.kDep, kDet: p.kDet, rFoul: p.rFoul, Sg: r.scWall.gypsum, Sc: r.scWall.calcite, kScale: p.kScale, rScale: p.rScale, mBlock: 50, edr: !!p.edr, revInterval: p.revInterval, revEff: p.revEff / 100, days: p.tCamp, n: 200 });
    const Umean = sum(st.map((s) => s.U)) / r.nSt, Rcp = Umean / Math.max(r.iAvg, 1e-9), rel = f.dU.map((d) => (100 * d) / Umean), trig = p.cipTrig, kT = rel.findIndex((x) => x >= trig), tClean = kT > 0 ? f.t[kT - 1] + ((f.t[kT] - f.t[kT - 1]) * (trig - rel[kT - 1])) / (rel[kT] - rel[kT - 1]) : kT === 0 ? 0 : null;
    const dSec = (f.dU[f.dU.length - 1] * sum(st.map((s) => s.I)) * r.Ncp) / (p.etaRect / 100) / 1000 / (r.Qprod * 3600);
    if (tClean !== null) W.push({ level: tClean < 14 ? 'warn' : 'info', msg: `Fouling and scaling raise the cell-pair voltage by ${trig} % after ${fmt(tClean, 3)} days: plan a clean-in-place at that interval.` });
    if (f.sup > 0 && !p.edr) W.push({ level: 'warn', msg: 'The concentrate is supersaturated at the membrane wall and no polarity reversal removes the scale: the scale layer grows without bound.' });
    kpis.push({ label: 'Cell-pair voltage rise from fouling and scaling', value: rel[rel.length - 1], unit: '%', status: rel[rel.length - 1] > trig ? 'warn' : 'ok', help: `After ${p.tCamp} days at constant current` }, { label: 'Days to cleaning', value: tClean ?? p.tCamp, unit: 'd', help: tClean === null ? 'The cleaning trigger is not reached within the simulated campaign' : `Voltage rise of ${trig} %` }, { label: 'Energy increase at the end of the campaign', value: dSec, unit: 'kWh/m³' });
    plots.push({ type: 'line', title: 'Fouling and scaling during the operating campaign', xlabel: 'Time (d)', ylabel: 'see legend', series: [{ name: 'Organic deposit on the anion membrane (g/m²)', x: f.t, y: f.mf }, { name: 'Mineral scale on the concentrate side (g/m²)', x: f.t, y: f.ms }, { name: 'Cell-pair voltage rise (%)', x: f.t, y: rel }, { name: 'Blocked membrane area (%)', x: f.t, y: f.block.map((b) => 100 * b) }], hlines: [{ y: trig, label: 'cleaning trigger' }], note: `Constant current ${fmt(r.iAvg, 3)} A/m²; wall saturation ratios: gypsum ${fmt(r.scWall.gypsum, 3)}, calcite ${fmt(r.scWall.calcite, 3)}${p.edr ? `; polarity reversal every ${p.revInterval} min removes ${p.revEff} % of the deposits (rate ${fmt(f.kRev, 3)} per day).` : '.'}` });
    tables.push({ title: 'Fouling and scaling', columns: ['Quantity', 'Value', 'Unit'], rows: [['Organic deposit at the end', f.mf[f.mf.length - 1], 'g/m²'], ['Steady-state organic deposit', Number.isFinite(f.mfInf) ? f.mfInf : null, 'g/m²'], ['Mineral scale at the end', f.ms[f.ms.length - 1], 'g/m²'], ['Blocked area at the end', 100 * f.block[f.block.length - 1], '%'], ['Added area resistance at the end', f.dR[f.dR.length - 1] * 1e4, 'Ω·cm²'], ['Clean cell-pair resistance', Rcp * 1e4, 'Ω·cm²'], ['Voltage rise at the end', f.dU[f.dU.length - 1], 'V per cell pair'], ['Removal rate by polarity reversal', f.kRev, '1/d'], ['Scaling driving force Σ(S − 1)²', f.sup, '–']] });
    out.daysToCleaningED = tClean; out.foulingVoltageRisePct = rel[rel.length - 1];
  }
  // ---- surrogate
  if (p.surrogate) try {
    const sg = stackSurrogate(r, clamp(Math.round(p.nTrain || 48), 16, 120)), Us = linspace(0.25, 1.2, 20), uNow = clamp(seg0.u * 100, 3, 16), mech = Us.map((U) => sg.at(1, U, uNow));
    if (sg.sec.r2 < 0.95) W.push({ level: 'warn', msg: `The surrogate reproduces held-out model runs only with R² = ${fmt(sg.sec.r2, 3)} — increase the number of training runs.` });
    kpis.push({ label: 'Surrogate R², specific energy (held-out)', value: sg.sec.r2, unit: '–', status: sg.sec.r2 < 0.95 ? 'warn' : 'ok', help: `${SURROGATES[sg.best.sec]} selected by validation; ${sg.train.length} training and ${sg.test.length} held-out runs of the mechanistic stack model` }, { label: 'Surrogate R², salt removal (held-out)', value: sg.removal.r2, unit: '–', help: `${SURROGATES[sg.best.removal]} selected by validation` }, { label: 'Selected surrogate (specific energy)', value: SURROGATES[sg.best.sec], help: `Held-out RMSE: ${Object.keys(sg.models).map((n) => `${SURROGATES[n]} ${fmt(sg.models[n].sec.rmse, 3)}`).join(', ')} kWh/m³` });
    const okM = Object.keys(sg.models).filter((n) => sg.models[n].sec.ok && sg.models[n].removal.ok), pmax = Math.max(1, ...sg.test.map((q) => q.sec));
    plots.push({ type: 'line', title: 'Surrogate parity on held-out runs: three learners', xlabel: 'Mechanistic model', ylabel: 'Surrogate', series: [...okM.map((n) => ({ name: `Specific DC energy, ${SURROGATES[n].toLowerCase()} (kWh/m³)`, x: sg.test.map((q) => q.sec), y: sg.models[n].sec.pred, mode: 'points' })), ...okM.map((n) => ({ name: `Salt removal, ${SURROGATES[n].toLowerCase()} (–)`, x: sg.test.map((q) => q.removal), y: sg.models[n].removal.pred, mode: 'points' })), { name: '1 : 1', x: [0, pmax], y: [0, pmax], dash: true }], note: `Kernel ridge regression, Gaussian process and neural network trained on the same Latin-hypercube sample of feed salinity (0.4–2.5 ×), voltage (0.25–1.2 V) and velocity (3–16 cm/s); the ${sg.test.length} plotted runs were used neither for training nor for selecting the learner (seed ${sg.seed}).` },
      { type: 'line', title: 'Surrogate versus mechanistic model along the voltage', xlabel: 'Voltage per cell pair (V)', ylabel: 'kWh/m³ · –', series: [{ name: 'Specific DC energy, mechanistic', x: Us, y: mech.map((q) => q.sec), mode: 'points' }, { name: 'Specific DC energy, surrogate', x: Us, y: Us.map((U) => sg.pred(1, U, uNow).sec) }, { name: 'Salt removal, mechanistic', x: Us, y: mech.map((q) => q.removal), mode: 'points' }, { name: 'Salt removal, surrogate', x: Us, y: Us.map((U) => sg.pred(1, U, uNow).removal) }], note: `Present feed, ${fmt(uNow, 3)} cm/s, ${r.nSt} stage${r.nSt > 1 ? 's' : ''}.` });
    tables.push({ title: 'Surrogate learners compared on the same design points', columns: ['Learner', 'Validation RMSE, ln SEC', 'Held-out R², SEC', 'Held-out RMSE, SEC (kWh/m³)', 'Validation RMSE, removal', 'Held-out R², removal', 'Held-out RMSE, removal (–)', 'Selected for'], rows: Object.keys(sg.models).map((n) => { const m = sg.models[n], f = (x) => (Number.isFinite(x) ? x : null); return [SURROGATES[n], f(m.sec.val), m.sec.ok ? m.sec.r2 : null, f(m.sec.rmse), f(m.removal.val), m.removal.ok ? m.removal.r2 : null, f(m.removal.rmse), [sg.best.sec === n ? 'specific energy' : null, sg.best.removal === n ? 'salt removal' : null].filter(Boolean).join(' + ') || '–']; }), note: `Selection: each learner is trained on ${sg.nFit} runs and scored on ${sg.nVal} validation runs; the one with the lowest validation error is retrained on all ${sg.train.length} runs. Held-out columns: ${sg.test.length} further runs. Neural network: two tanh layers (10 and 8 units), Adam with early stopping. Gaussian process: anisotropic squared-exponential kernel, maximum marginal likelihood.` },
      { title: 'Surrogate model details', columns: ['Quantity', 'Specific DC energy', 'Salt removal'], rows: [['Selected learner', SURROGATES[sg.best.sec], SURROGATES[sg.best.removal]], ['Kernel regression: length scale (unit cube)', sg.mS.ell, sg.mR.ell], ['Kernel regression: leave-one-out RMSE (training; ln kWh/m³ · –)', sg.mS.rmseLoo, sg.mR.rmseLoo], ['Held-out R² of the selected learner', sg.sec.r2, sg.removal.r2], ['Held-out RMSE of the selected learner (kWh/m³ · –)', sg.sec.rmse, sg.removal.rmse], ['Training runs', sg.train.length, sg.train.length], ['Held-out runs', sg.test.length, sg.test.length]] },
      { title: 'Surrogate held-out runs', columns: ['Salinity ×', 'Voltage (V)', 'Velocity (cm/s)', 'SEC model (kWh/m³)', 'SEC surrogate (kWh/m³)', 'Removal model (–)', 'Removal surrogate (–)'], rows: sg.test.map((q) => [q.sf, q.U, q.u, q.sec, q.p.sec, q.removal, q.p.removal]) });
    out.surrogateR2 = sg.sec.r2; out.surrogateModel = sg.best.sec;
  } catch (e) { W.push({ level: 'warn', msg: `The surrogate could not be trained: ${e.message}` }); }
  // ---- operating-condition optimisation
  if (p.optimise && p.mode === 'design') {
    const o = optimiseED(p), base = edCost(r, p);
    if (o.best) {
      const zf = o.z.flat().filter((x) => x !== null), zm = zf.length ? Math.max(...zf) : base;
      kpis.push({ label: 'Lowest unit cost (optimised operation)', value: Math.min(o.best.cost, base), unit: '$/m³', help: `Present operating point: ${fmt(base, 3)} $/m³ (electricity + membrane replacement)` }, { label: 'Optimal i / i_lim', value: o.best.cost <= base ? o.best.phi : p.safety, unit: '%' }, { label: 'Optimal linear velocity', value: o.best.cost <= base ? o.best.u : p.uLin, unit: 'cm/s' });
      plots.push({ type: 'field', title: 'Unit cost versus operating current fraction and velocity', xlabel: 'Operating current ÷ limiting current (%)', ylabel: 'Linear velocity (cm/s)', zlabel: 'Unit cost', zunit: '$/m³', x: o.phis, y: o.us, z: o.z.map((row) => row.map((x) => (x === null ? zm : x))), cmap: 'viridis', contours: 8, markers: [{ x: o.best.phi, y: o.best.u, label: 'optimum' }, { x: clamp(p.safety, o.phis[0], o.phis[o.phis.length - 1]), y: clamp(p.uLin, o.us[0], o.us[o.us.length - 1]), label: 'present' }], shapes: o.path.length > 1 ? [{ x: o.path.map((c) => c.phi), y: o.path.map((c) => c.u), closed: false, dash: true }] : [], note: `Backdrop: grid of complete re-designs for the same product target (coarse path resolution; cells that miss the target take the highest cost of the map). Optimum: Nelder–Mead simplex over the continuous variables, started from the best cell (${o.evals} designs in total; dashed line = designs visited). Best grid cell ${fmt(o.gridBest.cost, 4)} $/m³, optimised ${fmt(o.best.cost, 4)} $/m³.` });
      tables.push({ title: 'Operating-condition optimisation', columns: ['i / i_lim (%)', 'Velocity (cm/s)', 'Unit cost ($/m³)', 'Specific energy (kWh/m³)', 'Membrane area (m²)', 'Stages', 'Cell pairs per stage'], rows: [[o.best.phi, o.best.u, o.best.cost, o.best.sec, o.best.area, o.best.nSt, o.best.Ncp], ...[...o.cells].sort((a, b) => a.cost - b.cost).slice(0, 7).map((c) => [c.phi, c.u, c.cost, c.sec, c.area, c.nSt, c.Ncp])], note: `First row: optimum of the simplex search (${o.iterations} iterations); then the seven best of ${o.cells.length} feasible grid designs. A higher current fraction saves membrane area but costs energy; the optimum balances the two at ${p.cElecED} $/kWh and ${p.cMem} $/m² over ${p.memLife} years.` });
      if (o.best.cost < 0.97 * base) recs.push(`Operating at ${fmt(o.best.phi, 3)} % of the limiting current and ${fmt(o.best.u, 3)} cm/s lowers the unit cost from ${fmt(base, 3)} to ${fmt(o.best.cost, 3)} $/m³.`);
      out.optimalCurrentFraction = o.best.phi; out.optimalVelocity = o.best.u; out.optimalCost = o.best.cost;
    } else W.push({ level: 'warn', msg: 'No operating point of the optimisation grid reaches the product target.' });
  }
  return { kpis, plots, tables, recs, out, bal };
}

const suite = {
  id: 'ed', num: 7, title: 'Electrodialysis & Membrane Electrochemical Processes', short: 'Electrodialysis', icon: '⚡',
  tagline: 'Ion-by-ion design and rating of ED/EDR stacks, with bipolar-membrane acid/base production and membrane capacitive deionisation.',
  description: 'Resolves every hydraulic stage of an electrodialysis stack along the flow path: an analytical electroneutral Nernst–Planck film model gives concentration polarisation and the limiting current, Donnan potentials and permselectivity give the membrane voltage, and ohmic, electrode and shunt losses complete the voltage balance. Each ion is transported according to its mobility, concentration and membrane selectivity, with co-ion leakage, back-diffusion, electro-osmosis and osmosis, so product quality, current efficiency, energy, scaling risk and the required cell pairs follow directly. Batch operation, a film–membrane–film Nernst–Planck–Donnan profile, and reduced-order bipolar-membrane and capacitive-deionisation models are included.',
  guide: [
    'Enter the feed analysis and the product flow (or pull the case feed water or an RO stream). ED is most competitive below about 5–10 g/L.',
    'Choose the process variant and, for ED, the operating specification: design for a target product TDS, fixed voltage, fixed current or batch.',
    'Check the membrane and spacer properties on Model setup; calibrate resistance, permselectivity and the Sherwood coefficient against stack data if you have them.',
    'Run. Keep the current below about 70–80 % of the limiting current density, and check the scaling ratios of the concentrate and the voltage breakdown.',
  ],
  implemented: ['nernst-planck equation', 'poisson equation', 'electroneutral nernst-planck', 'nernst equation', 'donnan-equilibrium', 'ohm', 'faraday', 'butler-volmer', 'tafel', 'charge-conservation', 'current-continuity', 'ionic mass balance', 'convection-diffusion', 'water-dissociation', 'membrane partition', 'poisson-nernst-planck', 'goldman-hodgkin-katz', 'navier-stokes equation',
    'donnan-nernst-planck', 'electro-osmosis-ion-transport', 'electrodialysis-water-splitting', 'bipolar-membrane', 'electrode-reaction-ion-transport', 'nernst-planck-poisson', 'nernst-planck-navier-stokes', 'maxwell-stefan-electrochemical', 'electroconvection model', 'electrochemical-thermal', 'electrochemical-machine-learning',
    'ion concentration', 'electric potential', 'membrane charge', 'temperature', 'velocity', 'electrode state', 'fixed-potential', 'imposed-current', 'electrode butler-volmer', 'donnan-interface', 'ion-partition', 'specified-concentration', 'inlet-flow', 'membrane-interface continuity', 'insulating/zero-current', 'specified-ion-flux', 'no-flux', 'outlet-pressure',
    'electrolyte chemistry', 'ionic-species transport', 'diffusion', 'electromigration', 'convection', 'electric-potential calculation', 'current-density prediction', 'ion-exchange membrane modelling', 'membrane selectivity', 'membrane resistance', 'electrode reactions', 'electrode compartments', 'concentration and diluate channels', 'concentration polarisation', 'limiting-current assessment', 'water transport', 'electro-osmosis', 'acid-base chemistry', 'electrochemical reactions', 'electrical-energy consumption', 'stack configuration', 'dynamic simulation', 'fouling and scaling', 'operating-condition optimisation', 'navier-stokes equations', 'nernst-planck-navier-stokes models', 'electroconvection models', 'electrochemical-machine-learning models'],
  equationsNote: 'Stack model: channels are one-dimensional plug flow with a film (Sherwood) boundary layer; the film solution is the exact electroneutral Nernst–Planck result for an equivalent binary salt built from the equivalent-weighted ion diffusivities, and membrane transport numbers split the current between counter-ions in proportion to mobility × concentration × selectivity. Optional sub-models on Model setup resolve what this leaves out: (1) a steady 1-D Poisson–Nernst–Planck solver (finite volumes, Scharfetter–Gummel fluxes, Newton iteration, mesh graded to the Debye length) for a binary 1:1 electrolyte across film | membrane | film, at an insulating charged wall or in the depleted diffusion layer — it is a local analysis shown beside the stack result, not fed back into it; (2) the Goldman–Hodgkin–Katz constant-field membrane potential, solved ion by ion for the bulk compositions with its local slope carrying the polarisation correction; (3) Maxwell–Stefan friction transport of all feed ions in the cation membrane with electro-osmotic water drag (ideal activities, constant friction diffusivities); (4) electroconvection from the Rubinstein–Zaltzman electro-osmotic slip model with a short-wave cut-off of the slip (an input length, 0.2 δ by default, standing for the finite extended space charge; without it the limiting formula has no preferred wavelength and no grid-converged solution): the critical wavenumber is the minimum of the marginal-stability curve, the vortex-cell wavelength is the most-transporting of a scan, and the 2-D non-linear Stokes / salt-transport problem is solved with second-order differences on a grid clustered towards the depleted interface (all Fourier modes the cut-off passes, ADI time stepping) on three systematically refined grids; the Richardson-extrapolated Sherwood number, its observed order and grid-convergence index are reported and the extrapolated value replaces the empirical plateau length and over-limiting conductance, while the share of water splitting stays an input; (5) channel flow, either as a boundary-layer (parabolised) Navier–Stokes march with electroneutral Nernst–Planck salt transport in an open, spacer-free channel (fast; no axial diffusion or recirculation), or as the full two-dimensional steady Navier–Stokes and salt-transport solution of the finite-volume solver of suite 4 around transverse spacer filaments (zigzag, cavity, submerged) or in the open channel, with the salt flux of the operating current at both membranes — either one then supplies the mass-transfer coefficient and pressure gradient of the stack (the 2-D section does not resolve the three-dimensional weave of a real spacer); (6) a stack energy balance with temperature-dependent properties at the mean stack temperature; (7) reduced-order fouling and scaling kinetics at constant current with polarity reversal homogenised to a first-order removal rate; (8) three surrogates of the mechanistic single-pass rating trained on the same Latin-hypercube runs — a feed-forward neural network (Adam, early stopping), a Gaussian process (maximum marginal likelihood) and kernel ridge regression — selected on a validation split and compared on held-out runs; and (9) a Nelder–Mead simplex search of current fraction and velocity for the lowest unit cost, started from a coarse grid of re-designs that is kept as the cost map. Electrode kinetics follow Butler–Volmer with a selectable transfer coefficient or its Tafel limit. Activity coefficients are ideal except in the gypsum and calcite saturation ratios (Davies), which are screening values — use suite 2 for speciation. Bipolar-membrane ED and capacitive deionisation are reduced-order models (lumped unit voltage with empirical current efficiency; equilibrium modified-Donnan / Gouy–Chapman–Stern double layers with RC charging) intended for sizing, not for stack design. Valid for roughly 0.2–40 g/L and 5–45 °C.',

  inputs: [
    { group: 'Process and duty', help: 'What the electro-membrane system must do.', fields: [
      { key: 'process', label: 'Process', type: 'select', value: 'ed', options: [{ value: 'ed', label: 'Electrodialysis (ED / EDR)' }, { value: 'bpmed', label: 'Bipolar-membrane ED — acid and caustic from salt' }, { value: 'mcdi', label: 'Capacitive deionisation (CDI / MCDI)' }], help: 'Each variant shows its own inputs.' },
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.brackish.ions, help: 'Ion-by-ion analysis; every charged species is transported individually.' },
      { key: 'salinityFactor', label: 'Salinity multiplier', unit: '×', value: 1, min: 0.02, max: 20, help: 'Scales the whole analysis — convenient for sensitivity runs.' },
      { key: 'Qp', label: 'Diluate (product-side) flow', unit: 'm³/h', value: 100, min: 0.01, max: 50000, help: 'Flow through the diluate channels. For bipolar-membrane ED: salt-solution feed; for CDI: treated flow.' },
      { key: 'T', label: 'Temperature', unit: '°C', value: 25, min: 5, max: 45, help: 'Raises conductivity and diffusivity by about 2 % per kelvin.' },
      { key: 'pH', label: 'Feed pH', unit: '', value: 7.6, min: 2, max: 12, help: 'Used for the carbonate scaling ratio and the pH-shift estimate.' },
    ] },
    { group: 'ED operation', showIf: isED, help: 'How the stack is specified.', fields: [
      { key: 'mode', label: 'Operating specification', type: 'select', value: 'design', options: [{ value: 'design', label: 'Design: target product TDS → stages, cell pairs, voltages' }, { value: 'voltage', label: 'Rating: fixed voltage per cell pair' }, { value: 'current', label: 'Rating: fixed average current density' }, { value: 'batch', label: 'Batch: recirculate a tank at fixed voltage' }], help: 'Design mode keeps every stage at the chosen fraction of its limiting current and adds stages until the target is met.' },
      { key: 'targetTDS', label: 'Target product TDS', unit: 'mg/L', value: 500, min: 5, max: 20000, showIf: (v) => v.mode === 'design' || v.mode === 'batch', help: 'Drinking water is typically ≤ 500 mg/L.' },
      { key: 'recovery', label: 'Water recovery', unit: '%', value: 72, min: 30, max: 97, typical: [50, 94], help: 'Product ÷ total feed; set by the concentrate blow-down (continuous) or the tank volumes (batch).' },
      { key: 'safety', label: 'Operating current ÷ limiting current', unit: '%', value: 70, min: 5, max: 99, typical: [50, 80], showIf: (v) => v.mode === 'design', help: 'Design safety factor against concentration polarisation.' },
      { key: 'uLin', label: 'Linear velocity in the channels', unit: 'cm/s', value: 8, min: 0.5, max: 40, typical: [5, 15], showIf: (v) => v.mode === 'design' || v.mode === 'batch', help: 'Sets the number of cell pairs (design) or the circulation flow (batch).' },
      { key: 'maxStages', label: 'Maximum number of stages', unit: '', value: 8, min: 1, max: 16, step: 1, showIf: (v) => v.mode === 'design', help: 'Upper limit for the automatic staging.' },
      { key: 'nStages', label: 'Hydraulic stages in series', unit: '', value: 3, min: 1, max: 16, step: 1, showIf: (v) => v.mode === 'voltage' || v.mode === 'current', help: 'Each stage is one pass along the flow path with its own electrode pair.' },
      { key: 'cpStack', label: 'Cell pairs per stack', unit: '', value: 500, min: 10, max: 1200, step: 1, showIf: (v) => v.mode !== 'batch', help: 'Larger duties are split over parallel stacks of at most this size.' },
      { key: 'Ncp', label: 'Cell pairs per stage', unit: '', value: 400, min: 1, max: 5000, step: 1, showIf: (v) => v.mode !== 'design', help: 'One cell pair = diluate channel + cation membrane + concentrate channel + anion membrane.' },
      { key: 'Ucp', label: 'Voltage per cell pair', unit: 'V', value: 0.6, min: 0.01, max: 4, typical: [0.3, 1.2], showIf: (v) => v.mode === 'voltage' || v.mode === 'batch', help: 'Stack voltage minus electrode voltage, divided by the number of cell pairs.' },
      { key: 'iSet', label: 'Average current density', unit: 'A/m²', value: 120, min: 1, max: 2000, showIf: (v) => v.mode === 'current', help: 'The stage voltage is solved so that the path-averaged current density equals this value.' },
      { key: 'Vbatch', label: 'Diluate batch volume', unit: 'm³', value: 5, min: 0.001, max: 5000, showIf: (v) => v.mode === 'batch', help: 'Initial volume of the diluate tank.' },
      { key: 'tBatch', label: 'Maximum batch time', unit: 'min', value: 90, min: 1, max: 3000, showIf: (v) => v.mode === 'batch', help: 'Integration stops here if the target is not reached earlier.' },
      { key: 'edr', label: 'Polarity reversal (EDR)', type: 'bool', value: true, help: 'Periodic reversal sheds scale and foulants but diverts off-spec product after every reversal.' },
      { key: 'revInterval', label: 'Reversal interval', unit: 'min', value: 20, min: 2, max: 240, showIf: (v) => v.edr, help: 'Time between polarity reversals.' },
      { key: 'offSpec', label: 'Off-spec product per reversal', unit: 's', value: 40, min: 0, max: 600, showIf: (v) => v.edr, help: 'Product diverted to waste while the channels swap roles.' },
    ] },
    { group: 'Stack geometry', showIf: (v) => v.process !== 'mcdi', help: 'Flow-path dimensions of one cell.', fields: [
      { key: 'W', label: 'Flow-path width', unit: 'm', value: 0.5, min: 0.02, max: 2, help: 'Active membrane width normal to the flow.' },
      { key: 'Lpath', label: 'Flow-path length per stage', unit: 'm', value: 1.5, min: 0.05, max: 10, help: 'Active length along the flow; tortuous-path stacks reach several metres.' },
      { key: 'hsp', label: 'Spacer (channel) thickness', unit: 'mm', value: 0.75, min: 0.1, max: 3, typical: [0.3, 1], help: 'Thin channels lower the ohmic loss but raise the pressure drop.' },
      { key: 'eps', label: 'Spacer porosity', unit: '–', value: 0.8, min: 0.4, max: 1, help: 'Open volume fraction of the spacer-filled channel.' },
      { key: 'shadow', label: 'Spacer shadow factor', unit: '–', value: 0.75, min: 0.3, max: 1, help: 'Fraction of the membrane area and channel cross-section open to current.' },
    ] },
    { group: 'Bipolar-membrane ED', showIf: (v) => v.process === 'bpmed', help: 'Three-compartment unit: bipolar, anion and cation membranes.', fields: [
      { key: 'iB', label: 'Current density', unit: 'A/m²', value: 800, min: 50, max: 2000, typical: [500, 1000], help: 'Bipolar membranes are operated far above ED current densities.' },
      { key: 'cProd', label: 'Acid and base product concentration', unit: 'mol/L', value: 1, min: 0.05, max: 4, typical: [0.5, 2], help: 'Higher strength costs current efficiency through proton and hydroxide leakage.' },
      { key: 'bpConv', label: 'Salt conversion', unit: '%', value: 60, min: 5, max: 98, help: 'Share of the feed salt split into acid and base.' },
      { key: 'bpUnits', label: 'Repeating units per stack', unit: '', value: 100, min: 1, max: 500, step: 1, help: 'Used for the number of stacks and electrode pairs.' },
    ] },
    { group: 'Capacitive deionisation', showIf: (v) => v.process === 'mcdi', help: 'Porous carbon electrode pair, with or without ion-exchange membranes.', fields: [
      { key: 'cdiMem', label: 'Ion-exchange membranes on the electrodes (MCDI)', type: 'bool', value: true, help: 'Membranes block co-ion expulsion, raising the charge efficiency towards the permselectivity.' },
      { key: 'Vch', label: 'Charging voltage', unit: 'V', value: 1.2, min: 0.2, max: 1.6, help: 'Kept below about 1.23 V to avoid water electrolysis.' },
      { key: 'Vdis', label: 'Discharge voltage', unit: 'V', value: 0, min: 0, max: 1, help: '0 V = short-circuit regeneration.' },
      { key: 'cdiLoad', label: 'Electrode loading (both electrodes)', unit: 'g/m²', value: 300, min: 20, max: 2000, help: 'Carbon mass per m² of cell area.' },
      { key: 'cdiFlow', label: 'Flow per cell area', unit: 'L/m²·min', value: 1, min: 0.05, max: 20, help: 'Spacer-channel throughput during adsorption and desorption.' },
      { key: 'cdiTads', label: 'Adsorption time', unit: 'min', value: 5, min: 0.2, max: 120, help: 'Half-cycle producing desalinated water.' },
      { key: 'cdiTdes', label: 'Desorption time', unit: 'min', value: 5, min: 0.2, max: 120, help: 'Half-cycle producing concentrate.' },
      { key: 'cdiRecov', label: 'Energy recovered on discharge', unit: '%', value: 0, min: 0, max: 95, help: 'Share of the stored energy returned by the power electronics.' },
    ] },
    { group: 'Membranes', tab: 'setup', showIf: (v) => v.process !== 'mcdi' || v.cdiMem, help: 'Ion-exchange membrane properties.', fields: [
      { key: 'alphaC', label: 'Permselectivity, cation membrane', unit: '–', value: 0.95, min: 0.5, max: 1, help: '1 = ideal: only counter-ions carry the current.' },
      { key: 'alphaA', label: 'Permselectivity, anion membrane', unit: '–', value: 0.93, min: 0.5, max: 1, help: 'Falls at high concentrate salinity.', showIf: (v) => v.process !== 'mcdi' },
      { key: 'Rcem', label: 'Area resistance, cation membrane', unit: 'Ω·cm²', value: 4, min: 0.3, max: 30, showIf: (v) => v.process !== 'mcdi', help: 'Measured in 0.5 mol/L NaCl.' },
      { key: 'Raem', label: 'Area resistance, anion membrane', unit: 'Ω·cm²', value: 4, min: 0.3, max: 30, showIf: (v) => v.process !== 'mcdi', help: 'Measured in 0.5 mol/L NaCl.' },
      { key: 'selDivC', label: 'Divalent / monovalent cation selectivity', unit: '–', value: 1, min: 0.02, max: 5, showIf: isED, help: 'Below 1 for monovalent-selective membranes (Ca²⁺, Mg²⁺ held back).' },
      { key: 'selDivA', label: 'Divalent / monovalent anion selectivity', unit: '–', value: 1, min: 0.02, max: 5, showIf: isED, help: 'Below 1 for monovalent-selective membranes (SO₄²⁻ held back).' },
      { key: 'Ps', label: 'Salt permeability of each membrane', unit: '10⁻⁸ m/s', value: 1, min: 0, max: 50, showIf: isED, help: 'Back-diffusion of salt from concentrate to diluate.' },
      { key: 'tw', label: 'Electro-osmotic water transport', unit: 'mol H₂O per Faraday', value: 10, min: 0, max: 40, showIf: isED, help: 'Hydration water dragged with the ions, both membranes together.' },
      { key: 'Lp', label: 'Osmotic water permeability', unit: 'mL/m²·h·bar', value: 4, min: 0, max: 100, showIf: isED, help: 'Water flux per bar of osmotic-pressure difference, per membrane.' },
      { key: 'Xfix', label: 'Fixed-charge concentration (profile solver)', unit: 'mol/L', value: 3, min: 0.2, max: 10, showIf: isED, help: 'Charge density of the membrane used in the Nernst–Planck–Donnan profile.' },
      { key: 'dMem', label: 'Membrane thickness (profile solver)', unit: 'µm', value: 130, min: 10, max: 1000, showIf: isED, help: 'Ion diffusivity in the membrane is taken as 5 % of the solution value.' },
      { key: 'Rbpm', label: 'Area resistance, bipolar membrane', unit: 'Ω·cm²', value: 5, min: 0.5, max: 50, showIf: (v) => v.process === 'bpmed', help: 'Ohmic part of the bipolar-membrane voltage.' },
      { key: 'bpOver', label: 'Water-splitting overpotential', unit: 'V', value: 0.12, min: 0, max: 1, showIf: (v) => v.process === 'bpmed', help: 'Junction overpotential beyond the 0.83 V thermodynamic value.' },
      { key: 'bpEff', label: 'Current efficiency at low product strength', unit: '%', value: 95, min: 40, max: 100, showIf: (v) => v.process === 'bpmed', help: 'Share of the current that ends up as acid and base.' },
      { key: 'bpLoss', label: 'Efficiency loss per mol/L of product', unit: '%', value: 12, min: 0, max: 40, showIf: (v) => v.process === 'bpmed', help: 'Proton and hydroxide leakage through the monopolar membranes.' },
    ] },
    { group: 'Mass transfer and hydraulics', tab: 'setup', showIf: isED, help: 'Boundary-layer correlation of the spacer-filled channels.', fields: [
      { key: 'shA', label: 'Sherwood coefficient a', unit: '–', value: 0.29, min: 0.05, max: 2, help: 'Sh = a·Re^b·Sc^⅓. Calibrate from a measured limiting current.' },
      { key: 'shB', label: 'Sherwood Reynolds exponent b', unit: '–', value: 0.5, min: 0.3, max: 0.9, help: 'Spacer-filled channels give 0.5–0.7.' },
      { key: 'kdp', label: 'Pressure-drop multiplier', unit: '×', value: 1, min: 0.2, max: 10, help: 'Scales the spacer friction factor f = 6.23 Re^−0.3.' },
      { key: 'dpManifold', label: 'Manifold and piping loss per stage', unit: 'bar', value: 0.3, min: 0, max: 3, showIf: (v) => v.process !== 'mcdi', help: 'Added to the channel pressure drop.' },
      { key: 'plateau', label: 'Limiting-plateau length', unit: 'V per cell pair', value: 0.6, min: 0.05, max: 3, showIf: (v) => v.olModel !== 'rz', help: 'Extra voltage beyond the limiting current before over-limiting conduction sets in.' },
      { key: 'olSlope', label: 'Over-limiting conductance ÷ ohmic conductance', unit: '–', value: 0.5, min: 0, max: 2, showIf: (v) => v.olModel !== 'rz', help: 'Slope of the over-limiting branch of the polarisation curve.' },
      { key: 'ecEll', label: 'Electroconvection: slip cut-off length ÷ δ', unit: '–', value: 0.2, min: 0.1, max: 0.4, showIf: (v) => v.olModel === 'rz', help: 'Short-wave cut-off of the electro-osmotic slip (finite thickness of the extended space charge). The limiting slip formula alone has no preferred wavelength; 0.2 puts the critical vortex pair at the ≈ 2δ seen in experiments and direct simulations.' },
      { key: 'fws', label: 'Share of over-limiting current from water splitting', unit: '–', value: 0.5, min: 0, max: 1, help: 'The remainder is carried by salt through electro-convection.' },
    ] },
    { group: 'Membrane-scale and channel models', tab: 'setup', showIf: isED, help: 'Optional detailed models. The first three change the stack solution; the Poisson–Nernst–Planck and Maxwell–Stefan solvers are local analyses at the stack inlet (continuous operation).', fields: [
      { key: 'membModel', label: 'Membrane potential', type: 'select', value: 'tms', options: [{ value: 'tms', label: 'Permselectivity × Nernst potential' }, { value: 'ghk', label: 'Goldman–Hodgkin–Katz constant-field equation' }], help: 'GHK solves the zero-current potential of each membrane ion by ion for the bulk compositions; concentration polarisation enters through its local slope.' },
      { key: 'olModel', label: 'Over-limiting current', type: 'select', value: 'empirical', options: [{ value: 'empirical', label: 'Empirical plateau length and conductance' }, { value: 'rz', label: 'Electroconvection (Rubinstein–Zaltzman slip model)' }], help: 'The electroconvection model computes the instability threshold and the vortex-enhanced current from a 2-D Stokes / salt-transport solution (adds about a second).' },
      { key: 'flowModel', label: 'Channel flow and mass transfer', type: 'select', value: 'corr', options: [{ value: 'corr', label: 'Spacer correlation Sh = a·Re^b·Sc^⅓' }, { value: 'ns', label: 'Parabolised (boundary-layer) Navier–Stokes march, open channel — fast' }, { value: 'cfd2d', label: 'Full 2-D Navier–Stokes + salt transport, spacer-filled or open channel (finite volumes)' }], help: 'Parabolised: marches the developing laminar flow and the salt boundary layers between two membranes without a spacer (no recirculation, no streamwise diffusion). Full 2-D: solves the elliptic Navier–Stokes equations and the salt balance around the spacer filaments with the finite-volume solver of suite 4. Either way the stack model uses the resulting Sherwood number and pressure gradient.' },
      { key: 'spArr', label: 'Spacer filaments (2-D solution)', type: 'select', value: 'zigzag', options: [{ value: 'zigzag', label: 'Zigzag (alternating membranes)' }, { value: 'cavity', label: 'Cavity (all on one membrane)' }, { value: 'submerged', label: 'Submerged (mid-channel)' }, { value: 'none', label: 'No filaments — open channel over the whole flow path' }], showIf: (v) => isED(v) && v.flowModel === 'cfd2d', help: 'Transverse filaments of the 2-D section. Eight pitches are solved; the Sherwood number and pressure gradient are taken over whole pitches away from the inlet.' },
      { key: 'spDf', label: 'Filament diameter ÷ channel thickness', unit: '–', value: 0.5, min: 0.2, max: 0.8, showIf: (v) => isED(v) && v.flowModel === 'cfd2d' && v.spArr !== 'none', help: 'Woven and extruded ED spacers have filaments of about half the channel thickness.' },
      { key: 'spLm', label: 'Filament pitch ÷ channel thickness', unit: '–', value: 4, min: 2, max: 12, showIf: (v) => isED(v) && v.flowModel === 'cfd2d' && v.spArr !== 'none', help: 'Distance between neighbouring filaments along the flow.' },
      { key: 'pOut', label: 'Channel outlet pressure', unit: 'bar', value: 0.2, min: 0, max: 5, showIf: (v) => v.flowModel === 'ns' || v.flowModel === 'cfd2d', help: 'Gauge pressure at the stage outlet (boundary condition of the momentum equation); the inlet pressure follows.' },
      { key: 'pnp', label: 'Poisson–Nernst–Planck solver', type: 'select', value: 'off', options: [{ value: 'off', label: 'Off' }, { value: 'membrane', label: 'Film | cation membrane | film at the operating current' }, { value: 'wall', label: 'Double layer at an insulating charged wall' }, { value: 'depleted', label: 'Depleted diffusion layer: current–voltage curve with space charge' }], showIf: cont, help: 'Resolves the space-charge regions that the electroneutral model replaces by Donnan jumps. Equivalent 1:1 salt at the stack inlet.' },
      { key: 'pnpPsi0', label: 'Wall potential', unit: 'mV', value: 100, min: 1, max: 300, showIf: (v) => cont(v) && v.pnp === 'wall', help: 'Potential of the insulating wall relative to the bulk solution.' },
      { key: 'msModel', label: 'Maxwell–Stefan transport in the cation membrane', type: 'bool', value: false, showIf: cont, help: 'Multi-ion friction model with electro-osmotic water drag, compared with Nernst–Planck.' },
      { key: 'msWater', label: 'Water volume fraction of the membrane', unit: '–', value: 0.3, min: 0.05, max: 0.7, showIf: (v) => cont(v) && v.msModel, help: 'Sets the water concentration inside the membrane.' },
      { key: 'msDw', label: 'Ion–water diffusivity ÷ solution diffusivity', unit: '–', value: 0.06, min: 0.005, max: 1, showIf: (v) => cont(v) && v.msModel, help: 'Maxwell–Stefan ion–water friction in the membrane pores.' },
      { key: 'msDm', label: 'Ion–polymer diffusivity ÷ solution diffusivity', unit: '–', value: 0.03, min: 0.002, max: 1, showIf: (v) => cont(v) && v.msModel, help: 'Friction of the ions with the fixed charges and the polymer matrix.' },
      { key: 'msDij', label: 'Ion–ion diffusivity', unit: '10⁻¹¹ m²/s', value: 5, min: 0.05, max: 1000, showIf: (v) => cont(v) && v.msModel, help: 'Friction between different ions; large values recover Nernst–Planck.' },
      { key: 'msDwm', label: 'Water–polymer diffusivity', unit: '10⁻¹⁰ m²/s', value: 1.2, min: 0.01, max: 100, showIf: (v) => cont(v) && v.msModel, help: 'Controls how easily the ions drag water through the membrane.' },
    ] },
    { group: 'Stack temperature, fouling, surrogate and optimisation', tab: 'setup', showIf: cont, help: 'Optional modules for continuous ED operation.', fields: [
      { key: 'thermalModel', label: 'Electrochemical–thermal coupling', type: 'bool', value: false, help: 'Joule heat warms the stack; conductivity and diffusivity are re-evaluated at the mean stack temperature (re-solves the stack a few times).' },
      { key: 'thLoss', label: 'Heat lost to the surroundings', unit: '% of heat released', value: 0, min: 0, max: 100, showIf: (v) => v.thermalModel, help: '0 = adiabatic stack.' },
      { key: 'foulModel', label: 'Fouling and scaling module', type: 'bool', value: false, help: 'Growth of organic deposit and mineral scale at constant current, the voltage and energy rise, and the cleaning interval.' },
      { key: 'cFoul', label: 'Charged organic foulant in the feed', unit: 'mg/L', value: 3, min: 0, max: 200, showIf: (v) => v.foulModel, help: 'Humic substances and other anionic organics that deposit on the anion membrane.' },
      { key: 'kDep', label: 'Deposition coefficient', unit: 'g/m²·d per mg/L at 100 A/m²', value: 0.02, min: 0, max: 5, showIf: (v) => v.foulModel, help: 'Electro-deposition rate, proportional to foulant concentration and current density.' },
      { key: 'kDet', label: 'Detachment rate at 8 cm/s', unit: '1/d', value: 0.05, min: 0, max: 20, showIf: (v) => v.foulModel, help: 'First-order shear removal, proportional to the velocity.' },
      { key: 'rFoul', label: 'Resistance of the organic deposit', unit: 'Ω·cm² per g/m²', value: 2, min: 0, max: 100, showIf: (v) => v.foulModel, help: 'Added area resistance per unit deposit.' },
      { key: 'kScale', label: 'Scale growth coefficient', unit: 'g/m²·d', value: 2, min: 0, max: 500, showIf: (v) => v.foulModel, help: 'Rate = k·[(S_gypsum − 1)² + (√S_calcite − 1)²] at the concentrate-side wall; zero below saturation.' },
      { key: 'rScale', label: 'Resistance of the scale layer', unit: 'Ω·cm² per g/m²', value: 0.5, min: 0, max: 100, showIf: (v) => v.foulModel, help: 'Added area resistance per unit scale; scale also blocks membrane area.' },
      { key: 'revEff', label: 'Deposit removed per polarity reversal', unit: '%', value: 60, min: 0, max: 99.9, showIf: (v) => v.foulModel && v.edr, help: 'Share of deposit and scale shed at every reversal.' },
      { key: 'tCamp', label: 'Operating campaign', unit: 'd', value: 60, min: 1, max: 1000, showIf: (v) => v.foulModel, help: 'Simulated time between cleanings.' },
      { key: 'cipTrig', label: 'Cleaning trigger: cell-pair voltage rise', unit: '%', value: 10, min: 1, max: 100, showIf: (v) => v.foulModel, help: 'Clean-in-place is planned when the voltage at constant current has risen by this much.' },
      { key: 'surrogate', label: 'Train a surrogate (machine-learning) model', type: 'bool', value: false, help: 'Neural network, Gaussian process and kernel regression of specific energy and salt removal versus feed salinity, voltage and velocity, compared on held-out runs; the best on validation is used. Trained on runs of the mechanistic stack model and tested on held-out runs.' },
      { key: 'nTrain', label: 'Training runs', unit: '', value: 48, min: 16, max: 120, step: 1, showIf: (v) => v.surrogate, help: 'Latin-hypercube sample; 16 further runs are held out for the parity test.' },
      { key: 'optimise', label: 'Optimise the operating conditions', type: 'bool', value: false, showIf: (v) => v.mode === 'design', help: 'Nelder–Mead search (started from a coarse grid of re-designs, shown as the cost map) of the allowed current fraction (which sets the number of stages) and the velocity (which sets the cell pairs) for the lowest unit cost.' },
      { key: 'cElecED', label: 'Electricity price', unit: '$/kWh', value: 0.08, min: 0, max: 1, showIf: (v) => v.mode === 'design' && v.optimise, help: 'For the energy part of the unit cost.' },
      { key: 'cMem', label: 'Installed membrane cost', unit: '$/m²', value: 100, min: 1, max: 2000, showIf: (v) => v.mode === 'design' && v.optimise, help: 'Membranes with spacers and stack hardware per m² of membrane.' },
      { key: 'memLife', label: 'Membrane life', unit: 'y', value: 7, min: 0.5, max: 20, showIf: (v) => v.mode === 'design' && v.optimise, help: 'Straight-line replacement over this period at 90 % availability.' },
    ] },
    { group: 'Electrodes and power supply', tab: 'setup', showIf: (v) => v.process !== 'mcdi', help: 'Boundary conditions at the electrode compartments.', fields: [
      { key: 'i0a', label: 'Anode exchange current density', unit: 'A/m²', value: 0.001, min: 1e-6, max: 100, help: 'Oxygen evolution on a mixed-metal-oxide anode.' },
      { key: 'i0c', label: 'Cathode exchange current density', unit: 'A/m²', value: 0.1, min: 1e-5, max: 1000, help: 'Hydrogen evolution on stainless steel or nickel.' },
      { key: 'kinetics', label: 'Electrode kinetics', type: 'select', value: 'bv', options: [{ value: 'bv', label: 'Butler–Volmer equation' }, { value: 'tafel', label: 'Tafel equation (high-overpotential limit)' }], help: 'Butler–Volmer: i = i₀[exp(αFη/RT) − exp(−(1−α)Fη/RT)]; Tafel: η = (RT/αF)·ln(i/i₀).' },
      { key: 'alphaBV', label: 'Charge-transfer coefficient α', unit: '–', value: 0.5, min: 0.05, max: 0.95, help: '0.5 gives the symmetric Butler–Volmer equation; the Tafel slope is 2.303·RT/(αF).' },
      { key: 'Rrinse', label: 'Electrode-rinse compartment resistance', unit: 'Ω·cm²', value: 20, min: 0, max: 500, help: 'Both rinse compartments and end membranes together.' },
      { key: 'shunt', label: 'Shunt (manifold leakage) current', unit: '%', value: 2, min: 0, max: 30, help: 'Share of the electrode current bypassing the cells through the manifolds.' },
      { key: 'etaRect', label: 'Rectifier efficiency', unit: '%', value: 95, min: 60, max: 100, help: 'AC to DC conversion.' },
      { key: 'etaPump', label: 'Pump + motor efficiency', unit: '%', value: 70, min: 20, max: 92, help: 'For the circulation pumping energy.' },
    ] },
    { group: 'Electrode double layer', tab: 'setup', showIf: (v) => v.process === 'mcdi', help: 'Electrical double-layer and cell parameters of the carbon electrodes.', fields: [
      { key: 'vmi', label: 'Micropore volume', unit: 'mL/g', value: 0.4, min: 0.05, max: 1.5, help: 'Volume of the charge-storing micropores per gram of electrode.' },
      { key: 'cStern', label: 'Volumetric Stern capacitance', unit: 'F/mL', value: 150, min: 20, max: 500, help: 'Modified-Donnan model.' },
      { key: 'muAtt', label: 'Micropore attraction energy', unit: 'kT', value: 1.5, min: 0, max: 4, help: 'Non-electrostatic ion attraction into the micropores.' },
      { key: 'cSternA', label: 'Areal Stern capacitance (Gouy–Chapman–Stern)', unit: 'F/m²', value: 0.2, min: 0.02, max: 2, help: 'Used for the planar double-layer comparison curve.' },
      { key: 'aBET', label: 'Effective double-layer area', unit: 'm²/g', value: 600, min: 50, max: 3000, help: 'Ion-accessible area for the Gouy–Chapman–Stern curve.' },
      { key: 'cdiESR', label: 'Cell series resistance', unit: 'Ω·cm²', value: 60, min: 2, max: 2000, help: 'Spacer, electrode, membrane and contact resistances.' },
    ] },
    { group: 'Discretisation', tab: 'mesh', help: 'Numerical resolution along the flow path and in time.', fields: [
      { key: 'nSeg', label: 'Segments per stage along the flow path', unit: '', value: 12, min: 2, max: 200, step: 1, help: 'Midpoint (second-order) marching; use the sensitivity study to confirm convergence.' },
      { key: 'nt', label: 'Time steps (batch and CDI cycle)', unit: '', value: 120, min: 4, max: 2000, step: 1, help: 'RK4 steps for batch ED and for each CDI half-cycle.' },
      { key: 'nProf', label: 'Points across the membrane (profile solver)', unit: '', value: 40, min: 6, max: 400, step: 1, help: 'RK4 steps of the Nernst–Planck–Donnan and Maxwell–Stefan profiles inside the membrane.' },
      { key: 'pnpRatio', label: 'Poisson–Nernst–Planck mesh growth ratio', unit: '–', value: 1.25, min: 1.05, max: 2, showIf: (v) => cont(v) && v.pnp && v.pnp !== 'off', help: 'Ratio of neighbouring cell sizes away from each interface; the first cell is a fifth of the local Debye length. Smaller = finer.' },
      { key: 'cfdNy', label: '2-D Navier–Stokes cells across the channel', unit: '', value: 24, min: 12, max: 48, step: 1, showIf: (v) => isED(v) && v.flowModel === 'cfd2d', help: 'Finite volumes across the channel, clustered towards both membranes; the streamwise count follows (20 per filament pitch at 24 across).' },
      { key: 'ecN', label: 'Electroconvection grid cells across the layer', unit: '', value: 20, min: 12, max: 48, step: 1, showIf: (v) => isED(v) && v.olModel === 'rz', help: 'Cells across the depleted diffusion layer on the medium grid of the 2-D vortex solution. The model also solves a 1.5 × coarser and a 1.5 × finer grid and Richardson-extrapolates, so the reported over-limiting current does not depend on this number beyond the stated grid-convergence index.' },
    ] },
  ],

  presets: [
    { name: 'Brackish 3.5 g/L → 500 mg/L, EDR design', values: {} },
    { name: 'EDR on low-salinity brackish water, 90 % recovery', values: { ions: WATERS.lowbrackish.ions, pH: 7.8, T: 20, targetTDS: 250, recovery: 90, edr: true, Qp: 250 } },
    { name: 'Monovalent-selective ED: nitrate and chloride removal', values: { selDivC: 0.15, selDivA: 0.1, targetTDS: 1500, recovery: 85 } },
    { name: 'Batch ED, 5 m³ tank at 0.7 V per cell pair', values: { mode: 'batch', Ucp: 0.7, Ncp: 150, Vbatch: 5, tBatch: 120 } },
    { name: 'Bipolar-membrane ED of RO brine to 1 mol/L acid and base', values: { process: 'bpmed', ions: WATERS.robrine.ions, Qp: 10, pH: 7.9 } },
    { name: 'Membrane capacitive deionisation, 1 g/L', values: { process: 'mcdi', ions: WATERS.lowbrackish.ions, Qp: 5, pH: 7.8 } },
  ],

  pull: ({ feed, outputs }) => [
    feed?.ions ? { key: 'ions', value: feed.ions, from: 'Case feed water' } : null, feed?.T ? { key: 'T', value: clamp(feed.T, 5, 45), from: 'Case feed water' } : null, feed?.pH ? { key: 'pH', value: feed.pH, from: 'Case feed water' } : null,
    outputs?.ro?.streams?.permeate?.ions ? { key: 'ions', value: outputs.ro.streams.permeate.ions, from: 'RO permeate (polishing by ED)' } : null,
    outputs?.ro?.streams?.concentrate?.ions ? { key: 'ions', value: outputs.ro.streams.concentrate.ions, from: 'RO concentrate (brine valorisation by ED / BPMED)' } : null,
    outputs?.cfd?.sherwood && outputs?.cfd?.reynolds ? { key: 'shA', value: clamp(outputs.cfd.sherwood / (outputs.cfd.reynolds ** 0.5 * 8.4), 0.05, 2), from: 'CFD Sherwood number' } : null,
  ],
  site: (site) => [{ key: 'T', value: site?.data?.sst !== undefined && site?.data?.sst !== null ? clamp(site.data.sst, 5, 45) : undefined, from: 'Sea-surface temperature at site' }],

  run(v, ctx) { // synchronous by default; asynchronous with the 2-D Navier–Stokes channel option (finite-volume solver) and, when the caller can yield, with the longer opt-in sub-models
    const go = () => (v.process === 'bpmed' ? runBPMED(v) : v.process === 'mcdi' ? runCDI(v) : v.mode === 'batch' ? runBatch(v) : runED(v));
    if (isED(v) && v.flowModel === 'cfd2d') { ctx?.progress?.(0.05, 'Solving the 2-D Navier–Stokes channel flow'); return prepareChannelCFD(v, ctx).then(async () => { ctx?.progress?.(0.6, 'Solving the stack'); await ctx?.tick?.(); return go(); }); }
    if (isED(v) && (v.olModel === 'rz' || v.surrogate || v.optimise || (v.pnp && v.pnp !== 'off')) && ctx?.tick) return (async () => { ctx.progress?.(0.2, 'Solving the stack and the selected sub-models'); await ctx.tick(); return go(); })(); // yield once so the interface can show progress before the longer sub-models
    return go();
  },

  mesh: [
    { name: 'Segments along the flow path', keys: ['nSeg'], min: 2, note: 'Stages, cell pairs and voltages are re-solved at each resolution.', metrics: [{ label: 'Specific energy', unit: 'kWh/m³', get: (r) => r.outputs.sec }, { label: 'Product TDS', unit: 'mg/L', get: (r) => r.outputs.streams.diluate.tds }, { label: 'Mean current density', unit: 'A/m²', get: (r) => r.outputs.currentDensity }] },
    { name: 'Time steps (batch ED and CDI cycle)', keys: ['nt'], min: 4, metrics: [{ label: 'Specific energy', unit: 'kWh/m³', get: (r) => r.outputs.sec }, { label: 'Product TDS', unit: 'mg/L', get: (r) => r.outputs.streams.diluate.tds }] },
  ],

  calibration: {
    note: 'Fit the membrane resistance, the permselectivity and the Sherwood coefficient to stack tests. Each row is one steady single-pass test of the configured stack: voltage per cell pair, linear velocity and salinity multiplier set the condition; the path-averaged current density, the product TDS and the current efficiency are the measurements. Include rows near the limiting current so that the Sherwood coefficient is identifiable.',
    params: [{ key: 'Rcem', label: 'Membrane area resistance (cation; anion kept)', lo: 0.5, hi: 20 }, { key: 'alphaC', label: 'Permselectivity (cation membrane)', lo: 0.7, hi: 1 }, { key: 'shA', label: 'Sherwood coefficient a', lo: 0.08, hi: 1.2 }],
    columns: [{ key: 'Ucp', label: 'Voltage per cell pair', unit: 'V' }, { key: 'uLin', label: 'Linear velocity', unit: 'cm/s' }, { key: 'salinityFactor', label: 'Salinity ×', unit: '–' }, { key: 'iAvg', label: 'Current density', unit: 'A/m²' }, { key: 'tdsOut', label: 'Product TDS', unit: 'mg/L' }, { key: 'eff', label: 'Current efficiency', unit: '%' }],
    targets: [{ key: 'iAvg', label: 'Current density', unit: 'A/m²' }, { key: 'tdsOut', label: 'Product TDS', unit: 'mg/L' }, { key: 'eff', label: 'Current efficiency', unit: '%' }],
    model(v) {
      const par = params(v), G = geometry(v), cf = toMolar(scaleIons(cloneIons(v.ions), v.salinityFactor ?? 1)), neutral = (v.salinityFactor ?? 1) * ((+v.ions.SiO2 || 0) + (+v.ions.B || 0));
      const q = (v.uLin / 100) * G.W * G.h * G.eps, cc = cf.map((c) => c * 2.5), m = marchStage({ qd: q, qc: q, cd: cf, cc }, v.Ucp, G, par, v.T, Math.min(Math.max(2, Math.round(v.nSeg)), 8));
      const eq = sum(cf.map((c, j) => (Z[j] > 0 ? AZ[j] * (q * c - m.st.qd * m.st.cd[j]) : 0)));
      return { iAvg: m.iAvg, tdsOut: tdsOf(m.st.cd, neutral), eff: m.I > 0 ? (100 * F * eq) / m.I : 0 };
    },
    get sample() { return (this._s ||= synth(3, [[0.3, 8, 1], [0.45, 8, 1], [0.6, 8, 1], [0.8, 8, 1], [1.1, 8, 1], [0.6, 5, 1], [0.6, 12, 1], [0.9, 12, 1.5], [0.5, 8, 0.6], [1.3, 6, 1]])); },
    get validationSample() { return (this._v ||= synth(23, [[0.4, 10, 1], [0.7, 10, 1], [1.0, 10, 1], [0.55, 6, 0.8], [0.85, 14, 1.3], [1.2, 9, 1], [0.35, 7, 1.2]])); },
  },

  async verify() {
    const d = defaultsOf(suite), C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    add('Nernst potential per decade at 25 °C', 59.16, 1000 * nernst(10, 25), 0.01, 'RT/F · ln 10 (mV)');
    add('Donnan potential (Teorell–Meyer–Sievers)', -vt(25) * Math.log((3000 + Math.sqrt(3000 ** 2 + 4 * 100 ** 2)) / 200), donnanPotential(100, 3000), 1e-12, 'Δφ = −(RT/F)·asinh(X/2c) = −(RT/F)·ln(c₊ᵐ/c); X = 3 mol/L, c = 0.1 mol/L');
    // ideal stack: Faraday's law
    const ideal = { ...d, alphaC: 1, alphaA: 1, Ps: 0, tw: 0, Lp: 0, shunt: 0, mode: 'voltage', nStages: 1, Ncp: 100, Ucp: 0.5 }, ri = simulateED(ideal);
    add("Faraday's law: 1 equivalent removed per 96 485 C", 1, ri.effMem, 1e-9, 'Ideal membranes, no back-diffusion: F·ṅ_eq / I = 1');
    const r = simulateED(d), t = r.tr;
    const ionErr = Math.max(...r.cf.map((c, j) => (c > 0 ? Math.abs(t.Qf * c - t.Qp * t.cp[j] - t.Qbd * t.ccOut[j]) / (t.Qf * c) : 0)));
    add('Every ion is conserved', 0, ionErr, 1e-9, 'Feed = diluate + concentrate blow-down for each of the charged species (largest relative error)');
    add('Water balance closes', 0, (t.Qf - t.Qp - t.Qbd) / t.Qf, 1e-12, 'Feed = product + blow-down');
    add('Diluate stays electroneutral', 0, chargeBalance(r.ionsP).errorPct, 1e-6, 'Σ zᵢcᵢ = 0 in the product (% of total equivalents); the feed analysis is charge-balanced first');
    add('Design reaches the target product TDS', d.targetTDS, r.tdsP, 0.5, 'Solver on the operating fraction of the limiting current');
    const G = geometry(d), par = params(d), cf = r.cf, c2 = cf.map((c) => 2 * c), lc = (c, u) => cellPair(c, c.map((x) => 2 * x), u, G, par, 25).ilim;
    add('Limiting current density ∝ concentration', 2, lc(c2, 0.08) / lc(cf, 0.08), 0.06, 'Doubling the diluate concentration at fixed velocity (ionic-strength effects on transport numbers are small)');
    add('Limiting current density ∝ velocity^b', d.shB, Math.log(lc(cf, 0.16) / lc(cf, 0.08)) / Math.log(2), 1e-6, 'Exponent recovered from two velocities; literature 0.5–0.7');
    const na = toMolar(cloneIons({ Na: 1149.5, Cl: 1772.65 })), cpN = cellPair(na, na, 0.08, G, par, 25), tm = 0.5 * (1 + par.alphaC), Dn = (2 * IONS.Na.D * IONS.Cl.D) / (IONS.Na.D + IONS.Cl.D), ts = IONS.Na.D / (IONS.Na.D + IONS.Cl.D);
    add('Film solution reproduces i_lim = F·k·c / (t̄ − t)', (F * cpN.hd.k * 50) / (tm - ts), cpN.ilimC, 1e-6 * cpN.ilimC, `50 mol/m³ NaCl, D = ${fmt(Dn, 4)} m²/s, cation membrane`);
    add('No current at zero voltage with equal concentrations', 0, currentAt(cpN, 0, par).i, 0, 'Limiting case: zero driving force');
    add('Cell-pair voltage equals the sum of its parts', 0, (() => { const p = cpN.parts(0.5 * cpN.ilim); return p.U - (p.donnan + p.polar + p.ohmMem + p.ohmD + p.ohmC); })(), 1e-12, 'Voltage balance at 50 % of the limiting current (V)');
    const pf = npProfile({ cd: 50, cc: 100, i: 0.5 * cpN.ilimC, X: 3000, dm: 130e-6, Dp: IONS.Na.D, Dm: IONS.Cl.D, DpM: 0.05 * IONS.Na.D, DmM: 0.05 * IONS.Cl.D, deltaD: cpN.hd.delta, deltaC: cpN.hd.delta, n: 40 });
    add('Profile solver meets the concentrate boundary condition', 0, pf.residual / 100, 1e-8, 'Shooting on the co-ion flux; residual relative to the bulk concentration');
    add('Profile solver: Donnan potential at the diluate face', donnanPotential(pf.cwd, 3000), pf.donL, 1e-10, 'Interface jump equals the TMS formula at the wall concentration (V)');
    const p0 = npProfile({ cd: 100, cc: 100, i: 0, X: 3000, dm: 130e-6, Dp: IONS.Na.D, Dm: IONS.Cl.D, DpM: 0.05 * IONS.Na.D, DmM: 0.05 * IONS.Cl.D, deltaD: 5e-5, deltaC: 5e-5 });
    add('Symmetric electrolyte, zero current: no membrane potential', 0, p0.potential, 1e-9, 'Limiting case (V)');
    add('Brackish 3.5 g/L → 0.5 g/L specific energy', 1.35, r.sec, 0.65, 'Literature 0.7–2 kWh/m³ for about 3 g/L → 0.5 g/L');
    add('Energy use exceeds the reversible work of separation', 1, r.sec > r.wMin && r.wMin > 0 ? 1 : 0, 0, `Second law: ${fmt(r.sec, 3)} vs ${fmt(r.wMin, 3)} kWh/m³`);
    const a = simulateED(d, { nSeg: 12 }), b = simulateED(d, { nSeg: 24 });
    add('Result is insensitive to path refinement (12 → 24 segments)', 0, Math.abs(a.sec - b.sec) / b.sec, 5e-3, 'Relative change of the specific energy');
    add('Bipolar junction voltage for 1 mol/L acid and base', 0.828, bpmUnit(0, 1, 1, 500, d).Ews, 0.002, '(RT/F)·ln10·ΔpH with ΔpH = 14 (V)');
    const bp = simulateBPMED({ ...d, ...suite.presets[4].values });
    add('BPMED energy per kg NaOH above the thermodynamic minimum', 1, bp.ePerKg > bp.eTheo && bp.ePerKg < 4 ? 1 : 0, 0, `${fmt(bp.ePerKg, 3)} kWh/kg vs minimum ${fmt(bp.eTheo, 3)} kWh/kg; literature 1.3–2.5`);
    const cv = { ...d, muAtt: 0, cdiMem: false }, e0 = cdiEquilibrium(20, cv, 1.0, 0);
    add('CDI charge efficiency equals tanh(Δφ_D / 2) without attraction term', Math.tanh(e0.a.phiD / vt(25) / 2), e0.effCDI, 1e-9, 'Modified-Donnan model, μ_att = 0, discharge at 0 V');
    const gc = gcs(0.5, 20, d);
    add('Gouy–Chapman–Stern charge efficiency equals tanh(Δφ_d / 4)', Math.tanh(gc.phiD / vt(25) / 4), gc.w / gc.sigma, 1e-12, 'Salt excess ÷ surface charge of a planar double layer');
    // ---- Poisson–Nernst–Planck solver
    const V25 = vt(25), D2 = [IONS.Na.D, IONS.Cl.D], D2m = D2.map((x) => 0.05 * x), sys = (cl, cr, Vv) => ({ z: [1, -1], layers: [{ L: 5e-5, D: D2 }, { L: 130e-6, D: D2m, X: -3000 }, { L: 5e-5, D: D2 }], left: { type: 'bulk', c: [cl, cl], psi: Vv }, right: { type: 'bulk', c: [cr, cr], psi: 0 } });
    const pq = solvePNP(sys(100, 100, 0)), kMid = pq.x.findIndex((x) => x >= 115e-6);
    add('Poisson–Nernst–Planck: Donnan potential of the membrane at equilibrium', donnanPotential(100, 3000), pq.psi[kMid], 1e-9, `Potential in the membrane core against 0.1 mol/L on both sides (V); ${pq.nodes} nodes, zero applied voltage`);
    add('Poisson–Nernst–Planck: no current at equilibrium', 0, pq.current, 1e-6, 'A/m²');
    const lamV = debyeLength(20), gw = pnpRamp({ z: [1, -1], layers: [{ L: 40 * lamV, D: D2, n: 30 }], left: { type: 'wall', psi: 0.1, flux: [0, 0] }, right: { type: 'bulk', c: [10, 10], psi: 0 }, res: 20, ratio: 1.1 }, { left: { psi: 0 } });
    add('Poisson equation: Gouy–Chapman potential decay at a charged wall', 0, Math.max(...gw.x.map((x, k) => Math.abs(gw.psi[k] - 4 * V25 * Math.atanh(Math.tanh(0.1 / (4 * V25)) * Math.exp(-x / lamV))))), 1e-4, 'Largest deviation from ψ = 4(RT/F)·atanh[tanh(Fψ₀/4RT)·exp(−x/λ_D)] for ψ₀ = 100 mV in 10 mol/m³ (V)');
    const grah = Math.sqrt(8 * 78.4 * EPS0 * R * 298.15 * 10) * Math.sinh(0.1 / (2 * V25));
    add('Poisson–Nernst–Planck: surface charge equals the Grahame equation', grah, gw.sigmaLeft, 3e-3 * grah, 'σ = √(8εRTc)·sinh(Fψ₀/2RT), C/m²');
    add('Insulating wall: zero ion flux gives zero current', 0, gw.current, 1e-6, 'No-flux boundary for both ions (A/m²)');
    const ljs = (Vv) => ({ z: [1, -1], layers: [{ L: 1e-4, D: D2, n: 60 }], left: { type: 'bulk', c: [100, 100], psi: Vv }, right: { type: 'bulk', c: [10, 10], psi: 0 } });
    let ja = solvePNP(ljs(0)), Vq = 0.012, jb = pnpContinue(ljs, ja, 0, Vq), Vprev = 0;
    for (let q = 0; q < 8 && Math.abs(jb.current) > 1e-9; q++) { const Vn = Vq - (jb.current * (Vq - Vprev)) / (jb.current - ja.current); ja = jb; Vprev = Vq; jb = pnpContinue(ljs, jb, Vq, Vn); Vq = Vn; }
    add('Poisson–Nernst–Planck: diffusion potential of a salt gradient (Planck–Henderson)', (V25 * (D2[1] - D2[0]) * Math.log(10)) / (D2[0] + D2[1]), Vq, 4e-5, 'Zero-current potential (RT/F)·(D₋ − D₊)/(D₊ + D₋)·ln(c_L/c_R) of the electroneutral bulk (V)');
    add('Poisson–Nernst–Planck: electroneutral salt flux in the bulk', (2 * D2[0] * D2[1] * 90) / (D2[0] + D2[1]) / 1e-4, jb.J[0], 2e-7, 'J = D_s·Δc/L with D_s = 2D₊D₋/(D₊ + D₋), mol/m²·s');
    const pnE = npProfile({ cd: 50, cc: 100, i: 150, X: 3000, dm: 130e-6, Dp: D2[0], Dm: D2[1], DpM: D2m[0], DmM: D2m[1], deltaD: 5e-5, deltaC: 5e-5, n: 80 }), pnP = pnpRamp(sys(50, 100, pnE.potential), { left: { psi: 0, c: [70, 70] }, right: { c: [70, 70] } });
    add('Poisson–Nernst–Planck agrees with the electroneutral Nernst–Planck–Donnan model', 150, pnP.current, 1.5, 'Current through film | membrane | film at the voltage the electroneutral model needs for 150 A/m² (the Debye length is far below every layer thickness)');
    add('Poisson–Nernst–Planck: co-ion leakage through the membrane', pnE.Jm, pnP.J[1], 0.02 * Math.abs(pnE.Jm), 'Co-ion flux of both models (mol/m²·s)');
    add('Poisson–Nernst–Planck: fluxes are uniform over the mesh', 0, pnP.fluxSpread, 1e-6, 'Current continuity: largest face-to-face difference of an ion flux ÷ largest flux');
    const dls = (Vv) => ({ z: [1, -1], ratio: 1.15, layers: [{ L: 5e-5, D: D2, n: 30 }], left: { type: 'bulk', c: [50, 50], psi: 0 }, right: { type: 'wall', cFix: [3000, null], flux: [0, 0], psi: -(Math.log(60) * V25 + Vv) } });
    const d0 = pnpRamp(dls(0), { right: { psi: 0, cFix: [50, null] } }), d1 = pnpContinue(dls, d0, 0, V25), d2 = pnpContinue(dls, d1, V25, 12 * V25), ilimP = (2 * F * D2[0] * 50) / 5e-5;
    add('Poisson–Nernst–Planck: polarisation curve below the limiting current', 1 - Math.exp(-0.5), d1.current / ilimP, 2e-3, 'i/i_lim = 1 − exp(−FV/2RT) at V = RT/F for an ideal cation-exchange surface (specified counter-ion concentration, no co-ion flux)');
    add('Poisson–Nernst–Planck: limiting current plateau 2·F·D₊·c/δ', 1, d2.current / ilimP, 0.01, 'At 12 thermal voltages; the small excess is carried by the extended space charge');
    // ---- Goldman–Hodgkin–Katz, electrode kinetics
    add('Goldman–Hodgkin–Katz potential reduces to the Nernst potential', nernst(10, 25), ghkPotential([1, 0], [1, -1], [10, 10], [100, 100], 25), 1e-10, 'Ideally selective membrane, single salt, tenfold ratio (V)');
    add('Goldman–Hodgkin–Katz slope at small ratios equals the permselectivity', 0.95, ghkPotential([1, 0.05 / 1.95], [1, -1], [100, 100], [101, 101], 25) / (V25 * Math.log(1.01)), 1e-4, 'P₋/P₊ = (1 − α)/(1 + α)');
    const gU = ghkPotential([1, 0.5, 0.02, 0.02], [1, 2, -1, -2], [10, 5, 10, 5], [40, 20, 40, 20], 25) / V25, gI = sum([1, 2, -1, -2].map((zz, j) => zz * ghkFlux([1, 0.5, 0.02, 0.02][j], zz, [10, 5, 10, 5][j], [40, 20, 40, 20][j], gU)));
    add('Goldman–Hodgkin–Katz: no net current at the solved potential (mixed valences)', 0, gI, 1e-9, 'Σ zᵢJᵢ = 0 for a mono- and divalent mixture');
    const gk = simulateED({ ...ideal, alphaC: 0.95, alphaA: 0.93, membModel: 'ghk' }), tk = simulateED({ ...ideal, alphaC: 0.95, alphaA: 0.93 });
    add('GHK and permselectivity models give similar stack currents', 1, gk.iAvg / tk.iAvg, 0.1, 'Same stack at 0.5 V per cell pair; the difference is the bi-ionic potential of the mixed feed');
    add('Tafel slope at α = 0.5 and 25 °C', 118.32, 1000 * (overpotential(1000, 1, 25, 0.5, 'tafel') - overpotential(100, 1, 25, 0.5, 'tafel')), 0.01, '2.303·RT/(αF), mV per decade');
    add('Butler–Volmer approaches the Tafel line at high current', overpotential(1e4, 1, 25, 0.3, 'tafel'), overpotential(1e4, 1, 25, 0.3), 1e-9, 'α = 0.3, i/i₀ = 10⁴ (V)');
    add('Butler–Volmer is linear at low current: η = RT·i/(F·i₀)', 1, overpotential(1e-4, 0.1, 25, 0.3) / (V25 * 1e-3), 1e-3, 'Ratio to the charge-transfer resistance, any α');
    add('Butler–Volmer solution satisfies the rate equation', 100, (() => { const e = overpotential(100, 0.1, 25, 0.3); return 0.1 * (Math.exp((0.3 * e) / V25) - Math.exp((-0.7 * e) / V25)); })(), 1e-8, 'i₀[exp(αFη/RT) − exp(−(1−α)Fη/RT)] at the returned η (A/m²)');
    // ---- Maxwell–Stefan
    const msd = msMembrane({ z: [1, -1], cL: [50, 50], cR: [100, 100], X: -3000, dm: 130e-6, Diw: D2m, Dim: [Infinity, Infinity], Dwm: 1e-30, cw: 1e12, i: 100 }), npd = npProfile({ cd: 50, cc: 100, i: 100, X: 3000, dm: 130e-6, Dp: D2[0], Dm: D2[1], DpM: D2m[0], DmM: D2m[1], deltaD: 1e-12, deltaC: 1e-12, n: 40 });
    add('Maxwell–Stefan reduces to Nernst–Planck in the dilute limit (co-ion flux)', npd.Jm, msd.N[1], 1e-4 * Math.abs(npd.Jm), 'No ion–ion friction, immobile water, ions dilute in the membrane water: compared with the independent Nernst–Planck–Donnan shooting solver (mol/m²·s)');
    add('Maxwell–Stefan reduces to Nernst–Planck in the dilute limit (counter-ion flux)', npd.Jp, msd.N[0], 1e-6 * npd.Jp, 'mol/m²·s');
    const msf = msMembrane({ z: [1, 2, -1, -2], cL: [40, 5, 40, 5], cR: [80, 10, 80, 10], X: -3000, dm: 130e-6, Diw: [1.33e-9, 0.79e-9, 2.03e-9, 1.07e-9].map((x) => 0.06 * x), Dim: [1.33e-9, 0.79e-9, 2.03e-9, 1.07e-9].map((x) => 0.03 * x), Dij: 5e-11, Dwm: 1.2e-10, cw: 16650, i: 100 });
    add('Maxwell–Stefan: transport numbers sum to one', 1, sum(msf.t), 1e-9, 'Four ions with friction to water, polymer and each other; F·Σ zᵢNᵢ = i');
    add('Maxwell–Stefan: ions drag water with them (electro-osmosis)', 1, msf.converged && msf.tw > 0.5 && msf.tw < 30 ? 1 : 0, 0, `Water transport number ${fmt(msf.tw, 3)} mol per Faraday (literature 2–12 per membrane)`);
    // ---- electroconvection
    const sm = stokesMode(Math.PI);
    add('Stokes mode satisfies its wall conditions', 0, Math.abs(sm.W(0)) + Math.abs(sm.W(1)) + Math.abs(sm.dW(1)) + Math.abs(sm.dW(0) - 1), 1e-10, 'W(0) = W(1) = W′(1) = 0, W′(0) = 1');
    add('Electroconvection: short-wave limit of the marginal voltage', Math.sqrt(32 / 0.32), ecMarginal(20, 0.32), 1e-3, 'Linear stability of the limiting slip model: Pe·V² → 32 as k → ∞ (thermal voltages)');
    const vcE = ecMarginal(Math.PI, 0.32), eLo = ecSolve({ V: 0.85 * vcE, Pe: 0.32, k: Math.PI, ny: 16, tEnd: 3 }), eHi = ecSolve({ V: 1.5 * vcE, Pe: 0.32, k: Math.PI, ny: 16, tEnd: 3 });
    add('Electroconvection: the quiescent layer is stable below the threshold', 1, eLo.nu, 2e-3, 'Non-linear 2-D solution at 0.85 × the linear-stability voltage decays to i = i_lim');
    add('Electroconvection: vortices raise the current above the threshold', 1, eHi.nu > 1.15 && eHi.nu < 4 ? 1 : 0, 0, `i/i_lim = ${fmt(eHi.nu, 4)} at 1.5 × threshold (measured over-limiting currents reach 1.5–3 × i_lim)`);
    const ecr = electroconvection(0.32, { ny: 20 }), ecr2 = electroconvection(0.32, { ny: 14 });
    add('Electroconvection: three-grid sequence is monotone with a grid-convergence index below 3 %', 1, ecr.monotone && ecr.steady && ecr.gciPct < 3 ? 1 : 0, 0, `i/i_lim at 1.7 × threshold: ${ecr.richardson.nu.map((q) => fmt(q, 5)).join(' → ')}, extrapolated ${fmt(ecr.nuExtrap, 5)}, GCI ${fmt(ecr.gciPct, 3)} %`);
    add('Electroconvection: observed order of convergence matches the second-order scheme', 2, ecr.order, 0.4, 'Richardson estimate from the three grids (central differences, clustered grid, Peaceman–Rachford ADI)');
    add('Electroconvection: extrapolated over-limiting current is independent of the grid input', ecr.nuExtrap, ecr2.nuExtrap, 0.02 * ecr.nuExtrap, `14 and 20 cells across the layer on the medium grid (previously the result drifted by 15 % between 16 and 32 cells)`);
    add('Electroconvection: critical wavenumber is the minimum of the marginal curve', 1, ecMarginal(ecr.kc, 0.32, 0.2) <= Math.min(ecMarginal(0.9 * ecr.kc, 0.32, 0.2), ecMarginal(1.1 * ecr.kc, 0.32, 0.2)) ? 1 : 0, 0, `k_c·δ = ${fmt(ecr.kc, 4)} with the slip cut-off 0.2 δ`);
    const ecp = { ...par, ol: { Vc: vcE, slope: 0.1 } }, cpE = cellPair(na, na, 0.08, G, ecp, 25), uT = cpE.U(0.98 * cpE.ilim) + 2 * vcE * V25;
    add('Electroconvection sets the plateau length of the cell pair', 0, currentAt(cpE, uT, ecp).iOver + (currentAt(cpE, uT + 0.2, ecp).iOver > 0 ? 0 : 1), 1e-12, 'No over-limiting current up to two depleted layers at the threshold voltage, some beyond');
    // ---- Navier–Stokes / Nernst–Planck channel
    const nsv = channelNS({ U: 0.3, h: 2e-3, L: 3, D: 1.6e-9, ny: 96, growth: 1.04 });
    add('Navier–Stokes: fully developed friction f·Re = 96', 96, nsv.fRe, 0.1, 'Parallel plates, Darcy friction factor on the hydraulic diameter 2h');
    add('Navier–Stokes: fully developed centre-line velocity 1.5·U', 1.5, nsv.uc[nsv.uc.length - 1], 2e-3, 'Parabolic profile at the outlet');
    add('Navier–Stokes: entrance length L_e ≈ 0.011·Re·D_h', 0.011, nsv.entrance / (4e-3 * nsv.Re), 0.002, 'Distance to 99 % of the developed centre-line velocity');
    add('Navier–Stokes: incremental pressure drop of the entrance', 0.67, nsv.Kinc, 0.08, '(Δp − Δp_Poiseuille)/(½ρU²); literature 0.64–0.69 for parallel plates');
    // ---- full 2-D Navier–Stokes + salt transport (finite volumes of suite 4)
    const cfE = await channelCFD({ U: 0.08, h: 0.75e-3, L: 0.2, D: 1e-6, c0: 1, jw: 1e-6, ny: 20 }), cfS = await channelCFD({ U: 0.064, h: 0.75e-3, L: 1.5, D: 1.6e-9, arr: 'zigzag', ny: 16 }), cfO = await channelCFD({ U: 0.064, h: 0.75e-3, L: 1.5, D: 1.6e-9, ny: 16 });
    add('2-D Navier–Stokes, empty channel: friction factor × Reynolds number', 96, cfE.fRe, 1.5, 'Fully developed plane Poiseuille flow, f·Re = 96 on the 2h basis (same analytic value as the parabolised march)');
    add('2-D Navier–Stokes, empty channel: fully developed Sherwood number', 8.235, cfE.Sh, 0.12, 'Uniform flux through both walls, Sc ≈ 1 so that the salt profile develops inside the solved length');
    add('2-D channel solution conserves salt', 0, (cfE.saltIn - cfE.saltOut) / cfE.saltIn, 1e-3, 'Inflow = outflow + flux through both membranes');
    add('2-D Navier–Stokes: spacer filaments raise the Sherwood number and the pressure gradient', 1, cfS.Sh > 2 * cfO.Sh && cfS.dpdx > 2 * cfO.dpdx && cfS.converged ? 1 : 0, 0, `Zigzag filaments: Sh ${fmt(cfS.Sh, 4)} against ${fmt(cfO.Sh, 4)} in the open channel; pressure gradient ${fmt(cfS.dpdx, 4)} against ${fmt(cfO.dpdx, 4)} Pa/m`);
    const nsO = channelNS({ U: 0.064, h: 0.75e-3, L: 1.5, D: 1.6e-9, c0: 1, jw: [1e-9, 1e-9] });
    add('2-D and parabolised Navier–Stokes agree on the open-channel pressure gradient', nsO.dpFd / 1.5, cfO.dpdx, 0.03 * (nsO.dpFd / 1.5), 'Developed pressure gradient 12·μ·U/h² (Pa/m)');
    const nsm = channelNS({ U: 0.01, h: 1e-3, L: 2, D: 1e-7, c0: 50, jw: [1e-5, 1e-5] });
    add('Navier–Stokes–Nernst–Planck: developed Sherwood number 8.235', 8.235, nsm.ShEnd, 0.03, 'Uniform ion flux through both walls (specified-flux boundary), parallel plates');
    add('Navier–Stokes–Nernst–Planck: salt balance of the channel', 0, (nsm.saltIn - nsm.saltOut) / nsm.saltIn, 2e-3, 'Inflow = outflow + wall fluxes');
    add('Navier–Stokes: inlet pressure = outlet pressure + pressure drop', 0, channelNS({ U: 0.08, h: 0.75e-3, L: 1, D: 1.6e-9, pOut: 2e4 }).pIn - 2e4 - channelNS({ U: 0.08, h: 0.75e-3, L: 1, D: 1.6e-9 }).dp, 1e-6, 'Outlet-pressure boundary condition (Pa)');
    // ---- thermal coupling, fouling, surrogate, optimisation
    const tv = simulateEDThermal({ ...d, thLoss: 0 }, { nSeg: 4, tol: 1e-4 });
    add('Thermal coupling: adiabatic temperature rise = heat ÷ heat-capacity flow', tv.th.Qgen / tv.th.mcp, tv.th.dT, 1e-9, `ΔT = ${fmt(tv.th.dT, 3)} K for ${fmt(tv.th.Qgen / 1000, 3)} kW released`);
    add('Thermal coupling: a warmer stack conducts better and needs less energy', 1, tv.sec < tv.th.iso.sec && tv.th.kappa > tv.th.iso.kappa ? 1 : 0, 0, `${fmt(tv.sec, 5)} vs ${fmt(tv.th.iso.sec, 5)} kWh/m³ isothermal`);
    const fo = foulingED({ i: 100, u: 0.08, cFoul: 5, kDep: 0.02, kDet: 0.05, rFoul: 2, Sg: 0.8, Sc: 0.5, kScale: 1, rScale: 5, mBlock: 50, edr: false, revInterval: 20, revEff: 0.9, days: 60 });
    add('Fouling kinetics reproduce the analytical deposit growth', (0.02 * 5 / 0.05) * (1 - Math.exp(-3)), fo.mf[fo.mf.length - 1], 1e-6, 'm = (a/k)(1 − e^(−kt)) for constant deposition and first-order detachment (g/m²)');
    add('No scale grows below saturation', 0, fo.ms[fo.ms.length - 1], 0, 'Gypsum ratio 0.8, calcite ratio 0.5');
    add('Supersaturation and polarity reversal: scale reaches a finite steady level', 1, (() => { const q = foulingED({ i: 100, u: 0.08, cFoul: 0, kDep: 0, kDet: 0, rFoul: 0, Sg: 2, Sc: 0.5, kScale: 1, rScale: 5, mBlock: 50, edr: true, revInterval: 20, revEff: 0.5, days: 5 }); return q.ms[q.ms.length - 1] / (1 / q.kRev); })(), 1e-6, 'm_s → k_s(S − 1)²/k_rev');
    const sgt = stackSurrogate(r, 40, 12);
    add('Surrogate reproduces held-out mechanistic runs', 1, sgt.sec.r2, 0.03, `R² of the specific energy on ${sgt.test.length} runs not used for training (removal: ${fmt(sgt.removal.r2, 4)})`);
    add('Surrogate interpolates its training runs', 1, sgt.fitSec.r2, 1e-3, 'Kernel regression with a small ridge term');
    add('Surrogate comparison: all three learners train and the selected one is the best on validation', 1, Object.values(sgt.models).every((m) => m.sec.ok && m.removal.ok) && Object.values(sgt.models).every((m) => m.sec.val >= sgt.models[sgt.best.sec].sec.val) ? 1 : 0, 0, `Held-out R² of the specific energy: ${Object.keys(sgt.models).map((n) => `${SURROGATES[n]} ${fmt(sgt.models[n].sec.r2, 4)}`).join(', ')}; selected: ${SURROGATES[sgt.best.sec]}`);
    add('Neural-network and Gaussian-process surrogates reproduce held-out runs', 1, Math.min(sgt.models.nn.sec.r2, sgt.models.gp.sec.r2, sgt.models.nn.removal.r2, sgt.models.gp.removal.r2), 0.06, 'Lowest held-out R² of the two new learners over both outputs');
    const sgt2 = stackSurrogate(r, 40, 12);
    add('Surrogate training is deterministic for a given seed', sgt.sec.rmse, sgt2.sec.rmse, 1e-12, 'Two trainings with the same seed give identical held-out errors');
    const og = optimiseED(d, [50, 90], [6, 12]), ogN = optimiseED(d, [50, 90], [6, 12], false), lo6 = og.cells.find((q) => q.phi === 50 && q.u === 12), hi6 = og.cells.find((q) => q.phi === 90 && q.u === 12);
    add('Grid stage of the optimiser returns the cheapest feasible grid design', Math.min(...og.cells.map((q) => q.cost)), ogN.best.cost, 0, 'Coarse grid of re-designs over current fraction and velocity ($/m³)');
    add('Nelder–Mead optimum is at least as cheap as the best grid design', 1, og.best.cost <= og.gridBest.cost + 1e-12 && og.evals > og.cells.length ? 1 : 0, 0, `Simplex search over the continuous variables: ${fmt(og.gridBest.cost, 5)} → ${fmt(og.best.cost, 5)} $/m³ in ${og.evals} designs (at ${fmt(og.best.phi, 3)} % and ${fmt(og.best.u, 3)} cm/s)`);
    add('Energy–area trade-off behind the optimum', 1, lo6 && hi6 && hi6.area < lo6.area && hi6.sec > lo6.sec ? 1 : 0, 0, 'A higher allowed current fraction saves a stage of membrane area but costs energy (12 cm/s)');
    return C;
  },
};

// ---- result builders ----------------------------------------------------------------------------------------
function profilePlot(v, cd, cc, i, delta, T) {
  const X = v.Xfix * 1000, dm = v.dMem * 1e-6, e = electrolyte(CH.map((k) => (k === 'Na' || k === 'Cl' ? cd : 0)), T), base = { cd, cc, X, dm, Dp: e.Dc, Dm: e.Da, DpM: 0.05 * e.Dc, DmM: 0.05 * e.Da, deltaD: delta, deltaC: delta, T, n: Math.max(6, Math.round(v.nProf)) };
  let pf = npProfile({ ...base, i });
  if (!pf.ok) pf = npProfile({ ...base, i: 0.5 * i });
  if (!pf.ok) return null;
  const um = pf.x.map((x) => x * 1e6);
  return { pf, plot: { type: 'line', title: 'Nernst–Planck–Donnan profile: diluate film | cation membrane | concentrate film', xlabel: 'Distance from the diluate-side membrane face (µm)', ylabel: 'mol/m³ · mV', logy: false,
    series: [{ name: 'Co-ion (anion) concentration (mol/m³)', x: um, y: pf.c }, { name: 'Counter-ion concentration ÷ 20 (mol/m³)', x: um, y: pf.cplus.map((c, k) => (pf.x[k] >= 0 && pf.x[k] <= dm ? c / 20 : c)), dash: true }, { name: 'Electric potential (mV)', x: um, y: pf.phi.map((p) => p * 1000) }],
    vlines: [{ x: 0, label: 'membrane' }, { x: dm * 1e6, label: '' }], note: `Equivalent 1:1 salt at the stack inlet, ${fmt(pf.Jp * F, 3)} A/m² carried by counter-ions and ${fmt(-pf.Jm * F, 3)} A/m² by leaking co-ions: counter-ion transport number ${fmt(pf.tm, 4)}. Inside the membrane the counter-ion curve is divided by 20.` } };
}

function scaleWarnings(W, sc, scWall, edr) {
  const limG = edr ? 1.75 : 1, limC = edr ? 150 : 3; // polarity reversal tolerates supersaturation (LSI up to about +2.2)
  if (scWall.gypsum > limG) W.push({ level: scWall.gypsum > 2 * limG ? 'bad' : 'warn', msg: `Gypsum saturation ratio reaches ${fmt(scWall.gypsum, 3)} at the concentrate-side membrane wall (bulk ${fmt(sc.gypsum, 3)}) — lower the recovery, dose antiscalant or use polarity reversal.` });
  if (scWall.calcite > limC) W.push({ level: 'warn', msg: `Calcite saturation ratio is about ${fmt(scWall.calcite, 3)} in the concentrate — acidify the concentrate loop or dose antiscalant${edr ? '' : ', or use polarity reversal'}.` });
}

function runED(v) {
  const r = v.thermalModel ? simulateEDThermal(v) : simulateED(v), W = [], p = v, t = r.tr, T = r.T, st = r.stages;
  const segs = st.flatMap((s, k) => s.segs.map((g) => ({ ...g, X: k * r.G.L + g.x, stage: k + 1 })));
  if (!r.reached) W.push({ level: 'bad', msg: `The target of ${p.targetTDS} mg/L is not reached in ${r.nSt} stages at ${p.safety} % of the limiting current (product ${fmt(r.tdsP, 4)} mg/L) — allow more stages, a longer flow path or a higher velocity.` });
  if (r.ratioMax > 1) W.push({ level: 'bad', msg: `The current exceeds the limiting current density (up to ${fmt(100 * r.ratioMax, 3)} %): water splitting, pH shifts and scaling on the membranes will occur. Lower the voltage or raise the velocity.` });
  else if (r.ratioMax > 0.8) W.push({ level: 'warn', msg: `The current reaches ${fmt(100 * r.ratioMax, 3)} % of the limiting current density; normal practice is ≤ 70–80 %.` });
  if (r.hAdd > 1e-9) W.push({ level: 'warn', msg: `Water splitting shifts the diluate pH to about ${fmt(r.pHd, 3)} and the concentrate pH to about ${fmt(r.pHc, 3)}.` });
  scaleWarnings(W, r.sc, r.scWall, p.edr);
  if (r.tdsF > 12000) W.push({ level: 'info', msg: `Feed salinity is ${fmt(r.tdsF / 1000, 3)} g/L: electrodialysis energy rises in proportion to the salt removed, so reverse osmosis is normally cheaper above about 5–10 g/L.` });
  if (r.eff < 0.8) W.push({ level: 'warn', msg: `Current efficiency is only ${fmt(100 * r.eff, 3)} % — back-diffusion and co-ion leakage grow with the concentrate/diluate ratio (${fmt(r.tdsC / r.tdsP, 3)}).` });
  if (st.some((s) => s.dp > 2.5e5)) W.push({ level: 'warn', msg: `Channel pressure drop reaches ${fmt(Math.max(...st.map((s) => s.dp)) / 1e5, 3)} bar in one stage — stacks are usually limited to about 3 bar.` });
  const cb0 = chargeBalance(scaleIons(cloneIons(p.ions), 1)).errorPct;
  if (Math.abs(cb0) > 2) W.push({ level: Math.abs(cb0) > 5 ? 'warn' : 'info', msg: `The feed analysis has a charge imbalance of ${fmt(cb0, 2)} %; it was balanced with ${cb0 > 0 ? 'chloride' : 'sodium'} before the calculation.` });
  if (p.mode === 'design') W.push({ level: 'info', msg: `Design: ${r.nSt} stage${r.nSt > 1 ? 's' : ''} × ${r.Ncp} cell pairs at ${fmt(100 * r.phi, 3)} % of the limiting current (${st.map((s) => fmt(s.U, 3)).join(' / ')} V per cell pair).` });
  if (p.edr) W.push({ level: 'info', msg: `Polarity reversal every ${p.revInterval} min diverts ${fmt(100 * r.edrLoss, 3)} % of the product as off-spec water.` });

  // sweeps: energy versus feed salinity (same design rules) and a voltage–velocity map
  const fast = { nSeg: Math.min(r.nSeg, 4), tol: 1e-4 }, sal = [0.4, 0.7, 1, 1.6, 2.6].map((f) => f * (p.salinityFactor ?? 1));
  const sw = sal.map((f) => { try { const q = simulateED({ ...p, mode: 'design' }, { ...fast, salinityFactor: f, targetTDS: Math.min(p.targetTDS, 0.6 * r.tdsF * (f / (p.salinityFactor ?? 1))) }); return q.reached ? [q.tdsF / 1000, q.sec, q.wMin, q.nSt] : null; } catch { return null; } }).filter(Boolean);
  const Us = linspace(0.2, 1.4, 6), us = [3, 6, 9, 12, 16], cfIn = r.cf, ccIn = t.ccIn;
  const map = us.map((uu) => Us.map((U) => { const q = (uu / 100) * r.G.W * r.G.h * r.G.eps; let s = { qd: q, qc: q, cd: cfIn, cc: ccIn }, m; for (let k = 0; k < r.nSt; k++) { m = marchStage(s, U, r.G, r.par, T, 4); s = m.st; } return tdsOf(s.cd, r.neutral); }));
  const u0 = st[0].segs[0].u, polIn = polarisation(st[0].in.cd, st[0].in.cc, u0, r.G, r.par, T), last = st[r.nSt - 1], polOut = polarisation(last.st.cd, last.st.cc, last.segs[r.nSeg - 1].u, r.G, r.par, T);
  const ePro = electrolyte(cfIn, T), prof = profilePlot(p, ePro.ceq, electrolyte(ccIn, T).ceq, st[0].segs[0].iFilm, st[0].segs[0].k > 0 ? ePro.Ds / st[0].segs[0].k : 5e-5, T);
  const mean = (f) => sum(segs.map(f)) / segs.length, vb = { 'Donnan (membrane) potential': mean((g) => g.pr.donnan), 'Concentration polarisation': mean((g) => g.pr.polar), 'Membranes (ohmic)': mean((g) => g.pr.ohmMem), 'Diluate channel (ohmic)': mean((g) => g.pr.ohmD), 'Concentrate channel (ohmic)': mean((g) => g.pr.ohmC), 'Electrodes ÷ cell pairs': (sum(st.map((s) => s.Uel)) / r.nSt / r.Ncp) * r.nPar };
  const rem = (j) => (r.cf[j] > 0 ? 100 * (1 - (t.Qp * t.cp[j]) / (t.Qd * r.cf[j])) : null), act = CH.map((k, j) => j).filter((j) => r.cf[j] > 1e-6);
  const xs = segs.map((g) => g.X), out = { streams: { diluate: stream(r.Qprod * 3600, T, r.pHd, r.ionsP), concentrate: stream(r.Qconc * 3600, T, r.pHc, r.ionsC) }, sec: r.sec, power: (r.Pel + r.Ppump) / 1000, area: r.area, cellPairs: r.Ncp * r.nSt, currentDensity: r.iAvg,
    stages: r.nSt, recovery: r.waterRec, currentEfficiency: r.eff, limitingRatio: r.ratioMax, voltagePerCellPair: st.map((s) => s.U), stackVoltage: st.map((s) => s.Ustack), saltRemoval: 1 - r.tdsP / r.tdsF, process: 'ed' };
  const stageEdges = st.slice(1).map((_, k) => ({ x: (k + 1) * r.G.L, label: `stage ${k + 2}` })), adv = advancedED(r, p, W);
  Object.assign(out, adv.out);
  return {
    summary: `${fmt(r.Qprod * 3600, 4)} m³/h of product at ${fmt(r.tdsP, 4)} mg/L from ${fmt(r.tdsF, 4)} mg/L feed in ${r.nSt} stage${r.nSt > 1 ? 's' : ''} of ${r.Ncp} cell pairs (${fmt(r.area, 4)} m² of membrane), using ${fmt(r.sec, 3)} kWh/m³ at ${fmt(100 * r.eff, 3)} % current efficiency and ${fmt(100 * r.waterRec, 3)} % recovery.`,
    warnings: W,
    kpis: [
      { label: 'Product flow', value: r.Qprod * 3600, unit: 'm³/h' }, { label: 'Product TDS', value: r.tdsP, unit: 'mg/L', status: p.mode === 'design' && !r.reached ? 'bad' : 'ok' },
      { label: 'Salt removal', value: 100 * (1 - r.tdsP / r.tdsF), unit: '%' }, { label: 'Water recovery', value: 100 * r.waterRec, unit: '%' },
      { label: 'Specific energy', value: r.sec, unit: 'kWh/m³', help: 'Rectifier input plus pumping per m³ of product' }, { label: 'Thermodynamic minimum', value: r.wMin, unit: 'kWh/m³', help: 'Reversible work of this separation (ideal solution)' },
      { label: 'Current efficiency', value: 100 * r.eff, unit: '%', status: r.eff < 0.8 ? 'warn' : 'ok' }, { label: 'Mean current density', value: r.iAvg, unit: 'A/m²' },
      { label: 'Highest i / i_lim', value: 100 * r.ratioMax, unit: '%', status: r.ratioMax > 1 ? 'bad' : r.ratioMax > 0.8 ? 'warn' : 'ok' }, { label: 'Stages × cell pairs', value: `${r.nSt} × ${r.Ncp}`, help: `${r.nPar} parallel stack${r.nPar > 1 ? 's' : ''} per stage` },
      { label: 'Membrane area', value: r.area, unit: 'm²', help: 'Cation + anion membranes of all stages' }, { label: 'Stack voltage (stage 1)', value: st[0].Ustack, unit: 'V' },
      { label: 'Total DC power', value: r.Pdc / 1000, unit: 'kW' }, { label: 'Pumping power', value: r.Ppump / 1000, unit: 'kW' },
      { label: 'Concentrate TDS', value: tds(r.ionsC), unit: 'mg/L' }, { label: 'Gypsum saturation (wall)', value: r.scWall.gypsum, unit: '–', status: r.scWall.gypsum > (p.edr ? 1.75 : 1) ? 'warn' : 'ok', help: 'Above 1 the concentrate is supersaturated with CaSO₄·2H₂O; EDR tolerates about 1.75' },
      { label: 'Pressure drop', value: r.dp / 1e5, unit: 'bar' }, { label: 'Diluate pH (estimated)', value: r.pHd, unit: '', status: r.hAdd > 1e-9 ? 'warn' : 'ok' },
      ...adv.kpis,
    ],
    recommendations: [
      r.ratioMax > 0.8 ? 'Reduce the voltage of the last stage or raise the linear velocity: the limiting current is proportional to the diluate concentration and falls stage by stage.' : null,
      !r.reached ? 'Add stages or lengthen the flow path; alternatively accept a higher product TDS and polish with a small RO or ion-exchange step.' : null,
      r.scWall.gypsum > 1 && !p.edr ? 'Switch on polarity reversal (EDR) or lower the recovery to keep calcium sulphate below saturation in the concentrate.' : null,
      vb['Diluate channel (ohmic)'] > 0.45 * sum(Object.values(vb)) ? 'Most of the voltage is lost in the dilute channels: thinner spacers or conductive (ion-exchange resin filled) spacers lower the energy.' : null,
      r.eff < 0.85 ? 'Improve current efficiency with higher-permselectivity membranes or a lower concentrate salinity (lower recovery).' : null,
      ...adv.recs,
      'Send the concentrate to suite 2 (Brine chemistry) for a full scaling check, and compare the cost of water with RO in suite 13 (Economics).',
    ].filter(Boolean),
    plots: [
      { type: 'line', title: 'Salinity along the flow path', xlabel: 'Path length through all stages (m)', ylabel: 'TDS (mg/L)', logy: true, series: [{ name: 'Diluate', x: xs, y: segs.map((g) => tdsOf(g.cd, r.neutral)) }, { name: 'Concentrate', x: xs, y: segs.map((g) => tdsOf(g.cc, r.neutral)) }, { name: 'Diluate at the membrane wall', x: xs, y: segs.map((g) => (tdsOf(g.cd, r.neutral) * g.wd) / g.ceqD), dash: true }], hlines: p.mode === 'design' ? [{ y: p.targetTDS, label: 'target' }] : [], vlines: stageEdges },
      { type: 'line', title: 'Current density and limiting current density', xlabel: 'Path length through all stages (m)', ylabel: 'A/m²', series: [{ name: 'Operating current density', x: xs, y: segs.map((g) => g.i) }, { name: 'Limiting current density', x: xs, y: segs.map((g) => g.ilim), dash: true }, { name: 'i / i_lim (%)', x: xs, y: segs.map((g) => 100 * g.ratio) }], hlines: [{ y: 80, label: '80 % guideline' }], vlines: stageEdges },
      { type: 'bar', title: 'Voltage breakdown per cell pair (path average)', ylabel: 'V', categories: Object.keys(vb), series: [{ name: 'Voltage', values: Object.values(vb) }] },
      { type: 'line', title: 'Polarisation curve of one cell pair', xlabel: 'Voltage per cell pair (V)', ylabel: 'Current density (A/m²)', series: [{ name: 'Stack inlet', x: polIn.U, y: polIn.i }, { name: 'Stack outlet', x: polOut.U, y: polOut.i }], vlines: [{ x: polIn.Ulim, label: 'limiting plateau' }, { x: polIn.Uover, label: 'over-limiting' }, { x: st[0].U, label: 'operating' }], hlines: [{ y: polIn.cp.ilim, label: 'i_lim inlet' }], note: r.par.ol ? 'Ohmic region at low voltage, limiting plateau where the wall concentration approaches zero, then the over-limiting branch with the threshold and slope of the electroconvection model.' : 'Ohmic region at low voltage, limiting plateau where the wall concentration approaches zero, then the over-limiting branch (water splitting and electro-convection; empirical slope).' },
      ...(prof ? [prof.plot] : []),
      { type: 'bar', title: 'Ion removal from the diluate', ylabel: '% removed', categories: act.map((j) => IONS[CH[j]].label), series: [{ name: 'Removal', values: act.map((j) => rem(j)) }] },
      ...(sw.length > 1 ? [{ type: 'line', title: 'Desalination energy versus feed salinity', xlabel: 'Feed TDS (g/L)', ylabel: 'kWh/m³', logx: true, series: [{ name: 'Specific energy (design re-solved)', x: sw.map((q) => q[0]), y: sw.map((q) => q[1]), mode: 'both' }, { name: 'Reversible minimum', x: sw.map((q) => q[0]), y: sw.map((q) => q[2]), mode: 'both', dash: true }, { name: 'Stages required', x: sw.map((q) => q[0]), y: sw.map((q) => q[3]), mode: 'points' }], note: 'Same product target and design rules; ED energy is nearly proportional to the salt removed.' }] : []),
      { type: 'field', title: 'Product TDS versus voltage and velocity (this stack)', xlabel: 'Voltage per cell pair (V)', ylabel: 'Linear velocity (cm/s)', zlabel: 'Product TDS', zunit: 'mg/L', x: Us, y: us, z: map, cmap: 'salinity', contours: 8, markers: [{ x: clamp(sum(st.map((s) => s.U)) / r.nSt, 0.2, 1.4), y: clamp(u0 * 100, 3, 16), label: 'operating' }], note: `${r.nSt} stage${r.nSt > 1 ? 's' : ''} with the same voltage on every stage; coarse grid (4 segments per stage).` },
      { type: 'line', title: 'Concentration-polarisation and water transport', xlabel: 'Path length through all stages (m)', ylabel: 'see legend', series: [{ name: 'Wall ÷ bulk concentration, diluate', x: xs, y: segs.map((g) => g.wd / g.ceqD) }, { name: 'Wall ÷ bulk concentration, concentrate', x: xs, y: segs.map((g) => g.wc / g.ceqC) }, { name: 'Water flux to concentrate (L/m²·h)', x: xs, y: segs.map((g) => g.jw * 3.6e6) }], vlines: stageEdges },
      ...adv.plots,
    ],
    tables: [
      { title: 'Stage summary', columns: ['Stage', 'Diluate in (mg/L)', 'Diluate out (mg/L)', 'Concentrate out (mg/L)', 'V per cell pair', 'Electrode voltage (V)', 'Stack voltage (V)', 'Stack current (A)', 'Mean i (A/m²)', 'Max i / i_lim (%)', 'DC power (kW)', 'Pressure drop (bar)', 'Velocity (cm/s)', 'Reynolds'],
        rows: st.map((s) => [s.n, s.tdsIn, s.tdsOut, s.tdsC, s.U, s.Uel, s.Ustack, s.Istack, s.iAvg, 100 * s.ratioMax, s.P / 1000, s.dp / 1e5, s.segs[0].u * 100, s.segs[0].Re]) },
      { title: 'Ion-by-ion results', columns: ['Ion', 'Feed (mg/L)', 'Product (mg/L)', 'Concentrate (mg/L)', 'Removal (%)', 'Mean flux (mmol/m²·h)', 'Share of current (%)'],
        rows: [...act.map((j) => [`${IONS[CH[j]].name} ${IONS[CH[j]].label}`, r.ionsF[CH[j]], r.ionsP[CH[j]], r.ionsC[CH[j]], rem(j), (t.tr[j] * 3.6e6) / (r.nSt * r.G.W * r.G.L), (100 * F * AZ[j] * t.tr[j]) / Math.max(sum(st.map((s) => s.I)), 1e-30)]),
          ['TDS', r.tdsF, r.tdsP, tds(r.ionsC), 100 * (1 - r.tdsP / r.tdsF), null, null], ['Flow (m³/h)', t.Qf * 3600, r.Qprod * 3600, r.Qconc * 3600, null, null, null], ['pH (estimated)', p.pH, r.pHd, r.pHc, null, null, null]],
        note: 'Share of current = Faradaic equivalents of each ion ÷ cell current; co-ion leakage and back-diffusion make the cation and anion sums smaller than 100 %. Uncharged silica and boron are not transported.' },
      { title: 'Profile along the flow path', columns: ['Stage', 'Position (m)', 'Diluate (eq/m³)', 'Concentrate (eq/m³)', 'κ diluate (S/m)', 'i (A/m²)', 'i_lim (A/m²)', 'i / i_lim (%)', 'Wall conc. diluate (eq/m³)', 'Mass-transfer coeff. (µm/s)', 'Donnan (V)', 'Polarisation (V)', 'Ohmic (V)'],
        rows: segs.filter((_, k) => segs.length <= 48 || k % Math.ceil(segs.length / 48) === 0).map((g) => [g.stage, g.X, g.ceqD, g.ceqC, g.kD, g.i, g.ilim, 100 * g.ratio, g.wd, g.k * 1e6, g.pr.donnan, g.pr.polar, g.pr.ohmMem + g.pr.ohmD + g.pr.ohmC]) },
      { title: 'Energy, efficiency and scaling', columns: ['Quantity', 'Value', 'Unit'],
        rows: [['DC energy', r.secDC, 'kWh/m³'], ['Rectifier loss', r.sec - r.secDC - r.secPump, 'kWh/m³'], ['Pumping energy', r.secPump, 'kWh/m³'], ['Total specific energy', r.sec, 'kWh/m³'], ['Energy per kg of salt removed', (r.sec * 1000) / Math.max(r.tdsF - r.tdsP, 1e-9), 'kWh/kg'],
          ['Current efficiency of the membranes', 100 * r.effMem, '%'], ['Shunt-current loss', 100 * r.par.shunt, '%'], ['Overall current efficiency', 100 * r.eff, '%'], ['Water transferred to the concentrate', (100 * (t.Qd - t.Qp)) / t.Qd, '% of diluate'],
          ['Gypsum saturation ratio, bulk concentrate', r.sc.gypsum, '–'], ['Gypsum saturation ratio, membrane wall', r.scWall.gypsum, '–'], ['Calcite saturation ratio, bulk concentrate', r.sc.calcite, '–'], ['Concentrate ionic strength', r.sc.I, 'mol/L'], ['Concentrate recycle flow', t.Qrec * 3600, 'm³/h'], ['Concentrate blow-down', t.Qbd * 3600, 'm³/h']] },
      ...adv.tables,
    ],
    balances: [
      { name: 'Water (m³/h)', in: t.Qf * 3600, out: (t.Qp + t.Qbd) * 3600 },
      { name: 'Total salt (kg/h)', in: t.Qf * tdsOf(r.cf) * 3.6, out: (t.Qp * tdsOf(t.cp) + t.Qbd * tdsOf(t.ccOut)) * 3.6 },
      { name: 'Sodium (mol/s)', in: t.Qf * r.cf[CH.indexOf('Na')], out: t.Qp * t.cp[CH.indexOf('Na')] + t.Qbd * t.ccOut[CH.indexOf('Na')] },
      { name: 'Chloride (mol/s)', in: t.Qf * r.cf[CH.indexOf('Cl')], out: t.Qp * t.cp[CH.indexOf('Cl')] + t.Qbd * t.ccOut[CH.indexOf('Cl')] },
      { name: 'Charge: cation vs anion equivalents removed (eq/s)', in: sum(t.tr.map((x, j) => (Z[j] > 0 ? x * AZ[j] : 0))), out: sum(t.tr.map((x, j) => (Z[j] < 0 ? x * AZ[j] : 0))) },
      ...adv.bal,
    ],
    outputs: out,
  };
}

function runBatch(v) {
  const r = simulateBatch(v), W = [], p = v, T = r.T;
  if (r.tHit === null) W.push({ level: 'bad', msg: `The batch does not reach ${p.targetTDS} mg/L within ${p.tBatch} min (final ${fmt(r.tdsP, 4)} mg/L) — add cell pairs, raise the voltage or allow more time.` });
  if (r.ratioMax > 1) W.push({ level: 'bad', msg: `The current exceeds the limiting current density during the batch (up to ${fmt(100 * r.ratioMax, 3)} %) — step the voltage down as the diluate is depleted.` });
  else if (r.ratioMax > 0.8) W.push({ level: 'warn', msg: `The current reaches ${fmt(100 * r.ratioMax, 3)} % of the limiting current density near the end of the batch.` });
  const sc = scaling(r.ionsC, p.pH, T), el = electrolyte(r.sE.cc, T), wall = r.m0.segs.length ? Math.max(...r.m0.segs.map((g) => g.wc / g.ceqC)) : 1, scWall = scaling(scaleIons(r.ionsC, wall), p.pH, T);
  scaleWarnings(W, sc, scWall, p.edr);
  if (!W.length) W.push({ level: 'info', msg: 'Batch completes within the limiting-current and scaling guidelines.' });
  const k = r.tHit === null ? r.nt : r.t.findIndex((x) => x >= r.tHit), pol = polarisation(r.cf, r.cf, (p.uLin / 100), r.G, r.par, T), polE = polarisation(r.sE.cd, r.sE.cc, p.uLin / 100, r.G, r.par, T);
  const e0 = electrolyte(r.cf, T), prof = profilePlot(p, e0.ceq, e0.ceq * 1.0001, r.m0.segs[0].iFilm, e0.Ds / r.m0.segs[0].k, T);
  const act = CH.map((_, j) => j).filter((j) => r.cf[j] > 1e-6), rem = (j) => 100 * (1 - (r.sE.cd[j] * r.sE.Vd) / (r.cf[j] * r.Vd0));
  const balSalt = { in: tdsOf(r.cf) * (r.Vd0 + r.Vc0) / 1000, out: (tdsOf(r.sE.cd) * r.sE.Vd + tdsOf(r.sE.cc) * r.sE.Vc) / 1000 };
  return {
    summary: `${fmt(r.sE.Vd, 4)} m³ of product at ${fmt(r.tdsP, 4)} mg/L after ${fmt(r.tUse, 3)} min in a ${r.Ncp}-cell-pair stack at ${p.Ucp} V per cell pair, using ${fmt(r.sec, 3)} kWh/m³ at ${fmt(100 * r.eff, 3)} % current efficiency.`,
    warnings: W,
    kpis: [
      { label: 'Batch time', value: r.tUse, unit: 'min', status: r.tHit === null ? 'bad' : 'ok' }, { label: 'Product TDS', value: r.tdsP, unit: 'mg/L', status: r.tHit === null ? 'bad' : 'ok' },
      { label: 'Product volume', value: r.sE.Vd, unit: 'm³' }, { label: 'Average production', value: r.Qprod, unit: 'm³/h' }, { label: 'Water recovery', value: 100 * r.waterRec, unit: '%' },
      { label: 'Specific energy', value: r.sec, unit: 'kWh/m³' }, { label: 'Current efficiency', value: 100 * r.eff, unit: '%' }, { label: 'Mean current density', value: r.iMean, unit: 'A/m²' },
      { label: 'Highest i / i_lim', value: 100 * r.ratioMax, unit: '%', status: r.ratioMax > 1 ? 'bad' : r.ratioMax > 0.8 ? 'warn' : 'ok' }, { label: 'Mean DC power', value: r.power / 1000, unit: 'kW' },
      { label: 'Membrane area', value: r.area, unit: 'm²' }, { label: 'Concentrate TDS (end)', value: r.tdsCend, unit: 'mg/L' }, { label: 'Gypsum saturation (wall)', value: scWall.gypsum, unit: '–', status: scWall.gypsum > 1 ? 'warn' : 'ok' },
    ],
    recommendations: [
      r.ratioMax > 0.8 ? 'Use a stepped or tapering voltage: the limiting current falls with the diluate concentration during the batch.' : null,
      r.tHit === null ? 'Increase the number of cell pairs or the voltage, or relax the target.' : null,
      'For continuous duty switch the specification to design mode to size a multi-stage stack for the same water.',
    ].filter(Boolean),
    plots: [
      { type: 'line', title: 'Tank salinity during the batch', xlabel: 'Time (min)', ylabel: 'TDS (mg/L)', logy: true, series: [{ name: 'Diluate tank', x: r.t, y: r.tdsD }, { name: 'Concentrate tank', x: r.t, y: r.tdsC }], hlines: [{ y: p.targetTDS, label: 'target' }], vlines: r.tHit !== null ? [{ x: r.tHit, label: 'target reached' }] : [] },
      { type: 'line', title: 'Current density and limiting-current ratio', xlabel: 'Time (min)', ylabel: 'A/m² · %', series: [{ name: 'Mean current density (A/m²)', x: r.t, y: r.iAvg }, { name: 'Highest i / i_lim (%)', x: r.t, y: r.ratio.map((x) => 100 * x) }], hlines: [{ y: 80, label: '80 % guideline' }] },
      { type: 'line', title: 'Electrical power', xlabel: 'Time (min)', ylabel: 'kW', series: [{ name: 'Rectifier input', x: r.t, y: r.P.map((x) => x / 1000) }] },
      { type: 'line', title: 'Polarisation curve of one cell pair', xlabel: 'Voltage per cell pair (V)', ylabel: 'Current density (A/m²)', series: [{ name: 'Start of batch', x: pol.U, y: pol.i }, { name: 'End of batch', x: polE.U, y: polE.i }], vlines: [{ x: p.Ucp, label: 'operating' }] },
      ...(prof ? [prof.plot] : []),
      { type: 'bar', title: 'Ion removal from the diluate', ylabel: '% removed', categories: act.map((j) => IONS[CH[j]].label), series: [{ name: 'Removal', values: act.map(rem) }] },
    ],
    tables: [
      { title: 'Batch history', columns: ['Time (min)', 'Diluate TDS (mg/L)', 'Concentrate TDS (mg/L)', 'Mean i (A/m²)', 'Max i / i_lim (%)', 'Power (kW)'], rows: r.t.map((x, i) => [x, r.tdsD[i], r.tdsC[i], r.iAvg[i], 100 * r.ratio[i], r.P[i] / 1000]).filter((_, i) => i <= k && (r.nt <= 40 || i % Math.ceil(r.nt / 40) === 0 || i === k)) },
      { title: 'Ion-by-ion results', columns: ['Ion', 'Feed (mg/L)', 'Product (mg/L)', 'Concentrate (mg/L)', 'Removal (%)'], rows: [...act.map((j) => [`${IONS[CH[j]].name} ${IONS[CH[j]].label}`, r.ionsF[CH[j]], r.ionsP[CH[j]], r.ionsC[CH[j]], rem(j)]), ['TDS', r.tdsF, r.tdsP, r.tdsCend, 100 * (1 - r.tdsP / r.tdsF)]] },
      { title: 'Energy and scaling', columns: ['Quantity', 'Value', 'Unit'], rows: [['DC + rectifier energy', r.secDC, 'kWh/m³'], ['Pumping energy', r.sec - r.secDC, 'kWh/m³'], ['Gypsum saturation ratio, bulk concentrate', sc.gypsum, '–'], ['Calcite saturation ratio, bulk concentrate', sc.calcite, '–'], ['Concentrate conductivity (end)', el.kappa, 'S/m'], ['Final concentrate volume', r.sE.Vc, 'm³']] },
    ],
    balances: [{ name: 'Water (m³)', in: r.Vd0 + r.Vc0, out: r.sE.Vd + r.sE.Vc }, { name: 'Total salt (kg)', in: balSalt.in, out: balSalt.out }],
    outputs: { streams: { diluate: stream(r.Qprod, T, p.pH, r.ionsP), concentrate: stream((r.sE.Vc / r.tUse) * 60, T, p.pH, r.ionsC) }, sec: r.sec, power: r.power / 1000, area: r.area, cellPairs: r.Ncp, currentDensity: r.iMean, recovery: r.waterRec, currentEfficiency: r.eff, limitingRatio: r.ratioMax, batchTime: r.tUse, process: 'ed-batch' },
  };
}

function runBPMED(v) {
  const r = simulateBPMED(v), W = [], u = r.u, T = r.T;
  if (v.iB > 1200) W.push({ level: 'warn', msg: 'Above about 1000–1200 A/m² bipolar membranes approach their own limiting current (salt depletion in the junction).' });
  if (u.eta < 0.7) W.push({ level: 'warn', msg: `Current efficiency is ${fmt(100 * u.eta, 3)} % at ${v.cProd} mol/L — product strengths above 1–2 mol/L leak protons and hydroxide through the monopolar membranes.` });
  if (r.cOut < 100) W.push({ level: 'warn', msg: `The salt channel is depleted to ${fmt(r.cOut, 3)} eq/m³; its resistance rises steeply — lower the conversion.` });
  const hard = (r.ionsF.Ca || 0) + (r.ionsF.Mg || 0);
  if (hard > 5) W.push({ level: 'bad', msg: `The feed holds ${fmt(hard, 3)} mg/L of calcium + magnesium: they precipitate as hydroxides in the base compartment. Soften to below about 1–5 mg/L (nanofiltration or chelating resin) before bipolar-membrane ED.` });
  if (!W.length) W.push({ level: 'info', msg: 'Operating point lies inside the usual bipolar-membrane window.' });
  const is = linspace(0, 1500, 31), cs = linspace(0.2, 3, 15), at = (i, c) => { const q = bpmUnit(i, c, c, Math.max(r.cOut, 50), v); return (q.U * F) / (q.eta * 0.039997) / 3.6e6; };
  const fi = [200, 500, 800, 1100, 1400], fc = [0.25, 0.5, 1, 1.5, 2, 3];
  const saltIons = scaleIons(r.ionsF, 1 - r.conv), parts = { 'Water splitting (thermodynamic)': u.Ews, 'Junction overpotential': v.bpOver, 'Membranes (ohmic)': u.ohmMem, 'Solutions (ohmic)': u.ohmSol };
  return {
    summary: `${fmt(r.naoh * 3600, 4)} kg/h of NaOH and ${fmt(r.hcl * 3600, 4)} kg/h of HCl at ${v.cProd} mol/L from ${fmt(v.Qp, 4)} m³/h of salt solution, using ${fmt(r.ePerKg, 3)} kWh per kg NaOH on ${fmt(r.area, 4)} m² of bipolar membrane.`,
    warnings: W,
    kpis: [
      { label: 'NaOH production', value: r.naoh * 3600, unit: 'kg/h' }, { label: 'HCl production', value: r.hcl * 3600, unit: 'kg/h' }, { label: 'Base / acid flow (each)', value: r.Qprod * 3600, unit: 'm³/h' },
      { label: 'Energy per kg NaOH', value: r.ePerKg, unit: 'kWh/kg', status: r.ePerKg > 2.8 ? 'warn' : 'ok' }, { label: 'Thermodynamic minimum', value: r.eTheo, unit: 'kWh/kg' }, { label: 'Energy per m³ of feed', value: r.ePerM3, unit: 'kWh/m³' },
      { label: 'Unit voltage', value: u.U, unit: 'V' }, { label: 'Water-splitting voltage', value: u.Ews, unit: 'V' }, { label: 'Current efficiency', value: 100 * u.eta, unit: '%', status: u.eta < 0.7 ? 'warn' : 'ok' },
      { label: 'Bipolar-membrane area', value: r.area, unit: 'm²' }, { label: 'Repeating units / stacks', value: `${r.units} / ${r.stacks}` }, { label: 'Total current', value: r.I, unit: 'A' },
      { label: 'Power', value: (r.P + r.Ppump) / 1000, unit: 'kW' }, { label: 'Salt conversion', value: 100 * r.conv, unit: '%' }, { label: 'Depleted salt concentration', value: r.cOut, unit: 'eq/m³', status: r.cOut < 100 ? 'warn' : 'ok' },
    ],
    recommendations: [hard > 5 ? 'Soften the feed first: suite 2 gives the hardness of an RO concentrate and suite 9 the pretreatment for brine valorisation.' : null, u.eta < 0.75 ? 'Produce a weaker acid/base (≤ 1 mol/L) and concentrate downstream if needed.' : null, 'The acid and caustic can replace purchased chemicals for RO cleaning and pH control — credit them in suite 13 (Economics).'].filter(Boolean),
    plots: [
      { type: 'line', title: 'Voltage of one repeating unit versus current density', xlabel: 'Current density (A/m²)', ylabel: 'V', series: [{ name: 'Unit voltage', x: is, y: is.map((i) => bpmUnit(i, v.cProd, v.cProd, Math.max(r.cOut, 50), v).U) }], hlines: [{ y: u.Ews, label: 'water splitting' }], vlines: [{ x: v.iB, label: 'operating' }] },
      { type: 'bar', title: 'Voltage breakdown of one repeating unit', ylabel: 'V', categories: Object.keys(parts), series: [{ name: 'Voltage', values: Object.values(parts) }] },
      { type: 'line', title: 'Energy and current efficiency versus product strength', xlabel: 'Acid / base concentration (mol/L)', ylabel: 'kWh/kg · %/100', series: [{ name: 'Energy per kg NaOH (kWh/kg)', x: cs, y: cs.map((c) => at(v.iB, c)) }, { name: 'Current efficiency (–)', x: cs, y: cs.map((c) => bpmUnit(v.iB, c, c, r.cOut, v).eta) }], vlines: [{ x: v.cProd, label: 'operating' }] },
      { type: 'line', title: 'Energy and membrane area versus current density', xlabel: 'Current density (A/m²)', ylabel: 'kWh/kg · 100 m²', series: [{ name: 'Energy per kg NaOH (kWh/kg)', x: is.slice(2), y: is.slice(2).map((i) => at(i, v.cProd)) }, { name: 'Membrane area (100 m²)', x: is.slice(2), y: is.slice(2).map((i) => r.I / i / 100) }], vlines: [{ x: v.iB, label: 'operating' }] },
      { type: 'field', title: 'Energy per kg NaOH versus current density and product strength', xlabel: 'Current density (A/m²)', ylabel: 'Product concentration (mol/L)', zlabel: 'Energy', zunit: 'kWh/kg', x: fi, y: fc, z: fc.map((c) => fi.map((i) => at(i, c))), cmap: 'thermal', contours: 8, markers: [{ x: v.iB, y: v.cProd, label: 'operating' }] },
    ],
    tables: [
      { title: 'Products and streams', columns: ['Stream', 'Flow (m³/h)', 'Concentration', 'Unit'], rows: [['Salt feed', v.Qp, r.el.ceq, 'eq/m³'], ['Depleted salt', v.Qp, r.cOut, 'eq/m³'], ['Caustic product (as NaOH)', r.Qprod * 3600, v.cProd, 'mol/L'], ['Acid product (as HCl)', r.Qprod * 3600, v.cProd, 'mol/L']], note: 'All cations are reported as sodium and all anions as chloride equivalents.' },
      { title: 'Electrical performance', columns: ['Quantity', 'Value', 'Unit'], rows: [['Unit voltage', u.U, 'V'], ['Electrode voltage per stack', r.Uel, 'V'], ['Current efficiency', 100 * u.eta, '%'], ['Total current', r.I, 'A'], ['DC power incl. rectifier', r.P / 1000, 'kW'], ['Pumping power', r.Ppump / 1000, 'kW'], ['Energy per kg NaOH', r.ePerKg, 'kWh/kg'], ['Energy per tonne of salt converted', ((r.P + r.Ppump) / 1000 / (r.nEq * 0.05844 * 3.6)) , 'kWh/t']] },
    ],
    balances: [{ name: 'Salt equivalents (eq/s)', in: (v.Qp / 3600) * r.el.ceq, out: (v.Qp / 3600) * r.cOut + r.nEq }, { name: 'Charge: Faradaic vs produced (eq/s)', in: (r.I * u.eta) / F, out: r.nEq }],
    outputs: { streams: { diluate: stream(v.Qp, T, v.pH, saltIons), concentrate: stream(r.Qprod * 3600, T, 14 + Math.log10(v.cProd), cloneIons({ Na: v.cProd * 22990 })) }, sec: r.ePerM3, power: (r.P + r.Ppump) / 1000, area: 3 * r.area, cellPairs: r.units, currentDensity: v.iB, naohKgPerH: r.naoh * 3600, hclKgPerH: r.hcl * 3600, energyPerKgNaOH: r.ePerKg, process: 'bpmed' },
  };
}

function runCDI(v) {
  const r = simulateCDI(v), W = [], T = r.T, name = v.cdiMem ? 'MCDI' : 'CDI';
  if (v.Vch > 1.23) W.push({ level: 'warn', msg: 'Charging above 1.23 V risks water electrolysis and carbon oxidation at the anode.' });
  if (r.c0 > 100) W.push({ level: 'warn', msg: `Feed is ${fmt(r.c0, 3)} eq/m³ (${fmt(tds(r.ionsF) / 1000, 3)} g/L): capacitive deionisation is suited to brackish water below about 3–5 g/L.` });
  if (r.removal < 0.15) W.push({ level: 'info', msg: `Only ${fmt(100 * r.removal, 3)} % of the salt is removed per pass — lower the flow per cell area or use several passes.` });
  if (!v.cdiMem && r.eq.effCDI < 0.6) W.push({ level: 'info', msg: `Charge efficiency without membranes is ${fmt(100 * r.eq.effCDI, 3)} % (co-ion expulsion); membranes would raise it to about ${fmt(100 * v.alphaC, 3)} %.` });
  if (!W.length) W.push({ level: 'info', msg: 'Operating point lies inside the usual capacitive-deionisation window.' });
  const Vs = linspace(0.2, 1.4, 13), md = Vs.map((V) => cdiEquilibrium(r.c0, v, V, 0)), g = Vs.map((V) => gcs(V / 2, r.c0, v)), cs = logspace(Math.log10(2), Math.log10(200), 9), fV = [0.6, 0.8, 1.0, 1.2, 1.4];
  const ionsP = scaleIons(r.ionsF, r.cAvg / r.c0), ionsC = scaleIons(r.ionsF, 1 + (1 - r.cAvg / r.c0) * (v.cdiTads / v.cdiTdes));
  return {
    summary: `${name} removes ${fmt(100 * r.removal, 3)} % of the salt (${fmt(tds(r.ionsF), 4)} → ${fmt(tds(ionsP), 4)} mg/L) at ${fmt(r.sec, 3)} kWh/m³, with a salt adsorption of ${fmt(r.sacDyn, 3)} mg/g per cycle (equilibrium ${fmt(r.eq.sac, 3)} mg/g) and ${fmt(100 * r.effDyn, 3)} % charge efficiency.`,
    warnings: W,
    kpis: [
      { label: 'Product TDS (cycle average)', value: tds(ionsP), unit: 'mg/L' }, { label: 'Salt removal', value: 100 * r.removal, unit: '%' }, { label: 'Salt adsorption per cycle', value: r.sacDyn, unit: 'mg/g' },
      { label: 'Equilibrium salt adsorption capacity', value: r.eq.sac, unit: 'mg/g', help: 'Modified-Donnan model between the charging and discharge voltages' }, { label: 'Equilibrium charge', value: r.eq.charge, unit: 'C/g' }, { label: 'Charge efficiency', value: 100 * r.effDyn, unit: '%' },
      { label: 'Specific energy', value: r.sec, unit: 'kWh/m³' }, { label: 'Energy per mole of salt', value: r.ePerMol, unit: 'kJ/mol' }, { label: 'Water recovery', value: 100 * r.waterRec, unit: '%' },
      { label: 'Productivity', value: r.prod, unit: 'L/m²·h' }, { label: 'Cell area required', value: r.cellArea, unit: 'm²' }, { label: 'Electrode mass', value: r.mass, unit: 'kg' }, { label: 'Average power', value: r.power, unit: 'kW' },
    ],
    recommendations: [!v.cdiMem ? 'Add ion-exchange membranes (MCDI) to raise the charge efficiency and allow reversed-voltage regeneration.' : null, v.cdiRecov < 30 ? 'Recover the discharge energy with a DC/DC converter: 30–60 % of the charging energy can be returned.' : null, r.c0 > 100 ? 'For this salinity compare with electrodialysis (this suite) or RO (suite 1).' : null].filter(Boolean),
    plots: [
      { type: 'line', title: 'Effluent salinity during one cycle', xlabel: 'Time (min)', ylabel: 'mg/L as NaCl', series: [{ name: 'Effluent', x: r.tt, y: r.ceff }], hlines: [{ y: r.c0 * 58.44, label: 'feed' }], vlines: [{ x: v.cdiTads, label: 'discharge starts' }] },
      { type: 'line', title: 'Cell current and equilibrium voltage', xlabel: 'Time (min)', ylabel: 'A/m² · V', series: [{ name: 'Current density (A/m²)', x: r.tt, y: r.cur }, { name: 'Double-layer voltage × 10 (V)', x: r.tt, y: r.volt.map((x) => 10 * x) }] },
      { type: 'line', title: 'Equilibrium salt adsorption versus cell voltage', xlabel: 'Cell voltage (V)', ylabel: 'mg NaCl per g of electrodes', series: [{ name: 'Modified-Donnan, with membranes', x: Vs, y: md.map((q) => (q.charge / F) * v.alphaC * 58.44e3), mode: 'both' }, { name: 'Modified-Donnan, no membranes', x: Vs, y: md.map((q) => 0.5 * v.vmi * 1e-6 * (q.a.ions - q.b.ions) * 58.44e3), mode: 'both' }, { name: 'Gouy–Chapman–Stern (planar)', x: Vs, y: g.map((q) => 0.5 * q.w * v.aBET * 58.44e3), mode: 'both', dash: true }], vlines: [{ x: v.Vch, label: 'charging' }] },
      { type: 'line', title: 'Charge efficiency versus cell voltage', xlabel: 'Cell voltage (V)', ylabel: '–', series: [{ name: 'Modified-Donnan, no membranes', x: Vs, y: md.map((q) => q.effCDI), mode: 'both' }, { name: 'Gouy–Chapman–Stern: tanh(Δφ_d/4)', x: Vs, y: g.map((q) => q.eff), mode: 'both', dash: true }], hlines: [{ y: v.alphaC, label: 'with membranes' }] },
      { type: 'line', title: 'Equilibrium adsorption versus feed salinity', xlabel: 'Feed concentration (eq/m³)', ylabel: 'mg/g · –', logx: true, series: [{ name: `Salt adsorption capacity, ${name} (mg/g)`, x: cs, y: cs.map((c) => cdiEquilibrium(c, v).sac), mode: 'both' }, { name: 'Charge efficiency without membranes × 10', x: cs, y: cs.map((c) => 10 * cdiEquilibrium(c, v).effCDI), mode: 'both' }], vlines: [{ x: r.c0, label: 'feed' }] },
      { type: 'field', title: 'Salt adsorption capacity versus voltage and salinity (no membranes)', xlabel: 'Cell voltage (V)', ylabel: 'Feed concentration (eq/m³)', zlabel: 'SAC', zunit: 'mg/g', x: fV, y: cs, z: cs.map((c) => fV.map((V) => cdiEquilibrium(c, { ...v, cdiMem: false }, V, 0).sac)), cmap: 'viridis', contours: 8, markers: [{ x: clamp(v.Vch, 0.6, 1.4), y: clamp(r.c0, 2, 200), label: 'operating' }] },
    ],
    tables: [
      { title: 'Double-layer state at the charging voltage', columns: ['Quantity', 'Value', 'Unit'], rows: [['Donnan potential (per electrode)', r.eq.a.phiD, 'V'], ['Stern potential (per electrode)', r.eq.a.phiSt, 'V'], ['Micropore charge density', r.eq.a.sigma, 'mol/m³'], ['Micropore ion concentration', r.eq.a.ions, 'mol/m³'], ['Charge efficiency without membranes', 100 * r.eq.effCDI, '%'], ['Debye length in the feed', gcs(v.Vch / 2, r.c0, v).lamD * 1e9, 'nm']] },
      { title: 'Cycle performance', columns: ['Quantity', 'Value', 'Unit'], rows: [['Charge passed per cycle', r.dQ, 'C/m²'], ['Salt removed per cycle', r.salt * 1000, 'mmol/m²'], ['Energy supplied on charging', r.Ein, 'J/m²'], ['Energy recoverable on discharge', Math.max(0, r.Erec), 'J/m²'], ['Net energy', r.Enet, 'J/m²'], ['Water produced per cycle', r.vol * 1000, 'L/m²'], ['Energy per mole of salt', r.ePerMol, 'kJ/mol']] },
    ],
    balances: [{ name: 'Salt adsorbed: time-integrated effluent vs double-layer state (mmol/m²)', in: r.salt * 1000, out: r.saltState * 1000 }],
    outputs: { streams: { diluate: stream(v.Qp, T, v.pH, ionsP), concentrate: stream((v.Qp * v.cdiTdes) / v.cdiTads, T, v.pH, ionsC) }, sec: r.sec, power: r.power, area: r.cellArea, cellPairs: Math.ceil(r.cellArea), currentDensity: Math.max(...r.cur), saltAdsorption: r.sacDyn, chargeEfficiency: r.effDyn, recovery: r.waterRec, process: v.cdiMem ? 'mcdi' : 'cdi' },
  };
}

/** Synthetic stack-test data: the model with slightly different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([Ucp, uLin, salinityFactor]) => {
    const m = suite.calibration.model({ ...d, Rcem: 4.4, alphaC: 0.9, shA: 0.34, Ucp, uLin, salinityFactor });
    return { Ucp, uLin, salinityFactor, iAvg: +(m.iAvg * (1 + g.normal(0, 0.012))).toFixed(1), tdsOut: +(m.tdsOut * (1 + g.normal(0, 0.008))).toFixed(0), eff: +(m.eff * (1 + g.normal(0, 0.006))).toFixed(1) };
  });
}

export default suite;
