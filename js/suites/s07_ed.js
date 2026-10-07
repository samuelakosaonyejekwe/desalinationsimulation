// Suite 7 — Electrodialysis and membrane electrochemical processes.
// ED / EDR stacks resolved along the flow path and across stages, ion by ion: electroneutral Nernst–Planck
// boundary layers (analytical film solution), limiting current density, Donnan potentials, ohmic losses,
// electrode kinetics, Faraday's law with co-ion leakage, back-diffusion, shunt currents and water transport.
// Reduced-order models of bipolar-membrane ED (acid/base) and membrane capacitive deionisation are included,
// together with a 1-D Nernst–Planck–Donnan profile solver across film | membrane | film.
import { brent, clamp, linspace, logspace, sum, rng, fmt, rk4 } from '../core/num.js';
import { R, F, KELVIN, density, viscosity } from '../core/props.js';
import { IONS, ION_IDS, WATERS, cloneIons, tds, scaleIons, chargeBalance, balanceCharge } from '../core/water.js';

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
  const rho = density(T, 0), mu = viscosity(T, 0), dh = (4 * eps) / (2 / h + ((1 - eps) * 8) / h), Re = (rho * u * dh) / mu, Sc = mu / (rho * Ds);
  const Sh = Math.max(par.shA * Math.max(Re, 1e-6) ** par.shB * Sc ** (1 / 3), 3), k = (Sh * Ds) / dh;
  return { dh, Re, Sc, Sh, k, delta: Ds / k, dpPerM: (par.kdp * 6.23 * Math.max(Re, 1) ** -0.3 * rho * u * u) / (2 * dh) };
}

/**
 * Local cell-pair electrochemistry for given diluate/concentrate compositions.
 * Returns the limiting currents and a function U(i) with its components (V per cell pair).
 */
export function cellPair(cd, cc, u, G, par, T) {
  const d = electrolyte(cd, T), c = electrolyte(cc, T), hd = channel(u, G.h, G.eps, T, d.Ds, par), hc = channel(u, G.h, G.eps, T, c.Ds, par), V = vt(T);
  const tC = 0.5 * (1 + par.alphaC), tA = 0.5 * (1 + par.alphaA); // counter-ion transport numbers in the membranes
  // film coefficients of the analytical Nernst–Planck solution: K1 sets the slope of c, K2/K1 the potential drop
  const K = (tm, Dct, Dco) => ({ k1: tm / Dct - (1 - tm) / Dco, k2: tm / Dct + (1 - tm) / Dco });
  const kC = K(tC, d.Dc, d.Da), kA = K(tA, d.Da, d.Dc), kCc = K(tC, c.Dc, c.Da), kAc = K(tA, c.Da, c.Dc);
  const ilimC = (2 * F * d.ceq) / (hd.delta * kC.k1), ilimA = (2 * F * d.ceq) / (hd.delta * kA.k1), ilim = Math.min(ilimC, ilimA);
  const bulkD = Math.max(G.h - 2 * hd.delta, 0.2 * G.h), bulkC = Math.max(G.h - 2 * hc.delta, 0.2 * G.h);
  const Rohm = par.Rcem + par.Raem + (bulkD / d.kappa + bulkC / c.kappa) / G.shadow;
  const parts = (i) => {
    const wdC = d.ceq * (1 - i / ilimC), wdA = d.ceq * (1 - i / ilimA), wcC = c.ceq + (i * hc.delta * kCc.k1) / (2 * F), wcA = c.ceq + (i * hc.delta * kAc.k1) / (2 * F);
    const donnan = V * (par.alphaC + par.alphaA) * Math.log(c.ceq / d.ceq);
    const memb = V * (par.alphaC * Math.log(wcC / wdC) + par.alphaA * Math.log(wcA / wdA));
    const films = V * ((kC.k2 / kC.k1) * Math.log(d.ceq / wdC) + (kA.k2 / kA.k1) * Math.log(d.ceq / wdA) + (kCc.k2 / kCc.k1) * Math.log(wcC / c.ceq) + (kAc.k2 / kAc.k1) * Math.log(wcA / c.ceq));
    return { donnan, polar: memb - donnan + films, ohmMem: i * (par.Rcem + par.Raem), ohmD: (i * bulkD) / d.kappa / G.shadow, ohmC: (i * bulkC) / c.kappa / G.shadow, wd: Math.min(wdC, wdA), wc: Math.max(wcC, wcA), U: memb + films + i * Rohm };
  };
  return { d, c, hd, hc, tC, tA, ilimC, ilimA, ilim, Rohm, parts, U: (i) => parts(i).U, E0: V * (par.alphaC + par.alphaA) * Math.log(c.ceq / d.ceq) };
}

/** Current density (A/m²) for a cell-pair voltage: film branch below the limiting current plus an empirical over-limiting branch. */
export function currentAt(cp, Ucp, par) {
  if (!(Ucp > cp.E0)) return { i: 0, iFilm: 0, iOver: 0 };
  const top = cp.ilim * (1 - 1e-9), hi = Math.min(top, (Ucp - cp.E0) / cp.Rohm), iFilm = cp.U(hi) <= Ucp ? hi : brent((i) => cp.U(i) - Ucp, 0, hi, 1e-13 * cp.ilim, 100); // U ≥ E₀ + i·R_ohm bounds the root
  const Uol = cp.U(0.98 * cp.ilim) + par.plateau, iOver = Ucp > Uol ? (par.olSlope * (Ucp - Uol)) / cp.Rohm : 0;
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

/** Electrode pair: reversible water-electrolysis voltage + Butler–Volmer overpotentials + rinse-compartment ohmic drop, V. */
export function electrodeVoltage(i, par, T) {
  const b = 2 * vt(T); // symmetric Butler–Volmer, transfer coefficient 0.5: η = (2RT/F)·asinh(i / 2i₀)
  return 1.229 + b * Math.asinh(i / (2 * par.i0a)) + b * Math.asinh(i / (2 * par.i0c)) + i * par.Rrinse;
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
    tw: v.tw, Lp: (v.Lp * 1e-6) / 3600 / 1e5, plateau: v.plateau, olSlope: v.olSlope, fws: v.fws, i0a: v.i0a, i0c: v.i0c, Rrinse: v.Rrinse * 1e-4, shunt: clamp(v.shunt / 100, 0, 0.5) };
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
  const cp = cellPair(cd, cc, u, G, par, T), Umax = cp.U(0.98 * cp.ilim) + par.plateau + 0.8, U = linspace(cp.E0, Umax, n);
  return { cp, U, i: U.map((x) => currentAt(cp, x, par).i), Ulim: cp.U(0.98 * cp.ilim), Uover: cp.U(0.98 * cp.ilim) + par.plateau };
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
  referenceOnly: ['poisson equation', 'poisson-nernst-planck', 'nernst-planck-poisson', 'electroconvection'],
  implemented: ['nernst-planck equation', 'poisson equation', 'electroneutral nernst-planck', 'nernst equation', 'donnan-equilibrium', 'ohm', 'faraday', 'butler-volmer', 'tafel', 'charge-conservation', 'current-continuity', 'ionic mass balance', 'convection-diffusion', 'water-dissociation', 'membrane partition',
    'donnan-nernst-planck', 'electro-osmosis-ion-transport', 'electrodialysis-water-splitting', 'bipolar-membrane', 'electrode-reaction-ion-transport',
    'ion concentration', 'electric potential', 'membrane charge', 'temperature', 'velocity', 'electrode state', 'fixed-potential', 'imposed-current', 'electrode butler-volmer', 'donnan-interface', 'ion-partition', 'specified-concentration', 'inlet-flow', 'membrane-interface continuity',
    'electrolyte chemistry', 'ionic-species transport', 'diffusion', 'electromigration', 'convection', 'electric-potential calculation', 'current-density prediction', 'ion-exchange membrane modelling', 'membrane selectivity', 'membrane resistance', 'electrode reactions', 'electrode compartments', 'concentration and diluate channels', 'concentration polarisation', 'limiting-current assessment', 'water transport', 'electro-osmosis', 'acid-base chemistry', 'electrochemical reactions', 'electrical-energy consumption', 'stack configuration', 'dynamic simulation'],
  equationsNote: 'Channels are one-dimensional plug flow with a film (Sherwood) boundary layer; the film solution is the exact electroneutral Nernst–Planck result for an equivalent binary salt built from the equivalent-weighted ion diffusivities, so the space-charge region of the full Poisson problem is not resolved. Membrane transport numbers split the current between counter-ions in proportion to mobility × concentration × selectivity. The over-limiting branch (plateau length and slope) and the share of water splitting are empirical inputs. Activity coefficients are ideal except in the gypsum and calcite saturation ratios (Davies), which are screening values — use suite 2 for speciation; membrane fouling is not modelled. Bipolar-membrane ED and capacitive deionisation are reduced-order models (lumped unit voltage with empirical current efficiency; equilibrium modified-Donnan / Gouy–Chapman–Stern double layers with RC charging) intended for sizing, not for stack design. Valid for roughly 0.2–40 g/L and 5–45 °C.',

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
      { key: 'plateau', label: 'Limiting-plateau length', unit: 'V per cell pair', value: 0.6, min: 0.05, max: 3, help: 'Extra voltage beyond the limiting current before over-limiting conduction sets in.' },
      { key: 'olSlope', label: 'Over-limiting conductance ÷ ohmic conductance', unit: '–', value: 0.5, min: 0, max: 2, help: 'Slope of the over-limiting branch of the polarisation curve.' },
      { key: 'fws', label: 'Share of over-limiting current from water splitting', unit: '–', value: 0.5, min: 0, max: 1, help: 'The remainder is carried by salt through electro-convection.' },
    ] },
    { group: 'Electrodes and power supply', tab: 'setup', showIf: (v) => v.process !== 'mcdi', help: 'Boundary conditions at the electrode compartments.', fields: [
      { key: 'i0a', label: 'Anode exchange current density', unit: 'A/m²', value: 0.001, min: 1e-6, max: 100, help: 'Oxygen evolution on a mixed-metal-oxide anode.' },
      { key: 'i0c', label: 'Cathode exchange current density', unit: 'A/m²', value: 0.1, min: 1e-5, max: 1000, help: 'Hydrogen evolution on stainless steel or nickel.' },
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
      { key: 'nProf', label: 'Points across the membrane (profile solver)', unit: '', value: 40, min: 6, max: 400, step: 1, help: 'RK4 steps of the Nernst–Planck–Donnan profile inside the membrane.' },
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

  run(v) {
    if (v.process === 'bpmed') return runBPMED(v);
    if (v.process === 'mcdi') return runCDI(v);
    return v.mode === 'batch' ? runBatch(v) : runED(v);
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

  verify() {
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
  const r = simulateED(v), W = [], p = v, t = r.tr, T = r.T, st = r.stages;
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
  const stageEdges = st.slice(1).map((_, k) => ({ x: (k + 1) * r.G.L, label: `stage ${k + 2}` }));
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
    ],
    recommendations: [
      r.ratioMax > 0.8 ? 'Reduce the voltage of the last stage or raise the linear velocity: the limiting current is proportional to the diluate concentration and falls stage by stage.' : null,
      !r.reached ? 'Add stages or lengthen the flow path; alternatively accept a higher product TDS and polish with a small RO or ion-exchange step.' : null,
      r.scWall.gypsum > 1 && !p.edr ? 'Switch on polarity reversal (EDR) or lower the recovery to keep calcium sulphate below saturation in the concentrate.' : null,
      vb['Diluate channel (ohmic)'] > 0.45 * sum(Object.values(vb)) ? 'Most of the voltage is lost in the dilute channels: thinner spacers or conductive (ion-exchange resin filled) spacers lower the energy.' : null,
      r.eff < 0.85 ? 'Improve current efficiency with higher-permselectivity membranes or a lower concentrate salinity (lower recovery).' : null,
      'Send the concentrate to suite 2 (Brine chemistry) for a full scaling check, and compare the cost of water with RO in suite 13 (Economics).',
    ].filter(Boolean),
    plots: [
      { type: 'line', title: 'Salinity along the flow path', xlabel: 'Path length through all stages (m)', ylabel: 'TDS (mg/L)', logy: true, series: [{ name: 'Diluate', x: xs, y: segs.map((g) => tdsOf(g.cd, r.neutral)) }, { name: 'Concentrate', x: xs, y: segs.map((g) => tdsOf(g.cc, r.neutral)) }, { name: 'Diluate at the membrane wall', x: xs, y: segs.map((g) => (tdsOf(g.cd, r.neutral) * g.wd) / g.ceqD), dash: true }], hlines: p.mode === 'design' ? [{ y: p.targetTDS, label: 'target' }] : [], vlines: stageEdges },
      { type: 'line', title: 'Current density and limiting current density', xlabel: 'Path length through all stages (m)', ylabel: 'A/m²', series: [{ name: 'Operating current density', x: xs, y: segs.map((g) => g.i) }, { name: 'Limiting current density', x: xs, y: segs.map((g) => g.ilim), dash: true }, { name: 'i / i_lim (%)', x: xs, y: segs.map((g) => 100 * g.ratio) }], hlines: [{ y: 80, label: '80 % guideline' }], vlines: stageEdges },
      { type: 'bar', title: 'Voltage breakdown per cell pair (path average)', ylabel: 'V', categories: Object.keys(vb), series: [{ name: 'Voltage', values: Object.values(vb) }] },
      { type: 'line', title: 'Polarisation curve of one cell pair', xlabel: 'Voltage per cell pair (V)', ylabel: 'Current density (A/m²)', series: [{ name: 'Stack inlet', x: polIn.U, y: polIn.i }, { name: 'Stack outlet', x: polOut.U, y: polOut.i }], vlines: [{ x: polIn.Ulim, label: 'limiting plateau' }, { x: polIn.Uover, label: 'over-limiting' }, { x: st[0].U, label: 'operating' }], hlines: [{ y: polIn.cp.ilim, label: 'i_lim inlet' }], note: 'Ohmic region at low voltage, limiting plateau where the wall concentration approaches zero, then the over-limiting branch (water splitting and electro-convection; empirical slope).' },
      ...(prof ? [prof.plot] : []),
      { type: 'bar', title: 'Ion removal from the diluate', ylabel: '% removed', categories: act.map((j) => IONS[CH[j]].label), series: [{ name: 'Removal', values: act.map((j) => rem(j)) }] },
      ...(sw.length > 1 ? [{ type: 'line', title: 'Desalination energy versus feed salinity', xlabel: 'Feed TDS (g/L)', ylabel: 'kWh/m³', logx: true, series: [{ name: 'Specific energy (design re-solved)', x: sw.map((q) => q[0]), y: sw.map((q) => q[1]), mode: 'both' }, { name: 'Reversible minimum', x: sw.map((q) => q[0]), y: sw.map((q) => q[2]), mode: 'both', dash: true }, { name: 'Stages required', x: sw.map((q) => q[0]), y: sw.map((q) => q[3]), mode: 'points' }], note: 'Same product target and design rules; ED energy is nearly proportional to the salt removed.' }] : []),
      { type: 'field', title: 'Product TDS versus voltage and velocity (this stack)', xlabel: 'Voltage per cell pair (V)', ylabel: 'Linear velocity (cm/s)', zlabel: 'Product TDS', zunit: 'mg/L', x: Us, y: us, z: map, cmap: 'salinity', contours: 8, markers: [{ x: clamp(sum(st.map((s) => s.U)) / r.nSt, 0.2, 1.4), y: clamp(u0 * 100, 3, 16), label: 'operating' }], note: `${r.nSt} stage${r.nSt > 1 ? 's' : ''} with the same voltage on every stage; coarse grid (4 segments per stage).` },
      { type: 'line', title: 'Concentration-polarisation and water transport', xlabel: 'Path length through all stages (m)', ylabel: 'see legend', series: [{ name: 'Wall ÷ bulk concentration, diluate', x: xs, y: segs.map((g) => g.wd / g.ceqD) }, { name: 'Wall ÷ bulk concentration, concentrate', x: xs, y: segs.map((g) => g.wc / g.ceqC) }, { name: 'Water flux to concentrate (L/m²·h)', x: xs, y: segs.map((g) => g.jw * 3.6e6) }], vlines: stageEdges },
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
    ],
    balances: [
      { name: 'Water (m³/h)', in: t.Qf * 3600, out: (t.Qp + t.Qbd) * 3600 },
      { name: 'Total salt (kg/h)', in: t.Qf * tdsOf(r.cf) * 3.6, out: (t.Qp * tdsOf(t.cp) + t.Qbd * tdsOf(t.ccOut)) * 3.6 },
      { name: 'Sodium (mol/s)', in: t.Qf * r.cf[CH.indexOf('Na')], out: t.Qp * t.cp[CH.indexOf('Na')] + t.Qbd * t.ccOut[CH.indexOf('Na')] },
      { name: 'Chloride (mol/s)', in: t.Qf * r.cf[CH.indexOf('Cl')], out: t.Qp * t.cp[CH.indexOf('Cl')] + t.Qbd * t.ccOut[CH.indexOf('Cl')] },
      { name: 'Charge: cation vs anion equivalents removed (eq/s)', in: sum(t.tr.map((x, j) => (Z[j] > 0 ? x * AZ[j] : 0))), out: sum(t.tr.map((x, j) => (Z[j] < 0 ? x * AZ[j] : 0))) },
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
