// Suite 12 — Energy recovery and pump calculations.
// Hydraulic network (Darcy–Weisbach with Colebrook–White, minor losses, static head), pump curves and
// operating point, affinity laws, specific speed, NPSH, motor/drive efficiency maps, the four common
// energy-recovery devices with a full pressure-exchanger mass/salt/energy balance, a method-of-
// characteristics water-hammer solver, a variable-operation study and lifecycle energy cost.
import { brent, clamp, linspace, sum, rng, fmt, lstsq } from '../core/num.js';
import { density, viscosity, psatSeawater, osmoticPressure, salinityFromTDS, G } from '../core/props.js';

const kW = (Q, dPbar) => (Q * dPbar) / 36; // m³/h × bar → kW (hydraulic)
const KBULK = 2.34e9; // bulk modulus of seawater, Pa
const polyval = (c, x) => c.reduceRight((s, a) => s * x + a, 0);

/** Standard-atmosphere pressure at elevation z (m), Pa. */
export const atmPressure = (z = 0) => 101325 * (1 - 2.25577e-5 * clamp(z, -400, 8000)) ** 5.25588;

/** Darcy friction factor. method: 'colebrook' (implicit), 'swamee' (explicit). Laminar below Re 2000, blended to 4000. */
export function frictionFactor(Re, rr, method = 'colebrook') {
  if (!(Re > 0)) return 0;
  const turb = (re) => {
    const sj = 0.25 / Math.log10(rr / 3.7 + 5.74 / re ** 0.9) ** 2;
    if (method === 'swamee') return sj;
    let x = 1 / Math.sqrt(sj); // x = 1/√f, fixed-point iteration of Colebrook–White
    for (let i = 0; i < 60; i++) { const xn = -2 * Math.log10(rr / 3.7 + (2.51 * x) / re); if (Math.abs(xn - x) < 1e-13) { x = xn; break; } x = xn; }
    return 1 / (x * x);
  };
  if (Re < 2000) return 64 / Re;
  if (Re > 4000) return turb(Re);
  const w = (Re - 2000) / 2000;
  return (1 - w) * (64 / 2000) + w * turb(4000);
}

/** Head loss (m) of a pipe run: friction + minor losses. Q in m³/s. */
export function pipeLoss(Q, { L, D, eps, rho, mu, K = 0, method = 'colebrook', C = 140 }) {
  const A = (Math.PI * D * D) / 4, v = Q / A, Re = (rho * Math.abs(v) * D) / mu, hv = (v * v) / (2 * G);
  const f = frictionFactor(Re, eps / D, method === 'hazen' ? 'colebrook' : method);
  const hf = method === 'hazen' ? (10.67 * L * Math.abs(Q) ** 1.852) / (C ** 1.852 * D ** 4.8704) : f * (L / D) * hv;
  return { hf, hm: K * hv, v, Re, f, total: hf + K * hv };
}

/** Synthetic single-pump curve through a best-efficiency point; shape follows the specific speed nq. */
export function synthCurve(Qb, Hb, etab, nq = 30) {
  const hs = 1.14 + 0.36 * clamp((nq - 15) / 100, 0, 1), a = 0.2 * (hs - 1), b = 0.8 * (hs - 1);
  return { Qb, Hb, etab, shutoff: hs * Hb, src: 'synthesised',
    H: (Q) => { const q = Q / Qb; return Hb * (hs - a * q - b * q * q); },
    eta: (Q) => { const q = Q / Qb; return Math.max(0.02, etab * (1 - Math.abs(1 - q) ** 1.9 * (q > 1 ? 1.25 : 1))); } };
}

/** Least-squares polynomial pump curve from a table of { Q, H, eta(%) } rows. */
export function tableCurve(rows, order = 2) {
  const pts = (rows || []).filter((r) => Number.isFinite(+r.Q) && Number.isFinite(+r.H)).map((r) => ({ Q: +r.Q, H: +r.H, eta: +r.eta || 0 }));
  if (pts.length < order + 1) throw new Error(`The pump-curve table needs at least ${order + 1} rows with flow and head.`);
  const Qm = Math.max(...pts.map((p) => p.Q)) || 1, X = pts.map((p) => Array.from({ length: order + 1 }, (_, j) => (p.Q / Qm) ** j));
  const cH = lstsq(X, pts.map((p) => p.H)), cE = lstsq(X, pts.map((p) => p.eta / 100));
  const H = (Q) => polyval(cH, Q / Qm), eta = (Q) => clamp(polyval(cE, Q / Qm), 0.02, 0.95);
  let Qb = Qm / 2, best = -1;
  for (const Q of linspace(0.05 * Qm, Qm, 240)) { const e = polyval(cE, Q / Qm); if (e > best) { best = e; Qb = Q; } }
  const fitErr = Math.sqrt(sum(pts.map((p) => (H(p.Q) - p.H) ** 2)) / pts.length);
  return { Qb, Hb: H(Qb), etab: eta(Qb), shutoff: H(0), src: `table fit (order ${order})`, H, eta, fitErr, Qmax: Qm };
}

/** Motor efficiency versus load fraction (fixed + load-dependent losses). */
export const motorEff = (load, rated) => { const x = clamp(load, 0.05, 1.3), l = 1 / rated - 1; return x / (x + l * (0.35 + 0.65 * x * x)); };
/** Variable-frequency-drive efficiency versus load fraction. */
export const vfdEff = (load, rated) => { const x = clamp(load, 0.05, 1.3), l = 1 / rated - 1; return x / (x + l * (0.3 + 0.2 * x + 0.5 * x * x)); };

/**
 * Isobaric pressure exchanger around a membrane train. Flows m³/h, salinities in consistent units.
 * leak = lubrication flow / HP brine inlet, of = over-flush (LP feed in / HP brine in − 1), mix0 = volumetric mixing at balanced flow.
 */
export function pxBalance({ Qf, r, Sf, rej = 0.997, leak = 0.015, of = 0, mix0 = 0.06 }) {
  const Qp = Qf * r, Qc = Qf - Qp, L = leak * Qc, QhpOut = Qc - L, QlpIn = Qc * (1 + of), QlpOut = QlpIn + L, Qhp = Qf - QhpOut;
  const M = mix0 * Math.exp(-12 * Math.max(0, of));
  let Sm = Sf, Sb = Sf / (1 - r), Shp = Sf, Sp = 0;
  for (let i = 0; i < 200; i++) {
    Shp = Sf + M * (Sb - Sf);
    const SmN = (Qhp * Sf + QhpOut * Shp) / Qf;
    Sp = (1 - rej) * SmN;
    const SbN = Qc > 0 ? (Qf * SmN - Qp * Sp) / Qc : SmN, d = Math.abs(SbN - Sb) + Math.abs(SmN - Sm);
    Sm = SmN; Sb = SbN;
    if (d < 1e-13 * Sf) break;
  }
  Shp = Sf + M * (Sb - Sf);
  const Slp = QlpOut > 0 ? (Qc * Sb + QlpIn * Sf - QhpOut * Shp) / QlpOut : Sb;
  return { Qp, Qc, L, Qhp, QhpOut, QlpIn, QlpOut, Qsys: Qhp + QlpIn, M, Sm, Sb, Shp, Sp, Slp, salinityRise: Sm / Sf - 1 };
}

/** Energy account of one energy-recovery option. Pressures in bar, flows m³/h, efficiencies as fractions. */
export function erdCase(type, p) {
  const px = pxBalance({ Qf: p.Qf, r: p.r, Sf: p.S, rej: p.rej, leak: type === 'px' ? p.leak : 0, of: type === 'px' ? p.of : 0, mix0: type === 'px' ? p.mix0 : 0 });
  const Qp = px.Qp, Qc = px.Qc, eD = p.etaMotor * p.etaVfd;
  let dPmix = 0, Pmem = p.Pmem;
  if (type === 'px' && p.applyMix) { dPmix = (osmoticPressure(p.T, Math.min(150, p.S * (1 + px.salinityRise) * p.avgCF)) - osmoticPressure(p.T, Math.min(150, p.S * p.avgCF))) / 1e5; Pmem += dPmix; }
  const Pc = Pmem - p.dpMem, o = { type, Qp, Qc, Pmem, Pc, dPmix, px, hpFlow: p.Qf, hpDp: Pmem - p.Ps, boostFlow: 0, boostDp: 0, boostElec: 0, recovered: 0, erdEff: 0, valveLoss: 0 };
  if (type === 'px') {
    const PhpOut = Pc - p.dpHP, PlpOut = p.Ps - p.dpLP;
    o.hpFlow = px.Qhp; o.boostFlow = px.QhpOut; o.boostDp = Math.max(0, Pmem - PhpOut + p.dpCirc);
    o.boostElec = kW(o.boostFlow, o.boostDp) / (p.etaBooster * eD);
    o.recovered = kW(px.QhpOut, PhpOut - p.Ps);
    o.erdEff = (px.QhpOut * PhpOut + px.QlpOut * PlpOut) / (Qc * Pc + px.QlpIn * p.Ps);
    o.PhpOut = PhpOut; o.PlpOut = PlpOut; o.residual = kW(px.QlpOut, Math.max(0, PlpOut - p.Pbrine));
    o.hpShaft = kW(o.hpFlow, o.hpDp) / p.etaHP;
  } else if (type === 'pelton' || type === 'hprt') {
    const et = type === 'pelton' ? p.etaPelton : p.etaHPRT;
    o.recovered = kW(Qc, Math.max(0, Pc - p.Pbrine)) * et; o.erdEff = et;
    o.hpShaft = Math.max(0, kW(p.Qf, o.hpDp) / p.etaHP - o.recovered); o.residual = kW(Qc, Math.max(0, Pc - p.Pbrine)) * (1 - et);
  } else if (type === 'turbo') {
    const boost = p.etaTurbo * (Qc / p.Qf) * Math.max(0, Pc - p.Pbrine); // hydraulic energy transfer to the full feed flow
    o.boost = boost; o.hpDp = Pmem - boost - p.Ps; o.recovered = kW(p.Qf, boost); o.erdEff = p.etaTurbo;
    o.hpShaft = kW(p.Qf, o.hpDp) / p.etaHP; o.residual = kW(Qc, Math.max(0, Pc - p.Pbrine)) * (1 - p.etaTurbo);
  } else { o.valveLoss = kW(Qc, Math.max(0, Pc - p.Pbrine)); o.residual = o.valveLoss; o.hpShaft = kW(p.Qf, o.hpDp) / p.etaHP; }
  o.hpElec = o.hpShaft / eD; o.net = o.hpElec + o.boostElec; o.sec = o.net / Qp;
  o.hydraulicGross = kW(p.Qf, Pmem - p.Ps);
  return o;
}

/**
 * Method-of-characteristics water hammer in a single pipe (N reaches, Δt = Δx/a).
 * scenario 'valve': upstream reservoir, downstream valve closing in tc; 'trip': prescribed pump run-down at the
 * upstream end (check valve shuts at zero flow), downstream reservoir. Heads in m above the upstream datum.
 */
export function waterHammer({ L, D, a, f, Q0, Hd, hv0 = 2, z = 0, N = 40, tSim = 10, scenario = 'valve', tc = 0, tI = 1, shape = 1 }) {
  N = Math.max(2, Math.round(N));
  const A = (Math.PI * D * D) / 4, dx = L / N, dt = dx / a, B = a / (G * A), R = (f * dx) / (2 * G * D * A * A);
  const steps = Math.max(4, Math.min(40000, Math.ceil(tSim / dt)));
  const hv = scenario === 'valve' ? Math.max(hv0, 1e-9) : 0, Hup = Hd + hv + N * R * Q0 * Q0;
  let H = Array.from({ length: N + 1 }, (_, i) => Hup - i * R * Q0 * Q0), Q = new Array(N + 1).fill(Q0);
  const Hmax = [...H], Hmin = [...H], H0 = [...H], node = scenario === 'valve' ? N : 0, t = [0], Hn = [H[node]], Qn = [Q[node]];
  let closed = false;
  for (let s = 1; s <= steps; s++) {
    const time = s * dt, Hn1 = new Array(N + 1), Qn1 = new Array(N + 1);
    for (let i = 1; i < N; i++) {
      const Cp = H[i - 1] + B * Q[i - 1] - R * Q[i - 1] * Math.abs(Q[i - 1]), Cm = H[i + 1] - B * Q[i + 1] + R * Q[i + 1] * Math.abs(Q[i + 1]);
      Hn1[i] = 0.5 * (Cp + Cm); Qn1[i] = (Cp - Cm) / (2 * B);
    }
    const Cp = H[N - 1] + B * Q[N - 1] - R * Q[N - 1] * Math.abs(Q[N - 1]), Cm = H[1] - B * Q[1] + R * Q[1] * Math.abs(Q[1]);
    if (scenario === 'valve') {
      Hn1[0] = Hup; Qn1[0] = (Hup - Cm) / B;
      const tau = tc > 0 ? Math.max(0, 1 - time / tc) ** shape : 0, Cv = (tau * tau * Q0 * Q0) / hv, dH = Cp - Hd;
      Qn1[N] = Cv <= 0 ? 0 : dH >= 0 ? 0.5 * (-Cv * B + Math.sqrt(Cv * Cv * B * B + 4 * Cv * dH)) : -0.5 * (-Cv * B + Math.sqrt(Cv * Cv * B * B - 4 * Cv * dH));
      Hn1[N] = Cp - B * Qn1[N];
    } else {
      let q = Q0 / (1 + time / Math.max(tI, 1e-6));
      if (closed || q < 0.02 * Q0) { q = 0; closed = true; }
      Qn1[0] = q; Hn1[0] = Cm + B * q;
      Hn1[N] = Hd; Qn1[N] = (Cp - Hd) / B;
    }
    H = Hn1; Q = Qn1;
    for (let i = 0; i <= N; i++) { if (H[i] > Hmax[i]) Hmax[i] = H[i]; if (H[i] < Hmin[i]) Hmin[i] = H[i]; }
    t.push(time); Hn.push(H[node]); Qn.push(Q[node]);
  }
  const zs = Array.from({ length: N + 1 }, (_, i) => (z * i) / N);
  return { x: zs.map((_, i) => i * dx), z: zs, H0, Hmax, Hmin, t, Hnode: Hn, Qnode: Qn, dt, dx, steps, B, v0: Q0 / A, joukowsky: (a * Q0) / (A * G), period: (4 * L) / a, node };
}

/** Wave speed in an elastic pipe (thin wall), m/s. */
export const waveSpeed = (rho, D, eMM, Egpa) => Math.sqrt(KBULK / rho) / Math.sqrt(1 + (KBULK * D) / (Egpa * 1e9 * (eMM / 1000)));

const pumpType = (nq) => (nq < 10 ? 'very low specific speed — use more stages or a positive-displacement pump' : nq < 35 ? 'radial-flow centrifugal' : nq < 80 ? 'Francis-vane (radial–mixed) centrifugal' : nq < 160 ? 'mixed-flow' : 'axial-flow (propeller)');
const sumK = (rows) => sum((rows || []).map((r) => (+r.K || 0) * (+r.n || 0)));

/** Complete steady calculation. Returned object is reused by run, calibration and verification. */
export function simulatePumps(v) {
  const T = v.T, S = v.S, rho = density(T, S), mu = viscosity(T, S), patm = atmPressure(v.elevation), pv = psatSeawater(T, S);
  const r = clamp(v.recovery / 100, 0.02, 0.97), deg = (1 - v.degRate / 100) ** Math.max(0, v.age);
  const mBar = (bar) => (bar * 1e5) / (rho * G); // bar → m of this fluid
  const avgCF = 0.5 * (1 + 1 / (1 - r));
  const P = { Qf: v.Qf, r, S, T, rej: v.rejection / 100, leak: v.pxLeak / 100, of: v.pxOverflush / 100, mix0: v.pxMix / 100, applyMix: v.applyMix, avgCF,
    Pmem: v.Pmem, dpMem: v.dpMem, Ps: v.Ps, Pbrine: v.Pbrine, dpHP: v.pxDpHP, dpLP: v.pxDpLP, dpCirc: v.dpCirc, etaHP: (v.etaHP / 100) * deg, etaBooster: (v.etaBooster / 100) * deg,
    etaPelton: v.etaPelton / 100, etaTurbo: v.etaTurbo / 100, etaHPRT: v.etaHPRT / 100, etaMotor: motorEff(0.9, v.etaMotor / 100), etaVfd: v.hpDrive === 'vfd' ? vfdEff(0.9, v.etaVfd / 100) : 1 };
  const options = ['px', 'turbo', 'pelton', 'hprt', 'none'].map((t) => erdCase(t, P)), sel = options.find((o) => o.type === v.erd) || options[0];

  // ---- hydraulic network: intake → pretreatment → HP-pump suction
  const Qsys = sel.type === 'px' ? sel.px.Qsys : v.Qf, Qint = Qsys / (1 - clamp(v.preLoss / 100, 0, 0.5)), Ktot = sumK(v.fittings);
  const pipe = { L: v.Lint, D: v.Dint, eps: v.epsInt / 1000, rho, mu, K: Ktot, C: v.hwC }, Hdel = mBar(v.Ps), Hpre = mBar(v.dpPre);
  const sysParts = (Q, method = v.friction) => { // Q total in m³/h
    const pl = pipeLoss(Q / 3600, { ...pipe, method });
    return { stat: v.zStatic, hf: v.kFric * pl.hf, hm: v.kFric * pl.hm, pre: Hpre * (Q / Qint) ** 2, del: Hdel, v: pl.v, Re: pl.Re, f: pl.f };
  };
  const Hsys = (Q, method) => { const s = sysParts(Q, method); return s.stat + s.hf + s.hm + s.pre + s.del; };
  const nPar = Math.max(1, Math.round(v.nPar)), nSer = Math.max(1, Math.round(v.nSer)), sp = clamp((v.speedPct / 100) * (v.trimPct / 100), 0.2, 1.3);
  const nqOf = (N, Qm3h, H) => (N * Math.sqrt(Qm3h / 3600)) / Math.max(H, 1e-6) ** 0.75;
  let curve;
  if (v.curveSrc === 'table') curve = tableCurve(v.curve, Math.round(v.fitOrder));
  else {
    const Qd = v.curveSrc === 'duty' ? v.dutyQ : Qint / nPar, Hd = v.curveSrc === 'duty' ? v.dutyH : (Hsys(Qint) * (1 + v.headMargin / 100)) / nSer;
    curve = synthCurve(Qd, Hd, v.dutyEta / 100, nqOf(v.rpm, Qd, Hd));
  }
  const Hpump = (Q, s = sp) => nSer * s * s * curve.H(Q / (nPar * s));
  const g = (Q) => Hpump(Q) - Hsys(Q), Qhi = 2.6 * curve.Qb * nPar * sp;
  let Qop = 0, state = 'ok';
  if (g(1e-6 * Qhi) <= 0) state = 'noflow';
  else if (g(Qhi) >= 0) { Qop = Qhi; state = 'runout'; }
  else Qop = brent(g, 1e-6 * Qhi, Qhi, 1e-9);
  const q1 = Qop / nPar, Hop = Qop > 0 ? Hpump(Qop) : Hpump(0), etaOp = (Qop > 0 ? curve.eta(q1 / sp) : 0.02) * deg, bepRatio = q1 / (curve.Qb * sp);
  const intakeHyd = (rho * G * (Qop / 3600) * Hop) / 1000, intakeShaft = intakeHyd / etaOp;
  const ratedInt = (1.15 * rho * G * ((curve.Qb * nPar) / 3600) * curve.Hb * nSer) / 1000 / curve.etab, loadInt = clamp(intakeShaft / ratedInt, 0.05, 1.3);
  const etaMi = motorEff(loadInt, v.etaMotor / 100), etaVi = v.intakeDrive === 'vfd' ? vfdEff(loadInt, v.etaVfd / 100) : 1, intakeElec = intakeShaft / (etaMi * etaVi);
  const parts = sysParts(Math.max(Qop, 1e-9));
  // specific speed, suction specific speed, NPSH
  const nq = nqOf(v.rpm, curve.Qb, curve.Hb), nss = v.nssUS / 51.65, npshrBep = ((v.rpm * Math.sqrt(curve.Qb / 3600)) / nss) ** (4 / 3);
  const npshr = npshrBep * sp * sp * (0.6 + 0.4 * (q1 / (curve.Qb * sp)) ** 2);
  const suc = pipeLoss(q1 / 3600, { L: v.Lsuc, D: v.Dsuc, eps: v.epsInt / 1000, rho, mu, K: v.Ksuc });
  const npsha = (patm - pv) / (rho * G) + v.zSuction - suc.total;
  // ---- HP pump: stage specific speed and Euler head
  const Hhp = mBar(sel.hpDp), stages = Math.max(1, Math.round(v.hpStages)), nqHP = nqOf(v.rpmHP, sel.hpFlow, Hhp / stages);
  const u2 = (Math.PI * v.D2 * v.rpmHP) / 60, cm2 = sel.hpFlow / 3600 / (Math.PI * v.D2 * (v.b2 / 1000) * 0.92), beta = (v.beta2 * Math.PI) / 180;
  const slip = 1 - Math.sqrt(Math.sin(beta)) / Math.max(3, v.nBlades) ** 0.7, cu2 = Math.max(0, slip * u2 - cm2 / Math.tan(beta));
  const euler = (u2 * cu2) / G, eulerStage = euler * (v.etaHyd / 100), eulerStages = Hhp / Math.max(eulerStage, 1e-6);
  const hpCurve = synthCurve(sel.hpFlow, Hhp * (1 + v.headMargin / 100), P.etaHP, nqHP);
  const throttlePenalty = kW(sel.hpFlow, v.dpValve) / (P.etaHP * P.etaMotor * P.etaVfd);
  const net = sel.net + intakeElec + throttlePenalty, sec = net / sel.Qp;
  const piF = osmoticPressure(T, S) / 1e5, minSEC = (piF / 36) * (-Math.log(1 - r) / r);
  return { v, rho, mu, patm, pv, r, P, options, sel, Qsys, Qint, Ktot, pipe, sysParts, Hsys, Hpump, curve, nPar, nSer, sp, Qop, state, q1, Hop, etaOp, bepRatio, intakeHyd, intakeShaft, intakeElec, etaMi, etaVi, loadInt, parts,
    nq, nss, npshr, npsha, suc, Hhp, stages, nqHP, u2, cm2, cu2, slip, euler, eulerStage, eulerStages, hpCurve, throttlePenalty, net, sec, piF, minSEC, mBar, deg, Hdel, Hpre };
}

/** Variable-operation study of the HP pump: speed control versus discharge throttling. */
function profileStudy(R, rows) {
  const { v, rho, sel, hpCurve, P } = R, pts = (rows || []).filter((p) => Number.isFinite(+p.h) && +p.flow > 0).map((p) => ({ h: +p.h, flow: +p.flow, P: +p.P || v.Pmem })).sort((a, b) => a.h - b.h);
  if (!pts.length) return { rows: [], eV: 0, eT: 0, eO: 0, infeasible: 0 };
  const span = Math.max(24, pts[pts.length - 1].h + 1), ratedShaft = (1.1 * kW(sel.hpFlow, sel.hpDp)) / P.etaHP, out = [];
  pts.forEach((p, i) => {
    const dur = (i + 1 < pts.length ? pts[i + 1].h : span) - p.h, Q = (sel.hpFlow * p.flow) / 100, dP = Math.max(0.5, p.P - (sel.boost || 0) - v.Ps), Hreq = (dP * 1e5) / (rho * G);
    const fs = (s) => s * s * hpCurve.H(Q / s) - Hreq;
    let s = 1.25, ok = true;
    if (fs(1.25) < 0) ok = false; else if (fs(0.2) > 0) s = 0.2; else s = brent(fs, 0.2, 1.25, 1e-9);
    const shaftV = (rho * G * (Q / 3600) * Hreq) / 1000 / hpCurve.eta(Q / s), loadV = shaftV / ratedShaft;
    const elecV = shaftV / (motorEff(loadV, v.etaMotor / 100) * vfdEff(loadV, v.etaVfd / 100));
    const Hfull = hpCurve.H(Q), feas = Hfull >= Hreq, Ht = Math.max(Hfull, Hreq), shaftT = (rho * G * (Q / 3600) * Ht) / 1000 / hpCurve.eta(Q);
    const elecT = shaftT / motorEff(shaftT / ratedShaft, v.etaMotor / 100);
    const other = (sel.boostElec + R.intakeElec) * (p.flow / 100) ** 2; // auxiliaries follow their system curves under speed control
    out.push({ ...p, dur, Q, Hreq, speed: 100 * s, etaV: hpCurve.eta(Q / s), elecV, Hfull, throttled: Math.max(0, Hfull - Hreq), elecT, feasT: feas, feasV: ok && s <= 1.05, other, Qp: (sel.Qp * p.flow) / 100 });
  });
  const tot = sum(out.map((o) => o.dur)), hrs = 8760 * (v.availability / 100), w = (o) => (o.dur / tot) * hrs;
  return { rows: out, eV: sum(out.map((o) => o.elecV * w(o))), eT: sum(out.map((o) => o.elecT * w(o))), eO: sum(out.map((o) => o.other * w(o))), water: sum(out.map((o) => o.Qp * w(o))), infeasible: out.filter((o) => !o.feasV).length, infeasT: out.filter((o) => !o.feasT).length };
}

const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.type === 'table' ? f.value.map((r) => ({ ...r })) : f.value]));
const ERD_NAMES = { px: 'Isobaric pressure exchanger', turbo: 'Turbocharger', pelton: 'Pelton turbine', hprt: 'Reverse-running pump', none: 'None (throttle valve)' };

const suite = {
  id: 'pump', num: 12, title: 'Energy Recovery & Pump Calculations', short: 'Pumps & ERD', icon: '⚙️',
  tagline: 'Hydraulic network, pump operating points, NPSH, energy-recovery devices, surge and lifecycle energy from intake to outfall.',
  description: 'Builds the system curve of the intake and pretreatment line with Darcy–Weisbach friction (Colebrook–White) and minor losses, intersects it with a fitted or synthesised pump curve, and applies the affinity laws for speed control, impeller trim, and pumps in parallel or series. The high-pressure duty is combined with a full mass, salt and energy balance of an isobaric pressure exchanger, or with a Pelton turbine, turbocharger or reverse-running pump, to give gross, recovered and net power and the specific energy. A method-of-characteristics solver gives the water-hammer envelope after a pump trip or valve closure, and a load-profile study compares variable-speed with throttled operation over the year.',
  guide: [
    'Enter the membrane duty (feed flow, pressure, recovery) or pull it from the RO suite.',
    'Describe the intake line: length, diameter, roughness, lift and the fittings table; choose how the supply pump curve is defined.',
    'Pick the energy-recovery device and its performance figures; the other devices are always calculated for comparison.',
    'Run. Check the operating point against the preferred region, the NPSH margin, the surge envelope and the net specific energy.',
  ],
  implemented: ['bernoulli', 'mechanical-energy', 'euler turbomachinery', 'pump-head', 'hydraulic-power', 'shaft-power', 'pump-efficiency', 'darcy-weisbach', 'colebrook-white', 'moody friction-factor', 'minor-loss', 'affinity law', 'specific-speed', 'cavitation', 'net-positive-suction-head', 'turbine equation', 'pressure-exchanger mass/energy', 'specific-energy-consumption',
    'pump-pipeline', 'pump-ro model', 'pump-energy-recovery-device', 'pressure-exchanger-membrane', 'turbine-brine', 'variable-speed-drive-pump', 'hydraulic-electrical motor', 'transient water-hammer',
    'initial pressure', 'flow', 'shaft speed', 'valve position', 'reservoir level', 'rotating-machine state', 'prescribed upstream/downstream pressure', 'prescribed flow', 'reservoir-head boundary', 'pump characteristic-curve boundary', 'pressure-exchanger interface', 'valve-loss boundary', 'no-flow boundary', 'transient characteristic/water-hammer',
    'hydraulic-network calculation', 'pump selection and sizing', 'pump-performance curve', 'motor modelling', 'pressure-loss calculation', 'high-pressure pumping', 'booster pumping', 'energy-recovery-device modelling', 'pressure-exchanger modelling', 'turbine modelling', 'efficiency mapping', 'variable-speed-drive modelling', 'piping losses', 'valve losses', 'transient hydraulics', 'cavitation assessment', 'operating-point determination', 'equipment degradation', 'energy-flow accounting', 'specific-energy-consumption calculation', 'equipment comparison', 'lifecycle energy assessment'],
  equationsNote: 'Single-phase, incompressible steady hydraulics; one equivalent pipeline from intake to the high-pressure pump suction. Synthesised pump curves are generic shapes scaled by specific speed — replace them with the certified curve before procurement. The water-hammer solver uses steady friction and does not model column separation, air valves or surge vessels: it flags when pressure reaches vapour pressure but is not valid beyond that point. Turbine, turbocharger and reverse-running-pump options use constant efficiencies.',

  inputs: [
    { group: 'Fluid and membrane duty', help: 'What the high-pressure system must deliver. Pull these from the RO suite when available.', fields: [
      { key: 'Qf', label: 'Membrane feed flow', unit: 'm³/h', value: 1000, min: 1, max: 2e5, help: 'Total flow entering the membrane train (HP pump plus pressure-exchanger outlet).' },
      { key: 'recovery', label: 'Recovery', unit: '%', value: 45, min: 5, max: 95, typical: [35, 85] },
      { key: 'Pmem', label: 'Membrane feed pressure', unit: 'bar', value: 60, min: 3, max: 120, help: 'Gauge pressure at the membrane inlet.' },
      { key: 'dpMem', label: 'Membrane-array pressure drop', unit: 'bar', value: 1.5, min: 0, max: 10, help: 'Feed-to-concentrate loss; sets the brine pressure entering the energy-recovery device.' },
      { key: 'T', label: 'Water temperature', unit: '°C', value: 25, min: 1, max: 45 },
      { key: 'S', label: 'Feed salinity', unit: 'g/kg', value: 35, min: 0, max: 80 },
      { key: 'rejection', label: 'Membrane salt rejection', unit: '%', value: 99.7, min: 50, max: 99.99, help: 'Used only in the pressure-exchanger salt balance.' },
      { key: 'Ps', label: 'HP-pump suction pressure', unit: 'bar', value: 2.5, min: 0.5, max: 10, help: 'Gauge pressure the supply system must deliver to the HP pump and to the low-pressure side of the energy-recovery device.' },
      { key: 'Pbrine', label: 'Brine discharge back-pressure', unit: 'bar', value: 1, min: 0, max: 10, help: 'Gauge pressure needed to reach the outfall.' },
    ] },
    { group: 'Intake and pretreatment line', help: 'One equivalent pipeline from the intake sump through pretreatment to the high-pressure pump suction.', fields: [
      { key: 'Lint', label: 'Pipe length', unit: 'm', value: 400, min: 1, max: 20000 },
      { key: 'Dint', label: 'Internal diameter', unit: 'm', value: 0.5, min: 0.02, max: 4 },
      { key: 'epsInt', label: 'Wall roughness', unit: 'mm', value: 0.05, min: 0.001, max: 5, help: 'GRP/HDPE 0.01–0.05, coated steel 0.05–0.1, aged or fouled pipe 0.5–2.' },
      { key: 'zStatic', label: 'Static lift', unit: 'm', value: 8, min: -20, max: 200, help: 'Elevation of the delivery point above the intake water level.' },
      { key: 'dpPre', label: 'Pretreatment pressure loss at design flow', unit: 'bar', value: 1.2, min: 0, max: 6, help: 'Screens, media or membrane filters and cartridge filters; scaled with flow squared.' },
      { key: 'preLoss', label: 'Pretreatment water loss', unit: '%', value: 4, min: 0, max: 30, help: 'Backwash and reject; the intake flow is larger than the RO feed by this share.' },
      { key: 'fittings', label: 'Fittings and valves (minor losses)', type: 'table', columns: [{ key: 'name', label: 'Fitting' }, { key: 'K', label: 'K', unit: '–' }, { key: 'n', label: 'Count' }],
        value: [{ name: 'Bell-mouth entrance', K: 0.05, n: 1 }, { name: 'Intake screen / strainer', K: 1.2, n: 1 }, { name: '90° long-radius bend', K: 0.25, n: 8 }, { name: '45° bend', K: 0.15, n: 4 }, { name: 'Butterfly valve, open', K: 0.35, n: 3 }, { name: 'Swing check valve', K: 2.0, n: 1 }, { name: 'Tee, through flow', K: 0.4, n: 2 }, { name: 'Tee, branch flow', K: 1.0, n: 1 }, { name: 'Exit to tank', K: 1.0, n: 1 }],
        help: 'Loss coefficients K applied to the velocity head of the line: h = Σ K·v²/2g.' },
    ] },
    { group: 'Supply (intake) pump', help: 'The pump, or bank of pumps, that drives the intake line.', fields: [
      { key: 'curveSrc', label: 'Pump curve', type: 'select', value: 'auto', options: [{ value: 'auto', label: 'Size for the system duty (synthesised curve)' }, { value: 'duty', label: 'Synthesise from my rated point' }, { value: 'table', label: 'Fit my curve table' }] },
      { key: 'headMargin', label: 'Head margin on selection', unit: '%', value: 5, min: 0, max: 30, help: 'Rated head above the calculated duty when pumps are sized automatically (also applied to the HP pump for the variable-operation study).' },
      { key: 'dutyQ', label: 'Rated flow per pump (best-efficiency point)', unit: 'm³/h', value: 560, min: 0.5, max: 1e5, showIf: (v) => v.curveSrc === 'duty' },
      { key: 'dutyH', label: 'Rated head per pump', unit: 'm', value: 48, min: 1, max: 1500, showIf: (v) => v.curveSrc === 'duty' },
      { key: 'dutyEta', label: 'Best-efficiency-point efficiency', unit: '%', value: 84, min: 20, max: 93, showIf: (v) => v.curveSrc !== 'table' },
      { key: 'curve', label: 'Pump curve, one pump at rated speed', type: 'table', columns: [{ key: 'Q', label: 'Flow', unit: 'm³/h' }, { key: 'H', label: 'Head', unit: 'm' }, { key: 'eta', label: 'Efficiency', unit: '%' }],
        value: [{ Q: 0, H: 60, eta: 0 }, { Q: 150, H: 59, eta: 44 }, { Q: 300, H: 56.5, eta: 70 }, { Q: 450, H: 52.5, eta: 82 }, { Q: 560, H: 48, eta: 84.5 }, { Q: 650, H: 43, eta: 82 }, { Q: 750, H: 36, eta: 73 }], showIf: (v) => v.curveSrc === 'table' },
      { key: 'fitOrder', label: 'Curve-fit order', type: 'select', value: 2, options: [{ value: 2, label: 'Quadratic' }, { value: 3, label: 'Cubic' }], showIf: (v) => v.curveSrc === 'table' },
      { key: 'nPar', label: 'Pumps running in parallel', unit: '', value: 2, min: 1, max: 12, step: 1 },
      { key: 'nSer', label: 'Pumps in series', unit: '', value: 1, min: 1, max: 4, step: 1 },
      { key: 'rpm', label: 'Rated speed', unit: 'rpm', value: 1480, min: 300, max: 3600 },
      { key: 'speedPct', label: 'Running speed (VFD)', unit: '% of rated', value: 100, min: 40, max: 110 },
      { key: 'trimPct', label: 'Impeller diameter', unit: '% of full', value: 100, min: 75, max: 100, help: 'Trim is applied through the affinity laws (Q ∝ D, H ∝ D²); valid for moderate trims only.' },
      { key: 'intakeDrive', label: 'Drive', type: 'select', value: 'vfd', options: [{ value: 'vfd', label: 'Variable-frequency drive' }, { value: 'dol', label: 'Fixed speed (direct on line)' }] },
      { key: 'zSuction', label: 'Water level above pump centreline', unit: 'm', value: 2, min: -8, max: 30, help: 'Negative for a suction lift.' },
      { key: 'Lsuc', label: 'Suction pipe length', unit: 'm', value: 12, min: 0.5, max: 500 },
      { key: 'Dsuc', label: 'Suction pipe diameter', unit: 'm', value: 0.45, min: 0.02, max: 4 },
      { key: 'Ksuc', label: 'Suction-side minor-loss coefficient ΣK', unit: '–', value: 2.2, min: 0, max: 30 },
      { key: 'nssUS', label: 'Suction specific speed', unit: 'US units', value: 9000, min: 4000, max: 16000, help: 'Sets the required NPSH at the best-efficiency point: Nss = N·√Q / NPSHr^0.75 (rpm, gpm, ft). 8 000–11 000 is conventional.' },
    ] },
    { group: 'High-pressure pump', fields: [
      { key: 'etaHP', label: 'HP pump efficiency at duty', unit: '%', value: 86, min: 30, max: 93 },
      { key: 'hpStages', label: 'Number of stages', unit: '', value: 4, min: 1, max: 16, step: 1 },
      { key: 'rpmHP', label: 'Speed', unit: 'rpm', value: 2980, min: 600, max: 7200 },
      { key: 'hpDrive', label: 'Drive', type: 'select', value: 'vfd', options: [{ value: 'vfd', label: 'Variable-frequency drive' }, { value: 'dol', label: 'Fixed speed (direct on line)' }] },
      { key: 'dpValve', label: 'Control-valve throttling on HP discharge', unit: 'bar', value: 0, min: 0, max: 30, help: 'Pressure deliberately destroyed across a control valve; reported as an energy penalty.' },
      { key: 'D2', label: 'Impeller outlet diameter', unit: 'm', value: 0.32, min: 0.05, max: 1.5, help: 'For the Euler head estimate.' },
      { key: 'b2', label: 'Impeller outlet width', unit: 'mm', value: 22, min: 2, max: 300 },
      { key: 'beta2', label: 'Blade outlet angle β₂', unit: '°', value: 25, min: 10, max: 60 },
      { key: 'nBlades', label: 'Number of blades', unit: '', value: 7, min: 3, max: 12, step: 1 },
      { key: 'etaHyd', label: 'Hydraulic efficiency', unit: '%', value: 90, min: 50, max: 98 },
    ] },
    { group: 'Energy-recovery device', fields: [
      { key: 'erd', label: 'Device in service', type: 'select', value: 'px', options: Object.entries(ERD_NAMES).map(([value, label]) => ({ value, label })), help: 'All options are calculated for the comparison table; this one feeds the plant totals.' },
      { key: 'pxLeak', label: 'Lubrication (leakage) flow', unit: '% of HP brine', value: 1.5, min: 0, max: 8, showIf: (v) => v.erd === 'px' },
      { key: 'pxMix', label: 'Volumetric mixing at balanced flow', unit: '%', value: 6, min: 0, max: 15, help: 'Share of brine salinity carried into the HP feed leaving the device.', showIf: (v) => v.erd === 'px' },
      { key: 'pxOverflush', label: 'Over-flush (LP feed above HP brine)', unit: '%', value: 0, min: 0, max: 20, help: 'Extra low-pressure feed reduces mixing at the price of more pretreated water.', showIf: (v) => v.erd === 'px' },
      { key: 'pxDpHP', label: 'High-pressure differential', unit: 'bar', value: 0.7, min: 0, max: 3, showIf: (v) => v.erd === 'px' },
      { key: 'pxDpLP', label: 'Low-pressure differential', unit: 'bar', value: 0.6, min: 0, max: 3, showIf: (v) => v.erd === 'px' },
      { key: 'dpCirc', label: 'Circulation-loop piping loss', unit: 'bar', value: 0.3, min: 0, max: 3, showIf: (v) => v.erd === 'px' },
      { key: 'etaBooster', label: 'Circulation (booster) pump efficiency', unit: '%', value: 80, min: 30, max: 92, showIf: (v) => v.erd === 'px' },
      { key: 'applyMix', label: 'Add the mixing penalty to the feed pressure', type: 'bool', value: false, help: 'Turn on when the membrane pressure above was calculated without the salinity increase caused by the pressure exchanger.', showIf: (v) => v.erd === 'px' },
      { key: 'etaPelton', label: 'Pelton turbine efficiency', unit: '%', value: 88, min: 40, max: 93 },
      { key: 'etaTurbo', label: 'Turbocharger transfer efficiency', unit: '%', value: 74, min: 30, max: 88, help: 'Hydraulic energy delivered to the feed ÷ hydraulic energy taken from the brine.' },
      { key: 'etaHPRT', label: 'Reverse-running pump efficiency', unit: '%', value: 77, min: 30, max: 88 },
    ] },
    { group: 'Variable operation and energy cost', fields: [
      { key: 'profile', label: 'Operating profile', type: 'table', columns: [{ key: 'h', label: 'From hour' }, { key: 'flow', label: 'Flow', unit: '% of design' }, { key: 'P', label: 'Membrane feed pressure', unit: 'bar' }],
        value: [{ h: 0, flow: 70, P: 53 }, { h: 5, flow: 85, P: 56.5 }, { h: 8, flow: 100, P: 60 }, { h: 12, flow: 100, P: 60.5 }, { h: 17, flow: 100, P: 60 }, { h: 21, flow: 80, P: 55.5 }],
        help: 'Each row holds until the next one. Use hours of a day or any longer period — rows are weighted by duration and scaled to a year.' },
      { key: 'availability', label: 'Plant availability', unit: '%', value: 94, min: 20, max: 100 },
      { key: 'elecPrice', label: 'Electricity price', unit: '$/kWh', value: 0.08, min: 0, max: 1, help: 'Indicative; edit to the site tariff.' },
      { key: 'years', label: 'Evaluation period', unit: 'years', value: 25, min: 1, max: 50, step: 1 },
      { key: 'discount', label: 'Discount rate', unit: '%/y', value: 7, min: 0, max: 25 },
      { key: 'escalation', label: 'Electricity price escalation', unit: '%/y', value: 2, min: -5, max: 15 },
    ] },
    { group: 'Models', tab: 'setup', fields: [
      { key: 'friction', label: 'Friction model', type: 'select', value: 'colebrook', options: [{ value: 'colebrook', label: 'Darcy–Weisbach + Colebrook–White' }, { value: 'swamee', label: 'Darcy–Weisbach + Swamee–Jain (explicit)' }, { value: 'hazen', label: 'Hazen–Williams' }] },
      { key: 'hwC', label: 'Hazen–Williams coefficient C', unit: '–', value: 145, min: 60, max: 160, help: 'Used for the comparison curve and when Hazen–Williams is selected.' },
      { key: 'kFric', label: 'Hydraulic-resistance multiplier', unit: '×', value: 1, min: 0.3, max: 5, help: 'Scales pipe friction and minor losses; calibrate it against measured system head.' },
      { key: 'etaMotor', label: 'Motor efficiency at rated load', unit: '%', value: 96, min: 70, max: 98.5 },
      { key: 'etaVfd', label: 'Drive efficiency at rated load', unit: '%', value: 97.5, min: 85, max: 99.5 },
    ] },
    { group: 'Site and equipment state', tab: 'setup', help: 'Initial and boundary conditions of the hydraulic system.', fields: [
      { key: 'elevation', label: 'Site elevation', unit: 'm', value: 5, min: -400, max: 4500, help: 'Sets atmospheric pressure for the NPSH and vapour-pressure checks.' },
      { key: 'age', label: 'Equipment age since overhaul', unit: 'years', value: 0, min: 0, max: 30 },
      { key: 'degRate', label: 'Pump efficiency loss per year', unit: '%/y', value: 0.6, min: 0, max: 5, help: 'Wear-ring and surface degradation between overhauls.' },
      { key: 'overhaul', label: 'Overhaul interval', unit: 'years', value: 6, min: 1, max: 30, step: 1, help: 'Efficiency is restored at each overhaul in the lifecycle calculation.' },
    ] },
    { group: 'Limits', tab: 'setup', fields: [
      { key: 'porLo', label: 'Preferred operating region, lower', unit: '% of BEP flow', value: 70, min: 30, max: 100 },
      { key: 'porHi', label: 'Preferred operating region, upper', unit: '% of BEP flow', value: 120, min: 100, max: 150 },
      { key: 'npshMargin', label: 'Required NPSH margin ratio', unit: '–', value: 1.3, min: 1, max: 3, help: 'NPSH available ÷ NPSH required. 1.3 is a common minimum for seawater pumps.' },
      { key: 'vMax', label: 'Maximum pipe velocity', unit: 'm/s', value: 3, min: 0.5, max: 8 },
      { key: 'pnRating', label: 'Pipe pressure rating', unit: 'bar', value: 16, min: 2, max: 160 },
    ] },
    { group: 'Transient (water hammer)', tab: 'setup', help: 'Initial condition: steady flow at the operating point. Boundary conditions depend on the event.', fields: [
      { key: 'whCase', label: 'Event', type: 'select', value: 'trip', options: [{ value: 'trip', label: 'Supply-pump trip (power failure)' }, { value: 'valve', label: 'Downstream valve closure' }] },
      { key: 'tClose', label: 'Valve closure time', unit: 's', value: 4, min: 0, max: 600, help: '0 = instantaneous. Compare with the pipe period 2L/a.', showIf: (v) => v.whCase === 'valve' },
      { key: 'hValve', label: 'Valve head loss when open', unit: 'm', value: 2, min: 0.01, max: 100, showIf: (v) => v.whCase === 'valve' },
      { key: 'inertia', label: 'Rotating inertia of pump and motor', unit: 'kg·m²', value: 6, min: 0.05, max: 5000, help: 'Per pump. A larger flywheel effect slows the run-down and softens the down-surge.', showIf: (v) => v.whCase === 'trip' },
      { key: 'pipeE', label: 'Pipe-wall elastic modulus', unit: 'GPa', value: 20, min: 0.5, max: 220, help: 'GRP 10–25, HDPE 0.8–1.2, ductile iron 170, steel 200.' },
      { key: 'pipeWall', label: 'Pipe-wall thickness', unit: 'mm', value: 12, min: 1, max: 100 },
      { key: 'tSim', label: 'Simulated time', unit: 's', value: 12, min: 0.5, max: 300 },
    ] },
    { group: 'Water-hammer grid', tab: 'mesh', help: 'The pipeline is divided into equal reaches; the time step follows from Δt = Δx / a (Courant number 1).', fields: [
      { key: 'nReach', label: 'Number of reaches', unit: '', value: 40, min: 4, max: 800, step: 1 },
    ] },
  ],

  presets: [
    { name: 'SWRO train, 45 % recovery, pressure exchanger', values: {} },
    { name: 'SWRO with Pelton turbine, fixed-speed pumps', values: { erd: 'pelton', hpDrive: 'dol', intakeDrive: 'dol', dpValve: 2 } },
    { name: 'Brackish RO, 78 % recovery, turbocharger', values: { Qf: 320, recovery: 78, Pmem: 15, dpMem: 2.2, S: 3.5, erd: 'turbo', hpStages: 3, D2: 0.2, b2: 20, Lint: 150, Dint: 0.3, zStatic: 25, dpPre: 0.8, nPar: 1, rejection: 99, profile: [{ h: 0, flow: 60, P: 12.5 }, { h: 6, flow: 100, P: 15 }, { h: 20, flow: 75, P: 13.4 }] } },
    { name: 'Measured supply-pump curve at 90 % speed', values: { curveSrc: 'table', speedPct: 90, zStatic: 4, dpPre: 0.9 } },
    { name: 'Long steel intake line, valve-closure surge', values: { Lint: 1800, Dint: 0.6, epsInt: 0.1, pipeE: 200, pipeWall: 8, whCase: 'valve', tClose: 6, tSim: 30, nReach: 60, pnRating: 25 } },
  ],

  pull: ({ feed, outputs }) => [
    outputs?.ro?.feedFlow ? { key: 'Qf', value: outputs.ro.feedFlow, from: 'RO design: feed flow' } : feed?.Q ? { key: 'Qf', value: feed.Q, from: 'Case feed water' } : null,
    outputs?.ro?.recovery ? { key: 'recovery', value: 100 * outputs.ro.recovery, from: 'RO design: recovery' } : null,
    outputs?.ro?.feedPressureBar ? { key: 'Pmem', value: outputs.ro.feedPressureBar, from: 'RO design: feed pressure' } : null,
    outputs?.ro?.feedPressureBar && outputs?.ro?.concentratePressureBar ? { key: 'dpMem', value: Math.max(0, outputs.ro.feedPressureBar - outputs.ro.concentratePressureBar), from: 'RO design: array pressure drop' } : null,
    outputs?.ro?.streams?.feed?.tds ? { key: 'S', value: salinityFromTDS(outputs.ro.streams.feed.tds, outputs.ro.streams.feed.T ?? 25), from: 'RO design: feed salinity' } : null,
    outputs?.ro?.streams?.feed?.T ?? feed?.T ? { key: 'T', value: outputs?.ro?.streams?.feed?.T ?? feed.T, from: outputs?.ro?.streams?.feed?.T ? 'RO design: temperature' : 'Case feed water' } : null,
  ].filter(Boolean),
  site: (site) => [
    site?.data?.elevation !== undefined && site?.data?.elevation !== null ? { key: 'elevation', value: site.data.elevation, from: 'Site elevation' } : null,
    site?.data?.electricityPrice ? { key: 'elecPrice', value: site.data.electricityPrice, from: 'Site electricity price' } : null,
    site?.data?.lendingRate ? { key: 'discount', value: site.data.lendingRate, from: 'Site lending rate' } : null,
  ].filter(Boolean),

  run(v, ctx) {
    const R = simulatePumps(v), { sel, options, rho, curve, parts, P } = R, W = [];
    ctx?.progress?.(0.3, 'Steady hydraulics solved');
    // ---- transient
    const a = waveSpeed(rho, v.Dint, v.pipeWall, v.pipeE), Aint = (Math.PI * v.Dint ** 2) / 4, Q0 = Math.max(R.Qop, 1e-6) / 3600;
    const fEff = R.v.kFric * (parts.f + (R.Ktot * v.Dint) / v.Lint), omega = (2 * Math.PI * v.rpm * (v.speedPct / 100)) / 60;
    const tI = (v.inertia * omega * omega) / Math.max(1, (R.intakeShaft / R.nPar) * 1000);
    const wh = waterHammer({ L: v.Lint, D: v.Dint, a, f: fEff, Q0, Hd: v.zStatic + R.Hdel + R.Hpre, hv0: v.hValve, z: v.zStatic, N: v.nReach, tSim: v.tSim, scenario: v.whCase, tc: v.tClose, tI });
    const toBar = (H, i) => (rho * G * (H - wh.z[i])) / 1e5, pMaxArr = wh.Hmax.map(toBar), pMinArr = wh.Hmin.map(toBar), p0Arr = wh.H0.map(toBar);
    const pMax = Math.max(...pMaxArr), pMin = Math.min(...pMinArr), pVapG = (R.pv - R.patm) / 1e5, joukBar = (rho * a * wh.v0) / 1e5;
    ctx?.progress?.(0.6, 'Transient solved');
    // ---- variable operation and lifecycle
    const prof = profileStudy(R, v.profile), n = Math.max(1, Math.round(v.years)), i = v.discount / 100, e = v.escalation / 100;
    const annualDesign = R.net * 8760 * (v.availability / 100), annualProfile = prof.rows.length ? prof.eV + prof.eO : annualDesign;
    const lifecycle = (kWhYear) => { let pvCost = 0, energy = 0; for (let y = 1; y <= n; y++) { const ageY = (y - 1) % Math.max(1, Math.round(v.overhaul)), fac = 1 / (1 - v.degRate / 100) ** ageY; energy += kWhYear * fac; pvCost += (kWhYear * fac * v.elecPrice * (1 + e) ** (y - 1)) / (1 + i) ** y; } return { pvCost, energy }; };
    const lc = lifecycle(v.hpDrive === 'vfd' ? annualProfile : (prof.rows.length ? prof.eT + prof.eO : annualDesign));
    const lcOpt = options.map((o) => lifecycle((o.net + R.intakeElec) * 8760 * (v.availability / 100)));

    // ---- checks
    const bepPct = 100 * R.bepRatio, npshRatio = R.npsha / Math.max(R.npshr, 1e-6);
    if (R.state === 'noflow') W.push({ level: 'bad', msg: `The supply pump cannot overcome the system head at zero flow (shut-off ${fmt(R.Hpump(0), 3)} m against ${fmt(R.Hsys(0), 3)} m) — raise the speed, add a pump in series or select a higher-head pump.` });
    if (R.state === 'runout') W.push({ level: 'bad', msg: 'The supply pump runs out beyond its curve: system resistance is far below the pump head. Reduce speed or throttle.' });
    if (R.state === 'ok' && (bepPct < v.porLo || bepPct > v.porHi)) W.push({ level: 'warn', msg: `Supply pump runs at ${fmt(bepPct, 3)} % of its best-efficiency flow, outside the preferred ${v.porLo}–${v.porHi} % region — expect vibration, recirculation and bearing wear.` });
    if (R.state === 'ok' && Math.abs(R.Qop - R.Qint) / R.Qint > 0.075) W.push({ level: 'warn', msg: `Supply pumps deliver ${fmt(R.Qop, 4)} m³/h but the plant needs ${fmt(R.Qint, 4)} m³/h — adjust speed (${fmt(clamp(v.speedPct * (R.Qint / Math.max(R.Qop, 1e-6)), 20, 130), 3)} % is a first estimate), trim or the number of pumps.` });
    if (npshRatio < 1) W.push({ level: 'bad', msg: `Cavitation: NPSH available ${fmt(R.npsha, 3)} m is below the ${fmt(R.npshr, 3)} m required.` });
    else if (npshRatio < v.npshMargin) W.push({ level: 'warn', msg: `NPSH margin ratio ${fmt(npshRatio, 3)} is below the required ${v.npshMargin}.` });
    if (Math.abs(parts.v) > v.vMax) W.push({ level: 'warn', msg: `Line velocity ${fmt(parts.v, 3)} m/s exceeds ${v.vMax} m/s — larger pipe would cut friction and surge.` });
    if (Math.abs(parts.v) < 0.6 && R.Qop > 0) W.push({ level: 'info', msg: `Line velocity ${fmt(parts.v, 2)} m/s is low — sediment and biofouling can settle below about 0.6 m/s.` });
    if (pMax > v.pnRating) W.push({ level: 'bad', msg: `Surge pressure ${fmt(pMax, 3)} bar exceeds the ${v.pnRating} bar pipe rating — slow the closure, add a surge vessel or raise the pressure class.` });
    if (pMin < pVapG) W.push({ level: 'bad', msg: `Down-surge reaches vapour pressure (${fmt(pMin, 3)} bar g computed, limit ${fmt(pVapG, 3)} bar g): column separation is likely. Results after that instant are not valid — add a flywheel, air valve or surge vessel.` });
    else if (pMin < -0.5) W.push({ level: 'warn', msg: `Sub-atmospheric pressure down to ${fmt(pMin, 3)} bar g during the transient — check pipe collapse rating and air ingress at joints.` });
    if (sel.type === 'px' && sel.PlpOut < v.Pbrine) W.push({ level: 'warn', msg: `Low-pressure brine leaves the pressure exchanger at ${fmt(sel.PlpOut, 3)} bar g, below the ${v.Pbrine} bar g needed at the outfall — raise the suction pressure.` });
    if (sel.type === 'px' && sel.px.salinityRise > 0.035) W.push({ level: 'warn', msg: `Mixing in the pressure exchanger raises membrane feed salinity by ${fmt(100 * sel.px.salinityRise, 3)} % — consider a small over-flush.` });
    if (R.nqHP < 12) W.push({ level: 'warn', msg: `HP pump specific speed per stage is only ${fmt(R.nqHP, 3)} — add stages or speed for a better hydraulic efficiency.` });
    if (v.dpValve > 0) W.push({ level: 'info', msg: `Throttling ${v.dpValve} bar on the HP discharge costs ${fmt(R.throttlePenalty, 3)} kW (${fmt(R.throttlePenalty / sel.Qp, 2)} kWh/m³).` });
    if (prof.infeasible) W.push({ level: 'warn', msg: `${prof.infeasible} profile row(s) need more than 105 % speed on the HP pump — increase the head margin.` });
    if (R.sec <= R.minSEC) W.push({ level: 'bad', msg: 'Net specific energy is below the thermodynamic minimum — the entered pressure is inconsistent with the salinity and recovery.' });
    if (!W.some((w) => w.level === 'bad')) W.unshift({ level: 'info', msg: 'No hard limit is violated at the design point.' });

    const best = options.reduce((b, o) => (o.sec < b.sec ? o : b)), none = options.find((o) => o.type === 'none');
    const Qs = linspace(0, Math.max(R.Qop, R.Qint) * 1.5, 41), recs = linspace(Math.max(10, v.recovery - 25), Math.min(92, v.recovery + 25), 11);
    const piB = (rr) => osmoticPressure(v.T, Math.min(150, v.S / (1 - rr))) / 1e5;
    const sweep = recs.map((rc) => { const Pm = Math.max(v.dpMem + v.Pbrine + 1, v.Pmem + piB(rc / 100) - piB(R.r)); return ['px', 'turbo', 'pelton', 'none'].map((t) => (erdCase(t, { ...P, r: rc / 100, Pmem: Pm, avgCF: 0.5 * (1 + 1 / (1 - rc / 100)) }).net + R.intakeElec) / (v.Qf * rc / 100)); });
    const loads = linspace(0.1, 1.2, 23), spds = [1, 0.9, 0.8, 0.7], qh = linspace(0, sel.hpFlow * 1.45, 30);
    const px = sel.px, methods = [['Colebrook–White', 'colebrook'], ['Swamee–Jain', 'swamee'], ['Hazen–Williams', 'hazen']];
    const hpElecTot = sel.hpElec + R.throttlePenalty;
    const out = { hpPumpPower: hpElecTot, boosterPower: sel.boostElec, intakePower: R.intakeElec, erdRecovered: sel.recovered, netPower: R.net, sec: R.sec, pumpEfficiency: P.etaHP, erdEfficiency: sel.erdEff,
      erdType: sel.type, hpFlow: sel.hpFlow, hpHead: R.Hhp, intakeFlow: R.Qop, intakeHead: R.Hop, npsha: R.npsha, npshr: R.npshr, surgeMaxBar: pMax, surgeMinBar: pMin, annualEnergy: annualProfile, lifecycleEnergyCost: lc.pvCost, secRO: sel.sec, minSEC: R.minSEC, membraneFeedSalinityRise: sel.type === 'px' ? px.salinityRise : 0 };

    return {
      summary: `${ERD_NAMES[sel.type]}: net ${fmt(R.net, 4)} kW for ${fmt(sel.Qp, 4)} m³/h of permeate = ${fmt(R.sec, 3)} kWh/m³ (HP pump ${fmt(hpElecTot, 4)} kW, circulation ${fmt(sel.boostElec, 3)} kW, supply ${fmt(R.intakeElec, 3)} kW; ${fmt(sel.recovered, 4)} kW recovered). Supply pumps run at ${fmt(bepPct, 3)} % of best-efficiency flow with an NPSH ratio of ${fmt(npshRatio, 3)}.`,
      warnings: W,
      kpis: [
        { label: 'Net specific energy', value: R.sec, unit: 'kWh/m³', status: R.sec <= R.minSEC ? 'bad' : 'ok', help: 'HP pump + circulation pump + supply pump, per m³ of permeate' },
        { label: 'Net electrical power', value: R.net, unit: 'kW' },
        { label: 'HP pump power', value: hpElecTot, unit: 'kW', help: 'Electrical input including motor and drive losses' },
        { label: 'Recovered by ERD', value: sel.recovered, unit: 'kW', help: 'Hydraulic power returned to the feed (or to the pump shaft)' },
        { label: 'ERD efficiency', value: 100 * sel.erdEff, unit: '%' },
        { label: 'Circulation pump', value: sel.boostElec, unit: 'kW' },
        { label: 'Supply pump power', value: R.intakeElec, unit: 'kW' },
        { label: 'Supply operating point', value: `${fmt(R.Qop, 4)} m³/h @ ${fmt(R.Hop, 3)} m` },
        { label: 'Flow relative to BEP', value: bepPct, unit: '%', status: R.state !== 'ok' ? 'bad' : bepPct < v.porLo || bepPct > v.porHi ? 'warn' : 'ok' },
        { label: 'NPSH available / required', value: npshRatio, unit: '–', status: npshRatio < 1 ? 'bad' : npshRatio < v.npshMargin ? 'warn' : 'ok', help: `${fmt(R.npsha, 3)} m available, ${fmt(R.npshr, 3)} m required` },
        { label: 'HP pump head', value: R.Hhp, unit: 'm' },
        { label: 'HP specific speed per stage', value: R.nqHP, unit: 'nq', status: R.nqHP < 12 ? 'warn' : 'ok', help: 'nq = N·√Q / H^0.75 (rpm, m³/s, m)' },
        { label: 'Thermodynamic minimum', value: R.minSEC, unit: 'kWh/m³', help: 'Reversible work of separation at this recovery' },
        { label: 'Membrane-feed salinity rise', value: sel.type === 'px' ? 100 * px.salinityRise : 0, unit: '%', status: sel.type === 'px' && px.salinityRise > 0.035 ? 'warn' : 'ok' },
        { label: 'Peak surge pressure', value: pMax, unit: 'bar g', status: pMax > v.pnRating ? 'bad' : 'ok' },
        { label: 'Minimum transient pressure', value: pMin, unit: 'bar g', status: pMin < pVapG ? 'bad' : pMin < -0.5 ? 'warn' : 'ok' },
        { label: 'Annual energy', value: annualProfile / 1e6, unit: 'GWh/y' },
        { label: 'Lifecycle energy cost (PV)', value: lc.pvCost / 1e6, unit: 'M$' },
      ],
      recommendations: [
        best.type !== sel.type ? `${ERD_NAMES[best.type]} gives the lowest net energy here (${fmt(best.sec + R.intakeElec / best.Qp, 3)} kWh/m³ against ${fmt(R.sec, 3)}).` : null,
        sel.type === 'none' ? 'The brine valve destroys ' + fmt(sel.valveLoss, 4) + ' kW — fit an energy-recovery device.' : null,
        prof.rows.length && prof.eT > prof.eV * 1.01 ? `Speed control saves ${fmt((prof.eT - prof.eV) / 1e6, 3)} GWh/y (${fmt((100 * (prof.eT - prof.eV)) / prof.eT, 3)} %) on the HP pump compared with throttling over this profile.` : null,
        R.state === 'ok' && bepPct > v.porHi ? 'Supply pump is oversized in head for this system: reduce speed or trim the impeller to move back toward the best-efficiency point.' : null,
        R.state === 'ok' && bepPct < v.porLo ? 'Supply pump is throttled back by the system: run fewer pumps in parallel or select a smaller pump.' : null,
        Math.abs(R.eulerStages - R.stages) > 1.5 ? `Impeller geometry implies about ${fmt(R.eulerStage, 3)} m per stage, i.e. ${fmt(R.eulerStages, 2)} stages for this duty rather than ${R.stages} — review diameter, speed or stage count.` : null,
        wh.joukowsky > 0 && v.whCase === 'valve' && v.tClose < (2 * v.Lint) / a ? `Closure is faster than the pipe period 2L/a = ${fmt((2 * v.Lint) / a, 3)} s, so the full Joukowsky rise of ${fmt(joukBar, 3)} bar develops — close more slowly.` : null,
        'Send the net power and specific energy to suite 13 (Economics); use suite 11 (Optimisation) to trade recovery against pumping energy.',
      ].filter(Boolean),
      plots: [
        { type: 'line', title: 'Supply pump and system curves', xlabel: 'Flow (m³/h)', ylabel: 'Head (m)', zeroY: true,
          series: [{ name: 'System (Darcy–Weisbach)', x: Qs, y: Qs.map((q) => R.Hsys(q)) }, { name: 'System (Hazen–Williams)', x: Qs, y: Qs.map((q) => R.Hsys(q, 'hazen')), dash: true },
            { name: `${R.nPar} pump(s) at ${fmt(100 * R.sp, 3)} % speed`, x: Qs, y: Qs.map((q) => Math.max(0, R.Hpump(q))) }, { name: 'Same at rated speed', x: Qs, y: Qs.map((q) => Math.max(0, R.Hpump(q, 1))), dash: true },
            { name: 'Operating point', x: [R.Qop], y: [R.Hop], mode: 'points' }], vlines: [{ x: R.Qint, label: 'required flow' }] },
        { type: 'line', title: 'Supply pump efficiency and NPSH (per pump)', xlabel: 'Flow per pump (m³/h)', ylabel: 'Efficiency (%) · NPSH (m)',
          series: (() => { const x = linspace(0.1 * curve.Qb * R.sp, 1.5 * curve.Qb * R.sp, 29); return [{ name: 'Efficiency (%)', x, y: x.map((q) => 100 * curve.eta(q / R.sp) * R.deg) }, { name: 'NPSH required (m)', x, y: x.map((q) => R.npshr * (0.6 + 0.4 * (q / (curve.Qb * R.sp)) ** 2) / (0.6 + 0.4 * R.bepRatio ** 2)) }, { name: 'NPSH available (m)', x, y: x.map((q) => (R.patm - R.pv) / (rho * G) + v.zSuction - pipeLoss(q / 3600, { L: v.Lsuc, D: v.Dsuc, eps: v.epsInt / 1000, rho, mu: R.mu, K: v.Ksuc }).total) }]; })(),
          vlines: [{ x: R.q1, label: 'operating' }, { x: curve.Qb * R.sp, label: 'BEP' }] },
        { type: 'bar', title: 'System head breakdown at the operating point', ylabel: 'm', categories: ['Static lift', 'Pipe friction', 'Minor losses', 'Pretreatment', 'Delivery pressure'], series: [{ name: 'Head (m)', values: [parts.stat, parts.hf, parts.hm, parts.pre, parts.del] }] },
        { type: 'bar', title: 'Energy-recovery options: electrical power', ylabel: 'kW', stacked: true, categories: options.map((o) => ERD_NAMES[o.type]),
          series: [{ name: 'HP pump', values: options.map((o) => o.hpElec) }, { name: 'Circulation pump', values: options.map((o) => o.boostElec) }, { name: 'Supply pump', values: options.map(() => R.intakeElec) }] },
        { type: 'bar', title: 'Energy flow of the selected configuration', ylabel: 'kW', categories: ['Hydraulic duty at membranes', 'Recovered by ERD', 'HP pump electrical', 'Circulation pump', 'Supply pump', 'Throttling', 'Pump/motor/drive losses', 'Residual in brine'],
          series: [{ name: 'kW', values: [sel.hydraulicGross, -sel.recovered, sel.hpElec, sel.boostElec, R.intakeElec, R.throttlePenalty, sel.hpElec - kW(sel.hpFlow, sel.hpDp) + (sel.type === 'pelton' || sel.type === 'hprt' ? sel.recovered : 0) + sel.boostElec - kW(sel.boostFlow, sel.boostDp) + R.intakeElec - R.intakeHyd, sel.residual] }],
          note: 'Positive bars are demands or losses, the negative bar is power returned by the energy-recovery device.' },
        { type: 'line', title: 'Net specific energy versus recovery', xlabel: 'Recovery (%)', ylabel: 'kWh/m³', series: ['Pressure exchanger', 'Turbocharger', 'Pelton turbine', 'No recovery'].map((name, k) => ({ name, x: recs, y: sweep.map((s) => s[k]) })).concat([{ name: 'Thermodynamic minimum', x: recs, y: recs.map((rc) => (R.piF / 36) * (-Math.log(1 - rc / 100) / (rc / 100))), dash: true }]),
          note: 'Feed pressure is shifted with the change of brine osmotic pressure; supply pumping is included.' },
        { type: 'line', title: 'HP pump curves under speed control with the operating profile', xlabel: 'Flow (m³/h)', ylabel: 'Head (m)',
          series: [...spds.map((s) => ({ name: `${100 * s} % speed`, x: qh, y: qh.map((q) => Math.max(0, s * s * R.hpCurve.H(q / s))) })), { name: 'Profile duty points', x: prof.rows.map((o) => o.Q), y: prof.rows.map((o) => o.Hreq), mode: 'points' }] },
        { type: 'line', title: 'HP pump power over the profile: speed control versus throttling', xlabel: 'Hour', ylabel: 'kW', zeroY: true,
          series: [{ name: 'Variable speed', x: prof.rows.map((o) => o.h), y: prof.rows.map((o) => o.elecV), mode: 'step' }, { name: 'Throttled at fixed speed', x: prof.rows.map((o) => o.h), y: prof.rows.map((o) => o.elecT), mode: 'step' }] },
        { type: 'line', title: 'Motor and drive efficiency versus load', xlabel: 'Load (% of rated)', ylabel: '%', ymin: 80, ymax: 100,
          series: [{ name: 'Motor', x: loads.map((l) => 100 * l), y: loads.map((l) => 100 * motorEff(l, v.etaMotor / 100)) }, { name: 'Drive', x: loads.map((l) => 100 * l), y: loads.map((l) => 100 * vfdEff(l, v.etaVfd / 100)) }, { name: 'Motor × drive', x: loads.map((l) => 100 * l), y: loads.map((l) => 100 * motorEff(l, v.etaMotor / 100) * vfdEff(l, v.etaVfd / 100)) }] },
        { type: 'line', title: `Water-hammer pressure envelope — ${v.whCase === 'trip' ? 'pump trip' : 'valve closure'}`, xlabel: 'Distance along the line (m)', ylabel: 'Pressure (bar g)',
          series: [{ name: 'Maximum', x: wh.x, y: pMaxArr }, { name: 'Steady', x: wh.x, y: p0Arr, dash: true }, { name: 'Minimum', x: wh.x, y: pMinArr }], hlines: [{ y: pVapG, label: 'vapour pressure' }, { y: v.pnRating, label: 'pipe rating' }] },
        { type: 'line', title: `Transient at the ${v.whCase === 'trip' ? 'pump discharge' : 'valve'}`, xlabel: 'Time (s)', ylabel: 'Pressure (bar g) · flow (% of initial)',
          series: (() => { const k = Math.max(1, Math.ceil(wh.t.length / 600)), ix = wh.t.map((_, j) => j).filter((j) => j % k === 0); return [{ name: 'Pressure (bar g)', x: ix.map((j) => wh.t[j]), y: ix.map((j) => toBar(wh.Hnode[j], wh.node)) }, { name: 'Flow ÷ 10 (% of initial)', x: ix.map((j) => wh.t[j]), y: ix.map((j) => (10 * wh.Qnode[j]) / Q0) }]; })() },
      ],
      tables: [
        { title: 'Hydraulic network at the operating point', columns: ['Item', 'Value', 'Unit'], rows: [
          ['Required intake flow', R.Qint, 'm³/h'], ['Delivered flow', R.Qop, 'm³/h'], ['Velocity', parts.v, 'm/s'], ['Reynolds number', parts.Re, '–'], ['Relative roughness ε/D', v.epsInt / 1000 / v.Dint, '–'], ['Darcy friction factor', parts.f, '–'],
          ['Σ K (fittings)', R.Ktot, '–'], ['Static lift', parts.stat, 'm'], ['Pipe friction', parts.hf, 'm'], ['Minor losses', parts.hm, 'm'], ['Pretreatment loss', parts.pre, 'm'], ['Delivery pressure head', parts.del, 'm'], ['Total system head', R.Hop, 'm'],
          ['Atmospheric pressure at site', R.patm / 1e5, 'bar a'], ['Vapour pressure', R.pv / 1e5, 'bar a'], ['Suction-line loss', R.suc.total, 'm'], ['NPSH available', R.npsha, 'm'], ['NPSH required', R.npshr, 'm']] },
        { title: 'Friction-model comparison (pipe friction at the operating flow)', columns: ['Model', 'Friction factor', 'Friction head (m)', 'Difference from Colebrook (%)'],
          rows: (() => { const b = pipeLoss(Q0, { ...R.pipe, method: 'colebrook' }); return methods.map(([name, m]) => { const p = pipeLoss(Q0, { ...R.pipe, method: m }); return [name, m === 'hazen' ? (p.hf * 2 * G * v.Dint) / (v.Lint * Math.max(p.v * p.v, 1e-12)) : p.f, p.hf, b.hf > 0 ? (100 * (p.hf - b.hf)) / b.hf : 0]; }); })() },
        { title: 'Pump data', columns: ['Quantity', 'Supply pump', 'HP pump'], rows: [
          ['Flow per pump (m³/h)', R.q1, sel.hpFlow], ['Head (m)', R.Hop / R.nSer, R.Hhp], ['Differential pressure (bar)', (rho * G * R.Hop) / 1e5, sel.hpDp], ['Efficiency at duty (%)', 100 * R.etaOp, 100 * P.etaHP], ['Hydraulic power (kW)', R.intakeHyd, kW(sel.hpFlow, sel.hpDp)],
          ['Shaft power (kW)', R.intakeShaft, sel.hpShaft], ['Motor × drive efficiency (%)', 100 * R.etaMi * R.etaVi, 100 * P.etaMotor * P.etaVfd], ['Electrical power (kW)', R.intakeElec, sel.hpElec], ['Speed (rpm)', v.rpm * (v.speedPct / 100), v.rpmHP],
          ['Specific speed nq (per stage)', R.nq, R.nqHP], ['Specific speed, US units', R.nq * 51.65, R.nqHP * 51.65], ['Suggested pump type', pumpType(R.nq), pumpType(R.nqHP)], ['Best-efficiency flow (m³/h)', curve.Qb * R.sp, R.hpCurve.Qb], ['Shut-off head (m)', R.Hpump(0) / R.nSer, R.hpCurve.shutoff],
          ['Curve source', curve.src, 'synthesised at duty'], ['Euler head per stage (m)', null, R.euler], ['Tip speed u₂ (m/s)', null, R.u2], ['Slip factor', null, R.slip], ['Stages implied by geometry', null, R.eulerStages]] },
        { title: 'Energy-recovery options', columns: ['Device', 'HP pump flow (m³/h)', 'HP pump Δp (bar)', 'HP pump (kW)', 'Circulation pump (kW)', 'Recovered (kW)', 'Device efficiency (%)', 'Net incl. supply (kW)', 'Net SEC (kWh/m³)', 'Saving vs none (%)', 'Lifecycle energy cost (M$)'],
          rows: options.map((o, k) => [ERD_NAMES[o.type], o.hpFlow, o.hpDp, o.hpElec, o.boostElec, o.recovered, 100 * o.erdEff, o.net + R.intakeElec, (o.net + R.intakeElec) / o.Qp, (100 * (none.net - o.net)) / (none.net + R.intakeElec), lcOpt[k].pvCost / 1e6]),
          note: 'Design point, constant efficiencies. The pressure-exchanger case includes lubrication leakage, mixing and the circulation pump.' },
        { title: 'Pressure-exchanger streams', columns: ['Stream', 'Flow (m³/h)', 'Pressure (bar g)', 'Salinity (g/kg)', 'Hydraulic power (kW)'], rows: (() => { const o = options[0], x = o.px; return [
          ['HP brine in', x.Qc, o.Pc, x.Sb, kW(x.Qc, o.Pc)], ['HP feed out', x.QhpOut, o.PhpOut, x.Shp, kW(x.QhpOut, o.PhpOut)], ['LP feed in', x.QlpIn, v.Ps, v.S, kW(x.QlpIn, v.Ps)], ['LP brine out', x.QlpOut, o.PlpOut, x.Slp, kW(x.QlpOut, o.PlpOut)],
          ['HP pump discharge', x.Qhp, o.Pmem, v.S, kW(x.Qhp, o.Pmem)], ['Membrane feed (mixed)', v.Qf, o.Pmem, x.Sm, kW(v.Qf, o.Pmem)], ['Permeate', x.Qp, 0, x.Sp, 0], ['Lubrication flow', x.L, null, null, null]]; })(),
          note: `Volumetric mixing ${fmt(100 * options[0].px.M, 3)} %; membrane-feed salinity is ${fmt(100 * options[0].px.salinityRise, 3)} % above the raw feed, worth about ${fmt((osmoticPressure(v.T, Math.min(150, v.S * (1 + options[0].px.salinityRise))) - osmoticPressure(v.T, v.S)) / 1e5, 2)} bar of feed osmotic pressure.` },
        { title: 'Variable operation of the HP pump', columns: ['From hour', 'Duration (h)', 'Flow (m³/h)', 'Required head (m)', 'Speed (%)', 'Pump efficiency (%)', 'Variable-speed power (kW)', 'Head throttled at fixed speed (m)', 'Throttled power (kW)', 'Saving (%)', 'Feasible'],
          rows: prof.rows.map((o) => [o.h, o.dur, o.Q, o.Hreq, o.speed, 100 * o.etaV, o.elecV, o.throttled, o.elecT, (100 * (o.elecT - o.elecV)) / o.elecT, o.feasV ? (o.feasT ? 'both' : 'speed control only') : 'no']),
          note: `Annual HP-pump energy: ${fmt(prof.eV / 1e6, 4)} GWh with speed control, ${fmt(prof.eT / 1e6, 4)} GWh throttled; auxiliaries add ${fmt(prof.eO / 1e6, 3)} GWh. Lifecycle energy ${fmt(lc.energy / 1e6, 4)} GWh over ${n} years.` },
      ],
      balances: [
        { name: 'Pressure exchanger volume (m³/h)', in: options[0].px.Qc + options[0].px.QlpIn, out: options[0].px.QhpOut + options[0].px.QlpOut },
        { name: 'Pressure exchanger salt (flow × salinity)', in: options[0].px.Qc * options[0].px.Sb + options[0].px.QlpIn * v.S, out: options[0].px.QhpOut * options[0].px.Shp + options[0].px.QlpOut * options[0].px.Slp },
        { name: 'System salt, feed vs permeate + brine', in: options[0].px.Qsys * v.S, out: options[0].px.Qp * options[0].px.Sp + options[0].px.QlpOut * options[0].px.Slp },
        { name: 'Supply pump head vs system head (m)', in: R.Hop, out: R.Qop > 0 ? R.Hsys(R.Qop) : R.Hop },
        { name: 'Membrane feed volume (m³/h)', in: v.Qf, out: sel.Qp + sel.Qc },
      ],
      outputs: out,
    };
  },

  mesh: { name: 'Water-hammer reaches (Δt = Δx / a)', keys: ['nReach'], min: 4, note: 'The number of pipe reaches is refined; the time step follows from the Courant condition.',
    metrics: [{ label: 'Peak surge pressure', unit: 'bar g', get: (r) => r.outputs.surgeMaxBar }, { label: 'Minimum transient pressure', unit: 'bar g', get: (r) => r.outputs.surgeMinBar }] },

  calibration: {
    note: 'Fit leakage, pump efficiencies and the hydraulic-resistance multiplier to plant readings. Each row is one steady operating point: membrane feed flow, feed pressure and recovery set the condition; HP-pump flow, HP-pump and circulation-pump electrical power and the supply-system head at the delivered flow are the measurements. Vary recovery and flow between rows so that leakage and efficiency can be told apart.',
    params: [{ key: 'pxLeak', label: 'Lubrication flow (% of HP brine)', lo: 0, hi: 8 }, { key: 'etaHP', label: 'HP pump efficiency (%)', lo: 50, hi: 93 }, { key: 'etaBooster', label: 'Circulation pump efficiency (%)', lo: 40, hi: 92 }, { key: 'kFric', label: 'Hydraulic-resistance multiplier', lo: 0.3, hi: 5 }],
    columns: [{ key: 'Qf', label: 'Membrane feed flow', unit: 'm³/h' }, { key: 'Pmem', label: 'Feed pressure', unit: 'bar' }, { key: 'recovery', label: 'Recovery', unit: '%' }, { key: 'Qhp', label: 'HP pump flow', unit: 'm³/h' }, { key: 'hpKW', label: 'HP pump power', unit: 'kW' }, { key: 'boostKW', label: 'Circulation pump power', unit: 'kW' }, { key: 'Hsys', label: 'Supply system head', unit: 'm' }],
    targets: [{ key: 'Qhp', label: 'HP pump flow', unit: 'm³/h' }, { key: 'hpKW', label: 'HP pump power', unit: 'kW' }, { key: 'boostKW', label: 'Circulation pump power', unit: 'kW' }, { key: 'Hsys', label: 'Supply system head', unit: 'm' }],
    model(v) {
      const rho = density(v.T, v.S), mu = viscosity(v.T, v.S), r = clamp(v.recovery / 100, 0.02, 0.97), eD = motorEff(0.9, v.etaMotor / 100) * (v.hpDrive === 'vfd' ? vfdEff(0.9, v.etaVfd / 100) : 1);
      const o = erdCase('px', { Qf: v.Qf, r, S: v.S, T: v.T, rej: v.rejection / 100, leak: v.pxLeak / 100, of: v.pxOverflush / 100, mix0: v.pxMix / 100, applyMix: false, avgCF: 1, Pmem: v.Pmem, dpMem: v.dpMem, Ps: v.Ps, Pbrine: v.Pbrine, dpHP: v.pxDpHP, dpLP: v.pxDpLP, dpCirc: v.dpCirc, etaHP: v.etaHP / 100, etaBooster: v.etaBooster / 100, etaMotor: eD, etaVfd: 1 });
      const Q = o.px.Qsys / (1 - v.preLoss / 100), pl = pipeLoss(Q / 3600, { L: v.Lint, D: v.Dint, eps: v.epsInt / 1000, rho, mu, K: sumK(v.fittings), method: v.friction, C: v.hwC });
      return { Qhp: o.hpFlow, hpKW: o.hpElec, boostKW: o.boostElec, Hsys: v.zStatic + v.kFric * pl.total + ((v.dpPre + v.Ps) * 1e5) / (rho * G) };
    },
    get sample() { return (this._s ||= synth(5, [[1000, 60, 45], [1000, 58, 42], [900, 57, 45], [1100, 62, 45], [1000, 63, 50], [800, 54, 40], [1050, 61, 47], [950, 59, 38]])); },
    get validationSample() { return (this._v ||= synth(23, [[1000, 59, 44], [850, 55.5, 41], [1080, 62.5, 48], [920, 58, 46], [1000, 61, 36], [1120, 60, 43]])); },
  },

  verify() {
    const d = defaultsOf(suite), C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    const R = simulatePumps(d), rho = R.rho;
    add('Laminar friction factor f = 64/Re', 64 / 1200, frictionFactor(1200, 1e-4), 1e-12, 'Hagen–Poiseuille limit at Re = 1200');
    add('Colebrook–White at Re = 1e5, ε/D = 1e-4', 0.0185, frictionFactor(1e5, 1e-4), 1.5e-4, 'Moody-chart value 0.0185');
    const fc = frictionFactor(1e5, 1e-4);
    add('Colebrook residual is zero', 0, 1 / Math.sqrt(fc) + 2 * Math.log10(1e-4 / 3.7 + 2.51 / (1e5 * Math.sqrt(fc))), 1e-9, '1/√f + 2·log₁₀(ε/3.7D + 2.51/(Re√f)) = 0');
    add('Swamee–Jain agrees with Colebrook', 0, Math.abs(frictionFactor(1e5, 1e-4, 'swamee') / fc - 1), 0.02, 'Explicit approximation within 2 %');
    // Bernoulli / mechanical-energy hand calculation: water at 20 °C, 100 m of 0.2 m pipe, 2 m/s, f from Colebrook, 10 m lift
    const rw = density(20, 0), mw = viscosity(20, 0), A = (Math.PI * 0.04) / 4, pl = pipeLoss(2 * A, { L: 100, D: 0.2, eps: 4.5e-5, rho: rw, mu: mw, K: 1.5 });
    const fh = frictionFactor((rw * 2 * 0.2) / mw, 4.5e-5 / 0.2), hand = 10 + (fh * 500 + 1.5) * (4 / (2 * G));
    add('Extended Bernoulli head, hand calculation', hand, 10 + pl.total, 1e-9, 'H = Δz + (f·L/D + ΣK)·v²/2g for 100 m × 0.2 m at 2 m/s');
    add('Operating point lies on both curves', 0, (R.Hpump(R.Qop) - R.Hsys(R.Qop)) / R.Hop, 1e-6, 'Pump head = system head at the solved flow');
    add('Hydraulic power P = ρ·g·Q·H', (rho * G * (R.Qop / 3600) * R.Hop) / 1000, R.intakeHyd, 1e-9, 'Supply pump, kW');
    add('Hydraulic power in pressure form Q·Δp/36', (R.sel.hpFlow / 3600) * R.sel.hpDp * 1e5 / 1000, kW(R.sel.hpFlow, R.sel.hpDp), 1e-9, 'm³/h × bar ÷ 36 = kW');
    // affinity laws on the synthesised curve
    const cv = synthCurve(500, 50, 0.85, 30), s = 0.8, Qa = 420, Ha = cv.H(Qa), Pa = (Qa * Ha) / cv.eta(Qa), Hs = s * s * cv.H((Qa * s) / s), Ps = (Qa * s * Hs) / cv.eta((Qa * s) / s);
    add('Affinity law H ∝ N²', s * s, Hs / Ha, 1e-12, 'Homologous point at 80 % speed (Q ∝ N imposed)');
    add('Affinity law P ∝ N³', s ** 3, Ps / Pa, 1e-12, 'Shaft power at the homologous point');
    const dSlow = simulatePumps({ ...d, curveSrc: 'duty', zStatic: 0, dpPre: 0, Ps: 0.0001, speedPct: 100 }), dFast = simulatePumps({ ...d, curveSrc: 'duty', zStatic: 0, dpPre: 0, Ps: 0.0001, speedPct: 80, friction: 'hazen' }), dRef = simulatePumps({ ...d, curveSrc: 'duty', zStatic: 0, dpPre: 0, Ps: 0.0001, speedPct: 100, friction: 'hazen' });
    add('Flow follows speed on a friction-only system', 0.8, dFast.Qop / dRef.Qop, 0.02, 'No static head, H ∝ Q^1.85: operating flow ≈ ∝ N (exact for H ∝ Q²)');
    add('Zero-static system: head at zero flow is zero', 0, dSlow.Hsys(1e-9), 0.01, 'Limiting case with no lift, no delivery pressure');
    const px = R.options[0].px;
    add('Pressure-exchanger volume balance', 0, (px.Qc + px.QlpIn - px.QhpOut - px.QlpOut) / px.Qc, 1e-12, 'HP in + LP in = HP out + LP out');
    add('Pressure-exchanger salt balance', 0, (px.Qc * px.Sb + px.QlpIn * d.S - px.QhpOut * px.Shp - px.QlpOut * px.Slp) / (px.Qc * px.Sb), 1e-10, 'Σ Q·S in = Σ Q·S out');
    add('System salt balance with pressure exchanger', 0, (px.Qsys * d.S - px.Qp * px.Sp - px.QlpOut * px.Slp) / (px.Qsys * d.S), 1e-9, 'Raw feed salt = permeate salt + discharged brine salt');
    const ideal = erdCase('px', { ...R.P, leak: 0, of: 0, mix0: 0, dpHP: 0, dpLP: 0 });
    add('Ideal pressure exchanger transfers all hydraulic energy', 1, ideal.erdEff, 1e-12, 'No leakage and no differential pressure → efficiency 100 %');
    add('Ideal exchanger: HP pump flow equals permeate flow', ideal.Qp, ideal.hpFlow, 1e-9, 'Limiting case without lubrication flow');
    // Joukowsky versus MOC first peak, frictionless instantaneous closure
    const wh = waterHammer({ L: 1000, D: 0.5, a: 1000, f: 0, Q0: 0.3, Hd: 20, hv0: 1e-6, N: 50, tSim: 1.5, scenario: 'valve', tc: 0 });
    add('MOC first surge peak equals Joukowsky a·Δv/g', (1000 * (0.3 / ((Math.PI * 0.25) / 4))) / G, Math.max(...wh.Hmax) - wh.H0[50], 1e-6, 'Instantaneous closure, frictionless pipe (head rise in m)');
    const wh2 = waterHammer({ L: 1000, D: 0.5, a: 1000, f: 0, Q0: 0.3, Hd: 20, hv0: 1e-6, N: 50, tSim: 8, scenario: 'valve', tc: 0 });
    add('Frictionless surge is symmetric about the static head', 0, (Math.max(...wh2.Hmax) - 20 - (20 - Math.min(...wh2.Hmin))) / wh2.joukowsky, 1e-6, 'Undamped oscillation ±a·v/g with period 4L/a');
    add('Wave speed in a rigid pipe equals √(K/ρ)', Math.sqrt(KBULK / rho), waveSpeed(rho, 0.5, 12, 1e9), 1e-3, 'Limit of infinite wall stiffness');
    add('Net specific energy exceeds the reversible minimum', 1, R.sec > R.minSEC && R.sel.sec > R.minSEC ? 1 : 0, 0, `Second-law check: ${fmt(R.sec, 3)} > ${fmt(R.minSEC, 3)} kWh/m³`);
    add('Every recovery device beats the throttle valve', 1, R.options.slice(0, 4).every((o) => o.net < R.options[4].net) ? 1 : 0, 0, 'Recovered power is positive for all devices');
    add('Standard atmosphere at sea level', 1.01325, atmPressure(0) / 1e5, 1e-9, 'bar');
    return C;
  },
};

/** Synthetic "measured" operating points: the model with different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([Qf, Pmem, recovery]) => {
    const m = suite.calibration.model({ ...d, pxLeak: 2.3, etaHP: 83.5, etaBooster: 76, kFric: 1.45, Qf, Pmem, recovery });
    return { Qf, Pmem, recovery, Qhp: +(m.Qhp * (1 + g.normal(0, 0.003))).toFixed(1), hpKW: +(m.hpKW * (1 + g.normal(0, 0.006))).toFixed(1), boostKW: +(m.boostKW * (1 + g.normal(0, 0.01))).toFixed(2), Hsys: +(m.Hsys * (1 + g.normal(0, 0.004))).toFixed(2) };
  });
}

export default suite;
