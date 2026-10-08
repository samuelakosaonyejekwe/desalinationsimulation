// Suite 6 — Thermal desalination.
// Rigorous effect-by-effect / stage-by-stage steady-state models of multi-effect distillation (forward and
// parallel/cross feed, optional thermo-compressor), multi-stage flash (once-through and brine recirculation)
// and mechanical vapour compression. Every effect/stage closes total-mass, salt and enthalpy balances with
// boiling-point elevation, non-equilibrium allowance, demister/line losses, distillate and brine flashing,
// feed preheaters, a final condenser, published U(T) correlations, exergy accounting and a lumped start-up model.
import { brent, clamp, linspace, sum, rng, fmt, rk4, nelderMead } from '../core/num.js';
import { density, cp, psat, tsat, antoine, latentHeat, bpe, enthalpyLiquid as hL, enthalpyVapour as hV, osmoticPressure, osmoticCoefficient, salinityFromTDS, tdsFromSalinity, KELVIN, R as RGAS } from '../core/props.js';
import { ION_IDS, WATERS, cloneIons, tds, scaleIons } from '../core/water.js';
import roSuite, { simulateRO } from './s01_ro.js';

const CPV = 1884, RW = 461.52, GAMMA = 1.32, RHO_REF = 994; // vapour cp J/kg·K, gas constant of steam, isentropic exponent, product density kg/m³
const K = (T) => T + KELVIN;
const fill = (n, x) => new Array(n).fill(x);
const lmtd = (d1, d2) => (Math.abs(d1 - d2) < 1e-9 ? 0.5 * (d1 + d2) : (d1 - d2) / Math.log(d1 / d2));
/** Density of saturated water vapour (ideal gas), kg/m³. */
export const rhoV = (T) => psat(T) / (RW * K(T));
const dPdT = (T) => (psat(T + 0.05) - psat(T - 0.05)) / 0.1;
/** Temperature of liquid of salinity S with specific enthalpy h (inverse of props.enthalpyLiquid). */
const tFromH = (h, S) => { let T = h / cp(40, S); for (let i = 0; i < 14; i++) { const d = (hL(T, S) - h) / cp(T, S); T -= d; if (Math.abs(d) < 1e-11) break; } return T; };

/** Overall heat-transfer coefficients of El-Dessouky & Ettouney (W/m²·K, T in °C): falling-film evaporator and condenser. */
export const uEvaporator = (T) => 1e3 * (1.9695 + 1.2057e-2 * T - 8.5989e-5 * T * T + 2.5651e-7 * T ** 3);
export const uCondenser = (T) => 1e3 * (1.7194 + 3.2063e-3 * T + 1.5971e-5 * T * T - 1.9918e-7 * T ** 3);
const fouled = (U, o) => 1 / (1 / (U * o.fU) + o.Rf);
/** Error for a design that cannot be built with the present inputs; `kind` selects the search for a feasible neighbour. */
const infeasible = (msg, kind) => Object.assign(new Error(msg), { infeasible: true, kind });

const MOLAL_W = 55.508; // mol of water per kg
/** Moles of dissolved ions per kg of water for sea-salt of salinity S (g/kg). */
export const ionMolality = (S) => (31.843 * S) / (1000 - S);
/** Water activity of the brine: 'ideal' = Raoult's law with the water mole fraction, 'activity' = Raoult's law corrected with the osmotic coefficient (a_w = exp(−φ·m/55.51)). */
export function waterActivity(T, S, basis = 'activity') {
  const m = ionMolality(Math.min(Math.max(S, 0), 400));
  return basis === 'ideal' ? MOLAL_W / (MOLAL_W + m) : Math.exp((-osmoticCoefficient(T, S) * m) / MOLAL_W);
}
/** Vapour pressure above brine from Raoult's law, Pa: p = a_w·p_sat(T). */
export const raoultPressure = (T, S, basis = 'activity') => waterActivity(T, S, basis) * psat(T);
/** Boiling-point elevation from Raoult's law, K: brine at T is in equilibrium with vapour saturated at t_sat(a_w·p_sat(T)). */
export const bpeRaoult = (T, S, basis = 'activity') => (S > 0 ? T - tsat(raoultPressure(T, S, basis)) : 0);
/** Boiling-point elevation on the selected basis (o.bpeModel: 'corr' | 'ideal' | 'activity'), scaled by the user multiplier. */
const bpeOf = (T, X, o) => o.fBPE * (o.bpeModel === 'ideal' || o.bpeModel === 'activity' ? bpeRaoult(T, X, o.bpeModel) : bpe(T, X));

/** Pressure drop of a wire-mesh demister, Pa (El-Dessouky et al. 2000): packing density kg/m³, vapour velocity m/s, wire diameter mm, thickness m. */
export const demisterDrop = (rhoP, V, dw, thick) => 3.88178 * rhoP ** 0.375798 * V ** 0.81317 * dw ** -1.56114147 * thick;

/** Saturation-temperature loss of the vapour across the demister (o.dpDem, Pa) and the vapour lines (o.linePct % of pressure), K. */
export function vapourLoss(Tv, o) {
  const pm = psat(Tv - 0.05), pp = psat(Tv + 0.05);
  return (o.dpDem + (o.linePct / 100) * 0.5 * (pm + pp)) / ((pp - pm) / 0.1);
}

/** Non-equilibrium allowance of a flashing stage, K (Lior correlation as used by El-Dessouky & Ettouney). */
export function nonEquilibriumAllowance(T, dTstage, o) {
  const nea10 = 0.9784 ** T * 15.7378 ** o.neaH * 1.3777 ** (o.neaVb * 3600e-6), half = 0.5 * dTstage + nea10;
  return o.fNEA * (nea10 / half) ** (0.3281 * o.neaL) * half;
}

/** Steam-jet ejector entrainment ratio Ra = motive / entrained (El-Dessouky fit of the Power chart). Pressures in kPa. */
export function entrainmentRatio(Pm, Ps, Pev, Tev) {
  const pcf = 3e-7 * Pm * Pm - 0.0009 * Pm + 1.6101, tcf = 2e-8 * Tev * Tev - 0.0006 * Tev + 1.0047;
  return 0.296 * (Ps ** 1.19 / Pev ** 1.04) * (Pm / Pev) ** 0.015 * (pcf / tcf);
}

/** Brine salinity (g/kg) at 90 % of CaSO₄ solubility in seawater brine at temperature T (El-Dessouky & Ettouney). */
export const caso4Limit = (T) => 0.9 * (457628.5 - 11304.11 * T + 107.5781 * T * T - 0.360747 * T ** 3) / 1000;

/** Condenser-type exchanger: isothermal condensing side at Th, cold stream m (kg/s) heated tin → tout, integrated in nSeg segments. */
export function condenserHX(name, Th, m, S, tin, tout, U, nSeg = 1) {
  const Q = m * (hL(tout, S) - hL(tin, S));
  if (!(Q > 1e-9) || !(Th > tout)) return { name, duty: Math.max(0, Q || 0), area: 0, lmtd: 0, U, ntu: 0, eff: 0, effNtu: 0, tin, tout, Th };
  let A = 0;
  for (let s = 0; s < nSeg; s++) {
    const ta = tin + ((tout - tin) * s) / nSeg, tb = tin + ((tout - tin) * (s + 1)) / nSeg;
    A += (m * (hL(tb, S) - hL(ta, S))) / (U * lmtd(Th - ta, Th - tb));
  }
  const ntu = (U * A * (tout - tin)) / Q;
  return { name, duty: Q, area: A, lmtd: Q / (U * A), U, ntu, eff: (tout - tin) / (Th - tin), effNtu: 1 - Math.exp(-ntu), tin, tout, Th };
}

// ---- multi-effect evaporation (MED, MED-TVC, MVC) ---------------------------------------------------------
/** One effect: heat q (W) boils feed F (enthalpy hf) plus brine Bin cascaded from the previous effect at temperature T. */
function effect(q, F, hf, Bin, Xin, hbin, T, Xf, o, Vg) {
  const m = F + Bin, salt = F * Xf + Bin * Xin, Vmax = m - salt / 280; // brine cannot be concentrated beyond ~280 g/kg
  let V = Vg ?? q / latentHeat(T), B, X, be, Tv, hv, hb; // warm start from the previous pass when available
  for (let k = 0; k < 60; k++) {
    V = clamp(V, -0.5 * m, Vmax);
    B = m - V; X = salt / B; be = bpeOf(T, X, o); Tv = T - be; hv = hV(Tv) + CPV * be; hb = hL(T, X);
    const Vn = clamp((q + F * hf + Bin * hbin - m * hb) / (hv - hb), -0.5 * m, Vmax);
    if (Math.abs(Vn - V) < 1e-12 * m) { V = Vn; break; }
    V = Vn;
  }
  B = m - V; X = salt / B; be = bpeOf(T, X, o); Tv = T - be; hv = hV(Tv) + CPV * be; hb = hL(T, X);
  return { V, B, X, be, Tv, hv, hb, cap: V >= Vmax * (1 - 1e-12) };
}

/**
 * One effect of a parallel/cross-feed train: the feed rate is the one that lets the brine leave at the design salinity Xt, so the
 * salt and enthalpy balances are linear in the feed F and the vapour V (no iteration, no salinity limit to guard).
 */
function effectEven(q, hf, Bin, Xin, hbin, T, Xf, Xt, o) {
  const be = bpeOf(T, Xt, o), Tv = T - be, hv = hV(Tv) + CPV * be, hb = hL(T, Xt), a = 1 - Xf / Xt, b = Bin * (1 - Xin / Xt); // salt: V = a·F + b
  const F = Math.max(0, (q + Bin * hbin - b * hv - (Bin - b) * hb) / (a * hv + (1 - a) * hb - hf)), V = a * F + b, B = F + Bin - V;
  return { F, V, B, X: B > 0 ? (F * Xf + Bin * Xin) / B : Xt, be, Tv, hv, hb, cap: false };
}

/** March once through the train for given heating duty, temperature differences and feed temperatures. */
function medPass(c, s) {
  const N = c.N, o = c.o, r = { T: [], Tv: [], Tc: [], Th: [], V: [], fl: [], Vt: [], hm: [], X: [], B: [], q: [], be: [], dl: [], mh: [], Fi: [], qloss: 0, Ts: s.Ts, cap: -1 };
  let Tcp = s.Ts, Bin = 0, Xin = c.Xf, hbin = 0, Dacc = s.Mev, hD = hL(s.Ts), mprev = 0, hprev = 0;
  for (let i = 0; i < N; i++) {
    const T = Tcp - s.dT[i], qg = i === 0 ? s.Q1 : mprev * (hprev - hL(Tcp)), q = qg * (1 - c.loss);
    r.qloss += qg - q;
    const hf = hL(s.tf[i], c.Xf), e = c.even ? effectEven(q, hf, Bin, Xin, hbin, T, c.Xf, c.Xb, o) : effect(q, c.Fi[i], hf, Bin, Xin, hbin, T, c.Xf, o, s.Vg?.[i]), dl = vapourLoss(e.Tv, o), Tc = e.Tv - dl;
    r.Fi.push(c.even ? e.F : c.Fi[i]);
    let f = 0;
    if (i > 0) { // distillate flash box: condensate of the previous effect's vapour joins the accumulated distillate
      const C = r.Vt[i - 1] - (i - 1 === c.nEnt ? s.Mev : 0), m = Dacc + C, H = Dacc * hD + C * hL(Tcp);
      f = Math.max(0, (H - m * hL(e.Tv)) / (hV(e.Tv) - hL(e.Tv)));
      Dacc = m - f; hD = Dacc > 0 ? (H - f * hV(e.Tv)) / Dacc : hL(e.Tv);
    }
    const Vt = e.V + f, hm = (e.V * e.hv + f * hV(e.Tv)) / Vt;
    r.T.push(T); r.Th.push(Tcp); r.Tv.push(e.Tv); r.Tc.push(Tc); r.V.push(e.V); r.fl.push(f); r.Vt.push(Vt); r.hm.push(hm); r.X.push(e.X); r.B.push(e.B); r.q.push(q); r.be.push(e.be); r.dl.push(dl);
    mprev = Vt - (i === c.nEnt ? s.Mev : 0) - (i < N - 1 ? s.qph[i] / (hm - hL(Tc)) : 0); hprev = hm;
    r.mh.push(mprev); if (e.cap && r.cap < 0) r.cap = i; // vapour passed on as heating steam (to the next effect or the final condenser); first effect at the salt cap
    Tcp = Tc; Bin = e.B; Xin = e.X; hbin = e.hb;
  }
  const L = N - 1, mc = r.Vt[L] - (L === c.nEnt ? s.Mev : 0);
  r.vent = c.vent * mc; r.cond = mc - r.vent; r.Dacc = Dacc; r.hD = hD; r.Md = Dacc + r.cond;
  r.Qc = r.cond * (r.hm[L] - hL(r.Tc[L]));
  r.U = r.T.map((T) => fouled(uEvaporator(T), o));
  r.A = r.q.map((q, i) => q / (r.U[i] * s.dT[i]));
  return r;
}

/** Vapour compressor between the last effect and the first-effect tube side. */
function compressor(r, Ts, c) {
  const L = c.N - 1, P1 = psat(r.Tc[L]), P2 = psat(Ts), T1 = r.Tc[L] + Math.max(0, r.hm[L] - hV(r.Tc[L])) / CPV;
  const wIs = (GAMMA / (GAMMA - 1)) * RW * K(T1) * ((P2 / P1) ** ((GAMMA - 1) / GAMMA) - 1), w = wIs / c.etaIs;
  return { P1, P2, T1, T2: T1 + w / CPV, w, wIs, ratio: P2 / P1, h2: r.hm[L] + w };
}

/**
 * Converged multi-effect train. c.proc: 'med' | 'tvc' | 'mvc'; c.areaMode: 'equalDT' | 'equalArea' | 'rating' (areas c.Ades fixed).
 * Flows in kg/s, temperatures in °C, salinities in g/kg, heat in W.
 */
export function solveMED(c) {
  const N = c.N, mvc = c.proc === 'mvc', tvc = c.proc === 'tvc', forward = c.arr === 'forward' && !mvc;
  let F = c.F;
  const cc = { ...c, even: !forward && c.Xb > c.Xf, Fi: forward ? [F, ...fill(N - 1, 0)] : fill(N, F / N), nEnt: tvc ? clamp(c.nEnt, 0, N - 1) : -1 }, tol = c.tol || 1e-10, maxIt = c.maxIt || 250;
  let dT = c.dT0 ? [...c.dT0] : fill(N, mvc ? c.dTm : Math.max(0.5, (c.Ts - c.Tn) / N - 1.1));
  let Ts = mvc ? c.T1 + dT[0] : c.Ts, tf = fill(N, mvc ? c.T1 - 4 : c.Tn - 6), qph = fill(N, 0), Mev = 0, Q1 = 0, Qaux = 0, TfM = c.T1 - 4;
  let r = null, ej = null, comp = null, pre = { Mcw: 0, Tf0: tf[0], hx: [] }, conv = false, it = 0;
  // closure variable x: heating duty (MED), motive steam (TVC) or feed-preheat enthalpy / auxiliary heat (MVC)
  let x = mvc ? 0 : tvc ? c.MdT / (1.3 * N) : (c.MdT * latentHeat(Ts)) / (0.85 * N);
  let fz = { Tev: Ts - ((cc.nEnt + 1) * (Ts - c.Tn)) / N, hev: 0, ratio: 1 / N, h2: hV(c.T1 - 1) + 3.4e4, TfMax: c.T1 - c.ttdCond - (N - 1) * (c.dTm + 1) };
  fz.hev = hV(fz.Tev);
  const wKey = `${c.proc}|${N}|${c.arr}|${c.preheat}|${c.areaMode}|${cc.nEnt}`, ws = c.warm && c.warm.key === wKey && !c.dT0 ? c.warm : null;
  if (ws) { dT = [...ws.dT]; tf = [...ws.tf]; qph = [...ws.qph]; cc.Fi = ws.Fi.map((q) => (q * F) / ws.F); x = mvc ? ws.x : (ws.x * c.MdT) / ws.MdT; fz = { ...ws.fz }; if (mvc) Ts = c.T1 + dT[0]; } // warm start from the previous converged design of the same train
  const pass = (xv) => {
    if (tvc) { // ejector: suction from effect nEnt, discharge at the first-effect condensing pressure
      const Tm = tsat(c.Pm), Ra = entrainmentRatio(c.Pm / 1e3, psat(Ts) / 1e3, psat(fz.Tev) / 1e3, fz.Tev);
      Mev = xv / Ra;
      const hd = (xv * hV(Tm) + Mev * fz.hev) / (xv + Mev);
      Q1 = (xv + Mev) * (hd - hL(Ts));
      ej = { Ra, Mm: xv, Mev, Tm, Tev: fz.Tev, hd, Td: Ts + Math.max(0, hd - hV(Ts)) / CPV, Cr: psat(Ts) / psat(fz.Tev), Er: c.Pm / psat(fz.Tev), Qin: xv * (hV(Tm) - hL(Ts)) };
    } else if (mvc) {
      Qaux = Math.max(0, xv); TfM = Math.max(c.Tsw, fz.TfMax + Math.min(0, xv) / (F * cp(c.T1, c.Xf))); tf = fill(N, TfM);
      Q1 = c.MdT * fz.ratio * (fz.h2 - hL(Ts)) + Qaux;
    } else Q1 = xv;
    return (lastPass = medPass(cc, { Ts, dT, tf, qph, Q1, Mev, Vg: lastPass?.V }));
  };
  let slope = 0, lastPass = null, errOuter = ws ? 1e-3 : 1, floored = 0;
  for (; it < maxIt; it++) {
    // inner secant: distillate target at frozen temperatures (production is almost linear in x)
    let x0 = x, r0 = pass(x0), x1;
    r = r0;
    const tolIn = clamp(1e-3 * errOuter, 1e-12, 1e-6) * c.MdT; // loose while the outer loop is far from converged
    let lo = null, hi = null; // closest points below and above the target: production rises with x, so they bracket the root
    const mark = (xv, rv) => { const g = rv.Md - c.MdT; if (g < 0) { if (!lo || xv > lo.x) lo = { x: xv, g }; } else if (g > 0 && (!hi || xv < hi.x)) hi = { x: xv, g }; };
    mark(x0, r0);
    for (let k = 0; k < (lo && hi ? 60 : 8) && Math.abs(r.Md - c.MdT) > tolIn; k++) {
      x1 = slope > 0 ? x0 + (c.MdT - r0.Md) / slope : mvc ? x0 + 0.01 * c.MdT * latentHeat(Ts) : x0 * clamp(c.MdT / r0.Md || 1.5, 0.3, 3);
      if (!mvc && !(x1 > 0)) x1 = 0.5 * x0;
      if (lo && hi && (!(x1 > lo.x && x1 < hi.x) || k >= 6)) x1 = lo.x + (hi.x - lo.x) * (k % 3 === 2 ? 0.5 : clamp(-lo.g / (hi.g - lo.g), 0.05, 0.95)); // the secant left the bracket (a salinity limit flattens the curve): false position with bisection steps
      if (x1 === x0) break;
      const r1 = pass(x1), sl = (r1.Md - r0.Md) / (x1 - x0);
      if (Number.isFinite(sl) && sl > 0) slope = sl;
      x0 = x1; r0 = r1; r = r1; mark(x1, r1);
    }
    let err = Math.abs(r.Md - c.MdT) / c.MdT + Math.abs(x0 - x) / (Math.abs(x0) + (mvc ? c.MdT * 1e5 : 1e-30));
    x = x0;
    if (cc.even) { // parallel / cross feed: every effect takes the feed that concentrates it to the design brine salinity
      const Fs = sum(r.Fi);
      err = Math.max(err, ...r.Fi.map((q, i) => Math.abs(q - cc.Fi[i]) / Fs));
      cc.Fi = [...r.Fi]; F = cc.F = Fs;
    } else if (c.Xb > c.Xf) { // forward feed: the vented vapour is boiled as well, so the feed follows the gross vapour and the blow-down leaves at the design salinity
      const Fg = ((c.MdT + Math.max(0, r.vent)) * c.Xb) / (c.Xb - c.Xf);
      err = Math.max(err, Math.abs(Fg - F) / F);
      cc.Fi = cc.Fi.map((q) => (q * Fg) / F); F = cc.F = Fg;
    }
    if (c.areaMode === 'equalArea' && N > 1) {
      const Am = sum(r.A) / N, S = sum(dT);
      dT = dT.map((d, i) => d * clamp(r.A[i] / Am, 0.3, 3) ** 0.7);
      const S2 = sum(dT); dT = dT.map((d) => (d * S) / S2);
      err = Math.max(err, ...r.A.map((a) => Math.abs(a / Am - 1)));
    } else if (c.areaMode === 'flux' && N > 1) { // prescribed equal heat flux q″ = U·ΔT in every effect: ΔT_i ∝ 1/U_i
      const S = sum(dT), w = r.U.map((u) => 1 / u), ws_ = sum(w), fm = sum(r.U.map((u, i) => u * dT[i])) / N;
      err = Math.max(err, ...r.U.map((u, i) => Math.abs((u * dT[i]) / fm - 1)));
      dT = dT.map((d, i) => 0.3 * d + 0.7 * ((S * w[i]) / ws_));
    } else if (c.areaMode === 'rating') {
      dT = dT.map((d, i) => clamp(d * clamp(r.A[i] / c.Ades[i], 0.3, 3) ** 0.7, 0.05, 40));
      err = Math.max(err, ...r.A.map((a, i) => Math.abs(a / c.Ades[i] - 1)));
    }
    if (mvc) {
      Ts = c.T1 + dT[0];
      comp = compressor(r, Ts, cc);
      const Thot = (r.Dacc * r.Tv[N - 1] + r.cond * Ts + r.B[N - 1] * r.T[N - 1]) / (r.Md + r.B[N - 1]); // flow-weighted temperature of product and brine entering the feed preheater
      const nf = { ratio: r.cond / r.Md, h2: comp.h2, TfMax: Thot - c.ttdCond };
      err = Math.max(err, Math.abs(nf.h2 - fz.h2) / 1e6, Math.abs(nf.ratio - fz.ratio), Math.abs(nf.TfMax - fz.TfMax) / 50);
      fz = { ...fz, ...nf };
    } else {
      const d = clamp(r.T[N - 1] - c.Tn, -6, 6); err = Math.max(err, Math.abs(d) / 10);
      if (c.areaMode === 'rating') Ts -= d; else dT = dT.map((v) => Math.max(0.05, v + d / N));
      pre = preheatTrain(cc, r, forward, Mev);
      err = Math.max(err, ...pre.tf.map((t, i) => Math.abs(t - tf[i]) / 50));
      tf = pre.tf; qph = pre.qph;
      if (tvc) {
        const Tn_ = r.Tc[cc.nEnt], hn = r.hm[cc.nEnt];
        err = Math.max(err, Math.abs(Tn_ - fz.Tev) / 50, Math.abs(hn - fz.hev) / 1e6);
        fz.Tev += 0.7 * (Tn_ - fz.Tev); fz.hev += 0.7 * (hn - fz.hev);
      }
    }
    errOuter = err;
    if (err < tol && it > 1) { conv = true; break; }
    if (!mvc && Math.min(...dT) <= 0.0501) { if (++floored > 15) break; } else floored = 0; // the temperature losses use up the whole window: no design exists
  }
  if (c.warm) { if (conv) Object.assign(c.warm, { key: wKey, dT: [...dT], tf: [...tf], qph: [...qph], Fi: [...cc.Fi], F, x, MdT: c.MdT, fz: { ...fz } }); else c.warm.key = ''; }
  // ---- a design that cannot be built is reported as such, never returned with open balances
  const L = N - 1, finite = [x, r.Md, Q1, ...r.T, ...r.V, ...r.A, ...dT].every(Number.isFinite), lossK = sum(r.be.slice(0, L)) + sum(r.dl.slice(0, L)), fewer = N > 1 ? 'use fewer effects' : 'widen the temperature window';
  if (!finite || (!conv && Math.min(...dT) <= 0.0501)) {
    if (mvc) throw infeasible(`The vapour-compression train cannot be solved: with ${N} effect${N > 1 ? 's' : ''} and a condensing − boiling difference of ${fmt(c.dTm, 3)} K the last effect would have to run below the ${fmt(c.Tsw, 3)} °C seawater. Use fewer effects, a smaller temperature difference or a higher first-effect temperature.`, 'window');
    throw infeasible(`The ${fmt(c.Ts - c.Tn, 3)} K between the heating steam (${fmt(c.Ts, 4)} °C) and the last effect (${fmt(c.Tn, 4)} °C) cannot drive ${N} effects: boiling-point elevation and demister/line losses alone take about ${fmt(lossK, 3)} K at ${fmt(c.Xb, 3)} g/kg. Raise the heating-steam temperature, lower the last-effect temperature, ${fewer} or lower the maximum brine salinity.`, 'window');
  }
  if (mvc && r.T[L] < c.Tsw + 1) throw infeasible(`With ${N} effects and a condensing − boiling difference of ${fmt(c.dTm, 3)} K the last effect would run at ${fmt(r.T[L], 3)} °C, not above the ${fmt(c.Tsw, 3)} °C seawater that feeds it. Use fewer effects, a smaller temperature difference or a higher first-effect temperature.`, 'window');
  if (r.cap >= 0) throw infeasible(`The brine of effect ${r.cap + 1} would be concentrated beyond 280 g/kg, where salt crystallises. Lower the maximum brine salinity or ${fewer}.`, 'salt');
  const iDry = r.V.findIndex((q) => !(q > 0)), iHeat = r.mh.findIndex((q) => !(q > 0));
  if (tvc && iHeat === cc.nEnt) throw infeasible(`The thermo-compressor would entrain ${fmt(Mev, 3)} kg/s from effect ${cc.nEnt + 1}, which produces only ${fmt(r.Vt[cc.nEnt], 3)} kg/s of vapour, so nothing is left to heat the ${cc.nEnt < L ? 'next effect' : 'final condenser'}. Take the suction from a colder effect (larger suction-effect number) so that less vapour is entrained per kg of motive steam, or use fewer effects.`, 'ejector');
  if (iDry >= 0 || iHeat >= 0) { const k = iDry >= 0 ? iDry : iHeat + 1; throw infeasible(`Effect ${Math.min(k, L) + 1} would receive no heating vapour or boil nothing: ${forward && !c.preheat ? 'in forward feed without preheaters the cold feed absorbs the heat of the first effect. Enable the feed preheaters, choose parallel feed or use fewer effects.' : 'the vapour of the effect before it is used up elsewhere. Use fewer effects or a wider temperature window.'}`, 'dry'); }
  if (!conv) throw infeasible(mvc ? `The vapour-compression heat balance did not converge for ${N} effect${N > 1 ? 's' : ''} at ${fmt(c.T1, 3)} °C with a ${fmt(c.dTm, 3)} K temperature difference. Use fewer effects, a larger temperature difference or a higher first-effect temperature.`
    : `The effect-by-effect iteration did not settle for ${N} effects between ${fmt(c.Ts, 4)} °C and ${fmt(c.Tn, 4)} °C (smallest driving temperature difference ${fmt(Math.min(...dT), 2)} K). Widen the temperature window, ${fewer} or select the equal-temperature-difference sizing.`, 'converge');
  if (!mvc && r.Tc[L] - c.ttdCond <= c.Tsw + 0.3) throw infeasible(`The last-effect vapour condenses at ${fmt(r.Tc[L], 3)} °C, too close to the ${fmt(c.Tsw, 3)} °C seawater for the final condenser with a ${fmt(c.ttdCond, 2)} K approach. Raise the last-effect temperature to at least ${fmt(Math.ceil(c.Tsw + c.ttdCond + (c.Tn - r.Tc[L]) + 1), 3)} °C or reduce the condenser approach.`, 'condenser');
  return { ...r, c: cc, dT, Ts, tf, qph, Q1, ej, comp, pre, converged: conv, iterations: it, forward, TfM, Qaux };
}

/** Final condenser and feed-preheater chain for the present vapour temperatures. */
function preheatTrain(c, r, forward, Mev) {
  const N = c.N, L = N - 1, F = c.F, h0 = hL(c.Tsw, c.Xf);
  let Tf0 = Math.max(c.Tsw + 0.3, r.Tc[L] - c.ttdCond);
  let Mcw = r.Qc / (hL(Tf0, c.Xf) - h0) - F;
  if (Mcw < 0) { Mcw = 0; Tf0 = tFromH(h0 + r.Qc / F, c.Xf); }
  const tf = fill(N, Tf0), qph = fill(N, 0), tin = fill(N, Tf0), tout = fill(N, Tf0), flow = fill(N, 0);
  if (c.preheat) {
    let t = Tf0;
    for (let i = N - 2; i >= 0; i--) {
      const m = forward ? F : sum(c.Fi.slice(0, i + 1)), avail = 0.35 * Math.max(0, r.Vt[i] - (i === c.nEnt ? Mev : 0)) * (r.hm[i] - hL(r.Tc[i]));
      let to = Math.max(t, r.Tc[i] - c.ttdPh), q = m * (hL(to, c.Xf) - hL(t, c.Xf));
      if (q > avail) { q = avail; to = tFromH(hL(t, c.Xf) + q / m, c.Xf); }
      qph[i] = q; tin[i] = t; tout[i] = to; flow[i] = m; t = to;
      if (!forward) tf[i] = to;
    }
    if (forward) tf[0] = t;
  }
  return { Tf0, Mcw, tf, qph, tin, tout, flow };
}

// ---- multi-stage flash ------------------------------------------------------------------------------------
/** Stage-by-stage MSF (c.type 'br' brine recirculation with nRej heat-rejection stages, 'ot' once-through). */
export function solveMSF(c) {
  const N = c.N, o = c.o, br = c.type === 'br', nRej = br ? clamp(c.nRej, 1, N - 1) : 0, nRec = N - nRej, dTs = (c.TBT - c.Tn) / N;
  if (!(dTs >= 0.5)) { // every stage needs a temperature drop that exceeds its non-equilibrium and demister losses
    const nMax = Math.floor((c.TBT - c.Tn) / 0.5);
    throw infeasible(`The flashing range between the top brine temperature (${fmt(c.TBT, 4)} °C) and the last stage (${fmt(c.Tn, 4)} °C) is ${fmt(Math.max(0, c.TBT - c.Tn), 3)} K; ${N} stages need at least ${fmt(0.5 * N, 3)} K (0.5 K per stage). Raise the top brine temperature to at least ${fmt(Math.ceil(c.Tn + 0.5 * N), 4)} °C or lower the last-stage temperature to ${fmt(Math.floor(c.TBT - 0.5 * N), 4)} °C or less${nMax >= 3 ? `, or use at most ${nMax} stages` : ''}.`, 'order');
  }
  const T = Array.from({ length: N + 1 }, (_, i) => c.TBT - i * dTs);
  const Mf0 = br ? (c.MdT * c.Xb) / (c.Xb - c.Xf) : 0;
  let Mr = c.MdT / (1 - Math.exp((-cp(75, c.Xf) * (c.TBT - c.Tn)) / latentHeat(75))), st, Xr = c.Xf, conv = false;
  const march = () => {
    let B = Mr, X = Xr, hb = hL(T[0], X), Dacc = 0, Td = T[0];
    const s = { D: [], B: [], X: [], Tv: [], be: [], nea: [], dl: [], Qc: [], Dacc: [] };
    for (let i = 1; i <= N; i++) {
      const nea = nonEquilibriumAllowance(T[i], dTs, o);
      let D = (B * cp(T[i], X) * dTs) / latentHeat(T[i]), Bn, Xn, be, Tv, dl = 0, hv, hbn;
      for (let k = 0; k < 40; k++) {
        Bn = B - D; Xn = (B * X) / Bn; be = bpeOf(T[i], Xn, o); hbn = hL(T[i], Xn);
        Tv = T[i] - be - nea; dl = vapourLoss(Tv, o); Tv -= dl; hv = hV(Tv) + CPV * (T[i] - Tv);
        const Dn = (B * (hb - hbn)) / (hv - hbn);
        if (Math.abs(Dn - D) < 1e-13 * B) { D = Dn; break; }
        D = Dn;
      }
      Bn = B - D; Xn = (B * X) / Bn; hbn = hL(T[i], Xn);
      s.Qc.push(D * (hv - hL(Tv)) + Dacc * (hL(Td) - hL(Tv)));
      Dacc += D; Td = Tv; B = Bn; X = Xn; hb = hbn;
      s.D.push(D); s.B.push(B); s.X.push(X); s.Tv.push(Tv); s.be.push(be); s.nea.push(nea); s.dl.push(dl); s.Dacc.push(Dacc);
    }
    s.Md = Dacc;
    return s;
  };
  for (let it = 0; it < 80; it++) {
    Xr = br ? (c.Xb * (Mr - c.MdT)) / Mr : c.Xf;
    st = march();
    if (Math.abs(st.Md - c.MdT) < 1e-11 * c.MdT) { conv = true; break; }
    Mr *= c.MdT / st.Md;
  }
  if (!conv || ![Mr, st.Md, ...st.X, ...st.Tv].every(Number.isFinite)) throw infeasible(`The stage-by-stage flashing balance did not converge for ${N} stages between ${fmt(c.TBT, 4)} °C and ${fmt(c.Tn, 4)} °C. Widen the flashing range (higher top brine temperature or lower last-stage temperature) or use fewer stages.`, 'converge');
  if (br && Mr < Mf0 * (1 + 1e-9)) throw infeasible(`The brine recirculation (${fmt(Mr, 4)} kg/s) would be smaller than the make-up feed (${fmt(Mf0, 4)} kg/s) needed to hold the blow-down at ${fmt(c.Xb, 3)} g/kg. Raise the maximum brine salinity, reduce the flashing range or choose the once-through arrangement.`, 'recycle');
  const Mf = br ? Mf0 : Mr, Mb = Mf - st.Md, Xb = st.X[N - 1];
  // tube side, cold end first
  const t = fill(N + 2, c.Tsw);
  let Mcw = 0, Mrej = Mf, tout = c.Tsw, Tr = c.Tsw;
  if (br) {
    const Qrej = sum(st.Qc.slice(nRec)), h0 = hL(c.Tsw, c.Xf);
    tout = st.Tv[nRec] - c.ttdRej;
    if (tout <= c.Tsw + 0.2) throw infeasible(`The heat-rejection stages condense at ${fmt(st.Tv[nRec], 3)} °C, too close to the ${fmt(c.Tsw, 3)} °C seawater for a ${fmt(c.ttdRej, 2)} K approach. Raise the last-stage temperature, reduce the rejection approach or use fewer rejection stages.`, 'reject');
    Mrej = Qrej / (hL(tout, c.Xf) - h0);
    if (Mrej < Mf) { Mrej = Mf; tout = tFromH(h0 + Qrej / Mf, c.Xf); }
    Mcw = Mrej - Mf;
    let h = h0; t[N + 1] = c.Tsw;
    for (let i = N; i > nRec; i--) { h += st.Qc[i - 1] / Mrej; t[i] = tFromH(h, c.Xf); }
    Tr = tFromH((Mf * hL(tout, c.Xf) + (Mr - Mf) * hL(T[N], Xb)) / Mr, Xr);
  }
  let h = hL(Tr, Xr);
  const tInRec = Tr;
  for (let i = nRec; i >= 1; i--) { h += st.Qc[i - 1] / Mr; t[i] = tFromH(h, Xr); }
  const Qh = Mr * (hL(T[0], Xr) - hL(t[1], Xr)), Qin = Qh / (1 - c.loss);
  const stages = [];
  for (let i = 1; i <= N; i++) {
    const rec = i <= nRec, m = rec ? Mr : Mrej, S = rec ? Xr : c.Xf, tin = i === nRec ? tInRec : t[i + 1];
    const U = fouled(uCondenser(st.Tv[i - 1]), o), hx = condenserHX(`Stage ${i}`, st.Tv[i - 1], m, S, tin, t[i], U, c.nSeg);
    stages.push({ i, rec, Tin: T[i - 1], T: T[i], Tv: st.Tv[i - 1], P: psat(st.Tv[i - 1]), B: st.B[i - 1], X: st.X[i - 1], D: st.D[i - 1], Dacc: st.Dacc[i - 1], be: st.be[i - 1], nea: st.nea[i - 1], dl: st.dl[i - 1], tin, tout: t[i], ttd: st.Tv[i - 1] - t[i], U, A: hx.area, Q: st.Qc[i - 1], ntu: hx.ntu, eff: hx.eff, m });
  }
  const bad = stages.find((s) => s.ttd <= 0.05);
  if (bad) throw infeasible(`Stage ${bad.i}: the tube-side brine (${fmt(bad.tout, 4)} °C) is not colder than the condensing vapour (${fmt(bad.Tv, 4)} °C), so the stage cannot recover its heat. Add stages, raise the top brine temperature or raise the last-stage temperature.`, 'stage');
  const heater = condenserHX('Brine heater', c.Tsteam, Mr, Xr, t[1], T[0], fouled(uCondenser(c.Tsteam), o), c.nSeg);
  return { c, N, nRec, nRej, dTs, T, t, stages, Mr, Xr, Mf, Mb, Xb, Md: st.Md, Mcw, Mrej, tout, Tr, Qh, Qin, Ms: Qin / latentHeat(c.Tsteam), heater, converged: conv, Td: st.Tv[N - 1] };
}

// ---- unified simulation ------------------------------------------------------------------------------------
const exL = (m, T, T0) => m * cp(0.5 * (T + T0), 0) * (T - T0 - K(T0) * Math.log(K(T) / K(T0)));
const exV = (T, T0, sup = 0) => hV(T) + CPV * sup - hL(T0) - K(T0) * (cp(0.5 * (T + T0), 0) * Math.log(K(T) / K(T0)) + latentHeat(T) / K(T) + CPV * Math.log((K(T) + sup) / K(T)));
const Iq = (Q, Thot, Tcold, T0) => K(T0) * Q * (1 / K(Tcold) - 1 / K(Thot));
const COLLECTORS = { flat: { name: 'Flat-plate collector', eta0: 0.76, a1: 3.6, a2: 0.014, tmax: 85 }, etc: { name: 'Evacuated-tube collector', eta0: 0.7, a1: 1.3, a2: 0.007, tmax: 130 }, trough: { name: 'Parabolic trough', eta0: 0.72, a1: 0.25, a2: 0.0012, tmax: 400 } };
const ANTISCALANT = { poly: { name: 'Polyphosphate', tbt: 90 }, ht: { name: 'High-temperature polymer additive', tbt: 112 }, acid: { name: 'Acid dosing + decarbonation', tbt: 120 } };

function options(p) {
  return { fBPE: p.fBPE, bpeModel: p.bpeModel || 'corr', fNEA: p.fNEA, neaH: p.neaH, neaVb: p.neaVb, neaL: p.neaL, dpDem: demisterDrop(p.rhoP, p.Vdem, p.dw, p.thick / 1000), linePct: p.linePct, fU: p.fU, Rf: p.Rf / 1000 };
}

/** Run the selected process once. Returns one unified result object (SI units, flows in kg/s). */
export function simulateThermal(v, ov = {}) {
  const p = { ...v, ...ov }, o = options(p), proc = p.proc, T0 = p.Tsw, Xf = p.Xf, MdT = (p.Md * RHO_REF) / 86400, nSeg = Math.max(1, Math.round(p.nSeg));
  const loss = clamp(p.heatLoss / 100, 0, 0.5), vent = clamp(p.ventPct / 100, 0, 0.2), eP = p.etaPump / 100, rhoS = density(T0, Xf);
  const R = { proc, p, o, T0, Xf, MdT, hx: [], ex: [], rows: [], warnings: [] };
  let pumps = {};
  if (proc === 'msf') {
    const N = Math.max(2, Math.round(p.Nst)), Tsteam = p.TBT + p.dTsteam;
    let m;
    try { m = solveMSF({ N, nRej: Math.round(p.nRej), type: p.msfType, TBT: p.TBT, Tn: p.TnMsf, Tsw: T0, Xf, Xb: Math.max(p.Xb, Xf * 1.25), MdT, ttdRej: p.ttdRej, Tsteam, loss, o, nSeg }); } catch (e) { throw e.infeasible && !ov.fast && !ov.probe && !p._lean ? withHint(e, p, ov) : e; }
    Object.assign(R, { m, N, Md: m.Md * (1 - vent), Mf: m.Mf, Mb: m.Mb, Xb: m.Xb, Mcw: m.Mcw, Ms: m.Ms, Qin: m.Qin, Tsteam, Ttop: p.TBT, Tlast: p.TnMsf, Tprod: m.Td, Tbrine: p.TnMsf, Tcw: m.tout, converged: m.converged });
    R.Qcond = sum(m.stages.filter((s) => !s.rec).map((s) => s.Q));
    R.areaEvap = sum(m.stages.map((s) => s.A)); R.areaAux = m.heater.area;
    R.hx = [m.heater, ...(m.nRej ? [{ name: 'Heat-rejection section', duty: R.Qcond, area: sum(m.stages.filter((s) => !s.rec).map((s) => s.A)), lmtd: NaN, U: m.stages[N - 1].U, ntu: sum(m.stages.filter((s) => !s.rec).map((s) => s.ntu)), eff: m.Mcw + m.Mf > 0 ? (m.tout - T0) / (m.stages[m.nRec].Tv - T0) : 0, tin: T0, tout: m.tout, Th: m.stages[m.nRec].Tv }] : []),
      { name: 'Heat-recovery section', duty: sum(m.stages.filter((s) => s.rec).map((s) => s.Q)), area: sum(m.stages.filter((s) => s.rec).map((s) => s.A)), lmtd: NaN, U: m.stages[0].U, ntu: sum(m.stages.filter((s) => s.rec).map((s) => s.ntu)), eff: (m.t[1] - m.stages[m.nRec - 1].tin) / (m.stages[0].Tv - m.stages[m.nRec - 1].tin), tin: m.stages[m.nRec - 1].tin, tout: m.t[1], Th: m.stages[0].Tv }];
    pumps = { 'Seawater supply': ((m.Mf + m.Mcw) * p.dpSea * 1e5) / (rhoS * eP), 'Brine recirculation': (m.Mr * p.dpRec * 1e5) / (density(p.TnMsf, m.Xr) * eP), 'Distillate': (R.Md * p.dpProd * 1e5) / (RHO_REF * eP), 'Brine blowdown': (m.Mb * p.dpBrine * 1e5) / (density(p.TnMsf, m.Xb) * eP) };
    R.exIn = m.Qin * (1 - K(T0) / K(Tsteam));
    R.ex = [['Brine heater', Iq(m.Qh, Tsteam, 0.5 * (m.t[1] + p.TBT), T0)], ['Stage condensers (ΔT)', sum(m.stages.map((s) => Iq(s.Q, s.Tv, 0.5 * (s.tin + s.tout), T0)))],
      ['Flashing, BPE, NEA, demisters', sum(m.stages.map((s) => Iq(s.Q, 0.5 * (s.Tin + s.T), s.Tv, T0)))], ['Cooling-water reject', exL(m.Mcw, m.tout, T0)], ['Brine and distillate discharge', exL(m.Mb, p.TnMsf, T0) + exL(m.Md, m.Td, T0)]];
    R.balance = { massIn: m.Mf + m.Mcw, massOut: m.Md + m.Mb + m.Mcw, saltIn: m.Mf * Xf, saltOut: m.Mb * m.Xb, eIn: m.Qh + (m.Mf + m.Mcw) * hL(T0, Xf), eOut: m.Md * hL(m.Td) + m.Mb * hL(p.TnMsf, m.Xb) + m.Mcw * hL(m.tout, Xf) };
  } else {
    const mvc = proc === 'mvc', tvc = proc === 'medtvc', N = Math.max(1, Math.round(mvc ? p.Nmvc : p.N)), Xb = Math.max(p.Xb, Xf * 1.25), F0 = (MdT * Xb) / (Xb - Xf);
    const c = { proc: mvc ? 'mvc' : tvc ? 'tvc' : 'med', N, arr: p.arr, F: F0, Xf, Xb, Tsw: T0, MdT, Ts: p.Ts, Tn: p.Tn, T1: p.Tmvc, dTm: p.dTmvc, preheat: !!p.preheat, ttdPh: p.ttdPh, ttdCond: p.ttdCond, loss, vent, o,
      areaMode: ov.areaMode || ((p.areaMode === 'equalArea' || p.areaMode === 'flux') && !ov.fast ? p.areaMode : 'equalDT'), Ades: ov.Ades, dT0: ov.dT0, tol: ov.fast ? 3e-7 : 1e-10, maxIt: ov.fast ? 60 : 250, warm: ov.warm, Pm: p.Pm * 1e5, nEnt: (Math.round(p.nEnt) > 0 ? Math.round(p.nEnt) : N) - 1, etaIs: p.etaIs / 100 };
    if (!mvc && p.Tn > p.Ts - 0.8 * N - 1) { // each effect needs roughly 0.8 K for boiling-point elevation, demister and a minimum driving force
      const nMax = Math.floor((p.Ts - p.Tn - 1) / 0.8);
      throw infeasible(`The last-effect temperature (${fmt(p.Tn, 4)} °C) must lie at least ${fmt(0.8 * N + 1, 3)} K below the heating steam (${fmt(p.Ts, 4)} °C) for ${N} effect${N > 1 ? 's' : ''}. Lower the last-effect temperature to ${fmt(Math.floor(10 * (p.Ts - 0.8 * N - 1)) / 10, 4)} °C or less, or raise the heating-steam temperature to at least ${fmt(Math.ceil(10 * (p.Tn + 0.8 * N + 1)) / 10, 4)} °C${nMax >= 1 && nMax < N ? `, or use at most ${nMax} effect${nMax > 1 ? 's' : ''}` : ''}.`, 'order');
    }
    if (tvc && tsat(p.Pm * 1e5) < p.Ts + 5) throw infeasible(`Motive steam at ${fmt(p.Pm, 3)} bar condenses at ${fmt(tsat(p.Pm * 1e5), 4)} °C and cannot compress vapour to the ${fmt(p.Ts, 4)} °C heating-steam level. Raise the motive-steam pressure to at least ${fmt(Math.ceil(psat(p.Ts + 5) / 1e4) / 10, 3)} bar or lower the heating-steam temperature.`, 'order');
    let r;
    try { r = solveMED(c); } catch (e) { throw e.infeasible && !ov.fast && !ov.probe && !p._lean ? withHint(e, p, ov) : e; }
    const F = r.c.F, L = N - 1, Md = r.Md, Mb = r.B[L], Tprod = tFromH(mvc ? (r.Dacc * r.hD + r.cond * hL(r.Ts)) / Md : (r.Dacc * r.hD + r.cond * hL(r.Tc[L])) / Md, 0);
    Object.assign(R, { r, N, Md, Mf: F, Mb, Xb: r.X[L], Mcw: 0, Ttop: r.T[0], Tlast: r.T[L], Tsteam: r.Ts, Tprod, Tbrine: r.T[L], Tcw: T0, Qcond: 0, converged: r.converged, areaEvap: sum(r.A), areaAux: 0 });
    const effEx = sum(r.q.map((q, i) => Iq(q, r.Th[i], r.T[i], T0))), lossEx = sum(r.Vt.map((m, i) => Iq(m * latentHeat(r.Tv[i]), r.T[i], r.Tc[i], T0)));
    if (mvc) {
      const cm = r.comp, Wsh = r.cond * cm.w, hProd = (r.Dacc * r.hD + r.cond * hL(r.Ts)) / Md, hB = hL(r.T[L], r.X[L]), dHf = F * (hL(r.TfM, Xf) - hL(T0, Xf));
      // feed preheater: product and brine cooled to a common outlet temperature To against the incoming feed
      const g = (To) => Md * (hProd - hL(To, 0)) + Mb * (hB - hL(To, r.X[L])) - dHf;
      let To = T0 + Math.min(p.ttdCond, 0.5 * (r.T[L] - T0)), ToB = To, Qfeed = 0;
      if (g(To) < 0) Qfeed = -g(To);
      else if (g(r.T[L]) <= 0) To = ToB = brent(g, To, r.T[L], 1e-10);
      else { // the feed cannot take up even the heat of the distillate cooled to the last-effect temperature: the brine by-passes the preheater and the distillate leaves warmer
        const gD = (t) => Md * (hProd - hL(t, 0)) - dHf;
        ToB = r.T[L]; To = gD(Tprod) < 0 ? brent(gD, r.T[L], Tprod, 1e-10) : Tprod;
      }
      const Qaux = r.Qaux + Qfeed;
      const qD = Md * (hProd - hL(To, 0)), qB = Mb * (hB - hL(ToB, r.X[L])), U = p.Uph * 1000 * o.fU, aHot = Math.max(0.05, (Md * Tprod + Mb * r.T[L]) / (Md + Mb) - r.TfM); // both feed branches leave with the same hot-end approach
      const cex = (name, q, Th, Tout) => {
        if (!(q > 1e-9 * (Wsh + 1))) return { name, duty: 0, area: 0, lmtd: 0, U, ntu: 0, eff: 0, tin: T0, tout: T0, Th }; // branch not in service
        const l = lmtd(aHot, Tout - T0), A = q / (U * l), rise = Th - aHot - T0, Cmin = q / Math.max(Th - Tout, rise, 1e-9);
        return { name, duty: q, area: A, lmtd: l, U, ntu: (U * A) / Cmin, eff: Math.max(Th - Tout, rise) / (Th - T0), tin: T0, tout: Th - aHot, Th };
      };
      R.hx = [cex('Feed preheater · distillate side', qD, Tprod, To), cex('Feed preheater · brine side', qB, r.T[L], ToB)];
      Object.assign(R, { Ms: 0, Qin: Qaux, Wcomp: Wsh / ((p.etaMotor / 100) * 0.98), Wshaft: Wsh, To, Qaux, Tprod: To, Tbrine: ToB, comp: cm, areaAux: sum(R.hx.map((x) => x.area)) });
      R.exIn = Wsh + Qaux * (1 - K(T0) / K(r.Ts));
      R.ex = [['Evaporator tubes (ΔT)', effEx], ['BPE, demister and line losses', lossEx], ['Compressor', K(T0) * r.cond * (CPV * Math.log(K(cm.T2) / K(cm.T1)) - RW * Math.log(cm.ratio))],
        ['Feed preheater', Iq(qD, 0.5 * (Tprod + To), 0.5 * (T0 + Tprod - aHot), T0) + Iq(qB, 0.5 * (r.T[L] + ToB), 0.5 * (T0 + r.T[L] - aHot), T0)], ['Brine and distillate discharge', exL(Mb, ToB, T0) + exL(Md, To, T0)]];
      R.balance = { massIn: F, massOut: Md + Mb + r.vent, saltIn: F * Xf, saltOut: Mb * r.X[L], eIn: Wsh + Qaux + F * hL(T0, Xf), eOut: Md * hL(To, 0) + Mb * hL(ToB, r.X[L]) + r.vent * r.hm[L] + r.qloss };
      pumps = { 'Feed supply': (F * p.dpSea * 1e5) / (rhoS * eP), 'Distillate': (Md * p.dpProd * 1e5) / (RHO_REF * eP), 'Brine blowdown': (Mb * p.dpBrine * 1e5) / (density(ToB, R.Xb) * eP) };
    } else {
      const pre = r.pre, Uc = fouled(uCondenser(r.Tc[L]), o);
      R.Mcw = pre.Mcw; R.Qcond = r.Qc; R.Tcw = pre.Tf0;
      R.hx = [condenserHX('Final condenser', r.Tc[L], F + pre.Mcw, Xf, T0, pre.Tf0, Uc, nSeg)];
      if (c.preheat) for (let i = N - 2; i >= 0; i--) if (pre.qph[i] > 0) R.hx.push(condenserHX(`Feed preheater ${i + 1}`, r.Tc[i], pre.flow[i], Xf, pre.tin[i], pre.tout[i], fouled(uCondenser(r.Tc[i]), o), nSeg));
      R.areaAux = sum(R.hx.map((x) => x.area));
      R.Qin = tvc ? r.ej.Qin : r.Q1; R.Ms = tvc ? r.ej.Mm : r.Q1 / latentHeat(r.Ts); R.ej = r.ej;
      R.exIn = tvc ? r.ej.Mm * (exV(r.ej.Tm, T0) - exL(1, r.Ts, T0)) : r.Q1 * (1 - K(T0) / K(r.Ts));
      R.ex = [['Effect tubes (ΔT)', effEx], ['BPE, demister and line losses', lossEx], ['Feed preheaters', sum(pre.qph.map((q, i) => (q > 0 ? Iq(q, r.Tc[i], 0.5 * (pre.tin[i] + pre.tout[i]), T0) : 0)))],
        ['Final condenser', Iq(r.Qc, r.Tc[L], 0.5 * (T0 + pre.Tf0), T0)], ['Cooling-water reject', exL(pre.Mcw, pre.Tf0, T0)], ['Brine and distillate discharge', exL(Mb, r.T[L], T0) + exL(Md, Tprod, T0)]];
      if (tvc) R.ex.push(['Steam-jet ejector', Math.max(0, r.ej.Mm * exV(r.ej.Tm, T0) + r.ej.Mev * exV(r.ej.Tev, T0) - (r.ej.Mm + r.ej.Mev) * exV(r.Ts, T0, r.ej.Td - r.Ts))]);
      R.balance = { massIn: F + pre.Mcw, massOut: Md + Mb + pre.Mcw + r.vent, saltIn: F * Xf, saltOut: Mb * r.X[L], eIn: R.Qin + (F + pre.Mcw) * hL(T0, Xf), eOut: r.Dacc * r.hD + r.cond * hL(r.Tc[L]) + Mb * hL(r.T[L], r.X[L]) + pre.Mcw * hL(pre.Tf0, Xf) + r.vent * r.hm[L] + r.qloss };
      pumps = { 'Seawater supply': ((F + pre.Mcw) * p.dpSea * 1e5) / (rhoS * eP), 'Distillate': (Md * p.dpProd * 1e5) / (RHO_REF * eP), 'Brine blowdown': (Mb * p.dpBrine * 1e5) / (density(r.T[L], R.Xb) * eP) };
    }
  }
  // vacuum system: isothermal compression of the vented vapour/NCG mixture from the coldest stage to atmosphere
  const Pvac = psat(proc === 'msf' ? R.m.Td : R.r.Tc[R.N - 1]), mVent = vent * (proc === 'msf' ? R.m.Md : R.r.cond / (1 - vent || 1));
  pumps['Vacuum / NCG venting'] = Pvac < 101325 ? (mVent * RW * K(R.Tlast) * Math.log(101325 / Pvac)) / 0.45 : 0;
  if (R.Wcomp) pumps['Vapour compressor'] = R.Wcomp;
  R.pumps = pumps; R.Wel = sum(Object.values(pumps)); R.vent = mVent;
  R.area = R.areaEvap + R.areaAux;
  R.Qd = (R.Md * 3600) / RHO_REF; // m³/h
  R.rec = R.Md / R.Mf;
  R.GOR = R.Ms > 0 ? R.Md / R.Ms : null; R.PR = R.Qin > 0 ? (R.Md * 2326e3) / R.Qin : null;
  R.sThKJ = R.Qin / R.Md / 1000; R.sTh = R.Qin / 1000 / R.Qd; R.sEl = R.Wel / 1000 / R.Qd;
  R.sEq = R.sEl + (proc === 'mvc' ? R.sTh : R.sTh * (p.etaTurb / 100) * Math.max(0, 1 - K(p.TcondPP) / K(proc === 'medtvc' ? R.ej.Tm : R.Tsteam)));
  R.sArea = R.area / R.Md; // m² per kg/s
  const rr = clamp(R.rec, 1e-4, 0.95);
  R.wMin = (osmoticPressure(T0, Xf) / 3.6e6) * (-Math.log(1 - rr) / rr); // kWh/m³
  const exUsed = sum(R.ex.map((e) => e[1])), exTot = R.exIn + (proc === 'mvc' ? R.Wel - R.Wcomp + (R.Wcomp - R.Wshaft) : R.Wel);
  R.ex.push(['Other (flash boxes, mixing, heat loss, separation work)', Math.max(0, R.exIn - exUsed)]);
  if (proc === 'mvc') R.ex.push(['Motor and drive losses', R.Wcomp - R.Wshaft]);
  R.eta2 = (R.wMin * R.Qd * 1000) / exTot; R.exTot = exTot;
  return R;
}

/** Lumped thermal-inertia transient (RK4): start-up from seawater temperature, or cool-down from the design state after a steam trip (shutdown = true). */
export function startUp(R, p, shutdown = false) {
  const N = R.N, T0 = R.T0, nt = Math.max(10, Math.round(p.nt)), tEnd = p.tEnd * 3600, ramp = Math.max(1, p.ramp * 60), cth = p.cth * 1000;
  let f, Tdes, lam, Av, TvDes;
  const vv = Math.max(0, p.vapVol ?? 0), vapC = (T, A) => vv * A * (latentHeat(T) * (rhoV(T + 0.5) - rhoV(T - 0.5)) + rhoV(T) * CPV); // J/K stored by the saturated vapour space as it warms
  if (R.proc === 'msf') {
    const m = R.m, cpb = cp(70, m.Xr), C = m.stages.map((s) => s.A * cth + s.B * p.holdup * cpb + vapC(s.Tv, s.A)), eps = m.stages.map((s) => (s.tout - s.tin) / (s.Tv - s.tin)), lossT = m.stages.map((s) => s.T - s.Tv);
    Tdes = m.stages.map((s) => s.T); Av = m.stages.map((s) => s.A); TvDes = m.stages.map((s) => s.Tv);
    lam = Math.max(...C.map((ci, i) => ((m.Mr + m.stages[i].m * eps[i]) * cpb) / ci)) * 2;
    f = (t, T) => {
      const tu = fill(N + 2, T0), ti = fill(N + 1, T0); // tube outlet of stage i (1-based) and its inlet
      for (let i = N; i >= 1; i--) {
        ti[i] = i === m.nRec && m.nRej ? (m.Mf * tu[i + 1] + (m.Mr - m.Mf) * T[N - 1]) / m.Mr : tu[i + 1];
        tu[i] = ti[i] + eps[i - 1] * Math.max(0, T[i - 1] - lossT[i - 1] - ti[i]);
      }
      const Ttop = tu[1] + (shutdown ? 0 : Math.min(m.Qh * Math.min(1, t / ramp), m.Mr * cpb * Math.max(0, m.T[0] - tu[1])) / (m.Mr * cpb)), d = [];
      for (let i = 0; i < N; i++) d.push((m.Mr * cpb * ((i ? T[i - 1] : Ttop) - T[i]) - m.stages[i].m * cpb * (tu[i + 1] - ti[i + 1])) / C[i]);
      return d;
    };
  } else {
    const r = R.r, UA = r.q.map((q, i) => q / r.dT[i]), dl = r.T.map((T, i) => T - r.Tc[i]), cpb = cp(60, R.Xf);
    const C = r.A.map((a, i) => a * cth + r.B[i] * p.holdup * cpb + vapC(r.Tv[i], a)), qOut = r.q.map((_, i) => (i < N - 1 ? r.q[i + 1] : r.q[i]));
    const sink = r.q.map((q, i) => (q - (i < N - 1 ? qOut[i] : 0)) / (r.T[i] - T0)), UAc = (R.proc === 'mvc' ? 0 : r.q[N - 1]) / Math.max(0.5, r.Tc[N - 1] - T0);
    Tdes = r.T; Av = r.A; TvDes = r.Tv;
    lam = Math.max(...C.map((ci, i) => (UA[i] + (UA[i + 1] || UAc) + Math.abs(sink[i])) / ci)) * 2;
    f = (t, T) => {
      const qin = T.map((Ti, i) => (i === 0 ? (shutdown ? 0 : Math.min(r.q[0] * Math.min(1, t / ramp), UA[0] * Math.max(0, r.Ts - Ti) * 3)) : UA[i] * Math.max(0, T[i - 1] - dl[i - 1] - Ti)));
      return T.map((Ti, i) => (qin[i] - (i < N - 1 ? qin[i + 1] : UAc * Math.max(0, Ti - dl[i] - T0)) - (i < N - 1 || R.proc === 'mvc' ? sink[i] * (Ti - T0) : sink[i] * 0)) / C[i]);
    };
  }
  const sub = Math.max(1, Math.ceil(((tEnd / nt) * lam) / 2.2)), sol = rk4(f, shutdown ? [...Tdes] : fill(N, T0), 0, tEnd, nt * sub);
  const ts = [], Ts = [], prod = [], inv = [], vol = Av.map((a) => vv * a), mDes = vol.map((V, i) => V * rhoV(TvDes[i])), mTot = sum(mDes);
  for (let k = 0; k <= nt; k++) {
    const y = sol.y[k * sub];
    ts.push(sol.t[k * sub] / 60); Ts.push(y);
    prod.push(100 * clamp(sum(y.map((T, i) => (T - T0) / (Tdes[i] - T0))) / N, 0, 1.5));
    inv.push(mTot > 0 ? (100 * sum(y.map((T, i) => vol[i] * rhoV(Math.max(T0, T - (Tdes[i] - TvDes[i]))))) / mTot) : 0);
  }
  const lim = shutdown ? 50 : 95, i95 = prod.findIndex((x) => (shutdown ? x <= lim : x >= lim)); // 95 % approach on start-up, half of the temperature rise lost on cool-down
  const t95 = i95 > 0 ? ts[i95 - 1] + ((ts[i95] - ts[i95 - 1]) * (lim - prod[i95 - 1])) / (prod[i95] - prod[i95 - 1]) : i95 === 0 ? 0 : null;
  return { t: ts, T: Ts, prod, t95, sub, Tdes, final: prod[prod.length - 1], vap: { vol, m0: vol.map((V) => V * rhoV(T0)), mDes, mTot, inv, TvDes } };
}

// ---- Raoult comparison, heat pump, cogeneration, costing, RO hybrid, design-grid optimisation -----------------
/** Specific entropy of saturated liquid water relative to 0 °C, J/kg·K: s_f = ∫ c_p dT / T (Simpson's rule). */
export function sLiquid(T) {
  const n = 40, h = T / n; let a = cp(0, 0) / K(0) + cp(T, 0) / K(T);
  for (let i = 1; i < n; i++) a += (i % 2 ? 4 : 2) * (cp(i * h, 0) / K(i * h));
  return (a * h) / 3;
}
/** Specific entropy of saturated steam, J/kg·K: s_g = s_f + λ/T. */
export const sVapour = (T) => sLiquid(T) + latentHeat(T) / K(T);
/** Isentropic expansion of steam of entropy s into the two-phase region at saturation temperature T2: quality and enthalpy. */
const expandTo = (s, T2) => { const x = (s - sLiquid(T2)) / (latentHeat(T2) / K(T2)); return { x, h: hL(T2) + x * latentHeat(T2) }; };

/**
 * Power–water cogeneration by turbine extraction (lost-work / power-loss method). Saturated steam at the throttle pressure
 * expands with isentropic efficiency p.etaTurb to the extraction pressure p_sat(Text), where Qin (W) is withdrawn for the
 * desalination plant; the same steam would otherwise expand on to the condenser at p.TcondPP.
 */
export function cogeneration(Qin, Text, Wel, p) {
  const T1 = tsat(p.Pthr * 1e5), eta = clamp(p.etaTurb / 100, 0.05, 1), Tc = p.TcondPP, Te = clamp(Text, Tc + 0.5, T1 - 0.5), h1 = hV(T1), s1 = sVapour(T1);
  const eS = expandTo(s1, Te), hE = h1 - eta * (h1 - eS.h), xE = (hE - hL(Te)) / latentHeat(Te), sE = sLiquid(Te) + (xE * latentHeat(Te)) / K(Te);
  const cS = expandTo(sE, Tc), wLost = eta * (hE - cS.h), wFull = h1 - hE + wLost, qExt = hE - hL(Te); // J/kg
  const mExt = Qin / qExt, Wlost = mExt * wLost, Pgross = p.Ppp * 1e6, m0 = Pgross / wFull, Pnet = Pgross - Wlost, etaB = clamp(p.etaBoiler / 100, 0.3, 1);
  const fuel0 = (m0 * (h1 - hL(Tc))) / etaB, fuel = (m0 * h1 - (m0 - mExt) * hL(Tc) - mExt * hL(Te)) / etaB, fuelWater = (Wlost + Wel) * (fuel0 / Pgross);
  return { T1, Te, Tc, eta, h1, hE, xE, xC: cS.x, wLost, wFull, qExt, mExt, m0, Wlost, Pgross, Pnet, Pexport: Pnet - Wel, fuel0, fuel, fuelWater, fuelShare: fuelWater / fuel, euf: (Pnet + Qin) / fuel, etaPower: Pgross / fuel0,
    lostPerHeat: wLost / qExt, carnot: 1 - K(Tc) / K(Te), condensate: (hL(Te) - hL(Tc) - K(Tc) * (sLiquid(Te) - sLiquid(Tc))) / qExt, feasible: mExt <= m0 && Text < T1 - 0.5, clipped: Text >= T1 - 0.5 };
}

/**
 * Heat-pump-driven desalination: the heat rejected in the final condenser / heat-rejection section is lifted to the heating-steam
 * level. 'mhp' = mechanical (electric) heat pump, COP = η·T_h/(T_h − T_l); 'ahp' = absorption heat pump driven by heat at T_g,
 * COP = 1 + η·(COP_rev − 1) with the reversible three-temperature limit COP_rev = (1 − T_l/T_g)/(1 − T_l/T_h).
 */
export function heatPump(R, p) {
  const abs = p.source === 'ahp', a = Math.max(0, p.hpApproach), eta = clamp(p.hpEta / 100, 0.01, 1), Tsup = R.proc === 'medtvc' ? R.ej.Tm : R.Tsteam, Th = Tsup + a;
  const Tsrc = R.proc === 'msf' ? R.m.Td : R.proc === 'mvc' ? R.T0 + a + 1 : R.r.Tc[R.N - 1], Tl = Math.min(Tsrc - a, Th - 2), Tg = Math.max(p.hpTg, Th + 10);
  const carnot = K(Th) / (Th - Tl), copRev = abs ? ((Tg - Tl) / K(Tg)) * carnot : carnot, cop = abs ? 1 + eta * (copRev - 1) : Math.max(1, eta * copRev);
  const Qneed = R.Qin, Qavail = Math.max(0, R.Qcond), Qe = Math.min(Qneed * (1 - 1 / cop), Qavail), Qhp = cop > 1 + 1e-12 ? Qe / (1 - 1 / cop) : 0, drive = Qhp - Qe, Qdirect = Qneed - Qhp;
  const heatExt = Qdirect + (abs ? drive : 0), Wext = abs ? 0 : drive, carnotW = (T) => (p.etaTurb / 100) * Math.max(0, 1 - K(p.TcondPP) / K(T));
  return { abs, Th, Tl, Tg, Tsup, carnot, copRev, cop, Qneed, Qavail, Qe, Qhp, drive, Qdirect, heatExt, Wext, share: Qneed > 0 ? Qhp / Qneed : 0, cwSaving: Qavail > 0 ? Qe / Qavail : 0,
    sTh: heatExt / 1000 / R.Qd, sEl: (R.Wel + Wext) / 1000 / R.Qd, sEq: (R.Wel + Wext + Qdirect * carnotW(Tsup) + (abs ? drive * carnotW(Tg) : 0)) / 1000 / R.Qd, PR: heatExt > 1e-9 * (R.Md * 2326e3) ? (R.Md * 2326e3) / heatExt : null };
}

/** Capital-recovery factor for interest rate i (fraction per year) and n years. */
export const crf = (i, n) => (i > 1e-12 ? (i * (1 + i) ** n) / ((1 + i) ** n - 1) : 1 / n);

/** Thermodynamic–economic model: unit water cost from heat, electricity, annualised capital of the heat-transfer surface (and compressor), O&M and chemicals. Rates in $/h. */
export function waterCost(R, p, ex = {}) {
  const hrs = 8760 * clamp(p.avail / 100, 0.05, 1), A = crf(p.intRate / 100, p.life), capSurf = p.cArea * R.area * p.capFactor, capComp = ((R.Wcomp || 0) / 1000) * p.cComp * p.capFactor, capex = capSurf + capComp;
  const heat = ex.heat ?? R.Qin, power = ex.power ?? R.Wel, Zcap = (capex * A) / hrs, Zom = (capex * (p.omPct / 100)) / hrs, Cheat = (p.cHeat * heat) / 1e6, Cel = (p.cElec * power) / 1000, Cchem = p.cChem * R.Qd;
  const parts = { 'Capital recovery': Zcap / R.Qd, 'Heat': Cheat / R.Qd, 'Electricity': Cel / R.Qd, 'Operation and maintenance': Zom / R.Qd, 'Chemicals': p.cChem };
  return { hrs, crf: A, capex, capSurf, capComp, Zcap, Zom, Cheat, Cel, Cchem, parts, total: sum(Object.values(parts)), rate: Zcap + Zom + Cheat + Cel + Cchem, perCapacity: capex / (R.Qd * 24) };
}

/**
 * Exergy–economic (SPECO-type) analysis: the fuel is the exergy of the heat and electricity supplied, costed at its average unit cost
 * c_F; every component is charged the cost of the exergy it destroys (Ċ_D = c_F·Ėx_D) plus its share of the capital and O&M rate Ż;
 * the product is the minimum work of separation, so Ċ_P = Ċ_F + Ż and c_P = Ċ_P / Ėx_P.
 */
export function exergoEconomics(R, p, wc = waterCost(R, p)) {
  const CF = wc.Cheat + wc.Cel, ExF = R.exTot / 1000, cF = CF / ExF, ExP = R.wMin * R.Qd, Z = wc.Zcap + wc.Zom; // $/h, kW, $/kWh
  const comps = R.ex.map(([k, e]) => [k, e / 1000]), other = comps.find((c) => c[0].startsWith('Other'));
  if (other) { other[0] = 'Other (flash boxes, mixing, heat loss)'; other[1] = Math.max(0, other[1] - ExP); }
  // the itemised destructions are mean-temperature estimates; the total (fuel − product) is exact. When the estimates add up to more than the total they are scaled to it.
  const itemised = sum(comps.map((c) => c[1])), overshoot = itemised > ExF - ExP ? itemised / (ExF - ExP) - 1 : 0;
  if (overshoot > 0) for (const c of comps) c[1] *= (ExF - ExP) / itemised;
  comps.push(['Auxiliaries and unallocated remainder (pumps, vacuum system)', overshoot > 0 ? 0 : ExF - ExP - itemised]);
  const aux = Math.max(R.areaAux, 0), w = {}; // capital weights ($ of installed equipment) per component
  if (R.proc === 'msf') { w['Stage condensers (ΔT)'] = wc.capSurf * (R.areaEvap / R.area); w['Brine heater'] = wc.capSurf * (aux / R.area); }
  else if (R.proc === 'mvc') { w['Evaporator tubes (ΔT)'] = wc.capSurf * (R.areaEvap / R.area); w['Feed preheater'] = wc.capSurf * (aux / R.area); w['Compressor'] = wc.capComp; }
  else { const ac = R.hx[0]?.area || 0; w['Effect tubes (ΔT)'] = wc.capSurf * (R.areaEvap / R.area); w['Final condenser'] = wc.capSurf * (ac / R.area); w['Feed preheaters'] = wc.capSurf * (Math.max(0, aux - ac) / R.area); }
  const rows = comps.map(([k, e]) => { const Zk = (Z * (w[k] || 0)) / wc.capex, CD = cF * e; return { name: k, ExD: e, CD, Z: Zk, f: Zk + CD > 0 ? Zk / (Zk + Math.max(CD, 0)) : 0, share: e / ExF }; });
  const CP = CF + Z, cP = CP / ExP;
  return { CF, ExF, cF, ExP, Z, rows, overshoot, CP, cP, r: cF > 0 ? (cP - cF) / cF : 0, CDtot: sum(rows.map((q) => q.CD)), perM3: CP / R.Qd, closure: (sum(rows.map((q) => q.CD + q.Z)) + cF * ExP - CP) / CP };
}

/**
 * Thermal + RO hybrid sharing the seawater intake: the RO block (element-by-element model of suite 1) is fed with cold seawater or
 * with the warm cooling water rejected by the thermal block, and the two products are blended.
 */
export function hybridRO(R, p, tdsD, ionsF) {
  const share = clamp(p.roShare / 100, 0.02, 0.98), rec = clamp(p.roRec / 100, 0.1, 0.85), Qf = (R.Qd * share) / (1 - share) / rec, rhoS = density(R.T0, R.Xf);
  const Qrej = (R.Mcw * 3600) / density(R.Tcw, R.Xf), warm = p.roFeed === 'reject' ? Math.min(Qf, Qrej) : 0, Tro = clamp((warm * R.Tcw + (Qf - warm) * R.T0) / Qf, 1, 45);
  const size = roBlockSize(R.Qd, p); // elements per vessel follow the duty, so that a small block is not a full seven-element vessel at a fraction of its design flux
  if (size.below) throw infeasible(size.msg, 'range');
  const roAt = (T) => simulateRO({ ...defaultsOf(roSuite), ions: ionsF, salinityFactor: 1, Qf, T, pH: 8.1, recovery: 100 * rec, targetFlux: p.roFlux, design: 'auto', mode: 'recovery', nSeg: 2, elements: size.elements });
  const ro = roAt(Tro), cold = warm > 0 && Tro > R.T0 + 0.5 ? roAt(clamp(R.T0, 1, 45)) : null, Qp = ro.product.Q, tdsP = tds(ro.product.ions), Qtot = R.Qd + Qp;
  const QfTh = (R.Mf * 3600) / rhoS, intakeTh = ((R.Mf + R.Mcw) * 3600) / rhoS, intake = intakeTh + Qf - warm, Qb = (R.Mb * 3600) / density(R.Tbrine, R.Xb), tdsB = tdsFromSalinity(R.Xb, R.Tbrine), tdsC = tds(ro.conc.ions);
  return { ro, cold, share: Qp / Qtot, rec, Qf, Tro, warm, Qrej, Qp, tdsP, tdsD, Qtot, tdsBlend: (R.Qd * tdsD + Qp * tdsP) / Qtot, ionsBlend: Object.fromEntries(ION_IDS.map((k) => [k, (R.Qd * (ionsF[k] || 0) * (tdsD / Math.max(tds(ionsF), 1e-9)) + Qp * (ro.product.ions[k] || 0)) / Qtot])),
    QfTh, intakeTh, intake, intakeSaving: warm / (intakeTh + Qf), recFeed: Qtot / (QfTh + Qf), recIntake: Qtot / intake, Qbrine: Qb + ro.conc.Q, tdsBrine: (Qb * tdsB + ro.conc.Q * tdsC) / (Qb + ro.conc.Q),
    secEl: (R.Wel / 1000 + ro.power) / Qtot, secTh: R.Qin / 1000 / Qtot, secEq: (R.sEq * R.Qd + ro.power) / Qtot, overPressure: ro.p1.Pf > ro.cfg.M.pmax };
}

/**
 * Size of the RO block of a hybrid for a thermal capacity Qd (m³/h): elements needed at the design flux and elements per vessel.
 * The RO engine works with whole spiral-wound elements; a block that needs less than half an element lies below the range of the model.
 */
export function roBlockSize(Qd, p) {
  const d = defaultsOf(roSuite), share = clamp(p.roShare / 100, 0.02, 0.98), Qp = (Qd * share) / (1 - share), q1 = (p.roFlux * d.area) / 1000, need = Qp / q1; // q1: permeate of one element at the design flux, m³/h
  const elements = clamp(Math.ceil(need / Math.ceil(need / d.elements) - 1e-9), 1, d.elements), below = need < 0.5;
  return { Qp, q1, need, elements, below, msg: below ? `The RO block of the hybrid would produce only ${fmt(Qp * 24, 3)} m³/d, below the range of the RO model: one spiral-wound element delivers about ${fmt(q1 * 24, 3)} m³/d at ${fmt(p.roFlux, 3)} L/m²·h. Raise the RO share to at least ${fmt(Math.ceil((1000 * 0.5 * q1) / (Qd + 0.5 * q1)) / 10, 3)} % or the distillate capacity to at least ${fmt(Math.ceil((0.5 * q1 * (1 - share) * 24) / share), 3)} m³/d, or switch the hybrid off.` : '' };
}

/** Grid of fast re-designs over the two main design variables of the selected process (number of effects/stages × top temperature, or MVC temperature × ΔT). */
export function designGrid(v, N = Math.max(1, Math.round(v.proc === 'msf' ? v.Nst : v.proc === 'mvc' ? v.Nmvc : v.N)), empty = false) {
  const p = v, proc = p.proc, tryRun = empty ? () => null : tryRunFull; // empty: axes only, no re-designs
  if (proc === 'msf') {
    const xs = [12, 16, 20, 24, 28, 32].filter((n) => n > p.nRej + 1), ys = linspace(88, 118, 6);
    return { xs, ys, x0: N, y0: p.TBT, xlabel: 'Number of stages', ylabel: 'Top brine temperature (°C)', title: 'Performance ratio versus stage count and top brine temperature', zlabel: 'Performance ratio', zunit: 'kg/2326 kJ', cmap: 'viridis', metric: (q) => q.PR, integerX: true, make: (n, T) => tryRun(p, { Nst: n, TBT: T, fast: true }), cells: ys.map((T) => xs.map((n) => tryRun(p, { Nst: n, TBT: T, fast: true }))) };
  }
  if (proc === 'mvc') {
    const xs = linspace(45, 70, 6), ys = linspace(1.5, 5, 5);
    return { xs, ys, x0: p.Tmvc, y0: p.dTmvc, xlabel: 'First-effect brine temperature (°C)', ylabel: 'Condensing − boiling ΔT (K)', title: 'Specific electricity versus evaporation temperature and ΔT', zlabel: 'Specific electricity', zunit: 'kWh/m³', cmap: 'thermal', metric: (q) => q.sEl, integerX: false, make: (T, d) => tryRun(p, { Tmvc: T, dTmvc: d, fast: true }), cells: ys.map((d) => xs.map((T) => tryRun(p, { Tmvc: T, dTmvc: d, fast: true }))) };
  }
  const xs = [4, 6, 9, 12], ys = linspace(58, 74, 4);
  return { xs, ys, x0: N, y0: p.Ts, xlabel: 'Number of effects', ylabel: 'Heating-steam temperature (°C)', title: 'Gain-output ratio versus effects and heating-steam temperature', zlabel: 'GOR', zunit: 'kg/kg', cmap: 'viridis', metric: (q) => q.GOR, integerX: true, make: (n, T) => tryRun(p, { N: n, Ts: T, fast: true, nEnt: p.nEnt > 0 ? Math.max(1, Math.round((p.nEnt * n) / N)) : 0 }), cells: ys.map((T) => xs.map((n) => tryRun(p, { N: n, Ts: T, fast: true, nEnt: p.nEnt > 0 ? Math.max(1, Math.round((p.nEnt * n) / N)) : 0 }))) };
}

/** Process optimisation: constrained grid search for the lowest unit water cost over the design grid (top brine temperature ≤ scale limit); the base design is a candidate. */
export function optimiseGrid(grid, p, tLim, baseCost) {
  const cost = grid.cells.map((row) => row.map((q) => (q ? waterCost(q, p).total : null)));
  let best = { cost: baseCost, x: grid.x0, y: grid.y0, base: true }, nFeas = 0, nCells = 0;
  grid.cells.forEach((row, j) => row.forEach((q, i) => { if (!q) return; nCells++; if (q.Ttop > tLim + 1e-9) return; nFeas++; if (cost[j][i] < best.cost) best = { cost: cost[j][i], x: grid.xs[i], y: grid.ys[j], base: false, q }; }));
  return { cost, best, nFeas, nCells, saving: baseCost > 0 ? 1 - best.cost / baseCost : 0 };
}

const calWarm = new Map(); // converged trains of the calibration model, one per operating point
/**
 * Process optimisation proper: lowest unit water cost over the two design variables of the map, subject to the scale limit on the top brine temperature.
 * Starts from the best cell of the grid. A whole-number variable (effects, stages) is enumerated around the start and the continuous one (temperature)
 * minimised for the best count with a bounded Nelder–Mead simplex; two continuous variables (MVC) are searched together. Infeasible or unconverged designs
 * are penalised. The result is never worse than the grid optimum.
 */
export function optimiseDesign(grid, p, tLim, gridOpt) {
  const xLo = Math.min(...grid.xs), xHi = Math.max(...grid.xs), yLo = Math.min(...grid.ys), yHi = Math.max(...grid.ys), start = gridOpt.best, big = 1e3 * Math.max(gridOpt.best.cost, 1e-6);
  let best = { ...start }, evals = 0;
  const f = (x, y) => { const q = grid.make(x, y); evals++; if (!q) return big; const c = waterCost(q, p).total; if (!Number.isFinite(c)) return big; if (q.Ttop > tLim + 1e-9) return big * (1 + (q.Ttop - tLim) / 100); if (c < best.cost) best = { cost: c, x, y, base: false, q }; return c; };
  if (grid.integerX) {
    const x0 = clamp(Math.round(start.x), xLo, xHi), y0 = clamp(start.y, yLo, yHi), cand = [x0, x0 - 1, x0 + 1].filter((x, k, a) => x >= Math.max(1, xLo) && x <= xHi && a.indexOf(x) === k);
    const at0 = cand.map((x) => f(x, y0)), xb = cand[at0.indexOf(Math.min(...at0))]; // best whole number at the starting temperature, then the temperature for it
    nelderMead((u) => f(xb, u[0]), [y0], { lo: [yLo], hi: [yHi], tol: 1e-4, maxIter: 8, scale: 0.12 });
  } else nelderMead((u) => f(u[0], u[1]), [clamp(start.x, xLo, xHi), clamp(start.y, yLo, yHi)], { lo: [xLo, yLo], hi: [xHi, yHi], tol: 1e-5, maxIter: 30, scale: 0.1 });
  return { best, evals, improved: best.cost < gridOpt.best.cost - 1e-12, saving: gridOpt.best.cost > 0 ? 1 - best.cost / gridOpt.best.cost : 0, method: grid.integerX ? 'whole-number variable enumerated, temperature by Nelder–Mead' : 'Nelder–Mead over both variables' };
}
const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const isMed = (v) => v.proc === 'med' || v.proc === 'medtvc', PROC = { med: 'MED', medtvc: 'MED-TVC', msf: 'MSF', mvc: 'MVC' };
/** Adds the nearest design that does solve (same inputs except one) to the message of an infeasibility error. */
function withHint(e, p, ov) {
  const ok = (o) => { try { return simulateThermal(p, { ...ov, ...o, probe: true, fast: true }).converged; } catch { return false; } };
  const N = Math.max(1, Math.round(p.proc === 'msf' ? p.Nst : p.proc === 'mvc' ? p.Nmvc : p.N)), nKey = p.proc === 'msf' ? 'Nst' : p.proc === 'mvc' ? 'Nmvc' : 'N', unit = p.proc === 'msf' ? 'stages' : 'effects', hints = [];
  const withN = (n) => ({ [nKey]: n, ...(p.proc === 'medtvc' && p.nEnt > 0 ? { nEnt: Math.max(1, Math.min(Math.round(p.nEnt), n)) } : {}) });
  if (e.kind === 'ejector') { for (let k = Math.round(p.nEnt) + 1; k <= N; k++) if (ok({ nEnt: k })) { hints.push(`suction from effect ${k}`); break; } }
  if (e.kind === 'stage') { for (const n of [N + 2, N + 4, N + 8, N + 12].filter((q) => q <= 45)) if (ok({ Nst: n })) { hints.push(`${n} stages`); break; } for (const d of [3, 6, 10, 15]) if (p.TBT + d <= 125 && ok({ TBT: p.TBT + d })) { hints.push(`a top brine temperature of ${fmt(p.TBT + d, 4)} °C`); break; } }
  if (e.kind === 'reject') for (const d of [1, 2, 3, 5, 8, 12]) if (p.TnMsf + d <= 60 && ok({ TnMsf: p.TnMsf + d })) { hints.push(`a last-stage temperature of ${fmt(p.TnMsf + d, 4)} °C`); break; }
  if (['window', 'converge', 'dry', 'salt', 'ejector'].includes(e.kind) && N > 1 && !hints.length) { // largest smaller train that solves (feasibility grows as effects are removed)
    let lo = 0, hi = N - 1;
    if (ok(withN(hi))) lo = hi; else if (ok(withN(1))) { lo = 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ok(withN(m))) lo = m; else hi = m; } }
    if (lo >= 1) hints.push(`${lo} ${lo > 1 ? unit : unit.slice(0, -1)}`);
  }
  if (e.kind === 'window' && p.proc !== 'mvc') for (const d of [3, 6, 10, 15, 25, 40]) if (p.Ts + d <= 130 && ok({ Ts: p.Ts + d })) { hints.push(`a heating-steam temperature of ${fmt(p.Ts + d, 4)} °C`); break; }
  if (e.kind === 'salt' || e.kind === 'window') for (const x of [0.8, 0.6, 0.45]) if (p.Xb * x > 1.3 * p.Xf && p.Xb * x >= 40 && ok({ Xb: Math.round(p.Xb * x) })) { hints.push(`a maximum brine salinity of ${Math.round(p.Xb * x)} g/kg`); break; }
  return hints.length ? infeasible(`${e.message} A design with ${hints.join(', or with ')} (other inputs unchanged) can be solved.`, e.kind) : e;
}
const tryRunFull = (v, ov) => { try { const q = simulateThermal(v, ov); return q.converged ? q : null; } catch { return null; } }, tryRun = tryRunFull;
const nz = (x) => (Number.isFinite(x) ? x : null);

const suite = {
  id: 'thermal', num: 6, title: 'Thermal Desalination', short: 'Thermal (MED/MSF)', icon: '♨️',
  tagline: 'Effect-by-effect and stage-by-stage design of MED, MED-TVC, MSF and MVC plants with heat-transfer sizing, energy and exergy accounting.',
  description: 'Solves the steady-state total-mass, salt and enthalpy balances of every effect or flashing stage, including boiling-point elevation, non-equilibrium allowance, demister and vapour-line losses, flashing of brine and distillate, feed preheaters and the final condenser. Heat-transfer areas follow from published U(T) correlations with fouling (LMTD and effectiveness–NTU); thermo-compressors use the Power-chart entrainment correlation and mechanical compressors an isentropic-efficiency model. Gain-output ratio, performance ratio, specific energy, cooling water, exergy destruction and a lumped start-up transient are reported together with part-load and seawater-temperature sweeps.',
  guide: [
    'Choose the process (MED, MED-TVC, MSF or MVC), the distillate capacity and the seawater condition — or pull the case feed, the site sea temperature or an RO concentrate for a hybrid.',
    'Set the temperature window (heating steam or top brine temperature and the last effect/stage) and the number of effects or stages.',
    'On Model setup choose equal-ΔT, equal-area or equal-heat-flux design, the boiling-point-elevation basis, fouling, temperature losses, scale limits, an optional RO hybrid and the costing inputs.',
    'Run. Check the per-effect table, the scale warnings and the balances on the Verify tab; the distillate, brine and energy figures are offered to the plant, ZLD and economics suites.',
  ],
  implemented: ['total and component mass', 'energy balance', 'steam-table', 'vapour-liquid equilibrium', 'boiling-point-elevation', 'antoine', 'clausius-clapeyron', 'flash-vaporization', 'heat-exchanger equation', 'overall heat-transfer', 'logarithmic-mean', 'effectiveness-ntu', 'condensation equation', 'evaporation equation', 'latent-heat', 'non-equilibrium allowance', 'demister pressure-drop', 'compressor equation', 'exergy equation', 'raoult',
    'multi-effect mass-energy', 'multi-stage-flash stagewise', 'thermal-vapour-compression', 'mechanical-vapour-compression', 'med-tvc', 'solar-thermal desalination', 'waste-heat-desalination', 'med-ro', 'msf-ro', 'heat-pump-desalination', 'thermal-electrical cogeneration', 'thermodynamic-economic', 'exergy-economic',
    'stage/effect temperature', 'pressures', 'salinities', 'liquid inventories', 'vapour inventories', 'wall temperature', 'heat-flux condition', 'feed-flow', 'feed-temperature and feed-salinity', 'heating-steam', 'condenser cooling-water', 'terminal vacuum', 'vapour-liquid interfacial', 'product/brine outlet',
    'feed-water thermodynamics', 'evaporation', 'condensation', 'flashing', 'boiling-point elevation', 'heat transfer', 'heat-exchanger modelling', 'multi-stage and multi-effect', 'vapour compression', 'steam and utility', 'vacuum-system', 'brine recirculation', 'heat recovery', 'scaling assessment', 'condenser modelling', 'thermal-energy consumption', 'electrical-energy consumption', 'equipment sizing', 'transient operation', 'start-up and shutdown', 'waste-heat integration', 'solar-thermal integration', 'process optimisation', 'performance-ratio assessment'],
  equationsNote: 'Steady-state design model with saturated heating steam. Seawater properties follow Sharqawy et al.; U(T), demister, non-equilibrium and CaSO₄-envelope correlations follow El-Dessouky & Ettouney and are valid for roughly 30–120 °C; the ejector fit holds for compression ratios ≥ 1.81 and entrainment ratios ≤ 4. Part-load results of MED/MVC are true ratings at fixed area, the MSF turndown curve is a re-design at lower top brine temperature. Start-up and cool-down after a steam trip are lumped thermal-inertia estimates (no vacuum pull-down, venting or level control). The boiling-point elevation can be taken from the seawater correlation or from Raoult’s law (p = a_w·p_sat with the water mole fraction or the osmotic-coefficient activity); all three are tabulated. MED–RO / MSF–RO hybrids couple this model to the element-by-element RO engine of suite 1 through a shared intake (optionally the warm cooling-water reject) and product blending; the RO block uses the default seawater membrane and automatic array sizing and is not co-optimised with the thermal block. Heat pumps are reduced-order (second-law efficiency × reversible COP, no working-fluid cycle); cogeneration is a saturated-steam turbine expansion with one isentropic efficiency and the lost-work (power-loss) allocation, limited to 45 bar throttle pressure by the property correlations. The thermodynamic–economic cost is a single-year annualised model (surface cost × installation factor, heat, electricity, O&M, chemicals); the exergy–economic table applies one average fuel-exergy cost to every component (SPECO-type aggregate, no stream-by-stream cost matrix). Process optimisation: a grid of equal-ΔT re-designs gives the cost map and the starting point, then a bounded Nelder–Mead simplex minimises the unit water cost over the continuous variable (heating-steam or top brine temperature) for the best neighbouring whole number of effects or stages — over both variables for MVC — with designs above the scale limit penalised; every evaluation is a fast (equal-ΔT) re-design, so the optimum is that of the simplified sizing rule. The vapour inventory is a saturated ideal-gas hold-up added to the lumped start-up model. Scaling is screened with a CaSO₄ solubility envelope and top-temperature limits only — use suite 2 for speciation. Non-condensable gases enter only as a venting allowance; tube-bundle geometry and wetting rates are not resolved.',

  inputs: [
    { group: 'Process and capacity', help: 'Which thermal process is designed and for how much distillate.', fields: [
      { key: 'proc', label: 'Process', type: 'select', value: 'med', options: [{ value: 'med', label: 'MED — multi-effect distillation' }, { value: 'medtvc', label: 'MED-TVC — with steam-jet thermo-compressor' }, { value: 'msf', label: 'MSF — multi-stage flash' }, { value: 'mvc', label: 'MVC — mechanical vapour compression' }], help: 'Each process shows its own inputs. The examples load realistic settings for every process.' },
      { key: 'Md', label: 'Distillate capacity', unit: 'm³/d', value: 20000, min: 10, max: 1e6, typical: [500, 90000], help: 'Net distillate production the plant is designed for.' },
    ] },
    { group: 'Seawater feed', help: 'Condition of the water entering the plant (seawater or, for a hybrid, RO concentrate).', fields: [
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.seawater.ions, help: 'Sets the ion ratios of the brine and distillate streams passed to other suites; the salinity below sets the total.' },
      { key: 'Xf', label: 'Feed salinity', unit: 'g/kg', value: 36, min: 1, max: 120, typical: [30, 70], help: 'Total dissolved salts per kg of feed.' },
      { key: 'Tsw', label: 'Seawater temperature', unit: '°C', value: 26, min: 2, max: 40, typical: [12, 35], help: 'Cooling-water and feed inlet temperature; also the dead state of the exergy analysis.' },
      { key: 'Xb', label: 'Maximum brine salinity', unit: 'g/kg', value: 60, min: 5, max: 200, typical: [50, 72], help: 'Blow-down salinity; with the feed salinity it fixes the recovery. Above about 160 g/kg the seawater property correlations are extrapolated; 200 g/kg is the limit of the model.', showIf: (v) => !(v.proc === 'msf' && v.msfType === 'ot') },
    ] },
    { group: 'Multi-effect train', help: 'Temperature window and arrangement of the effects.', showIf: isMed, fields: [
      { key: 'N', label: 'Number of effects', unit: '', value: 10, min: 1, max: 24, step: 1, typical: [4, 16], help: 'More effects re-use the latent heat more often (higher GOR) but need more area.' },
      { key: 'Ts', label: 'Heating-steam temperature at effect 1', unit: '°C', value: 70, min: 45, max: 130, typical: [60, 72], help: 'Condensing temperature inside the first-effect tubes. For MED-TVC this is the ejector discharge.' },
      { key: 'Tn', label: 'Last-effect brine temperature', unit: '°C', value: 40, min: 25, max: 80, typical: [36, 45], help: 'Fixed by the vacuum of the final condenser; must stay several kelvin above the seawater.' },
      { key: 'arr', label: 'Feed arrangement', type: 'select', value: 'parallel', options: [{ value: 'parallel', label: 'Parallel / cross feed (brine cascades and flashes)' }, { value: 'forward', label: 'Forward feed (all feed to effect 1)' }], help: 'Parallel/cross feed distributes the feed over all effects; forward feed sends everything to the hottest effect.' },
      { key: 'preheat', label: 'Feed preheaters between effects', type: 'bool', value: true, help: 'Part of each effect’s vapour preheats the feed, raising the gain-output ratio.' },
      { key: 'Pm', label: 'Motive-steam pressure', unit: 'bar', value: 10, min: 1.5, max: 35, typical: [2.5, 25], help: 'Saturated motive steam supplied to the ejector.', showIf: (v) => v.proc === 'medtvc' },
      { key: 'nEnt', label: 'Ejector suction taken from effect', unit: '', value: 0, min: 0, max: 24, step: 1, help: '0 = last effect. Suction from a middle effect lowers the compression ratio and raises the entrainment.', showIf: (v) => v.proc === 'medtvc' },
    ] },
    { group: 'Multi-stage flash', showIf: (v) => v.proc === 'msf', fields: [
      { key: 'msfType', label: 'Configuration', type: 'select', value: 'br', options: [{ value: 'br', label: 'Brine recirculation (heat recovery + heat rejection)' }, { value: 'ot', label: 'Once-through' }], help: 'Brine recirculation re-uses most of the flashed brine and rejects heat in the last stages; once-through passes the feed a single time.' },
      { key: 'Nst', label: 'Number of stages', unit: '', value: 24, min: 3, max: 45, step: 1, typical: [16, 28], help: 'Total number of flashing stages (heat recovery plus heat rejection).' },
      { key: 'nRej', label: 'Heat-rejection stages', unit: '', value: 3, min: 1, max: 6, typical: [2, 4], step: 1, showIf: (v) => v.msfType === 'br', help: 'Stages cooled by seawater instead of recirculating brine.' },
      { key: 'TBT', label: 'Top brine temperature', unit: '°C', value: 110, min: 60, max: 125, typical: [90, 112], help: 'Brine temperature leaving the brine heater; limited by the scale-control method.' },
      { key: 'TnMsf', label: 'Last-stage brine temperature', unit: '°C', value: 40, min: 25, max: 60, typical: [35, 42], help: 'Brine temperature in the last, coldest stage.' },
      { key: 'dTsteam', label: 'Heating steam above top brine temperature', unit: 'K', value: 7, min: 2, max: 30, typical: [5, 10], help: 'Condensing-steam temperature in the brine heater minus the top brine temperature.' },
    ] },
    { group: 'Mechanical vapour compression', showIf: (v) => v.proc === 'mvc', fields: [
      { key: 'Nmvc', label: 'Number of effects', unit: '', value: 1, min: 1, max: 6, typical: [1, 4], step: 1, help: 'Effects in series between the compressor discharge and suction.' },
      { key: 'Tmvc', label: 'First-effect brine temperature', unit: '°C', value: 60, min: 40, max: 100, typical: [50, 70], help: 'Boiling temperature of the brine in the first effect.' },
      { key: 'dTmvc', label: 'Condensing − boiling temperature difference', unit: 'K', value: 2.5, min: 0.8, max: 8, typical: [1.5, 4], help: 'Smaller differences save compressor power but need more evaporator area.' },
      { key: 'etaIs', label: 'Compressor isentropic efficiency', unit: '%', value: 76, min: 40, max: 92, typical: [70, 85], help: 'Ratio of isentropic to actual compression work.' },
      { key: 'etaMotor', label: 'Motor + drive efficiency', unit: '%', value: 94, min: 60, max: 99, typical: [92, 97], help: 'Electric motor, drive and gearbox combined.' },
      { key: 'Uph', label: 'Feed-preheater overall U', unit: 'kW/m²·K', value: 1.8, min: 0.3, max: 6, typical: [1.2, 3], help: 'Plate exchangers recovering heat from product and brine.' },
    ] },
    { group: 'Heat source', help: 'Where the heat comes from and how it is valued.', fields: [
      { key: 'source', label: 'Heat source', type: 'select', value: 'steam', options: [{ value: 'steam', label: 'Boiler or turbine-extraction steam' }, { value: 'cogen', label: 'Power-plant cogeneration (turbine extraction, lost-work allocation)' }, { value: 'mhp', label: 'Mechanical (electric) heat pump on the rejected heat' }, { value: 'ahp', label: 'Absorption heat pump on the rejected heat' }, { value: 'solar', label: 'Solar-thermal collector field' }, { value: 'waste', label: 'Industrial waste-heat stream' }], help: 'Adds the matching integration model to the results: turbine-extraction cogeneration, a heat pump that upgrades the condenser heat, a solar field or a waste-heat match.' },
      { key: 'Pthr', label: 'Turbine throttle pressure (saturated steam)', unit: 'bar', value: 40, min: 3, max: 45, showIf: (v) => v.source === 'cogen', help: 'Saturated steam at this pressure expands in the turbine; the heating steam is extracted at the pressure the desalination plant needs. Property correlations limit the model to 45 bar.' },
      { key: 'Ppp', label: 'Gross power without extraction', unit: 'MW', value: 300, min: 1, max: 3000, typical: [50, 1500], showIf: (v) => v.source === 'cogen', help: 'Rating of the steam turbine when all steam expands to the condenser.' },
      { key: 'etaBoiler', label: 'Boiler efficiency', unit: '%', value: 90, min: 50, max: 99, showIf: (v) => v.source === 'cogen', help: 'Fuel heat converted into steam enthalpy.' },
      { key: 'hpEta', label: 'Heat-pump second-law efficiency', unit: '%', value: 55, min: 10, max: 100, showIf: (v) => v.source === 'mhp' || v.source === 'ahp', help: 'Mechanical: share of the Carnot COP reached. Absorption: share of the reversible three-temperature gain (COP − 1) reached. 100 % gives the thermodynamic limit.' },
      { key: 'hpApproach', label: 'Heat-pump exchanger approach', unit: 'K', value: 3, min: 0, max: 15, showIf: (v) => v.source === 'mhp' || v.source === 'ahp', help: 'Temperature difference in the heat-pump evaporator and condenser/absorber.' },
      { key: 'hpTg', label: 'Driving-heat temperature (generator)', unit: '°C', value: 150, min: 80, max: 250, showIf: (v) => v.source === 'ahp', help: 'Temperature of the steam or hot gas that drives the absorption heat pump.' },
      { key: 'ghi', label: 'Daily solar irradiation', unit: 'kWh/m²·d', value: 5.8, min: 0.5, max: 10, showIf: (v) => v.source === 'solar', help: 'Annual-mean global horizontal irradiation at the site.' },
      { key: 'collector', label: 'Collector type', type: 'select', value: 'etc', options: Object.entries(COLLECTORS).map(([k, c]) => ({ value: k, label: c.name })), showIf: (v) => v.source === 'solar', help: 'Sets the optical efficiency and heat-loss coefficients of the collector.' },
      { key: 'solarFrac', label: 'Solar fraction of the heat demand', unit: '%', value: 70, min: 5, max: 100, showIf: (v) => v.source === 'solar', help: 'Share of the daily heat demand supplied by the collector field.' },
      { key: 'Tamb', label: 'Ambient air temperature', unit: '°C', value: 28, min: -10, max: 50, showIf: (v) => v.source === 'solar', help: 'Collector heat losses grow with the difference to ambient temperature.' },
      { key: 'Twh', label: 'Waste-heat stream temperature', unit: '°C', value: 95, min: 40, max: 400, showIf: (v) => v.source === 'waste', help: 'Supply temperature of the waste-heat carrier.' },
      { key: 'Qwh', label: 'Waste-heat duty available', unit: 'MW', value: 40, min: 0.01, max: 5000, showIf: (v) => v.source === 'waste', help: 'Heat that can be extracted from the waste-heat stream.' },
      { key: 'TcondPP', label: 'Reference power-plant condenser temperature', unit: '°C', value: 38, min: 20, max: 60, help: 'Used to convert heat into the electricity a turbine would have produced from it (equivalent-electric energy).' },
      { key: 'etaTurb', label: 'Turbine efficiency relative to Carnot', unit: '%', value: 85, min: 40, max: 100, help: 'Fraction of the Carnot work a real steam turbine would deliver.' },
    ] },
    { group: 'Heat-transfer model', tab: 'setup', help: 'How the heat-transfer areas are obtained.', fields: [
      { key: 'areaMode', label: 'Effect sizing rule', type: 'select', value: 'equalArea', options: [{ value: 'equalArea', label: 'Equal area in every effect (iterated)' }, { value: 'equalDT', label: 'Equal temperature difference in every effect' }, { value: 'flux', label: 'Equal heat flux in every effect (prescribed-flux condition)' }], showIf: (v) => v.proc !== 'msf', help: 'Equal-area designs are the industrial standard; the temperature differences are iterated until all areas match. The heat-flux option instead imposes the same heat flux q″ = U·ΔT on every effect.' },
      { key: 'fU', label: 'Heat-transfer coefficient multiplier', unit: '×', value: 1, min: 0.3, max: 2, typical: [0.7, 1.2], help: 'Scales the published U(T) correlations. Calibrate against plant data.' },
      { key: 'Rf', label: 'Additional fouling resistance', unit: 'm²·K/kW', value: 0.03, min: 0, max: 0.5, typical: [0, 0.09], help: 'Added in series to the correlation value: 1/U = 1/U₀ + R_f.' },
      { key: 'heatLoss', label: 'Heat loss to surroundings', unit: '% of duty', value: 1, min: 0, max: 15, typical: [0.5, 3], help: 'Share of every effect or brine-heater duty lost through the insulation.' },
      { key: 'ttdCond', label: 'Final-condenser / feed-heater approach', unit: 'K', value: 3, min: 0.5, max: 15, typical: [2, 5], showIf: (v) => v.proc !== 'msf', help: 'Condensing temperature minus seawater outlet temperature. For MVC: minimum approach of the feed preheater.' },
      { key: 'ttdPh', label: 'Feed-preheater terminal difference', unit: 'K', value: 3, min: 0.5, max: 15, typical: [2, 5], showIf: (v) => isMed(v) && v.preheat, help: 'Condensing vapour temperature minus feed outlet temperature of each preheater.' },
      { key: 'ttdRej', label: 'Heat-rejection terminal difference', unit: 'K', value: 3, min: 0.5, max: 15, typical: [2, 5], showIf: (v) => v.proc === 'msf' && v.msfType === 'br', help: 'Condensing temperature of the first rejection stage minus the seawater leaving it.' },
    ] },
    { group: 'Temperature losses and venting', tab: 'setup', help: 'Thermodynamic penalties between boiling brine and condensing vapour.', fields: [
      { key: 'bpeModel', label: 'Boiling-point-elevation basis', type: 'select', value: 'corr', options: [{ value: 'corr', label: 'Seawater correlation (Sharqawy)' }, { value: 'activity', label: 'Raoult’s law with water activity (osmotic coefficient)' }, { value: 'ideal', label: 'Raoult’s law, ideal (water mole fraction)' }], help: 'Raoult’s law: the brine boils where a_w·p_sat(T) equals the stage pressure. All three bases are always compared in the results; this choice selects the one used in the balances.' },
      { key: 'fBPE', label: 'Boiling-point-elevation multiplier', unit: '×', value: 1, min: 0, max: 2, help: '0 gives the ideal pure-water limit.' },
      { key: 'fNEA', label: 'Non-equilibrium-allowance multiplier', unit: '×', value: 1, min: 0, max: 3, showIf: (v) => v.proc === 'msf', help: 'Scales the non-equilibrium allowance correlation of the flashing stages.' },
      { key: 'neaH', label: 'Flashing-brine pool depth', unit: 'm', value: 0.3, min: 0.05, max: 1, showIf: (v) => v.proc === 'msf', help: 'Depth of the flashing brine above the stage floor.' },
      { key: 'neaVb', label: 'Brine load per metre of stage width', unit: 'kg/m·s', value: 180, min: 30, max: 400, showIf: (v) => v.proc === 'msf', help: 'Flashing-brine mass flow per metre of stage width.' },
      { key: 'neaL', label: 'Stage length', unit: 'm', value: 3, min: 0.5, max: 8, showIf: (v) => v.proc === 'msf', help: 'Length of a stage in the flow direction.' },
      { key: 'rhoP', label: 'Demister packing density', unit: 'kg/m³', value: 150, min: 80, max: 208, help: 'Wire-mesh packing density (correlation range 80–208 kg/m³).' },
      { key: 'Vdem', label: 'Vapour velocity in demister', unit: 'm/s', value: 4, min: 0.9, max: 7.5, help: 'Superficial vapour velocity through the mesh (correlation range 0.98–7.5 m/s).' },
      { key: 'dw', label: 'Demister wire diameter', unit: 'mm', value: 0.28, min: 0.2, max: 0.32, help: 'Mesh wire diameter (correlation range 0.2–0.32 mm).' },
      { key: 'thick', label: 'Demister thickness', unit: 'mm', value: 150, min: 50, max: 300, help: 'Thickness of the demister pad.' },
      { key: 'linePct', label: 'Vapour-line pressure drop', unit: '% of stage pressure', value: 0.5, min: 0, max: 5, help: 'Friction loss of the vapour between demister and condensing surface.' },
      { key: 'ventPct', label: 'Vapour lost with vented gases', unit: '% of condensed vapour', value: 0.5, min: 0, max: 5, typical: [0.2, 1], help: 'Vapour leaving the cold end with the non-condensable gases through the vacuum system.' },
      { key: 'carry', label: 'Brine carry-over through demisters', unit: 'mg brine per kg vapour', value: 150, min: 0, max: 5000, help: 'Sets the distillate salinity.' },
    ] },
    { group: 'Scale limits', tab: 'setup', fields: [
      { key: 'tbtLimMED', label: 'Maximum top brine temperature, MED / MVC', unit: '°C', value: 70, min: 55, max: 130, showIf: (v) => v.proc !== 'msf', help: 'Above about 70 °C calcium-sulphate and soft scale form on falling-film tubes.' },
      { key: 'antiscalant', label: 'MSF scale control', type: 'select', value: 'ht', options: Object.entries(ANTISCALANT).map(([k, a]) => ({ value: k, label: `${a.name} (≤ ${a.tbt} °C)` })), showIf: (v) => v.proc === 'msf', help: 'Sets the highest allowable top brine temperature.' },
    ] },
    { group: 'Pumps', tab: 'setup', help: 'Differential pressures used for the electrical pumping estimate.', fields: [
      { key: 'dpSea', label: 'Seawater supply', unit: 'bar', value: 2.5, min: 0, max: 15, help: 'Head of the seawater supply pumps.' }, { key: 'dpRec', label: 'Brine recirculation (MSF)', unit: 'bar', value: 5, min: 0, max: 15, showIf: (v) => v.proc === 'msf', help: 'Head of the brine-recirculation pumps.' },
      { key: 'dpProd', label: 'Distillate extraction', unit: 'bar', value: 3, min: 0, max: 15, help: 'Head of the distillate pumps (suction is under vacuum).' }, { key: 'dpBrine', label: 'Brine blow-down', unit: 'bar', value: 2.5, min: 0, max: 15, help: 'Head of the blow-down pumps (suction is under vacuum).' },
      { key: 'etaPump', label: 'Pump + motor efficiency', unit: '%', value: 75, min: 30, max: 92, help: 'Combined hydraulic and motor efficiency.' },
    ] },
    { group: 'Start-up transient (initial conditions)', tab: 'setup', help: 'All effects/stages start at seawater temperature; the heating steam is ramped up linearly.', fields: [
      { key: 'transient', label: 'Simulate start-up and shutdown', type: 'bool', value: true, help: 'Integrates the warm-up of every effect/stage from a cold start.' },
      { key: 'cth', label: 'Thermal mass per m² of heat-transfer surface', unit: 'kJ/m²·K', value: 25, min: 2, max: 600, showIf: (v) => v.transient, help: 'Tubes, shell, internals and wetted film, referred to the heat-transfer area.' },
      { key: 'vapVol', label: 'Vapour-space volume per m² of heat-transfer surface', unit: 'm³/m²', value: 0.03, min: 0, max: 0.5, showIf: (v) => v.transient, help: 'Vapour inventory of each effect/stage: it starts saturated at seawater temperature and its mass and latent energy grow as the unit warms up.' },
      { key: 'holdup', label: 'Brine hold-up time per effect/stage', unit: 's', value: 90, min: 0, max: 900, showIf: (v) => v.transient, help: 'Liquid inventory expressed as residence time of the brine flow.' },
      { key: 'ramp', label: 'Steam ramp-up time', unit: 'min', value: 30, min: 1, max: 600, showIf: (v) => v.transient, help: 'Time over which the heating steam is brought to its design flow.' },
      { key: 'tEnd', label: 'Simulated time', unit: 'h', value: 4, min: 0.2, max: 48, showIf: (v) => v.transient, help: 'Length of the simulated start-up; it is extended automatically if the plant is not yet warm.' },
    ] },
    { group: 'Hybrid with reverse osmosis', tab: 'setup', help: 'MED–RO / MSF–RO (or MVC–RO) plant: a reverse-osmosis block shares the seawater intake and the two products are blended.', fields: [
      { key: 'hybrid', label: 'Plant configuration', type: 'select', value: 'none', options: [{ value: 'none', label: 'Thermal plant only' }, { value: 'ro', label: 'Thermal + RO hybrid with product blending' }], help: 'The RO block is solved element by element with the engine of suite 1 (default seawater membrane, automatic array sizing).' },
      { key: 'roShare', label: 'RO share of the total product', unit: '%', value: 50, min: 5, max: 95, typical: [20, 80], showIf: (v) => v.hybrid === 'ro', help: 'The thermal capacity above stays as entered; the RO block is sized to supply this share of the blended product.' },
      { key: 'roRec', label: 'RO recovery', unit: '%', value: 42, min: 10, max: 85, typical: [35, 50], showIf: (v) => v.hybrid === 'ro', help: 'Permeate ÷ RO feed.' },
      { key: 'roFlux', label: 'RO design flux', unit: 'L/m²·h', value: 14, min: 5, max: 35, typical: [12, 17], showIf: (v) => v.hybrid === 'ro', help: 'Average flux used to size the RO array.' },
      { key: 'roFeed', label: 'RO feed taken from', type: 'select', value: 'reject', options: [{ value: 'reject', label: 'Warm cooling-water reject of the thermal block' }, { value: 'intake', label: 'Cold seawater from the shared intake' }], showIf: (v) => v.hybrid === 'ro', help: 'Feeding the RO with the warm reject saves intake flow and lowers the RO pressure.' },
      { key: 'tdsLimit', label: 'Blended-product TDS limit', unit: 'mg/L', value: 500, min: 20, max: 2000, showIf: (v) => v.hybrid === 'ro', help: 'Blending with distillate lets a single-pass RO meet this limit.' },
    ] },
    { group: 'Costing (thermodynamic–economic and exergy–economic)', tab: 'setup', help: 'Prices and capital factors for the unit water cost, the exergy cost rates and the cost-optimal design on the sensitivity map.', fields: [
      { key: 'cHeat', label: 'Price of heat', unit: '$/MWh', value: 8, min: 0, max: 200, help: 'Value of the heating steam per MWh of heat (about 2 $/GJ for low-pressure extraction steam).' },
      { key: 'cElec', label: 'Price of electricity', unit: '$/kWh', value: 0.07, min: 0, max: 1, help: 'For pumps, vacuum system, vapour compressor and heat pump.' },
      { key: 'cArea', label: 'Cost of heat-transfer surface', unit: '$/m²', value: 120, min: 5, max: 2000, help: 'Purchased cost of evaporator, condenser and preheater surface.' },
      { key: 'cComp', label: 'Cost of the vapour compressor', unit: '$/kW', value: 900, min: 50, max: 5000, showIf: (v) => v.proc === 'mvc', help: 'Per kW of electrical compressor power.' },
      { key: 'capFactor', label: 'Installed plant cost ÷ equipment cost', unit: '×', value: 2.2, min: 1, max: 6, help: 'Covers shells, pumps, piping, intake, civil works and engineering.' },
      { key: 'intRate', label: 'Interest rate', unit: '%/y', value: 6, min: 0, max: 25, help: 'For the capital-recovery factor.' },
      { key: 'life', label: 'Plant life', unit: 'y', value: 25, min: 3, max: 50, help: 'Amortisation period.' },
      { key: 'avail', label: 'Availability', unit: '%', value: 92, min: 30, max: 100, help: 'Share of the year the plant produces.' },
      { key: 'omPct', label: 'Operation and maintenance', unit: '% of capital per year', value: 3, min: 0, max: 15, help: 'Labour, spares and cleaning.' },
      { key: 'cChem', label: 'Chemicals', unit: '$/m³', value: 0.03, min: 0, max: 1, help: 'Antiscalant, antifoam and post-treatment per m³ of distillate.' },
    ] },
    { group: 'Discretisation', tab: 'mesh', help: 'Numerical resolution of the tube-side integration and of the start-up integration.', fields: [
      { key: 'nSeg', label: 'Segments per condenser / preheater', unit: '', value: 4, min: 1, max: 60, step: 1, help: 'Heat exchangers are integrated in segments with local properties; one segment is the classical LMTD.' },
      { key: 'nt', label: 'Start-up time steps', unit: '', value: 240, min: 20, max: 6000, step: 1, help: 'Output steps of the RK4 start-up integration (sub-steps are added automatically when needed for stability).' },
    ] },
  ],

  presets: [
    { name: 'MED, 10 effects, parallel feed, 20 000 m³/d', values: {} },
    { name: 'MED-TVC, 10 effects, suction at effect 5', values: { proc: 'medtvc', N: 10, nEnt: 5, Pm: 10, Md: 36000, Xf: 42, Xb: 64, Tsw: 28, Tn: 40, areaMode: 'equalDT' } },
    { name: 'MSF brine recirculation, 24 stages, TBT 110 °C', values: { proc: 'msf', Md: 72000, Xf: 42, Xb: 66, Tsw: 30 } },
    { name: 'MSF once-through, 20 stages, TBT 90 °C', values: { proc: 'msf', msfType: 'ot', Nst: 20, TBT: 90, TnMsf: 36, Md: 10000, antiscalant: 'poly' } },
    { name: 'MVC single effect, 1 500 m³/d', values: { proc: 'mvc', Md: 1500, Xb: 70, heatLoss: 0.3, ventPct: 0.2, ttdCond: 2 } },
  ],

  pull: ({ feed, outputs }) => [
    feed?.ions ? { key: 'ions', value: feed.ions, from: 'Case feed water' } : null,
    feed?.ions ? { key: 'Xf', value: salinityFromTDS(tds(feed.ions), feed.T ?? 25), from: 'Case feed water salinity' } : null,
    feed?.T ? { key: 'Tsw', value: clamp(feed.T, 2, 40), from: 'Case feed water' } : null,
    outputs?.ro?.streams?.concentrate?.ions ? { key: 'ions', value: outputs.ro.streams.concentrate.ions, from: 'RO concentrate (hybrid RO–thermal)' } : null,
    outputs?.ro?.streams?.concentrate?.tds ? { key: 'Xf', value: salinityFromTDS(outputs.ro.streams.concentrate.tds, outputs.ro.streams.concentrate.T ?? 25), from: 'RO concentrate salinity (hybrid RO–thermal)' } : null,
  ],
  site: (site) => [
    { key: 'Tsw', value: site?.data?.sst, from: 'Sea-surface temperature at site' }, { key: 'Xf', value: site?.data?.salinity, from: 'Sea-surface salinity at site' },
    { key: 'ghi', value: site?.data?.ghiDaily, from: 'Daily solar irradiation at site' }, { key: 'Tamb', value: site?.data?.airTemp, from: 'Air temperature at site' },
    { key: 'cElec', value: site?.data?.electricityPrice, from: 'Electricity price at site' }, { key: 'intRate', value: site?.data?.lendingRate, from: 'Lending rate at site' },
  ],

  run(v) {
    if (v.hybrid === 'ro') { const size = roBlockSize(v.Md / 24, v); if (size.below) throw infeasible(size.msg, 'range'); } // lower limit of the RO model, stated before anything is solved
    const lean = v._lean === true, tryRun = lean ? () => null : tryRunFull; // lean (stress cases): the design is solved and reported in full, the sweeps and maps around it are skipped
    const R = simulateThermal(v), p = v, W = [], proc = R.proc, msf = proc === 'msf', mvc = proc === 'mvc', N = R.N, name = PROC[proc];
    const m = R.m, r = R.r, kgh = 3.6; // kg/s → t/h
    // ---- limits and warnings
    if (!R.converged) W.push({ level: 'bad', msg: 'The effect iteration did not fully converge — the temperature window is probably too narrow for this number of effects. Widen it or reduce the effects.' });
    const tLim = msf ? ANTISCALANT[p.antiscalant].tbt : p.tbtLimMED;
    if (R.Ttop > tLim + 1e-9) W.push({ level: 'bad', msg: `Top brine temperature ${fmt(R.Ttop, 4)} °C exceeds the ${tLim} °C scale limit${msf ? ` of ${ANTISCALANT[p.antiscalant].name.toLowerCase()}` : ' for falling-film evaporators'} — calcium-sulphate and alkaline scale will form.` });
    const scaleRows = (msf ? m.stages.map((s) => [s.i, s.T, s.X]) : r.T.map((T, i) => [i + 1, T, r.X[i]])).filter(([, T, X]) => X > caso4Limit(T));
    if (scaleRows.length) W.push({ level: msf ? 'info' : 'warn', msg: `Brine salinity exceeds the CaSO₄ solubility envelope in ${scaleRows.length} ${msf ? 'stage' : 'effect'}${scaleRows.length > 1 ? 's' : ''} (first: no. ${scaleRows[0][0]} at ${fmt(scaleRows[0][1], 3)} °C, ${fmt(scaleRows[0][2], 3)} g/kg vs ${fmt(caso4Limit(scaleRows[0][1]), 3)} g/kg)${msf ? ' — acceptable only with the selected scale control and sponge-ball cleaning.' : ' — lower the brine salinity or the top temperature.'}` });
    if (R.ej) {
      if (R.ej.Cr < 1.81) W.push({ level: 'warn', msg: `Ejector compression ratio ${fmt(R.ej.Cr, 3)} is below 1.81, outside the range of the entrainment correlation.` });
      if (R.ej.Ra > 4) W.push({ level: 'warn', msg: `Entrainment ratio ${fmt(R.ej.Ra, 3)} kg motive/kg entrained exceeds 4 — the thermo-compressor is inefficient here; take suction from a warmer effect.` });
      if (R.ej.Mev > 0.85 * r.Vt[r.c.nEnt]) W.push({ level: 'bad', msg: `The ejector wants ${fmt(R.ej.Mev, 3)} kg/s but effect ${r.c.nEnt + 1} only produces ${fmt(r.Vt[r.c.nEnt], 3)} kg/s of vapour — move the suction point or change the motive pressure.` });
    }
    if (!msf && Math.min(...r.dT) < 1) W.push({ level: 'warn', msg: `The smallest effect temperature difference is only ${fmt(Math.min(...r.dT), 2)} K — areas become very large; use fewer effects or a wider temperature window.` });
    if (!msf && r.V.some((x) => x <= 0)) W.push({ level: 'bad', msg: 'At least one effect produces no vapour — its feed is too cold for the heat it receives. Enable feed preheaters or reduce the number of effects.' });
    if (msf && Math.min(...m.stages.map((s) => s.ttd)) < 1.5) W.push({ level: 'warn', msg: `Smallest stage terminal difference is ${fmt(Math.min(...m.stages.map((s) => s.ttd)), 2)} K — condenser areas grow steeply below about 2 K.` });
    if (msf && p.msfType === 'br' && m.Mcw <= 0) W.push({ level: 'info', msg: 'No separate cooling water is needed: the make-up feed alone absorbs the heat-rejection duty.' });
    if (mvc && R.Qaux > 0) W.push({ level: 'warn', msg: `The compressor work cannot keep the unit in heat balance with a ${p.ttdCond} K preheater approach: ${fmt(R.Qaux / 1000, 3)} kW of auxiliary heat is required.` });
    if (mvc && N > 6) W.push({ level: 'info', msg: 'MVC units rarely have more than four effects.' });
    if (p.Xb < 1.25 * p.Xf && !(msf && p.msfType === 'ot')) W.push({ level: 'warn', msg: `The maximum brine salinity (${p.Xb} g/kg) is too close to the feed salinity (${fmt(p.Xf, 3)} g/kg); ${fmt(1.25 * p.Xf, 3)} g/kg was used so that the recovery stays above 20 %.` });
    if (R.Xb > 75) W.push({ level: 'warn', msg: `Blow-down salinity ${fmt(R.Xb, 3)} g/kg is above the usual 65–72 g/kg design range.` });
    if (!W.some((w) => w.level === 'bad')) W.unshift({ level: 'info', msg: `${name} design converged; mass, salt and energy balances close (see Verify tab).` });

    // ---- heat source helper
    const srcRows = [];
    let srcNote = '';
    if (p.source === 'solar') {
      const col = COLLECTORS[p.collector], Tm = (mvc ? 60 : proc === 'medtvc' ? R.ej.Tm : R.Tsteam) + 10, G = 800, eta = Math.max(0.05, col.eta0 - (col.a1 * (Tm - p.Tamb)) / G - (col.a2 * (Tm - p.Tamb) ** 2) / G);
      const Eday = (R.Qin / 1000) * 24 * (p.solarFrac / 100), area = Eday / (p.ghi * eta), store = (R.Qin * (p.solarFrac / 100) * 16 * 3600) / (cp(80, 0) * 20 * density(80, 0));
      srcRows.push(['Collector type', col.name], ['Mean collector temperature (°C)', Tm], ['Collector efficiency (%)', 100 * eta], ['Solar heat required (MWh/d)', Eday / 1000], ['Collector aperture area (m²)', area], ['Collector area per m³/d of distillate (m²)', area / p.Md], ['Hot-water storage for 16 h at ΔT 20 K (m³)', store]);
      if (Tm > col.tmax) W.push({ level: 'warn', msg: `${col.name}s are not suited to ${fmt(Tm, 3)} °C — choose a higher-temperature collector.` });
      if (mvc) W.push({ level: 'info', msg: 'MVC is driven by electricity; the solar-thermal helper only covers its small auxiliary heat demand.' });
      srcNote = `η = η₀ − a₁ΔT/G − a₂ΔT²/G at G = ${G} W/m²; ${col.name === 'Parabolic trough' ? 'troughs use direct irradiation, so treat the area as indicative.' : 'annual-mean irradiation.'}`;
    } else if (p.source === 'waste') {
      const Tneed = (proc === 'medtvc' ? R.ej.Tm : R.Tsteam) + 5, ok = p.Twh >= Tneed, cover = ok ? Math.min(1, (p.Qwh * 1e6) / Math.max(R.Qin, 1e-9)) : 0;
      srcRows.push(['Minimum stream temperature for this design (°C)', Tneed], ['Stream temperature (°C)', p.Twh], ['Heat demand of the plant (MW)', R.Qin / 1e6], ['Waste heat available (MW)', p.Qwh], ['Share of demand covered (%)', 100 * cover], ['Distillate producible from the waste heat alone (m³/d)', ok && R.sTh > 0 ? (p.Qwh * 1000 * 24) / R.sTh : 0]);
      if (!ok) W.push({ level: 'bad', msg: `The waste-heat stream (${p.Twh} °C) is colder than the ${fmt(Tneed, 3)} °C this design needs — lower the heating-steam temperature or use fewer effects.` });
      else if (cover < 1) W.push({ level: 'info', msg: `Waste heat covers ${fmt(100 * cover, 3)} % of the heat demand; the rest must come from steam.` });
    }
    const Tsup = proc === 'medtvc' ? R.ej.Tm : R.Tsteam, extraKpis = [], extraPlots = [], extraTables = [], extraBal = [], extraOut = {};
    let srcTitle = p.source === 'solar' ? 'Solar-thermal integration' : 'Waste-heat integration', costEx = {};
    if (p.source === 'cogen') {
      const cg = cogeneration(R.Qin, Tsup, R.Wel, p), mig = (R.Qd * 24) / 4546.09;
      srcTitle = 'Thermal–electrical cogeneration (turbine extraction, power-loss method)';
      srcRows.push(['Throttle steam temperature (°C)', cg.T1], ['Extraction (heating-steam) temperature (°C)', cg.Te], ['Extraction pressure (bar)', psat(cg.Te) / 1e5], ['Steam quality at extraction (–)', cg.xE], ['Turbine steam flow (kg/s)', cg.m0], ['Extraction steam flow (kg/s)', cg.mExt], ['Share of turbine steam extracted (%)', (100 * cg.mExt) / cg.m0],
        ['Lost work per kg of extraction steam (kJ/kg)', cg.wLost / 1000], ['Lost work ÷ heat delivered (–)', cg.lostPerHeat], ['Carnot factor of the delivered heat, 1 − T_c/T_ext (–)', cg.carnot], ['Exergy returned with the hot condensate ÷ heat (–)', cg.condensate], ['Power lost by extraction (MW)', cg.Wlost / 1e6], ['Electricity used by the desalination plant (MW)', R.Wel / 1e6], ['Net power for export (MW)', cg.Pexport / 1e6],
        ['Power-to-water ratio (MW per MIGD)', cg.Pexport / 1e6 / mig], ['Power-to-water ratio (kW per m³/d)', cg.Pexport / 1000 / (R.Qd * 24)], ['Fuel heat (MW)', cg.fuel / 1e6], ['Fuel charged to water, power-loss method (MW)', cg.fuelWater / 1e6], ['Share of fuel charged to water (%)', 100 * cg.fuelShare], ['Energy-utilisation factor (–)', cg.euf], ['Power-only cycle efficiency (%)', 100 * cg.etaPower],
        ['Equivalent electricity of the water, lost work + auxiliaries (kWh/m³)', (cg.Wlost + R.Wel) / 1000 / R.Qd]);
      srcNote = 'Saturated-steam Rankine expansion with the turbine efficiency entered under “Heat source”: s_g = ∫c_p dT/T + λ/T fixes the wet-steam states; the water is charged the turbine work its extraction steam would still have produced down to the condenser. For an ideal turbine that work equals the Carnot value of the heat plus the exergy of the hot condensate, which returns to the boiler and is credited in the fuel heat.';
      if (!cg.feasible) W.push({ level: 'bad', msg: cg.clipped ? `The plant needs heat at ${fmt(Tsup, 3)} °C, which is not below the ${fmt(cg.T1, 3)} °C throttle steam — raise the throttle pressure.` : `The extraction (${fmt(cg.mExt, 3)} kg/s) exceeds the turbine steam flow (${fmt(cg.m0, 3)} kg/s) — the power plant is too small for this water capacity.` });
      if (cg.feasible && cg.Pexport < 0) W.push({ level: 'warn', msg: `The desalination plant uses ${fmt(R.Wel / 1e6, 3)} MW of electricity but the power plant has only ${fmt(cg.Pnet / 1e6, 3)} MW left after the steam extraction: the station is a net importer of ${fmt(-cg.Pexport / 1e6, 3)} MW and the power-to-water ratio is negative. Raise the gross power-plant output to at least ${fmt(Math.ceil((cg.Pgross - cg.Pexport) / 1e5) / 10, 3)} MW.` });
      extraKpis.push({ label: 'Power lost by steam extraction', value: cg.Wlost / 1e6, unit: 'MW', status: cg.feasible ? 'ok' : 'bad' }, { label: 'Equivalent electricity (lost work)', value: (cg.Wlost + R.Wel) / 1000 / R.Qd, unit: 'kWh/m³', help: 'Turbine work lost by the extraction plus auxiliaries, per m³ of distillate' }, { label: 'Power-to-water ratio', value: cg.Pexport / 1e6 / mig, unit: 'MW/MIGD' }, { label: 'Energy-utilisation factor', value: cg.euf, unit: '–', help: '(Net power + process heat) ÷ fuel heat' });
      const Te = linspace(Math.min(cg.Tc + 8, cg.T1 - 10), Math.min(130, cg.T1 - 5), 9), cc = Te.map((T) => cogeneration(1e6, T, 0, p));
      extraPlots.push({ type: 'line', title: 'Cogeneration: turbine work lost per unit of extracted heat', xlabel: 'Extraction (heating-steam) temperature (°C)', ylabel: 'kWh of power per kWh of heat', series: [{ name: 'Lost work ÷ heat (turbine model)', x: Te, y: cc.map((q) => q.lostPerHeat), mode: 'both' }, { name: 'Reversible limit: Carnot factor + condensate exergy', x: Te, y: cc.map((q) => q.carnot + q.condensate), dash: true }], vlines: [{ x: cg.Te, label: 'this plant' }], note: 'Low-temperature processes (MED) take steam that has already done most of its work; MSF and TVC motive steam cost more power.' });
      extraOut.powerLoss = cg.Wlost / 1000; extraOut.powerToWater = cg.Pexport / 1e6 / mig; extraOut.secLostWork = (cg.Wlost + R.Wel) / 1000 / R.Qd;
    } else if (p.source === 'mhp' || p.source === 'ahp') {
      const hp = heatPump(R, p);
      srcTitle = hp.abs ? 'Absorption heat-pump desalination' : 'Mechanical heat-pump desalination';
      srcRows.push(['Heat-pump source temperature (°C)', hp.Tl], ['Heat-pump delivery temperature (°C)', hp.Th], ...(hp.abs ? [['Driving-heat (generator) temperature (°C)', hp.Tg]] : []), ['Reversible COP (–)', hp.copRev], ['COP of the heat pump (–)', hp.cop], ['Heat demand of the plant (MW)', hp.Qneed / 1e6], ['Rejected heat available (MW)', hp.Qavail / 1e6], ['Heat taken from the reject (MW)', hp.Qe / 1e6],
        [hp.abs ? 'Driving heat to the generator (MW)' : 'Compressor power (MW)', hp.drive / 1e6], ['Heat delivered by the heat pump (MW)', hp.Qhp / 1e6], ['Heat still supplied directly (MW)', hp.Qdirect / 1e6], ['Share of the heat demand covered (%)', 100 * hp.share], ['Cooling duty avoided (%)', 100 * hp.cwSaving],
        ['External heat (kWh/m³)', hp.sTh], ['Electricity incl. heat pump (kWh/m³)', hp.sEl], ['Equivalent electricity (kWh/m³)', hp.sEq], ['Performance ratio on external heat (kg/2326 kJ)', hp.PR]);
      srcNote = hp.abs ? 'COP = 1 + η·(COP_rev − 1), COP_rev = (1 − T_l/T_g)/(1 − T_l/T_h); the absorber and condenser heat (driving heat + lifted heat) goes to the first effect or brine heater.' : 'COP = η·T_h/(T_h − T_l); the compressor work and the lifted condenser heat go to the first effect or brine heater.';
      if (hp.Qavail <= 0) W.push({ level: 'info', msg: mvc ? 'MVC is already a mechanical heat pump with no condenser reject — the heat-pump option has nothing to upgrade.' : 'This configuration rejects no heat to cooling water, so the heat pump has no low-temperature source; all heat is supplied directly.' });
      else if (hp.share < 0.999) W.push({ level: 'info', msg: `The rejected heat limits the heat pump to ${fmt(100 * hp.share, 3)} % of the heat demand; the rest is supplied directly.` });
      if (proc === 'medtvc') W.push({ level: 'info', msg: 'With a thermo-compressor the heat pump has to deliver motive steam at ' + fmt(hp.Th, 3) + ' °C — the large lift makes the COP low; plain MED suits a heat pump better.' });
      extraKpis.push({ label: 'Heat-pump COP', value: hp.cop, unit: '–', help: `Reversible limit ${fmt(hp.copRev, 3)}` }, { label: hp.abs ? 'Heat-pump driving heat' : 'Heat-pump compressor power', value: hp.drive / 1e6, unit: 'MW' }, { label: 'Equivalent electricity with heat pump', value: hp.sEq, unit: 'kWh/m³' }, { label: 'Cooling duty avoided', value: 100 * hp.cwSaving, unit: '%' });
      extraBal.push({ name: 'Heat pump: source heat + drive = heat delivered (MW)', in: (hp.Qe + hp.drive) / 1e6, out: hp.Qhp / 1e6 });
      const ee = [20, 35, 50, 65, 80, 100];
      extraPlots.push({ type: 'line', title: 'Heat-pump COP and equivalent electricity versus second-law efficiency', xlabel: 'Second-law efficiency (%)', ylabel: 'COP · kWh/m³', series: [{ name: 'COP', x: ee, y: ee.map((e) => heatPump(R, { ...p, hpEta: e }).cop), mode: 'both' }, { name: 'Equivalent electricity (kWh/m³)', x: ee, y: ee.map((e) => heatPump(R, { ...p, hpEta: e }).sEq), mode: 'both' }], hlines: [{ y: R.sEq, label: 'without heat pump' }], vlines: [{ x: p.hpEta, label: 'selected' }] });
      costEx = { heat: hp.heatExt, power: R.Wel + hp.Wext };
      extraOut.heatPumpCOP = hp.cop; extraOut.secEquivalentHeatPump = hp.sEq;
    }

    // ---- sweeps (fast equal-ΔT designs; MED/MVC part load is a fixed-area rating)
    const base = { ...v }, pick = (q) => (q ? [nz(q.GOR), q.sArea / 1000, q.sEl, q.sTh] : [null, null, null, null]);
    const nKey = msf ? 'Nst' : mvc ? 'Nmvc' : 'N', nNow = N, nList = (msf ? [12, 16, 20, 24, 28, 32, 36] : mvc ? [1, 2, 3, 4] : [4, 6, 8, 11, 14]).filter((n) => !msf || n > p.nRej + 1);
    if (!nList.includes(nNow)) { nList.push(nNow); nList.sort((a, b) => a - b); }
    const sweepN = nList.map((n) => pick(tryRun(base, { [nKey]: n, fast: true, nEnt: p.nEnt > 0 && !msf ? Math.max(1, Math.round((p.nEnt * n) / N)) : 0 })));
    const tLastKey = msf ? 'TnMsf' : 'Tn', Tsws = linspace(Math.max(5, p.Tsw - 10), Math.min(38, p.Tsw + 8), 5);
    const sweepT = Tsws.map((T) => { const q = tryRun(base, mvc ? { Tsw: T, fast: true } : { Tsw: T, [tLastKey]: p[tLastKey] + (T - p.Tsw), fast: true }); return q ? [nz(q.GOR), q.sArea / 1000, q.sEl, (q.Mcw * 3.6) / density(T, p.Xf) / q.Qd, q.sTh] : [null, null, null, null, null]; });
    const loads = [50, 70, 85, 100, 110];
    let part;
    if (msf) part = loads.map((L) => { const q = tryRun(base, { TBT: p.TnMsf + ((p.TBT - p.TnMsf) * L) / 100, fast: true }); return q ? [L, q.PR, (100 * q.areaEvap) / R.areaEvap, p.TnMsf + ((p.TBT - p.TnMsf) * L) / 100, q.sTh] : [L, null, null, null, null]; });
    else part = loads.map((L) => { const q = tryRun(base, { Md: (p.Md * L) / 100, areaMode: 'rating', Ades: r.A, dT0: r.dT.map((d) => (d * L) / 100), fast: true }); return q ? [L, mvc ? q.sEl : q.GOR, q.Tsteam, q.Ttop, mvc ? q.comp.ratio : q.sTh] : [L, null, null, null, null]; });
    // 2-D sensitivity map (grid of fast re-designs) and its use for the cost-optimal design
    const grid = designGrid(base, N, lean), field = { type: 'field', title: grid.title, xlabel: grid.xlabel, ylabel: grid.ylabel, zlabel: grid.zlabel, zunit: grid.zunit, x: grid.xs, y: grid.ys, z: grid.cells.map((row) => row.map((q) => (q ? nz(grid.metric(q)) : null))), cmap: grid.cmap, contours: 8, markers: [{ x: grid.x0, y: grid.y0, label: 'design' }] };
    // repair isolated gaps of the map so that colour scaling stays finite
    const zf = field.z.flat().filter((x) => x !== null), zMean = zf.length ? sum(zf) / zf.length : 0;
    field.z = field.z.map((row) => row.map((x) => (x === null ? zMean : x)));
    field.note = 'Each cell is a full re-design with equal temperature differences; cells that cannot be designed are filled with the map average.';
    // ---- thermodynamic–economic and exergy–economic costing, process optimisation on the design grid
    const wc = waterCost(R, p, costEx), xe = exergoEconomics(R, p, waterCost(R, p)), opt = optimiseGrid(grid, p, tLim, waterCost(R, p).total), od = optimiseDesign(grid, p, tLim, opt);
    const cf_ = opt.cost.flat().filter((x) => x !== null), cMean = cf_.length ? sum(cf_) / cf_.length : wc.total;
    const costField = { type: 'field', title: 'Unit water cost on the design map (process optimisation)', xlabel: grid.xlabel, ylabel: grid.ylabel, zlabel: 'Water cost', zunit: '$/m³', x: grid.xs, y: grid.ys, z: opt.cost.map((row) => row.map((x) => (x === null ? cMean : x))), cmap: 'thermal', contours: 8, markers: [{ x: grid.x0, y: grid.y0, label: 'design' }, ...(opt.best.base ? [] : [{ x: opt.best.x, y: opt.best.y, label: 'lowest cost on the grid' }]), ...(od.improved ? [{ x: clamp(od.best.x, grid.xs[0], grid.xs[grid.xs.length - 1]), y: od.best.y, label: 'optimum' }] : [])],
      note: `Backdrop: grid of ${opt.nCells} re-designs (${opt.nFeas} within the ${tLim} °C scale limit) with the costing inputs of Model setup; lowest grid cost ${fmt(opt.best.cost, 3)} $/m³ at ${grid.xlabel.toLowerCase()} = ${fmt(opt.best.x, 3)}, ${grid.ylabel.toLowerCase()} = ${fmt(opt.best.y, 3)}${opt.best.base ? ' (the present design)' : ''}. Optimiser (${od.method}, ${od.evals} further re-designs): ${fmt(od.best.cost, 4)} $/m³ at ${fmt(od.best.x, 3)} / ${fmt(od.best.y, 4)}${od.improved ? '' : ' — no better design than the grid optimum inside the bounds'}.` };
    if (od.improved && od.best.cost < 0.97 * wc.total) W.push({ level: 'info', msg: `Process optimisation: the optimiser finds ${fmt(od.best.cost, 3)} $/m³ at ${grid.xlabel.toLowerCase()} ${fmt(od.best.x, 3)} and ${grid.ylabel.toLowerCase()} ${fmt(od.best.y, 4)} (present design ${fmt(waterCost(R, p).total, 3)} $/m³).` });
    if (!opt.best.base && opt.saving > 0.03) W.push({ level: 'info', msg: `Process optimisation: the design map holds a cheaper design (${fmt(opt.best.cost, 3)} instead of ${fmt(waterCost(R, p).total, 3)} $/m³) at ${grid.xlabel.toLowerCase()} ${fmt(opt.best.x, 3)} and ${grid.ylabel.toLowerCase()} ${fmt(opt.best.y, 3)}.` });

    let su = null, tSim = p.tEnd;
    if (p.transient) for (let k = 0; k < (lean ? 1 : 4); k++) { su = startUp(R, { ...p, tEnd: tSim }); if (su.t95 !== null) break; if (k < 3) tSim *= 2; }
    if (su && su.t95 === null) W.push({ level: 'info', msg: `Start-up does not reach 95 % of the design temperature rise within ${fmt(tSim, 3)} h.` });
    const sd = su ? startUp(R, { ...p, tEnd: tSim }, true) : null;
    if (su && su.t95 !== null && tSim > p.tEnd) W.push({ level: 'info', msg: `The start-up simulation was extended to ${fmt(tSim, 3)} h so that the plant reaches its design temperatures.` });

    // ---- outputs
    const ionsF = scaleIons(cloneIons(p.ions), tdsFromSalinity(p.Xf, 25) / Math.max(tds(p.ions), 1e-9)), cf = R.Xb / p.Xf, Xmean = msf ? sum(m.stages.map((s) => s.X)) / N : sum(r.X) / N;
    const tdsD = (p.carry * 1e-6 * Xmean * 1000 * RHO_REF) / 1000, Qb = (R.Mb * 3600) / density(R.Tbrine, R.Xb);
    const round = (o) => Object.fromEntries(ION_IDS.map((k) => [k, +o[k].toPrecision(6)]));
    const out = { distillate: R.Qd, GOR: nz(R.GOR), PR: nz(R.PR), secThermal: R.sTh, secElec: R.sEl, steam: R.Ms, area: R.area, recovery: R.rec, coolingWater: (R.Mcw * 3600) / density(p.Tsw, p.Xf), heat: R.Qin / 1000, power: R.Wel / 1000, process: proc, topBrineTemperature: R.Ttop, brineSalinity: R.Xb, secEquivalent: R.sEq, startUpMin: su?.t95 ?? null,
      streams: { distillate: { Q: R.Qd, T: R.Tprod, P: 1, pH: 6.5, tds: tdsD, ions: round(scaleIons(ionsF, tdsD / Math.max(tds(ionsF), 1e-9))) }, brine: { Q: Qb, T: R.Tbrine, P: 1, pH: Math.min(9, 8.1 + 0.3 * Math.log10(cf)), tds: tdsFromSalinity(R.Xb, R.Tbrine), ions: round(scaleIons(ionsF, tdsFromSalinity(R.Xb, R.Tbrine) / Math.max(tds(ionsF), 1e-9))) } } };

    Object.assign(out, extraOut, { waterCost: wc.total, exergyCostProduct: xe.cP, optimumWaterCost: opt.best.cost, optimisedWaterCost: od.best.cost, optimalDesign: { x: od.best.x, y: od.best.y }, heatFlux: R.areaEvap > 0 ? (msf ? sum(m.stages.map((s) => s.Q)) : sum(r.q)) / R.areaEvap / 1000 : 0 });
    // ---- thermal + RO hybrid
    let hy = null;
    if (p.hybrid === 'ro') {
      try { hy = hybridRO(R, p, tdsD, ionsF); } catch (e) { W.push({ level: 'bad', msg: e.infeasible ? e.message : `The RO block of the hybrid could not be solved at ${fmt(p.roRec, 3)} % recovery (${e.message}). Lower the RO recovery.` }); }
    }
    if (hy) {
      const ro = hy.ro, hn = `${name}–RO`;
      if (hy.overPressure) W.push({ level: 'bad', msg: `The RO block needs ${fmt(ro.p1.Pf, 3)} bar, above the ${ro.cfg.M.pmax} bar element rating — lower the RO recovery.` });
      if (Math.abs(ro.overallRec - hy.rec) > 0.01) W.push({ level: 'warn', msg: `The RO block reaches ${fmt(100 * ro.overallRec, 3)} % recovery instead of the ${fmt(100 * hy.rec, 3)} % asked for — at this salinity the pressure limit of the elements is reached. Lower the RO recovery.` });
      if (hy.Tro > 40) W.push({ level: 'warn', msg: `The RO feed is ${fmt(hy.Tro, 3)} °C; most elements are limited to 40–45 °C — blend in cold seawater.` });
      if (hy.tdsBlend > p.tdsLimit) W.push({ level: 'warn', msg: `The blended product has ${fmt(hy.tdsBlend, 3)} mg/L, above the ${p.tdsLimit} mg/L limit — lower the RO share or add a second pass.` });
      else W.push({ level: 'info', msg: `${hn} hybrid: ${fmt(100 * hy.share, 3)} % of the product comes from single-pass RO (${fmt(hy.tdsP, 3)} mg/L) and blends with distillate to ${fmt(hy.tdsBlend, 3)} mg/L.` });
      extraKpis.push({ label: 'Hybrid product (thermal + RO)', value: hy.Qtot * 24, unit: 'm³/d' }, { label: 'Blended product TDS', value: hy.tdsBlend, unit: 'mg/L', status: hy.tdsBlend > p.tdsLimit ? 'warn' : 'ok' }, { label: 'Hybrid recovery (product ÷ feed treated)', value: 100 * hy.recFeed, unit: '%' },
        { label: 'Hybrid equivalent-electric energy', value: hy.secEq, unit: 'kWh/m³', help: 'Thermal block (electricity + heat as lost turbine work) and RO electricity over the blended product' }, { label: 'RO feed pressure', value: ro.p1.Pf, unit: 'bar', status: hy.overPressure ? 'bad' : 'ok' }, { label: 'Intake flow saved by sharing', value: 100 * hy.intakeSaving, unit: '%' });
      extraTables.push({ title: `${hn} hybrid: contribution of each process`, columns: ['Process', 'Feed treated (m³/h)', 'Product (m³/h)', 'Recovery (%)', 'Share of product (%)', 'Product TDS (mg/L)', 'Electricity (kW)', 'Heat (kW)', 'Equivalent electricity (kWh/m³ of its product)', 'Brine (m³/h)'],
        rows: [[name, hy.QfTh, R.Qd, 100 * R.rec, 100 * (1 - hy.share), tdsD, R.Wel / 1000, R.Qin / 1000, R.sEq, Qb], ['Reverse osmosis', hy.Qf, hy.Qp, 100 * ro.overallRec, 100 * hy.share, hy.tdsP, ro.power, 0, ro.sec, ro.conc.Q], ['Hybrid total', hy.QfTh + hy.Qf, hy.Qtot, 100 * hy.recFeed, 100, hy.tdsBlend, R.Wel / 1000 + ro.power, R.Qin / 1000, hy.secEq, hy.Qbrine]],
        note: `RO: ${ro.nEl} elements, ${fmt(ro.area, 4)} m², feed at ${fmt(hy.Tro, 3)} °C (${fmt(hy.warm, 4)} m³/h taken from the ${fmt(hy.Qrej, 4)} m³/h warm reject) and ${fmt(ro.p1.Pf, 3)} bar${hy.cold ? `; fed with cold seawater it would need ${fmt(hy.cold.p1.Pf, 3)} bar and ${fmt(hy.cold.sec, 3)} instead of ${fmt(ro.sec, 3)} kWh/m³` : ''}. Shared intake ${fmt(hy.intake, 4)} m³/h (product ÷ intake ${fmt(100 * hy.recIntake, 3)} %); combined brine ${fmt(hy.tdsBrine / 1000, 3)} g/L.` });
      const sh = linspace(0, 90, 10), at = (x) => { const q = (R.Qd * (x / 100)) / (1 - x / 100); return [(R.Qd * tdsD + q * hy.tdsP) / (R.Qd + q), (R.sEq * R.Qd + ro.sec * q) / (R.Qd + q)]; };
      extraPlots.push({ type: 'line', title: `${hn} hybrid: blend quality and energy versus RO share`, xlabel: 'RO share of the blended product (%)', ylabel: 'mg/L · kWh/m³', series: [{ name: 'Blended product TDS (mg/L)', x: sh, y: sh.map((x) => at(x)[0]), mode: 'both' }, { name: 'Equivalent electricity × 10 (kWh/m³)', x: sh, y: sh.map((x) => 10 * at(x)[1]), mode: 'both' }], hlines: [{ y: p.tdsLimit, label: 'TDS limit' }], vlines: [{ x: 100 * hy.share, label: 'selected' }], note: 'Thermal capacity fixed; the RO permeate quality and specific energy of the solved array are kept along the curve.' });
      extraBal.push({ name: 'Hybrid RO water (m³/h)', in: hy.Qf, out: hy.Qp + ro.conc.Q }, { name: 'Blend salt (kg/h)', in: (R.Qd * tdsD + hy.Qp * hy.tdsP) / 1000, out: (hy.Qtot * hy.tdsBlend) / 1000 });
      Object.assign(out, { hybridProduct: hy.Qtot, hybridTDS: hy.tdsBlend, hybridRecovery: hy.recFeed, hybridSecEquivalent: hy.secEq, hybridSecElec: hy.secEl, hybridSecThermal: hy.secTh, roShare: hy.share });
      out.streams.blend = { Q: hy.Qtot, T: R.Tprod, P: 1, pH: 6.8, tds: hy.tdsBlend, ions: round(hy.ionsBlend) };
    }
    // ---- Raoult's-law comparison of the boiling-point elevation
    const rl = (msf ? m.stages.map((s) => [s.i, s.T, s.X]) : r.T.map((T, i) => [i + 1, T, r.X[i]])).map(([n, T, X]) => ({ n, T, X, corr: bpe(T, X), act: bpeRaoult(T, X, 'activity'), ideal: bpeRaoult(T, X, 'ideal'), aw: waterActivity(T, X, 'activity'), xw: waterActivity(T, X, 'ideal'), pR: raoultPressure(T, X, 'activity'), pC: psat(T - bpe(T, X)) }));
    const bpeName = { corr: 'seawater correlation', activity: 'Raoult’s law with water activity', ideal: 'ideal Raoult’s law' }[p.bpeModel] || 'seawater correlation', rlLast = rl[rl.length - 1];
    if (p.bpeModel === 'ideal') W.push({ level: 'info', msg: `Ideal Raoult’s law (mole fraction, no osmotic coefficient) gives a boiling-point elevation ${fmt(100 * (rlLast.ideal / rlLast.corr - 1), 2)} % away from the seawater correlation in the last ${msf ? 'stage' : 'effect'}.` });
    const xs = Array.from({ length: N }, (_, i) => i + 1), unit = msf ? 'Stage' : 'Effect';
    const plots = [];
    if (msf) plots.push({ type: 'line', title: 'Temperature profile along the stages', xlabel: 'Stage', ylabel: '°C', series: [{ name: 'Flashing brine', x: xs, y: m.stages.map((s) => s.T), mode: 'both' }, { name: 'Condensing vapour', x: xs, y: m.stages.map((s) => s.Tv), mode: 'both' }, { name: 'Tube-side brine leaving stage', x: xs, y: m.stages.map((s) => s.tout), mode: 'both' }], hlines: [{ y: tLim, label: 'scale limit' }], vlines: m.nRej ? [{ x: m.nRec + 0.5, label: 'recovery | rejection' }] : [] },
      { type: 'line', title: 'Distillate formed and brine salinity per stage', xlabel: 'Stage', ylabel: 'kg/s · g/kg', series: [{ name: 'Distillate formed × 10 (kg/s)', x: xs, y: m.stages.map((s) => s.D * 10), mode: 'both' }, { name: 'Brine salinity (g/kg)', x: xs, y: m.stages.map((s) => s.X), mode: 'both' }, { name: 'Accumulated distillate (kg/s)', x: xs, y: m.stages.map((s) => s.Dacc), mode: 'both' }] },
      { type: 'line', title: 'Temperature losses and terminal difference per stage', xlabel: 'Stage', ylabel: 'K', series: [{ name: 'Boiling-point elevation', x: xs, y: m.stages.map((s) => s.be), mode: 'both' }, { name: 'Non-equilibrium allowance', x: xs, y: m.stages.map((s) => s.nea), mode: 'both' }, { name: 'Demister + line', x: xs, y: m.stages.map((s) => s.dl), mode: 'both' }, { name: 'Terminal temperature difference', x: xs, y: m.stages.map((s) => s.ttd), mode: 'both' }] },
      { type: 'bar', title: 'Condenser area per stage', ylabel: 'm²', categories: xs.map(String), series: [{ name: 'Area', values: m.stages.map((s) => s.A) }] });
    else plots.push({ type: 'line', title: 'Temperature profile along the effects', xlabel: 'Effect', ylabel: '°C', series: [{ name: 'Heating vapour (tube side)', x: xs, y: r.Th, mode: 'both' }, { name: 'Boiling brine', x: xs, y: r.T, mode: 'both' }, { name: 'Vapour after demister', x: xs, y: r.Tc, mode: 'both' }, ...(mvc ? [] : [{ name: 'Feed entering effect', x: xs, y: r.tf, mode: 'both', dash: true }])], hlines: [{ y: tLim, label: 'scale limit' }] },
      { type: 'line', title: 'Vapour formed and brine salinity per effect', xlabel: 'Effect', ylabel: 'kg/s · g/kg', series: [{ name: 'Vapour by boiling (kg/s)', x: xs, y: r.V, mode: 'both' }, { name: 'Vapour by distillate flashing (kg/s)', x: xs, y: r.fl, mode: 'both' }, { name: 'Brine salinity (g/kg)', x: xs, y: r.X, mode: 'both' }, { name: 'CaSO₄ envelope (g/kg)', x: xs, y: r.T.map((T) => Math.min(caso4Limit(T), 200)), dash: true }] },
      { type: 'line', title: 'Temperature differences and losses per effect', xlabel: 'Effect', ylabel: 'K', series: [{ name: 'Driving ΔT', x: xs, y: r.dT, mode: 'both' }, { name: 'Boiling-point elevation', x: xs, y: r.be, mode: 'both' }, { name: 'Demister + line loss', x: xs, y: r.dl, mode: 'both' }] },
      { type: 'bar', title: 'Evaporator area per effect', ylabel: 'm²', categories: xs.map(String), series: [{ name: 'Area', values: r.A }] });
    plots.push({ type: 'bar', title: 'Exergy destruction by component', ylabel: 'kW', categories: R.ex.map((e) => e[0]), series: [{ name: 'Exergy destroyed or rejected', values: R.ex.map((e) => e[1] / 1000) }], note: `Dead state ${fmt(p.Tsw, 3)} °C. Second-law efficiency = minimum work of separation ÷ exergy supplied.` });
    plots.push({ type: 'line', title: `Effect of the number of ${msf ? 'stages' : 'effects'}`, xlabel: msf ? 'Stages' : 'Effects', ylabel: mvc ? 'kWh/m³ · 1000 m² per kg/s' : 'GOR (kg/kg) · 1000 m² per kg/s', series: [{ name: mvc ? 'Specific electricity (kWh/m³)' : 'Gain-output ratio', x: nList, y: sweepN.map((q) => (mvc ? q[2] : q[0])), mode: 'both' }, { name: 'Specific area (1000 m² per kg/s)', x: nList, y: sweepN.map((q) => q[1]), mode: 'both' }], vlines: [{ x: N, label: 'design' }] });
    plots.push({ type: 'line', title: 'Effect of seawater temperature', xlabel: 'Seawater temperature (°C)', ylabel: 'see legend', series: [mvc ? { name: 'Specific electricity (kWh/m³)', x: Tsws, y: sweepT.map((q) => q[2]), mode: 'both' } : { name: 'Gain-output ratio', x: Tsws, y: sweepT.map((q) => q[0]), mode: 'both' }, { name: 'Specific area (1000 m² per kg/s)', x: Tsws, y: sweepT.map((q) => q[1]), mode: 'both' }, ...(mvc ? [] : [{ name: 'Cooling water (m³ per m³ distillate)', x: Tsws, y: sweepT.map((q) => q[3]), mode: 'both' }])], note: mvc ? 'Design recalculated at each temperature.' : 'The last effect/stage temperature follows the seawater so that the condenser approach stays constant.' });
    plots.push(msf ? { type: 'line', title: 'Turn-down by top brine temperature (constant stage temperatures spacing)', xlabel: 'Flashing range (% of design)', ylabel: 'see legend', series: [{ name: 'Performance ratio', x: loads, y: part.map((q) => q[1]), mode: 'both' }, { name: 'Condenser area needed (% of installed ÷ 10)', x: loads, y: part.map((q) => (q[2] === null ? null : q[2] / 10)), mode: 'both' }, { name: 'Top brine temperature ÷ 10 (°C)', x: loads, y: part.map((q) => (q[3] === null ? null : q[3] / 10)), mode: 'both' }], note: 'Re-design at reduced flashing range for the same distillate; a ratio above 100 % means the installed area would limit production.' }
      : { type: 'line', title: 'Part-load rating at fixed heat-transfer area', xlabel: 'Load (% of design distillate)', ylabel: 'see legend', series: [{ name: mvc ? 'Specific electricity (kWh/m³)' : 'Gain-output ratio', x: loads, y: part.map((q) => q[1]), mode: 'both' }, { name: 'Heating temperature ÷ 10 (°C)', x: loads, y: part.map((q) => (q[2] === null ? null : q[2] / 10)), mode: 'both' }, { name: 'Top brine temperature ÷ 10 (°C)', x: loads, y: part.map((q) => (q[3] === null ? null : q[3] / 10)), mode: 'both' }], note: 'The installed areas are kept; the temperature differences and the heating-steam temperature adjust to the load.' });
    plots.push(field, costField);
    plots.push({ type: 'line', title: 'Boiling-point elevation: seawater correlation versus Raoult’s law', xlabel: unit, ylabel: 'K', series: [{ name: 'Seawater correlation', x: xs, y: rl.map((q) => q.corr), mode: 'both' }, { name: 'Raoult’s law with water activity a_w = exp(−φ·m/55.51)', x: xs, y: rl.map((q) => q.act), mode: 'both' }, { name: 'Raoult’s law, ideal (water mole fraction)', x: xs, y: rl.map((q) => q.ideal), mode: 'both', dash: true }], note: `BPE = T − t_sat(a_w·p_sat(T)). The balances use the ${bpeName}${p.fBPE !== 1 ? ` × ${p.fBPE}` : ''}.` });
    plots.push({ type: 'bar', title: 'Unit water cost breakdown (thermodynamic–economic model)', ylabel: '$/m³', categories: Object.keys(wc.parts), series: [{ name: 'Cost', values: Object.values(wc.parts) }], note: `Capital ${fmt(wc.capex / 1e6, 3)} M$ (${fmt(wc.perCapacity, 4)} $ per m³/d), capital-recovery factor ${fmt(wc.crf, 3)} per year, ${fmt(wc.hrs, 4)} h/y.` });
    plots.push({ type: 'bar', title: 'Exergy–economic cost rates by component', ylabel: '$/h', categories: xe.rows.map((q) => q.name), series: [{ name: 'Cost of exergy destroyed Ċ_D', values: xe.rows.map((q) => q.CD) }, { name: 'Capital and O&M rate Ż', values: xe.rows.map((q) => q.Z) }], stacked: true, note: `Fuel exergy costs ${fmt(xe.cF, 3)} $/kWh; the product (minimum work of separation) leaves at ${fmt(xe.cP, 3)} $/kWh.` });
    plots.push(...extraPlots);
    if (su) {
      const idx = [...new Set([0, Math.floor((N - 1) / 2), N - 1])];
      plots.push({ type: 'line', title: 'Start-up from cold and cool-down after a steam trip', xlabel: 'Time (min)', ylabel: '°C · %', series: [...idx.map((i) => ({ name: `${unit} ${i + 1} brine temperature (°C)`, x: su.t, y: su.T.map((y) => y[i]) })), { name: 'Start-up: approach to design (%)', x: su.t, y: su.prod, dash: true }, { name: 'Shutdown after a steam trip: remaining temperature rise (%)', x: sd.t, y: sd.prod, dash: true }, ...(su.vap.mTot > 0 ? [{ name: 'Start-up: vapour inventory (% of design)', x: su.t, y: su.vap.inv, dash: true }] : [])], note: `Lumped thermal inertia, linear steam ramp over ${p.ramp} min; after a steam trip feed and cooling water keep flowing; ${su.sub} RK4 sub-step${su.sub > 1 ? 's' : ''} per output step.` });
    }
    // any failed sweep point is dropped from its series
    for (const pl of plots) if (pl.type === 'line') {
      for (const s of pl.series) { const keep = s.y.map((y, i) => y !== null && Number.isFinite(y) && Number.isFinite(s.x[i])); s.x = s.x.filter((_, i) => keep[i]); s.y = s.y.filter((_, i) => keep[i]); }
      pl.series = pl.series.filter((s) => s.x.length > 0);
    }
    const shown = plots.filter((pl) => pl.type !== 'line' || pl.series.length > 0);

    const tables = [];
    if (msf) tables.push({ title: 'Stage-by-stage results', columns: ['Stage', 'Section', 'Brine T (°C)', 'Vapour T (°C)', 'Pressure (kPa)', 'Brine out (kg/s)', 'Salinity (g/kg)', 'Distillate formed (kg/s)', 'BPE (K)', 'NEA (K)', 'Demister+line (K)', 'Tube in (°C)', 'Tube out (°C)', 'TTD (K)', 'U (W/m²·K)', 'Area (m²)', 'Duty (MW)', 'NTU', 'Effectiveness', 'Heat flux (kW/m²)'],
      rows: m.stages.map((s) => [s.i, s.rec ? 'recovery' : 'rejection', s.T, s.Tv, s.P / 1000, s.B, s.X, s.D, s.be, s.nea, s.dl, s.tin, s.tout, s.ttd, s.U, s.A, s.Q / 1e6, s.ntu, s.eff, s.A > 0 ? s.Q / s.A / 1000 : 0]) });
    else tables.push({ title: 'Effect-by-effect results', columns: ['Effect', 'Heating T (°C)', 'Brine T (°C)', 'Vapour T (°C)', 'Pressure (kPa)', 'Feed in (kg/s)', 'Feed T (°C)', 'Brine out (kg/s)', 'Salinity (g/kg)', 'Vapour boiled (kg/s)', 'Flash vapour (kg/s)', 'BPE (K)', 'Demister+line (K)', 'ΔT (K)', 'U (W/m²·K)', 'Area (m²)', 'Duty (MW)', 'CaSO₄ envelope (g/kg)', 'Heat flux (kW/m²)'],
      rows: xs.map((n, i) => [n, r.Th[i], r.T[i], r.Tv[i], psat(r.Tv[i]) / 1000, r.c.Fi[i], r.c.Fi[i] > 0 ? r.tf[i] : null, r.B[i], r.X[i], r.V[i], r.fl[i], r.be[i], r.dl[i], r.dT[i], r.U[i], r.A[i], r.q[i] / 1e6, caso4Limit(r.T[i]), (r.U[i] * r.dT[i]) / 1000]),
      note: N > 1 && r.c.areaMode === 'equalArea' ? 'Temperature differences iterated until all effects have the same area.' : N > 1 && r.c.areaMode === 'flux' ? 'Prescribed-flux condition: temperature differences iterated until every effect carries the same heat flux q″ = U·ΔT.' : '' });
    tables.push({ title: 'Heat exchangers (LMTD and effectiveness–NTU)', columns: ['Exchanger', 'Duty (MW)', 'Hot side (°C)', 'Cold in (°C)', 'Cold out (°C)', 'LMTD (K)', 'U (W/m²·K)', 'Area (m²)', 'NTU', 'Effectiveness'], rows: R.hx.map((x) => [x.name, x.duty / 1e6, x.Th, x.tin, x.tout, nz(x.lmtd), x.U, x.area, x.ntu, x.eff]), note: 'For condensing duties ε = 1 − exp(−NTU); the NTU and LMTD routes give the same area.' });
    tables.push({ title: 'Streams and utilities', columns: ['Stream', 'Flow (kg/s)', 'Flow (t/h)', 'Temperature (°C)', 'Salinity (g/kg)'],
      rows: [['Feed (make-up)', R.Mf, R.Mf * kgh, p.Tsw, p.Xf], ['Distillate', R.Md, R.Md * kgh, R.Tprod, tdsD / 1000], ['Brine blow-down', R.Mb, R.Mb * kgh, R.Tbrine, R.Xb], ['Cooling water rejected', R.Mcw, R.Mcw * kgh, R.Tcw, p.Xf],
        ...(msf ? [['Recirculating brine', m.Mr, m.Mr * kgh, m.Tr, m.Xr]] : []), ...(R.ej ? [['Motive steam', R.ej.Mm, R.ej.Mm * kgh, R.ej.Tm, 0], ['Entrained vapour', R.ej.Mev, R.ej.Mev * kgh, R.ej.Tev, 0]] : []),
        ...(mvc ? [['Compressed vapour', r.cond, r.cond * kgh, R.comp.T2, 0]] : [[proc === 'medtvc' ? 'Heating vapour to effect 1' : 'Heating steam', proc === 'medtvc' ? R.ej.Mm + R.ej.Mev : R.Ms, (proc === 'medtvc' ? R.ej.Mm + R.ej.Mev : R.Ms) * kgh, R.Tsteam, 0]]), ['Vented vapour', R.vent, R.vent * kgh, R.Tlast, 0]] });
    tables.push({ title: 'Energy and exergy accounting', columns: ['Item', 'kW', 'kWh per m³ distillate'],
      rows: [['Heat supplied', R.Qin / 1000, R.sTh], ...Object.entries(R.pumps).map(([k, w]) => [`Electricity · ${k}`, w / 1000, w / 1000 / R.Qd]), ['Electricity · total', R.Wel / 1000, R.sEl], ['Equivalent electricity (heat valued as lost turbine work)', R.sEq * R.Qd, R.sEq], ['Minimum work of separation', R.wMin * R.Qd, R.wMin],
        ['Exergy supplied (heat + electricity)', R.exTot / 1000, R.exTot / 1000 / R.Qd], ...R.ex.map(([k, e]) => [`Exergy destroyed · ${k}`, e / 1000, e / 1000 / R.Qd])] });
    if (srcRows.length) tables.push({ title: srcTitle, columns: ['Quantity', 'Value'], rows: srcRows, note: srcNote });
    tables.push(...extraTables);
    tables.push({ title: 'Boiling-point elevation and brine vapour pressure: correlation versus Raoult’s law', columns: [unit, 'Brine T (°C)', 'Salinity (g/kg)', 'BPE correlation (K)', 'BPE Raoult, activity (K)', 'BPE Raoult, ideal (K)', 'Water activity a_w (–)', 'Water mole fraction x_w (–)', 'Vapour pressure a_w·p_sat (kPa)', 'Vapour pressure from correlation (kPa)'],
      rows: rl.map((q) => [q.n, q.T, q.X, q.corr, q.act, q.ideal, q.aw, q.xw, q.pR / 1000, q.pC / 1000]), note: `Raoult’s law: p = a_w·p_sat(T) with a_w = x_w (ideal) or exp(−φ·m/55.51) (osmotic coefficient φ, ion molality m). Used in the balances: ${bpeName}.` });
    tables.push({ title: 'Thermodynamic–economic model: unit water cost', columns: ['Item', '$/h', '$/m³'], rows: [['Capital recovery', wc.Zcap, wc.parts['Capital recovery']], ['Heat', wc.Cheat, wc.parts.Heat], ['Electricity', wc.Cel, wc.parts.Electricity], ['Operation and maintenance', wc.Zom, wc.parts['Operation and maintenance']], ['Chemicals', wc.Cchem, wc.parts.Chemicals], ['Total', wc.rate, wc.total],
      ['Installed capital (M$)', wc.capex / 1e6, null], ['Capital per m³/d of capacity ($)', wc.perCapacity, null], ['Lowest cost on the design map ($/m³)', null, opt.best.cost], [`… at ${grid.xlabel.toLowerCase()}`, opt.best.x, null], [`… at ${grid.ylabel.toLowerCase()}`, opt.best.y, null]],
      note: 'Cost = annualised capital of the heat-transfer surface (× installation factor) + heat + electricity + O&M + chemicals. The design-map optimum is a grid search over equal-ΔT re-designs within the scale limit' + (p.source === 'mhp' || p.source === 'ahp' ? '; the heat and electricity rows include the heat pump (its driving heat is priced like the heating steam, the heat-pump equipment itself is not costed), the map optimum does not.' : '.') });
    tables.push({ title: 'Exergy–economic analysis (cost rates per component)', columns: ['Component', 'Exergy destroyed (kW)', 'Share of fuel exergy (%)', 'Cost of destruction Ċ_D ($/h)', 'Capital + O&M rate Ż ($/h)', 'Ċ_D + Ż ($/h)', 'Exergoeconomic factor f (%)'],
      rows: [...xe.rows.map((q) => [q.name, q.ExD, 100 * q.share, q.CD, q.Z, q.CD + q.Z, 100 * q.f]), ['Product: minimum work of separation', xe.ExP, (100 * xe.ExP) / xe.ExF, xe.cF * xe.ExP, 0, xe.cF * xe.ExP, null], ['Fuel (heat + electricity) / total', xe.ExF, 100, xe.CF, xe.Z, xe.CP, null]],
      note: `Fuel exergy cost c_F = ${fmt(xe.cF, 3)} $/kWh, product exergy cost c_P = (Ċ_F + Ż)/Ėx_P = ${fmt(xe.cP, 3)} $/kWh, relative cost difference ${fmt(xe.r, 3)}. A low f means the component’s cost is dominated by irreversibility (spend capital there); a high f means capital dominates.${xe.overshoot > 0 ? ` The component estimates (mean-temperature formulas) added up to ${fmt(100 * xe.overshoot, 2)} % more than the exact total of fuel minus product exergy and were scaled to it.` : ''}` });
    if (su && su.vap.mTot > 0) tables.push({ title: 'Vapour inventories (initial condition and design state)', columns: [unit, 'Vapour-space volume (m³)', 'Design vapour temperature (°C)', 'Design pressure (kPa)', 'Vapour mass at cold start (kg)', 'Vapour mass at design (kg)'],
      rows: [...xs.map((n, i) => [n, su.vap.vol[i], su.vap.TvDes[i], psat(su.vap.TvDes[i]) / 1000, su.vap.m0[i], su.vap.mDes[i]]), ['Total', sum(su.vap.vol), null, null, sum(su.vap.m0), su.vap.mTot]], note: `Initial condition: every vapour space is saturated at the ${fmt(p.Tsw, 3)} °C seawater temperature (vacuum already pulled). The vapour space adds its latent and sensible capacity V·(λ·dρ_v/dT + ρ_v·c_p) to each effect/stage of the start-up model.` });

    const b = R.balance, sA = R.area / p.Md;
    return {
      summary: `${name} with ${N} ${msf ? 'stages' : N > 1 ? 'effects' : 'effect'} produces ${fmt(R.Qd * 24, 4)} m³/d of distillate at ${fmt(100 * R.rec, 3)} % recovery${mvc ? ` using ${fmt(R.sEl, 3)} kWh/m³ of electricity` : ` with a gain-output ratio of ${fmt(R.GOR, 3)} (${fmt(R.sTh, 3)} kWh of heat and ${fmt(R.sEl, 3)} kWh of electricity per m³)`} on ${fmt(R.area, 4)} m² of heat-transfer surface.`,
      warnings: W,
      kpis: [
        { label: 'Distillate', value: R.Qd * 24, unit: 'm³/d' }, { label: 'Recovery', value: 100 * R.rec, unit: '%' },
        mvc ? { label: 'Compressor power', value: R.Wcomp / 1000, unit: 'kW' } : { label: 'Gain-output ratio', value: R.GOR, unit: 'kg/kg', help: 'Distillate per kg of heating (motive) steam' },
        mvc ? { label: 'Compression ratio', value: R.comp.ratio, unit: '–' } : { label: 'Performance ratio', value: R.PR, unit: 'kg/2326 kJ' },
        mvc ? { label: 'Compressor discharge temperature', value: R.comp.T2, unit: '°C' } : { label: 'Steam consumption', value: R.Ms * kgh, unit: 't/h' },
        { label: 'Specific heat', value: R.sTh, unit: 'kWh/m³', help: `${fmt(R.sThKJ, 4)} kJ per kg of distillate` }, { label: 'Specific electricity', value: R.sEl, unit: 'kWh/m³', status: mvc && R.sEl > 14 ? 'warn' : 'ok' },
        { label: 'Equivalent-electric energy', value: R.sEq, unit: 'kWh/m³', help: 'Electricity plus the turbine work the heating steam could have produced' },
        { label: 'Heat-transfer area', value: R.area, unit: 'm²' }, { label: 'Specific area', value: sA, unit: 'm² per m³/d', help: `${fmt(R.sArea, 4)} m² per kg/s of distillate` },
        { label: 'Top brine temperature', value: R.Ttop, unit: '°C', status: R.Ttop > tLim ? 'bad' : 'ok' }, { label: 'Brine salinity', value: R.Xb, unit: 'g/kg', status: R.Xb > 75 ? 'warn' : 'ok' },
        ...(mvc ? [{ label: 'Feed temperature after preheater', value: r.TfM, unit: '°C' }, { label: 'Auxiliary heat', value: R.Qaux / 1000, unit: 'kW', status: R.Qaux > 0 ? 'warn' : 'ok' }] : [{ label: 'Cooling water', value: out.coolingWater, unit: 'm³/h' }, { label: 'Condenser load', value: R.Qcond / 1e6, unit: 'MW' }]),
        ...(R.ej ? [{ label: 'Entrainment ratio', value: R.ej.Ra, unit: 'kg motive/kg', status: R.ej.Ra > 4 ? 'warn' : 'ok' }, { label: 'Ejector compression ratio', value: R.ej.Cr, unit: '–', status: R.ej.Cr < 1.81 ? 'warn' : 'ok' }] : []),
        { label: 'Second-law efficiency', value: 100 * R.eta2, unit: '%' }, { label: 'Distillate TDS', value: tdsD, unit: 'mg/L' },
        { label: 'Unit water cost', value: wc.total, unit: '$/m³', help: 'Thermodynamic–economic model: capital recovery + heat + electricity + O&M + chemicals' }, { label: 'Exergy cost of the product', value: xe.cP, unit: '$/kWh', help: `Fuel exergy costs ${fmt(xe.cF, 3)} $/kWh` },
        { label: 'Lowest cost on the design map', value: opt.best.cost, unit: '$/m³', status: !opt.best.base && opt.saving > 0.03 ? 'warn' : 'ok', help: `Grid optimum at ${grid.xlabel.toLowerCase()} ${fmt(opt.best.x, 3)}, ${grid.ylabel.toLowerCase()} ${fmt(opt.best.y, 3)}${opt.best.base ? ' (the present design)' : ''}` },
        { label: 'Optimised unit water cost', value: od.best.cost, unit: '$/m³', help: `${od.method}: ${grid.xlabel.toLowerCase()} ${fmt(od.best.x, 3)}, ${grid.ylabel.toLowerCase()} ${fmt(od.best.y, 4)}; ${od.evals} re-designs beyond the grid, within the ${tLim} °C scale limit` },
        { label: 'Mean heat flux', value: out.heatFlux, unit: 'kW/m²', help: 'Evaporator / stage-condenser duty per m² of surface' }, { label: `Boiling-point elevation, last ${msf ? 'stage' : 'effect'}`, value: msf ? m.stages[N - 1].be : r.be[N - 1], unit: 'K', help: `Basis: ${bpeName}; Raoult (activity) ${fmt(rlLast.act, 3)} K, correlation ${fmt(rlLast.corr, 3)} K` },
        ...(su && su.vap.mTot > 0 ? [{ label: 'Vapour inventory at design', value: su.vap.mTot, unit: 'kg', help: `${fmt(sum(su.vap.m0), 3)} kg at the cold start` }] : []),
        ...extraKpis,
        ...(su && su.t95 !== null ? [{ label: 'Start-up to 95 %', value: su.t95, unit: 'min', help: 'Time for the lumped model to reach 95 % of the design temperature rise' }] : []),
        ...(sd && sd.t95 !== null ? [{ label: 'Cool-down to 50 % after steam trip', value: sd.t95, unit: 'min', help: 'Time until half of the design temperature rise is lost when the heating steam stops' }] : []),
      ],
      recommendations: [
        R.Ttop > tLim ? 'Lower the top temperature below the scale limit or select a stronger scale-control method.' : null,
        !msf && !mvc && p.arr === 'forward' && !p.preheat ? 'Add feed preheaters: in forward feed all the feed must otherwise be heated in the first effect, which costs steam.' : null,
        proc === 'med' && R.GOR < 0.75 * N ? 'The gain-output ratio is low for this number of effects — check the feed-preheater setting and the brine salinity (recovery).' : null,
        proc === 'med' && N >= 6 ? 'If medium-pressure steam is available, a thermo-compressor (MED-TVC) raises the gain-output ratio by 30–60 %.' : null,
        msf && R.PR < 7.5 ? 'Raise the performance ratio with more stages or a higher top brine temperature (see the sensitivity map).' : null,
        mvc && R.sEl > 12 ? 'Reduce the condensing − boiling ΔT or improve compressor efficiency: both lower the electricity demand directly.' : null,
        R.Mcw > 6 * R.Md ? 'Cooling-water demand is high; a warmer last effect/stage or more effects reduces the heat rejected.' : null,
        'Send the brine to suite 2 (Brine chemistry) for a full scaling check and to suite 5 (Sea discharge) — it is warm and saline.',
        !opt.best.base && opt.saving > 0.03 ? `The cost map points to ${grid.xlabel.toLowerCase()} ${fmt(opt.best.x, 3)} and ${grid.ylabel.toLowerCase()} ${fmt(opt.best.y, 3)} (${fmt(100 * opt.saving, 2)} % cheaper water with the present prices).` : null,
        'Use suite 13 (Economics) for the full project cash flow; the unit cost here covers only the thermal block.',
      ].filter(Boolean),
      plots: shown, tables,
      balances: [
        { name: 'Water + salt mass (kg/s)', in: b.massIn, out: b.massOut }, { name: 'Salt (kg/s)', in: (b.saltIn) / 1000, out: (b.saltOut) / 1000 }, { name: 'Energy (MW)', in: b.eIn / 1e6, out: b.eOut / 1e6 },
        ...(msf ? [] : [{ name: 'Distillate = Σ vapour boiled − vent (kg/s)', in: sum(r.V) - r.vent, out: r.Md }]),
        { name: 'Exergy cost rates: Σ(Ċ_D + Ż) + product = Ċ_F + Ż ($/h)', in: sum(xe.rows.map((q) => q.CD + q.Z)) + xe.cF * xe.ExP, out: xe.CP }, ...extraBal,
      ],
      outputs: out,
    };
  },

  mesh: [
    { name: 'Number of effects (MED / MED-TVC)', keys: ['N'], min: 2, note: 'A physical resolution study: how the result approaches the many-effect limit.', metrics: [{ label: 'Specific heat', unit: 'kWh/m³', get: (r) => r.outputs.secThermal }, { label: 'Heat-transfer area', unit: 'm²', get: (r) => r.outputs.area }] },
    { name: 'Number of stages (MSF)', keys: ['Nst'], min: 6, metrics: [{ label: 'Specific heat', unit: 'kWh/m³', get: (r) => r.outputs.secThermal }, { label: 'Heat-transfer area', unit: 'm²', get: (r) => r.outputs.area }] },
    { name: 'Heat-exchanger segments', keys: ['nSeg'], min: 1, metrics: [{ label: 'Heat-transfer area', unit: 'm²', get: (r) => r.outputs.area }] },
    { name: 'Start-up time steps', keys: ['nt'], min: 20, metrics: [{ label: 'Start-up time to 95 %', unit: 'min', get: (r) => r.outputs.startUpMin ?? NaN }] },
  ],

  calibration: {
    note: 'Fit the heat-transfer multiplier, the heat loss and the condenser approach to plant or pilot data. Each row is one steady operating point (seawater temperature, heating-steam or top brine temperature and distillate production); the measurements are the specific heat consumption (for MVC: specific electricity), the heat-transfer area needed per unit of production and the cooling-water flow. Use rows covering different loads and seasons.',
    params: [{ key: 'fU', label: 'Heat-transfer coefficient multiplier', lo: 0.4, hi: 1.8 }, { key: 'heatLoss', label: 'Heat loss (% of duty)', lo: 0, hi: 10 }, { key: 'ttdCond', label: 'Condenser approach (K)', lo: 1, hi: 10 }],
    columns: [{ key: 'Tsw', label: 'Seawater temperature', unit: '°C' }, { key: 'Ts', label: 'Heating-steam temperature', unit: '°C' }, { key: 'Md', label: 'Distillate', unit: 'm³/d' }, { key: 'sth', label: 'Specific energy', unit: 'kWh/m³' }, { key: 'sA', label: 'Specific area', unit: 'm² per m³/d' }, { key: 'Qcw', label: 'Cooling water', unit: 'm³/h' }],
    targets: [{ key: 'sth', label: 'Specific energy', unit: 'kWh/m³' }, { key: 'sA', label: 'Specific area', unit: 'm² per m³/d' }, { key: 'Qcw', label: 'Cooling-water flow', unit: 'm³/h' }],
    model(v) {
      const key = `${v.Tsw}|${v.Ts}|${v.Md}|${v.Xf}`;
      if (!calWarm.has(key)) { if (calWarm.size > 200) calWarm.clear(); calWarm.set(key, {}); }
      const q = simulateThermal(v, { areaMode: 'equalDT', warm: calWarm.get(key), ...(v.proc === 'msf' ? { TBT: v.Ts > 80 ? v.Ts : v.TBT } : {}) }); // tight tolerance (smooth finite-difference Jacobians), warm-started per operating point
      return { sth: q.proc === 'mvc' ? q.sEl : q.sTh, sA: q.area / v.Md, Qcw: (q.Mcw * 3600) / density(v.Tsw, v.Xf) };
    },
    get sample() { return (this._s ||= synth(5, [[24, 68, 18000], [26, 70, 20000], [28, 70, 20000], [30, 72, 21000], [22, 66, 16000], [27, 69, 19000], [31, 71, 20000], [25, 67, 17000]])); },
    get validationSample() { return (this._v ||= synth(17, [[23, 69, 19500], [29, 71, 20500], [26, 66, 16500], [32, 72, 21500], [25, 70, 18500], [28, 68, 17500]])); },
  },

  verify() {
    const d = defaultsOf(suite), C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    add('Saturation pressure at 100 °C', 101.325, psat(100) / 1000, 0.15, 'IAPWS-IF97 region 4 (kPa); 100 °C on ITS-90 gives 101.42 kPa');
    add('Latent heat at 100 °C', 2257, latentHeat(100) / 1000, 3, 'kJ/kg, steam tables');
    add('Antoine equation agrees with IAPWS at 60 °C', 0, antoine(60) / psat(60) - 1, 5e-3, 'Relative difference of the two saturation-pressure equations');
    add('Clausius–Clapeyron slope at 60 °C', 1, (latentHeat(60) * psat(60)) / (RW * K(60) ** 2) / dPdT(60), 0.02, 'dP/dT = λP/(R_v T²) against the numerical slope of the IAPWS curve');
    add('tsat inverts psat', 73.5, tsat(psat(73.5)), 1e-6, 'Property consistency');
    // single effect by hand: Ms·λs = F·(h_b − h_f) + D·(h_v − h_b)
    const one = simulateThermal(d, { N: 1, preheat: false, Ts: 70, Tn: 62, heatLoss: 0, ventPct: 0, areaMode: 'equalDT' }), r1 = one.r;
    const hand = (r1.c.F * (hL(r1.T[0], r1.X[0]) - hL(r1.tf[0], d.Xf)) + r1.V[0] * (hV(r1.Tv[0]) + CPV * r1.be[0] - hL(r1.T[0], r1.X[0]))) / latentHeat(70);
    add('Single-effect steam demand equals the hand energy balance', hand, one.Ms, 1e-6 * hand, 'Ms·λs = F·(h_b − h_f) + D·(h_v − h_b), kg/s');
    add('Single effect: GOR below one', 1, one.GOR < 1 && one.GOR > 0.75 ? 1 : 0, 0, `GOR = ${fmt(one.GOR, 4)}; sensible feed heating and BPE keep it under 1`);
    const med = simulateThermal(d), bm = med.balance;
    add('MED total mass balance closes', 0, (bm.massIn - bm.massOut) / bm.massIn, 1e-9, 'Feed + cooling water = distillate + brine + reject + vent');
    add('MED salt balance closes', 0, (bm.saltIn - bm.saltOut) / bm.saltIn, 1e-9, 'F·X_f = B·X_b');
    add('MED energy balance closes', 0, (bm.eIn - bm.eOut) / bm.eIn, 1e-7, 'Steam heat + seawater enthalpy = all outlet enthalpies + losses');
    add('MED equal-area iteration converged', 0, Math.max(...med.r.A) / Math.min(...med.r.A) - 1, 1e-6, 'Largest ÷ smallest effect area − 1');
    add('MED GOR ≈ 0.8–0.9 × number of effects', 0.85, med.GOR / med.N, 0.1, `Literature range for 8–12 effects; got GOR ${fmt(med.GOR, 4)} with ${med.N} effects`);
    const tvc = simulateThermal({ ...d, ...suite.presets[1].values });
    add('MED-TVC GOR within the 9–16 literature range', 12.5, tvc.GOR, 3.5, `10 effects, suction at effect 5, entrainment ratio ${fmt(tvc.ej.Ra, 3)}`);
    add('MED-TVC energy balance closes', 0, (tvc.balance.eIn - tvc.balance.eOut) / tvc.balance.eIn, 1e-7, 'Motive-steam heat is the only external heat');
    add('Ejector correlation, hand value', 0.296 * (31.2 ** 1.19 / 15.76 ** 1.04) * (1000 / 15.76) ** 0.015 * ((3e-7 * 1e6 - 0.9 + 1.6101) / (2e-8 * 55 * 55 - 0.033 + 1.0047)), entrainmentRatio(1000, 31.2, 15.76, 55), 1e-9, 'Ra = 0.296 Ps^1.19/Pev^1.04 (Pm/Pev)^0.015 PCF/TCF');
    const msf = simulateThermal({ ...d, ...suite.presets[2].values }), bs = msf.balance;
    add('MSF performance ratio, 24 stages at TBT 110 °C', 8.75, msf.PR, 1.25, 'Literature 8–10 kg per 2326 kJ for large units; the textbook 21 + 3 stage arrangement gives 7.5–8');
    add('MSF brine-heater temperature rise = rejection stages × stage drop', msf.m.nRej * msf.m.dTs + (msf.p.TnMsf - msf.m.Tr), msf.p.TBT - msf.m.t[1], 0.35, 'Analytical result for equal stage drops and equal tube/flashing flow (K)');
    add('MSF energy balance closes', 0, (bs.eIn - bs.eOut) / bs.eIn, 1e-8, 'Brine-heater duty + seawater = distillate + blow-down + cooling water');
    add('MSF salt balance closes', 0, (bs.saltIn - bs.saltOut) / bs.saltIn, 1e-9, 'Make-up salt leaves with the blow-down');
    const mvc = simulateThermal({ ...d, ...suite.presets[4].values }), bv = mvc.balance;
    add('MVC specific electricity within 7–12 kWh/m³', 9.5, mvc.sEl, 2.5, 'Single effect, literature range');
    add('MVC energy balance closes', 0, (bv.eIn - bv.eOut) / bv.eIn, 1e-7, 'Shaft work + feed enthalpy = product + brine enthalpies + losses');
    const hx = condenserHX('check', 60, 100, 36, 30, 50, 2500, 1);
    add('Effectiveness–NTU equals the LMTD result', hx.eff, hx.effNtu, 1e-9, 'Condenser: ε = 1 − exp(−NTU)');
    const hx8 = condenserHX('check', 60, 100, 36, 30, 50, 2500, 32);
    add('Segmented exchanger converges to the LMTD area', 0, hx8.area / hx.area - 1, 2e-3, 'Only the weak cp(T) variation separates the two');
    const ideal = simulateThermal(d, { fBPE: 0, heatLoss: 0, ventPct: 0, rhoP: 80, thick: 50, Vdem: 0.9, linePct: 0, fast: true });
    add('Losses reduce the GOR', 1, ideal.GOR > med.GOR ? 1 : 0, 0, `Without BPE, demister, line and heat losses GOR rises from ${fmt(med.GOR, 4)} to ${fmt(ideal.GOR, 4)}`);
    add('Second-law efficiency is between 0 and 1', 1, [med, tvc, msf, mvc].every((q) => q.eta2 > 0 && q.eta2 < 1) ? 1 : 0, 0, `MED ${fmt(100 * med.eta2, 3)} %, MED-TVC ${fmt(100 * tvc.eta2, 3)} %, MSF ${fmt(100 * msf.eta2, 3)} %, MVC ${fmt(100 * mvc.eta2, 3)} %`);
    // ---- Raoult's law
    add('Raoult’s law reproduces the ebullioscopic constant of water', 0.512, bpeRaoult(100, 0.5, 'ideal') / ionMolality(0.5), 0.01, 'Dilute limit at 100 °C: ΔT_b / m → R·T²·M_w/Δh_vap = 0.512 K·kg/mol');
    add('Raoult’s law with water activity agrees with the seawater BPE correlation', bpe(70, 50), bpeRaoult(70, 50, 'activity'), 0.05, '70 °C, 50 g/kg (K); the ideal mole-fraction form is ' + fmt(bpeRaoult(70, 50, 'ideal'), 3) + ' K');
    const ra = simulateThermal(d, { bpeModel: 'activity', fast: true }), La = ra.N - 1;
    add('Selected Raoult basis is the one used in the effect balances', bpeRaoult(ra.r.T[La], ra.r.X[La], 'activity'), ra.r.be[La], 1e-9, 'Last-effect BPE of a run with the Raoult (activity) option (K)');
    add('Raoult vapour pressure: p = x_w·p_sat for the ideal basis', (55.508 / (55.508 + ionMolality(60))) * psat(60), raoultPressure(60, 60, 'ideal'), 1e-9, 'Hand value at 60 °C, 60 g/kg (Pa)');
    // ---- prescribed heat flux
    const fx = simulateThermal(d, { areaMode: 'flux' }), qf = fx.r.q.map((q, i) => q / fx.r.A[i]);
    add('Equal-heat-flux design: same q″ in every effect', 0, Math.max(...qf) / Math.min(...qf) - 1, 1e-6, `q″ = U·ΔT = ${fmt(qf[0] / 1000, 4)} kW/m² in all ${fx.N} effects`);
    add('Equal-heat-flux design keeps the energy balance and the last-effect temperature', 0, Math.abs((fx.balance.eIn - fx.balance.eOut) / fx.balance.eIn) + Math.abs(fx.r.T[fx.N - 1] - d.Tn) / 100, 1e-6, 'Relative energy imbalance + temperature mismatch');
    // ---- vapour inventory
    const su0 = startUp(med, { ...d, vapVol: 0 }), su1 = startUp(med, { ...d, vapVol: 0.3 });
    add('Vapour inventory equals the ideal-gas hand value', (psat(med.r.Tv[0]) * 0.01801528 * 0.3 * med.r.A[0]) / (8.314462618 * K(med.r.Tv[0])), su1.vap.mDes[0], 2e-4 * su1.vap.mDes[0], 'm = p_sat·M·V/(R·T) for the first effect (kg)');
    add('A larger vapour space slows the start-up', 1, su1.t95 > su0.t95 && su1.vap.inv[0] < 50 && Math.abs(su1.vap.inv[su1.vap.inv.length - 1] - 100) < 8 ? 1 : 0, 0, `95 % reached after ${fmt(su0.t95, 4)} min without and ${fmt(su1.t95, 4)} min with 0.3 m³/m² of vapour space; the inventory rises from ${fmt(su1.vap.inv[0], 3)} % to ${fmt(su1.vap.inv[su1.vap.inv.length - 1], 4)} % of design`);
    // ---- heat pump
    const hm = heatPump(med, { ...d, source: 'mhp', hpEta: 100 }), ha = heatPump(med, { ...d, source: 'ahp' });
    add('Mechanical heat pump at 100 % second-law efficiency reaches the Carnot COP', K(hm.Th) / (hm.Th - hm.Tl), hm.cop, 1e-12, 'COP = T_h/(T_h − T_l)');
    add('Absorption heat pump: reversible three-temperature COP, hand value', (1 - K(ha.Tl) / K(ha.Tg)) / (1 - K(ha.Tl) / K(ha.Th)), ha.copRev, 1e-9, '(1 − T_l/T_g)/(1 − T_l/T_h); the real COP ' + fmt(ha.cop, 3) + ' lies between 1 and this limit');
    add('Heat-pump energy balance closes', 0, (ha.Qe + ha.drive - ha.Qhp) / ha.Qhp + (ha.Qhp + ha.Qdirect - med.Qin) / med.Qin + (ha.cop > 1 && ha.cop < ha.copRev ? 0 : 1), 1e-12, 'Source heat + driving heat = heat delivered; delivered + direct = plant demand');
    // ---- cogeneration
    add('Entropy of saturated steam at 100 °C', 7.355, sVapour(100) / 1000, 0.02, 's_g = ∫c_p dT/T + λ/T (kJ/kg·K), steam tables 7.355');
    const cgI = cogeneration(1e8, 70, 0, { ...d, etaTurb: 100 }), cgR = cogeneration(med.Qin, med.Tsteam, med.Wel, d);
    add('Ideal turbine: lost work = Carnot value of the heat + condensate exergy', cgI.carnot + cgI.condensate, cgI.lostPerHeat, 1e-3, 'Second-law identity for wet-steam extraction at 70 °C with a 38 °C condenser');
    add('Cogeneration: real lost work is below the reversible limit and steam flows balance', 1, cgR.lostPerHeat < cgR.carnot + cgR.condensate && cgR.lostPerHeat > 0 && Math.abs(cgR.mExt * cgR.qExt - med.Qin) < 1e-6 * med.Qin && cgR.euf > cgR.etaPower && cgR.euf < 1 ? 1 : 0, 0, `${fmt(cgR.lostPerHeat, 3)} kWh of power per kWh of heat at ${fmt(cgR.Te, 3)} °C; energy-utilisation factor ${fmt(cgR.euf, 3)} versus ${fmt(cgR.etaPower, 3)} for power only`);
    // ---- costing
    add('Capital-recovery factor, 6 % over 25 years', 0.078227, crf(0.06, 25), 1e-6, 'i(1+i)ⁿ/((1+i)ⁿ − 1)');
    const wcV = waterCost(med, d), xeV = exergoEconomics(med, d, wcV);
    add('Unit water cost equals the hand sum of its parts', (d.cArea * med.area * d.capFactor * (crf(0.06, 25) + d.omPct / 100)) / (8760 * 0.92) / med.Qd + (d.cHeat * med.sTh) / 1000 + d.cElec * med.sEl + d.cChem, wcV.total, 1e-9, 'Capital + O&M + heat + electricity + chemicals ($/m³)');
    add('Exergy-cost balance closes', 0, xeV.closure, 1e-12, 'Σ(Ċ_D + Ż) over the components + cost of the product exergy = Ċ_F + Ż');
    add('Exergy-economic and thermodynamic-economic routes give the same water cost', wcV.total, xeV.perM3 + d.cChem, 1e-9, 'Ċ_P per m³ + chemicals ($/m³); product exergy costs more than fuel exergy: ' + fmt(xeV.cP, 3) + ' vs ' + fmt(xeV.cF, 3) + ' $/kWh');
    // ---- process optimisation
    const gr = designGrid(d), og = optimiseGrid(gr, d, d.tbtLimMED, wcV.total), feas = gr.cells.flatMap((row, j) => row.map((q, i) => (q && q.Ttop <= d.tbtLimMED + 1e-9 ? og.cost[j][i] : Infinity)));
    add('Process optimisation returns the cheapest feasible design', Math.min(wcV.total, ...feas), og.best.cost, 1e-12, `Grid search over ${og.nCells} re-designs, ${og.nFeas} within the scale limit ($/m³)`);
    const odV = optimiseDesign(gr, d, d.tbtLimMED, og), mvV = { ...d, proc: 'mvc' }, grM = designGrid(mvV), ogM = optimiseGrid(grM, mvV, 1e9, waterCost(simulateThermal(mvV), mvV).total), odM = optimiseDesign(grM, mvV, 1e9, ogM);
    add('Optimiser (Nelder–Mead) is at least as cheap as the grid optimum and respects the scale limit', 1, odV.best.cost <= og.best.cost + 1e-12 && (odV.best.base || odV.best.q.Ttop <= d.tbtLimMED + 1e-9) && odV.evals > 0 ? 1 : 0, 0, `MED: grid ${fmt(og.best.cost, 5)} → optimiser ${fmt(odV.best.cost, 5)} $/m³ in ${odV.evals} further re-designs`);
    add('Optimiser improves on the grid for two continuous variables (MVC)', 1, odM.best.cost <= ogM.best.cost + 1e-12 && odM.evals > 5 ? 1 : 0, 0, `MVC: grid ${fmt(ogM.best.cost, 5)} → optimiser ${fmt(odM.best.cost, 5)} $/m³ at ${fmt(odM.best.x, 4)} °C and ${fmt(odM.best.y, 4)} K`);
    // ---- thermal + RO hybrid
    const hyM = hybridRO(med, d, 5, scaleIons(cloneIons(d.ions), 1)), msfV = simulateThermal({ ...d, ...suite.presets[2].values }), hyS = hybridRO(msfV, { ...d, ...suite.presets[2].values, roFeed: 'intake', roShare: 30 }, 5, scaleIons(cloneIons(d.ions), 1));
    add('MED–RO hybrid: blended TDS is the flow-weighted mean', (med.Qd * 5 + hyM.Qp * hyM.tdsP) / (med.Qd + hyM.Qp), hyM.tdsBlend, 1e-9, `Distillate 5 mg/L + RO permeate ${fmt(hyM.tdsP, 3)} mg/L (mg/L)`);
    add('MED–RO hybrid: RO water balance closes and the RO share is met', 0, Math.abs(hyM.Qf - hyM.Qp - hyM.ro.conc.Q) / hyM.Qf + Math.abs(hyM.share - 0.5), 1e-3, 'Feed = permeate + concentrate; RO supplies 50 % of the blend');
    add('MSF–RO hybrid: combined energy lies between the two processes', 1, hyS.secEq < msfV.sEq && hyS.secEq > hyS.ro.sec && hyS.recFeed > msfV.rec === (hyS.ro.overallRec > msfV.rec) ? 1 : 0, 0, `${fmt(hyS.secEq, 3)} kWh/m³ equivalent for the blend versus ${fmt(msfV.sEq, 3)} (MSF) and ${fmt(hyS.ro.sec, 3)} (RO)`);
    add('Warm reject feed lowers the RO pressure', 1, hyM.cold && hyM.ro.p1.Pf < hyM.cold.p1.Pf && hyM.Tro > med.T0 ? 1 : 0, 0, `${fmt(hyM.ro.p1.Pf, 3)} bar at ${fmt(hyM.Tro, 3)} °C versus ${fmt(hyM.cold?.p1.Pf ?? 0, 3)} bar on cold seawater`);
    // ---- stress cases: seeded random designs over the whole input range
    const st = stressCases(suite, 60, 20260);
    add('Stress cases: random designs across the input ranges solve cleanly or are rejected with an explanation', 0, st.fails.length, 0, `${st.n} seeded cases: ${st.solved} solved with finite results, closed balances and no negative flow, area or energy; ${st.rejected} rejected with a message that names the input to change${st.fails.length ? `; first failure (case ${st.fails[0].k}): ${st.fails[0].why}` : ''}`);
    add('Stress cases: a meaningful share of the sampled designs is feasible', 1, st.solved >= 0.15 * st.n ? 1 : 0, 0, `${st.solved} of ${st.n} sampled designs solve; the rest are infeasible combinations that are rejected`);
    return C;
  },
};

/**
 * Seeded stress cases: random designs over the declared input ranges (log-uniform where a range spans decades, range ends included),
 * select and yes/no inputs rotated so that their values meet in changing combinations. Every case must either solve — no NaN, Infinity
 * or undefined anywhere in the result, every reported balance closed to 1e-3, no negative flow, area or energy — or be rejected with
 * an Error whose message says in plain language what to change. Returns { n, solved, rejected, fails: [{ k, why, values }] }.
 */
export function stressCases(suiteDef, n = 60, seed = 1, fixed = {}) {
  const groups = suiteDef.inputs, fields = groups.flatMap((g) => g.fields.map((fl) => ({ ...fl, g }))), base = Object.fromEntries(fields.map((fl) => [fl.key, fl.type === 'ions' || fl.type === 'table' ? JSON.parse(JSON.stringify(fl.value)) : fl.value ?? null]));
  const g = rng(seed), cats = fields.filter((fl) => (fl.type === 'select' || fl.type === 'bool') && !(fl.key in fixed)), out = { n, solved: 0, rejected: 0, fails: [] };
  const bad = (x, path, acc, depth = 0) => { // non-finite numbers, undefined and their printed forms
    if (acc.length > 3 || depth > 8 || x === null) return;
    if (typeof x === 'number') { if (!Number.isFinite(x)) acc.push(`${path} = ${x}`); } else if (x === undefined) acc.push(`${path} undefined`);
    else if (typeof x === 'string') { if (/\b(NaN|undefined|Infinity)\b/.test(x)) acc.push(`${path}: "${x.slice(0, 60)}"`); }
    else if (Array.isArray(x) || ArrayBuffer.isView(x)) for (let i = 0; i < x.length; i++) bad(x[i], `${path}[${i}]`, acc, depth + 1);
    else if (typeof x === 'object') for (const k of Object.keys(x)) if (!k.startsWith('_')) bad(x[k], `${path}.${k}`, acc, depth + 1);
  };
  const signed = /net|power-to-water|powerToWater|energy per m³ of permeate|npv|irr|lsi|index|change|°C|temperature|potential|voltage/i, sized = /^(m²|m³\/h|m³\/d|kW|MW|kWh\/m³|kg\/s|kg\/h|t\/h|t\/d|L\/m²·h|%|bar|mg\/L|g\/kg)$/;
  for (let k = 0; k < n; k++) {
    const ov = { ...fixed };
    cats.forEach((fl, i) => { const vals = fl.type === 'bool' ? [false, true] : fl.options.map((o) => o.value); ov[fl.key] = vals[(k * (i + 1) + Math.floor(k / vals.length) * i + g.int(vals.length) * (k % 3 === 0 ? 1 : 0)) % vals.length]; });
    for (const fl of fields) {
      if (fl.type || fl.key in fixed || !Number.isFinite(fl.min) || !Number.isFinite(fl.max)) continue;
      const vis = { ...base, ...ov }, u = g.uniform(), e = g.uniform();
      if ((fl.g.showIf && !fl.g.showIf(vis)) || (fl.showIf && !fl.showIf(vis))) continue;
      const hi = fl.g.tab === 'mesh' ? fl.min + 0.03 * (fl.max - fl.min) : fl.max; // discretisation inputs stay at the coarse end to keep the check fast
      const x = e < 0.1 ? fl.min : e < 0.2 ? hi : e < 0.3 ? fl.value : fl.min > 0 && hi / fl.min >= 100 ? fl.min * (hi / fl.min) ** u : fl.min + u * (hi - fl.min);
      ov[fl.key] = fl.step === 1 ? Math.round(x) : x;
    }
    const why = [];
    let res = null;
    try { res = suiteDef.run({ ...JSON.parse(JSON.stringify(base)), ...ov, _lean: true }, undefined); }
    catch (err) {
      const m = String(err?.message || '');
      if (err?.constructor !== Error || m.length < 40 || /NaN|undefined|Infinity|not a function|Cannot read|is not (a|an|iterable|defined)|brent|bracket/i.test(m) || !/raise|lower|reduce|increase|use |choose|select|switch|simulate|shorten|lengthen|enable|add |take /i.test(m)) why.push(`unexplained error: ${m.slice(0, 120)}`);
      else out.rejected++;
    }
    if (res && typeof res.then === 'function') res = null; // asynchronous variants are not part of this check
    if (res) {
      bad({ kpis: res.kpis, tables: (res.tables || []).map((t) => t.rows), outputs: res.outputs, balances: res.balances, summary: res.summary, warnings: res.warnings }, 'result', why);
      for (const pl of res.plots || []) bad(pl.type === 'field' ? pl.z : (pl.series || []).map((q) => [q.x, q.y, q.values].filter(Boolean)), `plot "${pl.title}"`, why);
      for (const b of res.balances || []) if (!(Math.abs(b.in - b.out) <= 1e-3 * Math.max(1, Math.abs(b.in)))) why.push(`balance "${b.name}" open: ${b.in} vs ${b.out}`);
      for (const q of res.kpis || []) if (typeof q.value === 'number' && q.value < -1e-9 && sized.test(q.unit || '') && !signed.test(q.label)) why.push(`negative ${q.label}: ${q.value}`);
      const neg = (o, path) => { for (const [key, val] of Object.entries(o || {})) { if (typeof val === 'number') { if (val < -1e-9 && !signed.test(key) && !/^(T|P|pH)$/.test(key) && !(key === 'secElec' && /pro/.test(res.outputs.process || ''))) why.push(`negative output ${path}${key}: ${val}`); } else if (val && typeof val === 'object' && !Array.isArray(val)) neg(val, `${path}${key}.`); } };
      neg(res.outputs, '');
      if (!why.length) out.solved++;
    }
    if (why.length) out.fails.push({ k, why: why.slice(0, 3).join('; '), values: ov });
  }
  return out;
}

/** Synthetic plant data: the model with slightly different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([Tsw, Ts, Md]) => {
    const q = suite.calibration.model({ ...d, fU: 0.86, heatLoss: 2.6, ttdCond: 4.2, Tsw, Ts, Md });
    return { Tsw, Ts, Md, sth: +(q.sth * (1 + g.normal(0, 0.006))).toFixed(2), sA: +(q.sA * (1 + g.normal(0, 0.012))).toFixed(4), Qcw: +(q.Qcw * (1 + g.normal(0, 0.015))).toFixed(0) };
  });
}

export default suite;
