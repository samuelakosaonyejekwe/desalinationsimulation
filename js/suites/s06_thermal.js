// Suite 6 — Thermal desalination.
// Rigorous effect-by-effect / stage-by-stage steady-state models of multi-effect distillation (forward and
// parallel/cross feed, optional thermo-compressor), multi-stage flash (once-through and brine recirculation)
// and mechanical vapour compression. Every effect/stage closes total-mass, salt and enthalpy balances with
// boiling-point elevation, non-equilibrium allowance, demister/line losses, distillate and brine flashing,
// feed preheaters, a final condenser, published U(T) correlations, exergy accounting and a lumped start-up model.
import { brent, clamp, linspace, sum, rng, fmt, rk4 } from '../core/num.js';
import { density, cp, psat, tsat, antoine, latentHeat, bpe, enthalpyLiquid as hL, enthalpyVapour as hV, osmoticPressure, salinityFromTDS, tdsFromSalinity, KELVIN } from '../core/props.js';
import { ION_IDS, WATERS, cloneIons, tds, scaleIons } from '../core/water.js';

const CPV = 1884, RW = 461.52, GAMMA = 1.32, RHO_REF = 994; // vapour cp J/kg·K, gas constant of steam, isentropic exponent, product density kg/m³
const K = (T) => T + KELVIN;
const fill = (n, x) => new Array(n).fill(x);
const lmtd = (d1, d2) => (Math.abs(d1 - d2) < 1e-9 ? 0.5 * (d1 + d2) : (d1 - d2) / Math.log(d1 / d2));
const dPdT = (T) => (psat(T + 0.05) - psat(T - 0.05)) / 0.1;
/** Temperature of liquid of salinity S with specific enthalpy h (inverse of props.enthalpyLiquid). */
const tFromH = (h, S) => { let T = h / cp(40, S); for (let i = 0; i < 14; i++) { const d = (hL(T, S) - h) / cp(T, S); T -= d; if (Math.abs(d) < 1e-11) break; } return T; };

/** Overall heat-transfer coefficients of El-Dessouky & Ettouney (W/m²·K, T in °C): falling-film evaporator and condenser. */
export const uEvaporator = (T) => 1e3 * (1.9695 + 1.2057e-2 * T - 8.5989e-5 * T * T + 2.5651e-7 * T ** 3);
export const uCondenser = (T) => 1e3 * (1.7194 + 3.2063e-3 * T + 1.5971e-5 * T * T - 1.9918e-7 * T ** 3);
const fouled = (U, o) => 1 / (1 / (U * o.fU) + o.Rf);

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
    B = m - V; X = salt / B; be = o.fBPE * bpe(T, X); Tv = T - be; hv = hV(Tv) + CPV * be; hb = hL(T, X);
    const Vn = clamp((q + F * hf + Bin * hbin - m * hb) / (hv - hb), -0.5 * m, Vmax);
    if (Math.abs(Vn - V) < 1e-12 * m) { V = Vn; break; }
    V = Vn;
  }
  B = m - V; X = salt / B; be = o.fBPE * bpe(T, X); Tv = T - be; hv = hV(Tv) + CPV * be; hb = hL(T, X);
  return { V, B, X, be, Tv, hv, hb };
}

/** March once through the train for given heating duty, temperature differences and feed temperatures. */
function medPass(c, s) {
  const N = c.N, o = c.o, r = { T: [], Tv: [], Tc: [], Th: [], V: [], fl: [], Vt: [], hm: [], X: [], B: [], q: [], be: [], dl: [], qloss: 0, Ts: s.Ts };
  let Tcp = s.Ts, Bin = 0, Xin = c.Xf, hbin = 0, Dacc = s.Mev, hD = hL(s.Ts), mprev = 0, hprev = 0;
  for (let i = 0; i < N; i++) {
    const T = Tcp - s.dT[i], qg = i === 0 ? s.Q1 : mprev * (hprev - hL(Tcp)), q = qg * (1 - c.loss);
    r.qloss += qg - q;
    const e = effect(q, c.Fi[i], hL(s.tf[i], c.Xf), Bin, Xin, hbin, T, c.Xf, o, s.Vg?.[i]), dl = vapourLoss(e.Tv, o), Tc = e.Tv - dl;
    let f = 0;
    if (i > 0) { // distillate flash box: condensate of the previous effect's vapour joins the accumulated distillate
      const C = r.Vt[i - 1] - (i - 1 === c.nEnt ? s.Mev : 0), m = Dacc + C, H = Dacc * hD + C * hL(Tcp);
      f = Math.max(0, (H - m * hL(e.Tv)) / (hV(e.Tv) - hL(e.Tv)));
      Dacc = m - f; hD = Dacc > 0 ? (H - f * hV(e.Tv)) / Dacc : hL(e.Tv);
    }
    const Vt = e.V + f, hm = (e.V * e.hv + f * hV(e.Tv)) / Vt;
    r.T.push(T); r.Th.push(Tcp); r.Tv.push(e.Tv); r.Tc.push(Tc); r.V.push(e.V); r.fl.push(f); r.Vt.push(Vt); r.hm.push(hm); r.X.push(e.X); r.B.push(e.B); r.q.push(q); r.be.push(e.be); r.dl.push(dl);
    mprev = Vt - (i === c.nEnt ? s.Mev : 0) - (i < N - 1 ? s.qph[i] / (hm - hL(Tc)) : 0); hprev = hm;
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
  const N = c.N, mvc = c.proc === 'mvc', tvc = c.proc === 'tvc', forward = c.arr === 'forward' && !mvc, F = c.F;
  const cc = { ...c, Fi: forward ? [F, ...fill(N - 1, 0)] : fill(N, F / N), nEnt: tvc ? clamp(c.nEnt, 0, N - 1) : -1 }, tol = c.tol || 1e-10, maxIt = c.maxIt || 250;
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
  let slope = 0, lastPass = null, errOuter = ws ? 1e-3 : 1;
  for (; it < maxIt; it++) {
    // inner secant: distillate target at frozen temperatures (production is almost linear in x)
    let x0 = x, r0 = pass(x0), x1;
    r = r0;
    const tolIn = clamp(1e-3 * errOuter, 1e-12, 1e-6) * c.MdT; // loose while the outer loop is far from converged
    for (let k = 0; k < 8 && Math.abs(r.Md - c.MdT) > tolIn; k++) {
      x1 = slope > 0 ? x0 + (c.MdT - r0.Md) / slope : mvc ? x0 + 0.01 * c.MdT * latentHeat(Ts) : x0 * clamp(c.MdT / r0.Md || 1.5, 0.3, 3);
      if (!mvc && !(x1 > 0)) x1 = 0.5 * x0;
      if (x1 === x0) break;
      const r1 = pass(x1), sl = (r1.Md - r0.Md) / (x1 - x0);
      if (Number.isFinite(sl) && sl > 0) slope = sl;
      x0 = x1; r0 = r1; r = r1;
    }
    let err = Math.abs(r.Md - c.MdT) / c.MdT + Math.abs(x0 - x) / (Math.abs(x0) + (mvc ? c.MdT * 1e5 : 1e-30));
    x = x0;
    if (!forward && N > 1) { // distribute the feed in proportion to the vapour boiled so every effect concentrates alike
      const Vs = sum(r.V.map((q) => Math.max(q, 0))), Fn = r.V.map((q, i) => 0.3 * cc.Fi[i] + 0.7 * F * Math.max(Math.max(q, 0) / Vs, 0.2 / N)), Fs = sum(Fn);
      err = Math.max(err, ...Fn.map((q, i) => Math.abs((q * F) / Fs - cc.Fi[i]) / F));
      cc.Fi = Fn.map((q) => (q * F) / Fs);
    }
    if (c.areaMode === 'equalArea' && N > 1) {
      const Am = sum(r.A) / N, S = sum(dT);
      dT = dT.map((d, i) => d * clamp(r.A[i] / Am, 0.3, 3) ** 0.7);
      const S2 = sum(dT); dT = dT.map((d) => (d * S) / S2);
      err = Math.max(err, ...r.A.map((a) => Math.abs(a / Am - 1)));
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
  }
  if (c.warm) { if (conv) Object.assign(c.warm, { key: wKey, dT: [...dT], tf: [...tf], qph: [...qph], Fi: [...cc.Fi], F, x, MdT: c.MdT, fz: { ...fz } }); else c.warm.key = ''; }
  if (!conv && !r.T.every((T) => T > c.Tsw - 5 && T < 200)) throw new Error('The effect train could not be solved for these inputs — the temperature window is too narrow for this number of effects' + (tvc ? ' or the ejector suction point leaves too little vapour for the downstream effects.' : '.'));
  if (!mvc && r.Tc[N - 1] - c.ttdCond <= c.Tsw + 0.3) throw new Error(`The last-effect vapour condenses at ${fmt(r.Tc[N - 1], 3)} °C, too close to the ${fmt(c.Tsw, 3)} °C seawater for the final condenser. Raise the last-effect temperature or lower the condenser approach.`);
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
        Bn = B - D; Xn = (B * X) / Bn; be = o.fBPE * bpe(T[i], Xn); hbn = hL(T[i], Xn);
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
  const Mf = br ? Mf0 : Mr, Mb = Mf - st.Md, Xb = st.X[N - 1];
  // tube side, cold end first
  const t = fill(N + 2, c.Tsw);
  let Mcw = 0, Mrej = Mf, tout = c.Tsw, Tr = c.Tsw;
  if (br) {
    const Qrej = sum(st.Qc.slice(nRec)), h0 = hL(c.Tsw, c.Xf);
    tout = st.Tv[nRec] - c.ttdRej;
    if (tout <= c.Tsw + 0.2) throw new Error(`Heat-rejection stages run at ${fmt(st.Tv[nRec], 3)} °C, too close to the ${fmt(c.Tsw, 3)} °C seawater. Raise the last-stage temperature or lower the rejection approach.`);
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
  if (bad) throw new Error(`Stage ${bad.i}: the tube-side brine (${fmt(bad.tout, 4)} °C) is not colder than the condensing vapour (${fmt(bad.Tv, 4)} °C). Add stages, widen the flashing range or raise the last-stage temperature.`);
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
  return { fBPE: p.fBPE, fNEA: p.fNEA, neaH: p.neaH, neaVb: p.neaVb, neaL: p.neaL, dpDem: demisterDrop(p.rhoP, p.Vdem, p.dw, p.thick / 1000), linePct: p.linePct, fU: p.fU, Rf: p.Rf / 1000 };
}

/** Run the selected process once. Returns one unified result object (SI units, flows in kg/s). */
export function simulateThermal(v, ov = {}) {
  const p = { ...v, ...ov }, o = options(p), proc = p.proc, T0 = p.Tsw, Xf = p.Xf, MdT = (p.Md * RHO_REF) / 86400, nSeg = Math.max(1, Math.round(p.nSeg));
  const loss = clamp(p.heatLoss / 100, 0, 0.5), vent = clamp(p.ventPct / 100, 0, 0.2), eP = p.etaPump / 100, rhoS = density(T0, Xf);
  const R = { proc, p, o, T0, Xf, MdT, hx: [], ex: [], rows: [], warnings: [] };
  let pumps = {};
  if (proc === 'msf') {
    const N = Math.max(2, Math.round(p.Nst)), Tsteam = p.TBT + p.dTsteam;
    const m = solveMSF({ N, nRej: Math.round(p.nRej), type: p.msfType, TBT: p.TBT, Tn: p.TnMsf, Tsw: T0, Xf, Xb: Math.max(p.Xb, Xf * 1.25), MdT, ttdRej: p.ttdRej, Tsteam, loss, o, nSeg });
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
    const mvc = proc === 'mvc', tvc = proc === 'medtvc', N = Math.max(1, Math.round(mvc ? p.Nmvc : p.N)), Xb = Math.max(p.Xb, Xf * 1.25), F = (MdT * Xb) / (Xb - Xf);
    const c = { proc: mvc ? 'mvc' : tvc ? 'tvc' : 'med', N, arr: p.arr, F, Xf, Tsw: T0, MdT, Ts: p.Ts, Tn: Math.min(p.Tn, p.Ts - 0.8 * N - 1), T1: p.Tmvc, dTm: p.dTmvc, preheat: !!p.preheat, ttdPh: p.ttdPh, ttdCond: p.ttdCond, loss, vent, o,
      areaMode: ov.areaMode || (p.areaMode === 'equalArea' && !ov.fast ? 'equalArea' : 'equalDT'), Ades: ov.Ades, dT0: ov.dT0, tol: ov.fast ? 3e-7 : 1e-10, maxIt: ov.fast ? 60 : 250, warm: ov.warm, Pm: p.Pm * 1e5, nEnt: (Math.round(p.nEnt) > 0 ? Math.round(p.nEnt) : N) - 1, etaIs: p.etaIs / 100 };
    const r = solveMED(c), L = N - 1, Md = r.Md, Mb = r.B[L], Tprod = tFromH(mvc ? (r.Dacc * r.hD + r.cond * hL(r.Ts)) / Md : (r.Dacc * r.hD + r.cond * hL(r.Tc[L])) / Md, 0);
    Object.assign(R, { r, N, Md, Mf: F, Mb, Xb: r.X[L], Mcw: 0, Ttop: r.T[0], Tlast: r.T[L], Tsteam: r.Ts, Tprod, Tbrine: r.T[L], Tcw: T0, Qcond: 0, converged: r.converged, areaEvap: sum(r.A), areaAux: 0 });
    const effEx = sum(r.q.map((q, i) => Iq(q, r.Th[i], r.T[i], T0))), lossEx = sum(r.Vt.map((m, i) => Iq(m * latentHeat(r.Tv[i]), r.T[i], r.Tc[i], T0)));
    if (mvc) {
      const cm = r.comp, Wsh = r.cond * cm.w, hProd = (r.Dacc * r.hD + r.cond * hL(r.Ts)) / Md, hB = hL(r.T[L], r.X[L]), dHf = F * (hL(r.TfM, Xf) - hL(T0, Xf));
      // feed preheater: product and brine cooled to a common outlet temperature To against the incoming feed
      const g = (To) => Md * (hProd - hL(To, 0)) + Mb * (hB - hL(To, r.X[L])) - dHf;
      let To = T0 + Math.min(p.ttdCond, 0.5 * (r.T[L] - T0)), Qfeed = 0;
      if (g(To) >= 0) To = brent(g, To, r.T[L], 1e-10); else Qfeed = -g(To);
      const Qaux = r.Qaux + Qfeed;
      const qD = Md * (hProd - hL(To, 0)), qB = Mb * (hB - hL(To, r.X[L])), U = p.Uph * 1000 * o.fU, aHot = Math.max(0.05, (Md * Tprod + Mb * r.T[L]) / (Md + Mb) - r.TfM); // both feed branches leave with the same hot-end approach
      const cex = (name, q, Th) => {
        const l = lmtd(aHot, To - T0), A = q / (U * l), rise = Th - aHot - T0, Cmin = q / Math.max(Th - To, rise, 1e-9);
        return { name, duty: q, area: A, lmtd: l, U, ntu: (U * A) / Cmin, eff: Math.max(Th - To, rise) / (Th - T0), tin: T0, tout: Th - aHot, Th };
      };
      R.hx = [cex('Feed preheater · distillate side', qD, Tprod), cex('Feed preheater · brine side', qB, r.T[L])];
      Object.assign(R, { Ms: 0, Qin: Qaux, Wcomp: Wsh / ((p.etaMotor / 100) * 0.98), Wshaft: Wsh, To, Qaux, Tprod: To, Tbrine: To, comp: cm, areaAux: sum(R.hx.map((x) => x.area)) });
      R.exIn = Wsh + Qaux * (1 - K(T0) / K(r.Ts));
      R.ex = [['Evaporator tubes (ΔT)', effEx], ['BPE, demister and line losses', lossEx], ['Compressor', K(T0) * r.cond * (CPV * Math.log(K(cm.T2) / K(cm.T1)) - RW * Math.log(cm.ratio))],
        ['Feed preheater', Iq(qD, 0.5 * (Tprod + To), 0.5 * (T0 + Tprod - aHot), T0) + Iq(qB, 0.5 * (r.T[L] + To), 0.5 * (T0 + r.T[L] - aHot), T0)], ['Brine and distillate discharge', exL(Mb, To, T0) + exL(Md, To, T0)]];
      R.balance = { massIn: F, massOut: Md + Mb + r.vent, saltIn: F * Xf, saltOut: Mb * r.X[L], eIn: Wsh + Qaux + F * hL(T0, Xf), eOut: Md * hL(To, 0) + Mb * hL(To, r.X[L]) + r.vent * r.hm[L] + r.qloss };
      pumps = { 'Feed supply': (F * p.dpSea * 1e5) / (rhoS * eP), 'Distillate': (Md * p.dpProd * 1e5) / (RHO_REF * eP), 'Brine blowdown': (Mb * p.dpBrine * 1e5) / (density(To, R.Xb) * eP) };
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
  let f, Tdes, lam;
  if (R.proc === 'msf') {
    const m = R.m, cpb = cp(70, m.Xr), C = m.stages.map((s) => s.A * cth + s.B * p.holdup * cpb), eps = m.stages.map((s) => (s.tout - s.tin) / (s.Tv - s.tin)), lossT = m.stages.map((s) => s.T - s.Tv);
    Tdes = m.stages.map((s) => s.T);
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
    const C = r.A.map((a, i) => a * cth + r.B[i] * p.holdup * cpb), qOut = r.q.map((_, i) => (i < N - 1 ? r.q[i + 1] : r.q[i]));
    const sink = r.q.map((q, i) => (q - (i < N - 1 ? qOut[i] : 0)) / (r.T[i] - T0)), UAc = (R.proc === 'mvc' ? 0 : r.q[N - 1]) / Math.max(0.5, r.Tc[N - 1] - T0);
    Tdes = r.T;
    lam = Math.max(...C.map((ci, i) => (UA[i] + (UA[i + 1] || UAc) + Math.abs(sink[i])) / ci)) * 2;
    f = (t, T) => {
      const qin = T.map((Ti, i) => (i === 0 ? (shutdown ? 0 : Math.min(r.q[0] * Math.min(1, t / ramp), UA[0] * Math.max(0, r.Ts - Ti) * 3)) : UA[i] * Math.max(0, T[i - 1] - dl[i - 1] - Ti)));
      return T.map((Ti, i) => (qin[i] - (i < N - 1 ? qin[i + 1] : UAc * Math.max(0, Ti - dl[i] - T0)) - (i < N - 1 || R.proc === 'mvc' ? sink[i] * (Ti - T0) : sink[i] * 0)) / C[i]);
    };
  }
  const sub = Math.max(1, Math.ceil(((tEnd / nt) * lam) / 2.2)), sol = rk4(f, shutdown ? [...Tdes] : fill(N, T0), 0, tEnd, nt * sub);
  const ts = [], Ts = [], prod = [];
  for (let k = 0; k <= nt; k++) {
    const y = sol.y[k * sub];
    ts.push(sol.t[k * sub] / 60); Ts.push(y);
    prod.push(100 * clamp(sum(y.map((T, i) => (T - T0) / (Tdes[i] - T0))) / N, 0, 1.5));
  }
  const lim = shutdown ? 50 : 95, i95 = prod.findIndex((x) => (shutdown ? x <= lim : x >= lim)); // 95 % approach on start-up, half of the temperature rise lost on cool-down
  const t95 = i95 > 0 ? ts[i95 - 1] + ((ts[i95] - ts[i95 - 1]) * (lim - prod[i95 - 1])) / (prod[i95] - prod[i95 - 1]) : i95 === 0 ? 0 : null;
  return { t: ts, T: Ts, prod, t95, sub, Tdes, final: prod[prod.length - 1] };
}

const calWarm = new Map(); // converged trains of the calibration model, one per operating point
const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const isMed = (v) => v.proc === 'med' || v.proc === 'medtvc', PROC = { med: 'MED', medtvc: 'MED-TVC', msf: 'MSF', mvc: 'MVC' };
const tryRun = (v, ov) => { try { const q = simulateThermal(v, ov); return q.converged ? q : null; } catch { return null; } };
const nz = (x) => (Number.isFinite(x) ? x : null);

const suite = {
  id: 'thermal', num: 6, title: 'Thermal Desalination', short: 'Thermal (MED/MSF)', icon: '♨️',
  tagline: 'Effect-by-effect and stage-by-stage design of MED, MED-TVC, MSF and MVC plants with heat-transfer sizing, energy and exergy accounting.',
  description: 'Solves the steady-state total-mass, salt and enthalpy balances of every effect or flashing stage, including boiling-point elevation, non-equilibrium allowance, demister and vapour-line losses, flashing of brine and distillate, feed preheaters and the final condenser. Heat-transfer areas follow from published U(T) correlations with fouling (LMTD and effectiveness–NTU); thermo-compressors use the Power-chart entrainment correlation and mechanical compressors an isentropic-efficiency model. Gain-output ratio, performance ratio, specific energy, cooling water, exergy destruction and a lumped start-up transient are reported together with part-load and seawater-temperature sweeps.',
  guide: [
    'Choose the process (MED, MED-TVC, MSF or MVC), the distillate capacity and the seawater condition — or pull the case feed, the site sea temperature or an RO concentrate for a hybrid.',
    'Set the temperature window (heating steam or top brine temperature and the last effect/stage) and the number of effects or stages.',
    'On Model setup choose equal-ΔT or equal-area design, fouling, temperature losses and scale limits.',
    'Run. Check the per-effect table, the scale warnings and the balances on the Verify tab; the distillate, brine and energy figures are offered to the plant, ZLD and economics suites.',
  ],
  implemented: ['total and component mass', 'energy balance', 'steam-table', 'vapour-liquid equilibrium', 'boiling-point-elevation', 'antoine', 'clausius-clapeyron', 'flash-vaporization', 'heat-exchanger equation', 'overall heat-transfer', 'logarithmic-mean', 'effectiveness-ntu', 'condensation equation', 'evaporation equation', 'latent-heat', 'non-equilibrium allowance', 'demister pressure-drop', 'compressor equation', 'exergy equation',
    'multi-effect mass-energy', 'multi-stage-flash stagewise', 'thermal-vapour-compression', 'mechanical-vapour-compression', 'med-tvc', 'solar-thermal desalination', 'waste-heat-desalination',
    'stage/effect temperature', 'pressures', 'salinities', 'liquid inventories', 'wall temperature', 'feed-flow', 'feed-temperature and feed-salinity', 'heating-steam', 'condenser cooling-water', 'terminal vacuum', 'vapour-liquid interfacial', 'product/brine outlet',
    'feed-water thermodynamics', 'evaporation', 'condensation', 'flashing', 'boiling-point elevation', 'heat transfer', 'heat-exchanger modelling', 'multi-stage and multi-effect', 'vapour compression', 'steam and utility', 'vacuum-system', 'brine recirculation', 'heat recovery', 'scaling assessment', 'condenser modelling', 'thermal-energy consumption', 'electrical-energy consumption', 'equipment sizing', 'transient operation', 'start-up and shutdown', 'waste-heat integration', 'solar-thermal integration', 'performance-ratio assessment'],
  equationsNote: 'Steady-state design model with saturated heating steam. Seawater properties follow Sharqawy et al.; U(T), demister, non-equilibrium and CaSO₄-envelope correlations follow El-Dessouky & Ettouney and are valid for roughly 30–120 °C; the ejector fit holds for compression ratios ≥ 1.81 and entrainment ratios ≤ 4. Part-load results of MED/MVC are true ratings at fixed area, the MSF turndown curve is a re-design at lower top brine temperature. Start-up and cool-down after a steam trip are lumped thermal-inertia estimates (no vacuum pull-down, venting or level control). An RO concentrate can be pulled as feed, but RO–thermal hybrids are not optimised here. Scaling is screened with a CaSO₄ solubility envelope and top-temperature limits only — use suite 2 for speciation. Non-condensable gases enter only as a venting allowance; tube-bundle geometry and wetting rates are not resolved.',

  inputs: [
    { group: 'Process and capacity', help: 'Which thermal process is designed and for how much distillate.', fields: [
      { key: 'proc', label: 'Process', type: 'select', value: 'med', options: [{ value: 'med', label: 'MED — multi-effect distillation' }, { value: 'medtvc', label: 'MED-TVC — with steam-jet thermo-compressor' }, { value: 'msf', label: 'MSF — multi-stage flash' }, { value: 'mvc', label: 'MVC — mechanical vapour compression' }], help: 'Each process shows its own inputs. The examples load realistic settings for every process.' },
      { key: 'Md', label: 'Distillate capacity', unit: 'm³/d', value: 20000, min: 10, max: 1e6, typical: [500, 90000], help: 'Net distillate production the plant is designed for.' },
    ] },
    { group: 'Seawater feed', help: 'Condition of the water entering the plant (seawater or, for a hybrid, RO concentrate).', fields: [
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.seawater.ions, help: 'Sets the ion ratios of the brine and distillate streams passed to other suites; the salinity below sets the total.' },
      { key: 'Xf', label: 'Feed salinity', unit: 'g/kg', value: 36, min: 1, max: 120, typical: [30, 70], help: 'Total dissolved salts per kg of feed.' },
      { key: 'Tsw', label: 'Seawater temperature', unit: '°C', value: 26, min: 2, max: 40, typical: [12, 35], help: 'Cooling-water and feed inlet temperature; also the dead state of the exergy analysis.' },
      { key: 'Xb', label: 'Maximum brine salinity', unit: 'g/kg', value: 60, min: 5, max: 250, typical: [50, 72], help: 'Blow-down salinity; with the feed salinity it fixes the recovery.', showIf: (v) => !(v.proc === 'msf' && v.msfType === 'ot') },
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
      { key: 'nRej', label: 'Heat-rejection stages', unit: '', value: 3, min: 1, max: 6, step: 1, showIf: (v) => v.msfType === 'br', help: 'Stages cooled by seawater instead of recirculating brine.' },
      { key: 'TBT', label: 'Top brine temperature', unit: '°C', value: 110, min: 60, max: 125, typical: [90, 112], help: 'Brine temperature leaving the brine heater; limited by the scale-control method.' },
      { key: 'TnMsf', label: 'Last-stage brine temperature', unit: '°C', value: 40, min: 25, max: 60, typical: [35, 42], help: 'Brine temperature in the last, coldest stage.' },
      { key: 'dTsteam', label: 'Heating steam above top brine temperature', unit: 'K', value: 7, min: 2, max: 30, help: 'Condensing-steam temperature in the brine heater minus the top brine temperature.' },
    ] },
    { group: 'Mechanical vapour compression', showIf: (v) => v.proc === 'mvc', fields: [
      { key: 'Nmvc', label: 'Number of effects', unit: '', value: 1, min: 1, max: 6, step: 1, help: 'Effects in series between the compressor discharge and suction.' },
      { key: 'Tmvc', label: 'First-effect brine temperature', unit: '°C', value: 60, min: 40, max: 100, typical: [50, 70], help: 'Boiling temperature of the brine in the first effect.' },
      { key: 'dTmvc', label: 'Condensing − boiling temperature difference', unit: 'K', value: 2.5, min: 0.8, max: 8, typical: [1.5, 4], help: 'Smaller differences save compressor power but need more evaporator area.' },
      { key: 'etaIs', label: 'Compressor isentropic efficiency', unit: '%', value: 76, min: 40, max: 92, help: 'Ratio of isentropic to actual compression work.' },
      { key: 'etaMotor', label: 'Motor + drive efficiency', unit: '%', value: 94, min: 60, max: 99, help: 'Electric motor, drive and gearbox combined.' },
      { key: 'Uph', label: 'Feed-preheater overall U', unit: 'kW/m²·K', value: 1.8, min: 0.3, max: 6, help: 'Plate exchangers recovering heat from product and brine.' },
    ] },
    { group: 'Heat source', help: 'Where the heat comes from and how it is valued.', fields: [
      { key: 'source', label: 'Heat source', type: 'select', value: 'steam', options: [{ value: 'steam', label: 'Boiler or turbine-extraction steam' }, { value: 'solar', label: 'Solar-thermal collector field' }, { value: 'waste', label: 'Industrial waste-heat stream' }], help: 'Adds a sizing table for a solar field or a waste-heat match to the results.' },
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
      { key: 'areaMode', label: 'Effect sizing rule', type: 'select', value: 'equalArea', options: [{ value: 'equalArea', label: 'Equal area in every effect (iterated)' }, { value: 'equalDT', label: 'Equal temperature difference in every effect' }], showIf: (v) => v.proc !== 'msf', help: 'Equal-area designs are the industrial standard; the temperature differences are iterated until all areas match.' },
      { key: 'fU', label: 'Heat-transfer coefficient multiplier', unit: '×', value: 1, min: 0.3, max: 2, help: 'Scales the published U(T) correlations. Calibrate against plant data.' },
      { key: 'Rf', label: 'Additional fouling resistance', unit: 'm²·K/kW', value: 0.03, min: 0, max: 0.5, help: 'Added in series to the correlation value: 1/U = 1/U₀ + R_f.' },
      { key: 'heatLoss', label: 'Heat loss to surroundings', unit: '% of duty', value: 1, min: 0, max: 15, help: 'Share of every effect or brine-heater duty lost through the insulation.' },
      { key: 'ttdCond', label: 'Final-condenser / feed-heater approach', unit: 'K', value: 3, min: 0.5, max: 15, showIf: (v) => v.proc !== 'msf', help: 'Condensing temperature minus seawater outlet temperature. For MVC: minimum approach of the feed preheater.' },
      { key: 'ttdPh', label: 'Feed-preheater terminal difference', unit: 'K', value: 3, min: 0.5, max: 15, showIf: (v) => isMed(v) && v.preheat, help: 'Condensing vapour temperature minus feed outlet temperature of each preheater.' },
      { key: 'ttdRej', label: 'Heat-rejection terminal difference', unit: 'K', value: 3, min: 0.5, max: 15, showIf: (v) => v.proc === 'msf' && v.msfType === 'br', help: 'Condensing temperature of the first rejection stage minus the seawater leaving it.' },
    ] },
    { group: 'Temperature losses and venting', tab: 'setup', help: 'Thermodynamic penalties between boiling brine and condensing vapour.', fields: [
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
      { key: 'ventPct', label: 'Vapour lost with vented gases', unit: '% of condensed vapour', value: 0.5, min: 0, max: 5, help: 'Vapour leaving the cold end with the non-condensable gases through the vacuum system.' },
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
      { key: 'holdup', label: 'Brine hold-up time per effect/stage', unit: 's', value: 90, min: 0, max: 900, showIf: (v) => v.transient, help: 'Liquid inventory expressed as residence time of the brine flow.' },
      { key: 'ramp', label: 'Steam ramp-up time', unit: 'min', value: 30, min: 1, max: 600, showIf: (v) => v.transient, help: 'Time over which the heating steam is brought to its design flow.' },
      { key: 'tEnd', label: 'Simulated time', unit: 'h', value: 4, min: 0.2, max: 48, showIf: (v) => v.transient, help: 'Length of the simulated start-up; it is extended automatically if the plant is not yet warm.' },
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
  ],

  run(v) {
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
    // 2-D sensitivity map
    let field;
    if (msf) {
      const xs = [12, 16, 20, 24, 28, 32].filter((n) => n > p.nRej + 1), ys = linspace(88, 118, 6);
      field = { type: 'field', title: 'Performance ratio versus stage count and top brine temperature', xlabel: 'Number of stages', ylabel: 'Top brine temperature (°C)', zlabel: 'Performance ratio', zunit: 'kg/2326 kJ', x: xs, y: ys, z: ys.map((T) => xs.map((n) => tryRun(base, { Nst: n, TBT: T, fast: true })?.PR ?? null)), cmap: 'viridis', contours: 8, markers: [{ x: N, y: p.TBT, label: 'design' }] };
    } else if (mvc) {
      const xs = linspace(45, 70, 6), ys = linspace(1.5, 5, 5);
      field = { type: 'field', title: 'Specific electricity versus evaporation temperature and ΔT', xlabel: 'First-effect brine temperature (°C)', ylabel: 'Condensing − boiling ΔT (K)', zlabel: 'Specific electricity', zunit: 'kWh/m³', x: xs, y: ys, z: ys.map((d) => xs.map((T) => tryRun(base, { Tmvc: T, dTmvc: d, fast: true })?.sEl ?? null)), cmap: 'thermal', contours: 8, markers: [{ x: p.Tmvc, y: p.dTmvc, label: 'design' }] };
    } else {
      const xs = [4, 6, 9, 12], ys = linspace(58, 74, 4);
      field = { type: 'field', title: 'Gain-output ratio versus effects and heating-steam temperature', xlabel: 'Number of effects', ylabel: 'Heating-steam temperature (°C)', zlabel: 'GOR', zunit: 'kg/kg', x: xs, y: ys, z: ys.map((T) => xs.map((n) => nz(tryRun(base, { N: n, Ts: T, fast: true, nEnt: p.nEnt > 0 ? Math.max(1, Math.round((p.nEnt * n) / N)) : 0 })?.GOR))), cmap: 'viridis', contours: 8, markers: [{ x: N, y: p.Ts, label: 'design' }] };
    }
    // repair isolated gaps of the map so that colour scaling stays finite
    const zf = field.z.flat().filter((x) => x !== null), zMean = zf.length ? sum(zf) / zf.length : 0;
    field.z = field.z.map((row) => row.map((x) => (x === null ? zMean : x)));
    field.note = 'Each cell is a full re-design with equal temperature differences; cells that cannot be designed are filled with the map average.';

    let su = null, tSim = p.tEnd;
    if (p.transient) for (let k = 0; k < 4; k++) { su = startUp(R, { ...p, tEnd: tSim }); if (su.t95 !== null) break; if (k < 3) tSim *= 2; }
    if (su && su.t95 === null) W.push({ level: 'info', msg: `Start-up does not reach 95 % of the design temperature rise within ${fmt(tSim, 3)} h.` });
    const sd = su ? startUp(R, { ...p, tEnd: tSim }, true) : null;
    if (su && su.t95 !== null && tSim > p.tEnd) W.push({ level: 'info', msg: `The start-up simulation was extended to ${fmt(tSim, 3)} h so that the plant reaches its design temperatures.` });

    // ---- outputs
    const ionsF = scaleIons(cloneIons(p.ions), tdsFromSalinity(p.Xf, 25) / Math.max(tds(p.ions), 1e-9)), cf = R.Xb / p.Xf, Xmean = msf ? sum(m.stages.map((s) => s.X)) / N : sum(r.X) / N;
    const tdsD = (p.carry * 1e-6 * Xmean * 1000 * RHO_REF) / 1000, Qb = (R.Mb * 3600) / density(R.Tbrine, R.Xb);
    const round = (o) => Object.fromEntries(ION_IDS.map((k) => [k, +o[k].toPrecision(6)]));
    const out = { distillate: R.Qd, GOR: nz(R.GOR), PR: nz(R.PR), secThermal: R.sTh, secElec: R.sEl, steam: R.Ms, area: R.area, recovery: R.rec, coolingWater: (R.Mcw * 3600) / density(p.Tsw, p.Xf), heat: R.Qin / 1000, power: R.Wel / 1000, process: proc, topBrineTemperature: R.Ttop, brineSalinity: R.Xb, secEquivalent: R.sEq, startUpMin: su?.t95 ?? null,
      streams: { distillate: { Q: R.Qd, T: R.Tprod, P: 1, pH: 6.5, tds: tdsD, ions: round(scaleIons(ionsF, tdsD / Math.max(tds(ionsF), 1e-9))) }, brine: { Q: Qb, T: R.Tbrine, P: 1, pH: Math.min(9, 8.1 + 0.3 * Math.log10(cf)), tds: tdsFromSalinity(R.Xb, R.Tbrine), ions: round(scaleIons(ionsF, tdsFromSalinity(R.Xb, R.Tbrine) / Math.max(tds(ionsF), 1e-9))) } } };

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
    plots.push(field);
    if (su) {
      const idx = [...new Set([0, Math.floor((N - 1) / 2), N - 1])];
      plots.push({ type: 'line', title: 'Start-up from cold and cool-down after a steam trip', xlabel: 'Time (min)', ylabel: '°C · %', series: [...idx.map((i) => ({ name: `${unit} ${i + 1} brine temperature (°C)`, x: su.t, y: su.T.map((y) => y[i]) })), { name: 'Start-up: approach to design (%)', x: su.t, y: su.prod, dash: true }, { name: 'Shutdown after a steam trip: remaining temperature rise (%)', x: sd.t, y: sd.prod, dash: true }], note: `Lumped thermal inertia, linear steam ramp over ${p.ramp} min; after a steam trip feed and cooling water keep flowing; ${su.sub} RK4 sub-step${su.sub > 1 ? 's' : ''} per output step.` });
    }
    // any failed sweep point is dropped from its series
    for (const pl of plots) if (pl.type === 'line') {
      for (const s of pl.series) { const keep = s.y.map((y, i) => y !== null && Number.isFinite(y) && Number.isFinite(s.x[i])); s.x = s.x.filter((_, i) => keep[i]); s.y = s.y.filter((_, i) => keep[i]); }
      pl.series = pl.series.filter((s) => s.x.length > 0);
    }
    const shown = plots.filter((pl) => pl.type !== 'line' || pl.series.length > 0);

    const tables = [];
    if (msf) tables.push({ title: 'Stage-by-stage results', columns: ['Stage', 'Section', 'Brine T (°C)', 'Vapour T (°C)', 'Pressure (kPa)', 'Brine out (kg/s)', 'Salinity (g/kg)', 'Distillate formed (kg/s)', 'BPE (K)', 'NEA (K)', 'Demister+line (K)', 'Tube in (°C)', 'Tube out (°C)', 'TTD (K)', 'U (W/m²·K)', 'Area (m²)', 'Duty (MW)', 'NTU', 'Effectiveness'],
      rows: m.stages.map((s) => [s.i, s.rec ? 'recovery' : 'rejection', s.T, s.Tv, s.P / 1000, s.B, s.X, s.D, s.be, s.nea, s.dl, s.tin, s.tout, s.ttd, s.U, s.A, s.Q / 1e6, s.ntu, s.eff]) });
    else tables.push({ title: 'Effect-by-effect results', columns: ['Effect', 'Heating T (°C)', 'Brine T (°C)', 'Vapour T (°C)', 'Pressure (kPa)', 'Feed in (kg/s)', 'Feed T (°C)', 'Brine out (kg/s)', 'Salinity (g/kg)', 'Vapour boiled (kg/s)', 'Flash vapour (kg/s)', 'BPE (K)', 'Demister+line (K)', 'ΔT (K)', 'U (W/m²·K)', 'Area (m²)', 'Duty (MW)', 'CaSO₄ envelope (g/kg)'],
      rows: xs.map((n, i) => [n, r.Th[i], r.T[i], r.Tv[i], psat(r.Tv[i]) / 1000, r.c.Fi[i], r.c.Fi[i] > 0 ? r.tf[i] : null, r.B[i], r.X[i], r.V[i], r.fl[i], r.be[i], r.dl[i], r.dT[i], r.U[i], r.A[i], r.q[i] / 1e6, caso4Limit(r.T[i])]),
      note: p.areaMode === 'equalArea' && N > 1 ? 'Temperature differences iterated until all effects have the same area.' : '' });
    tables.push({ title: 'Heat exchangers (LMTD and effectiveness–NTU)', columns: ['Exchanger', 'Duty (MW)', 'Hot side (°C)', 'Cold in (°C)', 'Cold out (°C)', 'LMTD (K)', 'U (W/m²·K)', 'Area (m²)', 'NTU', 'Effectiveness'], rows: R.hx.map((x) => [x.name, x.duty / 1e6, x.Th, x.tin, x.tout, nz(x.lmtd), x.U, x.area, x.ntu, x.eff]), note: 'For condensing duties ε = 1 − exp(−NTU); the NTU and LMTD routes give the same area.' });
    tables.push({ title: 'Streams and utilities', columns: ['Stream', 'Flow (kg/s)', 'Flow (t/h)', 'Temperature (°C)', 'Salinity (g/kg)'],
      rows: [['Feed (make-up)', R.Mf, R.Mf * kgh, p.Tsw, p.Xf], ['Distillate', R.Md, R.Md * kgh, R.Tprod, tdsD / 1000], ['Brine blow-down', R.Mb, R.Mb * kgh, R.Tbrine, R.Xb], ['Cooling water rejected', R.Mcw, R.Mcw * kgh, R.Tcw, p.Xf],
        ...(msf ? [['Recirculating brine', m.Mr, m.Mr * kgh, m.Tr, m.Xr]] : []), ...(R.ej ? [['Motive steam', R.ej.Mm, R.ej.Mm * kgh, R.ej.Tm, 0], ['Entrained vapour', R.ej.Mev, R.ej.Mev * kgh, R.ej.Tev, 0]] : []),
        ...(mvc ? [['Compressed vapour', r.cond, r.cond * kgh, R.comp.T2, 0]] : [[proc === 'medtvc' ? 'Heating vapour to effect 1' : 'Heating steam', proc === 'medtvc' ? R.ej.Mm + R.ej.Mev : R.Ms, (proc === 'medtvc' ? R.ej.Mm + R.ej.Mev : R.Ms) * kgh, R.Tsteam, 0]]), ['Vented vapour', R.vent, R.vent * kgh, R.Tlast, 0]] });
    tables.push({ title: 'Energy and exergy accounting', columns: ['Item', 'kW', 'kWh per m³ distillate'],
      rows: [['Heat supplied', R.Qin / 1000, R.sTh], ...Object.entries(R.pumps).map(([k, w]) => [`Electricity · ${k}`, w / 1000, w / 1000 / R.Qd]), ['Electricity · total', R.Wel / 1000, R.sEl], ['Equivalent electricity (heat valued as lost turbine work)', R.sEq * R.Qd, R.sEq], ['Minimum work of separation', R.wMin * R.Qd, R.wMin],
        ['Exergy supplied (heat + electricity)', R.exTot / 1000, R.exTot / 1000 / R.Qd], ...R.ex.map(([k, e]) => [`Exergy destroyed · ${k}`, e / 1000, e / 1000 / R.Qd])] });
    if (srcRows.length) tables.push({ title: p.source === 'solar' ? 'Solar-thermal integration' : 'Waste-heat integration', columns: ['Quantity', 'Value'], rows: srcRows, note: srcNote });

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
        'Use suite 13 (Economics) to weigh the extra area of a higher GOR against the energy saved.',
      ].filter(Boolean),
      plots: shown, tables,
      balances: [
        { name: 'Water + salt mass (kg/s)', in: b.massIn, out: b.massOut }, { name: 'Salt (kg/s)', in: (b.saltIn) / 1000, out: (b.saltOut) / 1000 }, { name: 'Energy (MW)', in: b.eIn / 1e6, out: b.eOut / 1e6 },
        ...(msf ? [] : [{ name: 'Distillate = Σ vapour boiled − vent (kg/s)', in: sum(r.V) - r.vent, out: r.Md }]),
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
    return C;
  },
};

/** Synthetic plant data: the model with slightly different true parameters plus deterministic noise. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([Tsw, Ts, Md]) => {
    const q = suite.calibration.model({ ...d, fU: 0.86, heatLoss: 2.6, ttdCond: 4.2, Tsw, Ts, Md });
    return { Tsw, Ts, Md, sth: +(q.sth * (1 + g.normal(0, 0.006))).toFixed(2), sA: +(q.sA * (1 + g.normal(0, 0.012))).toFixed(4), Qcw: +(q.Qcw * (1 + g.normal(0, 0.015))).toFixed(0) };
  });
}

export default suite;
