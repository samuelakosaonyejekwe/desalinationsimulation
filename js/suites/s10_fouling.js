// Suite 10 — Fouling and membrane-performance monitoring.
// Operating logs are validated, normalised (ASTM D4516 style: net driving pressure, temperature-correction
// factor, flow-corrected differential pressure), converted to Darcy resistances, and analysed for trends,
// sudden changes and anomalies. Blocking laws, cake filtration, critical flux and Monod biofilm growth give
// the mechanistic interpretation; rule-based evidence scores name the likely foulant; robust trends give the
// days to the next cleaning and the remaining membrane life with uncertainty bands.
import { clamp, linspace, logspace, sum, mean, quantile, rng, fmt, rk4, interp1, levenbergMarquardt, isNum, linfit, tridiag, solveLinear } from '../core/num.js';
import { tcf, viscosity, conductivityFromTDS, diffusivityNaCl, KELVIN } from '../core/props.js';

const KB = 1.380649e-23, MU25 = viscosity(25);
const med = (a) => (a.length ? quantile(a, 0.5) : 0);
const mad = (a) => { const m = med(a); return 1.4826 * med(a.map((x) => Math.abs(x - m))); };
const tcfM = (T) => tcf(T, T >= 25 ? 2640 : 3020);
/** Osmotic pressure of a salt solution from its TDS (mg/L), bar — the ASTM D4516 approximation. */
export const piAstm = (C, T) => (0.002654 * C * (T + KELVIN)) / (1000 - C / 1000);
let _ec = null;
const ecTable = () => (_ec ||= (() => { const t = logspace(1, 3e5, 120); return { tds: t, ec: t.map(conductivityFromTDS) }; })());
/** TDS (mg/L) from electrical conductivity (µS/cm), inverse of the shared conductivity correlation. */
export const tdsFromEC = (ec) => (ec <= ecTable().ec[0] ? 0.5 * Math.max(ec, 0) : interp1(ecTable().ec, ecTable().tds, ec));
/** Flux relative to the clean value under Hermia's constant-pressure blocking laws; x = k·t. */
export const hermia = (law, x) => (law === 'complete' ? Math.exp(-x) : law === 'standard' ? 1 / (1 + x) ** 2 : law === 'cake' ? 1 / Math.sqrt(1 + x) : 1 / (1 + x));
export const HERMIA = { complete: { name: 'Complete blocking', n: 2 }, standard: { name: 'Standard (pore constriction)', n: 1.5 }, intermediate: { name: 'Intermediate blocking', n: 1 }, cake: { name: 'Cake filtration', n: 0 } };
/** Kozeny–Carman specific cake resistance, m/kg, for particle diameter dp (m), porosity eps and particle density rho. */
export const kozenyCarman = (dp, eps, rho) => (180 * (1 - eps)) / (rho * dp * dp * eps ** 3);

/** Theil–Sen slope with Sen's 95 % confidence interval (robust to outliers and non-normal noise). */
export function theilSen(x, y) {
  const n = Math.min(x.length, y.length);
  if (n < 3) return { slope: 0, intercept: n ? mean(y) : 0, lo: 0, hi: 0, n };
  const s = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (x[j] !== x[i]) s.push((y[j] - y[i]) / (x[j] - x[i]));
  if (!s.length) return { slope: 0, intercept: mean(y), lo: 0, hi: 0, n };
  s.sort((a, b) => a - b);
  const N = s.length, slope = quantile(s, 0.5), C = 1.96 * Math.sqrt((n * (n - 1) * (2 * n + 5)) / 18);
  const lo = s[clamp(Math.floor((N - C) / 2) - 1, 0, N - 1)], hi = s[clamp(Math.ceil((N + C) / 2), 0, N - 1)];
  return { slope, intercept: med(y.map((v, i) => v - slope * x[i])), lo, hi, n };
}
/** Hampel identifier: true where a point deviates from the centred running median by more than nsig robust sigmas. */
export function hampel(x, k = 3, nsig = 4.5) {
  return x.map((v, i) => { const w = x.slice(Math.max(0, i - k), Math.min(x.length, i + k + 1)), m = med(w), s = mad(w); return Math.abs(v - m) > nsig * Math.max(s, 1e-9 + 0.004 * Math.abs(m)); });
}
const stepScan = (x, w) => { const n = x.length, d = new Array(n).fill(0); for (let i = w; i <= n - w; i++) d[i] = med(x.slice(i, i + w)) - med(x.slice(i - w, i)); return d; };

// ---- mechanistic single-train model (used for the synthetic log, calibration and the pressure overlay) ---------
/** Feed pressure and differential pressure of a train at time tc (days since clean start). */
export function pressureModel(p, c) {
  const Y = clamp(c.Qp / c.Qf, 0.01, 0.98), Cfb = (tdsFromEC(c.Cf) * Math.log(1 / (1 - Y))) / Y, a = hermia(p.law, Math.max(0, p.kFoul) * Math.max(0, c.tc));
  const ndp = (c.Qp * 1000) / p.area / (p.Aclean * a * tcfM(c.T));
  const dP = p.dp0 * (1 + p.kDp * Math.max(0, c.tc)) * ((c.Qf - c.Qp / 2) / p.qRef) ** p.mFlow * (viscosity(c.T) / viscosity(25)) ** 0.3;
  const Pp = c.Pp ?? p.Pp, piP = piAstm(Cfb * (p.sp0 / 100) * tcfM(c.T), c.T);
  return { Pf: ndp + dP / 2 + Pp + piAstm(Cfb, c.T) - piP, dP, ndp };
}

/**
 * Deterministic synthetic operating log of a two-stage brackish RO train (150 m³/h permeate, 75 % recovery):
 * gradual colloidal/organic fouling, a tail-end scaling episode (days 96–119), a clean-in-place on day 120 and an
 * O-ring failure on day 150, with sensor noise and a few deliberate data-quality problems.
 */
export function syntheticLog({ days = 180, seed = 3, noise = 1, fouling = true, events = true, quality = true, step = null } = {}) {
  const g = rng(seed), rows = [], area = 8035, s1 = 0.65, z = () => noise * g.normal(0, 1), ec = conductivityFromTDS;
  let Rg = 0, Rs = 0, f1 = 0, f2g = 0, f2s = 0, spAge = 0, spScale = 0;
  for (let t = 0; t < days; t++) {
    if (events && t === 120) { Rg *= 0.15; Rs = 0; f1 *= 0.1; f2g *= 0.1; f2s = 0; spScale = 0; }
    const scaling = events && t >= 96 && t < 120, spStep = events && t >= 150 ? 1.45 : 1, extra = step && t >= step.day ? step.loss : 0;
    const T = 24 + 4 * Math.sin((2 * Math.PI * (t - 40)) / 365) + 0.25 * z(), tdsF = 3500 * (1 + 0.04 * Math.sin((2 * Math.PI * t) / 90)) * (1 + 0.006 * z());
    const Qp = 150 * (1 + 0.005 * z()), Y = 0.75 + 0.003 * z(), Qf = Qp / Y, a = (1 - extra) / (1 + Rg + Rs), mu = (viscosity(T) / viscosity(25)) ** 0.3;
    const ndp = (Qp * 1000) / area / (3.6 * a * tcfM(T));
    const dP1 = 1.6 * (1 + f1) * ((Qf - (s1 * Qp) / 2) / 151.25) ** 1.5 * mu, dP2 = 1.2 * (1 + f2g + f2s) * ((Qf - s1 * Qp - ((1 - s1) * Qp) / 2) / 76.25) ** 1.5 * mu;
    const Cfb = (tdsF * Math.log(1 / (1 - Y))) / Y, sp = 1.2 * (1 + spAge) * (1 + spScale) * spStep * tcfM(T) * (150 / Qp), tdsP = (sp / 100) * Cfb;
    const Pp = 1 + 0.015 * z(), Pf = ndp + (dP1 + dP2) / 2 + Pp + piAstm(Cfb, T) - piAstm(tdsP, T) + 0.03 * z(), Pi = Pf - dP1 + 0.012 * z(), Pc = Pf - dP1 - dP2 + 0.015 * z();
    const row = { t, Pf: +Pf.toFixed(2), Pi: +Pi.toFixed(2), Pc: +Pc.toFixed(2), Pp: +Pp.toFixed(2), Qf: +(Qf * (1 + 0.004 * z())).toFixed(1), Qp: +Qp.toFixed(1), Cf: +(ec(tdsF) * (1 + 0.006 * z())).toFixed(0), Cp: +(ec(tdsP) * (1 + 0.012 * z())).toFixed(1), T: +T.toFixed(1) };
    if (quality) {
      if (t >= 84 && t <= 86) { /* plant shutdown: no records */ } else {
        if (t === 37) row.Qp = null; // missing transmitter value
        if (t === 71) row.Cp = +(row.Cp * 3.2).toFixed(1); // conductivity spike
        if (t === 133) row.T = 0; // frozen temperature signal
        rows.push(row);
      }
    } else rows.push(row);
    if (fouling) { Rg += 0.0009; f1 += 0.001; f2g += 0.0004; spAge += 0.0002; if (scaling) { Rs += 0.0045; f2s += 0.012; spScale += 0.003; } }
  }
  return rows;
}
let _log = null;
const defaultLog = () => (_log ||= syntheticLog());

// ---- analysis ----------------------------------------------------------------------------------------------
const COLS = [
  { key: 't', label: 'Time', unit: 'd' }, { key: 'Pf', label: 'Feed pressure', unit: 'bar' }, { key: 'Pi', label: 'Interstage pressure', unit: 'bar' }, { key: 'Pc', label: 'Concentrate pressure', unit: 'bar' }, { key: 'Pp', label: 'Permeate pressure', unit: 'bar' },
  { key: 'Qf', label: 'Feed flow', unit: 'm³/h' }, { key: 'Qp', label: 'Permeate flow', unit: 'm³/h' }, { key: 'Cf', label: 'Feed conductivity', unit: 'µS/cm' }, { key: 'Cp', label: 'Permeate conductivity', unit: 'µS/cm' }, { key: 'T', label: 'Temperature', unit: '°C' },
];
const NEED = ['t', 'Pf', 'Pc', 'Pp', 'Qf', 'Qp', 'Cf', 'Cp', 'T'];
export const FOULANTS = {
  colloidal: { name: 'Colloidal / particulate fouling', cip: [['1', 'High-pH detergent', '0.1 % NaOH + 0.03 % SDS (or 1 % Na₄EDTA)', '11–12', '30–35', '60 min recirculation, 1–4 h soak'], ['2', 'Low-pH rinse', '2 % citric acid', '2.5–4', '30–35', '30–60 min recirculation']] },
  bio: { name: 'Biological fouling', cip: [['1', 'High-pH biofilm removal', '0.1 % NaOH + 0.03 % SDS', '12', '35', '60 min recirculation, overnight soak'], ['2', 'Non-oxidising biocide', 'DBNPA 20–30 mg/L or isothiazolinone', '6–8', 'ambient', '30–60 min'], ['3', 'Low-pH rinse', '0.2 % HCl or 2 % citric acid', '2–3', '30', '30 min']] },
  organic: { name: 'Organic fouling', cip: [['1', 'High-pH cleaning', '0.1 % NaOH + 1 % Na₄EDTA', '12', '30–35', '60 min recirculation, 2–6 h soak'], ['2', 'Low-pH rinse', '0.2 % HCl', '2–2.5', '30', '30 min']] },
  scaling: { name: 'Mineral scaling', cip: [['1', 'Low-pH scale dissolution', '2 % citric acid (carbonates) or 0.2 % HCl; 1 % Na₄EDTA + NaOH at pH 12 for sulphate scales', '2–3', '35–40', '60–120 min recirculation, start with the last stage'], ['2', 'High-pH polish', '0.1 % NaOH', '11–12', '30', '30 min']] },
  integrity: { name: 'Mechanical integrity (O-ring, glue line, interconnector)', cip: [['1', 'Do not clean', 'Probe each pressure vessel for permeate conductivity; shim and replace O-rings or the leaking element', '–', '–', '–']] },
  oxidation: { name: 'Oxidation damage of the polyamide layer', cip: [['1', 'Do not clean', 'Check dechlorination (ORP, bisulphite dose); dye-test and replace the affected elements', '–', '–', '–']] },
};

/** Validation, normalisation, event detection, trends and prognosis of an operating log. */
export function analyseLog(v) {
  const issues = [], raw = (Array.isArray(v.log) ? v.log : []).filter((r) => r && isNum(r.t)).slice().sort((a, b) => a.t - b.t), area = Math.max(1, v.area);
  // 1 — data validation
  const rows = [];
  for (const r of raw) {
    const miss = NEED.filter((k) => !isNum(r[k]));
    if (miss.length) { issues.push([r.t, `Missing value: ${miss.join(', ')}`, 'Row excluded']); continue; }
    const bad = [];
    if (r.T < 1 || r.T > 50) bad.push(`temperature ${r.T} °C outside 1–50 °C`);
    if (r.Pf <= 0 || r.Pf > 130) bad.push('feed pressure outside 0–130 bar');
    if (!(r.Pc < r.Pf)) bad.push('concentrate pressure not below feed pressure');
    if (!(r.Pp < r.Pc)) bad.push('permeate pressure not below concentrate pressure');
    if (!(r.Qp > 0 && r.Qp < 0.97 * r.Qf)) bad.push('permeate flow not between 0 and 97 % of feed flow');
    if (!(r.Cp > 0 && r.Cp < r.Cf)) bad.push('permeate conductivity not between 0 and feed conductivity');
    if (bad.length) { issues.push([r.t, `Range check failed: ${bad.join('; ')}`, 'Row excluded']); continue; }
    rows.push(r);
  }
  if (rows.length < 14) throw new Error('The operating log needs at least 14 complete, plausible rows (time, pressures, flows, conductivities, temperature).');
  const dts = rows.slice(1).map((r, i) => r.t - rows[i].t), dtMed = Math.max(med(dts), 1e-6);
  rows.slice(1).forEach((r, i) => { if (r.t - rows[i].t > Math.max(2.5 * dtMed, dtMed + 1.5)) issues.push([rows[i].t, `Gap of ${fmt(r.t - rows[i].t, 3)} d in the record`, 'Kept as a gap; trends bridge it without interpolation']); });
  // 2 — row quantities
  const s1 = clamp(v.stage1Share / 100, 0.3, 0.9), mF = v.mFlow, visc = (T) => (v.viscCorr ? (viscosity(T) / viscosity(25)) ** 0.3 : 1);
  const calc = (r) => {
    const Y = r.Qp / r.Qf, tdsF = tdsFromEC(r.Cf), tdsP = tdsFromEC(r.Cp), Cfb = (tdsF * Math.log(1 / (1 - Y))) / Y, dP = r.Pf - r.Pc;
    const ndp = r.Pf - dP / 2 - r.Pp - piAstm(Cfb, r.T) + piAstm(tdsP, r.T), tc = tcfM(r.T), sp = (100 * tdsP) / Cfb, qa = r.Qf - r.Qp / 2, stage = isNum(r.Pi) && r.Pi < r.Pf && r.Pi > r.Pc;
    const Jw = r.Qp / 3600 / area;
    return { t: r.t, Y, tdsF, tdsP, Cfb, dP, ndp, tcf: tc, sp, qa, flux: (r.Qp * 1000) / area, kA: r.Qp / (ndp * tc), kS: (sp * r.Qp) / tc, kD: dP / qa ** mF / visc(r.T),
      kD1: stage ? (r.Pf - r.Pi) / (r.Qf - (s1 * r.Qp) / 2) ** mF / visc(r.T) : null, kD2: stage ? (r.Pi - r.Pc) / (r.Qf - s1 * r.Qp - ((1 - s1) * r.Qp) / 2) ** mF / visc(r.T) : null,
      Rtot: (ndp * tc * 1e5) / (MU25 * Jw), piF: piAstm(Cfb, r.T), row: r };
  };
  let all = rows.map(calc);
  all.filter((q) => !(q.ndp > 0.05)).forEach((q) => issues.push([q.t, 'Net driving pressure is not positive — pressures or conductivities inconsistent', 'Row excluded']));
  all = all.filter((q) => q.ndp > 0.05);
  // 3 — outliers (isolated spikes; persistent steps are kept because the running median follows them)
  const flag = new Array(all.length).fill(false);
  for (const [k, name] of [['kA', 'normalised permeate flow'], ['kS', 'normalised salt passage'], ['kD', 'normalised differential pressure']]) hampel(all.map((q) => q[k]), 3, v.outlierSigma).forEach((f, i) => { if (f && !flag[i]) { flag[i] = true; issues.push([all[i].t, `Outlier in ${name}`, 'Excluded from trends and statistics']); } });
  const G = all.filter((_, i) => !flag[i]);
  if (G.length < 12) throw new Error('Too few valid rows remain after data validation.');
  const stageOK = G.filter((q) => q.kD1 != null).length > 0.8 * G.length;
  // 4 — reference state
  const nBase = clamp(Math.round(v.nBase), 3, Math.max(3, Math.floor(G.length / 3))), base = G.slice(0, nBase);
  let ref;
  if (v.refMode === 'design') {
    const ok = v.dQp > 0 && v.dQp < 0.97 * v.dQf && v.dPc < v.dPf && v.dPp < v.dPc && v.dCp > 0 && v.dCp < v.dCf, d = ok ? calc({ t: 0, Pf: v.dPf, Pc: v.dPc, Pp: v.dPp, Qf: v.dQf, Qp: v.dQp, Cf: v.dCf, Cp: v.dCp, T: v.dT }) : null;
    if (d && d.ndp > 0.05) ref = { kA: d.kA, kS: d.kS, kD: d.kD, ndp: d.ndp, Qp: v.dQp, qa: d.qa, Rm: d.Rtot };
    else issues.push(['–', 'The design reference point is inconsistent (no positive net driving pressure, or flows, pressures or conductivities out of order)', 'The baseline of the log is used as reference instead']);
  }
  if (!ref) ref = { kA: mean(base.map((q) => q.kA)), kS: mean(base.map((q) => q.kS)), kD: mean(base.map((q) => q.kD)), ndp: mean(base.map((q) => q.ndp)), Qp: mean(base.map((q) => q.row.Qp)), qa: mean(base.map((q) => q.qa)), Rm: mean(base.map((q) => q.Rtot)) };
  const sb = base.filter((q) => q.kD1 != null);
  ref.kD1 = stageOK && sb.length ? mean(sb.map((q) => q.kD1)) : 1; ref.kD2 = stageOK && sb.length ? mean(sb.map((q) => q.kD2)) : 1;
  const norm = (q) => ({ npf: (100 * q.kA) / ref.kA, nsp: (100 * q.kS) / ref.kS, dpn: (100 * q.kD) / ref.kD, dpn1: q.kD1 != null ? (100 * q.kD1) / ref.kD1 : null, dpn2: q.kD2 != null ? (100 * q.kD2) / ref.kD2 : null });
  G.forEach((q) => Object.assign(q, norm(q)));
  all.forEach((q, i) => { if (flag[i]) Object.assign(q, norm(q)); });
  const t = G.map((q) => q.t), npf = G.map((q) => q.npf), nsp = G.map((q) => q.nsp), ndp = G.map((q) => q.dpn), n = G.length;
  // 5 — sudden changes
  const w = clamp(Math.round(v.stepWin), 2, Math.max(2, Math.floor(n / 4))), series = { npf, nsp, ndp }, thr = { npf: v.stepNPF, nsp: v.stepNSP, ndp: v.stepNDP }, scans = {}, sig = {};
  for (const k of Object.keys(series)) { scans[k] = stepScan(series[k], w); sig[k] = mad(series[k].slice(1).map((x, i) => x - series[k][i])) / Math.SQRT2; }
  const score = t.map((_, i) => Math.max(...Object.keys(series).map((k) => Math.abs(scans[k][i]) / Math.max(thr[k], 4 * sig[k]))));
  const events = [];
  for (let i = w; i <= n - w; i++) {
    if (!(score[i] > 1)) continue;
    let j = i, best = i;
    while (j + 1 <= n - w && score[j + 1] > 1) { j++; if (score[j] > score[best]) best = j; }
    // refine the location: the largest single-interval jump of the dominant signal near the detected index
    const dom = Object.keys(series).reduce((a, k) => (Math.abs(scans[k][best]) / thr[k] > Math.abs(scans[a][best]) / thr[a] ? k : a), 'npf');
    let at = best, big = 0;
    for (let q = Math.max(1, best - w + 1); q <= Math.min(n - 1, best + w - 1); q++) { const jmp = Math.abs(series[dom][q] - series[dom][q - 1]); if (jmp > big) { big = jmp; at = q; } }
    const lvl = (k) => med(series[k].slice(at, Math.min(n, at + w))) - med(series[k].slice(Math.max(0, at - w), at));
    const e = { i: at, t: t[at], dNPF: lvl('npf'), dNSP: lvl('nsp'), dNDP: lvl('ndp') };
    // a steep ramp is not a step: over a three-times wider span a ramp grows threefold, a step does not
    const wide = at - 2 * w >= 0 && at + 2 * w <= n ? med(series[dom].slice(at + w, at + 2 * w)) - med(series[dom].slice(at - 2 * w, at - w)) : null, own = lvl(dom);
    if (wide != null && Math.abs(own) < 0.72 * Math.abs(wide) && Math.sign(own) === Math.sign(wide)) { i = j; continue; }
    e.type = e.dNSP > thr.nsp && e.dNDP > -thr.ndp && e.dNPF < 2 * thr.npf ? 'integrity' : e.dNPF > thr.npf ? 'cleaning' : e.dNPF < -thr.npf ? 'loss' : e.dNDP > thr.ndp ? 'dpjump' : 'other';
    e.label = { integrity: 'Sudden salt-passage rise — integrity fault (O-ring, glue line) suspected', cleaning: 'Performance restored — cleaning or element replacement', loss: 'Sudden permeability loss — upset, particulate slug or compaction', dpjump: 'Sudden pressure-drop rise — particulate slug, spacer blockage or telescoping', other: 'Sudden change' }[e.type];
    events.push(e); i = j + w - 1;
  }
  // gradual components: sudden non-cleaning steps removed
  const destep = (x, key) => x.map((val, i) => val - sum(events.filter((e) => e.type !== 'cleaning' && e.i <= i).map((e) => e[key])));
  const nspG = destep(nsp, 'dNSP'), npfG = destep(npf, 'dNPF');
  // 6 — cycles between cleanings
  const cuts = [0, ...events.filter((e) => e.type === 'cleaning').map((e) => e.i), n], cycles = [];
  const seg = (arr, a, b) => arr.slice(a, b), ts = (a, b, y) => (b - a >= 4 ? theilSen(seg(t, a, b), seg(y, a, b)) : { slope: 0, lo: 0, hi: 0, intercept: b > a ? med(seg(y, a, b)) : 0, n: b - a });
  const stageSeries = (k) => G.map((q) => q[k]);
  for (let c = 0; c + 1 < cuts.length; c++) {
    const a = cuts[c], b = cuts[c + 1];
    if (b - a < 2) continue;
    const lateA = Math.max(a, b - Math.max(8, Math.min(Math.round(v.lateWin / dtMed), Math.floor((b - a) / 2)))), earlyB = Math.max(a + 4, Math.min(lateA, a + Math.floor((b - a) / 2)));
    const f = (y, a0, b0) => ts(a0, b0, y), d1 = stageOK ? stageSeries('dpn1') : ndp, d2 = stageOK ? stageSeries('dpn2') : ndp, ok = (y, a0, b0) => { const idx = []; for (let i = a0; i < b0; i++) if (y[i] != null) idx.push(i); return idx.length >= 4 ? theilSen(idx.map((i) => t[i]), idx.map((i) => y[i])) : { slope: 0, lo: 0, hi: 0, intercept: 0, n: 0 }; };
    const cy = { a, b, t0: t[a], t1: t[b - 1], days: t[b - 1] - t[a], npf: f(npfG, a, b), nsp: f(nspG, a, b), ndp: f(ndp, a, b), ndp1: ok(d1, a, b), ndp2: ok(d2, a, b),
      late: { a: lateA, npf: f(npfG, lateA, b), nsp: f(nspG, lateA, b), ndp: f(ndp, lateA, b), ndp1: ok(d1, lateA, b), ndp2: ok(d2, lateA, b) }, early: { npf: f(npfG, a, earlyB), ndp1: ok(d1, a, earlyB) },
      startNPF: (() => { const m = Math.min(b, a + Math.max(8, Math.round(30 / dtMed))), q = ts(a, m, npf); return m - a >= 8 ? q.intercept + q.slope * t[a] : med(seg(npf, a, m)); })(), endNPF: med(seg(npf, Math.max(a, b - 5), b)), startNSP: med(seg(nsp, a, Math.min(b, a + 5))), endNSP: med(seg(nsp, Math.max(a, b - 5), b)), startNDP: med(seg(ndp, a, Math.min(b, a + 5))), endNDP: med(seg(ndp, Math.max(a, b - 5), b)),
      steps: events.filter((e) => e.type !== 'cleaning' && e.i >= a && e.i < b) };
    cycles.push(cy);
  }
  // 7 — foulant evidence scores per cycle (late window = the condition that led to the cleaning)
  const P = (x, s) => clamp(x / s, 0, 1), risk = { sdi: P(v.sdi - 3, 2), toc: P(v.toc - 2, 3), bio: clamp(0.5 * P(v.aoc - 10, 40) + 0.3 * P(mean(G.map((q) => q.row.T)) - 20, 12) + (v.biocide ? 0 : 0.2), 0, 1), scale: P(v.siMargin + 0.5, 1) };
  for (const cy of cycles) {
    const L = cy.late, rN = -L.npf.slope, r1 = L.ndp1.slope, r2 = L.ndp2.slope, rS = L.nsp.slope, lead = stageOK ? (r1 >= r2 ? 1 : 0.5) : 0.75, tail = stageOK ? P(r2 - r1, 0.2) : 0.5 * P(L.ndp.slope, 0.3);
    const accel = cy.early.ndp1.slope > 1e-4 ? r1 / cy.early.ndp1.slope : r1 > 0.02 ? 3 : 1, stepSP = sum(cy.steps.filter((e) => e.type === 'integrity').map((e) => e.dNSP));
    const s = {
      colloidal: 0.45 * P(r1, 0.15) * lead + 0.25 * P(rN, 0.15) + 0.15 * (1 - P(Math.abs(rS), 0.3)) + 0.15 * risk.sdi,
      bio: 0.4 * P(r1, 0.3) * lead + 0.25 * P(accel - 1.2, 1.5) * P(r1, 0.1) + 0.15 * P(rN, 0.2) + 0.2 * risk.bio * P(r1, 0.05),
      organic: 0.45 * P(rN, 0.15) * (1 - P(Math.max(r1, r2), 0.15)) + 0.2 * (1 - P(rS, 0.1)) * P(rN, 0.05) + 0.35 * risk.toc * P(rN, 0.05),
      scaling: 0.35 * tail + 0.25 * P(rS, 0.2) + 0.2 * P(rN, 0.3) + 0.2 * risk.scale * P(rN + r2, 0.1),
      integrity: stepSP > 0 ? 0.8 * P(stepSP, 2 * v.stepNSP) + 0.2 : 0,
      oxidation: 0.7 * P(rS, 0.2) * P(-rN, 0.05) + 0.3 * P(rS, 0.5) * (rN < 0.02 ? 1 : 0),
    };
    const tot = sum(Object.values(s)) || 1;
    cy.scores = s; cy.prob = Object.fromEntries(Object.entries(s).map(([k, x]) => [k, x / tot]));
    cy.top = Object.keys(s).reduce((a, k) => (s[k] > s[a] ? k : a), 'colloidal'); cy.topFoulant = ['colloidal', 'bio', 'organic', 'scaling'].reduce((a, k) => (s[k] > s[a] ? k : a), 'colloidal');
    cy.loss = Math.max(0, cy.startNPF - cy.endNPF); cy.features = { rN, r1, r2, rS, accel };
  }
  const cur = cycles[cycles.length - 1], wts = {};
  for (const cy of cycles) for (const k of ['colloidal', 'bio', 'organic', 'scaling']) wts[k] = (wts[k] || 0) + (cy.loss + 0.5) * cy.prob[k];
  const dominant = Object.keys(wts).reduce((a, k) => (wts[k] > wts[a] ? k : a), 'colloidal');
  // 8 — statistical process control on the normalised permeate flow (re-baselined after each cleaning)
  const spc = { z: [], ewma: [], cp: [], cm: [], alarms: [], limE: v.ewmaL * Math.sqrt(v.ewmaLambda / (2 - v.ewmaLambda)), h: v.cusumH };
  for (const cy of cycles) {
    const nb = Math.min(nBase, Math.max(3, Math.floor((cy.b - cy.a) / 3))), mu = mean(seg(npf, cy.a, cy.a + nb)), sd = Math.max(mad(seg(npf, cy.a, cy.a + nb)), 0.25);
    let e = 0, cp = 0, cm = 0, fired = false;
    for (let i = cy.a; i < cy.b; i++) {
      const z = (npf[i] - mu) / sd;
      e = v.ewmaLambda * z + (1 - v.ewmaLambda) * e; cp = Math.max(0, cp + z - v.cusumK); cm = Math.max(0, cm - z - v.cusumK);
      spc.z.push(z); spc.ewma.push(e); spc.cp.push(cp); spc.cm.push(cm);
      if (!fired && i >= cy.a + nb && (cm > v.cusumH || e < -spc.limE)) { fired = true; spc.alarms.push({ t: t[i], kind: cm > v.cusumH ? 'CUSUM' : 'EWMA', npf: npf[i], cycle: cycles.indexOf(cy) + 1 }); }
    }
    cy.alarm = fired ? spc.alarms[spc.alarms.length - 1] : null;
    const trigAt = t.find((x, i) => i >= cy.a && i < cy.b && npf[i] <= 100 - v.trigNPF);
    cy.lead = fired && trigAt != null ? trigAt - cy.alarm.t : null;
  }
  // 9 — resistances in series (Darcy) and cleaning effectiveness
  const Rm = ref.Rm, Rt = G.map((q) => q.Rtot), cleanings = [];
  cycles.slice(1).forEach((cy, c) => {
    const pre = cycles[c], rB = med(seg(Rt, Math.max(pre.a, pre.b - 5), pre.b)), rA = med(seg(Rt, cy.a, Math.min(cy.b, cy.a + 5))), mA = Math.min(cy.b - cy.a, Math.max(8, Math.round(30 / dtMed))), sdA = (1.6 * mad(seg(npf, cy.a, cy.a + mA).map((x, i) => x - cy.npf.slope * t[cy.a + i]))) / Math.sqrt(mA);
    cleanings.push({ t: cy.t0, npfBefore: pre.endNPF, npfAfter: cy.startNPF, recovery: pre.endNPF < 100 ? clamp((cy.startNPF - pre.endNPF) / (100 - pre.endNPF), -1, 1.5) : 1, ndpBefore: pre.endNDP, ndpAfter: cy.startNDP, dpRecovery: pre.endNDP > 100 ? clamp((pre.endNDP - cy.startNDP) / (pre.endNDP - 100), -1, 1.5) : 1,
      nspBefore: pre.endNSP, nspAfter: cy.startNSP, Rrev: Math.max(0, rB - rA), Rirr: Math.max(0, rA - Rm), irr: Math.max(0, 100 - cy.startNPF), sdA, foulant: pre.top });
  });
  const lastClean = cleanings[cleanings.length - 1] || null, rNow = med(Rt.slice(-5)), Rirr = lastClean ? lastClean.Rirr : (1 - v.revFrac / 100) * Math.max(0, rNow - Rm), Rrev = Math.max(0, rNow - Rm - Rirr);
  // 10 — triggers and forecast for the current cycle
  const now = { npf: med(npf.slice(-5)), nsp: med(nsp.slice(-5)), ndp: med(ndp.slice(-5)), nspG: med(nspG.slice(-5)), t: t[n - 1] }, base0 = v.refAfter === 'postclean' ? { npf: cur.startNPF, nsp: cur.startNSP, ndp: cur.startNDP } : { npf: 100, nsp: 100, ndp: 100 };
  const integrity = events.some((e) => e.type === 'integrity' && e.i >= cur.a);
  const eta = (margin, r, rlo, rhi) => { const d = (x) => (margin <= 0 ? 0 : x > 1e-9 ? Math.min(3650, margin / x) : 3650); return { days: d(r), lo: d(rhi), hi: d(rlo), margin }; };
  const trig = {
    npf: { name: 'Normalised permeate flow', now: now.npf, limit: base0.npf * (1 - v.trigNPF / 100), rate: cur.npf.slope, ...eta(now.npf - base0.npf * (1 - v.trigNPF / 100), -cur.npf.slope, -cur.npf.hi, -cur.npf.lo) },
    ndp: { name: 'Normalised differential pressure', now: now.ndp, limit: base0.ndp * (1 + v.trigDP / 100), rate: cur.ndp.slope, ...eta(base0.ndp * (1 + v.trigDP / 100) - now.ndp, cur.ndp.slope, cur.ndp.lo, cur.ndp.hi) },
    nsp: { name: 'Normalised salt passage', now: now.nsp, limit: base0.nsp * (1 + v.trigSP / 100), rate: cur.nsp.slope, ...eta(base0.nsp * (1 + v.trigSP / 100) - (integrity ? now.nspG : now.nsp), cur.nsp.slope, cur.nsp.lo, cur.nsp.hi) },
  };
  const first = ['npf', 'ndp', ...(integrity ? [] : ['nsp'])].reduce((a, k) => (trig[k].days < trig[a].days ? k : a), 'npf');
  const daysToCleaning = trig[first].days, since = now.t - cur.t0, lens = [...cycles.slice(0, -1).map((c) => c.days + dtMed), since + daysToCleaning].filter((x) => x > 0);
  const cleaningsPerYear = clamp(365 / Math.max(mean(lens), 1), 0.05, 52);
  // 11 — irreversible loss and remaining useful life
  const span = Math.max(now.t - t[0], 1);
  let irrRate, irrLo, irrHi, irrNow;
  if (lastClean && lastClean.t - t[0] > 5) { const el = lastClean.t - t[0]; irrNow = lastClean.irr; irrRate = lastClean.irr / el; irrLo = Math.max(0, lastClean.irr - 2 * lastClean.sdA - 0.3) / el; irrHi = (lastClean.irr + 2 * lastClean.sdA + 0.3) / el; }
  else { const f = 1 - v.revFrac / 100, c0 = cycles[0]; irrRate = Math.max(0, -c0.npf.slope) * f; irrLo = Math.max(0, -c0.npf.hi) * f; irrHi = Math.max(0, -c0.npf.lo) * f; irrNow = irrRate * span; }
  const lifeOf = (r) => (r > 1e-6 ? clamp(Math.max(0, v.eolLoss - irrNow) / r / 365, 0, 25) : 25), rul = lifeOf(irrRate), rulLo = lifeOf(irrHi), rulHi = lifeOf(irrLo), age = v.age0 + span / 365;
  return { v, issues, rows, all, flag, G, t, npf, nsp, ndp, npfG, nspG, ref, stageOK, events, cycles, cur, dominant, wts, spc, Rm, Rt, Rirr, Rrev, cleanings, lastClean, now, trig, first, daysToCleaning, cleaningsPerYear, integrity, irrRate, irrLo, irrHi, irrNow, rul, rulLo, rulHi, age, dtMed, risk, nBase, area, since };
}

/** Fit the four Hermia laws to a flux-decline series with a chronological train/validation split. */
export function fitHermia(t, y, trainFrac = 0.7) {
  const n = t.length, nt = clamp(Math.round(trainFrac * n), 4, n - 2), tt = t.slice(0, nt), yt = y.slice(0, nt), tv = t.slice(nt), yv = y.slice(nt);
  const r2 = (m, p) => { const mm = mean(m), st = sum(m.map((x) => (x - mm) ** 2)); return st > 0 ? 1 - sum(m.map((x, i) => (x - p[i]) ** 2)) / st : 0; };
  const k0 = Math.max(1e-5, (1 - y[n - 1] / y[0]) / Math.max(t[n - 1], 1e-9));
  const fits = Object.keys(HERMIA).map((law) => {
    const f = levenbergMarquardt((p) => tt.map((x, i) => p[0] * hermia(law, p[1] * x) - yt[i]), [yt[0], k0], { lo: [0.2 * yt[0], 0], hi: [2 * yt[0], 10] });
    const pt = tt.map((x) => f.p[0] * hermia(law, f.p[1] * x)), pv = tv.map((x) => f.p[0] * hermia(law, f.p[1] * x));
    return { law, J0: f.p[0], k: f.p[1], r2: r2(yt, pt), rmseVal: Math.sqrt(mean(pv.map((x, i) => (x - yv[i]) ** 2))), biasVal: mean(pv.map((x, i) => x - yv[i])), r2Val: r2(yv, pv), sse: f.sse };
  });
  const best = fits.reduce((a, b) => (b.r2 > a.r2 ? b : a), fits[0]);
  return { fits, best, split: t[nt - 1], nTrain: nt, nVal: n - nt, tTrainMax: t[nt - 1], tValMin: t[nt] };
}

/** Critical flux (m/s) from Brownian (Lévêque) and shear-induced back-transport of particles of diameter dp. */
export function criticalFlux(dp, u, { h = 7.1e-4, L = 6, T = 25, phiB = 1e-6, phiW = 0.6 } = {}) {
  const gam = (6 * u) / h, D = (KB * (T + KELVIN)) / (3 * Math.PI * viscosity(T) * dp), ln = Math.log(phiW / phiB);
  const kB = 0.807 * ((gam * D * D) / L) ** (1 / 3), kS = 0.078 * ((dp / 2) ** 4 / L) ** (1 / 3) * gam;
  return { J: Math.max(kB, kS) * ln, brownian: kB * ln, shear: kS * ln, shearRate: gam };
}

/** Monod biofilm growth with a carrying capacity, integrated with fixed-step RK4. Returns biomass over time (g/m²). */
export function biofilm({ X0, mumax, Ks, S, kd, Xmax }, days, nStep) {
  const r = (mumax * S) / (Ks + S) - kd, N = Math.max(2, Math.round(nStep), Math.ceil((Math.abs(r) * days) / 2)); // at least the explicit stability limit
  const sol = rk4((_, y) => [r * y[0] * (1 - y[0] / Xmax)], [X0], 0, days, N);
  return { t: sol.t, X: sol.y.map((y) => clamp(y[0], 0, 2 * Xmax)), r, exact: (tt) => Xmax / (1 + ((Xmax - X0) / X0) * Math.exp(-r * tt)) };
}

// ---- adsorption isotherms ------------------------------------------------------------------------------------
/** Langmuir isotherm q = qmax·K·C / (1 + K·C). */
export const langmuir = (C, qmax, K) => (qmax * K * C) / (1 + K * C);
/** Freundlich isotherm q = Kf·C^(1/n). */
export const freundlich = (C, Kf, n) => Kf * Math.max(C, 0) ** (1 / n);
const r2of = (m, p) => { const mm = mean(m), st = sum(m.map((x) => (x - mm) ** 2)); return st > 0 ? 1 - sum(m.map((x, i) => (x - p[i]) ** 2)) / st : 0; };
/** Least-squares fit of both isotherms to adsorption data [{ C (mg/L), q (mg/m²) }]. */
export function fitIsotherms(rows) {
  const d = (Array.isArray(rows) ? rows : []).filter((r) => r && isNum(r.C) && isNum(r.q) && r.C > 0 && r.q > 0).sort((a, b) => a.C - b.C);
  if (d.length < 3) return null;
  const C = d.map((r) => r.C), q = d.map((r) => r.q), qm = Math.max(...q), Cm = Math.max(...C);
  const fl = levenbergMarquardt((p) => C.map((c, i) => langmuir(c, p[0], p[1]) - q[i]), [1.3 * qm, 2 / Cm], { lo: [0.2 * qm, 1e-6 / Cm], hi: [50 * qm, 1e4 / Cm] });
  const lf = linfit(C.map(Math.log), q.map(Math.log)), ff = levenbergMarquardt((p) => C.map((c, i) => freundlich(c, p[0], p[1]) - q[i]), [Math.exp(lf.intercept), clamp(1 / Math.max(lf.slope, 0.05), 0.3, 20)], { lo: [1e-9, 0.2], hi: [1e6, 50] });
  const pl = C.map((c) => langmuir(c, fl.p[0], fl.p[1])), pf = C.map((c) => freundlich(c, ff.p[0], ff.p[1])), rm = (p) => Math.sqrt(mean(p.map((x, i) => (x - q[i]) ** 2)));
  const L = { qmax: fl.p[0], K: fl.p[1], r2: r2of(q, pl), rmse: rm(pl) }, Fr = { Kf: ff.p[0], n: ff.p[1], r2: r2of(q, pf), rmse: rm(pf) }, best = L.rmse <= Fr.rmse ? 'langmuir' : 'freundlich';
  return { C, q, langmuir: L, freundlich: Fr, best, at: (c) => (best === 'langmuir' ? langmuir(c, L.qmax, L.K) : freundlich(c, Fr.Kf, Fr.n)), n: d.length };
}

// ---- combined fouling laws --------------------------------------------------------------------------------------
/** Ho–Zydney combined pore-blocking and cake-filtration law: J/J0 = e^(−bt) + (1 − e^(−bt)) / ((1 + ρ0)·√(1 + g·t)). */
export const hoZydney = (t, b, rho0, g) => { const e = Math.exp(-b * t); return e + (1 - e) / ((1 + rho0) * Math.sqrt(1 + g * t)); };
/** Adsorption–pore-blocking law: open-pore fraction ε0·e^(−kb·t) times the permeability left by the adsorbed layer, 1 − λ(1 − e^(−ka·t)). */
export const adsorptionBlocking = (t, ka, kb, lam, eps0 = 1) => eps0 * Math.exp(-kb * t) * (1 - lam * (1 - Math.exp(-ka * t)));
/** Fit both combined laws to a normalised flux series with a chronological split. lam = adsorptive loss at equilibrium coverage. */
export function fitCombined(t, y, lam, trainFrac = 0.7) {
  const n = t.length, nt = clamp(Math.round(trainFrac * n), 5, n - 2), tt = t.slice(0, nt), yt = y.slice(0, nt), tv = t.slice(nt), yv = y.slice(nt), k0 = Math.max(1e-5, (1 - y[n - 1] / y[0]) / Math.max(t[n - 1], 1e-9));
  const pack = (name, f, p, npar) => { const pt = tt.map((x) => f(x, p)), pv = tv.map((x) => f(x, p)); return { name, p, f: (x) => f(x, p), r2: r2of(yt, pt), rmseVal: Math.sqrt(mean(pv.map((x, i) => (x - yv[i]) ** 2))), r2Val: r2of(yv, pv), npar }; };
  const fh = (x, p) => p[0] * hoZydney(x, p[1], p[2], p[3]), hz = levenbergMarquardt((p) => tt.map((x, i) => fh(x, p) - yt[i]), [yt[0], 3 * k0, 0.2, 3 * k0], { lo: [0.5 * yt[0], 0, 0, 0], hi: [1.5 * yt[0], 5, 50, 50], maxIter: 30 });
  const la = clamp(lam, 0, 0.95), fa = (x, p) => p[0] * adsorptionBlocking(x, p[1], p[2], la), ab = levenbergMarquardt((p) => tt.map((x, i) => fa(x, p) - yt[i]), [yt[0], 20 * k0 + 0.05, k0], { lo: [0.5 * yt[0], 1e-4, 0], hi: [1.5 * yt[0], 50, 5], maxIter: 40 });
  return { hoZydney: pack('Cake–pore blocking (Ho–Zydney)', fh, hz.p, 4), adsBlock: pack('Adsorption–pore blocking', fa, ab.p, 3), nTrain: nt, nVal: n - nt, lam: la };
}

// ---- deposition and detachment ------------------------------------------------------------------------------------
/**
 * Deposit mass per membrane area under a constant deposition flux a (kg/m²·d) and first-order shear detachment k (1/d):
 * dm/dt = a − k·m, m(t) = a/k·(1 − e^(−kt)) + m0·e^(−kt).
 */
export const depositMass = (t, a, k, m0 = 0) => (k > 1e-12 ? (a / k) * (1 - Math.exp(-k * t)) + m0 * Math.exp(-k * t) : m0 + a * t);
/** Fit deposition flux and detachment constant to a deposit-mass series. */
export function fitDeposition(t, m, m0 = 0) {
  if (t.length < 5) return null;
  const a0 = Math.max((m[m.length - 1] - m0) / Math.max(t[t.length - 1], 1e-9), 1e-12), sc = Math.max(...m.map(Math.abs), 1e-12);
  const f = levenbergMarquardt((p) => t.map((x, i) => (depositMass(x, p[0], p[1], m0) - m[i]) / sc), [a0, 0.5 / Math.max(t[t.length - 1], 1e-9)], { lo: [0, 1e-6], hi: [1e3 * a0 + 1e-9, 10] });
  const pr = t.map((x) => depositMass(x, f.p[0], f.p[1], m0));
  return { a: f.p[0], k: f.p[1], r2: r2of(m, pr), mss: f.p[0] / f.p[1], pred: pr };
}

// ---- threshold flux ---------------------------------------------------------------------------------------------
/** Hinge fit of a flux-stepping test: fouling rate = a + b·max(0, J − Jth) (Field & Pearce). rows = [{ J, rate }]. */
export function thresholdFlux(rows) {
  const d = (Array.isArray(rows) ? rows : []).filter((r) => r && isNum(r.J) && isNum(r.rate)).sort((p, q) => p.J - q.J);
  if (d.length < 5) return null;
  const J = d.map((r) => r.J), y = d.map((r) => r.rate), n = J.length;
  const at = (jt) => { // linear least squares for a and b at a given threshold
    const x = J.map((j) => Math.max(0, j - jt)), mx = mean(x), my = mean(y);
    let sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sxx += (x[i] - mx) ** 2; sxy += (x[i] - mx) * (y[i] - my); }
    const b = sxx > 0 ? sxy / sxx : 0, a = my - b * mx;
    return { a, b, sse: sum(y.map((v, i) => (v - a - b * x[i]) ** 2)) };
  };
  let lo = J[1], hi = J[n - 2], best = lo, bs = Infinity;
  for (let i = 0; i <= 60; i++) { const jt = lo + ((hi - lo) * i) / 60, s = at(jt).sse; if (s < bs) { bs = s; best = jt; } }
  let a = Math.max(lo, best - (hi - lo) / 60), b = Math.min(hi, best + (hi - lo) / 60);
  for (let i = 0; i < 40; i++) { const m1 = a + 0.382 * (b - a), m2 = a + 0.618 * (b - a); if (at(m1).sse < at(m2).sse) b = m2; else a = m1; } // golden section
  const jt = 0.5 * (a + b), f = at(jt), lin = linfit(J, y), sseLin = sum(y.map((v, i) => (v - lin.intercept - lin.slope * J[i]) ** 2));
  return { Jth: jt, a: f.a, b: f.b, sse: f.sse, r2: r2of(y, J.map((j) => f.a + f.b * Math.max(0, j - jt))), J, rate: y, rateAt: (j) => f.a + f.b * Math.max(0, j - jt), significant: f.b > 0 && f.sse < 0.6 * sseLin + 1e-30 };
}

// ---- concentration polarisation and its coupling to the deposit ----------------------------------------------------
/** Film-theory mass-transfer coefficient in a spacer-filled channel (Schock & Miquel): Sh = 0.065·Re^0.875·Sc^0.25. */
export function massTransfer(u, h, T, D) {
  const dh = 1.236 * h, mu = viscosity(T), rho = 997, Re = (rho * u * dh) / mu, Sc = mu / (rho * D), Sh = 0.065 * Re ** 0.875 * Sc ** 0.25;
  return { k: (Sh * D) / dh, Re, Sc, Sh, dh };
}
/**
 * Concentration polarisation with a deposit layer (cake-enhanced osmotic pressure, Hoek & Elimelech):
 * 1/k* = 1/k + δ·(1 − ln ε²)/(D·ε); β = exp(J/k). The deposit mass m is found so that its hydraulic pressure loss
 * μ·J·α·m plus the extra osmotic pressure π_b·(β* − β) equals the observed fouling pressure loss dPf (Pa).
 */
export function cakeEnhancedCP({ J, k, D, eps, rhoP, alpha, piB, dPf, mu = MU25 }) {
  const tort = 1 - Math.log(eps * eps), kStar = (m) => 1 / (1 / k + ((m / (rhoP * (1 - eps))) * tort) / (D * eps)), beta = Math.exp(J / k);
  const loss = (m) => mu * J * alpha * m + piB * (Math.exp(Math.min(J / kStar(m), 20)) - beta), mHyd = Math.max(dPf, 0) / (mu * J * alpha);
  let m = 0;
  if (dPf > 0) { let a = 0, b = mHyd; for (let i = 0; i < 60; i++) { const c = 0.5 * (a + b); if (loss(c) > dPf) b = c; else a = c; } m = 0.5 * (a + b); }
  const bs = Math.exp(Math.min(J / kStar(m), 20)), hyd = mu * J * alpha * m, osm = piB * (bs - beta);
  return { beta, betaStar: bs, k, kStar: kStar(m), m, mHydraulicOnly: mHyd, delta: m / (rhoP * (1 - eps)), hydraulic: hyd, osmotic: osm, shareOsmotic: hyd + osm > 0 ? osm / (hyd + osm) : 0, tort };
}

// ---- biofilm growth limited by substrate transport -----------------------------------------------------------------
/**
 * Steady substrate profile in a biofilm of thickness Lf: Df·S'' = (μmax·Xf/Y)·S/(Ks + S), zero flux at the membrane
 * (z = 0), film transfer at the surface Df·S'(Lf) = kL·(Sb − S). Finite differences with Newton iteration.
 * Units: S, Ks, Sb in g/m³; Xf g/m³; μmax 1/s. Returns the substrate flux into the film (g/m²·s).
 */
export function biofilmSubstrate({ Sb, kL, Df, mumax, Ks, Y, Xf, Lf, nz = 20 }) {
  const N = Math.max(4, Math.round(nz)), dz = Lf / N, q = (mumax * Xf) / Y, S = new Array(N).fill(Sb), a = new Array(N), b = new Array(N), c = new Array(N), r = new Array(N), w = Df / (dz * dz), ws = 1 / (dz * (dz / (2 * Df) + 1 / kL)); // surface conductance of half a cell plus the liquid film, per cell height
  for (let it = 0; it < 40; it++) {
    for (let i = 0; i < N; i++) {
      const lo = i > 0 ? w : 0, up = i < N - 1 ? w : 0, sf = i === N - 1 ? ws : 0, den = Ks + S[i];
      r[i] = (i > 0 ? lo * (S[i - 1] - S[i]) : 0) + (i < N - 1 ? up * (S[i + 1] - S[i]) : 0) + sf * (Sb - S[i]) - (q * S[i]) / den;
      a[i] = lo; c[i] = up; b[i] = -(lo + up + sf) - (q * Ks) / (den * den);
    }
    const d = tridiag(a, b, c, r.map((x) => -x));
    let mx = 0;
    for (let i = 0; i < N; i++) { const s1 = Math.max(S[i] + d[i], 1e-9 * Sb); mx = Math.max(mx, Math.abs(s1 - S[i])); S[i] = s1; }
    if (mx < 1e-12 * Math.max(Sb, 1e-30)) break;
  }
  let cons = 0;
  for (let i = 0; i < N; i++) cons += ((q * S[i]) / (Ks + S[i])) * dz;
  return { S, z: S.map((_, i) => (i + 0.5) * dz), flux: cons, Swall: S[0], Ssurf: S[N - 1], eta: cons / Math.max(((q * Sb) / (Ks + Sb)) * Lf, 1e-300), thiele: Lf * Math.sqrt(q / (Ks * Df)) };
}
/** Biofilm thickness over time: growth from the substrate flux, loss by decay and detachment. Times in days, lengths in m. */
export function biofilmTransport(p, days, nStep) {
  const N = Math.max(4, Math.round(nStep)), dt = (days / N) * 86400, t = [0], Lf = [p.Lf0], X = [p.Lf0 * p.Xf], eta = [], flux = [];
  const rate = (L) => { const s = biofilmSubstrate({ ...p, Lf: Math.max(L, 1e-9) }); return { d: (p.Y * s.flux) / p.Xf - p.kd * L, s }; };
  let L = p.Lf0;
  for (let i = 0; i < N; i++) {
    const k1 = rate(L), Lp = Math.max(L + dt * k1.d, 1e-9), k2 = rate(Lp);
    eta.push(k1.s.eta); flux.push(k1.s.flux);
    L = clamp(L + 0.5 * dt * (k1.d + k2.d), 1e-9, p.Lmax ?? 1e-2);
    t.push(((i + 1) * days) / N); Lf.push(L); X.push(L * p.Xf);
  }
  const end = biofilmSubstrate({ ...p, Lf: L });
  eta.push(end.eta); flux.push(end.flux);
  return { t, Lf, X, eta, flux, end, supplyLimit: p.kL * p.Sb };
}

// ---- deposition profile along the feed channel ------------------------------------------------------------------------
/**
 * One-dimensional feed channel from inlet to concentrate outlet with permeation through both walls, in one or more
 * stages of different total channel width (o.widths, m): cross-flow, salt and particle concentration, pressure (spacer
 * friction f = fMult·6.23·Re^−0.3), local flux through membrane + deposit, local wall shear and critical flux, and the
 * deposit m(x, t) from deposition above the critical flux and shear detachment. The outlet pressure is prescribed; the
 * inlet pressure follows by shooting. o.dPtarget (Pa) calibrates the friction multiplier to a measured pressure drop.
 * SI units; Q0 in m³/s, L = length of one stage.
 */
export function channelFouling(o) {
  const nS = o.widths.length, Ns = Math.max(4, Math.round((o.N ?? 40) / nS)), N = Ns * nS, dx = o.L / Ns, dh = 1.236 * o.h, mu = viscosity(o.T), muJ = o.mu ?? mu, rho = 997, m = new Array(N).fill(o.m0 ?? 0), nT = Math.max(1, Math.round(o.nT ?? 12)), dt = (o.days ?? 0) / nT;
  let fMult = o.fMult ?? 1;
  const march = (Pin, rec) => {
    let Q = o.Q0, P = Pin;
    const out = rec ? { x: [], u: [], J: [], P: [], cb: [], tau: [], Jc: [] } : null;
    for (let i = 0; i < N; i++) {
      const W = o.widths[Math.floor(i / Ns)], u = Math.max(Q / (W * o.h), 1e-6), cf = o.Q0 / Math.max(Q, 1e-30), R = o.Rm + o.alpha * m[i], J = Math.max(0, (P - o.Pp - o.pi0 * cf * (o.betaCP ?? 1)) / (muJ * R));
      const Re = (rho * u * dh) / mu, dP = (fMult * (6.23 / Re ** 0.3) * rho * u * u * dx) / (2 * dh);
      if (rec) { out.x.push((i + 0.5) * dx); out.u.push(u); out.J.push(J); out.P.push(P - dP / 2); out.cb.push(cf); out.tau.push((mu * 6 * u) / o.h); out.Jc.push(criticalFlux(o.dp, u, { h: o.h, L: o.L, T: o.T, phiB: Math.min(o.phiB * cf, 0.3) }).J); }
      Q -= Math.min(2 * J * W * dx, Q * (1 - 1e-6)); P -= dP;
    }
    return { Pout: P, Q, out };
  };
  const solveIn = () => { // shooting on the inlet pressure so that the outlet pressure equals the prescribed value
    let a = o.Pout, b = o.Pout + 80e5;
    const g = (p) => march(p, false).Pout - o.Pout;
    if (g(a) > 0) return a;
    for (let i = 0; i < 60; i++) { const c = 0.5 * (a + b); if (g(c) > 0) b = c; else a = c; if (b - a < 10) break; }
    return 0.5 * (a + b);
  };
  if (o.dPtarget > 0) { // friction multiplier that reproduces the measured pressure drop (fittings, spacer fouling)
    let a = 0.05, b = 50;
    for (let i = 0; i < 40; i++) { fMult = Math.sqrt(a * b); if (solveIn() - o.Pout > o.dPtarget) b = fMult; else a = fMult; if (b / a < 1.002) break; }
    fMult = Math.sqrt(a * b);
  }
  let Pin = solveIn(), st = march(Pin, true);
  const first = { Pin, recovery: 1 - st.Q / o.Q0, J: st.out.J.slice() };
  for (let k = 0; k < nT && dt > 0; k++) {
    for (let i = 0; i < N; i++) m[i] = depositMass(dt, o.omega * o.cp * st.out.cb[i] * Math.max(0, st.out.J[i] - st.out.Jc[i]) * 86400, o.kDet * st.out.tau[i], m[i]); // exact step for constant local conditions
    Pin = solveIn(); st = march(Pin, true);
  }
  return { ...st.out, m: m.slice(), Pin, Pout: st.Pout, recovery: 1 - st.Q / o.Q0, perm: o.Q0 - st.Q, Qout: st.Q, first, fMult, dx, N, nStages: nS };
}

// ---- state-space model with Kalman filtering ---------------------------------------------------------------------------
/**
 * Local-linear-trend state-space model x = [level, slope]: level' = level + slope·Δt + w₁, slope' = slope + w₂, y = level + v.
 * qL, qS = process-noise intensities per day, r = measurement variance. Returns filtered states, covariances and innovations.
 */
export function kalmanTrend(t, y, { qL = 0, qS = 1e-6, r = 1, x0 = null, P0 = null } = {}) {
  const n = y.length, level = [], slope = [], sdL = [], sdS = [], innov = [], zs = [];
  let l = x0 ? x0[0] : y[0], s = x0 ? x0[1] : 0, p11 = P0 ? P0[0] : 4 * r, p12 = P0 ? P0[1] : 0, p22 = P0 ? P0[2] : 1, ll = 0, K = [0, 0];
  for (let i = 0; i < n; i++) {
    const dt = i > 0 ? Math.max(t[i] - t[i - 1], 0) : 0;
    // prediction
    l += s * dt; const a11 = p11 + 2 * dt * p12 + dt * dt * p22 + qL * dt, a12 = p12 + dt * p22, a22 = p22 + qS * dt;
    // update
    const S = a11 + r, e = y[i] - l, k1 = a11 / S, k2 = a12 / S;
    l += k1 * e; s += k2 * e; p11 = (1 - k1) * a11; p12 = (1 - k1) * a12; p22 = a22 - k2 * a12; K = [k1, k2];
    level.push(l); slope.push(s); sdL.push(Math.sqrt(Math.max(p11, 0))); sdS.push(Math.sqrt(Math.max(p22, 0))); innov.push(e); zs.push(e / Math.sqrt(S));
    if (i > 1) ll += -0.5 * (Math.log(2 * Math.PI * S) + (e * e) / S);
  }
  return { level, slope, sdL, sdS, innov, z: zs, logLik: ll, gain: K, P: [p11, p12, p22], state: [l, s] };
}
/** Slope-noise intensity chosen by maximum likelihood of the innovations on a small grid. */
export function kalmanAuto(t, y, r) {
  let best = null;
  for (const qS of [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]) { const f = kalmanTrend(t, y, { qS, r }); if (!best || f.logLik > best.logLik) best = { ...f, qS }; }
  return best;
}

/**
 * Row-by-row (streaming) monitor: each pushed record is validated, normalised against the baseline collected from the
 * first accepted rows, and passed through the Kalman filter and the EWMA / CUSUM charts in O(1) work per sample.
 * Returns the alerts raised by that record. A large upward jump of the normalised flow re-baselines the monitor (cleaning).
 */
export function createMonitor(v) {
  const area = Math.max(1, v.area), nB = clamp(Math.round(v.nBase ?? 10), 3, 60), base = [], S = { n: 0, accepted: 0, rejected: 0, ref: null, kf: null, e: 0, cm: 0, mu: 0, sd: 1, last: null, alerts: [], cycleStart: null, buf: [], fired: false, trig: false, r: v.kfR ?? 1, qS: v.kfQ ?? 1e-6, resets: 0, series: [] };
  const quant = (r) => { const Y = r.Qp / r.Qf, Cfb = (tdsFromEC(r.Cf) * Math.log(1 / (1 - Y))) / Y, dP = r.Pf - r.Pc, ndp = r.Pf - dP / 2 - r.Pp - piAstm(Cfb, r.T) + piAstm(tdsFromEC(r.Cp), r.T); return { ndp, kA: r.Qp / (ndp * tcfM(r.T)) }; };
  const valid = (r) => r && NEED.every((k) => isNum(r[k])) && r.T >= 1 && r.T <= 50 && r.Pf > 0 && r.Pf <= 130 && r.Pc < r.Pf && r.Pp < r.Pc && r.Qp > 0 && r.Qp < 0.97 * r.Qf && r.Cp > 0 && r.Cp < r.Cf;
  const rebase = (t) => { S.buf = []; S.kf = null; S.e = 0; S.cm = 0; S.fired = false; S.trig = false; S.cycleStart = t; };
  const push = (row, again = false) => {
    if (again) S.accepted--; else S.n++;
    const al = [], add = (type, msg) => { const a = { t: row?.t ?? null, type, msg }; al.push(a); S.alerts.push(a); };
    if (!valid(row)) { S.rejected++; add('rejected', 'Record failed the completeness or range checks'); return al; }
    const q = quant(row);
    if (!(q.ndp > 0.05)) { S.rejected++; add('rejected', 'Net driving pressure is not positive'); return al; }
    S.accepted++;
    if (!S.ref) { base.push(q.kA); if (base.length >= nB) { S.ref = mean(base); S.cycleStart = row.t; add('baseline', `Reference established from ${nB} records`); } return al; }
    const npf = (100 * q.kA) / S.ref;
    // cycle statistics for the control charts come from the first records after a (re)start
    if (!again && S.buf.length < nB) {
      S.buf.push(npf);
      if (S.buf.length === nB) { S.mu = mean(S.buf); S.sd = Math.max(mad(S.buf), 0.25); if (v.kfR == null) S.r = Math.max(S.sd * S.sd, 0.05); }
    }
    if (!S.kf) S.kf = { l: npf, s: 0, p11: 4, p12: 0, p22: 1e-2, t: row.t };
    {
      const k = S.kf, dt = Math.max(row.t - k.t, 0), lp = k.l + k.s * dt, a11 = k.p11 + 2 * dt * k.p12 + dt * dt * k.p22, a12 = k.p12 + dt * k.p22, a22 = k.p22 + S.qS * dt, Sv = a11 + S.r, e = npf - lp, z = e / Math.sqrt(Sv);
      if (z > 6 && e > (v.stepNPF ?? 3)) { S.resets++; add('cleaning', `Normalised flow jumped by ${fmt(e, 3)} % of reference: cleaning or element replacement — monitor re-baselined`); rebase(row.t); S.buf.push(npf); S.kf = { l: npf, s: 0, p11: 4, p12: 0, p22: 1e-2, t: row.t }; S.last = { t: row.t, npf, level: npf, slope: 0 }; return al; }
      if (z < -6 && -e > (v.stepNPF ?? 3)) { add('step', `Sudden loss of ${fmt(-e, 3)} % of reference`); k.p11 += e * e; k.p22 += 0.25 * (e / Math.max(dt, 1)) ** 2; return al.concat(push(row, true)); } // inflate the state covariance and process the record again
      const k1 = a11 / Sv, k2 = a12 / Sv;
      k.l = lp + k1 * e; k.s += k2 * e; k.p11 = (1 - k1) * a11; k.p12 = (1 - k1) * a12; k.p22 = a22 - k2 * a12; k.t = row.t;
    }
    if (S.buf.length >= nB) {
      const z = (npf - S.mu) / S.sd, lam = v.ewmaLambda ?? 0.2;
      S.e = lam * z + (1 - lam) * S.e; S.cm = Math.max(0, S.cm - z - (v.cusumK ?? 0.5));
      if (!S.fired && (S.cm > (v.cusumH ?? 5) || S.e < -(v.ewmaL ?? 3) * Math.sqrt(lam / (2 - lam)))) { S.fired = true; add('alarm', `${S.cm > (v.cusumH ?? 5) ? 'CUSUM' : 'EWMA'} alarm: normalised flow drifting down (${fmt(npf, 4)} % of reference)`); }
    }
    if (!S.trig && S.kf.l <= 100 - (v.trigNPF ?? 10)) { S.trig = true; add('trigger', `Filtered normalised flow ${fmt(S.kf.l, 4)} % reached the cleaning trigger`); }
    S.last = { t: row.t, npf, level: S.kf.l, slope: S.kf.s, sdSlope: Math.sqrt(Math.max(S.kf.p22, 0)) };
    S.series.push(S.last);
    return al;
  };
  return { push, state: S };
}

// ---- mechanistic model with a learned residual -----------------------------------------------------------------------
/**
 * Ridge regression of the residual of a mechanistic prediction on standardised features, trained on the first part of
 * the record and tested on the later part (chronological, no leakage). The penalty is chosen by leave-one-out
 * cross-validation on the training part. X: rows of features; y: measurements; mech: mechanistic predictions.
 */
export function hybridResidual(X, y, mech, nTrain, names = []) {
  const n = y.length, d = X[0].length, nt = clamp(Math.round(nTrain), d + 3, n - 2), mu = [], sd = [];
  for (let j = 0; j < d; j++) { const col = X.slice(0, nt).map((r) => r[j]), m = mean(col); mu.push(m); sd.push(Math.sqrt(mean(col.map((x) => (x - m) ** 2))) || 1); }
  const Z = X.map((r) => r.map((x, j) => clamp((x - mu[j]) / sd[j], -4, 4))), res = y.map((v, i) => v - mech[i]), bias = mean(res.slice(0, nt)), rt = res.slice(0, nt).map((x) => x - bias);
  const G = Array.from({ length: d }, () => new Array(d).fill(0)), g = new Array(d).fill(0);
  for (let i = 0; i < nt; i++) for (let a = 0; a < d; a++) { g[a] += Z[i][a] * rt[i]; for (let b = 0; b < d; b++) G[a][b] += Z[i][a] * Z[i][b]; }
  let best = null;
  for (const lam of [0.01, 0.1, 1, 10, 100, 1000].map((x) => x * nt / 100)) {
    const A = G.map((row, a) => row.map((x, b) => x + (a === b ? lam : 0))), w = solveLinear(A, g), inv = Array.from({ length: d }, (_, c) => solveLinear(A, Array.from({ length: d }, (_, q) => (q === c ? 1 : 0))));
    let loo = 0;
    for (let i = 0; i < nt; i++) { let pr = 0, hii = 0; for (let a = 0; a < d; a++) { pr += w[a] * Z[i][a]; for (let b = 0; b < d; b++) hii += Z[i][a] * inv[b][a] * Z[i][b]; } loo += ((rt[i] - pr) / Math.max(1 - hii, 1e-6)) ** 2; }
    if (!best || loo < best.loo) best = { lam, w, loo };
  }
  const corr = (i) => bias + sum(best.w.map((w, a) => w * Z[i][a])), hyb = mech.map((m, i) => m + corr(i)), rm = (p, a, b) => Math.sqrt(mean(p.slice(a, b).map((x, i) => (x - y[a + i]) ** 2)));
  return { weights: best.w, bias, lam: best.lam, looRmse: Math.sqrt(best.loo / nt), hybrid: hyb, nTrain: nt, nTest: n - nt, rmseMechTrain: rm(mech, 0, nt), rmseHybTrain: rm(hyb, 0, nt), rmseMechTest: rm(mech, nt, n), rmseHybTest: rm(hyb, nt, n), names };
}

// ---- surface scaling ------------------------------------------------------------------------------------------------
/** Scale mass (g/m²) on the tail elements: dm/dt = k·(S − 1)² for S > 1 from the initial mass m0; covered area fraction 1 − exp(−m/mc). */
export function scaleMass(t, { m0 = 0, k, S, mc }) { const m = m0 + (S > 1 ? k * (S - 1) ** 2 * Math.max(t, 0) : 0); return { m, cover: 1 - Math.exp(-m / Math.max(mc, 1e-9)) }; }

const ISO_DEFAULT = [[0.5, 0.92], [1, 1.5], [2, 2.55], [3, 3.02], [5, 3.9], [8, 4.35], [12, 4.9], [20, 5.2]].map(([C, q]) => ({ C, q }));
const STEP_DEFAULT = [[10, 0.05], [13, 0.06], [16, 0.04], [19, 0.05], [22, 0.06], [25, 0.17], [28, 0.3], [31, 0.4], [34, 0.54]].map(([J, rate]) => ({ J, rate }));
const D_ORG = 5e-10, D_AOC = 6e-10; // m²/s: dissolved organic foulants and assimilable carbon
/** Compact number formatting for notes (significant digits, no locale lookup). */
const fq = (x, sig = 4) => (typeof x !== 'number' ? String(x ?? '–') : !Number.isFinite(x) ? '–' : x === 0 ? '0' : Math.abs(x) >= 1e7 || Math.abs(x) < 1e-4 ? x.toExponential(Math.max(1, sig - 1)) : String(+x.toPrecision(sig)));
/** Extended models of run(): adsorption, combined laws, polarisation coupling, deposition, threshold flux, biofilm transport, channel profile, state space, streaming, learned residual, scaling, dashboard. */
function extendedFouling(v, a, c) {
  const { G, t, npf, cycles, cur, now, trig } = a, K = [], PL = [], TB = [], W = [], BAL = [], out = {}, h = v.hChan * 1e-3, Tm = c.Tm, J = c.Jms, last5 = G.slice(-5), tcfN = med(last5.map((q) => q.tcf)), muEff = MU25 / tcfN;
  // 1 — adsorption isotherms and the adsorptive permeability loss
  const iso = fitIsotherms(v.iso), mtO = massTransfer(v.uCross, h, Tm, D_ORG), betaOrg = Math.exp(Math.min(J / mtO.k, 5)), Cw = Math.max(v.toc, 0) * betaOrg;
  let lam = 0;
  if (iso) {
    const qEq = iso.at(Cw), qRef = iso.best === 'langmuir' ? iso.langmuir.qmax : Math.max(...iso.q), theta = clamp(qEq / qRef, 0, 1), cs = logspace(Math.max(iso.C[0] / 3, 1e-3), iso.C[iso.C.length - 1] * 1.5, 40);
    lam = (clamp(v.adsLoss ?? 8, 0, 95) / 100) * theta;
    PL.push({ type: 'line', title: 'Adsorption isotherm of the organic foulant on the membrane', xlabel: 'Concentration at the membrane (mg/L)', ylabel: 'Adsorbed amount (mg/m²)', logx: true, series: [{ name: 'Measured', x: iso.C, y: iso.q, mode: 'points' }, { name: `Langmuir (R² ${fq(iso.langmuir.r2, 4)})`, x: cs, y: cs.map((x) => langmuir(x, iso.langmuir.qmax, iso.langmuir.K)), dash: iso.best !== 'langmuir' }, { name: `Freundlich (R² ${fq(iso.freundlich.r2, 4)})`, x: cs, y: cs.map((x) => freundlich(x, iso.freundlich.Kf, iso.freundlich.n)), dash: iso.best !== 'freundlich' }], vlines: [{ x: clamp(Cw, cs[0], cs[cs.length - 1]), label: 'at the wall' }] });
    TB.push({ title: 'Adsorption isotherms', columns: ['Model', 'Parameter 1', 'Parameter 2', 'R²', 'RMSE (mg/m²)', 'Adsorbed at the wall concentration (mg/m²)', 'Selected'], rows: [
      ['Langmuir q = q_max·K·C/(1 + K·C)', `q_max = ${fq(iso.langmuir.qmax, 4)} mg/m²`, `K = ${fq(iso.langmuir.K, 4)} L/mg`, iso.langmuir.r2, iso.langmuir.rmse, langmuir(Cw, iso.langmuir.qmax, iso.langmuir.K), iso.best === 'langmuir' ? 'lower RMSE' : ''],
      ['Freundlich q = K_f·C^(1/n)', `K_f = ${fq(iso.freundlich.Kf, 4)}`, `n = ${fq(iso.freundlich.n, 4)}`, iso.freundlich.r2, iso.freundlich.rmse, freundlich(Cw, iso.freundlich.Kf, iso.freundlich.n), iso.best === 'freundlich' ? 'lower RMSE' : '']],
      note: `${iso.n} data points. Organic carbon at the membrane: ${fq(Cw, 3)} mg/L (bulk ${v.toc} mg/L × polarisation factor ${fq(betaOrg, 3)} for a diffusivity of ${D_ORG} m²/s). Surface coverage ${fq(100 * theta, 3)} % → adsorptive permeability loss at equilibrium ${fq(100 * lam, 3)} % (${v.adsLoss ?? 8} % at full coverage); this loss enters the adsorption–pore-blocking law.` });
    K.push({ label: 'Adsorbed organic load at equilibrium', value: qEq, unit: 'mg/m²', help: `${iso.best === 'langmuir' ? 'Langmuir' : 'Freundlich'} isotherm at the wall concentration; coverage ${fq(100 * theta, 3)} %` });
    out.adsorbedLoad = qEq; out.adsorptionModel = iso.best; out.adsorptiveLoss = lam;
  }
  // 2 — combined laws on the first cycle and the pore availability carried into the current cycle
  const eps00 = clamp((v.poreAvail0 ?? 100) / 100, 0.05, 1);
  if (c.hf && c.ht.length >= 12) {
    const cb = fitCombined(c.ht, c.hy, lam), ab = cb.adsBlock, hz = cb.hoZydney, tc = t.slice(cur.a).map((x) => x - cur.t0), obs = npf.slice(cur.a), epsStart = eps00 * Math.min(1, cur.startNPF / Math.max(cycles[0].startNPF, 1e-9));
    const pred = tc.map((x) => cur.startNPF * adsorptionBlocking(x, ab.p[1], ab.p[2], cb.lam)), rm = Math.sqrt(mean(pred.map((x, i) => (x - obs[i]) ** 2))), epsNow = epsStart * Math.exp(-ab.p[2] * a.since);
    PL.push({ type: 'line', title: 'Combined fouling laws on the first cycle (chronological split)', xlabel: 'Time in cycle (d)', ylabel: 'Normalised flux J/J₀', series: [{ name: 'Data', x: c.ht, y: c.hy, mode: 'points' }, { name: `${hz.name} (R² ${fq(hz.r2, 3)})`, x: c.ht, y: c.ht.map(hz.f) }, { name: `${ab.name} (R² ${fq(ab.r2, 3)})`, x: c.ht, y: c.ht.map(ab.f), dash: true }], vlines: [{ x: c.ht[cb.nTrain - 1], label: 'train | validate' }] });
    TB.push({ title: 'Combined fouling laws (first cycle)', columns: ['Law', 'Parameters', 'R² training', 'RMSE validation', 'R² validation', 'Free parameters'], rows: [
      [hz.name, `β = ${fq(hz.p[1], 3)} 1/d, R_p0/R_m = ${fq(hz.p[2], 3)}, cake growth = ${fq(hz.p[3], 3)} 1/d`, hz.r2, hz.rmseVal, hz.r2Val, hz.npar], [ab.name, `k_ads = ${fq(ab.p[1], 3)} 1/d, k_block = ${fq(ab.p[2], 3)} 1/d, adsorptive loss ${fq(100 * cb.lam, 3)} % (from the isotherm)`, ab.r2, ab.rmseVal, ab.r2Val, ab.npar],
      ['Best single Hermia law for comparison', HERMIA[c.hf.best.law].name, c.hf.best.r2, c.hf.best.rmseVal, c.hf.best.r2Val, 2]],
      note: `Ho–Zydney: J/J₀ = e^(−βt) + (1 − e^(−βt))/((1 + R_p0/R_m)·√(1 + g·t)) — open pores are blocked at the rate β while a cake grows over the blocked area. Adsorption–pore blocking: J/J₀ = ε₀·e^(−k_block·t)·[1 − λ(1 − e^(−k_ads·t))]. Carried into the current cycle with the pore availability left by the last cleaning as initial condition (ε₀ = ${fq(100 * epsStart, 4)} % of all pores, entered ${fq(100 * eps00, 4)} % at the start of the log), the adsorption–pore-blocking law predicts the observed normalised flow of that cycle with an RMSE of ${fq(rm, 3)} % of reference.` });
    K.push({ label: 'Open-pore fraction now', value: 100 * epsNow, unit: '%', help: 'Initial pore availability after the last cleaning times exp(−k_block·t) from the adsorption–pore-blocking law' });
    out.poreAvailability = epsNow; out.combinedLawRmse = rm;
  }
  // 3 — concentration polarisation and its coupling to the deposit (cake-enhanced osmotic pressure)
  const Dn = diffusivityNaCl(Tm, 3), mt = massTransfer(v.uCross, h, Tm, Dn), piB = mean(last5.map((q) => q.piF)) * 1e5, dPf = muEff * J * a.Rrev;
  const ce = cakeEnhancedCP({ J, k: mt.k, D: Dn, eps: v.eps, rhoP: v.rhoP, alpha: c.alpha, piB, dPf, mu: muEff }), Cfb = mean(last5.map((q) => q.Cfb));
  TB.push({ title: 'Concentration polarisation and its coupling to the deposit', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [
    ['Reynolds / Schmidt / Sherwood number', `${fq(mt.Re, 3)} / ${fq(mt.Sc, 3)} / ${fq(mt.Sh, 3)}`, '–', 'Sh = 0.065·Re^0.875·Sc^0.25 (spacer-filled channel)'], ['Mass-transfer coefficient k', mt.k * 1e6, 'µm/s', `Hydraulic diameter ${fq(mt.dh * 1000, 3)} mm`], ['Polarisation factor β = exp(J/k), clean membrane', ce.beta, '–', ce.beta > 1.2 ? 'Above the usual design limit of 1.2' : 'Within the usual design limit of 1.2'],
    ['Polarisation factor with the deposit, β*', ce.betaStar, '–', 'Back-diffusion is hindered inside the deposit (tortuosity 1 − ln ε²)'], ['Salt concentration at the membrane', (Cfb * ce.betaStar) / 1000, 'g/L', `Feed–brine average ${fq(Cfb / 1000, 3)} g/L`], ['Osmotic pressure at the membrane', (piB * ce.betaStar) / 1e5, 'bar', `Bulk ${fq(piB / 1e5, 3)} bar`],
    ['Fouling pressure loss explained', dPf / 1e5, 'bar', 'μ·J·R_reversible from the Darcy analysis'], ['… hydraulic resistance of the deposit', ce.hydraulic / 1e5, 'bar', `${fq(100 * (1 - ce.shareOsmotic), 3)} %`], ['… cake-enhanced osmotic pressure', ce.osmotic / 1e5, 'bar', `${fq(100 * ce.shareOsmotic, 3)} %`],
    ['Deposit loading with the osmotic coupling', ce.m * 1000, 'g/m²', `${fq(ce.mHydraulicOnly * 1000, 3)} g/m² if all the loss were hydraulic`], ['Deposit thickness', ce.delta * 1e6, 'µm', 'δ = m / (ρ(1 − ε))']],
    note: 'The deposit mass is solved so that its hydraulic loss plus the extra osmotic pressure it causes add up to the observed fouling pressure loss.' });
  K.push({ label: 'Polarisation factor β', value: ce.beta, unit: '–', status: ce.beta > 1.2 ? 'warn' : 'ok', help: `Film theory with the Schock–Miquel correlation; ${fq(ce.betaStar, 4)} with the present deposit` }, { label: 'Fouling loss due to cake-enhanced osmotic pressure', value: 100 * ce.shareOsmotic, unit: '%', help: 'Share of the reversible fouling pressure loss that is osmotic rather than hydraulic' });
  if (ce.beta > 1.2) W.push({ level: 'warn', msg: `The concentration-polarisation factor is ${fq(ce.beta, 3)} (limit 1.2): the membrane wall sees ${fq(100 * (ce.beta - 1), 3)} % more salt than the bulk — lower the flux or raise the cross-flow.` });
  Object.assign(out, { cpFactor: ce.beta, cpFactorFouled: ce.betaStar, osmoticShareOfFouling: ce.shareOsmotic });
  // 4 — deposition and detachment: boundary condition at the membrane and its fit to the current cycle
  const cyD = cur.b - cur.a >= 8 ? cur : cycles[0], tcd = t.slice(cyD.a, cyD.b).map((x) => x - cyD.t0), base = cyD === cycles[0] ? a.Rm : a.Rm + a.Rirr, mObs = a.Rt.slice(cyD.a, cyD.b).map((R) => Math.max(0, R - base) / c.alpha), cP = v.phiB * v.rhoP, tauW = viscosity(Tm) * c.cf.shearRate;
  const aIn = v.omega * cP * Math.max(0, J - c.cf.J) * 86400, kIn = Math.max(v.kDet ?? 0.05, 0) * tauW, fd = fitDeposition(tcd, mObs, mObs.length ? Math.min(...mObs.slice(0, 3)) : 0);
  if (fd) {
    const tl = linspace(0, Math.max(tcd[tcd.length - 1], 1), 40), m0 = Math.min(...mObs.slice(0, 3));
    PL.push({ type: 'line', title: 'Deposit on the membrane: deposition and detachment', xlabel: 'Time in cycle (d)', ylabel: 'Deposit loading (g/m²)', series: [{ name: 'Inferred from the fouling resistance', x: tcd, y: mObs.map((x) => x * 1000), mode: 'points' }, { name: 'Deposition–detachment model, fitted', x: tl, y: tl.map((x) => depositMass(x, fd.a, fd.k, m0) * 1000) }, { name: 'Same model with the entered attachment and detachment parameters', x: tl, y: tl.map((x) => depositMass(x, aIn, kIn, m0) * 1000), dash: true }] });
    TB.push({ title: 'Deposition and detachment at the membrane', columns: ['Quantity', 'Fitted to the cycle', 'From the entered parameters', 'Unit', 'Note'], rows: [
      ['Deposition flux to the membrane', fd.a * 1000, aIn * 1000, 'g/m²·d', 'Boundary condition: ω·c·(J − J_crit)⁺'], ['Attachment efficiency ω', cP * J > 0 ? fd.a / (cP * J * 86400) : 0, v.omega, '–', `Foulant concentration ${fq(cP * 1000, 3)} g/m³, flux ${fq(J * 3.6e6, 3)} L/m²·h, critical flux ${fq(c.cf.J * 3.6e6, 3)} L/m²·h`],
      ['Detachment rate constant k = k_det·τ_w', fd.k, kIn, '1/d', `Wall shear stress ${fq(tauW, 3)} Pa`], ['Detachment coefficient k_det', tauW > 0 ? fd.k / tauW : 0, v.kDet ?? 0.05, '1/(Pa·d)', 'Detachment flux = k_det·τ_w·m'], ['Steady-state deposit a/k', fd.mss * 1000, kIn > 0 ? (aIn / kIn) * 1000 : null, 'g/m²', 'Where deposition and detachment balance'], ['Time constant 1/k', 1 / fd.k, kIn > 0 ? 1 / kIn : null, 'd', `R² of the fit ${fq(fd.r2, 3)}`]],
      note: `dm/dt = ω·c·(J − J_crit)⁺ − k_det·τ_w·m, fitted to cycle ${cycles.indexOf(cyD) + 1}. ${aIn <= 0 ? 'With the entered particle size the operating flux is below the critical flux, so the entered parameters predict no net deposition; the fitted flux shows what the resistance history requires.' : ''}` });
    K.push({ label: 'Deposition flux to the membrane', value: fd.a * 1000, unit: 'g/m²·d', help: `Deposition–detachment model fitted to the cycle; detachment time constant ${fq(1 / fd.k, 3)} d` });
    Object.assign(out, { depositionFlux: fd.a, detachmentRate: fd.k });
  }
  // 5 — threshold flux from the flux-stepping test
  const th = thresholdFlux(v.fluxStep);
  if (th) {
    const js = linspace(th.J[0], th.J[th.J.length - 1], 50);
    PL.push({ type: 'line', title: 'Threshold flux from the flux-stepping test', xlabel: 'Flux (L/m²·h)', ylabel: 'Fouling rate (% of resistance per day)', series: [{ name: 'Flux-step data', x: th.J, y: th.rate, mode: 'points' }, { name: 'Threshold-flux relation', x: js, y: js.map(th.rateAt) }], vlines: [{ x: th.Jth, label: 'threshold' }, { x: clamp(c.fluxNow, th.J[0], th.J[th.J.length - 1]), label: 'operating' }] });
    TB.push({ title: 'Threshold flux', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [['Threshold flux J_th', th.Jth, 'L/m²·h', th.significant ? 'Clear break in the fouling rate' : 'No clear break: the data are close to a straight line'], ['Fouling rate below the threshold', th.a, '%/d', 'Low, flux-independent rate'], ['Rise above the threshold', th.b, '%/d per L/m²·h', 'rate = a + b·(J − J_th)'], ['Operating flux', c.fluxNow, 'L/m²·h', c.fluxNow > th.Jth ? 'Above the threshold flux' : 'Below the threshold flux'], ['Fouling rate expected at the operating flux', th.rateAt(c.fluxNow), '%/d', `Observed permeability loss ${fq(c.foulRate, 2)} %/d`], ['R² of the relation', th.r2, '–', `${th.J.length} flux steps`]] });
    K.push({ label: 'Threshold flux', value: th.Jth, unit: 'L/m²·h', status: c.fluxNow > th.Jth ? 'warn' : 'ok', help: 'Flux above which the fouling rate rises sharply (hinge fit of the flux-stepping test)' });
    if (c.fluxNow > th.Jth) W.push({ level: 'warn', msg: `The operating flux ${fq(c.fluxNow, 3)} L/m²·h is above the threshold flux ${fq(th.Jth, 3)} L/m²·h found in the flux-stepping test: expect the fouling rate to rise to about ${fq(th.rateAt(c.fluxNow), 2)} %/d.` });
    out.thresholdFluxLMH = th.Jth;
  }
  // 6 — biofilm growth limited by substrate transport (zero flux of substrate at the membrane)
  const mtA = massTransfer(v.uCross, h, Tm, D_AOC), Xf = Math.max(v.bioXf ?? 30, 1) * 1000, bp = { Sb: Math.max(v.aoc, 1e-6) * 1e-3, kL: mtA.k, Df: 0.8 * D_AOC, mumax: v.mumax / 86400, Ks: v.KsAoc * 1e-3, Y: clamp(v.bioY ?? 0.5, 0.05, 1), Xf, kd: v.kd / 86400, Lf0: c.Xnow / Xf, Lmax: (2 * v.Xmax) / Xf, nz: v.nzBio ?? 20 };
  const bt = biofilmTransport(bp, v.horizon, Math.min(Math.max(v.nStep, 8), 240)), btDp = bt.X.map((x) => c.d1now + v.betaBio * (x - c.Xnow)), ib = btDp.findIndex((x) => x >= c.limDp), btDays = c.d1now >= c.limDp ? 0 : ib > 0 ? bt.t[ib - 1] + ((c.limDp - btDp[ib - 1]) * (bt.t[ib] - bt.t[ib - 1])) / (btDp[ib] - btDp[ib - 1]) : v.horizon;
  const s0 = biofilmSubstrate({ ...bp, Lf: Math.max(bp.Lf0, 1e-9) }), feedSupply = (mean(last5.map((q) => q.row.Qf)) * 24 * bp.Sb) / a.area;
  PL.push({ type: 'line', title: 'Biofilm growth with substrate transport versus the well-fed Monod model', xlabel: 'Days from now', ylabel: 'Biomass (g/m²)', series: [{ name: 'Growth limited by substrate transport', x: bt.t, y: bt.X }, { name: 'Monod growth at the bulk concentration', x: c.bio.t, y: c.bio.X, dash: true }], note: 'The well-fed model assumes every cell sees the bulk substrate concentration; with film and biofilm diffusion the supply of assimilable carbon limits growth.' });
  TB.push({ title: 'Biofilm growth and substrate transport', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [
    ['Biofilm thickness now → at the horizon', `${fq(bp.Lf0 * 1e6, 3)} → ${fq(bt.Lf[bt.Lf.length - 1] * 1e6, 3)}`, 'µm', `Biomass density ${v.bioXf ?? 30} kg/m³`], ['Biomass now → at the horizon', `${fq(c.Xnow, 3)} → ${fq(bt.X[bt.X.length - 1], 3)}`, 'g/m²', `Well-fed Monod model: ${fq(c.bio.X[c.bio.X.length - 1], 3)} g/m²`],
    ['Substrate at the biofilm surface / at the membrane', `${fq(s0.Ssurf * 1000, 3)} / ${fq(s0.Swall * 1000, 3)}`, 'µg/L', `Bulk ${v.aoc} µg/L; zero flux at the membrane`], ['Thiele modulus', s0.thiele, '–', s0.thiele > 1 ? 'Diffusion-limited inside the film' : 'Reaction-limited inside the film'], ['Effectiveness factor', s0.eta, '–', 'Actual uptake / uptake at the bulk concentration'],
    ['Substrate flux into the biofilm', s0.flux * 86400 * 1000, 'mg/m²·d', `Film-transfer limit k_L·S = ${fq(bt.supplyLimit * 86400 * 1000, 3)} mg/m²·d`], ['Substrate carried by the feed', feedSupply * 1000, 'mg/m²·d', 'Feed flow × assimilable carbon / membrane area: the ceiling for the whole train'], ['Days to the ΔP trigger', btDays >= v.horizon ? `> ${v.horizon}` : btDays, 'd', `Well-fed model: ${c.bioDays >= v.horizon ? `> ${v.horizon}` : fq(c.bioDays, 3)} d`]],
    note: `Steady diffusion–reaction in the biofilm (${bp.nz} cells, Newton iteration) coupled to the film transfer coefficient ${fq(mtA.k * 1e6, 3)} µm/s; thickness integrated with ${bt.t.length - 1} Heun steps.` });
  K.push({ label: 'Biofilm effectiveness factor', value: s0.eta, unit: '–', status: 'ok', help: 'Share of the well-fed Monod uptake that substrate transport allows' });
  Object.assign(out, { biofilmEffectiveness: s0.eta, bioTransportDaysToTrigger: btDays });
  // 7 — deposition profile along the feed channel with the outlet pressure prescribed
  if (v.chanOn !== false) {
    const PoutBar = med(last5.map((q) => q.row.Pc)), PpBar = med(last5.map((q) => q.row.Pp)), piIn = mean(last5.map((q) => piAstm(q.tdsF, q.row.T))), PfMeas = med(last5.map((q) => q.row.Pf)), Ymeas = med(last5.map((q) => q.Y)), Qf = med(last5.map((q) => q.row.Qf)) / 3600;
    // equivalent channel: total membrane area on two walls, split over the stages in the ratio of their vessel numbers
    const ratio = Math.max(v.stageRatio ?? 2, 1), nSt = a.stageOK ? 2 : 1, widths = nSt === 2 ? [(a.area * ratio) / (ratio + 1) / (2 * v.lChan), a.area / (ratio + 1) / (2 * v.lChan)] : [a.area / (2 * v.lChan)];
    const ch = channelFouling({ L: v.lChan, widths, h, Q0: Qf, T: Tm, mu: muEff, Rm: a.Rm + a.Rirr, alpha: c.alpha, Pp: PpBar * 1e5, Pout: PoutBar * 1e5, dPtarget: Math.max(PfMeas - PoutBar, 0.05) * 1e5, pi0: piIn * 1e5, betaCP: ce.beta, dp: v.dpNm * 1e-9, phiB: v.phiB, omega: v.omega, cp: cP, kDet: Math.max(v.kDet ?? 0.05, 0), days: Math.max(a.since, 0), nT: 12, N: v.nChan ?? 40 });
    const n = ch.x.length, lead = mean(ch.m.slice(0, Math.ceil(n / 4))), tail = mean(ch.m.slice(-Math.ceil(n / 4)));
    PL.push({ type: 'line', title: 'Channel model: flux, critical flux and deposit from inlet to outlet', xlabel: 'Distance from the feed inlet (m)', ylabel: 'Flux (L/m²·h) · deposit (g/m²)', series: [{ name: 'Local flux (L/m²·h)', x: ch.x, y: ch.J.map((q) => q * 3.6e6) }, { name: 'Local critical flux (L/m²·h)', x: ch.x, y: ch.Jc.map((q) => Math.min(q * 3.6e6, 500)), dash: true }, { name: 'Deposit (g/m²)', x: ch.x, y: ch.m.map((q) => q * 1000) }], note: 'Deposition occurs where the local flux exceeds the local critical flux; cross-flow and wall shear fall along the channel as water permeates.' });
    TB.push({ title: 'Channel model with prescribed outlet pressure', columns: ['Quantity', 'Model', 'Measured', 'Unit', 'Note'], rows: [
      ['Outlet (concentrate) pressure', ch.Pout / 1e5, PoutBar, 'bar', 'Boundary condition'], ['Inlet (feed) pressure, clean channel', ch.first.Pin / 1e5, PfMeas, 'bar', `Found by shooting from the outlet condition; friction scaled ×${fq(ch.fMult, 3)} to the measured pressure drop`], ['Inlet (feed) pressure with the deposit', ch.Pin / 1e5, PfMeas, 'bar', 'Same friction factor'], ['Recovery', 100 * ch.recovery, 100 * Ymeas, '%', `${ch.nStages} stage${ch.nStages > 1 ? `s, vessel ratio ${ratio} : 1` : ''}; an independent check of membrane resistance and osmotic pressure`], ['Inlet cross-flow velocity', ch.u[0], v.uCross, 'm/s', 'Model: feed flow / open channel area; right: value entered for the critical-flux estimate'],
      ['Flux at inlet → outlet', `${fq(ch.J[0] * 3.6e6, 3)} → ${fq(ch.J[n - 1] * 3.6e6, 3)}`, c.fluxNow, 'L/m²·h', 'Measured value is the train average'], ['Wall shear stress inlet → outlet', `${fq(ch.tau[0], 3)} → ${fq(ch.tau[n - 1], 3)}`, null, 'Pa', 'Wall-shear boundary condition for detachment'],
      ['Deposit in the first / last quarter', `${fq(lead * 1000, 3)} / ${fq(tail * 1000, 3)}`, null, 'g/m²', lead > tail ? 'Lead-end deposition (high flux)' : tail > 0 ? 'Tail-end deposition (low shear)' : 'No deposition above the critical flux']],
      note: `${n} cells; spacer friction f = ${fq(ch.fMult, 3)} × 6.23·Re^−0.3 (the factor absorbs fittings, interconnectors and spacer fouling); membrane resistance and polarisation factor from this analysis; deposit integrated over the ${fq(a.since, 3)} days of the current cycle in 12 quasi-steady steps. A reduced, one-dimensional form of a flow simulation: the cross-channel profile is represented by the film coefficient.` });
    BAL.push({ name: 'Channel model: feed flow = concentrate + permeate (scaled)', in: 1, out: (ch.Qout + ch.perm) / Qf });
    if (Math.abs(ch.recovery - Ymeas) > 0.07) W.push({ level: 'info', msg: `At the logged pressures the channel model gives ${fq(100 * ch.recovery, 3)} % recovery against ${fq(100 * Ymeas, 3)} % measured: ${ch.J[n - 1] < 0.1 * ch.J[0] ? 'the tail of the train has almost no net driving pressure once the local osmotic pressure and polarisation are resolved — check the concentrate pressure and conductivity readings, or the last elements contribute little' : 'check the membrane area, the channel dimensions and the pressure readings'}.` });
    K.push({ label: 'Channel model: feed pressure', value: ch.Pin / 1e5, unit: 'bar', help: `From the prescribed outlet pressure ${fq(PoutBar, 4)} bar; measured ${fq(PfMeas, 4)} bar` });
    Object.assign(out, { channelFeedPressureBar: ch.Pin / 1e5, channelRecovery: ch.recovery, channelDepositLead: lead, channelDepositTail: tail });
  }
  // 8 — state-space model of the normalised flow with a Kalman filter, restarted at every cleaning
  const kfL = [], kfS = [];
  let kf = null;
  for (const cy of cycles) {
    const tt = t.slice(cy.a, cy.b), yy = npf.slice(cy.a, cy.b), dif = yy.slice(1).map((x, i) => x - yy[i]), r = Math.max(mad(dif) / Math.SQRT2, 0.15) ** 2, f = tt.length >= 4 ? kalmanAuto(tt, yy, r) : null;
    if (f) { kfL.push(...f.level); kfS.push(...f.sdL); if (cy === cur) kf = { ...f, r, tt, yy }; } else { kfL.push(...yy); kfS.push(...yy.map(() => 0)); }
  }
  if (kf) {
    const n = kf.level.length, lv = kf.level[n - 1], sl = kf.slope[n - 1], sd = kf.sdS[n - 1], lim = trig.npf.limit, dz = (s) => (lv <= lim ? 0 : s < -1e-9 ? Math.min(3650, (lv - lim) / -s) : 3650), kd = dz(sl), kLo = dz(sl - 2 * sd), kHi = dz(sl + 2 * sd), nAn = kf.z.filter((z) => Math.abs(z) > 3.5).length, ds = (x) => (x >= 3650 ? '> 3650' : fq(x, 3));
    PL.push({ type: 'line', title: 'State-space estimate of the normalised permeate flow (Kalman filter)', xlabel: 'Time (d)', ylabel: '% of reference', series: [{ name: 'Normalised permeate flow', x: t, y: npf, mode: 'points' }, { name: 'Filtered level', x: t, y: kfL }, { name: 'Level + 2σ', x: t, y: kfL.map((x, i) => x + 2 * kfS[i]), dash: true }, { name: 'Level − 2σ', x: t, y: kfL.map((x, i) => x - 2 * kfS[i]), dash: true }], hlines: [{ y: lim, label: 'cleaning trigger' }], note: 'The filter is restarted at every detected cleaning (reset of the state).' });
    TB.push({ title: 'State-space model (local linear trend, Kalman filter)', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [
      ['Filtered level now', lv, '% of reference', `± ${fq(2 * kf.sdL[n - 1], 2)} (2σ)`], ['Filtered slope now', sl, '%/d', `± ${fq(2 * sd, 2)} (2σ); Theil–Sen trend ${fq(cur.npf.slope, 3)} %/d`], ['Days to the flow trigger', ds(kd), 'd', `Band ${ds(kLo)}–${ds(kHi)} d from the slope uncertainty; trend extrapolation ${ds(trig.npf.days)} d`],
      ['Measurement noise (1σ)', Math.sqrt(kf.r), '% of reference', 'From the first differences of the cycle'], ['Slope process noise', kf.qS, '(%/d)² per d', 'Chosen by maximum likelihood of the innovations'], ['Kalman gain (level / slope)', `${fq(kf.gain[0], 3)} / ${fq(kf.gain[1], 3)}`, '', 'At the last record'], ['Innovations beyond 3.5σ', nAn, 'records', 'One-step-ahead prediction errors flagged as anomalies']],
      note: 'State x = [level, slope]: level(t + Δt) = level + slope·Δt, slope(t + Δt) = slope + w, measurement = level + v.' });
    K.push({ label: 'Days to cleaning (state-space estimate)', value: kd >= 3650 ? '> 3650' : kd, unit: 'd', help: `Kalman-filtered level and slope of the normalised flow; band ${ds(kLo)}–${ds(kHi)} d` });
    Object.assign(out, { kalmanLevel: lv, kalmanSlope: sl, kalmanDaysToCleaning: kd });
  }
  // 9 — streaming replay: the records are fed one at a time, as a data-acquisition system would deliver them
  const mon = createMonitor(v), rows = (Array.isArray(v.log) ? v.log : []).filter((r) => r && isNum(r.t)).slice().sort((p, q) => p.t - q.t);
  for (const r of rows) mon.push(r);
  const ms = mon.state, show = ms.alerts.filter((x) => x.type !== 'rejected'), lastAlarm = ms.alerts.filter((x) => x.type === 'alarm').pop();
  TB.push({ title: 'Streaming replay of the log (record-by-record processing)', columns: ['Day', 'Event', 'Message'], rows: show.length ? show.slice(-40).map((x) => [x.t, { baseline: 'Baseline', alarm: 'Control-chart alarm', trigger: 'Cleaning trigger', cleaning: 'Cleaning detected', step: 'Sudden loss' }[x.type] || x.type, x.msg]) : [['–', 'No event', 'No alert was raised']],
    note: `${ms.n} records were pushed one at a time; ${ms.accepted} accepted, ${ms.rejected} rejected by the on-line validation. Each record updates the normalisation, the Kalman state and the EWMA/CUSUM charts with a fixed amount of work, so the same routine can follow a live data feed: append new rows to the operating log and run again.${ms.last ? ` State after the last record: level ${fq(ms.last.level, 4)} % of reference, slope ${fq(ms.last.slope, 3)} %/d.` : ''}` });
  K.push({ label: 'Streaming monitor: records / alerts', value: `${ms.accepted} / ${show.length}`, help: 'Record-by-record processing of the log with on-line validation, Kalman update and control charts' });
  Object.assign(out, { streamAccepted: ms.accepted, streamRejected: ms.rejected, streamAlerts: show.length, streamCleanings: ms.resets, streamLastAlarmDay: lastAlarm ? lastAlarm.t : -1 });
  // 10 — mechanistic trend with a learned residual, tested out of sample
  if (c.hf && c.ht.length >= 16) {
    const cy = cycles[0], bl = c.hf.best, mech = c.ht.map((x) => 100 * bl.J0 * hermia(bl.law, bl.k * x)), names = ['Temperature', 'Feed conductivity', 'Recovery', 'Flux', 'Normalised ΔP'], X = G.slice(cy.a, cy.b).map((q) => [q.row.T, q.row.Cf, 100 * q.Y, q.flux, q.dpn]), y = npf.slice(cy.a, cy.b);
    const hy = hybridResidual(X, y, mech, c.hf.nTrain, names), gain = hy.rmseMechTest > 0 ? 1 - hy.rmseHybTest / hy.rmseMechTest : 0;
    PL.push({ type: 'line', title: 'Mechanistic law with a learned residual: out-of-sample test', xlabel: 'Time in cycle (d)', ylabel: 'Normalised permeate flow (% of reference)', series: [{ name: 'Measured', x: c.ht, y, mode: 'points' }, { name: `Mechanistic (${HERMIA[bl.law].name.toLowerCase()})`, x: c.ht, y: mech }, { name: 'Mechanistic + learned residual', x: c.ht, y: hy.hybrid, dash: true }], vlines: [{ x: c.ht[hy.nTrain - 1], label: 'train | test' }] });
    TB.push({ title: 'Mechanistic model with a learned residual (ridge regression)', columns: ['Quantity', 'Mechanistic only', 'With learned residual', 'Unit'], rows: [['RMSE on the training part', hy.rmseMechTrain, hy.rmseHybTrain, '% of reference'], ['RMSE on the later, unseen part', hy.rmseMechTest, hy.rmseHybTest, '% of reference'], ...names.map((nm, j) => [`Weight of ${nm.toLowerCase()} (standardised)`, null, hy.weights[j], '% per σ']), ['Constant offset', null, hy.bias, '% of reference']],
      note: `Residual of the best blocking law regressed on five operating variables with a ridge penalty (${fq(hy.lam, 3)}, chosen by leave-one-out cross-validation; LOO error ${fq(hy.looRmse, 3)} %). Trained on the first ${hy.nTrain} records and tested on the following ${hy.nTest}. ${gain > 0.05 ? `The learned residual lowers the out-of-sample error by ${fq(100 * gain, 3)} %.` : 'The learned residual does not improve the out-of-sample error: the normalisation already removes the influence of these variables, and what remains is a change of mechanism that the training period did not contain.'}` });
    K.push({ label: 'Out-of-sample error, mechanistic / with learned residual', value: `${fq(hy.rmseMechTest, 3)} / ${fq(hy.rmseHybTest, 3)}`, unit: '% of ref.', help: 'Root-mean-square error on the later part of the first cycle, not used for training' });
    Object.assign(out, { hybridRmseTest: hy.rmseHybTest, mechanisticRmseTest: hy.rmseMechTest });
  }
  // 11 — surface scaling on the tail elements, with the polarisation raised by the deposit
  const Ssc = 10 ** ((v.siMargin + 2 * Math.log10(ce.betaStar / ce.beta)) / 2), m0s = cycles.length > 1 ? 0 : Math.max(v.ms0 ?? 0, 0), sc = scaleMass(a.since, { m0: m0s, k: Math.max(v.kScale ?? 0.2, 0), S: Ssc, mc: Math.max(v.mCover ?? 10, 0.1) }), tailS = clamp((v.tailShare ?? 20) / 100, 0, 1);
  TB.push({ title: 'Surface scaling on the tail elements', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [['Supersaturation ratio beyond the antiscalant limit', Ssc, '–', `Scaling margin ${v.siMargin} SI plus ${fq(2 * Math.log10(ce.betaStar / ce.beta), 2)} SI from the deposit-enhanced polarisation`], ['Initial scaling mass of this cycle', m0s, 'g/m²', cycles.length > 1 ? 'Reset by the cleaning' : 'Entered initial condition'], ['Scale mass now', sc.m, 'g/m²', Ssc > 1 ? `Growth ${fq((v.kScale ?? 0.2) * (Ssc - 1) ** 2, 3)} g/m²·d` : 'No growth: within the antiscalant limit'], ['Membrane area covered on the tail elements', 100 * sc.cover, '%', `1 − exp(−m/m_c), m_c = ${v.mCover ?? 10} g/m²`], ['Flow loss of the train from scaling', 100 * sc.cover * tailS, '% of reference', `Tail elements carry ${v.tailShare ?? 20} % of the area`]] });
  K.push({ label: 'Scale mass on the tail elements (model)', value: sc.m, unit: 'g/m²', status: sc.cover > 0.2 ? 'warn' : 'ok', help: 'Second-order surface growth beyond the antiscalant limit, from the initial scaling mass of the cycle' });
  Object.assign(out, { scaleMass: sc.m, scaleCoverage: sc.cover });
  // 12 — dashboard: one line per indicator and an overall health index
  const left = (q, pct, up) => clamp((100 * q.margin) / Math.max(((q.limit / (1 + (up ? pct : -pct) / 100)) * pct) / 100, 1e-9), 0, 100), sFlow = left(trig.npf, v.trigNPF, false), sDp = left(trig.ndp, v.trigDP, true), sSp = a.integrity ? 0 : left(trig.nsp, v.trigSP, true);
  const sData = (100 * G.length) / Math.max(a.rows.length + a.issues.filter((x) => /Missing|Range/.test(x[1])).length, 1), sLife = clamp((100 * a.rul) / Math.max(a.rul + a.age, 1e-9), 0, 100);
  const sc5 = [['Normalised permeate flow', now.npf, trig.npf.limit, sFlow], ['Normalised differential pressure', now.ndp, trig.ndp.limit, sDp], ['Normalised salt passage', now.nsp, trig.nsp.limit, sSp], ['Concentration polarisation β', ce.beta, 1.2, clamp((100 * (1.2 - ce.beta)) / 0.2, 0, 100)], ['Remaining membrane life (years)', a.rul, 0, sLife], ['Usable records (%)', sData, 100, clamp(sData, 0, 100)]].filter((q, i) => i !== 3);
  const cpRow = ['Concentration polarisation β (not in the index)', ce.beta, 1.2, null, ce.beta > 1.2 ? 'Act' : ce.beta > 1.15 ? 'Watch' : 'OK'];
  const health = 0.5 * Math.min(...sc5.map((q) => q[3])) + 0.5 * mean(sc5.map((q) => q[3])), stat = (s) => (s >= 60 ? 'OK' : s >= 25 ? 'Watch' : 'Act');
  TB.push({ title: 'Monitoring dashboard', columns: ['Indicator', 'Now', 'Limit', 'Score (0–100)', 'Status'], rows: [...sc5.map((q) => [q[0], q[1], q[2], q[3], stat(q[3])]), cpRow, ['Membrane health index', null, null, health, stat(health)]], note: 'Score = share of the allowed margin still unused (100 = at reference, 0 = at the limit). The health index is the mean of the scores and the worst score in equal parts, so one failing indicator cannot hide behind good ones.' });
  K.push({ label: 'Membrane health index', value: health, unit: '/100', status: health >= 60 ? 'ok' : health >= 25 ? 'warn' : 'bad', help: 'Dashboard score combining flow, pressure drop, salt passage, remaining life and data quality' });
  out.healthIndex = health;
  return { K, PL, TB, W, BAL, out };
}

const D = () => Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const modelPars = (v) => ({ area: v.area, Aclean: v.Aclean, kFoul: v.kFoul, law: v.law, dp0: v.dp0, kDp: v.kDp, qRef: v.qRef, mFlow: v.mFlow, sp0: v.sp0, Pp: v.PpRef });

const suite = {
  id: 'fouling', num: 10, title: 'Fouling & Membrane-Performance Monitoring', short: 'Fouling monitor', icon: '📈',
  tagline: 'Normalise operating data, separate real deterioration from operating changes, diagnose the foulant and forecast cleaning and membrane life.',
  description: 'Reads an operating log (pressures, flows, conductivities, temperature), validates it and normalises permeate flow, salt passage and differential pressure to reference conditions, so that temperature, salinity and set-point changes no longer hide or mimic fouling. Darcy resistances, blocking-law and combined cake/adsorption–pore-blocking fits, adsorption isotherms, a deposition–detachment law, threshold- and critical-flux relations, concentration polarisation with its coupling to the deposit, a channel deposition profile and biofilm growth with substrate transport interpret the decline; a Kalman-filtered state-space model, a record-by-record streaming monitor and a mechanistic model with a learned residual complement the statistics; robust trends, EWMA/CUSUM charts and step detection separate gradual from sudden changes; evidence rules score the likely foulant; and the trends are extrapolated to the cleaning triggers and to end of membrane life with uncertainty bands.',
  guide: [
    'Paste or import the operating log on the Inputs tab (one row per day or shift). The built-in 180-day example contains gradual fouling, a scaling episode, a cleaning and an O-ring failure.',
    'Enter the membrane area and choose the reference: the first days of the log (clean baseline) or the design point.',
    'Add what you know about the feed: SDI, MFI, organic carbon, scaling margin from suite 2.',
    'Run. Check the data-validation table first, then the three normalised trends, the detected events, the diagnosis and the forecast.',
  ],
  referenceOnly: ['real-time operating-data acquisition'],
  implemented: ['resistance-in-series', 'darcy membrane', 'cake-filtration', 'hermia complete-blocking', 'hermia standard-blocking', 'hermia intermediate-blocking', 'hermia cake-filtration', 'pore-blocking equations', 'cake-compressibility', 'kozeny–carman', 'deposition-rate', 'critical-flux', 'concentration-polarization equation', 'normalized permeate-flow', 'normalized salt-passage', 'normalized pressure-drop',
    'cake–pore-blocking', 'fouling–scaling', 'resistance–compressibility', 'mechanistic–statistical monitoring', 'remaining-useful-life/prognostic',
    'adsorption equations', 'langmuir isotherm', 'freundlich isotherm', 'detachment-rate', 'threshold-flux', 'fouling–concentration-polarization', 'adsorption–pore-blocking', 'biofilm-growth–transport', 'deposition–detachment', 'cfd–fouling', 'membrane-performance–state-space', 'mechanistic–machine-learning',
    'initial pore availability', 'initial scaling mass', 'membrane deposition-flux', 'foulant attachment/detachment', 'zero-flux wall', 'outlet-pressure condition', 'real-time operating-data acquisition', 'concentration-polarisation assessment', 'dashboards',
    'clean-membrane resistance', 'initial permeability', 'initial deposit thickness', 'initial biofilm biomass', 'baseline normalized performance', 'inlet foulant concentration', 'permeate-flux boundary', 'transmembrane-pressure boundary', 'wall shear condition', 'cleaning/reset',
    'data validation and cleaning', 'membrane-performance normalisation', 'permeate-flow monitoring', 'salt-passage and rejection monitoring', 'pressure-drop monitoring', 'flux monitoring', 'organic-fouling assessment', 'biological-fouling assessment', 'colloidal-fouling assessment', 'inorganic-scaling assessment', 'membrane-ageing analysis', 'fouling-resistance modelling', 'cleaning-in-place monitoring', 'cleaning-effectiveness assessment', 'anomaly detection', 'trend analysis', 'membrane-health indicators', 'fault diagnosis', 'remaining-useful-life prediction', 'predictive maintenance', 'alarms', 'historical performance comparison'],
  equationsNote: 'Normalisation follows the ASTM D4516 approach for one train: log-mean feed–brine concentration, osmotic pressure from TDS, an Arrhenius temperature-correction factor and a flow-exponent correction of the pressure drop. It assumes steady operating points; start-ups, flushing periods and rows with large recovery changes should be removed. Conductivity is converted to TDS with a generic correlation — a site-specific factor improves salt-passage accuracy. Foulant scores are evidence rules, not probabilities from a trained classifier, and should be confirmed by autopsy or targeted water analysis. Blocking laws are fitted to the normalised (constant-pressure-equivalent) flux decline. Forecasts extrapolate the current robust trend; they do not anticipate operating changes. Stage-resolved diagnosis needs the interstage pressure column. The isotherm and threshold-flux relations are fitted to the laboratory rows entered on the Inputs tab (the built-in rows are examples, not properties of your foulant). Polarisation uses film theory with the Schock–Miquel correlation for one average channel; the cake-enhanced osmotic pressure follows Hoek and Elimelech with an assumed deposit porosity. The deposition–detachment law lumps the whole membrane area into one deposit mass. The channel model is one-dimensional (feed to concentrate, one equivalent channel for all stages) and represents the cross-channel profile by the film coefficient — a reduced-order form of a flow simulation, not a resolved spacer geometry (suite 4). The biofilm model solves steady diffusion–reaction across the film with a zero-flux membrane but treats the film as uniform along the channel. The state-space model is a local linear trend with Gaussian noise; the learned residual is a linear ridge regression and can only capture effects present in its training period. The streaming monitor processes one record at a time from the log table; connecting it to a plant historian is outside this application. Scale growth is a screening law driven by the entered scaling margin.',

  inputs: [
    { group: 'Operating log', help: 'One row per logged operating point, in chronological order. Rows with missing or implausible values are reported and skipped.', fields: [
      { key: 'log', label: 'Operating data', type: 'table', columns: COLS, get value() { return defaultLog(); }, help: 'Interstage pressure is optional; with it, first-stage and last-stage pressure drops are separated, which is what distinguishes particulate/biological fouling from scaling.' },
    ] },
    { group: 'Membrane system', fields: [
      { key: 'area', label: 'Total membrane area', unit: 'm²', value: 8035, min: 1, max: 5e6, help: 'Used for flux, permeability and Darcy resistance (not needed for the normalised ratios).' },
      { key: 'stage1Share', label: 'Share of permeate from the first stage', unit: '%', value: 65, min: 30, max: 90, help: 'Used to estimate the average feed–brine flow of each stage for the stage pressure-drop normalisation.' },
      { key: 'age0', label: 'Membrane age at the start of the log', unit: 'years', value: 0, min: 0, max: 20 },
    ] },
    { group: 'Feed-water fouling indicators', help: 'Optional measurements that sharpen the diagnosis.', fields: [
      { key: 'sdi', label: 'Silt density index SDI₁₅', unit: '%/min', value: 3.2, min: 0, max: 6.7, help: '< 3 good, 3–5 marginal, > 5 unacceptable for spiral-wound RO.' },
      { key: 'mfi', label: 'Modified fouling index MFI-UF', unit: 's/L²', value: 1500, min: 0, max: 1e5, help: 'Measured through an ultrafiltration membrane at 2 bar; MFI₀.₄₅ values are typically below 5 s/L².' },
      { key: 'toc', label: 'Total organic carbon', unit: 'mg/L', value: 1.6, min: 0, max: 50 },
      { key: 'aoc', label: 'Assimilable organic carbon', unit: 'µg/L', value: 18, min: 0, max: 1000, help: 'Above about 10 µg/L biofouling becomes likely.' },
      { key: 'biocide', label: 'Biocide / shock treatment in use', type: 'bool', value: false },
      { key: 'siMargin', label: 'Worst scaling margin (SI − limit) of the concentrate', unit: 'SI', value: 0.1, min: -5, max: 5, help: 'From suite 2: positive means a mineral is beyond what the antiscalant controls.' },
    ] },
    { group: 'Laboratory tests (optional)', help: 'Bench data that anchor two mechanistic relations: an adsorption isotherm of the organic foulant on the membrane, and a flux-stepping test for the threshold flux. The built-in rows are examples.', fields: [
      { key: 'iso', label: 'Adsorption isotherm data', type: 'table', columns: [{ key: 'C', label: 'Equilibrium concentration', unit: 'mg/L' }, { key: 'q', label: 'Adsorbed amount', unit: 'mg/m²' }], value: ISO_DEFAULT, help: 'Static adsorption test of the organic foulant (as carbon) on membrane coupons. Langmuir and Freundlich isotherms are fitted; the better one gives the adsorbed load at the wall concentration.' },
      { key: 'fluxStep', label: 'Flux-stepping test', type: 'table', columns: [{ key: 'J', label: 'Flux', unit: 'L/m²·h' }, { key: 'rate', label: 'Fouling rate', unit: '% of resistance per day' }], value: STEP_DEFAULT, help: 'Fouling rate measured at stepwise increased flux. The break point of the rate is the threshold flux.' },
    ] },
    { group: 'Reference conditions', tab: 'setup', help: 'Initial condition of the monitoring: what counts as 100 % performance.', fields: [
      { key: 'refMode', label: 'Reference state', type: 'select', value: 'baseline', options: [{ value: 'baseline', label: 'First days of the log (clean baseline)' }, { value: 'design', label: 'Design point entered below' }] },
      { key: 'nBase', label: 'Rows in the baseline window', unit: '', value: 10, min: 3, max: 60, step: 1, showIf: (v) => v.refMode === 'baseline' },
      { key: 'dQf', label: 'Design feed flow', unit: 'm³/h', value: 200, min: 0.1, max: 1e5, showIf: (v) => v.refMode === 'design' }, { key: 'dQp', label: 'Design permeate flow', unit: 'm³/h', value: 150, min: 0.1, max: 1e5, showIf: (v) => v.refMode === 'design' },
      { key: 'dPf', label: 'Design feed pressure', unit: 'bar', value: 12.7, min: 1, max: 120, showIf: (v) => v.refMode === 'design' }, { key: 'dPc', label: 'Design concentrate pressure', unit: 'bar', value: 9.9, min: 0.5, max: 120, showIf: (v) => v.refMode === 'design' },
      { key: 'dPp', label: 'Design permeate pressure', unit: 'bar', value: 1, min: 0, max: 20, showIf: (v) => v.refMode === 'design' }, { key: 'dT', label: 'Design temperature', unit: '°C', value: 25, min: 1, max: 45, showIf: (v) => v.refMode === 'design' },
      { key: 'dCf', label: 'Design feed conductivity', unit: 'µS/cm', value: 5860, min: 10, max: 1e5, showIf: (v) => v.refMode === 'design' }, { key: 'dCp', label: 'Design permeate conductivity', unit: 'µS/cm', value: 153, min: 0.1, max: 5000, showIf: (v) => v.refMode === 'design' },
    ] },
    { group: 'Normalisation model', tab: 'setup', fields: [
      { key: 'mFlow', label: 'Flow exponent of the pressure drop', unit: '–', value: 1.5, min: 1, max: 2, help: 'ΔP ∝ (average feed–brine flow)^m; 1.4–1.7 for spacer-filled channels.' },
      { key: 'viscCorr', label: 'Correct the pressure drop for viscosity', type: 'bool', value: true, help: 'Removes the seasonal temperature effect on ΔP (ΔP ∝ μ^0.3).' },
      { key: 'outlierSigma', label: 'Outlier threshold (Hampel)', unit: 'σ', value: 4.5, min: 2.5, max: 10 },
    ] },
    { group: 'Change detection and alarms', tab: 'setup', help: 'Sudden changes are detected from the shift of the running median; gradual drifts by EWMA and CUSUM control charts.', fields: [
      { key: 'stepWin', label: 'Step-detection window', unit: 'rows', value: 4, min: 2, max: 15, step: 1 },
      { key: 'stepNPF', label: 'Sudden change: permeate flow', unit: '% of reference', value: 3, min: 0.5, max: 30 }, { key: 'stepNSP', label: 'Sudden change: salt passage', unit: '% of reference', value: 15, min: 2, max: 200 }, { key: 'stepNDP', label: 'Sudden change: differential pressure', unit: '% of reference', value: 4, min: 0.5, max: 50 },
      { key: 'ewmaLambda', label: 'EWMA weight λ', unit: '–', value: 0.2, min: 0.05, max: 1 }, { key: 'ewmaL', label: 'EWMA control-limit width L', unit: 'σ', value: 3, min: 2, max: 5 },
      { key: 'cusumK', label: 'CUSUM allowance k', unit: 'σ', value: 0.5, min: 0.1, max: 2 }, { key: 'cusumH', label: 'CUSUM decision interval h', unit: 'σ', value: 5, min: 2, max: 12 },
      { key: 'lateWin', label: 'Diagnosis window before a cleaning', unit: 'd', value: 25, min: 5, max: 120, help: 'The foulant is diagnosed from the trends in this last part of each cycle.' },
    ] },
    { group: 'Cleaning triggers and end of life', tab: 'setup', help: 'Limits against which the trends are extrapolated.', fields: [
      { key: 'trigNPF', label: 'Clean when normalised permeate flow has dropped by', unit: '%', value: 10, min: 3, max: 30, typical: [10, 15] },
      { key: 'trigDP', label: 'Clean when normalised differential pressure has risen by', unit: '%', value: 15, min: 5, max: 60 },
      { key: 'trigSP', label: 'Clean when normalised salt passage has risen by', unit: '%', value: 10, min: 3, max: 100, typical: [5, 10] },
      { key: 'refAfter', label: 'Triggers are measured from', type: 'select', value: 'startup', options: [{ value: 'startup', label: 'The original reference' }, { value: 'postclean', label: 'The level right after the last cleaning' }] },
      { key: 'eolLoss', label: 'End of life: irreversible permeability loss', unit: '%', value: 25, min: 5, max: 60 },
      { key: 'revFrac', label: 'Assumed reversible share of fouling (if no cleaning is in the log)', unit: '%', value: 85, min: 0, max: 100 },
    ] },
    { group: 'Fouling-resistance model', tab: 'setup', help: 'Parameters of the mechanistic pressure model (permeability decline by a blocking law, linear growth of the pressure drop). Calibrate them on the Calibrate tab.', fields: [
      { key: 'Aclean', label: 'Clean water permeability (25 °C)', unit: 'L/m²·h·bar', value: 3.3, min: 0.2, max: 15 }, { key: 'kFoul', label: 'Fouling rate constant k', unit: '1/d', value: 0.002, min: 0, max: 0.2 },
      { key: 'law', label: 'Blocking law', type: 'select', value: 'intermediate', options: Object.entries(HERMIA).map(([value, h]) => ({ value, label: `${h.name} (n = ${h.n})` })) },
      { key: 'dp0', label: 'Clean differential pressure at reference flow', unit: 'bar', value: 2.6, min: 0.1, max: 15 }, { key: 'kDp', label: 'Pressure-drop growth rate', unit: '1/d', value: 0.0015, min: 0, max: 0.1 },
      { key: 'qRef', label: 'Reference average feed–brine flow', unit: 'm³/h', value: 125, min: 0.1, max: 1e5 }, { key: 'sp0', label: 'Salt passage at reference', unit: '%', value: 1.2, min: 0.01, max: 50 }, { key: 'PpRef', label: 'Permeate pressure', unit: 'bar', value: 1, min: 0, max: 20 },
    ] },
    { group: 'Deposit, critical flux and biofilm', tab: 'setup', help: 'Cake filtration (Kozeny–Carman with compressibility), particle back-transport and Monod biofilm growth.', fields: [
      { key: 'dpNm', label: 'Foulant particle diameter', unit: 'nm', value: 20, min: 1, max: 1e5 }, { key: 'eps', label: 'Deposit porosity', unit: '–', value: 0.4, min: 0.1, max: 0.9 }, { key: 'rhoP', label: 'Particle density', unit: 'kg/m³', value: 2000, min: 900, max: 6000 },
      { key: 'sComp', label: 'Cake compressibility index', unit: '–', value: 0.5, min: 0, max: 1.2, help: 'α = α₀ (ΔP / 1 bar)^s; 0 = incompressible.' }, { key: 'omega', label: 'Particle deposition factor', unit: '–', value: 0.15, min: 0, max: 1, help: 'Share of the particles carried to the membrane that actually deposit under cross-flow (1 = dead-end filtration).' },
      { key: 'uCross', label: 'Cross-flow velocity', unit: 'm/s', value: 0.12, min: 0.01, max: 1 }, { key: 'hChan', label: 'Feed-channel height', unit: 'mm', value: 0.71, min: 0.3, max: 2 }, { key: 'lChan', label: 'Channel length (vessel)', unit: 'm', value: 6, min: 0.5, max: 10 }, { key: 'phiB', label: 'Particle volume fraction in the feed', unit: '–', value: 1e-6, min: 1e-9, max: 1e-2 },
      { key: 'mumax', label: 'Biofilm maximum growth rate', unit: '1/d', value: 0.6, min: 0.01, max: 5 }, { key: 'KsAoc', label: 'Half-saturation constant', unit: 'µg/L', value: 60, min: 1, max: 1000 }, { key: 'kd', label: 'Decay and detachment rate', unit: '1/d', value: 0.08, min: 0, max: 2 },
      { key: 'Xmax', label: 'Biofilm carrying capacity', unit: 'g/m²', value: 12, min: 0.5, max: 100 }, { key: 'X0', label: 'Initial biomass after cleaning', unit: 'g/m²', value: 0.05, min: 1e-4, max: 5 }, { key: 'betaBio', label: 'Pressure-drop rise per unit biomass', unit: '%·m²/g', value: 8, min: 0.1, max: 100 },
      { key: 'horizon', label: 'Forecast horizon', unit: 'd', value: 120, min: 10, max: 730 },
    ] },
    { group: 'Adsorption, attachment and scaling', tab: 'setup', help: 'Initial conditions and boundary conditions of the mechanistic fouling models: pore availability, adsorptive loss, detachment by wall shear and scale growth on the tail elements.', fields: [
      { key: 'poreAvail0', label: 'Initial pore availability', unit: '%', value: 100, min: 5, max: 100, help: 'Share of the membrane pores (permeable area) that is open at the start of the log. After each cleaning the model continues with the availability that the cleaning restored.' },
      { key: 'adsLoss', label: 'Permeability loss at full adsorptive coverage', unit: '%', value: 8, min: 0, max: 90, help: 'Loss caused by a complete adsorbed layer of the organic foulant; the isotherm gives the coverage reached at the wall concentration.' },
      { key: 'kDet', label: 'Detachment coefficient', unit: '1/(Pa·d)', value: 0.05, min: 0, max: 50, help: 'Deposit removed per day and per pascal of wall shear stress: detachment flux = k_det·τ_w·m.' },
      { key: 'ms0', label: 'Initial scaling mass', unit: 'g/m²', value: 0, min: 0, max: 500, help: 'Scale already present on the tail elements at the start of the log (reset to zero by a cleaning).' },
      { key: 'kScale', label: 'Scale growth constant', unit: 'g/m²·d', value: 0.2, min: 0, max: 100, help: 'Growth rate of the scale layer at a supersaturation ratio of 2 beyond the antiscalant limit: dm/dt = k·(S − 1)².' },
      { key: 'mCover', label: 'Scale mass that covers the surface', unit: 'g/m²', value: 10, min: 0.1, max: 500, help: 'Covered area fraction = 1 − exp(−m/m_c); 10 g/m² is a layer of a few µm.' },
      { key: 'tailShare', label: 'Share of area on the scaling tail elements', unit: '%', value: 20, min: 1, max: 100, help: 'Only the last elements, where the concentrate is strongest, scale.' },
      { key: 'bioXf', label: 'Biofilm dry density', unit: 'kg/m³', value: 30, min: 5, max: 200, help: 'Dry biomass per volume of biofilm; converts areal biomass to thickness.' }, { key: 'bioY', label: 'Biomass yield', unit: 'g/g', value: 0.5, min: 0.05, max: 1, help: 'Biomass formed per unit of assimilable carbon consumed.' },
      { key: 'stageRatio', label: 'Vessel ratio between the stages', unit: ': 1', value: 2, min: 1, max: 4, help: 'Number of first-stage vessels per second-stage vessel (2 for a 2 : 1 array); used by the channel model when the log has an interstage pressure.', showIf: (v) => v.chanOn },
      { key: 'chanOn', label: 'Solve the deposition profile along the feed channel', type: 'bool', value: true, help: 'One-dimensional channel from feed inlet to concentrate outlet: cross-flow, pressure, local flux, wall shear, critical flux and deposit, with the concentrate pressure of the log as outlet boundary condition.' },
    ] },
    { group: 'Time stepping', tab: 'mesh', help: 'Fixed-step fourth-order Runge–Kutta integration of the biofilm model over the forecast horizon.', fields: [
      { key: 'nStep', label: 'Time steps over the horizon', unit: '', value: 60, min: 4, max: 5000, step: 1 },
      { key: 'nChan', label: 'Cells along the feed channel', unit: '', value: 40, min: 8, max: 400, step: 1, help: 'Axial cells of the channel deposition model.' },
      { key: 'nzBio', label: 'Cells across the biofilm', unit: '', value: 20, min: 4, max: 200, step: 1, help: 'Finite-difference cells of the substrate diffusion–reaction profile.' },
    ] },
  ],

  presets: [
    { name: 'Two-stage brackish RO, 180 days: fouling, scaling, CIP, O-ring fault', values: {} },
    { name: 'Same plant, first 90 days only (early-warning view)', values: { get log() { return defaultLog().filter((r) => r.t < 90); } } },
    { name: 'Biofouling-prone warm feed, no interstage pressure', values: { get log() { return bioLog(); }, aoc: 120, toc: 3.5, sdi: 2.4, siMargin: -0.6, trigDP: 15 } },
    { name: 'Tighter triggers measured from the last cleaning', values: { trigNPF: 8, trigDP: 12, trigSP: 8, refAfter: 'postclean' } },
    { name: 'Design-point reference instead of baseline', values: { refMode: 'design', dQf: 200, dQp: 150, dPf: 12.7, dPc: 9.9, dPp: 1, dT: 25, dCf: 5860, dCp: 153 } },
  ],

  pull: ({ outputs }) => {
    const ro = outputs?.ro, f = ro?.streams?.feed, p = ro?.streams?.permeate, from = 'RO design point (used when the reference state is “design”)';
    return [
      isNum(outputs?.chem?.scalingMargin) ? { key: 'siMargin', value: +clamp(outputs.chem.scalingMargin, -5, 5).toFixed(3), from: 'Brine chemistry: worst saturation index minus its limit' } : null,
      ...(ro?.permeateFlow ? [{ key: 'dQp', value: ro.permeateFlow, from }, f?.Q ? { key: 'dQf', value: f.Q, from } : null, ro.feedPressureBar ? { key: 'dPf', value: ro.feedPressureBar, from } : null, ro.concentratePressureBar ? { key: 'dPc', value: ro.concentratePressureBar, from } : null,
        p?.P != null ? { key: 'dPp', value: p.P, from } : null, f?.T != null ? { key: 'dT', value: f.T, from } : null, f?.tds ? { key: 'dCf', value: Math.round(conductivityFromTDS(f.tds)), from } : null, p?.tds ? { key: 'dCp', value: +conductivityFromTDS(p.tds).toFixed(1), from } : null] : []),
    ].filter(Boolean);
  },
  site: () => [],

  run(v, ctx) {
    ctx?.progress?.(0.2, 'Validating and normalising the log…');
    const a = analyseLog(v), { G, t, npf, nsp, ndp, cycles, cur, events, trig, now, spc } = a, W = [];
    const FN = (k) => FOULANTS[k].name, pct = (x) => fmt(x, 3);
    // Hermia fits on the first cycle with a chronological split
    const c1 = cycles[0], hy = npf.slice(c1.a, c1.b).map((x) => x / 100), ht = t.slice(c1.a, c1.b).map((x) => x - t[c1.a]), hf = ht.length >= 10 ? fitHermia(ht, hy, 0.7) : null;
    // cake filtration, MFI and critical flux
    const dp = v.dpNm * 1e-9, alpha0 = kozenyCarman(dp, v.eps, v.rhoP), ndpNow = med(G.slice(-5).map((q) => q.ndp)), alpha = alpha0 * Math.max(ndpNow, 0.01) ** v.sComp;
    const mCake = a.Rrev / alpha, delta = mCake / (v.rhoP * (1 - v.eps)), fluxNow = med(G.slice(-5).map((q) => q.flux)), Jms = fluxNow / 3.6e6;
    const Imfi = (v.mfi * 1e6 * 2 * 207e3 * (13.8e-4) ** 2) / 1.002e-3, mfiRate = (100 * v.omega * Imfi * Jms * 86400) / Math.max(med(a.Rt.slice(-5)), 1); // % of total resistance per day
    const Tm = mean(G.map((q) => q.row.T)), cf = criticalFlux(dp, v.uCross, { h: v.hChan * 1e-3, L: v.lChan, T: Tm, phiB: v.phiB }), jc = cf.J * 3.6e6;
    const fx = linspace(-2, 1, 13), fy = linspace(0.04, 0.3, 9), fz = fy.map((u) => fx.map((lx) => Math.min(200, criticalFlux(10 ** lx * 1e-6, u, { h: v.hChan * 1e-3, L: v.lChan, T: Tm, phiB: v.phiB }).J * 3.6e6)));
    // biofilm projection of the first-stage pressure drop
    const d1now = a.stageOK ? med(G.slice(-5).map((q) => q.dpn1).filter((x) => x != null)) : now.ndp, d1start = a.stageOK ? med(G.slice(cur.a, cur.a + 5).map((q) => q.dpn1).filter((x) => x != null)) : cur.startNDP;
    const Xnow = clamp(Math.max(v.X0, (d1now - Math.min(d1start, d1now)) / v.betaBio), v.X0, 0.98 * v.Xmax), bio = biofilm({ X0: Xnow, mumax: v.mumax, Ks: v.KsAoc, S: v.aoc, kd: v.kd, Xmax: v.Xmax }, v.horizon, v.nStep);
    const bioDp = bio.X.map((x) => d1now + v.betaBio * (x - Xnow)), limDp = (v.refAfter === 'postclean' ? d1start : 100) * (1 + v.trigDP / 100), ib = bioDp.findIndex((x) => x >= limDp);
    const bioDays = d1now >= limDp ? 0 : ib > 0 ? bio.t[ib - 1] + ((limDp - bioDp[ib - 1]) * (bio.t[ib] - bio.t[ib - 1])) / (bioDp[ib] - bioDp[ib - 1]) : v.horizon;
    const d1rate = a.stageOK ? cur.ndp1.slope : cur.ndp.slope;
    // warnings
    const dueNow = a.daysToCleaning <= 0;
    if (a.integrity) W.push({ level: 'bad', msg: `Sudden salt-passage rise of ${pct(sum(cur.steps.filter((e) => e.type === 'integrity').map((e) => e.dNSP)))} % of reference on day ${fmt(cur.steps.find((e) => e.type === 'integrity').t, 4)} without loss of flow: mechanical integrity fault (O-ring, interconnector or glue line). Cleaning will not restore rejection — probe the vessels.` });
    for (const k of ['npf', 'ndp', 'nsp']) if (trig[k].margin <= 0 && !(k === 'nsp' && a.integrity)) W.push({ level: 'bad', msg: `${trig[k].name} is at ${pct(trig[k].now)} % of reference, beyond the cleaning trigger (${pct(trig[k].limit)} %) — clean now; delaying makes the deposit harder to remove.` });
    if (!dueNow && a.daysToCleaning < 21) W.push({ level: 'warn', msg: `The ${trig[a.first].name.toLowerCase()} trigger is expected in about ${fmt(a.daysToCleaning, 2)} days (95 % band ${fmt(trig[a.first].lo, 2)}–${fmt(trig[a.first].hi, 2)} d): schedule the cleaning.` });
    for (const c of a.cleanings) if (c.recovery < 0.8) W.push({ level: 'warn', msg: `The cleaning on day ${fmt(c.t, 4)} recovered only ${pct(100 * c.recovery)} % of the lost permeate flow — the remainder is irreversible or the recipe did not match the foulant (${FN(c.foulant)}).` });
    if (a.issues.length) W.push({ level: 'info', msg: `${a.issues.length} data-quality issue${a.issues.length > 1 ? 's were' : ' was'} found and handled (see the validation table).` });
    if (!a.stageOK) W.push({ level: 'info', msg: 'No interstage pressure in the log: first-stage and last-stage pressure drops cannot be separated, so colloidal/biological fouling and scaling are distinguished less reliably.' });
    if (fluxNow > jc) W.push({ level: 'info', msg: `Operating flux ${fmt(fluxNow, 3)} L/m²·h is above the estimated critical flux ${fmt(jc, 3)} L/m²·h for ${v.dpNm} nm particles: deposition is expected.` });
    if (v.sdi > 5) W.push({ level: 'bad', msg: `SDI₁₅ = ${v.sdi} exceeds the limit of 5 for spiral-wound elements — improve pretreatment.` });
    if (hf && Math.abs(hf.best.biasVal) > 2 * Math.sqrt(hf.best.sse / hf.nTrain) + 0.005) W.push({ level: 'info', msg: `The blocking law fitted to the first ${fmt(hf.split, 3)} days ${hf.best.biasVal > 0 ? 'over' : 'under'}-predicts the later flux by ${pct(100 * Math.abs(hf.best.biasVal))} % of reference: the fouling mechanism changed during the cycle.` });
    if (!W.some((w) => w.level !== 'info')) W.unshift({ level: 'info', msg: 'All normalised indicators are within their cleaning triggers.' });

    const curIssue = cur.top, foulRate = Math.max(0, -cur.npf.slope), life = a.age + a.rul;
    ctx?.progress?.(0.7, 'Mechanistic and state-space models…');
    const X = extendedFouling(v, a, { hf, ht, hy, alpha, Jms, Tm, cf, fluxNow, foulRate, Xnow, d1now, limDp, bio, bioDays });
    W.push(...X.W);
    const ev = events.map((e) => ({ x: e.t, label: e.type === 'cleaning' ? 'CIP' : e.type === 'integrity' ? 'fault' : 'step' }));
    const proj = linspace(0, Math.min(v.horizon, Math.max(20, Math.min(1.4 * a.daysToCleaning, v.horizon))), 12), last = now.t, line = (y0, r) => proj.map((d) => y0 + r * d);
    const pm = modelPars(v), pred = G.map((q) => { const cy = cycles.find((c) => q.t >= c.t0 && q.t <= c.t1) || cur; return pressureModel(pm, { tc: q.t - cy.t0, T: q.row.T, Qf: q.row.Qf, Qp: q.row.Qp, Cf: q.row.Cf, Pp: q.row.Pp }); });
    const probs = Object.keys(FOULANTS), recipe = FOULANTS[a.integrity && cur.topFoulant && cur.loss < 3 && now.ndp < trig.ndp.limit ? 'integrity' : cur.topFoulant].cip, recipeFor = a.integrity && cur.loss < 3 && now.ndp < trig.ndp.limit ? 'integrity' : cur.topFoulant;
    const Rnow = med(a.Rt.slice(-5)), dstr = (x) => (x >= 3650 ? '> 3650' : fmt(x, 3));
    const out = {
      foulingRate: foulRate, daysToCleaning: a.daysToCleaning, daysToCleaningLow: trig[a.first].lo, daysToCleaningHigh: trig[a.first].hi, cleaningsPerYear: a.cleaningsPerYear, membraneLife: life, remainingLife: a.rul, remainingLifeLow: a.rulLo, remainingLifeHigh: a.rulHi,
      normPermeability: now.npf / 100, normSaltPassage: now.nsp / 100, normDP: now.ndp / 100, dominantFoulant: FN(a.dominant), currentIssue: FN(curIssue), integrityFault: a.integrity, irreversibleLoss: a.irrNow, cleaningRecovery: a.lastClean ? a.lastClean.recovery : null,
      limitingTrigger: trig[a.first].name, permeability: (a.ref.kA * now.npf * 10) / a.area, cleanPermeability: (a.ref.kA * 1000) / a.area, Rm: a.Rm, Rreversible: a.Rrev, Rirreversible: a.Rirr, criticalFluxLMH: jc, bestBlockingLaw: hf ? HERMIA[hf.best.law].name : 'not fitted', bioDaysToTrigger: bioDays, bioDpAtHorizon: bioDp[bioDp.length - 1], dataIssues: a.issues.length, events: events.map((e) => ({ day: e.t, type: e.type })), ...X.out,
    };
    if (out.cleaningRecovery == null) delete out.cleaningRecovery;
    return {
      summary: `Normalised permeate flow is at ${pct(now.npf)} % of reference, salt passage at ${pct(now.nsp)} % and differential pressure at ${pct(now.ndp)} %. Permeability is falling by ${fmt(foulRate, 2)} %/d; ${dueNow ? 'a cleaning is due now' : `the next cleaning is expected in about ${dstr(a.daysToCleaning)} days (${dstr(trig[a.first].lo)}–${dstr(trig[a.first].hi)})`}. Dominant foulant over the record: ${FN(a.dominant).toLowerCase()}${a.integrity ? '; an integrity fault is active' : ''}.`,
      warnings: W,
      kpis: [
        { label: 'Normalised permeate flow', value: now.npf, unit: '% of ref.', status: trig.npf.margin <= 0 ? 'bad' : trig.npf.days < 21 ? 'warn' : 'ok' }, { label: 'Normalised salt passage', value: now.nsp, unit: '% of ref.', status: a.integrity || trig.nsp.margin <= 0 ? 'bad' : 'ok' },
        { label: 'Normalised differential pressure', value: now.ndp, unit: '% of ref.', status: trig.ndp.margin <= 0 ? 'bad' : trig.ndp.days < 21 ? 'warn' : 'ok' }, { label: 'Permeability loss rate', value: foulRate, unit: '%/d', help: `Robust trend of the current cycle (95 % interval ${fmt(Math.max(0, -cur.npf.hi), 2)}–${fmt(Math.max(0, -cur.npf.lo), 2)})` },
        { label: 'Days to next cleaning', value: a.daysToCleaning >= 3650 ? '> 3650' : a.daysToCleaning, unit: 'd', status: dueNow ? 'bad' : a.daysToCleaning < 21 ? 'warn' : 'ok', help: `Limited by ${trig[a.first].name.toLowerCase()}; 95 % band ${dstr(trig[a.first].lo)}–${dstr(trig[a.first].hi)} d` },
        { label: 'Cleanings per year', value: a.cleaningsPerYear, unit: '1/y', status: a.cleaningsPerYear > 4 ? 'warn' : 'ok' }, { label: 'Dominant foulant', value: FN(a.dominant) }, { label: 'Current main issue', value: FN(curIssue), status: curIssue === 'integrity' || curIssue === 'oxidation' ? 'bad' : 'ok' },
        { label: 'Last cleaning: flow recovery', value: a.lastClean ? 100 * a.lastClean.recovery : 'no cleaning in log', unit: a.lastClean ? '%' : '', status: a.lastClean && a.lastClean.recovery < 0.8 ? 'warn' : 'ok' }, { label: 'Irreversible permeability loss', value: a.irrNow, unit: '%' },
        { label: 'Remaining membrane life', value: a.rul, unit: 'years', help: `Until ${v.eolLoss} % irreversible loss; band ${fmt(a.rulLo, 2)}–${fmt(a.rulHi, 2)} years`, status: a.rul < 1 ? 'warn' : 'ok' }, { label: 'Current permeability (25 °C)', value: out.permeability, unit: 'L/m²·h·bar' },
        { label: 'Operating flux', value: fluxNow, unit: 'L/m²·h', status: fluxNow > jc ? 'warn' : 'ok' }, { label: 'Critical flux estimate', value: jc, unit: 'L/m²·h' },
        { label: 'Fouling resistance / membrane resistance', value: (Rnow - a.Rm) / a.Rm, unit: '–', help: 'Darcy resistance in series, referred to 25 °C: (R_total − R_membrane)/R_membrane' }, { label: 'Deposit loading (cake model)', value: mCake * 1000, unit: 'g/m²' },
        { label: 'Best blocking law', value: hf ? HERMIA[hf.best.law].name : 'not fitted', help: hf ? `R² = ${fmt(hf.best.r2, 4)} on the training part of the first cycle` : '' }, { label: 'Detected events', value: `${events.filter((e) => e.type === 'cleaning').length} cleaning · ${events.filter((e) => e.type !== 'cleaning').length} fault/step` },
        ...X.K,
      ],
      recommendations: [
        a.integrity ? 'Probe every pressure vessel for permeate conductivity to locate the leaking O-ring or element; replace it before the next cleaning.' : null,
        dueNow ? `Clean now with the ${FN(recipeFor).toLowerCase()} recipe below, starting with the stage that shows the higher pressure-drop rise.` : a.daysToCleaning < 45 ? `Plan a cleaning within ${fmt(a.daysToCleaning, 2)} days (see the recipe table).` : null,
        a.dominant === 'scaling' ? 'Scaling dominated the fouling history: check the antiscalant dose and the recovery set-point with suite 2 (Brine chemistry).' : null,
        a.dominant === 'bio' || cur.topFoulant === 'bio' ? 'Biofouling pattern: reduce assimilable carbon (avoid over-dosing antiscalant and bisulphite), consider periodic non-oxidising biocide and shorter intervals between preventive cleanings.' : null,
        a.dominant === 'colloidal' || v.sdi > 3 ? 'Particulate load is marginal: check the cartridge filters and coagulation; a lower flux in the lead elements reduces deposition.' : null,
        a.cleaningsPerYear > 4 ? 'More than four cleanings per year indicates a pretreatment problem rather than a cleaning problem.' : null,
        'Pass the normalised permeability to suite 1 (flow factor) and the cleaning frequency and membrane life to suite 13 (Economics).',
      ].filter(Boolean),
      plots: [
        { type: 'line', title: 'Normalised permeate flow', xlabel: 'Time (d)', ylabel: '% of reference', series: [{ name: 'Normalised permeate flow', x: t, y: npf, mode: 'both' }, { name: 'Measured permeate flow (% of reference flow)', x: t, y: G.map((q) => (100 * q.row.Qp) / a.ref.Qp), dash: true }, { name: 'Trend of current cycle', x: [cur.t0, now.t], y: [cur.npf.intercept + cur.npf.slope * cur.t0, cur.npf.intercept + cur.npf.slope * now.t] }], hlines: [{ y: trig.npf.limit, label: 'cleaning trigger' }], vlines: ev, note: 'The measured flow is held by the control system; the normalised flow shows what the membranes would deliver at reference pressure and temperature.' },
        { type: 'line', title: 'Normalised salt passage', xlabel: 'Time (d)', ylabel: '% of reference', series: [{ name: 'Normalised salt passage', x: t, y: nsp, mode: 'both' }, { name: 'Gradual component (steps removed)', x: t, y: a.nspG, dash: true }], hlines: [{ y: trig.nsp.limit, label: 'trigger' }], vlines: ev },
        { type: 'line', title: 'Normalised differential pressure', xlabel: 'Time (d)', ylabel: '% of reference', series: [{ name: 'Whole train', x: t, y: ndp, mode: 'both' }, ...(a.stageOK ? [{ name: 'First stage', x: t, y: G.map((q) => q.dpn1 ?? 100) }, { name: 'Last stage', x: t, y: G.map((q) => q.dpn2 ?? 100) }] : [])], hlines: [{ y: trig.ndp.limit, label: 'trigger' }], vlines: ev, note: a.stageOK ? 'First-stage rise points to particulate or biological fouling; last-stage rise to scaling.' : '' },
        { type: 'line', title: 'Darcy resistance in series', xlabel: 'Time (d)', ylabel: 'Resistance (10¹³ m⁻¹)', zeroY: true, series: [{ name: 'Total resistance', x: t, y: a.Rt.map((x) => x / 1e13), mode: 'both' }, { name: 'Membrane resistance R_m', x: [t[0], now.t], y: [a.Rm / 1e13, a.Rm / 1e13], dash: true }, { name: 'R_m + irreversible', x: [t[0], now.t], y: [(a.Rm + a.Rirr) / 1e13, (a.Rm + a.Rirr) / 1e13], dash: true }], vlines: ev },
        ...(hf ? [{ type: 'line', title: 'Blocking-law fits to the first cycle (chronological split)', xlabel: 'Time in cycle (d)', ylabel: 'Normalised flux J/J₀', series: [{ name: 'Training data', x: ht.slice(0, hf.nTrain), y: hy.slice(0, hf.nTrain), mode: 'points' }, { name: 'Validation data (not used for fitting)', x: ht.slice(hf.nTrain), y: hy.slice(hf.nTrain), mode: 'points' }, ...hf.fits.map((f) => ({ name: `${HERMIA[f.law].name} (R² ${fmt(f.r2, 3)})`, x: ht, y: ht.map((x) => f.J0 * hermia(f.law, f.k * x)), dash: f.law !== hf.best.law }))], vlines: [{ x: hf.split, label: 'train | validate' }] }] : []),
        { type: 'line', title: 'Control charts of the normalised permeate flow', xlabel: 'Time (d)', ylabel: 'Standardised units (σ)', series: [{ name: 'EWMA', x: t, y: spc.ewma.map((x) => clamp(x, -30, 30)) }, { name: 'CUSUM (downward)', x: t, y: spc.cm.map((x) => -Math.min(x, 60)) }], hlines: [{ y: -spc.limE, label: 'EWMA limit' }, { y: -spc.h, label: 'CUSUM limit' }], vlines: spc.alarms.map((al) => ({ x: al.t, label: 'alarm' })), ymin: -65, note: 'Charts restart from a new baseline after every detected cleaning.' },
        { type: 'line', title: 'Forecast to the cleaning trigger', xlabel: 'Time (d)', ylabel: '% of reference', series: [{ name: 'Permeate flow (history)', x: t.slice(cur.a), y: npf.slice(cur.a), mode: 'points' }, { name: 'Permeate flow forecast', x: proj.map((d) => last + d), y: line(now.npf, cur.npf.slope) }, { name: '95 % band', x: proj.map((d) => last + d), y: line(now.npf, cur.npf.lo), dash: true }, { name: '95 % band ', x: proj.map((d) => last + d), y: line(now.npf, cur.npf.hi), dash: true },
          { name: 'Differential pressure (history)', x: t.slice(cur.a), y: ndp.slice(cur.a), mode: 'points' }, { name: 'Differential pressure forecast', x: proj.map((d) => last + d), y: line(now.ndp, cur.ndp.slope) }], hlines: [{ y: trig.npf.limit, label: 'flow trigger' }, { y: trig.ndp.limit, label: 'ΔP trigger' }] },
        { type: 'bar', title: 'Evidence scores by cause', ylabel: '% of total evidence', categories: probs.map((k) => FN(k).split(' (')[0]), series: cycles.map((c, i) => ({ name: `Cycle ${i + 1} (day ${fmt(c.t0, 4)}–${fmt(c.t1, 4)})`, values: probs.map((k) => 100 * c.prob[k]) })) },
        { type: 'field', title: 'Critical flux versus particle size and cross-flow velocity', xlabel: 'log₁₀ particle diameter (µm)', ylabel: 'Cross-flow velocity (m/s)', zlabel: 'Critical flux', zunit: 'L/m²·h', x: fx, y: fy, z: fz, cmap: 'viridis', contours: 8, zmin: 0, zmax: Math.min(200, Math.max(...fz.flat())), markers: [{ x: clamp(Math.log10(v.dpNm / 1000), -2, 1), y: clamp(v.uCross, 0.04, 0.3), label: 'this plant' }], note: `Brownian diffusion protects against the smallest particles and shear-induced diffusion against the largest; particles around 0.1–1 µm deposit most easily. Operating flux: ${fmt(fluxNow, 3)} L/m²·h.` },
        { type: 'line', title: `${a.stageOK ? 'First-stage' : 'Train'} pressure drop: Monod biofilm projection versus linear trend`, xlabel: 'Days from now', ylabel: '% of reference', series: [{ name: 'Biofilm model (Monod growth)', x: bio.t, y: bioDp }, { name: 'Linear trend of the current cycle', x: bio.t, y: bio.t.map((d) => d1now + d1rate * d), dash: true }], hlines: [{ y: limDp, label: 'ΔP trigger' }] },
        { type: 'line', title: 'Feed pressure: measured versus fouling-resistance model', xlabel: 'Time (d)', ylabel: 'bar', series: [{ name: 'Measured', x: t, y: G.map((q) => q.row.Pf), mode: 'points' }, { name: 'Model (current parameters, reset at each cleaning)', x: t, y: pred.map((p) => p.Pf) }], vlines: ev, note: 'Fit the model parameters on the Calibrate tab; a model that tracks the data before an event and departs after it shows when the mechanism changed.' },
        ...X.PL,
      ],
      tables: [
        { title: 'Operating cycles', columns: ['Cycle', 'From (d)', 'To (d)', 'Permeate flow start → end (% ref.)', 'Flow trend (%/d)', '95 % interval', 'Salt-passage trend (%/d)', 'ΔP trend (%/d)', 'First-stage ΔP trend (%/d)', 'Last-stage ΔP trend (%/d)', 'Most likely cause', 'Evidence share (%)', 'Early-warning lead (d)'],
          rows: cycles.map((c, i) => [i + 1, c.t0, c.t1, `${pct(c.startNPF)} → ${pct(c.endNPF)}`, c.npf.slope, `${fmt(c.npf.lo, 2)} … ${fmt(c.npf.hi, 2)}`, c.nsp.slope, c.ndp.slope, a.stageOK ? c.ndp1.slope : null, a.stageOK ? c.ndp2.slope : null, FN(c.top), 100 * c.prob[c.top], c.lead]),
          note: 'Trends are Theil–Sen slopes of the gradual component (sudden steps removed) in % of reference per day. The cause is diagnosed from the last part of each cycle. Lead = days between the first control-chart alarm and the cleaning trigger.' },
        { title: 'Detected sudden changes', columns: ['Day', 'Classification', 'Δ permeate flow (% ref.)', 'Δ salt passage (% ref.)', 'Δ differential pressure (% ref.)', 'Nature'], rows: events.length ? events.map((e) => [e.t, e.label, e.dNPF, e.dNSP, e.dNDP, 'Sudden']) : [['–', 'No sudden change detected: all deterioration is gradual', null, null, null, 'Gradual']] },
        { title: 'Cleaning triggers and forecast', columns: ['Indicator', 'Now (% ref.)', 'Trigger (% ref.)', 'Trend (%/d)', 'Days to trigger', 'Earliest (d)', 'Latest (d)', 'Status'],
          rows: Object.entries(trig).map(([k, q]) => [q.name, q.now, q.limit, q.rate, dstr(q.days), dstr(q.lo), dstr(q.hi), k === 'nsp' && a.integrity ? 'Integrity fault — not a cleaning issue' : q.margin <= 0 ? 'Trigger exceeded' : q.days < 21 ? 'Due soon' : 'OK']),
          note: `Remaining membrane life ${fmt(a.rul, 3)} years (band ${fmt(a.rulLo, 3)}–${fmt(a.rulHi, 3)}) from an irreversible loss rate of ${fmt(a.irrRate * 365, 3)} %/year towards the ${v.eolLoss} % end-of-life limit; membrane age now ${fmt(a.age, 3)} years.` },
        { title: 'Cleaning effectiveness', columns: ['Day', 'Flow before (% ref.)', 'Flow after (% ref.)', 'Flow recovery (%)', 'ΔP before (% ref.)', 'ΔP after (% ref.)', 'ΔP recovery (%)', 'Salt passage before → after (% ref.)', 'Reversible resistance removed (10¹³ m⁻¹)', 'Irreversible resistance left (10¹³ m⁻¹)', 'Verdict'],
          rows: a.cleanings.length ? a.cleanings.map((c) => [c.t, c.npfBefore, c.npfAfter, 100 * c.recovery, c.ndpBefore, c.ndpAfter, 100 * c.dpRecovery, `${pct(c.nspBefore)} → ${pct(c.nspAfter)}`, c.Rrev / 1e13, c.Rirr / 1e13, c.recovery >= 0.9 ? 'Effective — fouling was reversible' : c.recovery >= 0.7 ? 'Partly effective — some irreversible fouling' : 'Poor — irreversible fouling or wrong recipe']) : [['–', null, null, null, null, null, null, 'No cleaning in the log', null, null, '–']] },
        { title: 'Blocking-law fits (first cycle)', columns: ['Law', 'Hermia exponent n', 'J₀ (–)', 'k (1/d)', 'R² training', 'RMSE validation', 'Bias validation', 'R² validation', 'Selected'], rows: hf ? hf.fits.map((f) => [HERMIA[f.law].name, HERMIA[f.law].n, f.J0, f.k, f.r2, f.rmseVal, f.biasVal, f.r2Val, f.law === hf.best.law ? 'best R²' : '']) : [['Too few points in the first cycle', null, null, null, null, null, null, null, '']],
          note: hf ? `Training: first ${hf.nTrain} points (to day ${fmt(hf.tTrainMax, 4)} of the cycle); validation: the following ${hf.nVal} points, never used for fitting. A negative validation R² means the later data follow a different mechanism than the fitted law.${Math.max(...hf.fits.map((f) => f.r2)) - Math.min(...hf.fits.map((f) => f.r2)) < 0.01 ? ' The four laws fit this record equally well: with a flux decline of only a few per cent they cannot be told apart, so the selection is not significant.' : ''}` : '' },
        { title: 'Deposit and feed-water interpretation', columns: ['Quantity', 'Value', 'Unit', 'Interpretation'], rows: [
          ['Membrane resistance R_m', a.Rm / 1e13, '10¹³ m⁻¹', 'Clean membrane (reference state)'], ['Reversible fouling resistance (current)', a.Rrev / 1e13, '10¹³ m⁻¹', 'Removable by cleaning'], ['Irreversible resistance', a.Rirr / 1e13, '10¹³ m⁻¹', a.lastClean ? 'Left after the last cleaning' : 'Assumed share'],
          ['Specific cake resistance α (Kozeny–Carman)', alpha, 'm/kg', `d = ${v.dpNm} nm, ε = ${v.eps}, compressibility ${v.sComp}`], ['Deposit loading', mCake * 1000, 'g/m²', 'm = R_rev / α'], ['Deposit thickness', delta * 1e6, 'µm', 'δ = m / (ρ (1 − ε))'],
          ['SDI₁₅', v.sdi, '%/min', v.sdi < 3 ? 'Good' : v.sdi <= 5 ? 'Marginal — frequent cleaning likely' : 'Unacceptable'], ['Plugging of the SDI filter in 15 min', Math.min(100, 15 * v.sdi), '%', 'P₁₅ = 15·SDI'], ['Fouling index I from MFI', Imfi, 'm⁻²', 'I = MFI·2ΔP₀A₀²/μ'],
          ['Resistance growth predicted from MFI', mfiRate, '%/d', `Observed permeability loss ${fmt(foulRate, 2)} %/d`], ['Wall shear rate', cf.shearRate, '1/s', 'γ = 6u/h'], ['Critical flux (Brownian / shear-induced)', `${fmt(cf.brownian * 3.6e6, 3)} / ${fmt(cf.shear * 3.6e6, 3)}`, 'L/m²·h', fluxNow > jc ? 'Operating above critical flux' : 'Operating below critical flux'],
          ['Biofilm net growth rate', bio.r, '1/d', bio.r > 0 ? `Doubling time ${fmt(Math.log(2) / bio.r, 3)} d at AOC ${v.aoc} µg/L` : 'No net growth'], ['Biofilm model: days to ΔP trigger', bioDays >= v.horizon ? `> ${v.horizon}` : bioDays, 'd', 'If the pressure-drop rise is biological']] },
        { title: `Suggested cleaning recipe — ${FN(recipeFor)}`, columns: ['Step', 'Purpose', 'Chemicals', 'pH', 'Temperature (°C)', 'Duration'], rows: recipe, note: 'Generic recipe for polyamide thin-film composite elements; respect the pH and temperature limits of the element manufacturer, clean each stage separately and flush with permeate between steps.' },
        { title: 'Data validation', columns: ['Day', 'Issue', 'Action'], rows: a.issues.length ? a.issues : [['–', 'No issue found', '–']], note: `${a.rows.length} of ${Array.isArray(v.log) ? v.log.length : 0} rows passed the completeness and range checks; ${G.length} are used for trends. Median sampling interval ${fmt(a.dtMed, 3)} d.` },
        { title: 'Normalised data', columns: ['Day', 'Recovery (%)', 'Net driving pressure (bar)', 'Temperature factor', 'Flux (L/m²·h)', 'Normalised permeate flow (% ref.)', 'Normalised salt passage (% ref.)', 'Normalised ΔP (% ref.)', 'Permeability (L/m²·h·bar)', 'Total resistance (10¹³ m⁻¹)', 'Flag'],
          rows: a.all.map((q, i) => [q.t, 100 * q.Y, q.ndp, q.tcf, q.flux, q.npf, q.nsp, q.dpn, (q.kA * 1000) / a.area, q.Rtot / 1e13, a.flag[i] ? 'outlier' : '']) },
      ].flatMap((tb, i, all) => (i === all.length - 2 ? [...X.TB, tb] : [tb])),
      balances: (() => {
        const q = G[G.length - 1], r = q.row;
        return [
          { name: 'Water (m³/h): feed = permeate + concentrate', in: r.Qf, out: r.Qp + (r.Qf - r.Qp) },
          { name: 'Salt (kg/h): feed = permeate + concentrate (log-mean closure)', in: (r.Qf * q.tdsF) / 1000, out: (r.Qp * q.tdsP + (r.Qf - r.Qp) * ((r.Qf * q.tdsF - r.Qp * q.tdsP) / (r.Qf - r.Qp))) / 1000 },
          { name: 'Resistances (10¹³ m⁻¹): total = membrane + irreversible + reversible', in: Rnow / 1e13, out: Rnow >= a.Rm + a.Rirr ? (a.Rm + a.Rirr + a.Rrev) / 1e13 : Rnow / 1e13 },
          { name: 'Pressure (bar): feed = NDP + ΔP/2 + permeate + Δπ', in: r.Pf, out: q.ndp + q.dP / 2 + r.Pp + q.piF - piAstm(q.tdsP, r.T) },
          ...X.BAL,
        ];
      })(),
      outputs: out,
    };
  },

  mesh: [{ name: 'Time step of the biofilm projection', keys: ['nStep'], min: 4, note: 'The Monod biofilm model is integrated with fixed-step RK4; the study refines the step over the forecast horizon.',
    metrics: [{ label: 'Projected pressure drop at the horizon', unit: '% of ref.', get: (r) => r.outputs.bioDpAtHorizon }, { label: 'Biofilm-model days to the ΔP trigger', unit: 'd', get: (r) => r.outputs.bioDaysToTrigger }] },
  { name: 'Axial grid of the channel model', keys: ['nChan'], min: 8, note: 'Cells from feed inlet to concentrate outlet.', metrics: [{ label: 'Feed pressure from the outlet condition', unit: 'bar', get: (r) => r.outputs.channelFeedPressureBar ?? 0 }, { label: 'Channel recovery', unit: '–', get: (r) => r.outputs.channelRecovery ?? 0 }] },
  { name: 'Grid across the biofilm', keys: ['nzBio'], min: 4, note: 'Cells of the substrate profile between the membrane and the biofilm surface.', metrics: [{ label: 'Biofilm effectiveness factor', unit: '–', get: (r) => r.outputs.biofilmEffectiveness }] }],

  calibration: {
    note: 'Fit the fouling-resistance model to operating data from one cycle. Each row gives the time since the clean start with the operating point (temperature, flows, feed conductivity) and the measured feed pressure and differential pressure. The sample uses the first 66 days of the example log for calibration and days 70–94 — later in time, never earlier — for validation, so the validation is a genuine forecast without information leakage.',
    params: [{ key: 'Aclean', label: 'Clean water permeability', lo: 0.5, hi: 12 }, { key: 'kFoul', label: 'Fouling rate constant k', lo: 0, hi: 0.05 }, { key: 'dp0', label: 'Clean differential pressure', lo: 0.3, hi: 10 }, { key: 'kDp', label: 'Pressure-drop growth rate', lo: 0, hi: 0.02 }],
    columns: [{ key: 'tc', label: 'Time since clean start', unit: 'd' }, { key: 'Tx', label: 'Temperature', unit: '°C' }, { key: 'QfX', label: 'Feed flow', unit: 'm³/h' }, { key: 'QpX', label: 'Permeate flow', unit: 'm³/h' }, { key: 'CfX', label: 'Feed conductivity', unit: 'µS/cm' }, { key: 'PfM', label: 'Feed pressure', unit: 'bar' }, { key: 'dPM', label: 'Differential pressure', unit: 'bar' }],
    targets: [{ key: 'PfM', label: 'Feed pressure', unit: 'bar' }, { key: 'dPM', label: 'Differential pressure', unit: 'bar' }],
    model(v) { const r = pressureModel(modelPars(v), { tc: v.tc ?? 0, T: v.Tx ?? 25, Qf: v.QfX ?? 200, Qp: v.QpX ?? 150, Cf: v.CfX ?? 5800 }); return { PfM: r.Pf, dPM: r.dP }; },
    get sample() { return (this._s ||= calRows([3, 8, 14, 20, 27, 33, 40, 46, 52, 59, 66])); },
    get validationSample() { return (this._v ||= calRows([70, 74, 78, 82, 88, 91, 94])); },
  },

  verify() {
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    const d = D();
    // clean plant with varying temperature, salinity and flow: normalised indicators must stay constant
    const clean = analyseLog({ ...d, log: syntheticLog({ noise: 0, fouling: false, events: false, quality: false }) }), spread = (y) => (Math.max(...y) - Math.min(...y)) / mean(y);
    add('Normalised permeate flow is invariant to temperature, salinity and set-point changes', 0, spread(clean.npf), 5e-3, 'Clean synthetic plant, 180 days of seasonal variation, no noise');
    add('Normalised salt passage is invariant', 0, spread(clean.nsp), 2e-2, 'Same case (conductivity rounding limits the precision)');
    add('Normalised differential pressure is invariant', 0, spread(clean.ndp), 1e-2, 'Same case');
    add('Zero-fouling limit: fouling resistance vanishes', 0, (med(clean.Rt.slice(-10)) - clean.Rm) / clean.Rm, 5e-3, 'Resistance-in-series with no foulant');
    add('No false alarms or events on the clean plant', 0, clean.events.length + clean.spc.alarms.length, 0, 'Control charts and step detection stay silent');
    // exact recovery of an injected fault
    const inj = analyseLog({ ...d, log: syntheticLog({ noise: 0, fouling: false, events: false, quality: false, step: { day: 100, loss: 0.12 } }) });
    add('Injected 12 % permeability loss is recovered', 88, inj.now.npf, 0.3, 'Normalised permeate flow after the step, % of reference');
    add('…and located on the right day', 100, inj.events[0]?.t ?? -1, 1, 'Step detection');
    // the default log
    const a = analyseLog(d), cip = a.events.find((e) => e.type === 'cleaning'), flt = a.events.find((e) => e.type === 'integrity');
    add('Cleaning event of the example log is found', 120, cip?.t ?? -1, 1, 'Injected on day 120');
    add('Integrity fault of the example log is found and classified', 150, flt?.t ?? -1, 1, 'O-ring failure injected on day 150 (salt passage ×1.45)');
    add('Scaling episode is diagnosed in the first cycle', 1, a.cycles[0].top === 'scaling' ? 1 : 0, 0, 'Last-stage pressure drop, salt passage and flow loss rise together');
    add('All deliberate data faults are caught', 3, a.issues.filter((x) => /Missing|Range|Outlier/.test(x[1])).length >= 3 ? 3 : a.issues.length, 0, 'Missing value (day 37), conductivity spike (day 71), frozen temperature (day 133)');
    // Hermia: each law must identify itself and return its rate constant
    const tt = linspace(0, 60, 31);
    let hit = 0, kerr = 0;
    for (const law of Object.keys(HERMIA)) { const f = fitHermia(tt, tt.map((x) => hermia(law, 0.02 * x)), 0.7); if (f.best.law === law) hit++; kerr = Math.max(kerr, Math.abs(f.fits.find((q) => q.law === law).k - 0.02)); }
    add('Blocking-law selection identifies each generating law', 4, hit, 0, 'Exact synthetic data from the four Hermia laws');
    add('Blocking-law fit returns the rate constant', 0, kerr, 1e-5, 'k = 0.02 1/d');
    const hf = fitHermia(a.t.slice(0, 80), a.npf.slice(0, 80), 0.7);
    add('Calibration split is chronological (no leakage)', 1, hf.tTrainMax < hf.tValMin ? 1 : 0, 0, 'Latest training time precedes the earliest validation time');
    add('Kozeny–Carman specific resistance (hand calculation)', 8.4375e11, kozenyCarman(1e-6, 0.4, 2000), 1e6, '180(1−ε)/(ρ d² ε³) for d = 1 µm, ε = 0.4, ρ = 2000 kg/m³');
    add('ASTM osmotic pressure of 35 g/L at 25 °C', 28.70, piAstm(35000, 25), 0.01, '0.002654·C·T/(1000 − C/1000) bar');
    const ts = theilSen(tt, tt.map((x, i) => 100 - 0.1 * x + (i % 7 === 3 ? 25 : 0)));
    add('Robust trend ignores 15 % gross outliers', -0.1, ts.slope, 2e-3, 'Theil–Sen slope of a contaminated straight line');
    const b = biofilm({ X0: 0.05, mumax: 0.6, Ks: 60, S: 18, kd: 0.05, Xmax: 12 }, 120, 60);
    add('Biofilm integration matches the analytical logistic solution', 0, Math.max(...b.t.map((x, i) => Math.abs(b.X[i] - b.exact(x)))), 1e-4, 'RK4 with 60 steps over 120 days');
    add('Critical flux vanishes without cross-flow', 0, criticalFlux(1e-7, 0).J, 1e-12, 'Limiting case of no shear');
    const q = a.G[a.G.length - 1];
    // adsorption isotherms
    const Cs = [0.3, 0.7, 1.5, 3, 6, 12, 25], fL = fitIsotherms(Cs.map((C) => ({ C, q: langmuir(C, 6, 0.35) }))), fF = fitIsotherms(Cs.map((C) => ({ C, q: freundlich(C, 1.8, 2.5) })));
    add('Langmuir isotherm: parameters recovered from exact data', 0, Math.abs(fL.langmuir.qmax / 6 - 1) + Math.abs(fL.langmuir.K / 0.35 - 1) + (fL.best === 'langmuir' ? 0 : 1), 1e-4, 'q_max = 6 mg/m², K = 0.35 L/mg; the Langmuir form is selected');
    add('Freundlich isotherm: parameters recovered from exact data', 0, Math.abs(fF.freundlich.Kf / 1.8 - 1) + Math.abs(fF.freundlich.n / 2.5 - 1) + (fF.best === 'freundlich' ? 0 : 1), 1e-4, 'K_f = 1.8, n = 2.5; the Freundlich form is selected');
    add('Langmuir isotherm: Henry limit and saturation', 2, langmuir(1e-6, 6, 0.35) / (6 * 0.35 * 1e-6) + langmuir(1e9, 6, 0.35) / 6, 1e-5, 'q → q_max·K·C at low and q → q_max at high concentration');
    // combined laws
    add('Ho–Zydney law: pure pore blocking and pure cake limits', 0, Math.abs(hoZydney(20, 0.03, 1e12, 0) - hermia('complete', 0.6)) + Math.abs(hoZydney(20, 1e3, 0, 0.05) - hermia('cake', 1)), 1e-9, 'Infinite deposit resistance → e^(−βt); instant blocking → 1/√(1 + g·t)');
    add('Adsorption–pore blocking: limits', 0, Math.abs(adsorptionBlocking(30, 0.5, 0.01, 0) - Math.exp(-0.3)) + Math.abs(adsorptionBlocking(1e3, 0.5, 0, 0.2, 0.9) - 0.9 * 0.8), 1e-9, 'No adsorption → complete blocking; long times without blocking → ε₀·(1 − λ)');
    const fc = fitCombined(tt, tt.map((x) => adsorptionBlocking(x, 0.15, 0.004, 0.1)), 0.1);
    add('Adsorption–pore blocking: rate constants recovered', 0, Math.abs(fc.adsBlock.p[1] / 0.15 - 1) + Math.abs(fc.adsBlock.p[2] / 0.004 - 1), 1e-3, 'k_ads = 0.15 1/d, k_block = 0.004 1/d from exact data with the adsorptive loss given by the isotherm');
    // deposition and detachment
    add('Deposition–detachment: steady state a/k and initial slope a', 2, depositMass(1e4, 0.03, 0.02) / 1.5 + depositMass(1e-6, 0.03, 0.02) / 3e-8, 1e-5, 'dm/dt = a − k·m with a = 0.03 kg/m²·d and k = 0.02 1/d');
    const fdv = fitDeposition(tt, tt.map((x) => depositMass(x, 0.03, 0.02, 0.1)), 0.1);
    add('Deposition–detachment: fit recovers deposition flux and detachment rate', 0, Math.abs(fdv.a / 0.03 - 1) + Math.abs(fdv.k / 0.02 - 1), 1e-4, 'Exact synthetic deposit history with an initial deposit of 0.1 kg/m²');
    // threshold flux
    const thv = thresholdFlux(linspace(8, 36, 15).map((J) => ({ J, rate: 0.04 + 0.05 * Math.max(0, J - 21.3) })));
    add('Threshold flux is located from a flux-stepping test', 21.3, thv.Jth, 0.05, 'Exact hinge data: rate = 0.04 + 0.05·(J − 21.3)⁺');
    // polarisation and cake-enhanced osmotic pressure
    const mtv = massTransfer(0.15, 7.1e-4, 25, 1.5e-9), ReV = (997 * 0.15 * 1.236 * 7.1e-4) / viscosity(25), ScV = viscosity(25) / (997 * 1.5e-9);
    add('Mass-transfer coefficient (Schock–Miquel, hand calculation)', (0.065 * ReV ** 0.875 * ScV ** 0.25 * 1.5e-9) / (1.236 * 7.1e-4), mtv.k, 1e-12, 'Sh = 0.065·Re^0.875·Sc^0.25 with d_h = 1.236·h');
    const cev = cakeEnhancedCP({ J: 5e-6, k: 3e-5, D: 1.5e-9, eps: 0.4, rhoP: 2000, alpha: 1e15, piB: 5e5, dPf: 4e4 }), cev0 = cakeEnhancedCP({ J: 5e-6, k: 3e-5, D: 1.5e-9, eps: 0.4, rhoP: 2000, alpha: 1e15, piB: 0, dPf: 4e4 });
    add('Film theory: β = exp(J/k)', Math.exp(5e-6 / 3e-5), cev.beta, 1e-12, 'J = 18 L/m²·h, k = 30 µm/s');
    add('Fouling–polarisation coupling: hydraulic + osmotic loss equals the observed loss', 4e4, cev.hydraulic + cev.osmotic, 1e-3, 'Deposit mass solved from μJαm + π(β* − β) = ΔP_fouling, Pa');
    add('…and the hindered mass-transfer coefficient follows 1/k* = 1/k + δ(1 − ln ε²)/(Dε)', 1 / (1 / 3e-5 + (cev.delta * (1 - Math.log(0.16))) / (1.5e-9 * 0.4)), cev.kStar, 1e-15, 'Cake-enhanced concentration polarisation');
    add('Without osmotic pressure all the loss is hydraulic', 1, cev0.m / cev0.mHydraulicOnly, 1e-9, 'Limiting case π = 0');
    // biofilm with substrate transport
    const bpv = { Sb: 0.02, kL: 2e-5, Df: 5e-10, mumax: 7e-6, Ks: 50, Y: 0.5, Xf: 3e4, Lf: 5e-5, nz: 60 }, bsv = biofilmSubstrate(bpv), k1 = (bpv.mumax * bpv.Xf) / (bpv.Y * bpv.Ks), mm = Math.sqrt(k1 / bpv.Df);
    add('Biofilm substrate flux: first-order diffusion–reaction solution', bpv.Sb / (1 / bpv.kL + 1 / (bpv.Df * mm * Math.tanh(mm * bpv.Lf))), bsv.flux, 2e-10, 'J = S_b / (1/k_L + 1/(D·m·tanh(m·L))) for S ≪ K_s, g/m²·s');
    add('Zero-flux membrane wall: without consumption the substrate is uniform', bpv.Sb, biofilmSubstrate({ ...bpv, mumax: 0 }).Swall, 1e-12, 'No reaction and no flux through the wall → no gradient');
    add('Biofilm: film transfer equals consumption', bsv.flux, (bpv.Sb - bsv.Ssurf) / (bpv.Lf / 60 / (2 * bpv.Df) + 1 / bpv.kL), 1e-12, 'Flux through the liquid film = substrate consumed in the film');
    // channel model
    const cho = { L: 6, widths: [400, 200], h: 7.1e-4, Q0: 0.05, T: 25, Rm: 1e14, alpha: 1e15, Pp: 1e5, Pout: 9e5, pi0: 2.5e5, betaCP: 1.1, dp: 2e-8, phiB: 1e-6, omega: 0.15, cp: 2e-3, kDet: 0.05, days: 30, N: 40 }, chv = channelFouling(cho);
    add('Channel model: prescribed outlet pressure is met', 9e5, chv.Pout, 10, 'Shooting on the inlet pressure, Pa');
    add('Channel model: water balance', 0.05, chv.Qout + chv.perm, 1e-12, 'Feed = concentrate + permeate, m³/s');
    const chi = channelFouling({ ...cho, widths: [400], Rm: 1e30, days: 0 }), uI = 0.05 / (400 * 7.1e-4), ReI = (997 * uI * 1.236 * 7.1e-4) / viscosity(25);
    add('Channel model: impermeable limit gives the spacer friction loss', ((6.23 / ReI ** 0.3) * 997 * uI * uI * 6) / (2 * 1.236 * 7.1e-4), chi.Pin - chi.Pout, 10, 'Δp = f·ρu²·L/(2·d_h) with f = 6.23·Re^−0.3, Pa');
    add('Channel model: friction factor calibrated to a measured pressure drop', 2.5e5, (() => { const q = channelFouling({ ...cho, dPtarget: 2.5e5, days: 0 }); return q.Pin - q.Pout; })(), 500, 'Bisection on the friction multiplier, Pa');
    // state-space model
    const kl = kalmanTrend(tt, tt.map((x) => 100 - 0.07 * x), { qS: 1e-6, r: 0.25 });
    add('Kalman filter recovers the slope of a noise-free decline', -0.07, kl.slope[kl.slope.length - 1], 1e-4, 'Local linear trend model');
    const g2 = rng(5), kn = kalmanTrend(linspace(0, 399, 400), linspace(0, 399, 400).map(() => g2.normal(0, 1)), { qL: 0.2, qS: 0, r: 1, x0: [0, 0], P0: [1, 0, 0] }), pInf = (0.2 + Math.sqrt(0.04 + 0.8)) / 2;
    add('Kalman gain converges to the algebraic Riccati solution', pInf / (pInf + 1), kn.gain[0], 1e-9, 'Local-level model with q = 0.2, r = 1: K = P/(P + r), P = (q + √(q² + 4qr))/2');
    // streaming monitor
    const cl = syntheticLog({ noise: 0.5, events: false, quality: false }), mo = createMonitor({ ...d, kfR: 1 });
    for (const row of cl) mo.push(row);
    const ser = mo.state.series, kb = kalmanTrend(ser.map((p) => p.t), ser.map((p) => p.npf), { qS: 1e-6, r: 1, P0: [4, 0, 1e-2] });
    add('Streaming monitor reproduces the batch Kalman filter', 0, Math.abs(mo.state.last.level - kb.level[kb.level.length - 1]) + Math.abs(mo.state.last.slope - kb.slope[kb.slope.length - 1]), 1e-9, 'Records pushed one at a time give the same state as filtering the whole series');
    const m2 = createMonitor(d);
    for (const row of defaultLog()) m2.push(row);
    add('Streaming monitor rejects the faulty records and detects the cleaning on line', 120, m2.state.rejected === 2 ? m2.state.alerts.find((x) => x.type === 'cleaning')?.t ?? -1 : -1, 1, 'Missing value and frozen temperature rejected; cleaning injected on day 120');
    // mechanistic model with learned residual
    const g3 = rng(9), Xs = tt.map(() => [25 + g3.normal(0, 2), 18 + g3.normal(0, 1)]), me = tt.map((x) => 100 * hermia('intermediate', 0.004 * x)), ys = me.map((m, i) => m + 0.8 * (Xs[i][0] - 25) - 0.3 * (Xs[i][1] - 18)), hv = hybridResidual(Xs, ys, me, 20, ['T', 'flux']);
    add('Learned residual removes a systematic effect out of sample', 1, hv.rmseHybTest < 0.05 && hv.rmseMechTest > 1 ? 1 : 0, 0, `Synthetic temperature and flux effects: test RMSE ${fmt(hv.rmseMechTest, 3)} % mechanistic, ${fmt(hv.rmseHybTest, 2)} % with the learned residual`);
    add('Train/test split of the learned residual is chronological', 20, hv.nTrain, 0, 'The first 20 records train, the following ones test');
    // scaling mass and dashboard
    add('Scale mass grows from its initial value only beyond saturation', 0, Math.abs(scaleMass(50, { m0: 2, k: 0.2, S: 1.5, mc: 10 }).m - (2 + 0.2 * 0.25 * 50)) + Math.abs(scaleMass(50, { m0: 2, k: 0.2, S: 0.9, mc: 10 }).m - 2), 1e-12, 'm = m₀ + k·(S − 1)²·t for S > 1, constant otherwise');
    const cleanRun = suite.run({ ...d, log: syntheticLog({ noise: 0, fouling: false, events: false, quality: false }) }, {});
    add('Dashboard: a clean plant scores a health index near 100', 100, cleanRun.outputs.healthIndex, 3, 'No fouling, no faults, complete data');
    add('Darcy law is recovered from the resistance', q.flux, ((q.ndp * q.tcf * 1e5) / (MU25 * q.Rtot)) * 3.6e6, 1e-9, 'J = NDP·TCF / (μ₂₅·R_total), resistance referred to 25 °C');
    return C;
  },
};

const HELP = {
  toc: 'Dissolved organic carbon of the RO feed; drives adsorption and organic fouling.', age0: 'Years in service before the first row of the log; added to the remaining-life estimate.', biocide: 'Lowers the biofouling evidence score.', refMode: 'What the normalised indicators are compared with.', nBase: 'Number of first valid rows averaged as the clean reference.',
  dQf: 'Feed flow at the design point.', dQp: 'Permeate flow at the design point.', dPf: 'Feed pressure at the design point.', dPc: 'Concentrate pressure at the design point.', dPp: 'Permeate back-pressure at the design point.', dT: 'Feed temperature at the design point.', dCf: 'Feed conductivity at the design point.', dCp: 'Permeate conductivity at the design point.',
  outlierSigma: 'Points further than this many robust standard deviations from the running median are excluded.', stepWin: 'Rows on each side of a candidate step that are compared.', stepNPF: 'Smallest sudden change of the normalised flow that counts as an event.', stepNSP: 'Smallest sudden change of the salt passage that counts as an event.', stepNDP: 'Smallest sudden change of the pressure drop that counts as an event.',
  ewmaLambda: 'Smaller values smooth more and detect smaller drifts.', ewmaL: 'Width of the control limits in standard deviations.', cusumK: 'Drift smaller than this (in σ) is ignored.', cusumH: 'Alarm when the cumulative sum exceeds this value.', trigNPF: 'Membrane suppliers recommend cleaning at a 10–15 % drop.', trigDP: 'Typical 15 %.', trigSP: 'Typical 5–10 %.',
  refAfter: 'The original reference is stricter once irreversible fouling has accumulated.', eolLoss: 'Membranes are usually replaced at 20–30 % irreversible loss.', revFrac: 'Only used when the log contains no cleaning.', Aclean: 'Permeability of the clean membrane at 25 °C.', kFoul: 'Rate constant of the selected blocking law.', law: 'Shape of the permeability decline with time.',
  dp0: 'Differential pressure of the clean train at the reference flow.', kDp: 'Relative rise of the pressure drop per day.', qRef: 'Average of feed and concentrate flow at which dp0 applies.', sp0: 'Salt passage of the clean membrane.', PpRef: 'Back-pressure on the permeate side.',
  dpNm: 'Size of the dominant colloids; sets the specific cake resistance and the critical flux.', eps: 'Void fraction of the deposit.', rhoP: 'Density of the foulant particles.', uCross: 'Mean velocity in the feed channel.', hChan: 'Feed-spacer thickness.', lChan: 'Length of the flow path through one pressure vessel.', phiB: 'Volume fraction of particles in the feed.',
  mumax: 'Maximum specific growth rate of the biofilm.', KsAoc: 'Assimilable carbon at which growth is half its maximum.', kd: 'Combined decay and detachment of biomass.', Xmax: 'Largest areal biomass the spacer channel can hold.', X0: 'Biomass left after a cleaning.', betaBio: 'Sensitivity of the pressure drop to biomass.', horizon: 'Length of the biofilm and deposit forecasts.', nStep: 'Steps of the biofilm integrations.',
};
for (const g of suite.inputs) for (const f of g.fields) if (!f.help && HELP[f.key]) f.help = HELP[f.key];

/** Rows for calibration/validation taken from the example log (gradual-fouling period before the scaling episode). */
function calRows(days) {
  const log = defaultLog();
  return days.map((d) => log.find((r) => r.t === d)).filter((r) => r && NEED.every((k) => isNum(r[k]))).map((r) => ({ tc: r.t, Tx: r.T, QfX: r.Qf, QpX: r.Qp, CfX: r.Cf, PfM: r.Pf, dPM: +(r.Pf - r.Pc).toFixed(2) }));
}
/** Second example: single pressure-drop signal with exponential (biological) growth and no interstage pressure. */
let _bio = null;
function bioLog() {
  if (_bio) return _bio;
  const g = rng(11), rows = [];
  for (let t = 0; t < 75; t++) {
    const T = 29 + 2 * Math.sin(t / 20) + 0.2 * g.normal(0, 1), X = 12 / (1 + (12 / 0.05 - 1) * Math.exp(-0.085 * t)), Qp = 150 * (1 + 0.004 * g.normal(0, 1)), Qf = Qp / 0.75;
    const a = 1 / (1 + 0.012 * X), ndp = (Qp * 1000) / 8035 / (3.6 * a * tcfM(T)), dP = 2.8 * (1 + 0.09 * X) * ((Qf - Qp / 2) / 125) ** 1.5 * (viscosity(T) / viscosity(25)) ** 0.3;
    const Cfb = (3200 * Math.log(4)) / 0.75, tdsP = 0.012 * tcfM(T) * Cfb, Pf = ndp + dP / 2 + 1 + piAstm(Cfb, T) - piAstm(tdsP, T) + 0.03 * g.normal(0, 1);
    rows.push({ t, Pf: +Pf.toFixed(2), Pi: null, Pc: +(Pf - dP + 0.015 * g.normal(0, 1)).toFixed(2), Pp: 1, Qf: +Qf.toFixed(1), Qp: +Qp.toFixed(1), Cf: +(conductivityFromTDS(3200) * (1 + 0.005 * g.normal(0, 1))).toFixed(0), Cp: +(conductivityFromTDS(tdsP) * (1 + 0.01 * g.normal(0, 1))).toFixed(1), T: +T.toFixed(1) });
  }
  return (_bio = rows);
}

export default suite;
