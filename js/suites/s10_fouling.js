// Suite 10 — Fouling and membrane-performance monitoring.
// Operating logs are validated, normalised (ASTM D4516 style: net driving pressure, temperature-correction
// factor, flow-corrected differential pressure), converted to Darcy resistances, and analysed for trends,
// sudden changes and anomalies. Blocking laws, cake filtration, critical flux and Monod biofilm growth give
// the mechanistic interpretation; rule-based evidence scores name the likely foulant; robust trends give the
// days to the next cleaning and the remaining membrane life with uncertainty bands.
import { clamp, linspace, logspace, sum, mean, quantile, rng, fmt, rk4, interp1, levenbergMarquardt, isNum } from '../core/num.js';
import { tcf, viscosity, conductivityFromTDS, KELVIN } from '../core/props.js';

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

const D = () => Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));
const modelPars = (v) => ({ area: v.area, Aclean: v.Aclean, kFoul: v.kFoul, law: v.law, dp0: v.dp0, kDp: v.kDp, qRef: v.qRef, mFlow: v.mFlow, sp0: v.sp0, Pp: v.PpRef });

const suite = {
  id: 'fouling', num: 10, title: 'Fouling & Membrane-Performance Monitoring', short: 'Fouling monitor', icon: '📈',
  tagline: 'Normalise operating data, separate real deterioration from operating changes, diagnose the foulant and forecast cleaning and membrane life.',
  description: 'Reads an operating log (pressures, flows, conductivities, temperature), validates it and normalises permeate flow, salt passage and differential pressure to reference conditions, so that temperature, salinity and set-point changes no longer hide or mimic fouling. Darcy resistances, blocking-law fits, cake-filtration and critical-flux estimates and a Monod biofilm model interpret the decline; robust trends, EWMA/CUSUM charts and step detection separate gradual from sudden changes; evidence rules score the likely foulant; and the trends are extrapolated to the cleaning triggers and to end of membrane life with uncertainty bands.',
  guide: [
    'Paste or import the operating log on the Inputs tab (one row per day or shift). The built-in 180-day example contains gradual fouling, a scaling episode, a cleaning and an O-ring failure.',
    'Enter the membrane area and choose the reference: the first days of the log (clean baseline) or the design point.',
    'Add what you know about the feed: SDI, MFI, organic carbon, scaling margin from suite 2.',
    'Run. Check the data-validation table first, then the three normalised trends, the detected events, the diagnosis and the forecast.',
  ],
  implemented: ['resistance-in-series', 'darcy membrane', 'cake-filtration', 'hermia complete-blocking', 'hermia standard-blocking', 'hermia intermediate-blocking', 'hermia cake-filtration', 'pore-blocking equations', 'cake-compressibility', 'kozeny–carman', 'deposition-rate', 'critical-flux', 'concentration-polarization equation', 'normalized permeate-flow', 'normalized salt-passage', 'normalized pressure-drop',
    'cake–pore-blocking', 'fouling–scaling', 'resistance–compressibility', 'mechanistic–statistical monitoring', 'remaining-useful-life/prognostic',
    'clean-membrane resistance', 'initial permeability', 'initial deposit thickness', 'initial biofilm biomass', 'baseline normalized performance', 'inlet foulant concentration', 'permeate-flux boundary', 'transmembrane-pressure boundary', 'wall shear condition', 'cleaning/reset',
    'data validation and cleaning', 'membrane-performance normalisation', 'permeate-flow monitoring', 'salt-passage and rejection monitoring', 'pressure-drop monitoring', 'flux monitoring', 'organic-fouling assessment', 'biological-fouling assessment', 'colloidal-fouling assessment', 'inorganic-scaling assessment', 'membrane-ageing analysis', 'fouling-resistance modelling', 'cleaning-in-place monitoring', 'cleaning-effectiveness assessment', 'anomaly detection', 'trend analysis', 'membrane-health indicators', 'fault diagnosis', 'remaining-useful-life prediction', 'predictive maintenance', 'alarms', 'historical performance comparison'],
  equationsNote: 'Normalisation follows the ASTM D4516 approach for one train: log-mean feed–brine concentration, osmotic pressure from TDS, an Arrhenius temperature-correction factor and a flow-exponent correction of the pressure drop. It assumes steady operating points; start-ups, flushing periods and rows with large recovery changes should be removed. Conductivity is converted to TDS with a generic correlation — a site-specific factor improves salt-passage accuracy. Foulant scores are evidence rules, not probabilities from a trained classifier, and should be confirmed by autopsy or targeted water analysis. Blocking laws are fitted to the normalised (constant-pressure-equivalent) flux decline. Forecasts extrapolate the current robust trend; they do not anticipate operating changes. Stage-resolved diagnosis needs the interstage pressure column.',

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
    { group: 'Time stepping', tab: 'mesh', help: 'Fixed-step fourth-order Runge–Kutta integration of the biofilm model over the forecast horizon.', fields: [
      { key: 'nStep', label: 'Time steps over the horizon', unit: '', value: 60, min: 4, max: 5000, step: 1 },
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
    const ev = events.map((e) => ({ x: e.t, label: e.type === 'cleaning' ? 'CIP' : e.type === 'integrity' ? 'fault' : 'step' }));
    const proj = linspace(0, Math.min(v.horizon, Math.max(20, Math.min(1.4 * a.daysToCleaning, v.horizon))), 12), last = now.t, line = (y0, r) => proj.map((d) => y0 + r * d);
    const pm = modelPars(v), pred = G.map((q) => { const cy = cycles.find((c) => q.t >= c.t0 && q.t <= c.t1) || cur; return pressureModel(pm, { tc: q.t - cy.t0, T: q.row.T, Qf: q.row.Qf, Qp: q.row.Qp, Cf: q.row.Cf, Pp: q.row.Pp }); });
    const probs = Object.keys(FOULANTS), recipe = FOULANTS[a.integrity && cur.topFoulant && cur.loss < 3 && now.ndp < trig.ndp.limit ? 'integrity' : cur.topFoulant].cip, recipeFor = a.integrity && cur.loss < 3 && now.ndp < trig.ndp.limit ? 'integrity' : cur.topFoulant;
    const Rnow = med(a.Rt.slice(-5)), dstr = (x) => (x >= 3650 ? '> 3650' : fmt(x, 3));
    const out = {
      foulingRate: foulRate, daysToCleaning: a.daysToCleaning, daysToCleaningLow: trig[a.first].lo, daysToCleaningHigh: trig[a.first].hi, cleaningsPerYear: a.cleaningsPerYear, membraneLife: life, remainingLife: a.rul, remainingLifeLow: a.rulLo, remainingLifeHigh: a.rulHi,
      normPermeability: now.npf / 100, normSaltPassage: now.nsp / 100, normDP: now.ndp / 100, dominantFoulant: FN(a.dominant), currentIssue: FN(curIssue), integrityFault: a.integrity, irreversibleLoss: a.irrNow, cleaningRecovery: a.lastClean ? a.lastClean.recovery : null,
      limitingTrigger: trig[a.first].name, permeability: (a.ref.kA * now.npf * 10) / a.area, cleanPermeability: (a.ref.kA * 1000) / a.area, Rm: a.Rm, Rreversible: a.Rrev, Rirreversible: a.Rirr, criticalFluxLMH: jc, bestBlockingLaw: hf ? HERMIA[hf.best.law].name : 'not fitted', bioDaysToTrigger: bioDays, bioDpAtHorizon: bioDp[bioDp.length - 1], dataIssues: a.issues.length, events: events.map((e) => ({ day: e.t, type: e.type })),
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
      ],
      balances: (() => {
        const q = G[G.length - 1], r = q.row;
        return [
          { name: 'Water (m³/h): feed = permeate + concentrate', in: r.Qf, out: r.Qp + (r.Qf - r.Qp) },
          { name: 'Salt (kg/h): feed = permeate + concentrate (log-mean closure)', in: (r.Qf * q.tdsF) / 1000, out: (r.Qp * q.tdsP + (r.Qf - r.Qp) * ((r.Qf * q.tdsF - r.Qp * q.tdsP) / (r.Qf - r.Qp))) / 1000 },
          { name: 'Resistances (10¹³ m⁻¹): total = membrane + irreversible + reversible', in: Rnow / 1e13, out: Rnow >= a.Rm + a.Rirr ? (a.Rm + a.Rirr + a.Rrev) / 1e13 : Rnow / 1e13 },
          { name: 'Pressure (bar): feed = NDP + ΔP/2 + permeate + Δπ', in: r.Pf, out: q.ndp + q.dP / 2 + r.Pp + q.piF - piAstm(q.tdsP, r.T) },
        ];
      })(),
      outputs: out,
    };
  },

  mesh: { name: 'Time step of the biofilm projection', keys: ['nStep'], min: 4, note: 'The Monod biofilm model is integrated with fixed-step RK4; the study refines the step over the forecast horizon.',
    metrics: [{ label: 'Projected pressure drop at the horizon', unit: '% of ref.', get: (r) => r.outputs.bioDpAtHorizon }, { label: 'Biofilm-model days to the ΔP trigger', unit: 'd', get: (r) => r.outputs.bioDaysToTrigger }] },

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
    add('Darcy law is recovered from the resistance', q.flux, ((q.ndp * q.tcf * 1e5) / (MU25 * q.Rtot)) * 3.6e6, 1e-9, 'J = NDP·TCF / (μ₂₅·R_total), resistance referred to 25 °C');
    return C;
  },
};

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
