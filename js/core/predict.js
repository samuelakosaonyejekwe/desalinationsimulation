// Predictions for any suite, made with the suite's own engine (no black box): the user asks "what if", the
// engine is solved at the operating points needed, and the answer is given as numbers, charts and plain sentences.
//   1 · What if one input changes   — sweep of one input, response of every key result, elasticities, optima
//   2 · Two inputs together          — response map of one result over two inputs, best and worst corner
//   3 · How sure is the prediction   — uncertain inputs sampled (Latin hypercube), P10 / P50 / P90 of every result
// Each run goes through the same background solver as the Run button and can be cancelled.
import { h, btn, fill, dataTable, toast, help } from './ui.js';
import { plotCard } from './plot.js';
import { fmt, lhs, histogram, isNum } from './num.js';
import { download } from './io.js';

const NUMERIC = (f) => (f.type === undefined || f.type === 'number') && typeof f.value === 'number' && Number.isFinite(f.value);
const pct = (a, b) => (Math.abs(a) > 1e-12 ? (100 * (b - a)) / Math.abs(a) : NaN);
const fmtPct = (x) => (Number.isFinite(x) ? `${x >= 0 ? '+' : ''}${fmt(x, 3)} %` : 'n/a');
const kpiMap = (res) => Object.fromEntries((res?.kpis || []).filter((k) => typeof k.value === 'number' && Number.isFinite(k.value)).map((k) => [k.label, k]));
const quantile = (a, q) => { const s = [...a].sort((x, y) => x - y); if (!s.length) return NaN; const p = (s.length - 1) * q, i = Math.floor(p); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (p - i); };
const unitOf = (f) => (f?.unit ? ' ' + f.unit : '');

/**
 * Build the Predict tab.
 * env = { suite, fields, values(), solveAt(v, progress) -> Promise<res>, cancel(), lastResult(), runLabel }
 */
export function predictTab(env) {
  const { suite, fields } = env;
  const base = env.values(), last = env.lastResult();
  const inputs = fields.filter(NUMERIC).filter((f) => !/^_/.test(f.key));
  const tabOf = new Map(suite.inputs.flatMap((g) => g.fields.map((f) => [f.key, g.tab || 'inputs'])));
  // physical inputs first (data tab, then model settings); numerical controls (seeds, iteration counts, grids, tolerances) last
  const numericalCtl = /seed|iteration|population|generation|toleran|samples|sample size|cells|nodes|grid|mesh|time step|steps\b|restarts|particles|bins|resolution/i;
  const hint = suite.predict || {}, firstKeys = hint.inputs || [];
  const order = (f) => (firstKeys.includes(f.key) ? -1 + firstKeys.indexOf(f.key) / 1000 : numericalCtl.test(f.label + ' ' + f.key) ? 2 : tabOf.get(f.key) === 'inputs' ? 0 : 1);
  const pref = inputs.filter((f) => tabOf.get(f.key) !== 'mesh').map((f, i) => [f, i]).sort((a, b) => order(a[0]) - order(b[0]) || a[1] - b[1]).map(([f]) => f);
  const outs0 = Object.values(kpiMap(last)), outFirst = (suite.predict?.outputs || []).map((l) => outs0.find((k) => k.label === l || k.label.startsWith(l))).filter(Boolean);
  const outs = [...new Set([...outFirst, ...outs0])];
  const secs = Math.max(0.05, (last?._ms || 1500) / 1000);
  const box = h('div', { class: 'groups predict' });

  if (!last) {
    box.append(h('p', { class: 'summary' }, `Predictions use this suite's own engine. Run it once first (“${env.runLabel}”) so the key results to predict are known and the time per solution can be estimated.`));
    return box;
  }
  if (!pref.length || !outs.length) { box.append(h('p', { class: 'summary' }, 'This suite has no numeric inputs or numeric key results to predict.')); return box; }

  box.append(h('p', { class: 'summary' }, `Ask what the ${suite.short || suite.title} results would be under other conditions. Every prediction below is computed by solving this suite's engine at the points needed — the same equations as the main run, not a fitted black box — so it is as valid as the model itself inside the input ranges shown. One solution of the current case takes about ${fmt(secs, 2)} s.`));

  // ---- shared controls -------------------------------------------------------------------------------------
  const inputSel = (id, pick = 0, label = 'Input') => h('select', { id, 'aria-label': label }, pref.map((f, i) => h('option', { value: f.key, selected: i === pick }, `${f.label}${unitOf(f)} — now ${fmt(base[f.key])}`)));
  const field = (key) => pref.find((f) => f.key === key);
  const rangeOf = (f) => {
    const v = base[f.key], lo0 = isNum(f.min) ? f.min : -Infinity, hi0 = isNum(f.max) ? f.max : Infinity;
    let lo = v === 0 ? -1 : v * 0.7, hi = v === 0 ? 1 : v * 1.3; if (lo > hi) [lo, hi] = [hi, lo];
    return [Math.max(lo, lo0), Math.min(hi, hi0)];
  };
  const num = (v, label) => h('input', { type: 'number', step: 'any', value: +(+v).toPrecision(6), 'aria-label': label, style: 'max-width:130px' });
  const outChecks = h('div', { class: 'chips' }, outs.map((k, i) => { const id = `po_${i}`; return h('label', { class: 'chip on', for: id, style: 'cursor:pointer' }, h('input', { type: 'checkbox', id, checked: outFirst.length ? outFirst.includes(k) : i < 8, value: k.label, style: 'margin-right:6px' }), k.label + (k.unit ? ` (${k.unit})` : '')); }));
  const chosenOuts = () => [...outChecks.querySelectorAll('input:checked')].map((c) => c.value);
  let busy = false;
  const guard = async (btnEl, n, work) => {
    if (busy) return toast('A prediction is already running.', 'warn');
    if (n * secs > 240 && !confirm(`This needs ${n} solutions, about ${Math.round((n * secs) / 60)} minutes. Continue?`)) return;
    busy = true; btnEl.disabled = true; cancel.hidden = false;
    try { await work(); } catch (e) { if (e.cancelled) toast('Prediction cancelled.', 'warn'); else { console.error(e); toast('Prediction failed: ' + (e.message || e), 'bad', 8000); } }
    finally { busy = false; btnEl.disabled = false; cancel.hidden = true; prog.hidden = true; }
  };
  const prog = h('div', { class: 'progress', hidden: true }, h('i')), progTxt = h('span', { class: 'run-status' });
  const step = (i, n, msg) => { prog.hidden = false; prog.firstChild.style.width = Math.round((100 * i) / n) + '%'; progTxt.textContent = `${msg} — solution ${i} of ${n}`; };
  const cancel = btn('✕  Cancel', () => env.cancel(), 'ghost danger', 'Stop the prediction'); cancel.hidden = true;
  const solveAll = async (points, msg) => { const res = []; for (let i = 0; i < points.length; i++) { step(i, points.length, msg); res.push(await env.solveAt({ ...base, ...points[i] })); } step(points.length, points.length, msg); return res; };

  box.append(h('fieldset', { class: 'group' }, h('legend', null, 'Results to predict', help('Tick the key results you want predicted. They are the numeric results of the last run.')), outChecks, h('div', { class: 'row-tools' }, cancel, progTxt), prog));

  // ---- 1 · one input -----------------------------------------------------------------------------------------
  {
    const sel = inputSel('pr1_in', 0), lo = num(0, 'From'), hi = num(0, 'To'), n = h('select', { 'aria-label': 'Points' }, [5, 7, 9, 13].map((x) => h('option', { value: x, selected: x === 7 }, `${x} points`)));
    const at = num(base[pref[0].key], 'Predict at'), out = h('div');
    const sync = () => { const f = field(sel.value), [a, b] = rangeOf(f); lo.value = +a.toPrecision(5); hi.value = +b.toPrecision(5); at.value = +(+base[f.key]).toPrecision(6); };
    sel.addEventListener('change', sync); sync();
    const go = btn('Predict the response', () => guard(go, +n.value, async () => {
      const f = field(sel.value), a = +lo.value, b = +hi.value, N = +n.value, wanted = chosenOuts();
      if (!(b > a)) return toast('“To” must be larger than “From”.', 'warn');
      const xs = Array.from({ length: N }, (_, i) => a + ((b - a) * i) / (N - 1));
      const res = await solveAll(xs.map((x) => ({ [f.key]: x })), `Varying ${f.label}`);
      const ks = res.map(kpiMap), b0 = kpiMap(last), rows = [], plots = [], sentences = [];
      for (const lbl of wanted) {
        const ys = ks.map((k) => k[lbl]?.value ?? NaN), unit = b0[lbl]?.unit || '';
        if (!ys.some(Number.isFinite)) continue;
        plots.push({ type: 'line', title: `${lbl} versus ${f.label}`, xlabel: f.label + unitOf(f), ylabel: lbl + (unit ? ` (${unit})` : ''), height: 240, series: [{ name: lbl, x: xs, y: ys, mode: 'both' }], vlines: [{ x: base[f.key], label: 'now' }],
          note: `Each point is a full solution of the engine with ${f.label} set to that value and every other input as it is now.` });
        const first = ys.find(Number.isFinite), lastY = [...ys].reverse().find(Number.isFinite);
        const iMin = ys.indexOf(Math.min(...ys.filter(Number.isFinite))), iMax = ys.indexOf(Math.max(...ys.filter(Number.isFinite)));
        // local elasticity at the current value: % change of the result per % change of the input
        const j = xs.findIndex((x) => x >= base[f.key]), j0 = Math.max(0, Math.min(xs.length - 2, j - 1));
        const el = Number.isFinite(ys[j0]) && Number.isFinite(ys[j0 + 1]) && ys[j0] && xs[j0] ? (((ys[j0 + 1] - ys[j0]) / ((ys[j0 + 1] + ys[j0]) / 2)) / ((xs[j0 + 1] - xs[j0]) / ((xs[j0 + 1] + xs[j0]) / 2))) : NaN;
        rows.push([lbl, unit, first, b0[lbl]?.value, lastY, pct(first, lastY), el, iMin > 0 && iMin < N - 1 ? `minimum near ${fmt(xs[iMin])}` : iMax > 0 && iMax < N - 1 ? `maximum near ${fmt(xs[iMax])}` : ys[N - 1] > ys[0] ? 'rises' : ys[N - 1] < ys[0] ? 'falls' : 'flat']);
        const dir = lastY > first ? 'rises' : lastY < first ? 'falls' : 'does not change';
        sentences.push(`${lbl} ${dir} from ${fmt(first)} to ${fmt(lastY)}${unit ? ' ' + unit : ''} (${fmtPct(pct(first, lastY))}) as ${f.label} goes from ${fmt(a)} to ${fmt(b)}${unitOf(f)}` + (Number.isFinite(el) ? `; around the current value a 1 % change of ${f.label} moves it by about ${fmt(el, 2)} %` : '') + (iMin > 0 && iMin < N - 1 ? `; it has a minimum near ${fmt(xs[iMin])}${unitOf(f)}` : iMax > 0 && iMax < N - 1 ? `; it has a maximum near ${fmt(xs[iMax])}${unitOf(f)}` : '') + '.');
      }
      const warn = (isNum(f.min) && a < f.min) || (isNum(f.max) && b > f.max) || (f.typical && (a < f.typical[0] || b > f.typical[1])) ? h('p', { class: 'warn warn-inline' }, `Part of this range lies outside the typical range of ${f.label}${f.typical ? ` (${f.typical[0]}–${f.typical[1]})` : ''}; predictions there are extrapolations of the model.`) : null;
      fill(out, h('div', { class: 'explain' }, h('h3', null, 'What the prediction says'), h('ul', null, sentences.map((s) => h('li', null, s)))), warn,
        dataTable({ title: `Response to ${f.label}`, columns: ['Result', 'Unit', `At ${fmt(a)}`, 'Now', `At ${fmt(b)}`, 'Change over the range %', 'Elasticity at current value', 'Shape'], rows,
          note: 'Elasticity = % change of the result for a 1 % change of the input around its current value (a value of 2 means the result changes twice as fast, in relative terms, as the input).' }),
        dataTable({ title: 'All predicted points', columns: [f.label + unitOf(f), ...wanted], rows: xs.map((x, i) => [x, ...wanted.map((l) => ks[i][l]?.value ?? null)]) }),
        h('div', { class: 'plots' }, plots.map((p) => plotCard(p, { onDownload: download }))));
    }), 'primary');
    const goAt = btn('Predict at this value', () => guard(goAt, 1, async () => {
      const f = field(sel.value), x = +at.value; if (!Number.isFinite(x)) return toast('Enter a value.', 'warn');
      const [r] = await solveAll([{ [f.key]: x }], `${f.label} = ${fmt(x)}`), k = kpiMap(r), b0 = kpiMap(last);
      const rows = chosenOuts().filter((l) => k[l]).map((l) => [l, k[l].unit || '', b0[l]?.value, k[l].value, k[l].value - (b0[l]?.value ?? NaN), pct(b0[l]?.value, k[l].value)]);
      const big = rows.filter((r2) => Number.isFinite(r2[5]) && Math.abs(r2[5]) >= 0.01).sort((p, q) => Math.abs(q[5]) - Math.abs(p[5])).slice(0, 3);
      fill(out, h('div', { class: 'explain' }, h('h3', null, `Prediction with ${f.label} = ${fmt(x)}${unitOf(f)} (now ${fmt(base[f.key])})`),
        h('p', null, big.length ? `The results that move most: ${big.map((r2) => `${r2[0]} ${fmtPct(r2[5])} (${fmt(r2[2])} → ${fmt(r2[3])}${r2[1] ? ' ' + r2[1] : ''})`).join('; ')}.` : 'None of the selected results changes by 0.01 % or more.'),
        (r.warnings || []).filter((w) => w.level === 'bad' || w.level === 'warn').length ? h('ul', null, r.warnings.filter((w) => w.level === 'bad' || w.level === 'warn').slice(0, 6).map((w) => h('li', { class: 'warn-inline' }, w.msg))) : null),
        dataTable({ title: 'Predicted against current', columns: ['Result', 'Unit', 'Now', 'Predicted', 'Change', 'Change %'], rows }));
    }), '');
    box.append(h('fieldset', { class: 'group' }, h('legend', null, '1 · What happens if one input changes?', help('Solves the engine across a range of one input with everything else unchanged, then states how each result responds, how sensitive it is (elasticity) and whether it has an optimum inside the range.')),
      h('div', { class: 'row-tools' }, h('label', { class: 'inline' }, 'Input ', sel), h('label', { class: 'inline' }, 'from ', lo), h('label', { class: 'inline' }, 'to ', hi), n, go),
      h('div', { class: 'row-tools' }, h('label', { class: 'inline' }, 'or predict the results at ', at), goAt), out));
  }

  // ---- 2 · two inputs ----------------------------------------------------------------------------------------
  {
    const s1 = inputSel('pr2_a', 0, 'First input'), s2 = inputSel('pr2_b', Math.min(1, pref.length - 1), 'Second input');
    const target = h('select', { 'aria-label': 'Result' }, outs.map((k, i) => h('option', { value: k.label, selected: i === 0 }, k.label + (k.unit ? ` (${k.unit})` : ''))));
    const goal = h('select', { 'aria-label': 'Goal' }, h('option', { value: 'min' }, 'lower is better'), h('option', { value: 'max' }, 'higher is better'));
    const n = h('select', { 'aria-label': 'Grid' }, [4, 5, 6, 7].map((x) => h('option', { value: x, selected: x === 5 }, `${x} × ${x} points`)));
    const out = h('div');
    const go = btn('Map the response', () => guard(go, (+n.value) ** 2, async () => {
      const fa = field(s1.value), fb = field(s2.value), N = +n.value, lbl = target.value;
      if (fa.key === fb.key) return toast('Choose two different inputs.', 'warn');
      const [a0, a1] = rangeOf(fa), [b0, b1] = rangeOf(fb), xa = Array.from({ length: N }, (_, i) => a0 + ((a1 - a0) * i) / (N - 1)), xb = Array.from({ length: N }, (_, i) => b0 + ((b1 - b0) * i) / (N - 1));
      const pts = []; for (const y of xb) for (const x of xa) pts.push({ [fa.key]: x, [fb.key]: y });
      const res = await solveAll(pts, `Mapping ${lbl}`), zs = res.map((r) => kpiMap(r)[lbl]?.value ?? NaN), Z = xb.map((_, j) => xa.map((__, i) => zs[j * N + i]));
      const ok = zs.map((z, i) => [z, i]).filter(([z]) => Number.isFinite(z)); if (!ok.length) return toast('The result is not available at these points.', 'warn');
      const best = ok.reduce((p, q) => ((goal.value === 'min' ? q[0] < p[0] : q[0] > p[0]) ? q : p)), worst = ok.reduce((p, q) => ((goal.value === 'min' ? q[0] > p[0] : q[0] < p[0]) ? q : p));
      const bp = pts[best[1]], wp = pts[worst[1]], now = kpiMap(last)[lbl]?.value, unit = kpiMap(last)[lbl]?.unit || '';
      fill(out, h('div', { class: 'explain' }, h('h3', null, `${lbl} over ${fa.label} and ${fb.label}`),
        h('p', null, `Best in this map: ${fmt(best[0])}${unit ? ' ' + unit : ''} at ${fa.label} = ${fmt(bp[fa.key])}${unitOf(fa)} and ${fb.label} = ${fmt(bp[fb.key])}${unitOf(fb)} (now ${fmt(now)}${unit ? ' ' + unit : ''}, ${fmtPct(pct(now, best[0]))}). Worst: ${fmt(worst[0])} at ${fmt(wp[fa.key])} / ${fmt(wp[fb.key])}. The colours show the value; the dot marks the current case.`)),
        h('div', { class: 'plots' }, plotCard({ type: 'field', title: `${lbl}${unit ? ' (' + unit + ')' : ''}`, xlabel: fa.label + unitOf(fa), ylabel: fb.label + unitOf(fb), zlabel: lbl, zunit: unit, x: xa, y: xb, z: Z, contours: 8, cmap: 'turbo',
          markers: [{ x: base[fa.key], y: base[fb.key], label: 'now' }, { x: bp[fa.key], y: bp[fb.key], label: 'best' }], note: `Each node is a full solution of the engine (${N * N} in all); values between nodes are interpolated for display only.` }, { onDownload: download })),
        dataTable({ title: 'Mapped points', columns: [fa.label + unitOf(fa), fb.label + unitOf(fb), lbl], rows: pts.map((p, i) => [p[fa.key], p[fb.key], zs[i]]) }));
    }), 'primary');
    box.append(h('fieldset', { class: 'group' }, h('legend', null, '2 · Two inputs together', help('Maps one result over a grid of two inputs (each ±30 % around its current value, within its allowed range) and points out the best and worst combination for your goal.')),
      h('div', { class: 'row-tools' }, h('label', { class: 'inline' }, 'Result ', target), goal, h('label', { class: 'inline' }, 'over ', s1), h('label', { class: 'inline' }, 'and ', s2), n, go), out));
  }

  // ---- 3 · uncertainty ---------------------------------------------------------------------------------------
  {
    const rows = pref.slice(0, Math.min(pref.length, 40));
    const unc = rows.map((f, i) => ({ f, on: h('input', { type: 'checkbox', checked: i < 4, 'aria-label': 'Uncertain: ' + f.label }), pm: h('input', { type: 'number', step: 'any', min: 0, max: 90, value: 10, 'aria-label': '± %', style: 'max-width:80px' }) }));
    const n = h('select', { 'aria-label': 'Samples' }, [20, 40, 80, 160].map((x) => h('option', { value: x, selected: x === 40 }, `${x} samples`)));
    const out = h('div');
    const go = btn('Forecast with uncertainty', () => guard(go, +n.value, async () => {
      const act = unc.filter((u) => u.on.checked), N = +n.value, wanted = chosenOuts();
      if (!act.length) return toast('Tick at least one uncertain input.', 'warn');
      const U = lhs(N, act.length, 11), pts = U.map((u) => Object.fromEntries(act.map((a, j) => { const v = base[a.f.key], d = Math.abs(v) * (+a.pm.value || 0) / 100; let x = v - d + 2 * d * u[j]; if (isNum(a.f.min)) x = Math.max(a.f.min, x); if (isNum(a.f.max)) x = Math.min(a.f.max, x); return [a.f.key, x]; })));
      const res = await solveAll(pts, 'Sampling the uncertain inputs'), ks = res.map(kpiMap), b0 = kpiMap(last), tbl = [], plots = [], sentences = [];
      for (const lbl of wanted) {
        const ys = ks.map((k) => k[lbl]?.value).filter(Number.isFinite); if (ys.length < 3) continue;
        const p10 = quantile(ys, 0.1), p50 = quantile(ys, 0.5), p90 = quantile(ys, 0.9), unit = b0[lbl]?.unit || '';
        // rank the inputs by the strength of their (rank) correlation with this result
        const rank = (a) => { const s = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]), r = new Array(a.length); s.forEach(([, i], k) => (r[i] = k)); return r; };
        const yAll = ks.map((k) => k[lbl]?.value ?? NaN), okIdx = yAll.map((v, i) => (Number.isFinite(v) ? i : -1)).filter((i) => i >= 0), ry = rank(okIdx.map((i) => yAll[i]));
        const corr = act.map((a) => { const rx = rank(okIdx.map((i) => pts[i][a.f.key])), m = (ry.length - 1) / 2; let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < ry.length; i++) { sxy += (rx[i] - m) * (ry[i] - m); sxx += (rx[i] - m) ** 2; syy += (ry[i] - m) ** 2; } return [a.f.label, sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0]; }).sort((p, q) => Math.abs(q[1]) - Math.abs(p[1]));
        tbl.push([lbl, unit, b0[lbl]?.value, p10, p50, p90, 100 * (p90 - p10) / Math.max(1e-12, Math.abs(p50)), corr[0] ? `${corr[0][0]} (ρ ${fmt(corr[0][1], 2)})` : '–']);
        const hg = histogram(ys, Math.min(15, Math.max(5, Math.round(Math.sqrt(ys.length)))));
        plots.push({ type: 'bar', title: `${lbl}: spread of the prediction`, ylabel: 'Samples', categories: hg.centers.map((c) => fmt(c, 3)), series: [{ name: lbl, values: hg.counts }], note: `P10 ${fmt(p10)}, P50 ${fmt(p50)}, P90 ${fmt(p90)}${unit ? ' ' + unit : ''}. Bars: number of samples in each band of ${lbl}.` });
        sentences.push(`${lbl}: most likely about ${fmt(p50)}${unit ? ' ' + unit : ''}; 80 % of the outcomes lie between ${fmt(p10)} and ${fmt(p90)}. It depends most on ${corr.slice(0, 2).map(([n2, c]) => `${n2} (${c >= 0 ? 'rises' : 'falls'} with it)`).join(' and ')}.`);
      }
      fill(out, h('div', { class: 'explain' }, h('h3', null, 'What the forecast says'), h('ul', null, sentences.map((s) => h('li', null, s))), h('p', { class: 'note' }, `${N} Latin-hypercube samples of ${act.map((a) => `${a.f.label} ±${a.pm.value} %`).join(', ')}, each a full solution of the engine. P10 / P90: 10 % of outcomes fall below / above these values.`)),
        dataTable({ title: 'Forecast with uncertainty', columns: ['Result', 'Unit', 'Now (nominal)', 'P10', 'P50 (median)', 'P90', 'Spread P10–P90 % of median', 'Most influential input'], rows: tbl }),
        h('div', { class: 'plots' }, plots.map((p) => plotCard(p, { onDownload: download }))));
    }), 'primary');
    box.append(h('fieldset', { class: 'group' }, h('legend', null, '3 · How sure is the prediction?', help('Tick the inputs you are unsure about and give their uncertainty (± % of the current value). The engine is solved at Latin-hypercube samples of them and every result is reported as P10 / P50 / P90 with the input that drives it most.')),
      h('div', { class: 'tbl-scroll', tabindex: '0', style: 'max-height:260px' }, h('table', { class: 'tbl params' }, h('thead', null, h('tr', null, ['Uncertain', 'Input', 'Current value', '± %'].map((x) => h('th', null, x)))),
        h('tbody', null, unc.map((u) => h('tr', null, h('td', null, u.on), h('td', { class: 'lead' }, u.f.label + unitOf(u.f)), h('td', { class: 'num' }, fmt(base[u.f.key])), h('td', null, u.pm)))))),
      h('div', { class: 'row-tools' }, n, go), out));
  }
  return box;
}
