// Generic suite workspace. A suite module only declares its inputs and engine (see docs/SUITE_CONTRACT.md);
// this file turns that declaration into the full workflow: guide → inputs → model setup → mesh →
// run → results → mesh-sensitivity → calibration → verification & validation → theory.
import { h, clear, btn, tabs, fieldRow, kpiGrid, dataTable, toast, badge, emptyState, help, fill } from './ui.js';
import { plotCard } from './plot.js';
import { store, sanitize } from './store.js';
import { fmt, gci, levenbergMarquardt, metrics, isNum } from './num.js';
import { download, toCSV } from './io.js';
import { CATALOG } from '../data/catalog.js';
import { geometryTab } from './geomview.js';
import { advicePanel } from './advisorview.js';

const lastResult = new Map(); // suite id -> full result of the last run (kept in memory)
const bigValues = new Map(); // `${suite}.${key}` -> large imported objects (geometry) kept out of localStorage

export const allFields = (suite) => suite.inputs.flatMap((g) => g.fields);
export function defaults(suite) {
  const d = {};
  for (const f of allFields(suite)) d[f.key] = f.type === 'ions' || f.type === 'table' ? structuredCloneSafe(f.value) : f.value ?? null;
  return d;
}
const structuredCloneSafe = (v) => (v === undefined ? null : JSON.parse(JSON.stringify(v)));

/** Current input values = defaults overlaid with what the user stored for this case. */
export function values(suite) {
  const v = defaults(suite), s = store.inputs(suite.id);
  for (const f of allFields(suite)) {
    if (f.type === 'file') { if (bigValues.has(suite.id + '.' + f.key)) v[f.key] = bigValues.get(suite.id + '.' + f.key); }
    else if (s[f.key] !== undefined && s[f.key] !== null) v[f.key] = s[f.key];
  }
  return v;
}
export function setInputValue(suite, key, value) { setValue(suite, key, value); }
function setValue(suite, key, value) {
  const f = allFields(suite).find((x) => x.key === key);
  if (f?.type === 'file') { if (value === null) bigValues.delete(suite.id + '.' + key); else bigValues.set(suite.id + '.' + key, value); }
  else store.setInput(suite.id, key, value);
}

/** Attach an imported geometry object to a suite's file-type input (used by the data portal). */
export function setGeometry(suite, key, g) { bigValues.set(suite.id + '.' + key, g); }

const context = (extra = {}) => ({
  feed: store.case.feed, site: store.case.site, outputs: store.case.outputs,
  progress: () => {}, tick: () => new Promise((r) => setTimeout(r, 0)), ...extra,
});

/** Values offered by upstream suites, the case feed water and the site data. */
export function linkItems(suite) {
  const items = [];
  const push = (list, kind) => { for (const it of list || []) if (it && it.key && it.value !== undefined && it.value !== null && !(typeof it.value === 'number' && !Number.isFinite(it.value))) items.push({ ...it, kind }); };
  try { push(suite.pull?.(context()), 'suite'); } catch (e) { console.warn('pull failed', suite.id, e); }
  try { if (store.case.site?.fetchedAt) push(suite.site?.(store.case.site), 'site'); } catch (e) { console.warn('site map failed', suite.id, e); }
  const known = new Set(allFields(suite).map((f) => f.key));
  return items.filter((it) => known.has(it.key));
}
export function applyLinks(suite, items = linkItems(suite)) {
  const prov = store.inputs(suite.id)._links || {};
  for (const it of items) { setValue(suite, it.key, sanitize(it.value)); prov[it.key] = it.from; }
  store.setInput(suite.id, '_links', prov);
  return items.length;
}

/** Run a suite with the current case inputs; stores its outputs for the other suites. */
export async function runSuite(suite, ctxExtra) {
  const v = values(suite), t0 = performance.now();
  const res = (await suite.run(v, context(ctxExtra))) || {};
  res.kpis ||= []; res.tables ||= []; res.plots ||= []; res.warnings ||= []; res.outputs ||= {};
  res._ms = performance.now() - t0; res._inputs = v;
  lastResult.set(suite.id, res);
  store.setOutputs(suite.id, { ...res.outputs, _kpis: res.kpis.slice(0, 8).map((k) => ({ label: k.label, value: k.value, unit: k.unit || '', status: k.status || '' })), _warnings: res.warnings.length });
  return res;
}
export const getResult = (id) => lastResult.get(id);

const norm = (s) => String(s).toLowerCase().normalize('NFKD').replace(/[‐-―]/g, '-').replace(/[’'`]/g, '').replace(/s\b/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
function implementedTest(suite) {
  const keys = (suite.implemented || []).map(norm).filter(Boolean);
  const ref = (suite.referenceOnly || []).map(norm).filter(Boolean); // fragments that must stay unticked
  return (item) => { const n = norm(item); return !ref.some((k) => n.includes(k)) && keys.some((k) => n.includes(k) || k.includes(n)); };
}

// ---- report export -----------------------------------------------------------------------------------
function exportReport(suite, res, plotEls) {
  const doc = document.implementation.createHTMLDocument(`${suite.title} — ${store.case.name}`);
  const add = (parent, tag, text, attrs = {}) => { const e = doc.createElement(tag); if (text !== undefined && text !== null) e.textContent = text; for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); parent.append(e); return e; };
  add(doc.head, 'meta', null, { charset: 'utf-8' });
  add(doc.head, 'style', 'body{font:14px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#0f172a}table{border-collapse:collapse;width:100%;margin:8px 0 20px;font-size:12px}th,td{border:1px solid #cbd5e1;padding:4px 8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f1f5f9}img{max-width:100%;border:1px solid #e2e8f0;margin:6px 0 18px}h1{font-size:22px}h2{font-size:16px;margin-top:28px;border-bottom:2px solid #0ea5e9;padding-bottom:4px}.w{color:#b45309}');
  const b = doc.body;
  add(b, 'h1', `${suite.num}. ${suite.title}`);
  add(b, 'p', `Case: ${store.case.name} · Site: ${store.case.site.name || 'not set'} · Generated ${new Date().toLocaleString()}`);
  if (res.warnings.length) { add(b, 'h2', 'Warnings and checks'); const ul = add(b, 'ul'); res.warnings.forEach((w) => add(ul, 'li', `[${w.level || 'info'}] ${w.msg}`, { class: 'w' })); }
  add(b, 'h2', 'Key results');
  const kt = add(b, 'table'); res.kpis.forEach((k) => { const tr = add(kt, 'tr'); add(tr, 'td', k.label); add(tr, 'td', `${typeof k.value === 'number' ? fmt(k.value) : k.value} ${k.unit || ''}`); });
  add(b, 'h2', 'Inputs used');
  const it = add(b, 'table');
  for (const f of allFields(suite)) { if (['ions', 'table', 'file'].includes(f.type)) continue; const tr = add(it, 'tr'); add(tr, 'td', f.label); add(tr, 'td', `${res._inputs[f.key]} ${f.unit || ''}`); }
  if (plotEls.length) add(b, 'h2', 'Plots');
  plotEls.forEach((c) => { const cv = c.querySelector('canvas'); if (cv && cv.width) { add(b, 'h3', c.querySelector('figcaption span')?.textContent || ''); add(b, 'img', null, { src: cv.toDataURL('image/png'), alt: 'plot' }); } });
  for (const t of res.tables) {
    add(b, 'h2', t.title || 'Table');
    const tb = add(b, 'table'), hr = add(tb, 'tr'); t.columns.forEach((c) => add(hr, 'th', c));
    t.rows.slice(0, 500).forEach((r) => { const tr = add(tb, 'tr'); r.forEach((c) => add(tr, 'td', typeof c === 'number' ? fmt(c) : c ?? '')); });
  }
  download('<!doctype html>' + doc.documentElement.outerHTML, `${suite.id}_report_${store.case.name}.html`, 'text/html');
}

// ---- the page ----------------------------------------------------------------------------------------
export function renderSuite(suite, root, app) {
  const cat = CATALOG[suite.num] || {};
  const isImpl = implementedTest(suite);
  let tabset;
  const status = h('span', { class: 'run-status', role: 'status', 'aria-live': 'polite' });
  const bar = h('div', { class: 'progress', hidden: true }, h('i'));

  const doRun = async () => {
    runBtn.disabled = true; bar.hidden = false; bar.firstChild.style.width = '4%';
    status.textContent = 'Running…';
    try {
      if (store.case.autolink) applyLinks(suite);
      const res = await runSuite(suite, { progress: (f, msg) => { bar.firstChild.style.width = Math.round(4 + 96 * Math.max(0, Math.min(1, f))) + '%'; if (msg) status.textContent = msg; } });
      status.textContent = `Solved in ${res._ms < 1000 ? Math.round(res._ms) + ' ms' : (res._ms / 1000).toFixed(1) + ' s'}`;
      const bad = res.warnings.filter((w) => w.level === 'bad').length;
      toast(bad ? `Run finished with ${bad} limit violation${bad > 1 ? 's' : ''}.` : 'Run finished.', bad ? 'warn' : 'ok');
      tabset.show('results');
    } catch (e) {
      console.error(e);
      status.textContent = 'Run failed';
      toast('Run failed: ' + (e.message || e), 'bad', 9000);
    } finally { runBtn.disabled = false; setTimeout(() => (bar.hidden = true), 500); }
  };
  const runBtn = btn('▶  Run simulation', doRun, 'primary', 'Solve this suite with the current inputs (shortcut: Ctrl/⌘ + Enter)');

  const presetSel = suite.presets?.length ? h('select', { 'aria-label': 'Load an example case', onchange: (e) => {
    const p = suite.presets[+e.target.value];
    if (p) { store.setInputs(suite.id, { ...defaults(suite), ...p.values }); toast(`Loaded example: ${p.name}`); tabset.show(tabset.active()); }
    e.target.value = '';
  } }, h('option', { value: '' }, 'Load example…'), suite.presets.map((p, i) => h('option', { value: i }, p.name))) : null;

  const groupTab = (tabName) => () => {
    const groups = suite.inputs.filter((g) => (g.tab || 'inputs') === tabName);
    const box = h('div', { class: 'groups' });
    const paint = () => {
      const v = values(suite), linked = store.inputs(suite.id)._links || {};
      clear(box);
      if (tabName === 'inputs') box.append(linkPanel(paint));
      for (const g of groups) {
        if (g.showIf && !g.showIf(v)) continue;
        const fields = g.fields.filter((f) => !f.showIf || f.showIf(v));
        if (!fields.length) continue;
        box.append(h('fieldset', { class: 'group' }, h('legend', null, g.group, help(g.help)),
          h('div', { class: 'fields' }, fields.map((f) => fieldRow(f, (k) => v[k], (k, val) => { v[k] = val; setValue(suite, k, val); const l = store.inputs(suite.id)._links; if (l && l[k]) { delete l[k]; } }, { rerender: paint, linked, feed: () => store.case.feed })))));
      }
      if (tabName === 'mesh' && suite.mesh) box.append(meshStudyPanel());
    };
    paint();
    return box;
  };

  const linkPanel = (repaint) => {
    const items = linkItems(suite);
    if (!items.length) return h('div', { class: 'linkbar quiet' }, h('span', null, '⛓ No linked data yet. Run an upstream suite or set the site on the Global Site Data page and matching inputs will be offered here automatically.'));
    const list = h('ul', { class: 'linklist' }, items.map((it) => h('li', null, h('b', null, allFields(suite).find((f) => f.key === it.key)?.label || it.key), ' ← ', typeof it.value === 'number' ? fmt(it.value) : typeof it.value === 'object' ? 'data set' : String(it.value), h('small', null, ' · ' + it.from))));
    return h('details', { class: 'linkbar' },
      h('summary', null, `⛓ ${items.length} input${items.length > 1 ? 's' : ''} available from linked suites and site data`,
        btn('Apply all', (e) => { e.preventDefault(); const n = applyLinks(suite, items); toast(`${n} linked value${n > 1 ? 's' : ''} applied.`, 'ok'); repaint(); }, 'mini primary')),
      list, h('p', { class: 'note' }, store.case.autolink ? 'Auto-link is on: these values are applied automatically each time you run. Turn it off on the Case page to keep manual values.' : 'Auto-link is off: values are only applied when you press “Apply all”.'));
  };

  // -- mesh / step sensitivity (grid-convergence index)
  const meshStudyPanel = () => {
    const studies = Array.isArray(suite.mesh) ? suite.mesh : [suite.mesh];
    const out = h('div', { class: 'study-out' });
    const ratio = h('input', { type: 'number', min: 1.2, max: 2.5, step: 0.1, value: 1.5, 'aria-label': 'Refinement ratio' });
    const sel = h('select', { 'aria-label': 'Study' }, studies.map((s, i) => h('option', { value: i }, s.name || 'Spatial grid')));
    const go = btn('Run 3-level sensitivity study', async () => {
      const st = studies[+sel.value], r = Math.max(1.2, Math.min(2.5, +ratio.value || 1.5)), base = values(suite);
      go.disabled = true; fill(out, h('p', { class: 'note' }, 'Solving coarse, medium and fine levels…'));
      try {
        const levels = [];
        for (const fac of [1 / r, 1, r]) {
          const v = { ...base };
          for (const k of st.keys) v[k] = st.refine === 'divide' ? base[k] / fac : Math.max(st.min || 4, Math.round(base[k] * fac));
          const res = await suite.run(v, context());
          const size = st.refine === 'divide' ? v[st.keys[0]] : 1 / v[st.keys[0]];
          levels.push({ v, size, cells: st.keys.reduce((p, k) => p * (st.refine === 'divide' ? 1 : v[k]), 1), vals: st.metrics.map((m) => +m.get(res)) });
          await new Promise((q) => setTimeout(q, 0));
        }
        const [c, m, f] = levels, rows = [], plots = [];
        st.metrics.forEach((mt, i) => {
          const g = gci([f.size, m.size, c.size], [f.vals[i], m.vals[i], c.vals[i]]);
          const verdict = g.gciFine < 0.01 ? 'Converged (< 1 %)' : g.gciFine < 0.03 ? 'Acceptable (< 3 %)' : g.gciFine < 0.1 ? 'Refine further' : 'Not converged';
          rows.push([mt.label + (mt.unit ? ` (${mt.unit})` : ''), c.vals[i], m.vals[i], f.vals[i], g.fExact, g.p, 100 * g.gciFine, 100 * g.gciCoarse, g.type, verdict]);
          plots.push({ type: 'line', title: `${mt.label} versus resolution`, xlabel: st.refine === 'divide' ? st.keys[0] : 'Normalised cell size h', ylabel: mt.unit || mt.label, height: 240,
            series: [{ name: 'Computed', x: [c.size, m.size, f.size], y: [c.vals[i], m.vals[i], f.vals[i]], mode: 'both' }, { name: 'Richardson extrapolation (h → 0)', x: [0, f.size], y: [g.fExact, f.vals[i]], dash: true }] });
        });
        fill(out, 
          dataTable({ title: `Grid-convergence study — ${st.name || 'spatial grid'}`, columns: ['Quantity', 'Coarse', 'Medium', 'Fine', 'Extrapolated', 'Observed order p', 'GCI fine %', 'GCI medium %', 'Convergence', 'Verdict'], rows,
            note: `Levels: ${st.keys.map((k) => `${k} = ${levels.map((l) => fmt(l.v[k])).join(' / ')}`).join('; ')}. Numerical uncertainty is the grid-convergence index (safety factor 1.25) from Richardson extrapolation on three systematically refined levels.` }),
          h('div', { class: 'plots' }, plots.map((p) => plotCard(p, { onDownload: download }))));
      } catch (e) { fill(out, h('p', { class: 'bad' }, 'Study failed: ' + e.message)); }
      go.disabled = false;
    }, 'primary');
    return h('fieldset', { class: 'group' }, h('legend', null, 'Mesh and step sensitivity', help('Solves the model on three systematically refined levels and quantifies the numerical uncertainty with Richardson extrapolation and the grid-convergence index (GCI), instead of simply declaring the mesh “independent”.')),
      h('div', { class: 'row-tools' }, studies.length > 1 ? sel : null, h('label', { class: 'inline' }, 'Refinement ratio ', ratio), go),
      h('p', { class: 'note' }, studies.map((st) => st.note).filter(Boolean).join(' ')), out);
  };

  // -- results
  const resultsTab = () => {
    const res = lastResult.get(suite.id);
    if (!res) return emptyState('No results yet', 'Press “Run simulation” to solve this suite with the current inputs. The defaults describe a realistic industrial case, so you can run straight away.', btn('▶  Run simulation', doRun, 'primary'));
    const plotEls = res.plots.map((p) => plotCard(p, { onDownload: download }));
    const box = h('div', { class: 'results' });
    if (res.warnings.length) box.append(h('div', { class: 'warns' }, res.warnings.map((w) => h('div', { class: 'warn ' + (w.level || 'info') }, h('b', null, w.level === 'bad' ? 'Limit exceeded' : w.level === 'warn' ? 'Check' : 'Note'), ' ', w.msg))));
    if (res.summary) box.append(h('p', { class: 'summary' }, res.summary));
    box.append(kpiGrid(res.kpis));
    box.append(h('div', { class: 'row-tools' },
      btn('Report (HTML)', () => exportReport(suite, res, plotEls), 'mini', 'Self-contained report with inputs, KPIs, plots and tables — print it to PDF from your browser'),
      btn('Results (JSON)', () => download(JSON.stringify({ suite: suite.id, case: store.case.name, inputs: sanitize(res._inputs), kpis: res.kpis, tables: res.tables, outputs: res.outputs, warnings: res.warnings }, null, 1), `${suite.id}_results.json`, 'application/json'), 'mini'),
      btn('All tables (CSV)', () => download(res.tables.map((t) => `# ${t.title}\n` + toCSV(t.columns, t.rows)).join('\n\n'), `${suite.id}_tables.csv`, 'text/csv'), 'mini')));
    if (plotEls.length) box.append(h('div', { class: 'plots' }, plotEls));
    res.tables.forEach((t) => box.append(dataTable(t)));
    try { box.append(advicePanel(suite, res.recommendations || [])); } catch (e) { console.warn('advisor', e); }
    const down = app?.downstream?.(suite.id) || [];
    if (down.length) box.append(h('div', { class: 'linkbar' }, h('span', null, '⛓ These results now feed: '), down.map((s) => h('a', { class: 'chip', href: '#/suite/' + s.id }, `${s.num}. ${s.short || s.title}`))));
    return box;
  };

  // -- calibration and validation
  const calTab = () => {
    const c = suite.calibration;
    if (!c) return emptyState('Calibration', 'This suite has no adjustable calibration parameters exposed.');
    const stored = store.inputs(suite.id);
    const out = h('div'), vout = h('div');
    const checks = c.params.map((p) => ({ p, on: h('input', { type: 'checkbox', checked: true, 'aria-label': 'Fit ' + p.label }), lo: h('input', { type: 'number', step: 'any', value: p.lo, 'aria-label': 'Lower bound' }), hi: h('input', { type: 'number', step: 'any', value: p.hi, 'aria-label': 'Upper bound' }) }));
    const tableDef = (key, sample) => ({ key, label: '', type: 'table', columns: c.columns, value: sample || [] });
    const calDef = tableDef('_cal', c.sample), valDef = tableDef('_val', c.validationSample || c.sample);
    const getT = (def) => (Array.isArray(stored[def.key]) ? stored[def.key] : def.value);
    const targetKeys = c.targets.map((t) => t.key);
    const predictRows = (base, rows) => rows.map((row) => {
      const v = { ...base };
      for (const col of c.columns) if (!targetKeys.includes(col.key) && isNum(row[col.key])) v[col.key] = row[col.key];
      return c.model(v);
    });
    const resid = (base, rows, scale) => (pvec, active) => {
      const b = { ...base }; active.forEach((a, i) => (b[a.p.key] = pvec[i]));
      const pred = predictRows(b, rows), r = [];
      rows.forEach((row, i) => targetKeys.forEach((k) => { if (isNum(row[k])) r.push(((pred[i][k] ?? NaN) - row[k]) / (scale[k] || 1)); }));
      return r.map((x) => (Number.isFinite(x) ? x : 1e6));
    };
    const scales = (rows) => Object.fromEntries(targetKeys.map((k) => { const a = rows.map((r) => r[k]).filter(isNum); return [k, a.length ? Math.max(1e-12, a.reduce((s, x) => s + Math.abs(x), 0) / a.length) : 1]; }));
    const parity = (rows, pred, title) => c.targets.map((t) => {
      const mm = [], pp = [];
      rows.forEach((r, i) => { if (isNum(r[t.key]) && isNum(pred[i][t.key])) { mm.push(r[t.key]); pp.push(pred[i][t.key]); } });
      if (!mm.length) return null;
      const lo = Math.min(...mm, ...pp), hi = Math.max(...mm, ...pp), mt = metrics(mm, pp);
      return { t, mt, plots: [
        { type: 'line', title: `${title}: ${t.label} — predicted vs measured`, xlabel: `Measured ${t.unit || ''}`, ylabel: `Predicted ${t.unit || ''}`, height: 260, series: [{ name: 'Data', x: mm, y: pp, mode: 'points' }, { name: 'Perfect agreement', x: [lo, hi], y: [lo, hi], dash: true }] },
        { type: 'line', title: `${title}: ${t.label} — residuals`, xlabel: `Measured ${t.unit || ''}`, ylabel: 'Predicted − measured', height: 260, series: [{ name: 'Residual', x: mm, y: mt.residuals, mode: 'points' }], hlines: [{ y: 0, label: 'zero', color: '#64748b' }] }] };
    }).filter(Boolean);
    const metricTable = (title, ps) => dataTable({ title, columns: ['Quantity', 'n', 'Bias', 'MAE', 'RMSE', 'NRMSE %', 'MAPE %', 'R²', '95 % CI of bias (low)', '95 % CI of bias (high)'], rows: ps.map((q) => [q.t.label, q.mt.n, q.mt.bias, q.mt.mae, q.mt.rmse, 100 * q.mt.nrmse, q.mt.mape, q.mt.r2, q.mt.ci95[0], q.mt.ci95[1]]) });

    const fit = () => {
      const rows = getT(calDef).filter((r) => targetKeys.some((k) => isNum(r[k]))), active = checks.filter((x) => x.on.checked);
      if (!active.length) return toast('Select at least one parameter to fit.', 'warn');
      if (rows.length * targetKeys.length < active.length) return toast('Not enough data rows for the number of fitted parameters.', 'warn');
      try {
        const base = values(suite), sc = scales(rows), fn = resid(base, rows, sc);
        const r = levenbergMarquardt((p) => fn(p, active), active.map((a) => base[a.p.key]), { lo: active.map((a) => +a.lo.value), hi: active.map((a) => +a.hi.value) });
        const fitted = { ...base }; active.forEach((a, i) => (fitted[a.p.key] = r.p[i]));
        const ps = parity(rows, predictRows(fitted, rows), 'Calibration');
        fill(out, 
          dataTable({ title: 'Fitted parameters', columns: ['Parameter', 'Initial', 'Fitted', '± Std. error', 'Relative error %', 'Identifiability'], rows: active.map((a, i) => [a.p.label, base[a.p.key], r.p[i], r.se[i], Math.abs((100 * r.se[i]) / (r.p[i] || 1)), !Number.isFinite(r.se[i]) ? 'Not identifiable' : Math.abs(r.se[i] / (r.p[i] || 1)) < 0.25 ? 'Well identified' : 'Weakly identified']), note: `Levenberg–Marquardt least squares, ${r.iterations} iterations, ${rows.length} data rows. Standard errors come from the parameter covariance matrix; a weakly identified parameter means the data do not constrain it — add operating points that span pressure, recovery, temperature and salinity.` }),
          metricTable('Goodness of fit on the calibration data (not a validation)', ps),
          h('div', { class: 'row-tools' }, btn('Apply fitted parameters to this case', () => { active.forEach((a, i) => setValue(suite, a.p.key, r.p[i])); toast('Fitted parameters applied to the inputs.', 'ok'); }, 'primary')),
          h('div', { class: 'plots' }, ps.flatMap((q) => q.plots).map((p) => plotCard(p, { onDownload: download }))));
      } catch (e) { toast('Calibration failed: ' + e.message, 'bad'); }
    };
    const validate = () => {
      const rows = getT(valDef).filter((r) => targetKeys.some((k) => isNum(r[k])));
      if (!rows.length) return toast('Add independent validation rows first.', 'warn');
      try {
        const ps = parity(rows, predictRows(values(suite), rows), 'Validation');
        fill(vout, metricTable('Validation metrics against independent data', ps), h('div', { class: 'plots' }, ps.flatMap((q) => q.plots).map((p) => plotCard(p, { onDownload: download }))));
      } catch (e) { toast('Validation failed: ' + e.message, 'bad'); }
    };
    const setT = (k, v) => store.setInput(suite.id, k, v);
    return h('div', { class: 'groups' },
      h('p', { class: 'summary' }, c.note || 'Estimate uncertain model parameters from measured data, then confirm the model on separate data that were not used for fitting.'),
      h('fieldset', { class: 'group' }, h('legend', null, '1 · Parameters to estimate'),
        h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Fit', 'Parameter', 'Current value', 'Lower bound', 'Upper bound'].map((x) => h('th', null, x)))),
          h('tbody', null, checks.map((x) => h('tr', null, h('td', null, x.on), h('td', { class: 'lead' }, x.p.label), h('td', { class: 'num' }, fmt(values(suite)[x.p.key])), h('td', null, x.lo), h('td', null, x.hi)))))),
      h('fieldset', { class: 'group' }, h('legend', null, '2 · Calibration data', help('Measured operating points. Columns that match model inputs set the operating condition of each row; the remaining columns are the measured targets.')),
        fieldRow(calDef, () => getT(calDef), setT), h('div', { class: 'row-tools' }, btn('Fit parameters', fit, 'primary')), out),
      h('fieldset', { class: 'group' }, h('legend', null, '3 · Independent validation data', help('Use operating points that were NOT used for calibration. Agreement with calibration data is not validation.')),
        fieldRow(valDef, () => getT(valDef), setT), h('div', { class: 'row-tools' }, btn('Validate model', validate, 'primary')), vout),
      cat.calibration ? h('details', { class: 'ref' }, h('summary', null, 'Recommended calibration practice for this suite'), h('p', null, cat.calibration)) : null,
      cat.validation ? h('details', { class: 'ref' }, h('summary', null, 'Recommended validation practice for this suite'), h('p', null, cat.validation)) : null);
  };

  const verifyTab = () => {
    const out = h('div');
    const run = async () => {
      fill(out, h('p', { class: 'note' }, 'Running verification checks…'));
      try {
        const checks = (await suite.verify?.()) || [];
        const pass = checks.filter((c) => c.pass).length;
        fill(out, 
          h('p', { class: 'summary' }, badge(`${pass} / ${checks.length} passed`, pass === checks.length ? 'ok' : 'bad'), ' Code and equation verification: conservation, limiting cases and independent hand calculations, executed live on this device.'),
          dataTable({ title: 'Verification checks', columns: ['Check', 'Expected', 'Computed', 'Tolerance', 'Result', 'Basis'], rows: checks.map((c) => [c.name, c.expected, c.got, c.tol, c.pass ? '✓ pass' : '✗ FAIL', c.note || '']) }));
      } catch (e) { fill(out, h('p', { class: 'bad' }, 'Verification failed to run: ' + e.message)); }
    };
    const res = lastResult.get(suite.id);
    return h('div', { class: 'groups' },
      h('fieldset', { class: 'group' }, h('legend', null, 'Verification — is the model solved correctly?'), h('div', { class: 'row-tools' }, btn('Run verification checks', run, 'primary')), out),
      res?.balances ? dataTable({ title: 'Conservation closure of the last run', columns: ['Balance', 'In', 'Out', 'Closure error %'], rows: res.balances.map((b) => [b.name, b.in, b.out, b.in ? (100 * (b.in - b.out)) / b.in : 0]) }) : null,
      cat.verification ? h('details', { class: 'ref', open: true }, h('summary', null, 'Verification practice for this suite'), h('p', null, cat.verification)) : null,
      h('p', { class: 'note' }, 'Validation against measured data and calibration live on the “Calibrate & validate” tab. Mesh and step-size uncertainty live on the “Mesh” tab.'));
  };

  const theoryTab = () => {
    const chips = (items) => h('div', { class: 'chips' }, (items || []).map((it) => h('span', { class: 'chip ' + (isImpl(it) ? 'on' : ''), title: isImpl(it) ? 'Solved by this suite' : 'Reference formulation — listed for completeness, not solved by the built-in engine' }, isImpl(it) ? '✓ ' : '', it)));
    const count = (items) => `${(items || []).filter(isImpl).length} of ${(items || []).length} solved in-app`;
    return h('div', { class: 'groups' },
      h('p', { class: 'summary' }, 'Ticked items are solved by the built-in engine of this suite. Unticked items are established formulations listed for reference; they can be added through the custom-model tools of suite 11.'),
      suite.equationsNote ? h('p', { class: 'note' }, suite.equationsNote) : null,
      h('fieldset', { class: 'group' }, h('legend', null, 'Classical governing equations ', h('small', null, count(cat.classical))), chips(cat.classical)),
      h('fieldset', { class: 'group' }, h('legend', null, 'Hybrid and coupled formulations ', h('small', null, count(cat.hybrid))), chips(cat.hybrid)),
      h('fieldset', { class: 'group' }, h('legend', null, 'Initial and boundary conditions ', h('small', null, count(cat.icbc))), chips(cat.icbc), h('p', { class: 'note' }, cat.icbcText || '')),
      h('fieldset', { class: 'group' }, h('legend', null, 'Modules ', h('small', null, count(cat.modules))), chips(cat.modules)));
  };

  const guideTab = () => h('div', { class: 'groups' },
    h('p', { class: 'summary' }, suite.description || suite.tagline),
    h('ol', { class: 'steps' }, (suite.guide || ['Review the inputs — defaults describe a realistic industrial case.', 'Choose the models and boundary conditions on the Model setup tab.', 'Press Run simulation and read the results, warnings and suggested actions.', 'Quantify numerical uncertainty, calibrate against your data and validate on independent data.']).map((s) => h('li', null, s))),
    h('div', { class: 'row-tools' }, btn('Go to inputs →', () => tabset.show('inputs'), 'primary'), btn('▶  Run with defaults', doRun)),
    h('details', { class: 'ref' }, h('summary', null, 'Input data this suite accepts'), h('p', null, cat.inputs || '')),
    h('details', { class: 'ref' }, h('summary', null, 'Output data this suite produces'), h('p', null, cat.outputs || '')));

  const has = (t) => suite.inputs.some((g) => (g.tab || 'inputs') === t);
  const defs = [
    { id: 'guide', label: 'Guide', render: guideTab, tip: 'What this suite does and how to use it' },
    { id: 'inputs', label: 'Inputs', render: groupTab('inputs'), tip: 'Feed, equipment and operating data' },
    has('setup') && { id: 'setup', label: 'Model setup', render: groupTab('setup'), tip: 'Model choices, initial and boundary conditions, solver settings' },
    (has('mesh') || suite.mesh) && { id: 'mesh', label: 'Mesh', render: groupTab('mesh'), tip: 'Discretisation and sensitivity study' },
    { id: 'geometry', label: 'Geometry', render: () => geometryTab(suite, { fields: allFields(suite), values: () => values(suite), setValue: (k, val) => setValue(suite, k, val) }), tip: 'Import CAD, mesh, GIS, point-cloud, image or network geometry, or generate one' },
    { id: 'results', label: 'Results', render: resultsTab, tip: 'KPIs, plots, tables and exports' },
    ...(suite.views || []).map((v) => ({ id: 'x_' + v.id, label: v.label, tip: v.tip, render: () => { const el = h('div', { class: 'groups' }); try { v.render(el, { values: () => values(suite), set: (k, val) => setValue(suite, k, val), result: () => lastResult.get(suite.id), run: doRun, h, plotCard: (p) => plotCard(p, { onDownload: download }), dataTable, kpiGrid, toast, download, store }); } catch (e) { el.append(h('p', { class: 'bad' }, e.message)); } return el; } })),
    { id: 'cal', label: 'Calibrate & validate', render: calTab, tip: 'Parameter estimation and independent validation' },
    { id: 'verify', label: 'Verify', render: verifyTab, tip: 'Conservation, limiting cases and hand-calculation checks' },
    { id: 'theory', label: 'Theory', render: theoryTab, tip: 'Governing equations, IC/BC and modules' },
  ].filter(Boolean);

  tabset = tabs(defs, store.pref('tab.' + suite.id) || 'guide', (id) => store.pref('tab.' + suite.id, id));
  const onKey = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && root.isConnected) { e.preventDefault(); doRun(); } };
  document.addEventListener('keydown', onKey);
  fill(root, 
    h('header', { class: 'suite-head' },
      h('div', { class: 'suite-title' }, h('span', { class: 'suite-num' }, suite.num), h('div', null, h('h1', null, suite.title), h('p', null, suite.tagline))),
      h('div', { class: 'actions' }, presetSel, btn('Reset', () => { store.clearInputs(suite.id); bigValues.forEach((_, k) => k.startsWith(suite.id + '.') && bigValues.delete(k)); toast('Inputs reset to defaults.'); tabset.show(tabset.active()); }, 'ghost', 'Restore the default inputs of this suite'), runBtn)),
    h('div', { class: 'statusline' }, status, bar), tabset);
  return () => document.removeEventListener('keydown', onKey);
}
