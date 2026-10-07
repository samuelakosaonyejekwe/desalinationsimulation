// Small DOM toolkit. All content is inserted as text nodes (never as HTML strings), so data that
// comes from files or remote services cannot inject markup or scripts.
import { fmt } from './num.js';
import { IONS, ION_IDS, WATERS, cloneIons, summarize } from './water.js';
import { readTable, toCSV, download, checkFile } from './io.js';
import { importGeometry, FORMATS } from './geom.js';
import { asOutline, asBathy } from './geomlinks.js';

export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'value') el.value = v;
    else if (k === 'checked' || k === 'disabled' || k === 'hidden' || k === 'selected' || k === 'open') el[k] = !!v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const kid of kids.flat(Infinity)) if (kid !== null && kid !== undefined && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
export const clear = (el) => { while (el.firstChild) el.firstChild.remove(); return el; };

let toastBox;
export function toast(msg, kind = 'info', ms = 4200) {
  if (!toastBox) { toastBox = h('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' }); document.body.append(toastBox); }
  const t = h('div', { class: 'toast ' + kind }, msg);
  toastBox.append(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 350); }, ms);
}

export const btn = (label, onclick, cls = '', title = '') => h('button', { type: 'button', class: 'btn ' + cls, onclick, title: title || null }, label);
export const help = (text) => (text ? h('span', { class: 'help', tabindex: '0', role: 'note', 'aria-label': text, 'data-tip': text }, '?') : null);
export const badge = (text, kind = '') => h('span', { class: 'badge ' + kind }, text);

/** KPI tile grid. items: [{ label, value, unit, status, help }] */
export function kpiGrid(items) {
  return h('div', { class: 'kpis' }, items.map((k) => h('div', { class: 'kpi ' + (k.status || ''), title: k.help || null },
    h('div', { class: 'kpi-l' }, k.label),
    h('div', { class: 'kpi-v' }, typeof k.value === 'number' ? fmt(k.value, k.sig || 4) : String(k.value ?? '–'), k.unit ? h('small', null, ' ' + k.unit) : null))));
}

/** Result table with CSV export. spec: { title, columns:[string], rows:[[...]], note } */
export function dataTable(spec) {
  const cap = h('div', { class: 'tbl-head' }, h('strong', null, spec.title || ''), btn('CSV', () => download(toCSV(spec.columns, spec.rows), (spec.title || 'table').replace(/[^\w-]+/g, '_') + '.csv', 'text/csv'), 'mini', 'Download this table'));
  const limit = 400, rows = spec.rows.slice(0, limit);
  const table = h('table', { class: 'tbl' },
    h('thead', null, h('tr', null, spec.columns.map((c) => h('th', { scope: 'col' }, c)))),
    h('tbody', null, rows.map((r) => h('tr', null, r.map((c, i) => h('td', { class: typeof c === 'number' ? 'num' : i === 0 ? 'lead' : '' }, typeof c === 'number' ? fmt(c) : c === null || c === undefined ? '–' : String(c)))))));
  return h('section', { class: 'tblbox' }, cap, h('div', { class: 'tbl-scroll', tabindex: '0' }, table),
    spec.rows.length > limit ? h('p', { class: 'note' }, `Showing the first ${limit} of ${spec.rows.length} rows — download the CSV for all rows.`) : null,
    spec.note ? h('p', { class: 'note' }, spec.note) : null);
}

export function tabs(defs, initial, onChange) {
  const bar = h('div', { class: 'tabs', role: 'tablist' }), body = h('div', { class: 'tab-body' });
  let active = null;
  const show = (id) => {
    const d = defs.find((x) => x.id === id) || defs[0];
    active = d.id;
    for (const b of bar.children) { const on = b.dataset.id === active; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); }
    clear(body); body.append(d.render());
    if (onChange) onChange(active);
  };
  defs.forEach((d) => bar.append(h('button', { type: 'button', class: 'tab', role: 'tab', dataset: { id: d.id }, onclick: () => show(d.id), title: d.tip || null }, d.label, d.count ? h('span', { class: 'pill' }, d.count) : null)));
  const el = h('div', { class: 'tabset' }, bar, body);
  el.show = show; el.active = () => active;
  show(initial || defs[0].id);
  return el;
}

// ---- field editors ---------------------------------------------------------------------------------
function numberField(f, value, set) {
  const input = h('input', { type: 'number', id: 'f_' + f.key, value: value ?? '', step: f.step ?? 'any', min: f.min ?? null, max: f.max ?? null, inputmode: 'decimal' });
  const msg = h('span', { class: 'field-msg' });
  const check = () => {
    const v = input.value === '' ? NaN : +input.value;
    let m = '';
    if (!Number.isFinite(v)) m = 'Enter a number';
    else if (f.min !== undefined && v < f.min) m = `Minimum is ${f.min}`;
    else if (f.max !== undefined && v > f.max) m = `Maximum is ${f.max}`;
    else if (f.typical && (v < f.typical[0] || v > f.typical[1])) m = `Typical range ${f.typical[0]}–${f.typical[1]}`;
    msg.textContent = m;
    msg.className = 'field-msg ' + (m && !m.startsWith('Typical') ? 'bad' : m ? 'warn' : '');
    input.setAttribute('aria-invalid', m && !m.startsWith('Typical') ? 'true' : 'false');
    return Number.isFinite(v) && !(f.min !== undefined && v < f.min) && !(f.max !== undefined && v > f.max);
  };
  input.addEventListener('input', () => { if (check()) set(+input.value); });
  check();
  return h('div', { class: 'ctl' }, h('div', { class: 'inp' }, input, f.unit ? h('span', { class: 'unit' }, f.unit) : null), msg);
}

function ionsField(f, value, set, ctx) {
  let ions = cloneIons(value || f.value);
  const grid = h('div', { class: 'ions' }), sumBox = h('div', { class: 'ion-sum' });
  const refresh = () => {
    const s = summarize(ions, 25);
    clear(sumBox).append(
      h('span', null, 'TDS ', h('b', null, fmt(s.tds)), ' mg/L'), h('span', null, 'Osmotic ', h('b', null, fmt(s.osmoticBar, 3)), ' bar'),
      h('span', null, 'Ionic strength ', h('b', null, fmt(s.ionicStrength, 3)), ' mol/L'),
      h('span', { class: Math.abs(s.chargeErrorPct) > 5 ? 'bad' : 'ok' }, 'Charge balance ', h('b', null, fmt(s.chargeErrorPct, 2)), ' %'));
  };
  const build = () => {
    clear(grid);
    for (const id of ION_IDS) {
      const inp = h('input', { type: 'number', min: 0, step: 'any', value: ions[id], 'aria-label': IONS[id].name + ' mg/L', inputmode: 'decimal' });
      inp.addEventListener('input', () => { ions[id] = Math.max(0, +inp.value || 0); set({ ...ions }); refresh(); });
      grid.append(h('label', { class: 'ion', title: IONS[id].name + ' (mg/L)' }, h('span', null, IONS[id].label), inp));
    }
    refresh();
  };
  const sel = h('select', { 'aria-label': 'Load a reference water' }, h('option', { value: '' }, 'Load reference water…'), Object.entries(WATERS).map(([k, w]) => h('option', { value: k }, w.name)));
  sel.addEventListener('change', () => { if (WATERS[sel.value]) { ions = cloneIons(WATERS[sel.value].ions); set({ ...ions }); build(); } sel.value = ''; });
  const tools = h('div', { class: 'row-tools' }, sel,
    ctx?.feed ? btn('Use case feed water', () => { ions = cloneIons(ctx.feed().ions); set({ ...ions }); build(); toast('Loaded the case feed-water analysis.'); }, 'mini') : null,
    importBtn('Import analysis', async (file) => {
      const t = await readTable(file), norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
      const alias = { sodium: 'Na', potassium: 'K', calcium: 'Ca', magnesium: 'Mg', barium: 'Ba', strontium: 'Sr', ammonium: 'NH4', iron: 'Fe', manganese: 'Mn', chloride: 'Cl', sulphate: 'SO4', sulfate: 'SO4', bicarbonate: 'HCO3', carbonate: 'CO3', nitrate: 'NO3', fluoride: 'F', phosphate: 'PO4', silica: 'SiO2', boron: 'B' };
      const find = (name) => ION_IDS.find((id) => norm(id) === norm(name)) || alias[norm(name)];
      let n = 0;
      // accept either one row with ion columns or two columns (ion, mg/L)
      for (const hd of t.headers) { const id = find(hd); if (id && typeof t.records[0]?.[hd] === 'number') { ions[id] = Math.max(0, t.records[0][hd]); n++; } }
      if (!n) for (const r of t.records) { const vals = Object.values(r), id = find(vals[0]); if (id && typeof vals[1] === 'number') { ions[id] = Math.max(0, vals[1]); n++; } }
      if (!n) throw new Error('No ion names recognised. Use columns such as Na, Ca, Cl, SO4 … in mg/L.');
      set({ ...ions }); build(); toast(`Imported ${n} constituents.`, 'ok');
    }));
  build();
  return h('div', { class: 'ctl wide' }, tools, grid, sumBox);
}

export function importBtn(label, handler, accept = '.csv,.tsv,.txt,.json,.xlsx') {
  const inp = h('input', { type: 'file', accept, hidden: true });
  inp.addEventListener('change', async () => {
    const file = inp.files[0];
    inp.value = '';
    if (!file) return;
    try { await handler(file); } catch (e) { toast(e.message || 'Could not read the file.', 'bad', 7000); }
  });
  return h('span', null, inp, btn(label, () => inp.click(), 'mini', 'Accepted: ' + accept));
}

function tableField(f, value, set) {
  let rows = Array.isArray(value) ? value.map((r) => ({ ...r })) : (f.value || []).map((r) => ({ ...r }));
  const wrap = h('div', { class: 'tbl-scroll edit', tabindex: '0' });
  const count = h('span', { class: 'note' });
  const commit = () => { set(rows.map((r) => ({ ...r }))); count.textContent = `${rows.length} row${rows.length === 1 ? '' : 's'}`; };
  const build = () => {
    const shown = rows.slice(0, 250);
    const body = h('tbody', null, shown.map((r, i) => h('tr', null,
      f.columns.map((c) => {
        const inp = h('input', { type: c.type === 'text' ? 'text' : 'number', step: 'any', value: r[c.key] ?? '', 'aria-label': `${c.label} row ${i + 1}` });
        inp.addEventListener('input', () => { r[c.key] = c.type === 'text' ? inp.value.slice(0, 200) : inp.value === '' ? null : +inp.value; commit(); });
        return h('td', null, inp);
      }),
      h('td', null, h('button', { type: 'button', class: 'mini ghost', title: 'Delete row', 'aria-label': 'Delete row', onclick: () => { rows.splice(i, 1); build(); commit(); } }, '✕')))));
    clear(wrap).append(h('table', { class: 'tbl' }, h('thead', null, h('tr', null, f.columns.map((c) => h('th', { scope: 'col' }, c.label, c.unit ? h('small', null, ' ' + c.unit) : null)), h('th', null, ''))), body));
    if (rows.length > 250) wrap.append(h('p', { class: 'note' }, `Editing the first 250 rows; all ${rows.length} rows are used in calculations.`));
    count.textContent = `${rows.length} row${rows.length === 1 ? '' : 's'}`;
  };
  const tools = h('div', { class: 'row-tools' },
    btn('+ Row', () => { rows.push(Object.fromEntries(f.columns.map((c) => [c.key, c.type === 'text' ? '' : 0]))); build(); commit(); }, 'mini'),
    importBtn('Import CSV / Excel', async (file) => {
      const t = await readTable(file), norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
      const map = f.columns.map((c, i) => t.headers.find((hd) => norm(hd) === norm(c.key) || norm(hd) === norm(c.label)) ?? t.headers[i]);
      rows = t.records.map((r) => Object.fromEntries(f.columns.map((c, i) => [c.key, map[i] === undefined ? null : r[map[i]]])));
      build(); commit(); toast(`Imported ${rows.length} rows from ${file.name}.`, 'ok');
    }),
    btn('Export', () => download(toCSV(f.columns.map((c) => c.key), rows), f.key + '.csv', 'text/csv'), 'mini'),
    f.value ? btn('Reset sample', () => { rows = f.value.map((r) => ({ ...r })); build(); commit(); }, 'mini ghost') : null, count);
  build();
  return h('div', { class: 'ctl wide' }, tools, wrap);
}

function fileField(f, value, set) {
  const info = h('span', { class: 'note' }, value?.name ? `Loaded: ${value.name}` : f.emptyText || (f.kind === 'table' ? 'No file loaded — site data or the built-in default is used.' : 'No file loaded — the built-in parametric geometry is used.'));
  return h('div', { class: 'ctl wide' }, h('div', { class: 'row-tools' },
    importBtn(f.buttonLabel || 'Import geometry', async (file) => {
      let g;
      const tabular = /\.(csv|tsv|txt|json|xlsx|xlsm)$/i.test(file.name);
      if (f.kind === 'table' && tabular) g = await readTable(file);
      else if (f.kind === 'table') { // terrain rasters, sounding clouds and terrain surfaces through the geometry importer
        checkFile(file); const raw = await importGeometry(file), b = asBathy(raw);
        if (!b) throw new Error(`${file.name} was read as “${raw.kind}”, which is not an elevation grid or sounding set.`);
        set(b); info.textContent = `Loaded: ${file.name} · ${(b.lon || b.x).length} × ${(b.lat || b.y).length} grid`; toast(`${file.name} imported.`, 'ok'); return;
      }
      else { checkFile(file); const raw = await importGeometry(file); g = asOutline(raw); if (!g) throw new Error(`${file.name} was read as “${raw.kind}”, which has no outline to use here. Open it in the Data portal to see where it can be used.`); g.name = file.name; }
      const v = f.parse ? f.parse(g, file.name) : g;
      set(v); info.textContent = `Loaded: ${file.name}` + (g.count ? ` · ${g.count.toLocaleString()} triangles` : g.polylines ? ` · ${g.polylines.length} outline(s)` : g.records ? ` · ${g.records.length} rows` : '');
      toast(`${file.name} imported.`, 'ok');
    }, (f.kind === 'table' ? (f.accept || '') + ',.asc,.grd,.tif,.tiff,.nc,.xyz,.las,.pts,.shp,.stl' : f.accept) || [...new Set(FORMATS.filter((x) => x.support !== 'convert').flatMap((x) => x.ext.map((e) => '.' + e)))].join(',')),
    btn('Clear', () => { set(null); info.textContent = 'Cleared — the built-in default is used.'; }, 'mini ghost'), info));
}

/**
 * Render one field definition. `get`/`set` read and write the current value; `ctx.rerender` is
 * called for fields whose value changes which other fields are visible.
 */
export function fieldRow(f, get, set, ctx = {}) {
  const value = get(f.key);
  let control;
  const wide = f.type === 'ions' || f.type === 'table' || f.type === 'file';
  if (f.type === 'select') {
    const opts = f.options.map((o) => (typeof o === 'object' ? o : { value: o, label: String(o) }));
    control = h('div', { class: 'ctl' }, h('select', { id: 'f_' + f.key, onchange: (e) => { const o = opts[e.target.selectedIndex]; set(f.key, o.value); if (ctx.rerender) ctx.rerender(); } },
      opts.map((o) => h('option', { value: String(o.value), selected: String(o.value) === String(value) }, o.label))));
  } else if (f.type === 'bool') {
    control = h('div', { class: 'ctl' }, h('label', { class: 'switch' }, h('input', { type: 'checkbox', id: 'f_' + f.key, checked: !!value, onchange: (e) => { set(f.key, e.target.checked); if (ctx.rerender) ctx.rerender(); } }), h('span', { class: 'slider' })));
  } else if (f.type === 'text') {
    control = h('div', { class: 'ctl' }, h('input', { type: 'text', id: 'f_' + f.key, value: value ?? '', maxlength: 200, oninput: (e) => set(f.key, e.target.value) }));
  } else if (f.type === 'ions') control = ionsField(f, value, (v) => set(f.key, v), ctx);
  else if (f.type === 'table') control = tableField(f, value, (v) => set(f.key, v));
  else if (f.type === 'file') control = fileField(f, value, (v) => { set(f.key, v); if (ctx.rerender) ctx.rerender(); });
  else control = numberField(f, value, (v) => set(f.key, v));
  const linked = ctx.linked?.[f.key];
  return h('div', { class: 'field' + (wide ? ' wide' : ''), dataset: { key: f.key } },
    h('label', { for: wide ? null : 'f_' + f.key }, f.label, help(f.help), linked ? h('span', { class: 'link-chip', title: 'Value linked from: ' + linked }, '⛓ ' + linked) : null), control);
}

export function emptyState(title, text, action) {
  return h('div', { class: 'empty' }, h('div', { class: 'empty-art', 'aria-hidden': 'true' }, '◌'), h('h3', null, title), h('p', null, text), action || null);
}
export { fmt };
