// Geometry hub UI: shared by the data portal and by the "Geometry" tab of every suite.
// Imported geometry of any pathway (CAD, surface, mesh, drawing, GIS, point cloud, voxel, network,
// procedural) is previewed, measured, and routed to the inputs of the suite that can use it.
import { h, clear, btn, kpiGrid, dataTable, toast, importBtn, help } from './ui.js';
import { plotCard } from './plot.js';
import { fmt } from './num.js';
import { download, LIMITS } from './io.js';
import { FORMATS, PATHWAYS, SUITE_GEOMETRY, formatOf, importGeometry, sectionOf, gridOf, microstructure, networkSummary, dimensions, generate } from './geom.preview.js';
import { geometryLinks } from './geomlinks.js';

const attached = new Map(); // suite id -> Geometry (kept in memory for the session; files are never uploaded)
export const geometryOf = (id) => attached.get(id);
export function attachGeometry(id, g) { if (g) attached.set(id, g); else attached.delete(id); }

export const ACCEPT = [...new Set(FORMATS.flatMap((f) => f.ext.map((e) => '.' + e)))].join(',');
const pc = (spec) => plotCard(spec, { onDownload: download });
const safe = (fn, fb = null) => { try { return fn(); } catch { return fb; } };

/** Everything that can be measured from a geometry, computed once per import. */
export function derive(g) {
  if (g._derived) return g._derived;
  const d = { dims: safe(() => dimensions(g)), micro: null, net: null, grid: null };
  if (g.kind === 'voxels') d.micro = safe(() => microstructure(g));
  if (g.kind === 'network') d.net = safe(() => networkSummary(g));
  if (g.kind === 'grid') { const z = g.grid.z.flat().filter(Number.isFinite); d.grid = z.length ? { min: Math.min(...z), max: Math.max(...z), nx: g.grid.x.length, ny: g.grid.y.length } : null; }
  Object.defineProperty(g, '_derived', { value: d, enumerable: false });
  return d;
}

/** Read one or several files (companions such as .shp + .dbf or .mhd + .raw are matched by base name). */
export async function readFiles(fileList, opts = {}) {
  const files = [...fileList];
  if (!files.length) throw new Error('No file selected.');
  for (const f of files) if (f.size > LIMITS.fileBytes) throw new Error(`${f.name} is larger than ${LIMITS.fileBytes / 1e6} MB.`);
  const rank = (f) => { const m = formatOf(f.name); return m ? (m.support === 'convert' ? 1 : 0) : 2; };
  const companionExt = /\.(shx|dbf|prj|cpg|raw|hdr|img|bin|mtl)$/i;
  const main = files.filter((f) => !companionExt.test(f.name) || files.length === 1).sort((a, b) => rank(a) - rank(b))[0] || files[0];
  const companion = Object.fromEntries(files.filter((f) => f !== main).map((f) => [f.name, f]));
  return importGeometry(main, { ...opts, companion });
}

function preview(g) {
  const out = [];
  if (g.kind === 'grid') out.push(pc({ type: 'field', title: `Elevation grid — ${g.name}`, xlabel: g.grid.geographic ? 'Longitude (°)' : 'x', ylabel: g.grid.geographic ? 'Latitude (°)' : 'y', zlabel: 'Elevation', x: g.grid.x, y: g.grid.y, z: g.grid.z.map((r) => Array.from(r)), cmap: 'salinity', contours: 8 }));
  else if (g.kind === 'voxels') {
    const v = g.voxels, k = Math.floor(v.nz / 2), sx = Math.max(1, Math.ceil(v.nx / 220)), sy = Math.max(1, Math.ceil(v.ny / 160));
    const xs = [], ys = [], z = [];
    for (let i = 0; i < v.nx; i += sx) xs.push(i * v.spacing[0]);
    for (let j = 0; j < v.ny; j += sy) { ys.push(j * v.spacing[1]); const row = []; for (let i = 0; i < v.nx; i += sx) row.push(v.data[(k * v.ny + j) * v.nx + i] ? 1 : 0); z.push(row); }
    out.push(pc({ type: 'field', title: `Microstructure${v.nz > 1 ? ` — slice ${k + 1} of ${v.nz}` : ''} (1 = solid)`, xlabel: 'x', ylabel: 'y', zlabel: 'solid', x: xs, y: ys, z, cmap: 'viridis', equal: true, zmin: 0, zmax: 1 }));
  } else if (g.kind === 'points') {
    const n = g.count, step = Math.max(1, Math.ceil(n / 4000)), x = [], y = [];
    for (let i = 0; i < n; i += step) { x.push(g.points[3 * i]); y.push(g.points[3 * i + 1]); }
    out.push(pc({ type: 'line', title: `Point cloud — plan view (${Math.min(n, x.length).toLocaleString()} of ${n.toLocaleString()} points)`, xlabel: 'x', ylabel: 'y', series: [{ name: 'Points', x, y, mode: 'points', size: 1.5 }] }));
    const gr = safe(() => gridOf(g, 60, 45));
    if (gr) out.push(pc({ type: 'field', title: 'Surface gridded from the points', xlabel: 'x', ylabel: 'y', zlabel: 'z', x: gr.x, y: gr.y, z: gr.z, cmap: 'salinity', contours: 8 }));
  } else if (g.kind === 'network') {
    const nodes = new Map(g.network.nodes.map((n) => [n.id, n])), x = [], y = [];
    const placed = g.network.nodes.some((n) => Number.isFinite(n.x) && Number.isFinite(n.y) && (n.x || n.y));
    if (placed) {
      for (const e of g.network.edges) { const a = nodes.get(e.from), b = nodes.get(e.to); if (a && b) { x.push(a.x, b.x, NaN); y.push(a.y, b.y, NaN); } }
      out.push(pc({ type: 'line', title: 'Network layout (plan view)', xlabel: 'x', ylabel: 'y', series: [{ name: 'Connections', x, y }, { name: 'Equipment / nodes', x: g.network.nodes.map((n) => n.x), y: g.network.nodes.map((n) => n.y), mode: 'points', size: 4 }] }));
    }
  } else if (g.kind === 'mesh' || g.kind === 'polylines') {
    const segs = safe(() => sectionOf(g), []) || [], x = [], y = [];
    segs.slice(0, 8000).forEach((s) => { x.push(s[0], s[2], NaN); y.push(s[1], s[3], NaN); });
    if (segs.length) out.push(pc({ type: 'line', title: g.kind === 'mesh' ? 'Mid-plane section (used for 2-D simulation)' : 'Outlines', xlabel: 'x', ylabel: 'y', series: [{ name: 'Geometry', x, y }] }));
  }
  return out;
}

function facts(g) {
  const d = derive(g), k = [{ label: 'Format', value: g.format || '—' }, { label: 'Pathway', value: PATHWAYS[g.pathway]?.title || g.pathway || '—' }, { label: 'Read as', value: { mesh: 'Triangulated surface', polylines: '2-D outlines', points: 'Point cloud', grid: 'Elevation grid', voxels: 'Voxel microstructure', network: 'Equipment / piping network', table: 'Data table' }[g.kind] || g.kind }];
  if (g.count) k.push({ label: g.kind === 'points' ? 'Points' : 'Triangles', value: g.count });
  if (g.polylines) k.push({ label: 'Outlines', value: g.polylines.length });
  for (const [key, val] of Object.entries(g.stats || {})) if (typeof val === 'number' || typeof val === 'string') k.push({ label: key.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase()), value: val });
  if (d.dims?.size) k.push({ label: 'Extent', value: d.dims.size.filter((s) => Number.isFinite(s)).map((s) => fmt(s, 4)).join(' × ') });
  if (d.dims?.area) k.push({ label: 'Surface area', value: d.dims.area }); if (d.dims?.volume) k.push({ label: 'Volume', value: d.dims.volume, help: d.dims.closed === false ? 'The surface is not closed, so the volume is approximate.' : '' });
  if (d.micro) k.push({ label: 'Porosity', value: d.micro.porosity }, { label: 'Tortuosity', value: (d.micro.tortuosity || []).filter(Number.isFinite).map((t) => fmt(t, 3)).join(' / ') || '—' }, { label: 'Mean pore size', value: d.micro.meanPoreSize }, { label: 'Specific surface', value: d.micro.specificSurface, unit: '1/length' }, { label: 'Percolates', value: (d.micro.percolates || []).map((p) => (p ? 'yes' : 'no')).join(' / ') });
  if (d.net) k.push({ label: 'Nodes', value: d.net.nodes }, { label: 'Connections', value: d.net.edges }, { label: 'Total pipe length', value: d.net.totalLength, unit: 'm' });
  if (d.grid) k.push({ label: 'Lowest point', value: d.grid.min, unit: 'm' }, { label: 'Highest point', value: d.grid.max, unit: 'm' }, { label: 'Grid', value: `${d.grid.nx} × ${d.grid.ny}` });
  return k.slice(0, 18);
}

/** Card describing one imported geometry: facts, warnings, preview and network tables. */
export function geometryCard(g) {
  const d = derive(g), box = h('div');
  box.append(kpiGrid(facts(g)));
  if (g.warnings?.length) box.append(h('div', { class: 'warns' }, g.warnings.slice(0, 8).map((w) => h('div', { class: 'warn warn' }, h('b', null, 'Note'), ' ', String(w)))));
  const pv = preview(g);
  if (pv.length) box.append(h('div', { class: 'plots' }, pv));
  if (d.net) {
    box.append(dataTable({ title: 'Equipment inventory', columns: ['Type', 'Count'], rows: Object.entries(d.net.byType || {}).map(([t, n]) => [t, n]) }));
    if (d.net.pipes?.length) box.append(dataTable({ title: 'Connections', columns: ['From', 'To', 'Length (m)', 'Diameter (m)', 'Name'], rows: d.net.pipes.slice(0, 400).map((p) => [String(p.from), String(p.to), p.length ?? null, p.diameter ?? null, p.name || '']) }));
  }
  if (g.kind === 'table') box.append(dataTable({ title: 'Data preview', columns: g.headers, rows: g.records.slice(0, 15).map((r) => g.headers.map((hd) => r[hd])) }));
  return box;
}

/** Controls that build a procedural geometry (no file needed). */
export function generatorPanel(onGeometry) {
  const defs = {
    spacer: { label: 'Spacer-filled membrane channel (2-D section)', p: { height: [0.00071, 'Channel height (m)'], length: [0.012, 'Length (m)'], diameter: [0.00036, 'Filament diameter (m)'], pitch: [0.003, 'Filament pitch (m)'] }, sel: { arrangement: ['zigzag', 'cavity', 'submerged'] } },
    lattice: { label: 'Filament lattice / spacer unit cell', p: { pitch: [0.003, 'Pitch (m)'], diameter: [0.0004, 'Filament diameter (m)'], angle: [90, 'Crossing angle (°)'], n: [96, 'Resolution (cells)'] } },
    tpms: { label: 'Triply-periodic minimal surface', p: { period: [1, 'Period'], porosity: [0.6, 'Target porosity'], n: [64, 'Resolution (cells)'] }, sel: { surface: ['gyroid', 'schwarzP', 'diamond'] } },
    spheres: { label: 'Random packed grains (stochastic porous medium)', p: { porosity: [0.45, 'Target porosity'], radius: [0.06, 'Grain radius (fraction of box)'], n: [96, 'Resolution (cells)'], seed: [7, 'Random seed'] } },
    voronoi: { label: 'Voronoi foam / cellular structure', p: { cells: [30, 'Number of cells'], wall: [0.03, 'Wall thickness (fraction of box)'], n: [96, 'Resolution (cells)'], seed: [3, 'Random seed'] } },
  };
  const sel = h('select', { 'aria-label': 'Generator type' }, Object.entries(defs).map(([k, v]) => h('option', { value: k }, v.label))), form = h('div', { class: 'fields' });
  const paint = () => { const d = defs[sel.value]; clear(form); for (const [k, [val, label]] of Object.entries(d.p)) form.append(h('div', { class: 'field' }, h('label', null, label), h('div', { class: 'ctl' }, h('input', { type: 'number', step: 'any', value: val, dataset: { k } })))); for (const [k, opts] of Object.entries(d.sel || {})) form.append(h('div', { class: 'field' }, h('label', null, k[0].toUpperCase() + k.slice(1)), h('div', { class: 'ctl' }, h('select', { dataset: { k } }, opts.map((o) => h('option', { value: o }, o)))))); };
  sel.addEventListener('change', paint); paint();
  return h('div', null, h('div', { class: 'row-tools' }, sel, btn('Generate geometry', () => {
    try { const spec = { type: sel.value }; for (const el of form.querySelectorAll('[data-k]')) spec[el.dataset.k] = el.tagName === 'SELECT' ? el.value : +el.value; const g = generate(spec); g.name ||= defs[sel.value].label; g.format ||= 'Procedural generator'; g.pathway ||= 'procedural'; onGeometry(g); }
    catch (e) { toast('Could not generate: ' + e.message, 'bad'); }
  }, 'primary')), form);
}

/** Supported-format catalogue grouped by pathway. */
export function formatCatalogue() {
  const badge = { full: ['reads directly', 'ok'], partial: ['reads (partial)', 'warn'], convert: ['convert first', ''] };
  return h('div', null, Object.entries(PATHWAYS).map(([id, p]) => {
    const list = FORMATS.filter((f) => f.pathway === id);
    if (!list.length) return null;
    return h('details', { class: 'ref' }, h('summary', null, `${p.title} — ${list.filter((f) => f.support !== 'convert').length} read directly, ${list.filter((f) => f.support === 'convert').length} via conversion`), h('p', null, p.blurb),
      h('div', { class: 'tbl-scroll' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Format', 'Extensions', 'Support', 'What happens'].map((x) => h('th', null, x)))),
        h('tbody', null, list.map((f) => h('tr', null, h('td', { class: 'lead' }, f.name), h('td', { class: 'lead' }, f.ext.map((e) => '.' + e).join(' ')), h('td', { class: 'lead' }, h('span', { class: 'badge ' + badge[f.support][1] }, badge[f.support][0])), h('td', { class: 'lead wrap' }, f.support === 'convert' ? f.convert : f.note)))))));
  }));
}

/**
 * The Geometry tab of a suite. api = { fields, values(), setValue(key, value), rerender() }.
 */
export function geometryTab(suite, api) {
  const info = SUITE_GEOMETRY[suite.id] || { classes: '', accepts: [] }, box = h('div', { class: 'groups' }), body = h('div');
  const paint = () => {
    const g = attached.get(suite.id);
    clear(body);
    if (!g) return body.append(h('p', { class: 'note' }, 'No geometry attached to this suite yet. Import a file here, generate one, or send one from the Data portal. Without geometry the suite uses its built-in parametric dimensions.'));
    const items = safe(() => geometryLinks(suite, g, derive(g)), []) || [];
    body.append(h('h3', null, g.name || 'Geometry'), geometryCard(g));
    if (items.length) body.append(h('fieldset', { class: 'group' }, h('legend', null, 'What this suite takes from the geometry'),
      h('ul', { class: 'linklist' }, items.map((it) => h('li', null, h('b', null, api.fields.find((f) => f.key === it.key)?.label || it.key), ' ← ', typeof it.value === 'number' ? fmt(it.value) : 'geometry data', h('small', null, ' · ' + it.from)))),
      h('div', { class: 'row-tools' }, btn('Apply to inputs', () => { for (const it of items) api.setValue(it.key, it.value); toast(`${items.length} input${items.length > 1 ? 's' : ''} set from the geometry.`, 'ok'); }, 'primary'))));
    else body.append(h('p', { class: 'note' }, `This geometry was read correctly, but ${suite.short || suite.title} has no input that a “${g.kind}” geometry can set directly. Its measured dimensions above can be typed into the inputs, or send it to a suite that uses this class of geometry from the Data portal.`));
    body.append(h('div', { class: 'row-tools' }, btn('Remove geometry', () => { attached.delete(suite.id); for (const f of api.fields.filter((x) => x.type === 'file')) api.setValue(f.key, null); paint(); }, 'ghost')));
  };
  const take = (g) => { attached.set(suite.id, g); const items = safe(() => geometryLinks(suite, g, derive(g)), []) || []; for (const it of items) api.setValue(it.key, it.value); toast(`${g.name} attached${items.length ? ` — ${items.length} input${items.length > 1 ? 's' : ''} set` : ''}.`, 'ok'); paint(); };
  const inp = h('input', { type: 'file', hidden: true, multiple: true, accept: ACCEPT });
  inp.addEventListener('change', async () => { const fl = [...inp.files]; inp.value = ''; if (!fl.length) return; try { take(await readFiles(fl)); } catch (e) { toast(e.message || 'Could not read the file.', 'bad', 12000); } });
  paint();
  box.append(
    h('p', { class: 'summary' }, 'Geometry for this suite: ', info.classes || 'parametric dimensions on the Inputs tab.'),
    h('fieldset', { class: 'group' }, h('legend', null, 'Attach geometry', help('Files are read on this device only. Select companion files together (for example .shp with .dbf, or .mhd with .raw).')),
      h('div', { class: 'row-tools' }, inp, btn('Import geometry file…', () => inp.click(), 'primary'), h('a', { class: 'btn', href: '#/portal' }, 'Open the Data portal')),
      h('details', { class: 'ref' }, h('summary', null, 'Or generate a geometry procedurally'), generatorPanel(take))),
    body,
    h('details', { class: 'ref' }, h('summary', null, 'All supported geometry formats'), formatCatalogue()));
  return box;
}
