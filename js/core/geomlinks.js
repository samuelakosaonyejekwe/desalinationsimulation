// Routing of imported geometry onto suite inputs. Each rule looks at what was imported (surface,
// outlines, elevation grid, voxel microstructure, network, parameter table) and offers values only
// for inputs that the suite really has. Every offer names where the number came from.
import { sectionOf, gridOf } from './geom.js';

const UNIT = { mm: 1e-3, millimetre: 1e-3, cm: 1e-2, m: 1, metre: 1, in: 0.0254, inch: 0.0254, ft: 0.3048, um: 1e-6, micron: 1e-6 };
const toM = (g) => UNIT[String(g.stats?.units || 'm').toLowerCase()] ?? 1;
const fin = (x) => typeof x === 'number' && Number.isFinite(x);
const between = (x, a, b) => fin(x) && x >= a && x <= b;

/** Outlines of any geometry as an object the 2-D solvers accept. */
export function asOutline(g) {
  if (g.kind === 'mesh' || g.kind === 'polylines') return g;
  const segs = sectionOf(g) || [];
  if (!segs.length) return null;
  const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const s of segs) { lo[0] = Math.min(lo[0], s[0], s[2]); hi[0] = Math.max(hi[0], s[0], s[2]); lo[1] = Math.min(lo[1], s[1], s[3]); hi[1] = Math.max(hi[1], s[1], s[3]); }
  return { kind: 'polylines', name: g.name, polylines: segs.map((s) => ({ x: [s[0], s[2]], y: [s[1], s[3]], closed: false })), bbox: { min: lo, max: hi } };
}

/** Elevation grid of a DEM, sounding cloud or terrain surface in the form the sea-discharge suite reads. */
export function asBathy(g) {
  if (!['grid', 'points', 'mesh'].includes(g.kind)) return null;
  const gr = g.kind === 'grid' && g.grid.x.length <= 120 && g.grid.y.length <= 120 ? { x: g.grid.x, y: g.grid.y, z: g.grid.z } : gridOf(g, 60, 60);
  if (!gr) return null;
  const x = Array.from(gr.x), y = Array.from(gr.y), elev = gr.z.map((r) => Array.from(r, (v) => (Number.isFinite(v) ? v : 0)));
  const geo = g.kind === 'grid' && g.grid.geographic !== undefined ? g.grid.geographic : Math.abs(x[0]) <= 180 && Math.abs(y[0]) <= 90 && Math.abs(x[x.length - 1] - x[0]) < 5;
  return geo ? { lat: y, lon: x, elev, name: g.name } : { x, y, elev, name: g.name };
}

function polylineLength(g) {
  if (g.kind !== 'polylines') return null;
  const geo = g.stats?.geographic || (g.bbox && Math.abs(g.bbox.min[0]) <= 180 && Math.abs(g.bbox.max[0]) <= 180 && Math.abs(g.bbox.min[1]) <= 90 && Math.abs(g.bbox.max[1]) <= 90 && g.pathway === 'gis');
  let L = 0;
  for (const p of g.polylines) for (let i = 1; i < p.x.length; i++) {
    if (geo) { const la = ((p.y[i] + p.y[i - 1]) / 2) * (Math.PI / 180); L += Math.hypot((p.x[i] - p.x[i - 1]) * 111320 * Math.cos(la), (p.y[i] - p.y[i - 1]) * 110540); }
    else L += Math.hypot(p.x[i] - p.x[i - 1], p.y[i] - p.y[i - 1]) * toM(g);
  }
  return L;
}

/** Name/value parameter tables and single-row tables whose columns match input keys. */
function fromTable(g, keys, fields) {
  if (g.kind !== 'table' || !g.records?.length) return [];
  const out = [], norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, ''), byNorm = new Map();
  for (const f of fields) { if (f.type && f.type !== 'number') continue; byNorm.set(norm(f.key), f.key); byNorm.set(norm(f.label), f.key); }
  const hd = g.headers;
  if (hd.length >= 2 && g.records.every((r) => typeof r[hd[0]] === 'string')) { for (const r of g.records) { const k = byNorm.get(norm(r[hd[0]])); if (k && fin(r[hd[1]])) out.push({ key: k, value: r[hd[1]], from: `Parameter “${r[hd[0]]}” in ${g.name}` }); } }
  else for (const col of hd) { const k = byNorm.get(norm(col)); if (k && fin(g.records[0][col])) out.push({ key: k, value: g.records[0][col], from: `Column “${col}” in ${g.name}` }); }
  return out;
}

const RULES = {
  cfd(g, d, has) {
    const out = [], o = asOutline(g);
    if (o && has('cad')) { out.push({ key: 'cad', value: o, from: `${g.format}: ${g.kind === 'mesh' ? 'mid-plane section of the surface' : g.kind === 'voxels' ? 'outline of the voxel slice' : 'imported outlines'}` }); if (has('geom')) out.push({ key: 'geom', value: 'import', from: 'Use the imported geometry as the flow obstacle' }); }
    return out;
  },
  sea(g, d, has) {
    const out = [], b = has('bathy') ? asBathy(g) : null;
    if (b) {
      out.push({ key: 'bathy', value: b, from: `${g.format}: seabed elevation grid ${(b.lon || b.x).length} × ${(b.lat || b.y).length}` });
      const mid = b.elev[Math.floor(b.elev.length / 2)][Math.floor(b.elev[0].length / 2)];
      if (has('depth') && fin(mid) && mid < 0) out.push({ key: 'depth', value: -mid, from: 'Water depth at the centre of the imported bathymetry' });
    }
    const L = polylineLength(g);
    if (L && has('outfallLength') && between(L, 5, 2e4)) out.push({ key: 'outfallLength', value: L, from: 'Length of the imported outfall route' });
    return out;
  },
  pump(g, d, has) {
    const out = [];
    if (d.net) {
      if (has('Lint') && between(d.net.totalLength, 1, 2e5)) out.push({ key: 'Lint', value: d.net.totalLength, from: `Total pipe length of ${d.net.edges} connections` });
      const dia = (d.net.pipes || []).map((p) => p.diameter).filter((x) => between(x, 0.01, 5)).sort((a, b) => a - b);
      if (has('Dint') && dia.length) out.push({ key: 'Dint', value: dia[Math.floor(dia.length / 2)], from: 'Median pipe diameter in the network' });
      if (has('zStatic') && between(d.net.maxElevationChange, 0.01, 500)) out.push({ key: 'zStatic', value: d.net.maxElevationChange, from: 'Largest elevation difference between nodes' });
    }
    const L = polylineLength(g);
    if (L && has('Lint') && between(L, 1, 2e5) && !d.net) out.push({ key: 'Lint', value: L, from: 'Length of the imported pipe route' });
    return out;
  },
  econ(g, d, has) {
    const L = polylineLength(g);
    return L && has('outfallLength') && between(L, 5, 5e4) ? [{ key: 'outfallLength', value: L, from: 'Length of the imported intake / outfall route' }] : [];
  },
  ro(g, d, has) {
    const out = [], s = d.dims?.size?.filter(fin).map((x) => x * toM(g)).sort((a, b) => a - b);
    if (s?.length && has('spacerMil') && between(s[0], 0.0004, 0.0017) && (g.kind === 'mesh' || g.kind === 'polylines' || g.kind === 'voxels')) out.push({ key: 'spacerMil', value: +(s[0] / 25.4e-6).toFixed(1), from: 'Smallest extent of the imported spacer / channel geometry' });
    return out;
  },
  ed(g, d, has) {
    const out = [], s = d.dims?.size?.filter(fin).map((x) => x * toM(g)).sort((a, b) => b - a);
    if (s?.length >= 2 && (g.kind === 'mesh' || g.kind === 'polylines')) {
      if (has('Lpath') && between(s[0], 0.05, 5)) out.push({ key: 'Lpath', value: s[0], from: 'Longest extent of the imported cell geometry' });
      if (has('W') && between(s[1], 0.02, 3) && s.length >= 3) out.push({ key: 'W', value: s[1], from: 'Width of the imported cell geometry' });
      if (has('hsp') && between(s[s.length - 1], 1e-4, 5e-3)) out.push({ key: 'hsp', value: s[s.length - 1] * 1000, from: 'Thickness of the imported flow channel' });
    }
    if (d.micro && has('eps') && between(d.micro.porosity, 0.05, 0.99)) out.push({ key: 'eps', value: d.micro.porosity, from: 'Open fraction of the imported spacer structure' });
    return out;
  },
  fomd(g, d, has) {
    const out = [], m = d.micro;
    if (!m) return out;
    const tau = (m.tortuosity || []).filter((t) => between(t, 1, 20));
    for (const k of ['epsM']) if (has(k) && between(m.porosity, 0.02, 0.98)) out.push({ key: k, value: m.porosity, from: 'Porosity of the imported membrane microstructure' });
    if (has('dPore') && between(m.meanPoreSize * toM(g) * 1e6, 0.01, 5)) out.push({ key: 'dPore', value: m.meanPoreSize * toM(g) * 1e6, from: 'Mean pore size (chord length) of the imported microstructure' });
    for (const k of ['tauM']) if (has(k) && tau.length) out.push({ key: k, value: Math.max(...tau), from: 'Through-thickness tortuosity of the imported microstructure' });
    return out;
  },
  fouling(g, d, has) {
    return d.micro && has('eps') && between(d.micro.porosity, 0.05, 0.95) ? [{ key: 'eps', value: d.micro.porosity, from: 'Porosity of the imaged deposit / cake layer' }] : [];
  },
};

/** Offers for one suite. Returns [{ key, value, from }] restricted to inputs the suite declares. */
export function geometryLinks(suite, g, derived) {
  const fields = suite.inputs.flatMap((grp) => grp.fields), keys = new Set(fields.map((f) => f.key)), has = (k) => keys.has(k);
  const out = [...(RULES[suite.id]?.(g, derived, has) || []), ...fromTable(g, keys, fields)];
  const custom = suite.geometry?.(g, derived) || [];
  const seen = new Set();
  return [...custom, ...out].filter((it) => it && keys.has(it.key) && it.value !== null && it.value !== undefined && !seen.has(it.key) && seen.add(it.key));
}
