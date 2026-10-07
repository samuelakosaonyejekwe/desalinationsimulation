// Data portal back-end: tabular files (CSV/TSV/JSON/XLSX), geometry (STL/OBJ/DXF/GeoJSON) and downloads.
// Everything is parsed locally in the browser. Files are size-limited and never executed or uploaded.

export const LIMITS = { fileBytes: 60e6, rows: 200000, triangles: 400000 };

export function checkFile(file) {
  if (!file) throw new Error('No file selected.');
  if (file.size > LIMITS.fileBytes) throw new Error(`File is larger than ${LIMITS.fileBytes / 1e6} MB.`);
  return file;
}
export const extOf = (name) => (String(name).toLowerCase().match(/\.([a-z0-9]+)$/) || [, ''])[1];

/** RFC-4180 style parser with automatic delimiter detection. Returns array of string rows. */
export function parseDelimited(text, delim) {
  text = text.replace(/^﻿/, '');
  if (!delim) {
    const head = text.slice(0, 4000).split(/\r?\n/)[0] || '';
    delim = [',', ';', '\t', '|'].map((d) => [d, head.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
  }
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; }
      else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === delim) { row.push(cell); cell = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); cell = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
      if (rows.length > LIMITS.rows) throw new Error('Too many rows.');
    } else cell += ch;
  }
  row.push(cell);
  if (row.length > 1 || row[0] !== '') rows.push(row);
  return rows;
}

const toNum = (s) => {
  if (typeof s === 'number') return s;
  const t = String(s).trim().replace(/\s/g, '');
  if (t === '') return null;
  const n = Number(/^-?\d{1,3}(\.\d{3})*,\d+$/.test(t) ? t.replace(/\./g, '').replace(',', '.') : /^-?\d+,\d+$/.test(t) ? t.replace(',', '.') : t);
  return Number.isFinite(n) ? n : null;
};

/** Rows -> { headers, records } where numeric-looking cells become numbers. */
export function tableFromRows(rows) {
  if (!rows.length) return { headers: [], records: [] };
  const firstNumeric = rows[0].every((c) => toNum(c) !== null);
  const headers = firstNumeric ? rows[0].map((_, i) => 'col' + (i + 1)) : rows[0].map((h, i) => String(h).trim() || 'col' + (i + 1));
  const body = firstNumeric ? rows : rows.slice(1);
  const records = body.map((r) => Object.fromEntries(headers.map((h, i) => { const n = toNum(r[i] ?? ''); return [h, n === null ? String(r[i] ?? '').trim() : n]; })));
  return { headers, records };
}

export function toCSV(headers, records) {
  const esc = (v) => { const s = v === null || v === undefined ? '' : String(v); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [headers.map(esc).join(','), ...records.map((r) => (Array.isArray(r) ? r : headers.map((h) => r[h])).map(esc).join(','))].join('\n');
}

// ---- XLSX (zip + deflate + XML), first worksheet only -------------------------------------------
async function inflateRaw(bytes) {
  const ds = new DecompressionStream('deflate-raw');
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
async function unzip(buf, wanted) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf), out = {};
  let eocd = -1;
  for (let i = u8.length - 22; i >= Math.max(0, u8.length - 70000); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('Not a valid .xlsx (zip) file.');
  const n = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const dec = new TextDecoder();
  for (let k = 0; k < n; k++) {
    if (dv.getUint32(p, true) !== 0x02014b50) break;
    const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true), usize = dv.getUint32(p + 24, true);
    const nl = dv.getUint16(p + 28, true), el = dv.getUint16(p + 30, true), cl = dv.getUint16(p + 32, true), lho = dv.getUint32(p + 42, true);
    const name = dec.decode(u8.subarray(p + 46, p + 46 + nl));
    if (wanted(name)) {
      if (usize > LIMITS.fileBytes * 4) throw new Error('Workbook is too large.');
      const lnl = dv.getUint16(lho + 26, true), lel = dv.getUint16(lho + 28, true), start = lho + 30 + lnl + lel;
      const data = u8.subarray(start, start + csize);
      out[name] = dec.decode(method === 0 ? data : await inflateRaw(data));
    }
    p += 46 + nl + el + cl;
  }
  return out;
}
export async function parseXLSX(buf) {
  const files = await unzip(buf, (n) => n === 'xl/sharedStrings.xml' || /^xl\/worksheets\/sheet\d+\.xml$/.test(n));
  const xml = (s) => new DOMParser().parseFromString(s, 'application/xml');
  const shared = files['xl/sharedStrings.xml'] ? [...xml(files['xl/sharedStrings.xml']).getElementsByTagName('si')].map((si) => [...si.getElementsByTagName('t')].map((t) => t.textContent).join('')) : [];
  const sheetName = Object.keys(files).filter((n) => n.includes('worksheets')).sort((a, b) => a.length - b.length || a.localeCompare(b))[0];
  if (!sheetName) throw new Error('No worksheet found.');
  const rows = [];
  for (const r of xml(files[sheetName]).getElementsByTagName('row')) {
    const row = [];
    for (const c of r.getElementsByTagName('c')) {
      const ref = c.getAttribute('r') || '', letters = ref.replace(/\d+/g, '');
      let col = 0;
      for (const ch of letters) col = col * 26 + (ch.charCodeAt(0) - 64);
      const t = c.getAttribute('t'), v = c.getElementsByTagName('v')[0]?.textContent ?? '';
      const val = t === 's' ? shared[+v] ?? '' : t === 'inlineStr' ? c.getElementsByTagName('t')[0]?.textContent ?? '' : v;
      row[Math.max(0, col - 1)] = val;
    }
    rows.push(Array.from(row, (x) => x ?? ''));
    if (rows.length > LIMITS.rows) throw new Error('Too many rows.');
  }
  const width = Math.max(0, ...rows.map((r) => r.length));
  return rows.filter((r) => r.some((c) => c !== '')).map((r) => Array.from({ length: width }, (_, i) => r[i] ?? ''));
}

/** Read any supported tabular file -> { headers, records }. */
export async function readTable(file) {
  checkFile(file);
  const ext = extOf(file.name);
  if (ext === 'xlsx' || ext === 'xlsm') return tableFromRows(await parseXLSX(await file.arrayBuffer()));
  const text = await file.text();
  if (ext === 'json') {
    const j = JSON.parse(text), arr = Array.isArray(j) ? j : Array.isArray(j?.data) ? j.data : Array.isArray(j?.records) ? j.records : null;
    if (!arr) throw new Error('JSON must be an array of records.');
    if (Array.isArray(arr[0])) return tableFromRows(arr.map((r) => r.map(String)));
    const headers = [...new Set(arr.slice(0, 200).flatMap((r) => Object.keys(r || {})))].filter((k) => k !== '__proto__');
    return { headers, records: arr.slice(0, LIMITS.rows).map((r) => Object.fromEntries(headers.map((h) => [h, typeof r[h] === 'number' ? r[h] : toNum(r[h] ?? '') ?? String(r[h] ?? '')]))) };
  }
  return tableFromRows(parseDelimited(text, ext === 'tsv' ? '\t' : undefined));
}

// ---- Geometry ------------------------------------------------------------------------------------
function bbox3(t) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < t.length; i += 3) for (let c = 0; c < 3; c++) { if (t[i + c] < lo[c]) lo[c] = t[i + c]; if (t[i + c] > hi[c]) hi[c] = t[i + c]; }
  return { min: lo, max: hi };
}
export function parseSTL(buf) {
  const u8 = new Uint8Array(buf), dv = new DataView(buf);
  const nBin = u8.length >= 84 ? dv.getUint32(80, true) : 0, isBin = u8.length === 84 + nBin * 50;
  let tri;
  if (isBin) {
    if (nBin > LIMITS.triangles) throw new Error('Mesh has too many triangles.');
    tri = new Array(nBin * 9);
    for (let i = 0; i < nBin; i++) for (let k = 0; k < 9; k++) tri[i * 9 + k] = dv.getFloat32(84 + i * 50 + 12 + k * 4, true);
  } else {
    const text = new TextDecoder().decode(u8), re = /vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)/g;
    tri = [];
    let m;
    while ((m = re.exec(text))) { tri.push(+m[1], +m[2], +m[3]); if (tri.length > LIMITS.triangles * 9) throw new Error('Mesh has too many triangles.'); }
    tri.length -= tri.length % 9;
  }
  if (!tri.length || tri.some((v) => !Number.isFinite(v))) throw new Error('Could not read any triangles from the STL file.');
  return { kind: 'mesh', triangles: tri, count: tri.length / 9, bbox: bbox3(tri) };
}
export function parseOBJ(text) {
  const vs = [], tri = [];
  for (const line of text.split(/\r?\n/)) {
    const p = line.trim().split(/\s+/);
    if (p[0] === 'v') vs.push([+p[1], +p[2], +p[3] || 0]);
    else if (p[0] === 'f') {
      const idx = p.slice(1).map((s) => { const i = parseInt(s, 10); return i < 0 ? vs.length + i : i - 1; });
      for (let k = 1; k + 1 < idx.length; k++) for (const q of [idx[0], idx[k], idx[k + 1]]) { const v = vs[q]; if (v) tri.push(v[0], v[1], v[2]); }
      if (tri.length > LIMITS.triangles * 9) throw new Error('Mesh has too many triangles.');
    }
  }
  tri.length -= tri.length % 9;
  if (!tri.length) throw new Error('No faces found in the OBJ file.');
  return { kind: 'mesh', triangles: tri, count: tri.length / 9, bbox: bbox3(tri) };
}
/** Minimal DXF reader: LINE, LWPOLYLINE, POLYLINE/VERTEX and CIRCLE entities -> 2-D polylines. */
export function parseDXF(text) {
  const lines = text.split(/\r?\n/), pairs = [];
  for (let i = 0; i + 1 < lines.length; i += 2) pairs.push([parseInt(lines[i], 10), lines[i + 1].trim()]);
  const polys = [];
  let cur = null, type = null, inEnt = false;
  const flush = () => {
    if (!cur) return;
    if (type === 'LINE' && cur.x.length === 2) polys.push({ x: cur.x, y: cur.y, closed: false });
    else if (type === 'CIRCLE' && cur.x.length) { const n = 48, x = [], y = []; for (let k = 0; k < n; k++) { x.push(cur.x[0] + cur.r * Math.cos((2 * Math.PI * k) / n)); y.push(cur.y[0] + cur.r * Math.sin((2 * Math.PI * k) / n)); } polys.push({ x, y, closed: true }); }
    else if ((type === 'LWPOLYLINE' || type === 'POLYLINE') && cur.x.length > 1) polys.push({ x: cur.x, y: cur.y, closed: !!(cur.flag & 1) });
    cur = null;
  };
  for (const [code, val] of pairs) {
    if (code === 0) {
      if (val === 'SECTION') continue;
      if (val === 'VERTEX' && type === 'POLYLINE') continue;
      if (val === 'SEQEND') { flush(); type = null; continue; }
      if (type !== 'POLYLINE' || val !== 'VERTEX') flush();
      type = val; inEnt = ['LINE', 'LWPOLYLINE', 'POLYLINE', 'CIRCLE'].includes(val);
      cur = inEnt ? { x: [], y: [], r: 0, flag: 0 } : null;
    } else if (cur) {
      if (code === 10 || code === 11) cur.x.push(+val);
      else if (code === 20 || code === 21) cur.y.push(+val);
      else if (code === 40) cur.r = +val;
      else if (code === 70 && !cur.flag) cur.flag = parseInt(val, 10) || 0;
    }
  }
  flush();
  const good = polys.filter((p) => p.x.length === p.y.length && p.x.every(Number.isFinite) && p.y.every(Number.isFinite));
  if (!good.length) throw new Error('No LINE / POLYLINE / CIRCLE entities found in the DXF file.');
  return { kind: 'polylines', polylines: good, bbox: bbox2(good) };
}
function bbox2(polys) {
  const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of polys) for (let i = 0; i < p.x.length; i++) { lo[0] = Math.min(lo[0], p.x[i]); hi[0] = Math.max(hi[0], p.x[i]); lo[1] = Math.min(lo[1], p.y[i]); hi[1] = Math.max(hi[1], p.y[i]); }
  return { min: lo, max: hi };
}
export function parseGeoJSON(text) {
  const g = JSON.parse(text), polys = [];
  const ring = (c, closed) => { if (Array.isArray(c) && c.length > 1) polys.push({ x: c.map((p) => +p[0]), y: c.map((p) => +p[1]), closed }); };
  const geom = (o) => {
    if (!o) return;
    if (o.type === 'FeatureCollection') (o.features || []).forEach((f) => geom(f.geometry));
    else if (o.type === 'Feature') geom(o.geometry);
    else if (o.type === 'GeometryCollection') (o.geometries || []).forEach(geom);
    else if (o.type === 'Polygon') o.coordinates.forEach((r) => ring(r, true));
    else if (o.type === 'MultiPolygon') o.coordinates.forEach((p) => p.forEach((r) => ring(r, true)));
    else if (o.type === 'LineString') ring(o.coordinates, false);
    else if (o.type === 'MultiLineString') o.coordinates.forEach((r) => ring(r, false));
  };
  geom(g);
  if (!polys.length) throw new Error('No polygon or line geometry found in the GeoJSON file.');
  return { kind: 'polylines', polylines: polys, bbox: bbox2(polys) };
}

/** Read any supported geometry file. */
export async function readGeometry(file) {
  checkFile(file);
  const ext = extOf(file.name);
  let g;
  if (ext === 'stl') g = parseSTL(await file.arrayBuffer());
  else if (ext === 'obj') g = parseOBJ(await file.text());
  else if (ext === 'dxf') g = parseDXF(await file.text());
  else if (ext === 'json' || ext === 'geojson') g = parseGeoJSON(await file.text());
  else if (ext === 'csv' || ext === 'txt' || ext === 'xy') {
    const rows = parseDelimited(await file.text()).map((r) => r.map(Number)).filter((r) => Number.isFinite(r[0]) && Number.isFinite(r[1]));
    if (rows.length < 3) throw new Error('Point file needs at least three x,y rows.');
    const p = [{ x: rows.map((r) => r[0]), y: rows.map((r) => r[1]), closed: true }];
    g = { kind: 'polylines', polylines: p, bbox: bbox2(p) };
  } else throw new Error(`Unsupported geometry format ".${ext}". Use STL, OBJ, DXF, GeoJSON or an x,y point file (export STEP/IGES from CAD as STL).`);
  g.name = file.name;
  return g;
}

/** Cut a triangle mesh with the plane {axis = value}; returns 2-D segments [x1,y1,x2,y2] in the two remaining axes. */
export function sliceMesh(mesh, axis = 2, value) {
  const t = mesh.triangles, a = [0, 1, 2].filter((k) => k !== axis), segs = [];
  if (value === undefined) value = 0.5 * (mesh.bbox.min[axis] + mesh.bbox.max[axis]);
  for (let i = 0; i < t.length; i += 9) {
    const P = [[t[i], t[i + 1], t[i + 2]], [t[i + 3], t[i + 4], t[i + 5]], [t[i + 6], t[i + 7], t[i + 8]]], pts = [];
    for (let e = 0; e < 3; e++) {
      const p = P[e], q = P[(e + 1) % 3], dp = p[axis] - value, dq = q[axis] - value;
      if ((dp < 0) !== (dq < 0)) { const s = dp / (dp - dq); pts.push([p[a[0]] + s * (q[a[0]] - p[a[0]]), p[a[1]] + s * (q[a[1]] - p[a[1]])]); }
    }
    if (pts.length === 2) segs.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
  }
  return segs;
}

/** Polylines -> segment list. */
export function polylinesToSegments(polys) {
  const segs = [];
  for (const p of polys) {
    for (let i = 0; i + 1 < p.x.length; i++) segs.push([p.x[i], p.y[i], p.x[i + 1], p.y[i + 1]]);
    if (p.closed && p.x.length > 2) segs.push([p.x[p.x.length - 1], p.y[p.y.length - 1], p.x[0], p.y[0]]);
  }
  return segs;
}

/**
 * Even-odd scan-line rasterisation of closed 2-D outlines onto a cell-centred grid.
 * Returns mask[j][i] = true where the cell centre lies inside the solid.
 */
export function rasterize(segs, x0, x1, y0, y1, nx, ny) {
  const mask = Array.from({ length: ny }, () => new Array(nx).fill(false)), dx = (x1 - x0) / nx, dy = (y1 - y0) / ny;
  for (let j = 0; j < ny; j++) {
    const y = y0 + (j + 0.5) * dy, xs = [];
    for (const s of segs) {
      const ya = s[1], yb = s[3];
      if ((ya <= y) !== (yb <= y)) xs.push(s[0] + ((y - ya) / (yb - ya)) * (s[2] - s[0]));
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const i0 = Math.max(0, Math.ceil((xs[k] - x0) / dx - 0.5)), i1 = Math.min(nx - 1, Math.floor((xs[k + 1] - x0) / dx - 0.5));
      for (let i = i0; i <= i1; i++) mask[j][i] = true;
    }
  }
  return mask;
}

/** Trigger a browser download of a Blob or string. */
export function download(data, filename, type = 'application/octet-stream') {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = document.createElement('a'), url = URL.createObjectURL(blob);
  a.href = url; a.download = String(filename).replace(/[^\w.\- ()]+/g, '_').slice(0, 120);
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
