// Look-up functions of the built-in world atlas. This module and the data modules behind it are loaded on
// demand (dynamic import) only when a live data service cannot be reached, so they cost nothing at start-up.
// Each data set is fetched and decoded the first time it is needed:
//   atlas_ocean.js    sea-surface salinity (SeaDataCloud, 1 degree) and temperature (CoralTemp, 0.5 degree near land), seasonal cycle
//   atlas_relief.js   seabed depth / land elevation, 0.25 degree world grid + 0.05 degree coastal tiles
//   atlas_coast.js    tides, currents and waves of coastal waters
//   atlas_nations.js  country outlines, national economic and energy figures, exchange rates
//   atlas_climate.js  long-term solar irradiation, wind speed and air temperature
// All of them are written by tools/atlas_build.py from open global sources (see the header of each file).
import { unpack, asI8, asI16, asU16, asI32 } from './atlas_codec.js';

const D2R = Math.PI / 180;
const wrap = (j, n) => ((j % n) + n) % n;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
/** Great-circle distance (km). */
export function distKm(la1, lo1, la2, lo2) {
  const a = Math.sin(((la2 - la1) * D2R) / 2) ** 2 + Math.cos(la1 * D2R) * Math.cos(la2 * D2R) * Math.sin(((lo2 - lo1) * D2R) / 2) ** 2;
  return 12742 * Math.asin(Math.min(1, Math.sqrt(a)));
}

const loaders = {
  ocean: () => import('./atlas_ocean.js').then((m) => {
    const O = m.OCEAN, n = O.nlat * O.nlon;
    const sh = unpack(O.sh), th = unpack(O.th);
    const F = O.fine, nf = F.nlat * F.nlon;
    return { ...O, S0: asU16(unpack(O.s0), 0, n), SH: asI8(sh, 0, 4 * n), T0: asI16(unpack(O.t0), 0, n), TH: asI8(th, 0, 4 * n), fine: { ...F, T0: asI16(unpack(F.t0), 0, nf), TH: asI8(unpack(F.th), 0, 4 * nf) } };
  }),
  relief: () => import('./atlas_relief.js').then((m) => {
    const R = m.RELIEF, B = R.base, T = R.tiles;
    const q = asI8(unpack(B.data), 0, B.nlat * B.nlon);
    for (let i = 0; i < B.nlat; i++) for (let j = 1, o = i * B.nlon; j < B.nlon; j++) q[o + j] = (q[o + j] + q[o + j - 1]) << 24 >> 24;
    const ids = asI32(unpack(T.ids), 0, T.count);
    for (let i = 1; i < T.count; i++) ids[i] += ids[i - 1];
    const raw = asI8(unpack(T.data), 0, T.count * 25), tq = new Int8Array(T.count * 25);
    for (let t = 0; t < T.count; t++) {
      const o = t * 25;
      let acc = 0;
      for (let k = 0; k < 25; k++) { acc = (k ? acc + raw[o + k] : raw[o]) << 24 >> 24; const r = (k / 5) | 0, c = k % 5; tq[o + r * 5 + (r % 2 ? 4 - c : c)] = acc; }
    }
    return { source: R.source, vintage: R.vintage, B, q, ids, tq, kb: B.k, kt: T.k };
  }),
  coast: () => import('./atlas_coast.js').then((m) => {
    const C = m.COAST;
    return { ...C, seaRows: asI16(unpack(C.sea.data), 0, C.sea.rows * C.sea.cols), waveRows: asI16(unpack(C.wave.data), 0, C.wave.rows * C.wave.cols) };
  }),
  nations: () => import('./atlas_nations.js').then((m) => {
    const N = m.NATIONS, polys = [], byCode = {}, raw = unpack(N.rings), v = asI16(raw, 0, raw.length / 2);
    let o = 0;
    for (const c of N.list) {
      byCode[c.a2] = c;
      for (let r = 0; r < c.rings; r++) {
        const n = v[o++], xs = new Float32Array(n), ys = new Float32Array(n);
        let x = 0, y = 0, x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
        for (let k = 0; k < n; k++) { x += v[o++]; y += v[o++]; const fx = x / N.scale, fy = y / N.scale; xs[k] = fx; ys[k] = fy; if (fx < x0) x0 = fx; if (fx > x1) x1 = fx; if (fy < y0) y0 = fy; if (fy > y1) y1 = fy; }
        polys.push({ c, xs, ys, x0, x1, y0, y1, area: (x1 - x0) * (y1 - y0) });
      }
    }
    polys.sort((a, b) => a.area - b.area); // enclaves and small states are tested before the large neighbours around them
    return { ...N, polys, byCode };
  }),
  climate: () => import('./atlas_climate.js').then((m) => {
    const C = m.CLIMATE, n = C.nlat * C.nlon, nh = (C.nlat / C.hstep) * (C.nlon / C.hstep), out = { ...C };
    for (const k of ['ghi', 'wind', 'temp']) out[k] = { ...C[k], A: unpack(C[k].ann).subarray(0, n), H: asI8(unpack(C[k].har), 0, 4 * nh) };
    return out;
  }),
};
const loaded = {};
/** Load and decode one data set of the atlas (memoised). Rejects when the module cannot be loaded. */
export function atlasData(name) {
  if (!loaded[name]) { loaded[name] = loaders[name](); loaded[name].catch(() => { delete loaded[name]; }); }
  return loaded[name];
}

// ------------------------------------------------------------------------------------------------ ocean
/** Valid cells of grid G around a point with interpolation weights, or (search) the nearest valid cell within about 330 km: { cells: [[k, w]], far km }. */
function oceanCells(G, lat, lon, valid, search = true) {
  const y = (lat - G.lat0) / G.step - 0.5, x = (lon - G.lon0) / G.step - 0.5, i0 = Math.floor(y), j0 = Math.floor(x), fy = y - i0, fx = x - j0;
  const ok = (i, j) => i >= 0 && i < G.nlat && valid(i * G.nlon + wrap(j, G.nlon));
  const cells = [];
  for (const [di, dj, w] of [[0, 0, (1 - fy) * (1 - fx)], [0, 1, (1 - fy) * fx], [1, 0, fy * (1 - fx)], [1, 1, fy * fx]]) if (ok(i0 + di, j0 + dj) && w > 1e-6) cells.push([(i0 + di) * G.nlon + wrap(j0 + dj, G.nlon), w]);
  if (cells.length) { const sum = cells.reduce((a, c) => a + c[1], 0); return { cells: cells.map(([k, w]) => [k, w / sum]), far: 0 }; }
  if (!search) return null;
  const ic = Math.round(y), jc = Math.round(x); // on land or in a gap: nearest sea cell, searched outwards
  let best = null;
  for (let r = 1; r <= 4; r++) {
    for (let di = -r; di <= r; di++) for (let dj = -r; dj <= r; dj++) {
      if (Math.max(Math.abs(di), Math.abs(dj)) !== r || !ok(ic + di, jc + dj)) continue;
      const d = distKm(lat, lon, G.lat0 + (ic + di + 0.5) * G.step, G.lon0 + (wrap(jc + dj, G.nlon) + 0.5) * G.step);
      if (!best || d < best.d) best = { d, k: (ic + di) * G.nlon + wrap(jc + dj, G.nlon) };
    }
    if (best && best.d < (r + 0.5) * 111 * G.step * Math.cos(lat * D2R)) break;
  }
  return best && best.d <= 330 ? { cells: [[best.k, 1]], far: Math.round(best.d) } : null;
}
/**
 * Sea-surface salinity and temperature climatology at a point (annual value and 12 months). Salinity (1° cells) and
 * temperature (0.5° cells near land, 1° elsewhere) come from different sources and are looked up independently; either
 * group of fields may be missing. Null when neither exists.
 */
export async function atlasOcean(lat, lon) {
  const O = await atlasData('ocean'), out = {};
  const series = (c, G, mean, H, hk) => { const n = G.nlat * G.nlon, v = new Array(12).fill(0); for (const [k, w] of c.cells) for (let m = 0; m < 12; m++) { const p = (2 * Math.PI * m) / 12; v[m] += w * (mean[k] / 100 + (H[k] * Math.cos(p) + H[n + k] * Math.sin(p) + H[2 * n + k] * Math.cos(2 * p) + H[3 * n + k] * Math.sin(2 * p)) / hk); } return v; };
  const cs = oceanCells(O, lat, lon, (k) => O.S0[k] !== 65535);
  if (cs) {
    const sM = series(cs, O, O.S0, O.SH, 20).map((v) => +Math.max(0, v).toFixed(2));
    Object.assign(out, { salinity: +cs.cells.reduce((a, [k, w]) => a + (w * O.S0[k]) / 100, 0).toFixed(2), salinityMin: Math.min(...sM), salinityMax: Math.max(...sM), salinityMonthly: sM, oceanCellKm: cs.far });
  }
  const F = O.fine, cf = oceanCells(F, lat, lon, (k) => F.T0[k] !== -32768, false), ct = cf || oceanCells(O, lat, lon, (k) => O.T0[k] !== -32768);
  if (ct) {
    const tM = (cf ? series(cf, F, F.T0, F.TH, 8) : series(ct, O, O.T0, O.TH, 8)).map((v) => +Math.max(-1.8, v).toFixed(2));
    Object.assign(out, { sstMonthly: tM, sst: tM[new Date().getUTCMonth()], sstMin: Math.min(...tM), sstMax: Math.max(...tM), sstCellKm: ct.far, sstPeriod: O.sstPeriod });
  }
  return cs || ct ? out : null;
}

// ------------------------------------------------------------------------------------------------ relief
const expand = (q, k) => (q < 0 ? -1 : 1) * (q / k) ** 2;
function tileOf(R, id) { // binary search in the sorted list of coastal cells
  let lo = 0, hi = R.ids.length - 1;
  while (lo <= hi) { const mid = (lo + hi) >> 1, v = R.ids[mid]; if (v === id) return mid; if (v < id) lo = mid + 1; else hi = mid - 1; }
  return -1;
}
function baseAt(R, lat, lon) { // bilinear in the 0.25° world grid
  const B = R.B, y = clamp((lat + 90) / B.step - 0.5, 0, B.nlat - 1), x = (lon + 180) / B.step - 0.5, i0 = Math.min(B.nlat - 2, Math.floor(y)), j0 = Math.floor(x), fy = y - i0, fx = x - j0;
  const v = (i, j) => expand(R.q[i * B.nlon + wrap(j, B.nlon)], R.kb);
  return (1 - fy) * ((1 - fx) * v(i0, j0) + fx * v(i0, j0 + 1)) + fy * ((1 - fx) * v(i0 + 1, j0) + fx * v(i0 + 1, j0 + 1));
}
/** Value of node (fi, fj) of the 0.05° lattice: a coastal tile where one exists, otherwise the world grid. */
function nodeAt(R, fi, fj) {
  fi = clamp(fi, 0, 3599); fj = wrap(fj, 7200);
  const t = tileOf(R, ((fi / 5) | 0) * 1440 + ((fj / 5) | 0));
  if (t >= 0) return expand(R.tq[t * 25 + (fi % 5) * 5 + (fj % 5)], R.kt);
  return baseAt(R, -90 + (fi + 0.5) * 0.05, -180 + (fj + 0.5) * 0.05);
}
function elevAt(R, lat, lon) {
  const y = (lat + 90) / 0.05 - 0.5, x = (lon + 180) / 0.05 - 0.5, i0 = Math.floor(y), j0 = Math.floor(x), fy = y - i0, fx = x - j0;
  return (1 - fy) * ((1 - fx) * nodeAt(R, i0, j0) + fx * nodeAt(R, i0, j0 + 1)) + fy * ((1 - fx) * nodeAt(R, i0 + 1, j0) + fx * nodeAt(R, i0 + 1, j0 + 1));
}
/** Elevation (m, negative below sea level) at a point from the built-in relief. */
export async function atlasElevation(lat, lon) { return elevAt(await atlasData('relief'), lat, lon); }

/**
 * Seabed and terrain around a point. Returns the same fields as the live relief connector, on a coarser grid:
 * bathy { lat[], lon[], elev[][], coarse }, depth, elevationRelief, maxDepthNearby, seaFraction, and for points in
 * the sea close to land a nearshore depth estimate (see below).
 */
export async function atlasRelief(lat, lon) {
  const R = await atlasData('relief'), half = 0.5, n = 41;
  const lats = Array.from({ length: n }, (_, i) => clamp(lat - half + (2 * half * i) / (n - 1), -89.9, 89.9)), lons = Array.from({ length: n }, (_, i) => lon - half + (2 * half * i) / (n - 1));
  const elev = lats.map((la) => lons.map((lo) => +elevAt(R, la, lo).toFixed(1)));
  const flat = elev.flat(), here = elevAt(R, lat, lon);
  const out = { bathy: { lat: lats, lon: lons.map((lo) => +(((lo + 540) % 360) - 180).toFixed(5)), elev, coarse: true, name: 'Built-in atlas relief (about 5 km at the coast, 25 km elsewhere)' },
    depth: here < 0 ? +(-here).toFixed(1) : 0, elevationRelief: +here.toFixed(1), maxDepthNearby: +Math.max(0, -Math.min(...flat)).toFixed(1), seaFraction: flat.filter((z) => z < 0).length / flat.length };
  // Nearshore: next to land the plain interpolation mixes the height of the land into the depth of the water. There the
  // depth is taken from the sea nodes only, tapered towards the shore by the share of land among the four nodes around
  // the site. (Checked against 2700 independent 15 arc-second soundings within 5 km of land: median error factor 1.25,
  // against 1.35 for plain interpolation and 3.5 for the 0.25° cell alone.)
  const y = (lat + 90) / 0.05 - 0.5, x = (lon + 180) / 0.05 - 0.5, i0 = Math.floor(y), j0 = Math.floor(x), fy = y - i0, fx = x - j0;
  let sw = 0, sv = 0;
  for (const [di, dj, w] of [[0, 0, (1 - fy) * (1 - fx)], [0, 1, (1 - fy) * fx], [1, 0, fy * (1 - fx)], [1, 1, fy * fx]]) { const z = nodeAt(R, i0 + di, j0 + dj); if (z < 0) { sw += w; sv += w * -z; } }
  if (sw > 0 && sw < 1 - 1e-9 && (here < 0 || sw >= 0.5)) {
    out.depth = +((sv / sw) * Math.min(1, 0.35 + 0.65 * sw)).toFixed(1);
    out.depthEstimated = 'nearshore estimate from the neighbouring sea cells of the built-in 5 km relief';
  } else if (here >= 0) out.elevation = +here.toFixed(1);
  return out;
}

// ------------------------------------------------------------------------------------------------ coast
/** Share of the straight line between two points that runs over land (0…1), from the world relief grid. */
function landShare(R, la1, lo1, la2, lo2) {
  let dlo = lo2 - lo1; if (dlo > 180) dlo -= 360; if (dlo < -180) dlo += 360;
  let n = 0;
  for (let k = 1; k <= 12; k++) if (baseAt(R, la1 + ((la2 - la1) * k) / 13, lo1 + (dlo * k) / 13) > 0) n++;
  return n / 12;
}
function nearestRows(rows, cols, count, lat, lon, maxKm, R) {
  const out = [], box = maxKm / 111, boxLon = box / Math.max(0.2, Math.cos(lat * D2R));
  for (let r = 0; r < count; r++) {
    const la = rows[r * cols] / 100, lo = rows[r * cols + 1] / 100;
    if (Math.abs(la - lat) > box) continue;
    let dlo = Math.abs(lo - lon); if (dlo > 180) dlo = 360 - dlo;
    if (dlo > boxLon) continue;
    const d = distKm(lat, lon, la, lo);
    if (d <= maxKm) out.push({ r, d, la, lo, eff: d });
  }
  // a table point on the other side of a peninsula or isthmus is a different sea: lines that cross land count as longer
  if (R) for (const p of out) p.eff = p.d * (1 + 3 * landShare(R, lat, lon, p.la, p.lo));
  return out.sort((a, b) => a.eff - b.eff);
}
/**
 * Tides, currents and waves of the nearest coastal table points. `tide` is a harmonic prediction for the same
 * window the live connector returns (three days back, five ahead), built from five fitted constituents.
 */
export async function atlasCoast(lat, lon) {
  const C = await atlasData('coast'), R = await atlasData('relief').catch(() => null), out = {};
  const s = nearestRows(C.seaRows, C.sea.cols, C.sea.rows, lat, lon, 220, R)[0];
  if (s) {
    const a = C.seaRows.subarray(s.r * C.sea.cols, (s.r + 1) * C.sea.cols);
    out.tideRange = a[2] / 100; out.tideSpring = a[3] / 100; out.tideNeap = a[4] / 100;
    if (a[16] >= 0) { out.currentSpeed = a[16] / 1000; out.currentMax = a[17] / 1000; }
    if (a[18] >= 0) out.currentDir = a[18];
    const now = Date.now(), H = 3600e3, start = Math.floor(now / H) * H - 72 * H, epoch = Date.parse(C.tideEpoch), nH = 192, eta = new Array(nH);
    for (let i = 0; i < nH; i++) {
      const th = (start + i * H - epoch) / H;
      let e = 0;
      for (let q = 0; q < 5; q++) e += (a[6 + 2 * q] / 1000) * Math.cos((C.tideSpeeds[q] * th - a[7 + 2 * q] / 10) * D2R);
      eta[i] = +e.toFixed(3);
    }
    out.tide = { t: Array.from({ length: nH }, (_, i) => i), eta, nowHour: (now - start) / H, synthetic: true, constituents: Object.fromEntries(C.tideNames.map((nm, q) => [nm, { amplitude: a[6 + 2 * q] / 1000, phase: a[7 + 2 * q] / 10 }])) };
    out.seaLevelMean = 0;
    out.coastPointKm = Math.round(s.d);
  }
  // waves: the table is coarser (one point per 2° coastal cell), so the nearest points are blended by inverse distance squared
  const near = nearestRows(C.waveRows, C.wave.cols, C.wave.rows, lat, lon, 330, R), ws = near.filter((p) => p.eff <= Math.max(120, 2.5 * near[0].eff)).slice(0, 3);
  if (ws.length) {
    const nc = C.wave.cols, wt = ws.map((p) => 1 / Math.max(10, p.eff) ** 2), sum = wt.reduce((x, y) => x + y, 0);
    const col = (c, need) => { let s = 0, w = 0; ws.forEach((p, i) => { const v = C.waveRows[p.r * nc + c]; if (!need || v >= 0) { s += wt[i] * v; w += wt[i]; } }); return w > 0 ? s / w / 100 : null; };
    out.waveHeight = +col(2).toFixed(2); out.waveHeightP95 = +col(3).toFixed(2); out.waveHeightMax = +col(4).toFixed(2);
    const tp = col(5, true), ts = col(6, true), a = C.waveRows.subarray(ws[0].r * nc, (ws[0].r + 1) * nc);
    if (tp !== null) out.wavePeriod = +tp.toFixed(2);
    if (ts !== null) out.wavePeriodStorm = +ts.toFixed(2);
    if (a[7] >= 0) out.waveDir = a[7];
    out.waveHeightMonthly = Array.from({ length: 12 }, (_, m) => +col(8 + m).toFixed(2));
    out.wavePointKm = Math.round(ws[0].d);
  }
  return Object.keys(out).length ? { ...out, seaPeriod: C.seaPeriod, wavePeriodOfRecord: C.wavePeriod } : null;
}

// ------------------------------------------------------------------------------------------------ nations
function inRing(p, x, y) {
  if (x < p.x0 || x > p.x1 || y < p.y0 || y > p.y1) return false;
  let inside = false;
  for (let i = 0, n = p.xs.length, j = n - 1; i < n; j = i++) if ((p.ys[i] > y) !== (p.ys[j] > y) && x < ((p.xs[j] - p.xs[i]) * (y - p.ys[i])) / (p.ys[j] - p.ys[i]) + p.xs[i]) inside = !inside;
  return inside;
}
function ringKm(p, x, y) { // distance from a point to the outline (local flat-earth approximation)
  const kx = 111.2 * Math.cos(y * D2R), ky = 111.2;
  let best = Infinity;
  for (let i = 0, n = p.xs.length; i < n - 1; i++) {
    const ax = (p.xs[i] - x) * kx, ay = (p.ys[i] - y) * ky, bx = (p.xs[i + 1] - x) * kx, by = (p.ys[i + 1] - y) * ky, dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
    const t = L > 0 ? clamp(-(ax * dx + ay * dy) / L, 0, 1) : 0, d = Math.hypot(ax + t * dx, ay + t * dy);
    if (d < best) best = d;
  }
  return best;
}
/** Country of a point: the outline that contains it, or the nearest one within 100 km for points offshore. */
export async function atlasPlace(lat, lon) {
  const N = await atlasData('nations');
  for (const p of N.polys) if (inRing(p, lon, lat)) return { country: p.c.name, countryCode: p.c.a2, iso3: p.c.a3, offshoreKm: 0 };
  const m = 100 / 111.2, mx = m / Math.max(0.15, Math.cos(lat * D2R));
  let best = null;
  for (const p of N.polys) {
    if (lon < p.x0 - mx || lon > p.x1 + mx || lat < p.y0 - m || lat > p.y1 + m) continue;
    const d = ringKm(p, lon, lat);
    if (d <= 100 && (!best || d < best.d)) best = { d, c: p.c };
  }
  return best ? { country: best.c.name, countryCode: best.c.a2, iso3: best.c.a3, offshoreKm: Math.round(best.d) } : null;
}
/** Latest bundled national figures of a country (ISO-2 code), in the field names of the live connectors. */
export async function atlasNational(code) {
  const N = await atlasData('nations'), c = N.byCode[String(code || '').toUpperCase()];
  if (!c) return null;
  const v = c.v || {}, d = { iso3: c.a3, countryName: c.name };
  for (const k of ['inflation', 'lendingRate', 'gdpPerCapita', 'waterStress', 'renewableElectricity', 'freshwaterPerCapita', 'safeWaterAccess', 'renewableShare']) if (v[k]) { d[k] = v[k][0]; d[k + 'Year'] = String(v[k][1]); }
  if (v.tariff) { d.electricityPriceWB = +(v.tariff[0] / 100).toFixed(4); d.electricityPriceWBYear = String(v.tariff[1]); }
  if (v.gridCarbon) { d.gridCarbon = +(v.gridCarbon[0] / 1000).toFixed(4); d.gridCarbonYear = String(v.gridCarbon[1]); }
  if (c.cur) { d.currency = c.cur; if (N.fx[c.cur] > 0) { d.fxPerUSD = N.fx[c.cur]; d.fxDate = N.fxDate; } }
  return d;
}
/** Bundled exchange rate of a currency: { fxPerUSD, fxDate } or null. */
export async function atlasRate(cur) { const N = await atlasData('nations'); return N.fx[cur] > 0 ? { fxPerUSD: N.fx[cur], fxDate: N.fxDate } : null; }
/** Country outlines for the base map: [{ code, xs, ys }] in degrees. */
export async function atlasBorders() { return (await atlasData('nations')).polys; }

// ------------------------------------------------------------------------------------------------ climate
/** Long-term solar irradiation (kWh/m²·d), wind speed at 10 m (m/s) and air temperature at 2 m (°C): annual mean and monthly cycle. */
export async function atlasClimate(lat, lon) {
  const C = await atlasData('climate'), y = (lat - C.lat0) / C.step - 0.5, x = (lon - C.lon0) / C.step - 0.5;
  if (y < -0.5 || y > C.nlat - 0.5) return null;
  const i0 = clamp(Math.floor(y), 0, C.nlat - 2), j0 = Math.floor(x), fy = clamp(y - i0, 0, 1), fx = x - j0, hn = (C.nlat / C.hstep) * (C.nlon / C.hstep);
  const hi = clamp(Math.floor((lat - C.lat0) / C.hstep), 0, C.nlat / C.hstep - 1), hj = wrap(Math.floor((lon - C.lon0) / C.hstep), C.nlon / C.hstep), hk = hi * (C.nlon / C.hstep) + hj;
  const one = (G, lo) => {
    let s = 0, w = 0;
    for (const [di, dj, ww] of [[0, 0, (1 - fy) * (1 - fx)], [0, 1, (1 - fy) * fx], [1, 0, fy * (1 - fx)], [1, 1, fy * fx]]) { const q = G.A[(i0 + di) * C.nlon + wrap(j0 + dj, C.nlon)]; if (q !== 255) { s += ww * (q / G.k - G.off); w += ww; } }
    if (w < 1e-6) return null;
    const ann = s / w, mon = Array.from({ length: 12 }, (_, m) => { const p = (2 * Math.PI * m) / 12; return +Math.max(lo, ann + (G.H[hk] * Math.cos(p) + G.H[hn + hk] * Math.sin(p) + G.H[2 * hn + hk] * Math.cos(2 * p) + G.H[3 * hn + hk] * Math.sin(2 * p)) / G.hk).toFixed(2); });
    return { ann: +ann.toFixed(2), mon };
  };
  const g = one(C.ghi, 0), wv = one(C.wind, 0), t = one(C.temp, -90), out = {};
  if (g) { out.ghiAnnual = g.ann; out.ghiMonthly = g.mon; }
  if (wv) { out.windAnnual = wv.ann; out.windMonthly = wv.mon; }
  if (t) { out.airTempAnnual = t.ann; out.airTempMonthly = t.mon; }
  return Object.keys(out).length ? { ...out, climatePeriod: C.period } : null;
}

/** Vintage (month of download) and period of every data set that has been loaded so far. */
export async function atlasVintage(names) {
  const out = {};
  for (const n of names) { try { const D = await atlasData(n); out[n] = D.vintage; if (n === 'nations') out.fx = D.fxDate; if (n === 'coast') { out.seaPeriod = D.seaPeriod; out.wavePeriod = D.wavePeriod; } if (n === 'climate') out.climatePeriod = D.period; } catch { /* data set unavailable */ } }
  return out;
}
