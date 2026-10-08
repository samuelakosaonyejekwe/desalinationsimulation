// Dependency-free canvas plotting: line/scatter curves, bar charts, 2-D fields
// (filled contours, iso-lines, streamlines, vectors) and shaded 3-D surfaces. Every chart supports PNG/CSV export;
// a spec may carry several `frames` (e.g. depth slices), which the chart card switches between with a slider.
import { fmt } from './num.js';

const PALETTE = ['#0ea5e9', '#f97316', '#10b981', '#a855f7', '#ef4444', '#eab308', '#14b8a6', '#ec4899', '#6366f1', '#84cc16', '#f43f5e', '#06b6d4'];
const CMAPS = {
  viridis: ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'],
  turbo: ['#30123b', '#4662d7', '#36aaf9', '#1ae4b6', '#72fe5e', '#c7ef34', '#faba39', '#f66b19', '#ca2a04', '#7a0403'],
  coolwarm: ['#3b4cc0', '#6f92f3', '#aac7fd', '#dddcdc', '#f7b89c', '#e7745b', '#b40426'],
  salinity: ['#f7fcf0', '#ccebc5', '#7bccc4', '#43a2ca', '#0868ac', '#084081', '#3f007d', '#7a0177'],
  // below the midpoint: sea (deep → shallow); above it: land (lowland → mountain)
  topo: ['#08306b', '#0a4a90', '#1565a8', '#2a7fbf', '#4a9bd0', '#72b7dc', '#9bd0e6', '#bfe6ee', '#a8d08d', '#c9dd9a', '#e9e3a0', '#d9bf77', '#b98f55', '#96673f', '#7a5a4a', '#5a463f'],
  land: ['#a8d08d', '#c9dd9a', '#e9e3a0', '#d9bf77', '#b98f55', '#96673f', '#7a5a4a', '#5a463f'],
  sea: ['#08306b', '#0a4a90', '#1565a8', '#2a7fbf', '#4a9bd0', '#72b7dc', '#9bd0e6', '#bfe6ee'],
  thermal: ['#042333', '#2c3395', '#744992', '#b15f82', '#eb7958', '#fbb43d', '#e8fa5b'],
  heat: ['#fff3b0', '#fdd66b', '#fba23c', '#f0652a', '#d0301f', '#8f1414'],
};
const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const LUT = {};
function lut(name) {
  if (LUT[name]) return LUT[name];
  const stops = (CMAPS[name] || CMAPS.viridis).map(hex), out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const p = (i / 255) * (stops.length - 1), a = Math.floor(p), b = Math.min(stops.length - 1, a + 1), t = p - a;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.round(stops[a][c] * (1 - t) + stops[b][c] * t);
  }
  return (LUT[name] = out);
}

function niceTicks(lo, hi, n = 6) {
  if (!(hi > lo)) { const d = Math.abs(lo) * 0.1 || 1; lo -= d; hi += d; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const t = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) t.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return t;
}
function logTicks(lo, hi) {
  const t = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) t.push(10 ** e);
  return t.filter((v) => v >= lo * 0.999 && v <= hi * 1.001);
}
const tickLabel = (v) => {
  const a = Math.abs(v);
  if (v === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace('e+', 'e');
  return String(+v.toPrecision(6));
};
const cssVar = (el, name, fb) => (getComputedStyle(el).getPropertyValue(name) || fb).trim() || fb;

function setup(canvas, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5), w = Math.max(260, canvas.clientWidth || canvas.parentElement.clientWidth || 600);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(height * dpr);
  canvas.style.height = height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, height);
  return {
    ctx, w, h: height,
    fg: cssVar(canvas, '--text', '#0f172a'), mute: cssVar(canvas, '--muted', '#64748b'), grid: cssVar(canvas, '--line', '#e2e8f0'), bg: cssVar(canvas, '--card', '#ffffff'),
  };
}

function extent(arrs, log) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (const v of a) if (Number.isFinite(v) && (!log || v > 0)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === Infinity) { lo = log ? 1 : 0; hi = log ? 10 : 1; }
  return [lo, hi];
}

function drawLine(canvas, spec) {
  const C = setup(canvas, spec.height || 320), { ctx, w, h } = C;
  const series = (spec.series || []).filter((s) => s && s.x && s.y);
  const m = { l: 62, r: 16, t: 14, b: 44 };
  let [x0, x1] = extent(series.map((s) => s.x), spec.logx), [y0, y1] = extent(series.map((s) => s.y).concat([(spec.hlines || []).map((q) => q.y)]), spec.logy);
  if (!spec.logy) { const pad = (y1 - y0) * 0.06 || Math.abs(y1) * 0.1 || 1; y0 -= pad; y1 += pad + (series.some((q) => q.name) ? (y1 - y0) * 0.12 : 0); if (spec.zeroY && y0 > 0) y0 = 0; }
  if (spec.xmin !== undefined) x0 = spec.xmin; if (spec.xmax !== undefined) x1 = spec.xmax;
  if (spec.ymin !== undefined) y0 = spec.ymin; if (spec.ymax !== undefined) y1 = spec.ymax;
  if (x1 === x0) { x0 -= 1; x1 += 1; }
  const tx = spec.logx ? Math.log10 : (v) => v, ty = spec.logy ? Math.log10 : (v) => v;
  const X = (v) => m.l + ((tx(v) - tx(x0)) / (tx(x1) - tx(x0))) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((ty(v) - ty(y0)) / (ty(y1) - ty(y0))) * (h - m.t - m.b);
  ctx.font = '11px system-ui, sans-serif'; ctx.lineWidth = 1;
  const xt = spec.logx ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(3, Math.floor(w / 90))), yt = spec.logy ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
  ctx.strokeStyle = C.grid; ctx.fillStyle = C.mute;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of xt) { const px = X(v); if (px < m.l - 1 || px > w - m.r + 1) continue; ctx.beginPath(); ctx.moveTo(px, m.t); ctx.lineTo(px, h - m.b); ctx.stroke(); ctx.fillText(tickLabel(v), px, h - m.b + 5); }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yt) { const py = Y(v); if (py < m.t - 1 || py > h - m.b + 1) continue; ctx.beginPath(); ctx.moveTo(m.l, py); ctx.lineTo(w - m.r, py); ctx.stroke(); ctx.fillText(tickLabel(v), m.l - 6, py); }
  ctx.strokeStyle = C.mute; ctx.strokeRect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
  ctx.fillStyle = C.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  if (spec.xlabel) ctx.fillText(spec.xlabel, m.l + (w - m.l - m.r) / 2, h - 4);
  if (spec.ylabel) { ctx.save(); ctx.translate(13, m.t + (h - m.t - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(spec.ylabel, 0, 0); ctx.restore(); }
  ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b); ctx.clip();
  for (const q of spec.hlines || []) {
    ctx.strokeStyle = q.color || '#ef4444'; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(m.l, Y(q.y)); ctx.lineTo(w - m.r, Y(q.y)); ctx.stroke(); ctx.setLineDash([]);
    if (q.label) { const under = Y(q.y) < m.t + 46; ctx.fillStyle = q.color || '#ef4444'; ctx.textAlign = 'right'; ctx.textBaseline = under ? 'top' : 'bottom'; ctx.fillText(q.label, w - m.r - 4, Y(q.y) + (under ? 3 : -2)); } // below the line when the line runs through the legend rows at the top
  }
  for (const q of spec.vlines || []) {
    ctx.strokeStyle = q.color || '#ef4444'; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(X(q.x), m.t); ctx.lineTo(X(q.x), h - m.b); ctx.stroke(); ctx.setLineDash([]);
  }
  series.forEach((s, i) => {
    const col = s.color || PALETTE[i % PALETTE.length], mode = s.mode || 'line';
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = s.width || 2; ctx.lineJoin = 'round';
    if (mode !== 'points') {
      ctx.setLineDash(s.dash ? [6, 4] : []);
      ctx.beginPath();
      let pen = false, py0 = 0;
      for (let k = 0; k < s.x.length; k++) {
        const vx = s.x[k], vy = s.y[k];
        if (!Number.isFinite(vx) || !Number.isFinite(vy) || (spec.logy && vy <= 0) || (spec.logx && vx <= 0)) { pen = false; continue; }
        const px = X(vx), py = Y(vy);
        if (!pen) { ctx.moveTo(px, py); pen = true; } else { if (mode === 'step') ctx.lineTo(px, py0); ctx.lineTo(px, py); }
        py0 = py;
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (mode === 'points' || mode === 'both') {
      for (let k = 0; k < s.x.length; k++) {
        if (!Number.isFinite(s.x[k]) || !Number.isFinite(s.y[k])) continue;
        ctx.beginPath(); ctx.arc(X(s.x[k]), Y(s.y[k]), s.size || 3, 0, 6.2832); ctx.fill();
      }
    }
  });
  for (const q of spec.vlines || []) { // labels last, written upwards along the line from the axis on a backing, clear of the legend and legible over the curves
    if (!q.label) continue;
    ctx.save(); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.translate(X(q.x) + 3, h - m.b - 5); ctx.rotate(-Math.PI / 2);
    ctx.globalAlpha = 0.8; ctx.fillStyle = C.bg || '#fff'; ctx.fillRect(-2, -1, ctx.measureText(q.label).width + 4, 14); ctx.globalAlpha = 1;
    ctx.fillStyle = q.color || '#ef4444'; ctx.fillText(q.label, 0, 0); ctx.restore();
  }
  ctx.restore();
  legend(C, series.map((s, i) => ({ name: s.name, color: s.color || PALETTE[i % PALETTE.length] })), m);
  canvas._hover = (px, py) => {
    if (px < m.l || px > w - m.r) return null;
    let best = null;
    series.forEach((s, i) => {
      for (let k = 0; k < s.x.length; k++) {
        if (!Number.isFinite(s.x[k]) || !Number.isFinite(s.y[k])) continue;
        const d = (X(s.x[k]) - px) ** 2 + (Y(s.y[k]) - py) ** 2;
        if (!best || d < best.d) best = { d, s, k, i };
      }
    });
    if (!best || best.d > 60 * 60) return null;
    return { x: X(best.s.x[best.k]), y: Y(best.s.y[best.k]), color: best.s.color || PALETTE[best.i % PALETTE.length], text: `${best.s.name ? best.s.name + ' · ' : ''}${fmt(best.s.x[best.k])} , ${fmt(best.s.y[best.k])}` };
  };
}

function legend(C, items, m) {
  const { ctx, w } = C;
  items = items.filter((i) => i.name);
  if (items.length < 1 || (items.length === 1 && !items[0].name)) return;
  ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  let x = m.l + 8, y = m.t + 10;
  for (const it of items) {
    const tw = ctx.measureText(it.name).width + 24;
    if (x + tw > w - m.r) { x = m.l + 8; y += 15; }
    ctx.globalAlpha = 0.78; ctx.fillStyle = C.bg; ctx.fillRect(x - 3, y - 7, tw, 14); ctx.globalAlpha = 1;
    ctx.fillStyle = it.color; ctx.fillRect(x, y - 2, 12, 4);
    ctx.fillStyle = C.fg; ctx.fillText(it.name, x + 16, y);
    x += tw + 6;
  }
}

function drawBar(canvas, spec) {
  const C = setup(canvas, spec.height || 320), { ctx, w, h } = C;
  const cats = spec.categories || [], series = spec.series || [];
  const longest = Math.max(0, ...cats.map((c) => String(c).length)), rot = cats.length * longest * 6.5 > w - 80;
  const m = { l: 62, r: 16, t: 14, b: rot ? Math.min(120, 22 + longest * 4.6) : 44 };
  let lo = 0, hi = 0;
  cats.forEach((_, i) => {
    if (spec.stacked) { let p = 0, n = 0; for (const s of series) { const v = s.values[i] || 0; if (v > 0) p += v; else n += v; } hi = Math.max(hi, p); lo = Math.min(lo, n); }
    else for (const s of series) { hi = Math.max(hi, s.values[i] || 0); lo = Math.min(lo, s.values[i] || 0); }
  });
  hi += (hi - lo) * 0.08 || 1;
  if (lo < 0) lo -= (hi - lo) * 0.05;
  const Y = (v) => h - m.b - ((v - lo) / (hi - lo)) * (h - m.t - m.b), bw = (w - m.l - m.r) / Math.max(1, cats.length);
  ctx.font = '11px system-ui, sans-serif'; ctx.strokeStyle = C.grid; ctx.fillStyle = C.mute; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of niceTicks(lo, hi, 5)) { ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(w - m.r, Y(v)); ctx.stroke(); ctx.fillText(tickLabel(v), m.l - 6, Y(v)); }
  const rects = [];
  cats.forEach((c, i) => {
    const x0 = m.l + i * bw + bw * 0.14, bwid = bw * 0.72;
    let p = 0, n = 0;
    series.forEach((s, j) => {
      const v = s.values[i] || 0, col = s.color || PALETTE[j % PALETTE.length];
      ctx.fillStyle = (spec.colors && series.length === 1 && spec.colors[i]) || col;
      let rx, rw, y0, y1;
      if (spec.stacked) { rx = x0; rw = bwid; if (v >= 0) { y0 = Y(p); p += v; y1 = Y(p); } else { y0 = Y(n); n += v; y1 = Y(n); } }
      else { rw = bwid / series.length; rx = x0 + j * rw; y0 = Y(0); y1 = Y(v); }
      ctx.fillRect(rx, Math.min(y0, y1), Math.max(1, rw - 1), Math.abs(y1 - y0));
      rects.push({ x: rx, y: Math.min(y0, y1), w: rw, h: Math.abs(y1 - y0), text: `${c}${s.name ? ' · ' + s.name : ''}: ${fmt(v)}` });
    });
    ctx.fillStyle = C.mute;
    if (rot) { ctx.save(); ctx.translate(x0 + bwid / 2, h - m.b + 6); ctx.rotate(-Math.PI / 3.2); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(c), 0, 0); ctx.restore(); }
    else { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(c), x0 + bwid / 2, h - m.b + 6); }
  });
  ctx.strokeStyle = C.mute; ctx.beginPath(); ctx.moveTo(m.l, Y(0)); ctx.lineTo(w - m.r, Y(0)); ctx.stroke();
  ctx.fillStyle = C.fg;
  if (spec.ylabel) { ctx.save(); ctx.translate(13, m.t + (h - m.t - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(spec.ylabel, 0, 0); ctx.restore(); }
  if (series.length > 1) legend(C, series.map((s, j) => ({ name: s.name, color: s.color || PALETTE[j % PALETTE.length] })), m);
  canvas._hover = (px, py) => { const r = rects.find((q) => px >= q.x && px <= q.x + q.w && py >= q.y - 4 && py <= q.y + q.h + 4); return r ? { x: r.x + r.w / 2, y: r.y, text: r.text, color: C.fg } : null; };
}

function drawField(canvas, spec) {
  const xs = spec.x, ys = spec.y, z = spec.z, nx = xs.length, ny = ys.length;
  const cw = Math.max(260, canvas.clientWidth || canvas.parentElement.clientWidth || 600);
  const m = { l: 58, r: 74, t: 14, b: 44 };
  let height = spec.height || 340;
  const dx = xs[nx - 1] - xs[0] || 1, dy = ys[ny - 1] - ys[0] || 1;
  if (spec.equal) height = Math.round(Math.min(560, Math.max(170, ((cw - m.l - m.r) * dy) / dx + m.t + m.b)));
  const C = setup(canvas, height), { ctx, w, h } = C;
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  let lo = spec.zmin, hi = spec.zmax;
  if (lo === undefined || hi === undefined) {
    let a = Infinity, b = -Infinity;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const v = z[j][i]; if (Number.isFinite(v) && !(spec.mask && spec.mask[j][i])) { if (v < a) a = v; if (v > b) b = v; } }
    if (lo === undefined) lo = a; if (hi === undefined) hi = b;
  }
  if (!(hi > lo)) hi = lo + (Math.abs(lo) * 1e-6 || 1e-9);
  const L = lut(spec.cmap || 'viridis');
  // optional two-slope scale: values below zmid use the lower half of the colour map, values above it the upper half
  const mid = spec.zmid !== undefined && spec.zmid > lo && spec.zmid < hi ? spec.zmid : null;
  // the 'topo' map has separate sea and land halves: values never blend across the shoreline
  const sA = spec.cmap === 'topo' ? 7 / 15 : 0.5, sB = spec.cmap === 'topo' ? 8 / 15 : 0.5;
  const norm = (v) => (mid === null ? (v - lo) / (hi - lo) : v < mid ? (sA * (v - lo)) / (mid - lo) : sB + ((1 - sB) * (v - mid)) / (hi - mid));
  const inv = (t) => (mid === null ? lo + t * (hi - lo) : t < 0.5 ? lo + (t / 0.5) * (mid - lo) : mid + ((t - 0.5) / 0.5) * (hi - mid));
  // Smooth fields are resampled to screen resolution by bilinear interpolation of the VALUES (then coloured),
  // which keeps colour boundaries such as a shoreline sharp; masked fields stay cell-exact.
  const smooth = !spec.mask && nx > 2 && ny > 2 && spec.interpolate !== false;
  const dprF = Math.min(window.devicePixelRatio || 1, 2), cap = spec.shade ? 520 : 640, W = smooth ? Math.max(nx, Math.min(cap, Math.round(pw * dprF))) : nx, H = smooth ? Math.max(ny, Math.min(cap, Math.round(ph * dprF))) : ny;
  let off = spec._raster && spec._raster.W === W && spec._raster.H === H && spec._raster.lo === lo && spec._raster.hi === hi ? spec._raster.canvas : null;
  const reuse = !!off;
  if (!off) { off = document.createElement('canvas'); off.width = W; off.height = H; }
  const octx = off.getContext('2d'), img = reuse ? null : octx.createImageData(W, H);
  const cellx = Math.abs(dx) / (nx - 1 || 1), celly = Math.abs(dy) / (ny - 1 || 1), geo = spec.shade === 'geo', kx = geo ? 111320 * Math.cos((((ys[0] + ys[ny - 1]) / 2) * Math.PI) / 180) : 1, ky = geo ? 110540 : 1;
  const zAt = (i, j) => z[Math.max(0, Math.min(ny - 1, j))][Math.max(0, Math.min(nx - 1, i))];
  // slope at every grid node by central differences; interpolating these (not the per-cell slope) gives smooth, natural shading
  let GX = null, GY = null;
  if (smooth && spec.shade) {
    GX = z.map((row, j) => row.map((_, i) => (zAt(i + 1, j) - zAt(i - 1, j)) / ((i === 0 || i === nx - 1 ? 1 : 2) * cellx * kx)));
    GY = z.map((row, j) => row.map((_, i) => (zAt(i, j + 1) - zAt(i, j - 1)) / ((j === 0 || j === ny - 1 ? 1 : 2) * celly * ky)));
  }
  for (let r = 0; r < (reuse ? 0 : H); r++) {
    const fj = smooth ? ((H - 1 - r) / (H - 1)) * (ny - 1) : ny - 1 - r, j0 = Math.min(ny - 2 < 0 ? 0 : ny - 2, Math.floor(fj)), b = smooth ? fj - j0 : 0;
    for (let cI = 0; cI < W; cI++) {
      const o = (r * W + cI) * 4;
      let v, shade = 1;
      if (smooth) {
        const fi = (cI / (W - 1)) * (nx - 1), i0 = Math.min(nx - 2, Math.floor(fi)), a = fi - i0;
        const v00 = z[j0][i0], v10 = z[j0][i0 + 1], v01 = z[j0 + 1][i0], v11 = z[j0 + 1][i0 + 1];
        v = Number.isFinite(v00 + v10 + v01 + v11) ? v00 * (1 - a) * (1 - b) + v10 * a * (1 - b) + v01 * (1 - a) * b + v11 * a * b : z[Math.round(fj)][Math.round(fi)];
        if (spec.shade && Number.isFinite(v)) { // lit from the north-west using smoothly interpolated node slopes
          const gx = GX[j0][i0] * (1 - a) * (1 - b) + GX[j0][i0 + 1] * a * (1 - b) + GX[j0 + 1][i0] * (1 - a) * b + GX[j0 + 1][i0 + 1] * a * b;
          const gy = GY[j0][i0] * (1 - a) * (1 - b) + GY[j0][i0 + 1] * a * (1 - b) + GY[j0 + 1][i0] * (1 - a) * b + GY[j0 + 1][i0 + 1] * a * b, ex = spec.exaggeration || 1;
          shade = Math.max(0.62, Math.min(1.22, 1 + 0.75 * ((-gx + gy) * ex) / Math.sqrt(1 + (gx * ex) ** 2 + (gy * ex) ** 2)));
        }
      } else {
        if (spec.mask && spec.mask[fj][cI]) { img.data[o] = 100; img.data[o + 1] = 116; img.data[o + 2] = 139; img.data[o + 3] = 255; continue; }
        v = zAt(cI, fj);
      }
      if (!Number.isFinite(v)) { img.data[o + 3] = 0; continue; }
      const t = Math.max(0, Math.min(255, Math.round(norm(v) * 255)));
      img.data[o] = Math.min(255, L[t * 3] * shade); img.data[o + 1] = Math.min(255, L[t * 3 + 1] * shade); img.data[o + 2] = Math.min(255, L[t * 3 + 2] * shade); img.data[o + 3] = 255;
    }
  }
  if (!reuse) { octx.putImageData(img, 0, 0); Object.defineProperty(spec, '_raster', { value: { canvas: off, W, H, lo, hi }, enumerable: false, configurable: true }); }
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(off, m.l, m.t, pw, ph);
  const X = (v) => m.l + ((v - xs[0]) / dx) * pw, Y = (v) => m.t + ph - ((v - ys[0]) / dy) * ph;
  const gi = (v) => ((v - xs[0]) / dx) * (nx - 1), gj = (v) => ((v - ys[0]) / dy) * (ny - 1);
  ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
  // iso-lines by marching squares
  if (spec.contours) {
    ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 0.8;
    const levels = Array.isArray(spec.contours) ? spec.contours : Array.from({ length: spec.contours }, (_, k) => lo + ((k + 1) * (hi - lo)) / (spec.contours + 1));
    const px = (i) => m.l + (i / (nx - 1)) * pw, py = (j) => m.t + ph - (j / (ny - 1)) * ph;
    for (const lvRaw of levels) {
      const lv = typeof lvRaw === 'object' ? lvRaw.level : lvRaw;
      ctx.strokeStyle = (typeof lvRaw === 'object' && lvRaw.color) || 'rgba(255,255,255,.55)'; ctx.lineWidth = (typeof lvRaw === 'object' && lvRaw.width) || 0.8;
      ctx.beginPath();
      for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
        const a = z[j][i], b = z[j][i + 1], c = z[j + 1][i + 1], d = z[j + 1][i];
        if (![a, b, c, d].every(Number.isFinite)) continue;
        const pts = [];
        const edge = (v1, v2, x1, y1, x2, y2) => { if ((v1 < lv) !== (v2 < lv)) { const t = (lv - v1) / (v2 - v1); pts.push([x1 + t * (x2 - x1), y1 + t * (y2 - y1)]); } };
        edge(a, b, i, j, i + 1, j); edge(b, c, i + 1, j, i + 1, j + 1); edge(c, d, i + 1, j + 1, i, j + 1); edge(d, a, i, j + 1, i, j);
        for (let k = 0; k + 1 < pts.length; k += 2) { ctx.moveTo(px(pts[k][0]), py(pts[k][1])); ctx.lineTo(px(pts[k + 1][0]), py(pts[k + 1][1])); }
      }
      ctx.stroke();
    }
  }
  const sample = (f, i, j) => {
    const i0 = Math.max(0, Math.min(nx - 2, Math.floor(i))), j0 = Math.max(0, Math.min(ny - 2, Math.floor(j))), a = i - i0, b = j - j0;
    return f[j0][i0] * (1 - a) * (1 - b) + f[j0][i0 + 1] * a * (1 - b) + f[j0 + 1][i0] * (1 - a) * b + f[j0 + 1][i0 + 1] * a * b;
  };
  if (spec.u && spec.v && (spec.stream || spec.vectors)) {
    const { u, v } = spec;
    let vmax = 1e-30;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) vmax = Math.max(vmax, Math.hypot(u[j][i], v[j][i]) || 0);
    const cellx = dx / (nx - 1), celly = dy / (ny - 1);
    if (spec.stream) {
      ctx.strokeStyle = 'rgba(255,255,255,.8)'; ctx.lineWidth = 0.9;
      const sx = Math.max(6, Math.round(pw / 46)), sy = Math.max(5, Math.round(ph / 26));
      for (let a = 0; a < sx; a++) for (let b = 0; b < sy; b++) {
        let i = ((a + 0.5) / sx) * (nx - 1), j = ((b + 0.5) / sy) * (ny - 1);
        ctx.beginPath(); ctx.moveTo(m.l + (i / (nx - 1)) * pw, m.t + ph - (j / (ny - 1)) * ph);
        for (let s = 0; s < 60; s++) {
          const uu = sample(u, i, j) / cellx, vv = sample(v, i, j) / celly, sp = Math.hypot(uu, vv);
          if (!(sp > 1e-12)) break;
          const ds = 0.6 / sp;
          const i2 = i + uu * ds * 0.5, j2 = j + vv * ds * 0.5;
          if (i2 < 0 || i2 > nx - 1 || j2 < 0 || j2 > ny - 1) break;
          const u2 = sample(u, i2, j2) / cellx, v2 = sample(v, i2, j2) / celly;
          i += u2 * ds; j += v2 * ds;
          if (i < 0 || i > nx - 1 || j < 0 || j > ny - 1) break;
          if (spec.mask && spec.mask[Math.round(j)][Math.round(i)]) break;
          ctx.lineTo(m.l + (i / (nx - 1)) * pw, m.t + ph - (j / (ny - 1)) * ph);
        }
        ctx.stroke();
      }
    }
    if (spec.vectors) {
      ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 1;
      const sx = Math.max(6, Math.round(pw / 34)), sy = Math.max(4, Math.round(ph / 30)), len = Math.min(pw / sx, ph / sy) * 0.85;
      for (let a = 0; a < sx; a++) for (let b = 0; b < sy; b++) {
        const i = ((a + 0.5) / sx) * (nx - 1), j = ((b + 0.5) / sy) * (ny - 1);
        if (spec.mask && spec.mask[Math.round(j)][Math.round(i)]) continue;
        const uu = sample(u, i, j), vv = sample(v, i, j), sp = Math.hypot(uu, vv);
        if (!(sp > vmax * 1e-3)) continue;
        const px0 = m.l + (i / (nx - 1)) * pw, py0 = m.t + ph - (j / (ny - 1)) * ph, k = (len * Math.sqrt(sp / vmax)) / sp;
        const ex = px0 + uu * k * (spec.equal ? 1 : 1), ey = py0 - vv * k, ang = Math.atan2(ey - py0, ex - px0);
        ctx.beginPath(); ctx.moveTo(px0, py0); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - 5 * Math.cos(ang - 0.45), ey - 5 * Math.sin(ang - 0.45)); ctx.lineTo(ex - 5 * Math.cos(ang + 0.45), ey - 5 * Math.sin(ang + 0.45)); ctx.fill();
      }
    }
  }
  for (const s of spec.shapes || []) {
    ctx.strokeStyle = s.color || '#ffffff'; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash ? [5, 4] : []);
    ctx.beginPath(); s.x.forEach((vx, k) => (k ? ctx.lineTo(X(vx), Y(s.y[k])) : ctx.moveTo(X(vx), Y(s.y[k]))));
    if (s.closed) ctx.closePath();
    if (s.fill) { ctx.fillStyle = s.fill; ctx.fill(); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.font = '11px system-ui, sans-serif';
  for (const p of spec.markers || []) {
    ctx.fillStyle = p.color || '#ffffff'; ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4.5, 0, 6.2832); ctx.fill(); ctx.stroke();
    if (p.label) { ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.strokeStyle = 'rgba(15,23,42,.8)'; ctx.lineWidth = 3; ctx.strokeText(p.label, X(p.x) + 7, Y(p.y) - 3); ctx.fillStyle = '#fff'; ctx.fillText(p.label, X(p.x) + 7, Y(p.y) - 3); }
  }
  ctx.restore();
  ctx.strokeStyle = C.mute; ctx.lineWidth = 1; ctx.strokeRect(m.l, m.t, pw, ph);
  ctx.fillStyle = C.mute; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of niceTicks(xs[0], xs[nx - 1], Math.max(4, Math.floor(pw / 70)))) if (v >= Math.min(xs[0], xs[nx - 1]) && v <= Math.max(xs[0], xs[nx - 1])) ctx.fillText(tickLabel(v), X(v), h - m.b + 5);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of niceTicks(ys[0], ys[ny - 1], Math.max(2, Math.floor(ph / 45)))) if (v >= Math.min(ys[0], ys[ny - 1]) && v <= Math.max(ys[0], ys[ny - 1])) ctx.fillText(tickLabel(v), m.l - 6, Y(v));
  ctx.fillStyle = C.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  if (spec.xlabel) ctx.fillText(spec.xlabel, m.l + pw / 2, h - 4);
  if (spec.ylabel) { ctx.save(); ctx.translate(13, m.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(spec.ylabel, 0, 0); ctx.restore(); }
  // colour bar
  const bx = w - m.r + 12, bwid = 12;
  for (let k = 0; k < ph; k++) { const t = Math.round((1 - k / ph) * 255); ctx.fillStyle = `rgb(${L[t * 3]},${L[t * 3 + 1]},${L[t * 3 + 2]})`; ctx.fillRect(bx, m.t + k, bwid, 1.5); }
  ctx.strokeStyle = C.mute; ctx.strokeRect(bx, m.t, bwid, ph);
  ctx.fillStyle = C.mute; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  const barTicks = mid === null ? niceTicks(lo, hi, 5) : [...niceTicks(lo, mid, 3).filter((v) => v < mid - (mid - lo) * 0.12), mid, ...niceTicks(mid, hi, 3).filter((v) => v > mid + (hi - mid) * 0.12)];
  for (const v of barTicks) if (v >= lo && v <= hi) ctx.fillText(tickLabel(v), bx + bwid + 4, m.t + ph - norm(v) * ph);
  void inv;
  if (spec.zlabel) { ctx.save(); ctx.translate(w - 5, m.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = C.fg; ctx.fillText(spec.zlabel, 0, 0); ctx.restore(); }
  canvas._pick = (px, py) => (px < m.l || px > m.l + pw || py < m.t || py > m.t + ph ? null : { x: xs[0] + ((px - m.l) / pw) * dx, y: ys[0] + ((m.t + ph - py) / ph) * dy });
  canvas._hover = (px, py) => {
    if (px < m.l || px > m.l + pw || py < m.t || py > m.t + ph) return null;
    const vx = xs[0] + ((px - m.l) / pw) * dx, vy = ys[0] + ((m.t + ph - py) / ph) * dy;
    const i = Math.round(gi(vx)), j = Math.round(gj(vy));
    if (spec.mask && spec.mask[j]?.[i]) return { x: px, y: py, text: 'solid', color: C.fg };
    return { x: px, y: py, color: C.fg, text: `x ${fmt(vx, 3)}, y ${fmt(vy, 3)} → ${fmt(z[j]?.[i])}${spec.zunit ? ' ' + spec.zunit : ''}` };
  };
}

/**
 * Shaded three-dimensional view of a ground surface with optional bodies on it (painter's algorithm on the 2-D canvas).
 * spec = { type: 'surface3d', x: [nx], y: [ny], z: [ny][nx] ground elevation at the grid nodes (NaN = hole), cmap, zmin, zmax, zmid,
 *   layers: [{ name, z: [ny][nx] upper surface, base: [ny][nx] (default: the ground), c: [ny][nx] colour value + cmap, cmin, cmax,
 *              clabel, cunit — or a flat `color` —, opacity, minThickness }]  — drawn where the surface stands above its base,
 *   planes: [{ z, name, color }] horizontal reference levels (outline), markers: [{ x, y, z0, z1, label, color }] vertical poles,
 *   azimuth (compass bearing of the viewpoint, °), elevation (° above the horizon), zscale (vertical exaggeration; automatic if absent),
 *   xlabel, ylabel, zlabel }. The card rotates the view by dragging (azimuth; with a mouse also the elevation) and with its ◀ ▶ buttons.
 */
function drawSurface3D(canvas, spec) {
  const xs = spec.x, ys = spec.y, zb = spec.z, nx = xs.length, ny = ys.length, layers = spec.layers || [];
  const cw = Math.max(260, canvas.clientWidth || canvas.parentElement.clientWidth || 600);
  const C = setup(canvas, spec.height || Math.round(Math.min(480, Math.max(240, 0.62 * cw)))), { ctx, w, h } = C;
  const bar = layers.find((l) => l.c) || null, m = { l: 10, r: bar ? 70 : 10, t: 10, b: 30 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const az = ((spec.azimuth ?? 205) * Math.PI) / 180, el = (Math.max(8, Math.min(85, spec.elevation ?? 30)) * Math.PI) / 180;
  const fin = Number.isFinite, cx = 0.5 * (xs[0] + xs[nx - 1]), cy = 0.5 * (ys[0] + ys[ny - 1]), Lx = Math.abs(xs[nx - 1] - xs[0]) || 1, Ly = Math.abs(ys[ny - 1] - ys[0]) || 1;
  let zlo = Infinity, zhi = -Infinity;
  const span = (a) => { for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const v = a[j][i]; if (fin(v)) { if (v < zlo) zlo = v; if (v > zhi) zhi = v; } } };
  span(zb); for (const l of layers) span(l.z); for (const q of spec.planes || []) { if (q.z < zlo) zlo = q.z; if (q.z > zhi) zhi = q.z; }
  if (!(zhi > zlo)) { zlo = (fin(zlo) ? zlo : 0) - 0.5; zhi = zlo + 1; }
  const zs = spec.zscale > 0 ? spec.zscale : (0.22 * Math.max(Lx, Ly)) / (zhi - zlo);
  // camera at compass bearing az: right = (−cos az, sin az), away = (−sin az, −cos az)
  const rx = -Math.cos(az), ry = Math.sin(az), dxv = -Math.sin(az), dyv = -Math.cos(az), se = Math.sin(el), ce = Math.cos(el);
  const P3 = (x, y, z) => { const X0 = x - cx, Y0 = y - cy, a = X0 * rx + Y0 * ry, d = X0 * dxv + Y0 * dyv, zz = (z - zlo) * zs; return [a, d * se + zz * ce, d * ce - zz * se]; };
  let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
  for (const x of [xs[0], xs[nx - 1]]) for (const y of [ys[0], ys[ny - 1]]) for (const z of [zlo, zhi]) { const q = P3(x, y, z); if (q[0] < a0) a0 = q[0]; if (q[0] > a1) a1 = q[0]; if (q[1] < b0) b0 = q[1]; if (q[1] > b1) b1 = q[1]; }
  const sc = Math.min(pw / (a1 - a0 || 1), ph / (b1 - b0 || 1)), ox = m.l + 0.5 * (pw - sc * (a1 - a0)), oy = m.t + 0.5 * (ph - sc * (b1 - b0));
  const SX = (q) => ox + (q[0] - a0) * sc, SY = (q) => oy + (b1 - q[1]) * sc;
  // light from the upper left of the view
  const lx = -0.45 * rx - 0.35 * dxv, ly = -0.45 * ry - 0.35 * dyv, lz = 0.82, ln = Math.hypot(lx, ly, lz);
  const shadeOf = (x0, y0, z00, z10, z01, x1, y1) => { const gx = ((z10 - z00) * zs) / (x1 - x0 || 1), gy = ((z01 - z00) * zs) / (y1 - y0 || 1), n = Math.hypot(gx, gy, 1); return 0.5 + 0.62 * Math.max(0, (-gx * lx - gy * ly + lz) / (n * ln)); };
  const Lb = lut(spec.cmap || 'topo'), glo = spec.zmin ?? zlo, ghi = spec.zmax ?? zhi, mid = spec.zmid !== undefined && spec.zmid > glo && spec.zmid < ghi ? spec.zmid : null, sA = (spec.cmap || 'topo') === 'topo' ? 7 / 15 : 0.5, sB = (spec.cmap || 'topo') === 'topo' ? 8 / 15 : 0.5;
  const gnorm = (v) => (mid === null ? (v - glo) / (ghi - glo || 1) : v < mid ? (sA * (v - glo)) / (mid - glo) : sB + ((1 - sB) * (v - mid)) / (ghi - mid));
  const rgb = (L, t, sh, al) => { const k = Math.max(0, Math.min(255, Math.round(t * 255))) * 3; return `rgba(${Math.min(255, Math.round(L[k] * sh))},${Math.min(255, Math.round(L[k + 1] * sh))},${Math.min(255, Math.round(L[k + 2] * sh))},${al})`; };
  const st = Math.max(1, Math.ceil(Math.sqrt(((nx - 1) * (ny - 1)) / 9000))), polys = [];
  const quad = (i, j, i2, j2, za, zb2, zc, zd, fill, bias, edge) => {
    const p0 = P3(xs[i], ys[j], za), p1 = P3(xs[i2], ys[j], zb2), p2 = P3(xs[i2], ys[j2], zc), p3 = P3(xs[i], ys[j2], zd);
    polys.push({ d: 0.25 * (p0[2] + p1[2] + p2[2] + p3[2]) - bias, f: fill, e: edge, p: [SX(p0), SY(p0), SX(p1), SY(p1), SX(p2), SY(p2), SX(p3), SY(p3)] });
  };
  for (let j = 0; j < ny - 1; j += st) for (let i = 0; i < nx - 1; i += st) {
    const i2 = Math.min(nx - 1, i + st), j2 = Math.min(ny - 1, j + st), za = zb[j][i], zb2 = zb[j][i2], zc = zb[j2][i2], zd = zb[j2][i];
    if (fin(za + zb2 + zc + zd)) quad(i, j, i2, j2, za, zb2, zc, zd, rgb(Lb, gnorm(0.25 * (za + zb2 + zc + zd)), shadeOf(xs[i], ys[j], za, zb2, zd, xs[i2], ys[j2]), 1), 0, null);
  }
  const eps = 1e-4 * Math.max(Lx, Ly), barRange = [0, 1];
  layers.forEach((l, li) => {
    const base = l.base || zb, minT = l.minThickness ?? 1e-9 * (zhi - zlo), Lc = l.c ? lut(l.cmap || 'heat') : null, col = l.color ? hex(l.color) : [234, 88, 12], al = l.opacity ?? 0.92;
    let clo = l.cmin, chi = l.cmax;
    if (Lc && (clo === undefined || chi === undefined)) { let a = Infinity, b = -Infinity; for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const v = l.c[j][i]; if (fin(v) && fin(l.z[j][i]) && l.z[j][i] - (fin(base[j][i]) ? base[j][i] : l.z[j][i]) > minT) { if (v < a) a = v; if (v > b) b = v; } } if (clo === undefined) clo = fin(a) ? a : 0; if (chi === undefined) chi = fin(b) ? b : 1; }
    if (Lc && !(chi > clo)) chi = clo + (Math.abs(clo) * 1e-6 || 1e-9);
    if (l === bar) { barRange[0] = clo; barRange[1] = chi; }
    for (let j = 0; j < ny - 1; j += st) for (let i = 0; i < nx - 1; i += st) {
      const i2 = Math.min(nx - 1, i + st), j2 = Math.min(ny - 1, j + st);
      const g = [base[j][i], base[j][i2], base[j2][i2], base[j2][i]], t = [l.z[j][i], l.z[j][i2], l.z[j2][i2], l.z[j2][i]];
      if (!fin(g[0] + g[1] + g[2] + g[3] + t[0] + t[1] + t[2] + t[3])) continue;
      for (let k = 0; k < 4; k++) if (t[k] < g[k]) t[k] = g[k];
      if (Math.max(t[0] - g[0], t[1] - g[1], t[2] - g[2], t[3] - g[3]) <= minT) continue;
      const sh = shadeOf(xs[i], ys[j], t[0], t[1], t[3], xs[i2], ys[j2]);
      let fill;
      if (Lc) { const cv = [l.c[j][i], l.c[j][i2], l.c[j2][i2], l.c[j2][i]].filter(fin), cm = cv.length ? cv.reduce((a, b) => a + b, 0) / cv.length : clo; fill = rgb(Lc, (cm - clo) / (chi - clo), sh, al); }
      else fill = `rgba(${Math.min(255, Math.round(col[0] * sh))},${Math.min(255, Math.round(col[1] * sh))},${Math.min(255, Math.round(col[2] * sh))},${al})`;
      quad(i, j, i2, j2, t[0], t[1], t[2], t[3], fill, eps * (li + 1), 'rgba(15,23,42,.22)');
    }
  });
  polys.sort((a, b) => b.d - a.d);
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, w, h - 2); ctx.clip();
  ctx.lineJoin = 'round';
  for (const q of polys) {
    const p = q.p;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(p[2], p[3]); ctx.lineTo(p[4], p[5]); ctx.lineTo(p[6], p[7]); ctx.closePath();
    ctx.fillStyle = q.f; ctx.fill(); ctx.strokeStyle = q.e || q.f; ctx.lineWidth = q.e ? 0.6 : 0.8; ctx.stroke(); // the same-colour stroke closes the hairline seams between neighbours
  }
  const path = (pts, close) => { ctx.beginPath(); pts.forEach((q, k) => (k ? ctx.lineTo(SX(q), SY(q)) : ctx.moveTo(SX(q), SY(q)))); if (close) ctx.closePath(); };
  ctx.font = '11px system-ui, sans-serif';
  for (const q of spec.planes || []) {
    const cor = [P3(xs[0], ys[0], q.z), P3(xs[nx - 1], ys[0], q.z), P3(xs[nx - 1], ys[ny - 1], q.z), P3(xs[0], ys[ny - 1], q.z)];
    ctx.strokeStyle = q.color || '#38bdf8'; ctx.lineWidth = 1.2; ctx.setLineDash([6, 4]); path(cor, true); ctx.stroke(); ctx.setLineDash([]);
    if (q.fill) { ctx.fillStyle = q.fill; path(cor, true); ctx.fill(); }
    if (q.name) { const top = cor.reduce((a, b) => (SY(b) < SY(a) ? b : a)); ctx.fillStyle = q.color || '#38bdf8'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(q.name, Math.max(m.l + 60, Math.min(w - m.r - 60, SX(top))), Math.max(m.t + 13, SY(top) - 3)); }
  }
  for (const q of spec.markers || []) {
    const a = P3(q.x, q.y, q.z0 ?? zlo), b = P3(q.x, q.y, q.z1 ?? zhi);
    ctx.strokeStyle = q.color || C.fg; ctx.lineWidth = 1.6; path([a, b]); ctx.stroke();
    ctx.fillStyle = q.color || C.fg; ctx.beginPath(); ctx.arc(SX(b), SY(b), 3, 0, 6.2832); ctx.fill();
    if (q.label) { ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.strokeStyle = C.bg; ctx.lineWidth = 3; ctx.strokeText(q.label, SX(b) + 6, SY(b)); ctx.fillStyle = C.fg; ctx.fillText(q.label, SX(b) + 6, SY(b)); }
  }
  ctx.restore();
  // orientation: north arrow on the ground plane, horizontal scale and the view parameters
  { const o = [m.l + 26, h - m.b - 22], nxs = ry, nys = dyv * se, nl = Math.hypot(nxs, nys) || 1, ax = (18 * nxs) / nl, ay = (-18 * nys) / nl;
    ctx.strokeStyle = C.fg; ctx.fillStyle = C.fg; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(o[0] - 0.5 * ax, o[1] - 0.5 * ay); ctx.lineTo(o[0] + ax, o[1] + ay); ctx.stroke();
    const an = Math.atan2(ay, ax); ctx.beginPath(); ctx.moveTo(o[0] + ax, o[1] + ay); ctx.lineTo(o[0] + ax - 6 * Math.cos(an - 0.5), o[1] + ay - 6 * Math.sin(an - 0.5)); ctx.lineTo(o[0] + ax - 6 * Math.cos(an + 0.5), o[1] + ay - 6 * Math.sin(an + 0.5)); ctx.fill();
    ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('N', o[0] + 1.6 * ax, o[1] + 1.6 * ay); }
  const azd = ((((spec.azimuth ?? 205) % 360) + 360) % 360), pts = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'][Math.round(azd / 45) % 8];
  ctx.fillStyle = C.mute; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.font = '11px system-ui, sans-serif';
  { const parts = [`Seen from the ${pts}`, `${Math.round((el * 180) / Math.PI)}° above the horizon`, `vertical scale × ${zs >= 10 ? Math.round(zs) : +zs.toPrecision(2)}`, tickLabel(+Lx.toPrecision(3)) + ' × ' + tickLabel(+Ly.toPrecision(3)) + (spec.xunit ? ' ' + spec.xunit : '')];
    while (parts.length > 1 && ctx.measureText(parts.join(' · ')).width > w - m.l - 6) parts.pop(); // drop the last items on narrow screens
    ctx.fillText(parts.join(' · '), m.l, h - 5); }
  if (bar) { // colour bar of the first layer coloured by a value
    const Lc = lut(bar.cmap || 'heat'), bx = w - m.r + 12, bwid = 12, lo = barRange[0], hi = barRange[1];
    for (let k = 0; k < ph; k++) { const t = Math.round((1 - k / ph) * 255); ctx.fillStyle = `rgb(${Lc[t * 3]},${Lc[t * 3 + 1]},${Lc[t * 3 + 2]})`; ctx.fillRect(bx, m.t + k, bwid, 1.5); }
    ctx.strokeStyle = C.mute; ctx.lineWidth = 1; ctx.strokeRect(bx, m.t, bwid, ph);
    ctx.fillStyle = C.mute; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (const v of niceTicks(lo, hi, 5)) if (v >= lo && v <= hi) ctx.fillText(tickLabel(v), bx + bwid + 4, m.t + ph - ((v - lo) / (hi - lo)) * ph);
    if (bar.clabel) { // the label shrinks, then loses its tail, so that it always fits beside the bar
      let txt = bar.clabel + (bar.cunit ? ' (' + bar.cunit + ')' : ''), px = 11;
      ctx.save(); ctx.translate(w - 5, m.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = C.fg;
      while (px > 9 && ctx.measureText(txt).width > ph) { px -= 0.5; ctx.font = `${px}px system-ui, sans-serif`; }
      while (txt.length > 8 && ctx.measureText(txt).width > ph) txt = txt.slice(0, -2).trimEnd() + '…';
      ctx.fillText(txt, 0, 0); ctx.restore(); }
  }
  canvas._hover = null; canvas._pick = null; canvas._polys = polys.length;
}

/** The spec shown for frame k of a plot with `frames` (the frame's fields override those of the spec). */
export function frameSpec(spec, k) {
  const f = spec.frames && spec.frames[k];
  return f ? { ...spec, ...f, frames: undefined, label: undefined } : spec;
}

export function draw(canvas, spec) {
  try {
    if (spec.type === 'bar') drawBar(canvas, spec);
    else if (spec.type === 'surface3d') drawSurface3D(canvas, spec);
    else if (spec.type === 'field') drawField(canvas, spec);
    else drawLine(canvas, spec);
  } catch (e) {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ef4444'; ctx.font = '12px system-ui'; ctx.fillText('Plot error: ' + e.message, 10, 20);
  }
}

/** Tabular form of a plot for CSV export. */
export function plotToCSV(spec) {
  const esc = (v) => (typeof v === 'string' && /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v ?? '');
  const rows = [];
  if (spec.type === 'bar') {
    rows.push(['category', ...spec.series.map((s) => s.name || 'value')]);
    spec.categories.forEach((c, i) => rows.push([c, ...spec.series.map((s) => s.values[i])]));
  } else if (spec.type === 'field') {
    rows.push([`${spec.ylabel || 'y'} \\ ${spec.xlabel || 'x'}`, ...spec.x]);
    spec.y.forEach((yv, j) => rows.push([yv, ...Array.from(spec.z[j])]));
  } else if (spec.type === 'surface3d') { // one block per surface: the ground, then the upper surface (and colour value) of every layer
    const block = (name, a) => { if (rows.length) rows.push([]); rows.push([`${name} — ${spec.ylabel || 'y'} \\ ${spec.xlabel || 'x'}`, ...spec.x]); spec.y.forEach((yv, j) => rows.push([yv, ...Array.from(a[j], (v) => (Number.isFinite(v) ? v : ''))])); };
    block(spec.zlabel || 'ground elevation', spec.z);
    for (const l of spec.layers || []) { block(`${l.name || 'layer'}: upper surface`, l.z); if (l.c) block(`${l.name || 'layer'}: ${l.clabel || 'value'}${l.cunit ? ' (' + l.cunit + ')' : ''}`, l.c); }
  } else {
    const n = Math.max(...spec.series.map((s) => s.x.length));
    rows.push(spec.series.flatMap((s) => [`${s.name || 'series'} · ${spec.xlabel || 'x'}`, `${s.name || 'series'} · ${spec.ylabel || 'y'}`]));
    for (let k = 0; k < n; k++) rows.push(spec.series.flatMap((s) => [s.x[k] ?? '', s.y[k] ?? '']));
  }
  return rows.map((r) => r.map(esc).join(',')).join('\n');
}

/** Build a self-contained chart card (title, canvas, hover read-out, export buttons). */
export function plotCard(spec, { onDownload } = {}) {
  const card = document.createElement('figure');
  card.className = 'plot';
  const head = document.createElement('figcaption');
  const title = document.createElement('span'); title.textContent = spec.title || '';
  const tools = document.createElement('span'); tools.className = 'plot-tools';
  const mk = (label, tip, fn) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'mini'; b.textContent = label; b.title = tip; b.addEventListener('click', fn); tools.append(b); };
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', spec.title || 'chart');
  const tip = document.createElement('div'); tip.className = 'plot-tip'; tip.hidden = true;
  const safe = (spec.title || 'plot').replace(/[^\w-]+/g, '_').slice(0, 60);
  // frames: several versions of the chart (depth slices, section positions, iso-levels …) chosen with a slider; `cur` is the one shown
  const frames = Array.isArray(spec.frames) && spec.frames.length > 1 ? spec.frames : null, is3d = spec.type === 'surface3d';
  let fk = frames ? Math.max(0, Math.min(frames.length - 1, Math.round(spec.frame ?? 0))) : 0, cur = frames ? frameSpec(spec, fk) : spec;
  const view = is3d ? { az: spec.azimuth ?? 205, el: spec.elevation ?? 30 } : null;
  const shown = () => (is3d ? Object.assign(cur === spec ? { ...spec } : cur, { azimuth: view.az, elevation: view.el }) : cur);
  const fileName = () => (frames ? safe + '_' + String(frames[fk].label ?? fk + 1).replace(/[^\w.-]+/g, '_').slice(0, 30) : safe);
  if (is3d) { mk('◀', 'Rotate the view to the left', () => { view.az -= 30; render(); }); mk('▶', 'Rotate the view to the right', () => { view.az += 30; render(); }); }
  mk('PNG', 'Download this chart as an image', () => canvas.toBlob((b) => onDownload && onDownload(b, fileName() + '.png')));
  mk('CSV', 'Download the plotted data as a table', () => onDownload && onDownload(new Blob([plotToCSV(shown())], { type: 'text/csv' }), fileName() + '.csv'));
  head.append(title, tools);
  const wrap = document.createElement('div'); wrap.className = 'plot-wrap'; wrap.append(canvas, tip);
  card.append(head);
  if (frames) {
    const row = document.createElement('label'), cap = document.createElement('span'), rng = document.createElement('input'), val = document.createElement('output');
    row.className = 'plot-frames'; Object.assign(row.style, { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', fontSize: '12.5px', margin: '2px 0 6px' });
    cap.textContent = spec.frameLabel || 'Frame';
    rng.type = 'range'; rng.min = '0'; rng.max = String(frames.length - 1); rng.step = '1'; rng.value = String(fk);
    Object.assign(rng.style, { flex: '1 1 140px', minWidth: '120px', minHeight: '28px', margin: '0' });
    val.style.fontWeight = '600';
    const show = () => { const f = frames[fk]; val.textContent = String(f.label ?? `${fk + 1} of ${frames.length}`); rng.setAttribute('aria-valuetext', val.textContent); title.textContent = f.title || spec.title || ''; canvas.setAttribute('aria-label', title.textContent || 'chart'); };
    rng.addEventListener('input', () => { fk = Math.max(0, Math.min(frames.length - 1, +rng.value || 0)); cur = frameSpec(spec, fk); show(); tip.hidden = true; render(); });
    show(); row.append(cap, rng, val); card.append(row);
    card._setFrame = (k) => { rng.value = String(k); rng.dispatchEvent(new Event('input')); };
  }
  card.append(wrap);
  if (spec.note) { const n = document.createElement('p'); n.className = 'note'; n.textContent = spec.note; card.append(n); }
  const render = () => { if (canvas.isConnected && canvas.clientWidth) draw(canvas, shown()); };
  if (is3d) { // drag to rotate: horizontally the azimuth (touch as well: vertical swipes keep scrolling the page), vertically the elevation with a mouse or pen
    let drag = null, pend = false;
    canvas.style.cursor = 'grab'; canvas.tabIndex = 0;
    const turn = () => { if (pend) return; pend = true; requestAnimationFrame(() => { pend = false; render(); }); };
    canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, az: view.az, el: view.el }; canvas.style.cursor = 'grabbing'; try { canvas.setPointerCapture(e.pointerId); } catch (er) { /* not capturable */ } });
    canvas.addEventListener('pointermove', (e) => { if (!drag) return; view.az = drag.az - 0.45 * (e.clientX - drag.x); if (e.pointerType !== 'touch') view.el = Math.max(8, Math.min(85, drag.el + 0.35 * (e.clientY - drag.y))); turn(); });
    const end = () => { drag = null; canvas.style.cursor = 'grab'; };
    canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('keydown', (e) => { const d = { ArrowLeft: [-15, 0], ArrowRight: [15, 0], ArrowUp: [0, 5], ArrowDown: [0, -5] }[e.key]; if (!d) return; e.preventDefault(); view.az += d[0]; view.el = Math.max(8, Math.min(85, view.el + d[1])); render(); });
    card._view = view;
  }
  const ro = new ResizeObserver(() => render());
  ro.observe(wrap);
  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect(), hv = canvas._hover && canvas._hover(e.clientX - r.left, e.clientY - r.top);
    if (!hv) { tip.hidden = true; return; }
    tip.hidden = false; tip.textContent = hv.text;
    tip.style.left = Math.min(r.width - 150, Math.max(0, hv.x + 10)) + 'px'; tip.style.top = Math.max(0, hv.y - 30) + 'px';
  });
  canvas.addEventListener('pointerleave', () => (tip.hidden = true));
  if (typeof spec.onPick === 'function') { // clickable field plots (e.g. choose a point on a map)
    canvas.style.cursor = 'crosshair';
    canvas.addEventListener('click', (e) => { const r = canvas.getBoundingClientRect(), p = canvas._pick && canvas._pick(e.clientX - r.left, e.clientY - r.top); if (p) spec.onPick(p.x, p.y); });
  }
  card._redraw = render;
  return card;
}

export { PALETTE };
