// Dependency-free canvas plotting: line/scatter curves, bar charts and 2-D fields
// (filled contours, iso-lines, streamlines, vectors). Every chart supports hover read-out and PNG/CSV export.
import { fmt } from './num.js';

const PALETTE = ['#0ea5e9', '#f97316', '#10b981', '#a855f7', '#ef4444', '#eab308', '#14b8a6', '#ec4899', '#6366f1', '#84cc16', '#f43f5e', '#06b6d4'];
const CMAPS = {
  viridis: ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'],
  turbo: ['#30123b', '#4662d7', '#36aaf9', '#1ae4b6', '#72fe5e', '#c7ef34', '#faba39', '#f66b19', '#ca2a04', '#7a0403'],
  coolwarm: ['#3b4cc0', '#6f92f3', '#aac7fd', '#dddcdc', '#f7b89c', '#e7745b', '#b40426'],
  salinity: ['#f7fcf0', '#ccebc5', '#7bccc4', '#43a2ca', '#0868ac', '#084081', '#3f007d', '#7a0177'],
  // below the midpoint: sea (deep → shallow); above it: land (lowland → mountain)
  topo: ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#e3f0fa', '#a8d08d', '#c9dd9a', '#e9e3a0', '#d9bf77', '#b98f55', '#96673f', '#7a5a4a', '#5a463f'],
  land: ['#a8d08d', '#c9dd9a', '#e9e3a0', '#d9bf77', '#b98f55', '#96673f', '#7a5a4a', '#5a463f'],
  sea: ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#e3f0fa'],
  thermal: ['#042333', '#2c3395', '#744992', '#b15f82', '#eb7958', '#fbb43d', '#e8fa5b'],
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
    if (q.label) { ctx.fillStyle = q.color || '#ef4444'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(q.label, w - m.r - 4, Y(q.y) - 2); }
  }
  for (const q of spec.vlines || []) {
    ctx.strokeStyle = q.color || '#ef4444'; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(X(q.x), m.t); ctx.lineTo(X(q.x), h - m.b); ctx.stroke(); ctx.setLineDash([]);
    if (q.label) { ctx.fillStyle = q.color || '#ef4444'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(q.label, X(q.x) + 4, m.t + 3); }
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
  const norm = (v) => (mid === null ? (v - lo) / (hi - lo) : v < mid ? (0.5 * (v - lo)) / (mid - lo) : 0.5 + (0.5 * (v - mid)) / (hi - mid));
  const inv = (t) => (mid === null ? lo + t * (hi - lo) : t < 0.5 ? lo + (t / 0.5) * (mid - lo) : mid + ((t - 0.5) / 0.5) * (hi - mid));
  // Smooth fields are resampled to screen resolution by bilinear interpolation of the VALUES (then coloured),
  // which keeps colour boundaries such as a shoreline sharp; masked fields stay cell-exact.
  const smooth = !spec.mask && nx > 2 && ny > 2 && spec.interpolate !== false;
  const dprF = Math.min(window.devicePixelRatio || 1, 2), W = smooth ? Math.max(nx, Math.min(900, Math.round(pw * dprF))) : nx, H = smooth ? Math.max(ny, Math.min(900, Math.round(ph * dprF))) : ny;
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const octx = off.getContext('2d'), img = octx.createImageData(W, H);
  const cellx = Math.abs(dx) / (nx - 1 || 1), celly = Math.abs(dy) / (ny - 1 || 1), geo = spec.shade === 'geo', kx = geo ? 111320 * Math.cos((((ys[0] + ys[ny - 1]) / 2) * Math.PI) / 180) : 1, ky = geo ? 110540 : 1;
  const zAt = (i, j) => z[Math.max(0, Math.min(ny - 1, j))][Math.max(0, Math.min(nx - 1, i))];
  for (let r = 0; r < H; r++) {
    const fj = smooth ? ((H - 1 - r) / (H - 1)) * (ny - 1) : ny - 1 - r, j0 = Math.min(ny - 2 < 0 ? 0 : ny - 2, Math.floor(fj)), b = smooth ? fj - j0 : 0;
    for (let cI = 0; cI < W; cI++) {
      const o = (r * W + cI) * 4;
      let v, shade = 1;
      if (smooth) {
        const fi = (cI / (W - 1)) * (nx - 1), i0 = Math.min(nx - 2, Math.floor(fi)), a = fi - i0;
        const v00 = z[j0][i0], v10 = z[j0][i0 + 1], v01 = z[j0 + 1][i0], v11 = z[j0 + 1][i0 + 1];
        v = Number.isFinite(v00 + v10 + v01 + v11) ? v00 * (1 - a) * (1 - b) + v10 * a * (1 - b) + v01 * (1 - a) * b + v11 * a * b : z[Math.round(fj)][Math.round(fi)];
        if (spec.shade && Number.isFinite(v)) { // hill-shading from the local slope, light from the north-west
          const gx = ((v10 - v00) * (1 - b) + (v11 - v01) * b) / (cellx * kx), gy = ((v01 - v00) * (1 - a) + (v11 - v10) * a) / (celly * ky), ex = spec.exaggeration || 1;
          shade = Math.max(0.55, Math.min(1.25, 1 + 0.9 * ((-gx + gy) * ex) / Math.sqrt(1 + (gx * ex) ** 2 + (gy * ex) ** 2)));
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
  octx.putImageData(img, 0, 0);
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
    for (const lv of levels) {
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

export function draw(canvas, spec) {
  try {
    if (spec.type === 'bar') drawBar(canvas, spec);
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
  mk('PNG', 'Download this chart as an image', () => canvas.toBlob((b) => onDownload && onDownload(b, safe + '.png')));
  mk('CSV', 'Download the plotted data as a table', () => onDownload && onDownload(new Blob([plotToCSV(spec)], { type: 'text/csv' }), safe + '.csv'));
  head.append(title, tools);
  const wrap = document.createElement('div'); wrap.className = 'plot-wrap'; wrap.append(canvas, tip);
  card.append(head, wrap);
  if (spec.note) { const n = document.createElement('p'); n.className = 'note'; n.textContent = spec.note; card.append(n); }
  const render = () => { if (canvas.isConnected && canvas.clientWidth) draw(canvas, spec); };
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
