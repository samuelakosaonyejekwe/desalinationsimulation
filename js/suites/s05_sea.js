// Suite 5 — Brine discharge into the sea.
// Near field: integral model of inclined negatively buoyant jets (volume, momentum, salt and heat fluxes
// with a jet/plume/cross-flow entrainment closure) from single-port or multiport diffusers, cross-checked
// against the empirical dense-jet coefficients. Intermediate field: entraining bottom gravity current.
// Far field: transient advection–dispersion of excess salinity over real or synthetic bathymetry, driven
// by tidal-harmonic, residual and wind-drift currents whose spatial pattern follows a rigid-lid,
// friction-dominated shallow-water balance. Regulatory mixing-zone, receptor and intake assessment.
import { rk45, clamp, linspace, fmt, rng, interp1, mean } from '../core/num.js';
import { density, G, salinityFromTDS } from '../core/props.js';
import { pcg5 } from './s04_cfd.js';

export const TIDES = { M2: 12.4206012, S2: 12.0, K1: 23.93447213, O1: 25.81933871 }; // constituent periods, h
const D2R = Math.PI / 180, HMIN = 0.5; // cells shallower than HMIN are treated as land
const bearing = (deg) => [Math.sin(deg * D2R), Math.cos(deg * D2R)]; // compass bearing → (east, north)
const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
/** Okubo (1971) scale-dependent horizontal diffusivity, m²/s for a length scale in m. */
export const okubo = (l) => 2.055e-4 * Math.max(l, 1) ** 1.15;
// Empirical coefficients for 60° dense jets in still water (Roberts, Ferrier & Daviero 1997), all × d·F or × F
export const ROBERTS60 = { zt: 2.2, xi: 2.4, Si: 1.6, xn: 9.0, Sn: 2.6, yL: 0.7 };

/**
 * Integral model of a round turbulent buoyant jet (top-hat profiles, Boussinesq).
 * State along the arc length s: Q, Mx, My, Mz (kinematic fluxes), Q·S, Q·T, x, y, z.
 * Entrainment E = P[α|V − u_a·e_x| + β u_a sinγ], α = α_j + k(g'b/V²)e_z bounded by the plume value, with an
 * enhancement on the descending limb of dense jets; P = 2πb, or 2s after neighbouring jets merge.
 * amb(z) → { S, T, ua } with the ambient current along +x. Angles in radians.
 */
export function denseJet(o) {
  const { d, U0, theta, Sb, Tb, amb, z0 = 0, alphaJ = 0.07, alphaP = 0.117, betaX = 0.5, descF = 2, spacing = Infinity, sigma = 0, depth = Infinity, bedSlope = 0, tol = 1e-6 } = o;
  const a0 = amb(z0), rho0 = density(a0.T, a0.S), rhoB = density(Tb, Sb), Q0 = ((Math.PI * d * d) / 4) * U0, gp0 = (G * (rhoB - rho0)) / rho0;
  const F = U0 / Math.sqrt(Math.max(Math.abs(gp0) * d, 1e-9)), kB = (alphaP - alphaJ) / (1.6 * alphaP);
  const shape = (y) => { const M = Math.hypot(y[1], y[2], y[3]), V = M / y[0], b = Math.sqrt(y[0] / (Math.PI * V)), merged = 2 * b >= spacing; return { M, V, b, merged, h: merged ? (2 * b * b) / spacing : b, ex: y[1] / M, ey: y[2] / M, ez: y[3] / M }; };
  const bed = (x, yy) => -bedSlope * Math.hypot(x, yy); // bed level relative to the port foot, falling away along the discharge
  const f = (s, y) => {
    const g = shape(y), a = amb(clamp(y[8], 0, Number.isFinite(depth) ? depth : 1e9)), gp = (G * (density(a.T, a.S) - density(y[5] / y[0], y[4] / y[0]))) / rho0;
    let al = clamp(alphaJ + kB * ((gp * g.b) / (g.V * g.V)) * g.ez, 0.6 * alphaJ, alphaP);
    if (g.ez < 0 && gp < 0) al *= descF;
    const P = g.merged ? 2 * spacing : 2 * Math.PI * g.b, E = P * (al * Math.abs(g.V - a.ua * g.ex) + betaX * a.ua * Math.sqrt(Math.max(0, 1 - g.ex * g.ex)));
    return [E, E * a.ua, 0, (gp * y[0]) / g.V, E * a.S, E * a.T, g.ex, g.ey, g.ez];
  };
  const y0 = [Q0, Q0 * U0 * Math.cos(theta) * Math.cos(sigma), Q0 * U0 * Math.cos(theta) * Math.sin(sigma), Q0 * U0 * Math.sin(theta), Q0 * Sb, Q0 * Tb, 0, 0, z0];
  const sMax = o.sMax ?? Math.min(8000 * d, d * (60 + 14 * Math.min(F, 400)) * (1 + (40 * a0.ua) / U0));
  let fate = 'range';
  const stop = (s, y) => { if (y[8] <= bed(y[6], y[7]) && y[3] < 0) { fate = 'seabed'; return true; } if (y[8] >= depth) { fate = 'surface'; return true; } return false; };
  const r = rk45(f, y0, 0, sMax, { rtol: tol, atol: 1e-12, hInit: d / 20, maxSteps: 4000, stop });
  // refine the last step so that the end point sits on the bed / surface
  const n = r.y.length;
  if (fate !== 'range' && n > 1) {
    const a = r.y[n - 2], b = r.y[n - 1], ta = fate === 'seabed' ? a[8] - bed(a[6], a[7]) : depth - a[8], tb = fate === 'seabed' ? b[8] - bed(b[6], b[7]) : depth - b[8], w = ta - tb !== 0 ? clamp(ta / (ta - tb), 0, 1) : 1;
    r.y[n - 1] = a.map((q, k) => q + w * (b[k] - q)); r.t[n - 1] = r.t[n - 2] + w * (r.t[n - 1] - r.t[n - 2]);
  }
  const path = { s: [], r: [], x: [], y: [], z: [], b: [], h: [], up: [], lo: [], V: [], S: [], Sc: [], rho: [], sal: [], merged: false };
  let zt = -Infinity, zc = -Infinity;
  for (let k = 0; k < n; k++) {
    const y = r.y[k], g = shape(y), c = Math.sqrt(Math.max(0, 1 - g.ez * g.ez)), e = Math.SQRT2 * g.h * c; // visual edge = 2 Gaussian half-widths
    path.s.push(r.t[k]); path.x.push(y[6]); path.y.push(y[7]); path.z.push(y[8]); path.r.push(Math.hypot(y[6], y[7])); path.b.push(g.b); path.h.push(g.h);
    path.up.push(y[8] + e); path.lo.push(y[8] - e); path.V.push(g.V); path.S.push(y[0] / Q0); path.Sc.push(Math.max(1, y[0] / Q0 / 1.7));
    path.sal.push(y[4] / y[0]); path.rho.push(density(y[5] / y[0], y[4] / y[0]));
    if (g.merged) path.merged = true;
    zt = Math.max(zt, y[8] + e); zc = Math.max(zc, y[8]);
  }
  const L = n - 1;
  return { F, gp0, Q0, rho0, rhoB, fate, path, zt: zt - z0, zc: zc - z0, xi: path.r[L], Si: path.Sc[L], Sbulk: path.S[L], Vi: path.V[L], bi: path.b[L], zEnd: path.z[L], sEnd: path.s[L], steps: n };
}

/**
 * Entraining bottom gravity current on a slope (Ellison–Turner layer equations), with lateral buoyancy spreading.
 * State along x: q = U h W, m = U² h W, W. Buoyancy flux B = g'·q is conserved in a uniform ambient.
 */
export function bottomCurrent({ q0, B, h0, W0, slope, Cd = 0.003, xMax = 3000, depth = Infinity }) {
  const sinb = slope / Math.hypot(1, slope), cosb = 1 / Math.hypot(1, slope), U0 = q0 / (h0 * W0);
  const st = (y) => { const U = y[1] / y[0], h = (y[0] * y[0]) / (y[1] * y[2]), gp = B / y[0], Ri = (gp * h * cosb) / (U * U); return { U, h, gp, Ri, E: 0.00153 / (0.0204 + Ri) }; }; // entrainment law of Parker et al. (1987)
  const f = (x, y) => { const s = st(y); return [s.E * s.U * y[2], s.gp * s.h * y[2] * sinb - Cd * s.U * s.U * y[2], (1.2 * Math.sqrt(Math.max(s.gp * s.h, 0))) / Math.max(s.U, 1e-6)]; };
  let arrest = false;
  const r = rk45(f, [q0, q0 * U0, W0], 0, xMax, { rtol: 1e-6, atol: 1e-10, hInit: Math.max(0.5, xMax / 2000), maxSteps: 3000, stop: (x, y) => { const s = st(y); if (!(s.U > 0.01) || s.h > 0.8 * depth || s.Ri > 4) { arrest = true; return true; } return false; } });
  const out = { x: [], h: [], U: [], W: [], q: [], Ri: [], t: [], arrest };
  let t = 0;
  r.y.forEach((y, k) => { const s = st(y); if (k) t += (r.t[k] - r.t[k - 1]) / Math.max(0.5 * (s.U + out.U[k - 1]), 1e-6); out.x.push(r.t[k]); out.h.push(s.h); out.U.push(s.U); out.W.push(y[2]); out.q.push(y[0]); out.Ri.push(s.Ri); out.t.push(t); });
  return out;
}

/** Brooks (1960) far-field centre-line dilution of a line source of width b in a steady current u, 4/3-law diffusion. */
export function brooks(x, b, u, K0) {
  if (!(x > 0)) return 1;
  const beta = (12 * K0) / (Math.max(u, 1e-4) * b), a = (1 + ((2 / 3) * beta * x) / b) ** 3 - 1;
  return 1 / Math.max(erf(Math.sqrt(1.5 / a)), 1e-9);
}

/** Tidal-harmonic, residual and wind-drift velocity (east, north) at time t (s). */
export function currentAt(c, t) {
  if (c.series) { const T = c.series.t, tt = T[0] + ((((t / 3600 - T[0]) % c.series.span) + c.series.span) % c.series.span); return [interp1(T, c.series.u, tt) + c.wind[0], interp1(T, c.series.v, tt) + c.wind[1]]; }
  let a = 0;
  for (const k of c.cons) a += k.amp * Math.cos((2 * Math.PI * t) / (k.T * 3600) - k.ph);
  return [a * c.axis[0] + c.res[0] + c.wind[0], a * c.axis[1] + c.res[1] + c.wind[1]];
}

/** Convert an x, y, z table (metres or lon/lat; elevation or positive depth) into a bathymetry grid. */
export function bathyFromTable(t, name = 'bathymetry') {
  const hs = t.headers || [], find = (re, k) => hs.find((h) => re.test(String(h).toLowerCase())) ?? hs[k];
  const hx = find(/^(x|lon|long|longitude|east|easting)/, 0), hy = find(/^(y|lat|latitude|north|northing)/, 1), hz = find(/^(z|elev|height|depth|bathy)/, 2), isDepth = /depth/.test(String(hz).toLowerCase());
  let pts = (t.records || []).map((r) => [+r[hx], +r[hy], isDepth ? -Math.abs(+r[hz]) : +r[hz]]).filter((p) => p.every(Number.isFinite));
  if (pts.length < 4) throw new Error('The bathymetry table needs at least four rows of x (or longitude), y (or latitude) and elevation (or depth).');
  const uniq = (k) => [...new Set(pts.map((p) => p[k]))].sort((a, b) => a - b), xs = uniq(0), ys = uniq(1);
  const geo = xs[0] >= -180 && xs[xs.length - 1] <= 180 && ys[0] >= -90 && ys[ys.length - 1] <= 90 && xs[xs.length - 1] - xs[0] < 5;
  let gx = xs, gy = ys, elev;
  if (xs.length * ys.length === pts.length && xs.length > 1 && ys.length > 1) {
    const ix = new Map(xs.map((x, i) => [x, i])), iy = new Map(ys.map((y, i) => [y, i]));
    elev = ys.map(() => new Array(xs.length).fill(0));
    for (const p of pts) elev[iy.get(p[1])][ix.get(p[0])] = p[2];
  } else { // scattered soundings: inverse-distance gridding onto 40 × 40 nodes
    if (pts.length > 4000) { const k = Math.ceil(pts.length / 4000); pts = pts.filter((_, i) => i % k === 0); }
    gx = linspace(xs[0], xs[xs.length - 1], 40); gy = linspace(ys[0], ys[ys.length - 1], 40);
    const sx = (xs[xs.length - 1] - xs[0]) || 1, sy = (ys[ys.length - 1] - ys[0]) || 1;
    elev = gy.map((y) => gx.map((x) => { let sw = 0, sz = 0; for (const p of pts) { const d2 = ((p[0] - x) / sx) ** 2 + ((p[1] - y) / sy) ** 2, w = 1 / (d2 * d2 + 1e-12); sw += w; sz += w * p[2]; } return sz / sw; }));
  }
  return geo ? { lon: gx, lat: gy, elev, name } : { x: gx, y: gy, elev, name };
}

/** Bathymetry sampler in metres relative to the grid centre: (x east, y north) → elevation (m, negative below sea level). */
function bathySampler(b) {
  if (!b || !Array.isArray(b.elev) || !b.elev.length) return null;
  let xs, ys, e = b.elev;
  if (Array.isArray(b.lat) && Array.isArray(b.lon)) {
    const lat0 = 0.5 * (Math.min(...b.lat) + Math.max(...b.lat)), lon0 = 0.5 * (Math.min(...b.lon) + Math.max(...b.lon));
    xs = b.lon.map((l) => (l - lon0) * 111320 * Math.cos(lat0 * D2R)); ys = b.lat.map((l) => (l - lat0) * 110540);
  } else if (Array.isArray(b.x) && Array.isArray(b.y)) { const xm = 0.5 * (Math.min(...b.x) + Math.max(...b.x)), ym = 0.5 * (Math.min(...b.y) + Math.max(...b.y)); xs = b.x.map((x) => x - xm); ys = b.y.map((y) => y - ym); }
  else return null;
  if (e.length === xs.length && e[0]?.length === ys.length && xs.length !== ys.length) e = ys.map((_, j) => xs.map((__, i) => b.elev[i][j])); // stored as [x][y]
  if (e.length !== ys.length || e[0]?.length !== xs.length || xs.length < 2 || ys.length < 2) return null;
  if (xs[0] > xs[xs.length - 1]) { xs = [...xs].reverse(); e = e.map((r) => [...r].reverse()); }
  if (ys[0] > ys[ys.length - 1]) { ys = [...ys].reverse(); e = [...e].reverse(); }
  if (!e.every((r) => r.every(Number.isFinite)) || ![...xs, ...ys].every(Number.isFinite)) return null;
  const loc = (a, v) => { let i = 0; while (i < a.length - 2 && a[i + 1] < v) i++; return [i, clamp((v - a[i]) / (a[i + 1] - a[i] || 1), 0, 1)]; };
  const fn = (x, y) => { const [i, fx] = loc(xs, x), [j, fy] = loc(ys, y); return (e[j][i] * (1 - fx) + e[j][i + 1] * fx) * (1 - fy) + (e[j + 1][i] * (1 - fx) + e[j + 1][i + 1] * fx) * fy; };
  fn.extent = [xs[0], xs[xs.length - 1], ys[0], ys[ys.length - 1]];
  return fn;
}

/** Model grid (origin at the outfall) with bed elevation from site/imported bathymetry or a synthetic beach. */
function makeGrid(v, nx, ny) {
  const Lx = v.Lx, Ly = v.Ly, dx = Lx / nx, dy = Ly / ny, x0 = -Lx * clamp(v.fx / 100, 0.05, 0.95), y0 = -Ly * clamp(v.fy / 100, 0.05, 0.95);
  const xs = Array.from({ length: nx }, (_, i) => x0 + (i + 0.5) * dx), ys = Array.from({ length: ny }, (_, j) => y0 + (j + 0.5) * dy);
  const synth = (x, y) => -clamp(v.depth + (v.slope / 100) * (y + v.bayAmp * (1 - Math.cos((2 * Math.PI * x) / Lx))), -6, Math.max(4 * v.depth, 60));
  let fn = synth, source = 'synthetic', note = '';
  const smp = bathySampler(v.bathy);
  if (v.bathy && !smp) note = 'The supplied bathymetry could not be interpreted (expected { lat[], lon[], elev[][] } or an x, y, z table); the synthetic beach is used.';
  if (smp) {
    const zo = smp(v.outX, v.outY);
    if (zo > -1) note = `The bathymetry gives ${fmt(-zo, 3)} m of water at the outfall position — move the outfall offset into deeper water. The synthetic beach is used instead.`;
    else { fn = (x, y) => smp(x + v.outX, y + v.outY); source = v.bathy.name || 'site bathymetry'; const e = smp.extent; if (x0 + v.outX < e[0] - dx || x0 + Lx + v.outX > e[1] + dx || y0 + v.outY < e[2] - dy || y0 + Ly + v.outY > e[3] + dy) note = 'The model domain is larger than the bathymetry coverage; edge depths are extended outward.'; }
  }
  const zb = new Float64Array(nx * ny), H = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const z = fn(xs[i], ys[j]); zb[j * nx + i] = z; H[j * nx + i] = -z > HMIN ? -z : 0; }
  const io = clamp(Math.floor((0 - x0) / dx), 0, nx - 1), jo = clamp(Math.floor((0 - y0) / dy), 0, ny - 1);
  return { nx, ny, dx, dy, x0, y0, xs, ys, zb, H, io, jo, source, note, depthOut: Math.max(-fn(0, 0), 0), fn };
}

/**
 * Spatial pattern of the depth-averaged current: quasi-steady, rigid-lid, friction-dominated shallow-water
 * balance u = −c H^(2/3) ∇η with ∇·(H u) = 0, i.e. ∇·(H^(5/3) ∇η) = 0. Two solutions (forcing along x and y)
 * are combined so that the current at the outfall equals the prescribed tidal/residual vector.
 */
export function flowBasis(g, closed = false) {
  const { nx, ny, dx, dy, H, x0, y0 } = g, n = nx * ny, C = Float64Array.from(H, (h) => (h > 0 ? h ** (5 / 3) : 0));
  const aE = new Float64Array(n), aN = new Float64Array(n), dg = new Float64Array(n), bw = new Float64Array(n), be = new Float64Array(n), bs = new Float64Array(n), bn = new Float64Array(n);
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    if (!C[P]) continue;
    if (i < nx - 1 && C[P + 1]) aE[P] = ((2 * C[P] * C[P + 1]) / (C[P] + C[P + 1])) * (dy / dx);
    if (j < ny - 1 && C[P + nx]) aN[P] = ((2 * C[P] * C[P + nx]) / (C[P] + C[P + nx])) * (dx / dy);
    if (!closed) { if (i === 0) bw[P] = (2 * C[P] * dy) / dx; if (i === nx - 1) be[P] = (2 * C[P] * dy) / dx; if (j === 0) bs[P] = (2 * C[P] * dx) / dy; if (j === ny - 1) bn[P] = (2 * C[P] * dx) / dy; }
  }
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) dg[P] = C[P] ? aE[P] + aN[P] + (i > 0 ? aE[P - 1] : 0) + (j > 0 ? aN[P - nx] : 0) + bw[P] + be[P] + bs[P] + bn[P] : 0;
  const raw = [0, 1].map((k) => {
    const eb = (x, y) => -(k ? y : x), rhs = new Float64Array(n), eta = new Float64Array(n);
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { const xc = x0 + (i + 0.5) * dx, yc = y0 + (j + 0.5) * dy; rhs[P] = bw[P] * eb(xc - dx / 2, yc) + be[P] * eb(xc + dx / 2, yc) + bs[P] * eb(xc, yc - dy / 2) + bn[P] * eb(xc, yc + dy / 2); eta[P] = dg[P] ? eb(xc, yc) : 0; }
    pcg5(nx, ny, aE, aN, dg, rhs, eta, 1e-9, 600);
    const qx = new Float64Array((nx + 1) * ny), qy = new Float64Array(nx * (ny + 1)); // face transport × face length
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (!C[P]) continue;
      const xc = x0 + (i + 0.5) * dx, yc = y0 + (j + 0.5) * dy;
      if (i === 0) qx[j * (nx + 1)] = -bw[P] * (eta[P] - eb(xc - dx / 2, yc));
      qx[j * (nx + 1) + i + 1] = i === nx - 1 ? -be[P] * (eb(xc + dx / 2, yc) - eta[P]) : -aE[P] * (eta[P + 1] - eta[P]);
      if (j === 0) qy[i] = -bs[P] * (eta[P] - eb(xc, yc - dy / 2));
      qy[(j + 1) * nx + i] = j === ny - 1 ? -bn[P] * (eb(xc, yc + dy / 2) - eta[P]) : -aN[P] * (eta[P + nx] - eta[P]);
    }
    return { qx, qy };
  });
  const cellU = (b, P) => { const i = P % nx, j = (P - i) / nx; return H[P] ? [(0.5 * (b.qx[j * (nx + 1) + i] + b.qx[j * (nx + 1) + i + 1])) / (H[P] * dy), (0.5 * (b.qy[P] + b.qy[P + nx])) / (H[P] * dx)] : [0, 0]; };
  // normalise: unit eastward / northward current at the outfall cell (domain mean if that cell is sheltered)
  const Po = g.jo * nx + g.io, [a, b] = cellU(raw[0], Po), [c, d] = cellU(raw[1], Po), det = a * d - b * c;
  let ma = 0, md = 0, m = 0;
  for (let P = 0; P < n; P++) if (H[P]) { ma += cellU(raw[0], P)[0]; md += cellU(raw[1], P)[1]; m++; }
  ma /= Math.max(m, 1); md /= Math.max(m, 1);
  const still = closed || !(Math.abs(ma) > 1e-9) || !(Math.abs(md) > 1e-9); // no through-flow is possible in a closed basin (rigid lid)
  const local = !still && Math.abs(det) > 0.05 * Math.abs(ma * md);
  const w = still ? [[0, 0], [0, 0]] : local ? [[d / det, -b / det], [-c / det, a / det]] : [[1 / ma, 0], [0, 1 / md]];
  const basis = w.map(([w1, w2]) => {
    const qx = raw[0].qx.map((q, k) => w1 * q + w2 * raw[1].qx[k]), qy = raw[0].qy.map((q, k) => w1 * q + w2 * raw[1].qy[k]), u = new Float64Array(n), vv = new Float64Array(n);
    let rate = 0, div = 0, ref = 0;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (!H[P]) continue;
      const fw = qx[j * (nx + 1) + i], fe = qx[j * (nx + 1) + i + 1], fs = qy[P], fn = qy[P + nx];
      u[P] = (0.5 * (fw + fe)) / (H[P] * dy); vv[P] = (0.5 * (fs + fn)) / (H[P] * dx);
      rate = Math.max(rate, (0.5 * (Math.abs(fw) + Math.abs(fe) + Math.abs(fs) + Math.abs(fn))) / (H[P] * dx * dy));
      div = Math.max(div, Math.abs(fe - fw + fn - fs)); ref = Math.max(ref, Math.abs(fw), Math.abs(fe), Math.abs(fs), Math.abs(fn));
    }
    return { qx, qy, u, v: vv, rate, div: ref ? div / ref : 0 };
  });
  return { basis, local };
}

/**
 * Far-field transport of excess salinity C (g/kg) in a bottom-attached layer occupying a fixed fraction φ of
 * the local depth (φ = 1: fully mixed): ∂(φHC)/∂t + ∇·(f_b φ q C) + ∇·(φ H u_g C) = ∇·(φ H K ∇C) + source.
 * Explicit finite volumes, upwind or van Leer TVD fluxes, CFL-limited step, open or closed boundaries.
 */
export async function farField(c, ctx) {
  const { g, flow, cur, K, phi = 1, bedF = 1, scheme = 'tvd', cfl = 0.5, tEnd, tStat = 0, src = [], rate = 0, probes = [], ring = [], drift = null, closed = false, thr = 0.1, particles = 0 } = c;
  const { nx, ny, dx, dy, H } = g, n = nx * ny, A = dx * dy, [bA, bB] = flow.basis;
  const C = c.C0 ? Float64Array.from(c.C0) : new Float64Array(n), dC = new Float64Array(n), Cmax = new Float64Array(n), Csum = new Float64Array(n), Csnap = new Float64Array(n);
  let Kmax = 0;
  for (let P = 0; P < n; P++) if (H[P] && K[P] > Kmax) Kmax = K[P];
  const dtDiff = Kmax > 0 ? 0.2 / (Kmax * (1 / (dx * dx) + 1 / (dy * dy))) : Infinity, tvd = scheme === 'tvd';
  // density-driven down-slope drift: u_g = √(g'·h_layer·|s| / 2C_d) directed down the bed gradient, g' = g β_S C
  const slopeAt = (P, i, j) => [i > 0 && i < nx - 1 && H[P - 1] && H[P + 1] ? (g.zb[P + 1] - g.zb[P - 1]) / (2 * dx) : 0, j > 0 && j < ny - 1 && H[P - nx] && H[P + nx] ? (g.zb[P + nx] - g.zb[P - nx]) / (2 * dy) : 0];
  const gCoef = (s, sOther, hf) => { const sm = Math.hypot(s, sOther); return drift && sm > 1e-6 ? (-s / Math.sqrt(sm)) * Math.sqrt((G * drift.betaS * phi * hf) / (2 * drift.Cd)) : 0; };
  const gRate = drift ? drift.vmax * (1 / dx + 1 / dy) : 0, vmaxG = drift ? drift.vmax : 0;
  // face tables: interior water–water faces (axis 0 = x, 1 = y) and open-boundary faces
  const tab = [0, 1].map((ax) => {
    const L = [], U1 = [], U2 = [], Fq = [], Dc = [], Gc = [], Ga = [], stride = ax ? nx : 1, len = ax ? dx : dy, dist = ax ? dy : dx;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (!H[P] || (ax ? j === ny - 1 : i === nx - 1) || !H[P + stride]) continue;
      const R = P + stride, hf = Math.min(H[P], H[R]), [sxP, syP] = slopeAt(P, i, j), [sxR, syR] = slopeAt(R, ax ? i : i + 1, ax ? j + 1 : j);
      L.push(P); Fq.push(ax ? (j + 1) * nx + i : j * (nx + 1) + i + 1);
      U1.push((ax ? j > 0 : i > 0) && H[P - stride] ? P - stride : -1); U2.push((ax ? j < ny - 2 : i < nx - 2) && H[R + stride] ? R + stride : -1);
      Dc.push((0.5 * (K[P] + K[R]) * hf * len) / dist); Ga.push(hf * len);
      Gc.push(gCoef((g.zb[R] - g.zb[P]) / dist, ax ? 0.5 * (sxP + sxR) : 0.5 * (syP + syR), hf));
    }
    return { m: L.length, stride, L: Int32Array.from(L), U1: Int32Array.from(U1), U2: Int32Array.from(U2), Fq: Int32Array.from(Fq), Dc: Float64Array.from(Dc), Gc: Float64Array.from(Gc), Ga: Float64Array.from(Ga), qa: ax ? bA.qy : bA.qx, qb: ax ? bB.qy : bB.qx };
  });
  const edge = []; // [cell, face index, axis, outward sign]
  if (!closed) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const P = j * nx + i;
    if (!H[P]) continue;
    if (i === 0) edge.push([P, j * (nx + 1), 0, -1]); if (i === nx - 1) edge.push([P, j * (nx + 1) + nx, 0, 1]);
    if (j === 0) edge.push([P, i, 1, -1]); if (j === ny - 1) edge.push([P, ny * nx + i, 1, 1]);
  }
  const inv = Float64Array.from(H, (h) => (h > 0 ? 1 / (h * A) : 0));
  const sample = (x, y) => { // bilinear over wet cells
    const fi = (x - g.x0) / dx - 0.5, fj = (y - g.y0) / dy - 0.5, i = clamp(Math.floor(fi), 0, nx - 2), j = clamp(Math.floor(fj), 0, ny - 2), a = clamp(fi - i, 0, 1), b = clamp(fj - j, 0, 1);
    let s = 0, w = 0;
    for (const [P, wt] of [[j * nx + i, (1 - a) * (1 - b)], [j * nx + i + 1, a * (1 - b)], [(j + 1) * nx + i, (1 - a) * b], [(j + 1) * nx + i + 1, a * b]]) if (H[P]) { s += wt * C[P]; w += wt; }
    return w > 0 ? s / w : 0;
  };
  // particles (random walk): continuous release at the source
  const np = Math.round(particles), px = new Float64Array(np), py = new Float64Array(np), alive = new Uint8Array(np), rn = np ? rng(99) : null;
  let released = 0;
  const ser = { t: [], u: [], v: [], probes: probes.map(() => []), ring: [] }, bal = { injected: 0, out: 0 };
  let t = 0, step = 0, nStat = 0, tSnap = 0, areaSnap = -1, lastSample = -Infinity, ringMax = 0, tRingMax = 0;
  const dtSample = tEnd / 360, maxSteps = c.maxSteps ?? 60000;
  while (t < tEnd - 1e-9 && step < maxSteps) {
    const [ux, uy] = currentAt(cur, t), adv = bedF * (Math.abs(ux) * bA.rate + Math.abs(uy) * bB.rate) + gRate;
    const dt = Math.min(adv > 0 ? cfl / adv : Infinity, dtDiff, tEnd - t, tEnd / 40);
    dC.fill(0);
    const cA = bedF * ux, cB = bedF * uy;
    for (const T of tab) {
      const { m, stride, L, U1, U2, Fq, Dc, Gc, Ga, qa, qb } = T;
      for (let k = 0; k < m; k++) {
        const P = L[k], R = P + stride, cl = C[P], cr = C[R], f = cA * qa[Fq[k]] + cB * qb[Fq[k]];
        let cf;
        if (f >= 0) { cf = cl; if (tvd && U1[k] >= 0) { const d1 = cl - C[U1[k]], d2 = cr - cl; if (d1 * d2 > 0) cf = cl + (d1 * d2) / (d1 + d2); } } // van Leer limiter
        else { cf = cr; if (tvd && U2[k] >= 0) { const d1 = cr - C[U2[k]], d2 = cl - cr; if (d1 * d2 > 0) cf = cr + (d1 * d2) / (d1 + d2); } }
        let flux = f * cf - Dc[k] * (cr - cl);
        if (Gc[k] !== 0) { const cu = Gc[k] > 0 ? cl : cr; if (cu > 0) { const ug = Gc[k] * Math.sqrt(cu); flux += (ug > vmaxG ? vmaxG : ug < -vmaxG ? -vmaxG : ug) * Ga[k] * cu; } }
        dC[P] -= flux; dC[R] += flux;
      }
    }
    for (const [P, q, ax, sg] of edge) { // open edge: outflow leaves the domain, inflow brings clean water
      const fo = sg * (ax ? cA * bA.qy[q] + cB * bB.qy[q] : cA * bA.qx[q] + cB * bB.qx[q]);
      if (fo > 0) { dC[P] -= fo * C[P]; bal.out += fo * C[P] * dt * phi; }
    }
    for (let P = 0; P < n; P++) { const cn = C[P] + dt * dC[P] * inv[P]; C[P] = cn > 0 ? cn : 0; }
    for (const s of src) C[s.P] += (dt * rate * s.w) / (phi * H[s.P] * A);
    bal.injected += dt * rate * (src.length ? 1 : 0);
    t += dt; step++;
    if (np) {
      const want = Math.min(np, Math.floor((t / tEnd) * np) + 1);
      while (released < want) { px[released] = c.srcXY[0] + c.srcR * (rn.uniform() - 0.5); py[released] = c.srcXY[1] + c.srcR * (rn.uniform() - 0.5); alive[released++] = 1; }
      for (let k = 0; k < released; k++) {
        if (!alive[k]) continue;
        const i = Math.floor((px[k] - g.x0) / dx), j = Math.floor((py[k] - g.y0) / dy), P = j * nx + i, sd = Math.sqrt(6 * K[P] * dt);
        const xn = px[k] + bedF * (ux * bA.u[P] + uy * bB.u[P]) * dt + sd * (2 * rn.uniform() - 1), yn = py[k] + bedF * (ux * bA.v[P] + uy * bB.v[P]) * dt + sd * (2 * rn.uniform() - 1);
        const i2 = Math.floor((xn - g.x0) / dx), j2 = Math.floor((yn - g.y0) / dy);
        if (i2 < 0 || i2 >= nx || j2 < 0 || j2 >= ny) { alive[k] = 0; continue; }
        if (H[j2 * nx + i2]) { px[k] = xn; py[k] = yn; }
      }
    }
    const inStat = t >= tStat;
    if (inStat) { nStat++; for (let P = 0; P < n; P++) { const cv = C[P]; if (cv > Cmax[P]) Cmax[P] = cv; Csum[P] += cv; } }
    if (t - lastSample >= dtSample || t >= tEnd - 1e-9) {
      lastSample = t;
      let rm = 0;
      for (const [x, y] of ring) rm = Math.max(rm, sample(x, y));
      ser.t.push(t / 3600); ser.u.push(ux); ser.v.push(uy); ser.ring.push(rm); probes.forEach((p, k) => ser.probes[k].push(sample(p[0], p[1])));
      if (inStat) {
        if (rm > ringMax) { ringMax = rm; tRingMax = t; }
        let area = 0;
        for (let P = 0; P < n; P++) if (C[P] > thr) area++;
        if (area > areaSnap) { areaSnap = area; tSnap = t; Csnap.set(C); }
      }
      if (ctx?.progress) ctx.progress(0.12 + (0.83 * t) / tEnd, `Far field: ${fmt(t / 3600, 3)} h of ${fmt(tEnd / 3600, 3)} h`);
      if (ctx?.tick) await ctx.tick();
    }
    if (!Number.isFinite(C[g.jo * nx + g.io])) throw new Error('The far-field solution became unstable — lower the CFL number.');
  }
  let mass = 0;
  for (let P = 0; P < n; P++) { mass += phi * H[P] * C[P] * A; Csum[P] = nStat ? Csum[P] / nStat : C[P]; if (!nStat) Cmax[P] = C[P]; }
  if (areaSnap < 0) Csnap.set(C);
  const part = np ? { x: [], y: [] } : null;
  if (part) for (let k = 0; k < released; k++) if (alive[k]) { part.x.push(px[k]); part.y.push(py[k]); }
  return { C, Cmax, Cmean: Csum, Csnap, tSnap, ser, bal: { ...bal, mass }, steps: step, tEnd: t, ringMax, tRingMax, dtMean: step ? t / step : 0, part, complete: t >= tEnd - 1e-6 };
}

/** Marching-squares iso-line of a row-major field (rows[j][i]), returned as plot shapes. */
function isolines(z, xs, ys, level, color, maxShapes = 60) {
  const nx = xs.length, ny = ys.length, segs = [];
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const q = [z[j][i], z[j][i + 1], z[j + 1][i + 1], z[j + 1][i]];
    if (!q.every(Number.isFinite)) continue;
    const pts = [], ed = (a, b, x1, y1, x2, y2) => { if ((a < level) !== (b < level)) { const t = (level - a) / (b - a); pts.push([x1 + t * (x2 - x1), y1 + t * (y2 - y1)]); } };
    ed(q[0], q[1], xs[i], ys[j], xs[i + 1], ys[j]); ed(q[1], q[2], xs[i + 1], ys[j], xs[i + 1], ys[j + 1]); ed(q[2], q[3], xs[i + 1], ys[j + 1], xs[i], ys[j + 1]); ed(q[3], q[0], xs[i], ys[j + 1], xs[i], ys[j]);
    for (let k = 0; k + 1 < pts.length; k += 2) segs.push([pts[k], pts[k + 1]]);
  }
  // chain segments that share end points into polylines
  const key = (p) => `${Math.round(p[0] * 100)},${Math.round(p[1] * 100)}`, ends = new Map();
  segs.forEach((s, k) => { for (const p of s) { const q = key(p); if (!ends.has(q)) ends.set(q, []); ends.get(q).push(k); } });
  const used = new Uint8Array(segs.length), lines = [];
  for (let k = 0; k < segs.length; k++) {
    if (used[k]) continue;
    used[k] = 1;
    const line = [segs[k][0], segs[k][1]];
    for (const dir of [1, 0]) for (;;) {
      const tip = dir ? line[line.length - 1] : line[0], nb = (ends.get(key(tip)) || []).find((m) => !used[m]);
      if (nb === undefined) break;
      used[nb] = 1;
      const other = key(segs[nb][0]) === key(tip) ? segs[nb][1] : segs[nb][0];
      if (dir) line.push(other); else line.unshift(other);
    }
    lines.push(line);
  }
  return lines.sort((a, b) => b.length - a.length).slice(0, maxShapes).map((l) => ({ x: l.map((p) => p[0]), y: l.map((p) => p[1]), closed: false, color, width: 1 }));
}

// ---------------------------------------------------------------------------------------------------
// Case set-up shared by the full run and the fast calibration model
// ---------------------------------------------------------------------------------------------------
/** Diffuser sizing from the 60° dense-jet coefficients: exit velocity window, Froude number, surface clearance, dilution. */
export function designDiffuser({ Q, gp, depth, z0, dS0, limit, vMin = 4, vMax = 6, Fmin = 20, clear = 0.8, nMax = 80 }) {
  const rows = [], dMin = 0.05;
  const rate = (n, d) => {
    const V = (4 * Q) / (n * Math.PI * d * d), F = V / Math.sqrt(Math.max(gp, 1e-9) * d), zt = ROBERTS60.zt * d * F, Si = ROBERTS60.Si * F, Sn = ROBERTS60.Sn * F, s = Math.max(1, Math.ceil(2 * d * F * 2) / 2), fails = [];
    if (V < vMin) fails.push('velocity low'); if (V > vMax) fails.push('velocity high'); if (F < Fmin) fails.push('Froude < ' + Fmin);
    if (zt + z0 > clear * depth) fails.push('jet too close to the surface'); if (dS0 / Sn > limit) fails.push('dilution short');
    const pen = Math.max(0, vMin / V - 1) + Math.max(0, V / vMax - 1) + Math.max(0, Fmin / F - 1) + 2 * Math.max(0, (zt + z0) / (clear * depth) - 1) + Math.max(0, dS0 / Sn / limit - 1);
    return { n, d, V, F, zt, Si, Sn, s, len: (n - 1) * s, ok: !fails.length, why: fails.join(', ') || 'meets all criteria', pen };
  };
  for (let n = 1; n <= nMax; n++) { // per port count: the largest 5 mm diameter step inside the velocity window that passes, else the least-violating one
    const hi = Math.max(dMin, Math.floor(Math.sqrt((4 * Q) / (n * Math.PI * vMin)) / 0.005) * 0.005), lo = Math.max(dMin, Math.ceil(Math.sqrt((4 * Q) / (n * Math.PI * vMax)) / 0.005) * 0.005);
    let pick = null;
    for (let d = hi; d >= Math.min(lo, hi) - 1e-9; d -= 0.005) { const r = rate(n, +d.toFixed(3)); if (!pick || (r.ok && !pick.ok) || (r.ok === pick.ok && r.pen < pick.pen - 1e-9)) pick = r; if (r.ok) break; }
    rows.push(pick);
    if (hi <= dMin && n > 3) break;
  }
  const best = rows.find((r) => r.ok) || rows.reduce((a, b) => (b.pen < a.pen - 1e-9 ? b : a), rows[0]);
  return { rows, best };
}

const waveOrbital = (Hs, T, h) => { // near-bed orbital velocity amplitude, linear wave theory
  if (!(Hs > 0) || !(T > 0) || !(h > 0)) return 0;
  const om = (2 * Math.PI) / T;
  let k = (om * om) / G;
  for (let i = 0; i < 30; i++) k = (om * om) / (G * Math.tanh(k * h));
  return (Math.PI * Hs) / (T * Math.sinh(Math.min(k * h, 30)));
};

function currents(v) {
  const cons = [['M2', v.uM2, v.phM2], ['S2', v.uS2, v.phS2], ['K1', v.uK1, v.phK1], ['O1', v.uO1, v.phO1]].filter((c) => c[1] > 0).map(([id, amp, ph]) => ({ id, amp, ph: ph * D2R, T: TIDES[id] }));
  const [rx, ry] = bearing(v.resDir), [wx, wy] = bearing(v.windDir + 180), wf = (v.windFactor / 100) * v.windSpeed;
  const c = { cons, axis: bearing(v.tideDir), res: [v.uRes * rx, v.uRes * ry], wind: [wf * wx, wf * wy], series: null };
  const rows = v.useSeries && Array.isArray(v.curSeries) ? v.curSeries.filter((r) => [r.t, r.speed, r.dir].every(Number.isFinite)).sort((a, b) => a.t - b.t) : [];
  if (rows.length >= 3 && rows[rows.length - 1].t > rows[0].t) c.series = { t: rows.map((r) => r.t), u: rows.map((r) => r.speed * bearing(r.dir)[0]), v: rows.map((r) => r.speed * bearing(r.dir)[1]), span: rows[rows.length - 1].t - rows[0].t };
  const sp = linspace(0, 2 * TIDES.O1 * 3600, 240).map((t) => { const [a, b] = currentAt(c, t); return Math.hypot(a, b); });
  c.mean = mean(sp); c.peak = Math.max(...sp); c.min = Math.min(...sp); c.rms = Math.sqrt(mean(sp.map((x) => x * x)));
  return c;
}

/** Discharge, ambient and diffuser definition; near-field solution for a given ambient current. */
function prep(v, depthOut = v.depth) {
  const depth = Math.max(depthOut, 0.5), Q = Math.max(v.Qb, 1e-6) / 3600, z0 = clamp(v.z0, 0, 0.5 * depth);
  const amb = (ua) => (z) => ({ S: v.Sa + v.dS * (0.5 - z / depth), T: v.Ta + v.dT * (z / depth - 0.5), ua });
  const a0 = amb(0)(z0), rhoA = density(a0.T, a0.S), rhoB = density(v.Tb, v.Sb), gp = (G * (rhoB - rhoA)) / rhoA, dS0 = v.Sb - a0.S;
  const limit = Math.max(1e-6, Math.min(v.limAbs, (v.limRel / 100) * v.Sa));
  const des = gp > 0 ? designDiffuser({ Q, gp, depth: depth - 0.5 * v.tideRange, z0, dS0: Math.abs(dS0), limit, vMin: v.vMin, vMax: v.vMax, Fmin: v.Fmin, clear: v.clear / 100 }) : null;
  const auto = v.design === 'auto' && des;
  const n = auto ? des.best.n : Math.max(1, Math.round(v.nPorts)), d = auto ? des.best.d : v.dPort / 1000, spacing = n > 1 ? (auto ? des.best.s : Math.max(v.spacing, d)) : Infinity, theta = (auto ? 60 : clamp(v.theta, 0, 90)) * D2R;
  const U0 = Q / n / ((Math.PI * d * d) / 4), Ldiff = n > 1 ? (n - 1) * spacing : 0;
  const jet = (ua = 0, sigma = 0, bedSlope = 0, tol = v.jetTol) => denseJet({ d, U0, theta, Sb: v.Sb, Tb: v.Tb, amb: amb(ua), z0, alphaJ: v.alphaJ, alphaP: Math.max(v.alphaP, v.alphaJ), betaX: v.betaX, descF: v.descF, spacing, sigma, depth, bedSlope, tol: clamp(tol, 1e-9, 1e-3) });
  return { depth, Q, z0, amb, rhoA, rhoB, gp, dS0, limit, des, auto: !!auto, n, d, spacing, theta, U0, Ldiff, jet };
}

/** Near-field end (bottom-layer transition) from the impact values using the empirical ratios of Roberts et al. */
function nearFieldEnd(j) {
  if (j.fate !== 'seabed') return { xn: j.xi, Sn: j.Si, yL: Math.max(2 * j.bi, 0.2) };
  return { xn: (j.xi * ROBERTS60.xn) / ROBERTS60.xi, Sn: (j.Si * ROBERTS60.Sn) / ROBERTS60.Si, yL: Math.max(0.2, (j.zt * ROBERTS60.yL) / ROBERTS60.zt) };
}

const F = (key, label, unit, value, min, max, help, extra = {}) => ({ key, label, unit, value, min, max, help, ...extra });
const SEL = (key, label, value, options, help, extra = {}) => ({ key, label, type: 'select', value, options: options.map(([v, l]) => ({ value: v, label: l })), help, ...extra });
const manual = (v) => v.design === 'manual';

const suite = {
  id: 'sea', num: 5, title: 'Brine Discharge into the Sea', short: 'Sea discharge', icon: '🌊',
  tagline: 'Dense-jet near field, bottom density current and tidal far-field dispersion of brine, with mixing-zone, receptor and intake assessment.',
  description: 'The near field is solved with an integral model of inclined negatively buoyant jets (volume, momentum, salt and heat conservation with an entrainment closure for jet, plume and cross-flow regimes) for single-port and multiport diffusers, and is cross-checked against the empirical dense-jet coefficients. The diluted brine then spreads as an entraining bottom density current and is carried by tidal, residual and wind-driven currents in a transient far-field advection–dispersion model over the site or a synthetic bathymetry. Results are tested against mixing-zone limits, sensitive receptors and recirculation to the intake.',
  guide: [
    'Enter the brine flow, salinity and temperature (or pull the concentrate from the RO or ZLD suites) and the ambient seawater.',
    'Let the tool size the diffuser (ports, diameter, 60° angle) or enter your own design.',
    'Describe the currents with tidal constituents, a residual current and wind drift, or apply the Global Site Data (bathymetry, currents, tide, wind, waves).',
    'Place the intake and the sensitive receptors relative to the outfall and set the mixing-zone radius and limits.',
    'Run. Read the near-field dilution first, then the far-field maps, the receptor time series and the compliance table.',
  ],
  implemented: ['continuity equation', 'momentum equation', 'boussinesq', 'hydrostatic-pressure', 'salinity advection-diffusion', 'temperature transport', 'scalar transport', 'equation of state', 'buoyancy equation', 'jet-integral', 'buoyant-plume', 'entrainment', 'densimetric-froude', 'gaussian plume', 'tidal-harmonic',
    'near-field/far-field', 'integral-plume-hydrodynamic', 'salinity-temperature-density', 'hydrodynamic-water-quality', 'hydrodynamic-particle-tracking', 'eulerian-lagrangian',
    'initial currents field', 'salinity', 'temperature', 'density stratification', 'tracer concentration', 'prescribed current/velocity', 'discharge-flow', 'brine salinity/temperature source', 'seabed no-normal-flow and friction', 'zero-gradient/outflow',
    'outfall and diffuser', 'near-field jet', 'buoyant-plume modelling', 'far-field hydrodynamics', 'salinity transport', 'temperature transport', 'density-driven', 'ocean-current', 'tidal modelling', 'wave effects', 'bathymetry', 'coastal-boundary', 'turbulent mixing', 'stratification', 'particle and contaminant transport', 'seabed interaction', 'plume dilution', 'recirculation towards desalination intakes', 'environmental-threshold', 'ecological exposure'],
  equationsNote: 'Scope and limits. Near field: steady integral jet model with top-hat profiles in the Boussinesq approximation; the default entrainment coefficients (jet 0.07, plume 0.117, descending-limb enhancement 2.0) are calibrated to the 60° still-water experiments of Roberts, Ferrier & Daviero (1997) and the bottom-layer transition uses their empirical ratios, so angles far from 45–65°, strongly merged jets and shallow water where the jet reaches the surface carry more uncertainty. Intermediate field: one-dimensional Ellison–Turner gravity current. Far field: two-dimensional transport of excess salinity in a bottom-attached layer that occupies a fixed fraction of the local depth (or the fully mixed water column), with an optional density-driven down-slope drift. Hydrodynamics are not a free-surface solution: the current pattern is a quasi-steady rigid-lid, friction-dominated shallow-water balance ∇·(H^5/3 ∇η) = 0 scaled in time by tidal harmonics (M2, S2, K1, O1), a residual current and wind drift, so tidal propagation, eddies shed by headlands, Coriolis effects and wetting–drying are not resolved. Wave effects enter only as enhanced near-bed mixing; temperature is carried in the near field only. A full three-dimensional hydrostatic or non-hydrostatic ocean model with turbulence closure, wave-action modelling, Flather boundaries, atmospheric heat flux and seasonal simulations are listed for reference and are not solved here.',

  inputs: [
    { group: 'Brine discharge', help: 'What leaves the plant. Pull the concentrate of the RO suite or the liquid discharge of the ZLD suite.', fields: [
      F('Qb', 'Brine flow', 'm³/h', 2000, 1, 5e5, 'Total discharge through the diffuser.'),
      F('Sb', 'Brine salinity', 'g/kg', 65, 0, 280, 'Seawater RO at 45 % recovery gives about 1.8 × ambient.'),
      F('Tb', 'Brine temperature', '°C', 24, 0, 60, 'RO concentrate is 1–2 °C above the intake; thermal plants discharge 5–10 °C warmer.'),
      F('cAnti', 'Antiscalant in brine', 'mg/L', 5.5, 0, 200, 'Dose × concentration factor.'),
      F('cCl', 'Residual chlorine / oxidant in brine', 'mg/L', 0.05, 0, 10, 'After dechlorination, typically below 0.1 mg/L.'),
    ] },
    { group: 'Receiving water', help: 'Ambient seawater at the outfall. Values are depth means; the stratification terms describe the difference between seabed and surface.', fields: [
      F('Sa', 'Ambient salinity', 'g/kg', 37, 0, 60, ''),
      F('Ta', 'Ambient temperature', '°C', 22, -2, 40, ''),
      F('depth', 'Water depth at the outfall', 'm', 12, 2, 200, 'Mean sea level. With imported bathymetry the depth is read from the grid.'),
      F('dS', 'Salinity stratification (seabed − surface)', 'g/kg', 0, -5, 10, 'Positive = saltier at the bed (stable).'),
      F('dT', 'Thermal stratification (surface − seabed)', '°C', 0, -5, 15, 'Positive = warmer at the surface (stable).'),
    ] },
    { group: 'Diffuser', help: 'Ports inclined upward from the seabed. 60° gives the longest trajectory and the highest dilution for dense jets.', fields: [
      SEL('design', 'Diffuser definition', 'auto', [['auto', 'Size the diffuser for me'], ['manual', 'I will enter ports and diameter']], 'Auto-design selects the fewest ports that satisfy the velocity, Froude-number, surface-clearance and dilution criteria.'),
      F('nPorts', 'Number of ports', '', 6, 1, 200, '', { showIf: manual, step: 1 }),
      F('dPort', 'Port diameter', 'mm', 155, 20, 2000, '', { showIf: manual }),
      F('theta', 'Port angle above horizontal', '°', 60, 0, 90, '', { showIf: manual }),
      F('spacing', 'Port spacing', 'm', 8, 0.2, 200, 'Neighbouring jets merge when their width exceeds the spacing.', { showIf: manual }),
      F('z0', 'Port height above the seabed', 'm', 1, 0, 10, 'Risers keep the jets clear of the bed and of sediment.'),
      F('jetDir', 'Discharge direction (bearing)', '°', 0, 0, 360, 'Compass bearing toward which the ports point; normally offshore. The synthetic beach faces north.'),
    ] },
    { group: 'Currents, tide, wind and waves', help: 'Tidal currents are rectilinear along the tidal axis. Phases are relative to the start of the simulation.', fields: [
      F('tideDir', 'Flood-current direction (bearing)', '°', 90, 0, 360, 'Direction toward which the flood tide flows; normally parallel to the coast.'),
      F('uM2', 'M2 current amplitude', 'm/s', 0.25, 0, 3, 'Principal lunar semi-diurnal constituent (12.42 h).'),
      F('phM2', 'M2 phase', '°', 0, -360, 360, ''),
      F('uS2', 'S2 current amplitude', 'm/s', 0.08, 0, 3, 'Principal solar semi-diurnal constituent (12.00 h); produces the spring–neap cycle.'),
      F('phS2', 'S2 phase', '°', 30, -360, 360, ''),
      F('uK1', 'K1 current amplitude', 'm/s', 0.04, 0, 3, 'Luni-solar diurnal constituent (23.93 h).'),
      F('phK1', 'K1 phase', '°', 60, -360, 360, ''),
      F('uO1', 'O1 current amplitude', 'm/s', 0.03, 0, 3, 'Lunar diurnal constituent (25.82 h).'),
      F('phO1', 'O1 phase', '°', 120, -360, 360, ''),
      F('uRes', 'Residual (net) current', 'm/s', 0.03, 0, 2, 'Tide-averaged drift.'),
      F('resDir', 'Residual-current direction (bearing)', '°', 90, 0, 360, ''),
      F('tideRange', 'Tidal range', 'm', 1.5, 0, 15, 'Used for the low-water surface-clearance check of the jets.'),
      F('windSpeed', 'Wind speed', 'm/s', 5, 0, 40, ''),
      F('windDir', 'Wind direction (from, bearing)', '°', 315, 0, 360, 'Meteorological convention: direction the wind blows from.'),
      F('windFactor', 'Wind-drift factor', '% of wind speed', 0.5, 0, 4, 'Wind-driven current of the transported layer: about 3 % at the surface, 1 % for the depth mean and less near the bed.'),
      F('waveHeight', 'Significant wave height', 'm', 0.8, 0, 12, 'Wave orbital motion enhances near-bed mixing in the Elder dispersion option.'),
      F('wavePeriod', 'Wave period', 's', 6, 2, 25, ''),
      { key: 'useSeries', label: 'Drive the far field with the current time series below', type: 'bool', value: false, help: 'Replaces the tidal constituents and the residual current with measured or site currents (repeated periodically).' },
      { key: 'curSeries', label: 'Current time series', type: 'table', showIf: (v) => v.useSeries, columns: [{ key: 't', label: 'Time', unit: 'h' }, { key: 'speed', label: 'Speed', unit: 'm/s' }, { key: 'dir', label: 'Direction (toward)', unit: '°' }],
        value: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24].map((t) => { const u = 0.28 * Math.cos((2 * Math.PI * t) / 12.42) + 0.04; return { t, speed: +Math.abs(u).toFixed(3), dir: u >= 0 ? 90 : 270 }; }) },
    ] },
    { group: 'Site layout', help: 'Positions are metres east (x) and north (y) of the outfall. Without imported bathymetry a plane beach with the shoreline to the south is used.', fields: [
      { key: 'bathy', label: 'Bathymetry (site grid or x, y, z table)', type: 'file', kind: 'table', value: null, buttonLabel: 'Import bathymetry table', accept: '.csv,.tsv,.txt,.json,.xlsx', parse: (t, name) => bathyFromTable(t, name), help: 'Columns x, y, z in metres or lon, lat, elevation (negative below sea level; a column called depth is taken positive down). The Global Site Data bathymetry is offered automatically.' },
      F('outX', 'Outfall offset east of the bathymetry centre', 'm', 0, -1e5, 1e5, '', { showIf: (v) => !!v.bathy }),
      F('outY', 'Outfall offset north of the bathymetry centre', 'm', 0, -1e5, 1e5, '', { showIf: (v) => !!v.bathy }),
      F('slope', 'Seabed slope (synthetic beach)', '%', 1.5, 0, 20, 'Offshore deepening per 100 m.', { showIf: (v) => !v.bathy }),
      F('bayAmp', 'Shoreline undulation (synthetic beach)', 'm', 120, 0, 2000, 'Amplitude of a gentle bay/headland shape along the coast; 0 = straight coast.', { showIf: (v) => !v.bathy }),
      F('inX', 'Intake position east of the outfall', 'm', -900, -1e5, 1e5, 'Position of the seawater intake head.'),
      F('inY', 'Intake position north of the outfall', 'm', -350, -1e5, 1e5, ''),
      { key: 'receptors', label: 'Sensitive receptors', type: 'table', columns: [{ key: 'name', label: 'Receptor', type: 'text' }, { key: 'x', label: 'East of outfall', unit: 'm' }, { key: 'y', label: 'North of outfall', unit: 'm' }, { key: 'thr', label: 'Threshold ΔS', unit: 'g/kg' }],
        value: [{ name: 'Seagrass meadow', x: 600, y: -300, thr: 0.5 }, { name: 'Reef patch', x: -450, y: 250, thr: 0.3 }, { name: 'Bathing beach', x: 300, y: -500, thr: 1 }], help: 'Each receptor is tested against its own excess-salinity threshold over the tidal cycle.' },
    ] },
    { group: 'Near-field model', tab: 'setup', help: 'Entrainment closure of the integral jet model. Defaults reproduce the 60° dense-jet experiments.', fields: [
      F('alphaJ', 'Jet entrainment coefficient α_j', '–', 0.07, 0.03, 0.15, 'Top-hat value for the momentum-dominated jet.'),
      F('alphaP', 'Plume entrainment coefficient α_p', '–', 0.117, 0.05, 0.25, 'Upper bound reached when buoyancy assists the flow.'),
      F('descF', 'Descending-limb mixing enhancement', '×', 2, 1, 4, 'Additional mixing of the falling dense plume by gravitational instability of its lower edge.'),
      F('betaX', 'Cross-flow entrainment coefficient', '–', 0.5, 0, 1.5, 'Entrainment of ambient flow normal to the jet axis.'),
    ] },
    { group: 'Far-field model', tab: 'setup', help: 'How the diluted brine is carried away from the near field.', fields: [
      SEL('layer', 'Vertical distribution', 'layer', [['layer', 'Bottom-attached layer (dense plume)'], ['mixed', 'Fully mixed water column']], 'The layer occupies a fixed fraction of the local depth, set from the near-field layer thickness.'),
      F('hLayer', 'Layer thickness at the outfall (0 = from the near field)', 'm', 0, 0, 100, '', { showIf: (v) => v.layer === 'layer' }),
      F('bedF', 'Near-bed velocity ÷ depth-mean velocity', '–', 0.7, 0.2, 1, 'Logarithmic profile: the bottom layer moves more slowly than the depth mean.', { showIf: (v) => v.layer === 'layer' }),
      { key: 'drift', label: 'Density-driven down-slope drift', type: 'bool', value: true, help: 'Adds the gravity-current velocity √(g′h·slope / 2C_d) down the local bed gradient.', showIf: (v) => v.layer === 'layer' },
      SEL('disp', 'Horizontal dispersion', 'elder', [['elder', 'Elder: K = 0.6 u* h (tide + waves)'], ['okubo', 'Okubo: scale-dependent 4/3-type law'], ['const', 'Constant coefficient']], ''),
      F('K0', 'Dispersion coefficient / background value', 'm²/s', 0.3, 0.001, 200, 'Constant value, or the floor added to the Elder coefficient.'),
      F('Kmult', 'Dispersion multiplier', '×', 1, 0.05, 20, 'Calibration factor on the dispersion coefficient.'),
      F('Cd', 'Seabed drag coefficient', '–', 0.0025, 0.0005, 0.02, 'Sets the friction velocity and the gravity-current drag.'),
      SEL('scheme', 'Advection scheme', 'tvd', [['tvd', 'Second-order TVD (van Leer)'], ['upwind', 'First-order upwind']], 'Upwind adds numerical diffusion of order u·Δx/2.'),
      { key: 'particles', label: 'Lagrangian random-walk particle tracker', type: 'bool', value: false, help: 'Releases particles continuously at the near-field end to show the plume envelope.' },
      F('nPart', 'Number of particles', '', 1500, 100, 10000, '', { showIf: (v) => v.particles, step: 1 }),
    ] },
    { group: 'Environmental limits', tab: 'setup', help: 'Regulatory mixing zone and water-quality criteria.', fields: [
      F('mzR', 'Mixing-zone radius', 'm', 100, 10, 5000, 'Compliance is assessed at this distance from the diffuser.'),
      F('limAbs', 'Excess-salinity limit (absolute)', 'g/kg', 2, 0.05, 20, 'For example ΔS ≤ 2 g/kg at the mixing-zone edge.'),
      F('limRel', 'Excess-salinity limit (relative)', '% of ambient', 5, 0.5, 50, 'The stricter of the absolute and relative limits applies.'),
      F('thrArea', 'Reporting threshold for exposed area', 'g/kg', 0.1, 0.01, 10, 'Seabed area and volume are reported above this value and above the limit.'),
      F('limAnti', 'Antiscalant criterion', 'mg/L', 0.2, 0.001, 50, 'Predicted no-effect concentration.'),
      F('limCl', 'Chlorine / oxidant criterion', 'mg/L', 0.0075, 0.0001, 5, 'For example 7.5 µg/L chlorine-produced oxidants.'),
    ] },
    { group: 'Diffuser design criteria', tab: 'setup', help: 'Used by the auto-design and reported for a manual design.', fields: [
      F('vMin', 'Minimum port velocity', 'm/s', 4, 0.5, 10, ''), F('vMax', 'Maximum port velocity', 'm/s', 6, 1, 15, ''),
      F('Fmin', 'Minimum densimetric Froude number', '–', 20, 5, 100, ''), F('clear', 'Jet top below this share of the low-water depth', '%', 80, 30, 100, ''),
    ] },
    { group: 'Far-field grid', tab: 'mesh', help: 'Cartesian finite-volume grid. The explicit scheme limits the time step through the CFL number.', fields: [
      F('nx', 'Cells east–west, nx', '', 80, 16, 220, '', { step: 1 }), F('ny', 'Cells north–south, ny', '', 60, 12, 160, '', { step: 1 }),
      F('Lx', 'Domain length east–west', 'm', 4000, 200, 2e5, ''), F('Ly', 'Domain length north–south', 'm', 3000, 200, 2e5, ''),
      F('fx', 'Outfall position, share of the domain from the west edge', '%', 50, 5, 95, ''), F('fy', 'Outfall position, share of the domain from the south edge', '%', 40, 5, 95, ''),
      F('cfl', 'CFL number', '–', 0.5, 0.05, 0.9, 'Advective time-step limit; 0.5 or less for the TVD scheme.'),
      F('nCycles', 'Simulated M2 tidal cycles', '', 3, 1, 30, 'Statistics are taken over the last two cycles (the last one if fewer than three are simulated).', { step: 1 }),
      F('jetTol', 'Integral-model tolerance', '–', 1e-6, 1e-9, 1e-3, 'Relative tolerance of the adaptive Runge–Kutta integration.'),
    ] },
  ],

  presets: [
    { name: 'Seawater RO brine, 2,000 m³/h, auto-designed diffuser', values: {} },
    { name: 'Large Gulf plant: 20,000 m³/h hypersaline brine, weak tide', values: { Qb: 20000, Sb: 68, Tb: 34, Sa: 42, Ta: 32, depth: 15, slope: 0.6, uM2: 0.12, uS2: 0.04, uRes: 0.04, Lx: 8000, Ly: 6000, inX: -2500, inY: -600, mzR: 300, receptors: [{ name: 'Seagrass meadow', x: 1500, y: -400, thr: 0.5 }, { name: 'Coral reef', x: -1200, y: 900, thr: 0.3 }] } },
    { name: 'Single-port outfall on a steep coast, strong tide', values: { design: 'manual', nPorts: 1, dPort: 280, theta: 45, Qb: 1200, depth: 18, slope: 4, uM2: 0.6, uS2: 0.2, bayAmp: 60, Lx: 5000, Ly: 2500, inX: 700, inY: -200, receptors: [{ name: 'Kelp bed', x: 500, y: -250, thr: 0.5 }, { name: 'Offshore reef', x: -600, y: 300, thr: 0.3 }] } },
    { name: 'Stratified summer case, fully mixed far field with particles', values: { dS: 0.6, dT: 4, layer: 'mixed', disp: 'okubo', particles: true, windSpeed: 9, windFactor: 1.2 } },
    { name: 'Thermal-plant brine in shallow water — jets reach the surface (problem case)', values: { design: 'manual', nPorts: 12, dPort: 200, spacing: 4, Qb: 6000, Sb: 52, Tb: 32, cCl: 0.1, depth: 9, disp: 'const', K0: 1.5 } },
  ],

  pull: ({ outputs, feed } = {}) => {
    const c = outputs?.ro?.streams?.concentrate, z = outputs?.zld, p = outputs?.plant?.streams?.brine, items = [], src = c?.Q > 0 ? [c, 'RO concentrate'] : p?.Q > 0 ? [p, 'Plant brine'] : null;
    if (src) {
      const [s, from] = src, T = Number.isFinite(s.T) ? s.T : 25;
      items.push({ key: 'Qb', value: s.Q, from: from + ' flow' });
      if (s.tds > 0) items.push({ key: 'Sb', value: salinityFromTDS(s.tds, T), from: from + ' salinity' });
      if (Number.isFinite(s.T)) items.push({ key: 'Tb', value: s.T + 1, from: from + ' temperature (+1 °C pumping)' });
    } else if (z?.liquidDischarge > 0) items.push({ key: 'Qb', value: z.liquidDischarge, from: 'ZLD liquid discharge' });
    const ro = outputs?.ro, dose = outputs?.chem?.antiscalantDose;
    if (dose > 0 && ro?.recovery > 0 && ro.recovery < 1) items.push({ key: 'cAnti', value: dose / (1 - ro.recovery), from: 'Antiscalant dose × concentration factor' });
    if (Number.isFinite(feed?.T)) items.push({ key: 'Ta', value: feed.T, from: 'Case feed water temperature' });
    return items;
  },
  site: (site) => {
    const d = site?.data || {}, it = [], add = (key, value, from) => { if (value !== undefined && value !== null && (typeof value !== 'number' || Number.isFinite(value))) it.push({ key, value, from }); };
    add('depth', d.depth > 2 ? d.depth : undefined, 'Water depth at site'); add('Ta', d.sst, 'Sea-surface temperature at site'); add('Sa', d.salinity, 'Salinity at site');
    add('uM2', d.currentSpeed > 0 ? +(d.currentSpeed * Math.PI * 0.5).toFixed(3) : undefined, 'Mean current speed at site × π/2 (tidal amplitude)'); add('tideDir', d.currentDir, 'Current direction at site');
    const eta = Array.isArray(d.tide?.eta) ? d.tide.eta.filter(Number.isFinite) : [];
    add('tideRange', Number.isFinite(d.tideRange) ? d.tideRange : eta.length > 3 ? +(Math.max(...eta) - Math.min(...eta)).toFixed(2) : undefined, Number.isFinite(d.tideRange) ? 'Tidal range at site' : 'Range of the site tide series'); add('waveHeight', d.waveHeight, 'Wave height at site'); add('wavePeriod', d.wavePeriod, 'Wave period at site'); add('windSpeed', d.windSpeed, 'Wind speed at site'); add('windDir', d.windDir, 'Wind direction at site');
    if (d.bathy?.elev?.length) add('bathy', { ...d.bathy, name: d.bathy.name || 'Site bathymetry' }, 'Bathymetry grid at site');
    const c = d.currents;
    if (Array.isArray(c?.t) && c.t.length >= 3) { const k = Math.ceil(c.t.length / 60); add('curSeries', c.t.map((t, i) => ({ t, speed: c.speed?.[i], dir: c.dir?.[i] })).filter((r, i) => i % k === 0 && [r.t, r.speed, r.dir].every(Number.isFinite)), 'Current time series at site'); }
    return it;
  },

  async run(v, ctx) {
    const W = [], nx = clamp(Math.round(v.nx), 12, 260), ny = clamp(Math.round(v.ny), 10, 200);
    const g = makeGrid(v, nx, ny), P = prep(v, g.depthOut || v.depth), cur = currents(v), [jx, jy] = bearing(v.jetDir), depth = P.depth;
    ctx?.progress?.(0.02, 'Near field: integral jet model');
    if (g.note) W.push({ level: 'warn', msg: g.note });
    const bedSlope = clamp((g.fn(0, 0) - g.fn(60 * jx, 60 * jy)) / 60, 0, 0.3);
    // ---- near field: slack water is the design case; flowing-ambient cases for comparison
    const jet = P.jet(0, 0, bedSlope), dense = P.gp > 0 && jet.fate === 'seabed', nf = nearFieldEnd(jet), dS0 = Math.abs(P.dS0), sign = P.dS0 >= 0 ? 1 : -1;
    const sig = (v.jetDir - v.tideDir) * D2R, scen = [['Slack water (design case)', 0, 0, jet], ['Mean current, flood', cur.mean, sig, null], ['Peak current, flood', cur.peak, sig, null], ['Peak current, ebb', cur.peak, sig + Math.PI, null]].map(([name, ua, s, j]) => { const q = j || P.jet(ua, s, bedSlope); return { name, ua, j: q, nf: nearFieldEnd(q) }; });
    const lowDepth = depth - 0.5 * v.tideRange, ztAbs = jet.zt + P.z0;
    // ---- intermediate field: bottom density current
    const W0 = Math.max(P.Ldiff, 2 * jet.bi) + nf.xn, gc = dense ? bottomCurrent({ q0: nf.Sn * P.Q, B: P.gp * P.Q, h0: Math.min(nf.yL, 0.7 * depth), W0, slope: bedSlope, Cd: v.Cd, xMax: 0.5 * Math.max(v.Lx, v.Ly), depth }) : null;
    const Kfun = (l) => v.Kmult * (v.disp === 'okubo' ? okubo(l) : v.disp === 'const' ? v.K0 : v.K0 + 0.6 * Math.sqrt(v.Cd) * Math.hypot(cur.rms, 0.7 * waveOrbital(v.waveHeight, v.wavePeriod, depth)) * depth);
    const gcEnd = gc ? { x: gc.x.at(-1), S: gc.q.at(-1) / P.Q, W: gc.W.at(-1) } : { x: 0, S: nf.Sn, W: W0 }, uB = Math.max(cur.mean, 0.02);
    /** Minimum dilution versus horizontal distance from the diffuser (near field → density current → Brooks). */
    const rJet = jet.path.r.map(((m) => (r) => (m = Math.max(m, r)))(0));
    const dilAt = (r) => {
      if (r <= jet.xi) return interp1(rJet, jet.path.Sc, r);
      if (r <= nf.xn) return jet.Si * (nf.Sn / jet.Si) ** ((r - jet.xi) / Math.max(nf.xn - jet.xi, 1e-9));
      const x = r - nf.xn;
      if (gc && x <= gcEnd.x) return interp1(gc.x, gc.q, x) / P.Q;
      return gcEnd.S * brooks(x - gcEnd.x, gcEnd.W, uB, Kfun(gcEnd.W));
    };
    // ---- far field
    const flow = flowBasis(g), n = nx * ny, hL = v.hLayer > 0 ? v.hLayer : gc ? interp1(gc.x, gc.h, Math.min(150, gcEnd.x)) : nf.yL;
    const phi = v.layer === 'layer' ? clamp(hL / depth, 0.04, 1) : 1, bedF = v.layer === 'layer' ? clamp(v.bedF, 0.1, 1) : 1, uw = waveOrbital(v.waveHeight, v.wavePeriod, depth);
    const K = new Float64Array(n), spd = new Float64Array(n), [ax, ay] = cur.axis;
    for (let j = 0, q = 0; j < ny; j++) for (let i = 0; i < nx; i++, q++) {
      if (!g.H[q]) continue;
      spd[q] = cur.rms * Math.hypot(ax * flow.basis[0].u[q] + ay * flow.basis[1].u[q], ax * flow.basis[0].v[q] + ay * flow.basis[1].v[q]);
      const uwq = uw * Math.min(3, (Math.sinh(Math.min((2 * Math.PI * depth) / (v.wavePeriod * Math.sqrt(G * depth)), 20)) / Math.sinh(Math.min((2 * Math.PI * g.H[q]) / (v.wavePeriod * Math.sqrt(G * g.H[q])), 20))) || 1);
      K[q] = v.Kmult * (v.disp === 'const' ? v.K0 : v.disp === 'okubo' ? Math.min(okubo(Math.max(g.dx, Math.hypot(g.xs[i], g.ys[j]))), okubo(5000)) : v.K0 + 0.6 * Math.sqrt(v.Cd) * Math.hypot(spd[q], 0.7 * uwq) * phi * g.H[q]);
    }
    const sx = clamp(jx * nf.xn, g.x0 + g.dx, g.x0 + v.Lx - g.dx), sy = clamp(jy * nf.xn, g.y0 + g.dy, g.y0 + v.Ly - g.dy), sg = Math.max(0.7 * Math.min(g.dx, g.dy), 0.4 * W0);
    let src = [], sw = 0;
    for (let j = 0, q = 0; j < ny; j++) for (let i = 0; i < nx; i++, q++) { const r2 = (g.xs[i] - sx) ** 2 + (g.ys[j] - sy) ** 2; if (g.H[q] && r2 < 9 * sg * sg) { const w = Math.exp(-r2 / (2 * sg * sg)); src.push({ P: q, w }); sw += w; } }
    if (!src.length) { let best = -1, bd = Infinity; for (let q = 0; q < n; q++) if (g.H[q]) { const d2 = (g.xs[q % nx] - sx) ** 2 + (g.ys[(q - (q % nx)) / nx] - sy) ** 2; if (d2 < bd) { bd = d2; best = q; } } if (best < 0) throw new Error('The model domain contains no water cells. Check the bathymetry, the outfall position and the depth.'); src = [{ P: best, w: 1 }]; sw = 1; W.push({ level: 'warn', msg: 'The outfall lies on land or in very shallow water on the model grid; the source was moved to the nearest water cell.' }); }
    src.forEach((s) => (s.w /= sw));
    const inDom = (x, y) => [clamp(x, g.x0 + 0.5 * g.dx, g.x0 + v.Lx - 0.5 * g.dx), clamp(y, g.y0 + 0.5 * g.dy, g.y0 + v.Ly - 0.5 * g.dy)];
    const recs = (Array.isArray(v.receptors) ? v.receptors : []).filter((r) => Number.isFinite(+r.x) && Number.isFinite(+r.y)).slice(0, 12).map((r, k) => ({ name: String(r.name || `Receptor ${k + 1}`).slice(0, 40), x: +r.x, y: +r.y, thr: +r.thr > 0 ? +r.thr : P.limit }));
    const probes = [inDom(v.inX, v.inY), ...recs.map((r) => inDom(r.x, r.y))], ring = linspace(0, 2 * Math.PI, 49).slice(0, 48).map((a) => [v.mzR * Math.cos(a), v.mzR * Math.sin(a)]).filter(([x, y]) => x > g.x0 && x < g.x0 + v.Lx && y > g.y0 && y < g.y0 + v.Ly);
    const TM2 = TIDES.M2 * 3600, nCyc = clamp(Math.round(v.nCycles), 1, 40), tEnd = nCyc * TM2, tStat = nCyc >= 3 ? tEnd - 2 * TM2 : nCyc === 2 ? TM2 : 0;
    const betaS = (density(v.Ta, v.Sa + 1) - density(v.Ta, v.Sa)) / density(v.Ta, v.Sa);
    const ff = await farField({ g, flow, cur, K, phi, bedF, scheme: v.scheme, cfl: clamp(v.cfl, 0.05, 0.9), tEnd, tStat, src, rate: P.Q * dS0, probes, ring, drift: v.layer === 'layer' && v.drift && dense ? { betaS, Cd: v.Cd, vmax: 0.4 } : null, thr: 0.1 * P.limit, particles: v.particles ? clamp(v.nPart, 100, 10000) : 0, srcXY: [sx, sy], srcR: Math.max(sg, g.dx) }, ctx);
    ctx?.progress?.(0.96, 'Environmental assessment');
    if (!ff.complete) W.push({ level: 'warn', msg: `The far-field run stopped after ${ff.steps} steps at ${fmt(ff.tEnd / 3600, 3)} h — coarsen the grid or raise the CFL number.` });
    // ---- assessment
    const rows2d = (a, sc = 1) => g.ys.map((_, j) => g.xs.map((__, i) => (g.H[j * nx + i] ? sc * a[j * nx + i] : NaN))), mask = g.ys.map((_, j) => g.xs.map((__, i) => !g.H[j * nx + i]));
    const mzNear = dS0 / dilAt(v.mzR), mzFar = ff.ringMax, mzEx = Math.max(mzNear, mzFar), cellA = g.dx * g.dy;
    const expo = (a, thr) => { let ar = 0, vol = 0, rmax = 0; for (let q = 0; q < n; q++) if (g.H[q] && a[q] > thr) { ar += cellA; vol += phi * g.H[q] * cellA; rmax = Math.max(rmax, Math.hypot(g.xs[q % nx], g.ys[(q - (q % nx)) / nx]) + 0.5 * Math.hypot(g.dx, g.dy)); } return { area: ar, vol, rmax }; };
    const eLim = expo(ff.Cmax, P.limit), eThr = expo(ff.Cmax, v.thrArea), eMeanThr = expo(ff.Cmean, v.thrArea), eMeanLim = expo(ff.Cmean, P.limit);
    let rA = 0;
    if (dS0 / dilAt(0) > P.limit) { const rs = [...linspace(0.5, nf.xn, 80), ...linspace(nf.xn, nf.xn + 0.7 * Math.max(v.Lx, v.Ly), 400)]; rA = rs.find((r) => dS0 / dilAt(r) <= P.limit) ?? rs.at(-1); }
    const compliance = Math.max(rA, eLim.rmax), iStat = ff.ser.t.findIndex((t) => t * 3600 >= tStat), tS = ff.ser.t.slice(Math.max(iStat, 0));
    const stats = (s, thr) => { const a = s.slice(Math.max(iStat, 0)); let ex = 0, run = 0, longest = 0; a.forEach((c, k) => { const dtk = k ? tS[k] - tS[k - 1] : 0; if (c > thr) { ex += dtk; run += dtk; longest = Math.max(longest, run); } else run = 0; }); const span = tS.at(-1) - tS[0] || 1; return { max: Math.max(...a, 0), mean: mean(a) || 0, frac: ex / span, longest }; };
    const intake = stats(ff.ser.probes[0], P.limit), recStats = recs.map((r, k) => ({ ...r, ...stats(ff.ser.probes[k + 1], r.thr), wet: g.H[clamp(Math.floor((probes[k + 1][1] - g.y0) / g.dy), 0, ny - 1) * nx + clamp(Math.floor((probes[k + 1][0] - g.x0) / g.dx), 0, nx - 1)] > 0 }));
    let shoreD = Infinity;
    for (let q = 0; q < n; q++) if (!g.H[q]) shoreD = Math.min(shoreD, Math.hypot(g.xs[q % nx], g.ys[(q - (q % nx)) / nx]));
    const outfallLength = (Number.isFinite(shoreD) ? shoreD : v.slope > 0 ? depth / (v.slope / 100) : 0) + P.Ldiff;
    const chem = [['Antiscalant', v.cAnti, v.limAnti], ['Chlorine / oxidant', v.cCl, v.limCl]].map(([name, c0, lim]) => ({ name, c0, lim, imp: c0 / jet.Si, nfe: c0 / nf.Sn, mz: dS0 > 0 ? (c0 * mzEx) / dS0 : c0 / dilAt(v.mzR), need: lim > 0 ? c0 / lim : 0 }));
    const uAtMax = Math.hypot(...currentAt(cur, ff.tRingMax)), impS = v.Sa + (v.dS * (0.5 - 0 / depth)) + (sign * dS0) / jet.Si;
    // ---- warnings
    if (!(P.gp > 0)) W.push({ level: 'warn', msg: `The brine (${fmt(P.rhoB, 5)} kg/m³) is not denser than the ambient water (${fmt(P.rhoA, 5)} kg/m³): the discharge behaves as a buoyant jet and rises to the surface. The dense-jet coefficients and the bottom-layer assessment do not apply.` });
    else if (jet.fate === 'surface' || ztAbs > lowDepth) W.push({ level: 'bad', msg: `The jets rise ${fmt(ztAbs, 3)} m above the bed and reach the surface at low water (${fmt(lowDepth, 3)} m): dilution is impaired and the plume becomes visible. Use more, smaller ports or a lower angle.` });
    else if (ztAbs > (v.clear / 100) * lowDepth) W.push({ level: 'warn', msg: `The jet top (${fmt(ztAbs, 3)} m) exceeds ${v.clear} % of the low-water depth (${fmt(lowDepth, 3)} m).` });
    if (jet.fate === 'range') W.push({ level: 'warn', msg: 'The jet did not return to the seabed within the integration range (very weak density difference); impact values refer to the end of the computed trajectory.' });
    if (jet.F < v.Fmin && P.gp > 0) W.push({ level: 'warn', msg: `Densimetric Froude number ${fmt(jet.F, 3)} is below ${v.Fmin}: the jets are weak and brine may fall back onto the diffuser.` });
    if (P.U0 < v.vMin || P.U0 > v.vMax) W.push({ level: 'warn', msg: `Port exit velocity ${fmt(P.U0, 3)} m/s is outside the ${v.vMin}–${v.vMax} m/s design window${P.U0 > v.vMax ? ' (head loss and fish-entrainment risk)' : ' (poor mixing and port fouling)'}.` });
    if (jet.path.merged) W.push({ level: 'info', msg: `Neighbouring jets merge before reaching the bed (spacing ${fmt(P.spacing, 3)} m); the merged entrainment perimeter is used. A spacing above about ${fmt(2 * P.d * jet.F, 3)} m avoids merging.` });
    if (mzEx > P.limit) W.push({ level: 'bad', msg: `Excess salinity at the ${v.mzR} m mixing-zone edge reaches ${fmt(mzEx, 3)} g/kg against a limit of ${fmt(P.limit, 3)} g/kg. Compliance is reached at about ${fmt(compliance, 3)} m.` });
    if (intake.max > 0.02 * v.Sa) W.push({ level: 'bad', msg: `Recirculation: the excess salinity at the intake peaks at ${fmt(intake.max, 3)} g/kg (${fmt((100 * intake.max) / v.Sa, 2)} % of ambient), raising RO feed pressure and energy use.` });
    else if (intake.max > 0.005 * v.Sa) W.push({ level: 'warn', msg: `Some brine returns to the intake: up to ${fmt(intake.max, 3)} g/kg above ambient.` });
    recStats.forEach((r) => { if (!r.wet) W.push({ level: 'info', msg: `Receptor “${r.name}” lies on land or outside the wet model domain; its values are taken from the nearest water.` }); if (r.max > r.thr) W.push({ level: r.frac > 0.25 ? 'bad' : 'warn', msg: `Receptor “${r.name}”: excess salinity up to ${fmt(r.max, 3)} g/kg exceeds its ${fmt(r.thr, 3)} g/kg threshold for ${fmt(100 * r.frac, 3)} % of the time (longest episode ${fmt(r.longest, 3)} h).` }); });
    chem.forEach((c) => { if (c.mz > c.lim) W.push({ level: 'warn', msg: `${c.name}: ${fmt(c.mz, 3)} mg/L at the mixing-zone edge exceeds the ${fmt(c.lim, 3)} mg/L criterion (required dilution ${fmt(c.need, 3)}).` }); });
    if (P.auto) W.push({ level: 'info', msg: `Diffuser sized automatically: ${P.n} port${P.n > 1 ? 's' : ''} of ${fmt(P.d * 1000, 3)} mm at 60°, spacing ${Number.isFinite(P.spacing) ? fmt(P.spacing, 3) + ' m' : '–'} (${P.des.best.why}).` });
    if (Math.abs(g.depthOut - v.depth) > 0.5 && g.source !== 'synthetic') W.push({ level: 'info', msg: `Depth at the outfall read from the bathymetry: ${fmt(g.depthOut, 3)} m (the depth input of ${v.depth} m is not used).` });
    if (!flow.local) W.push({ level: 'info', msg: 'The outfall cell is sheltered on the model grid; the current pattern was scaled to the prescribed speed as a domain average.' });
    if (v.scheme === 'upwind') W.push({ level: 'info', msg: `First-order upwind adds numerical diffusion of about ${fmt(0.5 * cur.rms * g.dx, 2)} m²/s, compared with a physical coefficient of ${fmt(K[g.jo * nx + g.io], 2)} m²/s at the outfall.` });
    if (!W.some((w) => w.level === 'bad' || w.level === 'warn')) W.unshift({ level: 'info', msg: `The discharge meets the ${fmt(P.limit, 3)} g/kg limit at the ${v.mzR} m mixing-zone edge with a margin of ${fmt(P.limit / Math.max(mzEx, 1e-9), 3)} ×.` });
    // ---- plots
    const pth = jet.path, flat = g.ys.map((_, j) => g.xs.map((__, i) => (g.H[j * nx + i] ? g.H[j * nx + i] : NaN))), zrows = g.ys.map((_, j) => g.xs.map((__, i) => g.zb[j * nx + i]));
    const iso = [...isolines(zrows, g.xs, g.ys, -HMIN, '#0f172a', 30).map((s) => ({ ...s, width: 1.6 })), ...[0.5, 1, 2].flatMap((m) => isolines(zrows, g.xs, g.ys, -m * depth, 'rgba(255,255,255,.45)', 12))];
    const circ = linspace(0, 2 * Math.PI, 49), shapes = [...iso, { x: circ.map((a) => v.mzR * Math.cos(a)), y: circ.map((a) => v.mzR * Math.sin(a)), closed: true, color: '#f97316', dash: true }];
    const markers = [{ x: 0, y: 0, label: 'Outfall' }, { x: probes[0][0], y: probes[0][1], label: 'Intake', color: '#38bdf8' }, ...recs.map((r, k) => ({ x: probes[k + 1][0], y: probes[k + 1][1], label: r.name, color: '#facc15' }))];
    const zmax = Math.max(1e-6, ...Array.from(ff.Cmax).filter((_, q) => g.H[q])), fbase = { type: 'field', xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', x: g.xs, y: g.ys, mask, equal: true, zlabel: 'Excess salinity', zunit: 'g/kg', cmap: 'salinity', zmin: 0, zmax, shapes, markers };
    const rr = [...linspace(0.2, nf.xn, 40), ...linspace(nf.xn * 1.02, Math.max(10 * nf.xn, 0.45 * Math.max(v.Lx, v.Ly)), 80)], bedLine = pth.r.map((r) => -bedSlope * r);
    const zA = linspace(0, depth, 25), plots = [
      { type: 'line', title: 'Near field: jet trajectory (slack water)', xlabel: 'Horizontal distance from the port (m)', ylabel: 'Height above the seabed at the port (m)', series: [{ name: 'Jet centre-line', x: pth.r, y: pth.z }, { name: 'Upper boundary', x: pth.r, y: pth.up, dash: true }, { name: 'Lower boundary', x: pth.r, y: pth.lo.map((z, k) => Math.max(z, bedLine[k])), dash: true }, { name: 'Seabed', x: pth.r, y: bedLine, color: '#92400e' }, { name: `Centre-line at mean current (${fmt(cur.mean, 2)} m/s)`, x: scen[1].j.path.r, y: scen[1].j.path.z, color: '#94a3b8' }],
        hlines: [{ y: lowDepth, label: 'water surface at low tide', color: '#0ea5e9' }], note: `Terminal rise ${fmt(jet.zt, 3)} m above the port, impact at ${fmt(jet.xi, 3)} m with a minimum dilution of ${fmt(jet.Si, 3)}.` },
      { type: 'line', title: 'Minimum dilution with distance from the diffuser', xlabel: 'Distance from the diffuser (m)', ylabel: 'Dilution (–)', logx: true, logy: true, series: [{ name: 'Jet → bottom layer → density current → far field', x: rr, y: rr.map(dilAt) }, { name: 'Far-field model, tidal maximum along the plume axis', x: [v.mzR], y: [mzFar > 0 ? dS0 / mzFar : null], mode: 'points' }],
        hlines: [{ y: dS0 / P.limit, label: 'dilution needed for the salinity limit' }], vlines: [{ x: jet.xi, label: 'impact' }, { x: nf.xn, label: 'end of near field' }, { x: v.mzR, label: 'mixing zone' }] },
      { ...fbase, title: `Far field: excess salinity at maximum plume extent (t = ${fmt(ff.tSnap / 3600, 3)} h)`, z: rows2d(ff.Csnap), contours: 6 },
      { ...fbase, title: 'Far field: tidal-mean excess salinity', z: rows2d(ff.Cmean), contours: 6 },
      { ...fbase, title: 'Far field: maximum envelope over the tidal cycle', z: rows2d(ff.Cmax), contours: 6, note: `Layer of ${fmt(phi * depth, 3)} m at the outfall (${fmt(100 * phi, 3)} % of the depth). Orange circle: ${v.mzR} m mixing zone; dark line: shoreline; light lines: isobaths at 0.5, 1 and 2 × outfall depth.` },
      { type: 'line', title: 'Excess salinity at the intake, receptors and mixing-zone edge', xlabel: 'Time (h)', ylabel: 'Excess salinity (g/kg)', zeroY: true, series: [{ name: 'Mixing-zone edge (maximum on the ring)', x: ff.ser.t, y: ff.ser.ring }, { name: 'Intake', x: ff.ser.t, y: ff.ser.probes[0] }, ...recs.map((r, k) => ({ name: r.name, x: ff.ser.t, y: ff.ser.probes[k + 1] }))], vlines: [{ x: tStat / 3600, label: 'statistics from here' }] },
      { type: 'line', title: 'Current at the outfall', xlabel: 'Time (h)', ylabel: 'm/s', series: [{ name: 'Eastward', x: ff.ser.t, y: ff.ser.u }, { name: 'Northward', x: ff.ser.t, y: ff.ser.v }, { name: 'Speed', x: ff.ser.t, y: ff.ser.u.map((u, k) => Math.hypot(u, ff.ser.v[k])), dash: true }], note: cur.series ? 'Driven by the supplied current time series.' : `Constituents: ${cur.cons.map((c) => `${c.id} ${fmt(c.amp, 2)} m/s`).join(', ') || 'none'}; residual ${fmt(v.uRes, 2)} m/s; wind drift ${fmt(Math.hypot(...cur.wind), 2)} m/s.` },
      { type: 'field', title: 'Bathymetry and flood-current pattern', xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', x: g.xs, y: g.ys, z: flat, mask, equal: true, zlabel: 'Water depth', zunit: 'm', cmap: 'viridis', contours: 8, vectors: true,
        u: g.ys.map((_, j) => g.xs.map((__, i) => ax * flow.basis[0].u[j * nx + i] + ay * flow.basis[1].u[j * nx + i])), v: g.ys.map((_, j) => g.xs.map((__, i) => ax * flow.basis[0].v[j * nx + i] + ay * flow.basis[1].v[j * nx + i])), markers, shapes: shapes.slice(-1), note: `Source: ${g.source}. Arrows: current per unit flood velocity at the outfall.` },
      { type: 'line', title: 'Density: ambient profile and jet centre-line', xlabel: 'Density (kg/m³)', ylabel: 'Height above the seabed (m)', series: [{ name: 'Ambient', x: zA.map((z) => { const a = P.amb(0)(z); return density(a.T, a.S); }), y: zA }, { name: 'Jet (cross-section mean)', x: pth.rho, y: pth.z }] },
    ];
    if (gc) plots.push({ type: 'line', title: 'Intermediate field: bottom density current', xlabel: 'Distance beyond the near field (m)', ylabel: 'Thickness (m) · velocity (cm/s) · width/10 (m)', series: [{ name: 'Layer thickness (m)', x: gc.x, y: gc.h }, { name: 'Velocity (cm/s)', x: gc.x, y: gc.U.map((u) => 100 * u) }, { name: 'Width ÷ 10 (m)', x: gc.x, y: gc.W.map((w) => w / 10) }], note: gc.arrest ? `The current arrests after ${fmt(gcEnd.x, 3)} m (it becomes sub-critical); beyond that the ambient current and dispersion control the spreading.` : `Bed slope ${fmt(100 * bedSlope, 2)} % along the discharge direction.` });
    if (P.des) { const dr = P.des.rows.filter((r) => r.n <= Math.max(12, 2 * P.des.best.n)); plots.push({ type: 'line', title: 'Diffuser design chart (60° ports)', xlabel: 'Number of ports', ylabel: 'Value', series: [{ name: 'Impact dilution S_i', x: dr.map((r) => r.n), y: dr.map((r) => r.Si), mode: 'both' }, { name: 'Froude number', x: dr.map((r) => r.n), y: dr.map((r) => r.F), mode: 'both' }, { name: 'Jet rise height × 10 (m)', x: dr.map((r) => r.n), y: dr.map((r) => 10 * r.zt), mode: 'both' }, { name: 'Port velocity × 10 (m/s)', x: dr.map((r) => r.n), y: dr.map((r) => 10 * r.V), mode: 'both' }], vlines: [{ x: P.n, label: P.auto ? 'selected' : 'entered' }] }); }
    if (ff.part) plots.push({ type: 'line', title: 'Random-walk particles at the end of the run (plume envelope)', xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', xmin: g.x0, xmax: g.x0 + v.Lx, ymin: g.y0, ymax: g.y0 + v.Ly, series: [{ name: `Particles (${ff.part.x.length})`, x: ff.part.x, y: ff.part.y, mode: 'points', size: 1.5 }, ...iso.slice(0, 6).map((s, k) => ({ name: k ? `Shoreline ${k + 1}` : 'Shoreline', x: s.x, y: s.y, color: '#92400e' })), { name: 'Outfall', x: [0], y: [0], mode: 'points', size: 5, color: '#ef4444' }, { name: 'Intake', x: [probes[0][0]], y: [probes[0][1]], mode: 'points', size: 5, color: '#0ea5e9' }] });
    // ---- tables
    const near60 = Math.abs(P.theta / D2R - 60) <= 5 && dense, dF = P.d * jet.F;
    const tables = [
      { title: 'Near field: integral model and empirical cross-check', columns: ['Quantity', 'Integral model', 'Empirical (60° dense jet)', 'Model ÷ (d·F or F)', 'Empirical coefficient'], rows: [
        ['Densimetric Froude number F', jet.F, null, null, null], ['Terminal rise height above port (m)', jet.zt, near60 ? ROBERTS60.zt * dF : null, jet.zt / dF, ROBERTS60.zt], ['Impact distance (m)', jet.xi, near60 ? ROBERTS60.xi * dF : null, jet.xi / dF, ROBERTS60.xi],
        ['Impact (minimum) dilution', jet.Si, near60 ? ROBERTS60.Si * jet.F : null, jet.Si / jet.F, ROBERTS60.Si], ['Near-field length (m)', nf.xn, near60 ? ROBERTS60.xn * dF : null, nf.xn / dF, ROBERTS60.xn], ['Near-field dilution', nf.Sn, near60 ? ROBERTS60.Sn * jet.F : null, nf.Sn / jet.F, ROBERTS60.Sn], ['Bottom-layer thickness (m)', nf.yL, near60 ? ROBERTS60.yL * dF : null, nf.yL / dF, ROBERTS60.yL]],
        note: near60 ? 'Empirical coefficients: Roberts, Ferrier & Daviero (1997), single 60° jets in still water. The near-field end and layer thickness of the model use the empirical ratios to the impact values.' : 'The empirical coefficients are quoted for 60° jets in still water only; they are shown as coefficients but not converted for this port angle or buoyancy.' },
      { title: 'Near-field scenarios', columns: ['Ambient condition', 'Current (m/s)', 'Rise height (m)', 'Impact distance (m)', 'Impact dilution', 'Near-field dilution', 'Salinity at impact (g/kg)', 'Excess at near-field end (g/kg)', 'Fate'], rows: scen.map((s) => [s.name, s.ua, s.j.zt, s.j.xi, s.j.Si, s.nf.Sn, v.Sa + (sign * dS0) / s.j.Si, dS0 / s.nf.Sn, s.j.fate === 'seabed' ? 'returns to seabed' : s.j.fate === 'surface' ? 'reaches surface' : 'carried by the current']) },
      { title: 'Environmental compliance', columns: ['Criterion', 'Predicted', 'Limit', 'Status'], rows: [
        [`Excess salinity at the ${v.mzR} m mixing-zone edge (g/kg)`, mzEx, P.limit, mzEx <= P.limit ? 'complies' : 'EXCEEDS'], ['… near-field / density-current estimate, slack water (g/kg)', mzNear, P.limit, mzNear <= P.limit ? 'complies' : 'EXCEEDS'], ['… far-field model, tidal maximum on the ring (g/kg)', mzFar, P.limit, mzFar <= P.limit ? 'complies' : 'EXCEEDS'],
        ['Distance to compliance (m)', compliance, v.mzR, compliance <= v.mzR ? 'complies' : 'EXCEEDS'], ['Maximum excess at the intake (g/kg)', intake.max, 0.02 * v.Sa, intake.max <= 0.02 * v.Sa ? 'acceptable' : 'RECIRCULATION'], ['Mean excess at the intake (g/kg)', intake.mean, null, ''],
        ...chem.map((c) => [`${c.name} at the mixing-zone edge (mg/L)`, c.mz, c.lim, c.mz <= c.lim ? 'complies' : 'EXCEEDS']),
        [`Seabed area above the limit, tidal maximum (ha)`, eLim.area / 1e4, null, ''], [`Seabed area above ${v.thrArea} g/kg, tidal maximum (ha)`, eThr.area / 1e4, null, ''], [`Seabed area above ${v.thrArea} g/kg, tidal mean (ha)`, eMeanThr.area / 1e4, null, ''], [`Volume above ${v.thrArea} g/kg, tidal maximum (1000 m³)`, eThr.vol / 1e3, null, ''], ['Seabed area above the limit, tidal mean (ha)', eMeanLim.area / 1e4, null, '']],
        note: `Worst case at the mixing-zone edge occurs ${fmt(ff.tRingMax / 3600, 3)} h into the run at a current speed of ${fmt(uAtMax, 2)} m/s (slack water is ${fmt(cur.min, 2)} m/s). Limit = min(${v.limAbs} g/kg, ${v.limRel} % of ambient).` },
      { title: 'Receptor and intake exposure (statistics window)', columns: ['Location', 'East (m)', 'North (m)', 'Threshold (g/kg)', 'Maximum ΔS (g/kg)', 'Mean ΔS (g/kg)', 'Time above threshold (%)', 'Longest episode (h)'], rows: [['Intake', v.inX, v.inY, P.limit, intake.max, intake.mean, 100 * intake.frac, intake.longest], ...recStats.map((r) => [r.name, r.x, r.y, r.thr, r.max, r.mean, 100 * r.frac, r.longest])],
        note: `Far-field numerics: ${nx} × ${ny} cells of ${fmt(g.dx, 3)} × ${fmt(g.dy, 3)} m, ${ff.steps} steps with a mean time step of ${fmt(ff.dtMean, 3)} s, ${nCyc} M2 cycle${nCyc > 1 ? 's' : ''} simulated and statistics from ${fmt(tStat / 3600, 3)} h onward. Dispersion coefficient at the outfall ${fmt(K[g.jo * nx + g.io], 2)} m²/s; tidal excursion about ${fmt((cur.rms * Math.SQRT2 * TIDES.M2 * 3600) / Math.PI, 3)} m; bathymetry: ${g.source}.` },
    ];
    if (P.des) tables.push({ title: 'Diffuser design options (60° ports)', columns: ['Ports', 'Diameter (mm)', 'Velocity (m/s)', 'Froude number', 'Rise height (m)', 'Impact dilution', 'Near-field dilution', 'Spacing (m)', 'Diffuser length (m)', 'Assessment'], rows: P.des.rows.filter((r) => r.n <= Math.max(12, 2 * P.des.best.n)).map((r) => [r.n, r.d * 1000, r.V, r.F, r.zt, r.Si, r.Sn, r.s, r.len, (r === P.des.best ? '★ ' : '') + r.why]), note: 'Sized with the empirical 60° coefficients: velocity inside the design window, Froude number above the minimum, jet top below the surface-clearance limit at low water, and the salinity limit met at the end of the near field.' });
    const mJetIn = jet.Q0 * (v.Sb - P.amb(0)(P.z0).S), L = pth.s.length - 1;
    const balances = [{ name: 'Far-field salt excess (g/kg·m³): injected vs stored + exported', in: ff.bal.injected, out: ff.bal.mass + ff.bal.out }];
    if (v.dS === 0 && v.dT === 0) balances.push({ name: 'Jet salt-excess flux (g/kg·m³/s per port)', in: mJetIn, out: pth.S[L] * jet.Q0 * (pth.sal[L] - v.Sa) });
    const outputs = { nearFieldDilution: nf.Sn, impactSalinity: impS, excessAtMixingZone: mzEx, complianceDistance: compliance, outfallLength, nPorts: P.n, portDiameter: P.d, impactDilution: jet.Si, froude: jet.F, exitVelocity: P.U0, riseHeight: jet.zt, mzFarField: mzFar, areaAboveThreshold: eThr.area, intakeExcessMax: intake.max, intakeExcessMean: intake.mean, limit: P.limit };
    for (const k of Object.keys(outputs)) if (!Number.isFinite(outputs[k])) delete outputs[k];
    return {
      summary: `${P.n} port${P.n > 1 ? 's' : ''} of ${fmt(P.d * 1000, 3)} mm at ${fmt(P.U0, 3)} m/s (F = ${fmt(jet.F, 3)}): impact dilution ${fmt(jet.Si, 3)}, near-field dilution ${fmt(nf.Sn, 3)}; excess salinity ${fmt(mzEx, 2)} g/kg at the ${v.mzR} m mixing-zone edge (limit ${fmt(P.limit, 3)}), up to ${fmt(intake.max, 2)} g/kg at the intake.`,
      kpis: [
        { label: 'Densimetric Froude number', value: jet.F, unit: '', status: P.gp > 0 && jet.F < v.Fmin ? 'warn' : 'ok', help: 'F = U / √(g′d)' },
        { label: 'Port exit velocity', value: P.U0, unit: 'm/s', status: P.U0 < v.vMin || P.U0 > v.vMax ? 'warn' : 'ok' },
        { label: 'Ports × diameter', value: `${P.n} × ${fmt(P.d * 1000, 3)} mm` },
        { label: 'Terminal rise height', value: jet.zt, unit: 'm', status: ztAbs > lowDepth ? 'bad' : ztAbs > (v.clear / 100) * lowDepth ? 'warn' : 'ok', help: 'Top of the jet above the port' },
        { label: 'Impact distance', value: jet.xi, unit: 'm' }, { label: 'Impact dilution', value: jet.Si, unit: '', help: 'Minimum (centre-line) dilution where the jet returns to the seabed' },
        { label: 'Salinity at impact', value: impS, unit: 'g/kg' }, { label: 'Near-field dilution', value: nf.Sn, unit: '' }, { label: 'Near-field length', value: nf.xn, unit: 'm' },
        { label: 'Bottom-layer thickness', value: phi * depth, unit: 'm', help: 'Thickness of the layer carried by the far-field model at the outfall' },
        { label: 'Excess at mixing-zone edge', value: mzEx, unit: 'g/kg', status: mzEx > P.limit ? 'bad' : 'ok' }, { label: 'Distance to compliance', value: compliance, unit: 'm', status: compliance > v.mzR ? 'bad' : 'ok' },
        { label: `Area above ${v.thrArea} g/kg`, value: eThr.area / 1e4, unit: 'ha', help: 'Seabed footprint of the tidal maximum' },
        { label: 'Maximum excess at intake', value: intake.max, unit: 'g/kg', status: intake.max > 0.02 * v.Sa ? 'bad' : intake.max > 0.005 * v.Sa ? 'warn' : 'ok' }, { label: 'Mean excess at intake', value: intake.mean, unit: 'g/kg' },
        { label: 'Worst receptor exceedance', value: recStats.length ? 100 * Math.max(...recStats.map((r) => r.frac)) : 0, unit: '% of time', status: recStats.some((r) => r.frac > 0) ? 'warn' : 'ok' },
        { label: 'Antiscalant at mixing-zone edge', value: chem[0].mz, unit: 'mg/L', status: chem[0].mz > chem[0].lim ? 'warn' : 'ok' }, { label: 'Outfall length', value: outfallLength, unit: 'm', help: 'Distance from the shoreline to the diffuser plus the diffuser length' },
      ],
      warnings: W,
      recommendations: [
        mzEx > P.limit ? 'Increase the near-field dilution: more and smaller ports raise the Froude number; the diffuser design table lists compliant options.' : null,
        ztAbs > (v.clear / 100) * lowDepth ? 'Move the outfall to deeper water or reduce the port diameter so that the jet top stays below the surface at low tide.' : null,
        intake.max > 0.005 * v.Sa ? 'Separate intake and outfall further, place the intake up-drift of the residual current or in shallower water away from the dense bottom layer.' : null,
        recStats.some((r) => r.max > r.thr) ? 'Re-orient the discharge away from the exceeded receptors or lengthen the outfall; check the maximum-envelope map for the plume path.' : null,
        gc?.arrest && bedSlope < 0.003 ? 'On this nearly flat seabed the brine layer stalls close to the diffuser; rely on tidal flushing and check for accumulation in depressions with field salinity profiles.' : null,
        'Calibrate the entrainment and dispersion coefficients against CTD and dye-tracer surveys on the Calibrate tab, and repeat the run for neap tides and calm weather.',
        'Run the grid study on the Mesh tab and quote the grid-convergence index with the far-field results.',
      ].filter(Boolean),
      plots, tables, balances, outputs,
    };
  },

  mesh: { name: 'Far-field grid (nx × ny)', keys: ['nx', 'ny'], min: 12, note: 'The domain size, the time-step criterion and the near-field solution are held constant while the far-field grid is refined.',
    metrics: [{ label: 'Far-field excess at the mixing-zone edge', unit: 'g/kg', get: (r) => r.outputs.mzFarField ?? 0 }, { label: 'Area above the reporting threshold', unit: 'm²', get: (r) => r.outputs.areaAboveThreshold ?? 0 }, { label: 'Maximum excess at the intake', unit: 'g/kg', get: (r) => r.outputs.intakeExcessMax ?? 0 }] },

  calibration: {
    note: 'Fit the entrainment closure of the jet model and the far-field dispersion multiplier. Each row is one survey or laboratory condition: brine flow, brine salinity, port angle and the ambient (residual) current; the measurements are the terminal rise height, the impact distance and the impact dilution of the jets in slack water, and the dilution 1 km down-current of the near field (dye or salinity transect). Enter the real diffuser first (Diffuser definition → “I will enter ports and diameter”).',
    params: [{ key: 'alphaJ', label: 'Jet entrainment coefficient', lo: 0.04, hi: 0.12 }, { key: 'descF', label: 'Descending-limb enhancement', lo: 1, hi: 3.5 }, { key: 'Kmult', label: 'Dispersion multiplier', lo: 0.1, hi: 10 }],
    columns: [{ key: 'Qb', label: 'Brine flow', unit: 'm³/h' }, { key: 'Sb', label: 'Brine salinity', unit: 'g/kg' }, { key: 'theta', label: 'Port angle', unit: '°' }, { key: 'uRes', label: 'Ambient current', unit: 'm/s' }, { key: 'zt', label: 'Rise height', unit: 'm' }, { key: 'xi', label: 'Impact distance', unit: 'm' }, { key: 'Si', label: 'Impact dilution', unit: '–' }, { key: 'dil1km', label: 'Dilution at 1 km', unit: '–' }],
    targets: [{ key: 'zt', label: 'Terminal rise height', unit: 'm' }, { key: 'xi', label: 'Impact distance', unit: 'm' }, { key: 'Si', label: 'Impact dilution', unit: '–' }, { key: 'dil1km', label: 'Dilution 1 km down-current', unit: '–' }],
    model(v) {
      const m = { ...v, design: 'manual' };
      if (v.design !== 'manual') { const a = prep({ ...v, theta: 60 }); m.nPorts = a.n; m.dPort = a.d * 1000; m.spacing = Number.isFinite(a.spacing) ? a.spacing : v.spacing; }
      const P = prep(m), j = P.jet(0, 0, 0, 1e-5), nf = nearFieldEnd(j), b = Math.max(P.Ldiff, 2 * j.bi) + nf.xn, u = Math.max(v.uRes, 0.02);
      const K = v.Kmult * (v.disp === 'okubo' ? okubo(b) : v.disp === 'const' ? v.K0 : v.K0 + 0.6 * Math.sqrt(v.Cd) * Math.hypot(u, 0.7 * waveOrbital(v.waveHeight, v.wavePeriod, P.depth)) * P.depth);
      return { zt: j.zt, xi: j.xi, Si: j.Si, dil1km: nf.Sn * brooks(1000, b, u, K) };
    },
    get sample() { return (this._s ||= synth(3, [[1500, 65, 60, 0.05], [2000, 65, 60, 0.08], [2500, 65, 60, 0.1], [2000, 60, 60, 0.15], [2000, 70, 60, 0.06], [3000, 66, 60, 0.12], [1200, 64, 45, 0.08], [2200, 67, 45, 0.2]])); },
    get validationSample() { return (this._v ||= synth(17, [[1800, 63, 60, 0.07], [2300, 68, 60, 0.11], [2800, 65, 60, 0.18], [1600, 66, 45, 0.09], [2100, 62, 60, 0.25], [2600, 69, 45, 0.05]])); },
  },

  async verify() {
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Number.isFinite(got) && Math.abs(got - expected) <= tol, note });
    add('Seawater density at 25 °C, 35 g/kg', 1023.3, density(25, 35), 0.3, 'Equation-of-state benchmark (kg/m³)');
    // dense 60° jet against the empirical coefficients
    const amb = () => ({ S: 37, T: 22, ua: 0 }), base = { d: 0.15, U0: 5, theta: 60 * D2R, Sb: 65, Tb: 22, amb };
    const j = denseJet(base), dF = base.d * j.F;
    add('60° dense jet: terminal rise height / (d·F)', ROBERTS60.zt, j.zt / dF, 0.2 * ROBERTS60.zt, 'Roberts, Ferrier & Daviero (1997): 2.2, within 20 %');
    add('60° dense jet: impact distance / (d·F)', ROBERTS60.xi, j.xi / dF, 0.2 * ROBERTS60.xi, 'Empirical 2.4, within 20 %');
    add('60° dense jet: impact dilution / F', ROBERTS60.Si, j.Si / j.F, 0.2 * ROBERTS60.Si, 'Empirical 1.6, within 20 %');
    const L = j.path.s.length - 1;
    add('Jet salt-excess flux is conserved', 1, (j.path.S[L] * (j.path.sal[L] - 37)) / (65 - 37), 1e-6, 'Q·(S − S_a) constant along the trajectory in a uniform ambient (ratio)');
    // Froude-number similarity: different size and density difference, same F
    const gp2 = (G * (density(22, 50) - density(22, 37))) / density(22, 37), d2 = 0.4, j2 = denseJet({ ...base, d: d2, Sb: 50, U0: j.F * Math.sqrt(gp2 * d2) });
    add('Froude-number scaling invariance: rise height', j.zt / dF, j2.zt / (d2 * j2.F), 0.01 * (j.zt / dF), 'Same z_t/(d·F) for a different port size, velocity and density difference');
    add('Froude-number scaling invariance: dilution', j.Si / j.F, j2.Si / j2.F, 0.01 * (j.Si / j.F), 'Same S_i/F');
    // non-buoyant jet: straight trajectory and the classical entrainment law
    const nb = denseJet({ ...base, Sb: 37, alphaJ: 0.08, alphaP: 0.117, sMax: 50 * 0.15 }), Ln = nb.path.s.length - 1;
    add('Non-buoyant jet: volume flux Q/Q₀ at 50 d', 1 + 0.32 * 50, nb.path.S[Ln], 0.02 * 17, 'Q/Q₀ = 1 + 4α s/d ≈ 0.32 s/d (Ricou & Spalding) for α = 0.08');
    add('Non-buoyant jet: straight trajectory', Math.tan(60 * D2R), nb.path.z[Ln] / nb.path.x[Ln], 1e-3, 'z/x = tan θ without buoyancy');
    add('Non-buoyant jet: centre-line decay C_c/C₀ · (s/d)', 5.3, (1 / nb.path.Sc[Ln]) * 50, 0.4, 'Classical range 5.0–5.6 for round turbulent jets');
    // far-field numerics on a flat basin
    const flatGrid = (nx, ny, d, H0, ring) => { const g = { nx, ny, dx: d, dy: d, x0: -nx * d / 2, y0: -ny * d / 2, H: new Float64Array(nx * ny).fill(H0), zb: new Float64Array(nx * ny).fill(-H0), io: nx >> 1, jo: ny >> 1 }; g.xs = Array.from({ length: nx }, (_, i) => g.x0 + (i + 0.5) * d); g.ys = Array.from({ length: ny }, (_, k) => g.y0 + (k + 0.5) * d); if (ring) for (let q = 0; q < nx * ny; q++) { const i = q % nx, k = (q - i) / nx; if (i < 2 || k < 2 || i >= nx - 2 || k >= ny - 2) g.H[q] = 0; else { g.zb[q] = -(H0 + 0.004 * g.ys[k]); g.H[q] = H0 + 0.004 * g.ys[k]; } } return g; };
    const still = { cons: [], axis: [1, 0], res: [0, 0], wind: [0, 0], series: null }, gauss = (g, s0, x0 = 0) => Float64Array.from({ length: g.nx * g.ny }, (_, q) => Math.exp(-((g.xs[q % g.nx] - x0) ** 2 + g.ys[(q - (q % g.nx)) / g.nx] ** 2) / (2 * s0 * s0)));
    const mom = (g, a) => { let m = 0, mx = 0, mxx = 0; for (let q = 0; q < a.length; q++) { const x = g.xs[q % g.nx]; m += a[q]; mx += a[q] * x; mxx += a[q] * x * x; } return { m, x: mx / m, var: mxx / m - (mx / m) ** 2 }; };
    const gc = flatGrid(40, 40, 50, 10, true), fc = flowBasis(gc, true), Kc = new Float64Array(1600).fill(2);
    const rc = await farField({ g: gc, flow: fc, cur: still, K: Kc, phi: 0.3, bedF: 0.7, tEnd: 6 * 3600, src: [{ P: 20 * 40 + 20, w: 1 }], rate: 5, drift: { betaS: 7.5e-4, Cd: 0.0025, vmax: 0.4 }, closed: true });
    add('Far field: salt mass conservation in a closed basin', 1, rc.bal.mass / rc.bal.injected, 1e-9, 'Stored excess ÷ injected excess on a sloping closed basin with dispersion and down-slope drift (ratio)');
    let ys = 0, ms = 0; for (let q = 0; q < 1600; q++) { ys += rc.C[q] * gc.H[q] * gc.ys[(q - (q % 40)) / 40]; ms += rc.C[q] * gc.H[q]; }
    add('Far field: dense layer drifts down-slope', 1, ys / ms > 20 ? 1 : 0, 0, `Plume centroid moved ${fmt(ys / ms, 3)} m toward deeper water from a source at the basin centre`);
    const go = flatGrid(80, 24, 50, 10, false), fo = flowBasis(go), K0 = new Float64Array(80 * 24), C0 = gauss(go, 200, -1000), m0 = mom(go, C0);
    const ra = await farField({ g: go, flow: fo, cur: { ...still, res: [0.4, 0] }, K: K0, tEnd: 4000, C0, cfl: 0.4 }), m1 = mom(go, ra.C);
    add('Far field: pure advection translates a blob by u·t', 0.4 * 4000, m1.x - m0.x, 10, 'Centroid displacement (m), uniform current 0.4 m/s for 4000 s, TVD scheme');
    add('Far field: advection preserves the blob (TVD)', 1, Math.max(...ra.C) / Math.max(...C0), 0.06, 'Peak after transport over 32 cells ÷ initial peak');
    add('Far field: uniform current field is reproduced', 0.4, 0.4 * fo.basis[0].u[12 * 80 + 5], 1e-6, 'Rigid-lid friction balance on a flat bed gives a uniform current (m/s)');
    const gd = flatGrid(60, 60, 50, 10, false), fd = flowBasis(gd), Kd = new Float64Array(3600).fill(5), D0 = gauss(gd, 200), v0 = mom(gd, D0);
    const rd = await farField({ g: gd, flow: fd, cur: still, K: Kd, tEnd: 4000, C0: D0 }), v1 = mom(gd, rd.C);
    add('Far field: pure diffusion spreads a Gaussian as σ² = σ₀² + 2Kt', 2 * 5 * 4000, v1.var - v0.var, 0.02 * 2 * 5 * 4000, 'Variance growth (m²), K = 5 m²/s');
    add('Far field: diffusion conserves mass', 1, v1.m / v0.m, 1e-9, 'Open boundaries not reached');
    // tidal harmonics and analytical far field
    const cu = { cons: [{ amp: 0.3, ph: 0.4, T: TIDES.M2 }], axis: bearing(90), res: [0, 0], wind: [0, 0], series: null };
    add('Tidal harmonic repeats after one M2 period', currentAt(cu, 5000)[0], currentAt(cu, 5000 + TIDES.M2 * 3600)[0], 1e-12, 'u(t + 12.4206 h) = u(t)');
    add('Brooks far-field dilution tends to 1 at the source', 1, brooks(1e-6, 50, 0.2, 0.05), 1e-6, 'Limit x → 0 of the line-source solution');
    const dg = designDiffuser({ Q: 2000 / 3600, gp: 0.21, depth: 12, z0: 1, dS0: 28, limit: 1.85 }).best;
    add('Auto-designed diffuser meets its criteria', 1, dg.ok && dg.V >= 4 && dg.V <= 6 && dg.F >= 20 ? 1 : 0, 0, `${dg.n} ports of ${fmt(dg.d * 1000, 3)} mm: velocity 4–6 m/s, F ≥ 20, jet below the surface`);
    return C;
  },
};

const HELP = {
  Sa: 'Depth-mean salinity of the receiving water.', Ta: 'Depth-mean temperature of the receiving water.', nPorts: 'Total number of discharge ports sharing the flow equally.', dPort: 'Internal diameter of each port (nozzle).', theta: '60° maximises the dilution of a dense jet; 30–45° suits shallow water.',
  phM2: 'Phase lag of the M2 current.', phS2: 'Phase lag of the S2 current.', phK1: 'Phase lag of the K1 current.', phO1: 'Phase lag of the O1 current.', resDir: 'Direction toward which the net drift flows.', windSpeed: 'Ten-metre wind speed.', wavePeriod: 'Peak wave period.',
  curSeries: 'Time in hours, speed in m/s, direction toward which the current flows.', outX: 'Moves the outfall within the imported bathymetry.', outY: 'Moves the outfall within the imported bathymetry.', inY: 'Negative values are toward the shore of the synthetic beach.',
  hLayer: 'Leave at 0 to use the density-current thickness computed 150 m beyond the near field.', disp: 'Elder scales with friction velocity and layer thickness; Okubo grows with distance from the outfall.', nPart: 'More particles give a smoother envelope.',
  vMin: 'Below about 4 m/s mixing weakens and ports foul.', vMax: 'Above about 6 m/s head loss and fish-entrainment risk rise.', Fmin: 'F = U/√(g′d); 20 or more keeps the jets fully turbulent and well mixed.', clear: 'Keeps the plume submerged and invisible at low tide.',
  nx: 'Aim for cells no larger than half the mixing-zone radius.', ny: 'Use a similar cell size in both directions.', Lx: 'Should contain the tidal excursion (current amplitude × 12.42 h ÷ π) on both sides of the outfall.', Ly: 'Should reach from the shoreline to well offshore of the plume.',
  fx: '50 % centres the outfall east–west.', fy: 'Leave room offshore: dense plumes drift down-slope.',
};
for (const f of suite.inputs.flatMap((g) => g.fields)) if (!f.help && HELP[f.key]) f.help = HELP[f.key];

/** Synthetic survey data: the fast model with different "true" coefficients plus deterministic noise. */
function synth(seed, pts) {
  const d = Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value])), g = rng(seed);
  return pts.map(([Qb, Sb, theta, uRes]) => {
    const m = suite.calibration.model({ ...d, design: 'manual', alphaJ: 0.078, descF: 1.75, Kmult: 1.8, Qb, Sb, theta, uRes });
    return { Qb, Sb, theta, uRes, zt: +(m.zt * (1 + g.normal(0, 0.03))).toFixed(2), xi: +(m.xi * (1 + g.normal(0, 0.03))).toFixed(2), Si: +(m.Si * (1 + g.normal(0, 0.05))).toFixed(1), dil1km: +(m.dil1km * (1 + g.normal(0, 0.06))).toFixed(0) };
  });
}

export default suite;
