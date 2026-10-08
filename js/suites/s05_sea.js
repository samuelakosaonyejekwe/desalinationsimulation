// Suite 5 — Brine discharge into the sea.
// Near field: integral model of inclined negatively buoyant jets (volume, momentum, salt and heat fluxes
// with a jet/plume/cross-flow entrainment closure) from single-port or multiport diffusers, cross-checked
// against the empirical dense-jet coefficients. Intermediate field: entraining bottom gravity current.
// Far field: transient advection–dispersion of excess salinity over real or synthetic bathymetry, driven
// by tidal-harmonic, residual and wind-drift currents whose spatial pattern follows a rigid-lid,
// friction-dominated shallow-water balance, or by a free-surface shallow-water solver (wetting–drying,
// Flather/elevation/radiation boundaries, wind stress, Coriolis). Optional wave-action model, vertical
// (x–z) hydrostatic/non-hydrostatic slice with turbulence closure, atmospheric heat exchange, ecological
// dose–response and seasonal sweep. Regulatory mixing-zone, receptor and intake assessment.
import { rk45, clamp, linspace, fmt, rng, interp1, mean } from '../core/num.js';
import { density, G, salinityFromTDS, cp as cpSea } from '../core/props.js';
import { pcg5, bandSolver } from './s04_cfd.js';

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
 * Explicit finite volumes, upwind or van Leer TVD fluxes, CFL-limited step, open or closed boundaries. The step is limited by the
 * tidal transport and the down-slope drift actually present; where that exceeds the reference step based on the drift-speed cap,
 * the limited flux is time-centred (Lax–Wendroff) for the excess, so the temporal truncation error stays that of the reference step.
 * Options: dyn = swCoupler(...) co-steps a shallowWater() solver and uses its time-mean face transports and its
 * moving depth, ∂D/∂t = −∇·(f_b q), so the scheme stays consistent with the moving free surface; extra = { rate, decay }
 * carries a second tracer (excess temperature) with a first-order surface-exchange decay (1/s per cell);
 * expThr = [thresholds] returns the share of the statistics window that each cell spends above each threshold.
 */
export async function farField(c, ctx) {
  const { g, flow, cur, K, phi = 1, bedF = 1, scheme = 'tvd', cfl = 0.5, tEnd, tStat = 0, src = [], rate = 0, probes = [], ring = [], drift = null, closed = false, thr = 0.1, particles = 0, dyn = null, extra = null, expThr = null } = c;
  const { nx, ny, dx, dy, H } = g, n = nx * ny, A = dx * dy, zero = () => ({ qx: new Float64Array((nx + 1) * ny), qy: new Float64Array(nx * (ny + 1)), rate: 0 });
  const bA = dyn ? zero() : flow.basis[0], bB = dyn ? zero() : flow.basis[1];
  const C = c.C0 ? Float64Array.from(c.C0) : new Float64Array(n), dC = new Float64Array(n), Cmax = new Float64Array(n), Csum = new Float64Array(n), Csnap = new Float64Array(n);
  const E = extra ? (extra.C0 ? Float64Array.from(extra.C0) : new Float64Array(n)) : null, dE = E ? new Float64Array(n) : null, Emax = E ? new Float64Array(n) : null, Esum = E ? new Float64Array(n) : null, lam = extra?.decay || null;
  const expo = expThr ? expThr.map(() => new Float64Array(n)) : null;
  let Kmax = 0, nExp = 0;
  for (let P = 0; P < n; P++) if (H[P] && K[P] > Kmax) Kmax = K[P];
  const dtDiff = Kmax > 0 ? 0.2 / (Kmax * (1 / (dx * dx) + 1 / (dy * dy))) : Infinity, tvd = scheme === 'tvd';
  // density-driven down-slope drift: u_g = √(g'·h_layer·|s| / 2C_d) directed down the bed gradient, g' = g β_S C
  const slopeAt = (P, i, j) => [i > 0 && i < nx - 1 && H[P - 1] && H[P + 1] ? (g.zb[P + 1] - g.zb[P - 1]) / (2 * dx) : 0, j > 0 && j < ny - 1 && H[P - nx] && H[P + nx] ? (g.zb[P + nx] - g.zb[P - nx]) / (2 * dy) : 0];
  const gCoef = (s, sOther, hf) => { const sm = Math.hypot(s, sOther); return drift && sm > 1e-6 ? (-s / Math.sqrt(sm)) * Math.sqrt((G * drift.betaS * phi * hf) / (2 * drift.Cd)) : 0; };
  const vmaxG = drift ? drift.vmax : 0;
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
    return { m: L.length, stride, L: Int32Array.from(L), U1: Int32Array.from(U1), U2: Int32Array.from(U2), Fq: Int32Array.from(Fq), Dc: Float64Array.from(Dc), Gc: Float64Array.from(Gc), Ga: Float64Array.from(Ga), qa: ax ? bA.qy : bA.qx, qb: ax ? bB.qy : bB.qx, fa: new Float64Array(L.length), fb: new Float64Array(L.length) };
  });
  const edge = []; // [cell, face index, axis, outward sign]
  if (!closed) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const P = j * nx + i;
    if (!H[P]) continue;
    if (i === 0) edge.push([P, j * (nx + 1), 0, -1]); if (i === nx - 1) edge.push([P, j * (nx + 1) + nx, 0, 1]);
    if (j === 0) edge.push([P, i, 1, -1]); if (j === ny - 1) edge.push([P, ny * nx + i, 1, 1]);
  }
  const inv = Float64Array.from(H, (h) => (h > 0 ? 1 / (h * A) : 0)), ivD = dyn ? new Float64Array(n) : null;
  // moving free surface (dyn): layer-equivalent depth D = H₀ + f_b (h − h₀) of the hydrodynamic cell, so that ∂D/∂t = −∇·(f_b q)
  const sw = dyn || null, H0 = sw ? Float64Array.from(H, (h, P) => (h > 0 ? Math.max(h + (sw.depth(P) > 0 ? sw.eta(P) : sw.eta0 ?? 0), 0.05) : 0)) : null, Do = sw ? Float64Array.from(H0) : null, Dn = sw ? Float64Array.from(H0) : null, Da = sw ? new Float64Array(n) : null, Db = sw ? new Float64Array(n) : null;
  // advective rate of the down-slope drift for the time-step limit: the largest drift speed on each axis found in the previous
  // step with a 25 % margin, never above the bound min(v_max, max|G_c|·√C_max) of the present field
  const gMax = tab.map((T) => { let m = 0; for (let k = 0; k < T.m; k++) { const a = Math.abs(T.Gc[k]); if (a > m) m = a; } return m; }), sq = drift ? new Float64Array(n) : null, ugSeen = [Infinity, Infinity];
  let cPeak = 0;
  for (let P = 0; P < n; P++) if (C[P] > cPeak) cPeak = C[P];
  const driftRate = () => { if (!drift) return 0; const r = Math.sqrt(cPeak); return Math.min(vmaxG, gMax[0] * r, 1.25 * ugSeen[0] + 1e-4) / dx + Math.min(vmaxG, gMax[1] * r, 1.25 * ugSeen[1] + 1e-4) / dy; };
  const gather = () => { for (const T of tab) { const { m, Fq, qa, qb, fa, fb } = T; for (let k = 0; k < m; k++) { fa[k] = qa[Fq[k]]; fb[k] = qb[Fq[k]]; } } }; // face transports in table order
  gather();
  const sampleOf = (F, x, y) => { // bilinear over wet cells
    const fi = (x - g.x0) / dx - 0.5, fj = (y - g.y0) / dy - 0.5, i = clamp(Math.floor(fi), 0, nx - 2), j = clamp(Math.floor(fj), 0, ny - 2), a = clamp(fi - i, 0, 1), b = clamp(fj - j, 0, 1), P = j * nx + i;
    let s = 0, w = 0;
    if (H[P]) { const wt = (1 - a) * (1 - b); s += wt * F[P]; w += wt; }
    if (H[P + 1]) { const wt = a * (1 - b); s += wt * F[P + 1]; w += wt; }
    if (H[P + nx]) { const wt = (1 - a) * b; s += wt * F[P + nx]; w += wt; }
    if (H[P + nx + 1]) { const wt = a * b; s += wt * F[P + nx + 1]; w += wt; }
    return w > 0 ? s / w : 0;
  };
  const sample = (x, y) => sampleOf(C, x, y);
  const cv = [0, 0], cellVel = (P) => { // velocity of the transported layer in cell P (shallow-water coupling)
    const i = P % nx, j = (P - i) / nx, hh = Math.max(H0[P] + (Dn[P] - H0[P]) / bedF, 0.2);
    cv[0] = (bedF * 0.5 * (bA.qx[j * (nx + 1) + i] + bA.qx[j * (nx + 1) + i + 1])) / (hh * dy); cv[1] = (bedF * 0.5 * (bA.qy[P] + bA.qy[P + nx])) / (hh * dx);
    return cv;
  };
  // particles (random walk with the drift correction ∇K + K ∇H/H of a depth-integrated layer): continuous release at the source
  const np = Math.round(particles), px = new Float64Array(np), py = new Float64Array(np), alive = new Uint8Array(np), tRel = new Float64Array(np), rn = np ? rng(99) : null;
  const pst = np ? { left: 0, mx: 0, my: 0, mxx: 0, myy: 0 } : null;
  let released = 0;
  const ser = { t: [], u: [], v: [], probes: probes.map(() => []), ring: [], extra: E ? probes.map(() => []) : null, eta: sw ? [] : null }, bal = { injected: 0, out: 0, injectedE: 0, outE: 0, lostE: 0 };
  /** One explicit transport step of length dts with current coefficients cA, cB (and depths a → b when the surface moves); stat = true adds the new field to the statistics; lwF = share of the step that is time-centred. */
  const advance = (dts, cA, cB, Da, Db, stat, lwF) => {
    const moving = !!Da, two = cB !== 0;
    dC.fill(0); if (E) dE.fill(0);
    if (sq) for (let P = 0; P < n; P++) sq[P] = Math.sqrt(C[P]);
    let iv = inv; // reciprocal volume of every cell at the start of the step
    if (moving && lwF > 0) { iv = ivD; for (let P = 0; P < n; P++) iv[P] = Da[P] > 0 ? 1 / (Da[P] * A) : 0; }
    for (let ax = 0; ax < 2; ax++) {
      const { m, stride, L, U1, U2, Dc, Gc, Ga, fa, fb } = tab[ax];
      let ugm = 0;
      for (let k = 0; k < m; k++) {
        const P = L[k], R = P + stride, cl = C[P], cr = C[R], f = two ? cA * fa[k] + cB * fb[k] : cA * fa[k];
        let cf, lw = 1; // lw = 1 − lwF·ν: Lax–Wendroff time-centring of the limited flux for the part of the step beyond the reference step (ν = Courant number of the donor cell)
        if (!tvd) { if (lwF > 0) { if (f >= 0) { lw = lwF * f * dts * iv[P]; cf = cl - 0.5 * lw * (cr - cl); } else { lw = -lwF * f * dts * iv[R]; cf = cr - 0.5 * lw * (cl - cr); } } else cf = f >= 0 ? cl : cr; } // upwind: keep the numerical diffusion ½|u|Δx(1 − ν) of the reference step
        else if (f >= 0) { cf = cl; if (U1[k] >= 0) { const d1 = cl - C[U1[k]], d2 = cr - cl; if (d1 * d2 > 0) { lw = 1 - lwF * f * dts * iv[P]; cf = cl + (lw > 0 ? lw : 0) * ((d1 * d2) / (d1 + d2)); } } } // van Leer limiter
        else { cf = cr; if (U2[k] >= 0) { const d1 = cr - C[U2[k]], d2 = cl - cr; if (d1 * d2 > 0) { lw = 1 + lwF * f * dts * iv[R]; cf = cr + (lw > 0 ? lw : 0) * ((d1 * d2) / (d1 + d2)); } } }
        let flux = f * cf - Dc[k] * (cr - cl), ugA = 0;
        if (sq && Gc[k] !== 0) { const up = Gc[k] > 0 ? P : R, cu = C[up]; if (cu > 0) { let ug = Gc[k] * sq[up]; if (ug > vmaxG) ug = vmaxG; else if (ug < -vmaxG) ug = -vmaxG; if (ug > ugm) ugm = ug; else if (-ug > ugm) ugm = -ug; ugA = ug * Ga[k]; flux += ugA * cu; } }
        dC[P] -= flux; dC[R] += flux;
        if (E) {
          const el = E[P], er = E[R];
          let ef;
          if (!tvd) ef = f >= 0 ? el - 0.5 * (lwF > 0 ? lw : 0) * (er - el) : er - 0.5 * (lwF > 0 ? lw : 0) * (el - er);
          else if (f >= 0) { ef = el; if (U1[k] >= 0) { const d1 = el - E[U1[k]], d2 = er - el; if (d1 * d2 > 0) { const le = 1 - lwF * f * dts * iv[P]; ef = el + (le > 0 ? le : 0) * ((d1 * d2) / (d1 + d2)); } } }
          else { ef = er; if (U2[k] >= 0) { const d1 = er - E[U2[k]], d2 = el - er; if (d1 * d2 > 0) { const le = 1 + lwF * f * dts * iv[R]; ef = er + (le > 0 ? le : 0) * ((d1 * d2) / (d1 + d2)); } } }
          const fe = f * ef - Dc[k] * (er - el) + ugA * (ugA > 0 ? el : er);
          dE[P] -= fe; dE[R] += fe;
        }
      }
      ugSeen[ax] = ugm;
    }
    for (let k = 0; k < edge.length; k++) { // open edge: outflow leaves the domain, inflow brings clean water
      const e = edge[k], P = e[0], fo = e[3] * (e[2] ? cA * bA.qy[e[1]] + cB * bB.qy[e[1]] : cA * bA.qx[e[1]] + cB * bB.qx[e[1]]);
      if (fo > 0) { dC[P] -= fo * C[P]; bal.out += fo * C[P] * dts * phi; if (E) { dE[P] -= fo * E[P]; bal.outE += fo * E[P] * dts * phi; } }
    }
    if (!moving) {
      for (let P = 0; P < n; P++) { const cn = C[P] + dts * dC[P] * inv[P]; C[P] = cn > 0 ? cn : 0; }
      for (const s of src) C[s.P] += (dts * rate * s.w) / (phi * H[s.P] * A);
      if (E) {
        for (let P = 0; P < n; P++) { let en = E[P] + dts * dE[P] * inv[P]; if (lam && lam[P] > 0) { const e2 = en / (1 + dts * lam[P]); bal.lostE += (en - e2) * phi * H[P] * A; en = e2; } E[P] = en; }
        for (const s of src) E[s.P] += (dts * extra.rate * s.w) / (phi * H[s.P] * A);
      }
    } else {
      for (let P = 0; P < n; P++) { if (!H[P] || !(Db[P] > 0.02)) continue; const cn = (Da[P] * C[P] + (dts * dC[P]) / A) / Db[P]; C[P] = cn > 0 ? cn : 0; }
      for (const s of src) C[s.P] += (dts * rate * s.w) / (phi * Math.max(Db[s.P], 0.02) * A);
      if (E) {
        for (let P = 0; P < n; P++) { if (!H[P] || !(Db[P] > 0.02)) continue; let en = (Da[P] * E[P] + (dts * dE[P]) / A) / Db[P]; if (lam && lam[P] > 0) { const e2 = en / (1 + dts * lam[P]); bal.lostE += (en - e2) * phi * Db[P] * A; en = e2; } E[P] = en; }
        for (const s of src) E[s.P] += (dts * extra.rate * s.w) / (phi * Math.max(Db[s.P], 0.02) * A);
      }
    }
    bal.injected += dts * rate * (src.length ? 1 : 0); if (E) bal.injectedE += dts * extra.rate * (src.length ? 1 : 0);
    // peak concentration (for the drift time-step limit) and, in the statistics window, envelope and running sum in one pass
    let cm = 0;
    if (stat) { for (let P = 0; P < n; P++) { const a = C[P]; if (a > cm) cm = a; if (a > Cmax[P]) Cmax[P] = a; Csum[P] += a; } if (E) for (let P = 0; P < n; P++) { const ev = E[P]; if (Math.abs(ev) > Math.abs(Emax[P])) Emax[P] = ev; Esum[P] += ev; } }
    else if (sq) for (let P = 0; P < n; P++) if (C[P] > cm) cm = C[P];
    cPeak = cm;
  };
  let t = 0, step = 0, nStat = 0, tSnap = 0, areaSnap = -1, lastSample = -Infinity, ringMax = 0, tRingMax = 0, rateDyn = 0, rateBulk = 0, subSteps = 0;
  const dtSample = tEnd / 360, maxSteps = c.maxSteps ?? 60000, gCap = drift ? vmaxG * (1 / dx + 1 / dy) : 0; // gCap: drift rate at the speed cap (reference step)
  while (t < tEnd - 1e-9 && step < maxSteps) {
    let ux = 1, uy = 0, dt, inStat;
    const gRate = driftRate();
    if (!sw) {
      const u2 = currentAt(cur, t); ux = u2[0]; uy = u2[1];
      const tide = bedF * (Math.abs(ux) * bA.rate + Math.abs(uy) * bB.rate), adv = tide + gRate, dtRef = Math.min(tide + gCap > 0 ? cfl / (tide + gCap) : Infinity, dtDiff, tEnd - t, tEnd / 40);
      dt = Math.min(adv > 0 ? cfl / adv : Infinity, dtDiff, tEnd - t, tEnd / 40);
      inStat = t + dt >= tStat;
      advance(dt, bedF * ux, bedF * uy, null, null, inStat, dt > dtRef ? 1 - dtRef / dt : 0);
    } else {
      // explicit hydrodynamics: step from the bulk cells, thin (drying) cells are sub-cycled. Semi-implicit hydrodynamics: one
      // hydrodynamic step at its own (advective) limit, and as many transport sub-steps on its time-mean transports as the tracer needs
      const dtH = sw.dtHydro ? sw.dtHydro() : 0;
      dt = dtH > 0 ? Math.min(dtH, 150, Math.max(tEnd - t, 1e-6), tEnd / 40) : Math.min(rateBulk + gRate > 0 ? cfl / (rateBulk + gRate) : 30, dtDiff, Math.max(tEnd - t, 1e-6), tEnd / 40);
      Do.set(Dn);
      sw.advance(dt, bA.qx, bA.qy);
      const hc = sw.sw.h, hc0 = sw.h0, mp = sw.map, lc = sw.sw.land, cR = (0.5 * bedF) / A;
      rateDyn = 0; rateBulk = 0;
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0, a = j * (nx + 1); i < nx; i++, P++, a++) {
        if (!H[P]) continue;
        const Q = mp[P];
        if (!lc[Q]) { const d = H0[P] + bedF * (hc[Q] - hc0[Q]); Dn[P] = d > 0.02 ? d : 0.02; }
        const dm = Do[P] < Dn[P] ? Do[P] : Dn[P];
        if (dm > 0.02) { const r = (cR * (Math.abs(bA.qx[a]) + Math.abs(bA.qx[a + 1]) + Math.abs(bA.qy[P]) + Math.abs(bA.qy[P + nx]))) / dm; if (r > rateDyn) rateDyn = r; if (dm > 0.3 && r > rateBulk) rateBulk = r; }
      }
      gather();
      inStat = t + dt >= tStat;
      const ns = dtH > 0 ? clamp(Math.ceil(dt * Math.max((rateDyn + gRate) / Math.max(cfl, 0.05), 1 / dtDiff)), 1, 80) : clamp(Math.ceil((dt * (rateDyn + gRate)) / Math.max(cfl, 0.05)), 1, 40);
      const dts = dt / ns, dtRef = dtH > 0 ? cfl / (rateDyn + gCap + 1e-300) : Math.min(rateBulk + gCap > 0 ? cfl / (rateBulk + gCap) : 30, dtDiff) / ns, lwF = dts > dtRef ? 1 - dtRef / dts : 0;
      if (ns === 1) advance(dt, bedF, 0, Do, Dn, inStat, lwF);
      else { const rn = 1 / ns; for (let s = 0; s < ns; s++) { const w0 = s * rn, w1 = (s + 1) * rn; for (let P = 0; P < n; P++) { const d0 = Do[P], dd = Dn[P] - d0; Da[P] = d0 + dd * w0; Db[P] = d0 + dd * w1; } advance(dts, bedF, 0, Da, Db, inStat && s === ns - 1, lwF); } }
      subSteps += ns;
    }
    t += dt; step++;
    if (np) {
      const want = Math.min(np, Math.floor((t / tEnd) * np) + 1);
      while (released < want) { px[released] = c.srcXY[0] + c.srcR * (rn.uniform() - 0.5); py[released] = c.srcXY[1] + c.srcR * (rn.uniform() - 0.5); tRel[released] = t; alive[released++] = 1; }
      for (let k = 0; k < released; k++) {
        if (!alive[k]) continue;
        const i = Math.floor((px[k] - g.x0) / dx), j = Math.floor((py[k] - g.y0) / dy), P = j * nx + i, Kp = K[P], sd = Math.sqrt(6 * Kp * dt);
        // drift correction of the random walk for a depth-integrated layer: (∇(H K)) / H, central differences over wet neighbours
        const hE = i < nx - 1 && H[P + 1] ? P + 1 : P, hW = i > 0 && H[P - 1] ? P - 1 : P, hN = j < ny - 1 && H[P + nx] ? P + nx : P, hS = j > 0 && H[P - nx] ? P - nx : P;
        const cx = hE !== hW ? (H[hE] * K[hE] - H[hW] * K[hW]) / ((hE - hW) * dx * H[P]) : 0, cy = hN !== hS ? (H[hN] * K[hN] - H[hS] * K[hS]) / (((hN - hS) / nx) * dy * H[P]) : 0;
        let vx, vy;
        if (!sw) { vx = bedF * (ux * bA.u[P] + uy * bB.u[P]); vy = bedF * (ux * bA.v[P] + uy * bB.v[P]); } else { const q = cellVel(P); vx = q[0]; vy = q[1]; }
        const xn = px[k] + (vx + cx) * dt + sd * (2 * rn.uniform() - 1), yn = py[k] + (vy + cy) * dt + sd * (2 * rn.uniform() - 1);
        const i2 = Math.floor((xn - g.x0) / dx), j2 = Math.floor((yn - g.y0) / dy);
        if (i2 < 0 || i2 >= nx || j2 < 0 || j2 >= ny) { alive[k] = 0; pst.left++; continue; }
        if (H[j2 * nx + i2]) { px[k] = xn; py[k] = yn; }
      }
    }
    if (inStat) nStat++;
    if (t - lastSample >= dtSample || t >= tEnd - 1e-9) {
      lastSample = t;
      let rm = 0;
      for (const [x, y] of ring) rm = Math.max(rm, sample(x, y));
      if (sw) { const Po = g.jo * nx + g.io, q = cellVel(Po); ux = q[0] / bedF; uy = q[1] / bedF; ser.eta.push(sw.depth(Po) > 0 ? sw.eta(Po) : (sw.eta0 ?? 0) + (Dn[Po] - H0[Po]) / bedF); }
      ser.t.push(t / 3600); ser.u.push(ux); ser.v.push(uy); ser.ring.push(rm); probes.forEach((p, k) => { ser.probes[k].push(sample(p[0], p[1])); if (E) ser.extra[k].push(sampleOf(E, p[0], p[1])); });
      if (inStat) {
        if (rm > ringMax) { ringMax = rm; tRingMax = t; }
        let area = 0;
        for (let P = 0; P < n; P++) if (C[P] > thr) area++;
        if (area > areaSnap) { areaSnap = area; tSnap = t; Csnap.set(C); }
        if (expo) { nExp++; expThr.forEach((th, k) => { const a = expo[k]; for (let P = 0; P < n; P++) if (C[P] > th) a[P]++; }); }
      }
      if (ctx?.progress) ctx.progress(0.12 + (0.83 * t) / tEnd, `Far field: ${(t / 3600).toFixed(1)} h of ${(tEnd / 3600).toFixed(1)} h`);
      if (ctx?.tick) await ctx.tick();
    }
    if (!Number.isFinite(C[g.jo * nx + g.io])) throw new Error('The far-field solution became unstable — lower the CFL number.');
  }
  let mass = 0, massE = 0;
  for (let P = 0; P < n; P++) { const D = sw ? (H[P] ? Dn[P] : 0) : H[P]; mass += phi * D * C[P] * A; if (E) { massE += phi * D * E[P] * A; Esum[P] = nStat ? Esum[P] / nStat : E[P]; if (!nStat) Emax[P] = E[P]; } Csum[P] = nStat ? Csum[P] / nStat : C[P]; if (!nStat) Cmax[P] = C[P]; }
  if (areaSnap < 0) Csnap.set(C);
  if (expo) for (const a of expo) for (let P = 0; P < n; P++) a[P] = nExp ? a[P] / nExp : 0;
  const part = np ? { x: [], y: [], age: [], released, left: pst.left } : null;
  if (part) for (let k = 0; k < released; k++) if (alive[k]) { part.x.push(px[k]); part.y.push(py[k]); part.age.push(t - tRel[k]); }
  return { C, Cmax, Cmean: Csum, Csnap, tSnap, ser, bal: { ...bal, mass, massE }, steps: step, tEnd: t, ringMax, tRingMax, dtMean: step ? t / step : 0, part, complete: t >= tEnd - 1e-6, E, Emax, Emean: Esum, expo, subSteps };
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
// Free-surface shallow-water solver
// ---------------------------------------------------------------------------------------------------
const OMEGA_E = 7.2921e-5, RHO_AIR = 1.22;
/** Kinematic wind stress τ/ρ_w (m²/s², east and north) from the 10 m wind (blowing from the bearing dirFrom), Smith–Banke drag law. */
export function windStress(W, dirFrom, rhoW = 1025) {
  const cd = (0.63 + 0.066 * clamp(W, 0, 30)) * 1e-3, [ex, ey] = bearing(dirFrom + 180), t = (RHO_AIR * cd * W * W) / rhoW;
  return [t * ex, t * ey, cd];
}

/**
 * Depth-averaged (2-D) shallow-water equations with a moving free surface on an Arakawa C grid:
 *   ∂η/∂t + ∇·(h u) = 0 (finite-volume, upwind face depth → exact volume conservation, non-negative depths),
 *   ∂u/∂t + u·∇u − f v = −g ∂η/∂x − c_f |u| u / h + (τ_wind + F_wave)/(ρ h)   (and likewise for v),
 * forward–backward time stepping, semi-implicit bed friction (constant drag coefficient, Manning or Chézy),
 * wetting and drying with a minimum depth, Coriolis, wind stress and wave (radiation-stress) forcing.
 * implicit = true switches to the semi-implicit θ scheme (Casulli 1990): the surface gradient in the momentum
 * equations and the velocity in the continuity equation are taken at θ·(n+1) + (1 − θ)·n, which gives one symmetric
 * positive-definite 5-point system for the new elevation per step; the step is then limited by the current speed
 * (advective Courant number), not by the gravity-wave speed. Fluxes are limited to the water a cell holds, so depths
 * stay non-negative, and the elevation is updated from the final fluxes, so volume is conserved to round-off.
 * Open sides: 'flather' (u_n = u_ext ± √(g/h)(η − η_ext)), 'elev' (clamped tidal elevation), 'rad' (radiation of
 * outgoing waves to a still exterior) or 'wall'. ext(t) → { e, gx, gy, U, V }: external elevation e + gx·x + gy·y and
 * external current (U, V) multiplied by the optional spatial patterns pat = { au, av, bu, bv }.
 */
export function shallowWater(o) {
  const { nx, ny, dx, dy, zb } = o, n = nx * ny, nu1 = nx + 1, A = dx * dy, hmin = o.hmin ?? 0.05, f = o.f || 0, land = o.land || new Uint8Array(n);
  const fr = o.fric || { type: 'cd', Cd: 0.0025 }, bc = { W: 'wall', E: 'wall', S: 'wall', N: 'wall', ...(o.bc || {}) }, x0 = o.x0 ?? 0, y0 = o.y0 ?? 0, pat = o.pat || null, fx = o.force?.fx || null, fy = o.force?.fy || null, fxu = o.force?.fxu || null, fyv = o.force?.fyv || null, adv = o.advect !== false, qsrc = o.qsrc || null; // fxu, fyv: optional depth-integrated forces given directly on the u and v faces; qsrc: volume source per cell (m³/s, semi-implicit scheme)
  const eta = new Float64Array(n), h = new Float64Array(n), u = new Float64Array(nu1 * ny), v = new Float64Array(nx * (ny + 1));
  let un = new Float64Array(u.length), vn = new Float64Array(v.length), uc = u, vc = v;
  const Fx = new Float64Array(u.length), Fy = new Float64Array(v.length), aX = new Float64Array(u.length), aY = new Float64Array(v.length), mu = new Uint8Array(u.length), mv = new Uint8Array(v.length);
  for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) mu[j * nu1 + i] = land[j * nx + i - 1] || land[j * nx + i] ? 0 : 1;
  for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) mv[j * nx + i] = land[(j - 1) * nx + i] || land[j * nx + i] ? 0 : 1;
  const e0 = o.eta0 ?? 0;
  for (let P = 0; P < n; P++) { const e = typeof e0 === 'function' ? e0(P % nx, (P - (P % nx)) / nx) : e0; eta[P] = land[P] ? zb[P] : Math.max(e, zb[P]); h[P] = eta[P] - zb[P]; }
  const cfOf = fr.type === 'manning' ? (hh) => (G * fr.n * fr.n) / Math.cbrt(hh) : fr.type === 'chezy' ? () => G / (fr.C * fr.C) : () => fr.Cd;
  const S = { nx, ny, dx, dy, eta, h, land, t: 0, steps: 0, volIn: 0, volClamp: 0, volSrc: 0, capped: 0, implicit: !!o.implicit, get u() { return uc; }, get v() { return vc; }, get solverIters() { return I ? I.iters : 0; } };
  const etaExt = (E, i, j) => E.e + E.gx * (x0 + (i + 0.5) * dx) + E.gy * (y0 + (j + 0.5) * dy);
  const still = { e: typeof e0 === 'number' ? e0 : 0, gx: 0, gy: 0, U: 0, V: 0 };
  const open = (s) => bc[s] !== 'wall';
  S.volume = () => { let s = 0; for (let P = 0; P < n; P++) s += h[P]; return s * A; };
  S.dtStable = () => {
    let m = 1e-12; const q = Math.sqrt(1 / (dx * dx) + 1 / (dy * dy));
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (!(h[P] > hmin)) continue; const r = Math.sqrt(G * h[P]) * q + Math.max(Math.abs(uc[j * nu1 + i]), Math.abs(uc[j * nu1 + i + 1])) / dx + Math.max(Math.abs(vc[P]), Math.abs(vc[P + nx])) / dy; if (r > m) m = r; }
    return (o.cfl ?? 0.9) / m; // forward–backward stability limit: c Δt √(1/Δx² + 1/Δy²) ≤ 1
  };
  const oW = open('W'), oE = open('E'), oS = open('S'), oN = open('N'), manning = fr.type === 'manning', cfC = manning ? 0 : cfOf(1), gn2 = manning ? G * fr.n * fr.n : 0, gdx = G / dx, gdy = G / dy, rdx = 1 / dx, rdy = 1 / dy;
  const imp = !!o.implicit, th = clamp(o.theta ?? 0.55, 0.5, 1), f64 = (m) => new Float64Array(m);
  const I = imp ? { us: f64(u.length), gu: f64(u.length), Hx: f64(u.length), vs: f64(v.length), gv: f64(v.length), Hy: f64(v.length), aE: f64(n), aN: f64(n), dg: f64(n), rhs: f64(n), en: f64(n), sc: f64(n), fix: new Uint8Array(n), bs: bandSolver(nx, ny, 9e6, 6), iters: 0, eo: f64(n), dtOld: 0 } : null;
  if (imp) {
    let hm = 0; for (let P = 0; P < n; P++) if (h[P] > hm) hm = h[P];
    S.dtMax = o.dtMax ?? (20 * Math.min(dx, dy)) / Math.sqrt(G * Math.max(hm, 1)); // gravity-wave Courant number of 20 at most
    S.dtStable = () => { // advective limit
      let m = 1e-12;
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (!(h[P] > hmin)) continue; const r = Math.max(Math.abs(uc[j * nu1 + i]), Math.abs(uc[j * nu1 + i + 1])) / dx + Math.max(Math.abs(vc[P]), Math.abs(vc[P + nx])) / dy; if (r > m) m = r; }
      return Math.min((o.cflU ?? 0.7) / m, S.dtMax);
    };
  }
  /** One semi-implicit step of length dt. */
  const stepImp = (dt, acc) => {
    const u = uc, v = vc, tw = typeof o.tau === 'function' ? o.tau(S.t) : o.tau || [0, 0], E = o.ext ? o.ext(S.t + dt) : still, twx = tw[0], twy = tw[1];
    const { us, gu, Hx, vs, gv, Hy, aE, aN, dg, rhs, en, sc, fix } = I, k = (th * dt) / A, dA = dt / A, gth = th * dt * G, g1 = (1 - th) * dt * G, t1 = 1 - th;
    // 1. explicit part of the momentum equations: u′ = us − gu (η′_R − η′_L); face depth from the upwind level above the higher bed
    for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) {
      const q = j * nu1 + i;
      Hx[q] = 0; gu[q] = 0; us[q] = 0;
      if (!mu[q]) continue;
      const R = j * nx + i, L = R - 1, eL = eta[L], eR = eta[R], zf = zb[L] > zb[R] ? zb[L] : zb[R];
      let uq = u[q];
      const e = uq > 0 ? eL : uq < 0 ? eR : eL > eR ? eL : eR, hf = e - zf;
      if (hf <= hmin) { uq = 0; u[q] = 0; } else Hx[q] = hf * dy;
      if ((eL > eR ? eL : eR) - zf <= hmin) continue;
      const vb = 0.25 * (v[L] + v[R] + v[L + nx] + v[R + nx]);
      let hb = 0.5 * (h[L] + h[R]); if (hb < hmin) hb = hmin;
      let a = f * vb;
      if (twx !== 0 || fx) a += (twx + (fx ? 0.5 * (fx[L] + fx[R]) : 0)) / (hb > 0.2 ? hb : 0.2);
      if (fxu) a += fxu[q] / (hb > 0.2 ? hb : 0.2);
      if (adv) a -= uq * (uq > 0 ? uq - u[q - 1] : u[q + 1] - uq) * rdx + vb * (vb > 0 ? (j > 0 ? uq - u[q - nu1] : 0) : j < ny - 1 ? u[q + nu1] - uq : 0) * rdy;
      const cf = manning ? gn2 / Math.cbrt(hb) : cfC, den = 1 / (1 + (dt * cf * Math.sqrt(uq * uq + vb * vb)) / hb);
      us[q] = (uq + dt * a - g1 * rdx * (eR - eL)) * den; gu[q] = gth * rdx * den;
    }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = j * nx + i;
      Hy[q] = 0; gv[q] = 0; vs[q] = 0;
      if (!mv[q]) continue;
      const L = q - nx, eL = eta[L], eR = eta[q], zf = zb[L] > zb[q] ? zb[L] : zb[q];
      let vq = v[q];
      const e = vq > 0 ? eL : vq < 0 ? eR : eL > eR ? eL : eR, hf = e - zf;
      if (hf <= hmin) { vq = 0; v[q] = 0; } else Hy[q] = hf * dx;
      if ((eL > eR ? eL : eR) - zf <= hmin) continue;
      const r = j * nu1 + i, ub = 0.25 * (u[r] + u[r + 1] + u[r - nu1] + u[r - nu1 + 1]);
      let hb = 0.5 * (h[L] + h[q]); if (hb < hmin) hb = hmin;
      let a = -f * ub;
      if (twy !== 0 || fy) a += (twy + (fy ? 0.5 * (fy[L] + fy[q]) : 0)) / (hb > 0.2 ? hb : 0.2);
      if (fyv) a += fyv[q] / (hb > 0.2 ? hb : 0.2);
      if (adv) a -= ub * (ub > 0 ? (i > 0 ? vq - v[q - 1] : 0) : i < nx - 1 ? v[q + 1] - vq : 0) * rdx + vq * (vq > 0 ? vq - v[q - nx] : v[q + nx] - vq) * rdy;
      const cf = manning ? gn2 / Math.cbrt(hb) : cfC, den = 1 / (1 + (dt * cf * Math.sqrt(vq * vq + ub * ub)) / hb);
      vs[q] = (vq + dt * a - g1 * rdy * (eR - eL)) * den; gv[q] = gth * rdy * den;
    }
    // 2. elevation system: η′_P + k Σ H g_f (η′_P − η′_nb) = η_P − (Δt/A) Σ H (θ u* + (1 − θ) uⁿ) + open-boundary terms
    for (let P = 0; P < n; P++) { dg[P] = land[P] ? 0 : 1; rhs[P] = eta[P]; aE[P] = 0; aN[P] = 0; fix[P] = 0; en[P] = eta[P]; }
    const pin = (P, i, j) => { if (!land[P]) { fix[P] = 1; en[P] = Math.max(etaExt(E, i, j), zb[P]); } };
    if (bc.W === 'elev') for (let j = 0; j < ny; j++) pin(j * nx, 0, j);
    if (bc.E === 'elev') for (let j = 0; j < ny; j++) pin(j * nx + nx - 1, nx - 1, j);
    if (bc.S === 'elev') for (let i = 0; i < nx; i++) pin(i, i, 0);
    if (bc.N === 'elev') for (let i = 0; i < nx; i++) pin((ny - 1) * nx + i, i, ny - 1);
    // interior faces: implicit weight w between the two cells and the explicit volume flux fex (clamped cells move to the right-hand side)
    for (let j = 0; j < ny; j++) for (let i = 1, q = j * nu1 + 1, R = j * nx + 1; i < nx; i++, q++, R++) {
      const H = Hx[q];
      if (!(H > 0)) continue;
      const L = R - 1, fex = dA * H * (th * us[q] + t1 * u[q]), w = k * H * gu[q];
      rhs[L] -= fex; rhs[R] += fex;
      if (fix[L] | fix[R]) { if (!fix[R]) { dg[R] += w; rhs[R] += w * en[L]; } else if (!fix[L]) { dg[L] += w; rhs[L] += w * en[R]; } }
      else { dg[L] += w; dg[R] += w; aE[L] = w; }
    }
    for (let q = nx; q < ny * nx; q++) {
      const H = Hy[q];
      if (!(H > 0)) continue;
      const L = q - nx, fex = dA * H * (th * vs[q] + t1 * v[q]), w = k * H * gv[q];
      rhs[L] -= fex; rhs[q] += fex;
      if (fix[L] | fix[q]) { if (!fix[q]) { dg[q] += w; rhs[q] += w * en[L]; } else if (!fix[L]) { dg[L] += w; rhs[L] += w * en[q]; } }
      else { dg[L] += w; dg[q] += w; aN[L] = w; }
    }
    // open sides: outward flux H_b (s·u_ext + c (η′_P − η_ext)) of a Flather or radiation boundary, c = √(g/h)
    const rim = (s, P, i, j, q, arr, sgn, len, isU) => {
      const t = bc[s];
      if (t !== 'flather' && t !== 'rad') return;
      if (land[P] || !(h[P] > hmin)) { arr[q] = 0; return; }
      const X = t === 'rad' ? still : E, ue = t === 'rad' ? 0 : isU ? X.U * (pat ? pat.au[P] : 1) + X.V * (pat ? pat.bu[P] : 0) : X.U * (pat ? pat.av[P] : 0) + X.V * (pat ? pat.bv[P] : 1);
      const c = Math.sqrt(G / (h[P] > 0.3 ? h[P] : 0.3)), Hb = h[P] * len;
      dg[P] += k * Hb * c; rhs[P] -= k * Hb * (sgn * ue - c * etaExt(X, i, j)) + t1 * dA * sgn * Hb * arr[q];
    };
    for (let j = 0; j < ny; j++) { rim('W', j * nx, 0, j, j * nu1, u, -1, dy, true); rim('E', j * nx + nx - 1, nx - 1, j, j * nu1 + nx, u, 1, dy, true); }
    for (let i = 0; i < nx; i++) { rim('S', i, i, 0, i, v, -1, dx, false); rim('N', (ny - 1) * nx + i, i, ny - 1, ny * nx + i, v, 1, dx, false); }
    if (qsrc) for (let P = 0; P < n; P++) if (qsrc[P] !== 0 && !land[P]) rhs[P] += dA * qsrc[P];
    for (let P = 0; P < n; P++) if (fix[P]) { dg[P] = 1; rhs[P] = en[P]; }
    // solve to a given fraction of the residual of the old elevation, starting from the elevation extrapolated in time
    // (η is recomputed from the fluxes below, so the tolerance does not affect volume conservation)
    const etol = o.etaTol ?? 1e-6;
    if (I.bs) {
      const ref = I.bs.residual(aE, aN, dg, rhs, en);
      if (I.dtOld > 0) { const w = dt / I.dtOld; for (let P = 0; P < n; P++) if (!fix[P] && !land[P] && h[P] > hmin) { const g = eta[P] + w * (eta[P] - I.eo[P]); en[P] = g > zb[P] ? g : zb[P]; } }
      I.iters += I.bs.solve(aE, aN, dg, rhs, en, 0, 200, false, etol * ref).iters;
    } else I.iters += pcg5(nx, ny, aE, aN, dg, rhs, en, etol, 600).iters;
    I.eo.set(eta); I.dtOld = dt;
    // 3. new velocities and the volume fluxes of the step, θ F′ + (1 − θ) Fⁿ
    for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) {
      const q = j * nu1 + i;
      if (!mu[q] || gu[q] === 0) { un[q] = 0; Fx[q] = 0; continue; }
      const R = j * nx + i;
      let w = us[q] - gu[q] * (en[R] - en[R - 1]);
      if (w > 2 || w < -2) { let hb = 0.5 * (h[R - 1] + h[R]); if (hb < hmin) hb = hmin; const cap = 3 * Math.sqrt(G * hb) + 0.5; if (w > cap) { w = cap; S.capped++; } else if (w < -cap) { w = -cap; S.capped++; } }
      un[q] = w; Fx[q] = Hx[q] * (th * w + t1 * u[q]);
    }
    for (let q = nx; q < ny * nx; q++) {
      if (!mv[q] || gv[q] === 0) { vn[q] = 0; Fy[q] = 0; continue; }
      let w = vs[q] - gv[q] * (en[q] - en[q - nx]);
      if (w > 2 || w < -2) { let hb = 0.5 * (h[q - nx] + h[q]); if (hb < hmin) hb = hmin; const cap = 3 * Math.sqrt(G * hb) + 0.5; if (w > cap) { w = cap; S.capped++; } else if (w < -cap) { w = -cap; S.capped++; } }
      vn[q] = w; Fy[q] = Hy[q] * (th * w + t1 * v[q]);
    }
    const rimV = (s, P, i, j, q, arr, old, Fa, sgn, len, isU) => {
      const t = bc[s];
      if ((t !== 'flather' && t !== 'rad') || land[P] || !(h[P] > hmin)) { arr[q] = 0; Fa[q] = 0; return; }
      const X = t === 'rad' ? still : E, ue = t === 'rad' ? 0 : isU ? X.U * (pat ? pat.au[P] : 1) + X.V * (pat ? pat.bu[P] : 0) : X.U * (pat ? pat.av[P] : 0) + X.V * (pat ? pat.bv[P] : 1);
      arr[q] = ue + sgn * Math.sqrt(G / (h[P] > 0.3 ? h[P] : 0.3)) * (en[P] - etaExt(X, i, j)); Fa[q] = h[P] * len * (th * arr[q] + t1 * old[q]);
    };
    for (let j = 0; j < ny; j++) { rimV('W', j * nx, 0, j, j * nu1, un, u, Fx, -1, dy, true); rimV('E', j * nx + nx - 1, nx - 1, j, j * nu1 + nx, un, u, Fx, 1, dy, true); }
    for (let i = 0; i < nx; i++) { rimV('S', i, i, 0, i, vn, v, Fy, -1, dx, false); rimV('N', (ny - 1) * nx + i, i, ny - 1, ny * nx + i, vn, v, Fy, 1, dx, false); }
    // 4. a cell cannot give more water than it holds: scale the outgoing fluxes of over-drawn cells
    let lim = false;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0, q = j * nu1; i < nx; i++, P++, q++) {
      if (land[P]) { sc[P] = 1; continue; }
      const out = dt * ((Fx[q + 1] > 0 ? Fx[q + 1] : 0) - (Fx[q] < 0 ? Fx[q] : 0) + (Fy[P + nx] > 0 ? Fy[P + nx] : 0) - (Fy[P] < 0 ? Fy[P] : 0)), av = h[P] * A;
      if (out > av) { sc[P] = out > 0 ? av / out : 0; lim = true; } else sc[P] = 1;
    }
    if (lim) {
      for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) { const q = j * nu1 + i, F = Fx[q]; if (F > 0) { if (i > 0) Fx[q] = F * sc[j * nx + i - 1]; } else if (F < 0 && i < nx) Fx[q] = F * sc[j * nx + i]; }
      for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) { const q = j * nx + i, F = Fy[q]; if (F > 0) { if (j > 0) Fy[q] = F * sc[q - nx]; } else if (F < 0 && j < ny) Fy[q] = F * sc[q]; }
    }
    // 5. clamped-elevation cells: the boundary flux that brings the cell to the prescribed level
    if (bc.W === 'elev' || bc.E === 'elev' || bc.S === 'elev' || bc.N === 'elev') {
      const cnt = (P, i, j) => (i === 0 && bc.W === 'elev' ? 1 : 0) + (i === nx - 1 && bc.E === 'elev' ? 1 : 0) + (j === 0 && bc.S === 'elev' ? 1 : 0) + (j === ny - 1 && bc.N === 'elev' ? 1 : 0);
      const todo = [];
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (!fix[P]) continue; const m = cnt(P, i, j); if (i === 0 && bc.W === 'elev') todo.push([P, j * nu1, Fx, -1, 1 / m]); if (i === nx - 1 && bc.E === 'elev') todo.push([P, j * nu1 + nx, Fx, 1, 1 / m]); if (j === 0 && bc.S === 'elev') todo.push([P, i, Fy, -1, 1 / m]); if (j === ny - 1 && bc.N === 'elev') todo.push([P, ny * nx + i, Fy, 1, 1 / m]); }
      const need = todo.map(([P]) => { const i = P % nx, r = ((P - i) / nx) * nu1 + i; return ((eta[P] - en[P]) * A) / dt - (Fx[r + 1] - Fx[r] + Fy[P + nx] - Fy[P]); });
      todo.forEach(([P, q, Fa, sgn, share], m) => { Fa[q] += sgn * need[m] * share; (Fa === Fx ? un : vn)[q] = h[P] > hmin ? Fa[q] / (h[P] * (Fa === Fx ? dy : dx)) : 0; });
    }
    // 6. continuity with the final fluxes
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0, q = j * nu1; i < nx; i++, P++, q++) {
      if (land[P]) continue;
      let e = eta[P] - dA * (Fx[q + 1] - Fx[q] + Fy[P + nx] - Fy[P]);
      if (qsrc && qsrc[P] !== 0) { e += dA * qsrc[P]; S.volSrc += dt * qsrc[P]; }
      if (e < zb[P]) { S.volClamp += (zb[P] - e) * A; e = zb[P]; }
      eta[P] = e; h[P] = e - zb[P];
    }
    for (let j = 0; j < ny; j++) S.volIn += dt * (Fx[j * nu1] - Fx[j * nu1 + nx]);
    for (let i = 0; i < nx; i++) S.volIn += dt * (Fy[i] - Fy[ny * nx + i]);
    if (acc) { for (let q = 0; q < Fx.length; q++) aX[q] += Fx[q]; for (let q = 0; q < Fy.length; q++) aY[q] += Fy[q]; }
    const tu = uc; uc = un; un = tu; const tv = vc; vc = vn; vn = tv;
    S.t += dt; S.steps++;
  };
  /** One step of length dt. acc = true accumulates the face transports for the transport model. */
  S.step = imp ? stepImp : (dt, acc) => {
    const u = uc, v = vc, tw = typeof o.tau === 'function' ? o.tau(S.t) : o.tau || [0, 0], E = o.ext ? o.ext(S.t + dt) : still, twx = tw[0], twy = tw[1];
    // 1. face transports with the upwind surface level above the higher of the two beds
    for (let j = 0; j < ny; j++) {
      const r = j * nu1, c = j * nx;
      Fx[r] = oW && h[c] > hmin ? h[c] * u[r] * dy : 0; Fx[r + nx] = oE && h[c + nx - 1] > hmin ? h[c + nx - 1] * u[r + nx] * dy : 0;
      for (let i = 1; i < nx; i++) {
        const q = r + i;
        if (!mu[q]) continue;
        const R = c + i, L = R - 1, uq = u[q], zf = zb[L] > zb[R] ? zb[L] : zb[R], e = uq > 0 ? eta[L] : uq < 0 ? eta[R] : eta[L] > eta[R] ? eta[L] : eta[R], hf = e - zf;
        if (hf <= hmin) { Fx[q] = 0; u[q] = 0; } else Fx[q] = hf * uq * dy;
      }
    }
    for (let i = 0; i < nx; i++) { Fy[i] = oS && h[i] > hmin ? h[i] * v[i] * dx : 0; const P = (ny - 1) * nx + i; Fy[P + nx] = oN && h[P] > hmin ? h[P] * v[P + nx] * dx : 0; }
    for (let q = nx; q < ny * nx; q++) {
      if (!mv[q]) continue;
      const L = q - nx, vq = v[q], zf = zb[L] > zb[q] ? zb[L] : zb[q], e = vq > 0 ? eta[L] : vq < 0 ? eta[q] : eta[L] > eta[q] ? eta[L] : eta[q], hf = e - zf;
      if (hf <= hmin) { Fy[q] = 0; v[q] = 0; } else Fy[q] = hf * vq * dx;
    }
    // 2. continuity (kinematic free-surface condition, depth-integrated)
    const dA = dt / A;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0, q = j * nu1; i < nx; i++, P++, q++) {
      if (land[P]) continue;
      let e = eta[P] - dA * (Fx[q + 1] - Fx[q] + Fy[P + nx] - Fy[P]);
      if (e < zb[P]) { S.volClamp += (zb[P] - e) * A; e = zb[P]; }
      eta[P] = e; h[P] = e - zb[P];
    }
    for (let j = 0; j < ny; j++) S.volIn += dt * (Fx[j * nu1] - Fx[j * nu1 + nx]);
    for (let i = 0; i < nx; i++) S.volIn += dt * (Fy[i] - Fy[ny * nx + i]);
    if (acc) { for (let q = 0; q < Fx.length; q++) aX[q] += Fx[q]; for (let q = 0; q < Fy.length; q++) aY[q] += Fy[q]; }
    const clampCell = (P, i, j) => { if (land[P]) return; const e = Math.max(etaExt(E, i, j), zb[P]); S.volClamp += (e - eta[P]) * A; eta[P] = e; h[P] = e - zb[P]; };
    if (bc.W === 'elev') for (let j = 0; j < ny; j++) clampCell(j * nx, 0, j);
    if (bc.E === 'elev') for (let j = 0; j < ny; j++) clampCell(j * nx + nx - 1, nx - 1, j);
    if (bc.S === 'elev') for (let i = 0; i < nx; i++) clampCell(i, i, 0);
    if (bc.N === 'elev') for (let i = 0; i < nx; i++) clampCell((ny - 1) * nx + i, i, ny - 1);
    // 3. momentum (dynamic free-surface condition: hydrostatic pressure gradient g∇η), u then v with the new u
    for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) {
      const q = j * nu1 + i;
      if (!mu[q]) continue;
      const R = j * nx + i, L = R - 1, eL = eta[L], eR = eta[R], zf = zb[L] > zb[R] ? zb[L] : zb[R];
      if ((eL > eR ? eL : eR) - zf <= hmin) { un[q] = 0; continue; }
      const uq = u[q], vb = 0.25 * (v[L] + v[R] + v[L + nx] + v[R + nx]);
      let hb = 0.5 * (h[L] + h[R]); if (hb < hmin) hb = hmin;
      let a = -gdx * (eR - eL) + f * vb;
      if (twx !== 0 || fx) a += (twx + (fx ? 0.5 * (fx[L] + fx[R]) : 0)) / (hb > 0.2 ? hb : 0.2);
      if (fxu) a += fxu[q] / (hb > 0.2 ? hb : 0.2);
      if (adv) a -= uq * (uq > 0 ? uq - u[q - 1] : u[q + 1] - uq) * rdx + vb * (vb > 0 ? (j > 0 ? uq - u[q - nu1] : 0) : j < ny - 1 ? u[q + nu1] - uq : 0) * rdy;
      const cf = manning ? gn2 / Math.cbrt(hb) : cfC;
      let w = cf > 0 ? (uq + dt * a) / (1 + (dt * cf * Math.sqrt(uq * uq + vb * vb)) / hb) : uq + dt * a;
      if (w > 2 || w < -2) { const cap = 3 * Math.sqrt(G * hb) + 0.5; if (w > cap) { w = cap; S.capped++; } else if (w < -cap) { w = -cap; S.capped++; } }
      un[q] = w;
    }
    for (let j = 0; j < ny; j++) { un[j * nu1] = u[j * nu1]; un[j * nu1 + nx] = u[j * nu1 + nx]; }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = j * nx + i;
      if (!mv[q]) continue;
      const L = q - nx, eL = eta[L], eR = eta[q], zf = zb[L] > zb[q] ? zb[L] : zb[q];
      if ((eL > eR ? eL : eR) - zf <= hmin) { vn[q] = 0; continue; }
      const vq = v[q], r = j * nu1 + i, ub = 0.25 * (un[r] + un[r + 1] + un[r - nu1] + un[r - nu1 + 1]);
      let hb = 0.5 * (h[L] + h[q]); if (hb < hmin) hb = hmin;
      let a = -gdy * (eR - eL) - f * ub;
      if (twy !== 0 || fy) a += (twy + (fy ? 0.5 * (fy[L] + fy[q]) : 0)) / (hb > 0.2 ? hb : 0.2);
      if (fyv) a += fyv[q] / (hb > 0.2 ? hb : 0.2);
      if (adv) a -= ub * (ub > 0 ? (i > 0 ? vq - v[q - 1] : 0) : i < nx - 1 ? v[q + 1] - vq : 0) * rdx + vq * (vq > 0 ? vq - v[q - nx] : v[q + nx] - vq) * rdy;
      const cf = manning ? gn2 / Math.cbrt(hb) : cfC;
      let w = cf > 0 ? (vq + dt * a) / (1 + (dt * cf * Math.sqrt(vq * vq + ub * ub)) / hb) : vq + dt * a;
      if (w > 2 || w < -2) { const cap = 3 * Math.sqrt(G * hb) + 0.5; if (w > cap) { w = cap; S.capped++; } else if (w < -cap) { w = -cap; S.capped++; } }
      vn[q] = w;
    }
    for (let i = 0; i < nx; i++) { vn[i] = v[i]; vn[ny * nx + i] = v[ny * nx + i]; }
    // 4. open boundaries
    const side = (s, P, i, j, q, arr, sgn, inner, isU) => {
      const t = bc[s];
      if (t === 'wall' || land[P] || !(h[P] > hmin)) { arr[q] = 0; return; }
      if (t === 'elev') { arr[q] = arr[inner]; return; }
      const X = t === 'rad' ? still : E, ue = t === 'rad' ? 0 : isU ? X.U * (pat ? pat.au[P] : 1) + X.V * (pat ? pat.bu[P] : 0) : X.U * (pat ? pat.av[P] : 0) + X.V * (pat ? pat.bv[P] : 1);
      arr[q] = ue + sgn * Math.sqrt(G / (h[P] > 0.3 ? h[P] : 0.3)) * (eta[P] - etaExt(X, i, j));
    };
    for (let j = 0; j < ny; j++) { side('W', j * nx, 0, j, j * nu1, un, -1, j * nu1 + 1, true); side('E', j * nx + nx - 1, nx - 1, j, j * nu1 + nx, un, 1, j * nu1 + nx - 1, true); }
    for (let i = 0; i < nx; i++) { side('S', i, i, 0, i, vn, -1, nx + i, false); side('N', (ny - 1) * nx + i, i, ny - 1, ny * nx + i, vn, 1, (ny - 1) * nx + i, false); }
    const tu = uc; uc = un; un = tu; const tv = vc; vc = vn; vn = tv;
    S.t += dt; S.steps++;
  };
  /** Advance by exactly dtTot in stable sub-steps; qx, qy receive the time-mean face transports (m³/s). */
  S.advance = (dtTot, qx, qy) => {
    const ds = S.dtStable(), ns = Math.max(1, Math.ceil(dtTot / ds)), dt = dtTot / ns;
    if (S.dt0 === undefined) S.dt0 = ds; else if (ds < (imp ? 1e-3 : 0.02) * S.dt0) throw new Error('The shallow-water solution became unstable (runaway velocities) — raise the minimum depth, coarsen the hydrodynamic grid or use Flather boundaries.');
    if (qx) { aX.fill(0); aY.fill(0); }
    for (let k = 0; k < ns; k++) S.step(dt, !!qx);
    if (qx) { for (let q = 0; q < aX.length; q++) qx[q] = aX[q] / ns; for (let q = 0; q < aY.length; q++) qy[q] = aY[q] / ns; }
    if (S.stat) { // envelope statistics once per call
      let wetN = 0; const st = S.stat;
      for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) { if (land[P]) continue; if (h[P] > hmin) { wetN++; const a = 0.5 * (uc[j * nu1 + i] + uc[j * nu1 + i + 1]), b = 0.5 * (vc[P] + vc[P + nx]), sp = Math.sqrt(a * a + b * b); if (sp > st.spd[P]) st.spd[P] = sp; } if (eta[P] > st.hi[P]) st.hi[P] = eta[P]; if (eta[P] < st.lo[P]) st.lo[P] = eta[P]; }
      if (wetN < st.wetMin) st.wetMin = wetN; if (wetN > st.wetMax) st.wetMax = wetN;
    }
    if (!Number.isFinite(eta[(ny >> 1) * nx + (nx >> 1)])) throw new Error('The shallow-water solution became unstable — coarsen the grid or raise the minimum depth.');
    return dtTot;
  };
  S.track = () => { S.stat = { spd: new Float64Array(n), hi: Float64Array.from(eta), lo: Float64Array.from(eta), wetMin: Infinity, wetMax: 0 }; };
  S.cellU = (P) => { const i = P % nx, j = (P - i) / nx; return [0.5 * (uc[j * nu1 + i] + uc[j * nu1 + i + 1]), 0.5 * (vc[P] + vc[P + nx])]; };
  return S;
}

/**
 * Couples a shallowWater() solver (cells m × the transport cells) to the far-field transport grid nx × ny: the
 * coarse face transports are interpolated linearly to the fine faces, which keeps the fine-grid divergence
 * equal to the coarse one (∂η/∂t uniform inside each hydrodynamic cell).
 */
export function swCoupler(sw, nx, ny, m = 1, eta0 = 0) {
  const ncx = sw.nx, ncy = sw.ny, map = new Int32Array(nx * ny), cx = m > 1 ? new Float64Array((ncx + 1) * ncy) : null, cy = m > 1 ? new Float64Array(ncx * (ncy + 1)) : null;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) map[j * nx + i] = Math.min(ncy - 1, Math.floor(j / m)) * ncx + Math.min(ncx - 1, Math.floor(i / m));
  const h0 = Float64Array.from(sw.h);
  return { sw, map, eta0, h0, depth: (P) => sw.h[map[P]], eta: (P) => sw.eta[map[P]], dh: (P) => (sw.land[map[P]] ? NaN : sw.h[map[P]] - h0[map[P]]), dtHydro: () => (sw.implicit ? sw.dtStable() : 0),
    advance(dt, qx, qy) {
      if (m === 1) return sw.advance(dt, qx, qy);
      sw.advance(dt, cx, cy);
      for (let j = 0; j < ny; j++) { const J = Math.floor(j / m); for (let i = 0; i <= nx; i++) { const I = Math.floor(i / m), a = i - I * m, FW = cx[J * (ncx + 1) + I]; qx[j * (nx + 1) + i] = a === 0 ? FW / m : (FW * (m - a) + cx[J * (ncx + 1) + I + 1] * a) / (m * m); } }
      for (let j = 0; j <= ny; j++) { const J = Math.floor(j / m), b = j - J * m; for (let i = 0; i < nx; i++) { const I = Math.floor(i / m), FS = cy[J * ncx + I]; qy[j * nx + i] = b === 0 ? FS / m : (FS * (m - b) + cy[(J + 1) * ncx + I] * b) / (m * m); } }
      return dt;
    } };
}

// ---------------------------------------------------------------------------------------------------
// Waves: steady wave-action balance (refraction, shoaling, depth-limited breaking, Doppler shift by currents)
// ---------------------------------------------------------------------------------------------------
/** Linear dispersion with a following/opposing current component Uk along the wave direction: (ω − k·Uk)² = g k tanh(kh). */
export function waveNumber(om, h, Uk = 0) {
  const k0 = (om * om) / G, x = k0 * h;
  let k = k0 / Math.sqrt(Math.tanh(x)); // Eckart first guess, then Newton
  for (let it = 0; it < 50; it++) {
    const kh = Math.min(k * h, 30), th = Math.tanh(kh), s = Math.sqrt(G * k * th), F = om - k * Uk - s, dF = -Uk - (0.5 * G * (th + k * h * (1 - th * th))) / s, kn = k - F / dF;
    if (!(kn > 0) || !Number.isFinite(kn)) break;
    const d = Math.abs(kn - k); k = kn; if (d < 1e-13 * k) break;
  }
  const kh = Math.min(k * h, 30), sig = Math.sqrt(G * k * Math.tanh(kh)), nn = 0.5 * (1 + (2 * kh) / Math.sinh(2 * kh));
  return { k, sig, c: sig / k, n: nn, cg: (nn * sig) / k };
}

/**
 * Steady wave-action balance ∇·[(c_g e_θ + U) N] = −D_b/σ for a monochromatic wave, N = E/σ, with the ray
 * (refraction) equation (c_g e_θ + U)·∇θ = −(c_g/c) ∂c/∂n − ∂(U·e_θ)/∂n, solved by upwind pseudo-time marching
 * with local steps. Depth-limited breaking caps the height at γ h. Returns height, direction, orbital velocity,
 * radiation-stress forces (per unit mass and area, m²/s²) and the wave-driven longshore current of Longuet-Higgins.
 * o = { nx, ny, dx, dy, h (0 = land), H0, T, theta0 (propagation direction, rad from +x), gamma, U, V, Cf }.
 */
export function waveField(o) {
  const { nx, ny, dx, dy, h, H0, T } = o, n = nx * ny, om = (2 * Math.PI) / T, gam = o.gamma ?? 0.78, U = o.U || null, V = o.V || null, Cf = o.Cf ?? 0.01, th0 = o.theta0;
  const N = new Float64Array(n), th = new Float64Array(n).fill(th0), k = new Float64Array(n), sg = new Float64Array(n), cc = new Float64Array(n), cg = new Float64Array(n), nn = new Float64Array(n), Nn = new Float64Array(n), tn = new Float64Array(n), brk = new Uint8Array(n);
  let href = 0;
  for (let P = 0; P < n; P++) if (h[P] > href) href = h[P];
  const disp = () => { for (let P = 0; P < n; P++) { if (!(h[P] > 0)) continue; const w = waveNumber(om, h[P], U ? U[P] * Math.cos(th[P]) + V[P] * Math.sin(th[P]) : 0); k[P] = w.k; sg[P] = w.sig; cc[P] = w.c; cg[P] = w.cg; nn[P] = w.n; } };
  disp();
  const w0 = waveNumber(om, href), E0 = (G * H0 * H0) / 8, N0 = E0 / om; // E/ρ; the incident wave is specified in still water of the largest depth
  const capN = (P) => (G * (gam * h[P]) ** 2) / 8 / sg[P];
  for (let P = 0; P < n; P++) N[P] = h[P] > 0 ? Math.min(o.init ? o.init.N[P] : N0 * (w0.cg / cg[P]), capN(P)) : 0;
  if (o.init) th.set(o.init.theta);
  const deep = (P) => h[P] >= 0.5 * href, iters = o.iters ?? 4 * (nx + ny), lim = (d1, d2) => (d1 * d2 > 0 ? (2 * d1 * d2) / (d1 + d2) : 0);
  const vxA = new Float64Array(n), vyA = new Float64Array(n), wet = Uint8Array.from(h, (x) => (x > 0 ? 1 : 0));
  // second-order (van Leer limited) upwind face values on the low (M|P) and high (P|Q) faces; g = value outside the grid
  const lowF = (F, P, M, MM, Q, g, vv) => (vv > 0 ? (M < 0 ? g : wet[M] ? F[M] + 0.5 * (MM >= 0 ? lim(F[M] - F[MM], F[P] - F[M]) : 0) : F[P]) : F[P] - 0.5 * (M >= 0 && Q >= 0 && wet[M] && wet[Q] ? lim(F[Q] - F[P], F[P] - F[M]) : 0));
  const highF = (F, P, M, Q, QQ, g, vv) => (vv > 0 ? F[P] + 0.5 * (M >= 0 && Q >= 0 && wet[M] && wet[Q] ? lim(F[P] - F[M], F[Q] - F[P]) : 0) : Q < 0 ? g : wet[Q] ? F[Q] - 0.5 * (QQ >= 0 ? lim(F[QQ] - F[Q], F[Q] - F[P]) : 0) : F[P]);
  const tolW = o.tol ?? 1e-8;
  let its = 0;
  for (let it = 0; it < iters; it++) {
    if (U && it % 25 === 24) disp();
    let dN = 0, dT = 0;
    for (let P = 0; P < n; P++) if (wet[P]) { vxA[P] = cg[P] * Math.cos(th[P]) + (U ? U[P] : 0); vyA[P] = cg[P] * Math.sin(th[P]) + (V ? V[P] : 0); }
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
      if (!wet[P]) { Nn[P] = 0; tn[P] = th0; continue; }
      const ct = Math.cos(th[P]), st = Math.sin(th[P]), vx = vxA[P], vy = vyA[P], dp = deep(P);
      // outside the grid: incident wave in deep water, zero gradient in shallow water; land sends nothing
      const gN = dp ? N0 * (w0.cg / cg[P]) : N[P], gT = dp ? th0 : th[P];
      let M = i > 0 ? P - 1 : -1, Q = i < nx - 1 ? P + 1 : -1, MM = i > 1 && wet[P - 1] && wet[P - 2] ? P - 2 : -1, QQ = i < nx - 2 && wet[P + 1] && wet[P + 2] ? P + 2 : -1;
      let vm = M < 0 ? vx : wet[M] ? 0.5 * (vx + vxA[M]) : vx < 0 ? vx : 0, vq = Q < 0 ? vx : wet[Q] ? 0.5 * (vx + vxA[Q]) : vx > 0 ? vx : 0;
      let div = (vq * highF(N, P, M, Q, QQ, gN, vq) - vm * lowF(N, P, M, MM, Q, gN, vm)) / dx, adv = (vx * (highF(th, P, M, Q, QQ, gT, vx) - lowF(th, P, M, MM, Q, gT, vx))) / dx;
      M = j > 0 ? P - nx : -1; Q = j < ny - 1 ? P + nx : -1; MM = j > 1 && wet[P - nx] && wet[P - 2 * nx] ? P - 2 * nx : -1; QQ = j < ny - 2 && wet[P + nx] && wet[P + 2 * nx] ? P + 2 * nx : -1;
      vm = M < 0 ? vy : wet[M] ? 0.5 * (vy + vyA[M]) : vy < 0 ? vy : 0; vq = Q < 0 ? vy : wet[Q] ? 0.5 * (vy + vyA[Q]) : vy > 0 ? vy : 0;
      div += (vq * highF(N, P, M, Q, QQ, gN, vq) - vm * lowF(N, P, M, MM, Q, gN, vm)) / dy; adv += (vy * (highF(th, P, M, Q, QQ, gT, vy) - lowF(th, P, M, MM, Q, gT, vy))) / dy;
      const cW = i > 0 && wet[P - 1] ? P - 1 : P, cE = i < nx - 1 && wet[P + 1] ? P + 1 : P, cS = j > 0 && wet[P - nx] ? P - nx : P, cN = j < ny - 1 && wet[P + nx] ? P + nx : P, ex = cE === cW ? 0 : 1 / ((cE - cW) * dx), ey = cN === cS ? 0 : 1 / (((cN - cS) / nx) * dy);
      let rhs = (-cg[P] / cc[P]) * (-st * (cc[cE] - cc[cW]) * ex + ct * (cc[cN] - cc[cS]) * ey);
      if (U) rhs -= -st * (ct * (U[cE] - U[cW]) * ex + st * (V[cE] - V[cW]) * ex) + ct * (ct * (U[cN] - U[cS]) * ey + st * (V[cN] - V[cS]) * ey);
      const dtau = 0.45 / (Math.abs(vx) / dx + Math.abs(vy) / dy + 1e-12);
      let Nv = N[P] - dtau * div; const cap = capN(P);
      brk[P] = Nv > cap ? 1 : 0; if (Nv > cap) Nv = cap; if (Nv < 0) Nv = 0;
      const tNew = th[P] + dtau * (rhs - adv), eN = Nv > N[P] ? Nv - N[P] : N[P] - Nv, eT = tNew > th[P] ? tNew - th[P] : th[P] - tNew;
      if (eN > dN) dN = eN; if (eT > dT) dT = eT;
      Nn[P] = Nv; tn[P] = tNew;
    }
    N.set(Nn); th.set(tn); its = it + 1;
    if (dN <= tolW * N0 && dT <= tolW && (!U || it % 25 !== 23)) break; // pseudo-time march has reached the steady state
  }
  if (U) disp();
  const Hs = new Float64Array(n), uorb = new Float64Array(n), fx = new Float64Array(n), fy = new Float64Array(n), Vls = new Float64Array(n), Sxx = new Float64Array(n), Sxy = new Float64Array(n), Syy = new Float64Array(n);
  for (let P = 0; P < n; P++) {
    if (!(h[P] > 0)) continue;
    const E = N[P] * sg[P], ct = Math.cos(th[P]), st = Math.sin(th[P]);
    Hs[P] = Math.sqrt((8 * E) / G); uorb[P] = (0.5 * Hs[P] * sg[P]) / Math.sinh(Math.min(k[P] * h[P], 30));
    Sxx[P] = E * (nn[P] * (ct * ct + 1) - 0.5); Syy[P] = E * (nn[P] * (st * st + 1) - 0.5); Sxy[P] = E * nn[P] * ct * st;
  }
  let surf = 0, HbMax = 0, VlsMax = 0;
  for (let j = 0, P = 0; j < ny; j++) for (let i = 0; i < nx; i++, P++) {
    if (!(h[P] > 0)) continue;
    const cW = i > 0 && h[P - 1] > 0 ? P - 1 : P, cE = i < nx - 1 && h[P + 1] > 0 ? P + 1 : P, cS = j > 0 && h[P - nx] > 0 ? P - nx : P, cN = j < ny - 1 && h[P + nx] > 0 ? P + nx : P;
    const ddx = (F) => (cE === cW ? 0 : (F[cE] - F[cW]) / ((cE - cW) * dx)), ddy = (F) => (cN === cS ? 0 : (F[cN] - F[cS]) / (((cN - cS) / nx) * dy));
    fx[P] = -(ddx(Sxx) + ddy(Sxy)); fy[P] = -(ddx(Sxy) + ddy(Syy));
    if (brk[P]) {
      surf++; if (Hs[P] > HbMax) HbMax = Hs[P];
      // Longuet-Higgins: alongshore radiation-stress force balanced by the wave-averaged bed shear (2/π) C_f u_orb V
      const hx = ddx(h), hy = ddy(h), hm = Math.hypot(hx, hy);
      if (hm > 1e-6 && uorb[P] > 1e-3) { const tx = -hy / hm, ty = hx / hm, Ft = fx[P] * tx + fy[P] * ty; Vls[P] = clamp(Ft / ((2 / Math.PI) * Cf * uorb[P]), -3, 3); if (Math.abs(Vls[P]) > VlsMax) VlsMax = Math.abs(Vls[P]); }
    }
  }
  return { N, theta: th, k, c: cc, cg, sig: sg, Hs, uorb, fx, fy, Vls, brk, surfCells: surf, HbMax, VlsMax, href, iters: its };
}

// ---------------------------------------------------------------------------------------------------
// Vertical (x–z) slice: non-hydrostatic or hydrostatic Boussinesq equations with a turbulence closure
// ---------------------------------------------------------------------------------------------------
const KE = { cmu: 0.09, c1: 1.44, c2: 1.92, sk: 1.0, se: 1.3, st: 0.85 };
/** Pacanowski & Philander (1981) Richardson-number mixing: ν = ν₀/(1 + 5Ri)² + ν_b, K = ν/(1 + 5Ri) + K_b. */
export function ppMixing(Ri, nu0 = 1e-2, nub = 1e-4, Kb = 1e-5) { const r = 1 + 5 * Math.max(Ri, 0), nu = nu0 / (r * r) + nub; return { nu, K: nu / r + Kb }; }

/**
 * Two-dimensional vertical slice of the Boussinesq Navier–Stokes equations (x along the section, z up, rigid lid):
 * 'nonhydro' — full vertical momentum equation with a pressure-projection step (Poisson equation for the
 *   non-hydrostatic pressure, conjugate gradients); 'hydro' — hydrostatic primitive equations: baroclinic pressure
 *   from the vertical integral of buoyancy, rigid-lid barotropic correction of the depth-integrated transport and
 *   w from continuity. Scalar s = density excess expressed as equivalent salinity (g/kg), buoyancy b = −g β s.
 * Turbulence: 'ke' (k–ε with shear and buoyancy production and wall values at the bed), 'pp' (Richardson-number
 * mixing of Pacanowski–Philander) or 'const'. Staggered grid, upwind momentum, van Leer TVD scalar transport.
 * o = { nx, nz, dx, dz, solid, s0 (cell array), beta, tEnd, turb, nu, Kh, Cd, k0, eps0, src: [{ P, rate }], model }.
 */
export async function verticalSlice(o, ctx) {
  const { nx, nz, dx, dz } = o, n = nx * nz, nu1 = nx + 1, beta = o.beta ?? 7.6e-4, gb = G * beta, hydro = o.model === 'hydro', turb = o.turb || 'const', Cd = o.Cd ?? 0;
  const solid = o.solid || new Uint8Array(n), nuB = o.nu ?? 1e-4, Kb = o.Kv ?? nuB, nuH = o.nuH ?? nuB, KH = o.Kh ?? nuH;
  const s = Float64Array.from(o.s0), sn = new Float64Array(n), u = new Float64Array(nu1 * nz), w = new Float64Array(nx * (nz + 1)), us = new Float64Array(u.length), ws = new Float64Array(w.length), p = new Float64Array(n), ph = new Float64Array(n);
  const ub = new Uint8Array(u.length), wb = new Uint8Array(w.length);
  for (let k = 0; k < nz; k++) for (let i = 0; i <= nx; i++) ub[k * nu1 + i] = i === 0 || i === nx || solid[k * nx + i - 1] || solid[k * nx + i] ? 1 : 0;
  for (let k = 0; k <= nz; k++) for (let i = 0; i < nx; i++) wb[k * nx + i] = k === 0 || k === nz || solid[(k - 1) * nx + i] || solid[k * nx + i] ? 1 : 0;
  const nut = new Float64Array(n).fill(nuB), Kt = new Float64Array(n).fill(Kb), tk = new Float64Array(n).fill(o.k0 ?? 1e-6), te = new Float64Array(n), tkn = new Float64Array(n), ten = new Float64Array(n);
  const Hd = nz * dz, eps0 = o.eps0 ?? (KE.cmu ** 0.75 * (o.k0 ?? 1e-6) ** 1.5) / (0.1 * Hd);
  te.fill(eps0);
  // Poisson operator (non-hydrostatic pressure)
  const aE = new Float64Array(n), aN = new Float64Array(n), dg = new Float64Array(n), rhs = new Float64Array(n);
  let pin = -1;
  for (let k = 0, P = 0; k < nz; k++) for (let i = 0; i < nx; i++, P++) { if (solid[P]) continue; if (pin < 0) pin = P; if (!ub[k * nu1 + i + 1]) aE[P] = dz / dx; if (!wb[(k + 1) * nx + i]) aN[P] = dx / dz; }
  for (let k = 0, P = 0; k < nz; k++) for (let i = 0; i < nx; i++, P++) dg[P] = solid[P] ? 0 : aE[P] + aN[P] + (i > 0 ? aE[P - 1] : 0) + (k > 0 ? aN[P - nx] : 0);
  for (let P = 0; P < n; P++) if (!solid[P] && !(dg[P] > 0)) dg[P] = 1; // isolated cell
  if (pin >= 0) dg[pin] *= 1.0001;
  const Bp = hydro ? null : bandSolver(nx, nz), Wcg = Bp ? null : { r: new Float64Array(n), z: new Float64Array(n), s: new Float64Array(n), q: new Float64Array(n), pc: new Float64Array(n) }; // the operator is constant: factorised once
  const srcs = o.src || [], lim = (d1, d2) => (d1 * d2 > 0 ? (d1 * d2) / (d1 + d2) : 0);
  /** Limited upwind value on the face between entries b and c of F (a and d are the next ones out, −1 if absent). */
  const fv = (F, a, b, c, d, vel) => (vel > 0 ? F[b] + (a >= 0 ? lim(F[b] - F[a], F[c] - F[b]) : 0) : F[c] - (d >= 0 ? lim(F[d] - F[c], F[c] - F[b]) : 0));
  let t = 0, steps = 0, salt0 = 0, injected = 0, pIter = 0;
  for (let P = 0; P < n; P++) if (!solid[P]) salt0 += s[P];
  const hist = { t: [], front: [], ke: [] }, tEnd = o.tEnd, sRange = () => { let a = Infinity, b = -Infinity; for (let P = 0; P < n; P++) if (!solid[P]) { if (s[P] < a) a = s[P]; if (s[P] > b) b = s[P]; } return b - a; };
  const cInt0 = Math.sqrt(Math.max(gb * Math.max(sRange(), o.sScale ?? 0, 1e-9) * Hd, 1e-12));
  const rx2 = 1 / (dx * dx), rz2 = 1 / (dz * dz);
  const turbStep = (dt) => {
    let numax = nuB;
    for (let k = 0, P = 0; k < nz; k++) for (let i = 0; i < nx; i++, P++) {
      if (solid[P]) { nut[P] = nuB; Kt[P] = Kb; tkn[P] = tk[P]; ten[P] = te[P]; continue; }
      const q = k * nu1 + i, r = k * nx + i, uc = 0.5 * (u[q] + u[q + 1]), wc = 0.5 * (w[r] + w[r + nx]);
      const up = k < nz - 1 && !solid[P + nx] ? 0.5 * (u[q + nu1] + u[q + nu1 + 1]) : uc, dn = k > 0 && !solid[P - nx] ? 0.5 * (u[q - nu1] + u[q - nu1 + 1]) : uc, dzz = ((k < nz - 1 && !solid[P + nx] ? 1 : 0) + (k > 0 && !solid[P - nx] ? 1 : 0)) * dz || dz;
      const uz = (up - dn) / dzz, ux = (u[q + 1] - u[q]) / dx, wz = (w[r + nx] - w[r]) / dz;
      const wE = i < nx - 1 && !solid[P + 1] ? 0.5 * (w[r + 1] + w[r + 1 + nx]) : wc, wW = i > 0 && !solid[P - 1] ? 0.5 * (w[r - 1] + w[r - 1 + nx]) : wc, wx = (wE - wW) / (2 * dx);
      const S2 = 2 * (ux * ux + wz * wz) + (uz + wx) * (uz + wx);
      const sU = k < nz - 1 && !solid[P + nx] ? s[P + nx] : s[P], sD = k > 0 && !solid[P - nx] ? s[P - nx] : s[P], N2 = (-gb * (sU - sD)) / dzz;
      if (turb === 'pp') { const m = ppMixing(N2 / (uz * uz + 1e-10), o.nu0 ?? 1e-2, nuB, Kb); nut[P] = N2 < 0 ? (o.nu0 ?? 1e-2) + nuB : m.nu; Kt[P] = N2 < 0 ? (o.nu0 ?? 1e-2) + Kb : m.K; }
      else if (turb === 'ke') {
        // k–ε: Dk/Dt = ∇·(ν_t/σ_k ∇k) + P + B − ε, Dε/Dt = ∇·(ν_t/σ_ε ∇ε) + (ε/k)(c₁P + c₃B − c₂ε); explicit, sinks implicit
        const nt = nut[P] - nuB, Pk = nt * S2, B = (-nt / KE.st) * N2, k0 = tk[P], e0 = te[P];
        const W = i > 0 && !solid[P - 1] ? P - 1 : P, Ea = i < nx - 1 && !solid[P + 1] ? P + 1 : P, Dn = k > 0 && !solid[P - nx] ? P - nx : P, Up = k < nz - 1 && !solid[P + nx] ? P + nx : P;
        const nE = 0.5 * (nut[P] + nut[Ea]) * rx2, nW = 0.5 * (nut[P] + nut[W]) * rx2, nU = 0.5 * (nut[P] + nut[Up]) * rz2, nD = 0.5 * (nut[P] + nut[Dn]) * rz2;
        const advK = (uc > 0 ? uc * (k0 - tk[W]) : uc * (tk[Ea] - k0)) / dx + (wc > 0 ? wc * (k0 - tk[Dn]) : wc * (tk[Up] - k0)) / dz, advE = (uc > 0 ? uc * (e0 - te[W]) : uc * (te[Ea] - e0)) / dx + (wc > 0 ? wc * (e0 - te[Dn]) : wc * (te[Up] - e0)) / dz;
        const difK = (nE * (tk[Ea] - k0) - nW * (k0 - tk[W]) + nU * (tk[Up] - k0) - nD * (k0 - tk[Dn])) / KE.sk, difE = (nE * (te[Ea] - e0) - nW * (e0 - te[W]) + nU * (te[Up] - e0) - nD * (e0 - te[Dn])) / KE.se;
        const kNew = (k0 + dt * (-advK + difK + Pk + (B > 0 ? B : 0))) / (1 + (dt * (e0 + (B < 0 ? -B : 0))) / k0);
        const eNew = (e0 + dt * (-advE + difE + (e0 / k0) * KE.c1 * (Pk + (B > 0 ? B : 0)))) / (1 + (dt * KE.c2 * e0) / k0);
        tkn[P] = kNew > 1e-10 ? kNew : 1e-10; ten[P] = eNew > 1e-14 ? eNew : 1e-14;
      }
      if (nut[P] > numax) numax = nut[P];
    }
    if (turb === 'ke') {
      tk.set(tkn); te.set(ten);
      for (let k = 0, P = 0; k < nz; k++) for (let i = 0; i < nx; i++, P++) {
        if (solid[P]) continue;
        if (Cd > 0 && (k === 0 || solid[P - nx])) { const uc = 0.5 * (u[k * nu1 + i] + u[k * nu1 + i + 1]), ut2 = Cd * uc * uc; if (ut2 > 1e-12) { tk[P] = Math.max(tk[P], ut2 / Math.sqrt(KE.cmu)); te[P] = Math.max(te[P], ut2 ** 1.5 / (0.41 * 0.5 * dz)); } }
        const sU = k < nz - 1 && !solid[P + nx] ? s[P + nx] : s[P], sD = k > 0 && !solid[P - nx] ? s[P - nx] : s[P], N2 = (-gb * (sU - sD)) / (2 * dz);
        let e = te[P];
        if (N2 > 0) { const eMin = (KE.cmu ** 0.75 * tk[P] * Math.sqrt(N2)) / (0.53 * Math.SQRT2); if (e < eMin) e = te[P] = eMin; } // Galperin length-scale limit in stable stratification
        const nt = Math.min((KE.cmu * tk[P] * tk[P]) / e, o.nuMax ?? 0.05);
        nut[P] = nuB + nt; Kt[P] = Kb + nt / KE.st;
        if (nut[P] > numax) numax = nut[P];
      }
    }
    return numax;
  };
  let numax = nuB;
  const front = o.front || null; // { row, i0, dir, level }
  while (t < tEnd - 1e-9 && steps < (o.maxSteps ?? 20000)) {
    let um = 1e-9, wm = 1e-9;
    for (let q = 0; q < u.length; q++) { const a = Math.abs(u[q]); if (a > um) um = a; }
    for (let q = 0; q < w.length; q++) { const a = Math.abs(w[q]); if (a > wm) wm = a; }
    const nuX = Math.max(numax, nuH, KH), cI = Math.max(cInt0, Math.sqrt(gb * Math.max(sRange(), 1e-9) * Hd));
    const dt = Math.min((o.cfl ?? 0.35) / (um / dx + wm / dz), 0.2 / (numax / (dz * dz) + nuX / (dx * dx)), (0.8 * dx) / cI, tEnd - t, o.dtMax ?? Infinity);
    if (turb !== 'const') numax = turbStep(dt);
    // hydrostatic pressure (÷ρ₀) from the top down
    if (hydro) for (let i = 0; i < nx; i++) { let acc = 0, prev = 0, first = true; for (let k = nz - 1; k >= 0; k--) { const P = k * nx + i; if (solid[P]) break; acc += first ? 0.5 * gb * s[P] * dz : 0.5 * gb * (s[P] + prev) * dz; ph[P] = acc; prev = s[P]; first = false; } }
    // ---- momentum predictor
    for (let k = 0; k < nz; k++) for (let i = 1; i < nx; i++) {
      const q = k * nu1 + i;
      if (ub[q]) { us[q] = 0; continue; }
      const R = k * nx + i, L = R - 1, uq = u[q];
      const upOk = k < nz - 1 && !ub[q + nu1], dnOk = k > 0 && !ub[q - nu1], nuC = 0.5 * (nut[L] + nut[R]);
      // conservative (flux-form) advection with van Leer limited upwind face values
      const Fe = 0.5 * (uq + u[q + 1]), Fw = 0.5 * (uq + u[q - 1]), Fn = k < nz - 1 ? 0.5 * (w[L + nx] + w[R + nx]) : 0, Fs = k > 0 ? 0.5 * (w[L] + w[R]) : 0;
      let a = -(Fe * fv(u, q - 1, q, q + 1, i + 2 <= nx ? q + 2 : -1, Fe) - Fw * fv(u, i >= 2 ? q - 2 : -1, q - 1, q, q + 1, Fw)) / dx
        - ((Fn !== 0 ? Fn * fv(u, dnOk ? q - nu1 : -1, q, q + nu1, k + 2 < nz ? q + 2 * nu1 : -1, Fn) : 0) - (Fs !== 0 ? Fs * fv(u, k >= 2 ? q - 2 * nu1 : -1, q - nu1, q, upOk ? q + nu1 : -1, Fs) : 0)) / dz;
      a += (nuH * (u[q + 1] - 2 * uq + u[q - 1])) / (dx * dx) + (nuC * ((upOk ? u[q + nu1] - uq : 0) - (dnOk ? uq - u[q - nu1] : 0))) / (dz * dz);
      a -= ((hydro ? ph[R] - ph[L] : p[R] - p[L])) / dx; // hydrostatic pressure, or the previous non-hydrostatic pressure (incremental projection)
      const drag = Cd > 0 && !dnOk ? (Cd * Math.abs(uq)) / dz : 0;
      us[q] = (uq + dt * a) / (1 + dt * drag);
    }
    if (!hydro) {
      for (let k = 1; k < nz; k++) for (let i = 0; i < nx; i++) {
        const q = k * nx + i;
        if (wb[q]) { ws[q] = 0; continue; }
        const D = q - nx, wq = w[q], r = k * nu1 + i;
        const eOk = i < nx - 1 && !wb[q + 1], wOk = i > 0 && !wb[q - 1], nuC = 0.5 * (nut[D] + nut[q]);
        const Fe = i < nx - 1 ? 0.5 * (u[r + 1] + u[r - nu1 + 1]) : 0, Fw = i > 0 ? 0.5 * (u[r] + u[r - nu1]) : 0, Fn = 0.5 * (wq + w[q + nx]), Fs = 0.5 * (wq + w[q - nx]);
        let a = -((Fe !== 0 ? Fe * fv(w, wOk ? q - 1 : -1, q, q + 1, i + 2 < nx ? q + 2 : -1, Fe) : 0) - (Fw !== 0 ? Fw * fv(w, i >= 2 ? q - 2 : -1, q - 1, q, eOk ? q + 1 : -1, Fw) : 0)) / dx
          - (Fn * fv(w, q - nx, q, q + nx, k + 2 <= nz ? q + 2 * nx : -1, Fn) - Fs * fv(w, k >= 2 ? q - 2 * nx : -1, q - nx, q, q + nx, Fs)) / dz;
        a += (nuH * ((eOk ? w[q + 1] - wq : 0) - (wOk ? wq - w[q - 1] : 0))) / (dx * dx) + (nuC * (w[q + nx] - 2 * wq + w[q - nx])) / (dz * dz);
        a -= 0.5 * gb * (s[D] + s[q]) + (p[q] - p[D]) / dz;
        ws[q] = wq + dt * a;
      }
      // incremental projection: ∇²φ = ∇·u*/Δt, u = u* − Δt ∇φ, p ← p + φ
      for (let k = 0, P = 0; k < nz; k++) for (let i = 0; i < nx; i++, P++) rhs[P] = solid[P] ? 0 : -((us[k * nu1 + i + 1] - us[k * nu1 + i]) * dz + (ws[P + nx] - ws[P]) * dx) / dt;
      if (Bp) pIter += Bp.solve(aE, aN, dg, rhs, ph, o.pTol ?? 2e-3, o.pIter ?? 60, true).iters; // pressure increment φ
      else { ph.fill(0); pIter += pcg5(nx, nz, aE, aN, dg, rhs, ph, o.pTol ?? 2e-3, o.pIter ?? 60, Wcg).iters; }
      for (let k = 0; k < nz; k++) for (let i = 1; i < nx; i++) { const q = k * nu1 + i; u[q] = ub[q] ? 0 : us[q] - (dt * (ph[k * nx + i] - ph[k * nx + i - 1])) / dx; }
      for (let k = 1; k < nz; k++) for (let i = 0; i < nx; i++) { const q = k * nx + i; w[q] = wb[q] ? 0 : ws[q] - (dt * (ph[q] - ph[q - nx])) / dz; }
      for (let P = 0; P < n; P++) p[P] += ph[P];
    } else {
      // rigid lid: depth-integrated transport through every section equals the prescribed through-flow (zero), then w from continuity
      for (let i = 1; i < nx; i++) { let Q = 0, m = 0; for (let k = 0; k < nz; k++) if (!ub[k * nu1 + i]) { Q += us[k * nu1 + i]; m++; } const c = m ? ((o.Q0 ?? 0) / dz - Q) / m : 0; for (let k = 0; k < nz; k++) { const q = k * nu1 + i; u[q] = ub[q] ? 0 : us[q] + c; } }
      for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) { const P = k * nx + i; w[P + nx] = solid[P] || wb[P + nx] ? 0 : w[P] - ((u[k * nu1 + i + 1] - u[k * nu1 + i]) * dz) / dx; }
    }
    // ---- scalar (van Leer TVD, conservative)
    sn.set(s);
    for (let k = 0; k < nz; k++) for (let i = 1; i < nx; i++) {
      const q = k * nu1 + i;
      if (ub[q]) continue;
      const R = k * nx + i, L = R - 1, uq = u[q];
      let sf;
      if (uq >= 0) sf = s[L] + (i > 1 && !solid[L - 1] ? lim(s[L] - s[L - 1], s[R] - s[L]) : 0); else sf = s[R] + (i < nx - 1 && !solid[R + 1] ? lim(s[R] - s[R + 1], s[L] - s[R]) : 0);
      const fl = (dt * (uq * sf - (KH * (s[R] - s[L])) / dx)) / dx;
      sn[L] -= fl; sn[R] += fl;
    }
    for (let k = 1; k < nz; k++) for (let i = 0; i < nx; i++) {
      const q = k * nx + i;
      if (wb[q]) continue;
      const D = q - nx, wq = w[q];
      let sf;
      if (wq >= 0) sf = s[D] + (k > 1 && !solid[D - nx] ? lim(s[D] - s[D - nx], s[q] - s[D]) : 0); else sf = s[q] + (k < nz - 1 && !solid[q + nx] ? lim(s[q] - s[q + nx], s[D] - s[q]) : 0);
      const fl = (dt * (wq * sf - (0.5 * (Kt[D] + Kt[q]) * (s[q] - s[D])) / dz)) / dz;
      sn[D] -= fl; sn[q] += fl;
    }
    for (const sc of srcs) { sn[sc.P] += dt * sc.rate; injected += dt * sc.rate; }
    s.set(sn);
    t += dt; steps++;
    if (!Number.isFinite(s[pin >= 0 ? pin : 0]) || !Number.isFinite(um)) throw new Error('The vertical-slice solution became unstable — reduce the eddy viscosity limit or coarsen the slice grid.');
    if (steps % 4 === 0 || t >= tEnd - 1e-9) {
      let xf = null;
      if (front) { const { row, level, dir, i0 } = front; let last = -1; for (let i = i0; i >= 0 && i < nx; i += dir) { let k = row; if (k < 0) { k = 0; while (k < nz - 1 && solid[k * nx + i]) k++; } const P = k * nx + i; if (solid[P]) break; if (s[P] - (o.sAmb ? o.sAmb[P] : 0) > level) last = i; } if (last >= 0) { // sub-cell position of the level crossing
          let k = row; if (k < 0) { k = 0; while (k < nz - 1 && solid[k * nx + last]) k++; }
          const P = k * nx + last, Q = P + dir, a = s[P] - (o.sAmb ? o.sAmb[P] : 0), b = last + dir >= 0 && last + dir < nx && !solid[Q] ? s[Q] - (o.sAmb ? o.sAmb[Q] : 0) : level;
          xf = (last + 0.5 + (a > b ? clamp((a - level) / (a - b), 0, 1) : 0) * dir) * dx; } }
      let ke = 0; for (let q = 0; q < u.length; q++) ke += u[q] * u[q];
      hist.t.push(t); hist.front.push(xf); hist.ke.push(0.5 * ke * dx * dz);
      if (steps % 40 === 0) { ctx?.progress?.(o.p0 !== undefined ? o.p0 + (o.p1 - o.p0) * (t / tEnd) : t / tEnd, `Vertical slice: ${fmt(t, 3)} s of ${fmt(tEnd, 3)} s`); if (ctx?.tick) await ctx.tick(); }
    }
  }
  let salt = 0, divMax = 0;
  for (let k = 0, P = 0; k < nz; k++) for (let i = 0; i < nx; i++, P++) { if (solid[P]) continue; salt += s[P]; const d = Math.abs((u[k * nu1 + i + 1] - u[k * nu1 + i]) / dx + (w[P + nx] - w[P]) / dz); if (d > divMax) divMax = d; }
  return { nx, nz, dx, dz, solid, s, u, w, p, nut, Kt, k: tk, eps: te, t, steps, hist, salt0, salt, injected, divMax, pIter, nu1 };
}

/** Least-squares front speed (m/s) from a front-position history over the window [f0, f1] of the run time. */
export function frontSpeed(hist, f0 = 0.3, f1 = 1) {
  const T = hist.t.at(-1) || 1, xs = [], ts = [];
  hist.t.forEach((t, k) => { if (hist.front[k] !== null && t >= f0 * T && t <= f1 * T) { ts.push(t); xs.push(hist.front[k]); } });
  if (ts.length < 3) return 0;
  const tm = mean(ts), xm = mean(xs); let a = 0, b = 0;
  ts.forEach((t, k) => { a += (t - tm) * (xs[k] - xm); b += (t - tm) ** 2; });
  return b > 0 ? a / b : 0;
}

// ---------------------------------------------------------------------------------------------------
// Three-dimensional hydrostatic (primitive-equation) Boussinesq model with a free surface, σ-layers
// ---------------------------------------------------------------------------------------------------
/**
 * Three-dimensional hydrostatic Boussinesq equations on the horizontal C grid of shallowWater() with nz terrain-following
 * σ-layers (layer k = 0 lies on the bed; thickness Δσ_k·D, D = η − z_b):
 *   ∂u/∂t + u·∇u + ω ∂u/∂z − f v = −g ∂η/∂x − (g/ρ₀) ∂/∂x|_z ∫_z^η ρ′ dz + ∂/∂z(ν ∂u/∂z) + ν_H ∇²u   (likewise v),
 *   ∂η/∂t + ∇·∫ u dz = 0,   layer continuity for the flux ω through the moving σ-surfaces,
 *   ∂(hc)/∂t + ∇·(h u c) + Δ(ω c) = Δ(K ∂c/∂z) + ∇·(h K_H ∇c) + source   for every tracer,   ρ′ = dens(tracers, z).
 * Mode splitting: the external (barotropic) mode — free surface, depth-mean velocity, wetting and drying and the open
 * boundaries — is the semi-implicit shallowWater() solver, forced on its faces by the depth integrals of the baroclinic
 * pressure gradient, of the three-dimensional momentum advection and horizontal viscosity, and by the difference between
 * the bottom stress of the velocity profile and that of the depth mean. The internal mode solves, per face, the vertical
 * tridiagonal system (implicit eddy viscosity, bottom drag and vertical advection, surface wind stress) with a uniform
 * pressure-gradient increment chosen so that the depth mean equals the barotropic velocity; the layer volume fluxes are
 * the barotropic time-mean transport split over the layers plus the zero-sum deviation, so the layer volumes follow the
 * free surface exactly. Baroclinic pressure gradient: density-Jacobian form on σ-layers, exact for a density that varies
 * linearly with z; dens() should return the anomaly from a horizontally uniform reference profile, which is then
 * balanced identically. Tracers: flux-form finite volumes, van Leer limited fluxes (first-order upwind next to thin
 * water), implicit vertical diffusion, sub-cycled to the layer Courant number. Vertical mixing: 'pp' — neutral
 * parabolic profile κ u* z (1 − z/D) from the bottom and wind friction velocities, damped by the Richardson-number
 * functions of Pacanowski & Philander (ppMixing) — or 'const'. Bottom drag: quadratic law on the velocity of the bottom
 * layer with C_b = (κ / ln(z_1/z₀))², or the constant Cd when z0 = 0; slip = true removes it.
 * Momentum advection: momentum-conserving form with van Leer limited face values (madv: 'upwind' for first order); the implicit
 * vertical advection carries its limited second-order part as an explicit correction.
 * Near-field inflow (coupling to an integral jet model at the end of its near field): inflow = { Q, S, u, v, eff, cols }.
 * Each column { P, w, src: [{ k, f }], sink: [{ k, f }] } (Σw = 1, Σf = 1) receives the near-field water S·Q·w in its
 * source layers and gives up the entrained water (S − 1)·Q·w from its sink layers, so the net volume source is the effluent Q·w.
 * The near-field water carries, for every tracer m, the effluent value eff[m] diluted with the entrained water at the concentration
 * found in the sink layers (re-entrainment included): the content added to the domain is exactly Q·eff[m] per second. It enters
 * with the horizontal velocity (u, v): the momentum cells of the source layers relax toward it at the rate inflow ÷ cell volume.
 * o = { nx, ny, nz, dx, dy, zb, land, sigma, hmin, f, rho0, Cd, z0, slip, turb, nu, Kv, nu0, nuH, Kh, madv, dens, refDensity, inflow,
 *       tracers: [{ c0, src: [{ P, k, rate }], open, diffH }], sw: { eta0, x0, y0, bc, ext, pat, tau, theta, dtMax } }.
 */
export function hydro3D(o) {
  const { nx, ny, dx, dy, zb } = o, K = Math.max(2, Math.round(o.nz)), n = nx * ny, nu1 = nx + 1, NU = nu1 * ny, NV = nx * (ny + 1), A = dx * dy, f64 = (m) => new Float64Array(m);
  const land = o.land || new Uint8Array(n), hmin = o.hmin ?? 0.05, hThin = o.hThin ?? Math.max(4 * hmin, 0.3), f = o.f || 0, rho0 = o.rho0 ?? 1025, gr = G / rho0, VK = 0.41;
  const Cd = o.Cd ?? 0.0025, z0 = o.z0 ?? 0, slip = !!o.slip, pp = (o.turb || 'pp') === 'pp', nuB = o.nu ?? 1e-4, KvB = o.Kv ?? (pp ? 1e-5 : nuB), nu0 = o.nu0 ?? 0, nuConv = o.nuConv ?? 1e-2, nuH = o.nuH ?? 0, Kh = o.Kh ?? 0, cfl = o.cfl ?? 0.6;
  const dens = o.dens || (() => 0), refD = o.refDensity || null, tvd = (o.madv || 'tvd') !== 'upwind';
  const ds = f64(K), sc = f64(K), sf = f64(K + 1), rds = f64(K), rdc = f64(K + 1);
  { const w = Array.isArray(o.sigma) && o.sigma.length === K ? o.sigma : null; let s = 0; for (let k = 0; k < K; k++) s += w ? w[k] : 1; for (let k = 0; k < K; k++) { ds[k] = (w ? w[k] : 1) / s; sf[k + 1] = sf[k] + ds[k]; sc[k] = sf[k] + 0.5 * ds[k]; } sf[K] = 1; for (let k = 0; k < K; k++) { rds[k] = 1 / ds[k]; if (k) rdc[k] = 1 / (sc[k] - sc[k - 1]); } }
  const Fu = f64(NU), Fv = f64(NV), swo = o.sw || {};
  // Near-field inflow (see the header): per column the outflow S·Q·w of near-field water over its source layers and the entrainment
  // (S − 1)·Q·w over its sink layers; qcol = net volume source of the column (the effluent Q·w), qlay = net source of every layer cell.
  const inf = o.inflow && o.inflow.Q > 0 && o.inflow.cols?.length ? o.inflow : null, qcol = inf ? f64(n) : null, qlay = inf ? f64(K * n) : null, sgU = inf ? f64(K * NU) : null, sgV = inf ? f64(K * NV) : null, sgT = [];
  if (inf) for (const c of inf.cols) { qcol[c.P] += inf.Q * c.w; for (const q of c.src) qlay[q.k * n + c.P] += inf.S * inf.Q * c.w * q.f; for (const q of c.sink) qlay[q.k * n + c.P] -= (inf.S - 1) * inf.Q * c.w * q.f; }
  const sw = shallowWater({ implicit: true, advect: false, ...swo, nx, ny, dx, dy, zb, land, hmin, f, fric: { type: 'cd', Cd: slip ? 0 : Cd }, force: { ...(swo.force || {}), fxu: Fu, fyv: Fv }, qsrc: qcol });
  const u = f64(K * NU), v = f64(K * NV), Qx = f64(K * NU), Qy = f64(K * NV), W = f64((K + 1) * n), rho = f64(K * n), Ip = f64(K * n), nuI = f64((K + 1) * n).fill(nuB), KvI = f64((K + 1) * n).fill(KvB), Gu = f64(K * NU), Gv = f64(K * NV);
  const gamU = f64(NU), gamV = f64(NV), tyU = new Uint8Array(NU), tyV = new Uint8Array(NV), Do = f64(n), Dn = f64(n), qx = f64(NU), qy = f64(NV), thick = new Uint8Array(n), acc = f64(K * n);
  const ta = f64(K), tb = f64(K), tc = f64(K), td = f64(K), te = f64(K), tg = f64(K);
  const tr = (o.tracers || []).map((t) => { const c = f64(K * n); if (t.c0 !== undefined) { if (typeof t.c0 === 'number') c.fill(t.c0); else if (typeof t.c0 === 'function') { for (let k = 0; k < K; k++) for (let P = 0; P < n; P++) c[k * n + P] = land[P] ? 0 : t.c0(P, k, zb[P] + sc[k] * sw.h[P]); } else c.set(t.c0); } return c; });
  const meta = (o.tracers || []).map((t) => ({ src: (t.src || []).map((s) => ({ idx: s.k * n + s.P, rate: s.rate })), open: typeof t.open === 'function' ? t.open : ((val) => () => val)(t.open ?? 0), diffH: t.diffH !== false && Kh > 0, injected: 0, out: 0, inn: 0 }));
  const cdBot = (h1) => (slip ? 0 : z0 > 0 ? Math.min((VK / Math.log(Math.max((0.5 * h1) / z0, 1.5))) ** 2, 0.05) : Cd);
  const lim = (d1, d2) => (d1 * d2 > 0 ? (d1 * d2) / (d1 + d2) : 0);
  if (inf) inf.wSum = inf.cols.reduce((a, c) => a + c.w, 0);
  const M = { nx, ny, nz: K, dx, dy, sw, u, v, W, rho, tr, ds, sc, sf, nuI, KvI, t: 0, steps: 0, subSteps: 0, uMax: 0, vMax: 0, drho: 0, meta, land, zb };
  { let lo = Infinity, hi = -Infinity; for (let k = 0; k < K; k++) for (let P = 0; P < n; P++) if (!land[P] && sw.h[P] > hmin) { const r = dens(tr, k * n + P, zb[P] + sc[k] * sw.h[P]); if (r < lo) lo = r; if (r > hi) hi = r; } M.drho = hi > lo ? hi - lo : 0; }
  M.dtStable = () => {
    let hm = 1; for (let P = 0; P < n; P++) if (sw.h[P] > hm) hm = sw.h[P];
    const dm = Math.min(dx, dy), dtI = (0.7 * dm) / (0.5 * Math.sqrt(gr * M.drho * hm) + 1e-9), dtV = nuH > 0 || Kh > 0 ? 0.2 / (Math.max(nuH, Kh) * (1 / (dx * dx) + 1 / (dy * dy))) : Infinity;
    return Math.min(sw.dtStable(), cfl / (M.uMax / dx + M.vMax / dy + 1e-12), dtI, dtV, o.dtMax ?? Infinity);
  };
  /** Explicit tendencies of one velocity component (ax = 0: u faces, 1: v faces) and the depth-integrated force for the barotropic mode. */
  const tend = (ax, dt) => {
    const h = sw.h, eta = sw.eta, a = ax ? v : u, b = ax ? u : v, NA = ax ? NV : NU, NB = ax ? NU : NV, Ga = ax ? Gv : Gu, Fa = ax ? Fv : Fu, gam = ax ? gamV : gamU, ty = ax ? tyV : tyU, ab = ax ? sw.v : sw.u, bb = ax ? sw.u : sw.v, dd = ax ? dy : dx, rd = 1 / dd;
    const sA = ax ? nx : 1, sT = ax ? 1 : nu1; // stride along and across the component on its own face lattice
    const sgA = inf ? (ax ? sgV : sgU) : null, sgW = inf ? (ax ? inf.v || 0 : inf.u || 0) : 0; // momentum of the near-field inflow: relaxation rate σ = q/V toward its velocity
    const Qa = ax ? Qy : Qx, Qb = ax ? Qx : Qy;
    for (let j = ax ? 1 : 0; j < ny; j++) for (let i = ax ? 0 : 1; i < nx; i++) {
      const q = ax ? j * nx + i : j * nu1 + i, R = j * nx + i, L = ax ? R - nx : R - 1;
      Fa[q] = 0; ty[q] = 0; gam[q] = 0;
      if (land[L] || land[R]) continue;
      const eL = eta[L], eR = eta[R], zf = zb[L] > zb[R] ? zb[L] : zb[R];
      if ((eL > eR ? eL : eR) - zf <= hmin) continue;
      const Hf = h[L] < h[R] ? h[L] : h[R], thin = Hf < hThin;
      ty[q] = thin ? 1 : 2;
      let hb = 0.5 * (h[L] + h[R]); if (hb < hmin) hb = hmin;
      // indices of the four transverse faces around this face
      const b0 = ax ? (j - 1) * nu1 + i : j * nx + i - 1, b1 = ax ? b0 + 1 : b0 + 1, b2 = ax ? b0 + nu1 : b0 + nx, b3 = b2 + 1;
      const hasM = ax ? i > 0 : j > 0, hasP = ax ? i < nx - 1 : j < ny - 1, hasMM = ax ? i > 1 : j > 1, hasPP = ax ? i < nx - 2 : j < ny - 2, has2M = ax ? j > 1 : i > 1, has2P = ax ? j < ny - 1 : i < nx - 1;
      let Fs = 0, ub0 = 0, vb0 = 0; const rV = 1 / (A * hb), hL = h[L], hR = h[R], dzb = zb[R] - zb[L], dhh = hR - hL, grd = gr * rd;
      for (let k = 0; k < K; k++) {
        const o1 = k * NA + q, o2 = k * NB, aq = a[o1], bq = 0.25 * (b[o2 + b0] + b[o2 + b1] + b[o2 + b2] + b[o2 + b3]);
        if (k === 0) { ub0 = aq; vb0 = bq; }
        // momentum-conserving advection (Stelling & Duinmeijer 2003): [Δ(q̄ û) − u Δq̄] ÷ layer volume, û = first-order upwind, q̄ = layer volume fluxes of the last step
        const o3 = k * NB, qp = 0.5 * (Qa[o1] + Qa[o1 + sA]), qm2 = 0.5 * (Qa[o1 - sA] + Qa[o1]), qtp = 0.5 * (Qb[o3 + b2] + Qb[o3 + b3]), qtm = 0.5 * (Qb[o3 + b0] + Qb[o3 + b1]);
        let g;
        if (tvd) { // second-order limited (van Leer) values of the advected velocity at the four sides of the momentum cell
          const aM = a[o1 - sA], aP = a[o1 + sA], aTm = hasM ? a[o1 - sT] : aq, aTp = hasP ? a[o1 + sT] : aq;
          const fe = qp > 0 ? aq + lim(aq - aM, aP - aq) : aP + (has2P ? lim(aP - a[o1 + 2 * sA], aq - aP) : 0), fw = qm2 > 0 ? aM + (has2M ? lim(aM - a[o1 - 2 * sA], aq - aM) : 0) : aq + lim(aq - aP, aM - aq);
          const fn = qtp > 0 || !hasP ? aq + (hasP && hasM ? lim(aq - aTm, aTp - aq) : 0) : aTp + (hasPP ? lim(aTp - a[o1 + 2 * sT], aq - aTp) : 0), fs = qtm > 0 && hasM ? aTm + (hasMM ? lim(aTm - a[o1 - 2 * sT], aq - aTm) : 0) : aq + (hasP && hasM ? lim(aq - aTp, aTm - aq) : 0);
          g = -(qp * fe - qm2 * fw - aq * (qp - qm2) + qtp * fn - qtm * fs - aq * (qtp - qtm)) * rds[k] * rV;
        } else g = -(qp * (qp > 0 ? aq : a[o1 + sA]) - qm2 * (qm2 > 0 ? a[o1 - sA] : aq) - aq * (qp - qm2) + qtp * (qtp > 0 || !hasP ? aq : a[o1 + sT]) - qtm * (qtm > 0 && hasM ? a[o1 - sT] : aq) - aq * (qtp - qtm)) * rds[k] * rV;
        if (nuH > 0) g += nuH * ((a[o1 + sA] - 2 * aq + a[o1 - sA]) * rd * rd + ((hasP && ty[q + sT] ? a[o1 + sT] - aq : 0) - (hasM && ty[q - sT] ? aq - a[o1 - sT] : 0)) / ((ax ? dx : dy) ** 2));
        if (!thin) { const kR = k * n + R, kL = k * n + L; g -= grd * (Ip[kR] - Ip[kL] + 0.5 * (rho[kL] + rho[kR]) * (dzb + sc[k] * dhh)); }
        Fs += ds[k] * g;
        if (sgA !== null && sgA[o1] > 0) Fs += ds[k] * sgA[o1] * (sgW - aq);
        Ga[o1] = g + (ax ? -f : f) * bq;
      }
      // bottom stress of the profile, γ u_b with γ = C_b |u_b|; the barotropic solver holds C_d |ū| ū implicitly
      const Ub = ab[q], Vb = 0.25 * (bb[b0] + bb[b1] + bb[b2] + bb[b3]), cb = thin ? (slip ? 0 : Cd) : cdBot(ds[0] * hb), g0 = cb * Math.sqrt(ub0 * ub0 + vb0 * vb0);
      gam[q] = g0;
      let Rr = -(g0 * ub0 - (slip ? 0 : Cd) * Math.sqrt(Ub * Ub + Vb * Vb) * Ub);
      const cap = (0.5 * hb * Math.max(Math.abs(Ub), Math.abs(ub0))) / dt;
      if (Rr > cap) Rr = cap; else if (Rr < -cap) Rr = -cap;
      Fa[q] = hb * Fs + Rr;
    }
  };
  /** Vertical implicit solve of one component with the depth-mean constraint; fills the layer volume fluxes. */
  const solve = (ax, dt, tw) => {
    const h = sw.h, a = ax ? v : u, NA = ax ? NV : NU, Ga = ax ? Gv : Gu, gam = ax ? gamV : gamU, ty = ax ? tyV : tyU, ab = ax ? sw.v : sw.u, Q = ax ? Qy : Qx, qm = ax ? qy : qx, len = ax ? dx : dy;
    let amax = 0; const hA = 0.5 / A, sgS = inf ? (ax ? sgV : sgU) : null, sgX = inf ? (ax ? inf.v || 0 : inf.u || 0) : 0;
    for (let j = 0; j < (ax ? ny + 1 : ny); j++) for (let i = 0; i < (ax ? nx : nu1); i++) {
      const q = ax ? j * nx + i : j * nu1 + i, edge = ax ? j === 0 || j === ny : i === 0 || i === nx, Ub = ab[q], F = qm[q];
      if (edge || ty[q] !== 2) { for (let k = 0; k < K; k++) { a[k * NA + q] = Ub; Q[k * NA + q] = ds[k] * F; } continue; }
      const R = j * nx + i, L = ax ? R - nx : R - 1;
      let hb = 0.5 * (h[L] + h[R]); const Hf = h[L] < h[R] ? h[L] : h[R];
      if (Hf < hThin) { for (let k = 0; k < K; k++) { a[k * NA + q] = Ub; Q[k * NA + q] = ds[k] * F; } continue; }
      if (hb < hmin) hb = hmin;
      // tridiagonal coefficients: −ta x_{k−1} + tb x_k − tc x_{k+1} = td (momentum) | te (unit forcing)
      let wl = 0, nl = 0, dl = 1; const rh1 = 1 / hb, rh = dt * rh1; // dl, du: reciprocal distance between the layer centres
      for (let k = 0; k < K; k++) {
        const rk = rds[k] * rh, top = k === K - 1, x1 = (k + 1) * n; // rk = Δt ÷ layer thickness
        const wu = top ? 0 : hA * (W[x1 + L] + W[x1 + R]), nuU = top ? 0 : 0.5 * (nuI[x1 + L] + nuI[x1 + R]), du = top ? 1 : rdc[k + 1] * rh1;
        const om = 0.5 * (wl + wu);
        let lo = k > 0 ? nl * dl * rk : 0, up = top ? 0 : nuU * du * rk;
        if (om > 0 && k > 0) lo += dt * om * dl; else if (om < 0 && !top) up -= dt * om * du;
        ta[k] = lo; tc[k] = up; tb[k] = 1 + lo + up + (k === 0 ? gam[q] * rk : 0);
        td[k] = a[k * NA + q] + dt * Ga[k * NA + q] + (top ? tw * rk : 0); te[k] = 1;
        if (sgS !== null) { const sg = sgS[k * NA + q]; if (sg > 0) { tb[k] += dt * sg; td[k] += dt * sg * sgX; } }
        if (tvd && om !== 0) { // the implicit vertical advection is first-order upwind; its limited second-order part is added explicitly (deferred correction)
          const x0 = k * NA + q, ak = a[x0];
          if (om > 0 && k > 0) { const am = a[x0 - NA], fh = top ? 0 : lim(ak - am, a[x0 + NA] - ak), fl = k > 1 ? lim(am - a[x0 - 2 * NA], ak - am) : 0; td[k] -= dt * om * dl * (fh - fl); }
          else if (om < 0 && !top) { const ap = a[x0 + NA], fh = k < K - 2 ? lim(ap - a[x0 + 2 * NA], ak - ap) : 0, fl = k > 0 ? lim(ak - ap, a[x0 - NA] - ak) : 0; td[k] -= dt * om * du * (fh - fl); }
        }
        wl = wu; nl = nuU; dl = du;
      }
      // Thomas algorithm with two right-hand sides
      let c0 = tc[0] / tb[0]; tg[0] = c0; td[0] /= tb[0]; te[0] /= tb[0];
      for (let k = 1; k < K; k++) { const m = 1 / (tb[k] - ta[k] * tg[k - 1]); tg[k] = tc[k] * m; td[k] = (td[k] + ta[k] * td[k - 1]) * m; te[k] = (te[k] + ta[k] * te[k - 1]) * m; }
      for (let k = K - 2; k >= 0; k--) { td[k] += tg[k] * td[k + 1]; te[k] += tg[k] * te[k + 1]; }
      let ma = 0, mb = 0; for (let k = 0; k < K; k++) { ma += ds[k] * td[k]; mb += ds[k] * te[k]; }
      const c = (Ub - ma) / mb, hl = Hf * len;
      for (let k = 0; k < K; k++) { const x = td[k] + c * te[k], ax2 = x < 0 ? -x : x; a[k * NA + q] = x; if (ax2 > amax) amax = ax2; Q[k * NA + q] = ds[k] * (F + hl * (x - Ub)); }
    }
    return amax;
  };
  /** One transport sub-step of tracer m from the layer depths Da to Db (per cell). */
  const transport = (m, dts, w0, w1) => {
    const c = tr[m], mt = meta[m], open = mt.open, dif = mt.diffH;
    acc.fill(0);
    for (let k = 0; k < K; k++) {
      const kn = k * n, oU = k * NU, oV = k * NV, dk = ds[k];
      for (let j = 0; j < ny; j++) {
        const r = j * nu1, c0 = j * nx;
        for (let i = 1; i < nx; i++) {
          const q = r + i, F = Qx[oU + q];
          if (F === 0 && !(dif && tyU[q] === 2)) continue;
          const R = kn + c0 + i, L = R - 1, cl = c[L], cr = c[R];
          let cf;
          if (F >= 0) { cf = cl; if (i > 1 && thick[c0 + i - 1] && thick[c0 + i - 2]) cf += lim(cl - c[L - 1], cr - cl); }
          else { cf = cr; if (i < nx - 1 && thick[c0 + i] && thick[c0 + i + 1]) cf += lim(cr - c[R + 1], cl - cr); }
          let fl = F * cf;
          if (dif && tyU[q] === 2) fl -= ((Kh * dk * Math.min(Dn[c0 + i - 1], Dn[c0 + i]) * dy) / dx) * (cr - cl);
          acc[L] -= fl; acc[R] += fl;
        }
        // open west / east faces
        const Fw = Qx[oU + r], Fe = Qx[oU + r + nx], Pw = kn + c0, Pe = kn + c0 + nx - 1;
        if (Fw > 0) { const val = open(zb[c0] + sc[k] * Dn[c0]); acc[Pw] += Fw * val; mt.inn += Fw * val * dts; } else if (Fw < 0) { acc[Pw] += Fw * c[Pw]; mt.out -= Fw * c[Pw] * dts; }
        if (Fe < 0) { const val = open(zb[c0 + nx - 1] + sc[k] * Dn[c0 + nx - 1]); acc[Pe] -= Fe * val; mt.inn -= Fe * val * dts; } else if (Fe > 0) { acc[Pe] -= Fe * c[Pe]; mt.out += Fe * c[Pe] * dts; }
      }
      for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = j * nx + i, F = Qy[oV + q];
        if (F === 0 && !(dif && tyV[q] === 2)) continue;
        const R = kn + q, L = R - nx, cl = c[L], cr = c[R];
        let cf;
        if (F >= 0) { cf = cl; if (j > 1 && thick[q - nx] && thick[q - 2 * nx]) cf += lim(cl - c[L - nx], cr - cl); }
        else { cf = cr; if (j < ny - 1 && thick[q] && thick[q + nx]) cf += lim(cr - c[R + nx], cl - cr); }
        let fl = F * cf;
        if (dif && tyV[q] === 2) fl -= ((Kh * dk * Math.min(Dn[q - nx], Dn[q]) * dx) / dy) * (cr - cl);
        acc[L] -= fl; acc[R] += fl;
      }
      for (let i = 0; i < nx; i++) { // open south / north faces
        const Fs = Qy[oV + i], Fn = Qy[oV + ny * nx + i], Ps = kn + i, Pn = kn + (ny - 1) * nx + i;
        if (Fs > 0) { const val = open(zb[i] + sc[k] * Dn[i]); acc[Ps] += Fs * val; mt.inn += Fs * val * dts; } else if (Fs < 0) { acc[Ps] += Fs * c[Ps]; mt.out -= Fs * c[Ps] * dts; }
        if (Fn < 0) { const val = open(zb[Pn - kn] + sc[k] * Dn[Pn - kn]); acc[Pn] -= Fn * val; mt.inn -= Fn * val * dts; } else if (Fn > 0) { acc[Pn] -= Fn * c[Pn]; mt.out += Fn * c[Pn] * dts; }
      }
    }
    for (const s of mt.src) { acc[s.idx] += s.rate; mt.injected += s.rate * dts; }
    if (inf) { // near-field water: the effluent plus what the jets entrain from the sink layers at the concentration found there
      const e = inf.eff?.[m] ?? 0;
      for (const cl of inf.cols) {
        const P = cl.P, qe = inf.Q * cl.w;
        if (!thick[P]) { for (let k = 0; k < K; k++) acc[k * n + P] += ds[k] * qe * e; continue; } // thin water: no layer exchange, the effluent alone
        let ent = 0;
        for (const q of cl.sink) { const x = q.k * n + P, fl = (inf.S - 1) * qe * q.f * c[x]; acc[x] -= fl; ent += fl; }
        for (const q of cl.src) acc[q.k * n + P] += (qe * e + ent) * q.f;
      }
      mt.injected += inf.Q * inf.wSum * e * dts;
    }
    // vertical exchange (advection through the σ-surfaces, implicit diffusion) and the update of every column
    for (let P = 0; P < n; P++) {
      if (land[P]) continue;
      const d0 = Do[P], dd = Dn[P] - d0, da = d0 + dd * w0, db = d0 + dd * w1;
      if (!thick[P]) { if (db > 1e-9) for (let k = 0; k < K; k++) { const x = k * n + P; c[x] = (da * c[x] + (dts * acc[x]) / (A * ds[k])) / db; } continue; }
      let cm = 0;
      for (let k = 1; k < K; k++) { const w = W[k * n + P], r = ((w > 0 ? w : -w) * dts) / (A * (ds[k] < ds[k - 1] ? ds[k] : ds[k - 1]) * (da < db ? da : db)); if (r > cm) cm = r; }
      const expl = cm <= 0.8;
      let dzl = 0, wlo = 0;
      for (let k = 0; k < K; k++) {
        const x = k * n + P, top = k === K - 1, wup = top ? 0 : W[x + n], dzu = top ? 0 : (A * KvI[x + n]) / ((sc[k + 1] - sc[k]) * db);
        let lo = dts * dzl, up = dts * dzu, dg = A * ds[k] * db + lo + up, rhs = A * ds[k] * da * c[x] + dts * acc[x];
        if (expl) {
          if (wlo !== 0) { const d = wlo > 0 ? k - 1 : k, xd = d * n + P; let cf = c[xd]; if (wlo > 0) { if (d > 0) cf += lim(cf - c[xd - n], c[x] - cf); } else if (d < K - 1) cf += lim(cf - c[xd + n], c[x - n] - cf); rhs += dts * wlo * cf; }
          if (wup !== 0) { const d = wup > 0 ? k : k + 1, xd = d * n + P; let cf = c[xd]; if (wup > 0) { if (d > 0) cf += lim(cf - c[xd - n], c[x + n] - cf); } else if (d < K - 1) cf += lim(cf - c[xd + n], c[x] - cf); rhs -= dts * wup * cf; }
        } else { if (wlo > 0) lo += dts * wlo; else dg -= dts * wlo; if (wup > 0) dg += dts * wup; else up -= dts * wup; }
        ta[k] = lo; tb[k] = dg; tc[k] = up; td[k] = rhs;
        dzl = dzu; wlo = wup;
      }
      tg[0] = tc[0] / tb[0]; td[0] /= tb[0];
      for (let k = 1; k < K; k++) { const mm = 1 / (tb[k] - ta[k] * tg[k - 1]); tg[k] = tc[k] * mm; td[k] = (td[k] + ta[k] * td[k - 1]) * mm; }
      c[(K - 1) * n + P] = td[K - 1];
      for (let k = K - 2; k >= 0; k--) { td[k] += tg[k] * td[k + 1]; c[k * n + P] = td[k]; }
    }
  };
  /** One step of length dt (not longer than dtStable()). */
  M.step = (dt) => {
    const h = sw.h, tau = typeof swo.tau === 'function' ? swo.tau(M.t) : swo.tau || [0, 0], taum = Math.hypot(tau[0], tau[1]);
    // 1. density anomaly, pressure integral I = ∫_z^η ρ′ dz, vertical mixing coefficients on the layer interfaces
    let rlo = Infinity, rhi = -Infinity;
    for (let P = 0; P < n; P++) {
      Do[P] = h[P];
      if (land[P] || !(h[P] > hmin)) { for (let k = 0; k < K; k++) { rho[k * n + P] = 0; Ip[k * n + P] = 0; } continue; }
      const D = h[P], zbP = zb[P];
      for (let k = 0; k < K; k++) { const x = k * n + P, r = dens(tr, x, zbP + sc[k] * D); rho[x] = r; if (r < rlo) rlo = r; if (r > rhi) rhi = r; }
      const xt = (K - 1) * n + P, rs = rho[xt] + ((rho[xt] - rho[xt - n]) * (1 - sc[K - 1])) / (sc[K - 1] - sc[K - 2]);
      let I = 0.5 * (rs + rho[xt]) * (1 - sc[K - 1]) * D; Ip[xt] = I;
      for (let k = K - 2; k >= 0; k--) { const x = k * n + P; I += 0.5 * (rho[x] + rho[x + n]) * (sc[k + 1] - sc[k]) * D; Ip[x] = I; }
      if (!pp || D < hThin) { for (let k = 1; k < K; k++) { nuI[k * n + P] = nuB + (pp ? nu0 : 0); KvI[k * n + P] = KvB + (pp ? nu0 : 0); } continue; }
      const i = P % nx, j = (P - i) / nx, qu = j * nu1 + i, ub = 0.5 * (u[qu] + u[qu + 1]), vb = 0.5 * (v[P] + v[P + nx]), us = Math.sqrt(cdBot(ds[0] * D) * (ub * ub + vb * vb) + taum);
      let ul = ub, vl = vb;
      for (let k = 1; k < K; k++) {
        const x = k * n + P, uc = 0.5 * (u[k * NU + qu] + u[k * NU + qu + 1]), vc = 0.5 * (v[k * NV + P] + v[k * NV + P + nx]), dz = (sc[k] - sc[k - 1]) * D;
        const N2 = (-gr * (rho[x] - rho[x - n] + (refD ? refD(zbP + sc[k] * D) - refD(zbP + sc[k - 1] * D) : 0))) / dz, S2 = ((uc - ul) * (uc - ul) + (vc - vl) * (vc - vl)) / (dz * dz);
        const nn = VK * us * sf[k] * (1 - sf[k]) * D + nu0;
        if (N2 < 0) { nuI[x] = nn + nuB + nuConv; KvI[x] = nn + KvB + nuConv; }
        else { const r = 1 + (5 * N2) / (S2 + 1e-10), nu = nn / (r * r) + nuB; nuI[x] = nu; KvI[x] = nu / r + KvB; } // = ppMixing(Ri, ν_neutral, ν_b, K_b)
        ul = uc; vl = vc;
      }
    }
    M.drho = rhi > rlo ? rhi - rlo : 0;
    if (inf) { // rate at which the inflow replaces the water of the momentum cells around its source layers (half of the cell inflow to each face)
      for (const x of sgT) { sgU[x[0]] = 0; sgU[x[0] + 1] = 0; sgV[x[1]] = 0; sgV[x[1] + nx] = 0; }
      sgT.length = 0;
      if (inf.u || inf.v) for (const cl of inf.cols) {
        const P = cl.P, i = P % nx, j = (P - i) / nx; if (!(h[P] >= hThin)) continue;
        for (const q of cl.src) {
          const r = (0.5 * inf.S * inf.Q * cl.w * q.f) / (A * ds[q.k]), xu = q.k * NU + j * nu1 + i, xv = q.k * NV + P; sgT.push([xu, xv]);
          if (i > 0 && !land[P - 1]) sgU[xu] += r / Math.max(0.5 * (h[P - 1] + h[P]), hmin); if (i < nx - 1 && !land[P + 1]) sgU[xu + 1] += r / Math.max(0.5 * (h[P + 1] + h[P]), hmin);
          if (j > 0 && !land[P - nx]) sgV[xv] += r / Math.max(0.5 * (h[P - nx] + h[P]), hmin); if (j < ny - 1 && !land[P + nx]) sgV[xv + nx] += r / Math.max(0.5 * (h[P + nx] + h[P]), hmin);
        }
      }
    }
    // 2. explicit tendencies and the forcing of the barotropic mode; 3. barotropic step
    tend(0, dt); tend(1, dt);
    sw.advance(dt, qx, qy);
    const hN = sw.h;
    for (let P = 0; P < n; P++) { Dn[P] = hN[P]; thick[P] = !land[P] && Do[P] >= hThin && Dn[P] >= hThin ? 1 : 0; }
    // 4. internal mode and layer volume fluxes
    M.uMax = solve(0, dt, tau[0]); M.vMax = solve(1, dt, tau[1]);
    // 5. flux through the σ-surfaces from layer continuity, and the layer Courant number
    let cr = 0;
    for (let j = 0, P = 0; j < ny; j++) for (let i = 0, q = j * nu1; i < nx; i++, P++, q++) {
      if (!thick[P]) { for (let k = 0; k <= K; k++) W[k * n + P] = 0; continue; }
      const dV = (A * (Dn[P] - Do[P])) / dt, vm = A * (Do[P] < Dn[P] ? Do[P] : Dn[P]);
      let w = 0;
      for (let k = 0; k < K; k++) {
        const a = Qx[k * NU + q], b = Qx[k * NU + q + 1], c = Qy[k * NV + P], d = Qy[k * NV + P + nx], wn = k === K - 1 ? 0 : w - (b - a + d - c) - ds[k] * dV + (qlay !== null ? qlay[k * n + P] : 0);
        const r = (0.5 * (Math.abs(a) + Math.abs(b) + Math.abs(c) + Math.abs(d))) / (ds[k] * vm);
        if (r > cr) cr = r;
        W[(k + 1) * n + P] = wn; w = wn;
      }
    }
    // 6. tracers, sub-cycled to the layer Courant number
    const ns = clamp(Math.ceil((dt * cr) / 0.7), 1, 40), dts = dt / ns;
    for (let m = 0; m < tr.length; m++) for (let s = 0; s < ns; s++) transport(m, dts, s / ns, (s + 1) / ns);
    M.subSteps += ns; M.t += dt; M.steps++;
    if (!Number.isFinite(M.uMax + M.vMax)) throw new Error('The three-dimensional hydrodynamic solution became unstable — coarsen the 3-D grid or raise the minimum depth.');
  };
  /** Content Σ c·V of tracer m (concentration × m³). */
  M.mass = (m) => { let s = 0; const c = tr[m], h = sw.h; for (let k = 0; k < K; k++) for (let P = 0; P < n; P++) if (!land[P]) s += c[k * n + P] * ds[k] * h[P] * A; return s; };
  /** Height of the centre of layer k in cell P (m, datum of zb). */
  M.z = (P, k) => zb[P] + sc[k] * sw.h[P];
  return M;
}

// ---------------------------------------------------------------------------------------------------
// Atmospheric heat exchange, ecological dose–response, vertical reconstruction of the layer
// ---------------------------------------------------------------------------------------------------
/** Bulk air–sea heat budget (W/m², positive into the sea). Tw, Ta °C; rh %; W m/s; cloud 0–1; solar W/m² (incident). */
export function surfaceHeatFlux({ Tw, Ta, rh = 70, W = 5, cloud = 0.3, solar = 200, P = 101325 }) {
  const SB = 5.670374e-8, TK = 273.15, es = (T) => 611.2 * Math.exp((17.62 * T) / (243.12 + T)), q = (e) => (0.622 * e) / (P - 0.378 * e), We = Math.max(W, 1);
  const sw = 0.94 * solar, lwDown = 0.97 * 0.937e-5 * SB * (Ta + TK) ** 6 * (1 + 0.17 * cloud * cloud), lwUp = 0.97 * SB * (Tw + TK) ** 4;
  const latent = RHO_AIR * 2.45e6 * 1.3e-3 * We * (q(es(Tw)) - (rh / 100) * q(es(Ta))), sensible = RHO_AIR * 1005 * 1.1e-3 * We * (Tw - Ta);
  return { sw, lwDown, lwUp, latent, sensible, net: sw + lwDown - lwUp - latent - sensible };
}
/** Linearised surface heat-exchange coefficient K = −∂Q_net/∂T_w (W/m²·K) and the equilibrium temperature. */
export function heatExchange(a) {
  const Q = (T) => surfaceHeatFlux({ ...a, Tw: T }).net, K = -(Q(a.Tw + 0.25) - Q(a.Tw - 0.25)) / 0.5;
  let lo = -5, hi = 70;
  for (let k = 0; k < 60; k++) { const m = 0.5 * (lo + hi); if (Q(m) > 0) lo = m; else hi = m; }
  return { K, Te: 0.5 * (lo + hi), flux: surfaceHeatFlux(a) };
}
/** Log-logistic dose–response: fraction affected at an excess salinity dS, from the 10 % and 50 % effect levels. */
export function doseResponse(dS, ec10, ec50) {
  if (!(dS > 0) || !(ec50 > 0)) return 0;
  const b = ec10 > 0 && ec10 < ec50 ? Math.log(9) / Math.log(ec50 / ec10) : 4;
  return 1 / (1 + (ec50 / dS) ** b);
}
/** Vertical profile of a bottom-attached layer: half-Gaussian above the bed whose depth integral equals φ·H·C. */
export function layerProfile(C, H, phi, zeta) {
  if (phi >= 0.999) return C;
  const sg = phi * H * Math.sqrt(2 / Math.PI), a = erf(H / (sg * Math.SQRT2));
  return (C / Math.max(a, 1e-9)) * Math.exp((-zeta * zeta) / (2 * sg * sg));
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
const manual = (v) => v.design === 'manual', isSW = (v) => v.hydro === 'sw', isWave = (v) => v.waveModel === 'action', isSlice = (v) => !!v.vslice, isHeat = (v) => !!v.heat, is3D = (v) => !!v.h3d, isFS = (v) => isSW(v) || is3D(v);

const suite = {
  id: 'sea', num: 5, title: 'Brine Discharge into the Sea', short: 'Sea discharge', icon: '🌊',
  tagline: 'Dense-jet near field, bottom density current and tidal far-field dispersion of brine, with mixing-zone, receptor and intake assessment.',
  description: 'Optional modules add a free-surface shallow-water solver with Flather/tidal-elevation/radiation boundaries, a three-dimensional hydrostatic σ-layer model of the plume over the bathymetry with three-dimensional views, a wave-action model, a vertical non-hydrostatic or hydrostatic slice with turbulence closure, atmospheric heat exchange, an ecological dose–response assessment and a seasonal sweep. The near field is solved with an integral model of inclined negatively buoyant jets (volume, momentum, salt and heat conservation with an entrainment closure for jet, plume and cross-flow regimes) for single-port and multiport diffusers, and is cross-checked against the empirical dense-jet coefficients. The diluted brine then spreads as an entraining bottom density current and is carried by tidal, residual and wind-driven currents in a transient far-field advection–dispersion model over the site or a synthetic bathymetry. Results are tested against mixing-zone limits, sensitive receptors and recirculation to the intake.',
  guide: [
    'Enter the brine flow, salinity and temperature (or pull the concentrate from the RO or ZLD suites) and the ambient seawater.',
    'Let the tool size the diffuser (ports, diameter, 60° angle) or enter your own design.',
    'Describe the currents with tidal constituents, a residual current and wind drift, or apply the Global Site Data (bathymetry, currents, tide, wind, waves).',
    'Place the intake and the sensitive receptors relative to the outfall and set the mixing-zone radius and limits.',
    'Optionally switch on, under Model setup, the free-surface shallow-water solver, the three-dimensional hydrostatic model (σ-layers) with its three-dimensional plume views, the wave-action model, the vertical slice of the bottom current, atmospheric heat exchange and the seasonal sweep.',
    'Run. Read the near-field dilution first, then the far-field maps, the receptor time series and the compliance table.',
  ],
  implemented: ['continuity equation', 'momentum equation', 'boussinesq', 'hydrostatic-pressure', 'salinity advection-diffusion', 'temperature transport', 'scalar transport', 'equation of state', 'buoyancy equation', 'jet-integral', 'buoyant-plume', 'entrainment', 'densimetric-froude', 'gaussian plume', 'tidal-harmonic',
    'near-field/far-field', 'integral-plume-hydrodynamic', 'salinity-temperature-density', 'hydrodynamic-water-quality', 'hydrodynamic-particle-tracking', 'eulerian-lagrangian',
    'initial currents field', 'salinity', 'temperature', 'density stratification', 'tracer concentration', 'prescribed current/velocity', 'discharge-flow', 'brine salinity/temperature source', 'seabed no-normal-flow and friction', 'zero-gradient/outflow',
    'outfall and diffuser', 'near-field jet', 'buoyant-plume modelling', 'far-field hydrodynamics', 'salinity transport', 'temperature transport', 'density-driven', 'ocean-current', 'tidal modelling', 'wave effects', 'bathymetry', 'coastal-boundary', 'turbulent mixing', 'stratification', 'particle and contaminant transport', 'seabed interaction', 'plume dilution', 'recirculation towards desalination intakes', 'environmental-threshold', 'ecological exposure',
    'shallow-water equation', 'non-hydrostatic navier-stokes', 'turbulence-closure', 'wave-action', 'cfd-coastal-circulation', 'wave-current interaction', 'tide-wave-current', 'plume-ecological-response', 'nested ocean-outfall',
    'initial sea level', 'turbulence', 'tidal-elevation', 'open-radiation', 'flather', 'free-surface kinematic', 'atmospheric heat-flux', 'wind-stress', 'atmospheric forcing', 'seasonal simulation', 'three-dimensional hydrostatic', 'three-dimensional plume'],
  equationsNote: 'Scope and limits. Near field: steady integral jet model with top-hat profiles in the Boussinesq approximation; the default entrainment coefficients (jet 0.07, plume 0.117, descending-limb enhancement 2.0) are calibrated to the 60° still-water experiments of Roberts, Ferrier & Daviero (1997) and the bottom-layer transition uses their empirical ratios, so angles far from 45–65°, strongly merged jets and shallow water where the jet reaches the surface carry more uncertainty. Intermediate field: one-dimensional Ellison–Turner gravity current. Far field: two-dimensional transport of excess salinity (and, optionally, excess temperature with a bulk atmospheric heat exchange) in a bottom-attached layer that occupies a fixed fraction of the local depth, or in the fully mixed water column, with an optional density-driven down-slope drift. Hydrodynamics: by default a quasi-steady rigid-lid, friction-dominated balance ∇·(H^5/3 ∇η) = 0 scaled in time by tidal harmonics (M2, S2, K1, O1), a residual current and wind drift. The optional shallow-water solver integrates the depth-averaged free-surface equations on the same grid (C grid, finite-volume continuity, semi-implicit θ = 0.55 treatment of the surface gradient with a conjugate-gradient elevation solve and implicit bed friction, so the step follows the current speed and not the gravity-wave speed; wetting–drying with flux limiting, Manning/Chézy/drag friction, Coriolis, wind stress, wave forces) with Flather, clamped-elevation or radiation open boundaries nested in the harmonic outer solution; it is a single-layer model, first-order in the advection terms, without horizontal eddy viscosity, and the tide is damped slightly by the time-step size. Hydrostatic and non-hydrostatic equations are solved in reduced form only: a two-dimensional vertical (along-discharge × depth) rigid-lid Boussinesq slice with a k–ε or Pacanowski–Philander closure, which omits lateral spreading and the alongshore tidal current. The optional three-dimensional model solves the hydrostatic Boussinesq (primitive) equations on the far-field domain with terrain-following σ-layers and a free surface: mode splitting with the semi-implicit shallow-water solver as the external mode, layer momentum equations with the baroclinic pressure gradient of the equation of state (density-Jacobian form), Coriolis, wind stress, log-law bottom drag, implicit vertical mixing (Pacanowski–Philander or constant) and flux-limited transport of the brine fraction (and of ambient salinity and temperature when the ambient is stratified). Its limits: the brine enters as a tracer source in the lowest layers at the end of the near field (no jet momentum, no volume source), the horizontal cells are larger than the near field, the pressure is hydrostatic (the non-hydrostatic item refers to the vertical slice), σ-layers lose accuracy where the bed steps by several layer thicknesses from one cell to the next, the momentum advection is first-order upwind, and open boundaries carry the barotropic tide with the ambient density profile. The three-dimensional plume views (plan maps in three layers, vertical sections, iso-surface elevation and thickness, oblique projection) are drawn from that field. Without this option no three-dimensional flow field is computed and the vertical sections shown with the plan maps are reconstructions that distribute the layer content of the two-dimensional far-field model over the depth; the regulatory assessment always uses the near-field and two-dimensional far-field values, with the 3-D metrics listed beside them. Waves: steady wave-action balance for one monochromatic component (refraction, shoaling, depth-limited breaking, Doppler shift and refraction by the peak tidal current) with the Longuet-Higgins longshore-current balance; no spectrum, diffraction or wind growth. The ecological response is a log-logistic dose–response on excess salinity with an exposure-duration test and indicative default thresholds. The seasonal simulation is a sweep of quasi-steady seasonal ambient states on a coarser far-field grid, not a continuous annual integration.',

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
    { group: 'Hydrodynamics: free-surface shallow-water solver', tab: 'setup', help: 'The default current field is a tidal-harmonic time signal on a rigid-lid friction pattern. The shallow-water option solves the depth-averaged free-surface equations on the far-field grid (finite volumes, wetting and drying) and drives the transport with the computed currents; the tidal harmonics then act as the outer model that supplies the open-boundary data.', fields: [
      SEL('hydro', 'Current field for the far field', 'harmonic', [['harmonic', 'Tidal harmonics on a rigid-lid pattern (fast)'], ['sw', 'Shallow-water equations (free surface, wetting–drying)']], 'The shallow-water run takes about twice as long as the harmonic field at the default hydrodynamic resolution.'),
      SEL('swBC', 'Open-boundary condition', 'flather', [['flather', 'Flather: tidal elevation and current, radiating'], ['elev', 'Clamped tidal elevation'], ['rad', 'Radiation only (no tide: wind-, wave- and Coriolis-driven flow)']], 'Flather boundaries let outgoing waves leave while imposing the outer tidal elevation and current; clamped boundaries prescribe the elevation only.', { showIf: isFS }),
      F('eta0', 'Initial sea level above mean sea level', 'm', 0, -3, 5, 'Still-water level at the start (storm surge or datum offset); also the mean level of the boundary tide.', { showIf: isFS }),
      F('etaLag', 'High water after peak flood current', '°', 0, -180, 180, '0° = progressive tidal wave (high water at peak flood); 90° = standing wave (high water at slack).', { showIf: isFS }),
      F('lat', 'Latitude (Coriolis)', '°', 25, -80, 80, 'Sets the Coriolis parameter f = 2Ω sin(latitude); 0 switches rotation off.', { showIf: isFS }),
      SEL('swFric', 'Bed-friction law', 'cd', [['cd', 'Constant drag coefficient (seabed drag above)'], ['manning', 'Manning roughness'], ['chezy', 'Chézy coefficient']], 'c_f = C_d, g n²/h^⅓ or g/C².', { showIf: isSW }),
      F('manN', 'Manning roughness n', 's/m^⅓', 0.025, 0.01, 0.1, 'Sand 0.02–0.025, rock and reef 0.03–0.05.', { showIf: (v) => isSW(v) && v.swFric === 'manning' }),
      F('chezy', 'Chézy coefficient C', 'm^½/s', 55, 20, 120, 'Typically 45–65 for sandy coasts.', { showIf: (v) => isSW(v) && v.swFric === 'chezy' }),
      F('swRef', 'Hydrodynamic cell size ÷ transport cell size', '×', 2, 1, 4, 'The free-surface solver is semi-implicit, so its time step follows the current speed rather than the gravity-wave speed; 2 runs it on cells twice as large (a quarter of the unknowns), and the transports are interpolated conservatively to the transport grid.', { showIf: isSW, step: 1 }),
      F('hDry', 'Minimum (drying) depth', 'm', 0.05, 0.01, 0.5, 'Cell faces shallower than this are closed; cells re-flood when the surface rises.', { showIf: isFS }),
      { key: 'windStress', label: 'Wind stress on the sea surface', type: 'bool', value: true, showIf: isFS, help: 'Surface stress ρ_air C_d W² from the wind inputs (replaces the empirical wind-drift factor).' },
    ] },
    { group: 'Hydrodynamics: three-dimensional hydrostatic model', tab: 'setup', help: 'Optional three-dimensional solution of the hydrostatic Boussinesq (primitive) equations on terrain-following σ-layers with a free surface: the brine is injected in the bottom layers at the end of the near field and spreads under the baroclinic pressure gradient of the equation of state, the tide, the wind and bottom friction. The run adds true three-dimensional views of the plume (plan maps at three levels, vertical sections, threshold volume) and compliance metrics from the 3-D field. The open-boundary condition, initial sea level, tidal phase lag, latitude, minimum depth and wind-stress switch of the shallow-water group above apply to this model as well (they appear there when this option is on).', fields: [
      { key: 'h3d', label: 'Solve the three-dimensional hydrostatic equations (σ-layers, free surface)', type: 'bool', value: false, help: 'Runs after the far field on its own grid; about 20 s at the default 40 × 30 × 8 cells and three tidal cycles.' },
      F('h3nx', 'Cells east–west (3-D grid)', '', 40, 12, 100, 'The 3-D model covers the far-field domain on its own, usually coarser, horizontal grid.', { showIf: is3D, step: 1 }),
      F('h3ny', 'Cells north–south (3-D grid)', '', 30, 10, 80, '', { showIf: is3D, step: 1 }),
      F('h3nz', 'σ-layers over the depth', '', 8, 3, 24, 'Every water column is divided into this number of layers, so the layers thin toward the shore.', { showIf: is3D, step: 1 }),
      SEL('h3Sigma', 'Layer distribution', 'uniform', [['uniform', 'Equal fractions of the depth'], ['bed', 'Refined toward the seabed (bottom layer one third of the top layer)']], 'Bed refinement resolves a thin dense layer with the same number of layers.', { showIf: is3D }),
      SEL('h3Turb', 'Vertical mixing', 'pp', [['pp', 'Richardson-number closure (Pacanowski–Philander) on a parabolic neutral profile'], ['const', 'Constant eddy viscosity and diffusivity']], 'The neutral profile κ u* z (1 − z/D) follows the bottom and wind friction velocities; stable stratification at the top of the brine layer damps it.', { showIf: is3D }),
      F('h3Nu', 'Background vertical eddy viscosity', 'm²/s', 1e-4, 1e-6, 1e-1, 'Added to the closure value (the background diffusivity is one tenth of it); the constant option uses it for both.', { showIf: is3D }),
      F('h3Kh', 'Horizontal eddy viscosity and diffusivity along the layers', 'm²/s', 0.2, 0, 50, 'Turbulent mixing only: the shear dispersion contained in the 2-D dispersion coefficient is resolved by the layers.', { showIf: is3D }),
      SEL('h3Src', 'Coupling of the near field to the 3-D model', 'coupled', [['coupled', 'Volume, salt and momentum at the end of the near field (entrained water withdrawn over the jet height)'], ['volume', 'Volume and salt, no jet momentum'], ['tracer', 'Salt only (tracer source in the bottom layers, no volume or momentum)']], 'Standard near-field → far-field hand-off: the jets entrain (S − 1)·Q of ambient water over their rise height and deliver S·Q of diluted water into the bottom layer at the end of the near field, moving in the discharge direction with the horizontal momentum the integral jet model has at impact. The brine added to the model is exactly the effluent flow in all three options.', { showIf: is3D }),
    ] },
    { group: 'Waves: wave-action balance', tab: 'setup', help: 'Refraction, shoaling and depth-limited breaking of a monochromatic wave over the bathymetry, Doppler shift and refraction by the tidal current, wave-induced mixing and the radiation-stress-driven longshore current.', fields: [
      SEL('waveModel', 'Wave model', 'orbital', [['orbital', 'Local orbital velocity from the wave height (no propagation)'], ['action', 'Wave-action balance over the bathymetry']], 'The wave-action solution feeds the dispersion coefficient and, with the shallow-water solver, the wave forces.'),
      F('waveDir', 'Wave direction (coming from, bearing)', '°', 340, 0, 360, 'The synthetic beach faces north, so waves arrive from northerly bearings.', { showIf: isWave }),
      F('gammaB', 'Breaker index γ = H/h', '–', 0.78, 0.4, 1.2, 'Depth-limited breaking caps the wave height at γ times the local depth.', { showIf: isWave }),
      F('waveCf', 'Bed-friction coefficient under waves', '–', 0.01, 0.001, 0.1, 'Used in the Longuet-Higgins longshore-current balance.', { showIf: isWave }),
      { key: 'waveCur', label: 'Wave–current interaction (waves on the peak tidal current)', type: 'bool', value: true, showIf: isWave, help: 'Solves the action balance with the Doppler-shifted dispersion relation and current refraction; the still-water solution is reported alongside.' },
    ] },
    { group: 'Vertical slice of the dense bottom current', tab: 'setup', help: 'Optional two-dimensional (along-discharge × vertical) Boussinesq simulation over the model bathymetry, fed by the near-field buoyancy flux. It resolves the layer thickness and front speed that the far-field layer model otherwise takes from the integral density-current model.', fields: [
      { key: 'vslice', label: 'Resolve the bottom current in a vertical slice', type: 'bool', value: false, help: 'Adds a few seconds to the run.' },
      SEL('vsModel', 'Pressure treatment', 'nonhydro', [['nonhydro', 'Non-hydrostatic (pressure projection)'], ['hydro', 'Hydrostatic (primitive equations)']], 'The hydrostatic option is the x–z section of the three-dimensional hydrostatic ocean equations; the non-hydrostatic option keeps the vertical acceleration.', { showIf: isSlice }),
      SEL('vsTurb', 'Turbulence closure', 'ke', [['ke', 'k–ε with buoyancy production'], ['pp', 'Pacanowski–Philander (Richardson number)'], ['const', 'Constant eddy viscosity']], '', { showIf: isSlice }),
      F('vsLen', 'Section length', 'm', 600, 100, 20000, 'One fifth of it lies behind the discharge point.', { showIf: isSlice }),
      F('vsTime', 'Simulated time', 'min', 60, 5, 720, '', { showIf: isSlice }),
      F('vsNx', 'Cells along the section', '', 100, 40, 200, '', { showIf: isSlice, step: 1 }),
      F('vsNz', 'Cells over the depth', '', 24, 12, 64, '', { showIf: isSlice, step: 1 }),
      F('vsK0', 'Initial turbulent kinetic energy', 'm²/s²', 1e-5, 1e-9, 1e-2, 'Initial condition of the k–ε closure; the dissipation rate follows from a length scale of one tenth of the depth.', { showIf: (v) => isSlice(v) && v.vsTurb === 'ke' }),
      F('vsNu', 'Background vertical eddy viscosity', 'm²/s', 1e-4, 1e-6, 1e-1, 'Added to the closure value; the constant-viscosity option uses it alone.', { showIf: isSlice }),
    ] },
    { group: 'Atmosphere and heat exchange', tab: 'setup', help: 'Bulk air–sea heat budget and far-field transport of the thermal excess of the brine.', fields: [
      { key: 'heat', label: 'Transport excess temperature with atmospheric heat exchange', type: 'bool', value: false, help: 'Carries a second tracer in the far field; adds about half of the far-field run time.' },
      F('airTemp', 'Air temperature', '°C', 24, -10, 50, '', { showIf: isHeat }),
      F('humidity', 'Relative humidity', '%', 70, 5, 100, '', { showIf: isHeat }),
      F('cloud', 'Cloud cover', 'fraction', 0.3, 0, 1, '', { showIf: isHeat }),
      F('solar', 'Mean incident solar radiation', 'W/m²', 220, 0, 450, 'Daily mean of the global horizontal irradiance.', { showIf: isHeat }),
    ] },
    { group: 'Ecological response', tab: 'setup', help: 'Dose–response of salinity-sensitive species: effect levels as excess salinity above ambient and the share of time the 10 % effect level may be exceeded.', fields: [
      { key: 'species', label: 'Species and thresholds', type: 'table', columns: [{ key: 'name', label: 'Species / community', type: 'text' }, { key: 'ec10', label: 'EC10 ΔS', unit: 'g/kg' }, { key: 'ec50', label: 'EC50 ΔS', unit: 'g/kg' }, { key: 'tol', label: 'Tolerated time above EC10', unit: '%' }],
        value: [{ name: 'Posidonia oceanica (seagrass)', ec10: 1, ec50: 2.5, tol: 25 }, { name: 'Cymodocea nodosa (seagrass)', ec10: 2, ec50: 5, tol: 25 }, { name: 'Reef-building corals', ec10: 1.5, ec50: 4, tol: 10 }, { name: 'Benthic infauna and echinoderms', ec10: 2, ec50: 6, tol: 25 }], help: 'Indicative literature values; replace them with site-specific toxicity data.' },
    ] },
    { group: 'Seasonal simulation', tab: 'setup', help: 'Repeats the near-field and far-field simulation for each season with the diffuser of the main run.', fields: [
      { key: 'seasonal', label: 'Run the seasonal sweep', type: 'bool', value: false, help: 'Adds about one third of the far-field run time (coarser grid, two tidal cycles per season).' },
      { key: 'seasons', label: 'Seasons', type: 'table', showIf: (v) => v.seasonal, columns: [{ key: 'name', label: 'Season', type: 'text' }, { key: 'Ta', label: 'Ambient temperature', unit: '°C' }, { key: 'Sa', label: 'Ambient salinity', unit: 'g/kg' }, { key: 'dT', label: 'Thermal stratification', unit: '°C' }, { key: 'uRes', label: 'Residual current', unit: 'm/s' }, { key: 'windSpeed', label: 'Wind speed', unit: 'm/s' }, { key: 'waveHeight', label: 'Wave height', unit: 'm' }],
        value: [{ name: 'Winter', Ta: 16, Sa: 37.2, dT: 0, uRes: 0.05, windSpeed: 8, waveHeight: 1.4 }, { name: 'Spring', Ta: 19, Sa: 37, dT: 1.5, uRes: 0.03, windSpeed: 6, waveHeight: 0.9 }, { name: 'Summer', Ta: 26, Sa: 37.3, dT: 5, uRes: 0.02, windSpeed: 4, waveHeight: 0.5 }, { name: 'Autumn', Ta: 22, Sa: 37.1, dT: 2, uRes: 0.04, windSpeed: 6, waveHeight: 1 }], help: 'Ambient temperature, salinity, stratification, residual current, wind and waves of each season.' },
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
    { name: 'Free-surface hydrodynamics, waves, heat exchange and a hydrostatic bottom-current slice', values: { hydro: 'sw', waveModel: 'action', heat: true, vslice: true, vsModel: 'hydro', vsTurb: 'pp', Tb: 30, nx: 60, ny: 44, nCycles: 2 } },
    { name: 'Three-dimensional hydrostatic model: brine plume on σ-layers with 3-D views', values: { h3d: true, h3nx: 32, h3ny: 24, h3nz: 6, nCycles: 2 } },
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
    add('tideRange', Number.isFinite(d.tideRange) ? d.tideRange : eta.length > 3 ? +(Math.max(...eta) - Math.min(...eta)).toFixed(2) : undefined, Number.isFinite(d.tideRange) ? 'Tidal range at site' : 'Range of the site tide series'); add('waveHeight', d.waveHeight, 'Wave height at site'); add('wavePeriod', d.wavePeriod, 'Wave period at site'); add('windSpeed', d.windSpeed, 'Wind speed at site'); add('windDir', d.windDir, 'Wind direction at site'); add('waveDir', d.waveDir, 'Wave direction at site'); add('airTemp', d.airTemp, 'Air temperature at site'); add('humidity', d.humidity, 'Relative humidity at site'); add('solar', d.solar ?? (Number.isFinite(d.ghiDaily) ? +((d.ghiDaily * 1000) / 24).toFixed(0) : undefined), 'Solar irradiance at site');
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
    const flow = flowBasis(g), n = nx * ny, betaR = (density(v.Ta, v.Sa + 1) - density(v.Ta, v.Sa)) / density(v.Ta, v.Sa), sEq = (P.rhoB - P.rhoA) / (P.rhoA * betaR); // density excess as equivalent salinity
    // ---- waves: wave-action balance over the bathymetry (optional), with and without the tidal current
    let wf = null;
    if (v.waveModel === 'action' && v.waveHeight > 0) {
      ctx?.progress?.(0.04, 'Waves: wave-action balance');
      const [wx, wy] = bearing(v.waveDir + 180), pk = v.waveCur ? cur.peak : 0, a0x = cur.axis[0], a0y = cur.axis[1], wo = { nx, ny, dx: g.dx, dy: g.dy, h: g.H, H0: v.waveHeight, T: v.wavePeriod, theta0: Math.atan2(wy, wx), gamma: clamp(v.gammaB, 0.3, 1.5), Cf: clamp(v.waveCf, 1e-3, 0.1) };
      const still = waveField(wo);
      wf = pk > 0 ? waveField({ ...wo, init: still, iters: 2 * (nx + ny), U: Float64Array.from(g.H, (_, q) => pk * (a0x * flow.basis[0].u[q] + a0y * flow.basis[1].u[q])), V: Float64Array.from(g.H, (_, q) => pk * (a0x * flow.basis[0].v[q] + a0y * flow.basis[1].v[q])) }) : still;
      for (const a of [wf.Hs, wf.uorb, wf.fx, wf.fy, wf.Vls, wf.theta]) for (let q = 0; q < n; q++) if (!Number.isFinite(a[q])) a[q] = 0;
      wf.still = still; wf.pk = pk; if (ctx?.tick) await ctx.tick();
    }
    // ---- vertical slice along the discharge direction (optional): resolves the dense bottom current
    let vs = null;
    if (v.vslice && dense) {
      const nxs = clamp(Math.round(v.vsNx), 40, 200), nzs = clamp(Math.round(v.vsNz), 12, 64), Ls = clamp(v.vsLen, 50, 20000), dxs = Ls / nxs, s00 = -0.2 * Ls, xsS = Array.from({ length: nxs }, (_, i) => s00 + (i + 0.5) * dxs);
      const zbs = xsS.map((x) => g.fn(x * jx, x * jy)), Hmx = Math.max(2, ...zbs.map((z) => -z)), dzs = Hmx / nzs, zsS = Array.from({ length: nzs }, (_, k) => -Hmx + (k + 0.5) * dzs), solid = new Uint8Array(nxs * nzs), s0 = new Float64Array(nxs * nzs), sAmb = new Float64Array(nxs * nzs);
      const rTop = density(P.amb(0)(depth).T, P.amb(0)(depth).S);
      for (let i = 0; i < nxs; i++) { const dry = -zbs[i] < 0.5; for (let k = 0; k < nzs; k++) { const q = k * nxs + i; solid[q] = dry || (zsS[k] < zbs[i] && k < nzs - 2) ? 1 : 0; const a = P.amb(0)(clamp(zsS[k] + depth, 0, depth)); sAmb[q] = solid[q] ? 0 : (density(a.T, a.S) - rTop) / (P.rhoA * betaR); s0[q] = sAmb[q]; } }
      const bedK = (i) => { let k = 0; while (k < nzs - 1 && solid[k * nxs + i]) k++; return k; }, is = clamp(Math.round((nf.xn - s00) / dxs - 0.5), 1, nxs - 2), kb = bedK(is), nc = clamp(Math.round(nf.yL / dzs), 1, Math.max(1, nzs - kb - 1));
      const qs = (P.Q * sEq) / Math.max(W0, 1), src = [], sRef = sEq / nf.Sn;
      if (!solid[kb * nxs + is]) for (let k = kb; k < kb + nc; k++) src.push({ P: k * nxs + is, rate: qs / (nc * dxs * dzs) });
      const r = await verticalSlice({ nx: nxs, nz: nzs, dx: dxs, dz: dzs, solid, s0, sAmb, beta: betaR, tEnd: clamp(v.vsTime, 1, 1440) * 60, model: v.vsModel, turb: v.vsTurb, nu: clamp(v.vsNu, 1e-6, 1e-1), nu0: 1e-2, nuH: 2e-3 * dxs, Kh: 2e-3 * dxs, Cd: v.Cd, k0: clamp(v.vsK0, 1e-9, 1), src, sScale: sRef, front: { row: -1, i0: is, dir: 1, level: 0.1 * sRef }, p0: 0.05, p1: 0.12 }, ctx);
      // layer thickness h = ∫ s′ dz / s′_bed and bed excess along the section
      const hS = [], sB = [];
      for (let i = 0; i < nxs; i++) { const k0 = bedK(i); let m = 0, I = 0; for (let k = k0; k < nzs; k++) { const q = k * nxs + i; if (solid[q]) continue; const e = Math.max(r.s[q] - sAmb[q], 0); I += e * dzs; if (e > m) m = e; } hS.push(m > 0.02 * sRef ? I / m : 0); sB.push(solid[k0 * nxs + i] ? 0 : Math.max(r.s[k0 * nxs + i] - sAmb[k0 * nxs + i], 0)); }
      const xf = r.hist.front.filter((x) => x !== null).at(-1) ?? null, frontX = xf === null ? xsS[is] : s00 + xf, im = clamp(Math.round((Math.min(xsS[is] + 100, 0.5 * (xsS[is] + frontX)) - s00) / dxs - 0.5), 0, nxs - 1);
      let nuMax = 0; for (let q = 0; q < r.nut.length; q++) if (!solid[q] && r.nut[q] > nuMax) nuMax = r.nut[q];
      vs = { r, nxs, nzs, dxs, dzs, xsS, zsS, solid, sAmb, hS, sB, is, im, sRef, qs, frontX, uf: frontSpeed(r.hist, 0.3, 1), hLayer: hS[im] > 0 ? hS[im] : 0, nuMax, bScale: Math.cbrt(Math.max(G * betaR * qs, 0)) };
    }
    const hL = v.hLayer > 0 ? v.hLayer : vs?.hLayer > 0 ? vs.hLayer : gc ? interp1(gc.x, gc.h, Math.min(150, gcEnd.x)) : nf.yL;
    const phi = v.layer === 'layer' ? clamp(hL / depth, 0.04, 1) : 1, bedF = v.layer === 'layer' ? clamp(v.bedF, 0.1, 1) : 1, uw = waveOrbital(v.waveHeight, v.wavePeriod, depth);
    const K = new Float64Array(n), spd = new Float64Array(n), [ax, ay] = cur.axis;
    for (let j = 0, q = 0; j < ny; j++) for (let i = 0; i < nx; i++, q++) {
      if (!g.H[q]) continue;
      spd[q] = cur.rms * Math.hypot(ax * flow.basis[0].u[q] + ay * flow.basis[1].u[q], ax * flow.basis[0].v[q] + ay * flow.basis[1].v[q]);
      const uwq = wf ? wf.uorb[q] : uw * Math.min(3, (Math.sinh(Math.min((2 * Math.PI * depth) / (v.wavePeriod * Math.sqrt(G * depth)), 20)) / Math.sinh(Math.min((2 * Math.PI * g.H[q]) / (v.wavePeriod * Math.sqrt(G * g.H[q])), 20))) || 1);
      K[q] = v.Kmult * (v.disp === 'const' ? v.K0 : v.disp === 'okubo' ? Math.min(okubo(Math.max(g.dx, Math.hypot(g.xs[i], g.ys[j]))), okubo(5000)) : v.K0 + 0.6 * Math.sqrt(v.Cd) * (wf ? Math.hypot(spd[q], 0.7 * uwq, wf.Vls[q]) : Math.hypot(spd[q], 0.7 * uwq)) * phi * g.H[q]);
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
    // ---- free-surface hydrodynamics (optional): shallow-water solver nested in the tidal-harmonic outer solution
    let swm = null, swI = null;
    const curT = { ...cur, wind: [0, 0] };
    if (v.hydro === 'sw') {
      const mc = clamp(Math.round(v.swRef), 1, 4), ncx = Math.ceil(nx / mc), ncy = Math.ceil(ny / mc), nc = ncx * ncy, zbc = new Float64Array(nc), land = new Uint8Array(nc), cnt = new Float64Array(nc), wetc = new Float64Array(nc);
      const coarse = (a) => { const o = new Float64Array(nc); for (let j = 0, q = 0; j < ny; j++) for (let i = 0; i < nx; i++, q++) o[Math.floor(j / mc) * ncx + Math.floor(i / mc)] += a[q]; for (let q = 0; q < nc; q++) o[q] /= cnt[q] || 1; return o; };
      for (let j = 0, q = 0; j < ny; j++) for (let i = 0; i < nx; i++, q++) { const Q = Math.floor(j / mc) * ncx + Math.floor(i / mc); cnt[Q]++; if (g.H[q]) wetc[Q]++; }
      zbc.set(coarse(g.zb)); for (let q = 0; q < nc; q++) land[q] = 2 * wetc[q] < cnt[q] || zbc[q] > -HMIN ? 1 : 0;
      const fC = 2 * OMEGA_E * Math.sin(clamp(v.lat, -85, 85) * D2R), lagT = (v.etaLag / 360) * TM2, ampS = cur.cons.reduce((a, c) => a + c.amp, 0);
      const fric = v.swFric === 'manning' ? { type: 'manning', n: clamp(v.manN, 0.005, 0.2) } : v.swFric === 'chezy' ? { type: 'chezy', C: clamp(v.chezy, 10, 150) } : { type: 'cd', Cd: v.Cd };
      const cf0 = fric.type === 'manning' ? (G * fric.n ** 2) / Math.cbrt(depth) : fric.type === 'chezy' ? G / fric.C ** 2 : v.Cd, tide = v.swBC !== 'rad';
      const along = (t) => { if (cur.series) { const [a, b] = currentAt(curT, t); return (a * cur.axis[0] + b * cur.axis[1]) / Math.max(cur.peak, 1e-6); } let a = 0; for (const k of cur.cons) a += k.amp * Math.cos((2 * Math.PI * t) / (k.T * 3600) - k.ph); return ampS > 0 ? a / ampS : 0; };
      // external (outer-model) data: harmonic current, tidal elevation and the surface tilt that balances it
      const ext = (t) => { const m = Math.min(1, t / 3600), [U, V] = currentAt(curT, t), [U1, V1] = currentAt(curT, t + 60), [U0, V0] = currentAt(curT, t - 60), dU = (U1 - U0) / 120, dV = (V1 - V0) / 120, r = (cf0 * Math.hypot(U, V)) / depth;
        return { e: v.eta0 + m * 0.5 * v.tideRange * along(t - lagT), gx: (-m * (dU - fC * V + r * U)) / G, gy: (-m * (dV + fC * U + r * V)) / G, U: m * U, V: m * V }; };
      const tw = v.windStress ? windStress(v.windSpeed, v.windDir, P.rhoA) : [0, 0, 0];
      swm = shallowWater({ implicit: true, nx: ncx, ny: ncy, dx: mc * g.dx, dy: mc * g.dy, zb: zbc, land, eta0: v.eta0, hmin: clamp(v.hDry, 0.01, 0.5), f: fC, x0: g.x0, y0: g.y0, fric, tau: [tw[0], tw[1]], force: wf ? { fx: coarse(wf.fx), fy: coarse(wf.fy) } : null, bc: { W: v.swBC, E: v.swBC, S: v.swBC, N: v.swBC }, ext: tide ? ext : null, pat: { au: coarse(flow.basis[0].u), av: coarse(flow.basis[0].v), bu: coarse(flow.basis[1].u), bv: coarse(flow.basis[1].v) } });
      swm.track();
      swI = { V0: swm.volume(), tw, fC, cf0, fric, dt0: swm.dtStable(), tide, mc, ncx, ncy, land, xs: Array.from({ length: ncx }, (_, i) => g.x0 + (i + 0.5) * mc * g.dx), ys: Array.from({ length: ncy }, (_, j) => g.y0 + (j + 0.5) * mc * g.dy), drv: swCoupler(swm, nx, ny, mc, v.eta0) };
    }
    // ---- atmospheric heat exchange (optional): bulk heat budget and surface-exchange decay of the thermal excess
    let heat = null;
    if (v.heat) {
      const hx = heatExchange({ Tw: v.Ta, Ta: v.airTemp, rh: clamp(v.humidity, 0, 100), W: v.windSpeed, cloud: clamp(v.cloud, 0, 1), solar: Math.max(v.solar, 0) }), rc = P.rhoA * cpSea(v.Ta, v.Sa), dT0 = v.Tb - P.amb(0)(P.z0).T, decay = new Float64Array(n), act = phi >= 0.999;
      if (act) for (let q = 0; q < n; q++) if (g.H[q]) decay[q] = Math.max(hx.K, 0) / (rc * g.H[q]);
      heat = { hx, rc, dT0, decay, act };
    }
    const specs = (Array.isArray(v.species) ? v.species : []).filter((r) => +r.ec50 > 0 && +r.ec10 > 0).slice(0, 6).map((r, k) => ({ name: String(r.name || `Species ${k + 1}`).slice(0, 48), ec10: Math.min(+r.ec10, 0.999 * +r.ec50), ec50: +r.ec50, tol: clamp(Number.isFinite(+r.tol) ? +r.tol : 25, 0, 100) }));
    const ff = await farField({ g, flow, cur, K, phi, bedF, scheme: v.scheme, cfl: clamp(v.cfl, 0.05, 0.9), tEnd, tStat, src, rate: P.Q * dS0, probes, ring, drift: v.layer === 'layer' && v.drift && dense ? { betaS, Cd: v.Cd, vmax: 0.4 } : null, thr: 0.1 * P.limit, particles: v.particles ? clamp(v.nPart, 100, 10000) : 0, srcXY: [sx, sy], srcR: Math.max(sg, g.dx), dyn: swm ? swI.drv : null, extra: heat ? { rate: P.Q * heat.dT0, decay: heat.act ? heat.decay : null } : null, expThr: specs.length ? specs.map((q) => q.ec10) : null }, ctx);
    // ---- three-dimensional hydrostatic model (optional): σ-layers, free surface, baroclinic pressure and 3-D transport of the brine
    let h3 = null;
    if (v.h3d) {
      ctx?.progress?.(0.955, 'Three-dimensional hydrostatic model');
      const n3x = clamp(Math.round(v.h3nx), 12, 100), n3y = clamp(Math.round(v.h3ny), 10, 80), K3 = clamp(Math.round(v.h3nz), 3, 24), g3 = makeGrid(v, n3x, n3y), n3 = n3x * n3y, fl3 = flowBasis(g3), land3 = Uint8Array.from(g3.H, (h) => (h > 0 ? 0 : 1));
      const sigma = v.h3Sigma === 'bed' ? Array.from({ length: K3 }, (_, k) => 3 ** (k / (K3 - 1))) : null, hmin3 = clamp(v.hDry, 0.01, 0.5);
      // outer (tidal-harmonic) solution on the open boundaries, as for the shallow-water option
      const fC = 2 * OMEGA_E * Math.sin(clamp(v.lat, -85, 85) * D2R), lagT = (v.etaLag / 360) * TM2, ampS = cur.cons.reduce((a, c) => a + c.amp, 0), tide = v.swBC !== 'rad';
      const along = (t) => { if (cur.series) { const [a, b] = currentAt(curT, t); return (a * cur.axis[0] + b * cur.axis[1]) / Math.max(cur.peak, 1e-6); } let a = 0; for (const k of cur.cons) a += k.amp * Math.cos((2 * Math.PI * t) / (k.T * 3600) - k.ph); return ampS > 0 ? a / ampS : 0; };
      const ext = (t) => { const m = Math.min(1, t / 3600), [U, V] = currentAt(curT, t), [U1, V1] = currentAt(curT, t + 60), [U0, V0] = currentAt(curT, t - 60), dU = (U1 - U0) / 120, dV = (V1 - V0) / 120, r = (v.Cd * Math.hypot(U, V)) / depth;
        return { e: v.eta0 + m * 0.5 * v.tideRange * along(t - lagT), gx: (-m * (dU - fC * V + r * U)) / G, gy: (-m * (dV + fC * U + r * V)) / G, U: m * U, V: m * V }; };
      const tw = v.windStress ? windStress(v.windSpeed, v.windDir, P.rhoA) : [0, 0, 0];
      // tracer 0: volume fraction of brine; with a stratified ambient, tracers 1 and 2 carry the ambient salinity and temperature
      const strat = v.dS !== 0 || v.dT !== 0, zh = (z) => clamp(z + depth, 0, depth) / depth, Sz = (z) => v.Sa + v.dS * (0.5 - zh(z)), Tz = (z) => v.Ta + v.dT * (zh(z) - 0.5), a0 = P.amb(0)(P.z0), dSb = v.Sb - a0.S, dTb = v.Tb - a0.T, rA = density(v.Ta, v.Sa);
      // equation of state: tabulated against the brine fraction for a uniform ambient (4096 intervals, linear interpolation: error below 10⁻⁶ kg/m³), evaluated directly otherwise
      const NT = 4096, rTab = Float64Array.from({ length: NT + 2 }, (_, k) => density(v.Ta + (dTb * k) / NT, v.Sa + (dSb * k) / NT) - rA), rRef = Float64Array.from({ length: 258 }, (_, k) => density(v.Ta + v.dT * (Math.min(k, 256) / 256 - 0.5), v.Sa + v.dS * (0.5 - Math.min(k, 256) / 256))), refD3 = (z) => { const s1 = 256 * zh(z), k = Math.floor(s1); return rRef[k] + (s1 - k) * (rRef[k + 1] - rRef[k]); };
      const dens3 = strat ? (tr, x, z) => density(tr[2][x] + dTb * tr[0][x], tr[1][x] + dSb * tr[0][x]) - refD3(z) : (tr, x) => { const b = tr[0][x]; if (!(b > 0)) return 0; /* round-off undershoots of the fraction must not index below the table */ const s1 = (b < 1 ? b : 1) * NT, k = Math.floor(s1); return rTab[k] + (s1 - k) * (rTab[k + 1] - rTab[k]); };
      const sg3 = Math.max(0.7 * Math.min(g3.dx, g3.dy), 0.4 * W0), dsg = sigma ? sigma.map((w) => w / sigma.reduce((a, b) => a + b, 0)) : new Array(K3).fill(1 / K3);
      let cells = [], sw3 = 0;
      for (let j = 0, q = 0; j < n3y; j++) for (let i = 0; i < n3x; i++, q++) { const r2 = (g3.xs[i] - sx) ** 2 + (g3.ys[j] - sy) ** 2; if (g3.H[q] > 1 && r2 < 9 * sg3 * sg3) { const w = Math.exp(-r2 / (2 * sg3 * sg3)); cells.push({ P: q, w }); sw3 += w; } }
      if (!cells.length) { let best = -1, bd = Infinity; for (let q = 0; q < n3; q++) if (g3.H[q]) { const d2 = (g3.xs[q % n3x] - sx) ** 2 + (g3.ys[(q - (q % n3x)) / n3x] - sy) ** 2; if (d2 < bd) { bd = d2; best = q; } } if (best >= 0) { cells = [{ P: best, w: 1 }]; sw3 = 1; } }
      const src3 = [], cpl = v.h3Src === 'tracer' || v.h3Src === 'volume' ? v.h3Src : 'coupled';
      if (cpl === 'tracer') for (const c of cells) { let nb = 1, th = dsg[0] * g3.H[c.P]; while (nb < K3 && th < Math.min(hL, 0.8 * g3.H[c.P])) th += dsg[nb++] * g3.H[c.P]; let fs = 0; for (let k = 0; k < nb; k++) fs += dsg[k]; for (let k = 0; k < nb; k++) src3.push({ P: c.P, k, rate: (P.Q * c.w * dsg[k]) / (sw3 * fs) }); }
      // near-field hand-off: S_n·Q of near-field water into the layers that the bottom layer of thickness y_L occupies (the surface layers for a
      // jet that ends at the surface), (S_n − 1)·Q of entrained water out of the layers the jet passes through, and the horizontal momentum flux
      // the integral jet model has where it ends, M = S_bulk·Q·V·cos φ, as the velocity M ÷ (S_n·Q) of the inflowing water
      let inflow = null, uNF = 0;
      if (cpl !== 'tracer' && cells.length) {
        const up = jet.fate === 'surface', Lp = jet.path.s.length - 1, cosF = Lp > 0 ? clamp((jet.path.r[Lp] - jet.path.r[Lp - 1]) / Math.max(jet.path.s[Lp] - jet.path.s[Lp - 1], 1e-12), 0, 1) : 1, SnF = Math.max(nf.Sn, 1);
        uNF = cpl === 'coupled' ? Math.min((jet.Sbulk * jet.Vi * cosF) / SnF, jet.Vi) : 0;
        const cols = cells.map((c) => {
          const Hc = g3.H[c.P], hS = Math.min(Math.max(nf.yL, 1e-6), 0.8 * Hc), hJ = Math.min(Math.max(ztAbs, nf.yL, 1e-6), Hc), src = [], sink = [];
          let zl = 0, fs = 0, fk = 0;
          for (let k = 0; k < K3; k++) { const z0k = zl, z1k = zl + dsg[k] * Hc, a = up ? Math.max(0, z1k - Math.max(z0k, Hc - hS)) : Math.max(0, Math.min(z1k, hS) - z0k), b = Math.max(0, Math.min(z1k, hJ) - z0k); if (a > 0) { src.push({ k, f: a }); fs += a; } if (b > 0) { sink.push({ k, f: b }); fk += b; } zl = z1k; }
          src.forEach((q) => (q.f /= fs)); sink.forEach((q) => (q.f /= fk));
          for (const q of src) src3.push({ P: c.P, k: q.k, rate: (SnF * P.Q * c.w * q.f) / sw3 });
          return { P: c.P, w: c.w / sw3, src, sink };
        });
        inflow = { Q: P.Q, S: SnF, u: uNF * jx, v: uNF * jy, eff: strat ? [1, a0.S, a0.T] : [1], cols };
      }
      const tracers = [{ c0: 0, src: inflow ? [] : src3, open: 0 }];
      if (strat) tracers.push({ c0: (q, k, z) => Sz(z), open: Sz, diffH: false }, { c0: (q, k, z) => Tz(z), open: Tz, diffH: false });
      const M = hydro3D({ nx: n3x, ny: n3y, nz: K3, dx: g3.dx, dy: g3.dy, zb: g3.zb, land: land3, sigma, hmin: hmin3, f: fC, rho0: P.rhoA, Cd: v.Cd, z0: depth * Math.exp(-1 - 0.41 / Math.sqrt(v.Cd)), turb: v.h3Turb, nu: clamp(v.h3Nu, 1e-6, 1e-1), Kv: v.h3Turb === 'const' ? clamp(v.h3Nu, 1e-6, 1e-1) : 0.1 * clamp(v.h3Nu, 1e-6, 1e-1), Kh: clamp(v.h3Kh, 0, 50), nuH: clamp(v.h3Kh, 0, 50), dens: dens3, refDensity: strat ? refD3 : null, tracers, inflow,
        sw: { eta0: v.eta0, x0: g3.x0, y0: g3.y0, bc: { W: v.swBC, E: v.swBC, S: v.swBC, N: v.swBC }, ext: tide ? ext : null, pat: { au: fl3.basis[0].u, av: fl3.basis[0].v, bu: fl3.basis[1].u, bv: fl3.basis[1].v }, tau: [tw[0], tw[1]] } });
      const b3 = M.tr[0], A3 = g3.dx * g3.dy, sE = Math.abs(dSb), thr3 = v.thrArea / Math.max(sE, 1e-12), lim3 = P.limit / Math.max(sE, 1e-12), V0 = M.sw.volume();
      const samp = (a, k, x, y) => { // bilinear over wet cells of layer k
        const fi = (x - g3.x0) / g3.dx - 0.5, fj = (y - g3.y0) / g3.dy - 0.5, i = clamp(Math.floor(fi), 0, n3x - 2), j = clamp(Math.floor(fj), 0, n3y - 2), wa = clamp(fi - i, 0, 1), wb = clamp(fj - j, 0, 1), q = j * n3x + i, o3 = k * n3;
        let s = 0, w = 0;
        for (const [d, wt] of [[0, (1 - wa) * (1 - wb)], [1, wa * (1 - wb)], [n3x, (1 - wa) * wb], [n3x + 1, wa * wb]]) if (g3.H[q + d]) { s += wt * a[o3 + q + d]; w += wt; }
        return w > 0 ? s / w : 0;
      };
      const volAbove = (a, D, lv) => { let vol = 0, area = 0, top = 0; for (let q = 0; q < n3; q++) { if (land3[q]) continue; let any = false; for (let k = 0; k < K3; k++) if (a[k * n3 + q] > lv) { vol += M.ds[k] * D[q] * A3; any = true; top = Math.max(top, M.sf[k + 1] * D[q]); } if (any) area += A3; } return { vol, area, top }; };
      const bMax = new Float64Array(K3 * n3), bSum = new Float64Array(K3 * n3), bSnap = new Float64Array(K3 * n3), DSnap = Float64Array.from(M.sw.h), uSnap = new Float64Array(K3 * n3), vSnap = new Float64Array(K3 * n3);
      const ser3 = { t: [], ring: [], vol: [], volLim: [], eta: [], ub: [], us: [], probesB: probes.map(() => []), probesS: probes.map(() => []) }, P3 = g3.jo * n3x + g3.io, NU3 = (n3x + 1) * n3y, NV3 = n3x * (n3y + 1), qU3 = g3.jo * (n3x + 1) + g3.io;
      let nSt = 0, last = -Infinity, volSnap = -1, tSnap3 = 0, ring3 = 0, tRing3 = 0, dtMin = Infinity, dtMaxU = 0, wall = Date.now();
      const snap = () => { bSnap.set(b3); DSnap.set(M.sw.h); for (let k = 0; k < K3; k++) for (let j = 0, q = 0; j < n3y; j++) for (let i = 0; i < n3x; i++, q++) { uSnap[k * n3 + q] = 0.5 * (M.u[k * NU3 + j * (n3x + 1) + i] + M.u[k * NU3 + j * (n3x + 1) + i + 1]); vSnap[k * n3 + q] = 0.5 * (M.v[k * NV3 + q] + M.v[k * NV3 + q + n3x]); } tSnap3 = M.t; };
      while (M.t < tEnd - 1e-6 && M.steps < 60000) {
        const dt = Math.min(M.dtStable(), 150, tEnd - M.t);
        M.step(dt);
        if (dt < dtMin && M.t < tEnd - 1e-6) dtMin = dt; if (dt > dtMaxU) dtMaxU = dt;
        const inSt = M.t >= tStat;
        if (M.t - last >= tEnd / 240 || M.t >= tEnd - 1e-6) {
          last = M.t;
          let rm = 0; for (const [x, y] of ring) rm = Math.max(rm, samp(b3, 0, x, y));
          const va = volAbove(b3, M.sw.h, thr3), vl = volAbove(b3, M.sw.h, lim3);
          ser3.t.push(M.t / 3600); ser3.ring.push(rm * sE); ser3.vol.push(va.vol); ser3.volLim.push(vl.vol); ser3.eta.push(g3.H[P3] ? M.sw.eta[P3] : 0); ser3.ub.push(Math.hypot(0.5 * (M.u[qU3] + M.u[qU3 + 1]), 0.5 * (M.v[P3] + M.v[P3 + n3x]))); ser3.us.push(Math.hypot(0.5 * (M.u[(K3 - 1) * NU3 + qU3] + M.u[(K3 - 1) * NU3 + qU3 + 1]), 0.5 * (M.v[(K3 - 1) * NV3 + P3] + M.v[(K3 - 1) * NV3 + P3 + n3x])));
          probes.forEach((p, k) => { ser3.probesB[k].push(samp(b3, 0, p[0], p[1]) * sE); ser3.probesS[k].push(samp(b3, K3 - 1, p[0], p[1]) * sE); });
          if (inSt) { nSt++; for (let x = 0; x < bMax.length; x++) { const c = b3[x]; if (c > bMax[x]) bMax[x] = c; bSum[x] += c; } }
          if (inSt) { if (rm * sE > ring3) { ring3 = rm * sE; tRing3 = M.t; } if (va.vol > volSnap) { volSnap = va.vol; snap(); } }
          ctx?.progress?.(0.955 + (0.04 * M.t) / tEnd, `Three-dimensional model: ${(M.t / 3600).toFixed(1)} h of ${(tEnd / 3600).toFixed(1)} h`);
          if (ctx?.tick) await ctx.tick();
        }
      }
      if (volSnap < 0) snap();
      for (let x = 0; x < bSum.length; x++) { bSum[x] = nSt ? bSum[x] / nSt : b3[x]; if (!nSt) bMax[x] = b3[x]; }
      // compliance metrics of the three-dimensional field (near-bed layer for the seabed criteria), used by the assessment below
      const k03 = Math.max(ser3.t.findIndex((t) => t * 3600 >= tStat), 0), st3 = (a, thr) => { const b = a.slice(k03), tt = ser3.t.slice(k03); let ex = 0; b.forEach((c, k) => { if (k && c > thr) ex += tt[k] - tt[k - 1]; }); return { max: Math.max(0, ...b), mean: mean(b) || 0, frac: ex / ((tt.at(-1) - tt[0]) || 1) }; };
      let aLim3 = 0, aThr3 = 0, rLim3 = 0;
      for (let q = 0; q < n3; q++) if (!land3[q]) { const c = sE * bMax[q]; if (c > v.thrArea) aThr3 += A3; if (c > P.limit) { aLim3 += A3; rLim3 = Math.max(rLim3, Math.hypot(g3.xs[q % n3x], g3.ys[(q - (q % n3x)) / n3x]) + 0.5 * Math.hypot(g3.dx, g3.dy)); } }
      const m3 = { k0: k03, st3, aLim: aLim3, aThr: aThr3, rLim: rLim3, in3: st3(ser3.probesB[0], P.limit), inS: st3(ser3.probesS[0], P.limit), rec: recs.map((r, k) => st3(ser3.probesB[k + 1], r.thr)) };
      h3 = { m: m3, cpl, inflow, uNF, M, g3, n3x, n3y, n3, K3, land3, sE, thr3, lim3, bMax, bMean: bSum, bSnap, DSnap, uSnap, vSnap, tSnap3, ser3, ring3, tRing3, strat, src3, tw, fC, tide, samp, volAbove, V0, dtMin: Number.isFinite(dtMin) ? dtMin : dtMaxU, dtMaxU, wall: (Date.now() - wall) / 1000, complete: M.t >= tEnd - 1e-3, z0: depth * Math.exp(-1 - 0.41 / Math.sqrt(v.Cd)) };
      if (!h3.complete) W.push({ level: 'warn', msg: `The three-dimensional run stopped after ${M.steps} steps at ${fmt(M.t / 3600, 3)} h — coarsen the 3-D grid.` });
    }
    ctx?.progress?.(0.96, 'Environmental assessment');
    if (!ff.complete) W.push({ level: 'warn', msg: `The far-field run stopped after ${ff.steps} steps at ${fmt(ff.tEnd / 3600, 3)} h — coarsen the grid or raise the CFL number.` });
    // ---- assessment
    const rows2d = (a, sc = 1) => g.ys.map((_, j) => g.xs.map((__, i) => (g.H[j * nx + i] ? sc * a[j * nx + i] : NaN))), mask = g.ys.map((_, j) => g.xs.map((__, i) => !g.H[j * nx + i]));
    // With the three-dimensional model on, every criterion takes the more conservative (larger) of the layer-model and the 3-D value.
    const mzNear = dS0 / dilAt(v.mzR), mzFar = ff.ringMax, mzLay = Math.max(mzNear, mzFar), mzEx = h3 ? Math.max(mzLay, h3.ring3) : mzLay, cellA = g.dx * g.dy;
    const expo = (a, thr) => { let ar = 0, vol = 0, rmax = 0; for (let q = 0; q < n; q++) if (g.H[q] && a[q] > thr) { ar += cellA; vol += phi * g.H[q] * cellA; rmax = Math.max(rmax, Math.hypot(g.xs[q % nx], g.ys[(q - (q % nx)) / nx]) + 0.5 * Math.hypot(g.dx, g.dy)); } return { area: ar, vol, rmax }; };
    const eLim = expo(ff.Cmax, P.limit), eThr = expo(ff.Cmax, v.thrArea), eMeanThr = expo(ff.Cmean, v.thrArea), eMeanLim = expo(ff.Cmean, P.limit);
    let rA = 0;
    if (dS0 / dilAt(0) > P.limit) { const rs = [...linspace(0.5, nf.xn, 80), ...linspace(nf.xn, nf.xn + 0.7 * Math.max(v.Lx, v.Ly), 400)]; rA = rs.find((r) => dS0 / dilAt(r) <= P.limit) ?? rs.at(-1); }
    const compLay = Math.max(rA, eLim.rmax), compliance = h3 ? Math.max(compLay, h3.m.rLim) : compLay, iStat = ff.ser.t.findIndex((t) => t * 3600 >= tStat), tS = ff.ser.t.slice(Math.max(iStat, 0));
    const stats = (s, thr) => { const a = s.slice(Math.max(iStat, 0)); let ex = 0, run = 0, longest = 0; a.forEach((c, k) => { const dtk = k ? tS[k] - tS[k - 1] : 0; if (c > thr) { ex += dtk; run += dtk; longest = Math.max(longest, run); } else run = 0; }); const span = tS.at(-1) - tS[0] || 1; return { max: Math.max(...a, 0), mean: mean(a) || 0, frac: ex / span, longest }; };
    const intake = stats(ff.ser.probes[0], P.limit), intakeMax = h3 ? Math.max(intake.max, h3.m.in3.max, h3.m.inS.max) : intake.max, recStats = recs.map((r, k) => ({ ...r, ...stats(ff.ser.probes[k + 1], r.thr), wet: g.H[clamp(Math.floor((probes[k + 1][1] - g.y0) / g.dy), 0, ny - 1) * nx + clamp(Math.floor((probes[k + 1][0] - g.x0) / g.dx), 0, nx - 1)] > 0 }));
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
    if (intakeMax > 0.02 * v.Sa) W.push({ level: 'bad', msg: `Recirculation: the excess salinity at the intake peaks at ${fmt(intakeMax, 3)} g/kg (${fmt((100 * intakeMax) / v.Sa, 2)} % of ambient), raising RO feed pressure and energy use.` });
    else if (intakeMax > 0.005 * v.Sa) W.push({ level: 'warn', msg: `Some brine returns to the intake: up to ${fmt(intakeMax, 3)} g/kg above ambient.` });
    recStats.forEach((r) => { if (!r.wet) W.push({ level: 'info', msg: `Receptor “${r.name}” lies on land or outside the wet model domain; its values are taken from the nearest water.` }); if (r.max > r.thr) W.push({ level: r.frac > 0.25 ? 'bad' : 'warn', msg: `Receptor “${r.name}”: excess salinity up to ${fmt(r.max, 3)} g/kg exceeds its ${fmt(r.thr, 3)} g/kg threshold for ${fmt(100 * r.frac, 3)} % of the time (longest episode ${fmt(r.longest, 3)} h).` }); });
    // ---- which model governs each criterion when the three-dimensional model is on, and where the two disagree
    const DIS3 = 2, gov3 = h3 ? (() => {
      const pick = (...a) => a.reduce((b, c) => (c[1] > b[1] ? c : b))[0], i3 = Math.max(h3.m.in3.max, h3.m.inS.max);
      const o = { mz: pick(['near-field estimate', mzNear], ['layer model', mzFar], ['3-D model', h3.ring3]), dist: pick(['near-field estimate', rA], ['layer model', eLim.rmax], ['3-D model', h3.m.rLim]), intake: pick(['layer model', intake.max], ['3-D model', i3]), dis: [] };
      const cmp = (name, a, b, floor, unit) => { const lo = Math.min(a, b), hi = Math.max(a, b); if (hi > floor && hi > DIS3 * Math.max(lo, 1e-300)) o.dis.push(`${name}: layer model ${fmt(a, 3)} ${unit}, 3-D model ${fmt(b, 3)} ${unit}`); };
      cmp(`excess salinity at the ${v.mzR} m mixing-zone edge`, mzFar, h3.ring3, 0.1 * P.limit, 'g/kg'); cmp('distance to compliance', eLim.rmax, h3.m.rLim, Math.max(g.dx, h3.g3.dx), 'm'); cmp('maximum excess at the intake', intake.max, i3, 0.1 * P.limit, 'g/kg');
      return o;
    })() : null;
    if (h3) {
      recs.forEach((r, k) => { const q = h3.m.rec[k]; if (q.max > r.thr && !(recStats[k].max > r.thr)) W.push({ level: q.frac > 0.25 ? 'bad' : 'warn', msg: `Receptor “${r.name}”: the three-dimensional model gives a near-bed excess salinity of up to ${fmt(q.max, 3)} g/kg, above its ${fmt(r.thr, 3)} g/kg threshold for ${fmt(100 * q.frac, 3)} % of the time, where the layer model stays below it (${fmt(recStats[k].max, 3)} g/kg).` }); });
      if (gov3.dis.length) W.push({ level: 'warn', msg: `The layer model and the three-dimensional model differ by more than a factor of ${DIS3} — ${gov3.dis.join('; ')}. The assessment uses the more conservative value of each; refine the 3-D grid (the plume must span several cells and layers) and check the layer thickness of the layer model before relying on either.` });
      W.push({ level: 'info', msg: `Regulatory assessment with the three-dimensional model on: each criterion uses the larger of the layer-model and 3-D values — mixing-zone edge governed by the ${gov3.mz}, distance to compliance by the ${gov3.dist}, intake by the ${gov3.intake}.` });
    }
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
    let pk = null;
    if (ff.part) { const dd = ff.part.x.map((x, k) => Math.hypot(x - sx, ff.part.y[k] - sy)).sort((a, b) => a - b), nA = dd.length; pk = { inside: ff.part.released ? (100 * nA) / ff.part.released : 0, d50: nA ? dd[Math.floor(0.5 * (nA - 1))] : 0, d90: nA ? dd[Math.floor(0.9 * (nA - 1))] : 0, age: nA ? mean(ff.part.age) / 3600 : 0, mz: nA ? (100 * dd.filter((q) => q <= v.mzR).length) / nA : 0 }; }
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
        ...(h3 ? [['… three-dimensional model, near-bed layer, tidal maximum on the ring (g/kg)', h3.ring3, P.limit, h3.ring3 <= P.limit ? 'complies' : 'EXCEEDS'], ['… distance to compliance, layer model and near-field estimate (m)', compLay, v.mzR, compLay <= v.mzR ? 'complies' : 'EXCEEDS'], ['… distance to compliance, three-dimensional model (m)', h3.m.rLim, v.mzR, h3.m.rLim <= v.mzR ? 'complies' : 'EXCEEDS'], ['… maximum excess at the intake, layer model (g/kg)', intake.max, 0.02 * v.Sa, intake.max <= 0.02 * v.Sa ? 'acceptable' : 'RECIRCULATION'], ['… maximum excess at the intake, three-dimensional model, bed or surface layer (g/kg)', Math.max(h3.m.in3.max, h3.m.inS.max), 0.02 * v.Sa, Math.max(h3.m.in3.max, h3.m.inS.max) <= 0.02 * v.Sa ? 'acceptable' : 'RECIRCULATION']] : []),
        ['Distance to compliance (m)', compliance, v.mzR, compliance <= v.mzR ? 'complies' : 'EXCEEDS'], ['Maximum excess at the intake (g/kg)', intakeMax, 0.02 * v.Sa, intakeMax <= 0.02 * v.Sa ? 'acceptable' : 'RECIRCULATION'], ['Mean excess at the intake (g/kg)', intake.mean, null, ''],
        ...chem.map((c) => [`${c.name} at the mixing-zone edge (mg/L)`, c.mz, c.lim, c.mz <= c.lim ? 'complies' : 'EXCEEDS']),
        [`Seabed area above the limit, tidal maximum (ha)`, eLim.area / 1e4, null, ''], [`Seabed area above ${v.thrArea} g/kg, tidal maximum (ha)`, eThr.area / 1e4, null, ''], [`Seabed area above ${v.thrArea} g/kg, tidal mean (ha)`, eMeanThr.area / 1e4, null, ''], [`Volume above ${v.thrArea} g/kg, tidal maximum (1000 m³)`, eThr.vol / 1e3, null, ''], ['Seabed area above the limit, tidal mean (ha)', eMeanLim.area / 1e4, null, '']],
        note: `Worst case at the mixing-zone edge occurs ${fmt(ff.tRingMax / 3600, 3)} h into the run at a current speed of ${fmt(uAtMax, 2)} m/s (slack water is ${fmt(cur.min, 2)} m/s). Limit = min(${v.limAbs} g/kg, ${v.limRel} % of ambient).${h3 ? ` Three-dimensional model on: the verdict lines (mixing-zone edge, distance to compliance, maximum at the intake, and the chemical criteria that scale with the mixing-zone value) take the larger of the layer-model and 3-D values; governed by — mixing-zone edge: ${gov3.mz}; distance: ${gov3.dist}; intake: ${gov3.intake}. Lines beginning with “…” show the individual models.` : ''}` },
      { title: 'Receptor and intake exposure (statistics window)', columns: ['Location', 'East (m)', 'North (m)', 'Threshold (g/kg)', 'Maximum ΔS (g/kg)', 'Mean ΔS (g/kg)', 'Time above threshold (%)', 'Longest episode (h)'], rows: [['Intake', v.inX, v.inY, P.limit, intake.max, intake.mean, 100 * intake.frac, intake.longest], ...recStats.map((r) => [r.name, r.x, r.y, r.thr, r.max, r.mean, 100 * r.frac, r.longest])],
        note: `Far-field numerics: ${nx} × ${ny} cells of ${fmt(g.dx, 3)} × ${fmt(g.dy, 3)} m, ${ff.steps} steps with a mean time step of ${fmt(ff.dtMean, 3)} s, ${nCyc} M2 cycle${nCyc > 1 ? 's' : ''} simulated and statistics from ${fmt(tStat / 3600, 3)} h onward. Dispersion coefficient at the outfall ${fmt(K[g.jo * nx + g.io], 2)} m²/s; tidal excursion about ${fmt((cur.rms * Math.SQRT2 * TIDES.M2 * 3600) / Math.PI, 3)} m; bathymetry: ${g.source}.` },
    ];
    if (P.des) tables.push({ title: 'Diffuser design options (60° ports)', columns: ['Ports', 'Diameter (mm)', 'Velocity (m/s)', 'Froude number', 'Rise height (m)', 'Impact dilution', 'Near-field dilution', 'Spacing (m)', 'Diffuser length (m)', 'Assessment'], rows: P.des.rows.filter((r) => r.n <= Math.max(12, 2 * P.des.best.n)).map((r) => [r.n, r.d * 1000, r.V, r.F, r.zt, r.Si, r.Sn, r.s, r.len, (r === P.des.best ? '★ ' : '') + r.why]), note: 'Sized with the empirical 60° coefficients: velocity inside the design window, Froude number above the minimum, jet top below the surface-clearance limit at low water, and the salinity limit met at the end of the near field.' });
    const mJetIn = jet.Q0 * (v.Sb - P.amb(0)(P.z0).S), L = pth.s.length - 1;
    const balances = [{ name: 'Far-field salt excess (g/kg·m³): injected vs stored + exported', in: ff.bal.injected, out: ff.bal.mass + ff.bal.out }];
    if (v.dS === 0 && v.dT === 0) balances.push({ name: 'Jet salt-excess flux (g/kg·m³/s per port)', in: mJetIn, out: pth.S[L] * jet.Q0 * (pth.sal[L] - v.Sa) });
    const outputs = { nearFieldDilution: nf.Sn, impactSalinity: impS, excessAtMixingZone: mzEx, complianceDistance: compliance, outfallLength, nPorts: P.n, portDiameter: P.d, impactDilution: jet.Si, froude: jet.F, exitVelocity: P.U0, riseHeight: jet.zt, mzFarField: mzFar, areaAboveThreshold: eThr.area, intakeExcessMax: intakeMax, intakeExcessMean: intake.mean, limit: P.limit };
    for (const k of Object.keys(outputs)) if (!Number.isFinite(outputs[k])) delete outputs[k];
    // ---- results of the additional physics: reconstructed sections, ecology, hydrodynamics, waves, slice, heat, seasons
    const xK = [], xO = {}, Po = g.jo * nx + g.io, fin = (x) => (Number.isFinite(x) ? x : 0);
    if (h3) { // true views of the three-dimensional solution: plan maps at three levels, vertical sections, threshold volume
      const { M, g3, n3x, n3y, n3, K3, land3, sE, bSnap, DSnap, ser3 } = h3, kM = K3 >> 1, kS = K3 - 1, mask3 = g3.ys.map((_, j) => g3.xs.map((__, i) => !!land3[j * n3x + i]));
      const lay = (a, k, sc = sE) => g3.ys.map((_, j) => g3.xs.map((__, i) => (land3[j * n3x + i] ? NaN : fin(sc * a[k * n3 + j * n3x + i]))));
      const zrows3 = g3.ys.map((_, j) => g3.xs.map((__, i) => g3.zb[j * n3x + i])), iso3 = isolines(zrows3, g3.xs, g3.ys, -HMIN, '#0f172a', 30).map((q) => ({ ...q, width: 1.6 })), shp3 = [...iso3, shapes.at(-1)];
      let zm3 = 1e-6; for (let q = 0; q < n3; q++) if (!land3[q]) { zm3 = Math.max(zm3, sE * bSnap[q]); }
      let zmE = 1e-6; for (let q = 0; q < n3; q++) if (!land3[q]) zmE = Math.max(zmE, sE * h3.bMax[q]);
      const f3 = { type: 'field', xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', x: g3.xs, y: g3.ys, mask: mask3, equal: true, zlabel: 'Excess salinity', zunit: 'g/kg', cmap: 'salinity', zmin: 0, zmax: zm3, shapes: shp3, markers, contours: 6 };
      const tS3 = fmt(h3.tSnap3 / 3600, 3), vecs = (k) => ({ vectors: true, u: g3.ys.map((_, j) => g3.xs.map((__, i) => (land3[j * n3x + i] ? 0 : fin(h3.uSnap[k * n3 + j * n3x + i])))), v: g3.ys.map((_, j) => g3.xs.map((__, i) => (land3[j * n3x + i] ? 0 : fin(h3.vSnap[k * n3 + j * n3x + i])))) });
      // plan map of any σ-layer: one frame per layer, chosen with the slider of the chart (the near-bed layer is shown first)
      const Ho3 = g3.H[g3.jo * n3x + g3.io], layName = (k) => (k === 0 ? 'near-bed layer' : k === kS ? 'surface layer' : `σ-layer ${k + 1} of ${K3}`), layTitle = (k) => `3-D model: excess salinity and current in the ${layName(k)} (t = ${tS3} h, largest plume volume)`;
      plots.push({ ...f3, ...vecs(0), title: layTitle(0), z: lay(bSnap, 0), frameLabel: 'Layer (bed → surface)', frame: 0,
        frames: Array.from({ length: K3 }, (_, k) => ({ label: `${k + 1} of ${K3} · ${fmt(M.sc[k] * Ho3, 3)} m above the bed at the outfall`, title: layTitle(k), z: lay(bSnap, k), ...vecs(k) })),
        note: `Three-dimensional hydrostatic solution on ${n3x} × ${n3y} cells with ${K3} σ-layers; move the slider to step through the layers from the seabed to the surface (the colour scale is the same for all). The near-bed layer is ${fmt(M.ds[0] * Ho3, 3)} m thick at the outfall. Arrows: velocity in the layer.` },
        { ...f3, zmax: zmE, title: '3-D model: near-bed excess salinity, maximum envelope over the tidal cycle', z: lay(h3.bMax, 0) },
        { ...f3, zmax: zmE, title: '3-D model: near-bed excess salinity, tidal mean', z: lay(h3.bMean, 0) });
      // vertical sections through the plume source, sampled from the σ-layers onto level surfaces
      const isx = clamp(Math.floor((sx - g3.x0) / g3.dx), 0, n3x - 1), jsy = clamp(Math.floor((sy - g3.y0) / g3.dy), 0, n3y - 1), nzv = 40;
      const colVal = (a, q, zz) => { const D = DSnap[q], s1 = (zz - g3.zb[q]) / Math.max(D, 1e-9); if (land3[q] || !(D > 0.05) || s1 < 0 || s1 > 1) return NaN; if (s1 <= M.sc[0]) return a[q]; if (s1 >= M.sc[K3 - 1]) return a[(K3 - 1) * n3 + q]; let k = 0; while (k < K3 - 2 && M.sc[k + 1] < s1) k++; const w = (s1 - M.sc[k]) / (M.sc[k + 1] - M.sc[k]); return a[k * n3 + q] * (1 - w) + a[(k + 1) * n3 + q] * w; };
      const sec3 = (alongX, at) => { // section along x through row `at`, or along y through column `at`, over its own depth range
        const m = alongX ? n3x : n3y, idx = (k) => (alongX ? at * n3x + k : k * n3x + at);
        let Hm = 1, eM = 0; for (let k = 0; k < m; k++) if (!land3[idx(k)]) { Hm = Math.max(Hm, -g3.zb[idx(k)]); eM = Math.max(eM, g3.zb[idx(k)] + DSnap[idx(k)]); }
        const y = Array.from({ length: nzv }, (_, k) => -Hm + ((k + 0.5) * (Hm + eM)) / nzv), z = y.map((zz) => Array.from({ length: m }, (_, k) => { const x = colVal(bSnap, idx(k), zz); return Number.isFinite(x) ? fin(sE * x) : NaN; }));
        return { y, z, mask: z.map((r) => r.map((x) => !Number.isFinite(x))) };
      };
      const pickIdx = (m, must) => { const stp = Math.max(1, Math.ceil(m / 40)), a = []; for (let k = 0; k < m; k += stp) a.push(k); if (!a.includes(must)) a.push(must); return a.sort((x, y) => x - y); }, rowsA = pickIdx(n3y, jsy), colsC = pickIdx(n3x, isx);
      const fA = rowsA.map((jj) => ({ at: jj, ...sec3(true, jj) })), fC = colsC.map((ii) => ({ at: ii, ...sec3(false, ii) }));
      let zsec = 1e-6; for (const f of [...fA, ...fC]) for (const r of f.z) for (const x of r) if (x > zsec) zsec = x;
      const bS = { type: 'field', ylabel: 'Elevation (m)', zlabel: 'Excess salinity', zunit: 'g/kg', cmap: 'salinity', zmin: 0, zmax: zsec, contours: 6 }, sA = fA.find((f) => f.at === jsy), sC = fC.find((f) => f.at === isx);
      const tA = (jj) => `3-D model: vertical section west–east (along-shore) at y = ${fmt(g3.ys[jj], 3)} m${jj === jsy ? ', through the plume source' : ''} (t = ${tS3} h)`, tC = (ii) => `3-D model: vertical section south–north (cross-shore) at x = ${fmt(g3.xs[ii], 3)} m${ii === isx ? ', through the plume source' : ''} (t = ${tS3} h)`;
      plots.push({ ...bS, title: tA(jsy), xlabel: 'East of outfall (m)', x: g3.xs, y: sA.y, z: sA.z, mask: sA.mask, frameLabel: 'Section position, north of the outfall', frame: rowsA.indexOf(jsy), frames: fA.map((f) => ({ label: `y = ${fmt(g3.ys[f.at], 3)} m`, title: tA(f.at), y: f.y, z: f.z, mask: f.mask })),
        note: 'Section of the three-dimensional solution: the σ-layer values are interpolated linearly in the vertical onto level surfaces; the seabed and the water above the free surface are masked. The slider moves the section across the model domain; it starts at the plume source.' },
        { ...bS, title: tC(isx), xlabel: 'North of outfall (m)', x: g3.ys, y: sC.y, z: sC.z, mask: sC.mask, frameLabel: 'Section position, east of the outfall', frame: colsC.indexOf(isx), frames: fC.map((f) => ({ label: `x = ${fmt(g3.xs[f.at], 3)} m`, title: tC(f.at), y: f.y, z: f.z, mask: f.mask })) });
      // threshold volume: footprint and top elevation of the water above the reporting threshold, outline of the volume above the limit
      const colMax = new Float64Array(n3), topZ = g3.ys.map(() => new Array(n3x).fill(NaN)), thk = g3.ys.map(() => new Array(n3x).fill(NaN));
      let zTopMin = 0, zTopMax = 0, tkMax = 1e-6;
      for (let j = 0, q = 0; j < n3y; j++) for (let i = 0; i < n3x; i++, q++) { if (land3[q]) continue; let kt = -1, m = 0, th = 0; for (let k = 0; k < K3; k++) { const c = bSnap[k * n3 + q]; if (c > m) m = c; if (c > h3.thr3) { kt = k; th += M.ds[k] * DSnap[q]; } } colMax[q] = sE * m; if (kt >= 0) { topZ[j][i] = g3.zb[q] + M.sf[kt + 1] * DSnap[q]; thk[j][i] = th; zTopMin = Math.min(zTopMin, topZ[j][i]); zTopMax = Math.max(zTopMax, topZ[j][i]); tkMax = Math.max(tkMax, th); } }
      const cmRows = g3.ys.map((_, j) => g3.xs.map((__, i) => (land3[j * n3x + i] ? -1 : colMax[j * n3x + i]))), outl = [...isolines(cmRows, g3.xs, g3.ys, v.thrArea, '#0ea5e9', 20).map((q) => ({ ...q, width: 1.6 })), ...isolines(cmRows, g3.xs, g3.ys, P.limit, '#ef4444', 20).map((q) => ({ ...q, width: 1.8 }))];
      const vS = h3.volAbove(bSnap, DSnap, h3.thr3), vL = h3.volAbove(bSnap, DSnap, h3.lim3), vE = h3.volAbove(h3.bMax, DSnap, h3.thr3), vEL = h3.volAbove(h3.bMax, DSnap, h3.lim3), k0 = Math.max(ser3.t.findIndex((t) => t * 3600 >= tStat), 0), volMean = mean(ser3.vol.slice(k0)) || 0;
      plots.push({ type: 'field', title: `3-D model: plume volume above ${v.thrArea} g/kg — footprint and elevation of its upper surface (t = ${tS3} h)`, xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', x: g3.xs, y: g3.ys, z: topZ, mask: topZ.map((r) => r.map((x) => !Number.isFinite(x))), equal: true, zlabel: 'Top of the plume', zunit: 'm', cmap: 'viridis', zmin: Math.min(zTopMin, -1e-6), zmax: zTopMax, shapes: [...iso3, ...outl, shapes.at(-1)], markers,
        note: `Iso-surface view of the three-dimensional field: the coloured area is the plan footprint of the water above ${v.thrArea} g/kg (${fmt(vS.area / 1e4, 3)} ha, ${fmt(vS.vol / 1e3, 3)} thousand m³) and the colour is the elevation of its upper surface. Blue line: outline at ${v.thrArea} g/kg; red line: outline of the volume above the ${fmt(P.limit, 3)} g/kg limit${vL.vol > 0 ? ` (${fmt(vL.vol / 1e3, 3)} thousand m³)` : ' (none at this time)'}.` },
        { type: 'field', title: `3-D model: thickness of the plume above ${v.thrArea} g/kg (t = ${tS3} h)`, xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', x: g3.xs, y: g3.ys, z: thk, mask: thk.map((r) => r.map((x) => !Number.isFinite(x))), equal: true, zlabel: 'Plume thickness', zunit: 'm', cmap: 'turbo', zmin: 0, zmax: tkMax, shapes: [...iso3, shapes.at(-1)], markers });
      // shaded three-dimensional view: the iso-surface of the excess salinity standing on the shaded seabed. Its elevation in every water
      // column is where the profile through the layer centres falls to the iso-level (the surface is the bed where no layer exceeds it).
      {
        // window: the footprint of the lowest iso-level with a margin of its own size (at least 14 × 10 cells), always including the outfall
        const lv0 = Math.min(0.5 * v.thrArea, P.limit); let wi0 = isx, wi1 = isx, wj0 = jsy, wj1 = jsy;
        for (let q = 0; q < n3; q++) if (!land3[q] && colMax[q] > lv0) { const i = q % n3x, j = (q - i) / n3x; if (i < wi0) wi0 = i; if (i > wi1) wi1 = i; if (j < wj0) wj0 = j; if (j > wj1) wj1 = j; }
        { const mi = Math.max(3, Math.ceil(0.5 * (wi1 - wi0)), Math.ceil((14 - (wi1 - wi0)) / 2)), mj = Math.max(3, Math.ceil(0.5 * (wj1 - wj0)), Math.ceil((10 - (wj1 - wj0)) / 2)); wi0 = Math.max(0, wi0 - mi); wi1 = Math.min(n3x - 1, wi1 + mi); wj0 = Math.max(0, wj0 - mj); wj1 = Math.min(n3y - 1, wj1 + mj); }
        const wxs = g3.xs.slice(wi0, wi1 + 1), wys = g3.ys.slice(wj0, wj1 + 1), win = (f) => wys.map((_, j) => wxs.map((__, i) => f(i + wi0, j + wj0)));
        // viewpoint on the deep-water side, a little to one side, so that rising ground stands behind the plume
        let hx = 0, hy = 0; for (let j = wj0; j <= wj1; j++) for (let i = wi0; i <= wi1; i++) { const zq = Math.min(g3.zb[j * n3x + i], 2); hx += zq * (g3.xs[i] - 0.5 * (wxs[0] + wxs.at(-1))); hy += zq * (g3.ys[j] - 0.5 * (wys[0] + wys.at(-1))); }
        const az3 = Math.hypot(hx, hy) > 0 ? Math.round((((Math.atan2(-hx, -hy) / D2R + 25) % 360) + 360) % 360) : 205;
        const isoTop = (lv) => win((i, j) => { const q = j * n3x + i, D = DSnap[q], zq = Math.min(g3.zb[q], 2); if (land3[q] || !(D > 0.05)) return zq; let kt = -1; for (let k = 0; k < K3; k++) if (bSnap[k * n3 + q] > lv) kt = k; if (kt < 0) return zq; if (kt === K3 - 1) return g3.zb[q] + D; const c0 = bSnap[kt * n3 + q], c1 = bSnap[(kt + 1) * n3 + q]; return g3.zb[q] + (M.sc[kt] + ((c0 - lv) / Math.max(c0 - c1, 1e-300)) * (M.sc[kt + 1] - M.sc[kt])) * D; });
        const bed3 = win((i, j) => Math.min(g3.zb[j * n3x + i], 2)), cM = win((i, j) => (land3[j * n3x + i] ? NaN : colMax[j * n3x + i])), cTop = Math.max(1e-9, ...colMax);
        const lvs = [...new Set([0.5 * v.thrArea, v.thrArea, 2 * v.thrArea, 5 * v.thrArea, P.limit].map((x) => +x.toPrecision(3)))].sort((a, b) => a - b).filter((x) => x === +v.thrArea.toPrecision(3) || x < cTop), kT = Math.max(0, lvs.indexOf(+v.thrArea.toPrecision(3)));
        const lay3 = (lv) => [{ name: `Water above ${fmt(lv, 3)} g/kg`, z: isoTop(lv / Math.max(sE, 1e-12)), c: cM, cmap: 'heat', cmin: 0, cmax: cTop, clabel: 'Largest excess salinity in the column', cunit: 'g/kg', opacity: 0.93, minThickness: 0.02 }];
        const title3 = (lv) => `3-D model: shaded view of the ${fmt(lv, 3)} g/kg iso-surface of the plume over the seabed (t = ${tS3} h)`;
        plots.push({ type: 'surface3d', title: title3(lvs[kT]), xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', zlabel: 'Seabed elevation (m)', xunit: 'm', x: wxs, y: wys, z: bed3, cmap: 'topo', zmid: 0, zmax: 2.5, azimuth: az3, elevation: 32, layers: lay3(lvs[kT]),
          planes: [{ z: 0, name: 'still-water level', color: '#38bdf8' }], markers: [{ x: g3.xs[isx], y: g3.ys[jsy], z0: g3.zb[jsy * n3x + isx], z1: 0, label: 'Outfall', color: '#0f172a' }],
          frameLabel: 'Iso-surface level', frame: kT, frames: lvs.map((lv) => ({ label: `${fmt(lv, 3)} g/kg${Math.abs(lv - P.limit) < 1e-9 * P.limit ? ' (limit)' : ''}`, title: title3(lv), layers: lay3(lv) })),
          note: `Filled, lit polygons drawn back to front: the seabed (land above the still-water level in earth colours) and, on it, the surface that encloses the water above the chosen excess salinity, coloured by the largest excess in the water column. Drag the picture sideways (or use ◀ ▶) to turn it; with a mouse, dragging up and down tilts it. The slider changes the iso-level. The vertical scale is exaggerated (factor in the caption). The view covers the part of the model domain around the plume, ${fmt(wxs[0], 3)} to ${fmt(wxs.at(-1), 3)} m east and ${fmt(wys[0], 3)} to ${fmt(wys.at(-1), 3)} m north of the outfall; the plume holds ${fmt(vS.vol / 1e3, 3)} thousand m³ above ${v.thrArea} g/kg at this time.` });
      }
      plots.push({ type: 'line', title: '3-D model: excess salinity at the intake, receptors and mixing-zone edge', xlabel: 'Time (h)', ylabel: 'Excess salinity (g/kg)', zeroY: true, series: [{ name: 'Mixing-zone edge, near bed (maximum on the ring)', x: ser3.t, y: ser3.ring }, { name: 'Intake, near bed', x: ser3.t, y: ser3.probesB[0] }, { name: 'Intake, surface layer', x: ser3.t, y: ser3.probesS[0], dash: true }, ...recs.map((r, k) => ({ name: `${r.name}, near bed`, x: ser3.t, y: ser3.probesB[k + 1] })), { name: '2-D layer model: mixing-zone edge', x: ff.ser.t, y: ff.ser.ring, dash: true, color: '#94a3b8' }], vlines: [{ x: tStat / 3600, label: 'statistics from here' }] },
        { type: 'line', title: '3-D model: plume volume, free surface and current at the outfall', xlabel: 'Time (h)', ylabel: 'Volume (1000 m³) · elevation (m) · speed (m/s)', series: [{ name: `Volume above ${v.thrArea} g/kg (1000 m³)`, x: ser3.t, y: ser3.vol.map((x) => x / 1e3) }, { name: 'Free-surface elevation (m)', x: ser3.t, y: ser3.eta }, { name: 'Near-bed current speed (m/s)', x: ser3.t, y: ser3.ub }, { name: 'Surface current speed (m/s)', x: ser3.t, y: ser3.us, dash: true }] });
      // vertical profiles at the plume source
      const Ps = jsy * n3x + isx, zP = Array.from({ length: K3 }, (_, k) => g3.zb[Ps] + M.sc[k] * DSnap[Ps]), jx3 = Math.hypot(cur.axis[0], cur.axis[1]) || 1;
      plots.push({ type: 'line', title: `3-D model: vertical profiles at the plume source (t = ${tS3} h)`, xlabel: 'Excess salinity (g/kg) · velocity (dm/s) · eddy diffusivity (cm²/s)', ylabel: 'Elevation (m)', series: [{ name: 'Excess salinity (g/kg)', x: zP.map((_, k) => sE * bSnap[k * n3 + Ps]), y: zP, mode: 'both' }, { name: 'Velocity along the tidal axis (dm/s)', x: zP.map((_, k) => (10 * (h3.uSnap[k * n3 + Ps] * cur.axis[0] + h3.vSnap[k * n3 + Ps] * cur.axis[1])) / jx3), y: zP, mode: 'both' }, { name: 'Tidal-mean excess salinity (g/kg)', x: zP.map((_, k) => sE * h3.bMean[k * n3 + Ps]), y: zP, dash: true }, { name: 'Vertical eddy diffusivity at the end of the run (cm²/s)', x: zP.slice(1).map((_, k) => 1e4 * M.KvI[(k + 1) * n3 + Ps]), y: zP.slice(1).map((z, k) => 0.5 * (z + zP[k])), dash: true }] });
      // compliance metrics from the three-dimensional field
      const { aLim, aThr, rLim, st3, in3, inS } = h3.m,
             eS3 = ser3.eta.slice(k0), rng3 = eS3.length ? Math.max(...eS3) - Math.min(...eS3) : 0, bal3 = M.meta[0], rHC = (() => { let r = 0; for (const c of h3.src3) { const q = c.P, i = q % n3x; for (const d of [1, n3x]) if (q + d < n3 && !land3[q + d] && (d > 1 || i < n3x - 1)) r = Math.max(r, Math.abs(g3.zb[q + d] - g3.zb[q]) / (M.ds[0] * g3.H[q])); } return r; })();
      const Vend = M.sw.volume(), vIn3 = h3.V0 + M.sw.volIn + M.sw.volClamp + M.sw.volSrc;
      tables.push({ title: 'Three-dimensional hydrostatic model: compliance metrics from the 3-D field', columns: ['Quantity', '3-D model', '2-D layer model', 'Limit'], rows: [
        [`Near-bed excess salinity at the ${v.mzR} m mixing-zone edge, tidal maximum (g/kg)`, h3.ring3, mzFar, P.limit], ['Near-bed area above the limit, tidal maximum (ha)', aLim / 1e4, eLim.area / 1e4, null], [`Near-bed area above ${v.thrArea} g/kg, tidal maximum (ha)`, aThr / 1e4, eThr.area / 1e4, null],
        [`Volume above ${v.thrArea} g/kg, tidal maximum envelope (1000 m³)`, vE.vol / 1e3, eThr.vol / 1e3, null], [`Volume above ${v.thrArea} g/kg at the largest plume (1000 m³)`, vS.vol / 1e3, null, null], [`Volume above ${v.thrArea} g/kg, mean over the statistics window (1000 m³)`, volMean / 1e3, null, null], ['Volume above the limit, tidal maximum envelope (1000 m³)', vEL.vol / 1e3, eLim.vol / 1e3, null],
        [`Largest height of the ${v.thrArea} g/kg surface above the bed (m)`, vS.top, phi * depth, null], ['Distance to compliance from the near-bed envelope (m)', rLim, eLim.rmax, v.mzR], ['Maximum excess at the intake, near bed (g/kg)', in3.max, intake.max, 0.02 * v.Sa], ['Maximum excess at the intake, surface layer (g/kg)', inS.max, null, 0.02 * v.Sa], ['Mean excess at the intake, near bed (g/kg)', in3.mean, intake.mean, null],
        ...recs.map((r, k) => { const q = st3(ser3.probesB[k + 1], r.thr); return [`${r.name}: maximum near-bed excess (g/kg) · time above threshold ${fmt(100 * q.frac, 3)} %`, q.max, recStats[k].max, r.thr]; })],
        note: 'The 3-D column is evaluated from the three-dimensional field (near-bed layer for the seabed criteria, all layers for the volumes); the 2-D column repeats the bottom-layer far-field model for comparison. The regulatory assessment above uses, for each criterion, the larger of the layer-model and 3-D values. The cells of the 3-D grid are larger than the near field, so concentrations within about one cell of the source are cell averages.' });
      tables.push({ title: 'Three-dimensional hydrostatic model: set-up and numerics', columns: ['Item', 'Value'], rows: [
        ['Equations', 'Hydrostatic Boussinesq primitive equations, free surface, σ-layers'], ['Grid (cells east–west × north–south × layers)', `${n3x} × ${n3y} × ${K3}`], ['Cell size (m)', `${fmt(g3.dx, 3)} × ${fmt(g3.dy, 3)}`], ['Layer thickness at the outfall, bed / surface (m)', `${fmt(M.ds[0] * g3.H[g3.jo * n3x + g3.io], 3)} / ${fmt(M.ds[K3 - 1] * g3.H[g3.jo * n3x + g3.io], 3)}`],
        ['Open-boundary condition (barotropic mode)', v.swBC === 'flather' ? 'Flather (elevation + current, radiating)' : v.swBC === 'elev' ? 'Clamped tidal elevation' : 'Radiation (no tidal forcing)'], ['Vertical mixing', v.h3Turb === 'pp' ? 'Parabolic neutral profile with Pacanowski–Philander Richardson-number damping' : 'Constant coefficients'], ['Active scalars', h3.strat ? 'Brine fraction, ambient salinity, ambient temperature (3 tracers)' : 'Brine fraction (salinity and temperature excess follow from it in a uniform ambient)'],
        ['Roughness length z₀ of the log-law bottom drag (mm)', 1000 * h3.z0], ['Coriolis parameter f (1/s)', h3.fC], ['Wind stress (N/m²)', Math.hypot(h3.tw[0], h3.tw[1]) * P.rhoA], ['Near-field coupling', h3.cpl === 'coupled' ? 'volume, salt and momentum' : h3.cpl === 'volume' ? 'volume and salt' : 'salt only (tracer source)'], ['Time steps', M.steps], ['Time step, smallest / largest (s)', `${fmt(h3.dtMin, 3)} / ${fmt(h3.dtMaxU, 3)}`], ['Tracer sub-steps per step', M.steps ? M.subSteps / M.steps : 0], ['Source cells × layers', h3.src3.length],
        ['Tidal range at the outfall, computed (m)', rng3], ['Peak near-bed / surface current at the outfall (m/s)', `${fmt(Math.max(0, ...ser3.ub.slice(k0)), 3)} / ${fmt(Math.max(0, ...ser3.us.slice(k0)), 3)}`], ['Bed step ÷ bottom-layer thickness at the source (hydrostatic-consistency number)', rHC],
        ['Brine balance error (relative)', bal3.injected > 0 ? (M.mass(0) + bal3.out - bal3.inn - bal3.injected) / bal3.injected : 0], ['Water-volume balance error (relative)', (Vend - vIn3) / Math.max(h3.V0, 1)], ['Run time of the 3-D model (s)', h3.wall]],
        note: 'Mode splitting: the semi-implicit shallow-water solver carries the free surface, the depth-mean flow, wetting and drying and the open boundaries; the vertical structure follows from the layer momentum equations with the baroclinic pressure gradient of the equation of state. ' + (h3.cpl === 'tracer' ? 'The brine enters as a tracer source in the lowest layers (no volume source and no momentum, as in the 2-D far field).' : `Near-field hand-off: ${fmt(h3.inflow.S * P.Q, 3)} m³/s of near-field water (dilution ${fmt(h3.inflow.S, 3)}) enters the layers within ${fmt(nf.yL, 3)} m of the ${jet.fate === 'surface' ? 'surface' : 'bed'} at the end of the near field${h3.uNF > 0 ? ` with a velocity of ${fmt(h3.uNF, 3)} m/s along the discharge direction (horizontal momentum flux of the jets where the integral model ends)` : ' without momentum'}, and ${fmt((h3.inflow.S - 1) * P.Q, 3)} m³/s of entrained water leaves the layers below ${fmt(Math.min(Math.max(ztAbs, nf.yL), depth), 3)} m at the local concentration (re-entrainment included); the net source is the effluent, ${fmt(P.Q, 4)} m³/s of water carrying exactly that volume of brine.`) });
      balances.push({ name: '3-D model: brine volume (m³), injected + inflow vs stored + exported', in: bal3.injected + bal3.inn, out: M.mass(0) + bal3.out }, { name: `3-D model: water volume (m³), initial + boundary inflow${h3.inflow ? ' + effluent' : ''} vs final`, in: vIn3, out: Vend });
      xK.push({ label: `Plume volume above ${v.thrArea} g/kg (3-D)`, value: vS.vol / 1e3, unit: '1000 m³', help: `Volume of the three-dimensional field above the reporting threshold at its largest extent; ${fmt(vL.vol / 1e3, 3)} thousand m³ above the limit` }, { label: 'Near-bed excess at mixing-zone edge (3-D)', value: h3.ring3, unit: 'g/kg', status: h3.ring3 > P.limit ? 'warn' : 'ok', help: `Tidal maximum on the ring in the bottom σ-layer; 2-D layer model ${fmt(mzFar, 3)} g/kg` }, { label: 'Plume top above the bed (3-D)', value: vS.top, unit: 'm', help: `Highest point of the ${v.thrArea} g/kg surface above the local seabed` }, { label: 'Maximum excess at intake (3-D, bed / surface)', value: `${fmt(in3.max, 3)} / ${fmt(inS.max, 3)}`, unit: 'g/kg' });
      Object.assign(xO, { plumeVolume3D: vS.vol, mz3D: h3.ring3, plumeTop3D: vS.top, intakeExcessMax3D: in3.max, areaAboveThreshold3D: aThr });
      if (rHC > 2) W.push({ level: 'info', msg: `Three-dimensional model: the seabed drops by ${fmt(rHC, 2)} bottom-layer thicknesses from one cell to the next at the source — the σ-layer pressure gradient is less accurate on such steep slopes; refine the 3-D grid horizontally or use fewer layers.` });
      W.push({ level: 'info', msg: `Three-dimensional hydrostatic model: ${n3x} × ${n3y} × ${K3} cells, ${M.steps} steps; near-bed excess at the mixing-zone edge ${fmt(h3.ring3, 3)} g/kg (2-D layer model ${fmt(mzFar, 3)}), plume volume above ${v.thrArea} g/kg ${fmt(vS.vol / 1e3, 3)} thousand m³, top ${fmt(vS.top, 3)} m above the bed.` });
    } else { // vertical sections through the plume source, reconstructed from the layer model (plan view = far-field maps)
      const nzv = 24, isx = clamp(Math.floor((sx - g.x0) / g.dx), 0, nx - 1), jsy = clamp(Math.floor((sy - g.y0) / g.dy), 0, ny - 1);
      const sec = (alongX) => {
        const m = alongX ? nx : ny, idx = (k) => (alongX ? jsy * nx + k : k * nx + isx);
        let Hm = 1; for (let k = 0; k < m; k++) Hm = Math.max(Hm, g.H[idx(k)]);
        const zs = Array.from({ length: nzv }, (_, k) => -Hm + ((k + 0.5) * Hm) / nzv), z = zs.map((zz) => Array.from({ length: m }, (_, k) => { const q = idx(k), Hq = g.H[q]; return !Hq || zz < -Hq ? NaN : fin(layerProfile(ff.Csnap[q], Hq, phi, zz + Hq)); }));
        return { zs, z, mask: z.map((r) => r.map((x) => !Number.isFinite(x))) };
      };
      const sA = sec(true), sC = sec(false), z3 = Math.max(1e-6, ...sA.z.flat().filter(Number.isFinite), ...sC.z.flat().filter(Number.isFinite)), b3 = { type: 'field', ylabel: 'Elevation (m)', zlabel: 'Excess salinity', zunit: 'g/kg', cmap: 'salinity', zmin: 0, zmax: z3 };
      plots.push({ ...b3, title: `Reconstructed plume section, west–east through the plume source (y = ${fmt(g.ys[jsy], 3)} m)`, xlabel: 'East of outfall (m)', x: g.xs, y: sA.zs, z: sA.z, mask: sA.mask, note: 'Reconstruction, not a three-dimensional solution: the layer excess of the two-dimensional far-field model is distributed above the bed as a half-Gaussian whose depth integral equals the transported layer content (uniform when the fully mixed option is chosen).' },
        { ...b3, title: `Reconstructed plume section, south–north (cross-shore) through the plume source (x = ${fmt(g.xs[isx], 3)} m)`, xlabel: 'North of outfall (m)', x: g.ys, y: sC.zs, z: sC.z, mask: sC.mask });
    }
    if (specs.length) { // plume–ecological response: log-logistic dose–response with exposure duration
      const sens = specs.reduce((a, b) => (b.ec10 < a.ec10 ? b : a)), eff = (c, sp) => 100 * doseResponse(c, sp.ec10, sp.ec50);
      const rowsE = specs.map((sp, k) => {
        let a10 = 0, a50 = 0, aT = 0;
        for (let q = 0; q < n; q++) if (g.H[q]) { if (ff.Cmean[q] >= sp.ec10) a10 += cellA; if (ff.Cmean[q] >= sp.ec50) a50 += cellA; if (ff.expo && ff.expo[k][q] > sp.tol / 100) aT += cellA; }
        const worst = recStats.length ? Math.max(...recStats.map((r) => eff(r.mean, sp))) : 0, e = eff(mzEx, sp);
        return [sp.name, sp.ec10, sp.ec50, e, a10 / 1e4, a50 / 1e4, sp.tol, aT / 1e4, worst, aT > 0 || e > 10 ? 'RISK' : 'acceptable'];
      });
      const effMap = g.ys.map((_, j) => g.xs.map((__, i) => (g.H[j * nx + i] ? eff(ff.Cmean[j * nx + i], sens) : NaN))), eMax = Math.max(1, ...effMap.flat().filter(Number.isFinite)), cs = Array.from({ length: 60 }, (_, k) => 0.05 * 10 ** ((k / 59) * 2.6));
      plots.push({ ...fbase, title: `Ecological response: predicted effect on ${sens.name} (tidal-mean exposure)`, z: effMap, zlabel: 'Affected fraction', zunit: '%', cmap: 'turbo', zmin: 0, zmax: eMax, contours: 5 });
      plots.push({ type: 'line', title: 'Dose–response curves and predicted exposure', xlabel: 'Excess salinity (g/kg)', ylabel: 'Affected fraction (%)', logx: true, series: specs.map((sp) => ({ name: sp.name, x: cs, y: cs.map((c) => eff(c, sp)) })), vlines: mzEx > 0.05 ? [{ x: mzEx, label: 'mixing-zone edge' }] : [], hlines: [{ y: 10, label: '10 % effect' }] });
      tables.push({ title: 'Ecological response (dose–response with exposure duration)', columns: ['Species / community', 'EC10 ΔS (g/kg)', 'EC50 ΔS (g/kg)', 'Effect at the mixing-zone edge (%)', 'Area ≥ 10 % effect, tidal mean (ha)', 'Area ≥ 50 % effect, tidal mean (ha)', 'Tolerated time above EC10 (%)', 'Area above EC10 longer than tolerated (ha)', 'Worst receptor effect (%)', 'Status'], rows: rowsE,
        note: 'Log-logistic dose–response fitted through the 10 % and 50 % effect levels of each species, applied to the tidal-mean excess salinity; the exposure-duration test counts the share of the statistics window in which each cell exceeds the EC10. The default thresholds are indicative literature values — replace them with site-specific toxicity data.' });
      const eS = eff(mzEx, sens), aRisk = Math.max(...rowsE.map((r) => r[7]));
      xK.push({ label: 'Ecological effect at mixing-zone edge', value: eS, unit: '%', status: eS > 10 ? 'warn' : 'ok', help: `Dose–response of the most sensitive species (${sens.name})` }, { label: 'Area exceeding tolerated exposure', value: aRisk, unit: 'ha', status: aRisk > 0 ? 'warn' : 'ok', help: 'Largest area, over all species, where the EC10 is exceeded for longer than the tolerated share of time' });
      Object.assign(xO, { ecoEffectMz: eS, ecoAreaAtRisk: aRisk * 1e4 });
      if (aRisk > 0) W.push({ level: 'warn', msg: `Ecological exposure: ${fmt(aRisk, 3)} ha of seabed exceeds a species EC10 for longer than its tolerated share of time (see the ecological-response table).` });
    }
    if (swm) {
      const st = swm.stat, hmin = clamp(v.hDry, 0.01, 0.5), k0 = Math.max(iStat, 0), spS = ff.ser.u.map((u, k) => Math.hypot(u, ff.ser.v[k])), spH = ff.ser.t.map((t) => Math.hypot(...currentAt(curT, t * 3600)));
      const pkS = Math.max(0, ...spS.slice(k0)), pkH = Math.max(0, ...spH.slice(k0)), eS = ff.ser.eta.slice(k0), rngS = eS.length ? Math.max(...eS) - Math.min(...eS) : 0;
      const { ncx, ncy, mc } = swI, rowsC = (fn) => swI.ys.map((_, j) => swI.xs.map((__, i) => fn(j * ncx + i)));
      let spMax = 0; for (let q = 0; q < ncx * ncy; q++) if (st.spd[q] > spMax) spMax = st.spd[q];
      const inter = Number.isFinite(st.wetMin) ? (st.wetMax - st.wetMin) * cellA * mc * mc : 0, Vend = swm.volume(), vIn = swI.V0 + swm.volIn + swm.volClamp, dryM = rowsC((q) => !!swI.land[q] || !(swm.h[q] > hmin));
      const cu = (k) => rowsC((q) => (swI.land[q] || !(swm.h[q] > hmin) ? 0 : fin(swm.cellU(q)[k])));
      plots.push({ type: 'field', title: `Shallow-water solution: free-surface elevation and depth-mean current (t = ${fmt(ff.tEnd / 3600, 3)} h)`, xlabel: 'East of outfall (m)', ylabel: 'North of outfall (m)', x: swI.xs, y: swI.ys, z: rowsC((q) => (swI.land[q] || !(swm.h[q] > hmin) ? NaN : swm.eta[q])), mask: dryM, equal: true, zlabel: 'Surface elevation', zunit: 'm', cmap: 'coolwarm', contours: 8, vectors: true, u: cu(0), v: cu(1), markers,
        note: `Free-surface finite-volume solution with ${v.swBC === 'flather' ? 'Flather' : v.swBC === 'elev' ? 'clamped tidal-elevation' : 'radiation'} open boundaries; land and cells shallower than ${fmt(hmin, 2)} m (dried) are masked.` });
      plots.push({ ...fbase, x: swI.xs, y: swI.ys, mask: rowsC((q) => !!swI.land[q]), title: 'Shallow-water solution: maximum current speed over the run', z: rowsC((q) => (swI.land[q] ? NaN : st.spd[q])), zlabel: 'Speed', zunit: 'm/s', cmap: 'viridis', zmin: 0, zmax: Math.max(spMax, 1e-6), contours: 6 });
      plots.push({ type: 'line', title: 'Outfall: shallow-water solution against the tidal-harmonic outer solution', xlabel: 'Time (h)', ylabel: 'Speed (m/s) · elevation (m)', series: [{ name: 'Current speed, shallow-water solver', x: ff.ser.t, y: spS }, { name: 'Current speed, harmonic input (boundary data)', x: ff.ser.t, y: spH, dash: true }, { name: 'Free-surface elevation η (m)', x: ff.ser.t, y: ff.ser.eta }], vlines: [{ x: tStat / 3600, label: 'statistics from here' }] });
      tables.push({ title: 'Shallow-water hydrodynamics', columns: ['Item', 'Value'], rows: [
        ['Open-boundary condition', v.swBC === 'flather' ? 'Flather (elevation + current, radiating)' : v.swBC === 'elev' ? 'Clamped tidal elevation' : 'Radiation (no tidal forcing)'], ['Hydrodynamic grid (cells) and cell size (m)', `${ncx} × ${ncy}, ${fmt(mc * g.dx, 3)} × ${fmt(mc * g.dy, 3)}`], ['Largest hydrodynamic time step (s)', swI.dt0], ['Hydrodynamic time steps', swm.steps], ['Elevation-solver iterations per step', swm.steps ? swm.solverIters / swm.steps : 0],
        ['Bed friction', swI.fric.type === 'manning' ? `Manning n = ${fmt(swI.fric.n, 3)}` : swI.fric.type === 'chezy' ? `Chézy C = ${fmt(swI.fric.C, 3)}` : `drag coefficient ${fmt(v.Cd, 3)}`], ['Friction coefficient c_f at the outfall depth', swI.cf0], ['Coriolis parameter f (1/s)', swI.fC],
        ['Wind stress (N/m²)', Math.hypot(swI.tw[0], swI.tw[1]) * P.rhoA], ['Initial sea level (m)', v.eta0], ['Peak current at the outfall, shallow-water (m/s)', pkS], ['Peak current at the outfall, harmonic input (m/s)', pkH], ['Tidal range at the outfall, computed (m)', rngS], ['Tidal range, input (m)', swI.tide ? v.tideRange : 0],
        ['Maximum current speed in the domain (m/s)', spMax], ['Intertidal (wetting–drying) area (ha)', inter / 1e4], ['Volume-balance error (relative)', (Vend - vIn) / Math.max(swI.V0, 1)], ['Velocity-limiter events', swm.capped]],
        note: 'The far-field transport is co-stepped with this solver: it uses the time-mean face transports of every transport step (interpolated linearly from the hydrodynamic cells) and a depth that follows their divergence, so a uniform concentration stays uniform while the tide rises and falls.' });
      balances.push({ name: 'Shallow-water volume (m³): initial + boundary inflow vs final', in: vIn, out: Vend });
      xK.push({ label: 'Peak current at outfall (shallow-water)', value: pkS, unit: 'm/s', help: `Harmonic input: ${fmt(pkH, 3)} m/s` }, { label: 'Computed tidal range at outfall', value: rngS, unit: 'm' }, { label: 'Intertidal (wetting–drying) area', value: inter / 1e4, unit: 'ha' });
      Object.assign(xO, { swPeakCurrent: pkS, swTidalRange: rngS, swMaxSpeed: spMax });
      W.push({ level: 'info', msg: `Free-surface shallow-water solution: peak current at the outfall ${fmt(pkS, 3)} m/s (harmonic input ${fmt(pkH, 3)} m/s), computed tidal range ${fmt(rngS, 3)} m, ${swm.steps} semi-implicit hydrodynamic steps.${swI.tide ? '' : ' Radiation boundaries carry no tide: the flow is driven by wind, waves and Coriolis only.'}` });
      if (swm.capped > 50) W.push({ level: 'warn', msg: `The shallow-water velocity limiter acted ${swm.capped} times (very shallow or steep cells) — raise the minimum depth or smooth the bathymetry.` });
    }
    if (wf) {
      const HsO = wf.Hs[Po], Hs0 = wf.still.Hs[Po], th = wf.theta[Po], turn = (((th - Math.atan2(bearing(v.waveDir + 180)[1], bearing(v.waveDir + 180)[0])) / D2R + 540) % 360) - 180, surfA = wf.surfCells * cellA, isx = g.io;
      const dir = (fn) => g.ys.map((_, j) => g.xs.map((__, i) => (g.H[j * nx + i] ? fn(wf.theta[j * nx + i]) : 0)));
      plots.push({ ...fbase, title: 'Waves: significant wave height and direction (wave-action balance)', z: rows2d(wf.Hs), zlabel: 'Wave height', zunit: 'm', cmap: 'viridis', zmin: 0, zmax: Math.max(1e-6, ...Array.from(wf.Hs)), contours: 6, vectors: true, u: dir(Math.cos), v: dir(Math.sin), note: `Refraction, shoaling and depth-limited breaking (H ≤ ${fmt(v.gammaB, 3)} h) of a ${fmt(v.wavePeriod, 3)} s wave${wf.pk > 0 ? `, on the peak tidal current of ${fmt(wf.pk, 2)} m/s` : ''}. Arrows: direction of propagation.` });
      plots.push({ ...fbase, title: 'Waves: radiation-stress-driven longshore current (Longuet-Higgins)', z: rows2d(Float64Array.from(wf.Vls, Math.abs)), zlabel: 'Longshore current', zunit: 'm/s', cmap: 'turbo', zmin: 0, zmax: Math.max(wf.VlsMax, 1e-6), contours: 5, note: 'Non-zero inside the surf zone only: alongshore radiation-stress force balanced by the wave-averaged bed shear stress.' });
      const col = (a) => g.ys.map((_, j) => (g.H[j * nx + isx] ? fin(a[j * nx + isx]) : null));
      plots.push({ type: 'line', title: 'Waves: cross-shore transect through the outfall', xlabel: 'North of outfall (m)', ylabel: 'm · m/s', series: [{ name: 'Wave height', x: g.ys, y: col(wf.Hs) }, { name: 'Wave height without current', x: g.ys, y: col(wf.still.Hs), dash: true }, { name: 'Breaking limit γ·h', x: g.ys, y: g.ys.map((_, j) => (g.H[j * nx + isx] ? Math.min(v.gammaB * g.H[j * nx + isx], 3 * v.waveHeight) : null)), dash: true }, { name: 'Near-bed orbital velocity (m/s)', x: g.ys, y: col(wf.uorb) }, { name: 'Longshore current (m/s)', x: g.ys, y: col(Float64Array.from(wf.Vls, Math.abs)) }] });
      tables.push({ title: 'Wave transformation (wave-action balance)', columns: ['Item', 'Value'], rows: [
        ['Incident wave height at the deepest boundary (m)', v.waveHeight], ['Period (s)', v.wavePeriod], ['Direction, coming from (°)', v.waveDir], ['Wavelength at the outfall (m)', wf.k[Po] > 0 ? (2 * Math.PI) / wf.k[Po] : 0], ['Wave height at the outfall (m)', HsO], ['… without the tidal current (m)', Hs0], ['Change by wave–current interaction (%)', Hs0 > 0 ? 100 * (HsO / Hs0 - 1) : 0],
        ['Refraction turning at the outfall (°)', turn], ['Near-bed orbital velocity at the outfall (m/s)', wf.uorb[Po]], ['… local estimate without refraction/shoaling (m/s)', uw], ['Breaker height (m)', wf.HbMax], ['Surf-zone area (ha)', surfA / 1e4], ['Maximum longshore current (m/s)', wf.VlsMax]],
        note: 'The orbital velocity and the longshore current of this solution replace the local estimate in the Elder dispersion coefficient; with the shallow-water solver the radiation-stress forces also drive the computed currents (wave set-up and longshore flow).' });
      xK.push({ label: 'Wave height at the outfall', value: HsO, unit: 'm', help: `Incident ${fmt(v.waveHeight, 3)} m; ${fmt(Hs0, 3)} m without the tidal current` }, { label: 'Breaker height', value: wf.HbMax, unit: 'm' }, { label: 'Maximum longshore current', value: wf.VlsMax, unit: 'm/s' });
      Object.assign(xO, { waveHeightOutfall: HsO, breakerHeight: wf.HbMax, longshoreCurrent: wf.VlsMax });
    }
    if (vs) {
      const { r, xsS, zsS, nxs, nzs, solid, sAmb } = vs, cell = (fn) => zsS.map((_, k) => xsS.map((__, i) => fn(k * nxs + i, i, k))), maskS = cell((q) => !!solid[q]);
      const exc = cell((q) => (solid[q] ? NaN : fin(r.s[q] - sAmb[q]))), eM = Math.max(1e-6, ...exc.flat().filter(Number.isFinite)), hyd = v.vsModel === 'hydro';
      const xr = xsS.map((x) => x - xsS[vs.is]), tMin = r.hist.t.map((t) => t / 60), fr = r.hist.front.map((x) => (x === null ? null : x - (xsS[vs.is] - xsS[0] + 0.5 * vs.dxs)));
      plots.push({ type: 'field', title: `Vertical slice (${hyd ? 'hydrostatic' : 'non-hydrostatic'}): density excess and flow after ${fmt(r.t / 60, 3)} min`, xlabel: 'Distance along the discharge direction (m)', ylabel: 'Elevation (m)', x: xsS, y: zsS, z: exc, mask: maskS, zlabel: 'Density excess (equivalent salinity)', zunit: 'g/kg', cmap: 'salinity', zmin: 0, zmax: eM, vectors: true,
        u: cell((q, i, k) => (solid[q] ? 0 : fin(0.5 * (r.u[k * r.nu1 + i] + r.u[k * r.nu1 + i + 1])))), v: cell((q) => (solid[q] ? 0 : fin(0.5 * (r.w[q] + r.w[q + nxs])))), markers: [{ x: xsS[vs.is], y: zsS[0], label: 'Near-field end (source)' }],
        note: 'Two-dimensional section: lateral spreading is not represented, so concentrations far from the source are upper bounds. The section follows the discharge bearing over the model bathymetry.' });
      if (v.vsTurb !== 'const') plots.push({ type: 'field', title: `Vertical slice: eddy viscosity (${v.vsTurb === 'ke' ? 'k–ε closure' : 'Pacanowski–Philander closure'})`, xlabel: 'Distance along the discharge direction (m)', ylabel: 'Elevation (m)', x: xsS, y: zsS, z: cell((q) => (solid[q] ? NaN : fin(r.nut[q] * 1e4))), mask: maskS, zlabel: 'Eddy viscosity', zunit: 'cm²/s', cmap: 'turbo' });
      plots.push({ type: 'line', title: 'Vertical slice: front of the dense bottom current', xlabel: 'Time (min)', ylabel: 'Front distance beyond the near-field end (m)', series: [{ name: 'Front position', x: tMin, y: fr }], note: `Front speed ${fmt(vs.uf, 3)} m/s (least-squares over the last 70 % of the run); buoyancy-flux scale (g′q)^⅓ = ${fmt(vs.bScale, 3)} m/s.` });
      plots.push({ type: 'line', title: 'Vertical slice: layer thickness and bed excess along the section', xlabel: 'Distance beyond the near-field end (m)', ylabel: 'Thickness (m) · excess (g/kg)', series: [{ name: 'Layer thickness, slice', x: xr, y: vs.hS }, { name: 'Bed density excess, slice (g/kg)', x: xr, y: vs.sB }, ...(gc ? [{ name: 'Layer thickness, integral density-current model', x: gc.x, y: gc.h, dash: true }] : [])] });
      const xm = xr[vs.im], hI = gc ? interp1(gc.x, gc.h, clamp(xm, 0, gcEnd.x)) : null, uI = gc ? interp1(gc.x, gc.U, clamp(xm, 0, gcEnd.x)) : null;
      tables.push({ title: 'Vertical slice: resolved dense bottom current', columns: ['Item', 'Value'], rows: [
        ['Equations', hyd ? 'Hydrostatic Boussinesq (baroclinic pressure, rigid lid)' : 'Non-hydrostatic Boussinesq (pressure projection)'], ['Turbulence closure', v.vsTurb === 'ke' ? 'k–ε with buoyancy production' : v.vsTurb === 'pp' ? 'Pacanowski–Philander Ri-dependent mixing' : 'constant eddy viscosity'], ['Grid (cells along × vertical)', `${nxs} × ${nzs}`], ['Cell size Δx × Δz (m)', `${fmt(vs.dxs, 3)} × ${fmt(vs.dzs, 3)}`], ['Time steps', r.steps],
        ['Buoyancy source per metre width, g′q (m³/s³)', G * betaR * vs.qs], ['Front speed, slice (m/s)', vs.uf], ['Buoyancy-flux velocity scale (g′q)^⅓ (m/s)', vs.bScale], ['Front speed ÷ (g′q)^⅓', vs.bScale > 0 ? vs.uf / vs.bScale : 0], ['Layer velocity, integral model (m/s)', uI], [`Layer thickness ${fmt(xm, 3)} m beyond the near field, slice (m)`, vs.hS[vs.im]], ['… integral density-current model (m)', hI],
        ['Maximum eddy viscosity (m²/s)', vs.nuMax], ['Initial turbulent kinetic energy (m²/s²)', v.vsTurb === 'ke' ? v.vsK0 : null], ['Density-excess conservation error (relative)', (r.salt - r.salt0 - r.injected) / Math.max(Math.abs(r.injected), 1e-12)], ['Largest velocity divergence (1/s)', r.divMax], ['Pressure iterations per step', hyd ? 0 : r.pIter / Math.max(r.steps, 1)]],
        note: v.hLayer > 0 ? 'The far-field layer thickness was entered by hand; the slice result is shown for comparison.' : 'The layer thickness resolved by the slice sets the thickness of the far-field layer (two-way nesting of the section in the coastal model).' });
      balances.push({ name: 'Vertical slice: density excess (g/kg·m²), initial + injected vs final', in: (r.salt0 + r.injected) * vs.dxs * vs.dzs, out: r.salt * vs.dxs * vs.dzs });
      xK.push({ label: 'Bottom-current front speed (slice)', value: vs.uf, unit: 'm/s', help: `${hyd ? 'Hydrostatic' : 'Non-hydrostatic'} vertical slice` }, { label: 'Layer thickness (slice)', value: vs.hS[vs.im], unit: 'm', help: `${fmt(xm, 3)} m beyond the near-field end` }, { label: 'Maximum eddy viscosity (slice)', value: vs.nuMax * 1e4, unit: 'cm²/s' });
      Object.assign(xO, { sliceFrontSpeed: vs.uf, sliceLayerThickness: vs.hS[vs.im] });
    } else if (v.vslice) W.push({ level: 'info', msg: 'The vertical slice was skipped: the discharge does not form a dense bottom current.' });
    if (heat) {
      const { hx, rc } = heat, fx = hx.flux, eM = Array.from(ff.Emax).filter((_, q) => g.H[q]), lo = Math.min(0, ...eM), hi = Math.max(lo + 1e-6, ...eM), tau = hx.K > 0 ? (rc * depth) / hx.K / 86400 : 0;
      let mzT = 0; for (const [x, y] of ring) { const q = clamp(Math.floor((y - g.y0) / g.dy), 0, ny - 1) * nx + clamp(Math.floor((x - g.x0) / g.dx), 0, nx - 1); if (g.H[q] && Math.abs(ff.Emax[q]) > Math.abs(mzT)) mzT = ff.Emax[q]; }
      const inT = ff.ser.extra[0].slice(Math.max(iStat, 0)), inMax = inT.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0);
      plots.push({ ...fbase, title: 'Far field: excess temperature, tidal extreme', z: rows2d(ff.Emax), zlabel: 'Excess temperature', zunit: '°C', cmap: 'thermal', zmin: lo, zmax: hi, contours: 6, note: heat.act ? `Surface heat exchange removes the excess with an e-folding time of ${fmt(tau, 3)} d at the outfall depth.` : 'Bottom-attached layer: the thermal excess is not in contact with the atmosphere, so no surface exchange is applied (choose the fully mixed option to include it).' });
      plots.push({ type: 'line', title: 'Excess temperature at the intake and receptors', xlabel: 'Time (h)', ylabel: 'Excess temperature (°C)', series: [{ name: 'Intake', x: ff.ser.t, y: ff.ser.extra[0] }, ...recs.map((r, k) => ({ name: r.name, x: ff.ser.t, y: ff.ser.extra[k + 1] }))] });
      tables.push({ title: 'Atmospheric heat budget and thermal plume', columns: ['Item', 'Value'], rows: [
        ['Absorbed solar radiation (W/m²)', fx.sw], ['Atmospheric long-wave radiation (W/m²)', fx.lwDown], ['Back radiation of the sea surface (W/m²)', -fx.lwUp], ['Evaporative (latent) heat flux (W/m²)', -fx.latent], ['Sensible heat flux (W/m²)', -fx.sensible], ['Net surface heat flux into the sea (W/m²)', fx.net],
        ['Surface heat-exchange coefficient K (W/m²·K)', hx.K], ['Equilibrium temperature (°C)', hx.Te], ['Ambient temperature drift (°C/day)', (fx.net * 86400) / (rc * depth)], ['e-folding time of a thermal excess at the outfall depth (d)', tau],
        ['Brine temperature excess at the port (°C)', heat.dT0], ['Thermal excess at the mixing-zone edge, tidal extreme (°C)', mzT], ['Thermal excess at the intake, extreme (°C)', inMax], ['Share of the discharged excess heat released to the atmosphere (%)', Math.abs(ff.bal.injectedE) > 0 ? (100 * ff.bal.lostE) / ff.bal.injectedE : 0]],
        note: 'Bulk formulae: absorbed short-wave radiation, Swinbank clear-sky long-wave radiation with a cloud correction, grey-body back radiation, and Dalton/Stanton-type latent and sensible fluxes with the 10 m wind. The excess temperature is transported with the same fluxes as the salinity excess.' });
      balances.push({ name: 'Far-field excess heat (°C·m³): injected vs stored + exported + released to the air', in: ff.bal.injectedE, out: ff.bal.massE + ff.bal.outE + ff.bal.lostE });
      xK.push({ label: 'Net surface heat flux', value: fx.net, unit: 'W/m²', help: 'Positive into the sea' }, { label: 'Equilibrium temperature', value: hx.Te, unit: '°C' }, { label: 'Thermal excess at mixing-zone edge', value: mzT, unit: '°C' });
      Object.assign(xO, { netHeatFlux: fx.net, thermalExcessMz: mzT, equilibriumTemperature: hx.Te });
    }
    if (v.seasonal) { // seasonal sweep: the same diffuser under each season's ambient conditions (coarser far-field grid)
      const num = (x, d) => (Number.isFinite(+x) && x !== null && x !== '' ? +x : d), seas = (Array.isArray(v.seasons) ? v.seasons : []).slice(0, 6), outS = [];
      for (const [k, sn] of seas.entries()) {
        const sv = { Ta: num(sn.Ta, v.Ta), Sa: num(sn.Sa, v.Sa), dT: num(sn.dT, v.dT), uRes: Math.max(0, num(sn.uRes, v.uRes)), windSpeed: Math.max(0, num(sn.windSpeed, v.windSpeed)), waveHeight: Math.max(0, num(sn.waveHeight, v.waveHeight)) }, name = String(sn.name || `Season ${k + 1}`).slice(0, 24);
        ctx?.progress?.(0.97, `Seasonal sweep: ${name}`);
        const sub = await suite.run({ ...v, ...sv, seasonal: false, vslice: false, h3d: false, heat: false, hydro: 'harmonic', waveModel: 'orbital', particles: false, design: 'manual', nPorts: P.n, dPort: P.d * 1000, spacing: Number.isFinite(P.spacing) ? P.spacing : v.spacing, theta: P.theta / D2R, nx: Math.max(24, Math.round(nx / 2)), ny: Math.max(16, Math.round(ny / 2)), nCycles: Math.min(nCyc, 2) }, { ...(ctx || {}), progress() {} });
        outS.push({ name, ...sv, rho: density(sv.Ta, sv.Sa), o: sub.outputs });
      }
      if (outS.length) {
        const gv = (s, k) => fin(s.o[k] ?? 0), worst = outS.reduce((a, b) => (gv(b, 'excessAtMixingZone') > gv(a, 'excessAtMixingZone') ? b : a));
        tables.push({ title: 'Seasonal simulation', columns: ['Season', 'Ambient T (°C)', 'Ambient S (g/kg)', 'Thermal stratification (°C)', 'Residual current (m/s)', 'Wind (m/s)', 'Waves (m)', 'Ambient density (kg/m³)', 'Froude number', 'Rise height (m)', 'Impact dilution', 'Near-field dilution', 'Excess at mixing-zone edge (g/kg)', 'Maximum excess at intake (g/kg)', 'Area above reporting threshold (ha)', 'Limit met'],
          rows: outS.map((s) => [s.name, s.Ta, s.Sa, s.dT, s.uRes, s.windSpeed, s.waveHeight, s.rho, gv(s, 'froude'), gv(s, 'riseHeight'), gv(s, 'impactDilution'), gv(s, 'nearFieldDilution'), gv(s, 'excessAtMixingZone'), gv(s, 'intakeExcessMax'), gv(s, 'areaAboveThreshold') / 1e4, gv(s, 'excessAtMixingZone') <= gv(s, 'limit') ? 'yes' : 'NO']),
          note: `Each season re-runs the near-field model and a far-field simulation (${Math.max(24, Math.round(nx / 2))} × ${Math.max(16, Math.round(ny / 2))} cells, ${Math.min(nCyc, 2)} tidal cycle${Math.min(nCyc, 2) > 1 ? 's' : ''}) with the diffuser of the main run held fixed.` });
        plots.push({ type: 'bar', title: 'Seasonal simulation: excess salinity at the mixing-zone edge and at the intake', ylabel: 'Excess salinity (g/kg)', categories: outS.map((s) => s.name), series: [{ name: 'Mixing-zone edge', values: outS.map((s) => gv(s, 'excessAtMixingZone')) }, { name: 'Intake (maximum)', values: outS.map((s) => gv(s, 'intakeExcessMax')) }] });
        plots.push({ type: 'bar', title: 'Seasonal simulation: near-field dilution', ylabel: 'Dilution (–)', categories: outS.map((s) => s.name), series: [{ name: 'Impact dilution', values: outS.map((s) => gv(s, 'impactDilution')) }, { name: 'Near-field dilution', values: outS.map((s) => gv(s, 'nearFieldDilution')) }] });
        xK.push({ label: 'Worst season at mixing-zone edge', value: `${worst.name}: ${fmt(gv(worst, 'excessAtMixingZone'), 3)} g/kg`, status: gv(worst, 'excessAtMixingZone') > gv(worst, 'limit') ? 'bad' : 'ok' });
        Object.assign(xO, { seasonalWorstExcess: gv(worst, 'excessAtMixingZone') });
        if (gv(worst, 'excessAtMixingZone') > gv(worst, 'limit')) W.push({ level: 'bad', msg: `Seasonal simulation: in ${worst.name} the excess salinity at the mixing-zone edge reaches ${fmt(gv(worst, 'excessAtMixingZone'), 3)} g/kg and exceeds the limit.` });
      }
    }
    if (pk) Object.assign(xO, { particlesInDomain: pk.inside / 100, particleDistance90: pk.d90 });
    for (const k of Object.keys(xO)) if (Number.isFinite(xO[k])) outputs[k] = xO[k];
    return {
      summary: `${P.n} port${P.n > 1 ? 's' : ''} of ${fmt(P.d * 1000, 3)} mm at ${fmt(P.U0, 3)} m/s (F = ${fmt(jet.F, 3)}): impact dilution ${fmt(jet.Si, 3)}, near-field dilution ${fmt(nf.Sn, 3)}; excess salinity ${fmt(mzEx, 2)} g/kg at the ${v.mzR} m mixing-zone edge (limit ${fmt(P.limit, 3)}), up to ${fmt(intakeMax, 2)} g/kg at the intake.`,
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
        { label: 'Maximum excess at intake', value: intakeMax, unit: 'g/kg', status: intakeMax > 0.02 * v.Sa ? 'bad' : intakeMax > 0.005 * v.Sa ? 'warn' : 'ok' }, { label: 'Mean excess at intake', value: intake.mean, unit: 'g/kg' },
        { label: 'Worst receptor exceedance', value: recStats.length ? 100 * Math.max(...recStats.map((r) => r.frac)) : 0, unit: '% of time', status: recStats.some((r) => r.frac > 0) ? 'warn' : 'ok' },
        { label: 'Antiscalant at mixing-zone edge', value: chem[0].mz, unit: 'mg/L', status: chem[0].mz > chem[0].lim ? 'warn' : 'ok' }, { label: 'Outfall length', value: outfallLength, unit: 'm', help: 'Distance from the shoreline to the diffuser plus the diffuser length' },
        ...(pk ? [{ label: 'Particles still inside the model domain', value: pk.inside, unit: '%', help: 'Lagrangian random-walk particles released continuously at the near-field end; the rest left through the open boundaries' }, { label: 'Particle distance from the source (median / 90 %)', value: `${fmt(pk.d50, 3)} / ${fmt(pk.d90, 3)}`, unit: 'm' }, { label: 'Mean age of the particles in the domain', value: pk.age, unit: 'h' }] : []),
        ...xK,
      ],
      warnings: W,
      recommendations: [
        mzEx > P.limit ? 'Increase the near-field dilution: more and smaller ports raise the Froude number; the diffuser design table lists compliant options.' : null,
        ztAbs > (v.clear / 100) * lowDepth ? 'Move the outfall to deeper water or reduce the port diameter so that the jet top stays below the surface at low tide.' : null,
        intakeMax > 0.005 * v.Sa ? 'Separate intake and outfall further, place the intake up-drift of the residual current or in shallower water away from the dense bottom layer.' : null,
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
    // ---- free-surface shallow-water solver
    {
      const flatB = (m, h0) => new Float64Array(m).fill(-h0), nf0 = { type: 'cd', Cd: 0 };
      const s1 = shallowWater({ nx: 50, ny: 3, dx: 200, dy: 200, zb: flatB(150, 10), eta0: (i) => 0.05 * Math.cos((Math.PI * (i + 0.5)) / 50), fric: nf0 }), Tm = (2 * 50 * 200) / Math.sqrt(G * 10), dt1 = s1.dtStable(), cr = [], v0 = s1.volume();
      for (let prev = s1.eta[50]; s1.t < 1.6 * Tm;) { s1.step(dt1); const e = s1.eta[50]; if (prev * e < 0) cr.push(s1.t - dt1 * (e / (e - prev))); prev = e; }
      add('Shallow water: seiche period equals Merian’s formula', 1, (cr[2] - cr[0]) / Tm, 0.005, 'T = 2L/√(gh) for the fundamental mode of a closed basin (ratio)');
      add('Shallow water: volume conservation (closed basin)', 0, s1.volume() / v0 - 1, 1e-12, 'Relative change of the stored volume after 1.6 seiche periods');
      const zl = new Float64Array(600); for (let q = 0; q < 600; q++) { const i = q % 30, k = (q - i) / 30; zl[q] = -5 + 6.5 * Math.exp(-((i - 15) ** 2 + (k - 10) ** 2) / 20) + 0.3 * Math.sin(i) + 0.05 * i; }
      const s2 = shallowWater({ nx: 30, ny: 20, dx: 100, dy: 100, zb: zl, eta0: 0 }); for (let k = 0; k < 200; k++) s2.step(s2.dtStable());
      add('Shallow water: lake at rest over an uneven bed with a dry island', 0, Math.max(...Array.from(s2.u, Math.abs), ...Array.from(s2.v, Math.abs)), 1e-12, 'Well-balanced scheme: no spurious current (m/s) with wet and dry cells');
      const zs = new Float64Array(240); for (let q = 0; q < 240; q++) zs[q] = -4 + 0.1 * (q % 60) + 0.2 * (((q - (q % 60)) / 60) % 2);
      const s3 = shallowWater({ nx: 60, ny: 4, dx: 50, dy: 50, zb: zs, eta0: (i) => 0.5 - 0.02 * i, fric: { type: 'manning', n: 0.02 } }), v3 = s3.volume(); let wmin = 240, wmax = 0;
      for (let k = 0; k < 1500; k++) { s3.step(s3.dtStable()); if (k % 10 === 0) { let wn = 0; for (const hh of s3.h) if (hh > 0.05) wn++; wmin = Math.min(wmin, wn); wmax = Math.max(wmax, wn); } }
      add('Shallow water: volume conservation with wetting and drying', 0, s3.volume() / v3 - 1, 1e-10, `Run-up and run-down on a beach with Manning friction; between ${wmin} and ${wmax} cells wet`);
      const tau = 1e-4, Ts = (2 * 20 * 250) / Math.sqrt(G * 5), s4 = shallowWater({ nx: 20, ny: 3, dx: 250, dy: 250, zb: flatB(60, 5), eta0: 0, fric: { type: 'cd', Cd: 0.003 }, tau: (t) => [tau * Math.min(1, t / (4 * Ts)), 0] }), dt4 = 0.9 * s4.dtStable(); let sl = 0, m4 = 0;
      while (s4.t < 8 * Ts) { s4.step(dt4); if (s4.t > 7 * Ts) { sl += (s4.eta[39] - s4.eta[20]) / (19 * 250); m4++; } }
      add('Shallow water: wind set-up balances the surface stress', 1, sl / m4 / (tau / (G * 5)), 0.01, '∂η/∂x = τ_wind/(ρ g h) in a closed basin (ratio)');
      const s5 = shallowWater({ nx: 120, ny: 3, dx: 100, dy: 100, zb: flatB(360, 10), eta0: (i) => 0.1 * Math.exp(-((i - 60) ** 2) / 50), fric: nf0, bc: { W: 'rad', E: 'rad' } }), dt5 = s5.dtStable();
      while (s5.t < (1.6 * 6000) / Math.sqrt(G * 10)) s5.step(dt5);
      add('Shallow water: radiation boundary lets a wave leave', 0, Math.max(...Array.from(s5.eta, Math.abs)) / 0.1, 0.02, 'Residual elevation ÷ initial hump after both pulses crossed the open ends');
      const T6 = 3600, om6 = (2 * Math.PI) / T6, s6 = shallowWater({ nx: 25, ny: 3, dx: 200, dy: 200, zb: flatB(75, 10), eta0: 0, fric: nf0, bc: { W: 'elev' }, ext: (t) => ({ e: 0.01 * Math.min(1, t / (5 * T6)) * Math.sin(om6 * t), gx: 0, gy: 0, U: 0, V: 0 }) }), dt6 = s6.dtStable(); let hi6 = -1, lo6 = 1;
      while (s6.t < 8 * T6) { s6.step(dt6); if (s6.t > 7 * T6) { hi6 = Math.max(hi6, s6.eta[49]); lo6 = Math.min(lo6, s6.eta[49]); } }
      add('Shallow water: tidal-elevation boundary, co-oscillating tide', 1 / Math.cos((om6 * (5000 - 100)) / Math.sqrt(G * 10)), (hi6 - lo6) / 0.02, 0.02, 'Amplification 1/cos(ωL/√(gh)) at the closed end of a frictionless channel');
      const U7 = 0.4, f7 = 1e-4, r7 = (0.0025 * U7) / 10, s7 = shallowWater({ nx: 24, ny: 8, dx: 250, dy: 250, zb: flatB(192, 10), eta0: 0, f: f7, fric: { type: 'cd', Cd: 0.0025 }, bc: { W: 'flather', E: 'flather' }, ext: (t) => { const m = Math.min(1, t / 3000); return { e: 0, gx: (-r7 * U7 * m) / G, gy: (-f7 * U7 * m) / G, U: U7 * m, V: 0 }; } }), dt7 = s7.dtStable();
      while (s7.t < 30000) s7.step(dt7);
      add('Shallow water: Flather boundaries carry the outer current', 1, s7.cellU(4 * 24 + 12)[0] / U7, 0.01, 'Interior velocity ÷ external current for a steady friction-balanced stream (ratio)');
      add('Shallow water: geostrophic cross-stream surface slope', 1, (s7.eta[6 * 24 + 12] - s7.eta[24 + 12]) / (5 * 250) / ((-f7 * U7) / G), 0.01, '∂η/∂y = −f U/g with Coriolis (ratio)');
      // transport on the moving free surface: a uniform concentration must stay uniform while the basin sloshes
      const gq = flatGrid(30, 6, 100, 8, false), s8 = shallowWater({ nx: 30, ny: 6, dx: 100, dy: 100, zb: gq.zb, eta0: (i) => 0.3 * Math.cos((Math.PI * (i + 0.5)) / 30), fric: nf0 });
      const r8 = await farField({ g: gq, flow: null, cur: still, K: new Float64Array(180).fill(1), tEnd: 900, C0: new Float64Array(180).fill(1), closed: true, dyn: swCoupler(s8, 30, 6, 1) }); let dev8 = 0; for (let q = 0; q < 180; q++) dev8 = Math.max(dev8, Math.abs(r8.C[q] - 1));
      const s9 = shallowWater({ nx: 15, ny: 3, dx: 200, dy: 200, zb: new Float64Array(45).fill(-8), eta0: (i) => 0.3 * Math.cos((Math.PI * (i + 0.5)) / 15), fric: nf0 }), r9 = await farField({ g: gq, flow: null, cur: still, K: new Float64Array(180).fill(1), tEnd: 900, C0: new Float64Array(180).fill(1), closed: true, dyn: swCoupler(s9, 30, 6, 2) }); let dev9 = 0; for (let q = 0; q < 180; q++) dev9 = Math.max(dev9, Math.abs(r9.C[q] - 1));
      add('Coarse hydrodynamic grid: interpolated transports stay consistent', 0, dev9, 1e-9, 'Solver cells twice the transport cells: max |C − 1| for a uniform concentration during the seiche');
      add('Shallow-water-driven transport preserves a uniform concentration', 0, dev8, 1e-9, 'Free-surface (η-consistent) far-field transport during a 0.3 m seiche: max |C − 1|');
      add('Shallow-water-driven transport conserves mass', 1, r8.bal.mass / (180 * 8 * 1e4), 1e-9, 'Σ D·C·A after the run ÷ initial content');
      add('Wind stress: Smith–Banke drag law', 1.22 * 1.29e-3 * 100, windStress(10, 270, 1025)[0] * 1025, 1e-6, 'τ = ρ_air C_d W² with C_d = (0.63 + 0.066 W)·10⁻³ at W = 10 m/s (N/m²)');
    }
    // ---- semi-implicit free-surface scheme and Lagrangian particles
    {
      const flatB = (m, h0) => new Float64Array(m).fill(-h0), nf0 = { type: 'cd', Cd: 0 };
      const i1 = shallowWater({ implicit: true, theta: 0.5, nx: 50, ny: 3, dx: 200, dy: 200, zb: flatB(150, 10), eta0: (i) => 0.05 * Math.cos((Math.PI * (i + 0.5)) / 50), fric: nf0 }), Tm = (2 * 50 * 200) / Math.sqrt(G * 10), dti = (4 * 200) / Math.sqrt(G * 10), cr = [], vi = i1.volume();
      for (let prev = i1.eta[50]; i1.t < 1.7 * Tm;) { i1.step(dti); const e = i1.eta[50]; if (prev * e < 0) cr.push(i1.t - dti * (e / (e - prev))); prev = e; }
      add('Semi-implicit shallow water: seiche period at a gravity-wave Courant number of 4', 1, (cr[2] - cr[0]) / Tm, 0.012, 'T = 2L/√(gh) with θ = ½; the step is four times the explicit limit (ratio, second-order phase error ≈ 0.5 %)');
      add('Semi-implicit shallow water: volume conservation', 0, i1.volume() / vi - 1, 1e-12, 'Relative change of the stored volume in the closed basin');
      const zs = new Float64Array(240); for (let q = 0; q < 240; q++) zs[q] = -4 + 0.1 * (q % 60) + 0.2 * (((q - (q % 60)) / 60) % 2);
      const i3 = shallowWater({ implicit: true, nx: 60, ny: 4, dx: 50, dy: 50, zb: zs, eta0: (i) => 0.5 - 0.02 * i, fric: { type: 'manning', n: 0.02 } }), v3 = i3.volume(); let hneg = 0, wmin = 240, wmax = 0;
      for (let k = 0; k < 300; k++) { i3.step(30); let wn = 0; for (const hh of i3.h) { if (hh < hneg) hneg = hh; if (hh > 0.05) wn++; } wmin = Math.min(wmin, wn); wmax = Math.max(wmax, wn); }
      add('Semi-implicit shallow water: volume conservation with wetting and drying', 0, i3.volume() / v3 - 1, 1e-10, `30 s steps (gravity-wave Courant number ≈ 4) on a beach with Manning friction; between ${wmin} and ${wmax} cells wet`);
      add('Semi-implicit shallow water: depths stay non-negative', 0, hneg, 0, 'Outgoing fluxes are limited to the water a cell holds');
      const zl = new Float64Array(600); for (let q = 0; q < 600; q++) { const i = q % 30, k = (q - i) / 30; zl[q] = -5 + 6.5 * Math.exp(-((i - 15) ** 2 + (k - 10) ** 2) / 20) + 0.3 * Math.sin(i) + 0.05 * i; }
      const i2 = shallowWater({ implicit: true, nx: 30, ny: 20, dx: 100, dy: 100, zb: zl, eta0: 0 }); for (let k = 0; k < 50; k++) i2.step(60);
      add('Semi-implicit shallow water: lake at rest over an uneven bed with a dry island', 0, Math.max(...Array.from(i2.u, Math.abs), ...Array.from(i2.v, Math.abs)), 1e-12, 'No spurious current (m/s)');
      const tau = 1e-4, Ts = (2 * 20 * 250) / Math.sqrt(G * 5), i4 = shallowWater({ implicit: true, nx: 20, ny: 3, dx: 250, dy: 250, zb: flatB(60, 5), eta0: 0, fric: { type: 'cd', Cd: 0.003 }, tau: (t) => [tau * Math.min(1, t / (4 * Ts)), 0] }); let sl = 0, m4 = 0;
      while (i4.t < 10 * Ts) { i4.step(150); if (i4.t > 8 * Ts) { sl += (i4.eta[39] - i4.eta[20]) / (19 * 250); m4++; } }
      add('Semi-implicit shallow water: wind set-up balances the surface stress', 1, sl / m4 / (tau / (G * 5)), 0.01, '∂η/∂x = τ_wind/(ρ g h), 150 s steps (ratio)');
      const U7 = 0.4, f7 = 1e-4, r7 = (0.0025 * U7) / 10, i7 = shallowWater({ implicit: true, nx: 24, ny: 8, dx: 250, dy: 250, zb: flatB(192, 10), eta0: 0, f: f7, fric: { type: 'cd', Cd: 0.0025 }, bc: { W: 'flather', E: 'flather' }, ext: (t) => { const m = Math.min(1, t / 3000); return { e: 0, gx: (-r7 * U7 * m) / G, gy: (-f7 * U7 * m) / G, U: U7 * m, V: 0 }; } });
      while (i7.t < 40000) i7.step(200);
      add('Semi-implicit shallow water: Flather boundaries carry the outer current', 1, i7.cellU(4 * 24 + 12)[0] / U7, 0.01, 'Interior velocity ÷ external current, 200 s steps (ratio)');
      add('Semi-implicit shallow water: geostrophic cross-stream surface slope', 1, (i7.eta[6 * 24 + 12] - i7.eta[24 + 12]) / (5 * 250) / ((-f7 * U7) / G), 0.015, '∂η/∂y = −f U/g (ratio)');
      const T6 = 3600, om6 = (2 * Math.PI) / T6, i6 = shallowWater({ implicit: true, theta: 0.5, nx: 25, ny: 3, dx: 200, dy: 200, zb: flatB(75, 10), eta0: 0, fric: nf0, bc: { W: 'elev' }, ext: (t) => ({ e: 0.01 * Math.min(1, t / (5 * T6)) * Math.sin(om6 * t), gx: 0, gy: 0, U: 0, V: 0 }) }); let hi6 = -1, lo6 = 1;
      while (i6.t < 8 * T6) { i6.step(40); if (i6.t > 7 * T6) { hi6 = Math.max(hi6, i6.eta[49]); lo6 = Math.min(lo6, i6.eta[49]); } }
      add('Semi-implicit shallow water: clamped tidal elevation, co-oscillating tide', 1 / Math.cos((om6 * (5000 - 100)) / Math.sqrt(G * 10)), (hi6 - lo6) / 0.02, 0.03, 'Amplification 1/cos(ωL/√(gh)) at the closed end, 40 s steps (Courant number 2)');
      const gq = flatGrid(30, 6, 100, 8, false), i8 = shallowWater({ implicit: true, nx: 30, ny: 6, dx: 100, dy: 100, zb: gq.zb, eta0: (i) => 0.3 * Math.cos((Math.PI * (i + 0.5)) / 30), fric: nf0 });
      const r8 = await farField({ g: gq, flow: null, cur: still, K: new Float64Array(180).fill(1), tEnd: 900, C0: new Float64Array(180).fill(1), closed: true, dyn: swCoupler(i8, 30, 6, 1) }); let dev8 = 0; for (let q = 0; q < 180; q++) dev8 = Math.max(dev8, Math.abs(r8.C[q] - 1));
      add('Transport on the semi-implicit free surface preserves a uniform concentration', 0, dev8, 1e-9, 'Fluxes and depth change of the transport are those of the hydrodynamic step: max |C − 1| during a 0.3 m seiche');
      // random-walk particles in a uniform current: mean drift and spreading
      const gp = flatGrid(120, 80, 50, 10, false), fp = flowBasis(gp), Kp = 2, up = 0.1, tp = 6 * 3600;
      const rp = await farField({ g: gp, flow: fp, cur: { ...still, res: [up, 0] }, K: new Float64Array(9600).fill(Kp), tEnd: tp, particles: 4000, srcXY: [-2000, 0], srcR: 1e-6 }), pa = rp.part, ma = mean(pa.age), mxp = mean(pa.x) + 2000, vy = mean(pa.y.map((y) => y * y));
      add('Particle tracking: mean drift equals current × age', 1, mxp / (up * ma), 0.02, `Random-walk particles released continuously in a uniform current of 0.1 m/s; ${pa.x.length} in the domain (ratio)`);
      add('Particle tracking: cross-current spreading σ² = 2K·age', 1, vy / (2 * Kp * ma), 0.06, 'Variance of the cross-stream position ÷ 2K × mean age for K = 2 m²/s (ratio; 4000 particles)');
    }
    // ---- wave-action balance
    {
      const wn = 30, wm = 50, sl = 0.02, hw = new Float64Array(wn * wm); for (let q = 0; q < wn * wm; q++) hw[q] = Math.max(0, sl * ((((q - (q % wn)) / wn) + 0.5) * 5 - 10));
      const a0 = 20 * D2R, wv = waveField({ nx: wn, ny: wm, dx: 10, dy: 5, h: hw, H0: 1, T: 10, theta0: -Math.PI / 2 - a0, Cf: 0.01 }), ref = waveNumber((2 * Math.PI) / 10, wv.href), Pa = 30 * wn + 9, anA = -Math.PI / 2 - wv.theta[Pa];
      add('Waves: Snell’s law of refraction on a plane beach', Math.sin(a0) / ref.c, Math.sin(anA) / wv.c[Pa], (0.01 * Math.sin(a0)) / ref.c, `sin θ / c is invariant: the wave turns from 20° to ${fmt(anA / D2R, 3)}° at ${fmt(hw[Pa], 3)} m depth`);
      add('Waves: shoaling and refraction coefficient', Math.sqrt((ref.cg * Math.cos(a0)) / (wv.cg[Pa] * Math.cos(anA))), wv.Hs[Pa] / 1, 0.01, 'H/H₀ = √(c_g0 cos θ₀ / (c_g cos θ)) from conservation of wave action');
      const Pb = 9 * wn + 9, anB = -Math.PI / 2 - wv.theta[Pb];
      add('Waves: depth-limited breaking H = γ h in the surf zone', 0.78 * hw[Pb], wv.Hs[Pb], 1e-6, 'Saturated surf zone');
      add('Waves: longshore current of Longuet-Higgins', ((5 * Math.PI) / 16) * (0.78 / 0.01) * G * hw[Pb] * (Math.sin(anB) / wv.c[Pb]) * sl, Math.abs(wv.Vls[Pb]), 0.08 * ((5 * Math.PI) / 16) * (0.78 / 0.01) * G * hw[Pb] * (Math.sin(anB) / wv.c[Pb]) * sl, 'V = (5π/16)(γ/C_f) g h (sin θ/c) tan β from the radiation-stress gradient, within 8 %');
      const hd = new Float64Array(240).fill(50), Uo = Float64Array.from({ length: 240 }, (_, q) => -0.5 * (1 + Math.tanh(((q % 80) - 40) / 6))), wc = waveField({ nx: 80, ny: 3, dx: 20, dy: 20, h: hd, H0: 1, T: 6, theta0: 0, U: Uo, V: new Float64Array(240), iters: 420 }), c0 = wc.c[82], c1 = wc.c[80 + 75];
      add('Wave–current interaction: Doppler-shifted phase speed on an opposing current', 0.5 * (1 + Math.sqrt(1 - 4 / c0)), c1 / c0, 1e-3, 'c/c₀ = ½[1 + √(1 + 4U/c₀)], U = −1 m/s, deep water');
      add('Wave–current interaction: wave-action conservation (steepening)', c0 / Math.sqrt(c1 * (c1 - 2)), wc.Hs[80 + 75] / wc.Hs[82], 5e-3, 'H/H₀ = c₀/√(c(c + 2U)) (Longuet-Higgins & Stewart 1961)');
    }
    // ---- vertical slice: lock exchange, turbulence closures
    {
      const ln = 96, lz = 20, Hl = 4, Ll = 32, bl = 0.1 / G, sl0 = Float64Array.from({ length: ln * lz }, (_, q) => (q % ln < ln / 2 ? 1 : 0)), Ub = 0.5 * Math.sqrt(0.1 * Hl);
      for (const model of ['nonhydro', 'hydro']) {
        const r = await verticalSlice({ nx: ln, nz: lz, dx: Ll / ln, dz: Hl / lz, s0: sl0, beta: bl, tEnd: 26, model, turb: 'const', nu: 1e-4, nuH: 1e-4, front: { row: 0, i0: 0, dir: 1, level: 0.5 } });
        add(`Vertical slice (${model === 'hydro' ? 'hydrostatic' : 'non-hydrostatic'}): lock-exchange front speed`, Ub, frontSpeed(r.hist, 0.3, 1), 0.12 * Ub, 'Benjamin (1968): u_f = ½√(g′H) for a full-depth lock exchange, within 12 %');
        if (model === 'nonhydro') { add('Vertical slice: salt conservation', 0, r.salt / r.salt0 - 1, 1e-10, 'Relative change of the total density excess'); add('Vertical slice: projected velocity is divergence-free', 0, r.divMax / (Ub / (Hl / lz)), 1e-4, 'max |∇·u| ÷ (u_f/Δz) after the pressure projection'); }
      }
      const kd = await verticalSlice({ nx: 6, nz: 6, dx: 1, dz: 1, s0: new Float64Array(36), tEnd: 50, turb: 'ke', k0: 1e-3, eps0: 1e-4, dtMax: 0.25, nuMax: 1e-9 });
      add('k–ε closure: decay of homogeneous turbulence', 1e-3 * (1 + (0.92 * 1e-4 * 50) / 1e-3) ** (-1 / 0.92), kd.k[14], 0.02 * 1e-3, 'k(t) = k₀[1 + (c₂ − 1) ε₀ t / k₀]^(−1/(c₂−1)) with c₂ = 1.92');
      add('Pacanowski–Philander mixing at Ri = 0.2', 1e-2 / 4 + 1e-4, ppMixing(0.2).nu, 1e-12, 'ν = ν₀/(1 + 5Ri)² + ν_b');
    }
    // ---- three-dimensional hydrostatic model
    {
      const flat3 = (m, h0) => new Float64Array(m).fill(-h0);
      { // full-depth lock exchange in a closed flat channel, at two vertical resolutions
        const lx = 80, Hl = 10, Ll = 4000, dxl = Ll / lx, drho = 1, Ub = 0.5 * Math.sqrt(((G * drho) / 1000) * Hl), got = {};
        for (const lz of [12, 24]) {
          const M = hydro3D({ nx: lx, ny: 3, nz: lz, dx: dxl, dy: dxl, zb: flat3(3 * lx, Hl), slip: true, turb: 'const', nu: 1e-4, Kv: 0, rho0: 1000, dens: (tr, x) => drho * tr[0][x], tracers: [{ c0: (P) => (P % lx < lx / 2 ? 1 : 0) }] }), m0 = M.mass(0), ts = [], xf = [], tE = (0.35 * Ll) / Ub;
          while (M.t < tE) { M.step(Math.min(M.dtStable(), 20)); let x = 0; for (let i = lx - 1; i >= 0; i--) { const a = M.tr[0][lx + i]; if (a > 0.5) { const b = i < lx - 1 ? M.tr[0][lx + i + 1] : 0; x = (i + 0.5) * dxl + (dxl * (a - 0.5)) / Math.max(a - b, 1e-12); break; } } ts.push(M.t); xf.push(x); }
          let st = 0, sx = 0, stt = 0, stx = 0, m = 0, cmin = 0, cmax = 1; for (let k = Math.floor(0.3 * ts.length); k < ts.length; k++) { st += ts[k]; sx += xf[k]; stt += ts[k] * ts[k]; stx += ts[k] * xf[k]; m++; }
          for (const c of M.tr[0]) { if (c < cmin) cmin = c; if (c > cmax) cmax = c; }
          got[lz] = (m * stx - st * sx) / (m * stt - st * st);
          if (lz === 12) {
            add('3-D hydrostatic model: tracer mass conserved in the lock exchange', 0, M.mass(0) / m0 - 1, 1e-11, 'Σ c·V over all layers, relative change — flux-form transport on the moving σ-layers');
            add('3-D hydrostatic model: limited transport creates no new extrema', 0, Math.max(-cmin, cmax - 1), 1e-9, 'Overshoot of the brine fraction beyond its initial range 0…1 (van Leer limiter)');
          }
        }
        const refL = 'Reference: Benjamin (1968), u_f = ½√(g′H), the energy-conserving value for a full-depth lock exchange (confirmed by Shin, Dalziel & Linden 2004); currents that mix at the head run at 0.44–0.48 √(g′H)';
        add('3-D hydrostatic model: lock-exchange front speed, 80 × 3 × 24 cells', Ub, got[24], 0.08 * Ub, `${refL}. Dense front along the free-slip bed, least-squares slope over the last 70 % of the run (front travel 124 H), 24 σ-layers: ${fmt(got[24] / Ub, 3)} of the reference = ${fmt(got[24] / Math.sqrt(((G * drho) / 1000) * Hl), 3)} √(g′H); tolerance 8 %`);
        add('3-D hydrostatic model: lock-exchange front speed, 80 × 3 × 12 cells', Ub, got[12], 0.12 * Ub, `Same case with 12 σ-layers: ${fmt(got[12] / Ub, 3)} of the reference = ${fmt(got[12] / Math.sqrt(((G * drho) / 1000) * Hl), 3)} √(g′H); tolerance 12 %. The deficit is set by the vertical resolution of the head, where the dense water that overruns the front is lifted and mixed over one layer: 0.795, 0.886, 0.926 and 0.939 of the reference with 6, 12, 24 and 48 layers, i.e. the model converges to about 0.47 √(g′H), inside the range of currents with a mixing head and 5–6 % below the energy-conserving value. Horizontal refinement (40 to 320 cells), the time step, second- instead of first-order momentum advection and the viscosity floor change the value by less than 1 %`);
        add('3-D hydrostatic model: the front speed converges upward with the number of layers', 1, got[24] > got[12] && got[24] < Ub ? 1 : 0, 0, `${fmt(got[12] / Ub, 4)} → ${fmt(got[24] / Ub, 4)} of ½√(g′H) from 12 to 24 layers`);
      }
      { // near-field → far-field coupling: volume, salt and momentum of the near-field water in a closed basin at rest
        const ix = 31, iy = 21, iz = 8, n2 = ix * iy, Pc = 10 * ix + 15, Qe = 1, Sd = 20, tI = 3600;
        const runI = (mode) => {
          const cols = [{ P: Pc, w: 1, src: [{ k: 0, f: 1 }], sink: [{ k: 1, f: 0.5 }, { k: 2, f: 0.5 }] }];
          const M = hydro3D({ nx: ix, ny: iy, nz: iz, dx: 50, dy: 50, zb: flat3(n2, 10), slip: true, turb: 'const', nu: 1e-3, Kv: 1e-5, rho0: 1000, dens: (tr, x) => 25 * tr[0][x], tracers: [{ c0: 0, src: mode === 'tracer' ? [{ P: Pc, k: 0, rate: Qe }] : [] }, { c0: 1 }], inflow: mode === 'tracer' ? null : { Q: Qe, S: Sd, u: mode === 'mom' ? 0.4 : 0, v: 0, eff: [1, 1], cols } }), v0 = M.sw.volume();
          while (M.t < tI) M.step(Math.min(M.dtStable(), 30, tI - M.t));
          let dev = 0, sx = 0, sm = 0; for (const c of M.tr[1]) dev = Math.max(dev, Math.abs(c - 1));
          for (let k = 0; k < iz; k++) for (let P = 0; P < n2; P++) { const w = M.tr[0][k * n2 + P] * M.ds[k]; sm += w; sx += w * ((P % ix) - 15) * 50; }
          return { M, dV: M.sw.volume() - v0, dev, xc: sx / sm, cs: M.tr[0][Pc] * Sd, up: M.tr[0][Pc + 5], dn: M.tr[0][Pc - 5] };
        };
        const a = runI('vol'), b = runI('tracer'), c = runI('mom');
        add('3-D model, near-field coupling: brine added equals the effluent exactly', 0, (a.M.mass(0) - a.M.meta[0].injected) / a.M.meta[0].injected, 1e-12, `Near-field water S·Q = ${Sd} m³/s enters the bottom layer and the entrained (S − 1)·Q leaves layers 2–3 at the local concentration; stored brine ÷ Q·t − 1 after one hour (with momentum: ${((c.M.mass(0) - c.M.meta[0].injected) / c.M.meta[0].injected).toExponential(1)})`);
        add('3-D model, near-field coupling: water volume grows by the effluent volume', Qe * tI, a.dV, 1e-6, 'Net volume source = S·Q − (S − 1)·Q = Q in the free-surface and layer continuity equations (m³ after one hour, closed basin)');
        add('3-D model, near-field coupling: a uniform tracer stays uniform', 0, Math.max(a.dev, c.dev), 1e-12, 'Tracer equal to 1 in the sea and in the effluent: sources, sinks, layer fluxes and the moving surface are mutually consistent (max |c − 1|)');
        add('3-D model, near-field coupling: dilution in the source cell equals the near-field dilution', 1, a.cs, 0.02, `Brine fraction of the bottom-layer source cell × S after one hour: the cell holds near-field water, as handed over by the jet model. The salt-only source gives ${fmt(b.cs, 3)} in the same cell — more concentrated than the water the near field delivers, because nothing displaces the brine it adds`);
        add('3-D model, near-field coupling: jet momentum carries the plume in the discharge direction', 1, c.xc > 100 && Math.abs(a.xc) < 1e-6 && c.up > 50 * Math.max(c.dn, 1e-12) ? 1 : 0, 0, `Centroid of the brine after one hour: ${fmt(c.xc, 3)} m downstream with an inflow velocity of 0.4 m/s, ${fmt(Math.abs(a.xc), 2)} m without (symmetric spreading); brine fraction 250 m ahead ${fmt(c.up * Sd, 3)}/S against ${fmt(c.dn * Sd, 3)}/S behind`);
      }
      { // barotropic seiche with a passive tracer
        const sx2 = 40, sz = 6, dxs = 250, Hs = 10, Tm = (2 * sx2 * dxs) / Math.sqrt(G * Hs), dts = (2 * dxs) / Math.sqrt(G * Hs), cr = [];
        const M = hydro3D({ nx: sx2, ny: 3, nz: sz, dx: dxs, dy: dxs, zb: flat3(3 * sx2, Hs), slip: true, turb: 'const', nu: 1e-5, tracers: [{ c0: 1 }], sw: { eta0: (i) => 0.05 * Math.cos((Math.PI * (i + 0.5)) / sx2), theta: 0.5 } }), v0 = M.sw.volume();
        for (let prev = M.sw.eta[sx2]; M.t < 1.7 * Tm;) { M.step(dts); const e = M.sw.eta[sx2]; if (prev * e < 0) cr.push(M.t - dts * (e / (e - prev))); prev = e; }
        let dev = 0, du = 0; for (const c of M.tr[0]) dev = Math.max(dev, Math.abs(c - 1)); const NUs = (sx2 + 1) * 3; for (let q = 0; q < NUs; q++) du = Math.max(du, Math.abs(M.u[(sz - 1) * NUs + q] - M.u[q]));
        add('3-D hydrostatic model: seiche period of the free surface', 1, (cr[2] - cr[0]) / Tm, 0.012, 'Merian: T = 2L/√(gH) in a closed basin, 6 σ-layers, step twice the explicit gravity-wave limit (ratio)');
        add('3-D hydrostatic model: a uniform tracer stays uniform under the moving surface', 0, dev, 1e-10, 'Layer volume fluxes and layer thickness change are consistent with the free surface: max |c − 1| during the seiche');
        add('3-D hydrostatic model: water volume conserved', 0, M.sw.volume() / v0 - 1, 1e-12, 'Relative change in the closed basin');
        add('3-D hydrostatic model: frictionless barotropic flow is depth-uniform', 0, du / Math.max(M.uMax, 1e-12), 1e-9, 'Surface-layer minus bottom-layer velocity ÷ largest velocity: without density differences and bed stress the layers move together');
      }
      { // wind-driven circulation in a closed basin: constant eddy viscosity, free-slip bed
        const wx = 20, wz = 12, Hw = 5, nuw = 0.01, tauw = 1e-4, M = hydro3D({ nx: wx, ny: 3, nz: wz, dx: 250, dy: 250, zb: flat3(3 * wx, Hw), slip: true, turb: 'const', nu: nuw, sw: { tau: (t) => [tauw * Math.min(1, t / 3000), 0] } });
        while (M.t < 30000) M.step(100);
        const NUw = (wx + 1) * 3, qw = wx + 1 + 10, ex = (s) => (tauw / (nuw * Hw)) * ((s * Hw) ** 2 / 2 - (Hw * Hw) / 6);
        add('3-D hydrostatic model: wind set-up of the free surface', 1, (M.sw.eta[wx + 15] - M.sw.eta[wx + 5]) / (10 * 250) / (tauw / (G * Hw)), 0.01, '∂η/∂x = τ_wind/(ρ g H) for zero bed stress (ratio)');
        add('3-D hydrostatic model: wind-driven surface current', ex(1 - 0.5 / wz), M.u[(wz - 1) * NUw + qw], 0.01 * ex(1 - 0.5 / wz), 'Steady closed-basin circulation with constant eddy viscosity: u(z) = (τ/ρνH)(z²/2 − H²/6), downwind at the surface (m/s)');
        add('3-D hydrostatic model: return current at the bed', ex(0.5 / wz), M.u[qw], 0.01 * Math.abs(ex(0.5 / wz)), 'Same parabolic profile: the flow reverses in the lower part of the water column so that the depth-integrated transport vanishes (m/s)');
      }
      { // stratified water at rest over a sloping, uneven bed
        const rx = 30, ry = 20, zr = new Float64Array(rx * ry); for (let q = 0; q < rx * ry; q++) { const i = q % rx, j = (q - i) / rx; zr[q] = -4 - 0.3 * i - 6 * Math.exp(-((i - 15) ** 2 + (j - 10) ** 2) / 20); }
        const M = hydro3D({ nx: rx, ny: ry, nz: 8, dx: 100, dy: 100, zb: zr, turb: 'const', nu: 1e-4, Kv: 0, dens: (tr, x) => 2 * tr[0][x], tracers: [{ c0: (P, k, z) => -z / 20 }] });
        for (let k = 0; k < 60; k++) M.step(60);
        add('3-D hydrostatic model: linearly stratified water stays at rest over a slope', 0, Math.max(M.uMax, M.vMax) / Math.sqrt(((G * 2) / 1025) * 20), 1e-10, 'Largest velocity after one hour ÷ internal wave speed: the density-Jacobian pressure gradient on σ-layers has no error for a density linear in z (depths 4–19 m, bed steps up to 1.6 layer thicknesses)');
      }
      { // the option through run(): three-dimensional views and the brine balance
        const dv3 = Object.fromEntries(suite.inputs.flatMap((q) => q.fields).map((q) => [q.key, q.type === 'table' ? JSON.parse(JSON.stringify(q.value)) : q.value])), r3 = await suite.run({ ...dv3, h3d: true, nx: 30, ny: 20, nCycles: 1, h3nx: 24, h3ny: 18, h3nz: 6 });
        const p3 = r3.plots.filter((q) => /^3-D model/.test(q.title)), f3 = p3.filter((q) => q.type === 'field'), ob = p3.find((q) => q.type === 'surface3d'), b3 = r3.balances.find((q) => /3-D model: brine/.test(q.name)), sec = f3.filter((q) => /vertical section/.test(q.title)), plan = f3.find((q) => /near-bed layer \(t/.test(q.title));
        const dimOK = (q, f) => (f.z || q.z).length === (f.y || q.y).length && (f.z || q.z)[0].length === q.x.length, frOK = (q) => Array.isArray(q.frames) && q.frames.length > 1 && q.frames.every((f) => f.label && dimOK(q, f));
        add('3-D plume views through run(): plan map with a layer slider, sections with a position slider', 1, f3.length >= 7 && plan && frOK(plan) && plan.frames.length === 6 && sec.length === 2 && sec.every(frOK) && f3.every((q) => dimOK(q, q)) ? 1 : 0, 0, `${f3.length} field views of the three-dimensional solution; the plan map carries ${plan?.frames?.length} frames (one per σ-layer, bed to surface) and the two vertical sections ${sec.map((q) => q.frames?.length).join(' and ')} positions, all with the grid dimensions of their chart`);
        { // shaded 3-D view: the iso-surface stands on the bed, never below it, and encloses the reported volume
          const L0 = ob?.layers?.[0], bedOK = !!L0 && L0.z.every((row, jq) => row.every((zq, iq) => zq >= ob.z[jq][iq] - 1e-9 && zq <= 3)), dA3 = L0 ? (ob.x[1] - ob.x[0]) * (ob.y[1] - ob.y[0]) : 0;
          let vol = 0; if (L0) L0.z.forEach((row, jq) => row.forEach((zq, iq) => { vol += (zq - ob.z[jq][iq]) * dA3; }));
          const kv = r3.kpis.find((q) => /^Plume volume above/.test(q.label));
          add('3-D plume views through run(): shaded iso-surface view with selectable level', 1, ob && bedOK && ob.frames.length >= 2 && ob.frames.every((f) => f.layers[0].z.length === ob.y.length) && ob.planes.length === 1 && ob.markers.length === 1 ? 1 : 0, 0, `Surface plot of the seabed with the plume iso-surface as filled polygons (${ob ? ob.frames.length : 0} iso-levels on the slider); the iso-surface lies between the bed and the water surface in every column`);
          add('3-D plume views through run(): volume under the drawn iso-surface', 1, kv ? vol / (1e3 * kv.value) : NaN, 0.35, `Volume between the plotted surface and the bed ÷ volume of the layer cells above the threshold (${kv ? fmt(kv.value, 3) : '–'} thousand m³): the surface is interpolated between the layer centres, the reported volume counts whole layer cells, so they agree to within a layer thickness`);
        }
        add('3-D model through run(): brine volume balance', 0, b3 ? (b3.out - b3.in) / b3.in : 1, 1e-9, '(stored + exported − injected − inflow) ÷ injected for the brine tracer over one tidal cycle');
        { // near-field coupling and compliance verdict through run()
          const rt = await suite.run({ ...dv3, h3d: true, h3Src: 'tracer', nx: 30, ny: 20, nCycles: 1, h3nx: 24, h3ny: 18, h3nz: 6 }), rl = await suite.run({ ...dv3, h3d: true, hLayer: 8, nx: 30, ny: 20, nCycles: 1, h3nx: 24, h3ny: 18, h3nz: 6 }), r0 = await suite.run({ ...dv3, nx: 30, ny: 20, nCycles: 1 });
          const o3 = r3.outputs, ot = rt.outputs, ol = rl.outputs, o0 = r0.outputs, row = (r, re) => r.tables.find((t) => t.title === 'Environmental compliance').rows.find((q) => re.test(q[0])), wv = b3 && r3.balances.find((q) => /3-D model: water volume/.test(q.name));
          add('3-D model through run(): far from the source the coupling does not matter', 1, o3.intakeExcessMax3D / ot.intakeExcessMax3D, 0.03, `Near-bed tidal maximum at the intake (${fmt(Math.hypot(dv3.inX, dv3.inY), 3)} m from the outfall), volume + salt + momentum hand-off ÷ salt-only source: ${fmt(o3.intakeExcessMax3D, 3)} against ${fmt(ot.intakeExcessMax3D, 3)} g/kg. At the ${dv3.mzR} m mixing-zone edge the hand-off gives ${fmt(o3.mz3D, 3)} against ${fmt(ot.mz3D, 3)} g/kg (ratio ${fmt(o3.mz3D / ot.mz3D, 3)}): near the source the volume and momentum of the near-field water matter`);
          add('3-D model through run(): water volume balance with the effluent source', 0, wv ? (wv.out - wv.in) / wv.in : 1, 1e-12, '(final − initial − boundary inflow − effluent) ÷ initial over one tidal cycle');
          add('Compliance verdict with the 3-D model: mixing-zone value is the larger of the models', Math.max(row(r3, /^… near-field/)[1], o3.mzFarField, o3.mz3D), o3.excessAtMixingZone, 0, 'excessAtMixingZone = max(near-field estimate, layer model, 3-D near-bed layer), exactly');
          add('Compliance verdict with the 3-D model: the 3-D model governs where it is the more conservative', ol.intakeExcessMax3D, ol.intakeExcessMax, 0, `Layer thickness of the layer model set to 8 m: its intake maximum falls to ${fmt(row(rl, /intake, layer model/)[1], 3)} g/kg, the 3-D model gives ${fmt(ol.intakeExcessMax3D, 3)} g/kg and that value is the verdict; the note of the compliance table names the governing model (${/intake: 3-D model/.test(rl.tables.find((t) => t.title === 'Environmental compliance').note) ? 'intake: 3-D model' : 'not named'})`);
          add('Compliance verdict with the 3-D model: disagreement beyond a factor of 2 is flagged', 1, rl.warnings.some((w) => w.level === 'warn' && /differ by more than a factor of 2/.test(w.msg)) && !r3.warnings.some((w) => /differ by more than a factor of 2/.test(w.msg)) ? 1 : 0, 0, `Mixing-zone edge with the 8 m layer: layer model ${fmt(ol.mzFarField, 3)} g/kg against ${fmt(ol.mz3D, 3)} g/kg in the 3-D model → warning; with the default layer (${fmt(o3.mzFarField, 3)} against ${fmt(o3.mz3D, 3)} g/kg) no warning`);
          add('Compliance verdict without the 3-D model is the layer-model verdict', Math.max(row(r0, /^… near-field/)[1], o0.mzFarField), o0.excessAtMixingZone, 0, 'excessAtMixingZone = max(near-field estimate, layer model) when the three-dimensional model is off; the compliance table then has no 3-D lines (' + (r0.tables.find((t) => t.title === 'Environmental compliance').rows.some((q) => /three-dimensional/.test(q[0])) ? 'present' : 'none') + ')');
        }
        const lm = (k) => { let m = 0; if (plan) for (const row of plan.frames[k].z) for (const x of row) if (Number.isFinite(x) && x > m) m = x; return m; };
        add('3-D model through run(): the brine stays near the bed', 1, lm(0) > lm(5) ? 1 : 0, 0, `Largest excess salinity ${fmt(lm(0), 3)} g/kg in the bottom σ-layer against ${fmt(lm(5), 3)} g/kg in the surface layer: the baroclinic pressure gradient and the stable stratification hold the dense plume down`);
      }
    }
    // ---- atmospheric heat exchange, ecology, vertical reconstruction, seasons
    {
      add('Heat budget: back radiation of the sea surface at 20 °C', 0.97 * 5.670374e-8 * 293.15 ** 4, surfaceHeatFlux({ Tw: 20, Ta: 20 }).lwUp, 1e-9, 'Grey-body emission εσT⁴ (W/m²)');
      const hx = heatExchange({ Tw: 22, Ta: 24, rh: 65, W: 5, cloud: 0.3, solar: 220 });
      add('Heat budget: zero net flux at the equilibrium temperature', 0, surfaceHeatFlux({ Tw: hx.Te, Ta: 24, rh: 65, W: 5, cloud: 0.3, solar: 220 }).net, 1e-6, `Equilibrium temperature ${fmt(hx.Te, 4)} °C, exchange coefficient ${fmt(hx.K, 3)} W/m²·K`);
      const gh = flatGrid(12, 12, 50, 10, false), lamH = hx.K / (1025 * 4000 * 10), rh = await farField({ g: gh, flow: flowBasis(gh, true), cur: still, K: new Float64Array(144), tEnd: 86400, closed: true, extra: { rate: 0, decay: new Float64Array(144).fill(lamH), C0: new Float64Array(144).fill(1) }, expThr: [0.5, 2], C0: new Float64Array(144).fill(1) });
      add('Far-field temperature: surface-exchange decay', Math.exp(-lamH * 86400), rh.E[70], 2e-4, 'ΔT(t) = ΔT₀ exp(−K t / ρ c_p h) for a uniform excess in still water');
      add('Far-field heat balance closes', 1, (rh.bal.massE + rh.bal.lostE) / (144 * 2500 * 10), 1e-9, 'Stored + released to the atmosphere ÷ initial excess heat');
      add('Exposure-duration statistics', 1, rh.expo[0][70] - rh.expo[1][70], 1e-12, 'A uniform excess of 1 g/kg is above 0.5 g/kg all the time and never above 2 g/kg');
      add('Dose–response passes through EC10 and EC50', 0.6, doseResponse(1, 1, 3) + doseResponse(3, 1, 3), 1e-9, 'Log-logistic curve: 10 % effect at EC10 and 50 % at EC50');
      let pi3 = 0; for (let k = 0; k < 400; k++) pi3 += (layerProfile(1, 12, 0.2, ((k + 0.5) * 12) / 400) * 12) / 400;
      add('Vertical reconstruction of the layer conserves its content', 0.2 * 12, pi3, 1e-3, 'Depth integral of the half-Gaussian profile equals φ·H·C');
      const dv = Object.fromEntries(suite.inputs.flatMap((q) => q.fields).map((q) => [q.key, q.type === 'table' ? JSON.parse(JSON.stringify(q.value)) : q.value])), sm = { ...dv, design: 'manual', nx: 24, ny: 16, nCycles: 1, seasonal: true, seasons: [{ name: 'Same as base', Ta: dv.Ta, Sa: dv.Sa, dT: dv.dT, uRes: dv.uRes, windSpeed: dv.windSpeed, waveHeight: dv.waveHeight }, { name: 'Cold', Ta: 12, Sa: dv.Sa, dT: 0, uRes: dv.uRes, windSpeed: dv.windSpeed, waveHeight: dv.waveHeight }] };
      const rs = await suite.run(sm), ts = rs.tables.find((t) => t.title === 'Seasonal simulation');
      add('Seasonal simulation: a season equal to the base case reproduces it', rs.outputs.nearFieldDilution, ts.rows[0][11], 1e-9 * rs.outputs.nearFieldDilution, 'Near-field dilution of the sweep row against the main run');
      add('Seasonal simulation: colder (denser) ambient lowers the density contrast', 1, ts.rows[1][8] > ts.rows[0][8] ? 1 : 0, 0, `Densimetric Froude number ${fmt(ts.rows[0][8], 4)} → ${fmt(ts.rows[1][8], 4)} at 12 °C`);
    }
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
  vsTurb: 'k–ε solves transport equations for the turbulent kinetic energy and its dissipation; the Richardson-number closure is algebraic.', vsTime: 'Long enough for the front to travel a few hundred metres (about 0.05–0.15 m/s).', vsNx: 'Cell length = section length ÷ cells.', vsNz: 'Cell height = largest depth on the section ÷ cells; the layer should span at least three cells.',
  airTemp: 'Air temperature at 2–10 m height.', humidity: 'Relative humidity of the air.', cloud: '0 = clear sky, 1 = overcast; raises the atmospheric long-wave radiation.',
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
