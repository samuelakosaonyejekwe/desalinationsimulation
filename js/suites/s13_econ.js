// Suite 13 — Economics and techno-economic analysis.
// Equipment-level CAPEX build-up with cost–capacity scaling, installation factors, indirects, cost-index
// escalation and interest during construction; itemised OPEX; levelised cost of water (annualised and
// discounted-cash-flow, real and nominal); project finance (debt, depreciation, tax, NPV, IRR, MIRR,
// payback, DSCR); carbon accounting; tornado, spider, scenario and Monte-Carlo analyses; real-options
// deferral value; levelised cost of on-site energy; exergy-based (thermoeconomic / exergoeconomic) costing;
// life-cycle assessment coupled to the cost model; a response-surface surrogate of the cost model; and
// production and emissions constraints. All cost data are indicative defaults and every one is editable.
import { brent, clamp, linspace, logspace, sum, mean, std, variance, quantile, histogram, rng, fmt, interp1, lstsq, lhs } from '../core/num.js';
import { osmoticPressure } from '../core/props.js';

// ---- financial primitives -------------------------------------------------------------------------
/** Capital-recovery factor for rate i (fraction) over n periods. */
export const crf = (i, n) => (Math.abs(i) < 1e-12 ? 1 / n : (i * (1 + i) ** n) / ((1 + i) ** n - 1));
/** Net present value of cash flows cf[0..n] (cf[0] at time zero). */
export const npv = (rate, cf) => cf.reduce((s, x, t) => s + x / (1 + rate) ** t, 0);
/** Fisher relation: real rate from nominal rate and inflation (fractions). */
export const fisherReal = (nominal, inflation) => (1 + nominal) / (1 + inflation) - 1;

/** Internal rate of return by bracketing scan + Brent. Returns NaN when the cash flow never changes sign. */
export function irr(cf) {
  const f = (r) => npv(r, cf), grid = [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.4, 0.7, 1.2, 3, 10, -0.02, -0.05, -0.1, -0.2, -0.35, -0.5, -0.7, -0.9, -0.99];
  if (!(cf.some((x) => x > 0) && cf.some((x) => x < 0))) return NaN;
  for (const seg of [grid.slice(0, 12), [0, ...grid.slice(12)]]) for (let k = 1; k < seg.length; k++) {
    const a = seg[k - 1], b = seg[k], fa = f(a), fb = f(b);
    if (fa === 0) return a;
    if (Number.isFinite(fa) && Number.isFinite(fb) && fa * fb <= 0) return brent(f, Math.min(a, b), Math.max(a, b), 1e-12);
  }
  return NaN;
}

/** Modified IRR: negatives financed at `fin`, positives reinvested at `reinv`. */
export function mirr(cf, fin, reinv) {
  const n = cf.length - 1, fv = cf.reduce((s, x, t) => s + (x > 0 ? x * (1 + reinv) ** (n - t) : 0), 0), pv = cf.reduce((s, x, t) => s + (x < 0 ? x / (1 + fin) ** t : 0), 0);
  return pv < 0 && fv > 0 ? (fv / -pv) ** (1 / n) - 1 : NaN;
}

/** Year in which the cumulative cash flow turns positive (linear interpolation); NaN if never. */
export function payback(cf, rate = 0) {
  let cum = 0;
  for (let t = 0; t < cf.length; t++) {
    const x = cf[t] / (1 + rate) ** t, prev = cum;
    cum += x;
    if (t > 0 && prev < 0 && cum >= 0) return t - 1 + -prev / x;
  }
  return NaN;
}

/** Depreciation schedule (array for years 1..horizon) that always sums to the depreciable base. */
export function depreciation(base, life, method = 'sl', horizon = life) {
  const n = Math.max(1, Math.round(life)), d = new Array(Math.max(horizon, n + 1)).fill(0);
  if (method === 'sl') for (let t = 0; t < n; t++) d[t] = base / n;
  else {
    // declining balance with switch to straight line; 'macrs' adds the half-year convention (n + 1 tax years, 150 % rate)
    const half = method === 'macrs', rate = (half ? 1.5 : 2) / n, yrs = half ? n + 1 : n;
    let book = base;
    for (let t = 0; t < yrs; t++) {
      const remaining = half ? n - Math.max(0, t - 0.5) : n - t, slCharge = book / Math.max(remaining, 0.5), db = book * rate * (half && t === 0 ? 0.5 : 1);
      const x = t === yrs - 1 ? book : Math.min(book, Math.max(db, half && t === 0 ? db : slCharge));
      d[t] = x; book -= x;
    }
  }
  return d.slice(0, Math.max(horizon, 1)).map((x, t, a) => (t === a.length - 1 ? x + sum(d.slice(a.length)) : x));
}

/** Level-payment loan. Returns per-year interest, principal and closing balance for years 1..tenor. */
export function loanSchedule(D, rate, tenor) {
  const n = Math.max(1, Math.round(tenor)), pay = D * crf(rate, n), rows = [];
  let bal = D;
  for (let t = 1; t <= n; t++) { const interest = bal * rate, principal = pay - interest; bal -= principal; rows.push({ t, interest, principal, payment: pay, balance: Math.abs(bal) < 1e-9 * Math.max(1, D) ? 0 : bal }); }
  return rows;
}

/** Cox–Ross–Rubinstein binomial option on asset value V with strike K and yield q. */
export function binomialOption({ V, K, r, sigma, T, steps = 60, q = 0, american = true }) {
  const n = Math.max(1, Math.round(steps)), dt = T / n, u = Math.exp(sigma * Math.sqrt(dt)), d = 1 / u, p = clamp((Math.exp((r - q) * dt) - d) / (u - d), 0, 1), disc = Math.exp(-r * dt);
  let val = Array.from({ length: n + 1 }, (_, j) => Math.max(0, V * u ** j * d ** (n - j) - K));
  for (let i = n - 1; i >= 0; i--) val = Array.from({ length: i + 1 }, (_, j) => { const cont = disc * (p * val[j + 1] + (1 - p) * val[j]); return american ? Math.max(cont, V * u ** j * d ** (i - j) - K) : cont; });
  return val[0];
}

const erf = (x) => { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a); return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)); };
const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const blackScholesCall = (V, K, r, s, T) => { const d1 = (Math.log(V / K) + (r + 0.5 * s * s) * T) / (s * Math.sqrt(T)); return V * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d1 - s * Math.sqrt(T)); };

/**
 * Correlated random multipliers. spec: [{ lo, hi }] in % around 1. dist: 'tri' | 'normal' | 'lognormal'.
 * corr: [i, j, rho] couples two columns through a Gaussian copula.
 */
export function sampleMultipliers(n, spec, dist = 'tri', corr = null, seed = 1) {
  const g = rng(seed), out = [];
  for (let s = 0; s < n; s++) {
    const z = spec.map(() => g.normal());
    if (corr) z[corr[1]] = corr[2] * z[corr[0]] + Math.sqrt(1 - corr[2] ** 2) * z[corr[1]];
    out.push(spec.map((d, j) => {
      const lo = d.lo / 100, hi = d.hi / 100;
      if (!(hi > lo)) return 1 + lo;
      if (dist === 'normal') return Math.max(0.05, 1 + ((hi - lo) / 3.29) * z[j]);
      if (dist === 'lognormal') return Math.exp((Math.log((1 + hi) / Math.max(0.05, 1 + lo)) / 3.29) * z[j]);
      const u = clamp(normCdf(z[j]), 1e-9, 1 - 1e-9), a = Math.min(lo, 0), b = Math.max(hi, 0), fc = -a / (b - a);
      return 1 + (u < fc ? a + Math.sqrt(u * (b - a) * -a) : b - Math.sqrt((1 - u) * (b - a) * b));
    }));
  }
  return out;
}

// ---- cost model -----------------------------------------------------------------------------------
const REF = { cap: 1e5, feed: 1e5 / 0.45, brine: 1e5 / 0.45 - 1e5, area: 297600, power: 13750, outfall: 1e5 / 0.45 - 1e5, thermal: 5e4, bc: 100, solids: 100 };
const DRIVERS = [
  { k: 'capex', name: 'Capital cost', lo: -15, hi: 30, good: -1 }, { k: 'elec', name: 'Electricity price', lo: -25, hi: 40, good: -1 }, { k: 'sec', name: 'Specific energy', lo: -8, hi: 12, good: -1 },
  { k: 'cf', name: 'Capacity factor', lo: -12, hi: 4, good: 1 }, { k: 'disc', name: 'Discount rate', lo: -20, hi: 25, good: -1 }, { k: 'memLife', name: 'Membrane life', lo: -30, hi: 30, good: 1 },
  { k: 'chem', name: 'Chemical cost', lo: -20, hi: 30, good: -1 }, { k: 'labour', name: 'Labour cost', lo: -15, hi: 20, good: -1 }, { k: 'maint', name: 'Maintenance cost', lo: -20, hi: 30, good: -1 }, { k: 'life', name: 'Plant life', lo: -20, hi: 20, good: 1 },
];
const BRINE = ['Sea outfall', 'Deep-well injection', 'Evaporation pond', 'Sewer / treatment-works discharge', 'Thermal zero-liquid discharge', 'None (handled by the equipment list)'];
const num = (x, d = 0) => (Number.isFinite(+x) ? +x : d);
/** Parse the table inputs once so that sensitivity and Monte-Carlo loops only redo the arithmetic. */
function prepare(v) {
  const t = (v.indexTable || []).filter((q) => Number.isFinite(+q.year) && +q.index > 0).map((q) => [+q.year, +q.index]).sort((x, y) => x[0] - y[0]), ix = (y) => (t.length ? interp1(t.map((q) => q[0]), t.map((q) => q[1]), y) : 1);
  return {
    escal: ix(v.analysisYear) / ix(v.baseYear),
    equip: (v.equipment || []).map((r) => { const b = String(r.basis || 'cap').trim().toLowerCase(), key = b in REF ? b : 'cap'; return { name: String(r.name || key), key, base: num(r.base) * 1e6, n: num(r.n, 0.7), install: num(r.install, 1), local: ['thermal', 'bc', 'solids'].includes(key) }; }),
    brine: BRINE.map((name, i) => { const r = (v.brineTable || [])[i] || {}; return { name, capex: num(r.capex), opex: num(r.opex), energy: num(r.energy) }; }),
    chems: (v.chemicals || []).map((r) => ({ name: String(r.name || 'chemical'), perProd: String(r.basis || 'feed').trim().toLowerCase().startsWith('p'), dose: num(r.dose), price: num(r.price) })),
    staff: (v.staff || []).map((r) => ({ role: String(r.role || 'staff'), n: num(r.n), salary: num(r.salary) })),
    repl: (v.replacements || []).map((r) => ({ name: String(r.name || 'item'), cost: num(r.cost) * 1e6, every: Math.max(1, Math.round(num(r.every, 10))) })).filter((r) => r.cost > 0),
  };
}

/** Effective electricity price and production factor under a two-band time-of-use tariff. */
function touCalc(v, m) {
  const k = m.elec ?? 1;
  if (!v.tou) return { price: v.elecPrice * k, prodFactor: 1, flat: v.elecPrice * k };
  const hp = clamp(v.peakHours, 0, 24), L = clamp(v.peakLoad / 100, 0, 1), run = hp * L + (24 - hp);
  return { price: (k * (hp * L * v.peakPrice + (24 - hp) * v.offPrice)) / run, prodFactor: run / 24, flat: (k * (hp * v.peakPrice + (24 - hp) * v.offPrice)) / 24 };
}

/** CAPEX and first-year OPEX build-up. m = optional driver multipliers (capex, elec, sec, cf, memLife, chem, labour, maint, land). */
export function buildCosts(v, m = {}, K = prepare(v)) {
  const cap = v.capacity, rec = clamp(v.recovery / 100, 0.05, 0.99), feed = cap / rec, brine = feed - cap, tou = touCalc(v, m);
  const cf = Math.min(1, (v.availability / 100) * (m.cf ?? 1)) * tou.prodFactor, prod = cap * 365 * cf;
  // on-site generation priced at its levelised cost of energy, blended with the grid by the self-supplied share
  const gen = lcoe({ capex: v.genCapex ?? 900, cf: (v.genCF ?? 24) / 100, om: (v.genOM ?? 1.5) / 100, fuel: v.genFuel ?? 0, life: v.genLife ?? 25, rate: fisherReal(((v.discNominal ?? 8) / 100) * (m.disc ?? 1), (v.inflation ?? 2.5) / 100) });
  const self = clamp((v.selfShare ?? 0) / 100, 0, 1), elecP = (1 - self) * tou.price + self * gen.lcoe, gridC = (1 - self) * v.gridCarbon + self * (v.genCarbon ?? 0.03);
  // thermoeconomic heat costing: heat valued at the electricity it could have produced (Carnot factor × expansion efficiency)
  const heatP = v.heatCosting === 'exergy' ? elecP * carnot(v.T0 ?? 25, v.steamT ?? 70) * 0.85 : v.heatPrice;
  const minProd = (cap * 365 * (v.minProdPct ?? 0)) / 100, shortfall = Math.max(0, minProd - prod);
  const area = v.membraneArea > 0 ? v.membraneArea : ((cap / 24) * 1000) / v.designFlux, sec = v.sec * (m.sec ?? 1), power = (sec * cap) / 24 + v.zldPower;
  const X = { cap, feed, brine, area, power, outfall: v.outfallLength > 0 ? brine : 0, thermal: v.thermalCapacity, bc: v.bcFeed, solids: v.solids };
  const escal = K.escal, learn = (1 - v.learnRate / 100) ** v.doublings, sat = Math.max(1, cap / Math.max(v.maxScale, 1));
  const kAll = v.kCapex * (m.capex ?? 1) * escal * v.locFactor;
  const items = K.equip.map((r) => {
    const x = X[r.key], sc = r.local ? 1 : sat, n = clamp(r.n + v.expShift, 0.2, 1.2);
    let cost = x > 0 ? r.base * (x / sc / REF[r.key]) ** n * sc : 0;
    if (r.key === 'outfall') cost *= v.outfallLength / 1000;
    if (r.key === 'area') cost *= learn;
    return { name: r.name, basis: r.key, n, equip: cost * kAll, installed: cost * kAll * r.install };
  });
  // brine-management options (indicative unit costs on brine volume)
  const brineOpts = K.brine.map((r) => ({ name: r.name, capex: r.capex * brine * v.locFactor * escal, opex: r.opex, energy: r.energy }));
  const bo = brineOpts[clamp(Math.round(v.brineOpt), 0, BRINE.length - 1)];
  const direct = sum(items.map((i) => i.installed)) + bo.capex;
  const indPct = { engineering: v.pEng, procurement: v.pProc, commissioning: v.pComm, owner: v.pOwner }, indirect = Object.fromEntries(Object.entries(indPct).map(([k, p]) => [k, (direct * p) / 100]));
  const indTot = sum(Object.values(indirect)), contingency = ((direct + indTot) * v.pCont) / 100, land = v.land * 1e6 * (m.land ?? 1), base = direct + indTot + contingency + land;
  // ---- OPEX at analysis-year prices, full production
  const elecKWh = sec * prod + v.zldPower * 8760 * cf + bo.energy * brine * 365 * cf, thermKWh = v.secThermal * prod;
  const chemRows = K.chems.map((r) => { const kgd = (r.dose * (r.perProd ? cap : feed)) / 1000; return { name: r.name, basis: r.perProd ? 'product' : 'feed', dose: r.dose, price: r.price, kgd, cost: kgd * 365 * cf * r.price * (m.chem ?? 1) }; });
  const staff = K.staff.map((r) => ({ ...r, cost: r.n * r.salary * (1 + v.burden / 100) * (m.labour ?? 1) }));
  const memLife = Math.max(0.5, v.memLife * (m.memLife ?? 1)), tCO2 = (elecKWh * gridC + thermKWh * v.heatCarbon) / 1000;
  const o = {
    elec: elecKWh * elecP, thermal: thermKWh * heatP, mem: (area * v.memPrice * learn * v.locFactor) / memLife,
    cart: (feed / 24 / v.cartFlow) * v.cartPrice * v.cartChanges * cf, cip: area * v.cipCost * v.cleanings, chem: sum(chemRows.map((r) => r.cost)), labour: sum(staff.map((r) => r.cost)),
    maint: (direct * v.maintPct * (m.maint ?? 1)) / 100, ins: (base * v.insPct) / 100, brine: bo.opex * brine * 365 * cf, sludge: (v.sludgeRate / 1000) * feed * 365 * cf * v.sludgeCost / 1000,
    solids: v.solids * 365 * cf * v.solidsCost, lab: v.labCost, carbon: tCO2 * v.carbonPrice, env: v.envCharge * brine * 365 * cf, shortfall: shortfall * (v.shortfallPenalty ?? 0),
  };
  o.overhead = ((o.labour + o.maint) * v.overheadPct) / 100;
  const groups = { energy: { f: 0, v: o.elec + o.thermal }, chem: { f: o.cip, v: o.chem + o.cart }, mem: { f: o.mem, v: 0 }, labour: { f: o.labour + o.overhead, v: 0 }, other: { f: o.maint + o.ins + o.lab + o.shortfall, v: o.brine + o.sludge + o.solids + o.carbon + o.env } };
  const opex = sum(Object.values(o));
  const repl = K.repl.map((r) => ({ ...r, cost: r.cost * kAll }));
  // working capital and interest during construction
  const wc = (opex * v.wcMonths) / 12, nc = Math.max(1, Math.round(v.constYears)), debtFrac = clamp(v.debtPct / 100, 0, 0.95), rl = v.loanRate / 100;
  const w = v.spendProfile === 'front' ? linspace(1.5, 0.5, nc) : v.spendProfile === 'back' ? linspace(0.5, 1.5, nc) : new Array(nc).fill(1), ws = sum(w);
  const spend = w.map((x) => (base * x) / ws), idc = sum(spend.map((s, k) => debtFrac * s * ((1 + rl) ** (nc - k - 0.5) - 1)));
  const tci = base + wc + idc, grant = (base * v.grantPct) / 100, saltRev = v.saltTpd * 365 * cf * v.saltPrice;
  return { cap, rec, feed, brine, cf, prod, area, sec, power, tou, escal, learn, sat, items, brineOpts, bo, direct, indirect, indTot, contingency, land, base, wc, idc, tci, grant, spend, o, groups, opex, chemRows, staff, repl, elecKWh, thermKWh, tCO2, saltRev, memLife, debtFrac, rl, gen, self, elecP, gridC, heatP, minProd, shortfall };
}

/** Rates and horizon. */
function finance(v, m = {}) {
  const dn = (v.discNominal / 100) * (m.disc ?? 1), infl = v.inflation / 100, N = Math.max(2, Math.round(v.life * (m.life ?? 1)));
  return { dn, infl, dr: fisherReal(dn, infl), N, esc: { energy: v.escElec / 100, chem: v.escChem / 100, mem: v.escMem / 100, labour: v.escLabour / 100, other: v.escOther / 100 } };
}

/** Annualised (capital-recovery-factor) levelised cost in real terms, with its breakdown in $/m³. */
export function lcowAnnual(c, f, v) {
  const salv = ((v.salvagePct / 100) * c.base + c.wc) / (1 + f.dr) ** f.N, k = crf(f.dr, f.N);
  const replPV = sum(c.repl.map((r) => { let s = 0; for (let t = r.every; t < f.N; t += r.every) s += r.cost / (1 + f.dr) ** t; return s; }));
  const capital = k * (c.tci - salv), replacements = k * replPV, o = c.o, q = c.prod;
  const parts = { Capital: capital, Electricity: o.elec, 'Thermal energy': o.thermal, 'Membrane replacement': o.mem, Chemicals: o.chem + o.cart + o.cip, Labour: o.labour + o.overhead, Maintenance: o.maint + replacements, 'Brine, sludge and solids': o.brine + o.sludge + o.solids, 'Insurance and laboratory': o.ins + o.lab, 'Carbon and environmental charges': o.carbon + o.env };
  if (o.shortfall > 0) parts['Take-or-pay shortfall penalty'] = o.shortfall;
  const total = sum(Object.values(parts));
  return { lcow: total / q, parts: Object.fromEntries(Object.entries(parts).map(([n, x]) => [n, x / q])), capital, replacements, annualCost: total, crf: k, fcr: (capital + o.ins + o.maint) / c.tci };
}

/** Year-by-year nominal cash flow, debt service, depreciation and tax. */
export function cashFlow(c, f, v, tariff = v.tariff, light = false) {
  const N = f.N, D = Math.max(0, c.debtFrac * (c.base - c.grant) + c.idc), loan = loanSchedule(D, c.rl, Math.min(Math.round(v.tenor), N));
  const dep = depreciation(Math.max(0, c.base + c.idc - c.land - c.grant), Math.min(Math.round(v.depLife), N), v.depMethod, N), tax = v.taxRate / 100, ramp = clamp(v.ramp / 100, 0.1, 1);
  const rows = [], proj = [-(c.tci - c.grant)], eq = [-(c.tci - c.grant - D)], costsNom = [c.tci], water = [0];
  let lossCF = 0;
  const G = Object.keys(c.groups), gf = G.map((g) => c.groups[g].f), gv = G.map((g) => c.groups[g].v), ge = G.map((g) => (1 + f.infl) * (1 + f.esc[g])), gm = G.map(() => 1), te = 1 + v.tariffEsc / 100;
  let fi = 1, ft = 1;
  for (let t = 1; t <= N; t++) {
    fi *= 1 + f.infl; ft *= te;
    const u = t === 1 ? ramp : 1, Q = c.prod * u, cost = new Array(G.length);
    let opex = 0, replace = 0;
    for (let j = 0; j < G.length; j++) { gm[j] *= ge[j]; cost[j] = (gf[j] + gv[j] * u) * gm[j]; opex += cost[j]; }
    if (t < N) for (const r of c.repl) if (t % r.every === 0) replace += r.cost * fi;
    opex += replace;
    const og = light ? null : Object.fromEntries(G.map((g, j) => [g, cost[j]])), revenue = (tariff * ft + v.subsidy * fi) * Q + c.saltRev * u * fi, ebitda = revenue - opex;
    const L = loan[t - 1] || { interest: 0, principal: 0, payment: 0, balance: 0 }, taxable = ebitda - dep[t - 1] - L.interest;
    let due = 0;
    if (taxable < 0) lossCF += -taxable; else { const used = Math.min(lossCF, taxable); lossCF -= used; due = (taxable - used) * tax; }
    const terminal = t === N ? ((v.salvagePct / 100) * c.base + c.wc) * fi : 0;
    const cfProj = ebitda - due + terminal, cfEq = cfProj - L.payment;
    proj.push(cfProj); eq.push(cfEq); costsNom.push(opex - terminal); water.push(Q);
    if (!light) rows.push({ t, Q, revenue, og, replace, opex, ebitda, dep: dep[t - 1], interest: L.interest, principal: L.principal, balance: L.balance, tax: due, terminal, cfProj, cfEq, dscr: L.payment > 0 ? (ebitda - due) / L.payment : NaN });
  }
  const pvCost = npv(f.dn, costsNom), lcowReal = pvCost / npv(f.dr, water), lcowNom = pvCost / npv(f.dn, water);
  return { rows, proj, eq, D, loan, depr: dep, pvCost, lcowReal, lcowNom, npv: npv(f.dn, proj), npvEq: npv(v.equityRate / 100, eq), water };
}

/** Light evaluation used by sensitivity, scenario and Monte-Carlo loops. */
export function evaluate(v, m = {}, withCash = false, K) {
  const c = buildCosts(v, m, K), f = finance(v, m), an = lcowAnnual(c, f, v);
  return { c, f, an, lcow: an.lcow, cash: withCash ? cashFlow(c, f, v) : null };
}

/** Levelised cost of energy of a generating plant: (CRF·CAPEX + fixed O&M) ÷ annual energy + fuel. capex $/kW, cf and om as fractions. */
export function lcoe({ capex, cf, om, fuel = 0, life, rate }) {
  const E = 8760 * Math.max(cf, 1e-6), k = crf(rate, Math.max(1, Math.round(life)));
  return { capital: (k * capex) / E, om: (om * capex) / E, fuel, lcoe: (k * capex + om * capex) / E + fuel, energy: E, crf: k };
}
const carnot = (T0, Ts) => Math.max(0, 1 - (T0 + 273.15) / (Math.max(Ts, T0 + 0.01) + 273.15));

/**
 * Thermoeconomic / exergoeconomic analysis (specific exergy costing). Per m³ of product: fuel exergy (electricity + Carnot-weighted heat),
 * product exergy (least work of separation), exergy destruction and cost rates of each subsystem. Components are costed in sequence:
 * pump → hydraulic exergy; energy recovery returns brine exergy at the membrane-feed unit cost (fuel rule); membrane → product.
 */
export function exergoeconomics(c, an, v) {
  const r = c.rec, T0 = v.T0 ?? 25, wmin = ((osmoticPressure(T0, clamp(v.feedSalinity ?? 35, 0.01, 120)) / 3.6e6) * -Math.log(1 - Math.min(r, 0.985))) / Math.min(r, 0.985);
  const eEl = c.elecKWh / c.prod, cn = carnot(T0, v.steamT ?? 70), eTh = (c.thermKWh / c.prod) * cn, thShare = c.cap > 0 ? clamp(v.thermalCapacity / c.cap, 0, 1) : 0, memShare = 1 - thShare;
  const etaP = clamp((v.etaPumpSet ?? 82) / 100, 0.2, 0.98), etaE = clamp((v.etaErd ?? 95) / 100, 0, 0.99), Whp = Math.min(eEl, (clamp((v.hpShare ?? 78) / 100, 0.05, 1) * c.sec) / 1), aux = eEl - Whp;
  const p = (Whp * etaP * r) / (1 - etaE * (1 - r)), brineEx = (p * (1 - r)) / r, wm = wmin * memShare, cEl = c.elecP, cHeatEx = eTh > 0 ? c.o.thermal / c.prod / eTh : 0;
  // non-energy cost rates ($ per m³) allocated to the subsystems by installed cost, with their own consumables
  const key = (b) => (b === 'power' ? 'pump' : b === 'area' ? 'mem' : b === 'brine' ? 'erd' : b === 'thermal' ? 'thermal' : 'aux'), inst = { pump: 0, mem: 0, erd: 0, thermal: 0, aux: c.bo.capex };
  for (const it of c.items) inst[key(it.basis)] += it.installed;
  if (!(sum(Object.values(inst)) > 0)) inst.aux = 1; // no equipment list: all shared costs go to the balance of plant
  const instTot = sum(Object.values(inst)), o = c.o, own = { pump: 0, erd: 0, thermal: 0, mem: o.mem + o.cip, aux: o.chem + o.cart + o.sludge + o.brine + o.solids + o.env };
  const shared = an.annualCost - o.elec - o.thermal - sum(Object.values(own)), z = Object.fromEntries(Object.keys(inst).map((k) => [k, (own[k] + (shared * inst[k]) / instTot) / c.prod]));
  if (!(eTh > 0)) { z.aux += z.thermal; z.thermal = 0; }
  const cPump = (cEl * Whp + z.pump) / Math.max(Whp * etaP, 1e-12), cHyd = (cEl * Whp + z.pump + z.erd) / Math.max(p, 1e-12), cErd = brineEx > 0 && etaE > 0 ? (cHyd * brineEx + z.erd) / (etaE * brineEx) : 0, cMem = (cHyd * p + z.mem) / Math.max(wm, 1e-12);
  const rows = [
    { name: 'High-pressure pumps and drives', fuel: Whp, prod: Whp * etaP, cF: cEl, cP: cPump, z: z.pump },
    { name: 'Energy recovery', fuel: brineEx, prod: etaE * brineEx, cF: cHyd, cP: cErd, z: z.erd },
    { name: 'Membrane array', fuel: p, prod: wm, cF: cHyd, cP: cMem, z: z.mem },
    { name: 'Intake, pretreatment, post-treatment and brine handling', fuel: aux, prod: 0, cF: cEl, cP: null, z: z.aux },
    ...(eTh > 0 ? [{ name: 'Thermal desalination units', fuel: eTh, prod: wmin * thShare, cF: cHeatEx, cP: (cHeatEx * eTh + z.thermal) / Math.max(wmin * thShare, 1e-12), z: z.thermal }] : []),
  ].map((q) => { const D = q.fuel - q.prod, cD = q.cF * D; return { ...q, D, eff: q.fuel > 0 ? q.prod / q.fuel : 0, cD, f: q.z + cD > 0 ? q.z / (q.z + cD) : 0, rel: q.cP !== null && q.cF > 0 ? (q.cP - q.cF) / q.cF : null }; });
  const fuel = eEl + eTh, dest = sum(rows.map((q) => q.D)), prodCost = cMem * wm + cEl * aux + z.aux + (eTh > 0 ? cHeatEx * eTh + z.thermal : 0);
  return { wmin, eEl, eTh, fuel, carnot: cn, eff: wmin / fuel, kStar: fuel / wmin, p, pBar: p * 36, rows, dest, balance: fuel - wmin - dest, prodCost, cProduct: prodCost / wmin, zTot: sum(Object.values(z)), consistent: p >= wm };
}

/**
 * Life-cycle assessment coupled to the cost model: cradle-to-grave greenhouse-gas emissions and primary energy per m³ over the operating
 * life (construction, membranes, chemicals, electricity, heat, end of life) and the cost of internalising them at the carbon price.
 */
export function lcaTea(c, f, v, an) {
  const N = f.N, W = c.prod * N, efC = v.efCapex ?? 0.3, efM = v.efMem ?? 15, efCh = v.efChem ?? 1.1, eol = (v.efEol ?? 5) / 100, chemKg = sum(c.chemRows.map((q) => q.kgd)) * 365 * c.cf;
  const con = efC * c.direct, st = { Construction: con, 'Membranes (initial + replacements)': efM * c.area * Math.max(1, N / c.memLife), Chemicals: efCh * chemKg * N, Electricity: c.elecKWh * c.gridC * N, Heat: c.thermKWh * v.heatCarbon * N, 'End of life': eol * con };
  const kg = sum(Object.values(st)), ci = kg / W, opCi = (c.tCO2 * 1000) / c.prod, embodied = ci - opCi;
  const pe = { Construction: 3.5 * c.direct, 'Membranes (initial + replacements)': 60 * c.area * Math.max(1, N / c.memLife), Chemicals: 6 * chemKg * N, Electricity: 2.5 * c.elecKWh * N, Heat: 1.1 * c.thermKWh * N, 'End of life': 0.05 * 3.5 * c.direct };
  return { stages: Object.fromEntries(Object.entries(st).map(([k, x]) => [k, x / W])), kgTotal: kg, ci, opCi, embodied, ced: sum(Object.values(pe)) / W, cedStages: Object.fromEntries(Object.entries(pe).map(([k, x]) => [k, x / W])),
    carbonCostLC: (ci * v.carbonPrice) / 1000, lcowLC: an.lcow + ((ci - opCi) * v.carbonPrice) / 1000, ecoEff: ci > 0 && an.lcow > 0 ? 1 / (an.lcow * ci) : 0 };
}

/**
 * Quadratic response surface y ≈ θ·[1, xᵢ, xᵢ², xᵢxⱼ] by least squares. X rows are points in normalised coordinates;
 * cross = indices of the variables whose pairwise interaction terms are included (default: all variables).
 */
export function fitQuadratic(X, y, cross = null) {
  const d = X[0].length, pairs = [], cr = cross || Array.from({ length: d }, (_, i) => i);
  for (let a = 0; a < cr.length; a++) for (let b = a + 1; b < cr.length; b++) pairs.push([cr[a], cr[b]]);
  const feat = (x) => { const q = [1, ...x]; for (let i = 0; i < d; i++) q.push(x[i] * x[i]); for (const [i, j] of pairs) q.push(x[i] * x[j]); return q; };
  const theta = lstsq(X.map(feat), y), np = pairs.length;
  return { theta, terms: theta.length, predict: (x) => { let s = theta[0]; for (let i = 0; i < d; i++) s += (theta[1 + i] + theta[1 + d + i] * x[i]) * x[i]; for (let k = 0; k < np; k++) s += theta[1 + 2 * d + k] * x[pairs[k][0]] * x[pairs[k][1]]; return s; } };
}

/** Response-surface surrogate of the levelised cost over the uncertain drivers, trained on Latin-hypercube runs of the cost model. */
export function surrogateTEA(v, K, unc, { nTrain = 80, nTest = 20, nMC = 4000, dist = 'tri', corr = null, seed = 1 } = {}) {
  const d = unc.length, lo = unc.map((q) => Math.min(q.lo, 0) / 100 - 0.02), hi = unc.map((q) => Math.max(q.hi, 0) / 100 + 0.02), mid = lo.map((a, j) => 1 + (a + hi[j]) / 2), half = lo.map((a, j) => Math.max((hi[j] - a) / 2, 1e-6));
  const toX = (mult) => mult.map((x, j) => (x - mid[j]) / half[j]), run = (mult) => lcowAnnual(buildCosts(v, Object.fromEntries(unc.map((q, j) => [q.k, mult[j]])), K), finance(v, Object.fromEntries(unc.map((q, j) => [q.k, mult[j]]))), v).lcow;
  const pts = lhs(nTrain + nTest, d, seed).map((u) => u.map((x, j) => mid[j] + (2 * x - 1) * half[j])), ys = pts.map(run), inter = ['capex', 'elec', 'sec', 'cf', 'disc', 'life'].map((k) => unc.findIndex((q) => q.k === k)).filter((i) => i >= 0), fit = fitQuadratic(pts.slice(0, nTrain).map(toX), ys.slice(0, nTrain), inter.length >= 2 ? inter : null);
  const yt = ys.slice(nTrain), yh = pts.slice(nTrain).map((q) => fit.predict(toX(q))), mu = mean(yt), r2 = 1 - sum(yt.map((a, i) => (a - yh[i]) ** 2)) / Math.max(sum(yt.map((a) => (a - mu) ** 2)), 1e-300), maxErr = 100 * Math.max(...yt.map((a, i) => Math.abs(yh[i] / a - 1)));
  const draws = sampleMultipliers(nMC, unc, dist, corr, seed + 17), ymc = draws.map((q) => fit.predict(toX(q)));
  // first-order variance-based sensitivity indices S_i = Var(E[y | x_i]) / Var(y), from the surrogate sample binned in x_i
  const vy = variance(ymc), my = mean(ymc), nb = 24, S1 = unc.map((_, j) => {
    let a = Infinity, b = -Infinity;
    for (let i = 0; i < nMC; i++) { const x = draws[i][j]; if (x < a) a = x; if (x > b) b = x; }
    if (!(b > a)) return 0;
    const cnt = new Array(nb).fill(0), acc = new Array(nb).fill(0), w = (b - a) / nb;
    for (let i = 0; i < nMC; i++) { const k = Math.min(nb - 1, Math.floor((draws[i][j] - a) / w)); cnt[k]++; acc[k] += ymc[i]; }
    let ve = 0;
    for (let k = 0; k < nb; k++) if (cnt[k] > 0) ve += (cnt[k] / nMC) * (acc[k] / cnt[k] - my) ** 2;
    return vy > 0 ? ve / vy : 0;
  });
  const srt = Float64Array.from(ymc).sort(), qt = (q) => { const x = (nMC - 1) * q, i = Math.floor(x); return i + 1 < nMC ? srt[i] + (x - i) * (srt[i + 1] - srt[i]) : srt[i]; };
  return { fit, toX, r2, maxErr, nTrain, nTest, nMC, p10: qt(0.1), p50: qt(0.5), p90: qt(0.9), mean: my, sd: Math.sqrt(vy), S1, ymc, parity: { model: yt, sur: yh } };
}

/** Scale the extensive technical inputs to another capacity (plant-size sweep, calibration rows). */
function atCapacity(v, cap, ref = v.capacity) {
  const r = cap / ref;
  return [{ ...v, capacity: cap, membraneArea: v.membraneArea * r, zldPower: v.zldPower * r, thermalCapacity: v.thermalCapacity * r, bcFeed: v.bcFeed * r, solids: v.solids * r, saltTpd: v.saltTpd * r }, { labour: r ** 0.3, land: r ** 0.6 }];
}

const defaultsOf = (s) => Object.fromEntries(s.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.type === 'table' ? f.value.map((r) => ({ ...r })) : f.value]));
const CHEMS = [{ name: 'Ferric chloride (coagulant)', basis: 'feed', dose: 5, price: 0.45 }, { name: 'Sulphuric acid', basis: 'feed', dose: 15, price: 0.12 }, { name: 'Antiscalant', basis: 'feed', dose: 2.5, price: 2.6 }, { name: 'Sodium bisulphite', basis: 'feed', dose: 3, price: 0.55 },
  { name: 'Sodium hypochlorite (as Cl₂)', basis: 'feed', dose: 2, price: 0.5 }, { name: 'Caustic soda', basis: 'product', dose: 8, price: 0.45 }, { name: 'Hydrated lime', basis: 'product', dose: 35, price: 0.16 }, { name: 'Carbon dioxide', basis: 'product', dose: 20, price: 0.22 }];

const suite = {
  id: 'econ', num: 13, title: 'Economics & Techno-Economic Analysis', short: 'Economics', icon: '💰',
  tagline: 'Capital and operating cost build-up, levelised cost of water, project finance, carbon, sensitivity and Monte-Carlo risk.',
  description: 'Builds the capital cost item by item from cost–capacity correlations, installation factors, indirect costs, contingency, cost-index escalation and interest during construction, and the operating cost from energy, membranes, chemicals, labour, maintenance and residuals. The levelised cost of water is computed both by annualising capital with the capital-recovery factor and from a year-by-year discounted cash flow with debt, depreciation, tax, inflation and category-specific escalation. Tornado, spider, scenario and seeded Monte-Carlo analyses show which assumptions drive the water cost and how uncertain it is; a binomial tree values the option to defer the investment.',
  guide: [
    'Enter plant capacity, recovery and specific energy — or pull them from the plant, RO, pump and ZLD suites.',
    'Review the equipment cost table, chemical prices and staffing. All figures are indicative defaults in US dollars: replace them with quotations wherever you have them.',
    'Set the financing (discount rate, inflation, debt share, tax) and the water tariff on the Setup tab.',
    'Run. Read the cost breakdown first, then the tornado chart and the Monte-Carlo range before quoting a single number.',
  ],
  implemented: ['capital-recovery-factor', 'annualized-capital-cost', 'net-present-value', 'internal-rate-of-return', 'discounted-payback', 'simple-payback', 'levelized-cost-of-water', 'fixed-charge-rate', 'depreciation', 'discounted-cash-flow', 'operating-cost', 'maintenance-cost', 'replacement-cost', 'salvage-value', 'working-capital', 'tax/cash-flow', 'learning-curve', 'economies-of-scale', 'cost-capacity scaling', 'inflation/escalation', 'real/nominal discount-rate', 'break-even',
    'techno-economic-process', 'environmental-economic', 'life-cycle-cost', 'water-energy-cost', 'stochastic techno-economic', 'monte-carlo cash-flow', 'probabilistic lcow', 'real-options', 'multi-objective cost-energy-environment',
    'initial capex', 'financing structure', 'debt/equity', 'initial electricity and chemical prices', 'initial production', 'discount rate', 'inflation rate', 'tax assumptions', 'asset value', 'project-lifetime condition', 'terminal/salvage-value condition', 'maximum water-cost constraint', 'debt-service constraint', 'replacement schedule', 'capacity limit', 'resource-price scenario', 'terminal cash-flow',
    'capital-cost estimation', 'equipment-cost estimation', 'installation and construction cost', 'operating-cost estimation', 'electricity cost', 'chemical cost', 'membrane-replacement cost', 'labour and maintenance cost', 'intake and outfall cost', 'brine-management cost', 'waste-disposal cost', 'product and resource-recovery revenue', 'financing', 'depreciation', 'taxation where applicable', 'discounting', 'cash-flow modelling', 'levelised cost of water', 'net-present-value analysis', 'internal-rate-of-return analysis', 'payback-period analysis', 'lifecycle costing', 'sensitivity analysis', 'scenario analysis', 'uncertainty and monte carlo analysis', 'carbon and environmental cost', 'capacity-factor analysis', 'plant-lifetime analysis', 'multi-objective techno-economic',
    'levelized-cost-of-energy', 'thermoeconomic', 'exergoeconomic', 'life-cycle-assessment-techno-economic', 'surrogate', 'economic initial conditions', 'horizon/constraint conditions', 'minimum production constraint', 'emissions constraint'],
  equationsNote: 'A screening- to feasibility-level estimate (roughly ±25–30 % on capital). The bundled cost correlations, cost index, chemical prices, salaries and brine-management unit costs are indicative, generic figures in US dollars — not quotations — and should be calibrated to recent regional projects. Capital is placed at the start of operation with interest during construction added; major replacements are expensed in the year they occur; tax losses are carried forward without limit. The real-options value uses a simple binomial tree with a constant volatility and should be read as an indication only. The levelised cost of energy is that of a single generator at constant output, bought as under a power-purchase agreement (its capital is not added to the plant CAPEX). The thermoeconomic / exergoeconomic analysis is a lumped specific-exergy-costing model of four subsystems: the membrane feed pressure is inferred from the entered specific energy, pumping share and efficiencies, the brine is costed by the fuel rule, and non-energy costs are allocated by installed cost — use suites 1, 3 and 12 for stream-level exergy. The life-cycle assessment covers greenhouse gases and primary energy with generic emission factors, not a full inventory. The surrogate is a quadratic response surface, valid only inside the uncertainty ranges it was trained on. The two catalogue lines on “economic initial conditions” and “horizon/constraint conditions” describe how the year-0 state, the project horizon and the constraints are treated here; they are reported in the table of initial conditions, horizon and constraints.',

  inputs: [
    { group: 'Plant and performance', help: 'Technical basis of the estimate. Pull from the other suites when they have been run.', fields: [
      { key: 'capacity', label: 'Product capacity', unit: 'm³/d', value: 100000, min: 100, max: 2e6, typical: [1000, 600000] },
      { key: 'availability', label: 'Capacity factor (availability × utilisation)', unit: '%', value: 92, min: 20, max: 100 },
      { key: 'recovery', label: 'Overall recovery', unit: '%', value: 45, min: 5, max: 99 },
      { key: 'sec', label: 'Specific electrical energy (whole plant)', unit: 'kWh/m³', value: 3.3, min: 0.1, max: 60, typical: [0.5, 5] },
      { key: 'secThermal', label: 'Specific thermal energy', unit: 'kWh/m³', value: 0, min: 0, max: 400, help: 'Heat per m³ of total product; zero for membrane-only plants.' },
      { key: 'membraneArea', label: 'Installed membrane area (0 = estimate from flux)', unit: 'm²', value: 0, min: 0, max: 2e7 },
      { key: 'designFlux', label: 'Design flux for the area estimate', unit: 'L/m²·h', value: 14, min: 3, max: 45, showIf: (v) => !(v.membraneArea > 0) },
      { key: 'memLife', label: 'Membrane life', unit: 'years', value: 6, min: 1, max: 15, help: 'Average age at replacement; the annual replacement rate is 1 ÷ life.' },
      { key: 'cleanings', label: 'Chemical cleanings per year', unit: '1/y', value: 3, min: 0, max: 24 },
      { key: 'outfallLength', label: 'Outfall length', unit: 'm', value: 1000, min: 0, max: 20000 },
      { key: 'thermalCapacity', label: 'Thermal-desalination capacity (hybrid plants)', unit: 'm³/d', value: 0, min: 0, max: 1e6 },
      { key: 'bcFeed', label: 'Brine-concentrator feed', unit: 'm³/h', value: 0, min: 0, max: 5000 },
      { key: 'solids', label: 'Crystalliser solids', unit: 't/d', value: 0, min: 0, max: 20000 },
      { key: 'zldPower', label: 'Additional power not in the specific energy (e.g. ZLD)', unit: 'kW', value: 0, min: 0, max: 5e5 },
      { key: 'saltTpd', label: 'Saleable salts and minerals', unit: 't/d', value: 0, min: 0, max: 20000 },
    ] },
    { group: 'Capital cost', help: 'Purchased-equipment cost of each item at the reference size, scaled as cost = base × (size ÷ reference size)^n and multiplied by its installation factor. Indicative figures — edit freely.', fields: [
      { key: 'equipment', label: 'Equipment cost correlations (reference plant: 100 000 m³/d, 45 % recovery)', type: 'table',
        columns: [{ key: 'name', label: 'Item' }, { key: 'basis', label: 'Scales with' }, { key: 'base', label: 'Cost at reference size', unit: 'M$' }, { key: 'n', label: 'Exponent n' }, { key: 'install', label: 'Installation factor' }],
        value: [{ name: 'Intake, screens and intake pumps', basis: 'feed', base: 8.0, n: 0.75, install: 1.15 }, { name: 'Outfall and diffuser (per 1000 m)', basis: 'outfall', base: 3.6, n: 0.65, install: 1.1 }, { name: 'Pretreatment (DAF/filtration/UF, cartridges)', basis: 'feed', base: 9.5, n: 0.8, install: 1.4 },
          { name: 'Membrane elements', basis: 'area', base: 5.1, n: 1.0, install: 1.05 }, { name: 'Pressure vessels and racks', basis: 'area', base: 3.6, n: 0.95, install: 1.3 }, { name: 'High-pressure pumps and motors', basis: 'power', base: 5.6, n: 0.75, install: 1.4 },
          { name: 'Energy-recovery devices', basis: 'brine', base: 3.2, n: 0.95, install: 1.3 }, { name: 'Piping and valves', basis: 'cap', base: 6.8, n: 0.75, install: 1.5 }, { name: 'Electrical supply and drives', basis: 'power', base: 6.4, n: 0.75, install: 1.4 },
          { name: 'Instrumentation and control', basis: 'cap', base: 2.6, n: 0.6, install: 1.3 }, { name: 'Civil works and buildings', basis: 'cap', base: 10.5, n: 0.72, install: 1.0 }, { name: 'Remineralisation and disinfection', basis: 'cap', base: 2.7, n: 0.75, install: 1.3 },
          { name: 'Product storage and pumping', basis: 'cap', base: 3.0, n: 0.75, install: 1.2 }, { name: 'Thermal desalination units', basis: 'thermal', base: 52, n: 0.8, install: 1.15 }, { name: 'Brine concentrator', basis: 'bc', base: 9, n: 0.7, install: 1.6 }, { name: 'Crystalliser and solids handling', basis: 'solids', base: 5, n: 0.65, install: 1.6 }],
        help: 'Scaling bases: cap (product m³/d), feed, brine, area (membrane m²), power (kW), outfall (brine flow × length), thermal (m³/d, reference 50 000), bc (m³/h, reference 100), solids (t/d, reference 100). Items whose basis is zero cost nothing.' },
      { key: 'kCapex', label: 'Cost-correlation multiplier', unit: '×', value: 1, min: 0.3, max: 3, help: 'Scales every equipment cost. Calibrate it against reference projects.' },
      { key: 'expShift', label: 'Scale-exponent adjustment', unit: '–', value: 0, min: -0.3, max: 0.3, help: 'Added to every exponent n; negative values strengthen the economy of scale.' },
      { key: 'maxScale', label: 'Largest single-plant size before replication', unit: 'm³/d', value: 250000, min: 5000, max: 2e6, help: 'Above this capacity additional output is built as parallel plants, so the economy of scale stops.' },
      { key: 'locFactor', label: 'Location factor', unit: '×', value: 1, min: 0.5, max: 2.5, help: 'Regional construction-cost level relative to the reference data.' },
      { key: 'pEng', label: 'Engineering and design', unit: '% of direct', value: 7, min: 0, max: 25 },
      { key: 'pProc', label: 'Procurement and construction management', unit: '% of direct', value: 5, min: 0, max: 25 },
      { key: 'pComm', label: 'Commissioning and start-up', unit: '% of direct', value: 2, min: 0, max: 15 },
      { key: 'pOwner', label: "Owner's costs, permits and development", unit: '% of direct', value: 4, min: 0, max: 25 },
      { key: 'pCont', label: 'Contingency', unit: '% of direct + indirect', value: 8, min: 0, max: 40 },
      { key: 'land', label: 'Land', unit: 'M$', value: 2.5, min: 0, max: 500 },
      { key: 'wcMonths', label: 'Working capital', unit: 'months of OPEX', value: 2, min: 0, max: 12 },
    ] },
    { group: 'Operating cost', help: 'Recurring costs at analysis-year prices. Indicative figures — edit freely.', fields: [
      { key: 'elecPrice', label: 'Electricity price', unit: '$/kWh', value: 0.08, min: 0, max: 1, showIf: (v) => !v.tou },
      { key: 'tou', label: 'Use a time-of-use tariff', type: 'bool', value: false },
      { key: 'peakPrice', label: 'Peak price', unit: '$/kWh', value: 0.14, min: 0, max: 2, showIf: (v) => v.tou },
      { key: 'offPrice', label: 'Off-peak price', unit: '$/kWh', value: 0.065, min: 0, max: 2, showIf: (v) => v.tou },
      { key: 'peakHours', label: 'Peak hours per day', unit: 'h', value: 5, min: 0, max: 24, showIf: (v) => v.tou },
      { key: 'peakLoad', label: 'Plant load during peak hours', unit: '%', value: 60, min: 0, max: 100, help: 'Turning down at peak saves on price but lowers annual production.', showIf: (v) => v.tou },
      { key: 'heatPrice', label: 'Heat price', unit: '$/kWh', value: 0.012, min: 0, max: 0.3, help: 'Cost of low-grade steam or waste heat per kWh thermal.' },
      { key: 'memPrice', label: 'Membrane element price', unit: '$/m²', value: 17, min: 2, max: 150 },
      { key: 'cipCost', label: 'Cleaning cost per cleaning', unit: '$/m²', value: 0.35, min: 0, max: 5 },
      { key: 'cartFlow', label: 'Flow per cartridge filter', unit: 'm³/h', value: 3.5, min: 0.2, max: 20 },
      { key: 'cartPrice', label: 'Cartridge price', unit: '$', value: 9, min: 0.5, max: 200 },
      { key: 'cartChanges', label: 'Cartridge change-outs per year', unit: '1/y', value: 6, min: 0, max: 52 },
      { key: 'chemicals', label: 'Chemicals', type: 'table', columns: [{ key: 'name', label: 'Chemical' }, { key: 'basis', label: 'Dosed per m³ of (feed / product)' }, { key: 'dose', label: 'Dose', unit: 'mg/L' }, { key: 'price', label: 'Price', unit: '$/kg' }], value: CHEMS, help: 'Dose as 100 % active product.' },
      { key: 'staff', label: 'Staffing', type: 'table', columns: [{ key: 'role', label: 'Role' }, { key: 'n', label: 'Headcount' }, { key: 'salary', label: 'Salary', unit: '$/y' }],
        value: [{ role: 'Plant manager', n: 1, salary: 120000 }, { role: 'Shift supervisors', n: 4, salary: 70000 }, { role: 'Operators', n: 16, salary: 45000 }, { role: 'Maintenance technicians', n: 8, salary: 48000 }, { role: 'Laboratory and quality', n: 3, salary: 50000 }, { role: 'Administration and HSE', n: 4, salary: 45000 }] },
      { key: 'burden', label: 'Payroll burden', unit: '%', value: 30, min: 0, max: 100, help: 'Social charges, benefits and training on top of salaries.' },
      { key: 'overheadPct', label: 'Overheads', unit: '% of labour + maintenance', value: 12, min: 0, max: 60 },
      { key: 'maintPct', label: 'Maintenance and spares', unit: '% of direct CAPEX /y', value: 1.8, min: 0, max: 8 },
      { key: 'insPct', label: 'Insurance', unit: '% of CAPEX /y', value: 0.5, min: 0, max: 3 },
      { key: 'labCost', label: 'Laboratory and monitoring', unit: '$/y', value: 350000, min: 0, max: 1e7 },
      { key: 'sludgeRate', label: 'Sludge production', unit: 'kg dry solids per 1000 m³ feed', value: 27, min: 0, max: 500 },
      { key: 'sludgeCost', label: 'Sludge disposal', unit: '$/t', value: 60, min: 0, max: 500 },
      { key: 'solidsCost', label: 'Crystalliser solids disposal', unit: '$/t', value: 40, min: 0, max: 500 },
      { key: 'replacements', label: 'Major replacement schedule', type: 'table', columns: [{ key: 'name', label: 'Item' }, { key: 'cost', label: 'Cost', unit: 'M$' }, { key: 'every', label: 'Every', unit: 'years' }],
        value: [{ name: 'High-pressure pump overhaul', cost: 0.9, every: 8 }, { name: 'Energy-recovery rotors and seals', cost: 0.5, every: 10 }, { name: 'Filter media / UF modules', cost: 1.6, every: 7 }, { name: 'Control system refresh', cost: 1.2, every: 10 }, { name: 'Drives and switchgear', cost: 1.8, every: 12 }],
        help: 'Lump-sum replacements in addition to membranes and routine maintenance; costs are for the 100 000 m³/d default — adjust for your plant.' },
    ] },
    { group: 'Brine management', fields: [
      { key: 'brineOpt', label: 'Brine route', type: 'select', value: 0, options: BRINE.map((label, value) => ({ value, label })) },
      { key: 'brineTable', label: 'Brine-management unit costs (rows in the order of the list above)', type: 'table', columns: [{ key: 'name', label: 'Route' }, { key: 'capex', label: 'Capital', unit: '$ per m³/d of brine' }, { key: 'opex', label: 'Operating', unit: '$/m³ brine' }, { key: 'energy', label: 'Energy', unit: 'kWh/m³ brine' }],
        value: [{ name: BRINE[0], capex: 0, opex: 0.008, energy: 0.03 }, { name: BRINE[1], capex: 550, opex: 0.25, energy: 0.9 }, { name: BRINE[2], capex: 2600, opex: 0.35, energy: 0.05 }, { name: BRINE[3], capex: 60, opex: 0.45, energy: 0.05 }, { name: BRINE[4], capex: 9000, opex: 1.1, energy: 26 }, { name: BRINE[5], capex: 0, opex: 0, energy: 0 }],
        help: 'Indicative unit costs used for the selected route and for the comparison table. The sea outfall structure itself is in the equipment list.' },
      { key: 'envCharge', label: 'Environmental discharge charge', unit: '$/m³ brine', value: 0, min: 0, max: 5 },
    ] },
    { group: 'Revenue, carbon and support', fields: [
      { key: 'tariff', label: 'Water tariff', unit: '$/m³', value: 1.05, min: 0, max: 20 },
      { key: 'tariffEsc', label: 'Tariff indexation', unit: '%/y', value: 2.5, min: -5, max: 20 },
      { key: 'saltPrice', label: 'Price of recovered salts', unit: '$/t', value: 45, min: 0, max: 5000 },
      { key: 'gridCarbon', label: 'Grid emission factor', unit: 'kgCO₂/kWh', value: 0.45, min: 0, max: 1.3 },
      { key: 'heatCarbon', label: 'Heat emission factor', unit: 'kgCO₂/kWh', value: 0.07, min: 0, max: 0.5 },
      { key: 'carbonPrice', label: 'Carbon price', unit: '$/tCO₂', value: 25, min: 0, max: 500 },
      { key: 'subsidy', label: 'Operating subsidy', unit: '$/m³', value: 0, min: 0, max: 5 },
      { key: 'grantPct', label: 'Capital grant', unit: '% of CAPEX', value: 0, min: 0, max: 80 },
      { key: 'fx', label: 'Local currency per US dollar', unit: '', value: 1, min: 0.0001, max: 1e6 },
      { key: 'currency', label: 'Local currency code', type: 'text', value: 'USD' },
    ] },
    { group: 'Financing and horizon', tab: 'setup', help: 'Economic initial conditions and horizon: treat these as explicit scenarios rather than calibrating them.', fields: [
      { key: 'life', label: 'Operating life', unit: 'years', value: 25, min: 5, max: 50, step: 1 },
      { key: 'constYears', label: 'Construction period', unit: 'years', value: 3, min: 1, max: 8, step: 1 },
      { key: 'spendProfile', label: 'Construction spending', type: 'select', value: 'even', options: [{ value: 'even', label: 'Even' }, { value: 'front', label: 'Front-loaded' }, { value: 'back', label: 'Back-loaded' }] },
      { key: 'discNominal', label: 'Discount rate (nominal)', unit: '%/y', value: 8, min: 0, max: 30, help: 'Weighted cost of capital in money-of-the-day terms.' },
      { key: 'inflation', label: 'General inflation', unit: '%/y', value: 2.5, min: -2, max: 30 },
      { key: 'debtPct', label: 'Debt share of capital', unit: '%', value: 70, min: 0, max: 95 },
      { key: 'loanRate', label: 'Loan interest rate', unit: '%/y', value: 6, min: 0, max: 30 },
      { key: 'tenor', label: 'Loan tenor', unit: 'years', value: 15, min: 1, max: 40, step: 1 },
      { key: 'equityRate', label: 'Required return on equity', unit: '%/y', value: 12, min: 0, max: 40 },
      { key: 'taxRate', label: 'Corporate tax rate', unit: '%', value: 20, min: 0, max: 60 },
      { key: 'depMethod', label: 'Depreciation', type: 'select', value: 'sl', options: [{ value: 'sl', label: 'Straight line' }, { value: 'db', label: 'Double declining balance' }, { value: 'macrs', label: 'Accelerated, half-year convention (MACRS-like)' }] },
      { key: 'depLife', label: 'Depreciation life', unit: 'years', value: 20, min: 3, max: 50, step: 1 },
      { key: 'salvagePct', label: 'Salvage value', unit: '% of CAPEX (real)', value: 5, min: 0, max: 50 },
      { key: 'ramp', label: 'Production in the first year', unit: '% of normal', value: 90, min: 10, max: 100, help: 'Ramp-up after commissioning.' },
    ] },
    { group: 'Energy supply: on-site generation and heat costing', tab: 'setup', help: 'Levelised cost of energy of a dedicated generator (photovoltaic, wind, gas engine …) and the share of the plant electricity it supplies. With a zero share the grid price is used and the LCOE is reported for comparison.', fields: [
      { key: 'selfShare', label: 'Electricity supplied by on-site generation', unit: '%', value: 0, min: 0, max: 100, help: 'This share is priced at the levelised cost of energy below and carries its emission factor.' },
      { key: 'genCapex', label: 'Generator specific capital cost', unit: '$/kW', value: 900, min: 100, max: 10000, help: 'Indicative: utility photovoltaics 700–1100, onshore wind 1200–1700, gas engine 800–1200.' },
      { key: 'genCF', label: 'Generator capacity factor', unit: '%', value: 24, min: 5, max: 98 },
      { key: 'genOM', label: 'Generator fixed O&M', unit: '% of capital /y', value: 1.5, min: 0, max: 10 },
      { key: 'genFuel', label: 'Generator fuel cost', unit: '$/kWh', value: 0, min: 0, max: 1, help: 'Fuel price ÷ efficiency; zero for solar and wind.' },
      { key: 'genLife', label: 'Generator life', unit: 'years', value: 25, min: 5, max: 50, step: 1 },
      { key: 'genCarbon', label: 'Generator emission factor', unit: 'kgCO₂/kWh', value: 0.03, min: 0, max: 1.3 },
      { key: 'heatCosting', label: 'Heat costing', type: 'select', value: 'price', options: [{ value: 'price', label: 'Entered heat price' }, { value: 'exergy', label: 'Thermoeconomic: by exergy (electricity the steam could have produced)' }], help: 'The thermoeconomic option values each kWh of heat at the electricity price × Carnot factor of the heat source × 0.85 expansion efficiency.' },
    ] },
    { group: 'Exergy and life-cycle basis', tab: 'setup', help: 'Reference state and subsystem efficiencies for the exergy costing, and emission factors for the life-cycle assessment. Indicative figures — edit freely.', fields: [
      { key: 'feedSalinity', label: 'Feed salinity', unit: 'g/kg', value: 35, min: 0.5, max: 120, help: 'Sets the least work of separation (product exergy).' },
      { key: 'T0', label: 'Ambient (dead-state) temperature', unit: '°C', value: 25, min: 0, max: 45 },
      { key: 'steamT', label: 'Heat-source temperature', unit: '°C', value: 70, min: 40, max: 250, help: 'For the Carnot factor of thermal energy.' },
      { key: 'hpShare', label: 'Share of electricity used by high-pressure pumping', unit: '%', value: 78, min: 20, max: 100, help: 'The rest drives intake, pretreatment, post-treatment and brine handling.' },
      { key: 'etaPumpSet', label: 'High-pressure pump × motor × drive efficiency', unit: '%', value: 82, min: 40, max: 95 },
      { key: 'etaErd', label: 'Energy-recovery efficiency', unit: '%', value: 95, min: 0, max: 99, help: '0 for a plant without energy recovery.' },
      { key: 'efCapex', label: 'Embodied carbon of construction', unit: 'kgCO₂e per $ direct cost', value: 0.3, min: 0, max: 2 },
      { key: 'efMem', label: 'Embodied carbon of membrane elements', unit: 'kgCO₂e/m²', value: 15, min: 0, max: 100 },
      { key: 'efChem', label: 'Embodied carbon of chemicals', unit: 'kgCO₂e/kg', value: 1.1, min: 0, max: 10 },
      { key: 'efEol', label: 'End-of-life emissions', unit: '% of construction', value: 5, min: 0, max: 50 },
    ] },
    { group: 'Escalation and cost basis', tab: 'setup', fields: [
      { key: 'escElec', label: 'Energy price escalation above inflation', unit: '%/y', value: 0.5, min: -5, max: 10 },
      { key: 'escLabour', label: 'Labour escalation above inflation', unit: '%/y', value: 1, min: -5, max: 10 },
      { key: 'escChem', label: 'Chemicals escalation above inflation', unit: '%/y', value: 0, min: -5, max: 10 },
      { key: 'escMem', label: 'Membrane price escalation above inflation', unit: '%/y', value: -1, min: -10, max: 10 },
      { key: 'escOther', label: 'Other costs escalation above inflation', unit: '%/y', value: 0, min: -5, max: 10 },
      { key: 'baseYear', label: 'Year of the cost data', unit: '', value: 2023, min: 2000, max: 2040, step: 1 },
      { key: 'analysisYear', label: 'Analysis (start-up) year', unit: '', value: 2026, min: 2000, max: 2050, step: 1 },
      { key: 'indexTable', label: 'Plant-cost index', type: 'table', columns: [{ key: 'year', label: 'Year' }, { key: 'index', label: 'Index' }],
        value: [{ year: 2010, index: 551 }, { year: 2012, index: 585 }, { year: 2014, index: 576 }, { year: 2016, index: 542 }, { year: 2018, index: 603 }, { year: 2019, index: 608 }, { year: 2020, index: 596 }, { year: 2021, index: 709 }, { year: 2022, index: 816 }, { year: 2023, index: 798 }, { year: 2024, index: 800 }, { year: 2025, index: 808 }, { year: 2026, index: 820 }, { year: 2030, index: 880 }],
        help: 'Chemical-plant cost index style series. Values to 2023 are rounded annual averages; later years are projections — replace with published figures.' },
      { key: 'learnRate', label: 'Membrane learning rate', unit: '% per doubling', value: 12, min: 0, max: 40, help: 'Cost reduction for each doubling of cumulative installed capacity.' },
      { key: 'doublings', label: 'Capacity doublings since the cost data', unit: '', value: 0, min: 0, max: 5, help: 'Learning-curve factor = (1 − rate)^doublings, applied to membrane elements and replacement.' },
    ] },
    { group: 'Uncertainty and scenarios', tab: 'setup', fields: [
      { key: 'sensPct', label: 'Sensitivity swing', unit: '±%', value: 20, min: 1, max: 60 },
      { key: 'uncert', label: 'Uncertainty ranges (keep the row order)', type: 'table', columns: [{ key: 'name', label: 'Driver' }, { key: 'lo', label: 'Low', unit: '%' }, { key: 'hi', label: 'High', unit: '%' }], value: DRIVERS.map((d) => ({ name: d.name, lo: d.lo, hi: d.hi })),
        help: 'Minimum and maximum for triangular sampling; 5th and 95th percentiles for normal and log-normal sampling.' },
      { key: 'mcDist', label: 'Sampling distribution', type: 'select', value: 'tri', options: [{ value: 'tri', label: 'Triangular' }, { value: 'normal', label: 'Normal' }, { value: 'lognormal', label: 'Log-normal' }] },
      { key: 'mcCorr', label: 'Correlation: electricity price ↔ chemical cost', unit: '–', value: 0.4, min: -0.95, max: 0.95, help: 'Energy and chemical prices tend to move together.' },
      { key: 'mcSeed', label: 'Random seed', unit: '', value: 2024, min: 1, max: 1e9, step: 1 },
      { key: 'targetLcow', label: 'Maximum acceptable water cost', unit: '$/m³', value: 1.0, min: 0.05, max: 20 },
      { key: 'minDscr', label: 'Minimum debt-service cover ratio', unit: '–', value: 1.2, min: 1, max: 3 },
      { key: 'minProdPct', label: 'Minimum annual production (take-or-pay)', unit: '% of nameplate', value: 85, min: 0, max: 100, help: 'Contractual minimum output. 0 = no constraint. The chance of missing it is taken from the Monte-Carlo sample.' },
      { key: 'shortfallPenalty', label: 'Penalty for production shortfall', unit: '$/m³ short', value: 0, min: 0, max: 10, showIf: (v) => v.minProdPct > 0, help: 'Charged on every m³ below the minimum and added to the operating cost.' },
      { key: 'maxCarbon', label: 'Emissions cap', unit: 'kgCO₂/m³', value: 2, min: 0, max: 50, help: 'Limit on the operational carbon intensity. 0 = no cap. When it is exceeded, the renewable share of electricity needed to comply and its water cost are solved.' },
      { key: 'renPrice', label: 'Renewable electricity price', unit: '$/kWh', value: 0.05, min: 0, max: 1, help: 'For the cost–energy–carbon trade-off.' },
      { key: 'renCarbon', label: 'Renewable emission factor', unit: 'kgCO₂/kWh', value: 0.03, min: 0, max: 0.3 },
      { key: 'capexPerSec', label: 'Capital premium per 1 % energy saved', unit: '%', value: 0.6, min: 0, max: 5, help: 'Extra membrane area, better pumps and recovery devices needed for each percent of specific-energy reduction.' },
      { key: 'volatility', label: 'Volatility of project value', unit: '%/y', value: 22, min: 1, max: 100, help: 'For the option to defer.' },
      { key: 'riskFree', label: 'Risk-free rate', unit: '%/y', value: 4, min: 0, max: 20 },
      { key: 'deferYears', label: 'Deferral window', unit: 'years', value: 5, min: 0.5, max: 15 },
    ] },
    { group: 'Sampling and tree resolution', tab: 'mesh', help: 'Numerical resolution of the stochastic analyses.', fields: [
      { key: 'nMC', label: 'Monte-Carlo samples', unit: '', value: 3000, min: 200, max: 50000, step: 100 },
      { key: 'optSteps', label: 'Binomial-tree steps', unit: '', value: 60, min: 4, max: 400, step: 1 },
    ] },
  ],

  presets: [
    { name: 'Large seawater RO, 100 000 m³/d', values: {} },
    { name: 'Small seawater RO, 10 000 m³/d, higher tariff', values: { capacity: 10000, sec: 3.8, elecPrice: 0.12, tariff: 1.9, land: 0.5, labCost: 120000, outfallLength: 400, constYears: 2,
      staff: [{ role: 'Plant manager', n: 1, salary: 90000 }, { role: 'Operators', n: 8, salary: 42000 }, { role: 'Maintenance technicians', n: 3, salary: 45000 }, { role: 'Laboratory and administration', n: 2, salary: 42000 }],
      replacements: [{ name: 'High-pressure pump overhaul', cost: 0.15, every: 8 }, { name: 'Filter media / UF modules', cost: 0.25, every: 7 }, { name: 'Control system and drives', cost: 0.3, every: 12 }] } },
    { name: 'Inland brackish RO, 30 000 m³/d, deep-well injection', values: { capacity: 30000, recovery: 80, sec: 0.95, designFlux: 26, memLife: 7, outfallLength: 0, brineOpt: 1, tariff: 0.62, land: 1, labCost: 200000, feedSalinity: 3.5,
      chemicals: [{ name: 'Sulphuric acid', basis: 'feed', dose: 40, price: 0.12 }, { name: 'Antiscalant', basis: 'feed', dose: 3.5, price: 2.6 }, { name: 'Caustic soda', basis: 'product', dose: 12, price: 0.45 }, { name: 'Sodium hypochlorite (as Cl₂)', basis: 'product', dose: 1.5, price: 0.5 }],
      staff: [{ role: 'Plant manager', n: 1, salary: 100000 }, { role: 'Operators', n: 10, salary: 45000 }, { role: 'Maintenance technicians', n: 4, salary: 48000 }, { role: 'Laboratory and administration', n: 3, salary: 45000 }],
      replacements: [{ name: 'Pump overhauls', cost: 0.25, every: 8 }, { name: 'Control system and drives', cost: 0.6, every: 12 }] } },
    { name: 'Solar-supplied seawater RO under an emissions cap', values: { selfShare: 45, genCapex: 850, genCF: 26, maxCarbon: 1.0, minProdPct: 90, shortfallPenalty: 0.4, availability: 88 } },
    { name: 'Hybrid RO–MED, 150 000 m³/d', values: { capacity: 150000, thermalCapacity: 50000, sec: 2.9, secThermal: 22, recovery: 42, tariff: 1.25, heatPrice: 0.01, land: 4, maxCarbon: 4 } },
    { name: 'Minimal-liquid-discharge plant with salt recovery', values: { capacity: 12000, recovery: 92, sec: 4.2, bcFeed: 45, solids: 70, zldPower: 1600, saltTpd: 55, saltPrice: 60, outfallLength: 0, brineOpt: 5, tariff: 3.2, elecPrice: 0.07, land: 1, labCost: 180000, maxCarbon: 4,
      staff: [{ role: 'Plant manager', n: 1, salary: 110000 }, { role: 'Operators', n: 12, salary: 48000 }, { role: 'Maintenance technicians', n: 6, salary: 50000 }, { role: 'Laboratory and administration', n: 3, salary: 46000 }],
      replacements: [{ name: 'Pump overhauls', cost: 0.2, every: 8 }, { name: 'Evaporator tube bundles', cost: 1.1, every: 12 }, { name: 'Control system and drives', cost: 0.4, every: 12 }] } },
  ],

  pull: ({ outputs }) => {
    const o = outputs || {}, P = [], q = o.plant?.productFlow ?? o.ro?.permeateFlow;
    if (q > 0) P.push({ key: 'capacity', value: q * 24, from: o.plant?.productFlow ? 'Plant simulation: product flow' : 'RO design: permeate flow' });
    const rec = o.plant?.recovery ?? o.ro?.recovery;
    if (rec > 0) P.push({ key: 'recovery', value: 100 * rec, from: o.plant?.recovery ? 'Plant simulation: recovery' : 'RO design: recovery' });
    const sec = o.plant?.secElec ?? o.pump?.sec ?? o.ro?.sec;
    if (sec > 0) P.push({ key: 'sec', value: sec, from: o.plant?.secElec ? 'Plant simulation: specific electricity' : o.pump?.sec ? 'Pumps and energy recovery: net specific energy' : 'RO design: specific energy' });
    const st = o.plant?.secThermal ?? (o.thermal?.secThermal && o.thermal?.distillate && q > 0 ? (o.thermal.secThermal * o.thermal.distillate) / q : undefined);
    if (st > 0) P.push({ key: 'secThermal', value: st, from: o.plant?.secThermal ? 'Plant simulation: specific heat' : 'Thermal desalination: heat per m³ of total product' });
    if (o.ro?.membraneArea > 0) P.push({ key: 'membraneArea', value: o.ro.membraneArea, from: 'RO design: membrane area' });
    if (o.thermal?.distillate > 0) P.push({ key: 'thermalCapacity', value: o.thermal.distillate * 24, from: 'Thermal desalination: distillate' });
    if (o.fouling?.membraneLife > 0) P.push({ key: 'memLife', value: clamp(o.fouling.membraneLife, 1, 15), from: 'Fouling monitor: membrane life' });
    if (o.fouling?.cleaningsPerYear >= 0) P.push({ key: 'cleanings', value: clamp(o.fouling.cleaningsPerYear, 0, 24), from: 'Fouling monitor: cleanings per year' });
    if (o.sea?.outfallLength > 0) P.push({ key: 'outfallLength', value: o.sea.outfallLength, from: 'Sea discharge: outfall length' });
    if (o.zld?.solids > 0) P.push({ key: 'solids', value: o.zld.solids, from: 'ZLD: solids produced' });
    if (o.zld?.power > 0) P.push({ key: 'zldPower', value: o.zld.power, from: 'ZLD: electrical power' });
    const salts = o.zld?.salts ? sum(Object.values(o.zld.salts).map((x) => +x || 0)) : 0;
    if (salts > 0) P.push({ key: 'saltTpd', value: salts, from: 'ZLD: recovered salts' });
    if (o.chem?.antiscalantDose > 0 || o.chem?.acidDose > 0) P.push({ key: 'chemicals', value: CHEMS.map((c) => ({ ...c, dose: c.name === 'Antiscalant' && o.chem.antiscalantDose > 0 ? o.chem.antiscalantDose : c.name === 'Sulphuric acid' && o.chem.acidDose > 0 ? o.chem.acidDose : c.dose })), from: 'Brine chemistry: antiscalant and acid dose' });
    if (o.ro?.streams?.feed?.tds > 0) P.push({ key: 'feedSalinity', value: clamp(o.ro.streams.feed.tds / 1000 / (1 + 0.0007 * (o.ro.streams.feed.tds / 1000)), 0.5, 120), from: 'RO design: feed salinity' });
    if (o.pump?.erdEfficiency >= 0 && o.pump?.erdEfficiency <= 1) P.push({ key: 'etaErd', value: clamp(100 * o.pump.erdEfficiency, 0, 99), from: 'Pumps and energy recovery: device efficiency' });
    return P;
  },
  site: (site) => {
    const d = site?.data || {}, P = [];
    if (d.inflation !== undefined && d.inflation !== null) P.push({ key: 'inflation', value: d.inflation, from: 'Site inflation rate' });
    if (d.lendingRate > 0) P.push({ key: 'loanRate', value: d.lendingRate, from: 'Site lending rate' });
    if (d.electricityPrice > 0) P.push({ key: 'elecPrice', value: d.electricityPrice, from: 'Site electricity price' });
    if (d.gridCarbon >= 0 && d.gridCarbon !== null && d.gridCarbon !== undefined) P.push({ key: 'gridCarbon', value: d.gridCarbon, from: 'Site grid emission factor' });
    if (d.fxPerUSD > 0) P.push({ key: 'fx', value: d.fxPerUSD, from: 'Site exchange rate' });
    if (d.currency) P.push({ key: 'currency', value: String(d.currency).slice(0, 8), from: 'Site currency' });
    return P;
  },

  run(v, ctx) {
    const K = prepare(v), E = evaluate(v, {}, true, K), { c, f, an, cash } = E, W = [], M = 1e6;
    const ccy = String(v.currency || 'USD').slice(0, 8), fx = v.fx > 0 ? v.fx : 1;
    // ---- finance indicators
    const pIrr = irr(cash.proj), eIrr = irr(cash.eq), pMirr = mirr(cash.proj, c.rl, f.dn), pbS = payback(cash.proj), pbD = payback(cash.proj, f.dn);
    const ds = cash.rows.filter((r) => Number.isFinite(r.dscr)).map((r) => r.dscr), minDscr = ds.length ? Math.min(...ds) : null, avgDscr = ds.length ? mean(ds) : null;
    const npvAt = (tar) => cashFlow(c, f, v, tar, true).npv;
    let breakEven = null;
    if (npvAt(0) < 0 && npvAt(60) > 0) breakEven = brent((x) => npvAt(x), 0, 60, 1e-9);
    const pvRev = sum(cash.rows.map((r) => r.revenue / (1 + f.dn) ** r.t)), bcr = pvRev / cash.pvCost;
    const ci = c.tCO2 * 1000 / c.prod, wacc = c.debtFrac * c.rl * (1 - v.taxRate / 100) + (1 - c.debtFrac) * (v.equityRate / 100);
    ctx?.progress?.(0.25, 'Cash flow built');
    // ---- sensitivity: tornado and spider
    const unc = DRIVERS.map((d, i) => ({ ...d, lo: num(v.uncert?.[i]?.lo, d.lo), hi: num(v.uncert?.[i]?.hi, d.hi) })), s = v.sensPct / 100;
    const L = (m) => evaluate(v, m, false, K).lcow;
    const tornado = unc.map((d) => ({ name: d.name, lo: L({ [d.k]: 1 - s }) - an.lcow, hi: L({ [d.k]: 1 + s }) - an.lcow })).map((t) => ({ ...t, swing: Math.abs(t.hi - t.lo) })).sort((a, b) => b.swing - a.swing);
    const sp = linspace(-30, 30, 7), spider = unc.slice(0, 7).map((d) => ({ name: d.name, x: sp, y: sp.map((p) => L({ [d.k]: 1 + p / 100 })) }));
    // ---- scenarios (each driver half-way to its favourable / adverse bound)
    const scen = (dir) => Object.fromEntries(unc.map((d) => [d.k, 1 + (0.5 * (dir * d.good > 0 ? d.hi : d.lo)) / 100]));
    const scenarios = [['Favourable', scen(1)], ['Expected', {}], ['Adverse', scen(-1)]].map(([name, m]) => { const e = evaluate(v, m, true, K); return { name, lcow: e.lcow, tci: e.c.tci, opex: e.c.opex, npv: e.cash.npv, irr: irr(e.cash.proj), prod: e.c.prod }; });
    ctx?.progress?.(0.45, 'Sensitivities done');
    // ---- Monte Carlo
    const nMC = Math.max(200, Math.round(v.nMC)), iE = DRIVERS.findIndex((d) => d.k === 'elec'), iC = DRIVERS.findIndex((d) => d.k === 'chem');
    const draws = sampleMultipliers(nMC, unc, v.mcDist, Math.abs(v.mcCorr) > 1e-6 ? [iE, iC, clamp(v.mcCorr, -0.99, 0.99)] : null, v.mcSeed), mcL = new Array(nMC), mcN = new Array(nMC);
    let nMiss = 0;
    for (let k = 0; k < nMC; k++) { const m = Object.fromEntries(DRIVERS.map((d, j) => [d.k, draws[k][j]])), cc = buildCosts(v, m, K), ff = finance(v, m); mcL[k] = lcowAnnual(cc, ff, v).lcow; mcN[k] = cashFlow(cc, ff, v, v.tariff, true).npv; if (cc.shortfall > 0) nMiss++; }
    const pMiss = nMiss / nMC;
    // ---- response-surface surrogate of the cost model: larger sample and variance-based sensitivity
    const sg = surrogateTEA(v, K, unc, { dist: v.mcDist, corr: Math.abs(v.mcCorr) > 1e-6 ? [iE, iC, clamp(v.mcCorr, -0.99, 0.99)] : null, seed: v.mcSeed });
    const sorted = [...mcL].sort((a, b) => a - b), p10 = quantile(mcL, 0.1), p50 = quantile(mcL, 0.5), p90 = quantile(mcL, 0.9), pExceed = mcL.filter((x) => x > v.targetLcow).length / nMC, pLoss = mcN.filter((x) => x < 0).length / nMC;
    const hist = histogram(mcL, 30), kq = Math.max(1, Math.floor(nMC / 200)), cdfX = sorted.filter((_, i) => i % kq === 0), cdfY = cdfX.map((_, i) => Math.min(1, (i * kq + 1) / nMC));
    ctx?.progress?.(0.7, 'Monte-Carlo done');
    // ---- sweeps and map
    const cfs = linspace(50, 100, 11), cfSweep = cfs.map((a) => evaluate({ ...v, availability: a }, {}, false, K).lcow);
    const sizes = logspace(Math.log10(Math.max(500, v.capacity / 20)), Math.log10(Math.min(2e6, v.capacity * 8)), 13), sizeRes = sizes.map((q) => { const [vv, mm] = atCapacity(v, q); const e = evaluate(vv, mm, false, K); return [e.lcow, e.c.tci / q]; });
    const eps = linspace(0.02, 0.2, 13), drs = linspace(2, 14, 11), field = drs.map((d) => eps.map((p) => evaluate({ ...v, tou: false, elecPrice: p, discNominal: d }, {}, false, K).lcow));
    const lives = [15, 20, 25, 30, 40], lifeSweep = lives.map((n) => evaluate({ ...v, life: n }, {}, false, K).lcow);
    // ---- brine-management comparison
    const brineCmp = c.brineOpts.map((b) => { const capA = an.crf * b.capex * (1 + (v.pEng + v.pProc + v.pComm + v.pOwner) / 100) * (1 + v.pCont / 100), op = b.opex * c.brine * 365 * c.cf, en = b.energy * c.brine * 365 * c.cf * c.elecP; return { name: b.name, capex: b.capex, add: (capA + op + en) / c.prod, co2: (b.energy * c.brine * 365 * c.cf * c.gridC) / 1000 }; });
    // ---- cost–energy–carbon trade-off
    const designs = [['Energy-efficient design (−10 % energy)', 0.9], ['Base design', 1], ['Low-capital design (+10 % energy)', 1.1]], shares = [0, 0.25, 0.5, 0.75, 1], pts = [];
    const trade = designs.map(([name, sf]) => ({ name, mode: 'both', x: [], y: [], sf }));
    trade.forEach((t) => shares.forEach((sh) => { const e = evaluate({ ...v, tou: false, selfShare: 0, elecPrice: (1 - sh) * c.elecP + sh * v.renPrice, gridCarbon: (1 - sh) * c.gridC + sh * v.renCarbon }, { sec: t.sf, capex: 1 + (v.capexPerSec * (1 - t.sf) * 100) / 100 }, false, K); const cI = (e.c.tCO2 * 1000) / e.c.prod; t.x.push(cI); t.y.push(e.lcow); pts.push({ design: t.name, share: sh, lcow: e.lcow, ci: cI, sec: e.c.sec }); }));
    pts.forEach((p) => { p.pareto = !pts.some((q) => q !== p && q.lcow <= p.lcow && q.ci <= p.ci && q.sec <= p.sec && (q.lcow < p.lcow || q.ci < p.ci || q.sec < p.sec)); });
    // ---- levelised cost of energy, exergy costing, life-cycle assessment
    const gen = c.gen, ex = exergoeconomics(c, an, v), lca = lcaTea(c, f, v, an), genKW = (c.self * c.elecKWh) / gen.energy;
    // ---- constraints: minimum production and emissions cap (renewable share needed to comply, and its cost)
    const renAt = (sh) => evaluate({ ...v, tou: false, selfShare: 0, elecPrice: (1 - sh) * c.elecP + sh * v.renPrice, gridCarbon: (1 - sh) * c.gridC + sh * v.renCarbon }, {}, false, K), ciOf = (e) => (e.c.tCO2 * 1000) / e.c.prod;
    const capC = v.maxCarbon ?? 0, r0 = renAt(0), r1 = renAt(1), ci1 = ciOf(r1), abate = ci - ci1 > 1e-9 ? (1000 * (r1.lcow - r0.lcow)) / (ci - ci1) : null;
    const emis = { cap: capC, active: capC > 0, ok: !(capC > 0) || ci <= capC, share: 0, lcow: an.lcow, feasible: true, best: null };
    if (emis.active && !emis.ok) { emis.feasible = ci1 <= capC; emis.share = emis.feasible ? clamp((ci - capC) / (ci - ci1), 0, 1) : 1; emis.lcow = renAt(emis.share).lcow; }
    if (emis.active) emis.best = pts.filter((q) => q.ci <= capC).reduce((b, q) => (b === null || q.lcow < b.lcow ? q : b), null);
    const needAvail = c.minProd > 0 ? (100 * c.minProd) / (c.cap * 365 * c.tou.prodFactor) : 0;
    // ---- option to defer
    const Vop = cash.npv + (c.tci - c.grant), strike = c.tci - c.grant, yieldQ = clamp(1 / f.N + 0.02, 0, 0.3);
    const optVal = Vop > 0 ? binomialOption({ V: Vop, K: strike, r: v.riskFree / 100, sigma: v.volatility / 100, T: v.deferYears, steps: v.optSteps, q: yieldQ, american: true }) : 0, flex = optVal - Math.max(cash.npv, 0);

    // ---- warnings
    if (an.lcow > v.targetLcow) W.push({ level: 'warn', msg: `Levelised cost ${fmt(an.lcow, 3)} $/m³ exceeds the ${v.targetLcow} $/m³ ceiling.` });
    if (cash.npv < 0) W.push({ level: 'bad', msg: `Net present value is negative (${fmt(cash.npv / M, 3)} M$) at a tariff of ${v.tariff} $/m³${breakEven ? ` — break-even tariff is ${fmt(breakEven, 3)} $/m³` : ''}.` });
    if (minDscr !== null && minDscr < v.minDscr) W.push({ level: minDscr < 1 ? 'bad' : 'warn', msg: `Minimum debt-service cover ratio ${fmt(minDscr, 3)} is below the ${v.minDscr} covenant — reduce gearing, lengthen the tenor or raise the tariff.` });
    if (!Number.isFinite(pIrr)) W.push({ level: 'warn', msg: 'The project cash flow never turns positive, so no internal rate of return exists.' });
    if (pExceed > 0.25) W.push({ level: 'warn', msg: `There is a ${fmt(100 * pExceed, 3)} % chance that the water cost exceeds ${v.targetLcow} $/m³.` });
    if (c.sat > 1) W.push({ level: 'info', msg: `Capacity is ${fmt(c.sat, 3)} × the largest single-plant size: cost is scaled as parallel plants beyond ${fmt(v.maxScale, 4)} m³/d.` });
    if (v.tou) W.push({ level: 'info', msg: `Time-of-use tariff: effective price ${fmt(c.tou.price, 3)} $/kWh (flat running would pay ${fmt(c.tou.flat, 3)}), production factor ${fmt(100 * c.tou.prodFactor, 4)} %.` });
    if (Math.abs(c.escal - 1) > 0.002) W.push({ level: 'info', msg: `Equipment costs escalated by ${fmt(100 * (c.escal - 1), 3)} % from ${v.baseYear} to ${v.analysisYear} with the cost index.` });
    if (c.shortfall > 0) W.push({ level: 'warn', msg: `Annual production ${fmt(c.prod / M, 4)} Mm³ is below the contractual minimum of ${fmt(c.minProd / M, 4)} Mm³ — a capacity factor of at least ${fmt(needAvail, 3)} % is needed${c.o.shortfall > 0 ? `; the shortfall costs ${fmt(c.o.shortfall / 1000, 3)} k$/y` : ''}.` });
    else if (pMiss > 0.1) W.push({ level: 'info', msg: `There is a ${fmt(100 * pMiss, 3)} % chance of missing the minimum production of ${fmt(c.minProd / M, 4)} Mm³/y under the capacity-factor uncertainty.` });
    if (emis.active && !emis.ok) W.push({ level: 'warn', msg: emis.feasible ? `Carbon intensity ${fmt(ci, 3)} kgCO₂/m³ exceeds the cap of ${capC} — a renewable share of ${fmt(100 * emis.share, 3)} % of the electricity would comply, at a water cost of ${fmt(emis.lcow, 3)} $/m³.` : `Carbon intensity ${fmt(ci, 3)} kgCO₂/m³ exceeds the cap of ${capC} and even fully renewable electricity (${fmt(ci1, 3)} kgCO₂/m³) does not comply — reduce the thermal energy or its emission factor.` });
    if (!ex.consistent) W.push({ level: 'warn', msg: `Exergy analysis: the entered specific energy implies a membrane feed pressure of ${fmt(ex.pBar, 3)} bar, below the least work of separation at this salinity and recovery — check the specific energy, feed salinity and pumping share.` });
    if (c.self > 0) W.push({ level: 'info', msg: `${fmt(100 * c.self, 3)} % of the electricity is self-supplied at a levelised cost of ${fmt(gen.lcoe, 3)} $/kWh (${fmt(genKW, 4)} kW of generation); effective price ${fmt(c.elecP, 3)} $/kWh.` });
    W.push({ level: 'info', msg: 'Cost data are indicative screening values (about ±25–30 % on capital). Calibrate against recent regional projects before using the result for a decision.' });
    const share = (x) => (100 * x) / an.lcow, top = Object.entries(an.parts).sort((a, b) => b[1] - a[1]);

    const out = { lcow: an.lcow, capex: c.tci, opex: c.opex, npv: cash.npv, carbonIntensity: ci, lcowNominal: cash.lcowNom, lcowDCF: cash.lcowReal, specificCapex: c.tci / c.cap, annualProduction: c.prod, tCO2: c.tCO2, p10, p50, p90, opexPerM3: c.opex / c.prod, lcowLocal: an.lcow * fx, currency: ccy, deferralValue: optVal,
      lcoe: gen.lcoe, exergyEfficiency: ex.eff, exergyCostOfProduct: ex.cProduct, lifeCycleCarbonIntensity: lca.ci, surrogateR2: sg.r2, surrogateP90: sg.p90, emissionsCapMet: emis.ok, minProductionMet: !(c.shortfall > 0), probMissProduction: pMiss };
    if (Number.isFinite(pIrr)) out.irr = pIrr;
    if (Number.isFinite(pbS)) out.payback = pbS;
    if (breakEven !== null) out.breakEvenTariff = breakEven;
    const capRows = [...c.items.filter((i) => i.installed > 0).map((i) => [i.name, i.basis, i.n, i.equip / M, i.installed / M, (100 * i.installed) / c.tci]),
      c.bo.capex > 0 ? [`Brine route: ${c.bo.name}`, 'brine', 1, c.bo.capex / M, c.bo.capex / M, (100 * c.bo.capex) / c.tci] : null,
      ['Direct cost', null, null, null, c.direct / M, (100 * c.direct) / c.tci], ['Engineering and design', null, null, null, c.indirect.engineering / M, (100 * c.indirect.engineering) / c.tci], ['Procurement and construction management', null, null, null, c.indirect.procurement / M, (100 * c.indirect.procurement) / c.tci],
      ['Commissioning and start-up', null, null, null, c.indirect.commissioning / M, (100 * c.indirect.commissioning) / c.tci], ["Owner's costs", null, null, null, c.indirect.owner / M, (100 * c.indirect.owner) / c.tci], ['Contingency', null, null, null, c.contingency / M, (100 * c.contingency) / c.tci],
      ['Land', null, null, null, c.land / M, (100 * c.land) / c.tci], ['Working capital', null, null, null, c.wc / M, (100 * c.wc) / c.tci], ['Interest during construction', null, null, null, c.idc / M, (100 * c.idc) / c.tci], ['Total capital investment', null, null, null, c.tci / M, 100]].filter(Boolean);
    const o = c.o, opRows = [['Electricity', o.elec], ['Thermal energy', o.thermal], ['Membrane replacement', o.mem], ['Cartridge filters', o.cart], ['Membrane cleaning', o.cip], ['Chemicals', o.chem], ['Labour', o.labour], ['Overheads', o.overhead], ['Maintenance and spares', o.maint], ['Insurance', o.ins], ['Laboratory and monitoring', o.lab], ['Brine management', o.brine], ['Sludge disposal', o.sludge], ['Crystalliser solids disposal', o.solids], ['Carbon cost', o.carbon], ['Environmental charges', o.env], ['Take-or-pay shortfall penalty', o.shortfall]];
    const yrs = cash.rows.map((r) => r.t);
    let cumU = cash.proj[0], cumD = cash.proj[0];
    const cumUnd = [cumU, ...cash.rows.map((r) => (cumU += r.cfProj))], cumDisc = [cumD, ...cash.rows.map((r) => (cumD += r.cfProj / (1 + f.dn) ** r.t))];

    return {
      summary: `Levelised cost of water ${fmt(an.lcow, 3)} $/m³${fx !== 1 ? ` (${fmt(an.lcow * fx, 3)} ${ccy}/m³)` : ''} for ${fmt(c.cap, 4)} m³/d: capital ${fmt(c.tci / M, 4)} M$ (${fmt(c.tci / c.cap, 4)} $ per m³/d), operating cost ${fmt(c.opex / M, 3)} M$/y. ${top[0][0]} is the largest component (${fmt(share(top[0][1]), 2)} %), then ${top[1][0].toLowerCase()} (${fmt(share(top[1][1]), 2)} %). Monte-Carlo P10–P90: ${fmt(p10, 3)}–${fmt(p90, 3)} $/m³.`,
      warnings: W,
      kpis: [
        { label: 'Levelised cost of water (real)', value: an.lcow, unit: '$/m³', status: an.lcow > v.targetLcow ? 'warn' : 'ok', help: 'Annualised capital + operating cost per m³, constant analysis-year dollars' },
        { label: 'LCOW from cash flow (real)', value: cash.lcowReal, unit: '$/m³', help: 'Discounted cost ÷ discounted production, including ramp-up and real escalation' },
        { label: 'LCOW (nominal)', value: cash.lcowNom, unit: '$/m³', help: 'Constant money-of-the-day price with the same present value' },
        { label: `LCOW in ${ccy}`, value: an.lcow * fx, unit: `${ccy}/m³` },
        { label: 'Total capital investment', value: c.tci / M, unit: 'M$' },
        { label: 'Specific capital cost', value: c.tci / c.cap, unit: '$ per m³/d' },
        { label: 'Operating cost', value: c.opex / M, unit: 'M$/y' },
        { label: 'Operating cost per m³', value: c.opex / c.prod, unit: '$/m³' },
        { label: 'Net present value', value: cash.npv / M, unit: 'M$', status: cash.npv < 0 ? 'bad' : 'ok', help: 'After-tax project cash flow at the nominal discount rate' },
        { label: 'Project IRR', value: Number.isFinite(pIrr) ? 100 * pIrr : 'none', unit: Number.isFinite(pIrr) ? '%' : '', status: Number.isFinite(pIrr) && pIrr >= f.dn ? 'ok' : 'warn' },
        { label: 'Equity IRR', value: Number.isFinite(eIrr) ? 100 * eIrr : 'none', unit: Number.isFinite(eIrr) ? '%' : '' },
        { label: 'Simple payback', value: Number.isFinite(pbS) ? pbS : `> ${f.N}`, unit: 'years' },
        { label: 'Discounted payback', value: Number.isFinite(pbD) ? pbD : `> ${f.N}`, unit: 'years' },
        { label: 'Minimum DSCR', value: minDscr ?? 'no debt', unit: minDscr === null ? '' : '–', status: minDscr !== null && minDscr < 1 ? 'bad' : minDscr !== null && minDscr < v.minDscr ? 'warn' : 'ok' },
        { label: 'Break-even tariff', value: breakEven ?? 'n/a', unit: breakEven === null ? '' : '$/m³', help: 'Tariff at which the after-tax net present value is zero' },
        { label: 'Carbon intensity', value: ci, unit: 'kgCO₂/m³' },
        { label: 'Monte-Carlo P50 / P90', value: `${fmt(p50, 3)} / ${fmt(p90, 3)}`, unit: '$/m³' },
        { label: 'Chance of exceeding the ceiling', value: 100 * pExceed, unit: '%', status: pExceed > 0.25 ? 'warn' : 'ok' },
        { label: 'LCOE of on-site generation', value: gen.lcoe, unit: '$/kWh', status: 'ok', help: `(CRF·CAPEX + O&M) ÷ annual energy + fuel, real terms; grid price ${fmt(c.tou.price, 3)} $/kWh; ${fmt(100 * c.self, 3)} % self-supplied` },
        { label: 'Exergy (second-law) efficiency', value: 100 * ex.eff, unit: '%', status: ex.consistent ? 'ok' : 'warn', help: `Least work of separation ${fmt(ex.wmin, 3)} kWh/m³ ÷ fuel exergy ${fmt(ex.fuel, 3)} kWh/m³` },
        { label: 'Exergy cost of product water', value: ex.cProduct, unit: '$/kWh', help: 'Exergoeconomic unit cost: all fuel and capital cost rates carried by the separation exergy of the product' },
        { label: 'Life-cycle carbon intensity', value: lca.ci, unit: 'kgCO₂e/m³', status: emis.active && lca.ci > capC ? 'warn' : 'ok', help: `Operation ${fmt(lca.opCi, 3)} + embodied ${fmt(lca.embodied, 3)} kgCO₂e/m³ (construction, membranes, chemicals, end of life)` },
        { label: 'Surrogate P50 / P90', value: `${fmt(sg.p50, 3)} / ${fmt(sg.p90, 3)}`, unit: '$/m³', status: sg.r2 > 0.99 ? 'ok' : 'warn', help: `Quadratic response surface of the cost model, ${sg.nMC} samples; hold-out R² ${fmt(sg.r2, 5)}` },
        { label: 'Emissions cap', value: !emis.active ? 'none set' : emis.ok ? 'met' : emis.feasible ? `needs ${fmt(100 * emis.share, 3)} % renewable` : 'cannot be met', unit: '', status: emis.ok ? 'ok' : 'warn' },
        { label: 'Minimum production', value: !(c.minProd > 0) ? 'none set' : c.shortfall > 0 ? `short by ${fmt(c.shortfall / M, 3)} Mm³/y` : 'met', unit: '', status: c.shortfall > 0 ? 'warn' : 'ok', help: `Chance of missing it under uncertainty: ${fmt(100 * pMiss, 3)} %` },
      ],
      recommendations: [
        `The water cost is most sensitive to ${tornado[0].name.toLowerCase()} and ${tornado[1].name.toLowerCase()}: firm these up first (quotations, tariff agreement, financing terms).`,
        share(an.parts.Electricity) > 30 ? `Electricity is ${fmt(share(an.parts.Electricity), 2)} % of the water cost — each 0.1 kWh/m³ saved is worth ${fmt((0.1 * c.elecP * c.prod) / 1000, 3)} k$/y. Check energy recovery and pump efficiency in suite 12.` : null,
        gen.lcoe < 0.9 * c.tou.price && c.self < 0.5 ? `On-site generation at these inputs has a levelised cost of ${fmt(gen.lcoe, 3)} $/kWh against ${fmt(c.tou.price, 3)} $/kWh from the grid — raise the self-supplied share to cut the water cost.` : null,
        ex.rows.length ? `Exergoeconomics: the largest cost of exergy destruction is in "${[...ex.rows].sort((a, b) => b.cD - a.cD)[0].name.toLowerCase()}" (${fmt([...ex.rows].sort((a, b) => b.cD - a.cD)[0].cD, 3)} $/m³) — that is where efficiency investment pays first.` : null,
        cash.npv < 0 && breakEven ? `Raise the tariff to at least ${fmt(breakEven, 3)} $/m³, or secure a capital grant or cheaper debt, for the project to earn its cost of capital.` : null,
        flex > 0.02 * strike && cash.npv > 0 ? `Waiting has option value (${fmt(flex / M, 3)} M$ above investing now) — consider phasing or deferring if demand or tariffs are uncertain.` : null,
        c.cf < 0.85 ? 'Capacity factor is below 85 %: fixed costs are spread over less water. Storage or demand contracts that raise utilisation cut the unit cost directly.' : null,
        pts.some((p) => p.pareto && p.share > 0 && p.lcow <= an.lcow * 1.001) ? 'Renewable supply at the entered price lowers both cost and carbon — it dominates grid-only supply.' : null,
        'Fit the cost-correlation multiplier and scale exponent to reference projects on the Calibrate tab before relying on absolute values.',
      ].filter(Boolean),
      plots: [
        { type: 'bar', title: 'Capital cost build-up', ylabel: 'M$', categories: [...c.items.filter((i) => i.installed > 0).map((i) => i.name), 'Indirect costs', 'Contingency', 'Land and working capital', 'Interest during construction'], series: [{ name: 'M$', values: [...c.items.filter((i) => i.installed > 0).map((i) => i.installed / M), c.indTot / M, c.contingency / M, (c.land + c.wc) / M, c.idc / M] }] },
        { type: 'bar', title: 'Levelised cost of water by component', ylabel: '$/m³', categories: Object.keys(an.parts), series: [{ name: '$/m³', values: Object.values(an.parts) }] },
        { type: 'line', title: 'Project cash flow', xlabel: 'Year of operation', ylabel: 'M$', series: [{ name: 'After-tax cash flow', x: yrs, y: cash.rows.map((r) => r.cfProj / M), mode: 'step' }, { name: 'Cumulative', x: [0, ...yrs], y: cumUnd.map((x) => x / M) }, { name: 'Cumulative, discounted', x: [0, ...yrs], y: cumDisc.map((x) => x / M) }, { name: 'Debt outstanding', x: [0, ...yrs], y: [cash.D, ...cash.rows.map((r) => r.balance)].map((x) => x / M), dash: true }], hlines: [{ y: 0, label: '' }] },
        { type: 'bar', title: `Tornado: change in LCOW for ±${v.sensPct} % on each driver`, ylabel: '$/m³', categories: tornado.map((t) => t.name), series: [{ name: `−${v.sensPct} %`, values: tornado.map((t) => t.lo) }, { name: `+${v.sensPct} %`, values: tornado.map((t) => t.hi) }] },
        { type: 'line', title: 'Spider plot', xlabel: 'Change in driver (%)', ylabel: 'LCOW ($/m³)', series: spider.map((q) => ({ ...q, mode: 'both' })) },
        { type: 'line', title: `Monte-Carlo distribution of LCOW (${nMC} samples)`, xlabel: 'LCOW ($/m³)', ylabel: 'Frequency (%)', zeroY: true, series: [{ name: 'Histogram', x: hist.centers, y: hist.counts.map((n) => (100 * n) / nMC), mode: 'step' }], vlines: [{ x: p10, label: 'P10' }, { x: p50, label: 'P50' }, { x: p90, label: 'P90' }] },
        { type: 'line', title: 'Cumulative probability of LCOW', xlabel: 'LCOW ($/m³)', ylabel: 'Probability of not exceeding', ymin: 0, ymax: 1, series: [{ name: 'Cumulative distribution', x: cdfX, y: cdfY }], vlines: [{ x: v.targetLcow, label: 'ceiling' }], hlines: [{ y: 0.5, label: 'P50' }, { y: 0.9, label: 'P90' }] },
        { type: 'line', title: 'Economies of scale', xlabel: 'Plant capacity (m³/d)', ylabel: 'LCOW ($/m³) · specific capital (k$ per m³/d)', logx: true, series: [{ name: 'LCOW ($/m³)', x: sizes, y: sizeRes.map((r) => r[0]), mode: 'both' }, { name: 'Specific capital (k$ per m³/d)', x: sizes, y: sizeRes.map((r) => r[1] / 1000), mode: 'both' }], vlines: [{ x: v.capacity, label: 'this plant' }], note: 'Staffing is scaled with capacity^0.3 and land with capacity^0.6 for this sweep.' },
        { type: 'line', title: 'Effect of capacity factor and plant life', xlabel: 'Capacity factor (%) · plant life (years × 2)', ylabel: 'LCOW ($/m³)', series: [{ name: 'Capacity factor', x: cfs, y: cfSweep, mode: 'both' }, { name: 'Plant life (x = years × 2)', x: lives.map((n) => 2 * n), y: lifeSweep, mode: 'both' }] },
        { type: 'field', title: 'LCOW map: electricity price × discount rate', xlabel: 'Electricity price ($/kWh)', ylabel: 'Nominal discount rate (%/y)', zlabel: 'LCOW', zunit: '$/m³', x: eps, y: drs, z: field, cmap: 'viridis', contours: 8, markers: [{ x: clamp(c.elecP, 0.02, 0.2), y: clamp(v.discNominal, 2, 14), label: 'this case' }] },
        { type: 'line', title: 'Cost – carbon trade-off (renewable share 0 → 100 %)', xlabel: 'Carbon intensity (kgCO₂/m³)', ylabel: 'LCOW ($/m³)', series: trade.map(({ name, mode, x, y }) => ({ name, mode, x, y })), note: 'Each line is one design; points step the renewable share of electricity from 0 to 100 % in quarters.' },
        { type: 'bar', title: 'Exergy destruction and exergoeconomic cost rates by subsystem', ylabel: 'kWh/m³ · $/m³ × 10', categories: ex.rows.map((q) => q.name), series: [{ name: 'Exergy destruction (kWh/m³)', values: ex.rows.map((q) => q.D) }, { name: 'Cost of exergy destruction ($/m³ × 10)', values: ex.rows.map((q) => 10 * q.cD) }, { name: 'Capital and O&M cost rate ($/m³ × 10)', values: ex.rows.map((q) => 10 * q.z) }] },
        { type: 'bar', title: 'Life-cycle greenhouse-gas emissions by stage', ylabel: 'kgCO₂e per m³', categories: Object.keys(lca.stages), series: [{ name: 'kgCO₂e/m³', values: Object.values(lca.stages) }] },
        { type: 'bar', title: 'Variance-based sensitivity of LCOW (first-order indices from the surrogate)', ylabel: 'Share of variance (–)', categories: unc.map((d) => d.name), series: [{ name: 'S₁', values: sg.S1 }] },
        { type: 'line', title: 'Surrogate parity on hold-out runs of the cost model', xlabel: 'Cost model LCOW ($/m³)', ylabel: 'Surrogate LCOW ($/m³)', series: [{ name: 'Hold-out runs', x: sg.parity.model, y: sg.parity.sur, mode: 'points' }, { name: '1 : 1', x: [Math.min(...sg.parity.model), Math.max(...sg.parity.model)], y: [Math.min(...sg.parity.model), Math.max(...sg.parity.model)], dash: true }] },
        { type: 'bar', title: 'Brine-management options: addition to the water cost', ylabel: '$/m³ of product', categories: brineCmp.slice(0, 5).map((b) => b.name), series: [{ name: '$/m³', values: brineCmp.slice(0, 5).map((b) => b.add) }] },
      ],
      tables: [
        { title: 'Capital cost build-up', columns: ['Item', 'Scales with', 'Exponent', 'Equipment (M$)', 'Installed (M$)', 'Share of total (%)'], rows: capRows, note: `Analysis-year dollars. Index escalation × ${fmt(c.escal, 4)}, location × ${v.locFactor}, correlation multiplier × ${v.kCapex}${c.learn < 1 ? `, membrane learning × ${fmt(c.learn, 3)}` : ''}. Indicative costs.` },
        { title: 'Operating cost (first full year, analysis-year prices)', columns: ['Item', 'k$/y', '$/m³', 'Share of OPEX (%)'], rows: [...opRows.filter((r) => r[1] > 0).map(([n, x]) => [n, x / 1000, x / c.prod, (100 * x) / c.opex]), ['Total operating cost', c.opex / 1000, c.opex / c.prod, 100]],
          note: `Annual production ${fmt(c.prod / M, 4)} Mm³; electricity ${fmt(c.elecKWh / M, 4)} GWh/y; emissions ${fmt(c.tCO2, 4)} tCO₂/y.` },
        { title: 'Chemicals and staffing', columns: ['Item', 'Basis', 'Dose (mg/L) / headcount', 'Unit price ($/kg or $/y)', 'Consumption (kg/d)', 'Cost (k$/y)'], rows: [...c.chemRows.map((r) => [r.name, r.basis, r.dose, r.price, r.kgd, r.cost / 1000]), ...c.staff.map((r) => [r.role, 'staff', r.n, r.salary, null, r.cost / 1000])] },
        { title: 'Financial indicators', columns: ['Indicator', 'Value', 'Unit'], rows: [
          ['Levelised cost of water, annualised (real)', an.lcow, '$/m³'], ['Levelised cost of water, cash-flow based (real)', cash.lcowReal, '$/m³'], ['Levelised cost of water (nominal)', cash.lcowNom, '$/m³'], [`Levelised cost of water in ${ccy}`, an.lcow * fx, `${ccy}/m³`],
          ['Capital-recovery factor', an.crf, '1/y'], ['Fixed-charge rate (capital + insurance + maintenance)', 100 * an.fcr, '%/y'], ['Annualised capital', an.capital / M, 'M$/y'], ['Real discount rate (Fisher)', 100 * f.dr, '%/y'], ['Nominal discount rate', 100 * f.dn, '%/y'], ['Weighted cost of capital implied by the financing', 100 * wacc, '%/y'],
          ['Debt at start of operation', cash.D / M, 'M$'], ['Equity', (c.tci - c.grant - cash.D) / M, 'M$'], ['Capital grant', c.grant / M, 'M$'], ['Net present value (project, after tax)', cash.npv / M, 'M$'], ['Net present value to equity', cash.npvEq / M, 'M$'],
          ['Project IRR', Number.isFinite(pIrr) ? 100 * pIrr : null, '%'], ['Equity IRR', Number.isFinite(eIrr) ? 100 * eIrr : null, '%'], ['Modified IRR', Number.isFinite(pMirr) ? 100 * pMirr : null, '%'], ['Simple payback', Number.isFinite(pbS) ? pbS : null, 'years'], ['Discounted payback', Number.isFinite(pbD) ? pbD : null, 'years'],
          ['Minimum debt-service cover ratio', minDscr, '–'], ['Average debt-service cover ratio', avgDscr, '–'], ['Break-even tariff', breakEven, '$/m³'], ['Benefit–cost ratio', bcr, '–'], ['Carbon emissions', c.tCO2, 'tCO₂/y'], ['Carbon intensity', ci, 'kgCO₂/m³'], ['Carbon cost', o.carbon / 1000, 'k$/y'],
          ['Value of the option to defer', optVal / M, 'M$'], ['Flexibility value above investing now', flex / M, 'M$'], ['Probability of negative NPV', 100 * pLoss, '%']] },
        { title: 'Year-by-year cash flow (nominal, M$)', columns: ['Year', 'Production (Mm³)', 'Revenue', 'Energy', 'Chemicals', 'Membranes', 'Labour', 'Other', 'Replacements', 'EBITDA', 'Depreciation', 'Interest', 'Principal', 'Tax', 'Project cash flow', 'Equity cash flow', 'Debt balance', 'DSCR'],
          rows: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cash.proj[0] / M, cash.eq[0] / M, cash.D / M, null], ...cash.rows.map((r) => [r.t, r.Q / M, r.revenue / M, r.og.energy / M, r.og.chem / M, r.og.mem / M, r.og.labour / M, r.og.other / M, r.replace / M, r.ebitda / M, r.dep / M, r.interest / M, r.principal / M, r.tax / M, r.cfProj / M, r.cfEq / M, r.balance / M, Number.isFinite(r.dscr) ? r.dscr : null])],
          note: 'Year 0 is the start of operation: total capital investment including interest during construction. The final year includes salvage value and recovered working capital.' },
        { title: 'Sensitivity ranking', columns: ['Driver', `LCOW at −${v.sensPct} % ($/m³)`, `LCOW at +${v.sensPct} % ($/m³)`, 'Swing ($/m³)', 'Swing (% of LCOW)'], rows: tornado.map((t) => [t.name, an.lcow + t.lo, an.lcow + t.hi, t.swing, (100 * t.swing) / an.lcow]) },
        { title: 'Scenarios and uncertainty', columns: ['Case', 'LCOW ($/m³)', 'Capital (M$)', 'OPEX (M$/y)', 'NPV (M$)', 'Project IRR (%)', 'Production (Mm³/y)'],
          rows: [...scenarios.map((q) => [q.name, q.lcow, q.tci / M, q.opex / M, q.npv / M, Number.isFinite(q.irr) ? 100 * q.irr : null, q.prod / M]), ['Monte-Carlo P10', p10, null, null, quantile(mcN, 0.9) / M, null, null], ['Monte-Carlo P50', p50, null, null, quantile(mcN, 0.5) / M, null, null], ['Monte-Carlo P90', p90, null, null, quantile(mcN, 0.1) / M, null, null], ['Monte-Carlo mean ± st. dev.', mean(mcL), null, null, mean(mcN) / M, null, null]],
          note: `Scenarios move every driver half-way to its favourable or adverse bound. Monte-Carlo: ${nMC} ${v.mcDist === 'tri' ? 'triangular' : v.mcDist} samples, seed ${v.mcSeed}, standard deviation ${fmt(std(mcL), 3)} $/m³, standard error of the mean ${fmt(std(mcL) / Math.sqrt(nMC), 2)} $/m³; ${fmt(100 * pExceed, 3)} % of samples exceed ${v.targetLcow} $/m³.` },
        { title: 'Brine-management options and cost–carbon trade-off', columns: ['Option', 'Capital (M$)', 'Addition to LCOW ($/m³)', 'Carbon (tCO₂/y) / intensity (kg/m³)', 'Specific energy (kWh/m³)', 'Non-dominated'],
          rows: [...brineCmp.slice(0, 5).map((b) => [`Brine: ${b.name}`, b.capex / M, b.add, b.co2, null, null]), ...pts.map((p) => [`${p.design}, ${100 * p.share} % renewable`, null, p.lcow, p.ci, p.sec, p.pareto ? 'yes' : 'no'])],
          note: 'Brine rows: cost added to the product water by each route on the current brine flow (indicative unit costs). Design rows: total LCOW; a point is non-dominated when no other point is at least as good in cost, carbon and energy.' },
        { title: 'Levelised cost of energy (on-site generation)', columns: ['Item', 'Value', 'Unit'], rows: [
          ['Capital recovery', gen.capital, '$/kWh'], ['Fixed operation and maintenance', gen.om, '$/kWh'], ['Fuel', gen.fuel, '$/kWh'], ['Levelised cost of energy (real)', gen.lcoe, '$/kWh'], ['Capital-recovery factor of the generator', gen.crf, '1/y'], ['Energy per installed kW', gen.energy, 'kWh/kW·y'],
          ['Grid electricity price (effective)', c.tou.price, '$/kWh'], ['Self-supplied share', 100 * c.self, '%'], ['Blended electricity price used in the cost model', c.elecP, '$/kWh'], ['Blended emission factor', c.gridC, 'kgCO₂/kWh'], ['Generation needed for the self-supplied share', genKW, 'kW'], ['Its capital (carried in the energy price, not in the plant CAPEX)', (genKW * (v.genCapex ?? 900)) / M, 'M$'],
          ['Heat price used', c.heatP, '$/kWh'], ['Carnot factor of the heat source', ex.carnot, '–']],
          note: 'LCOE = (CRF·CAPEX + O&M) ÷ (8760 × capacity factor) + fuel, at the real discount rate. The self-supplied share is bought at this cost, as under a power-purchase agreement.' },
        { title: 'Exergy and exergoeconomic analysis (per m³ of product)', columns: ['Subsystem', 'Fuel exergy (kWh/m³)', 'Product exergy (kWh/m³)', 'Exergy destruction (kWh/m³)', 'Exergy efficiency (%)', 'Fuel unit cost ($/kWh)', 'Product unit cost ($/kWh)', 'Capital + O&M rate Ż ($/m³)', 'Cost of destruction Ċ_D ($/m³)', 'Exergoeconomic factor f (%)', 'Relative cost difference r'],
          rows: [...ex.rows.map((q) => [q.name, q.fuel, q.prod, q.D, 100 * q.eff, q.cF, q.cP, q.z, q.cD, 100 * q.f, q.rel]), ['Whole plant', ex.fuel, ex.wmin, ex.dest, 100 * ex.eff, null, ex.cProduct, ex.zTot, sum(ex.rows.map((q) => q.cD)), null, null]],
          note: `Fuel exergy = electricity + heat × Carnot factor (${fmt(ex.carnot, 3)}); product exergy = least work of separation at ${fmt(v.feedSalinity ?? 35, 3)} g/kg and ${fmt(100 * c.rec, 3)} % recovery (${fmt(ex.wmin, 3)} kWh/m³). Unit exergetic cost k* = ${fmt(ex.kStar, 3)}. Implied membrane feed pressure ${fmt(ex.pBar, 3)} bar. Specific exergy costing: the pump product is hydraulic exergy; the energy-recovery device returns brine exergy at the membrane-feed unit cost; capital, membranes, chemicals and other non-energy costs are allocated by installed cost and added as Ż. The product cost rate equals the LCOW.` },
        { title: 'Life-cycle assessment and its cost', columns: ['Life-cycle stage', 'Greenhouse gases (kgCO₂e/m³)', 'Share (%)', 'Primary energy (kWh/m³)'],
          rows: [...Object.keys(lca.stages).map((k) => [k, lca.stages[k], lca.ci > 0 ? (100 * lca.stages[k]) / lca.ci : 0, lca.cedStages[k]]), ['Total, cradle to grave', lca.ci, 100, lca.ced], ['of which operation (electricity and heat)', lca.opCi, lca.ci > 0 ? (100 * lca.opCi) / lca.ci : 0, null], ['of which embodied', lca.embodied, lca.ci > 0 ? (100 * lca.embodied) / lca.ci : 0, null],
            ['Life-cycle carbon cost at the carbon price ($/m³)', lca.carbonCostLC, null, null], ['LCOW including embodied carbon ($/m³)', lca.lcowLC, null, null], ['Abatement cost of renewable electricity ($/tCO₂)', abate, null, null], ['Eco-efficiency (m³ per $·kgCO₂e)', lca.ecoEff, null, null]],
          note: `Emission factors: construction ${v.efCapex ?? 0.3} kgCO₂e per $ of direct cost, membranes ${v.efMem ?? 15} kgCO₂e/m², chemicals ${v.efChem ?? 1.1} kgCO₂e/kg, end of life ${v.efEol ?? 5} % of construction; totals over ${f.N} years of operation divided by the water produced. Primary-energy factors are generic (electricity 2.5, heat 1.1 kWh/kWh; 3.5 kWh/$ construction; 60 kWh/m² membrane; 6 kWh/kg chemicals). Indicative values.` },
        { title: 'Response-surface surrogate of the cost model', columns: ['Quantity', 'Cost model (direct Monte-Carlo)', 'Surrogate'], rows: [
          ['Samples', nMC, sg.nMC], ['LCOW P10 ($/m³)', p10, sg.p10], ['LCOW P50 ($/m³)', p50, sg.p50], ['LCOW P90 ($/m³)', p90, sg.p90], ['Mean ($/m³)', mean(mcL), sg.mean], ['Standard deviation ($/m³)', std(mcL), sg.sd], ['Hold-out R²', null, sg.r2], ['Largest hold-out error (%)', null, sg.maxErr],
          ...unc.map((d, j) => [`First-order sensitivity index: ${d.name}`, null, sg.S1[j]]), ['Sum of first-order indices', null, sum(sg.S1)]],
          note: `Quadratic polynomial in the ${unc.length} drivers (${sg.fit.terms} terms; interaction terms among capital cost, electricity price, specific energy, capacity factor, discount rate and plant life), fitted by least squares to ${sg.nTrain} Latin-hypercube runs of the cost model and checked on ${sg.nTest} further runs. The indices are Var(E[LCOW | driver]) ÷ Var(LCOW) from the surrogate sample; their sum below 1 indicates interactions and correlation.` },
        { title: 'Initial conditions, horizon and constraints', columns: ['Condition', 'Value', 'Limit', 'Unit', 'Status'], rows: [
          ['Initial condition: capital investment at start of operation', c.tci / M, null, 'M$', ''], ['Initial condition: debt / equity', `${fmt(cash.D / M, 4)} / ${fmt((c.tci - c.grant - cash.D) / M, 4)}`, null, 'M$', ''], ['Initial condition: electricity price', c.elecP, null, '$/kWh', ''], ['Initial condition: chemical cost', c.o.chem / 1000, null, 'k$/y', ''],
          ['Initial condition: first-year production', cash.rows[0].Q / M, null, 'Mm³', ''], ['Initial condition: depreciable asset value', Math.max(0, c.base + c.idc - c.land - c.grant) / M, null, 'M$', ''],
          ['Horizon: construction + operating life', `${Math.round(v.constYears)} + ${f.N}`, null, 'years', ''], ['Horizon: loan fully repaid in year', cash.loan.length, f.N, 'year', cash.loan.length <= f.N ? 'ok' : 'violated'], ['Horizon: terminal value in the final year (nominal)', cash.rows[f.N - 1].terminal / M, null, 'M$', ''],
          ['Constraint: maximum water cost', an.lcow, v.targetLcow, '$/m³', an.lcow <= v.targetLcow ? 'met' : 'violated'], ['Constraint: minimum debt-service cover', minDscr, v.minDscr, '–', minDscr === null ? 'no debt' : minDscr >= v.minDscr ? 'met' : 'violated'],
          ['Constraint: minimum annual production', c.prod / M, c.minProd / M, 'Mm³/y', !(c.minProd > 0) ? 'none set' : c.shortfall > 0 ? 'violated' : 'met'], ['  capacity factor needed for it', 100 * c.cf, needAvail, '%', ''], ['  chance of missing it (Monte-Carlo)', 100 * pMiss, null, '%', ''], ['  shortfall penalty', c.o.shortfall / 1000, null, 'k$/y', ''],
          ['Constraint: emissions cap', ci, emis.active ? capC : null, 'kgCO₂/m³', !emis.active ? 'none set' : emis.ok ? 'met' : 'violated'], ['  renewable share of electricity needed to comply', 100 * emis.share, 100, '%', emis.feasible ? '' : 'not sufficient'], ['  water cost when complying', emis.lcow, v.targetLcow, '$/m³', ''],
          ['  least-cost compliant design in the trade-off set', emis.best ? `${emis.best.design}, ${100 * emis.best.share} % renewable` : emis.active ? 'none' : 'n/a', null, '', ''], ['Constraint: largest single-plant size', c.cap, v.maxScale, 'm³/d', c.sat > 1 ? 'replicated' : 'met']],
          note: 'Economic initial conditions are the state at the start of operation (year 0 of the cash flow); the horizon closes the cash flow with the terminal value; constraints are checked on the deterministic case and, for production, on the Monte-Carlo sample.' },
      ],
      balances: [
        { name: 'Exergy: fuel = product + destruction (kWh/m³)', in: ex.fuel, out: ex.wmin + ex.dest },
        { name: 'Exergy costing: product cost rate = LCOW ($/m³)', in: an.lcow, out: ex.prodCost },
        { name: 'Life-cycle stages sum to the total (kgCO₂e/m³)', in: lca.ci, out: sum(Object.values(lca.stages)) },
        { name: 'LCOW components sum to the total ($/m³)', in: an.lcow, out: sum(Object.values(an.parts)) },
        { name: 'Capital items sum to the total investment (M$)', in: c.tci / M, out: (c.direct + c.indTot + c.contingency + c.land + c.wc + c.idc) / M },
        { name: 'Depreciation sums to the depreciable base (M$)', in: Math.max(0, c.base + c.idc - c.land - c.grant) / M, out: sum(cash.depr) / M },
        { name: 'Loan principal repaid equals the debt (M$)', in: cash.D / M, out: sum(cash.loan.map((r) => r.principal)) / M },
        { name: 'Present value of costs = levelised cost × discounted water (M$)', in: cash.pvCost / M, out: (cash.lcowReal * npv(f.dr, cash.water)) / M },
      ],
      outputs: out,
    };
  },

  mesh: [
    { name: 'Monte-Carlo sample count', keys: ['nMC'], min: 200, note: 'Sampling error falls with 1/√N, so the observed order is about 0.5 and convergence may be oscillatory.', metrics: [{ label: 'LCOW P50', unit: '$/m³', get: (r) => r.outputs.p50 }, { label: 'LCOW P90', unit: '$/m³', get: (r) => r.outputs.p90 }] },
    { name: 'Binomial-tree steps', keys: ['optSteps'], min: 4, metrics: [{ label: 'Option-to-defer value', unit: '$', get: (r) => r.outputs.deferralValue }] },
  ],

  calibration: {
    note: 'Fit the cost-correlation multiplier, the scale-exponent adjustment and the membrane price level to reference projects. Each row is one plant: capacity, specific energy and electricity price set the case; specific capital cost and levelised water cost are the reported figures. The bundled rows are ILLUSTRATIVE, synthetic seawater-RO plants for demonstration — they are not records of real projects; replace them with documented projects in your region and currency year. Staffing and land are scaled from the 100 000 m³/d defaults for each row.',
    params: [{ key: 'kCapex', label: 'Cost-correlation multiplier', lo: 0.4, hi: 2.5 }, { key: 'expShift', label: 'Scale-exponent adjustment', lo: -0.25, hi: 0.25 }, { key: 'maintPct', label: 'Maintenance (% of direct CAPEX per year)', lo: 0.5, hi: 6 }],
    columns: [{ key: 'capacity', label: 'Capacity', unit: 'm³/d' }, { key: 'sec', label: 'Specific energy', unit: 'kWh/m³' }, { key: 'elecPrice', label: 'Electricity price', unit: '$/kWh' }, { key: 'capexSpec', label: 'Specific capital cost', unit: '$ per m³/d' }, { key: 'lcow', label: 'Levelised cost', unit: '$/m³' }],
    targets: [{ key: 'capexSpec', label: 'Specific capital cost', unit: '$ per m³/d' }, { key: 'lcow', label: 'Levelised cost of water', unit: '$/m³' }],
    model(v) {
      const [vv, mm] = atCapacity({ ...v, membraneArea: 0, tou: false }, v.capacity, REF.cap), e = evaluate(vv, mm);
      return { capexSpec: e.c.tci / e.c.cap, lcow: e.lcow };
    },
    get sample() { return (this._s ||= synth(7, [[10000, 3.9, 0.11], [20000, 3.7, 0.09], [36000, 3.6, 0.1], [50000, 3.5, 0.08], [75000, 3.4, 0.07], [100000, 3.3, 0.08], [150000, 3.2, 0.06], [200000, 3.2, 0.075], [330000, 3.1, 0.055], [600000, 3.0, 0.05]])); },
    get validationSample() { return (this._v ||= synth(31, [[15000, 3.8, 0.1], [45000, 3.5, 0.085], [90000, 3.4, 0.09], [135000, 3.25, 0.065], [250000, 3.15, 0.06], [400000, 3.05, 0.07], [540000, 3.0, 0.052]])); },
  },

  verify() {
    const d = defaultsOf(suite), C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    add('Capital-recovery factor, closed form', 0.08 / (1 - 1.08 ** -20), crf(0.08, 20), 1e-14, 'i(1+i)ⁿ/((1+i)ⁿ−1) at 8 %, 20 years = 0.101852');
    add('Capital-recovery factor, published value', 0.101852, crf(0.08, 20), 1e-6, 'Interest tables, 8 %, 20 years');
    add('Annuity identity', 1, npv(0.07, [0, ...new Array(25).fill(crf(0.07, 25))]), 1e-12, 'Present value of n payments of CRF equals 1');
    add('CRF tends to 1/n at zero rate', 1 / 25, crf(1e-14, 25), 1e-12, 'Limiting case i → 0');
    const cf = [-1000, 300, 420, 680];
    add('NPV at zero rate equals the simple sum', 400, npv(0, cf), 1e-12, '−1000 + 300 + 420 + 680');
    add('NPV hand calculation at 10 %', -1000 + 300 / 1.1 + 420 / 1.21 + 680 / 1.331, npv(0.1, cf), 1e-10, 'Three-year example');
    const r = irr(cf);
    add('IRR makes the NPV zero', 0, npv(r, cf), 1e-7, `IRR = ${fmt(100 * r, 5)} %`);
    add('IRR of a one-period loan', 0.1, irr([-100, 110]), 1e-9, '−100 now, +110 in a year');
    add('MIRR hand calculation', (((300 * 1.1 ** 2 + 420 * 1.1 + 680) / 1000) ** (1 / 3)) - 1, mirr(cf, 0.06, 0.1), 1e-12, 'Reinvest at 10 %');
    add('Simple payback hand calculation', 2 + 280 / 680, payback(cf), 1e-12, 'Cumulative −1000, −700, −280, +400 → 2.41 years');
    add('Discounted payback is longer than simple payback', 1, payback(cf, 0.1) > payback(cf) ? 1 : 0, 0, 'Discounting delays recovery');
    add('Fisher relation', 0.08, (1 + fisherReal(0.08, 0.025)) * 1.025 - 1, 1e-14, '(1 + real)(1 + inflation) = 1 + nominal');
    for (const mth of ['sl', 'db', 'macrs']) add(`Depreciation (${mth}) sums to the depreciable base`, 1e6, sum(depreciation(1e6, 20, mth, 25)), 1e-6, 'Σ annual charges = base');
    add('Straight-line charge', 5e4, depreciation(1e6, 20, 'sl', 25)[7], 1e-9, 'base ÷ life');
    const loan = loanSchedule(7e7, 0.06, 15);
    add('Loan amortisation ends at zero', 0, loan[14].balance, 1e-3, 'Closing balance after the last level payment');
    add('Loan payment equals debt × CRF', 7e7 * crf(0.06, 15), loan[3].interest + loan[3].principal, 1e-6, 'Level annual debt service');
    // constant case: annualised LCOW = DCF LCOW
    const k = { ...d, ramp: 100, escElec: 0, escLabour: 0, escChem: 0, escMem: 0, escOther: 0 }, e = evaluate(k, {}, true);
    add('Annualised LCOW equals cash-flow LCOW for constant real costs', e.lcow, e.cash.lcowReal, 1e-9, 'No ramp-up, no real escalation: both methods must agree (with inflation, via the Fisher relation)');
    const z = evaluate({ ...k, inflation: 0, discNominal: 0, salvagePct: 0, wcMonths: 0, debtPct: 0, replacements: [] }, {}, true);
    add('Zero-discount LCOW = (CAPEX/life + OPEX) ÷ production', (z.c.tci / z.f.N + z.c.opex) / z.c.prod, z.lcow, 1e-12, 'Hand calculation without discounting, salvage or replacements');
    add('Nominal LCOW exceeds real LCOW under inflation', 1, e.cash.lcowNom > e.cash.lcowReal ? 1 : 0, 0, 'Levelising in money of the day');
    const be = brent((x) => cashFlow(e.c, e.f, k, x).npv, 0, 60, 1e-10);
    add('Break-even tariff gives zero NPV', 0, cashFlow(e.c, e.f, k, be).npv / e.c.tci, 1e-7, `Tariff ${fmt(be, 4)} $/m³`);
    add('Cost–capacity rule: doubling a 0.7-exponent item', 2 ** 0.7, buildCosts({ ...d, capacity: 2e5, equipment: [{ name: 'x', basis: 'cap', base: 1, n: 0.7, install: 1 }] }).items[0].equip / buildCosts({ ...d, equipment: [{ name: 'x', basis: 'cap', base: 1, n: 0.7, install: 1 }] }).items[0].equip, 1e-12, 'C₂/C₁ = (Q₂/Q₁)ⁿ');
    // Monte Carlo of a linear model against its analytic mean
    const spec = [{ lo: -15, hi: 30 }, { lo: -25, hi: 40 }, { lo: -10, hi: 10 }], a = [0.3, 0.25, 0.2], n = 4000, smp = sampleMultipliers(n, spec, 'tri', [0, 1, 0.4], 99).map((mm) => sum(mm.map((x, j) => a[j] * x)));
    const exact = sum(spec.map((q, j) => a[j] * (1 + (q.lo + q.hi) / 300)));
    add('Monte-Carlo mean of a linear model matches the analytic mean', exact, mean(smp), (4 * std(smp)) / Math.sqrt(n), 'Triangular inputs, E[x] = (min + mode + max)/3; tolerance = 4 standard errors');
    add('Binomial tree converges to Black–Scholes', blackScholesCall(100, 100, 0.04, 0.22, 5), binomialOption({ V: 100, K: 100, r: 0.04, sigma: 0.22, T: 5, steps: 400, american: false }), 0.05, 'European call, no yield');
    add('Carbon intensity = emission factor × specific energy', d.gridCarbon * d.sec + d.gridCarbon * 0.03 * (1 / 0.45 - 1), (e.c.tCO2 * 1000) / e.c.prod, 1e-9, 'Electricity only (membrane plant), including outfall pumping');
    // ---- levelised cost of energy
    const g1 = lcoe({ capex: 1000, cf: 0.25, om: 0.02, fuel: 0.01, life: 20, rate: 0.08 });
    add('LCOE hand calculation', (0.101852209 * 1000 + 20) / 2190 + 0.01, g1.lcoe, 1e-9, '(CRF·CAPEX + O&M) ÷ (8760 × 0.25) + fuel at 8 %, 20 years = 0.0656 $/kWh');
    add('LCOE at zero discount rate', (1000 / 20 + 20) / 2190, lcoe({ capex: 1000, cf: 0.25, om: 0.02, fuel: 0, life: 20, rate: 0 }).lcoe, 1e-12, '(CAPEX/life + O&M) ÷ annual energy');
    add('LCOE annuity identity', 1000, npv(0.08, [0, ...new Array(20).fill(g1.capital * g1.energy)]), 1e-6, 'Present value of the capital part of the LCOE revenue over the life equals the investment');
    const es = evaluate({ ...k, selfShare: 100 }), eg = evaluate({ ...k, selfShare: 0 });
    add('Self-supplied plant pays the LCOE for its electricity', es.c.gen.lcoe * es.c.elecKWh, es.c.o.elec, 1e-6, 'Electricity cost = LCOE × annual consumption at 100 % self-supply');
    add('Zero self-supply leaves the grid price unchanged', d.elecPrice, eg.c.elecP, 1e-15, 'Blended price at 0 % share');
    // ---- thermoeconomic / exergoeconomic costing
    const xe = exergoeconomics(e.c, e.an, k);
    add('Exergy balance closes: fuel = product + destruction', 0, xe.balance / xe.fuel, 1e-12, 'Sum over the subsystems, kWh per m³ of product');
    add('Exergoeconomic cost balance: product cost rate equals the LCOW', e.lcow, xe.prodCost, 1e-10, 'Σ fuel cost + Σ Ż carried to the product by specific exergy costing');
    add('Least work of separation, closed form', ((osmoticPressure(25, 35) / 3.6e6) * -Math.log(0.55)) / 0.45, xe.wmin, 1e-12, 'w_min = π_f·ln(1/(1 − r))/r for 35 g/kg at 45 % recovery ≈ 0.96 kWh/m³');
    add('Second-law efficiency is between 0 and 1', 1, xe.eff > 0 && xe.eff < 1 ? 1 : 0, 0, `${fmt(100 * xe.eff, 3)} %`);
    add('Pump subsystem: product unit cost = (c_F + Ż/W)/η', (e.c.elecP + xe.rows[0].z / xe.rows[0].fuel) / 0.82, xe.rows[0].cP, 1e-12, 'Cost balance of the high-pressure pump, $ per kWh of hydraulic exergy');
    add('Carnot factor of 70 °C heat at 25 °C ambient', 1 - 298.15 / 343.15, xe.carnot, 1e-12, '1 − T₀/T');
    const hx = buildCosts({ ...d, secThermal: 20, heatCosting: 'exergy' });
    add('Thermoeconomic heat price = electricity price × Carnot factor × 0.85', d.elecPrice * (1 - 298.15 / 343.15) * 0.85, hx.heatP, 1e-12, 'Heat valued by its exergy');
    // ---- life-cycle assessment
    const lc = lcaTea(e.c, e.f, k, e.an);
    add('Life-cycle stages sum to the total', lc.ci, sum(Object.values(lc.stages)), 1e-12, 'kgCO₂e per m³');
    add('Operational stage of the LCA equals the carbon accounting', (e.c.tCO2 * 1000) / e.c.prod, lc.stages.Electricity + lc.stages.Heat, 1e-12, 'Electricity + heat');
    add('Embodied construction carbon, hand calculation', (0.3 * e.c.direct) / (e.c.prod * e.f.N), lc.stages.Construction, 1e-15, 'Emission factor × direct cost ÷ lifetime production');
    add('LCOW with embodied carbon = LCOW + embodied intensity × carbon price', e.lcow + (lc.embodied * d.carbonPrice) / 1000, lc.lcowLC, 1e-12, 'Internalised life-cycle externality');
    // ---- surrogate
    { const Xq = lhs(60, 3, 5).map((u) => u.map((x) => 2 * x - 1)), fq = (x) => 2 - x[0] + 0.5 * x[1] * x[1] + 0.3 * x[0] * x[2] - 0.2 * x[2], q = fitQuadratic(Xq, Xq.map(fq));
      add('Quadratic response surface reproduces a quadratic function exactly', fq([0.3, -0.7, 0.5]), q.predict([0.3, -0.7, 0.5]), 1e-8, 'Least-squares fit of 10 terms to 60 Latin-hypercube points'); }
    const uncD = DRIVERS.map((q) => ({ ...q })), sgt = surrogateTEA(d, prepare(d), uncD, { seed: 3 }), direct = sampleMultipliers(1500, uncD, 'tri', null, 20).map((mm) => evaluate(d, Object.fromEntries(DRIVERS.map((q, j) => [q.k, mm[j]]))).lcow);
    add('Surrogate reproduces hold-out runs of the cost model', 1, sgt.r2, 0.005, 'R² on Latin-hypercube runs not used in the fit');
    add('Surrogate median agrees with direct Monte-Carlo', quantile(direct, 0.5), sgt.p50, 0.01, 'P50 of LCOW, independent samples ($/m³)');
    add('First-order sensitivity indices lie between 0 and 1', 1, sgt.S1.every((x) => x >= 0 && x <= 1) && sum(sgt.S1) > 0.7 && sum(sgt.S1) < 1.3 ? 1 : 0, 0, `Sum = ${fmt(sum(sgt.S1), 3)}`);
    // ---- initial conditions, horizon and constraints
    add('Initial condition: year-0 cash flow is the capital investment net of grants', -(e.c.tci - e.c.grant), e.cash.proj[0], 1e-6, 'State of the project at the start of operation');
    add('Horizon: the cash flow spans the operating life and ends with the terminal value', 1, e.cash.proj.length === e.f.N + 1 && e.cash.rows[e.f.N - 1].terminal > 0 && e.cash.rows[e.f.N - 2].terminal === 0 ? 1 : 0, 0, 'N + 1 entries; salvage and working capital recovered in year N only');
    const sh = buildCosts({ ...d, availability: 80, minProdPct: 90, shortfallPenalty: 0.5 });
    add('Minimum-production constraint: shortfall penalty, hand calculation', 0.5 * (0.9 - 0.8) * d.capacity * 365, sh.o.shortfall, 1e-6, 'Penalty × (minimum − actual) with 80 % capacity factor against a 90 % minimum');
    add('Minimum-production constraint is inactive when production is sufficient', 0, buildCosts({ ...d, minProdPct: 85, shortfallPenalty: 0.5 }).o.shortfall, 0, '92 % capacity factor against an 85 % minimum');
    const capT = 0.9, ci0 = (e.c.tCO2 * 1000) / e.c.prod, rA = (s2) => evaluate({ ...k, tou: false, selfShare: 0, elecPrice: (1 - s2) * e.c.elecP + s2 * d.renPrice, gridCarbon: (1 - s2) * e.c.gridC + s2 * d.renCarbon }), ciR = (q) => (q.c.tCO2 * 1000) / q.c.prod, sReq = (ci0 - capT) / (ci0 - ciR(rA(1)));
    add('Emissions constraint: the solved renewable share meets the cap exactly', capT, ciR(rA(sReq)), 1e-9, `Share ${fmt(100 * sReq, 4)} % for a cap of 0.9 kgCO₂/m³`);
    return C;
  },
};

/** Illustrative reference plants: the model with different "true" coefficients plus deterministic scatter. */
function synth(seed, pts) {
  const d = defaultsOf(suite), g = rng(seed);
  return pts.map(([capacity, sec, elecPrice]) => {
    const m = suite.calibration.model({ ...d, kCapex: 0.94, expShift: 0.03, maintPct: 2.3, capacity, sec, elecPrice });
    return { capacity, sec, elecPrice, capexSpec: Math.round((m.capexSpec * (1 + g.normal(0, 0.05))) / 10) * 10, lcow: +(m.lcow * (1 + g.normal(0, 0.03))).toFixed(3) };
  });
}

export default suite;
