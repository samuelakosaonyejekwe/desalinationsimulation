// Suite 2 — Brine chemistry, precipitation and scaling.
// Aqueous speciation (carbonate, borate, silicate, sulphate and fluoride acid–base systems, water
// dissociation, ion pairs) solved by mass action with mass and alkalinity balances; activity
// coefficients from Debye–Hückel, extended Debye–Hückel, Davies, Truesdell–Jones or the Pitzer
// ion-interaction model (Harvie–Møller–Weare 25 °C parameter set); temperature- and pressure-dependent
// solubility products; saturation indices; equilibrium precipitation; concentration paths; chemical
// dosing; nucleation and growth kinetics; scaling and corrosion indices.
import { brent, clamp, linspace, logspace, sum, rng, fmt, solveLinear as solveLin } from '../core/num.js';
import { density, viscosity, R, KELVIN } from '../core/props.js';
import { IONS, ION_IDS, WATERS, cloneIons, tds, chargeBalance, conductivity, hardness, molar } from '../core/water.js';

const LN10 = Math.LN10, MW_W = 0.0180153, KB = 1.380649e-23, NA = 6.02214076e23;
const tk = (T) => T + KELVIN;
const analytic = (a, b, c, d, e = 0) => (T) => { const K = tk(T); return a + b * K + c / K + d * Math.log10(K) + e / (K * K); };
/** van't Hoff extrapolation of log K from 25 °C with a constant reaction enthalpy dH (kJ/mol). */
const vh = (logK25, dH = 0) => (T) => logK25 - ((dH * 1000) / (R * LN10)) * (1 / tk(T) - 1 / 298.15);

// Acid–base and gas constants (Plummer & Busenberg 1982 and the WATEQ/PHREEQC compilations).
const logK1 = analytic(-356.3094, -0.06091964, 21834.37, 126.8339, -1684915); // CO2(aq) + H2O = H+ + HCO3-
const logK2 = analytic(-107.8871, -0.03252849, 5151.79, 38.92561, -563713.9); // HCO3- = H+ + CO3 2-
const logKw = analytic(-283.971, -0.05069842, 13323.0, 102.24447, -1119669); // H2O = H+ + OH-
const logKH = analytic(108.3865, 0.01985076, -6919.53, -40.45154, 669365); // CO2(g) = CO2(aq), mol/kg/atm
const logKb = vh(-9.236, 13.8); // B(OH)3 + H2O = B(OH)4- + H+
const logKsi = analytic(-302.3724, -0.050698, 15669.69, 108.18466, -1119669); // H4SiO4 = H3SiO4- + H+

// ---- species ---------------------------------------------------------------------------------------
// Master (component) species; carbonate is carried as CO3 2-, boron as B(OH)3, silica as SiO2(aq),
// phosphate as HPO4 2-.
const MAST = ['Na', 'K', 'Ca', 'Mg', 'Ba', 'Sr', 'NH4', 'Fe', 'Mn', 'Cl', 'SO4', 'NO3', 'F', 'PO4', 'C', 'B', 'Si'];
const MZ = [1, 1, 2, 2, 2, 2, 1, 2, 2, -1, -2, -1, -1, -2, -2, 0, 0];
const MNAME = { C: 'CO3', B: 'B(OH)3', Si: 'SiO2', PO4: 'HPO4' };
const MION = { B: 'B', Si: 'SiO2' }; // master -> water-analysis id where it differs
const NM = MAST.length, IH = NM, IC = MAST.indexOf('C');
const mi = (k) => MAST.indexOf(k);
// Derived species formed from masters and H+: [id, charge, master A, master B, nH, nH2O, logK Pitzer set, logK ion-pair set].
// m = K · a_A · a_B · a_H^nH · a_w^nH2O / γ. A null constant switches the species off for that model family.
const kc1 = (T) => -logK2(T), kc2 = (T) => -logK1(T) - logK2(T), kso4 = vh(1.988, 16.1), khf = vh(3.18, 13.3);
const DER = [
  ['OH', -1, '', '', -1, 1, logKw, logKw], ['HCO3', -1, 'C', '', 1, 0, kc1, kc1], ['CO2', 0, 'C', '', 2, -1, kc2, kc2],
  ['B(OH)4', -1, 'B', '', -1, 1, logKb, logKb], ['H3SiO4', -1, 'Si', '', -1, 0, logKsi, logKsi], ['HSO4', -1, 'SO4', '', 1, 0, kso4, kso4], ['HF', 0, 'F', '', 1, 0, khf, khf],
  ['MgOH', 1, 'Mg', '', -1, 1, vh(-11.809, 66.7), vh(-11.44, 66.7)], ['CaCO3°', 0, 'Ca', 'C', 0, 0, vh(3.151, 14.8), vh(3.224, 14.8)], ['MgCO3°', 0, 'Mg', 'C', 0, 0, vh(2.928, 11.35), vh(2.98, 11.35)],
  ['CaOH', 1, 'Ca', '', -1, 1, null, vh(-12.78, 64)], ['CaSO4°', 0, 'Ca', 'SO4', 0, 0, null, vh(2.3, 6.9)], ['MgSO4°', 0, 'Mg', 'SO4', 0, 0, null, vh(2.37, 19)],
  ['NaSO4', -1, 'Na', 'SO4', 0, 0, null, vh(0.7, 4.7)], ['KSO4', -1, 'K', 'SO4', 0, 0, null, vh(0.85, 9.4)], ['BaSO4°', 0, 'Ba', 'SO4', 0, 0, null, vh(2.7, 0)], ['SrSO4°', 0, 'Sr', 'SO4', 0, 0, null, vh(2.29, 8.7)],
  ['CaHCO3', 1, 'Ca', 'C', 1, 0, null, (T) => 1.106 - logK2(T)], ['MgHCO3', 1, 'Mg', 'C', 1, 0, null, (T) => 1.07 - logK2(T)], ['NaHCO3°', 0, 'Na', 'C', 1, 0, null, (T) => -0.25 - logK2(T)],
  ['NaCO3', -1, 'Na', 'C', 0, 0, null, vh(1.27, 37.3)], ['CaF', 1, 'Ca', 'F', 0, 0, null, vh(0.94, 17.2)], ['MgF', 1, 'Mg', 'F', 0, 0, null, vh(1.82, 13.4)],
];
const ND = DER.length, NS = NM + 1 + ND;
const SID = [...MAST.map((k) => MNAME[k] || k), 'H', ...DER.map((d) => d[0])];
const ZS = [...MZ, 1, ...DER.map((d) => d[1])];
const DA = DER.map((d) => mi(d[2])), DB = DER.map((d) => mi(d[3])), DNH = DER.map((d) => d[4]), DNW = DER.map((d) => d[5]);
const si = (id) => SID.indexOf(id);
const IOH = si('OH'), ICO2 = si('CO2'), JCO2 = ICO2 - NM - 1;
// Alkalinity equivalents carried by each species (reference level: CO2, B(OH)3, SiO2, SO4, F, H2O).
const ALK = SID.map((_, s) => (s < NM ? (s === IC ? 2 : 0) : s === IH ? -1 : (DA[s - NM - 1] === IC || DB[s - NM - 1] === IC ? 2 : 0) - DNH[s - NM - 1]));
const CARB = SID.map((_, s) => s === IC || (s > IH && (DA[s - NM - 1] === IC || DB[s - NM - 1] === IC)));
const SUP = { 1: '⁺', 2: '²⁺', 3: '³⁺', '-1': '⁻', '-2': '²⁻', '-3': '³⁻' };
const CHARGE_LABEL = (s) => (ZS[s] === 0 ? SID[s] : SID[s] + SUP[ZS[s]]);

// ---- activity models -------------------------------------------------------------------------------
export const ACTIVITY_MODELS = { pitzer: 'Pitzer ion interaction (Harvie–Møller–Weare)', tj: 'Truesdell–Jones / B-dot + ion pairs', davies: 'Davies + ion pairs', edh: 'Extended Debye–Hückel + ion pairs', dh: 'Debye–Hückel limiting law + ion pairs' };
/** Debye–Hückel osmotic slope Aφ (kg½/mol½), quadratic through the 0, 25 and 100 °C values. */
const aphi = (T) => 0.3767 + 5.087e-4 * T + 3.333e-6 * T * T;
// Ion-size parameter å (Å) and Truesdell–Jones b for the extended Debye–Hückel forms.
const SIZE = { Na: [4, 0.075], K: [3.5, 0.015], Ca: [5, 0.165], Mg: [5.5, 0.2], Ba: [5, 0.11], Sr: [5.26, 0.121], NH4: [2.5, 0.015], Fe: [6, 0.1], Mn: [6, 0.1], Cl: [3.5, 0.015], SO4: [5, -0.04], NO3: [3, 0.015], F: [3.5, 0.02], HPO4: [4, 0], CO3: [5.4, 0], H: [9, 0], OH: [3.5, 0.02], HCO3: [5.4, 0] };
const SA = SID.map((id) => (SIZE[id] || [4, 0.041])[0]), SB = SID.map((id) => (SIZE[id] || [4, 0.041])[1]);

// Pitzer parameters at 25 °C (Harvie, Møller & Weare 1984; borate from Felmy & Weare 1986; nitrate and
// fluoride from Pitzer's tabulations). Sr and Ba sulphate use the Ca–SO4 set; NH4 uses K, Fe/Mn use Mg.
const PZ_ID = { NH4: 'K', Fe: 'Mg', Mn: 'Mg', H3SiO4: 'HCO3' };
const PZ_BIN = 'Na Cl .0765 .2664 0 .00127|Na SO4 .01958 1.113 0 .00497|Na HSO4 .0454 .398 0 0|Na OH .0864 .253 0 .0044|Na HCO3 .0277 .0411 0 0|Na CO3 .0399 1.389 0 .0044|Na NO3 .0068 .1783 0 -.00072|Na F .0215 .2107 0 0|Na B(OH)4 -.0427 .089 0 .0114|Na HPO4 -.0583 1.4655 0 .0294|'
  + 'K Cl .04835 .2122 0 -.00084|K SO4 .04995 .7793 0 0|K HSO4 -.0003 .1735 0 0|K OH .1298 .32 0 .0041|K HCO3 .0296 -.013 0 -.008|K CO3 .1488 1.43 0 -.0015|K NO3 -.0816 .0494 0 .0066|K F .08089 .2021 0 .00093|K B(OH)4 .035 .14 0 0|'
  + 'Ca Cl .3159 1.614 0 -.00034|Ca SO4 .2 3.1973 -54.24 0|Ca HSO4 .2145 2.53 0 0|Ca OH -.1747 -.2303 -5.72 0|Ca HCO3 .4 2.977 0 0|Ca NO3 .2108 1.409 0 -.02014|'
  + 'Mg Cl .35235 1.6815 0 .00519|Mg SO4 .221 3.343 -37.23 .025|Mg HSO4 .4746 1.729 0 0|Mg HCO3 .329 .6072 0 0|Mg NO3 .367 1.585 0 -.02062|MgOH Cl -.1 1.658 0 0|'
  + 'Sr Cl .2858 1.667 0 -.0013|Sr SO4 .2 3.1973 -54.24 0|Ba Cl .2628 1.4963 0 -.01938|Ba SO4 .2 3.1973 -54.24 0|H Cl .1775 .2945 0 .0008|H SO4 .0298 0 0 .0438|H HSO4 .2065 .5556 0 0';
const PZ_THETA = 'Na K -.012|Na Ca .07|Na Mg .07|Na H .036|K Ca .032|K H .005|Ca Mg .007|Ca H .092|Mg H .1|Cl SO4 .02|Cl HSO4 -.006|Cl OH -.05|Cl HCO3 .03|Cl CO3 -.02|SO4 OH -.013|SO4 HCO3 .01|SO4 CO3 .02|OH CO3 .1|HCO3 CO3 -.04|Cl NO3 .016';
const PZ_PSI = 'Na K Cl -.0018|Na K SO4 -.01|Na K HCO3 -.003|Na K CO3 .003|Na Ca Cl -.007|Na Ca SO4 -.055|Na Mg Cl -.012|Na Mg SO4 -.015|Na H Cl -.004|Na H HSO4 -.0129|K Ca Cl -.025|K Mg Cl -.022|K Mg SO4 -.048|K H Cl -.011|K H SO4 .197|K H HSO4 -.0265|'
  + 'Ca Mg Cl -.012|Ca Mg SO4 .024|Ca H Cl -.015|Mg MgOH Cl .028|Mg H Cl -.011|Mg H HSO4 -.0178|Cl SO4 Na .0014|Cl SO4 Ca -.018|Cl SO4 Mg -.004|Cl HSO4 Na -.006|Cl HSO4 H .013|Cl OH Na -.006|Cl OH K -.006|Cl OH Ca -.025|Cl HCO3 Na -.015|Cl HCO3 Mg -.096|'
  + 'Cl CO3 Na .0085|Cl CO3 K .004|SO4 HSO4 Na -.0094|SO4 HSO4 K -.0677|SO4 HSO4 Mg -.0425|SO4 OH Na -.009|SO4 OH K -.05|SO4 HCO3 Na -.005|SO4 HCO3 Mg -.161|SO4 CO3 Na -.005|SO4 CO3 K -.009|OH CO3 Na -.017|OH CO3 K -.01|HCO3 CO3 Na .002|HCO3 CO3 K .012';
const PZ_LAM = 'CO2 Na .1|CO2 K .051|CO2 Ca .183|CO2 Mg .183|CO2 Cl -.005|CO2 SO4 .097|CO2 HSO4 -.003|B(OH)3 Na -.097|B(OH)3 K -.14|B(OH)3 Cl .091|B(OH)3 SO4 .018|SiO2 Na .104|SiO2 Mg .3|SiO2 Ca .3';
const PZ = (() => {
  const pid = SID.map((id) => PZ_ID[id] || id), rows = (s) => s.split('|').filter(Boolean).map((r) => r.split(' '));
  const idx = (name) => pid.map((p, i) => (p === name ? i : -1)).filter((i) => i >= 0);
  const B0 = new Float64Array(NS * NS), B1 = new Float64Array(NS * NS), B2 = new Float64Array(NS * NS), CM = new Float64Array(NS * NS), TH = new Float64Array(NS * NS), LAM = new Float64Array(NS * NS);
  const PSI = new Float64Array(NS * NS * NS), HASPSI = new Uint8Array(NS * NS), HASB = new Uint8Array(NS * NS);
  for (const [c, a, b0, b1, b2, cphi] of rows(PZ_BIN)) for (const i of idx(c)) for (const j of idx(a)) {
    const k = i * NS + j;
    B0[k] = +b0; B1[k] = +b1; B2[k] = +b2; CM[k] = +cphi / (2 * Math.sqrt(Math.abs(ZS[i] * ZS[j]))); HASB[k] = 1;
  }
  for (const [a, b, t] of rows(PZ_THETA)) for (const i of idx(a)) for (const j of idx(b)) { TH[i * NS + j] = +t; TH[j * NS + i] = +t; }
  for (const [a, b, c, p] of rows(PZ_PSI)) for (const i of idx(a)) for (const j of idx(b)) for (const k of idx(c)) {
    PSI[(i * NS + j) * NS + k] = +p; PSI[(j * NS + i) * NS + k] = +p; HASPSI[i * NS + j] = 1; HASPSI[j * NS + i] = 1;
  }
  for (const [n, i0, l] of rows(PZ_LAM)) for (const i of idx(n)) for (const j of idx(i0)) { LAM[i * NS + j] = +l; LAM[j * NS + i] = +l; }
  const all = SID.map((_, i) => i);
  return { B0, B1, B2, CM, TH, LAM, PSI, HASPSI, HASB, cat: all.filter((i) => ZS[i] > 0), an: all.filter((i) => ZS[i] < 0), neu: all.filter((i) => ZS[i] === 0), pc: new Int32Array(NS), pa: new Int32Array(NS) };
})();

/** Higher-order electrostatic mixing term Eθ and its ionic-strength derivative (Pitzer 1975 approximation of J). */
const pzJ = (x) => x / (4 + 4.581 * x ** -0.7237 * Math.exp(-0.012 * x ** 0.528));
const pzJp = (x) => { const e = 4.581 * x ** -0.7237 * Math.exp(-0.012 * x ** 0.528), d = 4 + e, dd = e * (-0.7237 / x - 0.012 * 0.528 * x ** -0.472); return 1 / d - (x * dd) / (d * d); };
function etheta(zi, zj, I, A, out) {
  const s = Math.sqrt(I), xij = 6 * zi * zj * A * s, xii = 6 * zi * zi * A * s, xjj = 6 * zj * zj * A * s;
  const e = ((zi * zj) / (4 * I)) * (pzJ(xij) - 0.5 * pzJ(xii) - 0.5 * pzJ(xjj));
  out[0] = e; out[1] = -e / I + ((zi * zj) / (8 * I * I)) * (xij * pzJp(xij) - 0.5 * xii * pzJp(xii) - 0.5 * xjj * pzJp(xjj));
}
const pzG = (x) => (2 * (1 - (1 + x) * Math.exp(-x))) / (x * x), pzGp = (x) => (-2 * (1 - (1 + x + 0.5 * x * x) * Math.exp(-x))) / (x * x);
const ETH = new Float64Array(2);

/** Pitzer model: fills lnG (natural log of molal activity coefficients); returns the ionic strength and sets PZ_OUT. */
const PZ_OUT = { I: 0, phi: 1, aw: 1 };
function pitzer(m, T, lnG) {
  const { B0, B1, B2, CM, TH, LAM, PSI, HASPSI, HASB, cat, an, neu, pc, pa } = PZ;
  let I = 0, Z = 0, sm = 0, nc = 0, na = 0;
  for (let i = 0; i < NS; i++) { const z = ZS[i], x = m[i]; lnG[i] = 0; if (!(x > 0)) continue; I += x * z * z; Z += x * (z < 0 ? -z : z); sm += x; if (z > 0) pc[nc++] = i; else if (z < 0) pa[na++] = i; }
  I *= 0.5;
  if (I < 1e-14) { PZ_OUT.I = I; PZ_OUT.phi = 1; PZ_OUT.aw = Math.exp(-MW_W * sm); return PZ_OUT; }
  const A = aphi(T), sI = Math.sqrt(I), b = 1.2;
  const g2 = pzG(2 * sI), gp2 = pzGp(2 * sI), e2 = Math.exp(-2 * sI), g14 = pzG(1.4 * sI), gp14 = pzGp(1.4 * sI), e14 = Math.exp(-1.4 * sI), g12 = pzG(12 * sI), gp12 = pzGp(12 * sI), e12 = Math.exp(-12 * sI);
  let F = -A * (sI / (1 + b * sI) + (2 / b) * Math.log(1 + b * sI)), osm = (-A * I * sI) / (1 + b * sI), CC = 0;
  for (let p = 0; p < nc; p++) {
    const c = pc[p], mc = m[c];
    for (let q = 0; q < na; q++) {
      const a = pa[q], k = c * NS + a;
      if (!HASB[k]) continue;
      const ma = m[a], b0 = B0[k], b1 = B1[k], b2 = B2[k], C = CM[k], two = ZS[c] === 2 && ZS[a] === -2;
      const B = b0 + b1 * (two ? g14 : g2) + b2 * g12, Bp = (b1 * (two ? gp14 : gp2) + b2 * gp12) / I, Bphi = b0 + b1 * (two ? e14 : e2) + b2 * e12, t = 2 * B + Z * C;
      F += mc * ma * Bp; CC += mc * ma * C; osm += mc * ma * (Bphi + Z * C);
      lnG[c] += ma * t; lnG[a] += mc * t;
    }
  }
  etheta(1, 2, I, A, ETH);
  const e12x = ETH[0], e12p = ETH[1];
  for (let pass = 0; pass < 2; pass++) { // cation–cation then anion–anion mixing with the opposite-charge triplets
    const set = pass ? pa : pc, ns = pass ? na : nc, oth = pass ? pc : pa, no = pass ? nc : na;
    for (let p = 0; p < ns; p++) {
      const i = set[p], mI = m[i], zi = Math.abs(ZS[i]);
      for (let q = p + 1; q < ns; q++) {
        const j = set[q], mJ = m[j], zj = Math.abs(ZS[j]), un = zi !== zj, ij = i * NS + j;
        const phi = TH[ij] + (un ? e12x : 0), phip = un ? e12p : 0;
        let sp = 0;
        if (HASPSI[ij]) for (let r = 0; r < no; r++) { const k = oth[r], ps = PSI[ij * NS + k]; if (ps !== 0) { sp += m[k] * ps; lnG[k] += mI * mJ * ps; } }
        F += mI * mJ * phip; osm += mI * mJ * (phi + I * phip + sp);
        lnG[i] += mJ * (2 * phi + sp); lnG[j] += mI * (2 * phi + sp);
      }
    }
  }
  for (const n of neu) {
    const mn = m[n];
    for (let p = 0; p < nc + na; p++) { const i = p < nc ? pc[p] : pa[p - nc], l = LAM[n * NS + i]; if (l === 0) continue; lnG[n] += 2 * m[i] * l; if (mn > 0) { lnG[i] += 2 * mn * l; osm += mn * m[i] * l; } }
  }
  for (let p = 0; p < nc + na; p++) { const i = p < nc ? pc[p] : pa[p - nc], z = ZS[i]; lnG[i] = clamp(lnG[i] + z * z * F + Math.abs(z) * CC, -45, 45); }
  // ions that are absent still need a coefficient for the mass-action constants: long-range term only
  for (let i = 0; i < NS; i++) if (!(m[i] > 0) && ZS[i] !== 0) lnG[i] = ZS[i] * ZS[i] * F;
  PZ_OUT.I = I; PZ_OUT.phi = 1 + (2 / sm) * osm; PZ_OUT.aw = Math.exp(-PZ_OUT.phi * sm * MW_W);
  return PZ_OUT;
}

/** Debye–Hückel family (log10 basis internally); neutral species follow a Setschenow salting-out term. */
function debye(model, m, T, lnG) {
  let I = 0, sm = 0;
  for (let i = 0; i < NS; i++) { I += m[i] * ZS[i] * ZS[i]; sm += m[i]; }
  I *= 0.5;
  const A = (3 * aphi(T)) / LN10, B = 0.3248 + 1.6e-4 * T, s = Math.sqrt(I);
  for (let i = 0; i < NS; i++) {
    const z2 = ZS[i] * ZS[i];
    let lg;
    if (z2 === 0) lg = model === 'dh' ? 0 : 0.1 * I;
    else if (model === 'dh') lg = -A * z2 * s;
    else if (model === 'edh') lg = (-A * z2 * s) / (1 + B * SA[i] * s);
    else if (model === 'davies') lg = -A * z2 * (s / (1 + s) - 0.3 * I);
    else lg = (-A * z2 * s) / (1 + B * SA[i] * s) + SB[i] * I;
    lnG[i] = clamp(lg * LN10, -45, 45);
  }
  const aw = Math.max(0.02, 1 - 0.017 * sm); // Garrels & Christ approximation
  return { I, aw, phi: sm > 1e-12 ? -Math.log(aw) / (MW_W * sm) : 1 };
}

// species lists used in the inner loops: derived species active per model family, and alkalinity carriers
const ACTIVE = [6, 7].map((f) => Int32Array.from(DER.map((d, j) => (d[f] ? j : -1)).filter((j) => j >= 0)));
const ALKI = Int32Array.from(ALK.map((a, s) => (a !== 0 ? s : -1)).filter((s) => s >= 0)), ALKV = Float64Array.from(ALKI, (s) => ALK[s]);
const KCACHE = new Map();
function kset(T, model) {
  const fam = model === 'pitzer' ? 5 + 1 : 7, key = fam + '|' + T;
  let k = KCACHE.get(key);
  if (!k) { if (KCACHE.size > 400) KCACHE.clear(); k = DER.map((d) => (d[fam] ? d[fam](T) : NaN)); KCACHE.set(key, k); }
  return k;
}

/** Aqueous equilibrium state at temperature T for one activity model. */
class Eq {
  constructor(T, model = 'pitzer') {
    this.T = T; this.model = ACTIVITY_MODELS[model] ? model : 'pitzer'; this.K = kset(T, this.model); this.kH = 10 ** logKH(T); this.act = ACTIVE[this.model === 'pitzer' ? 0 : 1];
    this.m = new Float64Array(NS); this.lnG = new Float64Array(NS); this._l = new Float64Array(NS); this.Kg = new Float64Array(ND); this.tot = new Float64Array(NM); this._b = new Float64Array(NM); this._hp = new Float64Array(4);
    this.aw = 1; this.phi = 1; this.I = 0; this.pH = 7.5; this.alk = 0; this.iterations = 0;
  }
  _kg() {
    const { K, Kg, lnG } = this, law = Math.log(this.aw);
    for (let j = 0; j < ND; j++) {
      if (Number.isNaN(K[j])) { Kg[j] = 0; continue; }
      const a = DA[j], b = DB[j];
      Kg[j] = Math.exp(LN10 * K[j] + (a >= 0 ? lnG[a] : 0) + (b >= 0 ? lnG[b] : 0) + DNW[j] * law - lnG[NM + 1 + j]);
    }
  }
  /** Distribute the totals over free ions and complexes at a given pH; returns total alkalinity (eq/kg). */
  _dist(pH, pCO2) {
    const { m, Kg, tot, _b: bound, _hp: hp, act } = this, na = act.length, aH = Math.exp(-LN10 * pH), gas = pCO2 != null;
    hp[0] = 1 / aH; hp[1] = 1; hp[2] = aH; hp[3] = aH * aH;
    m[IH] = aH / Math.exp(this.lnG[IH]);
    if (gas) m[IC] = (pCO2 * this.kH) / Math.exp(this.lnG[ICO2]) / (Kg[JCO2] * hp[3]);
    for (let it = 0; it < 400; it++) {
      bound.fill(0);
      for (let q = 0; q < na; q++) {
        const j = act[q], a = DA[j], b = DB[j];
        let x = Kg[j] * hp[DNH[j] + 1];
        if (a >= 0) x *= m[a];
        if (b >= 0) x *= m[b];
        m[NM + 1 + j] = x;
        if (a >= 0) bound[a] += x;
        if (b >= 0) bound[b] += x;
      }
      let dmax = 0;
      for (let i = 0; i < NM; i++) {
        if (gas && i === IC) continue;
        const t = tot[i];
        if (!(t > 0)) { m[i] = 0; continue; }
        const f = m[i], nf = (t * f) / (f + bound[i]), d = Math.abs(nf - f) / nf;
        if (d > dmax) dmax = d;
        m[i] = nf;
      }
      if (dmax < 1e-13) break;
    }
    if (gas) tot[IC] = m[IC] + bound[IC];
    let alk = 0;
    for (let q = 0; q < ALKI.length; q++) alk += ALKV[q] * m[ALKI[q]];
    return alk;
  }
  /** Solve the pH that reproduces a total alkalinity: safeguarded secant on the monotone alkalinity(pH) curve. */
  _ph(alkT, pCO2) {
    let lo = -Infinity, hi = Infinity, x0 = this.pH, f0 = this._dist(x0, pCO2) - alkT;
    if (f0 < 0) lo = x0; else hi = x0;
    let x1 = x0 + (f0 < 0 ? 0.05 : -0.05), f1 = this._dist(x1, pCO2) - alkT;
    for (let it = 0; it < 120 && f1 !== 0; it++) {
      if (f1 < 0) { if (x1 > lo) lo = x1; } else if (x1 < hi) hi = x1;
      let x2 = f1 !== f0 ? x1 - (f1 * (x1 - x0)) / (f1 - f0) : NaN;
      if (!(x2 > lo && x2 < hi)) x2 = Number.isFinite(lo) && Number.isFinite(hi) ? 0.5 * (lo + hi) : Number.isFinite(lo) ? lo + 1.5 : hi - 1.5;
      x2 = clamp(x2, -4, 17);
      const dx = Math.abs(x2 - x1);
      x0 = x1; f0 = f1; x1 = x2; f1 = this._dist(x1, pCO2) - alkT;
      if (dx < 2e-11 || hi - lo < 2e-11) break;
    }
    this.pH = x1;
    return f1 + alkT;
  }
  _act() {
    const r = this.model === 'pitzer' ? pitzer(this.m, this.T, this._l) : debye(this.model, this.m, this.T, this._l);
    this.I = r.I; this.phi = r.phi; this.aw = r.aw;
  }
  /**
   * Solve the speciation for component totals `tot` (mol/kg water).
   * o.pH fixed (with o.alkC = analysed carbonate alkalinity to back out total carbon), or o.alk = total
   * alkalinity to solve the pH; o.pCO2 (atm) imposes gas equilibrium; o.frozen keeps the activity coefficients.
   */
  run(tot, o = {}) {
    const { m, lnG, _l } = this, T = this.tot;
    T.set(tot);
    for (let i = 0; i < NM; i++) { if (!(T[i] > 0)) m[i] = 0; else if (!(m[i] > 0 && m[i] <= T[i])) m[i] = T[i]; }
    let it = 0;
    for (; it < 200; it++) {
      this._kg();
      let errC = 0;
      if (o.pH != null) {
        this.pH = o.pH; this.alk = this._dist(o.pH, o.pCO2);
        if (o.alkC != null && o.pCO2 == null) {
          for (let q = 0; q < 4; q++) {
            let ac = 0;
            for (let s = 0; s < NS; s++) if (CARB[s]) ac += ALK[s] * m[s];
            if (!(o.alkC > 0) || !(ac > 0)) { if (!(o.alkC > 0)) { T[IC] = 0; this.alk = this._dist(o.pH); } break; }
            const f = o.alkC / ac; errC = Math.abs(f - 1);
            if (errC < 1e-12) break;
            T[IC] *= f; m[IC] *= f; this.alk = this._dist(o.pH);
          }
        }
      } else this.alk = this._ph(o.alk, o.pCO2);
      if (o.frozen) break;
      this._act();
      let d = 0;
      const w = it < 40 ? 1 : 0.5;
      for (let s = 0; s < NS; s++) { const e = _l[s] - lnG[s]; if (m[s] > 0 && Math.abs(e) > d) d = Math.abs(e); lnG[s] += w * e; }
      if (d < 1e-10 && errC < 1e-10) { this._kg(); break; }
    }
    this.iterations = it;
    return this;
  }
  /** log10 activity of species index s. */
  la(s) { return Math.log10(Math.max(this.m[s], 1e-40)) + this.lnG[s] / LN10; }
  gammaOf(id) { return Math.exp(this.lnG[si(MNAME[id] || id)]); }
  molalityOf(id) { return this.m[si(MNAME[id] || id)]; }
  /** Net charge of all species, eq/kg water. */
  charge() { let c = 0; for (let s = 0; s < NS; s++) c += ZS[s] * this.m[s]; return c; }
  carbon() { let c = 0; for (let s = 0; s < NS; s++) if (CARB[s]) c += this.m[s]; return c; }
  pCO2() { return (this.m[ICO2] * Math.exp(this.lnG[ICO2])) / this.kH; }
  species() { return SID.map((id, s) => ({ id, label: CHARGE_LABEL(s), z: ZS[s], m: this.m[s], gamma: Math.exp(this.lnG[s]), a: this.m[s] * Math.exp(this.lnG[s]) })); }
}

// ---- minerals --------------------------------------------------------------------------------------
// logK(T) for dissolution into the master species (hydroxide minerals: into OH-). sigma = crystal–solution
// interfacial energy (mJ/m²), kg = growth constant (m/s at S − 1 = 1), dV = reaction volume (cm³/mol),
// siAS = saturation index that a threshold inhibitor can normally hold.
const mk = (name, formula, mw, stoich, logK, x = {}) => ({ name, formula, mw, stoich, logK, nOH: 0, nW: 0, rho: 2500, sigma: 80, kg: 1e-10, dV: 0, siAS: 0, group: 'scale', ...x, Ksp: (T = 25) => 10 ** logK(T) });
export const MINERALS = {
  calcite: mk('Calcite', 'CaCO₃', 100.087, { Ca: 1, C: 1 }, analytic(-171.9065, -0.077993, 2839.319, 71.595), { rho: 2710, sigma: 94, dV: -59.1, siAS: 1.8 }),
  aragonite: mk('Aragonite', 'CaCO₃', 100.087, { Ca: 1, C: 1 }, analytic(-171.9773, -0.077993, 2903.293, 71.595), { rho: 2930, sigma: 90, dV: -56.3, siAS: 1.8 }),
  gypsum: mk('Gypsum', 'CaSO₄·2H₂O', 172.17, { Ca: 1, SO4: 1 }, analytic(68.2401, 0, -3221.51, -25.0627), { nW: 2, rho: 2320, sigma: 40, kg: 1e-9, dV: -42.4, siAS: 0.36 }),
  anhydrite: mk('Anhydrite', 'CaSO₄', 136.14, { Ca: 1, SO4: 1 }, analytic(197.52, 0, -8669.8, -69.835), { rho: 2960, sigma: 60, kg: 3e-10, dV: -49.8, siAS: 0.36 }),
  barite: mk('Barite', 'BaSO₄', 233.39, { Ba: 1, SO4: 1 }, analytic(136.035, 0, -7680.41, -48.595), { rho: 4480, sigma: 120, kg: 3e-10, dV: -50.6, siAS: 1.78 }),
  celestite: mk('Celestite', 'SrSO₄', 183.68, { Sr: 1, SO4: 1 }, vh(-6.63, -4.3), { rho: 3960, sigma: 85, kg: 3e-10, dV: -49.7, siAS: 0.9 }),
  fluorite: mk('Fluorite', 'CaF₂', 78.07, { Ca: 1, F: 2 }, analytic(66.348, 0, -4298.2, -25.271), { rho: 3180, sigma: 140, dV: -44.7, siAS: 2.08 }),
  silica: mk('Amorphous silica', 'SiO₂(am)', 60.084, { Si: 1 }, analytic(-0.26, 0, -731, 0), { rho: 2200, sigma: 45, kg: 1e-12, siAS: 0.18 }),
  brucite: mk('Brucite', 'Mg(OH)₂', 58.32, { Mg: 1 }, vh(-10.88, -2), { nOH: 2, rho: 2370, sigma: 100, dV: -53.9 }),
  halite: mk('Halite', 'NaCl', 58.443, { Na: 1, Cl: 1 }, vh(1.57, 3.84), { rho: 2165, sigma: 38, kg: 1e-6, dV: -10.4, group: 'salt' }),
  strontianite: mk('Strontianite', 'SrCO₃', 147.63, { Sr: 1, C: 1 }, vh(-9.271, -1.7), { rho: 3760, group: 'minor' }),
  witherite: mk('Witherite', 'BaCO₃', 197.34, { Ba: 1, C: 1 }, vh(-8.562, 2.9), { rho: 4290, group: 'minor' }),
  siderite: mk('Siderite', 'FeCO₃', 115.85, { Fe: 1, C: 1 }, vh(-10.89, -10.4), { rho: 3870, group: 'minor' }),
  magnesite: mk('Magnesite', 'MgCO₃', 84.314, { Mg: 1, C: 1 }, vh(-7.834, -25.8), { rho: 2960, group: 'inhibited' }),
  dolomite: mk('Dolomite', 'CaMg(CO₃)₂', 184.40, { Ca: 1, Mg: 1, C: 2 }, vh(-17.083, -39.5), { rho: 2840, group: 'inhibited' }),
  nesquehonite: mk('Nesquehonite', 'MgCO₃·3H₂O', 138.36, { Mg: 1, C: 1 }, vh(-5.167, -24.2), { nW: 3, rho: 1850, group: 'salt' }),
  portlandite: mk('Portlandite', 'Ca(OH)₂', 74.093, { Ca: 1 }, vh(-5.19, -17.9), { nOH: 2, rho: 2230, group: 'salt' }),
  sylvite: mk('Sylvite', 'KCl', 74.551, { K: 1, Cl: 1 }, vh(0.9, 17.2), { rho: 1990, kg: 1e-6, group: 'salt' }),
  glauberite: mk('Glauberite', 'Na₂Ca(SO₄)₂', 278.18, { Na: 2, Ca: 1, SO4: 2 }, vh(-5.245, 0), { rho: 2800, group: 'salt' }),
  thenardite: mk('Thenardite', 'Na₂SO₄', 142.04, { Na: 2, SO4: 1 }, vh(-0.288, -2.4), { rho: 2660, kg: 1e-7, group: 'salt' }),
  mirabilite: mk('Mirabilite', 'Na₂SO₄·10H₂O', 322.19, { Na: 2, SO4: 1 }, vh(-1.214, 79.4), { nW: 10, rho: 1464, kg: 1e-7, group: 'salt' }),
  bloedite: mk('Bloedite', 'Na₂Mg(SO₄)₂·4H₂O', 334.47, { Na: 2, Mg: 1, SO4: 2 }, vh(-2.347, 0), { nW: 4, rho: 2230, group: 'salt' }),
  epsomite: mk('Epsomite', 'MgSO₄·7H₂O', 246.47, { Mg: 1, SO4: 1 }, vh(-1.881, 11.5), { nW: 7, rho: 1680, kg: 1e-7, group: 'salt' }),
  hexahydrite: mk('Hexahydrite', 'MgSO₄·6H₂O', 228.46, { Mg: 1, SO4: 1 }, vh(-1.635, -0.4), { nW: 6, rho: 1757, kg: 1e-7, group: 'salt' }),
  kieserite: mk('Kieserite', 'MgSO₄·H₂O', 138.38, { Mg: 1, SO4: 1 }, vh(-0.123, -60), { nW: 1, rho: 2570, group: 'salt' }),
  polyhalite: mk('Polyhalite', 'K₂MgCa₂(SO₄)₄·2H₂O', 602.94, { K: 2, Mg: 1, Ca: 2, SO4: 4 }, vh(-13.744, 0), { nW: 2, rho: 2780, group: 'salt' }),
  syngenite: mk('Syngenite', 'K₂Ca(SO₄)₂·H₂O', 328.42, { K: 2, Ca: 1, SO4: 2 }, vh(-7.448, 0), { nW: 1, rho: 2600, group: 'salt' }),
  kainite: mk('Kainite', 'KMgClSO₄·3H₂O', 248.97, { K: 1, Mg: 1, Cl: 1, SO4: 1 }, vh(-0.193, 0), { nW: 3, rho: 2150, group: 'salt' }),
  carnallite: mk('Carnallite', 'KMgCl₃·6H₂O', 277.85, { K: 1, Mg: 1, Cl: 3 }, vh(4.33, 0), { nW: 6, rho: 1600, kg: 1e-7, group: 'salt' }),
  bischofite: mk('Bischofite', 'MgCl₂·6H₂O', 203.30, { Mg: 1, Cl: 2 }, vh(4.455, 0), { nW: 6, rho: 1570, kg: 1e-7, group: 'salt' }),
};
for (const [id, M] of Object.entries(MINERALS)) {
  M.id = id; M.stoichiometry = { ...M.stoich, ...(M.nOH ? { OH: M.nOH } : {}), ...(M.nW ? { H2O: M.nW } : {}) };
  M._st = Object.entries(M.stoich).map(([k, n]) => [mi(k), n]); M._i = Int32Array.from(M._st.map((x) => x[0])); M._n = Float64Array.from(M._st.map((x) => x[1])); M._alk = 2 * (M.stoich.C || 0) + M.nOH; M._nu = sum(Object.values(M.stoich)) + M.nOH;
}
export const SCALE_MINERALS = ['calcite', 'aragonite', 'gypsum', 'anhydrite', 'barite', 'celestite', 'fluorite', 'silica', 'brucite', 'halite'];
export const EVAPORITE_MINERALS = ['calcite', 'gypsum', 'anhydrite', 'barite', 'celestite', 'silica', 'brucite', 'halite', 'glauberite', 'thenardite', 'mirabilite', 'bloedite', 'epsomite', 'hexahydrite', 'kieserite', 'polyhalite', 'syngenite', 'sylvite', 'kainite', 'carnallite', 'bischofite'];

/** Saturation index of one mineral in an equilibrium state (pressure P in bar, dk = optional log K offset). */
export function saturationIndex(eq, id, P = 1, dk = 0) {
  const M = MINERALS[id];
  const ix = M._i, nu = M._n, m = eq.m, g = eq.lnG;
  let s = 0;
  for (let q = 0; q < ix.length; q++) { const i = ix[q]; s += nu[q] * (Math.log10(m[i] > 1e-40 ? m[i] : 1e-40) + g[i] / LN10); }
  if (M.nOH) s += M.nOH * eq.la(IOH);
  if (M.nW) s += M.nW * Math.log10(eq.aw);
  const dP = M.dV ? (-M.dV * 1e-6 * (P - 1) * 1e5) / (R * tk(eq.T) * LN10) : 0; // pressure raises solubility when ΔV < 0
  return s - (M.logK(eq.T) + dP + dk);
}
const present = (eq, id) => MINERALS[id]._st.every(([i]) => eq.tot[i] > 0);
/** Saturation indices of every mineral whose components are present. */
export function saturation(eq, P = 1, dk = {}, ids = Object.keys(MINERALS)) {
  const out = {};
  for (const id of ids) if (present(eq, id)) out[id] = saturationIndex(eq, id, P, dk[id] || 0);
  return out;
}

// ---- solutions (mole basis) ------------------------------------------------------------------------
// A solution is { T, model, n: Float64Array (mol of each component), alk (eq), w (kg water), pH, eq }.
function molalBasis(ions, T) {
  const c = cloneIons(ions), t = tds(c);
  let S = t / 1000;
  for (let i = 0; i < 8; i++) S = t / density(T, Math.min(S, 400));
  const rho = density(T, Math.min(S, 400)), kgw = Math.max(0.05, (rho - t / 1000) / 1000); // kg water per litre
  const tot = new Float64Array(NM);
  for (let i = 0; i < NM; i++) { const id = MION[MAST[i]] || MAST[i]; if (i !== IC) tot[i] = c[id] / IONS[id].mw / 1000 / kgw; }
  const h = c.HCO3 / IONS.HCO3.mw / 1000 / kgw, c3 = c.CO3 / IONS.CO3.mw / 1000 / kgw;
  tot[IC] = h + c3;
  return { tot, alkC: h + 2 * c3, kgw, rho, S, tds: t };
}
const wrap = (T, model, n, alk, w, eq, extra = {}) => ({ T, model, n, alk, w, pH: eq.pH, eq, ...extra });
const totOf = (n, w) => { const t = new Float64Array(NM); for (let i = 0; i < NM; i++) t[i] = Math.max(0, n[i]) / w; return t; };

/** Build a solution from a water analysis (mg/L), temperature and measured pH. */
export function makeSolution({ ions, T = 25, pH = 8, model = 'pitzer', kgw = 1 }) {
  const b = molalBasis(ions, T), eq = new Eq(T, model).run(b.tot, { pH, alkC: b.alkC });
  return wrap(T, eq.model, eq.tot.map((x) => x * kgw), eq.alk * kgw, kgw, eq, { kgwPerL: b.kgw });
}
/** Re-equilibrate a solution: o.pH fixes the pH (alkalinity then follows), o.pCO2 opens it to a gas phase, o.T changes temperature. */
export function equilibrate(sol, o = {}) {
  const T = o.T ?? sol.T, eq = new Eq(T, o.model || sol.model);
  eq.pH = sol.pH; if (sol.eq && T === sol.T && eq.model === sol.model) { eq.lnG.set(sol.eq.lnG); eq.aw = sol.eq.aw; }
  eq.run(totOf(sol.n, sol.w), o.pH != null ? { pH: o.pH, pCO2: o.pCO2 } : { alk: sol.alk / sol.w, pCO2: o.pCO2 });
  const n = Float64Array.from(sol.n); n[IC] = eq.tot[IC] * sol.w;
  return wrap(T, eq.model, n, eq.alk * sol.w, sol.w, eq, { kgwPerL: sol.kgwPerL });
}
/**
 * Remove water by the factor cf (water-mass basis). co2: 'ro' — dissolved CO2 passes the membrane and stays at
 * its feed molality; 'closed' — all carbon is retained; 'open' — equilibrium with a gas phase at pCO2 (atm).
 * rej = average salt rejection (1 = everything retained).
 */
export function concentrateSolution(sol, cf, { co2 = 'closed', pCO2 = 4.2e-4, rej = 1 } = {}) {
  const w = sol.w / cf, keep = 1 - (1 - 1 / cf) * (1 - rej), n = sol.n.map((x) => x * keep);
  if (co2 === 'ro') n[IC] = Math.max(0, n[IC] - sol.eq.m[ICO2] * (sol.w * keep - w));
  return equilibrate({ ...sol, n, alk: sol.alk * keep, w }, co2 === 'open' ? { pCO2 } : {});
}
/** Blend two solutions; fb = share of solution b in the mixed water (0–1). */
export function mixSolutions(a, b, fb) {
  const n = a.n.map((x, i) => (x / a.w) * (1 - fb) + (b.n[i] / b.w) * fb);
  return equilibrate({ T: a.T * (1 - fb) + b.T * fb, model: a.model, n, alk: (a.alk / a.w) * (1 - fb) + (b.alk / b.w) * fb, w: 1, pH: a.pH, kgwPerL: a.kgwPerL });
}
// Reagents: equivalents of alkalinity added per mole and the components they carry.
export const REAGENTS = {
  h2so4: { name: 'Sulphuric acid H₂SO₄', mw: 98.08, alk: -2, add: { SO4: 1 } }, hcl: { name: 'Hydrochloric acid HCl', mw: 36.46, alk: -1, add: { Cl: 1 } },
  naoh: { name: 'Caustic soda NaOH', mw: 40.0, alk: 1, add: { Na: 1 } }, lime: { name: 'Hydrated lime Ca(OH)₂', mw: 74.09, alk: 2, add: { Ca: 1 } },
  soda: { name: 'Soda ash Na₂CO₃', mw: 105.99, alk: 2, add: { Na: 2, C: 1 } }, co2: { name: 'Carbon dioxide CO₂', mw: 44.01, alk: 0, add: { C: 1 } },
};
/** Add `mol` of a reagent per kg of water and re-equilibrate. */
export function doseSolution(sol, reagent, mol) {
  const r = REAGENTS[reagent], n = Float64Array.from(sol.n);
  for (const [k, nu] of Object.entries(r.add)) n[mi(k)] += nu * mol * sol.w;
  return equilibrate({ ...sol, n, alk: sol.alk + r.alk * mol * sol.w });
}

/**
 * Equilibrium precipitation (and dissolution of any `reservoir` solids) by mass action.
 * Active-set Newton iteration on the precipitated amounts ξ: every active mineral is driven to SI = 0 with
 * fully consistent activity coefficients, minerals are activated when supersaturated and dropped when their
 * solid is exhausted. Mass, alkalinity (hence charge) and water of hydration are conserved exactly.
 * Returns { sol, solids: { id: mol }, active, si0, iterations }; `hint` pre-activates minerals (e.g. from a previous step).
 */
export function precipitateSolution(sol, minerals, { pCO2 = null, reservoir = {}, P = 1, dk = {}, hint = [] } = {}) {
  const ids = minerals.filter((id) => MINERALS[id]), mins = ids.map((id) => MINERALS[id]), nK = ids.length, xi = new Float64Array(nK), res = ids.map((id) => reservoir[id] || 0);
  const n = new Float64Array(NM), tot = new Float64Array(NM), SIv = new Float64Array(nK), tol = 1e-8, E = new Eq(sol.T, sol.model);
  const ro = { alk: 0, pCO2 }, gas = pCO2 != null, dkv = ids.map((id) => dk[id] || 0);
  let alk = sol.alk, w = sol.w, evals = 0;
  E.pH = sol.pH; if (sol.eq) { E.lnG.set(sol.eq.lnG); E.aw = sol.eq.aw; E.m.set(sol.eq.m); }
  const evalAt = (x) => {
    n.set(sol.n); alk = sol.alk; w = sol.w; evals++;
    for (let k = 0; k < nK; k++) { const d = x[k]; if (d === 0) continue; const M = mins[k], ix = M._i, nu = M._n; for (let q = 0; q < ix.length; q++) n[ix[q]] -= nu[q] * d; alk -= M._alk * d; w -= M.nW * MW_W * d; }
    for (let i = 0; i < NM; i++) tot[i] = n[i] > 0 ? n[i] / w : 0;
    ro.alk = alk / w; E.run(tot, ro);
    for (let k = 0; k < nK; k++) {
      const ix = mins[k]._i;
      let ok = true;
      for (let q = 0; q < ix.length; q++) if (!(n[ix[q]] > 0) && !(gas && ix[q] === IC)) { ok = false; break; }
      SIv[k] = ok ? saturationIndex(E, ids[k], P, dkv[k]) : -99;
    }
  };
  for (let k = 0; k < nK; k++) if (res[k] > 0 && mins[k]._st.some(([i]) => !(sol.n[i] > 0))) xi[k] = -Math.min(res[k], 1e-4 * sol.w); // seed absent components
  evalAt(xi);
  const si0 = Object.fromEntries(ids.map((id, k) => [id, SIv[k]])); // saturation state before any phase change
  const active = [], isAct = new Uint8Array(nK), blocked = new Uint8Array(nK), has = (k) => xi[k] + res[k] > 1e-15;
  const add = (k) => { active.push(k); isAct[k] = 1; }, drop = (a) => { isAct[active[a]] = 0; active.splice(a, 1); };
  for (let k = 0; k < nK; k++) if (has(k) && SIv[k] > -90) add(k);
  for (const id of hint) { const k = ids.indexOf(id); if (k >= 0 && !isAct[k] && SIv[k] > tol) add(k); }
  let outer = 0, J = null, jsig = '';
  for (; outer < 60 + 30 * nK; outer++) {
    for (let a = active.length - 1; a >= 0; a--) if (!has(active[a]) && SIv[active[a]] < tol) drop(a);
    let merit0 = 0;
    for (const k of active) merit0 = Math.max(merit0, Math.abs(SIv[k]));
    if (merit0 < tol) {
      let best = -1;
      for (let k = 0; k < nK; k++) if (!isAct[k] && !blocked[k] && SIv[k] > tol && (best < 0 || SIv[k] > SIv[best])) best = k;
      if (best < 0) break;
      add(best); merit0 = SIv[best];
    }
    const nA = active.length, F0 = active.map((k) => SIv[k]), n0 = Float64Array.from(n), w0 = w, sig = active.join(',');
    if (!J || sig !== jsig) { // finite-difference Jacobian dSI/dξ; refreshed when the active set changes, Broyden-updated otherwise
      J = Array.from({ length: nA }, () => new Array(nA)); jsig = sig;
      for (let b = 0; b < nA; b++) {
        const k = active[b];
        let base = Infinity;
        for (const [i, nu] of mins[k]._st) if (n0[i] > 0) base = Math.min(base, n0[i] / nu);
        const h = 1e-6 * (Number.isFinite(base) ? base : 1e-6);
        xi[k] += h; evalAt(xi); xi[k] -= h;
        for (let a = 0; a < nA; a++) J[a][b] = (SIv[active[a]] - F0[a]) / h;
      }
    }
    let dx;
    try { dx = solveLin(J, F0.map((x) => -x)); if (!dx.every(Number.isFinite)) throw new Error('singular'); } catch { dx = F0.map((x, a) => (J[a][a] ? -x / J[a][a] : 0)); }
    // fraction-to-boundary limits: solids cannot go below their inventory, components and water stay positive
    let lam = 1, hit = -1;
    for (let b = 0; b < nA; b++) { const k = active[b], lo = -res[k]; if (dx[b] < 0 && xi[k] + lam * dx[b] < lo) { lam = Math.max(0, (lo - xi[k]) / dx[b]); hit = b; } }
    const dn = new Float64Array(NM);
    let dw = 0;
    for (let b = 0; b < nA; b++) { const M = mins[active[b]]; for (const [i, nu] of M._st) dn[i] += nu * dx[b]; dw += M.nW * MW_W * dx[b]; }
    for (let i = 0; i < NM; i++) if (dn[i] > 0 && n0[i] > 0 && !(pCO2 != null && i === IC) && lam * dn[i] > 0.97 * n0[i]) { lam = (0.97 * n0[i]) / dn[i]; hit = -1; }
    if (dw > 0 && lam * dw > 0.9 * w0) { lam = (0.9 * w0) / dw; hit = -1; }
    if (lam < 1e-13) { // an exhausted solid is asked to dissolve further: it cannot coexist with the others
      const b = hit >= 0 ? hit : 0, k = active[b];
      if (SIv[k] > tol) blocked[k] = 1;
      drop(b); J = null; evalAt(xi); continue;
    }
    const x0 = Float64Array.from(xi);
    let t = 0;
    for (; t < 14; t++) {
      for (let b = 0; b < nA; b++) { const k = active[b]; xi[k] = Math.max(-res[k], x0[k] + lam * dx[b]); }
      evalAt(xi);
      let merit = 0;
      for (const k of active) merit = Math.max(merit, Math.abs(SIv[k]));
      if (merit < merit0 || (t === 0 && hit >= 0) || t === 13) break;
      lam *= 0.5; hit = -1;
    }
    if (t > 0 || lam < 1) J = null; // damped or truncated step: rebuild the Jacobian next time
    else { // Broyden rank-one update
      let ss = 0;
      const sv = active.map((k) => xi[k] - x0[k]);
      for (const x of sv) ss += x * x;
      if (ss > 0) for (let a = 0; a < nA; a++) { let js = 0; for (let b = 0; b < nA; b++) js += J[a][b] * sv[b]; const c = (SIv[active[a]] - F0[a] - js) / ss; for (let b = 0; b < nA; b++) J[a][b] += c * sv[b]; }
    }
  }
  evalAt(xi);
  const nOut = Float64Array.from(n);
  if (pCO2 != null) nOut[IC] = E.tot[IC] * w;
  return { sol: wrap(sol.T, sol.model, nOut, alk, w, E, { kgwPerL: sol.kgwPerL }), solids: Object.fromEntries(ids.map((id, k) => [id, xi[k]])), active: active.map((k) => ids[k]), si0, iterations: outer, evals };
}

/** Composition of a solution in the water-analysis convention (mg/L), with density-based volume. */
export function solutionToIons(sol) {
  const e = sol.eq, out = cloneIons({});
  for (let i = 0; i < NM; i++) { if (i === IC) continue; const id = MION[MAST[i]] || MAST[i]; out[id] = e.tot[i] * IONS[id].mw; }
  for (let s = 0; s < NS; s++) if (CARB[s]) { if (ALK[s] === 1) out.HCO3 += e.m[s] * IONS.HCO3.mw; else if (ALK[s] === 2) out.CO3 += e.m[s] * IONS.CO3.mw; }
  const g = sum(Object.values(out)), S = (1000 * g) / (1000 + g), rho = density(sol.T, Math.min(S, 400)), kgw = (rho * (1 - S / 1000)) / 1000;
  for (const k of ION_IDS) out[k] *= kgw * 1000;
  return { ions: out, tds: g * kgw * 1000, salinity: S, density: rho, kgwPerL: kgw, gPerKgw: g };
}

/** Molality of a mineral that dissolves to saturation in a solution (mol/kg of the original water). */
export function solubility(sol, id, o = {}) {
  const r = precipitateSolution(sol, [id], { ...o, reservoir: { [id]: o.excess ?? 60 } });
  return { m: -r.solids[id] / sol.w, sol: r.sol };
}

// ---- classical indices -----------------------------------------------------------------------------
/** Langelier, Stiff–Davis, Ryznar, Puckorius, Larson–Skold and aggressive indices from a bulk analysis. */
export function scalingIndices(ions, T, pH) {
  const t = Math.max(tds(ions), 1), ca = Math.max(molar(ions, 'Ca') / 1000, 1e-12), alkEq = Math.max((molar(ions, 'HCO3') + 2 * molar(ions, 'CO3')) / 1000, 1e-12);
  const caCO3 = ca * 100087, alkCO3 = alkEq * 50043;
  const pHsL = 9.3 + (Math.log10(t) - 1) / 10 + (-13.12 * Math.log10(tk(T)) + 34.55) - (Math.log10(caCO3) - 0.4) - Math.log10(alkCO3);
  let I = 0;
  for (const k of ION_IDS) I += 0.5 * (molar(ions, k) / 1000) * IONS[k].z ** 2;
  I = Math.max(I, 1e-6);
  const K = (I <= 1.2 ? 2.022 * Math.exp((Math.log(I) + 7.544) ** 2 / 102.6) : -0.1 * I + 3.625) - 0.0002 * T * T + 0.00097 * T + 0.262; // Stiff–Davis K(I, T)
  const pHsSD = -Math.log10(ca) - Math.log10(alkEq) + K;
  const strong = (molar(ions, 'Cl') + 2 * molar(ions, 'SO4')) / 1000;
  return { lsi: pH - pHsL, sdsi: pH - pHsSD, rsi: 2 * pHsL - pH, psi: 2 * pHsL - (1.465 * Math.log10(Math.max(alkCO3, 1e-6)) + 4.54), pHs: pHsL, pHsSD, larsonSkold: strong / alkEq, aggressive: pH + Math.log10(Math.max(caCO3 * alkCO3, 1e-12)), ionicStrengthMolar: I };
}

/** Classical nucleation theory and power-law growth for a supersaturated mineral. Times in s, G in m/s. */
export function nucleationKinetics(id, SI, T, { het = 0.1, logA = 30, kgMult = 1, alpha = 1e-6 } = {}) {
  const M = MINERALS[id], S = 10 ** (SI / M._nu);
  if (!(S > 1.0001)) return null;
  const kT = KB * tk(T), vm = M.mw / 1000 / (M.rho * NA), lnS = Math.log(S), sig = M.sigma * 1e-3;
  const lnJ = LN10 * logA - (het * 16 * Math.PI * sig ** 3 * vm * vm) / (3 * kT ** 3 * lnS * lnS); // ln of nuclei per m³ per s
  const G = M.kg * kgMult * (S - 1) ** 2 * Math.exp((-45000 / R) * (1 / tk(T) - 1 / 298.15));
  const lnT = 0.25 * (Math.log((3 * alpha) / Math.PI) - lnJ - 3 * Math.log(G)); // polynuclear induction time (Kashchiev)
  return { S, lnJ, log10J: Math.max(-300, lnJ / LN10), G, tInd: Math.exp(clamp(lnT, -20, 27.6)), rCritNm: ((2 * sig * vm) / (kT * lnS)) * 1e9, flux: G * M.rho * 3.6e6 /* g/m²/h */ };
}

// ---- public mg/L-level API --------------------------------------------------------------------------
function describe(sol, P = 1, dk = {}, full = true) {
  const e = sol.eq, io = solutionToIons(sol), SI = saturation(e, P, dk), idx = scalingIndices(io.ions, sol.T, e.pH), gamma = {}, molality = {}, activities = {};
  for (let s = 0; s < NS; s++) { const id = s < NM ? MION[MAST[s]] || (MAST[s] === 'C' ? 'CO3' : MAST[s]) : SID[s]; gamma[id] = Math.exp(e.lnG[s]); molality[id] = e.m[s]; activities[id] = e.m[s] * gamma[id]; }
  const r = {
    pH: e.pH, T: sol.T, I: e.I, aw: e.aw, osmoticCoeff: e.phi, gamma, molality, activities, species: e.species(), SI,
    omega: Object.fromEntries(Object.entries(SI).map(([k, x]) => [k, 10 ** x])), lsi: idx.lsi, sdsi: idx.sdsi, rsi: idx.rsi, psi: idx.psi, larsonSkold: idx.larsonSkold, aggressiveIndex: idx.aggressive, siCalcite: SI.calcite ?? -99,
    density: io.density, viscosity: viscosity(sol.T, Math.min(io.salinity, 300)), conductivity: conductivity(io.ions, sol.T), tds: io.tds, salinity: io.salinity, ions: io.ions,
    alkalinity: (e.alk * io.kgwPerL) * 50043, dic: e.carbon(), pCO2: e.pCO2(), chargeErrorPct: chargeBalance(io.ions).errorPct, osmoticPressure: (-R * tk(sol.T) * Math.log(e.aw)) / 1.807e-5 / 1e5, hardness: hardness(io.ions), model: sol.model,
  };
  if (full) {
    // calcium-carbonate precipitation potential: CaCO3 that precipitates (+) or dissolves (−) on the way to calcite equilibrium
    r.ccpp = e.tot[mi('Ca')] > 0 || e.tot[IC] > 0 ? precipitateSolution(sol, ['calcite'], { reservoir: { calcite: 0.05 * sol.w }, P, dk }).solids.calcite / sol.w * 100087 * io.kgwPerL : 0;
  }
  return r;
}
/** Full chemical state of a water: speciation, activities, saturation and scaling/corrosion indices. */
export function analyzeWater({ ions, T = 25, pH = 8, P = 1, model = 'pitzer' }) {
  const r = describe(makeSolution({ ions, T, pH, model }), P);
  for (const id of Object.keys(MINERALS)) if (!(id in r.SI)) { r.SI[id] = -99; r.omega[id] = 0; } // −99 marks a mineral whose constituents are absent
  return r;
}
/** Concentrate a water by the factor cf with carbonate re-equilibration. Returns the new analysis (mg/L), pH and state. */
export function concentrate({ ions, T = 25, pH = 8, cf = 2, model = 'pitzer', co2 = 'ro', rej = 1, P = 1 }) {
  const s = concentrateSolution(makeSolution({ ions, T, pH, model }), Math.max(cf, 1e-6), { co2, rej }), d = describe(s, P, {}, false);
  return { ions: d.ions, pH: d.pH, T, tds: d.tds, cf, I: d.I, aw: d.aw, SI: d.SI, density: d.density, solution: s };
}
/** Equilibrium precipitation of the listed minerals. Solids in mg per litre of the original water. */
export function precipitate({ ions, T = 25, pH = 8, minerals = SCALE_MINERALS.filter((k) => k !== 'aragonite' && k !== 'anhydrite'), model = 'pitzer', P = 1 }) {
  const s0 = makeSolution({ ions, T, pH, model }), r = precipitateSolution(s0, minerals, { P }), io = solutionToIons(r.sol), f = s0.kgwPerL / s0.w;
  const solids = Object.fromEntries(Object.entries(r.solids).map(([k, x]) => [k, Math.max(0, x) * MINERALS[k].mw * 1000 * f]));
  return { solids, totalSolids: sum(Object.values(solids)), ions: io.ions, pH: r.sol.pH, tds: io.tds, SI: saturation(r.sol.eq, P), solution: r.sol, moles: r.solids };
}
/** Mean ionic activity coefficient and osmotic coefficient of a single salt solution (used for benchmarks and plots). */
export function saltActivity(cation, anion, m, { T = 25, model = 'pitzer' } = {}) {
  const tot = new Float64Array(NM), zc = MZ[mi(cation)], za = -MZ[mi(anion)], nc = za, na = zc; // electroneutral formula unit
  tot[mi(cation)] = nc * m; tot[mi(anion)] = na * m;
  const e = new Eq(T, model).run(tot, { pH: 7 });
  let lg, nu = nc + na;
  if (model === 'pitzer') lg = (nc * e.lnG[mi(cation)] + na * e.lnG[mi(anion)]) / nu;
  else lg = (nc * Math.log(e.m[mi(cation)] * Math.exp(e.lnG[mi(cation)]) / (nc * m)) + na * Math.log(e.m[mi(anion)] * Math.exp(e.lnG[mi(anion)]) / (na * m))) / nu; // stoichiometric γ± including ion pairing
  return { gamma: Math.exp(lg), phi: e.phi, aw: e.aw, I: e.I };
}
export const COMPONENTS = MAST;
export const componentIndex = mi;

// ---- suite ------------------------------------------------------------------------------------------
const MODEL_OPTS = Object.entries(ACTIVITY_MODELS).map(([value, label]) => ({ value, label }));
const NACL_LIT = { m: [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6], g: [0.778, 0.735, 0.681, 0.657, 0.668, 0.714, 0.783, 0.874, 0.986] }; // Robinson & Stokes, 25 °C
const limitsOf = (v) => { const p = (x) => Math.log10(Math.max(x, 1) / 100); return { calcite: v.limCalcite, aragonite: v.limCalcite, gypsum: p(v.limGypsum), anhydrite: p(v.limGypsum), barite: p(v.limBarite), celestite: p(v.limCelestite), fluorite: p(v.limFluorite), silica: p(v.limSilica), brucite: 0, halite: 0 }; };
const scaleSet = (v) => ['calcite', 'gypsum', ...(v.T > 50 ? ['anhydrite'] : []), 'barite', 'celestite', 'fluorite', 'silica', 'brucite', 'halite'];
const dkOf = (v) => ({ calcite: v.dkCalcite || 0, aragonite: v.dkCalcite || 0, gypsum: v.dkGypsum || 0, barite: v.dkBarite || 0, silica: v.dkSilica || 0 });
const cross = (xs, ys, thr) => { if (ys[0] >= thr) return xs[0]; for (let i = 1; i < xs.length; i++) if (ys[i] >= thr) return xs[i - 1] + ((thr - ys[i - 1]) * (xs[i] - xs[i - 1])) / (ys[i] - ys[i - 1]); return null; };
const mgL = (sol, mol, mw) => (mol / sol.w) * mw * 1000 * solutionToIons(sol).kgwPerL;

/** Acid or caustic dosing of the feed to a target pH, a target concentrate LSI, or a fixed dose. */
function applyDosing(feed, v, wallCF, copt) {
  const acid = v.acid, base = v.base, none = { sol: feed, reagent: null, mol: 0, mg: 0 };
  const pack = (sol, reagent, mol) => ({ sol, reagent, mol, mg: mol * REAGENTS[reagent].mw * 1000 * (feed.kgwPerL || 1) });
  if (v.doseMode === 'fixed') { const r = v.fixedChem, mol = Math.max(0, v.fixedDose) / REAGENTS[r].mw / 1000 / (feed.kgwPerL || 1); return mol > 0 ? pack(doseSolution(feed, r, mol), r, mol) : none; }
  if (v.doseMode === 'ph') {
    let cur = feed, tot = 0, reagent = null;
    for (let it = 0; it < 4; it++) {
      const d = (cur.alk - equilibrate(cur, { pH: v.targetPH }).alk) / cur.w; // equivalents of acid (+) or base (−) still required
      if (it === 0) reagent = d > 0 ? acid : base;
      const mol = d / -REAGENTS[reagent].alk;
      if (Math.abs(mol) < 1e-12) break;
      tot += mol; cur = doseSolution(cur, reagent, mol);
    }
    return Math.abs(tot) > 1e-12 && tot > 0 ? pack(cur, reagent, tot) : none;
  }
  if (v.doseMode === 'lsi') {
    const g = (mol) => saturationIndex(concentrateSolution(mol > 0 ? doseSolution(feed, acid, mol) : feed, wallCF, copt).eq, 'calcite', v.P, v.dkCalcite || 0) - v.targetLSI;
    if (!(feed.n[IC] > 0) || !(feed.n[mi('Ca')] > 0) || g(0) <= 0) return none;
    const top = (2 * Math.max(feed.alk, 1e-6)) / feed.w / -REAGENTS[acid].alk;
    if (g(top) > 0) return pack(doseSolution(feed, acid, top), acid, top);
    const mol = brent(g, 0, top, 1e-9);
    return pack(doseSolution(feed, acid, mol), acid, mol);
  }
  return none;
}

/** Complete scaling assessment used by run(), the mesh study and the verification checks. */
export function assessScaling(v) {
  const model = v.model, dk = dkOf(v), lim = limitsOf(v), set = scaleSet(v), P = v.P;
  let raw = makeSolution({ ions: v.ions, T: v.T, pH: v.pH, model });
  const unmixed = raw;
  let other = null;
  if (v.mixOn && tds(cloneIons(v.mixIons)) > 0 && v.mixFrac > 0) { other = makeSolution({ ions: v.mixIons, T: v.mixT, pH: v.mixPH, model }); raw = mixSolutions(raw, other, clamp(v.mixFrac / 100, 0, 1)); raw.T = raw.eq.T; }
  const R = clamp(v.recovery / 100, 0, 0.995), rej = clamp(v.rejection / 100, 0, 1), beta = Math.max(1, v.cpFactor), cfOf = (r) => 1 / (1 - r);
  const copt = { co2: v.co2, pCO2: v.pCO2 * 1e-6, rej };
  const dose = applyDosing(raw, v, cfOf(R) * beta, copt), feed = dose.sol;
  const concAt = (r, b = 1) => concentrateSolution(feed, cfOf(r) * b, copt);
  const useRo = v.useRoBrine && tds(cloneIons(v.roBrine)) > 0;
  const conc = useRo ? makeSolution({ ions: v.roBrine, T: v.T, pH: v.roBrinePH, model }) : concAt(R);
  const wall = useRo ? concentrateSolution(conc, beta, { co2: 'closed' }) : concAt(R, beta);
  // recovery sweep at the membrane wall
  const g0 = solutionToIons(feed).gPerKgw, Rmax = clamp(1 - (beta * g0) / 380, 0.3, 0.98), nRec = Math.max(4, Math.round(v.nRec));
  const Rs = linspace(0, Rmax, nRec), sweep = Rs.map((r) => saturation(concAt(r, beta).eq, P, dk, set));
  const maxRec = {};
  for (const id of set) {
    const ys = sweep.map((q) => q[id] ?? -99), here = present(feed.eq, id);
    maxRec[id] = { plain: here ? cross(Rs, ys, 0) : null, as: here ? cross(Rs, ys, Math.max(lim[id] ?? 0, 0)) : null, present: here };
  }
  const pick = (key) => { let best = null; for (const id of set) { const r = maxRec[id][key]; if (maxRec[id].present && r != null && (best == null || r < best.r)) best = { id, r }; } return best || { id: null, r: Rmax }; };
  return { v, model, dk, lim, set, P, raw, unmixed, other, feed, dose, conc, wall, R, rej, beta, Rs, Rmax, sweep, maxRec, limitPlain: pick('plain'), limitAS: pick('as'), concAt, copt, useRo, cf: cfOf(R) };
}

const D = () => Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));

const suite = {
  id: 'chem', num: 2, title: 'Brine Chemistry, Precipitation & Scaling', short: 'Brine chemistry', icon: '⚗️',
  tagline: 'Speciation, activity models up to saturated brines, saturation indices, scaling limits, dosing and precipitation.',
  description: 'Solves the full aqueous speciation of a water analysis — carbonate, borate, silicate and sulphate acid–base systems, water dissociation and ion pairs — with activity coefficients from the Pitzer ion-interaction model or four Debye–Hückel-type models. The water is concentrated along the recovery path with carbonate re-equilibration; saturation indices of every relevant mineral, the maximum recovery before each one scales (with and without antiscalant), acid or caustic doses, equilibrium precipitation masses, nucleation induction times and corrosion indices follow from the same thermodynamic state.',
  guide: [
    'Enter or pull the feed analysis, pH and temperature. A complete analysis matters: TDS alone cannot predict scaling.',
    'Set the recovery, the membrane-wall concentration factor β and, if used, the antiscalant and pH-adjustment strategy.',
    'On Model setup choose the activity model (Pitzer for anything above brackish salinity) and how dissolved CO₂ behaves.',
    'Run. Read the saturation table first, then the recovery sweep: the first mineral to cross its limit sets the maximum recovery. The concentrate is offered to the ZLD and discharge suites.',
  ],
  implemented: ['mass-action', 'law of mass action', 'component mass-balance', 'charge-balance', 'alkalinity equation', 'acid–base equilibrium', 'water-dissociation', 'henry', 'mineral-solubility-product', 'ion-activity-product', 'saturation-index', 'chemical-potential equality', 'debye–hückel equation', 'extended debye–hückel', 'davies', 'specific-ion-interaction', 'pitzer', 'setschenow', 'ion-pairing', 'aqueous complexation equations', 'precipitation/dissolution rate', 'classical nucleation theory', 'crystal-growth', 'induction-time',
    'equilibrium–kinetic precipitation', 'pitzer–speciation', 'activity–nucleation–growth', 'scaling–surface-deposition', 'speciation–corrosion', 'evaporation–speciation–precipitation', 'membrane-concentration–mineral-equilibrium',
    'initial ionic composition', 'alkalinity', 'temperature', 'pressure', 'dissolved gases', 'initial supersaturation', 'prescribed species concentration', 'equilibrium mineral boundary', 'gas–liquid equilibrium condition', 'inlet chemistry', 'fixed-temperature', 'prescribed-pressure',
    'complete ionic-speciation', 'electrolyte thermodynamics', 'activity and ionic-strength', 'acid-base equilibrium', 'ph and alkalinity', 'gas-liquid equilibrium', 'mineral saturation', 'precipitation and dissolution', 'scale identification', 'scale-quantity', 'crystallisation tendency', 'solubility modelling', 'temperature and pressure effects', 'chemical dosing', 'antiscalant assessment', 'corrosion tendency', 'brine mixing', 'reaction kinetics', 'high-salinity physical-property'],
  equationsNote: 'Pitzer parameters are the Harvie–Møller–Weare 25 °C set (Na–K–Mg–Ca–H–Cl–SO₄–HCO₃–CO₃–OH–CO₂, extended with Sr, Ba, NO₃, F and borate); away from 25 °C only the Debye–Hückel slope and the equilibrium constants change, so results are most reliable at 10–45 °C and indicative up to about 100 °C. The Debye–Hückel-type models with ion pairing are valid to an ionic strength of roughly 0.1 (limiting law 0.005, Davies 0.5, Truesdell–Jones about 1 mol/kg). pH is on the conventional single-ion activity scale without MacInnes scaling. Redox, phosphate speciation, surface complexation, ion exchange and solid solutions are not modelled; dolomite and magnesite are reported but never precipitated because they are kinetically inhibited. Induction times come from classical nucleation theory and are order-of-magnitude screening values; the antiscalant dose is a heuristic to be confirmed with the supplier.',

  inputs: [
    { group: 'Feed water', help: 'The water entering the membrane or concentration step.', fields: [
      { key: 'ions', label: 'Feed-water analysis (mg/L)', type: 'ions', value: WATERS.seawater.ions, help: 'Complete ionic analysis. Bicarbonate and carbonate define the carbonate alkalinity; the pH then fixes dissolved CO₂.' },
      { key: 'T', label: 'Temperature', unit: '°C', value: 25, min: 0, max: 120, typical: [5, 45], help: 'Solubility products, acid–base constants and the Debye–Hückel slope depend on temperature.' },
      { key: 'pH', label: 'Feed pH', unit: '', value: 8.1, min: 2, max: 12.5, help: 'Measured pH of the analysed sample.' },
      { key: 'P', label: 'Pressure', unit: 'bar', value: 60, min: 1, max: 1000, help: 'Pressure slightly raises mineral solubility (reaction-volume correction). Use the concentrate pressure.' },
      { key: 'Q', label: 'Feed flow', unit: 'm³/h', value: 1000, min: 0.01, max: 1e6, help: 'Only used to express doses and precipitation potential as mass flows.' },
    ] },
    { group: 'Concentration step', help: 'How far the water is concentrated and what the membrane wall sees.', fields: [
      { key: 'recovery', label: 'Recovery', unit: '%', value: 45, min: 0, max: 99, typical: [35, 90], help: 'Share of the feed removed as low-salinity product. Concentration factor = 1/(1 − recovery) for a fully rejecting membrane.' },
      { key: 'rejection', label: 'Average salt rejection', unit: '%', value: 99.6, min: 50, max: 100, help: 'Ions passing into the permeate are not concentrated. Use 100 % for evaporation.' },
      { key: 'cpFactor', label: 'Concentration-polarisation factor β', unit: '–', value: 1.1, min: 1, max: 2, help: 'Ratio of wall to bulk concentration in the tail element; scaling starts at the wall. Take it from the RO suite.' },
      { key: 'useRoBrine', label: 'Use the RO-suite concentrate at the design point', type: 'bool', value: false, help: 'When ticked, the concentrate composition comes from the element-by-element RO result (ion-specific rejection) instead of the uniform concentration factor.' },
      { key: 'roBrine', label: 'RO concentrate analysis (mg/L)', type: 'ions', value: WATERS.robrine.ions, showIf: (v) => v.useRoBrine },
      { key: 'roBrinePH', label: 'RO concentrate pH', unit: '', value: 7.9, min: 2, max: 12.5, showIf: (v) => v.useRoBrine },
    ] },
    { group: 'Chemical dosing', help: 'pH adjustment and scale inhibitor.', fields: [
      { key: 'antiscalant', label: 'Antiscalant (threshold inhibitor) is dosed', type: 'bool', value: true, help: 'With an inhibitor each mineral may exceed saturation up to the limits on the Model setup tab.' },
      { key: 'doseMode', label: 'pH adjustment', type: 'select', value: 'none', options: [{ value: 'none', label: 'None' }, { value: 'ph', label: 'Dose to a target feed pH' }, { value: 'lsi', label: 'Dose acid to a target concentrate calcite index' }, { value: 'fixed', label: 'Fixed dose of one chemical' }] },
      { key: 'targetPH', label: 'Target feed pH', unit: '', value: 7, min: 3, max: 11.5, showIf: (v) => v.doseMode === 'ph' },
      { key: 'targetLSI', label: 'Target calcite saturation index at the wall', unit: '', value: 0.5, min: -1, max: 2.5, showIf: (v) => v.doseMode === 'lsi' },
      { key: 'acid', label: 'Acid', type: 'select', value: 'h2so4', options: [{ value: 'h2so4', label: 'Sulphuric acid' }, { value: 'hcl', label: 'Hydrochloric acid' }], help: 'Sulphuric acid is cheaper but adds sulphate, which raises the sulphate-scale indices.', showIf: (v) => v.doseMode === 'ph' || v.doseMode === 'lsi' },
      { key: 'base', label: 'Alkali', type: 'select', value: 'naoh', options: [{ value: 'naoh', label: 'Caustic soda' }, { value: 'lime', label: 'Hydrated lime' }], showIf: (v) => v.doseMode === 'ph' },
      { key: 'fixedChem', label: 'Chemical', type: 'select', value: 'h2so4', options: Object.entries(REAGENTS).map(([value, r]) => ({ value, label: r.name })), showIf: (v) => v.doseMode === 'fixed' },
      { key: 'fixedDose', label: 'Dose (as 100 % chemical)', unit: 'mg/L', value: 20, min: 0, max: 5000, showIf: (v) => v.doseMode === 'fixed' },
    ] },
    { group: 'Stream mixing (optional)', help: 'Blend a second water into the feed before concentration, for example a recycle, a second well or an incompatible injection water.', fields: [
      { key: 'mixOn', label: 'Blend a second stream', type: 'bool', value: false },
      { key: 'mixIons', label: 'Second-stream analysis (mg/L)', type: 'ions', value: WATERS.brackish.ions, showIf: (v) => v.mixOn },
      { key: 'mixFrac', label: 'Share of second stream in the blend', unit: '%', value: 30, min: 0, max: 100, showIf: (v) => v.mixOn },
      { key: 'mixPH', label: 'Second-stream pH', unit: '', value: 7.6, min: 2, max: 12.5, showIf: (v) => v.mixOn },
      { key: 'mixT', label: 'Second-stream temperature', unit: '°C', value: 25, min: 0, max: 120, showIf: (v) => v.mixOn },
    ] },
    { group: 'Thermodynamic model', tab: 'setup', help: 'Activity-coefficient model and the boundary condition for dissolved CO₂.', fields: [
      { key: 'model', label: 'Activity model', type: 'select', value: 'pitzer', options: MODEL_OPTS, help: 'Pitzer is required for seawater and brines. The Debye–Hückel family (with explicit ion pairs) is offered for dilute waters and for cross-checking.' },
      { key: 'co2', label: 'Dissolved CO₂ during concentration', type: 'select', value: 'ro', options: [{ value: 'ro', label: 'Passes the membrane (RO/NF): CO₂ stays at feed level' }, { value: 'closed', label: 'Closed system: all carbon retained' }, { value: 'open', label: 'Open to gas phase at fixed CO₂ partial pressure' }], help: 'Controls how the pH moves as the water is concentrated.' },
      { key: 'pCO2', label: 'CO₂ partial pressure of the gas phase', unit: 'µatm', value: 420, min: 1, max: 1e6, showIf: (v) => v.co2 === 'open', help: 'Atmospheric air ≈ 420 µatm.' },
      { key: 'solids', label: 'Solids allowed to precipitate', type: 'select', value: 'scale', options: [{ value: 'scale', label: 'Common membrane scales' }, { value: 'all', label: 'Scales and evaporite salts' }], help: 'Phases considered in the equilibrium-precipitation calculations.' },
    ] },
    { group: 'Saturation limits with antiscalant', tab: 'setup', help: 'Highest supersaturation a threshold inhibitor can normally control. Without antiscalant the limit of every mineral is saturation (SI = 0).', fields: [
      { key: 'limCalcite', label: 'Calcium carbonate: saturation index', unit: 'SI', value: 1.8, min: 0, max: 3, help: 'Typical 1.8–2.5 depending on the product.' },
      { key: 'limGypsum', label: 'Calcium sulphate: ion product / Ksp', unit: '%', value: 230, min: 100, max: 600 },
      { key: 'limBarite', label: 'Barium sulphate: ion product / Ksp', unit: '%', value: 6000, min: 100, max: 20000 },
      { key: 'limCelestite', label: 'Strontium sulphate: ion product / Ksp', unit: '%', value: 800, min: 100, max: 3000 },
      { key: 'limFluorite', label: 'Calcium fluoride: ion product / Ksp', unit: '%', value: 12000, min: 100, max: 50000 },
      { key: 'limSilica', label: 'Silica: concentration / solubility', unit: '%', value: 150, min: 100, max: 300, help: '100 % without a silica dispersant.' },
    ] },
    { group: 'Solubility-product adjustments', tab: 'setup', help: 'Offsets added to log Ksp. Leave at zero unless calibrated against solubility measurements (Calibrate tab).', fields: [
      { key: 'dkCalcite', label: 'Δ log Ksp calcite', unit: '', value: 0, min: -1, max: 1 }, { key: 'dkGypsum', label: 'Δ log Ksp gypsum', unit: '', value: 0, min: -1, max: 1 },
      { key: 'dkBarite', label: 'Δ log Ksp barite', unit: '', value: 0, min: -1, max: 1 }, { key: 'dkSilica', label: 'Δ log Ksp amorphous silica', unit: '', value: 0, min: -1, max: 1 },
    ] },
    { group: 'Nucleation and growth kinetics', tab: 'setup', help: 'Initial condition: a crystal-free supersaturated solution. Classical nucleation theory with a heterogeneous-nucleation factor and a parabolic growth law.', fields: [
      { key: 'het', label: 'Heterogeneous-nucleation factor f(θ)', unit: '–', value: 0.1, min: 0.005, max: 1, help: '1 = homogeneous nucleation; 0.05–0.2 is typical on membranes and particles.' },
      { key: 'logA', label: 'Pre-exponential factor log₁₀ A', unit: 'log(m⁻³s⁻¹)', value: 30, min: 20, max: 36 },
      { key: 'kgMult', label: 'Growth-rate multiplier', unit: '×', value: 1, min: 0.001, max: 1000, help: 'Scales the built-in growth constants; antiscalants typically reduce growth by 10–100×.' },
      { key: 'tRes', label: 'Residence time of the concentrate', unit: 's', value: 120, min: 1, max: 1e6, help: 'Time the supersaturated concentrate spends in the tail elements and piping before disposal or treatment.' },
    ] },
    { group: 'Sweep resolution', tab: 'mesh', help: 'The maximum recovery of each mineral is interpolated on the recovery sweep, so its resolution is a discretisation parameter.', fields: [
      { key: 'nRec', label: 'Points in the recovery sweep', unit: '', value: 36, min: 6, max: 200, step: 1 },
      { key: 'nCF', label: 'Points in the precipitation path', unit: '', value: 12, min: 4, max: 60, step: 1 },
    ] },
  ],

  presets: [
    { name: 'Seawater RO, 45 % recovery, antiscalant only', values: {} },
    { name: 'Brackish well, 80 % recovery, acid to wall SI 1.0', values: { ions: WATERS.brackish.ions, T: 25, pH: 7.6, P: 14, Q: 250, recovery: 80, rejection: 99, cpFactor: 1.15, doseMode: 'lsi', targetLSI: 1.0 } },
    { name: 'Arabian Gulf seawater, 40 % recovery, 32 °C', values: { ions: WATERS.gulf.ions, T: 32, pH: 8.2, recovery: 40, P: 68 } },
    { name: 'Produced water blended with seawater (barite incompatibility)', values: { ions: WATERS.produced.ions, T: 40, pH: 6.8, P: 30, Q: 120, recovery: 20, rejection: 100, mixOn: true, mixIons: WATERS.seawater.ions, mixFrac: 40, mixPH: 8.1, mixT: 25, co2: 'closed' } },
    { name: 'Low-salinity river water, 90 % recovery, Davies model', values: { ions: WATERS.lowbrackish.ions, T: 20, pH: 7.8, P: 10, Q: 300, recovery: 90, rejection: 98.5, cpFactor: 1.15, model: 'davies', doseMode: 'ph', targetPH: 6.8 } },
  ],

  pull: ({ feed, outputs }) => [
    feed?.ions ? { key: 'ions', value: feed.ions, from: 'Case feed water' } : null, feed?.Q ? { key: 'Q', value: feed.Q, from: 'Case feed water' } : null,
    feed?.T != null ? { key: 'T', value: feed.T, from: 'Case feed water' } : null, feed?.pH ? { key: 'pH', value: feed.pH, from: 'Case feed water' } : null,
    outputs?.ro?.recovery ? { key: 'recovery', value: +(100 * outputs.ro.recovery).toFixed(2), from: 'RO design: recovery' } : null,
    outputs?.ro?.cpFactor ? { key: 'cpFactor', value: clamp(outputs.ro.cpFactor, 1, 2), from: 'RO design: polarisation factor β' } : null,
    outputs?.ro?.concentratePressureBar ? { key: 'P', value: outputs.ro.concentratePressureBar, from: 'RO design: concentrate pressure' } : null,
    outputs?.ro?.streams?.concentrate?.ions ? { key: 'roBrine', value: outputs.ro.streams.concentrate.ions, from: 'RO design: concentrate composition' } : null,
    outputs?.ro?.streams?.concentrate?.pH ? { key: 'roBrinePH', value: outputs.ro.streams.concentrate.pH, from: 'RO design: concentrate pH' } : null,
    outputs?.ro?.streams?.concentrate?.ions ? { key: 'useRoBrine', value: true, from: 'RO design: use element-resolved concentrate' } : null,
  ].filter(Boolean),
  site: (site) => [site?.data?.sst != null ? { key: 'T', value: site.data.sst, from: 'Sea-surface temperature at site' } : null].filter(Boolean),

  run(v, ctx) {
    const a = assessScaling(v), { feed, conc, wall, set, lim, dk, P, R, Rs, sweep } = a, W = [];
    ctx?.progress?.(0.35, 'Saturation sweeps…');
    const fd = describe(feed, P, dk), cd = describe(conc, P, dk), wd = describe(wall, P, dk, false), rawD = describe(a.raw, P, dk, false);
    const limOf = (id) => (v.antiscalant ? Math.max(lim[id] ?? 0, 0) : 0);
    const best = v.antiscalant ? a.limitAS : a.limitPlain, name = (id) => (id ? MINERALS[id].name : 'none');
    // antiscalant dose heuristic: severity = highest SI relative to its controllable limit
    let sev = 0;
    for (const id of set) if ((lim[id] ?? 0) > 0 && wd.SI[id] > 0) sev = Math.max(sev, wd.SI[id] / lim[id]);
    const keepF = 1 - R * (1 - a.rej), asDose = v.antiscalant && sev > 0 ? (clamp(2 + 6 * sev, 2, 12) * (1 - R)) / keepF : 0;
    const status = (id, s) => (s == null ? '–' : s <= 0 ? 'Undersaturated' : !v.antiscalant ? 'Scaling without inhibitor' : s <= limOf(id) ? 'Supersaturated — controlled by antiscalant' : 'Exceeds antiscalant limit');
    for (const id of set) {
      const s = wd.SI[id];
      if (s == null || s <= 0) continue;
      if (s > limOf(id)) W.push({ level: 'bad', msg: `${MINERALS[id].name} is ${fmt(10 ** s * 100, 3)} % saturated at the membrane wall (SI ${fmt(s, 3)}), above the ${v.antiscalant ? 'antiscalant limit' : 'saturation limit'} of SI ${fmt(limOf(id), 3)} — lower the recovery${id === 'calcite' ? ' or dose acid' : id === 'silica' ? ', raise the temperature or pH, or use a silica dispersant' : ''}.` });
      else W.push({ level: 'info', msg: `${MINERALS[id].name} is supersaturated at the wall (SI ${fmt(s, 3)}) but within the antiscalant limit.` });
    }
    if (Math.abs(rawD.chargeErrorPct) > 5) W.push({ level: 'warn', msg: `The feed analysis has a charge imbalance of ${fmt(rawD.chargeErrorPct, 3)} % — check the laboratory data; saturation indices inherit that error.` });
    if (v.model !== 'pitzer' && wd.I > (v.model === 'dh' ? 0.01 : v.model === 'davies' ? 0.5 : v.model === 'edh' ? 0.1 : 1)) W.push({ level: 'warn', msg: `Ionic strength ${fmt(wd.I, 3)} mol/kg is outside the validity range of the selected activity model — switch to Pitzer.` });
    if (v.T > 60 || v.T < 5) W.push({ level: 'info', msg: 'Pitzer interaction parameters are 25 °C values; at this temperature the saturation indices are indicative.' });
    if (best.r != null && R > best.r + 1e-9) W.push({ level: 'warn', msg: `Design recovery ${fmt(100 * R, 3)} % exceeds the scaling-limited recovery of ${fmt(100 * best.r, 3)} % (${name(best.id)}).` });
    if (!W.some((w) => w.level === 'bad')) W.unshift({ level: 'info', msg: v.antiscalant ? 'All minerals are within their controllable saturation limits at the membrane wall.' : 'No mineral exceeds saturation at the membrane wall.' });

    // pH sweep at constant total carbon (acid/base addition to the concentrate)
    const pHs = linspace(4, 11.5, 31), phRuns = pHs.map((x) => equilibrate(wall, { pH: x }).eq);
    const frac = phRuns.map((e) => { const c = e.carbon() || 1e-30, bt = e.tot[mi('B')] || 1e-30; let h = 0, c3 = 0; for (let s = 0; s < NS; s++) if (CARB[s]) { if (ALK[s] === 1) h += e.m[s]; else if (ALK[s] === 2) c3 += e.m[s]; } return [e.m[ICO2] / c, h / c, c3 / c, e.m[si('B(OH)4')] / bt]; });
    const hasC = wall.n[IC] > 0, hasB = wall.n[mi('B')] > 0;
    const Ts = linspace(5, 95, 13), tRuns = Ts.map((T) => saturation(equilibrate(wall, { T }).eq, P, dk, set));
    const shown = set.filter((id) => present(feed.eq, id));
    const line = (xs, rows, f = (q, id) => q[id]) => shown.map((id) => ({ name: MINERALS[id].name, x: xs, y: rows.map((q) => clamp(f(q, id) ?? -99, -8, 8)) }));
    ctx?.progress?.(0.6, 'Precipitation path…');
    // equilibrium precipitation: design point and along the concentration path
    const pset = v.solids === 'all' ? [...new Set([...EVAPORITE_MINERALS, 'fluorite'])] : set;
    const pr = precipitateSolution(conc, pset, { P, dk, pCO2: v.co2 === 'open' ? v.pCO2 * 1e-6 : null }), after = solutionToIons(pr.sol), Qc = v.Q * (1 - R);
    const solidRows = Object.entries(pr.solids).filter(([, x]) => x > 1e-12).map(([id, x]) => { const c = mgL(conc, x, MINERALS[id].mw); return [MINERALS[id].name, MINERALS[id].formula, c, (c * Qc * 24) / 1000, (x / conc.w) * 1000]; });
    const totalSolid = sum(solidRows.map((r) => r[2]));
    const nCF = Math.max(3, Math.round(v.nCF)), cfMax = Math.max(1.5, (1 / (1 - a.Rmax)) * a.beta), CFs = logspace(1, cfMax, nCF), fk = (feed.kgwPerL || 1) * 1000;
    let hint = [];
    const path = CFs.map((cf) => { const q = precipitateSolution(concentrateSolution(feed, cf, a.copt), pset, { P, dk, hint }); hint = q.active; return q; });
    const pathIds = pset.filter((id) => path.some((q) => q.solids[id] > 1e-10));
    // kinetics at the wall
    const kin = shown.map((id) => [id, wd.SI[id] > 0 ? nucleationKinetics(id, wd.SI[id], v.T, v) : null]).filter(([, k]) => k);
    const fast = kin.filter(([id, k]) => k.tInd < v.tRes && wd.SI[id] > limOf(id));
    for (const [id, k] of fast) W.push({ level: 'warn', msg: `${MINERALS[id].name}: estimated induction time ${fmt(k.tInd, 2)} s is shorter than the concentrate residence time (${v.tRes} s).` });
    // activity-coefficient comparison for NaCl
    const ms = logspace(0.001, 6, 19), gam = Object.keys(ACTIVITY_MODELS).map((mod) => ({ name: ACTIVITY_MODELS[mod].split(' (')[0].split(' +')[0], x: ms, y: ms.map((m) => Math.min(saltActivity('Na', 'Cl', m, { T: 25, model: mod }).gamma, 3)) }));
    ctx?.progress?.(0.8, 'Scaling map…');
    // scaling-margin map over recovery and feed pH
    const fx = linspace(0, a.Rmax, 13), fy = linspace(5.5, 9, 8);
    const fz = fy.map((ph) => { const f = equilibrate(a.raw, { pH: ph }); return fx.map((r) => { const q = saturation(concentrateSolution(f, a.beta / (1 - r), a.copt).eq, P, dk, set); let m = -9; for (const id of set) if (q[id] != null) m = Math.max(m, q[id] - limOf(id)); return clamp(m, -3, 3); }); });
    const mixPlot = a.other ? (() => { const fs = linspace(0, 1, 11), rows = fs.map((f) => saturation(mixSolutions(a.unmixed, a.other, f).eq, P, dk, set)); return { type: 'line', title: 'Compatibility of the two streams (before concentration)', xlabel: 'Share of second stream in the blend (%)', ylabel: 'Saturation index', series: set.filter((id) => rows.some((q) => q[id] != null)).map((id) => ({ name: MINERALS[id].name, x: fs.map((f) => 100 * f), y: rows.map((q) => clamp(q[id] ?? -8, -8, 8)) })), hlines: [{ y: 0, label: 'saturation' }], vlines: [{ x: v.mixFrac, label: 'blend' }], note: 'A maximum between the end members reveals incompatible waters (typically barium meeting sulphate).' }; })() : null;

    const brine = { Q: Qc, T: v.T, P: v.P, pH: +cd.pH.toFixed(3), tds: cd.tds, ions: Object.fromEntries(ION_IDS.map((k) => [k, +cd.ions[k].toPrecision(6)])) };
    const recStr = (r) => (r == null ? `> ${fmt(100 * a.Rmax, 3)}` : fmt(100 * r, 3));
    const allIds = Object.keys(wd.SI).filter((id) => MINERALS[id].group !== 'salt' || id === 'halite' || wd.SI[id] > -1);
    const spec = cd.species.filter((s) => s.m > 1e-12).sort((p, q) => q.m - p.m);
    const corr = fd.larsonSkold, sat = (id) => (wd.SI[id] != null ? 100 * 10 ** wd.SI[id] : 0);
    const out = {
      streams: { brine }, SI: Object.fromEntries(Object.entries(cd.SI).map(([k, x]) => [k, +x.toFixed(4)])), SIwall: Object.fromEntries(Object.entries(wd.SI).map(([k, x]) => [k, +x.toFixed(4)])),
      maxRecovery: best.r ?? a.Rmax, maxRecoveryNoAntiscalant: a.limitPlain.r ?? a.Rmax, maxRecoveryAntiscalant: a.limitAS.r ?? a.Rmax, limitingMineral: best.id || 'none', antiscalantDose: asDose, acidDose: a.dose.mg, doseChemical: a.dose.reagent || 'none',
      scalingMargin: Math.max(-9, ...set.filter((id) => wd.SI[id] != null).map((id) => wd.SI[id] - limOf(id))),
      lsi: cd.siCalcite, lsiClassic: cd.lsi, sdsi: cd.sdsi, ionicStrength: cd.I, waterActivity: cd.aw, osmoticPressureBar: cd.osmoticPressure, pHConcentrate: cd.pH, feedPH: fd.pH, precipitationPotential: totalSolid, ccpp: cd.ccpp, density: cd.density, model: v.model,
    };
    return {
      summary: `At ${fmt(100 * R, 3)} % recovery the concentrate reaches ${fmt(cd.tds / 1000, 3)} g/L TDS, pH ${fmt(cd.pH, 3)} and ionic strength ${fmt(cd.I, 3)} mol/kg; ${best.id ? `${name(best.id).toLowerCase()} is the first mineral to reach its ${v.antiscalant ? 'antiscalant' : 'saturation'} limit, at ${recStr(best.r)} % recovery` : `no mineral reaches its limit up to ${fmt(100 * a.Rmax, 3)} % recovery`}.`,
      warnings: W,
      kpis: [
        { label: 'Concentrate TDS', value: cd.tds / 1000, unit: 'g/L' }, { label: 'Concentrate pH', value: cd.pH, unit: '', help: 'From the carbonate system after concentration' },
        { label: 'Ionic strength', value: cd.I, unit: 'mol/kg' }, { label: 'Water activity', value: cd.aw, unit: '–', sig: 5 },
        { label: 'Calcite SI at wall', value: wd.SI.calcite ?? 0, unit: '', status: (wd.SI.calcite ?? -9) > limOf('calcite') ? 'bad' : (wd.SI.calcite ?? -9) > 0 ? 'warn' : 'ok', help: 'Thermodynamic Langelier index: log(IAP/Ksp) with the selected activity model' },
        { label: 'Stiff–Davis index', value: cd.sdsi, unit: '', help: 'Classical S&DSI of the bulk concentrate' }, { label: 'Ryznar index', value: cd.rsi, unit: '', help: '< 6 scale forming, > 7 corrosive' },
        { label: 'Gypsum saturation at wall', value: sat('gypsum'), unit: '%', status: (wd.SI.gypsum ?? -9) > limOf('gypsum') ? 'bad' : 'ok' }, { label: 'Barite saturation at wall', value: sat('barite'), unit: '%', status: (wd.SI.barite ?? -9) > limOf('barite') ? 'bad' : 'ok' },
        { label: 'Silica saturation at wall', value: sat('silica'), unit: '%', status: (wd.SI.silica ?? -9) > limOf('silica') ? 'bad' : 'ok' },
        { label: 'Max recovery, no antiscalant', value: recStr(a.limitPlain.r), unit: '%', help: `Limited by ${name(a.limitPlain.id)}` }, { label: 'Max recovery, with antiscalant', value: recStr(a.limitAS.r), unit: '%', status: a.limitAS.r != null && R > a.limitAS.r ? 'bad' : 'ok', help: `Limited by ${name(a.limitAS.id)}` },
        { label: 'Limiting mineral', value: name(best.id) }, { label: 'Antiscalant dose (feed)', value: asDose, unit: 'mg/L', help: 'Screening estimate' },
        { label: `${a.dose.reagent ? REAGENTS[a.dose.reagent].name.split(' ').slice(-1)[0] : 'Acid'} dose`, value: a.dose.mg, unit: 'mg/L', help: 'As 100 % chemical per litre of feed' },
        { label: 'Precipitation potential', value: totalSolid, unit: 'mg/L', help: 'Solids formed if the bulk concentrate went to equilibrium' }, { label: 'CCPP of concentrate', value: cd.ccpp, unit: 'mg/L CaCO₃' },
        { label: 'Osmotic pressure', value: cd.osmoticPressure, unit: 'bar', help: 'From the water activity' },
      ],
      recommendations: [
        best.r != null && R > best.r ? `Reduce the recovery to about ${fmt(100 * best.r - 1, 3)} % or treat for ${name(best.id).toLowerCase()} (see the recovery sweep).` : null,
        (wd.SI.calcite ?? -9) > limOf('calcite') && v.doseMode === 'none' ? 'Calcium carbonate is beyond the inhibitor limit: set pH adjustment to “target concentrate calcite index” to size the acid dose.' : null,
        !v.antiscalant && W.some((w) => w.level === 'bad') ? 'Enable antiscalant dosing: most sparingly soluble salts can be held well above saturation by a threshold inhibitor.' : null,
        corr > 1.2 && (fd.siCalcite ?? 0) < 0 ? `Larson–Skold index ${fmt(corr, 3)} with a negative calcite index: the water is corrosive to carbon steel and cement linings — specify duplex/GRP wetted parts.` : null,
        v.model !== 'pitzer' && cd.I > 0.5 ? 'Switch the activity model to Pitzer for this salinity.' : null,
        'Send the concentrate to suite 9 (ZLD) for the evaporation path and salt recovery, or to suite 5 for the discharge assessment.',
      ].filter(Boolean),
      plots: [
        { type: 'line', title: `Saturation index versus recovery (membrane wall, β = ${fmt(a.beta, 3)})`, xlabel: 'Recovery (%)', ylabel: 'Saturation index log(IAP/Ksp)', series: line(Rs.map((r) => 100 * r), sweep), hlines: [{ y: 0, label: 'saturation' }], vlines: [{ x: 100 * R, label: 'design' }, ...(best.r != null ? [{ x: 100 * best.r, label: 'limit' }] : [])], ymin: -4, ymax: 4 },
        { type: 'line', title: 'Saturation index versus pH (concentrate, constant total carbon)', xlabel: 'pH', ylabel: 'Saturation index', series: line(pHs, phRuns, (e, id) => saturationIndex(e, id, P, dk[id] || 0)), hlines: [{ y: 0, label: 'saturation' }], vlines: [{ x: wd.pH, label: 'concentrate' }], ymin: -6, ymax: 6 },
        { type: 'line', title: 'Saturation index versus temperature (concentrate)', xlabel: 'Temperature (°C)', ylabel: 'Saturation index', series: line(Ts, tRuns), hlines: [{ y: 0, label: 'saturation' }], vlines: [{ x: v.T, label: 'design' }], ymin: -4, ymax: 4, note: 'Calcium carbonate, anhydrite and brucite become less soluble on heating; silica, barite and gypsum behave the opposite way.' },
        { type: 'line', title: 'Carbonate and borate speciation versus pH (concentrate)', xlabel: 'pH', ylabel: 'Fraction of total (–)', ymin: 0, ymax: 1, series: [
          ...(hasC ? [{ name: 'CO₂(aq)', x: pHs, y: frac.map((f) => f[0]) }, { name: 'HCO₃⁻ (incl. pairs)', x: pHs, y: frac.map((f) => f[1]) }, { name: 'CO₃²⁻ (incl. pairs)', x: pHs, y: frac.map((f) => f[2]) }] : []),
          ...(hasB ? [{ name: 'B(OH)₄⁻ / total boron', x: pHs, y: frac.map((f) => f[3]), dash: true }] : []), ...(!hasC && !hasB ? [{ name: 'no carbonate or boron present', x: [4, 11.5], y: [0, 0] }] : [])], vlines: [{ x: wd.pH, label: 'concentrate' }] },
        { type: 'line', title: 'Equilibrium precipitation along the concentration path', xlabel: 'Concentration factor (×)', ylabel: 'Solids formed (g per m³ of feed)', logx: true, series: pathIds.length ? pathIds.map((id) => ({ name: MINERALS[id].name, x: CFs, y: path.map((q) => Math.max(0, q.solids[id]) / feed.w * MINERALS[id].mw * fk), mode: 'both' })) : [{ name: 'no solids form', x: CFs, y: CFs.map(() => 0) }], vlines: [{ x: a.cf, label: 'design' }], note: 'Precipitation sequence if every mineral reached equilibrium (no inhibitor, unlimited time).' },
        { type: 'line', title: 'Mean activity coefficient of NaCl by model (25 °C)', xlabel: 'Ionic strength (mol/kg)', ylabel: 'γ±', logx: true, ymin: 0, ymax: 1.6, series: [...gam, { name: 'Measured (Robinson & Stokes)', x: NACL_LIT.m, y: NACL_LIT.g, mode: 'points' }], vlines: [{ x: clamp(cd.I, 0.001, 6), label: 'this brine' }] },
        { type: 'field', title: `Scaling margin: worst SI minus its ${v.antiscalant ? 'antiscalant' : 'saturation'} limit`, xlabel: 'Recovery (%)', ylabel: 'Feed pH after adjustment', zlabel: 'margin', zunit: 'SI', x: fx.map((r) => 100 * r), y: fy, z: fz, zmin: -3, zmax: 3, cmap: 'coolwarm', contours: 8, markers: [{ x: 100 * R, y: clamp(fd.pH, 5.5, 9), label: 'design' }], note: 'Negative (blue) = every mineral within its limit; positive (red) = at least one mineral beyond it.' },
        { type: 'bar', title: 'Saturation at the membrane wall', ylabel: 'Saturation index', categories: shown.map((id) => MINERALS[id].name), series: [{ name: 'Saturation index', values: shown.map((id) => clamp(wd.SI[id] ?? -8, -8, 8)) }, { name: 'Limit', values: shown.map((id) => limOf(id)) }] },
        ...(mixPlot ? [mixPlot] : []),
      ],
      tables: [
        { title: 'Mineral saturation', columns: ['Mineral', 'Formula', 'SI feed', 'SI concentrate', 'SI at wall', 'Saturation at wall (%)', 'Limit (SI)', 'Max recovery, no antiscalant (%)', 'Max recovery, antiscalant (%)', 'Status'],
          rows: allIds.map((id) => [MINERALS[id].name, MINERALS[id].formula, fd.SI[id] ?? null, cd.SI[id] ?? null, wd.SI[id], 100 * 10 ** clamp(wd.SI[id], -12, 12), set.includes(id) ? limOf(id) : null, a.maxRec[id] ? recStr(a.maxRec[id].plain) : null, a.maxRec[id] ? recStr(a.maxRec[id].as) : null, MINERALS[id].group === 'inhibited' ? 'Kinetically inhibited — not a practical scale' : set.includes(id) ? status(id, wd.SI[id]) : wd.SI[id] > 0 ? 'Supersaturated' : 'Undersaturated']),
          note: `SI = log₁₀(ion-activity product / Ksp) with the ${ACTIVITY_MODELS[a.model]} model at ${v.T} °C and ${v.P} bar. “> x” means the limit is not reached within the sweep.` },
        { title: 'Speciation of the concentrate', columns: ['Species', 'Charge', 'Molality (mol/kg)', 'Activity coefficient γ', 'Activity', 'Share of dissolved species (%)'], rows: spec.map((s) => [s.label, s.z, s.m, s.gamma, s.a, (100 * s.m) / sum(spec.map((q) => q.m))]), note: `Water activity ${fmt(cd.aw, 5)}, osmotic coefficient ${fmt(cd.osmoticCoeff, 4)}, ionic strength ${fmt(cd.I, 4)} mol/kg, pCO₂ ${fmt(cd.pCO2 * 1e6, 3)} µatm.` },
        { title: 'Stream compositions (mg/L)', columns: ['Constituent', 'Raw feed', 'Feed after dosing', 'Concentrate', 'Concentrate after precipitation'],
          rows: [...ION_IDS.map((k) => [`${IONS[k].name} ${IONS[k].label}`, rawD.ions[k], fd.ions[k], cd.ions[k], after.ions[k]]), ['TDS', rawD.tds, fd.tds, cd.tds, after.tds], ['pH', rawD.pH, fd.pH, cd.pH, pr.sol.pH], ['Alkalinity (mg/L CaCO₃)', rawD.alkalinity, fd.alkalinity, cd.alkalinity, pr.sol.eq.alk * after.kgwPerL * 50043],
            ['Ionic strength (mol/kg)', rawD.I, fd.I, cd.I, pr.sol.eq.I], ['Density (kg/m³)', rawD.density, fd.density, cd.density, after.density], ['Viscosity (mPa·s)', rawD.viscosity * 1000, fd.viscosity * 1000, cd.viscosity * 1000, viscosity(v.T, Math.min(after.salinity, 300)) * 1000], ['Conductivity (µS/cm)', rawD.conductivity, fd.conductivity, cd.conductivity, conductivity(after.ions, v.T)], ['Osmotic pressure (bar)', rawD.osmoticPressure, fd.osmoticPressure, cd.osmoticPressure, (-R_GAS * tk(v.T) * Math.log(pr.sol.eq.aw)) / 1.807e-5 / 1e5]],
          note: 'Bicarbonate and carbonate are redistributed by the speciation at the stated pH (the carbonate alkalinity of the analysis is preserved). Hydroxide and borate alkalinity are included in the alkalinity row.' },
        { title: 'Equilibrium precipitation of the bulk concentrate', columns: ['Solid', 'Formula', 'mg per L of concentrate', 'kg/d', 'mmol/kg water'], rows: solidRows.length ? solidRows : [['No solid phase forms', '', 0, 0, 0]], note: 'Thermodynamic potential without inhibitor; the actual deposit is limited by kinetics and antiscalant.' },
        { title: 'Nucleation and growth at the membrane wall', columns: ['Mineral', 'Supersaturation ratio S', 'log₁₀ nucleation rate (m⁻³s⁻¹)', 'Critical nucleus radius (nm)', 'Induction time (s)', 'Linear growth rate (µm/d)', 'Potential deposit flux (g/m²·d)'],
          rows: kin.length ? kin.map(([id, k]) => [MINERALS[id].name, k.S, k.log10J, k.rCritNm, k.tInd >= 9e11 ? '> 10¹² (never)' : k.tInd, k.G * 8.64e10, k.flux * 24]) : [['No mineral is supersaturated', null, null, null, null, null, null]], note: 'Classical nucleation theory (order-of-magnitude). S = (IAP/Ksp)^(1/ν); induction time is the polynuclear expression t = [3α/(πJG³)]^¼ for a detectable solid fraction α = 10⁻⁶.' },
        { title: 'Scaling and corrosion indices', columns: ['Index', 'Feed', 'Concentrate', 'Interpretation'], rows: [
          ['Calcite SI (thermodynamic)', fd.siCalcite, cd.siCalcite, '> 0 scale forming, < 0 dissolving'], ['Langelier index (classical)', fd.lsi, cd.lsi, 'Valid below about 10 g/L TDS'], ['Stiff–Davis index', fd.sdsi, cd.sdsi, 'High-salinity form of the Langelier index'],
          ['Ryznar stability index', fd.rsi, cd.rsi, '< 6 scaling, 6–7 balanced, > 7 corrosive'], ['Puckorius index', fd.psi, cd.psi, '< 6 scaling, > 7 corrosive'], ['Larson–Skold index', fd.larsonSkold, cd.larsonSkold, '> 1.2 high corrosion rate on steel'], ['Aggressive index', fd.aggressiveIndex, cd.aggressiveIndex, '< 10 aggressive, > 12 non-aggressive'], ['CCPP (mg/L CaCO₃)', fd.ccpp, cd.ccpp, 'Mass of CaCO₃ that would precipitate (+) or dissolve (−)']] },
      ],
      balances: (() => {
        const iCa = mi('Ca'), iS = mi('SO4'), caS = sum(Object.entries(pr.solids).map(([id, x]) => x * (MINERALS[id].stoich.Ca || 0))), sS = sum(Object.entries(pr.solids).map(([id, x]) => x * (MINERALS[id].stoich.SO4 || 0)));
        const cS = sum(Object.entries(pr.solids).map(([id, x]) => x * (MINERALS[id].stoich.C || 0))), keep = a.useRo ? 1 : 1 - (1 - 1 / a.cf) * (1 - a.rej);
        return [
          { name: 'Calcium through precipitation (mmol)', in: conc.n[iCa] * 1000, out: (pr.sol.n[iCa] + caS) * 1000 }, { name: 'Sulphate through precipitation (mmol)', in: conc.n[iS] * 1000, out: (pr.sol.n[iS] + sS) * 1000 },
          { name: 'Inorganic carbon through precipitation (mmol)', in: conc.n[IC] * 1000, out: v.co2 === 'open' ? conc.n[IC] * 1000 : (pr.sol.n[IC] + cS) * 1000 },
          { name: 'Net charge through precipitation (meq)', in: conc.eq.charge() * conc.w * 1000 + 1, out: pr.sol.eq.charge() * pr.sol.w * 1000 + 1 },
          { name: 'Chloride through concentration (mmol per kg feed water)', in: feed.n[mi('Cl')] * 1000, out: a.useRo ? feed.n[mi('Cl')] * 1000 : (conc.n[mi('Cl')] + feed.n[mi('Cl')] * (1 - keep)) * 1000 },
          { name: 'Alkalinity through dosing (meq/kg)', in: (a.raw.alk / a.raw.w) * 1000 + (a.dose.reagent ? REAGENTS[a.dose.reagent].alk * a.dose.mol * 1000 : 0), out: (feed.alk / feed.w) * 1000 },
        ];
      })(),
      outputs: out,
    };
  },

  mesh: { name: 'Recovery-sweep resolution', keys: ['nRec'], min: 6, note: 'Scaling-limited recoveries are interpolated between sweep points; the study refines the sweep and reports the numerical uncertainty of those limits.',
    metrics: [{ label: 'Max recovery with antiscalant', unit: '–', get: (r) => r.outputs.maxRecoveryAntiscalant }, { label: 'Max recovery without antiscalant', unit: '–', get: (r) => r.outputs.maxRecoveryNoAntiscalant }] },

  calibration: {
    note: 'Fit the solubility-product offsets to laboratory solubility data. Each row is one equilibrium experiment in an NaCl background solution: gypsum is equilibrated in a closed vessel, calcite under a fixed CO₂ partial pressure. The measured dissolved calcium is compared with the model. Validate with experiments at other salinities and temperatures.',
    params: [{ key: 'dkGypsum', label: 'Δ log Ksp gypsum', lo: -0.4, hi: 0.4 }, { key: 'dkCalcite', label: 'Δ log Ksp calcite', lo: -0.4, hi: 0.4 }],
    columns: [{ key: 'mNaCl', label: 'NaCl background', unit: 'mol/kg' }, { key: 'Tc', label: 'Temperature', unit: '°C' }, { key: 'pCO2x', label: 'CO₂ pressure', unit: 'atm' }, { key: 'sGyp', label: 'Gypsum solubility', unit: 'mmol/kg' }, { key: 'sCal', label: 'Calcite solubility', unit: 'mmol/kg' }],
    targets: [{ key: 'sGyp', label: 'Gypsum solubility', unit: 'mmol/kg' }, { key: 'sCal', label: 'Calcite solubility', unit: 'mmol/kg' }],
    model(v) {
      const n = new Float64Array(NM), m = Math.max(0, v.mNaCl ?? 0), T = v.Tc ?? 25;
      n[mi('Na')] = m; n[mi('Cl')] = m;
      const base = equilibrate({ T, model: v.model, n, alk: 0, w: 1, pH: 7 });
      return { sGyp: solubility(base, 'gypsum', { dk: { gypsum: v.dkGypsum || 0 }, excess: 0.3 }).m * 1000, sCal: solubility(base, 'calcite', { pCO2: Math.max(1e-6, v.pCO2x ?? 4.2e-4), dk: { calcite: v.dkCalcite || 0 }, excess: 0.3 }).m * 1000 };
    },
    get sample() { return (this._s ||= synth(5, [[0, 25, 0.01], [0.25, 25, 0.01], [0.5, 25, 0.03], [1, 25, 0.03], [2, 25, 0.1], [3, 25, 0.1], [4, 25, 0.3], [0.5, 35, 0.3]])); },
    get validationSample() { return (this._v ||= synth(17, [[0.1, 25, 0.05], [0.75, 30, 0.05], [1.5, 25, 0.2], [2.5, 20, 0.02], [3.5, 25, 0.5], [5, 25, 0.1]])); },
  },

  verify() {
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    const n1 = saltActivity('Na', 'Cl', 1), n6 = saltActivity('Na', 'Cl', 6), ca = saltActivity('Ca', 'Cl', 1);
    add('NaCl 1 mol/kg: mean activity coefficient', 0.657, n1.gamma, 0.004, 'Pitzer model against Robinson & Stokes (25 °C)');
    add('NaCl 1 mol/kg: osmotic coefficient', 0.936, n1.phi, 0.003, 'Pitzer model against Robinson & Stokes');
    add('NaCl 6 mol/kg: mean activity coefficient', 0.986, n6.gamma, 0.01, 'Near halite saturation');
    add('CaCl₂ 1 mol/kg: mean activity coefficient', 0.5, ca.gamma, 0.01, '2:1 electrolyte benchmark');
    const dil = saltActivity('Na', 'Cl', 1e-4), lim = 10 ** (-(3 * aphi(25)) / LN10 * 0.01);
    add('Dilute limit: Pitzer → Debye–Hückel limiting law', lim, dil.gamma, 2e-4, 'γ± at 10⁻⁴ mol/kg equals 10^(−A√I)');
    add('Dilute limit: Davies and Pitzer agree', 0, saltActivity('Na', 'Cl', 1e-3, { model: 'davies' }).gamma - saltActivity('Na', 'Cl', 1e-3).gamma, 1e-3, 'Cross-check of formulations where both are valid');
    const pure = makeSolution({ ions: cloneIons({}), T: 25, pH: 7 });
    add('Pure water at 25 °C is neutral', 6.998, equilibrate(pure).pH, 0.005, 'pH = ½ pKw from the alkalinity balance [OH⁻] = [H⁺]');
    add('Halite solubility in water, 25 °C', 6.15, solubility(pure, 'halite', { excess: 12 }).m, 0.1, 'mol/kg; literature 6.15');
    add('Gypsum solubility in water, 25 °C', 15.1, solubility(pure, 'gypsum', { excess: 0.5 }).m * 1000, 0.4, 'mmol/kg; literature 15.1–15.3');
    const cc = solubility(pure, 'calcite', { pCO2: 10 ** -3.5, excess: 0.1 });
    add('Calcite + water + air (pCO₂ = 10⁻³·⁵ atm): pH', 8.28, cc.sol.pH, 0.05, 'Classical open-system benchmark (Henry’s law + carbonate equilibria)');
    const sw = makeSolution({ ions: WATERS.seawater.ions, T: 25, pH: 8.1 }), swSI = saturation(sw.eq);
    add('Seawater calcite saturation state Ω at pH 8.1', 5, 10 ** swSI.calcite, 1, 'Surface seawater is 4–6 times supersaturated');
    add('Seawater water activity', 0.9814, sw.eq.aw, 0.0008, 'S = 35 g/kg at 25 °C');
    add('Seawater is undersaturated in gypsum', 1, swSI.gypsum < 0 && swSI.gypsum > -1 ? 1 : 0, 0, 'Ω ≈ 0.2');
    const io = solutionToIons(sw), W0 = WATERS.seawater.ions, alkOf = (c) => c.HCO3 / IONS.HCO3.mw + (2 * c.CO3) / IONS.CO3.mw;
    const err = Math.max(Math.abs(alkOf(io.ions) / alkOf(W0) - 1), ...ION_IDS.filter((k) => k !== 'HCO3' && k !== 'CO3').map((k) => Math.abs(io.ions[k] - W0[k]) / Math.max(1, W0[k])));
    add('Analysis → molal speciation → analysis round trip', 0, err, 2e-3, 'Largest relative error of any ion and of the carbonate alkalinity (mg/L basis)');
    const brine = concentrateSolution(sw, 4.2, { co2: 'closed' }), pr = precipitateSolution(brine, ['calcite', 'gypsum', 'barite', 'celestite']);
    const caS = pr.solids.calcite + pr.solids.gypsum;
    add('Calcium is conserved through precipitation', 0, (brine.n[mi('Ca')] - pr.sol.n[mi('Ca')] - caS) / brine.n[mi('Ca')], 1e-10, 'Dissolved before = dissolved after + in solids');
    add('Charge is conserved through precipitation', 0, (brine.eq.charge() * brine.w - pr.sol.eq.charge() * pr.sol.w) / brine.eq.I, 1e-8, 'Only neutral salts leave the solution');
    add('Precipitated minerals end exactly at saturation', 0, Math.max(...['calcite', 'gypsum'].map((id) => Math.abs(saturationIndex(pr.sol.eq, id)))), 1e-6, 'SI = 0 for every solid present');
    let first = null;
    for (let cf = 8; cf < 14 && first == null; cf += 0.25) if (saturationIndex(precipitateSolution(concentrateSolution(sw, cf, { co2: 'open' }), ['calcite', 'gypsum'], { pCO2: 4.2e-4 }).sol.eq, 'halite') >= 0) first = cf;
    add('Seawater evaporation: halite appears at a concentration factor of about 10.8', 10.8, first ?? 0, 0.6, 'Harvie–Møller–Weare evaporation sequence');
    const e = equilibrate(sw, { pH: -logK1(25) + 0.2 }).eq;
    add('Carbonate fractions sum to one', 1, (e.carbon()) / e.tot[IC], 1e-10, 'CO₂ + HCO₃⁻ + CO₃²⁻ + ion pairs = total inorganic carbon');
    const d = D(), base = assessScaling(d), acid = assessScaling({ ...d, doseMode: 'ph', targetPH: 6.5 });
    add('Acid dosing reaches the target pH', 6.5, acid.feed.eq.pH, 1e-3, 'Dose solved from the alkalinity difference at constant total carbon');
    add('Acid dosing lowers the calcite index', 1, saturationIndex(acid.wall.eq, 'calcite') < saturationIndex(base.wall.eq, 'calcite') - 0.5 ? 1 : 0, 0, 'Qualitative limiting behaviour');
    return C;
  },
};
const R_GAS = R;

/** Synthetic laboratory data: the model with slightly shifted solubility products plus deterministic noise. */
function synth(seed, pts) {
  const d = D(), g = rng(seed);
  return pts.map(([mNaCl, Tc, pCO2x]) => {
    const m = suite.calibration.model({ ...d, dkGypsum: 0.035, dkCalcite: -0.06, mNaCl, Tc, pCO2x });
    return { mNaCl, Tc, pCO2x, sGyp: +(m.sGyp * (1 + g.normal(0, 0.012))).toFixed(2), sCal: +(m.sCal * (1 + g.normal(0, 0.015))).toFixed(3) };
  });
}

export default suite;
