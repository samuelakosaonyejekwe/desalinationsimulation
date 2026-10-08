// Suite 2 — Brine chemistry, precipitation and scaling.
// Aqueous speciation (carbonate, borate, silicate, sulphate and fluoride acid–base systems, water
// dissociation, ion pairs) solved by mass action with mass and alkalinity balances; activity
// coefficients from Debye–Hückel, extended Debye–Hückel, Davies, Truesdell–Jones or the Pitzer
// ion-interaction model (Harvie–Møller–Weare 25 °C parameter set), Bromley or SIT; temperature- and pressure-dependent
// solubility products; saturation indices; equilibrium precipitation; concentration paths; chemical
// dosing; nucleation and growth kinetics; scaling and corrosion indices.
import { brent, clamp, linspace, logspace, sum, rng, fmt, interp1, tridiag, trapz, solveLinear as solveLin } from '../core/num.js';
import { density, viscosity, diffusivityNaCl, osmoticPressure, R, KELVIN } from '../core/props.js';
import { IONS, ION_IDS, WATERS, cloneIons, tds, chargeBalance, conductivity, hardness, molar } from '../core/water.js';
import { solveChannel, buildMask, channel1D, yGrid } from './s04_cfd.js';

const LN10 = Math.LN10, MW_W = 0.0180153, KB = 1.380649e-23, NA = 6.02214076e23;
const tk = (T) => T + KELVIN;
const analytic = (a, b, c, d, e = 0) => (T) => { const K = tk(T); return a + b * K + c / K + d * Math.log10(K) + e / (K * K); };
/** van't Hoff extrapolation of log K from 25 °C with a constant reaction enthalpy dH (kJ/mol). */
const vh = (logK25, dH = 0) => (T) => logK25 - ((dH * 1000) / (R * LN10)) * (1 / tk(T) - 1 / 298.15);

// Acid–base and gas constants: Plummer & Busenberg (1982) carbonate system as tabulated in the USGS WATEQ4F
// database (wateq4f.dat, Ball & Nordstrom 1991); every coefficient below was compared with that file.
const logK1 = analytic(-356.3094, -0.06091964, 21834.37, 126.8339, -1684915); // CO2(aq) + H2O = H+ + HCO3-
const logK2 = analytic(-107.8871, -0.03252849, 5151.79, 38.92561, -563713.9); // HCO3- = H+ + CO3 2-
const logKw = analytic(-283.971, -0.05069842, 13323.0, 102.24447, -1119669); // H2O = H+ + OH-
const logKH = analytic(108.3865, 0.01985076, -6919.53, -40.45154, 669365); // CO2(g) = CO2(aq), mol/kg/atm
const logKb = vh(-9.236, 13.5); // B(OH)3 + H2O = B(OH)4- + H+ (MINTEQA2 v4: −9.236; ΔH from WATEQ4F, 3.224 kcal/mol)
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
// Ion-pair family: WATEQ4F (wateq4f.dat) log K and ΔH (CaOH⁺ ΔH from MINTEQA2 v4). Pitzer family: the constants that
// belong to the Harvie–Møller–Weare (1984) parameter set as distributed in the USGS PHRQPITZ/PHREEQC pitzer.dat
// (pK₂ 10.3393, pK₁ + pK₂ 16.6767, HSO₄⁻ 1.979, MgOH⁺ −11.809, CaCO₃° 3.151, MgCO₃° 2.928) — the Plummer–Busenberg
// temperature functions shifted to the HMW 25 °C values. Mixing the two sets would bias carbonate saturation by 0.08 log units.
const kc1 = (T) => -logK2(T), kc2 = (T) => -logK1(T) - logK2(T), kso4 = vh(1.988, 16.1), khf = vh(3.18, 13.3);
const kc1p = analytic(107.8975, 0.03252849, -5151.79, -38.92561, 563713.9), kc2p = analytic(464.1925, 0.09344813, -26986.16, -165.75951, 2248628.9), kso4p = vh(1.979, 16.1);
const DER = [
  ['OH', -1, '', '', -1, 1, logKw, logKw], ['HCO3', -1, 'C', '', 1, 0, kc1p, kc1], ['CO2', 0, 'C', '', 2, -1, kc2p, kc2],
  ['B(OH)4', -1, 'B', '', -1, 1, logKb, logKb], ['H3SiO4', -1, 'Si', '', -1, 0, logKsi, logKsi], ['HSO4', -1, 'SO4', '', 1, 0, kso4p, kso4], ['HF', 0, 'F', '', 1, 0, khf, khf],
  ['MgOH', 1, 'Mg', '', -1, 1, vh(-11.809, 64.5), vh(-11.44, 66.7)], ['CaCO3°', 0, 'Ca', 'C', 0, 0, vh(3.151, 14.8), vh(3.224, 14.8)], ['MgCO3°', 0, 'Mg', 'C', 0, 0, vh(2.928, 10.6), vh(2.98, 11.35)],
  ['CaOH', 1, 'Ca', '', -1, 1, null, vh(-12.78, 64.1)], ['CaSO4°', 0, 'Ca', 'SO4', 0, 0, null, vh(2.3, 6.9)], ['MgSO4°', 0, 'Mg', 'SO4', 0, 0, null, vh(2.37, 19)],
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
const CHARGE_LABEL = (s) => (ZS[s] === 0 ? SID[s] : SID[s] + SUP[ZS[s]]), LABELS = SID.map((_, s) => CHARGE_LABEL(s));

// ---- activity models -------------------------------------------------------------------------------
export const ACTIVITY_MODELS = { pitzer: 'Pitzer ion interaction (Harvie–Møller–Weare)', bromley: 'Bromley (strong electrolytes, to about 6 mol/kg)', sit: 'Specific ion interaction (SIT) + ion pairs', tj: 'Truesdell–Jones / B-dot + ion pairs', davies: 'Davies + ion pairs', edh: 'Extended Debye–Hückel + ion pairs', dh: 'Debye–Hückel limiting law + ion pairs' };
/** Debye–Hückel osmotic slope Aφ (kg½/mol½), quadratic through the 0, 25 and 100 °C values (0.3767, 0.3915, 0.4605: the grid of the LLNL EQ3/6 Pitzer file data0.ypf, which the fit reproduces within 0.0005 up to 100 °C). */
const aphi = (T) => 0.3767 + 5.087e-4 * T + 3.333e-6 * T * T;
// Ion-size parameter å (Å) and Truesdell–Jones b for the extended Debye–Hückel forms: the −gamma entries of the USGS
// WATEQ4F database (wateq4f.dat; Truesdell & Jones 1974), all eighteen compared with that file. Species without an
// entry there (ion pairs, borate, silicate) use å = 4, b = 0.041 — a model assumption, not a tabulated value.
const SIZE = { Na: [4, 0.075], K: [3.5, 0.015], Ca: [5, 0.165], Mg: [5.5, 0.2], Ba: [5, 0], Sr: [5.26, 0.121], NH4: [2.5, 0], Fe: [6, 0], Mn: [6, 0], Cl: [3.5, 0.015], SO4: [5, -0.04], NO3: [3, 0], F: [3.5, 0], HPO4: [5, 0], CO3: [5.4, 0], H: [9, 0], OH: [3.5, 0], HCO3: [5.4, 0] };
const SA = SID.map((id) => (SIZE[id] || [4, 0.041])[0]), SB = SID.map((id) => (SIZE[id] || [4, 0.041])[1]);

// Pitzer parameters at 25 °C. Every entry was compared with a database file:
//  · Na–K–Mg–Ca–H–Cl–SO4–OH–HCO3–CO3–CO2 system: Harvie, Møller & Weare (1984) as held in the LLNL EQ3/6 file
//    data0.hmw (β0, β1, β2, Cφ, θ, ψ and the CO2 λ: 229 numbers, all identical);
//  · Sr and Ba chlorides, Sr sulphate, borate (Felmy & Weare 1986): USGS PHREEQC pitzer.dat (PHRQPITZ lineage);
//  · fluorides, KNO3, Mg(NO3)2, phosphates and θ(Cl,NO3) (Pitzer 1991 tabulation), NaNO3 and Ca(NO3)2 (refits with
//    α1 = 2): LLNL EQ3/6 Yucca Mountain Pitzer file data0.ypf (25 °C terms);
//  · silica λ: PHREEQC pitzer.dat (Appelo 2015).
// Analogue assignments (no measured set in these files for the suite's species list): Ba–SO4 uses the Ca–SO4 set,
// NH4 uses K, Fe/Mn use Mg, H3SiO4 uses HCO3.
const PZ_ID = { NH4: 'K', Fe: 'Mg', Mn: 'Mg', H3SiO4: 'HCO3' };
const PZ_BIN = 'Na Cl .0765 .2664 0 .00127|Na SO4 .01958 1.113 0 .00497|Na HSO4 .0454 .398 0 0|Na OH .0864 .253 0 .0044|Na HCO3 .0277 .0411 0 0|Na CO3 .0399 1.389 0 .0044|Na NO3 .00357079 .231963 0 -.0000415038|Na F .0215 .2107 0 0|Na B(OH)4 -.0427 .089 0 .0114|Na HPO4 -.0583 1.4655 0 .02938|'
  + 'K Cl .04835 .2122 0 -.00084|K SO4 .04995 .7793 0 0|K HSO4 -.0003 .1735 0 0|K OH .1298 .32 0 .0041|K HCO3 .0296 -.013 0 -.008|K CO3 .1488 1.43 0 -.0015|K NO3 -.0816 .0494 0 .0066|K F .08089 .2021 0 .00093|K B(OH)4 .035 .14 0 0|K HPO4 .0248 1.2743 0 .016387|'
  + 'Ca Cl .3159 1.614 0 -.00034|Ca SO4 .2 3.1973 -54.24 0|Ca HSO4 .2145 2.53 0 0|Ca OH -.1747 -.2303 -5.72 0|Ca HCO3 .4 2.977 0 0|Ca NO3 .14844 2.44408 0 -.0041168|'
  + 'Mg Cl .35235 1.6815 0 .00519|Mg SO4 .221 3.343 -37.23 .025|Mg HSO4 .4746 1.729 0 0|Mg HCO3 .329 .6072 0 0|Mg NO3 .3671 1.5848 0 -.020625|MgOH Cl -.1 1.658 0 0|'
  + 'Sr Cl .2858 1.667 0 -.0013|Sr SO4 .2 3.1973 -54.24 0|Ba Cl .2628 1.49625 0 -.0193782|Ba SO4 .2 3.1973 -54.24 0|H Cl .1775 .2945 0 .0008|H SO4 .0298 0 0 .0438|H HSO4 .2065 .5556 0 0';
const PZ_THETA = 'Na K -.012|Na Ca .07|Na Mg .07|Na H .036|K Ca .032|K H .005|Ca Mg .007|Ca H .092|Mg H .1|Cl SO4 .02|Cl HSO4 -.006|Cl OH -.05|Cl HCO3 .03|Cl CO3 -.02|SO4 OH -.013|SO4 HCO3 .01|SO4 CO3 .02|OH CO3 .1|HCO3 CO3 -.04|Cl NO3 .016';
const PZ_PSI = 'Na K Cl -.0018|Na K SO4 -.01|Na K HCO3 -.003|Na K CO3 .003|Na Ca Cl -.007|Na Ca SO4 -.055|Na Mg Cl -.012|Na Mg SO4 -.015|Na H Cl -.004|Na H HSO4 -.0129|K Ca Cl -.025|K Mg Cl -.022|K Mg SO4 -.048|K H Cl -.011|K H SO4 .197|K H HSO4 -.0265|'
  + 'Ca Mg Cl -.012|Ca Mg SO4 .024|Ca H Cl -.015|Mg MgOH Cl .028|Mg H Cl -.011|Mg H HSO4 -.0178|Cl SO4 Na .0014|Cl SO4 Ca -.018|Cl SO4 Mg -.004|Cl HSO4 Na -.006|Cl HSO4 H .013|Cl OH Na -.006|Cl OH K -.006|Cl OH Ca -.025|Cl HCO3 Na -.015|Cl HCO3 Mg -.096|'
  + 'Cl CO3 Na .0085|Cl CO3 K .004|SO4 HSO4 Na -.0094|SO4 HSO4 K -.0677|SO4 HSO4 Mg -.0425|SO4 OH Na -.009|SO4 OH K -.05|SO4 HCO3 Na -.005|SO4 HCO3 Mg -.161|SO4 CO3 Na -.005|SO4 CO3 K -.009|OH CO3 Na -.017|OH CO3 K -.01|HCO3 CO3 Na .002|HCO3 CO3 K .012';
const PZ_LAM = 'CO2 Na .1|CO2 K .051|CO2 Ca .183|CO2 Mg .183|CO2 Cl -.005|CO2 SO4 .097|CO2 HSO4 -.003|B(OH)3 Na -.097|B(OH)3 K -.14|B(OH)3 Cl .091|B(OH)3 SO4 .018|SiO2 Na .0566|SiO2 K .0298|SiO2 Mg .238|SiO2 Ca .238|SiO2 SO4 -.085';
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
  if (model === 'davies' && sm > 1e-12) { // osmotic coefficient that is Gibbs–Duhem consistent with the Davies coefficients
    const phi = 1 + (A * LN10 * (-2 * dhOsm(s, 1) + 0.3 * I * I)) / sm;
    return { I, aw: Math.max(0.02, Math.exp(-phi * sm * MW_W)), phi };
  }
  const aw = Math.max(0.02, 1 - 0.017 * sm); // Garrels & Christ approximation
  return { I, aw, phi: sm > 1e-12 ? -Math.log(aw) / (MW_W * sm) : 1 };
}
/** Debye–Hückel part of Σm·(1 − φ)/(2A′): [x − 2 ln x − 1/x]/b³ with x = 1 + b√I (series below √I·b = 0.01). */
function dhOsm(s, b) { const y = b * s; return y < 0.01 ? (s * s * s) / 3 - (b * s * s * s * s) / 2 : (1 + y - 2 * Math.log(1 + y) - 1 / (1 + y)) / (b * b * b); }

// Bromley (1973) and the Brønsted–Guggenheim–Scatchard specific-ion-interaction (SIT) model.
// SIT ε(i,k), kg/mol at 25 °C: OECD-NEA thermochemical database, 2020 update of the SIT tables (Tables B-6/B-7), each
// value also compared with the ThermoChimie sit.dat distributed with USGS PHREEQC where that file lists the pair.
// Sr–Cl is not in either table and takes the Ca–Cl value as an analogue. Unlisted pairs use ε = 0.
// Bromley salt constants B: twelve of the fifteen (all but SrCl2, BaCl2 and MgSO4) were re-derived here by least
// squares from the NIST activity-coefficient tables (Hamer & Wu 1972; Goldberg & Nuttall 1978; Goldberg 1981) and agree
// within 0.003 kg/mol (K2SO4, with data only to 0.7 mol/kg, within 0.013). Bromley's individual-ion table
// (B = B₊ + B₋ + δ₊δ₋, used only for salts without a fitted constant) could not be compared with a retrievable copy
// of the 1973 paper and is reported as unconfirmed in the provenance table; the Bromley model is opt-in.
// Both models return the osmotic coefficient that satisfies the Gibbs–Duhem equation with their activity coefficients.
const PFAM = { pitzer: 1, bromley: 1 }; // model families that use the strong-electrolyte species set (no sulphate ion pairs)
const BR_ION = { H: [0.0875, 0.103], Na: [0, 0.028], K: [-0.0452, -0.079], NH4: [-0.042, -0.02], Mg: [0.057, 0.157], Ca: [0.0374, 0.119], Sr: [0.0245, 0.11], Ba: [0.0022, 0.098], Mn: [0.037, 0.21], Fe: [0.046, 0.21], F: [0.0295, -0.93], Cl: [0.0643, -0.067], NO3: [-0.025, 0.27], OH: [0.076, -1], SO4: [0, -0.4], CO3: [0.028, -0.67], HPO4: [-0.01, -0.57] };
const BR_SALT = 'Na Cl .0574|K Cl .024|H Cl .1433|Ca Cl .0948|Mg Cl .1129|Sr Cl .0847|Ba Cl .0638|NH4 Cl .02|Na SO4 -.0204|K SO4 -.032|Mg SO4 -.0153|Na NO3 -.0128|K NO3 -.0862|Na OH .0747|K OH .1131';
const SIT_EPS = 'Na Cl .03|K Cl 0|H Cl .12|NH4 Cl -.01|Ca Cl .14|Sr Cl .14|Mg Cl .19|Ba Cl .07|Na SO4 -.12|K SO4 -.06|Na HSO4 -.01|Na NO3 -.04|K NO3 -.11|H NO3 .07|Ca NO3 .02|Mg NO3 .17|Na OH .04|K OH .09|Na HCO3 0|K HCO3 -.06|Na CO3 -.08|K CO3 .02|Na F .02|K F .03|Na B(OH)4 -.07|Na HPO4 -.15|K HPO4 -.1|Fe Cl .17|Mn Cl .13|Na H3SiO4 -.08';
const PAIRPAR = (() => {
  const BR = new Float64Array(NS * NS), EPS = new Float64Array(NS * NS), rows = (t) => t.split('|').map((r) => r.split(' '));
  for (const c of PZ.cat) for (const a of PZ.an) { const p = BR_ION[SID[c]] || [0, 0], q = BR_ION[SID[a]] || [0, 0]; BR[c * NS + a] = p[0] + q[0] + p[1] * q[1]; }
  for (const [c, a, b] of rows(BR_SALT)) BR[si(c) * NS + si(a)] = +b;
  for (const [c, a, e] of rows(SIT_EPS)) EPS[si(c) * NS + si(a)] = +e;
  return { BR, EPS };
})();
/** Bromley or SIT single-ion activity coefficients; neutral species follow the Setschenow term. */
function pairModel(model, m, T, lnG) {
  let I = 0, sm = 0;
  for (let i = 0; i < NS; i++) { I += m[i] * ZS[i] * ZS[i]; sm += m[i]; }
  I *= 0.5;
  const A = (3 * aphi(T)) / LN10, s = Math.sqrt(I), sit = model === 'sit', dh = sit ? (-A * s) / (1 + 1.5 * s) : (-A * s) / (1 + s), { cat, an } = PZ, { BR, EPS } = PAIRPAR;
  for (let i = 0; i < NS; i++) lnG[i] = ZS[i] === 0 ? 0.1 * I : ZS[i] * ZS[i] * dh;
  let os = -2 * A * dhOsm(s, sit ? 1.5 : 1); // Σm·(φ − 1)/ln 10
  for (const c of cat) {
    const mc = m[c], zc = ZS[c];
    for (const a of an) {
      const ma = m[a];
      if (!(mc > 0) && !(ma > 0)) continue;
      let t;
      if (sit) { t = EPS[c * NS + a]; os += t * mc * ma; } else {
        const B = BR[c * NS + a], zz = -zc * ZS[a], Z = 0.5 * (zc - ZS[a]), x = (1.5 * I) / zz, q = 1 + x, cz = (0.06 + 0.6 * B) * zz;
        t = (cz / (q * q) + B) * Z * Z;
        if (mc > 0 && ma > 0) os += 2 * mc * ma * Z * Z * (cz * (x < 1e-3 ? 0.5 - (4 * x) / 3 : 1 / (q * q) - (Math.log(q) + 1 / q - 1) / (x * x)) + 0.5 * B);
      }
      lnG[c] += t * ma; lnG[a] += t * mc;
    }
  }
  for (let i = 0; i < NS; i++) lnG[i] = clamp(lnG[i] * LN10, -45, 45);
  const phi = sm > 1e-12 ? 1 + (os * LN10) / sm : 1;
  return { I, aw: clamp(Math.exp(-phi * sm * MW_W), 0.02, 1), phi };
}

// species lists used in the inner loops: derived species active per model family, and alkalinity carriers
const ACTIVE = [6, 7].map((f) => Int32Array.from(DER.map((d, j) => (d[f] ? j : -1)).filter((j) => j >= 0)));
const ALKI = Int32Array.from(ALK.map((a, s) => (a !== 0 ? s : -1)).filter((s) => s >= 0)), ALKV = Float64Array.from(ALKI, (s) => ALK[s]);
const KCACHE = new Map();
function kset(T, model) {
  const fam = PFAM[model] ? 5 + 1 : 7, key = fam + '|' + T;
  let k = KCACHE.get(key);
  if (!k) { if (KCACHE.size > 400) KCACHE.clear(); k = DER.map((d) => (d[fam] ? d[fam](T) : NaN)); KCACHE.set(key, k); }
  return k;
}

/** Aqueous equilibrium state at temperature T for one activity model. */
class Eq {
  constructor(T, model = 'pitzer') {
    this.T = T; this.model = ACTIVITY_MODELS[model] ? model : 'pitzer'; this.K = kset(T, this.model); this.kH = 10 ** logKH(T); this.act = ACTIVE[PFAM[this.model] ? 0 : 1];
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
    const r = this.model === 'pitzer' ? pitzer(this.m, this.T, this._l) : this.model === 'bromley' || this.model === 'sit' ? pairModel(this.model, this.m, this.T, this._l) : debye(this.model, this.m, this.T, this._l);
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
  species() { return SID.map((id, s) => ({ id, label: LABELS[s], z: ZS[s], m: this.m[s], gamma: Math.exp(this.lnG[s]), a: this.m[s] * Math.exp(this.lnG[s]) })); }
}

// ---- minerals --------------------------------------------------------------------------------------
// logK(T) for dissolution into the master species (hydroxide minerals: into OH-). sigma = crystal–solution
// interfacial energy (mJ/m²), kg = growth constant (m/s at S − 1 = 1), dV = reaction volume (cm³/mol),
// siAS = saturation index that a threshold inhibitor can normally hold.
// Sources of log K (each value compared with the database file named): scale-forming minerals — USGS WATEQ4F
// (wateq4f.dat: Plummer & Busenberg 1982 carbonates, Langmuir & Melchior 1985 sulphates, Nordstrom et al. 1990);
// evaporite salts and the hydroxides — Harvie, Møller & Weare (1984) as held in LLNL EQ3/6 data0.hmw and the USGS
// PHRQPITZ/PHREEQC pitzer.dat. A solubility product belongs to the aqueous model it was derived with, so where the
// two families differ a mineral carries both: kP is used with the Pitzer/Bromley species set (HMW values: calcite
// −8.406, aragonite −8.219) and kI with the ion-pair models (WATEQ4F values); logK is the one used otherwise.
// awx = exponent of the water activity in the ion-activity product that does not change the water inventory
// (amorphous silica: SiO2 + 2 H2O = H4SiO4, so IAP = a(H4SiO4)/aw²).
const mk = (name, formula, mw, stoich, logK, x = {}) => ({ name, formula, mw, stoich, logK, kP: null, kI: null, awx: 0, nOH: 0, nW: 0, rho: 2500, sigma: 80, kg: 1e-10, dV: 0, siAS: 0, group: 'scale', ...x, Ksp: (T = 25) => 10 ** logK(T) });
export const MINERALS = {
  calcite: mk('Calcite', 'CaCO₃', 100.087, { Ca: 1, C: 1 }, analytic(-171.9065, -0.077993, 2839.319, 71.595), { kP: analytic(-171.8329, -0.077993, 2839.319, 71.595), rho: 2710, sigma: 94, dV: -59.1, siAS: 1.8 }),
  aragonite: mk('Aragonite', 'CaCO₃', 100.087, { Ca: 1, C: 1 }, analytic(-171.9773, -0.077993, 2903.293, 71.595), { kP: analytic(-171.8607, -0.077993, 2903.293, 71.595), rho: 2930, sigma: 90, dV: -56.3, siAS: 1.8 }),
  gypsum: mk('Gypsum', 'CaSO₄·2H₂O', 172.17, { Ca: 1, SO4: 1 }, analytic(68.2401, 0, -3221.51, -25.0627), { nW: 2, rho: 2320, sigma: 40, kg: 1e-9, dV: -42.4, siAS: 0.36 }),
  anhydrite: mk('Anhydrite', 'CaSO₄', 136.14, { Ca: 1, SO4: 1 }, analytic(197.52, 0, -8669.8, -69.835), { rho: 2960, sigma: 60, kg: 3e-10, dV: -49.8, siAS: 0.36 }),
  barite: mk('Barite', 'BaSO₄', 233.39, { Ba: 1, SO4: 1 }, analytic(136.035, 0, -7680.41, -48.595), { rho: 4480, sigma: 120, kg: 3e-10, dV: -50.6, siAS: 1.78 }),
  celestite: mk('Celestite', 'SrSO₄', 183.68, { Sr: 1, SO4: 1 }, vh(-6.63, -4.3), { rho: 3960, sigma: 85, kg: 3e-10, dV: -49.7, siAS: 0.9 }),
  fluorite: mk('Fluorite', 'CaF₂', 78.07, { Ca: 1, F: 2 }, analytic(66.348, 0, -4298.2, -25.271), { rho: 3180, sigma: 140, dV: -44.7, siAS: 2.08 }),
  silica: mk('Amorphous silica', 'SiO₂(am)', 60.084, { Si: 1 }, analytic(-0.26, 0, -731, 0), { awx: -2, rho: 2200, sigma: 45, kg: 1e-12, siAS: 0.18 }),
  brucite: mk('Brucite', 'Mg(OH)₂', 58.32, { Mg: 1 }, vh(-10.88, -2), { kI: vh(-11.16, -1.6), nOH: 2, rho: 2370, sigma: 100, dV: -53.9 }),
  halite: mk('Halite', 'NaCl', 58.443, { Na: 1, Cl: 1 }, vh(1.57, 3.84), { kI: vh(1.582, 3.84), rho: 2165, sigma: 38, kg: 1e-6, dV: -10.4, group: 'salt' }),
  strontianite: mk('Strontianite', 'SrCO₃', 147.63, { Sr: 1, C: 1 }, vh(-9.271, -1.7), { rho: 3760, group: 'minor' }),
  witherite: mk('Witherite', 'BaCO₃', 197.34, { Ba: 1, C: 1 }, vh(-8.562, 2.9), { rho: 4290, group: 'minor' }),
  siderite: mk('Siderite', 'FeCO₃', 115.85, { Fe: 1, C: 1 }, vh(-10.89, -10.4), { rho: 3870, group: 'minor' }),
  magnesite: mk('Magnesite', 'MgCO₃', 84.314, { Mg: 1, C: 1 }, vh(-7.834, -25.8), { kI: vh(-8.029, -25.8), rho: 2960, group: 'inhibited' }),
  dolomite: mk('Dolomite', 'CaMg(CO₃)₂', 184.40, { Ca: 1, Mg: 1, C: 2 }, vh(-17.083, -39.5), { rho: 2840, group: 'inhibited' }),
  nesquehonite: mk('Nesquehonite', 'MgCO₃·3H₂O', 138.36, { Mg: 1, C: 1 }, vh(-5.167, -24.2), { kI: vh(-5.621, -24.2), nW: 3, rho: 1850, group: 'salt' }),
  portlandite: mk('Portlandite', 'Ca(OH)₂', 74.093, { Ca: 1 }, vh(-5.19, -17.9), { nOH: 2, rho: 2230, group: 'salt' }),
  sylvite: mk('Sylvite', 'KCl', 74.551, { K: 1, Cl: 1 }, analytic(3.984, 0, -919.55, 0), { rho: 1990, kg: 1e-6, group: 'salt' }),
  glauberite: mk('Glauberite', 'Na₂Ca(SO₄)₂', 278.18, { Na: 2, Ca: 1, SO4: 2 }, vh(-5.245, 0), { rho: 2800, group: 'salt' }),
  thenardite: mk('Thenardite', 'Na₂SO₄', 142.04, { Na: 2, SO4: 1 }, vh(-0.288, -2.4), { kI: vh(-0.179, -2.4), rho: 2660, kg: 1e-7, group: 'salt' }),
  mirabilite: mk('Mirabilite', 'Na₂SO₄·10H₂O', 322.19, { Na: 2, SO4: 1 }, vh(-1.214, 79.4), { kI: vh(-1.114, 79.4), nW: 10, rho: 1464, kg: 1e-7, group: 'salt' }),
  bloedite: mk('Bloedite', 'Na₂Mg(SO₄)₂·4H₂O', 334.47, { Na: 2, Mg: 1, SO4: 2 }, vh(-2.347, 0), { nW: 4, rho: 2230, group: 'salt' }),
  epsomite: mk('Epsomite', 'MgSO₄·7H₂O', 246.47, { Mg: 1, SO4: 1 }, vh(-1.881, 11.5), { kI: vh(-2.14, 11.8), nW: 7, rho: 1680, kg: 1e-7, group: 'salt' }),
  hexahydrite: mk('Hexahydrite', 'MgSO₄·6H₂O', 228.46, { Mg: 1, SO4: 1 }, analytic(-62.666, 0, 1828, 22.187), { nW: 6, rho: 1757, kg: 1e-7, group: 'salt' }),
  kieserite: mk('Kieserite', 'MgSO₄·H₂O', 138.38, { Mg: 1, SO4: 1 }, vh(-0.123, -29.2), { nW: 1, rho: 2570, group: 'salt' }),
  polyhalite: mk('Polyhalite', 'K₂MgCa₂(SO₄)₄·2H₂O', 602.94, { K: 2, Mg: 1, Ca: 2, SO4: 4 }, vh(-13.744, 0), { nW: 2, rho: 2780, group: 'salt' }),
  syngenite: mk('Syngenite', 'K₂Ca(SO₄)₂·H₂O', 328.42, { K: 2, Ca: 1, SO4: 2 }, vh(-7.448, 0), { nW: 1, rho: 2600, group: 'salt' }),
  kainite: mk('Kainite', 'KMgClSO₄·3H₂O', 248.97, { K: 1, Mg: 1, Cl: 1, SO4: 1 }, vh(-0.193, 0), { nW: 3, rho: 2150, group: 'salt' }),
  carnallite: mk('Carnallite', 'KMgCl₃·6H₂O', 277.85, { K: 1, Mg: 1, Cl: 3 }, vh(4.33, 0), { nW: 6, rho: 1600, kg: 1e-7, group: 'salt' }),
  bischofite: mk('Bischofite', 'MgCl₂·6H₂O', 203.30, { Mg: 1, Cl: 2 }, analytic(3.524, 0, 277.6, 0), { nW: 6, rho: 1570, kg: 1e-7, group: 'salt' }),
};
/** log K of a mineral for the species set of an activity model (see the note above the table). */
const logKfor = (M, T, model) => (PFAM[model] ? M.kP || M.logK : M.kI || M.logK)(T);
for (const [id, M] of Object.entries(MINERALS)) {
  M.id = id; M.logKfor = (T, model = 'pitzer') => logKfor(M, T, model); M.stoichiometry = { ...M.stoich, ...(M.nOH ? { OH: M.nOH } : {}), ...(M.nW ? { H2O: M.nW } : {}) };
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
  if (M.awx) s += M.awx * Math.log10(eq.aw);
  const dP = M.dV ? (-M.dV * 1e-6 * (P - 1) * 1e5) / (R * tk(eq.T) * LN10) : 0; // pressure raises solubility when ΔV < 0
  return s - (logKfor(M, eq.T, eq.model) + dP + dk);
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
    // calcium-carbonate precipitation potential: CaCO3 that precipitates (+) or dissolves (−) on the way to calcite equilibrium.
    // It needs a complete equilibrium-precipitation solve (about five times the cost of everything else in this function),
    // so it is evaluated on first access and then stored: callers that only read saturation indices do not pay for it.
    let ccpp;
    Object.defineProperty(r, 'ccpp', { enumerable: true, configurable: true,
      get() { if (ccpp === undefined) ccpp = e.tot[mi('Ca')] > 0 || e.tot[IC] > 0 ? precipitateSolution(sol, ['calcite'], { reservoir: { calcite: 0.05 * sol.w }, P, dk }).solids.calcite / sol.w * 100087 * io.kgwPerL : 0; return ccpp; },
      set(x) { ccpp = x; } });
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
  const tot = new Float64Array(NM), zc = MZ[mi(cation)], za = -MZ[mi(anion)], gcd = zc === za ? zc : 1, nc = za / gcd, na = zc / gcd; // smallest electroneutral formula unit (MgSO4, not Mg2(SO4)2)
  tot[mi(cation)] = nc * m; tot[mi(anion)] = na * m;
  const e = new Eq(T, model).run(tot, { pH: 7 });
  let lg, nu = nc + na;
  if (model === 'pitzer') lg = (nc * e.lnG[mi(cation)] + na * e.lnG[mi(anion)]) / nu;
  else lg = (nc * Math.log(e.m[mi(cation)] * Math.exp(e.lnG[mi(cation)]) / (nc * m)) + na * Math.log(e.m[mi(anion)] * Math.exp(e.lnG[mi(anion)]) / (na * m))) / nu; // stoichiometric γ± including ion pairing
  return { gamma: Math.exp(lg), phi: model === 'pitzer' ? e.phi : -Math.log(e.aw) / (MW_W * nu * m), aw: e.aw, I: e.I }; // stoichiometric φ (per formula ion) where ion pairs form
}
export const COMPONENTS = MAST;
export const componentIndex = mi;

// ---- Gibbs-energy minimisation ------------------------------------------------------------------------
// Standard chemical potentials (in units of RT) follow from the equilibrium constants with the master species,
// H⁺ and H₂O as the reference: derived species μ° = −ln K, solids μ° = ln Ksp − n(OH)·ln Kw.
const solidMu0 = (M, T, P, dk, model = 'pitzer') => LN10 * (logKfor(M, T, model) + (M.dV ? (-M.dV * 1e-6 * (P - 1) * 1e5) / (R * tk(T) * LN10) : 0) + dk - M.nOH * logKw(T));
/** Total Gibbs energy G/RT (mol) of a solution plus solids ({ id: mol }), on the reference above. */
export function gibbsEnergy(sol, solids = {}, { P = 1, dk = {} } = {}) {
  const e = sol.eq, K = e.K;
  let g = (sol.w / MW_W) * Math.log(e.aw);
  for (let s = 0; s < NS; s++) { const x = e.m[s]; if (x > 0) g += x * sol.w * ((s > IH ? -LN10 * K[s - NM - 1] : 0) + Math.log(x) + e.lnG[s]); }
  for (const [id, n] of Object.entries(solids)) if (MINERALS[id] && n) g += n * solidMu0(MINERALS[id], sol.T, P, dk[id] || 0, sol.model);
  return g;
}
/** Gaussian elimination with partial pivoting on a flat row-major matrix (destroys A and b); false when singular. */
function gaussInPlace(A, b, x, n) {
  for (let k = 0; k < n; k++) {
    let p = k, big = Math.abs(A[k * n + k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(A[i * n + k]); if (v > big) { big = v; p = i; } }
    if (!(big > 1e-300)) return false;
    if (p !== k) { for (let j = k; j < n; j++) { const t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t; } const t = b[k]; b[k] = b[p]; b[p] = t; }
    const piv = A[k * n + k];
    for (let i = k + 1; i < n; i++) { const f = A[i * n + k] / piv; if (f === 0) continue; for (let j = k + 1; j < n; j++) A[i * n + j] -= f * A[k * n + j]; b[i] -= f * b[k]; }
  }
  for (let i = n - 1; i >= 0; i--) { let v = b[i]; for (let j = i + 1; j < n; j++) v -= A[i * n + j] * x[j]; x[i] = v / A[i * n + i]; if (!Number.isFinite(x[i])) return false; }
  return true;
}
/**
 * Chemical equilibrium by direct minimisation of the total Gibbs energy over all aqueous species and the listed
 * solid phases, subject to the component (element) balances and the proton condition. Solved in the dual form
 * (element potentials λ: μᵢ = Σ aᵢⱼλⱼ for every species present, μ ≥ Σ aλ for absent solids) by a damped Newton
 * iteration with an active set of solids; activity coefficients are updated between Newton steps. The start is a
 * neutral solution of free ions, so the result is independent of the mass-action solver.
 */
export function gibbsMinimize(sol, minerals = [], { P = 1, dk = {}, reservoir = {}, pH0 = 7 } = {}) {
  const T = sol.T, E = new Eq(T, sol.model), K = E.K, w0 = sol.w, law0 = LN10 * logKw(T);
  const ids = minerals.filter((id) => MINERALS[id]), mins = ids.map((id) => MINERALS[id]), nK = ids.length, res = ids.map((id) => Math.max(0, reservoir[id] || 0));
  const nTot = Float64Array.from(sol.n, (x) => Math.max(0, x));
  let bH = 2 * nTot[IC] - sol.alk;
  mins.forEach((M, k) => { if (res[k] > 0) { for (const [i, nu] of M._st) nTot[i] += nu * res[k]; bH += -M.nOH * res[k]; } });
  const cIdx = new Int32Array(NM).fill(-1);
  let nc = 0;
  for (let i = 0; i < NM; i++) if (nTot[i] > 0) cIdx[i] = nc++;
  const iHc = nc, NC = nc + 1, b = new Float64Array(NC);
  for (let i = 0; i < NM; i++) if (cIdx[i] >= 0) b[cIdx[i]] = nTot[i];
  b[iHc] = bH;
  const sp = [];
  for (let i = 0; i < NM; i++) if (cIdx[i] >= 0) sp.push({ s: i, ci: [cIdx[i]], ai: [1], lk: 0, nW: 0 });
  sp.push({ s: IH, ci: [iHc], ai: [1], lk: 0, nW: 0 });
  for (const j of E.act) {
    const a = DA[j], c = DB[j];
    if ((a >= 0 && cIdx[a] < 0) || (c >= 0 && cIdx[c] < 0)) continue;
    const ci = [], ai = [];
    if (a >= 0) { ci.push(cIdx[a]); ai.push(1); }
    if (c >= 0) { ci.push(cIdx[c]); ai.push(1); }
    if (DNH[j]) { ci.push(iHc); ai.push(DNH[j]); }
    sp.push({ s: NM + 1 + j, ci, ai, lk: LN10 * K[j], nW: DNW[j] });
  }
  const so = mins.map((M, k) => { const ok = M._st.every(([i]) => cIdx[i] >= 0), ci = M._st.map(([i]) => cIdx[i]), ai = M._st.map(([, nu]) => nu); if (M.nOH) { ci.push(iHc); ai.push(-M.nOH); } return { ok, ci, ai, mu: solidMu0(M, T, P, dk[ids[k]] || 0, sol.model), nW: M.nW + M.nOH + M.awx }; });
  const nS = sp.length, lam = new Float64Array(NC), lnG = new Float64Array(NS), ms = new Float64Array(nS), nk = Float64Array.from(res), r = new Float64Array(NC), sc = new Float64Array(NC);
  for (let c = 0; c < nc; c++) lam[c] = Math.log(b[c] / w0);
  lam[iHc] = -LN10 * pH0;
  let law = 0, w = w0;
  const active = [], isAct = new Uint8Array(nK);
  for (let k = 0; k < nK; k++) if (so[k].ok && res[k] > 0) { active.push(k); isAct[k] = 1; }
  const aff = (k, l = lam) => { const q = so[k]; let x = -(q.mu - q.nW * law); for (let t = 0; t < q.ci.length; t++) x += q.ai[t] * l[q.ci[t]]; return x; }; // = ln(IAP/Ksp)
  const evalAt = (l, n) => { // residuals of the element balances; returns the scaled merit
    r.fill(0); sc.fill(0);
    for (let q = 0; q < nS; q++) {
      const p = sp[q];
      let x = p.lk + p.nW * law - lnG[p.s];
      for (let t = 0; t < p.ci.length; t++) x += p.ai[t] * l[p.ci[t]];
      const mm = Math.exp(Math.min(x, 4.6)) * w; // molalities are capped at 100 mol/kg while the iteration is far from the solution
      ms[q] = mm;
      for (let t = 0; t < p.ci.length; t++) { r[p.ci[t]] += p.ai[t] * mm; sc[p.ci[t]] += Math.abs(p.ai[t]) * mm; }
    }
    for (const k of active) { const q = so[k]; for (let t = 0; t < q.ci.length; t++) { r[q.ci[t]] += q.ai[t] * n[k]; sc[q.ci[t]] += Math.abs(q.ai[t] * n[k]); } }
    let merit = 0;
    dual = 0;
    for (let q = 0; q < nS; q++) dual += ms[q];
    for (let c = 0; c < NC; c++) { dual -= l[c] * b[c]; r[c] -= b[c]; sc[c] = Math.max(sc[c], Math.abs(b[c]), 1e-30); merit = Math.max(merit, Math.abs(r[c]) / sc[c]); }
    for (const k of active) merit = Math.max(merit, Math.abs(aff(k, l)));
    return merit;
  };
  let dual = 0, it = 0, merit = evalAt(lam, nk), dG = 1, done = false;
  const l2 = new Float64Array(NC), n2 = new Float64Array(nK), NMAX = NC + nK, AM = new Float64Array(NMAX * NMAX), RH = new Float64Array(NMAX), DX = new Float64Array(NMAX);
  let nAct = 0, warm = false; // first meet the balances with ideal activities, then let the activity model in
  for (; it < 400 && !done; it++) {
    const nA = active.length, n = NC + nA, A = AM.subarray(0, n * n), rhs = RH.subarray(0, n);
    A.fill(0);
    for (let q = 0; q < nS; q++) { const p = sp[q], mm = ms[q]; for (let t = 0; t < p.ci.length; t++) for (let u = 0; u < p.ci.length; u++) A[p.ci[t] * n + p.ci[u]] += p.ai[t] * p.ai[u] * mm; }
    for (let c = 0; c < NC; c++) rhs[c] = -r[c];
    active.forEach((k, a) => { const q = so[k]; for (let t = 0; t < q.ci.length; t++) { A[q.ci[t] * n + NC + a] += q.ai[t]; A[(NC + a) * n + q.ci[t]] += q.ai[t]; } rhs[NC + a] = -aff(k); });
    const dx = DX.subarray(0, n);
    if (!gaussInPlace(A, rhs, dx, n)) break;
    let al = 1, big = 0;
    for (let c = 0; c < NC; c++) big = Math.max(big, Math.abs(dx[c]));
    if (big > 2.5) al = 2.5 / big;
    let hit = -1;
    active.forEach((k, a) => { if (dx[NC + a] < 0 && nk[k] + al * dx[NC + a] < 0) { al = Math.max(0, -nk[k] / dx[NC + a]); hit = a; } });
    if (hit >= 0 && al < 1e-12) { isAct[active[hit]] = 0; nk[active[hit]] = 0; active.splice(hit, 1); merit = evalAt(lam, nk); continue; } // exhausted solid leaves the assemblage
    // line search: without solids the dual function Σn − λ·b is convex and the Newton step is a descent direction for it
    let slope = 0;
    if (!nA) for (let c = 0; c < NC; c++) slope += r[c] * dx[c];
    const d0 = dual;
    for (let t = 0; t < 30; t++) {
      for (let c = 0; c < NC; c++) l2[c] = lam[c] + al * dx[c];
      n2.set(nk); active.forEach((k, a) => { n2[k] = Math.max(0, nk[k] + al * dx[NC + a]); });
      const m2 = evalAt(l2, n2);
      if (nA ? m2 < merit || merit < 1e-9 || t >= 11 : dual <= d0 + 1e-4 * al * slope + 1e-14 * Math.abs(d0) || t === 29) break;
      al *= 0.5;
    }
    lam.set(l2); nk.set(n2);
    if (!warm) { merit = evalAt(lam, nk); if (merit < 1e-3 || it > 80) warm = true; else continue; }
    // activity coefficients, water activity and water inventory for the new composition
    E.m.fill(0);
    for (let q = 0; q < nS; q++) E.m[sp[q].s] = ms[q] / w;
    E._act();
    dG = 0;
    let fin = Number.isFinite(E.aw) && E.aw > 0;
    for (let s = 0; s < NS && fin; s++) if (!Number.isFinite(E._l[s])) fin = false;
    if (!fin) { dG = 1; merit = evalAt(lam, nk); continue; } // activity model out of range for this iterate: take another Newton step first
    nAct++;
    for (let s = 0; s < NS; s++) { const e = E._l[s] - lnG[s]; if (E.m[s] > 0 && Math.abs(e) > dG) dG = Math.abs(e); lnG[s] += (nAct < 8 ? 0.5 : nAct < 80 ? 1 : 0.5) * clamp(e, -1.5, 1.5); }
    law = Math.log(E.aw); w = w0;
    for (let k = 0; k < nK; k++) w -= mins[k].nW * MW_W * (nk[k] - res[k]);
    merit = evalAt(lam, nk);
    if (merit < 1e-11 && dG < 1e-10) {
      let best = -1, bv = 1e-9;
      for (let k = 0; k < nK; k++) if (so[k].ok && !isAct[k]) { const x = aff(k); if (x > bv) { bv = x; best = k; } }
      for (let a = active.length - 1; a >= 0; a--) if (!(nk[active[a]] > 0)) { isAct[active[a]] = 0; active.splice(a, 1); }
      if (best >= 0) { active.push(best); isAct[best] = 1; merit = evalAt(lam, nk); } else done = true;
    }
  }
  // final state in the form of an ordinary solution
  E.lnG.set(lnG); E.pH = -lam[iHc] / LN10; E.tot.fill(0);
  let alk = 0;
  for (let q = 0; q < nS; q++) { const s = sp[q].s; E.m[s] = ms[q] / w; alk += ALK[s] * ms[q]; }
  const nOut = new Float64Array(NM);
  for (let q = 0; q < nS; q++) { const s = sp[q].s; if (s < NM) nOut[s] += ms[q]; else if (s > IH) { const j = s - NM - 1; if (DA[j] >= 0) nOut[DA[j]] += ms[q]; if (DB[j] >= 0) nOut[DB[j]] += ms[q]; } }
  for (let i = 0; i < NM; i++) E.tot[i] = nOut[i] / w;
  E.alk = alk / w;
  const out = wrap(T, E.model, nOut, alk, w, E, { kgwPerL: sol.kgwPerL }), solids = Object.fromEntries(ids.map((id, k) => [id, nk[k] - res[k]])), amount = Object.fromEntries(ids.map((id, k) => [id, nk[k]]));
  const G = gibbsEnergy(out, amount, { P, dk }), ok = done && Number.isFinite(G) && Number.isFinite(E.pH);
  return { sol: out, pH: E.pH, solids, active: active.map((k) => ids[k]), G, iterations: it, converged: ok, residual: Number.isFinite(merit) ? merit : 1, lambda: Array.from(lam) };
}

// ---- surface complexation: generalised two-layer model -------------------------------------------------
// Hydrous ferric oxide (Dzombak & Morel 1990): 89 g/mol Fe, 600 m²/g, 0.2 mol weak sites per mol Fe.
// Reactions ≡FeOH + sorbate + h·H⁺ = surface species (charge change dz); intrinsic log K at 25 °C.
// Source: Dzombak & Morel (1990) as tabulated in the SURFACE_SPECIES block of the USGS PHREEQC database phreeqc.dat
// (acid–base constants table 5.7; Ca, Mg tables 10.1/10.5; borate table 10.7; sulphate table 10.8), silicate from
// Swedlund & Webster (1999) in the same file — all ten constants, the 600 m²/g and the 0.2 mol weak sites per mol Fe
// were compared with that file and are identical. The 0.005 mol/mol strong sites are not carried: none of the
// sorbates treated here (silica, boron, sulphate, magnesium) has a strong-site constant, and the strong-site
// calcium complex occupies at most 2.5 % of the sites.
export const HFO = { mw: 89, area: 600, sites: 0.2, rx: [
  ['≡FeOH₂⁺', '', 1, 1, 7.29], ['≡FeO⁻', '', -1, -1, -8.93], ['≡FeH₂BO₃', 'B', 0, 0, 0.62], ['≡FeH₃SiO₄', 'Si', 0, 0, 4.28], ['≡FeH₂SiO₄⁻', 'Si', -1, -1, -3.22], ['≡FeHSiO₄²⁻', 'Si', -2, -2, -11.69],
  ['≡FeOCa⁺', 'Ca', -1, 1, -5.85], ['≡FeOMg⁺', 'Mg', -1, 1, -4.6], ['≡FeSO₄⁻', 'SO4', 1, -1, 7.78], ['≡FeOHSO₄²⁻', 'SO4', 0, -2, 0.79]] };
const FARADAY = 96485.33212;
/**
 * Sorption on hydrous ferric oxide in equilibrium with the aqueous state `eq`: mass action with the Coulombic
 * factor exp(−Δz·Fψ/RT), the site balance, the Gouy–Chapman charge–potential relation σ = 0.1174·√I·sinh(Fψ/2RT)
 * and the mass balances of the trace sorbates boron and silica (major ions are not depleted).
 * feMgL = iron dose (mg Fe per litre). o.act overrides sorbate activities (used by the verification).
 */
export function surfaceComplexation(eq, feMgL, { kgwPerL = 1, act = null, I = null, pH = null, rx = null, sites = null, area = null } = {}) {
  const RX = rx || HFO.rx, fe = Math.max(feMgL, 1e-9) / 1000 / 55.845 / kgwPerL, Stot = sites ?? fe * HFO.sites, areaKg = area ?? HFO.area * fe * HFO.mw; // mol Fe, mol sites and m² of surface per kg water (sites, area and rx can be overridden for benchmarks)
  const aH = 10 ** -(pH ?? eq.pH), ion = I ?? eq.I, a0 = (k) => (act ? act[k] || 0 : eq.tot[mi(k)] > 0 ? eq.m[mi(k)] * Math.exp(eq.lnG[mi(k)]) : 0);
  const tot = { Si: act ? act.SiT ?? 0 : eq.tot[mi('Si')], B: act ? act.BT ?? 0 : eq.tot[mi('B')] }, g = { Si: tot.Si > 0 ? a0('Si') / tot.Si : 0, B: tot.B > 0 ? a0('B') / tot.B : 0 }; // activity of the sorbing species per mol of dissolved total
  const state = (x) => {
    const kf = RX.map(([, , h, dz, lk]) => 10 ** lk * aH ** h * Math.exp(-dz * x));
    let fix = 1; const dep = { Si: 0, B: 0 };
    RX.forEach(([, sb], q) => { if (sb === 'Si' || sb === 'B') dep[sb] += kf[q] * g[sb]; else fix += kf[q] * (sb ? a0(sb) : 1); });
    const h = (S) => S * fix + (tot.Si * S * dep.Si) / (1 + S * dep.Si) + (tot.B * S * dep.B) / (1 + S * dep.B) - Stot;
    const S = brent(h, 0, Stot, 1e-16 * Stot + 1e-300), c = { Si: tot.Si / (1 + S * dep.Si), B: tot.B / (1 + S * dep.B) };
    const conc = RX.map(([, sb], q) => S * kf[q] * (sb === 'Si' || sb === 'B' ? g[sb] * c[sb] : sb ? a0(sb) : 1));
    let z = 0;
    RX.forEach(([, , , dz], q) => { z += dz * conc[q]; });
    return { S, c, conc, sigma: (FARADAY * z) / areaKg };
  };
  const sd = (x) => 0.1174 * Math.sqrt(Math.max(ion, 1e-12)) * Math.sinh(x / 2), f = (x) => state(x).sigma - sd(x);
  const x = brent(f, -30, 30, 1e-12), st = state(x);
  return { x, psi: (x * R * tk(eq?.T ?? 25)) / FARADAY, sigma: st.sigma, sigmaDiffuse: sd(x), Stot, free: st.S, areaKg,
    species: [{ name: '≡FeOH', conc: st.S, frac: st.S / Stot }, ...RX.map(([name], q) => ({ name, conc: st.conc[q], frac: st.conc[q] / Stot }))],
    sorbed: { Si: tot.Si - st.c.Si, B: tot.B - st.c.B }, dissolved: st.c, total: tot, removal: { Si: tot.Si > 0 ? 1 - st.c.Si / tot.Si : 0, B: tot.B > 0 ? 1 - st.c.B / tot.B : 0 } };
}

// ---- ion exchange: Gaines–Thomas convention --------------------------------------------------------------
/** Equivalent fractions on an exchanger in equilibrium with a solution: βM·aNa² / (βNa²·aM) = K (divalent), βK·aNa / (βNa·aK) = K. */
export function gainesThomas(act, logK = {}) {
  const K = { Ca: 10 ** (logK.Ca ?? 0.8), Mg: 10 ** (logK.Mg ?? 0.6), K: 10 ** (logK.K ?? 0.7) }, aNa = Math.max(act.Na || 0, 1e-30);
  const A = (K.Ca * (act.Ca || 0) + K.Mg * (act.Mg || 0)) / (aNa * aNa), B = 1 + (K.K * (act.K || 0)) / aNa, y = 2 / (B + Math.sqrt(B * B + 4 * A));
  return { Na: y, K: (K.K * (act.K || 0) * y) / aNa, Ca: (K.Ca * (act.Ca || 0) * y * y) / (aNa * aNa), Mg: (K.Mg * (act.Mg || 0) * y * y) / (aNa * aNa) };
}
/** Batch equilibrium of V litres of solution with q equivalents of exchanger. T = total equivalents { Na, Ca, Mg }; Kc = concentration-based selectivities (L/eq). */
function batchExchange(T, V, q, Kc) {
  const bet = (y) => { const cNa = Math.max((T.Na - y * q) / V, 1e-300), d = V * cNa * cNa; return { Na: y, Ca: (Kc.Ca * y * y * T.Ca) / (d + Kc.Ca * y * y * q), Mg: (Kc.Mg * y * y * T.Mg) / (d + Kc.Mg * y * y * q) }; };
  const top = Math.min(1, T.Na / q) * (1 - 1e-12), f = (y) => { const x = bet(y); return x.Na + x.Ca + x.Mg - 1; };
  const y = f(top) <= 0 ? top : brent(f, 0, top, 1e-14), x = bet(y), s = x.Na + x.Ca + x.Mg;
  const beta = { Na: x.Na / s, Ca: x.Ca / s, Mg: x.Mg / s };
  return { beta, c: { Na: Math.max(0, (T.Na - beta.Na * q) / V), Ca: Math.max(0, (T.Ca - beta.Ca * q) / V), Mg: Math.max(0, (T.Mg - beta.Mg * q) / V) } };
}
/**
 * Sodium-cycle softener: exhausted and regenerated resin compositions (Gaines–Thomas with activity coefficients from
 * the speciation model), and the breakthrough curve from an equilibrium-stage column (feed parcels pass through
 * nStage stages in series). Concentrations in eq/L, capacity in eq per litre of resin; monovalent K⁺ is counted with Na⁺.
 */
export function softenerColumn(feed, { cap = 2, logKCa = 0.8, logKMg = 0.6, regenDose = 120, regenPct = 10, nStage = 12, nParcel = 240, leakFrac = 0.05, model = 'pitzer', T = 25 } = {}) {
  const e = feed.eq, kw = feed.kgwPerL || 1, c0 = { Na: (e.tot[mi('Na')] + e.tot[mi('K')]) * kw, Ca: 2 * e.tot[mi('Ca')] * kw, Mg: 2 * e.tot[mi('Mg')] * kw }; // eq/L
  const gam = (E) => ({ Na: Math.exp(E.lnG[mi('Na')]), Ca: Math.exp(E.lnG[mi('Ca')]) * (E.tot[mi('Ca')] > 0 ? E.m[mi('Ca')] / E.tot[mi('Ca')] : 1), Mg: Math.exp(E.lnG[mi('Mg')]) * (E.tot[mi('Mg')] > 0 ? E.m[mi('Mg')] / E.tot[mi('Mg')] : 1) });
  // concentration-based selectivity: β_M·c_Na² / (β_Na²·c_M) with c in eq/L (c_M = 2·mol/L)
  const kc = (gm, k) => ({ Ca: (10 ** logKCa * gm.Ca * k) / (2 * gm.Na * gm.Na), Mg: (10 ** logKMg * gm.Mg * k) / (2 * gm.Na * gm.Na) });
  const gF = gam(e), KcF = kc(gF, kw);
  const eqm = (c, Kc) => { const A = (Kc.Ca * c.Ca + Kc.Mg * c.Mg) / Math.max(c.Na * c.Na, 1e-300), y = 2 / (1 + Math.sqrt(1 + 4 * A)); return { Na: y, Ca: (Kc.Ca * c.Ca * y * y) / Math.max(c.Na * c.Na, 1e-300), Mg: (Kc.Mg * c.Mg * y * y) / Math.max(c.Na * c.Na, 1e-300) }; };
  const exhausted = eqm(c0, KcF);
  // regeneration: batch equilibrium of the exhausted resin with the NaCl regenerant
  const mR = (regenPct * 10) / 58.443 / (1 - regenPct / 100), tot = new Float64Array(NM); // mol NaCl per kg water
  tot[mi('Na')] = mR; tot[mi('Cl')] = mR + 2e-3; tot[mi('Ca')] = 5e-4; tot[mi('Mg')] = 5e-4;
  const eR = new Eq(T, model).run(tot, { pH: 7 }), cR = (regenPct * 10 * density(T, regenPct * 10)) / 1000 / 58.443, KcR = kc(gam(eR), (density(T, regenPct * 10) * (1 - regenPct / 100)) / 1000), vR = Math.max(regenDose, 0) / 58.443 / Math.max(cR, 1e-9); // eq/L of regenerant; litres of regenerant per litre of resin
  // counter-current regeneration of the fully exhausted bed: regenerant parcels pass the stages from the service outlet to the inlet
  const N = Math.max(2, Math.round(nStage)), qs = cap / N, bet = Array.from({ length: N }, () => ({ ...exhausted })), nReg = 40, dR = vR / nReg;
  if (vR > 0) for (let p = 0; p < nReg; p++) {
    let c = { Na: cR, Ca: 0, Mg: 0 };
    for (let st = N - 1; st >= 0; st--) { const r = batchExchange({ Na: c.Na * dR + bet[st].Na * qs, Ca: c.Ca * dR + bet[st].Ca * qs, Mg: c.Mg * dR + bet[st].Mg * qs }, dR, qs, KcR); bet[st] = r.beta; c = r.c; }
  }
  const avg = (k) => sum(bet.map((x) => x[k])) / N, reg = { Na: avg('Na'), Ca: avg('Ca'), Mg: avg('Mg') }, regOutlet = { ...bet[N - 1] };
  const hard0 = c0.Ca + c0.Mg, working = Math.max(0, cap * (exhausted.Ca + exhausted.Mg - reg.Ca - reg.Mg)), bvIdeal = hard0 > 0 ? working / hard0 : 0;
  // service run through the equilibrium-stage column
  const bvMax = Math.max(1.8 * bvIdeal, 4), dV = bvMax / Math.max(20, Math.round(nParcel));
  const bv = [], yCa = [], yMg = [];
  let fed = 0, out = 0, bvBreak = null, sumNa = 0, sumCa = 0, sumMg = 0, nPre = 0;
  for (let p = 0; p < Math.max(20, Math.round(nParcel)); p++) {
    let c = { ...c0 };
    for (let s = 0; s < N; s++) { const r = batchExchange({ Na: c.Na * dV + bet[s].Na * qs, Ca: c.Ca * dV + bet[s].Ca * qs, Mg: c.Mg * dV + bet[s].Mg * qs }, dV, qs, KcF); bet[s] = r.beta; c = r.c; }
    fed += hard0 * dV; out += (c.Ca + c.Mg) * dV;
    bv.push((p + 1) * dV); yCa.push(c0.Ca > 0 ? c.Ca / c0.Ca : 0); yMg.push(c0.Mg > 0 ? c.Mg / c0.Mg : 0);
    if (bvBreak == null && hard0 > 0 && (c.Ca + c.Mg) / hard0 > leakFrac) bvBreak = p * dV;
    if (bvBreak == null) { sumNa += c.Na; sumCa += c.Ca; sumMg += c.Mg; nPre++; }
  }
  const onResin = cap * (sum(bet.map((x) => x.Ca + x.Mg)) / N - reg.Ca - reg.Mg), soft = nPre ? { Na: sumNa / nPre, Ca: sumCa / nPre, Mg: sumMg / nPre } : { ...c0 };
  return { c0, exhausted, regenerated: reg, regOutlet, working, bvIdeal, bvBreak: bvBreak ?? bvMax, broke: bvBreak != null, bv, yCa, yMg, soft, hard0, leak: hard0 > 0 ? (soft.Ca + soft.Mg) / hard0 : 0, balance: { in: fed, out: out + onResin }, KcFeed: KcF, KcRegen: KcR, regenLitres: vR, saltEff: regenDose > 0 ? (working * 58.443) / regenDose : 0 };
}

// ---- population balance of a precipitating scale mineral -------------------------------------------------
/**
 * Population balance ∂n/∂t + G ∂n/∂L = 0 with the nucleation flux J at the critical size, solved by the method of
 * characteristics: every class of crystals (initial nuclei, initial precipitate, and the nuclei born in each time
 * step) keeps its number and grows with the size-independent rate G(t). The dissolved excess is depleted by the
 * crystal volume, which feeds back on J and G through `rates(ξ)` → { J (#/m³·s), G (m/s), Lc (m) }; ξ = precipitated mol per m³.
 * classes0 = [{ N (#/m³), L (m) }], rhoM = molar density of the crystal (mol/m³), xiMax = equilibrium amount.
 */
export function scalePBE({ rates, classes0 = [], rhoM, kv = Math.PI / 6, tEnd, xiMax = Infinity, nStep = 240 }) {
  const N = [], L = [], vol = () => { let s = 0; for (let i = 0; i < N.length; i++) s += N[i] * L[i] ** 3; return kv * s; };
  for (const c of classes0) if (c.N > 0 && c.L > 0) { N.push(c.N); L.push(c.L); }
  const v0 = vol(), hist = { t: [0], xi: [0], number: [sum(N)], L43: [], G: [], J: [], S: [] }, steps = Math.max(8, Math.round(nStep));
  const mom = (j) => { let s = 0; for (let i = 0; i < N.length; i++) s += N[i] * L[i] ** j; return s; };
  const l43 = () => { const m3 = mom(3); return m3 > 0 ? mom(4) / m3 : 0; };
  let t = 0, xi = 0, sub = 0, r = rates(0);
  hist.L43.push(l43()); hist.G.push(r.G); hist.J.push(r.J); hist.S.push(r.S ?? 1);
  for (let k = 1; k <= steps; k++) {
    const tk1 = tEnd * (k / steps) ** 2;
    while (t < tk1 - 1e-12 * tEnd && sub < 40 * steps) {
      const rate = rhoM * kv * (3 * r.G * mom(2) + r.J * r.Lc ** 3), room = Math.max(0, xiMax - xi);
      let dt = tk1 - t;
      if (rate > 0 && Number.isFinite(xiMax)) dt = Math.min(dt, Math.max((0.04 * Math.max(room, 1e-6 * xiMax)) / rate, 1e-9 * tEnd));
      // Heun step: growth increment and nuclei born with the mean of the rates at the start and at the predicted end
      const xiP = Math.min(xiMax, xi + rate * dt), rp = rates(xiP), G = 0.5 * (r.G + rp.G), J = 0.5 * (r.J + rp.J), dL = G * dt;
      for (let i = 0; i < L.length; i++) L[i] += dL;
      if (J * dt > 0) { N.push(J * dt); L.push(0.5 * (r.Lc + rp.Lc) + 0.5 * dL); }
      t += dt; sub++;
      xi = Math.min(xiMax, rhoM * (vol() - v0)); r = rates(xi);
    }
    hist.t.push(t); hist.xi.push(xi); hist.number.push(mom(0)); hist.L43.push(l43()); hist.G.push(r.G); hist.J.push(r.J); hist.S.push(r.S ?? 1);
  }
  return { ...hist, N, L, mu: [0, 1, 2, 3, 4].map(mom), volume0: v0, volume: vol(), complete: t >= tEnd * (1 - 1e-9), subSteps: sub };
}

// ---- one-dimensional advection–dispersion–reaction --------------------------------------------------------
/** Complementary error function (Abramowitz & Stegun 7.1.26, |error| < 1.5·10⁻⁷). */
export const erfc = (x) => { const z = Math.abs(x), t = 1 / (1 + 0.3275911 * z), y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z); return x >= 0 ? y : 2 - y; };
/** Ogata–Banks solution of advection–dispersion from a constant-concentration inlet into a semi-infinite column. */
export const ogataBanks = (x, t, u, D) => (t <= 0 ? 0 : 0.5 * erfc((x - u * t) / (2 * Math.sqrt(D * t))) + (u * x / D < 600 ? 0.5 * Math.exp((u * x) / D) * erfc((x + u * t) / (2 * Math.sqrt(D * t))) : 0));
/**
 * ∂c/∂t + u ∂c/∂x = D ∂²c/∂x² + s for several scalars sharing one transport operator. Finite volumes, upwind advection,
 * central dispersion, backward Euler (tridiagonal); the numerical dispersion u·Δx/2 + u²·Δt/2 is subtracted from D.
 * inlet: 'conc' (prescribed concentration, Dirichlet) or 'flux' (prescribed total flux u·c_in, Danckwerts);
 * outlet: convective (zero dispersive flux). react(cells, dt) applies the source by operator splitting.
 */
export function adr1d({ N = 40, L = 1, u = 1, D = 0, tEnd, cfl = 0.5, inlet = 'conc', fields, react = null, onStep = null }) {
  const n = Math.max(4, Math.round(N)), dx = L / n, nSteps = Math.max(4, Math.ceil((tEnd * u) / (cfl * dx))), dt = tEnd / nSteps;
  const Dnum = (u * dx) / 2 + (u * u * dt) / 2, Dd = Math.max(0, D - Dnum), Deff = Dd + Dnum, a = dt / dx, dd = Dd / dx, dir = inlet === 'conc' ? 2 * dd : 0;
  const lo = new Array(n).fill(-(u + dd) * a), di = new Array(n).fill(1 + (u + 2 * dd) * a), up = new Array(n).fill(-dd * a);
  lo[0] = 0; di[0] = 1 + (u + dd + dir) * a; di[n - 1] = 1 + (u + dd) * a; up[n - 1] = 0;
  const c = fields.map((f) => new Array(n).fill(f.c0 ?? 0)), bal = fields.map((f) => ({ in: 0, out: 0, store0: (f.c0 ?? 0) * L }));
  for (let k = 0; k < nSteps; k++) {
    fields.forEach((f, q) => {
      const rhs = c[q].slice();
      rhs[0] += (u + dir) * a * f.cin;
      const x = tridiag(lo, di, up, rhs);
      bal[q].in += (u * f.cin + dir * (f.cin - x[0])) * dt; bal[q].out += u * x[n - 1] * dt;
      c[q] = x;
    });
    if (react) react(c, dt, k);
    if (onStep) onStep(c, (k + 1) * dt, k);
  }
  return { x: Array.from({ length: n }, (_, i) => (i + 0.5) * dx), c, dx, dt, nSteps, Deff, Dnum, balance: bal.map((bq, q) => ({ ...bq, store: sum(c[q]) * dx })) };
}

/** Kinetic mineral for the population balance, reactive transport and channel models. */
function pickMineral(v, a, sol) {
  const SI = saturation(sol.eq, a.P, a.dk, a.set);
  if (v.pbMineral && v.pbMineral !== 'auto' && MINERALS[v.pbMineral] && SI[v.pbMineral] != null) return v.pbMineral;
  let best = null;
  for (const id of a.set) if (SI[id] != null && id !== 'halite' && (best == null || SI[id] / MINERALS[id]._nu > SI[best] / MINERALS[best]._nu)) best = id;
  return best;
}
/** Solution after ξ mol of a mineral (per kg of the current water) has left it (ξ < 0: dissolved into it). */
function withdraw(sol, id, xi) {
  const M = MINERALS[id], n = Float64Array.from(sol.n), amt = xi * sol.w;
  for (const [i, nu] of M._st) n[i] = Math.max(0, n[i] - nu * amt);
  return equilibrate({ ...sol, n, alk: sol.alk - M._alk * amt, w: sol.w - M.nW * MW_W * amt });
}
const rhoMolar = (id) => (MINERALS[id].rho / MINERALS[id].mw) * 1000; // mol per m³ of crystal

/** Precipitation of one mineral from the bulk concentrate during its residence time (population balance + speciation). */
export function scaleKinetics(sol, id, v, { P = 1, dk = {} } = {}) {
  const M = MINERALS[id], kw = solutionToIons(sol).kgwPerL * 1000, dkv = dk[id] || 0; // kg water per m³
  const xiEq = Math.max(0, precipitateSolution(sol, [id], { P, dk }).solids[id] / sol.w), si0 = saturationIndex(sol.eq, id, P, dkv);
  const xs = linspace(0, xiEq, 4), sis = xiEq > 0 ? xs.map((x, i) => (i === 0 ? si0 : i === 3 ? 0 : saturationIndex(withdraw(sol, id, x).eq, id, P, dkv))) : xs.map(() => si0);
  const siAt = (xi) => (xiEq > 0 ? interp1(xs, sis, clamp(xi / kw, 0, xiEq)) : si0);
  const rates = (xi) => { const s = siAt(xi), k = s > 0 ? nucleationKinetics(id, s, sol.T, v) : null; return k ? { J: Math.exp(clamp(k.lnJ, -700, 80)), G: k.G, Lc: Math.max(2e-9 * k.rCritNm, 1e-9), S: k.S } : { J: 0, G: 0, Lc: 1e-9, S: 10 ** (s / M._nu) }; };
  const seedMol = Math.max(0, v.seedMass ?? 0) / M.mw, Ls = Math.max(v.seedL ?? 5, 0.01) * 1e-6, kvS = Math.PI / 6; // mol/m³ of initial precipitate
  const classes0 = [{ N: Math.max(0, v.seedN ?? 0) * 1e6, L: Math.max(v.seedL0 ?? 0.2, 0.001) * 1e-6 }, { N: seedMol / rhoMolar(id) / (kvS * Ls ** 3), L: Ls }];
  const pb = scalePBE({ rates, classes0, rhoM: rhoMolar(id), kv: kvS, tEnd: Math.max(v.tRes, 1e-3), xiMax: xiEq * kw, nStep: v.nPB ?? 120 });
  const mg = (xi) => xi * M.mw; // mol/m³ → g/m³ = mg/L
  return { id, pb, xiEq, kw, si0, siEnd: siAt(pb.xi[pb.xi.length - 1]), siAt, mgL: pb.xi.map(mg), mgEq: mg(xiEq * kw), conversion: xiEq > 0 ? pb.xi[pb.xi.length - 1] / (xiEq * kw) : 0, seedMgL: (v.seedMass ?? 0) + (Math.max(0, v.seedN ?? 0) * 1e6 * kvS * (Math.max(v.seedL0 ?? 0.2, 0.001) * 1e-6) ** 3 * M.rho * 1e3) };
}

/**
 * Reactive transport along a flow channel: the inlet water displaces the initial water while a mineral precipitates
 * on (or dissolves from) the wall. With equal diffusivities the composition of every cell is fixed by two transported
 * scalars — the inlet-water fraction f and the reaction progress ξ (mol per kg water lost to the wall) — and the
 * saturation index is interpolated from speciation runs tabulated on (f, ξ) (3 × 6 points).
 */
export function reactiveTransport(init, inl, id, o) {
  const M = MINERALS[id], P = o.P ?? 1, dkv = o.dk?.[id] || 0, kw = solutionToIons(inl).kgwPerL * 1000, av = 2 / o.h; // kg water per m³; wall area per volume of a slit
  const nx = 6, nf = 3, T0 = o.table || (() => {
    const fs = [0, 0.5, 1], mixes = fs.map((f) => (f === 0 ? init : f === 1 ? inl : mixSolutions(init, inl, f)));
    // equilibrium amounts bound the table: precipitation capacity (ξ > 0) and dissolution capacity (ξ < 0)
    const cap = mixes.map((s) => { const r = precipitateSolution(s, [id], { P, dk: o.dk || {}, reservoir: { [id]: 0.02 * s.w } }); return r.solids[id] / s.w; });
    const xlo = Math.min(0, ...cap) * 1.15 - 1e-9, xhi = Math.max(0, ...cap) * 1.15 + 1e-9, xs = linspace(xlo, xhi, nx);
    return { cap, xlo, xhi, xs, tab: mixes.map((s) => xs.map((x) => { const q = withdraw(s, id, x); return present(q.eq, id) ? clamp(saturationIndex(q.eq, id, P, dkv), -8, 8) : -8; })) };
  })(), { cap, xlo, xhi, xs, tab } = T0;
  const tb = Float64Array.from(tab.flat()), dxi = (xhi - xlo) / (nx - 1), node = new Float64Array(nx);
  const siAt = (f, xi) => {
    const ff = (f < 0 ? 0 : f > 1 ? 1 : f) * (nf - 1), i = Math.min(nf - 2, Math.floor(ff)), tf = ff - i, xr = (xi - xlo) / dxi, xx = xr < 0 ? 0 : xr > nx - 1 ? nx - 1 : xr, j = Math.min(nx - 2, Math.floor(xx)), tx = xx - j, p = i * nx + j;
    return (1 - tf) * ((1 - tx) * tb[p] + tx * tb[p + 1]) + tf * ((1 - tx) * tb[p + nx] + tx * tb[p + nx + 1]);
  };
  const xiSat = (f, xi, s0) => { // reaction progress at which the cell is exactly saturated: root of the piecewise-linear table along ξ
    const ff = (f < 0 ? 0 : f > 1 ? 1 : f) * (nf - 1), i = Math.min(nf - 2, Math.floor(ff)), tf = ff - i;
    for (let j = 0; j < nx; j++) node[j] = (1 - tf) * tb[i * nx + j] + tf * tb[(i + 1) * nx + j];
    let j = Math.min(nx - 2, Math.max(0, Math.floor((xi - xlo) / dxi)));
    if (s0 > 0) { for (; j < nx - 1; j++) if (node[j + 1] <= 0) return node[j] > 0 ? xlo + (j + node[j] / (node[j] - node[j + 1])) * dxi : Math.max(xi, xlo + j * dxi); return xhi; }
    for (; j >= 0; j--) if (node[j] >= 0) return node[j + 1] < 0 ? xlo + (j + node[j] / (node[j] - node[j + 1])) * dxi : Math.min(xi, xlo + (j + 1) * dxi);
    return xlo;
  };
  const kg = M.kg * (o.kgMult ?? 1) * Math.exp((-45000 / R) * (1 / tk(inl.T) - 1 / 298.15)) * rhoMolar(id), kd = kg * (o.dissMult ?? 10); // mol/m²/s at S − 1 = 1
  const n = Math.max(4, Math.round(o.N ?? 40)), inv = new Array(n).fill(Math.max(0, o.M0 ?? 0)), rate = new Array(n).fill(0); // wall inventory, mol/m²
  let reacted = 0;
  const react = (c, dt) => {
    const f = c[0], xi = c[1];
    for (let i = 0; i < n; i++) {
      rate[i] = 0;
      if (o.wall === 'none') continue;
      const s = siAt(f[i], xi[i]);
      if (Math.abs(s) < 1e-9) continue;
      let d;
      if (o.wall === 'equilibrium') d = xiSat(f[i], xi[i], s) - xi[i];
      else {
        const S = Math.exp((LN10 * s) / M._nu);
        d = S > 1 ? (av * kg * (S - 1) * (S - 1) * dt) / kw : (-av * kd * (1 - S) * dt) / kw;
        if (siAt(f[i], xi[i] + d) * s < 0) { const lim = xiSat(f[i], xi[i], s) - xi[i]; d = S > 1 ? Math.min(d, Math.max(0, lim)) : Math.max(d, Math.min(0, lim)); } // the rate law may not overshoot saturation within a step
      }
      if (d < 0) d = Math.max(d, (-inv[i] * av) / kw); // a wall without inventory cannot dissolve
      xi[i] += d; inv[i] += (d * kw) / av; rate[i] = (d * kw) / av / dt; reacted += d * kw * (o.L / n);
    }
  };
  const tEnd = (o.pv * o.L) / o.u, hist = { t: [], f: [], si: [], xi: [] }, every = Math.max(1, Math.round(((tEnd * o.u) / (0.5 * (o.L / n))) / 80));
  const sol = adr1d({ N: n, L: o.L, u: o.u, D: o.D, tEnd, inlet: o.inlet, fields: [{ cin: 1, c0: 0 }, { cin: 0, c0: 0 }], react, onStep: (c, t, k) => { if (k % every === 0) { hist.t.push(t); hist.f.push(c[0][n - 1]); hist.si.push(siAt(c[0][n - 1], c[1][n - 1])); hist.xi.push(c[1][n - 1]); } } });
  const f = sol.c[0], xi = sol.c[1], si = f.map((ff, i) => siAt(ff, xi[i])), b = sol.balance[1];
  return { id, x: sol.x, f, xi, si, inv, rate, hist, tEnd, kw, av, table: T0, tab, xs, cap, Deff: sol.Deff, Dnum: sol.Dnum, dt: sol.dt, nSteps: sol.nSteps, siAt,
    depositGm2: inv.map((m) => m * M.mw), depositMean: (sum(inv) / n) * M.mw, tracer: sol.balance[0],
    balance: { in: reacted, out: (b.store + b.out - b.in) * kw }, kg, kd };
}

/**
 * Two-dimensional steady flow and solute transport in a membrane feed channel (half height H, membrane at y = 0,
 * symmetry plane at y = H), marched along x in boundary-layer (parabolised) form: at every station the momentum
 * equation d/dy(μ(c)·du/dy) = dp/dx is solved numerically with the local, concentration-dependent viscosity and the
 * flow left after permeation; continuity gives the wall-normal velocity; the solute follows
 * u ∂c/∂x + v ∂c/∂y = D ∂²c/∂y² with the rejecting, permeating wall D ∂c/∂y = −v_w·R·c_w. c is relative to the inlet.
 */
export function channelCFD({ L = 6, H = 3.55e-4, u0 = 0.15, vw = 4e-6, D = 1.5e-9, rej = 1, nx = 60, ny = 30, grow = 1.12, visc = () => 1e-3 }) {
  const NX = Math.max(6, Math.round(nx)), NY = Math.max(6, Math.round(ny)), dx = L / NX;
  // wall-refined grid: cell heights grow geometrically away from the membrane
  const h0 = (H * (grow - 1)) / (grow ** NY - 1), dy = Array.from({ length: NY }, (_, j) => h0 * grow ** j), yc = [], yf = [0];
  dy.forEach((d, j) => { yc.push(yf[j] + d / 2); yf.push(yf[j] + d); });
  const dc = (j) => yc[j + 1] - yc[j]; // distance between centres j and j+1
  let c = new Array(NY).fill(1), q = u0 * H;
  const x = [], cw = [], cb = [], dpdx = [], tauW = [], umax = [], field = [], ufield = [];
  const mu = new Float64Array(NY), ml = new Array(NY), md = new Array(NY), mup = new Array(NY), dcs = Array.from({ length: NY - 1 }, (_, j) => dc(j));
  const momentum = (cc, qq) => { // unit-pressure-gradient solution φ: d/dy(μ dφ/dy) = 1, φ(0) = 0, φ'(H) = 0; u = (dp/dx)·φ
    for (let j = 0; j < NY; j++) mu[j] = visc(cc[j]);
    for (let j = 0; j < NY; j++) {
      const s = j === 0 ? mu[0] / (dy[0] / 2) : (2 * mu[j] * mu[j - 1]) / (mu[j] + mu[j - 1]) / dcs[j - 1], nn = j === NY - 1 ? 0 : (2 * mu[j] * mu[j + 1]) / (mu[j] + mu[j + 1]) / dcs[j];
      ml[j] = j === 0 ? 0 : s; mup[j] = nn; md[j] = -(s + nn);
    }
    const phi = tridiag(ml, md, mup, dy);
    let flow = 0;
    for (let j = 0; j < NY; j++) flow += phi[j] * dy[j];
    const g = qq / flow; // g = dp/dx (negative)
    for (let j = 0; j < NY; j++) phi[j] *= g;
    return { u: phi, g, tau: (mu[0] * phi[0]) / (dy[0] / 2) };
  };
  let saltIn = 0, saltWall = 0, m = momentum(c, q);
  for (let i = 0; i <= NX; i++) {
    const u = m.u;
    if (i === 0) saltIn = sum(u.map((uu, j) => uu * c[j] * dy[j]));
    const cwall = c[0] / (1 - Math.min(0.9, (vw * rej * dy[0]) / (2 * D)));
    x.push(i * dx); cw.push(cwall); cb.push(sum(u.map((uu, j) => uu * c[j] * dy[j])) / Math.max(q, 1e-30)); dpdx.push(m.g); tauW.push(m.tau); umax.push(Math.max(...u)); field.push(c.slice()); ufield.push(u.slice());
    if (i === NX) break;
    // next station: flow after permeation, velocity profile with the lagged viscosity, wall-normal velocity from continuity
    const qn = Math.max(q - vw * dx, 1e-9 * u0 * H), mn = momentum(c, qn), un = mn.u, vf = new Array(NY + 1).fill(0);
    vf[0] = -vw;
    for (let j = 0; j < NY; j++) vf[j + 1] = vf[j] - ((un[j] - u[j]) * dy[j]) / dx;
    const lo = new Array(NY).fill(0), di = new Array(NY).fill(0), up = new Array(NY).fill(0), rhs = new Array(NY);
    for (let j = 0; j < NY; j++) {
      di[j] = (un[j] * dy[j]) / dx; rhs[j] = (u[j] * c[j] * dy[j]) / dx;
      if (j < NY - 1) { const v = vf[j + 1], dif = D / dc(j); di[j] += Math.max(v, 0) + dif; up[j] += Math.min(v, 0) - dif; } // flux through the upper face: v·c(upwind) − D ∂c/∂y
      if (j > 0) { const v = vf[j], dif = D / dc(j - 1); di[j] += -Math.min(v, 0) + dif; lo[j] += -Math.max(v, 0) - dif; }
    }
    // wall face: only the salt passing the membrane leaves, −v_w(1 − R)·c_w, with c_w eliminated through the wall condition
    const wf = 1 / (1 - Math.min(0.9, (vw * rej * dy[0]) / (2 * D)));
    di[0] += vw * (1 - rej) * wf;
    c = tridiag(lo, di, up, rhs);
    saltWall += vw * (1 - rej) * wf * c[0] * dx;
    q = qn; m = mn;
  }
  const uEnd = ufield[NX], saltOut = sum(uEnd.map((uu, j) => uu * field[NX][j] * dy[j]));
  return { x, y: yc, dy, cw, cb, dpdx, tauW, umax, field, ufield, dp: -trapz(x, dpdx), balance: { in: saltIn, out: saltOut + saltWall }, qOut: q, recovery: 1 - q / (u0 * H) };
}

/**
 * Spacer-filled membrane feed channel solved with the two-dimensional finite-volume Navier–Stokes and species
 * solver of suite 4 (SIMPLE-type pressure–velocity coupling on a staggered grid, immersed spacer filaments,
 * solution–diffusion membranes on both walls). Full channel of height h between two membranes, length nFil·lm.
 * The solute is carried relative to the inlet (c = 1). Permeation: J = A·(ΔP − Δπ(c_wall)) with the osmotic
 * pressure function `pi` (Pa, of the relative concentration), or a uniform flux vw when pi is null; salt passage
 * follows the rejection `rej`. Returns the wall-concentration profiles on both membranes, the mixing-cup bulk
 * concentration, the local flux and shear, the concentration field and the salt balance.
 */
export async function spacerChannelCFD({ h = 7.1e-4, u0 = 0.1, vw = 4e-6, D = 1.5e-9, rej = 1, rho = 1000, mu = 1e-3, arr = 'zigzag', lm = 3e-3, df = 3.6e-4, nFil = 6, nxFil = 24, ny = 32, stretch = 8, dP = 0, pi = null, maxIter = 500, tol = 2e-5, scalIter = 300, solver = {} } = {}, ctx) {
  const nF = clamp(Math.round(nFil), 1, 40), L = nF * lm, nx = clamp(Math.round(nxFil), 6, 120) * nF, NY = clamp(2 * Math.round(ny / 2), 8, 160), g = yGrid(h, NY, stretch);
  const mk = buildMask({ type: 'spacer', arr, L, H: h, df, lm, nFil: nF }, nx, NY, g.yc), R = clamp(rej, 0.5, 1), osm = typeof pi === 'function';
  // uniform-flux mode: a very large driving pressure makes the flux independent of the (small) hydraulic pressure variation
  const dPm = osm ? dP : 1e9, pi0 = osm ? pi(1) * R : 0, A = vw / Math.max(dPm - pi0, 1e-9), B = R < 1 ? (vw * (1 - R)) / R : 0;
  const r = await solveChannel({ L, H: h, nx, ny: NY, stretch, solid: mk.solid, rho, mu, Uin: u0, inlet: 'parabolic', scheme: 'hybrid', steady: true, maxIter, tol, scalIter, ...solver,
    species: { c0: 1, D, A, B, dP: dPm, pi: osm ? pi : () => 0, bot: 'membrane', top: 'membrane' } }, ctx);
  const { dx, dy, yc, u, nu1, solid } = r, phi = r.spc.phi, x = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx);
  const cb = new Array(nx), blockB = new Array(nx), blockT = new Array(nx);
  for (let i = 0; i < nx; i++) {
    let q = 0, qc = 0;
    for (let j = 0; j < NY; j++) { const P = j * nx + i; if (solid[P]) continue; const uc = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); q += uc * dy[j]; qc += uc * phi[P] * dy[j]; }
    cb[i] = q > 0 ? qc / q : 1; blockB[i] = !!solid[i]; blockT[i] = !!solid[(NY - 1) * nx + i];
  }
  let sin = 0, sout = 0, qin = 0, qout = 0, perm = 0;
  for (let j = 0; j < NY; j++) { sin += r.uin[j] * dy[j]; qin += r.uin[j] * dy[j]; const uo = u[j * nu1 + nx]; qout += uo * dy[j]; sout += uo * phi[j * nx + nx - 1] * dy[j]; }
  for (let i = 0; i < nx; i++) { sout += (r.Jb[i] * r.spc.pB[i] + r.Jt[i] * r.spc.pT[i]) * dx; perm += (r.Jb[i] + r.Jt[i]) * dx; }
  const open = (blk) => blk.filter((b) => !b).length || 1, Jmean = perm / ((open(blockB) + open(blockT)) * dx);
  return { x, y: Array.from(yc), dx, dy: Array.from(dy), nx, ny: NY, L, h, cwB: Array.from(r.spc.wB), cwT: Array.from(r.spc.wT), cb, JB: Array.from(r.Jb), JT: Array.from(r.Jt), tauB: Array.from(r.tauB), tauT: Array.from(r.tauT), blockB, blockT,
    field: Array.from({ length: NY }, (_, j) => Array.from({ length: nx }, (_, i) => phi[j * nx + i])), mask: Array.from({ length: NY }, (_, j) => Array.from({ length: nx }, (_, i) => !!solid[j * nx + i])), shapes: mk.shapes,
    Jmean, recovery: perm / qin, massError: (qin - qout - perm) / qin, balance: { in: sin, out: sout }, converged: r.converged, iters: r.iters, scalRes: r.scalRes, A, B, osmotic: osm, solidFraction: mk.solidFraction };
}

// ---- surrogate of the saturation index: Gaussian-kernel ridge regression ------------------------------------
function cholesky(A) {
  const n = A.length, Lm = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= Lm[i][k] * Lm[j][k];
    if (i === j) { if (!(s > 0)) return null; Lm[i][i] = Math.sqrt(s); } else Lm[i][j] = s / Lm[j][j];
  }
  return Lm;
}
const cholSolve = (Lm, b) => { const n = b.length, y = new Float64Array(n), x = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= Lm[i][k] * y[k]; y[i] = s / Lm[i][i]; } for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= Lm[k][i] * x[k]; x[i] = s / Lm[i][i]; } return x; };
/**
 * Kernel ridge regression with a Gaussian kernel on standardised inputs. The length scale is chosen by closed-form
 * leave-one-out cross-validation on the training set. X: rows of features, Y: rows of targets (several outputs).
 */
export function kernelRidge(X, Y, { scales = [0.6, 1, 1.6, 2.6], nugget = 1e-6 } = {}) {
  const n = X.length, d = X[0].length, nOut = Y[0].length, mu = [], sd = [];
  for (let j = 0; j < d; j++) { const col = X.map((r) => r[j]), m = sum(col) / n; mu.push(m); sd.push(Math.sqrt(sum(col.map((x) => (x - m) ** 2)) / n) || 1); }
  const Z = X.map((r) => r.map((x, j) => (x - mu[j]) / sd[j])), ym = [], Yc = Y.map((r) => r.slice());
  for (let o = 0; o < nOut; o++) { const m = sum(Y.map((r) => r[o])) / n; ym.push(m); Yc.forEach((r) => { r[o] -= m; }); }
  const d2 = Z.map((a) => Z.map((b) => { let s = 0; for (let j = 0; j < d; j++) s += (a[j] - b[j]) ** 2; return s; }));
  let best = null;
  for (const ell of scales) {
    const Kmat = d2.map((row, i) => row.map((x, j) => Math.exp(-x / (2 * ell * ell)) + (i === j ? nugget : 0))), Lm = cholesky(Kmat);
    if (!Lm) continue;
    const alpha = Array.from({ length: nOut }, (_, o) => cholSolve(Lm, Yc.map((r) => r[o]))), dinv = new Float64Array(n);
    // diagonal of K⁻¹ = column norms of L⁻¹: one forward substitution per column, started at its first non-zero entry
    const yv = new Float64Array(n);
    for (let i = 0; i < n; i++) { let d = 0; for (let k = i; k < n; k++) { let t = k === i ? 1 : 0; const Lk = Lm[k]; for (let j = i; j < k; j++) t -= Lk[j] * yv[j]; yv[k] = t / Lk[k]; d += yv[k] * yv[k]; } dinv[i] = d; }
    let loo = 0;
    for (let o = 0; o < nOut; o++) for (let i = 0; i < n; i++) loo += (alpha[o][i] / dinv[i]) ** 2;
    loo = Math.sqrt(loo / (n * nOut));
    if (!best || loo < best.loo) best = { ell, alpha, loo };
  }
  if (!best) throw new Error('kernel matrix is not positive definite');
  const predict = (x) => { const z = x.map((v, j) => (v - mu[j]) / sd[j]), k = Z.map((a) => { let s = 0; for (let j = 0; j < d; j++) s += (a[j] - z[j]) ** 2; return Math.exp(-s / (2 * best.ell * best.ell)); }); return ym.map((m, o) => { let s = m; for (let i = 0; i < n; i++) s += best.alpha[o][i] * k[i]; return s; }); };
  return { predict, ell: best.ell, loo: best.loo, n };
}
/** Latin-hypercube points in the unit cube (deterministic). */
function lhsPts(n, d, seed) {
  const g = rng(seed), cols = Array.from({ length: d }, () => { const p = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = g.int(i + 1); [p[i], p[j]] = [p[j], p[i]]; } return p.map((i) => (i + g.uniform()) / n); });
  return Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
}
/** Train and test a surrogate of the wall saturation indices over recovery, feed pH and temperature. */
export function trainSurrogate(a, ids, { nTrain = 48, nTest = 16, seed = 11 } = {}) {
  const rng0 = [[0, a.Rmax], [5.5, 9], [5, 60]], feat = (r, pH, T) => [Math.log(1 / (1 - r)), pH, T];
  const engine = (r, pH, T) => { const e = concentrateSolution(equilibrate(a.raw, { pH, T }), a.beta / (1 - r), a.copt).eq; return ids.map((id) => clamp(saturationIndex(e, id, a.P, a.dk[id] || 0), -12, 12)); };
  const pts = lhsPts(nTrain + nTest, 3, seed).map((p) => p.map((x, j) => rng0[j][0] + x * (rng0[j][1] - rng0[j][0])));
  const X = pts.map((p) => feat(...p)), Y = pts.map((p) => engine(...p)), model = kernelRidge(X.slice(0, nTrain), Y.slice(0, nTrain));
  const pred = X.slice(nTrain).map(model.predict), meas = Y.slice(nTrain), stats = ids.map((id, o) => { const m = meas.map((r) => r[o]), p = pred.map((r) => r[o]), mm = sum(m) / m.length, sse = sum(m.map((x, i) => (x - p[i]) ** 2)), sst = sum(m.map((x) => (x - mm) ** 2)); return { id, rmse: Math.sqrt(sse / m.length), r2: sst > 0 ? 1 - sse / sst : 1, maxErr: Math.max(...m.map((x, i) => Math.abs(x - p[i]))) }; });
  return { ids, model, feat, engine, X, Y, predict: (r, pH, T) => model.predict(feat(r, pH, T)), meas, pred, stats, nTrain, nTest, rmse: Math.sqrt(sum(stats.map((s) => s.rmse ** 2)) / stats.length), ranges: rng0 };
}

// ---- suite ------------------------------------------------------------------------------------------
const MODEL_OPTS = Object.entries(ACTIVITY_MODELS).map(([value, label]) => ({ value, label }));
const NACL_LIT = { m: [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6], g: [0.779, 0.734, 0.681, 0.657, 0.668, 0.714, 0.783, 0.874, 0.986] }; // Hamer & Wu (1972), 25 °C
// Critically evaluated mean activity coefficients γ± and osmotic coefficients φ at 25 °C, rows [molality, γ±, φ], as
// printed in the NIST compilations: Hamer & Wu (1972, J. Phys. Chem. Ref. Data 1, 1047: tables 16 and 28), Goldberg &
// Nuttall (1978, 7, 263: tables 17 and 20) and Goldberg (1981, 10, 671). lim = [model, highest molality used, tolerance]:
// each model is only tested inside its validity range (Davies I ≤ 0.3, SIT I ≤ 3, Bromley I ≤ 6 mol/kg).
const ACT_REF = {
  NaCl: { c: 'Na', a: 'Cl', src: 'Hamer & Wu 1972', d: [[0.01, 0.903, 0.968], [0.05, 0.822, 0.944], [0.1, 0.779, 0.933], [0.2, 0.734, 0.924], [0.5, 0.681, 0.921], [1, 0.657, 0.936], [2, 0.668, 0.984], [3, 0.714, 1.045], [4, 0.783, 1.116], [5, 0.874, 1.191], [6, 0.986, 1.27]], lim: [['pitzer', 6, 0.006], ['bromley', 6, 0.02], ['sit', 3, 0.035], ['davies', 0.2, 0.03]] },
  KCl: { c: 'K', a: 'Cl', src: 'Hamer & Wu 1972', d: [[0.1, 0.768, 0.927], [0.2, 0.717, 0.913], [0.5, 0.649, 0.9], [1, 0.604, 0.898], [2, 0.573, 0.912], [3, 0.568, 0.936], [4, 0.576, 0.965]], lim: [['pitzer', 4, 0.005], ['bromley', 4, 0.008], ['sit', 3, 0.045], ['davies', 0.2, 0.06]] },
  'MgCl₂': { c: 'Mg', a: 'Cl', src: 'Goldberg & Nuttall 1978', d: [[0.1, 0.5347, 0.8648], [0.2, 0.4935, 0.876], [0.5, 0.4855, 0.9475], [1, 1 * 0.5769, 1.1092], [2, 1.0655, 1.525], [3, 2.3498, 2.0125], [4, 5.6692, 2.5313], [5, 14.396, 3.0645]], lim: [['pitzer', 5, 0.03], ['bromley', 1, 0.03], ['sit', 1, 0.035], ['davies', 0.1, 0.035]] },
  'CaCl₂': { c: 'Ca', a: 'Cl', src: 'Goldberg & Nuttall 1978', d: [[0.1, 0.5171, 0.8516], [0.2, 0.4692, 0.8568], [0.5, 0.4442, 0.9134], [1, 0.4956, 1.0444], [2, 0.7842, 1.3754], [3, 1.455, 1.7685]], lim: [['pitzer', 3, 0.03], ['bromley', 2, 0.035], ['sit', 1, 0.02], ['davies', 0.1, 0.055]] },
  'Na₂SO₄': { c: 'Na', a: 'SO4', src: 'Goldberg 1981', d: [[0.1, 0.4457, 0.7869], [0.5, 0.2684, 0.6945], [1, 0.204, 0.6481], [2, 0.1546, 0.6257]], lim: [['pitzer', 2, 0.03], ['bromley', 1, 0.035], ['sit', 1, 0.045], ['davies', 0.1, 0.065]] },
  'K₂SO₄': { c: 'K', a: 'SO4', src: 'Goldberg 1981', d: [[0.1, 0.4239, 0.7687], [0.2, 0.3429, 0.7304], [0.5, 0.2514, 0.6875]], lim: [['pitzer', 0.5, 0.065], ['bromley', 0.5, 0.07], ['sit', 0.5, 0.04], ['davies', 0.1, 0.07]] },
};
/** Largest relative deviation of γ± and φ of one model from a reference table, up to the molality mMax. */
function actDeviation(ref, model, mMax) {
  const rows = ref.d.filter((r) => r[0] <= mMax + 1e-12);
  let dg = 0, dp = 0;
  for (const [m, g, ph] of rows) { const q = saltActivity(ref.c, ref.a, m, { model }); dg = Math.max(dg, Math.abs(q.gamma / g - 1)); dp = Math.max(dp, Math.abs(q.phi / ph - 1)); }
  return { dg, dp, n: rows.length, m0: rows[0][0], m1: rows[rows.length - 1][0] };
}
/** Single-salt Pitzer equations in their textbook closed form (Pitzer 1973; Pitzer & Mayorga 1974 for 2:2 salts) — an independent evaluation of the ion-interaction sums. */
function pitzerSingle(zc, za, m, b0, b1, b2, cphi, T = 25) {
  const nc = za === zc ? 1 : za, na = za === zc ? 1 : zc, nu = nc + na, I = 0.5 * m * (nc * zc * zc + na * za * za), s = Math.sqrt(I), A = aphi(T), a1 = zc === 2 && za === 2 ? 1.4 : 2, a2 = 12;
  const gf = (x) => (2 * (1 - (1 + x - 0.5 * x * x) * Math.exp(-x))) / (x * x), f = -A * (s / (1 + 1.2 * s) + (2 / 1.2) * Math.log(1 + 1.2 * s));
  const Bg = 2 * b0 + b1 * gf(a1 * s) + b2 * gf(a2 * s), Bp = b0 + b1 * Math.exp(-a1 * s) + b2 * Math.exp(-a2 * s);
  return { gamma: Math.exp(zc * za * f + m * ((2 * nc * na) / nu) * Bg + m * m * ((2 * (nc * na) ** 1.5) / nu) * 1.5 * cphi), phi: 1 - (zc * za * A * s) / (1 + 1.2 * s) + m * ((2 * nc * na) / nu) * Bp + m * m * ((2 * (nc * na) ** 1.5) / nu) * cphi };
}
/** Where every constant set of the suite comes from and how it was checked. Status: confirmed, replaced, analogue or unconfirmed. */
const PROVENANCE = [
  ['Pitzer β⁰, β¹, β², Cφ, θ, ψ and CO₂ λ of the Na–K–Mg–Ca–H–Cl–SO₄–OH–HCO₃–CO₃–CO₂ system', 'Harvie, Møller & Weare (1984), read from the LLNL EQ3/6 database file data0.hmw', '173 numbers compared by script: all identical', '25 °C; to salt saturation (I ≈ 20 mol/kg)', 'confirmed'],
  ['Pitzer parameters of SrCl₂, BaCl₂, SrSO₄ and borate (B(OH)₄⁻, B(OH)₃ λ)', 'USGS PHREEQC pitzer.dat, PHRQPITZ lineage (Plummer et al. 1988; Felmy & Weare 1986)', '18 numbers compared; BaCl₂ β¹ and Cφ had been rounded and were restored', '25 °C', 'confirmed'],
  ['Pitzer parameters of NaF, KF, KNO₃, Mg(NO₃)₂, Na₂HPO₄, K₂HPO₄, θ(Cl,NO₃)', 'Pitzer (1991) tabulation, read from the LLNL EQ3/6 Pitzer file data0.ypf', '19 numbers compared; Mg(NO₃)₂ restored to full precision, K₂HPO₄ added', '25 °C', 'confirmed'],
  ['Pitzer parameters of NaNO₃ and Ca(NO₃)₂', 'LLNL EQ3/6 data0.ypf (revision 0): refits with α₁ = 2 to the Archer (2000) and Oakes et al. (2000) evaluations', 'The earlier values could not be found in a retrievable file and were replaced by this set', '25 °C; NaNO₃ checked here against Hamer & Wu to 6 mol/kg', 'replaced'],
  ['Pitzer λ of dissolved silica with Na⁺, K⁺, Mg²⁺, Ca²⁺, SO₄²⁻', 'USGS PHREEQC pitzer.dat (Appelo 2015)', 'The earlier values could not be found in any database and were replaced', '25 °C', 'replaced'],
  ['Pitzer analogues: Ba–SO₄ (uses Ca–SO₄), NH₄⁺ (uses K⁺), Fe²⁺ and Mn²⁺ (use Mg²⁺), H₃SiO₄⁻ (uses HCO₃⁻)', 'Assignment by chemical similarity, not a measured parameter set', 'Not applicable', 'Trace constituents only', 'analogue'],
  ['Debye–Hückel slope Aφ(T)', 'Grid of the LLNL EQ3/6 Pitzer file data0.ypf (0.3767, 0.3915, 0.4190, 0.4605 at 0, 25, 60, 100 °C)', 'Fit reproduces the four grid values within 0.0005', '0–100 °C', 'confirmed'],
  ['Carbonate, water, silicate, borate, HSO₄⁻ and HF dissociation; CO₂ Henry constant (ion-pair models)', 'USGS WATEQ4F database wateq4f.dat (Plummer & Busenberg 1982; Ball & Nordstrom 1991); borate log K as in MINTEQA2 v4', 'All coefficients of the six temperature functions and four log K/ΔH pairs compared', '0–90 °C', 'confirmed'],
  ['The same constants for the Pitzer and Bromley species set (pK₂ 10.339, pK₁ 6.337, HSO₄⁻ 1.979)', 'USGS PHRQPITZ/PHREEQC pitzer.dat; 25 °C values equal to data0.hmw', 'Corrected: the ion-pair values had been used with the Pitzer model', '25 °C exact, 0–90 °C by the shifted temperature function', 'corrected'],
  ['Ion-pair constants (CaSO₄°, MgSO₄°, NaSO₄⁻, KSO₄⁻, CaHCO₃⁺, MgHCO₃⁺, NaHCO₃°, NaCO₃⁻, CaCO₃°, MgCO₃°, CaOH⁺, MgOH⁺, CaF⁺, MgF⁺, BaSO₄°, SrSO₄°)', 'USGS WATEQ4F database wateq4f.dat (CaOH⁺ ΔH from MINTEQA2 v4); Pitzer-set CaCO₃°, MgCO₃°, MgOH⁺ from data0.hmw', '16 log K and 12 ΔH compared: identical after rounding; two Pitzer-set ΔH adjusted to pitzer.dat', 'I < 0.7 mol/kg', 'confirmed'],
  ['Ion-size å and b of the Truesdell–Jones and extended Debye–Hückel models', 'USGS WATEQ4F database wateq4f.dat', '18 ions compared; b of Ba²⁺, NH₄⁺, Fe²⁺, Mn²⁺, NO₃⁻, F⁻, OH⁻ and å of HPO₄²⁻ corrected', 'I < 1 mol/kg', 'corrected'],
  ['Ion-size default (å = 4, b = 0.041) for species without a tabulated entry', 'Model assumption', 'No source', 'Ion pairs and minor species only', 'unconfirmed'],
  ['log K(T) of calcite, aragonite, gypsum, anhydrite, barite, celestite, fluorite, amorphous silica, strontianite, witherite, siderite, dolomite', 'USGS WATEQ4F database wateq4f.dat', 'All analytic coefficients and ΔH compared: identical; silica now carries the water activity of SiO₂ + 2 H₂O = H₄SiO₄', '0–90 °C', 'confirmed'],
  ['log K of calcite and aragonite with the Pitzer model (−8.406, −8.219)', 'Harvie, Møller & Weare (1984) in data0.hmw; temperature function of pitzer.dat', 'Corrected (was the ion-pair value −8.480, −8.336); verified against the seawater solubility of Mucci (1983)', '25 °C exact', 'corrected'],
  ['log K of halite, sylvite, the Na/Mg/K/Ca sulphate and chloride salts, brucite, portlandite, magnesite, nesquehonite', 'Harvie, Møller & Weare (1984) in data0.hmw and USGS PHREEQC pitzer.dat (PHRQPITZ lineage)', '19 values at 25 °C compared: identical to 0.001 (mirabilite follows pitzer.dat, −1.214)', '25 °C', 'confirmed'],
  ['The same minerals with the ion-pair models (brucite −11.16, magnesite −8.03, nesquehonite −5.62, epsomite −2.14, mirabilite −1.11, thenardite −0.18, halite 1.58)', 'USGS WATEQ4F database wateq4f.dat', 'Added: the Pitzer-set values had been used with every model', 'I < 0.7 mol/kg', 'corrected'],
  ['Temperature dependence of sylvite, hexahydrite, bischofite (analytic) and kieserite (ΔH −29 kJ/mol)', 'USGS PHREEQC pitzer.dat (PHRQPITZ expressions; kieserite slope from the Appelo 2015 expression)', 'Replaced: the earlier ΔH of hexahydrite and kieserite were not found in a database', '0–100 °C, indicative', 'replaced'],
  ['SIT interaction coefficients ε(cation, anion)', 'OECD-NEA thermochemical database, 2020 update of the SIT tables (B-6, B-7); ThermoChimie sit.dat of USGS PHREEQC', '26 of 27 pairs identical in the NEA tables; 19 also in sit.dat; FeCl₂, MnCl₂, NaH₃SiO₄ added from sit.dat', 'I ≤ 3 mol/kg', 'confirmed'],
  ['SIT ε(Sr²⁺, Cl⁻)', 'Not in the retrieved tables: the Ca²⁺ value is used as an analogue', 'Not applicable', 'Trace constituent', 'analogue'],
  ['Bromley salt constants B of NaCl, KCl, HCl, NH₄Cl, CaCl₂, MgCl₂, Na₂SO₄, K₂SO₄, NaNO₃, KNO₃, NaOH, KOH', 'Bromley (1973); the paper could not be retrieved, so each B was re-derived by least squares from the NIST tables (Hamer & Wu 1972; Goldberg & Nuttall 1978; Goldberg 1981)', 'Refit agrees within 0.003 kg/mol (K₂SO₄ within 0.013)', 'I ≤ 6 mol/kg; not for 2:2 salts', 'confirmed by refit'],
  ['Bromley salt constants B of SrCl₂, BaCl₂, MgSO₄ and the individual-ion table (B₊, B₋, δ₊, δ₋) used for all other salts', 'Bromley (1973), not retrievable', 'Not checked — the Bromley model is opt-in and never used by other suites', 'I ≤ 6 mol/kg', 'unconfirmed'],
  ['Hydrous ferric oxide: site density, surface area, protonation and sorption constants', 'Dzombak & Morel (1990) and Swedlund & Webster (1999), read from the SURFACE_SPECIES block of USGS PHREEQC phreeqc.dat', '10 constants, 600 m²/g and 0.2 mol/mol compared: identical; solver reproduces PHREEQC example 8', '25 °C, I < 0.7 mol/kg', 'confirmed'],
  ['Interfacial energies, growth constants and antiscalant limits of the minerals', 'Order-of-magnitude engineering defaults', 'Not source-checked; adjustable through the kinetic inputs and the limit fields', 'Screening only', 'unconfirmed'],
];
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
  // optional pretreatment: sorption of silica and boron on ferric hydroxide, then sodium-cycle softening
  let scm = null, ix = null;
  if (v.scmOn && v.feDose > 0) {
    scm = surfaceComplexation(raw.eq, v.feDose, { kgwPerL: raw.kgwPerL || 1 });
    const n = Float64Array.from(raw.n);
    n[mi('Si')] *= 1 - scm.removal.Si; n[mi('B')] *= 1 - scm.removal.B;
    scm.before = raw; raw = equilibrate({ ...raw, n }, { pH: raw.pH });
  }
  if (v.ixOn) {
    ix = softenerColumn(raw, { cap: v.ixCap, logKCa: v.ixKCa, logKMg: v.ixKMg, regenDose: v.ixRegen, regenPct: v.ixRegenPct, leakFrac: clamp((v.ixLeak ?? 5) / 100, 0.001, 0.9), nStage: v.ixStages ?? 12, model, T: raw.T });
    const kw = raw.kgwPerL || 1, n = Float64Array.from(raw.n), iCa = mi('Ca'), iMg = mi('Mg'), dCa = Math.max(0, n[iCa] - (ix.soft.Ca / 2 / kw) * raw.w), dMg = Math.max(0, n[iMg] - (ix.soft.Mg / 2 / kw) * raw.w);
    n[iCa] -= dCa; n[iMg] -= dMg; n[mi('Na')] += 2 * (dCa + dMg);
    ix.before = raw; raw = equilibrate({ ...raw, n });
  }
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
  return { v, model, dk, lim, set, P, raw, unmixed, other, scm, ix, feed, dose, conc, wall, R, rej, beta, Rs, Rmax, sweep, maxRec, limitPlain: pick('plain'), limitAS: pick('as'), concAt, copt, useRo, cf: cfOf(R) };
}

/** Compact number formatting for notes (significant digits, no locale lookup). */
const fq = (x, sig = 4) => (typeof x !== 'number' ? String(x ?? '–') : !Number.isFinite(x) ? '–' : x === 0 ? '0' : Math.abs(x) >= 1e7 || Math.abs(x) < 1e-4 ? x.toExponential(Math.max(1, sig - 1)) : String(+x.toPrecision(sig)));
const SCM_NOTE = 'Generalised two-layer model for hydrous ferric oxide (Dzombak & Morel constants at 25 °C; silicate constants of Swedlund & Webster). Only weak sites are counted; major ions are not depleted by sorption.';
/** Extended models of run(): Gibbs minimisation, population balance, reactive transport, channel flow, sorption, exchange, surrogate. */
function extendedModels(v, a, c) {
  const { conc, wall, feed, P, dk } = a, K = [], PL = [], TB = [], W = [], BAL = [], out = {}, mmol = (x, s) => (x / s.w) * 1000;
  let march = null, sur = null;
  // 1 — Gibbs-energy minimisation of the bulk concentrate (closed system), compared with the mass-action result
  if (v.gemOn !== false) {
    const ma = v.co2 === 'open' ? precipitateSolution(conc, c.pset, { P, dk, reservoir: c.reservoir }) : c.pr, g = gibbsMinimize(conc, c.pset, { P, dk, reservoir: c.reservoir });
    if (!g.converged) {
      W.push({ level: 'warn', msg: 'The Gibbs-energy minimisation did not converge for this brine (outside the range of the activity model); the mass-action result stands without cross-check.' });
      K.push({ label: 'Gibbs minimisation vs mass action', value: 'not converged', status: 'warn' });
    } else {
    const RT = R * tk(conc.T), amt = (sol) => Object.fromEntries(Object.keys(sol).map((id) => [id, sol[id] + (c.reservoir[id] || 0)])), G0 = gibbsEnergy(conc, c.reservoir, { P, dk }), Gma = gibbsEnergy(ma.sol, amt(ma.solids), { P, dk });
    const ids = c.pset.filter((id) => Math.abs(ma.solids[id]) > 1e-12 || Math.abs(g.solids[id]) > 1e-12), dpH = Math.abs(g.pH - ma.sol.pH), dS = Math.max(0, ...ids.map((id) => Math.abs(g.solids[id] - ma.solids[id]) / conc.w)) * 1000, rel = (-(g.G - G0) * RT) / conc.w;
    TB.push({ title: 'Equilibrium by Gibbs-energy minimisation versus mass action', columns: ['Quantity', 'Mass action', 'Gibbs minimisation', 'Difference', 'Unit'], rows: [
      ['pH at equilibrium', ma.sol.pH, g.pH, g.pH - ma.sol.pH, ''], ...ids.map((id) => [`${MINERALS[id].name} ${ma.solids[id] < 0 ? 'dissolved' : 'precipitated'}`, mmol(ma.solids[id], conc), mmol(g.solids[id], conc), mmol(g.solids[id] - ma.solids[id], conc), 'mmol/kg water']),
      ['Ionic strength', ma.sol.eq.I, g.sol.eq.I, g.sol.eq.I - ma.sol.eq.I, 'mol/kg'], ['Gibbs energy of the final state, G/RT', Gma, g.G, g.G - Gma, 'mol'], ['Gibbs energy before precipitation, G/RT', G0, G0, 0, 'mol'], ['Gibbs energy released', (-(Gma - G0) * RT) / conc.w, rel, (-(g.G - Gma) * RT) / conc.w, 'J per kg water']],
      note: `Closed system. The minimiser varies ${g.lambda.length} element potentials and the amounts of ${g.active.length} solid phase${g.active.length === 1 ? '' : 's'} (${g.iterations} Newton steps, largest balance residual ${fq(g.residual, 2)}); it starts from a neutral solution of free ions and does not use the mass-action solver. Reference state: master species, H⁺ and H₂O.` });
    K.push({ label: 'Gibbs energy released by precipitation', value: rel, unit: 'J/kg water', help: 'Drop of the total Gibbs energy between the supersaturated concentrate and its equilibrium state (Gibbs-energy minimisation)' }, { label: 'Gibbs minimisation vs mass action', value: Math.max(dpH, dS), unit: 'pH · mmol/kg', status: g.converged && Math.max(dpH, dS) < 1e-3 ? 'ok' : 'warn', help: 'Largest difference in equilibrium pH or precipitated amount between the two independent solvers' });
    out.gibbsRelease = rel; out.gemPH = g.pH; out.gemDifference = Math.max(dpH, dS);
    }
  }
  // 2 — population balance of the precipitating mineral in the bulk concentrate
  const kinId = pickMineral(v, a, conc);
  if (kinId) {
    const q = scaleKinetics(conc, kinId, v, { P, dk }), pb = q.pb, M = MINERALS[kinId], last = pb.t.length - 1, nTot = pb.mu[0];
    const ord = pb.L.map((L, i) => [L, pb.N[i]]).sort((x, y) => x[0] - y[0]);
    let cum = 0;
    const cdf = ord.map(([L, N]) => { cum += N; return [L * 1e6, nTot > 0 ? cum / nTot : 0]; });
    PL.push({ type: 'line', title: `Precipitation kinetics of ${M.name.toLowerCase()} in the concentrate (population balance)`, xlabel: 'Time (s)', ylabel: 'Solid formed (mg/L)', series: [{ name: 'Precipitated by nucleation and growth', x: pb.t, y: q.mgL }, { name: 'Potential still dissolved', x: pb.t, y: q.mgL.map((m) => Math.max(0, q.mgEq - m)), dash: true }], hlines: [{ y: q.mgEq, label: 'equilibrium potential' }], note: `Initial condition: ${fq(v.seedN, 3)} nuclei per mL of ${v.seedL0} µm and ${fq(v.seedMass, 3)} mg/L of ${v.seedL} µm precipitate.` });
    PL.push({ type: 'line', title: `Crystal population of ${M.name.toLowerCase()} after the residence time`, xlabel: 'Crystal size (µm)', ylabel: 'Cumulative number fraction (–)', logx: true, ymin: 0, ymax: 1, series: [{ name: 'Number distribution', x: cdf.length ? cdf.map((p) => p[0]) : [0.001, 1], y: cdf.length ? cdf.map((p) => p[1]) : [0, 0], mode: 'step' }] });
    TB.push({ title: 'Population balance of the precipitating scale', columns: ['Quantity', 'Value', 'Unit', 'Meaning'], rows: [
      ['Mineral', M.name, '', v.pbMineral === 'auto' || !v.pbMineral ? 'Highest supersaturation per ion' : 'Selected'], ['Saturation index at the start → end', `${fq(q.si0, 3)} → ${fq(q.siEnd, 3)}`, '', 'Driving force is consumed by the growing crystals'], ['Equilibrium precipitation potential', q.mgEq, 'mg/L', 'Mass-action equilibrium of this mineral alone'],
      ['Precipitated within the residence time', q.mgL[last], 'mg/L', `${fq(100 * q.conversion, 3)} % of the potential in ${v.tRes} s`], ['Initial solids (nuclei + precipitate)', q.seedMgL, 'mg/L', 'Initial condition of the population balance'], ['Crystal number at the start', pb.number[0] / 1e6, '#/mL', 'Initial nuclei and precipitate'], ['Crystal number at the end', pb.number[last] / 1e6, '#/mL', 'Initial + nucleated'],
      ['Mass-mean size L₄₃', pb.L43[last] * 1e6, 'µm', 'Final'], ['Linear growth rate', pb.G[0] * 1e9, 'nm/s', 'At the start'], ['log₁₀ nucleation rate', Math.log10(Math.max(pb.J[0], 1e-300)), 'log(m⁻³s⁻¹)', 'At the start']],
      note: `Method of characteristics with ${pb.N.length} crystal classes and ${pb.subSteps} time steps; nucleation and growth rates follow the saturation index recomputed by the speciation model as the mineral leaves the solution.${pb.complete ? '' : ' The integration stopped early (very stiff kinetics).'}` });
    K.push({ label: `${M.name} formed in the residence time`, value: q.mgL[last], unit: 'mg/L', status: q.conversion > 0.2 && q.si0 > 0 ? 'warn' : 'ok', help: `Population balance with nucleation and growth: ${fq(100 * q.conversion, 3)} % of the equilibrium potential` });
    if (q.conversion > 0.2 && q.si0 > 0) W.push({ level: 'warn', msg: `The population balance predicts that ${fq(100 * q.conversion, 3)} % of the ${M.name.toLowerCase()} potential precipitates within the ${v.tRes} s residence time — bulk crystallisation, not only wall scaling, is to be expected.` });
    out.kineticMineral = kinId; out.kineticPrecipitate = q.mgL[last]; out.kineticConversion = q.conversion;
  }
  // 3 — reactive transport along the channel
  const mode = v.rtMode || 'off', L = Math.max(v.rtL ?? 1, 0.01), u = Math.max(v.rtU ?? 0.1, 1e-4), h = Math.max(v.rtH ?? 0.71, 0.05) * 1e-3;
  if (mode !== 'off' && kinId) {
    const M = MINERALS[kinId], flush = mode === 'flush', rt = reactiveTransport(flush ? wall : feed, flush ? feed : wall, kinId, { P, dk, h, L, u, D: Math.max(v.rtDisp ?? 0.02, 0) * u + 1.5e-9, pv: Math.max(v.rtPV ?? 3, 0.2), inlet: v.rtInlet || 'conc', wall: v.rtWall || 'kinetic', N: v.rtN ?? 40, kgMult: v.kgMult, dissMult: v.rtDiss ?? 10, M0: flush ? Math.max(v.rtM0 ?? 0, 0) / M.mw : 0 });
    const day = rt.rate.map((r) => r * M.mw * 86400), pvs = rt.hist.t.map((t) => (t * u) / L), wallName = { kinetic: 'kinetic surface reaction', equilibrium: 'local equilibrium', none: 'zero flux (inert wall)' }[v.rtWall || 'kinetic'];
    PL.push({ type: 'line', title: `Reactive transport: ${M.name.toLowerCase()} saturation along the channel after ${fq(v.rtPV, 3)} channel volumes`, xlabel: 'Distance from inlet (m)', ylabel: 'Saturation index · inlet-water fraction', series: [{ name: 'Saturation index', x: rt.x, y: rt.si }, { name: 'Inlet-water fraction', x: rt.x, y: rt.f, dash: true }], hlines: [{ y: 0, label: 'saturation' }], note: `${flush ? 'Flush: feed water displaces the concentrate and dissolves the wall deposit.' : 'Start-up: the concentrate displaces the feed water.'} Inlet: ${v.rtInlet === 'flux' ? 'prescribed flux' : 'prescribed concentration'}; outlet: convective; wall: ${wallName}.` });
    PL.push({ type: 'line', title: 'Reactive transport: wall deposit', xlabel: 'Distance from inlet (m)', ylabel: 'Deposit (g/m²) · rate (g/m²·d)', series: [{ name: 'Deposit on the wall (g/m²)', x: rt.x, y: rt.depositGm2 }, { name: 'Wall flux at the end (g/m²·d)', x: rt.x, y: day, dash: true }] });
    PL.push({ type: 'line', title: 'Reactive transport: outlet history', xlabel: 'Channel volumes passed (–)', ylabel: 'Inlet-water fraction · saturation index', series: [{ name: 'Inlet-water fraction at the outlet', x: pvs, y: rt.hist.f }, { name: 'Saturation index at the outlet', x: pvs, y: rt.hist.si }], vlines: [{ x: 1, label: 'one volume' }] });
    TB.push({ title: 'Reactive transport along the channel', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [
      ['Mineral', M.name, '', flush ? 'Dissolving from the wall inventory' : 'Precipitating on the wall'], ['Inlet condition', v.rtInlet === 'flux' ? 'Prescribed species flux (Danckwerts)' : 'Prescribed concentration', '', 'Outlet: convective (zero dispersive flux)'], ['Wall condition', wallName, '', v.rtWall === 'none' ? 'No exchange with the wall' : 'Precipitation above, dissolution below saturation'],
      ['Saturation index at inlet → outlet', `${fq(rt.siAt(1, 0), 3)} → ${fq(rt.si[rt.si.length - 1], 3)}`, '', 'End of the simulation'], ['Mean wall deposit', rt.depositMean, 'g/m²', flush ? `Initial inventory ${fq(v.rtM0, 3)} g/m²` : 'Formed during the simulation'], ['Largest wall flux', Math.max(...day.map(Math.abs)), 'g/m²·d', 'Reactive mineral-surface flux'],
      ['Residence time', L / u, 's', `${fq(L, 3)} m at ${fq(u, 3)} m/s`], ['Dispersion coefficient used', rt.Deff, 'm²/s', `Numerical part ${fq(rt.Dnum, 2)} m²/s is included, not added`], ['Grid and time steps', `${rt.x.length} cells · ${rt.nSteps} steps`, '', `Δt = ${fq(rt.dt, 3)} s`]],
      note: 'Advection–dispersion–reaction with the saturation index interpolated from speciation runs of three inlet/initial mixtures at six reaction-progress levels. Equal diffusivities are assumed for all species.' });
    BAL.push({ name: `Reactive transport: ${M.name.toLowerCase()} exchanged with the wall vs change in the water (mol per m² of flow section)`, in: rt.balance.in + 1, out: rt.balance.out + 1 });
    K.push({ label: 'Reactive transport: mean wall deposit', value: rt.depositMean, unit: 'g/m²', help: `${M.name} on the channel wall after ${fq(v.rtPV, 3)} channel volumes` });
    out.rtDepositMean = rt.depositMean; out.rtOutletSI = rt.si[rt.si.length - 1];
  }
  // 4 — two-dimensional flow and concentration field of the tail channel
  if (v.cfdOn !== false) {
    const io = solutionToIons(feed), Rr = clamp(a.R, 0, 0.98), vw = Math.max(v.cfdFlux ?? 15, 0.1) / 3.6e6, H = h / 2, rCh = Math.min(0.6, (vw * L) / (u * H)), cfIn = Math.max(1, (1 - rCh) / (1 - Rr)), vwUse = (rCh * u * H) / L;
    const Dm = diffusivityNaCl(v.T, Math.min(io.salinity * cfIn, 200)) * Math.max(v.cfdMix ?? 1, 0.1), visc = (cc) => viscosity(v.T, Math.min(io.salinity * cfIn * cc, 300));
    const f = channelCFD({ L, H, u0: u, vw: vwUse, D: Dm, rej: a.rej, nx: 60, ny: v.cfdNy ?? 30, visc }), nxp = f.x.length, cwMax = Math.max(...f.cw);
    const cfs = logspace(Math.max(cfIn, 0.2), Math.max(cfIn * cwMax, cfIn * 1.05), 8), sis = kinId ? cfs.map((cf) => saturationIndex(concentrateSolution(feed, cf, a.copt).eq, kinId, P, dk[kinId] || 0)) : cfs.map(() => 0), lc = cfs.map(Math.log);
    const siW = f.cw.map((cc) => interp1(lc, sis, Math.log(Math.max(cfIn * cc, 1e-9)))), beta = f.cw.map((cc, i) => cc / f.cb[i]), bMax = Math.max(...beta), yu = linspace(0, H, 24);
    const dep = kinId ? siW.map((s) => { const k = s > 0 ? nucleationKinetics(kinId, s, v.T, v) : null; return k ? k.flux * 24 : 0; }) : siW.map(() => 0);
    PL.push({ type: 'field', title: 'Channel flow model: concentration relative to the channel inlet', xlabel: 'Distance along the channel (m)', ylabel: 'Distance from the membrane (µm)', zlabel: 'c / c inlet', zunit: '–', x: f.x, y: yu.map((y) => y * 1e6), z: yu.map((y) => f.field.map((col) => interp1(f.y, col, y))), cmap: 'salinity', contours: 8, note: 'Membrane at the bottom, channel mid-plane at the top. The thin layer of raised concentration at the membrane is the polarisation layer that the film-theory factor β summarises.' });
    PL.push({ type: 'line', title: 'Channel flow model: polarisation and wall saturation along the channel', xlabel: 'Distance along the channel (m)', ylabel: 'β (–) · saturation index', series: [{ name: 'Local polarisation factor c wall / c bulk', x: f.x, y: beta }, ...(kinId ? [{ name: `${MINERALS[kinId].name} saturation index at the wall`, x: f.x, y: siW }] : [])], hlines: [{ y: a.beta, label: 'β entered' }] });
    TB.push({ title: 'Channel flow model (two-dimensional, boundary-layer form)', columns: ['Quantity', 'Value', 'Unit', 'Note'], rows: [
      ['Channel recovery', 100 * f.recovery, '%', `Flux ${fq(vwUse * 3.6e6, 3)} L/m²·h over ${fq(L, 3)} m${cfIn > 1 ? '; the outlet is at the design recovery' : ''}`], ['Bulk concentration factor inlet → outlet (vs feed)', `${fq(cfIn, 4)} → ${fq(cfIn * f.cb[nxp - 1], 4)}`, '×', 'Mixing-cup average'],
      ['Largest local polarisation factor', bMax, '–', `Entered β = ${fq(a.beta, 3)}`], ['Wall concentration factor at the outlet (vs feed)', cfIn * f.cw[nxp - 1], '×', 'What the scaling minerals see'], ['Pressure drop', f.dp / 1e5, 'bar', 'From the numerically solved momentum equation with concentration-dependent viscosity'],
      ['Wall shear stress inlet → outlet', `${fq(f.tauW[0], 3)} → ${fq(f.tauW[nxp - 1], 3)}`, 'Pa', 'Falls as water is removed'], ['Peak / mean velocity at the outlet', f.umax[nxp - 1] / Math.max(f.qOut / H, 1e-30), '–', '1.5 for constant viscosity'],
      ...(kinId ? [[`${MINERALS[kinId].name} saturation index at the wall, outlet`, siW[nxp - 1], '', 'From the speciation model at the local wall concentration'], ['Largest potential deposit flux', Math.max(...dep), 'g/m²·d', 'Growth law at the local wall supersaturation']] : []), ['Grid', `${nxp} stations × ${f.y.length} cells`, '', 'Cells refined towards the membrane']],
      note: `Open slit without feed spacer (laminar): an upper bound of the polarisation in a spacer-filled element — switch on the spacer-filled section (Navier–Stokes solver) for the effect of the filaments. Solute diffusivity ${fq(Dm, 3)} m²/s${(v.cfdMix ?? 1) !== 1 ? ` (including the mixing factor ${v.cfdMix})` : ''}.` });
    BAL.push({ name: 'Channel flow model: salt entering vs leaving (relative)', in: 1, out: f.balance.out / f.balance.in });
    K.push({ label: 'Channel model: largest local β', value: bMax, unit: '–', status: bMax > a.beta * 1.15 ? 'warn' : 'ok', help: 'Wall-to-bulk concentration ratio from the two-dimensional flow and concentration field of an open channel' });
    if (bMax > a.beta * 1.15) W.push({ level: 'info', msg: `The open-channel flow model gives a local polarisation factor up to ${fq(bMax, 3)}, above the entered β = ${fq(a.beta, 3)} — a feed spacer lowers it; check β with suite 4 or the RO suite.` });
    out.cfdBetaMax = bMax; out.cfdPressureDropBar = f.dp / 1e5;
    march = { f, cfIn, vw: vwUse, u, io };
  }
  // 5 — sorption on ferric hydroxide and sodium-cycle softening (pretreatment)
  if (a.scm) {
    const s = a.scm, pHs = linspace(4, 11, 15), edge = pHs.map((x) => surfaceComplexation(equilibrate(s.before, { pH: x }).eq, v.feDose, { kgwPerL: s.before.kgwPerL || 1 }).removal);
    PL.push({ type: 'line', title: 'Sorption edge on ferric hydroxide (surface-complexation model)', xlabel: 'pH', ylabel: 'Removed from solution (%)', ymin: 0, ymax: 100, series: [{ name: 'Silica', x: pHs, y: edge.map((e) => 100 * e.Si) }, { name: 'Boron', x: pHs, y: edge.map((e) => 100 * e.B) }], vlines: [{ x: s.before.pH, label: 'feed pH' }] });
    TB.push({ title: 'Surface complexation on hydrous ferric oxide', columns: ['Surface species', 'Concentration (µmol/kg water)', 'Share of sites (%)'], rows: s.species.map((q) => [q.name, q.conc * 1e6, 100 * q.frac]),
      note: `${fq(v.feDose, 3)} mg/L Fe gives ${fq(s.Stot * 1e6, 3)} µmol/kg of sites on ${fq(s.areaKg, 3)} m² per kg water. Surface potential ${fq(s.psi * 1000, 3)} mV, surface charge ${fq(s.sigma * 1000, 3)} mC/m². Silica removal ${fq(100 * s.removal.Si, 3)} %, boron removal ${fq(100 * s.removal.B, 3)} %; the water sent to the concentration step is corrected for both. ${SCM_NOTE}` });
    K.push({ label: 'Silica removed by ferric hydroxide', value: 100 * s.removal.Si, unit: '%', help: 'Surface-complexation model' }, { label: 'Boron removed by ferric hydroxide', value: 100 * s.removal.B, unit: '%' });
    out.silicaRemoval = s.removal.Si; out.boronRemoval = s.removal.B;
  }
  if (a.ix) {
    const x = a.ix, pc = (b) => 100 * b;
    PL.push({ type: 'line', title: 'Softener breakthrough (equilibrium-stage column, Gaines–Thomas exchange)', xlabel: 'Bed volumes treated (–)', ylabel: 'Effluent / feed concentration (–)', series: [{ name: 'Calcium', x: x.bv, y: x.yCa }, { name: 'Magnesium', x: x.bv, y: x.yMg }], vlines: [{ x: x.bvBreak, label: 'breakthrough' }, { x: x.bvIdeal, label: 'stoichiometric' }], note: 'Magnesium is held less strongly and is displaced by calcium, so it breaks through first and can overshoot its feed concentration.' });
    TB.push({ title: 'Ion exchange: sodium-cycle softener', columns: ['Quantity', 'Na⁺ (+K⁺)', 'Ca²⁺', 'Mg²⁺', 'Unit'], rows: [
      ['Feed', x.c0.Na * 1000, x.c0.Ca * 1000, x.c0.Mg * 1000, 'meq/L'], ['Resin in equilibrium with the feed (exhausted)', pc(x.exhausted.Na), pc(x.exhausted.Ca), pc(x.exhausted.Mg), '% of capacity'], ['Resin after regeneration (bed average)', pc(x.regenerated.Na), pc(x.regenerated.Ca), pc(x.regenerated.Mg), '% of capacity'], ['Resin after regeneration (service-outlet end)', pc(x.regOutlet.Na), pc(x.regOutlet.Ca), pc(x.regOutlet.Mg), '% of capacity'],
      ['Softened water (mean to breakthrough)', x.soft.Na * 1000, x.soft.Ca * 1000, x.soft.Mg * 1000, 'meq/L'], ['Selectivity in the feed, concentration basis', 1, x.KcFeed.Ca, x.KcFeed.Mg, 'L/eq'], ['Selectivity in the regenerant', 1, x.KcRegen.Ca, x.KcRegen.Mg, 'L/eq']],
      note: `Working capacity ${fq(x.working, 3)} eq per litre of resin of ${fq(v.ixCap, 3)} eq/L total; ${x.broke ? `breakthrough (${v.ixLeak} % hardness leakage) after ${fq(x.bvBreak, 3)} bed volumes` : `no breakthrough within ${fq(x.bvBreak, 3)} bed volumes`} (stoichiometric ${fq(x.bvIdeal, 3)}); regenerant ${fq(x.regenLitres, 3)} L of ${v.ixRegenPct} % NaCl per litre of resin, salt efficiency ${fq(100 * x.saltEff, 3)} %. Activity coefficients come from the selected activity model. The softened water is what the concentration step receives.` });
    BAL.push({ name: 'Softener: hardness fed vs effluent + resin loading (eq per litre of resin)', in: x.balance.in, out: x.balance.out });
    K.push({ label: 'Softener throughput to breakthrough', value: x.bvBreak, unit: 'bed volumes', status: x.working < 0.15 * v.ixCap ? 'warn' : 'ok', help: 'Gaines–Thomas exchange in an equilibrium-stage column' }, { label: 'Softener working capacity', value: x.working, unit: 'eq/L resin' });
    if (x.working < 0.15 * v.ixCap) W.push({ level: 'warn', msg: `The sodium-cycle softener has a working capacity of only ${fq(x.working, 3)} eq/L at this salinity: sodium in the feed competes for the resin, so softening by ion exchange is impractical here.` });
    out.softenerBedVolumes = x.bvBreak; out.softenerCapacity = x.working;
  }
  // 6 — surrogate of the saturation indices trained on the speciation engine
  if (v.mlOn) {
    const ids = ['calcite', 'gypsum', 'barite', 'silica'].filter((id) => a.set.includes(id) && present(feed.eq, id));
    if (ids.length) {
      const nT = clamp(Math.round(v.mlN ?? 48), 16, 300), s = trainSurrogate(a, ids, { nTrain: nT, nTest: Math.max(8, Math.round(nT / 3)), seed: Math.round(v.mlSeed ?? 11) }), pH0 = clamp(a.raw.pH, 5.5, 9), T0 = clamp(v.T, 5, 60), r0 = clamp(a.R, 0, a.Rmax);
      const at = s.predict(r0, pH0, T0), dpH = s.predict(r0, Math.min(9, pH0 + 0.1), T0).map((x, i) => (x - s.predict(r0, Math.max(5.5, pH0 - 0.1), T0)[i]) / (Math.min(9, pH0 + 0.1) - Math.max(5.5, pH0 - 0.1))), dT = s.predict(r0, pH0, Math.min(60, T0 + 2)).map((x, i) => (x - s.predict(r0, pH0, Math.max(5, T0 - 2))[i]) / (Math.min(60, T0 + 2) - Math.max(5, T0 - 2)));
      const exact = s.engine(r0, pH0, T0), lo = Math.min(...s.meas.flat(), ...s.pred.flat()), hi = Math.max(...s.meas.flat(), ...s.pred.flat());
      PL.push({ type: 'line', title: 'Surrogate model of the wall saturation index: parity on held-out points', xlabel: 'Speciation engine (SI)', ylabel: 'Surrogate (SI)', series: [...ids.map((id, o) => ({ name: MINERALS[id].name, x: s.meas.map((r) => r[o]), y: s.pred.map((r) => r[o]), mode: 'points' })), { name: '1 : 1', x: [lo, hi], y: [lo, hi], dash: true }], note: `${s.nTest} test points that were not used for training.` });
      TB.push({ title: 'Surrogate (kernel regression) of the wall saturation index', columns: ['Mineral', 'Test RMSE (SI)', 'Test R²', 'Largest test error (SI)', 'Surrogate at design', 'Engine at design', '∂SI/∂pH', '∂SI/∂T (1/K)'], rows: s.stats.map((q, o) => [MINERALS[q.id].name, q.rmse, q.r2, q.maxErr, at[o], exact[o], dpH[o], dT[o]]),
        note: `Gaussian-kernel ridge regression trained on ${s.nTrain} Latin-hypercube samples of the speciation engine over recovery 0–${fq(100 * a.Rmax, 3)} %, feed pH 5.5–9 and 5–60 °C, tested on ${s.nTest} further samples; kernel length ${s.model.ell} (standardised units) chosen by leave-one-out cross-validation (LOO error ${fq(s.model.loo, 2)} SI). The design column is evaluated at the feed pH before dosing${a.dose.reagent ? ' (the main results include the dose)' : ''}. Sensitivities are finite differences of the surrogate.` });
      K.push({ label: 'Surrogate test error', value: s.rmse, unit: 'SI', status: s.rmse < 0.1 ? 'ok' : 'warn', help: 'Root-mean-square error of the trained surrogate on held-out speciation runs' });
      if (s.rmse > 0.1) W.push({ level: 'info', msg: `The surrogate of the saturation index has a test error of ${fq(s.rmse, 2)} SI units — raise the number of training points before using its sensitivities.` });
      out.surrogateRMSE = s.rmse; sur = s;
    }
  }
  return { K, PL, TB, W, BAL, out, march, sur, kinId };
}

/** Value below which the share q (0–1) of the entries lies. */
const pctl = (arr, q) => { const b = [...arr].sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.max(0, Math.round(q * (b.length - 1))))] : 0; };
/**
 * Scaling in a spacer-filled section at the concentrate end of the channel: Navier–Stokes + salt transport with
 * permeating membranes (suite 4 solver) → local wall concentration → speciation engine → saturation-index map,
 * deposition flux and hot spots; compared with the open-slit boundary-layer march and with film theory.
 */
export async function spacerSection(v, a, m, kinId, ctx) {
  const { feed, P, dk } = a, h = Math.max(v.rtH ?? 0.71, 0.05) * 1e-3, arr = v.cfdArr || 'zigzag', lm = clamp(v.cfdLm ?? 3, 0.5, 20) * 1e-3, nFil = clamp(Math.round(v.cfdNFil ?? 6), 2, 20);
  const cfSec = m.cfIn * m.f.cb[m.f.cb.length - 1], uSec = Math.max(m.u * (1 - m.f.recovery), 1e-4), Ssec = Math.min(m.io.salinity * cfSec, 250), T = v.T;
  const rho = density(T, Ssec), mu = viscosity(T, Ssec), Dm = diffusivityNaCl(T, Math.min(Ssec, 200)), piOf = (c) => osmoticPressure(T, clamp(Ssec * c, 0, 260)), dPa = Math.max(v.P ?? 0, 0) * 1e5;
  const osm = v.cfdOsm !== false && dPa - piOf(1) * a.rej > 0.1 * dPa;
  const f = await spacerChannelCFD({ h, u0: uSec, vw: m.vw, D: Dm, rej: a.rej, rho, mu, arr, lm, df: clamp((v.cfdDf ?? 50) / 100, 0.1, 0.85) * h, nFil, nxFil: v.cfdNxFil ?? 24, ny: v.cfdNyFull ?? 32, dP: dPa, pi: osm ? piOf : null, tol: 1e-3, maxIter: 400, solver: { alphaU: 0.85 } }, ctx); // settings tuned against a fully converged run: wall concentrations within 1e-4
  // speciation engine on the local wall composition: bulk analysis × local concentration factor (tabulated on 9 factors)
  const set = a.set.filter((id) => present(feed.eq, id)), cMax = Math.max(1.02, ...f.cwB, ...f.cwT, ...f.field.flat()), cfs = logspace(cfSec * 0.98, cfSec * cMax * 1.02, 9), lc = cfs.map(Math.log);
  const tab = cfs.map((cf) => saturation(concentrateSolution(feed, cf, a.copt).eq, P, dk, set)), id0 = kinId && set.includes(kinId) ? kinId : set.reduce((b, id) => (b == null || tab[8][id] / MINERALS[id]._nu > tab[8][b] / MINERALS[b]._nu ? id : b), null);
  const siOf = (id, c) => interp1(lc, tab.map((q) => q[id]), Math.log(Math.max(cfSec * c, 1e-9))), depOf = (si) => { const k = si > 0 ? nucleationKinetics(id0, si, T, v) : null; return k ? k.flux * 24 : 0; };
  const i0 = Math.round(f.nx / nFil), side = (cw, blk, J, tau) => { // statistics over the open membrane downstream of the entrance spacing
    const xs = [], c = [], si = [], dep = [], jj = [], tt = [];
    for (let i = 0; i < f.nx; i++) if (!blk[i]) { xs.push(f.x[i]); c.push(cw[i]); const s0 = siOf(id0, cw[i]); si.push(s0); dep.push(depOf(s0)); jj.push(J[i]); tt.push(Math.abs(tau[i])); }
    const k0 = xs.findIndex((x) => x >= f.x[Math.min(i0, f.nx - 1)]), st = k0 < 0 ? 0 : k0, cs = c.slice(st), iPk = st + cs.indexOf(Math.max(...cs));
    return { x: xs, c, si, dep, J: jj, tau: tt, cMean: sum(cs) / cs.length, c95: pctl(cs, 0.95), cPeak: c[iPk], xPeak: xs[iPk], siPeak: si[iPk], siMean: sum(si.slice(st)) / cs.length, depMean: sum(dep.slice(st)) / cs.length, depPeak: Math.max(...dep.slice(st)), tauPeak: tt[iPk], tauMean: sum(tt.slice(st)) / cs.length, share: cs.filter((x) => x > 1.1 * (sum(cs) / cs.length)).length / cs.length };
  };
  const B = side(f.cwB, f.blockB, f.JB, f.tauB), Tp = side(f.cwT, f.blockT, f.JT, f.tauT), cbMean = sum(f.cb.slice(i0)) / (f.nx - i0), hot = B.cPeak >= Tp.cPeak ? B : Tp;
  // references: open-slit boundary-layer march over the same section with the same mean flux, and film theory with the spacer Sherwood correlation
  const mar = channelCFD({ L: f.L, H: h / 2, u0: uSec, vw: f.Jmean, D: Dm, rej: a.rej, nx: 120, ny: v.cfdNy ?? 30, visc: () => mu }), marSI = mar.cw.map((c) => siOf(id0, c)), ms = mar.x.map((x, i) => i).filter((i) => mar.x[i] >= f.x[Math.min(i0, f.nx - 1)]);
  const marMean = sum(ms.map((i) => mar.cw[i])) / ms.length, marPeak = Math.max(...mar.cw);
  const c1 = channel1D({ T, c0: 1, propMode: 'custom', rho, mu: mu * 1000, Dsalt: Dm * 1e9, piCoef: 0, H: h * 1000, Uin: uSec, geom: 'spacer', arr, nFil, lm: lm * 1000, L: f.L * 1000, A: 0, B: 0, dPtm: 0 });
  const e = Math.exp(Math.min(8, f.Jmean / c1.k)), film = e / (1 + (1 - a.rej) * (e - 1));
  const mineral = (c) => Object.fromEntries(set.map((id) => [id, siOf(id, c)]));
  return { f, id0, set, cfSec, uSec, Ssec, rho, mu, Dm, osm, B, T: Tp, hot, cbMean, mar, marSI, marMean, marPeak, film, c1, siOf, depOf, siBulk: mineral(cbMean), siWallMean: mineral(0.5 * (B.cMean + Tp.cMean)), siHot95: mineral(Math.max(B.c95, Tp.c95)), siPeak: mineral(hot.cPeak), siFilm: mineral(cbMean * film), siMarch: mineral(marMean), Re: (rho * uSec * 2 * h) / mu };
}
async function spacerScaling(v, a, X, ctx) {
  let q;
  try { q = await spacerSection(v, a, X.march, X.kinId, ctx); } catch (err) { X.W.push({ level: 'warn', msg: `The Navier–Stokes model of the spacer section could not be solved (${err.message}); the open-channel march stands.` }); return; }
  const { f, id0, B, T: Tp, hot } = q, M = id0 ? MINERALS[id0] : null, nm = M ? M.name : 'mineral', mm = (xs) => xs.map((x) => x * 1000), arrName = { zigzag: 'zigzag filaments', cavity: 'filaments on one membrane', submerged: 'mid-channel filaments', none: 'no filaments' }[v.cfdArr || 'zigzag'];
  const siF = f.field.map((row, j) => row.map((c, i) => (f.mask[j][i] ? 0 : clamp(q.siOf(id0, c), -12, 12))));
  X.PL.push({ type: 'field', title: `Spacer-filled section: ${nm.toLowerCase()} saturation index (Navier–Stokes model)`, xlabel: 'Distance along the section (mm)', ylabel: 'Height above the lower membrane (mm)', zlabel: 'SI', zunit: '', x: mm(f.x), y: mm(f.y), z: siF, mask: f.mask, cmap: 'turbo', contours: 8, shapes: f.shapes.map((sh) => ({ ...sh, x: mm(sh.x), y: mm(sh.y) })), markers: [{ x: hot.xPeak * 1000, y: hot === B ? 0 : f.h * 1000, label: 'hot spot' }],
    note: `Membranes at the bottom and the top, flow from left to right, ${arrName}. The salt rejected by the permeating membranes accumulates in the slow fluid next to and behind the filaments; the saturation index follows from the speciation engine at the local concentration.` });
  X.PL.push({ type: 'line', title: `Spacer-filled section: ${nm.toLowerCase()} saturation index at the membranes`, xlabel: 'Distance along the section (mm)', ylabel: 'Saturation index', series: [{ name: 'Lower membrane (Navier–Stokes, spacer)', x: mm(B.x), y: B.si }, { name: 'Upper membrane (Navier–Stokes, spacer)', x: mm(Tp.x), y: Tp.si }, { name: 'Open slit (boundary-layer march)', x: mm(q.mar.x), y: q.marSI, dash: true }, { name: 'Bulk (mixing-cup)', x: mm(f.x), y: f.cb.map((c) => q.siOf(id0, c)), dash: true }], hlines: [{ y: q.siFilm[id0], label: 'film theory' }, { y: 0, label: 'saturation' }], note: 'Gaps in the membrane curves are the contact lines of filaments, where the membrane is covered.' });
  X.PL.push({ type: 'line', title: `Spacer-filled section: potential ${nm.toLowerCase()} deposition flux and permeate flux`, xlabel: 'Distance along the section (mm)', ylabel: 'Deposition (g/m²·d) · flux (L/m²·h)', series: [{ name: 'Deposition, lower membrane', x: mm(B.x), y: B.dep }, { name: 'Deposition, upper membrane', x: mm(Tp.x), y: Tp.dep }, { name: 'Deposition, open slit', x: mm(q.mar.x), y: q.marSI.map(q.depOf), dash: true }, { name: 'Permeate flux, lower membrane (L/m²·h)', x: mm(B.x), y: B.J.map((j) => j * 3.6e6), dash: true }] });
  const cf = q.cfSec, st = (b) => [b.cMean, b.c95, b.cPeak];
  X.TB.push({ title: 'Spacer-filled channel section (Navier–Stokes, finite-volume solver of suite 4)', columns: ['Quantity', 'Navier–Stokes, lower membrane', 'Navier–Stokes, upper membrane', 'Open-slit march', 'Film theory', 'Unit'], rows: [
    ['Polarisation factor c wall / c section inlet: mean', B.cMean, Tp.cMean, q.marMean, q.cbMean * q.film, '–'], ['Polarisation factor: 95th percentile', B.c95, Tp.c95, pctl(q.mar.cw, 0.95), null, '–'], ['Polarisation factor: peak (hot spot)', B.cPeak, Tp.cPeak, q.marPeak, null, '–'],
    ['Wall concentration factor versus the feed: mean', cf * B.cMean, cf * Tp.cMean, cf * q.marMean, cf * q.cbMean * q.film, '×'], [`${nm} saturation index: mean`, B.siMean, Tp.siMean, q.siMarch[id0], q.siFilm[id0], ''], [`${nm} saturation index: hot spot`, B.siPeak, Tp.siPeak, q.siOf(id0, q.marPeak), null, ''],
    ['Hot-spot position', B.xPeak * 1000, Tp.xPeak * 1000, f.L * 1000, null, 'mm'], ['Wall shear stress at the hot spot / mean', `${fq(B.tauPeak, 3)} / ${fq(B.tauMean, 3)}`, `${fq(Tp.tauPeak, 3)} / ${fq(Tp.tauMean, 3)}`, null, null, 'Pa'], ['Membrane more than 10 % above the mean wall concentration', 100 * B.share, 100 * Tp.share, null, null, '% of open area'],
    [`Potential ${nm.toLowerCase()} deposition flux: mean`, B.depMean, Tp.depMean, sum(q.marSI.map(q.depOf)) / q.marSI.length, q.depOf(q.siFilm[id0]), 'g/m²·d'], [`Potential ${nm.toLowerCase()} deposition flux: peak`, B.depPeak, Tp.depPeak, q.depOf(q.siOf(id0, q.marPeak)), null, 'g/m²·d'],
    ['Mean permeate flux', sum(B.J) / B.J.length * 3.6e6, sum(Tp.J) / Tp.J.length * 3.6e6, f.Jmean * 3.6e6, f.Jmean * 3.6e6, 'L/m²·h']],
    note: `Section of ${fq(f.L * 1000, 3)} mm (${v.cfdNFil ?? 6} filament spacings of ${fq((v.cfdLm ?? 3), 3)} mm) at the concentrate end: inlet at ${fq(cf, 4)} times the feed concentration (${fq(q.Ssec, 3)} g/kg), ${fq(q.uSec, 3)} m/s, channel Reynolds number ${fq(q.Re, 3)}. Grid ${f.nx} × ${f.ny} cells, ${f.converged ? 'converged' : 'NOT converged'} in ${f.iters} iterations, salt balance error ${fq(Math.abs(f.balance.out / f.balance.in - 1), 2)}. ${q.osm ? `Solution–diffusion walls with A = ${fq(f.A * 3.6e11, 3)} L/m²·h·bar at ${fq(v.P, 3)} bar, so the flux falls where the wall concentration rises.` : 'Uniform permeate flux (the applied pressure does not exceed the osmotic pressure by enough for the flux law).'} Constant density and viscosity; molecular diffusivity ${fq(q.Dm, 3)} m²/s without a mixing factor. Statistics exclude the first filament spacing (entrance). The march and film-theory columns use the mean flux of the Navier–Stokes solution; film theory uses the spacer Sherwood correlation of suite 4 (Sh = ${fq(q.c1.Sh, 3)}). The peak value at a filament contact line depends on the grid — judge hot spots by the 95th percentile and refine the grid on the Mesh tab.` });
  X.TB.push({ title: 'Saturation indices in the spacer-filled section', columns: ['Mineral', 'Bulk', 'Wall, mean (Navier–Stokes)', 'Wall, 95th percentile', 'Wall, hot spot', 'Wall, open-slit march (mean)', 'Wall, film theory'], rows: q.set.map((id) => [MINERALS[id].name, q.siBulk[id], q.siWallMean[id], q.siHot95[id], q.siPeak[id], q.siMarch[id], q.siFilm[id]]), note: 'Speciation engine evaluated on the bulk analysis scaled by the local concentration factor (nine tabulated factors, interpolated in ln CF).' });
  X.BAL.push({ name: 'Spacer section (Navier–Stokes): salt entering vs leaving (relative)', in: 1, out: f.balance.out / f.balance.in });
  const bMean = 0.5 * (B.cMean + Tp.cMean) / q.cbMean, b95 = Math.max(B.c95, Tp.c95) / q.cbMean;
  X.K.push({ label: 'Spacer section: mean polarisation factor', value: bMean, unit: '–', help: `Navier–Stokes solution with ${arrName}; open-slit march ${fq(q.marMean / q.cbMean, 3)}, film theory ${fq(q.film, 3)}` }, { label: `Spacer section: hot-spot ${nm.toLowerCase()} SI`, value: Math.max(q.siHot95[id0], -99), unit: '', status: q.siHot95[id0] > Math.max(a.lim[id0] ?? 0, 0) ? 'warn' : 'ok', help: '95th percentile of the wall saturation index behind the filaments' });
  if (!f.converged) X.W.push({ level: 'warn', msg: 'The Navier–Stokes solution of the spacer section did not reach its tolerance — the flow behind the filaments is probably unsteady; treat the hot-spot values as indicative.' });
  if (q.siHot95[id0] > 0 && q.siHot95[id0] - q.siWallMean[id0] > 0.05) X.W.push({ level: 'info', msg: `Behind the spacer filaments the wall ${nm.toLowerCase()} saturation index reaches ${fq(q.siHot95[id0], 3)} (95th percentile; mean ${fq(q.siWallMean[id0], 3)}, film theory ${fq(q.siFilm[id0], 3)}): scale starts at these stagnant spots.` });
  Object.assign(X.out, { spacerBetaMean: bMean, spacerBeta95: b95, spacerBetaPeak: hot.cPeak / q.cbMean, spacerSIMean: q.siWallMean[id0], spacerSIHot: q.siHot95[id0], spacerDepositPeak: Math.max(B.depPeak, Tp.depPeak), spacerMineral: id0 });
}

/**
 * Neural network and Gaussian process (learners of the optimisation suite) trained on the design points of the
 * kernel-regression surrogate. All three are fitted on the first 80 % of the training points, ranked on the remaining
 * 20 % (which the network also uses for early stopping) and reported on the held-out test runs.
 */
export async function trainLearners(s, { seed = 11, hidden = [12], epochs = 1500, tick = null } = {}) {
  const { nnTrain, gpFit } = await import('./s11_opt.js');
  const nT = s.nTrain, nFit = Math.max(8, Math.round(0.8 * nT)), Xf = s.X.slice(0, nFit), Xv = s.X.slice(nFit, nT), Xt = s.X.slice(nT), col = (Y, o) => Y.map((r) => r[o]);
  const krr = kernelRidge(Xf, s.Y.slice(0, nFit)), cap = Math.min(nFit, 60), per = { krr: s.ids.map((_, o) => (x) => krr.predict(x)[o]), gp: [], nn: [] }, info = { gp: [], nn: [] };
  for (let o = 0; o < s.ids.length; o++) {
    // hyper-parameters by maximum marginal likelihood on at most 60 points (the cost grows with n³ per likelihood evaluation), then conditioned on every fit point
    const yf = col(s.Y.slice(0, nFit), o), g0 = gpFit(Xf.slice(0, cap), yf.slice(0, cap), { maxIter: 160 }), gp = cap < nFit ? gpFit(Xf, yf, { theta: g0.theta }) : g0;
    per.gp.push((x) => gp.predict(x).mean); info.gp.push(gp);
    if (tick) await tick();
    const nn = nnTrain(Xf, yf, { hidden, epochs, lr: 0.02, seed: seed + o, Xval: Xv, yval: col(s.Y.slice(nFit, nT), o), patience: 120, l2: 1e-5, batch: 8 });
    per.nn.push((x) => nn.predict(x)); info.nn.push(nn);
    if (tick) await tick();
  }
  const score = (f, X, Y) => { const p = X.map(f), m = Y, mm = sum(m) / m.length, sse = sum(m.map((y, i) => (y - p[i]) ** 2)), sst = sum(m.map((y) => (y - mm) ** 2)); return { rmse: Math.sqrt(sse / m.length), r2: sst > 0 ? 1 - sse / sst : 1, maxErr: Math.max(...m.map((y, i) => Math.abs(y - p[i]))), pred: p }; };
  const names = { krr: 'Kernel ridge regression', gp: 'Gaussian process', nn: 'Neural network' }, L = {};
  for (const k of Object.keys(names)) {
    const val = s.ids.map((_, o) => score(per[k][o], Xv, col(s.Y.slice(nFit, nT), o))), test = s.ids.map((_, o) => score(per[k][o], Xt, col(s.Y.slice(nT), o)));
    L[k] = { key: k, name: names[k], val, test, valRmse: Math.sqrt(sum(val.map((q) => q.rmse ** 2)) / val.length), testRmse: Math.sqrt(sum(test.map((q) => q.rmse ** 2)) / test.length), predict: (r, pH, T) => { const x = s.feat(r, pH, T); return per[k].map((f) => f(x)); } };
  }
  const best = Object.values(L).reduce((b, q) => (q.valRmse < b.valRmse ? q : b));
  return { learners: L, best, nFit, nVal: nT - nFit, nTest: Xt.length, gpCap: cap, info, seed, hidden };
}
async function compareLearners(v, a, X, ctx) {
  const s = X.sur, seed = Math.round(v.mlSeed ?? 11);
  let c;
  try { c = await trainLearners(s, { seed, tick: ctx?.tick }); } catch (err) { X.W.push({ level: 'warn', msg: `The neural network and Gaussian process could not be trained (${err.message}); the kernel regression stands.` }); return; }
  const meas = s.Y.slice(s.nTrain).flat(), lo = Math.min(...meas), hi = Math.max(...meas), Ls = Object.values(c.learners), pH0 = clamp(a.raw.pH, 5.5, 9), T0 = clamp(v.T, 5, 60), r0 = clamp(a.R, 0, a.Rmax), exact = s.engine(r0, pH0, T0);
  X.PL.push({ type: 'line', title: 'Surrogate learners: parity on held-out engine runs', xlabel: 'Speciation engine (SI)', ylabel: 'Surrogate (SI)', series: [...Ls.map((q) => ({ name: `${q.name}${q === c.best ? ' (selected)' : ''}`, x: meas, y: s.Y.slice(s.nTrain).map((_, i) => s.ids.map((__, o) => q.test[o].pred[i])).flat(), mode: 'points' })), { name: '1 : 1', x: [lo, hi], y: [lo, hi], dash: true }], note: `${c.nTest} test runs × ${s.ids.length} minerals that no learner has seen.` });
  X.TB.push({ title: 'Surrogate learners compared on held-out engine runs', columns: ['Learner', 'Mineral', 'Validation RMSE (SI)', 'Test RMSE (SI)', 'Test R²', 'Largest test error (SI)', 'At design', 'Engine at design', 'Selected'],
    rows: [...Ls.flatMap((q) => { const at = q.predict(r0, pH0, T0); return s.ids.map((id, o) => [q.name, MINERALS[id].name, q.val[o].rmse, q.test[o].rmse, q.test[o].r2, q.test[o].maxErr, at[o], exact[o], q === c.best ? 'yes' : '']); }), ...Ls.map((q) => [q.name, 'All minerals', q.valRmse, q.testRmse, null, Math.max(...q.test.map((t) => t.maxErr)), null, null, q === c.best ? 'yes' : ''])],
    note: `Same Latin-hypercube design for all learners (seed ${seed}): ${c.nFit} points to fit, ${c.nVal} to rank the learners (the network also stops early on them) and ${c.nTest} untouched test runs. Gaussian process: anisotropic squared-exponential kernel, hyper-parameters by maximum marginal likelihood${c.gpCap < c.nFit ? ` on the first ${c.gpCap} points, then conditioned on all ${c.nFit}` : ''}; length scales for ${MINERALS[s.ids[0]].name.toLowerCase()} (recovery, pH, temperature; standardised) ${c.info.gp[0].lengthScales.map((l) => fq(l, 3)).join(', ')}. Neural network: ${c.hidden.join(' + ')} tanh units per output (${c.info.nn[0].nWeights} weights), mini-batch Adam, best epoch ${c.info.nn.map((n) => n.bestEpoch).join('/')}. The learner with the lowest validation error is selected; its test error is therefore an unbiased estimate. The kernel regression of the table above is trained on all ${s.nTrain} points.` });
  X.K.push({ label: 'Best surrogate learner', value: c.best.name, help: 'Lowest error on the validation points' }, { label: 'Best learner: test error', value: c.best.testRmse, unit: 'SI', status: c.best.testRmse < 0.1 ? 'ok' : 'warn', help: 'Root-mean-square error on engine runs not used for fitting or selection' });
  Object.assign(X.out, { surrogateBest: c.best.key, surrogateBestRMSE: c.best.testRmse, surrogateRMSEs: Object.fromEntries(Ls.map((q) => [q.key, q.testRmse])) });
}

const D = () => Object.fromEntries(suite.inputs.flatMap((g) => g.fields).map((f) => [f.key, f.value]));

const suite = {
  id: 'chem', num: 2, title: 'Brine Chemistry, Precipitation & Scaling', short: 'Brine chemistry', icon: '⚗️',
  tagline: 'Speciation, activity models up to saturated brines, saturation indices, scaling limits, dosing and precipitation.',
  description: 'Solves the full aqueous speciation of a water analysis — carbonate, borate, silicate and sulphate acid–base systems, water dissociation and ion pairs — with activity coefficients from the Pitzer ion-interaction model or four Debye–Hückel-type models. The water is concentrated along the recovery path with carbonate re-equilibration; saturation indices of every relevant mineral, the maximum recovery before each one scales (with and without antiscalant), acid or caustic doses, equilibrium precipitation masses, nucleation induction times and corrosion indices follow from the same thermodynamic state. The equilibrium is cross-checked by an independent Gibbs-energy minimisation; a population balance follows nucleation and growth during the residence time; reactive transport and a two-dimensional channel flow model resolve where the wall becomes supersaturated; optional ferric-hydroxide sorption and ion-exchange softening act on the feed; a spacer-filled channel section can be resolved with the Navier–Stokes solver to locate scaling hot spots behind the filaments; and trained surrogates (kernel regression, Gaussian process, neural network) give fast sensitivities.',
  guide: [
    'Enter or pull the feed analysis, pH and temperature. A complete analysis matters: TDS alone cannot predict scaling.',
    'Set the recovery, the membrane-wall concentration factor β and, if used, the antiscalant and pH-adjustment strategy.',
    'On Model setup choose the activity model (Pitzer for anything above brackish salinity) and how dissolved CO₂ behaves.',
    'Optional, on Model setup: initial solids and nuclei, the reactive-transport case and its boundary conditions, the channel flow model, ferric-hydroxide sorption, ion-exchange softening and the surrogate model.',
    'Run. Read the saturation table first, then the recovery sweep: the first mineral to cross its limit sets the maximum recovery. The concentrate is offered to the ZLD and discharge suites.',
  ],
  implemented: ['mass-action', 'law of mass action', 'component mass-balance', 'charge-balance', 'alkalinity equation', 'acid–base equilibrium', 'water-dissociation', 'henry', 'mineral-solubility-product', 'ion-activity-product', 'saturation-index', 'chemical-potential equality', 'debye–hückel equation', 'extended debye–hückel', 'davies', 'specific-ion-interaction', 'pitzer', 'setschenow', 'ion-pairing', 'aqueous complexation equations', 'precipitation/dissolution rate', 'classical nucleation theory', 'crystal-growth', 'induction-time',
    'equilibrium–kinetic precipitation', 'pitzer–speciation', 'activity–nucleation–growth', 'scaling–surface-deposition', 'speciation–corrosion', 'evaporation–speciation–precipitation', 'membrane-concentration–mineral-equilibrium',
    'gibbs-energy-minimization', 'bromley', 'surface-complexation', 'ion-exchange', 'population-balance equation', 'equilibrium–reactive-transport', 'precipitation–population-balance', 'precipitation–cfd', 'thermodynamic–machine-learning',
    'initial mineral inventories', 'initial nuclei population', 'initial precipitate mass', 'prescribed species flux', 'zero-flux boundary', 'reactive mineral-surface flux', 'dissolution/precipitation surface condition', 'outlet convective',
    'initial ionic composition', 'alkalinity', 'temperature', 'pressure', 'dissolved gases', 'initial supersaturation', 'prescribed species concentration', 'equilibrium mineral boundary', 'gas–liquid equilibrium condition', 'inlet chemistry', 'fixed-temperature', 'prescribed-pressure',
    'complete ionic-speciation', 'electrolyte thermodynamics', 'activity and ionic-strength', 'acid-base equilibrium', 'ph and alkalinity', 'gas-liquid equilibrium', 'mineral saturation', 'precipitation and dissolution', 'scale identification', 'scale-quantity', 'crystallisation tendency', 'solubility modelling', 'temperature and pressure effects', 'chemical dosing', 'antiscalant assessment', 'corrosion tendency', 'brine mixing', 'reaction kinetics', 'high-salinity physical-property'],
  equationsNote: 'Pitzer parameters are the Harvie–Møller–Weare 25 °C set (Na–K–Mg–Ca–H–Cl–SO₄–HCO₃–CO₃–OH–CO₂, extended with Sr, Ba, NO₃, F and borate); away from 25 °C only the Debye–Hückel slope and the equilibrium constants change, so results are most reliable at 10–45 °C and indicative up to about 100 °C. The Debye–Hückel-type models with ion pairing are valid to an ionic strength of roughly 0.1 (limiting law 0.005, Davies 0.5, Truesdell–Jones about 1 mol/kg). pH is on the conventional single-ion activity scale without MacInnes scaling. The Bromley model treats all salts as fully dissociated strong electrolytes (reliable to about 6 mol/kg for chloride brines), the specific-ion-interaction model is accurate to about 3 mol/kg; both return the osmotic coefficient and water activity that satisfy the Gibbs–Duhem equation with their activity coefficients (the Debye–Hückel, extended Debye–Hückel and Truesdell–Jones models keep an approximate water activity). The solubility products and carbonate constants are those that belong to the selected model family (Harvie–Møller–Weare values with Pitzer and Bromley, WATEQ4F values with the ion-pair models); the Data provenance table lists the source of every constant set. Redox, phosphate speciation and solid solutions are not modelled; dolomite and magnesite are reported but never precipitated because they are kinetically inhibited. Induction times come from classical nucleation theory and are order-of-magnitude screening values; the antiscalant dose is a heuristic to be confirmed with the supplier. The Gibbs-energy minimisation treats the closed system with the same thermodynamic data as the mass-action solver, so it checks the numerical solution, not the data. The population balance uses size-independent growth without agglomeration or breakage. Reactive transport is one-dimensional with equal diffusivities for all species and one reacting mineral; its saturation index is interpolated between tabulated speciation runs. The default channel flow model is a two-dimensional laminar boundary-layer (parabolised) solution for an open slit with uniform flux. The optional spacer-filled section solves the two-dimensional steady Navier–Stokes and salt-transport equations with the finite-volume solver of suite 4 (transverse filaments, solution–diffusion membranes on both walls, constant fluid properties) over a few filament spacings at the concentrate end; it is two-dimensional and steady, so three-dimensional net geometry and vortex shedding are not represented, and the peak concentration at a filament contact line is grid dependent. Surface complexation covers silica and boron on hydrous ferric oxide at 25 °C constants (weak sites only); ion exchange covers Na–Ca–Mg on a strong-acid resin with ideal (equilibrium-stage) column behaviour and counter-current regeneration of a fully exhausted bed. The surrogates (kernel ridge regression, Gaussian process, feed-forward neural network) are trained and tested on the speciation engine itself and are only valid inside their training ranges.',

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
    { group: 'Equilibrium cross-check and initial solids', tab: 'setup', help: 'An independent Gibbs-energy minimisation of the bulk concentrate, and solids that are already present when the equilibrium is calculated.', fields: [
      { key: 'gemOn', label: 'Cross-check the equilibrium by Gibbs-energy minimisation', type: 'bool', value: true, help: 'Minimises the total Gibbs energy over all dissolved species and solid phases subject to the element balances (closed system) and compares pH and solid amounts with the mass-action solver.' },
      { key: 'invCalcite', label: 'Initial calcite inventory in the concentrate', unit: 'mg/L', value: 0, min: 0, max: 1e5, help: 'Suspended or deposited calcium carbonate present before equilibration. It dissolves if the water is undersaturated and adds to the solids otherwise.' },
      { key: 'invGypsum', label: 'Initial gypsum inventory in the concentrate', unit: 'mg/L', value: 0, min: 0, max: 1e5, help: 'Calcium sulphate dihydrate present before equilibration (for example a seeded slurry).' },
    ] },
    { group: 'Precipitation population balance', tab: 'setup', help: 'Nucleation and growth of one mineral in the bulk concentrate during its residence time, solved as a population balance coupled to the speciation. The initial condition is the crystal population entered here.', fields: [
      { key: 'pbMineral', label: 'Mineral followed by the kinetic models', type: 'select', value: 'auto', options: [{ value: 'auto', label: 'Automatic: highest supersaturation' }, ...['calcite', 'gypsum', 'barite', 'celestite', 'fluorite', 'silica'].map((id) => ({ value: id, label: MINERALS[id].name }))], help: 'Used by the population balance, the reactive-transport model and the channel flow model.' },
      { key: 'seedN', label: 'Initial nuclei population', unit: '#/mL', value: 1000, min: 0, max: 1e12, help: 'Number of crystal nuclei or foreign particles that act as growth sites at time zero. Cartridge-filtered water still carries 10²–10⁴ sub-micron particles per mL.' },
      { key: 'seedL0', label: 'Size of the initial nuclei', unit: 'µm', value: 0.2, min: 0.001, max: 50, help: 'Diameter of the initial nuclei.' },
      { key: 'seedMass', label: 'Initial precipitate mass', unit: 'mg/L', value: 0, min: 0, max: 1e5, help: 'Mass of crystals of this mineral already suspended at time zero (seeding, carry-over, recycle).' },
      { key: 'seedL', label: 'Crystal size of the initial precipitate', unit: 'µm', value: 5, min: 0.01, max: 2000, help: 'Diameter of the crystals that make up the initial precipitate mass.', showIf: (v) => v.seedMass > 0 },
    ] },
    { group: 'Reactive transport and channel flow', tab: 'setup', help: 'One-dimensional advection–dispersion–reaction along a flow channel with wall precipitation or dissolution, and a two-dimensional flow and concentration field of the tail membrane channel.', fields: [
      { key: 'rtMode', label: 'Reactive-transport case', type: 'select', value: 'scaling', options: [{ value: 'off', label: 'Off' }, { value: 'scaling', label: 'Start-up: concentrate displaces feed water, scale forms on the wall' }, { value: 'flush', label: 'Flush: feed water displaces concentrate and dissolves the wall deposit' }], help: 'Inlet chemistry and initial water are taken from the feed and the wall-state concentrate of this run.' },
      { key: 'rtInlet', label: 'Inlet boundary condition', type: 'select', value: 'conc', options: [{ value: 'conc', label: 'Prescribed concentration' }, { value: 'flux', label: 'Prescribed species flux (no back-dispersion)' }], help: 'A prescribed flux lets only u·c of the inlet water in; a prescribed concentration also allows dispersion across the inlet. The outlet is always convective.', showIf: (v) => v.rtMode !== 'off' },
      { key: 'rtWall', label: 'Wall boundary condition', type: 'select', value: 'kinetic', options: [{ value: 'kinetic', label: 'Reactive mineral surface (rate law)' }, { value: 'equilibrium', label: 'Equilibrium with the mineral' }, { value: 'none', label: 'Zero flux (inert wall)' }], help: 'Rate law: second order in supersaturation for precipitation, first order in undersaturation for dissolution (only while deposit is left).', showIf: (v) => v.rtMode !== 'off' },
      { key: 'rtL', label: 'Channel length', unit: 'm', value: 1, min: 0.05, max: 50, help: 'One spiral-wound element is about 1 m long.' },
      { key: 'rtU', label: 'Mean velocity', unit: 'm/s', value: 0.1, min: 0.005, max: 2, help: 'Cross-flow velocity in the channel.' },
      { key: 'rtH', label: 'Channel height', unit: 'mm', value: 0.71, min: 0.2, max: 5, help: 'Feed-spacer thickness; sets the wall area per volume (2/h) and the half-height of the flow model.' },
      { key: 'rtDisp', label: 'Longitudinal dispersivity', unit: 'm', value: 0.02, min: 0, max: 2, help: 'Dispersion coefficient = dispersivity × velocity + molecular diffusion.', showIf: (v) => v.rtMode !== 'off' },
      { key: 'rtPV', label: 'Simulated time', unit: 'channel volumes', value: 3, min: 0.3, max: 200, help: 'Duration as a multiple of the residence time.', showIf: (v) => v.rtMode !== 'off' },
      { key: 'rtM0', label: 'Initial wall deposit', unit: 'g/m²', value: 5, min: 0, max: 5000, help: 'Initial mineral inventory on the wall that the flush can dissolve.', showIf: (v) => v.rtMode === 'flush' },
      { key: 'rtDiss', label: 'Dissolution rate relative to growth', unit: '×', value: 10, min: 0.01, max: 1e4, help: 'Ratio of the dissolution rate constant to the growth rate constant.', showIf: (v) => v.rtMode !== 'off' && v.rtWall === 'kinetic' },
      { key: 'cfdOn', label: 'Solve the flow and concentration field of the tail channel', type: 'bool', value: true, help: 'Two-dimensional laminar flow between membranes with permeation: momentum equation with concentration-dependent viscosity, continuity and solute transport, marched along the channel. Gives the local polarisation factor and the wall saturation index.' },
      { key: 'cfdFlux', label: 'Permeate flux in the channel', unit: 'L/m²·h', value: 15, min: 1, max: 80, help: 'Water flux through the membrane walls of the channel.', showIf: (v) => v.cfdOn },
      { key: 'cfdMix', label: 'Spacer mixing factor on the diffusivity', unit: '×', value: 1, min: 0.5, max: 20, help: '1 = open channel (upper bound of polarisation). Values of 2–5 mimic the extra transverse mixing of a feed spacer.', showIf: (v) => v.cfdOn },
      { key: 'cfdSpacer', label: 'Resolve a spacer-filled section with the Navier–Stokes solver', type: 'bool', value: false, showIf: (v) => v.cfdOn, help: 'Solves the two-dimensional Navier–Stokes and salt-transport equations (finite-volume solver of suite 4) in a section of the tail channel with feed-spacer filaments and permeating membranes on both walls, then evaluates the speciation engine on the local wall composition: saturation-index map, deposition flux and hot spots behind the filaments. Takes a few seconds.' },
      { key: 'cfdArr', label: 'Filament arrangement', type: 'select', value: 'zigzag', options: [{ value: 'zigzag', label: 'Zigzag (alternating walls)' }, { value: 'cavity', label: 'Cavity (all on one membrane)' }, { value: 'submerged', label: 'Submerged (mid-channel)' }, { value: 'none', label: 'No filaments (open slit)' }], showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdLm', label: 'Filament spacing', unit: 'mm', value: 3, min: 0.5, max: 20, typical: [2, 6], help: 'Centre-to-centre distance of successive transverse filaments.', showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdDf', label: 'Filament diameter', unit: '% of channel height', value: 50, min: 10, max: 85, help: 'About half the channel height for a two-layer net spacer.', showIf: (v) => v.cfdOn && v.cfdSpacer && v.cfdArr !== 'none' },
      { key: 'cfdNFil', label: 'Filament spacings resolved', unit: '', value: 6, min: 2, max: 20, step: 1, help: 'Length of the resolved section = spacings × filament spacing, placed at the concentrate end of the channel. The first spacing is an entrance length and is left out of the statistics.', showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdOsm', label: 'Flux responds to the local osmotic pressure', type: 'bool', value: true, help: 'Solution–diffusion wall: J = A·(ΔP − Δπ(c wall)) with the pressure of the Inputs tab, A fitted so that the flux at the section inlet equals the entered flux. Unticked: uniform flux.', showIf: (v) => v.cfdOn && v.cfdSpacer },
    ] },
    { group: 'Pretreatment: sorption and softening', tab: 'setup', help: 'Optional steps ahead of the concentration: sorption of silica and boron on ferric-hydroxide floc (surface complexation) and sodium-cycle ion-exchange softening. Both change the water that is concentrated.', fields: [
      { key: 'scmOn', label: 'Ferric coagulation: sorb silica and boron', type: 'bool', value: false, help: 'Generalised two-layer surface-complexation model for hydrous ferric oxide with the diffuse-layer charge–potential relation.' },
      { key: 'feDose', label: 'Iron dose', unit: 'mg/L Fe', value: 5, min: 0.05, max: 500, help: 'Ferric iron precipitated as hydrous ferric oxide (0.2 mol sorption sites per mol Fe, 600 m²/g).', showIf: (v) => v.scmOn },
      { key: 'ixOn', label: 'Sodium-cycle softener ahead of the concentration', type: 'bool', value: false, help: 'Strong-acid cation resin exchanging Ca²⁺ and Mg²⁺ for Na⁺ (Gaines–Thomas convention), with counter-current regeneration by NaCl brine and an equilibrium-stage breakthrough curve.' },
      { key: 'ixCap', label: 'Total resin capacity', unit: 'eq/L resin', value: 2, min: 0.2, max: 5, help: 'Strong-acid cation resins: 1.8–2.2 eq per litre of bed.', showIf: (v) => v.ixOn },
      { key: 'ixKCa', label: 'log K calcium / sodium', unit: '', value: 0.8, min: -1, max: 3, help: 'Gaines–Thomas selectivity for Ca²⁺ + 2 NaX = CaX₂ + 2 Na⁺.', showIf: (v) => v.ixOn },
      { key: 'ixKMg', label: 'log K magnesium / sodium', unit: '', value: 0.6, min: -1, max: 3, help: 'Gaines–Thomas selectivity for Mg²⁺ + 2 NaX = MgX₂ + 2 Na⁺.', showIf: (v) => v.ixOn },
      { key: 'ixRegen', label: 'Regenerant dose', unit: 'g NaCl/L resin', value: 120, min: 0, max: 400, help: 'Salt used per litre of resin and cycle; 80–160 g/L is typical.', showIf: (v) => v.ixOn },
      { key: 'ixRegenPct', label: 'Regenerant strength', unit: '% NaCl', value: 10, min: 2, max: 25, help: 'Mass fraction of NaCl in the regenerant brine.', showIf: (v) => v.ixOn },
      { key: 'ixLeak', label: 'Breakthrough criterion', unit: '% of feed hardness', value: 5, min: 0.1, max: 50, help: 'The service run ends when the effluent hardness exceeds this share of the feed hardness.', showIf: (v) => v.ixOn },
    ] },
    { group: 'Surrogate model', tab: 'setup', help: 'A regression model trained on the speciation engine for fast sensitivities, with an out-of-sample test.', fields: [
      { key: 'mlOn', label: 'Train a surrogate of the wall saturation index', type: 'bool', value: false, help: 'Gaussian-kernel ridge regression over recovery, feed pH and temperature; the kernel length is chosen by leave-one-out cross-validation and the model is tested on points it has not seen.' },
      { key: 'mlCompare', label: 'Also train a neural network and a Gaussian process and select the best', type: 'bool', value: true, showIf: (v) => v.mlOn, help: 'A feed-forward tanh network (mini-batch Adam, early stopping) and a Gaussian process (anisotropic squared-exponential kernel, marginal-likelihood hyper-parameters) from the optimisation suite are trained on the same design points; all three learners are compared on engine runs that none of them has seen.' },
      { key: 'mlSeed', label: 'Random seed', unit: '', value: 11, min: 1, max: 99999, step: 1, showIf: (v) => v.mlOn, help: 'Seed of the Latin-hypercube design, the network initialisation and the mini-batch order: the same seed reproduces the result exactly.' },
      { key: 'mlN', label: 'Training points', unit: '', value: 48, min: 16, max: 300, step: 1, help: 'Latin-hypercube samples of the speciation engine; a third as many again are used for testing.', showIf: (v) => v.mlOn },
    ] },
    { group: 'Sweep resolution', tab: 'mesh', help: 'The maximum recovery of each mineral is interpolated on the recovery sweep, so its resolution is a discretisation parameter.', fields: [
      { key: 'nRec', label: 'Points in the recovery sweep', unit: '', value: 36, min: 6, max: 200, step: 1 },
      { key: 'nCF', label: 'Points in the precipitation path', unit: '', value: 12, min: 4, max: 60, step: 1 },
      { key: 'rtN', label: 'Cells along the channel (reactive transport)', unit: '', value: 40, min: 8, max: 400, step: 1, help: 'Finite-volume cells of the advection–dispersion–reaction model.' },
      { key: 'cfdNxFil', label: 'Cells per filament spacing (spacer section)', unit: '', value: 24, min: 8, max: 80, step: 1, help: 'Axial finite-volume cells per filament spacing of the Navier–Stokes model.', showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdNyFull', label: 'Cells across the full channel (spacer section)', unit: '', value: 32, min: 12, max: 120, step: 1, help: 'Wall-clustered cells between the two membranes of the Navier–Stokes model.', showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdNy', label: 'Cells across the half-channel (flow model)', unit: '', value: 30, min: 8, max: 120, step: 1, help: 'Wall-refined cells between the membrane and the mid-plane.' },
      { key: 'nPB', label: 'Output steps of the population balance', unit: '', value: 120, min: 20, max: 2000, step: 1, help: 'Time levels (quadratically spaced) at which new nuclei classes are created.' },
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

  async run(v, ctx) {
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
    if (v.model !== 'pitzer' && wd.I > ({ dh: 0.01, davies: 0.5, edh: 0.1, sit: 3.5, bromley: 6 }[v.model] ?? 1)) W.push({ level: 'warn', msg: `Ionic strength ${fmt(wd.I, 3)} mol/kg is outside the validity range of the selected activity model — switch to Pitzer.` });
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
    const kwc = solutionToIons(conc).kgwPerL, reservoir = {};
    for (const [id, mg] of [['calcite', v.invCalcite], ['gypsum', v.invGypsum]]) if (mg > 0 && pset.includes(id)) reservoir[id] = (mg / 1000 / MINERALS[id].mw / kwc) * conc.w; // initial mineral inventory, mol
    const pr = precipitateSolution(conc, pset, { P, dk, pCO2: v.co2 === 'open' ? v.pCO2 * 1e-6 : null, reservoir }), after = solutionToIons(pr.sol), Qc = v.Q * (1 - R);
    const solidRows = Object.entries(pr.solids).filter(([, x]) => x > 1e-12).map(([id, x]) => { const c = mgL(conc, x, MINERALS[id].mw); return [MINERALS[id].name, MINERALS[id].formula, c, (c * Qc * 24) / 1000, (x / conc.w) * 1000]; });
    for (const [id, x] of Object.entries(pr.solids)) if (x < -1e-12) { const c = mgL(conc, x, MINERALS[id].mw); solidRows.push([`${MINERALS[id].name} (dissolves from the initial inventory)`, MINERALS[id].formula, c, (c * Qc * 24) / 1000, (x / conc.w) * 1000]); }
    const totalSolid = sum(solidRows.filter((r) => r[2] > 0).map((r) => r[2]));
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

    ctx?.progress?.(0.9, 'Kinetics, transport and cross-checks…');
    const X = extendedModels(v, a, { pset, pr, reservoir });
    if (v.cfdOn !== false && v.cfdSpacer && X.march) { ctx?.progress?.(0.92, 'Navier–Stokes solution of the spacer-filled section…'); await spacerScaling(v, a, X, ctx); }
    if (v.mlOn && v.mlCompare !== false && X.sur) { ctx?.progress?.(0.97, 'Training neural network and Gaussian process…'); await compareLearners(v, a, X, ctx); }
    W.push(...X.W);
    const brine = { Q: Qc, T: v.T, P: v.P, pH: +cd.pH.toFixed(3), tds: cd.tds, ions: Object.fromEntries(ION_IDS.map((k) => [k, +cd.ions[k].toPrecision(6)])) };
    const recStr = (r) => (r == null ? `> ${fmt(100 * a.Rmax, 3)}` : fmt(100 * r, 3));
    const allIds = Object.keys(wd.SI).filter((id) => MINERALS[id].group !== 'salt' || id === 'halite' || wd.SI[id] > -1);
    const spec = cd.species.filter((s) => s.m > 1e-12).sort((p, q) => q.m - p.m);
    const corr = fd.larsonSkold, sat = (id) => (wd.SI[id] != null ? 100 * 10 ** wd.SI[id] : 0);
    const out = {
      streams: { brine }, SI: Object.fromEntries(Object.entries(cd.SI).map(([k, x]) => [k, +x.toFixed(4)])), SIwall: Object.fromEntries(Object.entries(wd.SI).map(([k, x]) => [k, +x.toFixed(4)])),
      maxRecovery: best.r ?? a.Rmax, maxRecoveryNoAntiscalant: a.limitPlain.r ?? a.Rmax, maxRecoveryAntiscalant: a.limitAS.r ?? a.Rmax, limitingMineral: best.id || 'none', antiscalantDose: asDose, acidDose: a.dose.mg, doseChemical: a.dose.reagent || 'none',
      scalingMargin: Math.max(-9, ...set.filter((id) => wd.SI[id] != null).map((id) => wd.SI[id] - limOf(id))),
      lsi: cd.siCalcite, lsiClassic: cd.lsi, sdsi: cd.sdsi, ionicStrength: cd.I, waterActivity: cd.aw, osmoticPressureBar: cd.osmoticPressure, pHConcentrate: cd.pH, feedPH: fd.pH, precipitationPotential: totalSolid, ccpp: cd.ccpp, density: cd.density, model: v.model, ...X.out,
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
        ...X.K,
      ],
      recommendations: [
        best.r != null && R > best.r ? `Reduce the recovery to about ${fmt(100 * best.r - 1, 3)} % or treat for ${name(best.id).toLowerCase()} (see the recovery sweep).` : null,
        (wd.SI.calcite ?? -9) > limOf('calcite') && v.doseMode === 'none' ? 'Calcium carbonate is beyond the inhibitor limit: set pH adjustment to “target concentrate calcite index” to size the acid dose.' : null,
        !v.antiscalant && W.some((w) => w.level === 'bad') ? 'Enable antiscalant dosing: most sparingly soluble salts can be held well above saturation by a threshold inhibitor.' : null,
        corr > 1.2 && (fd.siCalcite ?? 0) < 0 ? `Larson–Skold index ${fmt(corr, 3)} with a negative calcite index: the water is corrosive to carbon steel and cement linings — specify duplex/GRP wetted parts.` : null,
        v.model !== 'pitzer' && v.model !== 'bromley' && cd.I > (v.model === 'sit' ? 3.5 : 0.5) ? 'Switch the activity model to Pitzer for this salinity.' : null,
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
        { type: 'line', title: 'Mean activity coefficient of NaCl by model (25 °C)', xlabel: 'Ionic strength (mol/kg)', ylabel: 'γ±', logx: true, ymin: 0, ymax: 1.6, series: [...gam, { name: 'Measured (Hamer & Wu 1972)', x: NACL_LIT.m, y: NACL_LIT.g, mode: 'points' }], vlines: [{ x: clamp(cd.I, 0.001, 6), label: 'this brine' }] },
        { type: 'field', title: `Scaling margin: worst SI minus its ${v.antiscalant ? 'antiscalant' : 'saturation'} limit`, xlabel: 'Recovery (%)', ylabel: 'Feed pH after adjustment', zlabel: 'margin', zunit: 'SI', x: fx.map((r) => 100 * r), y: fy, z: fz, zmin: -3, zmax: 3, cmap: 'coolwarm', contours: 8, markers: [{ x: 100 * R, y: clamp(fd.pH, 5.5, 9), label: 'design' }], note: 'Negative (blue) = every mineral within its limit; positive (red) = at least one mineral beyond it.' },
        { type: 'bar', title: 'Saturation at the membrane wall', ylabel: 'Saturation index', categories: shown.map((id) => MINERALS[id].name), series: [{ name: 'Saturation index', values: shown.map((id) => clamp(wd.SI[id] ?? -8, -8, 8)) }, { name: 'Limit', values: shown.map((id) => limOf(id)) }] },
        ...(mixPlot ? [mixPlot] : []),
        ...X.PL,
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
        ...X.TB,
        { title: 'Data provenance', columns: ['Parameter set', 'Source', 'How it was checked', 'Validity range', 'Status'], rows: PROVENANCE.map((r) => r.slice()),
          note: 'Every tabulated constant of the chemistry engine was compared with the named database file or table. “Replaced” = the earlier value could not be found and was exchanged for a documented one; “analogue” = assigned by chemical similarity; “unconfirmed” = no source could be retrieved, and the entry is not used by the default (Pitzer) calculation path. The activity models are tested against the NIST activity-coefficient tables on the Verify tab.' },
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
          ...X.BAL,
        ];
      })(),
      outputs: out,
    };
  },

  mesh: [{ name: 'Recovery-sweep resolution', keys: ['nRec'], min: 6, note: 'Scaling-limited recoveries are interpolated between sweep points; the study refines the sweep and reports the numerical uncertainty of those limits.',
    metrics: [{ label: 'Max recovery with antiscalant', unit: '–', get: (r) => r.outputs.maxRecoveryAntiscalant }, { label: 'Max recovery without antiscalant', unit: '–', get: (r) => r.outputs.maxRecoveryNoAntiscalant }] },
  { name: 'Reactive-transport grid', keys: ['rtN'], min: 8, note: 'Cells along the channel of the advection–dispersion–reaction model (needs the reactive-transport case switched on).', metrics: [{ label: 'Mean wall deposit', unit: 'g/m²', get: (r) => r.outputs.rtDepositMean ?? 0 }, { label: 'Outlet saturation index', unit: '', get: (r) => r.outputs.rtOutletSI ?? 0 }] },
  { name: 'Spacer-section grid (Navier–Stokes)', keys: ['cfdNxFil', 'cfdNyFull'], min: 8, note: 'Cells per filament spacing and across the channel of the Navier–Stokes model (needs the spacer-filled section switched on). The mean polarisation converges quickly; the hot-spot value at a filament contact line converges slowly.', metrics: [{ label: 'Mean polarisation factor', unit: '–', get: (r) => r.outputs.spacerBetaMean ?? 1 }, { label: '95th-percentile polarisation factor', unit: '–', get: (r) => r.outputs.spacerBeta95 ?? 1 }] },
  { name: 'Flow-model grid across the channel', keys: ['cfdNy'], min: 8, note: 'Wall-normal cells of the two-dimensional channel model.', metrics: [{ label: 'Largest local polarisation factor', unit: '–', get: (r) => r.outputs.cfdBetaMax ?? 1 }, { label: 'Channel pressure drop', unit: 'bar', get: (r) => r.outputs.cfdPressureDropBar ?? 0 }] }],

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

  async verify() {
    const C = [], add = (name, expected, got, tol, note) => C.push({ name, expected, got, tol, pass: Math.abs(got - expected) <= tol, note });
    const n1 = saltActivity('Na', 'Cl', 1), n6 = saltActivity('Na', 'Cl', 6), ca = saltActivity('Ca', 'Cl', 1);
    add('NaCl 1 mol/kg: mean activity coefficient', 0.657, n1.gamma, 0.004, 'Pitzer model against Hamer & Wu (1972), 25 °C');
    add('NaCl 1 mol/kg: osmotic coefficient', 0.936, n1.phi, 0.003, 'Pitzer model against Hamer & Wu (1972)');
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
    add('Seawater calcite saturation state Ω at pH 8.22', 4.9, 10 ** saturationIndex(equilibrate(sw, { pH: 8.22 }).eq, 'calcite'), 1, 'Surface seawater (activity-scale pH 8.2) is 4–6 times supersaturated');
    { // stoichiometric solubility product of calcite in seawater: an independent measurement of Ksp·γ products
      const e = sw.eq, S = solutionToIons(sw).salinity, kgw = 1 - S / 1000, co3T = e.m[IC] + e.m[si('CaCO3°')] + e.m[si('MgCO3°')] + e.m[si('NaCO3')], rS = Math.sqrt(S);
      const mucci = -(-171.9065 - 0.077993 * 298.15 + 2839.319 / 298.15 + 71.595 * Math.log10(298.15) + (-0.77712 + 0.0028426 * 298.15 + 178.34 / 298.15) * rS - 0.07711 * S + 0.0041249 * S * rS);
      add('Seawater: stoichiometric calcite solubility product pK*sp (mol²/kg² of seawater)', mucci, -(Math.log10(e.tot[mi('Ca')] * co3T) - swSI.calcite + 2 * Math.log10(kgw)), 0.02, 'Measured by Mucci (1983), salinity fit as used in the CO2SYS programs; tests log K of calcite together with the Pitzer activity coefficients of Ca²⁺ and CO₃²⁻');
      const hco3T = e.m[si('HCO3')], pk1 = 3633.86 / 298.15 - 61.2172 + 9.6777 * Math.log(298.15) - 0.011555 * S + 0.0001152 * S * S, pk2 = 471.78 / 298.15 + 25.929 - 3.16967 * Math.log(298.15) - 0.01781 * S + 0.0001122 * S * S;
      add('Seawater: ratio of the stoichiometric carbonic-acid constants log(K₁*/K₂*)', pk2 - pk1, Math.log10((hco3T * hco3T) / (e.m[ICO2] * co3T)), 0.03, 'Lueker, Dickson & Keeling (2000) refit of the Mehrbach measurements; the ratio [HCO₃⁻]²/([CO₂][CO₃²⁻]) does not depend on the pH scale');
    }
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
    // Bromley and specific-ion-interaction models
    add('Bromley model: NaCl 1 mol/kg mean activity coefficient', 0.657, saltActivity('Na', 'Cl', 1, { model: 'bromley' }).gamma, 0.004, 'Robinson & Stokes, 25 °C');
    add('Bromley model: CaCl₂ 1 mol/kg mean activity coefficient', 0.5, saltActivity('Ca', 'Cl', 1, { model: 'bromley' }).gamma, 0.01, '2:1 electrolyte, literature 0.500');
    add('Bromley model: NaCl 6 mol/kg mean activity coefficient', 0.986, saltActivity('Na', 'Cl', 6, { model: 'bromley' }).gamma, 0.02, 'Near halite saturation');
    add('Specific-ion-interaction model: NaCl 0.1 mol/kg', 0.778, saltActivity('Na', 'Cl', 0.1, { model: 'sit' }).gamma, 0.006, 'Robinson & Stokes');
    add('Specific-ion-interaction model: CaCl₂ 0.1 mol/kg', 0.518, saltActivity('Ca', 'Cl', 0.1, { model: 'sit' }).gamma, 0.006, 'Literature 0.518');
    // Gibbs-energy minimisation against the mass-action solver (same closed system)
    const set4 = ['calcite', 'gypsum', 'barite', 'celestite'], gm = gibbsMinimize(brine, set4), G0 = gibbsEnergy(brine), half = withdraw(withdraw(brine, 'gypsum', (0.5 * pr.solids.gypsum) / brine.w), 'calcite', (0.5 * pr.solids.calcite) / brine.w);
    add('Gibbs-energy minimisation reproduces the mass-action pH', pr.sol.pH, gm.pH, 1e-6, 'Seawater concentrated 4.2 times with calcite, gypsum, barite and celestite; independent solver and starting point');
    add('Gibbs-energy minimisation reproduces the precipitated amounts', 0, Math.max(...set4.map((id) => Math.abs(gm.solids[id] - pr.solids[id]) / Math.max(Math.abs(pr.solids[id]), 1e-9))), 1e-5, 'Largest relative difference of any solid');
    add('Gibbs energy falls monotonically towards equilibrium', 1, G0 > gibbsEnergy(half, { gypsum: 0.5 * pr.solids.gypsum, calcite: 0.5 * pr.solids.calcite }) && gibbsEnergy(half, { gypsum: 0.5 * pr.solids.gypsum, calcite: 0.5 * pr.solids.calcite }) > gm.G ? 1 : 0, 0, 'G(supersaturated) > G(half precipitated) > G(equilibrium)');
    const over = withdraw(brine, 'gypsum', (1.3 * pr.solids.gypsum) / brine.w);
    add('Equilibrium is a minimum: precipitating 30 % too much gypsum costs Gibbs energy', 1, gibbsEnergy(over, { gypsum: 1.3 * pr.solids.gypsum }) > gibbsEnergy(withdraw(brine, 'gypsum', gibbsMinimize(brine, ['gypsum']).solids.gypsum / brine.w), { gypsum: gibbsMinimize(brine, ['gypsum']).solids.gypsum }) ? 1 : 0, 0, 'Gypsum-only system, overshoot compared with the minimum');
    // initial mineral inventory: calcite in undersaturated acidified water dissolves to saturation
    const acidw = equilibrate(sw, { pH: 6.5 }), dis = precipitateSolution(acidw, ['calcite'], { reservoir: { calcite: 0.01 } }), gdis = gibbsMinimize(acidw, ['calcite'], { reservoir: { calcite: 0.01 } });
    add('Initial calcite inventory dissolves into undersaturated water until saturation', 0, dis.solids.calcite < 0 ? saturationIndex(dis.sol.eq, 'calcite') : 9, 1e-6, 'Negative precipitated amount = dissolution from the inventory; SI = 0 at the end');
    add('…and the Gibbs minimisation finds the same dissolved amount', 1, gdis.solids.calcite / dis.solids.calcite, 1e-5, 'Inventory supplied to both solvers');
    // population balance
    const kv = Math.PI / 6, pa = scalePBE({ rates: () => ({ J: 1e9, G: 1e-8, Lc: 1e-12 }), classes0: [], rhoM: 1, kv, tEnd: 100, nStep: 200 });
    add('Population balance: crystal number equals J·t', 1, pa.mu[0] / 1e11, 1e-9, 'Constant nucleation rate, no initial crystals');
    add('Population balance: crystal volume equals k_v·J·G³·t⁴/4', 1, pa.volume / ((kv * 1e9 * 1e-24 * 1e8) / 4), 1e-3, 'Analytical solution for constant nucleation and growth');
    const ps = scalePBE({ rates: () => ({ J: 0, G: 1e-8, Lc: 1e-9 }), classes0: [{ N: 1e6, L: 1e-6 }], rhoM: 1, kv, tEnd: 100 });
    add('Population balance: initial nuclei grow by G·t and keep their number', 2e-6, ps.L[0] * (ps.mu[0] / 1e6), 1e-12, 'Initial nuclei population of 1 µm crystals, G = 10 nm/s for 100 s');
    const vk = { ...d, tRes: 86400, seedN: 0, seedMass: 0 }, bare = scaleKinetics(brine, 'gypsum', vk), seeded = scaleKinetics(brine, 'gypsum', { ...vk, seedMass: 200, seedL: 5 });
    add('Initial precipitate mass accelerates desupersaturation', 1, seeded.conversion > bare.conversion + 0.05 && seeded.siEnd < bare.siEnd ? 1 : 0, 0, `Gypsum in 4.2-times concentrated seawater, 1 day: conversion ${fmt(100 * bare.conversion, 3)} % unseeded, ${fmt(100 * seeded.conversion, 3)} % with 200 mg/L of 5 µm seed`);
    add('Initial precipitate mass is carried by the initial population', 200, seeded.pb.volume0 * MINERALS.gypsum.rho * 1000, 1e-6, 'k_v·N·L³·ρ of the seed class, mg/L');
    // advection–dispersion–reaction
    const ob = adr1d({ N: 200, L: 1, u: 1, D: 0.02, tEnd: 0.6, inlet: 'conc', fields: [{ cin: 1, c0: 0 }] });
    add('Advection–dispersion: Ogata–Banks solution', 0, Math.max(...ob.x.map((x, i) => Math.abs(ob.c[0][i] - ogataBanks(x, 0.6, 1, ob.Deff)))), 0.012, 'Largest deviation of the concentration profile, prescribed-concentration inlet, Péclet number 50, 200 cells');
    add('Prescribed-concentration inlet: tracer balance closes', 0, ob.balance[0].in - ob.balance[0].out - ob.balance[0].store, 1e-12, 'Inflow − convective outflow = storage');
    const fx2 = adr1d({ N: 200, L: 1, u: 1, D: 0.02, tEnd: 0.6, inlet: 'flux', fields: [{ cin: 1, c0: 0 }] });
    add('Prescribed-flux inlet admits exactly u·c·t', 0.6, fx2.balance[0].in, 1e-12, 'Danckwerts condition: no dispersive flux across the inlet');
    add('Convective outlet: what entered is stored or has left by advection', 0.6, fx2.balance[0].store + fx2.balance[0].out, 1e-10, 'Zero dispersive flux at the outlet');
    const dec = adr1d({ N: 200, L: 1, u: 1, D: 0.02, tEnd: 6, inlet: 'conc', fields: [{ cin: 1, c0: 0 }], react: (cc, dt) => { for (let i = 0; i < cc[0].length; i++) cc[0][i] /= 1 + 2 * dt; } });
    add('Advection–dispersion–reaction: steady first-order decay', Math.exp(((1 - Math.sqrt(1 + 8 * dec.Deff)) / (2 * dec.Deff)) * dec.x[99]), dec.c[0][99], 0.004, 'c = exp[x(u − √(u² + 4kD))/2D] at mid-length');
    const rtb = { h: 7.1e-4, L: 1, u: 0.1, D: 0.002, pv: 4, N: 40 }, rk = reactiveTransport(sw, brine, 'gypsum', { ...rtb, wall: 'kinetic' }), re = reactiveTransport(sw, brine, 'gypsum', { ...rtb, wall: 'equilibrium', table: rk.table }), rz = reactiveTransport(sw, brine, 'gypsum', { ...rtb, wall: 'none', table: rk.table });
    add('Zero-flux wall: nothing deposits and the outlet keeps the inlet saturation', 0, rz.depositMean + Math.abs(rz.si[39] - rz.siAt(1, 0)), 1e-9, 'Inert-wall boundary condition');
    add('Equilibrium mineral wall: the water leaves exactly saturated', 0, re.si[39], 1e-6, 'Local-equilibrium boundary condition (gypsum from 4.2-times concentrated seawater)');
    add('Reactive mineral surface: deposit lies between the inert and the equilibrium wall', 1, rk.depositMean > 0 && rk.depositMean < re.depositMean ? 1 : 0, 0, `Kinetic ${fmt(rk.depositMean, 3)} g/m², equilibrium ${fmt(re.depositMean, 3)} g/m²`);
    add('Reactive transport: mineral balance closes', 0, (rk.balance.in - rk.balance.out) / rk.balance.in, 1e-9, 'Mineral taken up by the wall = loss from the water (stored + convected out)');
    const fl = reactiveTransport(brine, sw, 'gypsum', { ...rtb, wall: 'kinetic', M0: 5 / 172.17, pv: 20 });
    add('Dissolution surface condition: an undersaturated flush removes wall inventory, most at the inlet', 1, fl.depositGm2[0] < fl.depositGm2[39] && fl.depositGm2[39] < 5 && fl.si[39] < 0 && fl.si[39] > fl.si[0] ? 1 : 0, 0, `5 g/m² gypsum initially; after 20 channel volumes ${fmt(fl.depositGm2[0], 3)} g/m² at the inlet, ${fmt(fl.depositGm2[39], 3)} g/m² at the outlet`);
    // channel flow model
    const Hh = 3.55e-4, cf0 = channelCFD({ L: 1, H: Hh, u0: 0.15, vw: 0, D: 1.5e-9, visc: () => 1e-3, ny: 40 });
    add('Channel flow model: Poiseuille pressure gradient without permeation', (-3 * 1e-3 * 0.15) / Hh ** 2, cf0.dpdx[0], 12, 'dp/dx = −3μu/H² for a slit of half-height H, Pa/m');
    add('Channel flow model: peak velocity is 1.5 times the mean', 1.5, cf0.umax[0] / 0.15, 0.01, 'Parabolic profile from the numerically solved momentum equation');
    const cf1 = channelCFD({ L: 0.02, H: Hh, u0: 0.15, vw: 1e-7, D: 1.5e-9, visc: () => 1e-3, ny: 60, nx: 400, grow: 1.08 }), xe = cf1.x[400];
    add('Channel flow model: Lévêque solution for uniform wall flux', 1 / 0.6508, ((cf1.cw[400] / cf1.cb[400] - 1) * (1.5e-9 / xe) * ((((3 * 0.15) / Hh) * xe * xe) / 1.5e-9) ** (1 / 3)) / 1e-7, 0.02, 'Wall excess (c_w − c_b)/c_b = v_w·x/(0.6508·D)·(γx²/D)^(−1/3) in the entrance region');
    const cf2 = channelCFD({ L: 1, H: Hh, u0: 0.1, vw: 4e-6, D: 1.5e-9, visc: (cc) => 1e-3 * (1 + 0.1 * (cc - 1)), ny: 30 });
    add('Channel flow model: salt is conserved', 1, cf2.balance.out / cf2.balance.in, 1e-9, 'Complete rejection: salt flow in = salt flow out');
    add('Channel flow model: bulk concentration follows 1/(1 − recovery)', 1, cf2.cb[cf2.cb.length - 1] * (1 - cf2.recovery), 1e-9, 'Mixing-cup concentration at the outlet');
    // surface complexation
    const pzc = surfaceComplexation(null, 10, { act: {}, I: 0.1, pH: 8.11 }), sc6 = surfaceComplexation(null, 10, { act: {}, I: 0.1, pH: 6 });
    add('Surface complexation: zero surface potential at the point of zero charge', 0, pzc.psi, 1e-9, 'pH = ½(pKa1 + pKa2) = 8.11 for hydrous ferric oxide without specific sorbates');
    add('Surface complexation: surface charge equals the diffuse-layer charge', sc6.sigmaDiffuse, sc6.sigma, 1e-9, 'Gouy–Chapman relation σ = 0.1174·√I·sinh(Fψ/2RT) at pH 6, C/m²');
    const bw = makeSolution({ ions: WATERS.brackish.ions, T: 25, pH: 7.6 }), scb = surfaceComplexation(bw.eq, 10, { kgwPerL: bw.kgwPerL });
    add('Surface complexation: site and silica balances close', 2, sum(scb.species.map((q) => q.frac)) + (scb.sorbed.Si + scb.dissolved.Si) / scb.total.Si, 1e-9, 'Σ surface species = total sites; sorbed + dissolved silica = total');
    // ion exchange
    const gt = gainesThomas({ Na: 0.01, Ca: 0.002, Mg: 0.001, K: 0.0005 });
    add('Gaines–Thomas exchange: equivalent fractions sum to one', 1, gt.Na + gt.K + gt.Ca + gt.Mg, 1e-12, 'Closed-form solution of the exchanger composition');
    add('Gaines–Thomas exchange: selectivity constant is recovered', 10 ** 0.8, (gt.Ca * 0.01 ** 2) / (gt.Na ** 2 * 0.002), 1e-9, 'β_Ca·a_Na² / (β_Na²·a_Ca) = K');
    const ixc = softenerColumn(bw, {}), ixs = softenerColumn(sw, {});
    add('Softener column: hardness balance closes', 0, (ixc.balance.in - ixc.balance.out) / ixc.balance.in, 1e-9, 'Hardness fed = effluent + gain on the resin');
    add('Softener column: breakthrough close to the stoichiometric capacity', 1, ixc.bvBreak / ixc.bvIdeal, 0.2, 'Favourable isotherm gives a sharp front; brackish water');
    add('Electroselectivity: dilute water loads the resin with more hardness than seawater', 1, ixc.exhausted.Ca + ixc.exhausted.Mg > ixs.exhausted.Ca + ixs.exhausted.Mg + 0.3 ? 1 : 0, 0, 'Divalent ions are preferred more strongly at low ionic strength');
    // ---- independent literature data ------------------------------------------------------------------
    const mLab = { pitzer: 'Pitzer', bromley: 'Bromley', sit: 'SIT', davies: 'Davies' };
    for (const [salt, ref] of Object.entries(ACT_REF)) for (const [model, mMax, tol] of ref.lim) {
      const q = actDeviation(ref, model, mMax);
      add(`${salt} ${q.m0}${q.n > 1 ? `–${q.m1}` : ''} mol/kg, ${mLab[model]} model: γ± and φ against ${ref.src}`, 0, Math.max(q.dg, q.dp), tol, `Largest relative deviation over ${q.n} tabulated molalit${q.n > 1 ? 'ies' : 'y'}: γ± ${fmt(100 * q.dg, 2)} %, φ ${fmt(100 * q.dp, 2)} % (NIST critical evaluation, 25 °C)`);
    }
    { // MgSO4: no critically evaluated table could be retrieved, so these are not independent of the suite's parameters
      const ps = [0.1, 1, 3].map((m) => { const q = saltActivity('Mg', 'SO4', m), h = pitzerSingle(2, 2, m, 0.221, 3.343, -37.23, 0.025); return Math.max(Math.abs(q.gamma / h.gamma - 1), Math.abs(q.phi / h.phi - 1)); });
      add('MgSO₄ 0.1–3 mol/kg, Pitzer model: multi-ion sums against the closed-form single-salt equation', 0, Math.max(...ps), 2e-3, 'Implementation check, not an independent one: both sides use the Harvie–Møller–Weare parameters (β⁰ 0.221, β¹ 3.343, β² −37.23, Cφ 0.025); no tabulated MgSO₄ data could be retrieved');
      const p01 = saltActivity('Mg', 'SO4', 0.1);
      add('MgSO₄ 0.1 mol/kg: SIT, Davies and Bromley against the suite’s Pitzer model', 0, Math.max(...['sit', 'davies', 'bromley'].map((mod) => Math.abs(saltActivity('Mg', 'SO4', 0.1, { model: mod }).gamma / p01.gamma - 1))), 0.15, 'Consistency check, not an independent one: the reference is the suite’s own source-verified Pitzer model. 2:2 salts are outside the range of the Bromley correlation');
      const nn = { c: 'Na', a: 'NO3', d: [[0.1, 0.76, 0.921], [0.5, 0.618, 0.876], [1, 0.549, 0.852], [2, 0.478, 0.826], [3, 0.437, 0.81], [4, 0.408, 0.798], [5, 0.386, 0.789], [6, 0.372, 0.789]] }, qn = actDeviation(nn, 'pitzer', 6);
      add('NaNO₃ 0.1–6 mol/kg, Pitzer model (replaced parameter set): γ± and φ against Hamer & Wu 1972', 0, Math.max(qn.dg, qn.dp), 0.03, `γ± ${fmt(100 * qn.dg, 2)} %, φ ${fmt(100 * qn.dp, 2)} %`);
    }
    { // PHREEQC example 1: speciation of the Nordstrom et al. (1979) seawater with phreeqc.dat (ion-association model)
      const ppm = { Na: 10768, K: 399.1, Ca: 412.3, Mg: 1291.8, Cl: 19353, SO4: 2712, HCO3: 141.682, SiO2: 4.28 }, x1 = analyzeWater({ ions: Object.fromEntries(Object.entries(ppm).map(([k, x]) => [k, x * 1.023])), T: 25, pH: 8.22, model: 'tj' });
      const iap = (id) => x1.SI[id] + MINERALS[id].logKfor(25, 'tj');
      add('PHREEQC example 1 (seawater): ionic strength, Truesdell–Jones model', 0.6737, x1.I, 0.005, 'mol/kg water; published output of the USGS PHREEQC example with phreeqc.dat');
      add('PHREEQC example 1: log ion-activity product of calcite', -7.67, iap('calcite'), 0.08, 'Published −7.67; the two ion-association databases differ slightly in their pair constants');
      add('PHREEQC example 1: log ion-activity product of gypsum', -5.27, iap('gypsum'), 0.06, 'Published −5.27');
      add('PHREEQC example 1: log ion-activity product of halite', -0.91, iap('halite'), 0.03, 'Published −0.91');
      add('PHREEQC example 1: log pCO₂', -3.35, Math.log10(x1.pCO2), 0.06, 'Published −3.35 (atm)');
    }
    { // PHREEQC example 8: diffuse-double-layer model of hydrous ferric oxide in 0.1 mol/kg NaNO3 (zinc at trace level)
      const rx8 = [['≡FeOH₂⁺', '', 1, 1, 7.18], ['≡FeO⁻', '', -1, -1, -8.82]], pub = [[5, 0.2006, 0.1228, 0.555, 0.008], [6, 0.1019, 0.08914, 0.31, 0.032], [7, 0.03934, 0.04745, 0.179, 0.072]];
      let dS = 0, dP = 0, dF = 0;
      for (const [pH, sig, psi, f1, f2] of pub) { const q = surfaceComplexation(null, 1, { act: {}, I: 0.1, pH, rx: rx8, sites: 2.05e-4, area: 54 }); dS = Math.max(dS, Math.abs(q.sigma / sig - 1)); dP = Math.max(dP, Math.abs(q.psi - psi)); dF = Math.max(dF, Math.abs(q.species[1].frac - f1), Math.abs(q.species[2].frac - f2)); }
      add('PHREEQC example 8 (hydrous ferric oxide): surface potential at pH 5, 6 and 7', 0, dP * 1000, 0.3, 'mV; published 122.8, 89.14 and 47.45 mV for 2.05·10⁻⁴ mol sites on 54 m² in 0.1 mol/kg NaNO₃ (example constants log K 7.18 and −8.82)');
      add('PHREEQC example 8: surface charge density', 0, dS, 0.003, 'Relative deviation from the published 0.2006, 0.1019 and 0.0393 C/m²');
      add('PHREEQC example 8: protonated and deprotonated site fractions', 0, dF, 0.002, 'Published ≡FeOH₂⁺ 0.555/0.310/0.179 and ≡FeO⁻ 0.008/0.032/0.072');
    }
    { // tabulated constants that have a closed-form meaning
      add('Calcite log K, ion-pair models (Plummer & Busenberg 1982)', -8.48, MINERALS.calcite.logKfor(25, 'tj'), 0.001, 'WATEQ4F database');
      add('Calcite log K, Pitzer model (Harvie, Møller & Weare 1984)', -8.406, MINERALS.calcite.logKfor(25, 'pitzer'), 0.001, 'EQ3/6 data0.hmw: −8.4062');
      add('Second dissociation constant of carbonic acid, Pitzer set', 10.3392, kc1p(25), 0.001, 'EQ3/6 data0.hmw (HMW 1984): 10.3392; the ion-pair set uses 10.329');
    }
    // surrogate
    const sur = trainSurrogate(base, ['calcite', 'gypsum'], { nTrain: 48, nTest: 16 });
    add('Surrogate of the saturation index: error on held-out speciation runs', 0, sur.rmse, 0.1, 'Root-mean-square error in SI units on 16 points not used for training');
    { // neural network and Gaussian process on the same design points
      const l1 = await trainLearners(sur, { seed: 11 }), l2 = await trainLearners(sur, { seed: 11 }), L = l1.learners;
      add('Surrogate learners: the selected learner on held-out engine runs', 0, l1.best.testRmse, 0.08, `${l1.best.name} selected on the validation points; test RMSE in SI units (kernel regression ${fmt(L.krr.testRmse, 2)}, Gaussian process ${fmt(L.gp.testRmse, 2)}, neural network ${fmt(L.nn.testRmse, 2)})`);
      add('Surrogate learners: Gaussian process on held-out engine runs', 0, L.gp.testRmse, 0.08, 'Root-mean-square error in SI units on 16 runs not used for fitting or selection');
      add('Surrogate learners: neural network on held-out engine runs', 0, L.nn.testRmse, 0.3, 'A network trained on 38 points is the weakest of the three learners; reported, not selected');
      add('Surrogate learners: training is reproducible with the seed', 0, Math.abs(l1.learners.nn.testRmse - l2.learners.nn.testRmse) + Math.abs(l1.learners.gp.testRmse - l2.learners.gp.testRmse), 0, 'Two trainings with the same seed give identical test errors');
    }
    { // spacer-filled channel by the Navier–Stokes solver
      const h = 7.1e-4, u0 = 0.1, vw = 15 / 3.6e6, Dd = 1.4e-9, ns = await spacerChannelCFD({ h, u0, vw, D: Dd, arr: 'none', lm: 3e-3, nFil: 4, nxFil: 16, ny: 24 }), mr = channelCFD({ L: ns.L, H: h / 2, u0, vw, D: Dd, rej: 1, nx: 240, ny: 60, grow: 1.08, visc: () => 1e-3 });
      let dev = 0;
      for (let i = Math.round(0.15 * ns.nx); i < ns.nx; i++) { const t = ns.x[i] / (ns.L / 240), k = Math.floor(t), c = mr.cw[k] * (1 - (t - k)) + mr.cw[Math.min(240, k + 1)] * (t - k); dev = Math.max(dev, Math.abs(ns.cwB[i] / c - 1), Math.abs(ns.cwT[i] / c - 1)); }
      add('Navier–Stokes channel without spacer: wall concentration equals the boundary-layer march', 0, dev, 0.01, 'Largest relative difference of c wall along both membranes (beyond the first 15 % of the length): two independent discretisations of the same problem');
      const sp = await spacerChannelCFD({ h, u0, vw, D: Dd, arr: 'zigzag', lm: 3e-3, df: 3.6e-4, nFil: 3, nxFil: 16, ny: 24, rej: 0.995 });
      add('Navier–Stokes spacer channel: salt balance closes', 1, sp.balance.out / sp.balance.in, 1e-5, 'Salt in = salt out + salt in the permeate (zigzag filaments, 99.5 % rejection)');
      add('Navier–Stokes spacer channel: water balance closes', 0, sp.massError, 1e-5, '(inflow − outflow − permeate)/inflow');
      const open = [...sp.cwB.filter((_, i) => !sp.blockB[i]), ...sp.cwT.filter((_, i) => !sp.blockT[i])], wm = sum(open) / open.length, cbm = sum(sp.cb) / sp.cb.length;
      const siG = (c) => saturationIndex(concentrateSolution(sw, 1.8 * c, { co2: 'closed' }).eq, 'gypsum');
      add('Navier–Stokes spacer channel: saturation index at the wall ≥ bulk', 1, Math.min(...open) >= 1 - 1e-9 && siG(wm) > siG(cbm) && siG(Math.max(...open)) > siG(wm) ? 1 : 0, 0, `Gypsum SI in 1.8-times concentrated seawater: bulk ${fmt(siG(cbm), 4)}, mean wall ${fmt(siG(wm), 4)}, hot spot ${fmt(siG(Math.max(...open)), 4)}; no wall point lies below the inlet concentration`);
      add('Navier–Stokes spacer channel: filaments create hot spots above the open-slit wall concentration', 1, Math.max(...open) > Math.max(...ns.cwB) ? 1 : 0, 0, `Peak c wall/c inlet ${fmt(Math.max(...open), 4)} with zigzag filaments, ${fmt(Math.max(...ns.cwB), 4)} without`);
    }
    add('Surrogate reproduces the engine at the design point', 0, Math.max(...sur.predict(0.45, 8.1, 25).map((x, i) => Math.abs(x - sur.engine(0.45, 8.1, 25)[i]))), 0.1, 'Kernel regression versus a fresh speciation run');
    return C;
  },
};
const R_GAS = R;
const HELP = {
  roBrine: 'Composition of the concentrate from the element-by-element RO calculation.', roBrinePH: 'pH of that concentrate.', doseMode: 'How acid or alkali is dosed to the feed before concentration.', targetPH: 'The dose is solved from the alkalinity difference at constant total carbon.',
  targetLSI: 'Calcite saturation index to be reached at the membrane wall of the tail element.', base: 'Lime is cheaper but adds calcium.', fixedChem: 'Chemical added at a fixed dose.', fixedDose: 'Mass of pure chemical per litre of feed.',
  mixOn: 'Mixes a second water into the feed and re-equilibrates the blend.', mixIons: 'Complete analysis of the second stream.', mixFrac: 'Volume share of the second stream in the blend.', mixPH: 'Measured pH of the second stream.', mixT: 'The blend temperature is the flow-weighted mean.',
  limGypsum: 'Gypsum and anhydrite; typical 230–400 %.', limBarite: 'Typical 6000–8000 %.', limCelestite: 'Typical 800–1200 %.', limFluorite: 'Typical 12000 %.',
  dkCalcite: 'Positive values make calcite more soluble.', dkGypsum: 'Positive values make gypsum more soluble.', dkBarite: 'Positive values make barite more soluble.', dkSilica: 'Positive values make amorphous silica more soluble.',
  logA: 'Kinetic prefactor of classical nucleation theory; 10³⁰ m⁻³s⁻¹ is the usual estimate for sparingly soluble salts.', nRec: 'More points sharpen the interpolated scaling-limited recovery.', nCF: 'Each point is a full equilibrium-precipitation calculation.',
};
for (const g of suite.inputs) for (const f of g.fields) if (!f.help && HELP[f.key]) f.help = HELP[f.key];

/** Synthetic laboratory data: the model with slightly shifted solubility products plus deterministic noise. */
function synth(seed, pts) {
  const d = D(), g = rng(seed);
  return pts.map(([mNaCl, Tc, pCO2x]) => {
    const m = suite.calibration.model({ ...d, dkGypsum: 0.035, dkCalcite: -0.06, mNaCl, Tc, pCO2x });
    return { mNaCl, Tc, pCO2x, sGyp: +(m.sGyp * (1 + g.normal(0, 0.012))).toFixed(2), sCal: +(m.sCal * (1 + g.normal(0, 0.015))).toFixed(3) };
  });
}

export default suite;
