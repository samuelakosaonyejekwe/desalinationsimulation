// Suite 2 — Brine chemistry, precipitation and scaling.
// Aqueous speciation (carbonate, borate, silicate, sulphate and fluoride acid–base systems, water
// dissociation, ion pairs) solved by mass action with mass and alkalinity balances; activity
// coefficients from Debye–Hückel, extended Debye–Hückel, Davies, Truesdell–Jones or the Pitzer
// ion-interaction model (Harvie–Møller–Weare 25 °C parameter set), Bromley or SIT; temperature- and pressure-dependent
// solubility products; saturation indices; equilibrium precipitation; concentration paths; chemical
// dosing; nucleation and growth kinetics; scaling and corrosion indices.
import { brent, clamp, linspace, logspace, sum, rng, fmt, interp1, tridiag, trapz, gci, solveLinear as solveLin } from '../core/num.js';
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
//  · Ba chloride (Pitzer & Mayorga 1973) and borate (Felmy & Weare 1986): USGS PHREEQC pitzer.dat (PHRQPITZ lineage) and data0.ypf;
//  · strontium: the 25 °C interaction set of the THEREDA database (Scharge 2016, release of 2020-10-22 for PHREEQC): Sr–Cl,
//    Sr–SO4, θ(Sr,Na), θ(Sr,K), θ(Sr,Mg), θ(Sr,Ca) and the four ψ(Sr,M,Cl); it replaces the Sr–SO4 entry of pitzer.dat,
//    which is numerically the Ca–SO4 set. The celestite log K that THEREDA pairs with it (−6.550) is NOT used: with it the
//    model lies 10–15 % above the celestite solubilities of Reardon & Armstrong (1987) in NaCl solutions to 5 mol/kg and
//    22–25 % above those of Culberson et al. (1978) in synthetic seawater; with log K = −6.63 (WATEQ4F, pitzer.dat) the
//    same parameters reproduce both within 4.5 %. Checked against the SrCl2 table of Goldberg & Nuttall (1978);
//  · ammonium, iron(II), manganese(II): NH4–Cl and NH4–SO4 (refits of Thiessen & Simonson 1990 and Clegg et al. 1996),
//    NH4–NO3, NH4–HCO3 and θ(H,NH4) (Pitzer 1991), Fe–Cl (Pitzer & Mayorga 1973), Fe–SO4 (β2 −42), Mn–SO4 (Pitzer &
//    Mayorga 1974): LLNL data0.ypf; Mn–Cl (Pitzer & Mayorga 1973) and Fe–HSO4: USGS PHREEQC pitzer.dat. Checked against
//    the NIST tables of NH4Cl and NH4NO3 (Hamer & Wu 1972), FeCl2 (Goldberg, Nuttall & Staples 1979) and MnCl2 (Goldberg 1979);
//  · fluorides, KNO3, Mg(NO3)2, phosphates and θ(Cl,NO3) (Pitzer 1991 tabulation), NaNO3 and Ca(NO3)2 (refits with
//    α1 = 2): LLNL EQ3/6 Yucca Mountain Pitzer file data0.ypf (25 °C terms);
//  · silica λ: PHREEQC pitzer.dat (Appelo 2015).
// Ba–SO4: no published Ba–SO4 binary could be read — PHRQPITZ (Dal Pozzo 1991, Table 2.2), the current PHREEQC pitzer.dat
// (Appelo 2015) and data0.ypf carry no Ba–SO4 term at all, and the two barite models built on measurements in sulphate
// media treat the interaction as an explicit BaSO4(aq) ion pair (Felmy, Rai & Amonette 1990: log K 2.72 ± 0.09 with
// log Ksp −10.05 ± 0.05; Monnin 1999 as described by Monnin et al. 1999). The treatment is selectable (BARITE_MODELS
// below). The default is the row "Ba SO4" of the table that follows — the Ca–SO4 set, the approximation of Rogers (1981,
// LBL-12356): against the measured barite solubilities in sulphate-bearing media that could be obtained (BARITE_SULPHATE:
// Na2SO4 solutions, seawater, reverse-osmosis feed and concentrate, dilute sulphuric acid; 41 points) it has the smallest deviation of the four
// treatments, and it shares the best fit in water and NaCl solutions (bariteEvidence, verify). In water and
// in NaCl solutions the treatments differ by their solubility products alone; what the choice does to the barite index
// of a sulphate-bearing water is evaluated by bariteBand() below and reported with every result.
// Analogue assignments that remain: for NH4, Fe and Mn only the pairs
// and mixing terms without a value in the files named above are still borrowed (NH4 from K, Fe/Mn from Mg: θ, ψ and
// the binaries with OH, CO3, F, HPO4, borate; HCO3 for Fe/Mn); H3SiO4 borrows HCO3 throughout.
const PZ_ID = { NH4: 'K', Fe: 'Mg', Mn: 'Mg', H3SiO4: 'HCO3' };
const PZ_BIN = 'Na Cl .0765 .2664 0 .00127|Na SO4 .01958 1.113 0 .00497|Na HSO4 .0454 .398 0 0|Na OH .0864 .253 0 .0044|Na HCO3 .0277 .0411 0 0|Na CO3 .0399 1.389 0 .0044|Na NO3 .00357079 .231963 0 -.0000415038|Na F .0215 .2107 0 0|Na B(OH)4 -.0427 .089 0 .0114|Na HPO4 -.0583 1.4655 0 .02938|'
  + 'K Cl .04835 .2122 0 -.00084|K SO4 .04995 .7793 0 0|K HSO4 -.0003 .1735 0 0|K OH .1298 .32 0 .0041|K HCO3 .0296 -.013 0 -.008|K CO3 .1488 1.43 0 -.0015|K NO3 -.0816 .0494 0 .0066|K F .08089 .2021 0 .00093|K B(OH)4 .035 .14 0 0|K HPO4 .0248 1.2743 0 .016387|'
  + 'Ca Cl .3159 1.614 0 -.00034|Ca SO4 .2 3.1973 -54.24 0|Ca HSO4 .2145 2.53 0 0|Ca OH -.1747 -.2303 -5.72 0|Ca HCO3 .4 2.977 0 0|Ca NO3 .14844 2.44408 0 -.0041168|'
  + 'Mg Cl .35235 1.6815 0 .00519|Mg SO4 .221 3.343 -37.23 .025|Mg HSO4 .4746 1.729 0 0|Mg HCO3 .329 .6072 0 0|Mg NO3 .3671 1.5848 0 -.020625|MgOH Cl -.1 1.658 0 0|'
  + 'Sr Cl .282054 1.44967 0 -.000552|Sr SO4 .2 1.30949 -24.31 0|Ba Cl .2628 1.49625 0 -.0193782|Ba SO4 .2 3.1973 -54.24 0|H Cl .1775 .2945 0 .0008|H SO4 .0298 0 0 .0438|H HSO4 .2065 .5556 0 0|'
  + 'NH4 Cl .0524920619 .187339293 0 -.00307437349|NH4 SO4 .0391672883 .662846757 0 -.000757091057|NH4 NO3 -.0154 .112 0 -.00003|NH4 HCO3 -.038 .07 0 0|Fe Cl .3359 1.5323 0 -.00861|Fe SO4 .2568 3.063 -42 .0209|Fe HSO4 .4273 3.48 0 0|Mn Cl .327225 1.55025 0 -.0204972|Mn SO4 .201 2.98 -40 .0182';
const PZ_THETA = 'Na K -.012|Na Ca .07|Na Mg .07|Na H .036|K Ca .032|K H .005|Ca Mg .007|Ca H .092|Mg H .1|Na Sr .101684|K Sr .041523|Mg Sr .0014857|Ca Sr .45946|Cl SO4 .02|Cl HSO4 -.006|Cl OH -.05|Cl HCO3 .03|Cl CO3 -.02|SO4 OH -.013|SO4 HCO3 .01|SO4 CO3 .02|OH CO3 .1|HCO3 CO3 -.04|Cl NO3 .016|H NH4 -.019';
const PZ_PSI = 'Na K Cl -.0018|Na K SO4 -.01|Na K HCO3 -.003|Na K CO3 .003|Na Ca Cl -.007|Na Ca SO4 -.055|Na Mg Cl -.012|Na Mg SO4 -.015|Na H Cl -.004|Na H HSO4 -.0129|K Ca Cl -.025|K Mg Cl -.022|K Mg SO4 -.048|K H Cl -.011|K H SO4 .197|K H HSO4 -.0265|'
  + 'Na Sr Cl -.014575|K Sr Cl -.02144|Mg Sr Cl -.0031398|Ca Sr Cl -.081237|Ca Mg Cl -.012|Ca Mg SO4 .024|Ca H Cl -.015|Mg MgOH Cl .028|Mg H Cl -.011|Mg H HSO4 -.0178|Cl SO4 Na .0014|Cl SO4 Ca -.018|Cl SO4 Mg -.004|Cl HSO4 Na -.006|Cl HSO4 H .013|Cl OH Na -.006|Cl OH K -.006|Cl OH Ca -.025|Cl HCO3 Na -.015|Cl HCO3 Mg -.096|'
  + 'Cl CO3 Na .0085|Cl CO3 K .004|SO4 HSO4 Na -.0094|SO4 HSO4 K -.0677|SO4 HSO4 Mg -.0425|SO4 OH Na -.009|SO4 OH K -.05|SO4 HCO3 Na -.005|SO4 HCO3 Mg -.161|SO4 CO3 Na -.005|SO4 CO3 K -.009|OH CO3 Na -.017|OH CO3 K -.01|HCO3 CO3 Na .002|HCO3 CO3 K .012';
const PZ_LAM = 'CO2 Na .1|CO2 K .051|CO2 Ca .183|CO2 Mg .183|CO2 Cl -.005|CO2 SO4 .097|CO2 HSO4 -.003|B(OH)3 Na -.097|B(OH)3 K -.14|B(OH)3 Cl .091|B(OH)3 SO4 .018|SiO2 Na .0566|SiO2 K .0298|SiO2 Mg .238|SiO2 Ca .238|SiO2 SO4 -.085';
const PZ = (() => {
  const pid = SID.map((id) => PZ_ID[id] || id), rows = (s) => s.split('|').filter(Boolean).map((r) => r.split(' '));
  const idx = (name) => pid.map((p, i) => (p === name || SID[i] === name ? i : -1)).filter((i) => i >= 0); // a row addresses its own ion and the ions that borrow it; rows under an ion's own name come last and overwrite the borrowed values
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

// ---- Ba–SO4 interaction on the Pitzer path: selectable treatment -------------------------------------
// 'pair'     Pitzer coefficients for the free ions, no Ba–SO4 binary, an explicit BaSO4(aq) ion pair with log K 2.72
//            (± 0.09) and barite log Ksp −10.05 (± 0.05): Felmy, Rai & Amonette (1990, J. Solution Chem. 19, 175), the
//            treatment they prefer for BaSO4 on their barite solubilities in Na2SO4 solutions at 25 °C. Only the
//            abstract and the first two pages of that paper were read: the two constants are theirs; the free-ion
//            parameters (Ba–Cl of Pitzer & Mayorga 1973 and the Harvie–Møller–Weare set), an activity coefficient of 1
//            for the neutral pair, a temperature-independent log K and the temperature and pressure function of the
//            barite log Ksp (the WATEQ4F expression shifted to −10.05 at 25 °C) are those of this suite, not statements
//            of the paper. Monnin et al. (1999, Mar. Chem. 65, 253) describe the model of Monnin (1999) as of the same
//            kind (BaSO4(aq), SrSO4(aq) and CaSO4(aq) pairs, all other interactions by Pitzer parameters) but print
//            no constants, so it cannot be offered as a treatment of its own. The solubility data of Felmy et al. were
//            not read either (according to García 2005 they are given as plots only); against the measurements of other
//            laboratories in sulphate media (BARITE_SULPHATE) this treatment lies +0.02 in log10 on average, rms 0.06.
// 'pairw'    the same ion pair with the barite log Ksp −9.965 fitted in this work to the solubility in pure water. Kept
//            selectable as a negative result: in sulphate media it dissolves 26 % too much barium on average (+0.10 in
//            log10, 41 points) — the association constant belongs with the log Ksp it was derived with.
// 'analogue' the Ca–SO4 binary (β⁰ 0.2, β¹ 3.1973, β² −54.24) for Ba–SO4, barite log Ksp −9.97 (WATEQ4F): the
//            approximation of Rogers (1981, LBL-12356).
// 'none'     no Ba–SO4 term, barite log Ksp −9.97: USGS PHREEQC pitzer.dat (Appelo 2015) and LLNL data0.ypf.
// An equilibrium state keeps the treatment it was made with (eq.ba) and hands it on to every solution derived from it.
export const BARITE_MODELS = {
  pair: { label: 'BaSO₄(aq) ion pair, log K 2.72, barite log Ksp −10.05 (Felmy, Rai & Amonette 1990)', b: [0, 0, 0], logK: 2.72, sd: 0.09, logKsp: -10.05, sdKsp: 0.05 },
  analogue: { label: 'Ca–SO₄ analogue binary, barite log Ksp −9.97 (Rogers 1981)', b: [0.2, 3.1973, -54.24] },
  none: { label: 'No Ba–SO₄ term, barite log Ksp −9.97 (USGS pitzer.dat)', b: [0, 0, 0] },
  pairw: { label: 'BaSO₄(aq) ion pair, log K 2.72 (Felmy, Rai & Amonette 1990), barite log Ksp −9.965 fitted in this work to pure-water solubility', b: [0, 0, 0], logK: 2.72, sd: 0.09, logKsp: -9.965, sdKsp: 0.01, fitted: true },
};
export const BARITE_MODEL_DEFAULT = 'analogue';
let BARITE_MODEL = BARITE_MODEL_DEFAULT;
/** Treatment of the Ba–SO4 interaction used by default for new solutions (Pitzer model). */
export const bariteModel = () => BARITE_MODEL;
/** Select the default treatment (a key of BARITE_MODELS); returns the previous one. */
export function setBariteModel(id) { const prev = BARITE_MODEL; if (BARITE_MODELS[id]) BARITE_MODEL = id; return prev; }
const isPair = (ba) => BARITE_MODELS[ba]?.logK != null; // treatments with an explicit BaSO4(aq) species
const KBASO4 = si('Ba') * NS + si('SO4');
/** Pitzer parameter view with a given Ba–SO4 binary (all other tables shared). */
const pzWith = ([b0, b1, b2]) => { const c = (A, x) => { const B = Float64Array.from(A); B[KBASO4] = x; return B; }; return { ...PZ, B0: c(PZ.B0, b0), B1: c(PZ.B1, b1), B2: c(PZ.B2, b2) }; };
const PZ_BA = Object.fromEntries(Object.entries(BARITE_MODELS).map(([id, t]) => [id, pzWith(t.b)]));

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
function pitzer(m, T, lnG, par = PZ) {
  const { B0, B1, B2, CM, TH, LAM, PSI, HASPSI, HASB, cat, an, neu, pc, pa } = par;
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
// ε(Sr²⁺, Cl⁻) is in neither table: 0.10 is a least-squares fit made here to the SrCl2 activity coefficients of Goldberg &
// Nuttall (1978) up to I = 3 mol/kg (rms 0.004 in log γ±); the same fit returns 0.135 for CaCl2, 0.198 for MgCl2 and 0.057 for
// BaCl2 against the NEA values 0.14, 0.19 and 0.07. Unlisted pairs use ε = 0.
// Bromley constants: the salt constants B and the individual-ion table (B = B₊ + B₋ + δ₊δ₋, used for salts without a
// fitted constant) are those of Bromley (1973, AIChE J. 19, 313–320) as reproduced in Appendix 4.2 of Zemaitis, Clark,
// Rafal & Scrivner, Handbook of Aqueous Electrolyte Thermodynamics (DIPPR/AIChE 1986, pp. 170–174); the 1973 article
// itself was not read. The 19 salt constants and the 23 ion pairs (B, δ) below were compared with that reprint; with a second
// reproduction (BR_THOMSEN below: ten salt constants, nine ions) and a third (BR_REPRO3: all 19 salt constants and, through its
// ion-table sums, the ions of this suite). Independently of all three, every salt constant is re-derived by Bromley's method from
// γ± data read first-hand (Hamer & Wu 1972; Goldberg & Nuttall 1978; Goldberg 1981; Rard, Wijesinghe & Wolery 2004 — BR_REFIT,
// BR_REFIT2, fitBromleyB), and a published constant is kept only if it reproduces those data within the scatter of the refit:
// 15 are kept, 4 are replaced by the re-derived value (BR_SWITCH: SrCl2, BaCl2, Na2CO3, K2CO3). One entry of the reprint is not
// taken as printed: it gives −0.0108 for RbI; the complete table of Hamer & Wu returns +0.0108 with the printed σ, and the third
// reproduction has +0.0108 (RbI enters only the comparison of the published and refitted tables, not a calculation of the suite).
// Both models return the osmotic coefficient that satisfies the Gibbs–Duhem equation with their activity coefficients.
const PFAM = { pitzer: 1, bromley: 1 }; // model families that use the strong-electrolyte species set (no sulphate ion pairs)
const BR_ION = { H: [0.0875, 0.103], Na: [0, 0.028], K: [-0.0452, -0.079], NH4: [-0.042, -0.02], Mg: [0.057, 0.157], Ca: [0.0374, 0.119], Sr: [0.0245, 0.11], Ba: [0.0022, 0.098], Mn: [0.037, 0.21], Fe: [0.046, 0.21], F: [0.0295, -0.93], Cl: [0.0643, -0.067], NO3: [-0.025, 0.27], OH: [0.076, -1], SO4: [0, -0.4], CO3: [0.028, -0.67], HPO4: [-0.01, -0.57] };
const BR_PUB = { Li: [0.0691, 0.138], Rb: [-0.0537, -0.1], Cs: [-0.071, -0.138], Br: [0.0741, 0.064], I: [0.089, 0.196], ClO4: [0.002, 0.79] }; // Bromley's values for ions outside the species list, used only to compare the two ion tables
// A second reproduction of Bromley's tables (it cites the 1973 article as its source; whether it was set from the article or
// from the handbook cannot be told): Thomsen (2009, "Electrolyte Solutions: Thermodynamics, Crystallization, Separation
// methods", lecture notes, Technical University of Denmark; open, pp. 49–50 read): Table 6.2, 18 salt constants, and Table 6.3, (B, δ) of
// six cations and four anions. reprint: the handbook-reprint values of the six salts of that table that are not salt constants of this suite.
const BR_THOMSEN = { salt: 'H Cl .1433|H NO3 .0776|K Br .0296|K Cl .024|K NO3 -.0862|Na Br .0749|Na Cl .0574|Na NO3 -.0128|NH4 Cl .02|H SO4 .0606|K CO3 .0372|K SO4 -.032|Na CO3 .0089|Na SO4 -.0204|Ca Cl .0948|Mg NO3 .1014|Mg SO4 -.0153|Al SO4 -.0044',
  ion: { H: [0.0875, 0.103], Na: [0, 0.028], K: [-0.0452, -0.079], NH4: [-0.042, -0.02], Ca: [0.0374, 0.119], Al: [0.052, 0.12], Cl: [0.0643, -0.067], NO3: [-0.025, 0.27], SO4: [0, -0.4], CO3: [0.028, -0.67] }, reprint: { 'Na Br': 0.0749, 'H SO4': 0.0606, 'K CO3': 0.0372, 'Na CO3': 0.0089, 'Mg NO3': 0.1014, 'Al SO4': -0.0044 } };
// BR_SALT: Bromley's table values; the constants in use are these except where BR_SWITCH holds a re-derived value.
const BR_SALT = 'Na Cl .0574|K Cl .024|H Cl .1433|Ca Cl .0948|Mg Cl .1129|Sr Cl .0847|Ba Cl .0638|NH4 Cl .02|Na SO4 -.0204|K SO4 -.032|Mg SO4 -.0153|Na NO3 -.0128|K NO3 -.0862|Na OH .0747|K OH .1131|NH4 NO3 -.0358|Mg NO3 .1014|Na CO3 .0089|K CO3 .0372';
const BR_SWITCH = { 'Sr Cl': 0.0809, 'Ba Cl': 0.0606, 'Na CO3': -0.0073, 'K CO3': 0.0304 }; // re-derived constants in use instead of the published ones (rule and data: BR_DATA2, BR_REFIT2 below)
const SIT_EPS = 'Na Cl .03|K Cl 0|H Cl .12|NH4 Cl -.01|NH4 NO3 -.06|Ca Cl .14|Sr Cl .1|Mg Cl .19|Ba Cl .07|Na SO4 -.12|K SO4 -.06|Na HSO4 -.01|Na NO3 -.04|K NO3 -.11|H NO3 .07|Ca NO3 .02|Mg NO3 .17|Na OH .04|K OH .09|Na HCO3 0|K HCO3 -.06|Na CO3 -.08|K CO3 .02|Na F .02|K F .03|Na B(OH)4 -.07|Na HPO4 -.15|K HPO4 -.1|Fe Cl .17|Mn Cl .13|Na H3SiO4 -.08';
const PAIRPAR = (() => {
  const BR = new Float64Array(NS * NS), EPS = new Float64Array(NS * NS), rows = (t) => t.split('|').map((r) => r.split(' '));
  for (const c of PZ.cat) for (const a of PZ.an) { const p = BR_ION[SID[c]] || [0, 0], q = BR_ION[SID[a]] || [0, 0]; BR[c * NS + a] = p[0] + q[0] + p[1] * q[1]; }
  for (const [c, a, b] of rows(BR_SALT)) BR[si(c) * NS + si(a)] = BR_SWITCH[c + ' ' + a] ?? +b;
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

// Independent refit of the Bromley constants from the NIST evaluations (this work). BR_REFIT: per salt "cation anion B rms n Imax
// [B published, from the handbook reprint; RbI with the corrected sign]" — B by fitBromleyB on the tabulated γ± (Hamer & Wu 1972 for 1:1 salts; Goldberg & Nuttall 1978 for the
// alkaline-earth halides; Goldberg 1981 for the sulphates), rms = weighted root-mean-square deviation in log₁₀ γ±, n points to
// the ionic strength Imax ≤ 6 mol/kg. BR_DATA: the rows [m γ± m γ± …] behind the suite's own salts, so that the regression can
// be repeated. BR_ION_REFIT: [B, δ] per ion from fitBromleyIons over the 52 refitted salts whose ions occur in at least three
// of them (anchors B(Na⁺) = 0, δ(OH⁻) = −1 as in Bromley's table, δ(Na⁺) = 0.028 and δ(Cl⁻) = −0.067 taken from it).
const BR_REFIT = 'H Cl .1434 .0022 29 6 .1433|H Br .1807 .0137 24 5.5 .1734|H I .2037 .0057 26 6 .2054|H ClO4 .1651 .0221 17 6 .1639|H NO3 .0770 .0210 25 6 .0776|Li Cl .1279 .0095 25 6 .1283|Li Br .1547 .0207 13 6 .1527|Li I .1802 .0058 20 3 .1815|Li OH -.0061 .0196 23 4 -.0097|Li ClO4 .1682 .0069 20 4.5 .1702|Li NO3 .0934 .0155 27 6 .0938|Na F .0042 .0004 13 0.9 .0041|Na Cl .0572 .0029 29 6 .0574|Na OH .0744 .0097 28 6 .0747|Na ClO3 .0105 .0035 17 3 .0127|Na ClO4 .0328 .0058 23 5.5 .0330|Na BrO3 -.0271 .0021 20 2.5 -.0278|Na NO3 -.0130 .0023 27 6 -.0128|Na H2PO4 -.0462 .0320 26 6 -.0460|K F .0564 .0067 27 6 .0565|K Cl .0236 .0005 22 4.5 .0240|K Br .0294 .0027 27 5.5 .0296|K I .0419 .0078 26 4.5 .0428|K OH .1136 .0090 23 6 .1131|K ClO3 -.0725 .0025 13 0.7 -.0739|K BrO3 -.0886 .0021 11 0.5 -.0884|K NO3 -.0855 .0112 21 3.5 -.0862|K H2PO4 -.1121 .0068 12 1.8|Rb F .0683 .0109 24 3.5 .0650|Rb Br .0110 .0035 18 5 .0111|Rb I .0100 .0048 12 4 .0108|Rb NO3 -.0863 .0213 26 4.5 -.0869|Cs F .0936 .0034 21 3 .0906|Cs Cl .0009 .0130 21 5 .0025|Cs Br -.0030 .0137 25 5 -.0039|Cs I -.0189 .0057 17 2.5 -.0188|Cs OH .1280 .0030 14 1.2 .1299|Cs NO3 -.1174 .0041 16 1.4 -.1173|NH4 Cl .0200 .0043 24 6 .0200|NH4 ClO4 -.0621 .0096 15 1.8 -.0640|NH4 NO3 -.0351 .0111 24 6 -.0358|Mg Cl .1129 .0050 10 4.5 .1129|Mg Br .1388 .0075 9 3.75 .1419|Mg I .1596 .0078 20 3 .1695|Ca Cl .0930 .0053 22 5.25 .0948|Ca I .1433 .0051 19 5.7 .1440|Sr Cl .0809 .0052 17 6 .0847|Sr Br .1052 .0036 25 5.7 .1038|Sr I .1351 .0038 33 5.7 .1339|Ba Br .0840 .0034 32 6 .0852|Ba I .1286 .0057 30 5.99 .1254|Ba Cl .0606 .0038 21 5.35 .0638|Ca Br .1190 .0047 17 6 .1179|Na SO4 -.0206 .0088 17 6 -.0204|K SO4 -.0420 .0103 10 2.08 -.0320|Li SO4 .0204 .0036 17 6 .0207|Rb SO4 -.0138 .0106 16 5.12 -.0091|Cs SO4 -.0055 .0100 16 4.89 -.0012';
const BR_DATA = {
  'Na Cl': '0.001 .965 0.002 .952 0.005 .928 0.01 .903 0.02 .872 0.05 .822 0.1 .779 0.2 .734 0.3 .709 0.4 .693 0.5 .681 0.6 .673 0.7 .667 0.8 .662 0.9 .659 1 .657 1.2 .655 1.4 .656 1.6 .658 1.8 .662 2 .668 2.5 .688 3 .714 3.5 .746 4 .783 4.5 .826 5 .874 5.5 .928 6 .986',
  'K Cl': '0.001 .965 0.002 .951 0.005 .927 0.01 .901 0.02 .869 0.05 .816 0.3 .687 0.4 .665 0.5 .649 0.6 .636 0.7 .626 0.8 .617 0.9 .610 1 .604 1.6 .580 1.8 .576 2 .573 2.5 .568 3 .568 3.5 .571 4 .576 4.5 .584',
  'H Cl': '0.001 .965 0.002 .952 0.005 .929 0.01 .905 0.02 .876 0.05 .832 0.1 .797 0.2 .768 0.3 .758 0.4 .756 0.5 .759 0.6 .765 0.7 .774 0.8 .785 0.9 .797 1 .811 1.2 .842 1.4 .877 1.6 .917 1.8 .961 2 1.009 2.5 1.148 3 1.316 3.5 1.517 4 1.757 4.5 2.040 5 2.380 5.5 2.770 6 3.230',
  'NH4 Cl': '0.001 .965 0.002 .952 0.005 .927 0.02 .869 0.05 .816 0.1 .769 0.2 .718 0.4 .666 0.5 .649 0.6 .636 0.7 .626 0.8 .617 1 .603 1.2 .592 1.4 .584 1.6 .578 2 .569 2.5 .563 3 .560 3.5 .560 4 .560 5 .563 5.5 .564 6 .565',
  'Na NO3': '0.001 .965 0.002 .951 0.005 .926 0.01 .900 0.02 .867 0.05 .811 0.1 .760 0.2 .702 0.3 .666 0.5 .618 0.6 .600 0.7 .585 0.8 .571 0.9 .559 1 .549 1.2 .530 1.4 .515 1.6 .501 1.8 .489 2 .478 3 .437 3.5 .422 4 .408 4.5 .396 5 .386 5.5 .378 6 .372',
  'K NO3': '0.001 .964 0.002 .950 0.005 .924 0.01 .896 0.05 .797 0.1 .735 0.2 .662 0.3 .614 0.4 .577 0.5 .546 0.6 .521 0.8 .478 0.9 .460 1 .444 1.2 .415 1.4 .390 1.6 .369 1.8 .350 2.5 .297 3 .269 3.5 .246',
  'Na OH': '0.001 .965 0.002 .952 0.005 .927 0.01 .902 0.02 .870 0.05 .819 0.1 .775 0.2 .731 0.3 .708 0.4 .694 0.5 .685 0.6 .679 0.8 .674 0.9 .673 1 .674 1.2 .678 1.4 .684 1.6 .692 1.8 .702 2 .714 2.5 .749 3 .794 3.5 .847 4 .911 4.5 .987 5 1.076 5.5 1.181 6 1.302',
  'K OH': '0.001 .965 0.002 .952 0.005 .927 0.01 .902 0.05 .821 0.2 .740 0.4 .713 0.5 .710 0.6 .711 0.7 .714 0.8 .718 0.9 .725 1 .733 1.2 .752 1.4 .774 1.6 .800 2 .860 3 1.058 4 1.331 4.5 1.501 5 1.697 5.5 1.923 6 2.180',
  'Na F': '0.001 .965 0.002 .951 0.005 .926 0.01 .901 0.05 .813 0.1 .764 0.2 .710 0.3 .676 0.4 .652 0.6 .617 0.7 .604 0.8 .592 0.9 .582',
  'K F': '0.001 .965 0.002 .952 0.005 .927 0.01 .902 0.02 .870 0.05 .818 0.1 .773 0.2 .726 0.3 .700 0.4 .682 0.5 .670 0.6 .662 0.8 .650 0.9 .647 1 .645 1.2 .643 1.4 .644 1.6 .647 1.8 .651 2 .658 2.5 .678 3 .705 3.5 .738 4 .777 4.5 .821 5.5 .927 6 .989',
  'Mg Cl': '0.001 .8893 0.002 .8522 0.003 .8265 0.008 .7525 0.009 .7426 0.07 .5619 0.08 .5513 0.4 .4796 1.25 .6561 1.5 .7622',
  'Ca Cl': '0.001 .8885 0.002 .8508 0.003 .8245 0.004 .8039 0.005 .7869 0.006 .7724 0.007 .7596 0.008 .7483 0.009 .7380 0.01 .7287 0.02 .6644 0.04 .5982 0.05 .5773 0.08 .5355 0.09 .5256 0.2 .4692 0.4 .4442 0.601 .4486 0.701 .4564 0.9 .4801 1.5 .6070 1.75 .6861',
  'Sr Cl': '0.01 .7255 0.02 .6595 0.05 .5694 0.1 .5063 0.2 .4550 0.3 .4336 0.4 .4241 0.5 .4209 0.6 .4219 0.7 .4260 0.8 .4324 0.9 .4409 1 .4513 1.25 .4846 1.5 .5283 1.75 .5829 2 .6496 2.25 .7303 2.5 .8270 2.75 .9427 3 1.0808',
  'Ba Cl': '0.01 .7214 0.02 .6532 0.05 .5591 0.1 .4924 0.2 .4365 0.3 .4115 0.4 .3983 0.5 .3911 0.6 .3877 0.7 .3867 0.8 .3875 0.9 .3897 1 .3929 1.1 .3970 1.2 .4019 1.3 .4074 1.4 .4135 1.5 .4202 1.6 .4273 1.7 .4349 1.785 .4417',
  'Na SO4': '0.01 .7117 0.02 .6369 0.05 .5289 0.1 .4457 0.2 .3656 0.3 .3212 0.4 .2910 0.5 .2684 0.6 .2506 0.7 .2359 0.8 .2236 0.9 .2131 1 .2040 1.25 .1859 1.5 .1725 1.75 .1623 2 .1546',
  'K SO4': '0.01 .7029 0.02 .6251 0.05 .5109 0.1 .4239 0.2 .3429 0.3 .3000 0.4 .2719 0.5 .2514 0.6 .2355 0.692 .2237',
};
const BR_ION_REFIT = { c: { H: [.0888, .118], Li: [.0727, .163], Na: [.0000, .028], K: [-.0512, -.066], Rb: [-.0620, -.091], Cs: [-.0790, -.127], NH4: [-.0338, -.049], Mg: [.0556, .087], Ca: [.0363, .103], Sr: [.0242, .122], Ba: [.0056, .185] }, a: { Cl: [.0639, -.067], Br: [.0796, .025], I: [.0901, .166], ClO4: [.0067, .577], NO3: [-.0141, .148], OH: [.0914, -1.000], F: [.0345, -1.081], SO4: [.0040, -.412] } };
// A third reproduction of Bromley's tables, in open-source code: the matrix BromleyData of the Modelica library ElectrolyteMedia
// (A. M. Bremen & A. Mitsos, RWTH Aachen; github.com/andreasbremen/electrolytemedia, file ElectrolyteMedia/Media/LiquidPhase/Common/
// MixtureSolutesData/package.mo, commit eaaf9ad of 2022-04-22, BSD 3-Clause; "based on Bromley, 1973"). It holds one B per cation–anion
// pair of 36 cations and 38 anions: the fitted salt constant where Bromley's Table 1 has one, otherwise B₊ + B₋ + δ₊δ₋ from his Table 2,
// so it reproduces both tables. BR_REPRO3.rows: its entries for the cations of this suite against an = F, Cl, NO₃, OH, SO₄, CO₃, HPO₄;
// RbI and BeCl₂ (Be²⁺: B 0.1, δ 0.2 in the handbook reprint) are the two further entries used in the checks.
const BR_REPRO3 = { an: ['F', 'Cl', 'NO3', 'OH', 'SO4', 'CO3', 'HPO4'], RbI: 0.0108, BeCl: 0.1509, Be: [0.1, 0.2],
  rows: { H: '.02121 .1433 .0776 .0605 .0606 .04649 .01879', Na: '.0041 .0574 -.0128 .0747 -.0204 .0089 -.0265', K: '.0565 .024 -.0862 .1131 -.032 .0372 -.0096', NH4: '.0061 .02 -.0358 .054 -.0287 -.0006 -.0406', Mg: '-.05951 .1129 .1014 -.024 -.0153 -.02019 -.04249',
    Ca: '-.04377 .0948 .041 -.0056 -.0102 -.01433 -.04043', Sr: '-.0483 .0847 .0138 -.0095 -.0195 -.0212 -.0482', Ba: '-.05944 .0638 -.0545 -.024 -.037 -.03546 -.06366', Mn: '-.1288 .0869 .0687 -.097 -.047 -.0757 -.0927', Fe: '-.1198 .0961 .0777 -.088 -.038 -.0667 -.0837' } };
// Refits for the salt constants of BR_SALT that BR_REFIT does not hold, and RbI from the complete table (BR_REFIT has it from 12 of
// the 27 rows). BR_DATA2: γ± rows [m γ± …] read from page images — RbI: Hamer & Wu (1972), Table 42 (p. 1079), all 27 rows;
// Na₂CO₃: Goldberg (1981), recommended values (p. 715) to 2 mol/kg; Mg(NO₃)₂: Rard, Wijesinghe & Wolery (2004, review of the
// thermodynamic properties of Mg(NO₃)₂(aq), Lawrence Livermore report UCRL-JRNL-203290 — the open report version of J. Chem. Eng. Data
// 49, 1127), Table 3, smoothed values to 2 mol/kg. MgSO₄ is fitted on the γ± of the Archer & Rard (1998) model as tabulated by
// Miladinović et al. (MGSO4_REF.ar, I = 1–6 mol/kg); K₂CO₃, for which Goldberg (1981) gives no table (the data were judged too
// imprecise), on γ± computed with the Harvie–Møller–Weare parameters of this suite — a parametrisation, not a table of primary values.
// NH₄NO₃: Hamer & Wu (1972), Table 57 (p. 1086), to 6 mol/kg. BR_REFIT2: "cation anion B rms n Imax B-published", as BR_REFIT.
// BR_SWITCH: the constants in use that are NOT Bromley's. Rule (checked in verify()): a published constant stays in use if it
// reproduces the independent γ± data with at most twice the standard deviation of the best one-constant fit, the latter taken as
// at least 0.005 in log γ± (1.2 % in γ±, the accuracy of the evaluated tables); otherwise the refitted constant replaces it.
const BR_DATA2 = {
  'Rb I': '.001 .965 .002 .951 .005 .926 .01 .900 .02 .866 .05 .810 .1 .759 .2 .703 .3 .670 .4 .646 .5 .627 .6 .613 .7 .601 .8 .591 .9 .582 1 .574 1.2 .562 1.4 .552 1.6 .544 1.8 .537 2 .532 2.5 .523 3 .517 3.5 .515 4 .514 4.5 .515 5 .517',
  'Na CO3': '.01 .7165 .02 .6445 .05 .5411 .1 .4619 .2 .3855 .3 .3429 .4 .3137 .5 .2918 .6 .2744 .7 .2602 .8 .2482 .9 .2379 1 .2290 1.25 .2115 1.5 .1986 1.75 .1892 2 .1823',
  'NH4 NO3': '.001 .964 .002 .951 .005 .925 .01 .897 .02 .862 .05 .801 .1 .744 .2 .678 .3 .637 .4 .606 .5 .582 .6 .561 .7 .544 .8 .528 .9 .515 1 .502 1.4 .462 1.6 .446 1.8 .432 2 .419 2.5 .391 3 .368 3.5 .349 4 .332 4.5 .316 5 .303 5.5 .291 6 .280',
  'Mg NO3': '.01 .7235 .02 .6577 .05 .5704 .1 .5131 .2 .4720 .3 .4593 .4 .4576 .5 .4618 .6 .4700 .7 .4812 .8 .4949 .9 .5109 1 .5291 1.2 .5715 1.4 .6222 1.6 .6815 1.8 .7502 2 .8292',
};
const BR_REFIT2 = 'Rb I .0108 .0050 27 5 .0108|Na CO3 -.0073 .0031 17 6 .0089|Mg NO3 .0998 .0038 18 6 .1014|Mg SO4 -.0094 .0378 11 6 -.0153|K CO3 .0304 .0032 17 6 .0372';
/**
 * Bromley salt constant B of one electrolyte by weighted least squares on log₁₀ γ± (the equation is linear in B).
 * rows = [[molality, γ±], …]; points up to the ionic strength Imax are used, each weighted by the ionic-strength
 * interval it represents (trapezoid rule), so the result does not depend on how densely a table is printed. With `B` given
 * the constant is not fitted and the deviations of that value are returned.
 */
export function fitBromleyB(zc, za, rows, { Imax = 6, T = 25, B: fixed = null } = {}) {
  const nc = za === zc ? 1 : za, na = za === zc ? 1 : zc, If = 0.5 * (nc * zc * zc + na * za * za), zz = zc * za, A = (3 * aphi(T)) / LN10, d = rows.filter((r) => If * r[0] <= Imax + 1e-9).sort((p, q) => p[0] - q[0]);
  let sxx = 0, sxy = 0, sw = 0, ss = 0, max = 0;
  const pts = d.map((r, i) => { const I = If * r[0], sq = Math.sqrt(I), q = (1 + (1.5 * I) / zz) ** 2; return { w: 0.5 * If * (d[Math.min(i + 1, d.length - 1)][0] - d[Math.max(i - 1, 0)][0]), x: (0.6 * zz * I) / q + I, y: Math.log10(r[1]) + (A * zz * sq) / (1 + sq) - (0.06 * zz * I) / q }; });
  for (const q of pts) { sxx += q.w * q.x * q.x; sxy += q.w * q.x * q.y; sw += q.w; }
  const B = fixed ?? (sxx > 0 ? sxy / sxx : 0);
  for (const q of pts) { const e = q.y - B * q.x; ss += q.w * e * e; max = Math.max(max, Math.abs(e)); }
  return { B, rms: sw > 0 ? Math.sqrt(ss / sw) : 0, max, n: pts.length, Imax: d.length ? If * d[d.length - 1][0] : 0 };
}
/**
 * Individual-ion values of the Bromley model, B(salt) = B₊ + B₋ + δ₊·δ₋, by alternating linear least squares over a
 * list of salts [[cation, anion, B], …]. The bilinear form has four free constants (a shift between B₊ and B₋, a scale
 * between δ₊ and δ₋ and a shift of each δ set), which `fix` removes: { Bc: {Na: 0}, dc: {Na: 0.028}, da: {OH: −1, Cl: −0.067} }.
 */
export function fitBromleyIons(salts, fix = { Bc: { Na: 0 }, dc: { Na: 0.028 }, da: { OH: -1, Cl: -0.067 } }, iters = 400) {
  const cats = [...new Set(salts.map((q) => q[0]))], ans = [...new Set(salts.map((q) => q[1]))], F = { Bc: fix.Bc || {}, dc: fix.dc || {}, Ba: fix.Ba || {}, da: fix.da || {} };
  const V = { Bc: Object.fromEntries(cats.map((c) => [c, F.Bc[c] ?? 0])), dc: Object.fromEntries(cats.map((c) => [c, F.dc[c] ?? 0.05])), Ba: Object.fromEntries(ans.map((a) => [a, F.Ba[a] ?? 0])), da: Object.fromEntries(ans.map((a) => [a, F.da[a] ?? 0])) };
  const model = (c, a) => V.Bc[c] + V.Ba[a] + V.dc[c] * V.da[a];
  const pass = (side) => { // side 0: B₊, B₋ and δ₊ with δ₋ held; side 1: B₊, B₋ and δ₋ with δ₊ held
    const vars = [], idx = {}, add = (t, k) => { if (!(k in F[t])) { idx[t + ' ' + k] = vars.length; vars.push([t, k]); } };
    cats.forEach((c) => add('Bc', c)); ans.forEach((a) => add('Ba', a));
    if (side === 0) cats.forEach((c) => add('dc', c)); else ans.forEach((a) => add('da', a));
    const nv = vars.length, N = Array.from({ length: nv }, () => new Array(nv).fill(0)), rhs = new Array(nv).fill(0);
    for (const [c, a, B] of salts) {
      const row = [], set = (t, k, coef) => { const j = idx[t + ' ' + k]; if (j != null) row.push([j, coef]); return j != null; };
      let y = B;
      if (!set('Bc', c, 1)) y -= V.Bc[c];
      if (!set('Ba', a, 1)) y -= V.Ba[a];
      if (!(side === 0 ? set('dc', c, V.da[a]) : set('da', a, V.dc[c]))) y -= V.dc[c] * V.da[a];
      for (const [j, cj] of row) { rhs[j] += cj * y; for (const [k, ck] of row) N[j][k] += cj * ck; }
    }
    for (let j = 0; j < nv; j++) N[j][j] = N[j][j] * (1 + 1e-10) + 1e-12;
    const x = solveLin(N, rhs);
    vars.forEach(([t, k], j) => { V[t][k] = x[j]; });
  };
  let prev = Infinity, rms = 0, it = 0;
  for (; it < iters; it++) { pass(it % 2); rms = Math.sqrt(sum(salts.map(([c, a, B]) => (model(c, a) - B) ** 2)) / salts.length); if (it > 4 && Math.abs(prev - rms) < 1e-12) break; prev = rms; }
  const res = salts.map(([c, a, B]) => [c, a, model(c, a) - B]);
  return { ...V, rms, max: Math.max(...res.map((r) => Math.abs(r[2]))), res, iterations: it, n: salts.length, nPar: 2 * (cats.length + ans.length) - Object.values(F).reduce((q, o) => q + Object.keys(o).length, 0) };
}

// species lists used in the inner loops: derived species active per model family, and alkalinity carriers
const ACTIVE = [6, 7].map((f) => Int32Array.from(DER.map((d, j) => (d[f] ? j : -1)).filter((j) => j >= 0)));
const ALKI = Int32Array.from(ALK.map((a, s) => (a !== 0 ? s : -1)).filter((s) => s >= 0)), ALKV = Float64Array.from(ALKI, (s) => ALK[s]);
const JBASO4 = DER.findIndex((d) => d[0] === 'BaSO4°'), ACT_PAIR = Int32Array.from([...ACTIVE[0], JBASO4].sort((a, b) => a - b)); // Pitzer species list with the BaSO4(aq) pair
const KCACHE = new Map();
function kset(T, model, pair = false, pk = 0) {
  const fam = PFAM[model] ? 5 + 1 : 7, key = fam + (pair ? 'p' + pk + '|' : '|') + T;
  let k = KCACHE.get(key);
  if (!k) { if (KCACHE.size > 400) KCACHE.clear(); k = DER.map((d) => (d[fam] ? d[fam](T) : NaN)); if (pair) k[JBASO4] = BARITE_MODELS.pair.logK + pk; KCACHE.set(key, k); }
  return k;
}

/** Aqueous equilibrium state at temperature T for one activity model. */
class Eq {
constructor(T, model = 'pitzer', ba = BARITE_MODEL, pk = 0) {
    this.T = T; this.model = ACTIVITY_MODELS[model] ? model : 'pitzer'; this.ba = BARITE_MODELS[ba] ? ba : BARITE_MODEL; this.pair = this.model === 'pitzer' && isPair(this.ba); this.pz = PZ_BA[this.ba];
    this.pk = this.pair && Number.isFinite(pk) ? pk : 0; // offset of log K of BaSO4(aq) fitted to a user's own measurements (calibration); 0 = published constant
    this.K = kset(T, this.model, this.pair, this.pk); this.kH = 10 ** logKH(T); this.act = this.pair ? ACT_PAIR : ACTIVE[PFAM[this.model] ? 0 : 1];
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
    const r = this.model === 'pitzer' ? pitzer(this.m, this.T, this._l, this.pz) : this.model === 'bromley' || this.model === 'sit' ? pairModel(this.model, this.m, this.T, this._l) : debye(this.model, this.m, this.T, this._l);
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
const DK_PAIR = BARITE_MODELS.pair.logKsp - MINERALS.barite.logK(25); // shift of the barite log Ksp that belongs to the ion-pair treatment (−10.05 against −9.970)
const dkPair = (ba) => (isPair(ba) ? BARITE_MODELS[ba].logKsp - MINERALS.barite.logK(25) : 0); // the same for every pair treatment: its own log Ksp at 25 °C against −9.970
const logKfor = (M, T, model, ba = BARITE_MODEL) => (PFAM[model] ? M.kP || M.logK : M.kI || M.logK)(T) + (M === MINERALS.barite && model === 'pitzer' ? dkPair(ba) : 0);
for (const [id, M] of Object.entries(MINERALS)) {
  M.id = id; M.logKfor = (T, model = 'pitzer', ba = BARITE_MODEL) => logKfor(M, T, model, ba); M.stoichiometry = { ...M.stoich, ...(M.nOH ? { OH: M.nOH } : {}), ...(M.nW ? { H2O: M.nW } : {}) };
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
  return s - (logKfor(M, eq.T, eq.model, eq.ba) + dP + dk);
}
const present = (eq, id) => MINERALS[id]._st.every(([i]) => eq.tot[i] > 0);
/** Saturation indices of every mineral whose components are present. */
export function saturation(eq, P = 1, dk = {}, ids = Object.keys(MINERALS)) {
  const out = {};
  for (const id of ids) if (present(eq, id)) out[id] = saturationIndex(eq, id, P, dk[id] || 0);
  return out;
}

// ---- barite index: sensitivity to the Ba–SO4 interaction ---------------------------------------------
// The default treatment of the Ba–SO4 interaction on the Pitzer path is the Ca–SO4 analogue with log Ksp −9.97
// (BARITE_MODELS.analogue): of the published treatments it reproduces the measured solubility in water, under pressure
// and at temperature most closely, and it has the smallest deviation from the 41 counted solubilities in sulphate-bearing
// media held in this file (mean −0.02, rms 0.05 in log10 of model/measured; ion pair of Felmy et al. +0.02 and 0.06, no
// Ba–SO4 term −0.05 and 0.07: BARITE_SULPHATE, bariteEvidence). The verdicts do not rest
// on that choice alone: they use the highest index among the published treatments (bariteVerdict). The treatments found
// in the literature are listed here with the barite log Ksp each is used with, and the barite index is recomputed under every one of them:
//  · explicit BaSO4(aq) ion pair, log K 2.72 ± 0.09, with log Ksp −10.05 ± 0.05 — Felmy, Rai & Amonette (1990, J. Solution
//    Chem. 19, 175; abstract and first two pages read), from their own barite solubilities in Na2SO4 solutions at 25 °C;
//    the model of Monnin (1999) is described as of the same kind (Monnin et al. 1999, Mar. Chem. 65, 253; no constants given);
//  · Ca–SO4 analogue, log Ksp −9.97 — Rogers (1981, LBL-12356); the default of this suite;
//  · Sr–SO4 analogue, log Ksp −9.97 — the Sr–SO4 set of THEREDA (Scharge 2016) in place of the Ca–SO4 one;
//  · no Ba–SO4 term, log Ksp −9.97 — USGS PHREEQC pitzer.dat (Appelo 2015) and LLNL data0.ypf.
// Barium is a trace ion, so the change of the index is closed-form: a binary changes ln γ(Ba) by 2·m(SO4)·B(I), an ion
// pair lowers the free barium by the factor 1 + K·γ(Ba)·γ(SO4)·m(SO4) (free-ion coefficients without a Ba–SO4 term,
// γ of the neutral pair = 1). Measured barite solubilities in sulphate media from other laboratories (BARITE_SULPHATE)
// are compared with every treatment in bariteEvidence(), in verify() and in a results table; the data of Felmy et al.
// themselves were not read (closed paper; given as plots only according to García 2005); twelve points that Paige (1990)
// digitised from one of their graphs are listed but not counted (they contradict the published constants).
export const BASO4_TREATMENTS = [
  { id: 'pair', model: 'pair', label: 'BaSO₄(aq) ion pair, log K 2.72 ± 0.09, with log Ksp −10.05 (Felmy et al. 1990)', logK: BARITE_MODELS.pair.logK, sd: BARITE_MODELS.pair.sd, logKsp: BARITE_MODELS.pair.logKsp },
  { id: 'ca', model: 'analogue', label: 'Ca–SO₄ analogue (Rogers 1981)', b: BARITE_MODELS.analogue.b },
  { id: 'sr', label: 'Sr–SO₄ analogue (THEREDA Sr–SO₄ set)', b: [0.2, 1.30949, -24.31] },
  { id: 'zero', model: 'none', label: 'no Ba–SO₄ term (PHREEQC pitzer.dat, data0.ypf)', b: [0, 0, 0] },
];
/** Seawater-type waters (free sulphate up to so4 mol/kg, a twofold seawater concentrate): the barite index under every treatment stays within lo … hi of the value of the default treatment; a wider band than warn is reported with the results. */
export const BARITE_BAND = { so4: 0.06, lo: -0.16, hi: 0.09, warn: 0.1 };
/**
 * Change of the barite saturation index under the other treatments of the Ba–SO4 interaction (Pitzer model only).
 * Returns { ref, mSO4, I, shift: { ca, sr, zero, pair, pairLo, pairHi, pairSameK, pairw }, lo, hi, up, width, ion, paired } — shifts
 * relative to the index of the state itself, whose treatment is ref ('pair', 'analogue' or 'none'; its own shift is 0);
 * lo ≤ 0 ≤ hi span all treatments, each with its own log Ksp; up (0 ≤ up ≤ hi) is the shift to the highest index among the
 * published treatments ('pair' with its central constants, 'ca', 'zero') — the conservative envelope; pairw is the ion pair
 * with the log Ksp fitted in this work to pure-water solubility; ion is the spread of the three ion-interaction treatments
 * alone; paired the share of barium bound in the pair under the ion-pair treatment; pairSameK the ion pair with the
 * log Ksp of the other treatments. Null when the model is not Pitzer or the water holds no sulphate.
 */
export function bariteBand(eq) {
  if (eq.model !== 'pitzer') return null;
  const iBa = si('Ba'), iS = si('SO4'), mS = eq.m[iS], I = eq.I;
  if (!(mS > 0) || !(I > 0)) return null;
  const sI = Math.sqrt(I), g14 = pzG(1.4 * sI), g12 = pzG(12 * sI), P = eq.pz, T = Object.fromEntries(BASO4_TREATMENTS.map((t) => [t.id, t]));
  const Bown = P.B0[KBASO4] + P.B1[KBASO4] * g14 + P.B2[KBASO4] * g12, zB = ([b0, b1, b2]) => (2 * mS * (b0 + b1 * g14 + b2 * g12)) / LN10; // index relative to "no Ba–SO4 term, log Ksp −9.97"
  const q = Math.exp(eq.lnG[iBa] - 2 * mS * Bown + eq.lnG[iS]) * mS; // γ(Ba)·γ(SO4)·m(SO4) without a Ba–SO4 term (Cφ of the binary is zero in every treatment)
  const zP = (lk) => -Math.log10(1 + 10 ** lk * q), own = eq.pair ? zP(T.pair.logK + eq.pk) - dkPair(eq.ba) : (2 * mS * Bown) / LN10;
  const ca = zB(T.ca.b) - own, sr = zB(T.sr.b) - own, zero = -own, pairSameK = zP(T.pair.logK) - own, pair = pairSameK - DK_PAIR, pairLo = zP(T.pair.logK + T.pair.sd) - DK_PAIR - own, pairHi = zP(T.pair.logK - T.pair.sd) - DK_PAIR - own, pairw = pairSameK - dkPair('pairw');
  const all = [0, ca, sr, zero, pairLo, pairHi, pairw], lo = Math.min(...all), hi = Math.max(...all), up = Math.max(0, ca, zero, pair);
  return { ref: eq.ba, mSO4: mS, I, shift: { ca, sr, zero, pair, pairLo, pairHi, pairSameK, pairw }, lo, hi, up, width: hi - lo, ion: Math.max(ca, sr, zero) - Math.min(ca, sr, zero), paired: 1 - 1 / (1 + 10 ** T.pair.logK * q) };
}
/** Treatments of the Ba–SO4 interaction that are published as such (the conservative envelope is the highest barite index among them and the treatment in use). */
export const BARITE_PUBLISHED = ['pair', 'analogue', 'none'];
/**
 * Barite index used for verdicts: the best estimate SI (the state's own treatment) and the conservative envelope —
 * the highest index among the published treatments (ion pair with the constants of Felmy et al. 1990, Ca–SO4 analogue of
 * Rogers 1981, no Ba–SO4 term as in pitzer.dat) and the treatment in use. se > 0 marks an index calibrated against the
 * user's own measurements (standard error of the fitted Δ log Ksp): the range is then ± 2·se and the envelope SI + 2·se.
 * basis 'best' returns the best estimate as the verdict value. Outside the Pitzer model there is one treatment only.
 */
export function bariteVerdict(eq, P = 1, dk = 0, { basis = 'envelope', se = 0 } = {}) {
  const si0 = saturationIndex(eq, 'barite', P, dk), b = bariteBand(eq), cal = se > 0;
  const lo = cal ? -2 * se : b ? b.lo : 0, hi = cal ? 2 * se : b ? b.hi : 0, up = cal ? 2 * se : b ? b.up : 0;
  return { best: si0, lo: si0 + lo, hi: si0 + hi, envelope: si0 + up, verdict: basis === 'best' ? si0 : si0 + up, up: basis === 'best' ? 0 : up, basis: basis === 'best' ? 'best' : 'envelope', calibrated: cal, band: b };
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
const wrap = (T, model, n, alk, w, eq, extra = {}) => ({ T, model, n, alk, w, pH: eq.pH, eq, ...extra, ba: eq.ba, pk: eq.pk });
const pkOf = (sol) => sol.pk ?? sol.eq?.pk ?? 0; // offset of the BaSO4(aq) association constant a solution was made with
const baOf = (sol) => sol.ba ?? sol.eq?.ba ?? BARITE_MODEL; // Ba–SO4 treatment a solution was made with
const totOf = (n, w) => { const t = new Float64Array(NM); for (let i = 0; i < NM; i++) t[i] = Math.max(0, n[i]) / w; return t; };

/** Build a solution from a water analysis (mg/L), temperature and measured pH. */
export function makeSolution({ ions, T = 25, pH = 8, model = 'pitzer', kgw = 1, bariteModel: ba = BARITE_MODEL, pairDK = 0 }) {
  const b = molalBasis(ions, T), eq = new Eq(T, model, ba, pairDK).run(b.tot, { pH, alkC: b.alkC });
  return wrap(T, eq.model, eq.tot.map((x) => x * kgw), eq.alk * kgw, kgw, eq, { kgwPerL: b.kgw });
}
/** Re-equilibrate a solution: o.pH fixes the pH (alkalinity then follows), o.pCO2 opens it to a gas phase, o.T changes temperature. */
export function equilibrate(sol, o = {}) {
  const T = o.T ?? sol.T, eq = new Eq(T, o.model || sol.model, o.bariteModel ?? baOf(sol), o.pairDK ?? pkOf(sol));
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
  return equilibrate({ T: a.T * (1 - fb) + b.T * fb, model: a.model, n, alk: (a.alk / a.w) * (1 - fb) + (b.alk / b.w) * fb, w: 1, pH: a.pH, kgwPerL: a.kgwPerL, ba: baOf(a), pk: pkOf(a) });
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
  const n = new Float64Array(NM), tot = new Float64Array(NM), SIv = new Float64Array(nK), tol = 1e-8, E = new Eq(sol.T, sol.model, baOf(sol), pkOf(sol));
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
  return { m: -r.solids[id] / sol.w, sol: r.sol, molality: r.sol.n[MINERALS[id]._i[0]] / MINERALS[id]._n[0] / r.sol.w }; // m: mol dissolved per kg of the initial water; molality: per kg of water of the saturated solution (differs for hydrates)
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
export function analyzeWater({ ions, T = 25, pH = 8, P = 1, model = 'pitzer', bariteModel: ba }) {
  const r = describe(makeSolution({ ions, T, pH, model, bariteModel: ba }), P);
  for (const id of Object.keys(MINERALS)) if (!(id in r.SI)) { r.SI[id] = -99; r.omega[id] = 0; } // −99 marks a mineral whose constituents are absent
  return r;
}
/** Concentrate a water by the factor cf with carbonate re-equilibration. Returns the new analysis (mg/L), pH and state. */
export function concentrate({ ions, T = 25, pH = 8, cf = 2, model = 'pitzer', co2 = 'ro', rej = 1, P = 1, bariteModel: ba }) {
  const s = concentrateSolution(makeSolution({ ions, T, pH, model, bariteModel: ba }), Math.max(cf, 1e-6), { co2, rej }), d = describe(s, P, {}, false);
  return { ions: d.ions, pH: d.pH, T, tds: d.tds, cf, I: d.I, aw: d.aw, SI: d.SI, density: d.density, solution: s };
}
/** Equilibrium precipitation of the listed minerals. Solids in mg per litre of the original water. */
export function precipitate({ ions, T = 25, pH = 8, minerals = SCALE_MINERALS.filter((k) => k !== 'aragonite' && k !== 'anhydrite'), model = 'pitzer', P = 1, bariteModel: ba }) {
  const s0 = makeSolution({ ions, T, pH, model, bariteModel: ba }), r = precipitateSolution(s0, minerals, { P }), io = solutionToIons(r.sol), f = s0.kgwPerL / s0.w;
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
const solidMu0 = (M, T, P, dk, model = 'pitzer', ba = BARITE_MODEL) => LN10 * (logKfor(M, T, model, ba) + (M.dV ? (-M.dV * 1e-6 * (P - 1) * 1e5) / (R * tk(T) * LN10) : 0) + dk - M.nOH * logKw(T));
/** Total Gibbs energy G/RT (mol) of a solution plus solids ({ id: mol }), on the reference above. */
export function gibbsEnergy(sol, solids = {}, { P = 1, dk = {} } = {}) {
  const e = sol.eq, K = e.K;
  let g = (sol.w / MW_W) * Math.log(e.aw);
  for (let s = 0; s < NS; s++) { const x = e.m[s]; if (x > 0) g += x * sol.w * ((s > IH ? -LN10 * K[s - NM - 1] : 0) + Math.log(x) + e.lnG[s]); }
  for (const [id, n] of Object.entries(solids)) if (MINERALS[id] && n) g += n * solidMu0(MINERALS[id], sol.T, P, dk[id] || 0, sol.model, baOf(sol));
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
  const T = sol.T, E = new Eq(T, sol.model, baOf(sol), pkOf(sol)), K = E.K, w0 = sol.w, law0 = LN10 * logKw(T);
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
  const so = mins.map((M, k) => { const ok = M._st.every(([i]) => cIdx[i] >= 0), ci = M._st.map(([i]) => cIdx[i]), ai = M._st.map(([, nu]) => nu); if (M.nOH) { ci.push(iHc); ai.push(-M.nOH); } return { ok, ci, ai, mu: solidMu0(M, T, P, dk[ids[k]] || 0, sol.model, E.ba), nW: M.nW + M.nOH + M.awx }; });
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
export async function spacerChannelCFD({ h = 7.1e-4, u0 = 0.1, vw = 4e-6, D = 1.5e-9, rej = 1, rho = 1000, mu = 1e-3, arr = 'zigzag', lm = 3e-3, df = 3.6e-4, nFil = 6, nxFil = 24, ny = 32, stretch = 8, dP = 0, pi = null, contact = 0.15, precip = null, maxIter = 500, tol = 2e-5, scalIter = 300, solver = {} } = {}, ctx) {
  const nF = clamp(Math.round(nFil), 1, 40), L = nF * lm, nx = clamp(Math.round(nxFil), 6, 240) * nF, NY = clamp(2 * Math.round(ny / 2), 8, 200), g = yGrid(h, NY, stretch);
  const mk = buildMask({ type: 'spacer', arr, L, H: h, df, lm, nFil: nF }, nx, NY, g.yc), R = clamp(rej, 0.5, 1), osm = typeof pi === 'function', dx0 = L / nx;
  // filament footprint: the membrane under the flattened contact of a wall-touching filament (half-width a) does not permeate.
  // A cell that is partly covered permeates in proportion to its open length, so the footprint has the same width on every grid.
  const aC = arr === 'zigzag' || arr === 'cavity' ? clamp(contact, 0, 0.5) * Math.min(df, 0.9 * h) : 0, permB = new Float64Array(nx).fill(1), permT = new Float64Array(nx).fill(1), feet = [];
  if (aC > 0) for (let k = 0; k < nF; k++) {
    const xc = (k + 0.3) * lm, top = arr === 'zigzag' && k % 2 === 1, pf = top ? permT : permB;
    feet.push({ x: xc, a: aC, top });
    // flattened contact: the cells lying wholly under the footprint (and always the column under the filament axis) are solid from the
    // membrane to the filament axis, so the contact is sealed on every grid instead of leaking where the stair-step circle misses the wall row
    const rF = Math.min(df, 0.9 * h) / 2;
    for (let i = 0; i < nx; i++) if ((i * dx0 >= xc - aC - 1e-9 * dx0 && (i + 1) * dx0 <= xc + aC + 1e-9 * dx0) || Math.abs((i + 0.5) * dx0 - xc) <= 0.5 * dx0 * (1 + 1e-6)) for (let j = 0; j < NY; j++) if (top ? g.yc[j] >= h - rF : g.yc[j] <= rF) mk.solid[j * nx + i] = 1;
    for (let i = Math.max(0, Math.floor((xc - aC) / dx0)); i <= Math.min(nx - 1, Math.floor((xc + aC) / dx0)); i++) pf[i] = Math.max(0, pf[i] - Math.max(0, Math.min((i + 1) * dx0, xc + aC) - Math.max(i * dx0, xc - aC)) / dx0);
  }
  // uniform-flux mode: a very large driving pressure makes the flux independent of the (small) hydraulic pressure variation
  const dPm = osm ? dP : 1e9, pi0 = osm ? pi(1) - pi(1 - R) : 0, A = vw / Math.max(dPm - pi0, 1e-9), B = R < 1 ? (vw * (1 - R)) / R : 0;
  const r = await solveChannel({ L, H: h, nx, ny: NY, stretch, solid: mk.solid, rho, mu, Uin: u0, inlet: 'parabolic', scheme: 'hybrid', steady: true, maxIter, tol, scalIter, ...solver,
    species: { c0: 1, D, A, B, R, dP: dPm, pi: osm ? pi : () => 0, bot: 'membrane', top: 'membrane', ...(aC > 0 ? { permB, permT } : {}) }, ...(precip ? { precip: { D, c0: 1, kr: precip.kr, csat: precip.csat, bot: true, top: true } } : {}) }, ctx);
  const { dx, dy, yc, u, nu1, solid } = r, phi = r.spc.phi, x = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx);
  const cb = new Array(nx), blockB = new Array(nx), blockT = new Array(nx);
  for (let i = 0; i < nx; i++) {
    let q = 0, qc = 0;
    for (let j = 0; j < NY; j++) { const P = j * nx + i; if (solid[P]) continue; const uc = 0.5 * (u[j * nu1 + i] + u[j * nu1 + i + 1]); q += uc * dy[j]; qc += uc * phi[P] * dy[j]; }
    cb[i] = q > 0 ? qc / q : 1; blockB[i] = !!solid[i]; blockT[i] = !!solid[(NY - 1) * nx + i];
  }
  let sin = 0, sout = 0, qin = 0, qout = 0, perm = 0, aOpen = 0;
  for (let j = 0; j < NY; j++) { sin += r.uin[j] * dy[j]; qin += r.uin[j] * dy[j]; const uo = u[j * nu1 + nx]; qout += uo * dy[j]; sout += uo * phi[j * nx + nx - 1] * dy[j]; }
  for (let i = 0; i < nx; i++) { sout += (r.Jb[i] * r.spc.pB[i] + r.Jt[i] * r.spc.pT[i]) * dx; perm += (r.Jb[i] + r.Jt[i]) * dx; aOpen += ((blockB[i] ? 0 : permB[i]) + (blockT[i] ? 0 : permT[i])) * dx; }
  const Jmean = perm / Math.max(aOpen, dx);
  // thermodynamic ceiling of the wall concentration: the flux stops where the osmotic-pressure difference across the membrane equals the applied pressure
  let cCap = null;
  if (osm) { const net = (c) => dP - (pi(c) - pi((1 - R) * c)); if (net(1) > 0 && net(40) < 0) cCap = brent(net, 1, 40, 1e-10); }
  const out = { x, y: Array.from(yc), dx, dy: Array.from(dy), nx, ny: NY, L, h, cwB: Array.from(r.spc.wB), cwT: Array.from(r.spc.wT), cb, JB: Array.from(r.Jb), JT: Array.from(r.Jt), tauB: Array.from(r.tauB), tauT: Array.from(r.tauT), blockB, blockT, permB: Array.from(permB), permT: Array.from(permT), feet, contactHalfWidth: aC, cCap,
    field: Array.from({ length: NY }, (_, j) => Array.from({ length: nx }, (_, i) => phi[j * nx + i])), mask: Array.from({ length: NY }, (_, j) => Array.from({ length: nx }, (_, i) => !!solid[j * nx + i])), shapes: mk.shapes,
    Jmean, recovery: perm / qin, massError: (qin - qout - perm) / qin, balance: { in: sin, out: sout }, converged: r.converged, iters: r.iters, scalRes: r.scalRes, A, B, osmotic: osm, solidFraction: mk.solidFraction, Jopen: A * (osm ? dP : dPm) };
  if (precip && r.scal) Object.assign(out, { scaleB: Array.from(r.scal.wB), scaleT: Array.from(r.scal.wT), scaleNB: Array.from(r.scal.nB), scaleNT: Array.from(r.scal.nT) });
  return out;
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
// Nuttall (1978, 7, 263: tables 17, 20, 23 and 26), Goldberg, Nuttall & Staples (1979, 8, 923: FeCl2), Goldberg (1979,
// 8, 1005: MnCl2) and Goldberg (1981, 10, 671). lim = [model, highest molality used, tolerance]:
// each model is only tested inside its validity range (Davies I ≤ 0.3, SIT I ≤ 3, Bromley I ≤ 6 mol/kg).
const ACT_REF = {
  NaCl: { c: 'Na', a: 'Cl', src: 'Hamer & Wu 1972', d: [[0.01, 0.903, 0.968], [0.05, 0.822, 0.944], [0.1, 0.779, 0.933], [0.2, 0.734, 0.924], [0.5, 0.681, 0.921], [1, 0.657, 0.936], [2, 0.668, 0.984], [3, 0.714, 1.045], [4, 0.783, 1.116], [5, 0.874, 1.191], [6, 0.986, 1.27]], lim: [['pitzer', 6, 0.006], ['bromley', 6, 0.02], ['sit', 3, 0.035], ['davies', 0.2, 0.03]] },
  KCl: { c: 'K', a: 'Cl', src: 'Hamer & Wu 1972', d: [[0.1, 0.768, 0.927], [0.2, 0.717, 0.913], [0.5, 0.649, 0.9], [1, 0.604, 0.898], [2, 0.573, 0.912], [3, 0.568, 0.936], [4, 0.576, 0.965]], lim: [['pitzer', 4, 0.005], ['bromley', 4, 0.008], ['sit', 3, 0.045], ['davies', 0.2, 0.06]] },
  'MgCl₂': { c: 'Mg', a: 'Cl', src: 'Goldberg & Nuttall 1978', d: [[0.1, 0.5347, 0.8648], [0.2, 0.4935, 0.876], [0.5, 0.4855, 0.9475], [1, 1 * 0.5769, 1.1092], [2, 1.0655, 1.525], [3, 2.3498, 2.0125], [4, 5.6692, 2.5313], [5, 14.396, 3.0645]], lim: [['pitzer', 5, 0.03], ['bromley', 1, 0.03], ['sit', 1, 0.035], ['davies', 0.1, 0.035]] },
  'CaCl₂': { c: 'Ca', a: 'Cl', src: 'Goldberg & Nuttall 1978', d: [[0.1, 0.5171, 0.8516], [0.2, 0.4692, 0.8568], [0.5, 0.4442, 0.9134], [1, 0.4956, 1.0444], [2, 0.7842, 1.3754], [3, 1.455, 1.7685]], lim: [['pitzer', 3, 0.03], ['bromley', 2, 0.035], ['sit', 1, 0.02], ['davies', 0.1, 0.055]] },
  'SrCl₂': { c: 'Sr', a: 'Cl', src: 'Goldberg & Nuttall 1978', d: [[0.1, 0.5063, 0.8435], [0.2, 0.455, 0.8454], [0.5, 0.4209, 0.893], [1, 0.4513, 1.0041], [1.5, 0.5283, 1.1358], [2, 0.6496, 1.2835], [3, 1.0808, 1.6161]], lim: [['pitzer', 3, 0.03], ['bromley', 2, 0.045], ['sit', 1, 0.035], ['davies', 0.1, 0.07]] },
  'BaCl₂': { c: 'Ba', a: 'Cl', src: 'Goldberg & Nuttall 1978', d: [[0.1, 0.4924, 0.8326], [0.2, 0.4365, 0.8301], [0.5, 0.3911, 0.8645], [1, 0.3929, 0.9401], [1.5, 0.4202, 1.0169], [1.785, 0.4417, 1.0603]], lim: [['pitzer', 1.785, 0.04], ['bromley', 1, 0.03], ['sit', 1, 0.04], ['davies', 0.1, 0.1]] },
  'Na₂SO₄': { c: 'Na', a: 'SO4', src: 'Goldberg 1981', d: [[0.1, 0.4457, 0.7869], [0.5, 0.2684, 0.6945], [1, 0.204, 0.6481], [2, 0.1546, 0.6257]], lim: [['pitzer', 2, 0.03], ['bromley', 1, 0.035], ['sit', 1, 0.045], ['davies', 0.1, 0.065]] },
  'NH₄Cl': { c: 'NH4', a: 'Cl', src: 'Hamer & Wu 1972', d: [[0.1, 0.769, 0.927], [0.2, 0.718, 0.913], [0.5, 0.649, 0.9], [1, 0.603, 0.897], [2, 0.569, 0.908], [3, 0.56, 0.926], [4, 0.56, 0.944], [5, 0.563, 0.96], [6, 0.565, 0.97]], lim: [['pitzer', 6, 0.008], ['bromley', 6, 0.03], ['sit', 1, 0.025]] },
  'NH₄NO₃': { c: 'NH4', a: 'NO3', src: 'Hamer & Wu 1972', d: [[0.1, 0.744, 0.912], [0.2, 0.678, 0.889], [1, 0.502, 0.82], [2, 0.419, 0.777], [3, 0.368, 0.744], [4, 0.332, 0.717], [5, 0.303, 0.693], [6, 0.28, 0.672]], lim: [['pitzer', 6, 0.018], ['bromley', 3, 0.04], ['sit', 1, 0.09]] },
  'FeCl₂': { c: 'Fe', a: 'Cl', src: 'Goldberg, Nuttall & Staples 1979', d: [[0.1, 0.5093, 0.8475], [0.2, 0.4625, 0.8555], [0.5, 0.4427, 0.9208], [1, 0.4998, 1.0564], [1.5, 0.6114, 1.207], [2, 0.7818, 1.3712]], lim: [['pitzer', 2, 0.025], ['bromley', 2, 0.04], ['sit', 1, 0.095]] },
  'MnCl₂': { c: 'Mn', a: 'Cl', src: 'Goldberg 1979', d: [[0.1, 0.5126, 0.8487], [0.2, 0.4641, 0.8537], [0.5, 0.4372, 0.9086], [1, 0.4775, 1.0251], [1.5, 0.5567, 1.1456], [2, 0.6611, 1.2597], [3, 0.9224, 1.4496], [4, 1.2221, 1.5822]], lim: [['pitzer', 2, 0.018], ['pitzer', 4, 0.055], ['bromley', 1.5, 0.02], ['sit', 1, 0.02]] },
  'K₂SO₄': { c: 'K', a: 'SO4', src: 'Goldberg 1981', d: [[0.1, 0.4239, 0.7687], [0.2, 0.3429, 0.7304], [0.5, 0.2514, 0.6875]], lim: [['pitzer', 0.5, 0.065], ['bromley', 0.5, 0.07], ['sit', 0.5, 0.04], ['davies', 0.1, 0.07]] },
};
// Independent measurements used by the verification list (25 °C).
// MgSO4: osmotic coefficients measured isopiestically against KCl, [m, φ] (Miladinović, Ninković, Todorović & Rard 2007,
// LLNL report UCRL-JRNL-231697, Table 4; J. Solution Chem. 2008); γ± and φ of pure MgSO4 from the Archer & Rard (1998)
// extended ion-interaction model as tabulated in Table 6 of the same report, [I, γ±, φ] with m = I/4; φ of Phutela &
// Pitzer (1986) as quoted in Table I-15 of the report ANL-EBS-MD-000045 REV 02 (2004). iv: a second isopiestic series
// against KCl at 298.15 K, [m, φ] (Ivanović, Popović, Rard, Grujić, Miladinović & Miladinović 2017, J. Chem. Thermodyn. 113,
// 91; the 12 points of pure MgSO4 as transcribed from the journal table in the NIST ThermoML archive, stored uncertainty
// 0.010–0.016 in φ). sol: solubility at 25 °C in mol/kg from the same archive — Xue et al. 2016 (Fluid Phase Equilib. 408,
// 115; 3.035 ± 0.122), Chen et al. 2018 (J. Chem. Eng. Data 63, 3418; mass fraction 0.2743 ± 0.0082 at 298.2 K) and Wang
// et al. 2017 (J. Chem. Eng. Data 62, 3334; mass fraction 0.2768), mass fractions converted with M = 120.366 g/mol.
// pv: vapour pressure of the solution saturated with MgSO4·7H2O, [K, kPa] read from Fig. 3(D) of López-Borrell et al. (2024,
// Polymers 16, 2335; open access): their own hygrometer measurement (±2 % relative humidity) and the points of Apelblat &
// Manzurola (2003) and Diesnis (1937) replotted there; reading uncertainty ±0.05 to ±0.08 kPa and ±0.15 K (the markers overlap).
// wrs: isopiestic measurements against NaCl at 25 °C by Wu, Rush & Scatchard (1968, J. Phys. Chem. 72, 4048), Tables II and
// III, as reprinted in the Oak Ridge report ORNL-TM-4212 (Water Research Program biennial progress report 1968–1970, p. 14;
// open, read from the page image): the columns of pure MgSO4 of the Na2SO4–MgSO4 and MgSO4–MgCl2 tables, [m, φ] with
// m = νmφ/(2φ) from the printed νmφ (0.5934 … 4.0434) and φ; reference φ(NaCl) of Robinson & Stokes as printed alongside.
const MGSO4_REF = { wrs: [[0.5684, 0.522], [0.7445, 0.5181], [0.7791, 0.5189], [1.0223, 0.5307], [1.147, 0.5397], [1.3253, 0.5575], [1.6507, 0.598], [1.7891, 0.6204], [2.3942, 0.7435], [2.5916, 0.7801]], iv: [[2.3855, 0.7456], [2.2434, 0.7136], [2.1323, 0.6921], [2.0069, 0.6646], [1.8839, 0.6414], [1.803, 0.6247], [1.7421, 0.6173], [1.7001, 0.6094], [1.6432, 0.6012], [1.6071, 0.5946], [1.5846, 0.5925], [1.562, 0.5861]], sol: [['Xue et al. 2016', 3.035], ['Chen et al. 2018', 3.14], ['Wang et al. 2017', 3.18]], pv: [['López-Borrell et al. 2024 (measured)', 298.1, 2.98], ['Apelblat & Manzurola 2003 (replotted)', 298.4, 2.93], ['Diesnis 1937 (replotted)', 296.9, 2.57], ['Diesnis 1937 (replotted)', 299.6, 3.03]], iso: [[0.7222, 0.5219], [1.0718, 0.5347], [1.459, 0.5736], [1.9469, 0.6542], [2.3873, 0.7508], [3.14, 0.9669]], ar: [[1, 0.1073, 0.5487], [1.5, 0.0876, 0.5326], [2, 0.0758, 0.5241], [2.5, 0.0679, 0.5203], [3, 0.0622, 0.5204], [3.5, 0.058, 0.5236], [4, 0.0548, 0.5298], [4.5, 0.0523, 0.5386], [5, 0.0504, 0.5499], [5.5, 0.049, 0.5635], [6, 0.048, 0.5793]], pp: [[0.1, 0.596], [0.5, 0.527], [1, 0.527], [3, 0.925]],
  // second pass (NIST ThermoML transcriptions unless stated). zh: [m, a_w] of pure MgSO₄, isopiestic against CaCl₂ at 298.15 K (Zhang, Li, Yao, Sun, Zeng & Song 2016, J. Chem. Eng. Data 61, 2277, data set 1, rows with m(LiCl) = 0);
  // jah: [m, a_w] to three decimals, ±0.004 (Jahani et al. 2014, J. Chem. Thermodyn. 69, 125); sol2: further solubilities at 298.15 K; zal: [°C, mol/kg], solubility of MgSO₄·7H₂O by Zhang, Asselin & Li (2016, J. Chem. Eng. Data 61, 2363), ±0.042;
  // drh20: deliquescence relative humidity of MgSO₄·7H₂O at 20 °C, "approximate" (Barlas et al. 2023, Plants 12, 2357, Table 1; open, read); steiger: [°C, a_w] deliquescence humidities calculated with the model of Steiger et al. (2011), as tabulated on the salt-damage wiki of HAWK Hildesheim (model values, not measurements).
  zh: [[1.1149, 0.9786], [1.819, 0.9596], [2.5412, 0.9306], [2.9876, 0.9063]], jah: [[0.291, 0.997], [0.348, 0.995], [0.436, 0.994], [0.574, 0.992], [0.767, 0.988]], sol2: [['Meng et al. 2018', 3.012], ['Zhang, Asselin & Li 2016', 3.0676], ['Liu et al. 2018', 2.977]], zal: [[20, 2.8759], [25, 3.0676], [30, 3.2706]], drh20: 0.92, steiger: [[20, 0.913], [25, 0.903], [30, 0.891]] };
// Barite in NaCl solutions, [NaCl, BaSO4 in mmol]: the table of Templeton (1960) as summarised by Blount (1977) and entered in
// the data file Barite_NaCl.dat of Appelo (2015, Appl. Geochem. 55, 62, supplementary files) — second-hand; it agrees within
// 4 % with Templeton's points as plotted in Fig. 9 of Blount (digitised below). Celestite in NaCl solutions,
// [NaCl in mol/L, SrSO4 in mmol/L]: Brower & Renault (1971), New Mexico Bureau of Mines Circular 116, Table 1.
const BARITE_NACL = [[0, 0.0108], [0.05, 0.0302], [0.1, 0.0392], [0.2, 0.052], [0.4, 0.067], [0.6, 0.078], [0.8, 0.088], [1, 0.096], [1.5, 0.112], [2, 0.125], [2.5, 0.135], [3, 0.144], [3.5, 0.153], [4, 0.161], [4.5, 0.169], [5, 0.177]];
const CELESTITE_NACL = [[0, 0.66], [0.001, 0.72], [0.01, 0.96], [0.025, 1.3], [0.1, 1.7], [1, 4.5]];
// Celestite, molal scale, [NaCl in mol/kg, SrSO4 in mmol/kg]: Reardon & Armstrong (1987, Geochim. Cosmochim. Acta 51, 63) at
// 25.0–25.1 °C and Culberson, Latham & Bates (1978, J. Phys. Chem. 82, 2693) in water and 0.7 mol/kg NaCl, both as tabulated
// in Table C.1 of Dal Pozzo (1991, MS thesis, University of Arizona). Barite, [NaCl, BaSO4 in mmol/kg]: Davis & Collins
// (1971) from Table C.2 of the same thesis, which also lists Templeton's 25 °C values at 1, 2 and 4 mol/kg NaCl as molal
// 0.0960, 0.125 and 0.161 — the same numbers as above.
const CELESTITE_RA = [[0, 0.643], [0.0501, 1.14], [0.202, 1.86], [0.4733, 2.69], [1.974, 4.62], [1.981, 4.71], [2.397, 4.9], [2.64, 4.95], [3.076, 4.92], [3.538, 4.89], [4.194, 4.68], [4.977, 4.48]];
const CELESTITE_CLB = { water: 0.644, nacl: [0.7, 3.231] };
const BARITE_DC = [[0, 0.011], [1, 0.0835], [2, 0.1086]];
// Blount (1977, Am. Mineral. 62, 942), read from the page images of the open journal copy. BARITE_BLOUNT.P: Table 3, barite
// in water at 25 °C, [bar, mmol/kg] (his own runs at 24 °C: 0.0289 at 1002 bar, 0.0182 at 500 bar); T: Table 3 at 1 bar,
// [°C, mmol/kg] (60 °C from Melcher and Templeton, 100 °C his own); nacl: Table 11, 25 °C and 1 bar, [NaCl mol/kg, mmol/kg] —
// the measurements of Puchelt (1967) that Blount adopts for his activity-coefficient fit. fig9T and fig9P: the points of
// Templeton (1960) and Puchelt (1967) in Blount's Fig. 9 (25 °C, 1 bar), digitised here from the 262 dpi scan: axes
// calibrated on six ordinate ticks (logarithmic, residual ≤ 0.001 in log₁₀) and on the 0–0.7 abscissa √m/(1 + √m); marker
// centres by image analysis. Reading uncertainty ±0.01 in log₁₀ of the solubility and ±8 % in the NaCl molality: the two
// Puchelt points that Table 11 prints (0.037 at 0.2 and 0.077 at 1.0 mol/kg) are recovered as 0.0369 at 0.213 and 0.0766 at 1.07.
const BARITE_BLOUNT = { P: [[100, 0.0117], [500, 0.0184], [1000, 0.029]], T: [[60, 0.0152], [100, 0.0168]], nacl: [[0.2, 0.037], [1, 0.077]],
  fig9T: [[0.0499, 0.0299], [0.1035, 0.0392], [0.2121, 0.0518], [0.415, 0.0668], [0.63, 0.0777], [1.0755, 0.0949], [2.1636, 0.1248], [3.2442, 0.1453]],
  fig9P: [[0.103, 0.0215], [0.21, 0.0303], [0.213, 0.0369], [0.52, 0.0551], [1.074, 0.0766], [2.205, 0.1127], [4.36, 0.1491]] };
// Synthetic seawater of Culberson, Latham & Bates (1978), molalities, and their celestite solubilities in four seawaters of
// varying Mg/Ca, rows [Mg, Ca, Sr at saturation in mmol/kg], as given in Tables 6 and 8 of Rogers (1981, PhD thesis, LBL-12356;
// the sulphate row of Table 6 is misprinted as carbonate there, the charge balance identifies it; bromide 0.00094 is counted
// as chloride). calc: what Rogers computed for that seawater with the CaSO4 parameters for BaSO4 and K = 1.10·10⁻¹⁰ — barium
// at barite saturation (mol/kg) and γ±(BaSO4).
const SW_CLB = { Na: 0.48523, K: 0.01058, Mg: 0.05518, Ca: 0.01068, Sr: 0.00009, SO4: 0.02927, cel: [[0.06598, 0, 0.414], [0.05519, 0.01076, 0.416], [0.03801, 0.02795, 0.423], [0.02099, 0.04497, 0.422]], calc: { Ba: 2.09e-7, g: 0.134 } };
// Barite in sulphate-bearing media, measured — the data that separate the treatments of the Ba–SO4 interaction (BARITE_MODELS).
// na2so4: Savenko, Savenko & Pokrovsky (2019, Okeanologiya 59 (6), 939–943; publisher's open copy, read in full), Table 1:
// recrystallised reagent BaSO4, 50 g/L, 13 months at 22 ± 1 °C in Na2SO4–NaNO3 solutions of ionic strength 0.05, 0.22 µm
// filtration, barium by ICP-MS (± 3 %); rows [Na2SO4, NaNO3 in mmol/L as made up, dissolved Ba in µmol/L] (mol/L taken as
// mol/kg of water: 0.4 % at this dilution). sw: the same paper, Table 1 (salinity 35: Bay of Biscay 0.221, Mediterranean
// 0.224, synthetic seawater without strontium 0.227 µmol/L) and Savenko, Savenko & Pokrovsky (2023, Okeanologiya 63 (5),
// 745–748; publisher's open copy, read in full), Table 1: synthetic seawater diluted by weight to salinity 0.35–30, same
// solid, method and temperature; rows [salinity, Ba in µmol/L], converted to nmol per kg of solution with the density at
// 22 °C. The four most dilute waters (salinity ≤ 2.1) are kept apart (swd): there the measured ion-activity product falls
// below the solubility product the authors derive from their own Na2SO4 series (7.57·10⁻¹¹), by a factor of 1.8 at salinity
// 0.35, so something other than the medium fixes the barium there; they are listed but not counted.
// jiang: Jiang (1996, J. Solution Chem. 25, 105) at 20 °C — not the article (closed) but its 12 points as plotted in
// Fig. 4-5 of A. Villafáfila García (2005, PhD thesis, Technical University of Denmark, "Measurement and Modelling of Scaling Minerals",
// p. 51; open thesis, read): the figure is a vector drawing, so the marker centres were taken from the drawing commands
// of the page (axes 0–8·10⁻³ mol/kg Na2SO4 and 0–5·10⁻⁷ mol/kg BaSO4; resolution of the drawing about 2 nmol/kg); rows
// [Na2SO4 in mmol/kg, BaSO4 in nmol/kg]. The same thesis states that Lieser (1965) and Felmy et al. (1990) reported their
// barite-in-Na2SO4 data only as plots — there is no table of Felmy's measurements to be read.
// ro: Boerlage (2001, PhD thesis, Wageningen University / IHE Delft, "Scaling and particulate fouling in membrane
// filtration systems"; open scan, tables read from the page images), Table 2.1: acidified feed of a reverse-osmosis pilot
// plant on pretreated Rhine water after charge balance (mmol/L: HCO3 1.45, Cl 5.02, SO4 0.6, NO3 0.29, K 0.16, Mg 0.46,
// Ca 1.75, Na 3.39; pH 6.8–7.0; dissolved organic carbon 1–2 mg/L), and Table 2.2: barium solubility at 25 °C found by
// seeding the feed and the concentrates at 80 and 90 % recovery with BaSO4 crystals (free drift, constant after 3–24 h,
// ICP, ± 10 %): 81, 42 and 34 µg/L; rows [concentration factor, Ba in µg/L].
const BARITE_SULPHATE_SRC = {
  na2so4: [[1, 47, 0.434], [1.5, 45.5, 0.353], [2, 44, 0.298], [2.5, 42.5, 0.22], [3.75, 38.75, 0.16], [5, 35, 0.13], [12.5, 12.5, 0.061]],
  jiang: [[0.3, 440.7], [0.483, 254], [1.082, 147.6], [1.681, 113.4], [1.994, 101.4], [2.593, 85.3], [3.088, 77.3], [3.896, 69.3], [4.677, 63.2], [5.798, 57.2], [6.579, 55.2], [7.882, 51.2]],
  sw: [[5, 0.216], [10, 0.207], [14.1, 0.202], [15, 0.207], [20, 0.201], [21.1, 0.204], [25, 0.203], [28, 0.218], [30, 0.222], [35, 0.221], [35, 0.224], [35, 0.227]],
  swd: [[0.35, 0.308], [0.7, 0.29], [1.05, 0.266], [2.1, 0.242]],
  // Paige (1990, PhD thesis, McMaster University), read from page images. Table 6 (p. 69): points that Paige digitised from the published graphs of
  // Felmy et al. (1990) and Lieser (1965) — [Na₂SO₄ mol/kg, Ba nmol/kg], headed "at 20 °C". Tables 7 and 8 (pp. 70–71): his own measurements in
  // sulphuric acid at 25 and 60 °C — [H₂SO₄ mol/kg, Ba nmol/kg] (¹³³Ba-labelled barite, 5 months from undersaturation at 25 °C, from supersaturation at 60 °C, 0.2 µm filters).
  felmy: [[5.45e-4, 153], [8.79e-4, 120], [1.18e-3, 97.3], [2e-3, 67], [3.15e-3, 51.1], [3.94e-3, 51.1], [4.94e-3, 36.7], [6.15e-3, 36.7], [7.18e-3, 57.4], [7.94e-3, 54.3], [8e-3, 47.9], [8.06e-3, 41.5]],
  lieser: [[5.1e-4, 490], [1e-3, 340], [2.1e-3, 220], [5.4e-3, 140], [1.1e-2, 95], [2.2e-2, 72], [5.5e-2, 50], [0.11, 42], [0.23, 29], [0.56, 23], [0.91, 21], [1.1, 19]],
  h2so4: { 25: [[3e-4, 480], [1e-3, 214], [3e-3, 132], [7.92e-2, 92], [0.45, 88.6], [0.89, 95.7], [1.84, 113], [2.89, 114], [3.95, 111], [6.19, 128]], 60: [[1e-3, 564], [3e-3, 384], [7.92e-2, 333], [0.45, 355], [0.89, 487], [1.84, 521], [2.89, 725], [3.95, 825], [6.19, 767]] },
  ro: [[1, 81], [5, 42], [10, 34]], roFeed: { HCO3: 1.45 * 61.017, Cl: 5.02 * 35.453, SO4: 0.6 * 96.06, NO3: 0.29 * 62.005, K: 0.16 * 39.098, Mg: 0.46 * 24.305, Ca: 1.75 * 40.078, Na: 3.39 * 22.99 }, // mg/L from the mmol/L column
};
const BARITE_SULPHATE = [
  ...BARITE_SULPHATE_SRC.na2so4.map(([s, n, ba]) => ({ medium: 'na2so4', src: 'Savenko et al. 2019, Table 1', kind: 'primary', cond: `${s} mmol/L Na₂SO₄ + ${n} mmol/L NaNO₃, 22 °C`, T: 22, comp: { Na: (2 * s + n) / 1000, SO4: s / 1000, NO3: n / 1000 }, ba: ba * 1000, perKgWater: true })),
  ...BARITE_SULPHATE_SRC.jiang.map(([s, ba]) => ({ medium: 'jiang', src: 'Jiang 1996 (García 2005, Fig. 4-5, read from the drawing)', kind: 'digitised', cond: `${s} mmol/kg Na₂SO₄, 20 °C`, T: 20, comp: { Na: (2 * s) / 1000, SO4: s / 1000 }, ba, perKgWater: true })),
  ...BARITE_SULPHATE_SRC.felmy.map(([c, ba]) => ({ medium: 'felmy', src: 'Felmy et al. 1990 (digitised by Paige 1990, Table 6)', kind: 'digitised', cond: `${+(c * 1000).toPrecision(3)} mmol/kg Na₂SO₄, 20 °C as headed`, T: 20, comp: { Na: 2 * c, SO4: c }, ba, perKgWater: true })),
  ...BARITE_SULPHATE_SRC.lieser.map(([c, ba]) => ({ medium: 'lieser', src: 'Lieser 1965 (digitised by Paige 1990, Table 6)', kind: 'digitised', cond: `${+(c * 1000).toPrecision(3)} mmol/kg Na₂SO₄, 20 °C`, T: 20, comp: { Na: 2 * c, SO4: c }, ba, perKgWater: true })),
  ...[25, 60].flatMap((T) => BARITE_SULPHATE_SRC.h2so4[T].map(([c, ba]) => ({ medium: c < 0.1 ? 'h2so4' : 'h2so4c', src: `Paige 1990, Table ${T === 25 ? 7 : 8}`, kind: 'primary', cond: `${c} mol/kg H₂SO₄, ${T} °C`, T, comp: { SO4: c }, alk: -2 * c, ba, perKgWater: true }))),
  ...['sw', 'swd'].flatMap((k) => BARITE_SULPHATE_SRC[k].map(([S, ba]) => ({ medium: k, src: S === 35 ? 'Savenko et al. 2019, Table 1' : 'Savenko et al. 2023, Table 1', kind: 'primary', cond: `seawater of salinity ${S}, 22 °C`, T: 22, S, ba: (ba * 1e6) / density(22, S) }))),
  ...BARITE_SULPHATE_SRC.ro.map(([cf, ba]) => ({ medium: 'ro', src: 'Boerlage 2001, Tables 2.1 and 2.2', kind: 'primary', cond: cf === 1 ? 'RO feed, 25 °C' : `RO concentrate at ${100 * (1 - 1 / cf)} % recovery, 25 °C`, T: 25, ions: BARITE_SULPHATE_SRC.roFeed, pH: 6.9, cf, ba })),
];
const BARITE_SULPHATE_MEDIA = { jiang: 'Na₂SO₄ solutions, 0.3–7.9 mmol/kg, 20 °C', na2so4: 'Na₂SO₄–NaNO₃ solutions, 1–12.5 mmol/L sulphate at I = 0.05, 22 °C', sw: 'Seawater of salinity 5–35, 22 °C', ro: 'Reverse-osmosis feed and concentrate of a surface water (0.6–6 mmol/L sulphate), 25 °C', swd: 'Seawater diluted to salinity 0.35–2.1, 22 °C (listed, not counted)',
  felmy: 'Na₂SO₄ solutions, 0.5–8.1 mmol/kg, second-hand digitisation of the points of Felmy et al. (listed, not counted: inconsistent with the constants of their source)', h2so4: 'Sulphuric acid, 0.3–79 mmol/kg, 25 and 60 °C', h2so4c: 'Sulphuric acid, 0.45–6.2 mol/kg, 25 and 60 °C (listed, not counted: outside the range of the model)', lieser: 'Na₂SO₄ solutions, 0.5 mmol/kg–1.1 mol/kg, 20 °C, second-hand digitisation of a contested series (listed, not counted)' };
const BARITE_NOT_COUNTED = ['swd', 'felmy', 'h2so4c', 'lieser'];
/** Osmotic coefficient of MgSO4(aq) at 298.15 K from the extended ion-interaction model of Archer & Rard (1998), parameters as reprinted in Table 3 of Miladinović et al. (2007): β⁰ −0.03089, β¹ 3.7687, β² −37.3659, C⁰ 0.016406, C¹ 0.34549, α 1.4 and 12, ω 1, Aφ 0.391475; fitted to 3.6176 mol/kg. */
const archerRardPhi = (m) => { const s = Math.sqrt(4 * m); return 1 - (4 * 0.391475 * s) / (1 + 1.2 * s) + m * (-0.03089 + 3.7687 * Math.exp(-1.4 * s) - 37.3659 * Math.exp(-12 * s)) + 4 * m * m * (0.016406 + 0.34549 * Math.exp(-s)); };
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
const BR_STAT = (() => {
  const rf = BR_REFIT.split('|').map((r) => r.split(' ')), own = rf.filter(([c, a]) => BR_DATA[c + ' ' + a]), d = rf.filter((r) => r[6] != null).map((r) => +r[2] - +r[6]);
  return `Refitted B (published; rms in log γ±): ${own.map(([c, a, B, rms, , , pub]) => `${c}–${a} ${B} (${pub}; ${rms})`).join(', ')}. Over the ${d.length} salts with a published constant the difference is ${fmt(Math.sqrt(sum(d.map((x) => x * x)) / d.length), 2)} kg/mol rms (largest ${fmt(Math.max(...d.map(Math.abs)), 2)}). The refitted ion table reproduces the 52 salt constants with rms 0.0081 (largest 0.022), Bromley’s with 0.0110 (largest 0.037)`;
})();
/** Where every constant set of the suite comes from and how it was checked. Status: confirmed, corrected, replaced, fitted or refitted here, analogue or unconfirmed (with a qualification where the source was read only in part or second-hand). */
const PROVENANCE = [
  ['Pitzer β⁰, β¹, β², Cφ, θ, ψ and CO₂ λ of the Na–K–Mg–Ca–H–Cl–SO₄–OH–HCO₃–CO₃–CO₂ system', 'Harvie, Møller & Weare (1984), read from the LLNL EQ3/6 database file data0.hmw', '173 numbers compared by script: all identical', '25 °C; to salt saturation (I ≈ 20 mol/kg)', 'confirmed'],
  ['Mg–SO₄ within that set (β⁰ 0.221, β¹ 3.343, β² −37.23, Cφ 0.025)', 'Harvie, Møller & Weare (1984) in data0.hmw; the Pitzer & Mayorga (1974) paper itself was not retrievable', 'Results do not depend on the two unread articles (Rard & Miller 1981; Archer & Rard 1998): the parameters in use are those of Harvie, Møller & Weare, read in data0.hmw, and they are validated directly against measured data read from open sources — 31 isopiestic points of four laboratories from 0.57 to 3.14 mol/kg (Wu, Rush & Scatchard 1968; Miladinović et al. 2007; Ivanović et al. 2017; Zhang et al. 2016): largest deviation 0.99 % in φ and 0.0010 in a_w; a measured water activity next to saturation, 0.9063 at 2.9876 mol/kg (Zhang et al. 2016), against 0.9066; the saturated solution itself, 0.9073 by interpolation of those measurements to the model’s saturation molality against 0.9077; a measured deliquescence humidity of epsomite, 92 % at 20 °C (Barlas et al. 2023, approximate), against 91.9 %. The Archer & Rard model enters only as a further, secondary comparison (0.5 %). What the measured data do not settle is the solubility: six single-laboratory values of 2016–2018 span 2.977–3.18 mol/kg (median 3.05) and the suite gives 2.970 — 2.7 % below the median, 2.5–4.1 % below Zhang, Asselin & Li (2016) at 20–30 °C; over that span the saturation water activity moves from 0.907 to 0.894. Below 0.57 mol/kg only a three-decimal series was obtained (Jahani et al. 2014, 0.29–0.77 mol/kg, within its ±0.004). For a reader who holds the articles: compare the osmotic coefficients of the suite (results of saltActivity for Mg–SO₄) with the table of isopiestic molalities and osmotic coefficients of MgSO₄ at 25 °C in Rard & Miller (1981, J. Chem. Eng. Data 26, 33), and the nine parameters in archerRardPhi and the eleven (γ±, φ) pairs of MGSO4_REF.ar with the 298.15 K parameter and smoothed-value tables of Archer & Rard (1998, J. Chem. Eng. Data 43, 791). Detail — independent data: φ within 1.0 % of six isopiestic measurements from 0.72 to 3.14 mol/kg (Miladinović et al. 2007); γ± and φ within 0.5 % of the Archer & Rard (1998) evaluation from 0.25 to 1.5 mol/kg; epsomite solubility 2.97 mol/kg against 2.96 and 3.02 retrieved; saturated-solution water activity: from measured isopiestic osmotic coefficients at the saturation molality — ln a_w = −2·m·M_w·φ (exact) with φ of Miladinović et al. (0.6542, 0.7508 and 0.9669 at 1.947, 2.387 and 3.14 mol/kg) interpolated to each retrieved solubility (2.96, 2.97 model, 3.018, 3.035 mol/kg): a_w 0.9076, 0.9070, 0.9041, 0.9030, i.e. 0.9053 ± 0.0023 (solubility spread) ± 0.0003 (φ ± 0.003) ± 0.0006 (interpolation); the suite gives 0.9077, +0.0024 from the centre of that range and +0.0007 at its own saturation molality. Secondary: 0.9038–0.9074 computed here from the Archer & Rard model parameters (derived). Also: φ within 0.7 % of a second isopiestic series of 12 points from 1.56 to 2.39 mol/kg (Ivanović, Popović, Rard et al. 2017 — primary, read through the NIST ThermoML transcription); solubility 3.035 ± 0.122 (Xue et al. 2016), 3.14 (Chen et al. 2018) and 3.18 mol/kg (Wang et al. 2017) from the same archive — the model is 2–7 % below them; measured vapour pressure of the saturated solution near 25 °C: p/p° 0.94 (López-Borrell et al. 2024, hygrometer, ±2 % RH), 0.91 (Apelblat & Manzurola 2003) and 0.88 (Diesnis 1937), the last two replotted in the first — all read from a figure (digitised, ±0.03; a loose cross-check only, the primary reference is the one built on the measured osmotic coefficients). Added from an open primary reprint: φ within 0.8 % of nine isopiestic points from 0.57 to 2.39 mol/kg measured against NaCl by Wu, Rush & Scatchard (1968, J. Phys. Chem. 72, 4048, Tables II and III as reprinted in the Oak Ridge report ORNL-TM-4212, p. 14; read from the page image); their tenth point, at 2.59 mol/kg, lies 2.5 % below the suite and 2.9 % below the Archer & Rard model. Second pass, all from the NIST ThermoML archive unless stated (transcriptions of the journal tables; the articles themselves not read): a_w of four pure MgSO₄ solutions against CaCl₂ (Zhang, Li, Yao, Sun, Zeng & Song 2016, J. Chem. Eng. Data 61, 2277) within 0.0003; solubilities 3.012 (Meng et al. 2018), 3.0676 (Zhang, Asselin & Li 2016) and 2.977 mol/kg (Liu et al. 2018); deliquescence humidity 92 % at 20 °C (Barlas et al. 2023, Plants 12, 2357 — open article, read; approximate); model-calculated deliquescence humidities 91.3, 90.3 and 89.1 % at 20, 25 and 30 °C (Steiger et al. 2011, through the salt-damage wiki of HAWK Hildesheim — secondary, not measurements) against 91.9, 90.8 and 89.5 %. Still not obtained: Archer & Rard 1998 and Rard & Miller 1981 (known only through the model parameters and comparisons reprinted in Miladinović et al.; OSTI holds both records without full text, as it does for Phutela & Pitzer 1986, Pabalan & Pitzer 1987 and Pitzer & Mayorga 1974; neither is in the ThermoML archive, which starts in 2003), a tabulated vapour pressure or humidity measured over the epsomite-saturated solution at 25 °C (Chou & Seal 2003, Steiger et al. 2011, Apelblat & Manzurola 2003, Ha & Chan 1999 and Diesnis 1937 are closed or refuse scripted access; Wexler & Hasegawa 1954 and Greenspan 1977 do not list MgSO₄), and any critical evaluation of MgSO₄ (none in the NIST reprint index; the IUPAC–NIST Solubility Data Series has no volume on magnesium sulphate)', '25 °C; to epsomite saturation', 'validated against measured data of four laboratories, independently of the two unread articles (Miladinović et al. 2007: open report version of the primary, read; Wu, Rush & Scatchard 1968: open reprint of the primary tables, read; Ivanović et al. 2017 and Zhang et al. 2016: database transcriptions of the primary tables); water activity at saturation: measured next to saturation (0.9063 at 2.9876 mol/kg) and by a measured deliquescence humidity at 20 °C (approximate); vapour pressures at 25 °C only digitised; Rard & Miller 1981 and Archer & Rard 1998 not obtained'],
  ['Pitzer parameters of BaCl₂ and borate (B(OH)₄⁻, B(OH)₃ λ)', 'USGS PHREEQC pitzer_2012.dat and LLNL data0.ypf (Pitzer & Mayorga 1973 for BaCl₂; Felmy & Weare 1986 for borate)', 'Numbers compared with the files; BaCl₂ γ± and φ within 3.3 % and 1.2 % of Goldberg & Nuttall (1978) to saturation (1.785 mol/kg)', '25 °C', 'confirmed'],
  ['Strontium: Pitzer Sr–Cl, Sr–SO₄, θ(Sr,Na), θ(Sr,K), θ(Sr,Mg), θ(Sr,Ca), ψ(Sr,M,Cl)', 'THEREDA database, PHREEQC release of 2020-10-22 (reference SCH2016, Scharge 2016); β⁰(Sr–SO₄) = 0.2 is adopted there from CaSO₄, β¹ 1.30949 and β² −24.31 are fitted', 'Replaces the Sr–SO₄ entry of pitzer.dat, which is numerically the Ca–SO₄ set. SrCl₂: γ± within 2.4 %, φ within 1.1 % of Goldberg & Nuttall (1978) to 3 mol/kg (previous set 5.5 %). Celestite with log K −6.63 (next row): within 4.4 % of Reardon & Armstrong (1987) in 11 NaCl solutions from 0.05 to 5 mol/kg, +1.8 % in 0.7 mol/kg NaCl and +1.4 to +4.2 % in four synthetic seawaters (Culberson et al. 1978), −4.2 % in water', '25 °C; NaCl solutions to 5 mol/kg and seawater tested; fitted within the THEREDA oceanic-salt set and used here with the Harvie–Møller–Weare set', 'replaced'],
  ['log K of celestite with the Pitzer model: −6.63 (was −6.550)', 'USGS WATEQ4F wateq4f.dat and PHREEQC pitzer.dat/phreeqc.dat (−6.63); the THEREDA value −6.550 (Dyrssen et al. 1969) that accompanies the strontium parameters is not used', 'With −6.550 the model lies 6 % above the measured solubility in water, 10–15 % above Reardon & Armstrong (1987) in NaCl solutions and 22–25 % above Culberson et al. (1978) in seawater; it had been accepted on the two-figure data of Brower & Renault (1971), which lie well above both. Measurements read from secondary tabulations (Dal Pozzo 1991, Table C.1; Rogers 1981, Tables 6 and 8); Reardon & Armstrong (1987) and Culberson, Latham & Bates (1978) are closed (no open-access location in Unpaywall, OpenAlex, CORE or Semantic Scholar; the IUPAC–NIST Solubility Data Series has no volume on alkaline-earth sulphates). Felmy, Rai & Amonette (1990, primary, abstract) give log Ksp −6.62 ± 0.02 from celestite solubilities in Na₂SO₄ solutions', '25 °C; ΔH −4.3 kJ/mol for other temperatures', 'corrected'],
  ['Pitzer parameters of NH₄⁺, Fe²⁺ and Mn²⁺: NH₄–Cl, NH₄–SO₄, NH₄–NO₃, NH₄–HCO₃, θ(H,NH₄), Fe–Cl, Fe–SO₄, Fe–HSO₄, Mn–Cl, Mn–SO₄', 'LLNL EQ3/6 data0.ypf (NH₄Cl: refit of Thiessen & Simonson 1990; (NH₄)₂SO₄: refit of Clegg et al. 1996; NH₄NO₃, NH₄HCO₃, θ: Pitzer 1991; FeCl₂: Pitzer & Mayorga 1973; FeSO₄: listed as Millero & Yao 1995; MnSO₄: Pitzer & Mayorga 1974) and USGS PHREEQC pitzer.dat (MnCl₂ 0.327225, 1.55025, −0.0204972; Fe–HSO₄)', 'Replace the K⁺ and Mg²⁺ analogues. Against the NIST tables: NH₄Cl γ± 0.6 % to 6 mol/kg and NH₄NO₃ 1.4 % to 6 mol/kg (Hamer & Wu 1972), FeCl₂ 2.0 % to 2 mol/kg (Goldberg, Nuttall & Staples 1979), MnCl₂ 1.4 % to 2 and 4.7 % to 4 mol/kg (Goldberg 1979). The MnCl₂ set of data0.ypf (Kim & Frederick 1988, with θ(Na,Mn) and ψ(Na,Mn,Cl)) was not adopted: it is 7 % off below 1 mol/kg. No table of (NH₄)₂SO₄, FeSO₄ or MnSO₄ was at hand, so the sulphates are compared with the files only', '25 °C; chlorides to the molalities named', 'replaced'],
  ['Pitzer parameters of NaF, KF, KNO₃, Mg(NO₃)₂, Na₂HPO₄, K₂HPO₄, θ(Cl,NO₃)', 'Pitzer (1991) tabulation, read from the LLNL EQ3/6 Pitzer file data0.ypf', '19 numbers compared; Mg(NO₃)₂ restored to full precision, K₂HPO₄ added', '25 °C', 'confirmed'],
  ['Pitzer parameters of NaNO₃ and Ca(NO₃)₂', 'LLNL EQ3/6 data0.ypf (revision 0): refits with α₁ = 2 to the Archer (2000) and Oakes et al. (2000) evaluations', 'The earlier values could not be found in a retrievable file and were replaced by this set', '25 °C; NaNO₃ checked here against Hamer & Wu to 6 mol/kg', 'replaced'],
  ['Pitzer λ of dissolved silica with Na⁺, K⁺, Mg²⁺, Ca²⁺, SO₄²⁻', 'USGS PHREEQC pitzer.dat (Appelo 2015)', 'The earlier values could not be found in any database and were replaced', '25 °C', 'replaced'],
  ['Ba–SO₄ interaction of the Pitzer model. Default: the Ca–SO₄ binary as an analogue for Ba–SO₄ (β⁰ 0.2, β¹ 3.1973, β² −54.24) with barite log Ksp −9.97. Selectable instead (input “Barium–sulphate interaction”): an explicit BaSO₄(aq) ion pair, log K 2.72, with barite log Ksp −10.05 and no Ba–SO₄ binary; the same pair with log Ksp −9.965 fitted here to pure water; no Ba–SO₄ term with log Ksp −9.97. No θ or ψ with Ba²⁺', 'Default: the approximation of Rogers (1981, PhD thesis with K. S. Pitzer, LBL-12356; read), who computed barite solubility with the CaSO₄ parameters. Ion pair — a published model: Felmy, Rai & Amonette (1990, J. Solution Chem. 19, 175 — primary). Read: the abstract and the first two pages only (publisher’s preview). Taken from them: log K of BaSO₄(aq) 2.72 ± 0.09, barite log Ksp −10.05 ± 0.05, and the statement that for BaSO₄ the explicit ion-association species is preferred over a description by ion-interaction parameters alone, on their own barite solubilities in Na₂SO₄ solutions. Not read: the activity-coefficient expressions of their model, the parameters it uses, and the solubility data — which, according to García (2005, PhD thesis, Technical University of Denmark; read), were published as plots only. The structure used here for the pair — Pitzer parameters of this suite for every other interaction, an explicit association equilibrium, free-ion activity product against Ksp — is the one that Monnin et al. (1999, Mar. Chem. 65, 253 — primary, open copy, read) describe for the model of Monnin (1999); they print no constants. Assumptions of this suite, not of the sources: activity coefficient 1 for the neutral pair; log K independent of temperature; temperature and pressure dependence of the barite log Ksp as in WATEQ4F, shifted by −0.080. No Ba–SO₄ term: USGS PHREEQC pitzer.dat (Appelo 2015; current file), LLNL data0.ypf, PHRQPITZ (Dal Pozzo 1991, Table 2.2), THEREDA, frezchem, ColdChem and SOLMINEQ.88. Measured barite solubilities in sulphate-bearing media, obtained from open documents and read in full unless stated: Savenko, Savenko & Pokrovsky (2019, Okeanologiya 59, 939, Table 1 — primary: Na₂SO₄–NaNO₃ solutions and seawater at 22 °C) and (2023, Okeanologiya 63, 745, Table 1 — primary: diluted seawater); Boerlage (2001, PhD thesis, Wageningen University / IHE Delft, Tables 2.1 and 2.2 — primary: reverse-osmosis feed and concentrate at 25 °C); Jiang (1996, J. Solution Chem. 25, 105) at 20 °C — digitised: 12 points taken from the vector drawing of Fig. 4-5 of García (2005), the article itself not read; Paige (1990, PhD thesis, McMaster University, open, Tables 6–8 read from page images — primary: his own solubilities in sulphuric acid at 25 and 60 °C; secondary: points he digitised from the graphs of Felmy et al. 1990 and Lieser 1965). Still not obtained: Felmy et al. (1990) beyond the preview, Jiang (1996), Monnin & Galinier (1988), Monnin (1999), Burton, Marshall & Phillips (1968; known through two quotations), Church & Wolgemuth (1972), Rushdi et al. (2000), Dideriksen et al. (2024)', 'All four treatments against 73 counted measured solubilities (results table “Barite treatments against measured solubility”; the same numbers are verification checks), mean / root mean square of log₁₀(model/measured). Water and NaCl solutions, 32 points (Templeton, Davis & Collins, Blount, Puchelt): analogue +0.01 / 0.06, no term +0.01 / 0.06, ion pair with −10.05 −0.03 / 0.07, ion pair with −9.965 +0.01 / 0.06 — here the treatments differ by the solubility product alone. Sulphate-bearing media, 41 points: analogue −0.02 / 0.05, ion pair with −10.05 +0.02 / 0.06, no term −0.05 / 0.07, ion pair with −9.965 +0.10 / 0.12. By medium, analogue: Na₂SO₄ 0.3–7.9 mmol/kg at 20 °C (Jiang, digitised) −0.01 / 0.02, every point but the first within 6 %; Na₂SO₄–NaNO₃ at I = 0.05 +0.03 / 0.06; seawater of salinity 5–35 −0.04 / 0.05 (17 % less barium than measured at salinity 35, i.e. an index 0.08 too high — safe side); reverse-osmosis feed and concentrate −0.07 / 0.07; sulphuric acid of 0.3–79 mmol/kg at 25 and 60 °C (Paige 1990, primary, 7 points) −0.03 / 0.04, every point within 16 % — the first measured support above 25 °C and above 30 mmol/kg of total sulphate; there the ion pair with the published constants is slightly closer (0.00 / 0.03). The ion pair with the published constants fits the two series of Savenko et al. as well or better but is 41 % high at 7.9 mmol/kg Na₂SO₄; the ion pair with the pure-water log Ksp is 72 % high there and is rejected by the data. The default is therefore the analogue — unchanged, now on measured evidence in sulphate media instead of on water and NaCl alone. Conservative envelope (highest index among the published treatments): on the safe side of the measurements by 0.05 on average, never more than 0.09 on the other side (one point, 1 mmol/L sulphate in 47 mmol/L NaNO₃, where every treatment is high). Not counted, but listed in the results table and in the checks: four seawaters diluted to salinity ≤ 2.1, where the measured barium is below what the authors’ own solubility product allows; twelve points that Paige (1990, Table 6) digitised from a graph of Felmy et al. (1990) — they lie 0.21 below the ion-pair model with the constants published from those very data, three of them below the concentration of the pair alone, so they do not represent the data behind the constants (counted, they would leave the analogue with the smallest deviation, 0.08 against 0.08 for no term and 0.12 for the ion pair, and put the envelope up to 0.18 on the unsafe side); twelve points of Lieser (1965) digitised by Paige, a series that Felmy et al. and Monnin consider too high — it is the only one above 30 mmol/kg of sulphate in a neutral medium, and at 0.23–1.1 mol/kg Na₂SO₄ the analogue lies 0.1–0.46 above it, the ion pair and the no-term treatment about 1.1 above it at the top: taken at face value it favours the analogue and says that in sulphate brines even the analogue may report an index too low by up to 0.46; twelve points of Paige in sulphuric acid of 0.45–6.2 mol/kg, where the suite dissolves 2 to 19 times the measured barium — outside the range of the model, the barite index must not be used in such acid. The default is unchanged by the second pass: the new counted points and both digitised series point the same way. Limits of the evidence: counted neutral media up to 29 mmol/kg sulphate (seawater) and 12.5 mmol/kg (Na₂SO₄), dilute sulphuric acid to 79 mmol/kg; 20–25 °C, 60 °C in dilute acid only; nothing measured was obtained for K₂SO₄ or MgSO₄ solutions, seawater concentrates, or sulphate-rich brines above 25 °C (the IUPAC–NIST Solubility Data Series has no volume on barium sulphate; the open oilfield and geothermal reports found hold simulations or precipitation tests, not equilibrium solubilities); the solids differ between laboratories (log Ksp −10.12 for the aged reagent of Savenko et al., −9.87 from the seeded tests of Boerlage). Consistency checks: free-ion solubilities of the treatments coincide within 0.001 in log₁₀ in 0–4 mol/kg NaCl once each uses its own log Ksp; BaSO₄(aq) at barite saturation equals K·Ksp (4.68·10⁻⁸ mol/kg) in every medium; free plus paired barium equals the total (10⁻¹²) and barium is conserved through precipitation; the closed-form band equals a recalculation of the speciation (5·10⁻⁷ SI). The barite index is also given under every other treatment (bariteBand) in the saturation table, in the table “Barite index: sensitivity to the Ba–SO₄ interaction” and, where the verdict depends on the treatment or the range exceeds 0.1 SI, in a warning', `Constants at 25 °C; measured support in sulphate media to 29 mmol/kg sulphate at 20–25 °C in neutral solutions and to 79 mmol/kg at 25 and 60 °C in dilute sulphuric acid; not valid in sulphuric acid of 0.45 mol/kg and more. The other treatments lie within −${-BARITE_BAND.lo} … +${BARITE_BAND.hi} SI of the default for the example waters and for seawater up to a twofold concentrate (free SO₄²⁻ ≤ ${BARITE_BAND.so4} mol/kg); wider (to 0.4–0.6 SI) for dilute sulphate-type waters and sulphate brines, where no measurement was obtained — reported with every run`, 'default confirmed against measured solubilities in sulphate media (two primary tables, two thesis tables, one series digitised from an open figure; two further second-hand digitised series listed, not counted); the ion-pair constants are from the abstract and first pages of their source only, whose own data were not read'],
  ['Other Pitzer analogues that remain: θ, ψ and the binaries of NH₄⁺ (from K⁺), Fe²⁺ and Mn²⁺ (from Mg²⁺) that have no value in the files above; H₃SiO₄⁻ (uses HCO₃⁻)', 'Assignment by chemical similarity', 'No independent check; the binaries with chloride and sulphate of these ions are taken from the files named above and tested against the NIST tables', 'Trace constituents only', 'analogue'],
  ['Debye–Hückel slope Aφ(T)', 'Grid of the LLNL EQ3/6 Pitzer file data0.ypf (0.3767, 0.3915, 0.4190, 0.4605 at 0, 25, 60, 100 °C)', 'Fit reproduces the four grid values within 0.0005', '0–100 °C', 'confirmed'],
  ['Carbonate, water, silicate, borate, HSO₄⁻ and HF dissociation; CO₂ Henry constant (ion-pair models)', 'USGS WATEQ4F database wateq4f.dat (Plummer & Busenberg 1982; Ball & Nordstrom 1991); borate log K as in MINTEQA2 v4', 'All coefficients of the six temperature functions and four log K/ΔH pairs compared', '0–90 °C', 'confirmed'],
  ['The same constants for the Pitzer and Bromley species set (pK₂ 10.339, pK₁ 6.337, HSO₄⁻ 1.979)', 'USGS PHRQPITZ/PHREEQC pitzer.dat; 25 °C values equal to data0.hmw', 'Corrected: the ion-pair values had been used with the Pitzer model', '25 °C exact, 0–90 °C by the shifted temperature function', 'corrected'],
  ['Ion-pair constants (CaSO₄°, MgSO₄°, NaSO₄⁻, KSO₄⁻, CaHCO₃⁺, MgHCO₃⁺, NaHCO₃°, NaCO₃⁻, CaCO₃°, MgCO₃°, CaOH⁺, MgOH⁺, CaF⁺, MgF⁺, BaSO₄°, SrSO₄°)', 'USGS WATEQ4F database wateq4f.dat (CaOH⁺ ΔH from MINTEQA2 v4); Pitzer-set CaCO₃°, MgCO₃°, MgOH⁺ from data0.hmw', '16 log K and 12 ΔH compared: identical after rounding; two Pitzer-set ΔH adjusted to pitzer.dat', 'I < 0.7 mol/kg', 'confirmed'],
  ['Ion-size å and b of the Truesdell–Jones and extended Debye–Hückel models', 'USGS WATEQ4F database wateq4f.dat', '18 ions compared; b of Ba²⁺, NH₄⁺, Fe²⁺, Mn²⁺, NO₃⁻, F⁻, OH⁻ and å of HPO₄²⁻ corrected', 'I < 1 mol/kg', 'corrected'],
  ['Ion-size default (å = 4, b = 0.041) for species without a tabulated entry', 'Model assumption', 'No source', 'Ion pairs and minor species only', 'unconfirmed'],
  ['log K(T) of calcite, aragonite, gypsum, anhydrite, barite, celestite, fluorite, amorphous silica, strontianite, witherite, siderite, dolomite', 'USGS WATEQ4F database wateq4f.dat', 'All analytic coefficients and ΔH compared: identical; silica now carries the water activity of SiO₂ + 2 H₂O = H₄SiO₄. Barite (log K −9.970 of this file; used with the Ca–SO₄ analogue and no-term treatments and with every model other than Pitzer — the default Pitzer treatment shifts it to −10.05, see the Ba–SO₄ row, where the deviations of the default are listed): in water 0.0107 mmol/kg against 0.0108 (Templeton 1960) and 0.0106 (Blount 1977). Primary tables of Blount (1977, Am. Mineral. 62, 942; open journal copy, read from page images): barite in water at 100, 500 and 1000 bar within 4.4 % (Table 3), at 60 and 100 °C within 7.9 % (Table 3), in 0.2 and 1.0 mol/kg NaCl 11 and 16 % above the measurements of Puchelt (1967) that Blount tabulates (Table 11) and 21 and 7 % below the Templeton (1960) table — with this log K the model lies between the two series. Templeton’s table is still second-hand (Appelo 2015) but agrees within 4 % with his points in Blount’s Fig. 9, digitised here (±0.01 in log₁₀). Celestite −6.63 with every model (celestite row)', '0–90 °C', 'confirmed'],
  ['log K of calcite and aragonite with the Pitzer model (−8.406, −8.219)', 'Harvie, Møller & Weare (1984) in data0.hmw; temperature function of pitzer.dat', 'Corrected (was the ion-pair value −8.480, −8.336); verified against the seawater solubility of Mucci (1983)', '25 °C exact', 'corrected'],
  ['log K of halite, sylvite, the Na/Mg/K/Ca sulphate and chloride salts, brucite, portlandite, magnesite, nesquehonite', 'Harvie, Møller & Weare (1984) in data0.hmw and USGS PHREEQC pitzer.dat (PHRQPITZ lineage)', '19 values at 25 °C compared: identical to 0.001 (mirabilite follows pitzer.dat, −1.214)', '25 °C', 'confirmed'],
  ['The same minerals with the ion-pair models (brucite −11.16, magnesite −8.03, nesquehonite −5.62, epsomite −2.14, mirabilite −1.11, thenardite −0.18, halite 1.58)', 'USGS WATEQ4F database wateq4f.dat', 'Added: the Pitzer-set values had been used with every model', 'I < 0.7 mol/kg', 'corrected'],
  ['Temperature dependence of sylvite, hexahydrite, bischofite (analytic) and kieserite (ΔH −29 kJ/mol)', 'USGS PHREEQC pitzer.dat (PHRQPITZ expressions; kieserite slope from the Appelo 2015 expression)', 'Replaced: the earlier ΔH of hexahydrite and kieserite were not found in a database', '0–100 °C, indicative', 'replaced'],
  ['SIT interaction coefficients ε(cation, anion)', 'OECD-NEA thermochemical database, 2020 update of the SIT tables (B-6, B-7); ThermoChimie sit.dat of USGS PHREEQC', '26 of 27 pairs identical in the NEA tables; 19 also in sit.dat; FeCl₂, MnCl₂, NaH₃SiO₄ and NH₄NO₃ (−0.06) added from sit.dat; against the NIST tables to 1 mol/kg: MnCl₂ 1.3 %, NH₄Cl 1.8 %, NH₄NO₃ 8.5 %, FeCl₂ 8.8 % in γ±', 'I ≤ 3 mol/kg', 'confirmed'],
  ['SIT ε(Sr²⁺, Cl⁻) = 0.10 kg/mol', 'Fitted in this work to the SrCl₂ activity coefficients of Goldberg & Nuttall (1978, Table 23) — the pair is not in the NEA tables', 'Weighted least squares on log γ± to I = 3 mol/kg: ε = 0.103, rms 0.004. The same fit gives 0.135, 0.198 and 0.057 for CaCl₂, MgCl₂ and BaCl₂ (NEA: 0.14, 0.19, 0.07)', 'I ≤ 3 mol/kg', 'fitted here'],
  ['Bromley model: salt constants B of 19 salts (15 of Bromley’s table, 4 re-derived here) and the individual-ion table (B₊, δ₊, B₋, δ₋) of 17 ions', 'Tables 1 and 2 of Bromley (1973, AIChE J. 19, 313–320) — article itself not read. Reproduction 1: Appendix 4.2 of Zemaitis, Clark, Rafal & Scrivner, Handbook of Aqueous Electrolyte Thermodynamics (DIPPR/AIChE 1986), pp. 170–174 (read). Reproduction 2: Tables 6.2 and 6.3 of Thomsen (2009, Electrolyte Solutions: Thermodynamics, Crystallization, Separation methods, lecture notes, Technical University of Denmark, pp. 49–50; open, read). Reproduction 3: the matrix BromleyData of the open-source Modelica library ElectrolyteMedia (Bremen & Mitsos, RWTH Aachen; github.com/andreasbremen/electrolytemedia, ElectrolyteMedia/Media/LiquidPhase/Common/MixtureSolutesData/package.mo, commit eaaf9ad, BSD 3-Clause; read). Re-derivation from primary evaluated data read first-hand: γ± tables of Hamer & Wu (1972, J. Phys. Chem. Ref. Data 1, 1047), Goldberg & Nuttall (1978, 7, 263), Goldberg (1981, 10, 671) and Rard, Wijesinghe & Wolery (2004, Lawrence Livermore report UCRL-JRNL-203290, Table 3 — open report version of J. Chem. Eng. Data 49, 1127)', 'Results do not depend on the unread article as far as the salt constants go: each of the 19 constants in use is re-derived by Bromley’s method (his equation, least squares on log γ± to I = 6 mol/kg or the end of the table) from independent data, and a table value is kept only if it reproduces those data with at most twice the standard deviation of the best fit (floor 0.005 in log γ±). 15 table values pass and stay in use (largest difference from the re-derived B: K₂SO₄ 0.010 kg/mol, 1.8 × the scatter, tabulated only to 0.69 mol/kg; CaCl₂ 0.0018; Mg(NO₃)₂ 0.0016; all others ≤ 0.0008 except MgSO₄ 0.006, where the one-constant equation itself misses a 2:2 salt by 0.04 in log γ±). 4 fail and are replaced by the re-derived constant: SrCl₂ 0.0847 → 0.0809 and BaCl₂ 0.0638 → 0.0606 (Goldberg & Nuttall 1978), Na₂CO₃ 0.0089 → −0.0073 (Goldberg 1981; the table value puts γ± 14 % high at 1 mol/kg and 24 % at 2 mol/kg) and K₂CO₃ 0.0372 → 0.0304. With the constants in use the rms deviation from the independent γ± data is at most 0.019 in log γ± (K₂SO₄; 0.011 for every other 1:1, 1:2 or 2:1 salt) and 0.049 for MgSO₄. Kind of data: NIST evaluations (primary evaluated tables, read) for 16 salts; the Livermore evaluation for Mg(NO₃)₂ (open report version, read); for MgSO₄ the Archer & Rard model values as tabulated by Miladinović et al. (2007) — an evaluated model table; for K₂CO₃ γ± from the Harvie–Møller–Weare parameters of this suite — a parametrisation, because no table of primary values was obtained (Goldberg 1981 judged the K₂CO₃ data too imprecise for one). RbI, on which the reproductions disagree (handbook −0.0108; ElectrolyteMedia +0.0108; not in the lecture notes): all 27 rows of Table 42 of Hamer & Wu give B = +0.0108 with a standard deviation of 0.0050 in log γ± — the printed magnitude and the printed σ (0.005); with the negative sign the table is missed by 0.064 (13 × the scatter, 31 % in γ± at 5 mol/kg). The sign is positive; the handbook entry is a misprint (RbI is not a salt of the suite). The three reproductions: all 19 table values identical in the handbook and in ElectrolyteMedia, 10 of them also in the lecture notes; ElectrolyteMedia differs from the handbook in LiOH, MgBr₂ and its Cs⁺ sums, none a salt or ion of this suite. Individual-ion table (used only for cation–anion pairs without a salt constant — minor species): identical in the handbook; H, Na, K, NH₄, Ca, Cl, NO₃, SO₄, CO₃ identical in the lecture notes; Mg, Sr, Ba, Mn, Fe, F, OH, HPO₄ each fixed by 4–8 ion-table sums of ElectrolyteMedia that reproduce B₊ + B₋ + δ₊δ₋ to 5·10⁻⁵. The ion table is not re-derived value by value: an independent global refit over 52 salts (next row) differs from it by up to 0.017 kg/mol in B₊ and B₋, so for the ion table the statement is “three concordant reproductions”, not independence of the article. δ of Mn²⁺ and Fe²⁺ are marked as estimates in the reprint. Hunt for the article or an author’s report version (not found): no open copy in OpenAlex; no report of Bromley’s in OSTI that tabulates the constants; his Office of Saline Water report No. 747 (1972, Properties of seawater and its concentrates, open, read) and NBS Report 10 002 (Wu & Hamer 1969, activity coefficients of the non-halide 1:1 electrolytes, the precursor of Hamer & Wu 1972; open, read) contain no B values. For a reader who holds the article: compare the 19 table values of BR_SALT and the sign of RbI with its Table 1, and the 17 (B, δ) pairs of BR_ION with its Table 2 — some 55 numbers', 'I ≤ 6 mol/kg; not for 2:2 salts', 'salt constants: independently re-derived from primary evaluated data for 17 of the 19 salts in use, from an evaluated model table for MgSO₄ and from a Pitzer parametrisation for K₂CO₃ — 15 table values confirmed within the fitting scatter, 4 replaced by the re-derived value; RbI sign settled by the data; table values identical in three reproductions (all secondary); ion table: three reproductions, not re-derived; article itself not read'],
  ['Bromley constants refitted in this work (BR_REFIT, BR_REFIT2): the basis of the row above', 'γ± tables of Hamer & Wu (1972), Goldberg & Nuttall (1978) and Goldberg (1981): 58 salts, weighted least squares on log γ± to I = 6 mol/kg (fitBromleyB); RbI and NH₄NO₃ on their complete tables, Na₂CO₃ (Goldberg 1981), Mg(NO₃)₂ (Rard, Wijesinghe & Wolery 2004), MgSO₄ and K₂CO₃ in BR_REFIT2; ion values by a global fit of B = B₊ + B₋ + δ₊δ₋ over 52 salts (fitBromleyIons)', BR_STAT, 'I ≤ 6 mol/kg', 'refitted here; in use for SrCl₂, BaCl₂, Na₂CO₃ and K₂CO₃'],
  ['Hydrous ferric oxide: site density, surface area, protonation and sorption constants', 'Dzombak & Morel (1990) and Swedlund & Webster (1999), read from the SURFACE_SPECIES block of USGS PHREEQC phreeqc.dat', '10 constants, 600 m²/g and 0.2 mol/mol compared: identical; solver reproduces PHREEQC example 8', '25 °C, I < 0.7 mol/kg', 'confirmed'],
  ['Interfacial energies, growth constants and antiscalant limits of the minerals', 'Order-of-magnitude engineering defaults', 'Not source-checked; adjustable through the kinetic inputs and the limit fields', 'Screening only', 'unconfirmed'],
];
const BA_SHORT = { pair: 'BaSO₄(aq) ion pair, Felmy et al. 1990', analogue: 'Ca–SO₄ analogue', none: 'no Ba–SO₄ term', pairw: 'BaSO₄(aq) ion pair, log Ksp fitted to pure water' };
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
  const ba = BARITE_MODELS[v.bariteModel] ? v.bariteModel : BARITE_MODEL, pairDK = Number.isFinite(v.dkBaPair) ? v.dkBaPair : 0;
  // Verdict basis for barite: every decision (status, limiting mineral, recovery limits) is taken on the conservative envelope —
  // the highest index among the published treatments of the Ba–SO4 interaction — unless the user selects the best estimate;
  // after a calibration against the user's own measurements (seBarite > 0) the envelope is the fitted index + 2 standard errors.
  const vb = { basis: v.bariteVerdict === 'best' ? 'best' : 'envelope', se: Math.max(0, Number.isFinite(v.seBarite) ? v.seBarite : 0) };
  const upOf = (eq) => (vb.basis === 'best' ? 0 : vb.se > 0 ? 2 * vb.se : bariteBand(eq)?.up ?? 0);
  const satV = (eq) => { const q = saturation(eq, P, dk, set), u = q.barite != null ? upOf(eq) : 0; return u ? [q, { ...q, barite: q.barite + u }] : [q, q]; }; // [best estimate, verdict basis]
  let raw = makeSolution({ ions: v.ions, T: v.T, pH: v.pH, model, bariteModel: ba, pairDK });
  const unmixed = raw;
  let other = null;
  if (v.mixOn && tds(cloneIons(v.mixIons)) > 0 && v.mixFrac > 0) { other = makeSolution({ ions: v.mixIons, T: v.mixT, pH: v.mixPH, model, bariteModel: ba, pairDK }); raw = mixSolutions(raw, other, clamp(v.mixFrac / 100, 0, 1)); raw.T = raw.eq.T; }
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
  const conc = useRo ? makeSolution({ ions: v.roBrine, T: v.T, pH: v.roBrinePH, model, bariteModel: ba, pairDK }) : concAt(R);
  const wall = useRo ? concentrateSolution(conc, beta, { co2: 'closed' }) : concAt(R, beta);
  // recovery sweep at the membrane wall
  const g0 = solutionToIons(feed).gPerKgw, Rmax = clamp(1 - (beta * g0) / 380, 0.3, 0.98), nRec = Math.max(4, Math.round(v.nRec));
  const Rs = linspace(0, Rmax, nRec), both = Rs.map((r) => satV(concAt(r, beta).eq)), sweep = both.map((q) => q[0]), sweepV = both.map((q) => q[1]);
  const maxRec = {};
  for (const id of set) {
    const ys = sweepV.map((q) => q[id] ?? -99), here = present(feed.eq, id);
    maxRec[id] = { plain: here ? cross(Rs, ys, 0) : null, as: here ? cross(Rs, ys, Math.max(lim[id] ?? 0, 0)) : null, present: here };
  }
  const pick = (key) => { let best = null; for (const id of set) { const r = maxRec[id][key]; if (maxRec[id].present && r != null && (best == null || r < best.r)) best = { id, r }; } return best || { id: null, r: Rmax }; };
  return { v, model, bariteModel: ba, vb, upOf, satV, sweepV, dk, lim, set, P, raw, unmixed, other, scm, ix, feed, dose, conc, wall, R, rej, beta, Rs, Rmax, sweep, maxRec, limitPlain: pick('plain'), limitAS: pick('as'), concAt, copt, useRo, cf: cfOf(R) };
}

// ---- barite treatments against measured solubility ----------------------------------------------------
/** Standard seawater of salinity S as molalities: the composition of Culberson, Latham & Bates (1978; SW_CLB, salinity 35) scaled on the water-mass basis. */
function seawaterTot(S = 35) {
  const f = (S / (1000 - S)) / (35 / 965), tot = new Float64Array(NM);
  for (const k of ['Na', 'K', 'Mg', 'Ca', 'Sr', 'SO4']) tot[mi(k)] = SW_CLB[k] * f;
  tot[mi('Cl')] = (SW_CLB.Na + SW_CLB.K + 2 * (SW_CLB.Mg + SW_CLB.Ca + SW_CLB.Sr - SW_CLB.SO4)) * f;
  return tot;
}
export const BARITE_TREATMENT_IDS = ['pairw', 'pair', 'analogue', 'none'];
let BEV = null;
/**
 * Every treatment of the Ba–SO4 interaction against the measured barite solubilities embedded in this file: pure water at
 * 25 °C, pure water at other temperatures and pressures, NaCl solutions, and the sulphate-bearing media (BARITE_SULPHATE).
 * Returns { points, media: [{ id, label, src, n, nPrimary, sulphate, counted, stat: { treatment: { bias, rms } } }], pooled, n,
 * sulphate, nSulphate, plain, nPlain, envelope: { bias, max, min, over } }; deviations are log10(model / measured); the
 * pooled statistics leave out the media named in BARITE_NOT_COUNTED; envelope is the lowest solubility among the published
 * treatments against the measurements in sulphate media. Computed once per session.
 */
export function bariteEvidence() {
  if (BEV) return BEV;
  const memo = new Map(), once = (k, f) => { if (!memo.has(k)) memo.set(k, f()); return memo.get(k); };
  const mk = (tot, T, B, alk = 0) => equilibrate({ T, model: 'pitzer', n: tot, alk, w: 1, pH: 7 }, { bariteModel: B });
  const nacl = (c) => { const n = new Float64Array(NM); n[mi('Na')] = c; n[mi('Cl')] = c; return n; };
  const sat = (key, tot, T, P, B, alk = 0) => once(key + '|' + T + '|' + P + '|' + B, () => { const r = solubility(mk(tot, T, B, alk), 'barite', { excess: 0.01, P }); return [r.m * 1000, r.sol.eq.tot[mi('Ba')] * 1000]; }); // mmol/kg water: dissolved from the solid, total at saturation
  const pts = [], add = (medium, src, kind, cond, meas, f) => pts.push({ medium, src, kind, cond, meas, model: Object.fromEntries(BARITE_TREATMENT_IDS.map((B) => [B, f(B)])) });
  for (const [src, x] of [['Templeton 1960', 0.0108], ['Davis & Collins 1971', 0.011], ['Blount 1977', 0.0106]]) add('w25', src, 'primary', '25 °C, 1 bar', x, (B) => sat('w', nacl(0), 25, 1, B)[0]);
  for (const [P, x] of BARITE_BLOUNT.P) add('wTP', 'Blount 1977, Table 3', 'primary', `25 °C, ${P} bar`, x, (B) => sat('w', nacl(0), 25, P, B)[0]);
  for (const [T, x] of BARITE_BLOUNT.T) add('wTP', 'Blount 1977, Table 3', 'primary', `${T} °C, 1 bar`, x, (B) => sat('w', nacl(0), T, 1, B)[0]);
  for (const [c, x] of BARITE_NACL.slice(1)) add('nacl', 'Templeton 1960', 'primary', `${c} mol/kg NaCl, 25 °C`, x, (B) => sat('n' + c, nacl(c), 25, 1, B)[0]);
  for (const [c, x] of BARITE_DC.slice(1)) add('nacl', 'Davis & Collins 1971', 'primary', `${c} mol/kg NaCl, 25 °C`, x, (B) => sat('n' + c, nacl(c), 25, 1, B)[0]);
  for (const [c, x] of BARITE_BLOUNT.nacl) add('nacl', 'Puchelt 1967 (Blount 1977, Table 11)', 'primary', `${c} mol/kg NaCl, 25 °C`, x, (B) => sat('n' + c, nacl(c), 25, 1, B)[0]);
  for (const [c, x] of BARITE_BLOUNT.fig9P.filter((r) => Math.abs(r[0] - 0.213) > 1e-9 && Math.abs(r[0] - 1.074) > 1e-9)) add('nacl', 'Puchelt 1967 (Blount 1977, Fig. 9, digitised)', 'primary', `${c} mol/kg NaCl, 25 °C`, x, (B) => sat('n' + c, nacl(c), 25, 1, B)[0]);
  for (const d of BARITE_SULPHATE) {
    if (d.ions) { add(d.medium, d.src, d.kind, d.cond, d.ba, (B) => once(`ro|${d.cf}|${B}`, () => solutionToIons(solubility(concentrateSolution(makeSolution({ ions: d.ions, T: d.T, pH: d.pH, model: 'pitzer', bariteModel: B }), d.cf, { co2: 'closed' }), 'barite', { excess: 1e-3 }).sol).ions.Ba * 1000)); continue; } // a water analysis in mg/L, concentrated by the factor cf: µg/L of barium on both sides
    const tot = d.S != null ? seawaterTot(d.S) : (() => { const n = new Float64Array(NM); for (const [k, x] of Object.entries(d.comp)) n[mi(k)] = x; return n; })(), kgw = d.S != null ? 1 - d.S / 1000 : 1;
    add(d.medium, d.src, d.kind, d.cond, d.ba, (B) => sat(d.S != null ? 'sw' + d.S : JSON.stringify(d.comp), tot, d.T, d.P ?? 1, B, d.alk ?? 0)[1] * 1e6 * (d.perKgWater ? 1 : kgw)); // nmol of barium per kg of solution (per kg of water where the source says so)
  }
  const LABEL = { w25: 'Pure water, 25 °C, 1 bar', wTP: 'Pure water, 60–100 °C and 100–1000 bar', nacl: 'NaCl solutions, 0.05–5 mol/kg, 25 °C', ...BARITE_SULPHATE_MEDIA };
  const stat = (rows) => Object.fromEntries(BARITE_TREATMENT_IDS.map((B) => { const d = rows.map((p) => Math.log10(p.model[B] / p.meas)); return [B, { bias: sum(d) / Math.max(d.length, 1), rms: Math.sqrt(sum(d.map((x) => x * x)) / Math.max(d.length, 1)) }]; }));
  const isS = (id) => !['w25', 'wTP', 'nacl'].includes(id), counted = pts.filter((p) => !BARITE_NOT_COUNTED.includes(p.medium)), sulph = counted.filter((p) => isS(p.medium)), plain = counted.filter((p) => !isS(p.medium));
  const media = Object.keys(LABEL).map((id) => { const rows = pts.filter((p) => p.medium === id); return { id, label: LABEL[id], src: [...new Set(rows.map((p) => p.src))].join('; '), n: rows.length, nPrimary: rows.filter((p) => p.kind === 'primary').length, sulphate: isS(id), counted: !BARITE_NOT_COUNTED.includes(id), stat: stat(rows) }; }).filter((q) => q.n);
  const env = sulph.map((p) => Math.log10(Math.min(...BARITE_PUBLISHED.map((B) => p.model[B])) / p.meas)); // the conservative envelope: the lowest solubility (highest index) among the published treatments
  return (BEV = { points: pts, media, pooled: stat(counted), n: counted.length, sulphate: stat(sulph), nSulphate: sulph.length, plain: stat(plain), nPlain: plain.length, envelope: { bias: sum(env) / Math.max(env.length, 1), max: Math.max(...env), min: Math.min(...env), over: env.filter((x) => x > 0.02).length } });
}

/** Note under the table “Barite index: sensitivity to the Ba–SO₄ interaction”: what the columns are, which index decides, and what the measurements in sulphate media say. */
function BARITE_NOTE(a, bOpt) {
  const ev = bariteEvidence(), s = ev.sulphate, sg = (x) => (x >= 0 ? '+' : '−') + fmt(Math.abs(x), 2);
  return `Barium is a trace ion, so each column is the index of this run shifted in closed form: a Ba–SO₄ binary changes ln γ(Ba²⁺) by 2·m(SO₄)·B(I), an ion pair lowers the free barium by the factor 1 + K·γ(Ba)·γ(SO₄)·m(SO₄); every treatment is used with its own barite log Ksp. Best estimate of this run: ${BA_SHORT[a.bariteModel]}. ${bOpt.se > 0 ? 'The range is ± 2 standard errors of the calibration against the user’s own measurements. ' : ''}${bOpt.basis === 'best' ? 'The verdict uses the best estimate (input “Barite verdict basis”).' : 'The verdict uses the highest index among the published treatments (ion pair with the constants of Felmy et al. 1990, Ca–SO₄ analogue, no Ba–SO₄ term) and the one selected — the conservative envelope.'} Against ${ev.nSulphate} measured barite solubilities in sulphate-bearing media (Na₂SO₄ solutions, seawater of salinity 5–35, reverse-osmosis feed and concentrate at 20–25 °C; sulphuric acid up to 79 mmol/kg at 25 and 60 °C) the mean of log₁₀(model/measured) and its root mean square are ${BARITE_TREATMENT_IDS.map((B) => `${sg(s[B].bias)} and ${fmt(s[B].rms, 2)} for “${BA_SHORT[B]}”`).join(', ')}; a positive value means the treatment dissolves more barium than was measured, i.e. its index is too low by that amount. The conservative envelope lies ${fmt(-ev.envelope.bias, 2)} SI above those measurements on average (from ${fmt(Math.max(0, ev.envelope.max), 2)} below to ${fmt(-ev.envelope.min, 2)} above). Per medium: table “Barite treatments against measured solubility”.`;
}
/** Results table: every treatment of the Ba–SO4 interaction against the measured barite solubilities held in this file (bariteEvidence). */
function BARITE_EVIDENCE_TABLE() {
  const ev = bariteEvidence(), r3 = (x) => +x.toFixed(3), cells = (st) => BARITE_TREATMENT_IDS.flatMap((B) => [r3(st[B].bias), r3(st[B].rms)]), D = BARITE_MODEL_DEFAULT;
  const best = (st) => BARITE_TREATMENT_IDS.reduce((p, B) => (st[B].rms < st[p].rms ? B : p)), M = Object.fromEntries(ev.media.map((q) => [q.id, q])), last = (id, B) => { const r = ev.points.filter((p) => p.medium === id).at(-1); return fmt(100 * (r.model[B] / r.meas - 1), 2); };
  return { title: 'Barite treatments against measured solubility', columns: ['Medium', 'Measurements', 'Points', ...BARITE_TREATMENT_IDS.flatMap((B) => [`${BA_SHORT[B]}: mean`, `${BA_SHORT[B]}: rms`])],
    rows: [...ev.media.map((q) => [q.label, q.src, q.n, ...cells(q.stat)]), ['Water and NaCl solutions together', '—', ev.nPlain, ...cells(ev.plain)], ['Sulphate-bearing media together (counted points)', '—', ev.nSulphate, ...cells(ev.sulphate)], ['All counted points', '—', ev.n, ...cells(ev.pooled)]],
    note: `Mean and root mean square of log₁₀(model solubility / measured solubility) for each treatment of the barium–sulphate interaction of the Pitzer model, each with its own barite log Ksp; +0.04 is a model 10 % above the measurement, and a treatment that dissolves too much barium reports a barite index too low by the same amount. In water and NaCl solutions the dissolved sulphate is 10⁻⁵–10⁻⁴ mol/kg and the treatments differ by their solubility products alone; the sulphate-bearing media are where they differ in substance. Measurements in sulphate media: (1) Jiang (1996, J. Solution Chem. 25, 105), barite in 0.3–7.9 mmol/kg Na₂SO₄ at 20 °C — the article was not read; its 12 points were taken from the vector drawing of Fig. 4-5 in García (2005, PhD thesis, Technical University of Denmark), reading resolution about 2 nmol/kg; (2) Savenko, Savenko & Pokrovsky (2019, Okeanologiya 59, 939, Table 1), recrystallised BaSO₄ after 13 months at 22 °C in Na₂SO₄–NaNO₃ solutions of ionic strength 0.05 and in three seawaters of salinity 35, barium by ICP-MS (± 3 %); (3) the same authors (2023, Okeanologiya 63, 745, Table 1), synthetic seawater diluted to salinity 0.35–30; (4) Boerlage (2001, PhD thesis, Wageningen University / IHE Delft, Tables 2.1 and 2.2), feed and concentrates at 80 and 90 % recovery of a reverse-osmosis pilot plant on pretreated Rhine water at 25 °C, seeded with BaSO₄ (± 10 %); (5) Paige (1990, PhD thesis, McMaster University, Tables 7 and 8), ¹³³Ba-labelled barite in sulphuric acid at 25 °C (five months, from undersaturation) and 60 °C (from supersaturation) — counted up to 79 mmol/kg; from 0.45 mol/kg the suite dissolves 2 to 19 times the measured barium and the points are listed only (the model is outside its range in such acid); (6) points that Paige digitised from the graphs of Felmy et al. (1990) and Lieser (1965), his Table 6 — listed only: the first contradict the constants published from them, the second is a series that later authors consider too high (it is nevertheless the only one above 30 mmol/kg of sulphate in a neutral medium: there the analogue is closest and still up to 0.46 above it). The seawater is modelled with the composition of Culberson, Latham & Bates (1978) scaled to the salinity; the suite holds no Ba–NO₃ binary (with the values of Pitzer & Mayorga 1973 every treatment would lie about 0.02 lower in the Na₂SO₄–NaNO₃ series). Not counted: the four seawaters diluted to salinity ≤ 2.1, where every treatment is ${fmt(100 * (10 ** Math.min(...BARITE_TREATMENT_IDS.map((B) => M.swd.stat[B].bias)) - 1), 2)}–${fmt(100 * (10 ** Math.max(...BARITE_TREATMENT_IDS.map((B) => M.swd.stat[B].bias)) - 1), 2)} % above the measurement on average and the measured ion-activity product falls below the solubility product the authors derive from their own Na₂SO₄ series. Reading: the Ca–SO₄ analogue has the smallest root-mean-square deviation in the sulphate-bearing media (${fmt(ev.sulphate.analogue.rms, 2)}; best treatment there: ${BA_SHORT[best(ev.sulphate)]}) and over all counted points (${fmt(ev.pooled.analogue.rms, 2)}; best: ${BA_SHORT[best(ev.pooled)]}); it follows the Na₂SO₄ series of Jiang within ${fmt(M.jiang.stat.analogue.rms, 2)}. The ion pair with the constants of Felmy et al. (1990) fits the two series of Savenko et al. as well or better (those authors use its association constant to reduce their data) but is ${last('jiang', 'pair')} % high at 7.9 mmol/kg Na₂SO₄; no Ba–SO₄ term is ${last('jiang', 'none').replace('-', '')} % low there. The ion pair combined with the log Ksp fitted to pure water is ${fmt(100 * (10 ** ev.sulphate.pairw.bias - 1), 2)} % high on average in sulphate media (${last('jiang', 'pairw')} % at 7.9 mmol/kg Na₂SO₄): the association constant 2.72 belongs with the log Ksp −10.05 it was derived with, and its combination with a pure-water log Ksp is not supported by these data. The default is therefore the ${BA_SHORT[D]}. The two laboratories that bracket the result differ in the solid: Savenko et al. find a solubility product of 7.57·10⁻¹¹ at 22 °C (log −10.12) for their aged reagent, Boerlage 1.34·10⁻¹⁰ (log −9.87) from seeded tests of 3–24 h in a natural water with 1–2 mg/L of organic carbon; the default is ${fmt(100 * (1 - 10 ** M.ro.stat[D].bias), 2)} % below Boerlage and ${fmt(100 * (1 - 10 ** M.sw.stat[D].bias), 2)} % below Savenko’s seawater on average, i.e. on the safe side of both. Still unread: the barite-in-Na₂SO₄ data of Felmy, Rai & Amonette (1990), which according to García (2005) exist only as plots, and the article of Jiang (1996) with its series at 0–80 °C.` };
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
 * Hot spot on one membrane of a spacer-channel solution: the peak cell value and the largest average of the wall
 * concentration over a patch of fixed physical length Lp, both taken over the permeating membrane (outside the filament
 * footprints) downstream of x0. The wall profile is integrated exactly as a piecewise-constant function, so the patch
 * measure does not depend on how the cells happen to fall under the window.
 */
export function wallHotSpot(f, top, Lp, x0 = 0, cw = top ? f.cwT : f.cwB) {
  const blk = top ? f.blockT : f.blockB, dx = f.dx, nx = f.nx, iv = [], cum = [0];
  // open intervals: outside the footprints and outside runs of blocked wall cells
  const cuts = (f.feet || []).filter((q) => q.top === top).map((q) => [q.x - q.a, q.x + q.a]);
  for (let i = 0; i < nx; i++) if (blk[i]) { let k = i; while (k + 1 < nx && blk[k + 1]) k++; cuts.push([i * dx, (k + 1) * dx]); i = k; }
  cuts.sort((p, q) => p[0] - q[0]);
  let s = x0;
  for (const [c0, c1] of cuts) { if (c1 <= s) continue; if (c0 > s + 1e-12 * f.L) iv.push([s, c0]); s = Math.max(s, c1); }
  if (f.L > s + 1e-12 * f.L) iv.push([s, f.L]);
  for (let i = 0; i < nx; i++) cum.push(cum[i] + cw[i] * dx);
  const I = (x) => { const t = clamp(x / dx, 0, nx), i = Math.min(nx - 1, Math.floor(t)); return cum[i] + cw[i] * (t - i) * dx; };
  let patch = 0, xPatch = x0, peak = 0, xPeak = x0, iPeak = -1;
  for (const [lo, hi] of iv) {
    for (let i = Math.max(0, Math.floor(lo / dx)); i < nx && i * dx < hi; i++) if (!blk[i] && Math.min(hi, (i + 1) * dx) - Math.max(lo, i * dx) > 1e-6 * dx && cw[i] > peak) { peak = cw[i]; xPeak = f.x[i]; iPeak = i; }
    const w = Math.min(Lp, hi - lo), cand = [lo, hi - w];
    for (let i = Math.ceil(lo / dx); i * dx <= hi; i++) cand.push(i * dx, i * dx - w);
    for (const c of cand) if (c >= lo - 1e-12 * f.L && c + w <= hi + 1e-12 * f.L) { const q = (I(c + w) - I(c)) / w; if (q > patch) { patch = q; xPatch = c + 0.5 * w; } }
  }
  return { peak: peak || 1, xPeak, iPeak, patch: patch || peak || 1, xPatch, open: iv };
}
/**
 * Discretisation error of a quantity on three systematically refined grids (h and f ordered fine → coarse) with `gci`.
 * The observed order is reported as found; for the extrapolation and the grid-convergence index it is limited to the
 * range 0.5–2 (the scheme is at most second order, and a stair-step boundary makes very high observed orders an
 * accident of the sequence). A sequence that is not monotone is not extrapolated: the fine-grid value stands and the
 * index is the safety factor times half the spread of the three values.
 */
export function gridConvergence(h, f, Fs = 1.25) {
  const g = gci(h, f, Fs), r21 = h[1] / h[0], mono = g.type !== 'oscillatory', flat = g.type === 'grid-insensitive';
  const p = flat ? 2 : clamp(Number.isFinite(g.p) ? g.p : 1, 0.5, 2), f1 = f[0], den = r21 ** p - 1;
  const value = mono && !flat ? f1 + (f1 - f[1]) / den : f1, half = 0.5 * (Math.max(...f) - Math.min(...f));
  const err = mono ? Math.abs(f1 - f[1]) / den : Math.max(half, Math.abs(f1 - f[1]) / den);
  return { value, gci: (Fs * err) / Math.max(Math.abs(f1), 1e-300), pObs: Number.isFinite(g.p) ? g.p : 0, p, monotone: mono, type: g.type, change21: f[1] !== 0 ? f1 / f[1] - 1 : 0 };
}
/**
 * Grid-convergence study of the filament hot spot: the same spacer-channel problem (a short section of `nSec` filament
 * spacings) on three grids whose cell counts per spacing are multiples of ten (0.6, 0.8 and 1.0 times `nFine`), so that the
 * filament axes fall on cell faces on every grid and the stair-step outline changes systematically. Returns, per grid, the
 * peak and the patch-averaged hot spot, their Richardson estimates, the largest concentration anywhere (to compare
 * with the osmotic ceiling) and the largest flux found through covered membrane (zero when the footprint is respected).
 */
export async function spacerHotSpotStudy(par, { nFine = 50, nSec = 2, Lp = 1e-4, tol = 3e-4, scalIter = 600 } = {}, ctx) {
  const n3 = 10 * Math.max(5, Math.round(nFine / 10)), ns = [0.6 * n3, 0.8 * n3, n3].map((x) => 10 * Math.round(x / 10)), grids = [];
  for (let k = 0; k < 3; k++) {
    ctx?.progress?.(0.93 + 0.02 * k, `Hot-spot grid study: grid ${k + 1} of 3…`);
    if (ctx?.tick) await ctx.tick();
    const f = await spacerChannelCFD({ ...par, nFil: nSec, nxFil: ns[k], ny: 2 * Math.round(0.4 * ns[k]), tol, scalIter, maxIter: 400, solver: { alphaU: 0.85 } }, ctx), x0 = f.L / nSec;
    const B = wallHotSpot(f, false, Lp, x0), T = wallHotSpot(f, true, Lp, x0), pk = B.peak >= T.peak ? B : T, pa = B.patch >= T.patch ? B : T;
    let cMax = 0, jFoot = 0, nFoot = 0, cbHot = 1;
    for (let i = 0; i < f.nx; i++) { cMax = Math.max(cMax, f.cwB[i], f.cwT[i]); for (const [pf, J, blk] of [[f.permB, f.JB, f.blockB], [f.permT, f.JT, f.blockT]]) if (pf[i] < 1) { jFoot = Math.max(jFoot, blk[i] ? Math.abs(J[i]) : J[i] - pf[i] * f.Jopen); nFoot++; } } // flux beyond what the open part of a footprint cell can carry
    for (const row of f.field) for (const c of row) if (c > cMax) cMax = c;
    cbHot = f.cb[clamp(Math.round(pa.xPatch / f.dx - 0.5), 0, f.nx - 1)];
    grids.push({ nxFil: ns[k], nx: f.nx, ny: f.ny, dx: f.dx, dy0: f.dy[0], hRep: Math.sqrt(f.dx * f.dy[0]), peak: pk.peak, xPeak: pk.xPeak, peakTop: pk === T, patch: pa.patch, xPatch: pa.xPatch, patchTop: pa === T, cbHot, cMax, jFoot, nFoot, Jmean: f.Jmean, converged: f.converged, iters: f.iters, balance: f.balance.out / f.balance.in, cCap: f.cCap, f: k === 2 ? f : null });
  }
  const fine = grids.slice().reverse(), h = fine.map((q) => q.hRep), cap = grids[2].cCap, est = (key) => { const c = gridConvergence(h, fine.map((q) => q[key])); c.capped = !!cap && c.value > cap; if (c.capped) c.value = cap; return c; }; // the exact solution cannot exceed the osmotic ceiling, so neither may its estimate
  return { grids, Lp, nSec, peak: est('peak'), patch: est('patch'), cCap: cap, cMax: Math.max(...grids.map((q) => q.cMax)), jFoot: Math.max(...grids.map((q) => q.jFoot)), nFoot: grids[2].nFoot, cbHot: grids[2].cbHot, fine: grids[2].f, contactHalfWidth: grids[2].f.contactHalfWidth };
}
/**
 * Scaling in a spacer-filled section at the concentrate end of the channel: Navier–Stokes + salt transport with
 * permeating membranes (suite 4 solver) → local wall concentration → speciation engine → saturation-index map,
 * deposition flux and hot spots; compared with the open-slit boundary-layer march and with film theory. The hot spot
 * next to a filament is then resolved on three refined grids (spacerHotSpotStudy) and, on request, limited by the
 * growth of the scale itself.
 */
export async function spacerSection(v, a, m, kinId, ctx) {
  const { feed, P, dk } = a, h = Math.max(v.rtH ?? 0.71, 0.05) * 1e-3, arr = v.cfdArr || 'zigzag', lm = clamp(v.cfdLm ?? 3, 0.5, 20) * 1e-3, nFil = clamp(Math.round(v.cfdNFil ?? 6), 2, 20);
  const cfSec = m.cfIn * m.f.cb[m.f.cb.length - 1], uSec = Math.max(m.u * (1 - m.f.recovery), 1e-4), Ssec = Math.min(m.io.salinity * cfSec, 250), T = v.T;
  const rho = density(T, Ssec), mu = viscosity(T, Ssec), Dm = diffusivityNaCl(T, Math.min(Ssec, 200)), piOf = (c) => osmoticPressure(T, clamp(Ssec * c, 0, 260)), dPa = Math.max(v.P ?? 0, 0) * 1e5;
  const osm = v.cfdOsm !== false && dPa - piOf(1) * a.rej > 0.1 * dPa, Lp = clamp(v.cfdPatch ?? 100, 10, 1000) * 1e-6;
  const par = { h, u0: uSec, vw: m.vw, D: Dm, rej: a.rej, rho, mu, arr, lm, df: clamp((v.cfdDf ?? 50) / 100, 0.1, 0.85) * h, contact: clamp((v.cfdContact ?? 15) / 100, 0, 0.5), dP: dPa, pi: osm ? piOf : null };
  const f = await spacerChannelCFD({ ...par, nFil, nxFil: v.cfdNxFil ?? 24, ny: v.cfdNyFull ?? 32, tol: 1e-3, maxIter: 400, solver: { alphaU: 0.85 } }, ctx);
  const gs = v.cfdGci !== false ? await spacerHotSpotStudy(par, { nFine: v.cfdGciN ?? 50, Lp }, ctx) : null;
  // speciation engine on the local wall composition: bulk analysis × local concentration factor (tabulated on 9 factors)
  const set = a.set.filter((id) => present(feed.eq, id)), cMax = Math.max(1.02, ...f.cwB, ...f.cwT, ...f.field.flat(), gs ? gs.cMax : 1, f.cCap || 1), cfs = logspace(cfSec * 0.98, cfSec * cMax * 1.02, 9), lc = cfs.map(Math.log);
  const tab = cfs.map((cf) => saturation(concentrateSolution(feed, cf, a.copt).eq, P, dk, set)), id0 = kinId && set.includes(kinId) ? kinId : set.reduce((b, id) => (b == null || tab[8][id] / MINERALS[id]._nu > tab[8][b] / MINERALS[b]._nu ? id : b), null);
  const siOf = (id, c) => interp1(lc, tab.map((q) => q[id]), Math.log(Math.max(cfSec * c, 1e-9))), depOf = (si) => { const k = si > 0 ? nucleationKinetics(id0, si, T, v) : null; return k ? k.flux * 24 : 0; };
  const i0 = Math.round(f.nx / nFil), side = (cw, blk, pf, J, tau, top) => { // statistics over the permeating membrane downstream of the entrance spacing
    const xs = [], c = [], si = [], dep = [], jj = [], tt = [];
    for (let i = 0; i < f.nx; i++) if (!blk[i] && pf[i] >= 0.5) { xs.push(f.x[i]); c.push(cw[i]); const s0 = siOf(id0, cw[i]); si.push(s0); dep.push(depOf(s0)); jj.push(J[i]); tt.push(Math.abs(tau[i])); }
    const k0 = xs.findIndex((x) => x >= f.x[Math.min(i0, f.nx - 1)]), st = k0 < 0 ? 0 : k0, cs = c.slice(st), iPk = st + cs.indexOf(Math.max(...cs)), hs = wallHotSpot(f, top, Lp, f.L / nFil);
    return { x: xs, c, si, dep, J: jj, tau: tt, cMean: sum(cs) / cs.length, c95: pctl(cs, 0.95), cPeak: c[iPk], xPeak: xs[iPk], cPatch: hs.patch, xPatch: hs.xPatch, siPeak: si[iPk], siMean: sum(si.slice(st)) / cs.length, depMean: sum(dep.slice(st)) / cs.length, depPeak: Math.max(...dep.slice(st)), tauPeak: tt[iPk], tauMean: sum(tt.slice(st)) / cs.length, share: cs.filter((x) => x > 1.1 * (sum(cs) / cs.length)).length / cs.length };
  };
  const B = side(f.cwB, f.blockB, f.permB, f.JB, f.tauB, false), Tp = side(f.cwT, f.blockT, f.permT, f.JT, f.tauT, true), cbMean = sum(f.cb.slice(i0)) / (f.nx - i0), hot = B.cPeak >= Tp.cPeak ? B : Tp;
  // references: open-slit boundary-layer march over the same section with the same mean flux, and film theory with the spacer Sherwood correlation
  const mar = channelCFD({ L: f.L, H: h / 2, u0: uSec, vw: f.Jmean, D: Dm, rej: a.rej, nx: 120, ny: v.cfdNy ?? 30, visc: () => mu }), marSI = mar.cw.map((c) => siOf(id0, c)), ms = mar.x.map((x, i) => i).filter((i) => mar.x[i] >= f.x[Math.min(i0, f.nx - 1)]);
  const marMean = sum(ms.map((i) => mar.cw[i])) / ms.length, marPeak = Math.max(...mar.cw);
  const c1 = channel1D({ T, c0: 1, propMode: 'custom', rho, mu: mu * 1000, Dsalt: Dm * 1e9, piCoef: 0, H: h * 1000, Uin: uSec, geom: 'spacer', arr, nFil, lm: lm * 1000, L: f.L * 1000, A: 0, B: 0, dPtm: 0 });
  const e = Math.exp(Math.min(8, f.Jmean / c1.k)), film = e / (1 + (1 - a.rej) * (e - 1));
  const mineral = (c) => Object.fromEntries(set.map((id) => [id, siOf(id, c)]));
  // grid-converged hot spot carried to the end of the full section: the polarisation of the study (hot spot ÷ local bulk)
  // times the highest bulk concentration of the section, never above the osmotic ceiling
  let hs = null;
  if (gs) {
    const cbEnd = Math.max(...f.cb.slice(i0)), lift = (c) => { const x = (c / gs.cbHot) * cbEnd; return f.cCap ? Math.min(x, f.cCap) : x; };
    hs = { cPatch: lift(gs.patch.value), cPeak: lift(gs.peak.value), cbEnd, beta: gs.patch.value / gs.cbHot };
    hs.si = mineral(hs.cPatch); hs.siPeak = mineral(hs.cPeak);
    if (v.cfdSink && id0 && hs.si[id0] > 0) hs.sink = await scaleSink(v, a, par, gs, { id0, cfSec, T, P, dk, Lp }, ctx);
  }
  return { f, id0, set, cfSec, uSec, Ssec, rho, mu, Dm, osm, B, T: Tp, hot, cbMean, mar, marSI, marMean, marPeak, film, c1, siOf, depOf, gs, hs, Lp, cCap: f.cCap, siCap: f.cCap ? mineral(f.cCap) : null, siBulk: mineral(cbMean), siWallMean: mineral(0.5 * (B.cMean + Tp.cMean)), siHot95: mineral(Math.max(B.c95, Tp.c95)), siPeak: mineral(hot.cPeak), siFilm: mineral(cbMean * film), siMarch: mineral(marMean), Re: (rho * uSec * 2 * h) / mu };
}
/**
 * Supersaturation at the hot spot limited by the growth of the scale: the lattice ion in shortest supply is transported
 * as a second species that the membrane rejects completely and that the wall consumes at N = k_r (c_w − c_sat). k_r is the
 * secant of the suite's growth law G = k_g (S − 1)² (nucleationKinetics) between saturation and the hot-spot state and is
 * iterated until the state it was evaluated at is the state it produces; c_sat and the saturation index of the depleted
 * water come from the speciation engine with the mineral withdrawn from the hot-spot water.
 */
async function scaleSink(v, a, par, gs, { id0, cfSec, T, P, dk, Lp }, ctx) {
  const M = MINERALS[id0], cH = gs.patch.value, sol = concentrateSolution(a.feed, cfSec * cH, a.copt), st = M._st.map(([i, nu]) => [i, nu, sol.n[i] / sol.w / nu]).sort((p, q) => p[2] - q[2])[0];
  const xiMax = st[2], qs = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9], sis = qs.map((q) => saturationIndex(withdraw(sol, id0, q * xiMax).eq, id0, P, dk[id0] || 0));
  const siQ = (q) => interp1(qs, sis, clamp(q, 0, 0.9)), qSat = sis[8] < 0 ? brent(siQ, 0, 0.9, 1e-10) : 0.9, csat = cH * (1 - qSat);
  const solIn = concentrateSolution(a.feed, cfSec, a.copt), cL0 = (solIn.n[st[0]] / solIn.w) * solutionToIons(solIn).kgwPerL * 1000; // mol of the limiting ion per m³ at the section inlet
  const rate = (si) => { const k = si > 0 ? nucleationKinetics(id0, si, T, v) : null; return k ? (st[1] * k.G * rhoMolar(id0)) / cL0 : 0; }; // relative concentration × m/s
  const nMid = gs.grids[1].nxFil, hist = [];
  let c2 = cH, si = sis[0], kr = 0, fS = null;
  for (let it = 0; it < 4; it++) {
    kr = rate(si) / Math.max(c2 - csat, 1e-9);
    ctx?.progress?.(0.985, 'Scale growth at the hot spot…');
    fS = await spacerChannelCFD({ ...par, nFil: gs.nSec, nxFil: nMid, ny: 2 * Math.round(0.4 * nMid), tol: 3e-4, scalIter: 600, maxIter: 400, solver: { alphaU: 0.85 }, precip: { kr, csat } }, ctx);
    const x0 = fS.L / gs.nSec, hB = wallHotSpot(fS, false, Lp, x0, fS.scaleB), hT = wallHotSpot(fS, true, Lp, x0, fS.scaleT), c2n = Math.max(hB.patch, hT.patch), siN = siQ(1 - Math.min(1, c2n / cH));
    hist.push({ kr, c2: c2n, si: siN });
    const done = Math.abs(siN - si) < 0.005;
    c2 = c2n; si = siN;
    if (done) break;
  }
  const x0 = fS.L / gs.nSec, salt = Math.max(wallHotSpot(fS, false, Lp, x0).patch, wallHotSpot(fS, true, Lp, x0).patch);
  return { id: id0, ion: MAST[st[0]], qSat, csat, kr, c2, si, si0: sis[0], hist, cMid: salt, flux: (rate(si) * cL0 / st[1]) * M.mw * 86400 /* g/m²·d */, iterations: hist.length };
}
async function spacerScaling(v, a, X, ctx) {
  let q;
  try { q = await spacerSection(v, a, X.march, X.kinId, ctx); } catch (err) { X.W.push({ level: 'warn', msg: `The Navier–Stokes model of the spacer section could not be solved (${err.message}); the open-channel march stands.` }); return; }
  const { f, id0, B, T: Tp, hot } = q, M = id0 ? MINERALS[id0] : null, nm = M ? M.name : 'mineral', mm = (xs) => xs.map((x) => x * 1000), arrName = { zigzag: 'zigzag filaments', cavity: 'filaments on one membrane', submerged: 'mid-channel filaments', none: 'no filaments' }[v.cfdArr || 'zigzag'];
  const siF = f.field.map((row, j) => row.map((c, i) => (f.mask[j][i] ? 0 : clamp(q.siOf(id0, c), -12, 12))));
  X.PL.push({ type: 'field', title: `Spacer-filled section: ${nm.toLowerCase()} saturation index (Navier–Stokes model)`, xlabel: 'Distance along the section (mm)', ylabel: 'Height above the lower membrane (mm)', zlabel: 'SI', zunit: '', x: mm(f.x), y: mm(f.y), z: siF, mask: f.mask, cmap: 'turbo', contours: 8, shapes: f.shapes.map((sh) => ({ ...sh, x: mm(sh.x), y: mm(sh.y) })), markers: [{ x: hot.xPeak * 1000, y: hot === B ? 0 : f.h * 1000, label: 'hot spot' }],
    note: `Membranes at the bottom and the top, flow from left to right, ${arrName}. The salt rejected by the permeating membranes accumulates in the slow fluid next to and behind the filaments; the saturation index follows from the speciation engine at the local concentration.` });
  X.PL.push({ type: 'line', title: `Spacer-filled section: ${nm.toLowerCase()} saturation index at the membranes`, xlabel: 'Distance along the section (mm)', ylabel: 'Saturation index', series: [{ name: 'Lower membrane (Navier–Stokes, spacer)', x: mm(B.x), y: B.si }, { name: 'Upper membrane (Navier–Stokes, spacer)', x: mm(Tp.x), y: Tp.si }, { name: 'Open slit (boundary-layer march)', x: mm(q.mar.x), y: q.marSI, dash: true }, { name: 'Bulk (mixing-cup)', x: mm(f.x), y: f.cb.map((c) => q.siOf(id0, c)), dash: true }], hlines: [{ y: q.siFilm[id0], label: 'film theory' }, { y: 0, label: 'saturation' }], note: 'Gaps in the membrane curves are the footprints of the filaments, where the membrane is covered and does not permeate.' });
  X.PL.push({ type: 'line', title: `Spacer-filled section: potential ${nm.toLowerCase()} deposition flux and permeate flux`, xlabel: 'Distance along the section (mm)', ylabel: 'Deposition (g/m²·d) · flux (L/m²·h)', series: [{ name: 'Deposition, lower membrane', x: mm(B.x), y: B.dep }, { name: 'Deposition, upper membrane', x: mm(Tp.x), y: Tp.dep }, { name: 'Deposition, open slit', x: mm(q.mar.x), y: q.marSI.map(q.depOf), dash: true }, { name: 'Permeate flux, lower membrane (L/m²·h)', x: mm(B.x), y: B.J.map((j) => j * 3.6e6), dash: true }] });
  const cf = q.cfSec, st = (b) => [b.cMean, b.c95, b.cPeak];
  X.TB.push({ title: 'Spacer-filled channel section (Navier–Stokes, finite-volume solver of suite 4)', columns: ['Quantity', 'Navier–Stokes, lower membrane', 'Navier–Stokes, upper membrane', 'Open-slit march', 'Film theory', 'Unit'], rows: [
    ['Polarisation factor c wall / c section inlet: mean', B.cMean, Tp.cMean, q.marMean, q.cbMean * q.film, '–'], ['Polarisation factor: 95th percentile', B.c95, Tp.c95, pctl(q.mar.cw, 0.95), null, '–'], ['Polarisation factor: peak (hot spot)', B.cPeak, Tp.cPeak, q.marPeak, null, '–'],
    ['Wall concentration factor versus the feed: mean', cf * B.cMean, cf * Tp.cMean, cf * q.marMean, cf * q.cbMean * q.film, '×'], [`${nm} saturation index: mean`, B.siMean, Tp.siMean, q.siMarch[id0], q.siFilm[id0], ''], [`${nm} saturation index: hot spot`, B.siPeak, Tp.siPeak, q.siOf(id0, q.marPeak), null, ''],
    ['Hot-spot position', B.xPeak * 1000, Tp.xPeak * 1000, f.L * 1000, null, 'mm'], ['Wall shear stress at the hot spot / mean', `${fq(B.tauPeak, 3)} / ${fq(B.tauMean, 3)}`, `${fq(Tp.tauPeak, 3)} / ${fq(Tp.tauMean, 3)}`, null, null, 'Pa'], ['Membrane more than 10 % above the mean wall concentration', 100 * B.share, 100 * Tp.share, null, null, '% of open area'],
    [`Potential ${nm.toLowerCase()} deposition flux: mean`, B.depMean, Tp.depMean, sum(q.marSI.map(q.depOf)) / q.marSI.length, q.depOf(q.siFilm[id0]), 'g/m²·d'], [`Potential ${nm.toLowerCase()} deposition flux: peak`, B.depPeak, Tp.depPeak, q.depOf(q.siOf(id0, q.marPeak)), null, 'g/m²·d'],
    ['Mean permeate flux', sum(B.J) / B.J.length * 3.6e6, sum(Tp.J) / Tp.J.length * 3.6e6, f.Jmean * 3.6e6, f.Jmean * 3.6e6, 'L/m²·h']],
    note: `Section of ${fq(f.L * 1000, 3)} mm (${v.cfdNFil ?? 6} filament spacings of ${fq((v.cfdLm ?? 3), 3)} mm) at the concentrate end: inlet at ${fq(cf, 4)} times the feed concentration (${fq(q.Ssec, 3)} g/kg), ${fq(q.uSec, 3)} m/s, channel Reynolds number ${fq(q.Re, 3)}. Grid ${f.nx} × ${f.ny} cells, ${f.converged ? 'converged' : 'NOT converged'} in ${f.iters} iterations, salt balance error ${fq(Math.abs(f.balance.out / f.balance.in - 1), 2)}. ${q.osm ? `Solution–diffusion walls with A = ${fq(f.A * 3.6e11, 3)} L/m²·h·bar at ${fq(v.P, 3)} bar, so the flux falls where the wall concentration rises.` : 'Uniform permeate flux (the applied pressure does not exceed the osmotic pressure by enough for the flux law).'} Constant density and viscosity; molecular diffusivity ${fq(q.Dm, 3)} m²/s without a mixing factor. Statistics exclude the first filament spacing (entrance). The march and film-theory columns use the mean flux of the Navier–Stokes solution; film theory uses the spacer Sherwood correlation of suite 4 (Sh = ${fq(q.c1.Sh, 3)}). ${f.contactHalfWidth > 0 ? `Each wall-touching filament covers the membrane over a flattened contact of ±${fq(f.contactHalfWidth * 1e6, 3)} µm, where the flux is zero.` : ''} The values of this table are those of the section grid; the hot spot next to a filament needs the finer grids of the convergence study below.` });
  X.TB.push({ title: 'Saturation indices in the spacer-filled section', columns: ['Mineral', 'Bulk', 'Wall, mean (Navier–Stokes)', 'Wall, 95th percentile', 'Wall, hot spot', 'Wall, open-slit march (mean)', 'Wall, film theory'], rows: q.set.map((id) => [MINERALS[id].name, q.siBulk[id], q.siWallMean[id], q.siHot95[id], q.siPeak[id], q.siMarch[id], q.siFilm[id]]), note: 'Speciation engine evaluated on the bulk analysis scaled by the local concentration factor (nine tabulated factors, interpolated in ln CF).' });
  X.BAL.push({ name: 'Spacer section (Navier–Stokes): salt entering vs leaving (relative)', in: 1, out: f.balance.out / f.balance.in });
  const gs = q.gs, hs = q.hs;
  if (gs) {
    const pc = (x) => 100 * x, row = (g) => [`${g.nx} × ${g.ny}`, g.dx * 1e6, g.dy0 * 1e6, g.peak, g.patch, g.cMax, g.Jmean * 3.6e6, g.converged ? 'yes' : 'no'], est = (c) => `${fq(c.value, 5)} ± ${fq(pc(c.gci), 2)} %${c.capped ? ' (at the ceiling)' : ''}`;
    X.TB.push({ title: 'Hot spot next to a filament: grid-convergence study', columns: ['Grid (cells)', 'Axial cell (µm)', 'Wall cell height (µm)', 'Peak c wall / c inlet', `Mean over the hottest ${fq(gs.Lp * 1e6, 3)} µm patch`, 'Largest concentration anywhere', 'Mean flux (L/m²·h)', 'Converged'],
      rows: [...gs.grids.map(row), ['Richardson estimate ± grid-convergence index', null, null, est(gs.peak), est(gs.patch), null, null, null], ['Observed order of convergence', null, null, fq(gs.peak.pObs, 3), fq(gs.patch.pObs, 3), null, null, null], ['Change between the two finest grids (%)', null, null, pc(gs.peak.change21), pc(gs.patch.change21), null, null, null], ['Sequence', null, null, gs.peak.monotone ? 'monotone' : 'not monotone', gs.patch.monotone ? 'monotone' : 'not monotone', null, null, null],
        ['Osmotic ceiling of the wall concentration', null, null, q.cCap, q.cCap, q.cCap, null, null]],
      note: `The same problem — the first ${gs.nSec} filament spacings of the section — on three grids refined by the same factor along and across the channel (cells per spacing in multiples of ten, so the filament axes lie on cell faces on every grid). The peak is the highest wall-cell value on the permeating membrane; the patch value is the highest average over ${fq(gs.Lp * 1e6, 3)} µm of membrane, a length comparable with a nucleation site. Richardson extrapolation with the observed order limited to 0.5–2 and a safety factor of 1.25 (grid-convergence index); a sequence that is not monotone is not extrapolated and carries half its spread instead; an estimate above the osmotic ceiling is set to the ceiling. ${q.cCap ? `The flux law J = A·(ΔP − Δπ(c wall)) ≥ 0 stops permeation where the osmotic-pressure difference across the membrane reaches the applied pressure, so no wall concentration can exceed ${fq(q.cCap, 5)} times the section inlet; the largest value found on any grid is ${fq(gs.cMax, 5)}.` : 'Uniform flux: there is no osmotic ceiling; a flux that does not respond to the wall concentration keeps concentrating the stagnant corner, so these values are grid-dependent upper estimates.'} Flux through the membrane covered by the footprints: ${fq(Math.max(0, gs.jFoot) * 3.6e6, 2)} L/m²·h.` });
    const rowsH = [['Polarisation at the hot spot (patch ÷ local bulk)', hs.beta, '–'], ['Highest bulk concentration of the section ÷ inlet', hs.cbEnd, '–'], ['Hot-spot wall concentration ÷ section inlet (patch, grid-converged)', hs.cPatch, '–'], ['Hot-spot wall concentration ÷ section inlet (peak, grid-converged)', hs.cPeak, '–'], ['Hot-spot concentration factor versus the feed', q.cfSec * hs.cPatch, '×'],
      ...(q.cCap ? [['Osmotic ceiling ÷ section inlet', q.cCap, '–'], ['Share of the ceiling reached at the hot spot', 100 * (hs.cPatch - 1) / Math.max(q.cCap - 1, 1e-9), '%']] : []), ...q.set.map((id) => [`${MINERALS[id].name} saturation index at the hot spot${q.siCap ? ` (at the osmotic ceiling: ${fq(q.siCap[id], 3)})` : ''}`, hs.si[id], ''])];
    if (hs.sink) { const k = hs.sink; rowsH.push([`${nm} saturation index at the hot spot with scale growth`, k.si, ''], [`…lattice ion in shortest supply (${k.ion}) left at the wall`, 100 * k.c2 / k.cMid, '% of its unreacted value'], ['…surface rate constant k_r (secant of the growth law)', k.kr * 1e6, 'µm/s'], [`…${nm.toLowerCase()} growth at the hot spot`, k.flux, 'g/m²·d']); }
    X.TB.push({ title: 'Hot spot of the spacer section (grid-converged)', columns: ['Quantity', 'Value', 'Unit'], rows: rowsH, note: `The polarisation of the convergence study is applied to the highest bulk concentration of the full section${q.cCap ? ' and limited by the osmotic ceiling' : ''}.${hs.sink ? ` Scale growth: the lattice ion in shortest supply is transported as a second, fully rejected species that the wall consumes at N = k_r·(c wall − c sat); k_r is the secant of the suite’s growth law between saturation and the hot-spot state (${hs.sink.iterations} iterations), c sat and the saturation index of the depleted water come from the speciation engine. Same diffusivity as the salt; medium grid of the study.` : ''}` });
    X.PL.push({ type: 'line', title: 'Hot spot next to a filament on three grids', xlabel: 'Representative cell size √(Δx·Δy wall) (µm)', ylabel: 'c wall / c section inlet', series: [{ name: 'Peak', x: gs.grids.map((g) => g.hRep * 1e6), y: gs.grids.map((g) => g.peak) }, { name: `Mean over ${fq(gs.Lp * 1e6, 3)} µm`, x: gs.grids.map((g) => g.hRep * 1e6), y: gs.grids.map((g) => g.patch) }], hlines: [...(q.cCap ? [{ y: q.cCap, label: 'osmotic ceiling' }] : []), { y: gs.patch.value, label: 'patch, extrapolated' }] });
  }
  const bMean = 0.5 * (B.cMean + Tp.cMean) / q.cbMean, b95 = Math.max(B.c95, Tp.c95) / q.cbMean, siHot = hs ? hs.si[id0] : q.siHot95[id0];
  X.K.push({ label: 'Spacer section: mean polarisation factor', value: bMean, unit: '–', help: `Navier–Stokes solution with ${arrName}; open-slit march ${fq(q.marMean / q.cbMean, 3)}, film theory ${fq(q.film, 3)}` }, { label: `Spacer section: hot-spot ${nm.toLowerCase()} SI`, value: Math.max(siHot, -99), unit: '', status: siHot > Math.max(a.lim[id0] ?? 0, 0) ? 'warn' : 'ok', help: hs ? `Mean over the hottest ${fq(q.Lp * 1e6, 3)} µm of membrane next to a filament, Richardson estimate from three grids (grid-convergence index ${fq(100 * gs.patch.gci, 2)} %)` : '95th percentile of the wall saturation index behind the filaments' });
  if (hs) X.K.push({ label: 'Spacer section: hot-spot polarisation factor', value: hs.cPatch / hs.cbEnd, unit: '–', help: `Patch-averaged wall concentration ÷ bulk; grid-convergence index ${fq(100 * gs.patch.gci, 2)} %, change between the two finest grids ${fq(100 * gs.patch.change21, 2)} %` });
  if (q.cCap) X.K.push({ label: 'Spacer section: osmotic ceiling of the wall concentration', value: q.cCap, unit: '× section inlet', help: `Concentration at which the osmotic-pressure difference across the membrane equals the applied ${fq(v.P, 3)} bar and permeation stops; ${nm.toLowerCase()} SI there ${fq(q.siCap[id0], 3)}` });
  if (!f.converged) X.W.push({ level: 'warn', msg: 'The Navier–Stokes solution of the spacer section did not reach its tolerance — the flow behind the filaments is probably unsteady; treat the hot-spot values as indicative.' });
  if (siHot > 0 && siHot - q.siWallMean[id0] > 0.05) X.W.push({ level: 'info', msg: `Next to the spacer filaments the wall ${nm.toLowerCase()} saturation index reaches ${fq(siHot, 3)} (${hs ? 'grid-converged patch average' : '95th percentile'}; mean ${fq(q.siWallMean[id0], 3)}, film theory ${fq(q.siFilm[id0], 3)}): scale starts at these stagnant spots.` });
  if (gs && !(Math.abs(gs.patch.change21) < 0.03)) X.W.push({ level: 'warn', msg: `The patch-averaged hot spot still changes by ${fq(100 * Math.abs(gs.patch.change21), 2)} % between the two finest grids — raise the cells per spacing of the convergence study on the Mesh tab.` });
  if (gs && !q.osm) X.W.push({ level: 'info', msg: v.cfdOsm === false ? 'The spacer section was solved with a uniform flux: without the osmotic feedback the hot spot has no ceiling and does not converge with the grid — read it as an upper estimate.' : 'The entered pressure does not exceed the osmotic pressure of the section by enough for the flux law, so the spacer section was solved with a uniform flux: without the osmotic feedback the hot spot has no ceiling and does not converge with the grid — read it as an upper estimate.' });
  Object.assign(X.out, { spacerBetaMean: bMean, spacerBeta95: b95, spacerBetaPeak: hot.cPeak / q.cbMean, spacerSIMean: q.siWallMean[id0], spacerSIHot: siHot, spacerSIHot95: q.siHot95[id0], spacerDepositPeak: Math.max(B.depPeak, Tp.depPeak), spacerMineral: id0, spacerCap: q.cCap ?? 0, ...(gs ? { spacerHotPatch: gs.patch.value, spacerHotPeak: gs.peak.value, spacerHotPatchGCI: gs.patch.gci, spacerHotPeakGCI: gs.peak.gci, spacerHotChange: gs.patch.change21, spacerHotMax: gs.cMax } : {}), ...(hs && hs.sink ? { spacerSIHotLimited: hs.sink.si } : {}) });
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
      { key: 'bariteModel', label: 'Barium–sulphate interaction (Pitzer model): best estimate', type: 'select', value: BARITE_MODEL_DEFAULT, options: [{ value: 'analogue', label: 'Ca–SO₄ analogue binary, log Ksp −9.97 (Rogers 1981) — closest to the measured solubilities in water, NaCl and sulphate media' }, { value: 'pairw', label: 'BaSO₄(aq) ion pair, log K 2.72 (Felmy et al. 1990), log Ksp −9.965 fitted in this work to pure-water solubility — too soluble in sulphate media' }, { value: 'pair', label: 'BaSO₄(aq) ion pair, log K 2.72, log Ksp −10.05 (Felmy et al. 1990)' }, { value: 'none', label: 'No Ba–SO₄ term, log Ksp −9.97 (USGS pitzer.dat)' }], showIf: (v) => v.model === 'pitzer', help: 'How barium and sulphate interact in the Pitzer model, with the barite solubility product that belongs to each treatment; this choice sets the best estimate of the barite index. Three treatments are published as such; the fourth combines the published ion-pair constant with a solubility product fitted in this work to the measured solubility in pure water. Their agreement with measured solubilities is listed in the results table “Barite treatments against measured solubility”.' },
      { key: 'bariteVerdict', label: 'Barite verdict basis', type: 'select', value: 'envelope', options: [{ value: 'envelope', label: 'Conservative envelope: highest index among the published treatments' }, { value: 'best', label: 'Best estimate only' }], help: 'Which barite index decides the status, the recovery limits, the limiting mineral and the warnings. The envelope is the highest saturation index among the published treatments of the barium–sulphate interaction (and the selected one), so barite is never reported as within its limit while a published treatment says otherwise; after a calibration against own measurements it is the fitted index plus two standard errors. The reported saturation index is the best estimate in both cases.' },
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
      { key: 'dkBaPair', label: 'Δ log K of the BaSO₄(aq) ion pair', unit: '', value: 0, min: -1, max: 1, showIf: (v) => v.model === 'pitzer', help: 'Offset of the association constant of BaSO₄(aq) (published value 2.72 ± 0.09) for the two ion-pair treatments; without effect on the others. Fit it on the Calibrate tab only with barium measurements at clearly different sulphate concentrations.' },
      { key: 'seBarite', label: 'Standard error of a calibrated Δ log Ksp barite', unit: '', value: 0, min: 0, max: 0.5, help: '0 = not calibrated: the range of the barite index is the spread between the treatments of the barium–sulphate interaction. After fitting Δ log Ksp barite to own measurements (Calibrate tab) enter the “± Std. error” of the fit here: the range becomes ± 2 standard errors and the conservative verdict uses the fitted index plus 2 standard errors. Valid for waters close to the one measured.' },
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
      { key: 'cfdOsm', label: 'Flux responds to the local osmotic pressure', type: 'bool', value: true, help: 'Solution–diffusion wall: J = A·(ΔP − Δπ(c wall)) ≥ 0 with the pressure of the Inputs tab, A fitted so that the flux at the section inlet equals the entered flux. The flux stops where the osmotic pressure reaches the applied pressure, which bounds the wall concentration. Unticked: uniform flux (no such bound).', showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdContact', label: 'Filament contact half-width', unit: '% of filament diameter', value: 15, min: 0, max: 50, typical: [10, 20], help: 'A filament pressed against the membrane is flattened and covers it over a strip of this half-width on each side of its axis; the covered membrane does not permeate. 0 = ideal line contact with permeation right up to it (the hot spot then depends on the grid).', showIf: (v) => v.cfdOn && v.cfdSpacer && (v.cfdArr === 'zigzag' || v.cfdArr === 'cavity') },
      { key: 'cfdPatch', label: 'Hot-spot averaging length', unit: 'µm', value: 100, min: 10, max: 1000, typical: [50, 100], help: 'The hot spot is reported as the highest mean wall concentration over this length of membrane — comparable with a nucleation site or a young crystal.', showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdGci', label: 'Grid-convergence study of the hot spot', type: 'bool', value: true, help: 'Solves the first two filament spacings on three refined grids and reports the Richardson estimate of the hot spot with its grid-convergence index. Adds a few seconds.', showIf: (v) => v.cfdOn && v.cfdSpacer },
      { key: 'cfdSink', label: 'Let scale growth limit the hot-spot supersaturation', type: 'bool', value: false, help: 'Adds wall crystallisation of the kinetic mineral with the growth law of the kinetics group: the lattice ion in shortest supply is consumed at the wall, which lowers the saturation index the hot spot can sustain. Adds several seconds.', showIf: (v) => v.cfdOn && v.cfdSpacer && v.cfdGci !== false },
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
      { key: 'cfdGciN', label: 'Cells per filament spacing on the finest grid of the hot-spot study', unit: '', value: 50, min: 50, max: 120, step: 10, help: 'The three grids of the convergence study use 0.6, 0.8 and 1.0 times this number per spacing (multiples of ten) and 0.8 times as many cells across the channel.', showIf: (v) => v.cfdOn && v.cfdSpacer && v.cfdGci !== false },
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
    // Barite: the numbers shown are the best estimate; every verdict below is taken on wSI, where barite carries the verdict basis (conservative envelope unless the best estimate is selected)
    const bOpt = a.vb, bV = present(wall.eq, 'barite') ? { Feed: bariteVerdict(feed.eq, P, dk.barite || 0, bOpt), Concentrate: bariteVerdict(conc.eq, P, dk.barite || 0, bOpt), 'Membrane wall': bariteVerdict(wall.eq, P, dk.barite || 0, bOpt) } : null;
    const wSI = bV ? { ...wd.SI, barite: bV['Membrane wall'].verdict } : wd.SI, bUp = bV ? bV['Membrane wall'].up : 0;
    for (const id of set) if ((lim[id] ?? 0) > 0 && wSI[id] > 0) sev = Math.max(sev, wSI[id] / lim[id]);
    const keepF = 1 - R * (1 - a.rej), asDose = v.antiscalant && sev > 0 ? (clamp(2 + 6 * sev, 2, 12) * (1 - R)) / keepF : 0;
    const status = (id, s) => (s == null ? '–' : s <= 0 ? 'Undersaturated' : !v.antiscalant ? 'Scaling without inhibitor' : s <= limOf(id) ? 'Supersaturated — controlled by antiscalant' : 'Exceeds antiscalant limit');
    for (const id of set) {
      const s = wSI[id], env = id === 'barite' && bUp > 0 ? ` on the conservative envelope of the ${bOpt.se > 0 ? 'calibration (fitted index + 2 standard errors)' : 'published Ba–SO₄ treatments'}; best estimate SI ${fmt(wd.SI[id], 3)}` : '';
      if (s == null || s <= 0) continue;
      if (s > limOf(id)) W.push({ level: 'bad', msg: `${MINERALS[id].name} is ${fmt(10 ** s * 100, 3)} % saturated at the membrane wall (SI ${fmt(s, 3)}${env}), above the ${v.antiscalant ? 'antiscalant limit' : 'saturation limit'} of SI ${fmt(limOf(id), 3)} — lower the recovery${id === 'calcite' ? ' or dose acid' : id === 'silica' ? ', raise the temperature or pH, or use a silica dispersant' : ''}.` });
      else W.push({ level: 'info', msg: `${MINERALS[id].name} is supersaturated at the wall (SI ${fmt(s, 3)}${env}) but within the antiscalant limit.` });
    }
    // barite: best estimate, range and verdict basis for the three streams (Pitzer model, barium present)
    const bRows = [], bWall = bV ? bV['Membrane wall'] : null, bLim = limOf('barite');
    for (const [nm, [sol]] of Object.entries(bV ? { Feed: [feed], Concentrate: [conc], 'Membrane wall': [wall] } : {})) {
      const q = bV[nm], b = q.band, s = q.best, lim0 = nm === 'Membrane wall' ? bLim : 0, word = nm === 'Membrane wall' ? ['Within the limit', 'Above the limit'] : ['Undersaturated', 'Supersaturated'];
      if (!b) continue;
      const over = q.verdict > lim0, overBest = s > lim0, overAny = q.envelope > lim0;
      bRows.push([nm, b.mSO4, b.I, s, `${fmt(s + b.shift.pair, 3)} (${fmt(s + b.shift.pairLo, 3)} to ${fmt(s + b.shift.pairHi, 3)})`, s + b.shift.pairw, s + b.shift.ca, s + b.shift.sr, s + b.shift.zero, 100 * b.paired, q.lo, q.hi, q.verdict,
        (over ? word[1] : word[0]) + (over !== overBest ? ' on the conservative envelope (best estimate: ' + word[0].toLowerCase() + ')' : !over && overAny ? ' on the best estimate (a published treatment: ' + word[1].toLowerCase() + ')' : q.calibrated ? ' (calibrated)' : ' under every published treatment')]);
    }
    if (bWall && bWall.band) {
      const st = `best estimate SI ${fmt(bWall.best, 3)} at the wall (${BA_SHORT[a.bariteModel]}), range ${fmt(bWall.lo, 3)} to ${fmt(bWall.hi, 3)}${bWall.calibrated ? ' (± 2 standard errors of the calibration)' : ' over the treatments of the Ba–SO₄ interaction'}`;
      if (bWall.verdict > bLim !== bWall.best > bLim) W.push({ level: 'warn', msg: `Barite: ${st}. The best estimate is within the limit of SI ${fmt(bLim, 3)}, the conservative envelope (SI ${fmt(bWall.verdict, 3)}) is not: status, recovery limits and limiting mineral are taken on the envelope. A jar test on this water settles it (Calibrate tab).` });
      else if (bOpt.basis === 'best' && bWall.envelope > bLim !== bWall.best > bLim) W.push({ level: 'warn', msg: `Barite: ${st}. The verdict uses the best estimate only (input “Barite verdict basis”); a published treatment of the Ba–SO₄ interaction gives SI ${fmt(bWall.envelope, 3)}, above the limit of SI ${fmt(bLim, 3)} — treat barite as marginal.` });
      else if (bWall.hi - bWall.lo > BARITE_BAND.warn) W.push({ level: 'info', msg: `Barite: ${st}${bWall.calibrated ? '' : ` (free sulphate ${fmt(bWall.band.mSO4, 3)} mol/kg, ${fmt(100 * bWall.band.paired, 2)} % of the barium paired under the ion-pair treatments)`}. The verdict against the limit of SI ${fmt(bLim, 3)} is the same over the whole range.` });
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
    const fast = kin.filter(([id, k]) => k.tInd < v.tRes && wSI[id] > limOf(id));
    for (const [id, k] of fast) W.push({ level: 'warn', msg: `${MINERALS[id].name}: estimated induction time ${fmt(k.tInd, 2)} s is shorter than the concentrate residence time (${v.tRes} s).` });
    // activity-coefficient comparison for NaCl
    const ms = logspace(0.001, 6, 19), gam = Object.keys(ACTIVITY_MODELS).map((mod) => ({ name: ACTIVITY_MODELS[mod].split(' (')[0].split(' +')[0], x: ms, y: ms.map((m) => Math.min(saltActivity('Na', 'Cl', m, { T: 25, model: mod }).gamma, 3)) }));
    ctx?.progress?.(0.8, 'Scaling map…');
    // scaling-margin map over recovery and feed pH
    const fx = linspace(0, a.Rmax, 13), fy = linspace(5.5, 9, 8);
    const fz = fy.map((ph) => { const f = equilibrate(a.raw, { pH: ph }); return fx.map((r) => { const q = a.satV(concentrateSolution(f, a.beta / (1 - r), a.copt).eq)[1]; let m = -9; for (const id of set) if (q[id] != null) m = Math.max(m, q[id] - limOf(id)); return clamp(m, -3, 3); }); });
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
      scalingMargin: Math.max(-9, ...set.filter((id) => wSI[id] != null).map((id) => wSI[id] - limOf(id))),
      bariteSIwallBest: bWall ? bWall.best : -99, bariteSIwallLow: bWall ? bWall.lo : -99, bariteSIwallHigh: bWall ? bWall.hi : -99, bariteSIwallVerdict: bWall ? bWall.verdict : -99, bariteVerdictBasis: bOpt.basis, bariteModel: a.bariteModel,
      maxRecoveryBariteNoAntiscalant: a.maxRec.barite?.present ? a.maxRec.barite.plain ?? a.Rmax : a.Rmax, maxRecoveryBariteAntiscalant: a.maxRec.barite?.present ? a.maxRec.barite.as ?? a.Rmax : a.Rmax,
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
        { label: 'Gypsum saturation at wall', value: sat('gypsum'), unit: '%', status: (wd.SI.gypsum ?? -9) > limOf('gypsum') ? 'bad' : 'ok' }, { label: 'Barite saturation at wall', value: sat('barite'), unit: '%', status: (wSI.barite ?? -9) > limOf('barite') ? 'bad' : 'ok', ...(bWall && bWall.band ? { help: `Best estimate; range ${fmt(100 * 10 ** bWall.lo, 3)}–${fmt(100 * 10 ** bWall.hi, 3)} %. The status uses ${bOpt.basis === 'best' ? 'the best estimate' : `the conservative envelope, ${fmt(100 * 10 ** bWall.verdict, 3)} %`}` } : {}) },
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
        { type: 'line', title: `Saturation index versus recovery (membrane wall, β = ${fmt(a.beta, 3)})`, xlabel: 'Recovery (%)', ylabel: 'Saturation index log(IAP/Ksp)', series: [...line(Rs.map((r) => 100 * r), sweep), ...(bUp > 0 && shown.includes('barite') ? [{ name: 'Barite, conservative envelope (verdict basis)', x: Rs.map((r) => 100 * r), y: a.sweepV.map((q) => clamp(q.barite ?? -99, -8, 8)), dash: true }] : [])], hlines: [{ y: 0, label: 'saturation' }], vlines: [{ x: 100 * R, label: 'design' }, ...(best.r != null ? [{ x: 100 * best.r, label: 'limit' }] : [])], ymin: -4, ymax: 4 },
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
          rows: allIds.map((id) => [MINERALS[id].name, MINERALS[id].formula, fd.SI[id] ?? null, cd.SI[id] ?? null, wd.SI[id], 100 * 10 ** clamp(wd.SI[id], -12, 12), set.includes(id) ? limOf(id) : null, a.maxRec[id] ? recStr(a.maxRec[id].plain) : null, a.maxRec[id] ? recStr(a.maxRec[id].as) : null, (MINERALS[id].group === 'inhibited' ? 'Kinetically inhibited — not a practical scale' : set.includes(id) ? status(id, wSI[id]) : wd.SI[id] > 0 ? 'Supersaturated' : 'Undersaturated') + (id === 'barite' && bWall && bWall.band ? ` · best estimate SI ${fmt(bWall.best, 3)}, range ${fmt(bWall.lo, 3)} to ${fmt(bWall.hi, 3)}; verdict on ${bOpt.basis === 'best' ? 'the best estimate' : `the conservative envelope, SI ${fmt(bWall.verdict, 3)}`}` : '')]),
          note: `SI = log₁₀(ion-activity product / Ksp) with the ${ACTIVITY_MODELS[a.model]} model at ${v.T} °C and ${v.P} bar. “> x” means the limit is not reached within the sweep.${bWall && bWall.band ? ` Barite: the SI columns give the best estimate (${BA_SHORT[a.bariteModel]}), SI ${fmt(bWall.best, 3)} at the wall with a range of ${fmt(bWall.lo, 3)} to ${fmt(bWall.hi, 3)} ${bWall.calibrated ? 'from the calibration against the user’s own measurements (± 2 standard errors)' : 'over the treatments of the Ba–SO₄ interaction'}. ${bOpt.basis === 'best' ? 'The status and the recovery limits of barite use the best estimate (input “Barite verdict basis”).' : `The status, the recovery limits of barite and the limiting mineral use the upper end among the published treatments — the conservative envelope, SI ${fmt(bWall.verdict, 3)} at the wall — so that barite is never reported as within its limit while a published treatment says otherwise.`}` : ''}` },
        ...(bRows.length ? [{ title: 'Barite index: sensitivity to the Ba–SO₄ interaction', columns: ['Stream', 'Free SO₄²⁻ (mol/kg)', 'Ionic strength (mol/kg)', `SI, best estimate of this run (${BA_SHORT[a.bariteModel]})`, 'SI, BaSO₄(aq) ion pair with log Ksp −10.05 (range for log K ± 0.09)', 'SI, BaSO₄(aq) ion pair with log Ksp −9.965 (fitted here)', 'SI, Ca–SO₄ analogue', 'SI, Sr–SO₄ analogue', 'SI, no Ba–SO₄ term', 'Barium bound in the pair (%)', 'Lowest SI', 'Highest SI', `SI used for the verdict (${bOpt.basis === 'best' ? 'best estimate' : 'conservative envelope'})`, 'Verdict'], rows: bRows,
          note: BARITE_NOTE(a, bOpt) }] : []),
        BARITE_EVIDENCE_TABLE(),
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
          note: 'Every tabulated constant of the chemistry engine was compared with the named database file or table. “Replaced” = the earlier value could not be found and was exchanged for a documented one; “fitted here” / “refitted here” = derived in this work from the named measurements, with the fit statistics given; “analogue” = assigned by chemical similarity; “unconfirmed” = no source could be retrieved, and the entry is not used by the default (Pitzer) calculation path. The activity models are tested against the NIST activity-coefficient tables on the Verify tab.' },
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
  { name: 'Spacer-section grid (Navier–Stokes)', keys: ['cfdNxFil', 'cfdNyFull'], min: 8, note: 'Cells per filament spacing and across the channel of the Navier–Stokes model (needs the spacer-filled section switched on). The mean polarisation converges quickly; the hot spot next to a filament is resolved by the separate three-grid convergence study of the results.', metrics: [{ label: 'Mean polarisation factor', unit: '–', get: (r) => r.outputs.spacerBetaMean ?? 1 }, { label: '95th-percentile polarisation factor', unit: '–', get: (r) => r.outputs.spacerBeta95 ?? 1 }] },
  { name: 'Flow-model grid across the channel', keys: ['cfdNy'], min: 8, note: 'Wall-normal cells of the two-dimensional channel model.', metrics: [{ label: 'Largest local polarisation factor', unit: '–', get: (r) => r.outputs.cfdBetaMax ?? 1 }, { label: 'Channel pressure drop', unit: 'bar', get: (r) => r.outputs.cfdPressureDropBar ?? 0 }] }],

  calibration: {
    note: 'Fit the solubility-product offsets to laboratory solubility data. Gypsum and calcite rows: one equilibrium experiment each in an NaCl background solution (gypsum in a closed vessel, calcite under a fixed CO₂ partial pressure); the measured dissolved calcium is compared with the model. Barite rows use the water of this case (feed analysis, pH and activity model of the Scenario and Model setup tabs) concentrated by the factor given in the row (1 / (1 − recovery); below 1 for a dilution), at the row temperature: either the dissolved barium measured after equilibration with barite, or the concentration factor at which barite was first seen to precipitate. How to run the barite jar test: bring the water to the concentration factor of interest (pilot concentrate, evaporated or blended feed) at the test temperature without antiscalant, add about 1 g/L of fine reagent-grade barite as seed and stir in a closed bottle for at least 72 h, sampling at 24, 48 and 72 h until two successive barium readings agree within the analytical error. Filter each sample at temperature through 0.2 µm or finer (fines passing the filter are the classic cause of high readings), acidify, and measure barium by ICP-MS or ICP-OES together with sulphate. Enter one row per test, fit “Δ log Ksp barite” (and the ion-pair constant only if the tests span clearly different sulphate concentrations), apply it, and type its “± Std. error” into “Standard error of a calibrated Δ log Ksp barite” on the Model setup tab: the range of the barite index then reflects the fit instead of the spread between treatments. An onset observed without seed lies above saturation and makes barite look more soluble than it is — use seeded tests for the fit and onset rows as a cross-check. The sample rows are a synthetic illustration generated with the model, shifted constants and noise — not measurements; replace them with your own. Validate with experiments at other salinities, concentration factors and temperatures.',
    params: [{ key: 'dkGypsum', label: 'Δ log Ksp gypsum', lo: -0.4, hi: 0.4 }, { key: 'dkCalcite', label: 'Δ log Ksp calcite', lo: -0.4, hi: 0.4 }, { key: 'dkBarite', label: 'Δ log Ksp barite', lo: -0.5, hi: 0.5 }, { key: 'dkBaPair', label: 'Δ log K BaSO₄(aq) ion pair (ion-pair treatments only; needs tests at different sulphate levels)', lo: -0.6, hi: 0.6 }],
    columns: [{ key: 'mNaCl', label: 'NaCl background (gypsum, calcite)', unit: 'mol/kg' }, { key: 'Tc', label: 'Temperature', unit: '°C' }, { key: 'pCO2x', label: 'CO₂ pressure (calcite)', unit: 'atm' }, { key: 'sGyp', label: 'Gypsum solubility', unit: 'mmol/kg' }, { key: 'sCal', label: 'Calcite solubility', unit: 'mmol/kg' },
      { key: 'cfBa', label: 'Barite test: concentration factor of the case water', unit: '×' }, { key: 'baEq', label: 'Barite test: dissolved barium at equilibrium', unit: 'µg/L' }, { key: 'cfOn', label: 'Barite: concentration factor at observed onset of precipitation', unit: '×' }],
    targets: [{ key: 'sGyp', label: 'Gypsum solubility', unit: 'mmol/kg' }, { key: 'sCal', label: 'Calcite solubility', unit: 'mmol/kg' }, { key: 'baEq', label: 'Dissolved barium at barite equilibrium', unit: 'µg/L' }, { key: 'cfOn', label: 'Concentration factor at barite saturation', unit: '×' }],
    model(v) {
      const C = (this._c ||= new Map()), memo = (k, f) => { if (!C.has(k)) { if (C.size > 800) C.clear(); C.set(k, f()); } return C.get(k); };
      const m = Math.max(0, v.mNaCl ?? 0), T = v.Tc ?? 25, bar = v.cfBa > 0, out = { sGyp: NaN, sCal: NaN };
      if (!bar) { // gypsum and calcite in an NaCl background (rows without a barite test)
        const base = memo(`b|${m}|${T}|${v.model}`, () => { const n = new Float64Array(NM); n[mi('Na')] = m; n[mi('Cl')] = m; return equilibrate({ T, model: v.model, n, alk: 0, w: 1, pH: 7 }); });
        out.sGyp = memo(`g|${m}|${T}|${v.model}|${v.dkGypsum || 0}`, () => solubility(base, 'gypsum', { dk: { gypsum: v.dkGypsum || 0 }, excess: 0.3 }).m * 1000);
        out.sCal = memo(`c|${m}|${T}|${v.model}|${v.dkCalcite || 0}|${v.pCO2x}`, () => solubility(base, 'calcite', { pCO2: Math.max(1e-6, v.pCO2x ?? 4.2e-4), dk: { calcite: v.dkCalcite || 0 }, excess: 0.3 }).m * 1000);
      }
      // barite in the water of the case: equilibrium with barite seed at the concentration factor of the row, and the factor at which SI = 0
      const ba = BARITE_MODELS[v.bariteModel] ? v.bariteModel : BARITE_MODEL, pk = Number.isFinite(v.dkBaPair) ? v.dkBaPair : 0, dk = v.dkBarite || 0, ions = v.ions || WATERS.seawater.ions, wk = `${JSON.stringify(ions)}|${v.pH}|${T}|${v.model}|${ba}|${pk}`;
      const feed = memo('w|' + wk, () => makeSolution({ ions, T, pH: v.pH ?? 8, model: v.model, bariteModel: ba, pairDK: pk })), cf = bar ? v.cfBa : 1;
      out.baEq = memo(`e|${wk}|${cf}|${dk}`, () => { const r = solubility(concentrateSolution(feed, cf, { co2: 'closed' }), 'barite', { dk: { barite: dk }, excess: 1e-3 }); return solutionToIons(r.sol).ions.Ba * 1000; });
      const grid = memo('s|' + wk, () => { const xs = logspace(0.05, clamp(300 / Math.max(solutionToIons(feed).gPerKgw, 1e-6), 1.5, 20), 21); return [xs, xs.map((x) => saturationIndex(concentrateSolution(feed, x, { co2: 'closed' }).eq, 'barite'))]; });
      out.cfOn = feed.n[mi('Ba')] > 0 && feed.n[mi('SO4')] > 0 ? cross(grid[0], grid[1], dk) ?? grid[0].at(-1) : grid[0].at(-1);
      return out;
    },
    get sample() { return (this._s ||= [...synth(5, [[0, 25, 0.01], [0.25, 25, 0.01], [0.5, 25, 0.03], [1, 25, 0.03], [2, 25, 0.1], [3, 25, 0.1], [4, 25, 0.3], [0.5, 35, 0.3]]), ...synthBa(11, [[1, 25], [1.25, 25], [1.54, 25], [1.82, 25], [2.2, 25], [1.54, 15], [0.5, 25]], [25])]); },
    get validationSample() { return (this._v ||= [...synth(17, [[0.1, 25, 0.05], [0.75, 30, 0.05], [1.5, 25, 0.2], [2.5, 20, 0.02], [3.5, 25, 0.5], [5, 25, 0.1]]), ...synthBa(23, [[1.1, 25], [1.67, 25], [2, 20], [1.33, 30]], [20])]); },
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
    { // MgSO4: implementation checks first, then measurements that are independent of the suite's parameters
      const ps = [0.1, 1, 3].map((m) => { const q = saltActivity('Mg', 'SO4', m), h = pitzerSingle(2, 2, m, 0.221, 3.343, -37.23, 0.025); return Math.max(Math.abs(q.gamma / h.gamma - 1), Math.abs(q.phi / h.phi - 1)); });
      add('MgSO₄ 0.1–3 mol/kg, Pitzer model: multi-ion sums against the closed-form single-salt equation', 0, Math.max(...ps), 2e-3, 'Implementation check, not an independent one: both sides use the Harvie–Møller–Weare parameters (β⁰ 0.221, β¹ 3.343, β² −37.23, Cφ 0.025, as held in EQ3/6 data0.hmw)');
      const p01 = saltActivity('Mg', 'SO4', 0.1);
      add('MgSO₄ 0.1 mol/kg: SIT, Davies and Bromley against the suite’s Pitzer model', 0, Math.max(...['sit', 'davies', 'bromley'].map((mod) => Math.abs(saltActivity('Mg', 'SO4', 0.1, { model: mod }).gamma / p01.gamma - 1))), 0.15, 'Consistency check, not an independent one: the reference is the suite’s own source-verified Pitzer model. 2:2 salts are outside the range of the Bromley correlation');
      { // independent data: isopiestic osmotic coefficients, the Archer–Rard evaluation, epsomite solubility
        const dv = (rows, f) => Math.max(...rows.map(f)), q = (m) => saltActivity('Mg', 'SO4', m);
        const dIso = dv(MGSO4_REF.iso, ([m, ph]) => Math.abs(q(m).phi / ph - 1)), dG = dv(MGSO4_REF.ar, ([I, g]) => Math.abs(q(I / 4).gamma / g - 1)), dP = dv(MGSO4_REF.ar, ([I, , ph]) => Math.abs(q(I / 4).phi / ph - 1)), dPP = dv(MGSO4_REF.pp, ([m, ph]) => Math.abs(q(m).phi / ph - 1));
        add('MgSO₄ 0.72–3.14 mol/kg, Pitzer model: osmotic coefficient against isopiestic measurements', 0, dIso, 0.012, `Largest relative deviation from six points measured against KCl reference solutions (Miladinović, Ninković, Todorović & Rard 2007, LLNL UCRL-JRNL-231697, Table 4): ${fmt(100 * dIso, 2)} %. Independent of the model parameters: the Harvie–Møller–Weare set dates from 1984`);
        add('MgSO₄ 0.25–1.5 mol/kg, Pitzer model: γ± against the Archer & Rard (1998) evaluation', 0, dG, 0.006, `Largest relative deviation over 11 tabulated ionic strengths (values of the extended ion-interaction model of Archer & Rard as printed in Table 6 of the same report): γ± ${fmt(100 * dG, 2)} %, φ ${fmt(100 * dP, 2)} %. An independent evaluation with its own parameters, not a measurement`);
        add('MgSO₄ 0.25–1.5 mol/kg, Pitzer model: φ against the Archer & Rard (1998) evaluation', 0, dP, 0.006, 'Same table');
        add('MgSO₄ 0.1–3 mol/kg, Pitzer model: φ against Phutela & Pitzer (1986)', 0, dPP, 0.013, `Four values quoted in Table I-15 of the report ANL-EBS-MD-000045 REV 02 (2004): largest deviation ${fmt(100 * dPP, 2)} %. Secondary source; Phutela & Pitzer and Harvie–Møller–Weare evaluate partly the same measurements`);
        const eps = solubility(pure, 'epsomite', { excess: 8 });
        add('Epsomite (MgSO₄·7H₂O) solubility in water, 25 °C', 2.99, eps.molality, 0.05, `mol/kg water. Two retrieved values: 3.018 (data point of Pabalan & Pitzer 1987, fig. 7, as entered in the file MgSO4.csv of Appelo 2015) and 2.96 (26.3 mass %, CRC Handbook as converted in ANL-EBS-MD-000045, solid phase not named there). Tests the Pitzer activity product at saturation (log K −1.881 with γ± and the water activity at 3 mol/kg); water activity of the saturated solution in the model ${fmt(eps.sol.eq.aw, 4)}`);
        const awAR = (m) => Math.exp(-2 * m * archerRardPhi(m) * MW_W), dTab = dv(MGSO4_REF.ar, ([I, , ph]) => Math.abs(archerRardPhi(I / 4) - ph));
        add('Archer & Rard (1998) MgSO₄ model as coded here: φ against the values printed with the parameters', 0, dTab, 1.5e-4, 'Transcription check of the nine parameters (Table 3 of Miladinović et al. 2007) against the eleven φ of Table 6 of the same report, I = 1–6 mol/kg; largest absolute difference');
        add('MgSO₄ at 3.018 mol/kg (epsomite solubility): water activity against the Archer & Rard model', awAR(3.018), q(3.018).aw, 0.002, `Derived reference, not a measurement: a_w = exp(−2mφ·M_w) with φ = ${fmt(archerRardPhi(3.018), 4)} from the Archer & Rard (1998) parameters (reprinted, secondary) at the solubility 3.018 mol/kg; at 2.96 mol/kg the two models give ${fmt(awAR(2.96), 4)} and ${fmt(q(2.96).aw, 4)}`);
        add('Epsomite-saturated solution, 25 °C: water activity against the Archer & Rard model (secondary)', 0.5 * (awAR(2.96) + awAR(3.018)), eps.sol.eq.aw, 0.004, `Secondary to the measurement-based check below. Reference ${fmt(awAR(3.018), 4)}–${fmt(awAR(2.96), 4)}: the Archer & Rard model evaluated at the two retrieved solubilities (derived). Suite: ${fmt(eps.sol.eq.aw, 4)} at its own saturation ${fmt(eps.molality, 3)} mol/kg. The only retrieved statement of the deliquescence humidity is 91 % for bulk MgSO₄·7H₂O at room temperature (Linnow et al. 2014, Energy Procedia 48, 394, from the phase diagram of Steiger et al. 2011 — calculated, secondary). Measured osmotic coefficients and vapour pressures are compared in the checks below; Archer & Rard (1998) and Rard & Miller (1981) themselves have no open copy`);
        { // primary reference for the saturated solution: ln a_w = −ν·m·M_w·φ (exact) with the measured φ interpolated to the saturation molality
          const top = MGSO4_REF.iso.slice(-3), lag = (pts, x) => sum(pts.map(([xi, yi], i) => yi * pts.reduce((p, [xj], j) => (j === i ? p : (p * (x - xj)) / (xi - xj)), 1))), awOf = (m, ph) => Math.exp(-2 * m * ph * MW_W), uPhi = 0.003;
          const sat = [['CRC Handbook value quoted in ANL-EBS-MD-000045', 2.96], ['this model', eps.molality], ['Pabalan & Pitzer 1987, as entered by Appelo 2015', 3.018], ['Xue et al. 2016', MGSO4_REF.sol[0][1]]].map(([s, m]) => { const ph = lag(top, m), a = awOf(m, ph); return { s, m, ph, a, lin: awOf(m, lag(top.slice(1), m)), alt: awOf(m, lag([MGSO4_REF.iv[0], top[2]], m)), u: a * 2 * m * MW_W * uPhi }; });
          const lo = Math.min(...sat.map((r) => r.a)), hi = Math.max(...sat.map((r) => r.a)), uM = Math.max(...sat.map((r) => r.u)), uI = Math.max(...sat.map((r) => Math.max(Math.abs(r.lin - r.a), Math.abs(r.alt - r.a)))), own = sat[1], aS = eps.sol.eq.aw;
          add('Epsomite-saturated solution, 25 °C: water activity from measured isopiestic osmotic coefficients', 0.5 * (lo + hi), aS, 0.5 * (hi - lo) + Math.hypot(uM, uI), `Primary reference, derived from measurements through the exact relation ln a_w = −ν·m·M_w·φ (ν = 2, M_w = 0.0180153 kg/mol). φ: isopiestic points of pure MgSO₄ on both sides of saturation — ${top.map(([m, ph]) => `${ph} at ${m} mol/kg`).join(', ')} (Miladinović, Ninković, Todorović & Rard 2007, LLNL UCRL-JRNL-231697, Table 4, read first-hand) — interpolated by the parabola through the three points to each retrieved saturation molality: ${sat.map((r) => `${fmt(r.m, 4)} mol/kg (${r.s}): φ ${fmt(r.ph, 4)}, a_w ${fmt(r.a, 4)}`).join('; ')}. Reference a_w ${fmt(0.5 * (lo + hi), 4)} ± ${fmt(0.5 * (hi - lo), 2)} from the spread of the solubility values, ± ${fmt(uM, 1)} from ±${uPhi} in φ (assumed for isopiestic work; the source reports a mean deviation of 0.0013 from the Archer & Rard model) and ± ${fmt(uI, 1)} from the interpolation across the 0.75 mol/kg gap (parabola against the straight line between the two bracketing points, and against the bracket formed with the highest point of the second series, ${MGSO4_REF.iv[0][1]} at ${MGSO4_REF.iv[0][0]} mol/kg, Ivanović et al. 2017). Suite: ${fmt(aS, 4)} at its own saturation ${fmt(eps.molality, 3)} mol/kg — deviation ${fmt(aS - 0.5 * (lo + hi), 2)} from the centre of the reference range`);
          add('MgSO₄ at the model’s saturation molality: water activity against the measured osmotic coefficients', own.a, aS, 0.001, `Same reference at one molality (${fmt(own.m, 4)} mol/kg), so the solubility value drops out: measured φ ${fmt(own.ph, 4)} → a_w ${fmt(own.a, 4)} ± ${fmt(Math.hypot(own.u, uI), 1)}; suite φ ${fmt(q(own.m).phi, 4)} → a_w ${fmt(aS, 4)}, deviation ${fmt(aS - own.a, 2)}. The Harvie–Møller–Weare set is ${fmt(100 * (1 - q(own.m).phi / own.ph), 2)} % low in φ here, as at the measured point 3.14 mol/kg`);
        }
        { // measurements: second isopiestic series, recent solubilities, vapour pressure of the saturated solution
          const dIv = dv(MGSO4_REF.iv, ([m, ph]) => Math.abs(q(m).phi / ph - 1));
          { // a third isopiestic series, measured against NaCl at Oak Ridge
            const w = MGSO4_REF.wrs, lo = w.filter((r) => r[0] < 2.5), top = w.at(-1), dW = dv(lo, ([m, ph]) => Math.abs(saltActivity('Mg', 'SO4', m).phi / ph - 1));
            add('MgSO₄ 0.57–2.39 mol/kg, Pitzer model: osmotic coefficient against the isopiestic series of Wu, Rush & Scatchard (1968)', 0, dW, 0.009, `Largest relative deviation from ${lo.length} points measured against NaCl at 25 °C (J. Phys. Chem. 72, 4048, Tables II and III, columns of pure MgSO₄, as reprinted in the Oak Ridge report ORNL-TM-4212, p. 14 — primary table, read from the page image; molality = νmφ/(2φ)). The Archer & Rard (1998) model as coded here deviates by up to ${fmt(100 * dv(lo, ([m, ph]) => Math.abs(archerRardPhi(m) / ph - 1)), 2)} % from the same points. The authors used the NaCl osmotic coefficients of Robinson & Stokes and state a mean difference of 0.006 in φ from the MgSO₄ table of those authors`);
            add('MgSO₄ 2.59 mol/kg: excess of the suite over the highest point of Wu, Rush & Scatchard against that of the Archer & Rard model', archerRardPhi(top[0]) / top[1] - 1, saltActivity('Mg', 'SO4', top[0]).phi / top[1] - 1, 0.006, `Relative difference in φ (measured ${top[1]} at ${top[0]} mol/kg): the suite and the Archer & Rard evaluation lie equally far above this one point, which the later long-equilibration measurements (Miladinović et al. 2007: 0.7508 at 2.387 mol/kg against 0.7435 at 2.394 here) show to be low; it is reported, not used as a criterion`);
          }
          add('MgSO₄ 1.56–2.39 mol/kg, Pitzer model: osmotic coefficient against a second isopiestic series', 0, dIv, 0.008, `Largest relative deviation from 12 points measured against KCl at 298.15 K (Ivanović, Popović, Rard et al. 2017, J. Chem. Thermodyn. 113, 91 — primary measurements, read through the NIST ThermoML transcription of the journal table; stated uncertainty 0.010–0.016 in φ)`);
          add('Epsomite solubility in water, 25 °C: against the measurement of Xue et al. (2016)', MGSO4_REF.sol[0][1], eps.molality, 0.122, `mol/kg; tolerance = the uncertainty stored with the value (NIST ThermoML transcription of Fluid Phase Equilib. 408, 115). Two further recent measurements lie higher: ${MGSO4_REF.sol.slice(1).map(([s, x]) => `${x} (${s})`).join(', ')} — ${fmt(100 * (MGSO4_REF.sol[1][1] / eps.molality - 1), 2)} and ${fmt(100 * (MGSO4_REF.sol[2][1] / eps.molality - 1), 2)} % above the model; the archive does not name the solid phase. The retrieved solubilities span 2.96–3.18 mol/kg`);
          const pw = (T) => Math.exp(34.494 - 4924.99 / (T - 273.15 + 237.1)) / (T - 273.15 + 105) ** 1.57 / 1000, rh = MGSO4_REF.pv.map(([s, T, p]) => [s, T, p / pw(T)]), mid = 0.5 * (Math.min(...rh.map((r) => r[2])) + Math.max(...rh.map((r) => r[2])));
          add('Epsomite-saturated solution near 25 °C: water activity against vapour pressures read from a figure (secondary, loose cross-check)', mid, eps.sol.eq.aw, 0.035, `Not the primary reference — that is the check built on measured osmotic coefficients above (a_w known to about ±0.003 there, ±0.03 here). p/p°(water), p° from the vapour-pressure equation of the same paper: ${rh.map(([s, T, x]) => `${fmt(x, 3)} at ${T} K (${s})`).join('; ')}. Read from Fig. 3(D) of López-Borrell et al. (2024, Polymers 16, 2335): own hygrometer measurement (±2 % RH; the authors note their values are 3–8 % high) and replotted literature points — digitised, ±0.03 in the ratio. A measured isopiestic point just above saturation (3.14 mol/kg, φ 0.9669, Miladinović et al. 2007) corresponds to a_w ${fmt(Math.exp(-2 * 3.14 * 0.9669 * MW_W), 4)}; model at 3.14 mol/kg ${fmt(q(3.14).aw, 4)}. The tabulated vapour pressures of Apelblat & Manzurola (2003, J. Chem. Thermodyn. 35, 221) could not be read`);
          { // second pass: measured water activities up to saturation, further solubilities, deliquescence humidity
            const zh = MGSO4_REF.zh, phiOf = ([m, a]) => -Math.log(a) / (2 * m * MW_W), dA = dv(zh, ([m, a]) => Math.abs(q(m).aw - a)), dZ = dv(zh, (r) => Math.abs(q(r[0]).phi / phiOf(r) - 1)), near = zh.at(-1), lo = zh.at(-2), aSat = eps.sol.eq.aw, awFrom = (m, ph) => Math.exp(-2 * m * ph * MW_W);
            add('MgSO₄ 1.11–2.99 mol/kg, Pitzer model: water activity against the isopiestic measurements of Zhang et al. (2016)', 0, dA, 5e-4, `Largest difference over four pure-MgSO₄ solutions equilibrated against CaCl₂ at 298.15 K: a_w ${zh.map(([m, a]) => `${a} at ${m} mol/kg (suite ${fmt(q(m).aw, 4)})`).join(', ')}. Zhang, Li, Yao, Sun, Zeng & Song (2016, J. Chem. Eng. Data 61, 2277) — primary measurements, read through the NIST ThermoML transcription (the article itself not read); a laboratory independent of the Livermore and Belgrade series. In φ = −ln a_w/(2·m·M_w): ${zh.map((r) => fmt(phiOf(r), 4)).join(', ')}, largest deviation ${fmt(100 * dZ, 2)} %`);
            add('MgSO₄ at 2.9876 mol/kg, next to epsomite saturation: water activity against the measured value', near[1], q(near[0]).aw, 0.001, `Measured a_w ${near[1]} (same series; four decimals) at a molality ${fmt(near[0] - eps.molality, 2)} mol/kg above the model’s saturation and inside the range of the measured solubilities (2.977–3.18 mol/kg). At the next lower water activity of that series, 0.8789, no pure MgSO₄ solution was prepared — the most MgSO₄-rich one holds 2.64 mol/kg with 0.66 mol/kg LiCl. An independently measured water activity of the solution at saturation, not derived from the unread evaluations`);
            const awSatMeas = near[1] + ((lo[1] - near[1]) / (near[0] - lo[0])) * (near[0] - eps.molality);
            add('Epsomite-saturated solution, 25 °C: water activity against the measured isopiestic points interpolated to the model’s saturation molality', awSatMeas, aSat, 0.001, `Linear interpolation between the two highest measured points of Zhang et al. (${lo[1]} at ${lo[0]} and ${near[1]} at ${near[0]} mol/kg) to ${fmt(eps.molality, 4)} mol/kg: ${fmt(awSatMeas, 4)}; suite ${fmt(aSat, 4)}. Carried instead to the measured solubilities with the measured slope between ${near[0]} and 3.14 mol/kg (a_w ${fmt(awFrom(3.14, MGSO4_REF.iso.at(-1)[1]), 4)}, Miladinović et al. 2007): ${[...MGSO4_REF.sol2, MGSO4_REF.sol[0]].map(([nm, m]) => `${fmt(near[1] + ((awFrom(3.14, MGSO4_REF.iso.at(-1)[1]) - near[1]) / (3.14 - near[0])) * (m - near[0]), 4)} at ${m} mol/kg (${nm})`).join(', ')}`);
            const all = [...MGSO4_REF.wrs.filter((r) => r[0] < 2.5), ...MGSO4_REF.iso, ...MGSO4_REF.iv, ...zh.map((r) => [r[0], phiOf(r)])], dAll = dv(all, ([m, ph]) => Math.abs(q(m).phi / ph - 1)), dAw = dv(all, ([m, ph]) => Math.abs(q(m).aw - awFrom(m, ph)));
            add('MgSO₄ 0.57–3.14 mol/kg, Pitzer model: largest deviation of φ from all measured isopiestic points held', 0, dAll, 0.011, `${all.length} points of four laboratories (Wu, Rush & Scatchard 1968 against NaCl, 9; Miladinović et al. 2007 against KCl, 6; Ivanović et al. 2017 against KCl, 12; Zhang et al. 2016 against CaCl₂, 4): largest deviation ${fmt(100 * dAll, 2)} % in φ and ${fmt(dAw, 2)} in a_w. This is the direct validation of the Mg–SO₄ representation in use; it involves neither Archer & Rard (1998) nor Rard & Miller (1981). The tenth point of Wu, Rush & Scatchard, at 2.59 mol/kg, is treated in its own check`);
            add('MgSO₄ 0.29–0.77 mol/kg: water activity against the isopiestic values of Jahani et al. (2014)', 0, dv(MGSO4_REF.jah, ([m, a]) => Math.abs(q(m).aw - a)), 0.004, `Loose check of the dilute end: five values given to three decimals with ±0.004 (ThermoML transcription of J. Chem. Thermodyn. 69, 125) — ${MGSO4_REF.jah.map(([m, a]) => `${a} at ${m} (suite ${fmt(q(m).aw, 4)})`).join(', ')}. They lie 0.002–0.003 above the suite and above the other series (0.9897 at 0.57 mol/kg from Wu, Rush & Scatchard), within their stated uncertainty`);
            const six = [...MGSO4_REF.sol.map((r) => r[1]), ...MGSO4_REF.sol2.map((r) => r[1])].sort((a, b) => a - b), med = 0.5 * (six[2] + six[3]);
            add('Epsomite solubility in water, 25 °C: against the median of six measurements of 2016–2018', med, eps.molality, 0.1, `mol/kg: ${[...MGSO4_REF.sol, ...MGSO4_REF.sol2].map(([nm, x]) => `${x} (${nm})`).join(', ')}; median ${fmt(med, 4)}, the suite is ${fmt(100 * (1 - eps.molality / med), 2)} % below it and within ${fmt(100 * (1 - eps.molality / six[0]), 2)} % of the lowest. Single-laboratory values from the NIST ThermoML archive; no critical evaluation of the solubility is openly available (the IUPAC–NIST Solubility Data Series has no volume on magnesium sulphate)`);
            const satT = MGSO4_REF.zal.map(([T]) => (T === 25 ? eps : solubility(makeSolution({ ions: cloneIons({}), T, pH: 7 }), 'epsomite', { excess: 8 })));
            add('Epsomite solubility at 20, 25 and 30 °C: against Zhang, Asselin & Li (2016)', 0, Math.max(...MGSO4_REF.zal.map(([, m], k) => Math.abs(satT[k].molality / m - 1))), 0.05, `Largest relative deviation; measured ${MGSO4_REF.zal.map(([T, m]) => `${m} at ${T} °C`).join(', ')} mol/kg (±0.042), suite ${satT.map((r) => fmt(r.molality, 4)).join(', ')} — the suite is ${MGSO4_REF.zal.map(([, m], k) => fmt(100 * (1 - satT[k].molality / m), 2)).join(', ')} % lower`);
            add('Epsomite-saturated solution, 20 °C: water activity against a measured deliquescence humidity', MGSO4_REF.drh20, satT[0].sol.eq.aw, 0.025, `Deliquescence of MgSO₄·7H₂O observed at 92 % relative humidity at 20 °C (Barlas et al. 2023, Plants 12, 2357, Table 1; open, read): climate chamber raised in steps of 5 % RH with microscope observation, and a vapour-sorption ramp; the authors call the value approximate and give no uncertainty, hence the tolerance. Suite: ${fmt(100 * satT[0].sol.eq.aw, 3)} % at 20 °C (saturation ${fmt(satT[0].molality, 3)} mol/kg). A direct measurement on the saturated solution, independent of every osmotic-coefficient source`);
            add('Epsomite deliquescence humidity at 20, 25 and 30 °C: against the values calculated by Steiger et al. (2011) (model values, secondary)', 0, Math.max(...MGSO4_REF.steiger.map(([, a], k) => Math.abs(satT[k].sol.eq.aw - a))), 0.007, `Largest difference in a_w; ${MGSO4_REF.steiger.map(([T, a], k) => `${T} °C: ${fmt(100 * a, 3)} % against ${fmt(100 * satT[k].sol.eq.aw, 3)} %`).join(', ')}. Not measurements: phase-diagram values of the ion-interaction model of Steiger, Linnow, Ehrhardt & Rohde (2011, Geochim. Cosmochim. Acta 75, 3600; article not read), as tabulated on the salt-damage wiki of HAWK Hildesheim. The suite is 0.4–0.6 % RH higher at each temperature, in line with its lower saturation molality`);
          }
        }
      }
      const nn = { c: 'Na', a: 'NO3', d: [[0.1, 0.76, 0.921], [0.5, 0.618, 0.876], [1, 0.549, 0.852], [2, 0.478, 0.826], [3, 0.437, 0.81], [4, 0.408, 0.798], [5, 0.386, 0.789], [6, 0.372, 0.789]] }, qn = actDeviation(nn, 'pitzer', 6);
      add('NaNO₃ 0.1–6 mol/kg, Pitzer model (replaced parameter set): γ± and φ against Hamer & Wu 1972', 0, Math.max(qn.dg, qn.dp), 0.03, `γ± ${fmt(100 * qn.dg, 2)} %, φ ${fmt(100 * qn.dp, 2)} %`);
    }
    { // barite and celestite solubility in water and in NaCl solutions (Pitzer model)
      const nacl = (m) => { const n = new Float64Array(NM); n[mi('Na')] = m; n[mi('Cl')] = m; return equilibrate({ T: 25, model: 'pitzer', n, alk: 0, w: 1, pH: 7 }); };
      const rms = (a) => Math.sqrt(sum(a.map((x) => Math.log10(1 + x) ** 2)) / a.length), pc = (x) => fmt(100 * x, 2), mx = (rows, f = (r) => r[1]) => Math.max(...rows.map((r) => Math.abs(f(r))));
      const KS2 = 10 ** BARITE_MODELS.pair.sdKsp - 1, KPAIR = 10 ** (BARITE_MODELS.pair.logK + BARITE_MODELS.pair.logKsp) * 1000; // 12 %: twice the stated ±0.05 of the published log Ksp in solubility terms (√Ksp); BaSO4(aq) at barite saturation, mmol/kg
      const water = (T, B) => equilibrate({ T, model: 'pitzer', n: new Float64Array(NM), alk: 0, w: 1, pH: 7 }, { bariteModel: B }), bsat = (sol, P = 1) => solubility(sol, 'barite', { excess: 0.01, P });
      const bset = (B) => { // every barite solubility comparison under one treatment of the Ba–SO4 interaction
        const bn = (c) => bsat(equilibrate(nacl(c), { bariteModel: B })).m * 1000, bw = (T, P) => bsat(water(T, B), P).m * 1000, sN = BARITE_NACL.map(([c]) => [c, bn(c)]), at = (c) => sN.find((r) => r[0] === c)[1], ba = BARITE_NACL.map(([c, x], i) => [c, sN[i][1] / x - 1]);
        return { sN, ba, hi: ba.filter((r) => r[0] >= 1 && r[0] <= 4), lo: ba.filter((r) => r[0] > 0 && r[0] < 1), dP: BARITE_BLOUNT.P.map(([P, x]) => [P, bw(25, P) / x - 1]), dT: BARITE_BLOUNT.T.map(([T, x]) => [T, bw(T, 1) / x - 1]),
          pu: BARITE_BLOUNT.nacl.map(([c, x]) => [c, x, at(c), BARITE_NACL.find((r) => r[0] === c)[1]]), dPu: BARITE_BLOUNT.fig9P.map(([c, x]) => [c, bn(c) / x - 1]), dc: BARITE_DC.slice(1).map(([c, x]) => [c, x, BARITE_NACL.find((r) => r[0] === c)[1], at(c)]) };
      };
      const DEF = BARITE_MODEL_DEFAULT, CMP = DEF === 'pair' ? 'analogue' : 'pair', BS = Object.fromEntries([...new Set([DEF, 'pair', 'analogue'])].map((B) => [B, bset(B)])), other = (B) => (B === DEF ? CMP : DEF);
      for (const B of [DEF, CMP]) { // the default treatment first, then the same checks under the published ion-pair constants (or, were those the default, under the analogue), so that the effect of the choice is on record
        const s = BS[B], o = BS[other(B)], pr = B === 'pair', tag = B === DEF ? '' : ` — ${BA_SHORT[B]} treatment (${pr ? 'published constants' : 'comparison'})`, on = BA_SHORT[other(B)], wide = pr ? KS2 : 0;
        const ks = pr ? 'log Ksp −10.05 ± 0.05 with the BaSO₄(aq) pair (Felmy et al. 1990)' : B === 'pairw' ? 'log Ksp −9.965, fitted in this work to these three values, with the BaSO₄(aq) pair of Felmy et al. (1990)' : 'log Ksp −9.97 (WATEQ4F)', wt = pr ? ` Tolerance: that used for the analogue treatment plus ${pc(KS2)} % (twice the stated ±0.05 of the published log Ksp, in solubility terms): the published constant puts the model about 9 % lower than log Ksp −9.97 in every chloride medium` : '';
        add('Barite solubility in water, 25 °C' + tag, 0.0107, s.sN[0][1], pr ? 0.0107 * KS2 : 0.0004, `mmol/kg. Templeton (1960): 0.0108 (as entered by Appelo 2015); Blount (1977, Am. Mineral. 62, 942, Table 3): 0.0106; Davis & Collins (1971, as tabulated by Dal Pozzo 1991): 0.011. Tests ${ks}; no activity model parameter matters at this dilution. Model ${fmt(s.sN[0][1], 3)} (${pc(s.sN[0][1] / 0.0107 - 1)} %); with the ${on} treatment ${fmt(o.sN[0][1], 3)} (${pc(o.sN[0][1] / 0.0107 - 1)} %).${pr ? ` The published solubility product lies below all three measurements; tolerance = twice its stated ±0.05, in solubility terms (${pc(KS2)} %)` : ''}`);
        add('Barite solubility in 1–4 mol/kg NaCl' + tag, 0, mx(s.hi), 0.08 + wide, `Largest relative deviation from the table of Templeton (1960) at ${s.hi.length} NaCl concentrations: ${s.hi.map((r) => `${r[0]}: ${pc(r[1])} %`).join(', ')}. Depends on the Ba–Cl and Na–SO₄ parameters and the solubility product only (no θ(Ba,Na) or ψ in the set). With the ${on} treatment: ${pc(mx(o.hi))} %.${wt}`);
        add('Barite solubility in 0.05–5 mol/kg NaCl: root-mean-square deviation' + tag, 0, rms(s.ba.slice(1).map((r) => r[1])), 0.06 + (pr ? BARITE_MODELS.pair.sdKsp : 0), `In log₁₀ units over 15 concentrations of the Templeton table. Below 1 mol/kg the model is lower than the table (${s.lo.map((r) => `${r[0]}: ${pc(r[1])} %`).join(', ')}): at 0.05 mol/kg the tabulated rise over pure water (factor 2.8) exceeds what the Debye–Hückel term allows (2.3), which no specific-interaction parameter can change. With the ${on} treatment: ${fmt(rms(o.ba.slice(1).map((r) => r[1])), 2)}.${pr ? ' Tolerance: that used for the analogue treatment plus 0.05 (twice the stated uncertainty of the published log Ksp, halved for the solubility)' : ''}`);
        add('Barite solubility in water at 25 °C and 100–1000 bar (Blount 1977, Table 3)' + tag, 0, mx(s.dP), 0.05 + wide, `Largest relative deviation; ${s.dP.map(([P, d]) => `${P} bar ${pc(d)} %`).join(', ')}. Primary table (0.0117, 0.0184, 0.0290 mmol/kg; his runs at 24 °C gave 0.0289 at 1002 bar). Tests the reaction volume −50.6 cm³/mol together with the solubility product. With the ${on} treatment: ${pc(mx(o.dP))} %.${wt}`);
        add('Barite solubility in water at 60 and 100 °C, 1 bar (Blount 1977, Table 3)' + tag, 0, mx(s.dT), 0.09, `Largest relative deviation; ${s.dT.map(([T, d]) => `${T} °C ${pc(d)} %`).join(', ')}. Primary table (0.0152 from Melcher and Templeton, 0.0168 measured by Blount). Tests log K(T) of WATEQ4F${pr ? ', here shifted to −10.05 at 25 °C (the temperature function is an assumption of this suite, not of Felmy et al.)' : ''}. With the ${on} treatment: ${pc(mx(o.dT))} %`);
        add('Barite solubility in 0.2 and 1.0 mol/kg NaCl, 25 °C (Blount 1977, Table 11)' + tag, 0, mx(s.pu, ([, x, g]) => g / x - 1), 0.17, `Largest relative deviation from the measurements of Puchelt (1967) as tabulated by Blount: ${s.pu.map(([c, x, g]) => `${c} mol/kg ${fmt(g, 3)} against ${x}`).join(', ')} mmol/kg. Blount adopts this series, which lies 20–29 % below Templeton’s. With the ${on} treatment: ${pc(mx(o.pu, ([, x, g]) => g / x - 1))} %`);
        add('Barite in 0.2 and 1.0 mol/kg NaCl: position of the model between the Puchelt and Templeton series' + tag, 0.5, mx(s.pu, ([, x, g, t]) => (g - x) / (t - x) - 0.5) + 0.5, 0.5, `0 = Puchelt (Blount Table 11), 1 = Templeton: ${s.pu.map(([c, x, g, t]) => `${c} mol/kg ${fmt((g - x) / (t - x), 2)}`).join(', ')}; with the ${on} treatment ${o.pu.map(([c, x, g, t]) => fmt((g - x) / (t - x), 2)).join(', ')}. The two measurement series disagree by more than the two treatments differ: with log Ksp −9.97 the model sits between them, with −10.05 close to Puchelt’s`);
        add('Barite solubility in 0.5–4.4 mol/kg NaCl against Puchelt’s points in Fig. 9 of Blount (1977), digitised' + tag, 0, mx(s.dPu.filter((r) => r[0] >= 0.5)), 0.22, `Largest relative deviation over four points; all seven: ${s.dPu.map(([c, d]) => `${fmt(c, 3)}: ${pc(d)} %`).join(', ')}. The model is above this series throughout; its two lowest points (0.10 and 0.21 mol/kg, 0.0215 and 0.0303 mmol/kg) scatter below Blount’s own curve and the point he tabulates (0.037 at 0.2). Molalities read from the figure are uncertain by ±8 %. With the ${on} treatment: ${pc(mx(o.dPu.filter((r) => r[0] >= 0.5)))} %`);
        add('Barite solubility in 1 and 2 mol/kg NaCl against the mean of two measurement series' + tag, 0, mx(s.dc, ([, x, t, g]) => g / (0.5 * (x + t)) - 1), 0.1, `mmol/kg, model (Davis & Collins 1971; Templeton 1960): ${s.dc.map(([c, x, t, g]) => `${c} mol/kg ${fmt(g, 3)} (${x}; ${t})`).join(', ')}. Both series as tabulated on the molal scale by Dal Pozzo (1991, Table C.2) — secondary; they differ from each other by 14 %. With the ${on} treatment: ${pc(mx(o.dc, ([, x, t, g]) => g / (0.5 * (x + t)) - 1))} %`);
      }
      { // the Templeton table itself against his points in Blount's Fig. 9 (independent of the model)
        const tab = (c) => { const j = BARITE_NACL.findIndex((r) => r[0] >= c), [c0, x0] = BARITE_NACL[j - 1], [c1, x1] = BARITE_NACL[j]; return x0 + ((x1 - x0) * (c - c0)) / (c1 - c0); }, dg = BARITE_BLOUNT.fig9T.map(([c, x]) => [c, tab(c) / x - 1]);
        add('Templeton (1960) table in use against Templeton’s points in Fig. 9 of Blount (1977), digitised', 0, Math.max(...dg.map((r) => Math.abs(r[1]))), 0.045, `Largest relative difference over ${dg.length} points from 0.05 to 3.2 mol/kg (${dg.map(([c, d]) => `${fmt(c, 2)}: ${pc(d)} %`).join(', ')}); table interpolated linearly. Digitised from the journal figure (reading uncertainty ±0.01 in log₁₀, i.e. 2.3 %, and ±8 % in molality, worth another 3 % in solubility): an independent reproduction of the second-hand table, not the original`);
      }
      { // all primary points pooled: the evidence behind the choice of the default treatment
        const pool = (s) => [...[0.0108, 0.011, 0.0106].map((x) => s.sN[0][1] / x - 1), ...s.ba.slice(1).map((r) => r[1]), ...s.dc.map(([, x, , g]) => g / x - 1), ...s.pu.map(([, x, g]) => g / x - 1), ...s.dPu.filter((r) => Math.abs(r[0] - 0.213) > 1e-9 && Math.abs(r[0] - 1.074) > 1e-9).map((r) => r[1]), ...s.dP.map((r) => r[1]), ...s.dT.map((r) => r[1])], pp = pool(BS.pair), pa = pool(BS.analogue);
        add('Barite solubility, all ' + pp.length + ' primary points pooled: ion-pair model with log Ksp −10.05 against log Ksp −9.97', rms(pa), rms(pp), 0.012, `Root-mean-square deviation in log₁₀ over water (3 values), Templeton’s NaCl table (15), Davis & Collins (2), Puchelt (2 tabulated, 5 digitised), Blount’s pressure (3) and temperature (2) points: ${fmt(rms(pp), 2)} with the published ion-pair model, ${fmt(rms(pa), 2)} with the Ca–SO₄ analogue and log Ksp −9.97 (the default); mean deviation ${fmt(sum(pp.map((x) => Math.log10(1 + x))) / pp.length, 2)} against ${fmt(sum(pa.map((x) => Math.log10(1 + x))) / pa.length, 2)}. The published model is worse in water, against Templeton and under pressure, better against Davis & Collins, Puchelt and at 60–100 °C: these media do not decide between the two solubility products; the sulphate-bearing media of the next checks do`);
      }
      { // sulphate-bearing media: the measurements that separate the treatments (BARITE_SULPHATE, bariteEvidence)
        const ev = bariteEvidence(), M = Object.fromEntries(ev.media.map((q) => [q.id, q])), D = BARITE_MODEL_DEFAULT, rel = (id, B = D) => ev.points.filter((p) => p.medium === id).map((p) => [p.cond, p.model[B] / p.meas - 1]);
        const st = (s) => BARITE_TREATMENT_IDS.map((B) => `${BA_SHORT[B]} ${fq(s[B].bias, 2)} / ${fq(s[B].rms, 2)}`).join('; '), list = (id) => rel(id).map((r) => pc(r[1])).join(', '), rmsMin = Math.min(...BARITE_TREATMENT_IDS.map((B) => ev.sulphate[B].rms));
        add('Barite in 0.3–7.9 mmol/kg Na₂SO₄ at 20 °C (Jiang 1996, 12 points read from a figure): largest deviation of the default treatment', 0, mx(rel('jiang')), 0.12, `Relative deviation of the dissolved barium, ${BA_SHORT[D]}: ${list('jiang')} %. Points of Jiang (1996, J. Solution Chem. 25, 105) as plotted in Fig. 4-5 of García (2005, PhD thesis, Technical University of Denmark, p. 51), read from the vector drawing of the page (resolution about 2 nmol/kg, i.e. 0.5–4 %); the article itself was not read. Mean / rms of log₁₀(model/measured): ${st(M.jiang.stat)}`);
        add('Barite in Na₂SO₄–NaNO₃ solutions of ionic strength 0.05 at 22 °C (Savenko et al. 2019, Table 1, 7 points): rms deviation of the default treatment', 0, M.na2so4.stat[D].rms, 0.08, `In log₁₀; relative deviations ${list('na2so4')} % at 1–12.5 mmol/L sulphate. Recrystallised BaSO₄, 13 months, ICP-MS ± 3 % (Savenko, Savenko & Pokrovsky 2019, Okeanologiya 59, 939; primary table). Every treatment is 23–49 % high at the lowest sulphate concentration. The suite has no Ba–NO₃ binary; the published one would lower every treatment by about 0.02. Mean / rms: ${st(M.na2so4.stat)}`);
        add('Barite in seawater of salinity 5–35 at 22 °C (Savenko et al. 2019 and 2023, 12 points): largest deviation of the default treatment', 0, mx(rel('sw')), 0.2, `Relative deviations ${list('sw')} % (salinity 5 to 35; three waters at 35). The default dissolves ${pc(-rel('sw').at(-1)[1])} % less barium than measured at salinity 35, i.e. its barite index is ${fq(-Math.log10(1 + rel('sw').at(-1)[1]), 2)} too high — on the safe side. Seawater composition of Culberson et al. (1978) scaled to the salinity; µmol/L converted with the density at 22 °C. Burton, Marshall & Phillips (1968) found 48 ± 3 µg/L at 20 °C (quoted by Savenko et al. and, as 350 nmol/kg, by Rogers 1981 — secondary), 56 % above the 30.7 µg/L of Savenko et al. Mean / rms: ${st(M.sw.stat)}`);
        add('Barite in the feed and concentrate of a reverse-osmosis pilot plant at 25 °C (Boerlage 2001, Tables 2.1 and 2.2, 3 points): largest deviation of the default treatment', 0, mx(rel('ro')), 0.22, `Relative deviations ${list('ro')} % for the feed and the concentrates at 80 and 90 % recovery (measured 81, 42 and 34 µg/L, ± 10 %; sulphate 0.6, 3 and 6 mmol/L). Seeded tests of 3–24 h in pretreated Rhine water with 1–2 mg/L of organic carbon; the thesis itself derives a solubility product of 1.34·10⁻¹⁰, 24 % above the pure-water value. All treatments lie below the measurement, i.e. on the safe side. Mean / rms: ${st(M.ro.stat)}`);
        add('Barite in 0.3–79 mmol/kg sulphuric acid at 25 and 60 °C (Paige 1990, Tables 7 and 8, 7 points): largest deviation of the default treatment', 0, mx(rel('h2so4')), 0.17, `Relative deviations ${list('h2so4')} % (25 °C: 0.3, 1, 3 and 79 mmol/kg; 60 °C: 1, 3 and 79 mmol/kg). Primary measurements read from the page images of the thesis of Paige (1990, McMaster University, pp. 70–71): ¹³³Ba-labelled barite, five months from undersaturation at 25 °C, from supersaturation at 60 °C, 0.2 µm filtration, γ counting; 95 % confidence interval 1–2 % where given. The medium is modelled as H₂SO₄ with the H–SO₄–HSO₄ parameters of Harvie, Møller & Weare (computed pH 3.3 to 1.1); total sulphate reaches 79 mmol/kg, and these are the only measured points of the set above 25 °C in a sulphate medium. Mean / rms: ${st(M.h2so4.stat)}`);
        { const hi = ev.points.filter((p) => p.medium === 'lieser').slice(6), r6 = Object.fromEntries(BARITE_TREATMENT_IDS.map((B) => [B, Math.sqrt(sum(hi.map((p) => Math.log10(p.model[B] / p.meas) ** 2)) / hi.length)]));
          add('Barite in 0.055–1.1 mol/kg Na₂SO₄ at 20 °C (Lieser 1965, 6 points digitised second-hand, not counted): rms deviation of the default treatment against the best of the four', Math.min(...Object.values(r6)), r6[D], 0.005, `In log₁₀: ${BARITE_TREATMENT_IDS.map((B) => `${BA_SHORT[B]} ${fq(r6[B], 2)}`).join('; ')}. The only points obtained above 30 mmol/kg of sulphate in a neutral medium; Paige (1990, Table 6) digitised them from Lieser’s graph, and both Felmy et al. (1990, abstract) and, according to Paige, Monnin regard the series as too high — at 0.5–22 mmol/kg it lies ${fq(Math.min(...rel('lieser').slice(0, 6).map((r) => 1 / (1 + r[1]))), 2)} to ${fq(Math.max(...rel('lieser').slice(0, 6).map((r) => 1 / (1 + r[1]))), 2)} times above the default, in line with that. At the high end the sign reverses: the default dissolves ${list('lieser').split(', ').slice(8).join(', ')} % more barium than the digitised points at 0.23, 0.56, 0.91 and 1.1 mol/kg, the ion pair and the no-term treatment ${pc(ev.points.filter((p) => p.medium === 'lieser').at(-1).model.pair / ev.points.filter((p) => p.medium === 'lieser').at(-1).meas - 1)} and ${pc(ev.points.filter((p) => p.medium === 'lieser').at(-1).model.none / ev.points.filter((p) => p.medium === 'lieser').at(-1).meas - 1)} % at 1.1 mol/kg. Taken at face value the series supports the analogue over the other treatments in sulphate brines and says that even it may report a barite index too low by up to ${fq(Math.log10(1 + rel('lieser').at(-1)[1]), 2)} there; being a second-hand digitisation of a contested series it is listed, not counted. Whole series, mean / rms: ${st(M.lieser.stat)}`); }
        add('Barite in 0.5–8.1 mmol/kg Na₂SO₄ (Felmy et al. 1990, 12 points digitised second-hand, not counted): mean deviation of the ion pair with the constants published from these very data', 0.21, M.felmy.stat.pair.bias, 0.03, `log₁₀(model/measured) at the 20 °C of the table heading (at 25 °C, the temperature named in the abstract of Felmy et al., 0.08 more). Paige (1990, Table 6) digitised the points from a graph of Felmy et al.; ${ev.points.filter((p) => p.medium === 'felmy' && p.meas < 46.8).length} of the 12 lie below K·Ksp = 46.8 nmol/kg, the concentration of the BaSO₄(aq) pair alone under the published constants (log K 2.72 ± 0.09, log Ksp −10.05 ± 0.05), and the series is not monotonic. A model fitted to these data cannot lie 0.2 above them on average, so the digitised points do not represent the data behind the published constants; they are listed in the results table and kept out of every statistic. All treatments, mean / rms: ${st(M.felmy.stat)}. Had they been counted: default ${fq(Math.sqrt((ev.sulphate[D].rms ** 2 * ev.nSulphate + M.felmy.stat[D].rms ** 2 * 12) / (ev.nSulphate + 12)), 2)}, no Ba–SO₄ term ${fq(Math.sqrt((ev.sulphate.none.rms ** 2 * ev.nSulphate + M.felmy.stat.none.rms ** 2 * 12) / (ev.nSulphate + 12)), 2)}, ion pair ${fq(Math.sqrt((ev.sulphate.pair.rms ** 2 * ev.nSulphate + M.felmy.stat.pair.rms ** 2 * 12) / (ev.nSulphate + 12)), 2)} rms in sulphate media — the default would still have the smallest deviation, and the conservative envelope would lie up to 0.18 on the unsafe side of these points`);
        add('Barite in 0.45–6.2 mol/kg sulphuric acid at 25 and 60 °C (Paige 1990, 12 points, not counted): mean excess of the default treatment — a limit of the model', 0.73, M.h2so4c.stat[D].bias, 0.05, `log₁₀(model/measured), i.e. the suite dissolves ${fq(10 ** Math.min(...ev.points.filter((p) => p.medium === 'h2so4c').map((p) => Math.log10(p.model[D] / p.meas))), 2)} to ${fq(10 ** Math.max(...ev.points.filter((p) => p.medium === 'h2so4c').map((p) => Math.log10(p.model[D] / p.meas))), 2)} times the measured barium (every treatment: ${st(M.h2so4c.stat)}). Recorded as a negative result: in sulphuric acid of 0.45 mol/kg and more (computed pH below 0.5) the barite index of the suite is too low by 0.3 to 1.3 and must not be used; the 25 °C parameter set carries no Ba–HSO₄ interaction and is outside its range in such acid. The seven points up to 79 mmol/kg are within 15 %`);
        add(`Barite in sulphate-bearing media, ${ev.nSulphate} measured points pooled: rms deviation of the default treatment against the best of the four`, rmsMin, ev.sulphate[D].rms, 0.005, `In log₁₀ (mean / rms): ${st(ev.sulphate)}. Water and NaCl solutions (${ev.nPlain} points): ${st(ev.plain)}. All ${ev.n} points: ${st(ev.pooled)}. The default (${BA_SHORT[D]}) has the smallest rms in the sulphate media and overall; this is the evidence on which it is kept`);
        add('Barite in sulphate-bearing media: mean deviation of the ion pair combined with the log Ksp fitted to pure water', 0.1, ev.sulphate.pairw.bias, 0.02, `In log₁₀, i.e. ${pc(10 ** ev.sulphate.pairw.bias - 1)} % too much dissolved barium on average and ${pc(rel('jiang', 'pairw').at(-1)[1])} % at 7.9 mmol/kg Na₂SO₄: recorded as a negative result. The association constant 2.72 of Felmy et al. (1990) belongs with the log Ksp −10.05 it was derived with; combined with a solubility product that fits pure water it counts the pair on top of a free-ion solubility that already matches, and the measurements in sulphate media reject it`);
        add('Barite verdicts: mean position of the conservative envelope against the measurements in sulphate-bearing media', -0.05, ev.envelope.bias, 0.03, `log₁₀ of (lowest solubility among the published treatments / measured) over ${ev.nSulphate} points: mean ${fq(ev.envelope.bias, 2)}, range ${fq(ev.envelope.min, 2)} to ${fq(ev.envelope.max, 2)}. Negative = the envelope reports a higher barite index than the measurement implies (safe side)`);
        add('Barite verdicts: largest amount by which the conservative envelope dissolves more barium than was measured', 0, Math.max(0, ev.envelope.max), 0.1, `log₁₀; ${ev.envelope.over} of ${ev.nSulphate} points exceed +0.02 (${ev.points.filter((p) => M[p.medium].sulphate && M[p.medium].counted && Math.min(...BARITE_PUBLISHED.map((B) => p.model[B])) / p.meas > 10 ** 0.02).map((p) => p.cond).join('; ')}) — at the low-sulphate end of the Na₂SO₄–NaNO₃ series, where every treatment is high. Everywhere else among the counted points the envelope is at or on the safe side of the measurement, so the verdict basis is kept. Outside the counted set it is not on the safe side: by up to 0.18 against the second-hand digitisation of Felmy et al., by 0.1–0.4 against Lieser’s digitised points at 0.23–1.1 mol/kg Na₂SO₄, and by 0.3–1.3 in sulphuric acid of 0.45 mol/kg and more (checks above)`);
        add('Seawater diluted to salinity 0.35–2.1 (Savenko et al. 2023, 4 points, not counted): mean deviation under the ion-pair constants the authors themselves use', 0.17, M.swd.stat.pair.bias, 0.03, `log₁₀(model/measured) with log K 2.72 and log Ksp −10.05 (Savenko et al. 2019 reduce their Na₂SO₄ series with this association constant and obtain log Ksp −10.12 at 22 °C): ${rel('swd', 'pair').map((r) => pc(r[1])).join(', ')} % at salinity 0.35, 0.7, 1.05 and 2.1; all treatments: ${st(M.swd.stat)}. In these four waters the measured barium is below what the authors’ own solubility product allows for the sulphate present, by a factor that grows on dilution — the barium is not fixed by the medium alone (the solid, 50 g/L, is a possible source of sulphate), so the points are listed in the results table but kept out of every statistic`);
      }
      { // the three treatments in the limit of no sulphate excess, the pair at saturation, and the barium balance
        const cs = [0, 0.1, 1, 4], sZ = cs.map((c) => bsat(equilibrate(nacl(c), { bariteModel: 'none' })).m * 1000), sA = cs.map((c) => BS.analogue.sN.find((r) => r[0] === c)[1]), sP = cs.map((c) => BS.pair.sN.find((r) => r[0] === c)[1]);
        add('Stoichiometric barite in 0–4 mol/kg NaCl: Ca–SO₄ analogue against no Ba–SO₄ term', 0, Math.max(...cs.map((c, i) => Math.abs(Math.log10(sA[i] / sZ[i])))), 1e-3, `log₁₀ of the solubility ratio at ${cs.join(', ')} mol/kg NaCl: ${cs.map((c, i) => fq(Math.log10(sA[i] / sZ[i]), 2)).join(', ')}. Without a sulphate excess the dissolved sulphate is 10⁻⁵–10⁻⁴ mol/kg and the Ba–SO₄ binary has nothing to act on`);
        add('Stoichiometric barite in 0–4 mol/kg NaCl: ion-pair model against the other treatments once each uses its own log Ksp', 0, Math.max(...cs.map((c, i) => Math.abs(Math.log10((sP[i] - KPAIR) / sZ[i]) - DK_PAIR / 2))), 1e-3, `log₁₀[(solubility − BaSO₄(aq))/solubility without a Ba–SO₄ term] − ½·Δlog Ksp, Δlog Ksp = ${fmt(DK_PAIR, 3)}: ${cs.map((c, i) => fq(Math.log10((sP[i] - KPAIR) / sZ[i]) - DK_PAIR / 2, 2)).join(', ')} (against the analogue: ${cs.map((c, i) => fq(Math.log10((sP[i] - KPAIR) / sA[i]) - DK_PAIR / 2, 2)).join(', ')}). The free-ion solubilities coincide when each treatment is given its own solubility product; the residual in pure water is the activity coefficient at the 9 % lower ionic strength of the solution itself. As computed the ion-pair model is ${cs.map((c, i) => pc(sP[i] / sA[i] - 1)).join(', ')} % lower (BaSO₄(aq) adds ${fmt(KPAIR, 3)} mmol/kg, the lower log Ksp takes ${pc(1 - 10 ** (DK_PAIR / 2))} %). The treatments differ in sulphate media only`);
        const iP = si('BaSO4°'), na2 = new Float64Array(NM); na2[mi('Na')] = 0.2; na2[mi('SO4')] = 0.1;
        const sw0 = makeSolution({ ions: WATERS.seawater.ions, T: 25, pH: 8.1, bariteModel: 'pair' }), media = [['water', water(25, 'pair')], ['1 mol/kg NaCl', equilibrate(nacl(1), { bariteModel: 'pair' })], ['0.1 mol/kg Na₂SO₄', equilibrate({ T: 25, model: 'pitzer', n: na2, alk: 0, w: 1, pH: 7 }, { bariteModel: 'pair' })], ['seawater', sw0]].map(([nm, s]) => [nm, bsat(s)]);
        add('Ion-pair model: BaSO₄(aq) at barite saturation equals K·Ksp in every medium', 1, Math.max(...media.map(([, r]) => Math.abs((r.sol.eq.m[iP] * 1000) / KPAIR - 1))) + 1, 1e-6, `m(BaSO₄°) = 10^(2.72 − 10.05) = ${fq(KPAIR / 1000, 3)} mol/kg (activity coefficient of the neutral pair 1) in ${media.map(([nm]) => nm).join(', ')}: ${media.map(([, r]) => fq(r.sol.eq.m[iP], 4)).join(', ')}. Share of the dissolved barium that is paired: ${media.map(([nm, r]) => `${nm} ${pc(r.sol.eq.m[iP] / r.sol.eq.tot[mi('Ba')])} %`).join(', ')}`);
        const na2s = (B) => bsat(equilibrate({ T: 25, model: 'pitzer', n: na2, alk: 0, w: 1, pH: 7 }, { bariteModel: B })).sol.eq.tot[mi('Ba')] * 1e9, n2 = { pair: na2s('pair'), analogue: na2s('analogue'), none: na2s('none') };
        add('Barite in 0.1 mol/kg Na₂SO₄: dissolved barium of the ion-pair model against its closed form', KPAIR * 1e6 + (10 ** BARITE_MODELS.pair.logKsp / (media[2][1].sol.eq.m[mi('SO4')] * media[2][1].sol.eq.gammaOf('Ba') * media[2][1].sol.eq.gammaOf('SO4'))) * 1e9, n2.pair, 1e-4, `nmol/kg: K·Ksp + Ksp/(γ(Ba)·γ(SO₄)·m(SO₄)). The three treatments give ${fmt(n2.pair, 3)} (ion pair), ${fmt(n2.analogue, 3)} (Ca–SO₄ analogue) and ${fmt(n2.none, 3)} (no term) nmol/kg — the kind of medium in which they differ most. Felmy et al. (1990) measured in such solutions but, according to García (2005), reported the data only as plots; the counted neutral-medium points go to 7.9 (Jiang 1996) and 12.5 mmol/kg sulphate (Savenko et al. 2019) and to 79 mmol/kg in dilute sulphuric acid (Paige 1990) — see the checks on sulphate-bearing media above. The one value obtained for this kind of solution is Lieser’s point at 0.11 mol/kg and 20 °C, 42 nmol/kg (digitised second-hand, contested series, not counted), so 0.1 mol/kg remains an extrapolation for every treatment`);
        const e2 = concentrateSolution(sw0, 2, { co2: 'closed' }).eq, iB = mi('Ba'), prB = precipitateSolution(concentrateSolution(sw0, 4.2, { co2: 'closed' }), ['calcite', 'gypsum', 'barite', 'celestite']), n0 = sw0.n[iB];
        add('Ion-pair model: barium mass balance, free ion plus pair against the total', 1, (e2.m[iB] + e2.m[iP]) / e2.tot[iB], 1e-12, `Twofold seawater concentrate: Ba²⁺ ${fq(e2.m[iB], 4)} + BaSO₄° ${fq(e2.m[iP], 4)} = total ${fq(e2.tot[iB], 4)} mol/kg (${pc(e2.m[iP] / e2.tot[iB])} % paired)`);
        add('Ion-pair model: barium is conserved through equilibrium precipitation', 1, (prB.sol.n[iB] + prB.solids.barite) / n0, 1e-10, `Seawater concentrated 4.2 times with calcite, gypsum, barite and celestite: dissolved (free ${fq(prB.sol.eq.m[iB] * prB.sol.w, 3)} + paired ${fq(prB.sol.eq.m[iP] * prB.sol.w, 3)} mol) + barite ${fq(prB.solids.barite, 3)} mol = initial ${fq(n0, 3)} mol; barite index of the final solution ${fq(saturationIndex(prB.sol.eq, 'barite'), 2)}`);
      }
      { // a published calculation, repeated with its own approximation
        const tot = new Float64Array(NM); for (const k of ['Na', 'K', 'Mg', 'Ca', 'Sr', 'SO4']) tot[mi(k)] = SW_CLB[k];
        tot[mi('Cl')] = SW_CLB.Na + SW_CLB.K + 2 * (SW_CLB.Mg + SW_CLB.Ca + SW_CLB.Sr - SW_CLB.SO4);
        const swb = (B) => solubility(equilibrate({ T: 25, model: 'pitzer', n: tot, alk: 0, w: 1, pH: 7 }, { bariteModel: B }), 'barite', { excess: 1e-3 }), bs = swb('analogue'), bp = swb('pair'), gB = Math.sqrt(bs.sol.eq.gammaOf('Ba') * bs.sol.eq.gammaOf('SO4'));
        add('Barite in seawater of salinity 35, Ca–SO₄ analogue treatment: barium at saturation against the calculation of Rogers (1981)', SW_CLB.calc.Ba * 1e9, bs.m * 1e9, 16, `nmol/kg. Rogers (PhD thesis, LBL-12356, p. 235) obtained 209 with the CaSO₄ parameters for BaSO₄ and K = 1.10·10⁻¹⁰ — the approximation of the analogue treatment (log K −9.97, i.e. 1.07·10⁻¹⁰), which is therefore the like-for-like comparison. A published calculation, not a measurement: Rogers quotes 350 nmol/kg measured at 20 °C by Burton, Marshall & Phillips and attributes the difference to particle size. Measured since at 22 °C: 216–221 nmol/kg (Savenko et al. 2019; see the seawater check above). The ion-pair model with the constants of Felmy et al. gives ${fmt(bp.m * 1e9, 4)} nmol/kg (${pc(bp.m / SW_CLB.calc.Ba - 1)} % above Rogers), ${pc(bp.sol.eq.m[si('BaSO4°')] / bp.sol.eq.tot[mi('Ba')])} % of it as BaSO₄(aq)`);
        add('Barite in seawater, Ca–SO₄ analogue treatment: mean activity coefficient of BaSO₄ against Rogers (1981)', SW_CLB.calc.g, gB, 0.007, `Rogers: 0.134 (16 % above Hanor, 14 % below Whitfield). Small differences come from the later Harvie–Møller–Weare parameters for the major ions. Ion-pair model with the constants of Felmy et al.: free-ion coefficient ${fmt(Math.sqrt(bp.sol.eq.gammaOf('Ba') * bp.sol.eq.gammaOf('SO4')), 3)}, stoichiometric (referred to total barium) ${fmt(Math.sqrt((bp.sol.eq.gammaOf('Ba') * bp.sol.eq.m[mi('Ba')] / bp.sol.eq.tot[mi('Ba')]) * bp.sol.eq.gammaOf('SO4')), 3)}`);
        { // Ba–SO4 interaction: spread of the barite index over the published treatments (bariteBand)
          const swS = makeSolution({ ions: WATERS.seawater.ions, T: 25, pH: 8.1 }), sw2 = concentrateSolution(swS, 2, { co2: 'closed' }), b2 = bariteBand(sw2.eq), s2 = saturationIndex(sw2.eq, 'barite'), Pn = PZ_BA.none, keep = [Pn.B0[KBASO4], Pn.B1[KBASO4], Pn.B2[KBASO4]];
          const direct = BASO4_TREATMENTS.filter((q) => q.model).map((q) => saturationIndex(equilibrate(sw2, { bariteModel: q.model }).eq, 'barite') - s2 - b2.shift[q.id]);
          try { const q = BASO4_TREATMENTS.find((r) => r.id === 'sr'); [Pn.B0[KBASO4], Pn.B1[KBASO4], Pn.B2[KBASO4]] = q.b; direct.push(saturationIndex(equilibrate(sw2, { bariteModel: 'none' }).eq, 'barite') - s2 - b2.shift.sr); } finally { [Pn.B0[KBASO4], Pn.B1[KBASO4], Pn.B2[KBASO4]] = keep; }
          add('Barite index under the other Ba–SO₄ treatments: closed-form shift against a recalculation of the speciation', 0, Math.max(...direct.map(Math.abs)), 2e-5, `Seawater concentrated twofold (free SO₄²⁻ ${fmt(b2.mSO4, 3)} mol/kg, I ${fmt(b2.I, 3)}), default treatment ${BA_SHORT[b2.ref]}: the speciation repeated with the ion pair, the Ca–SO₄ analogue, no Ba–SO₄ term and the Sr–SO₄ analogue (parameters exchanged in the table); barium is a trace ion, so the shift is 2·m(SO₄)·ΔB(I)/ln 10 for a binary and −log₁₀(1 + K·γ(Ba)·γ(SO₄)·m(SO₄)) + Δlog Ksp for the pair`);
          add('Barite index in a twofold seawater concentrate: spread over the ion-interaction treatments of Ba–SO₄', 0, b2.ion, 0.045, `SI units, relative to the default (${BA_SHORT[b2.ref]}, SI ${fmt(s2, 3)}): Ca–SO₄ analogue (Rogers 1981) ${fmt(b2.shift.ca, 2)}, Sr–SO₄ analogue ${fmt(b2.shift.sr, 2)}, no Ba–SO₄ term (PHREEQC pitzer.dat, data0.ypf) ${fmt(b2.shift.zero, 2)}; all three with log Ksp −9.97`);
          const all = [...Object.entries(WATERS).map(([nm, w]) => [w.name || nm, makeSolution({ ions: w.ions, T: 25, pH: w.pH ?? 7.5 })]), ['seawater ×2', sw2]].map(([nm, s]) => [nm, bariteBand(s.eq)]).filter((r) => r[1]);
          const lo = all.reduce((p, r) => (r[1].lo < p[1].lo ? r : p)), hi = all.reduce((p, r) => (r[1].hi > p[1].hi ? r : p));
          add('Barite index: lowest value under any published Ba–SO₄ treatment, example waters and twofold seawater concentrate', 0, -lo[1].lo, -BARITE_BAND.lo, `SI below the value of the default treatment (${BA_SHORT[b2.ref]}); largest for ${lo[0]} (${fmt(lo[1].lo, 3)}). Includes the range log K 2.72 ± 0.09 of the pair; in sulphate-poor waters the other treatments lie lower by the difference of the solubility products (−9.97 against −10.05). Twofold seawater concentrate: ${fmt(b2.lo, 3)} … +${fmt(b2.hi, 3)}, ${fmt(100 * b2.paired, 2)} % of the barium paired. ${all.map(([nm, b]) => `${nm} ${fmt(b.lo, 2)}/+${fmt(b.hi, 2)}`).join('; ')}`);
          add('Barite index: highest value under any published Ba–SO₄ treatment, same waters', 0, hi[1].hi, BARITE_BAND.hi, `SI above the value of the default treatment; largest for ${hi[0]} (+${fmt(hi[1].hi, 3)}). In waters with sulphate as a major anion the ion pair gives the lowest barite index of all treatments and the default lies near the top of the range; the verdicts use the highest value in any case`);
          const dil = [0.001, 0.003, 0.01].map((ms) => { const ions = { Na: 2 * ms * IONS.Na.mw * 1000, SO4: ms * IONS.SO4.mw * 1000, Ba: 0.001 }, p = makeSolution({ ions, T: 25, pH: 7, bariteModel: 'analogue' }), d = makeSolution({ ions, T: 25, pH: 7, model: 'davies' }), x = makeSolution({ ions, T: 25, pH: 7, bariteModel: 'pair' }); return [ms, saturationIndex(d.eq, 'barite') - saturationIndex(p.eq, 'barite'), bariteBand(p.eq), saturationIndex(x.eq, 'barite') - saturationIndex(p.eq, 'barite')]; });
          add('Barite index with a BaSO₄(aq) ion pair: Pitzer path against the speciation of the Davies ion-pair model', 0, Math.max(...dil.map(([, d, b]) => Math.abs(d - b.shift.pairSameK))), 0.035, `Na₂SO₄ solutions of 0.001, 0.003 and 0.01 mol/kg, same log Ksp, relative to the Ca–SO₄ analogue: Davies model with BaSO₄° (log K 2.7, WATEQ4F) ${dil.map(([, d]) => fmt(d, 3)).join(', ')}; Pitzer path with log K 2.72 ${dil.map(([, , b]) => fmt(b.shift.pairSameK, 3)).join(', ')} (closed form) and, with its own log Ksp, ${dil.map(([, , , x]) => fmt(x, 3)).join(', ')} (speciation). In these dilute sulphate waters the treatments differ by up to ${fmt(Math.max(...dil.map(([, , b]) => b.width)), 2)} SI — the case flagged in the results`);
          add('Barite index in dilute Na₂SO₄ solutions: speciation of the ion-pair treatment against the closed form', 0, Math.max(...dil.map(([, , b, x]) => Math.abs(x - b.shift.pair))), 2e-5, 'Same three solutions: the index computed with the BaSO₄(aq) species in the Pitzer speciation minus the closed-form shift of bariteBand()');
          add('BaSO₄(aq) association constant of the ion-pair models against Felmy, Rai & Amonette (1990)', 2.72, DER.find((d) => d[0] === 'BaSO4°')[7](25), 0.09, 'log K at 25 °C: 2.7 in WATEQ4F (in use with the Debye–Hückel family) against 2.72 ± 0.09 from barite solubilities in Na₂SO₄ solutions (J. Solution Chem. 19, 175, abstract — primary). The Pitzer path uses 2.72 itself');
          add('Barite log Ksp of the ion-pair treatment against Felmy, Rai & Amonette (1990)', -10.05, logKfor(MINERALS.barite, 25, 'pitzer', 'pair'), 1e-9, `Felmy et al.: −10.05 ± 0.05 from solubilities in Na₂SO₄ solutions, evaluated together with the ion pair (log K 2.72 ± 0.09) — the two constants are used as a pair. The analogue and no-term treatments keep ${fmt(logKfor(MINERALS.barite, 25, 'pitzer', 'analogue'), 4)} (WATEQ4F; Blount 1977: −9.98; Templeton 1960: −9.96), which reproduces the solubility in water; the published compilation of Zhen-Wu et al. (2014) lists −9.96 to −10.05 for six studies`);
        }
      }
      const cel = (o) => { const n = new Float64Array(NM); for (const [k, x] of Object.entries(o)) n[mi(k)] = x; return solubility(equilibrate({ T: 25, model: 'pitzer', n, alk: 0, w: 1, pH: 7 }), 'celestite', { excess: 0.2 }).m * 1000; };
      const ra = CELESTITE_RA.map(([c, x]) => [c, cel({ Na: c, Cl: c }) / x - 1]), cw = (ra[0][1] + 1) * CELESTITE_RA[0][1];
      add('Celestite solubility in water, 25 °C', 0.6435, cw, 0.035, 'mmol/kg. Reardon & Armstrong (1987): 0.643; Culberson, Latham & Bates (1978): 0.644 (both as tabulated by Dal Pozzo 1991; Rogers 1981 quotes the latter as 0.644 ± 0.001); Brower & Renault (1971): 0.66 mmol/L. Tests log K = −6.63 with the Sr–SO₄ parameters of the THEREDA set; with the THEREDA log K −6.55 the model gives 0.684');
      add('Celestite log Ksp in use against Felmy, Rai & Amonette (1990)', -6.62, logKfor(MINERALS.celestite, 25, 'pitzer'), 0.02, 'Felmy et al.: −6.62 ± 0.02 from celestite solubilities in Na₂SO₄ solutions, ion-interaction treatment (J. Solution Chem. 19, 175, abstract — primary). In use: −6.63 (WATEQ4F, pitzer.dat)');
      add('Celestite solubility in 0.05–5 mol/kg NaCl against Reardon & Armstrong (1987)', 0, Math.max(...ra.slice(1).map((r) => Math.abs(r[1]))), 0.06, `Largest relative deviation over 11 solutions (data as tabulated in Table C.1 of Dal Pozzo 1991 — secondary): ${ra.slice(1).map((r) => `${r[0]}: ${pc(r[1])} %`).join(', ')}. With log K −6.55 the deviations were +10 to +15 %`);
      add('Celestite solubility in 0–5 mol/kg NaCl: root-mean-square deviation', 0, rms(ra.map((r) => r[1])), 0.02, 'In log₁₀ units over the 12 points of Reardon & Armstrong (1987); 0.049 with log K −6.55');
      add('Celestite solubility in 0.7 mol/kg NaCl (Culberson, Latham & Bates 1978)', CELESTITE_CLB.nacl[1], cel({ Na: 0.7, Cl: 0.7 }), 0.13, 'mmol/kg; measured 3.231 (as tabulated by Dal Pozzo 1991 — secondary)');
      const sw = SW_CLB.cel.map(([mg, ca, x]) => [mg, ca, cel({ Na: SW_CLB.Na, K: SW_CLB.K, Mg: mg, Ca: ca, SO4: SW_CLB.SO4, Cl: SW_CLB.Na + SW_CLB.K + 2 * (mg + ca - SW_CLB.SO4) }) / x - 1]);
      add('Celestite solubility in seawater of salinity 35 (Culberson, Latham & Bates 1978)', 0, Math.max(...sw.map((r) => Math.abs(r[2]))), 0.06, `Largest relative deviation over four synthetic seawaters with Mg from 0.021 to 0.066 and Ca from 0 to 0.045 mol/kg (measured 0.414–0.423 mmol/kg Sr; Rogers 1981, Tables 6 and 8 — secondary): ${sw.map((r) => `${pc(r[2])} %`).join(', ')}. Tests θ(Sr,Na), θ(Sr,Mg), θ(Sr,Ca) and Sr–SO₄ together. With log K −6.55 the model was 22–25 % high; the seawater data of Reardon & Armstrong (1987) could not be retrieved`);
      const ce = CELESTITE_NACL.map(([c, x]) => { const rho = density(25, (c * 58.443) / (1 + c * 0.058443)) / 1000, kgw = rho - c * 0.058443; return [c, (cel({ Na: c / kgw, Cl: c / kgw }) * kgw) / x - 1]; });
      add('Celestite solubility in 0–1 mol/L NaCl against Brower & Renault (1971): root-mean-square deviation', 0, rms(ce.map((r) => r[1])), 0.09, `In log₁₀ units over six concentrations (primary, two significant figures): ${ce.map((r) => `${r[0]}: ${pc(r[1])} %`).join(', ')}. This series lies well above Reardon & Armstrong (1.3 mmol/L at 0.025 mol/L against 1.14 mmol/kg at 0.05 mol/kg; 4.5 at 1 mol/L against 4.62 at 1.97 mol/kg), which is why it favoured log K −6.55 (rms 0.043) before the molal data were retrieved`);
    }
    { // Bromley constants: published table against an independent refit from the NIST evaluations
      const Zc = { Mg: 2, Ca: 2, Sr: 2, Ba: 2 }, rf = BR_REFIT.split('|').map((r) => r.split(' ')).map(([c, a, B, rm, n, Im, pub]) => ({ c, a, B: +B, rms: +rm, n: +n, Imax: +Im, pub: pub == null ? null : +pub }));
      let dFit = 0, dPub = 0, worst = '';
      for (const [k, txt] of Object.entries(BR_DATA)) {
        const [c, a] = k.split(' '), v = txt.split(' ').map(Number), rows = []; for (let i = 0; i < v.length; i += 2) rows.push([v[i], v[i + 1]]);
        const f = fitBromleyB(Zc[c] || 1, a === 'SO4' ? 2 : 1, rows), st = rf.find((r) => r.c === c && r.a === a), used = PAIRPAR.BR[si(c) * NS + si(a)];
        dFit = Math.max(dFit, Math.abs(f.B - st.B));
        if (Math.abs(used - f.B) > dPub) { dPub = Math.abs(used - f.B); worst = `${c}–${a}: ${fmt(used, 4)} published, ${fmt(f.B, 4)} refitted`; }
      }
      add('Bromley refit: the regression on the embedded NIST tables returns the stored constants', 0, dFit, 6e-5, `fitBromleyB on γ± of ${Object.keys(BR_DATA).length} salts (Hamer & Wu 1972; Goldberg & Nuttall 1978; Goldberg 1981), weighted least squares on log γ± to I = 6 mol/kg`);
      add('Bromley constants in use (Bromley 1973) against the refit from NIST data', 0, dPub, 0.011, `Largest difference of B over the 16 salts of the suite with tabulated data — ${worst}; K₂SO₄ is tabulated only to 0.69 mol/kg`);
      const cmp = rf.filter((r) => r.pub != null), dAll = cmp.map((r) => r.B - r.pub), rbi = rf.find((r) => r.c === 'Rb' && r.a === 'I');
      add('Bromley published salt constants against the refit: root-mean-square difference over all salts', 0, Math.sqrt(sum(dAll.map((x) => x * x)) / dAll.length), 0.004, `${cmp.length} salts with a value in Bromley’s Table 1 as reproduced in Zemaitis et al. (1986, Appendix 4.2), RbI with the corrected sign; largest ${fmt(Math.max(...dAll.map(Math.abs)), 3)} kg/mol`);
      add('Bromley constant of RbI: value held (sign corrected) against the first refit, on 12 of the 27 rows of the NIST table', rbi.pub, rbi.B, 0.002, `kg/mol; superseded by the fit to the complete table further down, which returns +0.0108. The handbook reprint (Zemaitis et al. 1986, Appendix 4.2) prints −0.0108 (σ 0.005), which is inconsistent with the refit to the NIST data (Hamer & Wu 1972: +${fmt(rbi.B, 3)}) and with the sum of Bromley’s own individual-ion values (+${fmt(BR_PUB.Rb[0] + BR_PUB.I[0] + BR_PUB.Rb[1] * BR_PUB.I[1], 3)}), so the sign is taken as a misprint in the reprint and the positive value is used. With the printed sign the difference from the refit would be ${fmt(Math.abs(-rbi.pub - rbi.B), 3)} kg/mol, against at most ${fmt(Math.max(...cmp.filter((r) => r !== rbi).map((r) => Math.abs(r.B - r.pub))), 3)} kg/mol for any other of the ${cmp.length - 1} salts`);
      let salts = rf.map((r) => [r.c, r.a, r.B]);
      for (;;) { const cnt = {}; for (const q of salts) { cnt['c' + q[0]] = (cnt['c' + q[0]] || 0) + 1; cnt['a' + q[1]] = (cnt['a' + q[1]] || 0) + 1; } const keep = salts.filter((q) => cnt['c' + q[0]] >= 3 && cnt['a' + q[1]] >= 3); if (keep.length === salts.length) break; salts = keep; }
      const ion = fitBromleyIons(salts), dIon = Math.max(...Object.entries(BR_ION_REFIT.c).map(([k, [B, d]]) => Math.max(Math.abs(ion.Bc[k] - B), Math.abs(ion.dc[k] - d))), ...Object.entries(BR_ION_REFIT.a).map(([k, [B, d]]) => Math.max(Math.abs(ion.Ba[k] - B), Math.abs(ion.da[k] - d))));
      add('Bromley ion values: the global regression returns the stored refit', 0, dIon, 6e-4, `B = B₊ + B₋ + δ₊δ₋ over ${ion.n} salts, ${ion.nPar} free ion values (anchors B(Na⁺) = 0, δ(OH⁻) = −1, δ(Na⁺) = 0.028, δ(Cl⁻) = −0.067), ${ion.iterations} alternating least-squares sweeps`);
      { // a second reproduction of Bromley's tables
        const th = BR_THOMSEN.salt.split('|').map((r) => r.split(' ')), own = Object.fromEntries(BR_SALT.split('|').map((r) => { const [c, a, b] = r.split(' '); return [c + ' ' + a, +b]; })), pubOf = (c, a) => own[c + ' ' + a] ?? rf.find((r) => r.c === c && r.a === a)?.pub ?? BR_THOMSEN.reprint[c + ' ' + a];
        const dS = th.map(([c, a, b]) => Math.abs(+b - pubOf(c, a))), inSuite = th.filter(([c, a]) => own[c + ' ' + a] != null), ions = Object.entries(BR_THOMSEN.ion).filter(([id]) => BR_ION[id]), dI = ions.map(([id, [B, d]]) => Math.max(Math.abs(B - BR_ION[id][0]), Math.abs(d - BR_ION[id][1])));
        add('Bromley salt constants: second reproduction (Thomsen 2009, Table 6.2) against the values in use and the handbook reprint', 0, Math.max(...dS), 5e-5, `kg/mol; ${th.length} salt constants printed in the lecture notes of Thomsen (Technical University of Denmark, open; p. 49), all identical to the handbook reprint (Zemaitis et al. 1986). ${inSuite.length} of them are salt constants of this suite (${inSuite.map(([c, a]) => c + '–' + a).join(', ')}); HNO₃ and KBr enter the refit comparison; NaBr, H₂SO₄, K₂CO₃, Na₂CO₃, Mg(NO₃)₂ and Al₂(SO₄)₃ are not held as salt constants here. Not covered by the second reproduction: Mg–Cl, Sr–Cl, Ba–Cl, Na–OH, K–OH, NH₄–NO₃ (handbook reprint and refit only) and RbI`);
        add('Bromley individual-ion values: second reproduction (Thomsen 2009, Table 6.3) against the values in use', 0, Math.max(...dI), 5e-5, `B and δ of ${ions.map(([id]) => id).join(', ')} (p. 50 of the same notes): identical. The notes also print Al³⁺ (0.052, 0.12), which is not an ion of this suite. Mg, Sr, Ba, Mn, Fe, F, OH and HPO₄ rest on the handbook reprint alone`);
        const est = (c, a) => BR_ION[c][0] + BR_ION[a][0] + BR_ION[c][1] * BR_ION[a][1], carb = [['Na', 'CO3'], ['K', 'CO3']].map(([c, a]) => est(c, a) - BR_THOMSEN.reprint[c + ' ' + a]);
        add('Bromley model, carbonates: ion-table estimate against Bromley’s fitted salt constants of Na₂CO₃ and K₂CO₃', 0, Math.max(...carb.map(Math.abs)), 0.002, `kg/mol: Na₂CO₃ ${fmt(est('Na', 'CO3'), 3)} against 0.0089, K₂CO₃ ${fmt(est('K', 'CO3'), 3)} against 0.0372 (all three reproductions): Bromley’s ion table agrees with his fitted constants within 0.0003 and 0.0015 kg/mol. Neither is in use: both salts carry constants re-derived from independent data (BR_SWITCH; checks below)`);
        add('Bromley model, Mg(NO₃)₂: ion-table estimate against Bromley’s fitted salt constant (the fitted constant is in use)', BR_THOMSEN.reprint['Mg NO3'], est('Mg', 'NO3'), 0.03, `kg/mol: ${fmt(est('Mg', 'NO3'), 3)} from B₊ + B₋ + δ₊δ₋ against the fitted 0.1014 (σ 0.004 in the handbook reprint; both reproductions). A known shortfall of the ion table for this salt, which is why the suite uses the fitted constant for it; the default Pitzer model is not affected`);
      }
      add('Bromley ion values: how well the refitted ion table reproduces the salt constants', 0.0081, ion.rms, 0.0004, `Root-mean-square residual in B, kg/mol (largest ${fmt(ion.max, 3)}). Bromley’s published ion table gives ${fmt(Math.sqrt(sum(salts.map(([c, a, B]) => ((BR_ION[c] || BR_PUB[c])[0] + (BR_ION[a] || BR_PUB[a])[0] + (BR_ION[c] || BR_PUB[c])[1] * (BR_ION[a] || BR_PUB[a])[1] - B) ** 2)) / salts.length), 4)} on the same salts: the additive ion scheme, not the data, limits the accuracy of salts without a fitted constant`);
      const pubIon = Math.max(...['H', 'Na', 'K', 'NH4', 'Mg', 'Ca', 'Sr', 'Ba'].map((k) => Math.abs(BR_ION[k][0] - BR_ION_REFIT.c[k][0])), ...['Cl', 'NO3', 'OH', 'F', 'SO4'].map((k) => Math.abs(BR_ION[k][0] - BR_ION_REFIT.a[k][0])));
      add('Bromley ion values in use against the refit: B₊ and B₋ of the suite’s ions', 0, pubIon, 0.017, 'Largest difference in kg/mol over H⁺, Na⁺, K⁺, NH₄⁺, Mg²⁺, Ca²⁺, Sr²⁺, Ba²⁺, Cl⁻, NO₃⁻, OH⁻, F⁻, SO₄²⁻ (the δ values are less well determined and differ by up to 0.09 for cations)');
      { // a third reproduction (open-source code) and the refits of the salts that BR_REFIT does not hold
        const own = BR_SALT.split('|').map((r) => r.split(' ')), r3 = (c, a) => +BR_REPRO3.rows[c].split(' ')[BR_REPRO3.an.indexOf(a)], lib = 'matrix BromleyData of the Modelica library ElectrolyteMedia (Bremen & Mitsos, RWTH Aachen; file MixtureSolutesData/package.mo, commit eaaf9ad, BSD 3-Clause; “based on Bromley, 1973”)';
        add('Bromley salt constants: third reproduction (open-source library) against the table values held', 0, Math.max(...own.map(([c, a, b]) => Math.abs(r3(c, a) - b))), 5e-5, `kg/mol; the ${own.length} table values of this suite’s salts (BR_SALT) in the ${lib}: identical. This reproduction covers the constants that the lecture notes do not print (Mg–Cl, Sr–Cl, Ba–Cl, Na–OH, K–OH, NH₄–NO₃)`);
        add('Bromley constant of RbI: third reproduction against the value held', rbi.pub, BR_REPRO3.RbI, 5e-5, 'kg/mol; the same matrix holds +0.0108 for Rb⁺–I⁻ (its neighbours RbBr 0.0111 and RbCl 0.0157 as in the handbook reprint), where the handbook reprint prints −0.0108. Of the other 56 Table 1 values of the refit list, compared by script, the matrix differs from the handbook reprint in two: LiOH (0.0071, the ion-table sum, against −0.0097) and MgBr₂ (0.1416 against 0.1419); its Cs⁺ entries for anions without a fitted salt are 0.142 too high (B of Cs⁺ entered with the wrong sign) — it is a transcription with slips of its own, none of them in a salt or ion of this suite');
        const est = (c, a) => BR_ION[c][0] + BR_ION[a][0] + BR_ION[c][1] * BR_ION[a][1], exact = Object.keys(BR_REPRO3.rows).flatMap((c) => BR_REPRO3.an.filter((a) => Math.abs(r3(c, a) - est(c, a)) < 5e-5).map((a) => [c, a])), solo = ['Mg', 'Sr', 'Ba', 'Mn', 'Fe', 'F', 'OH', 'HPO4'], cnt = (k) => exact.filter((q) => q[0] === k || q[1] === k).length;
        add('Bromley individual-ion values: third reproduction, entries that equal B₊ + B₋ + δ₊δ₋ of the table in use', 38, exact.length, 0, `Of the ${Object.keys(BR_REPRO3.rows).length * BR_REPRO3.an.length} entries for the cations H, Na, K, NH₄, Mg, Ca, Sr, Ba, Mn, Fe against F, Cl, NO₃, OH, SO₄, CO₃, HPO₄, those without a fitted salt constant in Bromley’s Table 1 are ion-table sums; each one reproduces the sum of the (B, δ) in use to 5·10⁻⁵ kg/mol. The eight ions that rested on the handbook reprint alone take part in ${solo.map((k) => `${k} ${cnt(k)}`).join(', ')} of these sums, each with several counter-ions whose values are confirmed by the lecture notes, which fixes both B and δ. Cl⁻ has a fitted constant with every one of these cations; BeCl₂ ${BR_REPRO3.BeCl} = ${fmt(BR_REPRO3.Be[0] + BR_ION.Cl[0] + BR_REPRO3.Be[1] * BR_ION.Cl[1], 4)} from the reprint’s Be²⁺ (0.1, 0.2) confirms its pair`);
        add('Bromley individual-ion values: fewest third-reproduction sums for an ion that rested on the handbook reprint alone', 4, Math.min(...solo.map(cnt)), 0, 'Each of Mg, Sr, Ba, Mn, Fe, F, OH, HPO₄ is determined by at least this many independent sums');
        const rf2 = BR_REFIT2.split('|').map((r) => r.split(' ')).map(([c, a, B, rm, n, Im, pub]) => ({ c, a, B: +B, rms: +rm, n: +n, Imax: +Im, pub: +pub })), pz = Object.fromEntries(PZ_BIN.split('|').map((r) => r.split(' ')).map(([c, a, ...v]) => [c + ' ' + a, v.map(Number)])), pair = (t) => { const v = t.split(' ').map(Number), rows = []; for (let i = 0; i < v.length; i += 2) rows.push([v[i], v[i + 1]]); return rows; };
        const rows2 = { 'Rb I': pair(BR_DATA2['Rb I']), 'Na CO3': pair(BR_DATA2['Na CO3']), 'Mg NO3': pair(BR_DATA2['Mg NO3']), 'Mg SO4': MGSO4_REF.ar.map(([I, g]) => [I / 4, g]), 'K CO3': pair(BR_DATA2['Na CO3']).map(([m]) => [m, pitzerSingle(1, 2, m, ...pz['K CO3']).gamma]) };
        const brG = (zc, za, m, B) => { const nc = za === zc ? 1 : za, na = za === zc ? 1 : zc, I = 0.5 * m * (nc * zc * zc + na * za * za), zz = zc * za, sq = Math.sqrt(I); return 10 ** ((-(3 * aphi(25)) / LN10) * zz * sq / (1 + sq) + ((0.06 + 0.6 * B) * zz * I) / (1 + (1.5 * I) / zz) ** 2 + B * I); };
        const zOf = (c, a) => [Zc[c] || 1, a === 'SO4' || a === 'CO3' ? 2 : 1], fit2 = Object.fromEntries(rf2.map((r) => [r.c + ' ' + r.a, fitBromleyB(...zOf(r.c, r.a), rows2[r.c + ' ' + r.a])])), g2 = (k) => rf2.find((r) => r.c + ' ' + r.a === k);
        add('Bromley refit of RbI, Na₂CO₃, Mg(NO₃)₂, MgSO₄ and K₂CO₃: the regression returns the stored constants', 0, Math.max(...rf2.map((r) => Math.max(Math.abs(fit2[r.c + ' ' + r.a].B - r.B), Math.abs(fit2[r.c + ' ' + r.a].rms - r.rms)))), 6e-5, `B and rms in log γ±: ${rf2.map((r) => `${r.c}–${r.a} ${r.B} (${r.rms}; ${r.n} points to I = ${r.Imax})`).join(', ')}`);
        { const k = 'Rb I', f = fit2[k], neg = fitBromleyB(1, 1, rows2[k], { B: -rbi.pub });
          add('Bromley constant of RbI: refit to the complete table of Hamer & Wu (1972) against the value in use', rbi.pub, f.B, 0.0005, `kg/mol. All ${f.n} rows of Table 42 (0.001–5 mol/kg, read from the page image) by Bromley’s method: B = +${fmt(f.B, 4)} with a standard deviation of ${fmt(f.rms, 2)} in log γ± — the magnitude and the σ (0.005) that the handbook reprint prints for RbI. With the printed sign, −${rbi.pub}, the same table is missed by ${fmt(neg.rms, 2)} in log γ± (largest ${fmt(neg.max, 2)}, i.e. ${fmt(100 * (10 ** neg.max - 1), 2)} % in γ± at 5 mol/kg), ${fmt(neg.rms / f.rms, 2)} times the fitted scatter. The sign is therefore settled by the data: +0.0108, as the third reproduction also has it. (The ${rbi.n}-row fit of the first pass gave +${rbi.B}.)`);
          add('Bromley constant of RbI with the sign printed in the handbook reprint: rms deviation from the Hamer & Wu table relative to the fitted one', 12.7, neg.rms / f.rms, 0.5, 'Ratio of the two standard deviations in log γ±; a value near 1 would leave the sign open'); }
        add('Bromley constant of Mg(NO₃)₂ in use against a refit to the evaluation of Rard, Wijesinghe & Wolery (2004)', g2('Mg NO3').pub, fit2['Mg NO3'].B, 0.004, `kg/mol; smoothed γ± of their Table 3 (Lawrence Livermore report UCRL-JRNL-203290, open report version of J. Chem. Eng. Data 49, 1127; read from the page image), ${fit2['Mg NO3'].n} points to 2 mol/kg: rms ${fmt(fit2['Mg NO3'].rms, 2)} in log γ±, against σ 0.004 in the handbook reprint`);
        add('Bromley constant of MgSO₄ in use against a refit to the Archer & Rard (1998) model values', g2('Mg SO4').pub, fit2['Mg SO4'].B, 0.01, `kg/mol; γ± at I = 1–6 mol/kg as tabulated by Miladinović et al. (2007) — an evaluated model table, the article itself not read: rms ${fmt(fit2['Mg SO4'].rms, 2)} in log γ± (σ 0.051 in the handbook reprint). The one-constant equation does not describe a 2:2 salt; B is determined to about ±0.01 only`);
        add('Bromley’s constant of K₂CO₃ against a refit to γ± of the Harvie–Møller–Weare parameters (the refit is in use)', g2('K CO3').pub, fit2['K CO3'].B, 0.015, `kg/mol; secondary: no table of primary values was obtained (Goldberg 1981 gives none for K₂CO₃), so γ± at 0.01–2 mol/kg comes from the single-salt Pitzer equation with β⁰ ${pz['K CO3'][0]}, β¹ ${pz['K CO3'][1]}, Cφ ${pz['K CO3'][3]} of this suite. Tolerance = σ 0.015 of the handbook reprint; in log γ± the published constant misses these values by more than the criterion below, so the refit is used`);
        { // every constant in use against its independent re-derivation
          const data = { ...Object.fromEntries(Object.entries(BR_DATA).map(([k, t]) => [k, pair(t)])), ...rows2, 'NH4 NO3': pair(BR_DATA2['NH4 NO3']) }, src = (k) => (k === 'Mg SO4' ? 'model table' : k === 'K CO3' ? 'Pitzer parametrisation' : k === 'Mg NO3' ? 'Livermore evaluation' : 'NIST');
          const all = own.map(([c, a, b]) => { const k = c + ' ' + a, z = zOf(c, a), f = fitBromleyB(...z, data[k]), t = fitBromleyB(...z, data[k], { B: +b }), u = fitBromleyB(...z, data[k], { B: PAIRPAR.BR[si(c) * NS + si(a)] }), lim = 2 * Math.max(f.rms, 0.005); return { k, c, a, pub: +b, used: u.B, B: f.B, rms: f.rms, n: f.n, Imax: f.Imax, rT: t.rms / lim, rU: u.rms / lim, rmsT: t.rms }; });
          const out = all.filter((r) => r.rT > 1), sw = Object.keys(BR_SWITCH), row = (r) => `${r.c}–${r.a} ${r.pub} / ${fmt(r.B, 4)} / ${fmt(Math.abs(r.pub - r.B), 2)} / ${fmt(r.rms, 2)} / ${fmt(r.rmsT, 2)}`;
          add('Bromley salt constants: stored refits of the 19 salts against the regression on the embedded tables', 0, Math.max(...all.map((r) => { const st = rf.find((q) => q.c === r.c && q.a === r.a) || g2(r.k); return Math.abs(st.B - r.B); })), 9e-4, 'kg/mol; NH₄NO₃ is refitted here on the complete table (28 rows) and stored from 24 rows, hence the tolerance');
          add('Bromley salt constants of Bromley’s table: number that reproduce the independent γ± data within the scatter of the refit', all.length - sw.length, all.length - out.length, 0, `Criterion: rms deviation in log γ± with the table constant ≤ 2 × max(rms of the refit, 0.005). Per salt — table B / re-derived B / |difference| / rms of the refit / rms with the table constant: ${all.map(row).join('; ')}. Data: ${all.map((r) => src(r.k)).filter((x, i, v) => v.indexOf(x) === i).map((x) => `${x} (${all.filter((r) => src(r.k) === x).length})`).join(', ')} — NIST: Hamer & Wu 1972, Goldberg & Nuttall 1978, Goldberg 1981. Outside the criterion: ${out.map((r) => `${r.c}–${r.a} (${fmt(r.rT * 2, 2)} × the scatter)`).join(', ')}; closest inside: ${all.filter((r) => r.rT <= 1).sort((p, q) => q.rT - p.rT).slice(0, 2).map((r) => `${r.c}–${r.a} (${fmt(r.rT * 2, 2)} ×)`).join(', ')}`);
          add('Bromley salt constants replaced by their re-derived values are exactly those outside the criterion', 0, sw.filter((k) => !out.some((r) => r.k === k)).length + out.filter((r) => !sw.includes(r.k)).length, 0, `Number of mismatches between the two lists. Old → new: ${out.map((r) => `${r.c}–${r.a} ${r.pub} → ${BR_SWITCH[r.k]}`).join(', ')} kg/mol. SrCl₂ and BaCl₂: Goldberg & Nuttall (1978) against the tables available to Bromley in 1973; K₂CO₃: re-derived from the Harvie–Møller–Weare parametrisation only (no table of primary values obtained), which makes the Bromley option consistent with the default model for this salt; Na₂CO₃: Goldberg (1981) names it as the salt where his evaluation departs most from the earlier ones, because of the isopiestic results of Robinson & Macaskill (1979) — the published 0.0089 puts γ± ${fmt(100 * (brG(1, 2, 1, 0.0089) / data['Na CO3'].find((r) => r[0] === 1)[1] - 1), 2)} % above the recommended value at 1 mol/kg and ${fmt(100 * (brG(1, 2, 2, 0.0089) / data['Na CO3'].find((r) => r[0] === 2)[1] - 1), 2)} % at 2 mol/kg; the Harvie–Møller–Weare parameters give −0.0062 by the same fit`);
          add('Bromley salt constants in use: largest rms deviation from the independent γ± data relative to the criterion', 0.9, Math.max(...all.map((r) => r.rU)), 0.1, `Every one of the ${all.length} constants in use — ${all.length - sw.length} of Bromley’s table and ${sw.length} re-derived — lies within the criterion (ratio ≤ 1); the largest is ${all.slice().sort((p, q) => q.rU - p.rU)[0].k.replace(' ', '–')}`);
          add('Bromley constants replaced: the values in use equal the refits', 0, Math.max(...sw.map((k) => Math.abs(BR_SWITCH[k] - all.find((r) => r.k === k).B))), 6e-5, 'kg/mol; SrCl₂ and BaCl₂ from BR_REFIT, Na₂CO₃ and K₂CO₃ from BR_REFIT2');
        }
      }
      { // SIT ε(Sr,Cl): fit to the same table up to I = 3 mol/kg
        const v = BR_DATA['Sr Cl'].split(' ').map(Number), A = (3 * aphi(25)) / LN10; let sxx = 0, sxy = 0; const d = [];
        for (let i = 0; i < v.length; i += 2) if (3 * v[i] <= 3 + 1e-9) d.push([v[i], v[i + 1]]);
        d.forEach(([m, g], i) => { const sq = Math.sqrt(3 * m), w = 0.5 * (d[Math.min(i + 1, d.length - 1)][0] - d[Math.max(i - 1, 0)][0]), x = (4 / 3) * m, y = Math.log10(g) + (2 * A * sq) / (1 + 1.5 * sq); sxx += w * x * x; sxy += w * x * y; });
        add('SIT ε(Sr²⁺, Cl⁻): least-squares fit to the SrCl₂ table of Goldberg & Nuttall (1978)', sxy / sxx, PAIRPAR.EPS[si('Sr') * NS + si('Cl')], 0.005, 'kg/mol; log γ± = −2A√I/(1 + 1.5√I) + (4/3)·ε·m up to I = 3 mol/kg. Replaces the Ca²⁺ analogue 0.14');
      }
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
    { // hot spot next to a filament: footprint, osmotic ceiling and grid convergence
      const piS = (c) => osmoticPressure(25, clamp(63.7 * c, 0, 260)), par = { h: 7.1e-4, u0: 0.0883, vw: 15 / 3.6e6, D: 1.48e-9, rej: 0.995, rho: 1045, mu: 1e-3, arr: 'zigzag', lm: 3e-3, df: 3.55e-4, contact: 0.15, dP: 60e5, pi: piS };
      const gs = await spacerHotSpotStudy(par, { nFine: 50, Lp: 1e-4 }), pk = gs.grids.map((g) => g.peak), pa = gs.grids.map((g) => g.patch), F = gs.fine;
      add('Spacer hot spot: osmotic ceiling is where the flux law gives zero flux', 60e5, piS(gs.cCap) - piS(0.005 * gs.cCap), 1, 'π(c cap) − π((1 − R)·c cap) = ΔP = 60 bar for 99.5 % rejection, Pa');
      add('Spacer hot spot: no concentration on any of the three grids exceeds the osmotic ceiling', 1, gs.cMax <= gs.cCap * (1 + 1e-4) && gs.cMax > 1.05 ? 1 : 0, 0, `Largest value in the field and on the walls ${fmt(gs.cMax, 5)} against the ceiling ${fmt(gs.cCap, 5)} (c / c inlet); J = A·(ΔP − Δπ) ≥ 0`);
      add('Spacer hot spot: membrane under the filament footprints does not permeate', 0, Math.max(0, gs.jFoot) * 3.6e6 + (gs.nFoot > 0 ? 0 : 1), 1e-9, `${gs.nFoot} wall cells touched by a footprint of ±${fmt(gs.contactHalfWidth * 1e6, 3)} µm: flux above the share of their open length, L/m²·h`);
      { const fp = F.feet.find((q) => !q.top && q.x > F.L / 2) || F.feet[0], cov = F.x.reduce((a2, x, i) => a2 + (Math.abs(x - fp.x) < fp.a + F.dx ? (1 - (fp.top ? F.permT : F.permB)[i]) * F.dx : 0), 0);
        add('Spacer hot spot: covered length of a footprint equals twice the contact half-width', 2 * fp.a * 1e6, cov * 1e6, 1e-6, 'Partly covered cells permeate in proportion to their open length, so the footprint has the same width on every grid, µm'); }
      add('Spacer hot spot: patch-averaged value changes by less than 3 % between the two finest grids', 0, Math.abs(gs.patch.change21), 0.03, `Mean over the hottest 100 µm of membrane on grids of ${gs.grids.map((g) => `${g.nx}×${g.ny}`).join(', ')} cells: ${pa.map((x) => fmt(x, 5)).join(', ')}; Richardson estimate ${fmt(gs.patch.value, 5)}, grid-convergence index ${fmt(100 * gs.patch.gci, 2)} %, observed order ${fmt(gs.patch.pObs, 3)}`);
      add('Spacer hot spot: peak sequence is monotone and stays below the ceiling', 1, gs.peak.monotone && pk[0] < pk[1] && pk[1] < pk[2] && pk[2] <= gs.cCap && gs.peak.value <= gs.cCap ? 1 : 0, 0, `Peak wall-cell value ${pk.map((x) => fmt(x, 5)).join(', ')}; estimate ${fmt(gs.peak.value, 5)}${gs.peak.capped ? ' (set to the ceiling)' : ''}, grid-convergence index ${fmt(100 * gs.peak.gci, 2)} %`);
      add('Spacer hot spot: the grid-convergence index covers the change to the finest grid', 1, gs.patch.gci * gs.patch.value >= 0.5 * Math.abs(pa[2] - pa[1]) ? 1 : 0, 0, 'Error band of the estimate against the last refinement step');
      const gq = gridConvergence([1, 1.5, 2.25], [1, 1.5, 2.25].map((hh) => 2 + 0.1 * hh * hh));
      add('Grid convergence: Richardson extrapolation recovers a second-order limit', 2, gq.value, 1e-9, `f = 2 + 0.1·h² on three grids; observed order ${fmt(gq.pObs, 4)}`);
      const hp = wallHotSpot({ dx: 1, nx: 10, L: 10, x: Array.from({ length: 10 }, (_, i) => i + 0.5), blockB: new Array(10).fill(false), feet: [{ x: 5, a: 1, top: false }], cwB: [1, 1, 1, 2, 9, 9, 4, 1, 1, 1] }, false, 1.5);
      add('Hot-spot patch average: exact integral of the cell values outside the footprint', (4 * 1 + 1 * 0.5) / 1.5, hp.patch, 1e-12, 'Cells under the footprint (x = 4–6) are ignored; the hottest 1.5-long window on the open membrane starts at its edge');
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
    return { mNaCl, Tc, pCO2x, sGyp: +(m.sGyp * (1 + g.normal(0, 0.012))).toFixed(2), sCal: +(m.sCal * (1 + g.normal(0, 0.015))).toFixed(3), cfBa: null, baEq: null, cfOn: null };
  });
}
/** Synthetic barite jar tests on the default water (an illustration, not measurements): the default model with Δ log Ksp barite = +0.06 and 4 % noise on barium; pts = [concentration factor, °C] of seeded equilibrium tests, onsetT = temperatures of onset observations (3 % noise). */
function synthBa(seed, pts, onsetT = []) {
  const d = D(), g = rng(seed), blank = { mNaCl: null, pCO2x: null, sGyp: null, sCal: null };
  return [...pts.map(([cfBa, Tc]) => ({ ...blank, Tc, cfBa, baEq: +(suite.calibration.model({ ...d, dkBarite: 0.06, cfBa, Tc }).baEq * (1 + g.normal(0, 0.04))).toFixed(1), cfOn: null })),
    ...onsetT.map((Tc) => ({ ...blank, Tc, cfBa: null, baEq: null, cfOn: +(suite.calibration.model({ ...d, dkBarite: 0.06, Tc }).cfOn * (1 + g.normal(0, 0.03))).toFixed(3) }))];
}

export default suite;
