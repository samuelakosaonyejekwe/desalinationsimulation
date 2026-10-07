// Decision support. Turns the published results of the suites, together with the live site context
// (grid carbon, renewable resource, tariffs, water stress, interest and inflation), into ranked,
// reasoned recommendations and a sustainability scorecard. Pure functions: no DOM, no network.
//
// Every recommendation states the finding (with the numbers), the action, the expected benefit, the
// benchmark it was judged against, where to act, and a literature topic for live evidence look-up.

const fin = (x) => typeof x === 'number' && Number.isFinite(x);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const f = (x, d = 2) => (fin(x) ? (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(d)) : '–');

/** Reference ranges used as benchmarks. Typical published industry practice; values are editable planning guides, not limits of law. */
export const BENCHMARKS = {
  swroSec: { label: 'SWRO membrane-system energy', unit: 'kWh/m³', good: 2.5, typical: 3.2, poor: 4.0, basis: 'Modern SWRO with isobaric energy recovery runs at about 2.2–3.0 kWh/m³ for the RO system and 3–4 kWh/m³ for the whole plant.' },
  bwroSec: { label: 'BWRO membrane-system energy', unit: 'kWh/m³', good: 0.6, typical: 1.0, poor: 1.6, basis: 'Brackish RO typically needs 0.4–1.5 kWh/m³ depending on salinity and recovery.' },
  lcow: { label: 'Levelised cost of water', unit: '$/m³', good: 0.6, typical: 1.0, poor: 1.6, basis: 'Large recent SWRO projects have been contracted at roughly 0.4–1.0 $/m³; small or remote plants commonly exceed 1.5 $/m³.' },
  carbon: { label: 'Carbon intensity of product water', unit: 'kgCO₂/m³', good: 0.8, typical: 1.8, poor: 3.0, basis: 'Grid-powered SWRO is typically 1.4–2.5 kgCO₂/m³; renewable-powered plants fall below 0.5.' },
  excessSalinity: { label: 'Excess salinity at the mixing-zone edge', unit: 'g/kg', good: 1.0, typical: 2.0, poor: 4.0, basis: 'Common regulatory practice limits the increase to about 2 g/kg (or 5 % above ambient) at the edge of a mixing zone of the order of 100 m.' },
  dilution: { label: 'Near-field dilution', unit: '×', good: 40, typical: 20, poor: 10, higherIsBetter: true, basis: 'Well-designed inclined dense-jet diffusers achieve near-field dilutions of 20–40 or more.' },
  gor: { label: 'Gain-output ratio', unit: 'kg/kg', good: 10, typical: 8, poor: 5, higherIsBetter: true, basis: 'MED-TVC plants reach GOR 9–16 and large MSF 8–10; values below about 6 indicate an uneconomic thermal design.' },
  pumpEff: { label: 'High-pressure pump efficiency', unit: '%', good: 87, typical: 83, poor: 76, higherIsBetter: true, basis: 'Large multistage HP pumps reach 86–90 %; small plunger or low-specific-speed pumps are lower.' },
  erdEff: { label: 'Energy-recovery efficiency', unit: '%', good: 96, typical: 92, poor: 80, higherIsBetter: true, basis: 'Isobaric pressure exchangers transfer 95–98 % of brine hydraulic energy; turbines 80–88 %.' },
  cleanings: { label: 'Membrane cleanings per year', unit: '1/y', good: 2, typical: 4, poor: 8, basis: 'Well-pretreated plants clean one to four times a year; more frequent cleaning signals a pretreatment or scaling problem.' },
  chemicals: { label: 'Chemical use', unit: 'kg per 1000 m³', good: 25, typical: 60, poor: 140, basis: 'Typical SWRO chemical consumption (coagulant, antiscalant, acid/caustic, bisulphite, remineralisation) is of the order of 30–80 kg per 1000 m³ of product.' },
};
/** Product-water guideline values (WHO Guidelines for Drinking-water Quality, 4th ed. incl. addenda). */
export const WATER_GUIDELINES = { tds: 600, boron: 2.4, chloride: 250, sodium: 200, nitrate: 50, fluoride: 1.5, sulphate: 250 };

/** 0–100 score from a benchmark (100 at "good" or better, 50 at "typical", 0 at "poor" or worse). */
export function score(value, b) {
  if (!fin(value)) return null;
  const s = b.higherIsBetter ? -1 : 1, v = s * value, g = s * b.good, t = s * b.typical, p = s * b.poor;
  if (v <= g) return 100;
  if (v >= p) return 0;
  return v <= t ? 100 - (50 * (v - g)) / (t - g) : 50 - (50 * (v - t)) / (p - t);
}
const rate = (s) => (s === null ? 'not assessed' : s >= 80 ? 'strong' : s >= 55 ? 'acceptable' : s >= 30 ? 'needs attention' : 'weak');

/** Shared facts assembled from whatever suites have been run. */
export function facts(outputs = {}, site = {}, feed = {}) {
  const o = outputs, d = site.data || {}, ro = o.ro, feedTds = ro?.streams?.feed?.tds ?? null;
  const seawater = fin(feedTds) ? feedTds > 15000 : true;
  const product = [o.plant?.productFlow, ro?.permeateFlow, o.thermal?.distillate, o.ed?.streams?.diluate?.Q].find(fin) ?? null; // m³/h
  const secElec = [o.plant?.secElec, o.pump?.sec, ro?.sec, o.thermal?.secElec, o.ed?.sec].find(fin) ?? null;
  const secTh = [o.plant?.secThermal, o.thermal?.secThermal].find(fin) ?? 0;
  const gridCarbon = fin(d.gridCarbon) ? d.gridCarbon : 0.45, heatCarbon = 0.2;
  const carbon = fin(o.econ?.carbonIntensity) ? o.econ.carbonIntensity : fin(secElec) ? secElec * gridCarbon + secTh * heatCarbon : null;
  const power = [o.plant?.power, o.pump?.netPower, ro?.pumpPower].find(fin) ?? (fin(secElec) && fin(product) ? secElec * product : null); // kW
  return { o, d, ro, seawater, product, secElec, secTh, gridCarbon, gridCarbonLive: !!d.gridCarbonLive, carbon, power, feedTds, price: fin(d.electricityPrice) ? d.electricityPrice : null, feed };
}

/** Renewable-supply option sized from the plant power and the site's long-term resource. */
export function renewableOption(F) {
  const ghi = fin(F.d.ghiAnnual) ? F.d.ghiAnnual : fin(F.d.ghiDaily) ? F.d.ghiDaily : null;
  if (!fin(F.power) || !fin(ghi) || ghi <= 0) return null;
  const pr = 0.78, yieldKWp = ghi * 365 * pr, annual = F.power * 8760 * 0.94; // kWh/y at 94 % availability
  const kWp = annual / yieldKWp, areaHa = (kWp * 1.6) / 1000; // ≈ 1.6 ha per MWp utility-scale, fixed tilt
  const capex = 750 * kWp, crf = (0.07 * 1.07 ** 25) / (1.07 ** 25 - 1), lcoe = (capex * crf + 12 * kWp) / (yieldKWp * kWp);
  const wind = F.d.windAnnual;
  return { ghi, yieldKWp, kWp, areaHa, lcoe, avoided: (annual * F.gridCarbon) / 1000, directShare: 0.28 + 0.02 * clamp(ghi - 4, 0, 3), windClass: fin(wind) ? (wind * 1.35 > 7 ? 'good' : wind * 1.35 > 5.5 ? 'moderate' : 'low') : null, wind, gridPrice: F.price, annual };
}

const R = (priority, area, title, why, action, benefit, goTo, basis, topic, suites) => ({ priority, area, title, why, action, benefit, goTo, basis, topic, suites });

/** All rules. Each returns zero or more recommendations from the facts. */
function rules(F) {
  const { o, d, ro } = F, out = [], B = BENCHMARKS;
  // ---- energy
  if (ro && fin(ro.sec)) {
    const b = F.seawater ? B.swroSec : B.bwroSec;
    if (ro.erdType === 'none' && F.seawater) out.push(R(1, 'Energy', 'Add isobaric energy recovery', `The concentrate leaves at ${f(ro.concentratePressureBar)} bar and the RO system uses ${f(ro.sec)} kWh/m³ with no energy recovery.`, 'Fit a pressure exchanger on the concentrate line and resize the high-pressure pump for the permeate flow only.', 'Typically cuts membrane-system energy by 50–60 % and pays back within one to three years.', 'ro', b.basis, 'seawater reverse osmosis isobaric energy recovery device', ['ro', 'pump', 'plant']));
    else if (ro.sec > b.typical) out.push(R(2, 'Energy', 'Membrane-system energy is above typical practice', `${f(ro.sec)} kWh/m³ against a typical ${b.typical} and best practice ${b.good} kWh/m³ (feed pressure ${f(ro.feedPressureBar)} bar, recovery ${f(100 * ro.recovery, 0)} %, flux ${f(ro.fluxLMH, 1)} L/m²·h).`, 'Test a lower flux (more area), a low-energy element class, a slightly lower recovery, and inter-stage or internally-staged designs in the optimiser.', 'Each bar of feed pressure saved lowers energy by roughly 0.03–0.06 kWh/m³.', 'opt', b.basis, 'low energy seawater reverse osmosis design optimisation', ['ro', 'opt']));
    if (fin(ro.cpFactor) && ro.cpFactor > 1.2) out.push(R(2, 'Reliability', 'Concentration polarisation is high', `The polarisation factor reaches ${f(ro.cpFactor)} (guideline ≤ 1.2), which raises wall salinity, scaling tendency and salt passage.`, 'Lower the lead-element flux (permeate back-pressure or more vessels), raise cross-flow, or select a feed spacer with better mixing using the CFD suite.', 'Lower scaling risk and better permeate quality.', 'cfd', 'Membrane manufacturers limit β to about 1.2 per element.', 'concentration polarization feed spacer reverse osmosis mass transfer', ['ro', 'cfd']));
  }
  if (o.pump) {
    if (fin(o.pump.pumpEfficiency) && o.pump.pumpEfficiency < B.pumpEff.typical / (o.pump.pumpEfficiency > 1 ? 1 : 100)) { const e = o.pump.pumpEfficiency > 1 ? o.pump.pumpEfficiency : 100 * o.pump.pumpEfficiency; out.push(R(2, 'Energy', 'High-pressure pump efficiency is low', `${f(e, 0)} % against ${B.pumpEff.good} % achievable for large multistage pumps.`, 'Re-select the pump closer to its best-efficiency point, or use a variable-speed drive instead of throttling.', `Each efficiency point is worth about ${f((o.pump.hpPumpPower || 0) * 0.012, 0)} kW continuously.`, 'pump', B.pumpEff.basis, 'high pressure pump efficiency reverse osmosis variable speed', ['pump'])); }
    if (fin(o.pump.erdEfficiency)) { const e = o.pump.erdEfficiency > 1 ? o.pump.erdEfficiency : 100 * o.pump.erdEfficiency; if (e < B.erdEff.typical) out.push(R(2, 'Energy', 'Energy-recovery efficiency is below isobaric practice', `${f(e, 0)} % recovered against ${B.erdEff.good} % for pressure exchangers.`, 'Compare a pressure exchanger with the present device in the pump suite’s option table.', 'Net specific energy typically falls by 0.3–0.8 kWh/m³ when replacing a turbine with an isobaric device.', 'pump', B.erdEff.basis, 'pressure exchanger versus Pelton turbine desalination', ['pump'])); }
  }
  if (o.thermal && fin(o.thermal.GOR) && o.thermal.GOR < B.gor.typical) out.push(R(2, 'Energy', 'Thermal performance ratio is low', `GOR ${f(o.thermal.GOR, 1)} against ${B.gor.typical}–${B.gor.good} for modern plants; thermal energy ${f(o.thermal.secThermal, 0)} kWh/m³.`, 'Add effects or stages, add thermal vapour compression, raise the top brine temperature within the scale limit, or hybridise with RO.', 'Steam demand falls roughly in proportion to the GOR gained.', 'thermal', B.gor.basis, 'multi-effect distillation thermal vapor compression gain output ratio', ['thermal']));
  // ---- scaling and chemistry
  if (o.chem) {
    if (fin(o.chem.maxRecovery) && ro && fin(ro.recovery) && ro.recovery > o.chem.maxRecovery + 0.005) out.push(R(1, 'Scaling', 'Design recovery exceeds the scaling limit', `The RO design recovers ${f(100 * ro.recovery, 0)} % but ${o.chem.limitingMineral || 'a mineral'} limits recovery to about ${f(100 * o.chem.maxRecovery, 0)} % with the present dosing.`, 'Reduce recovery, lower the feed pH, or raise the antiscalant dose and confirm the new limit in the chemistry suite.', 'Avoids irreversible scale on the tail elements and unplanned cleanings.', 'chem', 'Saturation indices from the electrolyte model of suite 2 with antiscalant thresholds.', `${o.chem.limitingMineral || 'calcium sulfate'} scaling antiscalant reverse osmosis recovery limit`, ['ro', 'chem']));
    else if (fin(o.chem.maxRecovery) && ro && fin(ro.recovery) && o.chem.maxRecovery - ro.recovery > 0.08 && !F.seawater) out.push(R(3, 'Sustainability', 'Recovery could be raised', `Scaling allows about ${f(100 * o.chem.maxRecovery, 0)} % recovery while the design uses ${f(100 * ro.recovery, 0)} %.`, 'Evaluate a higher recovery (add a stage or concentrate recycle) in the RO suite.', `Brine volume would fall by roughly ${f(100 * (1 - (1 - o.chem.maxRecovery + 0.03) / (1 - ro.recovery)), 0)} %.`, 'ro', 'Scaling-limited recovery from suite 2.', 'high recovery brackish reverse osmosis concentrate minimisation', ['ro', 'chem']));
    if (fin(o.chem.lsi) && o.chem.lsi < -1.5) out.push(R(3, 'Reliability', 'Water is strongly corrosive', `Langelier index ${f(o.chem.lsi)}.`, 'Check materials (duplex / super-duplex, GRP) and remineralise the product before distribution.', 'Protects pipework and meets stability targets.', 'plant', 'LSI between about −0.5 and +0.5 is considered stable.', 'desalinated water remineralization corrosion Langelier', ['chem', 'plant']));
  }
  // ---- water quality
  const p = ro?.streams?.permeate;
  if (p) {
    if (fin(p.tds) && p.tds > WATER_GUIDELINES.tds) out.push(R(1, 'Water quality', 'Product salinity is above the palatability guideline', `Product TDS ${f(p.tds, 0)} mg/L against ${WATER_GUIDELINES.tds} mg/L.`, 'Use a higher-rejection element class, lower recovery or temperature, or add a partial second pass.', 'Brings product within drinking-water guidance.', 'ro', 'WHO drinking-water guidance: TDS below about 600 mg/L is generally considered good.', 'reverse osmosis permeate quality second pass design', ['ro']));
    if (fin(p.ions?.B) && p.ions.B > WATER_GUIDELINES.boron) out.push(R(1, 'Water quality', 'Product boron is above the WHO guideline', `Boron ${f(p.ions.B)} mg/L against ${WATER_GUIDELINES.boron} mg/L.`, 'Add a second pass at pH 10–10.5 or use boron-selective resin on part of the permeate.', 'Meets the guideline and protects boron-sensitive crops if the water is used for irrigation.', 'ro', 'WHO guideline value for boron: 2.4 mg/L.', 'boron removal seawater reverse osmosis second pass pH', ['ro']));
  }
  // ---- environment
  if (o.sea) {
    if (fin(o.sea.excessAtMixingZone) && o.sea.excessAtMixingZone > B.excessSalinity.typical) out.push(R(1, 'Environment', 'Brine plume exceeds the mixing-zone criterion', `Excess salinity at the mixing-zone edge is ${f(o.sea.excessAtMixingZone)} g/kg against a common limit of ${B.excessSalinity.typical} g/kg` + (fin(o.sea.complianceDistance) ? `; compliance is reached at about ${f(o.sea.complianceDistance, 0)} m.` : '.'), 'Increase port exit velocity and number of ports, incline the jets at about 60°, move the diffuser to deeper water, or pre-dilute with cooling or intake by-pass water.', 'Protects benthic habitat and keeps the discharge permit compliant.', 'sea', B.excessSalinity.basis, 'brine discharge multiport diffuser inclined dense jet dilution', ['sea']));
    if (fin(o.sea.nearFieldDilution) && o.sea.nearFieldDilution < B.dilution.typical) out.push(R(2, 'Environment', 'Near-field dilution is low', `Dilution ${f(o.sea.nearFieldDilution, 0)}× against ${B.dilution.typical}–${B.dilution.good}× for a well-designed diffuser.`, 'Raise the port densimetric Froude number (smaller or more ports) and use the diffuser design helper.', 'Faster mixing and a smaller seabed footprint.', 'sea', B.dilution.basis, 'negatively buoyant jet densimetric Froude number dilution', ['sea']));
  }
  if (ro && fin(ro.recovery) && !o.zld && !F.seawater && ro.recovery < 0.85) out.push(R(3, 'Sustainability', 'Inland concentrate needs a management route', `About ${f(100 * (1 - ro.recovery), 0)} % of the feed leaves as concentrate.`, 'Assess brine concentration and crystallisation or selective salt recovery in the ZLD suite.', 'Removes the liquid discharge and can turn salts into revenue.', 'zld', 'Inland plants cannot rely on marine dilution.', 'brackish concentrate minimal liquid discharge brine concentrator', ['ro', 'zld']));
  if (o.zld && fin(o.zld.liquidDischarge) && o.zld.liquidDischarge > 0.01) out.push(R(2, 'Sustainability', 'Liquid discharge remains after the ZLD train', `${f(o.zld.liquidDischarge)} m³/h of purge or mother liquor is still discharged.`, 'Route the purge to an evaporation pond or dryer, or tighten the crystalliser purge after checking impurity build-up.', 'Achieves true zero liquid discharge.', 'zld', 'Zero-liquid-discharge balance of suite 9.', 'zero liquid discharge crystallizer purge management', ['zld']));
  // ---- reliability
  if (o.fouling) {
    if (fin(o.fouling.daysToCleaning) && o.fouling.daysToCleaning < 30) out.push(R(1, 'Reliability', 'A membrane cleaning is due soon', `Cleaning thresholds are forecast to be reached in about ${f(o.fouling.daysToCleaning, 0)} days` + (o.fouling.dominantFoulant ? `; the dominant mechanism is ${o.fouling.dominantFoulant}.` : '.'), 'Schedule a clean-in-place matched to the foulant and review pretreatment performance.', 'Cleaning before a 10–15 % normalised-flow loss keeps fouling reversible.', 'fouling', 'Common trigger: 10–15 % drop in normalised permeate flow or 15 % rise in differential pressure.', `${o.fouling.dominantFoulant || 'membrane'} fouling reverse osmosis cleaning strategy`, ['fouling']));
    if (fin(o.fouling.cleaningsPerYear) && o.fouling.cleaningsPerYear > B.cleanings.typical) out.push(R(2, 'Reliability', 'Cleaning frequency is high', `${f(o.fouling.cleaningsPerYear, 1)} cleanings per year against ${B.cleanings.good}–${B.cleanings.typical} for well-pretreated plants.`, 'Improve pretreatment (coagulation, UF or DAF), review biocide and antiscalant programmes.', 'Longer membrane life and less chemical waste.', 'plant', B.cleanings.basis, 'pretreatment ultrafiltration seawater reverse osmosis fouling reduction', ['fouling', 'plant']));
  }
  // ---- cost
  if (o.econ && fin(o.econ.lcow)) {
    if (o.econ.lcow > B.lcow.typical) out.push(R(2, 'Cost', 'Water cost is above typical contracts', `Levelised cost ${f(o.econ.lcow)} $/m³ against ${B.lcow.good}–${B.lcow.typical} $/m³ for large modern plants.`, 'Open the economics tornado chart to find the dominant driver; then test capacity, energy price, financing and recovery scenarios.', 'Targets the few parameters that move the cost most.', 'econ', B.lcow.basis, 'levelized cost of water seawater desalination drivers', ['econ']));
    if (fin(o.econ.npv) && o.econ.npv < 0) out.push(R(1, 'Cost', 'The project does not recover its cost of capital', `Net present value is ${f(o.econ.npv / 1e6, 1)} M$ at the stated tariff and discount rate.`, 'Review the water tariff, financing terms, capacity factor and energy supply; the break-even tariff is reported in the economics suite.', 'Bankable project structure.', 'econ', 'NPV ≥ 0 at the weighted cost of capital.', 'desalination project finance water tariff', ['econ']));
  }
  if (fin(d.lendingRate) && d.lendingRate > 10) out.push(R(3, 'Cost', 'Local cost of debt is high', `The latest published lending rate is ${f(d.lendingRate, 1)} %/y${d.lendingRateYear ? ` (${d.lendingRateYear})` : ''}.`, 'Capital-light choices (higher flux, phased capacity) and concessional or export-credit finance weigh more heavily here; test them in the economics suite.', 'Lower capital charge per m³.', 'econ', 'World Bank lending interest rate indicator.', 'desalination financing cost of capital developing countries', ['econ']));
  // ---- climate and renewables
  const ren = renewableOption(F);
  if (fin(F.carbon) && F.carbon > B.carbon.typical) out.push(R(2, 'Sustainability', 'Carbon intensity of the water is high', `${f(F.carbon)} kgCO₂/m³ with grid electricity at ${f(F.gridCarbon)} kgCO₂/kWh${F.gridCarbonLive ? ` (${d.gridCarbonYear || 'latest'} national average)` : ' (default value — pull site data for the national figure)'}.`, ren ? `Supply part of the load from solar: about ${f(ren.kWp / 1000, 1)} MWp would match the annual energy.` : 'Assess renewable supply and efficiency measures.', ren ? `Up to ${f(ren.avoided, 0)} tCO₂ avoided per year.` : 'Lower emissions.', 'econ', B.carbon.basis, 'renewable energy powered desalination carbon footprint', ['econ', 'plant', 'pump', 'ro']));
  if (ren && fin(ren.gridPrice) && ren.lcoe < ren.gridPrice * 0.9) out.push(R(2, 'Sustainability', 'Solar electricity is cheaper than the grid at this site', `Long-term irradiation is ${f(ren.ghi)} kWh/m²·d, giving an indicative solar cost of ${f(ren.lcoe, 3)} $/kWh against a grid price of ${f(ren.gridPrice, 3)} $/kWh.`, `A plant of about ${f(ren.kWp / 1000, 1)} MWp on ${f(ren.areaHa, 1)} ha matches the annual energy; around ${f(100 * ren.directShare, 0)} % can be used directly without storage, more with product-water storage and flexible operation.`, `Lower energy cost and up to ${f(ren.avoided, 0)} tCO₂/y avoided.`, 'econ', 'NASA POWER long-term irradiation; indicative utility-scale PV cost of 750 $/kWp at 7 % over 25 years.', 'photovoltaic powered reverse osmosis flexible operation', ['econ', 'plant', 'pump', 'ro', 'thermal']));
  if (fin(d.waterStress) && d.waterStress > 80) out.push(R(3, 'Sustainability', 'The country is under high water stress', `Freshwater withdrawals are ${f(d.waterStress, 0)} % of available renewable resources${d.waterStressYear ? ` (${d.waterStressYear})` : ''}.`, 'Prioritise high recovery, low losses, reuse of backwash water and coupling with wastewater reuse where possible.', 'Each percentage point of recovery reduces intake and pretreatment duty.', 'plant', 'SDG indicator 6.4.2 (World Bank / FAO).', 'desalination water scarcity integrated water reuse', ['plant', 'ro']));
  if (o.plant && fin(o.plant.chemicals) && fin(F.product) && F.product > 0) { const kg = (o.plant.chemicals / (F.product * 24)) * 1000; if (kg > B.chemicals.typical) out.push(R(3, 'Sustainability', 'Chemical intensity is high', `${f(kg, 0)} kg of chemicals per 1000 m³ of product against a typical ${B.chemicals.typical}.`, 'Review coagulant and antiscalant doses against the chemistry suite, and consider UF pretreatment to cut coagulant use.', 'Less chemical transport, cost and residual discharge.', 'plant', B.chemicals.basis, 'chemical consumption seawater reverse osmosis pretreatment', ['plant', 'chem'])); }
  return out;
}

/** Ranked recommendations. Pass a suite id to keep only those that concern that suite. */
export function advise(outputs, site, feed, suiteId) {
  const F = facts(outputs, site, feed);
  let recs = rules(F);
  if (suiteId) recs = recs.filter((r) => r.suites.includes(suiteId) || r.goTo === suiteId);
  return recs.sort((a, b) => a.priority - b.priority);
}

/** Sustainability scorecard: pillars scored 0–100 from benchmarks, with the evidence behind each. */
export function sustainability(outputs, site, feed) {
  const F = facts(outputs, site, feed), { o, d, ro } = F, B = BENCHMARKS, P = [];
  const add = (id, title, sdg, value, unit, s, detail, need) => P.push({ id, title, sdg, value, unit, score: s, rating: rate(s), detail, need });
  const bSec = F.seawater ? B.swroSec : B.bwroSec;
  add('energy', 'Energy efficiency', 'SDG 7', F.secElec, 'kWh/m³', fin(F.secElec) ? score(F.secElec, bSec) : null, fin(F.secElec) ? `Electricity ${f(F.secElec)} kWh/m³${F.secTh ? ` plus ${f(F.secTh, 0)} kWh/m³ heat` : ''}; best practice ${bSec.good}.` : '', 'Run the RO, pump or plant suite');
  add('climate', 'Climate impact', 'SDG 13', F.carbon, 'kgCO₂/m³', fin(F.carbon) ? score(F.carbon, B.carbon) : null, fin(F.carbon) ? `Grid factor ${f(F.gridCarbon)} kgCO₂/kWh${F.gridCarbonLive ? ' (live national value)' : ' (default)'}${fin(d.renewableShare) ? `; ${f(d.renewableShare, 0)} % of national electricity is renewable` : ''}.` : '', 'Run a suite that reports energy');
  const ex = o.sea?.excessAtMixingZone, dil = o.sea?.nearFieldDilution;
  add('marine', 'Marine and brine impact', 'SDG 14', fin(ex) ? ex : null, 'g/kg excess', fin(ex) ? score(ex, B.excessSalinity) : fin(dil) ? score(dil, B.dilution) : null, fin(ex) ? `Excess salinity ${f(ex)} g/kg at the mixing-zone edge${fin(dil) ? `, near-field dilution ${f(dil, 0)}×` : ''}.` : '', 'Run the sea-discharge suite');
  const rec = [o.plant?.recovery, ro?.recovery].find(fin);
  const recB = F.seawater ? { good: 0.5, typical: 0.42, poor: 0.3, higherIsBetter: true } : { good: 0.88, typical: 0.78, poor: 0.6, higherIsBetter: true };
  add('water', 'Water-use efficiency', 'SDG 6', fin(rec) ? 100 * rec : null, '% recovery', fin(rec) ? score(rec, recB) : null, fin(rec) ? `${f(100 * rec, 0)} % of the intake becomes product${fin(d.waterStress) ? `; national water stress ${f(d.waterStress, 0)} %` : ''}.` : '', 'Run the RO or plant suite');
  const chem = o.plant && fin(o.plant.chemicals) && fin(F.product) && F.product > 0 ? (o.plant.chemicals / (F.product * 24)) * 1000 : null;
  add('chemicals', 'Chemical footprint', 'SDG 12', chem, 'kg/1000 m³', fin(chem) ? score(chem, B.chemicals) : null, fin(chem) ? `Typical ${B.chemicals.typical} kg per 1000 m³.` : '', 'Run the plant suite');
  const solids = o.zld?.solids, liquid = o.zld?.liquidDischarge;
  add('circular', 'Resource recovery and circularity', 'SDG 12', fin(solids) ? solids : null, 't/d salts', o.zld ? clamp(40 + (fin(liquid) && liquid < 0.01 ? 40 : 10) + (o.zld.salts && Object.keys(o.zld.salts).length > 1 ? 20 : 5), 0, 100) : null, o.zld ? `${f(solids, 1)} t/d of solids recovered; residual liquid discharge ${f(liquid)} m³/h.` : '', 'Run the ZLD suite to assess salt and water recovery from brine');
  const lc = o.econ?.lcow;
  add('afford', 'Affordability', 'SDG 6.1', fin(lc) ? lc : null, '$/m³', fin(lc) ? score(lc, B.lcow) : null, fin(lc) ? `Levelised cost ${f(lc)} $/m³${fin(d.gdpPerCapita) ? `; 50 m³ per person-year would cost ${f((100 * 50 * lc) / d.gdpPerCapita, 2)} % of GDP per capita` : ''}.` : '', 'Run the economics suite');
  const cl = o.fouling?.cleaningsPerYear, margin = fin(o.chem?.maxRecovery) && fin(ro?.recovery) ? o.chem.maxRecovery - ro.recovery : null;
  const sRel = [fin(cl) ? score(cl, B.cleanings) : null, fin(margin) ? clamp(50 + 500 * margin, 0, 100) : null].filter((x) => x !== null);
  add('resilience', 'Operational resilience', 'SDG 9', fin(cl) ? cl : null, 'cleanings/y', sRel.length ? sRel.reduce((a, b) => a + b, 0) / sRel.length : null, [fin(cl) ? `${f(cl, 1)} cleanings per year` : '', fin(margin) ? `scaling margin ${f(100 * margin, 0)} recovery points` : ''].filter(Boolean).join('; '), 'Run the fouling and chemistry suites');
  const scored = P.filter((p) => p.score !== null);
  const overall = scored.length ? scored.reduce((a, p) => a + p.score, 0) / scored.length : null;
  return { pillars: P, overall, rating: rate(overall), assessed: scored.length, renewable: renewableOption(F), facts: F };
}
