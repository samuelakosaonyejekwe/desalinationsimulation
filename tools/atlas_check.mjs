// Verification of the built-in world atlas (js/data/atlas_*.js) against the live sources.
//
//   node tools/atlas_check.mjs             compare the atlas-only site record of 15 coastal sites with the stored
//                                          reference values and print the deviations (exit code 1 on a failure)
//   node tools/atlas_check.mjs --strict    as above, but a missed target tolerance (see below) also counts as a failure
//   node tools/atlas_check.mjs --refresh   fetch the reference values again from the live services and print the
//                                          REF table to paste below (needs a connection; takes a few minutes)
//
// Reference values (REF) were taken from the live services at build time, at the exact site coordinates:
//   sal, sst   SeaDataCloud climatology, 0.25° (the app's own live salinity connector): median salinity, 12 monthly temperatures
//   sstYear    Open-Meteo marine sea-surface temperature at the site (the app's live sea-state source), monthly means of the wave year
//   depth      NOAA NCEI global DEM mosaic at its native resolution (the app's live relief connector); 0 = on land
//   tide       Open-Meteo marine sea level at the site, same period as the atlas: median over 8-day windows of the 2nd-98th percentile range
//   hs         Open-Meteo marine significant wave height at the site, mean of the hourly values of the atlas wave year
// The two live temperature references disagree with each other by up to several degrees (single 0.25° nodes of the
// SeaDataCloud months contain artefacts, e.g. 20.7 °C in October off Dubai, and run below present-day summer temperatures in
// the Arabian Gulf; the marine model is one particular year). The atlas therefore takes its temperature from a third source
// (NOAA CoralTemp satellite climatology) and every month must lie within ±2 °C of at least one of the two references.
// Hard limits (exit code 1): salinity ±1 g/kg (±2 in enclosed seas), monthly SST as above, sea / land agreement of the depth,
// tidal range ±40 % (or ±0.15 m), wave height within a factor of two, exact country code, presence and shape of the
// national, climate, relief and series fields, and usable values from every suite's site hook.
// Targets (reported; failures only with --strict): depth in the same class (under 30 m / 30-200 m / deeper) or within a
// factor of two; wave height ±40 % (or ±0.15 m). The 5 km relief overestimates the depth close inshore on steep, narrow
// shelves, and the 2° wave table cannot follow local sheltering by islands, reefs and bays.
import { fetchSite } from '../js/core/live.js';
import { atlasData } from '../js/data/atlas_lookup.js';

const SITES = [
  // name, lat, lon, country, enclosed sea?
  ['Dubai', 25.10, 55.05, 'AE', true], ['Jeddah', 21.55, 39.08, 'SA', true], ['Limassol', 34.64, 33.05, 'CY', true], ['Perth', -32.20, 115.72, 'AU', false], ['Carlsbad CA', 33.14, -117.36, 'US', false],
  ['Ashkelon', 31.64, 34.49, 'IL', true], ['Chennai', 12.70, 80.27, 'IN', false], ['Singapore', 1.27, 103.88, 'SG', false], ['Cape Town', -33.90, 18.38, 'ZA', false], ['Antofagasta', -23.60, -70.43, 'CL', false],
  ['Tampa', 27.60, -82.80, 'US', false], ['Barcelona', 41.28, 2.15, 'ES', true], ['Agadir', 30.25, -9.70, 'MA', false], ['Lagos', 6.38, 3.40, 'NG', false], ['Sydney', -34.03, 151.25, 'AU', false],
];

// ---- reference values from the live services (written by --refresh) ---------------------------------------
const REF = {
  "Dubai": {cc: "AE", sal: 38.61, sst: [22,21.4,22.8,22.4,27.1,27.8,30.4,30.1,32.5,20.7,26.4,24.4], sstYear: [21.4,21.9,23,27,30.5,32.8,34,34.3,33.5,31.3,27.7,24.4], depth: 9, tide: 1.51, hs: 0.47},
  "Jeddah": {cc: "SA", sal: 39.14, sst: [27,25.1,26,27,27.9,29.4,29.4,31.6,31,30.8,30,28.5], sstYear: [27.3,26.1,26.2,27.6,28.3,29.1,30.8,31.8,30.7,29.5,30.1,28.6], depth: 705, tide: 0.23, hs: 0.7},
  "Limassol": {cc: "CY", sal: 39.19, sst: [18.3,17.1,17,18,19.4,24.1,26.8,27.9,26.7,24.3,22.5,20.1], sstYear: [19.2,17.9,18.1,18.3,20.3,24.2,26.6,27.6,27.4,25.5,24.7,22.1], depth: 34, tide: 0.35, hs: 0.61},
  "Perth": {cc: "AU", sal: 35.68, sst: [21.3,21.4,22.1,21.9,21.8,21.9,20.4,19.7,18.9,19,18.4,20.4], sstYear: [23.7,23.9,23.3,22.6,20.7,18.8,17.1,16.5,17.2,19,20.3,22.2], depth: 20, tide: 0.63, hs: 1.54},
  "Carlsbad CA": {cc: "US", sal: 33.5, sst: [14.6,14.6,14.8,15.3,16.8,17.4,18.8,20.7,20.2,19.5,17.9,16.3], sstYear: [14.2,14.6,14.8,16,18.2,20,21.4,22.1,22.2,19.3,18.5,17.4], depth: 25.8, tide: 1.73, hs: 0.92},
  "Ashkelon": {cc: "IL", sal: 39.16, sst: [18.1,17.2,17.4,18.7,20.6,24.7,27.4,28.3,28,25.7,23.3,20.3], sstYear: [18.4,17,18.4,20.3,22.7,25.9,28.9,29.9,29.4,27.4,25.2,21.8], depth: 32, tide: 0.39, hs: 0.75},
  "Chennai": {cc: "IN", sal: 33.46, sst: [26.7,27,28.7,29.8,30.4,29.7,29,29.1,28.9,29.3,28.8,27.7], sstYear: [26.6,28.1,29.1,29.9,30.3,29.8,29.4,29.3,29.8,29.7,28.8,26.6], depth: 19, tide: 0.89, hs: 0.66},
  "Singapore": {cc: "SG", sal: 32.38, sst: [27.3,28.1,28.7,29.4,29.8,29.4,29.6,28.7,28.9,29.3,29.1,27.8], sstYear: [28,28.1,28.8,30,31.2,31,30.6,30.4,30,29.8,30.3,29], depth: 15, tide: 2.3, hs: 0.17},
  "Cape Town": {cc: "ZA", sal: 35.17, sst: [16.9,16.8,16.1,14.9,16.1,15.6,15.3,15.1,15.2,15.7,15.7,16.7], sstYear: [15.8,11.8,15.8,15,14.1,14.2,14.5,14.5,14.7,14.5,13,13.2], depth: 34, tide: 1.56, hs: 2.3},
  "Antofagasta": {cc: "CL", sal: 34.73, sst: [18.6,20.2,20.8,18.3,17.8,16.6,15.8,15.3,16.9,16.9,17.1,18.1], sstYear: [19.4,19.9,18.8,17,16,15.5,14.9,15.6,15.4,16.2,17,18.3], depth: 83, tide: 1.07, hs: 1.63},
  "Tampa": {cc: "US", sal: 35.97, sst: [19.1,17.7,21,22,24.6,27.1,29.7,30.3,29.3,26.9,24.4,23.2], sstYear: [15.5,20,20.1,24.2,28.2,30.2,31,31.6,29.3,26.6,21.7,20.4], depth: 5.7, tide: 0.72, hs: 0.41},
  "Barcelona": {cc: "ES", sal: 38.01, sst: [13.2,12.8,12.9,13.9,16.2,19.2,21.8,24.9,24,19.4,17.4,14.9], sstYear: [14.6,14.1,13.8,15.7,18.7,23.5,26.8,27.3,25.8,22.7,18.5,15.8], depth: 38, tide: 0.21, hs: 0.64},
  "Agadir": {cc: "MA", sal: 36.41, sst: [17.4,16.7,17,18.3,18.1,18.4,17.7,18.3,21.4,19.4,19.4,18.2], sstYear: [18.2,16.8,17.6,18.5,18.7,19,19.5,21.5,20.3,20.2,20,17.5], depth: 69, tide: 2.63, hs: 1.6},
  "Lagos": {cc: "NG", sal: 35.01, sst: [28.2,28.6,29.1,29.1,29.2,28.4,26.6,25,26.3,27.4,28.4,28.6], sstYear: [28.9,28.7,29.5,30,29.8,29.1,26.4,25.8,26.6,28.1,29.1,29.2], depth: 12, tide: 1.46, hs: 1.11},
  "Sydney": {cc: "AU", sal: 35.56, sst: [22.4,23,23.1,22.1,20.3,18.6,17.8,17.1,18.2,18.7,20.1,20.6], sstYear: [22,22.5,22.1,22.4,21.2,19.2,18.2,18.6,19,19.7,20,21.5], depth: 49, tide: 1.38, hs: 1.59},
};
const REF_DATE = '2026-10-08';

const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length ? (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 : NaN; };
const q = (a, p) => { const s = a.slice().sort((x, y) => x - y), i = (s.length - 1) * p, lo = Math.floor(i); return s[lo] + (s[Math.min(s.length - 1, lo + 1)] - s[lo]) * (i - lo); };
const depthClass = (z) => (z <= 0 ? 'land' : z < 30 ? 'under 30 m' : z < 200 ? '30-200 m' : 'deeper');

async function refresh() {
  const C = await atlasData('coast'), [t0, t1] = C.seaPeriod.split(' to '), [w0, w1] = C.wavePeriod.split(' to '), out = {};
  const get = async (u) => { for (let k = 0; k < 5; k++) { try { const r = await fetch(u); if (r.ok) return await r.json(); } catch { /* retry */ } await new Promise((res) => setTimeout(res, 5000 * (k + 1))); } return null; };
  for (const [name, lat, lon] of SITES) {
    const s = await fetchSite(lat, lon, () => {}, () => {}, { fresh: true }), d = s.data, af = new Set(d.atlasFields || []);
    const live = (k) => (af.has(k) || d[k] == null ? null : d[k]);
    const sea = await get(`https://marine-api.open-meteo.com/v1/marine?latitude=${lat}&longitude=${lon}&hourly=sea_level_height_msl&start_date=${t0}&end_date=${t1}&timezone=GMT&cell_selection=sea`);
    const eta = (sea?.hourly?.sea_level_height_msl || []).map((x) => (typeof x === 'number' ? x : NaN)), win = [];
    for (let a = 0; a + 192 <= eta.length; a += 24) { const w = eta.slice(a, a + 192).filter(Number.isFinite); if (w.length > 150) win.push(q(w, 0.98) - q(w, 0.02)); }
    const wv = await get(`https://marine-api.open-meteo.com/v1/marine?latitude=${lat}&longitude=${lon}&hourly=wave_height&start_date=${w0}&end_date=${w1}&timezone=GMT&cell_selection=sea`);
    const hs = (wv?.hourly?.wave_height || []).filter((x) => typeof x === 'number');
    const st = await get(`https://marine-api.open-meteo.com/v1/marine?latitude=${lat}&longitude=${lon}&hourly=sea_surface_temperature&start_date=${w0}&end_date=${w1}&timezone=GMT&cell_selection=sea`);
    const byM = Array.from({ length: 12 }, () => []); (st?.hourly?.time || []).forEach((t, i) => { const v = st.hourly.sea_surface_temperature[i]; if (typeof v === 'number') byM[+t.slice(5, 7) - 1].push(v); });
    const sstYear = byM.every((a) => a.length > 300) ? byM.map((a) => +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(1)) : null;
    out[name] = { cc: s.countryCode && !af.has('country') ? s.countryCode : null, sal: live('salinity') && +live('salinity').toFixed(2), sst: live('sstMonthly') && live('sstMonthly').map((x) => +x.toFixed(1)), depth: live('depth') === null ? null : +live('depth').toFixed(1),
      sstYear, tide: win.length ? +median(win).toFixed(2) : null, hs: hs.length > 4000 ? +(hs.reduce((a, b) => a + b, 0) / hs.length).toFixed(2) : null };
    console.error(name, JSON.stringify(out[name]));
  }
  console.log('const REF = ' + JSON.stringify(out).replace(/"(\w+)":/g, '$1:').replace(/\},/g, '},\n  ') + ';');
  console.log(`const REF_DATE = '${new Date().toISOString().slice(0, 10)}';`);
}

async function check() {
  let fails = 0;
  const rows = [], bad = (name, msg) => { fails++; console.log(`  FAIL ${name}: ${msg}`); };
  const dev = { sal: [], sst: [], sdc: [], near: [], tide: [], hs: [] }, outside = [], deep = [];
  for (const [name, lat, lon, cc, enclosed] of SITES) {
    const r = REF[name] || {}, s = await fetchSite(lat, lon, () => {}, () => {}, { atlasOnly: true }), d = s.data, af = new Set(d.atlasFields || []);
    const row = { site: name };
    row.country = s.countryCode; if (s.countryCode !== cc) bad(name, `country ${s.countryCode}, expected ${cc}`);
    if (r.sal != null) { const e = d.salinity - r.sal; row.salinity = `${d.salinity.toFixed(1)} (${e >= 0 ? '+' : ''}${e.toFixed(2)})`; dev.sal.push(Math.abs(e)); if (!(Math.abs(e) <= (enclosed ? 2 : 1))) bad(name, `salinity ${d.salinity} vs ${r.sal}`); } else row.salinity = d.salinity?.toFixed(1) + ' (no ref)';
    if (d.sstMonthly && (r.sstYear || r.sst)) { // each month within ±2 °C of at least one of the two live references
      const eY = r.sstYear ? r.sstYear.map((x, m) => Math.abs(d.sstMonthly[m] - x)) : null, eC = r.sst ? r.sst.map((x, m) => Math.abs(d.sstMonthly[m] - x)) : null;
      const best = d.sstMonthly.map((_, m) => Math.min(eY ? eY[m] : Infinity, eC ? eC[m] : Infinity)), worst = Math.max(...best);
      if (eY) { row['SST vs model year (max)'] = Math.max(...eY).toFixed(1) + ' °C'; dev.sst.push(Math.max(...eY)); }
      if (eC) { row['SST vs SDC (median / max)'] = `${median(eC).toFixed(1)} / ${Math.max(...eC).toFixed(1)}`; dev.sdc.push(median(eC)); }
      row['SST nearest ref (max)'] = worst.toFixed(1) + (worst <= 2 ? '' : ' ✗'); dev.near.push(worst);
      if (!(worst <= 2)) bad(name, `a monthly SST is ${worst.toFixed(2)} °C from both live references`);
    } else bad(name, 'monthly SST missing');
    if (r.depth != null) { // hard limit: sea or land must agree; target: same depth class or within a factor of two
      const a = d.depth, okc = depthClass(a) === depthClass(r.depth) || (a > 0 && r.depth > 0 && a / r.depth < 2 && r.depth / a < 2);
      row.depth = `${a} m vs ${r.depth} m${okc ? '' : ' ✗'}`;
      if ((a > 0) !== (r.depth > 0)) bad(name, `depth ${a} m vs ${r.depth} m: sea / land disagree`); else if (!okc) deep.push(name);
    } else row.depth = d.depth + ' m';
    for (const [k, key, label] of [['tide', 'tideRange', 'tide'], ['hs', 'waveHeight', 'Hs']]) {
      if (r[k] != null && d[key] != null) {
        const e = d[key] - r[k], rel = r[k] > 0 ? (100 * e) / r[k] : 0, within = Math.abs(rel) <= 40 || Math.abs(e) <= 0.15;
        row[label] = `${d[key].toFixed(2)} vs ${r[k].toFixed(2)} m (${rel >= 0 ? '+' : ''}${rel.toFixed(0)} %)${within ? '' : ' ✗'}`; dev[k].push(Math.abs(rel));
        // waves: the 2° table cannot follow local sheltering, so ±40 % is the target (counted below) and a factor of two the hard limit
        if (k === 'hs') { if (!within) outside.push(name); if (!(d[key] / r[k] <= 2 && r[k] / d[key] <= 2) && Math.abs(e) > 0.15) bad(name, `${label} ${d[key]} vs ${r[k]} (beyond a factor of two)`); }
        else if (!within) bad(name, `${label} ${d[key]} vs ${r[k]}`);
      }
      else { row[label] = d[key] != null ? d[key].toFixed(2) + ' m (no ref)' : 'missing'; if (d[key] == null) bad(name, label + ' missing'); }
    }
    const need = ['inflation', 'gdpPerCapita', 'currency', 'fxPerUSD', 'gridCarbon', 'electricityPrice', 'ghiDaily', 'ghiAnnual', 'windAnnual', 'airTemp', 'windSpeed', 'bathy', 'tide', 'currentSpeed', 'wavePeriod', 'maxDepthNearby', 'salinityMonthly', 'sst'];
    const miss = need.filter((k) => d[k] == null);
    if (miss.length) bad(name, 'missing fields: ' + miss.join(', '));
    if (d.bathy && (d.bathy.elev.length !== d.bathy.lat.length || d.bathy.elev[0].length !== d.bathy.lon.length || d.bathy.elev.flat().some((z) => !Number.isFinite(z)))) bad(name, 'bathy grid malformed');
    if (d.tide && (d.tide.eta.length !== d.tide.t.length || d.tide.eta.some((z) => !Number.isFinite(z)))) bad(name, 'tide series malformed');
    if (!af.has('salinity') || !Object.values(s.status).every((x) => x.atlas || !x.ok)) bad(name, 'atlas book-keeping (atlasFields / status) incomplete');
    row.fields = `${af.size} atlas`; row.GHI = d.ghiDaily; row.infl = d.inflation != null ? `${d.inflation} (${d.inflationYear})` : '–';
    rows.push(row);
  }
  console.table(rows);
  // every suite's site hook must accept an atlas-only record and offer only finite, usable values
  const { SUITES, loadSuite } = await import('../js/suites/index.js');
  let hooks = 0, offered = 0;
  for (const [name, lat, lon] of [SITES[0], SITES[4], SITES[7]]) {
    const rec = await fetchSite(lat, lon, () => {}, () => {}, { atlasOnly: true });
    for (const meta of SUITES) {
      const suite = await loadSuite(meta.id);
      if (!suite.site) continue;
      hooks++;
      try {
        for (const it of (suite.site(rec) || []).filter(Boolean)) {
          const v = it.value;
          if (v === undefined || v === null) continue; // nothing to offer for this input (e.g. no lending rate published for the country)
          offered++;
          const fine = typeof v === 'number' ? Number.isFinite(v) : typeof v === 'string' ? v.length > 0 : typeof v === 'object';
          if (!fine) bad(name, `suite ${meta.id}: site hook offers an unusable value for "${it.key}"`);
        }
      } catch (e) { bad(name, `suite ${meta.id}: site hook threw ${e.message}`); }
    }
  }
  console.log(`Suite site hooks on atlas-only records: ${hooks} hook calls, ${offered} linked values offered.`);
  const mx = (a) => (a.length ? Math.max(...a) : NaN), mn = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  console.log(`Reference values of ${REF_DATE}. Deviations over ${SITES.length} sites — salinity: mean ${mn(dev.sal).toFixed(2)}, max ${mx(dev.sal).toFixed(2)} g/kg; monthly SST: worst month against the nearer of the two references ${mx(dev.near).toFixed(2)} °C (largest monthly deviation from the marine model year: mean over sites ${mn(dev.sst).toFixed(2)}, worst ${mx(dev.sst).toFixed(2)} °C; median monthly deviation from the SeaDataCloud climatology: mean over sites ${mn(dev.sdc).toFixed(2)} °C); tidal range: mean ${mn(dev.tide).toFixed(0)} %, max ${mx(dev.tide).toFixed(0)} %; wave height: mean ${mn(dev.hs).toFixed(0)} %, max ${mx(dev.hs).toFixed(0)} %${outside.length ? ` — outside the ±40 % target at ${outside.length} site(s): ${outside.join(', ')}` : ''}; depth outside its class and beyond a factor of two at ${deep.length} site(s)${deep.length ? ': ' + deep.join(', ') : ''}.`);
  if (process.argv.includes('--strict')) for (const nm of [...outside, ...deep]) bad(nm, 'target tolerance missed (strict mode)');
  // enclosed and marginal seas against textbook values
  const seas = [['Arabian Gulf', 26.5, 52, 38, 43], ['Red Sea', 22, 38, 38, 40.5], ['Eastern Mediterranean', 34, 30, 38, 39.6], ['Baltic Sea', 57, 19.5, 6, 8], ['Black Sea', 43, 34, 17, 18.6]];
  const { atlasOcean } = await import('../js/data/atlas_lookup.js');
  for (const [nm, la, lo, a, b] of seas) { const o = await atlasOcean(la, lo); console.log(`  ${nm}: ${o ? o.salinity : 'no data'} g/kg (expected ${a}–${b})`); if (!o || o.salinity < a || o.salinity > b) bad(nm, 'salinity outside the expected range'); }
  console.log(fails ? `${fails} check(s) FAILED` : 'All atlas checks passed.');
  process.exit(fails ? 1 : 0);
}

if (process.argv.includes('--refresh')) await refresh(); else await check();
