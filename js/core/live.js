// Live global data connectors. Every request goes straight from the user's browser to a public,
// key-less HTTPS service on the allow-list below; nothing passes through any server of this app, so the
// data stay fresh wherever the app is opened. Responses are treated as untrusted numbers/text only.
import { clamp, mean, quantile } from './num.js';
import { ATLAS } from '../data/atlas.js';

export const SOURCES = [
  { id: 'place', name: 'Place and country', host: 'api.bigdatacloud.net', provider: 'BigDataCloud reverse geocoding', gives: 'Locality, country' },
  { id: 'weather', name: 'Weather and solar', host: 'api.open-meteo.com', provider: 'Open-Meteo forecast (national weather models)', gives: 'Air temperature, wind, humidity, pressure, solar irradiance, land elevation' },
  { id: 'marine', name: 'Sea state, currents and tides', host: 'marine-api.open-meteo.com', provider: 'Open-Meteo marine (wave and ocean models)', gives: 'Sea-surface temperature, waves, ocean currents, sea-level / tide series' },
  { id: 'bathy', name: 'Seabed and terrain', host: 'gis.ngdc.noaa.gov', provider: 'NOAA NCEI global DEM mosaic (best available resolution); fallback SRTM30+ via PacIOOS', gives: 'Bathymetry / topography grid around the site' },
  { id: 'salinity', name: 'Seawater salinity', host: 'erddap.emodnet-physics.eu', provider: 'SeaDataCloud global T–S climatology (EMODnet Physics ERDDAP)', gives: 'Monthly near-surface salinity and temperature climatology' },
  { id: 'economy', name: 'Inflation and interest', host: 'api.worldbank.org', provider: 'World Bank Open Data', gives: 'Consumer-price inflation, lending interest rate' },
  { id: 'fx', name: 'Currency', host: 'open.er-api.com', provider: 'Open exchange-rate API', gives: 'Local currency per US dollar' },
  { id: 'energy', name: 'Grid carbon and renewables', host: 'ourworldindata.org', provider: 'Our World in Data (Ember / Energy Institute series)', gives: 'Carbon intensity of electricity, renewable share of generation' },
  { id: 'climate', name: 'Solar and wind climatology', host: 'power.larc.nasa.gov', provider: 'NASA POWER long-term climatology', gives: 'Monthly and annual solar irradiation, wind speed, air temperature' },
];
export const EVIDENCE_SOURCES = [
  { name: 'OpenAlex', host: 'api.openalex.org', gives: 'Open index of the global research literature' },
  { name: 'Crossref', host: 'api.crossref.org', gives: 'DOI registry of scholarly and technical publications' },
];
const GEOCODE_HOST = 'geocoding-api.open-meteo.com';
const ALLOWED = new Set([...SOURCES.map((s) => s.host), ...EVIDENCE_SOURCES.map((s) => s.host), GEOCODE_HOST, 'pae-paha.pacioos.hawaii.edu']);

/** GET text (CSV) from an allow-listed HTTPS host with a timeout and a size cap. */
export async function getText(url, ms = 20000, maxChars = 400000) {
  const u = new URL(url);
  if (u.protocol !== 'https:' || !ALLOWED.has(u.hostname)) throw new Error('Blocked request to a host that is not on the allow-list.');
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), ms);
  try { const r = await fetch(u.href, { signal: ctl.signal, credentials: 'omit', referrerPolicy: 'no-referrer', cache: 'no-store' }); if (!r.ok) throw new Error('HTTP ' + r.status); return (await r.text()).slice(0, maxChars); }
  finally { clearTimeout(timer); }
}

/** GET JSON from an allow-listed HTTPS host with a timeout. */
export async function getJSON(url, ms = 20000, form = null) {
  const u = new URL(url);
  if (u.protocol !== 'https:' || !ALLOWED.has(u.hostname)) throw new Error('Blocked request to a host that is not on the allow-list.');
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), ms);
  try {
    const opt = { signal: ctl.signal, credentials: 'omit', referrerPolicy: 'no-referrer', cache: 'no-store' };
    const r = await fetch(u.href, form ? { ...opt, method: 'POST', body: new URLSearchParams(form) } : opt);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return await r.json();
  } finally { clearTimeout(timer); }
}
const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : null);
const txt = (x, n = 80) => (typeof x === 'string' ? x.slice(0, n) : '');

/** Place search by name -> [{ name, country, lat, lon }]. */
export async function searchPlace(q) {
  const j = await getJSON(`https://${GEOCODE_HOST}/v1/search?name=${encodeURIComponent(String(q).slice(0, 80))}&count=8&language=en&format=json`);
  return (j.results || []).map((r) => ({ name: txt(r.name), admin: txt(r.admin1), country: txt(r.country), code: txt(r.country_code, 3), lat: num(r.latitude), lon: num(r.longitude) })).filter((r) => r.lat !== null && r.lon !== null);
}

// Indicative industrial electricity tariff ($/kWh) and grid carbon intensity (kgCO₂/kWh) by country.
// These are hand-entered planning defaults — every value is editable in the economics suite. The grid-carbon
// figure is replaced by the Our World in Data series (live, or the copy in the built-in atlas) whenever that is
// available; the tariff stays as the indicative value because the only open global tariff series (World Bank,
// 2019, small-business connection) is older and not an industrial rate — it is used for countries missing here.
// CURRENCY is a quick table for the common cases; the built-in atlas carries the full ISO 4217 list.
const ENERGY = {
  AE: [0.08, 0.40], SA: [0.05, 0.57], QA: [0.04, 0.49], KW: [0.03, 0.57], BH: [0.07, 0.49], OM: [0.06, 0.44], IL: [0.11, 0.45], EG: [0.05, 0.43], MA: [0.10, 0.62], DZ: [0.04, 0.48], TN: [0.09, 0.47], LY: [0.03, 0.55],
  ES: [0.13, 0.17], IT: [0.20, 0.30], GR: [0.16, 0.34], CY: [0.24, 0.60], MT: [0.15, 0.39], TR: [0.10, 0.42], PT: [0.13, 0.15], FR: [0.14, 0.06], GB: [0.24, 0.21], DE: [0.20, 0.36], NL: [0.17, 0.27],
  US: [0.08, 0.37], MX: [0.11, 0.42], CL: [0.13, 0.30], PE: [0.08, 0.20], BR: [0.13, 0.10], AR: [0.07, 0.31], CA: [0.09, 0.12],
  AU: [0.14, 0.55], NZ: [0.11, 0.11], CN: [0.09, 0.56], IN: [0.10, 0.71], PK: [0.14, 0.40], BD: [0.09, 0.57], SG: [0.17, 0.41], JP: [0.19, 0.46], KR: [0.11, 0.43], ID: [0.07, 0.68], MY: [0.09, 0.59], TH: [0.12, 0.47], VN: [0.08, 0.47], PH: [0.15, 0.61],
  ZA: [0.09, 0.71], NG: [0.07, 0.37], GH: [0.13, 0.30], KE: [0.16, 0.09], NA: [0.11, 0.06], TZ: [0.10, 0.34], SN: [0.17, 0.53], DJ: [0.25, 0.50], CV: [0.28, 0.55], IR: [0.02, 0.49], IQ: [0.06, 0.60], JO: [0.11, 0.39], YE: [0.15, 0.60],
};
const CURRENCY = {
  AE: 'AED', SA: 'SAR', QA: 'QAR', KW: 'KWD', BH: 'BHD', OM: 'OMR', IL: 'ILS', EG: 'EGP', MA: 'MAD', DZ: 'DZD', TN: 'TND', LY: 'LYD', ES: 'EUR', IT: 'EUR', GR: 'EUR', CY: 'EUR', MT: 'EUR', PT: 'EUR', FR: 'EUR', DE: 'EUR', NL: 'EUR', TR: 'TRY', GB: 'GBP',
  US: 'USD', MX: 'MXN', CL: 'CLP', PE: 'PEN', BR: 'BRL', AR: 'ARS', CA: 'CAD', AU: 'AUD', NZ: 'NZD', CN: 'CNY', IN: 'INR', PK: 'PKR', BD: 'BDT', SG: 'SGD', JP: 'JPY', KR: 'KRW', ID: 'IDR', MY: 'MYR', TH: 'THB', VN: 'VND', PH: 'PHP',
  ZA: 'ZAR', NG: 'NGN', GH: 'GHS', KE: 'KES', NA: 'NAD', TZ: 'TZS', SN: 'XOF', DJ: 'DJF', CV: 'CVE', IR: 'IRR', IQ: 'IQD', JO: 'JOD', YE: 'YER',
};

const connectors = {
  async place(lat, lon) {
    const j = await getJSON(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`);
    const code = txt(j.countryCode, 2).toUpperCase(), e = ENERGY[code];
    return { meta: { name: txt(j.city || j.locality || j.principalSubdivision || ''), country: txt(j.countryName), countryCode: code },
      data: e ? { electricityPrice: e[0], gridCarbon: e[1], currency: CURRENCY[code] || 'USD' } : { currency: CURRENCY[code] || 'USD' } };
  },
  async weather(lat, lon) {
    const j = await getJSON(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,shortwave_radiation&daily=shortwave_radiation_sum,temperature_2m_max,temperature_2m_min&wind_speed_unit=ms&past_days=7&forecast_days=7&timezone=GMT`);
    const c = j.current || {}, d = j.daily || {};
    const ghi = (d.shortwave_radiation_sum || []).filter((x) => num(x) !== null).map((x) => x / 3.6); // MJ/m² -> kWh/m²
    return { data: { airTemp: num(c.temperature_2m), humidity: num(c.relative_humidity_2m), pressure: num(c.surface_pressure), windSpeed: num(c.wind_speed_10m), windDir: num(c.wind_direction_10m), solar: num(c.shortwave_radiation),
      ghiDaily: ghi.length ? mean(ghi) : null, airTempMax: d.temperature_2m_max?.length ? Math.max(...d.temperature_2m_max.filter((x) => num(x) !== null)) : null, airTempMin: d.temperature_2m_min?.length ? Math.min(...d.temperature_2m_min.filter((x) => num(x) !== null)) : null, elevation: num(j.elevation) } };
  },
  async marine(lat, lon) {
    const j = await getJSON(`https://marine-api.open-meteo.com/v1/marine?latitude=${lat}&longitude=${lon}&current=wave_height,wave_direction,wave_period,sea_surface_temperature,ocean_current_velocity,ocean_current_direction&hourly=sea_level_height_msl,ocean_current_velocity,ocean_current_direction,sea_surface_temperature&past_days=3&forecast_days=5&timezone=GMT&cell_selection=sea`);
    const c = j.current || {}, hh = j.hourly || {}, n = (hh.time || []).length;
    const keep = (a) => (a || []).map((x) => num(x));
    const eta = keep(hh.sea_level_height_msl), sp = keep(hh.ocean_current_velocity).map((x) => (x === null ? null : x / 3.6)), dir = keep(hh.ocean_current_direction), sst = keep(hh.sea_surface_temperature).filter((x) => x !== null);
    const etaOk = eta.filter((x) => x !== null), spOk = sp.filter((x) => x !== null);
    // gaps in the source series are bridged by linear interpolation (never by zeros)
    const bridge = (a) => { const o = a.slice(); let last = -1; for (let i = 0; i < o.length; i++) { if (o[i] === null) continue; if (last < 0) for (let k = 0; k < i; k++) o[k] = o[i]; else for (let k = last + 1; k < i; k++) o[k] = o[last] + ((o[i] - o[last]) * (k - last)) / (i - last); last = i; } if (last >= 0) for (let k = last + 1; k < o.length; k++) o[k] = o[last]; return o.map((x) => x ?? 0); };
    const t0 = Date.parse(String(hh.time?.[0] || '') + 'Z'), nowHour = Number.isFinite(t0) ? clamp((Date.now() - t0) / 3600e3, 0, Math.max(0, n - 1)) : 72;
    const t = Array.from({ length: n }, (_, i) => i);
    const data = { waveHeight: num(c.wave_height), waveDir: num(c.wave_direction), wavePeriod: num(c.wave_period), sst: num(c.sea_surface_temperature) ?? (sst.length ? mean(sst) : null),
      currentSpeed: spOk.length ? mean(spOk) : c.ocean_current_velocity != null ? num(c.ocean_current_velocity) / 3.6 : null, currentMax: spOk.length ? Math.max(...spOk) : null, currentDir: num(c.ocean_current_direction),
      sstMin: sst.length ? Math.min(...sst) : null, sstMax: sst.length ? Math.max(...sst) : null };
    if (etaOk.length > 24) { data.tideRange = quantile(etaOk, 0.98) - quantile(etaOk, 0.02); data.seaLevelMean = mean(etaOk); data.tide = { t, eta: bridge(eta), nowHour }; }
    if (spOk.length > 24) data.currents = { t, speed: bridge(sp), dir: bridge(dir), nowHour, gaps: sp.length - spOk.length };
    if (data.sst === null && !etaOk.length && !spOk.length) throw new Error('No marine data at this point (inland site?)');
    return { data };
  },
  async bathy(lat, lon) {
    const half = 0.12, n = 57; // 57 × 57 samples ≈ the 15-arc-second native resolution of the mosaic over this window
    const lats = Array.from({ length: n }, (_, i) => clamp(lat - half + (2 * half * i) / (n - 1), -89.9, 89.9)), lons = Array.from({ length: n }, (_, i) => clamp(lon - half + (2 * half * i) / (n - 1), -179.9, 179.9));
    let elev;
    try { // primary: one multipoint sample request against the NOAA global DEM mosaic
      const points = lats.flatMap((la) => lons.map((lo) => [+lo.toFixed(5), +la.toFixed(5)]));
      const flat = new Array(n * n).fill(null), CH = 1000, jobs = []; // the service returns at most 1000 samples per request
      for (let o = 0; o < points.length; o += CH) jobs.push(getJSON('https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/getSamples', 12000,
        { geometry: JSON.stringify({ points: points.slice(o, o + CH), spatialReference: { wkid: 4326 } }), geometryType: 'esriGeometryMultipoint', returnFirstValueOnly: 'true', f: 'json' })
        .then((j) => { for (const sm of j.samples || []) { const v = parseFloat(sm.value), id = o + sm.locationId; if (Number.isInteger(sm.locationId) && id >= 0 && id < n * n && Number.isFinite(v) && Math.abs(v) < 12000) flat[id] = v; } }));
      await Promise.all(jobs);
      if (flat.filter((v) => v !== null).length < 0.6 * n * n) throw new Error('Incomplete relief grid');
      elev = lats.map((_, a) => lons.map((_, b) => flat[a * n + b] ?? 0));
    } catch { // fallback: SRTM30+ 1 km relief resampled onto the same grid
      const j = await getJSON(`https://pae-paha.pacioos.hawaii.edu/erddap/griddap/srtm30plus_v11_bathy.json?elev%5B(${lats[0].toFixed(4)}):1:(${lats[n - 1].toFixed(4)})%5D%5B(${lons[0].toFixed(4)}):1:(${lons[n - 1].toFixed(4)})%5D`, 20000);
      const rows = (j.table?.rows || []).filter((r) => num(r[0]) !== null && num(r[1]) !== null && num(r[2]) !== null);
      if (rows.length < 9) throw new Error('No relief data returned');
      elev = lats.map((la) => lons.map((lo) => { let best = rows[0], bd = Infinity; for (const r of rows) { const d = (r[0] - la) ** 2 + (r[1] - lo) ** 2; if (d < bd) { bd = d; best = r; } } return best[2]; }));
    }
    const mid = (n - 1) / 2, here = elev[mid][mid], flat = elev.flat();
    return { data: { bathy: { lat: lats, lon: lons, elev }, depth: here < 0 ? -here : 0, elevationRelief: here, maxDepthNearby: Math.max(0, -Math.min(...flat)), seaFraction: flat.filter((z) => z < 0).length / flat.length } };
  },
  async salinity(lat, lon) {
    const g = (v) => Math.round(v * 4) / 4, la = clamp(g(lat), -79.5, 79.5), lo = clamp(g(lon), -179.5, 179.25);
    const q = `%5B0:1:11%5D%5B0%5D%5B(${la - 0.25}):1:(${la + 0.25})%5D%5B(${lo - 0.25}):1:(${lo + 0.25})%5D`;
    const url = `https://erddap.emodnet-physics.eu/erddap/griddap/SDC_GLO_CLIM_TS_V2_1.json?Salinity${q},Temperature${q}`;
    const j = await getJSON(url, 20000); // slow server: its answer is cached on the device for 90 days, so this wait happens once per place
    const rows = (j.table?.rows || []).filter((r) => num(r[4]) !== null && r[4] > 0 && r[4] < 60);
    if (!rows.length) throw new Error('No ocean cell near this point');
    const byMonth = Array.from({ length: 12 }, () => ({ s: [], t: [] }));
    for (const r of rows) { const m = new Date(r[0]).getUTCMonth(); if (m >= 0) { byMonth[m].s.push(r[4]); if (num(r[5]) !== null) byMonth[m].t.push(r[5]); } }
    const sM = byMonth.map((b) => (b.s.length ? mean(b.s) : null)), tM = byMonth.map((b) => (b.t.length ? mean(b.t) : null)), sOk = sM.filter((x) => x !== null);
    return { data: { salinity: quantile(sOk, 0.5), salinityMin: Math.min(...sOk), salinityMax: Math.max(...sOk), salinityMonthly: sM.map((x) => x ?? 0), sstMonthly: tM.map((x) => x ?? 0) } };
  },
  async economy(lat, lon, site) {
    const code = site.countryCode;
    if (!code) throw new Error('Country unknown');
    const one = async (ind) => { const j = await getJSON(`https://api.worldbank.org/v2/country/${encodeURIComponent(code)}/indicator/${ind}?format=json&mrnev=1`); const r = j?.[1]?.[0]; return r && num(r.value) !== null ? { value: r.value, year: txt(String(r.date), 6), iso3: txt(r.countryiso3code, 3) } : null; };
    // key -> World Bank indicator
    const IND = { inflation: 'FP.CPI.TOTL.ZG', lendingRate: 'FR.INR.LEND', waterStress: 'ER.H2O.FWST.ZS', freshwaterPerCapita: 'ER.H2O.INTR.PC', renewableElectricity: 'EG.ELC.RNEW.ZS', tariffWB: 'IC.ELC.PRI.KH.DB1619', gdpPerCapita: 'NY.GDP.PCAP.CD', safeWaterAccess: 'SH.H2O.SMDW.ZS' };
    const keys = Object.keys(IND), res = await Promise.allSettled(keys.map((k) => one(IND[k]))), data = {};
    res.forEach((r, i) => { const v = r.status === 'fulfilled' ? r.value : null; if (!v) return; data[keys[i]] = v.value; data[keys[i] + 'Year'] = v.year; if (v.iso3) data.iso3 = v.iso3; });
    if (data.tariffWB != null) { data.electricityPriceWB = data.tariffWB / 100; delete data.tariffWB; data.electricityPriceWBYear = data.tariffWBYear; delete data.tariffWBYear; }
    if (!Object.keys(data).length) throw new Error('No indicators published for this country');
    return { data };
  },
  async energy(lat, lon, site) {
    const iso3 = site.data?.iso3, name = site.country;
    if (!iso3 && !name) throw new Error('Country unknown');
    // The chart addresses are requested directly: the short aliases answer with a redirect that browsers refuse cross-origin.
    const owid = (direct, alias) => getText('https://ourworldindata.org/grapher/' + direct).catch(() => getText('https://ourworldindata.org/grapher/' + alias));
    const pick = (csv) => { for (const line of csv.split('\n')) { const c = line.split(','); if ((iso3 && c[1] === iso3) || (!iso3 && c[0] === name)) { const v = parseFloat(c[3]); if (Number.isFinite(v)) return { value: v, year: txt(c[4] || c[2], 6) }; } } return null; };
    const [ci, rn] = await Promise.allSettled([owid('electricity-mix.csv?frequency=annual&metric=carbon_intensity&source=total&csvType=filtered&time=latest', 'carbon-intensity-electricity.csv?csvType=filtered&time=latest'), owid('electricity-mix.csv?frequency=annual&metric=share_of_generation&source=renewables&csvType=filtered&time=latest', 'share-electricity-renewables.csv?csvType=filtered&time=latest')]);
    const data = {}, a = ci.status === 'fulfilled' ? pick(ci.value) : null, b = rn.status === 'fulfilled' ? pick(rn.value) : null;
    if (a && a.value >= 0 && a.value < 1500) { data.gridCarbon = a.value / 1000; data.gridCarbonYear = a.year; data.gridCarbonLive = true; }
    if (b && b.value >= 0 && b.value <= 100) { data.renewableShare = b.value; data.renewableShareYear = b.year; }
    if (!Object.keys(data).length) throw new Error('Country not listed');
    return { data };
  },
  async climate(lat, lon) {
    const j = await getJSON(`https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=ALLSKY_SFC_SW_DWN,WS10M,T2M&community=RE&longitude=${lon.toFixed(3)}&latitude=${lat.toFixed(3)}&format=JSON`, 25000);
    const p = j?.properties?.parameter || {}, M = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
    const series = (o) => (o ? M.map((m) => (num(o[m]) !== null && o[m] > -900 ? o[m] : 0)) : null), ann = (o) => (o && num(o.ANN) !== null && o.ANN > -900 ? o.ANN : null);
    const data = { ghiAnnual: ann(p.ALLSKY_SFC_SW_DWN), ghiMonthly: series(p.ALLSKY_SFC_SW_DWN), windAnnual: ann(p.WS10M), windMonthly: series(p.WS10M), airTempAnnual: ann(p.T2M), airTempMonthly: series(p.T2M) };
    if (data.ghiAnnual === null) throw new Error('No climatology for this point');
    return { data };
  },
  async fx(lat, lon, site) {
    let cur = CURRENCY[site.countryCode] || site.data?.currency || 'USD';
    if (site.countryCode && !CURRENCY[site.countryCode]) { // not in the short table above: the bundled ISO 4217 list knows every country
      try { cur = (await (await loadAtlas()).atlasNational(site.countryCode))?.currency || cur; } catch { /* keep the default */ }
    }
    const j = await getJSON('https://open.er-api.com/v6/latest/USD');
    const rate = num(j?.rates?.[cur]);
    if (rate === null) throw new Error('Currency not listed');
    return { data: { currency: cur, fxPerUSD: rate, fxDate: txt(j.time_last_update_utc, 40) } };
  },
};

// ---- response cache ---------------------------------------------------------------------------------
// Slow-changing answers (relief, climatologies, national indicators) are kept on the device so that a
// repeat fetch of the same place is instant; fast-changing ones (weather, sea state) are kept briefly.
const CACHE_KEY = 'brinelab.live.v2', H = 3600e3;
const TTL = { place: 30 * 24 * H, weather: 0.5 * H, marine: 0.5 * H, bathy: 90 * 24 * H, salinity: 90 * 24 * H, economy: 7 * 24 * H, fx: 12 * H, energy: 7 * 24 * H, climate: 90 * 24 * H };
let memCache = null;
function cacheAll() {
  if (memCache) return memCache;
  try { memCache = JSON.parse(localStorage.getItem(CACHE_KEY) || '{}') || {}; } catch { memCache = {}; }
  return memCache;
}
function cacheGet(key, ttl) { const e = cacheAll()[key]; return e && Date.now() - e.t < ttl ? e : null; }
function cachePut(key, v) {
  const all = cacheAll(); all[key] = { t: Date.now(), v };
  const keys = Object.keys(all);
  if (keys.length > 160) keys.sort((a, b) => all[a].t - all[b].t).slice(0, keys.length - 120).forEach((k) => delete all[k]);
  try { localStorage.setItem(CACHE_KEY, JSON.stringify(all)); }
  catch { try { for (const k of Object.keys(all)) if (k.startsWith('bathy')) delete all[k]; localStorage.setItem(CACHE_KEY, JSON.stringify(all)); } catch { /* storage unavailable: memory cache only */ } }
}
/** Cache key of a connector: grid-cell for site data, country for national data. */
function keyOf(id, lat, lon, site) {
  if (id === 'economy' || id === 'energy') return site.countryCode ? `${id}:${site.countryCode}` : null;
  if (id === 'fx') return `fx:${site.data?.currency || site.countryCode || 'USD'}`;
  const r = id === 'salinity' ? 4 : id === 'climate' ? 2 : id === 'bathy' ? 250 : id === 'place' ? 100 : 20; // cells per degree
  return `${id}:${Math.round(lat * r)}:${Math.round(lon * r)}`;
}

// ---- built-in world atlas -----------------------------------------------------------------------------
// When a live source cannot be reached (no connection, a blocked host, a service that is down) the fields
// it would have supplied are answered from data bundled with the app: js/data/atlas_*.js, built from open
// global data sets by tools/atlas_build.py. Those modules are large, so they are imported only when needed.
let atlasLib = null;
/** The look-up module of the built-in atlas (js/data/atlas_lookup.js), loaded on first use. */
export const loadAtlas = () => (atlasLib ??= import('../data/atlas_lookup.js').catch((e) => { atlasLib = null; throw e; }));

/** Fields by which a source counts as "answered by the atlas", and the atlas data set that stands in for it. */
const ATLAS_SOURCE = {
  place: ['nations', ['country']], economy: ['nations', ['inflation', 'lendingRate', 'gdpPerCapita', 'waterStress', 'renewableElectricity', 'electricityPriceWB', 'freshwaterPerCapita', 'safeWaterAccess']],
  fx: ['fx', ['fxPerUSD']], energy: ['nations', ['gridCarbon', 'renewableShare']], weather: ['climate', ['airTemp', 'windSpeed', 'ghiDaily', 'elevation']],
  marine: ['coast', ['tideRange', 'waveHeight', 'currentSpeed', 'tide', 'sst']], bathy: ['relief', ['bathy', 'depth']], salinity: ['ocean', ['salinity', 'salinityMonthly']], climate: ['climate', ['ghiAnnual', 'windAnnual', 'airTempAnnual']],
};
/** Plain-language name of every site-data field the atlas can supply (used for labels and the notice on the site page). */
export const ATLAS_LABELS = {
  country: 'country', currency: 'currency', electricityPrice: 'electricity price', inflation: 'inflation', lendingRate: 'lending rate', gdpPerCapita: 'GDP per capita', waterStress: 'water stress', renewableElectricity: 'renewable electricity (World Bank)',
  electricityPriceWB: 'business electricity tariff', freshwaterPerCapita: 'freshwater per capita', safeWaterAccess: 'safe water access', fxPerUSD: 'exchange rate', gridCarbon: 'grid carbon', renewableShare: 'renewable share',
  salinity: 'salinity', salinityMonthly: 'monthly salinity', sstMonthly: 'monthly sea temperature', sst: 'sea temperature', bathy: 'seabed and terrain grid', depth: 'water depth', maxDepthNearby: 'deepest point nearby', elevation: 'land elevation',
  tideRange: 'tidal range', tide: 'tide series', currentSpeed: 'mean current', currentMax: 'peak current', currentDir: 'current direction', waveHeight: 'wave height', wavePeriod: 'wave period', waveDir: 'wave direction',
  ghiAnnual: 'solar resource', ghiDaily: 'solar resource', windAnnual: 'long-term wind', airTempAnnual: 'long-term air temperature', airTemp: 'air temperature', windSpeed: 'wind speed',
};
const month = () => new Date().getUTCMonth();

/**
 * Fill every site-data field that is still missing from the built-in atlas. Live values are never overwritten.
 *   site    { lat, lon, data, country, countryCode, name }  — changed in place
 *   status  per-source status of the live fetch; sources answered by the atlas get { atlas: true, message: 'built-in atlas (yyyy-mm)' }
 *   ctx     { mark: Map(field -> atlas data set), notes: {}, done: Set(source ids that have settled), vintage: {} }
 * A field is filled only once every live source that could still deliver it has settled (ctx.done; default: all).
 * Afterwards site.data.atlasFields lists the fields taken from the atlas, site.data.atlasVintage the dates of the
 * bundled data sets in use and site.data.atlasNotes any caveat per field.
 */
export async function atlasFill(site, status = {}, ctx = {}) {
  const d = (site.data ??= {}), { lat, lon } = site;
  const mark = (ctx.mark ??= new Map()), notes = (ctx.notes ??= {}), done = (ctx.done ??= new Set(SOURCES.map((x) => x.id))), vintage = (ctx.vintage ??= {});
  const ready = (...ids) => ids.every((id) => done.has(id)), missing = (...keys) => keys.some((k) => d[k] == null);
  const put = (set, k, v, note) => { if (v == null || (d[k] != null && !mark.has(k))) return; d[k] = v; mark.set(k, set); if (note) notes[k] = note; else delete notes[k]; };
  let A = null;
  try { A = await loadAtlas(); } catch { /* the atlas modules could not be loaded: the small table shipped with the core is used below */ }
  const part = async (fn) => { try { await fn(); } catch { /* this data set is unavailable; the others still apply */ } };
  if (A) {
    await part(async () => { // country and national figures
      if (!ready('place')) return;
      if (!site.countryCode) { const p = await A.atlasPlace(lat, lon); if (p) { site.country = p.country; site.countryCode = p.countryCode; mark.set('country', 'nations'); if (p.offshoreKm > 0) notes.country = `nearest country, ${p.offshoreKm} km from its outline`; } }
      if (!site.countryCode) return;
      const eco = ['inflation', 'lendingRate', 'gdpPerCapita', 'waterStress', 'renewableElectricity', 'freshwaterPerCapita', 'safeWaterAccess', 'electricityPriceWB'];
      const wantEco = ready('economy') && missing(...eco), wantEn = ready('energy') && (!d.gridCarbonLive || missing('renewableShare')), wantFx = ready('fx') && missing('fxPerUSD');
      if (!wantEco && !wantEn && !wantFx && !missing('currency', 'electricityPrice')) return;
      const n = await A.atlasNational(site.countryCode);
      if (!n) return;
      if (!site.country) site.country = n.countryName;
      if (n.currency && (d.currency == null || (d.currency === 'USD' && !CURRENCY[site.countryCode] && !status.fx?.ok))) { d.currency = n.currency; mark.set('currency', 'nations'); }
      if (wantEco) { for (const k of eco) { put('nations', k, n[k]); if (mark.has(k)) d[k + 'Year'] = n[k + 'Year']; } d.iso3 ??= n.iso3; }
      if (wantEn) {
        if (!d.gridCarbonLive && n.gridCarbon != null) { d.gridCarbon = n.gridCarbon; d.gridCarbonYear = n.gridCarbonYear; mark.set('gridCarbon', 'nations'); }
        put('nations', 'renewableShare', n.renewableShare); if (mark.has('renewableShare')) d.renewableShareYear = n.renewableShareYear;
      }
      if (wantFx) { const r = d.currency === n.currency ? n : await A.atlasRate(d.currency); if (r?.fxPerUSD > 0) { put('fx', 'fxPerUSD', r.fxPerUSD); if (mark.has('fxPerUSD')) d.fxDate = r.fxDate; } }
      const e = ENERGY[site.countryCode];
      if (e) { put('nations', 'electricityPrice', e[0], 'indicative industrial tariff (hand-entered planning default)'); if (ready('energy') && d.gridCarbon == null) put('nations', 'gridCarbon', e[1], 'indicative planning default'); }
      else if (ready('economy') && mark.has('electricityPriceWB')) put('nations', 'electricityPrice', d.electricityPriceWB, `World Bank business tariff, ${d.electricityPriceWBYear}`);
    });
    await part(async () => { // sea-surface salinity and temperature
      const wantS = ready('salinity') && missing('salinity', 'salinityMonthly', 'sstMonthly'), wantT = ready('salinity', 'marine') && missing('sst');
      if (!wantS && !wantT) return;
      const o = await A.atlasOcean(lat, lon);
      if (!o) return;
      const note = o.oceanCellKm > 0 ? `nearest sea cell, ${o.oceanCellKm} km away` : '';
      if (wantS) { put('ocean', 'salinity', o.salinity, note); if (mark.has('salinity')) { d.salinityMin = o.salinityMin; d.salinityMax = o.salinityMax; } put('ocean', 'salinityMonthly', o.salinityMonthly); put('ocean', 'sstMonthly', o.sstMonthly); }
      if (wantT && (d.sstMonthly || o.sstMonthly)) { const own = mark.has('sstMonthly') || d.sstMonthly == null, m = own ? o.sstMonthly : d.sstMonthly; put(own ? 'ocean' : 'derived', 'sst', m[month()], 'climatological mean for this month'); if (mark.has('sst')) { d.sstMin = Math.min(...m); d.sstMax = Math.max(...m); } }
    });
    await part(async () => { // seabed and terrain
      const wantB = ready('bathy') && missing('bathy', 'depth'), wantE = ready('bathy', 'weather') && missing('elevation');
      if (!wantB && !wantE) return;
      const r = await A.atlasRelief(lat, lon);
      if (wantB) {
        put('relief', 'bathy', r.bathy, 'coarse grid (about 5 km at the coast)'); put('relief', 'depth', r.depth, r.depthEstimated || '');
        if (mark.has('depth')) { d.elevationRelief = r.elevationRelief; d.seaFraction = r.seaFraction; d.depthEstimated = r.depthEstimated || 'built-in atlas relief'; }
        put('relief', 'maxDepthNearby', r.maxDepthNearby, 'within about 50 km');
      }
      if (wantE) put('relief', 'elevation', r.elevation);
    });
    await part(async () => { // tides, currents, waves
      if (!ready('marine') || !missing('tideRange', 'tide', 'currentSpeed', 'waveHeight', 'wavePeriod')) return;
      const c = await A.atlasCoast(lat, lon);
      if (!c) return;
      const near = c.coastPointKm != null ? `, model point ${c.coastPointKm} km away` : '';
      put('coast', 'tideRange', c.tideRange, `typical range ${c.seaPeriod}${near}`);
      if (mark.has('tideRange')) { d.tideSpring = c.tideSpring; d.tideNeap = c.tideNeap; }
      if (d.tide == null || mark.has('tide')) { put('coast', 'tide', c.tide, 'synthetic: harmonic prediction from five tidal constituents'); if (mark.has('tide')) d.seaLevelMean = 0; }
      put('coast', 'currentSpeed', c.currentSpeed, `mean ${c.seaPeriod}`); put('coast', 'currentMax', c.currentMax, `99th percentile ${c.seaPeriod}`); put('coast', 'currentDir', c.currentDir, 'predominant direction');
      const wn = c.wavePointKm != null ? `, model point ${c.wavePointKm} km away` : '';
      put('coast', 'waveHeight', c.waveHeight, `annual mean ${c.wavePeriodOfRecord.slice(0, 4)}${wn}`);
      if (mark.has('waveHeight')) { d.waveHeightP95 = c.waveHeightP95; d.waveHeightMax = c.waveHeightMax; d.waveHeightMonthly = c.waveHeightMonthly; d.wavePeriodStorm = c.wavePeriodStorm; }
      put('coast', 'wavePeriod', c.wavePeriod, 'annual mean'); put('coast', 'waveDir', c.waveDir, 'predominant direction');
    });
    await part(async () => { // long-term solar, wind, air temperature
      const wantC = ready('climate') && missing('ghiAnnual', 'windAnnual', 'airTempAnnual'), wantW = ready('climate', 'weather') && missing('airTemp', 'windSpeed', 'ghiDaily');
      if (!wantC && !wantW) return;
      const c = await A.atlasClimate(lat, lon);
      if (!c) return;
      if (wantC) for (const k of ['ghiAnnual', 'ghiMonthly', 'windAnnual', 'windMonthly', 'airTempAnnual', 'airTempMonthly']) put('climate', k, c[k], k.endsWith('Annual') ? `long-term mean ${c.climatePeriod}` : '');
      if (wantW) {
        const own = (k) => (mark.has(k) || d[k] == null ? 'climate' : 'derived'), at = d.airTempMonthly || c.airTempMonthly, wm = d.windMonthly || c.windMonthly;
        if (at) put(own('airTempMonthly'), 'airTemp', at[month()], 'long-term mean for this month'); if (wm) put(own('windMonthly'), 'windSpeed', wm[month()], 'long-term mean for this month');
        put(own('ghiAnnual'), 'ghiDaily', d.ghiAnnual ?? c.ghiAnnual, 'long-term mean');
      }
    });
  }
  // last resort (also when the atlas modules are unavailable): the small table shipped with the core
  if (ready('salinity') && d.salinity == null) {
    const at = A ? null : atlasSite(lat, lon), reg = regionalSalinity(lat, lon), enclosed = reg !== 35.5 && reg !== 34.3;
    put('core', 'salinity', at && !enclosed ? at.salinity : reg, at && !enclosed ? 'coarse 4° table' : 'regional estimate');
    if (at) { put('core', 'salinityMonthly', at.salinityMonthly); put('core', 'sstMonthly', at.sstMonthly); if (ready('marine')) put('core', 'sst', at.sst, 'coarse 4° table'); }
    else if (d.sstMonthly && d.salinityMonthly == null) put('core', 'salinityMonthly', new Array(12).fill(d.salinity), 'regional estimate, no seasonal cycle');
  }
  if (ready('climate', 'weather') && d.ghiAnnual == null && d.ghiDaily == null) put('core', 'ghiDaily', solarEstimate(lat), 'estimate from latitude');
  // dates of the bundled data, and the per-source status
  const sets = new Set(mark.values());
  if (A) for (const s of sets) if (s !== 'core' && s !== 'derived' && vintage[s] === undefined) Object.assign(vintage, await A.atlasVintage([s === 'fx' ? 'nations' : s]).catch(() => ({})));
  for (const k of [...mark.keys()]) if (mark.get(k) === 'derived') mark.delete(k); // computed from live climatology: not atlas data
  for (const [id, [set, keys]] of Object.entries(ATLAS_SOURCE)) {
    const st = status[id];
    if (!st || st.ok || st.atlas || !keys.some((k) => mark.has(k))) continue;
    status[id] = { ok: false, atlas: true, message: `built-in atlas (${vintage[set] || vintage.nations || 'bundled'})`, reason: st.message, at: st.at };
    ctx.onStatus?.(id, 'atlas', status[id].message);
  }
  atlasStamp(site, ctx);
  return site;
}
/** Write the atlas book-keeping (atlasFields, atlasNotes, atlasVintage and the older flags) into site.data. */
function atlasStamp(site, ctx) {
  const d = site.data, mark = ctx.mark, sets = new Set(mark.values());
  d.atlasFields = [...mark.keys()].sort();
  d.atlasNotes = Object.fromEntries(Object.entries(ctx.notes).filter(([k]) => mark.has(k)));
  const v = Object.fromEntries(Object.entries(ctx.vintage || {}).filter(([k]) => sets.has(k) || (k === 'seaPeriod' || k === 'wavePeriod' ? sets.has('coast') : k === 'climatePeriod' ? sets.has('climate') : false)));
  if (Object.keys(v).length) d.atlasVintage = v; else delete d.atlasVintage;
  d.atlas = mark.size > 0; d.salinityEstimated = mark.has('salinity'); d.sstEstimated = mark.has('sst'); d.solarEstimated = mark.get('ghiDaily') === 'core';
  if (!mark.has('depth')) delete d.depthEstimated;
}
/**
 * Merge a fresh fetch into the stored data of the same place: fresh live values replace anything, but a fresh
 * atlas value never replaces a stored live one. Returns the merged data object (with a consistent atlasFields).
 */
export function mergeSiteData(old = {}, fresh = {}) {
  const oldAtlas = new Set(old.atlasFields || []), newAtlas = new Set(fresh.atlasFields || []), out = { ...old }, fields = new Set();
  for (const k of oldAtlas) if (old[k] != null) fields.add(k);
  for (const [k, v] of Object.entries(fresh)) {
    if (v === null || v === undefined || k.startsWith('atlas')) continue;
    if (newAtlas.has(k) && old[k] != null && !oldAtlas.has(k)) continue; // stored live value stays
    out[k] = v; if (newAtlas.has(k)) fields.add(k); else fields.delete(k);
  }
  out.atlasFields = [...fields].sort();
  out.atlasNotes = Object.fromEntries(Object.entries({ ...(old.atlasNotes || {}), ...(fresh.atlasNotes || {}) }).filter(([k]) => fields.has(k)));
  out.atlasVintage = { ...(old.atlasVintage || {}), ...(fresh.atlasVintage || {}) }; if (!fields.size) delete out.atlasVintage;
  out.atlas = fields.size > 0; out.salinityEstimated = fields.has('salinity'); out.sstEstimated = fields.has('sst'); out.solarEstimated = fields.has('ghiDaily') && fresh.solarEstimated === true;
  if (!fields.has('depth')) delete out.depthEstimated;
  return out;
}

/**
 * Pull everything for one location, as fast as the sources allow:
 *  - every connector starts at once (national data start the moment the country is known);
 *  - answers already on the device are used immediately;
 *  - onData(site) is called after each answer so the page fills in progressively instead of waiting for the slowest source;
 *  - whatever a source fails to deliver is filled from the built-in atlas (atlasFill), field by field, and listed in
 *    data.atlasFields; a live answer always replaces an atlas value.
 * opt: { fresh: ignore the device cache, atlasOnly: do not use the network at all }.
 * Returns { meta, data, status: { id: { ok, message, at, cached, atlas } } }.
 */
export async function fetchSite(lat, lon, onStatus = () => {}, onData = () => {}, opt = {}) {
  const { fresh = false } = opt;
  lat = clamp(+lat, -90, 90); lon = ((((+lon + 180) % 360) + 360) % 360) - 180;
  const site = { lat, lon, data: {}, name: '', country: '', countryCode: '' }, status = {};
  const ctx = { mark: new Map(), notes: {}, done: new Set(), vintage: {}, onStatus };
  const snapshot = () => { atlasStamp(site, ctx); return { ...site, data: { ...site.data }, status: { ...status }, fetchedAt: new Date().toISOString() }; };
  const absorb = (r) => { // live (or stored live) answer: it replaces any atlas value of the same field
    for (const [k, v] of Object.entries(r.data || {})) if (v !== null && v !== undefined) { site.data[k] = v; ctx.mark.delete(k); }
    if (r.meta) { Object.assign(site, r.meta); if (r.meta.countryCode) ctx.mark.delete('country'); }
  };
  const run = async (id) => {
    const key = keyOf(id, lat, lon, site), hit = !fresh && key ? cacheGet(key, TTL[id]) : null;
    if (hit) { absorb(hit.v); ctx.done.add(id); status[id] = { ok: true, message: 'Live', at: new Date(hit.t).toISOString(), cached: true }; onStatus(id, 'ok', 'Live'); finish(); onData(snapshot()); return; }
    onStatus(id, 'loading');
    try {
      const r = await connectors[id](lat, lon, site);
      absorb(r);
      if (key) cachePut(key, { data: r.data, meta: r.meta });
      status[id] = { ok: true, message: 'Live', at: new Date().toISOString() };
    } catch (e) {
      const stale = key ? cacheAll()[key] : null; // an older stored answer beats no answer
      if (stale) { absorb(stale.v); status[id] = { ok: true, message: 'Stored copy', at: new Date(stale.t).toISOString(), cached: true }; }
      else status[id] = { ok: false, message: e.name === 'AbortError' ? 'Timed out' : txt(e.message || 'Unavailable', 90), at: new Date().toISOString() };
    }
    ctx.done.add(id);
    onStatus(id, status[id].ok ? 'ok' : 'fail', status[id].message);
    if (!status[id].ok) await atlasFill(site, status, ctx); // stand in for this source straight away (the country, in particular, unlocks the national sources)
    finish(); onData(snapshot());
  };
  const finish = () => { // values derived from other fields
    const d = site.data, mk = ctx.mark, derive = (k, from, v) => { d[k] = v; if (mk.has(from)) mk.set(k, mk.get(from)); else mk.delete(k); };
    if (d.sst == null && d.sstMonthly && d.sstMonthly[month()]) derive('sst', 'sstMonthly', d.sstMonthly[month()]);
    if (d.ghiAnnual != null && (d.ghiDaily == null || mk.has('ghiDaily') || !mk.has('ghiAnnual'))) derive('ghiDaily', 'ghiAnnual', d.ghiAnnual); // the long-term mean is the better design basis than this week's weather
    if (d.electricityPrice == null && d.electricityPriceWB != null) derive('electricityPrice', 'electricityPriceWB', d.electricityPriceWB);
  };
  if (opt.atlasOnly) { // no network at all: answer from the built-in atlas
    for (const src of SOURCES) { status[src.id] = { ok: false, message: 'No connection', at: new Date().toISOString() }; ctx.done.add(src.id); onStatus(src.id, 'fail', 'No connection'); }
    await atlasFill(site, status, ctx); finish();
    return snapshot();
  }
  const national = run('place').then(() => Promise.all([run('fx'), run('economy').then(() => run('energy'))])); // these need the country
  await Promise.all([national, ...['weather', 'marine', 'bathy', 'salinity', 'climate'].map(run)]);
  await atlasFill(site, status, ctx); finish();
  return snapshot();
}

/**
 * Coarse stand-by value for a point: nearest ocean cell of the 4° climatology shipped with the core (searched outwards
 * up to three cells). Returns null far inland. Used only if the modules of the full built-in atlas cannot be loaded.
 */
export function atlasSite(lat, lon) {
  const A = ATLAS, i0 = Math.round((lat - A.lat0) / A.step), j0 = Math.round((((lon - A.lon0) % 360) + 360) % 360 / A.step);
  let best = null;
  for (let r = 0; r <= 3 && !best; r++) for (let di = -r; di <= r; di++) for (let dj = -r; dj <= r; dj++) {
    if (Math.max(Math.abs(di), Math.abs(dj)) !== r) continue;
    const i = i0 + di, j = (((j0 + dj) % A.nlon) + A.nlon) % A.nlon;
    if (i < 0 || i >= A.nlat) continue;
    const k = i * A.nlon + j;
    if (A.sal[k] < 0) continue;
    const d = Math.hypot(di, dj * Math.cos((lat * Math.PI) / 180));
    if (!best || d < best.d) best = { d, k };
  }
  if (!best) return null;
  const k = best.k, t0 = A.t0[k] / 10, ta = A.ta[k] / 10, tp = A.tp[k] / 10, month = new Date().getUTCMonth();
  const sstMonthly = Array.from({ length: 12 }, (_, m) => +(t0 + ta * Math.cos((2 * Math.PI * (m - tp)) / 12)).toFixed(2));
  return { salinity: A.sal[k] / 10, sst: sstMonthly[month], sstMin: +(t0 - ta).toFixed(2), sstMax: +(t0 + ta).toFixed(2), sstMonthly, salinityMonthly: new Array(12).fill(A.sal[k] / 10) };
}
/** Clear-sky-based estimate of the long-term mean solar irradiation (kWh/m²·d) from latitude alone (used outside the atlas grid, beyond 60° S / 80° N). */
export const solarEstimate = (lat) => +clamp(6.4 * Math.cos((Math.abs(lat) * Math.PI) / 180) ** 1.15 + 0.6, 1.5, 6.8).toFixed(2);

/** Regional salinity estimate, used only where neither the live service nor the built-in atlas has a value (e.g. the Caspian Sea, far inland). */
export function regionalSalinity(lat, lon) {
  const box = (a, b, c, d) => lat >= a && lat <= b && lon >= c && lon <= d;
  if (box(23.5, 30.5, 47.5, 56.5)) return 42;
  if (box(12, 30, 32, 43.5)) return 40;
  if (box(30, 46, -6, 36.5)) return 38.3;
  if (box(53, 66, 9, 30)) return 7.5;
  if (box(40.5, 47, 27, 42)) return 18;
  if (box(36, 47, 46.5, 55)) return 12.5;
  return Math.abs(lat) < 35 ? 35.5 : 34.3;
}

/**
 * Live literature evidence for a decision topic: recent, well-cited peer-reviewed work from OpenAlex,
 * with Crossref as a fallback. Returns [{ title, year, venue, cited, url }] (plain text only).
 */
export async function evidence(topic, n = 5) {
  const q = encodeURIComponent(String(topic).slice(0, 160));
  const year = new Date().getUTCFullYear() - 6;
  const safeUrl = (doi) => { const d = txt(String(doi || '').replace(/^https?:\/\/(dx\.)?doi\.org\//i, ''), 200); return /^10\.\d{4,9}\/\S+$/.test(d) ? 'https://doi.org/' + encodeURI(d) : ''; };
  try {
    const j = await getJSON(`https://api.openalex.org/works?filter=title_and_abstract.search:${q},from_publication_date:${year}-01-01,type:article|review,cited_by_count:%3E4&sort=relevance_score:desc&per-page=${n}&select=title,publication_year,doi,cited_by_count,primary_location`, 15000);
    const out = (j.results || []).map((w) => ({ title: txt(w.title, 260), year: num(w.publication_year), venue: txt(w.primary_location?.source?.display_name, 120), cited: num(w.cited_by_count), url: safeUrl(w.doi) })).filter((w) => w.title);
    if (out.length) return { source: 'OpenAlex', items: out };
  } catch { /* fall through to Crossref */ }
  const j = await getJSON(`https://api.crossref.org/works?query=${q}&rows=${n}&select=title,DOI,issued,container-title,is-referenced-by-count&filter=from-pub-date:${year}`, 15000);
  return { source: 'Crossref', items: (j.message?.items || []).map((w) => ({ title: txt(w.title?.[0], 260), year: num(w.issued?.['date-parts']?.[0]?.[0]), venue: txt(w['container-title']?.[0], 120), cited: num(w['is-referenced-by-count']), url: safeUrl(w.DOI) })).filter((w) => w.title) };
}
