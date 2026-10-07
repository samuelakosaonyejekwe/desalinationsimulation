// Non-suite pages: home, case & feed water, global site data, data portal, integrated run, app & offline.
import { h, clear, btn, kpiGrid, dataTable, toast, badge, fieldRow, importBtn, help, fill } from '../core/ui.js';
import { store } from '../core/store.js';
import { SUITES, CHAIN, byId, loadSuite, downstream } from '../suites/index.js';
import { runSuite, applyLinks, linkItems, allFields, setInputValue } from '../core/suiteview.js';
import { readFiles, geometryCard, generatorPanel, formatCatalogue, attachGeometry, derive, ACCEPT } from '../core/geomview.js';
import { geometryLinks } from '../core/geomlinks.js';
import { SUITE_GEOMETRY, formatOf } from '../core/geom.js';
import { fetchSite, searchPlace, SOURCES } from '../core/live.js';
import { plotCard } from '../core/plot.js';
import { fmt } from '../core/num.js';
import { summarize, WATERS, cloneIons, ION_IDS, IONS } from '../core/water.js';
import { download, readTable, extOf, checkFile } from '../core/io.js';
import { MIRRORS, APP } from '../data/app.js';

const card = (...kids) => h('section', { class: 'card' }, ...kids);
const pc = (spec) => plotCard(spec, { onDownload: download });
const ago = (iso) => { if (!iso) return 'never'; const s = (Date.now() - new Date(iso)) / 1000; return s < 90 ? 'just now' : s < 5400 ? Math.round(s / 60) + ' min ago' : s < 172800 ? Math.round(s / 3600) + ' h ago' : Math.round(s / 86400) + ' d ago'; };

// ---------------------------------------------------------------------------------------------- home
export function home(root) {
  const c = store.case, done = SUITES.filter((s) => c.outputs[s.id]).length;
  const cards = SUITES.map((s) => {
    const o = c.outputs[s.id], k = o?._kpis?.slice(0, 3) || [];
    return h('a', { class: 'suite-card' + (o ? ' done' : ''), href: '#/suite/' + s.id },
      h('div', { class: 'sc-top' }, h('span', { class: 'sc-ico', 'aria-hidden': 'true' }, s.icon), h('span', { class: 'sc-num' }, String(s.num).padStart(2, '0')), o ? badge('solved ' + ago(o._at), 'ok') : badge('not run', '')),
      h('h3', null, s.title), h('p', null, s.blurb),
      k.length ? h('ul', { class: 'sc-kpi' }, k.map((q) => h('li', null, q.label + ': ', h('b', null, typeof q.value === 'number' ? fmt(q.value, 3) : String(q.value)), ' ' + q.unit))) : null,
      h('div', { class: 'sc-links' }, s.uses.length ? '⛓ uses ' + s.uses.map((u) => byId(u).short).join(', ') : '⛓ starts the chain'));
  });
  fill(root, 
    h('section', { class: 'hero' },
      h('div', null, h('h1', null, 'One connected workspace for every desalination calculation'),
        h('p', null, 'Thirteen engineering suites — membranes, chemistry, whole-plant balances, CFD, sea discharge, thermal, electrochemical, emerging processes, ZLD, fouling, optimisation, pumping and economics — share one case. Results flow from suite to suite automatically, so a single industrial case can be analysed end to end.'),
        h('div', { class: 'row-tools' }, h('a', { class: 'btn primary', href: '#/case' }, '1 · Define the case'), h('a', { class: 'btn', href: '#/site' }, '2 · Pull site data'), h('a', { class: 'btn', href: '#/chain' }, '3 · Run everything'), h('a', { class: 'btn', href: '#/advisor' }, '4 · Decide'))),
      h('div', { class: 'hero-stat' }, kpiGrid([{ label: 'Active case', value: c.name }, { label: 'Feed', value: `${fmt(c.feed.Q)} m³/h` }, { label: 'Feed TDS', value: summarize(c.feed.ions, c.feed.T).tds, unit: 'mg/L' }, { label: 'Site', value: c.site.name || (c.site.lat !== null ? `${fmt(c.site.lat, 4)}, ${fmt(c.site.lon, 4)}` : 'not set') }, { label: 'Suites solved', value: `${done} / 13` }, { label: 'Site data', value: c.site.fetchedAt ? ago(c.site.fetchedAt) : 'not pulled' }]))),
    h('h2', { class: 'sect' }, 'The 13 suites'), h('div', { class: 'suite-grid' }, cards,
      [['🌍', 'Global site data', 'Live sea, weather, seabed and economic data for any location', '#/site', c.site.fetchedAt ? 'pulled ' + ago(c.site.fetchedAt) : 'not pulled'], ['🔗', 'Integrated run', 'Solve all suites in sequence with results passed along', '#/chain', `${done} / 13 solved`], ['🧭', 'Decision support', 'Ranked recommendations and sustainability scorecard', '#/advisor', 'whole case']].map(([ico, t, b, href, st]) =>
        h('a', { class: 'suite-card whole', href }, h('div', { class: 'sc-top' }, h('span', { class: 'sc-ico', 'aria-hidden': 'true' }, ico), h('span', { class: 'sc-num' }, ''), badge(st, '')), h('h3', null, t), h('p', null, b), h('div', { class: 'sc-links' }, 'whole case')))),
    h('h2', { class: 'sect' }, 'How the suites are wired together'),
    card(h('div', { class: 'flowmap' }, CHAIN.map((id) => { const s = byId(id), d = downstream(id); return h('div', { class: 'fm-row' }, h('a', { class: 'chip on', href: '#/suite/' + id }, `${s.num}. ${s.short}`), h('span', { class: 'fm-arrow', 'aria-hidden': 'true' }, '→'), h('div', { class: 'fm-to' }, d.length ? d.map((x) => h('a', { class: 'chip', href: '#/suite/' + x.id }, `${x.num}. ${x.short}`)) : h('span', { class: 'note' }, 'final results — feeds the decision report'))); })),
      h('p', { class: 'note' }, 'Each row reads “this suite feeds →”. For example the RO concentrate becomes the brine analysed by Chemistry, concentrated by ZLD and discharged by Outfall; its pressures and flows size the pumps; energy, membranes and chemicals end up in Economics.')));
}

// ------------------------------------------------------------------------------------- case & feed
export function casePage(root) {
  const c = store.case, f = c.feed, sumBox = h('div');
  const paintSum = () => { const s = summarize(store.case.feed.ions, store.case.feed.T); fill(sumBox, kpiGrid([{ label: 'TDS', value: s.tds, unit: 'mg/L' }, { label: 'Salinity', value: s.salinity, unit: 'g/kg' }, { label: 'Osmotic pressure', value: s.osmoticBar, unit: 'bar' }, { label: 'Conductivity', value: s.conductivity, unit: 'µS/cm' }, { label: 'Density', value: s.density, unit: 'kg/m³' }, { label: 'Hardness', value: s.hardness, unit: 'mg/L CaCO₃' }, { label: 'Alkalinity', value: s.alkalinity, unit: 'mg/L CaCO₃' }, { label: 'Charge balance', value: s.chargeErrorPct, unit: '%', status: Math.abs(s.chargeErrorPct) > 5 ? 'warn' : 'ok' }])); };
  const get = (k) => store.case.feed[k], set = (k, v) => { store.setFeed({ [k]: v }); paintSum(); };
  const F = (def) => fieldRow(def, get, set, {});
  const lib = store.library(), libSel = h('select', { 'aria-label': 'Saved cases' }, h('option', { value: '' }, Object.keys(lib).length ? 'Open a saved case…' : 'No saved cases yet'), Object.keys(lib).map((n) => h('option', { value: n }, n)));
  libSel.addEventListener('change', () => { if (libSel.value) { store.loadFromLibrary(libSel.value); toast('Case loaded.', 'ok'); casePage(root); } });
  paintSum();
  fill(root, 
    h('header', { class: 'page-head' }, h('h1', null, 'Case & feed water'), h('p', null, 'A case is one industrial study: its site, its feed water and the inputs and results of every suite. Everything is stored on this device only.')),
    card(h('h2', null, 'Case'),
      h('div', { class: 'fields' },
        h('div', { class: 'field' }, h('label', { for: 'c_name' }, 'Case name'), h('div', { class: 'ctl' }, h('input', { id: 'c_name', type: 'text', maxlength: 120, value: c.name, oninput: (e) => store.update({ name: e.target.value.slice(0, 120) || 'Untitled case' }) }))),
        h('div', { class: 'field' }, h('label', { for: 'c_auto' }, 'Auto-link suites', help('When on, each suite takes matching values from the case feed water, the site data and the suites upstream of it every time it runs.')), h('div', { class: 'ctl' }, h('label', { class: 'switch' }, h('input', { id: 'c_auto', type: 'checkbox', checked: c.autolink, onchange: (e) => store.update({ autolink: e.target.checked }) }), h('span', { class: 'slider' })))),
        h('div', { class: 'field wide' }, h('label', { for: 'c_notes' }, 'Notes'), h('div', { class: 'ctl wide' }, h('textarea', { id: 'c_notes', rows: 2, maxlength: 5000, oninput: (e) => store.update({ notes: e.target.value }) }, c.notes)))),
      h('div', { class: 'row-tools' },
        btn('Save to this device', () => toast(store.saveToLibrary() ? 'Case saved in the library on this device.' : 'Could not save — device storage is full.', 'ok'), 'primary'), libSel,
        btn('Export case file', () => download(store.exportJSON(), `${store.case.name}.brinelab.json`, 'application/json'), '', 'One portable JSON file with every input — share it or open it on another device'),
        importBtn('Import case file', async (file) => { checkFile(file); store.importJSON(await file.text()); toast('Case imported.', 'ok'); casePage(root); }, '.json'),
        btn('New blank case', () => { if (confirm('Start a new blank case? Unsaved inputs of the current case are discarded.')) { store.reset(); casePage(root); } }, 'ghost'))),
    card(h('h2', null, 'Feed water'), h('p', { class: 'note' }, 'This analysis is offered to every suite that needs feed chemistry. Pick a reference water, import a laboratory sheet (CSV / Excel with ion names and mg/L), or type the values.'),
      h('div', { class: 'fields' },
        F({ key: 'Q', label: 'Feed flow', unit: 'm³/h', value: f.Q, min: 0.01, max: 1e6, help: 'Raw-water flow to the plant.' }), F({ key: 'T', label: 'Temperature', unit: '°C', value: f.T, min: 0, max: 60 }),
        F({ key: 'pH', label: 'pH', unit: '', value: f.pH, min: 1, max: 13 }), F({ key: 'P', label: 'Pressure', unit: 'bar', value: f.P, min: 0, max: 100 }),
        F({ key: 'turbidity', label: 'Turbidity', unit: 'NTU', value: f.turbidity, min: 0, max: 1000 }), F({ key: 'sdi', label: 'Silt density index SDI₁₅', unit: '', value: f.sdi, min: 0, max: 6.7 }), F({ key: 'toc', label: 'Total organic carbon', unit: 'mg/L', value: f.toc, min: 0, max: 200 }),
        F({ key: 'ions', label: 'Ionic analysis (mg/L)', type: 'ions', value: f.ions })),
      sumBox));
}

// ---------------------------------------------------------------------------------- global site data
// Map tiles are fetched (not hot-linked) so the HTTP status can be checked: a refused tile is never
// shown, the next provider is tried instead, and good tiles are kept in memory for the session.
const TILE_SOURCES = [(z, x, y) => `https://tile.openstreetmap.org/${z}/${x}/${y}.png`, (z, x, y) => `https://basemaps.cartocdn.com/rastertiles/voyager/${z}/${x}/${y}.png`];
const tileCache = new Map();
let tileSource = 0;
function loadTile(z, x, y) {
  const key = `${z}/${x}/${y}`;
  if (tileCache.has(key)) return tileCache.get(key);
  const p = (async () => {
    for (let k = tileSource; k < TILE_SOURCES.length; k++) {
      try {
        const r = await fetch(TILE_SOURCES[k](z, x, y), { mode: 'cors', credentials: 'omit', referrerPolicy: 'strict-origin-when-cross-origin' });
        if (r.ok && (r.headers.get('content-type') || '').startsWith('image/')) return URL.createObjectURL(await r.blob());
        if (r.status === 403 || r.status === 429) tileSource = Math.max(tileSource, k + 1); // provider refuses this app: stop asking it
      } catch { /* offline or blocked: try the next provider */ }
    }
    tileCache.delete(key);
    return null;
  })();
  if (tileCache.size > 600) { const first = tileCache.keys().next().value; tileCache.get(first).then((u) => u && URL.revokeObjectURL(u)); tileCache.delete(first); }
  tileCache.set(key, p);
  return p;
}

function slippyMap(lat, lon, onPick) {
  let zoom = lat === null || lat === undefined ? 2 : 8, cLat = lat ?? 22, cLon = lon ?? 30, mLat = lat, mLon = lon;
  const box = h('div', { class: 'map', tabindex: '0', role: 'application', 'aria-label': 'World map. Click or tap to choose the site.' }), layer = h('div', { class: 'map-layer' }), pin = h('div', { class: 'map-pin', hidden: true }, '📍');
  box.append(layer, pin, h('div', { class: 'map-zoom' }, h('button', { type: 'button', 'aria-label': 'Zoom in', onclick: (e) => { e.stopPropagation(); zoom = Math.min(15, zoom + 1); draw(); } }, '+'), h('button', { type: 'button', 'aria-label': 'Zoom out', onclick: (e) => { e.stopPropagation(); zoom = Math.max(2, zoom - 1); draw(); } }, '−')),
    h('div', { class: 'map-attr' }, '© OpenStreetMap contributors · © CARTO'));
  const X = (lo, z) => ((lo + 180) / 360) * 2 ** z * 256, Y = (la, z) => { const r = (la * Math.PI) / 180; return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** z * 256; };
  const invX = (x, z) => (x / (2 ** z * 256)) * 360 - 180, invY = (y, z) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / (2 ** z * 256)))) * 180) / Math.PI;
  function draw() {
    const w = box.clientWidth || 600, hgt = box.clientHeight || 360, cx = X(cLon, zoom), cy = Y(cLat, zoom), n = 2 ** zoom;
    clear(layer);
    for (let tx = Math.floor((cx - w / 2) / 256); tx <= Math.floor((cx + w / 2) / 256); tx++) for (let ty = Math.floor((cy - hgt / 2) / 256); ty <= Math.floor((cy + hgt / 2) / 256); ty++) {
      if (ty < 0 || ty >= n) continue;
      const img = h('img', { alt: '', draggable: 'false' });
      img.style.left = Math.round(tx * 256 - cx + w / 2) + 'px'; img.style.top = Math.round(ty * 256 - cy + hgt / 2) + 'px';
      layer.append(img);
      loadTile(zoom, ((tx % n) + n) % n, ty).then((url) => { if (url && img.isConnected) img.src = url; else img.remove(); });
    }
    if (mLat !== null && mLat !== undefined) { pin.hidden = false; pin.style.left = X(mLon, zoom) - cx + w / 2 + 'px'; pin.style.top = Y(mLat, zoom) - cy + hgt / 2 + 'px'; } else pin.hidden = true;
  }
  let drag = null;
  box.addEventListener('pointerdown', (e) => { if (e.target.closest('.map-zoom')) return; drag = { x: e.clientX, y: e.clientY, cx: X(cLon, zoom), cy: Y(cLat, zoom), moved: false }; box.setPointerCapture(e.pointerId); });
  box.addEventListener('pointermove', (e) => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 5) drag.moved = true; if (drag.moved) { cLon = invX(drag.cx - dx, zoom); cLat = Math.max(-84, Math.min(84, invY(drag.cy - dy, zoom))); draw(); } });
  box.addEventListener('pointerup', (e) => {
    if (drag && !drag.moved) { const r = box.getBoundingClientRect(), px = X(cLon, zoom) + (e.clientX - r.left - r.width / 2), py = Y(cLat, zoom) + (e.clientY - r.top - r.height / 2); mLat = invY(py, zoom); mLon = ((invX(px, zoom) + 540) % 360) - 180; draw(); onPick(mLat, mLon); }
    drag = null;
  });
  box.addEventListener('wheel', (e) => { e.preventDefault(); zoom = Math.max(2, Math.min(15, zoom + (e.deltaY < 0 ? 1 : -1))); draw(); }, { passive: false });
  box.addEventListener('keydown', (e) => { const s = 40 / 2 ** zoom; if (e.key === 'ArrowLeft') cLon -= s; else if (e.key === 'ArrowRight') cLon += s; else if (e.key === 'ArrowUp') cLat += s; else if (e.key === 'ArrowDown') cLat -= s; else if (e.key === '+') zoom++; else if (e.key === '-') zoom--; else return; e.preventDefault(); draw(); });
  new ResizeObserver(draw).observe(box);
  box.setView = (la, lo, z) => { cLat = la; cLon = lo; mLat = la; mLon = lo; if (z) zoom = z; draw(); };
  return box;
}

export function sitePage(root) {
  const s = store.case.site;
  const latI = h('input', { type: 'number', step: 'any', min: -90, max: 90, value: s.lat ?? '', id: 's_lat', placeholder: 'e.g. 25.05' }), lonI = h('input', { type: 'number', step: 'any', min: -180, max: 180, value: s.lon ?? '', id: 's_lon', placeholder: 'e.g. 55.05' });
  const srcBox = h('div', { class: 'sources' }), dataBox = h('div'), results = h('ul', { class: 'search-results' });
  const map = slippyMap(s.lat, s.lon, (la, lo) => { latI.value = la.toFixed(4); lonI.value = lo.toFixed(4); });
  const paintSources = (live = {}) => fill(srcBox, SOURCES.map((src) => { const st = live[src.id] || store.case.site.status?.[src.id]; return h('div', { class: 'source ' + (st === 'loading' ? 'loading' : st?.ok ? 'ok' : st ? 'fail' : '') }, h('b', null, src.name), h('span', null, src.gives), h('small', null, src.provider + ' · ' + src.host), h('em', null, st === 'loading' ? 'fetching…' : st ? (st.ok ? 'live · ' + ago(st.at) : 'unavailable — ' + st.message) : 'not fetched')); }));
  const paintData = () => {
    const site = store.case.site, d = site.data || {};
    clear(dataBox);
    if (!site.fetchedAt) return dataBox.append(h('p', { class: 'note' }, 'Pick a coastal or inland point on the map (or search / type coordinates) and press “Fetch live site data”.'));
    const K = (label, v, unit, extra) => (v === null || v === undefined ? null : { label, value: v, unit, help: extra });
    dataBox.append(h('h2', { class: 'sect' }, `${site.name || 'Site'}${site.country ? ', ' + site.country : ''} — ${fmt(site.lat, 5)}°, ${fmt(site.lon, 5)}°`),
      kpiGrid([K('Sea-surface temperature', d.sst, '°C'), K(d.salinityEstimated ? 'Salinity (regional estimate)' : 'Salinity (climatology)', d.salinity, 'g/kg'), K('Water depth at point', d.depth, 'm'), K('Deepest nearby', d.maxDepthNearby, 'm'), K('Mean current', d.currentSpeed, 'm/s'), K('Peak current', d.currentMax, 'm/s'), K('Tidal range', d.tideRange, 'm'), K('Wave height', d.waveHeight, 'm'), K('Wave period', d.wavePeriod, 's'),
        K('Air temperature', d.airTemp, '°C'), K('Wind speed', d.windSpeed, 'm/s'), K('Humidity', d.humidity, '%'), K('Solar resource (long-term)', d.ghiDaily, 'kWh/m²·d'), K('Wind (long-term, 10 m)', d.windAnnual, 'm/s'), K('Land elevation', d.elevation, 'm'),
        K('Inflation' + (d.inflationYear ? ` (${d.inflationYear})` : ''), d.inflation, '%/y'), K('Lending rate' + (d.lendingRateYear ? ` (${d.lendingRateYear})` : ''), d.lendingRate, '%/y'), K(`${d.currency || ''} per USD`, d.fxPerUSD, ''), K('Electricity (indicative)', d.electricityPrice, '$/kWh'), K(d.gridCarbonLive ? `Grid carbon (${d.gridCarbonYear || 'latest'})` : 'Grid carbon (indicative)', d.gridCarbon, 'kgCO₂/kWh'), K('Renewable electricity', d.renewableShare, '%'), K('Water stress', d.waterStress, '%')].filter(Boolean)));
    const plots = [];
    if (d.bathy) {
      const flat = d.bathy.elev.flat(), zlo = Math.min(...flat), zhi = Math.max(...flat), allLand = zlo >= 0, allSea = zhi <= 0;
      plots.push({ type: 'field', title: allLand ? 'Terrain around the site (elevation above sea level, m)' : allSea ? 'Seabed around the site (m, negative = below sea level)' : 'Seabed and terrain around the site (m; blue = sea, green to brown = land)', xlabel: 'Longitude (°)', ylabel: 'Latitude (°)', zlabel: 'Elevation (m)', zunit: 'm', x: d.bathy.lon, y: d.bathy.lat, z: d.bathy.elev,
        cmap: allLand ? 'land' : allSea ? 'sea' : 'topo', zmid: allLand || allSea ? undefined : 0, contours: allLand || allSea ? 8 : [0], equal: true, markers: [{ x: site.lon, y: site.lat, label: 'site' }],
        onPick: (lo, la) => { latI.value = la.toFixed(4); lonI.value = lo.toFixed(4); map.setView(la, lo); toast(`Site moved to ${la.toFixed(4)}°, ${lo.toFixed(4)}° — fetching live data for the new point…`); doFetch(); },
        note: 'Click anywhere on this panel to move the site to that point. ' + allLand ? `This point is inland: the ground is ${fmt(zlo, 3)}–${fmt(zhi, 3)} m above sea level and there is no sea within about 13 km, so the marine figures above come from the nearest sea cell (or are unavailable). For a coastal or outfall study, click a point at the coast or just offshore.` : allSea ? 'Open water: the whole window is below sea level.' : 'The white line is the shoreline (0 m).' });
    }
    if (d.tide) plots.push({ type: 'line', title: 'Sea level (tide + surge), last 3 days and next 5', xlabel: 'Hours', ylabel: 'm above mean sea level', series: [{ name: 'Sea level', x: d.tide.t, y: d.tide.eta }] });
    if (d.currents) plots.push({ type: 'line', title: 'Ocean-current speed', xlabel: 'Hours', ylabel: 'm/s', series: [{ name: 'Current speed', x: d.currents.t, y: d.currents.speed }] });
    if (d.salinityMonthly) plots.push({ type: 'line', title: 'Monthly climatology near the site', xlabel: 'Month', ylabel: 'Salinity (g/kg) · temperature (°C)', series: [{ name: 'Salinity', x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], y: d.salinityMonthly, mode: 'both' }, { name: 'Temperature', x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], y: d.sstMonthly, mode: 'both' }] });
    if (plots.length) dataBox.append(h('div', { class: 'plots' }, plots.map(pc)));
    dataBox.append(card(h('h2', null, 'Where these values go'), h('p', { class: 'note' }, 'Open any suite: matching inputs appear in its “linked data” bar and, with auto-link on, are applied when you run. Use the buttons below to also set the case feed water from the site.'),
      h('div', { class: 'row-tools' },
        d.salinity ? btn(`Set case feed to seawater at ${fmt(d.salinity, 3)} g/kg${d.sst ? ' and ' + fmt(d.sst, 3) + ' °C' : ''}`, () => { const base = WATERS.seawater.ions, tdsT = d.salinity * 1000 * 1.025, f = tdsT / ION_IDS.reduce((a, k) => a + base[k], 0); store.setFeed({ ions: cloneIons(Object.fromEntries(ION_IDS.map((k) => [k, +(base[k] * f).toPrecision(5)]))), T: d.sst ?? store.case.feed.T, name: `Seawater at ${site.name || 'site'}`, source: 'site' }); toast('Case feed water updated from site data.', 'ok'); }, 'primary') : null,
        btn('Download site data (JSON)', () => download(JSON.stringify(site, null, 1), 'site-data.json', 'application/json')))));
  };
  const doFetch = async () => {
    const la = +latI.value, lo = +lonI.value;
    if (!Number.isFinite(la) || !Number.isFinite(lo) || latI.value === '' || lonI.value === '' || Math.abs(la) > 90 || Math.abs(lo) > 180) return toast('Enter a valid latitude (−90…90) and longitude (−180…180), or click the map.', 'warn');
    if (!navigator.onLine) return toast('You are offline. Stored site data stay available; live data refresh when you reconnect.', 'warn');
    fetchBtn.disabled = true; const live = {};
    try {
      const site = await fetchSite(la, lo, (id, st, msg) => { live[id] = st === 'loading' ? 'loading' : { ok: st === 'ok', message: msg, at: new Date().toISOString() }; paintSources(live); });
      const old = store.case.site, same = old.lat !== null && Math.abs(old.lat - site.lat) < 0.02 && Math.abs(old.lon - site.lon) < 0.02;
      if (same) site.data = { ...old.data, ...site.data, ...(site.data.salinityEstimated && old.data?.salinity && !old.data.salinityEstimated ? { salinity: old.data.salinity, salinityEstimated: false } : {}) }; // keep earlier answers of sources that are down right now
      store.setSite(site); map.setView(site.lat, site.lon);
      const ok = Object.values(site.status).filter((x) => x.ok).length;
      toast(`Site data updated: ${ok} of ${SOURCES.length} sources answered.`, ok ? 'ok' : 'bad');
    } catch (e) { toast('Could not fetch site data: ' + e.message, 'bad'); }
    fetchBtn.disabled = false; paintSources(); paintData();
  };
  const fetchBtn = btn('Fetch live site data', doFetch, 'primary');
  const q = h('input', { type: 'search', placeholder: 'Search a city, port or plant location…', 'aria-label': 'Search place', maxlength: 80 });
  const doSearch = async () => {
    if (!q.value.trim()) return;
    try { const r = await searchPlace(q.value.trim()); fill(results, r.length ? r.map((p) => h('li', null, h('button', { type: 'button', class: 'linklike', onclick: () => { latI.value = p.lat.toFixed(4); lonI.value = p.lon.toFixed(4); map.setView(p.lat, p.lon, 9); clear(results); } }, `${p.name}${p.admin ? ', ' + p.admin : ''}, ${p.country}`))) : h('li', { class: 'note' }, 'No match — try another spelling.')); }
    catch { toast('Search is unavailable (offline?). Type coordinates instead.', 'warn'); }
  };
  q.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
  paintSources(); paintData();
  fill(root, 
    h('header', { class: 'page-head' }, h('h1', null, 'Global site data'), h('p', null, 'Choose any location on Earth. Your browser pulls current sea state, tides, currents, seabed relief, salinity and temperature climatology, weather, solar resource and national economic indicators directly from open global data services, and offers them to every suite.')),
    card(
      h('div', { class: 'site-bar' },
        h('div', { class: 'site-search' }, q, btn('Search', doSearch)),
        h('label', { class: 'site-coord', for: 's_lat' }, h('span', null, 'Latitude °N'), latI), h('label', { class: 'site-coord', for: 's_lon' }, h('span', null, 'Longitude °E'), lonI),
        fetchBtn, btn('Use my location', () => navigator.geolocation?.getCurrentPosition((p) => { latI.value = p.coords.latitude.toFixed(4); lonI.value = p.coords.longitude.toFixed(4); map.setView(p.coords.latitude, p.coords.longitude, 9); }, () => toast('Location permission was not granted.', 'warn')), 'ghost')),
      results, map,
      h('p', { class: 'note' }, 'Click or tap the map to place the site. For an outfall study choose a point in the sea a few hundred metres offshore; for a plant on land the marine sources report the nearest sea cell.'),
      h('h3', null, 'Live sources'), srcBox),
    dataBox);
}

// --------------------------------------------------------------------------------------- data portal
export function portalPage(root) {
  const out = h('div');
  const TABULAR = ['csv', 'tsv', 'txt', 'dat', 'xlsx', 'xlsm'];
  // Show an imported or generated geometry and offer every suite that can use it.
  const showGeometry = async (g) => {
    clear(out);
    const d = derive(g), routes = [];
    for (const m of SUITES) { try { const su = await loadSuite(m.id), items = geometryLinks(su, g, d), accepts = (SUITE_GEOMETRY[m.id]?.accepts || []).includes(g.kind); if (items.length || accepts) routes.push({ su, items }); } catch { /* suite not available */ } }
    routes.sort((a, b) => b.items.length - a.items.length);
    const send = (su, items) => { attachGeometry(su.id, g); for (const it of items) setInputValue(su, it.key, it.value); store.pref('tab.' + su.id, 'geometry'); toast(`${g.name} sent to ${su.short}${items.length ? ` — ${items.length} input${items.length > 1 ? 's' : ''} set` : ''}.`, 'ok'); location.hash = '#/suite/' + su.id; };
    out.append(card(h('h2', null, `Geometry: ${g.name}`), geometryCard(g),
      h('h3', null, 'Send to a suite'),
      routes.length ? h('div', { class: 'tbl-scroll' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Suite', 'What it will take from this geometry', ''].map((x) => h('th', null, x)))),
        h('tbody', null, routes.map(({ su, items }) => h('tr', null, h('td', { class: 'lead' }, `${su.icon} ${su.num}. ${su.title}`), h('td', { class: 'lead wrap' }, items.length ? items.map((it) => `${allFields(su).find((f) => f.key === it.key)?.label || it.key} ← ${it.from}`).join('; ') : 'Attached for reference: preview and measured dimensions on its Geometry tab'), h('td', null, btn(items.length ? 'Send & apply' : 'Attach', () => send(su, items), items.length ? 'mini primary' : 'mini')))))))
        : h('p', { class: 'note' }, 'No suite uses this class of geometry directly. Its measured dimensions are shown above.')));
  };
  const handle = async (files) => {
    for (const f of files) checkFile(f);
    const file = files[0], ext = extOf(file.name);
    clear(out);
    if (files.length === 1 && ext === 'json' && /brinelab/i.test(file.name)) { store.importJSON(await file.text()); toast('Case imported.', 'ok'); return out.append(card(h('h2', null, 'Case imported'), h('p', null, `“${store.case.name}” is now the active case.`), h('a', { class: 'btn primary', href: '#/case' }, 'Open the case'))); }
    if (files.length > 1 || !TABULAR.includes(ext)) return showGeometry(await readFiles(files));
    const t = await readTable(file);
    if (!t.records.length) throw new Error('The file contains no data rows.');
    const targets = [];
    for (const m of SUITES) { try { const su = await loadSuite(m.id); allFields(su).filter((f) => f.type === 'table').forEach((f) => targets.push({ su, f })); } catch { /* skip */ } }
    const sel = h('select', { 'aria-label': 'Destination' }, targets.map((x, i) => h('option', { value: i }, `${x.su.num}. ${x.su.short} → ${x.f.label}`)));
    const mapBox = h('div');
    const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
    const paintMap = () => {
      const { f } = targets[+sel.value] || {};
      if (!f) return;
      fill(mapBox, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, h('th', null, 'Destination column'), h('th', null, 'Take from file column'))),
        h('tbody', null, f.columns.map((c, i) => h('tr', null, h('td', { class: 'lead' }, c.label + (c.unit ? ` (${c.unit})` : '')), h('td', null, h('select', { dataset: { key: c.key } }, h('option', { value: '' }, '— leave empty —'), t.headers.map((hd, j) => h('option', { value: hd, selected: norm(hd) === norm(c.key) || norm(hd) === norm(c.label) || (j === i && !t.headers.some((x) => norm(x) === norm(c.key))) }, hd)))))))));
    };
    sel.addEventListener('change', paintMap); paintMap();
    out.append(card(h('h2', null, `Table: ${file.name}`), h('p', { class: 'note' }, `${t.records.length} rows × ${t.headers.length} columns detected.`),
      dataTable({ title: 'Preview (first 12 rows)', columns: t.headers, rows: t.records.slice(0, 12).map((r) => t.headers.map((hd) => r[hd])) }),
      h('div', { class: 'row-tools' }, btn('This file is geometry (coordinates, soundings, point cloud, network table) — read it as geometry', async () => { try { await showGeometry(await readFiles(files)); } catch (e) { toast(e.message, 'bad', 12000); } })),
      targets.length ? h('div', null, h('h3', null, 'Send to a suite as data'), h('div', { class: 'row-tools' }, sel), mapBox,
        h('div', { class: 'row-tools' }, btn('Load into suite', () => { const { su, f } = targets[+sel.value]; const m = Object.fromEntries([...mapBox.querySelectorAll('select')].map((s) => [s.dataset.key, s.value])); store.setInput(su.id, f.key, t.records.map((r) => Object.fromEntries(f.columns.map((c) => [c.key, m[c.key] ? r[m[c.key]] : null])))); toast(`${t.records.length} rows loaded into ${su.short}.`, 'ok'); location.hash = '#/suite/' + su.id; }, 'primary'))) : null,
      h('p', { class: 'note' }, 'Water analyses (ion names with mg/L) can be imported directly inside any feed-analysis editor with its “Import analysis” button.')));
  };
  const input = h('input', { type: 'file', hidden: true, multiple: true, accept: ACCEPT + ',.xlsx,.xlsm,.tsv' });
  const safe = async (files) => { try { await handle([...files]); } catch (e) { clear(out); const m = formatOf(files[0]?.name || ''); out.append(card(h('h2', null, `Could not read ${files[0]?.name || 'the file'} directly`), h('p', { class: m?.support === 'convert' ? 'summary' : 'bad' }, e.message || 'Unreadable file.'), m ? h('p', { class: 'note' }, `Recognised as ${m.name}.`) : h('p', { class: 'note' }, 'The file type was not recognised. See the list of supported formats below.'))); toast(m?.support === 'convert' ? 'This format needs one conversion step — see the instruction.' : 'Could not read the file.', m?.support === 'convert' ? 'warn' : 'bad', 8000); } };
  input.addEventListener('change', () => { if (input.files.length) safe([...input.files]); input.value = ''; });
  const drop = h('div', { class: 'drop', tabindex: '0', role: 'button', 'aria-label': 'Choose or drop files', onclick: () => input.click(), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); } }, h('div', { class: 'drop-ico', 'aria-hidden': 'true' }, '⬆'), h('b', null, 'Drop files here or click to choose'), h('span', null, 'Files are read on this device only — nothing is uploaded. Select companion files together (e.g. .shp + .dbf, .mhd + .raw).'), input);
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files.length) safe([...e.dataTransfer.files]); });
  fill(root, 
    h('header', { class: 'page-head' }, h('h1', null, 'Data portal'), h('p', null, 'One place to bring case data in: geometry of every kind, laboratory analyses, plant historian logs, pump curves, cost tables and complete case files. Each import is recognised, previewed, measured and routed to the suites that can use it.')),
    card(drop, h('div', { class: 'formats' },
      h('div', null, h('b', null, 'CAD and surfaces'), h('span', null, 'STEP, IGES, STL, OBJ, PLY, OFF, 3MF, AMF, glTF/GLB, COLLADA, VRML/X3D, VTK')),
      h('div', null, h('b', null, 'Computational meshes'), h('span', null, 'Gmsh, SU2, UNV, Nastran, VTK/VTU, Tecplot, Plot3D, node–element files')),
      h('div', null, h('b', null, 'Drawings'), h('span', null, 'DXF, SVG, HPGL, x-y profiles')),
      h('div', null, h('b', null, 'GIS, bathymetry, point clouds'), h('span', null, 'Shapefile, GeoJSON, KML/KMZ, GPX, GML, GeoTIFF, ASCII grid, NetCDF, XYZ, LAS, PTS/PTX')),
      h('div', null, h('b', null, 'Images and voxels'), h('span', null, 'PNG/JPEG/BMP, TIFF stacks, RAW, NRRD, MetaImage, NIfTI, NPY/NPZ')),
      h('div', null, h('b', null, 'Plant and piping networks'), h('span', null, 'JSON, YAML, XML, CSV connectivity, PCF, IFC, AutomationML')),
      h('div', null, h('b', null, 'Tables'), h('span', null, 'CSV, TSV, TXT, JSON, Excel .xlsx — logs, analyses, curves, schedules')),
      h('div', null, h('b', null, 'Cases'), h('span', null, '.brinelab.json — every input of all 13 suites in one file')))),
    out,
    card(h('h2', null, 'Generate geometry without a file'), h('p', { class: 'note' }, 'Procedural and parametric geometry: spacer-filled channels, filament lattices, periodic minimal surfaces, random packed grains and cellular foams — useful for spacer optimisation, porous membranes and deposits.'), generatorPanel((g) => showGeometry(g).catch((e) => toast(e.message, 'bad')))),
    card(h('h2', null, 'Supported geometry formats'), h('p', { class: 'note' }, 'Open, documented formats are read directly on this device. Closed or kernel-specific formats cannot be decoded without the vendor’s licensed translator, so for those the portal tells you exactly which neutral format to export.'), formatCatalogue()));
}

// ------------------------------------------------------------------------------------ integrated run
export function chainPage(root) {
  const rows = h('tbody'), bar = h('div', { class: 'progress', hidden: true }, h('i')), summary = h('div');
  const sel = Object.fromEntries(CHAIN.map((id) => [id, h('input', { type: 'checkbox', checked: store.pref('chain.' + id) !== false, 'aria-label': 'Include ' + byId(id).title, onchange: (e) => store.pref('chain.' + id, e.target.checked) })]));
  const cell = {};
  CHAIN.forEach((id, i) => { const s = byId(id); cell[id] = { st: h('td', null, store.case.outputs[id] ? badge('solved ' + ago(store.case.outputs[id]._at), 'ok') : badge('waiting', '')), kp: h('td', { class: 'kp' }, kpText(store.case.outputs[id])), ln: h('td', null, '') }; rows.append(h('tr', null, h('td', null, sel[id]), h('td', { class: 'num' }, i + 1), h('td', { class: 'lead' }, h('a', { href: '#/suite/' + id }, `${s.icon} ${s.num}. ${s.title}`)), cell[id].ln, cell[id].st, cell[id].kp)); });
  function kpText(o) { return o?._kpis?.length ? o._kpis.slice(0, 4).map((k) => `${k.label} ${typeof k.value === 'number' ? fmt(k.value, 3) : k.value} ${k.unit}`).join(' · ') : '—'; }
  const go = btn('▶  Run the selected suites in sequence', async () => {
    go.disabled = true; bar.hidden = false; clear(summary);
    const list = CHAIN.filter((id) => sel[id].checked); let okN = 0, viol = 0;
    for (let i = 0; i < list.length; i++) {
      const id = list[i]; bar.firstChild.style.width = Math.round((100 * i) / list.length) + '%';
      clear(cell[id].st).append(badge('running…', 'warn'));
      try {
        const su = await loadSuite(id), items = linkItems(su), n = applyLinks(su, items);
        cell[id].ln.textContent = n ? `${n} linked` : '–'; cell[id].ln.title = items.map((x) => `${x.key} ← ${x.from}`).join('\n');
        await new Promise((r) => setTimeout(r, 15));
        const res = await runSuite(su), bad = res.warnings.filter((w) => w.level === 'bad').length;
        viol += bad; okN++;
        clear(cell[id].st).append(badge(bad ? `${bad} limit issue${bad > 1 ? 's' : ''}` : 'solved', bad ? 'warn' : 'ok'), h('small', null, ` ${Math.round(res._ms)} ms`));
        cell[id].kp.textContent = kpText(store.case.outputs[id]);
      } catch (e) { console.error(e); clear(cell[id].st).append(badge('failed', 'bad')); cell[id].kp.textContent = String(e.message || e).slice(0, 160); }
    }
    bar.firstChild.style.width = '100%'; setTimeout(() => (bar.hidden = true), 600); go.disabled = false;
    const o = store.case.outputs;
    summary.append(h('h2', { class: 'sect' }, 'Case summary'), kpiGrid([
      o.ro && { label: 'Product water', value: o.ro.permeateFlow, unit: 'm³/h' }, o.ro && { label: 'Recovery', value: 100 * o.ro.recovery, unit: '%' }, o.ro && { label: 'Feed pressure', value: o.ro.feedPressureBar, unit: 'bar' },
      o.chem?.limitingMineral && { label: 'First scale to form', value: o.chem.limitingMineral }, o.chem?.maxRecovery && { label: 'Scaling-limited recovery', value: 100 * o.chem.maxRecovery, unit: '%' },
      o.pump?.sec && { label: 'Net pumping energy', value: o.pump.sec, unit: 'kWh/m³' }, o.plant?.secElec && { label: 'Plant electricity', value: o.plant.secElec, unit: 'kWh/m³' },
      o.sea?.nearFieldDilution && { label: 'Outfall near-field dilution', value: o.sea.nearFieldDilution, unit: '×' }, o.fouling?.daysToCleaning != null && { label: 'Days to next cleaning', value: o.fouling.daysToCleaning, unit: 'd' },
      o.zld?.solids != null && { label: 'ZLD solids', value: o.zld.solids, unit: 't/d' }, o.econ?.lcow && { label: 'Levelised cost of water', value: o.econ.lcow, unit: '$/m³' }, o.econ?.capex && { label: 'CAPEX', value: o.econ.capex / 1e6, unit: 'M$' },
    ].filter(Boolean)), h('p', { class: 'summary' }, `${okN} of ${list.length} suites solved` + (viol ? `, with ${viol} design-limit issue${viol > 1 ? 's' : ''} to review (open the flagged suites).` : ' with no design-limit violations.')),
    h('div', { class: 'row-tools' }, btn('Export all results (JSON)', () => download(JSON.stringify({ case: store.case.name, site: { name: store.case.site.name, lat: store.case.site.lat, lon: store.case.site.lon }, outputs: store.case.outputs }, null, 1), `${store.case.name}_all_results.json`, 'application/json'))));
    summary.append(h('div', { class: 'linkbar' }, 'Next: ', h('a', { class: 'btn mini primary', href: '#/advisor' }, 'See ranked recommendations and the sustainability scorecard →')));
    toast('Integrated run finished.', 'ok');
  }, 'primary');
  fill(root, 
    h('header', { class: 'page-head' }, h('h1', null, 'Integrated run'), h('p', null, 'Solves the suites one after another in data-flow order. Before each suite runs, it receives the case feed water, the site data and the results of the suites before it — so one industrial case is analysed across all disciplines in a single pass.')),
    card(h('div', { class: 'row-tools' }, go, btn('Select all', () => CHAIN.forEach((id) => { sel[id].checked = true; store.pref('chain.' + id, true); }), 'ghost'), btn('Select none', () => CHAIN.forEach((id) => { sel[id].checked = false; store.pref('chain.' + id, false); }), 'ghost')), bar,
      h('div', { class: 'tbl-scroll' }, h('table', { class: 'tbl chain' }, h('thead', null, h('tr', null, ['', '#', 'Suite', 'Links', 'Status', 'Key results'].map((x) => h('th', null, x)))), rows))),
    summary);
}

// ------------------------------------------------------------------------------- app, offline, mirrors
export function appPage(root, app) {
  const mirrorBox = h('div', { class: 'sources' }), stBox = h('div');
  const check = async () => {
    fill(mirrorBox, MIRRORS.map((m) => h('div', { class: 'source loading', dataset: { url: m.url } }, h('b', null, m.name), h('span', null, m.note), h('small', null, m.url), h('em', null, 'checking…'))));
    for (const el of mirrorBox.children) {
      const url = el.dataset.url;
      try { const ctl = new AbortController(), tm = setTimeout(() => ctl.abort(), 9000); const r = await fetch(url + 'version.json?t=' + Date.now(), { signal: ctl.signal, cache: 'no-store', credentials: 'omit', mode: 'cors' }); clearTimeout(tm); const j = await r.json(); el.className = 'source ok'; el.lastChild.textContent = `online · build ${String(j.version).slice(0, 12)}`; }
      catch { el.className = 'source fail'; el.lastChild.textContent = navigator.onLine ? 'not reachable from this network' : 'offline'; }
      el.append(h('a', { class: 'btn mini', href: url + 'index.html', rel: 'noopener' }, 'Open'));
    }
  };
  const paintStorage = async () => {
    let est = null; try { est = await navigator.storage?.estimate?.(); } catch { /* not supported */ }
    const persisted = await navigator.storage?.persisted?.().catch(() => false);
    fill(stBox, kpiGrid([{ label: 'App build', value: app.version || 'dev' }, { label: 'Connection', value: navigator.onLine ? 'online' : 'offline', status: navigator.onLine ? 'ok' : 'warn' }, { label: 'Offline copy', value: app.offlineReady() ? 'ready' : 'preparing…', status: app.offlineReady() ? 'ok' : 'warn' }, { label: 'Installed', value: app.installed() ? 'yes' : 'not yet' }, est ? { label: 'Storage used', value: (est.usage || 0) / 1e6, unit: 'MB' } : null, { label: 'Protected storage', value: persisted ? 'yes' : 'no', help: 'Protected storage is not cleared automatically by the browser when the device runs low on space.' }].filter(Boolean)));
  };
  paintStorage();
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  fill(root, 
    h('header', { class: 'page-head' }, h('h1', null, 'Install, offline & availability'), h('p', null, `${APP.name} is a self-contained application: after the first visit it runs from this device, with or without a network, and keeps your cases locally.`)),
    card(h('h2', null, 'Install on this device'), stBox,
      h('div', { class: 'row-tools' }, btn('⬇  Install app', () => app.install(), 'primary'), btn('Check for updates', () => app.checkUpdate(true)), btn('Keep my data protected', async () => { const ok = await navigator.storage?.persist?.(); toast(ok ? 'Storage is now protected from automatic clean-up.' : 'The browser did not grant protected storage (install the app first).', ok ? 'ok' : 'warn'); paintStorage(); }, 'ghost'),
        h('a', { class: 'btn', href: 'standalone.html', download: 'BrineLab-standalone.html', title: 'A single HTML file containing the whole application. Copy it to any computer or phone and open it — no server and no internet needed.' }, '⬇  Download single-file edition')),
      h('ul', { class: 'steps' },
        h('li', null, h('b', null, 'Windows · macOS · Linux · ChromeOS (Chrome, Edge): '), 'press “Install app” above, or the install icon at the right of the address bar.'),
        h('li', null, h('b', null, 'Android (Chrome, Edge, Samsung Internet): '), 'press “Install app”, or menu ⋮ → “Add to Home screen / Install app”.'),
        h('li', { class: ios ? 'hl' : '' }, h('b', null, 'iPhone · iPad (Safari): '), 'tap the Share button, then “Add to Home Screen”. Apple does not allow an install button inside web pages.'),
        h('li', null, h('b', null, 'macOS Safari: '), 'File → “Add to Dock”.'), h('li', null, h('b', null, 'Firefox desktop: '), 'no install prompt, but the app still works offline in a normal tab once loaded; or use the single-file edition.'))),
    card(h('h2', null, 'Works in aeroplane mode'), h('p', null, 'All 13 calculation engines, the plotting, file import and your cases run entirely on the device. Only two things need a connection: pulling live site data and checking for a newer build. Site data already fetched remain stored with the case.'),
      h('p', { class: 'note' }, 'While you are online the app checks for a newer build in the background and refreshes stored site data that are more than six hours old; installed copies on supporting browsers also refresh periodically in the background.')),
    card(h('h2', null, 'Availability and mirrors'), h('p', null, MIRRORS.length > 1 ? 'The same build is published at more than one independent address. If one host is down, open another — or simply keep using the installed copy, which needs no host at all.' : 'Once installed (or saved as the single-file edition) the application needs no host at all: it keeps running if the web address is unreachable. The build is host-independent, so it can also be published to a second, independent host; none has been set up yet.'), mirrorBox, h('div', { class: 'row-tools' }, btn('Check mirrors now', check))),
    card(h('h2', null, 'Security and privacy'), h('ul', { class: 'steps' },
      h('li', null, 'No account, no tracking, no analytics, no cookies. Cases, inputs and results never leave this device unless you export them.'),
      h('li', null, 'A strict content-security policy blocks inline and third-party scripts; the app loads no external code libraries.'),
      h('li', null, 'Imported files are parsed as data only, size-limited, and never executed; all text is displayed as plain text.'),
      h('li', null, 'Live data requests go over HTTPS only, to a fixed allow-list of public data services, without credentials or referrer.'),
      h('li', null, 'Custom formulas in suite 11 are evaluated by a built-in safe expression parser, never by executing code.')),
      h('div', { class: 'row-tools' }, btn('Erase all data stored by this app on this device', async () => { if (!confirm('Erase every stored case, input and preference on this device? This cannot be undone.')) return; try { localStorage.clear(); } catch { /* ignore */ } toast('Local data erased. Reloading…'); setTimeout(() => location.reload(), 600); }, 'ghost danger'))),
    card(h('h2', null, 'Live data sources'), h('div', { class: 'sources' }, SOURCES.map((s) => h('div', { class: 'source' }, h('b', null, s.name), h('span', null, s.gives), h('small', null, s.provider + ' · ' + s.host))))));
  check();
}
