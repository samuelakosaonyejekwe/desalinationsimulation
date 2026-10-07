# Suite module contract

Every suite is one ES module in `js/suites/` that `export default`s a plain object. The generic
workspace (`js/core/suiteview.js`) turns that object into the full workflow (guide, inputs, model
setup, mesh, results, mesh-sensitivity, calibration, validation, verification, theory). A suite module
therefore contains **physics and declarations only** — no DOM code, no network access, no storage.

Suite modules may import only from `../core/num.js`, `../core/props.js`, `../core/water.js`,
`../core/io.js` (pure geometry helpers) and other suite modules' named engine exports. They must load
and run under plain Node (`node tests/run.mjs`) as well as in the browser.

```js
export default {
  id: 'ro',                  // short unique id (fixed, see table below)
  num: 1,                    // 1..13
  title: 'Reverse Osmosis & Membrane Design',
  short: 'RO design',        // <= 18 characters, used in navigation
  icon: '💧',
  tagline: 'One sentence shown under the title.',
  description: 'Two to four sentences: what is solved and how.',
  guide: ['Step 1 …', 'Step 2 …'],          // optional
  implemented: ['solution-diffusion', …],   // lower-case fragments matched against the reference
                                            // catalogue (js/data/catalog.js) to tick what is solved
  referenceOnly: ['poisson'],                // optional: fragments that must stay unticked even if matched
  equationsNote: 'optional note on model scope and validity limits',

  inputs: [                                 // groups of fields
    { group: 'Feed water', tab: 'inputs' | 'setup' | 'mesh', help: '…', showIf: (v) => true,
      fields: [
        { key: 'Qf', label: 'Feed flow', unit: 'm³/h', value: 1000, min: 0, max: 1e6, typical: [10, 50000], help: '…' },
        { key: 'model', label: 'Transport model', type: 'select', value: 'sd', options: [{ value: 'sd', label: 'Solution–diffusion' }] },
        { key: 'erd', label: 'Energy recovery', type: 'bool', value: true },
        { key: 'ions', label: 'Feed analysis', type: 'ions', value: WATERS.seawater.ions },   // mg/L object
        { key: 'log', label: 'Operating log', type: 'table', columns: [{ key: 't', label: 'Time', unit: 'd' }], value: [ {t: 0} ] },
        { key: 'geom', label: 'CAD geometry', type: 'file', value: null },                    // parsed STL/OBJ/DXF/GeoJSON
        // any field may have showIf: (v) => boolean
      ] },
  ],
  presets: [{ name: 'Brackish 75 % recovery', values: { Qf: 200 } }],

  // Values offered from the case feed water and from other suites' outputs (all optional chaining!)
  pull: ({ feed, site, outputs }) => [{ key: 'Qf', value: feed.Q, from: 'Case feed water' }],
  // Values offered from the Global Site Data page (site.data fields are listed below)
  site: (site) => [{ key: 'T', value: site.data.sst, from: 'Sea-surface temperature at site' }],

  // The engine. May be async. v = current input values keyed by field key.
  // ctx = { feed, site, outputs, progress(fraction, message), tick() -> Promise (yield to the UI) }
  run(v, ctx) {
    return {
      summary: 'One plain-language sentence describing the outcome.',
      kpis: [{ label: 'Recovery', value: 45.0, unit: '%', status: 'ok' | 'warn' | 'bad', help: '…' }],
      warnings: [{ level: 'bad' | 'warn' | 'info', msg: 'Lead-element flux 38 LMH exceeds the 34 LMH limit.' }],
      recommendations: ['Plain-language next action …'],
      plots: [ /* plot specs, see below */ ],
      tables: [{ title: 'Stage summary', columns: ['Stage', 'Flow (m³/h)'], rows: [[1, 550]], note: '…' }],
      balances: [{ name: 'Water mass', in: 1000, out: 999.9999 }],   // shown on the Verify tab
      outputs: { /* machine-readable values consumed by other suites — see table below */ },
    };
  },

  // Numerical-uncertainty study. keys are integer resolution inputs multiplied by the refinement ratio
  // (or, with refine: 'divide', step sizes divided by it). May be an array of several studies.
  mesh: { name: 'Axial discretisation', keys: ['nSeg'], min: 4, metrics: [{ label: 'Permeate flow', unit: 'm³/h', get: (res) => res.outputs.permeateFlow }] },

  // Parameter estimation + validation. model must be synchronous and fast.
  calibration: {
    note: '…',
    params: [{ key: 'A', label: 'Water permeability A', lo: 0.5, hi: 10 }],     // keys of input fields
    columns: [{ key: 'P', label: 'Feed pressure', unit: 'bar' }, { key: 'Qp', label: 'Permeate flow', unit: 'm³/h' }],
    targets: [{ key: 'Qp', label: 'Permeate flow', unit: 'm³/h' }],             // subset of columns that are measurements
    model: (v) => ({ Qp: 123.4 }),                                              // predictions for one operating point
    sample: [ { P: 55, Qp: 450 }, … ],            // realistic synthetic data (>= 6 rows)
    validationSample: [ … ],                     // different operating points
  },

  // Code/equation verification: conservation, limiting cases, analytical or hand calculations.
  verify() { return [{ name: 'Water balance closes', expected: 0, got: 1e-12, tol: 1e-9, pass: true, note: 'Qf = Qp + Qc' }]; },

  // Optional live feed: the shell lets the user follow a local export file and reloads this table input as it grows.
  live: { key: 'log', label: 'Plant operating log', help: '…' },

  // Optional extra tabs with custom content. `el` is an empty container; build DOM only with api.h(...)
  views: [{ id: 'flowsheet', label: 'Flowsheet', tip: '…', render(el, api) { /* api = { h, values(), set(k,v), result(), run(), plotCard(spec), dataTable(spec), kpiGrid(items), toast, download } */ } }],
};
```

## Plot specs

```js
{ type: 'line', title, xlabel, ylabel, logx, logy, ymin, ymax, xmin, xmax, zeroY,
  series: [{ name, x: [], y: [], mode: 'line' | 'points' | 'both' | 'step', dash: true, color }],
  hlines: [{ y, label, color }], vlines: [{ x, label }], note }
{ type: 'bar', title, ylabel, categories: [], series: [{ name, values: [] }], stacked: true }
{ type: 'field', title, xlabel, ylabel, zlabel, zunit, x: [nx], y: [ny], z: [ny][nx] /* rows */, zmin, zmax,
  cmap: 'viridis' | 'turbo' | 'coolwarm' | 'salinity' | 'thermal', contours: 8, equal: true,
  u: [ny][nx], v: [ny][nx], stream: true, vectors: true, mask: [ny][nx] /* true = solid */,
  shapes: [{ x: [], y: [], closed, color, dash, fill }], markers: [{ x, y, label }] }
```

## Fixed ids and the outputs each suite publishes

Streams are `{ Q (m³/h), T (°C), P (bar), pH, tds (mg/L), ions: { Na: mg/L, … } }`.

| num | id | outputs (all optional for consumers — always read with `?.`) |
|---|---|---|
| 1 | `ro` | `streams.{feed,permeate,concentrate}`, `feedPressureBar`, `concentratePressureBar`, `recovery` (0–1), `permeateFlow`, `fluxLMH`, `membraneArea`, `nElements`, `nVessels`, `sec` (kWh/m³), `pumpPower` (kW), `dpBar`, `cpFactor` |
| 2 | `chem` | `streams.brine`, `SI` (object mineral → SI), `maxRecovery` (0–1), `limitingMineral`, `antiscalantDose` (mg/L), `acidDose` (mg/L), `lsi`, `ionicStrength` |
| 3 | `plant` | `productFlow`, `brineFlow` (m³/h), `recovery`, `power` (kW), `heat` (kW), `secElec`, `secThermal` (kWh/m³), `chemicals` (kg/d), `streams.{product,brine}` |
| 4 | `cfd` | `dpPerM` (Pa/m), `frictionFactor`, `sherwood`, `kMass` (m/s), `cpFactor`, `wallShear` (Pa) |
| 5 | `sea` | `nearFieldDilution`, `impactSalinity` (g/kg), `excessAtMixingZone` (g/kg), `complianceDistance` (m), `outfallLength` (m), `nPorts`, `portDiameter` (m) |
| 6 | `thermal` | `distillate` (m³/h), `GOR`, `PR`, `secThermal`, `secElec` (kWh/m³), `steam` (kg/s), `area` (m²), `streams.{distillate,brine}` |
| 7 | `ed` | `streams.{diluate,concentrate}`, `sec`, `power` (kW), `area` (m²), `cellPairs`, `currentDensity` (A/m²) |
| 8 | `fomd` | `flux` (LMH), `secThermal`, `secElec`, `area`, `streams.{product,concentrate}` |
| 9 | `zld` | `waterRecovered` (m³/h), `solids` (t/d), `secElec`, `secThermal`, `power` (kW), `salts` (object name → t/d), `liquidDischarge` (m³/h) |
| 10 | `fouling` | `foulingRate` (%/d permeability loss), `daysToCleaning`, `cleaningsPerYear`, `membraneLife` (y), `normPermeability` (fraction of clean), `dominantFoulant` |
| 11 | `opt` | `best` (object of optimal decision variables), `objective`, `pareto` (array) |
| 12 | `pump` | `hpPumpPower`, `boosterPower`, `intakePower`, `erdRecovered`, `netPower` (kW), `sec` (kWh/m³), `pumpEfficiency`, `erdEfficiency` |
| 13 | `econ` | `lcow` ($/m³), `capex` ($), `opex` ($/y), `npv`, `irr`, `payback` (y), `carbonIntensity` (kgCO₂/m³) |

## Site data (`site.data`, any field may be missing)

`sst` (°C), `salinity` (g/kg), `depth` (m, positive down at the site), `bathy` ({ lat[], lon[], elev[][] } local grid, elevation m, negative below sea level),
`currentSpeed` (m/s), `currentDir` (°), `tideRange` (m), `tide` ({ t[] h, eta[] m }), `currents` ({ t[], speed[], dir[] }),
`waveHeight` (m), `wavePeriod` (s), `waveDir` (°), `airTemp` (°C), `windSpeed` (m/s), `windDir` (°), `humidity` (%),
`pressure` (hPa), `solar` (W/m²), `ghiDaily` (kWh/m²/d), `elevation` (m), `inflation` (%/y), `lendingRate` (%/y),
`fxPerUSD`, `currency`, `electricityPrice` ($/kWh), `gridCarbon` (kgCO₂/kWh).

## Shared toolbox

- `num.js`: `brent, solve1, newton1, newtonN, solveLinear, tridiag, rk4, rk45, nelderMead, diffEvolution, levenbergMarquardt, lstsq, linfit, trapz, interp1, linspace, logspace, metrics, gci, lhs, rng, histogram, mean, std, variance, quantile, sum, clamp, fmt, isNum`
- `props.js`: `density, viscosity, cp, conductivityThermal, psat, tsat, psatSeawater, antoine, latentHeat, bpe, enthalpyLiquid, enthalpyVapour, vapourDensity, osmoticCoefficient, osmoticPressure, diffusivityNaCl, conductivityFromTDS, salinityFromTDS, tdsFromSalinity, tcf, R, F, G, KELVIN` (T in °C, S in g/kg, SI otherwise)
- `water.js`: `IONS, ION_IDS, CATIONS, ANIONS, WATERS, cloneIons, tds, molar, totalMolar, ionicStrength, chargeBalance, balanceCharge, scaleIons, mixIons, seawaterAtTDS, osmoticPressureIons, vantHoff, conductivity, hardness, alkalinity, summarize`
- `io.js` (pure parts): `sliceMesh, polylinesToSegments, rasterize`
