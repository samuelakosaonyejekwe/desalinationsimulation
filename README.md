# BrineLab — integrated desalination simulation suite

BrineLab is an installable, offline-capable web application that connects thirteen engineering
suites around one shared case, so a single industrial desalination study can be analysed end to end.

**Open it:** https://samuelakosaonyejekwe.github.io/desalinationsimulation/

| # | Suite | What it solves |
|---|---|---|
| 1 | Reverse osmosis & membrane design | Element-by-element arrays, solution–diffusion / Spiegler–Kedem, polarisation, ion-by-ion permeate, staging, second pass, energy recovery |
| 2 | Brine chemistry, precipitation & scaling | Speciation, Debye–Hückel → Pitzer activity models, saturation indices, precipitation, scaling-limited recovery, dosing |
| 3 | Whole-plant process simulation | Sequential-modular flowsheets with recycle, mass / energy / exergy balances, off-design and dynamic cases |
| 4 | Flow in membranes, channels & equipment | 2-D Navier–Stokes with species and heat transport, spacers, membrane walls, imported CAD geometry, grid-convergence |
| 5 | Brine discharge into the sea | Dense-jet integral near field, density current, tidal far-field plume on real bathymetry, compliance |
| 6 | Thermal desalination | MED, MED-TVC, MSF, MVC stage-by-stage balances, exergy |
| 7 | Electrodialysis & electrochemical | Nernst–Planck stack model, limiting current, BMED, CDI |
| 8 | FO, MD & emerging | ICP/ECP forward osmosis, dusty-gas membrane distillation, hybrids |
| 9 | Brine concentration, crystallisation & ZLD | Evaporation path, salt sequence, population balance, ZLD balance |
| 10 | Fouling & performance monitoring | Normalisation, blocking laws, diagnosis, cleaning forecast |
| 11 | Optimisation, AI & custom modelling | NLP, NSGA-II, sensitivity, uncertainty, surrogates, MPC, safe custom equations |
| 12 | Energy recovery & pumps | System curves, NPSH, pressure exchangers, turbines, water hammer |
| 13 | Economics | CAPEX / OPEX, LCOW, cash flow, Monte-Carlo, carbon |

Every suite has the same workflow: **Guide → Inputs → Model setup → Mesh → Results → Calibrate &
validate → Verify → Theory**, with plots, tables, CSV/PNG/JSON/HTML-report export, a three-level
grid-convergence (GCI) study where a discretisation exists, least-squares calibration with parameter
identifiability, independent validation metrics, and live verification checks.

Beyond the suites the app provides:

* **Decision support and sustainability** — every result is benchmarked against published practice and the live site
  context, then turned into ranked recommendations and a scorecard; each recommendation can pull current literature.
* **Universal geometry import** — CAD, surface, mesh, drawing, GIS, point-cloud, voxel and network formats are
  read on the device, previewed, measured and routed to the suites that can use them; closed formats get an
  exact conversion instruction. Procedural geometry (spacers, lattices, minimal surfaces, packed beds) is built in.
* **Background solving** — engines run in a worker thread with progress and Cancel.
* **Live plant feed** — the monitoring suite can follow an export file on the user's computer and re-analyse as it grows.
* **Data provenance** — thermodynamic constants are checked against published databases and listed with their sources.

## How the suites are connected

One case holds the site, the feed-water analysis, and each suite's inputs and published outputs.
When a suite runs it is offered matching values from the case feed, from the *Global site data* page
and from the suites upstream of it (for example the RO concentrate becomes the brine for chemistry,
ZLD and the outfall; RO pressures size the pumps; energy, membranes and chemicals feed economics).
The *Integrated run* page solves the whole chain in data-flow order.

## Live global data

The *Global site data* page lets the user pick any point on Earth. The user's own browser then calls
open, key-less services directly: Open-Meteo (weather, solar, waves, sea-surface temperature, ocean
currents, sea level), SRTM30+ global relief via PacIOOS ERDDAP (bathymetry / topography grid),
SeaDataCloud climatology via EMODnet-Physics ERDDAP (salinity, temperature), BigDataCloud (country),
World Bank (inflation, lending rate) and an open exchange-rate API. No server belonging to this
project is involved, so the data are as fresh as the sources whenever and wherever the app is opened.

## Install and offline use

* Chrome / Edge / Android: the **Install** button in the app bar.
* iPhone / iPad: Safari → Share → *Add to Home Screen*.
* A service worker stores the whole application, so every calculation works in aeroplane mode.
* `standalone.html` is the entire application in one file — copy it anywhere and open it.

## Availability

The static build is host-independent (relative paths only). It is published on GitHub Pages. Installed
copies and the single-file edition (`standalone.html`) keep working with no host at all, so an outage of
the web address does not stop existing users. To add a second, fully independent web address, run
`tools/deploy-mirrors.sh` with a Cloudflare Pages, Netlify, GitLab or Codeberg account, then list the new
address in `js/data/app.js` so the app's *Install & offline* page checks it.

## Security

* Strict Content-Security-Policy: no inline scripts, no third-party code, no `eval`.
* No accounts, cookies, analytics or tracking; cases stay in the browser's local storage.
* Imported files are size-limited and parsed as data only; all text is rendered as plain text.
* Network requests are HTTPS-only to a fixed allow-list, without credentials or referrer.
* User formulas (suite 11) run through a built-in expression parser, never through code execution.

## Development

No dependencies and no build step are needed to run the app:

```bash
python3 -m http.server 8080      # then open http://localhost:8080
node tests/run.mjs               # regression tests for every suite
node tools/build.mjs             # stamp version.json + sw.js (and standalone.html if esbuild is installed)
```

`docs/SUITE_CONTRACT.md` describes how a suite module is written. Cost data and the bundled
electricity / grid-carbon table are indicative planning defaults; every value is editable.

## Scope and limits

The engines are engineering models intended for design studies, screening, teaching and
decision support. Each suite's *Theory* tab ticks the formulations it actually solves and lists the
rest as reference. Results for a real plant should be calibrated and validated against that plant's
data using the built-in tools before they are relied on.
