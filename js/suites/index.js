// Suite registry: navigation metadata is static (so the shell renders instantly) and each engine
// is loaded on demand the first time its page is opened.
export const SUITES = [
  { id: 'ro', num: 1, title: 'Reverse Osmosis & Membrane Design', short: 'RO design', icon: '💧', blurb: 'Element-by-element array design, permeate quality, energy', load: () => import('./s01_ro.js'), uses: ['cfd', 'fouling', 'opt'] },
  { id: 'chem', num: 2, title: 'Brine Chemistry, Precipitation & Scaling', short: 'Chemistry', icon: '⚗️', blurb: 'Speciation, activity models, saturation, scaling limits', load: () => import('./s02_chem.js'), uses: ['ro'] },
  { id: 'plant', num: 3, title: 'Whole-Plant Process Simulation', short: 'Plant', icon: '🏭', blurb: 'Flowsheet mass, energy and exergy balances', load: () => import('./s03_plant.js'), uses: ['ro', 'chem', 'pump', 'thermal', 'zld'] },
  { id: 'cfd', num: 4, title: 'Flow in Membranes, Channels & Equipment', short: 'CFD', icon: '🌀', blurb: 'Navier–Stokes, species transport, spacers, CAD import', load: () => import('./s04_cfd.js'), uses: ['ro'] },
  { id: 'sea', num: 5, title: 'Brine Discharge into the Sea', short: 'Outfall', icon: '🌊', blurb: 'Dense-jet near field, far-field plume, compliance', load: () => import('./s05_sea.js'), uses: ['ro', 'zld', 'thermal'] },
  { id: 'thermal', num: 6, title: 'Thermal Desalination', short: 'Thermal', icon: '♨️', blurb: 'MED, MED-TVC, MSF, MVC stage-by-stage', load: () => import('./s06_thermal.js'), uses: ['ro'] },
  { id: 'ed', num: 7, title: 'Electrodialysis & Electrochemical Processes', short: 'Electrodialysis', icon: '⚡', blurb: 'Nernst–Planck stack model, limiting current, energy', load: () => import('./s07_ed.js'), uses: [] },
  { id: 'fomd', num: 8, title: 'Forward Osmosis, Membrane Distillation & Emerging', short: 'FO · MD', icon: '🧪', blurb: 'ICP/ECP, dusty-gas vapour transport, hybrids', load: () => import('./s08_fomd.js'), uses: ['ro'] },
  { id: 'zld', num: 9, title: 'Brine Concentration, Crystallization & ZLD', short: 'ZLD', icon: '🧂', blurb: 'Evaporation path, salts sequence, population balance', load: () => import('./s09_zld.js'), uses: ['ro', 'chem'] },
  { id: 'fouling', num: 10, title: 'Fouling & Membrane-Performance Monitoring', short: 'Fouling', icon: '📈', blurb: 'Normalisation, diagnosis, cleaning forecast', load: () => import('./s10_fouling.js'), uses: ['ro'] },
  { id: 'opt', num: 11, title: 'Optimization, AI & Custom Numerical Modelling', short: 'Optimise · AI', icon: '🧠', blurb: 'Optimisers, Pareto fronts, surrogates, custom models', load: () => import('./s11_opt.js'), uses: ['ro', 'econ', 'fouling'] },
  { id: 'pump', num: 12, title: 'Energy Recovery & Pump Calculations', short: 'Pumps · ERD', icon: '⚙️', blurb: 'System curves, NPSH, pressure exchangers, surge', load: () => import('./s12_pump.js'), uses: ['ro'] },
  { id: 'econ', num: 13, title: 'Economics & Techno-Economic Analysis', short: 'Economics', icon: '💲', blurb: 'CAPEX, OPEX, LCOW, cash flow, uncertainty', load: () => import('./s13_econ.js'), uses: ['ro', 'chem', 'plant', 'sea', 'thermal', 'zld', 'fouling', 'pump'] },
];
export const byId = (id) => SUITES.find((s) => s.id === id);
/** Order that respects the main data flow: membrane design → chemistry → hydraulics → … → economics → optimisation. */
export const CHAIN = ['ro', 'chem', 'cfd', 'pump', 'fouling', 'thermal', 'ed', 'fomd', 'zld', 'sea', 'plant', 'econ', 'opt'];
export const downstream = (id) => SUITES.filter((s) => s.uses.includes(id));

const cache = new Map();
export async function loadSuite(id) {
  if (cache.has(id)) return cache.get(id);
  const meta = byId(id);
  if (!meta) throw new Error('Unknown suite: ' + id);
  const mod = (await meta.load()).default;
  cache.set(id, mod);
  return mod;
}
