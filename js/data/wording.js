// Suite-specific wording for the workspace tabs and the run button, so each discipline reads in its
// own vocabulary. Order of the labels: overview, data, model settings, numerical resolution, geometry,
// results, calibration/validation, verification, theory.
const W = (run, guide, inputs, setup, mesh, geometry, results, cal, verify, theory) => ({ run, guide, inputs, setup, mesh, geometry, results, cal, verify, theory });
export const WORDING = {
  default: W('Run simulation', 'Guide', 'Inputs', 'Model setup', 'Mesh', 'Geometry', 'Results', 'Calibrate & validate', 'Verify', 'Theory'),
  ro: W('Run design', 'Overview', 'Feed & array', 'Transport & limits', 'Discretisation', 'Element geometry', 'Performance', 'Calibrate & validate', 'Verify', 'Equations'),
  chem: W('Run chemistry', 'Overview', 'Water & dosing', 'Thermodynamics', 'Resolution', 'Equipment & crystals', 'Speciation & scaling', 'Calibrate & validate', 'Verify', 'Equations'),
  plant: W('Solve flowsheet', 'Overview', 'Flowsheet data', 'Solver & off-design', 'Time step', 'Layout & topology', 'Balances & KPIs', 'Calibrate & validate', 'Verify', 'Equations'),
  cfd: W('Run CFD', 'Overview', 'Domain & flow', 'Physics & boundaries', 'Mesh', 'CAD & geometry', 'Fields & results', 'Calibrate & validate', 'Verify', 'Equations'),
  sea: W('Run plume model', 'Overview', 'Discharge & sea', 'Mixing & limits', 'Grid', 'Bathymetry & outfall', 'Plume & compliance', 'Calibrate & validate', 'Verify', 'Equations'),
  thermal: W('Run process', 'Overview', 'Process data', 'Heat-transfer models', 'Resolution', 'Equipment geometry', 'Thermal performance', 'Calibrate & validate', 'Verify', 'Equations'),
  ed: W('Run stack model', 'Overview', 'Feed & stack', 'Membranes & electrodes', 'Discretisation', 'Cell geometry', 'Stack performance', 'Calibrate & validate', 'Verify', 'Equations'),
  fomd: W('Run process', 'Overview', 'Streams & module', 'Transport models', 'Discretisation', 'Membrane structure', 'Flux & energy', 'Calibrate & validate', 'Verify', 'Equations'),
  zld: W('Run ZLD train', 'Overview', 'Brine & train', 'Thermodynamics & kinetics', 'Grids', 'Vessels & crystals', 'Water, salts & energy', 'Calibrate & validate', 'Verify', 'Equations'),
  fouling: W('Analyse log', 'Overview', 'Operating log', 'Normalisation & detection', 'Time step', 'Scans & images', 'Diagnosis & forecast', 'Fit & back-test', 'Self-checks', 'Methods'),
  opt: W('Run study', 'Overview', 'Problem data', 'Algorithm settings', 'Convergence', 'Shapes & data', 'Solutions', 'Fit & test', 'Algorithm checks', 'Methods'),
  pump: W('Run hydraulics', 'Overview', 'Duty & piping', 'Curves, drives & limits', 'Pipe reaches', 'Piping network', 'Operating point & energy', 'Calibrate & validate', 'Verify', 'Equations'),
  econ: W('Run analysis', 'Briefing', 'Costs & quantities', 'Finance & scenarios', 'Sample size', 'Layouts & asset lists', 'Cost of water & cash flow', 'Benchmark & back-cast', 'Audit checks', 'Methods'),
};
export const wording = (id) => WORDING[id] || WORDING.default;
