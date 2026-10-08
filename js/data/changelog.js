// Record of changes to the calculation engines that move results. Every entry says what changed, why,
// how much typical results moved, and how to reproduce the earlier behaviour where that is possible.
// Results, reports and case files are stamped with the build that produced them, so a number can always be traced.
export const CHANGELOG = [
  { date: '2026-10-08', suites: ['chem', 'ro', 'zld', 'plant', 'opt'], title: 'Calcite and carbonate constants made consistent with the Pitzer parameter set',
    what: 'The default (Pitzer) path was combining Harvie–Møller–Weare activity parameters with calcite, aragonite and carbonate constants from the ion-pair family of databases. It now uses the constants that belong to the Harvie–Møller–Weare set.',
    effect: 'Calcite saturation index about 0.08 lower (default seawater concentrate 0.89 → 0.80); antiscalant dose 2.82 → 2.67 mg/L. Stoichiometric calcite solubility in seawater now matches the measured reference (pK*sp 6.364 against 6.368).', revert: 'Not offered: the earlier combination was an inconsistency.' },
  { date: '2026-10-08', suites: ['chem', 'ro', 'zld'], title: 'Strontium parameters from a published database; celestite constant chosen by measured solubility',
    what: 'Strontium–sulphate and strontium mixing parameters were calcium analogues; they now come from the THEREDA database. The celestite solubility constant on the Pitzer path is the value that reproduces measured celestite solubility in water, sodium-chloride solutions and seawater within about 4 %.',
    effect: 'Celestite saturation index in the default seawater concentrate −0.44 → −0.43 (after an intermediate −0.51); celestite-limited recovery without antiscalant 72.8 → 70.7 %.', revert: 'Not offered.' },
  { date: '2026-10-08', suites: ['chem'], title: 'Ammonium, iron(II) and manganese(II) interaction parameters from published databases',
    what: 'Analogue entries replaced by published Pitzer parameters, checked against NIST activity-coefficient tables.', effect: 'No change for waters without these ions.', revert: 'Not offered.' },
  { date: '2026-10-08', suites: ['fouling'], title: 'Example operating log regenerated to be physically attainable',
    what: 'The built-in example log had a concentrate pressure below the osmotic pressure at the membrane, which no membrane could produce. It is now generated from the suite\'s own channel model (feed 1 000 mg/L).',
    effect: 'All default results of the fouling suite changed with the example. Analyses of your own logs are unaffected, and the suite now flags logs that are not physically attainable.', revert: 'Not applicable: example data only.' },
  { date: '2026-10-08', suites: ['econ'], title: 'Costing scope made explicit; default staffing scaled with plant size',
    what: 'Results linked from the thermal and ZLD suites are only costed when "Plant scope being costed" includes them. Default staffing, land and laboratory budgets are scaled from the 100 000 m³/d reference plant to the entered capacity unless you have edited them.',
    effect: 'A linked 10 000 m³/d membrane plant: levelised cost 4.36 → 1.50 $/m³ (the earlier figure silently included a thermal plant and a ZLD train).', revert: 'Set the scope to "Hybrid plant with ZLD" and switch off the size scaling.' },
  { date: '2026-10-08', suites: ['ed'], title: 'Electroconvection model given a physical short-wave cut-off and solved to grid convergence',
    what: 'The over-limiting current is now extrapolated from three grids with a stated numerical uncertainty; the threshold comes from the corrected stability curve.',
    effect: 'Over-limiting current at 1.7 × threshold: grid-dependent 2.06–2.38 → 2.085 ± 1.5 %. Only affects runs with the electroconvection model selected.', revert: 'Not offered: the earlier value depended on the grid setting.' },
  { date: '2026-10-08', suites: ['fomd'], title: 'Capacitive-deionisation model extended to dilute feeds',
    what: 'Thin-layer and overlapping-layer descriptions of the electrical double layer are blended by pore size relative to the Debye length.',
    effect: 'RO + CDI polishing preset: charge efficiency 89.6 → 69.4 %, CDI energy 0.031 → 0.040 kWh/m³. Feeds that were previously refused now solve.', revert: 'Select the Gouy–Chapman–Stern layer model on the setup tab.' },
  { date: '2026-10-08', suites: ['cfd', 'sea'], title: 'Item-by-item audit; faster solvers',
    what: 'Seven theory items whose names claimed more than was solved were unticked pending real implementations; eleven were strengthened with new checks; a wall-velocity error for particles next to slip walls was corrected. Pressure and shallow-water solvers were replaced by faster ones.',
    effect: 'Default results unchanged to solver tolerance. With the full shallow-water option, area above 0.1 g/kg 62.5 → 65 ha (+4 %).', revert: 'Not applicable.' },
  { date: '2026-10-07', suites: ['pump'], title: 'Turbine recovery included in annual energy for turbine-type devices',
    what: 'Annual and lifecycle energy from the operating profile now subtract the recovered power for Pelton and reverse-running-pump cases.', effect: 'Pelton preset: 15.5 → 9.7 GWh/y.', revert: 'Not offered: the earlier figure was gross pumping energy.' },
];
/** Entries that concern one suite, newest first. */
export const changesFor = (id) => CHANGELOG.filter((c) => c.suites.includes(id));
