// Plain-language explanation of a run, shown at the top of every results tab: what the run says in a few
// sentences, what each key result means and how it is obtained, how it compares with the previous run, and which
// inputs differ from the defaults. Suites can give each KPI its own `help` text; otherwise a shared glossary is used.
import { h } from './ui.js';
import { fmt } from './num.js';

// Shared glossary, matched against the KPI label (first match wins). Each entry: [pattern, meaning].
const GLOSSARY = [
  [/specific energy|kwh\/m³|kwh per m³|sec\b/i, 'Energy used per cubic metre of product water — the main running-cost and carbon driver. Lower is better; seawater RO plants today typically use 2.5–4 kWh/m³.'],
  [/levelised cost|lcow|cost of water/i, 'Levelised cost of water: all capital and running costs spread over the water produced during the plant life, discounted to today’s money. Lower is better.'],
  [/recovery/i, 'Share of the feed (or brine) that leaves as product water. Higher recovery means less intake and less brine, but more concentrated brine and higher scaling risk.'],
  [/rejection|passage/i, 'How much of the dissolved salt the membrane holds back (rejection) or lets through (passage). Rejection = 100 % − passage.'],
  [/permeate|product (water )?(tds|salinity|quality)/i, 'The purified water. Its salinity is checked against the drinking-water or process target.'],
  [/flux/i, 'Water (or salt) passing per square metre of membrane per hour. Too high a flux accelerates fouling; too low wastes membrane area.'],
  [/concentration polari[sz]ation|cp factor|β|beta/i, 'How much saltier the water is right at the membrane surface than in the bulk flow. Values above about 1.2 raise scaling and fouling risk.'],
  [/saturation index|\bsi\b/i, 'log₁₀ of how far a mineral is above its solubility: 0 = just saturated, positive = can precipitate (scale), negative = stays dissolved.'],
  [/scaling|antiscalant/i, 'Precipitation of sparingly soluble minerals on surfaces. Limits are set by saturation indices and the antiscalant that can be dosed.'],
  [/dilution/i, 'How many times the discharged brine is mixed with seawater by the time it reaches the stated point. Higher dilution means a weaker plume.'],
  [/excess salinity|mixing.zone/i, 'How much saltier than the ambient sea the water is at that point. Regulators set a limit at the edge of the mixing zone.'],
  [/froude/i, 'Ratio of jet momentum to buoyancy. High Froude numbers give strong mixing near the outlet.'],
  [/\bgor\b|gain output ratio|performance ratio|\bpr\b/i, 'Kilograms of distilled water per kilogram of heating steam (GOR) or per 2326 kJ of heat (PR). Higher is more efficient.'],
  [/bpe|boiling.point elevation/i, 'How much hotter brine boils than pure water at the same pressure. It eats into the temperature difference available for evaporation.'],
  [/heat.transfer area|area\b/i, 'Surface needed to transfer the heat or water at the stated rate; the main capital cost of evaporators, membranes and exchangers.'],
  [/npsh/i, 'Net positive suction head: pressure margin above boiling at the pump inlet. Available NPSH must exceed the pump’s required NPSH with margin, or the pump cavitates.'],
  [/efficiency/i, 'Useful output divided by input for this device (hydraulic, isentropic, current or overall). Higher is better.'],
  [/pressure|head\b/i, 'Pressure (bar) or head (m of liquid) at the stated point. It sets pump energy and equipment ratings.'],
  [/current density|limiting current/i, 'Electric current per square metre of membrane. Above the limiting current, ions are depleted at the membrane and water splits; stacks run below it.'],
  [/purity/i, 'Share of the wanted product in the solid or liquid produced. Sales grades set a minimum.'],
  [/npv|net present value/i, 'Net present value: all future cash flows discounted to today. Positive means the project earns more than the discount rate.'],
  [/irr|internal rate/i, 'Internal rate of return: the discount rate at which the net present value is zero.'],
  [/payback/i, 'Years until cumulative cash flow turns positive.'],
  [/capex|capital/i, 'Up-front investment: equipment, construction, engineering and contingency.'],
  [/opex|operating cost|running cost/i, 'Yearly running costs: energy, chemicals, membranes, labour, maintenance and disposal.'],
  [/carbon|co₂|co2|emission/i, 'Greenhouse-gas emissions attributed to the result, mainly from the electricity and heat used.'],
  [/reynolds/i, 'Ratio of inertial to viscous forces in the flow. It decides whether the flow is laminar or turbulent and how well it mixes.'],
  [/sherwood|mass.transfer coefficient/i, 'Rate at which salt is carried away from the membrane surface. Higher means less concentration polarisation.'],
  [/wall shear|shear stress/i, 'Drag of the flowing water on a surface. Higher shear scours deposits but costs pressure.'],
  [/pressure drop|Δp|head loss/i, 'Pressure lost to friction along the flow path; it must be supplied by the pumps.'],
  [/permeability|normali[sz]ed/i, 'Membrane performance corrected to reference temperature, pressure and salinity, so that changes show fouling or damage rather than changing operating conditions.'],
  [/fouling|cleaning/i, 'Deposits that reduce membrane or heat-transfer performance over time, and the cleaning that restores it.'],
  [/temperature/i, 'Temperature at the stated point.'],
  [/salinity|tds|concentration/i, 'Dissolved salt content at the stated point.'],
  [/flow|throughput|capacity|production/i, 'Volume or mass passing per unit time at the stated point.'],
  [/power|kw\b|mw\b/i, 'Rate of energy use (electric or thermal) at the stated point.'],
  [/solids|cake|salt product|crystal/i, 'Solid salts produced, their amount, size or quality.'],
  [/residual|error|closure|balance/i, 'Numerical check: how well mass or energy balances close. It should be close to zero.'],
  [/time|duration|life|days|hours/i, 'Time quantity at the stated point.'],
];

/** Meaning of a KPI: its own help text, the suite's dictionary, or the shared glossary. */
export function meaningOf(k, suite) {
  if (k.help) return k.help;
  const own = suite?.kpiHelp && (suite.kpiHelp[k.label] || Object.entries(suite.kpiHelp).find(([p]) => k.label.toLowerCase().includes(p.toLowerCase()))?.[1]);
  if (own) return own;
  const g = GLOSSARY.find(([re]) => re.test(k.label));
  return g ? g[1] : 'Computed by this suite’s engine from the inputs; the Equations tab lists the formulations used.';
}

const statusWord = { ok: 'within limits', good: 'good', warn: 'needs attention', bad: 'outside a limit' };
const pct = (a, b) => (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a) > 1e-12 ? (100 * (b - a)) / Math.abs(a) : NaN);

/**
 * Explanation panel for a result.
 * opts = { suite, res, prev (previous result or null), fields (all field defs), defaults (object), runLabel }
 */
export function explainPanel({ suite, res, prev, fields, defaults }) {
  const kpis = res.kpis || [], warns = res.warnings || [];
  const bad = warns.filter((w) => w.level === 'bad'), att = warns.filter((w) => w.level === 'warn');
  const flagged = kpis.filter((k) => k.status === 'bad' || k.status === 'warn');
  // 1 · in plain words
  const lines = [];
  if (res.summary) lines.push(res.summary);
  lines.push(bad.length ? `${bad.length} limit${bad.length > 1 ? 's are' : ' is'} exceeded — see the red notes below and the suggested actions at the end of the page.` : att.length ? `No limit is exceeded; ${att.length} point${att.length > 1 ? 's need' : ' needs'} attention (amber notes).` : 'No limit is exceeded and no check needs attention.');
  if (flagged.length) lines.push(`Results flagged: ${flagged.map((k) => `${k.label} (${statusWord[k.status] || k.status})`).join('; ')}.`);
  // 2 · each result
  const items = kpis.map((k) => h('li', null, h('b', null, k.label + ': '), `${typeof k.value === 'number' ? fmt(k.value, k.sig || 4) : String(k.value ?? '–')}${k.unit ? ' ' + k.unit : ''} `, k.status && statusWord[k.status] ? h('span', { class: 'badge ' + (k.status === 'bad' ? 'bad' : k.status === 'warn' ? 'warn' : 'ok') }, statusWord[k.status]) : null, h('div', { class: 'note' }, meaningOf(k, suite))));
  // 3 · compared with the previous run
  let cmp = null;
  if (prev?.kpis?.length) {
    const pm = Object.fromEntries(prev.kpis.filter((k) => typeof k.value === 'number').map((k) => [k.label, k.value]));
    const moved = kpis.filter((k) => typeof k.value === 'number' && Number.isFinite(pm[k.label]) && Math.abs(pct(pm[k.label], k.value)) > 0.05).map((k) => [k, pm[k.label], pct(pm[k.label], k.value)]).sort((p, q) => Math.abs(q[2]) - Math.abs(p[2]));
    const changedIn = fields.filter((f) => f.type !== 'table' && f.type !== 'ions' && f.type !== 'file' && prev._inputs && res._inputs && JSON.stringify(prev._inputs[f.key]) !== JSON.stringify(res._inputs[f.key]));
    cmp = h('div', null, h('h4', null, 'Compared with the previous run'),
      changedIn.length ? h('p', null, `Inputs changed: ${changedIn.slice(0, 8).map((f) => `${f.label} ${fmtV(prev._inputs[f.key])} → ${fmtV(res._inputs[f.key])}${f.unit ? ' ' + f.unit : ''}`).join('; ')}${changedIn.length > 8 ? ` and ${changedIn.length - 8} more` : ''}.`) : h('p', null, 'No input changed (the same case was solved again).'),
      moved.length ? h('ul', null, moved.slice(0, 8).map(([k, a, p]) => h('li', null, `${k.label}: ${fmt(a)} → ${fmt(k.value)}${k.unit ? ' ' + k.unit : ''} (${p >= 0 ? '+' : ''}${fmt(p, 3)} %)`))) : h('p', null, 'No key result moved by more than 0.05 %.'));
  }
  // 4 · what differs from the defaults
  const changed = fields.filter((f) => !['table', 'ions', 'file'].includes(f.type) && defaults && res._inputs && f.key in defaults && JSON.stringify(defaults[f.key]) !== JSON.stringify(res._inputs[f.key]));
  const own = h('p', null, changed.length ? `Your case differs from the built-in example in ${changed.length} input${changed.length > 1 ? 's' : ''}: ${changed.slice(0, 10).map((f) => `${f.label} = ${fmtV(res._inputs[f.key])}${f.unit ? ' ' + f.unit : ''}`).join('; ')}${changed.length > 10 ? ' …' : ''}.` : 'All inputs are at the built-in example values (a realistic industrial case); change them on the input tabs to describe your own plant.');
  return h('details', { class: 'explain card-lite', open: true },
    h('summary', null, 'What these results mean'),
    h('div', { class: 'explain-body' },
      h('h4', null, 'In plain words'), h('ul', null, lines.map((l) => h('li', null, l))),
      h('h4', null, 'Each key result'), h('ul', { class: 'kpi-explain' }, items),
      cmp,
      h('h4', null, 'How these numbers were obtained'),
      h('p', null, `${suite.description || suite.tagline || ''} ${suite.equationsNote || ''}`.trim()),
      own,
      h('p', { class: 'note' }, 'The charts below carry their own notes; hover over a curve or map for exact values. The Predict tab answers “what if” questions with the same engine, and the suggested actions at the end turn these results into decisions.')));
}
const fmtV = (v) => (typeof v === 'number' ? fmt(v) : typeof v === 'boolean' ? (v ? 'on' : 'off') : String(v));
