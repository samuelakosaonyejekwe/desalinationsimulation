// Case store: one "case" holds the site, the feed-water analysis, every suite's inputs and the
// machine-readable outputs of the last run of each suite (which is what links the suites together).
// Persisted in localStorage on the user's device only; nothing is sent to any server.
import { WATERS, cloneIons } from './water.js';

const KEY = 'brinelab.case.v1', LIB = 'brinelab.cases.v1', PREF = 'brinelab.prefs.v1';
const MAX_IMPORT = 8e6; // characters
const hasLS = (() => { try { return typeof localStorage !== 'undefined' && localStorage !== null; } catch { return false; } })();
const listeners = new Map();

export function blankCase() {
  return {
    schema: 1,
    name: 'Untitled case',
    notes: '',
    created: new Date().toISOString(),
    site: { name: '', lat: null, lon: null, country: '', countryCode: '', data: {}, fetchedAt: null },
    feed: { source: 'seawater', name: WATERS.seawater.name, Q: 1000, T: 25, pH: 8.1, P: 1, ions: cloneIons(WATERS.seawater.ions), turbidity: 1, sdi: 3, toc: 1.5 },
    inputs: {},
    outputs: {},
    autolink: true,
  };
}

/** Deep-copy JSON data while dropping prototype-polluting keys and non-finite numbers. */
export function sanitize(v, depth = 0) {
  if (depth > 12) return null;
  if (v === null || typeof v === 'boolean' || typeof v === 'string') return typeof v === 'string' ? v.slice(0, 200000) : v;
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (Array.isArray(v)) return v.slice(0, 200000).map((x) => sanitize(x, depth + 1));
  if (typeof v === 'object') {
    const o = {};
    for (const k of Object.keys(v)) {
      if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
      o[k] = sanitize(v[k], depth + 1);
    }
    return o;
  }
  return null;
}

function normalise(raw) {
  const b = blankCase(), c = sanitize(raw) || {};
  const out = { ...b, ...c };
  out.site = { ...b.site, ...(c.site || {}) };
  out.feed = { ...b.feed, ...(c.feed || {}) };
  out.feed.ions = cloneIons(out.feed.ions);
  for (const k of ['Q', 'T', 'pH', 'P']) out.feed[k] = Number.isFinite(+out.feed[k]) ? +out.feed[k] : b.feed[k];
  out.inputs = c.inputs && typeof c.inputs === 'object' && !Array.isArray(c.inputs) ? c.inputs : {};
  out.outputs = c.outputs && typeof c.outputs === 'object' && !Array.isArray(c.outputs) ? c.outputs : {};
  out.name = String(out.name || 'Untitled case').slice(0, 120);
  out.notes = String(out.notes || '').slice(0, 5000);
  out.autolink = out.autolink !== false;
  return out;
}

let current = blankCase();
if (hasLS) {
  try { const s = localStorage.getItem(KEY); if (s) current = normalise(JSON.parse(s)); } catch { current = blankCase(); }
}
let saveTimer = null;
function persist() {
  if (!hasLS) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(KEY, JSON.stringify(current)); }
    catch {
      // Storage quota reached: drop bulky stored outputs and retry once.
      try { localStorage.setItem(KEY, JSON.stringify({ ...current, outputs: {} })); } catch { /* storage unavailable */ }
    }
  }, 250);
}
function emit(ev, data) { for (const fn of listeners.get(ev) || []) { try { fn(data); } catch (e) { console.error(e); } } }

export const store = {
  get case() { return current; },
  on(ev, fn) { if (!listeners.has(ev)) listeners.set(ev, new Set()); listeners.get(ev).add(fn); return () => listeners.get(ev).delete(fn); },
  update(patch) { Object.assign(current, patch); persist(); emit('case', current); },
  setFeed(patch) { current.feed = { ...current.feed, ...patch }; persist(); emit('feed', current.feed); },
  setSite(patch) { current.site = { ...current.site, ...patch }; persist(); emit('site', current.site); },
  inputs(id) { return current.inputs[id] || (current.inputs[id] = {}); },
  setInput(id, key, value) { this.inputs(id)[key] = value; persist(); },
  setInputs(id, values) { current.inputs[id] = { ...values }; persist(); },
  clearInputs(id) { delete current.inputs[id]; persist(); },
  outputs(id) { return current.outputs[id]; },
  setOutputs(id, o) { current.outputs[id] = { ...sanitize(o), _at: new Date().toISOString() }; persist(); emit('outputs', id); },
  reset() { current = blankCase(); persist(); emit('case', current); },
  exportJSON() { return JSON.stringify({ app: 'BrineLab', ...current }, null, 1); },
  importJSON(text) {
    if (typeof text !== 'string' || text.length > MAX_IMPORT) throw new Error('Case file is too large or not text.');
    const raw = JSON.parse(text);
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Not a BrineLab case file.');
    current = normalise(raw); persist(); emit('case', current);
  },
  // Named case library on this device
  library() { if (!hasLS) return {}; try { return sanitize(JSON.parse(localStorage.getItem(LIB) || '{}')) || {}; } catch { return {}; } },
  saveToLibrary() {
    const lib = this.library(); lib[current.name] = { ...current, outputs: {}, savedAt: new Date().toISOString() };
    try { localStorage.setItem(LIB, JSON.stringify(lib)); return true; } catch { return false; }
  },
  loadFromLibrary(name) { const lib = this.library(); if (lib[name]) { current = normalise(lib[name]); persist(); emit('case', current); } },
  deleteFromLibrary(name) { const lib = this.library(); delete lib[name]; try { localStorage.setItem(LIB, JSON.stringify(lib)); } catch { /* ignore */ } },
  pref(k, v) {
    if (!hasLS) return v;
    let p = {};
    try { p = JSON.parse(localStorage.getItem(PREF) || '{}') || {}; } catch { p = {}; }
    if (arguments.length === 1) return p[k];
    p[k] = v;
    try { localStorage.setItem(PREF, JSON.stringify(p)); } catch { /* ignore */ }
    return v;
  },
};
