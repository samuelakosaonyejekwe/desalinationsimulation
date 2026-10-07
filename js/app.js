// Application shell: routing, navigation (including back/forward arrows), theme, install,
// offline status, update checks and background refresh of live site data.
import { h, clear, fill, toast, btn } from './core/ui.js';
import { store } from './core/store.js';
import { SUITES, byId, loadSuite, downstream } from './suites/index.js';
import { renderSuite } from './core/suiteview.js';
import { home, casePage, sitePage, portalPage, chainPage, appPage } from './pages/pages.js';
import { fetchSite } from './core/live.js';
import { APP } from './data/app.js';
import { advisorPage } from './core/advisorview.js';

const PAGES = [
  { path: 'home', label: 'Overview', icon: '🏠', render: home },
  { path: 'case', label: 'Case & feed water', icon: '🗂️', render: casePage },
  { path: 'site', label: 'Global site data', icon: '🌍', render: sitePage },
  { path: 'portal', label: 'Data portal', icon: '📥', render: portalPage },
  ...SUITES.map((s) => ({ path: 'suite/' + s.id, label: `${s.num}. ${s.short}`, full: s.title, icon: s.icon, suite: s.id })),
  { path: 'chain', label: 'Integrated run', icon: '🔗', render: chainPage },
  { path: 'advisor', label: 'Decision support', icon: '🧭', full: 'Decision support & sustainability', render: advisorPage },
  { path: 'app', label: 'Install & offline', icon: '📲', render: appPage },
];
const main = document.getElementById('main'), nav = document.getElementById('nav'), pager = document.getElementById('pager');
let cleanup = null, deferredInstall = null, swReg = null, version = '', offlineReady = false;

const app = {
  get version() { return version; },
  offlineReady: () => offlineReady,
  installed: () => window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true,
  downstream,
  async install() {
    if (app.installed()) return toast('The app is already installed on this device.', 'ok');
    if (deferredInstall) { deferredInstall.prompt(); const r = await deferredInstall.userChoice.catch(() => null); if (r?.outcome === 'accepted') toast('Installing…', 'ok'); deferredInstall = null; return; }
    location.hash = '#/app';
    toast(/iphone|ipad|ipod/i.test(navigator.userAgent) ? 'On iPhone/iPad: tap Share, then “Add to Home Screen”.' : 'Use your browser menu → “Install app” / “Add to Home screen”. Steps are listed on this page.', 'info', 9000);
  },
  async checkUpdate(manual) {
    if (!navigator.onLine) { if (manual) toast('You are offline — the installed copy keeps working.', 'warn'); return; }
    try {
      const j = await (await fetch('version.json?t=' + Date.now(), { cache: 'no-store' })).json();
      if (version && j.version && j.version !== version) { await swReg?.update(); offerReload(); }
      else if (manual) toast('You have the latest build.', 'ok');
    } catch { if (manual) toast('Could not check for updates right now.', 'warn'); }
  },
};

function offerReload() {
  if (document.getElementById('upd')) return;
  const bar = h('div', { id: 'upd', class: 'update-bar', role: 'status' }, 'A newer build is ready. ', btn('Reload now', () => { swReg?.waiting?.postMessage('skipWaiting'); setTimeout(() => location.reload(), 300); }, 'mini primary'), btn('Later', () => bar.remove(), 'mini ghost'));
  document.body.append(bar);
}

// ---- navigation --------------------------------------------------------------------------------------
const current = () => { const p = location.hash.replace(/^#\/?/, '') || 'home'; return PAGES.find((x) => x.path === p) ? p : 'home'; };
function buildNav() {
  const group = (title, items) => h('div', { class: 'nav-group' }, h('div', { class: 'nav-title' }, title), items.map((p) => h('a', { href: '#/' + p.path, class: 'nav-link', dataset: { path: p.path }, title: p.full || p.label }, h('span', { class: 'nav-ico', 'aria-hidden': 'true' }, p.icon), h('span', { class: 'nav-label' }, p.label), p.suite ? h('span', { class: 'nav-dot', dataset: { suite: p.suite }, title: 'Solved in this case' }) : null)));
  fill(nav, group('Start', PAGES.slice(0, 4)), group('Simulation suites', PAGES.filter((p) => p.suite)), group('Whole case', PAGES.slice(-3)));
  markSolved();
}
function markSolved() { for (const d of nav.querySelectorAll('.nav-dot')) d.classList.toggle('on', !!store.case.outputs[d.dataset.suite]); }

async function route() {
  const path = current(), i = PAGES.findIndex((p) => p.path === path), page = PAGES[i];
  if (cleanup) { try { cleanup(); } catch { /* ignore */ } cleanup = null; }
  for (const a of nav.querySelectorAll('.nav-link')) { const on = a.dataset.path === path; a.classList.toggle('on', on); if (on) { a.setAttribute('aria-current', 'page'); a.scrollIntoView({ block: 'nearest' }); } else a.removeAttribute('aria-current'); }
  document.body.classList.remove('nav-open');
  document.title = `${page.full || page.label} · ${APP.name}`;
  const prev = PAGES[i - 1], next = PAGES[i + 1];
  const arrow = (p, dir) => (p ? h('a', { class: 'pager-btn ' + dir, href: '#/' + p.path, rel: dir === 'prev' ? 'prev' : 'next', 'aria-label': (dir === 'prev' ? 'Previous page: ' : 'Next page: ') + p.label }, h('span', { class: 'pg-arrow', 'aria-hidden': 'true' }, dir === 'prev' ? '←' : '→'), h('span', { class: 'pg-text' }, h('small', null, dir === 'prev' ? 'Previous' : 'Next'), h('b', null, `${p.icon} ${p.label}`))) : h('span', { class: 'pager-btn disabled ' + dir }));
  fill(pager, arrow(prev, 'prev'), h('span', { class: 'pager-pos' }, `${i + 1} / ${PAGES.length}`), arrow(next, 'next'));
  const set = (id, p) => { const el = document.getElementById(id); el.disabled = !p; el.title = p ? p.label + '  (Alt + ' + (id === 'navPrev' ? '←' : '→') + ')' : ''; el.onclick = () => p && (location.hash = '#/' + p.path); };
  set('navPrev', prev); set('navNext', next);
  document.getElementById('crumb').textContent = page.full || page.label;
  clear(main); main.scrollTop = 0; window.scrollTo(0, 0);
  if (page.suite) {
    main.append(h('div', { class: 'loading' }, h('div', { class: 'spinner' }), `Loading ${byId(page.suite).title}…`));
    try {
      const suite = await loadSuite(page.suite);
      if (current() !== path) return;
      const host = h('div', { class: 'suite' }); fill(main, host);
      cleanup = renderSuite(suite, host, app);
    } catch (e) {
      console.error(e);
      fill(main, h('div', { class: 'empty' }, h('h3', null, 'This suite could not be loaded'), h('p', null, navigator.onLine ? String(e.message || e) : 'You are offline and this suite has not been stored on the device yet. Connect once and it will be saved for offline use.'), btn('Try again', route, 'primary')));
    }
  } else {
    const host = h('div', { class: 'page' }); main.append(host);
    try { page.render(host, app); } catch (e) { console.error(e); host.append(h('p', { class: 'bad' }, 'Page error: ' + e.message)); }
  }
  main.focus({ preventScroll: true });
}

// ---- theme, connection, install -----------------------------------------------------------------------
function applyTheme(t) { if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; document.querySelector('meta[name=theme-color]')?.setAttribute('content', getComputedStyle(document.documentElement).getPropertyValue('--bar').trim() || '#0b3a5b'); }
function setupChrome() {
  applyTheme(store.pref('theme'));
  document.getElementById('themeBtn').addEventListener('click', () => { const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; const t = dark ? 'light' : 'dark'; store.pref('theme', t); applyTheme(t); window.dispatchEvent(new Event('resize')); for (const f of document.querySelectorAll('figure.plot')) f._redraw?.(); });
  document.getElementById('menuBtn').addEventListener('click', () => document.body.classList.toggle('nav-open'));
  document.getElementById('scrim').addEventListener('click', () => document.body.classList.remove('nav-open'));
  document.getElementById('installBtn').addEventListener('click', () => app.install());
  const net = document.getElementById('net');
  const paintNet = () => { net.textContent = navigator.onLine ? 'Online' : 'Offline'; net.className = 'net ' + (navigator.onLine ? 'on' : 'off'); net.title = navigator.onLine ? 'Connected: live data and update checks are available.' : 'No connection: all calculations still work from this device.'; };
  window.addEventListener('online', () => { paintNet(); toast('Back online.', 'ok'); refreshSite(); app.checkUpdate(false); });
  window.addEventListener('offline', () => { paintNet(); toast('You are offline — everything keeps working from this device.', 'warn'); });
  paintNet();
  window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); deferredInstall = e; document.getElementById('installBtn').classList.add('ready'); });
  window.addEventListener('appinstalled', () => { deferredInstall = null; document.getElementById('installBtn').hidden = true; toast(`${APP.name} is installed.`, 'ok'); });
  if (app.installed()) document.getElementById('installBtn').hidden = true;
  document.addEventListener('keydown', (e) => { if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) { const b = document.getElementById(e.key === 'ArrowLeft' ? 'navPrev' : 'navNext'); if (!b.disabled) { e.preventDefault(); b.click(); } } if (e.key === 'Escape') document.body.classList.remove('nav-open'); });
  const caseChip = document.getElementById('caseChip'), paintCase = () => (caseChip.textContent = store.case.name);
  store.on('case', paintCase); store.on('outputs', markSolved); store.on('case', markSolved); paintCase();
}

/** Refresh stored site data in the background when they are older than six hours. */
async function refreshSite() {
  const s = store.case.site;
  if (!navigator.onLine || s.lat === null || s.lon === null || !s.fetchedAt) return;
  if (Date.now() - new Date(s.fetchedAt) < 6 * 3600e3) return;
  try { const fresh = await fetchSite(s.lat, s.lon); if (Object.values(fresh.status).some((x) => x.ok)) { store.setSite({ ...fresh, data: { ...s.data, ...fresh.data }, name: fresh.name || s.name }); if (current() === 'site') route(); } } catch { /* stay on stored data */ }
}

async function setupServiceWorker() {
  try { version = (await (await fetch('version.json', { cache: 'no-store' })).json()).version || ''; } catch { version = ''; }
  if (!('serviceWorker' in navigator) || !/^https?:$/.test(location.protocol)) return;
  try {
    swReg = await navigator.serviceWorker.register('sw.js');
    navigator.serviceWorker.addEventListener('message', (e) => { if (e.data?.type === 'offline-ready') offlineReady = true; if (e.data?.type === 'version' && !version) version = e.data.version; });
    navigator.serviceWorker.ready.then((r) => { r.active?.postMessage('status'); r.periodicSync?.register('refresh', { minInterval: 6 * 3600e3 }).catch(() => {}); });
    swReg.addEventListener('updatefound', () => { const w = swReg.installing; w?.addEventListener('statechange', () => { if (w.state === 'installed' && navigator.serviceWorker.controller) offerReload(); }); });
    if (swReg.waiting && navigator.serviceWorker.controller) offerReload();
  } catch (e) { console.warn('Service worker unavailable', e); }
}

buildNav(); setupChrome();
window.addEventListener('hashchange', route);
route();
setupServiceWorker();
setTimeout(refreshSite, 4000);
setInterval(() => { app.checkUpdate(false); refreshSite(); }, 30 * 60e3);
document.addEventListener('visibilitychange', () => { if (!document.hidden) { app.checkUpdate(false); refreshSite(); } });
