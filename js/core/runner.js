// Runs suite engines in a background thread when the browser allows it (served over http/https with
// module workers), and on the page itself otherwise (single-file edition, old browsers).
let worker = null, seq = 0, broken = false;
const pending = new Map();

function getWorker() {
  if (broken || typeof Worker === 'undefined' || !/^https?:$/.test(location.protocol)) return null;
  if (worker) return worker;
  try {
    worker = new Worker(new URL('./run.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const p = pending.get(e.data?.id);
      if (!p) return;
      if (e.data.type === 'progress') return p.progress(e.data.f, e.data.msg);
      pending.delete(e.data.id);
      if (e.data.type === 'done') p.resolve(e.data.res);
      else if (e.data.type === 'unclonable') p.reject(Object.assign(new Error('unclonable'), { fallback: true }));
      else p.reject(new Error(e.data.message || 'Run failed'));
    };
    worker.onerror = () => { broken = true; killWorker('fallback'); };
  } catch { broken = true; worker = null; }
  return worker;
}
function killWorker(reason) {
  try { worker?.terminate(); } catch { /* already gone */ }
  worker = null;
  for (const [, p] of pending) p.reject(Object.assign(new Error(reason === 'cancel' ? 'Run cancelled.' : 'worker unavailable'), { fallback: reason !== 'cancel', cancelled: reason === 'cancel' }));
  pending.clear();
}

/** True while a background run is in progress. */
export const isRunning = () => pending.size > 0;
/** Stop the current background run immediately. */
export function cancelRun() { if (pending.size) killWorker('cancel'); }

/**
 * Solve a suite. Returns { res, threaded }. `ctx` must be plain data plus an optional progress callback.
 */
export async function solve(suite, values, ctx) {
  const { progress = () => {}, tick, ...plain } = ctx;
  const w = getWorker();
  if (w) {
    try {
      const id = ++seq;
      const res = await new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject, progress });
        try { w.postMessage({ id, suiteId: suite.id, values, ctx: plain }); }
        catch (e) { pending.delete(id); reject(Object.assign(new Error('unclonable'), { fallback: true })); }
      });
      return { res, threaded: true };
    } catch (e) {
      if (!e.fallback) throw e; // genuine model error or cancellation
    }
  }
  return { res: await suite.run(values, { ...plain, progress, tick: tick || (() => new Promise((r) => setTimeout(r, 0))) }), threaded: false };
}
