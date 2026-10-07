// Background solver thread. Runs a suite engine off the page's main thread so that the interface
// never freezes during a long calculation, and so that a run can be cancelled.
import { loadSuite } from '../suites/index.js';

self.onmessage = async (e) => {
  const { id, suiteId, values, ctx } = e.data || {};
  try {
    const suite = await loadSuite(suiteId);
    let last = 0;
    const res = await suite.run(values, {
      ...ctx,
      progress: (f, msg) => { const now = Date.now(); if (now - last > 60 || f >= 1) { last = now; self.postMessage({ id, type: 'progress', f, msg }); } },
      tick: () => Promise.resolve(), // nothing to yield to in a worker
    });
    try { self.postMessage({ id, type: 'done', res }); }
    catch { self.postMessage({ id, type: 'unclonable' }); } // result holds something that cannot cross threads: the page reruns it itself
  } catch (err) {
    self.postMessage({ id, type: 'error', message: String(err?.message || err) });
  }
};
