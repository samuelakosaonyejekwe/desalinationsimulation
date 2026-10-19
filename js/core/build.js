// Identity of the running build, used to stamp results, reports and case files.
let cached = null;
export async function buildId() {
  if (cached) return cached;
  try {
    if (/^https?:$/.test(location.protocol) && document.querySelector('link[rel=manifest]')) cached = (await (await fetch('version.json', { cache: 'no-store' })).json()).version || 'unknown';
    else cached = 'single-file';
  } catch { cached = 'unknown'; }
  return cached;
}
export const buildIdNow = () => cached || 'unknown';
/** Set the identity of the running build when it is known better (the stored copy that serves the page). */
export function setBuildId(v) { if (v) cached = v; }
