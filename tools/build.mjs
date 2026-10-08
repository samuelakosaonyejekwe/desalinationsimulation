// Build step: stamps a content-hash version, generates the service worker pre-cache list and,
// when esbuild is available, produces standalone.html (the whole application in one file).
//   node tools/build.mjs            -> version.json + sw.js (+ standalone.html if esbuild is found)
//   ESBUILD=/path/to/esbuild/lib/main.js node tools/build.mjs
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, relative, dirname } from 'node:path';
import { gzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const walk = (dir) => readdirSync(join(root, dir)).flatMap((n) => { const p = join(dir, n); return statSync(join(root, p)).isDirectory() ? walk(p) : [p]; });
const files = ['index.html', 'manifest.webmanifest', ...walk('css'), ...walk('js'), ...walk('assets')].map((f) => f.split('\\').join('/')).sort();

const hash = createHash('sha256');
for (const f of files) { hash.update(f); hash.update(readFileSync(join(root, f))); }
hash.update(readFileSync(join(root, 'tools/sw.template.js')));
const version = hash.digest('hex').slice(0, 12);

// ---- standalone single-file edition ------------------------------------------------------------------
let standalone = false;
const esbuildPath = process.env.ESBUILD || ['node_modules/esbuild/lib/main.js'].map((p) => join(root, p)).find(existsSync);
if (esbuildPath && existsSync(esbuildPath)) {
  const esbuild = await import(pathToFileURL(esbuildPath).href);
  const out = await esbuild.build({ entryPoints: [join(root, 'js/app.js')], bundle: true, format: 'iife', minify: true, write: false, target: ['es2020'], legalComments: 'none', charset: 'utf8' });
  // The bundle is stored gzip-compressed (base64) in an inert data block and unpacked in the browser by a tiny loader,
  // which keeps the single-file edition small. The security policy lists exactly two script fingerprints: the loader and
  // the unpacked application code. The browser checks the unpacked code against its fingerprint before running it, so the
  // policy is as strict as an uncompressed build: no other script, from any source, can run.
  const bundle = out.outputFiles[0].text, packed = gzipSync(Buffer.from(bundle, 'utf8'), { level: 9 }).toString('base64');
  const loader = "(async()=>{try{const t=document.getElementById('app-gz').textContent,b=Uint8Array.from(atob(t),c=>c.charCodeAt(0)),r=new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))),e=document.createElement('script');e.textContent=await r.text();document.body.append(e)}catch(x){document.getElementById('main').textContent='This browser is too old to unpack the single-file edition (it needs DecompressionStream). Use a current Chrome, Edge, Firefox or Safari, or the web address.'}})();";
  const js = loader;
  const css = readFileSync(join(root, 'css/app.css'), 'utf8');
  const icon = 'data:image/svg+xml;base64,' + readFileSync(join(root, 'assets/icon.svg')).toString('base64');
  const sha = (s) => "'sha256-" + createHash('sha256').update(s, 'utf8').digest('base64') + "'";
  let html = readFileSync(join(root, 'index.html'), 'utf8');
  html = html
    .replace(/<link rel="manifest"[^>]*>\n/, '').replace(/<link rel="apple-touch-icon"[^>]*>\n/, '').replace(/<link rel="modulepreload"[^>]*>\n/, '')
    .replace(/<link rel="icon"[^>]*>/, `<link rel="icon" href="${icon}">`)
    .replace(/src="assets\/icon\.svg"/g, `src="${icon}"`)
    .replace('<link rel="stylesheet" href="css/app.css">', () => `<style>${css}</style>`)
    .replace('<script type="module" src="js/app.js"></script>', () => `<script type="application/octet-stream" id="app-gz">${packed}</script>\n<script>${js}</script>`)
    .replace("script-src 'self'", () => `script-src ${sha(js)} ${sha(bundle)}`).replace("style-src 'self'", () => `style-src ${sha(css)}`);
  writeFileSync(join(root, 'standalone.html'), html);
  standalone = `${(bundle.length / 1e6).toFixed(2)} MB of code packed to ${(html.length / 1e6).toFixed(2)} MB`;
}

const precache = ['./', ...files, 'version.json', ...(existsSync(join(root, 'standalone.html')) ? [] : [])];
writeFileSync(join(root, 'version.json'), JSON.stringify({ version, built: new Date().toISOString(), files: files.length }) + '\n');
writeFileSync(join(root, 'sw.js'), readFileSync(join(root, 'tools/sw.template.js'), 'utf8').replace('__VERSION__', version).replace('__FILES__', JSON.stringify(precache)));
console.log(`build ${version}: ${files.length} files pre-cached${standalone ? `, standalone.html written (${standalone})` : ' (standalone.html not rebuilt: esbuild not found)'}`);
