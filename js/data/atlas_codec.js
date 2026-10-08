// Decoder for the packed tables of the built-in world atlas (js/data/atlas_*.js). The build script
// (tools/atlas_build.py) stores each table as base64 text of a raw-DEFLATE stream; this module turns it back
// into typed arrays. Self-contained and synchronous so that it behaves the same in every browser and in Node.

/** base64 text -> Uint8Array */
export function b64bytes(s) {
  const b = atob(s), n = b.length, u = new Uint8Array(n);
  for (let i = 0; i < n; i++) u[i] = b.charCodeAt(i);
  return u;
}

const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/** Canonical Huffman table with a 9-bit fast lookup: entry = (symbol << 4) | length, 0 = longer code. */
function table(lens) {
  const count = new Uint16Array(16), offs = new Uint16Array(16), symbol = new Uint16Array(lens.length), fast = new Uint16Array(512);
  for (let i = 0; i < lens.length; i++) count[lens[i]]++;
  count[0] = 0;
  for (let i = 1; i < 16; i++) offs[i] = offs[i - 1] + count[i - 1];
  const next = offs.slice();
  for (let s = 0; s < lens.length; s++) if (lens[s]) symbol[next[lens[s]]++] = s;
  let code = 0, idx = 0;
  for (let len = 1; len <= 9; len++) {
    for (let k = 0; k < count[len]; k++, code++, idx++) {
      let rev = 0;
      for (let b = 0; b < len; b++) rev |= ((code >> b) & 1) << (len - 1 - b);
      for (let fill = rev; fill < 512; fill += 1 << len) fast[fill] = (symbol[idx] << 4) | len;
    }
    code <<= 1;
  }
  return { count, symbol, fast };
}

/** Inflate a raw DEFLATE stream (RFC 1951) into a Uint8Array of the known length n. */
export function inflate(src, n) {
  const out = new Uint8Array(n);
  let ip = 0, op = 0, buf = 0, cnt = 0;
  const bits = (k) => { while (cnt < k) { buf |= src[ip++] << cnt; cnt += 8; } const v = buf & ((1 << k) - 1); buf >>>= k; cnt -= k; return v; };
  const decode = (h) => {
    while (cnt < 9) { buf |= (src[ip++] || 0) << cnt; cnt += 8; }
    const f = h.fast[buf & 511];
    if (f) { const len = f & 15; buf >>>= len; cnt -= len; return f >> 4; }
    let code = 0, first = 0, index = 0;
    for (let len = 1; len < 16; len++) {
      code |= bits(1);
      const c = h.count[len];
      if (code - c < first) return h.symbol[index + (code - first)];
      index += c; first = (first + c) << 1; code <<= 1;
    }
    throw new Error('Atlas data are damaged (bad code).');
  };
  let fixed = null;
  for (let last = 0; !last;) {
    last = bits(1);
    const type = bits(2);
    if (type === 0) {
      buf = 0; cnt = 0;
      const len = src[ip] | (src[ip + 1] << 8); ip += 4;
      out.set(src.subarray(ip, ip + len), op); ip += len; op += len;
      continue;
    }
    let lt, dt;
    if (type === 1) {
      if (!fixed) { const l = new Uint8Array(288); l.fill(8, 0, 144); l.fill(9, 144, 256); l.fill(7, 256, 280); l.fill(8, 280, 288); fixed = [table(l), table(new Uint8Array(30).fill(5))]; }
      [lt, dt] = fixed;
    } else if (type === 2) {
      const nlen = bits(5) + 257, ndist = bits(5) + 1, ncode = bits(4) + 4, cl = new Uint8Array(19);
      for (let i = 0; i < ncode; i++) cl[ORDER[i]] = bits(3);
      const ct = table(cl), lens = new Uint8Array(nlen + ndist);
      for (let i = 0; i < nlen + ndist;) {
        const sym = decode(ct);
        if (sym < 16) lens[i++] = sym;
        else { const prev = sym === 16 ? lens[i - 1] : 0; let rep = sym === 16 ? 3 + bits(2) : sym === 17 ? 3 + bits(3) : 11 + bits(7); while (rep--) lens[i++] = prev; }
      }
      lt = table(lens.subarray(0, nlen)); dt = table(lens.subarray(nlen));
    } else throw new Error('Atlas data are damaged (bad block).');
    for (;;) {
      const sym = decode(lt);
      if (sym < 256) { out[op++] = sym; continue; }
      if (sym === 256) break;
      const len = LBASE[sym - 257] + bits(LEXT[sym - 257]), ds = decode(dt), dist = DBASE[ds] + bits(DEXT[ds]);
      for (let k = 0, from = op - dist; k < len; k++) out[op++] = out[from + k];
    }
  }
  if (op !== n) throw new Error('Atlas data are damaged (length).');
  return out;
}

/** Packed table { n, z } -> Uint8Array. */
export const unpack = (p) => inflate(b64bytes(p.z), p.n);
/** View helpers on an unpacked byte array (copies, so alignment never matters). */
export const asI8 = (u, o, n) => new Int8Array(u.buffer.slice(u.byteOffset + o, u.byteOffset + o + n));
export const asI16 = (u, o, n) => new Int16Array(u.buffer.slice(u.byteOffset + o, u.byteOffset + o + 2 * n));
export const asU16 = (u, o, n) => new Uint16Array(u.buffer.slice(u.byteOffset + o, u.byteOffset + o + 2 * n));
export const asI32 = (u, o, n) => new Int32Array(u.buffer.slice(u.byteOffset + o, u.byteOffset + o + 4 * n));
