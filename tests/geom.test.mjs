// Geometry-import engine tests: node tests/geom.test.mjs
// Every reader gets a small file synthesised in memory; the analysis helpers and generators are checked on known shapes.
import zlib from 'node:zlib';
import { FORMATS, PATHWAYS, SUITE_GEOMETRY, formatOf, importGeometry, sectionOf, maskOf, gridOf, microstructure, networkSummary, dimensions, generate } from '../js/core/geom.js';

let pass = 0, fails = 0;
const t0 = Date.now();
const ok = (cond, msg) => { if (cond) { pass++; console.log('ok   ' + msg); } else { fails++; console.log('FAIL ' + msg); } };
const near = (a, b, tol, msg) => ok(typeof a === 'number' && Math.abs(a - b) <= tol, `${msg} (got ${a}, expected ${b} ± ${tol})`);
const enc = new TextEncoder();
const bytesOf = (d) => (typeof d === 'string' ? enc.encode(d) : d instanceof Uint8Array ? new Uint8Array(d) : new Uint8Array(d));
const F = (name, data) => { const u8 = bytesOf(data), ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength); return { name, size: u8.length, arrayBuffer: async () => ab, text: async () => new TextDecoder().decode(u8) }; };
const SAMPLES = [];
const imp = (name, data, opts) => { SAMPLES.push([name, bytesOf(data), opts]); return importGeometry(F(name, data), opts); };
async function group(label, fn) { try { await fn(); } catch (e) { fails++; console.log(`FAIL ${label} — threw: ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`); } }
async function throws(label, fn, re) {
  try { await fn(); ok(false, `${label}: should have thrown`); }
  catch (e) { ok(e instanceof Error && e.message.length > 0 && e.message.length < 500 && (!re || re.test(e.message)), `${label}: clean error "${String(e && e.message).slice(0, 70)}"`); }
}
const cat = (...parts) => { const bs = parts.map(bytesOf), out = new Uint8Array(bs.reduce((s, b) => s + b.length, 0)); let o = 0; for (const b of bs) { out.set(b, o); o += b.length; } return out; };
const le = (type, vals, little = true) => { const sz = { u8: 1, i8: 1, u16: 2, i16: 2, u32: 4, i32: 4, f32: 4, f64: 8, u64: 8 }[type], b = new Uint8Array(sz * vals.length), dv = new DataView(b.buffer); vals.forEach((v, i) => { const o = i * sz; if (type === 'u8') dv.setUint8(o, v); else if (type === 'i8') dv.setInt8(o, v); else if (type === 'u16') dv.setUint16(o, v, little); else if (type === 'i16') dv.setInt16(o, v, little); else if (type === 'u32') dv.setUint32(o, v, little); else if (type === 'i32') dv.setInt32(o, v, little); else if (type === 'f32') dv.setFloat32(o, v, little); else if (type === 'u64') dv.setBigUint64(o, BigInt(v), little); else dv.setFloat64(o, v, little); }); return b; };
const be = (type, vals) => le(type, vals, false);
const b64 = (u8) => Buffer.from(u8).toString('base64');

// stored / deflated zip writer
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = (u8) => { let c = 0xffffffff; for (const b of u8) c = CRC[(c ^ b) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function zip(entries, deflate = false) {
  const locals = [], central = [];
  let off = 0;
  for (const [name, data] of entries) {
    const raw = bytesOf(data), nm = enc.encode(name), body = deflate ? new Uint8Array(zlib.deflateRawSync(raw)) : raw, m = deflate ? 8 : 0;
    const lh = cat(le('u32', [0x04034b50]), le('u16', [20, 0, m, 0, 0]), le('u32', [crc32(raw), body.length, raw.length]), le('u16', [nm.length, 0]), nm, body);
    central.push(cat(le('u32', [0x02014b50]), le('u16', [20, 20, 0, m, 0, 0]), le('u32', [crc32(raw), body.length, raw.length]), le('u16', [nm.length, 0, 0, 0, 0]), le('u32', [0, off]), nm));
    locals.push(lh); off += lh.length;
  }
  const cd = cat(...central);
  return cat(...locals, cd, le('u32', [0x06054b50]), le('u16', [0, 0, entries.length, entries.length]), le('u32', [cd.length, off]), le('u16', [0]));
}

// unit cube: 8 vertices, 6 outward quads, 12 outward triangles, and a 6-tetrahedron split
const V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const Q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
const T = Q.flatMap((q) => [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
const TETS = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]];
const vtxt = (sep = ' ') => V.map((v) => v.join(sep)).join('\n');
function isCube(g, label, { count = 12, lo = [0, 0, 0], hi = [1, 1, 1], area = 6, volume = 1 } = {}) {
  ok(g.kind === 'mesh' && g.count === count && g.triangles.length === count * 9 && Array.isArray(g.triangles), `${label}: mesh with ${count} triangles (got ${g.kind}, ${g.count})`);
  ok(g.bbox.min.every((v, k) => Math.abs(v - lo[k]) < 1e-6) && g.bbox.max.every((v, k) => Math.abs(v - hi[k]) < 1e-6), `${label}: bbox ${JSON.stringify(g.bbox.min)}–${JSON.stringify(g.bbox.max)}`);
  const d = dimensions(g);
  ok(Math.abs(d.area - area) < 1e-6 && d.closed && Math.abs(d.volume - volume) < 1e-6 && d.consistentNormals, `${label}: area ${d.area}, closed ${d.closed}, volume ${d.volume}, consistent normals ${d.consistentNormals}`);
  ok(typeof g.name === 'string' && typeof g.format === 'string' && Array.isArray(g.warnings) && g.stats && typeof g.pathway === 'string', `${label}: carries name / format "${g.format}" / pathway "${g.pathway}"`);
}

// ---------------------------------------------------------------------------------------------------------
await group('registry', async () => {
  ok(Array.isArray(FORMATS) && FORMATS.length > 60 && FORMATS.every((f) => Array.isArray(f.ext) && f.ext.length && f.name && f.pathway && ['full', 'partial', 'convert'].includes(f.support) && typeof f.note === 'string' && (f.support !== 'convert' || (typeof f.convert === 'string' && f.convert.length > 20))), 'FORMATS entries are well formed');
  ok(FORMATS.every((f) => PATHWAYS[f.pathway]) && ['cad', 'surface', 'mesh', 'drawing', 'gis', 'points', 'voxel', 'network', 'numeric'].every((p) => PATHWAYS[p] && PATHWAYS[p].title && PATHWAYS[p].blurb), 'every format pathway is described in PATHWAYS');
  const ids = ['ro', 'chem', 'plant', 'cfd', 'sea', 'thermal', 'ed', 'fomd', 'zld', 'fouling', 'opt', 'pump', 'econ'];
  ok(ids.every((s) => SUITE_GEOMETRY[s] && SUITE_GEOMETRY[s].classes.length > 10 && SUITE_GEOMETRY[s].accepts.length) && Object.keys(SUITE_GEOMETRY).length === 13, 'SUITE_GEOMETRY covers the 13 suites');
  ok(formatOf('a.STEP').name === 'STEP (ISO 10303)' && formatOf('scan.nii.gz').name === 'NIfTI-1' && formatOf('part.x_t').support === 'convert' && formatOf('noext') === null && formatOf('a.unknownext') === null && formatOf('polyMesh/points').name === 'OpenFOAM polyMesh', 'formatOf handles case, double extensions, underscores and unknown names');
  const convExt = FORMATS.filter((f) => f.support === 'convert').flatMap((f) => f.ext), readExt = FORMATS.filter((f) => f.support !== 'convert').flatMap((f) => f.ext);
  ok(convExt.every((e) => !readExt.includes(e)), 'no extension is both readable and convert-only');
});

await group('surface formats', async () => {
  const facet = (t) => `facet normal 0 0 0\nouter loop\n${t.map((i) => 'vertex ' + V[i].join(' ')).join('\n')}\nendloop\nendfacet`;
  isCube(await imp('cube.stl', `solid cube\n${T.map(facet).join('\n')}\nendsolid cube\n`), 'STL ascii');
  const bin = new Uint8Array(84 + 50 * 12), dv = new DataView(bin.buffer);
  dv.setUint32(80, 12, true);
  T.forEach((t, i) => t.forEach((vi, k) => V[vi].forEach((c, q) => dv.setFloat32(84 + i * 50 + 12 + k * 12 + q * 4, c, true))));
  isCube(await imp('cube.stl', bin), 'STL binary');
  isCube(await imp('padded.stl', cat(bin, new Uint8Array(2))), 'STL binary with trailing padding bytes');
  isCube(await imp('cube.obj', `# cube\n${V.map((v) => 'v ' + v.join(' ')).join('\n')}\n${Q.map((q) => 'f ' + q.map((i) => i + 1).join(' ')).join('\n')}\n`), 'OBJ');
  const plyHead = (fmt) => `ply\nformat ${fmt} 1.0\ncomment test\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nelement face 6\nproperty list uchar int vertex_indices\nend_header\n`;
  isCube(await imp('cube.ply', plyHead('ascii') + V.map((v) => v.join(' ') + ' 255').join('\n') + '\n' + Q.map((q) => '4 ' + q.join(' ')).join('\n') + '\n'), 'PLY ascii');
  for (const little of [true, false]) {
    const body = cat(...V.map((v) => cat(le('f32', v, little), le('u8', [200]))), ...Q.map((q) => cat(le('u8', [4]), le('i32', q, little))));
    isCube(await imp('cube.ply', cat(plyHead(little ? 'binary_little_endian' : 'binary_big_endian'), body)), `PLY binary ${little ? 'little' : 'big'} endian`);
  }
  const pc = await imp('pts.ply', 'ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\nproperty double y\nproperty double z\nend_header\n0 0 0\n1 2 3\n4 5 6\n');
  ok(pc.kind === 'points' && pc.count === 3 && pc.points[5] === 3 && pc.bbox.max[2] === 6, 'PLY without faces → point cloud');
  isCube(await imp('cube.off', `OFF\n# comment\n8 6 12\n${vtxt()}\n${Q.map((q) => '4 ' + q.join(' ')).join('\n')}\n`), 'OFF');
  const model = `<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources><object id="1" type="model"><mesh><vertices>${V.map((v) => `<vertex x="${v[0]}" y="${v[1]}" z="${v[2]}"/>`).join('')}</vertices><triangles>${T.map((t) => `<triangle v1="${t[0]}" v2="${t[1]}" v3="${t[2]}"/>`).join('')}</triangles></mesh></object></resources><build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 10 20 30"/></build></model>`;
  isCube(await imp('cube.3mf', zip([['[Content_Types].xml', '<Types/>'], ['3D/3dmodel.model', model]])), '3MF (stored zip, build transform)', { lo: [10, 20, 30], hi: [11, 21, 31] });
  isCube(await imp('cube.3mf', zip([['3D/3dmodel.model', model]], true)), '3MF (deflated zip)', { lo: [10, 20, 30], hi: [11, 21, 31] });
  const amf = `<?xml version="1.0"?><amf unit="millimeter"><object id="0"><mesh><vertices>${V.map((v) => `<vertex><coordinates><x>${v[0]}</x><y>${v[1]}</y><z>${v[2]}</z></coordinates></vertex>`).join('')}</vertices><volume>${T.map((t) => `<triangle><v1>${t[0]}</v1><v2>${t[1]}</v2><v3>${t[2]}</v3></triangle>`).join('')}</volume></mesh></object></amf>`;
  isCube(await imp('cube.amf', amf), 'AMF xml');
  isCube(await imp('cube.amf', zip([['cube.amf', amf]])), 'AMF zip-wrapped');
  const ci = Q.map((q) => q.join(' ') + ' -1').join(' ');
  isCube(await imp('cube.wrl', `#VRML V2.0 utf8\n# a cube\nShape { geometry IndexedFaceSet { coordIndex [ ${Q.map((q) => q.join(', ') + ', -1').join(', ')} ]\n coord Coordinate { point [ ${V.map((v) => v.join(' ')).join(', ')} ] } } }\n`), 'VRML 2 IndexedFaceSet');
  isCube(await imp('cube.iv', `#Inventor V2.1 ascii\nSeparator { Coordinate3 { point [ ${V.map((v) => v.join(' ')).join(', ')} ] } IndexedFaceSet { coordIndex [ ${Q.map((q) => q.join(', ') + ', -1').join(', ')} ] } }\n`), 'Open Inventor ascii');
  isCube(await imp('cube.x3d', `<?xml version="1.0"?><!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" "http://www.web3d.org/specifications/x3d-3.3.dtd"><X3D><Scene><Shape><IndexedFaceSet coordIndex="${ci}"><Coordinate point="${V.map((v) => v.join(' ')).join(', ')}"/></IndexedFaceSet></Shape></Scene></X3D>`), 'X3D');
  const daeSrc = `<source id="pos"><float_array id="pos-a" count="24">${V.flat().join(' ')}</float_array><technique_common><accessor source="#pos-a" count="8" stride="3"/></technique_common></source><source id="nrm"><float_array id="nrm-a" count="3">0 0 1</float_array></source><vertices id="vtx"><input semantic="POSITION" source="#pos"/></vertices>`;
  const dae = (prim) => `<?xml version="1.0"?><COLLADA version="1.4.1"><asset><unit meter="0.01" name="cm"/><up_axis>Z_UP</up_axis></asset><library_geometries><geometry id="g"><mesh>${daeSrc}${prim}</mesh></geometry></library_geometries></COLLADA>`;
  isCube(await imp('cube.dae', dae(`<triangles count="12"><input semantic="VERTEX" source="#vtx" offset="0"/><input semantic="NORMAL" source="#nrm" offset="1"/><p>${T.flat().map((i) => i + ' 0').join(' ')}</p></triangles>`)), 'COLLADA triangles (2 inputs)');
  isCube(await imp('cube.dae', dae(`<polylist count="6"><input semantic="VERTEX" source="#vtx" offset="0"/><vcount>4 4 4 4 4 4</vcount><p>${Q.flat().join(' ')}</p></polylist>`)), 'COLLADA polylist');
  // glTF: positions (float32) + indices (uint16) in one buffer
  const gbin = cat(le('f32', V.flat()), le('u16', T.flat()));
  const gj = (uri, extra = {}) => ({ asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, ...extra }], meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }], buffers: [uri ? { uri, byteLength: gbin.length } : { byteLength: gbin.length }], bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 96 }, { buffer: 0, byteOffset: 96, byteLength: 72 }], accessors: [{ bufferView: 0, componentType: 5126, count: 8, type: 'VEC3' }, { bufferView: 1, componentType: 5123, count: 36, type: 'SCALAR' }] });
  isCube(await imp('cube.gltf', JSON.stringify(gj('data:application/octet-stream;base64,' + b64(gbin)))), 'glTF embedded base64');
  isCube(await imp('cube.gltf', JSON.stringify(gj('data:application/octet-stream;base64,' + b64(gbin), { translation: [5, 0, 0], scale: [2, 2, 2] }))), 'glTF node translation + scale', { lo: [5, 0, 0], hi: [7, 2, 2], area: 24, volume: 8 });
  isCube(await imp('cube.gltf', JSON.stringify(gj('cube.bin')), { companion: { 'cube.bin': F('cube.bin', gbin) } }), 'glTF with companion .bin');
  await throws('glTF without its external buffer', () => imp('cube.gltf', JSON.stringify(gj('cube.bin'))), /companion|\.glb/);
  let js = enc.encode(JSON.stringify(gj(null, { rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2] })));
  js = cat(js, ' '.repeat((4 - (js.length % 4)) % 4));
  const glb = cat(le('u32', [0x46546c67, 2, 12 + 8 + js.length + 8 + gbin.length]), le('u32', [js.length, 0x4e4f534a]), js, le('u32', [gbin.length, 0x004e4942]), gbin);
  isCube(await imp('cube.glb', glb), 'GLB binary, node rotated 90° about z', { lo: [-1, 0, 0], hi: [0, 1, 1] });
  // VTK legacy
  const vtkHead = (kind, mode = 'ASCII') => `# vtk DataFile Version 3.0\ncube\n${mode}\nDATASET ${kind}\n`;
  isCube(await imp('cube.vtk', vtkHead('POLYDATA') + `POINTS 8 float\n${vtxt()}\nPOLYGONS 6 30\n${Q.map((q) => '4 ' + q.join(' ')).join('\n')}\n`), 'VTK legacy ASCII POLYDATA');
  isCube(await imp('pv.vtk', vtkHead('POLYDATA') + `FIELD FieldData 1\nTIME 1 1 double\n0.5\nPOINTS 8 float\n${vtxt()}\n\nMETADATA\nINFORMATION 2\nNAME L2_NORM_RANGE LOCATION vtkDataArray\nDATA 2 0 1.73205\nNAME L2_NORM_FINITE_RANGE LOCATION vtkDataArray\nDATA 2 0 1.73205\n\nPOLYGONS 6 30\n${Q.map((q) => '4 ' + q.join(' ')).join('\n')}\n\nPOINT_DATA 8\nSCALARS p float\nLOOKUP_TABLE default\n0 1 2 3 4 5 6 7\n`), 'VTK legacy with FIELD and METADATA blocks (ParaView style)');
  isCube(await imp('cube.vtk', cat(vtkHead('POLYDATA', 'BINARY'), 'POINTS 8 float\n', be('f32', V.flat()), '\nPOLYGONS 6 30\n', be('i32', Q.flatMap((q) => [4, ...q])), '\n')), 'VTK legacy BINARY POLYDATA');
  const ug = await imp('hex.vtk', vtkHead('UNSTRUCTURED_GRID') + `POINTS 8 double\n${vtxt()}\nCELLS 1 9\n8 0 1 2 3 4 5 6 7\nCELL_TYPES 1\n12\n`);
  isCube(ug, 'VTK legacy UNSTRUCTURED_GRID hexahedron → boundary');
  ok(ug.stats.cells === 1 && ug.stats.nodes === 8 && ug.stats.cellTypes.hex === 1 && ug.stats.dimension === 3, 'VTK volume-mesh stats (nodes, cells, cellTypes, dimension)');
  isCube(await imp('cube51.vtk', `# vtk DataFile Version 5.1\ncube\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 8 float\n${vtxt()}\nCELLS 7 24\nOFFSETS vtktypeint64\n0 4 8 12 16 20 24\nCONNECTIVITY vtktypeint64\n${TETS.flat().join(' ')}\nCELL_TYPES 6\n10 10 10 10 10 10\n`), 'VTK legacy 5.1 offsets/connectivity, six tetrahedra');
  const sp = await imp('img.vtk', vtkHead('STRUCTURED_POINTS') + 'DIMENSIONS 3 2 2\nORIGIN 0 0 0\nSPACING 0.5 0.5 0.5\nPOINT_DATA 12\nSCALARS phase unsigned_char 1\nLOOKUP_TABLE default\n1 0 0 1 0 0 1 0 0 1 0 0\n');
  ok(sp.kind === 'voxels' && sp.voxels.nx === 3 && sp.voxels.ny === 2 && sp.voxels.nz === 2 && sp.voxels.data[0] === 1 && sp.voxels.data[1] === 0 && Math.abs(sp.stats.porosity - 2 / 3) < 1e-12 && sp.voxels.spacing[0] === 0.5, 'VTK STRUCTURED_POINTS → voxels');
  isCube(await imp('grid.vtk', vtkHead('RECTILINEAR_GRID') + 'DIMENSIONS 3 2 2\nX_COORDINATES 3 float\n0 0.5 1\nY_COORDINATES 2 float\n0 1\nZ_COORDINATES 2 float\n0 1\n'), 'VTK RECTILINEAR_GRID → boundary', { count: 20 });
  // VTK XML
  const vtp = `<?xml version="1.0"?><VTKFile type="PolyData" version="0.1" byte_order="LittleEndian"><PolyData><Piece NumberOfPoints="8" NumberOfPolys="6"><Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">${V.flat().join(' ')}</DataArray></Points><Polys><DataArray type="Int32" Name="connectivity" format="ascii">${Q.flat().join(' ')}</DataArray><DataArray type="Int32" Name="offsets" format="ascii">4 8 12 16 20 24</DataArray></Polys></Piece></PolyData></VTKFile>`;
  isCube(await imp('cube.vtp', vtp), 'VTP ascii');
  const hdr = (n, t = 'u32') => le(t, [n]);
  const vtu = (attrs, pts, con, off, typ, tail = '') => `<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" ${attrs}><UnstructuredGrid><Piece NumberOfPoints="8" NumberOfCells="6"><Points>${pts}</Points><Cells>${con}${off}${typ}</Cells></Piece></UnstructuredGrid>${tail}</VTKFile>`;
  const da = (type, name, fmt, body, extra = '') => `<DataArray type="${type}" Name="${name}" ${name === 'Points' ? 'NumberOfComponents="3" ' : ''}format="${fmt}" ${extra}>${body}</DataArray>`;
  const P8 = le('f64', V.flat()), C8 = le('u64', TETS.flat()), O8 = le('u64', [4, 8, 12, 16, 20, 24]), Ty = le('u8', [10, 10, 10, 10, 10, 10]);
  const raw = (b, sep = true, ht = 'u32') => (sep ? b64(hdr(b.length, ht)) + b64(b) : b64(cat(hdr(b.length, ht), b)));
  isCube(await imp('cube.vtu', vtu('', da('Float64', 'Points', 'ascii', V.flat().join(' ')), da('Int64', 'connectivity', 'ascii', TETS.flat().join(' ')), da('Int64', 'offsets', 'ascii', '4 8 12 16 20 24'), da('UInt8', 'types', 'ascii', '10 10 10 10 10 10'))), 'VTU ascii (six tetrahedra)');
  isCube(await imp('cube.vtu', vtu('header_type="UInt32"', da('Float64', 'Points', 'binary', raw(P8)), da('Int64', 'connectivity', 'binary', raw(C8)), da('Int64', 'offsets', 'binary', raw(O8)), da('UInt8', 'types', 'binary', raw(Ty)))), 'VTU inline base64 (header encoded separately)');
  isCube(await imp('cube.vtu', vtu('header_type="UInt64"', da('Float64', 'Points', 'binary', raw(P8, false, 'u64')), da('Int64', 'connectivity', 'binary', raw(C8, false, 'u64')), da('Int64', 'offsets', 'binary', raw(O8, false, 'u64')), da('UInt8', 'types', 'binary', raw(Ty, false, 'u64')))), 'VTU inline base64 (UInt64 header, one stream)');
  const zl = (b) => { const c = new Uint8Array(zlib.deflateSync(b)); return b64(le('u32', [1, b.length, b.length, c.length])) + b64(c); };
  isCube(await imp('cube.vtu', vtu('compressor="vtkZLibDataCompressor"', da('Float64', 'Points', 'binary', zl(P8)), da('Int64', 'connectivity', 'binary', zl(C8)), da('Int64', 'offsets', 'binary', zl(O8)), da('UInt8', 'types', 'binary', zl(Ty)))), 'VTU inline base64, zlib-compressed');
  const blocks = [P8, C8, O8, Ty].map((b) => cat(hdr(b.length), b)), offs = blocks.map((_, k) => blocks.slice(0, k).reduce((s, b) => s + b.length, 0));
  const app = vtu('', da('Float64', 'Points', 'appended', '', `offset="${offs[0]}"`), da('Int64', 'connectivity', 'appended', '', `offset="${offs[1]}"`), da('Int64', 'offsets', 'appended', '', `offset="${offs[2]}"`), da('UInt8', 'types', 'appended', '', `offset="${offs[3]}"`), '<AppendedData encoding="raw">_@@@</AppendedData>');
  const [a1, a2] = app.split('@@@');
  isCube(await imp('cube.vtu', cat(a1, ...blocks, a2)), 'VTU appended raw data');
  const b64blocks = blocks.map((b) => b64(b)), boffs = b64blocks.map((_, k) => b64blocks.slice(0, k).reduce((s, b) => s + b.length, 0));
  isCube(await imp('cube.vtu', vtu('', da('Float64', 'Points', 'appended', '', `offset="${boffs[0]}"`), da('Int64', 'connectivity', 'appended', '', `offset="${boffs[1]}"`), da('Int64', 'offsets', 'appended', '', `offset="${boffs[2]}"`), da('UInt8', 'types', 'appended', '', `offset="${boffs[3]}"`), `<AppendedData encoding="base64">_${b64blocks.join('')}</AppendedData>`)), 'VTU appended base64 data');
  const vti = await imp('img.vti', `<VTKFile type="ImageData" byte_order="LittleEndian"><ImageData WholeExtent="0 2 0 1 0 1" Origin="0 0 0" Spacing="2 2 2"><Piece Extent="0 2 0 1 0 1"><PointData Scalars="s"><DataArray type="UInt8" Name="s" format="ascii">0 0 1 0 0 1 0 0 1 0 0 1</DataArray></PointData></Piece></ImageData></VTKFile>`);
  ok(vti.kind === 'voxels' && vti.voxels.nx === 3 && vti.voxels.data[2] === 1 && vti.voxels.spacing[1] === 2 && Math.abs(vti.stats.porosity - 2 / 3) < 1e-12, 'VTI ImageData → voxels');
  isCube(await imp('grid.vts', `<VTKFile type="StructuredGrid" byte_order="LittleEndian"><StructuredGrid WholeExtent="0 1 0 1 0 1"><Piece Extent="0 1 0 1 0 1"><Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0 1 0 0 0 1 0 1 1 0 0 0 1 1 0 1 0 1 1 1 1 1</DataArray></Points></Piece></StructuredGrid></VTKFile>`), 'VTS StructuredGrid → boundary');
  isCube(await imp('grid.vtr', `<VTKFile type="RectilinearGrid" byte_order="LittleEndian"><RectilinearGrid WholeExtent="0 1 0 1 0 1"><Piece Extent="0 1 0 1 0 1"><Coordinates><DataArray type="Float32" format="ascii">0 1</DataArray><DataArray type="Float32" format="ascii">0 1</DataArray><DataArray type="Float32" format="ascii">0 1</DataArray></Coordinates></Piece></RectilinearGrid></VTKFile>`), 'VTR RectilinearGrid → boundary');
  // GTS: vertices, edges, faces (all 1-based)
  const em = new Map(), edges = [], eid = (a, b) => { const k = a < b ? a + '_' + b : b + '_' + a; if (!em.has(k)) { edges.push([a, b]); em.set(k, edges.length); } return em.get(k); };
  const gf = T.map((t) => [eid(t[0], t[1]), eid(t[1], t[2]), eid(t[2], t[0])]);
  isCube(await imp('cube.gts', `8 ${edges.length} 12\n${vtxt()}\n${edges.map((e) => `${e[0] + 1} ${e[1] + 1}`).join('\n')}\n${gf.map((f) => f.join(' ')).join('\n')}\n`), 'GTS');
  isCube(await imp('cube.byu', `1 8 6 24\n1 6\n${vtxt()}\n${Q.map((q) => q.map((i, k) => (k === 3 ? -(i + 1) : i + 1)).join(' ')).join('\n')}\n`), 'BYU');
  isCube(await imp('cube.tri', `8 12\n${vtxt()}\n${T.map((t) => t.map((i) => i + 1).join(' ')).join('\n')}\n`), 'TRI (Cart3D-style counts line)');
  isCube(await imp('cube.fac', `8\n${V.map((v, k) => `${k + 1} ${v.join(' ')}`).join('\n')}\n12\n${T.map((t, k) => `${k + 1} ${t.map((i) => i + 1).join(' ')}`).join('\n')}\n`), 'FAC (separate counts, indexed rows)');
  await throws('unrecognised .surf dialect', () => imp('x.surf', 'surface data v2\nfoo bar\n'), /STL|OBJ|PLY/);
});

await group('computational meshes', async () => {
  const nodes22 = V.map((v, k) => `${k + 1} ${v.join(' ')}`).join('\n');
  const g2 = await imp('cube.msh', `$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n8\n${nodes22}\n$EndNodes\n$Elements\n8\n1 15 2 0 1 1\n2 3 2 0 1 1 4 3 2\n${TETS.map((t, k) => `${k + 3} 4 2 0 1 ${t.map((i) => i + 1).join(' ')}`).join('\n')}\n$EndElements\n`);
  isCube(g2, 'Gmsh 2.2 (six tetrahedra + boundary quad + point)');
  ok(g2.format === 'Gmsh MSH' && g2.stats.cellTypes.tet === 6 && g2.stats.cellTypes.quad === 1 && g2.stats.dimension === 3 && g2.stats.nodes === 8, 'Gmsh 2.2 stats');
  isCube(await imp('cube.msh', `$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Entities\n0 0 0 1\n1 0 0 0 1 1 1 0 0\n$EndEntities\n$Nodes\n1 8 1 8\n3 1 0 8\n1\n2\n3\n4\n5\n6\n7\n8\n${vtxt()}\n$EndNodes\n$Elements\n1 1 1 1\n3 1 5 1\n1 1 2 3 4 5 6 7 8\n$EndElements\n`), 'Gmsh 4.1 (one hexahedron)');
  isCube(await imp('cube.msh', `$MeshFormat\n4 0 8\n$EndMeshFormat\n$Nodes\n1 8\n1 3 0 8\n${nodes22}\n$EndNodes\n$Elements\n1 1\n1 3 5 1\n1 1 2 3 4 5 6 7 8\n$EndElements\n`), 'Gmsh 4.0 (one hexahedron)');
  const sq = await imp('square.msh', `$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n1 0 0 0\n2 2 0 0\n3 2 2 0\n4 0 2 0\n$EndNodes\n$Elements\n2\n1 2 2 0 1 1 2 3\n2 2 2 0 1 1 3 4\n$EndElements\n`);
  ok(sq.kind === 'polylines' && sq.polylines.length === 1 && sq.polylines[0].closed && sq.polylines[0].x.length === 4 && sq.stats.dimension === 2 && Math.abs(dimensions(sq).area - 4) < 1e-12, '2-D Gmsh mesh → closed boundary outline (area 4)');
  await throws('binary Gmsh', () => imp('b.msh', '$MeshFormat\n2.2 1 8\n\x01\x00\x00\x00\n$EndMeshFormat\n'), /ASCII/);
  // Fluent ASCII: one hexahedral cell, six boundary quads (hex indices, c0 = cell 1, c1 = 0)
  const fl = `(0 "test mesh")\n(2 3)\n(10 (0 1 8 0 3))\n(12 (0 1 1 0))\n(13 (0 1 6 0))\n(10 (1 1 8 1 3)(\n${vtxt()}\n))\n(13 (2 1 6 3 4)(\n${Q.map((q) => q.map((i) => (i + 1).toString(16)).join(' ') + ' 1 0').join('\n')}\n))\n(12 (3 1 1 1 4))\n`;
  const gfl = await imp('cube.msh', fl);
  isCube(gfl, 'Fluent ASCII mesh (sniffed from .msh)');
  ok(gfl.format === 'Fluent mesh (ASCII)' && gfl.stats.boundaryFaces === 6 && gfl.stats.cells === 1, 'Fluent stats and format name');
  isCube(await imp('cube.su2', `% test\nNDIME= 3\nNELEM= 6\n${TETS.map((t, k) => `10 ${t.join(' ')} ${k}`).join('\n')}\nNPOIN= 8\n${V.map((v, k) => v.join(' ') + ' ' + k).join('\n')}\nNMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 2\n5 0 1 2\n5 0 2 3\n`), 'SU2');
  const s2 = await imp('sq.su2', 'NDIME= 2\nNELEM= 1\n9 0 1 2 3 0\nNPOIN= 4\n0 0 0\n1 0 1\n1 1 2\n0 1 3\n');
  ok(s2.kind === 'polylines' && s2.polylines[0].closed && Math.abs(dimensions(s2).area - 1) < 1e-12, 'SU2 2-D quad → outline');
  const pad = (v, w = 10) => String(v).padStart(w);
  const unv = `    -1\n  2411\n${V.map((v, k) => `${pad(k + 1)}${pad(1)}${pad(1)}${pad(11)}\n${v.map((c) => c.toExponential(16).replace('e', 'D').padStart(25)).join('')}`).join('\n')}\n    -1\n    -1\n  2412\n${pad(1)}${pad(115)}${pad(1)}${pad(1)}${pad(7)}${pad(8)}\n${[1, 2, 3, 4, 5, 6, 7, 8].map((i) => pad(i)).join('')}\n${pad(2)}${pad(94)}${pad(1)}${pad(1)}${pad(7)}${pad(4)}\n${[1, 4, 3, 2].map((i) => pad(i)).join('')}\n    -1\n`;
  const gu = await imp('cube.unv', unv);
  isCube(gu, 'UNV (2411 nodes, 2412 hexahedron + shell)');
  ok(gu.stats.cellTypes.hex === 1 && gu.stats.cellTypes.quad === 1, 'UNV element types');
  const f8 = (s) => String(s).padEnd(8).slice(0, 8), f16 = (s) => String(s).padEnd(16).slice(0, 16);
  const bdfFree = `$ free field\nBEGIN BULK\n${V.map((v, k) => `GRID,${k + 1},,${v.map((c) => c.toFixed(1)).join(',')}`).join('\n')}\nCHEXA,1,1,1,2,3,4,5,6,+C1\n+C1,7,8\nENDDATA\n`;
  isCube(await imp('cube.bdf', bdfFree), 'Nastran free field (CHEXA with continuation)');
  const bdfSmall = `SOL 101\nCEND\nBEGIN BULK\n${V.map((v, k) => f8('GRID') + f8(k + 1) + f8('') + v.map((c) => f8(c ? '1.0' : '0.')).join('')).join('\n')}\n${TETS.map((t, k) => f8('CTETRA') + f8(k + 1) + f8(1) + t.map((i) => f8(i + 1)).join('')).join('\n')}\nENDDATA\n`;
  isCube(await imp('cube.nas', bdfSmall), 'Nastran small field (CTETRA)');
  const bdfLarge = `BEGIN BULK\n${V.map((v, k) => f8('GRID*') + f16(k + 1) + f16('') + f16(v[0] ? '1.+0' : '0.0') + f16(v[1] ? '1.0D0' : '0.0') + f8('*G' + k) + '\n' + f8('*G' + k) + f16(v[2] ? '10.-1' : '0.0')).join('\n')}\n${f8('CHEXA')}${f8(1)}${f8(1)}${[1, 2, 3, 4, 5, 6].map(f8).join('')}${f8('+H')}\n${f8('+H')}${f8(7)}${f8(8)}\nENDDATA\n`;
  isCube(await imp('cube.bdf', bdfLarge), 'Nastran large-field GRID* with Nastran-style exponents');
  const tecFE = `TITLE = "cube"\nVARIABLES = "X", "Y", "Z", "P"\nZONE T="solid", N=8, E=1, DATAPACKING=POINT, ZONETYPE=FEBRICK\n${V.map((v) => v.join(' ') + ' 1.5').join('\n')}\n1 2 3 4 5 6 7 8\n`;
  isCube(await imp('cube.dat', tecFE), 'Tecplot FEBRICK (point packing, sniffed from .dat)');
  const tecBlock = `VARIABLES = "X" "Y" "Z"\nZONE N=8, E=6, F=FEBLOCK, ET=TETRAHEDRON\n${[0, 1, 2].map((c) => V.map((v) => v[c]).join(' ')).join('\n')}\n${TETS.map((t) => t.map((i) => i + 1).join(' ')).join('\n')}\n`;
  isCube(await imp('cube.tec', tecBlock), 'Tecplot FEBLOCK tetrahedra');
  const ijk = []; for (let k = 0; k < 2; k++) for (let j = 0; j < 2; j++) for (let i = 0; i < 3; i++) ijk.push([i / 2, j, k]);
  const tecO = await imp('cube.plt', `TITLE="ordered"\nVARIABLES="X","Y","Z"\nZONE I=3, J=2, K=2, F=POINT\n${ijk.map((p) => p.join(' ')).join('\n')}\n`);
  isCube(tecO, 'Tecplot ordered I×J×K zone (sniffed from .plt)', { count: 20 });
  ok(tecO.format === 'Tecplot ASCII' && tecO.stats.structured && tecO.stats.cells === 2, 'Tecplot ordered zone stats');
  await throws('Tecplot with an absurd VARLOCATION range', () => imp('cc.dat', 'VARIABLES = "X", "Y", "Z", "P"\nZONE N=8, E=1, DATAPACKING=BLOCK, ZONETYPE=FEBRICK, VARLOCATION=([1-999999999999]=CELLCENTERED)\n0 1 1 0 0 1 1 0\n0 0 1 1 0 0 1 1\n0 0 0 0 1 1 1 1\n1.5\n1 2 3 4 5 6 7 8\n'), /Cell-centred/);
  const tecCC = await imp('cc.dat', 'VARIABLES = "X", "Y", "Z", "P"\nZONE N=8, E=1, DATAPACKING=BLOCK, ZONETYPE=FEBRICK, VARLOCATION=([4]=CELLCENTERED)\n0 1 1 0 0 1 1 0\n0 0 1 1 0 0 1 1\n0 0 0 0 1 1 1 1\n1.5\n1 2 3 4 5 6 7 8\n');
  isCube(tecCC, 'Tecplot block packing with a cell-centred variable');
  const tec2 = await imp('sq.dat', 'VARIABLES = "X", "Y"\nZONE N=4, E=1, F=FEPOINT, ET=QUADRILATERAL\n0 0\n3 0\n3 1\n0 1\n1 2 3 4\n');
  ok(tec2.kind === 'polylines' && tec2.polylines[0].closed && Math.abs(dimensions(tec2).area - 3) < 1e-12, 'Tecplot 2-D quadrilateral → outline');
  await throws('binary Tecplot .plt', () => imp('b.plt', cat('#!TDV112', new Uint8Array(40))), /ASCII/);
  const p3 = [0, 1, 2].map((c) => ijk.map((p) => p[c]).join(' ')).join('\n');
  isCube(await imp('grid.p3d', `3 2 2\n${p3}\n`), 'Plot3D single block', { count: 20 });
  isCube(await imp('grid.x', `1\n3 2 2\n${p3}\n`), 'Plot3D multi-block header', { count: 20 });
  const px = await imp('grid.xyz', `3 2 2\n${p3}\n`);
  ok(px.kind === 'mesh' && px.format === 'Plot3D grid (ASCII)' && px.count === 20, '.xyz with a Plot3D header is read as a structured grid');
  isCube(await imp('mesh.json', JSON.stringify({ nodes: V, elements: [[0, 1, 2, 3, 4, 5, 6, 7]] })), 'JSON nodes + elements (hexahedron)');
  isCube(await imp('mesh.json', JSON.stringify({ vertices: V, faces: Q.map((q) => q.map((i) => i + 1)) })), 'JSON vertices + faces (1-based quads)');
  isCube(await imp('mesh.json', JSON.stringify({ nodes: V, cells: TETS })), 'JSON nodes + cells (tetrahedra recognised by shape)');
  // OpenFOAM polyMesh
  const foam = (cls, body, note = '') => `/*--------------------------------*- C++ -*----------------------------------*\\\n| OpenFOAM |\n\\*---------------------------------------------------------------------------*/\nFoamFile\n{\n    version 2.0;\n    format ascii;\n    class ${cls};\n    ${note}\n    object x;\n}\n// * * * //\n${body}\n`;
  const fp = F('points', foam('vectorField', `8\n(\n${V.map((v) => `(${v.join(' ')})`).join('\n')}\n)`)), ff = F('faces', foam('faceList', `6\n(\n${Q.map((q) => `4(${q.join(' ')})`).join('\n')}\n)`)), fo = F('owner', foam('labelList', '6\n(\n0\n0\n0\n0\n0\n0\n)', 'note "nPoints: 8 nCells: 1 nFaces: 6 nInternalFaces: 0";')), fn = F('neighbour', foam('labelList', '0\n(\n)'));
  const gfo = await importGeometry(fp, { companion: { faces: ff, owner: fo, neighbour: fn } });
  isCube(gfo, 'OpenFOAM polyMesh (points + faces + owner + neighbour)');
  ok(gfo.stats.cells === 1 && gfo.stats.internalFaces === 0, 'OpenFOAM stats');
  isCube(await importGeometry(ff, { companion: [fp, fo] }), 'OpenFOAM started from faces, companions as a list, nInternalFaces from the owner note');
  await throws('OpenFOAM points without companions', () => importGeometry(fp), /faces/);
});

// ---- CAD writers ----------------------------------------------------------------------------------------
const RN = (v) => { const s = String(v); return /[.e]/.test(s) ? s : s + '.'; };
function stepFile(build, schema = 'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }') {
  let id = 0;
  const lines = [], E = (s) => { lines.push(`#${++id}=${s};`); return '#' + id; };
  build(E);
  return `ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('test; with ''quotes'''),'2;1');\nFILE_NAME('t.stp','2024-01-01T00:00:00',('a'),('o'),'p','s','');\nFILE_SCHEMA(('${schema}'));\nENDSEC;\nDATA;\n/* geometry */\n${lines.join('\n')}\nENDSEC;\nEND-ISO-10303-21;\n`;
}
const sP = (E, p) => E(`CARTESIAN_POINT('',(${p.map(RN).join(',')}))`);
const sD = (E, d) => E(`DIRECTION('',(${d.map(RN).join(',')}))`);
const sAX = (E, o, z, x) => E(`AXIS2_PLACEMENT_3D('',${sP(E, o)},${sD(E, z)},${sD(E, x)})`);
const sVX = (E, p) => E(`VERTEX_POINT('',${sP(E, p)})`);
const sLine = (E, va, vb, pa, pb) => E(`EDGE_CURVE('',${va},${vb},${E(`LINE('',${sP(E, pa)},${E(`VECTOR('',${sD(E, pb.map((v, k) => v - pa[k]))},1.)`)})`)},.T.)`);
const sLoop = (E, edges) => E(`EDGE_LOOP('',(${edges.map(([e, o]) => E(`ORIENTED_EDGE('',*,*,${e},${o ? '.T.' : '.F.'})`)).join(',')}))`);
const sUnits = (E) => E('(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))');
const crossv = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
/** Planar quad faces (outward, counter-clockwise) -> ADVANCED_FACE refs. */
function sQuadFaces(E, pts, quads) {
  return quads.map((q) => {
    const p = q.map((i) => pts[i]), vs = p.map((x) => sVX(E, x)), es = p.map((x, k) => [sLine(E, vs[k], vs[(k + 1) % 4], x, p[(k + 1) % 4]), true]);
    const u = p[1].map((v, k) => v - p[0][k]), w = p[3].map((v, k) => v - p[0][k]), n = crossv(u, w), l = Math.hypot(...n), ul = Math.hypot(...u);
    return E(`ADVANCED_FACE('',(${E(`FACE_OUTER_BOUND('',${sLoop(E, es)},.T.)`)}),${E(`PLANE('',${sAX(E, p[0], n.map((v) => v / l), u.map((v) => v / ul))})`)},.T.)`);
  });
}
function igesFile(entities) {
  const row = (body, sec, n) => body.padEnd(72).slice(0, 72) + sec + String(n).padStart(7), f = (v) => String(v).padStart(8);
  const chunk = (s, w) => { const out = []; let cur = ''; for (const part of s.split(/(?<=[,;])/)) { if (cur.length + part.length > w) { out.push(cur); cur = ''; } cur += part; } if (cur) out.push(cur); return out; };
  const G = chunk('1H,,1H;,4HTEST,8HTEST.IGS,6HSYSTEM,3H1.0,32,38,6,308,15,4HTEST,1.,2,2HMM,1,0.01,15H20240101.000000,1E-06,100.,4HAUTH,3HORG,11,0,15H20240101.000000;', 72);
  const D = [], P = [];
  entities.forEach((e, i) => {
    const pl = chunk(e.params + ';', 64), start = P.length + 1;
    for (const s of pl) P.push(s.padEnd(64) + ' ' + String(2 * i + 1).padStart(7) + 'P' + String(P.length + 1).padStart(7));
    D.push(f(e.type) + f(start) + f(0) + f(0) + f(0) + f(0) + f(e.xf || 0) + f(0) + (e.status || '00000000') + 'D' + String(2 * i + 1).padStart(7));
    D.push(f(e.type) + f(0) + f(0) + f(pl.length) + f(e.form || 0) + f('') + f('') + f('') + f(0) + 'D' + String(2 * i + 2).padStart(7));
  });
  return [row('IGES test file', 'S', 1), ...G.map((g, k) => row(g, 'G', k + 1)), ...D, ...P, row(`S${String(1).padStart(7)}G${String(G.length).padStart(7)}D${String(D.length).padStart(7)}P${String(P.length).padStart(7)}`, 'T', 1)].join('\n') + '\n';
}
const A48 = 24 * Math.sin((2 * Math.PI) / 48), P48 = 96 * Math.sin(Math.PI / 48);   // area and perimeter of the 48-gon inscribed in a unit circle

await group('CAD: STEP and IGES', async () => {
  const B = V.map((v) => [v[0] * 2, v[1] * 3, v[2] * 4]);
  const box = await imp('box.step', stepFile((E) => { sUnits(E); const fs = sQuadFaces(E, B, Q); E(`MANIFOLD_SOLID_BREP('box',${E(`CLOSED_SHELL('',(${fs.join(',')}))`)})`); }));
  isCube(box, 'STEP box (ADVANCED_FACE on PLANE)', { hi: [2, 3, 4], area: 52, volume: 24 });
  ok(box.stats.units === 'mm' && box.stats.faces === 6 && box.stats.tessellatedFaces === 6 && box.format === 'STEP (ISO 10303)' && box.pathway === 'cad', 'STEP box: units mm, 6 of 6 faces, format and pathway');
  const part = await imp('part.stp', stepFile((E) => {
    sUnits(E); sQuadFaces(E, B, Q);
    const far = sP(E, [10, 11, 12]);
    E(`ADVANCED_FACE('',(),${E(`B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((${far},${far}),(${far},${far})),.UNSPECIFIED.,.F.,.F.,.F.,(2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.)`)},.T.)`);
  }));
  ok(part.kind === 'mesh' && part.count === 12 && part.warnings.some((w) => /6 of 7 faces tessellated/.test(w) && /b spline surface/.test(w)), 'STEP with a B-spline face: "6 of 7 faces tessellated" reported');
  ok(part.bbox.max[0] === 10 && part.bbox.max[2] === 12 && part.bbox.min[0] === 0, 'STEP bbox still covers every CARTESIAN_POINT when faces are skipped');
  // solid cylinder R = 1, h = 2
  const cyl = await imp('cyl.stp', stepFile((E) => {
    sUnits(E);
    const v0 = sVX(E, [1, 0, 0]), v1 = sVX(E, [1, 0, 2]), ax0 = sAX(E, [0, 0, 0], [0, 0, 1], [1, 0, 0]), ax1 = sAX(E, [0, 0, 2], [0, 0, 1], [1, 0, 0]);
    const e0 = E(`EDGE_CURVE('',${v0},${v0},${E(`CIRCLE('',${ax0},1.)`)},.T.)`), e1 = E(`EDGE_CURVE('',${v1},${v1},${E(`CIRCLE('',${ax1},1.)`)},.T.)`), e2 = sLine(E, v0, v1, [1, 0, 0], [1, 0, 2]);
    E(`ADVANCED_FACE('',(${E(`FACE_OUTER_BOUND('',${sLoop(E, [[e0, false]])},.T.)`)}),${E(`PLANE('',${ax0})`)},.F.)`);
    E(`ADVANCED_FACE('',(${E(`FACE_OUTER_BOUND('',${sLoop(E, [[e1, true]])},.T.)`)}),${E(`PLANE('',${ax1})`)},.T.)`);
    E(`ADVANCED_FACE('',(${E(`FACE_OUTER_BOUND('',${sLoop(E, [[e0, true], [e2, true], [e1, false], [e2, false]])},.T.)`)}),${E(`CYLINDRICAL_SURFACE('',${ax0},1.)`)},.T.)`);
  }));
  const dc = dimensions(cyl);
  ok(cyl.kind === 'mesh' && cyl.count === 46 * 2 + 96 && cyl.stats.tessellatedFaces === 3, `STEP cylinder: 3 faces → ${cyl.count} triangles`);
  ok(dc.closed && dc.consistentNormals && Math.abs(dc.volume - 2 * A48) < 1e-9 && Math.abs(dc.area - (2 * A48 + 2 * P48)) < 1e-9, `STEP cylinder: closed, volume ${dc.volume} (π·r²·h = ${2 * Math.PI}), area ${dc.area}`);
  ok(Math.abs(dc.volume - 2 * Math.PI) / (2 * Math.PI) < 0.005 && cyl.bbox.min.every((v, k) => Math.abs(v - [-1, -1, 0][k]) < 1e-9) && cyl.bbox.max.every((v, k) => Math.abs(v - [1, 1, 2][k]) < 1e-9), 'STEP cylinder: volume within 0.5 % of exact, bbox [-1,-1,0]–[1,1,2]');
  // plates of thickness 1 with N × N round holes of radius 1 on a pitch of 4: planar faces with holes (several bridged to the
  // same outline vertex) + inward-facing cylinders (same_sense .F.)
  for (const N of [1, 3]) {
    const Wd = 4 * N, plate = await imp(`plate${N}.stp`, stepFile((E) => {
      sUnits(E);
      const C = [[0, 0, 0], [Wd, 0, 0], [Wd, Wd, 0], [0, Wd, 0], [0, 0, 1], [Wd, 0, 1], [Wd, Wd, 1], [0, Wd, 1]], rings = [];
      sQuadFaces(E, C, Q.slice(2));
      for (let ia = 0; ia < N; ia++) for (let ib = 0; ib < N; ib++) {
        const cx = 2 + 4 * ia, cy = 2 + 4 * ib, w0 = sVX(E, [cx + 1, cy, 0]), w1 = sVX(E, [cx + 1, cy, 1]), ax0 = sAX(E, [cx, cy, 0], [0, 0, 1], [1, 0, 0]), ax1 = sAX(E, [cx, cy, 1], [0, 0, 1], [1, 0, 0]);
        const c0 = E(`EDGE_CURVE('',${w0},${w0},${E(`CIRCLE('',${ax0},1.)`)},.T.)`), c1 = E(`EDGE_CURVE('',${w1},${w1},${E(`CIRCLE('',${ax1},1.)`)},.T.)`), seam = sLine(E, w0, w1, [cx + 1, cy, 0], [cx + 1, cy, 1]);
        rings.push([c0, c1]);
        E(`ADVANCED_FACE('',(${E(`FACE_BOUND('',${sLoop(E, [[c0, true], [seam, true], [c1, false], [seam, false]])},.T.)`)}),${E(`CYLINDRICAL_SURFACE('',${ax0},1.)`)},.F.)`);
      }
      for (const [zq, q, sense] of [[0, Q[0], '.F.'], [1, Q[1], '.T.']]) {
        const p = q.map((i) => C[i]), vs = p.map((x) => sVX(E, x)), es = p.map((x, k) => [sLine(E, vs[k], vs[(k + 1) % 4], x, p[(k + 1) % 4]), true]);
        E(`ADVANCED_FACE('',(${[E(`FACE_OUTER_BOUND('',${sLoop(E, es)},.T.)`), ...rings.map((r) => E(`FACE_BOUND('',${sLoop(E, [[r[zq], zq === 0]])},.T.)`))].join(',')}),${E(`PLANE('',${sAX(E, [0, 0, zq], [0, 0, 1], [1, 0, 0])})`)},${sense})`);
      }
    }));
    const dp = dimensions(plate), vol = Wd * Wd - N * N * A48;
    ok(plate.kind === 'mesh' && plate.stats.tessellatedFaces === 6 + N * N && dp.closed && dp.consistentNormals, `STEP plate with ${N * N} hole(s): ${6 + N * N} faces, ${plate.count} triangles, closed with consistent normals (${dp.closed}, ${dp.consistentNormals})`);
    ok(Math.abs(dp.volume - vol) < 1e-9 && Math.abs(dp.area - (2 * vol + 4 * Wd + N * N * P48)) < 1e-9, `STEP plate with ${N * N} hole(s): volume ${dp.volume} (exact ${Wd * Wd - N * N * Math.PI}), area ${dp.area}`);
  }
  const broken = await imp('broken.stp', stepFile((E) => { sUnits(E); sQuadFaces(E, V, Q); E("PRODUCT('x',(1,2"); }));
  ok(broken.kind === 'mesh' && broken.count === 12, 'STEP with a malformed, unrelated entity still reads the solid');
  // AP242 tessellated geometry
  const cl = `COORDINATES_LIST('',8,(${V.map((v) => `(${v.map(RN).join(',')})`).join(',')}))`;
  isCube(await imp('tess.stp', stepFile((E) => { sUnits(E); E(`TRIANGULATED_FACE('',${E(cl)},8,((0.,0.,1.)),$,(),(${T.map((t) => `(${t.map((i) => i + 1).join(',')})`).join(',')}))`); }, 'AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }')), 'STEP AP242 TRIANGULATED_FACE');
  isCube(await imp('tess2.stp', stepFile((E) => {
    const strips = Q.slice(0, 3).map((q) => `(${[q[0], q[1], q[3], q[2]].map((i) => 8 - i).join(',')})`), fans = Q.slice(3).map((q) => `(${q.map((i) => 8 - i).join(',')})`);
    E(`COMPLEX_TRIANGULATED_FACE('',${E(cl)},8,((0.,0.,1.)),$,(8,7,6,5,4,3,2,1),(${strips.join(',')}),(${fans.join(',')}))`);
  })), 'STEP AP242 COMPLEX_TRIANGULATED_FACE (strips, fans, pnindex)');
  const fb = await imp('faceted.stp', stepFile((E) => { const ps = V.map((v) => sP(E, v)); const fs = Q.map((q) => E(`FACE('',(${E(`FACE_OUTER_BOUND('',${E(`POLY_LOOP('',(${q.map((i) => ps[i]).join(',')}))`)},.T.)`)}))`)); E(`FACETED_BREP('',${E(`CLOSED_SHELL('',(${fs.join(',')}))`)})`); }));
  isCube(fb, 'STEP FACETED_BREP / POLY_LOOP');
  ok(fb.warnings.some((w) => /length unit/i.test(w)), 'STEP without units says so');
  // IGES: two lines, a half-circle arc and a quadratic NURBS curve
  const ig = await imp('wire.igs', igesFile([{ type: 110, params: '110,0.,0.,0.,10.,0.,0.' }, { type: 110, params: '110,10.,0.,0.,10.,-1.,0.' }, { type: 100, params: '100,0.,5.,0.,10.,0.,0.,0.' }, { type: 126, params: '126,2,2,1,0,1,0,0.,0.,0.,1.,1.,1.,1.,1.,1.,0.,0.,0.,5.,-5.,0.,10.,0.,0.,0.,1.,0.,0.,1.' }, { type: 116, params: '116,5.,2.,0.,0' }, { type: 144, params: '144,1,0,0,0' }]));
  ok(ig.kind === 'polylines' && ig.polylines.length === 4 && ig.stats.units === 'mm' && ig.stats.droppedAxis === 'z' && ig.stats.surfaces === 1 && ig.stats.points === 1, `IGES: 4 curves, mm, z dropped (${ig.polylines.length}, ${ig.stats.units}, ${ig.stats.droppedAxis})`);
  ok(Math.abs(ig.bbox.min[0]) < 1e-9 && Math.abs(ig.bbox.max[0] - 10) < 1e-9 && Math.abs(ig.bbox.max[1] - 5) < 1e-9 && Math.abs(ig.bbox.min[1] + 2.5) < 1e-9, `IGES bbox: arc top y = 5, NURBS dip y = −2.5 (${ig.bbox.min}, ${ig.bbox.max})`);
  ok(ig.polylines[2].x.length === 25 && Math.abs(ig.polylines[2].x[12] - 5) < 1e-9 && Math.abs(ig.polylines[2].y[12] - 5) < 1e-9, 'IGES arc discretised counter-clockwise through (5, 5)');
  ok(ig.warnings.some((w) => /not tessellated/.test(w)) && ig.warnings.some((w) => /projected/.test(w)) && formatOf('a.igs').support === 'partial', 'IGES warns that surfaces are not tessellated');
  const ig2 = await imp('moved.iges', igesFile([{ type: 124, params: '124,1.,0.,0.,100.,0.,1.,0.,200.,0.,0.,1.,0.' }, { type: 110, params: '110,0.,0.,0.,1.,1.,0.', xf: 1 }, { type: 112, params: '112,1,1,3,1,0.,2.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.' }]));
  ok(ig2.polylines.length === 2 && ig2.bbox.max[0] === 101 && ig2.bbox.max[1] === 201, 'IGES transformation matrix (124) applied; parametric spline (112) evaluated');
});

await group('2-D drawings', async () => {
  const dxf = (ents) => `0\nSECTION\n2\nENTITIES\n${ents}0\nENDSEC\n0\nEOF\n`;
  const d1 = await imp('sq.dxf', dxf('0\nLWPOLYLINE\n8\n0\n90\n4\n70\n1\n10\n0\n20\n0\n10\n2\n20\n0\n10\n2\n20\n2\n10\n0\n20\n2\n0\nARC\n8\n0\n10\n5\n20\n0\n40\n1\n50\n0\n51\n180\n0\nELLIPSE\n10\n10\n20\n0\n11\n2\n21\n0\n40\n0.5\n41\n0\n42\n6.283185307179586\n'));
  ok(d1.kind === 'polylines' && d1.polylines.length === 3 && d1.polylines[0].closed && d1.polylines[0].x.length === 4, 'DXF: LWPOLYLINE square + ARC + ELLIPSE → 3 polylines');
  const arc = d1.polylines[1], ell = d1.polylines[2];
  ok(arc.x.length === 25 && Math.abs(arc.x[0] - 6) < 1e-9 && Math.abs(arc.y[12] - 1) < 1e-9 && Math.abs(arc.x[24] - 4) < 1e-9 && !arc.closed, 'DXF ARC from 0° to 180°');
  ok(ell.closed && Math.abs(Math.max(...ell.x) - 12) < 1e-9 && Math.abs(Math.max(...ell.y) - 1) < 1e-9, 'DXF ELLIPSE (major 2, ratio 0.5)');
  const d2 = await imp('face.dxf', dxf('0\n3DFACE\n10\n0\n20\n0\n30\n0\n11\n1\n21\n0\n31\n0\n12\n1\n22\n1\n32\n0\n13\n0\n23\n1\n33\n0\n'));
  ok(d2.kind === 'mesh' && d2.count === 2 && Math.abs(dimensions(d2).area - 1) < 1e-12 && !dimensions(d2).closed, 'DXF 3DFACE → mesh (open surface, area 1)');
  const d3 = await imp('spl.dxf', dxf('0\nSPLINE\n70\n8\n71\n2\n72\n6\n73\n3\n40\n0\n40\n0\n40\n0\n40\n1\n40\n1\n40\n1\n10\n0\n20\n0\n30\n0\n10\n5\n20\n10\n30\n0\n10\n10\n20\n0\n30\n0\n'));
  ok(d3.polylines.length === 1 && Math.abs(Math.max(...d3.polylines[0].y) - 5) < 1e-9, 'DXF SPLINE evaluated (quadratic peak at y = 5)');
  const dInf = await imp('odd.dxf', dxf('0\nARC\n10\n0\n20\n0\n40\n1\n50\n0\n51\n-1e308\n0\nARC\n10\n0\n20\n0\n40\n1\n50\n-Infinity\n51\n90\n0\nLINE\n10\n0\n20\n0\n11\n1\n21\n1\n'));
  ok(dInf.kind === 'polylines' && dInf.polylines.length >= 1 && dInf.polylines.every((p) => p.x.length <= 200), 'DXF arcs with absurd angles neither hang nor explode');
  await throws('DXF without supported entities', () => imp('e.dxf', dxf('0\nTEXT\n1\nhello\n')), /No LINE/);
  const svg = await imp('shapes.svg', `<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><defs><rect x="0" y="0" width="999" height="999"/></defs><rect x="10" y="10" width="10" height="10"/><g transform="translate(100,0) scale(2)"><path d="M0 0 h10 v10 h-10 z"/></g><circle cx="50" cy="50" r="5"/><polygon points="0,90 10,90 10,100"/><path d="M 60 0 A 5 5 0 0 1 70 0 M 80,20 C 80,10 90,10 90,20 q 5,5 10,0 L 100 30"/><line x1="0" y1="0" x2="1" y2="1" transform="matrix(1 0 0 1 150 50)"/></svg>`);
  ok(svg.kind === 'polylines' && svg.polylines.length === 7, `SVG: rect, group path, circle, polygon, 2 sub-paths, line → 7 polylines (${svg.polylines.length})`);
  const [r, gp, c] = svg.polylines, ar = (p) => { let s = 0; for (let i = 0; i < p.x.length; i++) { const j = (i + 1) % p.x.length; s += p.x[i] * p.y[j] - p.x[j] * p.y[i]; } return Math.abs(s) / 2; };
  ok(r.closed && r.x.length === 4 && ar(r) === 100 && gp.closed && Math.min(...gp.x) === 100 && Math.max(...gp.x) === 120 && ar(gp) === 400, 'SVG rect area 100; translate + scale group gives x 100–120, area 400');
  ok(c.closed && Math.abs(ar(c) - 25 * A48) < 1e-9 && svg.bbox.min[1] === -5 && svg.bbox.max[1] === 100, 'SVG circle flattened; elliptical arc bulges 5 units beyond y = 0');
  ok(Math.max(...r.y) === 85 && Math.min(...r.y) === 75, 'SVG y axis flipped within the bounding box');
  const hp = await imp('sq.hpgl', 'IN;SP1;PU0,0;PD4000,0,4000,4000,0,4000,0,0;PU;PA8000,0;CI400;PU;');
  ok(hp.kind === 'polylines' && hp.polylines.length === 2 && hp.polylines[0].closed && hp.polylines[0].x.length === 4 && Math.abs(dimensions(hp).area - (10000 + 100 * A48)) < 1e-6 && hp.stats.units === 'mm', 'HPGL: 100 mm square + circle of radius 10 mm');
  const hpA = await imp('arc.hpgl', 'IN;PU10,0;PD;AA0,0,1e15;PU;');
  ok(hpA.polylines[0].x.length <= 200, 'HPGL arc with an absurd sweep stays bounded');
  const hp2 = await imp('sq.plt', 'IN;PU10,10;PR;PD100,0,0,100,-100,0,0,-100;PU;');
  ok(hp2.format === 'HPGL plot file' && hp2.pathway === 'drawing' && hp2.polylines[0].closed && Math.abs(hp2.bbox.max[0] - 2.75) < 1e-12, 'HPGL sniffed from .plt, relative moves');
  const xy = await imp('profile.xy', '# x y\n0 0\n2 0\n2 1\n0 1\n0 0\n\n5 5\n6 6\n');
  ok(xy.kind === 'polylines' && xy.polylines.length === 2 && xy.polylines[0].closed && xy.polylines[0].x.length === 4 && !xy.polylines[1].closed, 'x-y file: blank line separates polylines, repeated first point closes');
});

// ---- raster writers ----------------------------------------------------------------------------------------
function lzwEncode(data) {
  const out = [];
  let buf = 0, cnt = 0, width = 9, next = 258, table = new Map(), w = '';
  const put = (code) => { buf = (buf << width) | code; cnt += width; while (cnt >= 8) { out.push((buf >> (cnt - 8)) & 255); cnt -= 8; } buf &= (1 << cnt) - 1; };
  const codeOf = (s) => (s.length === 1 ? s.charCodeAt(0) : table.get(s));
  put(256);
  for (const b of data) {
    const c = String.fromCharCode(b);
    if (w === '') { w = c; continue; }
    if (table.has(w + c)) { w += c; continue; }
    put(codeOf(w));
    table.set(w + c, next++);
    if (next === 4094) { put(256); table = new Map(); next = 258; width = 9; } else if (next > (1 << width) - 1) width++;
    w = c;
  }
  if (w !== '') put(codeOf(w));
  put(257);
  if (cnt) out.push((buf << (8 - cnt)) & 255);
  return new Uint8Array(out);
}
function tiffFile(pages, little = true) {
  const chunks = [new Uint8Array(8)], ifds = [];
  let pos = 8;
  const put = (b) => { const o = pos; chunks.push(b); pos += b.length; if (pos % 2) { chunks.push(new Uint8Array(1)); pos++; } return o; };
  for (const pg of pages) {
    const offs = pg.strips.map((s) => put(s)), tags = [...pg.tags, [pg.tiled ? 324 : 273, 4, offs], [pg.tiled ? 325 : 279, 4, pg.strips.map((s) => s.length)]].sort((a, b) => a[0] - b[0]);
    ifds.push(tags.map(([tag, type, vals]) => {
      const bytes = type === 2 ? enc.encode(vals + '\0') : le({ 3: 'u16', 4: 'u32', 12: 'f64' }[type], vals, little), count = type === 2 ? bytes.length : vals.length;
      let vf = new Uint8Array(4);
      if (bytes.length <= 4) vf.set(bytes); else vf = le('u32', [put(bytes)], little);
      return cat(le('u16', [tag, type], little), le('u32', [count], little), vf);
    }));
  }
  const at = [];
  let p = pos;
  for (const e of ifds) { at.push(p); p += 2 + 12 * e.length + 4; }
  ifds.forEach((e, k) => chunks.push(le('u16', [e.length], little), ...e, le('u32', [k + 1 < ifds.length ? at[k + 1] : 0], little)));
  const out = cat(...chunks);
  out.set(cat(little ? 'II' : 'MM', le('u16', [42], little), le('u32', [at[0]], little)), 0);
  return out;
}
/** One TIFF page: values row-major, type 'u8' | 'u16' | 'i16' | 'f32' | 'f64'. */
function tiffPage({ w, h, values, type = 'f32', comp = 1, pred = 1, tile = 0, little = true, geo = null, nodata, rowsPerStrip }) {
  const bp = { u8: 1, u16: 2, i16: 2, f32: 4, f64: 8 }[type], sf = type[0] === 'f' ? 3 : type[0] === 'i' ? 2 : 1;
  const encodeRows = (vals, cw, rows) => {
    let v = Array.from(vals);
    if (pred === 2) for (let r = 0; r < rows; r++) for (let i = cw - 1; i > 0; i--) v[r * cw + i] -= v[r * cw + i - 1];
    let b = le(type, pred === 2 && type === 'u16' ? v.map((x) => (x + 65536) % 65536) : pred === 2 && type === 'u8' ? v.map((x) => (x + 256) % 256) : v, little);
    if (pred === 3) {
      const src = le(type, v, false), o = new Uint8Array(src.length);
      for (let r = 0; r < rows; r++) { for (let s = 0; s < cw; s++) for (let k = 0; k < bp; k++) o[r * cw * bp + k * cw + s] = src[(r * cw + s) * bp + k]; for (let i = cw * bp - 1; i > 0; i--) o[r * cw * bp + i] = (o[r * cw * bp + i] - o[r * cw * bp + i - 1]) & 255; }
      b = o;
    }
    return comp === 5 ? lzwEncode(b) : comp === 8 ? new Uint8Array(zlib.deflateSync(b)) : comp === 32773 ? cat(...Array.from({ length: Math.ceil(b.length / 100) }, (_, k) => { const s = b.subarray(k * 100, k * 100 + 100); return cat(le('u8', [s.length - 1]), s); })) : b;
  };
  const strips = [], tags = [[256, 4, [w]], [257, 4, [h]], [258, 3, [bp * 8]], [259, 3, [comp]], [262, 3, [1]], [277, 3, [1]], [284, 3, [1]], [317, 3, [pred]], [339, 3, [sf]]];
  if (tile) {
    for (let ty = 0; ty < h; ty += tile) for (let tx = 0; tx < w; tx += tile) { const t = new Array(tile * tile).fill(0); for (let r = 0; r < tile && ty + r < h; r++) for (let c = 0; c < tile && tx + c < w; c++) t[r * tile + c] = values[(ty + r) * w + tx + c]; strips.push(encodeRows(t, tile, tile)); }
    tags.push([322, 3, [tile]], [323, 3, [tile]]);
  } else {
    const rps = rowsPerStrip || h;
    for (let r = 0; r < h; r += rps) strips.push(encodeRows(values.slice(r * w, Math.min(h, r + rps) * w), w, Math.min(rps, h - r)));
    tags.push([278, 4, [rps]]);
  }
  if (geo) tags.push([33550, 12, [geo.dx, geo.dy, 0]], [33922, 12, [0, 0, 0, geo.x0, geo.y1, 0]], [34735, 3, [1, 1, 0, 2, 1024, 0, 1, geo.geographic ? 2 : 1, 1025, 0, 1, 1]]);
  if (nodata !== undefined) tags.push([42113, 2, String(nodata)]);
  return { tags, strips, tiled: !!tile };
}
function netcdf(dims, vars, version = 1) {
  const pad4 = (b) => cat(b, new Uint8Array((4 - (b.length % 4)) % 4)), nm = (s) => cat(be('u32', [s.length]), pad4(enc.encode(s)));
  const NC = { i8: 1, char: 2, i16: 3, i32: 4, f32: 5, f64: 6 }, attList = (atts) => (atts && atts.length ? cat(be('u32', [12, atts.length]), ...atts.map(([n, t, v]) => cat(nm(n), be('u32', [NC[t], t === 'char' ? v.length : v.length]), pad4(t === 'char' ? enc.encode(v) : be(t, v))))) : be('u32', [0, 0]));
  const datas = vars.map((v) => pad4(be(v.type, v.data)));
  const header = (begins) => cat('CDF', le('u8', [version]), be('u32', [0]), be('u32', [10, dims.length]), ...dims.map(([n, l]) => cat(nm(n), be('u32', [l]))), attList([['title', 'char', 'test']]), be('u32', [11, vars.length]), ...vars.map((v, k) => cat(nm(v.name), be('u32', [v.dims.length, ...v.dims]), attList(v.atts), be('u32', [NC[v.type], datas[k].length]), version === 1 ? be('u32', [begins[k]]) : be('u64', [begins[k]]))));
  const hl = header(vars.map(() => 0)).length, begins = [];
  let p = hl;
  for (const d of datas) { begins.push(p); p += d.length; }
  return cat(header(begins), ...datas);
}

await group('GIS vectors', async () => {
  const gj = await imp('coast.geojson', JSON.stringify({ type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [[[10, 50], [11, 50], [11, 51], [10, 51], [10, 50]]] } }, { type: 'Feature', geometry: { type: 'LineString', coordinates: [[10, 50], [12, 52]] } }, { type: 'Feature', geometry: { type: 'Point', coordinates: [10.5, 50.5] } }] }));
  ok(gj.kind === 'polylines' && gj.polylines.length === 2 && gj.polylines[0].closed && gj.polylines[0].x.length === 4 && gj.geographic && gj.format === 'GeoJSON' && gj.markers.length === 1, 'GeoJSON polygon + line + point');
  const gp = await imp('pts.json', JSON.stringify({ type: 'MultiPoint', coordinates: [[1, 2, 3], [4, 5, 6]] }));
  ok(gp.kind === 'points' && gp.count === 2 && gp.points[5] === 6 && gp.format === 'GeoJSON', 'GeoJSON points only (sniffed from .json)');
  const kml = `<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Area &amp; outfall</name><Polygon><outerBoundaryIs><LinearRing><coordinates>\n 54.0,24.0,0 54.5,24.0,0 54.5,24.4,0 54.0,24.4,0 54.0,24.0,0\n</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark><Placemark><LineString><coordinates>54.1,24.1 54.2,24.3</coordinates></LineString></Placemark><Placemark><Point><coordinates>54.25,24.2,-12</coordinates></Point></Placemark></Document></kml>`;
  const k1 = await imp('site.kml', kml);
  ok(k1.kind === 'polylines' && k1.polylines.length === 2 && k1.polylines[0].closed && k1.polylines[0].x.length === 4 && k1.bbox.max[0] === 54.5 && k1.bbox.min[1] === 24 && k1.geographic && k1.stats.points === 1, 'KML polygon ring, line string and point');
  const k2 = await imp('site.kmz', zip([['doc.kml', kml]], true));
  ok(k2.kind === 'polylines' && k2.polylines.length === 2 && k2.format === 'KML / KMZ', 'KMZ (deflated zip)');
  const gpx = await imp('track.gpx', '<?xml version="1.0"?><gpx version="1.1"><wpt lat="24.5" lon="54.5"><ele>3</ele></wpt><trk><trkseg><trkpt lat="24.0" lon="54.0"><ele>1</ele></trkpt><trkpt lat="24.1" lon="54.2"/><trkpt lat="24.3" lon="54.3"/></trkseg></trk><rte><rtept lat="1" lon="2"/><rtept lat="3" lon="4"/></rte></gpx>');
  ok(gpx.kind === 'polylines' && gpx.polylines.length === 2 && gpx.polylines[0].x[1] === 54.2 && gpx.polylines[0].y[2] === 24.3 && gpx.markers[0][0] === 54.5, 'GPX track, route and waypoint (x = lon, y = lat)');
  const gml = await imp('zone.gml', '<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml"><gml:featureMember><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension="2">0 0 4 0 4 3 0 3 0 0</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:featureMember><gml:LineString><gml:coordinates>1,1 2,2 3,1</gml:coordinates></gml:LineString><gml:Point><gml:pos>9 9</gml:pos></gml:Point></gml:FeatureCollection>');
  ok(gml.kind === 'polylines' && gml.polylines.length === 2 && gml.polylines[0].closed && Math.abs(dimensions(gml).area - 12) < 1e-12 && gml.polylines[1].x.length === 3 && gml.markers.length === 1, 'GML posList ring, coordinates line string and pos point');
  // shapefile: one polygon record (ring of 5 points) and one polyline with two parts
  const ring = [[0, 0], [0, 2], [3, 2], [3, 0], [0, 0]], rec = (n, content) => cat(be('i32', [n, content.length / 2]), content);
  const polyRec = (type, parts, pts) => cat(le('i32', [type]), le('f64', [0, 0, 3, 2]), le('i32', [parts.length, pts.length, ...parts]), le('f64', pts.flat()));
  const shp = (type, recs) => { const body = cat(...recs), h = new Uint8Array(100), dv = new DataView(h.buffer); dv.setInt32(0, 9994, false); dv.setInt32(24, (100 + body.length) / 2, false); dv.setInt32(28, 1000, true); dv.setInt32(32, type, true); return cat(h, body); };
  const dbf = cat(le('u8', [3, 124, 1, 1]), le('u32', [1]), le('u16', [65, 11]), new Uint8Array(20), cat('NAME', new Uint8Array(7), 'C', new Uint8Array(4), le('u8', [10]), new Uint8Array(15)), le('u8', [0x0d]), ' intake    ');
  const s1 = await imp('zone.shp', shp(5, [rec(1, polyRec(5, [0], ring)), rec(2, polyRec(3, [0, 2], [[5, 5], [6, 6], [7, 7], [8, 9]]))]), { companion: { 'zone.dbf': F('zone.dbf', dbf), 'zone.prj': F('zone.prj', 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984"]]') } });
  ok(s1.kind === 'polylines' && s1.polylines.length === 3 && s1.polylines[0].closed && s1.polylines[0].x.length === 4 && !s1.polylines[1].closed && s1.polylines[2].y[1] === 9, 'Shapefile polygon ring + two-part polyline');
  ok(s1.stats.records === 2 && s1.stats.attributes[0] === 'NAME' && s1.stats.attributeRecords === 1 && s1.geographic === true && /GEOGCS/.test(s1.stats.projection), 'Shapefile .dbf field names and .prj read from companions');
  const s2 = await imp('wells.shp', shp(11, [rec(1, cat(le('i32', [11]), le('f64', [500000, 2700000, -15, 0]))), rec(2, cat(le('i32', [11]), le('f64', [500010, 2700020, -18, 0])))]));
  ok(s2.kind === 'points' && s2.count === 2 && s2.points[2] === -15 && s2.points[4] === 2700020 && s2.geographic === false, 'Shapefile PointZ, lone .shp');
  const mif = await imp('zone.mif', 'Version 300\nCharset "WindowsLatin1"\nDelimiter ","\nColumns 1\n  Name Char(10)\nData\n\nRegion 1\n  5\n0 0\n4 0\n4 4\n0 4\n0 0\n    Pen (1,2,0)\n    Brush (2,16777215,16777215)\nPline 3\n1 1\n2 2\n3 1\nLine 0 0 9 9\nPoint 5 5\n');
  ok(mif.kind === 'polylines' && mif.polylines.length === 3 && mif.polylines[0].closed && mif.polylines[0].x.length === 4 && mif.polylines[1].x.length === 3 && mif.bbox.max[0] === 9, 'MapInfo MIF region, pline and line');
});

await group('rasters and DEMs', async () => {
  // 5 × 4 DEM, value = 100·row(from the top) + col; lower-left corner (1000, 2000), cell 10
  const W = 5, H = 4, dem = Array.from({ length: W * H }, (_, i) => 100 * Math.floor(i / W) + (i % W) + 0.5);
  const isDem = (g, label, { nod = 0, x0 = 1005, y0 = 2005, d = 10 } = {}) => {
    ok(g.kind === 'grid' && g.grid.x.length === W && g.grid.y.length === H && g.grid.z.length === H && g.grid.z[0].length === W, `${label}: 5 × 4 grid (kind ${g.kind})`);
    ok(Math.abs(g.grid.x[0] - x0) < 1e-9 && Math.abs(g.grid.x[4] - (x0 + 4 * d)) < 1e-9 && Math.abs(g.grid.y[0] - y0) < 1e-9 && Math.abs(g.grid.y[3] - (y0 + 3 * d)) < 1e-9 && g.grid.y[1] > g.grid.y[0], `${label}: cell-centre coordinates ascending (${g.grid.x[0]}, ${g.grid.y[0]})`);
    ok(g.grid.z[0][1] === 301.5 && g.grid.z[3][4] === 4.5 && (nod ? Number.isNaN(g.grid.z[3][0]) : g.grid.z[3][0] === 0.5) && g.grid.nodata === nod, `${label}: values (top row of the file is the highest y), ${g.grid.nodata} nodata`);
  };
  const ascBody = (vals) => Array.from({ length: H }, (_, r) => vals.slice(r * W, r * W + W).join(' ')).join('\n');
  isDem(await imp('dem.asc', `ncols 5\nnrows 4\nxllcorner 1000\nyllcorner 2000\ncellsize 10\nNODATA_value -9999\n${ascBody(dem)}\n`), 'ESRI ASCII grid');
  const withHole = dem.slice(); withHole[0] = -9999;
  isDem(await imp('dem.asc', `NCOLS 5\nNROWS 4\nXLLCENTER 1005\nYLLCENTER 2005\nCELLSIZE 10\nNODATA_VALUE -9999\n${ascBody(withHole)}\n`), 'ESRI ASCII grid (xllcenter, nodata)', { nod: 1 });
  const geo = { dx: 10, dy: 10, x0: 1000, y1: 2040 };
  const t1 = await imp('dem.tif', tiffFile([tiffPage({ w: W, h: H, values: dem, geo })]));
  isDem(t1, 'GeoTIFF float32 uncompressed');
  ok(t1.format === 'GeoTIFF / TIFF' && t1.pathway === 'gis' && t1.grid.geographic === false, 'GeoTIFF format / pathway / projected');
  isDem(await imp('dem.tif', tiffFile([tiffPage({ w: W, h: H, values: dem, geo, comp: 8 })])), 'GeoTIFF Deflate');
  isDem(await imp('dem.tif', tiffFile([tiffPage({ w: W, h: H, values: dem, geo, comp: 5 })])), 'GeoTIFF LZW');
  isDem(await imp('dem.tif', tiffFile([tiffPage({ w: W, h: H, values: dem, geo, comp: 32773, rowsPerStrip: 1 })])), 'GeoTIFF PackBits, one row per strip');
  isDem(await imp('dem.tiff', tiffFile([tiffPage({ w: W, h: H, values: dem, geo, comp: 8, pred: 3, little: false })], false)), 'GeoTIFF big-endian, Deflate, floating-point predictor');
  isDem(await imp('dem.tif', tiffFile([tiffPage({ w: W, h: H, values: withHole, geo, comp: 5, pred: 3, type: 'f64', tile: 16, nodata: -9999 })])), 'GeoTIFF float64, tiled, LZW + predictor 3, GDAL_NODATA', { nod: 1 });
  const i16 = dem.map((v) => Math.round(v * 10)), ti = await imp('dem16.tif', tiffFile([tiffPage({ w: W, h: H, values: i16, type: 'i16', geo: { ...geo, geographic: true }, comp: 8, pred: 2 })]));
  ok(ti.kind === 'grid' && ti.grid.z[0][1] === 3015 && ti.grid.z[3][4] === 45 && ti.grid.geographic === true, 'GeoTIFF int16 with horizontal predictor, geographic key');
  // LZW across the 9 → 10 → 11-bit code-width changes, and a multi-page stack
  const big = Array.from({ length: 64 * 64 }, (_, i) => (i * 7919 + ((i * i) % 251)) % 256 > 127 ? 255 : (i * 31) % 97);
  const pg = (vals, comp) => tiffPage({ w: 64, h: 64, values: vals, type: 'u8', comp });
  const st = await imp('stack.tif', tiffFile([pg(big, 5), pg(big.map((v) => 255 - v), 8), pg(big, 32773)]), { threshold: 127 });
  const solid0 = big.filter((v) => v > 127).length;
  ok(st.kind === 'voxels' && st.voxels.nz === 3 && st.voxels.nx === 64 && st.format === 'TIFF stack' && st.pathway === 'voxel', 'multi-page 8-bit TIFF → 3-D voxel stack');
  let s0 = 0, s1 = 0, same = 0;
  for (let i = 0; i < 4096; i++) { s0 += st.voxels.data[i]; s1 += st.voxels.data[4096 + i]; same += st.voxels.data[i] === st.voxels.data[8192 + i] ? 1 : 0; }
  ok(s0 === solid0 && s1 === 4096 - big.filter((v) => 255 - v <= 127).length && same === 4096 && st.voxels.data[63 * 64] === (big[0] > 127 ? 1 : 0), `TIFF stack: LZW page decodes exactly across code-width changes (${s0} = ${solid0} solid), rows flipped to y-up`);
  const im = await imp('mask.tif', tiffFile([pg(big, 1)]));
  ok(im.kind === 'voxels' && im.voxels.nz === 1 && im.format === 'TIFF image' && im.stats.method === 'otsu', 'single 8-bit TIFF without georeferencing → 2-D voxels (Otsu)');
  await throws('BigTIFF', () => imp('big.tif', cat('II', le('u16', [43, 8, 0]), new Uint8Array(16))), /BigTIFF/);
  await throws('JPEG-compressed TIFF', () => imp('j.tif', tiffFile([{ ...tiffPage({ w: 2, h: 2, values: [1, 2, 3, 4], type: 'u8' }), tags: tiffPage({ w: 2, h: 2, values: [1, 2, 3, 4], type: 'u8' }).tags.map((t) => (t[0] === 259 ? [259, 3, [7]] : t)) }])), /JPEG/);
  // NetCDF classic: y descending in the file, packed shorts with scale / offset / fill
  const ys = [2035, 2025, 2015, 2005], xs = [1005, 1015, 1025, 1035, 1045];
  const nc1 = await imp('dem.nc', netcdf([['y', 4], ['x', 5]], [{ name: 'x', dims: [1], type: 'f64', data: xs, atts: [['units', 'char', 'm']] }, { name: 'y', dims: [0], type: 'f64', data: ys }, { name: 'elevation', dims: [0, 1], type: 'f32', data: dem, atts: [['units', 'char', 'm']] }]));
  isDem(nc1, 'NetCDF classic (CDF-1) float variable with coordinate variables');
  ok(nc1.stats.variable === 'elevation' && nc1.stats.units === 'm' && nc1.format === 'NetCDF classic', 'NetCDF stats (variable, units)');
  const packed = withHole.map((v) => (v === -9999 ? -32768 : Math.round((v - 100) / 0.5)));
  isDem(await imp('dem.grd', netcdf([['lat', 4], ['lon', 5], ['t', 1]], [{ name: 'lon', dims: [1], type: 'f32', data: xs }, { name: 'lat', dims: [0], type: 'f32', data: ys }, { name: 'z', dims: [2, 0, 1], type: 'i16', data: packed, atts: [['scale_factor', 'f64', [0.5]], ['add_offset', 'f64', [100]], ['_FillValue', 'i16', [-32768]]] }], 2)), 'NetCDF CDF-2 (.grd), 3-D packed shorts with scale_factor / add_offset / _FillValue', { nod: 1 });
  await throws('NetCDF-4 / HDF5 in a .nc file', () => imp('v4.nc', cat(le('u8', [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]), new Uint8Array(64))), /nccopy|classic/);
  // Surfer grids (rows from the lowest y upwards)
  const up = Array.from({ length: H }, (_, r) => dem.slice((H - 1 - r) * W, (H - r) * W)).flat();
  isDem(await imp('dem.grd', `DSAA\n5 4\n1005 1045\n2005 2035\n0.5 304.5\n${ascBody(up)}\n`), 'Surfer ASCII grid (DSAA)');
  isDem(await imp('dem.grd', cat('DSBB', le('i16', [5, 4]), le('f64', [1005, 1045, 2005, 2035, 0.5, 304.5]), le('f32', up))), 'Surfer 6 binary grid (DSBB)');
  isDem(await imp('dem.grd', cat('DSRB', le('i32', [4, 1]), 'GRID', le('i32', [72, 4, 5]), le('f64', [1005, 2005, 10, 10, 0.5, 304.5, 0, 1.70141e38]), 'DATA', le('i32', [160]), le('f64', up))), 'Surfer 7 binary grid (DSRB)');
  // band-interleaved rasters with header companions
  const bil = cat(...Array.from({ length: H }, (_, r) => cat(le('f32', dem.slice(r * W, r * W + W)), le('f32', new Array(W).fill(7)))));
  isDem(await imp('dem.bil', bil, { companion: { 'dem.hdr': F('dem.hdr', 'BYTEORDER I\nLAYOUT BIL\nNROWS 4\nNCOLS 5\nNBANDS 2\nNBITS 32\nPIXELTYPE FLOAT\nULXMAP 1005\nULYMAP 2035\nXDIM 10\nYDIM 10\n') } }), 'ESRI BIL (2 bands) + .hdr');
  isDem(await imp('dem.bsq', cat(new Uint8Array(8), be('f64', dem)), { companion: { 'dem.hdr': F('dem.hdr', 'ENVI\nsamples = 5\nlines = 4\nbands = 1\nheader offset = 8\ndata type = 5\ninterleave = bsq\nbyte order = 1\nmap info = {UTM, 1, 1, 1000, 2040, 10, 10, 40, North}\n') } }), 'ENVI BSQ float64 big-endian + .hdr');
  await throws('BIL without its header', () => imp('dem.bil', bil), /\.hdr/);
  // soundings
  const rows = []; for (let j = 0; j < 4; j++) for (let i = 0; i < 5; i++) rows.push(`${54 + i * 0.01},${24 + j * 0.01},${10 + i + j}`);
  const snd = await imp('soundings.csv', 'Longitude,Latitude,Depth (m)\n' + rows.join('\n') + '\n');
  ok(snd.kind === 'grid' && snd.grid.x.length === 5 && snd.grid.y.length === 4 && snd.grid.z[0][0] === -10 && snd.grid.z[3][4] === -17 && snd.grid.geographic && snd.warnings.some((w) => /positive down/.test(w)), 'lon, lat, depth CSV on a lattice → grid with depths as negative elevations');
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const scat = Array.from({ length: 600 }, () => { const x = rnd() * 100, y = rnd() * 50; return `${x.toFixed(3)} ${y.toFixed(3)} ${(5 + 0.1 * x).toFixed(3)}`; });
  const sc = await imp('survey.xyz', 'easting northing depth\n' + scat.join('\n') + '\n');
  ok(sc.kind === 'grid' && sc.count === 600 && sc.points.length === 1800 && sc.warnings.some((w) => /binned/.test(w)) && sc.grid.z.flat().every((v) => v <= -4.9 && v >= -15.1), `scattered easting / northing / depth → binned grid ${sc.stats.nx} × ${sc.stats.ny}, points kept`);
});

await group('point clouds', async () => {
  const hd = new Uint8Array(227), dv = new DataView(hd.buffer);
  hd.set(enc.encode('LASF')); hd[24] = 1; hd[25] = 2; dv.setUint16(94, 227, true); dv.setUint32(96, 227, true); hd[104] = 1; dv.setUint16(105, 28, true); dv.setUint32(107, 10, true);
  [0.01, 0.01, 0.001].forEach((v, k) => dv.setFloat64(131 + 8 * k, v, true)); [100, 200, 0].forEach((v, k) => dv.setFloat64(155 + 8 * k, v, true));
  const recs = cat(...Array.from({ length: 10 }, (_, i) => cat(le('i32', [i * 100, i * 50, -i * 1000]), new Uint8Array(16))));
  const las = await imp('scan.las', cat(hd, recs));
  ok(las.kind === 'points' && las.count === 10 && las.points[0] === 100 && las.points[1] === 200 && Math.abs(las.points[27] - 109) < 1e-9 && Math.abs(las.points[28] - 204.5) < 1e-9 && Math.abs(las.points[29] + 9) < 1e-9 && las.stats.pointFormat === 1, 'LAS 1.2 format 1: 10 points scaled and offset');
  const cutLas = await imp('cut.las', cat(hd, recs.subarray(0, 28 * 6 + 5)));
  ok(cutLas.count === 6 && cutLas.warnings.some((w) => /truncated/.test(w)), 'truncated LAS keeps the complete records and warns');
  hd[104] = 0x80 | 1;
  await throws('LAZ-compressed point format inside .las', () => imp('c.las', cat(hd, recs)), /laszip|Decompress/);
  const pts = await imp('scan.pts', '3\n1 2 3 100 255 0 0\n4 5 6 90 0 255 0\n7 8 9 80 0 0 255\n');
  ok(pts.kind === 'points' && pts.count === 3 && pts.points[3] === 4 && pts.points[8] === 9 && pts.warnings.some((w) => /ignored/.test(w)), 'PTS with count line, intensity and colour');
  const ptx = await imp('scan.ptx', '2\n2\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n10 20 30 1\n1 1 1 0.5\n0 0 0 0.5\n2 2 2 0.5\n3 3 3 0.5\n');
  ok(ptx.kind === 'points' && ptx.count === 3 && ptx.points[0] === 11 && ptx.points[1] === 21 && ptx.points[8] === 33 && ptx.warnings.some((w) => /empty/.test(w)), 'PTX: registration transform applied, empty return dropped');
  const xyz = await imp('cloud.xyz', '0.5 1.5 2.5\n1 2 3\n-4 5 6\n');
  ok(xyz.kind === 'points' && xyz.count === 3 && xyz.bbox.min[0] === -4 && xyz.bbox.max[2] === 6 && xyz.format === 'XYZ points / soundings', 'XYZ point list');
  const asc = await imp('cloud.asc', '// X Y Z I\n1 2 3 9\n4 5 6 9\n');
  ok(asc.kind === 'points' && asc.count === 2 && asc.pathway === 'points', '.asc without a grid header → point list');
  const csv = await imp('pts.csv', 'x,y,z\n0,0,1\n1,0,2\n0,1,3\n1,1,5\n2,2,2\n');
  ok(csv.kind === 'points' && csv.count === 5 && csv.points[14] === 2 && csv.pathway === 'points', 'CSV with x, y, z header → points');
  const txt = await imp('pts.txt', '1\t2\t3\n4\t5\t6\n');
  ok(txt.kind === 'points' && txt.count === 2, 'tab-separated TXT → points');
});

// ---- voxel volumes ---------------------------------------------------------------------------------------
function npy(descr, shape, bytes, fortran = false, major = 1) {
  const pre = major === 1 ? 10 : 12;
  let h = `{'descr': '${descr}', 'fortran_order': ${fortran ? 'True' : 'False'}, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  h = h.padEnd(Math.ceil((pre + h.length + 1) / 64) * 64 - pre - 1) + '\n';
  return cat(le('u8', [0x93]), 'NUMPY', le('u8', [major, 0]), major === 1 ? le('u16', [h.length]) : le('u32', [h.length]), h, bytes);
}
await group('image / voxel volumes', async () => {
  // 8 × 6 × 4 block (x, y, z), solid where x < 2 → porosity 0.75
  const NX = 8, NY = 6, NZ = 4, blk = new Uint8Array(NX * NY * NZ);
  for (let k = 0; k < NZ; k++) for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) blk[(k * NY + j) * NX + i] = i < 2 ? 1 : 0;
  const isBlock = (g, label, { sp } = {}) => {
    ok(g.kind === 'voxels' && g.voxels.nx === NX && g.voxels.ny === NY && g.voxels.nz === NZ && g.voxels.data instanceof Uint8Array && g.voxels.data.length === blk.length, `${label}: ${NX} × ${NY} × ${NZ} voxels (kind ${g.kind}, ${g.voxels && [g.voxels.nx, g.voxels.ny, g.voxels.nz]})`);
    ok(g.voxels.data.every((v, i) => v === blk[i]) && Math.abs(g.stats.porosity - 0.75) < 1e-12 && Math.abs(microstructure(g).porosity - 0.75) < 1e-12, `${label}: solid block reproduced, porosity ${g.stats.porosity}`);
    if (sp) ok(g.voxels.spacing.every((v, k) => Math.abs(v - sp[k]) < 1e-6), `${label}: spacing ${g.voxels.spacing}`);
  };
  isBlock(await imp('block.npy', npy('|b1', [NZ, NY, NX], blk)), 'NPY bool, C order (z, y, x)');
  const grey = Float32Array.from(blk, (v, i) => (v ? 200 : 20) + (i % 7) * 0.25);
  isBlock(await imp('block.npy', npy('<f4', [NZ, NY, NX], le('f32', Array.from(grey)), false, 2)), 'NPY float32 v2 header, Otsu threshold');
  const fort = new Uint8Array(blk.length);            // Fortran order with shape (nx, ny, nz) has the same memory layout
  fort.set(blk);
  isBlock(await imp('block.npy', npy('>i2', [NX, NY, NZ], be('i16', Array.from(fort)), true, 3)), 'NPY big-endian int16, Fortran order, v3 header');
  const inv = await imp('block.npy', npy('|u1', [NZ, NY, NX], blk), { invert: true });
  ok(Math.abs(inv.stats.porosity - 0.25) < 1e-12 && inv.voxels.data[0] === 0, 'opts.invert swaps solid and pore');
  const np = await imp('pts.npy', npy('<f8', [5, 3], le('f64', [0, 0, 0, 1, 0, 0.5, 0, 1, 0.25, 1, 1, 2, 3, 3, 3])));
  ok(np.kind === 'points' && np.count === 5 && np.points[5] === 0.5 && np.bbox.max[2] === 3, 'NPY N × 3 float array → points');
  const img = await imp('img.npy', npy('|u1', [3, 4], le('u8', [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])));
  ok(img.kind === 'voxels' && img.voxels.nx === 4 && img.voxels.ny === 3 && img.voxels.nz === 1 && img.stats.porosity === 0.5, 'NPY 2-D integer array → 2-D voxels');
  const dm = await imp('dem.npy', npy('<f8', [2, 4], le('f64', [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])));
  ok(dm.kind === 'grid' && dm.grid.z[1][2] === 6.5 && dm.grid.x.length === 4 && dm.grid.y.length === 2, 'NPY 2-D real array → grid');
  const lab = await imp('labels.npy', npy('|u1', [1, 2, 4], le('u8', [0, 1, 2, 3, 0, 0, 5, 5]))), two = await imp('two.npy', npy('|i1', [1, 2, 4], le('i8', [-1, 1, 1, -1, -1, -1, 1, -1])));
  ok(lab.stats.method === 'labels' && lab.stats.porosity === 3 / 8 && two.stats.method === 'binary' && two.stats.porosity === 5 / 8, 'labelled volume: non-zero = solid; two-valued volume: the larger value = solid');
  const z1 = await imp('arrays.npz', zip([['volume.npy', npy('|b1', [NZ, NY, NX], blk)], ['other.npy', npy('<f8', [2], le('f64', [1, 2]))]], true));
  isBlock(z1, 'NPZ (deflated), first array');
  ok(z1.stats.array === 'volume' && z1.stats.arrays === 2 && z1.warnings.some((w) => /2 arrays/.test(w)), 'NPZ reports which array was used');
  const nh = (enc2, extra = '') => `NRRD0004\n# comment\ntype: unsigned char\ndimension: 3\nsizes: ${NX} ${NY} ${NZ}\nspacings: 0.5 0.5 2\nencoding: ${enc2}\nendian: little\n${extra}\n`;
  isBlock(await imp('block.nrrd', cat(nh('raw'), blk)), 'NRRD raw', { sp: [0.5, 0.5, 2] });
  isBlock(await imp('block.nrrd', cat(nh('gzip'), new Uint8Array(zlib.gzipSync(blk)))), 'NRRD gzip');
  isBlock(await imp('block.nrrd', nh('ascii') + Array.from(blk).join(' ') + '\n'), 'NRRD ascii');
  isBlock(await imp('block.nhdr', `NRRD0004\ntype: uint8\ndimension: 3\nsizes: ${NX} ${NY} ${NZ}\nspace directions: (0.1,0,0) (0,0.2,0) (0,0,0.3)\nencoding: raw\ndata file: block.raw\n`, { companion: { 'block.raw': F('block.raw', blk) } }), 'NRRD detached header + companion', { sp: [0.1, 0.2, 0.3] });
  const mh = (file, extra = '') => `ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n${extra}ElementSpacing = 0.25 0.25 0.5\nDimSize = ${NX} ${NY} ${NZ}\nElementType = MET_USHORT\nElementDataFile = ${file}\n`;
  const u16 = le('u16', Array.from(blk, (v) => v * 1000));
  isBlock(await imp('block.mha', cat(mh('LOCAL'), u16)), 'MetaImage .mha inline', { sp: [0.25, 0.25, 0.5] });
  isBlock(await imp('block.mhd', mh('block.zraw', 'CompressedData = True\n'), { companion: { 'block.zraw': F('block.zraw', new Uint8Array(zlib.deflateSync(u16))) } }), 'MetaImage .mhd + zlib-compressed companion');
  await throws('MetaImage .mhd without its data file', () => imp('block.mhd', mh('block.raw')), /block\.raw/);
  const nii = (little) => { const h = new Uint8Array(352), dv = new DataView(h.buffer); dv.setInt32(0, 348, little); [3, NX, NY, NZ, 1, 1, 1, 1].forEach((v, k) => dv.setInt16(40 + 2 * k, v, little)); dv.setInt16(70, 16, little); dv.setInt16(72, 32, little); [1, 0.5, 0.5, 1.5].forEach((v, k) => dv.setFloat32(76 + 4 * k, v, little)); dv.setFloat32(108, 352, little); dv.setFloat32(112, 2, little); dv.setFloat32(116, -1, little); h.set(enc.encode('n+1'), 344); return cat(h, le('f32', Array.from(blk), little)); };
  isBlock(await imp('block.nii', nii(true)), 'NIfTI-1 float32 with scl_slope / scl_inter', { sp: [0.5, 0.5, 1.5] });
  isBlock(await imp('block.nii.gz', new Uint8Array(zlib.gzipSync(nii(false)))), 'NIfTI-1 big-endian, gzip (.nii.gz)');
  const cube = new Uint8Array(27).map((_, i) => (i % 3 === 0 ? 255 : 0)), rc = await imp('cube.raw', cube);
  ok(rc.kind === 'voxels' && rc.voxels.nx === 3 && rc.voxels.nz === 3 && Math.abs(rc.stats.porosity - 2 / 3) < 1e-12 && rc.warnings.some((w) => /inferred/.test(w)), 'RAW 27 bytes → 3 × 3 × 3 cube inferred');
  isBlock(await imp('block.vol', u16, { dims: [NX, NY, NZ], dtype: 'uint16', spacing: [2, 2, 2] }), 'RAW with dims + dtype', { sp: [2, 2, 2] });
  await throws('RAW without dimensions', () => imp('odd.raw', new Uint8Array(30)), /dims/);
  await throws('RAW smaller than its dimensions', () => imp('small.raw', u16, { dims: [100, 100, 100], dtype: 'uint16' }), /too small/);
  // DICOM: explicit VR little endian, one 4 × 3 slice of unsigned shorts, with an undefined-length sequence to skip
  const tag = (g, e) => le('u16', [g, e]), even = (b) => (b.length % 2 ? cat(b, ' ') : bytesOf(b));
  const el = (g, e, vr, val) => { const v = even(val); return ['OB', 'OW', 'SQ', 'UN'].includes(vr) ? cat(tag(g, e), vr, le('u16', [0]), le('u32', [v.length]), v) : cat(tag(g, e), vr, le('u16', [v.length]), v); };
  const us = (v) => le('u16', [v]), px = [0, 0, 900, 900, 0, 0, 900, 900, 10, 10, 800, 800];
  const seq = cat(tag(0x0008, 0x1140), 'SQ', le('u16', [0]), le('u32', [0xffffffff]), tag(0xfffe, 0xe000), le('u32', [0xffffffff]), el(0x0008, 0x1150, 'UI', '1.2.3'), tag(0xfffe, 0xe00d), le('u32', [0]), tag(0xfffe, 0xe0dd), le('u32', [0]));
  const dcm = (ts, pixel) => cat(new Uint8Array(128), 'DICM', el(0x0002, 0x0010, 'UI', ts), seq, el(0x0018, 0x0050, 'DS', '2.5'), el(0x0028, 0x0002, 'US', us(1)), el(0x0028, 0x0004, 'CS', 'MONOCHROME2'), el(0x0028, 0x0010, 'US', us(3)), el(0x0028, 0x0011, 'US', us(4)), el(0x0028, 0x0030, 'DS', '0.5\\0.25'), el(0x0028, 0x0100, 'US', us(16)), el(0x0028, 0x0103, 'US', us(0)), pixel);
  const dc = await imp('slice.dcm', dcm('1.2.840.10008.1.2.1', el(0x7fe0, 0x0010, 'OW', le('u16', px))), { threshold: 400 });
  ok(dc.kind === 'voxels' && dc.voxels.nx === 4 && dc.voxels.ny === 3 && dc.voxels.nz === 1 && dc.stats.porosity === 0.5 && dc.voxels.spacing[0] === 0.25 && dc.voxels.spacing[1] === 0.5 && dc.voxels.spacing[2] === 2.5, 'DICOM explicit-VR slice: 4 × 3, pixel spacing and thickness, sequence skipped');
  ok(dc.voxels.data[2] === 1 && dc.voxels.data[0] === 0 && dc.stats.method === 'threshold', 'DICOM thresholded at the given value with rows flipped to y-up');
  await throws('compressed DICOM transfer syntax', () => imp('j.dcm', dcm('1.2.840.10008.1.2.4.70', el(0x7fe0, 0x0010, 'OW', le('u16', px)))), /gdcmconv|Decompress|TIFF/);
  await throws('PNG outside a browser', () => imp('photo.png', cat(le('u8', [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), new Uint8Array(32))), /browser/);
  const mtx = await imp('mask.txt', Array.from({ length: 4 }, (_, j) => Array.from({ length: 10 }, (_q, i) => (i < 5 ? 1 : 0)).join(' ')).join('\n') + '\n');
  ok(mtx.kind === 'voxels' && mtx.voxels.nx === 10 && mtx.voxels.ny === 4 && mtx.stats.porosity === 0.5, 'numeric 0/1 matrix in a text file → 2-D voxels');
  const jv = await imp('vox.json', JSON.stringify({ voxels: [[[1, 0], [1, 0]], [[1, 0], [0, 0]]], spacing: [2, 2, 2] }));
  ok(jv.kind === 'voxels' && jv.voxels.nx === 2 && jv.voxels.nz === 2 && jv.stats.porosity === 5 / 8 && jv.voxels.spacing[0] === 2, 'JSON nested 3-D array → voxels');
});

// ---- networks -----------------------------------------------------------------------------------------------
await group('plant / piping networks', async () => {
  const net = { nodes: [{ id: 'feed', type: 'source', x: 0, y: 0, z: 0 }, { id: 'P1', type: 'Pump', name: 'HP pump', x: 10, y: 0, z: 0 }, { id: 'RO', kind: 'membrane', position: [10, 0, 5] }, { tag: 'brine', class: 'sink', x: 20, y: 0, elevation: 5 }], pipes: [{ from: 'feed', to: 'P1', L: 10, D: 0.2 }, { source: 'P1', target: 'RO', dn: 150 }, { upstream: 'RO', downstream: 'brine', length: 12, diameter: 0.1, type: 'brine line', name: 'B-1' }] };
  const isNet = (g, label, { coords = true } = {}) => {
    const s = networkSummary(g);
    ok(g.kind === 'network' && s.nodes === 4 && s.edges === 3 && g.network.nodes.every((n) => typeof n.id === 'string' && typeof n.type === 'string' && typeof n.name === 'string'), `${label}: network of 4 nodes, 3 edges`);
    ok(Math.abs(s.totalLength - 27) < 1e-9 && s.byType.pump === 1 && s.byType.pipe === 2 && s.byType['brine line'] === 1 && s.order.join('>') === 'feed>P1>RO>brine' && s.acyclic, `${label}: total length ${s.totalLength}, byType, topological order ${s.order.join('>')}`);
    ok(s.pipes.length === 3 && s.pipes[0].diameter === 0.2 && Math.abs(s.pipes[1].diameter - 0.15) < 1e-12 && s.pipes[2].name === 'B-1' && (!coords || s.maxElevationChange === 5), `${label}: pipe list (DN 150 → 0.15 m), elevation change ${s.maxElevationChange}`);
  };
  const nj = await imp('plant.json', JSON.stringify(net));
  isNet(nj, 'JSON network');
  ok(nj.warnings.some((w) => /computed from node coordinates/.test(w)) && nj.pathway === 'numeric' && nj.network.nodes[1].name === 'HP pump' && nj.network.nodes[2].z === 5, 'JSON network: liberal keys (kind, position, tag, elevation), computed length noted');
  const yaml = `# desalination train\nplant:\n  name: "Train A"   # comment\n  units:\n    - id: feed\n      type: source\n      x: 0\n      y: 0\n      z: 0\n    - {id: P1, type: Pump, name: 'HP pump', x: 10, y: 0, z: 0}\n    - id: RO\n      kind: membrane\n      position: [10, 0, 5]\n    - tag: brine\n      class: sink\n      x: 20\n      y: 0\n      elevation: 5\n  streams:\n  - from: feed\n    to: P1\n    L: 10\n    D: 0.2\n  - {source: P1, target: RO, dn: 150}\n  - upstream: RO\n    downstream: brine\n    length: 12.0\n    diameter: 1.0e-1\n    type: brine line\n    name: B-1\n`;
  isNet(await imp('plant.yaml', yaml), 'YAML network (block + inline maps, nested under "plant")');
  isNet(await imp('plant.xml', `<?xml version="1.0"?><plant><equipment><unit id="feed" type="source" x="0" y="0" z="0"/><unit id="P1" type="Pump" x="10" y="0" z="0"><name>HP pump</name></unit><unit id="RO" kind="membrane" x="10" y="0" z="5"/><unit tag="brine" class="sink" x="20" y="0" elevation="5"/></equipment><piping><pipe from="feed" to="P1" L="10" D="0.2"/><pipe source="P1" target="RO" dn="150"/><pipe upstream="RO" downstream="brine" length="12" diameter="0.1" type="brine line" name="B-1"/></piping></plant>`), 'XML network');
  const csvNet = await imp('links.csv', 'from,to,length,diameter,type,name\nfeed,P1,10,0.2,pipe,L1\nP1,RO,5,0.15,pipe,L2\nRO,brine,12,0.1,brine line,B-1\n');
  ok(csvNet.kind === 'network' && csvNet.pathway === 'network' && networkSummary(csvNet).totalLength === 27 && networkSummary(csvNet).order.join('>') === 'feed>P1>RO>brine' && networkSummary(csvNet).byType['brine line'] === 1, 'CSV connectivity table → network');
  const loop = networkSummary(await imp('loop.json', JSON.stringify({ edges: [['a', 'b', 1], ['b', 'c', 1], ['c', 'a', 1]] })));
  ok(loop.nodes === 3 && loop.order.length === 0 && loop.acyclic === false && loop.totalLength === 3, 'cyclic network: no topological order');
  await imp('evil.json', '{"__proto__": {"polluted": 1}, "constructor": {"prototype": {"polluted": 1}}, "nodes": [{"id": "a", "__proto__": {"polluted": 1}}], "edges": [{"from": "a", "to": "b"}]}');
  await imp('evil.yaml', '__proto__:\n  polluted: 1\nnodes:\n  - id: a\n    __proto__: {polluted: 1}\nedges:\n  - {from: a, to: b}\n');
  ok(({}).polluted === undefined && Object.prototype.polluted === undefined, '__proto__ / constructor keys in JSON and YAML do not pollute prototypes');
  await throws('YAML with an anchor', () => imp('a.yaml', 'base: &b\n  x: 1\nuse: *b\n'), /anchors/);
  // PCF: three pipes, an elbow and a valve (millimetres)
  const pcf = `ISOGEN-FILES ISOGEN.FLS\nUNITS-BORE MM\nUNITS-CO-ORDS MM\nUNITS-WEIGHT KGS\nPIPELINE-REFERENCE LINE-100\n    PIPING-SPEC CS150\nPIPE\n    END-POINT 0 0 0 100\n    END-POINT 1000 0 0 100\n    ITEM-CODE P1\nELBOW\n    END-POINT 1000 0 0 100\n    END-POINT 1150 150 0 100\n    CENTRE-POINT 1150 0 0\n    SKEY ELBW\nPIPE\n    END-POINT 1150.0004 150 0 100\n    END-POINT 1150 2150 0 100\nVALVE\n    END-POINT 1150 2150 0 100\n    END-POINT 1150 2350 0 100\n    TAG XV-101\nPIPE\n    END-POINT 1150 2350 0 100\n    END-POINT 1150 2350 3000 100\nMATERIALS\nITEM-CODE P1\n    DESCRIPTION PIPE\n`;
  const gp = await imp('line.pcf', pcf), sp = networkSummary(gp);
  ok(gp.kind === 'network' && sp.nodes === 6 && sp.edges === 5 && sp.byType.pipe === 3 && sp.byType.elbow === 1 && sp.byType.valve === 1, `PCF: 6 merged nodes, 5 components (${sp.nodes}, ${sp.edges})`);
  ok(Math.abs(sp.totalLength - (1 + 0.15 * Math.PI / 2 + 2 + 0.2 + 3)) < 1e-6 && sp.pipes.every((p) => p.diameter === 0.1) && sp.maxElevationChange === 3 && gp.stats.units === 'm' && sp.pipes[3].name === 'XV-101', `PCF: lengths in metres incl. elbow arc (${sp.totalLength}), bore 0.1 m, rise 3 m`);
  ok(sp.order.length === 6 && sp.order[0] === 'N1' && sp.sources.length === 1 && sp.sinks.length === 1, 'PCF: line is a single chain from source to sink');
  // IFC: pump, two pipe segments (extrusion depth / length quantity), a tank, ports and connections; millimetre model
  const ifc = `ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');\nFILE_NAME('p.ifc','2024-01-01T00:00:00',(''),(''),'','','');\nFILE_SCHEMA(('IFC4'));\nENDSEC;\nDATA;\n#1=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);\n#2=IFCUNITASSIGNMENT((#1));\n#10=IFCCARTESIANPOINT((0.,0.,0.));\n#11=IFCAXIS2PLACEMENT3D(#10,$,$);\n#12=IFCLOCALPLACEMENT($,#11);\n#13=IFCCARTESIANPOINT((1000.,0.,500.));\n#14=IFCAXIS2PLACEMENT3D(#13,$,$);\n#15=IFCLOCALPLACEMENT(#12,#14);\n#16=IFCDIRECTION((0.,1.,0.));\n#17=IFCDIRECTION((1.,0.,0.));\n#18=IFCAXIS2PLACEMENT3D(#13,#17,#16);\n#19=IFCLOCALPLACEMENT(#15,#18);\n#21=IFCLOCALPLACEMENT(#19,#14);\n#20=IFCPUMP('guidPump',$,'Feed pump P-101',$,$,#15,$,'P-101',.CIRCULATOR.);\n#30=IFCCIRCLEPROFILEDEF(.AREA.,$,$,50.);\n#31=IFCDIRECTION((0.,0.,1.));\n#32=IFCEXTRUDEDAREASOLID(#30,#11,#31,2500.);\n#33=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#32));\n#34=IFCPRODUCTDEFINITIONSHAPE($,$,(#33));\n#40=IFCPIPESEGMENT('guidPipe1',$,'Pipe 1',$,$,#12,#34,$,.RIGIDSEGMENT.);\n#41=IFCPIPESEGMENT('guidPipe2',$,'Pipe 2',$,$,#21,$,$,.RIGIDSEGMENT.);\n#50=IFCQUANTITYLENGTH('Length',$,$,4000.,$);\n#51=IFCELEMENTQUANTITY('q',$,'Qto_PipeSegmentBaseQuantities',$,$,(#50));\n#52=IFCRELDEFINESBYPROPERTIES('r',$,$,$,(#41),#51);\n#60=IFCDISTRIBUTIONPORT('p1',$,'out',$,$,$,$,.SOURCE.,.PIPE.,.PIPING.);\n#61=IFCDISTRIBUTIONPORT('p2',$,'in',$,$,$,$,.SINK.,.PIPE.,.PIPING.);\n#62=IFCDISTRIBUTIONPORT('p3',$,'out',$,$,$,$,.SOURCE.,.PIPE.,.PIPING.);\n#63=IFCDISTRIBUTIONPORT('p4',$,'in',$,$,$,$,.SINK.,.PIPE.,.PIPING.);\n#70=IFCRELNESTS('n1',$,$,$,#20,(#60));\n#71=IFCRELNESTS('n2',$,$,$,#40,(#61,#62));\n#72=IFCRELNESTS('n3',$,$,$,#41,(#63));\n#80=IFCRELCONNECTSPORTS('c1',$,$,$,#60,#61,$);\n#81=IFCRELCONNECTSPORTS('c2',$,$,$,#63,#62,$);\n#90=IFCTANK('guidTank',$,'Permeate tank',$,$,$,$,$,$);\n#91=IFCPUMPTYPE('guidType',$,'type',$,$,$,$,$,$,.CIRCULATOR.);\nENDSEC;\nEND-ISO-10303-21;\n`;
  const gi = await imp('plant.ifc', ifc), si = networkSummary(gi), byId = Object.fromEntries(gi.network.nodes.map((n) => [n.id, n]));
  ok(gi.kind === 'network' && si.nodes === 4 && si.edges === 2 && si.byType.pump === 1 && si.byType.pipe === 2 && si.byType.tank === 1 && gi.format === 'IFC (STEP text)', `IFC: pump + 2 pipe segments + tank, 2 port connections (${si.nodes}, ${si.edges})`);
  ok(byId.guidPump.name === 'Feed pump P-101' && byId.guidPump.tag === 'P-101' && Math.abs(byId.guidPump.x - 1) < 1e-12 && Math.abs(byId.guidPump.z - 0.5) < 1e-12 && byId.guidTank.x === null, 'IFC: names, tags and placements in metres (nested IfcLocalPlacement)');
  ok(Math.abs(byId.guidPipe2.x - 2.5) < 1e-12 && Math.abs(byId.guidPipe2.y - 1) < 1e-12 && Math.abs(byId.guidPipe2.z - 1) < 1e-12,`IFC: rotated child placement composed with its parent (${byId.guidPipe2.x}, ${byId.guidPipe2.y}, ${byId.guidPipe2.z})`);
  ok(Math.abs(byId.guidPipe1.length - 2.5) < 1e-12 && Math.abs(byId.guidPipe1.diameter - 0.1) < 1e-12 && Math.abs(byId.guidPipe2.length - 4) < 1e-12 && Math.abs(si.totalLength - 6.5) < 1e-12, 'IFC: pipe lengths from extrusion depth and Qto length quantity');
  ok(gi.network.edges[0].from === 'guidPump' && gi.network.edges[0].to === 'guidPipe1' && gi.network.edges[1].from === 'guidPipe1' && gi.network.edges[1].to === 'guidPipe2' && gi.warnings.some((w) => /2 of 2 segments/.test(w)), 'IFC: connectivity through nested ports, directed from SOURCE to SINK; recovery reported');
  const aml = `<?xml version="1.0" encoding="utf-8"?><CAEXFile FileName="plant.aml" SchemaVersion="2.15"><InstanceHierarchy Name="Plant"><InternalElement Name="Train1" ID="t1"><InternalElement Name="HP pump" ID="e1"><ExternalInterface Name="out" ID="i1"/><RoleRequirements RefBaseRoleClassPath="ProcessRoles/Pump"/></InternalElement><InternalElement Name="RO vessel" ID="e2"><ExternalInterface Name="in" ID="i2"/><ExternalInterface Name="brine" ID="i3"/></InternalElement><InternalElement Name="ERD" ID="e3" RefBaseSystemUnitPath="Lib/EnergyRecovery"><ExternalInterface Name="in" ID="i4"/></InternalElement><InternalLink Name="feed" RefPartnerSideA="i1" RefPartnerSideB="i2"/><InternalLink Name="brine" RefPartnerSideA="e2:brine" RefPartnerSideB="{e3}:in"/></InternalElement></InstanceHierarchy></CAEXFile>`;
  const ga = await imp('plant.aml', aml), sa = networkSummary(ga);
  ok(ga.kind === 'network' && sa.nodes === 4 && sa.edges === 2 && sa.byType.pump === 1 && sa.byType.energyrecovery === 1 && ga.network.nodes[1].parent === 't1', 'AutomationML: InternalElement hierarchy → 4 nodes with roles');
  ok(ga.network.edges[0].from === 'e1' && ga.network.edges[0].to === 'e2' && ga.network.edges[1].from === 'e2' && ga.network.edges[1].to === 'e3' && ga.network.edges[1].name === 'brine', 'AutomationML: InternalLinks resolved by interface ID and by element:interface');
  const viaXml = await imp('plant.xml', aml);
  ok(viaXml.kind === 'network' && viaXml.format === 'AutomationML / CAEX', 'generic .xml routed by its root element');
  const prm = await imp('element.json', JSON.stringify({ element: { diameter_in: 8, length_m: 1.016, area_m2: 37 }, spacer: { thickness_mil: 34, angle: 90 } }));
  ok(prm.kind === 'table' && prm.records.length === 1 && prm.records[0]['spacer.angle'] === 90 && prm.params.element.area_m2 === 37, 'JSON parameter set (parametric membrane definition) → one-row table + params');
  const tb = await imp('notes.csv', 'name,value,unit\nrecovery,0.45,-\npressure,62,bar\n');
  ok(tb.kind === 'table' && tb.headers.length === 3 && tb.records.length === 2 && tb.records[1].value === 62, 'unrecognised CSV columns → plain table');
});

// ---- analysis helpers -------------------------------------------------------------------------------------
await group('sectionOf / maskOf / gridOf', async () => {
  const cube = await imp('cube.obj', `${V.map((v) => 'v ' + v.join(' ')).join('\n')}\n${Q.map((q) => 'f ' + q.map((i) => i + 1).join(' ')).join('\n')}\n`);
  for (const axis of [0, 1, 2]) {
    const s = sectionOf(cube, { axis });
    const len = s.reduce((a, q) => a + Math.hypot(q[2] - q[0], q[3] - q[1]), 0), flat = s.flat();
    ok(s.length === 8 && Math.abs(len - 4) < 1e-12 && Math.min(...flat) === 0 && Math.max(...flat) === 1, `sectionOf(cube, axis ${axis}): unit-square outline, perimeter ${len}`);
  }
  ok(sectionOf(cube, { axis: 2, value: 5 }).length === 0, 'sectionOf outside the mesh gives no segments');
  const frac = (m) => m.flat().filter(Boolean).length / (m.length * m[0].length);
  const m1 = maskOf(cube, 0, 2, 0, 2, 40, 40, { margin: 0.25 });
  ok(m1.length === 40 && m1[0].length === 40 && frac(m1) === 0.25 && m1[20][20] === true && m1[2][2] === false && typeof m1[0][0] === 'boolean', `maskOf(cube) fitted with a 25 % margin: area fraction ${frac(m1)}`);
  ok(frac(maskOf(cube, 0, 2, 0, 2, 40, 40)) === 1 && Math.abs(frac(maskOf(cube, 0, 4, 0, 2, 80, 40)) - 0.5) < 1e-12 && frac(maskOf(cube, 0, 4, 0, 2, 80, 40, { stretch: true })) === 1, 'maskOf fit: uniform scale by default, stretch fills the window');
  ok(Math.abs(frac(maskOf(cube, -1, 2, -1, 2, 30, 30, { fit: false })) - 1 / 9) < 1e-12 && Math.abs(frac(maskOf(cube, -1, 2, -1, 2, 30, 30, { fit: false, invert: true })) - 8 / 9) < 1e-12, 'maskOf without fit keeps model coordinates; invert flips');
  const sq = await imp('sq.xy', '0 0\n2 0\n2 1\n0 1\n0 0\n');
  ok(sectionOf(sq).length === 4 && frac(maskOf(sq, 0, 4, 0, 4, 40, 40)) === 0.5, 'sectionOf / maskOf on polylines (2 × 1 rectangle fills half of a square window)');
  const sp = generate({ type: 'spacer', height: 1, length: 8, diameter: 0.5, pitch: 2, n: 320 });
  const mv = maskOf(sp, 0, 8, 0, 1, sp.voxels.nx, sp.voxels.ny);
  let same = 0;
  for (let j = 0; j < sp.voxels.ny; j++) for (let i = 0; i < sp.voxels.nx; i++) same += mv[j][i] === (sp.voxels.data[j * sp.voxels.nx + i] === 1) ? 1 : 0;
  ok(same === sp.voxels.nx * sp.voxels.ny, 'maskOf(voxels) on the native grid reproduces the voxel data');
  const sv = sectionOf(sp), per = sv.reduce((a, q) => a + Math.hypot(q[2] - q[0], q[3] - q[1]), 0);
  ok(sv.length > 100 && Math.abs(per - 4 * Math.PI * 0.5) / (2 * Math.PI) < 0.08, `sectionOf(voxels): marching-squares outline of 4 filaments, perimeter ${per.toFixed(3)} vs ${(2 * Math.PI).toFixed(3)}`);
  const blk = generate({ type: 'csg', tree: { type: 'box', min: [0.25, 0.25, 0.25], max: [0.75, 0.75, 0.75] }, bounds: { min: [0, 0, 0], max: [1, 1, 1] }, n: 20 });
  const s3 = sectionOf(blk, { axis: 0, value: 0.5 }), f3 = s3.flat();
  ok(s3.length > 0 && Math.abs(Math.min(...f3) - 0.25) < 0.03 && Math.abs(Math.max(...f3) - 0.75) < 0.03 && sectionOf(blk, { axis: 0, value: 0.05 }).length === 0, 'sectionOf(3-D voxels, axis, value) cuts the chosen slice');
  // gridOf: scattered samples of the plane z = 2x + 3y + 1, with an empty patch that must be filled
  let seed = 11; const rnd = () => ((seed = (seed * 48271) % 2147483647) / 2147483647), pts = [];
  while (pts.length < 3 * 6000) { const x = rnd(), y = rnd(); if (x > 0.42 && x < 0.58 && y > 0.42 && y < 0.58) continue; pts.push(x, y, 2 * x + 3 * y + 1); }
  const pc = await imp('plane.npy', npy('<f8', [pts.length / 3, 3], le('f64', pts))), gr = gridOf(pc, 10, 10);
  let worst = 0;
  for (let j = 0; j < 10; j++) for (let i = 0; i < 10; i++) worst = Math.max(worst, Math.abs(gr.z[j][i] - (2 * gr.x[i] + 3 * gr.y[j] + 1)));
  ok(gr.x.length === 10 && gr.y.length === 10 && gr.z.length === 10 && gr.z[0].length === 10 && gr.z.flat().every(Number.isFinite) && worst < 1e-9, `gridOf(scattered points) recovers the plane, worst cell error ${worst.toExponential(2)}`);
  const hole = Array.from(pts); for (let k = 0; k < hole.length; k += 3) if (hole[k] > 0.3 && hole[k] < 0.7 && hole[k + 1] > 0.3 && hole[k + 1] < 0.7) hole[k + 2] = NaN;
  const gh = gridOf({ kind: 'points', points: Float64Array.from(hole) }, 10, 10);
  ok(Math.abs(gh.z[4][4] - (2 * gh.x[4] + 3 * gh.y[4] + 1)) < 1e-9 && Math.abs(gh.z[5][5] - (2 * gh.x[5] + 3 * gh.y[5] + 1)) < 1e-9, 'gridOf fills empty cells from the surrounding bins');
  const dem = await imp('dem.asc', 'ncols 3\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\n3 4 5\n0 1 2\n'), rg = gridOf(dem, 5, 3);
  ok(rg.z[0][0] === 0 && rg.z[2][4] === 5 && rg.z[1][2] === 2.5 && rg.z[0][1] === 0.5 && rg.x[4] === 2.5, 'gridOf(grid) resamples bilinearly');
  const dg = dimensions(dem), cs = sectionOf(dem, { value: 2.5 });
  ok(dg.area === 2 && cs.length >= 2 && cs.every((s) => s.every(Number.isFinite)), 'grid: extent area and iso-line section');
  const gm = gridOf(cube, 2, 2);
  ok(gm.z.flat().every((v) => v === 0.5), 'gridOf(mesh) bins the mesh vertices');
});

await group('microstructure', async () => {
  // slabs normal to y: channels run along x and z
  const nx = 12, ny = 8, nz = 6, data = new Uint8Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) data[(k * ny + j) * nx + i] = j % 4 < 2 ? 1 : 0;
  const m = microstructure({ kind: 'voxels', voxels: { nx, ny, nz, data, spacing: [1, 1, 1] } });
  ok(m.porosity === 0.5 && m.solidFraction === 0.5 && m.connectedPorosity === 0.5, 'parallel channels: porosity exactly 0.5, all of it connected');
  ok(m.tortuosity.length === 3 && m.tortuosity[0] === 1 && m.tortuosity[2] === 1 && m.tortuosity[1] === null && m.percolates.join() === 'true,false,true', `parallel channels: tortuosity 1 along the channels, no percolation across (${m.tortuosity})`);
  ok(Math.abs(m.specificSurface - 0.375) < 1e-12 && Math.abs(m.meanPoreSize - 4) < 1e-12 && Math.abs(m.meanSolidSize - 4) < 1e-12, `parallel channels: specific surface ${m.specificSurface}, mean chord ${m.meanPoreSize}`);
  const m2 = microstructure({ kind: 'voxels', voxels: { nx, ny, nz, data, spacing: [2, 2, 2] } });
  ok(Math.abs(m2.specificSurface - 0.1875) < 1e-12 && m2.meanPoreSize === 8 && m2.tortuosity[0] === 1, 'spacing scales lengths and specific surface');
  // serpentine channel in a 2-D mask (true = solid)
  const W = 21, H = 9, mask = Array.from({ length: H }, () => new Array(W).fill(false));
  for (const [x, gapTop] of [[5, true], [10, false], [15, true]]) for (let j = 0; j < H; j++) if (gapTop ? j < H - 1 : j > 0) mask[j][x] = true;
  const ms = microstructure(mask);
  ok(ms.tortuosity.length === 2 && ms.percolates[0] === true && ms.tortuosity[0] > 1.5 && ms.tortuosity[0] < 3 && Math.abs(ms.porosity - (1 - 24 / 189)) < 1e-12, `serpentine 2-D mask: tortuosity ${ms.tortuosity[0].toFixed(3)} along x`);
  const closed = mask.map((r) => r.slice()); closed[H - 1][5] = true;
  const mc = microstructure(closed);
  ok(mc.percolates[0] === false && mc.tortuosity[0] === null && mc.percolates[1] === true && mc.tortuosity[1] >= 1 && mc.tortuosity[1] < 1.05 && mc.poreClusters === 2, 'blocked serpentine: no percolation along x, still open along y');
  const ring = Array.from({ length: 7 }, (_, j) => Array.from({ length: 7 }, (_q, i) => Math.max(Math.abs(i - 3), Math.abs(j - 3)) === 2)), mr = microstructure(ring);
  ok(Math.abs(mr.porosity - 33 / 49) < 1e-12 && Math.abs(mr.connectedPorosity - 24 / 49) < 1e-12 && mr.poreClusters === 2, 'sealed cavity: connected porosity counts only clusters that span the sample');
  const diag = Array.from({ length: 30 }, (_, j) => Array.from({ length: 30 }, (_q, i) => Math.abs(i - j) > 2));
  const md = microstructure(diag);
  ok(md.tortuosity[0] > 1.3 && md.tortuosity[0] < 1.5, `45° channel: chamfer tortuosity ${md.tortuosity[0].toFixed(3)} ≈ √2`);
  const dv = dimensions({ kind: 'voxels', voxels: { nx, ny, nz, data, spacing: [0.5, 0.5, 0.5] }, bbox: { min: [0, 0, 0], max: [6, 4, 3] } });
  ok(dv.volume === 36 && dv.size.join() === '6,4,3' && Math.abs(dv.area - 0.75 * 72) < 1e-9, 'dimensions(voxels): solid volume by voxel count');
});

await group('generate()', async () => {
  const por = (g) => microstructure(g, { tortuosity: false }).porosity, sameData = (a, b) => a.voxels.data.length === b.voxels.data.length && a.voxels.data.every((v, i) => v === b.voxels.data[i]);
  const shape = (g, label) => ok(g.kind === 'voxels' && g.voxels.data instanceof Uint8Array && g.voxels.data.length === g.voxels.nx * g.voxels.ny * g.voxels.nz && g.voxels.spacing.length === 3 && g.pathway === 'procedural' && Array.isArray(g.warnings) && Math.abs(g.stats.porosity - por(g)) < 1e-12, `${label}: voxel geometry ${g.voxels.nx} × ${g.voxels.ny} × ${g.voxels.nz}, porosity ${g.stats.porosity.toFixed(4)}`);
  const gy = generate({ type: 'tpms', surface: 'gyroid', porosity: 0.6, n: 32 });
  shape(gy, 'tpms gyroid');
  ok(Math.abs(por(gy) - 0.6) < 0.002 && sameData(gy, generate({ type: 'tpms', surface: 'gyroid', porosity: 0.6, n: 32 })) && microstructure(gy).percolates.every(Boolean), 'gyroid hits its porosity target, is deterministic and percolates in x, y, z');
  for (const s of ['schwarzP', 'diamond']) { const g = generate({ type: 'tpms', surface: s, n: 32, period: 0.5 }); ok(Math.abs(por(g) - 0.5) < 0.02 && g.voxels.nx === 32, `tpms ${s} at level 0: porosity ${por(g).toFixed(3)} ≈ 0.5`); }
  const sheet = generate({ type: 'tpms', surface: 'gyroid', sheet: true, porosity: 0.8, n: 32 });
  ok(Math.abs(por(sheet) - 0.8) < 0.002, `sheet gyroid hits its porosity target (${por(sheet)})`);
  const vo = generate({ type: 'voronoi', cells: 12, porosity: 0.8, seed: 3, n: 32 });
  shape(vo, 'voronoi foam');
  ok(Math.abs(por(vo) - 0.8) < 0.002 && sameData(vo, generate({ type: 'voronoi', cells: 12, porosity: 0.8, seed: 3, n: 32 })) && !sameData(vo, generate({ type: 'voronoi', cells: 12, porosity: 0.8, seed: 4, n: 32 })), 'voronoi: porosity target met, same seed → same structure, other seed → different');
  const vw = generate({ type: 'voronoi', cells: 8, wall: 0.06, seed: 2, n: [96, 96, 1] });
  ok(vw.voxels.nz === 1 && por(vw) > 0.5 && por(vw) < 0.95 && microstructure(vw).tortuosity.length === 2, `2-D voronoi with wall thickness: porosity ${por(vw).toFixed(3)}`);
  const la = generate({ type: 'lattice', pitch: 1, diameter: 0.3, angle: 90, n: 96 });
  shape(la, 'filament lattice (spacer unit cell)');
  ok(Math.abs(por(la) - (1 - (Math.PI * 0.3) / 4)) < 0.02 && Math.abs(la.bbox.max[2] - 0.6) < 1e-12 && Math.abs(la.bbox.max[0] - 2 * Math.SQRT2) < 1e-9, `lattice: porosity ${por(la).toFixed(4)} vs analytic ${(1 - (Math.PI * 0.3) / 4).toFixed(4)} for touching crossed filaments`);
  const sph = generate({ type: 'spheres', porosity: 0.45, radius: 0.08, seed: 5, n: 48 });
  shape(sph, 'sphere pack');
  ok(Math.abs(por(sph) - 0.45) < 0.01 && sameData(sph, generate({ type: 'spheres', porosity: 0.45, radius: 0.08, seed: 5, n: 48 })) && !sameData(sph, generate({ type: 'spheres', porosity: 0.45, radius: 0.08, seed: 6, n: 48 })), `spheres: porosity ${por(sph).toFixed(4)} within 0.01 of 0.45, seeded`);
  const disc = generate({ type: 'spheres', porosity: 0.7, radius: 0.05, seed: 1, n: [128, 128, 1], overlap: false });
  ok(disc.voxels.nz === 1 && Math.abs(por(disc) - 0.7) < 0.01 && disc.stats.overlap === false, `non-overlapping discs: porosity ${por(disc).toFixed(4)}`);
  const csg = generate({ type: 'csg', tree: { op: 'difference', children: [{ type: 'box', min: [0, 0, 0], max: [1, 1, 1] }, { type: 'sphere', center: [0.5, 0.5, 0.5], radius: 0.4 }] }, bounds: { min: [0, 0, 0], max: [1, 1, 1] }, n: 40 });
  shape(csg, 'CSG box minus sphere');
  ok(Math.abs(1 - por(csg) - (1 - (4 / 3) * Math.PI * 0.064)) < 0.01, `CSG: solid fraction ${(1 - por(csg)).toFixed(4)} vs ${(1 - (4 / 3) * Math.PI * 0.064).toFixed(4)}`);
  const csg2 = generate({ type: 'csg', tree: { op: 'union', children: [{ type: 'cylinder', center: [0, 0, 0], radius: 0.5, height: 2, axis: 'z' }, { op: 'intersection', children: [{ type: 'box', center: [2, 0, 0], size: [1, 1, 1] }, { type: 'sphere', center: [2, 0, 0], radius: 0.6 }] }] }, n: 80 });
  const dv = dimensions(csg2);
  ok(csg2.bbox.min[0] < -0.5 && csg2.bbox.max[0] > 2.5 && csg2.voxels.origin[0] < -0.5 && Math.abs(dv.volume - (Math.PI * 0.5 + 0.9)) < 0.12, `CSG union with automatic bounds: volume ${dv.volume.toFixed(3)} (cylinder π/2 + clipped box)`);
  const im = generate({ type: 'implicit', fn: (x, y, z) => Math.hypot(x, y, z) - 0.4, bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] }, n: 40 });
  shape(im, 'implicit sphere');
  ok(Math.abs(1 - por(im) - (4 / 3) * Math.PI * 0.064) < 0.01 && im.bbox.min[0] === -0.5, 'implicit function sampled inside its bounds (solid where fn < 0)');
  let threw = null;
  try { generate({ type: 'implicit', fn: 'Math.hypot(x,y,z)-0.4' }); } catch (e) { threw = e; }
  ok(threw instanceof Error && /function/.test(threw.message), 'implicit geometry refuses a string instead of a function');
  const sp = generate({ type: 'spacer', height: 1, length: 8, diameter: 0.5, pitch: 2, arrangement: 'zigzag', n: 400 });
  shape(sp, 'spacer-filled channel');
  const msp = microstructure(sp);
  ok(sp.voxels.nz === 1 && sp.voxels.nx === 400 && sp.voxels.ny === 50 && Math.abs(1 - msp.porosity - (4 * Math.PI * 0.0625) / 8) < 0.003 && msp.percolates[0] && msp.tortuosity[0] > 1 && msp.tortuosity[0] < 1.2 && sp.stats.filaments === 4, `spacer: 4 filaments, solid fraction ${(1 - msp.porosity).toFixed(4)}, tortuosity ${msp.tortuosity[0].toFixed(3)}`);
  const cav = generate({ type: 'spacer', height: 1, length: 8, diameter: 0.5, pitch: 2, arrangement: 'cavity', n: 400 });
  ok(cav.voxels.data.subarray(49 * 400).every((v) => v === 0) && cav.voxels.data.subarray(0, 400).some((v) => v === 1), 'cavity arrangement keeps every filament on the bottom wall');
  for (const bad of [{ type: 'nonsense' }, { type: 'csg', tree: { type: 'torus' } }, { type: 'tpms', surface: 'nope' }, { type: 'tpms', n: 5000 }]) { let e = null; try { generate(bad); } catch (x) { e = x; } ok(e instanceof Error && e.message.length < 300, `generate(${JSON.stringify(bad)}) throws: ${e && e.message.slice(0, 50)}`); }
});

await group('convert-only formats', async () => {
  const conv = FORMATS.filter((f) => f.support === 'convert');
  let n = 0, good = 0;
  for (const f of conv) for (const ext of f.ext) {
    n++;
    try { await importGeometry(F('model.' + ext, cat('\x89HDF\r\n\x1a\n binary payload ', new Uint8Array(64)))); }
    catch (e) { if (e instanceof Error && e.message === f.convert) good++; else console.log(`     .${ext}: ${e && e.message}`); }
  }
  ok(n >= 60 && good === n,`all ${n} convert-only extensions (${conv.length} formats) throw their conversion instruction`);
  for (const ext of ['x_t', 'sldprt', 'dwg', 'laz', 'h5', 'cgns', 'e57', 'gpkg', 'fbx', 'CATPart', '3dm', 'pdf', 'mat', 'parquet', '000', 'rvt', 'jt', 'exo', 'med', 'bag', 'grib2', 'sat', 'prt', 'ipt', 'dgn']) await throws(`.${ext}`, () => importGeometry(F('file.' + ext, 'x')), /Export|export|Convert|convert|Decompress|Save|save|use /);
  await throws('unknown extension', () => importGeometry(F('file.qqq', 'x')), /Unsupported/);
  await throws('shapefile side-car selected alone', () => importGeometry(F('zone.dbf', 'x')), /\.shp/);
  await throws('empty file', () => importGeometry(F('a.stl', '')), /empty/);
  await throws('no file', () => importGeometry(null), /No file/);
  await throws('oversized file', () => importGeometry({ name: 'a.stl', size: 1e9, arrayBuffer: async () => new ArrayBuffer(0) }), /larger/);
});

await group('malformed and truncated input', async () => {
  const seen = new Set(), settle = (p) => Promise.race([p.then((g) => ['ok', g], (e) => ['err', e]), new Promise((r) => setTimeout(() => r(['hang']), 4000))]);
  const kinds = ['mesh', 'polylines', 'points', 'grid', 'voxels', 'network', 'table'];
  let cases = 0;
  for (const [name, bytes, opts] of SAMPLES) {
    const key = name.replace(/^[^.]*\./, '') + ':' + bytes.length;
    if (seen.has(key) || bytes.length < 8) continue;
    seen.add(key);
    const variants = [bytes.subarray(0, bytes.length >> 1), bytes.subarray(0, Math.floor(bytes.length / 3)), bytes.subarray(0, Math.min(bytes.length, 12)), bytes.subarray(0, bytes.length - 3)];
    for (const stride of [7, 31]) { const c = bytes.slice(); for (let i = stride; i < c.length; i += stride * 3) c[i] = (c[i] * 31 + 101) & 255; variants.push(c); }
    const flip = bytes.slice(); for (let i = 0; i < flip.length; i++) if (flip[i] >= 0x31 && flip[i] <= 0x38 && i % 5 === 0) flip[i] = 0x39; variants.push(flip);
    let bad = '';
    for (const v of variants) {
      cases++;
      const [st, r] = await settle(importGeometry(F(name, v), opts));
      if (st === 'hang') bad = 'hang';
      else if (st === 'err' && !(r instanceof Error && r.message.length > 0 && r.message.length < 500 && !/undefined|is not a function|Cannot read|Invalid typed array|out of bounds|RangeError|TypeError/.test(r.message))) bad = `raw error: ${r && r.message}`;
      else if (st === 'ok' && !(r && kinds.includes(r.kind) && Array.isArray(r.warnings))) bad = 'invalid geometry returned';
      if (bad) break;
    }
    ok(!bad, `damaged ${name} (${variants.length} variants) → clean result or clean error${bad ? ' — ' + bad : ''}`);
  }
  ok(cases > 500, `${cases} damaged inputs processed without a crash or hang`);
  await throws('STEP that is not STEP', () => importGeometry(F('a.step', 'hello world')), /ISO-10303-21/);
  await throws('XML bomb-style nesting', () => importGeometry(F('a.kml', '<a>'.repeat(5000))), /nested|No line|truncated/);
  await throws('garbage glTF', () => importGeometry(F('a.glb', cat('glTF', le('u32', [2, 9999, 500, 0x4e4f534a]), '{}'))), /truncated|glTF/);
  await throws('PLY with an absurd vertex count', () => importGeometry(F('a.ply', 'ply\nformat binary_little_endian 1.0\nelement vertex 4000000000\nproperty float x\nproperty float y\nproperty float z\nend_header\n')), /impossible|truncated|malformed/);
  await throws('NPY with an absurd shape', () => importGeometry(F('a.npy', npy('<f8', [100000, 100000, 100], new Uint8Array(8)))), /truncated|large|too many/);
  await throws('corrupt gzip', () => importGeometry(F('a.nii.gz', cat(le('u8', [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3]), new Uint8Array(40).fill(0xaa)))), /corrupt|truncated|NIfTI/);
  const bomb = `<model><resources>${Array.from({ length: 9 }, (_, k) => `<object id="${k}"><components>${Array.from({ length: 60 }, () => `<component objectid="${k + 1}"/>`).join('')}</components></object>`).join('')}</resources><build><item objectid="0"/></build></model>`;
  await throws('3MF component bomb', () => importGeometry(F('bomb.3mf', zip([['3D/3dmodel.model', bomb]]))), /too many|No triangles/);
  await throws('zip without directory', () => importGeometry(F('a.3mf', cat('PK\x03\x04', new Uint8Array(40)))), /zip/);
});

console.log(`\n${pass} checks passed, ${fails} failed, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
process.exit(fails ? 1 : 0);
