// Interim geometry reader used until the full import engine is in place: STL, OBJ, DXF, GeoJSON and x-y files.
import { readGeometry, sliceMesh, polylinesToSegments, extOf } from './io.js';

export const FORMATS = [
  { ext: ['stl'], name: 'STL', pathway: 'surface', support: 'full', note: 'ASCII and binary triangle surfaces.' },
  { ext: ['obj'], name: 'Wavefront OBJ', pathway: 'surface', support: 'full', note: 'Vertices and faces.' },
  { ext: ['dxf'], name: 'DXF', pathway: 'drawing', support: 'full', note: 'Lines, polylines and circles.' },
  { ext: ['geojson'], name: 'GeoJSON', pathway: 'gis', support: 'full', note: 'Polygons and lines.' },
  { ext: ['xy'], name: 'x-y profile', pathway: 'drawing', support: 'full', note: 'Closed outline from x, y rows.' },
];
export const PATHWAYS = { surface: { title: 'Surface geometry', blurb: 'Triangulated surfaces exported from CAD.' }, drawing: { title: '2-D drawings and profiles', blurb: 'Outlines for channel and equipment sections.' }, gis: { title: 'GIS', blurb: 'Coastlines and site outlines.' }, procedural: { title: 'Procedural geometry', blurb: 'Generated without a file.' } };
export const SUITE_GEOMETRY = {};
export const formatOf = (name) => FORMATS.find((f) => f.ext.includes(extOf(name))) || null;
export async function importGeometry(file) {
  const m = formatOf(file.name);
  if (!m) throw new Error('The full geometry import engine (CAD, meshes, GIS rasters, point clouds, voxels, networks) arrives in the next update. For now use STL, OBJ, DXF or GeoJSON.');
  const g = await readGeometry(file);
  return { ...g, format: m.name, pathway: m.pathway, warnings: [], stats: {} };
}
export const sectionOf = (g) => (g.kind === 'mesh' ? sliceMesh(g) : g.kind === 'polylines' ? polylinesToSegments(g.polylines) : []);
export const maskOf = () => null;
export const gridOf = () => null;
export const microstructure = () => null;
export const networkSummary = () => null;
export function dimensions(g) { return g.bbox ? { size: g.bbox.max.map((v, i) => v - g.bbox.min[i]) } : null; }
export function generate() { throw new Error('Procedural geometry arrives with the full import engine in the next update.'); }
