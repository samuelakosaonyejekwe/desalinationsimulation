// Application identity and the addresses the build is published at. Add each further host here (and to the
// connect-src list in index.html) after publishing to it with tools/deploy-mirrors.sh.
export const APP = { name: 'BrineLab', tagline: 'Integrated desalination simulation suite' };
/** Address the Internet Archive is asked to re-capture whenever a browser first sees a new build. */
export const ARCHIVE_SOURCE = 'https://samuelakosaonyejekwe.github.io/desalinationsimulation/standalone.html';
export const MIRRORS = [
  { name: 'Primary — GitHub Pages', url: 'https://samuelakosaonyejekwe.github.io/desalinationsimulation/', note: 'Main address' },
  { name: 'Independent copy — Internet Archive', kind: 'archive', url: 'https://web.archive.org/web/29991231235959id_/https://samuelakosaonyejekwe.github.io/desalinationsimulation/standalone.html', note: 'The single-file edition preserved by the Internet Archive, on infrastructure unrelated to GitHub (the address resolves to the most recent capture). All 13 engines, file import and decision support work there. That host blocks calls to outside data services, so site data come from the built-in atlas and the map uses the built-in coastlines.' },
];
