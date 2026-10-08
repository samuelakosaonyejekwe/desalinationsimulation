// Application identity and the addresses the build is published at. Add each further host here (and to the
// connect-src list in index.html) after publishing to it with tools/deploy-mirrors.sh.
export const APP = { name: 'BrineLab', tagline: 'Integrated desalination simulation suite' };
export const MIRRORS = [
  { name: 'Primary — GitHub Pages', url: 'https://samuelakosaonyejekwe.github.io/desalinationsimulation/', note: 'Main address' },
  { name: 'Independent copy — Internet Archive', kind: 'archive', url: 'https://web.archive.org/web/2id_/https://samuelakosaonyejekwe.github.io/desalinationsimulation/standalone.html', note: 'The single-file edition preserved by the Internet Archive, on infrastructure unrelated to GitHub. All 13 engines, file import and decision support work there; live site data and map tiles are not available from that address.' },
];
