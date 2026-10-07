// Application identity and the addresses the build is published at. Add each further host here (and to the
// connect-src list in index.html) after publishing to it with tools/deploy-mirrors.sh.
export const APP = { name: 'BrineLab', tagline: 'Integrated desalination simulation suite' };
export const MIRRORS = [
  { name: 'Primary — GitHub Pages', url: 'https://samuelakosaonyejekwe.github.io/desalinationsimulation/', note: 'Main address' },
];
