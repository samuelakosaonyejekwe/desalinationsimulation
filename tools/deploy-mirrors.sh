#!/usr/bin/env bash
# Publish the same static build to additional, fully independent hosts.
# The app uses relative paths only, so the repository root can be served from any static host.
#
#   tools/deploy-mirrors.sh cloudflare <project-name>     # needs: npx wrangler login
#   tools/deploy-mirrors.sh netlify <site-id>             # needs: npx netlify login
#   tools/deploy-mirrors.sh gitlab <git-remote-url>       # GitLab Pages (uses .gitlab-ci.yml)
#   tools/deploy-mirrors.sh codeberg <git-remote-url>     # Codeberg Pages (pushes branch "pages")
#   tools/deploy-mirrors.sh archive [url]                 # ask the Internet Archive to capture the single-file edition (no account)
#
# After a mirror is live, add its address to js/data/app.js (MIRRORS) and to the connect-src list
# in index.html, then run `node tools/build.mjs` and commit.
set -euo pipefail
cd "$(dirname "$0")/.."
node tools/build.mjs
case "${1:-}" in
  cloudflare) npx --yes wrangler pages deploy . --project-name "${2:?project name}" --branch main ;;
  netlify)    npx --yes netlify-cli deploy --prod --dir . --site "${2:?site id}" ;;
  gitlab)     git push "${2:?remote url}" HEAD:main ;;
  codeberg)   git push "${2:?remote url}" HEAD:pages ;;
  archive)    curl -s -o /dev/null -w 'Internet Archive capture: %{http_code} %{redirect_url}\n' "https://web.archive.org/save/${2:-https://samuelakosaonyejekwe.github.io/desalinationsimulation/standalone.html}" ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
