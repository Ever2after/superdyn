#!/usr/bin/env bash
# Project-local entry point for the pinned, unofficial epic-fab downloader.
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bun_bin="$project_root/.tools/bun/bun"
cli_path="$project_root/.tools/epic-fab/src/cli.ts"
export XDG_CONFIG_HOME="$project_root/.auth/fab"

usage() {
  cat <<'EOF'
Usage: bash unreal_synthetic/scripts/fab.sh <command>

  login                 Print the Epic login URL and accept an authorization code
  status                Show whether local credentials have been saved (no tokens)
  library               List owned assets as JSON
  download <asset-id>    Download one owned asset for UE 5.8
  logout                Remove this project's saved login

Paste the authorization code into your terminal, not into chat.
EOF
}

command_name="${1:---help}"
case "$command_name" in
  --help|-h|help) usage; exit 0 ;;
  login|status|library|download|logout) shift ;;
  *) usage >&2; exit 1 ;;
esac

if [[ "$command_name" == download ]]; then
  if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo 'download requires one asset ID from the library listing.' >&2
    exit 1
  fi
elif [[ $# -ne 0 ]]; then
  usage >&2
  exit 1
fi

if [[ "$command_name" == status ]]; then
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ['XDG_CONFIG_HOME']) / 'epic-fab/auth.json'
try:
    data = json.loads(path.read_text())
    saved = isinstance(data, dict) and all(
        isinstance(data.get(key), str) and data[key]
        for key in ('accessToken', 'refreshToken', 'accountId', 'expiresAt')
    )
except (OSError, ValueError):
    data, saved = {}, False
result = {'credentials_saved': bool(saved)}
if saved:
    result['expires_at'] = data['expiresAt']
    try:
        expiry = datetime.fromisoformat(data['expiresAt'].replace('Z', '+00:00'))
        result['access_token_expired'] = expiry <= datetime.now(timezone.utc)
    except (ValueError, TypeError):
        result['access_token_expired'] = None
print(json.dumps(result, indent=2))
PY
  exit 0
fi

if [[ ! -x "$bun_bin" || ! -f "$cli_path" ]]; then
  echo 'Project-local downloader is missing; see unreal_synthetic/FAB_DOWNLOAD.md.' >&2
  exit 1
fi

mkdir -p "$project_root/.auth" "$XDG_CONFIG_HOME"
chmod 700 "$project_root/.auth" "$XDG_CONFIG_HOME"

case "$command_name" in
  login)
    if [[ ! -t 0 ]]; then
      echo 'Run login in your interactive IDE/server terminal.' >&2
      exit 1
    fi
    exec "$bun_bin" run "$cli_path" auth
    ;;
  library) exec "$bun_bin" run "$cli_path" list --json ;;
  download)
    asset_id="$1"
    exec "$bun_bin" run "$cli_path" download "$asset_id" \
      --engine UE_5.8 --concurrency 2 \
      --into "$project_root/inputs/source_demos/fab_downloads/$asset_id"
    ;;
  logout) exec "$bun_bin" run "$cli_path" logout ;;
esac
