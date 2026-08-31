#!/usr/bin/env bash
# run_agent.sh — Launch the agentY headless chat host (ComfyUI sidebar backend)
#
# The macOS/Linux counterpart of run_agent.ps1, step for step. The UI lives inside
# ComfyUI (the "agentY" tab in the left sidebar, from the separate
# agentY-comfyuiConnect repo); this script starts the backend it talks to over
# HTTP/SSE on http://127.0.0.1:<port>.
#
# Usage:
#   ./run_agent.sh                                        # port from the settings
#   ./run_agent.sh --port 5001
#   ./run_agent.sh --llm-query-templates "ollama,qwen3-coder:32b"
#   ./run_agent.sh --llm-assemble-workflow "claude,claude-sonnet-4-5"
#
# Written for bash 3.2 — the version macOS still ships as /bin/bash. No associative
# arrays, no ${var,,}, no mapfile. Tempting on a modern Linux box, and a syntax
# error on the Mac this exists for.

# No `set -e`: half the steps here are allowed to fail (an offline remote, a repo
# with no upstream, a port nobody holds). PowerShell's -AllowFail is explicit about
# which those are, and `set -e` would turn every one of them into a dead start.
set -uo pipefail

# Empty, not 5000: the port is decided in ONE place — the settings files, read by
# src/utils/settings.agent_server_url() once the venv is active (below). Deciding
# it here as well is how a Mac ended up on 5000 no matter what the settings said,
# and 5000 is the one port a Mac cannot have: ControlCenter's AirPlay Receiver
# holds it and answers there. --port still overrides everything.
PORT=""
BIND_HOST="127.0.0.1"
DEBUG=0
NO_UPDATE=0
LLM_QUERY_TEMPLATES=""
LLM_ASSEMBLE_WORKFLOW=""

C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
C_RED=$'\033[31m'; C_GRAY=$'\033[90m'; C_OFF=$'\033[0m'
if [ ! -t 1 ]; then C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_OFF=""; fi

say() { printf '%s%s%s\n' "${2:-}" "$1" "$C_OFF"; }

usage() {
  cat <<'EOF'

Usage: ./run_agent.sh [OPTIONS]

Options:
  --port <number>                      Backend port the ComfyUI sidebar connects to.
                                       Default: config/settings.default.toml
                                       (5000; 5001 on macOS, where AirPlay holds 5000).
  --host <addr>                        Bind address (default: 127.0.0.1; use 0.0.0.0 for LAN).
  --llm-query-templates "prov,model"   LLM for the QueryTemplates stage (sets env vars).
  --llm-assemble-workflow "prov,model" LLM for the AssembleWorkflow stage (sets env vars).
  --debug                              Enable hang/stall tracing to .logs/debug.log.
  --no-update                          Skip the startup check for updates on the remote.
  --help                               Show this help message and exit.

The chat UI is the agentY tab in ComfyUI's left sidebar (separate repo). Install once:
  git clone https://github.com/szprivate/agentY-comfyuiConnect into <ComfyUI>/custom_nodes/ and restart ComfyUI.

EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --port)                  PORT="${2:-}"; shift 2 ;;
    --host|--bind-host)      BIND_HOST="${2:-}"; shift 2 ;;
    --llm-query-templates)   LLM_QUERY_TEMPLATES="${2:-}"; shift 2 ;;
    --llm-assemble-workflow) LLM_ASSEMBLE_WORKFLOW="${2:-}"; shift 2 ;;
    --debug)                 DEBUG=1; shift ;;
    --no-update)             NO_UPDATE=1; shift ;;
    --help|-h)               usage; exit 0 ;;
    *) say "[run_agent] Unknown option: $1" "$C_RED"; usage; exit 2 ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

if [ "$DEBUG" = "1" ]; then
  export AGENTY_DEBUG=1
  say "[run_agent] AGENTY_DEBUG enabled - tracing hangs/stalls to .logs/debug.log" "$C_YELLOW"
fi

# ── Startup update check ────────────────────────────────────────────────────
# Fast-forwards the repos that make up the RUNNING agent (this one, plus the
# agenty_core tool layer it installs editable) to whatever the remote has.
#
# Deliberately conservative, because this runs unattended on every start and the
# working copy is the user's:
#   * local commits that aren't pushed are left alone — never rebase, never reset;
#   * --ff-only, so a diverged branch reports and stops rather than merging;
#   * a fetch failure (offline, VPN, remote down) is a shrug, not a failed start.
#
# A dirty working tree does not block the update: the agent writes generated
# artifacts into its own checkouts, so a repo is nearly always dirty. What matters
# is the INTERSECTION of what is locally modified with what the incoming commits
# touch — only that is a real conflict, and git fast-forwards around the rest.

DEPS_CHANGED=""
COMFYUI_DIR=""
REPO_MOVED=0        # set by update_repo: did HEAD actually move?

# Repo-relative paths: modified tracked + untracked. core.quotepath=false stops git
# escaping non-ASCII names into "\303\244" forms, which would never match the
# incoming list.
dirty_paths() {
  git -c core.quotepath=false -C "$1" diff --name-only HEAD 2>/dev/null
  git -c core.quotepath=false -C "$1" ls-files --others --exclude-standard 2>/dev/null
}

# Fast-forward one repo. Appends to DEPS_CHANGED when requirements/pyproject moved,
# and sets REPO_MOVED. Deliberately NOT via stdout: every status line here is
# ordinary output, and a `$(update_repo ...)` capture would swallow the lot. This is
# the one place the PowerShell original gets for free — Write-Host bypasses the
# pipeline, echo does not.
update_repo() {
  REPO_MOVED=0
  name="$1"; dir="$2"
  [ -d "$dir/.git" ] || return 0

  upstream="$(git -C "$dir" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"
  if [ -z "$upstream" ]; then
    say "[update] $name has no upstream branch - skipping." "$C_GRAY"; return 0
  fi

  if ! git -C "$dir" fetch --quiet 2>/dev/null; then
    say "[update] $name - could not reach the remote; continuing with the local copy." "$C_GRAY"
    return 0
  fi

  behind="$(git -C "$dir" rev-list --count 'HEAD..@{u}' 2>/dev/null)"
  ahead="$(git -C "$dir" rev-list --count '@{u}..HEAD' 2>/dev/null)"
  [ -n "$behind" ] && [ -n "$ahead" ] || return 0

  if [ "$behind" -eq 0 ]; then
    say "[update] $name is up to date." "$C_GRAY"; return 0
  fi
  if [ "$ahead" -gt 0 ]; then
    say "[update] $name has diverged ($ahead local, $behind remote) - skipping. Merge it yourself." "$C_YELLOW"
    return 0
  fi

  before="$(git -C "$dir" rev-parse HEAD 2>/dev/null)"
  tmp="$(mktemp -d)"; inc="$tmp/incoming"; dty="$tmp/dirty"; col="$tmp/collisions"
  git -c core.quotepath=false -C "$dir" diff --name-only 'HEAD..@{u}' 2>/dev/null | sort -u > "$inc"
  dirty_paths "$dir" | grep -v '^$' | sort -u > "$dty"
  # grep -Fxf: whole-line, fixed-string intersection. No associative array needed,
  # and it survives the spaces and non-ASCII a path can contain.
  : > "$col"
  if [ -s "$dty" ] && [ -s "$inc" ]; then
    grep -Fxf "$inc" "$dty" > "$col" 2>/dev/null
  fi
  n_dirty="$(wc -l < "$dty" | tr -d ' ')"
  n_col="$(wc -l < "$col" | tr -d ' ')"

  if [ "$n_dirty" -gt 0 ] && [ "$n_col" -eq 0 ]; then
    say "[update] $name has $n_dirty local change(s), none of them touched by this update - keeping them." "$C_GRAY"
  fi

  stashed=0
  if [ "$n_col" -gt 0 ]; then
    # Park ONLY the colliding paths. Nothing is discarded: a stash entry is a
    # commit, recoverable with `git stash pop` long after this run.
    say "[update] $name - $n_col local file(s) also changed upstream; parking them in a stash:" "$C_YELLOW"
    while IFS= read -r c; do say "           $c" "$C_YELLOW"; done < "$col"
    # NUL-separated through xargs: a path with a space stays one argument.
    if tr '\n' '\0' < "$col" | xargs -0 git -C "$dir" stash push --include-untracked \
         -m "agentY auto-update $(date '+%Y-%m-%d %H:%M:%S')" --; then
      stashed=1
    else
      say "[update] $name - could not stash; leaving the repo untouched." "$C_YELLOW"
      rm -rf "$tmp"; return 0
    fi
  fi

  say "[update] $name is $behind commit(s) behind $upstream - fast-forwarding..." "$C_CYAN"
  if ! git -C "$dir" merge --ff-only '@{u}'; then
    say "[update] $name could not fast-forward - leaving it as it was." "$C_YELLOW"
    [ "$stashed" = "1" ] && git -C "$dir" stash pop
    rm -rf "$tmp"; return 0
  fi
  after="$(git -C "$dir" rev-parse HEAD 2>/dev/null)"
  REPO_MOVED=1
  say "[update] $name updated -> $(printf '%s' "$after" | cut -c1-7)" "$C_GREEN"

  if [ "$stashed" = "1" ]; then
    if ! git -C "$dir" stash pop; then
      # The restore conflicts with what just arrived. Conflict markers in the tree
      # would be worse than useless — these are mostly generated JSON the app parses
      # on startup — so put those paths back to the new HEAD and leave the stash
      # entry alone. Nothing is lost; it just needs a human. Only the colliding
      # paths are reset, so any OTHER local work in the tree is untouched.
      tr '\n' '\0' < "$col" | xargs -0 git -C "$dir" reset -q -- 2>/dev/null
      tr '\n' '\0' < "$col" | xargs -0 git -C "$dir" checkout -q --force -- 2>/dev/null
      say "[update] $name - your version of those file(s) conflicts with the update." "$C_YELLOW"
      say "         They are SAVED in the stash and the working tree now matches the remote." "$C_YELLOW"
      say "         Recover with:  git -C \"$dir\" stash list   /   git -C \"$dir\" stash pop" "$C_YELLOW"
    else
      say "[update] $name - local changes restored on top of the update." "$C_GRAY"
    fi
  fi
  rm -rf "$tmp"

  # Only reinstall dependencies when the pulled range actually touched them.
  if [ -n "$(git -C "$dir" diff --name-only "$before..$after" -- requirements.txt pyproject.toml 2>/dev/null)" ]; then
    DEPS_CHANGED="$DEPS_CHANGED $name"
  fi
  return 0
}

SKIP_UPDATE=0
[ "$NO_UPDATE" = "1" ] && SKIP_UPDATE=1
case "${AGENTY_NO_UPDATE:-}" in ""|0|false|no) ;; *) SKIP_UPDATE=1 ;; esac

SETTINGS_LOCAL="$PROJECT_ROOT/config/settings.local.json"
if [ -f "$SETTINGS_LOCAL" ]; then
  # An explicit auto_update = false in the settings file opts out permanently.
  # Parsed with python rather than a regex: this is the app's own settings file, and
  # a grep that half-understands JSON is how a reordered key silently flips a switch
  # the user thinks they set.
  _sj="$(python3 - "$SETTINGS_LOCAL" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    d = {}
print("0" if d.get("auto_update") is False else "1")
print(d.get("comfyui_dir") or "")
PY
)"
  if [ -n "$_sj" ]; then
    [ "$(printf '%s\n' "$_sj" | sed -n 1p)" = "0" ] && SKIP_UPDATE=1
    COMFYUI_DIR="$(printf '%s\n' "$_sj" | sed -n 2p)"
  fi
fi

if [ "$SKIP_UPDATE" = "1" ]; then
  say "[update] Update check skipped." "$C_GRAY"
else
  PARENT="$(dirname "$PROJECT_ROOT")"
  update_repo 'agentY'      "$PROJECT_ROOT"
  update_repo 'agenty_core' "$PARENT/agenty_core"

  # The sidebar extension is a THIRD checkout, and often two: the clone ComfyUI
  # actually loads (<ComfyUI>/custom_nodes/agentY-comfyuiConnect) and, for anyone
  # who works on it, a dev clone beside agentY. Update every one we can find — a
  # stale installed clone is the usual reason the panel is missing a feature the
  # host already has. Candidates are cheap path guesses; nothing is searched.
  ext_seen=""; ext_updated=0; ext_i=0
  for cand in "$PARENT/agentY-comfyuiConnect" \
              "${AGENTY_COMFYUI_DIR:-}/custom_nodes/agentY-comfyuiConnect" \
              "${COMFYUI_DIR:-}/custom_nodes/agentY-comfyuiConnect" \
              "$PARENT/comfyui/custom_nodes/agentY-comfyuiConnect" \
              "$PARENT/ComfyUI/custom_nodes/agentY-comfyuiConnect" \
              "${HOME:-}/ComfyUI/custom_nodes/agentY-comfyuiConnect" \
              "${HOME:-}/comfyui/custom_nodes/agentY-comfyuiConnect"; do
    case "$cand" in /custom_nodes/*) continue ;; esac   # an unset root leaves a bare path
    [ -d "$cand/.git" ] || continue
    full="$(cd "$cand" && pwd)"
    case " $ext_seen " in *" $full "*) continue ;; esac
    ext_seen="$ext_seen $full"
    ext_i=$((ext_i + 1))
    ext_label="agentY-comfyuiConnect"
    [ "$ext_i" -gt 1 ] && ext_label="agentY-comfyuiConnect #$ext_i"
    update_repo "$ext_label" "$full"
    [ "$REPO_MOVED" = "1" ] && ext_updated=1
  done
  if [ "$ext_updated" = "1" ]; then
    say "[update] The ComfyUI sidebar extension changed - restart ComfyUI (or reload the browser for JS-only changes) to pick it up." "$C_YELLOW"
  fi
  echo
fi

# ── The virtual environment ─────────────────────────────────────────────────
# Every install below names the venv's interpreter explicitly: with a conda
# environment active (miniconda auto-activates `base` in a fresh shell) both uv and
# a bare `pip` install into THAT instead, and the packages land somewhere agentY
# never looks — the failure mode is a dependency that is "installed" and still
# missing at import time.
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  say "[run_agent] .venv not found - creating virtual environment..." "$C_YELLOW"
  # python3, not python: on macOS a bare `python` is either absent or the 2.7 stub
  # Apple used to ship, and neither builds a usable venv.
  if ! python3 -m venv .venv; then
    say "[run_agent] Could not create .venv - is python3 installed? (xcode-select --install)" "$C_RED"
    exit 1
  fi
  say "[run_agent] Installing dependencies..." "$C_YELLOW"
  "$VENV_PY" -m pip install -r requirements.txt
fi
# Clear the macOS "hidden" flag from the venv's .pth files before anything imports
# through them. Since 3.11, site.addpackage() SKIPS a hidden .pth without a word,
# and agenty_core is installed editable - it reaches the interpreter through
# exactly one .pth file. Flag that file and the whole shared tool layer is missing
# at import time while sitting correctly on disk.
#
# The installer does this too. It is repeated here because the flag can arrive
# AFTER the install: anything that hides dotfiles reaches .venv, and every .pth
# inside it if it recurses. Naming site-packages rather than walking the tree
# keeps it at milliseconds, so it costs nothing to do on every start.
if command -v chflags >/dev/null 2>&1; then
  for pth in "$PROJECT_ROOT"/.venv/lib/python*/site-packages/*.pth; do
    [ -e "$pth" ] && chflags nohidden "$pth" 2>/dev/null
  done
fi

# shellcheck disable=SC1091
. "$PROJECT_ROOT/.venv/bin/activate"

# A pull that changed requirements.txt / pyproject.toml needs the environment
# brought back in line BEFORE the app imports anything.
if [ -n "$(printf '%s' "$DEPS_CHANGED" | tr -d ' ')" ]; then
  say "[update] Dependencies changed in:$DEPS_CHANGED - reinstalling..." "$C_CYAN"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV_PY" -r requirements.txt
  else
    "$VENV_PY" -m pip install -r requirements.txt
  fi
  if [ $? -ne 0 ]; then
    say "[update] Dependency install failed - check the output above." "$C_YELLOW"
  fi
  echo
fi

# Cheap sanity check on the venv (spec lookups only, ~0.3s). Silent unless
# something required is missing — a venv can fall behind requirements.txt without a
# pull ever touching it, and the symptom is otherwise a feature that quietly does
# nothing.
if [ -f "$PROJECT_ROOT/scripts/check_env.py" ]; then
  python "$PROJECT_ROOT/scripts/check_env.py" --quiet || echo
fi

# ── Per-stage LLM overrides ─────────────────────────────────────────────────
# "provider,model" -> the env vars create_pipeline() reads.
apply_llm() {   # $1 = "provider,model"   $2 = stage prefix
  spec="$1"; stage="$2"
  [ -z "$spec" ] && return 0
  provider="$(printf '%s' "$spec" | cut -d, -f1 | sed 's/^ *//;s/ *$//')"
  model="$(printf '%s' "$spec" | cut -s -d, -f2- | sed 's/^ *//;s/ *$//')"
  export "${stage}_LLM=$provider"
  [ -z "$model" ] && return 0
  case "$provider" in
    claude) export "${stage}_ANTHROPIC_MODEL=$model" ;;
    *)      export "${stage}_OLLAMA_MODEL=$model" ;;
  esac
}
apply_llm "$LLM_QUERY_TEMPLATES"   "QUERYTEMPLATES"
apply_llm "$LLM_ASSEMBLE_WORKFLOW" "ASSEMBLEWORKFLOW"

# ── Free the target port ────────────────────────────────────────────────────
# A previous host whose Ctrl+C didn't fully stop it can linger and keep answering
# with OLD code, so a plain restart isn't enough (new routes 404). Stop any leftover
# agentY host still bound to $PORT.
#
# A function rather than inline, because it is the only step here that destroys
# something: `kill -9` on a PID chosen by a pattern match. That deserves to be
# testable, and it is — lsof/ps/kill are looked up by name, so a test can define
# them as shell functions and watch what this would have done.
#
# A process on the port that is NOT an agentY host is reported and left alone. It
# might be somebody's database.
free_port() {   # $1 = port
  local port="$1" procId cmdline killed_any=0
  command -v lsof >/dev/null 2>&1 || {
    say "[run_agent] Port check skipped (no lsof on PATH)." "$C_GRAY"; return 0; }
  for procId in $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -u); do
    [ -n "$procId" ] || continue
    cmdline="$(ps -o command= -p "$procId" 2>/dev/null)"
    case "$cmdline" in
      *agenty_ui_server*)
        say "[run_agent] Port $port held by a leftover agentY host (PID $procId) - stopping it." "$C_YELLOW"
        kill -9 "$procId" 2>/dev/null
        killed_any=1 ;;
      "")
        # lsof named a PID that ps cannot describe: it exited between the two
        # calls, or belongs to another user. Either way there is nothing to
        # identify, and killing on no evidence is exactly what this guards against.
        say "[run_agent] Port $port held by PID $procId, which cannot be identified - leaving it alone." "$C_GRAY" ;;
      *)
        say "[run_agent] WARNING: port $port is held by PID $procId ($(printf '%s' "$cmdline" | cut -c1-40)), which is not an agentY host - leaving it alone. Use --port to pick another port." "$C_RED" ;;
    esac
  done
  [ "$killed_any" = "1" ] && sleep 1
  return 0
}
# Settle the port now: free_port, the message below and the exec all need a real
# number, and the answer lives in the settings files. Asking the same resolver the
# host itself uses is what keeps the launcher and the host from disagreeing about
# where the host is — which the sidebar would then be the one to discover.
if [ -z "$PORT" ]; then
  PORT="$(python - <<'PY' 2>/dev/null
from urllib.parse import urlsplit
try:
    from src.utils.settings import agent_server_url, default_agent_port
    print(urlsplit(agent_server_url()).port or default_agent_port())
except Exception:
    print("")
PY
)"
fi
if [ -z "$PORT" ]; then
  # Settings unreadable. Say so rather than starting somewhere nobody expects.
  say "[run_agent] Could not read the port from config/ - falling back to 5000." "$C_YELLOW"
  PORT=5000
fi

free_port "$PORT"

# ── Refresh ComfyUI model caches ────────────────────────────────────────────
say "[run_agent] Refreshing ComfyUI model cache..." "$C_CYAN"
export COMFYUI_MODELS_REFRESHED=""   # clear any leftover value from a previous run
python scripts/refresh_models.py
export COMFYUI_MODELS_REFRESHED=1    # prevent re-runs in child processes
echo

echo
say "Starting agentY chat host on http://${BIND_HOST}:${PORT} ..." "$C_CYAN"
say "Open ComfyUI and click the agentY tab in the left sidebar." "$C_GREEN"
echo

exec python -m src.agenty_ui_server --host "$BIND_HOST" --port "$PORT"
