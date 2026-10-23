#!/usr/bin/env bash
# install_agent.sh — One-shot installer / bootstrapper for the full agentY stack.
#
# The macOS (and Linux) counterpart of install_agent.ps1: the same eight stages, in
# the same order, with the same switches. Sets up the five repos that make up
# agentY, prompts for the secrets it needs, and drops the chat UI into your ComfyUI:
#
#   * agentY                (this repo)  - the Strands chat host / pipeline
#   * agentY-core           (sibling)    - shared ComfyUI/HF/web/file tool layer
#                                          (installed editable; required)
#   * agentY-mcp            (sibling)    - the MCP-server / Claude-Desktop variant
#                                          (optional; skip with --skip-mcp)
#   * agentY-comfyuiConnect (into ComfyUI/custom_nodes) - the sidebar chat UI
#
# The UI is the "agentY" tab inside ComfyUI. Conversations persist to a local
# SQLite file.
#
# Usage:
#   ./install_agent.sh
#   ./install_agent.sh --comfyui-path ~/ComfyUI --skip-mcp
#
# Written for bash 3.2 — the version macOS still ships as /bin/bash.

set -uo pipefail

COMFYUI_PATH=""
PARENT_DIR=""
SKIP_MCP=0
SKIP_COMFY_NODE=0
NON_INTERACTIVE=0

C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
C_RED=$'\033[31m'; C_GRAY=$'\033[90m'; C_WHITE=$'\033[97m'; C_OFF=$'\033[0m'
if [ ! -t 1 ]; then C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_WHITE=""; C_OFF=""; fi

header()  { printf '\n%s===  %s  ===%s\n' "$C_CYAN" "$1" "$C_OFF"; }
success() { printf '%s  [ok] %s%s\n' "$C_GREEN" "$1" "$C_OFF"; }
info()    { printf '%s  [i]  %s%s\n' "$C_YELLOW" "$1" "$C_OFF"; }
fail()    { printf '%s  [!]  %s%s\n' "$C_RED" "$1" "$C_OFF"; }
plain()   { printf '%s%s%s\n' "${2:-$C_WHITE}" "$1" "$C_OFF"; }
die()     { echo; fail "$1"; exit "${2:-1}"; }

usage() {
  cat <<'EOF'

Usage: ./install_agent.sh [OPTIONS]

Options:
  --comfyui-path <dir>   Path to your ComfyUI install (the folder containing
                         custom_nodes/). Auto-detected when omitted.
  --parent-dir <dir>     Where the sibling repos (agentY-core, agentY-mcp) live or
                         will be cloned. Defaults to this repo's parent directory.
  --skip-mcp             Do not clone / set up the agentY-mcp sibling repo.
  --skip-comfy-node      Do not touch ComfyUI (skip locating it and installing the
                         sidebar node).
  --non-interactive      Never prompt. Use existing values / defaults only.
  --help                 Show this help message and exit.

EOF
}

ORIGINAL_ARGS=("$@")     # handed to the new installer when an update replaces this one
# The Python a NEW .venv is made with; an existing .venv keeps its interpreter.
PYTHON_VERSION="${AGENTY_PYTHON_VERSION:-3.12}"

while [ $# -gt 0 ]; do
  case "$1" in
    --comfyui-path)    COMFYUI_PATH="${2:-}"; shift 2 ;;
    --parent-dir)      PARENT_DIR="${2:-}"; shift 2 ;;
    --skip-mcp)        SKIP_MCP=1; shift ;;
    --skip-comfy-node) SKIP_COMFY_NODE=1; shift ;;
    --non-interactive) NON_INTERACTIVE=1; shift ;;
    --help|-h)         usage; exit 0 ;;
    *) fail "Unknown option: $1"; usage; exit 2 ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "$PARENT_DIR" ] || PARENT_DIR="$(dirname "$PROJECT_ROOT")"
IS_MAC=0
[ "$(uname -s)" = "Darwin" ] && IS_MAC=1

# -- Low-level helpers --------------------------------------------------------

get_env_value() {   # $1 = file  $2 = key
  [ -f "$1" ] || return 0
  sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*//p" "$1" | head -n 1
}

set_env_value() {   # $1 = file  $2 = key  $3 = value
  local file="$1" key="$2" value="$3" tmp
  tmp="$(mktemp)"
  if [ -f "$file" ] && grep -q "^[[:space:]]*$key[[:space:]]*=" "$file"; then
    # awk, not sed -i: a token can contain the / and & that sed's replacement treats
    # as syntax, and BSD sed's -i wants an argument GNU sed's does not.
    #
    # The value arrives through the ENVIRONMENT rather than -v, because awk expands
    # escape sequences inside a -v assignment: a token containing \d came back out
    # as d, one character shorter and silently wrong. That is the worst shape a bug
    # can take here — the key reads back as "set", looks right at a glance, and only
    # fails later as an authentication error nobody traces to the installer.
    _AGENTY_ENV_VALUE="$value" awk -v k="$key" '
      BEGIN { v = ENVIRON["_AGENTY_ENV_VALUE"] }
      $0 ~ "^[[:space:]]*" k "[[:space:]]*=" && !done { print k "=" v; done=1; next }
      { print }
    ' "$file" > "$tmp"
  else
    [ -f "$file" ] && cat "$file" > "$tmp"
    printf '%s=%s\n' "$key" "$value" >> "$tmp"
  fi
  mv "$tmp" "$file"
}

is_placeholder() {  # the .env_example stubs and blanks are "not yet set"
  case "${1:-}" in
    "" | "hf_..." | "sk-ant-..." | "comfyui-..." | "sk-...") return 0 ;;
    *) return 1 ;;
  esac
}

masked() {
  if is_placeholder "${1:-}"; then printf '(not set)'; return; fi
  if [ "${#1}" -le 8 ]; then printf '%s' "$(printf '%*s' "${#1}" '' | tr ' ' '*')"; return; fi
  printf '%s****...' "$(printf '%s' "$1" | cut -c1-4)"
}

# Prompt for a secret, keeping the existing value on <Enter>. Echoes the resolved
# value on stdout and writes it into $1 when the user supplies a new one; every
# status line goes to stderr so the capture stays clean.
read_secret() {   # $1 = file  $2 = key  $3 = label  $4 = help
  local file="$1" key="$2" label="$3" help="$4" cur entered
  cur="$(get_env_value "$file" "$key")"
  if [ "$NON_INTERACTIVE" = "1" ]; then
    is_placeholder "$cur" || printf '%s' "$cur"
    return 0
  fi
  {
    echo
    plain "  $label" "$C_WHITE"
    [ -n "$help" ] && plain "    $help" "$C_GRAY"
  } >&2
  if is_placeholder "$cur"; then
    printf '%s' "    $key [Enter = skip]: " >&2
  else
    printf '%s' "    $key [Enter = keep $(masked "$cur")]: " >&2
  fi
  IFS= read -r entered
  entered="$(printf '%s' "${entered:-}" | sed 's/^ *//;s/ *$//')"
  if [ -n "$entered" ]; then
    set_env_value "$file" "$key" "$entered"
    success "$key set" >&2
    printf '%s' "$entered"
    return 0
  fi
  is_placeholder "$cur" || printf '%s' "$cur"
}

# -- Release channel ----------------------------------------------------------
# "stable" (the default) follows each repository's `stable` branch, which only
# moves when a release is made - all five repositories together, so what runs is
# always a set that was tested together. "dev" follows the repository's `dev`
# branch, where new work lands first, commit by commit. Set update_channel in
# config/settings.local.json, or AGENTY_UPDATE_CHANNEL.
#
# Following is done by pointing the checked-out branch's upstream at
# origin/stable (or back at the default branch). Nothing is checked out and no
# file changes here; the fast-forward below then simply has a different target.
update_channel() {
  local c="${AGENTY_UPDATE_CHANNEL:-}" f="$PROJECT_ROOT/config/settings.local.json"
  if [ -z "$c" ] && [ -f "$f" ]; then
    c="$(python3 - "$f" <<'PY' 2>/dev/null
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("update_channel") or "")
except Exception:
    print("")
PY
)"
  fi
  case "$(printf '%s' "$c" | tr '[:upper:]' '[:lower:]' | tr -d ' \r\n')" in dev) echo dev ;; *) echo stable ;; esac
}

# Call after a fetch. Leaves a detached HEAD and a repository without a `stable`
# branch exactly as they are.
set_channel_upstream() {   # $1 = dir  $2 = channel
  local dir="$1" channel="$2" branch up head target
  branch="$(git -C "$dir" branch --show-current 2>/dev/null)"
  [ -n "$branch" ] || return 0
  up="$(git -C "$dir" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)" || up=""
  if [ "$channel" = "stable" ]; then
    if git -C "$dir" rev-parse -q --verify refs/remotes/origin/stable >/dev/null 2>&1 && [ "$up" != "origin/stable" ]; then
      git -C "$dir" branch -q --set-upstream-to=origin/stable >/dev/null 2>&1
    fi
  else
    # dev: the repository's `dev` branch, where new work lands first. Only a
    # branch that was following a release line (stable, or the default branch)
    # is re-pointed - a feature branch keeps its own upstream.
    head="$(git -C "$dir" symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null)" || head=""
    target="$head"
    git -C "$dir" rev-parse -q --verify refs/remotes/origin/dev >/dev/null 2>&1 && target="origin/dev"
    if [ -n "$target" ] && [ "$up" != "$target" ] && { [ "$up" = "origin/stable" ] || [ "$up" = "$head" ]; }; then
      git -C "$dir" branch -q --set-upstream-to="$target" >/dev/null 2>&1
    fi
  fi
  return 0
}

# Fast-forward the checkout at $2 to its upstream, and say what happened.
#
# The same rules as run_agent.sh's update_repo, for the same reason: the agent
# rewrites tracked files in its own checkouts (config/models.json on every start,
# saved templates in agentY-core), so a plain `git pull` refuses as soon as the
# remote touches one of them - and an installer that then says "up to date"
# leaves an old agentY next to a new agentY-core. Only the local files the
# incoming commits also change are parked in a stash; nothing is discarded.
update_checkout() {   # $1 = name  $2 = dir
  local name="$1" dir="$2" upstream behind ahead f stashed=0
  local -a collisions=()
  upstream="$(git -C "$dir" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)" || upstream=""
  if [ -z "$upstream" ]; then info "$name has no upstream branch - left as it is"; return 0; fi
  if ! git -C "$dir" fetch --quiet 2>/dev/null; then
    fail "$name - could not reach the remote. NOT updated; continuing with the local copy."
    return 0
  fi
  set_channel_upstream "$dir" "$CHANNEL"
  upstream="$(git -C "$dir" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)" || upstream=""
  behind="$(git -C "$dir" rev-list --count 'HEAD..@{u}' 2>/dev/null || echo 0)"
  ahead="$(git -C "$dir" rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)"
  if [ "${behind:-0}" = "0" ]; then
    if [ "${ahead:-0}" != "0" ] && [ "$upstream" = "origin/stable" ]; then
      info "$name is $ahead commit(s) ahead of the stable release - left as it is"
    else
      success "$name up to date"
    fi
    return 0
  fi
  if [ "${ahead:-0}" != "0" ]; then
    fail "$name has $ahead local commit(s) and is $behind behind $upstream. NOT updated - merge it yourself."
    return 0
  fi

  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if git -c core.quotepath=false -C "$dir" diff --name-only 'HEAD..@{u}' 2>/dev/null | grep -Fxq -- "$f"; then
      collisions+=("$f")
    fi
  done < <( { git -c core.quotepath=false -C "$dir" diff --name-only HEAD 2>/dev/null
              git -c core.quotepath=false -C "$dir" ls-files --others --exclude-standard 2>/dev/null; } | sort -u )

  if [ "${#collisions[@]}" -gt 0 ]; then
    info "$name - ${#collisions[@]} locally changed file(s) also changed upstream; parking them in a stash:"
    for f in "${collisions[@]}"; do plain "         $f" "$C_GRAY"; done
    if ! git -C "$dir" stash push --include-untracked -m "agentY installer $(date '+%Y-%m-%d %H:%M:%S')" -- "${collisions[@]}"; then
      fail "$name - could not stash those files. NOT updated."
      return 0
    fi
    stashed=1
  fi

  info "$name is $behind commit(s) behind $upstream - fast-forwarding"
  if ! git -C "$dir" merge --ff-only '@{u}'; then
    [ "$stashed" = "1" ] && git -C "$dir" stash pop
    fail "$name could not be fast-forwarded. NOT updated - see git's message above."
    return 0
  fi
  if [ "$stashed" = "1" ] && ! git -C "$dir" stash pop; then
    # The restore conflicts with what just arrived. Conflict markers in generated
    # JSON the app parses at startup would stop it starting, so those paths go
    # back to the new HEAD and the stash entry is kept.
    git -C "$dir" reset -q -- "${collisions[@]}" 2>/dev/null
    git -C "$dir" checkout -q --force -- "${collisions[@]}" 2>/dev/null
    info "$name - your version of those file(s) conflicts with the update. It is SAVED in"
    info "the stash (git -C \"$dir\" stash list); the working tree now matches the remote."
  fi
  success "$name updated -> $(git -C "$dir" rev-parse --short HEAD 2>/dev/null)"
}

# Clone $2 into $3 if missing; otherwise fast-forward it (update_checkout).
ensure_repo() {   # $1 = name  $2 = url  $3 = dir  $4 = "required"|""
  local name="$1" url="$2" dir="$3" required="${4:-}"
  if [ -d "$dir/.git" ]; then
    update_checkout "$name" "$dir"
    return 0
  fi
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    info "$name exists at $dir but is not a git checkout - leaving it untouched"
    return 0
  fi
  info "Cloning $name -> $dir"
  # A new clone starts on its channel's branch: `stable` (the release) or `dev`.
  # A repository without that branch is cloned as it is.
  if git clone -q --branch "$CHANNEL" "$url" "$dir" 2>/dev/null; then
    success "$name cloned"
    return 0
  fi
  [ -d "$dir/.git" ] && rm -rf "$dir"
  if git clone "$url" "$dir"; then
    success "$name cloned"
    return 0
  fi
  # A clone that failed at checkout leaves a .git behind; the next run would take
  # that for an install and report it up to date. The folder was missing or empty
  # before this call, so removing it loses nothing.
  [ -d "$dir/.git" ] && rm -rf "$dir"
  if [ "$required" = "required" ]; then
    die "Could not clone required repo $name from $url - see git's message above."
  fi
  fail "$name could not be cloned - see git's message above. Continuing without it."
}

venv_python() { printf '%s/.venv/bin/python' "$1"; }

# macOS: leave exactly one OpenMP runtime in the venv. A no-op anywhere else.
#
# torch and faiss-cpu each ship their own libomp.dylib in their macOS wheels, and
# both get mapped into the agent host, which imports torch for SAM3 grounding and
# faiss for the memory index. The SECOND one to actually run OpenMP work - not to
# be imported, to run work - aborts the process:
#
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
#   initialized ... Abort trap: 6
#
# It fires in either import order, because it is about which library reaches
# OpenMP first at runtime, so it reads as a random crash rather than a broken
# install.
#
# The documented workaround, KMP_DUPLICATE_LIB_OK=TRUE, is documented as unsafe:
# it lets two runtimes coexist, each with its own thread pool, and its own error
# text warns it "may cause crashes or silently produce incorrect results". Wrong
# vector-search results that nobody notices is a bad trade for a fixed crash.
#
# So there is one runtime instead: faiss's copy becomes a symlink to torch's.
# Both are LLVM libomp at the same ABI (3.20), and torch's is the newer build.
# Verified with both libraries doing heavy threaded work in one process - one
# runtime mapped, no abort, and faiss's nearest-neighbour answers still exact.
#
# The original is kept alongside as libomp.dylib.orig, so this is reversible, and
# it is re-applied after every install because reinstalling faiss restores its own.
# Only when BOTH are present: torch is optional here, and a venv without it has
# nothing to collide with.
dedupe_openmp() {   # $1 = venv dir
  local sp faiss_omp torch_omp
  [ "$IS_MAC" = "1" ] || return 0
  for sp in "$1"/lib/python*/site-packages; do
    [ -d "$sp" ] || continue
    faiss_omp="$sp/faiss/.dylibs/libomp.dylib"
    torch_omp="$sp/torch/lib/libomp.dylib"
    [ -e "$torch_omp" ] || continue
    [ -e "$faiss_omp" ] || continue
    [ -L "$faiss_omp" ] && continue          # already pointed at torch's
    cp -p "$faiss_omp" "$faiss_omp.orig" 2>/dev/null
    if ln -sf ../../torch/lib/libomp.dylib "$faiss_omp" 2>/dev/null; then
      info "Linked faiss's libomp to torch's (one OpenMP runtime, avoids OMP Error #15)"
    fi
  done
  return 0
}


# Clear the macOS "hidden" flag from a venv's .pth files. A no-op anywhere else:
# chflags is BSD, and no other platform has the flag to begin with.
#
# Since 3.11, site.addpackage() SKIPS any .pth file carrying UF_HIDDEN, silently
# and with no warning. agenty_core is installed editable, so it reaches the
# interpreter through exactly one .pth file - flag that file and `import
# agenty_core` fails while the package, its dist-info and its finder all sit
# correctly on disk. The install looks broken in a way it is not, and the obvious
# remedy makes it worse: the requirement is already satisfied, so reinstalling
# changes nothing and you are left staring at a package you can see and cannot
# import.
#
# Anything that hides dotfiles can do this - .venv is one, and a pass that
# recurses reaches every .pth inside it. Clearing costs milliseconds, so it is
# done on every install and every start rather than diagnosed a second time.
unhide_pth() {   # $1 = venv dir
  local pth
  command -v chflags >/dev/null 2>&1 || return 0
  [ -d "$1" ] || return 0
  # site-packages by name rather than `find`: the .pth files that matter are all
  # there, and walking a 45,000-file venv to reach them takes two seconds.
  for pth in "$1"/lib/python*/site-packages/*.pth; do
    [ -e "$pth" ] && chflags nohidden "$pth" 2>/dev/null
  done
  return 0
}

report_torch() {  # $1 = dir
  # No CUDA branch here, and none is missing: on a Mac the PyPI wheel is the right
  # build. torch talks to the GPU through Metal (MPS), which ships in that same
  # wheel, so the ~3 GB CUDA-index dance install_agent.ps1 does on Windows has no
  # macOS equivalent to do. Worth SAYING, though — SAM3 grounding is the feature
  # that cares, and "did it find the GPU?" is exactly what you want to know now
  # rather than a minute into the first call.
  local py; py="$(venv_python "$1")"
  [ -x "$py" ] || return 0
  "$py" - <<'PY' 2>/dev/null
try:
    import torch
except Exception:
    raise SystemExit(0)
mps = getattr(torch.backends, "mps", None)
if mps is not None and mps.is_available():
    print(f"  [ok] torch {torch.__version__} — Metal (MPS) available; SAM3 grounding will use the GPU")
elif torch.cuda.is_available():
    print(f"  [ok] torch {torch.__version__} — CUDA available")
else:
    print(f"  [i]  torch {torch.__version__} — no GPU backend; SAM3 grounding falls back to CPU (~1 min/call)")
PY
}

setup_venv() {   # $1 = name  $2 = dir  $3 = "with-torch"|""
  local name="$1" dir="$2" with_torch="${3:-}" venv py
  venv="$dir/.venv"; py="$(venv_python "$dir")"
  if [ ! -d "$venv" ] || [ ! -x "$py" ]; then
    if [ -d "$venv" ]; then info "$name .venv incomplete - recreating"; rm -rf "$venv"; fi
    # Named, not whatever `python` uv meets first on this machine: 3.12 is what
    # agentY is developed and tested on. uv downloads it when it is not installed.
    info "Creating $name .venv (uv venv, Python $PYTHON_VERSION)"
    if ! ( cd "$dir" && uv venv .venv --python "$PYTHON_VERSION" ); then
      info "Python $PYTHON_VERSION is not available - using uv's default interpreter"
      ( cd "$dir" && uv venv .venv ) || die "uv venv ($name) failed."
    fi
  else
    info "$name .venv already exists"
  fi
  # requirements.txt names `-e ../agentY-core`. Where the folder still has its old
  # name (agenty_core), scripts/core_dir.py renames it and leaves a link behind.
  if [ -x "$py" ] && [ -f "$PROJECT_ROOT/scripts/core_dir.py" ]; then
    "$py" "$PROJECT_ROOT/scripts/core_dir.py" || true
  fi
  [ -f "$dir/requirements.txt" ] || die "requirements.txt not found in $dir."
  info "Installing $name dependencies (uv pip install -r requirements.txt)"
  # --python: name the target interpreter. With a conda environment active
  # (miniconda auto-activates `base`), uv installs into THAT rather than the .venv
  # we just made, and the whole dependency set lands somewhere agentY never looks —
  # an install that reports success and imports nothing.
  # requirements.lock: the versions the release was tested with, as constraints.
  # requirements.txt alone only sets minimums.
  local lock_args=""
  if [ -f "$dir/requirements.lock" ]; then
    info "Using the versions pinned in requirements.lock"
    lock_args="-c requirements.lock"
  fi
  # shellcheck disable=SC2086  (two words on purpose)
  ( cd "$dir" && uv pip install --python "$py" -r requirements.txt $lock_args ) \
    || die "uv pip install ($name) failed."
  unhide_pth "$venv"
  dedupe_openmp "$venv"
  [ "$with_torch" = "with-torch" ] && report_torch "$dir"
  success "$name environment ready"
}

# Import-check every dependency agentY names, in the venv that will run it. Most of
# them are also somebody else's transitive dep, so a gap in requirements.txt
# otherwise stays invisible until the machine that resolved differently quietly
# loses a feature.
test_environment() {   # $1 = dir
  local py script; py="$(venv_python "$1")"; script="$1/scripts/check_env.py"
  if [ ! -x "$py" ] || [ ! -f "$script" ]; then
    info "Dependency check skipped (no venv python or scripts/check_env.py)"
    return 0
  fi
  if "$py" "$script" --gpu; then
    success "Every required dependency imports"
  else
    fail "Required dependencies are missing - see the list above."
    info "Re-run after fixing:  .venv/bin/python scripts/check_env.py"
  fi
}

# Echoes the .env path; returns non-zero instead of calling `die`.
#
# It is used as `ENV_FILE="$(ensure_env_file "$dir")"`, and command substitution
# runs in a SUBSHELL: an `exit` there ends only that subshell. The script would
# have carried on with ENV_FILE holding the text of the error message and written
# the user's API keys to a path made of it.
ensure_env_file() {   # $1 = dir -> echoes the .env path
  local dir="$1"
  if [ ! -f "$dir/.env" ]; then
    if [ ! -f "$dir/.env_example" ]; then
      fail ".env_example not found in $dir." >&2
      return 1
    fi
    cp "$dir/.env_example" "$dir/.env" || { fail "could not create $dir/.env" >&2; return 1; }
    info "Created .env from .env_example in $dir" >&2
  fi
  printf '%s/.env' "$dir"
}

# Accept the ComfyUI root, or a common wrapper folder one level up.
test_comfyui_dir() {   # $1 = path -> echoes the resolved root, or nothing
  local p="${1:-}" cand
  [ -n "$p" ] || return 0
  # Expand a leading ~ the shell did not, e.g. when the path arrived from `read`.
  case "$p" in "~"/*) p="$HOME/${p#\~/}" ;; "~") p="$HOME" ;; esac
  [ -d "$p" ] || return 0
  for cand in "$p" "$p/ComfyUI" "$p/comfyui"; do
    if [ -d "$cand/custom_nodes" ]; then ( cd "$cand" && pwd ); return 0; fi
  done
}

find_comfyui() {   # $1 = hint -> echoes the resolved root, or nothing
  local hint="${1:-}" c hit
  for c in "$hint" \
           "$PARENT_DIR/ComfyUI" "$PARENT_DIR/comfyui" \
           "${HOME:-}/ComfyUI" "${HOME:-}/comfyui" \
           "${HOME:-}/Documents/ComfyUI" "${HOME:-}/Documents/comfyui" \
           "${HOME:-}/Library/Application Support/ComfyUI" \
           "/Applications/ComfyUI"; do
    [ -n "$c" ] || continue
    hit="$(test_comfyui_dir "$c")"
    if [ -n "$hit" ]; then printf '%s' "$hit"; return 0; fi
  done
}

# -- Multi-select menu --------------------------------------------------------
# select_many: Up/Down (or k/j) move, Space ticks, 'a' ticks all, Enter confirms.
#   in : SM_TITLE; SM_LABELS, SM_HINTS, SM_SEL = parallel arrays (SM_SEL is 0/1)
#   out: SM_RESULT = the chosen indexes, space separated (possibly empty)
# Without an interactive terminal it asks for numbers instead ("1,3"; Enter =
# keep the pre-ticked ones). With --non-interactive it asks nothing and returns the
# pre-ticked items, or none when SM_NONE_SILENT=1. SM_ASSUME_TTY=1 forces the
# key-driven menu on piped input (the tests drive it that way).
SM_TITLE=""; SM_LABELS=(); SM_HINTS=(); SM_SEL=(); SM_RESULT=""; SM_NONE_SILENT=0

sm_collect() {
  local i; SM_RESULT=""
  for ((i = 0; i < ${#SM_LABELS[@]}; i++)); do
    [ "${SM_SEL[$i]}" = "1" ] && SM_RESULT="$SM_RESULT $i"
  done
  SM_RESULT="${SM_RESULT# }"
}

sm_draw() {   # $1 = cursor row  $2 = "redraw": move up first
  local i n=${#SM_LABELS[@]} cur="$1" mark line cols
  cols="$(tput cols 2>/dev/null || echo 80)"
  [ "${2:-}" = "redraw" ] && printf '\033[%dA' "$n"
  for ((i = 0; i < n; i++)); do
    mark=" "; [ "${SM_SEL[$i]}" = "1" ] && mark="x"
    line="  $([ "$i" = "$cur" ] && echo ">" || echo " ") [$mark] ${SM_LABELS[$i]}"
    [ "$i" = "$cur" ] && [ -n "${SM_HINTS[$i]:-}" ] && line="$line  - ${SM_HINTS[$i]}"
    line="${line:0:$((cols - 1))}"
    if [ "$i" = "$cur" ]; then printf '\033[2K\033[36m%s\033[0m\n' "$line"; else printf '\033[2K%s\n' "$line"; fi
  done
}

select_many() {
  local n=${#SM_LABELS[@]} i cur=0 key rest ans tok keep bad tries=0 picked
  sm_collect
  [ "$n" -eq 0 ] && { SM_RESULT=""; return 0; }
  if [ "$NON_INTERACTIVE" = "1" ]; then
    [ "$SM_NONE_SILENT" = "1" ] && SM_RESULT=""
    return 0
  fi
  echo; plain "  $SM_TITLE" "$C_WHITE"
  if [ "${SM_ASSUME_TTY:-0}" != "1" ] && { [ ! -t 0 ] || [ ! -t 1 ] || ! command -v tput >/dev/null 2>&1 || [ "${TERM:-dumb}" = "dumb" ]; }; then
    for ((i = 0; i < n; i++)); do
      mark=" "; [ "${SM_SEL[$i]}" = "1" ] && mark="*"
      printf '   %s%d) %s\n' "$mark" "$((i + 1))" "${SM_LABELS[$i]}"
      [ -n "${SM_HINTS[$i]:-}" ] && printf '         %s\n' "${SM_HINTS[$i]}"
    done
    keep=""; for i in $SM_RESULT; do keep="$keep,$((i + 1))"; done; keep="${keep#,}"
    while :; do
      tries=$((tries + 1))
      printf '%s' "    Numbers separated by commas, 'none', or Enter = [$keep]: "
      ans=""; IFS= read -r ans || ans=""
      ans="$(printf '%s' "$ans" | sed 's/^ *//;s/ *$//')"
      [ -z "$ans" ] && return 0
      case "$ans" in none|n|no) SM_RESULT=""; return 0 ;; esac
      picked=""; bad=0
      for tok in $(printf '%s' "$ans" | tr ',' ' '); do
        case "$tok" in
          ''|*[!0-9]*) bad=1 ;;
          *) if [ "$tok" -ge 1 ] && [ "$tok" -le "$n" ]; then picked="$picked $((tok - 1))"; else bad=1; fi ;;
        esac
      done
      if [ "$bad" = "0" ]; then
        SM_RESULT="$(printf '%s\n' $picked | sort -nu | tr '\n' ' ' | sed 's/ *$//')"
        return 0
      fi
      if [ "$tries" -ge 5 ]; then info "No valid answer - keeping [$keep]."; return 0; fi
      fail "Type numbers from the list, e.g. 1,3"
    done
  fi
  plain "    Up/Down move, Space ticks, A ticks all, Enter confirms" "$C_GRAY"
  trap 'printf "\033[?25h\n"; exit 130' INT
  printf '\033[?25l'
  sm_draw "$cur"
  while :; do
    key=""
    IFS= read -rsn1 key || { key=""; }          # EOF confirms rather than spinning
    case "$key" in
      $'\033')
        rest=""; IFS= read -rsn2 -t 1 rest || true
        case "$rest" in
          '[A'|'OA') cur=$(((cur + n - 1) % n)) ;;
          '[B'|'OB') cur=$(((cur + 1) % n)) ;;
        esac ;;
      k) cur=$(((cur + n - 1) % n)) ;;
      j) cur=$(((cur + 1) % n)) ;;
      ' ') if [ "${SM_SEL[$cur]}" = "1" ]; then SM_SEL[$cur]=0; else SM_SEL[$cur]=1; fi ;;
      a|A)
        ans=0; for ((i = 0; i < n; i++)); do [ "${SM_SEL[$i]}" = "0" ] && ans=1; done   # any unticked: tick all, else clear
        for ((i = 0; i < n; i++)); do SM_SEL[$i]=$ans; done ;;
      '') break ;;                              # Enter (read strips the newline)
    esac
    sm_draw "$cur" redraw
  done
  printf '\033[?25h'; trap - INT
  sm_collect
}

# -- LLM providers ------------------------------------------------------------
# The providers agentY can run its agents on: the provider token of a
# 'provider,model' setting (src/agent.py: ollama, anthropic, the DashScope
# aliases, openai, google/gemini) and the key each one reads. Ollama is local.
# id|label|key variables (comma)|question|help
LLM_PROVIDERS='anthropic|Anthropic (Claude) - recommended|ANTHROPIC_API_KEY|Anthropic API key (Claude)|Create at https://platform.claude.com/
dashscope|Alibaba Model Studio (DashScope, Qwen)|DASHSCOPE_API_KEY|DashScope / Alibaba Model Studio key|For Qwen models: https://bailian.console.alibabacloud.com/
openai|OpenAI (GPT)|OPENAI_API_KEY|OpenAI API key|Create at https://platform.openai.com/api-keys
google|Google (Gemini)|GEMINI_API_KEY,GOOGLE_API_KEY|Gemini API key|Create at https://aistudio.google.com/apikey
ollama|Ollama (local models, no key)||Ollama|'

# Which providers to ask for keys. Ticked from the start: the ones whose key is
# already in .env (Anthropic when there is none yet). Asks the key of each ticked
# one only; the rest of .env is left alone.
select_llm_providers() {   # $1 = .env
  local env_file="$1" id label keys q help k isset any=0 i
  local -a P_KEYS=() P_Q=() P_HELP=()
  SM_LABELS=(); SM_HINTS=(); SM_SEL=()
  while IFS='|' read -r id label keys q help; do
    [ -z "$id" ] && continue
    isset=0
    for k in ${keys//,/ }; do is_placeholder "$(get_env_value "$env_file" "$k")" || isset=1; done
    [ "$isset" = "1" ] && any=1
    SM_LABELS+=("$label"); SM_SEL+=("$isset")
    SM_HINTS+=("$([ "$isset" = "1" ] && echo "key already in .env")")
    P_KEYS+=("$keys"); P_Q+=("$q"); P_HELP+=("$help")
  done <<EOF
$LLM_PROVIDERS
EOF
  [ "$any" = "0" ] && SM_SEL[0]=1
  SM_TITLE="Which LLM providers will you use? (a key is asked for each ticked one)"
  SM_NONE_SILENT=0
  select_many
  for i in $SM_RESULT; do
    if [ -z "${P_KEYS[$i]}" ]; then
      [ "$NON_INTERACTIVE" = "1" ] || info "Ollama needs no key. Install it from https://ollama.com/ and start it before agentY."
      continue
    fi
    k="${P_KEYS[$i]%%,*}"
    # Gemini also reads GOOGLE_API_KEY: an existing one counts as set.
    for id in ${P_KEYS[$i]//,/ }; do is_placeholder "$(get_env_value "$env_file" "$id")" || { k="$id"; break; }; done
    read_secret "$env_file" "$k" "${P_Q[$i]}" "${P_HELP[$i]}" >/dev/null
  done
}

# -- MCP presets --------------------------------------------------------------
# Offers the catalog in config/mcp_presets.json; scripts/mcp_presets.py does the
# merge into config/mcp.json so the logic exists once. Nothing here asks for an
# add-on to be installed into a desktop app: those steps are printed.
install_mcp_presets() {   # $1 = .env
  local env_file="$1" py tool id name cat def desc key label help opt line state warm steps
  py="$(venv_python "$PROJECT_ROOT")"; tool="$PROJECT_ROOT/scripts/mcp_presets.py"
  if [ ! -x "$py" ] || [ ! -f "$tool" ]; then info "MCP presets skipped (no venv python or scripts/mcp_presets.py)"; return 0; fi
  if [ "$NON_INTERACTIVE" = "1" ]; then info "Non-interactive: no MCP presets added. Add them later in Settings > MCP servers."; return 0; fi
  local -a P_IDS=() chosen=() later_id=() later_steps=()
  SM_LABELS=(); SM_HINTS=(); SM_SEL=()
  while IFS='|' read -r id name cat def desc; do
    [ -z "$id" ] && continue
    P_IDS+=("$id"); SM_LABELS+=("$name  [$cat]"); SM_HINTS+=("$desc"); SM_SEL+=("$def")
  done <<EOF
$("$py" "$tool" list)
EOF
  [ "${#P_IDS[@]}" -eq 0 ] && return 0
  SM_TITLE="MCP servers to add (extra tools for the agent; Enter with none ticked adds nothing)"
  SM_NONE_SILENT=1
  select_many
  SM_NONE_SILENT=0
  if [ -z "$SM_RESULT" ]; then info "No MCP presets chosen - add them later in Settings > MCP servers."; return 0; fi
  for line in $SM_RESULT; do chosen+=("${P_IDS[$line]}"); done
  for id in "${chosen[@]}"; do
    while IFS='|' read -r -u 3 key label help opt; do
      [ -z "$key" ] && continue
      label="$label (for MCP preset $id)"; [ "$opt" = "1" ] && label="$label - optional"
      read_secret "$env_file" "$key" "$label" "$help" >/dev/null
    done 3<<EOF
$("$py" "$tool" keys "$id")
EOF
  done
  while IFS='|' read -r -u 3 state id; do
    case "$state" in
      added)   success "MCP server '$id' added to config/mcp.json" ;;
      skipped) info "MCP server '$id' is already configured - kept as it is" ;;
      needs_bundle) info "MCP server '$id' needs its .mcpb bundle first - not written to config/mcp.json yet"
                    steps="$("$py" "$tool" steps "$id")"
                    [ -n "$steps" ] && { later_id+=("$id"); later_steps+=("$steps"); }
                    continue ;;
      *)       fail "Unknown MCP preset '$id'"; continue ;;
    esac
    if warm="$("$py" "$tool" warmup "$id" 2>&1 | tail -n 1)"; then
      [ -n "$warm" ] && [ "$warm" != "nothing to fetch" ] && success "Pre-fetched '$id'"
    else
      info "Could not pre-fetch '$id' ($warm) - it is fetched on first use"
    fi
    steps="$("$py" "$tool" steps "$id")"
    if [ -n "$steps" ]; then later_id+=("$id"); later_steps+=("$steps"); fi
  done 3<<EOF
$("$py" "$tool" apply "${chosen[@]}")
EOF
  if [ "${#later_id[@]}" -gt 0 ]; then
    echo; plain "  Finish later (the installer does not touch other applications):" "$C_WHITE"
    local i
    for ((i = 0; i < ${#later_id[@]}; i++)); do
      plain "    ${later_id[$i]}:" "$C_CYAN"
      printf '%s\n' "${later_steps[$i]}" | sed 's/^/      /'
    done
    info "Or ask the agent: it can do these steps for you once it runs (Settings > MCP servers > Add from presets does it too)."
  fi
}

# =============================================================================
echo
plain "  agentY stack installer" "$C_CYAN"
plain "  repo root: $PROJECT_ROOT" "$C_GRAY"
plain "  siblings : $PARENT_DIR" "$C_GRAY"
plain "  platform : $(uname -s) $(uname -m)" "$C_GRAY"

embedder_preset() {   # $1 = settings.local.json -> echoes memory.embedder.preset or nothing
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys
try:
    print(((json.load(open(sys.argv[1], encoding="utf-8")).get("memory") or {}).get("embedder") or {}).get("preset") or "")
except Exception:
    print("")
PY
}

# Exit 0 when settings.local.json can be written to: absent, empty, or a JSON
# object. A file that is there but unreadable must be left alone, not replaced by
# a fresh one - it is the user's settings.
settings_writable() {   # $1 = settings.local.json
  python3 - "$1" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    sys.exit(0)
try:
    raw = open(path, encoding="utf-8-sig").read()
    sys.exit(0 if not raw.strip() or isinstance(json.loads(raw), dict) else 1)
except Exception:
    sys.exit(1)
PY
}

set_embedder_preset() {   # $1 = settings.local.json  $2 = preset
  python3 - "$1" "$2" <<'PY'
import json, os, sys
path, preset = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path, encoding="utf-8-sig"))
    if not isinstance(data, dict):
        data = {}
except Exception:
    data = {}
mem = data["memory"] = data.get("memory") if isinstance(data.get("memory"), dict) else {}
emb = mem["embedder"] = mem.get("embedder") if isinstance(mem.get("embedder"), dict) else {}
emb["preset"] = preset
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
PY
}

ollama_running() {
  python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',11434))==0 else 1)" 2>/dev/null
}

# Long-term memory needs an embedder: a model that turns text into vectors so
# memories can be found by meaning. Asked here because the old default (Ollama)
# is not on every machine, and without one memory silently does nothing.
select_embedder() {   # $1 = .env  $2 = settings.local.json
  local env_file="$1" local_settings="$2" ollama_note current default
  if ollama_running; then ollama_note=" (running)"; else ollama_note=" (not running now - start it before agentY)"; fi
  local ids="local ollama dashscope gemini openai"
  local keys=" - - DASHSCOPE_API_KEY GEMINI_API_KEY OPENAI_API_KEY"
  if ! settings_writable "$local_settings"; then
    fail "$local_settings is not valid JSON - left untouched, so no embedder was chosen."
    info "Fix or delete that file, then pick one in Settings > Models > Memory embedder."
    return
  fi
  current="$(embedder_preset "$local_settings")"
  if [ -n "$current" ]; then default="$current"; elif ollama_running; then default="ollama"; else default="local"; fi
  if [ "$NON_INTERACTIVE" = "1" ]; then
    if [ -z "$current" ]; then set_embedder_preset "$local_settings" "$default"; success "Memory embedder: $default"
    else success "Memory embedder: $current (kept)"; fi
    return
  fi
  plain "  Long-term memory turns what the agent learns into vectors so it can find it" "$C_GRAY"
  plain "  again by meaning. Pick the model that does it (changeable later in Settings >" "$C_GRAY"
  plain "  Models > Memory embedder; switching re-embeds what is stored)." "$C_GRAY"
  local i=1 id key note mark def_idx=1
  for id in $ids; do
    key="$(printf '%s' "$keys" | awk -v n="$i" '{print $n}')"
    case "$id" in
      local) label="Local - built into agentY, no service or key (downloads ~130 MB on first use)" ;;
      ollama) label="Ollama - nomic-embed-text$ollama_note" ;;
      dashscope) label="Alibaba DashScope - text-embedding-v4" ;;
      gemini) label="Google Gemini - gemini-embedding-001" ;;
      openai) label="OpenAI - text-embedding-3-small" ;;
    esac
    note=""
    if [ "$key" != "-" ] && is_placeholder "$(get_env_value "$env_file" "$key")"; then note="  [needs $key - asked next]"; fi
    mark=" "; [ "$id" = "$default" ] && { mark="*"; def_idx=$i; }
    plain "   $mark$i) $label$note" "$C_WHITE"
    i=$((i + 1))
  done
  local ans pick pick_key val
  while true; do
    printf '    Memory embedder [1-5, Enter = %s]: ' "$def_idx"
    IFS= read -r ans
    ans="$(printf '%s' "${ans:-}" | sed 's/^ *//;s/ *$//')"
    [ -n "$ans" ] || ans="$def_idx"
    case "$ans" in 1|2|3|4|5) ;; *) fail "Type a number from the list."; continue ;; esac
    pick="$(printf '%s' "$ids" | awk -v n="$ans" '{print $n}')"
    pick_key="$(printf '%s' "$keys" | awk -v n="$ans" '{print $n}')"
    if [ "$pick_key" != "-" ] && is_placeholder "$(get_env_value "$env_file" "$pick_key")"; then
      val="$(read_secret "$env_file" "$pick_key" "$pick_key for the memory embedder" "")"
      if [ -z "$val" ]; then fail "That embedder needs $pick_key. Pick another, or enter the key."; continue; fi
    fi
    set_embedder_preset "$local_settings" "$pick"
    success "Memory embedder: $pick (settings.local.json)"
    [ "$pick" = "local" ] && info "Its model downloads into models/embeddings/ the first time memory is used."
    return
  done
}

# -- 1. Preflight -------------------------------------------------------------
header "1 / 8  Preflight"
command -v git >/dev/null 2>&1 || die "'git' is not on PATH. Install Git and re-run."
success "git found"
if ! command -v uv >/dev/null 2>&1; then
  fail "'uv' is not on PATH."
  plain "       Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh" "$C_WHITE"
  plain "       or:          brew install uv" "$C_WHITE"
  die "Install uv and re-run."
fi
success "uv found: $(uv --version)"
command -v python3 >/dev/null 2>&1 || die "'python3' is not on PATH. On macOS: xcode-select --install"
success "python3 found: $(python3 --version 2>&1)"

if [ "$IS_MAC" = "1" ]; then
  # insightface and sam3 ship as SOURCE distributions only — there is no macOS
  # wheel for either — so pip compiles them here. Without the Command Line Tools
  # that fails deep inside a build log, with a message about a missing header
  # rather than a missing toolchain.
  if xcode-select -p >/dev/null 2>&1; then
    success "Xcode Command Line Tools present (insightface / sam3 compile from source)"
  else
    fail "Xcode Command Line Tools are missing."
    plain "       insightface and sam3 have no macOS wheel and must compile here." "$C_WHITE"
    plain "       Install them first:  xcode-select --install" "$C_WHITE"
    die "Install the Command Line Tools and re-run."
  fi
  if [ "$(uname -m)" != "arm64" ]; then
    info "Intel Mac: onnxruntime and faiss-cpu publish arm64-only wheels at current"
    info "versions, so face-likeness QA may need older pins. Everything else is fine."
  fi
  # The collector nodes' file dialog runs under ComfyUI's Python, not this venv, so
  # this is a heads-up rather than something to install.
  if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    info "This python3 has no tkinter. If ComfyUI's Python also lacks it, the"
    info "collector nodes fall back to an AppleScript dialog (no install needed)."
  fi
fi

# -- 2. Sibling repos (agentY-core, agentY-mcp) -------------------------------
header "2 / 8  Sibling repos"
CHANNEL="$(update_channel)"
info "Release channel: $CHANNEL"
# agentY itself first: the siblings below are brought to the remote's newest
# commit, and a new agentY-core under an old agentY is an agent that does not
# start. When the pull replaced this very script, the new one takes over - the
# rest of an old installer would install what the old agentY needed.
if [ -d "$PROJECT_ROOT/.git" ]; then
  SELF="$PROJECT_ROOT/install_agent.sh"
  SELF_BEFORE="$(cksum < "$SELF")"
  update_checkout "agentY" "$PROJECT_ROOT"
  if [ "$SELF_BEFORE" != "$(cksum < "$SELF")" ] && [ -z "${AGENTY_INSTALLER_RESTARTED:-}" ]; then
    info "The installer itself was updated - starting the new one."
    AGENTY_INSTALLER_RESTARTED=1 exec bash "$SELF" ${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"}
  fi
else
  info "agentY at $PROJECT_ROOT is not a git checkout - it cannot be updated from here"
fi
# The tool layer's repository is agentY-core. An install from before it was
# renamed has the folder as agenty_core: it is updated where it is here, and moved
# to the new name in stage 3 (scripts/core_dir.py), once there is a Python to run
# that with.
CORE_DIR="$PARENT_DIR/agentY-core"
if [ ! -e "$CORE_DIR" ] && [ -d "$PARENT_DIR/agenty_core/.git" ]; then CORE_DIR="$PARENT_DIR/agenty_core"; fi
ensure_repo "agentY-core" "https://github.com/szprivate/agentY-core.git" "$CORE_DIR" required
[ -f "$CORE_DIR/pyproject.toml" ] || die "agentY-core looks incomplete at $CORE_DIR (no pyproject.toml). agentY's requirements.txt installs it editable via '-e ../agentY-core'."

MCP_DIR="$PARENT_DIR/agentY-mcp"
if [ "$SKIP_MCP" = "0" ]; then
  ensure_repo "agentY-mcp" "https://github.com/szprivate/agentY-mcp.git" "$MCP_DIR"
else
  info "Skipping agentY-mcp (--skip-mcp)"
fi

# -- 3. agentY environment ----------------------------------------------------
header "3 / 8  agentY environment"
setup_venv "agentY" "$PROJECT_ROOT" with-torch

# -- 4. Secrets (.env) --------------------------------------------------------
header "4 / 8  Secrets (.env)"
ENV_FILE="$(ensure_env_file "$PROJECT_ROOT")" || die "Could not prepare agentY's .env."
if [ "$NON_INTERACTIVE" = "1" ]; then
  info "Non-interactive: leaving .env values as-is. Edit $ENV_FILE to set keys."
else
  plain "  Press Enter to keep an existing value or skip an optional one." "$C_GRAY"
fi
select_llm_providers "$ENV_FILE"
HF_TOKEN_VAL="$(read_secret "$ENV_FILE" "HF_TOKEN"          "Hugging Face token (gated model downloads)"      "Create at https://huggingface.co/  (account -> Access Tokens)")"
COMFY_KEY_VAL="$(read_secret "$ENV_FILE" "COMFYUI_API_KEY"  "ComfyUI API key (optional - auth / API nodes)"   "https://platform.comfy.org/profile/api-keys  - blank for a local ComfyUI")"
install_mcp_presets "$ENV_FILE"
success "agentY .env ready ($ENV_FILE)"

# -- 5. Long-term memory embedder ----------------------------------------------
header "5 / 8  Memory embedder"
select_embedder "$ENV_FILE" "$PROJECT_ROOT/config/settings.local.json"

# -- 6. ComfyUI: locate + install the sidebar node ----------------------------
header "6 / 8  ComfyUI sidebar node"
RESOLVED_COMFY=""
if [ "$SKIP_COMFY_NODE" = "1" ]; then
  info "Skipping ComfyUI node install (--skip-comfy-node)"
else
  RESOLVED_COMFY="$(test_comfyui_dir "$COMFYUI_PATH")"
  [ -n "$RESOLVED_COMFY" ] || RESOLVED_COMFY="$(find_comfyui "$COMFYUI_PATH")"
  if [ -n "$RESOLVED_COMFY" ]; then
    success "Found ComfyUI: $RESOLVED_COMFY"
    if [ "$NON_INTERACTIVE" = "0" ]; then
      printf '    Use this ComfyUI? [Y/n] (or type another path): '
      IFS= read -r ans
      ans="$(printf '%s' "${ans:-}" | sed 's/^ *//;s/ *$//')"
      case "$(printf '%s' "$ans" | tr '[:upper:]' '[:lower:]')" in
        "" | y | yes) ;;
        n | no) RESOLVED_COMFY="" ;;
        *) RESOLVED_COMFY="$(test_comfyui_dir "$ans")" ;;
      esac
    fi
  fi
  if [ -z "$RESOLVED_COMFY" ] && [ "$NON_INTERACTIVE" = "0" ]; then
    printf '    ComfyUI folder (contains custom_nodes/), or Enter to skip: '
    IFS= read -r entered
    entered="$(printf '%s' "${entered:-}" | sed 's/^ *//;s/ *$//')"
    if [ -n "$entered" ]; then
      RESOLVED_COMFY="$(test_comfyui_dir "$entered")"
      [ -n "$RESOLVED_COMFY" ] || fail "That folder doesn't look like a ComfyUI install - skipping."
    fi
  fi

  if [ -n "$RESOLVED_COMFY" ]; then
    NODE_DIR="$RESOLVED_COMFY/custom_nodes/agentY-comfyuiConnect"
    ensure_repo "agentY-comfyuiConnect" "https://github.com/szprivate/agentY-comfyuiConnect.git" "$NODE_DIR"
    success "Sidebar node installed under $RESOLVED_COMFY/custom_nodes - restart ComfyUI once."

    # The bEpic viewer is agentY's standard viewer (Settings > Load & save nodes
    # saves into it by default). A failed clone only costs the viewer.
    ensure_repo "ComfyUI-ImageViewer" "https://github.com/szprivate/ComfyUI-ImageViewer.git" "$RESOLVED_COMFY/custom_nodes/ComfyUI-ImageViewer"

    # Record where the agentY host lives so the sidebar's "Start server" button can
    # relaunch run_agent.sh when the host is down. The agentY host also rewrites
    # this on startup; this bootstrap makes the button work day-1, before the host
    # has ever run. (Gitignored - machine-specific.)
    if [ -d "$NODE_DIR" ]; then
      python3 - "$NODE_DIR/.agenty_host.json" "$PROJECT_ROOT" <<'PY'
import json, sys
json.dump({"project_root": sys.argv[2], "run_script": "run_agent.sh"},
          open(sys.argv[1], "w", encoding="utf-8"), indent=2)
PY
      success "Recorded agentY host location for the 'Start server' button"
    fi

    # Offer to set this ComfyUI's URL as a local override (settings.local.json).
    # Committed defaults live in config/settings.default.toml (localhost); the local
    # JSON is deep-merged over them and is gitignored.
    LOCAL_SETTINGS="$PROJECT_ROOT/config/settings.local.json"
    if ! settings_writable "$LOCAL_SETTINGS"; then
      fail "$LOCAL_SETTINGS is not valid JSON - left untouched (the ComfyUI URL was not asked)."
    elif [ "$NON_INTERACTIVE" = "0" ]; then
      CUR_URL="$(python3 - "$LOCAL_SETTINGS" <<'PY' 2>/dev/null
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("comfyui_url") or "http://127.0.0.1:8188")
except Exception:
    print("http://127.0.0.1:8188")
PY
)"
      [ -n "$CUR_URL" ] || CUR_URL="http://127.0.0.1:8188"
      printf '    ComfyUI URL for settings.local.json [Enter = keep %s]: ' "$CUR_URL"
      IFS= read -r NEW_URL
      NEW_URL="$(printf '%s' "${NEW_URL:-}" | sed 's/^ *//;s/ *$//')"
      if [ -n "$NEW_URL" ] && [ "$NEW_URL" != "$CUR_URL" ]; then
        python3 - "$LOCAL_SETTINGS" "$NEW_URL" <<'PY'
import json, os, sys
path, url = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
except Exception:
    data = {}
data["comfyui_url"] = url
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
PY
        success "settings.local.json comfyui_url -> $NEW_URL"
      fi
    fi
  else
    info "No ComfyUI configured. Install the node later:"
    plain "       git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>/custom_nodes/agentY-comfyuiConnect" "$C_WHITE"
  fi
fi

# -- 7. agentY-mcp environment (optional) -------------------------------------
header "7 / 8  agentY-mcp environment"
if [ "$SKIP_MCP" = "1" ] || [ ! -f "$MCP_DIR/requirements.txt" ]; then
  info "Skipping agentY-mcp environment."
else
  setup_venv "agentY-mcp" "$MCP_DIR"
  MCP_ENV="$(ensure_env_file "$MCP_DIR")" || die "Could not prepare agentY-mcp's .env."
  # The MCP host (Claude Desktop) supplies the model, so agentY-mcp only needs
  # HF_TOKEN + COMFYUI_API_KEY. Reuse what we just collected for agentY.
  if ! is_placeholder "$HF_TOKEN_VAL"; then
    set_env_value "$MCP_ENV" "HF_TOKEN" "$HF_TOKEN_VAL"; info "Propagated HF_TOKEN to agentY-mcp .env"
  fi
  if ! is_placeholder "$COMFY_KEY_VAL"; then
    set_env_value "$MCP_ENV" "COMFYUI_API_KEY" "$COMFY_KEY_VAL"; info "Propagated COMFYUI_API_KEY to agentY-mcp .env"
  fi
  success "agentY-mcp ready ($MCP_DIR)"
fi

# -- 8. Verify ----------------------------------------------------------------
header "8 / 8  Dependency check"
chmod +x "$PROJECT_ROOT/run_agent.sh" 2>/dev/null
test_environment "$PROJECT_ROOT"

# -- Done ---------------------------------------------------------------------
header "Setup complete"
echo
plain "  Installed:" "$C_CYAN"
plain "    - agentY        $PROJECT_ROOT"
plain "    - agentY-core   $PARENT_DIR/agentY-core  (editable dependency)"
[ "$SKIP_MCP" = "0" ] && [ -f "$MCP_DIR/requirements.txt" ] && plain "    - agentY-mcp    $MCP_DIR"
[ -n "$RESOLVED_COMFY" ] && plain "    - sidebar node  $RESOLVED_COMFY/custom_nodes/agentY-comfyuiConnect"
echo
plain "  Next steps:" "$C_CYAN"
plain "    1. Start the agent chat host:" "$C_YELLOW"
plain "         ./run_agent.sh"
if [ -n "$RESOLVED_COMFY" ]; then
  plain "    2. Restart ComfyUI, then click the 'agentY' tab in its left sidebar." "$C_YELLOW"
else
  plain "    2. Install agentY-comfyuiConnect into ComfyUI/custom_nodes and restart ComfyUI." "$C_YELLOW"
fi
if [ "$SKIP_MCP" = "0" ] && [ -f "$MCP_DIR/requirements.txt" ]; then
  plain "    3. (optional) Register agentY-mcp with Claude Desktop - see $MCP_DIR/README.md" "$C_YELLOW"
fi
echo
plain "  Review secrets/paths anytime in:  $ENV_FILE" "$C_GRAY"
plain "  Defaults: $PROJECT_ROOT/config/settings.default.toml  (overrides: settings.local.json)" "$C_GRAY"
echo
