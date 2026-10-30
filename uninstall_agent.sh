#!/usr/bin/env bash
# uninstall_agent.sh - Uninstaller for the agentY stack (macOS and Linux).
#
# The counterpart of install_agent.sh. An install only creates folders - no launch agents,
# environment variables, services or shortcuts - so removing agentY means removing folders, and
# this script only ever removes folders it has IDENTIFIED as agentY's (git origin
# github.com/szprivate/<repo>, or agentY marker files):
#
#   * agentY, agentY-core (older: agenty_core), agentY-mcp   - the code and each venv
#   * your data inside them: conversations, memory, settings, .env keys, MCP tokens, output
#   * agentY-comfyuiConnect (older: comfyui-agent-canvas, agentY_bridge) in custom_nodes
#   * the bEpic viewer (ComfyUI-ImageViewer, any folder name) - kept unless you say so
#
# It shows a plan first (every path with its size, and what it KEEPS), then asks per group. It
# never removes ComfyUI, models, Ollama, Strata, Blender, or anything it cannot verify.
# Works on old installs and standalone: download it anywhere and point it at the install with
# --path, or let it look (its own folder, the current folder, the usual places).
#
# Usage:
#   ./uninstall_agent.sh --dry-run
#   ./uninstall_agent.sh --path ~/agentY
#
# Written for bash 3.2 - the version macOS still ships as /bin/bash.

set -uo pipefail

AGENTY_PATH=""
PARENT_DIR=""
COMFYUI_PATH=""
YES=0
DRY_RUN=0
KEEP_USER_DATA=0
NO_BACKUP=0
INCLUDE_VIEWER=0
INCLUDE_DESKTOP=0
PORT=5000
FROM_TEMP=""

C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
C_RED=$'\033[31m'; C_GRAY=$'\033[90m'; C_WHITE=$'\033[97m'; C_OFF=$'\033[0m'
if [ ! -t 1 ]; then C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_WHITE=""; C_OFF=""; fi

header()  { printf '\n%s===  %s  ===%s\n' "$C_CYAN" "$1" "$C_OFF"; }
success() { printf '%s  [ok] %s%s\n' "$C_GREEN" "$1" "$C_OFF"; }
info()    { printf '%s  [i]  %s%s\n' "$C_YELLOW" "$1" "$C_OFF"; }
fail()    { printf '%s  [!]  %s%s\n' "$C_RED" "$1" "$C_OFF"; }
line()    { printf '%s       %s%s\n' "${2:-$C_WHITE}" "$1" "$C_OFF"; }
die()     { echo; fail "$1"; exit "${2:-1}"; }

usage() {
  cat <<'EOF'

Usage: ./uninstall_agent.sh [OPTIONS]

  --path <dir>            The agentY folder to remove. Without it the script looks at its
                          own folder, the current folder (and its parents), ComfyUI's
                          .agenty_host.json and the usual places.
  --parent-dir <dir>      Where agentY-core / agentY-mcp live (default: parent of agentY).
  --comfyui-path <dir>    Only look for the panel and viewer in this ComfyUI.
  --yes                   Do not ask. Removes agentY and its siblings, your data (after a
                          backup zip) and the panel; keeps the viewer and ComfyUI Desktop
                          unless --include-viewer / --include-comfyui-desktop.
  --dry-run               Only print the plan and the checks. Nothing is asked or removed.
  --keep-user-data        With --yes: keep conversations, memory, settings and keys.
  --no-backup             Do not make the backup zip of your data.
  --include-viewer        With --yes: also remove the bEpic viewer.
  --include-comfyui-desktop
                          With --yes: also run the brew uninstall of ComfyUI Desktop.
  --port <n>              The port the agentY host listens on (default 5000).
  -h, --help              Show this help.

EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --path)            AGENTY_PATH="${2:-}"; shift 2 ;;
    --parent-dir)      PARENT_DIR="${2:-}"; shift 2 ;;
    --comfyui-path)    COMFYUI_PATH="${2:-}"; shift 2 ;;
    --yes|-y)          YES=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --keep-user-data)  KEEP_USER_DATA=1; shift ;;
    --no-backup)       NO_BACKUP=1; shift ;;
    --include-viewer)  INCLUDE_VIEWER=1; shift ;;
    --include-comfyui-desktop) INCLUDE_DESKTOP=1; shift ;;
    --port)            PORT="${2:-5000}"; shift 2 ;;
    --from-temp)       FROM_TEMP="${2:-}"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

TAB=$'\t'

# -- Small helpers ----------------------------------------------------------------------------
abspath() {   # $1 = path (may not exist) -> physical absolute path, no trailing slash
  local p="$1" d b
  case "$p" in "~"/*) p="$HOME/${p#\~/}" ;; "~") p="$HOME" ;; esac
  if [ -d "$p" ]; then ( cd "$p" 2>/dev/null && pwd -P ); return 0; fi
  d="$(dirname "$p")"; b="$(basename "$p")"
  if [ -d "$d" ]; then printf '%s/%s\n' "$(cd "$d" && pwd -P)" "$b"; else printf '%s\n' "$p"; fi
}

under() {   # $1 child, $2 parent -> 0 when child is parent or inside it
  case "$1/" in "$2"/*) return 0 ;; esac
  return 1
}

size_kb() {   # $1 = path -> KB used (links are not followed)
  if [ -L "$1" ]; then echo 0; return; fi
  du -sk "$1" 2>/dev/null | awk '{print $1}' | head -n 1 | grep . || echo 0
}

fmt_kb() {
  local kb="${1:-0}"
  awk -v k="$kb" 'BEGIN{ if(k>=1048576) printf "%.1f GB", k/1048576; else if(k>=1024) printf "%.1f MB", k/1024; else printf "%d KB", k }'
}

ask() {   # $1 question, $2 default (y|n) -> 0 for yes. EOF = the default.
  local q="$1" def="$2" hint a
  if [ "$def" = "y" ]; then hint="Y/n"; else hint="y/N"; fi
  printf '    %s [%s] ' "$q" "$hint"
  a=""; read -r a || a=""
  a="${a%$'\r'}"
  a="$(printf '%s' "$a" | tr '[:upper:]' '[:lower:]' | tr -d ' \r')"
  [ -n "$a" ] || a="$def"
  case "$a" in y|yes) return 0 ;; *) return 1 ;; esac
}

# -- Identification: git origin first, marker files as the fallback ---------------------------
ORIGIN=""
git_origin() {   # $1 = dir -> sets ORIGIN; returns 1 when it is not a git checkout
  local d="$1" cfg in line
  ORIGIN=""
  [ -e "$d/.git" ] || return 1
  if [ -d "$d/.git" ] && [ -f "$d/.git/config" ]; then
    in=0
    while IFS= read -r line; do
      case "$line" in
        *'[remote "origin"]'*) in=1 ;;
        '['*) in=0 ;;
        *url*=*) if [ "$in" = 1 ]; then ORIGIN="$(printf '%s' "$line" | sed 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*$//')"; break; fi ;;
      esac
    done < "$d/.git/config"
    return 0
  fi
  ORIGIN="$(git -C "$d" remote get-url origin 2>/dev/null | head -n 1 || true)"
  return 0
}

origin_pattern() {   # $1 = kind
  case "$1" in
    agentY)    echo 'github\.com[:/]szprivate/agenty(\.git)?/?$' ;;
    core)      echo 'github\.com[:/]szprivate/(agenty-core|agenty_core)(\.git)?/?$' ;;
    mcp)       echo 'github\.com[:/]szprivate/agenty-mcp(\.git)?/?$' ;;
    connect)   echo 'github\.com[:/]szprivate/(agenty-comfyuiconnect|comfyui-agent-canvas)(\.git)?/?$' ;;
    viewer)    echo 'github\.com[:/]szprivate/comfyui-imageviewer(\.git)?/?$' ;;
    datalayer) echo 'github\.com[:/]chainlit/chainlit-datalayer(\.git)?/?$' ;;
  esac
}

origin_matches() {   # $1 = kind, $2 = origin
  printf '%s' "$2" | grep -Eiq "$(origin_pattern "$1")"
}

markers_ok() {   # $1 = kind, $2 = dir -> 0 when a marker row is complete
  local k="$1" d="$2"
  case "$k" in
    agentY)  [ -f "$d/requirements.txt" ] && { [ -f "$d/src/pipeline.py" ] || [ -f "$d/src/agenty_ui_server.py" ] || [ -f "$d/src/chainlit_app.py" ]; } ;;
    core)    [ -f "$d/pyproject.toml" ] && [ -f "$d/agenty_core/__init__.py" ] ;;
    mcp)     { [ -f "$d/manifest.json" ] && [ -f "$d/claude_desktop_config.example.json" ]; } || { [ -f "$d/requirements.txt" ] && [ -d "$d/agents" ] && [ -d "$d/src" ]; } ;;
    connect) [ -f "$d/__init__.py" ] && { [ -f "$d/web/agent_canvas.js" ] || [ -f "$d/web/agent_chat.js" ] || [ -f "$d/js/agentY_bridge.js" ]; } ;;
    viewer)  [ -f "$d/pyproject.toml" ] && grep -q 'szprivate/ComfyUI-ImageViewer' "$d/pyproject.toml" 2>/dev/null ;;
    *)       return 1 ;;
  esac
}

ID_HOW=""; ID_WHY=""
identify() {   # $1 = dir, $2 = kind -> 0 verified (ID_HOW), 1 refused (ID_WHY)
  ID_HOW=""; ID_WHY=""
  if git_origin "$1" && [ -n "$ORIGIN" ]; then
    if origin_matches "$2" "$ORIGIN"; then ID_HOW="git origin $ORIGIN"; return 0; fi
    ID_WHY="its git origin is '$ORIGIN', not an agentY repository"; return 1
  fi
  if markers_ok "$2" "$1"; then ID_HOW="marker files"; return 0; fi
  ID_WHY="no git origin from szprivate and none of the agentY marker files"; return 1
}

# == ComfyUI discovery (copied from install_agent.sh - the installer is not edited) =========
test_comfyui_dir() {
  local p="${1:-}" cand
  [ -n "$p" ] || return 0
  case "$p" in "~"/*) p="$HOME/${p#\~/}" ;; "~") p="$HOME" ;; esac
  [ -d "$p" ] || return 0
  for cand in "$p" "$p/ComfyUI" "$p/comfyui"; do
    if [ -d "$cand/custom_nodes" ]; then ( cd "$cand" && pwd -P ); return 0; fi
  done
}

COMFY_FOUND=""
add_comfy() {
  local hit
  hit="$(test_comfyui_dir "${1:-}")"
  [ -n "$hit" ] || return 0
  case "
$COMFY_FOUND
" in *"
$hit
"*) return 0 ;; esac
  COMFY_FOUND="${COMFY_FOUND:+$COMFY_FOUND
}$hit"
}

comfy_desktop_base_paths() {
  local cfg
  for cfg in "${HOME:-}/Library/Application Support/ComfyUI/config.json" \
             "${XDG_CONFIG_HOME:-${HOME:-}/.config}/ComfyUI/config.json"; do
    [ -f "$cfg" ] || continue
    sed -n 's/.*"basePath"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg"
  done
}

comfy_from_processes() {
  ps -axo command= 2>/dev/null | grep -i 'main\.py' | grep -i comfy \
    | sed -n 's/.*[[:space:]]\(\/[^[:space:]]*\)\/main\.py.*/\1/p'
}

find_all_comfyui() {   # $1 = hint -> fills COMFY_FOUND
  local c line
  COMFY_FOUND=""
  if [ -n "${1:-}" ]; then add_comfy "$1"; return 0; fi    # an explicit folder is the only one looked at
  for c in "$PARENT_DIR/ComfyUI" "$PARENT_DIR/comfyui" \
           "${HOME:-}/ComfyUI" "${HOME:-}/comfyui" \
           "${HOME:-}/Documents/ComfyUI" "${HOME:-}/Documents/comfyui" \
           "${HOME:-}/Library/Application Support/ComfyUI" \
           "${AGENTY_COMFYUI_DIR:-}" \
           "/Applications/ComfyUI"; do
    add_comfy "$c"
  done
  for c in "${HOME:-}/ComfyUI-Installs"/*; do add_comfy "$c"; done
  while IFS= read -r line; do [ -z "$line" ] || add_comfy "$line"; done <<EOF_BASE
$(comfy_desktop_base_paths)
EOF_BASE
  while IFS= read -r line; do [ -z "$line" ] || add_comfy "$line"; done <<EOF_PROC
$(comfy_from_processes)
EOF_PROC
}
# == end ComfyUI discovery ====================================================================

# -- Where is agentY? --------------------------------------------------------------------------
FOUND_OK=""; FOUND_REFUSED=""
consider_agenty() {   # $1 = candidate dir
  local full
  [ -d "$1" ] || return 0
  full="$(abspath "$1")"
  case "
$SEEN_CAND
" in *"
$full
"*) return 0 ;; esac
  SEEN_CAND="${SEEN_CAND:+$SEEN_CAND
}$full"
  if identify "$full" agentY; then FOUND_OK="${FOUND_OK:+$FOUND_OK
}$full"
  else FOUND_REFUSED="${FOUND_REFUSED:+$FOUND_REFUSED
}$full - $ID_WHY"; fi
}

find_agenty() {
  local cur i d line hostcfg root rel
  SEEN_CAND=""; FOUND_OK=""; FOUND_REFUSED=""
  if [ -n "$AGENTY_PATH" ]; then
    consider_agenty "$AGENTY_PATH"; consider_agenty "$AGENTY_PATH/agentY"; return 0
  fi
  # Tier 1: where the script is and where the user stands.
  [ -z "${SELF_DIR:-}" ] || consider_agenty "$SELF_DIR"
  cur="$(pwd -P)"; i=0
  while [ "$i" -lt 6 ] && [ -n "$cur" ]; do
    consider_agenty "$cur"; consider_agenty "$cur/agentY"
    [ "$cur" = "/" ] && break
    cur="$(dirname "$cur")"; i=$((i + 1))
  done
  [ -z "$FOUND_OK" ] || return 0
  # Tier 2: the panel records where its host lives.
  while IFS= read -r root; do
    [ -n "$root" ] || continue
    for d in "$root/custom_nodes"/*/; do
      hostcfg="${d%/}/.agenty_host.json"
      [ -f "$hostcfg" ] || continue
      line="$(sed -n 's/.*"project_root"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$hostcfg" | head -n 1)"
      [ -z "$line" ] || consider_agenty "$line"
    done
  done <<EOF_ROOTS
$COMFY_FOUND
EOF_ROOTS
  [ -z "$FOUND_OK" ] || return 0
  # Tier 3: the usual places.
  for rel in "agentY" "Documents/agentY" "dev/agentY" "src/agentY" "Developer/agentY"; do consider_agenty "${HOME:-}/$rel"; done
}

# -- Safety: paths that are never removed ------------------------------------------------------
forbidden_reason() {   # $1 = path -> prints why, or nothing
  local p c
  p="$(abspath "$1")"
  [ "$p" = "/" ] && { echo "it is the filesystem root"; return; }
  case "$p" in /usr|/usr/*|/bin|/sbin|/etc|/var|/System|/System/*|/Library|/Applications|/Users|/home|/opt|/private|/private/*|/tmp|/dev)
    [ "$p" = "/tmp" ] || { echo "it is a system folder ($p)"; return; } ;;
  esac
  if [ -n "${HOME:-}" ] && under "$(abspath "$HOME")" "$p"; then echo "it is, or contains, your home folder"; return; fi
  while IFS= read -r c; do
    [ -n "$c" ] || continue
    if under "$c" "$p"; then echo "it is, or contains, the ComfyUI folder $c"; return; fi
  done <<EOF_C
$COMFY_FOUND
EOF_C
  if [ -n "$PARENT_DIR" ] && under "$PARENT_DIR" "$p"; then echo "it is, or contains, the folder the siblings live in ($PARENT_DIR)"; return; fi
}

# == Plan ====================================================================================
ITEMS=""        # group TAB path TAB label TAB link(0|1) TAB kb TAB how
REFUSED=""; REMOVED=""; KEPT=""; FAILED=""
add_line() { eval "$1=\"\${$1:+\$$1
}\$2\""; }

add_item() {   # group path label link how
  local kb=0
  if [ "$4" != "1" ]; then kb="$(size_kb "$2")"; fi
  add_line ITEMS "$1${TAB}$2${TAB}$3${TAB}$4${TAB}$kb${TAB}${5:--}"
}

add_verified() {   # path kind group label -> 0 when it became an item
  local full why target
  full="$(abspath "$1")"
  why="$(forbidden_reason "$full")"
  if [ -n "$why" ]; then add_line REFUSED "$full - refused: $why"; return 1; fi
  if [ -L "$full" ]; then
    target="$(cd "$full" 2>/dev/null && pwd -P)"
    [ -n "$target" ] || { add_line REFUSED "$full - refused: a link whose target cannot be read"; return 1; }
    if identify "$target" "$2"; then add_item "$3" "$1" "$4 (link to $target - only the link is removed)" 1 "$ID_HOW"; return 0; fi
    add_line REFUSED "$full - refused: $ID_WHY"; return 1
  fi
  if identify "$full" "$2"; then add_item "$3" "$full" "$4" 0 "$ID_HOW"; return 0; fi
  add_line REFUSED "$full - refused: $ID_WHY"; return 1
}

items_of() {   # $1 = group -> the item lines of it
  printf '%s\n' "$ITEMS" | awk -F"$TAB" -v g="$1" '$1==g'
}
count_of() { items_of "$1" | grep -c . || true; }

# -- start ---------------------------------------------------------------------------------------
SELF="$0"
SELF_DIR=""
case "$SELF" in */*) [ -f "$SELF" ] && SELF_DIR="$(cd "$(dirname "$SELF")" && pwd -P)"; SELF="$SELF_DIR/$(basename "$SELF")" ;; *) [ -f "$SELF" ] && SELF_DIR="$(pwd -P)" && SELF="$SELF_DIR/$SELF" ;; esac

echo
printf '%s  agentY uninstaller%s\n' "$C_CYAN" "$C_OFF"
[ "$DRY_RUN" = 0 ] || printf '%s  DRY RUN - nothing is asked, stopped or removed.%s\n' "$C_YELLOW" "$C_OFF"

[ -z "$PARENT_DIR" ] || PARENT_DIR="$(abspath "$PARENT_DIR")"
find_all_comfyui "$COMFYUI_PATH"
if [ -n "$COMFYUI_PATH" ] && [ -z "$COMFY_FOUND" ]; then fail "--comfyui-path $COMFYUI_PATH does not look like a ComfyUI install (no custom_nodes/ in it)."; fi

find_agenty
PROJECT=""
n_ok="$(printf '%s\n' "$FOUND_OK" | grep -c . || true)"
if [ "$n_ok" = "1" ]; then PROJECT="$FOUND_OK"
elif [ "$n_ok" -gt 1 ]; then
  info "More than one agentY folder was found:"
  printf '%s\n' "$FOUND_OK" | nl -w2 -s') ' | while IFS= read -r l; do line "$l"; done
  if [ "$YES" = 1 ] || [ "$DRY_RUN" = 1 ]; then die "Pass --path <folder> to say which one." 2; fi
  printf '    Which one? [1-%s] ' "$n_ok"; read -r pick || pick=""
  PROJECT="$(printf '%s\n' "$FOUND_OK" | sed -n "${pick:-0}p")"
  [ -n "$PROJECT" ] || die "No choice made - nothing removed." 2
fi
if [ -z "$PROJECT" ]; then
  if [ -n "$AGENTY_PATH" ] && [ -n "$FOUND_REFUSED" ]; then printf '%s\n' "$FOUND_REFUSED" | while IFS= read -r l; do fail "$l"; done; fi
  die "No agentY folder found. Run this from inside it, or pass --path <the agentY folder>." 2
fi
[ -n "$PARENT_DIR" ] || PARENT_DIR="$(dirname "$PROJECT")"

# From inside the folder it is about to delete: re-run from a copy in a temp folder.
if [ -n "$SELF" ] && [ -f "$SELF" ] && [ -z "$FROM_TEMP" ] && [ "$DRY_RUN" = 0 ] && under "$SELF" "$PROJECT"; then
  tmp="$(mktemp "${TMPDIR:-/tmp}/agenty_uninstall.XXXXXX")" || die "Cannot make a temp file."
  cp "$SELF" "$tmp"
  info "Running from a copy in the temp folder so this folder can be removed."
  set -- --from-temp "$SELF" --path "$PROJECT" --parent-dir "$PARENT_DIR" --port "$PORT"
  [ -z "$COMFYUI_PATH" ] || set -- "$@" --comfyui-path "$COMFYUI_PATH"
  [ "$YES" = 0 ] || set -- "$@" --yes
  [ "$KEEP_USER_DATA" = 0 ] || set -- "$@" --keep-user-data
  [ "$NO_BACKUP" = 0 ] || set -- "$@" --no-backup
  [ "$INCLUDE_VIEWER" = 0 ] || set -- "$@" --include-viewer
  [ "$INCLUDE_DESKTOP" = 0 ] || set -- "$@" --include-comfyui-desktop
  ( cd "${TMPDIR:-/tmp}" && bash "$tmp" "$@" )
  code=$?
  rm -f "$tmp"
  exit "$code"
fi
[ -z "$FROM_TEMP" ] || cd "${TMPDIR:-/tmp}" 2>/dev/null || cd /

printf '%s  agentY   : %s%s\n' "$C_GRAY" "$PROJECT" "$C_OFF"
printf '%s  siblings : %s%s\n' "$C_GRAY" "$PARENT_DIR" "$C_OFF"
if [ -n "$COMFY_FOUND" ]; then printf '%s\n' "$COMFY_FOUND" | while IFS= read -r c; do printf '%s  ComfyUI  : %s%s\n' "$C_GRAY" "$c" "$C_OFF"; done
else printf '%s  ComfyUI  : none found (the panel and viewer are not looked for)%s\n' "$C_GRAY" "$C_OFF"; fi

# Group 2: code and venvs ------------------------------------------------------------------------
add_verified "$PROJECT" agentY code "agentY" || true
if [ -d "$PARENT_DIR" ]; then
  for d in "$PARENT_DIR"/* "$PARENT_DIR"/.[!.]*; do
    [ -d "$d" ] || continue
    [ "$(abspath "$d")" = "$PROJECT" ] && continue
    git_origin "$d" || true
    o="$ORIGIN"; kind=""; base="$(basename "$d")"
    case "$base" in agentY-core|agenty_core) kind=core ;; agentY-mcp) kind=mcp ;; esac
    if [ -z "$kind" ] && [ -n "$o" ]; then
      if origin_matches core "$o"; then kind=core; elif origin_matches mcp "$o"; then kind=mcp; fi
    fi
    [ -n "$kind" ] || continue
    if [ ! -L "$d" ] && [ -n "$o" ] && ! origin_matches "$kind" "$o"; then
      add_line REFUSED "$d - refused: its git origin is '$o', not an agentY repository"; continue
    fi
    if [ "$kind" = core ]; then lbl="agentY-core (the tool layer)"; else lbl="agentY-mcp"; fi
    add_verified "$d" "$kind" code "$lbl" || true
  done
fi

# Group 3: user data inside the repos -----------------------------------------------------------
DATA_REL="memory output output_images output_workflows logs .logs .files .chainlit .env .env.* keys*.txt .agenty_token* config/settings.local.json* config/.mcp_tokens config/fitness_weights.json comfyui_workflow_templates_custom/templates/local"
DATA_PATHS=""
while IFS="$TAB" read -r g p l k kb how; do
  [ -n "$g" ] || continue
  [ "$k" = "1" ] && continue
  for rel in $DATA_REL; do
    for m in "$p"/$rel; do
      [ -e "$m" ] || continue
      [ -L "$m" ] && continue
      case "$(basename "$m")" in .env_example|.env.example) continue ;; esac
      add_line DATA_PATHS "$m"
    done
  done
  if [ -e "$p/chainlit-datalayer" ]; then
    if identify "$p/chainlit-datalayer" datalayer; then add_line DATA_PATHS "$p/chainlit-datalayer"
    else add_line REFUSED "$p/chainlit-datalayer - refused: $ID_WHY"; fi
  fi
done <<EOF_CODE
$(items_of code)
EOF_CODE
DATA_PATHS="$(printf '%s\n' "$DATA_PATHS" | grep . | sort -u)"
while IFS= read -r p; do
  [ -n "$p" ] || continue
  add_item data "$p" "data: ${p#$PARENT_DIR/}" 0 -
done <<EOF_DATA
$DATA_PATHS
EOF_DATA
DATA_KEEP="$(printf '%s\n' "$DATA_PATHS" | grep .)"

# Groups 4 and 5: the panel and the viewer in every ComfyUI found --------------------------------
while IFS= read -r root; do
  [ -n "$root" ] || continue
  for d in "$root/custom_nodes"/*/ "$root/custom_nodes"/.[!.]*/; do
    d="${d%/}"
    [ -d "$d" ] || continue
    base="$(basename "$d")"; git_origin "$d" || true; o="$ORIGIN"
    is_panel=0; is_viewer=0
    case "$base" in agentY-comfyuiConnect|comfyui-agent-canvas|agentY_bridge) is_panel=1 ;; esac
    if [ -n "$o" ]; then
      origin_matches connect "$o" && is_panel=1
      origin_matches viewer "$o" && is_viewer=1
    elif [ "$is_panel" = 0 ] && markers_ok viewer "$d"; then is_viewer=1; fi
    if [ "$is_viewer" = 1 ]; then add_verified "$d" viewer viewer "bEpic viewer ($base)" || true
    elif [ "$is_panel" = 1 ]; then add_verified "$d" connect panel "sidebar panel ($base)" || true; fi
  done
done <<EOF_COMFY
$COMFY_FOUND
EOF_COMFY

# Plan output ---------------------------------------------------------------------------------
show_group() {   # title group default
  local g p l k kb how sz dataKb sub
  echo
  printf '%s  %s  (default answer: %s)%s\n' "$C_WHITE" "$1" "$3" "$C_OFF"
  if [ "$(count_of "$2")" = "0" ]; then line "(nothing found)" "$C_GRAY"; return; fi
  while IFS="$TAB" read -r g p l k kb how; do
    [ -n "$g" ] || continue
    if [ "$k" = "1" ]; then sz="link"; else
      sz="$(fmt_kb "$kb")"
      if [ "$2" = code ]; then
        dataKb=0
        while IFS="$TAB" read -r g2 p2 l2 k2 kb2 h2; do
          [ -n "$g2" ] || continue
          if under "$p2" "$p"; then dataKb=$((dataKb + kb2)); fi
        done <<EOF_D
$(items_of data)
EOF_D
        [ "$dataKb" -le 0 ] || sz="$sz in all, of which $(fmt_kb "$dataKb") is your data (item 3)"
      fi
    fi
    line "$p  [$sz]" "$C_GRAY"
    [ "$how" = "-" ] || line "  verified: $how" "$C_GRAY"
  done <<EOF_G
$(items_of "$2")
EOF_G
}

own_pids() {   # this process and its ancestors, one per line
  local cur=$$ i=0
  while [ "$i" -lt 12 ] && [ -n "$cur" ] && [ "$cur" != "0" ]; do
    echo "$cur"
    cur="$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ')"; i=$((i + 1))
  done
}

host_pids() {   # pids of the agentY host: run_agent / agenty_ui_server under the code dirs, or the :PORT listener that is ours
  local own dirs ours pid cmd d listener
  own="$(own_pids)"
  dirs="$(items_of code | awk -F"$TAB" '$4==0{print $2}')"
  listener="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | sort -u || true)"
  ps -axo pid=,command= 2>/dev/null | while read -r pid cmd; do
    case "
$own
" in *"
$pid
"*) continue ;; esac
    ours=0
    while IFS= read -r d; do [ -n "$d" ] && case "$cmd" in *"$d"*) ours=1 ;; esac; done <<EOF_X
$dirs
EOF_X
    [ "$ours" = 1 ] || continue
    case "$cmd" in *agenty_ui_server*|*run_agent.*) echo "$pid $cmd" ;; esac
  done
  : "$listener"
}

foreign_listener() {
  local pid
  pid="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)"
  [ -n "$pid" ] || return 0
  printf '%s %s\n' "$pid" "$(ps -o command= -p "$pid" 2>/dev/null | cut -c1-80)"
}

header "Plan"
HOST_LINES="$(host_pids)"
echo
printf '%s  1. The running agentY host  (default answer: Y)%s\n' "$C_WHITE" "$C_OFF"
if [ -n "$HOST_LINES" ]; then printf '%s\n' "$HOST_LINES" | while IFS= read -r l; do line "PID $l" "$C_GRAY"; done; else line "(not running)" "$C_GRAY"; fi
fl="$(foreign_listener)"
if [ -n "$fl" ]; then
  fpid="${fl%% *}"
  if ! printf '%s\n' "$HOST_LINES" | grep -q "^$fpid "; then line "port $PORT is used by PID $fl - not agentY's, left alone" "$C_YELLOW"; fi
fi

show_group "2. agentY, agentY-core and agentY-mcp: the code and the venvs" code "Y"
show_group "3. Your data (conversations, memory, settings, .env keys, MCP tokens, output)" data "N - asked separately, with a backup zip first"
show_group "4. The agentY panel in ComfyUI's custom_nodes" panel "Y"
show_group "5. The bEpic viewer (works without agentY)" viewer "N - kept"
if [ -n "$REFUSED" ]; then printf '%s\n' "$REFUSED" | while IFS= read -r l; do echo; fail "NOT TOUCHED  $l"; done; fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_BASE="$PARENT_DIR/agentY-userdata-backup-$STAMP"

echo
printf '%s  6. ComfyUI Desktop  (default answer: N)%s\n' "$C_WHITE" "$C_OFF"
DESKTOP=0
if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1 && brew list --cask comfy >/dev/null 2>&1; then DESKTOP=1; fi
if [ "$DESKTOP" = 1 ]; then line "installed via brew. Remove it with:  brew uninstall --cask comfy" "$C_GRAY"; else line "not installed via brew (nothing to remove)" "$C_GRAY"; fi
echo
printf '%s  KEPT, always:%s\n' "$C_WHITE" "$C_OFF"
if [ -n "$COMFY_FOUND" ]; then printf '%s\n' "$COMFY_FOUND" | while IFS= read -r c; do line "ComfyUI itself and its models: $c" "$C_GRAY"; done; fi
line "Ollama, Strata, Blender, and everything outside the folders listed above" "$C_GRAY"
line "uv's cache and the Python builds uv downloaded (see the hints at the end)" "$C_GRAY"

# Locks ---------------------------------------------------------------------------------------
get_locks() {   # stdin: selected item lines -> prints one message per lock
  local own dirs d pid cmd root
  own="$(own_pids)"
  dirs="$(awk -F"$TAB" '$4==0{print $2}')"
  ps -axo pid=,command= 2>/dev/null | while read -r pid cmd; do
    case "
$own
" in *"
$pid
"*) continue ;; esac
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      case "$cmd" in "$d"/*|*" $d/"*) echo "PID $pid runs from $d"; break ;; esac
    done <<EOF_L
$dirs
EOF_L
    case "$cmd" in *main.py*)
      case "$cmd" in *omfy*|*OMFY*)
        while IFS= read -r root; do
          [ -n "$root" ] || continue
          case "$cmd" in *"$root"*)
            if printf '%s\n' "$dirs" | grep -q "^$root/custom_nodes/"; then echo "ComfyUI is running (PID $pid) from $root - close it first"; fi ;;
          esac
        done <<EOF_R
$COMFY_FOUND
EOF_R
      ;; esac ;;
    esac
  done | sort -u
}

show_hints() {
  local cd
  header "Things this script never does for you"
  cd="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
  [ "$(uname -s)" = "Darwin" ] || cd="${XDG_CONFIG_HOME:-$HOME/.config}/Claude/claude_desktop_config.json"
  if [ -f "$cd" ] && grep -qi 'agenty' "$cd" 2>/dev/null; then
    info "Claude Desktop still lists agentY-mcp: open $cd and remove the \"agentY\" entry under \"mcpServers\"."
  else
    info "If you registered agentY-mcp with Claude Desktop: remove the \"agentY\" key under \"mcpServers\" in $cd."
  fi
  info "uv keeps a download cache of every package (shared with your other Python projects). To free it:  uv cache clean"
  info "The Deadline plugin is removed by its own script, BEFORE agentY is deleted:  python3 scripts/install_deadline_plugin.py --uninstall"
  if printf '%s\n' "$ITEMS" | grep -q 'chainlit-datalayer'; then
    info "The old Chainlit data layer ran Postgres in Docker. Stop it and drop its volumes first:  docker compose -f <chainlit-datalayer>/docker-compose.yml down -v"
  fi
}

if [ "$DRY_RUN" = 1 ]; then
  header "Checks"
  locks="$( { items_of code; items_of data; items_of panel; } | get_locks )"
  if [ -n "$locks" ]; then printf '%s\n' "$locks" | while IFS= read -r l; do fail "would block the removal: $l"; done
  else success "nothing holds the folders open"; fi
  info "A real run asks before every group. --yes answers: stop the host, remove agentY + siblings, back up and remove data, remove the panel; keeps the viewer."
  show_hints
  echo
  printf '%s  Dry run finished. Nothing was changed.%s\n' "$C_CYAN" "$C_OFF"
  exit 0
fi

# == Questions ===============================================================================
header "Questions"
DO_HOST=0; DO_CODE=0; DO_DATA=0; DO_BACKUP=0; DO_PANEL=0; DO_VIEWER=0; DO_DESKTOP=0
HAS_CODE=0; HAS_DATA=0; HAS_PANEL=0; HAS_VIEWER=0
[ "$(count_of code)" = 0 ] || HAS_CODE=1
[ "$(count_of data)" = 0 ] || HAS_DATA=1
[ "$(count_of panel)" = 0 ] || HAS_PANEL=1
[ "$(count_of viewer)" = 0 ] || HAS_VIEWER=1
if [ "$YES" = 1 ]; then
  DO_HOST=1; DO_CODE=$HAS_CODE; DO_PANEL=$HAS_PANEL
  if [ "$HAS_DATA" = 1 ] && [ "$KEEP_USER_DATA" = 0 ]; then DO_DATA=1; [ "$NO_BACKUP" = 1 ] || DO_BACKUP=1; fi
  [ "$HAS_VIEWER" = 1 ] && [ "$INCLUDE_VIEWER" = 1 ] && DO_VIEWER=1
  [ "$DESKTOP" = 1 ] && [ "$INCLUDE_DESKTOP" = 1 ] && DO_DESKTOP=1
else
  if [ -n "$HOST_LINES" ]; then ask "1. Stop the running agentY host?" y && DO_HOST=1; fi
  if [ "$HAS_CODE" = 1 ]; then ask "2. Remove agentY, agentY-core and agentY-mcp (code and venvs)?" y && DO_CODE=1; fi
  if [ "$HAS_DATA" = 1 ]; then
    if ask "3. Also DELETE your data (conversations, memory, settings, keys, tokens)? This cannot be undone." n; then
      DO_DATA=1
      if [ "$NO_BACKUP" = 0 ]; then ask "   Back it up as a zip next to the folder first?" y && DO_BACKUP=1; fi
    fi
  fi
  if [ "$HAS_PANEL" = 1 ]; then ask "4. Remove the agentY panel from ComfyUI's custom_nodes?" y && DO_PANEL=1; fi
  if [ "$HAS_VIEWER" = 1 ]; then ask "5. Also remove the bEpic viewer (it works without agentY)?" n && DO_VIEWER=1; fi
  if [ "$DESKTOP" = 1 ]; then ask "6. Run 'brew uninstall' for ComfyUI Desktop now?" n && DO_DESKTOP=1; fi
fi
if [ "$((DO_HOST + DO_CODE + DO_DATA + DO_PANEL + DO_VIEWER + DO_DESKTOP))" = 0 ]; then
  info "Nothing selected - nothing changed."; show_hints; exit 0
fi

SELECTED=""
[ "$DO_CODE" = 0 ] || SELECTED="$(items_of code)"
if [ "$DO_DATA" = 1 ] && [ "$DO_CODE" = 0 ]; then SELECTED="$(printf '%s\n%s' "$SELECTED" "$(items_of data)")"; fi
[ "$DO_PANEL" = 0 ] || SELECTED="$(printf '%s\n%s' "$SELECTED" "$(items_of panel)")"
[ "$DO_VIEWER" = 0 ] || SELECTED="$(printf '%s\n%s' "$SELECTED" "$(items_of viewer)")"
SELECTED="$(printf '%s\n' "$SELECTED" | grep .)"

# 1. Stop the host ---------------------------------------------------------------------------
if [ "$DO_HOST" = 1 ] && [ -n "$HOST_LINES" ]; then
  printf '%s\n' "$HOST_LINES" | while read -r pid rest; do kill "$pid" 2>/dev/null; done
  sleep 1
  printf '%s\n' "$HOST_LINES" | while read -r pid rest; do kill -9 "$pid" 2>/dev/null; done
  success "agentY host stopped"
  add_line REMOVED "stopped the running agentY host"
fi

# Locks: before anything is removed, so a refusal leaves everything as it was.
if [ -n "$SELECTED" ]; then
  locks="$(printf '%s\n' "$SELECTED" | get_locks)"
  if [ -n "$locks" ]; then
    echo
    printf '%s\n' "$locks" | while IFS= read -r l; do fail "$l"; done
    echo
    die "Close what is listed above (ComfyUI, Claude Desktop, a terminal in these folders, the agent window) and run the uninstaller again. NOTHING was removed." 3
  fi
fi

# Backup --------------------------------------------------------------------------------------
BACKUP_DONE=""
make_archive() {   # $1 = out base; stdin: paths relative to PARENT_DIR -> prints the archive path
  local base="$1" rels py
  rels="$(cat)"
  if command -v zip >/dev/null 2>&1; then
    ( cd "$PARENT_DIR" && printf '%s\n' "$rels" | zip -rq "$base.zip" -@ ) >/dev/null 2>&1 && { echo "$base.zip"; return 0; }
    rm -f "$base.zip"
  fi
  for py in python3 python; do
    command -v "$py" >/dev/null 2>&1 || continue
    "$py" -c "import zipfile" >/dev/null 2>&1 || continue
    ( cd "$PARENT_DIR" && printf '%s\n' "$rels" | "$py" -c '
import os, sys, zipfile
out = sys.argv[1]
z = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True)
for rel in sys.stdin.read().splitlines():
    if not rel:
        continue
    if os.path.isdir(rel) and not os.path.islink(rel):
        for d, ds, fs in os.walk(rel):
            ds[:] = [x for x in ds if not os.path.islink(os.path.join(d, x))]
            for f in fs:
                p = os.path.join(d, f)
                try:
                    z.write(p, p.replace(os.sep, "/"))
                except OSError as e:
                    sys.stderr.write("skipped %s: %s\n" % (p, e))
    elif os.path.isfile(rel):
        z.write(rel, rel.replace(os.sep, "/"))
z.close()
' "$base.zip" ) >/dev/null 2>&1 && { echo "$base.zip"; return 0; }
    rm -f "$base.zip"
  done
  ( cd "$PARENT_DIR" && printf '%s\n' "$rels" | tar -czf "$base.tar.gz" -T - ) >/dev/null 2>&1 && { echo "$base.tar.gz"; return 0; }
  rm -f "$base.tar.gz"
  return 1
}

if [ "$DO_DATA" = 1 ] && [ "$DO_BACKUP" = 1 ]; then
  header "Backup"
  rels="$(items_of data | awk -F"$TAB" '{print $2}' | while IFS= read -r p; do printf '%s\n' "${p#$PARENT_DIR/}"; done)"
  if arc="$(printf '%s\n' "$rels" | make_archive "$BACKUP_BASE")" && [ -s "$arc" ]; then
    BACKUP_DONE="$arc"
    success "backed up -> $arc ($(fmt_kb "$(size_kb "$arc")"))"
  else
    fail "The backup failed. Your data is NOT being deleted."
    DO_DATA=0
  fi
fi

# Removal --------------------------------------------------------------------------------------
header "Removing"
cd "${TMPDIR:-/tmp}" 2>/dev/null || cd /

remove_one() {   # path label
  local why
  why="$(forbidden_reason "$1")"
  if [ -n "$why" ]; then add_line FAILED "$1 - refused at the last check: $why"; return; fi
  if [ -L "$1" ]; then rm -f "$1"; else chmod -R u+w "$1" 2>/dev/null; rm -rf "$1"; fi
  if [ -e "$1" ] || [ -L "$1" ]; then add_line FAILED "$1 - not completely removed"; fail "$2 - not completely removed"
  else add_line REMOVED "$2  ($1)"; success "removed $2"; fi
}

remove_keeping() {   # dir - keeps every path in DATA_KEEP
  local dir="$1" e
  for e in "$dir"/* "$dir"/.[!.]* "$dir"/..?*; do
    [ -e "$e" ] || [ -L "$e" ] || continue
    if printf '%s\n' "$DATA_KEEP" | grep -qxF "$e"; then continue; fi
    if [ -d "$e" ] && [ ! -L "$e" ] && printf '%s\n' "$DATA_KEEP" | grep -qF "$e/"; then remove_keeping "$e"; continue; fi
    if [ -L "$e" ]; then rm -f "$e"; else chmod -R u+w "$e" 2>/dev/null; rm -rf "$e"; fi
    if [ -e "$e" ] || [ -L "$e" ]; then add_line FAILED "$e - not completely removed"; fi
  done
}

remove_group_items() {   # stdin item lines
  local g p l k kb how
  while IFS="$TAB" read -r g p l k kb how; do [ -n "$g" ] && remove_one "$p" "$l"; done
}

[ "$DO_PANEL" = 0 ] || items_of panel | remove_group_items
[ "$DO_VIEWER" = 0 ] || items_of viewer | remove_group_items
# `|` runs the loop in a subshell: removals recorded there are not seen here, so the loops below run in this shell.
if [ "$DO_CODE" = 1 ]; then
  while IFS="$TAB" read -r g p l k kb how; do
    [ -n "$g" ] || continue
    if [ "$k" = "1" ]; then remove_one "$p" "$l"; fi
  done <<EOF_LINKS
$(items_of code)
EOF_LINKS
  while IFS="$TAB" read -r g p l k kb how; do
    [ -n "$g" ] || continue
    [ "$k" = "1" ] && continue
    [ "$p" = "$PROJECT" ] && continue
    if [ "$DO_DATA" = 1 ]; then remove_one "$p" "$l"
    else remove_keeping "$p"; add_line REMOVED "$l code and venv ($p)"; success "removed $l (your data kept)"; fi
  done <<EOF_CODE2
$(items_of code)
EOF_CODE2
  if [ "$DO_DATA" = 1 ]; then remove_one "$PROJECT" "agentY"
  else remove_keeping "$PROJECT"; add_line REMOVED "agentY code and venv ($PROJECT)"; success "removed agentY (your data kept)"; fi
elif [ "$DO_DATA" = 1 ]; then
  while IFS="$TAB" read -r g p l k kb how; do
    [ -n "$g" ] && remove_one "$p" "$l"
  done <<EOF_DATA2
$(items_of data)
EOF_DATA2
fi

if [ "$DO_DESKTOP" = 1 ]; then
  info "brew uninstall --cask comfy"
  if brew uninstall --cask comfy; then add_line REMOVED "ComfyUI Desktop (brew)"; else add_line FAILED "ComfyUI Desktop - brew failed"; fi
fi

# Summary --------------------------------------------------------------------------------------
while IFS="$TAB" read -r g p l k kb how; do
  [ -n "$g" ] || continue
  { [ -e "$p" ] || [ -L "$p" ]; } || continue
  case "$g" in
    viewer) [ "$DO_VIEWER" = 1 ] || add_line KEPT "the bEpic viewer - works without agentY: $p" ;;
    data)   [ "$DO_DATA" = 1 ] || add_line KEPT "your data: $p" ;;
    panel)  [ "$DO_PANEL" = 1 ] || add_line KEPT "the panel: $p" ;;
    code)   [ "$DO_CODE" = 1 ] || add_line KEPT "$l: $p" ;;
  esac
done <<EOF_SUM
$ITEMS
EOF_SUM
if [ -n "$COMFY_FOUND" ]; then while IFS= read -r c; do add_line KEPT "ComfyUI and its models: $c"; done <<EOF_K
$COMFY_FOUND
EOF_K
fi
add_line KEPT "Ollama, Strata, Blender and every folder outside the ones listed"

header "Summary"
printf '%s  Removed:%s\n' "$C_CYAN" "$C_OFF"
if [ -n "$REMOVED" ]; then printf '%s\n' "$REMOVED" | while IFS= read -r l; do line "$l"; done; else line "(nothing)" "$C_GRAY"; fi
printf '%s  Kept:%s\n' "$C_CYAN" "$C_OFF"
printf '%s\n' "$KEPT" | awk '!s[$0]++' | while IFS= read -r l; do line "$l"; done
if [ -n "$REFUSED" ]; then printf '%s  Not touched (not verified as agentY'"'"'s):%s\n' "$C_CYAN" "$C_OFF"; printf '%s\n' "$REFUSED" | while IFS= read -r l; do line "$l" "$C_YELLOW"; done; fi
if [ -n "$BACKUP_DONE" ]; then printf '%s  Backup of your data:%s\n' "$C_CYAN" "$C_OFF"; line "$BACKUP_DONE"; fi
if [ -n "$FAILED" ]; then printf '%s  Could not remove:%s\n' "$C_RED" "$C_OFF"; printf '%s\n' "$FAILED" | while IFS= read -r l; do line "$l" "$C_RED"; done; fi
show_hints
echo
[ -z "$FAILED" ] || exit 1
exit 0
