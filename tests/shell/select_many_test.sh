#!/usr/bin/env bash
# Exercise install_agent.sh's multi-select menu and the provider / MCP preset
# steps built on it. $1 = repo root.
#
# The functions are sourced out of the real script. Real arrow keys need a
# terminal, so the key-driven menu is fed through a pipe with SM_ASSUME_TTY=1:
# the same read loop, the same escape-sequence parsing, no tty.
set -u
ROOT="${1:?usage: select_many_test.sh <repo-root>}"

NON_INTERACTIVE=0
C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_WHITE=""; C_OFF=""
header(){ :; }; success(){ :; }; info(){ :; }; fail(){ :; }; plain(){ :; }
die(){ echo "DIE: $1"; exit 1; }
PARENT_DIR="/nonexistent"
tput(){ echo 80; }

FNS="$(mktemp)"
sed -n '/^# -- Low-level helpers/,/^# =====/p' "$ROOT/install_agent.sh" > "$FNS"
sed -n '/^# -- Multi-select menu/,/^# =====/p' "$ROOT/install_agent.sh" >> "$FNS"
# shellcheck disable=SC1090
. "$FNS"

T="$(mktemp -d)"; ok=0; bad=0
check(){ if [ "$2" = "$3" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1: got [$2] want [$3]"; fi; }

UP=$'\033[A'; DOWN=$'\033[B'
fresh(){ SM_TITLE="t"; SM_LABELS=("One" "Two" "Three"); SM_HINTS=("h1" "" "h3"); SM_SEL=("$@"); SM_NONE_SILENT=0; SM_RESULT=""; }
keys(){ SM_ASSUME_TTY=1 select_many >/dev/null 2>&1 <<<"$1"; }   # one here-string: keys, then a newline = Enter

# Enter at once keeps what was pre-ticked.
fresh 0 1 0; keys ""
check "enter keeps preticks" "$SM_RESULT" "1"
# Space ticks the cursor row; Down moves; Enter confirms.
fresh 0 0 0; keys " ${DOWN} ${DOWN} "
check "space+down ticks 0,1,2" "$SM_RESULT" "0 1 2"
# Space on a ticked row clears it.
fresh 1 1 0; keys " "
check "space clears" "$SM_RESULT" "1"
# Up from the first row wraps to the last.
fresh 0 0 0; keys "${UP} "
check "up wraps" "$SM_RESULT" "2"
# Down past the last wraps to the first.
fresh 0 0 0; keys "${DOWN}${DOWN}${DOWN} "
check "down wraps" "$SM_RESULT" "0"
# 'a' ticks all, again clears all.
fresh 0 1 0; keys "a"
check "a ticks all" "$SM_RESULT" "0 1 2"
fresh 1 1 1; keys "a"
check "a clears all" "$SM_RESULT" ""
# vi keys
fresh 0 0 0; keys "j "
check "j moves down" "$SM_RESULT" "1"
# EOF (closed input) confirms instead of spinning.
fresh 0 1 0; SM_ASSUME_TTY=1 select_many >/dev/null 2>&1 </dev/null
check "eof confirms" "$SM_RESULT" "1"

# The numbered fallback (no terminal here: stdin is a pipe).
fresh 0 1 0; select_many >/dev/null 2>&1 <<<"1,3"
check "numbers" "$SM_RESULT" "0 2"
fresh 0 1 0; select_many >/dev/null 2>&1 <<<""
check "numbers: enter keeps" "$SM_RESULT" "1"
fresh 0 1 0; select_many >/dev/null 2>&1 <<<"none"
check "numbers: none" "$SM_RESULT" ""
fresh 0 1 0; select_many >/dev/null 2>&1 <<<"9
2"
check "numbers: retry after bad" "$SM_RESULT" "1"
fresh 0 1 0; select_many >/dev/null 2>&1 </dev/null
check "numbers: closed stdin keeps" "$SM_RESULT" "1"

# --non-interactive: no prompt, pre-ticked kept (or nothing when asked).
NON_INTERACTIVE=1
fresh 1 0 1; select_many >/dev/null 2>&1 </dev/null
check "noninteractive keeps preticks" "$SM_RESULT" "0 2"
fresh 1 0 1; SM_NONE_SILENT=1; select_many >/dev/null 2>&1 </dev/null
check "noninteractive none-silent" "$SM_RESULT" ""
NON_INTERACTIVE=0

# Providers: pre-ticked from .env, only ticked ones asked.
printf 'ANTHROPIC_API_KEY=sk-ant-REAL\nDASHSCOPE_API_KEY=\nGOOGLE_API_KEY=gk-real\n' > "$T/.env"
asked=""
read_secret(){ asked="$asked $2"; }
SM_ASSUME_TTY=1 select_llm_providers "$T/.env" >/dev/null 2>&1 <<<""
check "providers asked = those set in .env" "$asked" " ANTHROPIC_API_KEY GOOGLE_API_KEY"
asked=""
printf 'HF_TOKEN=x\n' > "$T/.env2"
SM_ASSUME_TTY=1 select_llm_providers "$T/.env2" >/dev/null 2>&1 <<<""
check "providers: nothing set -> Anthropic" "$asked" " ANTHROPIC_API_KEY"
asked=""
SM_ASSUME_TTY=1 select_llm_providers "$T/.env2" >/dev/null 2>&1 <<<"jjjj "
check "providers: tick Ollama = no key asked" "$asked" " ANTHROPIC_API_KEY"
NON_INTERACTIVE=1; asked=""
select_llm_providers "$T/.env" >/dev/null 2>&1 </dev/null
check "providers non-interactive: keys of set ones only" "$asked" " ANTHROPIC_API_KEY GOOGLE_API_KEY"
NON_INTERACTIVE=0

# MCP presets, end to end with the real helper on a scratch root: the menu, the
# keys asked, the merge into an existing mcp.json, the finish-later text.
# python is a native program on Windows: it needs C:/-style paths inside -c strings.
npath(){ if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
servers(){ python -c "import json,sys;print(sorted(json.load(open(sys.argv[1]))['servers']))" "$(npath "$R/config/mcp.json")"; }
R="$T/root"; mkdir -p "$R/scripts" "$R/config" "$R/.venv/bin"
cp "$ROOT/scripts/mcp_presets.py" "$R/scripts/"
printf '#!/bin/sh\nexec python "$@"\n' > "$R/.venv/bin/python"; chmod +x "$R/.venv/bin/python"
cat > "$R/config/mcp_presets.json" <<'JSON'
{"presets":[
 {"id":"alpha","name":"Alpha","category":"Test","description":"a hosted one","server":{"enabled":true,"transport":"http","url":"https://a.example/mcp","auth":"none","headers":{"Authorization":"Bearer ${ALPHA_KEY}"}},
  "env_keys":[{"key":"ALPHA_KEY","label":"Alpha key","help_url":"","optional":false}],"requires":[],"setup_steps":"open the Alpha app and press go","default_selected":false},
 {"id":"beta","name":"Beta","category":"Test","description":"a local one","server":{"enabled":true,"transport":"stdio","command":"true","args":[],"auth":"none"},
  "env_keys":[],"requires":[],"setup_steps":"","default_selected":true}]}
JSON
printf '%s\n' '{"servers":{"mine":{"enabled":true,"transport":"http","url":"https://m.example"}}}' > "$R/config/mcp.json"
PROJECT_ROOT="$R"; asked=""; OUT="$T/out"; : > "$OUT"
plain(){ printf '%s\n' "$1" >> "$OUT"; }; info(){ printf '%s\n' "$1" >> "$OUT"; }; success(){ printf '%s\n' "$1" >> "$OUT"; }
# Beta is pre-ticked; Space ticks Alpha (the cursor starts on it); Enter.
SM_ASSUME_TTY=1 install_mcp_presets "$T/.env" >>"$OUT" 2>>"$T/dbg" <<<" "
check "preset keys asked for ticked ones only" "$asked" " ALPHA_KEY"
check "mcp.json keeps mine, adds both"   "$(servers)" "['alpha', 'beta', 'mine']"
check "placeholder kept" "$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['servers']['alpha']['headers']['Authorization'])" "$(npath "$R/config/mcp.json")")" 'Bearer ${ALPHA_KEY}'
check "finish-later printed" "$(grep -c 'open the Alpha app' "$OUT")" "1"
# A second run skips what exists and drops nothing.
asked=""
SM_ASSUME_TTY=1 install_mcp_presets "$T/.env" >>"$OUT" 2>>"$T/dbg" <<<" "
check "second run keeps all"  "$(servers)" "['alpha', 'beta', 'mine']"
check "second run says so"    "$(grep -c "already configured" "$OUT")" "2"
# --non-interactive leaves mcp.json alone.
NON_INTERACTIVE=1; printf '%s\n' '{"servers":{}}' > "$R/config/mcp.json"
install_mcp_presets "$T/.env" >/dev/null 2>&1 </dev/null
check "noninteractive: mcp.json untouched" "$(cat "$R/config/mcp.json")" '{"servers":{}}'
NON_INTERACTIVE=0

[ -f "$T/dbg" ] && cat "$T/dbg"
rm -rf "$T" "$FNS"
echo "select_many: $ok passed, $bad failed"
[ "$bad" -eq 0 ]
