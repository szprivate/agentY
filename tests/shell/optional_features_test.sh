#!/usr/bin/env bash
# Exercise install_agent.sh's optional-features step (install_optional_deps) with a
# stub interpreter that stands in for scripts/optional_deps.py. $1 = repo root.
#
# The function and the menu are sourced out of the real script. What is checked:
# a package starts ticked, an action ("Set up agentY for Claude Code") starts
# unticked, Enter alone installs only the package, ticking the action runs its
# install, a non-interactive run keeps only an earlier yes.
set -u
ROOT="${1:?usage: optional_features_test.sh <repo-root>}"

NON_INTERACTIVE=0
C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_WHITE=""; C_OFF=""
header(){ :; }; success(){ :; }; info(){ :; }; fail(){ echo "FAILMSG $1"; }; plain(){ :; }
die(){ echo "DIE: $1"; exit 1; }
tput(){ echo 80; }

FNS="$(mktemp)"
sed -n '/^# -- Low-level helpers/,/^# =====/p' "$ROOT/install_agent.sh" > "$FNS"
sed -n '/^# -- Multi-select menu/,/^# =====/p' "$ROOT/install_agent.sh" >> "$FNS"
sed -n '/^install_optional_deps()/,/^}/p' "$ROOT/install_agent.sh" >> "$FNS"
# shellcheck disable=SC1090
. "$FNS"

T="$(mktemp -d)"; ok=0; bad=0
check(){ if [ "$2" = "$3" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1: got [$2] want [$3]"; fi; }

LOG="$T/log"
# The stub: the first argument is the script path (ignored), then the command.
cat > "$T/fakepy" <<STUB
#!/usr/bin/env bash
case "\$2" in
  list) printf 'reverse_image_search|Reverse image search|hint one|%s\n' "\${CHOICE_RIS:-}"
        printf 'claude_code_plugin|Set up agentY for Claude Code|hint two|%s\n' "\${CHOICE_CC-0}" ;;
  set) echo "set \$3 \$4" >> "$LOG" ;;
  install) echo "install \$3" >> "$LOG" ;;
esac
STUB
chmod +x "$T/fakepy"
venv_python(){ echo "$T/fakepy"; }
PROJECT_ROOT="$ROOT"

DOWN=$'\033[B'
run(){ : > "$LOG"; SM_ASSUME_TTY=1 install_optional_deps >/dev/null 2>&1 <<<"$1"; tr '\n' ';' < "$LOG"; }

check "enter alone: package yes, action no, only the package installs" \
  "$(run "")" "set reverse_image_search yes;set claude_code_plugin no;install reverse_image_search;"
check "ticking the action runs its install" \
  "$(run "${DOWN} ")" "set reverse_image_search yes;set claude_code_plugin yes;install reverse_image_search;install claude_code_plugin;"

NON_INTERACTIVE=1
check "non-interactive without an earlier yes installs nothing" "$(CHOICE_RIS=0 run "")" ""
check "non-interactive keeps an earlier yes for the action" \
  "$(CHOICE_RIS=0 CHOICE_CC=1 run "")" "install claude_code_plugin;"

echo "$ok passed, $bad failed"
[ "$bad" -eq 0 ]
