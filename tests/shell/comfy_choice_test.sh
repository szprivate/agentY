#!/usr/bin/env bash
# Exercise install_agent.sh's ComfyUI discovery and choice (find_all_comfyui, choose_comfyui) against fake
# folder trees under a throwaway HOME. $1 = repo root.
#
# The functions are sourced out of the real script. Nothing real is looked at: HOME, PARENT_DIR and the
# process list are all stand-ins, and the answers come from stdin. What is checked is the behaviour of
# GitHub issue 79: every install is found and listed, the USER picks, declining everything picks nothing
# (no fallback to an install that was never offered), and a non-interactive run does not choose among several.
set -u
ROOT="${1:?usage: comfy_choice_test.sh <repo-root>}"

NON_INTERACTIVE=0
C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_WHITE=""; C_OFF=""
success(){ :; }; info(){ :; }; fail(){ echo "FAILMSG $1" >> "$MSGS"; }; plain(){ :; }

FNS="$(mktemp)"
sed -n '/^# Accept the ComfyUI root/,/^select_execution_mode()/p' "$ROOT/install_agent.sh" | sed '$d' > "$FNS"
# shellcheck disable=SC1090
. "$FNS"

T="$(mktemp -d)"; ok=0; bad=0
MSGS="$T/msgs"; : > "$MSGS"
check(){ if [ "$2" = "$3" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1: got [$2] want [$3]"; fi; }

export HOME="$T/home"
PARENT_DIR="$T/parent"
mkdir -p "$HOME" "$PARENT_DIR"
mk(){ mkdir -p "$1/custom_nodes"; }
mk "$HOME/ComfyUI"                                   # a clone in the home folder
mk "$HOME/Documents/ComfyUI"                         # ComfyUI Desktop's default data folder
mk "$HOME/ComfyUI-Installs/desk1"                    # a Desktop-managed install
mk "$T/elsewhere/mydata"                             # only the Desktop config.json knows this one
mkdir -p "$HOME/Library/Application Support/ComfyUI"
printf '{\n  "basePath": "%s",\n  "other": 1\n}\n' "$T/elsewhere/mydata" > "$HOME/Library/Application Support/ComfyUI/config.json"
mk "$T/proc/ComfyUI"; : > "$T/proc/ComfyUI/main.py"  # a running one
printf '__version__ = "0.3.60"\n' > "$HOME/ComfyUI/comfyui_version.py"
comfy_process_lines(){ printf '%s\n' "/usr/bin/python3 $T/proc/ComfyUI/main.py --port 8188" "/usr/bin/python3 /somewhere/else.py"; }

count(){ printf '%s\n' "$1" | grep -c . ; }

find_all_comfyui ""
check "five distinct installs" "$(count "$COMFY_FOUND")" "5"
case "$COMFY_FOUND" in *"$T/elsewhere/mydata"*) r=yes ;; *) r=no ;; esac
check "Desktop's config.json basePath is found" "$r" "yes"
case "$COMFY_FOUND" in *"$T/proc/ComfyUI"*) r=yes ;; *) r=no ;; esac
check "a running process is found" "$r" "yes"
check "a version is read without starting it" "$(comfy_label "$HOME/ComfyUI")" "v0.3.60"
case "$(comfy_label "$HOME/Documents/ComfyUI")" in *"data folder only"*) r=yes ;; *) r=no ;; esac
check "a data-only folder says so" "$r" "yes"

# The same folder reached twice (basePath = ~/Documents/ComfyUI) is one entry.
printf '{"basePath": "%s"}' "$HOME/Documents/ComfyUI" > "$HOME/Library/Application Support/ComfyUI/config.json"
find_all_comfyui ""
check "a folder named twice is listed once" "$(count "$COMFY_FOUND")" "4"

ask(){ NON_INTERACTIVE=0; choose_comfyui "${2:-}" >/dev/null 2>&1 <<<"$1"; printf '%s' "$RESOLVED_COMFY"; }
ask2(){ NON_INTERACTIVE=0; RESOLVED_COMFY=""; choose_comfyui "${2:-}" >/dev/null 2>&1 <<<"$1"; }

ask2 "2";           check "number 2 picks the second listed" "$RESOLVED_COMFY" "$(printf '%s\n' "$COMFY_FOUND" | sed -n 2p)"
ask2 "1";           check "number 1 picks the first listed" "$RESOLVED_COMFY" "$(printf '%s\n' "$COMFY_FOUND" | sed -n 1p)"
ask2 "";            check "Enter among several picks nothing" "$RESOLVED_COMFY" ""
ask2 "s";           check "s skips" "$RESOLVED_COMFY" ""
ask2 "n";           check "n skips" "$RESOLVED_COMFY" ""
ask2 "9";           check "a number outside the list picks nothing" "$RESOLVED_COMFY" ""
ask2 "0";           check "0 picks nothing" "$RESOLVED_COMFY" ""
ask2 "$T/proc/ComfyUI"; check "a typed path is accepted" "$RESOLVED_COMFY" "$T/proc/ComfyUI"
mkdir -p "$T/notcomfy"
ask2 "$T/notcomfy"; check "a folder without custom_nodes picks nothing" "$RESOLVED_COMFY" ""
grep -q "doesn't look like a ComfyUI" "$MSGS"; check "and says why" "$?" "0"

# One install: Enter and y use it, n declines (and nothing else is substituted).
only="$T/only"; mk "$only/ComfyUI"
HOME="$T/emptyhome"; mkdir -p "$HOME"; PARENT_DIR="$only"
comfy_process_lines(){ :; }
ask2 "";   check "a single install: Enter uses it" "$RESOLVED_COMFY" "$only/ComfyUI"
ask2 "y";  check "a single install: y uses it" "$RESOLVED_COMFY" "$only/ComfyUI"
ask2 "n";  check "a single install: n declines" "$RESOLVED_COMFY" ""
check "declining leaves nothing found to fall back on being used" "$(count "$COMFY_FOUND")" "1"

# --comfyui-path is the user's choice and the only candidate.
HOME="$T/home"; PARENT_DIR="$T/parent"
comfy_process_lines(){ printf '%s\n' "/usr/bin/python3 $T/proc/ComfyUI/main.py"; }
ask2 "" "$T/elsewhere/mydata"; check "an explicit path is used on Enter" "$RESOLVED_COMFY" "$T/elsewhere/mydata"
ask2 "n" "$T/elsewhere/mydata"; check "an explicit path can still be declined" "$RESOLVED_COMFY" ""

# Non-interactive: an explicit path, or the only install; several without a path = nothing.
NON_INTERACTIVE=1; RESOLVED_COMFY=""; choose_comfyui "$T/elsewhere/mydata" >/dev/null 2>&1 </dev/null
check "non-interactive with a path" "$RESOLVED_COMFY" "$T/elsewhere/mydata"
NON_INTERACTIVE=1; RESOLVED_COMFY=""; choose_comfyui "" >/dev/null 2>&1 </dev/null
check "non-interactive with several installs installs into none" "$RESOLVED_COMFY" ""
HOME="$T/emptyhome"; PARENT_DIR="$only"; comfy_process_lines(){ :; }
NON_INTERACTIVE=1; RESOLVED_COMFY=""; choose_comfyui "" >/dev/null 2>&1 </dev/null
check "non-interactive with exactly one install uses it" "$RESOLVED_COMFY" "$only/ComfyUI"
HOME="$T/emptyhome"; PARENT_DIR="$T/nothing"
NON_INTERACTIVE=1; RESOLVED_COMFY=""; choose_comfyui "" >/dev/null 2>&1 </dev/null
check "no install at all picks nothing" "$RESOLVED_COMFY" ""

echo "$ok passed, $bad failed"
[ "$bad" -eq 0 ]
