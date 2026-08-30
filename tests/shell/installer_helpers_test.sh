#!/usr/bin/env bash
# Exercise install_agent.sh's helper functions. $1 = repo root.
#
# The helpers are sourced out of the real script rather than copied here, so this
# tests what ships. Everything above the "=====" banner is definitions; the stages
# below it are what we must NOT run.
set -u
ROOT="${1:?usage: installer_helpers_test.sh <repo-root>}"

NON_INTERACTIVE=1
C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_WHITE=""; C_OFF=""
header(){ :; }; success(){ :; }; info(){ :; }; fail(){ :; }; plain(){ :; }
die(){ echo "DIE: $1"; exit 1; }
PARENT_DIR="/nonexistent"

FNS="$(mktemp)"
sed -n '/^# -- Low-level helpers/,/^# =====/p' "$ROOT/install_agent.sh" > "$FNS"
# shellcheck disable=SC1090
. "$FNS"

T="$(mktemp -d)"; ok=0; bad=0
check(){ if [ "$2" = "$3" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1: got [$2] want [$3]"; fi; }

# ── .env read/write ─────────────────────────────────────────────────────────
printf 'HF_TOKEN=hf_...\nANTHROPIC_API_KEY=sk-ant-real\n# a comment\nEMPTY=\n' > "$T/.env"
check "read plain"        "$(get_env_value "$T/.env" ANTHROPIC_API_KEY)" "sk-ant-real"
check "read placeholder"  "$(get_env_value "$T/.env" HF_TOKEN)"          "hf_..."
check "read empty"        "$(get_env_value "$T/.env" EMPTY)"             ""
check "read missing key"  "$(get_env_value "$T/.env" NOPE)"              ""
check "read missing file" "$(get_env_value "$T/none.env" HF_TOKEN)"      ""

set_env_value "$T/.env" HF_TOKEN "hf_abc123"
check "overwrite"      "$(get_env_value "$T/.env" HF_TOKEN)" "hf_abc123"
check "sibling kept"   "$(get_env_value "$T/.env" ANTHROPIC_API_KEY)" "sk-ant-real"
check "comment kept"   "$(grep -c '^# a comment' "$T/.env")" "1"
check "no line growth" "$(wc -l < "$T/.env" | tr -d ' ')" "4"

set_env_value "$T/.env" NEW_KEY "added"
check "append"         "$(get_env_value "$T/.env" NEW_KEY)" "added"

# A token containing the characters sed's replacement and awk's -v both treat as
# syntax. awk -v ate the backslash and returned a key one character short — set,
# plausible-looking, and wrong.
set_env_value "$T/.env" HF_TOKEN 'a/b&c\d$e'
check "sed/awk-hostile" "$(get_env_value "$T/.env" HF_TOKEN)" 'a/b&c\d$e'

set_env_value "$T/fresh.env" K "v"
check "creates file"   "$(get_env_value "$T/fresh.env" K)" "v"

# ── placeholder / masking ───────────────────────────────────────────────────
for p in "" "hf_..." "sk-ant-..." "comfyui-..." "sk-..."; do
  if is_placeholder "$p"; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL placeholder [$p] not detected"; fi
done
for r in "hf_abc" "sk-ant-real" "x"; do
  if is_placeholder "$r"; then bad=$((bad+1)); echo "FAIL real value [$r] called placeholder"; else ok=$((ok+1)); fi
done
check "mask unset" "$(masked 'hf_...')"            "(not set)"
check "mask short" "$(masked '12345')"             "*****"
check "mask long"  "$(masked 'sk-ant-abcdefghij')" "sk-a****..."

# ── ComfyUI detection ───────────────────────────────────────────────────────
mkdir -p "$T/Comfy/custom_nodes" "$T/wrap/ComfyUI/custom_nodes" "$T/plain"
check "direct root"  "$(basename "$(test_comfyui_dir "$T/Comfy")")" "Comfy"
check "wrapper dir"  "$(basename "$(test_comfyui_dir "$T/wrap")")"  "ComfyUI"
check "not comfyui"  "$(test_comfyui_dir "$T/plain")"               ""
check "missing dir"  "$(test_comfyui_dir "$T/ghost")"               ""
check "empty arg"    "$(test_comfyui_dir "")"                       ""

# A path with a space — ordinary on a Mac, and the usual way a shell installer
# breaks on one.
mkdir -p "$T/My Stuff/ComfyUI/custom_nodes"
check "spaced path"  "$(basename "$(test_comfyui_dir "$T/My Stuff")")" "ComfyUI"

# ── read_secret, non-interactive ────────────────────────────────────────────
check "keeps real value" "$(read_secret "$T/.env" ANTHROPIC_API_KEY L H)" "sk-ant-real"
check "skips unset key"  "$(read_secret "$T/fresh.env" MISSING L H)"      ""

# ── ensure_env_file signals failure by RETURNING, not by exiting ────────────
# It is called as ENV_FILE="$(ensure_env_file "$dir")", and command substitution
# runs in a subshell: an `exit` in there ends only the subshell, and the installer
# would carry on with ENV_FILE holding the text of the error — then write the
# user's API keys into a path made of it.
mkdir -p "$T/proj"
printf 'HF_TOKEN=hf_...\n' > "$T/proj/.env_example"
envpath="$(ensure_env_file "$T/proj" 2>/dev/null)"; rc=$?
check "creates from example" "$rc" "0"
check "returns the path"     "$envpath" "$T/proj/.env"
check "the file exists"      "$([ -f "$T/proj/.env" ] && echo yes)" "yes"

envpath="$(ensure_env_file "$T/proj" 2>/dev/null)"; rc=$?
check "second call is fine"  "$rc" "0"
check "and same path"        "$envpath" "$T/proj/.env"

mkdir -p "$T/bare"       # no .env and no .env_example
envpath="$(ensure_env_file "$T/bare" 2>/dev/null)"; rc=$?
check "missing example fails" "$rc" "1"
check "and echoes no path"    "$envpath" ""

rm -rf "$T" "$FNS"
echo "installer helpers: $ok passed, $bad failed"
[ "$bad" -eq 0 ]
