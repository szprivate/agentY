#!/usr/bin/env bash
# Exercise run_agent.sh's free_port. $1 = repo root.
#
# free_port is the only step in the launcher that destroys something: `kill -9` on
# a PID picked by matching a command line. lsof, ps and kill are all looked up by
# name, so this test defines them as shell functions and watches what would have
# happened — no real process is ever signalled.
set -u
ROOT="${1:?usage: free_port_test.sh <repo-root>}"

C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_OFF=""
say() { printf '%s\n' "$1"; }

FNS="$(mktemp)"
awk '/^free_port\(\)/,/^\}$/' "$ROOT/run_agent.sh" > "$FNS"
# shellcheck disable=SC1090
. "$FNS"

ok=0; bad=0
check(){ if [ "$2" = "$3" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1: got [$2] want [$3]"; fi; }
contains(){ case "$2" in *"$3"*) ok=$((ok+1)) ;; *) bad=$((bad+1)); echo "FAIL $1: [$2] lacks [$3]" ;; esac; }

KILLED=""
# Stand-ins. A function beats an external command in bash's lookup order, so the
# real lsof/ps/kill are never reached from here.
kill() { KILLED="$KILLED $2"; }
command() {   # only used as `command -v lsof`
  if [ "${2:-}" = "lsof" ] && [ "$FAKE_HAS_LSOF" = "0" ]; then return 1; fi
  return 0
}
sleep() { :; }         # the settle pause after a kill
lsof() { printf '%s\n' $FAKE_PIDS; }
# Called as `ps -o command= -p <pid>`, so the PID is the FOURTH argument. Reading
# $3 handed every process an empty command line, which free_port then correctly
# declined to kill — a harness that looked green while testing one branch six times.
ps() { eval "printf '%s\n' \"\${FAKE_CMD_${4}:-}\""; }

# free_port must run in THIS shell: `out="$(free_port ...)"` would put it in a
# subshell and $KILLED would die with it. Output goes through a file instead.
OUT="$(mktemp)"; out=""
run_free_port() { KILLED=""; free_port "$1" > "$OUT" 2>&1; out="$(cat "$OUT")"; }

# ── 1. A leftover agentY host is stopped ────────────────────────────────────
FAKE_HAS_LSOF=1; FAKE_PIDS="4242"; FAKE_CMD_4242="python -m src.agenty_ui_server --port 5000"
run_free_port 5000
check    "kills the old host"  "$KILLED" " 4242"
contains "and says so"         "$out" "leftover agentY host (PID 4242)"

# ── 2. A stranger on the port is REPORTED, never killed ─────────────────────
# The whole point of the pattern match. Someone's postgres is not ours to end.
FAKE_HAS_LSOF=1; FAKE_PIDS="777"; FAKE_CMD_777="/usr/local/pgsql/bin/postgres -D /data"
run_free_port 5000
check    "leaves it alone"     "$KILLED" ""
contains "warns"               "$out" "WARNING"
contains "names the pid"       "$out" "PID 777"
contains "suggests --port"     "$out" "--port"

# ── 3. A PID ps cannot describe is left alone ──────────────────────────────
# lsof named it and ps came back empty: it exited between the two calls, or it
# belongs to another user. Killing on no evidence is the failure this guards.
FAKE_HAS_LSOF=1; FAKE_PIDS="999"; FAKE_CMD_999=""
run_free_port 5000
check    "no kill on no info"  "$KILLED" ""
contains "says it cannot tell" "$out" "cannot be identified"

# ── 4. Several holders: only ours dies ──────────────────────────────────────
FAKE_HAS_LSOF=1; FAKE_PIDS="111 222"
FAKE_CMD_111="python -m src.agenty_ui_server"
FAKE_CMD_222="node /srv/app.js"
run_free_port 5000
check    "only ours"           "$KILLED" " 111"

# ── 5. Nothing on the port ──────────────────────────────────────────────────
FAKE_HAS_LSOF=1; FAKE_PIDS=""
run_free_port 5000
check    "nothing killed"      "$KILLED" ""
check    "and nothing said"    "$out"    ""

# ── 6. No lsof: skipped, and said out loud ─────────────────────────────────
FAKE_HAS_LSOF=0; FAKE_PIDS="4242"
run_free_port 5000
check    "no lsof, no kill"    "$KILLED" ""
contains "explains the skip"   "$out" "no lsof on PATH"

rm -f "$FNS"
echo "free_port: $ok passed, $bad failed"
[ "$bad" -eq 0 ]
