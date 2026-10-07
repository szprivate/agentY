#!/usr/bin/env bash
# Exercise run_agent.sh's update_repo against real git repos. $1 = repo root.
#
# This is the part of the launcher that touches the user's working copy, so it is
# tested against actual commits rather than mocked: what matters is that a dirty
# checkout keeps its local work, a diverged branch is refused, and nothing is ever
# discarded — claims only real git can settle.
set -u
ROOT="${1:?usage: update_repo_test.sh <repo-root>}"

C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_OFF=""
say() { printf '%s\n' "$1"; }

FNS="$(mktemp)"
awk '/^update_channel\(\)/,/^  return 0$/' "$ROOT/run_agent.sh" > "$FNS"
echo "}" >> "$FNS"          # the awk range stops before update_repo's closing brace
# shellcheck disable=SC1090
. "$FNS"

CHANNEL=dev          # what the cases below were written for; the channel cases set it
PROJECT_ROOT="$ROOT"

ok=0; bad=0
check(){ if [ "$2" = "$3" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1: got [$2] want [$3]"; fi; }

T="$(mktemp -d)"
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t

new_pair() {   # $1 = name -> a bare "remote" plus a clone with an upstream
  git init -q --bare "$T/$1.git"
  git init -q "$T/$1"; cd "$T/$1" || exit 1
  echo one > a.txt; echo req0 > requirements.txt; echo keep > untouched.txt
  git add -A >/dev/null; git commit -qm first
  git remote add origin "$T/$1.git"; git push -q -u origin HEAD:master 2>/dev/null
  git branch --set-upstream-to=origin/master >/dev/null 2>&1
  cd "$T" || exit 1
}

push_upstream() {   # $1 = name  $2 = file  $3 = content
  rm -rf "$T/push"; git clone -q "$T/$1.git" "$T/push"
  cd "$T/push" || exit 1
  printf '%s\n' "$3" > "$2"; git add -A >/dev/null; git commit -qm "upstream $2"
  git push -q origin HEAD:master; cd "$T" || exit 1
}

# ── 1. A clean fast-forward ─────────────────────────────────────────────────
new_pair r1
push_upstream r1 a.txt two
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r1 "$T/r1" > "$T/out1" 2>&1
check "ff sets moved"   "$REPO_MOVED" "1"
check "ff took content" "$(cat "$T/r1/a.txt")" "two"
check "ff no dep flag"  "$DEPS_CHANGED" ""

# ── 2. requirements.txt moved -> the venv needs reinstalling ────────────────
push_upstream r1 requirements.txt req1
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r1 "$T/r1" > "$T/out2" 2>&1
check "deps flagged"    "$DEPS_CHANGED" " r1"

# ── 3. Dirty file NOT touched upstream -> kept, and still fast-forwards ─────
push_upstream r1 a.txt three
echo "my local work" > "$T/r1/untouched.txt"
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r1 "$T/r1" > "$T/out3" 2>&1
check "local work kept" "$(cat "$T/r1/untouched.txt")" "my local work"
check "updated anyway"  "$(cat "$T/r1/a.txt")" "three"
check "and says so"     "$(grep -c 'none of them touched by this update' "$T/out3")" "1"

# ── 4. Dirty file that IS touched upstream -> stashed, never discarded ──────
git -C "$T/r1" checkout -q -- untouched.txt
push_upstream r1 a.txt four
echo "my version" > "$T/r1/b_local.txt"
rm -rf "$T/push"; git clone -q "$T/r1.git" "$T/push"
( cd "$T/push" && echo "upstream version" > b_local.txt && git add -A >/dev/null \
  && git commit -qm "upstream adds b_local" && git push -q origin HEAD:master )
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r1 "$T/r1" > "$T/out4" 2>&1
check "collision named" "$(grep -c 'also changed upstream' "$T/out4")" "1"
check "work is stashed" "$(git -C "$T/r1" stash list | wc -l | tr -d ' ')" "1"
check "repo still sane" "$(git -C "$T/r1" rev-parse --verify HEAD >/dev/null 2>&1 && echo yes)" "yes"

# ── 5. Diverged -> refuses, and changes nothing ─────────────────────────────
new_pair r2
push_upstream r2 a.txt remote
cd "$T/r2" && echo mine > mine.txt && git add -A >/dev/null && git commit -qm "local only" && cd "$T"
head_before="$(git -C "$T/r2" rev-parse HEAD)"
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r2 "$T/r2" > "$T/out5" 2>&1
check "says diverged"   "$(grep -c 'has diverged' "$T/out5")" "1"
check "HEAD untouched"  "$(git -C "$T/r2" rev-parse HEAD)" "$head_before"
check "not marked moved" "$REPO_MOVED" "0"

# ── 6. No upstream configured -> a skip, not an error ───────────────────────
git init -q "$T/r3"
cd "$T/r3" && echo x > x.txt && git add -A >/dev/null && git commit -qm only && cd "$T"
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r3 "$T/r3" > "$T/out6" 2>&1
check "no upstream"     "$(grep -c 'no upstream branch' "$T/out6")" "1"

# ── 7. Not a git checkout -> silent no-op ───────────────────────────────────
mkdir -p "$T/r4"
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r4 "$T/r4" > "$T/out7" 2>&1
check "non-repo silent" "$(wc -c < "$T/out7" | tr -d ' ')" "0"

# ── 8. Already current ──────────────────────────────────────────────────────
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r1 "$T/r1" > "$T/out8" 2>&1
check "up to date"      "$(grep -c 'is up to date' "$T/out8")" "1"
check "current !moved"  "$REPO_MOVED" "0"

# ── 9. A dirty path containing a space survives the intersection ───────────
new_pair r5
push_upstream r5 a.txt v2
mkdir -p "$T/r5/sub dir"; echo local > "$T/r5/sub dir/my file.txt"
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r5 "$T/r5" > "$T/out9" 2>&1
check "spaced path kept" "$(cat "$T/r5/sub dir/my file.txt")" "local"
check "spaced path ff'd" "$(cat "$T/r5/a.txt")" "v2"

# ── 10. REPO_MOVED does not leak from the previous repo ────────────────────
# The launcher calls update_repo in a loop over the sidebar-extension checkouts and
# reads REPO_MOVED after each one. Without a reset at the top of the function, the
# first checkout that moved would make every later one look like it moved too — and
# the user is told to restart ComfyUI for a change that never arrived.
# NOTE: no reset between these calls. That is the point.
new_pair r6
push_upstream r6 a.txt moved
DEPS_CHANGED=""; REPO_MOVED=0
update_repo r6 "$T/r6" > "$T/out10a" 2>&1
check "first one moved"  "$REPO_MOVED" "1"
update_repo r6 "$T/r6" > "$T/out10b" 2>&1      # now up to date
check "second not moved" "$REPO_MOVED" "0"
mkdir -p "$T/r7"
update_repo r6 "$T/r6" > /dev/null 2>&1
update_repo r7 "$T/r7" > "$T/out10c" 2>&1      # not a repo at all
check "non-repo clears"  "$REPO_MOVED" "0"

# ── 11. The stable channel follows origin/stable, not the branch's own head ──
new_pair r8
git -C "$T/r8" push -q origin HEAD:refs/heads/stable 2>/dev/null     # the release, at "one"
push_upstream r8 a.txt dev-only                                        # master moves on
CHANNEL=stable
update_repo r8 "$T/r8" > "$T/out11" 2>&1
check "stable ignores dev"   "$(cat "$T/r8/a.txt")" "one"
check "stable not moved"     "$REPO_MOVED" "0"
check "follows origin/stable" "$(git -C "$T/r8" rev-parse --abbrev-ref '@{u}')" "origin/stable"

# a release is made: stable moves up to master
git -C "$T/r8.git" update-ref refs/heads/stable refs/heads/master
update_repo r8 "$T/r8" > "$T/out11b" 2>&1
check "release arrives"      "$(cat "$T/r8/a.txt")" "dev-only"
check "release moved it"     "$REPO_MOVED" "1"

# ── 12. A checkout that is past the release is left alone, and says so ────────
push_upstream r8 a.txt newer
git -C "$T/r8" fetch -q; git -C "$T/r8" merge -q --ff-only origin/master
update_repo r8 "$T/r8" > "$T/out12" 2>&1
check "ahead is kept"        "$(cat "$T/r8/a.txt")" "newer"
check "and it says so"       "$(grep -c 'ahead of the stable release' "$T/out12")" "1"

# ── 13. Back on dev, it follows its own branch again ────────────────────────
git -C "$T/r8" remote set-head origin master >/dev/null 2>&1
push_upstream r8 a.txt newest
CHANNEL=dev
update_repo r8 "$T/r8" > "$T/out13" 2>&1
check "dev follows the branch" "$(cat "$T/r8/a.txt")" "newest"
check "upstream is the branch" "$(git -C "$T/r8" rev-parse --abbrev-ref '@{u}')" "origin/master"

# ── 13b. With a dev branch, the dev channel follows that, not the default branch
git -C "$T/r8.git" update-ref refs/heads/dev refs/heads/master
rm -rf "$T/push"; git clone -q "$T/r8.git" "$T/push"; ( cd "$T/push" && git checkout -q dev && echo on-dev > a.txt \
  && git commit -qam "dev work" && git push -q origin dev )
update_repo r8 "$T/r8" > "$T/out13b" 2>&1
check "dev branch followed"  "$(cat "$T/r8/a.txt")" "on-dev"
check "upstream is dev"      "$(git -C "$T/r8" rev-parse --abbrev-ref '@{u}')" "origin/dev"

# ── 14. A repository with no stable branch is followed as before ─────────────
new_pair r9
push_upstream r9 a.txt two
CHANNEL=stable
update_repo r9 "$T/r9" > "$T/out14" 2>&1
check "no stable branch: ff" "$(cat "$T/r9/a.txt")" "two"

# ── 15. The channel is read from the environment, then the settings, else stable
check "default channel"  "$(AGENTY_UPDATE_CHANNEL='' PROJECT_ROOT="$T" update_channel)" "stable"
check "env says dev"     "$(AGENTY_UPDATE_CHANNEL=dev PROJECT_ROOT="$T" update_channel)" "dev"
check "nonsense is stable" "$(AGENTY_UPDATE_CHANNEL=nightly PROJECT_ROOT="$T" update_channel)" "stable"

cd /; rm -rf "$T" "$FNS"
echo "update_repo: $ok passed, $bad failed"
[ "$bad" -eq 0 ]
