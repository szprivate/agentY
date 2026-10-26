"""Manage the people who may use a server install of agentY.

    .venv/Scripts/python.exe scripts/agenty_users.py add anna --slack U01ABCDEF --role approver
    .venv/Scripts/python.exe scripts/agenty_users.py list
    .venv/Scripts/python.exe scripts/agenty_users.py remove anna
    .venv/Scripts/python.exe scripts/agenty_users.py usage --by project --days 30

`add` prints the person's token ONCE. Only a digest is kept in config/users.json,
so a lost token is replaced (run `add` again for the same name), not recovered.
The token goes in `Authorization: Bearer <token>` (or `X-AgentY-Token`). It only
means anything when `security.user_tokens` is on - see docs/setup/server.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import team_store, user_auth  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="create a user, or issue a new token for one")
    add.add_argument("name")
    add.add_argument("--slack", default="", help="their Slack member id (U01ABCDEF)")
    add.add_argument("--role", action="append", default=None,
                     help="a role; 'approver' lets them approve other people's requests")
    sub.add_parser("list", help="show the users (never their tokens)")
    rm = sub.add_parser("remove", help="revoke a user")
    rm.add_argument("name")
    use = sub.add_parser("usage", help="what the team has spent, from the usage ledger")
    use.add_argument("--by", choices=("user", "project"), default="user")
    use.add_argument("--days", type=float, default=30)
    args = ap.parse_args(argv)

    if args.cmd == "add":
        token = user_auth.add_user(args.name, slack_id=args.slack, roles=args.role)
        print(f"user {args.name} saved in {user_auth.users_path()}")
        print("token (shown once - copy it now):")
        print(token)
        return 0
    if args.cmd == "list":
        users = user_auth.load()
        if not users:
            print("no users yet")
        for name, rec in sorted(users.items()):
            issued = time.strftime("%Y-%m-%d", time.localtime(rec.get("token_issued") or 0))
            print(f"{name:20} slack={rec.get('slack_id') or '-':12} "
                  f"roles={','.join(rec.get('roles') or []) or '-':12} token issued {issued}")
        return 0
    if args.cmd == "remove":
        ok = user_auth.remove_user(args.name)
        print("removed" if ok else "no such user")
        return 0 if ok else 1
    rows = team_store.usage_summary(args.by, since=time.time() - args.days * 86400)
    if not rows:
        print("nothing recorded")
    for r in rows:
        print(f"{r[args.by]:24} turns={r['turns']:4} in={r['input']:>10} out={r['output']:>9} "
              f"cost=${r['cost']:.2f} partner=${r['partner_cost']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
