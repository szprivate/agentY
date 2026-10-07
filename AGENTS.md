# Rules for coding agents

These apply to every repository of the agentY stack: **agentY**, **agentY-core**,
**agentY-comfyuiConnect** and **agentY-mcp**. The same file is in each of them.

## Branches

| branch | what it is | who moves it |
|---|---|---|
| `dev` | where all new work goes first | you, by committing and pushing |
| `main` (`master` in agentY-core) | the latest release | only a release |
| `stable` | the latest release; what installed machines follow | only a release |

1. **Commit and push new work to `dev`.** Never commit or push directly to
   `main`, `master` or `stable`.
2. **Do not create other long-lived branches.** A short-lived branch for one
   piece of work is fine; merge it into `dev` and delete it.
3. **A change is not released because it is pushed.** It reaches other machines
   only with the next release. When you finish work, say so, and say that it is
   on `dev`.

## Releases

4. **Moving tested work from `dev` to `main` / `stable` is a release** of the
   whole system, unless the user says otherwise. Do it only when the user asks
   for it, and only after the tests have been run.
5. **A release covers all four repositories under one version number**, also
   the ones that did not change, so version numbers stay the same across the
   stack. Never tag or release one repository by itself.
6. **Use the release script**, from the agentY checkout, with every repository on
   its `dev` branch and pushed:

   ```
   .venv/Scripts/python.exe scripts/make_release.py <x.y.z> --dry-run
   .venv/Scripts/python.exe scripts/make_release.py <x.y.z>
   ```

   It writes `release.toml` and `requirements.lock`, tags every repository
   `v<x.y.z>`, moves `main` / `master` and `stable` to the release, and creates
   the GitHub releases. Do not do those steps by hand.
7. **Pick the version number with the user** if they have not given one: a fix is
   a patch (1.0.**1**), new features a minor (1.**1**.0), an incompatible change
   a major (**2**.0.0).

## Names

- The tool layer's repository and folder are `agentY-core`. The Python package
  inside it is `agenty_core` — that is not a mistake, and imports stay as they are.

More on channels and releases: `docs/reference.md` in agentY, section *Releases*.
