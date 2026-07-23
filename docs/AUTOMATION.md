# Repo automation

Everything here runs on GitHub's free tier. **No paid API key is required.**

| Workflow / bot | What it does | Cost | Needs a key? |
|---|---|---|---|
| **CI** (`ci.yml`) | ruff + black + mypy + pytest on 3.11/3.12, eval-coverage gate, packaging build | free Actions | no |
| **Release** (`release.yml`) | build + GitHub release + PyPI publish on tags | free | PyPI trusted publishing (no secret) |
| **Dependabot** | weekly pip + actions dependency PRs | free | no |
| **CodeQL** | security scanning | free | no |
| **PR Labeler** | labels PRs by touched paths | free | no |
| **Stale triage** | flags/closes inactive issues & PRs | free | no |
| **PR review (free)** (`pr-review-free.yml`) | AI review of each PR diff via **GitHub Models** | **free** (built-in `GITHUB_TOKEN`, rate-limited) | **no** |
| **Claude review** (`claude-review.yml`) | fork-safe Claude review of the diff; sets `claude-approved` / `claude-changes-requested` | paid | `ANTHROPIC_API_KEY` secret |
| **Auto-merge** (`auto-merge.yml`) | squash-merges Dependabot on green, and Claude-approved PRs that touch no CI/packaging/deps path | free | no |

## Free LLM for maintenance

The default AI reviewer uses **[GitHub Models](https://docs.github.com/en/github-models)**:
GitHub Actions can call hosted models (e.g. `openai/gpt-4o-mini`) for free using
the built-in `GITHUB_TOKEN` and a `models: read` permission — no secret, no
billing, on the free tier for public repos (subject to rate limits). It posts a
concise review comment on every PR and degrades gracefully (skips, never fails
the PR) if the rate limit is hit.

The paid **Claude review** is an optional upgrade: add an `ANTHROPIC_API_KEY`
secret (Settings → Secrets → Actions) and it activates; without it, it simply
skips and the free reviewer handles PRs. So you can run the whole project's
maintenance at zero LLM cost.

## Contribution pipeline & key safety

When `ANTHROPIC_API_KEY` is set, incoming PRs flow through:

1. **CI** (`ci.yml`) runs ruff/black/mypy/pytest on 3.11 + 3.12. These three
   checks (`test (3.11)`, `test (3.12)`, `build`) are **required** by branch
   protection on `main`, so nothing merges without them green.
2. **Claude review** reads the diff and posts findings, labelling the PR
   `claude-approved` or `claude-changes-requested`.
3. **Auto-merge** enables GitHub squash auto-merge (which waits for the required
   checks) for Dependabot, and for Claude-approved PRs that don't touch
   sensitive paths. PRs changing `.github/**`, `pyproject.toml`, `setup.*`,
   lockfiles, or `MANIFEST.in` get `needs-maintainer` and wait for a human.

**Why the key can't leak or be misused:**

- It lives only as an **encrypted GitHub Actions secret**, never in the repo
  (`.env.secret` is git-ignored) and never in a `run:` argument. GitHub also
  auto-redacts it from all logs.
- `claude-review.yml` runs on `pull_request_target` (so the secret is available
  for fork PRs) but **never checks out or executes PR-authored code** — it only
  fetches the diff *text* via `gh pr diff` and sends it to the API. A malicious
  fork has no code-execution surface in the job that holds the key.
- There is deliberately **no agentic path** (e.g. `claude-code-action` running a
  fork's build/tests) that could execute untrusted code with the key.
- Auto-merge never merges PRs that touch CI/packaging/dependency files, so the
  supply-chain and secret-config surfaces always require a human.
- The LLM review is advisory, **not** a security boundary; CI and the
  sensitive-path block are the enforced gates.
