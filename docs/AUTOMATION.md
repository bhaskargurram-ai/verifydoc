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
| **Claude review** (`claude-review.yml`) | higher-quality AI review + `@claude` replies | paid | optional `ANTHROPIC_API_KEY` |

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
