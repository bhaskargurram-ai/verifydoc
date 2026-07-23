# Security Policy

## Supported versions

VerifyDoc is pre-1.0; security fixes are applied to the latest released version
on PyPI and `main`. Please always test against the newest release.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's [private vulnerability reporting](https://github.com/bhaskargurram-ai/verifydoc/security/advisories/new)
("Report a vulnerability" under the repo's **Security** tab). Include a
description, affected version, and a minimal reproduction if possible.

We aim to acknowledge reports within **72 hours** and to ship a fix or
mitigation for confirmed issues within **30 days**, coordinating disclosure
with you.

## Scope & handling notes

- **Secrets never leave your machine.** VerifyDoc reads API keys (e.g.
  `ANTHROPIC_API_KEY`, bot tokens) only from environment variables / a
  git-ignored `.env.secret`; none are logged or committed. The server's inbound
  webhooks are signature-verified.
- **Untrusted documents** are processed locally by the configured adapter. When
  using a hosted API adapter (`api-vlm`) or the WhatsApp Cloud API, document
  content is sent to that third party by design — prefer a local adapter
  (`text-search`, `rapidocr`, a local HF VLM) for sensitive data.
- Dependency and code-scanning alerts are handled via Dependabot and CodeQL.
