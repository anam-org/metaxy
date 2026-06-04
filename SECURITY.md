# Security Policy

Metaxy is a backend library for feature metadata management. It is embedded into
other applications and pipelines rather than run as a standalone service, and it
declares dependency version ranges rather than pinning exact versions. Keep this in
mind when assessing security reports.

## Scope

In scope:

- Vulnerabilities in Metaxy's own code — for example, unsafe handling of untrusted
  metadata or feature definitions, query construction that could enable injection,
  or unsafe deserialization.

Generally out of scope:

- Advisories reported by scanners against entries in `uv.lock`. The lockfile pins
  Metaxy's own development and CI environment; it does not constrain the versions
  that downstream projects resolve. Consumers control their own dependency versions
  through the ranges Metaxy declares in `pyproject.toml`.
- Security decisions that belong to the embedding application, such as how it
  authenticates, where it stores credentials, or which metadata store backend it
  exposes to untrusted input.

## Supported Versions

Fixes are released against the latest published version on PyPI. There is no
backporting to older releases. If a fix affects how Metaxy resolves its own
dependencies, it ships as an updated lockfile and, where relevant, an adjusted
version range in `pyproject.toml`.

## Reporting a Vulnerability

Please report security vulnerabilities privately. **Do not open a public issue,
pull request, or discussion for a security report.**

Use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/anam-org/metaxy/security) of the repository.
2. Click **Report a vulnerability** to open a private advisory.
3. Include affected version(s), a description of the impact, and steps to reproduce
   or a proof of concept.

We will acknowledge reports and work with you on a fix on a best-effort basis, and
credit you in the published advisory unless you prefer to remain anonymous. Please
give us a reasonable opportunity to release a fix before any public disclosure.
