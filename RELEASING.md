# Releasing

## Prerequisites

Install [git-cliff](https://git-cliff.org/) for changelog generation:

```bash
brew install git-cliff
```

## Changelog

Preview unreleased changelog entries:

```bash
just changelog-preview
```

Regenerate the full `CHANGELOG.md` from scratch:

```bash
just changelog
```

## Creating a release

```bash
just release rc      # pre-release
just release stable  # stable release
```

The `release` recipe bumps the version, updates `_version.py`, and prepends the new release section to `CHANGELOG.md`.

## Release summary

To include a custom summary for a release, create an annotated tag:

```bash
git tag -a v0.2.0 -m "This release does X and Y."
```

The tag message appears above the commit groups in both `CHANGELOG.md` and the GitHub Release notes. With a lightweight tag, no summary is included.

## Publishing

Push the tag to trigger the release workflow:

```bash
git push origin v0.2.0
```

Or create a Release from GitHub's UI. The GitHub Release body is automatically populated by CI using git-cliff.
