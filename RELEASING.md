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

Regenerate the full `CHANGELOG.md`:

```bash
just changelog
```

## Pre-release

```bash
just release rc
```

## Stable release

```bash
just release stable
```

The `release` recipe bumps the version, updates `_version.py`, and regenerates `CHANGELOG.md`.

Then create a Release and the corresponding tag from GitHub's UI. The GitHub Release body is automatically populated by CI using git-cliff.
