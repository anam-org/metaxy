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

## Creating a release

```bash
just release rc                                    # pre-release
just release stable                                # stable release
just release stable message="Adds PostgreSQL support."  # with a release summary
```

This bumps the version and generates the changelog. Commit and tag manually after reviewing the changes. The optional `message` parameter adds a release summary above the commit groups in both `CHANGELOG.md` and the GitHub Release notes.

## Publishing

Push the tag to trigger the release workflow:

```bash
git push origin --follow-tags
```

The CI pipeline runs QA, publishes to PyPI, creates a GitHub Release, and deploys docs (stable releases only).
