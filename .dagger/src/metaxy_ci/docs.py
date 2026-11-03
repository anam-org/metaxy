"""Building and deploying the documentation."""

import time

from dagger_clients.core import Container, Secret, Workspace, core

from metaxy_ci.containers import dev_image, pinned_image, source, with_apt_packages
from metaxy_ci.git import UPSTREAM

# ca-certificates: mkdocstrings fetches other projects' inventories over HTTPS.
# libcairo2: the social plugin renders cards with cairosvg.
# The rest: the Chromium that Slidev exports the slides with, as `playwright install-deps` lists them.
_DOCS_APT_PACKAGES = [
    "ca-certificates",
    "libcairo2",
    "libasound2t64",
    "libatk1.0-0t64",
    "libatspi2.0-0t64",
    "libdbus-1-3",
    "libgbm1",
    "libglib2.0-0t64",
    "libnspr4",
    "libnss3",
    "libxcomposite1",
    "libxdamage1",
    "libxfixes3",
    "libxkbcommon0",
    "libxrandr2",
]

# Builds the Slidev slides that the docs embed.
_BUN_IMAGE = pinned_image("bun")


async def docs_container(ws: Workspace) -> Container:
    """A container that builds the documentation, with the source tree."""
    return (
        with_apt_packages(await dev_image(ws), _DOCS_APT_PACKAGES)
        .with_file("/usr/local/bin/bun", core().container().from_(_BUN_IMAGE).file("/usr/local/bin/bun"))
        # Kept across builds, so `bun install` and the Chromium download run once.
        .with_mounted_cache("/root/.bun/install/cache", core().cache_volume("bun-install-cache"))
        .with_mounted_cache("/root/.cache/ms-playwright", core().cache_volume("ms-playwright"))
        # Silence the Material for MkDocs "MkDocs 2.0" upgrade notice.
        .with_env_variable("NO_MKDOCS_2_WARNING", "1")
        .with_env_variable("METAXY_SYNC_DOCS_MOUNTS", "1")
        .with_directory(".", source(ws))
    )


async def deploy(ws: Workspace, version: str, github_token: Secret, alias: str, default: str, prerelease: bool) -> None:
    """Build the documentation and push it to the gh-pages branch with mike."""
    ctr = (await docs_container(ws)).with_directory(".git", ws.git().directory())
    if prerelease:
        ctr = ctr.with_env_variable("DOCS_PRERELEASE", "true")
    ctr = (
        ctr.with_secret_variable("GITHUB_TOKEN", github_token)
        .with_exec(["git", "config", "--global", "user.name", "github-actions[bot]"])
        .with_exec(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"])
        # Reads the token from the environment at push time, so it is never written to disk.
        .with_exec(
            [
                "git",
                "config",
                "--global",
                "credential.helper",
                '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f',
            ]
        )
        .with_exec(
            ["sh", "-c", 'git remote set-url origin "$1" 2>/dev/null || git remote add origin "$1"', "-", UPSTREAM]
        )
        # A new value on every call, so gh-pages is always fetched and pushed again.
        .with_env_variable("DEPLOY_STARTED_AT", str(time.time_ns()))
        .with_exec(["sh", "-c", "git fetch origin gh-pages:gh-pages || true"])
        .with_exec(
            [
                "mike",
                "deploy",
                "--push",
                "--ignore-remote-status",
                *(["--update-aliases", version, alias] if alias else [version]),
            ]
        )
    )
    if default:
        ctr = ctr.with_exec(["mike", "set-default", "--push", default])
    await ctr.sync()
