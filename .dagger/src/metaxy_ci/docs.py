"""Building and deploying the documentation."""

import time

from dagger.telemetry import get_tracer
from dagger_clients.core import Container, Secret, Workspace, core

from metaxy_ci.containers import dev_image, pinned_image, source, with_apt_packages
from metaxy_ci.git import UPSTREAM

# ca-certificates: mkdocstrings fetches other projects' inventories over HTTPS.
# The rest: what the Chromium that Slidev exports the slides with needs, from
# Playwright's native dependency table for Debian 13 (the same for x64 and arm64).
# It includes libcairo2, which the social plugin also renders cards with.
_DOCS_APT_PACKAGES = [
    "ca-certificates",
    "libasound2t64",
    "libatk-bridge2.0-0t64",
    "libatk1.0-0t64",
    "libatspi2.0-0t64",
    "libcairo2",
    "libcups2t64",
    "libdbus-1-3",
    "libdrm2",
    "libgbm1",
    "libglib2.0-0t64",
    "libnspr4",
    "libnss3",
    "libpango-1.0-0",
    "libx11-6",
    "libxcb1",
    "libxcomposite1",
    "libxdamage1",
    "libxext6",
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
    tracer = get_tracer()
    with tracer.start_as_current_span("build the documentation container"):
        ctr = await (await docs_container(ws)).with_directory(".git", ws.git().directory()).sync()
    if prerelease:
        ctr = ctr.with_env_variable("DOCS_PRERELEASE", "true")
    with tracer.start_as_current_span("fetch gh-pages"):
        ctr = await (
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
            .sync()
        )
    with tracer.start_as_current_span("deploy the documentation with mike"):
        ctr = await ctr.with_exec(
            [
                "mike",
                "deploy",
                "--push",
                "--ignore-remote-status",
                *(["--update-aliases", version, alias] if alias else [version]),
            ]
        ).sync()
    if default:
        with tracer.start_as_current_span("set the default version"):
            await ctr.with_exec(["mike", "set-default", "--push", default]).sync()
