"""Containers with Metaxy and its dependencies installed."""

import asyncio
import tomllib
from pathlib import Path

import tomlkit
from dagger.telemetry import get_tracer
from dagger_clients.core import Container, Directory, Workspace, core
from dagger_clients.uv import uv

# Files that do not affect tests, so editing them does not rerun the suites.
# Suites get what they need from tests.toml as environment variables.
# .git changes with every commit and git operation; what a step needs from it is passed in explicitly.
_SOURCE_EXCLUDE = [".git", ".dagger", "dagger.toml", "dagger.lock", "tests.toml"]


def pinned_image(name: str) -> str:
    """The image reference in this package's `images/<name>/Dockerfile`, which Dependabot keeps current."""
    dockerfile = Path(__file__).resolve().parent / "images" / name / "Dockerfile"
    for line in dockerfile.read_text().splitlines():
        if line.startswith("FROM "):
            return line.split()[1]
    raise ValueError(f"no FROM line in {dockerfile}")


# Not slim: `uv lock` needs git for the git sources in pyproject.toml.
_UV_IMAGE = pinned_image("uv")


async def _with_overrides(ws: Workspace, overrides: list[str]) -> Workspace:
    """A copy of `ws` locked with `overrides` replacing the matching requirements."""
    pyproject = tomlkit.parse(await ws.file("pyproject.toml").contents())
    pyproject["tool"]["uv"]["override-dependencies"] = overrides
    manifests = ws.directory(
        "/", include=["**/pyproject.toml", "uv.lock"], exclude=_SOURCE_EXCLUDE, gitignore=True
    ).with_new_file("pyproject.toml", tomlkit.dumps(pyproject))
    locked = (
        core()
        .container()
        .from_(_UV_IMAGE)
        .with_env_variable("UV_CACHE_DIR", "/mnt/cache/uv")
        .with_mounted_cache("/mnt/cache/uv", core().cache_volume("uv-cache"))
        .with_directory("/src", manifests)
        .with_workdir("/src")
        .with_exec(["uv", "lock"])
        .directory("/src")
    )
    with get_tracer().start_as_current_span(f"lock with overrides: {' '.join(overrides)}") as span:
        span.set_attribute("uv.overrides", overrides)
        locked = await locked.sync()
    # `as_workspace` reads dagger.toml, so only the module source is left out.
    return (
        ws.directory("/", exclude=[".git", ".dagger"], gitignore=True)
        .with_file("pyproject.toml", locked.file("pyproject.toml"))
        .with_file("uv.lock", locked.file("uv.lock"))
        .as_workspace()
    )


def with_apt_packages(ctr: Container, packages: list[str]) -> Container:
    return ctr.with_exec(
        ["sh", "-c", 'apt-get update && apt-get install -y --no-install-recommends "$@"', "-", *packages]
    )


async def image(
    ws: Workspace,
    python_version: str,
    extras: list[str] | None,
    groups: list[str] | None,
    overrides: list[str] | None,
    apt_packages: list[str] | None,
) -> Container:
    base = (
        core()
        .container()
        .from_(f"python:{python_version}-slim")
        .with_env_variable("UV_PROJECT_ENVIRONMENT", "/usr/local")
        .with_workdir("/src")
    )
    if apt_packages:
        base = with_apt_packages(base, apt_packages)
    return (
        uv(await _with_overrides(ws, overrides) if overrides else ws)
        .workspace(".")
        .builder(base_container=base)
        .with_python_pin(python_version)
        .with_extras(extras or [])
        .with_groups(groups or [])
        .without_default_groups()
        .with_dependencies()
        .container()
    )


def source(ws: Workspace) -> Directory:
    return ws.directory("/", exclude=_SOURCE_EXCLUDE, gitignore=True)


async def dev_image(ws: Workspace) -> Container:
    """Metaxy with every extra and dependency group installed."""
    pyproject, tests = await asyncio.gather(ws.file("pyproject.toml").contents(), ws.file("tests.toml").contents())
    pyproject_config = tomllib.loads(pyproject)
    return await image(
        ws,
        "3.11",
        list(pyproject_config["project"]["optional-dependencies"]),
        list(pyproject_config["dependency-groups"]),
        None,
        # The same packages as the test suites, which include git for the git sources in pyproject.toml.
        tomllib.loads(tests).get("apt-packages"),
    )


async def dev_container(ws: Workspace) -> Container:
    """Metaxy with every extra and dependency group installed, and the source tree."""
    return (await dev_image(ws)).with_directory(".", source(ws))
