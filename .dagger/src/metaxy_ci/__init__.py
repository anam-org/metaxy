import tomllib
from typing import Annotated

from dagger import Doc, check, collection, field, function, get, keys, object_type
from dagger_clients.core import Container, Workspace, core
from dagger_clients.uv import uv

PythonVersion = Annotated[str, Doc("Python version to install, e.g. 3.13.")]
Extras = Annotated[list[str] | None, Doc("Optional dependency extras to install, e.g. duckdb.")]
Groups = Annotated[list[str] | None, Doc("Dependency groups to install, e.g. dev.")]
Overrides = Annotated[
    list[str] | None,
    Doc("Requirements that replace the matching locked requirements, e.g. ibis-framework==11.*."),
]
AptPackages = Annotated[list[str] | None, Doc("Debian packages to install, e.g. libpq5.")]

# Files that do not affect tests, so editing them does not rerun the suites.
# Suites get what they need from tests.toml as environment variables.
_SOURCE_EXCLUDE = [".dagger", "dagger.toml", "dagger.lock", "tests.toml"]

# Not slim: `uv lock` needs git for the git sources in pyproject.toml.
_UV_IMAGE = "ghcr.io/astral-sh/uv:0.11.26-python3.13-trixie"

# Sets `[tool.uv] override-dependencies` in ./pyproject.toml to the arguments.
_SET_OVERRIDES = """
import sys
import tomlkit
with open("pyproject.toml") as f:
    doc = tomlkit.load(f)
doc["tool"]["uv"]["override-dependencies"] = sys.argv[1:]
with open("pyproject.toml", "w") as f:
    tomlkit.dump(doc, f)
"""


def _with_overrides(ws: Workspace, overrides: list[str]) -> Workspace:
    """A copy of `ws` locked with `overrides` replacing the matching requirements."""
    manifests = ws.directory("/", include=["**/pyproject.toml", "uv.lock"], exclude=_SOURCE_EXCLUDE, gitignore=True)
    locked = (
        core()
        .container()
        .from_(_UV_IMAGE)
        .with_env_variable("UV_CACHE_DIR", "/mnt/cache/uv")
        .with_mounted_cache("/mnt/cache/uv", core().cache_volume("uv-cache"))
        .with_directory("/src", manifests)
        .with_workdir("/src")
        .with_exec(["uv", "run", "--no-project", "--with", "tomlkit", "python", "-c", _SET_OVERRIDES, *overrides])
        .with_exec(["uv", "lock"])
        .directory("/src")
    )
    # `as_workspace` reads dagger.toml, so only the module source is left out.
    return (
        ws.directory("/", exclude=[".dagger"], gitignore=True)
        .with_file("pyproject.toml", locked.file("pyproject.toml"))
        .with_file("uv.lock", locked.file("uv.lock"))
        .as_workspace()
    )


def _image(
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
    )
    if apt_packages:
        base = base.with_exec(
            ["sh", "-c", 'apt-get update && apt-get install -y --no-install-recommends "$@"', "-", *apt_packages]
        )
    return (
        uv(_with_overrides(ws, overrides) if overrides else ws)
        .workspace(".")
        .builder(base_container=base)
        .with_python_pin(python_version)
        .with_extras(extras or [])
        .with_groups(groups or [])
        .without_default_groups()
        .with_dependencies()
        .container()
    )


@object_type
class TestSuite:
    """A pytest run in an image with the suite's dependencies installed."""

    name: Annotated[str, Doc("Suite name in tests.toml.")] = field()
    container: Annotated[
        Container, Doc("Image with the dependencies, the source tree and the service images to test against.")
    ] = field()
    pytest_args: Annotated[list[str], Doc("Arguments passed to pytest.")] = field()

    @check
    @function
    async def run(self) -> None:
        """Run the suite's tests.

        Tests get nested Dagger access to start the services they need.
        """
        await self.container.with_exec(
            ["python", "-m", "pytest", *self.pytest_args], experimental_privileged_nesting=True
        ).sync()


@object_type
@collection
class TestSuites:
    """The test suites in tests.toml, keyed by name."""

    names: Annotated[list[str], Doc("Suite names.")] = keys()

    # Not `field()`s: they are only read when `suite` builds a suite.
    ws: Workspace | None = None
    config: str = ""
    python_version: str = "3.11"

    @function
    @get
    def suite(self, name: Annotated[str, Doc("Suite name in tests.toml.")]) -> TestSuite:
        """The suite called `name`."""
        assert self.ws is not None
        config = tomllib.loads(self.config)
        spec = config["suites"][name]
        ctr = _image(
            self.ws,
            self.python_version,
            spec.get("extras"),
            spec.get("groups", ["dev"]),
            spec.get("overrides"),
            config.get("apt-packages"),
        ).with_directory(".", self.ws.directory("/", exclude=_SOURCE_EXCLUDE, gitignore=True))
        for service, image in {**config["services"], **spec.get("services", {})}.items():
            ctr = ctr.with_env_variable(f"METAXY_TEST_{service.upper()}_IMAGE", image)
        return TestSuite(name=name, container=ctr, pytest_args=spec["pytest-args"])


@object_type
class MetaxyCi:
    """CI pipelines for Metaxy."""

    ws: Workspace

    @function
    def image(
        self,
        python_version: PythonVersion = "3.11",
        extras: Extras = None,
        groups: Groups = None,
        overrides: Overrides = None,
        apt_packages: AptPackages = None,
    ) -> Container:
        """A container with Metaxy and the selected extras and dependency groups installed.

        Default dependency groups are installed only when listed in `groups`.
        """
        return _image(self.ws, python_version, extras, groups, overrides, apt_packages)

    @function
    async def tests(self, python_version: PythonVersion = "3.11") -> TestSuites:
        """The test suites defined in tests.toml."""
        config = await self.ws.file("tests.toml").contents()
        return TestSuites(
            names=list(tomllib.loads(config)["suites"]), ws=self.ws, config=config, python_version=python_version
        )
