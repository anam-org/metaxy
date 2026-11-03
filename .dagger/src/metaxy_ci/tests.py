"""The test suites defined in tests.toml."""

import asyncio
import tomllib
from typing import Annotated, Any, Self

from dagger import Doc, check, collection, field, function, get, keys, object_type
from dagger_clients.core import Container, Secret, Workspace

from metaxy_ci.containers import image, source
from metaxy_ci.git import tach_args

# The region of the MotherDuck account the MotherDuck tests use.
_MOTHERDUCK_REGION = "eu-central-1"

# Suite flags in tests.toml that leave a suite out of `dagger check`.
_EXTRA_FLAGS = ("heavy", "compat")


@object_type
class ServiceImage:
    """The image of a service that the test fixtures start."""

    name: Annotated[str, Doc("Service name in tests.toml, e.g. clickhouse.")] = field()
    image: Annotated[str, Doc("Image reference.")] = field()


@object_type
class TestSuite:
    """A test suite in tests.toml: what to install and which tests to run."""

    name: Annotated[str, Doc("Suite name in tests.toml.")] = field()
    extras: Annotated[list[str], Doc("Optional dependency extras to install.")] = field()
    groups: Annotated[list[str], Doc("Dependency groups to install.")] = field()
    overrides: Annotated[list[str], Doc("Requirements that replace the matching locked requirements.")] = field()
    apt_packages: Annotated[list[str], Doc("Debian packages to install.")] = field()
    services: Annotated[list[ServiceImage], Doc("Images of the services the tests start.")] = field()
    pytest_args: Annotated[list[str], Doc("Arguments passed to pytest.")] = field()

    @classmethod
    def from_config(cls, config: dict[str, Any], name: str) -> Self:
        """The suite called `name` in the parsed tests.toml `config`."""
        spec = config["suites"][name]
        return cls(
            name=name,
            extras=spec.get("extras", []),
            groups=spec.get("groups", ["dev"]),
            overrides=spec.get("overrides", []),
            apt_packages=config.get("apt-packages", []),
            services=[
                ServiceImage(name=service, image=service_image)
                for service, service_image in {**config["services"], **spec.get("services", {})}.items()
            ],
            pytest_args=spec["pytest-args"],
        )


def suite_names(config_toml: str, *, extra: bool) -> list[str]:
    return [
        name
        for name, spec in tomllib.loads(config_toml)["suites"].items()
        if any(spec.get(flag, False) for flag in _EXTRA_FLAGS) == extra
    ]


async def _container(
    ws: Workspace, suite: TestSuite, python_version: str, motherduck_token: Secret | None
) -> Container:
    ctr = (
        await image(ws, python_version, suite.extras, suite.groups, suite.overrides, suite.apt_packages)
    ).with_directory(".", source(ws))
    for service in suite.services:
        ctr = ctr.with_env_variable(f"METAXY_TEST_{service.name.upper()}_IMAGE", service.image)
    if motherduck_token is not None:
        ctr = ctr.with_secret_variable("MOTHERDUCK_TOKEN", motherduck_token).with_env_variable(
            "MOTHERDUCK_REGION", _MOTHERDUCK_REGION
        )
    return ctr


async def _run(ws: Workspace, suite: TestSuite, python_version: str, motherduck_token: Secret | None) -> None:
    """Run `suite`, skipping the tests unaffected by the changes since the merge base with `main`.

    Tests get nested Dagger access to start the services they need.
    """
    ctr, tach = await asyncio.gather(_container(ws, suite, python_version, motherduck_token), tach_args(ws))
    if tach:
        # Last, so only the pytest run depends on the commit.
        ctr = ctr.with_directory(".git", ws.git().directory())
    await ctr.with_exec(
        ["python", "-m", "pytest", *suite.pytest_args, *tach], experimental_privileged_nesting=True
    ).sync()


@object_type
class Suite:
    """A test suite that `dagger check` runs."""

    suite: Annotated[TestSuite, Doc("The suite.")] = field()

    # Not `field()`s: only `container` and `run` read them, so listing the
    # checks builds nothing.
    ws: Workspace | None = None
    python_version: str = "3.11"
    motherduck_token: Secret | None = None

    @function
    async def container(self) -> Container:
        """Image with the suite's dependencies, the source tree and the service images to test against."""
        assert self.ws is not None
        return await _container(self.ws, self.suite, self.python_version, self.motherduck_token)

    @check
    @function
    async def run(self) -> None:
        """Run the suite's tests."""
        assert self.ws is not None
        await _run(self.ws, self.suite, self.python_version, self.motherduck_token)


@object_type
class ExtraSuite:
    """A heavy or compat test suite. It is not for a check: it runs only when `run` is called."""

    suite: Annotated[TestSuite, Doc("The suite.")] = field()

    # Not `field()`s: only `container` and `run` read them.
    ws: Workspace | None = None
    python_version: str = "3.11"
    motherduck_token: Secret | None = None

    @function
    async def container(self) -> Container:
        """Image with the suite's dependencies, the source tree and the service images to test against."""
        assert self.ws is not None
        return await _container(self.ws, self.suite, self.python_version, self.motherduck_token)

    @function
    async def run(self) -> None:
        """Run the suite's tests."""
        assert self.ws is not None
        await _run(self.ws, self.suite, self.python_version, self.motherduck_token)


@object_type
@collection
class TestSuites:
    """The test suites in tests.toml that `dagger check` runs, keyed by name."""

    names: Annotated[list[str], Doc("Suite names.")] = keys()

    # Not `field()`s: they are only read when `suite` looks up a suite.
    ws: Workspace | None = None
    config: str = ""
    python_version: str = "3.11"
    motherduck_token: Secret | None = None

    @function
    @get
    def suite(self, name: Annotated[str, Doc("Suite name in tests.toml.")]) -> Suite:
        """The suite called `name`."""
        return Suite(
            suite=TestSuite.from_config(tomllib.loads(self.config), name),
            ws=self.ws,
            python_version=self.python_version,
            motherduck_token=self.motherduck_token,
        )


@object_type
@collection
class ExtraTestSuites:
    """The heavy and compat test suites in tests.toml, keyed by name."""

    names: Annotated[list[str], Doc("Suite names.")] = keys()

    # Not `field()`s: they are only read when `suite` looks up a suite.
    ws: Workspace | None = None
    config: str = ""
    python_version: str = "3.11"
    motherduck_token: Secret | None = None

    @function
    @get
    def suite(self, name: Annotated[str, Doc("Suite name in tests.toml.")]) -> ExtraSuite:
        """The suite called `name`."""
        return ExtraSuite(
            suite=TestSuite.from_config(tomllib.loads(self.config), name),
            ws=self.ws,
            python_version=self.python_version,
            motherduck_token=self.motherduck_token,
        )
