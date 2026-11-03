"""The test suites defined in tests.toml."""

import asyncio
import tomllib
from typing import Annotated, Any, Self

from dagger import Doc, check, collection, field, function, get, keys, object_type
from dagger.telemetry import get_tracer
from dagger_clients.core import Container, Secret, Workspace

from metaxy_ci.containers import image, source
from metaxy_ci.git import base_git_dir, merge_base, tach_args

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


async def _run(
    ws: Workspace,
    suite: TestSuite,
    python_version: str,
    motherduck_token: Secret | None,
    codecov_token: Secret | None,
) -> None:
    """Run `suite`, skipping the tests unaffected by the changes since the merge base with `main`.

    On `main`, with `codecov_token`, every test runs under coverage and the report is uploaded to Codecov.
    Tests get nested Dagger access to start the services they need.
    """
    tracer = get_tracer()

    async def build() -> Container:
        with tracer.start_as_current_span("build the test image"):
            return await (await _container(ws, suite, python_version, motherduck_token)).sync()

    async def find_merge_base() -> tuple[str, str]:
        with tracer.start_as_current_span("find the merge base with main"):
            return await merge_base(ws)

    ctr, (base, head) = await asyncio.gather(build(), find_merge_base())
    tach = tach_args(base, head)
    if tach:
        # Last, so only the pytest run depends on the merge base. Intent-to-add
        # entries make the files that are new since the merge base part of the diff.
        ctr = ctr.with_mounted_directory(".git", base_git_dir(base)).with_exec(["git", "add", "--intent-to-add", "."])
    if codecov_token is not None and base == head:
        with tracer.start_as_current_span("run the tests under coverage"):
            ctr = await _with_coverage(ctr, suite).sync()
        with tracer.start_as_current_span("upload coverage to Codecov"):
            await _with_codecov_upload(ctr, suite, head, codecov_token).sync()
    else:
        with tracer.start_as_current_span("run the tests"):
            await ctr.with_exec(
                ["python", "-m", "pytest", *suite.pytest_args, *tach], experimental_privileged_nesting=True
            ).sync()


_CODECOV_CLI = "codecov-cli==11.3.2"
_CODECOV_SLUG = "anam-org/metaxy"


def _with_coverage(ctr: Container, suite: TestSuite) -> Container:
    """Run `suite` under coverage and write the report to coverage.xml."""
    return (
        # Absolute, so subprocesses started in other directories write their data here too.
        ctr.with_env_variable("COVERAGE_FILE", "/src/.coverage")
        .with_env_variable("COVERAGE_PROCESS_START", "/src/pyproject.toml")
        # Measures the subprocesses the tests start.
        .with_exec(
            [
                "sh",
                "-c",
                'cp .github/coverage_subprocess.pth "$(python -c \'import sysconfig; print(sysconfig.get_path("purelib"))\')"',
            ]
        )
        .with_exec(["coverage", "run", "-m", "pytest", *suite.pytest_args], experimental_privileged_nesting=True)
        .with_exec(["coverage", "combine"])
        .with_exec(["coverage", "xml", "-o", "coverage.xml"])
    )


def _with_codecov_upload(ctr: Container, suite: TestSuite, head: str, codecov_token: Secret) -> Container:
    """Upload coverage.xml to Codecov, flagged with the suite name."""
    return (
        ctr.with_secret_variable("CODECOV_TOKEN", codecov_token)
        # Cloud Checks have no CI environment for the Codecov CLI to read the commit from.
        .with_exec(
            [
                "sh",
                "-c",
                # The token is read from the secret at run time, so it is not part of the arguments.
                'exec uv tool run --from "$0" codecovcli upload-process -t "$CODECOV_TOKEN" "$@"',
                _CODECOV_CLI,
                "--disable-search",
                "--file",
                "coverage.xml",
                "--flag",
                suite.name,
                "--sha",
                head,
                "--branch",
                "main",
                "--slug",
                _CODECOV_SLUG,
                "--git-service",
                "github",
            ]
        )
    )


@object_type
class Suite:
    """A test suite that `dagger check` runs."""

    suite: Annotated[TestSuite, Doc("The suite.")] = field()

    # Not `field()`s: only `container` and `run` read them, so listing the
    # checks builds nothing.
    ws: Workspace | None = None
    python_version: str = "3.11"
    motherduck_token: Secret | None = None
    codecov_token: Secret | None = None

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
        await _run(self.ws, self.suite, self.python_version, self.motherduck_token, self.codecov_token)


@object_type
class ExtraSuite:
    """A heavy or compat test suite. It is not for a check: it runs only when `run` is called."""

    suite: Annotated[TestSuite, Doc("The suite.")] = field()

    # Not `field()`s: only `container` and `run` read them.
    ws: Workspace | None = None
    python_version: str = "3.11"
    motherduck_token: Secret | None = None
    codecov_token: Secret | None = None

    @function
    async def container(self) -> Container:
        """Image with the suite's dependencies, the source tree and the service images to test against."""
        assert self.ws is not None
        return await _container(self.ws, self.suite, self.python_version, self.motherduck_token)

    @function
    async def run(self) -> None:
        """Run the suite's tests."""
        assert self.ws is not None
        await _run(self.ws, self.suite, self.python_version, self.motherduck_token, self.codecov_token)


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
    codecov_token: Secret | None = None

    @function
    @get
    def suite(self, name: Annotated[str, Doc("Suite name in tests.toml.")]) -> Suite:
        """The suite called `name`."""
        return Suite(
            suite=TestSuite.from_config(tomllib.loads(self.config), name),
            ws=self.ws,
            python_version=self.python_version,
            motherduck_token=self.motherduck_token,
            codecov_token=self.codecov_token,
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
    codecov_token: Secret | None = None

    @function
    @get
    def suite(self, name: Annotated[str, Doc("Suite name in tests.toml.")]) -> ExtraSuite:
        """The suite called `name`."""
        return ExtraSuite(
            suite=TestSuite.from_config(tomllib.loads(self.config), name),
            ws=self.ws,
            python_version=self.python_version,
            motherduck_token=self.motherduck_token,
            codecov_token=self.codecov_token,
        )
