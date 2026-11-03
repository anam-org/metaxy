"""The `metaxy-ci` module's entrypoint."""

from typing import Annotated

from dagger import Doc, check, function, object_type
from dagger_clients.core import Container, Secret, Workspace

from metaxy_ci import docs, plugin
from metaxy_ci.args import AptPackages, Extras, Groups, Overrides, PythonVersion
from metaxy_ci.containers import dev_container, image
from metaxy_ci.tests import ExtraTestSuites, TestSuites, suite_names


@object_type
class MetaxyCi:
    """CI pipelines for Metaxy."""

    ws: Workspace
    motherduck_token: Annotated[
        Secret | None, Doc("MotherDuck token for the MotherDuck tests, which are skipped without it.")
    ] = None
    codecov_token: Annotated[
        Secret | None, Doc("Codecov token. With it, the test suites upload coverage to Codecov on the main branch.")
    ] = None
    python_version: Annotated[str, Doc("Python version the test suites run on, e.g. 3.13.")] = "3.11"

    @function
    async def image(
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
        return await image(self.ws, python_version, extras, groups, overrides, apt_packages)

    @function
    async def tests(self) -> TestSuites:
        """The test suites in tests.toml that `dagger check` runs."""
        config = await self.ws.file("tests.toml").contents()
        return TestSuites(
            names=suite_names(config, extra=False),
            ws=self.ws,
            config=config,
            python_version=self.python_version,
            motherduck_token=self.motherduck_token,
            codecov_token=self.codecov_token,
        )

    @function
    async def extra_tests(self) -> ExtraTestSuites:
        """The heavy and compat test suites in tests.toml, which `dagger check` leaves out."""
        config = await self.ws.file("tests.toml").contents()
        return ExtraTestSuites(
            names=suite_names(config, extra=True),
            ws=self.ws,
            config=config,
            python_version=self.python_version,
            motherduck_token=self.motherduck_token,
            codecov_token=self.codecov_token,
        )

    @check
    @function
    async def ty(self) -> None:
        """Type check with ty."""
        await (await dev_container(self.ws)).with_exec(["ty", "check"]).sync()

    @check
    @function
    async def tach(self) -> None:
        """Check module boundaries with tach."""
        await (await dev_container(self.ws)).with_exec(["tach", "check"]).sync()

    @check
    @function
    async def deadcode(self) -> None:
        """Find unused code in src with deadcode."""
        await (await dev_container(self.ws)).with_exec(["deadcode", "src"]).sync()

    @check
    @function
    async def dprint(self) -> None:
        """Check TOML and JSON formatting with dprint."""
        # Every file, so dprint also checks the files the suites leave out.
        await (
            (await dev_container(self.ws))
            .with_directory(".", self.ws.directory("/", exclude=[".git"], gitignore=True))
            .with_exec(["dprint", "check"])
            .sync()
        )

    @function
    async def docs_container(self) -> Container:
        """The container that builds the documentation, with the source tree."""
        return await docs.docs_container(self.ws)

    @check
    @function
    async def docs(self) -> None:
        """Build the documentation with `mkdocs build --strict`."""
        await (await docs.docs_container(self.ws)).with_exec(["mkdocs", "build", "--strict"]).sync()

    @check
    @function
    async def docs_snippets(self) -> None:
        """Test the code blocks in the documentation."""
        await (await docs.docs_container(self.ws)).with_exec(["python", "-m", "pytest", "./docs"]).sync()

    @check
    @function
    async def claude_plugin(self) -> None:
        """Validate the Claude Code plugin manifest and the files it references."""
        await plugin.validate(self.ws)

    @function(cache="never")
    async def deploy_docs(
        self,
        version: Annotated[str, Doc("Documentation version, e.g. v1.0.0 or main.")],
        github_token: Annotated[Secret, Doc("GitHub token that can push to the gh-pages branch.")],
        alias: Annotated[str, Doc("Alias to point at this version, e.g. latest.")] = "",
        default: Annotated[str, Doc("Version or alias that the docs root redirects to, e.g. stable.")] = "",
        prerelease: Annotated[bool, Doc("Show the unstable version banner.")] = False,
    ) -> None:
        """Build the documentation and push it to the gh-pages branch with mike."""
        await docs.deploy(self.ws, version, github_token, alias, default, prerelease)
