"""CLI application context for sharing state across commands."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaxy.config import MetaxyConfig
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureGraph


# Context variable for storing the app context
_app_context: ContextVar["AppContext | None"] = ContextVar("_app_context", default=None)


@dataclass
class AppContext:
    """CLI application context.

    Stores the config initialized by the meta app launcher.
    """

    config: "MetaxyConfig"
    cli_project: (
        str | None
    )  # some CLI commands can be executed with a project different from the one in the Metaxy config
    all_projects: bool = False  # some CLI commands can work with all projects

    @classmethod
    def set(
        cls, config: "MetaxyConfig", cli_project: str | None, all_projects: bool = False
    ) -> None:
        """Initialize the app context.

        Should be called at CLI startup.

        Args:
            config: Metaxy configuration
            cli_project: CLI project override
            all_projects: Whether to include all projects
        """
        if _app_context.get() is not None:
            raise RuntimeError(
                "AppContext already initialized. It is not allowed to call AppContext.set() again."
            )
        else:
            from metaxy import load_features
            from metaxy.config import MetaxyConfig

            MetaxyConfig.set(config)
            load_features()
            _app_context.set(AppContext(config, cli_project, all_projects))

    @classmethod
    def get(cls) -> "AppContext":
        """Get the app context.

        Returns:
            AppContext instance

        Raises:
            RuntimeError: If context not initialized
        """
        ctx = _app_context.get()
        if ctx is None:
            raise RuntimeError(
                "CLI context not initialized. AppContext.set(config) should be called at CLI startup."
            )
        else:
            return ctx

    def reset(self) -> None:
        """Reset the app context.

        Raises:
            RuntimeError: If context not initialized
        """
        ctx = _app_context.get()
        if ctx is None:
            raise RuntimeError(
                "CLI context not initialized. AppContext.set(config) should be called at CLI startup."
            )
        else:
            _app_context.set(None)

    def get_store(self, name: str | None = None) -> "MetadataStore":
        """Get and open a metadata store from config.

        Store is retrieved from config context.

        Returns:
            Opened metadata store instance (within context manager)

        Raises:
            RuntimeError: If context not initialized
        """
        return self.config.get_store(name)

    @cached_property
    def graph(self) -> "FeatureGraph":
        """Get the graph instance.

        Returns:
            Graph instance

        Raises:
            RuntimeError: If context not initialized
        """
        from metaxy.models.feature import FeatureGraph

        return FeatureGraph.get_active()

    def raise_command_cannot_override_project(self) -> None:
        """Raise an error if the command cannot override project.

        Raises:
            SystemExit: If command cannot override project
        """
        if self.cli_project or self.all_projects:
            from metaxy.cli.console import console

            console.print(
                f"[red]Error:[/red] This command can only be used with the project from Metaxy configuration: {self.config.project}.",
                style="bold",
            )
            raise SystemExit(1)

    @cached_property
    def project(self) -> str | None:
        """Get the project for the app invocation, in order of precedence:
            - None if all projects are selected
            - project from CLI input
            - project from Metaxy configuration

        Returns:
            Project name or None if not set
        """
        return (
            (self.cli_project or self.config.project) if not self.all_projects else None
        )

    def get_required_project(self) -> str:
        """Get the project for commands that require a specific project.

        This method ensures we have a valid project string, raising an error
        if all_projects is True (which would make project None).

        Returns:
            Project name (never None)

        Raises:
            SystemExit: If all_projects is True (project would be None)
        """
        if self.all_projects:
            from metaxy.cli.console import console

            console.print(
                "[red]Error:[/red] This command requires a specific project. "
                "Cannot use --all-projects flag.",
                style="bold",
            )
            raise SystemExit(1)

        # Return the project (either from CLI or config, guaranteed to have a default)
        return self.cli_project or self.config.project
