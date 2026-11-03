from typing import Annotated

from dagger import Doc, function, object_type
from dagger_clients.core import Container, Workspace, core
from dagger_clients.uv import uv


@object_type
class MetaxyCi:
    """CI pipelines for Metaxy."""

    ws: Workspace

    @function
    def image(
        self,
        python_version: Annotated[str, Doc("Python version to install, e.g. 3.13.")] = "3.11",
        extras: Annotated[list[str] | None, Doc("Optional dependency extras to install, e.g. duckdb.")] = None,
        groups: Annotated[list[str] | None, Doc("Dependency groups to install, e.g. dev.")] = None,
    ) -> Container:
        """A container with Metaxy and the selected extras and dependency groups installed.

        Default dependency groups are installed only when listed in `groups`.
        """
        base = (
            core()
            .container()
            .from_(f"python:{python_version}-slim")
            .with_env_variable("UV_PROJECT_ENVIRONMENT", "/usr/local")
        )
        return (
            uv(self.ws)
            .workspace(".")
            .builder(base_container=base)
            .with_python_pin(python_version)
            .with_extras(extras or [])
            .with_groups(groups or [])
            .without_default_groups()
            .with_dependencies()
            .container()
        )
