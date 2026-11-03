"""Argument types shared by the module's functions."""

from typing import Annotated

from dagger import Doc

PythonVersion = Annotated[str, Doc("Python version to install, e.g. 3.13.")]
Extras = Annotated[list[str] | None, Doc("Optional dependency extras to install, e.g. duckdb.")]
Groups = Annotated[list[str] | None, Doc("Dependency groups to install, e.g. dev.")]
Overrides = Annotated[
    list[str] | None,
    Doc("Requirements that replace the matching locked requirements, e.g. ibis-framework==11.*."),
]
AptPackages = Annotated[list[str] | None, Doc("Debian packages to install, e.g. libpq5.")]
