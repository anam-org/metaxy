"""Configuration commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import cyclopts

from metaxy.cli.console import console, data_console, error_console

# Config subcommand app
app = cyclopts.App(
    name="config",
    help="Manage Metaxy configuration",
    console=console,
    error_console=error_console,
)

ConfigOutputFormat = Literal["toml", "json"]


@app.command(name="print")
def print_config(
    *,
    format: Annotated[
        ConfigOutputFormat,
        cyclopts.Parameter(
            name=["-f", "--format"],
            help="Output format: 'toml' (with syntax highlighting) or 'json'.",
        ),
    ] = "toml",
) -> None:
    """Print the current Metaxy configuration.

    Examples:
        $ metaxy config print
        $ metaxy config print --format json
    """
    from rich.syntax import Syntax

    from metaxy.cli.context import AppContext

    context = AppContext.get()
    config = context.config

    if format == "json":
        config_dict = config.model_dump(mode="json")
        data_console.print(json.dumps(config_dict, indent=2))
    else:
        toml_str = config.to_toml()
        syntax = Syntax(toml_str, "toml", line_numbers=False)
        data_console.print(syntax)
