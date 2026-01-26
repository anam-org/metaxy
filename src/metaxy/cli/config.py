"""Configuration commands for Metaxy CLI."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

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


def _remove_none_values(obj: Any) -> Any:
    """Recursively remove None values from a dict (TOML doesn't support None)."""
    if isinstance(obj, dict):
        return {k: _remove_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_remove_none_values(item) for item in obj if item is not None]
    return obj


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
    import tomli_w
    from rich.syntax import Syntax

    from metaxy.cli.context import AppContext

    context = AppContext.get()
    config = context.config

    # Dump config to dict, converting nested models
    config_dict = config.model_dump(mode="json")

    if format == "json":
        data_console.print(json.dumps(config_dict, indent=2))
    else:
        # Remove None values (TOML doesn't support them)
        config_dict = _remove_none_values(config_dict)
        # Convert to TOML and display with syntax highlighting
        toml_str = tomli_w.dumps(config_dict)
        syntax = Syntax(toml_str, "toml", line_numbers=False)
        data_console.print(syntax)
