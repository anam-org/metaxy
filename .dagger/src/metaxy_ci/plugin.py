"""Validating the Claude Code plugin."""

import json
import posixpath
from pathlib import Path

from dagger_clients.core import Directory, ExistsType, Workspace, core

from metaxy_ci.containers import pinned_image

# Runs `claude plugin validate`.
_NODE_IMAGE = pinned_image("node")

# package.json and package-lock.json that pin the Claude Code CLI; Dependabot bumps them.
_CLAUDE_CODE = Path(__file__).resolve().parent / "images" / "claude-code"


async def _check_plugin_paths(plugin: Directory) -> None:
    """Fail unless the skills and MCP server paths in plugin.json hold what Claude Code loads."""
    manifest = json.loads(await plugin.file(".claude-plugin/plugin.json").contents())
    if skills := manifest.get("skills"):
        skills_path = posixpath.normpath(skills)
        names = [entry.rstrip("/") for entry in await plugin.entries(path=skills_path)]
        skill_dirs = [
            name
            for name in names
            if await plugin.exists(f"{skills_path}/{name}", expected_type=ExistsType.DIRECTORY_TYPE)
        ]
        if not skill_dirs:
            raise ValueError(f"No skills found in {skills}")
        missing = [name for name in skill_dirs if not await plugin.exists(f"{skills_path}/{name}/SKILL.md")]
        if missing:
            raise ValueError(f"Skills without a SKILL.md in {skills}: {', '.join(missing)}")
    if mcp_servers := manifest.get("mcpServers"):
        json.loads(await plugin.file(posixpath.normpath(mcp_servers)).contents())


async def validate(ws: Workspace) -> None:
    """Validate the plugin manifest with `claude plugin validate` and the paths it references."""
    plugin = ws.directory("/", include=[".claude-plugin/**"])
    await (
        core()
        .container()
        .from_(_NODE_IMAGE)
        .with_workdir("/opt/claude-code")
        .with_new_file("package.json", (_CLAUDE_CODE / "package.json").read_text())
        .with_new_file("package-lock.json", (_CLAUDE_CODE / "package-lock.json").read_text())
        .with_exec(["npm", "ci", "--no-audit", "--no-fund"])
        .with_env_variable("PATH", "/opt/claude-code/node_modules/.bin:${PATH}", expand=True)
        .with_directory("/src", plugin)
        .with_workdir("/src")
        .with_exec(["claude", "plugin", "validate", "."])
        .sync()
    )
    await _check_plugin_paths(plugin)
