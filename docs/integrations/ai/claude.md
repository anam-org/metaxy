---
title: Claude Code Plugin
description: Use Metaxy with Claude Code through the official plugin.
---

# Claude Code Plugin

A [Claude Code](https://claude.com/product/claude-code) plugin that provides additional context for working with Metaxy projects.

## Features

- **`/metaxy` skill**: Guidance on working with Metaxy, including feature definitions, versioning, and metadata stores
- **MCP tools**: Explore feature graphs and query metadata directly from Claude Code via the [MCP server](mcp.md)

## Installation

```
/plugin install anam-org/metaxy
```

## Requirements

`uv` must be installed. The plugin starts the MCP server via `uv run` to use the project's Python environment.
