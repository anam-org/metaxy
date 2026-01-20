---
title: "MCP Server"
description: "MCP server for AI assistants"
---

# MCP Server

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server exposes Metaxy's feature graph and metadata store operations to AI assistants, enabling them to explore your feature definitions and query metadata.

## Installation

Install Metaxy with the `mcp` extra:

=== "uv"

    ```bash
    uv add metaxy[mcp]
    ```

=== "pip"

    ```bash
    pip install metaxy[mcp]
    ```

## Running the Server

Run the MCP server from your Metaxy project directory:

```bash
metaxy mcp # (1)!
```

1. Use `uv run metaxy mcp` to run the server within the project's Python environment.

The server uses the standard Metaxy configuration discovery, loading `metaxy.toml` from the current directory or parent directories.

## Configuration

### Claude Code

Add the MCP server to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "metaxy": {
      "command": "metaxy",
      "args": ["mcp"]
    }
  }
}
```

## Available Tools

The MCP server provides the following tools:

::: metaxy-mcp-tools
