"""Entry point for metaxy_config markdown extension.

This module provides a top-level import point for the markdown extension
so it can be used as 'metaxy_config' in markdown_extensions config.
"""

from __future__ import annotations

from mkdocs_metaxy.config.markdown_ext import MetaxyConfigExtension, makeExtension

__all__ = ["MetaxyConfigExtension", "makeExtension"]
