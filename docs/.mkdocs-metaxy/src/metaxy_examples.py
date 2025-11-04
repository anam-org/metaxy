"""Entry point for metaxy_examples markdown extension.

This module provides a top-level import point for the markdown extension
so it can be used as 'metaxy_examples' in markdown_extensions config.
"""

from __future__ import annotations

from mkdocs_metaxy.examples.markdown_ext import MetaxyExamplesExtension, makeExtension

__all__ = ["MetaxyExamplesExtension", "makeExtension"]
