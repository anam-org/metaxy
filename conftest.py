"""Root conftest for Sybil docstring code block testing.

This module configures Sybil to parse and test markdown code blocks
(```python and ```py) found in docstrings throughout the src/metaxy codebase.
"""

from sybil import Sybil
from sybil.document import PythonDocStringDocument
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.markdown.codeblock import CodeBlockParser
from sybil.parsers.markdown.skip import SkipParser


def sybil_setup(namespace):
    """Set up an isolated FeatureGraph for each docstring document.

    This provides a clean graph context so examples don't need boilerplate.
    Examples can define features and they'll be registered to this isolated graph.
    """
    from metaxy.models.feature import FeatureGraph

    # Create isolated graph and enter its context
    graph = FeatureGraph()
    context_manager = graph.use()
    context_manager.__enter__()
    namespace["_sybil_graph"] = graph
    namespace["_sybil_context_manager"] = context_manager


def sybil_teardown(namespace):
    """Clean up the isolated FeatureGraph context."""
    context_manager = namespace.get("_sybil_context_manager")
    if context_manager:
        context_manager.__exit__(None, None, None)


# Create parsers for both ```python and ```py code blocks
# SkipParser must come first to handle skip directives before code blocks
python_evaluator = PythonEvaluator()
parsers = [
    SkipParser(),
    CodeBlockParser(language="python", evaluator=python_evaluator),
    CodeBlockParser(language="py", evaluator=python_evaluator),
]

# Configure Sybil to extract docstrings from Python files and parse code blocks
pytest_collect_file = Sybil(
    parsers=parsers,
    patterns=["src/metaxy/**/*.py"],
    document_types={".py": PythonDocStringDocument},
    setup=sybil_setup,
    teardown=sybil_teardown,
    excludes=[
        "**/ext/mcp/**",  # MCP module has import conflict with 'mcp' package
        "**/_testing/**",  # Internal testing utilities - examples not meant to be executed
    ],
).pytest()
