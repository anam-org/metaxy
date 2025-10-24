"""Entrypoint discovery and loading for Metaxy features.

This module provides functionality to automatically discover and load Feature
classes from modules, supporting both:
- Config-based entrypoints (list of module paths)
- Environment-based entrypoints (environment variables starting with METAXY_ENTRYPOINT)
- Package-based entrypoints (via importlib.metadata)

Features are automatically registered to the active FeatureGraph when their
containing modules are imported (via the Feature metaclass).
"""

import importlib
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaxy.models.feature import FeatureGraph

# Conditional import for Python 3.10+ compatibility
from importlib.metadata import entry_points  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# Default entry point group name for package-based discovery
DEFAULT_ENTRY_POINT_GROUP = "metaxy.features"


class EntrypointLoadError(Exception):
    """Raised when an entrypoint fails to load."""

    pass


def load_module_entrypoint(
    module_path: str,
    *,
    graph: "FeatureGraph | None" = None,
) -> None:
    """Load a single module entrypoint.

    Imports the specified module, which should contain Feature class definitions.
    Features will be automatically registered to the active graph via the
    Feature metaclass.

    Args:
        module_path: Fully qualified module path (e.g., "myapp.features.video")
        graph: Target graph. If None, uses FeatureGraph.get_active()

    Raises:
        EntrypointLoadError: If module import fails

    Example:
        >>> from metaxy.entrypoints import load_module_entrypoint
        >>> load_module_entrypoint("myapp.features.core")
        >>> # Features from myapp.features.core are now registered
    """
    from metaxy.models.feature import FeatureGraph

    target_graph = graph or FeatureGraph.get_active()

    try:
        # Set graph as active during import so Features register to it
        with target_graph.use():
            logger.debug(f"Loading entrypoint module: {module_path}")
            importlib.import_module(module_path)
            logger.info(f"Successfully loaded entrypoint: {module_path}")
    except ImportError as e:
        raise EntrypointLoadError(
            f"Failed to import entrypoint module '{module_path}': {e}"
        ) from e
    except Exception as e:
        raise EntrypointLoadError(
            f"Error loading entrypoint module '{module_path}': {e}"
        ) from e


def load_entrypoints(
    entrypoints: list[str],
    *,
    graph: "FeatureGraph | None" = None,
) -> None:
    """Load multiple module entrypoints from a list.

    Args:
        entrypoints: List of module paths to import
        graph: Target graph. If None, uses FeatureGraph.get_active()

    Raises:
        EntrypointLoadError: If any module import fails

    Example:
        >>> from metaxy.entrypoints import load_config_entrypoints
        >>> load_config_entrypoints([
        ...     "myapp.features.video",
        ...     "myapp.features.audio",
        ...     "myapp.features.text"
        ... ])
    """
    from metaxy.models.feature import FeatureGraph

    target_graph = graph or FeatureGraph.get_active()

    logger.info(f"Loading {len(entrypoints)} entrypoints")

    for module_path in entrypoints:
        load_module_entrypoint(module_path, graph=target_graph)


def load_package_entrypoints(
    group: str = DEFAULT_ENTRY_POINT_GROUP,
    *,
    graph: "FeatureGraph | None" = None,
) -> None:
    """Load entrypoints from installed packages using importlib.metadata.

    Discovers and loads all entry points registered in the specified group.
    This is the package-based entrypoint mechanism using standard Python
    packaging infrastructure.

    Packages declare entrypoints in their pyproject.toml:
        [project.entry-points."metaxy.features"]
        myfeature = "mypackage.features.module"

    Args:
        group: Entry point group name (default: "metaxy.features")
        graph: Target graph. If None, uses FeatureGraph.get_active()

    Raises:
        EntrypointLoadError: If any entrypoint fails to load

    Example:
        >>> from metaxy.entrypoints import load_package_entrypoints
        >>> # Discover and load all installed plugins
        >>> load_package_entrypoints()
    """
    from metaxy.models.feature import FeatureGraph

    target_graph = graph or FeatureGraph.get_active()

    logger.info(f"Discovering package entrypoints in group: {group}")

    # Discover entry points
    # Note: Python 3.10+ returns SelectableGroups, 3.9 returns dict
    discovered = entry_points()

    # Handle different return types across Python versions
    if hasattr(discovered, "select"):
        # Python 3.10+: SelectableGroups with select() method
        eps = discovered.select(group=group)
    else:
        # Python 3.9: dict-like interface
        eps = discovered.get(group, [])

    eps_list = list(eps)

    if not eps_list:
        logger.debug(f"No package entrypoints found in group: {group}")
        return

    logger.info(f"Found {len(eps_list)} package entrypoints in group: {group}")

    for ep in eps_list:
        try:
            logger.debug(f"Loading package entrypoint: {ep.name} = {ep.value}")
            # Load the entry point (imports the module)
            with target_graph.use():
                ep.load()
            logger.info(f"Successfully loaded package entrypoint: {ep.name}")
        except Exception as e:
            raise EntrypointLoadError(
                f"Failed to load package entrypoint '{ep.name}' ({ep.value}): {e}"
            ) from e


def load_env_entrypoints() -> None:
    """Load entrypoints from environment variables.

    Discovers and loads all entry points from environment variables matching
    the pattern METAXY_ENTRYPOINT*. Each variable should contain a
    comma-separated list of module paths.

    Environment variables:
        METAXY_ENTRYPOINT="myapp.features.core,myapp.features.extra"
        METAXY_ENTRYPOINT_PLUGINS="plugin1.features,plugin2.features"

    Args:
        graph: Target graph. If None, uses FeatureGraph.get_active()

    Raises:
        EntrypointLoadError: If any entrypoint fails to load

    Example:
        >>> import os
        >>> os.environ["METAXY_ENTRYPOINT"] = "myapp.features.core"
        >>> from metaxy.entrypoints import load_env_entrypoints
        >>> load_env_entrypoints()
    """

    logger.info("Discovering environment-based entrypoints (METAXY_ENTRYPOINT*)")

    # Find all environment variables matching METAXY_ENTRYPOINT*
    env_vars = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("METAXY_ENTRYPOINT")
    }

    if not env_vars:
        logger.debug("No environment entrypoints found (METAXY_ENTRYPOINT* not set)")
        return

    logger.info(f"Found {len(env_vars)} METAXY_ENTRYPOINT* environment variable(s)")

    # Collect all module paths from all matching env vars
    all_module_paths = []
    for env_var, value in sorted(env_vars.items()):
        logger.debug(f"Processing {env_var}={value}")
        # Split by comma and strip whitespace
        module_paths = [path.strip() for path in value.split(",") if path.strip()]
        all_module_paths.extend(module_paths)

    if not all_module_paths:
        logger.debug("No module paths found in METAXY_ENTRYPOINT* variables")
        return

    logger.info(
        f"Loading {len(all_module_paths)} module(s) from environment entrypoints"
    )

    # Load each module path
    for module_path in all_module_paths:
        load_module_entrypoint(module_path)


def load_features(
    entrypoints: list[str] | None = None,
    package_entrypoint_group: str = DEFAULT_ENTRY_POINT_GROUP,
    *,
    load_config: bool = True,
    load_packages: bool = True,
    load_env: bool = True,
) -> "FeatureGraph":
    """Discover and load all entrypoints from config, packages, and environment.

    This is the main entry point for loading features. It combines config-based,
    package-based, and environment-based entrypoint discovery.

    Args:
        entrypoints: List of module paths (optional)
        package_entrypoint_group: Entry point group for package discovery
        load_config: Whether to load config-based entrypoints (default: True)
        load_packages: Whether to load package-based entrypoints (default: True)
        load_env: Whether to load environment-based entrypoints (default: True)

    Returns:
        The graph that was populated (useful for testing/inspection)

    Raises:
        EntrypointLoadError: If any entrypoint fails to load

    Example:
        >>> from metaxy.entrypoints import load_features
        >>>
        >>> # Load from all sources
        >>> graph = load_features(
        ...     entrypoints=["myapp.features.core"],
        ...     load_packages=True,
        ...     load_env=True
        ... )
        >>>
        >>> # Load only from config
        >>> graph = load_features(
        ...     entrypoints=["myapp.features.core"],
        ...     load_packages=False,
        ...     load_env=False
        ... )
    """
    from metaxy.config import MetaxyConfig
    from metaxy.models.feature import FeatureGraph

    target_graph = FeatureGraph.get_active()

    logger.info("Starting entrypoint discovery and loading")

    # Load explicit entrypoints
    if entrypoints:
        load_entrypoints(entrypoints)

    # Load config-based entrypoints
    if load_config:
        config = MetaxyConfig.load(search_parents=True)
        load_entrypoints(config.entrypoints)

    # Load package-based entrypoints
    if load_packages:
        load_package_entrypoints(package_entrypoint_group)

    # Load environment-based entrypoints
    if load_env:
        load_env_entrypoints()

    num_features = len(target_graph.features_by_key)
    logger.info(
        f"Entrypoint loading complete. Registry now contains {num_features} features."
    )

    return target_graph
