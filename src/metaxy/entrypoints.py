"""Entrypoint discovery and loading for Metaxy features.

This module provides functionality to automatically discover and load Feature
classes from modules, supporting both:
- Config-based entrypoints (list of module paths)
- Package-based entrypoints (via importlib.metadata)

Features are automatically registered to the active FeatureRegistry when their
containing modules are imported (via the Feature metaclass).
"""

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaxy.models.feature import FeatureRegistry

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
    registry: "FeatureRegistry | None" = None,
) -> None:
    """Load a single module entrypoint.

    Imports the specified module, which should contain Feature class definitions.
    Features will be automatically registered to the active registry via the
    Feature metaclass.

    Args:
        module_path: Fully qualified module path (e.g., "myapp.features.video")
        registry: Target registry. If None, uses FeatureRegistry.get_active()

    Raises:
        EntrypointLoadError: If module import fails

    Example:
        >>> from metaxy.entrypoints import load_module_entrypoint
        >>> load_module_entrypoint("myapp.features.core")
        >>> # Features from myapp.features.core are now registered
    """
    from metaxy.models.feature import FeatureRegistry

    target_registry = registry or FeatureRegistry.get_active()

    try:
        # Set registry as active during import so Features register to it
        with target_registry.use():
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


def load_config_entrypoints(
    entrypoints: list[str],
    *,
    registry: "FeatureRegistry | None" = None,
) -> None:
    """Load multiple module entrypoints from a list.

    This is the config-based entrypoint loading mechanism. Each string in the
    list should be a fully qualified module path.

    Args:
        entrypoints: List of module paths to import
        registry: Target registry. If None, uses FeatureRegistry.get_active()

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
    from metaxy.models.feature import FeatureRegistry

    target_registry = registry or FeatureRegistry.get_active()

    logger.info(f"Loading {len(entrypoints)} config-based entrypoints")

    for module_path in entrypoints:
        load_module_entrypoint(module_path, registry=target_registry)


def load_package_entrypoints(
    group: str = DEFAULT_ENTRY_POINT_GROUP,
    *,
    registry: "FeatureRegistry | None" = None,
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
        registry: Target registry. If None, uses FeatureRegistry.get_active()

    Raises:
        EntrypointLoadError: If any entrypoint fails to load

    Example:
        >>> from metaxy.entrypoints import load_package_entrypoints
        >>> # Discover and load all installed plugins
        >>> load_package_entrypoints()
    """
    from metaxy.models.feature import FeatureRegistry

    target_registry = registry or FeatureRegistry.get_active()

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
            with target_registry.use():
                ep.load()
            logger.info(f"Successfully loaded package entrypoint: {ep.name}")
        except Exception as e:
            raise EntrypointLoadError(
                f"Failed to load package entrypoint '{ep.name}' ({ep.value}): {e}"
            ) from e


def discover_and_load_entrypoints(
    config_entrypoints: list[str] | None = None,
    package_entrypoint_group: str = DEFAULT_ENTRY_POINT_GROUP,
    *,
    load_config: bool = True,
    load_packages: bool = True,
    registry: "FeatureRegistry | None" = None,
) -> "FeatureRegistry":
    """Discover and load all entrypoints from both config and packages.

    This is the main entry point for loading features. It combines both
    config-based and package-based entrypoint discovery.

    Args:
        config_entrypoints: List of module paths from config (optional)
        package_entrypoint_group: Entry point group for package discovery
        load_config: Whether to load config-based entrypoints (default: True)
        load_packages: Whether to load package-based entrypoints (default: True)
        registry: Target registry. If None, uses FeatureRegistry.get_active()

    Returns:
        The registry that was populated (useful for testing/inspection)

    Raises:
        EntrypointLoadError: If any entrypoint fails to load

    Example:
        >>> from metaxy.entrypoints import discover_and_load_entrypoints
        >>>
        >>> # Load from both sources
        >>> registry = discover_and_load_entrypoints(
        ...     config_entrypoints=["myapp.features.core"],
        ...     load_packages=True
        ... )
        >>>
        >>> # Load only from config
        >>> registry = discover_and_load_entrypoints(
        ...     config_entrypoints=["myapp.features.core"],
        ...     load_packages=False
        ... )
    """
    from metaxy.models.feature import FeatureRegistry

    target_registry = registry or FeatureRegistry.get_active()

    logger.info("Starting entrypoint discovery and loading")

    # Load config-based entrypoints
    if load_config and config_entrypoints:
        load_config_entrypoints(config_entrypoints, registry=target_registry)

    # Load package-based entrypoints
    if load_packages:
        load_package_entrypoints(package_entrypoint_group, registry=target_registry)

    num_features = len(target_registry.features_by_key)
    logger.info(
        f"Entrypoint loading complete. Registry now contains {num_features} features."
    )

    return target_registry
