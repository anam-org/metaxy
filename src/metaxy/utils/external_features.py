"""External feature synchronization utilities.

This module handles syncing external feature definitions from metadata stores,
including version mismatch detection and warning/error handling.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import narwhals as nw

from metaxy._decorators import public
from metaxy._exceptions import ExternalFeatureVersionMismatchError
from metaxy._warnings import (
    ExternalFeatureVersionMismatchWarning,
    UnresolvedExternalFeatureWarning,
)
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store import MetadataStore
    from metaxy.models.feature_definition import FeatureDefinition


def _format_field_mismatches(
    expected_by_field: dict[str, str],
    actual_by_field: dict[str, str],
) -> list[str]:
    """Format field-level mismatches for a single feature."""
    mismatched = []
    all_fields = set(expected_by_field.keys()) | set(actual_by_field.keys())
    for field in sorted(all_fields):
        expected = expected_by_field.get(field, "<missing>")
        actual = actual_by_field.get(field, "<missing>")
        if expected != actual:
            mismatched.append(f"      {field}: expected '{expected}', got '{actual}'")
    return mismatched


def _format_version_mismatch_message(
    mismatches: list[tuple[FeatureKey, str, str, dict[str, str], dict[str, str]]],
) -> str:
    """Format a consolidated message for version mismatches.

    Args:
        mismatches: List of (key, expected_version, actual_version, expected_by_field, actual_by_field)

    Returns:
        Formatted message string.
    """
    lines = [
        f"Version mismatch detected for {len(mismatches)} external feature(s). "
        "The external feature definition(s) may be out of sync with the metadata store.",
        "",
    ]

    for key, expected_version, actual_version, expected_by_field, actual_by_field in mismatches:
        lines.append(f"  {key.to_string()}:")
        lines.append(f"    feature version: expected '{expected_version}', got '{actual_version}'")
        field_mismatches = _format_field_mismatches(expected_by_field, actual_by_field)
        if field_mismatches:
            lines.append("    field mismatches:")
            lines.extend(field_mismatches)
        lines.append("")

    return "\n".join(lines)


def _check_version_mismatches(
    graph: FeatureGraph,
    external_versions_before: dict[FeatureKey, tuple[str, dict[str, str], FeatureDefinition]],
    on_version_mismatch: Literal["warn", "error"] | None,
) -> None:
    """Check for version mismatches on external features that were replaced.

    Args:
        graph: The feature graph after loading definitions.
        external_versions_before: Dict mapping feature keys to (version, by_field, definition) tuples
            recorded before loading.
        on_version_mismatch: Override for the mismatch handling mode.

    Raises:
        ExternalFeatureVersionMismatchError: If any features have mismatches with error mode.
    """

    warn_mismatches: list[tuple[FeatureKey, str, str, dict[str, str], dict[str, str]]] = []
    error_mismatches: list[tuple[FeatureKey, str, str, dict[str, str], dict[str, str]]] = []

    for key, (expected_version, expected_by_field, external_defn) in external_versions_before.items():
        # Only check if the external feature was replaced (no longer external)
        current_defn = graph.feature_definitions_by_key.get(key)
        if current_defn is None or current_defn.is_external:
            continue

        actual_version = graph.get_feature_version(key)
        actual_by_field = graph.get_feature_version_by_field(key)

        if expected_version != actual_version:
            mismatch_data = (key, expected_version, actual_version, expected_by_field, actual_by_field)
            if on_version_mismatch is not None:
                effective_mode = on_version_mismatch
            else:
                effective_mode = external_defn.on_version_mismatch
            if effective_mode == "error":
                error_mismatches.append(mismatch_data)
            else:
                warn_mismatches.append(mismatch_data)

    # Issue consolidated warning for all "warn" mismatches
    if warn_mismatches:
        message = _format_version_mismatch_message(warn_mismatches)
        warnings.warn(message, ExternalFeatureVersionMismatchWarning, stacklevel=4)

    # Raise consolidated error for all "error" mismatches
    if error_mismatches:
        message = _format_version_mismatch_message(error_mismatches)
        raise ExternalFeatureVersionMismatchError(message)


@public
def sync_external_features(
    store: MetadataStore,
    *,
    on_version_mismatch: Literal["warn", "error"] | None = None,
) -> list[FeatureDefinition]:
    """Sync external feature definitions from a metadata store if the graph has any.

    This function loads feature definitions from the metadata store to replace
    external feature placeholders in the active graph. It also validates that
    the versions match and warns or errors on mismatches.

    Additionally, this function loads any feature keys specified in the
    `features` config field, warning if any of them are not found in the metadata store.

    Args:
        store: Metadata store to load from. Will be opened automatically if not already open.
        on_version_mismatch: Optional override for the `on_version_mismatch` setting on external feature definitions.

            !!! info
                Setting [`MetaxyConfig.locked`][metaxy.MetaxyConfig] to `True` takes precedence over this argument.

    Returns:
        List of loaded FeatureDefinition objects.

    Example:
        ```python
        import metaxy as mx

        # Sync external features before running a pipeline
        mx.sync_external_features(store)

        # Or with explicit error handling
        mx.sync_external_features(store, on_version_mismatch="error")
        ```
    """
    from metaxy.config import MetaxyConfig
    from metaxy.metadata_store.system import SystemTableStorage

    graph = FeatureGraph.get_active()
    config = MetaxyConfig.get(_allow_default_config=True)

    if not graph.has_external_features:
        return []

    # Check if locked mode is enabled
    if config.locked:
        on_version_mismatch = "error"

    # Record versions of external features BEFORE loading
    external_versions_before: dict[FeatureKey, tuple[str, dict[str, str], FeatureDefinition]] = {}
    external_keys: list[str] = []
    for key, defn in graph.feature_definitions_by_key.items():
        if defn.is_external:
            external_versions_before[key] = (
                graph.get_feature_version(key),
                graph.get_feature_version_by_field(key),
                defn,
            )
            external_keys.append(key.to_string())

    # Use nullcontext if store is already open, otherwise open it
    cm = nullcontext(store) if store._is_open else store
    with cm:
        storage = SystemTableStorage(store)
        # Load features by key, not by project - external features may have placeholder projects
        result = storage._load_feature_definitions_raw(
            filters=[nw.col("feature_key").is_in(external_keys)],
        )

    # Check for version mismatches
    _check_version_mismatches(graph, external_versions_before, on_version_mismatch)

    # Warn if there are still unresolved external features after sync
    remaining_external = list(sorted(d.spec.key for d in graph.feature_definitions_by_key.values() if d.is_external))
    if remaining_external:
        keys_str = ", ".join(str(k) for k in remaining_external)
        warnings.warn(
            f"After syncing, {len(remaining_external)} external feature(s) could not be resolved "
            f"from the metadata store: {keys_str}. "
            f"These features may not exist in the store.",
            UnresolvedExternalFeatureWarning,
            stacklevel=2,
        )

    return result
