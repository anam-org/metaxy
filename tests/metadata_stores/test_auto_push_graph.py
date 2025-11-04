from __future__ import annotations

import json
from pathlib import Path

from metaxy import (
    Feature,
    FeatureKey,
    FieldKey,
    FieldSpec,
)
from metaxy import (
    TestingFeatureSpec as TFeatureSpec,
)
from metaxy.config import MetaxyConfig
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY


class CountingStore(InMemoryMetadataStore):
    """Store that tracks read_metadata_in_store calls for caching assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.read_calls = 0

    def read_metadata_in_store(self, *args, **kwargs):
        self.read_calls += 1
        return super().read_metadata_in_store(*args, **kwargs)


def _patch_cache_dir(monkeypatch, cache_dir: Path) -> None:
    """Redirect snapshot cache to a temp directory for tests."""

    def _cache_path(appname: str) -> str:  # pragma: no cover - trivial wrapper
        return str(cache_dir)

    monkeypatch.setattr(
        "metaxy.metadata_store.base.user_cache_dir",
        _cache_path,
    )


def test_auto_push_graph_on_open(monkeypatch, tmp_path, config, graph):
    """Auto-push should record a snapshot as soon as a store opens."""
    cache_dir = tmp_path / "cache_auto"
    _patch_cache_dir(monkeypatch, cache_dir)

    auto_config = config.model_copy(update={"auto_push_graph": True})
    MetaxyConfig.set(auto_config)

    try:

        class AutoFeature(
            Feature,
            spec=TFeatureSpec(
                key=FeatureKey(["auto", "feature"]),
                fields=[FieldSpec(key=FieldKey(["auto", "value"]), code_version="1")],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None

            versions_df = versions_lazy.collect().to_polars()
            assert versions_df.height == 1
            assert versions_df.get_column("feature_key").to_list() == ["auto/feature"]

            # Manual push after auto-push should be an already-recorded no-op
            result = store.record_feature_graph_snapshot()
            assert result.already_recorded is True
            assert result.metadata_changed is False
            assert result.snapshot_version
    finally:
        # Restore base config for other tests
        MetaxyConfig.set(config)


def test_snapshot_push_cache_skips_duplicate_reads(
    monkeypatch, tmp_path, config, graph
):
    """Cache should short-circuit repeated pushes of identical graph states."""
    cache_dir = tmp_path / "cache_skip"
    _patch_cache_dir(monkeypatch, cache_dir)

    # Ensure base configuration has auto push disabled for explicit control
    MetaxyConfig.set(config.model_copy(update={"auto_push_graph": False}))

    try:

        class CachedFeature(
            Feature,
            spec=TFeatureSpec(
                key=FeatureKey(["cached", "feature"]),
                fields=[FieldSpec(key=FieldKey(["cached", "value"]), code_version="1")],
            ),
        ):
            pass

        cache_file = cache_dir / "push_cache.json"

        with CountingStore() as store:
            result1 = store.record_feature_graph_snapshot()
            assert result1.already_recorded is False
            assert store.read_calls == 1

            result2 = store.record_feature_graph_snapshot()
            assert result2.already_recorded is True
            assert result2.metadata_changed is False
            assert store.read_calls == 2  # Second call still checks store state once

        assert cache_file.exists()
        cache_payload = json.loads(cache_file.read_text(encoding="utf-8"))
        metadata_section = cache_payload.get("_metadata", {})
        project_entry = metadata_section.get(config.project, {})
        assert project_entry, "Expected project-specific cache entry"

        scope_entry = next(iter(project_entry.values()), None)
        assert scope_entry is not None, "Expected at least one scoped cache entry"
        assert isinstance(scope_entry, dict) and scope_entry, (
            "Expected scoped cache entries"
        )

        snapshot_entry = scope_entry.get(result1.snapshot_version, {})
        spec_versions = snapshot_entry.get("spec_versions", {})
        assert (
            spec_versions.get("cached/feature") == CachedFeature.feature_spec_version()
        )
    finally:
        MetaxyConfig.set(config)
