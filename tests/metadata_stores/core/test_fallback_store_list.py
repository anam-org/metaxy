"""Tests for FallbackStoreList lazy resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from metaxy.metadata_store.fallback import FallbackStoreList
from metaxy.versioning.types import HashAlgorithm


def _make_mock_store(hash_algorithm: HashAlgorithm = HashAlgorithm.XXHASH64) -> MagicMock:
    store = MagicMock()
    store.hash_algorithm = hash_algorithm
    return store


class TestPreResolvedStores:
    def test_len(self) -> None:
        assert len(FallbackStoreList([_make_mock_store(), _make_mock_store()])) == 2

    def test_bool_nonempty(self) -> None:
        assert FallbackStoreList([_make_mock_store()])

    def test_getitem(self) -> None:
        store_a = _make_mock_store()
        store_b = _make_mock_store()
        fsl = FallbackStoreList([store_a, store_b])
        assert fsl[0] is store_a
        assert fsl[1] is store_b

    def test_getitem_slice(self) -> None:
        stores = [_make_mock_store(), _make_mock_store(), _make_mock_store()]
        fsl = FallbackStoreList(stores)
        result = fsl[0:2]
        assert len(result) == 2
        assert result[0] is stores[0]
        assert result[1] is stores[1]

    def test_iter(self) -> None:
        stores = [_make_mock_store(), _make_mock_store()]
        assert list(FallbackStoreList(stores)) == stores

    def test_all_resolved_true(self) -> None:
        assert FallbackStoreList([_make_mock_store()]).all_resolved


class TestEmptyList:
    def test_len(self) -> None:
        assert len(FallbackStoreList()) == 0

    def test_bool_false(self) -> None:
        assert not FallbackStoreList()

    def test_bool_false_none_entries(self) -> None:
        assert not FallbackStoreList(None)

    def test_all_resolved_empty(self) -> None:
        assert FallbackStoreList().all_resolved


class TestLazyResolution:
    def test_not_resolved_at_construction(self) -> None:
        assert not FallbackStoreList(["dev", "prod"]).all_resolved

    def test_resolve_on_first_access(self) -> None:
        mock_store = _make_mock_store()
        mock_config = MagicMock()
        mock_config.get_store.return_value = mock_store

        fsl = FallbackStoreList(
            ["dev"],
            config=mock_config,
            parent_hash_algorithm=HashAlgorithm.XXHASH64,
        )

        assert fsl[0] is mock_store
        mock_config.get_store.assert_called_once_with("dev")

    def test_caching_second_access(self) -> None:
        mock_store = _make_mock_store()
        mock_config = MagicMock()
        mock_config.get_store.return_value = mock_store

        fsl = FallbackStoreList(
            ["dev"],
            config=mock_config,
            parent_hash_algorithm=HashAlgorithm.XXHASH64,
        )

        assert fsl[0] is fsl[0]
        mock_config.get_store.assert_called_once()

    def test_all_resolved_after_access(self) -> None:
        mock_config = MagicMock()
        mock_config.get_store.return_value = _make_mock_store()

        fsl = FallbackStoreList(
            ["dev"],
            config=mock_config,
            parent_hash_algorithm=HashAlgorithm.XXHASH64,
        )

        assert not fsl.all_resolved
        _ = fsl[0]
        assert fsl.all_resolved

    def test_hash_algorithm_mismatch_on_resolve(self) -> None:
        mock_config = MagicMock()
        mock_config.get_store.return_value = _make_mock_store(HashAlgorithm.XXHASH32)

        fsl = FallbackStoreList(
            ["prod"],
            config=mock_config,
            parent_hash_algorithm=HashAlgorithm.XXHASH64,
        )

        with pytest.raises(ValueError, match="hash_algorithm"):
            _ = fsl[0]

    def test_missing_config_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="no MetaxyConfig provided"):
            _ = FallbackStoreList(["dev"])[0]

    def test_mixed_resolved_and_unresolved(self) -> None:
        resolved_store = _make_mock_store()
        lazy_store = _make_mock_store()
        mock_config = MagicMock()
        mock_config.get_store.return_value = lazy_store

        fsl = FallbackStoreList(
            [resolved_store, "prod"],
            config=mock_config,
            parent_hash_algorithm=HashAlgorithm.XXHASH64,
        )

        assert not fsl.all_resolved
        assert fsl[0] is resolved_store
        mock_config.get_store.assert_not_called()

        assert fsl[1] is lazy_store
        mock_config.get_store.assert_called_once_with("prod")
        assert fsl.all_resolved

    def test_iter_resolves_all(self) -> None:
        mock_config = MagicMock()
        mock_config.get_store.return_value = _make_mock_store()

        fsl = FallbackStoreList(
            ["a", "b"],
            config=mock_config,
            parent_hash_algorithm=HashAlgorithm.XXHASH64,
        )

        assert len(list(fsl)) == 2
        assert fsl.all_resolved
