"""Simple tests for IbisJsonCompatStore to verify it's correctly configured."""

import pytest

from metaxy.metadata_store.ibis_json_compat import IbisJsonCompatStore
from metaxy.versioning.flat_engine import IbisFlatVersioningEngine


def test_ibis_json_compat_store_is_abstract():
    """Test that IbisJsonCompatStore cannot be instantiated (abstract class)."""
    with pytest.raises(TypeError, match="abstract"):
        IbisJsonCompatStore(  # pyright: ignore[reportAbstractUsage]
            backend="duckdb",
            connection_params={"database": ":memory:"},
        )


def test_ibis_json_compat_store_forces_dict_based_engine():
    """Test that subclasses inherit dict-based engine configuration."""
    from metaxy.versioning.types import HashAlgorithm

    class ConcreteJsonCompatStore(IbisJsonCompatStore):
        """Minimal concrete implementation for testing."""

        def _create_hash_functions(self):
            import ibis

            @ibis.udf.scalar.builtin
            def md5(_x: str) -> str: ...

            return {HashAlgorithm.MD5: lambda x: md5(x.cast(str))}

        def _get_json_unpack_exprs(self, json_column, field_names):
            # Minimal implementation
            return {}

        def _get_json_pack_expr(self, struct_name, field_columns):
            # Minimal implementation
            import ibis

            return ibis.literal("{}")

    store = ConcreteJsonCompatStore(
        backend="duckdb",
        connection_params={"database": ":memory:"},
    )

    # Verify it uses dict-based engine
    assert store.versioning_engine_cls == IbisFlatVersioningEngine


def test_abstract_methods_are_defined():
    """Test that IbisJsonCompatStore defines the required abstract methods."""
    # Get all abstract methods
    abstract_methods = IbisJsonCompatStore.__abstractmethods__

    # Should have the JSON pack/unpack methods
    assert "_get_json_unpack_exprs" in abstract_methods
    assert "_get_json_pack_expr" in abstract_methods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
