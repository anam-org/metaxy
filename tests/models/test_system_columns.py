from metaxy.models import constants as sys_cols


def test_is_system_column_recognises_canonical_names_only() -> None:
    """Test that is_system_column recognizes canonical system column names."""
    # Should recognize actual system columns
    assert sys_cols.is_system_column(sys_cols.METAXY_SNAPSHOT_VERSION)
    assert sys_cols.is_system_column(sys_cols.METAXY_FEATURE_VERSION)
    assert sys_cols.is_system_column(sys_cols.METAXY_PROVENANCE_BY_FIELD)

    # Should not recognize user-defined columns
    assert not sys_cols.is_system_column("user_defined_column")
    assert not sys_cols.is_system_column("my_column")


def test_is_droppable_system_column_requires_canonical_name() -> None:
    """Test that is_droppable_system_column identifies droppable system columns."""
    # Droppable columns
    assert sys_cols.is_droppable_system_column(sys_cols.METAXY_FEATURE_VERSION)
    assert sys_cols.is_droppable_system_column(sys_cols.METAXY_SNAPSHOT_VERSION)

    # Not droppable (essential columns)
    assert not sys_cols.is_droppable_system_column(sys_cols.METAXY_PROVENANCE_BY_FIELD)

    # Not system columns at all
    assert not sys_cols.is_droppable_system_column("user_column")
