from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from typing_extensions import Self

from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.models.plan import FeaturePlan
from metaxy.provenance.ibis import IbisHashFn
from metaxy.provenance.tracker import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


def _create_duckdb_hash_functions() -> dict[HashAlgorithm, IbisHashFn]:
    import ibis

    @ibis.udf.scalar.builtin
    def MD5(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def HEX(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def LOWER(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def xxh32(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def xxh64(x: str) -> str: ...

    def md5_hash(expr):
        return LOWER(HEX(MD5(expr.cast(str))))

    def xxhash32_hash(expr):
        return xxh32(expr).cast(str)

    def xxhash64_hash(expr):
        return xxh64(expr).cast(str)

    return {
        HashAlgorithm.MD5: md5_hash,
        HashAlgorithm.XXHASH32: xxhash32_hash,
        HashAlgorithm.XXHASH64: xxhash64_hash,
    }


class DuckDBMetadataStore(IbisMetadataStore):
    def __init__(
        self,
        database: str,
        config: dict[str, str] | None = None,
        hash_algorithm: HashAlgorithm = HashAlgorithm.XXHASH64,
        hash_length: int = 16,
        auto_create_tables: bool = True,
        **kwargs: Any,
    ):
        self.database = database
        self.config = config

        super().__init__(
            hash_functions=_create_duckdb_hash_functions(),
            hash_algo=hash_algorithm,
            hash_length=hash_length,
            auto_create_tables=auto_create_tables,
        )

    @contextmanager
    def open(self) -> Iterator[Self]:
        import ibis

        conn = ibis.duckdb.connect(self.database, **(self.config or {}))
        with super()._open_with_connection(conn) as store:
            yield store

    def create_tracker(self, plan: FeaturePlan) -> ProvenanceTracker:
        # Import inside function to avoid module-level import
        import ibis.backends.duckdb

        # Cast to DuckDB backend to access raw_sql method
        duckdb_conn = cast(ibis.backends.duckdb.Backend, self.conn)
        duckdb_conn.raw_sql("INSTALL hashfuncs FROM community")
        duckdb_conn.raw_sql("LOAD hashfuncs")
        return super().create_tracker(plan)
