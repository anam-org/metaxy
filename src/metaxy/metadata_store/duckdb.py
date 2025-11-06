from typing import Any

from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.provenance.ibis import IbisHashFn
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
    def __init__(self, database: str, **kwargs: Any):
        import ibis

        backend = ibis.duckdb.connect(database)
        backend.raw_sql("INSTALL hashfuncs FROM community")
        backend.raw_sql("LOAD hashfuncs")

        super().__init__(backend, _create_duckdb_hash_functions())
        self.database = database
