from typing import Any

from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.provenance.ibis import IbisHashFn
from metaxy.provenance.types import HashAlgorithm


def _create_clickhouse_hash_functions() -> dict[HashAlgorithm, IbisHashFn]:
    import ibis

    @ibis.udf.scalar.builtin
    def MD5(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def HEX(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def lower(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def xxHash32(x: str) -> int: ...

    @ibis.udf.scalar.builtin
    def xxHash64(x: str) -> int: ...

    @ibis.udf.scalar.builtin
    def toString(x: int) -> str: ...

    def md5_hash(expr):
        return lower(HEX(MD5(expr.cast(str))))

    def xxhash32_hash(expr):
        return toString(xxHash32(expr))

    def xxhash64_hash(expr):
        return toString(xxHash64(expr))

    return {
        HashAlgorithm.MD5: md5_hash,
        HashAlgorithm.XXHASH32: xxhash32_hash,
        HashAlgorithm.XXHASH64: xxhash64_hash,
    }


class ClickHouseMetadataStore(IbisMetadataStore):
    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        import ibis

        if connection_string:
            backend = ibis.connect(connection_string)
        elif connection_params:
            backend = ibis.clickhouse.connect(**connection_params)
        else:
            raise ValueError("Must provide connection_string or connection_params")

        super().__init__(backend, _create_clickhouse_hash_functions())
