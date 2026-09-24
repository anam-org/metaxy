ARG DUCKDB_VERSION=1.4.3

FROM duckdb/duckdb:${DUCKDB_VERSION} AS duckdb-extensions
COPY duckdb-extensions.sql /duckdb-extensions.sql
RUN ["/duckdb", "-init", "/duckdb-extensions.sql", "-c", "SELECT 1"]

FROM python:3.12-slim
ARG DUCKDB_VERSION
COPY --from=duckdb-extensions /root/.duckdb /root/.duckdb
RUN pip install --no-cache-dir "duckdb==${DUCKDB_VERSION}" "metaxy[duckdb]"
WORKDIR /app
COPY . /app
