import warnings

import narwhals as nw
from narwhals.typing import Frame


class PolarsMaterializationWarning(Warning):
    pass

    @classmethod
    def warn_on_implementation_mismatch(cls, expected: nw.Implementation, actual: nw.Implementation, message: str = ""):
        if expected != actual:
            warning = f"Narwhals implementation mismatch: native is {expected}, got {actual}. This will lead to materialization into an eager Polars frame."

            if message:
                warning += f" {message}"

            warnings.warn(warning, cls, stacklevel=3)


class MetaxyColumnMissingWarning(Warning):
    pass

    @classmethod
    def warn_on_missing_column(cls, expected: str, df: Frame, message: str = ""):
        # Use collect_schema().names() to avoid PerformanceWarning on lazy frames
        columns = df.collect_schema().names()
        if expected in columns:
            return
        else:
            warning = f"Metaxy column missing: expected {expected}, got {columns}."

            if message:
                warning += f" {message}"

            warnings.warn(warning, cls, stacklevel=3)
