from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Generic

import narwhals as nw
from narwhals.typing import FrameT


@dataclass(frozen=True)
class RenamedDataFrame(Generic[FrameT]):
    """An immutable wrapper for a dataframe with renamed columns.

    Each transformation method returns a new instance, preserving immutability.
    ID columns are tracked alongside the DataFrame for joining later."""

    df: FrameT
    id_columns: tuple[str, ...]

    def rename(self, mapping: Mapping[str, str]) -> "RenamedDataFrame[FrameT]":
        new_df = self.df.rename(mapping) if mapping else self.df  # ty: ignore[invalid-argument-type]
        new_id_columns = tuple(mapping.get(col, col) for col in self.id_columns)
        return replace(self, df=new_df, id_columns=new_id_columns)

    def filter(self, filters: Sequence[nw.Expr] | None) -> "RenamedDataFrame[FrameT]":
        if filters:
            new_df = self.df.filter(*filters)  # ty: ignore[invalid-argument-type]
            return replace(self, df=new_df)
        return self

    def select(self, columns: Sequence[str] | None) -> "RenamedDataFrame[FrameT]":
        if columns:
            new_df = self.df.select(*columns)  # ty: ignore[invalid-argument-type]
            new_id_columns = tuple(col for col in self.id_columns if col in columns)
            return replace(self, df=new_df, id_columns=new_id_columns)
        return self
