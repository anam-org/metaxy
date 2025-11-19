from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic

import narwhals as nw
from narwhals.typing import FrameT
from typing_extensions import Self


@dataclass
class RenamedDataFrame(Generic[FrameT]):
    """A class representing a dataframe with renamed columns.

    We keep ID columns together with this DataFrame to join on them later on."""

    df: FrameT
    id_columns: list[str]

    def rename(self, mapping: Mapping[str, str]) -> Self:
        if mapping:
            # Convert Mapping to dict for narwhals rename
            self.df = self.df.rename(dict(mapping))
        self.id_columns = [mapping.get(col, col) for col in self.id_columns]
        return self

    def filter(self, filters: Sequence[nw.Expr] | None) -> Self:
        if filters:
            self.df = self.df.filter(*filters)
        return self

    def select(self, columns: Sequence[str] | None) -> Self:
        if columns:
            self.df = self.df.select(*columns)
            self.id_columns = [col for col in self.id_columns if col in columns]
        return self
