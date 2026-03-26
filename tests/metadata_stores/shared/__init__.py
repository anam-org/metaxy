"""Shared test packs for metadata store tests.

Each pack is a mixin class with test methods that depend on a `store` fixture.
Per-store test modules inherit from these packs and provide the `store` fixture.
"""

from .crud import CRUDTests
from .deletion import DeletionTests
from .display import DisplayTests
from .filters import FilterTests
from .resolve_update import ResolveUpdateTests
from .versioning import VersioningTests
from .write import WriteTests

__all__ = [
    "CRUDTests",
    "DeletionTests",
    "DisplayTests",
    "FilterTests",
    "ResolveUpdateTests",
    "VersioningTests",
    "WriteTests",
]
