"""Shared fixtures for project-related tests."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType

import pytest


class FakePackageFactory:
    """Factory for creating fake packages in sys.modules."""

    def __init__(self) -> None:
        self._created_modules: list[str] = []

    def create(
        self,
        package_name: str,
        project_name: str | None = None,
    ) -> ModuleType:
        """Create a fake package and register it in sys.modules.

        Args:
            package_name: The name for the fake module.
            project_name: If provided, sets __metaxy_project__ on the module.
        """
        module = ModuleType(package_name)
        if project_name is not None:
            module.__metaxy_project__ = project_name  # type: ignore[attr-defined]
        sys.modules[package_name] = module
        self._created_modules.append(package_name)
        return module

    def cleanup(self) -> None:
        """Remove all created modules from sys.modules."""
        for name in self._created_modules:
            sys.modules.pop(name, None)
        self._created_modules.clear()


@pytest.fixture
def fake_package_factory() -> Iterator[FakePackageFactory]:
    """Factory fixture for creating fake packages with automatic cleanup."""
    factory = FakePackageFactory()
    yield factory
    factory.cleanup()


@pytest.fixture
def project_a_package(fake_package_factory: FakePackageFactory) -> ModuleType:
    """Create a fake package with __metaxy_project__ = 'project_a'."""
    return fake_package_factory.create("fake_project_a_pkg", "project_a")


@pytest.fixture
def project_b_package(fake_package_factory: FakePackageFactory) -> ModuleType:
    """Create a fake package with __metaxy_project__ = 'project_b'."""
    return fake_package_factory.create("fake_project_b_pkg", "project_b")


@pytest.fixture
def project_c_package(fake_package_factory: FakePackageFactory) -> ModuleType:
    """Create a fake package with __metaxy_project__ = 'project_c'."""
    return fake_package_factory.create("fake_project_c_pkg", "project_c")
