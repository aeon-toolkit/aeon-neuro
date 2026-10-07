"""Tests that soft dependencies are handled correctly in modules."""

import re
from importlib import import_module

import pytest

from aeon_neuro.testing.tests import ALL_AEON_NEURO_MODULES


def test_module_crawl():
    """Test that we are crawling modules correctly."""
    assert "aeon.transformations" in ALL_AEON_NEURO_MODULES
    assert "aeon.transformations.collection" in ALL_AEON_NEURO_MODULES
    assert "aeon.transformations.series" in ALL_AEON_NEURO_MODULES
    assert "aeon.utils" in ALL_AEON_NEURO_MODULES
    assert "aeon.utils.discovery" in ALL_AEON_NEURO_MODULES


@pytest.mark.parametrize("module", ALL_AEON_NEURO_MODULES)
def test_module_soft_deps(module):
    """Test soft dependency imports in aeon modules.

    Imports all modules and catch exceptions due to missing dependencies.
    """
    try:
        import_module(module)
    except ModuleNotFoundError as e:  # pragma: no cover
        dependency = "unknown"
        match = re.search(r"\'(.+?)\'", str(e))
        if match:
            dependency = match.group(1)

        raise ModuleNotFoundError(
            f"The module: {module} should not require any soft dependencies, "
            f"but tried importing: '{dependency}'. Make sure soft dependencies are "
            f"properly isolated."
        ) from e
