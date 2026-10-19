"""Tests for the aeon-neuro package and testing module utilities."""

import pkgutil

import aeon_neuro

# collect all modules
ALL_AEON_NEURO_MODULES = pkgutil.walk_packages(
    aeon_neuro.__path__, aeon_neuro.__name__ + "."
)
ALL_AEON_NEURO_MODULES = [x[1] for x in ALL_AEON_NEURO_MODULES]

ALL_AEON_NEURO_MODULES_NO_TESTS = [
    x
    for x in ALL_AEON_NEURO_MODULES
    if not any(part == "tests" for part in x.split("."))
]
