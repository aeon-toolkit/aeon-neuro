"""Test all estimators in aeon."""

from aeon.testing.estimator_checking import parametrize_with_checks
from aeon.utils.discovery import all_estimators

ALL_TEST_ESTIMATORS = all_estimators(return_names=False, include_sklearn=False)


@parametrize_with_checks(ALL_TEST_ESTIMATORS)
def test_all_estimators(check):
    """Run general estimator checks on all aeon estimators."""
    check()
