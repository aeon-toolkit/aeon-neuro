"""Test all estimators in aeon-neuro."""

from aeon.testing import testing_config as aeon_testing_config
from aeon.testing.estimator_checking import parametrize_with_checks

from aeon_neuro.testing.testing_config import EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
from aeon_neuro.utils.discovery import all_estimators

aeon_testing_config.EXCLUDED_TESTS.update(EXCLUDED_TESTS)

ALL_TEST_ESTIMATORS = [
    est
    for est in all_estimators(return_names=False, include_sklearn=False)
    if est.__name__ not in EXCLUDE_ESTIMATORS
]


@parametrize_with_checks(ALL_TEST_ESTIMATORS)
def test_all_estimators(check):
    """Run general estimator checks on all aeon-neuro estimators."""
    check()
