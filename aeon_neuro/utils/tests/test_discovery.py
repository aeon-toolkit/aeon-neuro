"""Testing of estimator lookup functionality."""

import pytest
from aeon.base import BaseAeonEstimator
from aeon.classification import BaseClassifier
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer
from aeon.utils.base import BASE_CLASS_REGISTER
from sklearn.base import BaseEstimator

from aeon_neuro.classification import (
    HIVECOTEV2,
    CSPClassifier,
    EEGNetClassifier,
    RiemannianMDMClassifier,
)
from aeon_neuro.transformations.collection.channel_selection import (
    DetachRocket,
    DetachRocketChannelSelector,
)
from aeon_neuro.transformations.series import EpochSeriesTransformer
from aeon_neuro.utils.discovery import all_estimators


def test_all_estimators():
    """Test return_names argument in all_estimators."""
    estimators = all_estimators()

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert estimator[0] == estimator[1].__name__
        assert "Base" not in estimator[0]
        assert issubclass(estimator[1], BaseEstimator)
        assert estimator[1].__module__.startswith("aeon_neuro.")

    estimators2 = all_estimators(return_names=False)

    assert [cls for _, cls in estimators] == estimators2

    estimators3 = all_estimators(include_sklearn=False)

    assert isinstance(estimators3, list)
    assert 0 < len(estimators3) <= len(estimators)

    for estimator in estimators3:
        assert issubclass(estimator[1], BaseAeonEstimator)


def test_all_estimators_contents():
    """Check aeon-neuro estimators are found and aeon imports and aliases are not."""
    estimators = all_estimators(return_names=False)

    assert CSPClassifier in estimators
    assert RiemannianMDMClassifier in estimators
    assert EpochSeriesTransformer in estimators

    # estimators imported from aeon are excluded
    assert HIVECOTEV2 not in estimators

    # aliases are only returned once, under the class name
    assert DetachRocket is DetachRocketChannelSelector
    names = [name for name, _ in all_estimators()]
    assert "DetachRocketChannelSelector" in names
    assert "DetachRocket" not in names
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "item",
    [
        ("classifier", BaseClassifier),
        ("transformer", BaseTransformer),
        ("collection-transformer", BaseCollectionTransformer),
        ("series-transformer", BaseSeriesTransformer),
    ],
)
def test_all_estimators_by_type(item):
    """Check that type_filter argument returns the correct type."""
    estimators = all_estimators(type_filter=item[0])

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert issubclass(estimator[1], item[1])

    estimators2 = all_estimators(type_filter=item[1])

    assert estimators == estimators2


def test_all_estimators_by_missing_type():
    """Check that a type with no aeon-neuro estimators returns an empty list."""
    assert all_estimators(type_filter="regressor") == []


@pytest.mark.parametrize(
    "input",
    [
        [BaseClassifier, BaseSeriesTransformer],
        [BaseClassifier, "collection-transformer"],
        ["series-transformer", "collection-transformer"],
    ],
)
def test_all_estimators_by_multiple_types(input):
    """Check that type_filter argument returns the correct type with multiple inputs."""
    estimators = all_estimators(type_filter=input)

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    types = tuple([i if isinstance(i, type) else BASE_CLASS_REGISTER[i] for i in input])
    for estimator in estimators:
        assert issubclass(estimator[1], types)


def test_all_estimators_invalid_type():
    """Check that invalid type_filter values raise an error."""
    with pytest.raises(ValueError, match="type_filter"):
        all_estimators(type_filter="not-a-type")
    with pytest.raises(ValueError, match="type_filter"):
        all_estimators(type_filter=int)


def test_all_estimators_exclude_type():
    """Test exclude_types argument in all_estimators."""
    estimators = all_estimators(exclude_types="transformer")

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert not issubclass(estimator[1], BaseTransformer)

    estimators2 = all_estimators(exclude_types=["series-transformer", "classifier"])

    assert isinstance(estimators2, list)
    assert 0 < len(estimators2) < len(all_estimators())

    for estimator in estimators2:
        assert not issubclass(estimator[1], (BaseSeriesTransformer, BaseClassifier))


@pytest.mark.parametrize(
    "tags",
    [
        {"capability:multivariate": True},
        {"non_deterministic": True},
        {"non_deterministic": True, "capability:multivariate": True},
    ],
)
def test_all_estimators_by_tag(tags):
    """Test ability to return estimator value of passed tags."""
    estimators = all_estimators(tag_filter=tags)

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert issubclass(estimator[1], BaseAeonEstimator)
        assert all(
            [estimator[1].get_class_tag(tag) == value for tag, value in tags.items()]
        )


def test_all_estimators_invalid_tag():
    """Check that invalid tag_filter values raise an error."""
    with pytest.raises(TypeError, match="tag_filter"):
        all_estimators(tag_filter="non_deterministic")
    with pytest.raises(ValueError, match="not a valid tag"):
        all_estimators(tag_filter={"not_a_tag": True})


def test_all_estimators_exclude_tags():
    """Test exclude_tags argument in all_estimators."""
    estimators = all_estimators(exclude_tags={"non_deterministic": True})

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert estimator[1].get_class_tag("non_deterministic") is not True

    names = [name for name, _ in estimators]
    assert "EEGNetClassifier" not in names


def test_all_estimators_filter_type_and_tag():
    """Test that type_filter and tag_filter can be used together."""
    estimators = all_estimators(
        type_filter="classifier", tag_filter={"non_deterministic": True}
    )

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert issubclass(estimator[1], BaseClassifier)
        assert estimator[1].get_class_tag("non_deterministic") is True

    # impossible combination
    estimators2 = all_estimators(
        type_filter="regressor", tag_filter={"capability:multivariate": True}
    )
    assert len(estimators2) == 0


def test_all_estimators_list_tag_lookup():
    """Check that all estimators can handle tags lists rather than single strings.

    EEGNetClassifier depends on both braindecode and torch, so it should be returned
    when filtering on either of these dependencies.
    """
    estimators = all_estimators(
        tag_filter={"python_dependencies": "braindecode"},
        return_names=False,
    )
    assert EEGNetClassifier in estimators

    estimators2 = all_estimators(
        tag_filter={"python_dependencies": "torch"},
        return_names=False,
    )
    assert EEGNetClassifier in estimators2

    estimators3 = all_estimators(
        tag_filter={"python_dependencies": ["braindecode", "pyriemann"]},
        return_names=False,
    )
    assert EEGNetClassifier in estimators3
    assert RiemannianMDMClassifier in estimators3

    assert len(estimators3) > len(estimators)

    estimators4 = all_estimators(
        tag_filter={"python_dependencies": "pyriemann"},
        return_names=False,
    )
    assert EEGNetClassifier not in estimators4

    assert len(estimators4) > 0
