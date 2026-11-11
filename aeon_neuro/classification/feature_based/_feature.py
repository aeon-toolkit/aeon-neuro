"""Feature extraction classifier for aeon."""

from __future__ import annotations

__all__ = ["FeatureClassifier"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from sklearn.base import BaseEstimator, is_classifier


class FeatureClassifier(BaseClassifier):
    """Classifier built on feature extraction.

    Extracts features from multivariate equal-length time series and classifies
    them with a scikit-learn classifier, by default
    ``LinearDiscriminantAnalysis``. Features can be combined either side-by-side
    (each extractor is applied to the input and the resulting feature sets are
    concatenated) or chained (each extractor is applied to the output of the
    previous one).

    The input shape is aeon's standard collection format:
    (n_cases, n_channels, n_timepoints).

    Parameters
    ----------
    transformers : transformer or list of transformers, default=None
        Feature extractor(s) with a scikit-learn ``fit``/``transform`` interface
        (for example aeon collection transformers, scikit-learn transformers or
        pyriemann estimators). Combined according to ``extraction``.
    classifier : sklearn classifier or None, default=None
        Unfitted scikit-learn classifier fitted on the extracted features. It is
        cloned before fitting, so the object passed in is not modified. If None,
        ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis()`` is used.
    extraction : {"side_by_side", "chained"}, default="side_by_side"
        How feature extractors are combined. ``"side_by_side"`` applies each
        extractor to the input and concatenates the resulting features;
        ``"chained"`` feeds the output of each extractor into the next.
    random_state : int or None, default=None
        Seed used to set the cloned classifier's ``random_state`` parameters, if
        it has any. If None, the classifier's own setting is kept.

    Attributes
    ----------
    transformers_ : list of transformers
        The fitted feature-extraction transformers, in the order applied.
    classifier_ : sklearn classifier
        The fitted clone of ``classifier``.

    Notes
    -----
    The transformers are applied as given, and it is assumed the user knows the
    shapes involved: each transformer must accept the shape it is handed. For
    ``"chained"`` extraction this means every transformer must accept the output
    shape of the previous one; an incompatible pairing raises a ``ValueError``
    naming the offending transformer. Each transformer's output is flattened to
    2D ``(n_cases, n_features)`` before the classifier, so any shape beyond the
    case axis is acceptable for that final step.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon.transformations.collection.feature_based import Catch22
    >>> from aeon_neuro.classification.feature_based import FeatureClassifier
    >>> X, y = make_example_3d_numpy(n_cases=12, n_channels=4, random_state=0)
    >>> # side-by-side: concatenate the features from several transformers
    >>> clf = FeatureClassifier(
    ...     transformers=[Catch22(), Catch22(catch24=True)],
    ...     extraction="side_by_side",
    ... )
    >>> clf.fit(X, y)  # doctest: +SKIP
    >>> clf.predict(X)  # doctest: +SKIP
    >>> # chained: feed each transformer's output into the next
    >>> clf = FeatureClassifier(transformers=[Catch22()], extraction="chained")
    >>> clf.fit(X, y)  # doctest: +SKIP
    >>> clf.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": False,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        transformers=None,
        classifier=None,
        extraction: str = "side_by_side",
        random_state: int | None = None,
    ):
        self.transformers = transformers
        self.classifier = classifier
        self.extraction = extraction
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Extract features and fit the classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training cases.
        y : np.ndarray of shape (n_cases,)
            Class labels.

        Returns
        -------
        self : FeatureClassifier
            Fitted classifier.
        """
        if self.extraction not in ("side_by_side", "chained"):
            raise ValueError(
                "extraction must be 'side_by_side' or 'chained', got "
                f"{self.extraction!r}."
            )

        Xt = self._extract(X, y, fit=True)
        self.classifier_ = self._build_classifier()
        self.classifier_.fit(Xt, y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class of each case.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Cases to classify.

        Returns
        -------
        np.ndarray of shape (n_cases,)
            Predicted class label for each case.
        """
        return self.classifier_.predict(self._extract(X, fit=False))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each case.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Cases to classify.

        Returns
        -------
        np.ndarray of shape (n_cases, n_classes)
            Predicted class probabilities for each case.
        """
        Xt = self._extract(X, fit=False)
        if callable(getattr(self.classifier_, "predict_proba", None)):
            return self.classifier_.predict_proba(Xt)

        dists = np.zeros((len(Xt), self.n_classes_))
        for i, pred in enumerate(self.classifier_.predict(Xt)):
            dists[i, self._class_dictionary[pred]] = 1
        return dists

    def _extract(self, X: np.ndarray, y: np.ndarray | None = None, fit: bool = False):
        """Dispatch to the configured feature extraction strategy.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Cases to extract features from.
        y : np.ndarray of shape (n_cases,) or None, default=None
            Class labels, available when ``fit`` is True.
        fit : bool, default=False
            If True, fit any internal extractors and transform; if False, reuse
            the extractors fitted during ``_fit`` and only transform.

        Returns
        -------
        np.ndarray of shape (n_cases, n_features)
            The extracted feature matrix.
        """
        if self.extraction == "side_by_side":
            return self._extract_side_by_side(X, y=y, fit=fit)
        return self._extract_chained(X, y=y, fit=fit)

    def _extract_side_by_side(
        self, X: np.ndarray, y: np.ndarray | None = None, fit: bool = False
    ) -> np.ndarray:
        """Side-by-side feature extraction (features concatenated)."""
        if self.transformers is None:
            return np.empty((len(X), 0))
        transformers = (
            self.transformers
            if isinstance(self.transformers, (list, tuple))
            else [self.transformers]
        )

        if fit:
            for t in transformers:
                if not (hasattr(t, "fit") and hasattr(t, "transform")):
                    raise TypeError(f"{t!r} is not a fit/transform transformer.")
            self.transformers_ = transformers

        feats = []
        for t in self.transformers_:
            Xt = t.fit_transform(X, y) if fit else t.transform(X)
            feats.append(Xt.reshape(len(Xt), -1) if Xt.ndim > 2 else Xt)

        return np.concatenate(feats, axis=1) if feats else np.empty((len(X), 0))

    def _extract_chained(
        self, X: np.ndarray, y: np.ndarray | None = None, fit: bool = False
    ) -> np.ndarray:
        """Chained feature extraction (extractors applied sequentially)."""
        if self.transformers is None:
            return np.empty((len(X), 0))
        transformers = (
            self.transformers
            if isinstance(self.transformers, (list, tuple))
            else [self.transformers]
        )

        if fit:
            for t in transformers:
                if not (hasattr(t, "fit") and hasattr(t, "transform")):
                    raise TypeError(f"{t!r} is not a fit/transform transformer.")
            self.transformers_ = transformers

        feat = X
        for i, t in enumerate(self.transformers_):
            try:
                feat = t.fit_transform(feat, y) if fit else t.transform(feat)
            except Exception as e:
                raise ValueError(
                    f"Transformer {i} ({type(t).__name__}) failed on input of "
                    f"shape {np.asarray(feat).shape} in chained extraction. Each "
                    "transformer must accept the previous one's output shape."
                ) from e

        return feat.reshape(len(feat), -1) if feat.ndim > 2 else feat

    def _build_classifier(self):
        """Return an unfitted clone of ``self.classifier`` or the default LDA."""
        if self.classifier is None:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            classifier = LinearDiscriminantAnalysis()
        elif isinstance(self.classifier, BaseEstimator) and is_classifier(
            self.classifier
        ):
            classifier = self.classifier
        else:
            raise TypeError(
                "classifier must be a scikit-learn classifier, e.g. "
                f"LinearDiscriminantAnalysis(), or None, got {self.classifier!r}."
            )
        return _clone_estimator(classifier, self.random_state)
