"""Optional EEG side information around aeon's classifier lifecycle."""

from copy import deepcopy

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from aeon_neuro.datasets._dataset_info import EEGDatasetInfo


class BaseEEGClassifier(BaseClassifier):
    """Base classifier accepting optional aligned EEG information.

    Concrete estimators implement ``_fit(X, y)``, ``_predict(X)`` and optionally
    ``_predict_proba(X)``. During fitting they can read detached ``info_``,
    ``sfreq_``, ``channel_names_`` and ``subject_ids_``. Predict methods can read
    ``_prediction_info`` for the current call. Omitting info preserves aeon's
    ordinary collection validation and constructor settings. Supplying info
    never silently overwrites an explicit rate or channel-name parameter.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y, *, info=None):
        """Fit through aeon's base class with optional aligned EEG information.

        Parameters
        ----------
        X : np.ndarray or list
            Training collection in an aeon-supported format. With info, use
            a 3D array or list of channel/sample arrays.
        y : np.ndarray
            One classification label per training row.
        info : EEGDatasetInfo or None, default=None
            Aligned current layout and optional source/subject information.

        Returns
        -------
        BaseEEGClassifier
            Fitted estimator with detached training information.
        """
        checked = self._validate_info(X, info)
        self._pending_info = checked
        try:
            return super().fit(X, y)
        finally:
            self.__dict__.pop("_pending_info", None)

    def _fit_setup(self, X, y):
        # aeon resets the estimator inside _fit_setup. Retain the call-local
        # input across that reset, then attach it before the ordinary _fit runs.
        info = getattr(self, "_pending_info", None)
        X, y = super()._fit_setup(X, y)
        self.info_ = info
        params = self.get_params(deep=False)
        self.sfreq_ = info.sfreq if info is not None else params.get("sfreq")
        self.channel_names_ = (
            info.channel_names
            if info is not None
            else deepcopy(params.get("channel_names", params.get("ch_names")))
        )
        self.subject_ids_ = (
            np.asarray(info.participant_ids, dtype=object) if info is not None else None
        )
        return X, y

    def _validate_info(self, X, info):
        if info is None:
            return None
        if not isinstance(info, EEGDatasetInfo):
            raise ValueError("info must be an EEGDatasetInfo object or None")
        info.validate_layout(X)
        # Accessors also reject heterogeneous case layouts at the estimator
        # boundary instead of choosing an arbitrary recording's header.
        rate, names = info.sfreq, info.channel_names
        _ = info.units  # reject physically inconsistent case layouts before fitting
        params = self.get_params(deep=False)
        for key in ("sfreq", "sampling_frequency"):
            if params.get(key) is not None and params[key] != rate:
                raise ValueError(f"info sampling frequency disagrees with {key}")
        for key in ("channel_names", "ch_names"):
            if params.get(key) is not None and tuple(params[key]) != names:
                raise ValueError(f"info channel order disagrees with {key}")
        return EEGDatasetInfo(info.to_dict())

    def _prediction_context(self, X, info):
        checked = self._validate_info(X, info)
        if checked is not None and self.info_ is not None:
            if checked.sfreq != self.info_.sfreq:
                raise ValueError("Prediction sampling frequency differs from fit")
            if checked.channel_names != self.info_.channel_names:
                raise ValueError("Prediction channel names/order differ from fit")
            # Unknown units remain unknown; known changes cannot be accepted as
            # equivalent physical inputs merely because the shapes still match.
            if any(
                a is not None and b is not None and a != b
                for a, b in zip(checked.units, self.info_.units, strict=True)
            ):
                raise ValueError("Prediction channel units differ from fit")
        self._prediction_info = checked

    def predict(self, X, *, info=None):
        """Predict labels, checking supplied rate and ordered channels against fit.

        Parameters
        ----------
        X : np.ndarray or list
            Prediction collection supported by the fitted estimator.
        info : EEGDatasetInfo or None, default=None
            Information aligned with prediction rows, separate from fit rows.

        Returns
        -------
        np.ndarray
            One predicted class label per row.
        """
        self._check_is_fitted()
        self._prediction_context(X, info)
        try:
            return super().predict(X)
        finally:
            self.__dict__.pop("_prediction_info", None)

    def predict_proba(self, X, *, info=None):
        """Predict probabilities with the same optional information checks.

        Parameters
        ----------
        X : np.ndarray or list
            Prediction collection supported by the fitted estimator.
        info : EEGDatasetInfo or None, default=None
            Information aligned with prediction rows.

        Returns
        -------
        np.ndarray
            Probabilities in fitted classes_ order, one row per prediction case.
        """
        self._check_is_fitted()
        self._prediction_context(X, info)
        try:
            return super().predict_proba(X)
        finally:
            self.__dict__.pop("_prediction_info", None)

    def validation_split(
        self, y, *, validation_size=0.2, random_state=None, strategy="auto"
    ):
        """Split training rows by subject with info, or stratify without it.

        Parameters
        ----------
        y : np.ndarray
            Labels aligned with the current fitting collection.
        validation_size : float, default=0.2
            Fraction of subjects, or stratified cases without information,
            held out for validation.
        random_state : int or None, default=None
            Seed for the declared single split.
        strategy : {"auto", "subject", "stratified"}, default="auto"
            Auto holds out subjects with info and stratifies cases without it.
            Stratified explicitly allows case validation with retained metadata
            for a within-person question. Subject requires known subject IDs.

        Returns
        -------
        train : np.ndarray
            Training indices relative to the fitting collection.
        validation : np.ndarray
            Validation indices relative to the fitting collection.

        Notes
        -----
        Only the rows supplied to this fit are eligible. Subject validation needs
        every subject to be known. With subject validation,
        subjects cannot cross the training/validation boundary.
        Both partitions must contain all fitting classes. An impossible split
        raises instead of falling back to random cases or changing the protocol.
        Returned indices refer to the fitting collection, never an outer test set.
        """
        y = np.asarray(y)
        indices = np.arange(len(y))
        if strategy not in ("auto", "subject", "stratified"):
            raise ValueError("Validation strategy must be auto, subject or stratified")
        if strategy == "stratified" or (strategy == "auto" and self.info_ is None):
            return train_test_split(
                indices,
                test_size=validation_size,
                stratify=y,
                random_state=random_state,
            )
        if self.info_ is None:
            raise ValueError("Subject validation requires aligned subject information")
        if len(y) != len(self.info_):
            raise ValueError("Validation labels must match the fitting information")
        subjects = self.subject_ids_
        if any(subject is None for subject in subjects) or len(set(subjects)) < 2:
            raise ValueError("Subject validation needs at least two known subjects")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=random_state,
        )
        train, validation = next(splitter.split(indices, y, groups=subjects))
        classes = set(y)
        if set(y[train]) != classes or set(y[validation]) != classes:
            raise ValueError("The declared subject split does not retain all classes")
        return train, validation
