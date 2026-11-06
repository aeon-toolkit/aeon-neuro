"""Experimental classifiers that reproduce NeuralBench's baseline set-ups.

These wrappers exist to run comparison experiments (tsml-eval) on the EEG
archives. They are not part of the public API and may change or be removed
once the experiments are done.

Motivation
----------
EEGBench compares general time series classifiers (HC2, QUANT, MultiROCKET-Hydra
and others) against EEG-specific methods. So far, its EEG-specific baselines have
been weak variants:

- minimum distance to mean and kNN Riemannian classifiers, rather than tangent
  space;
- CSP without band-pass filtering;
- deep learners run with aeon defaults, without input scaling or early stopping.

To make the comparison fair, we use the strongest non-foundation-model set-ups
from NeuralBench (Banville et al., 2026, arXiv:2605.08495), which are configured
by people who work on these methods:

1. ``NeuralBenchTSLRClassifier``: covariance, tangent space and logistic
   regression ("CovTsLR"). This is NeuralBench's classical baseline and mirrors
   MOABB's ``TSLR.yml`` pipeline, which is among the best classical pipelines
   in the MOABB benchmark (Chevallier et al., 2024).
2. ``NeuralBenchCTNetClassifier``: CTNet (Zhao et al., 2024) trained from
   scratch with NeuralBench's recipe. NeuralBench names CTNet as a small
   task-specific model (about 150K parameters) that performs close to the best
   EEG foundation models. Foundation models are excluded because several were
   pretrained on datasets used for downstream evaluation (NeuralBench flags but
   does not remove these cases); a model trained from scratch avoids that
   leakage.
3. ``MOABBTSLRClassifier``: MOABB's own ``TSLR.yml`` pipeline with its motor
   imagery band-pass (8-32 Hz), and none of NeuralBench's additions. It gives
   numbers directly comparable to MOABB's published table, and shows how much
   NeuralBench's version changes. It is meant for motor imagery problems,
   since the band is specific to motor imagery.

The NeuralBench settings follow its code (``neuralbench/defaults/config.yaml``,
``defaults/eeg/config.yaml``, the classification task configs and
``neuralbench/baselines.py``, version 0.4.0) rather than the paper text where
the two differ. The paper says 50 epochs and patience 10; the code uses 40 and
5.

Differences from NeuralBench
----------------------------
NeuralBench preprocesses whole continuous recordings and then cuts windows. The
EEG archives hold pre-cut cases with no recording or subject IDs, so the
preprocessing is applied per case and some steps are adapted:

- **High-pass at 0.1 Hz** is replaced by removing each channel's mean in each
  case. A 0.1 Hz FIR filter is about 33 s long, much longer than most cases;
  NeuralBench itself refuses to filter segments shorter than its filter. On
  short cases, removing the mean is what such a filter would mainly do.
- **Notch filters** are zero-phase IIR notches (``scipy.signal.iirnotch``, Q=30)
  rather than MNE's FIR notch, which also needs several seconds of signal.
  Harmonics of 50 and 60 Hz are notched below the lower of the input and
  output Nyquist frequencies; resampling removes anything higher.
- **Low-pass at 75 Hz** is skipped when resampling to 120 Hz, because
  resampling's anti-aliasing already removes everything above 60 Hz.
- **Robust scaling** uses each channel's median and interquartile range over
  all training cases and time points, applied unchanged to test cases.
  NeuralBench scales each recording separately, which also normalises
  differences between subjects; that needs recording IDs, which the archives
  do not have (see issue #283). Scaling over the training set keeps power
  differences between cases, which motor imagery and clinical problems rely on.
- **Validation split** for CTNet's early stopping holds out 25% of the
  training subjects when subject IDs are given through ``train_groups``, as
  NeuralBench does (its 60/20/20 train/validation/test split by subject
  gives 25% of the non-test subjects). Most archive problems have subject IDs
  in their sidecar files. Without them, 25% of the training cases are held out
  at random, stratified by class; the validation set then shares subjects with
  the training set, so early stopping may stop late.
- **Windows** are the archive's cases as given, not NeuralBench's task-specific
  windows (for example 4 s from the cue for motor imagery).
- **Class weights and label smoothing** (balanced weights, smoothing 0.1) are
  used for every problem. NeuralBench sets them per task, but uses this
  combination for nearly every single-label classification task, including all
  clinical ones.
- **Short cases**: CTNet needs at least 64 time points (0.53 s at 120 Hz).
  Shorter cases are zero padded at the end after scaling, where zero is the
  training median.
"""

from __future__ import annotations

__all__ = [
    "MOABBTSLRClassifier",
    "NeuralBenchCTNetClassifier",
    "NeuralBenchPreprocessor",
    "NeuralBenchTSLRClassifier",
]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs
from sklearn.utils import check_random_state

# NeuralBench's EEG extractor settings (defaults/eeg/config.yaml).
_TARGET_SFREQ = 120.0
_NOTCH_FREQS = (50.0, 60.0)
_H_FREQ = 75.0
_CLAMP = 20.0
# NeuralBench caps notch harmonics at 300 Hz.
_MAX_NOTCH = 301.0
# CTNet's two pooling layers (8 x 8) need at least this many time points.
_CTNET_MIN_TIMEPOINTS = 64
# NeuralBench's LogisticRegressionCV grid (baselines.py).
_LR_CS = tuple(10.0**k for k in range(-3, 4))


class NeuralBenchPreprocessor(BaseCollectionTransformer):
    """Preprocess EEG cases as NeuralBench does, adapted to pre-cut cases.

    Steps, in NeuralBench's order:

    1. remove each channel's mean in each case (in place of a 0.1 Hz high-pass);
    2. notch filter harmonics of ``notch_freqs``;
    3. low-pass at ``h_freq`` if it is below the output Nyquist frequency;
    4. resample to ``target_sfreq``;
    5. robust scale each channel (median and interquartile range);
    6. clamp to ``[-clamp, clamp]``.

    See the module docstring for why each step differs from NeuralBench.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input in Hz. Required.
    target_sfreq : float or None, default=120.0
        Output sampling frequency. If None, the data is not resampled.
    demean : bool, default=True
        If True, remove each channel's mean in each case.
    notch_freqs : tuple of float, default=(50.0, 60.0)
        Base frequencies to notch, with their harmonics. Empty to skip.
    h_freq : float or None, default=75.0
        Low-pass cutoff in Hz, applied only below the output Nyquist frequency.
    scaling : {"train", "case", None}, default="train"
        ``"train"`` scales each channel by its median and interquartile range
        over the training cases. ``"case"`` uses each case's own statistics.
        None skips scaling.
    clamp : float or None, default=20.0
        Clamp the scaled data to this absolute value. None to skip.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "fit_is_empty": False,
        "output_data_type": "Collection",
    }

    def __init__(
        self,
        sfreq: float | None = None,
        target_sfreq: float | None = _TARGET_SFREQ,
        demean: bool = True,
        notch_freqs: tuple = _NOTCH_FREQS,
        h_freq: float | None = _H_FREQ,
        scaling: str | None = "train",
        clamp: float | None = _CLAMP,
    ):
        self.sfreq = sfreq
        self.target_sfreq = target_sfreq
        self.demean = demean
        self.notch_freqs = notch_freqs
        self.h_freq = h_freq
        self.scaling = scaling
        self.clamp = clamp
        super().__init__()

    def _fit(self, X, y=None):
        self._fit_transform(X, y)
        return self

    def _fit_transform(self, X, y=None):
        self._check_params()
        X = self._filter_and_resample(X)
        if self.scaling == "train":
            # sklearn's RobustScaler statistics, one per channel
            pooled = np.moveaxis(X, 1, 0).reshape(X.shape[1], -1)
            q25, self.center_, q75 = np.percentile(pooled, [25, 50, 75], axis=1)
            scale = q75 - q25
            scale[scale == 0] = 1.0
            self.scale_ = scale
        return self._scale_and_clamp(X)

    def _transform(self, X, y=None):
        return self._scale_and_clamp(self._filter_and_resample(X))

    def _check_params(self):
        if self.sfreq is None or not self.sfreq > 0:
            raise ValueError(f"sfreq must be a positive number, got {self.sfreq!r}.")
        if self.target_sfreq is not None and not self.target_sfreq > 0:
            raise ValueError(
                f"target_sfreq must be positive or None, got {self.target_sfreq!r}."
            )
        if self.scaling not in ("train", "case", None):
            raise ValueError(
                f"scaling must be 'train', 'case' or None, got {self.scaling!r}."
            )

    @property
    def output_sfreq(self):
        """Sampling frequency of the transformed data."""
        return self.sfreq if self.target_sfreq is None else self.target_sfreq

    def _filter_and_resample(self, X):
        import mne
        from scipy.signal import filtfilt, iirnotch

        X = np.asarray(X, dtype=np.float64)
        if self.demean:
            X = X - X.mean(axis=2, keepdims=True)

        # frequencies above the output Nyquist frequency are removed by
        # resampling, so only notch and low-pass below it
        nyquist = min(self.sfreq, self.output_sfreq) / 2
        for base in self.notch_freqs:
            for freq in np.arange(base, min(nyquist, _MAX_NOTCH), base):
                b, a = iirnotch(freq, Q=30.0, fs=self.sfreq)
                X = filtfilt(b, a, X, axis=-1)

        if self.h_freq is not None and self.h_freq < nyquist:
            X = mne.filter.filter_data(
                X, self.sfreq, l_freq=None, h_freq=self.h_freq, verbose=False
            )

        if self.target_sfreq is not None and self.target_sfreq != self.sfreq:
            X = mne.filter.resample(
                X,
                up=float(self.target_sfreq),
                down=float(self.sfreq),
                axis=-1,
                verbose=False,
            )
        return X

    def _scale_and_clamp(self, X):
        if self.scaling == "train":
            X = (X - self.center_[None, :, None]) / self.scale_[None, :, None]
        elif self.scaling == "case":
            q25, median, q75 = np.percentile(X, [25, 50, 75], axis=2, keepdims=True)
            scale = q75 - q25
            scale[scale == 0] = 1.0
            X = (X - median) / scale
        if self.clamp is not None:
            X = np.clip(X, -self.clamp, self.clamp)
        return X

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return [
            {"sfreq": 100.0},
            {"sfreq": 100.0, "target_sfreq": None, "scaling": "case", "clamp": None},
        ]


class NeuralBenchTSLRClassifier(BaseClassifier):
    """Tangent space logistic regression, as NeuralBench's CovTsLR baseline.

    The pipeline is ``Covariances(oas) -> Shrinkage(0.1) -> TangentSpace(riemann)
    -> StandardScaler -> LogisticRegressionCV``, fit on preprocessed data
    (``NeuralBenchPreprocessor`` without clamping). It follows NeuralBench's
    ``CovTsLR`` (``neuralbench/baselines.py``), which mirrors MOABB's
    ``TSLR.yml``. See the module docstring for why this set-up is used and how
    the preprocessing differs from NeuralBench.

    As in NeuralBench:

    - clamping is skipped, because clipped trials can give rank-deficient
      covariance matrices;
    - training cases with variance at most ``min_trial_variance`` are dropped;
    - if the Riemannian mean fails because covariances are not positive
      definite, the fit is retried with the log-Euclidean and then Euclidean
      metric;
    - test cases whose tangent space features are not finite get zero
      features, so they receive the training-mean prediction.

    Unlike NeuralBench, all training cases are used. NeuralBench subsamples
    to at most 20,000, stratified by class, to limit memory and because the
    iterative Riemannian mean failed on one very large task; the metric
    fallback above covers that failure.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input in Hz. Required when ``preprocess`` is
        True.
    preprocess : bool, default=True
        If True, apply ``NeuralBenchPreprocessor`` (without clamping) first.
    cov_estimator : str, default="oas"
        Covariance estimator for ``pyriemann.estimation.Covariances``.
    shrinkage : float, default=0.1
        Shrinkage for ``pyriemann.estimation.Shrinkage``.
    metric : str, default="riemann"
        First metric tried for ``pyriemann.tangentspace.TangentSpace``.
    Cs : tuple of float, default=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)
        Inverse regularisation strengths searched by ``LogisticRegressionCV``.
    cv : int, default=5
        Number of stratified folds used to choose ``C``.
    min_trial_variance : float, default=1e-8
        Training cases with variance at or below this value are dropped.
    n_jobs : int, default=1
        Number of jobs for ``LogisticRegressionCV``.
    random_state : int or None, default=None
        Seed for breaking ties between equally probable classes in predict.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        sfreq: float | None = None,
        preprocess: bool = True,
        cov_estimator: str = "oas",
        shrinkage: float = 0.1,
        metric: str = "riemann",
        Cs: tuple = _LR_CS,
        cv: int = 5,
        min_trial_variance: float = 1e-8,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.sfreq = sfreq
        self.preprocess = preprocess
        self.cov_estimator = cov_estimator
        self.shrinkage = shrinkage
        self.metric = metric
        self.Cs = Cs
        self.cv = cv
        self.min_trial_variance = min_trial_variance
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        if self.preprocess:
            self.preprocessor_ = NeuralBenchPreprocessor(sfreq=self.sfreq, clamp=None)
            X = self.preprocessor_.fit_transform(X)

        variance = X.reshape(X.shape[0], -1).var(axis=1)
        keep = variance > self.min_trial_variance
        X, y = X[keep], y[keep]
        self.n_dropped_ = int((~keep).sum())

        error = None
        for metric in dict.fromkeys([self.metric, "logeuclid", "euclid"]):
            try:
                self.pipeline_ = self._build_pipeline(metric).fit(X, y)
                self.metric_ = metric
                return self
            except ValueError as err:
                if "positive definite" not in str(err):
                    raise
                error = err
        raise error

    def _predict_proba(self, X):
        if self.preprocess:
            X = self.preprocessor_.transform(X)
        features = X
        for _, step in self.pipeline_.steps[:-1]:
            features = step.transform(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        final = self.pipeline_.steps[-1][1]
        # final.classes_ are the labels seen in fit, in the same sorted order
        # as self.classes_ unless dropping flat cases removed a class
        proba = np.zeros((len(X), self.n_classes_))
        columns = [self._class_dictionary[c] for c in final.classes_]
        proba[:, columns] = final.predict_proba(features)
        return proba

    def _predict(self, X):
        proba = self._predict_proba(X)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(p == p.max())))]
                for p in proba
            ]
        )

    def _build_pipeline(self, metric):
        from pyriemann.estimation import Covariances, Shrinkage
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            Covariances(estimator=self.cov_estimator),
            Shrinkage(shrinkage=self.shrinkage),
            TangentSpace(metric=metric),
            StandardScaler(),
            LogisticRegressionCV(
                Cs=list(self.Cs),
                cv=self.cv,
                max_iter=1000,
                class_weight="balanced",
                scoring="accuracy",
                n_jobs=check_n_jobs(self.n_jobs),
            ),
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return [
            {"sfreq": 100.0, "Cs": (0.1, 1.0), "cv": 2, "random_state": 0},
            {"preprocess": False, "Cs": (1.0,), "cv": 2, "random_state": 0},
        ]


class MOABBTSLRClassifier(BaseClassifier):
    """MOABB's tangent space logistic regression pipeline (``TSLR.yml``).

    Band-pass filters each case to ``fmin``-``fmax`` Hz as MOABB's motor imagery
    paradigm does (MNE's default IIR filter, a zero-phase 4th-order
    Butterworth), then fits ``Covariances("oas") -> TangentSpace("riemann") ->
    LogisticRegression(C=1.0)`` with sklearn's defaults otherwise. There is no
    resampling, scaling, shrinkage, feature standardisation, class weighting
    or tuning of ``C``.

    See the module docstring for why this is run alongside
    ``NeuralBenchTSLRClassifier``.

    Differences from MOABB: MOABB filters the continuous recording before
    cutting epochs; here each case is filtered on its own, so the filter's edge
    effects are larger on short cases. Evaluation follows the archive's splits,
    not MOABB's within-session cross-validation.

    Parameters
    ----------
    sfreq : float or None
        Sampling frequency of the input in Hz. Required unless both ``fmin``
        and ``fmax`` are None.
    fmin : float or None, default=8.0
        Low cutoff in Hz. None for no high-pass.
    fmax : float or None, default=32.0
        High cutoff in Hz. None for no low-pass.
    C : float, default=1.0
        Inverse regularisation strength of the logistic regression.
    random_state : int or None, default=None
        Seed for breaking ties between equally probable classes in predict.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        sfreq: float | None = None,
        fmin: float | None = 8.0,
        fmax: float | None = 32.0,
        C: float = 1.0,
        random_state: int | None = None,
    ):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.C = C
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipeline_ = make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            LogisticRegression(C=self.C),
        ).fit(self._band_pass(X), y)
        return self

    def _predict_proba(self, X):
        return self.pipeline_.predict_proba(self._band_pass(X))

    def _predict(self, X):
        proba = self._predict_proba(X)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(p == p.max())))]
                for p in proba
            ]
        )

    def _band_pass(self, X):
        import mne

        X = np.asarray(X, dtype=np.float64)
        if self.fmin is None and self.fmax is None:
            return X
        if self.sfreq is None or not self.sfreq > 0:
            raise ValueError(f"sfreq must be a positive number, got {self.sfreq!r}.")
        return mne.filter.filter_data(
            X,
            self.sfreq,
            l_freq=self.fmin,
            h_freq=self.fmax,
            method="iir",
            verbose=False,
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return [
            {"sfreq": 100.0, "random_state": 0},
            {"fmin": None, "fmax": None, "C": 0.5, "random_state": 0},
        ]


class NeuralBenchCTNetClassifier(BaseClassifier):
    """CTNet trained from scratch with NeuralBench's recipe.

    The data is preprocessed with ``NeuralBenchPreprocessor`` (120 Hz, robust
    scaling, clamping at 20), then CTNet from braindecode is trained through
    ``BraindecodeClassifier`` with NeuralBench's settings:

    - AdamW, learning rate 1e-4, weight decay 0.05;
    - one-cycle schedule with 10% warmup and cosine annealing, stepped per
      batch;
    - gradient norm clipping at 1.0 and batch size 64;
    - up to 40 epochs, with early stopping on validation balanced accuracy
      (patience 5), keeping the best epoch;
    - cross-entropy with balanced class weights and label smoothing 0.1.

    See the module docstring for why this set-up is used and how it differs from
    NeuralBench. Pass subject IDs as ``train_groups`` so the validation split
    is by subject, as in NeuralBench; without them it is by case.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input in Hz. Required.
    n_epochs : int, default=40
        Maximum number of epochs.
    early_stopping_patience : int, default=5
        Epochs without improvement in validation balanced accuracy before
        stopping.
    validation_size : float, default=0.25
        Fraction of training subjects (or, without ``train_groups``, training
        cases) held out for early stopping.
    train_groups : array-like or None, default=None
        Subject ID of each training case, in the order passed to ``fit``. If
        given, whole subjects are held out for validation. If None, or if
        there is only one subject, cases are held out at random. Set it for
        each training set; it is not used in predict.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-4
        Peak learning rate.
    weight_decay : float, default=0.05
        AdamW weight decay.
    model_params : dict or None, default=None
        Extra keyword arguments for braindecode's CTNet.
    device : str or None, default=None
        Torch device. If None, uses CUDA when available.
    verbose : bool, default=False
        If True, print training progress.
    random_state : int or None, default=None
        Random seed.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": ["braindecode", "torch"],
    }

    def __init__(
        self,
        sfreq: float | None = None,
        n_epochs: int = 40,
        early_stopping_patience: int = 5,
        validation_size: float = 0.25,
        train_groups=None,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        model_params: dict | None = None,
        device: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.sfreq = sfreq
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_size = validation_size
        self.train_groups = train_groups
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_params = model_params
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from aeon_neuro.classification.deep_learning import BraindecodeClassifier

        self.preprocessor_ = NeuralBenchPreprocessor(sfreq=self.sfreq)
        X = self._pad(self.preprocessor_.fit_transform(X))
        self.classifier_ = BraindecodeClassifier(
            model="CTNet",
            model_params=self.model_params,
            sfreq=self.preprocessor_.output_sfreq,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            optimizer="adamw",
            weight_decay=self.weight_decay,
            device=self.device,
            verbose=self.verbose,
            random_state=self.random_state,
            lr_scheduler="onecycle",
            gradient_clip=1.0,
            class_weight="balanced",
            label_smoothing=0.1,
            validation_size=self.validation_size,
            early_stopping_patience=self.early_stopping_patience,
            train_groups=self.train_groups,
        )
        self.classifier_.fit(X, y)
        return self

    def _predict_proba(self, X):
        X = self._pad(self.preprocessor_.transform(X))
        return self.classifier_.predict_proba(X)

    def _predict(self, X):
        X = self._pad(self.preprocessor_.transform(X))
        return self.classifier_.predict(X)

    @staticmethod
    def _pad(X):
        """Zero pad cases shorter than CTNet's minimum length at the end."""
        short = _CTNET_MIN_TIMEPOINTS - X.shape[2]
        if short > 0:
            X = np.pad(X, ((0, 0), (0, 0), (0, short)))
        return X

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "sfreq": 100.0,
            "n_epochs": 2,
            "early_stopping_patience": 1,
            "validation_size": 0.3,
            "batch_size": 4,
            "model_params": {
                "num_layers": 1,
                "num_heads": 2,
                "embed_dim": 8,
                "kernel_size": 8,
            },
            "random_state": 0,
        }
