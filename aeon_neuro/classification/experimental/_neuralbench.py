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

The classifiers already in aeon-neuro are configured for experiments the same
way, with an external reference set-up rather than their defaults:

4. ``NeuralBenchDeepClassifier``: any braindecode model with the same recipe
   as CTNet. NeuralBench trains EEGNet, Deep4Net (aeon-neuro's
   ``DeepConvNetClassifier``), EEGConformer and ShallowFBCSPNet this way, with
   braindecode's default settings. These replace runs of aeon-neuro's
   ``EEGNetClassifier``, ``DeepConvNetClassifier`` and ``EEGConformerClassifier``
   with their defaults (200 epochs of Adam, no scaling or early stopping).
5. ``MOABBPipelineClassifier``: MOABB's published pipelines, each with MOABB's
   band-pass. ``"MDM"`` is ``RiemannianMDMClassifier`` with OAS covariances,
   ``"CSP_LDA"`` and ``"CSP_SVM"`` are MOABB's CSP pipelines (pyriemann's CSP,
   which differs from the MNE CSP in ``CSPClassifier``), and ``"XdawnCov_MDM"``
   is MOABB's ERP pipeline, which replaces ``XDAWNClassifier`` on ERP problems.

Not configured: the foundation models (``CBraModClassifier``,
``LabramClassifier``), because of pretraining leakage; and
``RiemannianKNNClassifier``, which has no reference set-up in MOABB or
NeuralBench.

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
    "MOABBPipelineClassifier",
    "MOABBTSLRClassifier",
    "NeuralBenchCTNetClassifier",
    "NeuralBenchDeepClassifier",
    "NeuralBenchPreprocessor",
    "NeuralBenchTSLRClassifier",
]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs
from pyriemann.spatialfilters import CSP
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state

# NeuralBench's EEG extractor settings (defaults/eeg/config.yaml).
_TARGET_SFREQ = 120.0
_NOTCH_FREQS = (50.0, 60.0)
_H_FREQ = 75.0
_CLAMP = 20.0
# NeuralBench caps notch harmonics at 300 Hz.
_MAX_NOTCH = 301.0
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


# MOABB pipelines (github.com/NeuroTechX/moabb/tree/develop/pipelines) and the
# paradigms they are run with in MOABB's benchmark. The SSVEP pipelines are not
# included: they need each case's stimulation frequencies, which the archives do
# not hold.
MOABB_PIPELINES = {
    # motor imagery (8-32 Hz band-pass)
    "TSLR": "TSLR.yml: Covariances(oas) -> TangentSpace(riemann) -> "
    "LogisticRegression(C=1.0)",
    "TSSVM": "TSSVM_grid.yml: Covariances(oas) -> TangentSpace(riemann) -> SVC, "
    "grid search over kernel in (rbf, linear), C=1",
    "TSEN": "EN_grid.yml: Covariances(oas) -> TangentSpace(riemann) -> "
    "LogisticRegression(elasticnet, saga), grid search over l1_ratio",
    "MDM": "MDM.yml: Covariances(oas) -> MDM(riemann)",
    "FgMDM": "FgMDM.yml: Covariances(oas) -> FgMDM(riemann)",
    "CSP_LDA": "CSP.yml: Covariances(oas) -> CSP(nfilter=6) -> "
    "LinearDiscriminantAnalysis(solver=svd)",
    "CSP_SVM": "CSP_SVM_grid.yml: Covariances(oas) -> CSP -> SVC, grid search over "
    "nfilter 2-8 and kernel in (rbf, linear), C=1",
    "DLCSPauto_shLDA": "regCSP+shLDA.yml: Covariances(oas) -> CSP(nfilter=6) -> "
    "LinearDiscriminantAnalysis(solver=lsqr, shrinkage=auto)",
    "TRCSP_LDA": "WTRCSP.py: Covariances(scm) -> TRCSP(nfilter=6) -> "
    "LinearDiscriminantAnalysis, two classes only",
    "FBCSP_SVM": "FBCSP.py: filter bank of six 4 Hz bands from 8 to 32 Hz, "
    "Covariances(oas) -> CSP(nfilter=4) per band -> SelectKBest(mutual "
    "information, k=10) -> SVC(linear), C tuned over logspace(-2, 2, 10)",
    "LogVar_LDA": "LogVar.yml: log variance -> LinearDiscriminantAnalysis(svd)",
    "LogVar_SVM": "LogVar_grid.yml: log variance -> SVC(linear), grid search over "
    "C in (1, 5, 10, 50, 100)",
    "AUG_TANG_SVM": "AUG_TANG_SVM_grid.yml: augmented (time-delay embedded) "
    "signal -> Covariances(cov) -> TangentSpace(riemann) -> SVC, grid search "
    "over order 1-10, lag 1-10 and kernel in (rbf, linear), C=1",
    # P300/ERP (1-24 Hz band-pass)
    "XdawnCov_MDM": "XdawnCov_MDM.yml: XdawnCovariances(oas, xdawn_estimator=scm) "
    "-> MDM",
    "XdawnCov_TS_SVM": "XdawnCov_TS_SVM.yml: XdawnCovariances(oas, "
    "xdawn_estimator=scm) -> TangentSpace -> SVC, grid search over kernel in "
    "(rbf, linear), C=10",
    "ERPCov_MDM": "ERPCov_MDM.yml: ERPCovariances(oas) -> MDM",
    "xDAWN_LDA": "xDAWN_LDA.yml: Xdawn(oas) -> Vectorizer -> "
    "LinearDiscriminantAnalysis",
}

# The bands of MOABB's FilterBankMotorImagery paradigm, used by FBCSP_SVM.
MOABB_FILTER_BANK = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32))


class _LogVariance(BaseEstimator, TransformerMixin):
    """MOABB's LogVariance: the log of each channel's variance."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(np.var(X, -1))


class _AugmentedDataset(BaseEstimator, TransformerMixin):
    """MOABB's AugmentedDataset: stack ``order`` copies lagged by ``lag``.

    Carrara and Papadopoulo (2023), arXiv:2302.04508. The output has ``order``
    times the channels and ``(order - 1) * lag`` fewer time points.
    """

    def __init__(self, order=1, lag=1):
        self.order = order
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.order == 1:
            return X
        return np.concatenate(
            [
                X[:, :, p * self.lag : -(self.order - p) * self.lag]
                for p in range(self.order)
            ],
            axis=1,
        )


class _TRCSP(CSP):
    """MOABB's TRCSP: weighted Tikhonov-regularised CSP, two classes only.

    Lotte and Guan (2011). Copied from ``moabb.pipelines.csp`` so that MOABB is
    not needed.
    """

    def __init__(self, nfilter=4, metric="euclid", log=True, alpha=1):
        super().__init__(nfilter, metric, log)
        self.alpha = alpha

    def fit(self, X, y):
        from pyriemann.utils.mean import mean_covariance
        from scipy import linalg

        X, y = np.asarray(X), np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"TRCSP only supports 2 classes, got {len(classes)}.")
        C = np.array([mean_covariance(X[y == c], metric=self.metric) for c in classes])
        filters, patterns = [], []
        # as MOABB: filters from C1 against regularised C0, then C0 against C1
        for a, b in ((C[1], C[0]), (C[0], C[1])):
            evals, evecs = linalg.eigh(a, b + np.eye(len(b)) * self.alpha)
            evecs = evecs[:, np.argsort(evals)[::-1]]
            A = np.linalg.pinv(evecs.T)
            filters.append(evecs[:, : self.nfilter // 2])
            patterns.append(A[:, : self.nfilter // 2])
        self.filters_ = np.concatenate(filters, axis=1).T
        self.patterns_ = np.concatenate(patterns, axis=1).T
        return self


class _FilterBank(BaseEstimator, TransformerMixin):
    """MOABB's FilterBank: fit ``estimator`` on each band (the last axis).

    The per-band outputs are concatenated on the feature axis.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.models_ = [
            clone(self.estimator).fit(X[..., i], y) for i in range(X.shape[-1])
        ]
        return self

    def transform(self, X):
        return np.concatenate(
            [m.transform(X[..., i]) for i, m in enumerate(self.models_)], axis=1
        )


class _MutualInfo:
    """``mutual_info_classif`` with a fixed seed, as a picklable score function."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def __call__(self, X, y):
        from sklearn.feature_selection import mutual_info_classif

        return mutual_info_classif(X, y, random_state=self.random_state)


class MOABBPipelineClassifier(BaseClassifier):
    """A MOABB benchmark pipeline, with MOABB's band-pass filter.

    Band-pass filters each case to ``fmin``-``fmax`` Hz as MOABB's paradigms do
    (MNE's default IIR filter, a zero-phase 4th-order Butterworth), then fits
    one of MOABB's published pipelines with exactly its settings. There is no
    resampling or scaling, as in MOABB.

    ``pipeline`` is one of the keys of ``MOABB_PIPELINES``, named after
    MOABB's pipeline files. Motor imagery pipelines:

    - ``"TSLR"``: ``TSLR.yml``, ``Covariances(oas) -> TangentSpace(riemann) ->
      LogisticRegression(C=1.0)``.
    - ``"TSSVM"``: ``TSSVM_grid.yml``, ``Covariances(oas) ->
      TangentSpace(riemann) -> SVC(C=1)``, kernel tuned over (rbf, linear).
    - ``"TSEN"``: ``EN_grid.yml``, ``Covariances(oas) -> TangentSpace(riemann)
      -> LogisticRegression`` with an elastic net penalty (saga solver,
      ``intercept_scaling=1000``, ``max_iter=1000``), ``l1_ratio`` tuned over
      (0.2, 0.3, 0.45, 0.65, 0.75).
    - ``"MDM"``: ``MDM.yml``, ``Covariances(oas) -> MDM(riemann)``. This is the
      same as aeon-neuro's ``RiemannianMDMClassifier`` with
      ``covariance_estimator="oas"``.
    - ``"FgMDM"``: ``FgMDM.yml``, ``Covariances(oas) -> FgMDM(riemann)``.
    - ``"CSP_LDA"``: ``CSP.yml``, ``Covariances(oas) -> CSP(nfilter=6) ->
      LinearDiscriminantAnalysis(solver="svd")``. This is pyriemann's CSP on
      covariance matrices, as in MOABB, not MNE's CSP, which aeon-neuro's
      ``CSPClassifier`` uses.
    - ``"CSP_SVM"``: ``CSP_SVM_grid.yml``, ``Covariances(oas) -> CSP ->
      SVC(C=1)``, tuned over CSP filters 2-8 and kernel in (rbf, linear).
    - ``"DLCSPauto_shLDA"``: ``regCSP+shLDA.yml``, ``Covariances(oas) ->
      CSP(nfilter=6) -> LinearDiscriminantAnalysis(solver="lsqr",
      shrinkage="auto")``.
    - ``"TRCSP_LDA"``: ``WTRCSP.py``, ``Covariances(scm) -> TRCSP(nfilter=6)
      -> LinearDiscriminantAnalysis``, weighted Tikhonov-regularised CSP. Two
      classes only; more raise a ``ValueError``.
    - ``"FBCSP_SVM"``: ``FBCSP.py``, a filter bank of MOABB's six 4 Hz bands
      from 8 to 32 Hz (``MOABB_FILTER_BANK``, which replaces ``fmin`` and
      ``fmax``), ``Covariances(oas) -> CSP(nfilter=4)`` on each band, then
      ``SelectKBest(mutual_info_classif, k=10) -> SVC(linear)`` with ``C``
      tuned over ``logspace(-2, 2, 10)`` by 5-fold cross-validation inside the
      pipeline, as in MOABB.
    - ``"LogVar_LDA"``: ``LogVar.yml``, the log variance of each channel ->
      ``LinearDiscriminantAnalysis(solver="svd")``.
    - ``"LogVar_SVM"``: ``LogVar_grid.yml``, log variance -> ``SVC(linear)``,
      ``C`` tuned over (1, 5, 10, 50, 100).
    - ``"AUG_TANG_SVM"``: ``AUG_TANG_SVM_grid.yml``, the signal augmented with
      ``order`` copies lagged by ``lag`` (Carrara and Papadopoulo, 2023) ->
      ``Covariances(cov) -> TangentSpace(riemann) -> SVC(C=1)``, tuned over
      order 1-10, lag 1-10 and kernel in (rbf, linear): 200 settings, so this
      is by far the slowest, and an order of 10 makes covariance matrices of
      ten times the channels.

    P300/ERP pipelines, to run with ``fmin=1`` and ``fmax=24``:

    - ``"XdawnCov_MDM"``: ``XdawnCov_MDM.yml``,
      ``XdawnCovariances(oas, xdawn_estimator=scm) -> MDM``, MOABB's best
      pipeline type for P300/ERP problems. It needs at least two channels.
    - ``"XdawnCov_TS_SVM"``: ``XdawnCov_TS_SVM.yml``,
      ``XdawnCovariances(oas, xdawn_estimator=scm) -> TangentSpace ->
      SVC(C=10)``, kernel tuned over (rbf, linear).
    - ``"ERPCov_MDM"``: ``ERPCov_MDM.yml``, ``ERPCovariances(oas) -> MDM``.
    - ``"xDAWN_LDA"``: ``xDAWN_LDA.yml``, ``Xdawn(oas) -> Vectorizer ->
      LinearDiscriminantAnalysis``.

    MOABB's SSVEP pipelines are not included, since they need each case's
    stimulation frequencies.

    The grid searches (the ``_grid`` files) follow MOABB's evaluations: 3-fold
    stratified shuffled cross-validation scored by ROC AUC for two classes and
    accuracy otherwise, then a refit on all the training data. Their SVMs use
    ``probability=True`` for ``predict_proba``, which adds Platt scaling but
    leaves the fitted SVM and its predictions unchanged.

    MOABB filters with a band chosen by paradigm: 8-32 Hz for motor imagery
    (the default here), 1-24 Hz for P300/ERP and 7-45 Hz for SSVEP. Set
    ``fmin`` and ``fmax`` to match the problem.

    See the module docstring for why these are run. Differences from MOABB:
    MOABB filters the continuous recording before cutting epochs; here each case
    is filtered on its own, so the filter's edge effects are larger on short
    cases. Evaluation follows the archive's splits, not MOABB's within-session
    cross-validation.

    Parameters
    ----------
    pipeline : str, default="TSLR"
        A key of ``MOABB_PIPELINES``, e.g. ``"TSLR"``, ``"CSP_LDA"`` or
        ``"XdawnCov_MDM"``.
    sfreq : float or None
        Sampling frequency of the input in Hz. Required unless both ``fmin``
        and ``fmax`` are None.
    fmin : float or None, default=8.0
        Low cutoff in Hz. None for no high-pass.
    fmax : float or None, default=32.0
        High cutoff in Hz. None for no low-pass.
    n_jobs : int, default=1
        Number of jobs for MDM, FgMDM and the grid searches.
    random_state : int or None, default=None
        Seed for the grid search cross-validation, Platt scaling, mutual
        information and the saga solver.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        pipeline: str = "TSLR",
        sfreq: float | None = None,
        fmin: float | None = 8.0,
        fmax: float | None = 32.0,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.pipeline = pipeline
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        self.pipeline_ = self._build_pipeline().fit(self._band_pass(X), y)
        return self

    def _predict_proba(self, X):
        # columns follow the pipeline's classes_, which are sorted like
        # self.classes_
        return self.pipeline_.predict_proba(self._band_pass(X))

    def _predict(self, X):
        # labels come from the pipeline's own predict, as in MOABB; for CSP_SVM
        # that is the SVM decision function, not the Platt probabilities
        return self.pipeline_.predict(self._band_pass(X))

    def _build_pipeline(self):
        from mne.decoding import Vectorizer
        from pyriemann.classification import MDM, FgMDM
        from pyriemann.estimation import (
            Covariances,
            ERPCovariances,
            XdawnCovariances,
        )
        from pyriemann.spatialfilters import Xdawn
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.feature_selection import SelectKBest
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC

        n_jobs = check_n_jobs(self.n_jobs)
        rs = self.random_state

        def svc(**kwargs):
            # probability=True only adds Platt scaling for predict_proba; the
            # fitted SVM and predict are unchanged
            return SVC(probability=True, random_state=rs, **kwargs)

        def grid(steps, param_grid):
            return GridSearchCV(
                Pipeline(steps),
                param_grid,
                cv=StratifiedKFold(3, shuffle=True, random_state=rs),
                scoring="roc_auc" if self.n_classes_ == 2 else "accuracy",
                refit=True,
                n_jobs=n_jobs,
            )

        def oas():
            return ("covariances", Covariances(estimator="oas"))

        def ts():
            return ("tangentspace", TangentSpace(metric="riemann"))

        kernels = ["rbf", "linear"]
        p = self.pipeline
        if p == "TSLR":
            steps = [oas(), ts(), ("logisticregression", LogisticRegression(C=1.0))]
        elif p == "TSSVM":
            return grid(
                [oas(), ts(), ("svc", svc(kernel="linear"))],
                {"svc__C": [1], "svc__kernel": kernels},
            )
        elif p == "TSEN":
            lr = LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.70,
                intercept_scaling=1000.0,
                solver="saga",
                max_iter=1000,
                random_state=rs,
            )
            return grid(
                [oas(), ts(), ("logisticregression", lr)],
                {"logisticregression__l1_ratio": [0.20, 0.30, 0.45, 0.65, 0.75]},
            )
        elif p == "MDM":
            steps = [oas(), ("mdm", MDM(metric="riemann", n_jobs=n_jobs))]
        elif p == "FgMDM":
            steps = [oas(), ("fgmdm", FgMDM(metric="riemann", n_jobs=n_jobs))]
        elif p == "CSP_LDA":
            steps = [oas(), ("csp", CSP(nfilter=6)), ("lda", LDA(solver="svd"))]
        elif p == "CSP_SVM":
            return grid(
                [oas(), ("csp", CSP(nfilter=6)), ("svc", svc(kernel="linear"))],
                {
                    "csp__nfilter": [2, 3, 4, 5, 6, 7, 8],
                    "svc__C": [1],
                    "svc__kernel": kernels,
                },
            )
        elif p == "DLCSPauto_shLDA":
            steps = [
                oas(),
                ("csp", CSP(nfilter=6)),
                ("lda", LDA(solver="lsqr", shrinkage="auto")),
            ]
        elif p == "TRCSP_LDA":
            steps = [
                ("covariances", Covariances(estimator="scm")),
                ("trcsp", _TRCSP(nfilter=6)),
                ("lda", LDA()),
            ]
        elif p == "FBCSP_SVM":
            band = Pipeline([oas(), ("csp", CSP(nfilter=4))])
            tuned_svc = GridSearchCV(
                svc(kernel="linear"), {"C": np.logspace(-2, 2, 10)}, n_jobs=n_jobs
            )
            steps = [
                ("filterbank", _FilterBank(band)),
                ("selectkbest", SelectKBest(_MutualInfo(rs), k=10)),
                ("gridsearchcv", tuned_svc),
            ]
        elif p == "LogVar_LDA":
            steps = [("logvariance", _LogVariance()), ("lda", LDA(solver="svd"))]
        elif p == "LogVar_SVM":
            return grid(
                [("logvariance", _LogVariance()), ("svc", svc(kernel="linear"))],
                {"svc__C": [1, 5, 10, 50, 100]},
            )
        elif p == "AUG_TANG_SVM":
            return grid(
                [
                    ("augmenteddataset", _AugmentedDataset()),
                    ("covariances", Covariances(estimator="cov")),
                    ts(),
                    ("svc", svc(kernel="rbf")),
                ],
                {
                    "augmenteddataset__order": list(range(1, 11)),
                    "augmenteddataset__lag": list(range(1, 11)),
                    "svc__C": [1],
                    "svc__kernel": kernels,
                },
            )
        elif p == "XdawnCov_MDM":
            steps = [
                (
                    "xdawncovariances",
                    XdawnCovariances(estimator="oas", xdawn_estimator="scm"),
                ),
                ("mdm", MDM(n_jobs=n_jobs)),
            ]
        elif p == "XdawnCov_TS_SVM":
            return grid(
                [
                    (
                        "xdawncovariances",
                        XdawnCovariances(estimator="oas", xdawn_estimator="scm"),
                    ),
                    ("tangentspace", TangentSpace()),
                    ("svc", svc()),
                ],
                {"svc__C": [10], "svc__kernel": kernels},
            )
        elif p == "ERPCov_MDM":
            steps = [
                ("erpcovariances", ERPCovariances(estimator="oas")),
                ("mdm", MDM(n_jobs=n_jobs)),
            ]
        elif p == "xDAWN_LDA":
            steps = [
                ("xdawn", Xdawn(estimator="oas")),
                ("vectorizer", Vectorizer()),
                ("lda", LDA()),
            ]
        else:
            raise ValueError(
                f"pipeline must be one of {sorted(MOABB_PIPELINES)}, got {p!r}."
            )
        return Pipeline(steps)

    def _band_pass(self, X):
        import mne

        X = np.asarray(X, dtype=np.float64)
        if self.pipeline == "FBCSP_SVM":
            # MOABB's FilterBankMotorImagery: one copy per band, on a last axis
            bands = MOABB_FILTER_BANK
        elif self.fmin is None and self.fmax is None:
            return X
        else:
            bands = [(self.fmin, self.fmax)]
        if self.sfreq is None or not self.sfreq > 0:
            raise ValueError(f"sfreq must be a positive number, got {self.sfreq!r}.")
        filtered = [
            mne.filter.filter_data(
                X, self.sfreq, l_freq=low, h_freq=high, method="iir", verbose=False
            )
            for low, high in bands
        ]
        if self.pipeline == "FBCSP_SVM":
            return np.stack(filtered, axis=-1)
        return filtered[0]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return [
            {"pipeline": "TSLR", "sfreq": 100.0, "random_state": 0},
            {"pipeline": "MDM", "fmin": None, "fmax": None, "random_state": 0},
            {"pipeline": "CSP_LDA", "fmin": None, "fmax": None, "random_state": 0},
            {"pipeline": "LogVar_LDA", "fmin": None, "fmax": None},
        ]


class MOABBTSLRClassifier(MOABBPipelineClassifier):
    """MOABB's tangent space logistic regression pipeline (``TSLR.yml``).

    ``MOABBPipelineClassifier`` with ``pipeline="TSLR"``: an 8-32 Hz band-pass,
    then ``Covariances(oas) -> TangentSpace(riemann) -> LogisticRegression(C=1.0)``
    and nothing else. See ``MOABBPipelineClassifier`` for details.

    Parameters
    ----------
    sfreq : float or None
        Sampling frequency of the input in Hz. Required unless both ``fmin``
        and ``fmax`` are None.
    fmin : float or None, default=8.0
        Low cutoff in Hz. None for no high-pass.
    fmax : float or None, default=32.0
        High cutoff in Hz. None for no low-pass.
    random_state : int or None, default=None
        Unused by this pipeline; kept so experiment code can set it.
    """

    _tags = {"capability:multithreading": False}

    def __init__(
        self,
        sfreq: float | None = None,
        fmin: float | None = 8.0,
        fmax: float | None = 32.0,
        random_state: int | None = None,
    ):
        super().__init__(
            pipeline="TSLR",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_jobs=1,
            random_state=random_state,
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return [
            {"sfreq": 100.0, "random_state": 0},
            {"fmin": None, "fmax": None, "random_state": 0},
        ]


# Minimum series length at 120 Hz of the braindecode models NeuralBench trains
# from scratch, with braindecode's default settings. Shorter cases are padded.
NEURALBENCH_MIN_TIMEPOINTS = {
    "CTNet": 64,
    "Deep4Net": 8,
    "EEGConformer": 99,
    "EEGNet": 27,
    "ShallowFBCSPNet": 99,
}


class NeuralBenchDeepClassifier(BaseClassifier):
    """A braindecode model trained from scratch with NeuralBench's recipe.

    The data is preprocessed with ``NeuralBenchPreprocessor`` (120 Hz, robust
    scaling, clamping at 20), then the braindecode model named by ``model`` is
    built with braindecode's default settings, as NeuralBench does, and trained
    through ``BraindecodeClassifier`` with NeuralBench's settings:

    - AdamW, learning rate 1e-4, weight decay 0.05;
    - one-cycle schedule with 10% warmup and cosine annealing, stepped per
      batch;
    - gradient norm clipping at 1.0 and batch size 64;
    - up to 40 epochs, with early stopping on validation balanced accuracy
      (patience 5), keeping the best epoch;
    - cross-entropy with balanced class weights and label smoothing 0.1.

    NeuralBench trains eight task-specific models this way. The five in
    braindecode are ``"CTNet"``, ``"Deep4Net"`` (aeon-neuro's
    ``DeepConvNetClassifier``), ``"EEGConformer"``, ``"EEGNet"`` and
    ``"ShallowFBCSPNet"``. Cases shorter than these models' minimum length
    (``NEURALBENCH_MIN_TIMEPOINTS``) are zero padded at the end after scaling.

    See the module docstring for why this set-up is used and how it differs from
    NeuralBench. Pass subject IDs as ``train_groups`` so the validation split
    is by subject, as in NeuralBench; without them it is by case.

    Parameters
    ----------
    model : str, default="CTNet"
        Name of a braindecode model.
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
        Extra keyword arguments for the braindecode model.
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
        model: str = "CTNet",
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
        self.model = model
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
            model=self.model,
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

    def _pad(self, X):
        """Zero pad cases shorter than the model's minimum length at the end."""
        short = NEURALBENCH_MIN_TIMEPOINTS.get(self.model, 0) - X.shape[2]
        if short > 0:
            X = np.pad(X, ((0, 0), (0, 0), (0, short)))
        return X

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        small = {
            "sfreq": 100.0,
            "n_epochs": 2,
            "early_stopping_patience": 1,
            "validation_size": 0.3,
            "batch_size": 4,
            "random_state": 0,
        }
        return [
            {
                "model": "EEGNet",
                "model_params": {"F1": 4, "D": 1, "kernel_length": 8},
                **small,
            },
            {
                "model": "ShallowFBCSPNet",
                "model_params": {"n_filters_time": 4, "n_filters_spat": 4},
                **small,
            },
        ]


class NeuralBenchCTNetClassifier(NeuralBenchDeepClassifier):
    """CTNet trained from scratch with NeuralBench's recipe.

    ``NeuralBenchDeepClassifier`` with ``model="CTNet"``. See
    ``NeuralBenchDeepClassifier`` and the module docstring for details.

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
        given, whole subjects are held out for validation.
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
        super().__init__(
            model="CTNet",
            sfreq=sfreq,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            validation_size=validation_size,
            train_groups=train_groups,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            model_params=model_params,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )

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
