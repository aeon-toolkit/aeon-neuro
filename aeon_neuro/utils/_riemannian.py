"""Utility functions for Riemannian geometry and covariance estimation."""

from __future__ import annotations

import numpy as np


def build_covariance_transformer(
    covariance_type: str = "cov",
    covariance_estimator: str = "scm",
    covariance_params: dict | None = None,
):
    """Construct a covariance transformer.

    Parameters
    ----------
    covariance_type : {"cov", "erp"}, default="cov"
        Which covariance step to use. ``"cov"`` estimates a plain covariance
        matrix with ``pyriemann.estimation.Covariances``. ``"erp"`` uses the
        supervised ``aeon_neuro.transformations.collection.ERPCovariances``.
    covariance_estimator : str, default="scm"
        Covariance estimator passed to the covariance transformer.
    covariance_params : dict or None, default=None
        Optional keyword arguments forwarded to the covariance transformer.

    Returns
    -------
    transformer
        A fitted covariance estimation transformer.
    """
    kwargs = {} if covariance_params is None else dict(covariance_params)

    if covariance_type == "cov":
        from pyriemann.estimation import Covariances

        return Covariances(estimator=covariance_estimator, **kwargs)
    if covariance_type == "erp":
        from aeon_neuro.transformations.collection import ERPCovariances

        return ERPCovariances(estimator=covariance_estimator, **kwargs)
    raise ValueError(
        "covariance_type must be one of {'cov', 'erp'}, " f"got {covariance_type!r}."
    )


def compute_covariance_matrices(
    X: np.ndarray,
    covariance_transformer,
    fit: bool = False,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate covariance matrices, on GPU if available.

    Plain covariances are computed on CUDA when available, then returned as
    a numpy array. The supervised ERP step runs on numpy and uses ``y`` at
    fit time.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints).
    covariance_transformer : transformer
        A covariance estimation transformer (e.g., from
        ``build_covariance_transformer``).
    fit : bool, default=False
        Whether to fit the transformer before transforming.
    y : np.ndarray or None, default=None
        Target labels, required if ``fit=True`` and using supervised
        covariance estimation (e.g., ``"erp"`` type).

    Returns
    -------
    X_cov : np.ndarray
        Estimated covariance matrices of shape (n_samples, n_channels, n_channels).
    """
    import torch

    # The ERP transformer is a numpy/aeon estimator, so only the plain
    # covariance path is moved onto the GPU.
    if torch.cuda.is_available():
        X = torch.as_tensor(X, dtype=torch.float64, device="cuda")

    if fit:
        X_cov = covariance_transformer.fit_transform(X, y)
    else:
        X_cov = covariance_transformer.transform(X)

    if torch.is_tensor(X_cov):
        X_cov = X_cov.cpu().numpy()
    return X_cov
