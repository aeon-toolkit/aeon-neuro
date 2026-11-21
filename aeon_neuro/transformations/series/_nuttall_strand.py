"""Nuttall-Strand power spectral density estimation."""

from numbers import Integral, Real

import numpy as np
from aeon.transformations.series.base import BaseSeriesTransformer
from scipy.linalg import LinAlgError, solve, solve_sylvester

__all__ = ["NuttallStrandSeriesTransformer"]


class NuttallStrandSeriesTransformer(BaseSeriesTransformer):
    """Estimate spectral density matrices with the Nuttall-Strand algorithm.

    Nuttall-Strand, also known as multivariate Burg, fits a vector
    autoregressive model using weighted forward and backward prediction errors.
    Its spectrum is returned as one complex Hermitian matrix per frequency.

    Parameters
    ----------
    model_order : int, default=10
        Order of the vector autoregressive model.
    n_freqs : int, default=10
        Number of equally spaced frequencies in the output.
    sfreq : float, default=120
        Sampling frequency in Hz.
    fmin : float, default=0
        Lowest frequency in Hz.
    fmax : float, default=30
        Highest frequency in Hz. Must not exceed the Nyquist frequency.

    Notes
    -----
    The input should be approximately stationary. Channel means are not removed;
    demean the input before transforming it, particularly when ``fmin=0``, to
    prevent an offset from producing a large 0 Hz term.

    The input must have full row rank. Rank-deficient channel data do not define
    the nonsingular covariance matrices required by the Nuttall-Strand recursion.

    The output has shape ``(n_freqs, n_channels, n_channels)`` and uses the
    frequency grid ``np.linspace(fmin, fmax, n_freqs)``. It is a two-sided
    spectral density divided by ``sfreq``, so positive-frequency values are half
    the corresponding one-sided Welch PSD. For inputs measured in units ``U``,
    the spectral matrix entries have units ``U**2 / Hz``.

    References
    ----------
    .. [1] Nuttall, A. H. (1976). Multivariate linear predictive spectral
       analysis employing weighted forward and backward averaging: A
       generalization of Burg's algorithm. NUSC Technical Report 5501.
    .. [2] Strand, O. N. (1977). Multichannel complex maximum entropy
       (autoregressive) spectral analysis. IEEE Transactions on Automatic
       Control, 22(4), 634-640. https://doi.org/10.1109/TAC.1977.1101545

    Examples
    --------
    >>> import numpy as np
    >>> from aeon_neuro.transformations.series import NuttallStrandSeriesTransformer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((3, 200))
    >>> psd = NuttallStrandSeriesTransformer(
    ...     model_order=3, n_freqs=5
    ... ).fit_transform(X)
    >>> psd.shape
    (5, 3, 3)
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        model_order=10,
        n_freqs=10,
        sfreq=120,
        fmin=0,
        fmax=30,
    ):
        self.model_order = model_order
        self.n_freqs = n_freqs
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        super().__init__(axis=1)

    @staticmethod
    def _solve(matrix, rhs):
        """Solve a Hermitian system, falling back to a pseudoinverse."""
        try:
            return solve(matrix, rhs, assume_a="her")
        except LinAlgError:
            return np.linalg.pinv(matrix) @ rhs

    def _solve_forward_reflection(self, forward, backward, p_forward, p_backward):
        """Solve the Nuttall-Strand Sylvester equation."""
        n_observations = forward.shape[1]
        error_forward = forward @ forward.conj().T / n_observations
        error_backward = backward @ backward.conj().T / n_observations
        cross_error = backward @ forward.conj().T / n_observations

        left = self._solve(p_backward, error_backward)
        right = self._solve(p_forward, error_forward)
        rhs = -2 * self._solve(p_backward, cross_error)
        return solve_sylvester(left, right, rhs)

    @staticmethod
    def _update_filter_coefficients(
        forward_coefficients,
        backward_coefficients,
        forward_reflection,
        backward_reflection,
    ):
        """Update forward and backward autoregressive coefficients."""
        n_channels = forward_reflection.shape[0]
        zeros = np.zeros((n_channels, n_channels), dtype=forward_reflection.dtype)
        new_forward = (
            np.vstack((forward_coefficients, zeros))
            + np.vstack((zeros, backward_coefficients)) @ forward_reflection
        )
        new_backward = np.vstack((forward_coefficients, zeros)) @ backward_reflection
        new_backward += np.vstack((zeros, backward_coefficients))
        return new_forward, new_backward

    def _compute_spectrum(self, forward_coefficients, noise_covariance):
        """Evaluate the fitted vector autoregressive spectrum."""
        n_channels = noise_covariance.shape[0]
        coefficients = (
            forward_coefficients.reshape(self.model_order + 1, n_channels, n_channels)[
                1:
            ]
            .conj()
            .swapaxes(1, 2)
        )
        frequencies = np.linspace(self.fmin, self.fmax, self.n_freqs)
        angular_frequencies = 2 * np.pi * frequencies / self.sfreq
        powers = np.arange(1, self.model_order + 1)
        exponentials = np.exp(-1j * np.outer(angular_frequencies, powers))
        ar_polynomials = np.eye(n_channels) + np.einsum(
            "fk,kij->fij", exponentials, coefficients
        )

        spectrum = np.empty((self.n_freqs, n_channels, n_channels), dtype=np.complex128)
        identity = np.eye(n_channels)
        for index, polynomial in enumerate(ar_polynomials):
            transfer = solve(polynomial, identity)
            estimate = transfer @ noise_covariance @ transfer.conj().T / self.sfreq
            spectrum[index] = (estimate + estimate.conj().T) / 2
        return spectrum

    def _validate_parameters(self, n_timepoints):
        """Validate parameters that depend on the input series."""
        if not isinstance(self.model_order, Integral) or self.model_order <= 0:
            raise ValueError("model_order must be a positive integer.")
        if self.model_order >= n_timepoints:
            raise ValueError("model_order must be smaller than the series length.")
        if not isinstance(self.n_freqs, Integral) or self.n_freqs < 1:
            raise ValueError("n_freqs must be a positive integer.")
        if not isinstance(self.sfreq, Real) or self.sfreq <= 0:
            raise ValueError("sfreq must be positive.")
        if not isinstance(self.fmin, Real) or self.fmin < 0:
            raise ValueError("fmin must be non-negative.")
        if not isinstance(self.fmax, Real) or self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin.")
        if self.fmax > self.sfreq / 2:
            raise ValueError("fmax must not exceed the Nyquist frequency.")

    def _transform(self, X, y=None):
        """Transform a multichannel series into spectral density matrices."""
        n_channels, n_timepoints = X.shape
        self._validate_parameters(n_timepoints)

        dtype = np.result_type(X.dtype, np.float64)
        X = np.asarray(X, dtype=dtype)
        if np.linalg.matrix_rank(X) < n_channels:
            raise ValueError(
                "X must have full row rank; rank-deficient inputs are not supported."
            )
        p_forward = X @ X.conj().T / n_timepoints
        p_backward = p_forward.copy()
        identity = np.eye(n_channels, dtype=dtype)
        forward_coefficients = identity
        backward_coefficients = identity
        forward = X[:, 1:]
        backward = X[:, :-1]

        for order in range(1, self.model_order + 1):
            forward_reflection = self._solve_forward_reflection(
                forward, backward, p_forward, p_backward
            )
            backward_reflection = self._solve(
                p_forward, forward_reflection.conj().T @ p_backward
            )

            old_p_forward = p_forward
            old_p_backward = p_backward
            p_forward = old_p_forward - (
                forward_reflection.conj().T @ old_p_backward @ forward_reflection
            )
            p_backward = old_p_backward - (
                backward_reflection.conj().T @ old_p_forward @ backward_reflection
            )
            p_forward = (p_forward + p_forward.conj().T) / 2
            p_backward = (p_backward + p_backward.conj().T) / 2

            forward_coefficients, backward_coefficients = (
                self._update_filter_coefficients(
                    forward_coefficients,
                    backward_coefficients,
                    forward_reflection,
                    backward_reflection,
                )
            )

            if order < self.model_order:
                next_forward = forward[:, 1:] + (
                    forward_reflection.conj().T @ backward[:, 1:]
                )
                next_backward = backward[:, :-1] + (
                    backward_reflection.conj().T @ forward[:, :-1]
                )
                forward, backward = next_forward, next_backward

        return self._compute_spectrum(forward_coefficients, p_forward)
