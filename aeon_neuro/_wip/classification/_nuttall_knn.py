"""Cumstomed K-nearest neighbors classifier for Burg-type Nuttall-Strand algorithm."""

from sklearn.neighbors import KNeighborsClassifier
from aeon_neuro._wip.distances._get_optimal_weights import get_optimal_weight1, get_optimal_weight2
from aeon_neuro._wip.distances._riemannian_matrix import (
    pairwise_riemannian_distance_1,
    pairwise_riemannian_distance_2,
    pairwise_weighted_riemannian_distance_1,
    pairwise_weighted_riemannian_distance_2,    
)


class NuttallKNN(KNeighborsClassifier):
    """Custom K-nearest neighbors classifier for Burg-type Nuttall-Strand algorithm.

    Parameters
    ----------
    mode : int 1 or 2
        Choose a Riemannian distance function to be used, by default 1.
    weighted : bool
        If True, use optimal weighted Riemannian distance, by default False.
    n_neighbors : int, optional
        Number of neighbors to use, by default 3.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 1.
        
    Examples
    --------
    >>> from aeon_neuro._wip.classification._nuttall_knn import NuttallKNN
    >>> from aeon_neuro._wip.transformations.series._nuttall_strand import (
    ...     NuttallStrand)
    >>> from sklearn.metrics import accuracy_score
    >>> from aeon.datasets import load_classification
    >>> import numpy as np
    >>> import warnings
    >>> warnings.filterwarnings("ignore")
    >>> n_freqs=25
    >>> transformer = NuttallStrand(model_order=5, n_freqs=n_freqs, fmax=60)
    >>> X, y= load_classification("EyesOpenShut")
    >>> X_train = X[:56]
    >>> X_test = X[56:58]
    >>> y_train = y[:56]
    >>> y_test = y[56:58]
    >>> X_train_psd = []
    >>> X_test_psd = []
    >>> for i in range(X_train.shape[0]):
    ...     X_train_psd.append(transformer.fit_transform(X_train[i]))
    >>> for i in range(X_test.shape[0]):
    ...     X_test_psd.append(transformer.fit_transform(X_test[i]))
    >>> X_train_psd = np.array(X_train_psd)
    >>> X_test_psd = np.array(X_test_psd)
    >>> knn = NuttallKNN(mode=1, weighted=True, n_neighbors=1, n_jobs=-1)
    >>> knn.fit(X_train_psd, y_train)
    >>> y_pred = knn.predict(X_test_psd)
    >>> accuracy_score(y_test, y_pred)
    0.5
    """
    
    def __init__(self, mode: int = 1, weighted: bool = False, n_neighbors=3, n_jobs=-1):
        assert mode in [1, 2], "Mode must be 1 or 2"
        self.mode = mode
        self.weighted = weighted
        super().__init__(metric='precomputed', n_neighbors=n_neighbors, n_jobs=n_jobs)

    def fit(self, X, y):
        self.X_train_ = X
        distances = None
        if self.mode == 1:
            if self.weighted is False:
                distances = pairwise_riemannian_distance_1(X)
            else:
                self.W_ = get_optimal_weight1(X, y)
                distances = pairwise_weighted_riemannian_distance_1(X, self.W_)
        else:
            if self.weighted is False:
                distances = pairwise_riemannian_distance_2(X)
            else:
                self.W_ = get_optimal_weight2(X, y)
                distances = pairwise_weighted_riemannian_distance_2(X, self.W_)
        super().fit(distances, y)


    def predict(self, X):
        distances = None
        if self.mode == 1:
            if self.weighted is False:
                distances = pairwise_riemannian_distance_1(X, self.X_train_)
            else:
                distances = pairwise_weighted_riemannian_distance_1(X, self.W_, self.X_train_)
        else:
            if self.weighted is False:
                distances = pairwise_riemannian_distance_2(X, self.X_train_)
            else:
                distances = pairwise_weighted_riemannian_distance_2(X, self.W_, self.X_train_)
        return super().predict(distances)