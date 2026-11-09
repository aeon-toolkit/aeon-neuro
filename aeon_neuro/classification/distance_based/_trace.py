import numpy as np
import pyriemann
from aeon.classification.base import BaseClassifier
from aeon.distances import dtw_distance
from scipy.sparse.csgraph import dijkstra


class TRACE(BaseClassifier):
    """
    TRACE (Temporal Riemannian Adaptive Classification Ensemble).

    This classifier combines the Riemannian distance metric
    with an intervalling approach to form an ensemble of weighted segments.
    Two optional weighting schemes,convexity and surprise,
    can be used to weight this ensemble.

    Parameters
    ----------
    cross_validate : {True,False}, default=True
        Whether the classifier should utilise train only cross validation
        to set the number of intervals the time series is split into
    use_convexity : {True,False}, default=True
        If the classifier should use convexity based weighting
    use_surprise : {True,False}, default=True
        If the classifier should use surprise based weighting
    intervals : int, default=5
        The number of intervals used to form the ensemble
    scale : int, default=1
        The scaling factor for the weighting schemes, a larger value
        indictes larger emphasis on weighting methods
    k : int, default=10
        Number of neighbours considered for convexity based weighing
    random_state: : int, RandomState instance or None, default=None
        Unused as clasifier contains no randomisation


    """

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        cross_validate=True,
        use_convexity=True,
        use_surprise=True,
        intervals=5,
        scale=1,
        k=10,
        random_state=None,
    ):
        self.cross_validate = cross_validate
        self.use_convexity = use_convexity
        self.use_surprise = use_surprise
        self.intervals = intervals
        self.scale = scale
        self.k = k
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        self.means_ = []
        self.convexity_weights_ = []
        self.surprise_weights_ = []
        self.classes_ = np.unique(y)
        convexity_measure = []

        if self.cross_validate:
            from sklearn.model_selection import GridSearchCV

            parameters = {"intervals": list(range(1, 15, 2)), "k": list(range(2, 10))}
            cls = TRACE(
                cross_validate=False,
                use_convexity=self.use_convexity,
                use_surprise=self.use_surprise,
            )
            clf = GridSearchCV(cls, parameters)
            clf.fit(X, y)
            self.intervals = clf.best_params_["intervals"]
            self.k = clf.best_params_["k"]

        remainder = int(len(X[0][0]) % self.intervals)

        if self.intervals == 1:
            self.use_convexity = False
            self.use_surprise = False

        if remainder != 0:
            intervals = np.array_split(X[:, :, :-remainder], self.intervals, axis=2)
        else:
            intervals = np.array_split(X, self.intervals, axis=2)

        if self.use_surprise:
            self.surprise_weights_ = surprise_calculation(intervals)

        for i in range(len(intervals)):
            interval_mean = []
            X_cov_train = pyriemann.geometry.covariance.covariances(
                intervals[i], estimator="lwf"
            )
            self.fg_ = pyriemann.tangentspace.FGDA()
            X_cov_train = self.fg.fit_transform(X_cov_train, y)
            for class_val in self.classes_:
                idx = np.where(y == class_val)
                slice_X_class = X_cov_train[idx[0]]
                class_mean = pyriemann.geometry.mean.mean_riemann(slice_X_class)
                interval_mean.append(class_mean)
            self.means.append(interval_mean)
            if self.use_convexity:
                convexity_measure.append(calculate_convexity(intervals[i], y, k=self.k))

        if self.use_convexity:
            mean = np.mean(convexity_measure)
            std = np.std(convexity_measure)
            z = (convexity_measure - mean) / std
            normalised = 1 / (1 + np.exp(-0.5 * z))
            self.convexity_weights_ = normalised

    def _predict_proba(self, X):
        X_test_distance = []
        intervals = np.array_split(X, self.intervals, axis=2)
        interval_covs = []

        for j in range(len(intervals)):
            X_cov_test = pyriemann.geometry.covariance.covariances(
                intervals[j], estimator="lwf"
            )
            X_cov_test = self.fg_.transform(X_cov_test)
            interval_covs.append(X_cov_test)

        for i in range(len(X)):
            interval_dist = []
            for j in range(len(intervals)):
                distances = []
                count = 0
                for r in range(-1, 2, 1):
                    if (j + r) > -1 and (j + r) < self.intervals:
                        count += 1
                        for c in range(len(self.classes_)):
                            distances.append(
                                pyriemann.geometry.distance.distance_riemann(
                                    interval_covs[j][i], np.array(self.means[j + r][c])
                                )
                            )
                for c in range(len(self.classes_)):
                    distances[c] = distances[c] / count
                interval_dist.append(distances)
            X_test_distance.append(interval_dist)
        probs = []

        if self.use_convexity and self.use_surprise:
            self.convexity_weights_ = [1 - x for x in self.convexity_weights_]
            self.surprise_weights_ = [1 - x for x in self.surprise_weights_]
            alpha = 0.8
            beta = 0.2
            weights = (np.array(self.convexity_weights_) ** alpha) * (
                np.array(self.surprise_weights_) ** beta
            )
            weights = [int((x) * 100) for x in weights]
            weights = scale_values_np(weights, self.scale)

        elif self.use_convexity:
            self.convexity_weights_ = [1 - x for x in self.convexity_weights_]
            weights = [int((x) * 100) for x in np.array(self.convexity_weights_)]
            weights = scale_values_np(weights, self.scale)

        elif self.use_surprise:
            self.surprise_weights_ = [1 - x for x in self.surprise_weights_]
            weights = [int((x) * 100) for x in np.array(self.surprise_weights_)]
            weights = scale_values_np(weights, self.scale)

        else:
            weights = [1 for x in range(self.intervals)]

        for inst in range(len(X_test_distance)):
            inst_votes = []
            instance = X_test_distance[inst]
            count = 0
            for interval in range(len(instance)):
                for _w in range(weights[interval]):
                    smallest = np.argmin(instance[interval]) % len(self.classes_)
                    temp = list(np.zeros(len(self.classes_)))
                    temp[smallest] = 1
                    inst_votes.append(temp)
                    count += 1

            counts = np.sum(inst_votes, axis=0)

            for i in range(len(counts)):
                counts[i] = float(counts[i] / count)
            probs.append(list(counts))

        return np.array(probs)

    def _predict(self, X):
        probas = self._predict_proba(X)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds


def scale_values_np(arr, scale):
    arr = np.asarray(arr)
    ranks = np.argsort(np.argsort(arr))
    factors = (1 / scale) + (scale - (1 / scale)) * ranks / (len(arr) - 1)
    return (arr * factors).astype(int)


def surprise_calculation(intervals):

    n_intervals, n_instances, n_channels, n_timepoints = np.shape(intervals)

    all_dists = []
    for j in range(n_intervals):
        X_trains = np.delete(intervals, j, axis=0)
        X_tests = intervals[j]
        distance_c = []
        for i in range(n_instances):
            X_train = np.mean(X_trains[:, i, :, :], axis=0)
            X_test = X_tests[i, :, :]
            distance = dtw_distance(X_train, X_test)
            distance_c.append(distance)

        all_dists.append(np.mean(distance_c))
    all_dists = np.array(all_dists)
    mean = np.mean(all_dists)
    std = np.std(all_dists)
    z = (all_dists - mean) / std
    normalised = 1 / (1 + np.exp(-0.5 * z))

    return normalised


def _reconstruct_path(predecessors, start, end):
    path = []
    i = end
    while i != start:
        if i == -9999:
            return None
        path.append(i)
        i = predecessors[start, i]
    path.append(start)
    return path[::-1]


def calculate_convexity(X, y, k):

    covariances = pyriemann.geometry.covariance.covariances(X, estimator="lwf")
    distance_matrix = []
    for i in range(len(X)):
        test = pyriemann.geometry.distance.distance(
            covariances, covariances[i], metric="riemann"
        )
        distance_matrix.append(test)
    distance_matrix = np.squeeze(np.array(distance_matrix))
    new_matrix = np.full_like(distance_matrix, np.inf)
    nearest = np.argsort(distance_matrix, axis=1)[:, :k]
    for i in range(len(distance_matrix)):
        new_matrix[i, nearest[i]] = distance_matrix[i, nearest[i]]
    new_matrix = np.minimum(new_matrix, new_matrix.T)
    np.fill_diagonal(new_matrix, np.inf)
    dist, pres = dijkstra(new_matrix, return_predecessors=True)
    score = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        cscore = []
        for start in idx:
            for end in idx:
                if start != end:
                    path = _reconstruct_path(pres, start, end)
                    if path is None:
                        cscore.append(1)
                    else:
                        temp = 0
                        for i in path:
                            if y[i] != c:
                                temp += 1
                        cscore.append(float(temp / len(path)))
        score.append(np.mean(cscore))
    return np.mean(score)
