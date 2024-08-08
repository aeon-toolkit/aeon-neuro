"""BandpowerEnsemble classifier."""

import numpy as np

# TODO: from aeon_neuro.transformations import BandPowerSeriesTransformer


class BandpowerEnsemble:
    """Fit/predict from an ensemble of bandpower transformers."""

    def __init__(self, fs, classifier):
        self.extractors = []
        self.classifiers = []
        self.bands = ["delta", "theta", "alpha", "beta", "gamma"]
        # TODO: refactor
        # for i in self.bands:
        #     self.extractors.append(bandpower.BandpowerExtraction(fs, i))
        for _ in range(5):
            self.classifiers.append(classifier.clone())

    def fit(self, X, y):
        """Fit method."""
        self.X = X
        self.y = y
        self.bandX = [0] * 5
        self.classes_ = np.unique(y)
        for i in range(5):
            self.bandX[i] = self.extractors[i].transform(self.X)
            self.classifiers[i].fit(self.bandX[i], y)

    def predict_proba(self, X):
        """Predict method."""
        final = np.zeros((len(X), len(self.classes_)))
        for i in range(5):
            bandX = self.extractors[i].transform(X)
            probs = self.classifiers[i].predict_proba(bandX)
            for instance in range(len(probs)):
                for classval in range(len(probs[instance])):
                    final[instance][classval] += probs[instance][classval]
        return final
