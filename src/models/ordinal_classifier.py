"""
Ordinal classification via binary decomposition (Frank & Hall, 2001).

No ordinal-regression package (mord, statsmodels) is available in this environment, so this
implements the well-established Frank & Hall decomposition directly on top of sklearn, which
keeps the project fully reproducible without external ordinal-regression dependencies.

Reference:
  Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. In Proceedings
  of the 12th European Conference on Machine Learning (ECML 2001), LNAI 2167, pp. 145-156.
  Springer. https://doi.org/10.1007/3-540-44795-4_13

  Survey/validation of this family of methods:
  Gutierrez, P. A., Perez-Ortiz, M., Sanchez-Monedero, J., Fernandez-Navarro, F., &
  Hervas-Martinez, C. (2016). Ordinal regression methods: survey and experimental study.
  IEEE Transactions on Knowledge and Data Engineering, 28(1), 127-146.

Method: for K ordered classes c_0 < c_1 < ... < c_{K-1}, train K-1 binary classifiers, where
classifier k predicts P(y > c_k). The class probabilities are then recovered as:
  P(y = c_0)     = 1 - P(y > c_0)
  P(y = c_k)     = P(y > c_{k-1}) - P(y > c_k)      for 0 < k < K-1
  P(y = c_{K-1}) = P(y > c_{K-2})
This preserves the ordering of the classes (predicting "close" misses rather than treating all
errors as equally bad, unlike one-vs-rest nominal multiclass classification), while still using
only standard, interpretable binary logistic regression underneath.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class OrdinalLogisticClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal classifier using Frank & Hall (2001) binary decomposition with
    logistic regression as the base binary learner."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.binary_clfs_ = []
        for k in self.classes_[:-1]:
            y_bin = (y > k).astype(int)
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                      random_state=self.random_state)
            clf.fit(X, y_bin)
            self.binary_clfs_.append(clf)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        K = len(self.classes_)
        probs_gt = np.column_stack([clf.predict_proba(X)[:, 1] for clf in self.binary_clfs_])

        proba = np.zeros((n, K))
        proba[:, 0] = 1 - probs_gt[:, 0]
        for k in range(1, K - 1):
            proba[:, k] = probs_gt[:, k - 1] - probs_gt[:, k]
        proba[:, K - 1] = probs_gt[:, K - 2]

        # Independently-trained binary classifiers can occasionally produce small negative
        # "probabilities" where monotonicity (P(y>k-1) >= P(y>k)) is violated; clip and
        # renormalize rather than silently allowing invalid probabilities.
        proba = np.clip(proba, 0, None)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
