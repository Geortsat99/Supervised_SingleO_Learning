from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class PCASphering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = Pipeline([
            ("mean_center", StandardScaler(with_mean=True, with_std=False)),
            ("pca", PCA(whiten=False)),
            ("scale_components", StandardScaler(with_mean=False, with_std=True))
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

