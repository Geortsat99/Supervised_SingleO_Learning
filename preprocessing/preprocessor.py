#pca sphering

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

class Preprocessor:
    def __init__(self, data, target_str):
        self.data = data
        self.target_str = target_str

    def splitter(self):
        y=self.data[self.target_str]
        X=self.data.drop(self.target_str,axis=1)
        return X,y

    def distribution_fixer(self):
        X, y = self.splitter()
        fixed_X = X.copy()
        fixed_y = y.copy()
        return fixed_X, fixed_y

    def one_hot_encoder(self):
        X,y=self.distribution_fixer()
        encoder = OneHotEncoder()
        #encoder.fit(self.X)
        encoded_X=X.copy()
        encoded_y=Y.copy()
        return encoded_X,encoded_y

    def pca_shpering(self):
        X, y = self.one_hot_encoder()
