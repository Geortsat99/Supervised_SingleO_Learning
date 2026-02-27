import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate


class CVResults:
    def __init__(self, pipe, X, Y, k_settings, fit_metrics, config_params):
        self.pipe = pipe
        self.k_settings = k_settings
        self.fit_metrics = fit_metrics
        self.config_params = config_params
        self.X=X
        self.Y=Y

    def cross_validation(self):
        pipe = clone(self.pipe)
        pipe.set_params(**self.config_params)
        results_cv = cross_validate(pipe, self.X, self.Y, cv=self.k_settings, scoring=self.fit_metrics)
        results_df=pd.DataFrame(results_cv)
        results_df=results_df.mean()
        return results_df
