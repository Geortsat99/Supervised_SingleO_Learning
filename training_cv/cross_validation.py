import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate


class CVResults:
    def __init__(self, model_name, pipe, X, Y, k_settings, fit_metrics, config_params):
        self.model_name = model_name
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
        result_df = results_df.mean().to_frame().T
        result_df['parameters']=self.config_params
        result_df['model_name']=self.model_name
        return result_df
