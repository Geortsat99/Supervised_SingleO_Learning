#IMPORTS
from sklearn.preprocessing import MinMaxScaler
import numpy as np





class Evaluation:
    def __init__(self, all_results_df, sort_metric, time_gravity=0):
        self.all_results_df = all_results_df
        self.sort_metric = sort_metric
        self.time_gravity = time_gravity


    def evaluate(self):
        all_res_df=self.all_results_df.copy()
        all_res_df['logged_total_time']=np.log(all_res_df['fit_time']+all_res_df['score_time'])
        scaler = MinMaxScaler()
        all_res_df['scaled_time'] = scaler.fit_transform(all_res_df[["logged_total_time"]])
        all_res_df['time_grade']=1-all_res_df['scaled_time']
        all_res_df['scaled_metric']=scaler.fit_transform(all_res_df[[f'test_{self.sort_metric}']])
        if all_res_df[f'test_{self.sort_metric}'].mean()<0:
            all_res_df['metric_grade']=1-all_res_df['scaled_metric']
        all_res_df['final_score']=self.time_gravity*all_res_df["time_grade"]+(1-self.time_gravity)*all_res_df["metric_grade"]
        return all_res_df

    def get_best_model_name(self):
        res_df=self.evaluate()
        best_model_name=res_df.loc[res_df['final_score'].max()]['model_name']
        return best_model_name

    def get_best_model_params(self):
        res_df=self.evaluate()
        best_model_params=res_df.loc[res_df['final_score'].max()]['params']
        return best_model_params



