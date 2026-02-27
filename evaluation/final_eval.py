





class Evaluation:
    def __init__(self, results_df,sort_metric,time_gravity):
        self.results_df = results_df
        self.sort_metric = sort_metric
        self.time_gravity = time_gravity

    def evaluate(self):
        self.results_df['custom_metric']= -self.time_gravity * (self.results_df['fit_time'] + self.results_df['score_time']) +
