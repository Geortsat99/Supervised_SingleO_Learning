import pandas as pd
from sklearn.model_selection import KFold

my_df={'target':[1,2,3,4,5,6,7,8,9],'predictor':[-11,-23,-29,-40,-56,-69,-68,-82,-87]}
data_df=pd.DataFrame(my_df)
models_selection=['LinearRegression','DecisionTreeRegressor']
status='regression'
selected_tune_metric='mae'
final_eval_metric='mse'
selected_tune_metric=f'test_{selected_tune_metric}'
final_eval_metric=f'test_{final_eval_metric}'
k_settings=KFold(n_splits=5,shuffle=True, random_state=42)
Y=data_df['target']
X=data_df.drop('target',axis=1)

