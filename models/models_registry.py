
#IMPORTS
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LogisticRegression,
    Perceptron)
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier)
from sklearn.neural_network import (
    MLPRegressor,
    MLPClassifier)
from sklearn.svm import (
    SVR,
    SVC)
from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier)
from sklearn.naive_bayes import GaussianNB
from xgboost import (
    XGBRegressor,
    XGBClassifier)
from lightgbm import (
    LGBMRegressor,
    LGBMClassifier)
from catboost import (
    CatBoostRegressor,
    CatBoostClassifier)




#MODELS
REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(),
    "RandomForestRegressor": RandomForestRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "MLPRegressor": MLPRegressor(),
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "AdaBoostRegressor": AdaBoostRegressor(),
    "XGBRegressor": XGBRegressor(),
    "LightGBMRegressor": LGBMRegressor(),
    "CatBoostRegressor": CatBoostRegressor()}

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "MLPClassifier": MLPClassifier(),
    "SVC": SVC(),
    "Perceptron": Perceptron(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "XGBClassifier": XGBClassifier(),
    "LightGBMClassifier": LGBMClassifier(),
    "CatBoostClassifier": CatBoostClassifier()}


#GRID
REGRESSION_PARAM_GRIDS = {
    "LinearRegression": {},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "Ridge": {"alpha": [0.1, 1, 10, 100]},
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1],
                   "l1_ratio": [0.1, 0.5, 0.9]},
    "RandomForestRegressor": {"n_estimators": [100, 200, 300],
                              "max_depth": [None, 10, 20, 40],
                              "min_samples_split": [2, 5, 10]},
    "DecisionTreeRegressor": {"max_depth": [None, 5, 10, 20],
                              "min_samples_split": [2, 5, 10]},
    "MLPRegressor": {"hidden_layer_sizes": [(50,), (100,), (100, 50)],
                     "activation": ["relu", "tanh"],
                     "alpha": [0.0001, 0.001, 0.01]},
    "SVR": {"C": [0.1, 1, 10],
            "kernel": ["rbf", "poly", "linear"],
            "gamma": ["scale", "auto"]},
    "KNeighborsRegressor": {"n_neighbors": [3, 5, 7, 9],
                            "weights": ["uniform", "distance"]},
    "GradientBoostingRegressor": {"n_estimators": [100, 200],
                                  "learning_rate": [0.01, 0.1, 0.2],
                                  "max_depth": [2, 3, 5]},
    "AdaBoostRegressor": {"n_estimators": [50, 100, 200],
                          "learning_rate": [0.01, 0.1, 1.0]},
    "XGBRegressor": {"n_estimators": [100, 200],
                     "max_depth": [3, 5, 7],
                     "learning_rate": [0.01, 0.1, 0.2]},
    "LightGBMRegressor": {"n_estimators": [100, 200],
                          "num_leaves": [31, 50, 100],
                          "learning_rate": [0.01, 0.1]},
    "CatBoostRegressor": {"depth": [4, 6, 10],
                          "learning_rate": [0.01, 0.1],
                          "iterations": [200, 500]}}

CLASSIFICATION_PARAM_GRIDS = {
    "LogisticRegression": {"C": [0.1, 1, 10],
                           "penalty": ["l2"],
                           "solver": ["lbfgs"],
                           "max_iter": [200, 500]},
    "RandomForestClassifier": {"n_estimators": [100, 200, 300],
                               "max_depth": [None, 10, 20, 40],
                               "min_samples_split": [2, 5, 10]},
    "DecisionTreeClassifier": {"max_depth": [None, 5, 10, 20],
                               "criterion": ["gini", "entropy"],
                               "min_samples_split": [2, 5, 10]},
    "MLPClassifier": {"hidden_layer_sizes": [(50,), (100,), (100, 50)],
                      "activation": ["relu", "tanh"],
                      "alpha": [0.0001, 0.001, 0.01],
                      "max_iter": [300]},
    "SVC": {"C": [0.1, 1, 10],
            "kernel": ["rbf", "poly", "linear"],
            "gamma": ["scale", "auto"]},
    "Perceptron": {"penalty": [None, "l2", "l1", "elasticnet"],
                   "alpha": [0.0001, 0.001, 0.01]},
    "KNeighborsClassifier": {"n_neighbors": [3, 5, 7, 9],
                             "weights": ["uniform", "distance"]},
    "GaussianNB": {},
    "GradientBoostingClassifier": {"n_estimators": [100, 200],
                                   "learning_rate": [0.01, 0.1, 0.2],
                                   "max_depth": [2, 3, 5]},
    "AdaBoostClassifier": {"n_estimators": [50, 100, 200],
                           "learning_rate": [0.01, 0.1, 1.0]},
    "XGBClassifier": {"n_estimators": [100, 200],
                      "max_depth": [3, 5, 7],
                      "learning_rate": [0.01, 0.1, 0.2]},
    "LightGBMClassifier": {"n_estimators": [100, 200],
                           "num_leaves": [31, 50, 100],
                           "learning_rate": [0.01, 0.1]},
    "CatBoostClassifier": {"depth": [4, 6, 10],
                           "learning_rate": [0.01, 0.1],
                           "iterations": [200, 500]}}
