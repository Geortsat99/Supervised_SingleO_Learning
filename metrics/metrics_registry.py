#IMPORTS
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer)

#SCORERS
REGRESSION_SCORERS = {
    "r2": make_scorer(r2_score),
    "explained_variance": make_scorer(explained_variance_score),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "mse": make_scorer(mean_squared_error, greater_is_better=False),
    "rmse": make_scorer(lambda y, y_pred: mean_squared_error(y, y_pred, squared=False),greater_is_better=False),
    "mape": make_scorer(mean_absolute_percentage_error, greater_is_better=False)}

CLASSIFICATION_SCORERS = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="binary"),
    "recall": make_scorer(recall_score, average="binary"),
    "f1": make_scorer(f1_score, average="binary"),
    "balanced_accuracy": make_scorer(accuracy_score),
    "precision_weighted": make_scorer(precision_score, average="weighted"),
    "recall_weighted": make_scorer(recall_score, average="weighted"),
    "f1_weighted": make_scorer(f1_score, average="weighted"),}
