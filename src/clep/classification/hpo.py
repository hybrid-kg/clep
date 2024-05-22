from typing import Callable, cast
import joblib

import numpy as np
import pandas as pd
from optuna import create_study, Trial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

from xgboost import XGBClassifier


class OptunaObjective:
    def __init__(self, X, y, metric, cv):
        self.X = X
        self.y = y
        self.metric = metric
        self.cv = cv
        self.classifier = None

    def __call__(self, trial):
        self.classifier = trial.suggest_categorical('classifier', ['SVM', 'ElasticNet', 'LogisticRegression', 'RandomForest', 'GradientBoosting'])
        # Define the hyperparameters to optimize based on the suggested classifier
        clf = self._get_clf(trial)

        if len(np.unique(self.y)) > 2:
            from classify import _do_multiclass_classification
            cv_results = _do_multiclass_classification(estimator=clf,
                                                       x=self.X,
                                                       y=self.y,
                                                       cv=self.cv,
                                                       scoring=self.metric)
        else:
            cv_results =  model_selection.cross_validate(estimator=clf,
                                                         X=self.X,
                                                         y=self.y,
                                                         cv=model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True),
                                                         scoring=self.metric,
                                                         return_estimator=True)

        for key, value in cv_results.items():
            trial.set_user_attr(key, value)

        metric_scores = self._get_metric(cv_results)

        return metric_scores

    def _get_clf(self, trial):
        match self.classifier:
            case 'LogisticRegression':
                C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
                clf = LogisticRegression(C=C, solver='lbfgs', max_iter=5000)
            case 'ElasticNet':
                C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                clf = LogisticRegression(C=C, solver='saga', penalty='elasticnet', l1_ratio=l1_ratio, max_iter=5000)
            case 'RandomForest':
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                max_features = trial.suggest_categorical('max_features', ['auto', 'log2'])
                clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
            case 'GradientBoosting':
                learning_rate = trial.suggest_float('learning_rate', 0.0, 1.0)
                subsample = trial.suggest_float('subsample', 0.1, 0.9)
                max_depth = trial.suggest_int('max_depth', 0, 10)
                min_child_weight = trial.suggest_int('min_child_weight', 0, 25)
                clf = XGBClassifier(learning_rate=learning_rate, subsample=subsample, max_depth=max_depth, min_child_weight=min_child_weight)
            case 'SVM':
                C = trial.suggest_float('C', 1e-3, 1e+3, log=True)
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
                degree = trial.suggest_int('degree', 1, 5)
                clf = SVC(C=C, kernel=kernel, degree=degree)
            case _:
                raise ValueError("Invalid classifier specified.")

        return clf

    def _get_metric(self, cv_results):
        results = []
        for metric in self.metric:
            if f'test_{metric}' in cv_results:
                results.append(np.mean(cv_results[f'test_{metric}']))

        return tuple(results)


if __name__ == '__main__':
    # Load the dataset
    # X, y = load_iris(return_X_y=True, as_frame=True)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str)
    args = parser.parse_args()
    data_df = pd.read_csv(args.data_dir, sep='\t', index_col=0)
    X = data_df.drop(columns='label')
    y = list(data_df['label'].values)

    metrics = ['roc_auc', 'accuracy']

    objective = OptunaObjective(X, y, metrics, 10)

    # Create a study object and optimize the hyperparameters for the chosen classifier
    directions = ['maximize'] * len(metrics)
    study = create_study(directions=directions)
    study.optimize(cast(Callable[[Trial], float], objective), n_trials=10)

    # # Get the best hyperparameters and the best accuracy score
    # best_params = study.best_params
    # best_accuracy = study.best_value

    # print("Best Hyperparameters:", best_params)
    # print("Best Accuracy:", best_accuracy)
    # p"D:\embedding.tsv"rint(study.best_trials)

    joblib.dump(study, 'study.pkl')

    # TODO: Add multi-processing to the code
    # def multiprocess():
    #     mydb = mysql.connector.connect(
    #     host="localhost",
    #     user="yourusername",
    #     password="yourpassword"
    #     )

    #     mycursor = mydb.cursor()

    #     mycursor.execute("CREATE DATABASE mydatabase")

    #     Pool(5).map(init, sql_path)
