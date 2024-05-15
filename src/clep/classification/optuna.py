from typing import Callable, cast

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet


class OptunaObjective:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classifier = None

    def __call__(self, trial):
        self.classifier = trial.suggest_categorical('classifier', ['SVM', 'ElasticNet'])
        # Define the hyperparameters to optimize based on the suggested classifier
        if self.classifier == 'SVM':
            C = trial.suggest_loguniform('C', 0.01, 10.0)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            degree = trial.suggest_int('degree', 1, 5)
            clf = SVC(C=C, kernel=kernel, degree=degree)
        elif self.classifier == 'ElasticNet':
            alpha = trial.suggest_loguniform('alpha', 0.01, 1.0)
            l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
            clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        else:
            raise ValueError("Invalid classifier specified.")

        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Evaluate the classifier on the validation set
        y_pred = clf.predict(X_val)

        # Calculate the accuracy score
        accuracy = accuracy_score(y_val, y_pred)

        return accuracy


# Load the dataset
X, y = load_iris(return_X_y=True)

objective = OptunaObjective(X, y)

# Create a study object and optimize the hyperparameters for the chosen classifier
study = optuna.create_study(direction='maximize')
study.optimize(cast(Callable[[optuna.Trial], float], objective), n_trials=100)

# Get the best hyperparameters and the best accuracy score
best_params = study.best_params
best_accuracy = study.best_value

print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)
