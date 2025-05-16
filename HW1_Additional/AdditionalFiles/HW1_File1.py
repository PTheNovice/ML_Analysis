"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split, \
    LearningCurveDisplay, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model, tree, ensemble


def get_data_wine():
    df = pd.read_csv('data/FinalWine.csv')
    return df


def get_data_car():
    df = pd.read_csv('data/FinalCar.csv')
    return df


def test_project_car():
    df_car = get_data_car()
    X = df_car.values[:, :-1]
    Y = df_car.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=7)

    dt_entropy = DecisionTreeClassifier(criterion="gini", random_state=7, max_depth=10, min_samples_leaf=5)

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(dt_entropy, x_train, y_train,
                                                                                         train_sizes=train_sizes, cv=5,
                                                                                         scoring='f1_weighted',
                                                                                         shuffle=True, random_state=7,
                                                                                         return_times=True)
    # print(train_size)
    # print(train_scores)
    # print(validation_scores)
    # print(fit_times)
    # print(score_times)

    # LearningCurveDisplay.from_estimator(clf_entropy, x_train, y_train, train_sizes=train_size, cv=5)
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='red', linestyle='--')

    plt.title('Decision Tree: Car Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()

    # param_range = np.linspace(.01, 1.0, 10)
    #
    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    """
    Building the DT Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 21)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(dt_entropy, x_train, y_train,
                                                           param_name='max_depth',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange',
             linestyle='--')
    plt.title('Decision Tree Validation Curve: Car Data')
    plt.xlabel('Hyperparameter: Max Depth')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()


def test_project_wine():
    df_wine = get_data_wine()
    X = df_wine.values[:, :-1]
    Y = df_wine.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=7)

    clf_entropy = DecisionTreeClassifier(criterion="gini", random_state=7, max_depth=10, min_samples_leaf=5)

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(clf_entropy, x_train, y_train,
                                                                                         train_sizes=train_sizes, cv=5,
                                                                                         scoring='f1_weighted',
                                                                                         shuffle=True, random_state=7,
                                                                                         return_times=True)
    # print(train_size)
    # print(train_scores)
    # print(validation_scores)
    # print(fit_times)
    # print(score_times)

    # LearningCurveDisplay.from_estimator(clf_entropy, x_train, y_train, train_sizes=train_size, cv=5)
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--' )

    plt.title('Decision Tree: Wine Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    # param_range = np.linspace(.01, 1.0, 10)
    #
    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)
    # ValidationCurveDisplay.from_estimator(clf_entropy, x_train, y_train, train_sizes=train_size, param_name="max_depth", param_range=[0, 1, 2])

    """
    Building the Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(1, 11)
    # print(param_range)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train,
                                                           param_name='max_depth',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange',
             linestyle='--')
    plt.title('Decision Tree Validation Curve: Wine Data')
    plt.xlabel('Hyperparameter: Max Depth')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_project_wine()