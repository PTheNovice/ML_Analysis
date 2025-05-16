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
from sklearn import tree, svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split, \
    LearningCurveDisplay, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, tree, ensemble


def get_data_redwine():
    df = pd.read_csv('data/FinalWine_Red.csv')
    return df


def get_data_rice():
    df = pd.read_csv('data/FinalRice.csv')
    return df


def test_project_rice():
    df_car = get_data_rice()
    X = df_car.values[:, :-1]
    Y = df_car.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=7)

    # clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=10, min_samples_split=10,
    #                                      min_samples_leaf=5)
    dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=7)

    svm_model = svm.SVC(random_state=7)

    """
    Building the DT Learning Curve
    """

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
    # LearningCurveDisplay.from_estimator(dt_entropy, x_train, y_train, train_sizes=train_size, cv=5)
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='red', linestyle='--')
    plt.title('Decision Tree: Car Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the DT Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 20)

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
    plt.title('Decision Tree Validation Curve: Rice Data')
    plt.xlabel('Hyperparameter: Max Depth')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the SVM Learning Curve
    """
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(svm_model, x_train, y_train,
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
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--')
    plt.title('SVM: Rice Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()

    """
    Building the SVM Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    # param_range = np.arange(0, 20)
    param_range = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    # print(param_range)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(svm_model, x_train, y_train,
                                                           param_name='kernel',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange',
             linestyle='--')
    plt.title('SVM Validation Curve: Rice Data')
    plt.xlabel('Hyperparameter: Kernel')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()


def test_project_redwine():
    df_wine = get_data_redwine()
    X = df_wine.values[:, :-1]
    Y = df_wine.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=7)

    """
    Building the DT Model
    """

    dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=3, min_samples_split=10,
                                        min_samples_leaf=5)
    """
    Building the DT Learning Curve
    """
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
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--' )
    plt.title('Decision Tree: Wine Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()

    """
    Building the DT Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 21)
    # print(param_range)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(dt_entropy, x_train, y_train,
                                                           param_name='max_depth',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    # plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color='blue', linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange', linestyle='--')
    # plt.title('Decision Tree Validation Curve: Red Wine Data')
    # plt.xlabel('Hyperparameter: Max Depth')
    # plt.ylabel('Weighted F1 Score')
    # plt.legend()
    # plt.show()

    # ValidationCurveDisplay.from_estimator(clf_entropy, x_train, y_train, train_sizes=train_size, param_name="max_depth", param_range=[0, 1, 2])

    """
    Building the SVM Model
    """
    svm_model_1 = svm.SVC(kernel='poly')
    svm_model_2 = svm.SVC(kernel='rbf')
    svm_model_3 = svm.SVC(kernel='sigmoid')
    # svm_model_4 = svm.SVC(kernel='linear')
    """
    Building the SVM Learning Curve
    """
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(svm_model_1, x_train, y_train,
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

    # plt.figure(figsize=(10, 6))
    # plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-')
    # plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--')
    # plt.title('SVM Learning Curve: Red Wine Data')
    # plt.xlabel('Training Sample Size')
    # plt.ylabel('Weighted F1 Score')
    # plt.legend()
    # plt.show()

    """
    Building the SVM Validation Curve
    """
    param_range = np.arange(0, 21)

    train_scores_2, validation_scores_2 = validation_curve(svm_model_1, x_train, y_train,
                                                           param_name='C', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    train_scores_3, validation_scores_3 = validation_curve(svm_model_2, x_train, y_train,
                                                           param_name='C', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    train_scores_4, validation_scores_4 = validation_curve(svm_model_3, x_train, y_train,
                                                           param_name='C', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    # plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores: Poly', color=(0, 0, 1), linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores: Poly', color=(1, .3, 0),
    #          linestyle='--')
    # plt.plot(param_range, np.mean(train_scores_3, axis=1), label='Train Scores: RBF', color=(0, 0, .7), linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_3, axis=1), label='Validation Scores: RBF', color=(.7, .3, 0),
    #          linestyle='--')
    # plt.plot(param_range, np.mean(train_scores_4, axis=1), label='Train Scores: Sigmoid', color=(0, 0, .4), linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_4, axis=1), label='Validation Scores: Sigmoid', color=(.4, .3, 0),
    #          linestyle='--')
    # plt.title('SVM Validation Curve: Red Wine Data')
    # plt.xlabel('Hyperparameter: Kernel')
    # plt.ylabel('Weighted F1 Score')
    # plt.legend()
    # plt.show()

    """
    Building the KNN Model
    """
    knn = KNeighborsClassifier(weights='uniform')
    """
    Building the KNN Learning Curve
    """
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(knn, x_train, y_train,
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
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--')
    plt.title('KNN Learning Curve: Red Wine Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()

    """
    Building the KNN Validation Curve
    """
    param_range = np.arange(0, 21)

    train_scores_2, validation_scores_2 = validation_curve(knn, x_train, y_train,
                                                           param_name='n_neighbors', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color=(0, 0, 1), linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color=(1, .3, 0),
             linestyle='--')
    plt.title('KNN Validation Curve: Red Wine Data')
    plt.xlabel('Hyperparameter: Kernel')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_project_redwine()