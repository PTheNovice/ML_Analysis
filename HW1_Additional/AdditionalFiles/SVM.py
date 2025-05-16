"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, learning_curve, validation_curve



def get_data_redwine():
    df = pd.read_csv('data/FinalWine_Red.csv')
    return df

# def get_data_rice():
#     df = pd.read_csv('data/FinalRice.csv')
#     return df
#
#
# def test_project_rice():
#     df_car = get_data_rice()
#     X = df_car.values[:, :-1]
#     Y = df_car.values[:, -1]


def get_data_car():
    df = pd.read_csv('data/FinalCar.csv')
    return df


def test_project_car():
    df_car = get_data_car()
    X = df_car.values[:, :-1]
    Y = df_car.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=7)

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
    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(svm_model_2, x_train, y_train,
                                                                                         train_sizes=train_sizes, cv=5,
                                                                                         scoring='f1_weighted',
                                                                                         shuffle=True, random_state=7,
                                                                                         return_times=True)
    # print(train_size)
    # print(train_scores)
    # print(validation_scores)
    # print(fit_times)
    # print(score_times)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--')
    plt.title('SVM Learning Curve: Car Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the SVM Validation Curve
    """
    # param_range = np.arange(0, 21)
    param_range = np.arange(0, 1.2, .1)
    # param_range = [0, .1, .2, .3, .4,1, 2, 3]

    # train_scores_2, validation_scores_2 = validation_curve(svm_model_1, x_train, y_train,
    #                                                        param_name='C', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    # train_scores_3, validation_scores_3 = validation_curve(svm_model_2, x_train, y_train,
    #                                                        param_name='C', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    # train_scores_4, validation_scores_4 = validation_curve(svm_model_3, x_train, y_train,
    #                                                        param_name='C', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')

    # train_scores_2, validation_scores_2 = validation_curve(svm_model_1, x_train, y_train,
    #                                                        param_name='gamma', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    train_scores_3, validation_scores_3 = validation_curve(svm_model_2, x_train, y_train,
                                                           param_name='gamma', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    # train_scores_4, validation_scores_4 = validation_curve(svm_model_3, x_train, y_train,
    #                                                        param_name='gamma', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    # plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores: Poly', color=(0, 0, 1), linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores: Poly', color=(1, .68, 0),
    #          linestyle='--')
    plt.plot(param_range, np.mean(train_scores_3, axis=1), label='Train Scores: RBF', color=(0, 0, .7), linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_3, axis=1), label='Validation Scores: RBF', color=(1, .75, 0),
             linestyle='--')
    # plt.plot(param_range, np.mean(train_scores_4, axis=1), label='Train Scores: Sigmoid', color=(0, 0, .4),
    #          linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_4, axis=1), label='Validation Scores: Sigmoid', color=(1, .61, 0),
    #          linestyle='--')
    plt.title('SVM Validation Curve: Car Data')
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
    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(svm_model_2, x_train, y_train,
                                                                                         train_sizes=train_sizes, cv=5,
                                                                                         scoring='f1_weighted',
                                                                                         shuffle=True, random_state=7,
                                                                                         return_times=True)
    # print(train_size)
    # print(train_scores)
    # print(validation_scores)
    # print(fit_times)
    # print(score_times)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--')
    plt.title('SVM Learning Curve: Red Wine Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the SVM Validation Curve
    """
    param_range = np.arange(0, 1.2, .1)
    # param_range = [0, .1, .2, .3, .4]

    # train_scores_2, validation_scores_2 = validation_curve(svm_model_1, x_train, y_train,
    #                                                        param_name='C', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    # train_scores_3, validation_scores_3 = validation_curve(svm_model_2, x_train, y_train,
    #                                                        param_name='C', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    # train_scores_4, validation_scores_4 = validation_curve(svm_model_3, x_train, y_train,
    #                                                        param_name='C', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    # train_scores_2, validation_scores_2 = validation_curve(svm_model_1, x_train, y_train,
    #                                                        param_name='gamma', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    train_scores_3, validation_scores_3 = validation_curve(svm_model_2, x_train, y_train,
                                                           param_name='gamma', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    # train_scores_4, validation_scores_4 = validation_curve(svm_model_3, x_train, y_train,
    #                                                        param_name='gamma', param_range=param_range, cv=5,
    #                                                        scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    # plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores: Poly', color=(0, 0, 1), linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores: Poly', color=(1, .68, 0),
    #          linestyle='--')
    plt.plot(param_range, np.mean(train_scores_3, axis=1), label='Train Scores: RBF', color=(0, 0, .7), linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_3, axis=1), label='Validation Scores: RBF', color=(1, .75, 0),
             linestyle='--')
    # plt.plot(param_range, np.mean(train_scores_4, axis=1), label='Train Scores: Sigmoid', color=(0, 0, .4), linestyle='-')
    # plt.plot(param_range, np.mean(validation_scores_4, axis=1), label='Validation Scores: Sigmoid', color=(1, .61, 0),
    #          linestyle='--')
    plt.title('SVM Validation Curve: Red Wine Data')
    plt.xlabel('Hyperparameter: Kernel')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_project_redwine()