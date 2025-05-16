"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
        Building the KNN Model
        """
    knn = KNeighborsClassifier(weights='uniform', n_neighbors=18, algorithm='auto', leaf_size=3)
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

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-', marker='o')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--',
             marker='o')
    plt.plot(train_size, np.mean(fit_times, axis=1), label='Clock Times', color='red', linestyle='dashdot')
    plt.title('KNN Learning Curve: Car Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    print(fit_times)

    """
    Building the KNN Validation Curve
    """
    param_range = np.arange(0, 300, 10)
    # param_range = [-1, 0, 1]

    train_scores_2, validation_scores_2 = validation_curve(knn, x_train, y_train,
                                                           param_name='n_neighbors', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color=(0, 0, 1), linestyle='-', marker='o')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('KNN Validation Curve: Car Data')
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
    Building the KNN Model
    """
    knn = KNeighborsClassifier(weights='uniform', n_neighbors=18, algorithm='auto', leaf_size=3)
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

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-', marker='o')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--', marker='o')
    plt.title('KNN Learning Curve: Red Wine Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the KNN Validation Curve
    """
    param_range = np.arange(0, 300, 10)
    # param_range = [-1, 0, 1]

    train_scores_2, validation_scores_2 = validation_curve(knn, x_train, y_train,
                                                           param_name='n_neighbors', param_range=param_range, cv=5,
                                                           scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color=(0, 0, 1), linestyle='-', marker='o')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('KNN Validation Curve: Red Wine Data')
    plt.xlabel('Hyperparameter: Kernel')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_project_redwine()