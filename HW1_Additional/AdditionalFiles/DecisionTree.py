"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier


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
    Building the DT Model
    """
    # clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=10, min_samples_split=10,
    #                                      min_samples_leaf=5)
    dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=8)

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

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-', marker='o')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--', marker='o')
    # plt.title('Decision Tree Learning Curve: Car Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the DT Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 21, 1)
    param_range_2 = np.arange(0, 1.4, .1)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(dt_entropy, x_train, y_train,
                                                           param_name='max_depth',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_3, validation_scores_3 = validation_curve(dt_entropy, x_train, y_train,
                                                           param_name='max_features',
                                                           param_range=param_range_2, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 2)
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores: Max Depth', color='blue', linestyle='-', marker='o')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Max Depth', color='orange',
             linestyle='--', marker='o')
    # plt.plot(param_range_2, np.mean(train_scores_3, axis=1), label='Train Scores: Min Samples', color=(0, .5, 1), linestyle='-', marker='o')
    # plt.plot(param_range_2, np.mean(validation_scores_3, axis=1), label='Validation Min Samples', color=(1, .88, 0),
    #          linestyle='--', marker='o')
    # plt.title('Decision Tree Validation Curve: Car Data')
    plt.xlabel('Hyperparameters: Max Depth')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    # plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores: Max Depth', color='blue', linestyle='-',
    #          marker='o')
    # plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Max Depth', color='orange',
    #          linestyle='--', marker='o')
    plt.plot(param_range_2, np.mean(train_scores_3, axis=1), label='Train Scores: Max Features', color='blue',
             linestyle='-', marker='o')
    plt.plot(param_range_2, np.mean(validation_scores_3, axis=1), label='Validation Max Features', color='orange',
             linestyle='--', marker='o')
    # plt.title('Decision Tree Validation Curve: Car Data')
    plt.xlabel('Hyperparameters: Max Features')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.suptitle("Decision Tree Learning and Validation Curves: Car Data")
    plt.tight_layout()

    plt.show()


def test_project_redwine():
    df_wine = get_data_redwine()
    X = df_wine.values[:, :-1]
    Y = df_wine.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=7)

    """
    Building the DT Model
    """

    # dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=3, min_samples_split=10,
    #                                     min_samples_leaf=5)
    dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=3, max_features=4)
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

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_size, np.mean(train_scores, axis=1), label='Train Scores', color='blue', linestyle='-', marker='o')
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--', marker='o')
    # plt.title('Decision Tree Learning Curve: Wine Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the DT Validation Curve
    """
    param_range_2 = np.arange(0, 1, .1)
    param_range = np.arange(0, 21, 1)
    # print(param_range)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(dt_entropy, x_train, y_train,
                                                           param_name='max_depth',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_3, validation_scores_3 = validation_curve(dt_entropy, x_train, y_train,
                                                           param_name='max_features',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 2)
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color='blue', linestyle='-', marker='o')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange', linestyle='--', marker='o')
    # plt.title('Decision Tree Validation Curve: Red Wine Data')
    plt.xlabel('Hyperparameter: Max Depth')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(param_range, np.mean(train_scores_3, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(param_range, np.mean(validation_scores_3, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    # plt.title('Decision Tree Validation Curve: Red Wine Data')
    plt.xlabel('Hyperparameter: Max Features')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.suptitle("Decision Tree Learning and Validation Curves: Red Wine Data")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_project_redwine()