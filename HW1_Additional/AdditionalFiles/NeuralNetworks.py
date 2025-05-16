"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.neural_network import MLPClassifier


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
    Building the ANN Model
    """

    # Highest f1_score so far
    # mlp_ann = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='lbfgs', random_state=7)
    # Testing
    mlp_ann = MLPClassifier(hidden_layer_sizes=(7,), activation='logistic', solver='lbfgs', random_state=7)

    # mlp_ann.fit(x_train, y_train)
    # # mlp_ann_wine.fit(x_train_wine, y_train_wine)
    # x = mlp_ann.loss_curve_
    # print(x)
    # plt.figure(figsize=(10, 6))
    # plt.plot(mlp_ann.loss_curve_, label='Car', color='blue', linestyle='-', marker='o')
    # # plt.plot(mlp_ann_wine.loss_curve_, label='Wine', color='orange', linestyle='--', marker='o')
    # plt.plot(mlp_ann.loss_curve_)
    # # plt.plot(mlp_ann_wine.loss_curve_)
    # plt.title('Neural Networks Loss Curve')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    """
    Building the ANN Learning Curve
    """
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(mlp_ann, x_train, y_train,
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
    plt.title('Artificial Neural Networks Learning Curve: Car Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the ANN Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 10, 1)
    # param_range = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    # param_range = [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,)]
    # param_range = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # print(param_range)
    # param_range = np.arange(0, 20000, 1000)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    # train_scores_2, validation_scores_2 = validation_curve(mlp_ann, x_train, y_train,
    #                                                        param_name='hidden_layer_sizes',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(mlp_ann, x_train, y_train,
                                                           param_name='alpha',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange',
             linestyle='--')
    plt.title('Artificial Neural Networks Validation Curve: Car Data')
    plt.xlabel('Hyperparameter: Hidden Layer Sizes')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()


def test_project_redwine():
    df_wine = get_data_redwine()
    X = df_wine.values[:, :-1]
    Y = df_wine.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=7)

    # x_train_tensor = torch.FloatTensor(x_train)
    # y_train_tensor = torch.LongTensor(y_train)

    """
    Building the ANN Model
    """

    # Highest f1_score so far
    # mlp_ann = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='lbfgs', random_state=7)
    # Testing
    mlp_ann = MLPClassifier(hidden_layer_sizes=(7,), activation='logistic', solver='lbfgs', random_state=7)

    """
    Building the ANN Learning Curve
    """
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size, train_scores, validation_scores, fit_times, score_times = learning_curve(mlp_ann, x_train, y_train,
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
    plt.plot(train_size, np.mean(validation_scores, axis=1), label='Validation Scores', color='orange', linestyle='--' )
    plt.title('Artificial Neural Networks Learning Curve: Red Wine Data')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the ANN Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 1, .01)
    # param_range = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    # param_range = [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
    # param_range = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # print(param_range)

    # train_scores_2, validation_scores_2 = validation_curve(clf_entropy, x_train, y_train, param_name='min_samples_split',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    # train_scores_2, validation_scores_2 = validation_curve(mlp_ann, x_train, y_train,
    #                                                        param_name='hidden_layer_sizes',
    #                                                        param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2, validation_scores_2 = validation_curve(mlp_ann, x_train, y_train,
                                                           param_name='tol',
                                                           param_range=param_range, cv=5, scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(param_range, np.mean(train_scores_2, axis=1), label='Train Scores', color='blue', linestyle='-')
    plt.plot(param_range, np.mean(validation_scores_2, axis=1), label='Validation Scores', color='orange', linestyle='--')
    plt.title('Artificial Neural Networks Validation Curve: Red Wine Data')
    plt.xlabel('Hyperparameter: Hidden Layer Sizes')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_project_redwine()