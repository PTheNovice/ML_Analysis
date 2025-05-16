"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


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


def test_project():
    df_car = get_data_car()
    X_car = df_car.values[:, :-1]
    Y_car = df_car.values[:, -1]

    scalar = StandardScaler()

    # x_train_car, x_test_car, y_train_car, y_test_car = train_test_split(X_car, Y_car, test_size=.2, random_state=7)
    x_tr_car, x_te_car, y_train_car, y_test_car = train_test_split(X_car, Y_car, test_size=.2, random_state=7)
    scalar.fit(x_tr_car)
    x_train_car = scalar.transform(x_tr_car)
    x_test_car = scalar.transform(x_te_car)

    df_wine = get_data_redwine()
    X_wine = df_wine.values[:, :-1]
    Y_wine = df_wine.values[:, -1]

    # x_train_wine, x_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, Y_wine, test_size=.2,
    #                                                                         random_state=7)
    x_tr_wine, x_te_wine, y_train_wine, y_test_wine = train_test_split(X_wine, Y_wine, test_size=.2,
                                                                       random_state=7)
    scalar.fit(x_tr_wine)
    x_train_wine = scalar.transform(x_tr_wine)
    x_test_wine = scalar.transform(x_te_wine)

    """
    Building the NN Model
    """
    # mlp_ann_car = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='lbfgs', alpha=.15, random_state=7)
    mlp_ann_car = MLPClassifier(hidden_layer_sizes=(80,), activation='relu', solver='adam', alpha=.15,
                                random_state=7)
    mlp_ann_car_2 = MLPClassifier(activation='relu', solver='adam', random_state=7)
    mlp_ann_car_3 = MLPClassifier(activation='relu', solver='adam', random_state=7)
    # mlp_ann_wine = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='lbfgs', alpha=.4, random_state=7)
    mlp_ann_wine = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam', alpha=.4,
                                 random_state=7)
    mlp_ann_wine_2 = MLPClassifier(activation='relu', solver='adam', random_state=7)
    mlp_ann_wine_3 = MLPClassifier(activation='relu', solver='adam', random_state=7)

    mlp_ann_car_loss = MLPClassifier(hidden_layer_sizes=(80,), activation='logistic', solver='adam', alpha=.15,
                                random_state=7)
    mlp_ann_wine_loss = MLPClassifier(hidden_layer_sizes=(30,), activation='logistic', solver='adam', alpha=.4,
                                 random_state=7)
    """
    Building the NN Learning Curve
    """

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_car, train_scores_car, validation_scores_car, fit_times_car, score_times_car = learning_curve(
        mlp_ann_car, x_train_car, y_train_car,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    """ Max Fit Time"""
    print("NN Car Evaluation Fit Time: ", np.max(fit_times_car))

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_size_car, np.mean(train_scores_car, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_car, np.mean(validation_scores_car, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('Neural Networks Learning Curve: Car Evaluation')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_wine, train_scores_wine, validation_scores_wine, fit_times_wine, score_times_wine = learning_curve(
        mlp_ann_wine, x_train_wine, y_train_wine,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    """ Max Fit Time"""
    print("NN Wine Quality Fit Time: ", np.max(fit_times_wine))

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(train_size_wine, np.mean(train_scores_wine, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('Neural Networks Learning Curve: Red Wine Quality')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show

    plt.suptitle("Neural Networks Learning Curves")
    plt.tight_layout()
    plt.savefig('images/NeuralNetworks_LearningCurve.png')
    # plt.show()

    """
    Building the NN Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 400, 20)
    # param_range_2 = np.linspace(.0001, .2, 21) # worked with lbfgs
    # param_range_2 = np.linspace(.0001, 100, 21)
    param_range_2 = np.linspace(0, 1, 20)
    # param_range_2 = np.linspace(0, 1000, 21)

    train_scores_2_car, validation_scores_2_car = validation_curve(mlp_ann_car_2, x_train_car, y_train_car,
                                                                   param_name='hidden_layer_sizes',
                                                                   param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_3_car, validation_scores_3_car = validation_curve(mlp_ann_car_3, x_train_car, y_train_car,
                                                                   param_name='alpha',
                                                                   param_range=param_range_2, cv=5,
                                                                   scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(param_range, np.mean(train_scores_2_car, axis=1), label='Train', color='blue',
             linestyle='-', marker='o')
    plt.plot(param_range, np.mean(validation_scores_2_car, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Neural Networks Validation Curve: Car Evaluation')
    plt.xlabel('Hyperparameters: Hidden Layer Sizes')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(param_range_2, np.mean(train_scores_3_car, axis=1), label='Train', color='blue',
             linestyle='-', marker='o')
    plt.plot(param_range_2, np.mean(validation_scores_3_car, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Neural Networks Validation Curve: Car Evaluation')
    plt.xlabel('Hyperparameters: Alpha')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    # param_range_2 = np.linspace(.0001, 5, 21) # worked with lbfgs
    # param_range_2 = np.linspace(.0001, 100, 21)
    # param_range_2 = np.linspace(0, 1000, 21)
    param_range = np.arange(0, 400, 20)
    param_range_2 = np.linspace(0, 1, 20)
    # print(param_range)

    train_scores_2_wine, validation_scores_2_wine = validation_curve(mlp_ann_wine_2, x_train_wine, y_train_wine,
                                                                     param_name='hidden_layer_sizes',
                                                                     param_range=param_range, cv=5,
                                                                     scoring='f1_weighted')
    train_scores_3_wine, validation_scores_3_wine = validation_curve(mlp_ann_wine_3, x_train_wine, y_train_wine,
                                                                     param_name='alpha',
                                                                     param_range=param_range, cv=5,
                                                                     scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 2)
    plt.plot(param_range, np.mean(train_scores_2_wine, axis=1), label='Train', color='blue', linestyle='-',
             marker='o')
    plt.plot(param_range, np.mean(validation_scores_2_wine, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Neural Networks Validation Curve: Red Wine Quality')
    plt.xlabel('Hyperparameter: Hidden Layer Sizes')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(param_range_2, np.mean(train_scores_3_wine, axis=1), label='Train', color='blue',
             linestyle='-',
             marker='o')
    plt.plot(param_range_2, np.mean(validation_scores_3_wine, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Neural Networks Validation Curve: Red Wine Quality')
    plt.xlabel('Hyperparameter: Alpha')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.suptitle("Neural Networks Validation Curves")
    plt.tight_layout()
    plt.savefig('images/NeuralNetworks_ValidationCurve.png')
    # plt.show()

    """
    Building the NN Loss Curve
    """

    mlp_ann_car_loss.fit(x_train_car, y_train_car)
    mlp_ann_wine_loss.fit(x_train_wine, y_train_wine)
    x = mlp_ann_wine_loss.loss_curve_
    # print(x)
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_ann_car_loss.loss_curve_, label='Car', color='blue', linestyle='-')
    plt.plot(mlp_ann_wine_loss.loss_curve_, label='Wine', color='orange', linestyle='--')
    plt.title('Neural Networks Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/NeuralNetworks_LossCurve.png')
    # plt.show()


if __name__ == "__main__":
    test_project()
