
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
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
    Building the SVM Model
    """

    svm_model_1_car = svm.SVC(kernel='poly')
    svm_model_2_car = svm.SVC(kernel='rbf', gamma=.4)
    # svm_model_2_car = svm.SVC(kernel='rbf')
    svm_model_3_car = svm.SVC(kernel='sigmoid')

    svm_model_1_wine = svm.SVC(kernel='poly')
    svm_model_2_wine = svm.SVC(kernel='rbf', gamma=.1)
    # svm_model_2_wine = svm.SVC(kernel='rbf')
    svm_model_3_wine = svm.SVC(kernel='sigmoid')


    """
    Building the SVM Learning Curve
    """

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_car, train_scores_car, validation_scores_car, fit_times_car, score_times_car = learning_curve(
        svm_model_2_car, x_train_car, y_train_car,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    """ Max Fit Time"""
    print("SVM Car Evaluation Fit Time: ", np.max(fit_times_car))

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_size_car, np.mean(train_scores_car, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_car, np.mean(validation_scores_car, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('SVM Learning Curve: Car Evaluation')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_wine, train_scores_wine, validation_scores_wine, fit_times_wine, score_times_wine = learning_curve(
        svm_model_2_wine, x_train_wine, y_train_wine,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    """ Max Fit Time"""
    print("SVM Wine Quality Fit Time: ", np.max(fit_times_wine))

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(train_size_wine, np.mean(train_scores_wine, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('SVM Learning Curve: Red Wine Quality')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show

    plt.suptitle("SVM Learning Curves")
    plt.tight_layout()
    plt.savefig('images/SVM_LearningCurve.png')
    # plt.show()

    """
    Building the SVM Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    # param_range = np.arange(0, 21, 1)
    # param_range = np.arange(0, 5, .5)
    param_range = np.arange(0, 1.2, .1) #  for Gamma
    # param_range_2 = np.arange(0, 1.4, .1)

    train_scores_2_car, validation_scores_2_car = validation_curve(svm_model_2_car, x_train_car, y_train_car,
                                                                   param_name='gamma',
                                                                   param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2_1_car, validation_scores_2_1_car = validation_curve(svm_model_3_car, x_train_car, y_train_car,
                                                                   param_name='gamma',
                                                                   param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_2_2_car, validation_scores_2_2_car = validation_curve(svm_model_1_car, x_train_car, y_train_car,
                                                                       param_name='gamma',
                                                                       param_range=param_range, cv=5,
                                                                       scoring='f1_weighted')

    # train_scores_3_car, validation_scores_3_car = validation_curve(svm_model_2_car, x_train_car, y_train_car,
    #                                                                param_name='C',
    #                                                                param_range=param_range_2, cv=5,
    #                                                                scoring='f1_weighted')
    # train_scores_3_1_car, validation_scores_3_1_car = validation_curve(svm_model_3_car, x_train_car, y_train_car,
    #                                                                param_name='C',
    #                                                                param_range=param_range_2, cv=5,
    #                                                                scoring='f1_weighted')
    # train_scores_3_2_car, validation_scores_3_2_car = validation_curve(svm_model_1_car, x_train_car, y_train_car,
    #                                                                    param_name='C',
    #                                                                    param_range=param_range_2, cv=5,
    #                                                                    scoring='f1_weighted')

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(param_range, np.mean(train_scores_2_car, axis=1), label='Train: RBF', color='blue',
             linestyle='-', marker='o')
    plt.plot(param_range, np.mean(validation_scores_2_car, axis=1), label='Validation: RBF', color='orange',
             linestyle='--', marker='o')
    plt.plot(param_range, np.mean(train_scores_2_1_car, axis=1), label='Train: Sigmoid', color=(.7, .8, .9),
             linestyle='-', marker='X')
    plt.plot(param_range, np.mean(validation_scores_2_1_car, axis=1), label='Validation: Sigmoid', color=(1, .95, 0),
             linestyle='--', marker='X')
    plt.plot(param_range, np.mean(train_scores_2_2_car, axis=1), label='Train: Poly', color=(.1, .37, 1),
             linestyle='-', marker='s')
    plt.plot(param_range, np.mean(validation_scores_2_2_car, axis=1), label='Validation: Poly', color=(1, .81, 0),
             linestyle='--', marker='s')
    plt.title('SVM Validation Curve: Car Evaluation')
    plt.xlabel('Hyperparameters: Gamma')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    # plt.subplot(2, 2, 3)
    # plt.plot(param_range_2, np.mean(train_scores_3_car, axis=1), label='Train: RBF', color='blue',
    #          linestyle='-', marker='o')
    # plt.plot(param_range_2, np.mean(validation_scores_3_car, axis=1), label='Validation: RBF', color='orange',
    #          linestyle='--', marker='o')
    # plt.plot(param_range_2, np.mean(train_scores_3_1_car, axis=1), label='Train: Sigmoid', color=(.7, .8, .9),
    #          linestyle='-', marker='X')
    # plt.plot(param_range_2, np.mean(validation_scores_3_1_car, axis=1), label='Validation: Sigmoid', color=(1, .95, 0),
    #          linestyle='--', marker='X')
    # plt.plot(param_range_2, np.mean(train_scores_3_2_car, axis=1), label='Train: Poly', color=(.1, .37, 1),
    #          linestyle='-', marker='s')
    # plt.plot(param_range_2, np.mean(validation_scores_3_2_car, axis=1), label='Validation: Poly', color=(1, .81, 0),
    #          linestyle='--', marker='s')
    # plt.title('SVM Validation Curve: Car Evaluation')
    # plt.xlabel('Hyperparameters: C')
    # plt.ylabel('Weighted F1 Score')
    # plt.legend()

    # param_range_2 = np.arange(0, 1.4, .1)
    param_range_2 = np.arange(0, 12, 1)
    param_range = np.arange(0, 1.2, .1)
    # param_range = np.arange(0, 21, 2)
    # print(param_range)

    train_scores_2_wine, validation_scores_2_wine = validation_curve(svm_model_2_wine, x_train_wine, y_train_wine,
                                                                     param_name='gamma',
                                                                     param_range=param_range, cv=5,
                                                                     scoring='f1_weighted')
    train_scores_2_1_wine, validation_scores_2_1_wine = validation_curve(svm_model_3_wine, x_train_wine, y_train_wine,
                                                                     param_name='gamma',
                                                                     param_range=param_range, cv=5,
                                                                     scoring='f1_weighted')
    train_scores_2_2_wine, validation_scores_2_2_wine = validation_curve(svm_model_1_wine, x_train_wine, y_train_wine,
                                                                         param_name='gamma',
                                                                         param_range=param_range, cv=5,
                                                                         scoring='f1_weighted')
    # train_scores_3_wine, validation_scores_3_wine = validation_curve(svm_model_2_wine, x_train_wine, y_train_wine,
    #                                                                  param_name='C',
    #                                                                  param_range=param_range, cv=5,
    #                                                                  scoring='f1_weighted')
    # train_scores_3_1_wine, validation_scores_3_1_wine = validation_curve(svm_model_3_wine, x_train_wine, y_train_wine,
    #                                                                  param_name='C',
    #                                                                  param_range=param_range, cv=5,
    #                                                                  scoring='f1_weighted')
    # train_scores_3_2_wine, validation_scores_3_2_wine = validation_curve(svm_model_1_wine, x_train_wine, y_train_wine,
    #                                                                      param_name='C',
    #                                                                      param_range=param_range, cv=5,
    #                                                                      scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(param_range, np.mean(train_scores_2_wine, axis=1), label='Train: RBF', color='blue', linestyle='-',
             marker='o')
    plt.plot(param_range, np.mean(validation_scores_2_wine, axis=1), label='Validation: RBF', color='orange',
             linestyle='--', marker='o')
    plt.plot(param_range, np.mean(train_scores_2_1_wine, axis=1), label='Train: Sigmoid', color=(.7, .8, .9),
             linestyle='-', marker='X')
    plt.plot(param_range, np.mean(validation_scores_2_1_wine, axis=1), label='Validation: Sigmoid', color=(1, .95, 0),
             linestyle='--', marker='X')
    plt.plot(param_range, np.mean(train_scores_2_2_wine, axis=1), label='Train: Poly', color=(.1, .37, 1),
             linestyle='-', marker='s')
    plt.plot(param_range, np.mean(validation_scores_2_2_wine, axis=1), label='Validation: Poly', color=(1, .81, 0),
             linestyle='--', marker='s')
    plt.title('SVM Validation Curve: Red Wine Quality')
    plt.xlabel('Hyperparameter: Gamma')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    # plt.subplot(2, 2, 4)
    # plt.plot(param_range, np.mean(train_scores_3_wine, axis=1), label='Train', color='blue',
    #          linestyle='-',
    #          marker='o')
    # plt.plot(param_range, np.mean(validation_scores_3_wine, axis=1), label='Validation', color='orange',
    #          linestyle='--', marker='o')
    # plt.plot(param_range_2, np.mean(train_scores_3_1_wine, axis=1), label='Train: Sigmoid', color=(.7, .8, .9),
    #          linestyle='-', marker='X')
    # plt.plot(param_range_2, np.mean(validation_scores_3_1_wine, axis=1), label='Validation: Sigmoid', color=(1, .95, 0),
    #          linestyle='--', marker='X')
    # plt.plot(param_range_2, np.mean(train_scores_3_2_wine, axis=1), label='Train: Poly', color=(.1, .37, 1),
    #          linestyle='-', marker='s')
    # plt.plot(param_range_2, np.mean(validation_scores_3_2_wine, axis=1), label='Validation: Poly', color=(1, .81, 0),
    #          linestyle='--', marker='s')
    # plt.title('SVM Validation Curve: Red Wine Quality')
    # plt.xlabel('Hyperparameter: C')
    # plt.ylabel('Weighted F1 Score')
    # plt.legend()

    plt.suptitle("SVM Validation Curves")
    plt.tight_layout()
    plt.savefig('images/SVM_ValidationCurve.png')
    # plt.show()


if __name__ == "__main__":
    test_project()
