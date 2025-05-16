
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer


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
    # print(x_tr_car)
    scalar.fit(x_tr_car)
    x_train_car = scalar.transform(x_tr_car)
    # print(x_train_car)
    x_test_car = scalar.transform(x_te_car)

    df_wine = get_data_redwine()
    X_wine = df_wine.values[:, :-1]
    Y_wine = df_wine.values[:, -1]

    # x_train_wine, x_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, Y_wine, test_size=.2,
    #                                                                         random_state=7)
    x_tr_wine, x_te_wine, y_train_wine, y_test_wine = train_test_split(X_wine, Y_wine, test_size=.2,
                                                                       random_state=7)
    # print(x_tr_wine)
    scalar.fit(x_tr_wine)
    x_train_wine = scalar.transform(x_tr_wine)
    # print(x_train_wine)
    x_test_wine = scalar.transform(x_te_wine)

    """
    Building the Boost Model
    """
    # dt_entropy_car = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=6, max_features=1)
    dt_entropy_car = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=8, max_leaf_nodes=11)
    # boost_dt_car = AdaBoostClassifier(dt_entropy_car, n_estimators=4, random_state=7, learning_rate=3)
    boost_dt_car = AdaBoostClassifier(dt_entropy_car, n_estimators=20, random_state=7, learning_rate=1.6)
    boost_dt_car_2 = AdaBoostClassifier(dt_entropy_car, n_estimators=4, random_state=7)
    boost_dt_car_3 = AdaBoostClassifier(dt_entropy_car, random_state=7, learning_rate=3)

    # dt_entropy_wine = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=3, max_features=8)
    dt_entropy_wine = DecisionTreeClassifier(criterion="entropy", random_state=7, max_depth=4, max_leaf_nodes=6)
    # boost_dt_wine = AdaBoostClassifier(dt_entropy_wine, n_estimators=8, random_state=7, learning_rate=3)
    boost_dt_wine = AdaBoostClassifier(dt_entropy_wine, n_estimators=40, random_state=7, learning_rate=.4)
    boost_dt_wine_2 = AdaBoostClassifier(dt_entropy_wine, n_estimators=8, random_state=7)
    boost_dt_wine_3 = AdaBoostClassifier(dt_entropy_wine, random_state=7, learning_rate=3)

    """
    Building the Boost Learning Curve
    """

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_car, train_scores_car, validation_scores_car, fit_times_car, score_times_car = learning_curve(
        boost_dt_car, x_train_car, y_train_car,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    """ Max Fit Time"""
    print("Ensemble Car Evaluation Fit Time: ", np.max(fit_times_car))

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_size_car, np.mean(train_scores_car, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_car, np.mean(validation_scores_car, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('Boosted Decision Tree Learning Curve: Car Evaluation')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_wine, train_scores_wine, validation_scores_wine, fit_times_wine, score_times_wine = learning_curve(
        boost_dt_wine, x_train_wine, y_train_wine,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    """ Max Fit Time"""
    print("Ensemble Wine Quality Fit Time: ", np.max(fit_times_wine))

    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 2)
    plt.plot(train_size_wine, np.mean(train_scores_wine, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('Boosted Decision Tree Learning Curve: Red Wine Quality')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show

    plt.suptitle("Boosted Decision Tree Learning Curves")
    plt.tight_layout()
    plt.savefig('images/Ensemble_LearningCurve.png')
    # plt.show()

    """
    Building the Boost Validation Curve
    """
    # param_range = np.linspace(.01, 1.0, 10)
    param_range = np.arange(0, 100, 5)
    # param_range = np.arange(0, 40, 1)
    # param_range_2 = np.arange(0, 20, 1)
    param_range_2 = np.arange(0, 2, .1)
    # param_range_2 = np.linspace(.01, 1.0, 20)

    train_scores_2_car, validation_scores_2_car = validation_curve(boost_dt_car_2, x_train_car, y_train_car,
                                                                   param_name='n_estimators',
                                                                   param_range=param_range, cv=5, scoring='f1_weighted')
    train_scores_3_car, validation_scores_3_car = validation_curve(boost_dt_car_3, x_train_car, y_train_car,
                                                                   param_name='learning_rate',
                                                                   param_range=param_range_2, cv=5,
                                                                   scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(param_range, np.mean(train_scores_2_car, axis=1), label='Train', color='blue',
             linestyle='-', marker='o')
    plt.plot(param_range, np.mean(validation_scores_2_car, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Boosted Decision Tree Validation Curve: Car Evaluation')
    plt.xlabel('Hyperparameters: N_Estimators')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(param_range_2, np.mean(train_scores_3_car, axis=1), label='Train', color='blue',
             linestyle='-', marker='o')
    plt.plot(param_range_2, np.mean(validation_scores_3_car, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Boosted Decision Tree Validation Curve: Car Evaluation')
    plt.xlabel('Hyperparameters: Learning Rate')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    param_range_2 = np.arange(0, 2, .1)
    # param_range_2 = np.linspace(.01, 1.0, 20)
    param_range = np.arange(0, 100, 5)
    # param_range = np.arange(0, 40, 1)
    # param_range_2 = np.arange(0, 20, 1)
    # param_range_2 = np.arange(0, 50, 1)
    # print(param_range)

    train_scores_2_wine, validation_scores_2_wine = validation_curve(boost_dt_wine_2, x_train_wine, y_train_wine,
                                                                     param_name='n_estimators',
                                                                     param_range=param_range, cv=5,
                                                                     scoring='f1_weighted')
    train_scores_3_wine, validation_scores_3_wine = validation_curve(boost_dt_wine_3, x_train_wine, y_train_wine,
                                                                     param_name='learning_rate',
                                                                     param_range=param_range, cv=5,
                                                                     scoring='f1_weighted')
    # print(train_scores_2, validation_scores_2)

    # plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 2)
    plt.plot(param_range, np.mean(train_scores_2_wine, axis=1), label='Train', color='blue', linestyle='-',
             marker='o')
    plt.plot(param_range, np.mean(validation_scores_2_wine, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Boosted Decision Tree Validation Curve: Red Wine Quality')
    plt.xlabel('Hyperparameter: N_Estimators')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(param_range_2, np.mean(train_scores_3_wine, axis=1), label='Train', color='blue', linestyle='-',
             marker='o')
    plt.plot(param_range_2, np.mean(validation_scores_3_wine, axis=1), label='Validation', color='orange',
             linestyle='--', marker='o')
    plt.title('Boosted Decision Tree Validation Curve: Red Wine Quality')
    plt.xlabel('Hyperparameter: Learning Rate')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.suptitle("Boosted Decision Tree Validation Curves")
    plt.tight_layout()
    plt.savefig('images/Ensemble_ValidationCurve.png')
    # plt.show()


if __name__ == "__main__":
    test_project()
