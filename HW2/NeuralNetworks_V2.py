"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import sklearn.metrics as skmt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import mlrose_hiive
from mlrose_hiive import RHCRunner, SARunner, GARunner, NNGSRunner, NeuralNetwork
from mlrose_hiive import GeomDecay, ExpDecay, ArithDecay

def get_data_car():
    df = pd.read_csv('data/FinalCar.csv')
    return df


def test_project():
    df_car = get_data_car()
    X_car = df_car.values[:, :-1]
    Y_car = df_car.values[:, -1]

    scalar = StandardScaler()

    # x_train_car, x_test_car, y_train_car, y_test_car = train_test_split(X_car, Y_car, test_size=.2, random_state=7)
    x_tr_car, x_te_car, y_tr_car, y_te_car = train_test_split(X_car, Y_car, test_size=.2, random_state=7)
    # scalar.fit(x_tr_car)
    scalar = MinMaxScaler()
    x_train_car = scalar.fit_transform(x_tr_car)
    # x_train_car = scalar.transform(x_tr_car)
    x_test_car = scalar.transform(x_te_car)

    one_hot = OneHotEncoder()
    y_train_car = one_hot.fit_transform(y_tr_car.reshape(-1, 1)).todense()
    y_test_car = one_hot.transform(y_te_car.reshape(-1, 1)).todense()


    """
    Building the RHC Model
    """
    # The NNGSRunner if wanted
    # grid_search_parameters = {'max_iters': [1000], 'learning_rate': [.1, .5, .9], 'activation': [mlrose_hiive.relu],
    # 'restarts': [1, 5, 10, 20], 'hidden_layer_sizes':[[80,]]}
    #
    # nnr = NNGSRunner(x_train=x_train_car, y_train=y_train_car, x_test=x_test_car, y_test=y_test_car, experiment_name='nn_rhc',
    #                  algorithm=mlrose_hiive.algorithms.random_hill_climb, grid_search_parameters=grid_search_parameters,
    #                  grid_search_scorer_method=skmt.accuracy_score,
    #                  iteration_list=[1000], bias=True, early_stopping=True, clip_max=5,
    #                  max_attempts=100, n_jobs=5, seed=7, generate_curves=True, output_directory=None)
    # run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()
    # cv_results_df.to_csv('name_rhc.csv', index=False)

    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=[80], activation='relu', algorithm='random_hill_climb',
                                          max_iters=1000, bias=True, is_classifier=True, learning_rate=.5,
                                          early_stopping=True, clip_max=5, max_attempts=100, random_state=7,
                                          restarts=5)
    nn_model.fit(x_train_car, y_train_car)
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size_rhc, train_scores_rhc, validation_scores_rhc, fit_times_rhc, score_times_rhc = learning_curve(
        nn_model, x_train_car, y_train_car,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(train_size_rhc, np.mean(train_scores_rhc, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_rhc, np.mean(validation_scores_rhc, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='+')

    plt.title('Neural Networks: RHC')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.savefig('images/NN_RHC_LearningCurve.png')
    # plt.show()

    """
    Building the SA Model
    """
    # The NNGSRunner if wanted
    # grid_search_parameters = {'max_iters': [1000], 'learning_rate': [.1, .5, .9],
    #                           'activation': [mlrose_hiive.relu],
    #                           'schedule': [ExpDecay(1), GeomDecay(1)],
    #                           'hidden_layer_sizes': [[80, ]]}
    #
    # nnr = NNGSRunner(x_train=x_train_car, y_train=y_train_car, x_test=x_test_car, y_test=y_test_car,
    #                  experiment_name='nn_sa',
    #                  algorithm=mlrose_hiive.algorithms.simulated_annealing,
    #                  grid_search_parameters=grid_search_parameters,
    #                  grid_search_scorer_method=skmt.accuracy_score, iteration_list=[50, 150, 200],
    #                  bias=True, early_stopping=False, clip_max=5, max_attempts=200, seed=7,
    #                  generate_curves=True, output_directory=None)
    # run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()
    # cv_results_df.to_csv('name_sa.csv', index=False)

    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=[80], activation='relu', algorithm='simulated_annealing',
                                          max_iters=1000, bias=True, is_classifier=True, learning_rate=.5,
                                          early_stopping=True, clip_max=5, max_attempts=100, random_state=7,
                                          schedule=GeomDecay(7))
    nn_model.fit(x_train_car, y_train_car)
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size_sa, train_scores_sa, validation_scores_sa, fit_times_sa, score_times_sa = learning_curve(
        nn_model, x_train_car, y_train_car,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    # plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 2)
    plt.plot(train_size_sa, np.mean(train_scores_sa, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_sa, np.mean(validation_scores_sa, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='+')

    plt.title('Neural Networks: SA')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.savefig('images/NN_SA_LearningCurve.png')
    # plt.show()

    """
    Building the GA Model
    """
    # grid_search_parameters = {'max_iters': [1000], 'learning_rate': [.1, .5, .9], 'activation': [mlrose_hiive.relu],
    #                           'population_sizes': [25, 75, 100], 'mutation_rates': [.1, .5, .9],
    #                           'hidden_layer_sizes':[[80,]]}
    # nnr = NNGSRunner(x_train=x_train_car, y_train=y_train_car, x_test=x_test_car, y_test=y_test_car,
    #                  experiment_name='nn_rhc',
    #                  algorithm=mlrose_hiive.algorithms.genetic_alg, grid_search_parameters=grid_search_parameters,
    #                  grid_search_scorer_method=skmt.accuracy_score, iteration_list=[1000],
    #                  bias=True, early_stopping=True, clip_max=5, max_attempts=200, n_jobs=5, seed=7,
    #                  generate_curves=True, output_directory=None)
    # run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()
    # cv_results_df.to_csv('name_ga.csv', index=False)

    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=[80], activation='relu', algorithm='genetic_alg',
                                          max_iters=1000, bias=True, is_classifier=True, learning_rate=.5,
                                          early_stopping=True, clip_max=5, max_attempts=100, random_state=7,
                                          mutation_prob=.45, pop_size=75)
    nn_model.fit(x_train_car, y_train_car)
    train_sizes = np.linspace(.01, 1.0, 10)
    train_size_ga, train_scores_ga, validation_scores_ga, fit_times_ga, score_times_ga = learning_curve(
        nn_model, x_train_car, y_train_car,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    # plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 3)
    plt.plot(train_size_ga, np.mean(train_scores_ga, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_ga, np.mean(validation_scores_ga, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='+')

    plt.title('Neural Networks: GA')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show()

    """
    Building the NN Model from A1
    """
    scalar.fit(x_tr_car)
    x_train_car = scalar.transform(x_tr_car)
    mlp_ann_car = MLPClassifier(hidden_layer_sizes=(80,), activation='relu', solver='adam', alpha=.15,
                                random_state=7)

    """
    Building the NN Learning Curve from A1
    """

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_car, train_scores_car, validation_scores_car, fit_times_car, score_times_car = learning_curve(
        mlp_ann_car, x_train_car, y_tr_car,
        train_sizes=train_sizes, cv=5,
        scoring='f1_weighted',
        shuffle=True, random_state=7,
        return_times=True)

    plt.subplot(2, 2, 4)
    plt.plot(train_size_car, np.mean(train_scores_car, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_car, np.mean(validation_scores_car, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.title('Neural Networks: A1 ')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.suptitle("Neural Networks Learning Curves: CAE Dataset")
    plt.tight_layout()
    plt.savefig('images/NN_A2_LearningCurves.png')
    plt.show()


if __name__ == "__main__":
    test_project()