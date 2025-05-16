"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, LatentDirichletAllocation
import sklearn.random_projection as rp
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error
from sklearn.manifold import LocallyLinearEmbedding, Isomap


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
    Dimensionality Reduction Algos
    """
    pca = PCA(n_components=4, random_state=7)
    pca_x_train = pca.fit_transform(x_train_wine)
    df = pd.DataFrame(np.array(pca_x_train))
    print(df.shape)

    # ica = FastICA(n_components=10, random_state=7)
    # ica_x_train = ica.fit_transform(x_train_wine)
    ica = FastICA(n_components=11, random_state=7)
    ica.fit(x_train_wine)
    kur_val = kurtosis(ica.components_, axis=1)
    select = np.where(np.abs(kur_val) > 1)[0]
    mix = ica.mixing_
    select_mix = mix[select]
    ica_x_train = np.dot(x_train_wine, select_mix.T)
    print(ica_x_train)
    df = pd.DataFrame(np.array(ica_x_train))
    print(df.shape)

    rca = rp.GaussianRandomProjection(n_components=11, compute_inverse_components=True, random_state=7)
    rca_x_train = rca.fit_transform(x_train_wine)
    df = pd.DataFrame(np.array(rca_x_train))
    print(df.shape)

    iso = Isomap(n_components=3, eigen_solver='dense')
    iso_x_train = iso.fit_transform(x_train_wine)
    df = pd.DataFrame(np.array(iso_x_train))
    print(df.shape)

    """
    Building the NN Model
    """

    mlp_ann_wine = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam', alpha=.4,
                                 random_state=7)

    mlp_ann_wine_loss = MLPClassifier(hidden_layer_sizes=(30,), activation='logistic', solver='adam', alpha=.4,
                                 random_state=7)
    """
    Building the NN Learning Curve
    """

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_wine, train_scores_wine, validation_scores_wine, fit_times_wine, score_times_wine = learning_curve(
        mlp_ann_wine, x_train_wine, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted', shuffle=True,
        random_state=7, return_times=True)

    train_size_wine_pca, train_scores_wine_pca, validation_scores_wine_pca, fit_times_wine_pca, score_times_wine_pca = \
        learning_curve(mlp_ann_wine, pca_x_train, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted',
                       shuffle=True, random_state=7, return_times=True)

    train_size_wine_ica, train_scores_wine_ica, validation_scores_wine_ica, fit_times_wine_ica, score_times_wine_ica = \
        learning_curve(mlp_ann_wine, ica_x_train, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted',
                       shuffle=True, random_state=7, return_times=True)

    train_size_wine_rca, train_scores_wine_rca, validation_scores_wine_rca, fit_times_wine_rca, score_times_wine_rca = \
        learning_curve(mlp_ann_wine, rca_x_train, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted',
                       shuffle=True, random_state=7, return_times=True)

    train_size_wine_iso, train_scores_wine_iso, validation_scores_wine_iso, fit_times_wine_iso, score_times_wine_iso = \
        learning_curve(mlp_ann_wine, iso_x_train, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted',
                       shuffle=True, random_state=7, return_times=True)

    """ Max Fit Time"""
    print("NN Wine Quality Fit Time: ", np.max(fit_times_wine))
    print("NN Wine Quality Fit Time - PCA: ", np.max(fit_times_wine_pca))
    print("NN Wine Quality Fit Time - ICA: ", np.max(fit_times_wine_ica))
    print("NN Wine Quality Fit Time - RCA: ", np.max(fit_times_wine_rca))
    print("NN Wine Quality Fit Time - Iso: ", np.max(fit_times_wine_iso))

    times = [np.max(fit_times_wine), np.max(fit_times_wine_pca), np.max(fit_times_wine_ica), np.max(fit_times_wine_rca),
         np.max(fit_times_wine_iso)]
    plt.figure(figsize=(10, 6))
    plt.bar(x=['Baseline', 'PCA', 'ICA', 'RCA', 'Iso'], width=.8,  height=times)
    plt.title("Maximum Fit Times")
    plt.ylabel("Time (s)")
    plt.xlabel("Baseline & Dimensionality Reduction Algos")
    plt.savefig("images/DimRedMaxFitTimes")
    plt.show()

    """ Plots """
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.plot(train_size_wine, np.mean(train_scores_wine, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(y=((np.mean(train_scores_wine, axis=1))[-1] + (np.mean(validation_scores_wine, axis=1))[-1])/2, color='black', linestyle='--')
    plt.title('RWQM Baseline')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(train_size_wine, np.mean(train_scores_wine_pca, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_pca, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(y=((np.mean(train_scores_wine_pca, axis=1))[-1] + (np.mean(validation_scores_wine_pca, axis=1))[-1]) / 2,
                color='black', linestyle='--')
    plt.title('RWQM PCA')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(train_size_wine, np.mean(train_scores_wine_ica, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_ica, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(
        y=((np.mean(train_scores_wine_ica, axis=1))[2] + (np.mean(validation_scores_wine_ica, axis=1))[2]) / 2,
        color='black', linestyle='--')
    plt.title('RWQM ICA')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(train_size_wine, np.mean(train_scores_wine_rca, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_rca, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(
        y=((np.mean(train_scores_wine_rca, axis=1))[-1] + (np.mean(validation_scores_wine_rca, axis=1))[-1]) / 2,
        color='black', linestyle='--')
    plt.title('RWQM RCA')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(train_size_wine, np.mean(train_scores_wine_iso, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_iso, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(
        y=((np.mean(train_scores_wine_iso, axis=1))[-1] + (np.mean(validation_scores_wine_iso, axis=1))[-1]) / 2,
        color='black', linestyle='--')
    plt.title('RWQM Isomap')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show

    plt.suptitle("Neural Networks Learning Curves")
    plt.tight_layout()
    plt.savefig('images/NeuralNetworks_LearningCurve.png')
    plt.show()

    """ Reported Plots """
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_size_wine, np.mean(train_scores_wine, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(y=((np.mean(train_scores_wine, axis=1))[-1] + (np.mean(validation_scores_wine, axis=1))[-1]) / 2,
                color='black', linestyle='--')
    plt.title('RWQM Baseline')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_size_wine, np.mean(train_scores_wine_ica, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_ica, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(
        y=((np.mean(train_scores_wine_ica, axis=1))[6] + (np.mean(validation_scores_wine_ica, axis=1))[6]) / 2,
        color='black', linestyle='--')
    plt.title('RWQM ICA')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_size_wine, np.mean(train_scores_wine_iso, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_iso, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(
        y=((np.mean(train_scores_wine_iso, axis=1))[-1] + (np.mean(validation_scores_wine_iso, axis=1))[-1]) / 2,
        color='black', linestyle='--')
    plt.title('RWQM Isomap')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    # plt.show

    plt.suptitle("Neural Networks Learning Curves")
    plt.tight_layout()
    plt.savefig('images/NeuralNetworks_LearningCurve_Reported.png')
    plt.show()

    """
    Building the NN Loss Curve
    """

    mlp_ann_wine_loss.fit(x_train_wine, y_train_wine)
    a = mlp_ann_wine_loss.loss_curve_
    mlp_ann_wine_loss.fit(pca_x_train, y_train_wine)
    b = mlp_ann_wine_loss.loss_curve_
    mlp_ann_wine_loss.fit(ica_x_train, y_train_wine)
    c = mlp_ann_wine_loss.loss_curve_
    mlp_ann_wine_loss.fit(rca_x_train, y_train_wine)
    d = mlp_ann_wine_loss.loss_curve_
    mlp_ann_wine_loss.fit(iso_x_train, y_train_wine)
    e = mlp_ann_wine_loss.loss_curve_
    # print(x)
    plt.figure(figsize=(10, 6))
    plt.plot(a, label='Baseline', color='orange', linestyle='--')
    # plt.plot(b, label='PCA', color='red', linestyle='--')
    plt.plot(c, label='ICA', color='blue', linestyle='--')
    # plt.plot(d, label='RCA', color='black', linestyle='--')
    plt.plot(e, label='Isomap', color='purple', linestyle='--')
    plt.title('Neural Networks Loss Curve: RQWM')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/NeuralNetworks_LossCurve.png')
    plt.show()


if __name__ == "__main__":
    test_project()