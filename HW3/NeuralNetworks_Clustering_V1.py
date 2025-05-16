
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
    # x, xt, y, yt = train_test_split(X_car, Y_car, test_size=.2, random_state=7)
    # df_wine = pd.DataFrame(np.array(y))
    # print(df_wine)
    kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=7)
    k = kmeans.fit_predict(x_train_wine)
    scalar.fit(k.reshape(-1, 1))
    a = scalar.transform(k.reshape(-1, 1))
    k_train = np.concatenate((x_train_wine, a), axis=1)
    # df = pd.DataFrame(np.array(k_train))
    # print(df.shape)
    # print(df)
    # print(df_wine[0].unique())
    # d = df_wine[0].value_counts()[0]
    # e = df_wine[0].value_counts()[1]
    # f = df_wine[0].value_counts()[2]
    # g = df_wine[0].value_counts()[3]
    # # h = df_wine[0].value_counts()[7]
    # # i = df_wine[0].value_counts()[8]
    # print(d, e, f, g)
    # df[11] = df[11].round(2)
    # print(df[11].unique())
    # d = df[11].value_counts()[-0.3]
    # e = df[11].value_counts()[0.37]
    # f = df[11].value_counts()[1.03]
    # g = df[11].value_counts()[1.69]
    # h = df[11].value_counts()[-0.96]
    # i = df[11].value_counts()[-1.62]
    # print(d, e, f, g, h, i)

    gmm = GaussianMixture(n_components=23, random_state=7)
    g = gmm.fit_predict(x_train_wine)
    scalar.fit(g.reshape(-1, 1))
    b = scalar.transform(g.reshape(-1, 1))
    g_train = np.concatenate((x_train_wine, b), axis=1)
    # df = pd.DataFrame(np.array(g_train))
    # print(df.shape)
    # print(df)

    """
    Building the NN Model
    """

    mlp_ann_wine = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam', alpha=.4,
                                 random_state=7)

    # mlp_ann_wine_loss = MLPClassifier(hidden_layer_sizes=(30,), activation='logistic', solver='adam', alpha=.4,
    #                              random_state=7)
    """
    Building the NN Learning Curve
    """

    train_sizes = np.linspace(.01, 1.0, 10)

    train_size_wine, train_scores_wine, validation_scores_wine, fit_times_wine, score_times_wine = learning_curve(
        mlp_ann_wine, x_train_wine, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted', shuffle=True,
        random_state=7, return_times=True)

    train_size_wine_km, train_scores_wine_km, validation_scores_wine_km, fit_times_wine_km, score_times_wine_km = \
        learning_curve(mlp_ann_wine, k_train, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted',
                       shuffle=True, random_state=7, return_times=True)

    train_size_wine_em, train_scores_wine_em, validation_scores_wine_em, fit_times_wine_em, score_times_wine_em = \
        learning_curve(mlp_ann_wine, g_train, y_train_wine, train_sizes=train_sizes, cv=5, scoring='f1_weighted',
                       shuffle=True, random_state=7, return_times=True)

    """ Max Fit Time"""
    print("NN Wine Quality Fit Time: ", np.max(fit_times_wine))
    print("NN Wine Quality Fit Time - K-Means: ", np.max(fit_times_wine_km))
    print("NN Wine Quality Fit Time - EM: ", np.max(fit_times_wine_em))
    # print("NN Wine Quality Fit Time - RCA: ", np.max(fit_times_wine_rca))
    # print("NN Wine Quality Fit Time - Iso: ", np.max(fit_times_wine_iso))
    #
    times = [np.max(fit_times_wine), np.max(fit_times_wine_km), np.max(fit_times_wine_em)]
    plt.figure(figsize=(10, 6))
    plt.bar(x=['Baseline', 'KMeans', 'EM'], width=.8,  height=times)
    plt.title("Maximum Fit Times")
    plt.ylabel("Time (s)")
    plt.xlabel("Baseline & Clustering Algos")
    plt.savefig("images/ClusteringMaxFitTimes")
    plt.show()

    """ Plots """
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_size_wine, np.mean(train_scores_wine, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(y=((np.mean(train_scores_wine, axis=1))[-1] + (np.mean(validation_scores_wine, axis=1))[-1])/2, color='black', linestyle='--')
    plt.title('RWQM Baseline')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_size_wine, np.mean(train_scores_wine_km, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_km, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(y=((np.mean(train_scores_wine_km, axis=1))[-1] + (np.mean(validation_scores_wine_km, axis=1))[-1]) / 2,
                color='black', linestyle='--')
    plt.title('RWQM K-Means')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_size_wine, np.mean(train_scores_wine_em, axis=1), label='Train Scores', color='blue', linestyle='-',
             marker='o')
    plt.plot(train_size_wine, np.mean(validation_scores_wine_em, axis=1), label='Validation Scores', color='orange',
             linestyle='--', marker='o')
    plt.axhline(
        y=((np.mean(train_scores_wine_em, axis=1))[-1] + (np.mean(validation_scores_wine_em, axis=1))[-1]) / 2,
        color='black', linestyle='--')
    plt.title('RWQM EM')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Weighted F1 Score')
    plt.legend()

    # plt.subplot(2, 3, 4)
    # plt.plot(train_size_wine, np.mean(train_scores_wine_rca, axis=1), label='Train Scores', color='blue', linestyle='-',
    #          marker='o')
    # plt.plot(train_size_wine, np.mean(validation_scores_wine_rca, axis=1), label='Validation Scores', color='orange',
    #          linestyle='--', marker='o')
    # plt.axhline(
    #     y=((np.mean(train_scores_wine_rca, axis=1))[-1] + (np.mean(validation_scores_wine_rca, axis=1))[-1]) / 2,
    #     color='black', linestyle='--')
    # plt.title('RWQM RCA')
    # plt.xlabel('Training Sample Size')
    # plt.ylabel('Weighted F1 Score')
    # plt.legend()
    #
    # plt.subplot(2, 3, 5)
    # plt.plot(train_size_wine, np.mean(train_scores_wine_iso, axis=1), label='Train Scores', color='blue', linestyle='-',
    #          marker='o')
    # plt.plot(train_size_wine, np.mean(validation_scores_wine_iso, axis=1), label='Validation Scores', color='orange',
    #          linestyle='--', marker='o')
    # plt.axhline(
    #     y=((np.mean(train_scores_wine_iso, axis=1))[-1] + (np.mean(validation_scores_wine_iso, axis=1))[-1]) / 2,
    #     color='black', linestyle='--')
    # plt.title('RWQM Isomap')
    # plt.xlabel('Training Sample Size')
    # plt.ylabel('Weighted F1 Score')
    # plt.legend()
    # # plt.show

    plt.suptitle("Neural Networks Learning Curves: Clustering")
    plt.tight_layout()
    plt.savefig('images/NeuralNetworks_LearningCurve_Clustering.png')
    plt.show()


    # """
    # Building the NN Loss Curve
    # """
    #
    # mlp_ann_wine_loss.fit(x_train_wine, y_train_wine)
    # a = mlp_ann_wine_loss.loss_curve_
    # mlp_ann_wine_loss.fit(pca_x_train, y_train_wine)
    # b = mlp_ann_wine_loss.loss_curve_
    # mlp_ann_wine_loss.fit(ica_x_train, y_train_wine)
    # c = mlp_ann_wine_loss.loss_curve_
    # mlp_ann_wine_loss.fit(rca_x_train, y_train_wine)
    # d = mlp_ann_wine_loss.loss_curve_
    # mlp_ann_wine_loss.fit(iso_x_train, y_train_wine)
    # e = mlp_ann_wine_loss.loss_curve_
    # # print(x)
    # plt.figure(figsize=(10, 6))
    # plt.plot(a, label='Baseline', color='orange', linestyle='--')
    # plt.plot(b, label='PCA', color='red', linestyle='--')
    # plt.plot(c, label='ICA', color='blue', linestyle='--')
    # plt.plot(d, label='RCA', color='black', linestyle='--')
    # plt.plot(e, label='Isomap', color='purple', linestyle='--')
    # plt.title('Neural Networks Loss Curve: RQWM')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('images/NeuralNetworks_LossCurve.png')
    # plt.show()


if __name__ == "__main__":
    test_project()
