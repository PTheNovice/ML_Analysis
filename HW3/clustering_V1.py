

import Data_Transformation_V1 as dt
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LinearRegression


def get_data_redwine():
    df = pd.read_csv('data/FinalWine_Red.csv')
    return df


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
    x_train_car = scalar.fit_transform(x_tr_car)
    x_test_car = scalar.transform(x_te_car)

    df_wine = get_data_redwine()
    X_wine = df_wine.values[:, :-1]
    Y_wine = df_wine.values[:, -1]

    # x_train_wine, x_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, Y_wine, test_size=.2,
    #                                                                         random_state=7)
    x_tr_wine, x_te_wine, y_train_wine, y_test_wine = train_test_split(X_wine, Y_wine, test_size=.2,
                                                                       random_state=7)
    scalar.fit(x_tr_wine)
    x_train_wine = scalar.fit_transform(x_tr_wine)
    x_test_wine = scalar.transform(x_te_wine)

    n_comp_range = range(1, 40)

    """
    Determine the number of clusters by looking at the 'within-cluster sum of squares' (WCSS), measures variability
    of the data points within each cluster. Kmeans only!
    """
    wcss_c = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(x_train_car)
        wcss_c.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(n_comp_range, wcss_c, label='Car')
    plt.title('K - Means: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)


    """
    Determine the number of clusters by looking at the 'Bayesian Information Criteria' (BIC), balance the fit 
    of the mmodel with the complexity. Gaussian Mixture Only!
    """
    low_bic = np.infty
    bic_c = []
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(x_train_car)
        bic_c.append(gmm.bic(x_train_car))
        if bic_c[-1] < low_bic:
            low_bic = bic_c[-1]
            best_gmm = gmm

    # for i in n_comp_range:
    #     gmm = GaussianMixture(n_components=i, random_state=7)
    #     gmm.fit(x_train_car)
    #     bic_c.append(gmm.lower_bound_)

    plt.subplot(2, 2, 3)
    plt.plot(n_comp_range, bic_c, label='Car')
    plt.title('EM: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    # plt.savefig("images/step1_BIC_car")
    # plt.show()

    opt_n = np.argmin(bic_c) + 1
    print("optimal num of components: ", opt_n)

    """
    Determine the number of clusters by looking at the 'within-cluster sum of squares' (WCSS), measures variability
    of the data points within each cluster. Kmeans only!
    """
    wcss_w = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(x_train_wine)
        wcss_w.append(kmeans.inertia_)

    plt.subplot(2, 2, 2)
    plt.plot(n_comp_range, wcss_w, label='Wine')
    plt.title('K - Means: RQWM')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    # plt.savefig("images/step1_wcss_wine")
    # plt.show()

    """
    Determine the number of clusters by looking at the 'Bayesian Information Criteria' (BIC), balance the fit 
    of the mmodel with the complexity. Gaussian Mixture Only!
    """
    low_bic = np.infty
    bic_w = []
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(x_train_wine)
        bic_w.append(gmm.bic(x_train_wine))
        if bic_w[-1] < low_bic:
            low_bic = bic_w[-1]
            best_gmm = gmm

    # for i in n_comp_range:
    #     gmm = GaussianMixture(n_components=i, random_state=7)
    #     gmm.fit(x_train_wine)
    #     bic_w.append(gmm.lower_bound_)

    plt.subplot(2, 2, 4)
    plt.plot(n_comp_range, bic_w, label='Wine')
    plt.title('EM: RQWM')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)

    # plt.legend()
    plt.suptitle("Clustering: CAE and RQWM")
    plt.tight_layout()
    plt.savefig("images/step1_Clustering_Report")
    plt.show()

    opt_n = np.argmin(bic_w) + 1
    print("optimal num of components: ", opt_n)


if __name__ == "__main__":
    test_project()
