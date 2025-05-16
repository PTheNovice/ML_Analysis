

import Data_Transformation_V1 as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA, FastICA, LatentDirichletAllocation
import sklearn.random_projection as rp
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error
from sklearn.manifold import LocallyLinearEmbedding, Isomap
import seaborn as sns


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

    """
    PCA with EM and K-Means: Wine
    """
    pca = PCA(n_components=11, random_state=7)
    new_x = pca.fit_transform(x_train_wine)
    print(len(new_x[0]))
    print(len(new_x[1]))

    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(new_x)
        bic.append(gmm.bic(new_x))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_PCA_EM_Wine")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)


    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(new_x)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='Wine')
    plt.title('Elbow WCSS: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("images/step3_PCA_KMeans_Wine")
    plt.show()


    """
    ICA with EM and K-Means: Wine
    """
    # print(x_train_wine)
    ica = FastICA(n_components=11, random_state=7)
    ica.fit(x_train_wine)
    kur_val = kurtosis(ica.components_, axis=1)
    select = np.where(np.abs(kur_val) > 1)[0]
    mix = ica.mixing_
    select_mix = mix[select]
    ica_wine = np.dot(x_train_wine, select_mix.T)
    # print(select)
    # ica_wine = ica.transform(x_train_wine[:, select])
    # ica_wine = ica.fit_transform(x_train_wine)

    # print(ica_wine)


    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(ica_wine)
        bic.append(gmm.bic(ica_wine))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_ICA_EM_Wine")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(ica_wine)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='Wine')
    plt.title('Elbow WCSS: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("images/step3_ICA_KMeans_Wine")
    plt.show()



    """
    RCA with EM and K-Means: Wine
    """

    rca = rp.GaussianRandomProjection(n_components=8, compute_inverse_components=True, random_state=7)
    rca_projected = rca.fit_transform(x_train_wine)

    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(rca_projected)
        bic.append(gmm.bic(rca_projected))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_RCA_EM_Wine")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(rca_projected)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='Wine')
    plt.title('Elbow WCSS: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("images/step3_RCA_KMeans_Wine")
    plt.show()



    """
    Iso with EM and K-Means: Wine
    """

    iso = Isomap(n_components=4, eigen_solver='dense')
    iso_wine = iso.fit_transform(x_train_wine)

    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(iso_wine)
        bic.append(gmm.bic(iso_wine))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_Iso_EM_Wine")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(iso_wine)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='wine')
    plt.title('Elbow WCSS: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("images/step3_Iso_KMeans_Wine")
    plt.show()

    """
    PCA with EM and K-Means: Car
    """
    pca = PCA(n_components=16, random_state=7)
    new_x = pca.fit_transform(x_train_car)
    print(len(new_x[0]))
    print(len(new_x[1]))

    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(new_x)
        bic.append(gmm.bic(new_x))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_PCA_EM_Car")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(new_x)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='car')
    plt.title('Elbow WCSS: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("images/step3_PCA_KMeans_Car")
    plt.show()

    """
    ICA with EM and K-Means: Car
    """
    ica = FastICA(n_components=21, random_state=7)
    # print(x_train_car)
    # # ica_car = ica.fit_transform(x_train_car)
    # ica.fit(x_train_car)
    # kur_val = np.abs(ica.components_).mean(axis=1)
    # select = np.where(kur_val > 1)[0]
    # ica_car = ica.transform(x_train_car[:, select])
    # print(ica_car)
    # ica_car = ica.fit_transform(x_train_wine[2:14])
    ica.fit(x_train_car)
    kur_val = kurtosis(ica.components_, axis=1)
    select = np.where(np.abs(kur_val) > 1)[0]
    mix = ica.mixing_
    select_mix = mix[select]
    ica_car = np.dot(x_train_car, select_mix.T)

    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(ica_car)
        bic.append(gmm.bic(ica_car))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_ICA_EM_Car")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(ica_car)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='car')
    plt.title('Elbow WCSS: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("images/step3_ICA_KMeans_Car")
    plt.show()

    """
    RCA with EM and K-Means: Car
    """

    rca = rp.GaussianRandomProjection(n_components=8, compute_inverse_components=True, random_state=7)
    rca_projected = rca.fit_transform(x_train_car)

    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(rca_projected)
        bic.append(gmm.bic(rca_projected))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_RCA_EM_Car")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(rca_projected)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='CAE')
    plt.title('Elbow WCSS: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("images/step3_RCA_KMeans_Car")
    plt.show()

    """
    Iso with EM and K-Means: Car
    """

    iso = Isomap(n_components=4, eigen_solver='dense')
    iso_car = iso.fit_transform(x_train_car)

    # EM
    low_bic = np.infty
    bic = []
    n_comp_range = range(1, 40)
    for i in n_comp_range:
        gmm = GaussianMixture(n_components=i, random_state=7)
        gmm.fit(iso_car)
        bic.append(gmm.bic(iso_car))
        if bic[-1] < low_bic:
            low_bic = bic[-1]
            best_gmm = gmm

    plt.plot(n_comp_range, bic)
    plt.title('BIC: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.savefig("images/step3_Iso_EM_Car")
    plt.show()

    opt_n = np.argmin(bic) + 1
    print("optimal num of components: ", opt_n)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(iso_car)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, wcss, label='CAE')
    plt.title('Elbow WCSS: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    # plt.savefig("images/PCA_Wine")
    plt.savefig("images/step3_Iso_KMeans_Car")
    plt.show()
    #
    """ Reporting Section """

    rca = rp.GaussianRandomProjection(n_components=21, compute_inverse_components=True, random_state=7)
    rca_projected = rca.fit_transform(x_train_car)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(rca_projected)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(n_comp_range, wcss, label='CAE')
    plt.title('RP - Elbow WCSS: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)

    rca = rp.GaussianRandomProjection(n_components=8, compute_inverse_components=True, random_state=7)
    rca_projected = rca.fit_transform(x_train_wine)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(rca_projected)
        wcss.append(kmeans.inertia_)

    plt.subplot(2, 2, 2)
    plt.plot(n_comp_range, wcss, label='RWQM')
    plt.title('RP - Elbow WCSS: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)

    iso = Isomap(n_components=3, eigen_solver='dense')
    iso_car = iso.fit_transform(x_train_car)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(iso_car)
        wcss.append(kmeans.inertia_)

    plt.subplot(2, 2, 3)
    plt.plot(n_comp_range, wcss, label='CAE')
    plt.title('Isomap - Elbow WCSS: CAE')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)

    iso = Isomap(n_components=3, eigen_solver='dense')
    iso_wine = iso.fit_transform(x_train_wine)

    # K-Means
    wcss = []
    for i in n_comp_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=7)
        kmeans.fit(iso_wine)
        wcss.append(kmeans.inertia_)

    plt.subplot(2, 2, 4)
    plt.plot(n_comp_range, wcss, label='RWQM')
    plt.title('Isomap - Elbow WCSS: RWQM')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)

    plt.suptitle("K-Means for # of Clusters on Dimensionality Reduction: CAE and RWQM")
    plt.tight_layout()
    plt.savefig('images/step3_DimRed_KM_CarAndWine_Reported.png')
    plt.show()


if __name__ == "__main__":
    test_project()
