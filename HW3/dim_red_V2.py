"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""

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
from sklearn.metrics import mean_squared_error, r2_score
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
    PCA 
    """
    pca_1 = 0
    iter_wine = range(1, len(x_train_wine[0]) + 1)
    for i in iter_wine:
        pca_1 = PCA(n_components=i, random_state=7)
        pca_1.fit(x_train_wine)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(pca_1.explained_variance_) + 1), pca_1.explained_variance_, marker='o', linestyle='--', label='Wine')
    # plt.title('Scree')
    # plt.xlabel('Prinicipal Comp')
    # plt.ylabel('Explained Variance')
    # plt.grid(True)
    # plt.savefig("images/step2_PCA_Wine")
    # plt.show()

    pca_2 = 0
    iter_car = range(1, len(x_train_car[0]) + 1)
    for i in iter_car:
        pca_2 = PCA(n_components=i, random_state=7)
        pca_2.fit(x_train_car)

    # plt.subplot(2, 2, 1)
    plt.plot(range(1, len(pca_2.explained_variance_) + 1), pca_2.explained_variance_, marker='o', linestyle='--', label='Car')
    plt.title('Scree: PCA')
    plt.xlabel('# of Components')
    plt.ylabel('Explained Variance')
    plt.grid(True)
    plt.xticks(iter_car)
    # plt.savefig("images/step2_PCA_Combined")
    plt.legend()
    # plt.show()

    """
    ICA
    """
    ica_1 = 0
    kur_values_1 = []
    for i in iter_wine:
        ica_1 = FastICA(n_components=i, random_state=7)
        ica_1.fit(x_train_wine)
        kur = kurtosis(ica_1.components_, axis=1)
        avg_kurtosis = np.mean(np.abs(kur))
        kur_values_1.append(avg_kurtosis)

    # Kurtosis is maximized here, the maximum value will be chosen as the optimal components
    plt.subplot(2, 2, 2)
    plt.plot(iter_wine, kur_values_1, marker='o', linestyle='--', label='Wine')
    # plt.title('Kurtosis')
    # plt.ylabel('Average Kurtosis')
    # plt.xlabel('# of Components')
    # plt.grid(True)
    # plt.xticks(iter_wine)
    # plt.savefig("images/step2_ICA_Wine")
    # plt.show()

    ica_2 = 0
    kur_values_2 = []
    for i in iter_car:
        ica_2 = FastICA(n_components=i, random_state=7)
        ica_2.fit(x_train_car)
        kur = kurtosis(ica_2.components_, axis=1)
        avg_kurtosis = np.mean(np.abs(kur))
        kur_values_2.append(avg_kurtosis)

    # Kurtosis is maximized here, the maximum value will be chosen as the optimal components
    plt.subplot(2, 2, 2)
    plt.plot(iter_car, kur_values_2, marker='o', linestyle='--', label='Car')
    plt.title('Kurtosis: ICA')
    plt.ylabel('Average Kurtosis')
    plt.xlabel('# of Components')
    plt.grid(True)
    plt.xticks(iter_car)
    # plt.savefig("images/step2_ICA_Car")
    plt.legend()
    # plt.show()

    """
    RCA
    """
    rca_1 = 0
    recon_error_values_1 = []
    for i in iter_wine:
        rca_1 = rp.GaussianRandomProjection(n_components=i, compute_inverse_components=True, random_state=7)
        rca_projected = rca_1.fit_transform(x_train_wine)
        rca_reconstructed = rca_1.inverse_transform(rca_projected)
        recon_error = mean_squared_error(x_train_wine, rca_reconstructed)
        # recon_error = r2_score(x_train_wine, rca_reconstructed)
        recon_error_values_1.append(recon_error)

    opt_components_idx = np.argmin(np.array(recon_error_values_1))
    opt_components = iter_wine[opt_components_idx]
    #
    plt.subplot(2, 2, 3)
    plt.plot(iter_wine, recon_error_values_1, marker='o', linestyle='--', label='Wine')
    # plt.title('Reconstruction Error')
    # plt.ylabel('Error')
    # plt.xlabel('# of Components')
    # plt.grid(True)
    # plt.xticks(iter_wine)
    # plt.savefig("images/step2_RP_Wine")
    # plt.show()

    rca_2 = 0
    recon_error_values_2 = []
    for i in iter_car:
        rca_2 = rp.GaussianRandomProjection(n_components=i, compute_inverse_components=True, random_state=7)
        rca_projected = rca_2.fit_transform(x_train_car)
        rca_reconstructed = rca_2.inverse_transform(rca_projected)
        recon_error = mean_squared_error(x_train_car, rca_reconstructed)
        # recon_error = r2_score(x_train_car, rca_reconstructed)
        recon_error_values_2.append(recon_error)

    opt_components_idx = np.argmin(np.array(recon_error_values_2))
    opt_components = iter_car[opt_components_idx]
    plt.subplot(2, 2, 3)
    plt.plot(iter_car, recon_error_values_2, marker='o', linestyle='--', label='Car')
    plt.title('Reconstruction Error: RCA')
    plt.ylabel('Error')
    plt.xlabel('# of Components')
    plt.grid(True)
    plt.xticks(iter_car)
    # plt.savefig("images/step2_RP_Car")
    plt.legend()
    # plt.show()

    """
    Iso
    """

    recon_error_values_1 = []
    for i in iter_wine:
        iso = Isomap(n_components=i, eigen_solver='dense')
        iso.fit(x_train_wine)
        recon_error = iso.reconstruction_error()
        recon_error_values_1.append(recon_error)

    plt.subplot(2, 2, 4)
    plt.plot(iter_wine, recon_error_values_1, marker='o', linestyle='--', label='Wine')
    # plt.title('Reconstruction Error: Isomap')
    # plt.ylabel('Error')
    # plt.xlabel('# of Components')
    # plt.grid(True)
    # plt.xticks(iter_wine)
    # plt.savefig("images/step2_Iso_Wine")
    # plt.show()

    recon_error_values_2 = []
    for i in iter_car:
        iso = Isomap(n_components=i, eigen_solver='dense')
        iso.fit(x_train_car)
        recon_error = iso.reconstruction_error()
        recon_error_values_2.append(recon_error)

    plt.subplot(2, 2, 4)
    plt.plot(iter_car, recon_error_values_2, marker='o', linestyle='--', label='Car')
    plt.title('Reconstruction Error: Isomap')
    plt.ylabel('Error')
    plt.xlabel('# of Components')
    plt.grid(True)
    plt.legend()
    plt.xticks(iter_car)

    plt.suptitle("Dimensionality Reduction: CAE and RWQM")
    plt.tight_layout()
    plt.savefig("images/step2_Reported")
    plt.show()


if __name__ == "__main__":
    test_project()