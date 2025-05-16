"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import Data_Transformation_V1 as data
import DecisionTree_V1 as dt
import EnsembleBoosting_V1 as eb_dt
import NeuralNetworks_V1 as nn
import KNN_V1 as knn
import SVM_V1 as svm


def run_program():
    data.test_car()
    data.test_wine_redonly()
    dt.test_project()
    eb_dt.test_project()
    knn.test_project()
    svm.test_project()
    nn.test_project()



if __name__ == '__main__':
    run_program()