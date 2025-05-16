
import Data_Transformation_V1 as data
import clustering_V1 as cl
import dim_red_V2 as dr
import dim_red_clustering as drcl
import NeuralNetworks_Clustering_V1 as nncl
import NeuralNetworks_DimRed_V1 as nndr


def run_program():
    data.test_car()
    data.test_wine_redonly()
    cl.test_project()
    dr.test_project()
    drcl.test_project()
    nncl.test_project()
    nndr.test_project()


if __name__ == '__main__':
    run_program()
