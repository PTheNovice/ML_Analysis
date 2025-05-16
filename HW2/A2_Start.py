
import Data_Transformation_V1 as data
import FLipFlop_V2_S as fs
import FLipFlop_V2_M as fm
import FLipFlop_V2_L as fl
import Knapsack_V2_S as ks
import Knapsack_V2_M as km
import Knapsack_V2_L as kl
import TSP_V2_S as ts
import TSP_V2_M as tm
import TSP_V2_L as tl
import NeuralNetworks_V2 as nn


def run_program():
    data.test_car()
    fs.test_flipflop()
    fm.test_flipflop()
    fl.test_flipflop()
    ks.test_knapsack()
    km.test_knapsack()
    kl.test_knapsack()
    ts.test_tsp()
    tm.test_tsp()
    tl.test_tsp()
    nn.test_project()


if __name__ == '__main__':
    run_program()
