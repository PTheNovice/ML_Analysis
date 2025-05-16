"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import mdp_comp_V2 as mdp
import mdp_comp_V2_BJ as mdp_bj
import mdp_alg_comp__ee_FL_V1 as ee_fl
import mdp_alg_comp__ee_BJ_V1 as ee_bj


def test_run_program():
    mdp.test_frozen_lake()
    mdp_bj.test_frozen_lake()
    ee_fl.test_frozen_lake()
    ee_bj.test_frozen_lake()
    mdp.mddata.test_car()


if __name__ == '__main__':
    print("Welcome to A4!")
    test_run_program()
    print("Thanks for reviewing my code!")