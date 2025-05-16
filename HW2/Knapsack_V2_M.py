
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
from mlrose_hiive import QueensGenerator, TSPGenerator, FlipFlopGenerator, MaxKColorGenerator, KnapsackGenerator
from mlrose_hiive import RHCRunner, SARunner, GARunner, MIMICRunner
from mlrose_hiive import GeomDecay, ExpDecay, ArithDecay


def plot_graph_fitness_iterations(title, array1, array2, array3, array4):
    param_range = 1
    plt.figure(figsize=(10, 6))
    # plt.subplot(2, 2, 1)
    x = np.arange(0, 100, 1)

    # upper1 = array1 + std1
    # lower1 = array1 - std1
    # upper2 = array2 + std2
    # lower2 = array2 - std2
    # upper3 = array3 + std3
    # lower3 = array3 - std3
    # upper4 = array4 + std4
    # lower4 = array4 - std4

    plt.plot(array1, label='RHC', color='red',
             linestyle='-') #, marker='o')
    # plt.fill_between(np.arange(len(array1)), lower1, upper1, color='salmon', alpha=.4, label='TEST')
    plt.plot(array2, label='SA', color='green',
             linestyle='-') #, marker='+')
    # plt.fill_between(np.arange(len(array2)), lower2, upper2, color='palegreen', alpha=.4, label='TEST')
    plt.plot(array3, label='GA', color='blue',
             linestyle='-') #, marker='^')
    # plt.fill_between(np.arange(len(array2)), lower3, upper3, color='skyblue', alpha=.4, label='TEST')
    plt.plot(array4, label='MIMIC', color='black',
             linestyle='-') #, marker='*')
    # plt.fill_between(np.arange(len(array4)), lower4, upper4, color='grey', alpha=.4, label='TEST')

    # plt.title('Fitness vs Iteration: Knapsack using RHC vs SA vs GA vs MIMIC: Medium Size')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.legend()

    # plt.suptitle("Neural Networks Validation Curves")
    # plt.tight_layout()

    plt.savefig('images/Knapsack_Fit_Medium.png')
    plt.show()

    # plt.subplot(2, 2, 3)


def plot_bar_max_fitness(title, array1, array2, array3, array4):

    max_values = {'RHC': array1, 'SA': array2, 'GA': array3, 'MIMIC': array4}
    labels = list(max_values.keys())
    values = list(max_values.values())

    # plt.title('Fitness vs Iteration: Knapsack using RHC vs SA vs GA vs MIMIC: Small Size')
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='orange')
    lab = [0, 1, 2, 3]
    for la, value in zip(lab, values):
        # plt.annotate(f'{value}', (label, value), textcoords='offset points', xytext=(0, 3), ha='center')
        plt.text(x=la, y=value, s=str(value), ha='center', va='bottom')

    plt.title(title)
    plt.xlabel('RO Algorithms')
    plt.ylabel('Max Fitness')
    plt.legend()
    plt.savefig('images/Knapsack_Bar_Medium.png')
    plt.show()


def plot_bar_max_fitness_iteration(title, array1, array2, array3, array4):

    max_values = {'RHC': np.argmax(array1), 'SA': np.argmax(array2), 'GA': np.argmax(array3),
                  'MIMIC': np.argmax(array4)}
    labels = list(max_values.keys())
    values = list(max_values.values())

    # plt.title('Fitness vs Iteration: Knapsack using RHC vs SA vs GA vs MIMIC: Small Size')
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='orange')
    lab = [0, 1, 2, 3]
    for la, value in zip(lab, values):
        # plt.annotate(f'{value}', (label, value), textcoords='offset points', xytext=(0, 3), ha='center')
        plt.text(x=la, y=value, s=str(value), ha='center', va='bottom')

    plt.title(title)
    plt.xlabel('RO Algorithms')
    plt.ylabel('Max Fitness')
    plt.legend()
    plt.savefig('images/Knapsack_MaxIteration_Medium.png')
    plt.show()


def plot_graph_fevals_iterations(title, array1, array2, array3, array4):
    plt.figure(figsize=(10, 6))
    plt.plot(array1, label='RHC', color='red',
             linestyle='-') #, marker='o')
    plt.plot(array2, label='SA', color='green',
             linestyle='-') #, marker='+')
    plt.plot(array3, label='GA', color='blue',
             linestyle='-') #, marker='^')
    plt.plot(array4, label='MIMIC', color='black',
             linestyle='-') #, marker='*')

    # plt.title('MaxKColor using RHC vs SA vs GA vs MIMIC: Small Size')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('FEvals')
    plt.legend()

    plt.savefig('images/Knapsack_FEvals_Medium.png')
    plt.show()


def plot_clock_time_table(title, array1, array2, array3, array4):
    array_2d = [['RHC', array1/1000], ['SA', array2/1000], ['GA', array3/1000], ['MIMIC', array4/1000]]
    # print(array_2d)

    table = plt.table(cellText=array_2d, colLabels=['a', 'b', 'c', 'd'], loc='center')
    plt.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)
    plt.savefig('images/Knapsack_ClockTime_Medium.png')
    plt.show()


def test_knapsack():

    # problem = MaxKColorGenerator.generate(seed=7, number_of_nodes=10, max_connections_per_node=2, max_colors=2)
    problem = KnapsackGenerator.generate(seed=7, number_of_items_types=6, max_item_count=25, max_weight_per_item=7,
                                         max_value_per_item=7, max_weight_pct=.8, multiply_by_max_item_count=True)

    """
    Randomized Hill Climbing (RHC) - Small, Need Med, Large
    """
    rhc = RHCRunner(problem=problem, experiment_name='RHC', output_directory='images/RHC.csv', seed=1,
                    iteration_list=[1000], max_attempts=1000, restart_list=[7])
    df_run_stats_rhc, df_run_curves_rhc = rhc.run()
    # print(df_run_curves_rhc)
    # iter = df_run_curves['Iteration'].to_numpy()
    # fevals_rhc = df_run_curves_rhc['FEvals'].to_numpy()
    # fit_rhc = df_run_curves_rhc['Fitness'].to_numpy()
    # clock_time_rhc = sum(df_run_curves_rhc['Time'].to_numpy())
    # print("RHC: ", clock_time_rhc)
    df_sdev = df_run_curves_rhc['Fitness']

    for i in range(100, 1000, 100):
        rhc = RHCRunner(problem=problem, experiment_name='RHC', output_directory=None, seed=i,
                        iteration_list=[1000], max_attempts=1000, restart_list=[7])
        df_run_stats, df_run_curves = rhc.run()

        # print(df_run_curves)
        # df_run_curves_rhc = df_run_curves_rhc + df_run_curves
        df_run_curves_rhc['Fitness'] += df_run_curves['Fitness']
        df_run_curves_rhc['FEvals'] += df_run_curves['FEvals']
        df_run_curves_rhc['Time'] += df_run_curves['Time']

        df_sdev = pd.concat([df_sdev, df_run_curves['Fitness']], axis=1)

    # print(df_run_curves_rhc)
    df_run_curves_rhc[['Fitness', 'FEvals', 'Time']] = df_run_curves_rhc[['Fitness', 'FEvals', 'Time']] / 10
    # df_run_curves_rhc = df_run_curves_rhc/10
    # print(df_run_curves_rhc)

    # Use for Evaluation of Restarts
    # fevals_rhc = df_run_curves_rhc['FEvals'].to_numpy()
    # fit_rhc = df_run_curves_rhc['Fitness'].to_numpy()
    # clock_time_rhc = sum(df_run_curves_rhc['Time'].to_numpy())
    # After Final Tuning, for RHC only, select the specific rows that relate to the restart
    fevals_rhc = df_run_curves_rhc['FEvals'].iloc[7001:8000].to_numpy()
    fit_rhc = df_run_curves_rhc['Fitness'].iloc[7001:8000].to_numpy()
    clock_time_rhc = sum(df_run_curves_rhc['Time'].iloc[7001:8000].to_numpy())

    df_sdev.iloc[:, 0] -= df_sdev.iloc[:, 1]
    std = df_sdev.std(axis=1)
    df_sdev['std'] = std
    # print(df_sdev)
    stdev_rhc = df_sdev['std'].to_numpy()

    """
    Simulated Annealing (SA) Algo - Small, Need Med, Large
    """
    init_state_q = [3]  # For SA, if the temperature list is increased, it will run iterations based on this value
    sa = SARunner(problem=problem, experiment_name='QueensSA', output_directory=None, seed=1,
                  iteration_list=[1000], max_attempts=1000, temperature_list=init_state_q, decay_list=[ArithDecay])
    df_run_stats_sa, df_run_curves_sa = sa.run()
    # fevals_sa = df_run_curves_sa['FEvals'].to_numpy()
    # fit_sa = df_run_curves_sa['Fitness'].to_numpy()
    # clock_time_sa = sum(df_run_curves_sa['Time'].to_numpy())

    # print(df_run_curves_sa)
    df_sdev = df_run_curves_rhc['Fitness']

    for i in range(100, 1000, 100):
        sa = SARunner(problem=problem, experiment_name='QueensSA', output_directory=None, seed=i,
                      iteration_list=[1000], max_attempts=1000, temperature_list=init_state_q,
                      decay_list=[ArithDecay])
        df_run_stats, df_run_curves = sa.run()

        df_run_curves_sa['Fitness'] += df_run_curves['Fitness']
        df_run_curves_sa['FEvals'] += df_run_curves['FEvals']
        df_run_curves_sa['Time'] += df_run_curves['Time']

        df_sdev = pd.concat([df_sdev, df_run_curves['Fitness']], axis=1)

    df_run_curves_sa[['Fitness', 'FEvals', 'Time']] = df_run_curves_sa[['Fitness', 'FEvals', 'Time']] / 10
    # print(df_run_curves_sa)
    fevals_sa = df_run_curves_sa['FEvals'].to_numpy()
    fit_sa = df_run_curves_sa['Fitness'].to_numpy()
    clock_time_sa = sum(df_run_curves_sa['Time'].to_numpy())

    df_sdev.iloc[:, 0] -= df_sdev.iloc[:, 1]
    std = df_sdev.std(axis=1)
    df_sdev['std'] = std
    # print(df_sdev)
    stdev_sa = df_sdev['std'].to_numpy()

    """
    Genetic Algorithm (GA) - Small, Need Med, Large
    """
    ga = GARunner(problem=problem, experiment_name='MaxK_GA', seed=1, iteration_list=[1000],
                  max_attempts=1000, population_sizes=[50],
                  mutation_rates=[.47])
    df_run_stats_ga, df_run_curves_ga = ga.run()

    # fevals_ga = df_run_curves_ga['FEvals'].to_numpy()
    # fit_ga = df_run_curves_ga['Fitness'].to_numpy()
    # clock_time_ga = sum(df_run_curves_ga['Time'].to_numpy())
    # print("GA: ", clock_time_ga)
    df_sdev = df_run_curves_rhc['Fitness']

    for i in range(100, 1000, 100):
        ga = GARunner(problem=problem, experiment_name='MaxK_GA', seed=i, iteration_list=[1000],
                      max_attempts=1000, population_sizes=[50],
                      mutation_rates=[.47])
        df_run_stats, df_run_curves = ga.run()

        # df_run_curves_ga = df_run_curves_ga + df_run_curves
        df_run_curves_ga['Fitness'] += df_run_curves['Fitness']
        df_run_curves_ga['FEvals'] += df_run_curves['FEvals']
        df_run_curves_ga['Time'] += df_run_curves['Time']

        df_sdev = pd.concat([df_sdev, df_run_curves['Fitness']], axis=1)

    df_run_curves_ga[['Fitness', 'FEvals', 'Time']] = df_run_curves_ga[['Fitness', 'FEvals', 'Time']] / 10
    # df_run_curves_ga = df_run_curves_ga / 10
    fevals_ga = df_run_curves_ga['FEvals'].to_numpy()
    fit_ga = df_run_curves_ga['Fitness'].to_numpy()
    clock_time_ga = sum(df_run_curves_ga['Time'].to_numpy())

    df_sdev.iloc[:, 0] -= df_sdev.iloc[:, 1]
    std = df_sdev.std(axis=1)
    df_sdev['std'] = std
    # print(df_sdev)
    stdev_ga = df_sdev['std'].to_numpy()

    """
    MIMIC - Small, Need Med, Large
    """
    mimic = MIMICRunner(problem=problem, experiment_name='RHC', output_directory=None, seed=1,
                        iteration_list=[1000], max_attempts=1000, population_sizes=[1500], keep_percent_list=[.5])
    df_run_stats_mimic, df_run_curves_mimic = mimic.run()

    df_sdev = df_run_curves_rhc['Fitness']

    for i in range(100, 1000, 100):
        mimic = MIMICRunner(problem=problem, experiment_name='RHC', output_directory=None, seed=i,
                           iteration_list=[1000], max_attempts=1000, population_sizes=[1500],
                           keep_percent_list=[.5])
        df_run_stats, df_run_curves = mimic.run()

        # df_run_curves_mimic = df_run_curves_mimic + df_run_curves
        df_run_curves_mimic['Fitness'] += df_run_curves['Fitness']
        df_run_curves_mimic['FEvals'] += df_run_curves['FEvals']
        df_run_curves_mimic['Time'] += df_run_curves['Time']

        df_sdev = pd.concat([df_sdev, df_run_curves['Fitness']], axis=1)

    df_run_curves_mimic[['Fitness', 'FEvals', 'Time']] = df_run_curves_mimic[['Fitness', 'FEvals', 'Time']] / 10
    # df_run_curves_ga = df_run_curves_ga / 10
    fevals_mimic = df_run_curves_mimic['FEvals'].to_numpy()
    fit_mimic = df_run_curves_mimic['Fitness'].to_numpy()
    clock_time_mimic = sum(df_run_curves_mimic['Time'].to_numpy())
    # print("MIMIC: ", clock_time_mimic)

    df_sdev.iloc[:, 0] -= df_sdev.iloc[:, 1]
    std = df_sdev.std(axis=1)
    df_sdev['std'] = std
    # print(df_sdev)
    stdev_mimic = df_sdev['std'].to_numpy()

    """
    Graphs - Small, Need Med, Need Large
    """
    plot_graph_fitness_iterations('Fitness vs Iteration: Knapsack using RHC vs SA vs GA vs MIMIC-Medium', array1=fit_rhc,
                                  array2=fit_sa, array3=fit_ga, array4=fit_mimic)
    plot_bar_max_fitness('Max Fitness: Knapsack using RHC vs SA vs GA vs MIMIC-Medium', array1=max(fit_rhc),
                         array2=max(fit_sa), array3=max(fit_ga), array4=max(fit_mimic))
    plot_bar_max_fitness_iteration('Max Fitness Iteration: Knapsack using RHC vs SA vs GA vs MIMIC-Small',
                                   array1=fit_rhc,
                                   array2=fit_sa, array3=fit_ga, array4=fit_mimic)
    # plot_graph_fitness_iterations('Fitness vs Iteration: Knapsack using RHC vs SA vs GA vs MIMIC-Small', array1=fit_rhc,
    #                               array2=fit_sa, array3=fit_ga, array4=fit_mimic, std1=stdev_rhc, std2=stdev_sa,
    #                               std3=stdev_ga, std4=stdev_mimic)
    plot_graph_fevals_iterations('FEvals vs Iteration: Knapsack using RHC vs SA vs GA vs MIMIC-Medium', array1=fevals_rhc,
                                  array2=fevals_sa, array3=fevals_ga, array4=fevals_mimic)
    plot_clock_time_table('Clock Time: Knapsack using RHC vs SA vs GA vs MIMIC-Medium', array1=clock_time_rhc,
                                  array2=clock_time_sa, array3=clock_time_ga, array4=clock_time_mimic)


if __name__ == "__main__":
    # test_queen()
    # test_tsp()
    test_knapsack()
