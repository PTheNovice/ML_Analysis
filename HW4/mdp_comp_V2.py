"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""
import bettermdptools.utils.test_env
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm as tqdm
import bettermdptools as mdp
from bettermdptools.algorithms import planner, rl
from bettermdptools.utils import blackjack_wrapper, callbacks, decorators, grid_search, plots, test_env
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.plots import Plots
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.utils.grid_search import GridSearch
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
import gymnasium as gym
from bettermdptools.utils.decorators import add_to
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def v_iters_plot(data, title):
    df = pd.DataFrame(data=data)
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, legend=None).set_title(title)
    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    # plt.show()


def plot_policy(val_max, directions, size, title):
    sns.heatmap(val_max, annot=directions, fmt="", cmap=sns.color_palette("Blues", as_cmap=True), linewidths=.7,
                linecolor="black", xticklabels=[], yticklabels=[],annot_kws={"fontsize": "xx-large"}, ).set(title=title)
    img_title = f"Policy_{size[0]}x{size[1]}.png"
    # plt.show()


@add_to(Plots)
@staticmethod
def mod_plot_policy(val_max, directions, size, title):
    sns.heatmap(val_max, annot=directions, fmt="", cmap=sns.color_palette("magma_r", as_cmap=True), linewidths=.7,
                linecolor="black",).set(title=title)
    img_title = f"Policy_{size[0]}x{size[1]}.png"
    # plt.show()


def test_project():
    print("Hello HW4")


def test_frozen_lake():
    # print("Wassguud brotha!")

    direction = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    """ Large Frozen Lake"""
    size_l = (20, 20)
    size_s = (8, 8)

    np.random.seed(7)
    env_l = gym.make('FrozenLake-v7', desc=generate_random_map(size=20, p=.96), is_slippery=True)
    env_l.reset(seed=7)
    env_s = gym.make('FrozenLake-v1', map_name="8x8", render_mode=None, is_slippery=True)
    env_s.reset(seed=7)

    """ Large Frozen Lake"""
    print("Runtime: FL_L")
    V, V_track, vi = Planner(env_l.P).value_iteration(gamma=1, n_iters=200, theta=.000000001)
    P, P_track, pi = Planner(env_l.P).policy_iteration(gamma=1, n_iters=200, theta=.000000001)
    Q, Q_V, Q_pi, Q_track, Q_pi_track = RL(env_l).q_learning(gamma=.99, epsilon_decay_ratio=.75,
                                                                            n_episodes=50000, init_alpha=.5,
                                                                            min_alpha=.1, alpha_decay_ratio=.75,
                                                                            init_epsilon=1.0, min_epsilon=.9)

    # fl_seed = [7, 77, 777, 7777, 77777]
    # print("Q-Learning for FrozenLake Large")
    # Q_flr = np.zeros_like(Q)
    # Q_V_flr = np.zeros_like(Q_V)
    # Q_track_flr = np.zeros_like(Q_track)
    # Q_pi_flr = []
    # for i in range(len(fl_seed)):
    #     np.random.seed(fl_seed[i])
    #     env_lr = gym.make('FrozenLake-v7', desc=generate_random_map(size=20, p=.96), is_slippery=True)
    #     Q_fl, Q_V_fl, Q_pi_fl, Q_track_fl, Q_pi_track_fl = RL(env_lr).q_learning(gamma=.99, epsilon_decay_ratio=.9,
    #                                                                              n_episodes=30000, init_alpha=.5,
    #                                                                              min_alpha=.1, alpha_decay_ratio=.9,
    #                                                                              init_epsilon=1.0, min_epsilon=.9)
    #     Q_flr = np.add(Q_flr, Q_fl)
    #     Q_V_flr = np.add(Q_V_flr, Q_V_fl)
    #     Q_track_flr = np.add(Q_track_flr, Q_track_fl)
    #     Q_pi_flr.append(Q_pi_fl)
    #
    # Q_flr /= 5
    # Q_V_flr /= 5
    # Q_track_flr /= 5
    #
    # action_counts = {}
    # for Q_pir in Q_pi_flr:
    #     for state, action in Q_pir.items():
    #         action_counts.setdefault(state, {}).setdefault(action, 0)
    #         action_counts[state][action] += 1
    # mode_actions_ql = {}
    # for state, count_dict in action_counts.items():
    #     mode_actions_ql[state] = max(count_dict, key=count_dict.get)

    """ Convergence for Max V vs Iterations"""

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.grid(True)
    max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
    v_iters_plot(max_val_per_iter, "VI: FrozenLake - Large")
    print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))

    plt.subplot(1, 3, 2)
    plt.grid(True)
    max_val_per_iter = np.trim_zeros(np.max(P_track, axis=1), 'b')
    v_iters_plot(max_val_per_iter, "PI: FrozenLake - Large")
    print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))

    plt.subplot(1, 3, 3)
    plt.grid(True)
    # max_val_per_iter = np.trim_zeros(np.max(Q, axis=1), 'b')
    max_val_per_iter = np.max(Q_track, axis=(1, 2))
    v_iters_plot(max_val_per_iter, "Q-Learning: FrozenLake - Large")
    print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))
    # max_val_per_iter = np.max(Q_track_flr, axis=(1, 2))
    # v_iters_plot(max_val_per_iter, "Q-Learning: FrozenLake - Large")

    plt.suptitle("FrozenLake w/ Large State Space Convergence: Max V against Iterations")
    plt.tight_layout()
    plt.savefig("images/FL_Converge_L.png")
    # plt.show()


    """ Policy Plots """

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    val_max, policy_map = Plots.get_policy_map(vi, V, direction, size_l)
    plot_policy(val_max, policy_map, size_l, title="VI: FrozenLake - Large")

    plt.subplot(1, 3, 2)
    val_max, policy_map = Plots.get_policy_map(pi, P, direction, size_l)
    plot_policy(val_max, policy_map, size_l, title="PI: FrozenLake - Large")

    plt.subplot(1, 3, 3)
    val_max, policy_map = Plots.get_policy_map(Q_pi, Q_V, direction, size_l)
    plot_policy(val_max, policy_map, size_l, title="Q-Learning: FrozenLake - Large")
    # val_max, policy_map = Plots.get_policy_map(mode_actions_ql, Q_V_flr, direction, size_l)
    # plot_policy(val_max, policy_map, size_l, title="Q-Learning: FrozenLake - Large")

    plt.suptitle("FrozenLake w/ Large State Space Policy Maps")
    plt.tight_layout()
    plt.savefig("images/FL_Policy_L.png")
    # plt.show()

    """ Hyperparameter Tuning """

    # plt.subplot(2, 2, 3)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    gammas = [.2, .4, .6, .8, .95, .99, 1]
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    for i in range(len(gammas)):
        V, V_track, vi = Planner(env_l.P).value_iteration(gamma=gammas[i], n_iters=500, theta=.000000001)
        max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
        plt.plot(max_val_per_iter, label=f"Gamma: {gammas[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None, color=colors[i])
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Frozen Lake - Large: VI - Gamma Change")
    # plt.savefig("images/FL_VI_Tuning_L_Gamma.png")
    # plt.show()

    plt.subplot(1, 2, 2)
    theta = [.0000001, .000001, .00001, .0001, .001, .01, .1]
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    for i in range(len(gammas)):
        V, V_track, vi = Planner(env_l.P).value_iteration(gamma=1, n_iters=500, theta=theta[i])
        max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
        plt.plot(max_val_per_iter, label=f"Theta: {theta[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None, color=colors[i])
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Frozen Lake - Large: VI - Theta Change")
    plt.suptitle("Hyperparameter Tuning")
    plt.savefig("images/FL_VI_Tuning_Theta_L_HyperTuningTotal.png")
    # plt.show()


    # ep = [.2, .4, .6, .8, .95, .99, 1]
    # alp = [.2, .4, .6, .8, .95, .99, 1]
    # colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    # for i in range(len(ep)):
    #     Q, Q_V, Q_pi, Q_track, Q_pi_track = RL(env_l).q_learning(epsilon_decay_ratio=.99, min_alpha=alp[i], gamma=.5, n_episodes=10000)
    #     max_val_per_iter = np.trim_zeros(np.max(Q, axis=1), 'b')
    #     df = pd.DataFrame(data=max_val_per_iter)
    #     sns.set_theme(style="whitegrid")
    #     sns.lineplot(data=df, legend=None, color=colors[i])
    #     # plt.gca().lines[i].set_color(colors[i])
    # plt.title("Frozen Lake - Large: QLearning - Epsilon Change")
    # plt.show()

    """ Small Frozen Lake"""
    print("Runtime: FL_S")
    V, V_track, vi = Planner(env_s.P).value_iteration()
    P, P_track, pi = Planner(env_s.P).policy_iteration()
    Q, Q_V, Q_pi, Q_track, Q_pi_track = RL(env_s).q_learning(n_episodes=30000)

    # print("Q-Learning for FrozenLake Small")
    # Q_fsr = np.zeros_like(Q)
    # Q_V_fsr = np.zeros_like(Q_V)
    # Q_track_fsr = np.zeros_like(Q_track)
    # Q_pi_fsr = []
    # for i in range(len(fl_seed)):
    #     np.random.seed(fl_seed[i])
    #     env_sr = gym.make('FrozenLake-v1', map_name="4x4", render_mode=None, is_slippery=True)
    #     Q_fs, Q_V_fs, Q_pi_fs, Q_track_fs, Q_pi_track_fs = RL(env_sr).q_learning(gamma=1, epsilon_decay_ratio=.5,
    #                                                                              n_episodes=30000)
    #     Q_fsr = np.add(Q_fsr, Q_fs)
    #     Q_V_fsr = np.add(Q_V_fsr, Q_V_fs)
    #     Q_track_fsr = np.add(Q_track_fsr, Q_track_fs)
    #     Q_pi_fsr.append(Q_pi_fs)
    #
    # Q_fsr /= 5
    # Q_V_fsr /= 5
    # Q_track_fsr /= 5
    #
    # action_counts = {}
    # for Q_pis in Q_pi_fsr:
    #     for state, action in Q_pis.items():
    #         action_counts.setdefault(state, {}).setdefault(action, 0)
    #         action_counts[state][action] += 1
    # mode_actions_qs = {}
    # for state, count_dict in action_counts.items():
    #     mode_actions_qs[state] = max(count_dict, key=count_dict.get)

    """ Convergence for Max V vs Iterations"""

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.grid(True)
    max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
    v_iters_plot(max_val_per_iter, "VI: FrozenLake - Small")
    print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))

    plt.subplot(1, 3, 2)
    plt.grid(True)
    max_val_per_iter = np.trim_zeros(np.max(P_track, axis=1), 'b')
    v_iters_plot(max_val_per_iter, "PI: FrozenLake - Small")
    print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))

    plt.subplot(1, 3, 3)
    plt.grid(True)
    # max_val_per_iter = np.trim_zeros(np.max(Q, axis=1), 'b')
    max_val_per_iter = np.max(Q_track, axis=(1, 2))
    v_iters_plot(max_val_per_iter, "Q-Learning: FrozenLake - Small")
    print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))
    # max_val_per_iter = np.max(Q_track_fsr, axis=(1, 2))
    # v_iters_plot(max_val_per_iter, "Q-Learning: FrozenLake - Small")

    plt.suptitle("FrozenLake w/ Small State Space Convergence: Max V against Iterations")
    plt.tight_layout()
    plt.savefig("images/FL_Converge_S.png")
    # plt.show()


    """ Policy Plots """

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    val_max, policy_map = Plots.get_policy_map(vi, V, direction, size_s)
    plot_policy(val_max, policy_map, size_s, title="VI: FrozenLake - Small")

    plt.subplot(1, 3, 2)
    val_max, policy_map = Plots.get_policy_map(pi, P, direction, size_s)
    plot_policy(val_max, policy_map, size_s, title="PI: FrozenLake - Small")

    plt.subplot(1, 3, 3)
    val_max, policy_map = Plots.get_policy_map(Q_pi, Q_V, direction, size_s)
    plot_policy(val_max, policy_map, size_s, title="Q-Learning: FrozenLake - Small")
    # val_max, policy_map = Plots.get_policy_map(mode_actions_qs, Q_V_fsr, direction, size_s)
    # plot_policy(val_max, policy_map, size_s, title="Q-Learning: FrozenLake - Small")

    plt.suptitle("FrozenLake w/ Small State Space Policy Maps")
    plt.tight_layout()
    plt.savefig("images/FL_Policy_S.png")
    # plt.show()


    """ Hyperparameter Tuning """

    gammas = [.2, .4, .6, .8, .95, .99, 1]
    # colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    # for i in range(len(gammas)):
    #     V, V_track, vi = Planner(env_s.P).value_iteration(gamma=gammas[i], n_iters=100, theta=.000000001)
    #     max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
    #     df = pd.DataFrame(data=max_val_per_iter)
    #     sns.set_theme(style="whitegrid")
    #     sns.lineplot(data=df, legend=None, color=colors[i])
    #     plt.gca().lines[i].set_color(colors[i])
    # plt.title("Frozen Lake - Large: VI - Gamma Change")
    # plt.savefig("images/FL_VI_Tuning_S.png")
    # plt.show()


    ep = [.2, .4, .6, .8, .95, .99, 1]
    # colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    # for i in range(len(ep)):
    #     Q, Q_V, Q_pi, Q_track, Q_pi_track = RL(env_s).q_learning(epsilon_decay_ratio=ep[i], gamma=.5, n_episodes=100000)
    #     max_val_per_iter = np.trim_zeros(np.max(Q, axis=1), 'b')
    #     df = pd.DataFrame(data=max_val_per_iter)
    #     sns.set_theme(style="whitegrid")
    #     sns.lineplot(data=df, legend=None, color=colors[i])
    #     plt.gca().lines[i].set_color(colors[i])
    # plt.title("Frozen Lake - Large: QLearning - Epsilon Change")
    # plt.show()

    # alp = [.2, .4, .6, .8, .95, .99, 1]
    # colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    # for i in range(len(ep)):
    #     Q, Q_V, Q_pi, Q_track, Q_pi_track = RL(env_s).q_learning(min_epsilon=ep[i], min_alpha=alp[i], gamma=.5,
    #                                                              n_episodes=30000)
    #     max_val_per_iter = np.trim_zeros(np.max(Q, axis=1), 'b')
    #     df = pd.DataFrame(data=max_val_per_iter)
    #     sns.set_theme(style="whitegrid")
    #     sns.lineplot(data=df, legend=None, color=colors[i])
    #     plt.gca().lines[i].set_color(colors[i])
    # plt.title("Frozen Lake - Large: QLearning - Epsilon Change")
    # plt.show()

    """ Blackjack """

    actions = {0: "S", 1: "H"}
    size = (29, 10)

    """ Models """
    print("Runtime: BJ_S")
    env_s = gym.make('Blackjack-v1', render_mode=None)
    blackjack_s = BlackjackWrapper(env_s)

    V, V_track, vi = Planner(blackjack_s.P).value_iteration()
    P, P_track, pi = Planner(blackjack_s.P).policy_iteration()
    Q, Q_V, qi, Q_track, Q_pi_track = RL(blackjack_s).q_learning(gamma=.99, epsilon_decay_ratio=.75,
                                                                                    n_episodes=5000,
                                                                                    init_alpha=.9, min_alpha=.1,
                                                                                    alpha_decay_ratio=.9, init_epsilon=1.0,
                                                                                    min_epsilon=.9)

    # print("Q-Learning for Blackjack")
    # Q_j = np.zeros_like(Q)
    # Q_V_j = np.zeros_like(Q_V)
    # Q_track_j = np.zeros_like(Q_track)
    # Q_pi_j = []
    # for i in range(len(fl_seed)):
    #     np.random.seed(fl_seed[i])
    #     env_sr = gym.make('FrozenLake-v1', map_name="4x4", render_mode=None, is_slippery=True)
    #     Q, Q_V, Q_pi, Q_track, Q_pi_track = RL(env_sr).q_learning(gamma=1, epsilon_decay_ratio=.5,
    #                                                               n_episodes=30000)
    #     Q_j = np.add(Q_j, Q)
    #     Q_V_j = np.add(Q_V_j, Q_V)
    #     Q_track_j = np.add(Q_track_j, Q_track)
    #     Q_pi_j.append(Q_pi)
    #
    # Q_j /= 5
    # Q_V_j /= 5
    # Q_track_j /= 5
    #
    # action_counts = {}
    # for Q_pij in Q_pi_j:
    #     for state, action in Q_pij.items():
    #         action_counts.setdefault(state, {}).setdefault(action, 0)
    #         action_counts[state][action] += 1
    # mode_actions_qj = {}
    # for state, count_dict in action_counts.items():
    #     mode_actions_qj[state] = max(count_dict, key=count_dict.get)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    gammas = [.2, .4, .6, .8, .95, .99, 1]
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    for i in range(len(gammas)):
        V, V_track, vi = Planner(blackjack_s.P).value_iteration(gamma=gammas[i], n_iters=500, theta=.000000001)
        max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
        plt.plot(max_val_per_iter, label=f"Gamma: {gammas[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None, color=colors[i])
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Blackjack -Small: VI - Gamma Change")
    # plt.savefig("images/FL_VI_Tuning_L_Gamma.png")
    # plt.show()

    plt.subplot(1, 2, 2)
    theta = [.0000001, .000001, .00001, .0001, .001, .01, .1]
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    for i in range(len(gammas)):
        V, V_track, vi = Planner(blackjack_s.P).value_iteration(gamma=1, n_iters=500, theta=theta[i])
        max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
        plt.plot(max_val_per_iter, label=f"Gamma: {theta[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None, color=colors[i])
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Blackjack -Small: VI - Theta Change")
    plt.suptitle("Hyperparameter Tuning")
    plt.savefig("images/BJ_VI_Tuning_Theta_L_HyperTuningTotal.png")
    # plt.show()

    # """ Convergence for Max V vs Iterations"""
    #
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 3, 1)
    # plt.grid(True)
    # max_val_per_iter = np.trim_zeros(np.max(V_track, axis=1), 'b')
    # v_iters_plot(max_val_per_iter, "VI: Blackjack")
    # print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))
    #
    # plt.subplot(1, 3, 2)
    # plt.grid(True)
    # max_val_per_iter = np.trim_zeros(np.max(P_track, axis=1), 'b')
    # v_iters_plot(max_val_per_iter, "PI: Blackjack")
    # print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))
    #
    # plt.subplot(1, 3, 3)
    # plt.grid(True)
    # # max_val_per_iter = np.trim_zeros(np.max(Q, axis=1), 'b')
    # # v_iters_plot(max_val_per_iter, "Q-Learning: Blackjack")
    # max_val_per_iter = np.max(Q_track, axis=(1, 2))
    # # max_val_per_iter = np.max(Q_track_j, axis=(1, 2))
    # v_iters_plot(max_val_per_iter, "Q-Learning: Blackjack - Small")
    # print("Iteration:", next((index for index, value in enumerate(max_val_per_iter) if value >= .99), None))
    #
    # plt.suptitle("Blackjack w/ Small State Space Convergence: Max V against Iterations")
    # plt.tight_layout()
    # plt.savefig("images/BJ_Converge_S.png")
    # plt.show()
    #
    # """ Policy Plots """
    #
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 3, 1)
    # val_max, policy = Plots.get_policy_map(vi, V, actions, size)
    # # Plots.plot_policy(val_max, policy, size, "VI: Blackjack")
    # Plots.mod_plot_policy(val_max, policy, size, "VI: Blackjack")
    #
    # plt.subplot(1, 3, 2)
    # val_max, policy = Plots.get_policy_map(pi, P, actions, size)
    # # Plots.plot_policy(val_max, policy, size, "PI: Blackjack")
    # Plots.mod_plot_policy(val_max, policy, size, "PI: Blackjack")
    #
    # plt.subplot(1, 3, 3)
    # val_max, policy = Plots.get_policy_map(qi, Q_V, actions, size)
    # Plots.mod_plot_policy(val_max, policy, size, "Q-Learning: Blackjack")
    # # val_max, policy_map = Plots.get_policy_map(mode_actions_qj, Q_V_j, actions, size)
    # # Plots.mod_plot_policy(val_max, policy_map, size, title="Q-Learning: Blackjack - Small")
    #
    # plt.suptitle("Blackjack w/ Small State Space Policy Maps")
    # plt.tight_layout()
    # plt.savefig("images/BJ_Policy_S.png")
    # plt.show()


if __name__ == "__main__":
    # test_project()
    test_frozen_lake()
    # test_blackjack()
    # taxi()