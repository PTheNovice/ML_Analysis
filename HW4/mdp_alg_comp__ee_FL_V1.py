
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
import itertools


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


@staticmethod
def ql_grid_search(env, gamma, epsilon_decay, iters, init_alpha, min_alpha, alpha_decay_ratio, init_epsilon, min_epsilon):
    for i in itertools.product(gamma, epsilon_decay, iters, init_alpha, min_alpha, alpha_decay_ratio, init_epsilon, min_epsilon):
        print("running q_learning with gamma:", i[0],  "epsilon decay:", i[1],  " iterations:", i[2], "init_alpha:",
              i[3], "min_alpha:", i[4], "alpha_decay:", i[5], "init_epsilon:", i[6], "min_epsilon:", i[7])
        Q, V, pi, Q_track, pi_track = RL(env).q_learning(gamma=i[0], epsilon_decay_ratio=i[1], n_episodes=i[2],
                                                         init_alpha=i[3], min_alpha=i[4], alpha_decay_ratio=i[5],
                                                         init_epsilon=i[6], min_epsilon=i[7])
        episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
        print("Avg. episode reward: ", np.mean(episode_rewards))
        print("###################")


def test_project():
    print("Wassup Guys!")


def test_frozen_lake():
    # print("Wassguud brotha!")

    direction = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    """ Large Frozen Lake"""

    size_l = (20, 20)
    # size_l = (8, 8)

    actions = {0: "S", 1: "H"}
    size = (29, 10)

    """ Models """
    env_s = gym.make('Blackjack-v1', render_mode=None)
    blackjack_s = BlackjackWrapper(env_s)
    # env_l = gym.make('FrozenLake-v1', desc=custom_map)
    np.random.seed(7)
    env_l = gym.make('FrozenLake-v7', desc=generate_random_map(size=20, p=.96), is_slippery=True)
    env_l.reset(seed=7)
    # env_l = gym.make('FrozenLake-v1', map_name="8x8", render_mode=None, is_slippery=True)


    # x = [.1, .2, .3, .4, .6, .8, .9]
    # colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
    # for i in range(len(x)):
    #     Q_fl, Q_V_fl, Q_pi_fl, Q_track_fl, Q_pi_track_fl = RL(env_l).q_learning(gamma=.99, epsilon_decay_ratio=.9, n_episodes=20000,
    #                init_alpha=.5, min_alpha=.1, alpha_decay_ratio=.9, init_epsilon=1.0, min_epsilon=.9)
    #     # max_val_per_iter = np.trim_zeros(np.max(Q_V_fl, axis=1), 'b')
    #     max_val_per_iter = np.max(Q_track_fl, axis=(1, 2))
    #     df = pd.DataFrame(data=max_val_per_iter)
    #     sns.set_theme(style="whitegrid")
    #     sns.lineplot(data=df, legend=None, color=colors[i])
    #     # plt.gca().lines[i].set_color(colors[i])
    # plt.title("Frozen Lake - Large: VI - Gamma Change")
    # # plt.savefig("images/FL_VI_Tuning_L.png")
    # plt.show()

    """ Learning Rate """
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    x = [.1, .25, .5, .75, .9]
    n = [1000, 5000, 10000, 30000]
    colors = ['red', 'blue', 'green', 'orange', 'black']
    colors_n = ['red', 'blue', 'green', 'orange']
    plt.grid(True)
    for i in range(len(x)):
        Q_fl, Q_V_fl, Q_pi_fl, Q_track_fl, Q_pi_track_fl = RL(env_l).q_learning(gamma=.99, epsilon_decay_ratio=.75,
                                                                                    n_episodes=30000,
                                                                                    init_alpha=x[i], min_alpha=.1,
                                                                                    alpha_decay_ratio=.9, init_epsilon=1.0,
                                                                                    min_epsilon=.9)
        # max_val_per_iter = np.trim_zeros(np.max(Q_V_fl, axis=1), 'b')
        max_val_per_iter = np.max(Q_track_fl, axis=(1, 2))
        plt.plot(max_val_per_iter, label=f"Learning Rate: {x[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None)
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()

    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Frozen Lake - Large: Q-Learning - Learning Rate Change")
    # plt.savefig("images/FL_VI_Tuning_L_LearningRate.png")
    # plt.show()

    """ Epsilon Decay Rate"""
    plt.subplot(2, 2, 2)
    x = [.1, .25, .5, .75, .9]
    n = [1000, 5000, 10000, 30000]
    colors = ['red', 'blue', 'green', 'orange', 'black']
    colors_n = ['red', 'blue', 'green', 'orange']
    plt.grid(True)
    for i in range(len(x)):
        Q_fl, Q_V_fl, Q_pi_fl, Q_track_fl, Q_pi_track_fl = RL(env_l).q_learning(gamma=.99, epsilon_decay_ratio=x[i],
                                                                                n_episodes=30000,
                                                                                init_alpha=.9, min_alpha=.1,
                                                                                alpha_decay_ratio=.9, init_epsilon=1.0,
                                                                                min_epsilon=.9)
        # max_val_per_iter = np.trim_zeros(np.max(Q_V_fl, axis=1), 'b')
        max_val_per_iter = np.max(Q_track_fl, axis=(1, 2))
        plt.plot(max_val_per_iter, label=f"Epsilon Decay: {x[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None)
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()

    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Frozen Lake - Large: Q-Learning - Epsilon Decay")
    # plt.savefig("images/FL_VI_Tuning_L_EpsilonDecay.png")
    # plt.show()

    # """ Epsilon Decay Rate"""
    #
    # x = [.1, .25, .5, .75, .9]
    # n = [1000, 5000, 10000, 30000]
    # colors = ['red', 'blue', 'green', 'orange', 'black']
    # colors_n = ['red', 'blue', 'green', 'orange']
    # plt.grid(True)
    # for i in range(len(x)):
    #     Q_fl, Q_V_fl, Q_pi_fl, Q_track_fl, Q_pi_track_fl = RL(env_l).q_learning(gamma=.99, epsilon_decay_ratio=.75,
    #                                                                             n_episodes=30000,
    #                                                                             init_alpha=.9, min_alpha=.1,
    #                                                                             alpha_decay_ratio=.9, init_epsilon=1.0,
    #                                                                             min_epsilon=x[i])
    #     # max_val_per_iter = np.trim_zeros(np.max(Q_V_fl, axis=1), 'b')
    #     max_val_per_iter = np.max(Q_track_fl, axis=(1, 2))
    #     plt.plot(max_val_per_iter, label=f"Min Epsilon: {x[i]}")
    #     # df = pd.DataFrame(data=max_val_per_iter)
    #     # sns.set_theme(style="whitegrid")
    #     # sns.lineplot(data=df, legend=None)
    #     plt.gca().lines[i].set_color(colors[i])
    #     plt.legend()
    #
    # plt.xlabel("Iterations")
    # plt.ylabel("V(s)")
    # plt.title("Frozen Lake - Large: Q-Learning - Min Epsilon")
    # plt.savefig("images/FL_VI_Tuning_L_MinEpsilon.png")
    # plt.show()

    """ Alpha Decay Rate"""
    plt.subplot(2, 2, 3)
    x = [.1, .25, .5, .75, .9]
    n = [1000, 5000, 10000, 30000]
    colors = ['red', 'blue', 'green', 'orange', 'black']
    colors_n = ['red', 'blue', 'green', 'orange']
    plt.grid(True)
    for i in range(len(x)):
        Q_fl, Q_V_fl, Q_pi_fl, Q_track_fl, Q_pi_track_fl = RL(env_l).q_learning(gamma=.99, epsilon_decay_ratio=.75,
                                                                                n_episodes=30000,
                                                                                init_alpha=.9, min_alpha=.1,
                                                                                alpha_decay_ratio=x[i], init_epsilon=1.0,
                                                                                min_epsilon=.9)
        # max_val_per_iter = np.trim_zeros(np.max(Q_V_fl, axis=1), 'b')
        max_val_per_iter = np.max(Q_track_fl, axis=(1, 2))
        plt.plot(max_val_per_iter, label=f"Alpha Decay: {x[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None)
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()

    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Frozen Lake - Large: Q-Learning - Alpha Decay")
    # plt.savefig("images/FL_VI_Tuning_L_AlphaDecay.png")
    # plt.show()

    """ Init Epsilon Rate"""
    plt.subplot(2, 2, 4)
    x = [.1, .25, .5, .75, .9]
    n = [1000, 5000, 10000, 30000]
    colors = ['red', 'blue', 'green', 'orange', 'black']
    colors_n = ['red', 'blue', 'green', 'orange']
    plt.grid(True)
    for i in range(len(x)):
        Q_fl, Q_V_fl, Q_pi_fl, Q_track_fl, Q_pi_track_fl = RL(env_l).q_learning(gamma=.99, epsilon_decay_ratio=.75,
                                                                                n_episodes=30000,
                                                                                init_alpha=.9, min_alpha=.1,
                                                                                alpha_decay_ratio=.75,
                                                                                init_epsilon=x[i],
                                                                                min_epsilon=.9)
        # max_val_per_iter = np.trim_zeros(np.max(Q_V_fl, axis=1), 'b')
        max_val_per_iter = np.max(Q_track_fl, axis=(1, 2))
        plt.plot(max_val_per_iter, label=f"Initial Epsilon: {x[i]}")
        # df = pd.DataFrame(data=max_val_per_iter)
        # sns.set_theme(style="whitegrid")
        # sns.lineplot(data=df, legend=None)
        plt.gca().lines[i].set_color(colors[i])
        plt.legend()

    plt.xlabel("Iterations")
    plt.ylabel("V(s)")
    plt.title("Frozen Lake - Large: Q-Learning - Initial Epsilon")
    # plt.savefig("images/FL_VI_Tuning_L_InitialEpsilon.png")
    # plt.show()
    plt.suptitle("Exploration-Exploitation Trade-Off")
    plt.tight_layout()
    plt.savefig("images/FL_VI_Tuning_L_4_GreedyEpsilon_Exploration.png")
    # plt.show()

    # max_val_per_iter = np.trim_zeros(np.max(Q_fl, axis=1), 'b')
    # # max_val_per_iter = np.max(Q_track_fl, axis=(1, 2))
    # df = pd.DataFrame(data=max_val_per_iter)
    # sns.set_theme(style="whitegrid")
    # sns.lineplot(data=df, legend=None)
    # # plt.gca().lines[i].set_color(colors[i])
    # plt.title("Frozen Lake - Large: VI - Gamma Change")
    # # plt.savefig("images/FL_VI_Tuning_L.png")
    # plt.show()


if __name__ == "__main__":
    # test_project()
    test_frozen_lake()
    # test_blackjack()
    # taxi()
