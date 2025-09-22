from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import matplotlib.pyplot as plt
import numpy as np
from Agents.sarsa import NStepSarsaAgent
from Agents.qlearning import QLearningAgent

import sys


def run_experiments():
    env_class = CliffEnv
    actions = ["up", "down", "left", "right"]
    episodes = 500
    runs = 100
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    algos = {
        'Q-learning': lambda: QLearningAgent(actions, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes),
        'SARSA': lambda: NStepSarsaAgent(actions, n=1, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes),
        '4-step SARSA': lambda: NStepSarsaAgent(actions, n=4, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    }

    results = {name: [] for name in algos}

    for name, agent_fn in algos.items():
        print(f"\nEjecutando {name}...")
        for run in range(runs):
            env = env_class()
            agent = agent_fn()
            rewards = agent.learn(env)
            results[name].append(rewards)
            if (run+1) % 10 == 0:
                print(f"  Run {run+1}/{runs} - Retorno promedio últimos 10 episodios: {np.mean(rewards[-10:]):.2f}")
        all_rewards = np.array(results[name])
        avg_per_ep = np.mean(all_rewards, axis=0)
        print(f"\nResumen {name}:")
        print(f"  Retorno promedio total: {np.mean(all_rewards):.2f}")
        print(f"  Retorno promedio últimos 10 episodios (promediado sobre runs): {np.mean(avg_per_ep[-10:]):.2f}")
        print(f"  Retorno promedio primer episodio: {avg_per_ep[0]:.2f}")
        print(f"  Retorno promedio episodio 500: {avg_per_ep[-1]:.2f}")

    plt.figure(figsize=(10,6))
    for name in algos:
        avg = np.mean(results[name], axis=0)
        plt.plot(avg, label=name)
    plt.xlabel('Episodio')
    plt.ylabel('Retorno promedio')
    plt.ylim(-200, None)
    plt.title('Comparación: Q-learning vs SARSA vs 4-step SARSA en CliffEnv')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cliffenv_comparacion.png')
    print("Gráfico guardado como 'cliffenv_comparacion.png'")


def show(env, current_state, reward=None):
    env.show()
    print(f"Raw state: {current_state}")
    if reward:
        print(f"Reward: {reward}")


def get_action_from_user(valid_actions):
    key = input()
    while key not in valid_actions:
        key = input()
    return valid_actions[key]


def play_simple_env(simple_env):
    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    s = simple_env.reset()
    show(simple_env, s)
    done = False
    while not done:
        print("Action: ", end="")
        action = get_action_from_user(key2action)
        s, r, done = simple_env.step(action)
        show(simple_env, s, r)


if __name__ == "__main__":
    print("Selecciona modo:")
    print("1. Jugar manualmente CliffEnv")
    print("2. Comparar Q-learning, SARSA y 4-step SARSA en CliffEnv")
    modo = input("Opción (1/2): ")
    if modo == "1":
        env = CliffEnv()
        play_simple_env(env)
    elif modo == "2":
        run_experiments()
    else:
        print("Opción no válida.")

