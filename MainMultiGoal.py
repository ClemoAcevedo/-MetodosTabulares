from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from MainSimpleEnvs import play_simple_env
import matplotlib.pyplot as plt
import numpy as np
from Agents.qlearning import QLearningAgent
from Agents.sarsa import NStepSarsaAgent
from Agents.qlearning_multigoal import QLearningMultiGoalAgent
from Agents.sarsa_multigoal import SarsaMultiGoalAgent

def play_room_env():
    n_episodes = 10
    for _ in range(n_episodes):
        env = RoomEnv()
        play_simple_env(env)


def run_roomenv_experiments():
    from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
    actions = ["up", "down", "left", "right"]
    tmp_env = RoomEnv()
    goals = tmp_env.get_goals() if hasattr(tmp_env, 'get_goals') else list(getattr(tmp_env, 'goals', []))
    episodes = 500
    runs = 100
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    algos = {
        'Q-learning': lambda: QLearningAgent(actions, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes, init_q=1.0),
        'SARSA': lambda: NStepSarsaAgent(actions, n=1, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes, init_q=1.0),
        '8-step SARSA': lambda: NStepSarsaAgent(actions, n=8, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes, init_q=1.0),
        'Q-learning MultiGoal': lambda: QLearningMultiGoalAgent(actions, goals, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes),
        'SARSA MultiGoal': lambda: SarsaMultiGoalAgent(actions, goals, n=1, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes),
    }

    results = {name: [] for name in algos}

    for name, agent_fn in algos.items():
        print(f"\nEjecutando {name}...")
        for run in range(runs):
            env = RoomEnv()
            agent = agent_fn()
            if name in ['Q-learning', 'SARSA', '8-step SARSA']:
                _, lengths = agent.learn(env, return_lengths=True)
                results[name].append(lengths)
            else:
                lengths = agent.learn(env)
                results[name].append(lengths)
            if (run+1) % 10 == 0:
                ultimos10 = np.mean(results[name][-1][-10:]) if len(results[name][-1]) >= 10 else np.mean(results[name][-1])
                print(f"  Run {run+1}/{runs} - Largo promedio últimos 10 episodios: {ultimos10:.2f}")
        all_lengths = np.array(results[name])
        avg_per_ep = np.mean(all_lengths, axis=0)
        print(f"\nResumen {name}:")
        print(f"  Largo promedio total: {np.mean(all_lengths):.2f}")
        print(f"  Largo promedio últimos 10 episodios (promediado sobre runs): {np.mean(avg_per_ep[-10:]):.2f}")
        print(f"  Largo promedio primer episodio: {avg_per_ep[0]:.2f}")
        print(f"  Largo promedio episodio 500: {avg_per_ep[-1]:.2f}")

    plt.figure(figsize=(10,6))
    for name in algos:
        arr = np.array(results[name])
        avg = np.mean(arr, axis=0)
        plt.plot(avg, label=name)
    plt.xlabel('Episodio')
    plt.ylabel('Largo promedio de episodio')
    plt.title('Comparación: Largo promedio de episodios en RoomEnv')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roomenv_comparacion.png')
    print("Gráfico guardado como 'roomenv_comparacion.png'")


if __name__ == '__main__':
    print("Selecciona modo:")
    print("1. Jugar manualmente RoomEnv")
    print("2. Comparar algoritmos en RoomEnv")
    modo = input("Opción (1/2): ")
    if modo == "1":
        play_room_env()
    elif modo == "2":
        run_roomenv_experiments()
    else:
        print("Opción no válida.")
