from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import matplotlib.pyplot as plt
import numpy as np
from Agents.sarsa import NStepSarsaAgent
from Agents.qlearning import QLearningAgent
from Agents.rmax import RMaxAgent
from Agents.dynaq import DynaQAgent


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


def run_rmax_dynaq_experiment():
    env_class = EscapeRoomEnv
    actions = ["down", "up", "right", "left"]
    episodes = 20
    runs = 5
    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1

    planning_steps = [0, 1, 10, 100, 1000, 10000]

    print("\n" + "="*80)
    print("EXPERIMENTO: RMax vs Dyna-Q en EscapeRoomEnv")
    print("="*80)
    print(f"Configuración: {runs} runs, {episodes} episodios, γ={gamma}, α={alpha}, ε={epsilon}")
    print()

    results = {}

    print("Ejecutando RMax...")
    rmax_rewards = []
    for run in range(runs):
        env = env_class()
        agent = RMaxAgent(actions, gamma=gamma, episodes=episodes)
        rewards = agent.learn(env)
        rmax_rewards.append(rewards)
        print(f"  Run {run+1}/{runs} - Retorno promedio: {np.mean(rewards):.2f}")

    results['RMax'] = np.array(rmax_rewards)
    rmax_mean = np.mean(results['RMax'])
    print(f"\nRMax - Retorno promedio total: {rmax_mean:.2f}")

    for n_steps in planning_steps:
        print(f"\nEjecutando Dyna-Q con {n_steps} pasos de planning...")
        dynaq_rewards = []
        for run in range(runs):
            env = env_class()
            agent = DynaQAgent(actions, alpha=alpha, gamma=gamma, epsilon=epsilon,
                             n=n_steps, episodes=episodes, init_q=0.0)
            rewards = agent.learn(env)
            dynaq_rewards.append(rewards)
            print(f"  Run {run+1}/{runs} - Retorno promedio: {np.mean(rewards):.2f}")

        results[f'Dyna-Q (n={n_steps})'] = np.array(dynaq_rewards)
        dynaq_mean = np.mean(results[f'Dyna-Q (n={n_steps})'])
        print(f"Dyna-Q (n={n_steps}) - Retorno promedio total: {dynaq_mean:.2f}")

    print("\n" + "="*80)
    print("TABLA DE RESULTADOS - RETORNO MEDIO POR EPISODIO")
    print("="*80)
    print(f"{'Método':<20} {'Retorno Medio':<15}")
    print("-" * 35)
    print(f"{'RMax':<20} {np.mean(results['RMax']):<15.2f}")
    for n_steps in planning_steps:
        method_name = f"Dyna-Q (n={n_steps})"
        mean_return = np.mean(results[method_name])
        print(f"{method_name:<20} {mean_return:<15.2f}")

    plt.figure(figsize=(12,8))

    rmax_avg = np.mean(results['RMax'], axis=0)
    plt.plot(rmax_avg, label='RMax', linewidth=2, marker='o', markersize=4)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, n_steps in enumerate(planning_steps):
        dynaq_avg = np.mean(results[f'Dyna-Q (n={n_steps})'], axis=0)
        plt.plot(dynaq_avg, label=f'Dyna-Q (n={n_steps})',
                color=colors[i], linewidth=2, marker='s', markersize=3)

    plt.xlabel('Episodio')
    plt.ylabel('Retorno promedio')
    plt.title('RMax vs Dyna-Q en EscapeRoomEnv')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rmax_dynaq_comparacion.png', dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado como 'rmax_dynaq_comparacion.png'")
    print("="*80)


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
    print("3. Comparar RMax vs Dyna-Q en EscapeRoomEnv")
    modo = input("Opción (1/2/3): ")
    if modo == "1":
        env = CliffEnv()
        play_simple_env(env)
    elif modo == "2":
        run_experiments()
    elif modo == "3":
        run_rmax_dynaq_experiment()
    else:
        print("Opción no válida.")

