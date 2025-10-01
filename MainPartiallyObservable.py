from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MainSimpleEnvs import show, get_action_from_user, play_simple_env
from MemoryWrappers.BinaryMemory import BinaryMemory
from MemoryWrappers.KOrderMemory import KOrderMemory
from MemoryWrappers.SelectiveMemory import SelectiveMemory
from Agents.qlearning import QLearningAgent
from Agents.sarsa import NStepSarsaAgent
import matplotlib.pyplot as plt
import numpy as np


def run_experiments_and_plot(algos, env_factory, runs, episodes, title, filename):
    results = {name: [] for name in algos}

    for name, agent_fn in algos.items():
        print(f"\nEjecutando {name}...")
        for run in range(runs):
            env = env_factory()
            agent = agent_fn()
            _, lengths = agent.learn(env, return_lengths=True)
            results[name].append(lengths)
            if (run+1) % 10 == 0:
                # Calcular estadísticas de los últimos 100 episodios del run actual
                last_100 = results[name][-1][-100:] if len(results[name][-1]) >= 100 else results[name][-1]
                avg_last_100 = np.mean(last_100)
                min_last_100 = np.min(last_100)
                max_last_100 = np.max(last_100)
                std_last_100 = np.std(last_100)

                print(f"  Run {run+1}/{runs} - Últimos 100 eps: avg={avg_last_100:.2f}, min={min_last_100:.0f}, max={max_last_100:.0f}, std={std_last_100:.2f}")

        all_lengths = np.array(results[name])
        avg_per_ep = np.mean(all_lengths, axis=0)

        # Calcular porcentaje de episodios que probablemente terminaron por timeout
        # Asumimos que episodios > 1000 pasos son muy probablemente timeouts o cerca
        total_episodes = all_lengths.size
        long_episodes = np.sum(all_lengths > 1000)
        timeout_percentage = (long_episodes / total_episodes) * 100

        print(f"\n{'='*60}")
        print(f"RESUMEN {name}")
        print(f"{'='*60}")
        print(f"  Largo promedio total: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}")
        print(f"  Largo promedio últimos 100 eps (sobre todos los runs): {np.mean(avg_per_ep[-100:]):.2f}")
        print(f"  Largo promedio primer episodio: {avg_per_ep[0]:.2f}")
        print(f"  Largo promedio último episodio: {avg_per_ep[-1]:.2f}")
        print(f"  Mínimo alcanzado en cualquier episodio: {np.min(all_lengths):.0f}")
        print(f"  Máximo alcanzado en cualquier episodio: {np.max(all_lengths):.0f}")
        print(f"  Episodios > 1000 pasos: {long_episodes}/{total_episodes} ({timeout_percentage:.1f}%)")
        print(f"{'='*60}")

    plt.figure(figsize=(12, 6))

    for name in algos:
        arr = np.array(results[name])
        avg = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        plt.plot(avg, label=name, linewidth=2)
        plt.fill_between(range(len(avg)), avg - std, avg + std, alpha=0.2)
    plt.xlabel('Episodio')
    plt.ylabel('Largo promedio de episodio')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Gráfico guardado como '{filename}'")
    print(f"{'='*80}\n")

    print(f"\n{'='*90}")
    print("TABLA COMPARATIVA - LARGO PROMEDIO DE EPISODIO")
    print(f"{'='*90}")
    print(f"{'Método':<20} {'Primeros 100':<15} {'Últimos 100':<15} {'Mínimo':<12} {'% > 1000':<12}")
    print("-" * 90)
    for name in algos:
        arr = np.array(results[name])
        avg_per_ep = np.mean(arr, axis=0)
        first_100 = np.mean(avg_per_ep[:100])
        last_100 = np.mean(avg_per_ep[-100:])
        minimum = np.min(arr)
        timeout_pct = (np.sum(arr > 1000) / arr.size) * 100
        print(f"{name:<20} {first_100:<15.2f} {last_100:<15.2f} {minimum:<12.0f} {timeout_pct:<12.1f}")
    print(f"{'='*90}\n")


def play_env_with_binary_memory():
    num_of_bits = 1
    env = InvisibleDoorEnv()
    env = BinaryMemory(env, num_of_bits)

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    key2memory = {str(i): i for i in range(2**num_of_bits)}
    s = env.reset()
    show(env, s)
    done = False
    while not done:
        print("Environment action: ", end="")
        env_action = get_action_from_user(key2action)
        print(f"Memory action ({', '.join(key2memory.keys())}): ", end="")
        mem_action = get_action_from_user(key2memory)
        action = env_action, mem_action
        s, r, done = env.step(action)
        show(env, s, r)


def play_env_with_k_order_memory():
    memory_size = 2
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)
    play_simple_env(env)


def play_env_without_extra_memory():
    env = InvisibleDoorEnv()
    play_simple_env(env)


def play_env_with_selective_memory():
    buffer_size = 1
    env = InvisibleDoorEnv()
    env = SelectiveMemory(env, buffer_size)

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    key2memory = {'s': "save", 'i': "ignore"}
    s = env.reset()
    show(env, s)
    done = False
    while not done:
        print("Environment action: ", end="")
        env_action = get_action_from_user(key2action)
        print(f"Memory action (s=save, i=ignore): ", end="")
        mem_action = get_action_from_user(key2memory)
        action = env_action, mem_action
        s, r, done = env.step(action)
        show(env, s, r)


def run_pomdp_experiments():
    print("\n" + "="*80)
    print("EXPERIMENTO: Observación Parcial en InvisibleDoorEnv")
    print("="*80)

    actions = ["up", "down", "left", "right"]
    episodes = 1000
    runs = 30
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.01
    init_q = 1.0

    print(f"Configuración: {runs} runs, {episodes} episodios")
    print(f"Parámetros: γ={gamma}, α={alpha}, ε={epsilon}, Q_init={init_q}")
    print()

    algos = {
        'Q-learning': lambda: QLearningAgent(actions, alpha=alpha, gamma=gamma,
                                             epsilon=epsilon, episodes=episodes, init_q=init_q),
        'SARSA': lambda: NStepSarsaAgent(actions, n=1, alpha=alpha, gamma=gamma,
                                        epsilon=epsilon, episodes=episodes, init_q=init_q),
        '16-step SARSA': lambda: NStepSarsaAgent(actions, n=16, alpha=alpha, gamma=gamma,
                                                epsilon=epsilon, episodes=episodes, init_q=init_q)
    }

    run_experiments_and_plot(algos, InvisibleDoorEnv, runs, episodes,
                            "POMDP sin memoria", "pomdp_comparacion.png")


def run_pomdp_with_2order_memory_experiments():
    print("\n" + "="*80)
    print("EXPERIMENTO: Observación Parcial en InvisibleDoorEnv con 2-order memory")
    print("="*80)

    actions = ["up", "down", "left", "right"]
    episodes = 1000
    runs = 30
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.01
    init_q = 1.0
    memory_size = 2

    print(f"Configuración: {runs} runs, {episodes} episodios, 2-order memory")
    print(f"Parámetros: γ={gamma}, α={alpha}, ε={epsilon}, Q_init={init_q}")
    print()

    algos = {
        'Q-learning': lambda: QLearningAgent(actions, alpha=alpha, gamma=gamma,
                                             epsilon=epsilon, episodes=episodes, init_q=init_q),
        'SARSA': lambda: NStepSarsaAgent(actions, n=1, alpha=alpha, gamma=gamma,
                                        epsilon=epsilon, episodes=episodes, init_q=init_q),
        '16-step SARSA': lambda: NStepSarsaAgent(actions, n=16, alpha=alpha, gamma=gamma,
                                                epsilon=epsilon, episodes=episodes, init_q=init_q)
    }

    env_factory = lambda: KOrderMemory(InvisibleDoorEnv(), memory_size)
    run_experiments_and_plot(algos, env_factory, runs, episodes,
                            "POMDP con 2-order memory", "pomdp_2order_memory_comparacion.png")


def run_pomdp_with_binary_memory_experiments():
    print("\n" + "="*80)
    print("EXPERIMENTO: Observación Parcial en InvisibleDoorEnv con 1-bit binary memory")
    print("="*80)

    episodes = 1000
    runs = 30
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.01
    init_q = 1.0
    num_of_bits = 1

    print(f"Configuración: {runs} runs, {episodes} episodios, 1-bit binary memory")
    print(f"Parámetros: γ={gamma}, α={alpha}, ε={epsilon}, Q_init={init_q}")
    print()

    env_actions = ["up", "down", "left", "right"]
    memory_actions = list(range(2**num_of_bits))
    actions = [(env_action, memory_action) for env_action in env_actions for memory_action in memory_actions]

    algos = {
        'Q-learning': lambda: QLearningAgent(actions, alpha=alpha, gamma=gamma,
                                             epsilon=epsilon, episodes=episodes, init_q=init_q),
        'SARSA': lambda: NStepSarsaAgent(actions, n=1, alpha=alpha, gamma=gamma,
                                        epsilon=epsilon, episodes=episodes, init_q=init_q),
        '16-step SARSA': lambda: NStepSarsaAgent(actions, n=16, alpha=alpha, gamma=gamma,
                                                epsilon=epsilon, episodes=episodes, init_q=init_q)
    }

    env_factory = lambda: BinaryMemory(InvisibleDoorEnv(), num_of_bits)
    run_experiments_and_plot(algos, env_factory, runs, episodes,
                            "POMDP con 1-bit binary memory", "pomdp_binary_memory_comparacion.png")


def run_pomdp_with_selective_memory_experiments():
    print("\n" + "="*80)
    print("EXPERIMENTO: Observación Parcial en InvisibleDoorEnv con selective memory (buffer=1)")
    print("="*80)

    episodes = 1000
    runs = 30
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.01
    init_q = 1.0
    buffer_size = 1

    print(f"Configuración: {runs} runs, {episodes} episodios, selective memory (buffer=1)")
    print(f"Parámetros: γ={gamma}, α={alpha}, ε={epsilon}, Q_init={init_q}")
    print()

    env_actions = ["up", "down", "left", "right"]
    memory_actions = ["save", "ignore"]
    actions = [(env_action, memory_action) for env_action in env_actions for memory_action in memory_actions]

    algos = {
        'Q-learning': lambda: QLearningAgent(actions, alpha=alpha, gamma=gamma,
                                             epsilon=epsilon, episodes=episodes, init_q=init_q),
        'SARSA': lambda: NStepSarsaAgent(actions, n=1, alpha=alpha, gamma=gamma,
                                        epsilon=epsilon, episodes=episodes, init_q=init_q),
        '16-step SARSA': lambda: NStepSarsaAgent(actions, n=16, alpha=alpha, gamma=gamma,
                                                epsilon=epsilon, episodes=episodes, init_q=init_q)
    }

    env_factory = lambda: SelectiveMemory(InvisibleDoorEnv(), buffer_size)
    run_experiments_and_plot(algos, env_factory, runs, episodes,
                            "POMDP con selective memory", "pomdp_selective_memory_comparacion.png")


if __name__ == '__main__':
    print("Selecciona modo:")
    print("1. Jugar manualmente (sin memoria)")
    print("2. Jugar con K-order memory")
    print("3. Jugar con Binary memory")
    print("4. Jugar con Selective memory")
    print("5. Ejecutar experimento POMDP (Q-learning, SARSA, 16-step SARSA)")
    print("6. Ejecutar experimento POMDP con 2-order memory")
    print("7. Ejecutar experimento POMDP con 1-bit binary memory")
    print("8. Ejecutar experimento POMDP con selective memory (buffer=1)")
    modo = input("Opción (1/2/3/4/5/6/7/8): ")

    if modo == "1":
        play_env_without_extra_memory()
    elif modo == "2":
        play_env_with_k_order_memory()
    elif modo == "3":
        play_env_with_binary_memory()
    elif modo == "4":
        play_env_with_selective_memory()
    elif modo == "5":
        run_pomdp_experiments()
    elif modo == "6":
        run_pomdp_with_2order_memory_experiments()
    elif modo == "7":
        run_pomdp_with_binary_memory_experiments()
    elif modo == "8":
        run_pomdp_with_selective_memory_experiments()
    else:
        print("Opción no válida.")


