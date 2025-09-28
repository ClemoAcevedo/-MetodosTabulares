from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from MainSimpleEnvs import show, get_action_from_user
from Agents.qlearning_multiagent import CentralizedAgent, DecentralizedAgent
import matplotlib.pyplot as plt
import numpy as np


def run_multiagent_experiment(env_class, agent_class, agent_kwargs, runs=30):
    lengths_per_run = []

    for run in range(runs):
        env = env_class()
        agent = agent_class(**agent_kwargs)
        lengths, rewards = agent.learn(env, run_num=run+1)
        lengths_per_run.append(lengths)

    lengths_per_run = np.array(lengths_per_run)
    avg_lengths = np.mean(lengths_per_run, axis=0)

    return avg_lengths, lengths_per_run


def play_hunter_env():
    hunter_env = HunterAndPreyEnv()

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down", '': "None"}
    num_of_agents = hunter_env.num_of_agents
    s = hunter_env.reset()
    show(hunter_env, s)
    done = False
    while not done:
        print("Hunter A: ", end="")
        hunter1 = get_action_from_user(key2action)
        print("Hunter B: ", end="")
        hunter2 = get_action_from_user(key2action)
        action = hunter1, hunter2
        if num_of_agents == 3:
            print("Prey: ", end="")
            prey = get_action_from_user(key2action)
            action = hunter1, hunter2, prey
        s, r, done = hunter_env.step(action)
        show(hunter_env, s, r)


def run_centralized_cooperative_experiment():
    print("\n" + "="*80)
    print("EXPERIMENTO 1: Centralized Cooperative Multi-Agent RL")
    print("="*80)

    episodes = 50000
    runs = 30
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    init_q = 1.0

    print(f"Configuración: {runs} runs, {episodes} episodios, γ={gamma}, α={alpha}, ε={epsilon}, Q_init={init_q}")

    env = CentralizedHunterEnv()
    agent_kwargs = {
        'actions': env.action_space,
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'episodes': episodes,
        'init_q': init_q
    }

    avg_lengths, lengths_per_run = run_multiagent_experiment(
        CentralizedHunterEnv, CentralizedAgent, agent_kwargs, runs
    )

    print(f"\nResultados Centralized Cooperative:")
    print(f"  Largo promedio total: {np.mean(lengths_per_run):.2f}")
    print(f"  Largo promedio últimos 100 episodios: {np.mean(avg_lengths[-100:]):.2f}")

    return avg_lengths


def run_decentralized_cooperative_experiment():
    print("\n" + "="*80)
    print("EXPERIMENTO 2: Decentralized Cooperative Multi-Agent RL")
    print("="*80)

    episodes = 50000
    runs = 30
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    init_q = 1.0

    print(f"Configuración: {runs} runs, {episodes} episodios, γ={gamma}, α={alpha}, ε={epsilon}, Q_init={init_q}")

    env = HunterEnv()
    agent_kwargs = {
        'num_agents': env.num_of_agents,
        'single_agent_actions': env.single_agent_action_space,
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'episodes': episodes,
        'init_q': init_q
    }

    avg_lengths, lengths_per_run = run_multiagent_experiment(
        HunterEnv, DecentralizedAgent, agent_kwargs, runs
    )

    print(f"\nResultados Decentralized Cooperative:")
    print(f"  Largo promedio total: {np.mean(lengths_per_run):.2f}")
    print(f"  Largo promedio últimos 100 episodios: {np.mean(avg_lengths[-100:]):.2f}")

    return avg_lengths


def run_decentralized_competitive_experiment():
    print("\n" + "="*80)
    print("EXPERIMENTO 3: Decentralized Competitive Multi-Agent RL")
    print("="*80)

    episodes = 50000
    runs = 30
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    init_q = 1.0

    print(f"Configuración: {runs} runs, {episodes} episodios, γ={gamma}, α={alpha}, ε={epsilon}, Q_init={init_q}")

    env = HunterAndPreyEnv()
    agent_kwargs = {
        'num_agents': env.num_of_agents,
        'single_agent_actions': env.single_agent_action_space,
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'episodes': episodes,
        'init_q': init_q
    }

    avg_lengths, lengths_per_run = run_multiagent_experiment(
        HunterAndPreyEnv, DecentralizedAgent, agent_kwargs, runs
    )

    print(f"\nResultados Decentralized Competitive:")
    print(f"  Largo promedio total: {np.mean(lengths_per_run):.2f}")
    print(f"  Largo promedio últimos 100 episodios: {np.mean(avg_lengths[-100:]):.2f}")

    return avg_lengths


def run_all_multiagent_experiments():
    print("EJECUTANDO TODOS LOS EXPERIMENTOS MULTI-AGENTE")
    print("="*80)

    centralized_lengths = run_centralized_cooperative_experiment()
    decentralized_coop_lengths = run_decentralized_cooperative_experiment()
    decentralized_comp_lengths = run_decentralized_competitive_experiment()

    episodes_range = np.arange(1, len(centralized_lengths) + 1)
    every_100 = episodes_range[::100] - 1

    plt.figure(figsize=(12, 8))
    plt.plot(every_100 + 1, centralized_lengths[every_100],
             label='Centralized Cooperative', linewidth=2, marker='o', markersize=3)
    plt.plot(every_100 + 1, decentralized_coop_lengths[every_100],
             label='Decentralized Cooperative', linewidth=2, marker='s', markersize=3)
    plt.plot(every_100 + 1, decentralized_comp_lengths[every_100],
             label='Decentralized Competitive', linewidth=2, marker='^', markersize=3)

    plt.xlabel('Episodio')
    plt.ylabel('Largo promedio de episodio')
    plt.title('Comparación Multi-Agent RL: Largo promedio de episodios (cada 100 episodios)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multiagent_comparacion.png', dpi=300, bbox_inches='tight')

    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"Centralized Cooperative - Largo promedio últimos 100 episodios: {np.mean(centralized_lengths[-100:]):.2f}")
    print(f"Decentralized Cooperative - Largo promedio últimos 100 episodios: {np.mean(decentralized_coop_lengths[-100:]):.2f}")
    print(f"Decentralized Competitive - Largo promedio últimos 100 episodios: {np.mean(decentralized_comp_lengths[-100:]):.2f}")
    print(f"\nGráfico guardado como 'multiagent_comparacion.png'")
    print("="*80)


if __name__ == '__main__':
    print("Selecciona modo:")
    print("1. Jugar manualmente HunterAndPreyEnv")
    print("2. Ejecutar experimento Centralized Cooperative")
    print("3. Ejecutar experimento Decentralized Cooperative")
    print("4. Ejecutar experimento Decentralized Competitive")
    print("5. Ejecutar todos los experimentos Multi-Agent RL")
    modo = input("Opción (1/2/3/4/5): ")

    if modo == "1":
        play_hunter_env()
    elif modo == "2":
        run_centralized_cooperative_experiment()
    elif modo == "3":
        run_decentralized_cooperative_experiment()
    elif modo == "4":
        run_decentralized_competitive_experiment()
    elif modo == "5":
        run_all_multiagent_experiments()
    else:
        print("Opción no válida.")
