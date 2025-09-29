import numpy as np
from Agents.qlearning import QLearningAgent


class CentralizedAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, init_q=0.0):
        self.agent = QLearningAgent(actions, alpha, gamma, epsilon, episodes, init_q)
        self.episodes = episodes

    def learn(self, env, run_num=None):
        lengths = []
        for ep in range(self.episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = env.step(action)

                old_q = self.agent.get_Q(state, action)
                next_qs = [self.agent.get_Q(next_state, a) for a in self.agent.actions]
                max_next_q = max(next_qs) if next_qs else 0.0
                new_q = old_q + self.agent.alpha * (reward + self.agent.gamma * max_next_q - old_q)
                self.agent.Q[(state, action)] = new_q

                state = next_state
                steps += 1
            lengths.append(steps)

        if run_num is not None:
            recent_avg_length = np.mean(lengths[-100:]) if len(lengths) >= 100 else np.mean(lengths)
            convergence_std = np.std(lengths[-100:]) if len(lengths) >= 100 else np.std(lengths)
            min_length = np.min(lengths[-100:]) if len(lengths) >= 100 else np.min(lengths)
            print(f"  Run {run_num} - Largo promedio últimos 100: {recent_avg_length:.2f}, Mín últimos 100: {min_length}, Convergencia (std): {convergence_std:.2f}")

        return lengths


class DecentralizedAgent:
    def __init__(self, num_agents, single_agent_actions, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, init_q=0.0):
        self.num_agents = num_agents
        self.agents = []
        self.episodes = episodes

        for _ in range(num_agents):
            agent = QLearningAgent(single_agent_actions, alpha, gamma, epsilon, episodes, init_q)
            self.agents.append(agent)

    def learn(self, env, run_num=None):
        lengths = []

        for ep in range(self.episodes):
            state = env.reset()
            done = False
            steps = 0

            while not done:
                actions = []
                for agent in self.agents:
                    action = agent.choose_action(state)
                    actions.append(action)

                joint_action = tuple(actions)
                next_state, rewards, done = env.step(joint_action)

                for i, agent in enumerate(self.agents):
                    old_q = agent.get_Q(state, actions[i])
                    next_qs = [agent.get_Q(next_state, a) for a in agent.actions]
                    max_next_q = max(next_qs) if next_qs else 0.0
                    new_q = old_q + agent.alpha * (rewards[i] + agent.gamma * max_next_q - old_q)
                    agent.Q[(state, actions[i])] = new_q

                state = next_state
                steps += 1

            lengths.append(steps)

        if run_num is not None:
            recent_avg_length = np.mean(lengths[-100:]) if len(lengths) >= 100 else np.mean(lengths)
            convergence_std = np.std(lengths[-100:]) if len(lengths) >= 100 else np.std(lengths)
            min_length = np.min(lengths[-100:]) if len(lengths) >= 100 else np.min(lengths)
            print(f"  Run {run_num} - Largo promedio últimos 100: {recent_avg_length:.2f}, Mín últimos 100: {min_length}, Convergencia (std): {convergence_std:.2f}")

        return lengths