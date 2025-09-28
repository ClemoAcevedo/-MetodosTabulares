import numpy as np
from Agents.qlearning import QLearningAgent


class CentralizedAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, init_q=0.0):
        self.agent = QLearningAgent(actions, alpha, gamma, epsilon, episodes, init_q)
        self.episodes = episodes

    def learn(self, env, run_num=None):
        lengths = []
        total_rewards = []
        for ep in range(self.episodes):
            state = env.reset()
            done = False
            steps = 0
            episode_reward = 0
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
                episode_reward += reward
            lengths.append(steps)
            total_rewards.append(episode_reward)

        if run_num is not None:
            recent_avg_length = np.mean(lengths[-100:]) if len(lengths) >= 100 else np.mean(lengths)
            recent_avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            print(f"  Run {run_num} - Largo promedio últimos 100 episodios: {recent_avg_length:.2f}, Recompensa promedio: {recent_avg_reward:.2f}")

        return lengths, total_rewards


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
        agent_rewards = [[] for _ in range(self.num_agents)]

        for ep in range(self.episodes):
            state = env.reset()
            done = False
            steps = 0
            episode_rewards = [0] * self.num_agents

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
                    episode_rewards[i] += rewards[i]

                state = next_state
                steps += 1

            lengths.append(steps)
            for i in range(self.num_agents):
                agent_rewards[i].append(episode_rewards[i])

        if run_num is not None:
            recent_avg_length = np.mean(lengths[-100:]) if len(lengths) >= 100 else np.mean(lengths)
            print(f"  Run {run_num} - Largo promedio últimos 100 episodios: {recent_avg_length:.2f}")
            for i in range(self.num_agents):
                recent_avg_reward = np.mean(agent_rewards[i][-100:]) if len(agent_rewards[i]) >= 100 else np.mean(agent_rewards[i])
                print(f"    Agente {i+1} - Recompensa promedio últimos 100 episodios: {recent_avg_reward:.2f}")

        return lengths, agent_rewards