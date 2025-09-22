import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500):
        """
        actions: lista de acciones posibles
        alpha: tasa de aprendizaje
        gamma: factor de descuento
        epsilon: tasa de exploración
        episodes: número de episodios de entrenamiento
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.Q = dict()

    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qs = [self.get_Q(state, a) for a in self.actions]
        max_q = max(qs)
        max_actions = [a for a, q in zip(self.actions, qs) if q == max_q]
        return random.choice(max_actions)

    def learn(self, env):
        rewards_per_episode = []
        for ep in range(self.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                old_q = self.get_Q(state, action)
                next_qs = [self.get_Q(next_state, a) for a in self.actions]
                max_next_q = max(next_qs) if next_qs else 0.0
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.Q[(state, action)] = new_q
                state = next_state
                total_reward += reward
            rewards_per_episode.append(total_reward)
        return rewards_per_episode

    def get_policy(self):
        policy = {}
        states_q_values = {}
        for (state, action), q_value in self.Q.items():
            if state not in states_q_values:
                states_q_values[state] = []
            states_q_values[state].append((action, q_value))
        for state, action_q_pairs in states_q_values.items():
            best_action = max(action_q_pairs, key=lambda item: item[1])[0]
            policy[state] = best_action
        return policy
