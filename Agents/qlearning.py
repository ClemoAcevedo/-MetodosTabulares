import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, init_q=0.0):
        """
        actions: lista de acciones posibles
        alpha: tasa de aprendizaje
        gamma: factor de descuento
        epsilon: tasa de exploración
        episodes: número de episodios de entrenamiento
        init_q: valor inicial de Q
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.Q = dict()
        self.init_q = init_q

    def get_Q(self, state, action):
        return self.Q.get((state, action), self.init_q)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qs = [self.get_Q(state, a) for a in self.actions]
        max_q = max(qs)
        max_actions = [a for a, q in zip(self.actions, qs) if q == max_q]
        return random.choice(max_actions)

    def learn(self, env, return_lengths=False):
        rewards_per_episode = []
        lengths_per_episode = []
        for ep in range(self.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
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
                steps += 1
            rewards_per_episode.append(total_reward)
            lengths_per_episode.append(steps)
        if return_lengths:
            return rewards_per_episode, lengths_per_episode
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
