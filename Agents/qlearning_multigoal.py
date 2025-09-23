import random

class QLearningMultiGoalAgent:
    def __init__(self, actions, goals, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500):
        """
        actions: lista de acciones posibles
        goals: lista de objetivos posibles (g)
        alpha: tasa de aprendizaje
        gamma: factor de descuento
        epsilon: tasa de exploración
        episodes: número de episodios de entrenamiento
        """
        self.actions = actions
        self.goals = goals
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.Q = dict()  

    def get_Q(self, state, goal, action):
        return self.Q.get((state, goal, action), 1.0)

    def choose_action(self, state, goal):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qs = [self.get_Q(state, goal, a) for a in self.actions]
        max_q = max(qs)
        max_actions = [a for a, q in zip(self.actions, qs) if q == max_q]
        return random.choice(max_actions)

    def learn(self, env):
        episode_lengths = []
        for ep in range(self.episodes):
            state, goal = env.reset()
            done = False
            steps = 0
            while not done:
                action = self.choose_action(state, goal)
                step_result = env.step(action)
                if len(step_result) == 4:
                    full_next_state, reward, done, next_goal = step_result
                else:
                    full_next_state, reward, done = step_result
                    next_goal = goal
                next_agent_pos = full_next_state[0]  # Extrae solo la posición del agente
                for g in self.goals:
                    if next_agent_pos == g:
                        target = 1.0
                    else:
                        next_qs = [self.get_Q(next_agent_pos, g, a) for a in self.actions]
                        target = self.gamma * max(next_qs)
                    old_q = self.get_Q(state, g, action)
                    self.Q[(state, g, action)] = old_q + self.alpha * (target - old_q)
                state = next_agent_pos
                goal = next_goal
                steps += 1
            episode_lengths.append(steps)
        return episode_lengths

    def get_policy(self):
        policy = {}
        for (state, goal, action), q_value in self.Q.items():
            if (state, goal) not in policy or q_value > self.Q.get((state, goal, policy[(state, goal)]), float('-inf')):
                policy[(state, goal)] = action
        return policy
