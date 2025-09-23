import random
from collections import deque

class SarsaMultiGoalAgent:
    def __init__(self, actions, goals, n=1, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500):
        """
        actions: lista de acciones posibles
        goals: lista de objetivos posibles (g)
        n: pasos para n-step SARSA (n=1 es SARSA estándar)
        alpha: tasa de aprendizaje
        gamma: factor de descuento
        epsilon: tasa de exploración
        episodes: número de episodios de entrenamiento
        """
        self.actions = actions
        self.goals = goals
        self.n = n
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
            action = self.choose_action(state, goal)
            states = deque([state], maxlen=self.n+1)
            goals = deque([goal], maxlen=self.n+1)
            actions = deque([action], maxlen=self.n+1)
            rewards = deque([0], maxlen=self.n+1)
            T = float('inf')
            t = 0
            steps = 0
            while True:
                if t < T:
                    step_result = env.step(actions[-1])
                    if len(step_result) == 4:
                        full_next_state, reward, done, next_goal = step_result
                    else:
                        full_next_state, reward, done = step_result
                        next_goal = goal
                    next_agent_pos = full_next_state[0]  # Extrae solo la posición del agente
                    states.append(next_agent_pos)
                    goals.append(next_goal)
                    rewards.append(reward)
                    steps += 1
                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(next_agent_pos, next_goal)
                        actions.append(next_action)
                tau = t - self.n + 1
                if tau >= 0:
                    for g in self.goals:
                        G = 0.0
                        for i in range(1, min(self.n, T-tau)+1):
                            r = 1.0 if states[i] == g else 0.0
                            G += (self.gamma**(i-1)) * r
                        if tau + self.n < T:
                            G += (self.gamma**self.n) * self.get_Q(states[-1], g, actions[-1])
                        old_q = self.get_Q(states[0], g, actions[0])
                        self.Q[(states[0], g, actions[0])] = old_q + self.alpha * (G - old_q)
                if tau == T - 1:
                    break
                t += 1
            episode_lengths.append(steps)
        return episode_lengths

    def get_policy(self):
        policy = {}
        for (state, goal, action), q_value in self.Q.items():
            if (state, goal) not in policy or q_value > self.Q.get((state, goal, policy[(state, goal)]), float('-inf')):
                policy[(state, goal)] = action
        return policy
