
import numpy as np
import random
from collections import deque


class NStepSarsaAgent:
	def __init__(self, actions, n=1, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, init_q=0.0):
		"""
		actions: lista de acciones posibles
		n: número de pasos (n=1 es SARSA estándar)
		alpha: tasa de aprendizaje
		gamma: factor de descuento
		epsilon: tasa de exploración
		episodes: número de episodios de entrenamiento
		init_q: valor inicial de Q
		"""
		self.actions = actions
		self.n = n
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
			action = self.choose_action(state)
			states = deque([state], maxlen=self.n+1)
			actions = deque([action], maxlen=self.n+1)
			rewards = deque([0], maxlen=self.n+1)
			T = float('inf')
			t = 0
			total_reward = 0
			steps = 0
			while True:
				if t < T:
					next_state, reward, done = env.step(actions[-1])
					states.append(next_state)
					rewards.append(reward)
					total_reward += reward
					steps += 1
					if done:
						T = t + 1
					else:
						next_action = self.choose_action(next_state)
						actions.append(next_action)
				tau = t - self.n + 1
				if tau >= 0:
					G = 0.0
					for i in range(1, min(self.n, T-tau)+1):
						G += (self.gamma**(i-1)) * rewards[i]
					if tau + self.n < T:
						G += (self.gamma**self.n) * self.get_Q(states[-1], actions[-1])
					old_q = self.get_Q(states[0], actions[0])
					self.Q[(states[0], actions[0])] = old_q + self.alpha * (G - old_q)
				if tau == T - 1:
					break
				t += 1
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

