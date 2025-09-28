import random
from collections import defaultdict

class DynaQAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, n=5, episodes=500, init_q=0.0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.episodes = episodes
        self.init_q = init_q

        self.Q = defaultdict(lambda: self.init_q)
        self.Model = {}
        self.observed_states = set()
        self.state_actions = set()

    def get_Q(self, state, action):
        return self.Q[(state, action)]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qs = [self.get_Q(state, a) for a in self.actions]
        max_q = max(qs)
        max_actions = [a for a, q in zip(self.actions, qs) if q == max_q]
        return random.choice(max_actions)

    def update_q(self, state, action, reward, next_state):
        old_q = self.get_Q(state, action)
        next_qs = [self.get_Q(next_state, a) for a in self.actions]
        max_next_q = max(next_qs) if next_qs else 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.Q[(state, action)] = new_q

    def update_model(self, state, action, reward, next_state):
        self.Model[(state, action)] = (reward, next_state)
        self.observed_states.add(state)
        self.state_actions.add((state, action))

    def planning_step(self):
        for _ in range(self.n):
            if not self.state_actions:
                break

            state, action = random.choice(list(self.state_actions))
            reward, next_state = self.Model[(state, action)]
            self.update_q(state, action, reward, next_state)

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

                self.update_q(state, action, reward, next_state)
                self.update_model(state, action, reward, next_state)
                self.planning_step()

                state = next_state
                total_reward += reward
                steps += 1

            rewards_per_episode.append(total_reward)
            lengths_per_episode.append(steps)
            print(f"    Ep {ep+1}: reward={total_reward}, steps={steps}")

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