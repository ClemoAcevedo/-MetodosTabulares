import random
from collections import defaultdict

class RMaxAgent:
    def __init__(self, actions, k=1, rmax=1.0, gamma=0.99, episodes=500, terminal_state='TERMINAL'):
        self.actions = actions
        self.k = k
        self.rmax = rmax
        self.gamma = gamma
        self.episodes = episodes
        self.terminal_state = terminal_state

        self.Nt = defaultdict(int)
        self.Model = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.V = defaultdict(float)

        self.known_states = set()
        self.known_actions_count = defaultdict(int)


    def get_all_transitions(self, state, action):
        if self.Nt[(state, action)] < self.k:
            return [(self.terminal_state, self.rmax, 1.0)]
        else:
            transitions = []
            total_count = self.Nt[(state, action)]
            for (next_state, reward), count in self.Model[state][action].items():
                prob = count / total_count
                transitions.append((next_state, reward, prob))
            return transitions

    def run_value_iteration(self, max_iterations=100, tolerance=1e-6):
        states = set(s for s, a in self.Nt.keys())
        states.add(self.terminal_state)

        for iteration in range(max_iterations):
            delta = 0
            for state in states:
                if state == self.terminal_state:
                    self.V[state] = 0.0
                    continue

                old_v = self.V[state]
                self._bao_backup(state, tolerance)
                delta = max(delta, abs(old_v - self.V[state]))

            if delta < tolerance:
                break

    def _bao_backup(self, state, tolerance):
        for _ in range(100):
            old_v_state = self.V[state]
            q_values = {a: self._get_q_value(state, a, self.V) for a in self.actions}
            self.V[state] = max(q_values.values())
            if abs(old_v_state - self.V[state]) < tolerance:
                break

    def choose_action(self, state):
        if state not in self.known_states:
            unknown_actions = [a for a in self.actions if self.Nt[(state, a)] < self.k]
            if unknown_actions:
                return random.choice(unknown_actions)

        best_action = None
        best_value = float('-inf')

        for action in self.actions:
            transitions = self.get_all_transitions(state, action)
            action_value = 0.0
            for next_state, reward, prob in transitions:
                action_value += prob * (reward + self.gamma * self.V[next_state])

            if action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action if best_action is not None else random.choice(self.actions)

    def _get_q_value(self, state, action, value_function):
        transitions = self.get_all_transitions(state, action)
        q_value = 0.0
        for next_state, reward, prob in transitions:
            q_value += prob * (reward + self.gamma * value_function[next_state])
        return q_value

    def learn(self, env, return_lengths=False, max_steps=2000):
        rewards_per_episode = []
        lengths_per_episode = []

        for ep in range(self.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < max_steps:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                is_newly_known = self.Nt[(state, action)] == self.k - 1

                self.Nt[(state, action)] += 1
                self.Model[state][action][(next_state, reward)] += 1

                if is_newly_known:
                    self.known_actions_count[state] += 1
                    if self.known_actions_count[state] == len(self.actions):
                        self.known_states.add(state)
                        self.run_value_iteration()

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
        self.run_value_iteration()

        policy = {}
        states = set(key[0] for key in self.Nt.keys())

        for state in states:
            if state == self.terminal_state:
                continue

            best_action = self.choose_action(state)
            policy[state] = best_action

        return policy