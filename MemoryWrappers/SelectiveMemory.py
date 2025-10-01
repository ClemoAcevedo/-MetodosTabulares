from collections import deque
from Environments.AbstractEnv import AbstractEnv


class SelectiveMemory(AbstractEnv):

    def __init__(self, env, buffer_size):
        self.__env = env
        self.__buffer_size = buffer_size
        self.__buffer = None
        self.__current_observation = None

    @property
    def action_space(self):
        env_actions = self.__env.action_space
        memory_actions = ["save", "ignore"]
        return [(env_action, memory_action) for env_action in env_actions for memory_action in memory_actions]

    def reset(self):
        self.__current_observation = self.__env.reset()
        self.__buffer = deque(maxlen=self.__buffer_size)
        return self.__get_state()

    def __get_state(self):
        if len(self.__buffer) == 0:
            return (self.__current_observation,)
        return (self.__current_observation,) + tuple(self.__buffer)

    def step(self, action):
        env_action, memory_action = action

        if memory_action == "save":
            self.__buffer.append(self.__current_observation)

        self.__current_observation, r, done = self.__env.step(env_action)
        return self.__get_state(), r, done

    def show(self):
        self.__env.show()
        print(f"Buffer: {list(self.__buffer)}")
