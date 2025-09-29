from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv


class CompetitiveHunterAndPreyEnv(HunterAndPreyEnv):

    def step(self, action):
        hunter1_action, hunter2_action, prey_action = action

        self._HunterAndPreyEnv__prey = self.get_next_location(self._HunterAndPreyEnv__prey, prey_action)
        self._HunterAndPreyEnv__hunter1 = self.get_next_location(self._HunterAndPreyEnv__hunter1, hunter1_action)
        self._HunterAndPreyEnv__hunter2 = self.get_next_location(self._HunterAndPreyEnv__hunter2, hunter2_action)

        hunter1_reward = 0.0
        hunter2_reward = 0.0
        done = False

        if self._HunterAndPreyEnv__prey == self._HunterAndPreyEnv__hunter1:
            hunter1_reward = 1.0
            hunter2_reward = -1.0
            done = True
        elif self._HunterAndPreyEnv__prey == self._HunterAndPreyEnv__hunter2:
            hunter2_reward = 1.0
            hunter1_reward = -1.0
            done = True

        prey_reward = -1.0 if done else 0.0

        return self._HunterAndPreyEnv__get_state(), (hunter1_reward, hunter2_reward, prey_reward), done