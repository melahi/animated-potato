import numpy as np

from common.schedules import LinearSchedule


class Player:
    def __init__(self, env_id, model_dir, exploration: LinearSchedule, param_noise):
        self.__env_id = env_id
        self.
        self.__continue_playing = None
        self.__policy = policy
        self.__exploration = exploration
        self.__param_noise = param_noise
        self.__experience_list = list()

    def terminating(self):
        self.__continue_playing = False

    def playing(self):
        t = 0
        while self.__continue_playing:
            # Take action and update exploration to the newest value
            reset = True
            observation = self.__env.reset()
            done = False
            while not done:
                kwargs = {}
                if not self.__param_noise:
                    update_eps = self.__exploration.value(t)
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(1. - self.__exploration.value(t) +
                                                           self.__exploration.value(t) / float(self.__env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                action = self.__policy(self.prepare_observation(observation), update_eps=update_eps, **kwargs)[0]
                reset = False
                new_observation, reward, done, _ = self.__env.step(action)
                # Store transition in the replay buffer.
                self.__experience_list.append((observation, action, reward, new_observation, float(done), False))
                observation = new_observation
        self.__env.close()

    @staticmethod
    def prepare_observation(observation):
        if type(observation) == dict:
            return {key: np.array(value, copy=False)[None] for key, value in observation.items()}
        return np.array(observation, copy=False)[None]

