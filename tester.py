import gym
import gym_gvgai
import numpy as np
from time import sleep

import simple
from common import wrap_atari_dqn
from common.atari_wrappers import ActionDirectionEnv


def main():
    env_id = "gvgai-testgame1-lvl0-v0"
    env = gym.make(env_id)
    env = wrap_atari_dqn(env)
    env = ActionDirectionEnv(env, 5)
    act = simple.load("models/gvgai-testgame1/Policy.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            action = act(prepare_observation(obs))[0]
            obs, rew, done, info = env.step(action)
            episode_rew += rew
            sleep(0.05)
            print("action:", action, "instance reward:", rew, "info:", info)
        env.render()
        print("Episode reward", episode_rew)
        sleep(2)


def prepare_observation(observation):
    if type(observation) == dict:
        return {key: np.array(value, copy=False)[None] for key, value in observation.items()}
    return np.array(observation, copy=False)[None]


if __name__ == '__main__':
    main()
