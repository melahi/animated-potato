import gym

import simple


def main():
    env_id = "gvgai-testgame1-lvl0-v0"
    env = gym.make(env_id)
    act = simple.load("models/gvgai-testgame1/Policy.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
            print("instance reward:", rew)
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
