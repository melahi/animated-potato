import os
import models
import bench
import argparse
import logger
from simple import learn, load
from common import set_global_seeds, wrap_atari_dqn
from common.atari_wrappers import make_atari, ActionDirectionEnv


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='gvgai-testgame1-lvl0-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    initial_direction = {'gvgai-testgame1': 5, 'gvgai-testgame2': 3}

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)
    game_name = args.env.split('-lvl')[0]
    if game_name in initial_direction:
        print("We should model with action direction")
        env = ActionDirectionEnv(env, initial_direction=initial_direction[game_name])
        model = models.cnn_to_mlp_with_action_direction(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=bool(args.dueling),
        )
    else:
        model = models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=bool(args.dueling),
        )
    checkpoint_path = "models/{}/".format(game_name)
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    os.makedirs(checkpoint_path, exist_ok=True)

    learn(env,
          q_func=model,
          lr=1e-4,
          max_timesteps=args.num_timesteps,
          buffer_size=1000,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=4,
          learning_starts=10000,
          target_network_update_freq=1000,
          gamma=0.99,
          prioritized_replay=bool(args.prioritized),
          prioritized_replay_alpha=args.prioritized_replay_alpha,
          checkpoint_freq=args.checkpoint_freq,
          checkpoint_path=checkpoint_path
          )

    env.close()


if __name__ == '__main__':
    main()
