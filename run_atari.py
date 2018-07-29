import os
import argparse
from common import set_global_seeds, create_gvgai_environment
from common.schedules import LinearSchedule
from player import Player


def create_players(env_id, model_dir, exploration_fraction, max_timesteps, exploration_final_eps, param_noise, agents_count):
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
    policy_path = os.path.join(model_dir, "Policy.pkl")
    player_processes = list()
    player_connections = list()
    for i in range(agents_count):
        process, connection = Player.player_process_factory(env_id, policy_path, exploration, param_noise)
        player_processes.append(process)
        player_connections.append(connection)
    return player_processes, player_connections


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='gvgai-testgame1-lvl0-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--model_dir', type=str, default=None)

    args = parser.parse_args()
    set_global_seeds(args.seed)
    env, does_need_action_direction, game_name = create_gvgai_environment(args.env)

    model_dir = "models/{}/".format(game_name)
    os.makedirs(model_dir, exist_ok=True)
    player_processes, player_connections = create_players(args.env, model_dir, 0.1, args.num_timesteps, 0.01, False, 1)

    import models
    from simple import learn

    if does_need_action_direction:
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
    env.close()
    if args.model_dir is not None:
        model_dir = args.model_dir

    learn(args.env,
          q_func=model,
          lr=1e-4,
          max_timesteps=args.num_timesteps,
          buffer_size=1000,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=1,
          learning_starts=500,
          target_network_update_freq=100,
          gamma=0.99,
          prioritized_replay=bool(args.prioritized),
          prioritized_replay_alpha=args.prioritized_replay_alpha,
          checkpoint_freq=args.checkpoint_freq,
          model_dir=model_dir,
          player_processes=player_processes,
          player_connections=player_connections
          )


if __name__ == '__main__':
    main()
