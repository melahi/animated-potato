import os
import signal
import pickle
import tensorflow as tf
import numpy as np

from player import Message
import common.tf_util as U
import logger
from common.schedules import LinearSchedule
from common import create_gvgai_environment

from build_graph import build_train
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils import ObservationInput
from common.tf_util import load_state, save_state
from expert_decision_maker import ExpertDecisionMaker
from act_wrapper import ActWrapper

terminate_learning = False


def signal_handler(signal_id, frame):
    global terminate_learning
    terminate_learning = True
    print("We should terminate the learning process ...")


def send_message_to_all(connections, message):
    for connection in connections:
        connection.send(message)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    session = tf.Session()
    session.__enter__()
    return ActWrapper.load(path)


def learn(env_id,
          q_func,
          lr=5e-4,
          max_timesteps=10000,
          buffer_size=5000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          train_steps=10,
          learning_starts=500,
          batch_size=32,
          print_freq=10,
          checkpoint_freq=100,
          model_dir=None,
          gamma=1.0,
          target_network_update_freq=50,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          player_processes=None,
          player_connections=None):
    env, _, _ = create_gvgai_environment(env_id)

    # Create all the functions necessary to train the model
    # expert_decision_maker = ExpertDecisionMaker(env=env)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space = env.observation_space

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    session = tf.Session()
    session.__enter__()
    policy_path = os.path.join(model_dir, "Policy.pkl")
    model_path = os.path.join(model_dir, "model", "model")
    if os.path.isdir(os.path.join(model_dir, "model")):
        load_state(model_path)
    else:
        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.action_space.n,
        }
        act = ActWrapper(act, act_params)
        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()
        act.save(policy_path)
        save_state(model_path)
    env.close()
    # Create the replay buffer
    if prioritized_replay:
        replay_buffer_path = os.path.join(model_dir, "Prioritized_replay.pkl")
        if os.path.isfile(replay_buffer_path):
            with open(replay_buffer_path, 'rb') as input_file:
                replay_buffer = pickle.load(input_file)
        else:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer_path = os.path.join(model_dir, "Normal_replay.pkl")
        if os.path.isfile(replay_buffer_path):
            with open(replay_buffer_path, 'rb') as input_file:
                replay_buffer = pickle.load(input_file)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    episode_rewards = list()
    saved_mean_reward = None

    signal.signal(signal.SIGQUIT, signal_handler)
    global terminate_learning

    total_timesteps = 0
    for timestep in range(max_timesteps):
        if terminate_learning:
            break

        for connection in player_connections:
            experiences, reward = connection.recv()
            episode_rewards.append(reward)
            for experience in experiences:
                replay_buffer.add(*experience)
                total_timesteps += 1

        if total_timesteps < learning_starts:
            if timestep % 10 == 0:
                print("not strated yet")
            continue

        if timestep % train_freq == 0:
            for i in range(train_steps):
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(total_timesteps))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

        if timestep % target_network_update_freq == 0:
            # Update target network periodically.
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if print_freq is not None and timestep % print_freq == 0:
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(total_timesteps)))
            logger.dump_tabular()

        if timestep % checkpoint_freq == 0 and mean_100ep_reward > saved_mean_reward:
            act.save(policy_path)
            save_state(model_path)
            saved_mean_reward = mean_100ep_reward
            send_message_to_all(player_connections, Message.UPDATE)

    send_message_to_all(player_connections, Message.TERMINATE)
    if mean_100ep_reward > saved_mean_reward:
        act.save(policy_path)
    with open(replay_buffer_path, 'wb') as output_file:
        pickle.dump(replay_buffer, output_file, pickle.HIGHEST_PROTOCOL)
    for player_process in player_processes:
        player_process.join()
        # player_process.terminate()

    return act.load(policy_path)


