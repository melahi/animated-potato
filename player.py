import multiprocessing
import numpy as np
import tensorflow as tf

from common import create_gvgai_environment
from common.schedules import LinearSchedule
from simple import ActWrapper, Message


class Player:
    def __init__(self, env_id, policy_path, exploration: LinearSchedule, param_noise, connection):
        self.__env, _, _ = create_gvgai_environment(env_id)
        self.__continue_playing = None
        self.__graph = tf.Graph()
        self.__session = tf.Session(graph=self.__graph)
        self.__policy = None
        self.__policy_path = policy_path
        self.__load_policy()
        self.__exploration = exploration
        self.__param_noise = param_noise
        self.__connection = connection

    def terminating(self):
        self.__continue_playing = False

    def play(self):
        with self.__session.as_default():
            self.__play()
        self.__env.close()

    def __play(self):
        t = 0
        while self.__continue_playing:
            # Take action and update exploration to the newest value
            reset = True
            observation = self.__env.reset()
            done = False
            episode_experiences = list()
            episode_reward = 0
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
                episode_experiences.append((observation, action, reward, new_observation, float(done), False))
                episode_reward += reward
                observation = new_observation
            self.__connection.send((episode_experiences, episode_reward))
            self.__check_commander_messages()

    @staticmethod
    def prepare_observation(observation):
        if type(observation) == dict:
            return {key: np.array(value, copy=False)[None] for key, value in observation.items()}
        return np.array(observation, copy=False)[None]

    def __load_policy(self):
        with self.__graph.as_default():
            with tf.device("cpu"):
                with self.__session.as_default():
                    self.__policy = ActWrapper.load(self.__policy_path)

    def __check_commander_messages(self):
        while self.__connection.poll():
            message = self.__connection.recv()
            if message == Message.UPDATE:
                self.__load_policy()
            elif message == Message.TERMINATE:
                self.terminating()

    @staticmethod
    def player_process_factory(env_id, policy_path, exploration: LinearSchedule, param_noise):
        commander_connection, player_connection = multiprocessing.Pipe(True)

        def create_and_play():
            player = Player(env_id, policy_path, exploration, param_noise, player_connection)
            player.play()

        player_process = multiprocessing.Process(target=lambda: create_and_play())
        player_process.start()
        return player_process, commander_connection
