# flake8: noqa F403
import bench
import logger
from common.console_util import *
from common.dataset import Dataset
from common.math_util import *
from common.misc_util import *


def create_gvgai_environment(env_id):
    from common.atari_wrappers import wrap_deepmind, make_atari, ActionDirectionEnv
    initial_direction = {'gvgai-testgame1': 3, 'gvgai-testgame2': 3}
    logger.configure()
    game_name = env_id.split('-lvl')[0]
    does_need_action_direction = False

    # Environment creation
    env = make_atari(env_id)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=True)
    if game_name in initial_direction:
        print("We should model with action direction")
        env = ActionDirectionEnv(env, initial_direction=initial_direction[game_name])
        does_need_action_direction = True
    return env, does_need_action_direction, game_name
