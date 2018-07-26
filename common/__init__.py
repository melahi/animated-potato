# flake8: noqa F403
from common.console_util import *
from common.dataset import Dataset
from common.math_util import *
from common.misc_util import *


def wrap_atari_dqn(env):
    from common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
