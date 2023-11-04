from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class FindMiddleSquareGameEnv(py_environment.PyEnvironment):

  def __init__(self):
    # Must be an even number
    self.square_size = 10
    self.middle_index = self.square_size / 2

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.int32, minimum=0, maximum=self.square_size, name='observation')
    self._state = np.array((random.randint(0, self.square_size), random.randint(0, self.square_size)), dtype=np.int32)
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = np.array((random.randint(0, self.square_size), random.randint(0, self.square_size)), dtype=np.int32)
    self._episode_ended = False
    return ts.restart(np.array(self._state, dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    # Right
    if action == 0:
      if self._state[0] < self.square_size:
        self._state[0] += 1
    # Up
    elif action == 1:
      if self._state[1] < self.square_size:
        self._state[1] += 1
    # Left
    elif action == 2:
      if self._state[0] > 0:
        self._state[0] -= 1
    # Down
    elif action == 3:
      if self._state[1] > 0:
        self._state[1] -= 1
    else:
      raise ValueError('`action` should be between 0 and 3.')
    
    if np.array_equal(self._state, np.array((self.middle_index, self.middle_index), dtype=np.int32)):
      self._episode_ended = True

    if self._episode_ended:
      reward = 10
      return ts.termination(np.array(self._state, dtype=np.int32), reward)
    else:
      # more negative reward the further we are from center
      # Negative reward should make DQN try to end the game faster?
      reward = -1 * (abs(self.middle_index - self._state[0]) + abs(self.middle_index - self._state[1]))
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=reward, discount=1.0)