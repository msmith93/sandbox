from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
from tf_agents.policies import policy_saver
import logging
from tensorflow import keras 
import numpy as np
import tensorflow as tf
import os

import random


# Check if TF is using the GPU                                                  
gpu_devices = tf.config.experimental.list_physical_devices('GPU')               
for device in gpu_devices:                                                      
  tf.config.experimental.set_memory_growth(device, True)                        
print("GPU count: " + str(len(gpu_devices)))

max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(environment_name, max_episode_steps=max_episode_steps,
                        gym_env_wrappers=[AtariPreprocessing, FrameStack4],
                        gym_kwargs={'render_mode': 'human'})
#env = suite_atari.load(environment_name, max_episode_steps=max_episode_steps,
#                       gym_env_wrappers=[AtariPreprocessing, FrameStack4])
tf_env = TFPyEnvironment(env)
time_step = tf_env.reset()

print("Environment has been reset")


policy_dir = os.path.join(os.getcwd(), 'policy11022023')
saved_policy = tf.compat.v2.saved_model.load(policy_dir)
policy_state = saved_policy.get_initial_state(batch_size=3)

while True:
  policy_step = saved_policy.action(time_step, policy_state)
  policy_state = policy_step.state
  time_step = tf_env.step(policy_step.action)


# while True:
#   action = random.randint(0, 3)

#   tf_env.step(action)
