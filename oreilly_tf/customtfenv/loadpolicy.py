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
from tfenv import FindMiddleSquareGameEnv
from tf_agents.environments import utils
from tf_agents.environments import wrappers

# Check if TF is using the GPU                                                  
gpu_devices = tf.config.experimental.list_physical_devices('GPU')               
for device in gpu_devices:                                                      
  tf.config.experimental.set_memory_growth(device, True)                        
print("GPU count: " + str(len(gpu_devices)))

max_episode_steps = 27000

environment = FindMiddleSquareGameEnv()
timelimit_env = wrappers.TimeLimit(environment, duration=max_episode_steps)
tf_env = TFPyEnvironment(timelimit_env)

print("Environment has been reset")


policy_dir = os.path.join(os.getcwd(), 'policy')
saved_policy = tf.compat.v2.saved_model.load(policy_dir)
policy_state = saved_policy.get_initial_state(batch_size=3)

num_episodes = 15

for i in range(num_episodes):
    episode_end = False
    reward = 0
    episode_steps = 0

    time_step = tf_env.reset()
    starting_point = time_step.observation
    while not episode_end:
        episode_steps += 1
        policy_step = saved_policy.action(time_step, policy_state)
        policy_state = policy_step.state

        time_step = tf_env.step(policy_step.action)
        reward += time_step.reward

        if time_step.step_type[0] == 2: # Episode end
            episode_end = True
        
    
    print(f"Episode {i}\nReward: {reward}\nNum steps: {episode_steps}\nStarting point; {starting_point}")
