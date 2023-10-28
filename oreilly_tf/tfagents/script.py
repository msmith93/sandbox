from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import random

max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(environment_name, max_episode_steps=max_episode_steps,
                       gym_env_wrappers=[AtariPreprocessing, FrameStack4],
                       gym_kwargs={'render_mode': 'human'})
tf_env = TFPyEnvironment(env)

out = tf_env.reset()

out = tf_env.step(1)

for i in range(0, 10):
  tf_env.step(0)

while True:
  action = random.randint(0, 3)

  tf_env.step(action)
