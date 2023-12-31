from tfenv import FindMiddleSquareGameEnv
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
import random
from tensorflow import keras 
import numpy as np
import tensorflow as tf
from tf_agents.utils.common import function
from tf_agents.policies import policy_saver
import os

class ShowProgress:
  def __init__(self, total):
    self.counter = 0
    self.total = total
  def __call__(self, trajectory):
    if not trajectory.is_boundary():
      self.counter += 1
    if self.counter % 100 == 0:
      print("\r{}/{}".format(self.counter, self.total), end="")

# Check if TF is using the GPU                                                  
gpu_devices = tf.config.experimental.list_physical_devices('GPU')               
for device in gpu_devices:                                                      
  tf.config.experimental.set_memory_growth(device, True)                        
print("GPU count: " + str(len(gpu_devices)))

max_episode_steps = 50

environment = FindMiddleSquareGameEnv()
timelimit_env = wrappers.TimeLimit(environment, duration=max_episode_steps)
tf_env = TFPyEnvironment(timelimit_env)

# utils.validate_py_environment(environment, episodes=1)

time_step = timelimit_env.reset()
print(time_step)

fc_layer_params = (32,)

q_net = QNetwork(
  tf_env.observation_spec(),
  tf_env.action_spec(),
  fc_layer_params=fc_layer_params
)
train_step = tf.Variable(0)
optimizer = keras.optimizers.Adam(lr=1e-3) 
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1.0, decay_steps=25000,
                                                        end_learning_rate=0.01)

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000,
                 td_errors_loss_fn=keras.losses.mean_squared_error,
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))

print("Agent initializing...")
agent.initialize()
print("Agent initialized")

policy_dir = os.path.join(os.getcwd(), 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

print("Creating replay buffer...")
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
  data_spec=agent.collect_data_spec,
  batch_size=tf_env.batch_size,
  max_length=200000
)
print("Replay buffer created...")

replay_buffer_observer = replay_buffer.add_batch

train_metrics = [
  tf_metrics.NumberOfEpisodes(),
  tf_metrics.EnvironmentSteps(),
  tf_metrics.AverageReturnMetric(),
  tf_metrics.AverageEpisodeLengthMetric()
]

collect_driver = DynamicStepDriver(
  tf_env,
  agent.collect_policy,
  observers=[replay_buffer_observer] + train_metrics
)

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

init_driver = DynamicStepDriver(
  tf_env,
  initial_collect_policy,
  observers=[replay_buffer.add_batch, ShowProgress(2000)],
  num_steps=2000
)

print("Running init driver...")
final_time_step, final_policy_state = init_driver.run()
print("Init Driver complete")

dataset = replay_buffer.as_dataset(
  sample_batch_size=64,
  num_steps=2,
  num_parallel_calls=3
).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

def train_agent(n_iterations):
  time_step = None
  policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
  iterator = iter(dataset)
  for iteration in range(n_iterations):
    time_step, policy_state = collect_driver.run(time_step, policy_state)
    trajectories, buffer_info = next(iterator)
    train_loss = agent.train(trajectories)
    if iteration % 100 == 0:
      print("\r{} loss:{:.5f}".format(
        iteration, train_loss.loss.numpy()), end=""
      )

train_agent(50000)

tf_policy_saver.save(policy_dir)

# def random_policy():
#     return random.randint(0,3)

# while not time_step.is_last():
#     action = random_policy()
#     print(f"Taking action {action}")
#     time_step = timelimit_env.step(action)
#     print(time_step)
