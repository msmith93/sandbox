import sys
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

def basic_policy(obs):
  angle = obs[2]
  return 0 if angle < 0 else 1

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
  tf.config.experimental.set_memory_growth(device, True)

n_inputs = 4
model = keras.models.Sequential([
  keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
  keras.layers.Dense(1, activation="sigmoid"),
  ])

print(model)

env = gym.make("GymV21Environment-v0", env_id="CartPole-v1", render_mode="human")

def play_one_step(env, obs, model, loss_fn):
  with tf.GradientTape() as tape:
    left_proba = model(obs[np.newaxis])
    action = (tf.random.uniform([1, 1]) > left_proba)
    y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
    loss = tf.reduce_mean(loss_fn(y_target, left_proba))
  grads = tape.gradient(loss, model.trainable_variables)
  obs, reward, done, info = env.step(int(action[0, 0].numpy()))
  return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
  pass

totals = []
for episode in range(2):
  episode_rewards = 0
  obs, info = env.reset()
  for step in range(200):
    action = basic_policy(obs)
    env_ret = env.step(action)
    obs, reward, done, truncated, info = env_ret
    #env.render()
    episode_rewards += reward
    if done or truncated:
      print("Done!")
      break
  print("reward: " + str(episode_rewards))
  totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
#env.render()
#for i in range(0, 100):
#  env.render()
