import sys
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
  tf.config.experimental.set_memory_growth(device, True)

model = tf.keras.models.load_model('final_modelv4.keras')


env = gym.make("GymV21Environment-v0", env_id="CartPole-v1", render_mode="human")

def play_one_step(env, obs, model):
  with tf.GradientTape() as tape:
    left_proba = model(obs[np.newaxis])
    action = (tf.random.uniform([1, 1]) > left_proba)
    action = 1 - int(left_proba[0, 0].numpy() + 0.5)
  return env.step(action)


totals = []
for episode in range(5):
  episode_rewards = 0
  obs, info = env.reset()
  for step in range(500):
    obs, reward, done, truncated, info = play_one_step(env, obs, model)
    episode_rewards += reward
    if done or truncated:
      print("Done!")
      break
  print("reward: " + str(episode_rewards))
  totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
