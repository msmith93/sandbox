import numpy as np
import sys
import gymnasium as gym
import tensorflow as tf
import csv
from tensorflow import keras
from collections import deque

# Check if TF is using the GPU
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
  tf.config.experimental.set_memory_growth(device, True)
print("GPU count: " + str(len(gpu_devices)))

input_shape = [4]
n_outputs = 2
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error
env = gym.make("GymV21Environment-v0", env_id="CartPole-v1", render_mode="rgb_array")

model = keras.models.Sequential([
  keras.layers.Dense(32, activation="elu", input_shape=input_shape),
  keras.layers.Dense(32, activation="elu"),
  keras.layers.Dense(n_outputs)
])
target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

replay_buffer = deque(maxlen=2000)

def epsilon_greedy_policy(state, epsilon=0):
  if np.random.rand() < epsilon:
    return np.random.randint(2)
  else:
    Q_values = model.predict(state[np.newaxis])
    return np.argmax(Q_values[0])

def sample_experiences(batch_size):
  indices = np.random.randint(len(replay_buffer), size=batch_size)
  batch = [replay_buffer[index] for index in indices]
  states, actions, rewards, next_states, dones = [
    np.array([experience[field_index] for experience in batch])
    for field_index in range(5)]
  
  return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
  action = epsilon_greedy_policy(state, epsilon)
  next_state, reward, done, truncated, info = env.step(action)

  replay_buffer.append((state, action, reward, next_state, done))
  return next_state, reward, done, info

def training_step(batch_size):
  experiences = sample_experiences(batch_size)
  states, actions, rewards, next_states, dones = experiences
  next_Q_values = target.predict(next_states)
  max_next_Q_values = np.max(next_Q_values, axis=1)
  target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)

  target_Q_values = target_Q_values.reshape(-1, 1)
  mask = tf.one_hot(actions, n_outputs)
  with tf.GradientTape() as tape:
    all_Q_values = model(states)
    Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
    loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

for episode in range(600):
  obs, info = env.reset()
  for step in range(200):
    epsilon = max(1 - episode / 500, 0.01)

    obs, reward, done, info = play_one_step(env, obs, epsilon)

    if done:
      break
  if episode > 50:
    training_step(batch_size)
  if episode % 50:
    target.set_weights(model.get_weights())

model.save("final_model.keras")