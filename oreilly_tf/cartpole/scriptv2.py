import numpy as np
import sys
import gymnasium as gym
import tensorflow as tf
import csv
from tensorflow import keras

# Check if TF is using the GPU
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
  tf.config.experimental.set_memory_growth(device, True)
print("GPU count: " + str(len(gpu_devices)))

# Set up the neural net (does it get set with random weights and biases at first?)
n_inputs = 4
model = keras.models.Sequential([
  keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
  keras.layers.Dense(1, activation="sigmoid"),
  ])

try:
  f_obj = open('output.csv', 'a')
  writer_obj = csv.writer(f_obj)

  env = gym.make("GymV21Environment-v0", env_id="CartPole-v1", render_mode="rgb_array")

  def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
      # Use our neural net to decide at what probability we should go left
      left_proba = model(obs[np.newaxis])
      # Use randomness to decide the action based on this probability
      action = (tf.random.uniform([1, 1]) > left_proba)
      # Pretend that our targetted probability of going left is
      #   100% if we are going to go left, and 0% if we are going to go right
      y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
      # Figure out how bad our neural network was at determining the target
      #   action. Remember, the "target action" was chosen through randomness.
      #   so it may not actually be the ideal action
      loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    # Determine the gradient (or path) that the weights and biases need
    #   to take in order to get closer to suggesting the "target action"
    # GradientTape records all actions in the context. Then tensorflow
    # uses magic to analyze that tape and figure out gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Perform the action (which we've decided is the target action
    obs, reward, done, truncated, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, truncated, grads

  def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
      current_rewards = []
      current_grads = []
      # Set up a new environment with the initial observation
      obs, info = env.reset()
      for step in range(n_max_steps):
        # play a step
        obs, reward, done, truncated, grads = play_one_step(env, obs, model, loss_fn)
        # Keep track of the rewards and gradients
        current_rewards.append(reward)
        current_grads.append(grads)
        if done or truncated:
          break
      all_rewards.append(current_rewards)
      all_grads.append(current_grads)
    return all_rewards, all_grads

  # > dicount_rewards([10, 0, -50], discount_factor=0.8)
  # array([-22, -40, -50])
  # -22 = 10 + 0 * 0.8 + (-50) * 0.8^2
  # -40 = 0 + -50 * 0.8
  def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
      discounted[step] += discounted[step + 1] * discount_factor
    return discounted

  # I don't really know what this is for, but it's basically doing normalization on the discounted rewards
  # https://earnandexcel.com/blog/how-to-normalize-data-excel-normalization-in-excel/
  # Seems to just get all the discounted rewards closer to being between -2 and 2? idk what the range is, but smaller
  def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    rewards_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / rewards_std for discounted_rewards in all_discounted_rewards]

  n_iterations = 150
  n_episodes_per_update = 10
  n_max_steps = 200
  discount_factor = 0.95

  optimizer = keras.optimizers.Adam(lr=0.01)
  loss_fn = keras.losses.binary_crossentropy

  for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)

    for episode_index, rewards in enumerate(all_rewards):
      new_row = [iteration, episode_index, len(rewards)]
      writer_obj.writerow(new_row)

    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):

      mean_grads = tf.reduce_mean(
        [final_reward * all_grads[episode_index][step][var_index]
        for episode_index, final_rewards in enumerate(all_final_rewards)
          for step, final_reward in enumerate(final_rewards)], axis=0)

      all_mean_grads.append(mean_grads)

    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
finally:
  print(model.trainable_variables)
  f_obj.close()
  #env.render()
  #for i in range(0, 100):
  #  env.render()
