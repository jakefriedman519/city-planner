import os
import pygame
import tensorflow as tf
import numpy as np
from dql import *
from gui import setup, refresh, main
from city_planner_gym import CityPlanningEnv
import matplotlib.pyplot as plt

tf.keras.utils.disable_interactive_logging()

# Initialize environment
env = CityPlanningEnv()
gui = False

if gui:
    setup()

# Action space and state dimensions
num_actions = env.action_space.nvec.prod()  # Total possible actions
grid_shape = env.observation_space["grid"].shape
budget_shape = env.observation_space["budget"].shape

dql_agent = DQL(num_actions, grid_shape, budget_shape)

target_dql_agent = DQL(num_actions, grid_shape, budget_shape)
target_dql_agent.set_weights(dql_agent.get_weights())

learning_rate = 0.001
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Hyperparameters
discount_factor = 0.99
exploration_prob = 1.0
exploration_decay = 0.995
min_exploration_prob = 0.01
num_episodes = 200
max_steps_per_episode = 100
update_target_frequency = 5
episode_rewards = []
losses = []


# cast for proper
def castStates(state):
    state = np.expand_dims(state, axis=0)
    state = np.expand_dims(state, axis=-1)
    state = tf.cast(state, dtype=tf.float32)
    return state


os.system("cls" if os.name == "nt" else "clear")

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Prepare inputs
        grid_state = castStates(state["grid"])
        budget_state = castStates(state["budget"])

        # Choose action
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Random exploration
        else:
            q_values = dql_agent.predict(
                {"grid_input": grid_state, "budget_input": budget_state}
            )
            action = np.unravel_index(np.argmax(q_values), env.action_space.nvec)   

        # Take action
        next_state, reward, done, info = env.step(action)

        # Prepare next state inputs
        next_grid_state = castStates(next_state["grid"])
        next_budget_state = castStates(next_state["budget"])

        # Calculate target Q-values using the Bellman equation
        
        with tf.GradientTape() as tape:
            current_q_values = dql_agent(
                {"grid_input": grid_state, "budget_input": budget_state}
            )

            next_q_values = target_dql_agent(
                {"grid_input": next_grid_state, "budget_input": next_budget_state}
            )

            max_next_q = tf.reduce_max(next_q_values, axis=-1)

            target_q_values = tf.identity(current_q_values)

            action_index = np.ravel_multi_index(action, env.action_space.nvec)

            action_index_tensor = tf.constant([[0, action_index]], dtype=tf.int32)  # Batch index 0

            updated_q_value = tf.reshape(
                reward + discount_factor * max_next_q * (1 - done), shape=(1,)
            )

            target_q_values = tf.tensor_scatter_nd_update(
                target_q_values, action_index_tensor, updated_q_value
            )

            loss = loss_fn(current_q_values, target_q_values)

        # Update the model
        gradients = tape.gradient(loss, dql_agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dql_agent.trainable_variables))

        # Update state and reward
        state = next_state
        episode_reward += reward
        losses.append(loss)

        if gui:
            refresh(state, reward, done, info)
            pygame.display.update()
            pygame.event.pump()

        if done:
            print
            break

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

    # Logging progress
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Exploration prob = {exploration_prob}")
    episode_rewards.append(episode_reward)

    if episode % update_target_frequency == 0:
        target_dql_agent.set_weights(dql_agent.get_weights())

# Save the model
dql_agent.save("city_planning_dql_model_5.keras")

eval_agent = tf.keras.models.load_model(
    "city_planning_dql_model_5.keras", custom_objects={"DQL": DQL}
)

# Evaluation loop
num_eval_episodes = 1
eval_rewards = []

# Evaluate over multiple episodes
for _ in range(num_eval_episodes):
    state = env.reset()
    eval_reward = 0

    for step in range(max_steps_per_episode):
        # Use the trained model to predict the action
        # Ensure the state is in the correct shape (batch size of 1)
        grid_state = castStates(state["grid"])  # Adding batch dimension
        budget_state = castStates(state["budget"])  # Adding batch dimension

        # Get the Q-values from the trained model
        q_values = eval_agent({"grid_input": grid_state, "budget_input": budget_state})
        # Choose the action with the highest Q-value
        action = np.unravel_index(np.argmax(q_values), env.action_space.nvec)

        print(action)

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, info = env.step(action)

        eval_reward += reward
        state = next_state

        # End the episode if done
        if done:
            break

    eval_rewards.append(eval_reward)

# Calculate and print the average reward over all evaluation episodes
average_eval_reward = np.mean(eval_rewards)
print(eval_rewards)
print(f"Average Evaluation Reward: {average_eval_reward}")

# Plot rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Episode Rewards Throughout Training")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="best")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(losses, label="Reward")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Losses throughout Training")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="best")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(eval_rewards, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Episode Rewards Throughout Evaluation")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="best")
plt.show()

main()
