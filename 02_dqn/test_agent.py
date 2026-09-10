import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer


# =========================
# Hyperparameters
# =========================

num_episodes = 300

buffer_capacity = 10000

batch_size = 64

learning_starts = 1000

target_update_frequency = 10


# =========================
# Environment
# =========================

env = gym.make(
    "CartPole-v1"
)


# =========================
# Agent
# =========================

agent = DQNAgent(
    state_dim=4,
    action_dim=2,

    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.99,

    gamma=0.99,
    learning_rate=1e-3
)


# =========================
# Replay Buffer
# =========================

buffer = ReplayBuffer(
    capacity=buffer_capacity
)


# =========================
# Store training results
# =========================

episode_rewards = []


# =========================
# Training
# =========================

for episode in range(num_episodes):

    state, info = env.reset()

    episode_reward = 0

    last_loss = None

    for step in range(500):

        # 1. Select action
        action = agent.select_action(state)

        # 2. Environment step
        next_state, reward, terminated, truncated, info = \
            env.step(action)

        done = terminated or truncated

        # 3. Store transition
        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        # 4. Accumulate reward
        episode_reward += reward

        # 5. Train network
        if len(buffer) >= learning_starts:

            last_loss = agent.update(
                buffer,
                batch_size=batch_size
            )

        # 6. Move to next state
        state = next_state

        # 7. Episode ends
        if done:
            break

    # Save reward
    episode_rewards.append(
        episode_reward
    )

    # =========================
    # Update Target Network
    # =========================

    if episode % target_update_frequency == 0:

        agent.update_target_network()

    # =========================
    # Epsilon Decay
    # =========================

    agent.decay_epsilon()

    # =========================
    # Print Training Info
    # =========================

    if last_loss is not None:

        print(
            f"Episode: {episode}, "
            f"Reward: {episode_reward:.1f}, "
            f"Epsilon: {agent.epsilon:.3f}, "
            f"Buffer: {len(buffer)}, "
            f"Loss: {last_loss:.4f}"
        )

    else:

        print(
            f"Episode: {episode}, "
            f"Reward: {episode_reward:.1f}, "
            f"Epsilon: {agent.epsilon:.3f}, "
            f"Buffer: {len(buffer)}, "
            f"Loss: Not training yet"
        )


env.close()


# ==================================================
# Visualization 1:
# Episode Reward Curve
# ==================================================

plt.figure()

plt.plot(
    episode_rewards,
    label="Episode Reward"
)

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.title("DQN Training Reward")

plt.legend()

plt.show()


# ==================================================
# Visualization 2:
# Moving Average
# ==================================================

window_size = 20

if len(episode_rewards) >= window_size:

    moving_average = np.convolve(
        episode_rewards,
        np.ones(window_size) / window_size,
        mode="valid"
    )

    plt.figure()

    plt.plot(
        episode_rewards,
        alpha=0.4,
        label="Episode Reward"
    )

    plt.plot(
        range(
            window_size - 1,
            len(episode_rewards)
        ),
        moving_average,
        label="20-Episode Moving Average"
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("DQN Training Performance")

    plt.legend()

    plt.show()


# ==================================================
# Visualization 3:
# Watch trained agent
# ==================================================

test_env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

# Test 时不再探索
old_epsilon = agent.epsilon

agent.epsilon = 0.0


state, info = test_env.reset()

test_reward = 0


for step in range(500):

    action = agent.select_action(state)

    next_state, reward, terminated, truncated, info = \
        test_env.step(action)

    test_reward += reward

    state = next_state

    if terminated or truncated:
        break


print(
    f"Test Reward: {test_reward}"
)


test_env.close()

agent.epsilon = old_epsilon