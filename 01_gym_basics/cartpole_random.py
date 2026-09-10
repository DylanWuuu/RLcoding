import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

for step in range(200):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step: {step}, "
        f"Action: {action}, "
        f"Reward: {reward}"
    )

    time.sleep(0.05)

    if terminated or truncated:
        print("Episode finished!")
        break

input("Press Enter to close...")

env.close()