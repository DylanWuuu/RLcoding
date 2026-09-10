import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        gamma=0.99,
        learning_rate=1e-3
    ):
        self.action_dim = action_dim

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma

        # Online Q Network
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim
        )

        # Target Q Network
        self.target_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim
        )

        # 初始时复制参数
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        # optimizer 只更新 online network
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):

        # epsilon-greedy
        if random.random() < self.epsilon:

            action = random.randrange(
                self.action_dim
            )

        else:

            state = torch.tensor(
                state,
                dtype=torch.float32
            )

            with torch.no_grad():
                q_values = self.q_network(state)

            action = torch.argmax(
                q_values
            ).item()

        return action

    def update(self, replay_buffer, batch_size):

        # 1. sample batch
        states, actions, rewards, next_states, dones = \
            replay_buffer.sample(batch_size)

        # 2. convert to tensor
        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # 3. Online network:
        # Q(s, a) for all actions
        q_values = self.q_network(states)

        # 4. Pick Q(s, actual_action)
        current_q = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # 5. Target Q
        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            )

            max_next_q = next_q_values.max(
                dim=1
            ).values

            target_q = (
                rewards
                + self.gamma
                * (1 - dones)
                * max_next_q
            )

        # 6. loss
        loss = self.loss_fn(
            current_q,
            target_q
        )

        # 7. gradient descent
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )