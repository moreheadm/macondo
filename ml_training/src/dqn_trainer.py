import torch
import torch.optim as optim
import torch.nn as nn
from replay_buffer import *
from network import *

class DQNTrainer:
    def __init__(self, policy_net, target_net, replay_buffer, optimizer, batch_size=32, target_update=10):
        self.policy_net = policy_net

        self.target_net = target_net
        self.target_net.eval()

        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.target_update = target_update
        self.loss_fn = nn.MSELoss()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        experiences = self.replay_buffer.sample(self.batch_size)

        afterstate_batch = torch.cat([exp.afterstate.unsqueeze(0) for exp in experiences])
        reward_batch = torch.cat([exp.reward for exp in experiences])
        next_state_batch = torch.cat([exp.next_state.unsqueeze(0) for exp in experiences])
        done_batch = torch.cat([exp.done for exp in experiences])

        state_action_values = self.policy_net(afterstate_batch)

        with torch.no_grad():
            next_state_accum_batch = self.target_net.first_layer(next_state_batch)
            next_state_values = []
            for i, exp in enumerate(experiences):
                accum = next_state_accum_batch[i]
                next_actions_values = self.target_net.first_layer(exp.next_actions)
                out = self.target_net(next_actions_values + accum, skip_input=True)
                next_state_values.append(out.max())
            next_state_values = torch.cat(next_state_values)

            expected_state_action_values = (next_state_values * 0.99) + reward_batch
        print(state_action_values.size(), expected_state_action_values.size())

        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, steps):
        for step in range(steps):
            self.train_step()
            if step % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

