import torch
import torch.optim as optim
import torch.nn as nn
import tqdm

from replay_buffer import *
from network import *

DEVICE = 'cuda'

class DQNTrainer:
    def __init__(self, policy_net, target_net, replay_buffer, optimizer, batch_size=1024, target_update=10):
        self.policy_net = policy_net

        self.target_net = target_net
        self.target_net.eval()

        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.target_update = target_update
        self.loss_fn = nn.MSELoss()
        self.training_stats = []

    def train_step(self, pbar):
        if len(self.replay_buffer) < self.batch_size:
            return

        experiences = self.replay_buffer.sample(self.batch_size)

        afterstate_batch = torch.cat([exp.afterstate.unsqueeze(0) for exp in experiences]).to(DEVICE)
        reward_batch = torch.tensor([exp.reward for exp in experiences]).to(DEVICE)
        next_state_batch = torch.cat([exp.next_state.unsqueeze(0) for exp in experiences]).to(DEVICE)
        #done_batch = torch.tensor([exp.done for exp in experiences])

        state_action_values = self.policy_net(afterstate_batch)

        with torch.no_grad():
            next_state_accum_batch = self.target_net.first_layer(next_state_batch)
            next_state_values = []
            for i, exp in enumerate(experiences):
                if exp.done == 0.0:
                    accum = next_state_accum_batch[i]
                    next_actions_values = self.target_net.first_layer(exp.next_actions.to(DEVICE))
                    out = self.target_net(next_actions_values + accum, skip_input=True)
                    next_state_values.append(out.max())
                else:
                    next_state_values.append(0.0)
            next_state_values = torch.tensor(next_state_values).to(DEVICE)

            expected_state_action_values = \
                    ((next_state_values * 0.99) + reward_batch).reshape(
                            (self.batch_size, 1))

        #print(state_action_values[0][0], expected_state_action_values[0][0])
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        pbar.set_description(f'Loss: {loss.item():10f}')
        self.training_stats.append({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, steps):
        pbar = tqdm.trange(steps)
        for step in pbar:
            self.train_step(pbar)
            if step % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

