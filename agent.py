#!/usr/bin/env python3

# helper
import collections
import copy

import numpy as np
import torch

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state']
)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.total_rewards = []

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None
        output_profits = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.single([self.state])
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, output_dict = self.env.step(action)

        self.total_reward += reward
        for i, r in enumerate(output_dict['all_profits']):
            if len(self.total_rewards) <= i:
                self.total_rewards.append(0)
            self.total_rewards[i] += r

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            output_profits = copy.deepcopy(self.total_rewards)
            self._reset()
        return done_reward, output_profits
