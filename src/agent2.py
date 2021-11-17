import collections
import copy
import os
import random

import numpy as np
import torch

import model
import utils as ut
from experience_buffer import ExperienceBuffer


class Agent2:
    def __init__(self):
        pass

    def policy(self, state, epsilon=0):
        pass


class QLearningAgent(Agent2):
    Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

    def __init__(self, n_observation, n_actions, optim=None, device='cpu', load_path=None):
        self.device = device
        self.n_actions = n_actions
        print('I initiate a QLearningAgent using {} device'.format(self.device))
        self.net = model.simple_network(n_observation, n_actions).to(self.device)
        if load_path:
            self.net.load_state_dict(torch.load(load_path), map_location=self.device)
        if optim:
            self.optimizer = optim(self.net.parameters(), lr=ut.LEARNING_RATE)
            self.tgt_net = model.simple_network(n_observation, n_actions).to(self.device)
            if load_path:
                self.tgt_net.load_state_dict(torch.load(load_path), map_location=self.device)
            self.buffer = ExperienceBuffer(ut.REPLAY_SIZE)
            self.buffer_for_feedback = None

    @torch.no_grad()
    def policy(self, state, epsilon=0):
        assert self.buffer_for_feedback is None
        if np.random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = int(torch.argmax(self.net(torch.Tensor(state).to(self.device))))
        self.buffer_for_feedback = (copy.deepcopy(state), action)
        return action

    def give_feedback(self, reward, is_done, new_state):
        exp = self.Experience(*self.buffer_for_feedback, reward, is_done, copy.deepcopy(new_state))
        # print(exp)
        self.buffer.append(exp)
        self.buffer_for_feedback = None

    def train_batch(self):
        self.optimizer.zero_grad()
        batch = self.buffer.sample(ut.BATCH_SIZE)
        loss_t, selected_q_val_mean = self.calc_loss(batch)
        loss_t.backward()
        self.optimizer.step()
        return loss_t.item(), selected_q_val_mean.item()

    def synchronize_tgt_net(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def calc_loss(self, batch, device='cpu'):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(np.single(states)).to(device)
        next_states_v = torch.tensor(np.single(next_states)).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.tgt_net(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * ut.GAMMA + rewards_v
        return torch.nn.MSELoss()(state_action_values, expected_state_action_values), state_action_values.mean()

    def save(self, path='QLearning_parameters'):
        if not os.path.isdir('trainedModels'):
            os.mkdir('trainedModels')
        torch.save(self.net.state_dict(), './trainedModels/' + path + '-best_%.2f_marketplace''.dat')
