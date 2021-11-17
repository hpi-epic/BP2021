import collections

import torch

import model
import utils as ut
from experience_buffer import ExperienceBuffer


class Agent2:
    def __init__(self):
        pass

    def policy(self, state):
        pass


class QLearningAgent(Agent2):
    Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

    def __init__(self, n_observation, n_actions):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('I initiate a QLearningAgent using {} device'.format(device))
        self.net = model.simple_network(n_observation, n_actions).to(device)
        self.tgt_net = model.simple_network(n_observation, n_actions).to(device)
        self.buffer = ExperienceBuffer(ut.REPLAY_SIZE)
        self.buffer_for_feedback = None

    def policy(self, state):
        assert self.buffer_for_feedback is None
        action = int(torch.argmax(self.net(torch.Tensor(state))))
        self.buffer_for_feedback = (state, action)
        return action

    def give_feedback(self, reward, is_done, new_state):
        exp = self.Experience(*self.buffer_for_feedback, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.buffer_for_feedback = None
