import collections
import random

import numpy as np
import torch
from attrdict import AttrDict

import recommerce.rl.model as model
from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_rule
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.market.sim_market import SimMarket
from recommerce.rl.q_learning.experience_buffer import ExperienceBuffer
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class QLearningAgent(ReinforcementLearningAgent, CircularAgent, LinearAgent):
	Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

	# If you enter load_path, the model will be loaded. For example, if you want to use a pretrained net or test a given agent.
	# If you set an optim, this means you want training.
	# Give no optim if you don't want training.
	def __init__(
			self,
			config_market: AttrDict,
			config_rl: AttrDict,
			marketplace: SimMarket,
			device='cuda' if torch.cuda.is_available() else 'cpu',
			load_path=None,
			name='',
			network_architecture=model.simple_network):
		assert isinstance(marketplace, SimMarket), f'marketplace must be a SimMarket, but is {type(marketplace)}'

		n_observations = marketplace.get_observations_dimension()
		self.n_actions = marketplace.get_n_actions()
		self.actions_dimension = marketplace.get_actions_dimension()
		self.config_market = config_market
		self.config_rl = config_rl
		self.device = device
		self.buffer_for_feedback = None
		self.name = name if name != '' else type(self).__name__
		print(f'I initiate a {type(self).__name__} using {self.device} device')
		self.net = network_architecture(n_observations, self.n_actions).to(self.device)
		if load_path:
			self.optimizer = None
			self.net.load_state_dict(torch.load(load_path, map_location=self.device))

		# Here is assumed that training happens if and only if no load_path is given.
		if load_path is None:
			self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config_rl.learning_rate)
			self.tgt_net = network_architecture(n_observations, self.n_actions).to(self.device)
			self.buffer = ExperienceBuffer(self.config_rl.replay_size)

	@torch.no_grad()
	def policy(self, observation, epsilon=0):
		assert self.buffer_for_feedback is None or self.optimizer is None, 'one of buffer_for_feedback or optimizer must be None'
		if np.random.random() < epsilon:
			action = random.randint(0, self.n_actions - 1)
		else:
			action = int(torch.argmax(self.net(torch.Tensor(observation).to(self.device))))
		if self.optimizer is not None:
			self.buffer_for_feedback = (observation, action)
		return self.agent_output_to_market_form(action)

	def agent_output_to_market_form(self, action):
		"""
		Takes a raw action and transforms it to a form that is accepted by the market.
		A raw action is for example three numbers in one.

		Args:
			action (np.array or int): the raw action

		Returns:
			tuple or int: the action accepted by the market.
		"""
		if self.actions_dimension == 1:
			return action
		action_list = []
		for _ in range(self.actions_dimension):
			action_list.append(action % self.config_market.max_price)
			action = action // self.config_market.max_price
		action_list.reverse()
		return tuple(action_list)

	def set_feedback(self, reward, is_done, new_observation):
		exp = self.Experience(*self.buffer_for_feedback, reward, is_done, new_observation)
		self.buffer.append(exp)
		self.buffer_for_feedback = None

	def train_batch(self):
		self.optimizer.zero_grad()
		batch = self.buffer.sample(self.config_rl.batch_size)
		loss_t, selected_q_val_mean = self.calc_loss(batch, self.device)
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

		expected_state_action_values = next_state_values * self.config_rl.gamma + rewards_v
		return torch.nn.MSELoss()(state_action_values, expected_state_action_values), state_action_values.mean()

	def save(self, model_path: str) -> None:
		"""
		Save a trained model to the specified folder within 'trainedModels'.

		Args:
			model_path (str): The path including the name where the model should be saved.
		"""
		assert model_path.endswith('.dat'), f'the modelname must end in ".dat": {model_path}'
		torch.save(self.net.state_dict(), model_path)

	@staticmethod
	def get_configurable_fields() -> list:
		return [
			('gamma', float, between_zero_one_rule),
			('batch_size', int, greater_zero_rule),
			('replay_size', int, greater_zero_rule),
			('learning_rate', float, greater_zero_rule),
			('sync_target_frames', int, greater_zero_rule),
			('replay_start_size', int, greater_zero_rule),
			('epsilon_decay_last_frame', int, greater_zero_rule),
			('epsilon_start', float, between_zero_one_rule),
			('epsilon_final', float, between_zero_one_rule),
		]
