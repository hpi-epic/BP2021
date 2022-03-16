import collections
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch

import alpha_business.rl.model as model
from alpha_business.configuration.hyperparameter_config import config
from alpha_business.market.circular.circular_vendors import CircularAgent
from alpha_business.market.linear.linear_vendors import LinearAgent
from alpha_business.rl.experience_buffer import ExperienceBuffer
from alpha_business.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class QLearningAgent(ReinforcementLearningAgent, ABC):
	Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

	# If you enter load_path, the model will be loaded. For example, if you want to use a pretrained net or test a given agent.
	# If you set an optim, this means you want training.
	# Give no optim if you don't want training.
	def __init__(
			self,
			n_observations,
			n_actions,
			optim=None,
			device='cuda' if torch.cuda.is_available() else 'cpu',
			load_path=None,
			name='q_learning'):
		self.device = device
		self.n_actions = n_actions
		self.buffer_for_feedback = None
		self.optimizer = None
		self.name = name
		print(f'I initiate a QLearningAgent using {self.device} device')
		self.net = model.simple_network(n_observations, n_actions).to(self.device)
		if load_path:
			self.net.load_state_dict(torch.load(load_path, map_location=self.device))
		if optim:
			self.optimizer = optim(self.net.parameters(), lr=config.learning_rate)
			self.tgt_net = model.simple_network(n_observations, n_actions).to(self.device)
			if load_path:
				self.tgt_net.load_state_dict(torch.load(load_path), map_location=self.device)
			self.buffer = ExperienceBuffer(config.replay_size)

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

	@abstractmethod
	def agent_output_to_market_form(self, action) -> tuple or int:  # pragma: no cover
		"""
		Takes a raw action and transforms it to a form that is accepted by the market.
		A raw action is for example three numbers in one.

		Args:
			action (np.array or int): the raw action

		Returns:
			tuple or int: the action accepted by the market.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

	def set_feedback(self, reward, is_done, new_observation):
		exp = self.Experience(*self.buffer_for_feedback, reward, is_done, new_observation)
		self.buffer.append(exp)
		self.buffer_for_feedback = None

	def train_batch(self):
		self.optimizer.zero_grad()
		batch = self.buffer.sample(config.batch_size)
		loss_t, selected_q_val_mean = self.calc_loss(batch, self.device)
		loss_t.backward()
		self.optimizer.step()
		return loss_t.item(), selected_q_val_mean.item()

	def synchronize_tgt_net(self):
		# Not printing this anymore since it clutters the output when training
		# print('Now I synchronize the tgt net')
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

		expected_state_action_values = next_state_values * config.gamma + rewards_v
		return torch.nn.MSELoss()(state_action_values, expected_state_action_values), state_action_values.mean()

	def save(self, model_path, model_name) -> None:
		"""
		Save a trained model to the specified folder within 'trainedModels'.

		Also caps the amount of models in the folder to a maximum of 10.

		Args:
			model_path (str): The path to the folder within 'trainedModels' where the model should be saved.
			model_name (str): The name of the .dat file of this specific model.
		"""
		model_name += '.dat'
		if not os.path.isdir(os.path.abspath(os.path.join('results', 'trainedModels'))):
			os.mkdir(os.path.abspath(os.path.join('results', 'trainedModels')))

		if not os.path.isdir(os.path.abspath(model_path)):
			os.mkdir(os.path.abspath(model_path))

		torch.save(self.net.state_dict(), os.path.join(model_path, model_name))

		full_directory = os.walk(model_path)
		for _, _, filenames in full_directory:
			if len(filenames) > 10:
				# split the filenames to isolate the reward-part
				split_filenames = [file.rsplit('_', 1) for file in filenames]
				# preserve the signature for later
				signature = split_filenames[0][0]
				# isolate the reward and convert it to float
				rewards = [file[1] for file in split_filenames]
				rewards = [float(reward.rsplit('.', 1)[0]) for reward in rewards]
				# sort the rewards to keep only the best ones
				rewards = sorted(rewards)

				for reward in range(len(rewards) - 10):
					os.remove(os.path.join(model_path, f'{signature}_{rewards[reward]:.3f}.dat'))


class QLearningLEAgent(QLearningAgent, LinearAgent):
	def agent_output_to_market_form(self, action):
		return action


class QLearningCEAgent(QLearningAgent, CircularAgent):
	def agent_output_to_market_form(self, action):
		return (int(action % config.max_price), int(action / config.max_price))


class QLearningCERebuyAgent(QLearningAgent, CircularAgent):
	def agent_output_to_market_form(self, action):
		return (
			int(action / (config.max_price * config.max_price)),
			int(action / config.max_price % config.max_price),
			int(action % config.max_price))