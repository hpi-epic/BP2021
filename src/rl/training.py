import math
import os
import signal
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut


class RLTrainer(ABC):
	def __init__(self, marketplace, RL_agent, log_dir_prepend=''):
		# TODO: assert Agent and marketplace fit together
		self.best_mean_reward = None
		self.marketplace = marketplace
		self.RL_agent = RL_agent
		assert self.trainer_agent_fit()
		# Signal handler for e.g. KeyboardInterrupt
		signal.signal(signal.SIGINT, self._signal_handler)

		self.initialize_io_related(log_dir_prepend)
		self.reset_time_tracker()

	def _signal_handler(self, signum, frame):  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		print('\nAborting training...')
		self._end_of_training()
		sys.exit(0)

	def initialize_io_related(self, log_dir_prepend):
		self.curr_time = time.strftime('%b%d_%H-%M-%S')
		self.signature = f'{type(self.marketplace).__name__}_{type(self.RL_agent).__name__}'
		self.writer = SummaryWriter(log_dir=os.path.join('results', 'runs', f'{log_dir_prepend}training_{self.signature}_{self.curr_time}'))

	def reset_time_tracker(self):
		self.frame_number_last_speed_update = 0
		self.time_last_speed_update = time.time()

	def calculate_dict_average(self, all_dicts) -> dict:
		sliced_dicts = all_dicts[-100:]
		averaged_info = sliced_dicts[0]
		for i, next_dict in enumerate(sliced_dicts):
			if i != 0:
				averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
		averaged_info = ut.divide_content_of_dict(averaged_info, len(sliced_dicts))
		return averaged_info

	def calculate_frames_per_second(self, frame_idx) -> float:
		speed = (frame_idx - self.frame_number_last_speed_update) / (
			(time.time() - self.time_last_speed_update) if (time.time() - self.time_last_speed_update) > 0 else 1
		)
		self.frame_number_last_speed_update = frame_idx
		self.time_last_speed_update = time.time()
		return speed

	def consider_print_info(self, frame_idx, episode_number, averaged_info, epsilon=None):
		if (episode_number) % 10 == 0:
			speed = self.calculate_frames_per_second(frame_idx)
			print(f"{frame_idx + 1}: {episode_number} episodes trained, mean return {averaged_info['profits/all']['vendor_0']:.3f}, " + (
				f'eps {epsilon:.2f}, ' if epsilon is not None else '') + f'speed {speed:.2f} f/s')

	def consider_update_best_model(self, averaged_info):
		mean_reward = averaged_info['profits/all']['vendor_0']
		if self.best_mean_reward is None:
			self.best_mean_reward = mean_reward - 1

		if mean_reward > self.best_mean_reward:
			self.RL_agent.save(path_name=f'{self.signature}_{self.curr_time}', model_name=f'{self.signature}_{mean_reward:.3f}')
			if self.best_mean_reward != 0:
				print(f'Best reward updated {self.best_mean_reward:.3f} -> {mean_reward:.3f}')
			self.best_mean_reward = mean_reward

	def consider_sync_tgt_net(self, frame_idx):
		if (frame_idx + 1) % config.SYNC_TARGET_FRAMES == 0:
			self.RL_agent.synchronize_tgt_net()

	@abstractmethod
	def train_agent(self, maxsteps=2 * config.EPSILON_DECAY_LAST_FRAME) -> None:
		raise NotImplementedError

	def _end_of_training(self):
		"""
		Inform the user of the best_mean_reward the agent achieved during training.
		"""
		if self.best_mean_reward == 0:
			print('The mean reward of the agent was never higher than 0, so no models were saved!')
		else:
			print(f'The best mean reward reached by the agent was {self.best_mean_reward:.3f}')
			print('The models were saved to:')
			print(os.path.abspath(os.path.join('trainedModels', f'{type(self.marketplace).__name__}_{type(self.RL_agent).__name__}')))


class QLearningTrainer(RLTrainer):
	def trainer_agent_fit(self):
		return isinstance(self.RL_agent, vendors.QLearningAgent), f'the passed agent must be a QLearningAgent: {self.RL_agent}'

	def train_agent(self, maxsteps=2 * config.EPSILON_DECAY_LAST_FRAME) -> None:
		"""
		Train a QLearningAgent on a marketplace.

		Args:
			maxsteps (int, optional): The maximum number of steps the training will run for. Defaults to 2*config.EPSILON_DECAY_LAST_FRAME.
		"""
		state = self.marketplace.reset()

		vendors_cumulated_info = None
		all_dicts = []
		losses = []
		rmse_losses = []
		selected_q_vals = []

		for frame_idx in range(maxsteps):
			epsilon = max(config.EPSILON_FINAL, config.EPSILON_START - frame_idx / config.EPSILON_DECAY_LAST_FRAME)

			action = self.RL_agent.policy(state, epsilon)
			state, reward, is_done, info = self.marketplace.step(action)
			self.RL_agent.set_feedback(reward, is_done, state)
			vendors_cumulated_info = info if vendors_cumulated_info is None else ut.add_content_of_two_dicts(vendors_cumulated_info, info)

			if is_done:
				all_dicts.append(vendors_cumulated_info)
				averaged_info = self.calculate_dict_average(all_dicts)

				if frame_idx > config.REPLAY_START_SIZE:
					averaged_info['Loss/MSE'] = np.mean(losses[-1000:])
					averaged_info['Loss/RMSE'] = np.mean(rmse_losses[-1000:])
					averaged_info['Loss/selected_q_vals'] = np.mean(selected_q_vals[-1000:])
					averaged_info['epsilon'] = epsilon
					ut.write_dict_to_tensorboard(self.writer, averaged_info, frame_idx / config.EPISODE_LENGTH)
					self.consider_print_info(frame_idx, len(all_dicts), averaged_info, epsilon)
					self.consider_update_best_model(averaged_info)

				vendors_cumulated_info = None
				self.marketplace.reset()

			self.consider_sync_tgt_net(frame_idx)

			if len(self.RL_agent.buffer) < config.REPLAY_START_SIZE:
				continue

			loss, selected_q_val_mean = self.RL_agent.train_batch()
			losses.append(loss)
			rmse_losses.append(math.sqrt(loss))
			selected_q_vals.append(selected_q_val_mean)

		self._end_of_training()
