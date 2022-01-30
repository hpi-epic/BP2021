import os
import signal
import sys
import time
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter

import configuration.config as config
import configuration.utils as ut
import rl.actorcritic_agent as actorcritic_agent
from agents.vendors import ReinforcementLearningAgent


class RLTrainer(ABC):
	def __init__(self, marketplace_class, agent_class, log_dir_prepend=''):
		# TODO: assert Agent and marketplace fit together
		assert issubclass(agent_class, ReinforcementLearningAgent)
		if issubclass(agent_class, actorcritic_agent.ContinuosActorCriticAgent):
			outputs = marketplace_class().get_actions_dimension()
		else:
			outputs = marketplace_class().get_n_actions()

		self.best_mean_reward = None
		self.marketplace_class = marketplace_class
		if issubclass(agent_class, actorcritic_agent.ActorCriticAgent):
			self.RL_agent = agent_class(marketplace_class().observation_space.shape[0], outputs)
		else:
			self.RL_agent = agent_class(marketplace_class().observation_space.shape[0], outputs, torch.optim.Adam)
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
		ut.ensure_results_folders_exist()
		self.curr_time = time.strftime('%b%d_%H-%M-%S')
		self.signature = f'{type(self.RL_agent).__name__}'
		self.writer = SummaryWriter(log_dir=os.path.join('results', 'runs', f'{log_dir_prepend}training_{self.curr_time}'))
		path_name = f'{self.signature}_{self.curr_time}'
		self.model_path = os.path.join('results', 'trainedModels', log_dir_prepend + path_name)

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
			(time.time() - self.time_last_speed_update) if (time.time() - self.time_last_speed_update) > 0 else 1)
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
			self.RL_agent.save(model_path=self.model_path, model_name=f'{self.signature}_{mean_reward:.3f}')
			if self.best_mean_reward != 0:
				print(f'Best reward updated {self.best_mean_reward:.3f} -> {mean_reward:.3f}')
			self.best_mean_reward = mean_reward

	def consider_sync_tgt_net(self, frame_idx):
		if (frame_idx + 1) % config.SYNC_TARGET_FRAMES == 0:
			self.RL_agent.synchronize_tgt_net()

	@abstractmethod
	def train_agent(self, maxsteps=2 * config.EPSILON_DECAY_LAST_FRAME) -> None:
		raise NotImplementedError('This method is abstract. Use a subclass')

	def _end_of_training(self):
		"""
		Inform the user of the best_mean_reward the agent achieved during training.
		"""
		if self.best_mean_reward == 0:
			print('The mean reward of the agent was never higher than 0, so no models were saved!')
		else:
			print(f'The best mean reward reached by the agent was {self.best_mean_reward:.3f}')
			print('The models were saved to:')
			print(os.path.abspath(self.model_path))
