import os
import signal
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import recommerce.configuration.utils as ut
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import config
from recommerce.configuration.path_manager import PathManager
from recommerce.monitoring.agent_monitoring.am_evaluation import Evaluator
from recommerce.monitoring.agent_monitoring.am_monitoring import Monitor
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class RLTrainer(ABC):
	def __init__(self, marketplace_class, agent_class, log_dir_prepend=''):
		"""
		Initialize an RLTrainer to train one specific configuration.
		Args:
			marketplace_class (subclass of SimMarket): The market scenario you want to train.
			agent_class (subclass of RLAgent): The agent you want to train.
			log_dir_prepend (str, optional): A prefix that is written before the saved data. Defaults to ''.
		"""
		# TODO: assert Agent and marketplace fit together
		assert issubclass(agent_class, ReinforcementLearningAgent)
		if issubclass(agent_class, actorcritic_agent.ContinuosActorCriticAgent):
			outputs = marketplace_class().get_actions_dimension()
		else:
			outputs = marketplace_class().get_n_actions()

		self.best_mean_interim_reward = None
		self.best_mean_overall_reward = None
		self.marketplace_class = marketplace_class
		self.agent_class = agent_class
		self.saved_parameter_paths = []
		if issubclass(agent_class, actorcritic_agent.ActorCriticAgent):
			self.RL_agent = agent_class(marketplace_class().observation_space.shape[0], outputs)
		else:
			self.RL_agent = agent_class(marketplace_class().observation_space.shape[0], outputs, torch.optim.Adam)
		assert self.trainer_agent_fit()

		# Signal handler for e.g. KeyboardInterrupt
		signal.signal(signal.SIGINT, self._signal_handler)

		self.initialize_io_related(log_dir_prepend)
		self.reset_time_tracker()

	def _signal_handler(self, signum, frame) -> None:  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		print('\nAborting training...')
		self._end_of_training()
		sys.exit(0)

	def initialize_io_related(self, log_dir_prepend) -> None:
		"""
		Initializes the local variables self.curr_time, self.signature, self.writer, self.model_path
		which are needed for saving the models and writing to tensorboard
		Args:
			log_dir_prepend (str): A prefix that is written before the saved data
		"""
		ut.ensure_results_folders_exist()
		self.curr_time = time.strftime('%b%d_%H-%M-%S')
		self.signature = f'{type(self.RL_agent).__name__}'
		self.writer = SummaryWriter(log_dir=os.path.join(PathManager.results_path, 'runs', f'{log_dir_prepend}training_{self.curr_time}'))
		path_name = f'{self.signature}_{self.curr_time}'
		self.model_path = os.path.join(PathManager.results_path, 'trainedModels', log_dir_prepend + path_name)
		os.makedirs(os.path.abspath(self.model_path), exist_ok=True)

	def reset_time_tracker(self) -> None:
		self.frame_number_last_speed_update = 0
		self.time_last_speed_update = time.time()

	def calculate_dict_average(self, all_dicts) -> dict:
		"""
		Takes a list of dictionaries and calculates the average for each entry over all dicts.
		Assumes that all dicts have the same shape.
		Args:
			all_dicts (list of dicts): The dictionaries which entries you want to average
		Returns:
			dict: A dict of the same shape containing the average in each entry.
		"""
		sliced_dicts = all_dicts[-100:]
		averaged_info = sliced_dicts[0]
		for i, next_dict in enumerate(sliced_dicts):
			if i != 0:
				averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
		averaged_info = ut.divide_content_of_dict(averaged_info, len(sliced_dicts))
		return averaged_info

	def consider_print_info(self, frame_idx, episode_number, averaged_info, epsilon=None) -> None:
		if (episode_number) % 10 == 0:
			tqdm.write(f"{frame_idx + 1}: {episode_number} episodes trained, mean return {averaged_info['profits/all']['vendor_0']:.3f}, " + (
				f'eps {epsilon:.2f}' if epsilon is not None else ''))

	def consider_save_model(self, episodes_idx: int, force=False) -> None:
		if ((episodes_idx % 500 == 0 and episodes_idx > 0) or force) and self.best_mean_interim_reward is not None:
			path_to_parameters = self.RL_agent.save(model_path=self.model_path, model_name=f'{self.signature}_{episodes_idx:05d}')
			tqdm.write(f'I write the interim model after {episodes_idx} episodes to the disk.')
			tqdm.write(f'You can find the parameters here: {path_to_parameters}.')
			tqdm.write(f'This model achieved a mean reward of {self.best_mean_interim_reward}.')
			self.saved_parameter_paths.append(path_to_parameters)
			self.best_mean_interim_reward = None

	def consider_update_best_model(self, averaged_info: dict) -> None:
		"""
		Evaluates if the current model is the best one until now.
		If it is, it updates the high score and syncs the model to a different variable until it is written to the disk.
		Args:
			averaged_info (dict): A dictionary containing averaged_info['profits/all']['vendor_0'] to evaluate the performance.
			frame_idx (int): The current frame.
		"""
		mean_reward = averaged_info['profits/all']['vendor_0']
		if self.best_mean_interim_reward is None or mean_reward > self.best_mean_interim_reward:
			self.RL_agent.sync_to_best_interim()
			self.best_mean_interim_reward = mean_reward
			if self.best_mean_overall_reward is None or self.best_mean_interim_reward > self.best_mean_overall_reward:
				if self.best_mean_overall_reward is not None:
					tqdm.write(f'Best overall reward updated {self.best_mean_overall_reward:.3f} -> {self.best_mean_interim_reward:.3f}')
				self.best_mean_overall_reward = self.best_mean_interim_reward

	def consider_sync_tgt_net(self, frame_idx) -> None:
		if (frame_idx + 1) % config.sync_target_frames == 0:
			self.RL_agent.synchronize_tgt_net()

	@abstractmethod
	def train_agent(self, maxsteps=2 * config.epsilon_decay_last_frame) -> None:
		raise NotImplementedError('This method is abstract. Use a subclass')

	def analyze_trained_agents(self):
		all_rewards = []
		for parameter_path in self.saved_parameter_paths:
			print(f'The analysis of model {parameter_path}:')
			monitor = Monitor()
			monitor.configurator.setup_monitoring(False, 250, 250, self.marketplace_class, [(self.agent_class, [parameter_path])])
			rewards = monitor.run_marketplace()
			all_rewards.append(rewards[0])
			print(f"It's mean was {np.mean(rewards[0])}")

		episode_numbers = [int(parameter_path[-9:][:5]) for parameter_path in self.saved_parameter_paths]
		monitor = Monitor()
		monitor.configurator.setup_monitoring(False, 250, 250, self.marketplace_class, [])
		Evaluator(monitor).create_violin_plot(all_rewards, episode_numbers)
		# To make this more beautiful, it could be all moved to agent monitoring.

	def _end_of_training(self) -> None:
		"""
		Inform the user of the best_mean_overall_reward the agent achieved during training.
		"""
		if self.best_mean_overall_reward is None:
			print('The `best_mean_overall_reward` has never been set. Is this expected?')
		elif self.best_mean_overall_reward == 0:
			print('The mean reward of the agent was never higher than 0, so no models were saved!')
		else:
			print(f'The best mean reward reached by the agent was {self.best_mean_overall_reward:.3f}')
			print('The models were saved to:')
			print(os.path.abspath(self.model_path))
