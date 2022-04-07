import os
import signal
import sys
import time
import warnings

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.auto import trange

import recommerce.configuration.utils as ut
from recommerce.configuration.hyperparameter_config import config
from recommerce.configuration.path_manager import PathManager
from recommerce.market.sim_market import SimMarket
from recommerce.monitoring.agent_monitoring.am_evaluation import Evaluator
from recommerce.monitoring.agent_monitoring.am_monitoring import Monitor
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent

warnings.filterwarnings('ignore')


class PerStepCheck(BaseCallback):
	def __init__(self, agent_class, marketplace_class, log_dir_prepend='', training_steps=10000, iteration_length=500):
		assert issubclass(agent_class, ReinforcementLearningAgent)
		assert issubclass(marketplace_class, SimMarket)
		assert isinstance(log_dir_prepend, str), \
			f'log_dir_prepend should be a string, but {log_dir_prepend} is {type(log_dir_prepend)}'
		assert isinstance(training_steps, int) and training_steps > 0
		assert isinstance(iteration_length, int) and iteration_length > 0
		super(PerStepCheck, self).__init__(True)
		self.best_mean_interim_reward = None
		self.best_mean_overall_reward = None
		self.marketplace_class = marketplace_class
		self.agent_class = agent_class
		self.iteration_length = iteration_length
		self.tqdm_instance = trange(training_steps)
		self.saved_parameter_paths = []
		signal.signal(signal.SIGINT, self._signal_handler)

		self.initialize_io_related(log_dir_prepend)

	def _signal_handler(self, signum, frame) -> None:  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		print('\nAborting training...')
		# self._end_of_training()
		sys.exit(0)

	def initialize_io_related(self, log_dir_prepend) -> None:
		"""
		Initializes the local variables self.curr_time, self.signature, self.writer, self.save_path
		and self.tmp_parameters which are needed for saving the models and writing to tensorboard
		Args:
			log_dir_prepend (str): A prefix that is written before the saved data
		"""
		ut.ensure_results_folders_exist()
		self.curr_time = time.strftime('%b%d_%H-%M-%S')
		self.signature = 'Stable_Baselines_Training'
		self.writer = SummaryWriter(log_dir=os.path.join(PathManager.results_path, 'runs', f'{log_dir_prepend}training_{self.curr_time}'))
		path_name = f'{self.signature}_{self.curr_time}'
		self.save_path = os.path.join(PathManager.results_path, 'trainedModels', log_dir_prepend + path_name)
		os.makedirs(os.path.abspath(self.save_path), exist_ok=True)
		self.tmp_parameters = os.path.join(self.save_path, 'tmp_model.zip')

	def _on_step(self) -> bool:
		self.tqdm_instance.update()
		if (self.num_timesteps - 1) % config.episode_length != 0 or self.num_timesteps <= config.episode_length:
			return True
		self.tqdm_instance.refresh()
		finished_episodes = self.num_timesteps // config.episode_length
		x, y = ts2xy(load_results(self.save_path), 'timesteps')
		assert len(x) > 0 and len(x) == len(y)
		mean_reward = np.mean(y[-100:])

		# consider print info
		if (finished_episodes) % 10 == 0:
			tqdm.write(f'{self.num_timesteps}: {finished_episodes} episodes trained, mean return {mean_reward:.3f}')

		# consider update best model
		if self.best_mean_interim_reward is None or mean_reward > self.best_mean_interim_reward + 15:
			self.model.save(self.tmp_parameters)
			self.best_mean_interim_reward = mean_reward
			if self.best_mean_overall_reward is None or self.best_mean_interim_reward > self.best_mean_overall_reward:
				if self.best_mean_overall_reward is not None:
					tqdm.write(f'Best overall reward updated {self.best_mean_overall_reward:.3f} -> {self.best_mean_interim_reward:.3f}')
				self.best_mean_overall_reward = self.best_mean_interim_reward

		# consider save model
		if (finished_episodes % self.iteration_length == 0 and finished_episodes > 0) and self.best_mean_interim_reward is not None:
			self.save_parameters(finished_episodes)

		return True

	def _on_training_end(self) -> None:
		self.tqdm_instance.close()
		if self.best_mean_interim_reward is not None:
			finished_episodes = self.num_timesteps // config.episode_length
			self.save_parameters(finished_episodes)

		# analyze trained agents
		if len(self.saved_parameter_paths) == 0:
			print('No agents saved! Nothing to monitor.')
			return
		monitor = Monitor()
		agent_list = [(self.agent_class, [parameter_path]) for parameter_path in self.saved_parameter_paths]
		monitor.configurator.setup_monitoring(False, 250, 250, self.marketplace_class, agent_list, support_continuouos_action_space=True)
		rewards = monitor.run_marketplace()
		episode_numbers = [int(parameter_path[-9:][:5]) for parameter_path in self.saved_parameter_paths]
		Evaluator(monitor.configurator).evaluate_session(rewards, episode_numbers)

	def save_parameters(self, finished_episodes):
		path_to_parameters = os.path.join(self.save_path, f'{self.signature}_{finished_episodes:05d}.zip')
		os.rename(self.tmp_parameters, path_to_parameters)
		self.saved_parameter_paths.append(path_to_parameters)
		tqdm.write(f'I write the interim model after {finished_episodes} episodes to the disk.')
		tqdm.write(f'You can find the parameters here: {path_to_parameters}.')
		tqdm.write(f'This model achieved a mean reward of {self.best_mean_interim_reward}.')
		self.best_mean_interim_reward = None
