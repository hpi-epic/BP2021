import os
import signal
import sys
import time
import warnings

import numpy as np
import stable_baselines3.common.monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
from recommerce.configuration.hyperparameter_config import config
from recommerce.configuration.path_manager import PathManager
from recommerce.rl.stable_baselines_model import StableBaselinesDDPG

warnings.filterwarnings('ignore')


class PerStepCheck(BaseCallback):
	"""
	Callback for saving a model (the check is done every ``check_freq`` steps)
	based on the training reward (in practice, we recommend using ``EvalCallback``).

	:param check_freq:
	:param log_dir: Path to the folder where the model will be saved.
		It must contains the file created by the ``Monitor`` wrapper.
	:param verbose: Verbosity level.
	"""
	def __init__(self, marketplace_class, agent_class, log_dir_prepend='', iteration_length=500):
		assert isinstance(log_dir_prepend, str), \
			f'log_dir_prepend should be a string, but {log_dir_prepend} is {type(log_dir_prepend)}'
		super(PerStepCheck, self).__init__(True)
		self.best_mean_interim_reward = None
		self.best_mean_overall_reward = None
		self.marketplace_class = marketplace_class
		self.agent_class = agent_class
		self.iteration_length = iteration_length
		self.saved_parameter_paths = []
		signal.signal(signal.SIGINT, self._signal_handler)

		self.initialize_io_related(log_dir_prepend)
		self.reset_time_tracker()

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
		which are needed for saving the models and writing to tensorboard
		Args:
			log_dir_prepend (str): A prefix that is written before the saved data
		"""
		ut.ensure_results_folders_exist()
		self.curr_time = time.strftime('%b%d_%H-%M-%S')
		self.signature = 'Stable_Baselines_Training'
		self.writer = SummaryWriter(log_dir=os.path.join(PathManager.results_path, 'runs', f'{log_dir_prepend}training_{self.curr_time}'))
		path_name = f'{self.signature}_{self.curr_time}'
		self.save_path = os.path.join(PathManager.results_path, 'trainedModels', log_dir_prepend + path_name)
		self.tmp_path = os.path.join(self.save_path, 'tmp_model')
		os.makedirs(os.path.abspath(self.save_path), exist_ok=True)

	def reset_time_tracker(self) -> None:
		self.frame_number_last_speed_update = 0
		self.time_last_speed_update = time.time()

	def _on_step(self) -> bool:
		if (self.num_timesteps - 1) % config.episode_length != 0 or self.num_timesteps <= config.episode_length:
			return True
		finished_episodes = self.num_timesteps // config.episode_length
		x, y = ts2xy(load_results(self.save_path), 'timesteps')
		mean_reward = np.mean(y[-100:])
		assert len(x) > 0

		# consider print info
		if (finished_episodes) % 10 == 0:
			tqdm.write(f'{self.num_timesteps}: {finished_episodes} episodes trained, mean return {mean_reward:.3f}')

		# consider update best model
		if self.best_mean_interim_reward is None or mean_reward > self.best_mean_interim_reward:
			self.model.save(self.tmp_path)
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
		if self.best_mean_interim_reward is not None:
			finished_episodes = self.num_timesteps // config.episode_length
			self.save_parameters(finished_episodes)

	def save_parameters(self, finished_episodes):
		path_to_parameters = os.path.join(self.save_path, f'{self.signature}_{finished_episodes:05d}')
		os.rename(self.tmp_path + '.zip', path_to_parameters + '.zip')
		self.saved_parameter_paths.append(path_to_parameters)
		tqdm.write(f'I write the interim model after {finished_episodes} episodes to the disk.')
		tqdm.write(f'You can find the parameters here: {path_to_parameters}.')
		tqdm.write(f'This model achieved a mean reward of {self.best_mean_interim_reward}.')
		self.best_mean_interim_reward = None


def train_ddpg_agent(marketplace_class, training_steps=100000, iteration_length=500):
	marketplace = marketplace_class(True)
	callback = PerStepCheck(marketplace_class, StableBaselinesDDPG, iteration_length=iteration_length)
	marketplace = stable_baselines3.common.monitor.Monitor(marketplace, callback.save_path)
	agent = StableBaselinesDDPG(marketplace)
	print('Now I start the training')
	agent.model.learn(training_steps, callback=callback)
	# agent.save(os.path.join(callback.save_path, "final_model"))


if __name__ == '__main__':
	train_ddpg_agent(circular_market.CircularEconomyRebuyPriceOneCompetitor)
