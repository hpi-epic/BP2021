import os
import signal
import sys
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.auto import trange

import recommerce.configuration.utils as ut
from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.configuration.path_manager import PathManager
from recommerce.market.sim_market import SimMarket
from recommerce.monitoring.agent_monitoring.am_evaluation import Evaluator
from recommerce.monitoring.agent_monitoring.am_monitoring import Monitor
from recommerce.rl.actorcritic.actorcritic_agent import ActorCriticAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class RecommerceCallback(BaseCallback):
	"""
	This callback checks if the current mean return is better than the best mean return.
	This check happens every episode.
	After 'iteration_length' episodes, the best model in that time span is saved to disk.
	"""
	def __init__(
		self,
		agent_class,
		marketplace_class,
		config: HyperparameterConfig,
		training_steps=10000,
		iteration_length=500,
		file_ending='zip',
		signature='train',
		analyze_after_training=True):

		assert issubclass(agent_class, ReinforcementLearningAgent)
		assert issubclass(marketplace_class, SimMarket)
		assert isinstance(training_steps, int) and training_steps > 0
		assert isinstance(iteration_length, int) and iteration_length > 0
		super(RecommerceCallback, self).__init__(True)
		self.best_mean_interim_reward = None
		self.best_mean_overall_reward = None
		self.agent_class = agent_class
		self.marketplace_class = marketplace_class
		self.config = config
		self.iteration_length = iteration_length
		self.file_ending = file_ending
		self.signature = signature
		self.tqdm_instance = trange(training_steps)
		self.saved_parameter_paths = []
		self.last_finished_episode = 0
		self.analyze_after_training = analyze_after_training
		self.all_dicts = []
		signal.signal(signal.SIGINT, self._signal_handler)

		self.initialize_io_related()

	def _signal_handler(self, signum, frame) -> None:  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		print('\nAborting training...')
		self._end_of_training()
		sys.exit(0)

	def initialize_io_related(self) -> None:
		"""
		Initializes the local variables self.curr_time, self.signature, self.writer, self.save_path
		and self.tmp_parameters which are needed for saving the models and writing to tensorboard
		"""
		ut.ensure_results_folders_exist()
		self.curr_time = time.strftime('%b%d_%H-%M-%S')
		self.writer = SummaryWriter(log_dir=os.path.join(PathManager.results_path, 'runs', f'training_{self.curr_time}'))
		path_name = f'{self.signature}_{self.curr_time}'
		self.save_path = os.path.join(PathManager.results_path, 'trainedModels', path_name)
		os.makedirs(os.path.abspath(self.save_path), exist_ok=True)
		self.tmp_parameters = os.path.join(self.save_path, f'tmp_model.{self.file_ending}')

	def _on_step(self, finished_episodes: int = None, mean_return: float = None) -> bool:
		"""
		This method is called during training after step in the environment is called.
		If you don't provide finished_episodes and mean_return, the agent will conclude this from the number of timesteps.
		Note that you must provide finished_episodes if and only if you provide mean_return.

		Args:
			finished_episodes (int, optional): The episodes that are already finished. Defaults to None.
			mean_return (float, optional): The recieved return over the last episodes. Defaults to None.

		Returns:
			bool: True should be returned. False will be interpreted as error.
		"""
		assert (finished_episodes is None) == (mean_return is None), 'finished_episodes must be exactly None if mean_return is None'

		# This means if it is a subclass of StableBaselinesAgent. Unfortunately, circular imports are not possible.
		if not issubclass(self.agent_class, QLearningAgent) and not issubclass(self.agent_class, ActorCriticAgent):
			# self.locals is a feature offered by stablebaselines
			# locals is a dict with all local variables in the training method of stablebaselines
			info = self.locals['infos'][0]
			self.all_dicts.append(info)

		self.tqdm_instance.update()
		if finished_episodes is None:
			finished_episodes = self.num_timesteps // self.config.episode_length
			x, y = ts2xy(load_results(self.save_path), 'timesteps')
			if len(x) <= 0:
				return True
			assert len(x) == len(y)
			mean_return = np.mean(y[-100:])
		assert isinstance(finished_episodes, int)
		assert isinstance(mean_return, float)

		assert finished_episodes >= self.last_finished_episode
		if finished_episodes == self.last_finished_episode or finished_episodes < 5:
			return True
		else:
			self.last_finished_episode = finished_episodes

		# consider print info
		if (finished_episodes) % 10 == 0:
			tqdm.write(f'{self.num_timesteps}: {finished_episodes} episodes trained, mean return {mean_return:.3f}')

		# consider update best model
		if self.best_mean_interim_reward is None or mean_return > self.best_mean_interim_reward + 15:
			self.model.save(self.tmp_parameters)
			self.best_mean_interim_reward = mean_return
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
			self.save_parameters(self.last_finished_episode)

		# analyze trained agents
		if len(self.saved_parameter_paths) == 0:
			print('No agents saved! Nothing to monitor.')
			return

		if self.analyze_after_training:
			monitor = Monitor()
			agent_list = [(self.agent_class, [parameter_path]) for parameter_path in self.saved_parameter_paths]
			# The next line is a bit hacky. We have to provide if the marketplace is continuos or not.
			# Only Stable Baselines agents use continuous actions at the moment. And only Stable Baselines agents have the attribute env.
			# The correct way of doing this would be by checking for `isinstance(StableBaselinesAgent)`, but that would result in a circular import.
			monitor.configurator.setup_monitoring(False, 250, 250, self.marketplace_class, agent_list,
				support_continuous_action_space=hasattr(self.model, 'env'), config=self.config)
			rewards = monitor.run_marketplace()
			episode_numbers = [int(parameter_path[-9:][:5]) for parameter_path in self.saved_parameter_paths]
			Evaluator(monitor.configurator).evaluate_session(rewards, episode_numbers)

	def save_parameters(self, finished_episodes: int):
		assert isinstance(finished_episodes, int)
		path_to_parameters = os.path.join(self.save_path, f'{self.signature}_{finished_episodes:05d}.{self.file_ending}')
		os.rename(self.tmp_parameters, path_to_parameters)
		self.saved_parameter_paths.append(path_to_parameters)
		tqdm.write(f'I write the interim model after {finished_episodes} episodes to the disk.')
		tqdm.write(f'You can find the parameters here: {path_to_parameters}.')
		tqdm.write(f'This model achieved a mean reward of {self.best_mean_interim_reward}.')
		self.best_mean_interim_reward = None

	def _end_of_training(self) -> None:
		"""
		Inform the user of the best_mean_overall_reward the agent achieved during training.
		"""
		if self.best_mean_overall_reward is None:
			print('The `best_mean_overall_reward` has never been set. Is this expected?')
		else:
			print(f'The best mean reward reached by the agent was {self.best_mean_overall_reward:.3f}')
			print('The models were saved to:')
			print(os.path.abspath(self.save_path))
