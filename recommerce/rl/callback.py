import os
import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.auto import trange

import recommerce.configuration.utils as ut
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly, CircularEconomyRebuyPriceVariableDuopoly
from recommerce.market.sim_market import SimMarket
from recommerce.monitoring.agent_monitoring.am_evaluation import Evaluator
from recommerce.monitoring.agent_monitoring.am_monitoring import Monitor
from recommerce.monitoring.watcher import Watcher
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
		# Change the 128 to the right parameter as soon as #211 is implemented.
		self.watcher = Watcher(128) if issubclass(agent_class, ActorCriticAgent) else Watcher()
		self.best_mean_interim_reward = None
		self.best_mean_overall_reward = None
		self.marketplace_class = marketplace_class
		self.agent_class = agent_class
		self.iteration_length = iteration_length
		self.file_ending = file_ending
		self.signature = signature
		self.tqdm_instance = trange(training_steps)
		self.saved_parameter_paths = []
		self.last_finished_episode = 0
		self.analyze_after_training = analyze_after_training
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

	def _on_step(self, finished_episodes: int = None, info: dict = None, env_index: int = 0) -> bool:
		"""
		This method is called during training after step in the environment is called.
		If you don't provide finished_episodes and mean_return, the agent will conclude this from the number of timesteps.
		Note that you must provide finished_episodes if and only if you provide mean_return.

		Args:
			finished_episodes (int, optional): The episodes that are already finished. Defaults to None.
			info (dict, optional): The info generated by the last step. Defaults to None.
			env_index (int, optional): The index of the environment the last step was done in. Defaults to 0.

		Returns:
			bool: True should be returned. False will be interpreted as error.
		"""
		assert (finished_episodes is None) == (info is None), 'finished_episodes must be exactly None if info is None'

		# This means if it is a subclass of StableBaselinesAgent. Unfortunately, circular imports are not possible.
		if not issubclass(self.agent_class, QLearningAgent) and not issubclass(self.agent_class, ActorCriticAgent):
			# self.locals is a feature offered by stablebaselines
			# locals is a dict with all local variables in the training method of stablebaselines
			info = self.locals['infos'][0]
			finished_episodes = len(self.watcher.all_dicts)

		self.watcher.add_info(info)
		self.tqdm_instance.update()

		assert isinstance(finished_episodes, int)

		assert finished_episodes >= self.last_finished_episode
		# Do the following part only if a new episode is finished.
		if finished_episodes == self.last_finished_episode or finished_episodes < 5:
			return True

		ut.write_dict_to_tensorboard(self.writer, self.watcher.get_average_dict(), finished_episodes, is_cumulative=True)
		self.last_finished_episode = finished_episodes
		mean_return = self.watcher.get_average_dict()['profits/all']['vendor_0']
		assert isinstance(mean_return, float)

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

		monitor = Monitor()
		monitor.configurator.get_folder()

		print('Creating scatterplots...')
		ignore_first_samples = 15  # the number of samples you want to skip because they can be severe outliers
		cumulative_properties = self.watcher.get_cumulative_properties()
		for property, samples in ut.unroll_dict_with_list(cumulative_properties).items():
			x_values = np.array(range(len(samples[ignore_first_samples:]))) + ignore_first_samples
			plt.clf()
			plt.scatter(x_values, samples[ignore_first_samples:])
			plt.title(f'All samples of {property} for each episode')
			plt.xlabel('Episode')
			plt.ylabel(property)
			plt.savefig(os.path.join(monitor.configurator.folder_path, f'scatterplot_samples_{property.replace("/", "_")}.svg'))

		print('Creating lineplots...')
		for property, samples in cumulative_properties.items():
			plt.clf()
			if not isinstance(samples[0], list):
				plot_values = [self.watcher.get_progress_values_of_property(property)[ignore_first_samples:]]
			else:
				plot_values = [self.watcher.get_progress_values_of_property(property, vendor)[ignore_first_samples:]
					for vendor in range(self.watcher.get_number_of_vendors())]

			for vendor, values in enumerate(plot_values):
				plt.plot(x_values, values, label=f'Vendor {vendor}')

			if isinstance(samples[0], list):
				plt.legend()
			plt.title(f'Rolling average training progress of {property}')
			plt.xlabel('Episode')
			plt.ylabel(property)
			plt.savefig(os.path.join(monitor.configurator.folder_path, f'lineplot_progress_{property.replace("/", "_")}.svg'))

		if self.analyze_after_training:
			agent_list = [(self.agent_class, [parameter_path]) for parameter_path in self.saved_parameter_paths]
			# The next line is a bit hacky. We have to provide if the marketplace is continuos or not.
			# Only Stable Baselines agents use continuous actions at the moment. And only Stable Baselines agents have the attribute env.
			# The correct way of doing this would be by checking for `isinstance(StableBaselinesAgent)`, but that would result in a circular import.
			monitor.configurator.setup_monitoring(
				False, 250, 250,
				(CircularEconomyRebuyPriceDuopoly if issubclass(self.marketplace_class, CircularEconomyRebuyPriceVariableDuopoly)
					else self.marketplace_class),
				agent_list,
				support_continuous_action_space=hasattr(self.model, 'env'))
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
