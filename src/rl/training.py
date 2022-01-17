import math
import os
import signal
import sys
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut


class RLTrainer():

	def __init__(self, environment, RL_agent):
		assert isinstance(RL_agent, vendors.QLearningAgent), f'the passed agent must be a QLearningAgent: {RL_agent}'
		# TODO: assert Agent and environment fit together
		self.best_mean_reward = None
		self.environment = environment
		self.RL_agent = RL_agent
		# Signal handler for e.g. KeyboardInterrupt
		signal.signal(signal.SIGINT, self._signal_handler)

	def _end_of_training(self):
		"""
		Inform the user of the best_mean_reward the agent achieved during training.
		"""
		if self.best_mean_reward == 0:
			print('The mean reward of the agent was never higher than 0, so no models were saved!')
		else:
			print(f'The best mean reward reached by the agent was {self.best_mean_reward:.3f}')
			print('The models were saved to:')
			print(os.path.abspath(os.path.join('trainedModels', f'{type(self.environment).__name__}_{type(self.RL_agent).__name__}')))

	def _signal_handler(self, signum, frame):  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		print('\nAborting training...')
		self._end_of_training()
		sys.exit(0)

	def train_QLearning_agent(self, maxsteps=2 * config.EPSILON_DECAY_LAST_FRAME, log_dir_prepend='') -> None:
		"""
		Train a QLearningAgent on a market environment.

		Args:
			maxsteps (int, optional): The maximum number of steps the training will run for. Defaults to 2*config.EPSILON_DECAY_LAST_FRAME.
			log_dir_prepend (str, optional): A string that is prepended to the log directory created by Tensorboard. Defaults to ''.
		"""
		state = self.environment.reset()

		frame_number_last_speed_update = 0
		time_last_speed_update = time.time()
		vendors_cumulated_info = None
		all_dicts = []

		losses = []
		rmse_losses = []
		selected_q_vals = []
		self.best_mean_reward = 0

		curr_time = time.strftime('%b%d_%H-%M-%S')
		signature = f'{type(self.environment).__name__}_{type(self.RL_agent).__name__}'
		writer = SummaryWriter(log_dir=os.path.join('results', 'runs', f'{log_dir_prepend}training_{curr_time}'))

		for frame_idx in range(maxsteps):
			epsilon = max(config.EPSILON_FINAL, config.EPSILON_START - frame_idx / config.EPSILON_DECAY_LAST_FRAME)

			action = self.RL_agent.policy(state, epsilon)
			state, reward, is_done, info = self.environment.step(action)
			self.RL_agent.set_feedback(reward, is_done, state)
			vendors_cumulated_info = info if vendors_cumulated_info is None else ut.add_content_of_two_dicts(vendors_cumulated_info, info)

			if is_done:
				all_dicts.append(vendors_cumulated_info)
				speed = (frame_idx - frame_number_last_speed_update) / (
					(time.time() - time_last_speed_update) if (time.time() - time_last_speed_update) > 0 else 1
				)
				frame_number_last_speed_update = frame_idx
				time_last_speed_update = time.time()

				# calculate the average of the last 100 items
				sliced_dicts = all_dicts[-100:]
				averaged_info = sliced_dicts[0]
				for i, next_dict in enumerate(sliced_dicts):
					if i != 0:
						averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
				averaged_info = ut.divide_content_of_dict(averaged_info, len(sliced_dicts))

				mean_reward = averaged_info['profits/all']['vendor_0']

				writer.add_scalar('Profit_mean/agent', mean_reward, frame_idx / config.EPISODE_LENGTH)
				ut.write_dict_to_tensorboard(writer, averaged_info, frame_idx / config.EPISODE_LENGTH, is_cumulative=True)
				if frame_idx > config.REPLAY_START_SIZE:
					writer.add_scalar(
						'Loss/MSE', np.mean(losses[-1000:]), frame_idx / config.EPISODE_LENGTH
					)
					writer.add_scalar(
						'Loss/RMSE', np.mean(rmse_losses[-1000:]), frame_idx / config.EPISODE_LENGTH
					)
					writer.add_scalar(
						'Loss/selected_q_vals',
						np.mean(selected_q_vals[-1000:]),
						frame_idx / config.EPISODE_LENGTH,
					)
				writer.add_scalar('epsilon', epsilon, frame_idx / config.EPISODE_LENGTH)
				print(f'''{frame_idx}: done {len(all_dicts)} games, this episode return {all_dicts[-1]['profits/all']['vendor_0']:.3f}, mean return {mean_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f} f/s''')

				if (frame_idx > config.EPSILON_DECAY_LAST_FRAME + 101) and (self.best_mean_reward < mean_reward):
					self.RL_agent.save(path_name=f'{signature}_{curr_time}', model_name=f'{signature}_{mean_reward:.3f}')
					if self.best_mean_reward != 0:
						print(f'Best reward updated {self.best_mean_reward:.3f} -> {mean_reward:.3f}')
					self.best_mean_reward = mean_reward
				if mean_reward > config.MEAN_REWARD_BOUND:
					print(f'Solved in {frame_idx} frames!')
					break

				vendors_cumulated_info = None
				self.environment.reset()

			if len(self.RL_agent.buffer) < config.REPLAY_START_SIZE:
				continue

			if frame_idx % config.SYNC_TARGET_FRAMES == 0:
				self.RL_agent.synchronize_tgt_net()

			loss, selected_q_val_mean = self.RL_agent.train_batch()
			losses.append(loss)
			rmse_losses.append(math.sqrt(loss))
			selected_q_vals.append(selected_q_val_mean)

		self._end_of_training()
