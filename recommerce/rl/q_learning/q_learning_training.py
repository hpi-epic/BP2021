import numpy as np

import recommerce.configuration.utils as ut
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.rl.training import RLTrainer


class QLearningTrainer(RLTrainer):
	
	def trainer_agent_fit(self) -> bool:
		return issubclass(self.agent_class, QLearningAgent), f'the passed agent must be a QLearningAgent: {self.agent_class}'

	def train_agent(self, number_of_training_steps=None) -> None:
		"""
		Train a QLearningAgent on a marketplace.

		Args:
			number_of_training_steps (int, optional): The maximum number of steps the training will run for.
			Defaults to 2*self.config.epsilon_decay_last_frame.
		"""
		if number_of_training_steps is None:
			number_of_training_steps = 2 * self.config.epsilon_decay_last_frame
		self.initialize_callback(number_of_training_steps)
		marketplace = self.marketplace_class()
		state = marketplace.reset()

		vendors_cumulated_info = None
		all_dicts = []
		losses = []
		rmse_losses = []
		selected_q_vals = []
		finished_episodes = 0
		mean_return = -np.inf

		for frame_idx in range(number_of_training_steps):
			epsilon = max(self.config.epsilon_final, self.config.epsilon_start - frame_idx / self.config.epsilon_decay_last_frame)

			action = self.callback.model.policy(state, epsilon)
			state, reward, is_done, info = marketplace.step(action)
			self.callback.model.set_feedback(reward, is_done, state)
			vendors_cumulated_info = info if vendors_cumulated_info is None else ut.add_content_of_two_dicts(vendors_cumulated_info, info)

			if is_done:
				all_dicts.append(vendors_cumulated_info)
				finished_episodes = len(all_dicts)
				averaged_info = self.calculate_dict_average(all_dicts)

				if frame_idx > self.config.replay_start_size:
					averaged_info['Loss/MSE'] = np.mean(losses[-1000:])
					averaged_info['Loss/RMSE'] = np.mean(rmse_losses[-1000:])
					averaged_info['Loss/selected_q_vals'] = np.mean(selected_q_vals[-1000:])
					averaged_info['epsilon'] = epsilon
					ut.write_dict_to_tensorboard(self.callback.writer, averaged_info, frame_idx / self.config.episode_length, is_cumulative=True)
					mean_return = averaged_info['profits/all']['vendor_0']

				vendors_cumulated_info = None
				marketplace.reset()

			self.callback.num_timesteps = frame_idx + 1
			self.callback._on_step(finished_episodes, mean_return)

			if len(self.callback.model.buffer) < self.config.replay_start_size:
				continue

			loss, selected_q_val_mean = self.callback.model.train_batch()
			losses.append(loss)
			rmse_losses.append(np.sqrt(loss))
			selected_q_vals.append(selected_q_val_mean)

			self.consider_sync_tgt_net(frame_idx)

		self.callback._on_training_end()
