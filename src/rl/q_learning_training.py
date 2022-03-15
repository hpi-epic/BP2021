import numpy as np

import configuration.utils as ut
from configuration.hyperparameter_config import config
from rl.q_learning_agent import QLearningAgent
from rl.training import RLTrainer


class QLearningTrainer(RLTrainer):
	def trainer_agent_fit(self) -> bool:
		return isinstance(self.RL_agent, QLearningAgent), f'the passed agent must be a QLearningAgent: {self.RL_agent}'

	def train_agent(self, number_of_training_steps=2 * config.epsilon_decay_last_frame) -> None:
		"""
		Train a QLearningAgent on a marketplace.

		Args:
			number_of_training_steps (int, optional): The maximum number of steps the training will run for.
			Defaults to 2*config.epsilon_decay_last_frame.
		"""
		self.marketplace = self.marketplace_class()
		state = self.marketplace.reset()

		vendors_cumulated_info = None
		all_dicts = []
		losses = []
		rmse_losses = []
		selected_q_vals = []

		for frame_idx in range(number_of_training_steps):
			epsilon = max(config.epsilon_final, config.epsilon_start - frame_idx / config.epsilon_decay_last_frame)

			action = self.RL_agent.policy(state, epsilon)
			state, reward, is_done, info = self.marketplace.step(action)
			self.RL_agent.set_feedback(reward, is_done, state)
			vendors_cumulated_info = info if vendors_cumulated_info is None else ut.add_content_of_two_dicts(vendors_cumulated_info, info)

			if is_done:
				all_dicts.append(vendors_cumulated_info)
				averaged_info = self.calculate_dict_average(all_dicts)

				if frame_idx > config.replay_start_size:
					averaged_info['Loss/MSE'] = np.mean(losses[-1000:])
					averaged_info['Loss/RMSE'] = np.mean(rmse_losses[-1000:])
					averaged_info['Loss/selected_q_vals'] = np.mean(selected_q_vals[-1000:])
					averaged_info['epsilon'] = epsilon
					ut.write_dict_to_tensorboard(self.writer, averaged_info, frame_idx / config.episode_length, is_cumulative=True)
					self.consider_print_info(frame_idx, len(all_dicts), averaged_info, epsilon)
					self.consider_update_best_model(averaged_info, frame_idx)

				vendors_cumulated_info = None
				self.marketplace.reset()

			if len(self.RL_agent.buffer) < config.replay_start_size:
				continue

			loss, selected_q_val_mean = self.RL_agent.train_batch()
			losses.append(loss)
			rmse_losses.append(np.sqrt(loss))
			selected_q_vals.append(selected_q_val_mean)

			self.consider_sync_tgt_net(frame_idx)

		self._end_of_training()
