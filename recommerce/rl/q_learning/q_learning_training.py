import numpy as np

import recommerce.configuration.utils as ut
from recommerce.configuration.hyperparameter_config import config
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.rl.training import RLTrainer


class QLearningTrainer(RLTrainer):
	def trainer_agent_fit(self) -> bool:
		return issubclass(self.agent_class, QLearningAgent), f'the passed agent must be a QLearningAgent: {self.agent_class}'

	def train_agent(self, number_of_training_steps=2 * config.epsilon_decay_last_frame) -> None:
		"""
		Train a QLearningAgent on a marketplace.

		Args:
			number_of_training_steps (int, optional): The maximum number of steps the training will run for.
			Defaults to 2*config.epsilon_decay_last_frame.
		"""
		self.initialize_callback(number_of_training_steps)
		marketplace = self.marketplace_class()
		state = marketplace.reset()

		losses = []
		rmse_losses = []
		selected_q_vals = []
		finished_episodes = 0

		for frame_idx in range(number_of_training_steps):
			epsilon = max(config.epsilon_final, config.epsilon_start - frame_idx / config.epsilon_decay_last_frame)

			action = self.callback.model.policy(state, epsilon)
			state, reward, is_done, info = marketplace.step(action)
			self.callback.model.set_feedback(reward, is_done, state)

			if is_done:
				finished_episodes += 1

			self.callback._on_step(finished_episodes, info)
			if is_done:
				averaged_info = self.callback.watcher.get_average_dict()

				if frame_idx > config.replay_start_size:
					averaged_info['Loss/MSE'] = np.mean(losses[-1000:])
					averaged_info['Loss/RMSE'] = np.mean(rmse_losses[-1000:])
					averaged_info['Loss/selected_q_vals'] = np.mean(selected_q_vals[-1000:])
					averaged_info['epsilon'] = epsilon
					ut.write_dict_to_tensorboard(self.callback.writer, averaged_info, frame_idx / config.episode_length, is_cumulative=True)

				marketplace.reset()

			self.callback.num_timesteps = frame_idx + 1

			if len(self.callback.model.buffer) < config.replay_start_size:
				continue

			loss, selected_q_val_mean = self.callback.model.train_batch()
			losses.append(loss)
			rmse_losses.append(np.sqrt(loss))
			selected_q_vals.append(selected_q_val_mean)

			self.consider_sync_tgt_net(frame_idx)

		self.callback._on_training_end()
