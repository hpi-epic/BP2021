import numpy as np

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
			number_of_training_steps = 2 * self.config_rl.epsilon_decay_last_frame
		marketplace = self.initialize_callback(number_of_training_steps)
		state = marketplace.reset()

		last_loss = 0
		last_q_val_selected_action = 0
		finished_episodes = 0

		for frame_idx in range(number_of_training_steps):
			epsilon = max(self.config_rl.epsilon_final, self.config_rl.epsilon_start - frame_idx / self.config_rl.epsilon_decay_last_frame)

			action = self.callback.model.policy(state, epsilon)
			state, reward, is_done, info = marketplace.step(action)

			# The following numbers are divided by the episode length because they will be summed up later in the watcher
			info['Loss/MSE'] = last_loss / self.config_market.episode_length
			info['Loss/RMSE'] = np.sqrt(last_loss) / self.config_market.episode_length
			info['Loss/selected_q_vals'] = last_q_val_selected_action / self.config_market.episode_length
			info['epsilon'] = epsilon / self.config_market.episode_length

			self.callback.model.set_feedback(reward, is_done, state)

			if is_done:
				finished_episodes += 1

			self.callback.num_timesteps = frame_idx
			self.callback._on_step(finished_episodes, info)
			if is_done:
				marketplace.reset()

			if len(self.callback.model.buffer) < self.config_rl.replay_start_size:
				continue

			last_loss, last_q_val_selected_action = self.callback.model.train_batch()

			self.consider_sync_tgt_net(frame_idx)

		self.callback._on_training_end()
