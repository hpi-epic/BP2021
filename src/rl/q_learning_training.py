import numpy as np

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut
from rl.training import RLTrainer


class QLearningTrainer(RLTrainer):
	def trainer_agent_fit(self):
		return isinstance(self.RL_agent, vendors.QLearningAgent), f'the passed agent must be a QLearningAgent: {self.RL_agent}'

	def train_agent(self, maxsteps=2 * config.EPSILON_DECAY_LAST_FRAME) -> None:
		"""
		Train a QLearningAgent on a marketplace.

		Args:
			maxsteps (int, optional): The maximum number of steps the training will run for. Defaults to 2*config.EPSILON_DECAY_LAST_FRAME.
		"""
		state = self.marketplace.reset()

		vendors_cumulated_info = None
		all_dicts = []
		losses = []
		rmse_losses = []
		selected_q_vals = []

		for frame_idx in range(maxsteps):
			epsilon = max(config.EPSILON_FINAL, config.EPSILON_START - frame_idx / config.EPSILON_DECAY_LAST_FRAME)

			action = self.RL_agent.policy(state, epsilon)
			state, reward, is_done, info = self.marketplace.step(action)
			self.RL_agent.set_feedback(reward, is_done, state)
			vendors_cumulated_info = info if vendors_cumulated_info is None else ut.add_content_of_two_dicts(vendors_cumulated_info, info)

			if is_done:
				all_dicts.append(vendors_cumulated_info)
				averaged_info = self.calculate_dict_average(all_dicts)

				if frame_idx > config.REPLAY_START_SIZE:
					averaged_info['Loss/MSE'] = np.mean(losses[-1000:])
					averaged_info['Loss/RMSE'] = np.mean(rmse_losses[-1000:])
					averaged_info['Loss/selected_q_vals'] = np.mean(selected_q_vals[-1000:])
					averaged_info['epsilon'] = epsilon
					ut.write_dict_to_tensorboard(self.writer, averaged_info, frame_idx / config.EPISODE_LENGTH)
					self.consider_print_info(frame_idx, len(all_dicts), averaged_info, epsilon)
					self.consider_update_best_model(averaged_info)

				vendors_cumulated_info = None
				self.marketplace.reset()

			if len(self.RL_agent.buffer) < config.REPLAY_START_SIZE:
				continue

			loss, selected_q_val_mean = self.RL_agent.train_batch()
			losses.append(loss)
			rmse_losses.append(np.sqrt(loss))
			selected_q_vals.append(selected_q_val_mean)

			self.consider_sync_tgt_net(frame_idx)

		self._end_of_training()
