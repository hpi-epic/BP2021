import random

import numpy as np
import torch

import configuration.hyperparameters_config as config
import configuration.utils as ut
import rl.actorcritic_agent as actorcritic_agent
from rl.training import RLTrainer


class ActorCriticTrainer(RLTrainer):
	def trainer_agent_fit(self) -> bool:
		return isinstance(self.RL_agent, actorcritic_agent.ActorCriticAgent)

	def choose_random_envs(self, total_envs) -> set:
		"""
		This method samples config.BATCH_SIZE distinct numbers out of 0, ..., total_envs - 1

		Args:
			total_envs (int): The number of envs

		Returns:
			set: the distinct shuffled numbers
		"""
		chosen_envs = set()
		while len(chosen_envs) < config.BATCH_SIZE:
			number = random.randint(0, total_envs - 1)
			if number not in chosen_envs:
				chosen_envs.add(number)
		return chosen_envs

	def train_agent(self, number_of_training_steps=200, verbose=False, total_envs=128) -> None:
		"""
		This is the central method you need to start training of actorcritic_agent.
		You can customize the training by several parameters.

		Args:
			number_of_training_steps (int, optional): The number of batches the agent is trained with. Defaults to 200.
			verbose (bool, optional): Should additional information about agent steps be written to the tensorboard? Defaults to False.
			total_envs (int, optional): The number of environments you use in parallel to fulfill the iid assumption. Defaults to 128.
		"""

		all_dicts = []
		if verbose:
			all_network_outputs = []
			all_v_estimates = []
		all_value_losses = []
		all_policy_losses = []

		finished_episodes = 0
		environments = [self.marketplace_class() for _ in range(total_envs)]
		info_accumulators = [None for _ in range(total_envs)]
		for step_number in range(number_of_training_steps):
			chosen_envs = self.choose_random_envs(total_envs)

			states = []
			actions = []
			rewards = []
			states_dash = []
			for env in chosen_envs:
				state = environments[env]._observation()
				if not verbose:
					action = self.RL_agent.policy(state, verbose=False, raw_action=True)
				else:
					action, net_output, v_estimate = self.RL_agent.policy(state, verbose=True, raw_action=True)
					all_network_outputs.append(net_output.reshape(-1))
					all_v_estimates.append(v_estimate)
				next_state, reward, is_done, info = environments[env].step(self.RL_agent.agent_output_to_market_form(action))

				states.append(state)
				actions.append(action)
				rewards.append(reward)
				states_dash.append(next_state)
				info_accumulators[env] = info if info_accumulators[env] is None else ut.add_content_of_two_dicts(info_accumulators[env], info)

				if is_done:
					finished_episodes += 1
					all_dicts.append(info_accumulators[env])

					averaged_info = self.calculate_dict_average(all_dicts)
					averaged_info['loss/value'] = np.mean(all_value_losses[-1000:])
					averaged_info['loss/policy'] = np.mean(all_policy_losses[-1000:])

					if verbose:
						averaged_info['verbose/v_estimate'] = np.mean(all_v_estimates[-1000:])
						myactions = np.array(all_network_outputs[-1000:])
						for action_num in range(len(all_network_outputs[0])):
							averaged_info[f'verbose/mean/information_{str(action_num)}'] = np.mean(myactions[:, action_num])
							averaged_info[f'verbose/min/information_{str(action_num)}'] = np.min(myactions[:, action_num])
							averaged_info[f'verbose/max/information_{str(action_num)}'] = np.max(myactions[:, action_num])

					ut.write_dict_to_tensorboard(self.writer, averaged_info, finished_episodes, is_cumulative=True)

					environments[env].reset()
					info_accumulators[env] = None

					self.consider_print_info(step_number, finished_episodes, averaged_info)
					self.consider_update_best_model(averaged_info, finished_episodes * config.EPISODE_LENGTH)

			policy_loss, valueloss = self.RL_agent.train_batch(
				torch.Tensor(np.array(states)),
				torch.from_numpy(np.array(actions, dtype=np.int64)),
				torch.Tensor(np.array(rewards)),
				torch.Tensor(np.array(next_state)),
				finished_episodes <= 500)
			all_value_losses.append(valueloss)
			all_policy_losses.append(policy_loss)

			self.consider_sync_tgt_net(step_number)

		self._end_of_training()
