import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import configuration.config as config
import configuration.utils as ut
import rl.actorcritic_agent as actorcritic_agent
from rl.training import RLTrainer


class ActorCriticTrainer(RLTrainer):
	def trainer_agent_fit(self):
		return isinstance(self.RL_agent, actorcritic_agent.ActorCriticAgent)

	def choose_random_envs(self, total_envs):
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

	def train_agent(self, number_of_training_steps=200, verbose=False, total_envs=128):
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
		best_mean_reward = 0

		curr_time = time.strftime('%b%d_%H-%M-%S')
		writer = SummaryWriter(log_dir=os.path.join('results', 'runs', f'training_AC_{curr_time}'))

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
					action = self.RL_agent.policy(state, True)
				else:
					action, net_output, v_estimate = self.RL_agent.policy_verbose(state)
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
					if finished_episodes % 10 == 0:
						print(f'Finished {finished_episodes} episodes')
					all_dicts.append(info_accumulators[env])

					# calculate the average of the last 100 items
					sliced_dicts = all_dicts[-100:]
					averaged_info = sliced_dicts[0]
					for dict_number, next_dict in enumerate(sliced_dicts):
						if dict_number != 0:
							averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
					averaged_info = ut.divide_content_of_dict(averaged_info, len(sliced_dicts))
					ut.write_dict_to_tensorboard(writer, averaged_info, finished_episodes, is_cumulative=True)
					if verbose:
						writer.add_scalar('verbose/v_estimate', np.mean(all_v_estimates[-1000:]), finished_episodes)
						myactions = np.array(all_network_outputs[-1000:])
						for action_num in range(len(all_network_outputs[0])):
							writer.add_scalar('verbose/mean/information_' + str(action_num), np.mean(myactions[:, action_num]), finished_episodes)
							writer.add_scalar('verbose/min/information_' + str(action_num), np.min(myactions[:, action_num]), finished_episodes)
							writer.add_scalar('verbose/max/information_' + str(action_num), np.max(myactions[:, action_num]), finished_episodes)
					writer.add_scalar('loss/value', np.mean(all_value_losses[-1000:]), finished_episodes)
					writer.add_scalar('loss/policy', np.mean(all_policy_losses[-1000:]), finished_episodes)

					environments[env].reset()
					info_accumulators[env] = None

					mean_reward = averaged_info['profits/all']['vendor_0']
					if best_mean_reward < mean_reward:
						self.RL_agent.save(path_name=f'{self.signature}_{curr_time}', model_name=f'{self.signature}_{mean_reward:.3f}')
						if best_mean_reward is not None:
							print(f'Best reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}')
						best_mean_reward = mean_reward

			policy_loss, valueloss = self.RL_agent.train_batch(
				torch.Tensor(np.array(states)),
				torch.from_numpy(np.array(actions, dtype=np.int64)),
				torch.Tensor(np.array(rewards)),
				torch.Tensor(np.array(next_state)),
				finished_episodes <= 500)
			all_value_losses.append(valueloss)
			all_policy_losses.append(policy_loss)
			if (step_number + 1) % config.SYNC_TARGET_FRAMES == 0:
				self.RL_agent.synchronize_critic_tgt_net()
