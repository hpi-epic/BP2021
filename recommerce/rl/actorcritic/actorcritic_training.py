import random

import numpy as np
import torch

import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import config
from recommerce.rl.training import RLTrainer


class ActorCriticTrainer(RLTrainer):
	def trainer_agent_fit(self) -> bool:
		return issubclass(self.agent_class, actorcritic_agent.ActorCriticAgent)

	def choose_random_envs(self, total_envs) -> set:
		"""
		This method samples config.batch_size distinct numbers out of 0, ..., total_envs - 1

		Args:
			total_envs (int): The number of envs

		Returns:
			set: the distinct shuffled numbers
		"""
		chosen_envs = set()
		while len(chosen_envs) < config.batch_size:
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
		self.initialize_callback(number_of_training_steps * config.batch_size)

		last_value_loss = 0
		last_policy_loss = 0

		finished_episodes = 0
		self.callback.num_timesteps = 0
		environments = [self.marketplace_class() for _ in range(total_envs)]

		for step_number in range(number_of_training_steps):
			chosen_envs = self.choose_random_envs(total_envs)

			states = []
			actions = []
			rewards = []
			states_dash = []
			for env in chosen_envs:
				self.callback.num_timesteps += 1
				state = environments[env]._observation()
				if not verbose:
					action = self.callback.model.policy(state, verbose=False, raw_action=True)
				else:
					action, net_output, v_estimate = self.callback.model.policy(state, verbose=True, raw_action=True)
				next_state, reward, is_done, info = environments[env].step(self.callback.model.agent_output_to_market_form(action))

				# The following numbers are divided by the episode length because they will be summed up later in the watcher
				info['loss/value'] = last_value_loss / config.episode_length
				info['loss/policy'] = last_policy_loss / config.episode_length
				if verbose:
					if isinstance(net_output, np.float32):
						info['verbose/net_output'] = net_output
					else:
						for action_num, output in enumerate(net_output):
							info[f'verbose/information_{str(action_num)}'] = output / config.episode_length
					info['verbose/v_estimate'] = v_estimate / config.episode_length

				states.append(state)
				actions.append(action)
				rewards.append(reward)
				states_dash.append(next_state)

				if is_done:
					finished_episodes += 1

				self.callback._on_step(finished_episodes, info, env)
				if is_done:
					environments[env].reset()

			last_policy_loss, last_value_loss = self.callback.model.train_batch(
				torch.Tensor(np.array(states)),
				torch.from_numpy(np.array(actions, dtype=np.int64)),
				torch.Tensor(np.array(rewards)),
				torch.Tensor(np.array(next_state)),
				finished_episodes <= 500)

			self.consider_sync_tgt_net(step_number)

		self.callback._on_training_end()
