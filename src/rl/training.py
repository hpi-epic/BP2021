import math
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut


# Gets the profit array of all vendors and returns the necessary dict for direct comparison in tb
def direct_comparison_dict(profits) -> dict:
	comparison_dict = {}
	n_vendors = len(profits[0])
	for i in range(n_vendors):
		last = profits[-100:]
		matrix = np.concatenate(last).reshape(-1, n_vendors)
		comparison_dict['vendor_' + str(i)] = np.mean(matrix[:, i])
	return comparison_dict


def train_QLearning_agent(RL_agent, environment, maxsteps=2 * config.EPSILON_DECAY_LAST_FRAME, log_dir_prepend='') -> None:
	"""
	Train a QLearningAgent on a market environment.

	Args:
		RL_agent (agents.vendors instance): The agent that should be trained.
		environment (market.sim_market instance): The market environment used for the training.
		maxsteps (int, optional): The maximum number of steps the training will run for. Defaults to 2*config.EPSILON_DECAY_LAST_FRAME.
		log_dir_prepend (str, optional): A string that is prepended to the log directory created by Tensorboard. Defaults to ''.
	"""
	assert isinstance(RL_agent, vendors.QLearningAgent), f'the passed agent must be a QLearningAgent: {RL_agent}'
	state = environment.reset()

	frame_number_last_speed_update = 0
	time_last_speed_update = time.time()
	vendors_cumulated_info = None
	all_dicts = []

	losses = []
	rmse_losses = []
	selected_q_vals = []
	best_mean_reward = 0

	curr_time = time.strftime('%b%d_%H-%M-%S')
	signature = f'{type(environment).__name__}_{type(RL_agent).__name__}'
	writer = SummaryWriter(log_dir=os.path.join('results', 'runs', f'{log_dir_prepend}training_{curr_time}'))

	for frame_idx in range(maxsteps):
		epsilon = max(config.EPSILON_FINAL, config.EPSILON_START - frame_idx / config.EPSILON_DECAY_LAST_FRAME)

		action = RL_agent.policy(state, epsilon)
		state, reward, is_done, info = environment.step(action)
		RL_agent.set_feedback(reward, is_done, state)
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

			if (frame_idx > config.EPSILON_DECAY_LAST_FRAME + 101) and (best_mean_reward < mean_reward):
				RL_agent.save(path_name=f'{signature}_{curr_time}', model_name=f'{signature}_{mean_reward:.3f}')
				if best_mean_reward != 0:
					print(f'Best reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}')
				best_mean_reward = mean_reward
			if mean_reward > config.MEAN_REWARD_BOUND:
				print(f'Solved in {frame_idx} frames!')
				break

			vendors_cumulated_info = None
			environment.reset()

		if len(RL_agent.buffer) < config.REPLAY_START_SIZE:
			continue

		if frame_idx % config.SYNC_TARGET_FRAMES == 0:
			RL_agent.synchronize_tgt_net()

		loss, selected_q_val_mean = RL_agent.train_batch()
		losses.append(loss)
		rmse_losses.append(math.sqrt(loss))
		selected_q_vals.append(selected_q_val_mean)

	if best_mean_reward == 0:
		print('The mean reward of the agent was never higher than 0, so no models were saved!')
	else:
		print(f'The best mean reward reached by the agent was {best_mean_reward:.3f}')
		print('The models were saved to:')
		print(os.path.abspath(os.path.join('results', 'trainedModels', f'{type(environment).__name__}_{type(RL_agent).__name__}_{curr_time}')))
