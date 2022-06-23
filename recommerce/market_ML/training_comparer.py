
from typing import Tuple

import numpy as np
import pandas as pd
import torch

# from recommerce.configuration.environment_config import EnvironmentConfigLoader, TrainingEnvironmentConfig
# from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.market.sim_market_kalibrated import SimMarketKalibrated
from recommerce.market.sim_market_kalibrator import SimMarketKalibrator

# import recommerce.rl.stable_baselines.stable_baselines_model as sbmodel
# from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC


class CircularEconomyComparerMarket(CircularEconomyRebuyPriceDuopoly):

	def __init__(self, config, support_continuous_action_space: bool = False) -> None:
		data_path = 'data/kalibration_data/training_data_native_marketplace_exploration_after_merge.csv'
		print('Loading data from:', data_path)
		data_frame = pd.read_csv(data_path)
		data = torch.tensor(data_frame.values)[1:, :]
		M123 = (0, 1, 2, 3, 4, 7, 8, 9, 22, 23, 24)  # 1, storage agent, comp prices old, agent prices, comp prices updated
		y1_index = (11, 12, 13)  # sales agent(used, new, rebuy)

		M4 = (0, 1, 2, 3, 4, 7, 8, 9)
		M4x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
		y4_index = 22
		M5 = (0, 1, 2, 3, 4, 7, 8, 9)
		M5x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
		y5_index = 23
		M6 = (0, 1, 2, 3, 4, 7, 8, 9)
		M6x = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10)
		y6_index = 24
		kalibrator = SimMarketKalibrator(config, M123, M123, M123, M4, M5, M6, M4x, M5x, M6x)

		self.kalibrated_market: SimMarketKalibrated = kalibrator.kalibrate_market(
			data, y1_index, y1_index, y1_index, y4_index, y5_index, y6_index)
		super(CircularEconomyRebuyPriceDuopoly, self).__init__(config, support_continuous_action_space)

	def reset(self) -> np.array:
		true_obs = super(CircularEconomyRebuyPriceDuopoly, self).reset()
		pred_obs = self.kalibrated_market.reset_with_foreign_state(true_obs)
		print('reset')
		self.compare_outputs((-1, -1, -1), true_obs, pred_obs)
		return true_obs

	def step(self, action) -> Tuple[np.array, np.float32, bool, dict]:
		step_output_tuple_a = super(CircularEconomyRebuyPriceDuopoly, self).step(action)
		step_output_tuple_b = self.kalibrated_market.step(action)

		self.compare_outputs(action, true_obs=step_output_tuple_a, predicted_obs=step_output_tuple_b)

		return step_output_tuple_a

	def compare_outputs(self, action, true_obs, predicted_obs):
		assert true_obs[2] == predicted_obs[2], f'{true_obs[2]} != {predicted_obs[2]}'
		# self.compare_sales(action, true_obs, predicted_obs)
		self.compare_observations_and_reward(action, true_obs, predicted_obs)

	def compare_sales(self, true_obs, predicted_obs):
		true_sales_used = self._output_dict['customer/purchases_refurbished']['vendor_0']
		true_sales_new = self._output_dict['customer/purchases_new']['vendor_0']
		true_sales_rebuy = self._output_dict['owner/rebuys']['vendor_0']
		output_dict_pred = predicted_obs[3]
		pred_sales_used = output_dict_pred['customer/purchases_refurbished']['vendor_0']
		pred_sales_new = output_dict_pred['customer/purchases_new']['vendor_0']
		pred_sales_rebuy = output_dict_pred['owner/rebuys']['vendor_0']
		true_sales_t = torch.tensor([true_sales_used, true_sales_new, true_sales_rebuy])
		pred_sales_t = torch.tensor([pred_sales_used, pred_sales_new, pred_sales_rebuy])
		print('True sales:', true_sales_t)
		print('Pred sales:', pred_sales_t)
		diff_sales = pred_sales_t - true_sales_t
		print('Diff sales:', diff_sales)

	def compare_observations_and_reward(self, action, true_obs, predicted_obs):
		true_obs_t = torch.tensor(true_obs[0])
		pred_obs_t = torch.tensor(predicted_obs[0])
		diff_obs = pred_obs_t - true_obs_t
		diff_reward = predicted_obs[1] - true_obs[1]
		print(f'diff_obs: {diff_obs}\t diff_reward: {diff_reward}')
