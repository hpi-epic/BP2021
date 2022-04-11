import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceOneCompetitor


class CircularEconomyDatagenerator(CircularEconomyRebuyPriceOneCompetitor):

	def __init__(self, support_continuous_action_space: bool = False) -> None:
		super.__init__(support_continuous_action_space)
		self.cumulated_states = torch.tensor([np.zeros(25)])
		self.cumulated_states = self.cumulated_states.transpose(0, 1)

	def step(self, action) -> Tuple[np.array, np.float32, bool, dict]:
		step_output_tuple = super.step(action)
		self.output_dict_append(self._output_dict, self.step_counter)
		return step_output_tuple

	def output_dict_append(self, output_dict: dict, i):
		x = torch.tensor(np.zeros(25))

		# agent period: 1st half: agent x[7]vs x[2]| 2nd half: agent x[7] vs x[2, i+1]
		# comp period: 1st half: com x[2, i-1] vs x[7, i-1] | 2nd half: com x[2, i-1] vs x[7]

		x[0] = i
		x[1] = output_dict['state/in_storage']['vendor_0']  # agent inventory

		x[2] = self.cumulated_states[22, i-1]  # comp price new (old)
		x[3] = self.cumulated_states[23, i-1]   # comp price used (old)
		x[4] = self.cumulated_states[24, i-1]   # comp price rebuy (old)

		x[5] = output_dict['state/in_storage']['vendor_1']  # comp inventory

		x[6] = output_dict['state/in_circulation']  # resource in use

		x[7] = output_dict['actions/price_refurbished']['vendor_0']  # agent price new
		x[8] = output_dict['actions/price_new']['vendor_0']  # agent price used
		x[9] = output_dict['actions/price_new']['vendor_0']  # agent price rebuy

		x[10] = output_dict['profits/storage_cost']['vendor_0']  # agent storage cost

		x[22] = output_dict['actions/price_new']['vendor_1']  # comp price new (updated)
		x[23] = output_dict['actions/price_refurbished']['vendor_1']  # comp price used (updated)
		x[24] = output_dict['actions/price_rebuy']['vendor_1']  # comp price rebuy (updated)

		x[11] = output_dict['customer/purchases_new']['vendor_0']  # agent sales new
		x[12] = output_dict['customer/purchases_refurbished']['vendor_0']  # agent sales used
		x[13] = output_dict['owner/rebuys']['vendor_0']  # agent repurchases

		x[14] = output_dict['profits/storage_cost']['vendor_1']  # comp holding cost

		x[15] = output_dict['customer/purchases_new']['vendor_1']  # comp sales new
		x[16] = output_dict['customer/purchases_refurbished']['vendor_1']  # comp sales used
		x[17] = output_dict['owner/rebuys']['vendor_0']  # comp sales rebuy

		x[18] = output_dict['profits/all']['vendor_0']  # agent total reward
		x[19] = output_dict['profits/all']['vendor_1']  # comp total reward

		x[20] = self.cumulated_states[20, i-1] + x[18]  # agent culmulated reward
		x[21] = self.cumulated_states[21, i-1] + x[19]  # comp culmulated reward
		pd.DataFrame(x).to_csv(f'kalibration_data/training_data-{datetime.datetime.now()}.csv', index=False)
		self.cumulated_states.append(x, axis=1)
