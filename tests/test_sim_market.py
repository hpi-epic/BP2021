import pytest
import utils_tests as ut_t

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.configuration.hyperparameter_config import HyperparameterConfig

config_hyperparameter: HyperparameterConfig = ut_t.mock_config_hyperparameter()

unique_output_dict_testcases = [
	linear_market.LinearEconomyDuopoly,
	linear_market.LinearEconomyOligopoly,
	circular_market.CircularEconomyMonopoly,
	circular_market.CircularEconomyRebuyPriceMonopoly,
	circular_market.CircularEconomyRebuyPriceDuopoly
]


@pytest.mark.parametrize('marketclass', unique_output_dict_testcases)
def test_unique_output_dict(marketclass):
	market = marketclass(config=config_hyperparameter)
	_, _, _, info_dict_1 = market.step(ut_t.create_mock_action(marketclass))
	_, _, _, info_dict_2 = market.step(ut_t.create_mock_action(marketclass))
	assert id(info_dict_1) != id(info_dict_2)
