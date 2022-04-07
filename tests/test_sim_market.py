import pytest
import utils_tests as ut_t

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market

unique_output_dict_testcases = [
	linear_market.ClassicScenario,
	linear_market.MultiCompetitorScenario,
	circular_market.CircularEconomyMonopolyScenario,
	circular_market.CircularEconomyRebuyPriceMonopolyScenario,
	circular_market.CircularEconomyRebuyPriceOneCompetitor
]


@pytest.mark.parametrize('marketclass', unique_output_dict_testcases)
def test_unique_output_dict(marketclass):
	market = marketclass()
	_, _, _, info_dict_1 = market.step(ut_t.create_mock_action(marketclass))
	_, _, _, info_dict_2 = market.step(ut_t.create_mock_action(marketclass))
	assert id(info_dict_1) != id(info_dict_2)
