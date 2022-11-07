import math
import random
from abc import ABC

from attrdict import AttrDict

from recommerce.market.vendors import Agent, FixedPriceAgent, HumanPlayer, RuleBasedAgent


class LinearAgent(Agent, ABC):
    pass


class HumanPlayerLE(LinearAgent, HumanPlayer):
    def __init__(self, name=''):
        self.name = name if name != '' else 'YOU - Linear'
        print('Welcome to this funny game! Now, you are the one playing the game!')

    def policy(self, observation, *_) -> int:
        print('The observation is', observation, 'and you have to decide what to do! Please enter your actions, seperated by spaces!')
        return input()


class FixedPriceLEAgent(LinearAgent, FixedPriceAgent):
    def __init__(self, config_market: AttrDict, fixed_price: int = None, name=''):

        if fixed_price is None:
            fixed_price = config_market.production_price + 3

        assert isinstance(fixed_price, int), f'the fixed_price must be an integer: {fixed_price} ({type(fixed_price)})'
        self.name = name if name != '' else type(self).__name__
        self.fixed_price = fixed_price

    def policy(self, *_) -> int:
        return self.fixed_price


class LinearRatio1LEAgent(LinearAgent, RuleBasedAgent):
    def policy(self, state, epsilon=0):
        # this stratgy calculates the value per money for each competing vendor and tries to adapt to it
        ratios = []
        # we skip the first ratio, the ratio of the competitor itself
        max_competing_ratio = 0
        state = state[1:]
        for i in range(math.floor(len(state) / 2)):
            quality_opponent = state[2 * i + 2]
            price_opponent = state[2 * i + 1] + 1
            ratio = quality_opponent / price_opponent  # value for price for vendor i
            ratios.append(ratio)
            if ratio > max_competing_ratio:
                max_competing_ratio = ratio

        intended = math.floor(1 / max_competing_ratio * state[0]) - 1
        return min(max(self.config_market.production_price + 1, intended), self.config_market.max_price - 1)  # actual price


class LERandomAgent(LinearAgent, RuleBasedAgent):
    def policy(self, state, epsilon=0):
        return random.randint(self.config_market.production_price + 1, self.config_market.max_price - 1)


class Just2PlayersLEAgent(LinearAgent, RuleBasedAgent):
    def policy(self, state, epsilon=0) -> int:
        """
        This competitor is based on quality and agents actions.

        While he can act in every linear economy, you should not expect good performance in a multicompetitor setting.

        Args:
            state (np.array): The state of the marketplace the agent sells its products at.
            epsilon (int, optional): Not used it this method. Defaults to 0.

        Returns:
            int: The price of the product he sells in the next round.
        """
        # assert len(state) == 4, "You can't use this competitor in this market!"
        agent_price = state[1]
        agent_quality = state[2]
        comp_quality = state[0]

        new_price = 0

        if comp_quality > agent_quality + 15:
            # significantly better quality
            new_price = agent_price + 2
        elif comp_quality > agent_quality:
            # slightly better quality
            new_price = agent_price + 1
        elif comp_quality < agent_quality and comp_quality > agent_quality - 15:
            # slightly worse quality
            new_price = agent_price - 1
        elif comp_quality < agent_quality:
            # significantly worse quality
            new_price = agent_price - 2
        elif comp_quality == agent_quality:
            # same quality
            new_price = agent_price
        if new_price <= self.config_market.production_price:
            new_price = self.config_market.production_price + 1
        elif new_price >= self.config_market.max_price:
            new_price = self.config_market.max_price - 1
        new_price = int(new_price)
        assert isinstance(new_price, int), 'new_price must be an int'
        return new_price
