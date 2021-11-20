import numpy as np
import sim_market as sim
import agent
import utils as ut

situation = 'linear'

marketplace = sim.CircularEconomy() if situation == 'circular' else sim.ClassicScenario()
agent = agent.QLearningAgent(marketplace.observation_space.shape[0], marketplace.action_space.n, load_path='/Users/lion/Documents/HPI5/EPIC/BP2021/trainedModels/args.env-best_2760.77_marketplace.dat.dat')

counter = 0
our_profit = 0
number_episodes = 10000
is_done = False
profits_per_episode = []

def metrics_average(profits, number_episodes) -> np.float64:
	return np.float64(sum(profits) / number_episodes)

state = marketplace.reset()
print('The production price is', ut.PRODUCTION_PRICE)

for episode in range(number_episodes):
	while not is_done:
		action = agent.policy(state)
		print(
			'This is the state:',
			marketplace.state,
			' and I will do ',
			action
		)
		state, reward, is_done, dict = marketplace.step(action)
		print('The agents profit this round is', reward)
		our_profit += reward
		counter += 1
	print(
		'The total profit of the agent per episode is',
		our_profit
	)
	profits_per_episode.append(our_profit)
	our_profit = 0
	is_done = False

# for number_episode, profit in enumerate(profits_per_episode):
	# print(f"In episode {number_episode} our profit is: " + str(profit))

print(f"The average reward over {number_episodes} episodes is: " + str(metrics_average(profits_per_episode, number_episodes)))