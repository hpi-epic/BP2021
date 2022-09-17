import numpy as np

import recommerce.market.circular.circular_sim_market as circular_market
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader

# Q-Learning Cube Limits
MAX_WAREHOUSE = 101
MAX_IN_CIRC = 701
MAX_PRICE = 10

# number of episode we will run
n_episodes = 10000
# maximum of iteration per episode
max_iter_episode = 500
# initialize the exploration probability to 1
exploration_proba = 0.4
# exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.00005
# minimum of exploration proba
min_exploration_proba = 0.01
# discounted factor
gamma = 0.999
# learning rate
lr = 0.2

print('we are going to try to find an optimal policy')
config_market = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceMonopoly)
env = circular_market.CircularEconomyRebuyPriceMonopoly(config=config_market)

Q_table = np.zeros((MAX_IN_CIRC, MAX_WAREHOUSE, MAX_PRICE, MAX_PRICE, MAX_PRICE))
Q_table_prev = np.zeros((MAX_IN_CIRC, MAX_WAREHOUSE, MAX_PRICE, MAX_PRICE, MAX_PRICE))
Q_table = np.fromfile('./test.dat').reshape(MAX_IN_CIRC, MAX_WAREHOUSE, MAX_PRICE, MAX_PRICE, MAX_PRICE)
ls = []

rewards_per_episode = []

# we iterate over episodes
for e in range(n_episodes):
    # we initialize the first state of the episode

    # reset returns -> [in_circ, in_warehouse]
    current_state = env.reset()
    done = False

    # sum the rewards that the agent gets from the environment
    total_episode_reward = 0

    for i in range(max_iter_episode):
        if np.random.uniform(0, 1) < exploration_proba:
            action = env.action_space.sample()
        else:
            a = Q_table[int(current_state[0]), int(current_state[1]), :]
            action = np.unravel_index(a.argmax(), a.shape)
        # The environment runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        next_state, reward, done, _ = env.step(action)

        # We update our Q-table using the Q-learning iteration

        Q_table[int(current_state[0]), int(current_state[1]), action[0], action[1], action[2]] =\
            (1 - lr) * Q_table[int(current_state[0]), int(current_state[1]), action[0], action[1], action[2]] + \
            lr * (reward + gamma * np.max(Q_table[int(next_state[0]), int(next_state[1]), :]))

        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state

    diff = np.max(abs(Q_table - Q_table_prev))
    ls.append(diff)
    Q_table_prev = np.copy(Q_table)

    # We update the exploration proba using exponential decay formula
    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))
    rewards_per_episode.append(total_episode_reward)
    if e % 50 == 0:
        print(f'best reward until now (ep {e}):')
        print(f'diff in epsisode: {diff}')
        print(max(rewards_per_episode))
        # plt.plot(ls)
        # plt.show()

with open('output.txt', 'w') as f:
    print(rewards_per_episode, file=f)

Q_table.tofile('test2.dat')
