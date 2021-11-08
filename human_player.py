from first_prototype import SimMarket

env = SimMarket()
our_profit = 0
is_done = False
state = env.reset()

print('The production price is', str(env.production_price))
while not is_done:
    agent_price = state[0]
    comp_price = state[2]
    agent_quality = state[1]
    comp_quality = state[3]
    print('agent_price:', agent_price, 'agent_quality', agent_quality, 
        'comp_price:', comp_price, 'comp_quality', comp_quality)
    action = input('What do you want to do? ')
    if action == '':
        action = agent_price
    state, reward, is_done, _ = env.step(int(action))
    print('Your profit this round is', reward)
    our_profit += reward
print('Your total earnings:', str(our_profit), 
    'total comp earnings:', env.comp_profit)