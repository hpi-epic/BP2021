import torch
import torch.nn as nn
from first_prototype import SimMarket

model = nn.Sequential(
	nn.Linear(4, 128),
	nn.ReLU(),
	nn.Linear(128, 128),
	nn.ReLU(),
	nn.Linear(128, 3)).to('cpu')
model.load_state_dict(torch.load(
	"best_marketplace.dat", map_location=torch.device('cpu')))

env = SimMarket()
our_profit = 0
is_done = False
state = env.reset(False)
print("The production price is " + str(env.production_price))
while not is_done:
	action = int(torch.argmax(model(torch.Tensor(state))))
	print("This is the state: " + str(state) + " and this is how I estimate the actions: " +
		  str(model(torch.Tensor(state))) + " so I do " + str(action))
	state, reward, is_done, _ = env.step(action)
	print("The agents profit this round is " + str(reward))
	our_profit += reward
print("In total the agent earned " + str(our_profit) + " with a profit/quality of: " + str(round(our_profit/state[1],3)) + 
      " and his competitor " + str(env.comp_profit) + " with a profit/quality of: " + str(round(env.comp_profit/state[3], 3)))
print("The value of the net is estimated as: " + str(our_profit/(state[3] - state[1] + 100)))
