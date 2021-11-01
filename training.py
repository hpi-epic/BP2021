#!/usr/bin/env python3
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from first_prototype import SimMarket


def model(device):
	return nn.Sequential(
		nn.Linear(4, 128),
		nn.ReLU(),
		nn.Linear(128, 128),
		nn.ReLU(),
		nn.Linear(128, 3)).to(device)
	# return nn.Sequential(
	#     nn.Linear(4, 512),
	#     nn.ReLU(),
	#     nn.Linear(512, 512),
	#     nn.ReLU(),
	#     nn.Linear(512, 512),
	#     nn.ReLU(),
	#     nn.Linear(512, 512),
	#     nn.ReLU(),
	#     nn.Linear(512, 3)).to(device)


MEAN_REWARD_BOUND = 50 * 100 * 20

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50000
LEARNING_RATE = 1e-5
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 75000
EPSILON_START = 1.0
EPSILON_FINAL = 0.1


Experience = collections.namedtuple(
	'Experience', field_names=['state', 'action', 'reward',
							   'done', 'new_state'])


class ExperienceBuffer:
	def __init__(self, capacity):
		self.buffer = collections.deque(maxlen=capacity)

	def __len__(self):
		return len(self.buffer)

	def append(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		indices = np.random.choice(len(self.buffer), batch_size,
								   replace=False)
		states, actions, rewards, dones, next_states = \
			zip(*[self.buffer[idx] for idx in indices])
		return np.array(states), np.array(actions, dtype=np.int64), \
			np.array(rewards, dtype=np.float32), \
			np.array(dones, dtype=np.uint8), \
			np.array(next_states)


class Agent:
	def __init__(self, env, exp_buffer):
		self.env = env
		self.exp_buffer = exp_buffer
		self._reset()

	def _reset(self):
		self.state = env.reset(False)
		self.total_reward = 0.0

	@torch.no_grad()
	def play_step(self, net, epsilon=0.0, device="cpu"):
		done_reward = None
		compet_reward = None

		if np.random.random() < epsilon:
			action = env.action_space.sample()
		else:
			state_a = np.single([self.state])
			state_v = torch.tensor(state_a).to(device)
			q_vals_v = net(state_v)
			_, act_v = torch.max(q_vals_v, dim=1)
			action = int(act_v.item())

		# do step in the environment
		new_state, reward, is_done, _ = self.env.step(action)
		self.total_reward += reward

		exp = Experience(self.state, action, reward,
						 is_done, new_state)
		self.exp_buffer.append(exp)
		self.state = new_state
		if is_done:
			done_reward = self.total_reward
			compet_reward = env.comp_profit
			self._reset()
		return done_reward, compet_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
	states, actions, rewards, dones, next_states = batch

	states_v = torch.tensor(np.single(
		states)).to(device)
	next_states_v = torch.tensor(np.single(
		next_states)).to(device)
	actions_v = torch.tensor(actions).to(device)
	rewards_v = torch.tensor(rewards).to(device)
	done_mask = torch.BoolTensor(dones).to(device)

	state_action_values = net(states_v).gather(
		1, actions_v.unsqueeze(-1)).squeeze(-1)

	with torch.no_grad():
		next_state_values = tgt_net(next_states_v).max(1)[0]
		next_state_values[done_mask] = 0.0
		next_state_values = next_state_values.detach()

	expected_state_action_values = next_state_values * GAMMA + rewards_v
	return nn.MSELoss()(state_action_values, expected_state_action_values)


device = torch.device(
	"cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using {} device".format(device))

env = SimMarket()

observations = env.observation_space.shape[0]

print("Observation Space: " + str(observations) + str(type(observations)))
print("Action Space: " + str(env.action_space.n))

net = model(device)
tgt_net = model(device)

print(net)

buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
epsilon = EPSILON_START

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
total_rewards = []
compet_rewards = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_m_reward = None

while True:
	frame_idx += 1
	epsilon = max(EPSILON_FINAL, EPSILON_START -
				  frame_idx / EPSILON_DECAY_LAST_FRAME)

	reward, compet_reward = agent.play_step(net, epsilon, device=device)
	if reward is not None:
		print("My profit is " + str(reward) + ", my competitor has " + str(reward) +
			  ". The quality values were " + str(env.state[1]) + " and " + str(env.state[3]))
		total_rewards.append(reward)
		compet_rewards.append(compet_reward)
		speed = (frame_idx - ts_frame) / \
			((time.time() - ts) if (time.time() - ts) > 0 else 1)
		ts_frame = frame_idx
		ts = time.time()
		m_reward = np.mean(total_rewards[-100:])
		m_compet_reward = np.mean(compet_rewards[-100:])
		print("%d: done %d games, reward %.3f, compet reward %.3f "
			  "eps %.2f, speed %.2f f/s" % (
				  frame_idx, len(total_rewards), m_reward, m_compet_reward, epsilon,
				  speed
			  ))
		if (best_m_reward is None or best_m_reward < m_reward) and frame_idx > 1.2*EPSILON_DECAY_LAST_FRAME:
			torch.save(net.state_dict(), "args.env" +
					   "-best_%.2f_marketplace.dat" % m_reward)
			if best_m_reward is not None:
				print("Best reward updated %.3f -> %.3f" % (
					best_m_reward, m_reward))
			best_m_reward = m_reward
		if m_reward > MEAN_REWARD_BOUND:
			print("Solved in %d frames!" % frame_idx)
			break

	if len(buffer) < REPLAY_START_SIZE:
		continue

	if frame_idx % SYNC_TARGET_FRAMES == 0:
		tgt_net.load_state_dict(net.state_dict())

	optimizer.zero_grad()
	batch = buffer.sample(BATCH_SIZE)
	loss_t = calc_loss(batch, net, tgt_net, device=device)
	loss_t.backward()
	optimizer.step()
