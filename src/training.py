#!/usr/bin/env python3

# helper
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import model
import sim_market as sim
import utils as ut
from agent import Agent
from experience_buffer import ExperienceBuffer


def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.single(states)).to(device)
    next_states_v = torch.tensor(np.single(next_states)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * ut.GAMMA + rewards_v
    return (
        nn.MSELoss()(state_action_values, expected_state_action_values),
        state_action_values.mean(),
    )


def write_tensorboard_profits(profits):
    mydict = {}
    n_vendors = len(profits[0])
    for i in range(n_vendors):
        last = profits[-100:]
        # print(last)
        matrix = np.concatenate(last).reshape(-1, n_vendors)
        # print(matrix)
        mydict['vendor_' + str(i)] = np.mean(matrix[:, i])
    return mydict


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using {} device'.format(device))

env = sim.CircularEconomy()

observations = env.observation_space.shape[0]

print('Observation Space:', observations, type(observations))
print('Action Space: ', env.action_space.n)

net = model.simple_network(observations, env.action_space.n).to(device)
tgt_net = model.simple_network(observations, env.action_space.n).to(device)

print(net)

buffer = ExperienceBuffer(ut.REPLAY_SIZE)
agent = Agent(env, buffer)
epsilon = ut.EPSILON_START

optimizer = optim.Adam(net.parameters(), lr=ut.LEARNING_RATE)
total_rewards = []
all_rewards = []
losses = []
rmse_losses = []
selected_q_vals = []
loss_val_ratio = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_m_reward = None

# tensorboard init
writer = SummaryWriter()

while True:
    frame_idx += 1  # counts the steps
    epsilon = max(
        ut.EPSILON_FINAL, ut.EPSILON_START - frame_idx / ut.EPSILON_DECAY_LAST_FRAME
    )

    reward, all_profits = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        print(
            'My profit is:',
            reward,
            '\t my competitor has',
            all_profits[1] if len(all_profits) > 1 else 0,
            '\tThe quality values were',
            env.state[0],
            '\tand',
            env.state[2] if len(all_profits) > 1 else 0,
        )
        total_rewards.append(reward)
        all_rewards.append(all_profits)
        speed = (frame_idx - ts_frame) / (
            (time.time() - ts) if (time.time() - ts) > 0 else 1
        )
        ts_frame = frame_idx
        ts = time.time()
        m_reward = np.mean(total_rewards[-100:])
        writer.add_scalar('Profit_mean/agent', m_reward, frame_idx / ut.EPISODE_LENGTH)
        writer.add_scalars(
            'Profit_mean/direct_comparison',
            write_tensorboard_profits(all_rewards),
            frame_idx / ut.EPISODE_LENGTH,
        )
        if frame_idx > ut.REPLAY_START_SIZE:
            writer.add_scalar(
                'Loss/MSE', np.mean(losses[-1000:]), frame_idx / ut.EPISODE_LENGTH
            )
            writer.add_scalar(
                'Loss/RMSE', np.mean(rmse_losses[-1000:]), frame_idx / ut.EPISODE_LENGTH
            )
            writer.add_scalar(
                'Loss/selected_q_vals',
                np.mean(selected_q_vals[-1000:]),
                frame_idx / ut.EPISODE_LENGTH,
            )
        writer.add_scalar('epsilon', epsilon, frame_idx / ut.EPISODE_LENGTH)
        print(
            '%d: done %d games, reward %.3f, comp reward %.3f '
            'eps %.2f, speed %.2f f/s'
            % (frame_idx, len(total_rewards), m_reward, 0, epsilon, speed)
        )

        if not os.path.isdir('trainedModels'):
            os.mkdir('trainedModels')

        if (
            best_m_reward is None or best_m_reward < m_reward
        ) and frame_idx > ut.EPSILON_DECAY_LAST_FRAME + 101:
            torch.save(
                net.state_dict(),
                './trainedModels/'
                + 'args.env'
                + '-best_%.2f_marketplace.dat' % m_reward,
            )
            if best_m_reward is not None:
                print('Best reward updated %.3f -> %.3f' % (best_m_reward, m_reward))
            best_m_reward = m_reward
        if m_reward > ut.MEAN_REWARD_BOUND or frame_idx >= ut.EPSILON_DECAY_LAST_FRAME * 4:
            print('Solved in %d frames!' % frame_idx)
            break

    if len(buffer) < ut.REPLAY_START_SIZE:
        continue

    if frame_idx % ut.SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())

    optimizer.zero_grad()
    batch = buffer.sample(ut.BATCH_SIZE)
    loss_t, selected_q_val_mean = calc_loss(batch, net, tgt_net, device=device)
    losses.append(loss_t.item())
    rmse_losses.append(torch.sqrt(loss_t).item())
    selected_q_vals.append(selected_q_val_mean.item())
    loss_t.backward()
    optimizer.step()
