#!/usr/bin/env python3

# helper
import math
import time

import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import agent2
import sim_market as sim
import utils as ut

# def write_tensorboard_profits(profits):
#     mydict = {}
#     n_vendors = len(profits[0])
#     for i in range(n_vendors):
#         last = profits[-100:]
#         matrix = np.concatenate(last).reshape(-1, n_vendors)
#         mydict['vendor_' + str(i)] = np.mean(matrix[:, i])
#     return mydict


env = sim.CircularEconomy()
state = env.reset()
agent = agent2.QLearningAgent(env.observation_space.shape[0], env.action_space.n, optim.Adam)

all_agent_returns = []
# all_vendors_reward = []
losses = []
rmse_losses = []
selected_q_vals = []
loss_val_ratio = []
ts_frame = 0
ts = time.time()
best_m_reward = None
episode_return = 0

# tensorboard init
writer = SummaryWriter()
for frame_idx in range(ut.EPSILON_DECAY_LAST_FRAME * 4):
    epsilon = max(
        ut.EPSILON_FINAL, ut.EPSILON_START - frame_idx / ut.EPSILON_DECAY_LAST_FRAME
    )

    action = agent.policy(state, epsilon)
    state, reward, is_done, info = env.step(action)
    episode_return += reward
    agent.give_feedback(reward, is_done, state)
    print(type(reward))

    if is_done:
        all_agent_returns.append(episode_return)
        # all_vendors_reward.append(info["all_profits"])
        speed = (frame_idx - ts_frame) / (
            (time.time() - ts) if (time.time() - ts) > 0 else 1
        )
        ts_frame = frame_idx
        ts = time.time()
        m_reward = np.mean(all_agent_returns[-100:])
        writer.add_scalar('Profit_mean/agent', m_reward, frame_idx / ut.EPISODE_LENGTH)
        # writer.add_scalars(
        #     'Profit_mean/direct_comparison',
        #     write_tensorboard_profits(all_vendors_reward),
        #     frame_idx / ut.EPISODE_LENGTH,
        # )
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
        print('%d: done %d games, this episode return %.3f, mean return %.3f, eps %.2f, speed %.2f f/s' % (frame_idx, len(all_agent_returns), episode_return, m_reward, epsilon, speed))

        if (
            best_m_reward is None or best_m_reward < m_reward
        ) and frame_idx > ut.EPSILON_DECAY_LAST_FRAME + 101:
            agent.save('args.env-best_%.2f_marketplace.dat' % m_reward)
            if best_m_reward is not None:
                print('Best reward updated %.3f -> %.3f' % (best_m_reward, m_reward))
            best_m_reward = m_reward
        if m_reward > ut.MEAN_REWARD_BOUND:
            print('Solved in %d frames!' % frame_idx)
            break

        episode_return = 0
        env.reset()

    if len(agent.buffer) < ut.REPLAY_START_SIZE:
        continue

    if frame_idx % ut.SYNC_TARGET_FRAMES == 0:
        agent.synchronize_tgt_net()

    loss, selected_q_val_mean = agent.train_batch()
    losses.append(loss)
    rmse_losses.append(math.sqrt(loss))
    selected_q_vals.append(selected_q_val_mean)
