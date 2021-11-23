#!/usr/bin/env python3

# helper
import json
import os

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 200000
LEARNING_RATE = None
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 75000
EPSILON_START = 1.0
EPSILON_FINAL = 0.1

config_rl = {}


def load_config_rl(path_rl=os.path.dirname(__file__) + os.sep + '..' + os.sep + 'config_rl.json'):
    with open(path_rl) as config_file:
        return json.load(config_file)


config_rl = load_config_rl()

# ordered alphabetically in the config_rl.json
assert 'learning_rate' in config_rl, 'your config is missing learning_rate'

LEARNING_RATE = float(config_rl['learning_rate'])
GAMMA = float(config_rl['gamma'])
BATCH_SIZE = int(config_rl['batch_size'])
REPLAY_SIZE = int(config_rl['replay_size'])
SYNC_TARGET_FRAMES = int(config_rl['sync_target_frames'])
REPLAY_START_SIZE = int(config_rl['replay_start_size'])

EPSILON_DECAY_LAST_FRAME = int(config_rl['epsilon_decay_last_frame'])
EPSILON_START = float(config_rl['epsilon_start'])
EPSILON_FINAL = float(config_rl['epsilon_final'])

assert LEARNING_RATE > 0 and LEARNING_RATE < 1, 'learning_rate should be between 0 and 1 (excluded)'
