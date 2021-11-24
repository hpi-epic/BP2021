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

config = {}


def load_config(path_rl=os.path.dirname(__file__) + os.sep + '..' + os.sep + 'config_rl.json'):
    with open(path_rl) as config_file:
        return json.load(config_file)


config = load_config()

# ordered alphabetically in the config_rl.json
assert 'learning_rate' in config, 'your config is missing learning_rate'
assert 'gamma' in config, 'your config is missing gamma'
assert 'batch_size' in config, 'your config is missing batch_size'
assert 'replay_size' in config, 'your config is missing replay_size'
assert 'sync_target_frames' in config, 'your config is missing sync_target_frames'
assert 'replay_start_size' in config, 'your config is missing replay_start_size'
assert 'epsilon_decay_last_frame' in config, 'your config is missing epsilon_decay_last_frame'
assert 'epsilon_start' in config, 'your config is missing epsilon_start'
assert 'epsilon_final' in config, 'your config is missing epsilon_final'


GAMMA = float(config['gamma'])
LEARNING_RATE = float(config['learning_rate'])
BATCH_SIZE = int(config['batch_size'])
REPLAY_SIZE = int(config['replay_size'])
SYNC_TARGET_FRAMES = int(config['sync_target_frames'])
REPLAY_START_SIZE = int(config['replay_start_size'])

EPSILON_DECAY_LAST_FRAME = int(config['epsilon_decay_last_frame'])
EPSILON_START = float(config['epsilon_start'])
EPSILON_FINAL = float(config['epsilon_final'])

assert LEARNING_RATE > 0 and LEARNING_RATE < 1, 'learning_rate should be between 0 and 1 (excluded)'
assert GAMMA >= 0 and GAMMA < 1, 'gamma should be between 0 (included) and 1 (excluded)'
assert BATCH_SIZE > 0, 'batch_size should be greater than 0'
assert REPLAY_SIZE > 0, 'replay_size should be greater than 0'
assert SYNC_TARGET_FRAMES > 0, 'sync_target_frames should be greater than 0'
assert REPLAY_START_SIZE > 0, 'replay_start_size should be greater than 0'
assert EPSILON_DECAY_LAST_FRAME >= 0, 'epsilon_decay_last_frame should not be negative'
