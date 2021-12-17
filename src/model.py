from torch import nn


def simple_network(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_size),
    )


def medium_network(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, output_size),
    )
