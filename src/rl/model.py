from torch import nn


def simple_network(input_size, output_size) -> nn.modules.container.Sequential:
	"""
	A network with one small hidden layer and user-defined input and output sizes.

	Args:
		input_size (int): Number of input nodes.
		output_size (int): Number of output nodes.

	Returns:
		nn.modules.container.Sequential: The network to use for the model.
	"""
	return nn.Sequential(
		nn.Linear(input_size, 128),
		nn.ReLU(),
		nn.Linear(128, 128),
		nn.ReLU(),
		nn.Linear(128, output_size),
	)


def medium_network(input_size, output_size) -> nn.modules.container.Sequential:
	"""A network with two medium sized hidden layers and user-defined input and output sizes.

	Args:
		input_size (int): Number of input nodes.
		output_size (int): Number of output nodes.

	Returns:
		nn.modules.container.Sequential: The network to use for the model.
	"""
	return nn.Sequential(
		nn.Linear(input_size, 2048),
		nn.ReLU(),
		nn.Linear(2048, 2048),
		nn.ReLU(),
		nn.Linear(2048, 2048),
		nn.ReLU(),
		nn.Linear(2048, output_size),
	)
