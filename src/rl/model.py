from torch import nn


def very_simple_network(input_size, output_size) -> nn.modules.container.Sequential:
	"""A network with one hidden layer (64 neurons) and user-defined input and output sizes.

	Args:
		input_size (int): Number of input nodes.
		output_size (int): Number of output nodes.

	Returns:
		nn.modules.container.Sequential: The network to use for the model.
	"""
	return nn.Sequential(
		nn.Linear(input_size, 64),
		nn.ReLU(),
		nn.Linear(64, output_size),
	)


def simple_network(input_size, output_size) -> nn.modules.container.Sequential:
	"""
  A network with two small hidden layers (128 neurons) and user-defined input and output sizes.

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
	"""A network with three medium sized hidden layers (512 neurons) and user-defined input and output sizes.

	Args:
		input_size (int): Number of input nodes.
		output_size (int): Number of output nodes.

	Returns:
		nn.modules.container.Sequential: The network to use for the model.
	"""
	return nn.Sequential(
		nn.Linear(input_size, 512),
		nn.ReLU(),
		nn.Linear(512, 512),
		nn.ReLU(),
		nn.Linear(512, 512),
		nn.ReLU(),
		nn.Linear(512, output_size),
	)
