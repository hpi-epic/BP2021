from django.contrib.auth.models import User
from django.db import models


class Config(models.Model):
	environment = models.ForeignKey('EnvironmentConfig', on_delete=models.CASCADE, null=True)
	hyperparameter = models.ForeignKey('HyperparameterConfig', on_delete=models.CASCADE, null=True)
	name = models.CharField(max_length=100, editable=False, default='')
	user = models.ForeignKey(User, on_delete=models.CASCADE, null=True,)

	def as_dict(self) -> dict:
		environment_dict = self.environment.as_dict() if self.environment is not None else None
		hyperparameter_dict = self.hyperparameter.as_dict() if self.hyperparameter is not None else None
		return remove_none_values_from_dict({'environment': environment_dict, 'hyperparameter': hyperparameter_dict})

	def is_referenced(self):
		# Query set is empty so we are not referenced by any container
		return bool(self.container_set.all())

	@staticmethod
	def get_empty_structure_dict():
		return {
			'environment': EnvironmentConfig.get_empty_structure_dict(),
			'hyperparameter': HyperparameterConfig.get_empty_structure_dict()
			}


class EnvironmentConfig(models.Model):
	agents = models.ForeignKey('AgentsConfig', on_delete=models.CASCADE, null=True)
	separate_markets = models.BooleanField(null=True)
	episodes = models.IntegerField(null=True)
	plot_interval = models.IntegerField(null=True)
	marketplace = models.CharField(max_length=150, null=True)
	task = models.CharField(max_length=14, choices=((1, 'training'), (2, 'agent_monitoring'), (3, 'exampleprinter')), null=True)

	def as_dict(self) -> dict:
		agents_list = self.agents.as_list() if self.agents is not None else None
		return remove_none_values_from_dict({
			'separate_markets': self.separate_markets,
			'episodes': self.episodes,
			'plot_interval': self.plot_interval,
			'marketplace': self.marketplace,
			'task': self.task,
			'agents': agents_list
		})

	@staticmethod
	def get_empty_structure_dict():
		return {
			'separate_markets': None,
			'episodes': None,
			'plot_interval': None,
			'marketplace': None,
			'task': None,
			'agents': AgentsConfig.get_empty_structure_list()
			}


class AgentsConfig(models.Model):
	def as_list(self) -> dict:
		referencing_agents = self.agentconfig_set.all()
		return [agent.as_dict() for agent in referencing_agents]

	@staticmethod
	def get_empty_structure_list():
		return []


class AgentConfig(models.Model):
	agents_config = models.ForeignKey('AgentsConfig', on_delete=models.CASCADE, null=True)
	name = models.CharField(max_length=100, default='')
	agent_class = models.CharField(max_length=100, null=True)
	argument = models.CharField(max_length=200, default='')

	def as_dict(self) -> dict:
		return remove_none_values_from_dict({
				'name': self.name,
				'agent_class': self.agent_class,
				'argument': self.argument
				})


class HyperparameterConfig(models.Model):
	rl = models.ForeignKey('RLConfig', on_delete=models.CASCADE, null=True)
	sim_market = models.ForeignKey('SimMarketConfig', on_delete=models.CASCADE, null=True)

	def as_dict(self) -> dict:
		sim_market_dict = self.sim_market.as_dict() if self.sim_market is not None else {'sim_market': None}
		rl_dict = self.rl.as_dict() if self.rl is not None else {'rl': None}
		return remove_none_values_from_dict({
			'rl': rl_dict,
			'sim_market': sim_market_dict
		})

	@staticmethod
	def get_empty_structure_dict():
		return {
			'rl': RlConfig.get_empty_structure_dict(),
			'sim_market': SimMarketConfig.get_empty_structure_dict()
		}


class RlConfig(models.Model):
	gamma = models.FloatField(null=True)
	batch_size = models.IntegerField(null=True)
	replay_size = models.IntegerField(null=True)
	learning_rate = models.FloatField(null=True)
	sync_target_frames = models.IntegerField(null=True)
	replay_start_size = models.IntegerField(null=True)
	epsilon_decay_last_frame = models.IntegerField(null=True)
	epsilon_start = models.FloatField(null=True)
	epsilon_final = models.FloatField(null=True)

	def as_dict(self) -> dict:
		return remove_none_values_from_dict({
			'gamma': self.gamma,
			'batch_size': self.batch_size,
			'replay_size': self.replay_size,
			'learning_rate': self.learning_rate,
			'sync_target_frames': self.sync_target_frames,
			'replay_start_size': self.replay_start_size,
			'epsilon_decay_last_frame': self.epsilon_decay_last_frame,
			'epsilon_start': self.epsilon_start,
			'epsilon_final': self.epsilon_final
		})

	@staticmethod
	def get_empty_structure_dict():
		return {
			'gamma': None,
			'batch_size': None,
			'replay_size': None,
			'learning_rate': None,
			'sync_target_frames': None,
			'replay_start_size': None,
			'epsilon_decay_last_frame': None,
			'epsilon_start': None,
			'epsilon_final': None
		}


class SimMarketConfig(models.Model):
	max_storage = models.IntegerField(null=True)
	episode_length = models.IntegerField(null=True)
	max_price = models.IntegerField(null=True)
	max_quality = models.IntegerField(null=True)
	number_of_customers = models.IntegerField(null=True)
	production_price = models.IntegerField(null=True)
	storage_cost_per_product = models.FloatField(null=True)

	def as_dict(self) -> dict:
		return remove_none_values_from_dict({
			'max_storage': self.max_storage,
			'episode_length': self.episode_length,
			'max_price': self.max_price,
			'max_quality': self.max_quality,
			'number_of_customers': self.number_of_customers,
			'production_price': self.production_price,
			'storage_cost_per_product': self.storage_cost_per_product
		})

	@staticmethod
	def get_empty_structure_dict():
		return {
			'max_storage': None,
			'episode_length': None,
			'max_price': None,
			'max_quality': None,
			'number_of_customers': None,
			'production_price': None,
			'storage_cost_per_product': None
		}


def capitalize(word: str) -> str:
	return word.upper() if len(word) <= 1 else word[0].upper() + word[1:]


def to_config_class_name(name: str) -> str:
	return ''.join([capitalize(x) for x in name.split('_')]) + 'Config'


def remove_none_values_from_dict(dict_with_none_values: dict) -> dict:
	return {key: value for key, value in dict_with_none_values.items() if value is not None}
