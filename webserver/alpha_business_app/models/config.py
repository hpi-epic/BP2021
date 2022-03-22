from django.db import models


class Config(models.Model):
	environment = models.ForeignKey('EnvironmentConfig', on_delete=models.DO_NOTHING, null=True)
	hyperparameter = models.ForeignKey('HyperparameterConfig', on_delete=models.DO_NOTHING, null=True)
	name = models.CharField(max_length=100, editable=False, default='')

	def as_dict(self) -> dict:
		environment_dict = self.environment.as_dict() if self.environment is not None else {'environment': None}
		hyperparameter_dict = self.hyperparameter.as_dict() if self.hyperparameter is not None else {'hyperparameter': None}
		return remove_none_values_from_dict({'environment': environment_dict, 'hyperparameter': hyperparameter_dict})


class EnvironmentConfig(models.Model):
	agents = models.ForeignKey('AgentsConfig', on_delete=models.DO_NOTHING, null=True)
	enable_live_draw = models.BooleanField(null=True)
	episodes = models.IntegerField(null=True)
	plot_interval = models.IntegerField(null=True)
	marketplace = models.CharField(max_length=100, null=True)
	task = models.CharField(max_length=14, choices=((1, 'training'), (2, 'monitoring'), (3, 'exampleprinter')), null=True)

	def as_dict(self) -> dict:
		agents_dict = self.agents.as_dict() if self.agents is not None else None
		return remove_none_values_from_dict({
			'enable_live_draw': self.enable_live_draw,
			'episodes': self.episodes,
			'plot_interval': self.plot_interval,
			'marketplace': self.marketplace,
			'task': self.task,
			'agents': agents_dict
		})


class AgentsConfig(models.Model):
	def as_dict(self) -> dict:
		rulebased = self.rulebasedagentconfig_set.all().first()
		cerebuy = self.cerebuyagentqlearningconfig_set.all().first()
		# at the moment, this can be at most one each
		rulebased_dict = rulebased.as_dict() if rulebased is not None else None
		cerbuy_dict = cerebuy.as_dict() if cerebuy is not None else None
		return remove_none_values_from_dict({
			'CE Rebuy Agent (QLearning)': cerbuy_dict,
			'Rule_Based Agent': rulebased_dict
		})


class RuleBasedAgentConfig(models.Model):
	agents_config = models.ForeignKey('AgentsConfig', on_delete=models.DO_NOTHING, null=True)
	agent_class = models.CharField(max_length=100, null=True)
	argument = models.CharField(max_length=200, null=True)

	def as_dict(self) -> dict:
		return remove_none_values_from_dict({
			'agent_class': self.agent_class,
			'argument': self.argument
		})


class CERebuyAgentQLearningConfig(models.Model):
	agents_config = models.ForeignKey('AgentsConfig', on_delete=models.DO_NOTHING, null=True)
	agent_class = models.CharField(max_length=100, null=True)
	argument = models.CharField(max_length=200, null=True)

	def as_dict(self) -> dict:
		return remove_none_values_from_dict({
			'agent_class': self.agent_class,
			'argument': self.argument
		})


class HyperparameterConfig(models.Model):
	rl = models.ForeignKey('RLConfig', on_delete=models.DO_NOTHING, null=True)
	sim_market = models.ForeignKey('SimMarketConfig', on_delete=models.DO_NOTHING, null=True)

	def as_dict(self) -> dict:
		sim_market_dict = self.sim_market.as_dict() if self.sim_market is not None else {'sim_market': None}
		rl_dict = self.rl.as_dict() if self.rl is not None else {'rl': None}
		return remove_none_values_from_dict({
			'rl': rl_dict,
			'sim_market': sim_market_dict
		})


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


def get_config_field_names(model):
	ret = []
	for f in model._meta.fields:
		ret += [f.name]
	# the id is auto generated and we do not use it for our config.
	if 'id' in ret:
		ret.remove('id')
	return ret


def capitalize(word: str) -> str:
	return word.upper() if len(word) <= 1 else word[0].upper() + word[1:]


def to_config_class_name(name: str) -> str:
	# replace all brackets
	class_name = name.replace('(', '').replace(')', '')
	# remove all whitespaces:
	class_name = ''.join([capitalize(x) for x in class_name.split(' ')])
	return ''.join([capitalize(x) for x in class_name.split('_')]) + 'Config'


def remove_none_values_from_dict(dict_with_none_values: dict) -> dict:
	return {k: v for k, v in dict_with_none_values.items() if v is not None}
