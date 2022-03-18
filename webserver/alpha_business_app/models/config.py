from django.db import models


class Config(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	environment = models.ForeignKey('EnvironmentConfig', on_delete=models.DO_NOTHING, null=True)
	hyperparameter = models.ForeignKey('HyperparameterConfig', on_delete=models.DO_NOTHING, null=True)


class EnvironmentConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	agent = models.ForeignKey('AgentsConfig', on_delete=models.DO_NOTHING, null=True)
	enable_live_draw = models.BooleanField(null=True)
	episodes = models.IntegerField(null=True)
	plot_interval = models.IntegerField(null=True)
	marketplace = models.CharField(max_length=100, null=True)
	task = models.CharField(max_length=14, choices=((1, 'training'), (2, 'monitoring'), (3, 'exampleprinter')), null=True)


class AgentsConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	name = models.CharField(max_length=50, null=True)
	agent_class = models.CharField(max_length=100, null=True)
	modelfile = models.CharField(max_length=200, null=True)


class HyperparameterConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	rl = models.ForeignKey('RLConfig', on_delete=models.DO_NOTHING, null=True)
	sim_market = models.ForeignKey('SimMarketConfig', on_delete=models.DO_NOTHING, null=True)


class RlConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	gamma = models.FloatField(null=True)
	batch_size = models.IntegerField(null=True)
	replay_size = models.IntegerField(null=True)
	learning_rate = models.FloatField(null=True)
	sync_target_frames = models.IntegerField(null=True)
	replay_start_size = models.IntegerField(null=True)
	epsilon_decay_last_frame = models.IntegerField(null=True)
	epsilon_start = models.FloatField(null=True)
	epsilon_final = models.FloatField(null=True)


class SimMarketConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	max_storage = models.IntegerField(null=True)
	episode_size = models.IntegerField(null=True)
	max_price = models.IntegerField(null=True)
	max_quality = models.IntegerField(null=True)
	number_of_customers = models.IntegerField(null=True)
	production_price = models.IntegerField(null=True)
	storage_cost_per_product = models.FloatField(null=True)


def get_config_field_names(model):
	ret = []
	for f in model._meta.fields:
		ret += [f.name]
	# the id is auto generated and we do not use it for our config.
	if 'id' in ret:
		ret.remove('id')
	return ret
