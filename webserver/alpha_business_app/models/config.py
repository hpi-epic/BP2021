from django.db import models


class Config(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	environment = models.ForeignKey('EnvironmentConfig', on_delete=models.DO_NOTHING, null=True)
	hyperparameter = models.ForeignKey('HyperparameterConfig', on_delete=models.DO_NOTHING, null=True)


class EnvironmentConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	task = models.CharField(max_length=14, choices=((1, 'training'), (2, 'monitoring'), (3, 'exampleprinter')), default='training')
	marketplace = models.CharField(max_length=100, default='market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario')
	agent = models.ForeignKey('AgentsConfig', on_delete=models.DO_NOTHING, null=True)


class AgentsConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	name = models.CharField(max_length=50, null=True)
	agent_class = models.CharField(max_length=100, null=True)
	modelfile = models.ForeignKey('ModelFilesConfig', on_delete=models.DO_NOTHING, null=True)


class ModelFilesConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	name = models.CharField(max_length=50, null=True)
	path_to_file = models.CharField(max_length=100, null=True)


class HyperparameterConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	rl = models.ForeignKey('RLConfig', on_delete=models.DO_NOTHING)
	sim_market = models.ForeignKey('SimMarketConfig', on_delete=models.DO_NOTHING)


class RlConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	gamma = models.FloatField(default=0.99, null=True)
	batch_size = models.IntegerField(default=32, null=True)
	replay_size = models.IntegerField(default=100000, null=True)
	learning_rate = models.FloatField(default=1e-6, null=True)
	sync_target_frames = models.IntegerField(default=1000, null=True)
	replay_start_size = models.IntegerField(default=10000, null=True)
	epsilon_decay_last_frame = models.IntegerField(default=75000, null=True)
	epsilon_start = models.FloatField(default=1.0, null=True)
	epsilon_final = models.FloatField(default=0.1, null=True)


class SimMarketConfig(models.Model):
	# id = models.CharField(max_length=50, primary_key=True)
	max_storage = models.IntegerField(default=100, null=True)
	episode_size = models.IntegerField(default=50, null=True)
	max_price = models.IntegerField(default=10, null=True)
	max_quality = models.IntegerField(default=50, null=True)
	number_of_customers = models.IntegerField(default=20, null=True)
	production_price = models.IntegerField(default=3, null=True)
	storage_cost_per_product = models.FloatField(default=0.1, null=True)
