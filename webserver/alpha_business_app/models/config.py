from django.db import models


class Config(models.Model):
	id = models.CharField(max_length=50, primary_key=True)
	environment = models.ForeignKey('EnvironmentConfig', on_delete=models.DO_NOTHING)
	hyperparameter = models.ForeignKey('HyperparameterConfig', on_delete=models.DO_NOTHING)


class EnvironmentConfig(models.Model):
	id = models.CharField(max_length=50, primary_key=True)
	task = models.CharField(max_length=14, choices=('training', 'monitoring', 'exampleprinter'), default='training')
	marketplace = models.CharField(max_length=100, default='market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario')
	agent = models.ForeignKey('AgentsConfig', on_delete=models.DO_NOTHING)


class AgentsConfig(models.Model):
	id = models.CharField(max_length=50, primary_key=True)
	name = models.CharField(max_length=50)
	agent_class = models.CharField(max_length=100)
	modelfile = models.ForeignKey('ModelFilesConfig', on_delete=models.DO_NOTHING)


class ModelFilesConfig(models.Model):
	id = models.CharField(max_length=50, primary_key=True)
	name = models.CharField(max_length=50)
	path_to_file = models.CharField(max_length=100)


class HyperparameterConfig(models.Model):
	id = models.CharField(max_length=50, primary_key=True)
	rl = models.ForeignKey('RLConfig', on_delete=models.DO_NOTHING)
	sim_market = models.ForeignKey('SimMarketConfig', on_delete=models.DO_NOTHING)


class RLConfig(models.Model):
	id = models.CharField(max_length=50, primary_key=True)
	gamma = models.FloatField(default=0.99)
	batch_size = models.IntegerField(default=32)
	replay_size = models.IntegerField(default=100000)
	learning_rate = models.FloatField(default=1e-6)
	sync_target_frames = models.IntegerField(default=1000)
	replay_start_size = models.IntegerField(default=10000)
	epsilon_decay_last_frame = models.IntegerField(default=75000)
	epsilon_start = models.FloatField(default=1.0)
	epsilon_final = models.FloatField(default=0.1)


class SimMarketConfig(models.Model):
	id = models.CharField(max_length=50, primary_key=True)
	max_storage = models.IntegerField(default=100)
	episode_size = models.IntegerField(default=50)
	max_price = models.IntegerField(default=10)
	max_quality = models.IntegerField(default=50)
	number_of_customers = models.IntegerField(default=20)
	production_price = models.IntegerField(default=3)
	storage_cost_per_product = models.FloatField(default=0.1)
