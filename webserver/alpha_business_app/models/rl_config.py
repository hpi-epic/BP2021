from django.db import models

from .abstract_config import AbstractConfig


class RlConfig(AbstractConfig, models.Model):
	sync_target_frames = models.IntegerField(null=True, default=None)
	replay_size = models.IntegerField(null=True, default=None)
	stable_baseline_test = models.FloatField(null=True, default=None)
	epsilon_start = models.FloatField(null=True, default=None)
	batch_size = models.IntegerField(null=True, default=None)
	learning_rate = models.FloatField(null=True, default=None)
	replay_start_size = models.IntegerField(null=True, default=None)
	testvalue2 = models.FloatField(null=True, default=None)
	epsilon_decay_last_frame = models.IntegerField(null=True, default=None)
	epsilon_final = models.FloatField(null=True, default=None)
	gamma = models.FloatField(null=True, default=None)
