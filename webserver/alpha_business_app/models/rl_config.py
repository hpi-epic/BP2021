from django.db import models

from .abstract_config import AbstractConfig


class RlConfig(AbstractConfig, models.Model):
	sync_target_frames = models.IntegerField(null=True, default=None)
	learning_rate = models.FloatField(null=True, default=None)
	n_epochs = models.IntegerField(null=True, default=None)
	replay_start_size = models.IntegerField(null=True, default=None)
	epsilon_start = models.FloatField(null=True, default=None)
	epsilon_decay_last_frame = models.IntegerField(null=True, default=None)
	epsilon_final = models.FloatField(null=True, default=None)
	neurones_per_hidden_layer = models.IntegerField(null=True, default=None)
	buffer_size = models.IntegerField(null=True, default=None)
	tau = models.FloatField(null=True, default=None)
	batch_size = models.IntegerField(null=True, default=None)
	ent_coef = models.FloatField(null=True, default=None)
	learning_starts = models.IntegerField(null=True, default=None)
	gamma = models.FloatField(null=True, default=None)
	replay_size = models.IntegerField(null=True, default=None)
	n_steps = models.IntegerField(null=True, default=None)
	clip_range = models.FloatField(null=True, default=None)
