from django.db import models

# Create your models here.


class Container(models.Model):
	container_id = models.CharField()
	config_file = models.CharField()
