import os
import shutil

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .constants import DATA_DIR


class Container(models.Model):
	"""
	This class represents one of the containers in our database.
	"""
	command = models.CharField(max_length=20, editable=False)
	config_file = models.CharField(max_length=500, editable=False)
	container_id = models.CharField(max_length=50, primary_key=True)
	created_at = models.DateTimeField(auto_now_add=True, editable=False)
	health_status = models.CharField(max_length=20, default='unknown')
	last_check_at = models.DateTimeField(auto_now_add=True)
	name = models.CharField(max_length=20)
	tensorboard_link = models.CharField(max_length=100, default='')

	def id(self):
		return self.container_id

	def is_archived(self):
		return 'archived' == self.health_status

	def has_tensorboard_link(self):
		return self.tensorboard_link


@receiver(post_delete, sender=Container)
def delete_container(sender, instance, **kwargs) -> None:
	"""
	This will be called when you delete a container from the database,
	We need to make sure, that we delete the objects' data folder
	"""
	container_id = instance.container_id
	container_data_path = os.path.join(DATA_DIR, container_id)
	if os.path.exists(container_data_path):
		shutil.rmtree(container_data_path)


def update_container(id: str, updated_values: dict) -> None:
	"""
	This will update the container belonging to the given id with the data given in `updated_values`.

	Args:
		id (str): id for the container that should be updated
		updated_values (dict): All keys need to be member variables of `Container`.
	"""
	saved_container = Container.objects.get(container_id=id)
	for key, value in updated_values.items():
		setattr(saved_container, key, value)
	saved_container.save()
