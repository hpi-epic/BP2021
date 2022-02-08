import os
import shutil

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .constants import DATA_DIR


class Container(models.Model):
	command = models.CharField(max_length=20, editable=False)
	config_file = models.CharField(max_length=500, editable=False)
	container_id = models.CharField(max_length=50, primary_key=True)
	created_at = models.DateTimeField(auto_now_add=True, editable=False)
	health_status = models.CharField(max_length=20, default='unknown')
	last_check_at = models.DateTimeField(auto_now_add=True)
	name = models.CharField(max_length=20)
	tensorboard_link = models.CharField(max_length=100, default='')

	def is_archived(self):
		return 'archived' == self.health_status

	def has_data(self):
		container_data_path = os.path.join(DATA_DIR, self.container_id)
		return os.path.exists(container_data_path)

	def has_tensorboard_link(self):
		return self.tensorboard_link


@receiver(post_delete, sender=Container)
def delete_container(sender, instance, **kwargs) -> None:
	# method will be called when you delete an object,
	# we need to make sure, that we delete the objects' data folder
	container_id = instance.container_id
	container_data_path = os.path.join(DATA_DIR, container_id)
	if os.path.exists(container_data_path):
		shutil.rmtree(container_data_path)


def update_container(id: str, updated_values: dict) -> None:
	saved_container = Container.objects.get(container_id=id)
	for key, value in updated_values.items():
		setattr(saved_container, key, value)
	saved_container.save()
