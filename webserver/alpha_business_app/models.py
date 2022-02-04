import os
import shutil

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .constants import DATA_DIR


class Container(models.Model):
	config_file = models.CharField(max_length=500)
	container_id = models.CharField(max_length=50, primary_key=True)
	created_at = models.DateTimeField(auto_now_add=True, editable=False)
	health_status = models.CharField(max_length=20, default='unknown')
	last_check_at = models.DateTimeField(auto_now_add=True)
	name = models.CharField(max_length=20)


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
	print(updated_values)
	for key, value in updated_values.items():
		setattr(saved_container, key, value)
	saved_container.save()
