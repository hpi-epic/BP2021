from django.db import models


class Container(models.Model):
	"""
	This class represents one of the containers in our database.
	"""
	command = models.CharField(max_length=20, editable=False)
	config_file = models.CharField(max_length=500, editable=False)
	id = models.CharField(max_length=50, primary_key=True)
	created_at = models.DateTimeField(auto_now_add=True, editable=False)
	health_status = models.CharField(max_length=20, default='unknown')
	last_check_at = models.DateTimeField(auto_now_add=True)
	name = models.CharField(max_length=20)
	tensorboard_link = models.CharField(max_length=100, default='')

	def is_archived(self):
		return self.health_status == 'archived'

	def has_tensorboard_link(self):
		return self.tensorboard_link != ''


def update_container(container_id: str, updated_values: dict) -> None:
	"""
	This will update the container belonging to the given id with the data given in `updated_values`.

	Args:
		id (str): id for the container that should be updated
		updated_values (dict): All keys need to be member variables of `Container`.
	"""
	saved_container = Container.objects.get(id=container_id)
	for key, value in updated_values.items():
		setattr(saved_container, key, value)
	saved_container.save()
