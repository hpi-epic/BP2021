from django.db import models

# Create your models here.


class Container(models.Model):
	config_file = models.CharField(max_length=500)
	container_id = models.CharField(max_length=50, primary_key=True)
	created_at = models.DateTimeField(auto_now_add=True, editable=False)
	health_status = models.CharField(max_length=20, default='unknown')
	last_check_at = models.DateTimeField(auto_now_add=True)
	name = models.CharField(max_length=20)
