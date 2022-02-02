from django.db import models

# Create your models here.


class Container(models.Model):
	container_id = models.CharField(max_length=50, primary_key=True)
	config_file = models.CharField(max_length=500)
	created_at = models.DateTimeField(auto_now_add=True, editable=False)
	last_check_at = models.DateTimeField(auto_now_add=True)
	health_status = models.CharField(max_length=20, default='unknown')
