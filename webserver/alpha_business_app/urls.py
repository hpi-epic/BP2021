from django.urls import include, path

from . import views

urlpatterns = [
	path('', views.index, name='index'),
	path('download', views.download, name='download'),
	path('observe', views.observe, name='observe'),
	path('upload', views.upload, name='upload'),
	path('details/<str:container_id>', views.detail, name='detail'),
	path('configurator', views.configurator, name='configurator'),
	path('delete_config/<int:config_id>', views.delete_config, name='delete_config'),

	# AJAX relevant url's
	path('agent', views.agent, name='agent'),
	path('api_availability', views.api_availability, name='api_availability'),
	path('marketplace_changed', views.marketplace_changed, name='marketplace'),
	path('validate_config', views.config_validation, name='config_validation'),
	path('container_notification', views.container_notification, name='container_notification'),
	path('api_info', views.get_api_url, name='api_info'),

	# User relevant urls
	path('accounts/', include('django.contrib.auth.urls'))
]
