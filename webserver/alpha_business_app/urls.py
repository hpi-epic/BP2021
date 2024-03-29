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
	path('new_agent', views.new_agent, name='new_agent'),
	path('agent_changed', views.agent_changed, name='agent_changed'),
	path('api_availability', views.api_availability, name='api_availability'),
	path('marketplace_changed', views.marketplace_changed, name='marketplace'),
	path('validate_config', views.config_validation, name='config_validation'),

	# User relevant urls
	path('accounts/', include('django.contrib.auth.urls'))
]
