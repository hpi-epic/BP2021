from django.urls import path

from . import views

urlpatterns = [
	path('', views.index, name='index'),
	path('download', views.download, name='download'),
	path('observe', views.observe, name='observe'),
	path('upload', views.upload, name='upload'),
	path('details/<str:container_id>', views.detail, name='detail'),
	path('configurator', views.configurator, name='configurator'),
	path('delete_config/<int:config_id>', views.delete_config, name='delete_config'),
]
