from django.urls import path

from . import views

urlpatterns = [
	path('', views.index, name='index'),
	path('download', views.download, name='download'),
	path('observe', views.observe, name='observe'),
	path('upload', views.upload, name='upload'),
	path('start_container', views.start_container, name='start'),
	path('details/<str:container_id>', views.detail, name='detail')
]
