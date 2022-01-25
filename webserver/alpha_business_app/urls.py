from django.urls import path

from . import views

urlpatterns = [
	path('', views.index, name='index'),
	path('upload', views.upload, name='upload'),
	path('observe', views.observe, name='observe'),
	path('download', views.download, name='download')
]
