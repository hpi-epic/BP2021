from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm

import os
from .handle_uploading import handle_uploaded_file

def index(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		# print(request.FILES, request.POST)
		handle_uploaded_file(request.FILES['upload_config'])
		return render(request, 'index.html', {'form': form})# HttpResponseRedirect('/success/url/')
	else:
		form = UploadFileForm()
	return render(request, 'index.html', {'form': form})