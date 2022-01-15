from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render


def index(request):
	if request.method == 'POST' and 'run_script' in request.POST:
		# import function to run
		from src.monitoring.exampleprinter import run_example

		# call function
		run_example() 

		# return user to required page
		return HttpResponse("Hello, world. You're at the polls index.")# HttpResponseRedirect(reverse(app_name:view_name)
	else:
		return render(request, 'index.html') # HttpResponse("Hello, world. You're at the polls index.")
