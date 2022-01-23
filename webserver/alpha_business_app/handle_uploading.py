import os

def handle_uploaded_file(f):
	path_to_configurations = './configurations'
	if not os.path.exists(path_to_configurations):
		os.mkdir(path_to_configurations)

	with open(os.path.join(path_to_configurations, 'my_config.json'), 'wb') as destination:
		for chunk in f.chunks():
			destination.write(chunk)