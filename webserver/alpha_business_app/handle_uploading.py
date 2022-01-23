def handle_uploaded_file(f):
	print(f)
	with open('./my_config.json', 'wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)