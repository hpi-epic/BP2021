from importlib import reload

import exampleprinter

for i in range(0, 100):
	with open('myfile.txt', 'w') as f:
		f.write(str(i))
	reload(exampleprinter)
