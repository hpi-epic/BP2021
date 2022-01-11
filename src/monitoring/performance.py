import cProfile
import os
import pstats
import time

# include the file you want to run the performance check on here!
import monitoring.exampleprinter


def _remove_files() -> None:
	"""
	Remove the unneeded result files created by the performance runs.
	"""
	for file_name in os.listdir('./performance/'):
		if not file_name.endswith('.prof'):
			os.remove('./performance/' + file_name)


def run_profiling(function='monitoring.exampleprinter.run_example()') -> None:
	"""
	Run the profiler on a specified function. Automatically starts a web server to visualize the results.

	Args:
		function (str, optional): The function to be run. The format must be module.function. Defaults to 'monitoring.exampleprinter.run_example()'.
	"""
	if not os.path.isdir('performance'):
		os.mkdir('performance')

	date_time = time.strftime('%Y%m%d-%H%M%S')
	start_time = time.perf_counter()

	cProfile.run(function, filename='./performance/' + function + '_' + date_time, sort=3)
	p = pstats.Stats('./performance/' + function + '_' + date_time)

	# Estimate of how long the function took to run for the filename
	time_frame = str(round(time.perf_counter() - start_time, 3))
	filename = './performance/' + function + '_' + time_frame + '_secs_' + date_time + '.prof'

	p.sort_stats('cumulative').dump_stats(filename)

	# Remove the initial file created by cProfile, not the .prof file used for snakeviz
	_remove_files()
	# Visualize the results
	os.system('snakeviz ' + filename)


if __name__ == '__main__':  # pragma: no cover
	run_profiling()
