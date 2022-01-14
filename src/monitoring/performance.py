import cProfile
import os
import pstats
import time

# include the file you want to run the performance check on here!
import monitoring.exampleprinter


def remove_files() -> None:
	"""
	Remove the unneeded result files created by the performance runs.
	"""
	for file_name in os.listdir(os.path.join('results', 'performance')):
		if not file_name.endswith('.prof'):
			os.remove(os.path.join('results', 'performance', file_name))


def run_profiling(function='monitoring.exampleprinter.run_example()') -> None:
	"""
	Run the profiler on a specified function. Automatically starts a web server to visualize the results.

	Args:
		function (str, optional): The function to be run. The format must be module.function. Defaults to 'monitoring.exampleprinter.run_example()'.
	"""
	if not os.path.isdir(os.path.join('results', 'performance')):
		os.mkdir(os.path.join('results', 'performance'))

	date_time = time.strftime('%b%d_%H-%M-%S')
	filename = os.path.join('results', 'performance', f'{function}_{date_time}')

	start_time = time.perf_counter()
	cProfile.run(function, filename=filename, sort=3)
	# Estimate of how long the function took to run for the filename
	time_frame = str(round(time.perf_counter() - start_time, 3))

	p = pstats.Stats(filename)
	dumped_filename = os.path.join('results', 'performance', f'{function}_{time_frame}_secs_{date_time}.prof')
	p.sort_stats('cumulative').dump_stats(filename=dumped_filename)

	# Remove the initial file created by cProfile, not the .prof file used for snakeviz
	remove_files()
	# Visualize the results
	os.system(f'snakeviz {dumped_filename}')


if __name__ == '__main__':  # pragma: no cover
	run_profiling()
