import cProfile
import os
import pstats
import signal
import sys
import time

# include the file you want to run the performance check on the line below!
import recommerce.monitoring.agent_monitoring.am_monitoring
from recommerce.configuration.path_manager import PathManager


class PerformanceMonitor():

	def __init__(self, function='monitoring.agent_monitoring.am_monitoring.run_monitoring_session()'):
		self.function = function
		# Signal handler for e.g. KeyboardInterrupt
		self.abort_counter = 0
		signal.signal(signal.SIGINT, self._signal_handler)

	def _signal_handler(self, signum, frame):  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		if self.abort_counter == 0:
			print(f'\nStopping performance monitoring of {self.function}...\n')
		self.abort_counter += 1
		sys.exit(0)

	def _remove_files(self) -> None:
		"""
		Remove the unneeded result files created by the performance runs.
		"""
		for file_name in os.listdir(os.path.join(PathManager.results_path, 'performance')):
			if not file_name.endswith('.prof'):
				os.remove(os.path.join(PathManager.results_path, 'performance', file_name))

	def run_profiling(self) -> None:
		"""
		Run the profiler on a specified function. Automatically starts a web server to visualize the results.
		"""
		os.makedirs(os.path.join(PathManager.results_path, 'performance'), exist_ok=True)

		date_time = time.strftime('%b%d_%H-%M-%S')
		filename = os.path.join(PathManager.results_path, 'performance', f'{self.function}_{date_time}')

		start_time = time.perf_counter()
		cProfile.run(self.function, filename=filename, sort=3)
		# Estimate of how long the function took to run for the filename
		time_frame = str(round(time.perf_counter() - start_time, 3))

		p = pstats.Stats(filename)
		dumped_filename = os.path.join(PathManager.results_path, 'performance', f'{self.function}_{time_frame}_secs_{date_time}.prof')
		p.sort_stats('cumulative').dump_stats(filename=dumped_filename)

		# Remove the initial file created by cProfile, not the .prof file used for snakeviz
		self._remove_files()
		# Visualize the results
		os.system(f'snakeviz {dumped_filename}')


if __name__ == '__main__':  # pragma: no cover
	PerformanceMonitor().run_profiling()
