import cProfile
import os
import pstats
import signal
import sys
import time

# include the file you want to run the performance check on here!
import monitoring.agent_monitoring


class PerformanceMonitor():

	def __init__(self, function='monitoring.agent_monitoring.main()'):
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
		for file_name in os.listdir('./performance/'):
			if not file_name.endswith('.prof'):
				os.remove('./performance/' + file_name)

	def run_profiling(self) -> None:
		"""
		Run the profiler on a specified function. Automatically starts a web server to visualize the results.

		Args:
			function (str, optional): The function to be run. The format must be module.function. Defaults to 'monitoring.exampleprinter.run_example()'.
		"""
		if not os.path.isdir('performance'):
			os.mkdir('performance')

		date_time = time.strftime('%Y%m%d-%H%M%S')
		start_time = time.perf_counter()

		cProfile.run(self.function, filename='./performance/' + self.function + '_' + date_time, sort=3)
		p = pstats.Stats('./performance/' + self.function + '_' + date_time)

		# Estimate of how long the function took to run for the filename
		time_frame = str(round(time.perf_counter() - start_time, 3))
		filename = './performance/' + self.function + '_' + time_frame + '_secs_' + date_time + '.prof'

		p.sort_stats('cumulative').dump_stats(filename)

		# Remove the initial file created by cProfile, not the .prof file used for snakeviz
		self._remove_files()
		print(f'The result file was saved at {os.path.abspath(filename)}\n')
		# Visualize the results
		os.system('snakeviz ' + filename)


if __name__ == '__main__':  # pragma: no cover
	PerformanceMonitor().run_profiling()
