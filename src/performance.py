import cProfile
import os
import pstats
import signal
import sys

import agent_monitoring as am
import exampleprinter
import training


# deletes both result binaries
def handler(signum, frame) -> None:
	os.chdir('..')
	os.remove('results.prof')
	os.remove('results')
	sys.exit(0)


def run_profiling() -> None:
	cProfile.run('exampleprinter.main()', filename='../results', sort=3)
	p = pstats.Stats('../results')
	p.sort_stats('cumulative').dump_stats(filename='../results.prof')
	signal.signal(signal.SIGINT, handler)
	os.system('snakeviz ../results.prof')


def main() -> None:
	run_profiling()


if __name__ == '__main__':
	main()
