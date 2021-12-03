import cProfile
import os
import pstats
import signal

cProfile.run('agent_monitoring.main()', filename='../results', sort=3)
p = pstats.Stats('../results')
p.sort_stats('cumulative').dump_stats(filename='../results.dmp')

os.system('snakeviz ../results.dmp')


def handler(signum, frame):

	os.remove('../results.dmp')
	exit(0)


signal.signal(signal.SIGINT, handler)
