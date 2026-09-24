"""Matched local review probe. Run in a fresh, one-thread process per source/case.

python probe.py --source /path/to/checkout --case moments --output result.json
Warm runtime and allocation tracing are separate calls; neither includes JIT.
RSS is process high-water memory, including imports/JIT. No battery bypass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--case', choices=['moments', 'quadrature', 'unique', 'pid', 'dense', 'mendelian'], required=True)
parser.add_argument('--n', type=int, default=400)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--values', type=Path)
args = parser.parse_args()
if sys.platform == 'darwin':
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    settings = subprocess.check_output(['pmset', '-g', 'custom'], text=True)
    if "'AC Power'" not in power or any('lowpowermode' in line and line.split()[-1] == '1' for line in settings.splitlines()):
        raise SystemExit('Timing requires AC power and Low Power Mode off')
else:
    power = 'not macOS'
sys.path.insert(0, str(args.source.resolve()))
import numpy as np
import scipy
import numba
import ltpred
from ltpred import quadrature
from ltpred.family import _pid_key
from ltpred.simulate import simulate_register_liabilities

if args.case == 'moments':
    means = np.linspace(-4., 4., 65536)
    def run():
        if hasattr(quadrature, '_moments_array'):
            return quadrature._moments_array(means, .75, 1.3, np.inf)
        return np.array([quadrature._moments(x, .75, 1.3, np.inf) for x in means])
elif args.case in ('quadrature', 'unique'):
    patterns = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool)
    patterns = np.tile(patterns, (16, 1))
    thr = np.full(patterns.shape, 1.3)
    if args.case == 'unique':
        thr += np.arange(len(patterns))[:, None] * .001
    lo, hi = np.where(patterns, thr, -np.inf), np.where(patterns, np.inf, thr)
    def run():
        r = quadrature.estimate_liability_quadrature_arrays(['o', 'm', 's1'], lo, hi, .5)
        return np.column_stack((r.est, r.var, r.error, r.n_nodes))
elif args.case == 'pid':
    pids = [f'person_{i}' for i in range(100000)]
    def run():
        return [_pid_key(pid) for pid in pids]
else:
    n = args.n
    ids = list(range(n))
    father = [None if i < 2 else 2 * ((i - 2) // 4) for i in ids]
    mother = [None if i < 2 else 2 * ((i - 2) // 4) + 1 for i in ids]
    def run():
        options = {} if args.case == 'dense' else {'method': 'mendelian'}
        result = simulate_register_liabilities(np.random.default_rng(17), ids, father, mother,
            h2=.5, cip_ages=[0., 100.], cip_values=[.001, .1], eval_age=70., **options)
        return np.column_stack((result.genetic, result.residual_var))

result = run()
times = []
for _ in range(3):
    start = time.perf_counter()
    run()
    times.append(time.perf_counter() - start)
tracemalloc.start()
run()
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
if args.values is not None and args.case != 'pid':
    np.save(args.values, result)
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((args.source / 'ltpred').glob('*.py'))}
record = dict(case=args.case, n=args.n, source=str(args.source), hashes=source_hashes,
              seconds=times, median_seconds=statistics.median(times),
              traced_peak_bytes=peak, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
              numba=numba.__version__, ltpred=ltpred.__version__, threads=numba.get_num_threads(), power=power)
args.output.write_text(json.dumps(record, indent=2)+'\n')
