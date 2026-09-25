"""Local before/after probe; each invocation imports an isolated source tree."""
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import runpy
import sys
import time
import tracemalloc

source, mode, case, destination = sys.argv[1:]
sys.path.insert(0, source)
import numpy as np
import scipy
import numba
from ltpred import Family, Member, estimate_liability
from ltpred.chunked import estimate_liability_pa_chunked, estimate_liability_gibbs_chunked

root = Path(__file__).resolve().parents[2]
destination = Path(destination)
if mode == 'agreement':
    result = {}
    rng = np.random.default_rng(934)
    for dtype in (np.float64, np.float32):
        cases = rng.random((41, 4)) < .2
        lower = np.where(cases, 1.64, -np.inf).astype(dtype)
        upper = np.where(cases, np.inf, 1.64).astype(dtype)
        lower[::7, 0] = upper[::7, 0] = .7
        for engine, fun in [('pa', estimate_liability_pa_chunked), ('gibbs', estimate_liability_gibbs_chunked)]:
            options = {} if engine == 'pa' else dict(seed=59, n_sim=103, burn_in=10, max_rounds=3, tol=1e6, return_var=True)
            for out in ('genetic', 'full'):
                values = fun(['o','s1','m','f'], lower, upper, h2=.5, c2=.1, m2=.05, out=out, chunk_size=7, **options)
                for i, value in enumerate(values):
                    result[f'{dtype.__name__}-{engine}-{out}-{i}'] = value
    # Mixed structures and tuple ids preserve arrival order and seeded streams.
    families = [Family(('f', i), [Member(role, -np.inf, 1.6, pid=(i, role))
                for role in (['o','m'] if i%2 else ['f','o','s1'])]) for i in range(20)]
    for method in ('pa', 'gibbs'):
        r = estimate_liability(families, h2=.5, method=method, out=('genetic','full'), seed=42, n_sim=103, burn_in=10, tol=1e6)
        for field in ('est','se','var'):
            for name, values in getattr(r, field).items():
                result[f'object-{method}-{field}-{name}'] = values
    from ltpred import fit_pairwise, fit_pairwise_multi
    single = runpy.run_path(str(root/'tests/test_pairwise.py'))
    multi = runpy.run_path(str(root/'tests/test_pairwise_multi.py'))
    # Reuse the independent exact-table fixture, not outputs from either fitter.
    components = {'A': np.array([[.4,-.12],[-.12,.3]]), 'E': np.array([[.6,.13],[.13,.7]])}
    fams, weights = multi['exact_pair_tables'](components)
    fit = fit_pairwise_multi(fams, weights=weights, sampling='ipw', tol=1e-12)
    result['multi-covariance'] = fit.parameter_covariance
    for name, values in fit.components.items():
        result[f'multi-{name}'] = values
    for name, values in fit.se.items():
        result[f'multi-se-{name}'] = values
    np.savez(destination, **result)
    print(f'{len(result)} output arrays saved to {destination}')
    sys.exit()

n = 1_000_000 if case == 'pa' else 50_000
lower, upper = np.full((n, 1), -np.inf), np.full((n, 1), 1.64)
fun = estimate_liability_pa_chunked if case == 'pa' else estimate_liability_gibbs_chunked
options = dict(chunk_size=4096) if case == 'pa' else dict(chunk_size=512, n_sim=20, burn_in=0, seed=53, tol=1e6, max_rounds=1, return_var=True)
# Warm all reached signatures before measuring; compile outside the memory trace.
fun(['o'], lower[:16], upper[:16], h2=.5, **options)
def call():
    return fun(['o'], lower, upper, h2=.5, **options)
record = dict(source=str(Path(source).resolve()), mode=mode, case=case, n=n,
              python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
              options=options, source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(source)/'ltpred').glob('*.py')})
if mode == 'memory':
    gc.collect()
    tracemalloc.start()
    values = call()
    record['peak_allocation_mib'] = tracemalloc.get_traced_memory()[1]/2**20
    tracemalloc.stop()
    record['peak_process_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    record['output_hashes'] = [hashlib.sha256(x.tobytes()).hexdigest() for x in values]
else:
    elapsed=[]
    for _ in range(5):
        gc.collect()
        t=time.perf_counter()
        values=call()
        elapsed.append(time.perf_counter()-t)
        del values
    record['seconds']=elapsed
    record['median_seconds']=float(np.median(elapsed))
destination.write_text(json.dumps(record, indent=2)+'\n')
print({k:v for k,v in record.items() if k not in ('source_sha256','output_hashes')})
