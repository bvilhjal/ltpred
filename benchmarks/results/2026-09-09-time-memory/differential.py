import sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from ltpred.pearson_aitken import pa_estimate_batched

rng = np.random.default_rng(9031)
results = []
for k in range(200):
    d = int(rng.integers(1, 40))
    matrix = rng.normal(size=(d, d))
    cov = matrix @ matrix.T + np.eye(d)
    scale = np.sqrt(np.diag(cov))
    cov /= np.outer(scale, scale)
    target = int(rng.integers(d))
    state = rng.integers(0, 5, size=(20, d))
    state[1:4] = state[0]
    values = rng.normal(size=state.shape)
    lower, upper = np.full(state.shape, -np.inf), np.full(state.shape, np.inf)
    upper[state == 1] = values[state == 1]
    lower[state == 2] = values[state == 2]
    lower[state == 3] = upper[state == 3] = values[state == 3]
    lower[state == 4], upper[state == 4] = values[state == 4] - .1, values[state == 4] + .1
    dtype = np.float32 if k % 2 else np.float64
    lower, upper = lower.astype(dtype), upper.astype(dtype)
    if k % 3:
        lower, upper = np.asfortranarray(lower), np.asfortranarray(upper)
    lower.flags.writeable = upper.flags.writeable = False
    results.append(np.array(pa_estimate_batched(cov, lower, upper, target=target)))
np.save(sys.argv[2], np.concatenate(results, axis=1))
