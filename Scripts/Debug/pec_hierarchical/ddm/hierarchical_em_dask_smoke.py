"""Parity check: distributed (Dask, per-subject) EM vs serial EM (M3).

Both paths share the same M-step and per-subject optimizer with identical seeds, so the results
should match closely. Runs a small problem on a local Dask cluster. For multi-node SLURM, launch
under `python -m psyneulink.dask_run` and pass the launcher client instead of a LocalCluster.
"""

import os
import sys

import numpy as np
from dask.distributed import LocalCluster, Client

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core",):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
from ddm_subjects import (
    generate_group_data, build_serial_subjects, ddm_pec_factory, transform, fit_bounds, FIT_PARAMS,
)
from pec_likelihood import PECGroupLikelihood
from laplace_em import fit_laplace_em
from dask_estep import make_distributed_estep_runner

ESTEP_OPTIONS = {"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200}


def run(n_subjects=4, num_trials=30, num_estimates=60, max_em_iterations=4, n_workers=4, seed=0):
    rng = np.random.default_rng(seed)
    tf = transform()
    lower, upper = fit_bounds()
    group = generate_group_data(n_subjects, [0.3, -0.2], [0.4, 0.4], num_trials, rng, num_estimates)
    payloads = group["payloads"]

    common = dict(
        n_subjects=n_subjects, n_params=len(FIT_PARAMS), transform=tf,
        max_em_iterations=max_em_iterations, em_tol=1e-3, variance_floor=1e-3,
        hessian_step=1e-3, estep_options=ESTEP_OPTIONS,
    )

    print("serial EM ...")
    like = PECGroupLikelihood(build_serial_subjects(payloads), tf)
    serial = fit_laplace_em(like.log_likelihood_s, **common)

    print("distributed EM ...")
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    try:
        runner = make_distributed_estep_runner(
            client, ddm_pec_factory, payloads, lower, upper,
            worker_cores=max(1, 8 // n_workers), hessian_step=1e-3, variance_floor=1e-3,
            estep_options=ESTEP_OPTIONS,
        )
        dist = fit_laplace_em(None, estep_runner=runner, **common)
    finally:
        client.close()
        cluster.close()

    print("serial beta / sigma:", serial.beta.ravel(), serial.sigma)
    print("dist   beta / sigma:", dist.beta.ravel(), dist.sigma)
    print("max|beta diff| :", np.max(np.abs(serial.beta - dist.beta)))
    print("max|sigma diff|:", np.max(np.abs(serial.sigma - dist.sigma)))
    print("max|z_hat diff|:", np.max(np.abs(serial.z_hat - dist.z_hat)))


if __name__ == "__main__":
    run()
