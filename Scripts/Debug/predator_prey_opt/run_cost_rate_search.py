from dask.distributed import Client, LocalCluster

import time
import joblib
import hypertunity as ht
import importlib
import os
import sys

def import_script(path):
    """Import a module or script by a given path.

    Args:
     path: :obj:`str`, can be either a module import of the form
     [package.]*[module] if the outer most package is in the
     `PYTHONPATH`, or a path to an arbitrary python script.

    Returns:
    The loaded python script as a module.
    """
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError:
       if not os.path.isfile(path):
           raise FileNotFoundError(f"Cannot find script {path}.")
       if not os.path.basename(path).endswith(".py"):
           raise ValueError(
               f"Expected a python script ending with *.py, "
               f"found {os.path.basename(path)}.")
       import_path = os.path.dirname(os.path.abspath(path))
       sys.path.append(import_path)
       module = importlib.import_module(
       f"{os.path.basename(path).rstrip('.py')}",
       package=f"{os.path.basename(import_path)}")
       sys.path.pop()

    return module

predator_prey = getattr(import_script('predator_prey_dmt.py'), 'main')

if __name__ == '__main__':

    client = Client(scheduler_file='scheduler.json')
    #client = Client()  # This is actually the following two commands
    print(client)

    domain = ht.Domain({
        "cost_rate": [-.9, -.1]})

    optimiser = ht.BayesianOptimisation(domain)

    domain = ht.Domain({
                    "cost_rate": set([-.8])
    })

    with joblib.parallel_backend('dask'):
        with joblib.Parallel() as parallel:
            print("Doing the work ... ")
            results = parallel(joblib.delayed(predator_prey)(*domain.sample().as_namedtuple()) for s in range(2))

    print(results)
