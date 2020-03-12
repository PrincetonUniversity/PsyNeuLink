from dask.distributed import Client, LocalCluster

import time
import joblib
import hypertunity as ht

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
            jobs = [ht.Job(task="predator_prey_dmt.py:run_games", args=(*domain.sample().as_namedtuple(),)) for s in range(10)]
            results = parallel(joblib.delayed(job)() for job in jobs)

    print(results)
