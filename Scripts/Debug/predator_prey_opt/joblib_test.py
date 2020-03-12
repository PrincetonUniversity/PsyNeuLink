from dask.distributed import Client
client = Client(scheduler_file='scheduler.json')

print(client)

client.restart()

import time
import joblib
import hypertunity as ht

def long_running_function(i):
    time.sleep(.1)
    return i

print("Doing some work ... ")

if __name__ == "__main__":

    domain = ht.Domain({
                    "cost_rate": set([-.8])
    })

    with joblib.parallel_backend('dask'):
        with joblib.Parallel() as parallel:
            jobs = [ht.Job(task="joblib_test2.py:run_games", args=(*domain.sample().as_namedtuple(),)) for s in range(10)]
            print([job.id for job in jobs])
            results = parallel(joblib.delayed(job)() for job in jobs)

    print(results)

#with joblib.parallel_backend('dask'):
#    joblib.Parallel(verbose=100)(
#        joblib.delayed(long_running_function)(i)
#        for i in range(10))

