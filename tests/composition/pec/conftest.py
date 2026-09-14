import pytest


@pytest.fixture
def single_threaded_torch():
    """Hold torch to one thread for the duration of one test.

    Sampling makes many small calls to a trained estimator.  Under pytest-xdist there is already
    one process per core, while torch sizes its own thread pool from the whole machine, so every
    worker tries to use every core and the contention costs far more than the parallelism gains:
    these tests run in a tenth of the time on one thread.

    Requested per test rather than applied to the directory. The number of threads changes
    floating-point summation order, and the tests that compare a distributed fit against an
    in-process one need both sides to agree exactly.
    """
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
