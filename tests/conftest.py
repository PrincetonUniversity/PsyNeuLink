import pytest
import random
import time
import numpy as np

# def pytest_addoption(parser):
#     parser.addoption(
#         '--pnl-seed',
#         action='store',
#         default=int(time.time() * 256),
#         help='the seed to use for each test'
#     )

def pytest_runtest_call(item):
    # seed = int(item.config.getoption('--pnl-seed'))
    seed = 0
    random.seed(seed)
    np.random.seed(seed)