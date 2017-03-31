# import numpy as np
# import random
#
# # Insures that runs are deterministic for use in Jenkins testing
# random.seed(0)
# np.random.seed(0)
#
# scripts =  [
#             'EVC System Laming Validation Test Script.py',
#             'Multilayer Learning Test Script.py',
#             'Reinforcement Learning Test Script.py',
#             'DDM Test Script.py',
#             'Stroop Model Test Script.py',
#             'Stroop Model Learning Test Script.py',
#             'Mixed NN & DDM script.py',
#             'System Graph and Input Test Script.py',
#             'Documentation Examples Script.py'
#             ]
#
# # foo_bar = __import__(script)
# for script in scripts:
#     file = open(script)
#     print("\nRUNNING {}\n".format(script))
#     exec(file.read())
#     file.close()
#     print ("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#

import numpy as np
import random
import subprocess
import sys

# Insures that runs are deterministic for use in Jenkins testing
random.seed(0)
np.random.seed(0)

scripts =  [
            'EVC System Laming Validation Test Script.py',
            'Multilayer Learning Test Script.py',
            'Reinforcement Learning Test Script.py',
            'DDM Test Script.py',
            'Stroop Model Test Script.py',
            'Stroop Model Learning Test Script.py',
            'Mixed NN & DDM script.py',
            'System Graph and Input Test Script.py',
            'Documentation Examples Script.py'
            ]

# foo_bar = __import__(script)
for script in scripts:
    file = open(script)
    print("\nRUNNING {}\n".format(script))
    # exec(file.read())
    for line in subprocess.check_output([sys.executable, script]).splitlines():
        print(str(line, 'utf-8'))
    file.close()
    print ("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

