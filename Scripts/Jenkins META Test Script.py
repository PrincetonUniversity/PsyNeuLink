import numpy as np
import random

# Insures that runs are deterministic for use in Jenkins testing
random.seed(0)
np.random.seed(0)

scripts =  ['Scripts/EVC System Laming Validation Test Script.py',
            'Scripts/Multilayer Learning Test Script.py',
            'Scripts/Reinforcement Learning Test Script.py',
            'Scripts/DDM Test Script.py',
            'Scripts/Stroop Model Test Script.py',
            'Scripts/Mixed NN & DDM script.py',
            'Scripts/System Graph and Input Test Script.py',
            'Scripts/Documentation Examples Script.py']

# foo_bar = __import__(script)
for script in scripts:
    file = open(script)
    print("\nRUNNING {}\n".format(script))
    exec(file.read())
    file.close()
    print ("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

