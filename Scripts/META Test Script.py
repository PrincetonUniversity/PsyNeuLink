scripts =  ['EVC System Laming Validation Test Script.py',
            'Multilayer Learning Test Script.py',
            'Reinforcement Learning Test Script.py',
            'DDM Test Script.py',
            'Stroop Model Test Script.py',
            'Mixed NN & DDM script.py',
            'System Graph and Input Test Script.py',
            ]

# foo_bar = __import__(script)
for script in scripts:
    file = open(script)
    print("\nRUNNING {}\n".format(script))
    exec(file.read())
    file.close()
    print ("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

