scripts =  ['EVC System Laming Validation Test Script.py',
            'Multilayer Learning Test Script.py',
            'Reinforcement Learning Test Script.py',
            'DDM Test Script.py']

# foo_bar = __import__(script)
for script in scripts:
    exec(open(script).read())
    # exec(open(script).close())
