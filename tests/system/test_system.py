import numpy as np
import random

from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Functions.Function import Linear, Logistic, BogaczEtAl, Integrator
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import *

def test_danglingControlledMech():
    #
    #   first section is from Stroop Demo
    #
    Color_Input = TransferMechanism(name='Color Input', function=Linear(slope = 0.2995))
    Word_Input = TransferMechanism(name='Word Input', function=Linear(slope = 0.2995))

    # Processing Mechanisms (Control)
    Color_Hidden = TransferMechanism(
        name='Colors Hidden',
        function=Logistic(gain=(1.0, ControlProjection)),
    )
    Word_Hidden = TransferMechanism(
        name='Words Hidden',
        function=Logistic(gain=(1.0, ControlProjection)),
    )
    Output = TransferMechanism(
        name='Output',
        function=Logistic(gain=(1.0, ControlProjection)),
    )

    # Decision Mechanisms
    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=(1.0),
            threshold=(0.1654),
            noise=(0.5),
            starting_point=(0),
            t0=0.25,
        ),
        name='Decision',
    )
    # Outcome Mechanisms:
    Reward = TransferMechanism(name='Reward')

    # Processes:
    ColorNamingProcess = process(
        default_input_value = [0],
        pathway = [Color_Input, Color_Hidden, Output, Decision],
        name = 'Color Naming Process',
    )

    WordReadingProcess = process(
        default_input_value = [0],
        pathway = [Word_Input, Word_Hidden, Output, Decision],
        name = 'Word Reading Process',
    )

    RewardProcess = process(
        default_input_value = [0],
        pathway = [Reward],
        name = 'RewardProcess',
    )

    # add another DDM but do not add to system
    second_DDM = DDM(
        function=BogaczEtAl(
            drift_rate=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            noise=(0.5),
            starting_point=(0),
            t0=0.45
        ),
        name='second_DDM',
    )

    # System:
    mySystem = system(
        processes=[ColorNamingProcess, WordReadingProcess, RewardProcess],
        controller=EVCMechanism,
        enable_controller=True,
        # monitor_for_control=[Reward, (DDM_PROBABILITY_UPPER_THRESHOLD, 1, -1)],
        name='EVC Gratton System',
    )

    # no assert, should only complete without error

class TestDocumentationExamples:
    def test_mechs_in_pathway(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = DDM()
        some_params = {PARAMETER_STATE_PARAMS:{THRESHOLD:2,NOISE:0.1}}
        my_process = process(pathway=[mechanism_1, TransferMechanism, (mechanism_2, some_params, 0)])
        result = my_process.execute()

        assert(result == np.array([2]))

    def test_default_projection(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_inline_projection_using_existing_projection(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        projection_A = MappingProjection()
        my_process = process(pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_inline_projection_using_keyword(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        my_process = process(pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_standalone_projection(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        projection_A = MappingProjection(sender=mechanism_1, receiver=mechanism_2)
        my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_process_learning(seed0):
        mechanism_1 = TransferMechanism(function=Logistic)
        mechanism_2 = TransferMechanism(function=Logistic)
        mechanism_3 = TransferMechanism(function=Logistic)
        my_process = process(
            pathway=[mechanism_1, mechanism_2, mechanism_3],
            learning=LEARNING_PROJECTION,
            target=[0],
        )
        result = my_process.execute()

        np.testing.assert_allclose(result, np.array([0.65077768]))

class TestGraphAndInput:
    def test_branch(self):
        a = TransferMechanism(name='a', default_input_value=[0,0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')

        p1 = process(pathway=[a, b, c], name='p1')
        p2 = process(pathway=[a, b, d], name='p2')

        s = system(
            processes=[p1, p2],
            name='Branch System',
            initial_values={a:[1,1]},
        )

        inputs={a:[2,2]}
        s.run(inputs)

        assert [a] == s.originMechanisms.mechanisms
        assert set([c, d]) == set(s.terminalMechanisms.mechanisms)

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == TERMINAL
        assert d.systems[s] == TERMINAL

    def test_bypass(self):
        a = TransferMechanism(name='a', default_input_value=[0,0])
        b = TransferMechanism(name='b', default_input_value=[0,0])
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')

        p1 = process(pathway=[a, b, c, d], name='p1')
        p2 = process(pathway=[a, b, d], name='p2')

        s = system(
            processes=[p1, p2],
            name='Bypass System',
            initial_values={a:[1,1]},
        )

        inputs={a:[[2,2],[0,0]]}
        s.run(inputs=inputs)

        assert [a] == s.originMechanisms.mechanisms
        assert [d] == s.terminalMechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == INTERNAL
        assert d.systems[s] == TERMINAL

    def test_chain(self):
        a = TransferMechanism(name='a', default_input_value=[0,0,0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')
        e = TransferMechanism(name='e')

        p1 = process(pathway=[a, b, c], name='p1')
        p2 = process(pathway=[c, d, e], name='p2')

        s = system(
            processes=[p1, p2],
            name='Chain System',
            initial_values={a:[1,1,1]},
        )

        inputs={a:[[2,2,2],[0,0,0]]}
        s.run(inputs=inputs)

        assert [a] == s.originMechanisms.mechanisms
        assert [e] == s.terminalMechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == INTERNAL
        assert d.systems[s] == INTERNAL
        assert e.systems[s] == TERMINAL

    def test_convergent(self):
        a = TransferMechanism(name='a', default_input_value=[0,0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        c = TransferMechanism(name='c', default_input_value=[0])
        d = TransferMechanism(name='d')
        e = TransferMechanism(name='e')

        p1 = process(pathway=[a, b, e], name='p1')
        p2 = process(pathway=[c, d, e], name='p2')

        s = system(
            processes=[p1, p2],
            name='Convergent System',
            initial_values={a:[1,1]},
        )

        inputs={a:[[2,2]], c:[[0]]}
        s.run(inputs=inputs)

        assert set([a, c]) == set(s.originMechanisms.mechanisms)
        assert [e] == s.terminalMechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == ORIGIN
        assert d.systems[s] == INTERNAL
        assert e.systems[s] == TERMINAL

    def cyclic_one_process(self):
        a = TransferMechanism(name='a', default_input_value=[0,0])
        b = TransferMechanism(name='b', default_input_value=[0,0])

        p1 = process(pathway=[a, b, a], name='p1')

        s = system(
            processes=[p1],
            name='Cyclic System with one Process',
            initial_values={a:[1,1]},
        )

        inputs={a:[1,1]}
        s.run(inputs=inputs)

        assert [a] == s.originMechanisms.mechanisms
        assert [] == s.terminalMechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INITIALIZE_CYCLE

    def cyclic_two_processes(self):
        a = TransferMechanism(name='a', default_input_value=[0,0])
        b = TransferMechanism(name='b', default_input_value=[0,0])
        c = TransferMechanism(name='c', default_input_value=[0,0])

        p1 = process(pathway=[a, b, a], name='p1')
        p2 = process(pathway=[a, c, a], name='p2')

        s = system(
            processes=[p1, p2],
            name='Cyclic System with two Processes',
            initial_values={a:[1,1]},
        )

        inputs={a:[1,1]}
        s.run(inputs=inputs)

        assert [a] == s.originMechanisms.mechanisms
        assert [] == s.terminalMechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INITIALIZE_CYCLE
        assert c.systems[s] == INITIALIZE_CYCLE

    def cyclic_extended_loop(self):
        a = TransferMechanism(name='a', default_input_value=[0,0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')
        e = TransferMechanism(name='e', default_input_value=[0])
        f = TransferMechanism(name='f')

        p1 = process(pathway=[a, b, c, d], name='p1')
        p2 = process(pathway=[e, c, f, b, d], name='p2')

        s = system(
            processes=[p1, p2],
            name='Cyclic System with Extended Loop',
            initial_values={a:[1,1]},
       )

        inputs={a:[2,2], e:[0]}
        s.run(inputs=inputs)

        assert set([a, c]) == set(s.originMechanisms.mechanisms)
        assert [d] == s.terminalMechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == CYCLE
        assert c.systems[s] == INTERNAL
        assert d.systems[s] == TERMINAL
        assert e.systems[s] == ORIGIN
        assert f.systems[s] == INITIALIZE_CYCLE
