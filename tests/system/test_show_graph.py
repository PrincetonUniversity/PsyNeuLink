import numpy as np
import pytest

from psyneulink.components.component import ComponentError
from psyneulink.components.functions.function import FunctionError
from psyneulink.components.functions.function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, Reinforcement, SoftMax
from psyneulink.components.functions.function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist, UniformToNormalDist
from psyneulink.components.mechanisms.mechanism import MechanismError
from psyneulink.components.mechanisms.processing.transfermechanism import TransferError, TransferMechanism
from psyneulink.library.subsystems.agt.lccontrolmechanism import LCControlMechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.globals.utilities import UtilitiesError
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.globals.keywords import ENABLED

class TestSimpleSystems:

    def test_process(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        p = Process(name="p",
                    pathway=[a, b])
        s = System(name="s",
                   processes=[p])

        s.show_graph()
        s.show_graph(show_dimensions=True)

    def test_diverging_pathways(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c",
                              default_variable=[0, 0, 0, 0, 0])
        p = Process(name="p",
                    pathway=[a, b])
        p2 = Process(name="p2",
                    pathway=[a, c])
        s = System(name="s",
                   processes=[p, p2])

        s.show_graph()
        s.show_graph(show_dimensions=True)

    def test_converging_pathways(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c",
                              default_variable=[0, 0, 0, 0, 0])
        p = Process(name="p",
                    pathway=[a, c])
        p2 = Process(name="p2",
                    pathway=[b, c])
        s = System(name="s",
                   processes=[p, p2])

        s.show_graph()
        s.show_graph(show_dimensions=True)

class TestLearning:

    def test_process(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        p = Process(name="p",
                    pathway=[a, b],
                    learning=ENABLED)
        s = System(name="s",
                   processes=[p])

        s.show_graph()
        s.show_graph(show_dimensions=True)
        s.show_graph(show_learning=True, show_dimensions=True)

    def test_diverging_pathways(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c",
                              default_variable=[0, 0, 0, 0, 0])
        p = Process(name="p",
                    pathway=[a, b],
                    learning=ENABLED)
        p2 = Process(name="p2",
                    pathway=[a, c],
                     learning=ENABLED)
        s = System(name="s",
                   processes=[p, p2])

        s.show_graph()
        s.show_graph(show_dimensions=True)
        s.show_graph(show_learning=True, show_dimensions=True)

    def test_converging_pathways(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c",
                              default_variable=[0, 0, 0, 0, 0])
        p = Process(name="p",
                    pathway=[a, c],
                    learning=ENABLED)
        p2 = Process(name="p2",
                    pathway=[b, c],
                     learning=ENABLED)
        s = System(name="s",
                   processes=[p, p2])

        s.show_graph()
        s.show_graph(show_dimensions=True)
        s.show_graph(show_learning=True, show_dimensions=True)

class TestControl:

    def test_process(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        LC = LCControlMechanism(modulated_mechanisms=[a,b],
                                objective_mechanism=ObjectiveMechanism(function=Linear,
                                                                       monitored_output_states=[b],
                                                                       name='lc_om'),
                                name="lc"
                                )
        p = Process(name="p",
                    pathway=[a, b])
        s = System(name="s",
                   processes=[p])

        s.show_graph()
        s.show_graph(show_dimensions=True)

    def test_diverging_pathways(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c",
                              default_variable=[0, 0, 0, 0, 0])
        LC = LCControlMechanism(modulated_mechanisms=[a,b],
                                objective_mechanism=ObjectiveMechanism(function=Linear,
                                                                       monitored_output_states=[b],
                                                                       name='lc_om'),
                                name="lc"
                                )
        p = Process(name="p",
                    pathway=[a, b])
        p2 = Process(name="p2",
                    pathway=[a, c])
        s = System(name="s",
                   processes=[p, p2])

        s.show_graph()
        s.show_graph(show_dimensions=True)

    def test_converging_pathways(self):
        a = TransferMechanism(name="a",
                              default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c",
                              default_variable=[0, 0, 0, 0, 0])
        LC = LCControlMechanism(modulated_mechanisms=[a,b],
                                objective_mechanism=ObjectiveMechanism(function=Linear,
                                                                       monitored_output_states=[b],
                                                                       name='lc_om'),
                                name="lc"
                                )
        p = Process(name="p",
                    pathway=[a, c])
        p2 = Process(name="p2",
                    pathway=[b, c])
        s = System(name="s",
                   processes=[p, p2])

        s.show_graph()
        s.show_graph(show_dimensions=True)