from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.transferfunctions import Exponential
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.modulatorysignals.gatingsignal import GatingSignal

from psyneulink.core.globals.keywords import SLOPE, RESULT

class TestControlSignals:
    def test_control_signal_intensity_cost_function(self):

        mech = TransferMechanism()
        ctl_sig = ControlSignal(projections=[(SLOPE, mech)],
                                intensity_cost_function=Exponential(rate=1))
        ctl_mech = ControlMechanism(control_signals=[ctl_sig])
        ctl_mech.execute()
        assert True

    def test_alias_equivalence_for_modulates_and_projections(self):
        inputs = [1, 9, 4, 3, 2]
        comp1 = Composition()
        tMech1 = TransferMechanism()
        tMech2 = TransferMechanism()
        cMech1 = ControlMechanism(control_signals=ControlSignal(modulates=(SLOPE, tMech2)),
                                  objective_mechanism=ObjectiveMechanism(monitor=(RESULT, tMech2)))
        comp1.add_nodes([tMech1, tMech2, cMech1])
        comp1.add_linear_processing_pathway([cMech1, tMech1, tMech2])
        comp1.run(inputs=inputs)
        comp2 = Composition()
        tMech3 = TransferMechanism()
        tMech4 = TransferMechanism()
        cMech2 = ControlMechanism(control_signals=ControlSignal(projections=(SLOPE, tMech4)),
                                  objective_mechanism=ObjectiveMechanism(monitor=(RESULT, tMech4)))
        comp2.add_nodes([tMech3, tMech4, cMech2])
        comp2.add_linear_processing_pathway([cMech2, tMech3, tMech4])
        comp2.run(inputs=inputs)
        assert comp1.results == comp2.results

class TestGatingSignals:
    def test_alias_equivalence_for_modulates_and_projections(self):
        inputs = [1123, 941, 43, 311, 21]
        Tx1 = TransferMechanism()
        Ty1 = TransferMechanism()
        G1 = GatingMechanism(gating_signals=[GatingSignal(modulates=Tx1)])
        comp1 = Composition()
        comp1.add_nodes([Tx1, Ty1, G1])
        comp1.add_linear_processing_pathway([Tx1, Ty1, G1, Tx1])
        comp1.run(inputs={Tx1: inputs})

        Tx2 = TransferMechanism()
        Ty2 = TransferMechanism()
        G2 = GatingMechanism(gating_signals=[GatingSignal(projections=Tx2)])
        comp2 = Composition()
        comp2.add_nodes([Tx2, Ty2, G2])
        comp2.add_linear_processing_pathway([Tx2, Ty2, G2, Tx2])
        comp2.run(inputs={Tx2: inputs})
        assert comp1.results == comp2.results
