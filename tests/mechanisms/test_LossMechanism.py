import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear
from psyneulink.core.components.functions.nonstateful.selectionfunctions import max_vs_next
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.scheduling.condition import Never, WhenFinished
from psyneulink.library.components.mechanisms.processing.transfer.lcamechanism import \
    LCAMechanism, MAX_VS_AVG, MAX_VS_NEXT, CONVERGENCE

class TestLossMechanism:

    @pytest.mark.composition
    def test_LossMechanism_construction(self):
        """Test sample and target assignments to LossMechanism"""
        pway1_mech_A = pnl.ProcessingMechanism(name='pway1_mech_A')
        pway1_mech_B = pnl.ProcessingMechanism(name='pway1_mech_B')
        pway1_mech_C = pnl.ProcessingMechanism(name='pway1_mech_C')
        pway2_mech_A = pnl.ProcessingMechanism(name='pway2_mech_A')
        pway2_mech_B = pnl.ProcessingMechanism(name='pway2_mech_B')
        pway2_mech_C = pnl.ProcessingMechanism(name='pway2_mech_C')
        pway3_mech_A = pnl.ProcessingMechanism(name='pway3_mech_A')
        pway3_mech_B = pnl.ProcessingMechanism(name='pway3_mech_B')
        solo_input_mech = pnl.ProcessingMechanism(name='solo_input_mech')
        loss_mech = pnl.LossMechanism(sample=pway3_mech_B.output_port, target=solo_input_mech)
        pway1 = [pway1_mech_A, pway1_mech_B, pway1_mech_C]
        pway2 = [pway2_mech_A, pway2_mech_B, pway2_mech_C]
        pway3 = [pway3_mech_A, pway3_mech_B]
        comp = pnl.AutodiffComposition(pathways=[pway1,pway2,pway3,solo_input_mech],
                                       targets=[(pway1_mech_B, pway2_mech_B.output_port),
                                                (pway1_mech_C, solo_input_mech),
                                                (pway2_mech_C, pnl.TARGET),
                                                loss_mech
                                                ]
                                       )
        loss_mechs = comp.loss_mechanisms

        assert loss_mechs[0].sample == pway1_mech_B.output_port
        assert loss_mechs[0].target == pway2_mech_B.output_port
        assert loss_mechs[1].sample == pway1_mech_C.output_port
        assert loss_mechs[1].target == solo_input_mech.output_port
        assert loss_mechs[2].sample == pway2_mech_C.output_port
        assert loss_mechs[2].target == comp.target_input_mechanisms[0].output_port
        assert loss_mechs[3].sample == pway3_mech_B.output_port
        assert loss_mechs[3].target == solo_input_mech.output_port

    def test_illegal_manual_LossMechanism(self):
        pway1_mech_A = pnl.ProcessingMechanism(name='pway1_mech_A')
        pway1_mech_B = pnl.ProcessingMechanism(name='pway1_mech_B')
        pway2_mech_A = pnl.ProcessingMechanism(name='pway2_mech_A')
        pway2_mech_B = pnl.ProcessingMechanism(name='pway2_mech_B')
        loss_mech = pnl.LossMechanism(sample=pway1_mech_B.output_port, target=pway2_mech_A.output_port)
        pway1 = [pway1_mech_A, pway1_mech_B]
        comp = pnl.AutodiffComposition(pathways=[pway1,loss_mech],
                                   targets={
                                       pway1_mech_B: TARGET,
                                       pway2_mech_B.output_port: TARGET
                                   })

        inputs = {pway1_mech_A: [[1]],
                  pway2_mech_A: [[1]],
                  }
        targets = {pway1_mech_B: [[1]],
                   comp.get_target_nodes()[1]: [[1]]}
        comp.learn(inputs=inputs,
                   targets=targets,
                   execution_mode=pnl.ExecutionMode.PyTorch)
