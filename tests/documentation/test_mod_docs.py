import pytest
import doctest


import psyneulink as pnl


@pytest.mark.parametrize("mod", [# Ports
                                 pnl.core.components.ports.parameterport,
                                 pnl.core.components.ports.outputport,
                                 pnl.core.components.ports.modulatorysignals.controlsignal,
                                 pnl.core.components.ports.modulatorysignals.gatingsignal,
                                 # Mechanisms
                                 # FIX 5/8/20 ELIMINATE SYSTEM [JDC] -- REFERENCES TO LABELS REQUIRE REFACTORING
                                 # pnl.core.components.mechanisms.mechanism,
                                 pnl.core.components.mechanisms.processing.transfermechanism,
                                 pnl.core.components.mechanisms.processing.integratormechanism,
                                 pnl.core.components.mechanisms.processing.objectivemechanism,
                                 pnl.core.components.mechanisms.modulatory.control.controlmechanism,
                                 # Function
                                 pnl.core.components.functions.function,
                                ])
def test_core_docs(mod, capsys):
    fail, total = doctest.testmod(mod)
    if fail > 0:
        captured = capsys.readouterr()
        pytest.fail("{} out of {} examples failed:\n{}\n{}".format(
            fail, total, captured.err, captured.out), pytrace=False)

@pytest.mark.parametrize("mod", [# Mechanisms
                                 pnl.library.components.mechanisms.processing.integrator.ddm,
                                 pnl.library.components.mechanisms.processing.objective.comparatormechanism,
                                 # Scheduling
                                 pnl.core.scheduling.scheduler,
                                 # Logs
                                 pnl.core.globals.log,
                                ])
def test_other_docs(mod, capsys):
    fail, total = doctest.testmod(mod, optionflags=doctest.REPORT_NDIFF)
    if fail > 0:
        captured = capsys.readouterr()
        pytest.fail("{} out of {} examples failed:\n{}\n{}".format(
            fail, total, captured.err, captured.out), pytrace=False)
