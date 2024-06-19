import doctest
import re

import graph_scheduler
import pytest

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
                                 # Functions
                                 pnl.core.components.functions.function,
                                 pnl.core.components.functions.stateful.memoryfunctions
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
                                 pnl.library.components.mechanisms.processing.integrator.episodicmemorymechanism,
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


def consistent_doc_attrs(*objs):
    return (
        all(o.__doc__ is None for o in objs)
        or all(isinstance(o.__doc__, str) for o in objs)
    )


@pytest.mark.parametrize(
    'mod',
    [
        pnl.core.scheduling.scheduler,
        pnl.core.scheduling.condition,
        pnl.core.scheduling.time,
    ]
)
def test_scheduler_substitutions(mod):
    for cls, repls in mod._doc_subs.items():
        if cls is None:
            cls = mod
        else:
            cls = getattr(mod, cls)

        for pattern, repl in repls:
            # remove any group substitution strings
            assert re.sub(r'\\\d', '', repl) in cls.__doc__
            # where the pattern is being replaced, check that the pattern is gone
            if not re.match(r'\\\d', repl):
                assert not re.match(pattern, cls.__doc__)

    for pattern, repl in pnl.core.scheduling._global_doc_subs:
        for cls_name in mod.__all__:
            cls = getattr(mod, cls_name)
            try:
                ext_cls = getattr(graph_scheduler, cls_name)
            except AttributeError:
                continue

            assert consistent_doc_attrs(cls, ext_cls)
            # global replacements may not happen in every docstring
            if cls.__doc__ is not None:
                assert re.sub(r'\\\d', '', repl) in cls.__doc__ or not re.match(pattern, ext_cls.__doc__)

        ext_module = getattr(graph_scheduler, mod.__name__.split('.')[-1]).__doc__
        assert consistent_doc_attrs(mod, ext_module)
        assert re.sub(r'\\\d', '', repl) in mod.__doc__ or not re.match(pattern, ext_module.__doc__)
