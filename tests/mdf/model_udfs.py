import graph_scheduler
import modeci_mdf.functions.standard

import psyneulink as pnl

A = pnl.TransferMechanism(name='A')
B = pnl.ProcessingMechanism(
    name='B',
    function=pnl.UserDefinedFunction(
        modeci_mdf.functions.standard.mdf_functions['sin']['function'],
        scale=1
    )
)
C = pnl.ProcessingMechanism(
    name='C',
    function=pnl.UserDefinedFunction(
        modeci_mdf.functions.standard.mdf_functions['cos']['function'],
        scale=1
    )
)
comp = pnl.Composition(name='comp', pathways=[A, B, C])
comp.scheduler.add_condition(B, pnl.EveryNCalls(A, 2))
comp.scheduler.add_condition(C, graph_scheduler.EveryNCalls(B, 2))
