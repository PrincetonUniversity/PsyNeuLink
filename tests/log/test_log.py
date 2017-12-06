import pytest

import psyneulink as pnl
import numpy as np

class TestLog:

    def test_log(self):

        T1 = pnl.TransferMechanism(name='T1', size=2)
        T2 = pnl.TransferMechanism(name='T2', size=2)
        Ps = pnl.Process(pathway=[T1, T2])
        Pj = T2.path_afferents[0]

        assert T1.loggable_items == {'Process-0_Input Projection': 'OFF',
                                     'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF'}
        assert T2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'MappingProjection from T1 to T2': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF'}

        T1.log_items(pnl.NOISE)
        T1.log_items(pnl.RESULTS)
        T2.log_items(Pj)

        assert T1.loggable_items == {'Process-0_Input Projection': 'OFF',
                                     'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'EXECUTION',
                                     'time_constant': 'OFF'}
        assert T2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'MappingProjection from T1 to T2': 'EXECUTION',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF'}

        Ps.execute()
        Ps.execute()
        Ps.execute()

        assert T1.logged_items == {'RESULTS': 'EXECUTION', 'noise': 'EXECUTION'}
        assert T2.logged_items == {'MappingProjection from T1 to T2': 'EXECUTION'}

        # assert T1.log.print_entries() ==
        # # Log for mech_A:
        # #
        # # Entry     Variable:                                          Context                                                                 Value
        # # 0         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # #
        # #
        # # 0         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        #
        # assert T2.log.print_entries() ==
        # # Log for mech_A:
        # #
        # # Entry     Variable:                                          Context                                                                 Value
        # # 0         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # #
        # #
        # # 0         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0

        assert T1.log.csv(entries=['noise', 'RESULTS'], owner_name=False, quotes=None) == \
                        "\'Entry\', \'noise\', \'RESULTS\'\n0,  0.,  0.  0.\n1,  0.,  0.  0.\n2,  0.,  0.  0.\n"

        assert T2.log.csv(entries=Pj, owner_name=True, quotes=True) == \
                "\'Entry\', \'MappingProjection from T1 to T2\'\n0,  1.\n1,  1."

        assert T1.log.nparray(entries=['noise', 'RESULTS'], header=False, owner_name=True) == \
               np.array([[[0], [1], [2]], [[ 0.], [ 0.], [ 0.]], [[ 0.,  0.], [ 0.,  0.],[ 0.,  0.]]])

