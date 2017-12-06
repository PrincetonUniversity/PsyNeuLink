import pytest

import psyneulink as pnl
import numpy as np

class TestLog:

    def test_log(self):

        T_1 = pnl.TransferMechanism(name='T_1', size=2)
        T_2 = pnl.TransferMechanism(name='T_2', size=2)
        Ps = pnl.Process(name='Ps', pathway=[T_1, T_2])
        Pj = T_2.path_afferents[0]

        assert T_1.loggable_items == {'Ps_Input Projection': 'OFF',
                                     'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF'}
        assert T_2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'MappingProjection from T_1 to T_2': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF'}

        T_1.log_items(pnl.NOISE)
        T_1.log_items(pnl.RESULTS)
        T_2.log_items(Pj)

        assert T_1.loggable_items == {'Ps_Input Projection': 'OFF',
                                     'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'EXECUTION',
                                     'time_constant': 'OFF'}
        assert T_2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'MappingProjection from T_1 to T_2': 'EXECUTION',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF'}

        Ps.execute()
        Ps.execute()
        Ps.execute()

        assert T_1.logged_items == {'RESULTS': 'EXECUTION', 'noise': 'EXECUTION'}
        assert T_2.logged_items == {'MappingProjection from T_1 to T_2': 'EXECUTION'}

        # assert T_1.log.print_entries() ==
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
        # assert T_2.log.print_entries() ==
        # # Log for mech_A:
        # #
        # # Entry     Variable:                                          Context                                                                 Value
        # # 0         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # #
        # #
        # # 0         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0

        assert T_1.log.csv(entries=['noise', 'RESULTS'], owner_name=False, quotes=None) == \
                        "\'Entry\', \'noise\', \'RESULTS\'\n0,  0.,  0.  0.\n1,  0.,  0.  0.\n2,  0.,  0.  0.\n"

        assert T_2.log.csv(entries=Pj, owner_name=True, quotes=True) == \
               "\'Entry\', 'T_2[MappingProjection from T_1 to T_2]\'\n" \
               "0, \' 1.  0.\'\n \' 0.  1.\'\n" \
               "1, \' 1.  0.\'\n \' 0.  1.\'\n" \
               "2, \' 1.  0.\'\n \' 0.  1.\'\n"

        # assert T_1.log.nparray(entries=['noise', 'RESULTS'], header=False, owner_name=True) == \
        #        np.array([[[0], [1], [2]], [[ 0.], [ 0.], [ 0.]], [[ 0.,  0.], [ 0.,  0.],[ 0.,  0.]]])





