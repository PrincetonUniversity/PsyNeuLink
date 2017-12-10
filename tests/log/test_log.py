import pytest

import psyneulink as pnl
import numpy as np

class TestLog:

    def test_log(self):

        T_1 = pnl.TransferMechanism(name='T_1', size=2)
        T_2 = pnl.TransferMechanism(name='T_2', size=2)
        PS = pnl.Process(name='PS', pathway=[T_1, T_2])
        PJ = T_2.path_afferents[0]

        assert T_1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF',
                                     'value': 'OFF'}
        assert T_2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF',
                                     'value': 'OFF'}
        assert PJ.loggable_items == {'matrix': 'OFF',
                                     'value': 'OFF'}

        T_1.log_items(pnl.NOISE)
        T_1.log_items(pnl.RESULTS)
        PJ.log_items(pnl.MATRIX)

        assert T_1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'EXECUTION',
                                     'time_constant': 'OFF',
                                     'value': 'OFF'}
        assert T_2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'time_constant': 'OFF',
                                     'value': 'OFF'}
        assert PJ.loggable_items == {'matrix': 'EXECUTION',
                                     'value': 'OFF'}

        PS.execute()
        PS.execute()
        PS.execute()

        assert T_1.logged_items == {'RESULTS': 'EXECUTION', 'noise': 'EXECUTION'}
        assert PJ.logged_items == {'matrix': 'EXECUTION'}

        # assert T_1.log.print_entries() ==
        # # Log for mech_A:
        # #
        # # Index     Variable:                                          Context                                                                  Value
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
        # # Index     Variable:                                          Context                                                                  Value
        # # 0         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # #
        # #
        # # 0         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0

        print(T_1.log.csv(entries=['noise', 'RESULTS'], owner_name=False, quotes=None))
        assert T_1.log.csv(entries=['noise', 'RESULTS'], owner_name=False, quotes=None) == \
                        "\'Index\', \'noise\', \'RESULTS\'\n0, 0.0, 0.0 0.0\n1, 0.0, 0.0 0.0\n2, 0.0, 0.0 0.0\n"

        assert PJ.log.csv(entries='matrix', owner_name=True, quotes=True) == \
               "\'Index\', \'MappingProjection from T_1 to T_2[matrix]\'\n" \
               "\'0\', \'1.0 0.0\' \'0.0 1.0\'\n" \
               "\'1\', \'1.0 0.0\' \'0.0 1.0\'\n" \
               "\'2\', \'1.0 0.0\' \'0.0 1.0\'\n"

        result = T_1.log.nparray(entries=['noise', 'RESULTS'], header=False, owner_name=True)
        np.testing.assert_array_equal(result,
                                      np.array([[[0], [1], [2]],
                                                [[ 0.], [ 0.], [ 0.]],
                                                [[ 0.,  0.], [ 0.,  0.],[ 0., 0.]]]))





