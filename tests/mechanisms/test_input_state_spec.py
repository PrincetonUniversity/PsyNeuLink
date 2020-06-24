import numpy as np
import psyneulink as pnl
import pytest
import re

from psyneulink.core.components.functions.combinationfunctions import Reduce
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.mechanisms.mechanism import MechanismError
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import ProjectionError
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.port import PortError
from psyneulink.core.globals.keywords import FUNCTION, INPUT_PORTS, MECHANISM, NAME, OUTPUT_PORTS, PROJECTIONS, RESULT, VARIABLE

mismatches_specified_default_variable_error_text = 'not compatible with its specified default variable'
mismatches_default_variable_format_error_text = 'is not compatible with its expected format'
mismatches_size_error_text = 'not compatible with the default variable determined from size parameter'
mismatches_more_input_ports_than_default_variable_error_text = 'There are more InputPorts specified'
mismatches_fewer_input_ports_than_default_variable_error_text = 'There are fewer InputPorts specified'
mismatches_specified_matrix_pattern = r'The number of rows \(\d\) of the matrix provided for .+ does not equal the length \(\d\) of the sender vector'


class TestInputPortSpec:
    # ------------------------------------------------------------------------------------------------

    # InputPort SPECIFICATIONS

    # ------------------------------------------------------------------------------------------------
    # TEST 1a
    # Match of default_variable and specification of multiple InputPorts by value and string

    def test_match_with_default_variable(self):

        T = TransferMechanism(
            default_variable=[[0, 0], [0]],
            input_ports=[[32, 24], 'HELLO']
        )
        assert T.defaults.variable.shape == np.array([[0, 0], [0]], dtype=object).shape
        assert len(T.input_ports) == 2
        assert T.input_ports[1].name == 'HELLO'
        # # PROBLEM WITH input FOR RUN:
        # my_mech_2.execute()

    # ------------------------------------------------------------------------------------------------
    # # TEST 1b
    # # Match of default_variable and specification of multiple InputPorts by value and string
    #
    # def test_match_with_default_variable(self):
    #
    #     T = TransferMechanism(
    #         default_variable=[[0], [0]],
    #         input_ports=[[32, 24], 'HELLO']
    #     )
    #     assert T.defaults.variable.shape == np.array([[0, 0], [0]], dtype=object).shape
    #     assert len(T.input_ports) == 2
    #     assert T.input_ports[1].name == 'HELLO'

    # # ------------------------------------------------------------------------------------------------
    # # TEST 2
    # # Mismatch between InputPort variable specification and corresponding item of owner Mechanism's variable
    #
    # # Deprecated this test as length of variable of each InputPort should be allowed to vary from
    # # corresponding item of Mechanism's default variable, so long as each is 1d and then number of InputPorts
    # # is consistent with number of items in Mechanism's default_variable (i.e., its length in axis 0).
    #
    # def test_mismatch_with_default_variable_error(self):
    #
    #     with pytest.raises(InputPortError) as error_text:
    #         TransferMechanism(
    #             default_variable=[[0], [0]],
    #             input_ports=[[32, 24], 'HELLO']
    #         )
    #     assert mismatches_default_variable_format_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 2a
    # Mismatch between InputPort variable specification and corresponding item of owner Mechanism's variable

    # Replacement for original TEST 2, which insures that the number InputPorts specified corresponds to the
    # number of items in the Mechanism's default_variable (i.e., its length in axis 0).
    def test_fewer_input_ports_than_default_variable_error(self):

        with pytest.raises(PortError) as error_text:
            TransferMechanism(
                default_variable=[[0], [0]],
                input_ports=['HELLO']
            )
        assert mismatches_fewer_input_ports_than_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 2b
    # Mismatch between InputPort variable specification and corresponding item of owner Mechanism's variable

    # Replacement for original TEST 2, which insures that the number InputPorts specified corresponds to the
    # number of items in the Mechanism's default_variable (i.e., its length in axis 0).
    def test_more_input_ports_than_default_variable_error(self):

        with pytest.raises(PortError) as error_text:
            TransferMechanism(
                default_variable=[[0], [0]],
                input_ports=[[32], [24], 'HELLO']
            )
        assert mismatches_more_input_ports_than_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 2c
    # Mismatch between InputPort variable specification and corresponding item of owner Mechanism's variable

    # Replacement for original TEST 2, which insures that the number InputPorts specified corresponds to the
    # number of items in the Mechanism's default_variable (i.e., its length in axis 0).
    def test_mismatch_num_input_ports_with_default_variable_error(self):

        with pytest.raises(MechanismError) as error_text:
            TransferMechanism(
                default_variable=[[0], [0]],
                input_ports=[[32]]
            )
        assert mismatches_specified_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 2d
    # Mismatch between dimensionality of InputPort variable owner Mechanism's variable

    # FIX: This needs to be handled better in Port._parse_port_spec (~Line 3018):
    #      seems to be adding the two axis2 values
    def test_mismatch_dim_input_ports_with_default_variable_error(self):

        with pytest.raises(PortError) as error_text:
            TransferMechanism(
                default_variable=[[0], [0]],
                input_ports=[[[32],[24]],'HELLO']
            )
        assert 'Port value' in str(error_text.value) and 'does not match reference_value' in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 3
    # Override of input_ports (mis)specification by INPUT_PORTS entry in params specification dict

    def test_override_by_dict_spec(self):

        T = TransferMechanism(
            default_variable=[[0, 0], [0]],
            input_ports=[[32], 'HELLO'],
            params={INPUT_PORTS: [[32, 24], 'HELLO']}
        )
        assert T.defaults.variable.shape == np.array([[0, 0], [0]], dtype=object).shape
        assert len(T.input_ports) == 2
        assert T.input_ports[1].name == 'HELLO'
        # # PROBLEM WITH input FOR RUN:
        # my_mech_2.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 4
    # Specification using input_ports without default_variable

    def test_transfer_mech_input_ports_no_default_variable(self):

        # PROBLEM: SHOULD GENERATE TWO INPUT_PORTS (
        #                ONE WITH [[32],[24]] AND OTHER WITH [[0]] AS VARIABLE INSTANCE DEFAULT
        #                INSTEAD, SEEM TO IGNORE InputPort SPECIFICATIONS AND JUST USE DEFAULT_VARIABLE
        #                NOTE:  WORKS FOR ObjectiveMechanism, BUT NOT TransferMechanism
        T = TransferMechanism(input_ports=[[32, 24], 'HELLO'])
        assert T.defaults.variable.shape == np.array([[0, 0], [0]], dtype=object).shape
        assert len(T.input_ports) == 2
        assert T.input_ports[1].name == 'HELLO'

    # ------------------------------------------------------------------------------------------------
    # TEST 5
    # Specification using INPUT_PORTS entry in params specification dict without default_variable

    def test_transfer_mech_input_ports_specification_dict_no_default_variable(self):

        # PROBLEM: SHOULD GENERATE TWO INPUT_PORTS (
        #                ONE WITH [[32],[24]] AND OTHER WITH [[0]] AS VARIABLE INSTANCE DEFAULT
        #                INSTEAD, SEEM TO IGNORE InputPort SPECIFICATIONS AND JUST USE DEFAULT_VARIABLE
        #                NOTE:  WORKS FOR ObjectiveMechanism, BUT NOT TransferMechanism
        T = TransferMechanism(params={INPUT_PORTS: [[32, 24], 'HELLO']})
        assert T.defaults.variable.shape == np.array([[0, 0], [0]], dtype=object).shape
        assert len(T.input_ports) == 2
        assert T.input_ports[1].name == 'HELLO'

    # ------------------------------------------------------------------------------------------------
    # TEST 6
    # Mechanism specification

    def test_mech_spec_list(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[[0]],
            input_ports=[R1]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0]]))
        assert len(T.input_ports) == 1
        assert T.input_port.path_afferents[0].sender == R1.output_port
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 7
    # Mechanism specification outside of a list

    def test_mech_spec_standalone(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        # Mechanism outside of list specification
        T = TransferMechanism(
            default_variable=[[0]],
            input_ports=R1
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0]]))
        assert len(T.input_ports) == 1
        assert T.input_port.path_afferents[0].sender == R1.output_port
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 8
    # OutputPort specification

    def test_output_port_spec_list_two_items(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[[0], [0]],
            input_ports=[
                R1.output_ports['FIRST'],
                R1.output_ports['SECOND']
            ]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0], [0]]))
        assert len(T.input_ports) == 2
        assert T.input_ports.names[0] == 'InputPort-0'
        assert T.input_ports.names[1] == 'InputPort-1'
        for input_port in T.input_ports:
            for projection in input_port.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 9
    # OutputPort specification outside of a list

    def test_output_port_spec_standalone(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[0],
            input_ports=R1.output_ports['FIRST']
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0]]))
        assert len(T.input_ports) == 1
        assert T.input_ports.names[0] == 'InputPort-0'
        T.input_port.path_afferents[0].sender == R1.output_port
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 10
    # OutputPorts in PROJECTIONS entries of a specification dictiontary, using with names (and one outside of a list)

    def test_specification_dict(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[[0], [0]],
            input_ports=[
                {
                    NAME: 'FROM DECISION',
                    PROJECTIONS: [R1.output_ports['FIRST']]
                },
                {
                    NAME: 'FROM RESPONSE_TIME',
                    PROJECTIONS: R1.output_ports['SECOND']
                }
            ])
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0], [0]]))
        assert len(T.input_ports) == 2
        assert T.input_ports.names[0] == 'FROM DECISION'
        assert T.input_ports.names[1] == 'FROM RESPONSE_TIME'
        for input_port in T.input_ports:
            for projection in input_port.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 11
    # default_variable override of value of OutputPort specification

    def test_default_variable_override_mech_list(self):

        R2 = TransferMechanism(size=3)

        # default_variable override of OutputPort.value
        T = TransferMechanism(
            default_variable=[[0, 0]],
            input_ports=[R2]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1
        assert len(T.input_port.path_afferents[0].sender.defaults.variable) == 3
        assert len(T.input_port.defaults.variable[0]) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 12
    # 2-item tuple specification with default_variable override of OutputPort.value

    def test_2_item_tuple_spec(self):
        R2 = TransferMechanism(size=3)
        T = TransferMechanism(size=2, input_ports=[(R2, np.zeros((3, 2)))])
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1
        assert len(T.input_port.path_afferents[0].sender.defaults.variable) == 3
        assert T.input_port.socket_width == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 12.1
    # 2-item tuple specification with value as first item (and no size specification for T)

    def test_2_item_tuple_value_for_first_item(self):
        R2 = TransferMechanism(size=3)
        T = TransferMechanism(input_ports=[([0,0], R2)])
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1
        assert T.input_port.path_afferents[0].sender.socket_width == 3
        assert T.input_port.socket_width == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 13
    # 4-item tuple Specification

    def test_projection_tuple_with_matrix_spec(self):
        R2 = TransferMechanism(size=3)
        T = TransferMechanism(size=2, input_ports=[(R2, None, None, np.zeros((3, 2)))])
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1
        assert T.input_port.path_afferents[0].sender.defaults.variable.shape[-1] == 3
        assert T.input_port.socket_width == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 14
    # Standalone Projection specification

    def test_projection_list(self):
        R2 = TransferMechanism(size=3)
        P = MappingProjection(sender=R2)
        T = TransferMechanism(
            size=2,
            input_ports=[P]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1
        assert len(T.input_port.path_afferents[0].sender.defaults.variable) == 3
        assert len(T.input_port.defaults.variable) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 15
    # Projection specification in Tuple

    def test_projection_in_tuple(self):
        R2 = TransferMechanism(size=3)
        P = MappingProjection(sender=R2)
        T = TransferMechanism(
            size=2,
            input_ports=[(R2, None, None, P)]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1
        assert len(T.input_port.path_afferents[0].sender.defaults.variable) == 3
        assert len(T.input_port.defaults.variable) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 16
    # PROJECTIONS specification in InputPort specification dictionary

    def test_projection_in_specification_dict(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(
            input_ports=[{
                NAME: 'My InputPort with Two Projections',
                PROJECTIONS: [
                    R1.output_ports['FIRST'],
                    R1.output_ports['SECOND']
                ]
            }]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0]]))
        assert len(T.input_ports) == 1
        assert T.input_port.name == 'My InputPort with Two Projections'
        for input_port in T.input_ports:
            for projection in input_port.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 17
    # MECHANISMS/OUTPUT_PORTS entries in params specification dict

    def test_output_port_in_specification_dict(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(
            input_ports=[{
                MECHANISM: R1,
                OUTPUT_PORTS: ['FIRST', 'SECOND']
            }]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0]]))
        assert len(T.input_ports) == 1
        for input_port in T.input_ports:
            for projection in input_port.path_afferents:
                assert projection.sender.owner is R1

    # ------------------------------------------------------------------------------------------------
    # TEST 18
    # String specification with variable specification

    def test_dict_with_variable(self):
        T = TransferMechanism(input_ports=[{NAME: 'FIRST', VARIABLE: [0, 0]}])
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 19
    # String specification with variable specification conflicts with default_variable

    def test_dict_with_variable_matches_default(self):
        T = TransferMechanism(
            default_variable=[0, 0],
            input_ports=[{NAME: 'FIRST', VARIABLE: [0, 0]}]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 20

    def test_dict_with_variable_matches_default_2(self):
        T = TransferMechanism(
            default_variable=[[0, 0]],
            input_ports=[{NAME: 'FIRST', VARIABLE: [0, 0]}]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 21

    def test_dict_with_variable_matches_default_multiple_input_ports(self):
        T = TransferMechanism(
            default_variable=[[0, 0], [0]],
            input_ports=[
                {NAME: 'FIRST', VARIABLE: [0, 0]},
                {NAME: 'SECOND', VARIABLE: [0]}
            ]
        )
        assert T.defaults.variable.shape == np.array([[0, 0], [0]], dtype=object).shape
        assert len(T.input_ports) == 2

    # ------------------------------------------------------------------------------------------------
    # TEST 22

    def test_dict_with_variable_mismatches_default(self):
        with pytest.raises(MechanismError) as error_text:
            TransferMechanism(
                default_variable=[[0]],
                input_ports=[{NAME: 'FIRST', VARIABLE: [0, 0]}]
            )
        assert mismatches_specified_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 23

    def test_dict_with_variable_mismatches_default_multiple_input_ports(self):
        with pytest.raises(MechanismError) as error_text:
            TransferMechanism(
                default_variable=[[0], [0]],
                input_ports=[
                    {NAME: 'FIRST', VARIABLE: [0, 0]},
                    {NAME: 'SECOND', VARIABLE: [0]}
                ]
            )
        assert mismatches_specified_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 24

    def test_dict_with_variable_matches_size(self):
        T = TransferMechanism(
            size=2,
            input_ports=[{NAME: 'FIRST', VARIABLE: [0, 0]}]
        )
        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 25

    def test_dict_with_variable_mismatches_size(self):
        with pytest.raises(MechanismError) as error_text:
            TransferMechanism(
                size=1,
                input_ports=[{NAME: 'FIRST', VARIABLE: [0, 0]}]
            )
        assert mismatches_size_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 26

    def test_params_override(self):
        T = TransferMechanism(
            input_ports=[[0], [0]],
            params={INPUT_PORTS: [[0, 0], [0]]}
        )
        assert T.defaults.variable.shape == np.array([[0, 0], [0]], dtype=object).shape
        assert len(T.input_ports) == 2

    # ------------------------------------------------------------------------------------------------
    # TEST 28

    def test_inputPort_class(self):
        T = TransferMechanism(input_ports=[InputPort])

        np.testing.assert_array_equal(T.defaults.variable, [InputPort.defaults.variable])
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 29

    def test_inputPort_class_with_variable(self):
        T = TransferMechanism(default_variable=[[0, 0]], input_ports=[InputPort])

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 30

    def test_InputPort_mismatches_default(self):
        with pytest.raises(MechanismError) as error_text:
            i = InputPort(reference_value=[0, 0, 0])
            TransferMechanism(default_variable=[0, 0], input_ports=[i])
        assert mismatches_specified_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 31

    def test_projection_with_matrix_and_sender(self):
        m = TransferMechanism(size=2)
        p = MappingProjection(sender=m, matrix=[[0, 0, 0], [0, 0, 0]])
        T = TransferMechanism(input_ports=[p])

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 32

    def tests_for_projection_with_matrix_and_sender_mismatches_default(self):
        with pytest.raises(MechanismError) as error_text:
            m = TransferMechanism(size=2)
            p = MappingProjection(sender=m, matrix=[[0, 0, 0], [0, 0, 0]])
            TransferMechanism(default_variable=[0, 0], input_ports=[p])
        assert mismatches_specified_default_variable_error_text in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            m = TransferMechanism(size=3, output_ports=[pnl.MEAN])
            p = MappingProjection(sender=m, matrix=[[0,0,0], [0,0,0]])
            T = TransferMechanism(input_ports=[p])
        assert re.match(
            mismatches_specified_matrix_pattern,
            str(error_text.value)
        )

        with pytest.raises(FunctionError) as error_text:
            m2 = TransferMechanism(size=2, output_ports=[pnl.MEAN])
            p2 = MappingProjection(sender=m2, matrix=[[1,1,1],[1,1,1]])
            T2 = TransferMechanism(input_ports=[p2])
        assert re.match(
            mismatches_specified_matrix_pattern,
            str(error_text.value)
        )

    # ------------------------------------------------------------------------------------------------
    # TEST 33

    def test_projection_with_sender_and_default(self):
        t = TransferMechanism(size=3)
        p = MappingProjection(sender=t)
        T = TransferMechanism(default_variable=[[0, 0]], input_ports=[p])

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 34

    def test_projection_no_args_projection_spec(self):
        p = MappingProjection()
        T = TransferMechanism(input_ports=[p])

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 35

    def test_projection_no_args_projection_spec_with_default(self):
        p = MappingProjection()
        T = TransferMechanism(default_variable=[[0, 0]], input_ports=[p])

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 36

    def test_projection_no_args_dict_spec(self):
        p = MappingProjection()
        T = TransferMechanism(input_ports=[{VARIABLE: [0, 0, 0], PROJECTIONS:[p]}])

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0, 0]]))
        assert len(T.input_ports) == 1

    # ------------------------------------------------------------------------------------------------
    # TEST 37

    def test_projection_no_args_dict_spec_mismatch_with_default(self):
        with pytest.raises(MechanismError) as error_text:
            p = MappingProjection()
            TransferMechanism(default_variable=[0, 0], input_ports=[{VARIABLE: [0, 0, 0], PROJECTIONS: [p]}])
        assert mismatches_specified_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 38

    def test_outputPort_(self):
        with pytest.raises(MechanismError) as error_text:
            p = MappingProjection()
            TransferMechanism(default_variable=[0, 0], input_ports=[{VARIABLE: [0, 0, 0], PROJECTIONS: [p]}])
        assert mismatches_specified_default_variable_error_text in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 26

    def test_add_input_port_with_projection_in_mech_constructor(self):
        T1 = TransferMechanism()
        I = InputPort(projections=[T1])
        T2 = TransferMechanism(input_ports=[I])
        assert T2.input_ports[0].path_afferents[0].sender.owner is T1

    # ------------------------------------------------------------------------------------------------
    # TEST 27

    def test_add_input_port_with_projection_using_add_ports(self):
        T1 = TransferMechanism()
        I = InputPort(projections=[T1])
        T2 = TransferMechanism()
        T2.add_ports([I])
        assert T2.input_ports[1].path_afferents[0].sender.owner is T1

    # ------------------------------------------------------------------------------------------------
    # TEST 28

    def test_add_input_port_with_projection_by_assigning_owner(self):
        T1 = TransferMechanism()
        T2 = TransferMechanism()
        InputPort(owner=T2, projections=[T1])

        assert T2.input_ports[1].path_afferents[0].sender.owner is T1
    # ------------------------------------------------------------------------------------------------
    # TEST 29

    def test_add_input_port_with_projection_by_assigning_owner_error(self):
        with pytest.raises(PortError) as error_text:
            S1 = TransferMechanism()
            S2 = TransferMechanism()
            TransferMechanism(name='T',
                              input_ports=[{'MY INPUT 1':[S1],
                                             'MY INPUT 2':[S2]}])

    # ------------------------------------------------------------------------------------------------
    # TEST 30

    def test_use_set_to_specify_projections_for_input_port_error(self):
        with pytest.raises(ProjectionError) as error_text:
            T1 = TransferMechanism()
            T2 = TransferMechanism()
            TransferMechanism(input_ports=[{'MY PORT':{T1, T2}}])
        assert ('Connection specification for InputPort of' in str(error_text.value)
                and 'is a set' in str(error_text.value)
                and 'it should be a list' in str(error_text.value))

    # ------------------------------------------------------------------------------------------------
    # TEST 31

    def test_multiple_states_specified_using_port_Name_format_error(self):
        with pytest.raises(PortError) as error_text:
            # Don't bother to specify anything as the value for each entry in the dict, since doesn't get there
            TransferMechanism(input_ports=[{'MY PORT A':{},
                                             'MY PORT B':{}}])
        assert ('There is more than one entry of the InputPort specification dictionary' in str(error_text.value)
                and'that is not a keyword; there should be only one (used to name the Port, with a list of '
                   'Projection specifications' in str(error_text.value))

    # ------------------------------------------------------------------------------------------------
    # TEST 32

    def test_default_name_and_projections_listing_for_input_port_in_constructor(self):
        T1 = TransferMechanism()
        my_input_port = InputPort(projections=[T1])
        T2 = TransferMechanism(input_ports=[my_input_port])
        assert T2.input_ports[0].name == 'InputPort-0'
        assert T2.input_ports[0].projections[0].sender.name == 'RESULT'

    # ------------------------------------------------------------------------------------------------
    # TEST 33

    def test_2_item_tuple_with_port_Name_list_and_mechanism(self):

        # T1 has OutputPorts of with same lengths,
        #    so T2 should use that length for its InputPort variable (since it is not otherwise specified)
        T1 = TransferMechanism(input_ports=[[0,0],[0,0]])
        T2 = TransferMechanism(input_ports=[(['RESULT-0', 'RESULT-1'], T1)])
        assert len(T2.input_ports[0].value) == 2
        assert T2.input_ports[0].path_afferents[0].sender.name == 'RESULT-0'
        assert T2.input_ports[0].path_afferents[1].sender.name == 'RESULT-1'

        # T1 has OutputPorts with different lengths both of which are specified by T2 to project to a singe InputPort,
        #    so T2 should use its variable default as format for the InputPort (since it is not otherwise specified)
        T1 = TransferMechanism(input_ports=[[0,0],[0,0,0]])
        T2 = TransferMechanism(input_ports=[(['RESULT-0', 'RESULT-1'], T1)])
        assert len(T2.input_ports[0].value) == 1
        assert T2.input_ports[0].path_afferents[0].sender.name == 'RESULT-0'
        assert T2.input_ports[0].path_afferents[1].sender.name == 'RESULT-1'

    # ------------------------------------------------------------------------------------------------
    # TEST 34

    def test_lists_of_mechanisms_and_output_ports(self):

        # Test "bare" list of Mechanisms
        T0 = TransferMechanism(name='T0')
        T1 = TransferMechanism(name='T1', input_ports=[[0,0],[0,0,0]])
        T2 = TransferMechanism(name='T2', input_ports=[[T0, T1]])
        assert len(T2.input_ports[0].path_afferents)==2
        assert T2.input_ports[0].path_afferents[0].sender.owner.name=='T0'
        assert T2.input_ports[0].path_afferents[1].sender.owner.name=='T1'
        assert T2.input_ports[0].path_afferents[1].matrix.shape == (2,1)

        # Test list of Mechanisms in 4-item tuple specification
        T3 = TransferMechanism(name='T3', input_ports=[([T0, T1], None, None, InputPort)])
        assert len(T3.input_ports[0].path_afferents)==2
        assert T3.input_ports[0].path_afferents[0].sender.owner.name=='T0'
        assert T3.input_ports[0].path_afferents[1].sender.owner.name=='T1'
        assert T3.input_ports[0].path_afferents[1].matrix.shape == (2,1)

        # Test "bare" list of OutputPorts
        T4= TransferMechanism(name='T4', input_ports=[[T0.output_ports[0], T1.output_ports[1]]])
        assert len(T4.input_ports[0].path_afferents)==2
        assert T4.input_ports[0].path_afferents[0].sender.owner.name=='T0'
        assert T4.input_ports[0].path_afferents[1].sender.owner.name=='T1'
        assert T4.input_ports[0].path_afferents[1].matrix.shape == (3,1)

        # Test list of OutputPorts in 4-item tuple specification
        T5 = TransferMechanism(name='T5', input_ports=[([T0.output_ports[0], T1.output_ports[1]],
                                                         None, None,
                                                         InputPort)])
        assert len(T5.input_ports[0].path_afferents)==2
        assert T5.input_ports[0].path_afferents[0].sender.owner.name=='T0'
        assert T5.input_ports[0].path_afferents[1].sender.owner.name=='T1'
        assert T5.input_ports[0].path_afferents[1].matrix.shape == (3,1)

    # ------------------------------------------------------------------------------------------------
    # TEST 35

    def test_list_of_mechanisms_with_gating_mechanism(self):

        T1 = TransferMechanism(name='T6')
        G = GatingMechanism(gating_signals=['a','b'])
        T2 = TransferMechanism(input_ports=[[T1, G]],
                               output_ports=[G.gating_signals['b']])
        assert T2.input_ports[0].path_afferents[0].sender.owner.name=='T6'
        assert T2.input_ports[0].mod_afferents[0].sender.name=='a'
        assert T2.output_ports[0].mod_afferents[0].sender.name=='b'

    # ------------------------------------------------------------------------------------------------
    # THOROUGH TESTING OF mech, 2-item, 3-item and 4-item tuple specifications with and without default_variable/size
    # (some of these may be duplicative of tests above)

    # pytest does not support fixtures in parametrize, but a class member is enough for this test
    transfer_mech = TransferMechanism(size=3)

    @pytest.mark.parametrize('default_variable, size, input_ports, variable_len_state, variable_len_mech', [
        # default_variable tests
        ([0, 0], None, [transfer_mech], 2, 2),
        ([0, 0], None, [(transfer_mech, None)], 2, 2),
        ([0, 0], None, [(transfer_mech, 1, 1)], 2, 2),
        ([0, 0], None, [((RESULT, transfer_mech), 1, 1)], 2, 2),
        ([0, 0], None, [(transfer_mech, 1, 1, None)], 2, 2),
        # size tests
        (None, 2, [transfer_mech], 2, 2),
        (None, 2, [(transfer_mech, None)], 2, 2),
        (None, 2, [(transfer_mech, 1, 1)], 2, 2),
        (None, 2, [(transfer_mech, 1, 1, None)], 2, 2),
        # no default_variable or size tests
        (None, None, [transfer_mech], 3, 3),
        (None, None, [(transfer_mech, None)], 3, 3),
        (None, None, [(transfer_mech, 1, 1)], 3, 3),
        (None, None, [(transfer_mech, 1, 1, None)], 3, 3),
        # tests of input ports with different variable and value shapes
        # ([[0,0]], None, [{VARIABLE: [[0], [0]], FUNCTION: LinearCombination}], 2, 2),
        # (None, 2, [{VARIABLE: [[0], [0]], FUNCTION: LinearCombination}], 2, 2),
        (None, 1, [{VARIABLE: [0, 0], FUNCTION: Reduce(weights=[1, -1])}], 2, 1),
        # (None, None, [transfer_mech], 3, 3),
        # (None, None, [(transfer_mech, None)], 3, 3),
        # (None, None, [(transfer_mech, 1, 1)], 3, 3),
        # (None, None, [(transfer_mech, 1, 1, None)], 3, 3),
        # # tests of input ports with different variable and value shapes
        # ([[0]], None, [{VARIABLE: [[0], [0]], FUNCTION: LinearCombination}], 2, 1),
        # (None, 1, [{VARIABLE: [0, 0], FUNCTION: Reduce(weights=[1, -1])}], 2, 1),
    ])
    def test_mech_and_tuple_specifications_with_and_without_default_variable_or_size(
        self,
        default_variable,
        size,
        input_ports,
        variable_len_state,
        variable_len_mech,
    ):
        # ADD TESTING WITH THIS IN PLACE OF transfer_mech:
        # p = MappingProjection(sender=transfer_mech)

        T = TransferMechanism(
            default_variable=default_variable,
            size=size,
            input_ports=input_ports
        )
        assert T.input_ports[0].socket_width == variable_len_state
        assert T.defaults.variable.shape[-1] == variable_len_mech

    def test_input_ports_arg_no_list(self):
        T = TransferMechanism(input_ports={VARIABLE: [0, 0, 0]})

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0, 0]]))
        assert len(T.input_ports) == 1

    def test_input_ports_params_no_list(self):
        T = TransferMechanism(params={INPUT_PORTS: {VARIABLE: [0, 0, 0]}})

        np.testing.assert_array_equal(T.defaults.variable, np.array([[0, 0, 0]]))
        assert len(T.input_ports) == 1
