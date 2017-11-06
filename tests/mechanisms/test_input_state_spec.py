import numpy as np
import pytest

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.states.inputstate import InputStateError
from psyneulink.globals.keywords import INPUT_STATES, MECHANISM, NAME, OUTPUT_STATES, PROJECTIONS


class TestInputStateSpec:
    # ------------------------------------------------------------------------------------------------

    # InputState SPECIFICATIONS

    # ------------------------------------------------------------------------------------------------
    # TEST 1
    # Match of default_variable and specification of multiple InputStates by value and string

    def test_match_with_default_variable(self):

        T = TransferMechanism(
            default_variable=[[0, 0], [0]],
            input_states=[[32, 24], 'HELLO']
        )
        assert T.instance_defaults.variable.shape == np.array([[0, 0], [0]]).shape
        assert len(T.input_states) == 2
        assert T.input_states[1].name == 'HELLO'
        # # PROBLEM WITH input FOR RUN:
        # my_mech_2.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 2
    # Mismatch between InputState variable specification and corresponding item of owner Mechanism's variable

    def test_mismatch_with_default_variable_error(self):

        with pytest.raises(InputStateError) as error_text:
            T = TransferMechanism(
                default_variable=[[0], [0]],
                input_states=[[32, 24], 'HELLO']
            )
        assert "Value specified for" in str(error_text.value) and "with its expected format" in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 3
    # Override of input_states (mis)specification by INPUT_STATES entry in params specification dict

    def test_override_by_dict_spec(self):

        T = TransferMechanism(
            default_variable=[[0, 0], [0]],
            input_states=[[32], 'HELLO'],
            params={INPUT_STATES: [[32, 24], 'HELLO']}
        )
        assert T.instance_defaults.variable.shape == np.array([[0, 0], [0]]).shape
        assert len(T.input_states) == 2
        assert T.input_states[1].name == 'HELLO'
        # # PROBLEM WITH input FOR RUN:
        # my_mech_2.execute()

    # # ------------------------------------------------------------------------------------------------
    # # TEST 4
    # # Specification using input_states without default_variable
    #
    # def test_transfer_mech_input_states_no_default_variable(self):
    #
    #     # PROBLEM: SHOULD GENERATE TWO INPUT_STATES (
    #     #                ONE WITH [[32],[24]] AND OTHER WITH [[0]] AS VARIABLE INSTANCE DEFAULT
    #     #                INSTEAD, SEEM TO IGNORE InputState SPECIFICATIONS AND JUST USE DEFAULT_VARIABLE
    #     #                NOTE:  WORKS FOR ObjectiveMechanism, BUT NOT TransferMechanism
    #     T = TransferMechanism(input_states=[[32, 24], 'HELLO'])
    #     assert T.instance_defaults.variable.shape == np.array([[0, 0], [0]]).shape
    #     assert len(T.input_states) == 2
    #     assert T.input_states[1].name == 'HELLO'

    # # ------------------------------------------------------------------------------------------------
    # # TEST 5
    # # Specification using INPUT_STATES entry in params specification dict without default_variable
    #
    # def test_transfer_mech_input_states_specification_dict_no_default_variable(self):
    #
    #     # PROBLEM: SHOULD GENERATE TWO INPUT_STATES (
    #     #                ONE WITH [[32],[24]] AND OTHER WITH [[0]] AS VARIABLE INSTANCE DEFAULT
    #     #                INSTEAD, SEEM TO IGNORE InputState SPECIFICATIONS AND JUST USE DEFAULT_VARIABLE
    #     #                NOTE:  WORKS FOR ObjectiveMechanism, BUT NOT TransferMechanism
    #     T = TransferMechanism(params = {INPUT_STATES:[[32, 24], 'HELLO']})
    #     assert T.instance_defaults.variable.shape == np.array([[0, 0], [0]]).shape
    #     assert len(T.input_states) == 2
    #     assert T.input_states[1].name == 'HELLO'

    # ------------------------------------------------------------------------------------------------
    # TEST 6
    # Mechanism specification

    def test_mech_spec_list(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[[0]],
            input_states=[R1]
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0]]))
        assert len(T.input_states) == 1
        assert T.input_state.path_afferents[0].sender == R1.output_state
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 7
    # Mechanism specification outside of a list

    def test_mech_spec_standalone(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        # Mechanism outside of list specification
        T = TransferMechanism(
            default_variable=[[0]],
            input_states=R1
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0]]))
        assert len(T.input_states) == 1
        assert T.input_state.path_afferents[0].sender == R1.output_state
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 8
    # OutputState specification

    def test_output_state_spec_list_two_items(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[[0], [0]],
            input_states=[
                R1.output_states['FIRST'],
                R1.output_states['SECOND']
            ]
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0], [0]]))
        assert len(T.input_states) == 2
        assert T.input_states.names[0] == 'InputState'
        assert T.input_states.names[1] == 'InputState-1'
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 9
    # OutputState specification outside of a list

    def test_output_state_spec_standalone(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[0],
            input_states=R1.output_states['FIRST']
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0]]))
        assert len(T.input_states) == 1
        assert T.input_states.names[0] == 'InputState'
        T.input_state.path_afferents[0].sender == R1.output_state
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 10
    # OutputStates in PROJECTIONS entries of a specification dictiontary, using with names (and one outside of a list)

    def test_specification_dict(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(
            default_variable=[[0], [0]],
            input_states=[
                {
                    NAME: 'FROM DECISION',
                    PROJECTIONS: [R1.output_states['FIRST']]
                },
                {
                    NAME: 'FROM RESPONSE_TIME',
                    PROJECTIONS: R1.output_states['SECOND']
                }
            ])
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0], [0]]))
        assert len(T.input_states) == 2
        assert T.input_states.names[0] == 'FROM DECISION'
        assert T.input_states.names[1] == 'FROM RESPONSE_TIME'
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 11
    # default_variable override of value of OutputState specification

    def test_default_variable_override_mech_list(self):

        R2 = TransferMechanism(size=3)

        # default_variable override of OutputState.value
        T = TransferMechanism(
            default_variable=[[0, 0]],
            input_states=[R2]
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0, 0]]))
        assert len(T.input_states) == 1
        assert len(T.input_state.path_afferents[0].sender.instance_defaults.variable) == 3
        assert len(T.input_state.instance_defaults.variable) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 12
    # 2-item tuple specification with default_variable override of OutputState.value

    def test_2_item_tuple_spec(self):
        R2 = TransferMechanism(size=3)
        T = TransferMechanism(size=2, input_states=[(R2, np.zeros((3, 2)))])
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0, 0]]))
        assert len(T.input_states) == 1
        assert len(T.input_state.path_afferents[0].sender.instance_defaults.variable) == 3
        assert len(T.input_state.instance_defaults.variable) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 13
    # ConnectionTuple Specification

    def test_connection_tuple_spec(self):
        R2 = TransferMechanism(size=3)
        T = TransferMechanism(size=2, input_states=[(R2, None, None, np.zeros((3, 2)))])
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0, 0]]))
        assert len(T.input_states) == 1
        assert len(T.input_state.path_afferents[0].sender.instance_defaults.variable) == 3
        assert len(T.input_state.instance_defaults.variable) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 14
    # Standalone Projection specification

    def test_projection_list(self):
        R2 = TransferMechanism(size=3)
        P = MappingProjection(sender=R2)
        T = TransferMechanism(
            size=2,
            input_states=[P]
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0, 0]]))
        assert len(T.input_states) == 1
        assert len(T.input_state.path_afferents[0].sender.instance_defaults.variable) == 3
        assert len(T.input_state.instance_defaults.variable) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 15
    # Projection specification in Tuple

    def test_projection_in_tuple(self):
        R2 = TransferMechanism(size=3)
        P = MappingProjection(sender=R2)
        T = TransferMechanism(
            size=2,
            input_states=[(R2, None, None, P)]
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0, 0]]))
        assert len(T.input_states) == 1
        assert len(T.input_state.path_afferents[0].sender.instance_defaults.variable) == 3
        assert len(T.input_state.instance_defaults.variable) == 2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 16
    # PROJECTIONS specification in InputState specification dictionary

    def test_projection_in_specification_dict(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(
            input_states=[{
                NAME: 'My InputState with Two Projections',
                PROJECTIONS: [
                    R1.output_states['FIRST'],
                    R1.output_states['SECOND']
                ]
            }]
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0]]))
        assert len(T.input_states) == 1
        assert T.input_state.name == 'My InputState with Two Projections'
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 17
    # MECHANISMS/OUTPUT_STATES entries in params specification dict

    def test_output_state_in_specification_dict(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(
            input_states=[{
                MECHANISM: R1,
                OUTPUT_STATES: ['FIRST', 'SECOND']
            }]
        )
        np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[0]]))
        assert len(T.input_states) == 1
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1
