import psyneulink as pnl
import pytest
import numpy as np

class TestCompositionMethods:

    def test_get_output_values_prop(self):
        A = pnl.ProcessingMechanism()
        c = pnl.Composition()
        c.add_node(A)
        result = c.run(inputs={A: [1]}, num_trials=2)
        assert result == c.output_values == [np.array([1])]

    def test_add_pathway_methods_return_pathway(self):
        c = pnl.Composition()
        p = c.add_linear_processing_pathway(pathway=[pnl.ProcessingMechanism(), pnl.ProcessingMechanism()])
        assert isinstance(p, pnl.Pathway)

        c = pnl.Composition()
        p = c.add_linear_learning_pathway(pathway=[pnl.ProcessingMechanism(), pnl.ProcessingMechanism()],
                                          learning_function=pnl.BackPropagation)
        assert isinstance(p, pnl.Pathway)


    # test whether xor model created as autodiff composition learns properly
    @pytest.mark.pytorch
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    @pytest.mark.parametrize("minibatch_size", [
        1,
        2,
        3,
        4
    ])
    def test_learning_output_shape(self, mode, minibatch_size):
        '''
        Tests for correct output from composition.learn
        Expected: All results from last epoch
        '''
        xor_in = pnl.TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = pnl.TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=pnl.Logistic())

        xor_out = pnl.TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=pnl.Logistic())

        hid_map = pnl.MappingProjection(matrix=np.random.rand(2,10), sender=xor_in, receiver=xor_hid)
        out_map = pnl.MappingProjection(matrix=np.random.rand(10,1))

        xor = pnl.AutodiffComposition()

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0], [0, 1], [1, 0], [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0], [1], [1], [0]])

        results = xor.learn(inputs={"inputs": {xor_in:xor_inputs},
                                    "targets": {xor_out:xor_targets},
                                    "epochs": 10
                                    },
                                    minibatch_size=minibatch_size,
                                    bin_execute=mode)


        assert len(results) == 4 // minibatch_size

    def test_composition_level_stateful_function_resets(self):
        A = pnl.TransferMechanism(
            name='A',
            integrator_mode=True,
            integration_rate=0.5
        )
        B = pnl.TransferMechanism(
            name='B',
            integrator_mode=True,
            integration_rate=0.5
        )
        C = pnl.TransferMechanism(name='C')

        comp = pnl.Composition(
            pathways=[[A, C], [B, C]]
        )

        A.log.set_log_conditions('value')
        B.log.set_log_conditions('value')

        comp.run(
            inputs={A: [1.0],
                    B: [1.0]},
            reset_stateful_functions_when={
                A: pnl.AtTrial(3),
                B: pnl.AtTrial(4)
            },
            reset_stateful_functions_to = {
                A: 0.5,
                B: 0.5
            },
            num_trials = 5
        )
        # Mechanism A - resets to 0.5 at the beginning of Trial 3. Its value at the end of Trial 3 will
        # be exactly one step of integration forward from 0.5.
        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.875, Trial 3: 0.75, Trial 4:  0.875
        assert np.allclose(
            A.log.nparray_dictionary('value')[comp.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.875])],
                [np.array([0.75])],
                [np.array([0.875])]
            ]
        )

        # Mechanism B - resets to 0.5 at the beginning of Trial 4. Its value at the end of Trial 4 will
        # be exactly one step of integration forward from 0.5.
        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.875, Trial 3: 0.9375. Trial 4: 0.75
        assert np.allclose(
            B.log.nparray_dictionary('value')[comp.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.875])],
                [np.array([0.9375])],
                [np.array([0.75])]
            ]
        )

        comp.reset()
        comp.run(
            inputs={A: [1.0],
                    B: [1.0]}
        )

        # Mechanisms A and B should have been reset, so verify that their new values are equal to their values at the
        # end of trial 0

        assert A.parameters.value.get(comp) == B.parameters.value.get(comp) == [[0.5]]
