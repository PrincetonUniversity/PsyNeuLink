import psyneulink as pnl
import numpy as np

class TestContrastiveHebbian:

    def test_scheduled_contrastive_hebbian(self):
        o = pnl.TransferMechanism()
        m = pnl.ContrastiveHebbianMechanism(
            integrator_mode=True,
            enable_learning=False,
            auto=0,
            hetero=-1,
            size=2,
        )

        s = pnl.sys(m, o)
        ms = pnl.Scheduler(system=s)
        ms.add_condition(o, pnl.WhenFinished(m))
        s.scheduler_processing = ms
        # m.reinitialize_when=pnl.Never()
        print('matrix:\n', m.afferents[1].matrix)
        results = s.run(inputs=[2, 2], num_trials=4)
        print(results)


    def test_using_Hebbian_learning_of_orthognal_inputs_without_integrator_mode(self):
        '''Same as tests/mechanisms/test_recurrent_transfer_mechanism/test_learning_of_orthognal_inputs

        Tests that ContrastiveHebbianMechanism behaves like RecurrentTransferMechanism with Hebbian LearningFunction
        (allowing for epsilon differences due CONVERGENCE CRITERION (executes one extra time for each input).
        '''
        size=4
        R = pnl.ContrastiveHebbianMechanism(
                size=size,
                function=pnl.Linear,
                learning_function=pnl.Hebbian,
                enable_learning=True,
                convergence_criterion=.01,
                # auto=0,
                hetero=np.full((size,size),0.0)
        )
        P=pnl.Process(pathway=[R])
        S=pnl.System(processes=[P])

        inputs_dict = {R:[1,0,1,0]}
        S.run(num_trials=4,
              inputs=inputs_dict)
        assert R.current_execution_time.pass_ == 4
        np.testing.assert_allclose(R.output_states[pnl.ACTIVITY_DIFFERENCE_OUTPUT].value,
                                   [1.20074767, 0.0, 1.20074767, 0.0])
        np.testing.assert_allclose(R.plus_phase_activity, [1.20074767, 0.0, 1.20074767, 0.0])
        np.testing.assert_allclose(R.minus_phase_activity, [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R.output_state.value, [1.20074767, 0.0, 1.20074767, 0.0])
        np.testing.assert_allclose(
            R.recurrent_projection.mod_matrix,
            [
                [0.0,         0.0,         0.2399363,  0.0 ],
                [0.0,         0.0,         0.0,       0.0  ],
                [0.2399363,    0.0,         0.0,       0.0 ],
                [0.0,         0.0,         0.0,       0.0  ]
            ]
        )

        # Reset state so learning of new pattern is "uncontaminated" by activity from previous one
        R.output_state.value = [0,0,0,0]
        inputs_dict = {R:[0,1,0,1]}
        S.run(num_trials=4,
              inputs=inputs_dict)
        np.testing.assert_allclose(
            R.recurrent_projection.mod_matrix,
            [
                [0.0,        0.0,        0.2399363,   0.0      ],
                [0.0,        0.0,        0.0,        0.2399363 ],
                [0.2399363,   0.0,        0.0,        0.0      ],
                [0.0,        0.2399363,   0.0,        0.0      ]
            ]
        )
        np.testing.assert_allclose(R.output_state.value, [0.0, 1.20074767, 0.0, 1.20074767])
        np.testing.assert_allclose(R.output_states[pnl.ACTIVITY_DIFFERENCE_OUTPUT].value,
                                   [ 0.0, 1.20074767, 0.0, 1.20074767])
        np.testing.assert_allclose(R.plus_phase_activity, [0.0, 1.20074767, 0.0, 1.20074767])
        np.testing.assert_allclose(R.minus_phase_activity, [0.0, 0.0, 0.0, 0.0])

    def test_using_Hebbian_learning_of_orthognal_inputs_with_integrator_mode(self):
        '''Same as tests/mechanisms/test_recurrent_transfer_mechanism/test_learning_of_orthognal_inputs

        Tests that ContrastiveHebbianMechanism behaves like RecurrentTransferMechanism with Hebbian LearningFunction
        (allowing for epsilon differences due to INTEGRATION and convergence criterion).
        '''
        size=4
        R = pnl.ContrastiveHebbianMechanism(
                size=size,
                function=pnl.Linear,
                integrator_mode=True,
                learning_function=pnl.Hebbian,
                enable_learning=True,
                integration_rate=0.2,
                convergence_criterion=.01,
                # auto=0,
                hetero=np.full((size,size),0.0)
        )
        P=pnl.Process(pathway=[R])
        S=pnl.System(processes=[P])

        inputs_dict = {R:[1,0,1,0]}
        S.run(num_trials=4,
              inputs=inputs_dict)
        assert R.current_execution_time.pass_ == 18
        np.testing.assert_allclose(R.output_states[pnl.ACTIVITY_DIFFERENCE_OUTPUT].value,
                                   [1.14142296, 0.0, 1.14142296, 0.0])
        np.testing.assert_allclose(R.plus_phase_activity, [1.14142296, 0.0, 1.14142296, 0.0])
        np.testing.assert_allclose(R.minus_phase_activity, [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R.output_state.value, [1.1414229612568625, 0.0, 1.1414229612568625, 0.0])
        np.testing.assert_allclose(
            R.recurrent_projection.mod_matrix,
            [
                [0.0,         0.0,         0.22035998,  0.0        ],
                [0.0,         0.0,         0.0,         0.0        ],
                [0.22035998,  0.0,         0.0,         0.0        ],
                [0.0,         0.0,         0.0,         0.0        ]
            ]
        )
        # Reset state so learning of new pattern is "uncontaminated" by activity from previous one
        R.output_state.value = [0,0,0,0]
        inputs_dict = {R:[0,1,0,1]}
        S.run(num_trials=4,
              inputs=inputs_dict)
        np.testing.assert_allclose(
            R.recurrent_projection.mod_matrix,
            [
                [0.0,        0.0,        0.22035998, 0.0       ],
                [0.0,        0.0,        0.0,        0.22035998],
                [0.22035998, 0.0,        0.0,        0.        ],
                [0.0,        0.22035998, 0.0,        0.        ]
            ]
        )
        np.testing.assert_allclose(R.output_state.value, [0.0, 1.1414229612568625, 0.0, 1.1414229612568625])
        np.testing.assert_allclose(R.output_states[pnl.ACTIVITY_DIFFERENCE_OUTPUT].value,
                                   [ 0.0, 1.14142296, 0.0, 1.14142296])
        np.testing.assert_allclose(R.plus_phase_activity, [0.0, 1.14142296, 0.0, 1.14142296])
        np.testing.assert_allclose(R.minus_phase_activity, [0.0, 0.0, 0.0, 0.0])
