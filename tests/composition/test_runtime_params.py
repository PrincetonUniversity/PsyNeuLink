import numpy as np

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.scheduling.condition import Any, AtTrial, AfterTrial
from psyneulink.compositions.composition import Composition
from psyneulink.scheduling.scheduler import Scheduler

class TestRuntimeParams:

    def test_composition_run_function_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_mechanism(T)

        assert T.function_object.slope == 1.0
        assert T.parameter_states['slope'].value == 1.0

        # Runtime param used for slope
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0}, runtime_params={T: {"slope": 10.0}})
        assert T.function_object.slope == 1.0
        assert T.parameter_states['slope'].value == 1.0
        assert T.value == 20.0

        # Runtime param NOT used for slope
        C.run(inputs={T: 2.0})
        assert T.function_object.slope == 1.0
        assert T.parameter_states['slope'].value == 1.0
        assert T.value == 2.0

    def test_composition_run_mechanism_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_mechanism(T)

        assert T.noise == 0.0
        assert T.parameter_states['noise'].value == 0.0

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0}, runtime_params={T: {"noise": 10.0}})
        assert T.noise == 0.0
        assert T.parameter_states['noise'].value == 0.0
        assert T.value == 12.0

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0}, )
        assert T.noise == 0.0
        assert T.parameter_states['noise'].value == 0.0
        assert T.value == 2.0

    def test_composition_run_with_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        # S = Scheduler(composition=C)
        C.add_mechanism(T)

        results = []

        def call_after_trial():

            results.append(T.output_state.value)

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AtTrial(1))}},
              # scheduler_processing=S,
              num_trials=4,
              call_after_trial=call_after_trial)

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0},
              call_after_trial=call_after_trial)

        assert np.allclose(results, [np.array([2.]),     # Trial 0 - condition not satisfied yet
                                     np.array([12.]),    # Trial 1 - condition satisfied
                                     np.array([2.]),     # Trial 2 - condition no longer satisfied (not sticky)
                                     np.array([2.]),     # Trial 3 - condition no longer satisfied (not sticky)
                                     np.array([2.])])    # New run (runtime param no longer applies)

    def test_composition_run_with_sticky_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_mechanism(T)

        assert T.noise == 0.0
        assert T.parameter_states['noise'].value == 0.0

        results = []

        def call_after_trial():

            results.append(T.output_state.value)

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AfterTrial(1))}},
              num_trials=4,
              call_after_trial=call_after_trial)

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0},
              call_after_trial=call_after_trial)

        assert np.allclose(results, [np.array([2.]),      # Trial 0 - condition not satisfied yet
                                     np.array([2.]),      # Trial 1 - condition not satisfied yet
                                     np.array([12.]),     # Trial 2 - condition satisfied
                                     np.array([12.]),     # Trial 3 - condition satisfied (sticky)
                                     np.array([2.])])     # New run (runtime param no longer applies)

    def test_composition_run_with_combined_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_mechanism(T)

        results = []

        def call_after_trial():

            results.append(T.output_state.value)

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, Any(AtTrial(1), AfterTrial(2)))}},
              num_trials=5,
              call_after_trial=call_after_trial)

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0},
              call_after_trial=call_after_trial)

        assert np.allclose(results, [np.array([2.]),      # Trial 0 - NOT condition 0, NOT condition 1
                                     np.array([12.]),     # Trial 1 - condition 0, NOT condition 1
                                     np.array([2.]),      # Trial 2 - NOT condition 0, NOT condition 1
                                     np.array([12.]),     # Trial 3 - NOT condition 0, condition 1
                                     np.array([12.]),     # Trial 4 - NOT condition 0, condition 1
                                     np.array([2.])])     # New run (runtime param no longer applies)