import numpy as np

from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.scheduling.condition import AfterTrial, Any, AtTrial

class TestMechanismRuntimeParams:

    def test_mechanism_execute_function_param(self):

        # Construction
        T = TransferMechanism()
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.function.slope == 10.0
        assert T.parameter_ports['slope'].value == 10.0
        assert T.value == 20.0

        # Runtime param NOT used for slope
        T.execute(input=2.0)
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        assert T.value == 2.0

    def test_mechanism_execute_mechanism_param(self):

        # Construction
        T = TransferMechanism()
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # Runtime param used for noise
        T.execute(runtime_params={"noise": 10.0}, input=2.0)
        assert T.noise == 10.0
        assert T.parameter_ports['noise'].value == 10.0
        assert T.value == 12.0

        # Runtime param NOT used for noise
        T.execute(input=2.0)
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0
        assert T.value == 2.0

    def test_runtime_params_reset_isolated(self):

        T = TransferMechanism()

        # Intercept attr updated
        T.function.intercept = 2.0
        assert T.function.intercept == 2.0

        # Runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.function.slope == 10.0
        assert T.parameter_ports['slope'].value == 10.0

        # Intercept attr NOT affected by runtime params
        assert T.function.intercept == 2.0
        assert T.value == 22.0

        # Runtime param NOT used for slope
        T.execute(input=2.0)
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Intercept attr NOT affected by runtime params reset
        assert T.function.intercept == 2.0
        assert T.value == 4.0

    def test_runtime_params_reset_to_most_recent_val(self):
        # NOT instance defaults

        # Construction
        T = TransferMechanism()
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Set slope attribute value directly
        T.function.slope = 2.0
        assert T.function.slope == 2.0

        # Runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.function.slope == 10.0
        assert T.parameter_ports['slope'].value == 10.0
        assert T.value == 20.0

        # Runtime param NOT used for slope - reset to most recent slope value (2.0)
        T.execute(input=2.0)
        assert T.function.slope == 2.0
        assert T.value == 4.0

class TestCompositionRuntimeParams:

    def test_composition_run_function_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Runtime param used for slope
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0}, runtime_params={T: {"slope": 10.0}})
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].parameters.value.get(C) == 1.0
        assert T.parameters.value.get(C.default_execution_id) == 20.0

        # Runtime param NOT used for slope
        C.run(inputs={T: 2.0})
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].parameters.value.get(C) == 1.0
        assert T.parameters.value.get(C.default_execution_id) == 2.0

    def test_composition_run_mechanism_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": 10.0}})
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].parameters.value.get(C) == 0.0
        assert T.parameters.value.get(C.default_execution_id) == 12.0

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0}, )
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].parameters.value.get(C) == 0.0
        assert T.parameters.value.get(C.default_execution_id) == 2.0

    def test_composition_run_with_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        # S = Scheduler(composition=C)
        C.add_node(T)

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AtTrial(1))}},
              # scheduler=S,
              num_trials=4)

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0})

        print(C.results)

        assert np.allclose(C.results, [np.array([[2.]]),     # Trial 0 - condition not satisfied yet
                                       np.array([[12.]]),    # Trial 1 - condition satisfied
                                       np.array([[2.]]),     # Trial 2 - condition no longer satisfied (not sticky)
                                       np.array([[2.]]),     # Trial 3 - condition no longer satisfied (not sticky)
                                       np.array([[2.]])])    # New run (runtime param no longer applies)

    def test_composition_run_with_sticky_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AfterTrial(1))}},
              num_trials=4)

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0})

        assert np.allclose(C.results, [np.array([[2.]]),      # Trial 0 - condition not satisfied yet
                                       np.array([[2.]]),      # Trial 1 - condition not satisfied yet
                                       np.array([[12.]]),     # Trial 2 - condition satisfied
                                       np.array([[12.]]),     # Trial 3 - condition satisfied (sticky)
                                       np.array([[2.]])])     # New run (runtime param no longer applies)

    def test_composition_run_with_combined_condition(self):

        # Construction
        T = TransferMechanism()
        C = Composition()
        C.add_node(T)

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        C.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, Any(AtTrial(1), AfterTrial(2)))}},
              num_trials=5)

        # Runtime param NOT used for noise
        C.run(inputs={T: 2.0})

        assert np.allclose(C.results,[np.array([[2.]]),      # Trial 0 - NOT condition 0, NOT condition 1
                                      np.array([[12.]]),     # Trial 1 - condition 0, NOT condition 1
                                      np.array([[2.]]),      # Trial 2 - NOT condition 0, NOT condition 1
                                      np.array([[12.]]),     # Trial 3 - NOT condition 0, condition 1
                                      np.array([[12.]]),     # Trial 4 - NOT condition 0, condition 1
                                      np.array([[2.]])])     # New run (runtime param no longer applies)
