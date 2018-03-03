from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.functions.function import Linear
from psyneulink.components.component import ComponentError
import numpy as np
import pytest

class TestParameterStates:
    def test_inspect_function_params_slope_noise(self):
        A = TransferMechanism()
        B = TransferMechanism()
        assert A.function_object.slope == 1.0
        assert B.function_object.slope == 1.0
        assert A.function_object._slope == 1.0
        assert B.function_object._slope == 1.0
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        assert A.noise == 0.0
        assert B.noise == 0.0
        assert A._noise == 0.0
        assert B._noise == 0.0
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        A.function_object.slope = 0.2

        assert A.function_object.slope == 0.2
        assert B.function_object.slope == 1.0
        assert A.function_object._slope == 0.2
        assert B.function_object._slope == 1.0
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        A.noise = 0.5

        assert A.noise == 0.5
        assert B.noise == 0.0
        assert A._noise == 0.5
        assert B._noise == 0.0
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        B.function_object.slope = 0.7

        assert A.function_object.slope == 0.2
        assert B.function_object.slope == 0.7
        assert A.function_object._slope == 0.2
        assert B.function_object._slope == 0.7
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        B.noise = 0.6

        assert A.noise == 0.5
        assert B.noise == 0.6
        assert A._noise == 0.5
        assert B._noise == 0.6
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        A.execute(1.0)
        assert A.mod_slope == [0.2]

        B.execute(1.0)

        assert A.function_object.slope == 0.2
        assert B.function_object.slope == 0.7
        assert A.function_object._slope == 0.2
        assert B.function_object._slope == 0.7
        assert A.mod_slope == [0.2]
        assert B.mod_slope == [0.7]

        assert A.noise == 0.5
        assert B.noise == 0.6
        assert A._noise == 0.5
        assert B._noise == 0.6
        assert A.mod_noise == 0.5
        assert B.mod_noise == 0.6

class TestConfigurableParameters:
    def test_configurable_params(self):
        old_value = 0.2
        new_value = 0.7
        T = TransferMechanism(function=Linear(slope=old_value,
                                              intercept=old_value),
                              noise=old_value,
                              smoothing_factor=old_value)

        # SLOPE - - - - - - - -

        assert np.allclose(T.user_params["function_params"]["slope"], old_value)
        assert np.allclose(T.function_object.slope, old_value)
        assert np.allclose(T.function_object._slope, old_value)
        assert np.allclose(T.mod_slope, old_value)

        T.function_object.slope = new_value

        # KAM changed 3/2/18 --
        # function_params looks at parameter state value, so this will not update until next execution
        assert np.allclose(T.user_params["function_params"]["slope"], old_value)
        assert np.allclose(T.function_object.slope, new_value)
        assert np.allclose(T.function_object._slope, new_value)
        assert np.allclose(T.mod_slope, old_value)

        # INTERCEPT - - - - - - - -

        assert np.allclose(T.user_params["function_params"]["intercept"], old_value)
        assert np.allclose(T.function_object.intercept, old_value)
        assert np.allclose(T.function_object._intercept, old_value)
        assert np.allclose(T.mod_intercept, old_value)

        T.function_object.intercept = new_value

        # KAM changed 3/2/18 --
        # function_params looks at parameter state value, so this will not update until next execution
        assert np.allclose(T.user_params["function_params"]["intercept"], old_value)
        assert np.allclose(T.function_object.intercept, new_value)
        assert np.allclose(T.function_object._intercept, new_value)
        assert np.allclose(T.mod_intercept, old_value)

        # SMOOTHING FACTOR - - - - - - - -

        assert np.allclose(T.user_params["smoothing_factor"], old_value)
        assert np.allclose(T.smoothing_factor, old_value)
        assert np.allclose(T._smoothing_factor, old_value)
        assert np.allclose(T.mod_smoothing_factor, old_value)

        T.smoothing_factor = new_value

        # KAM changed 3/2/18 --
        # function_params looks at parameter state value, so this will not update until next execution
        assert np.allclose(T.user_params["smoothing_factor"], old_value)
        assert np.allclose(T.smoothing_factor, new_value)
        assert np.allclose(T._smoothing_factor, new_value)
        assert np.allclose(T.mod_smoothing_factor, old_value)

        # NOISE - - - - - - - -

        assert np.allclose(T.user_params["noise"], old_value)
        assert np.allclose(T.noise, old_value)
        assert np.allclose(T._noise, old_value)
        assert np.allclose(T.mod_noise, old_value)

        T.noise = new_value

        # KAM changed 3/2/18 --
        # function_params looks at parameter state value, so this will not update until next execution
        assert np.allclose(T.user_params["noise"], old_value)
        assert np.allclose(T.noise, new_value)
        assert np.allclose(T._noise, new_value)
        assert np.allclose(T.mod_noise, old_value)

        T.execute(1.0)

        assert np.allclose(T.user_params["function_params"]["slope"], new_value)
        assert np.allclose(T.function_object.slope, new_value)
        assert np.allclose(T.function_object._slope, new_value)
        assert np.allclose(T.mod_slope, new_value)

        assert np.allclose(T.user_params["function_params"]["intercept"], new_value)
        assert np.allclose(T.function_object.intercept, new_value)
        assert np.allclose(T.function_object._intercept, new_value)
        assert np.allclose(T.mod_intercept, new_value)

        assert np.allclose(T.user_params["smoothing_factor"], new_value)
        assert np.allclose(T.smoothing_factor, new_value)
        assert np.allclose(T._smoothing_factor, new_value)
        assert np.allclose(T.mod_smoothing_factor, new_value)

        assert np.allclose(T.user_params["noise"], new_value)
        assert np.allclose(T.noise, new_value)
        assert np.allclose(T._noise, new_value)
        assert np.allclose(T.mod_noise, new_value)

class TestModParams:
    def test_mod_param_error(self):
        T = TransferMechanism()
        with pytest.raises(ComponentError) as error_text:
            T.mod_slope = 20.0
        assert "directly because it is computed by the ParameterState" in str(error_text.value)
