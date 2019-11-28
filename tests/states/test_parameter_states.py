import numpy as np
import pytest

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism

class TestParameterPorts:
    def test_inspect_function_params_slope_noise(self):
        A = TransferMechanism()
        B = TransferMechanism()
        assert A.function.slope == 1.0
        assert B.function.slope == 1.0
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        assert A.noise == 0.0
        assert B.noise == 0.0
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        A.function.slope = 0.2

        assert A.function.slope == 0.2
        assert B.function.slope == 1.0
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        A.noise = 0.5

        assert A.noise == 0.5
        assert B.noise == 0.0
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        B.function.slope = 0.7

        assert A.function.slope == 0.2
        assert B.function.slope == 0.7
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        B.noise = 0.6

        assert A.noise == 0.5
        assert B.noise == 0.6
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        A.execute(1.0)
        assert A.mod_slope == [0.2]

        B.execute(1.0)

        assert A.function.slope == 0.2
        assert B.function.slope == 0.7
        assert A.mod_slope == [0.2]
        assert B.mod_slope == [0.7]

        assert A.noise == 0.5
        assert B.noise == 0.6
        assert A.mod_noise == 0.5
        assert B.mod_noise == 0.6

    def test_direct_call_to_constructor_error(self):
        from psyneulink.core.components.ports.parameterport import ParameterPort, ParameterPortError
        with pytest.raises(ParameterPortError) as error_text:
            ParameterPort(owner='SOMETHING')
        assert "Contructor for ParameterPort cannot be called directly(context: None" in str(error_text.value)

class TestConfigurableParameters:
    def test_configurable_params(self):
        old_value = 0.2
        new_value = 0.7
        T = TransferMechanism(function=Linear(slope=old_value,
                                              intercept=old_value),
                              noise=old_value,
                              integration_rate=old_value)

        # SLOPE - - - - - - - -

        assert np.allclose(T.function.slope, old_value)
        assert np.allclose(T.mod_slope, old_value)

        T.function.slope = new_value

        assert np.allclose(T.function.slope, new_value)
        assert np.allclose(T.mod_slope, old_value)

        # INTERCEPT - - - - - - - -

        assert np.allclose(T.function.intercept, old_value)
        assert np.allclose(T.mod_intercept, old_value)

        T.function.intercept = new_value

        assert np.allclose(T.function.intercept, new_value)
        assert np.allclose(T.mod_intercept, old_value)

        # SMOOTHING FACTOR - - - - - - - -

        assert np.allclose(T.integration_rate, old_value)
        assert np.allclose(T.mod_integration_rate, old_value)

        T.integration_rate = new_value

        # KAM changed 3/2/18 --
        # function_params looks at ParameterPort value, so this will not update until next execution
        assert np.allclose(T.integration_rate, new_value)
        assert np.allclose(T.mod_integration_rate, old_value)

        # NOISE - - - - - - - -

        assert np.allclose(T.noise, old_value)
        assert np.allclose(T.mod_noise, old_value)

        T.noise = new_value

        # KAM changed 3/2/18 --
        # function_params looks at ParameterPort value, so this will not update until next execution
        assert np.allclose(T.noise, new_value)
        assert np.allclose(T.mod_noise, old_value)

        T.execute(1.0)

        assert np.allclose(T.function.slope, new_value)
        assert np.allclose(T.mod_slope, new_value)

        assert np.allclose(T.function.intercept, new_value)
        assert np.allclose(T.mod_intercept, new_value)

        assert np.allclose(T.integration_rate, new_value)
        assert np.allclose(T.mod_integration_rate, new_value)

        assert np.allclose(T.noise, new_value)
        assert np.allclose(T.mod_noise, new_value)

class TestModParams:
    def test_mod_param_error(self):
        T = TransferMechanism()
        with pytest.raises(ComponentError) as error_text:
            T.mod_slope = 20.0
        assert "directly because it is computed by the ParameterPort" in str(error_text.value)
