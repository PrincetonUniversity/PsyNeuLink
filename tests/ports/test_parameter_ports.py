import numpy as np
import psyneulink as pnl
import pytest

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism


class TestParameterPorts:
    def test_inspect_function_params_slope_noise(self):
        A = TransferMechanism()
        B = TransferMechanism()
        assert A.function.slope.base == 1.0
        assert B.function.slope.base == 1.0
        assert A.function.slope.modulated == [1.0]
        assert B.function.slope.modulated == [1.0]

        assert A.noise.base == 0.0
        assert B.noise.base == 0.0
        assert A.noise.modulated == 0.0
        assert B.noise.modulated == 0.0

        A.function.slope.base = 0.2

        assert A.function.slope.base == 0.2
        assert B.function.slope.base == 1.0
        assert A.function.slope.modulated == [1.0]
        assert B.function.slope.modulated == [1.0]

        A.noise.base = 0.5

        assert A.noise.base == 0.5
        assert B.noise.base == 0.0
        assert A.noise.modulated == 0.0
        assert B.noise.modulated == 0.0

        B.function.slope.base = 0.7

        assert A.function.slope.base == 0.2
        assert B.function.slope.base == 0.7
        assert A.function.slope.modulated == [1.0]
        assert B.function.slope.modulated == [1.0]

        B.noise.base = 0.6

        assert A.noise.base == 0.5
        assert B.noise.base == 0.6
        assert A.noise.modulated == 0.0
        assert B.noise.modulated == 0.0

        A.execute(1.0)
        assert A.function.slope.modulated == [0.2]

        B.execute(1.0)

        assert A.function.slope.base == 0.2
        assert B.function.slope.base == 0.7
        assert A.function.slope.modulated == [0.2]
        assert B.function.slope.modulated == [0.7]

        assert A.noise.base == 0.5
        assert B.noise.base == 0.6
        assert A.noise.modulated == 0.5
        assert B.noise.modulated == 0.6

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

        assert np.allclose(T.function.slope.base, old_value)
        assert np.allclose(T.function.slope.modulated, old_value)

        T.function.slope.base = new_value

        assert np.allclose(T.function.slope.base, new_value)
        assert np.allclose(T.function.slope.modulated, old_value)

        # INTERCEPT - - - - - - - -

        assert np.allclose(T.function.intercept.base, old_value)
        assert np.allclose(T.function.intercept.modulated, old_value)

        T.function.intercept.base = new_value

        assert np.allclose(T.function.intercept.base, new_value)
        assert np.allclose(T.function.intercept.modulated, old_value)

        # SMOOTHING FACTOR - - - - - - - -

        assert np.allclose(T.integration_rate.base, old_value)
        assert np.allclose(T.integration_rate.modulated, old_value)

        T.integration_rate.base = new_value

        # KAM changed 3/2/18 --
        # function_params looks at ParameterPort value, so this will not update until next execution
        assert np.allclose(T.integration_rate.base, new_value)
        assert np.allclose(T.integration_rate.modulated, old_value)

        # NOISE - - - - - - - -

        assert np.allclose(T.noise.base, old_value)
        assert np.allclose(T.noise.modulated, old_value)

        T.noise.base = new_value

        # KAM changed 3/2/18 --
        # function_params looks at ParameterPort value, so this will not update until next execution
        assert np.allclose(T.noise.base, new_value)
        assert np.allclose(T.noise.modulated, old_value)

        T.execute(1.0)

        assert np.allclose(T.function.slope.base, new_value)
        assert np.allclose(T.function.slope.modulated, new_value)

        assert np.allclose(T.function.intercept.base, new_value)
        assert np.allclose(T.function.intercept.modulated, new_value)

        assert np.allclose(T.integration_rate.base, new_value)
        assert np.allclose(T.integration_rate.modulated, new_value)

        assert np.allclose(T.noise.base, new_value)
        assert np.allclose(T.noise.modulated, new_value)

class TestModParams:
    def test_mod_param_error(self):
        T = TransferMechanism()
        with pytest.raises(ComponentError) as error_text:
            T.function.slope.modulated = 20.0
        assert "directly because it is computed by the ParameterPort" in str(error_text.value)


class TestParameterPortList:
    @pytest.fixture
    def transfer_mech(self):
        return TransferMechanism(function=pnl.Logistic)

    def test_duplicate(self, transfer_mech):
        assert 'offset-function' in transfer_mech.parameter_ports
        assert 'offset-integrator_function' in transfer_mech.parameter_ports

    def test_duplicate_base_access_fails(self, transfer_mech):
        with pytest.raises(
            pnl.ParameterPortError,
            match='Did you want offset-function or offset-integrator_function'
        ):
            transfer_mech.parameter_ports['offset']

    def test_duplicate_sources(self, transfer_mech):
        assert transfer_mech.parameter_ports['offset-function'].source is transfer_mech.function.parameters.offset
        assert transfer_mech.parameter_ports['offset-integrator_function'].source is transfer_mech.integrator_function.parameters.offset

    def test_sharedparameter_different_name(self, transfer_mech):
        assert transfer_mech.parameter_ports['integration_rate'] is transfer_mech.parameter_ports['rate']
        assert transfer_mech.parameter_ports['integration_rate'].source is transfer_mech.integrator_function.parameters.rate

    def test_alias_unique(self):
        mech = pnl.LCAMechanism()

        assert mech.parameter_ports['leak'] is mech.parameter_ports['integration_rate']

    def test_alias_duplicate(self):
        mech = pnl.LCAMechanism(function=pnl.ReLU)

        assert mech.parameter_ports['leak-function'].source is mech.function.parameters.leak
        assert mech.parameter_ports['leak-integrator_function'] is mech.parameter_ports['integration_rate']
        assert mech.parameter_ports['leak-integrator_function'].source is mech.integrator_function.parameters.rate

    def test_alias_duplicate_base_access_fails(self):
        mech = pnl.LCAMechanism(function=pnl.ReLU)

        with pytest.raises(
            pnl.ParameterPortError,
            match='Did you want leak-function or rate'
        ):
            mech.parameter_ports['leak']
