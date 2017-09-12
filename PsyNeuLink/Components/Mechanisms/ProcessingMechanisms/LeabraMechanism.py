# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************  LeabraMechanism  ******************************************************

"""
"""

import leabra
import numbers
import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import Component, function_type, method_type, parameter_keywords
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, Function_Base, Linear
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError, Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.States.OutputState import PRIMARY_OUTPUT_STATE, StandardOutputStates, standard_output_states
from PsyNeuLink.Globals.Keywords import FUNCTION, GAIN, INITIALIZER, INITIALIZING, LEABRA_FUNCTION, LEABRA_FUNCTION_TYPE, LEABRA_MECHANISM, MEAN, MEDIAN, NETWORK, NOISE, RATE, RESULT, STANDARD_DEVIATION, TRANSFER_FUNCTION_TYPE, TRANSFER_MECHANISM, VARIANCE, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Globals.Utilities import append_type_to_name, iscompatible
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

class LeabraError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class LeabraFunction(Function_Base):

    componentType = LEABRA_FUNCTION_TYPE
    componentName = LEABRA_FUNCTION

    multiplicative_param = NotImplemented
    additive_param = NotImplemented  # very hacky

    classPreferences = {
        kwPreferenceSetName: 'LeabraFunctionClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = [0]


    def __init__(self,
                 default_variable=None,
                 network=None,
                 params=None,
                 owner=None,
                 prefs=None,
                 context=componentName + INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(network=network,
                                                  params=params)

        if default_variable is None:
            default_variable = np.zeros(self.network.layers[0].size)  # self.network.layers[0].size is the size of the input layer

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    def _validate_variable(self, variable, context=None):
        if not isinstance(variable, (list, np.ndarray, numbers.Number)):
            raise LeabraError("Input Error: the input variable ({}) was of type {}, but instead should be a list, "
                              "numpy array, or number.".format(variable, type(variable)))

        input_size = self.network.layers[0].size
        if len(variable) != input_size and len(np.atleast_2d(variable[0])) != input_size:
            # the second check np.atleast_2d(variable[0]) is just in case variable was a 2D array rather than a vector
            raise LeabraError("Input Error: the input variable was {}, which was of an incompatible length with the "
                              "input_size, which should be {}.".format(variable, input_size))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        if not isinstance(request_set[NETWORK], leabra.Network):
            raise LeabraError("Error: the network given ({}) was of type {}, but instead must be a leabra Network.".
                              format(request_set[NETWORK], type(request_set[NETWORK])))
        super()._validate_params(request_set, target_set, context)

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        print('variable about to be tested is: ', variable)
        variable = np.atleast_2d(variable)[0]  # SUPER HACKY
        return test_network(self.network, input_pattern=variable)

class LeabraMechanism(ProcessingMechanism_Base):
    """
    """

    componentType = LEABRA_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'TransferCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    # LeabraMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()

    standard_output_states = standard_output_states.copy()

    def __init__(self,
                 # input_states: tc.optional(tc.any(list, dict)) = None    (input states will be two)
                 input_size=1,
                 output_size=1,
                 hidden_layers=0,
                 function=Linear,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING):

        leabra_network = build_network(input_size, output_size, hidden_layers)

        function = LeabraFunction(network=leabra_network)  #, owner=self)

        # size = [input_size, input_size]
        # input_states = ['main_input', 'learning target']

        params = self._assign_args_to_param_dicts(function=function,
                                                  input_size=input_size,
                                                  output_size=output_size,
                                                  params=params)

        super().__init__(size=input_size,
                         input_states=['main_input'],
                         output_states=['main_output'],
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

def build_network(n_input, n_output, n_hidden):

    # specifications
    unit_spec  = leabra.UnitSpec(adapt_on=True, noisy_act=True)
    layer_spec = leabra.LayerSpec(lay_inhib=True)
    conn_spec  = leabra.ConnectionSpec(proj='full', rnd_type='uniform',  rnd_mean=0.75, rnd_var=0.2)

    # input/outputs
    input_layer  = leabra.Layer(n_input, spec=layer_spec, unit_spec=unit_spec, name='input_layer')
    output_layer = leabra.Layer(n_output, spec=layer_spec, unit_spec=unit_spec, name='output_layer')

    # creating the required numbers of hidden layers and connections
    layers = [input_layer]
    connections = []
    for i in range(n_hidden):
        hidden_layer = leabra.Layer(n_input, spec=layer_spec, unit_spec=unit_spec, name='hidden_layer_{}'.format(i))
        hidden_conn  = leabra.Connection(layers[-1],  hidden_layer, spec=conn_spec)
        layers.append(hidden_layer)
        connections.append(hidden_conn)

    last_conn  = leabra.Connection(layers[-1],  output_layer, spec=conn_spec)
    layers.append(output_layer)

    network_spec = leabra.NetworkSpec(quarter_size=50)
    network = leabra.Network(layers=layers, connections=connections)

    return network

def test_network(network, input_pattern):
    assert len(network.layers[0].units) == len(input_pattern)
    network.set_inputs({'input_layer': input_pattern})
    for i in range(3):
        network.quarter()
    acts = network.layers[-1].activities
    network.quarter()
    return acts