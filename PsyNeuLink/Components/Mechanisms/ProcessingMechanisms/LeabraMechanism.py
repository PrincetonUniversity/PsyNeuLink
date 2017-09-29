# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************  LeabraMechanism  ******************************************************

"""
"""

import warnings
import leabra
import numbers
import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import Component, function_type, method_type, parameter_keywords
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, Function_Base, Linear
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError, Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.States.OutputState import PRIMARY_OUTPUT_STATE, StandardOutputStates, standard_output_states
from PsyNeuLink.Globals.Keywords import FUNCTION, GAIN, INITIALIZER, INITIALIZING, INPUT_STATES, LEABRA_FUNCTION, LEABRA_FUNCTION_TYPE, LEABRA_MECHANISM, MEAN, MEDIAN, NETWORK, NOISE, OUTPUT_STATES, RATE, RESULT, STANDARD_DEVIATION, TRANSFER_FUNCTION_TYPE, TRANSFER_MECHANISM, VARIANCE, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Globals.Utilities import append_type_to_name, iscompatible
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

# Used to name input_states and output_states:
MAIN_INPUT = 'main_input'
LEARNING_TARGET = 'learning_target'
MAIN_OUTPUT = 'main_output'
input_state_names =  [MAIN_INPUT, LEARNING_TARGET]
output_state_name = [MAIN_OUTPUT]

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
        print("about to validate variable, which is ", variable)
        if not isinstance(variable, (list, np.ndarray, numbers.Number)):
            raise LeabraError("Input Error: the input variable ({}) was of type {}, but instead should be a list, "
                              "numpy array, or number.".format(variable, type(variable)))

        input_size = self.network.layers[0].size
        output_size = self.network.layers[-1].size
        if (not hasattr(self, "owner")) or (not hasattr(self.owner, "training_flag")) or self.owner.training_flag is False:
            if len(convert_to_2d_input(variable)[0]) != input_size:
                # convert_to_2d_input(variable[0]) is just in case variable is a 2D array rather than a vector
                raise LeabraError("Input Error: the input was {}, which was of an incompatible length with the "
                                  "input_size, which should be {}.".format(convert_to_2d_input(variable)[0], input_size))
        else:
            if len(convert_to_2d_input(variable)[0]) != input_size or len(convert_to_2d_input(variable)[1]) != output_size:
                raise LeabraError("Input Error: the input variable was {}, which was of an incompatible length with "
                                  "the input_size or output_size, which should be {} and {} respectively.".
                                  format(variable, input_size, output_size))
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
        print('about to execute with variable, which is ', variable)
        if (not hasattr(self, "owner")) or (not hasattr(self.owner, "training_flag")) or self.owner.training_flag is False:
            print('variable about to be tested is: ', variable)
            variable = convert_to_2d_input(variable)[0]  # FIX: buggy, doesn't handle lists well. hacky conversion from 2D arrays into 1D arrays
            return test_network(self.network, input_pattern=variable)  # potentially append an array of zeros to make output format consistent
        else:
            variable = convert_to_2d_input(variable)  # FIX: buggy, doesn't handle lists well
            if len(variable) != 2:
                raise LeabraError("Input Error: the input given ({}) for training was not the right format: the input "
                                  "should be a 2D array containing two vectors, corresponding to the input and the "
                                  "training target.".format(variable))
            if len(variable[0]) != self.network.layers[0].size or len(variable[1]) != self.network.layers[-1].size:
                raise LeabraError("Input Error: the input given ({}) was not the right format: it should be a 2D array "
                                  "containing two vectors, corresponding to the input (which should be length {}) and "
                                  "the training target (which should be length {})".
                                  format(variable, self.network.layers[0], self.network.layers[-1].size))
            return train_network(self.network, input_pattern=variable[0], learning_target=variable[1])

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
    paramClassDefaults.update({FUNCTION: LeabraFunction,
                               INPUT_STATES: input_state_names,
                               OUTPUT_STATES: output_state_name})

    standard_output_states = standard_output_states.copy()

    def __init__(self,
                 input_size=1,
                 output_size=1,
                 hidden_layers=0,
                 hidden_sizes=None,
                 training_flag=False,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING):

        leabra_network = build_network(input_size, output_size, hidden_layers, hidden_sizes)

        function = LeabraFunction(network=leabra_network)

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY_OUTPUT_STATE)

        params = self._assign_args_to_param_dicts(function=function,
                                                  input_size=input_size,
                                                  output_size=output_size,
                                                  hidden_sizes=hidden_sizes,
                                                  training_flag=training_flag,
                                                  params=params)

        super().__init__(size=[input_size, output_size],
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

#assumes that within lists and arrays, all elements are the same type
# also this is written sub-optimally: some cases should be broken off into more if statements for speed
def convert_to_2d_input(array_like):
    if isinstance(array_like, (np.ndarray, list)):
        if isinstance(array_like[0], (np.ndarray, list)):
            if isinstance(array_like[0][0], (np.ndarray, list)):
                print("array_like ({}) is at least 3D, which may cause conversion errors".format(array_like))
            out = []
            for a in array_like:
                out.append(np.array(a))
            return out
        elif isinstance(array_like[0], numbers.Number):
            return [np.array(array_like)]
    elif isinstance(array_like, numbers.Number):
        return [np.array([array_like])]


def build_network(n_input, n_output, n_hidden, hidden_sizes=None):

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
        if hidden_sizes is not None:
            hidden_size = hidden_sizes[i]
        else:
            hidden_size = n_input
        hidden_layer = leabra.Layer(hidden_size, spec=layer_spec, unit_spec=unit_spec, name='hidden_layer_{}'.format(i))
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
    for i in range(3):  # FIX: should this be 4 quarters, not 3 quarters?
        network.quarter()
    acts = network.layers[-1].activities
    network.quarter()
    return acts

def train_network(network, input_pattern, learning_target):
    assert len(network.layers[0].units) == len(input_pattern)

    return np.zeros(len(input_pattern))