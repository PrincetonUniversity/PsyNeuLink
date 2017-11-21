# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************  LeabraMechanism  ******************************************************

"""

Overview
--------
A LeabraMechanism is a subclass of `ProcessingMechanism` that wraps a leabra network. Leabra is an artificial neural
network algorithm (`O'Reilly, 1996 <ftp://grey.colorado.edu/pub/oreilly/thesis/oreilly_thesis.all.pdf>`). For more
info about leabra, please see `O'Reilly and Munakata, 2016 <https://grey.colorado.edu/emergent/index.php/Leabra>`_.

.. note::
    The LeabraMechanism uses the leabra Python package, which can be found
    `here <https://github.com/benureau/leabra>` at Github. While the LeabraMechanism should always match the output
    of an equivalent network in the leabra package, the leabra package itself is still in development, so it is not
    guaranteed to be correct yet.

.. _Leabra_Mechanism_Creation:

Creating a LeabraMechanism
--------------------------

A LeabraMechanism can be created in two ways. Users can specify the size of the input layer (**input_size**), size
of the output layer (**output_size**), number of hidden layers (**hidden_layers**), and sizes of the hidden layers
(**hidden_sizes**). In this case, the LeabraMechanism will initialize the connections as uniform random values between
0.55 and 0.95. Alternatively, users can provide a leabra Network object from the leabra package as an argument
(**leabra_net**), in which case the **leabra_net** will be used as the network wrapped by the LeabraMechanism.
This option requires users to be familiar with the leabra package, but allows more flexibility in specifying parameters.
In either method of creating a LeabraMechanism, the **training_flag** argument specifies whether the network should be
learning (updating its weights) or not.

.. _Leabra_Mechanism_Structure:

Structure
---------

The LeabraMechanism has an attribute `training_flag <LeabraMechanism.training_flag>` which can be set to True/False to
determine whether the network is currently learning.

.. note::
    If the training_flag is True, the network will learn using the Leabra learning algorithm. Other algorithms may be
    added later.

The LeabraMechanism has two `InputState`s: the *MAIN_INPUT* InputState and the *LEARNING_TARGET* InputState. The
*MAIN_INPUT* InputState is the input to the leabra network, while the *LEARNING_TARGET* InputState is the learning
target for the LeabraMechanism. The input to the *MAIN_INPUT* InputState should have length equal to
`input_size <LeabraMechanism.input_size>` and the input to the *LEARNING_TARGET* InputState should have length equal to
`output_size <LeabraMechanism.output_size>`.

.. note::
    Currently, there is a bug where LeabraMechanism (and other `Mechanism`s with multiple input states) cannot be
    used as `ORIGIN Mechanisms <System_Mechanisms>` for a `System`. If you desire to use a LeabraMechanism as an ORIGIN
    Mechanism, you can work around this bug by creating two `TransferMechanism`s as ORIGIN Mechanisms instead, and have
    these two TransferMechanisms pass their output to the InputStates of the LeabraMechanism. Here is an example of
    how to do this::
        L = LeabraMechanism(input_size=input_size, output_size=output_size)
        T1 = TransferMechanism(name='T1', size=input_size, function=Linear)
        T2 = TransferMechanism(name='T2', size=output_size, function=Linear)
        p1 = Process(pathway=[T1, L])
        proj = MappingProjection(sender=T2, receiver=L.input_states[1])
        p2 = Process(pathway=[T2, proj, L])
        s = System(processes=[p1, p2])
        s.run(inputs={T1: input_data, T2: training_data})

.. _Leabra_Mechanism_Execution:

Execution
---------

The LeabraMechanism passes input and training data to the leabra Network it wraps, and the LeabraMechanism passes its
leabra Network's output (after one "trial", default 200 cycles in PsyNeuLink) to its primary `OutputState`. For details
on Leabra, please see `O'Reilly and Munakata, 2016 <https://grey.colorado.edu/emergent/index.php/Leabra>` and
the `leabra code on Github <https://github.com/benureau/leabra>`.

.. _Leabra_Mechanism_Reference:

Class Reference
---------------

"""

import numbers
import numpy as np

try:
    import leabra
    leabra_available = True
except ImportError:
    leabra_available = False

from psyneulink.components.functions.function import Function_Base
from psyneulink.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.components.states.outputstate import PRIMARY, StandardOutputStates, standard_output_states
from psyneulink.globals.keywords import FUNCTION, INITIALIZING, INPUT_STATES, LEABRA_FUNCTION, LEABRA_FUNCTION_TYPE, LEABRA_MECHANISM, NETWORK, OUTPUT_STATES, kwPreferenceSetName
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.scheduling.timescale import TimeScale

__all__ = [
    'build_leabra_network', 'convert_to_2d_input', 'input_state_names', 'LeabraError', 'LeabraFunction', 'LeabraMechanism',
    'LEARNING_TARGET', 'MAIN_INPUT', 'MAIN_OUTPUT', 'output_state_name', 'test_leabra_network', 'train_leabra_network',
]

# Used to name input_states and output_states:
MAIN_INPUT = 'main_input'
LEARNING_TARGET = 'learning_target'
MAIN_OUTPUT = 'main_output'
input_state_names = [MAIN_INPUT, LEARNING_TARGET]
output_state_name = [MAIN_OUTPUT]


class LeabraError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LeabraFunction(Function_Base):
    """
    LeabraFunction(             \
        default_variable=None,  \
        network=None,           \
        params=None,            \
        owner=None,             \
        prefs=None)

    .. _LeabraFunction:

    Transform variable by providing it as input to the leabra network inside the LeabraFunction.

    Arguments
    ---------

    default_variable : number or np.array : default np.zeros() (array of zeros)
        specifies a template for the input to the leabra network.

    network : leabra.Network
        specifies the leabra network to be used.

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the LeabraMechanism; see `prefs <LeabraMechanism.prefs>` for details.


    Attributes
    ----------

    variable : number or np.array
        contains value to be transformed.

    network : leabra.Network
        the leabra network that is being used

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the LeabraMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

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

        if not leabra_available:
            raise LeabraError('leabra python module is not installed')

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(network=network,
                                                  params=params)

        if default_variable is None:
            input_size = len(self.network.layers[0].units)
            default_variable = np.zeros(input_size)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    def _validate_variable(self, variable, context=None):
        if not isinstance(variable, (list, np.ndarray, numbers.Number)):
            raise LeabraError("Input Error: the input variable ({}) was of type {}, but instead should be a list, "
                              "numpy array, or number.".format(variable, type(variable)))

        input_size = len(self.network.layers[0].units)
        output_size = len(self.network.layers[-1].units)
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

        # HACK: otherwise the INITIALIZING function executions affect aspects of the leabra learning algorithm
        if INITIALIZING in context:
            output_size = len(self.network.layers[-1].units)
            return np.zeros(output_size)

        if (not hasattr(self, "owner")) or (not hasattr(self.owner, "training_flag")) or self.owner.training_flag is False:
            variable = convert_to_2d_input(variable)[0]  # FIX: buggy, doesn't handle lists well. hacky conversion from 2D arrays into 1D arrays
            return test_leabra_network(self.network, input_pattern=variable)  # potentially append an array of zeros to make output format consistent

        else:
            variable = convert_to_2d_input(variable)  # FIX: buggy, doesn't handle lists well
            if len(variable) != 2:
                raise LeabraError("Input Error: the input given ({}) for training was not the right format: the input "
                                  "should be a 2D array containing two vectors, corresponding to the input and the "
                                  "training target.".format(variable))
            if len(variable[0]) != len(self.network.layers[0].units) or len(variable[1]) != len(self.network.layers[-1].units):
                raise LeabraError("Input Error: the input given ({}) was not the right format: it should be a 2D array "
                                  "containing two vectors, corresponding to the input (which should be length {}) and "
                                  "the training target (which should be length {})".
                                  format(variable, self.network.layers[0], len(self.network.layers[-1].units)))
            return train_leabra_network(self.network, input_pattern=variable[0], output_pattern=variable[1])


class LeabraMechanism(ProcessingMechanism_Base):
    """
    LeabraMechanism(                \
    leabra_net=None,                \
    input_size=1,                   \
    output_size=1,                  \
    hidden_layers=0,                \
    hidden_sizes=None,              \
    training_flag=False,            \
    params=None,                    \
    name=None,                      \
    prefs=None)

    Subclass of `ProcessingMechanism` that is a wrapper for a Leabra network in PsyNeuLink.

    Arguments
    ---------

    leabra_net : Optional[leabra.Network]
        a network object from the leabra package. If specified, the LeabraMechanism's network becomes **leabra_net**,
        and the other arguments that specify the network are ignored (**input_size**, **output_size**,
        **hidden_layers**, **hidden_sizes**).

    input_size : int : default 1
        an integer specifying how many units are in (the size of) the first layer (input) of the leabra network.

    output_size : int : default 1
        an integer specifying how many units are in (the size of) the final layer (output) of the leabra network.

    hidden_layers : int : default 0
        an integer specifying how many hidden layers are in the leabra network.

    hidden_sizes : int or List[int] : default input_size
        if specified, this should be a list of integers, specifying the size of each hidden layer. If **hidden_sizes**
        if a list, the number of integers in **hidden_sizes** should be equal to the number of hidden layers. If not
        specified, hidden layers will default to the same size as the input layer.

    training_flag : boolean : default None
        a boolean specifying whether the leabra network should be learning. If True, the leabra network will adjust
        its weights using the "leabra" algorithm, based on the training pattern (which is read from its second output
        state). The `training_flag` attribute can be changed after initialization, causing the leabra network to
        start/stop learning. If None, `training_flag` will default to False if **leabra_net** argument is not provided.
        If **leabra_net** argument is provided and `training_flag` is None, then the existing learning rules of the
        **leabra_net** will be preserved.

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default KWTA-<index>
        a string used for the name of the mechanism.
        If is not specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to this Mechanism's `function <LeabraMechanism.function>`.

    function : LeabraFunction
        the function that wraps and executes the leabra mechanism

    value : 2d np.array [array(float64)]
        result of executing `function <LeabraMechanism.function>`.

    input_size : int : default 1
        an integer specifying how many units are in (the size of) the first layer (input) of the leabra network.

    output_size : int : default 1
        an integer specifying how many units are in (the size of) the final layer (output) of the leabra network.

    hidden_layers : int : default 0
        an integer specifying how many hidden layers are in the leabra network.

    hidden_sizes : int or List[int] : default input_size
        an integer or list of integers, specifying the size of each hidden layer.

    training_flag : boolean
        a boolean specifying whether the leabra network should be learning. If True, the leabra network will adjust
        its weights using the "leabra" algorithm, based on the training pattern (which is read from its second output
        state). The `training_flag` attribute can be changed after initialization, causing the leabra network to
        start/stop learning.

    network : leabra.Network
        the leabra.Network object which is executed by the LeabraMechanism. For more info about leabra Networks,
        please see the `leabra package <https://github.com/benureau/leabra>` on Github.

    output_states : *ContentAddressableList[OutputState]* : default [`RESULT <TRANSFER_MECHANISM_RESULT>`]
        list of Mechanism's `OutputStates <OutputStates>`.  By default there is a single OutputState,
        `RESULT <TRANSFER_MECHANISM_RESULT>`, that contains the result of a call to the Mechanism's
        `function <LeabraMechanism.function>`;  additional `standard <TransferMechanism_Standard_OutputStates>`
        and/or custom OutputStates may be included, based on the specifications made in the **output_states** argument
        of the Mechanism's constructor.

    output_values : List[array(float64)]
        each item is the `value <OutputState.value>` of the corresponding OutputState in `output_states
        <LeabraMechanism.output_states>`.  The default is a single item containing the result of the
        TransferMechanism's `function <LeabraMechanism.function>`;  additional
        ones may be included, based on the specifications made in the
        **output_states** argument of the Mechanism's constructor (see `TransferMechanism Standard OutputStates
        <TransferMechanism_Standard_OutputStates>`).

    name : str : default LeabraMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).

    Returns
    -------
    instance of LeabraMechanism : LeabraMechanism
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
                 leabra_net=None,
                 input_size=1,
                 output_size=1,
                 hidden_layers=0,
                 hidden_sizes=None,
                 training_flag=False,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING):
        if not leabra_available:
            raise LeabraError('leabra python module is not installed')

        if leabra_net is not None:
            leabra_network = leabra_net
            input_size = len(leabra_network.layers[0].units)
            output_size = len(leabra_network.layers[-1].units)
            hidden_layers = len(leabra_network.layers) - 2
            hidden_sizes = list(map(lambda x: len(x.units), leabra_network.layers))[1:-2]
            training_flag = None
        else:
            if hidden_sizes is None:
                hidden_sizes = input_size
            if training_flag is None:
                training_flag = False
            leabra_network = build_leabra_network(input_size, output_size, hidden_layers, hidden_sizes, training_flag)

        function = LeabraFunction(network=leabra_network)

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        params = self._assign_args_to_param_dicts(function=function,
                                                  input_size=input_size,
                                                  output_size=output_size,
                                                  hidden_layers=hidden_layers,
                                                  hidden_sizes=hidden_sizes,
                                                  training_flag=training_flag,
                                                  params=params)

        super().__init__(size=[input_size, output_size],
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    @property
    def training_flag(self):
        return self._training_flag

    @training_flag.setter
    def training_flag(self, value):
        if self._training_flag is value:
            return
        elif value is True:
            conns = self.function_object.network.connections  # the connections between layers in the Leabra network
            for i in range(len(conns)):
                conns[i].spec.lrule = 'leabra'  # change each connection's learning rule
            self._training_flag = value
        elif value is False:
            conns = self.function_object.network.connections
            for i in range(len(conns)):
                conns[i].spec.lrule = 'None'  # change each connection's learning rule to None
            self._training_flag = value
        elif value is None:
            self._training_flag = value

    @property
    def network(self):
        return self.function_object.network

    @network.setter
    def network(self, value):
        self.function_object.network = value
# assumes that within lists and arrays, all elements are the same type
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


def build_leabra_network(n_input, n_output, n_hidden, hidden_sizes=None, training_flag=None):

    # specifications
    if training_flag is True:
        learning_rule = 'leabra'
    else:
        learning_rule = None
    unit_spec = leabra.UnitSpec(adapt_on=True, noisy_act=True)
    layer_spec = leabra.LayerSpec(lay_inhib=True)
    conn_spec = leabra.ConnectionSpec(proj='full', rnd_type='uniform', rnd_mean=0.75, rnd_var=0.2, lrule=learning_rule)

    # input/outputs
    input_layer = leabra.Layer(n_input, spec=layer_spec, unit_spec=unit_spec, name='input_layer')
    output_layer = leabra.Layer(n_output, spec=layer_spec, unit_spec=unit_spec, name='output_layer')

    # creating the required numbers of hidden layers and connections
    layers = [input_layer]
    connections = []
    if isinstance(hidden_sizes, numbers.Number):
        hidden_sizes = [hidden_sizes] * n_hidden
    for i in range(n_hidden):
        if hidden_sizes is not None:
            hidden_size = hidden_sizes[i]
        else:
            hidden_size = n_input
        hidden_layer = leabra.Layer(hidden_size, spec=layer_spec, unit_spec=unit_spec, name='hidden_layer_{}'.format(i))
        hidden_conn = leabra.Connection(layers[-1],  hidden_layer, spec=conn_spec)
        layers.append(hidden_layer)
        connections.append(hidden_conn)

    last_conn = leabra.Connection(layers[-1],  output_layer, spec=conn_spec)
    connections.append(last_conn)
    layers.append(output_layer)

    network_spec = leabra.NetworkSpec(quarter_size=50)
    network = leabra.Network(layers=layers, connections=connections)

    return network


def test_leabra_network(network, input_pattern):
    assert len(network.layers[0].units) == len(input_pattern)
    network.set_inputs({'input_layer': input_pattern})

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]


def train_leabra_network(network, input_pattern, output_pattern):
    """Run one trial on the network"""
    assert len(network.layers[0].units) == len(input_pattern)

    assert len(network.layers[-1].units) == len(output_pattern)
    network.set_inputs({'input_layer': input_pattern})
    network.set_outputs({'output_layer': output_pattern})

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]

# FIX: add/test compatibility with the np.matrix data type
# def convert_to_2d_input(array_like, num_input_states = None):
#     if isinstance(array_like, numbers.Number) or (isinstance(array_like, np.ndarray) and np.ndim(array_like) == 0):
#         return [np.atleast_1d([array_like])]
#     elif isinstance(array_like, (np.ndarray, list)):
#         if isinstance(array_like[0], (np.ndarray, list)):
#             if isinstance(array_like[0][0], (np.ndarray, list)) and not isinstance(array_like[0][0], np.matrix):
#                 print("WARNING: array_like ({}) is at least 3D, which may cause conversion errors".format(array_like))
#             if num_input_states is None or num_input_states == len(array_like):
#                 out = []
#                 for a in array_like:
#                     out.append(np.atleast_1d(a))
#                 return out
#             elif num_input_states == 1:
#                 return [np.atleast_2d(array_like)]
#             else:
#                 print("WARNING: The number of input states ({}) does not seem compatible with the input ({}).".
#                               format(num_input_states, array_like))
#                 out = []
#                 for a in array_like:
#                     out.append(np.atleast_1d(a))
#                 return out
#         elif isinstance(array_like[0], numbers.Number):
#             if num_input_states is None or num_input_states == 1:
#                 return [np.atleast_1d(array_like)]
#             elif num_input_states == len(array_like):
#                 out = []
#                 for a in array_like:
#                     out.append(np.atleast_1d(a))
#                 return out
#             else:
#                 print("WARNING: The number of input states ({}) does not seem compatible with the input ({}).".
#                               format(num_input_states, array_like))
#                 return [np.atleast_1d(array_like)]
#         else:
#             return np.atleast_2d(array_like)  # this is hacky; mainly for supporting legacy code
#     else:
#         return np.atleast_2d(array_like)  # this is hacky; mainly for supporting legacy code
