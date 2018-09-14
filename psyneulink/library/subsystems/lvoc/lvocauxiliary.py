# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  EVCAuxiliary ******************************************************

"""
Auxiliary functions for `EVCControlMechanism`.

"""

import numpy as np
import typecheck as tc
import warnings

from psyneulink.components.functions.function import Buffer, Function_Base, Integrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import OUTCOME
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.defaults import MPI_IMPLEMENTATION, defaultControlAllocation
from psyneulink.globals.keywords import COMBINE_OUTCOME_AND_COST_FUNCTION, COST_FUNCTION, EVC_SIMULATION, FUNCTION, FUNCTION_OUTPUT_TYPE_CONVERSION, FUNCTION_PARAMS, NOISE, PARAMETER_STATE_PARAMS, PREDICTION_MECHANISM, RATE, SAVE_ALL_VALUES_AND_POLICIES, VALUE_FUNCTION, kwPreferenceSetName, kwProgressBarChar
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'AVERAGE_INPUTS', 'CONTROL_SIGNAL_GRID_SEARCH_FUNCTION', 'CONTROLLER',
    'EVCAuxiliaryError', 'EVCAuxiliaryFunction', 'WINDOW_SIZE',
    'kwEVCAuxFunction', 'kwEVCAuxFunctionType', 'kwValueFunction',
    'INPUT_SEQUENCE', 'OUTCOME', 'PredictionMechanism', 'PY_MULTIPROCESSING',
    'TIME_AVERAGE_INPUT', 'ValueFunction', 'FILTER_FUNCTION'
]

PY_MULTIPROCESSING = False

if PY_MULTIPROCESSING:
    from multiprocessing import Pool


if MPI_IMPLEMENTATION:
    from mpi4py import MPI

kwEVCAuxFunction = "EVC AUXILIARY FUNCTION"
kwEVCAuxFunctionType = "EVC AUXILIARY FUNCTION TYPE"
kwValueFunction = "EVC VALUE FUNCTION"
CONTROL_SIGNAL_GRID_SEARCH_FUNCTION = "EVC CONTROL SIGNAL GRID SEARCH FUNCTION"
CONTROLLER = 'controller'


class EVCAuxiliaryError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EVCAuxiliaryFunction(Function_Base):
    """Base class for EVC auxiliary functions
    """
    componentType = kwEVCAuxFunctionType

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = None

    classPreferences = {
        kwPreferenceSetName: 'ValueFunctionCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
       }

    @tc.typecheck
    def __init__(self,
                 function,
                 variable=None,
                 params=None,
                 owner=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)
        self.aux_function = function

        super().__init__(default_variable=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context,
                         function=function,
                         )


class ValueFunction(EVCAuxiliaryFunction):
    """Calculate the `EVC <EVCControlMechanism_EVC>` for a given performance outcome and set of costs.

    ValueFunction takes as its arguments an outcome (a value representing the performance of a `System`)
    and list of costs (each reflecting the `cost <ControlSignal.cost>` of a `ControlSignal` of the `controller
    System.controller` of that System), and returns an `expected value of control (EVC) <EVCControlMechanism_EVC>` based
    on these (along with the outcome and aggregation of costs used to calculate the EVC).

    ValueFunction is the default for an EVCControlMechanism's `value_function <EVCControlMechanism.value_function>` attribute, and
    it is called by `ControlSignalGridSearch` (the EVCControlMechanism's default `function <EVCControlMechanism.function>`).

    The ValueFunction's default `function <ValueFunction.function>` calculates the EVC using the result of the `function
    <ObjectiveMechanism.function` of the EVCControlMechanism's `objective_mechanism <EVCControlMechanism.objective_mechanism>`, and
    two auxiliary functions specified in corresponding attributes of the EVCControlMechanism: `cost_function
    <EVCControlMechanism.cost_function>` and `combine_outcome_and_cost_function
    <EVCControlMechanism.combine_outcome_and_cost_function>`. The calculation of the EVC
    provided by ValueFunction can be modified by customizing or replacing any of these functions, or by replacing the
    ValueFunction's `function <ValueFunction.function>` itself (in the EVCControlMechanism's `value_function
    <EVCControlMechanism.value_function>` attribute). Replacement functions must use the same format (number and type of
    items) for its arguments and return values (see `note <EVCControlMechanism_Calling_and_Assigning_Functions>`).

    """

    componentName = kwValueFunction

    def __init__(self, function=None):
        function = function or self.function
        super().__init__(function=function,
                         context=ContextFlags.CONSTRUCTOR)

    def function(
        self,
        controller=None,
        outcome=None,
        costs=None,
        variable=None,
        params=None,
        context=None
    ):
        """
        function (controller, outcome, costs)

        Calculate the EVC as follows:

        * call the `cost_function` for the EVCControlMechanism specified in the **controller** argument,
          to combine the list of costs specified in the **costs** argument into a single cost value;

        * call the `combine_outcome_and_cost_function` for the EVCControlMechanism specified in the **controller** argument,
          to combine the value specified in the **outcome** argument with the value returned by the `cost_function`;

        * return the results in a three item tuple: (EVC, outcome and cost).


        Arguments
        ---------

        controller : EVCControlMechanism
            the EVCControlMechanism for which the EVC is to be calculated;  this is required so that the controller's
            `cost_function <EVCControlMechanism.cost_function>` and `combine_outcome_and_cost_function
            <EVCControlMechanism.combine_outcome_and_cost_function>` functions can be called.

        outcome : value : default float
            should represent the outcome of performance of the `System` for which an `allocation_policy` is being
            evaluated.

        costs : list or array of values : default 1d np.array of floats
            each item should be the `cost <ControlSignal.cost>` of one of the controller's `ControlSignals
            <EVCControlMechanism_ControlSignals>`.


        Returns
        -------

        (EVC, outcome, cost) : Tuple(float, float, float)



        """

        if self.context.initialization_status == ContextFlags.INITIALIZING:
            return (np.array([0]), np.array([0]), np.array([0]))

        cost_function = controller.paramsCurrent[COST_FUNCTION]
        combine_function = controller.paramsCurrent[COMBINE_OUTCOME_AND_COST_FUNCTION]

        from psyneulink.components.functions.function import UserDefinedFunction

        # Aggregate costs
        if isinstance(cost_function, UserDefinedFunction):
            cost = cost_function._execute(controller=controller, costs=costs)
        else:
            cost = cost_function._execute(variable=costs, context=context)

        # Combine outcome and cost to determine value
        if isinstance(combine_function, UserDefinedFunction):
            value = combine_function._execute(controller=controller, outcome=outcome, cost=cost)
        else:
            value = combine_function._execute(variable=[outcome, -cost])

        return (value, outcome, cost)


AVERAGE_INPUTS = 'AVERAGE_INPUTS'
INPUT_SEQUENCE = 'INPUT_SEQUENCE'
TIME_AVERAGE_INPUT = 'TIME_AVERAGE_INPUT'
input_types = {TIME_AVERAGE_INPUT, AVERAGE_INPUTS, INPUT_SEQUENCE}

WINDOW_SIZE = 'window_size'
FILTER_FUNCTION = 'filter_function'


class PredictionMechanism(IntegratorMechanism):
    """PredictionMechanism(      \
    default_variable=None,       \
    size=None,                   \
    function=TIME_AVERAGE_INPUT, \
    initial_value=None,          \
    window_size=1,               \
    filter_function=None,        \
    params=None,                 \
    name=None,                   \
    prefs=None)

    Tracks the inputs to an `ORIGIN` Mechanism of the `system <EVCControlMechanism.system>` controlled by an
    `EVCControlMechanism`, and used to generate the input for that `ORIGIN` Mechanism in a `simulated run
    <EVCControlMechanism_Execution>` of that System.

    .. _PredictionMechanism_Creation:

    **Creating a PredictionMechanism**

    PredictionMechanisms are created automatically when an `EVCControlMechanism` is created, one for each `ORIGIN`
    Mechanism in the `system <EVCControlMechanism.system>` for which the EVCControlMechanism is a `controller
    <System.controller>`. PredictionMechanisms should not be created on their own, as their execution requires
    tight integration with an EVCControlMechanism and the System to which it belongs, and they will not function
    properly if this is not insured.

    **Structure**

    The `System` to which a PredictionMechanism belongs is referenced in its `system <PredictionMechanism.system>`
    attribute, and the System's `ORIGIN` Mechanism which the PredictionMechanism is associated is referenced in its
    `origin_mechanism <PredictionMechanism.origin_mechanism>` attribute.  A PredictionMechanism has the same number
    of `InputStates <InputState>` as its `origin_mechanism <PredictionMechanism.origin_mechanism>`, each of which
    receives a `Projection <Projection>` from the same source as the corresponding InputState of its `origin_mechanism
    <PredictionMechanism.origin_mechanism>` (see `EVCControlMechanism_Prediction_Mechanisms`); and it has one
    `OutputState` for each of its InputStates.

    .. _PredictionMechanism_Function:

    **Function**

    The `function <PredictionMechanism.function>` of a PredictionMechanism records the input received by its
    `origin_mechanism <PredictionMechanism.origin_mechanism>`, and possibly transforms this in some way.  Any
    function can be assigned, so long as it can take as its input a value with the same format as the `variable
    <Mechanism_Base.variable>` of its `origin_mechanism <PredictionMechanism.origin_mechanism>`, and returns a
    similarly formatted value or a list of them.  The result of a PredictionMechanism's `function
    <PredictionMechanism.function>` is provided as the input to its `origin_mechanism
    <PredictionMechanism.origin_mechanism>` in each trial of a simulated run of its `system
    <PredictionMechanism.system>` (see `EVCControlMechanism Execution <EVCControlMechanism_Execution>`).  If a
    PredictionMechanism's `function <PredictionMechanism.function>` returns more than one item,
    then the EVCControlMechanism runs as many simulations as there are items, using each as the input for one
    simulation, and computes the mean EVC over all of them.  Therefore, **the** `function
    <PredictionMechanism.function>` **of every PredictionMechanism associated with an** `EVCControlMechanism`
    **must return the same number of items.**

    In place of a function, the following keywords can be used to specify one of three standard configurations:

    * *TIME_AVERAGE_INPUT:* uses an `AdaptiveIntegrator` Function to compute an exponentially weighted time-average
      of the input to the PredictionMechanism; the PredictionMechanism's **rate** and **noise** arguments can be used
      to specify the corresponding `rate <AdaptiveIntegrator.rate>` and `noise <AdaptiveIntegrator.noise>` parameters
      of the function.  The function returns the time-averaged input as a single item.

    * *AVERAGE_INPUTS:* uses a `Buffer` Function to compute the average of the number of preceding inputs specified in
      the PredictionMechanism's **window_size** argument.  If the **rate** and/or **noise** arguments are specified,
      they are applied to each item in the list before it is averaged, as follows: :math:`item * rate + noise`.

    .. _PredictionMechanism_Input_Sequence:

    * *INPUT_SEQUENCE:* uses a `Buffer` Function to maintain a running record of preceding inputs, which are returned
      in a list. When the EVCControlMechanism `runs a simulation <EVCControlMechanism_Execution>`, one trial is run
      using each item in the list as the input to the PredictionMechanism's `origin_mechanism
      <PredictionMechanism.origin_mechanism>`. The **window_size** argument can be used to specify how many preceding
      inputs should be maintained in the record; if the number of preceding inputs exceeds the value of
      **window_size**, the oldest is deleted and the most recent one is added to the list.  If **window_size** is not
      specified (`None`), a record of all preceding inputs is maintained and returned. The **filter_function**
      argument can be used to specify a function that filters the list (e.g., modifies and/or deletes items);  the
      modified list is then returned by the PredictionMechanism's `function <PredictionMechanism.function>`.  If the
      **rate** and/or **noise** arguments are specified, they are applied to each item in the list, as follows:
      :math:`item * rate + noise`;  note that since the list is maintained across trials in the actual run of the
      `system <PredictionMechanism.system>`, the effects of these parameters are cumulative over trials.

    **Execution**

    A PredictionMechanism is executed each time its `system <PredictionMechanism>` is run; however, it is **not**
    executed when that System is simulated (that is, it is run using the EVCControlMechanism's `run_simulation
    <EVCControlMechanism.run_simulation>` method).  Thus, its inputs are updated only once per *actual* run of the
    System, and each simulated run (i.e., the simulation for each `allocation_policy
    <EVCControlMechanism.allocation_policy>`; see `EVCControlMechanism Execution <EVCControlMechanism_Execution>`)
    uses the *exact same* set of inputs -- the value of the PredictionMechanisms resulting from the last actual run
    of its `system <PredictionMechanism.system>`) -- so that the results of the simulated run for each
    `allocation_policy <EVCControlMechanism.allocation_policy>` can properly be compared. If the PredictionMechanisms
    for an EVCControlMechanism generate a list of inputs (e.g., using `INPUT_SEQUENCE
    <PredictionMechanism_Input_Sequence>`), then each simulated run involves several trials, each of which uses one
    item from the list of each PredictionMechanism as the input to its `origin_mechanism
    <PredictionMechanism.origin_mechanism>`;  items in the list are used as inputs in the exact same sequence for the
    simulated run of every `allocation_policy <EVCControlMechanism.allocation_policy>` so that, once again,
    proper comparisons can be made between policies.

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default None
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method; format must match the
        `variable <Mechanism.variable>` for its `origin_mechanism <PredictionMechanism.origin_mechanism>`.

    size : int, list or np.ndarray of ints
        see `size <Mechanism.size>`.

    function : function, *TIME_AVERAGE_INPUT*, *AVERAGE_INPUTS*, or *INPUT_SEQUENCE* : default *TIME_AVERAGE_INPUT*
        specifies the function used to generate the input provided to its `origin_mechanism
        <PredictionMechanism.origin_mechanism>`; the function must take as its input a single value with the same
        format as the `variable <Mechanism.variable>` of its `origin_mechanism <PredictionMechanism.origin_mechanism>`,
        and must return a similarly formatted value or a list of them (see `above <PredictionMechanism_Function>`
        for additional details).

    initial_value :  value, list or np.ndarray : default None
        specifies value used to initialize the PredictionMechanism's `value <PredictionMechanism.value>` attribute;
        if `None` is specified, 0 is used if the `value <Function_Base.value>` of the PredictionMechanism's `function
        <PredictionMechanism.function>` is numeric, and an empty list is used if *INPUT_SEQUENCE* is specified.

    window_size : int : default None
        specifies number of input values to maintain when *INPUT_SEQUENCE* option is used for
        `function <PredictionMechanism.function>`

    filter_function: function : default None
        specifies a function that takes a list of values, each of which has the same format as the `variable
        <Mechanism_Base.variable>` of the PredictionMechanism's `origin_mechanism
        <PredictionMechanism.origin_mechanism>`, and returns a list of similarly formatted values, though not
        necessarily of the same length (see `above <PredictionMechanism_Function>` for additional details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its `function <Mechanism_Base.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <TransferMechanism.name>`
        specifies the name of the TransferMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the TransferMechanism; see `prefs <TransferMechanism.prefs>` for details.

    Attributes
    ----------

    system : System
        the `System` to which the PredictionMechanism belongs.

    origin_mechanism : Mechanism
        the `ORIGIN` Mechanism with which the PredictionMechanism is associated (in its `system
        <PredictionMechanism.system>`), and the input of which it tracks.

    variable :  2d np.array
        the input received from the PredictionMechanism's `system <PredictionMechanism.system>`, that is identical
        to the input received by its `origin_mechanism <PredictionMechanism.origin_mechanism>`.

    function : Function
        used to track the inputs to the PredictionMechanism's `origin_mechanism <PredictionMechanism.origin_mechanism>`;
        the default is an `AdaptiveIntegrator` (see `above <PredictionMechanism_Function>` for additional details).

    value : 3d np.array
        result returned by the PredictionMechanism's `function <PredictionMechanism.function>`, and provided as
        input to its `origin_mechanism <PredictionMechanism.origin_mechanism>`.  The format conforms to that of a
        System's `run <System.run>` method: items in the outermost dimension (axis 0) correspond to the inputs for
        each trial of a simulation, each of which is a 2d np.array containing the input for each `InputState` of the
        `Mechanism`.

    """

    componentType = PREDICTION_MECHANISM

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict))=None,
                 function:tc.optional(tc.enum(TIME_AVERAGE_INPUT, AVERAGE_INPUTS, INPUT_SEQUENCE))=TIME_AVERAGE_INPUT,
                 initial_value=None,
                 # rate:tc.optional(tc.any(int, float))=1.0,
                 # noise:tc.optional(tc.any(int, float, callable))=0.0,
                 rate:tc.any(int, float)=1.0,
                 noise:tc.any(int, float, callable)=0.0,
                 window_size=1,
                 filter_function:tc.optional(callable)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        if not context in {ContextFlags.COMPONENT, ContextFlags.COMPOSITION, ContextFlags.COMMAND_LINE}:
            warnings.warn("PredictionMechanism should not be constructed on its own.  If you insist,"
                          "set context=ContextFlags.COMMAND_LINE, but proceed at your peril!")
            return

        if params and FUNCTION in params:
            function = params[FUNCTION]

        input_type = None
        if function in input_types:
            input_type = function

        params = self._assign_args_to_param_dicts(window_size=window_size,
                                                  input_type=input_type,
                                                  filter_function=filter_function,
                                                  params=params)

        if function in input_types:

            if function is TIME_AVERAGE_INPUT:
                # Use default for IntegratorMechanism: AdaptiveIntegrator
                function = self.ClassDefaults.function

            elif function in {AVERAGE_INPUTS, INPUT_SEQUENCE}:

                # Maintain the preceding sequence of inputs (of length window_size), and use those for each simulation
                function = Buffer(default_variable=[[0]],
                                  initializer=initial_value,
                                  rate=rate,
                                  noise=noise,
                                  history=self.window_size)

        params.update({FUNCTION_PARAMS:{RATE:rate,
                                        NOISE:noise}})

        super().__init__(
                default_variable=default_variable,
                size=size,
                input_states=input_states,
                function=function,
                params=params,
                name=name,
                prefs=prefs)

    def _execute(self, variable=None, runtime_params=None, context=None):
        '''Update predicted value on "real" but not simulation runs '''

        if self.context.execution_phase == ContextFlags.SIMULATION:
            # Just return current value for simulation runs
            value = self.value
        else:
            # Update deque with new input for any other type of run
            value = super()._execute(variable, runtime_params=runtime_params, context=context)

            # If inputs are being recorded (#recorded = window_size):
            if len(value) > 1:
                if self.input_type is AVERAGE_INPUTS:
                    # Compute average input over window_size
                    value = np.sum(value)/value.shape[0]

                elif self.input_type is INPUT_SEQUENCE:
                    if self.filter_function:
                        # Use filter_function to return input values
                        value = self.filter_function(value)
                    else:
                        # Return all input values in window_size
                        pass
        return value

    @property
    def system(self):
        try:
            from psyneulink.components.system import System
            return next((p.sender.owner for p in self.afferents if isinstance(p.sender.owner, System)), None)
        except:
            return self._system

    @system.setter
    def system(self, value):
        self._system = value

    @property
    def origin_mechanism(self):
        try:
            return self.system.origin_mechanisms[self.system.controller.prediction_mechanisms.index(self)]
        except:
            return None
