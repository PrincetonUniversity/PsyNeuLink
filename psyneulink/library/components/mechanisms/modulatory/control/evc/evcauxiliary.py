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
import copy
import warnings

from psyneulink.core.components.functions.function import Function_Base
from psyneulink.core.components.functions.statefulfunctions.statefulfunction import StatefulFunction
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import Buffer
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.defaults import MPI_IMPLEMENTATION, defaultControlAllocation
from psyneulink.core.globals.keywords import COMBINE_OUTCOME_AND_COST_FUNCTION, COST_FUNCTION, EVC_SIMULATION, FUNCTION, FUNCTION_PARAMS, NOISE, PREDICTION_MECHANISM, RATE, \
    PREFERENCE_SET_NAME, PROGRESS_BAR_CHAR
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'AVERAGE_INPUTS', 'CONTROL_SIGNAL_GRID_SEARCH_FUNCTION', 'CONTROLLER', 'ControlSignalGridSearch',
    'EVCAuxiliaryError', 'WINDOW_SIZE',
    'kwEVCAuxFunction', 'kwEVCAuxFunctionType', 'kwValueFunction',
    'INPUT', 'INPUT_SEQUENCE', 'PredictionMechanism', 'PY_MULTIPROCESSING',
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

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                filter_function
                    see `filter_function <PredictionMechanism.filter_function>`

                    :default value: None
                    :type:

                rate
                    see `rate <PredictionMechanism.rate>`

                    :default value: 1.0
                    :type: float

                window_size
                    see `window_size <PredictionMechanism.window_size>`

                    :default value: 1
                    :type: int

        """
        variable = Parameter(None, read_only=True, pnl_internal=True, constructor_argument='default_variable')

    classPreferences = {
        PREFERENCE_SET_NAME: 'ValueFunctionCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
       }

    @tc.typecheck
    def __init__(self,
                 function,
                 variable=None,
                 params=None,
                 owner=None,
                 prefs:is_pref_set=None,
                 context=None):
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
    System.controller` of that System), and returns an `expected value of control (EVC) <EVCControlMechanism_EVC>`
    based on these (along with the outcome and combination of costs used to calculate the EVC).

    ValueFunction is the default for an EVCControlMechanism's `value_function <EVCControlMechanism.value_function>`
    attribute, and it is called by `ControlSignalGridSearch` (the EVCControlMechanism's default `function
    <EVCControlMechanism.function>`).

    The ValueFunction's default `function <ValueFunction.function>` calculates the EVC using the result of the
    `function <ObjectiveMechanism.function` of the EVCControlMechanism's `objective_mechanism
    <EVCControlMechanism.objective_mechanism>`, and two auxiliary functions specified in corresponding attributes of
    the EVCControlMechanism: `cost_function <EVCControlMechanism.cost_function>` and `combine_outcome_and_cost_function
    <EVCControlMechanism.combine_outcome_and_cost_function>`. The calculation of the EVC
    provided by ValueFunction can be modified by customizing or replacing any of these functions, or by replacing the
    ValueFunction's `function <ValueFunction.function>` itself (in the EVCControlMechanism's `value_function
    <EVCControlMechanism.value_function>` attribute). Replacement functions must use the same format (number and type
    of items) for its arguments and return values (see `note <EVCControlMechanism_Calling_and_Assigning_Functions>`).

    """

    componentName = kwValueFunction

    def __init__(self, function=None):
        function = function or self.function
        super().__init__(function=function,
                         )

    def _function(
        self,
        controller=None,
        outcome=None,
        costs=None,
        variable=None,
        context=None,
        params=None,

    ):
        """
        function (controller, outcome, costs)

        Calculate the EVC as follows:

        * call the `cost_function` for the EVCControlMechanism specified in the **controller** argument,
          to combine the list of costs specified in the **costs** argument into a single cost value;

        * call the `combine_outcome_and_cost_function` for the EVCControlMechanism specified in the **controller**
          argument, to combine the value specified in the **outcome** argument with the value returned by the
         `cost_function`;

        * return the results in a three item tuple: (EVC, outcome and cost).


        Arguments
        ---------

        controller : EVCControlMechanism
            the EVCControlMechanism for which the EVC is to be calculated;  this is required so that the controller's
            `cost_function <EVCControlMechanism.cost_function>` and `combine_outcome_and_cost_function
            <EVCControlMechanism.combine_outcome_and_cost_function>` functions can be called.

        outcome : value : default float
            should represent the outcome of performance of the `System` for which an `control_allocation` is being
            evaluated.

        costs : list or array of values : default 1d np.array of floats
            each item should be the `cost <ControlSignal.cost>` of one of the controller's `ControlSignals
            <EVCControlMechanism_ControlSignals>`.


        Returns
        -------

        (EVC, outcome, cost) : Tuple(float, float, float)



        """

        if self.is_initializing:
            return (np.array([0]), np.array([0]), np.array([0]))

        # remove this in favor of attribute or parameter?
        cost_function = controller.cost_function
        combine_function = controller.combine_outcome_and_cost_function

        from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction

        # Aggregate costs
        if isinstance(cost_function, UserDefinedFunction):
            cost = cost_function._execute(controller=controller, costs=costs, context=context)
        else:
            cost = cost_function._execute(variable=costs, context=context)

        # Combine outcome and cost to determine value
        if isinstance(combine_function, UserDefinedFunction):
            value = combine_function._execute(controller=controller, outcome=outcome, cost=cost, context=context)
        else:
            value = combine_function._execute(variable=[outcome, -cost], context=context)

        return (value, outcome, cost)


class ControlSignalGridSearch(EVCAuxiliaryFunction):
    """Conduct an exhaustive search of allocation polices and return the one with the maximum `EVC
    <EVCControlMechanism_EVC>`.

    This is the default `function <EVCControlMechanism.function>` for an EVCControlMechanism. It identifies the
    `control_allocation` with the maximum `EVC <EVCControlMechanism_EVC>` by a conducting a grid search over every
    possible `control_allocation` given the `allocation_samples` specified for each of its ControlSignals (i.e.,
    the `Cartesian product <https://en.wikipedia.org/wiki/Cartesian_product>`_ of the `allocation
    <ControlSignal.allocation>` values specified by the `allocation_samples` attribute of each ControlSignal).  The
    full set of allocation policies is stored in the EVCControlMechanism's `control_signal_search_space` attribute.
    The EVCControlMechanism's `evaluate` method is used to simulate its `system <EVCControlMechanism.system>`
    under each `control_allocation` in `control_signal_search_space`, calculate the EVC for each of those policies,
    and return the policy with the greatest EVC. By default, only the maximum EVC is saved and returned.  However,
    setting the `save_all_values_and_policies` attribute to `True` saves each policy and its EVC for each simulation
    run (in the EVCControlMechanism's `EVC_policies` and `EVC_values` attributes, respectively). The EVC is
    calculated for each policy by iterating over the following steps:

    * Select an control_allocation:

        draw a successive item from `control_signal_search_space` in each iteration, and assign each of its values as
        the `allocation` value for the corresponding ControlSignal for that simulation of the `system
        <EVCControlMechanism.system>`.

    * Simulate performance:

        execute the `system <EVCControlMechanism.system>` under the selected `control_allocation` using the
        EVCControlMechanism's `evaluate <EVCControlMechanism.evaluate>` method, and the `value
        <PredictionMechanism.value>`\\s of its `prediction_mechanisms <EVCControlMechanism.prediction_mechanisms>` as
        the input to the corresponding `ORIGIN` Mechanisms of the `system <EVCControlMechanism.system>` it controls;
        the values of all :ref:`stateful attributes` of 'Components` in the System are :ref:`re-initialized` to the
        same value prior to each simulation, so that the results for each `control_allocation
        <EVCControlMechanism.control_allocation>` are based on the same initial conditions.  Each simulation includes
        execution of the EVCControlMechanism's `objective_mechanism`, which provides the result to the
        EVCControlMechanism.  If `system <EVCControlMechanism.system>`\\.recordSimulationPref is `True`,
        the results of each simulation are appended to the `simulation_results <System.simulation_results>`
        attribute of `system <EVCControlMechanism.system>`.

    * Calculate the EVC:

        call the EVCControlMechanism's `value_function <EVCControlMechanism_Value_Function>` to calculate the EVC for
        the current iteration, using three values (see `EVCControlMechanism_Functions` for additional details):

        - the EVCControlMechanism's `input <EVC_Mechanism_Input>`, which is the result of its `objective_mechanism
          <EVCControlMechanism.objective_mechanism>`'s `function <ObjectiveMechanism.function>`) and provides an
          evaluation of the outcome of processing in the `system <EVCControlMechanism.system>` under the current
          `control_allocation`;
        |
        - the result of the `cost <EVCControlMechanism_Cost_Function>` function, called by the `value_function
          <EVCControlMechanism_Value_Function>`, that returns the cost for the `control_allocation` based on
          the current `cost <ControlSignal.cost>` associated with each of its ControlSignals;
        |
        - the result of the `combine <EVCControlMechanism_Combine_Function>` function, called by the `value_function
          <EVCControlMechanism_Value_Function>`, that returns the EVC by subtracting the cost from the outcome

    * Save the values:

        if the `save_all_values_and_policies` attribute is `True`, save allocation policy in the EVCControlMechanism's
        `EVC_policies` attribute, and its value is saved in the `EVC_values` attribute; otherwise, retain only
        maximum EVC value.

    The ControlSignalGridSearch function returns the `control_allocation` that yielded the maximum EVC.
    Its operation can be modified by customizing or replacing any or all of the functions referred to above
    (also see `EVCControlMechanism_Functions`).

    """

    componentName = CONTROL_SIGNAL_GRID_SEARCH_FUNCTION

    def __init__(self,
                 default_variable=None,
                 params=None,
                 function=None,
                 owner=None):
        function = function or self.function
        super().__init__(function=function,
                         owner=owner,
                         )

    def _function(
        self,
        controller=None,
        variable=None,
        context=None,
        runtime_params=None,
        params=None,
    ):
        """Grid search combinations of control_signals in specified allocation ranges to find one that maximizes EVC

        * Called by ControlSignalGridSearch.
        * Call System.execute for each `control_allocation` in `control_signal_search_space`.
        * Store an array of values for output_ports in `monitored_output_ports`
          (i.e., the input_ports in `input_ports`) for each `control_allocation`.
        * Call `_compute_EVC` for each control_allocation to calculate the EVC, identify the  maximum,
          and assign to `EVC_max`.
        * Set `EVC_max_policy` to the `control_allocation` (outputPort.values) corresponding to EVC_max.
        * Set value for each control_signal (outputPort.value) to the values in `EVC_max_policy`.
        * Return an control_allocation.

        .. note::
            * runtime_params is used for self.__execute (that calculates the EVC for each call to System.execute);
              it is NOT used for System.execute -- that uses the runtime_params provided for the Mechanisms in each
              Process.configuration

        Return (2D np.array): value of outputPort for each monitored port (in self.input_ports) for EVC_max

        """

        if self.is_initializing:
            return [defaultControlAllocation]

        # Get value of, or set default for standard args
        if controller is None:
            raise EVCAuxiliaryError("Call to ControlSignalGridSearch() missing controller argument")

        #region RUN SIMULATION

        EVC_max = None
        EVC_values = []
        EVC_policies = []

        # Get allocation_samples for all ControlSignals
        num_control_signals = len(controller.control_signals)
        control_signal_sample_lists = []
        for control_signal in controller.control_signals:
            control_signal_sample_lists.append(control_signal.allocation_samples.generator)
        control_signal_search_space = np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1,num_control_signals)

        # Print progress bar
        if controller.prefs.reportOutputPref:
            progress_bar_rate_str = ""
            search_space_size = len(control_signal_search_space)
            progress_bar_rate = int(10**(np.log10(search_space_size) - 2))
            if progress_bar_rate > 1:
                progress_bar_rate_str = str(progress_bar_rate) + " "
            print("\n{0} evaluating EVC for {1} (one dot for each {2}of {3} samples): ".
                  format(controller.name, controller.system.name, progress_bar_rate_str, search_space_size))

        # Evaluate all combinations of control_signals (policies)
        sample = 0
        EVC_max_port_values = variable.copy()

        EVC_max_policy = control_signal_search_space[0] * 0.0

        # Parallelize using multiprocessing.Pool
        # NOTE:  currently fails on attempt to pickle lambda functions
        #        preserved here for possible future restoration
        if PY_MULTIPROCESSING:
            EVC_pool = Pool()
            results = EVC_pool.map(compute_EVC, [(controller, arg, runtime_params, context)
                                                 for arg in control_signal_search_space], context=context)

        else:

            # Parallelize using MPI
            if MPI_IMPLEMENTATION:
                Comm = MPI.COMM_WORLD
                rank = Comm.Get_rank()
                size = Comm.Get_size()

                chunk_size = (len(control_signal_search_space) + (size - 1)) // size
                print("Rank: {}\nSize: {}\nChunk size: {}".format(rank, size, chunk_size))
                start = chunk_size * rank
                end = chunk_size * (rank + 1)
                if start > len(control_signal_search_space):
                    start = len(control_signal_search_space)
                if end > len(control_signal_search_space):
                    end = len(control_signal_search_space)
            else:
                start = 0
                end = len(control_signal_search_space)

            if MPI_IMPLEMENTATION:
                print("START: {0}\nEND: {1}".format(start,end))

            #region EVALUATE EVC

            # Compute EVC for each allocation policy in control_signal_search_space
            # Notes on MPI:
            # * breaks up search into chunks of size chunk_size for each process (rank)
            # * each process computes max for its chunk and returns
            # * result for each chunk contains EVC max and associated allocation policy for that chunk

            result = None
            EVC_max = float('-Infinity')
            EVC_max_policy = np.zeros_like(control_signal_search_space[0])
            EVC_max_port_values = np.zeros_like(controller.get_input_values(context))
            max_value_port_policy_tuple = (EVC_max, EVC_max_port_values, EVC_max_policy)
            # FIX:  INITIALIZE TO FULL LENGTH AND ASSIGN DEFAULT VALUES (MORE EFFICIENT):
            EVC_values = np.array([])
            EVC_policies = np.array([[]])

            for allocation_vector in control_signal_search_space[start:end,:]:

            # for iter in range(rank, len(control_signal_search_space), size):
            #     allocation_vector = control_signal_search_space[iter,:]:

                if controller.prefs.reportOutputPref:
                    increment_progress_bar = (progress_bar_rate < 1) or not (sample % progress_bar_rate)
                    if increment_progress_bar:
                        print(PROGRESS_BAR_CHAR, end='', flush=True)
                sample +=1

                # Calculate EVC for specified allocation policy
                result_tuple = compute_EVC(controller, allocation_vector, runtime_params, context)
                EVC, outcome, cost = result_tuple

                EVC_max = max(EVC, EVC_max)
                # max_result([t1, t2], key=lambda x: x1)


                # Add to list of EVC values and allocation policies if save option is set
                if controller.save_all_values_and_policies:
                    # FIX:  ASSIGN BY INDEX (MORE EFFICIENT)
                    EVC_values = np.append(EVC_values, np.atleast_1d(EVC), axis=0)
                    # Save policy associated with EVC for each process, as order of chunks
                    #     might not correspond to order of policies in control_signal_search_space
                    if len(EVC_policies[0])==0:
                        EVC_policies = np.atleast_2d(allocation_vector)
                    else:
                        EVC_policies = np.append(EVC_policies, np.atleast_2d(allocation_vector), axis=0)

                # If EVC is greater than the previous value:
                # - store the current set of monitored port value in EVC_max_port_values
                # - store the current set of control_signals in EVC_max_policy
                # if EVC_max > EVC:
                # FIX: PUT ERROR HERE IF EVC AND/OR EVC_MAX ARE EMPTY (E.G., WHEN EXECUTION_ID IS WRONG)
                if EVC == EVC_max:
                    # Keep track of port values and allocation policy associated with EVC max
                    # EVC_max_port_values = controller.input_value.copy()
                    # EVC_max_policy = allocation_vector.copy()
                    EVC_max_port_values = controller.get_input_values(context)
                    EVC_max_policy = allocation_vector
                    max_value_port_policy_tuple = (EVC_max, EVC_max_port_values, EVC_max_policy)

            #endregion

            # Aggregate, reduce and assign global results

            if MPI_IMPLEMENTATION:
                # combine max result tuples from all processes and distribute to all processes
                max_tuples = Comm.allgather(max_value_port_policy_tuple)
                # get tuple with "EVC max of maxes"
                max_of_max_tuples = max(max_tuples, key=lambda max_tuple: max_tuple[0])
                # get EVC_max, port values and allocation policy associated with "max of maxes"
                EVC_max = max_of_max_tuples[0]
                EVC_max_port_values = max_of_max_tuples[1]
                EVC_max_policy = max_of_max_tuples[2]

                if controller.save_all_values_and_policies:
                    EVC_values = np.concatenate(Comm.allgather(EVC_values), axis=0)
                    EVC_policies = np.concatenate(Comm.allgather(EVC_policies), axis=0)

            controller.parameters.EVC_max._set(EVC_max, context)
            controller.parameters.EVC_max_port_values._set(EVC_max_port_values, context)
            controller.parameters.EVC_max_policy._set(EVC_max_policy, context)
            if controller.save_all_values_and_policies:
                controller.parameters.EVC_values._set(EVC_values, context)
                controller.parameters.EVC_policies._set(EVC_policies, context)
            # # TEST PRINT:
            # import re
            # print("\nFINAL:\n\tmax tuple:\n\t\tEVC_max: {}\n\t\tEVC_max_port_values: {}\n\t\tEVC_max_policy: {}".
            #       format(re.sub('[\[,\],\n]','',str(max_value_port_policy_tuple[0])),
            #              re.sub('[\[,\],\n]','',str(max_value_port_policy_tuple[1])),
            #              re.sub('[\[,\],\n]','',str(max_value_port_policy_tuple[2]))),
            #       flush=True)

            # FROM MIKE ANDERSON (ALTERNTATIVE TO allgather:  REDUCE USING A FUNCTION OVER LOCAL VERSION)
            # a = np.random.random()
            # mymax=Comm.allreduce(a, MPI.MAX)
            # print(mymax)

        if controller.prefs.reportOutputPref:
            print("\nEVC simulation completed")
    #endregion

        # -----------------------------------------------------------------

        #region ASSIGN CONTROL SIGNAL VALUES

        # Assign allocations to control_signals for optimal allocation policy:
        EVC_maxStateValue = iter(EVC_max_port_values)

        # Assign max values for optimal allocation policy to controller.input_ports (for reference only)
        for i in range(len(controller.input_ports)):
            controller.input_ports[controller.input_ports.names[i]].parameters.value._set(np.atleast_1d(next(EVC_maxStateValue)), context)


        # Report EVC max info
        if controller.prefs.reportOutputPref:
            print ("\nMaximum EVC for {0}: {1}".format(controller.system.name, float(EVC_max)))
            print ("ControlProjection allocation(s) for maximum EVC:")
            for i in range(len(controller.control_signals)):
                print("\t{0}: {1}".format(controller.control_signals[i].name,
                                        EVC_max_policy[i]))
            print()

        #endregion

        # # TEST PRINT:
        # print ("\nEND OF TRIAL 1 EVC outputPort: {0}\n".format(controller.outputPort.value))

        #region ASSIGN AND RETURN control_allocation
        # Convert EVC_max_policy into 2d array with one control_signal allocation per item,
        #     assign to controller.control_allocation, and return (where it will be assigned to controller.value).
        #     (note:  the conversion is to be consistent with use of controller.value for assignments to control_signals.value)
        allocation_policy = np.array(EVC_max_policy).reshape(len(EVC_max_policy), -1)
        controller.parameters.value._set(allocation_policy, context)
        return allocation_policy
        #endregion


def compute_EVC(ctlr, allocation_vector, runtime_params, context):
    """Compute EVC for a specified `control_allocation <EVCControlMechanism.control_allocation>`.

    IMPLEMENTATION NOTE:  implemented as a function so it can be used with multiprocessing Pool
    IMPLEMENTATION NOTE:  this could be further parallelized if input is for multiple trials

    Simulates and calculates one trial for each set of inputs in ctrl.predicted_input.
    Returns the average EVC over all trials

    Args:
        ctlr (EVCControlMechanism)
        allocation_vector (1D np.array): allocation policy for which to compute EVC
        runtime_params (dict): runtime params passed to ctlr.update
        context (value): context passed to ctlr.update

    Returns (float, float, float):
        (EVC_current, outcome, combined_costs)

    """
    # # TEST PRINT:
    # print("Allocation vector: {}\nPredicted input: {}".
    #       format(allocation_vector, [mech.outputPort.value for mech in predicted_input]),
    #       flush=True)


    # Run one simulation and get EVC for each trial's worth of inputs in predicted_input
    predicted_input = ctlr.parameters.predicted_input._get(context)

    origin_mechs = list(predicted_input.keys())
    # number of trials' worth of inputs in predicted_input should be the same for all ORIGIN Mechanisms, so use first:
    num_trials = len(predicted_input[origin_mechs[0]])
    EVC_list = []


    # FIX: 6/16/18: ADD PREDICTION MECHANISM HERE IF IT'S FUNCTION IS STATEFUL
    # Get any values that need to be reinitialized for each run
    reinitialization_values = {}
    for mechanism in ctlr.system.stateful_mechanisms + ctlr.prediction_mechanisms.mechanisms:
        # "save" the current state of each stateful mechanism by storing the values of each of its stateful
        # attributes in the reinitialization_values dictionary; this gets passed into run and used to call
        # the reinitialize method on each stateful mechanism.
        reinitialization_value = []

        if isinstance(mechanism.function, StatefulFunction):
            for attr in mechanism.function.stateful_attributes:
                reinitialization_value.append(mechanism.function._get_current_function_param(attr, context))
        elif hasattr(mechanism, "integrator_function"):
            if isinstance(mechanism.integrator_function, IntegratorFunction):
                for attr in mechanism.integrator_function.stateful_attributes:
                    reinitialization_value.append(mechanism.integrator_function._get_current_function_param(attr, context))

        reinitialization_values[mechanism] = reinitialization_value

    # Run simulation trial by trial in order to get EVC for each trial
    # IMPLEMENTATION NOTE:  Consider calling execute rather than run (for efficiency)
    for i in range(num_trials):
        sim_context = copy.copy(context)
        sim_context.execution_id = ctlr.get_next_sim_id(context)
        # sim_context.add_flag(ContextFlags.SIMULATION)
        try:
            ctlr.parameters.simulation_ids._get(context).append(sim_context.execution_id)
        except AttributeError:
            ctlr.parameters.simulation_ids._set([sim_context.execution_id], context)

        ctlr.system._initialize_from_context(sim_context, context)

        inputs = {key:value[i] for key, value in predicted_input.items()}

        outcome = ctlr.evaluate(
            inputs=inputs,
            allocation_vector=allocation_vector,
            context=sim_context,
            runtime_params=runtime_params,
            reinitialize_values=reinitialization_values,

        )
        EVC_list.append(
            ctlr.value_function(
                controller=ctlr,
                outcome=outcome,
                costs=ctlr.parameters.control_signal_costs._get(sim_context),
                context=sim_context,

            )
        )
        # sim_context.remove_flag(ContextFlags.SIMULATION)
        # assert True
        # # TEST PRINT EVC:
        # print ("Trial: {}\tInput: {}\tAllocation: {}\tOutcome: {}\tCost: {}\tEVC: {}".
        #        format(i, list(inputs.values())[0], allocation_vector,
        #               EVC_list[i][1], EVC_list[i][2], EVC_list[i][0]))

    # Re-assign values of reinitialization attributes to their value at entry
    for mechanism in reinitialization_values:
        mechanism.reinitialize(*reinitialization_values[mechanism], context=context)

    EVC_avg = list(map(lambda x: (sum(x)) / num_trials, zip(*EVC_list)))

    # TEST PRINT EVC:
    # print("EVC_avg: {}".format(EVC_avg[0]))

    if PY_MULTIPROCESSING:
        return

    else:
        return (EVC_avg)


INPUT = 'INPUT'
AVERAGE_INPUTS = 'AVERAGE_INPUTS'
INPUT_SEQUENCE = 'INPUT_SEQUENCE'
TIME_AVERAGE_INPUT = 'TIME_AVERAGE_INPUT'
input_types = {INPUT, TIME_AVERAGE_INPUT, AVERAGE_INPUTS, INPUT_SEQUENCE}

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
    of `InputPorts <InputPort>` as its `origin_mechanism <PredictionMechanism.origin_mechanism>`, each of which
    receives a `Projection <Projection>` from the same source as the corresponding InputPort of its `origin_mechanism
    <PredictionMechanism.origin_mechanism>` (see `EVCControlMechanism_Prediction_Mechanisms`); and it has one
    `OutputPort` for each of its InputPorts.

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

    * *INPUT:* uses an `Linear` Function to simply pass the input it receives as its output, possibly modified by the
      values specified in the PredictionMechanism's **rate** and **noise** arguments.  If those are specified, they
      are assigned as the `Linear` function's `slope <Linear.slope>` and `intercept <Linear.intercept>` parameters,
      respectively.

    * *TIME_AVERAGE_INPUT:* uses an `AdaptiveIntegrator` Function to compute an exponentially weighted
      time-average of the input to the PredictionMechanism; the PredictionMechanism's **rate** and **noise**
      arguments can be used to specify the corresponding `rate <AdaptiveIntegrator.rate>` and `noise
      <AdaptiveIntegrator.noise>` parameters of the function.  The function returns the time-averaged input
      as a single item.

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
    executed when that System is simulated (that is, it is run using the EVCControlMechanism's `evaluate
    <EVCControlMechanism.evaluate>` method).  Thus, its inputs are updated only once per *actual* run of the
    System, and each simulated run (i.e., the simulation for each `control_allocation
    <EVCControlMechanism.control_allocation>`; see `EVCControlMechanism Execution <EVCControlMechanism_Execution>`)
    uses the *exact same* set of inputs -- the value of the PredictionMechanisms resulting from the last actual run
    of its `system <PredictionMechanism.system>`) -- so that the results of the simulated run for each
    `control_allocation <EVCControlMechanism.control_allocation>` can properly be compared. If the PredictionMechanisms
    for an EVCControlMechanism generate a list of inputs (e.g., using `INPUT_SEQUENCE
    <PredictionMechanism_Input_Sequence>`), then each simulated run involves several trials, each of which uses one
    item from the list of each PredictionMechanism as the input to its `origin_mechanism
    <PredictionMechanism.origin_mechanism>`;  items in the list are used as inputs in the exact same sequence for the
    simulated run of every `control_allocation <EVCControlMechanism.control_allocation>` so that, once again,
    proper comparisons can be made between policies.

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default None
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method; format must match the
        `variable <Mechanism.variable>` for its `origin_mechanism <PredictionMechanism.origin_mechanism>`.

    size : int, list or np.ndarray of ints
        see `size <Mechanism.size>`.

    function : function, *INPUT*, *TIME_AVERAGE_INPUT*, *AVERAGE_INPUTS*, or *INPUT_SEQUENCE* : default
    *TIME_AVERAGE_INPUT*
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
        a `parameter dictionary <ParameterPort_Specification>` that can be used to specify the parameters for
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
        each trial of a simulation, each of which is a 2d np.array containing the input for each `InputPort` of the
        `Mechanism <Mechanism>`.

    """

    componentType = PREDICTION_MECHANISM

    class Parameters(IntegratorMechanism.Parameters):
        """
            Attributes
            ----------

                filter_function
                    see `filter_function <PredictionMechanism.filter_function>`

                    :default value: None
                    :type:

                input_type
                    see `input_type <PredictionMechanism.input_type>`

                    :default value: None
                    :type:

                rate
                    see `rate <PredictionMechanism.rate>`

                    :default value: 1.0
                    :type: ``float``

                window_size
                    see `window_size <PredictionMechanism.window_size>`

                    :default value: 1
                    :type: ``int``
        """
        window_size = Parameter(1, stateful=False, loggable=False)
        filter_function = Parameter(None, stateful=False, loggable=False)
        input_type = None

        rate = Parameter(1.0, modulable=True)

    @tc.typecheck
    @handle_external_context(source=None)
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:tc.optional(tc.any(list, dict))=None,
                 function:tc.optional(tc.enum(
                         INPUT, TIME_AVERAGE_INPUT, AVERAGE_INPUTS, INPUT_SEQUENCE))=TIME_AVERAGE_INPUT,
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

        if not context.source in {ContextFlags.COMPONENT, ContextFlags.COMPOSITION, ContextFlags.COMMAND_LINE}:
            warnings.warn("PredictionMechanism should not be constructed on its own.  If you insist,"
                          "set context=Context(source=ContextFlags.COMMAND_LINE), but proceed at your peril!")
            return

        if params is None:
            params = {}

        if params and FUNCTION in params:
            function = params[FUNCTION]

        input_type = None
        if function in input_types:
            input_type = function

        if function in input_types:

            if function is INPUT:
                function = Linear(slope=rate, intercept=noise)

            elif function is TIME_AVERAGE_INPUT:
                # Use default for IntegratorMechanism: AdaptiveIntegrator
                function = self.class_defaults.function

            elif function in {AVERAGE_INPUTS, INPUT_SEQUENCE}:

                # Maintain the preceding sequence of inputs (of length window_size), and use those for each simulation
                function = Buffer(default_variable=[[0]],
                                  initializer=initial_value,
                                  rate=rate,
                                  noise=noise,
                                  history=window_size)

        params.update({FUNCTION_PARAMS:{RATE:rate,
                                        NOISE:noise}})

        super().__init__(
                default_variable=default_variable,
                size=size,
                input_ports=input_ports,
                function=function,
                params=params,
                name=name,
                prefs=prefs)

    def _execute(self, variable=None, context=None, runtime_params=None):
        """Update predicted value on "real" but not simulation runs"""

        if ContextFlags.SIMULATION in context.execution_phase:
            # Just return current value for simulation runs
            value = self.parameters.value._get(context)

        else:
            # Update deque with new input for any other type of run
            value = super()._execute(variable=variable, context=context, runtime_params=runtime_params)

            # If inputs are being recorded (#recorded = window_size):
            if len(value) > 1:
                if self.input_type is AVERAGE_INPUTS:
                    # Compute average input over window_size
                    value = np.sum(value) / value.shape[0]

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
            from psyneulink.core.components.system import System
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


