# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  EVCAuxiliary ******************************************************

"""
Auxiliary functions for `EVCMechanism`.

"""

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Functions.Function import Function_Base
from PsyNeuLink.Globals.Defaults import MPI_IMPLEMENTATION, defaultControlAllocation
from PsyNeuLink.Globals.Keywords import CLOCK, COMBINE_OUTCOME_AND_COST_FUNCTION, CONTEXT, COST_FUNCTION, EVC_SIMULATION, EXECUTING, FUNCTION_OUTPUT_TYPE_CONVERSION, INITIALIZING, PARAMETER_STATE_PARAMS, PARAMS, SAVE_ALL_VALUES_AND_POLICIES, TIME_SCALE, VALUE_FUNCTION, VARIABLE, kwPreferenceSetName, kwProgressBarChar
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

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
OUTCOME = 'outcome'
COSTS = 'costs'


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

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
                               FUNCTION_OUTPUT_TYPE_CONVERSION: False,
                               PARAMETER_STATE_PARAMS: None})

    # MODIFIED 11/29/16 NEW:
    classPreferences = {
        kwPreferenceSetName: 'ValueFunctionCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }
    # MODIFIED 11/29/16 END

    @tc.typecheck
    def __init__(self,
                 function,
                 variable=None,
                 params=None,
                 owner=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)
        self.aux_function = function

        super().__init__(default_variable=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None


class ValueFunction(EVCAuxiliaryFunction):
    """Calculate the `EVC <EVCMechanism_EVC>` for a given performance outcome and set of costs.

    ValueFunction takes as its arguments an outcome (a value representing the performance of a `System`)
    and list of costs (each reflecting the `cost <ControlSignal.cost>` of a `ControlSignal` of the `controller
    System_Base.controller` of that System), and returns an `expected value of control (EVC) <EVCMechanism_EVC>` based
    on these (along with the outcome and aggregation of costs used to calculate the EVC).

    ValueFunction is the default for an EVCMechanism's `value_function <EVCMechanism.value_function>` attribute, and
    it is called by `ControlSignalGridSearch` (the EVCMechanism's default `function <EVCMechanism.function>`).

    The ValueFunction's default `function <ValueFunction.function>` calculates the EVC using the result of the `function
    <ObjectiveMechanism.function` of the EVCMechanism's `objective_mechanism <EVCMechanism.objective_mechanism>`, and
    two auxiliary functions specified in corresponding attributes of the EVCMechanism: `cost_function
    <EVCMechanism.cost_function>` and `combine_outcome_and_cost_function
    <EVCMechanism.combine_outcome_and_cost_function>`. The calculation of the EVC
    provided by ValueFunction can be modified by customizing or replacing any of these functions, or by replacing the
    ValueFunction's `function <ValueFunction.function>` itself (in the EVCMechanism's `value_function
    <EVCMechanism.value_function>` attribute). Replacement functions must use the same format (number and type of
    items) for its arguments and return values (see `note <EVCMechanism_Calling_and_Assigning_Functions>`).

    """

    componentName = kwValueFunction

    def __init__(self, function=None):
        function = function or self.function
        super().__init__(function=function,
                         context=self.componentName+INITIALIZING)

    def function(self, **kwargs):
        """
        function (controller, outcome, costs)

        Calculate the EVC as follows:

        * call the `cost_function` for the EVCMechanism specified in the **controller** argument,
          to combine the list of costs specified in the **costs** argument into a single cost value;

        * call the `combine_outcome_and_cost_function` for the EVCMechanism specified in the **controller** argument,
          to combine the value specified in the **outcome** argument with the value returned by the `cost_function`;

        * return the results in a three item tuple: (EVC, outcome and cost).


        Arguments
        ---------

        controller : EVCMechanism
            the EVCMechanism for which the EVC is to be calculated;  this is required so that the controller's
            `cost_function <EVCMechanism.cost_function>` and `combine_outcome_and_cost_function
            <EVCMechanism.combine_outcome_and_cost_function>` functions can be called.

        outcome : value : default float
            should represent the outcome of performance of the `System` for which an `allocation_policy` is being
            evaluated.

        costs : list or array of values : default 1d np.array of floats
            each item should be the `cost <ControlSignal.cost>` of one of the controller's `ControlSignals
            <EVCMechanism_ControlSignals>`.


        Returns
        -------

        (EVC, outcome, cost) : Tuple(float, float, float)



        """

        context = kwargs['context']

        if INITIALIZING in context:
            return (np.array([0]), np.array([0]), np.array([0]))

        controller = kwargs[CONTROLLER]
        outcome = kwargs[OUTCOME]
        costs = kwargs[COSTS]

        cost_function = controller.paramsCurrent[COST_FUNCTION]
        combine_function = controller.paramsCurrent[COMBINE_OUTCOME_AND_COST_FUNCTION]

        from PsyNeuLink.Components.Functions.Function import UserDefinedFunction

        # Aggregate costs
        if isinstance(cost_function, UserDefinedFunction):
            cost = cost_function.function(controller=controller, costs=costs)
        else:
            cost = cost_function.function(variable=costs, context=context)

        # Combine outcome and cost to determine value
        if isinstance(combine_function, UserDefinedFunction):
            value = combine_function.function(controller=controller, outcome=outcome, cost=cost)
        else:
            value = combine_function.function(variable=[outcome, -cost])

        return (value, outcome, cost)


class ControlSignalGridSearch(EVCAuxiliaryFunction):
    """Conduct an exhaustive search of allocation polices and return the one with the maximum `EVC <EVCMechanism_EVC>`.

    This is the default `function <EVCMechanism.function>` for an EVCMechanism. It identifies the `allocation_policy`
    with the maximum `EVC <EVCMechanism_EVC>` by a conducting a grid search over every possible `allocation_policy`
    given the `allocation_samples` specified for each of its ControlSignals (i.e., the `Cartesian product
    <https://en.wikipedia.org/wiki/Cartesian_product>`_ of the `allocation <ControlSignal.allocation>` values specified
    by the `allocation_samples` attribute of each ControlSignal).  The full set of allocation policies is stored in the
    EVCMechanism's `control_signal_search_space` attribute.  The EVCMechanism's `run_simulation` method is used to
    simulate its `system <EVCMechanism.system>` under each `allocation_policy` in `control_signal_search_space`,
    calculate the EVC for each of those policies, and return the policy with the greatest EVC. By default, only the
    maximum EVC is saved and returned.  However, setting the `save_all_values_and_policies` attribute to `True` saves
    each policy and its EVC for each simulation run (in the EVCMechanism's `EVC_policies` and `EVC_values` attributes,
    respectively). The EVC is calculated for each policy by iterating over the following steps:

    * Select an allocation_policy:
        draw a successive item from `control_signal_search_space` in each iteration, and assign each of its values as
        the `allocation` value for the corresponding ControlSignal for that simulation of the `system
        <EVCMechanism.system>`.

    * Simulate performance:
        execute the `system <EVCMechanism.system>` under the selected `allocation_policy` using the EVCMechanism's
        `run_simulation` method, and the `value <Mechanism_Base.value>`\\s of its `prediction_mechanisms` as the input
        to the `system <EVCMechanism.system>`;  these use the history of previous trials to generate an average
        expected input for each `ORIGIN` Mechanism of the `system <EVCMechanism.system>`.

    * Calculate the EVC:
        call the EVCMechanism's `value_function <EVCMechanism_Value_Function>` to calculate the EVC for the current
        iteration, using three values (see `EVCMechanism_Functions` for additional details):

        - the EVCMechanism's `input <EVC_Mechanism_Input>`, which is the result of its `objective_mechanism
          <EVCMechanism.objective_mechanism>`'s `function <ObjectiveMechanism.function>`) and provides an evaluation
          of the outcome of processing in the `system <EVCMechanism.system>` under the current `allocation_policy`;
        |
        - the result of the `cost <EVCMechanism_Cost_Function>` function, called by the `value_function
          <EVCMechanism_Value_Function>`, that returns the cost for the `allocation_policy` based on
          the current `cost <ControlSignal.cost>` associated with each of its ControlSignals;
        |
        - the result of the `combine <EVCMechanism_Combine_Function>` function, called by the `value_function
          <EVCMechanism_Value_Function>`, that returns the EVC by subtracting the cost from the outcome

    * Save the values:
        if the `save_all_values_and_policies` attribute is `True`, save allocation policy in the EVCMechanism's
        `EVC_policies` attribute, and its value is saved in the `EVC_values` attribute; otherwise, retain only
        maximum EVC value.

    The ControlSignalGridSearch function returns the `allocation_policy` that yielded the maximum EVC.
    Its operation can be modified by customizing or replacing any or all of the functions referred to above
    (also see `EVCMechanism_Functions`).

    """

    componentName = CONTROL_SIGNAL_GRID_SEARCH_FUNCTION

    def __init__(self,
                 default_variable=None,
                 params=None,
                 function=None,
                 owner=None,
                 context=None):
        function = function or self.function
        super().__init__(function=function,
                         owner=owner,
                         context=self.componentName+INITIALIZING)

    def function(self, **kwargs):
        """Grid search combinations of control_signals in specified allocation ranges to find one that maximizes EVC

        Description
        -----------
            * Called by ControlSignalGridSearch.
            * Call System_Base.execute for each `allocation_policy` in `control_signal_search_space`.
            * Store an array of values for outputStates in `monitored_output_states`
                (i.e., the input_states in `input_states`) for each `allocation_policy`.
            * Call `_compute_EVC` for each allocation_policy to calculate the EVC, identify the  maximum,
                and assign to `EVC_max`.
            * Set `EVC_max_policy` to the `allocation_policy` (outputState.values) corresponding to EVC_max.
            * Set value for each control_signal (outputState.value) to the values in `EVC_max_policy`.
            * Return an allocation_policy.

            Note:
            * runtime_params is used for self.__execute (that calculates the EVC for each call to System_Base.execute);
              it is NOT used for System_Base.execute -- that uses the runtime_params provided for the Mechanisms in each
                Process.configuration

            Return (2D np.array): value of outputState for each monitored state (in self.input_states) for EVC_max

        """

        context = kwargs[CONTEXT]

        if INITIALIZING in context:
            return defaultControlAllocation

        # Get value of, or set default for standard args
        try:
            controller = kwargs[CONTROLLER]
        except KeyError:
            raise EVCAuxiliaryError("Call to ControlSignalGridSearch() missing controller argument")
        try:
            variable = self._update_variable(kwargs[VARIABLE])
        except KeyError:
            variable = self._update_variable(None)
        try:
            runtime_params = kwargs[PARAMS]
        except KeyError:
            runtime_params = None
        try:
            clock = kwargs[CLOCK]
        except KeyError:
            clock = CentralClock
        try:
            time_scale = kwargs[TIME_SCALE]
        except KeyError:
            time_scale = TimeScale.TRIAL
        try:
            context = kwargs[CONTEXT]
        except KeyError:
            context = None

        #region RUN SIMULATION

        controller.EVC_max = None
        controller.EVC_values = []
        controller.EVC_policies = []

        # Reset context so that System knows this is a simulation (to avoid infinitely recursive loop)
        context = context.replace(EXECUTING, '{0} {1}'.format(controller.name, EVC_SIMULATION))

        # Print progress bar
        if controller.prefs.reportOutputPref:
            progress_bar_rate_str = ""
            search_space_size = len(controller.control_signal_search_space)
            progress_bar_rate = int(10 ** (np.log10(search_space_size)-2))
            if progress_bar_rate > 1:
                progress_bar_rate_str = str(progress_bar_rate) + " "
            print("\n{0} evaluating EVC for {1} (one dot for each {2}of {3} samples): ".
                  format(controller.name, controller.system.name, progress_bar_rate_str, search_space_size))

        # Evaluate all combinations of control_signals (policies)
        sample = 0
        controller.EVC_max_state_values = variable.copy()
        controller.EVC_max_policy = controller.control_signal_search_space[0] * 0.0

        # Parallelize using multiprocessing.Pool
        # NOTE:  currently fails on attempt to pickle lambda functions
        #        preserved here for possible future restoration
        if PY_MULTIPROCESSING:
            EVC_pool = Pool()
            results = EVC_pool.map(_compute_EVC, [(controller, arg, runtime_params, time_scale, context)
                                                 for arg in controller.control_signal_search_space])

        else:

            # Parallelize using MPI
            if MPI_IMPLEMENTATION:
                Comm = MPI.COMM_WORLD
                rank = Comm.Get_rank()
                size = Comm.Get_size()

                chunk_size = (len(controller.control_signal_search_space) + (size-1)) // size
                print("Rank: {}\nChunk size: {}".format(rank, chunk_size))
                start = chunk_size * rank
                end = chunk_size * (rank+1)
                if start > len(controller.control_signal_search_space):
                    start = len(controller.control_signal_search_space)
                if end > len(controller.control_signal_search_space):
                    end = len(controller.control_signal_search_space)
            else:
                start = 0
                end = len(controller.control_signal_search_space)

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
            EVC_max_policy = np.empty_like(controller.control_signal_search_space[0])
            EVC_max_state_values = np.empty_like(controller.input_values)
            max_value_state_policy_tuple = (EVC_max, EVC_max_state_values, EVC_max_policy)
            # FIX:  INITIALIZE TO FULL LENGTH AND ASSIGN DEFAULT VALUES (MORE EFFICIENT):
            EVC_values = np.array([])
            EVC_policies = np.array([[]])

            # # TEST PRINT:
            # print("\nEVC SIMULATION\n")

            for allocation_vector in controller.control_signal_search_space[start:end,:]:
            # for iter in range(rank, len(controller.control_signal_search_space), size):
            #     allocation_vector = controller.control_signal_search_space[iter,:]:

                if controller.prefs.reportOutputPref:
                    increment_progress_bar = (progress_bar_rate < 1) or not (sample % progress_bar_rate)
                    if increment_progress_bar:
                        print(kwProgressBarChar, end='', flush=True)
                sample +=1

                # Calculate EVC for specified allocation policy
                result_tuple = _compute_EVC(args=(controller, allocation_vector,
                                                  runtime_params,
                                                  time_scale,
                                                  context))
                EVC, outcome, cost = result_tuple

                EVC_max = max(EVC, EVC_max)
                # max_result([t1, t2], key=lambda x: x1)

                # Add to list of EVC values and allocation policies if save option is set
                if controller.paramsCurrent[SAVE_ALL_VALUES_AND_POLICIES]:
                    # FIX:  ASSIGN BY INDEX (MORE EFFICIENT)
                    EVC_values = np.append(EVC_values, np.atleast_1d(EVC), axis=0)
                    # Save policy associated with EVC for each process, as order of chunks
                    #     might not correspond to order of policies in control_signal_search_space
                    if len(EVC_policies[0])==0:
                        EVC_policies = np.atleast_2d(allocation_vector)
                    else:
                        EVC_policies = np.append(EVC_policies, np.atleast_2d(allocation_vector), axis=0)

                # If EVC is greater than the previous value:
                # - store the current set of monitored state value in EVC_max_state_values
                # - store the current set of control_signals in EVC_max_policy
                # if EVC_max > EVC:
                # FIX: PUT ERROR HERE IF EVC AND/OR EVC_MAX ARE EMPTY (E.G., WHEN EXECUTION_ID IS WRONG)
                if EVC == EVC_max:
                    # Keep track of state values and allocation policy associated with EVC max
                    # EVC_max_state_values = controller.input_value.copy()
                    # EVC_max_policy = allocation_vector.copy()
                    EVC_max_state_values = controller.input_values
                    EVC_max_policy = allocation_vector
                    max_value_state_policy_tuple = (EVC_max, EVC_max_state_values, EVC_max_policy)

            #endregion

            # Aggregate, reduce and assign global results

            if MPI_IMPLEMENTATION:
                # combine max result tuples from all processes and distribute to all processes
                max_tuples = Comm.allgather(max_value_state_policy_tuple)
                # get tuple with "EVC max of maxes"
                max_of_max_tuples = max(max_tuples, key=lambda max_tuple: max_tuple[0])
                # get EVC_max, state values and allocation policy associated with "max of maxes"
                controller.EVC_max = max_of_max_tuples[0]
                controller.EVC_max_state_values = max_of_max_tuples[1]
                controller.EVC_max_policy = max_of_max_tuples[2]

                if controller.paramsCurrent[SAVE_ALL_VALUES_AND_POLICIES]:
                    controller.EVC_values = np.concatenate(Comm.allgather(EVC_values), axis=0)
                    controller.EVC_policies = np.concatenate(Comm.allgather(EVC_policies), axis=0)
            else:
                controller.EVC_max = EVC_max
                controller.EVC_max_state_values = EVC_max_state_values
                controller.EVC_max_policy = EVC_max_policy
                if controller.paramsCurrent[SAVE_ALL_VALUES_AND_POLICIES]:
                    controller.EVC_values = EVC_values
                    controller.EVC_policies = EVC_policies
            # # TEST PRINT:
            # import re
            # print("\nFINAL:\n\tmax tuple:\n\t\tEVC_max: {}\n\t\tEVC_max_state_values: {}\n\t\tEVC_max_policy: {}".
            #       format(re.sub('[\[,\],\n]','',str(max_value_state_policy_tuple[0])),
            #              re.sub('[\[,\],\n]','',str(max_value_state_policy_tuple[1])),
            #              re.sub('[\[,\],\n]','',str(max_value_state_policy_tuple[2]))),
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
        EVC_maxStateValue = iter(controller.EVC_max_state_values)

        # Assign max values for optimal allocation policy to controller.input_states (for reference only)
        for i in range(len(controller.input_states)):
            controller.input_states[controller.input_states.names[i]].value = np.atleast_1d(next(EVC_maxStateValue))


        # Report EVC max info
        if controller.prefs.reportOutputPref:
            print ("\nMaximum EVC for {0}: {1}".format(controller.system.name, float(controller.EVC_max)))
            print ("ControlProjection allocation(s) for maximum EVC:")
            for i in range(len(controller.control_signals)):
                print("\t{0}: {1}".format(controller.control_signals[i].name,
                                        controller.EVC_max_policy[i]))
            print()

        #endregionj

        # # TEST PRINT:
        # print ("\nEND OF TRIAL 1 EVC outputState: {0}\n".format(controller.outputState.value))

        #region ASSIGN AND RETURN allocation_policy
        # Convert EVC_max_policy into 2d array with one control_signal allocation per item,
        #     assign to controller.allocation_policy, and return (where it will be assigned to controller.value).
        #     (note:  the conversion is to be consistent with use of controller.value for assignments to control_signals.value)
        controller.allocation_policy = np.array(controller.EVC_max_policy).reshape(len(controller.EVC_max_policy), -1)
        return controller.allocation_policy
        #endregion


def _compute_EVC(args):
    """Compute EVC for a specified `allocation_policy <EVCMechanism.allocation_policy>`.

    IMPLEMENTATION NOTE:  implemented as a function so it can be used with multiprocessing Pool

    Args:
        ctlr (EVCMechanism)
        allocation_vector (1D np.array): allocation policy for which to compute EVC
        runtime_params (dict): runtime params passed to ctlr.update
        time_scale (TimeScale): time_scale passed to ctlr.update
        context (value): context passed to ctlr.update

    Returns (float, float, float):
        (EVC_current, outcome, aggregated_costs)

    """

    ctlr, allocation_vector, runtime_params, time_scale, context = args

    # # TEST PRINT:
    # print("Allocation vector: {}\nPredicted input: {}".
    #       format(allocation_vector, [mech.outputState.value for mech in ctlr.predicted_input]),
    #       flush=True)

    outcome = ctlr.run_simulation(inputs=ctlr.predicted_input,
                        allocation_vector=allocation_vector,
                        runtime_params=runtime_params,
                        time_scale=time_scale,
                        context=context)

    EVC_current = ctlr.paramsCurrent[VALUE_FUNCTION].function(controller=ctlr,
                                                              # MODIFIED 5/7/17 OLD:
                                                              # outcome=ctlr.input_values,
                                                              # MODIFIED 5/7/17 NEW:
                                                              outcome=outcome,
                                                              # MODIFIED 5/7/17 END
                                                              costs=ctlr.control_signal_costs,
                                                              context=context)


    if PY_MULTIPROCESSING:
        return

    else:
        return (EVC_current)
