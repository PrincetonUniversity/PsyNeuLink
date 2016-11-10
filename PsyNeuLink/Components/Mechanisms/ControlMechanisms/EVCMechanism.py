# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  EVCMechanism ******************************************************

"""
Overview
--------

An EVCMechanism is a :doc:`ControlMechanism` that seeks to optimize a "portfolio" of control signals
so as to maximize the performance of the system to which it belongs.  Each control signal is implemented as a
:doc:`ControlSignal` projection, that regulates the parameter of a mechanism in the system;  collectively,
the control signals govern the system's behavior.  A particular combination of control signal values is called a
*control allocation policy*.  The EVCMechanism evaluates the system's performance under each control allocation policy,
selects the one that generates the best performance, and then assigns the allocation values designated by that policy
to  each of its control signals.  Those are then used when the System is next executed. This implements a form of the
EVC Theory described in Shenhav et al. (2013).

The set of control allocation policies that are tested is contained in the EVCMechanism's ``controlSignalSearchSpace``
attribute (see :ref:`EVCMechanism_ControlSignalSearchSpace`. The performance of the system is tested for each
control allocation policy in the ``controlSignalSearchSpace``, and the outcome of performance is used to calculate the
expected value of control (EVC) is calculated for that policy.  The EVC can be thought of as the cost-benefit ratio
for a given policy, that is calculated as the difference between the outcome of performance and the cost
of the control signals used to generate that outcome. The outcome is assessed by aggregating the ``value`` of each
outputState listed in the``monitoredOutputStates`` attribute.  The cost of the control signals is assessed by
aggregating the cost associated with each ControlSignal projection (based on its ``value`` for the policy,  and c
ontained in its ``cost`` attribute).  The policy that generates the greatest EVC is implemented, and used on the next
round of execution.

.. _EVCMechanism_Creating_An_EVCMechanism:

Creating an EVCMechanism
------------------------

An EVCMechanism can be instantiated using the standard Python method of calling its constructor.  However,  more
commonly, it is generated automatically when a system is created and an EVCMechanism is specified as its ``controller``
(see System :ref:`System_Class_Reference`).  When this occurs, PsyNeuLink configures the EVCMechanism as follows:

  * **Monitored OutputStates** -- these are the outputStates of the system's mechanisms that are monitored by the
    EVCMechanism, and used to determine the outcome of performance under each control allocation policy. An inputState
    is added to the EVCMechanism for each outputState specified in its ``monitored_output_states`` parameter, and a
    :doc:`Mapping` projection is created that projects from that outputState to the EVCMechanism's inputState
    (see _ControlMechanism_Monitored_OutputStates for specifying :keyword:`MONITORED_OUTPUT_STATES`).
  ..
  * **Prediction Mechanisms** -- these are used to generate the input for each simulation of the system run by the
    EVCMechanism (see :ref:`EVCMechanism_Execution`).  A prediction mechanism is created for each :keyword:`ORIGIN`
    (input) mechanism in the system; a Mapping projection is created that projects to it from the corresponding
    :keyword:`ORIGIN` mechanism; and the pair are assigned to their own *prediction process*.  The prediction mechanisms
    and prediction processes for an EVCMechanism are listed in its ``predictionMechanisms`` and ``predictionProcesses``
    attributes, respectively.
  ..
  * **ControlSignal Projections** -- these are used by the EVCMechanism to regulate the parameters of mechanism
    functions that have been specified for control.  A control signal can be assigned to a parameter where its function
    is specified (see :ref:`Mechanism_ParameterStates`). When an EVCMechanism is created for a system system,
    it assumes the control signals for those parameters:  For each, an outputState is added to the EVCMechanism and a
    ControlSignal projection is assigned from that outputState to the parameterState for the parameter to be controlled.

.. _EVCMechanism_Parameters:

EVC Mechanism Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

An EVCMechanism computes the EVC for each control allocation policy using three functions specified in its
``function``, ``outcome_aggregation_function`` and ``cost_aggregation_function`` parameters. These functions,  their
parameters, and the other parameters that govern the operation of the EVCMechanism can be set in the arguments of its
constructor.  However, as noted above, EVCMechanisms are more commonly created automatically as part of a system.
In that case, the EVCMechanism is assigned as the system's ``controller`` attribute, and can be configured by
assigning a params dictionary to the controller's ``params`` attribute using the following keys for its entries
(see :ref:`Mechanism_Specifying_Parameters` for details of parameter specification):

    * :keyword:`MONITORED_OUTPUT_STATES` - the outputStates of the system's mechanisms used  to evaluate the outcome
      of performance for the EVC calculation (see :ref:`ControlMechanism_Monitored_OutputStates` for specifying
      monitored outputStates).  The default is: :keyword:`MonitoredOutputStateOption.PRIMARY_OUTPUT_STATES`,
      which uses the value of the primary outputState of every :keyword:`TERMINAL` mechanism in the system (see
      :ref:`_Mechanism_Role_In_Processes_And_Systems`).  Each outputState in ``monitoredOutputStates`` can be
      assigned an exponent and a weight to parameterize its contribution to the aggregate value (see
      :ref:`EVCMechanism_Parameterizing_EVC_Objective_Function`).
..
    * :keyword:`FUNCTION` - combines the aggregated value of the monitored outputStates with the aggregated cost of
      the control signal values for a given control allocation policy, to determine the **EVC** for that policy.  The
      default is the :class:`LinearCombination` function, which subtracts the aggregated cost from the aggregate value.
    ..
    * :keyword:`OUTCOME_AGGREGATION_FUNCTION` - combines the values of the outputStates in its ``monitoredOutputStates``
      attribute to generate an aggregate **value** of the outcome for a given control allocation policy.  The default
      is the :class:`LinearCombination` function that computes an elementwise (Hadamard) product of the outputState
      values.  The ``weights`` and ``exponents`` arguments of the function can be used, respectively, to scale and/or
      exponentiate the contribution of each outputState's value to the aggregated outcome  The length of thes array
      for these arguments must equal the number of outputStates in``monitoredOutputStates``.  These specifications
      will supercede any made for individual outputStates in a tuple of the ``monitored_output_states`` argument, or
      MONITORED_OUTPUT_STATES entry of a params specification dictionary (see
      :ref:`ControlMechanism_Monitored_OutputStates`).
    ..
    * :keyword:`COST_AGGREGATION_FUNCTION` - combines the costs of the ControlSignals to genreat an aggregate **cost**
      for a given *control allocation policy*.  The default is the :class:`LinearCombination` function that sums the
      costs.  The ``weights`` and ``exponents`` arguments of the function can be used, respectively, to scale and/or
      exponentiate the contribution of each control signal's cost to the value to the aggregated cost; the length of
      the array for those arguments must equal the number of control signals in the ``controlSignals`` attribute.
    ..
    * :keyword:`PREDICTION_MECHANISM_TYPE` - the type of prediction mechanism to use for generating the input
      to the system in each simulation run (see :ref:`EVCMechanism_Execution`).  The default is a
      :class:`AdaptiveIntegratorMechanism`, which exponentially time-averages its inputs.
    ..
    * :keyword:`PREDICTION_MECHANISM_PARAMS` - parameters to use for the prediction mechanism.
    ..
    * :keyword:`SAVE_ALL_VALUES_AND_POLICIES` - specifies whether to save the results of the full EVC evaluation
      for each simulation run (see ``EVCvalues`` and ``EVCpolicies`` attributes in :ref:`EVCMechanism_Class_Reference`).


.. _EVCMechanism_ControlSignalSearchSpace:

Constructing the ControlSignalSearchSpace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``controlSignalSearchSpace`` is constructed from the ``allocationSamples`` attribute of each of the
EVCMechanism's control signals (that is, the ControlSignal projections in each of its outputState
``sendToProjections`` attribute).  The ``allocationSamples`` attribute of a control signal is an array of
values to sample for the parameter controlled by a given control signal.  A control allocation policy is made up of
one value from the ``allocationSamples`` attribute of each of the EVCMechanism's control signals.  When an
EVCMechanism is created, it constructs all possible control allocation policies (i.e., all possible combinations of
values for its control signals), which is placed in its ``controlSignalSearchSpace`` attribute.

COMMENT:
  [TBI:]  The ``controlSignalSearchSpace`` described above is constructed by default.  However, this can be customized
          by assigning either a 2d array or a function that returns a 2d array to the ``controlSignalSearchSpace``
          attribute.  The first dimension (or axis 0) of the 2d array must be an array of control allocation
          policies (of any length), each of which contains a value for each ControlSignal projection in the
          EVCMechanism, assigned in the same order they are listed in its ``controlSignals`` attribute.
COMMENT

.. _EVCMechanism_Parameterizing_EVC_Calculation:

Parameterizing the EVC Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EVC calculation can be parameterized by specifying any of the three functions described above, and/or specifying
how each outputState in its ``monitoredOutputStates`` attribute contributes to the outcome of a control allocation
policy calculated by the :keyword:`OUTCOME_AGGREGATION_FUNCTION`.  The latter can be done by using the tuples format
to specify an outputState in the ``monitored_states_argument`` of an EVCMechanism or system constructor, or the
:keyword:`MONITORED_OUTPUT_STATES` entry of a specification dict assigned to their ``params`` argument (see
:ref:`ControlMechanism_Monitored_OutputStates`). The tuples format can be used to assign an exponent to an outputState
(e.g., to make it a divisor), and/or a weight (i.e., to scale its) for use when it is combined with the others by the
:keyword:`OUTCOME_AGGREGATION_FUNCTION`.  OutputStates not specified in a tuple are assigned an exponent and weight of
1 (see :ref:`EVC_Mechanism_Examples`).

.. _EVCMechanism_Execution:

Execution
---------

When an EVCMechanism is executed, it tests all of the control allocation policies in its ``controlSignalSearchSpace``,
evaluates the EVC for each policy, picks the policy that maximizes the EVC, and implements that policy for the next
execution of the System.  Details of the procedure for each sample are as follows:

* **Select a control allocation policy**: pick the next control allocation policy in the
``controlSignalSearchSpace``, and assign the specified value to each control signal.
..
* **Simulate execution of the system**:  use the prediction process for each :keyword:`ORIGIN` mechanism in the
  system to specify its input (this uses the history of previous trials to generate an average expected input value),
  and execute the System using those inputs, and parameter values (controlled by the EVCMechanism) specified by the
  selected control allcoation policy.
..
* **Calculate the EVC for the control allocation policy**:
    * calculate the **value** of the policy using the EVCMechanism's ``outcome_aggregation_function`` to aggregate
      the value of the outputStates listed in the EVCMechanism's ``monitoredOutputStates`` attribute;
    * calculate the **cost** of the policy using the EVCMechanism's ``cost_aggregation_function`` to aggregate the
      cost associated with each control signal.
    * calculate the **EVC** using the EVCMechanism's ``function`` to subtract the aggregated cost from the
      aggregated value for the policy.
..
* **Save** the values associated with each policy, if the **save_all_values_and_policies** attribute is True.
  Otherwise, only the ``EVCMax`` value is stored.

Once the all control allocation policies in the controlSignalSearchSpace have been evaluated, the policy generating the
maximum EVC is implemented, by assigning the value it specifies for each control signal.  These are then used by the
parameterStates to which they project in the next execution of the system.

.. _EVCMechanism_Examples

Examples
--------

The following example implements a system with an EVCMechanism (and two processes not shown)::

    mySystem = system(processes=[myRewardProcess, myDecisionProcess],
                      controller=EVCMechanism,
                      monitored_output_states=[Reward, DECISION_VARIABLE,(RESPONSE_TIME, -1, 1)],

It uses the system's ``monitored_output_states`` argument to assign three outputStates to be monitored (belonging
to mechanisms not show here).  The first one references a mechanism (belonging to a mechanism not shown;  its
primary outputState will be used by default).  The second and third uses keywords that are the names of
outputStates (in this case, for a :doc:`DDM` ProcessingMechanism).  The last one (RESPONSE_TIME) is assgined an
exponent of -1 and weight of 1. As a result, each calculation of the EVC computation will multiply the value of the
primary outputState of the Reward mechanism by the value of the DECISION_VARIABLE outputState of the DDM mechanism,
and then divide that by the value of the RESPONSE_TIME outputState of the DDM mechanism.

COMMENT:
ADD: This example specifies the EVCMechanism on its own, and then uses it for a system.
COMMENT


.. _EVCMechanism_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlMechanism import *
from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Components.Mechanisms.Mechanism import MonitoredOutputStatesOption
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.AdaptiveIntegrator import AdaptiveIntegratorMechanism
from PsyNeuLink.Components.ShellClasses import *

PY_MULTIPROCESSING = False

if PY_MULTIPROCESSING:
    from multiprocessing import Pool


if MPI_IMPLEMENTATION:
    from mpi4py import MPI


OBJECT = 0
EXPONENT = 1
WEIGHT = 2


class EVCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EVCMechanism(ControlMechanism_Base):
    """EVCMechanism(                                                                    \
    prediction_mechanism_type=AdaptiveIntegratorMechanism,                              \
    prediction_mechanism_params=None,                                                   \
    monitored_output_states=None,                                                       \
    function=LinearCombination(offset=0.0,scale=1,operation=SUM),                       \
    outcome_aggregation_function=LinearCombination(offset=0,scale=1,operation=PRODUCT), \
    cost_aggregation_function=LinearCombination(offset=0.0,scale=1.0,operation=SUM),    \
    save_all_values_and_policies:bool=False,                                            \
    params=None,                                                                        \
    name=None,                                                                          \
    prefs=None)

    Optimizes the ControlSignals for a System.

    COMMENT:
        Class attributes:
            + componentType (str): System Default Mechanism
            + paramClassDefaults (dict):
                + SYSTEM (System)
                + MONITORED_OUTPUT_STATES (list of Mechanisms and/or OutputStates)

        Class methods:
            None
    COMMENT

    Arguments
    ---------

    COMMENT:
       **********************************************************************************************

       PUT SOME OF THIS STUFF IN ATTRIBUTES, BUT USE DEFAULTS HERE

        # - specification of system:  required param: SYSTEM
        # - kwDefaultController:  True =>
        #         takes over all projections from default Controller;
        #         does not take monitored states (those are created de-novo)
        # TBI: - CONTROL_SIGNAL_PROJECTIONS:
        #         list of projections to add (and for which outputStates should be added)

        # - inputStates: one for each performance/environment variable monitiored

        ControlSignal Specification:
        #    - wherever a ControlSignal projection is specified, using kwEVC instead of CONTROL_SIGNAL
        #        this should override the default sender kwSystemDefaultController in ControlSignal._instantiate_sender
        #    ? expclitly, in call to "EVC.monitor(input_state, parameter_state=NotImplemented) method

        # - specification of function: default is default allocation policy (BADGER/GUMBY)
        #     constraint:  if specified, number of items in variable must match number of inputStates in INPUT_STATES
        #                  and names in list in kwMonitor must match those in INPUT_STATES

       **********************************************************************************************

       NOT CURRENTLY IN USE:

        system : System
            system for which the EVCMechanism is the controller;  this is a required parameter.

        default_input_value : Optional[number, list or np.ndarray] : ``defaultControlAllocation`` [LINK]

    COMMENT

    prediction_mechanism_type : CombinationFunction: default AdaptiveIntegratorMechanism
        mechanism class used for prediction mechanism(s).
        Each instance is named using the name of the :keyword:ORIGIN mechanism + PREDICTION_MECHANISM
        and assigned an outputState named based on the same.

    prediction_mechanism_params : Optional[Dict[param keyword, param value]] : default None
        parameter dictionary passed to PredictionMechanismType constructor.
        The same set is passed to all PredictionMechanisms.

    monitored_output_states : List[OutputState or Tuple[OutputState, list or 1d np.array, list or 1d np.array]] : \
    default :keyword:`MonitoredOutputStatesOptions.PRIMARY_OUTPUT_STATES`
        specifies set of outputState values to monitor, and that are passed to outcome_aggregation_function
        (see :ref:`ControlMechanism_Monitored_OutputStates` for specification options, and
        and :ref:`EVCMechanism_Parameterizing_EVC_Calculation`.

    function : CombinationFunction : LinearCombination(offset=0.0,scale=1,operation=SUM)
        specifies the function used to calculate the value discounted by the cost of a control
        allocation policy to determine its EVC.

    outcome_aggregation_function : CombinationFunction : LinearCombination(offset=0,scale=,operation=PRODUCT)
        specifies the function used to aggregate the value of the outputStates in ``monitoredOutputStates``.
        The ``weight`` and/or the ``exponent arguments can be used to parameterize the contribution that each
        outputState makes to the aggregated value;  the length of each argument must equal the number of outputStates
        in ``monitoredOutputStates``.

    cost_aggregation_function : CombinationFunction : LinearCombination(offset=0.0,scale=1.0,operation=SUM)
        specifies the function used to aggregate the cost of the EVCMechanism's control signals.
        The ``weight`` and/or the ``exponent arguments can be used to parameterize the contribution that each
        control signal makes to the aggregated value;  the length of each argument must equal the number of
        control signals in ``controlSignals``.

    save_all_values_and_policies : bool : default False
        when True, saves all ontrol allocation policies in ``controlSignalSearchSpace`` and associated EVC values
        (in addition to the max).

    params : Optional[Dict[param keyword, param value]]
        Dictionary that can be used to specify the parameters for the mechanism, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a parms dict).

    name : str : default Transfer-<index>
        String used for the name of the mechanism.
        If not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : Process.classPreferences]
        Preference set for process.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    Attributes
    ----------

    make_default_controller : bool : default True
        if True, assigns EVCMechanism when instantiated as the DefaultController

    system : System
        system for which EVCMechanism is the ``controller``.

    predictionMechanisms : List[ProcessingMechanism]
        list of predictionMechanisms added to the system, one for each of its :keyword:`ORIGIN` mechanisms

    predictionProcesses : List[Process]
        list of prediction processes added to the system, each comprise of one of its :keyword:`ORIGIN` mechanisms
        and the associated ``predictionMechanism``.

    monitoredOutputStates : List[OutputState]
        each item is an outputState of a mechanism in the system that has been assigned a projection to a corresponding
        inputState of the EVCMechanism.

    monitoredValues : 3D np.nparray
        array of values of the outputStates in ``monitoredOutputStates`` (equivalent to the values of
        self.inputStates).

    function : CombinationFunction : default LinearCombination(offset=0.0,scale=1,operation=SUM)

    outcome_aggregation_function : CombinationFunction : default LinearCombination(offset=0.0,scale=1.0,
    operation=PRODUCT)
        function used to combine the values of the outputStates in ``monitoredOutputStates``.

    cost_aggregation_function : CombinationFunction : default LinearCombination(offset=0.0,scale=1.0,operation=SUM)
        function used to combine the cost of the mechanism's ControlSignal projections.  The :keyword:``weights``
        argument can be used to scale the contribution of the cost of each control signal;  it must be an array of
        scalar values, the length of which is equal to the number of the EVCMechanism's outputStates.

    controlSignalSearchSpace : 2d np.array
        array that contains arrays of control allocation policies.  Each control allocation policy contains one
        value for each of the mechanism's control signals (i.e., ControlSignal projections).  By default,
        it is assigned a set of all possible control allocation policies (using np.meshgrid to construct all
        permutations of control signal values).

    EVCmax : 1d np.array with single value
        the maximum EVC value over all control allocation policies in ``controlSignalSearchSpace``.

    EVCmaxStateValues : 2d np.array
        an array of the values for the outputStates in ``monitoredOutputStates`` using the control allocation policy
        that generated ``EVCmax``.

    EVCmaxPolicy : 1d np.array
        an array of the control signal values (value of ControlSignal projections) for the control allocation policy
        that generated ``EVCmax``.

    prediction_mechanism_type : ProcessingMechanism : default AdaptiveIntegratorMechanism
        processingMechanism class used for prediction mechanism(s).
        Note: each instance will be named based on origin mechanism + kwPredictionMechanism,
              and assigned an outputState named based on the same

    prediction_mechanism_params : Dict[param key, param value] : default ``None``
        parameter dictionary passed to ``prediction_mechanism_type`` on instantiation.
        The same dictionary will be passed to all instances of ``prediction_mechanism_type`` created.

    save_all_values_and_policies : bool : default False
        specifies whether or not to save all ControlAllocationPolicies and associated EVC values (in addition to max).

    """

    componentType = "EVCMechanism"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = [defaultControlAllocation]

    from PsyNeuLink.Components.Functions.Function import LinearCombination
    # from Components.__init__ import DefaultSystem
    paramClassDefaults = ControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({MAKE_DEFAULT_CONTROLLER: True,
                               PARAMETER_STATES: False})

    @tc.typecheck
    def __init__(self,
                 # system:System,
                 # default_input_value=None,
                 prediction_mechanism_type=AdaptiveIntegratorMechanism,
                 prediction_mechanism_params:tc.optional(dict)=None,
                 monitored_output_states:tc.optional(list)=None,
                 function=LinearCombination(offset=0.0,
                                            scale=1,
                                            operation=SUM,
                                            context=componentType+FUNCTION),
                 outcome_aggregation_function=LinearCombination(offset=0,
                                                                scale=1,
                                                                operation=PRODUCT),
                 cost_aggregation_function=LinearCombination(offset=0.0,
                                                             scale=1.0,
                                                             operation=SUM,
                                                             context=componentType+COST_AGGREGATION_FUNCTION),
                 save_all_values_and_policies:bool=False,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+kwInit):

        monitored_output_states = monitored_output_states or [MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]
        prediction_mechanism_params = prediction_mechanism_params or {MONITORED_OUTPUT_STATES:None}

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(# system=system,
                                                  prediction_mechanism_type=prediction_mechanism_type,
                                                  prediction_mechanism_params=prediction_mechanism_params,
                                                  monitored_output_states=monitored_output_states,
                                                  function=function,
                                                  outcome_aggregation_function=outcome_aggregation_function,
                                                  cost_aggregation_function=cost_aggregation_function,
                                                  save_all_values_and_policies=save_all_values_and_policies,
                                                  params=params)

        self.controlSignalChannels = OrderedDict()

        super(EVCMechanism, self).__init__(# default_input_value=default_input_value,
                                           monitored_output_states=monitored_output_states,
                                           function=function,
                                           params=params,
                                           name=name,
                                           prefs=prefs,
                                           context=self)

    def _instantiate_input_states(self, context=None):
        """Instantiate inputState and Mapping Projections for list of Mechanisms and/or States to be monitored

        Instantiate PredictionMechanisms for origin mechanisms in System
        - these will now be terminal mechanisms, and their associated input mechanisms will no longer be
        - if an associated input mechanism needs to be monitored by the EVCMechanism, it must be specified explicilty
            in an outputState, mechanism, controller or systsem MONITORED_OUTPUT_STATES param (see below)

        Parse paramsCurent[MONITORED_OUTPUT_STATES] for system, controller, mechanisms and/or their outputStates:
        - if specification in outputState is None:
             do NOT monitor this state (this overrides any other specifications)
        - if an outputState is specified in *any* MONITORED_OUTPUT_STATES, monitor it (this overrides any other specs)
        - if a mechanism is terminal and/or specified in the system or controller:
            if MonitoredOutputStatesOptions is PRIMARY_OUTPUT_STATES:  monitor only its primary (first) outputState
            if MonitoredOutputStatesOptions is ALL_OUTPUT_STATES:  monitor all of its outputStates
        Note: precedence is given to MonitoredOutputStatesOptions specification in mechanism > controller > system

        Assign inputState to controller for each state to be monitored;  for each item in self.monitoredOutputStates:
        - if it is a OutputState, call _instantiate_monitoring_input_state()
        - if it is a Mechanism, call _instantiate_monitoring_input_state for relevant Mechanism.outputStates
            (determined by whether it is a terminal mechanism and/or MonitoredOutputStatesOption specification)

        Notes:
        * MonitoredOutputStatesOption is an AutoNumbered Enum declared in ControlMechanism
            - it specifies options for assigning outputStates of terminal Mechanisms in the System
                to self.monitoredOutputStates;  the options are:
                + PRIMARY_OUTPUT_STATES: assign only the primary outputState for each terminal Mechanism
                + ALL_OUTPUT_STATES: assign all of the outputStates of each terminal Mechanism
            - precedence is given to MonitoredOutputStatesOptions specification in mechanism > controller > system
        * self.monitoredOutputStates is a list, each item of which is a Mechanism.outputState from which a projection
            will be instantiated to a corresponding inputState of the ControlMechanism
        * self.inputStates is the usual ordered dict of states,
            each of which receives a projection from a corresponding item in self.monitoredOutputStates

        """

        self._instantiate_prediction_mechanisms(context=context)

        from PsyNeuLink.Components.Mechanisms.Mechanism import MonitoredOutputStatesOption
        from PsyNeuLink.Components.States.OutputState import OutputState

        # Clear self.variable, as items will be assigned in call(s) to _instantiate_monitoring_input_state()
        self.variable = None

        # PARSE SPECS

        controller_specs = []
        system_specs = []
        mech_specs = []
        all_specs = []

        # Get controller's MONITORED_OUTPUT_STATES specifications (optional, so need to try)
        try:
            controller_specs = self.paramsCurrent[MONITORED_OUTPUT_STATES]
        except KeyError:
            pass

        # Get system's MONITORED_OUTPUT_STATES specifications (specified in paramClassDefaults, so must be there)
        system_specs = self.system.paramsCurrent[MONITORED_OUTPUT_STATES]

        # If controller has a MonitoredOutputStatesOption specification, remove any such spec from system specs
        if (any(isinstance(item, MonitoredOutputStatesOption) for item in controller_specs)):
            option_item = next((item for item in system_specs if isinstance(item, MonitoredOutputStatesOption)),None)
            if not option_item is None:
                del system_specs[option_item]

        # Combine controller and system specs
        all_specs = controller_specs + system_specs

        # Extract references to mechanisms and/or outputStates from any tuples
        # Note: leave tuples in all_specs for use in generating exponent and weight arrays below
        all_specs_extracted_from_tuples = []
        for item in all_specs:
            # Extract references from specification tuples
            if isinstance(item, tuple):
                all_specs_extracted_from_tuples.append(item[OBJECT])
                continue
            # Validate remaining items as one of the following:
            elif isinstance(item, (Mechanism, OutputState, MonitoredOutputStatesOption, str)):
                all_specs_extracted_from_tuples.append(item)
            # IMPLEMENTATION NOTE: This should never occur, as should have been found in _validate_monitored_state()
            else:
                raise EVCError("PROGRAM ERROR:  illegal specification ({0}) encountered by {1} "
                               "in MONITORED_OUTPUT_STATES for a mechanism, controller or system in its scope".
                               format(item, self.name))

        # Get MonitoredOutputStatesOptions if specified for controller or System, and make sure there is only one:
        option_specs = [item for item in all_specs if isinstance(item, MonitoredOutputStatesOption)]
        if not option_specs:
            ctlr_or_sys_option_spec = None
        elif len(option_specs) == 1:
            ctlr_or_sys_option_spec = option_specs[0]
        else:
            raise EVCError("PROGRAM ERROR: More than one MonitoredOutputStateOption specified in {}: {}".
                           format(self.name, option_specs))

        # Get MONITORED_OUTPUT_STATES specifications for each mechanism and outputState in the System
        # Assign outputStates to self.monitoredOutputStates
        self.monitoredOutputStates = []
        
        # Notes:
        # * Use all_specs to accumulate specs from all mechanisms and their outputStates
        #     for use in generating exponents and weights below)
        # * Use local_specs to combine *only current* mechanism's specs with those from controller and system specs;
        #     this allows the specs for each mechanism and its outputStates to be evaluated independently of any others
        controller_and_system_specs = all_specs_extracted_from_tuples.copy()

        for mech in self.system.mechanisms:

            # For each mechanism:
            # - add its specifications to all_specs (for use below in generating exponents and weights)
            # - extract references to Mechanisms and outputStates from any tuples, and add specs to local_specs
            # - assign MonitoredOutputStatesOptions (if any) to option_spec, (overrides one from controller or system)
            # - use local_specs (which now has this mechanism's specs with those from controller and system specs)
            #     to assign outputStates to self.monitoredOutputStates

            mech_specs = []
            output_state_specs = []
            local_specs = controller_and_system_specs.copy()
            option_spec = ctlr_or_sys_option_spec

            # PARSE MECHANISM'S SPECS

            # Get MONITORED_OUTPUT_STATES specification from mechanism
            try:
                mech_specs = mech.paramsCurrent[MONITORED_OUTPUT_STATES]

                if mech_specs is NotImplemented:
                    raise AttributeError

                # Setting MONITORED_OUTPUT_STATES to None specifies mechanism's outputState(s) should NOT be monitored
                if mech_specs is None:
                    raise ValueError

            # Mechanism's MONITORED_OUTPUT_STATES is absent or NotImplemented, so proceed to parse outputState(s) specs
            except (KeyError, AttributeError):
                pass

            # Mechanism's MONITORED_OUTPUT_STATES is set to None, so do NOT monitor any of its outputStates
            except ValueError:
                continue

            # Parse specs in mechanism's MONITORED_OUTPUT_STATES
            else:

                # Add mech_specs to all_specs
                all_specs.extend(mech_specs)

                # Extract refs from tuples and add to local_specs
                for item in mech_specs:
                    if isinstance(item, tuple):
                        local_specs.append(item[OBJECT])
                        continue
                    local_specs.append(item)

                # Get MonitoredOutputStatesOptions if specified for mechanism, and make sure there is only one:
                #    if there is one, use it in place of any specified for controller or system
                option_specs = [item for item in mech_specs if isinstance(item, MonitoredOutputStatesOption)]
                if not option_specs:
                    option_spec = ctlr_or_sys_option_spec
                elif option_specs and len(option_specs) == 1:
                    option_spec = option_specs[0]
                else:
                    raise EVCError("PROGRAM ERROR: More than one MonitoredOutputStateOption specified in {}: {}".
                                   format(mech.name, option_specs))

            # PARSE OUTPUT STATE'S SPECS

            # for output_state_name, output_state in list(mech.outputStates.items()):
            for output_state_name, output_state in mech.outputStates.items():

                # Get MONITORED_OUTPUT_STATES specification from outputState
                try:
                    output_state_specs = output_state.paramsCurrent[MONITORED_OUTPUT_STATES]
                    if output_state_specs is NotImplemented:
                        raise AttributeError

                    # Setting MONITORED_OUTPUT_STATES to None specifies outputState should NOT be monitored
                    if output_state_specs is None:
                        raise ValueError

                # outputState's MONITORED_OUTPUT_STATES is absent or NotImplemented, so ignore
                except (KeyError, AttributeError):
                    pass

                # outputState's MONITORED_OUTPUT_STATES is set to None, so do NOT monitor it
                except ValueError:
                    continue

                # Parse specs in outputState's MONITORED_OUTPUT_STATES
                else:

                    # Note: no need to look for MonitoredOutputStatesOption as it has no meaning
                    #       as a specification for an outputState

                    # Add outputState specs to all_specs and local_specs
                    all_specs.extend(output_state_specs)

                    # Extract refs from tuples and add to local_specs
                    for item in output_state_specs:
                        if isinstance(item, tuple):
                            local_specs.append(item[OBJECT])
                            continue
                        local_specs.append(item)

            # Ignore MonitoredOutputStatesOption if any outputStates are explicitly specified for the mechanism
            for output_state_name, output_state in list(mech.outputStates.items()):
                if (output_state in local_specs or output_state.name in local_specs):
                    option_spec = None


            # ASSIGN SPECIFIED OUTPUT STATES FOR MECHANISM TO self.monitoredOutputStates

            for output_state_name, output_state in list(mech.outputStates.items()):

                # If outputState is named or referenced anywhere, include it
                if (output_state in local_specs or output_state.name in local_specs):
                    self.monitoredOutputStates.append(output_state)
                    continue

                if option_spec is None:
                    continue
                # if option_spec is MonitoredOutputStatesOption.ONLY_SPECIFIED_OUTPUT_STATES:
                #     continue

                # If mechanism is named or referenced in any specification or it is a terminal mechanism
                elif (mech.name in local_specs or mech in local_specs or
                              mech in self.system.terminalMechanisms.mechanisms):
                    # If MonitoredOutputStatesOption is PRIMARY_OUTPUT_STATES and outputState is primary, include it 
                    if option_spec is MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES:
                        if output_state is mech.outputState:
                            self.monitoredOutputStates.append(output_state)
                            continue
                    # If MonitoredOutputStatesOption is ALL_OUTPUT_STATES, include it
                    elif option_spec is MonitoredOutputStatesOption.ALL_OUTPUT_STATES:
                        self.monitoredOutputStates.append(output_state)
                    else:
                        raise EVCError("PROGRAM ERROR: unrecognized specification of MONITORED_OUTPUT_STATES for "
                                       "{0} of {1}".
                                       format(output_state_name, mech.name))


        # ASSIGN WEIGHTS AND EXPONENTS

        # Note: these values will be superceded by any assigned as arguments to the outcome_aggregation_function
        #       if it is specified in the constructor for the mechanism

        num_monitored_output_states = len(self.monitoredOutputStates)
        exponents = np.ones((num_monitored_output_states,1))
        weights = np.ones_like(exponents)

        # Get and assign specification of exponents and weights for mechanisms or outputStates specified in tuples
        for spec in all_specs:
            if isinstance(spec, tuple):
                object_spec = spec[OBJECT]
                # For each outputState in monitoredOutputStates
                for item in self.monitoredOutputStates:
                    # If either that outputState or its owner is the object specified in the tuple
                    if item is object_spec or item.name is object_spec or item.owner is object_spec:
                        # Assign the exponent and weight specified in the tuple to that outputState
                        i = self.monitoredOutputStates.index(item)
                        exponents[i] = spec[EXPONENT]
                        weights[i] = spec[WEIGHT]

        self.paramsCurrent[OUTCOME_AGGREGATION_FUNCTION].exponents = exponents
        self.paramsCurrent[OUTCOME_AGGREGATION_FUNCTION].weights = weights


        # INSTANTIATE INPUT STATES

        # Instantiate inputState for each monitored state in the list
        # from Components.States.OutputState import OutputState
        for monitored_state in self.monitoredOutputStates:
            if isinstance(monitored_state, OutputState):
                self._instantiate_monitoring_input_state(monitored_state, context=context)
            elif isinstance(monitored_state, Mechanism):
                for output_state in monitored_state.outputStates:
                    self._instantiate_monitoring_input_state(output_state, context=context)
            else:
                raise EVCError("PROGRAM ERROR: outputState specification ({0}) slipped through that is "
                               "neither a OutputState nor Mechanism".format(monitored_state))


        if self.prefs.verbosePref:
            print ("{0} monitoring:".format(self.name))
            for state in self.monitoredOutputStates:
                exponent =  np.ndarray.item(self.paramsCurrent[OUTCOME_AGGREGATION_FUNCTION].weights[
                                                self.monitoredOutputStates.index(state)])
                weight = np.ndarray.item(self.paramsCurrent[OUTCOME_AGGREGATION_FUNCTION].exponents[
                                             self.monitoredOutputStates.index(state)])
                print ("\t{0} (exp: {1}; wt: {2})".format(state.name, exponent, weight))

        self.inputValue = self.variable.copy() * 0.0

        return self.inputStates

    def _instantiate_prediction_mechanisms(self, context=None):
        """Add prediction Process for each origin (input) Mechanism in System

        Args:
            context:
        """

        from PsyNeuLink.Components.Process import Process_Base

        # Instantiate a predictionMechanism for each origin (input) Mechanism in self.system,
        #    instantiate a Process (that maps the origin to the prediction mechanism),
        #    and add that Process to System.processes list
        self.predictionMechanisms = []
        self.predictionProcesses = []
        inputs = self.system.variable

        for mech in self.system.originMechanisms.mechanisms:

            # Get any params specified for predictionMechanism(s) by EVCMechanism
            try:
                prediction_mechanism_params = self.paramsCurrent[PREDICTION_MECHANISM_PARAMS]
            except KeyError:
                prediction_mechanism_params = {}


            # Add outputState with name based on originMechanism
            output_state_name = mech.name + '_' + kwPredictionMechanismOutput
            prediction_mechanism_params[OUTPUT_STATES] = [output_state_name]

            # Instantiate predictionMechanism
            prediction_mechanism = self.paramsCurrent[PREDICTION_MECHANISM_TYPE](
                                                            name=mech.name + "_" + kwPredictionMechanism,
                                                            params = prediction_mechanism_params,
                                                            context=context)

            # Assign list of processes for which prediction_mechanism will provide input during the simulation
            # - used in _get_simulation_system_inputs()
            # - assign copy, since don't want to include the prediction process itself assigned to mech.processes below
            prediction_mechanism.use_for_processes = list(mech.processes.copy())

            self.predictionMechanisms.append(prediction_mechanism)

            # Instantiate process with originMechanism projecting to predictionMechanism, and phase = originMechanism
            prediction_process = Process_Base(default_input_value=NotImplemented,
                                              params={
                                                  PATHWAY:[(mech, mech.phaseSpec),
                                                                   IDENTITY_MATRIX,
                                                                   (prediction_mechanism, mech.phaseSpec)]},
                                              name=mech.name + "_" + kwPredictionProcess,
                                              context=context
                                              )
            prediction_process._isControllerProcess = True
            # Add the process to the system's processes param (with None as input)
            self.system.params[kwProcesses].append((prediction_process, None))
            # Add the process to the controller's list of prediction processes
            self.predictionProcesses.append(prediction_process)

        # Re-instantiate system with predictionMechanism Process(es) added
        # MODIFIED 10/2/16 OLD:
        self.system._instantiate_processes(inputs=self.system.variable, context=context)
        # # MODIFIED 10/2/16 NEW:
        # self.system._instantiate_processes(inputs=inputs, context=context)
        # MODIFIED 10/2/16 END
        self.system._instantiate_graph(context=context)

    def _instantiate_monitoring_input_state(self, monitored_state, context=None):
        """Instantiate inputState with projection from monitoredOutputState

        Validate specification for outputState to be monitored
        Instantiate inputState with value of monitoredOutputState
        Instantiate Mapping projection to inputState from monitoredOutputState

        Args:
            monitored_state (OutputState):
            context:
        """

        self._validate_monitored_state_spec(monitored_state, context=context)

        state_name = monitored_state.name + '_Monitor'

        # Instantiate inputState
        input_state = self._instantiate_control_mechanism_input_state(state_name,
                                                                      monitored_state.value,
                                                                      context=context)

        # Instantiate Mapping Projection from monitored_state to new input_state
        from PsyNeuLink.Components.Projections.Mapping import Mapping
        Mapping(sender=monitored_state, receiver=input_state)

    def _instantiate_function(self, context=None):
        super()._instantiate_function(context=context)

    def _instantiate_attributes_after_function(self, context=None):

        super()._instantiate_attributes_after_function(context=context)

        # Insure that length of the weights and/or exponents arguments for the outcome_aggregation_function
        #    matches the number of monitoredOutputStates
        num_monitored_output_states = len(self.monitoredOutputStates)
        if not self.outcome_aggregation_function.weights is None:
            num_outcome_weights = len(self.outcome_aggregation_function.weights)
            if  num_outcome_weights != num_monitored_output_states:
                raise EVCError("The length of the weights argument {} for the {} of {} "
                               "must equal the number of its monitoredOutputStates {}".
                               format(num_outcome_weights,
                                      OUTCOME_AGGREGATION_FUNCTION,
                                      self.name,
                                      num_monitored_output_states))
        if not self.outcome_aggregation_function.exponents is None:
            num_outcome_exponents = len(self.outcome_aggregation_function.exponents)
            if  num_outcome_exponents != num_monitored_output_states:
                raise EVCError("The length of the exponents argument {} for the {} of {} "
                               "must equal the number of its control signals {}".
                               format(num_outcome_exponents,
                                      OUTCOME_AGGREGATION_FUNCTION,
                                      self.name,
                                      num_monitored_output_states))

        # Insure that length of the weights and/or exponents arguments for the cost_aggregation_function
        #    matches the number of control signals
        num_control_signals = len(self.controlSignals)
        if not self.cost_aggregation_function.weights is None:
            num_cost_weights = len(self.cost_aggregation_function.weights)
            if  num_cost_weights != num_control_signals:
                raise EVCError("The length of the weights argument {} for the {} of {} "
                               "must equal the number of its control signals {}".
                               format(num_cost_weights, COST_AGGREGATION_FUNCTION, self.name, num_control_signals))
        if not self.cost_aggregation_function.exponents is None:
            num_cost_exponents = len(self.cost_aggregation_function.exponents)
            if  num_cost_exponents != num_control_signals:
                raise EVCError("The length of the exponents argument {} for the {} of {} "
                               "must equal the number of its control signals {}".
                               format(num_cost_exponents, COST_AGGREGATION_FUNCTION, self.name, num_control_signals))


        # Map indices of outputValue to outputState(s)
        self._outputStateValueMapping = OrderedDict()
        for i in range(len(self.outputStates)):
            self._outputStateValueMapping[list(self.outputStates.keys())[i]] = i
        # for output_state in self.outputStates:
        #     self._outputStateValueMapping[list(self.outputStates.keys())[i]] = i

        self.outputValue = [None] * len(self._outputStateValueMapping)

    def _get_simulation_system_inputs(self, phase):
        """Return array of predictionMechanism values for use as inputs to processes in simulation run of System

        Returns: 2D np.array

        """

        simulation_inputs = np.empty_like(self.system.inputs, dtype=float)

        # For each prediction mechanism
        for prediction_mech in self.predictionMechanisms:

            # Get the index for each process that uses simulated input from the prediction mechanism
            for predicted_process in prediction_mech.use_for_processes:
                # process_index = self.system.processes.index(predicted_process)
                process_index = self.system._processList.processes.index(predicted_process)
                # Assign the prediction mechanism's value as the simulated input for the process
                #    in the phase at which it is used
                if prediction_mech.phaseSpec == phase:
                    simulation_inputs[process_index] = prediction_mech.value
                # For other phases, assign zero as the simulated input to the process
                else:
                    simulation_inputs[process_index] = np.atleast_1d(0)
        return simulation_inputs

    def __execute__(self,
                    variable=NotImplemented,
                    time_scale=TimeScale.TRIAL,
                    runtime_params=NotImplemented,
                    context=None):
        """Construct and search space of control signals for maximum EVC and set value of outputStates accordingly

        * Get ``allocationSamples`` for each control signal (i.e., the ControlSignal projection for each outputState in
          ``outputStates``).
        * Construct ``controlSignalSearchSpace``: a 2D np.array of control allocation policies, each policy of which
          is a different combination of values, one from the ``allocationSamples`` of each control signal.
        * Call ``system``.execute for each control allocation policy in ``controlSignalSearchSpace``
        * Store an array of values for outputStates in ``monitoredOutputStates`` (i.e., the inputStates in
          ``inputStates``) for each control allocation policy.
        * Call ``execute`` to calculate the EVC for each control allocation policy, identify the maxium, and assign to
          ``EVCmax``.
        * Set ``EVCmaxPolicy`` to the control allocation policy (outputState.values) corresponding to EVCmax
        * Set value for each control signal (outputState.value) to the values in ``EVCmaxPolicy``
        * Return ``EVCmax``

         Note:
         * runtime_params is used for self.execute (that calculates the EVC for each call to system.execute);
             it is NOT used for system.execute -- that uses the runtime_params
              provided for the Mechanisms in each Process.congiruation

        Args:
            time_scale:
            runtime_params:
            context:

        Returns (2D np.array): value of outputState for each monitored state (in self.inputStates) for EVCMax
        """

        #region CONSTRUCT SEARCH SPACE
        # IMPLEMENTATION NOTE: MOVED FROM _instantiate_function
        #                      TO BE SURE LATEST VALUES OF allocationSamples ARE USED (IN CASE THEY HAVE CHANGED)
        #                      SHOULD BE PROFILED, AS MAY BE INEFFICIENT TO EXECUTE THIS FOR EVERY RUN
        control_signal_sample_lists = []
        # Get allocationSamples for all ControlSignal Projections of all outputStates in self.outputStates
        num_output_states = len(self.outputStates)

        for output_state in self.outputStates:
            for projection in self.outputStates[output_state].sendsToProjections:
                control_signal_sample_lists.append(projection.allocationSamples)

        # Construct controlSignalSearchSpace:  set of all permutations of ControlSignal allocations
        #                                     (one sample from the allocationSample of each ControlSignal)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.controlSignalSearchSpace = \
            np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1,num_output_states)
        # END MOVE
        #endregion

        #region ASSIGN SIMULATION INPUT(S)
        # For each prediction mechanism, assign its value as input to corresponding process for the simulation
        for mech in self.predictionMechanisms:
            # For each outputState of the predictionMechanism, assign its value as the value of the corresponding
            # Process.inputState for the origin Mechanism corresponding to mech
            for output_state in mech.outputStates:
                for input_state_name, input_state in list(mech.inputStates.items()):
                    for projection in input_state.receivesFromProjections:
                        input = mech.outputStates[output_state].value
                        projection.sender.owner.inputState.receivesFromProjections[0].sender.value = input

        #endregion

        #region RUN SIMULATION

        self.EVCmax = None
        self.EVCvalues = []
        self.EVCpolicies = []

        # Reset context so that System knows this is a simulation (to avoid infinitely recursive loop)
        context = context.replace('EXECUTING', '{0} {1}'.format(self.name, kwEVCSimulation))

        if self.prefs.reportOutputPref:
            progress_bar_rate_str = ""
            search_space_size = len(self.controlSignalSearchSpace)
            progress_bar_rate = int(10 ** (np.log10(search_space_size)-2))
            if progress_bar_rate > 1:
                progress_bar_rate_str = str(progress_bar_rate) + " "
            print("\n{0} evaluating EVC for {1} (one dot for each {2}of {3} samples): ".
                  format(self.name, self.system.name, progress_bar_rate_str, search_space_size))

        # Evaluate all combinations of controlSignals (policies)
        sample = 0
        self.EVCmaxStateValues = self.variable.copy()
        self.EVCmaxPolicy = self.controlSignalSearchSpace[0] * 0.0

        # Parallelize using multiprocessing.Pool
        # NOTE:  currently fails on attempt to pickle lambda functions
        #        preserved here for possible future restoration
        if PY_MULTIPROCESSING:
            EVC_pool = Pool()
            results = EVC_pool.map(_compute_EVC, [(self, arg, runtime_params, time_scale, context)
                                                 for arg in self.controlSignalSearchSpace])

        else:

            # Parallelize using MPI
            if MPI_IMPLEMENTATION:
                Comm = MPI.COMM_WORLD
                rank = Comm.Get_rank()
                size = Comm.Get_size()

                chunk_size = (len(self.controlSignalSearchSpace) + (size-1)) // size
                print("Rank: {}\nChunk size: {}".format(rank, chunk_size))
                start = chunk_size * rank
                end = chunk_size * (rank+1)
                if start > len(self.controlSignalSearchSpace):
                    start = len(self.controlSignalSearchSpace)
                if end > len(self.controlSignalSearchSpace):
                    end = len(self.controlSignalSearchSpace)
            else:
                start = 0
                end = len(self.controlSignalSearchSpace)

            if MPI_IMPLEMENTATION:
                print("START: {0}\nEND: {1}".format(start,end))

            #region EVALUATE EVC

            # Compute EVC for each allocation policy in controlSignalSearchSpace
            # Notes on MPI:
            # * breaks up search into chunks of size chunk_size for each process (rank)
            # * each process computes max for its chunk and returns
            # * result for each chunk contains EVC max and associated allocation policy for that chunk

            result = None
            EVC_max = float('-Infinity')
            EVC_max_policy = np.empty_like(self.controlSignalSearchSpace[0])
            EVC_max_state_values = np.empty_like(self.inputValue)
            max_value_state_policy_tuple = (EVC_max, EVC_max_state_values, EVC_max_policy)
            # FIX:  INITIALIZE TO FULL LENGTH AND ASSIGN DEFAULT VALUES (MORE EFFICIENT):
            EVC_values = np.array([])
            EVC_policies = np.array([[]])

            for allocation_vector in self.controlSignalSearchSpace[start:end,:]:
            # for iter in range(rank, len(self.controlSignalSearchSpace), size):
            #     allocation_vector = self.controlSignalSearchSpace[iter,:]:

                if self.prefs.reportOutputPref:
                    increment_progress_bar = (progress_bar_rate < 1) or not (sample % progress_bar_rate)
                    if increment_progress_bar:
                        print(kwProgressBarChar, end='', flush=True)
                sample +=1

                # Calculate EVC for specified allocation policy
                result_tuple = _compute_EVC(args=(self, allocation_vector, runtime_params, time_scale, context))
                EVC, value, cost = result_tuple

                EVC_max = max(EVC, EVC_max)
                # max_result([t1, t2], key=lambda x: x1)

                # Add to list of EVC values and allocation policies if save option is set
                if self.paramsCurrent[SAVE_ALL_VALUES_AND_POLICIES]:
                    # FIX:  ASSIGN BY INDEX (MORE EFFICIENT)
                    EVC_values = np.append(EVC_values, np.atleast_1d(EVC), axis=0)
                    # Save policy associated with EVC for each process, as order of chunks
                    #     might not correspond to order of policies in controlSignalSearchSpace
                    if len(EVC_policies[0])==0:
                        EVC_policies = np.atleast_2d(allocation_vector)
                    else:
                        EVC_policies = np.append(EVC_policies, np.atleast_2d(allocation_vector), axis=0)

                # If EVC is greater than the previous value:
                # - store the current set of monitored state value in EVCmaxStateValues
                # - store the current set of controlSignals in EVCmaxPolicy
                # if EVC_max > EVC:
                if EVC == EVC_max:
                    # Keep track of state values and allocation policy associated with EVC max
                    # EVC_max_state_values = self.inputValue.copy()
                    # EVC_max_policy = allocation_vector.copy()
                    EVC_max_state_values = self.inputValue
                    EVC_max_policy = allocation_vector
                    max_value_state_policy_tuple = (EVC_max, EVC_max_state_values, EVC_max_policy)

            #endregion

            # Aggregate, reduce and assign global results

            if MPI_IMPLEMENTATION:
                # combine max result tuples from all processes and distribute to all processes
                max_tuples = Comm.allgather(max_value_state_policy_tuple)
                # get tuple with "EVC max of maxes"
                max_of_max_tuples = max(max_tuples, key=lambda max_tuple: max_tuple[0])
                # get EVCmax, state values and allocation policy associated with "max of maxes"
                self.EVCmax = max_of_max_tuples[0]
                self.EVCmaxStateValues = max_of_max_tuples[1]
                self.EVCmaxPolicy = max_of_max_tuples[2]

                if self.paramsCurrent[SAVE_ALL_VALUES_AND_POLICIES]:
                    self.EVCvalues = np.concatenate(Comm.allgather(EVC_values), axis=0)
                    self.EVCpolicies = np.concatenate(Comm.allgather(EVC_policies), axis=0)
            else:
                self.EVCmax = EVC_max
                self.EVCmaxStateValues = EVC_max_state_values
                self.EVCmaxPolicy = EVC_max_policy
                if self.paramsCurrent[SAVE_ALL_VALUES_AND_POLICIES]:
                    self.EVCvalues = EVC_values
                    self.EVCpolicies = EVC_policies
            # TEST PRINT:
            print("\nFINAL:\n\tmax tuple:\n\t\tEVC_max: {}\n\t\tEVC_max_state_values: {}\n\t\tEVC_max_policy: {}".
                  format(max_value_state_policy_tuple[0],
                         max_value_state_policy_tuple[1],
                         max_value_state_policy_tuple[2]),
                  flush=True)


            # FROM MIKE ANDERSON (ALTERNTATIVE TO allgather:  REDUCE USING A FUNCTION OVER LOCAL VERSION)
            # a = np.random.random()
            # mymax=Comm.allreduce(a, MPI.MAX)
            # print(mymax)

        if self.prefs.reportOutputPref:
            print("\nEVC simulation completed")
#endregion

        #region ASSIGN CONTROL SIGNAL VALUES

        # Assign allocations to controlSignals (self.outputStates) for optimal allocation policy:
        # MODIFIED 10/25/16 OLD:
        # for output_state in self.outputStates.values():
            # output_state.value = np.atleast_1d(next(iter(self.EVCmaxPolicy)))
        # MODIFIED 10/25/16 NEW:
        for output_state_name, output_state in self.outputStates.items():
            output_state.value = np.atleast_1d(self.EVCmaxPolicy[self._outputStateValueMapping[output_state_name]])
        # MODIFIED 10/25/16 END

        # MODIFIED 11/4/16 NEWER:
        EVCmaxStateValue = iter(self.EVCmaxStateValues)
        # MODIFIED 11/4/16 END
        # Assign max values for optimal allocation policy to self.inputStates (for reference only)
        for i in range(len(self.inputStates)):
            # list(self.inputStates.values())[i].value = np.atleast_1d(self.EVCmaxStateValues[i])
            # # MODIFIED 10/25/16 OLD:
            # next(iter(self.inputStates.values())).value = np.atleast_1d(next(iter(self.EVCmaxStateValues)))
            # # MODIFIED 10/25/16 NEW:
            # self.inputStates[list(self.inputStates.keys())[i]].value =
            #  np.atleast_1d(next(iter(self.EVCmaxStateValues)))
            # MODIFIED 11/4/16 NEWER:
            self.inputStates[list(self.inputStates.keys())[i]].value = np.atleast_1d(next(EVCmaxStateValue))
            # MODIFIED 10/25/16 END


        # Report EVC max info

        if self.prefs.reportOutputPref:
            print ("\nMaximum EVC for {0}: {1}".format(self.system.name, float(self.EVCmax)))
            print ("ControlSignal allocation(s) for maximum EVC:")
            for i in range(len(self.outputStates)):
                print("\t{0}: {1}".format(list(self.outputStates.values())[i].name,
                                        self.EVCmaxPolicy[i]))
            print()

        #endregion


        # TEST PRINT:
        # print ("\nEND OF TRIAL 1 EVC outputState: {0}\n".format(self.outputState.value))


        # # MODIFIED 10/5/16 OLD:
        # return self.EVCmax
        # # MODIFIED 10/5/16 NEW:
        # return self.EVCmaxPolicy
        # MODIFIED 10/25/15 NEWER:

        # for name in self._outputStateValueMapping:
        #     self.outputValue[self._outputStateValueMapping[name]] =
        #                    self.EVCmaxPolicy[self._outputStateValueMapping[name]]
        # Get EVCmaxPolicy for each outputState (which are in an OrderedDict) and assign to corresponding outputValue
        for i in range(len(self.outputStates)):
            self.outputValue[self._outputStateValueMapping[list(self.outputStates.keys())[i]]] = self.EVCmaxPolicy[i]
        return self.outputValue

        # for i in range(len(self.EVCmaxPolicy)):
        #     self.outputValue[self.outputState[self._outputStateValueMapping[i]]] = self.EVCmaxPolicy[i]

        # MODIFIED 10/5-25/16 END

    # IMPLEMENTATION NOTE: NOT IMPLEMENTED, AS PROVIDED BY params[FUNCTION]
    # IMPLEMENTATION NOTE: RETURNS EVC FOR CURRENT STATE OF monitoredOutputStates
    #                      self.value IS SET TO THIS, WHICH IS NOT THE SAME AS outputState(s).value
    #                      THE LATTER IS STORED IN self.allocationPolicy
    # def execute(self, params, time_scale, context):
    #     """Calculate EVC for values of monitored states (in self.inputStates)
    #     """

    # def _update_output_states(self, time_scale=None, context=None):
    #     """Assign outputStateValues to allocationPolicy
    #
    #     This method overrides super._update_output_states, instantiate allocationPolicy attribute
    #         and assign it outputStateValues
    #     Notes:
    #     * this is necessary, since self.execute returns (and thus self.value equals) the EVC for monitoredOutputStates
    #         and a given allocation policy (i.e., set of outputState values / ControlSignal specifications);
    #         this devaites from the usual case in which self.value = self.execute = self.outputState.value(s)
    #         therefore, self.allocationPolicy is used to represent to current set of self.outputState.value(s)
    #
    #     Args:
    #         time_scale:
    #         context:
    #     """
    #     for i in range(len(self.allocationPolicy)):
    #         self.allocationPolicy[i] = next(iter(self.outputStates.values())).value
    #
    #     super()._update_output_states(time_scale= time_scale, context=context)

    def _add_monitored_states(self, states_spec, context=None):
        """Validate and then instantiate outputStates to be monitored by EVC

        Use by other objects to add a state or list of states to be monitored by EVC
        states_spec can be a Mechanism, OutputState or list of either or both
        If item is a Mechanism, each of its outputStates will be used
        All of the outputStates specified must be for a Mechanism that is in self.System

        Args:
            states_spec (Mechanism, MechanimsOutputState or list of either or both:
            context:
        """
        states_spec = list(states_spec)
        self._validate_monitored_state_spec(states_spec, context=context)
        # FIX: MODIFIED 7/18/16:  NEED TO IMPLEMENT  _instantiate_monitored_output_states
        #                         SO AS TO CALL _instantiate_input_states()
        self._instantiate_monitored_output_states(states_spec, context=context)

def _compute_EVC(args):
    """compute EVC for a specified allocation policy

    IMPLEMENTATION NOTE:  implemented as a function so it can be used with multiprocessing Pool

    Args:
        ctlr (EVCMechanism)
        allocation_vector (1D np.array): allocation policy for which to compute EVC
        runtime_params (dict): runtime params passed to ctlr.update
        time_scale (TimeScale): time_scale passed to ctlr.update
        context (value): context passed to ctlr.update

    Returns (float, float, float):
        (EVC_current, total_current_value, total_current_control_costs)

    """
    ctlr, allocation_vector, runtime_params, time_scale, context = args

    # #TEST PRINT
    # print("-------- EVC SIMULATION --------");

    # Implement the current policy over ControlSignal Projections
    for i in range(len(ctlr.outputStates)):
        # # MODIFIED 10/25/16 OLD:
        # next(iter(ctlr.outputStates.values())).value = np.atleast_1d(allocation_vector[i])
        # MODIFIED 10/25/16 NEW:
        ctlr.outputStates[list(ctlr.outputStates.keys())[i]].value = np.atleast_1d(allocation_vector[i])
        # MODIFIED 10/25/16 END

    # Execute self.system for the current policy
    time_step_buffer = CentralClock.time_step
    for i in range(ctlr.system._phaseSpecMax+1):
        CentralClock.time_step = i
        simulation_inputs = ctlr._get_simulation_system_inputs(phase=i)
        ctlr.system.execute(inputs=simulation_inputs, time_scale=time_scale, context=context)
    CentralClock.time_step = time_step_buffer

    # Get control cost for this policy
    # Iterate over all outputStates (controlSignals)
    j = 0
    ctlr_output_states_iter = iter(ctlr.outputStates.values())
    for i in range(len(ctlr.outputStates)):
        # Get projections for this outputState
        output_state_projections = next(ctlr_output_states_iter).sendsToProjections
        # Iterate over all projections for the outputState
        for projection in output_state_projections:
            # Get ControlSignal cost
            ctlr.controlSignalCosts[j] = np.atleast_2d(projection.cost)
            j += 1

    total_current_control_cost = ctlr.paramsCurrent[COST_AGGREGATION_FUNCTION].function(ctlr.controlSignalCosts)

    # Get value of current policy = weighted sum of values of monitored output states
    # Note:  ctlr.inputValue = value of monitored output states (self.inputStates) = self.variable
    ctlr._update_input_states(runtime_params=runtime_params, time_scale=time_scale,context=context)
    total_current_value = ctlr.paramsCurrent[OUTCOME_AGGREGATION_FUNCTION].function(variable=ctlr.inputValue,
                                                                                    params=runtime_params,
                                                                                    time_scale=time_scale,
                                                                                    context=context)

    # Calculate EVC for the result (default: total value - total cost)
    EVC_current = ctlr.function([total_current_value,
                                 -total_current_control_cost])

    # #TEST PRINT:
    # print("allocation_vector: {}".format(allocation_vector))
    # print("total_current_control_cost: {}".format(total_current_control_cost))
    # print("total_current_value: {}".format(total_current_value))
    # print("EVC_current: {}".format(EVC_current))

    if PY_MULTIPROCESSING:
        return

    else:
        return (EVC_current, total_current_value, total_current_control_cost)
