# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  WeightedErrorMechanism ***************************************************

"""
Overview
--------

A WeightedErrorMechanism monitors the `outputState <OutputState>` of it's
`errorSource <WeightedErrorMechanism.errorSource>`, a `ProcessingMechanism` that projects to another mechanism in a
`process <Process>`.  It computes the contribution of each element of the output of the
`errorSource <WeightedErrorMechanism.errorSource>` to the error in the output of
the mechanism to which the `errorSource <WeightedErrorMechanism.errorSource>` projects
(the `weighted_error_signal`).  The WeightedErrorMechanism `function <WeightedErrorMechanism.function>` returns an error
array that can be used by a `LearningProjection` to adjust a `MappingProjection` to the
`errorSource <WeightedErrorMechanism.errorSource>`, so as to reduce its future contribution to the
`weighted_error_signal`.

.. _WeightedError_Creation:

Creating A WeightedErrorMechanism
---------------------------------

A WeightedErrorMechanism can be created directly by calling its constructor, or using the `mechanism` function and
specifying :keyword:`WeightedErrorMechanism` as its `mech_spec` argument.  It can also be created by
`in context specification of a LearningProjection <Projection_Creation>` for a MappingProjection; that MappingProjection
must project to a ProcessingMechanism that is not a `TERMINAL` mechanism. One or more WeightedErrorMechanisms,
are also `created automatically <LearningProjection_Automatic_Creation>`, along with associated LearningProjections,
when learning is specified for a process or system with any
COMMENT:
    `INTERNAL` [ADD THIS WHEN mechanisms CAN RECEIVE MORE THAN ONE DESIGNATION
COMMENT
ProcessingMechanisms that are not `TERMINAL` or `SINGLETON` mechanisms.

.. _WeightedError_Structure

Structure
---------

A WeightedErrorMechanism has a single `inputState <InputState>` and a single `outputState <OutputState>` (labelled
`WEIGHTED_ERROR`).  It also has four primary attributes:  an

  * `errorSource <WeightedErrorMechanism.errorSource>`: the ProcessingMechanism for which the WeightedErrorMechanism
    computes the `weighted_error_Signal`;
  ..
  * `next_mechanism`: a ProcessingMechanism to which the `errorSource <WeightedErrorMechanism.errorSource>` projects;
  ..
  * `projection_to_next_mechanism`: the `MappingProjection` from the `errorSource <WeightedErrorMechanism.errorSource>`
    to the `next_mechanism`;
  ..
  * `error_signal <WeightedErrorMechanism.error_signal>`: the error for the output of the `next_mechanism`;
  ..
  * `weighted_error_Signal`: the error for the output of the `errorSource`, weighted by the contribution that each
    element of its elements makes to the `error_signal <WeightedErrorMechanism.error_signal>`.

The WeightedErrorMechanism receives a MappingProjection to its inputState from the
`errorSource  <WeightedErrorMechanism.errorSource>`, and its `WEIGHTED_ERROR` outputState
is assigned a `LearningProjection` to the MappingProjection it is being used to train (and that projects to the
`errorSource <WeightedErrorMechanism.errorSource>`)

COMMENT:
    Each row of the  `matrix <MappingProjection.MappingProjection.matrix>` for that MappingProjection corresponds to an
    element of the `value` for the `errorSource <WeightedErrorMechanism.errorSource>`, each column corresponds to an
    element of the `value` of the mechanism to which it projects, and each element of the matrix is the weight of the
    association between the two.
COMMENT

.. _WeightedError_Execution

Execution
---------

A WeightedErrorMechanism always executes after the mechanism it is monitoring.  It's
`function <WeightedErrorMechanism.function>` computes its `weighted_error_signal`: the contribution that each element
of the output of the `errorSource <WeightedErrorMechanism.errorSource>` makes to the
`error_signal <WeightedErrorMechanism>` (i.e., the error for the output of the `next_mechanism`), weighted both by the
strength of its association to each element of the `next_mechanism` (specified in `projection_to_next_mechanism`),
and the differential of the :keyword:`function` of `next_mechanism`.  This implements a core computation of the
`Generalized Delta Rule <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_  implemented by the
`BackPropagation` learning function. The `function <WeightedErrorMechanism>` of the WeightedErrorMechanism returns
the `weighted_error_signal`, that is assigned to its `value <WeightedErrorMechanism.value>` and
`outputValue <WeightedErrorMechanism.outputValue>` attributes, and the :keyword:`value` of of its `WEIGHTED_ERROR`
outputState.

.. _WeightedError_Class_Reference:

Class Reference
---------------

"""



# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
# from PsyNeuLink.Components.States.InputState import InputState

# WeightedErrorMechanism output (used to create and name outputStates):
WEIGHTED_ERRORS = 'WeightedErrors'
PROJECTION_TO_NEXT_MECHANISM = 'projection_to_next_mechanism'

# WeightedErrorMechanism output indices (used to index output values):
class WeightedErrorOutput(AutoNumber):
    ERROR_SIGNAL = ()

class WeightedErrorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class WeightedErrorMechanism(MonitoringMechanism_Base):
    """
    WeightedErrorMechanism( \
    error_signal=None,      \
    params=None,            \
    name=None,              \
    prefs=None)

    Implements WeightedErrorMechanism subclass of MonitoringMechanism.

    COMMENT:
        Description:
            WeightedErrorMechanism is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the
            Function class
            It's function computes the contribution of each sender element (rows of PROJECTION_TO_NEXT_MECHANISM param)
               to the error values of the receivers
                 (elements of the error_signal array, columns of the matrix of the PROJECTION_TO_NEXT_MECHANISM param),
               weighted by the association of each sender with each receiver
               (specified in PROJECTION_TO_NEXT_MECHANISM.matrix)
            The function returns an array with the weighted errors for each sender element

        Class attributes:
            + componentType (str): WeightedErrorMechanism
            + classPreference (PreferenceSet): WeightedError_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + variableClassDefault (1D np.array):
            + paramClassDefaults (dict): {PROJECTION_TO_NEXT_MECHANISM: MappingProjection}
            + paramNames (dict): names as above

        Class methods:
            None

        MechanismRegistry:
            All instances of WeightedErrorMechanism are registered in MechanismRegistry, which maintains an
                entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    error_signal : 1d np.array

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the mechanism, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a params dict).
        Includes the following entry:

        * :keyword:`PROJECTION_TO_NEXT_MECHANISM`:  MappingProjection;
          its :py:data:`matrix <MappingProjection.MappingProjection.matrix>` parameter is used to calculate the
          `weighted_error_signal <WeightedErrorMechanism.weighted_error_signal>`; it's width (number of columns)
          must match the length of ``error_signal``.

    name : str : default WeightedErrorMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

    errorSource : ProcessingMechanism
        the mechanism that projects to the WeightedErrorMechanism,
        and for which it calculates the :py:data:`weighted_error_signal <WeightedErrorMechanism.weighted_error_signal>`.

    projection_to_next_mechanism : MappingProjection
        projection from the ``errorSource to the next mechanism in the process;  its ``matrix`` parameter is
        used to calculate the :py:data:`weighted_error_signal <WeightedErrorMechanism.weighted_error_signal>`.

    variable : 1d np.array
        error_signal from mechanism to which projection_to_next_mechanism projects;  used by ``function`` to compute
        :py:data:`weighted_error_signal <WeightedErrorMechanism.weighted_error_signal>`.

    value : 1d np.array
        output of ``function``, same as :py:data:`weighted_error_signal <WeightedErrorMechanism.weighted_error_signal>`.

    weighted_error_signal : 1d np.array
        specifies the weighted contribution made by each element of the ``value`` of the error source to the
        ``error_signal`` received from the next mechanism in the process (the one to which ``projection_to_next_mechanism``
        projects.

    outputValue : List[1d np.array]
        single item in list is ``errorSignal`` (output of ``function``;  same as ``value``).

    name : str : default WeightedErrorMechanism-<index>
        the name of the mechanism.
        Specified in the `name` argument of the constructor for the projection;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for mechanism.
        Specified in the `prefs` argument of the constructor for the mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    """

    componentType = WEIGHTED_ERROR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'WeightedErrorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    variableClassDefault = [0]  # error_signal

    # WeightedErrorMechanism parameter assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_TO_NEXT_MECHANISM: None,
        OUTPUT_STATES:[WEIGHTED_ERRORS],
    })

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 error_signal=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences and call super.__init__
        """

        self.function = self.execute

# # FIX: MODIFY get_param_value_for_keyword TO TAKE PARAMS DICT

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)

        super().__init__(variable=error_signal,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure that width of PROJECTION_TO_NEXT_MECHANISM matrix equals length of error_signal

        Validate that width of matrix for projection in PROJECTION_TO_NEXT_MECHANISM param equals length of error_signal
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        cols = target_set[PROJECTION_TO_NEXT_MECHANISM].matrix.shape[1]
        error_signal_len = len(self.variable[0])
        if  cols != error_signal_len:
            raise WeightedErrorError("Number of columns ({}) of weight matrix for {}"
                                     " must equal length of error_signal ({})".
                                     format(cols,self.name,error_signal_len))

    def _instantiate_attributes_before_function(self, context=None):
        self.next_mechanism = self.paramsCurrent[PROJECTION_TO_NEXT_MECHANISM].receiver.owner
        super()._instantiate_attributes_before_function(context=context)

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):

        """Compute weighted_error_signal for errorSource from derivative of error_signal for mechanism it projects to.

        Return weighted_error_signal.
        """

        if not context:
            context = EXECUTING + self.name

        self._check_args(variable=variable, params=runtime_params, context=context)

        # Get error signal from monitoring mechanism for next mechanism in the process
        error = self.variable[0]

        # Get weight matrix for projection from next mechanism in the process (to one after that)
        next_level_matrix = self.paramsCurrent[PROJECTION_TO_NEXT_MECHANISM].matrix

        # Get output of the next mechanism in the process
        next_level_output = self.next_mechanism.outputState.value

        # Get derivative for next mechanism's function
        derivative_fct = self.next_mechanism.function_object.derivative

        # Compute derivative of error with respect to current output of next mechanism
        output_derivative = derivative_fct(output=next_level_output)
        error_derivative = error * output_derivative

        # Compute error terms for each unit of current mechanism weighted by contribution to error in the next one
        self.weighted_error_signal = np.dot(next_level_matrix, error_derivative)

        # Compute summed error for use by callers to decide whether to update
        self.summedErrorSignal = np.sum(self.weighted_error_signal)

        return self.weighted_error_signal

