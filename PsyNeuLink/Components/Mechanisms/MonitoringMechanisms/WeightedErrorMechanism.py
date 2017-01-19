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

A WeightedErrorMechanism monitors the outputState of it's :py:data:`errorSource <WeightedErrorMechanism.errorSource>`:
a ProcessingMechanism that projects to another mechanism in a :doc:`Process`.  It computes the contribution of each
element of the output of the :py:data:`errorSource <WeightedErrorMechanism.errorSource>` to the error in the output of
the mechanism to which the :py:data:`errorSource <WeightedErrorMechanism.errorSource>` projects
(the :py:data:`weightedErrorSignal <WeightedErrorMechanism.weightedErrorSignal>`).  The WeightedErrorMechanism
``function``returns an error array that can be used by a :doc:`LearningProjection` to adjust a MappingProjection the
:py:data:`errorSource <WeightedErrorMechanism.errorSource>`, so as to reduce its future contribution to the
:py:data:`weightedErrorSignal <WeightedErrorMechanism.weightedErrorSignal>`.

.. _WeightedError_Creation:

Creating A WeightedErrorMechanism
---------------------------------

A WeightedErrorMechanism can be created either directly, by calling its constructor, or using the :class:`mechanism`
function and specifying "WeightedErrorMechanism" as its ``mech_spec`` argument.  It can also be created by :ref:`in
context specification of a LearningProjection <_Projection_Creation>` for a projection  to a ProcessingMechanism in
a process that has at least one other ProcessingMechanism to which it projects. One or more WeightedErrors
are also created automatically when a process system is created for which learning is specified; each is
assigned a projection from the outputState of a ProcessingMechanism that receives a MappingProjection being
learned, and a LearningProjection to that MappingProjection
(see :ref:`learning in a process <Process_Learning>`, and
:ref:`automatic creation of LearningSignals <LearningProjection_Automatic_Creation> for details).

.. _WeightedError_Structure

Structure
---------

A WeightedErrorMechanism has a single inputState, a :keyword:`NEXT_LEVEL_PROJECTION` parameter, and a single
(:keyword:`WEIGHTED_ERROR`) outputState.  The **inputState** receives a MappingProjection from its
:py:data:`errorSource <WeightedErrorMechanism.errorSource>` -- the Processing mechanism for which it computes the
error.  :keyword:`NEXT_LEVEL_PROJECTION` is assigned to the MappingProjection projection from the
:ref:`primary outputState <OutputState_Primary>` of the :py:data:`errorSource <WeightedErrorMechanism.errorSource>`
to the next mechanism in the process.  Each row of it's :py:data:`matrix <MappingProjection.MappingProjection.matrix>`
parameter corresponds to an element of the ``value`` of the :py:data:`errorSource <WeightedErrorMechanism.errorSource>`
each column corresponds to an element of the ``value`` of the mechanism to which it projects, and each element of the
matrix is the weight of the association between the two.  The **outputState** of a WeightedErrorMechanism is
typically assigned a :doc:`LearningProjection` that is used to modify the
:py:data:`matrix <MappingProjection.MappingProjection.matrix>` parameter of a MappingProjection to the
:py:data:`errorSource <WeightedErrorMechanism.errorSource>` (as shown in :ref:`this figure <Process_Learning_Figure>`).

.. _WeightedError_Execution

Execution
---------

A WeightedErrorMechanism always executes after the mechanism it is monitoring.  It's ``function`` computes the
contribution of each element of the ``value`` of the :py:data:`errorSource <WeightedErrorMechanism.errorSource>`
to the ``error_signal``:  the error associated with each element of the ``value`` of the mechanism to which the
:py:data:`errorSource <WeightedErrorMechanism.errorSource>` projects, scaled both by the weight of its association to
that element specified by from :keyword:`NEXT_LEVEL_PROJECTION`) and the differential of the
``function`` for that mechanism.  This implements a core computation of the
`Generalized Delta Rule <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_  (or "backpropagation")
learning algorithm. The ``function`` returns an array with the weighted errors for each element of the
:py:data:`errorSource <WeightedErrorMechanism.errorSource>`, which is placed in its ``value``
and  :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` attributes, and the value of of its
(:keyword:`WEIGHTED_ERROR`) outputState.

.. _WeightedError_Class_Reference:

Class Reference
---------------

"""



# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
# from PsyNeuLink.Components.States.InputState import InputState

# WeightedErrorMechanism output (used to create and name outputStates):
WEIGHTED_ERRORS = 'WeightedErrors'
NEXT_LEVEL_PROJECTION = 'next_level_projection'

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
    WeightedErrorMechanism(     \
    error_signal=None, \
    params=None,       \
    name=None,         \
    prefs=None)

    Implements WeightedErrorMechanism subclass of MonitoringMechanism.

    COMMENT:
        Description:
            WeightedErrorMechanism is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the
            Function class
            It's function computes the contribution of each sender element (rows of the NEXT_LEVEL_PROJECTION param)
               to the error values of the receivers
                 (elements of the error_signal array, columns of the matrix of the NEXT_LEVEL_PROJECTION param),
               weighted by the association of each sender with each receiver (specified in NEXT_LEVEL_PROJECTION.matrix)
            The function returns an array with the weighted errors for each sender element

        Class attributes:
            + componentType (str): WeightedErrorMechanism
            + classPreference (PreferenceSet): WeightedError_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + variableClassDefault (1D np.array):
            + paramClassDefaults (dict): {NEXT_LEVEL_PROJECTION: MappingProjection}
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

        * :keyword:`NEXT_LEVEL_PROJECTION`:  MappingProjection;
          its :py:data:`matrix <MappingProjection.MappingProjection.matrix>` parameter is used to calculate the
          :py:data:`weightedErrorSignal <WeightedErrorMechanism.weightedErrorSignal>`; it's width (number of columns)
          must match the length of ``error_signal``.

    name : str : default WeightedErrorMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

    errorSource : ProcessingMechanism
        the mechanism that projects to the WeightedErrorMechanism,
        and for which it calculates the :py:data:`weightedErrorSignal <WeightedErrorMechanism.weightedErrorSignal>`.
        
    next_level_projection : MappingProjection
        projection from the ``errorSource to the next mechanism in the process;  its ``matrix`` parameter is 
        used to calculate the :py:data:`weightedErrorSignal <WeightedErrorMechanism.weightedErrorSignal>`.

    variable : 1d np.array
        error_signal from mechanism to which next_level_projection projects;  used by ``function`` to compute
        :py:data:`weightedErrorSignal <WeightedErrorMechanism.weightedErrorSignal>`.

    value : 1d np.array
        output of ``function``, same as :py:data:`weightedErrorSignal <WeightedErrorMechanism.weightedErrorSignal>`.

    weightedErrorSignal : 1d np.array
        specifies the weighted contribution made by each element of the ``value`` of the error source to the
        ``error_signal`` received from the next mechanism in the process (the one to which ``next_level_projection``
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
        (see :py:class:`PreferenceSet <LINK>` for details).


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
        NEXT_LEVEL_PROJECTION: None,
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
        """Insure that width of NEXT_LEVEL_PROJECTION matrix equals length of error_signal

        Validate that width of matrix for projection in NEXT_LEVEL_PROJECTION param equals length of error_signal
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        cols = target_set[NEXT_LEVEL_PROJECTION].matrix.shape[1]
        error_signal_len = len(self.variable[0])
        if  cols != error_signal_len:
            raise WeightedErrorError("Number of columns ({}) of weight matrix for {}"
                                     " must equal length of error_signal ({})".
                                     format(cols,self.name,error_signal_len))

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):

        """Compute weightedErrorSignal for errorSource from derivative of error_signal for mechanism it projects to.

        Return weightedErrorSignal.
        """

        if not context:
            context = EXECUTING + self.name

        self._check_args(variable=variable, params=runtime_params, context=context)

        # Get error signal from monitoring mechanism for next mechanism in the process
        error = self.variable[0]

        # Get weight matrix for projection from next mechanism in the process (to one after that)
        next_level_matrix = self.paramsCurrent[NEXT_LEVEL_PROJECTION].matrix

        # Get output of the next mechanism in the process
        next_level_output = self.paramsCurrent[NEXT_LEVEL_PROJECTION].receiver.owner.outputState.value

        # Get derivative for next mechanism's function
        derivative_fct = self.paramsCurrent[NEXT_LEVEL_PROJECTION].receiver.owner.function_object.derivative

        # Compute derivative of error with respect to current output of next mechanism
        output_derivative = derivative_fct(output=next_level_output)
        error_derivative = error * output_derivative

        # Compute error terms for each unit of current mechanism weighted by contribution to error in the next one
        self.weightedErrorSignal = np.dot(next_level_matrix, error_derivative)

        # Compute summed error for use by callers to decide whether to update
        self.summedErrorSignal = np.sum(self.weightedErrorSignal)

        return self.weightedErrorSignal

