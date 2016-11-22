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

A WeightedErrorMechanism monitors the outputState of it's ``errorSource``: a ProcessingMechanism that projects to
another mechanism in a :doc:`Process`.  It computes the contribution of each element of the output of the
``errorSource`` to the error in the output of the mechanism to which the ``errorSource`` projects
(the ``weightedErrorSignal``).  The WeightedErrorMechanism ``function``returns an error array that can be used by a
:doc:`LearningSignal` to adjust a Mapping projection the ``errorSource``, so as to reduce its future contribution to
the weightedErrorSignal.

.. _WeightedError_Creation:

Creating A WeightedErrorMechanism
---------------------------------

A WeightedErrorMechanism can be created either directly, by calling its constructor, or using the :class:`mechanism`
function and specifying "WeightedErrorMechanism" as its ``mech_spec`` argument.  It can also be created by :ref:`in
context specification of a LearningSignal <_Projection_Creation>` for a projection  to a ProcessingMechanism in
a process that has at least one other ProcessingMechanism to which it projects. One or more WeightedErrors
are also created automatically when a process system is created for which learning is specified; each is
assigned a projection from the outputState of a ProcessingMechanism that receives a Mapping projection being
learned, and a LearningSignal projection to that Mapping projection
(see :ref:`learning in a process <Process_Learning>`, and
:ref:`automatic creation of LearningSignals <LearningSignal_Automatic_Creation> for details).

.. _WeightedError_Structure

Structure
---------

A WeightedErrorMechanism has a single inputState, a :keyword:`NEXT_LEVEL_PROJECTION` parameter, and a single
(:keyword:`WEIGHTED_ERROR`) outputState.  The **inputState** receives a Mapping projection from its ``errorSource`` --
the Processing mechanism for which it computes the error.  :keyword:`NEXT_LEVEL_PROJECTION` is assigned to the Mapping
projection from the primary outputState of the ``errorSource`` to the next mechanism in the process.  Each row of it's
``matrix`` parameter corresponds to an element of the ``value`` of the ``errorSource``, each column corresponds to an
element of the ``value`` of the mechanism to which it projects, and each element of the matrix is the weight of the
association between the two.  The **outputState** of a WeightedErrorMechanism is typically assigned a
:doc:`LearningSignal` projection that is used to modify the ``matrix`` parameter of a Mapping projection to the
``errorSource`` (as shown in :ref:`this figure <Process_Learning_Figure>`).

.. _WeightedError_Execution

Execution
---------

A WeightedErrorMechanism always executes after the mechanism it is monitoring.  It's ``function`` computes the
contribution of each element of the ``value`` of the ``errorSource`` to the ``error_signal``:  the error associated
with each element of the ``value`` of the mechanism to which the ``errorSource`` projects, scaled both by the weight of
its association to that element  specified by from :keyword:`NEXT_LEVEL_PROJECTION`) and the differential of the
``function`` for that mechanism.  This implements a core computation of the Generalized Delta Rule (or
"backpropagation") learning algorithm (REFS AND [LINK]). The ``function`` returns an array with the weighted errors
for each element of the ``errorSource``, which is placed in its ``value`` and  ``outputValue`` attributes,
and the value of of its (:keyword:`WEIGHTED_ERROR) outputState.

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
            + paramClassDefaults (dict): {NEXT_LEVEL_PROJECTION: Mapping}
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
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a parms dict).
        Includes the following entry:

        * :keyword:`NEXT_LEVEL_PROJECTION`:  Mapping projection;
          its ``matrix`` parameter is used to calculate the ``weightedErrorSignal``;
          it's width (number of columns) must match the length of ``error_signal``.

    Attributes
    ----------

    errorSource : ProcessingMechanism
        the mechanism that projects to the WeightedErrorMechanism,
        and for which it calculates the ``weightedErrorSignal``.
        
    next_level_projection : Mapping projection
        projection from the ``errorSource to the next mechanism in the process;  its ``matrix`` parameter is 
        used to calculate the ``weightedErrorSignal``.

    variable : 1d np.array
        error_signal from mechanism to which next_level_projection projects;  used by ``function`` to compute
        weightedErrorSignal.

    value : 1d np.array
        output of ``function``, same as ``weightedErrorSignal``.

    weightedErrorSignal : 1d np.array
        specifies the weighted contribution made by each element of the ``value`` of the error source to the
        ``error_signal`` received from the next mechanism in the process (the one to which ``next_level_projection``
        projects.

    outputValue : List[1d np.array]
        single item in list is ``errorSignal`` (output of ``function``;  same as ``value``).

    name : str : default WeightedErrorMechanism-<index>
        the name of the mechanism.
        Specified in the name argument of the call to create the projection;
        if not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the PreferenceSet for mechanism.
        Specified in the prefs argument of the call to create the mechanism;
        if it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    """

    componentType = "WeightedErrorMechanism"

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

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
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

    def _instantiate_attributes_before_function(self, context=None):

        # Map indices of output to outputState(s)
        self._outputStateValueMapping = {}
        self._outputStateValueMapping[WEIGHTED_ERRORS] = WeightedErrorOutput.ERROR_SIGNAL.value

        super()._instantiate_attributes_before_function(context=context)


    def __execute__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=None):

        """Compute weightedErrorSignal for errorSource from derivative of error_signal for mechanism it projects to.

        Return weightedErrorSignal.
        """

        if not context:
            context = kwExecuting + self.name

        self._check_args(variable=variable, params=params, context=context)

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

        # Assign output values
        self.outputValue[WeightedErrorOutput.ERROR_SIGNAL.value] = self.weightedErrorSignal

        return self.outputValue