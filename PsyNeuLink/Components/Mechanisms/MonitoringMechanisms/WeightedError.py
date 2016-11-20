# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  WeightedError *******************************************************

"""
Overview
--------

WeightedError mechanisms monitor the outputState of a ProcessingMechanism in a :doc:`System` or :doc:`Process` ,
and compare this to a target provided as input to the ``run`` method of the system or process when it is executed.
The comparison can be done using subtraction or division.

.. _WeightedError_Creating_A_WeightedError:

Creating A WeightedError
---------------------

A WeightedError mechanism can be created either directly, by calling its constructor, or using the :class:`mechanism`
function and specifying "WeightedError" as its ``mech_spec`` argument.  It can also be created by :ref:`in context
specification of a LearningSignal <_Projection_Creating_A_Projection>` for a projection  to a ProcessingMechanism in
a system or process that has at least one other ProcessingMechanism to which it projects. One or more WeightedErrors
are also created automatically when a system or a process is created for which learning is specified; each is
assigned a projection from the outputState of a ProcessingMechanism that receives a Mapping projection being
learned, and a LearningSignal projection to that Mapping projection
(see :ref:`learning in a process <Process_Learning>`, and
:ref:`automatic creation of LearningSignals <LearningSignal_Automatic_Creation> for details).

.. _WeightedError_Execution

Execution
---------

A WeightedError always executes after the mechanism it is monitoring.  It's ``function`` computes the contribution of
each sender element (the rows of its :keyword:`NEXT_LEVEL_PROJECTION` matrix attribute) to the error values of the
receivers (elements of the ``error_signal`` array, columns of the ``NEXT_LEVEL_PROJECTION matrix), weighted by the
association of each sender with each receiver (specified in NEXT_LEVEL_PROJECTION.matrix).  The ``function`` returns
an array with the weighted errors for each sender element, which is placed in its ``value`` and ``outputValue``
attributes, and the value of of its ouptputState.

.. _WeightedError_Class_Reference:

Class Reference
---------------
"""



# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
# from PsyNeuLink.Components.States.InputState import InputState

# WeightedError output (used to create and name outputStates):
WEIGHTED_ERRORS = 'WeightedErrors'
NEXT_LEVEL_PROJECTION = 'next_level_projection'

# WeightedError output indices (used to index output values):
class WeightedErrorOutput(AutoNumber):
    ERROR_SIGNAL = ()

class WeightedErrorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class WeightedError(MonitoringMechanism_Base):
    """
    WeightedError{     \
    error_signal=None, \
    params=None,       \
    name=None,         \
    prefs=None)

    Implements WeightedError subclass of MonitoringMechanism

    COMMENT:
        Description:
            WeightedError is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the Function class
            It's function computes the contribution of each sender element (rows of the NEXT_LEVEL_PROJECTION param)
                to the error values of the receivers
                 (elements of the error_signal array, columns of the matrix of the NEXT_LEVEL_PROJECTION param),
                weighted by the association of each sender with each receiver (specified in NEXT_LEVEL_PROJECTION.matrix)
            The function returns an array with the weighted errors for each sender element

        Class attributes:
            + componentType (str): WeightedError
            + classPreference (PreferenceSet): WeightedError_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + variableClassDefault (1D np.array):
            + paramClassDefaults (dict): {NEXT_LEVEL_PROJECTION: Mapping}
            + paramNames (dict): names as above

        Class methods:
            None

        MechanismRegistry:
            All instances of WeightedError are registered in MechanismRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    - error_signal (1D np.array)
    - params (dict):
        + NEXT_LEVEL_PROJECTION (Mapping Projection):
            projection, the matrix of which is used to calculate error_array
            width (number of columns) must match error_signal

    Attributes
    ----------

    variable : 1d np.array
        error_signal used by ``function``.

    value : XXXX
        output of ``function``.

    name : str : default WeightedError-<index>
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

    componentType = "WeightedError"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'WeightedErrorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    variableClassDefault = [0]  # error_signal

    # WeightedError parameter assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NEXT_LEVEL_PROJECTION: None,
        OUTPUT_STATES:[WEIGHTED_ERRORS],
    })

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 error_signal=NotImplemented,
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

        """Compute error_signal for current layer from derivative of error_signal at next layer

        Return weighted error array.
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
        error_array = np.dot(next_level_matrix, error_derivative)

        # Compute summed error for use by callers to decide whether to update
        self.summedErrorSignal = np.sum(error_array)

        # Assign output values
        self.outputValue[WeightedErrorOutput.ERROR_SIGNAL.value] = error_array

        return self.outputValue