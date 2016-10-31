# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **********************************************  Mapping **************************************************************
#

from PsyNeuLink.Functions.Projections.Projection import *
from PsyNeuLink.Functions.Utilities.Utility import *

class MappingError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class Mapping(Projection_Base):
    """Implement projection conveying values from output of a mechanism to input of another (default: IdentityMapping)

    Description:
        The Mapping class is a functionType in the Projection category of Function,
        It's function conveys (and possibly transforms) the OutputState.value of a sender
            to the InputState.value of a receiver

    Instantiation:
        - Mapping Projections can be instantiated in one of several ways:
            - directly: requires explicit specification of the sender
            - as part of the instantiation of a mechanism:
                the mechanism outputState will automatically be used as the receiver:
                    if the mechanism is being instantiated on its own, the sender must be explicity specified
                    if the mechanism is being instantiated within a pathway:
                        if a sender is explicitly specified for the mapping, that will be used;
                        otherwise, if it is the first mechanism in the list, process.input will be used as the sender;
                        otherwise, the preceding mechanism in the list will be used as the sender

    Initialization arguments:
        - sender (State) - source of projection input (default: systemDefaultSender)
        - receiver: (State or Mechanism) - destination of projection output (default: systemDefaultReceiver)
            if it is a Mechanism, and has >1 inputStates, projection will be mapped to the first inputState
# IMPLEMENTATION NOTE:  ABOVE WILL CHANGE IF SENDER IS ALLOWED TO BE A MECHANISM (SEE FIX ABOVE)
        - params (dict) - dictionary of projection params:
# IMPLEMENTTION NOTE: ISN'T kwProjectionSenderValue REDUNDANT WITH sender and receiver??
            + kwProjectionSenderValue (list): (default: [1]) ?? OVERRIDES sender ARG??
            + FUNCTION (Utility): (default: LinearMatrix)
            + FUNCTION_PARAMS (dict): (default: {MATRIX: IDENTITY_MATRIX})
# IMPLEMENTATION NOTE:  ?? IS THIS STILL CORRECT?  IF NOT, SEARCH FOR AND CORRECT IN OTHER CLASSES
        - name (str) - if it is not specified, a default based on the class is assigned in register_category
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
# IMPLEMENTATION NOTE:  AUGMENT SO THAT SENDER CAN BE A Mechanism WITH MULTIPLE OUTPUT STATES, IN WHICH CASE:
#                RECEIVER MUST EITHER BE A MECHANISM WITH SAME NUMBER OF INPUT STATES AS SENDER HAS OUTPUTSTATES
#                (FOR WHICH SENDER OUTPUTSTATE IS MAPPED TO THE CORRESONDING RECEIVER INPUT STATE
#                              USING THE SAME MAPPING PROJECTION MATRIX, OR AN ARRAY OF THEM)
#                OR BOTH MUST BE 1D ARRAYS (I.E., SINGLE VECTOR)
#       SHOULD BE CHECKED IN OVERRIDE OF _validate_variable THEN HANDLED IN _instantiate_sender and _instantiate_receiver


    Parameters:
        The default for FUNCTION is LinearMatrix using IDENTITY_MATRIX:
            the sender state is passed unchanged to the receiver's state
# IMPLEMENTATION NOTE:  *** CONFIRM THAT THIS IS TRUE:
        FUNCTION can be set to another function, so long as it has type kwMappingFunction
        The parameters of FUNCTION can be set:
            - by including them at initialization (param[FUNCTION] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[FUNCTION].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)



    ProjectionRegistry:
        All Mapping projections are registered in ProjectionRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Mapping projections can be named explicitly (using the name='<name>' argument).  If this argument is omitted,
        it will be assigned "Mapping" with a hyphenated, indexed suffix ('Mapping-n')

    Class attributes:
        + className = MAPPING
        + functionType = PROJECTION
        # + defaultSender (State)
        # + defaultReceiver (State)
        + paramClassDefaults (dict)
            paramClassDefaults.update({
                               FUNCTION:LinearMatrix,
                               FUNCTION_PARAMS: {
                                   # LinearMatrix.kwReceiver: receiver.value,
                                   LinearMatrix.MATRIX: LinearMatrix.DEFAULT_MATRIX},
                               kwProjectionSender: kwInputState, # Assigned to class ref in __init__ module
                               kwProjectionSenderValue: [1],
                               })
        + paramNames (dict)
        # + senderDefault (State) - set to Process inputState
        + classPreference (PreferenceSet): MappingPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

    Class methods:
        function (executes function specified in params[FUNCTION]

    Instance attributes:
        + sender (State)
        + receiver (State)
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramsCurrent (dict) - set currently in effect
        + variable (value) - used as input to projection's function
        + value (value) - output of function
        + monitoringMechanism (MonitoringMechanism) - source of error signal for matrix weight changes
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying MappingPreferenceSet

    Instance methods:
        none
    """

    functionType = MAPPING
    className = functionType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({FUNCTION: LinearMatrix,
                               kwProjectionSender: kwOutputState, # Assigned to class ref in __init__.py module
                               kwProjectionSenderValue: [1],
                               })
    @tc.typecheck
    def __init__(self,
                 sender=NotImplemented,
                 receiver=NotImplemented,
                 # sender=None,
                 # receiver=None,
                 matrix=DEFAULT_MATRIX,
                 param_modulation_operation=ModulationOperation.ADD,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """
IMPLEMENTATION NOTE:  *** DOCUMENTATION NEEDED (SEE CONTROL SIGNAL)

        :param sender:
        :param receiver:
        :param params:
        :param name:
        :param context:
        :return:
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(
                                                 # function=function,
                                                 function_params={MATRIX: matrix},
                                                 param_modulation_operation=param_modulation_operation,
                                                 params=params)

        self.monitoringMechanism = None

        # If sender or receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        if sender is NotImplemented or receiver is NotImplemented:
        # if not sender or receiver:
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = self
            self.init_args['name'] = name
            # Delete these as they have been moved to params dict (and will not be recognized by Projection.__init__)
            del self.init_args['matrix']
            del self.init_args['param_modulation_operation']

            # Flag for deferred initialization
            self.value = kwDeferredInit
            return

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super(Mapping, self).__init__(sender=sender,
                                      receiver=receiver,
                                      params=params,
                                      name=name,
                                      prefs=prefs,
                                      context=self)

    def _instantiate_receiver(self, context=None):
        """Handle situation in which self.receiver was specified as a Mechanism (rather than State)

        If receiver is specified as a Mechanism, it is reassigned to the (primary) inputState for that Mechanism
        If the Mechanism has more than one inputState, assignment to other inputStates must be done explicitly
            (i.e., by: _instantiate_receiver(State)

        """
        # Assume that if receiver was specified as a Mechanism, it should be assigned to its (primary) inputState
        if isinstance(self.receiver, Mechanism):
            if (len(self.receiver.inputStates) > 1 and
                    (self.prefs.verbosePref or self.receiver.prefs.verbosePref)):
                print("{0} has more than one inputState; {1} was assigned to the first one".
                      format(self.receiver.owner.name, self.name))
            self.receiver = self.receiver.inputState

        self.reshapedWeightMatrix = False

        # Get sender and receiver lengths
        # Note: if either is a scalar, manually set length to 1 to avoid TypeError in call to len()
        try:
            mapping_input_len = len(self.variable)
        except TypeError:
            mapping_input_len = 1
        try:
            receiver_len = len(self.receiver.variable)
        except TypeError:
            receiver_len = 1

        # Compare length of Mapping output and receiver's variable to be sure matrix has proper dimensions
        try:
            mapping_output_len = len(self.value)
        except TypeError:
            mapping_output_len = 1

        # FIX: CONVERT ALL REFS TO paramsCurrent[FUNCTION_PARAMS][MATRIX] TO self.matrix (CHECK THEY'RE THE SAME)
        # FIX: CONVERT ALL REFS TO matrix_spec TO self._matrix_spec
        # FIX: CREATE @PROPERTY FOR self._learning_spec AND ASSIGN IN INIT??
        # FIX: HOW DOES mapping_output_len RELATE TO receiver_len?/

        if self._matrix_spec is AUTO_ASSIGN_MATRIX:
            if mapping_input_len == receiver_len:
                self._matrix_spec = IDENTITY_MATRIX
            else:
                self._matrix_spec = FULL_CONNECTIVITY_MATRIX

        # Length of the output of the projection doesn't match the length of the receiving input state
        #    so consider reshaping the matrix
        if mapping_output_len != receiver_len:

            if 'projection' in self.name or 'Projection' in self.name:
                projection_string = ''
            else:
                projection_string = 'projection'

            if self._matrix_spec is IDENTITY_MATRIX:
                # Identity matrix is not reshapable
                raise ProjectionError("Output length ({}) of \'{}{}\' from {} to mechanism \'{}\'"
                                      " must equal length of it inputState ({}) to use {}".
                                      format(mapping_output_len,
                                             self.name,
                                             projection_string,
                                             self.sender.name,
                                             self.receiver.owner.name,
                                             receiver_len,
                                             IDENTITY_MATRIX))
            else:
                # Flag that matrix is being reshaped
                self.reshapedWeightMatrix = True
                if self.prefs.verbosePref:
                    print("Length ({}) of the output of {}{} does not match the length ({}) "
                          "of the inputState for the receiver {}; the width of the matrix (number of columns); "
                          "the width of the matrix (number of columns) will be adjusted to accomodate the receiver".
                          format(mapping_output_len,
                                 self.name,
                                 projection_string,
                                 receiver_len,
                                 self.receiver.owner.name))

                self.matrix = get_matrix(self._matrix_spec, mapping_input_len, receiver_len, context=context)

                # Since matrix shape has changed, output of self.function may have chnaged, so update self.value
                self._update_value()

        super()._instantiate_receiver(context=context)

    def execute(self, input=NotImplemented, params=NotImplemented, time_scale=None, context=None):
        # IMPLEMENT: check for flag that it has changed (needs to be implemented, and set by ErrorMonitoringMechanism)
        # DOCUMENT: update, including use of monitoringMechanism.monitoredStateChanged and weightChanged flag
        """
        If there is an functionParrameterStates[LEARNING_SIGNAL], update the matrix parameterState:
                 it should set params[PARAMETER_STATE_PARAMS] = {kwLinearCombinationOperation:SUM (OR ADD??)}
                 and then call its super().execute
           - use its value to update MATRIX using CombinationOperation (see State update ??execute method??)

        """

        # ASSUMES IF self.monitoringMechanism IS ASSIGNED AND parameterState[MATRIX] HAS BEEN INSTANTIATED
        # THAT LEARNING SIGNAL EXISTS
        # AVERTS DUCK TYPING WHICH OTHERWISE WOULD BE REQUIRED FOR THE MOST FREQUENT CASES (I.E., NO LearningSignal)

        # Check whether weights changed
        if self.monitoringMechanism and self.monitoringMechanism.summedErrorSignal:

            # Assume that if monitoringMechanism attribute is assigned,
            #    both a LearningSignal and parameterState[MATRIX] to receive it have been instantiated
            matrix_parameter_state = self.parameterStates[MATRIX]

            # Assign current MATRIX to parameter state's baseValue, so that it is updated in call to execute()
            matrix_parameter_state.baseValue = self.matrix

            # Pass params for parameterState's funtion specified by instantiation in LearningSignal
            weight_change_params = matrix_parameter_state.paramsCurrent

            # Update parameter state: combines weightChangeMatrix from LearningSignal with matrix baseValue
            matrix_parameter_state.update(weight_change_params, context=context)

            # Update MATRIX
            self.matrix = matrix_parameter_state.value

        return self.function(self.sender.value, params=params, context=context)

    @property
    def matrix(self):
        return self.function.__self__.matrix

    @matrix.setter
    def matrix(self, matrix):
        if not (isinstance(matrix, np.matrix) or
                    (isinstance(matrix,np.ndarray) and matrix.ndim == 2) or
                    (isinstance(matrix,list) and np.array(matrix).ndim == 2)):
            raise MappingError("Matrix parameter for {} ({}) Mapping projection must be "
                               "an np.matrix, a 2d np.array, or a correspondingly configured list".
                               format(self.name, matrix))
        self.function.__self__.matrix = matrix

    @property
    def _matrix_spec(self):
        """Returns matrix specification in self.paramsCurrent[FUNCTION_PARAMS][MATRIX]

        Returns matrix param for Mapping, getting second item if it is
         a ParamValueprojection or unnamed (matrix, projection) tuple
        """
        return get_function_param(self.paramsCurrent[FUNCTION_PARAMS][MATRIX])

    @_matrix_spec.setter
    def _matrix_spec(self, value):
        """Assign matrix specification for self.paramsCurrent[FUNCTION_PARAMS][MATRIX]

        Assigns matrix param for Mapping, assiging second item if it is
         a ParamValueprojection or unnamed (matrix, projection) tuple
        """
        if isinstance(self.paramsCurrent[FUNCTION_PARAMS][MATRIX], ParamValueProjection):
            self.paramsCurrent[FUNCTION_PARAMS][MATRIX].value =  value

        elif (isinstance(self.paramsCurrent[FUNCTION_PARAMS][MATRIX], tuple) and
                      len(self.paramsCurrent[FUNCTION_PARAMS][MATRIX]) is 2 and
                  (self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1] in {MAPPING, CONTROL_SIGNAL, LEARNING_SIGNAL}
                   or isinstance(self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1], Projection) or
                       (inspect.isclass(self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1]) and
                            issubclass(self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1], Projection)))
              ):
            self.paramsCurrent[FUNCTION_PARAMS][MATRIX] = (value, self.paramsCurrent[FUNCTION_PARAMS][MATRIX][1])

        else:
            self.paramsCurrent[FUNCTION_PARAMS][MATRIX] = value
