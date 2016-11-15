# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  Mapping **************************************************************

"""
.. _Mapping_Overview:

Overview
--------

Mapping projections transmit value from an outputState of one ProcessingMechanism (its ``sender``) to the inputState of
another (its ``receiver``).  Its default ``function`` is :class:`LinearMatrix`, which uses the projection's ``matrix``
parameter to transform an array received from its ``sender``, transforms it, and transmits the result to its
``receiver``.

.. _Mapping_Creating_A_Mapping_Projection:

Creating a Mapping Projection
-----------------------------

COMMENT:
    ??LEGACY:
    - as part of the instantiation of a mechanism:
        the mechanism outputState will automatically be used as the receiver:
            if the mechanism is being instantiated on its own, the sender must be explicity specified
COMMENT

A mapping projection can be created in any of the ways that can be used for creating a
:ref:`projection <_Projection_Creating_A_Projection>) or in specifying a projection in the
:ref:`pathway <_Process_Projections>` of a process. Mapping projections are also automatically created by
PsyNeuLink in a number of circumstances (matrix types are described in :ref:`Mapping_Structure):

* in a **process**, between adjacent mechanisms in the ``pathway`` for which none has been assigned;
  the matrix will use :keyword:`AUTO_ASSIGN_MATRIX`, which determines the appropriate matrix by context.
..
* by a **ControlMechanism**, from outputStates listed in its ``monitoredOutputStates`` attribute to assigned
  inputStates in the ControlMechanism (see :ref:`ControlMechanism_Creating_A_ControlMechanism`);  a
  :keyword:`IDENTITY_MATRIX` will be used.

* by a **LearningSignal**, from a mechanism that is the source of an error signal, to a :doc:`MonitoringMechanism`
  that is used to evaluate that error and generate a learning signal from it (see [LINK]);  the matrix used
  depends on the ``function`` parameter of the :doc:`LearningSignal`.[LINK]

When a mapping projection is created, its ``matrix`` and ``param_modulation_operation`` attributes can be specified,
or they can be assigned by default (see below).

.. _Mapping_Structure:

Structure
---------

In addition to its ``function``, mapping projections use the following two the primary parameters:

``matrix``

  Used by ``function`` to execute a matrix transformation of its input.  It can be assigned a list of 1d arrays,
  an np.ndarray, np.matrix, a function that resolves to one of these, or one of the following keywords:

  .. _Matrix_Keywords:

  * :keyword:`IDENTITY_MATRIX` - a square matrix of 1's; this requires that the length of the sender and receiver
    values are the same.
  * :keyword:`FULL_CONNECTIVITY_MATRIX` - a matrix that has a number of rows equal to the length of the sender's value,
    and a number of columns equal to the length of the receiver's value, all the elements of which are 1's.
  * :keyword:`RANDOM_CONNECTIVITY_MATRIX` - a matrix that has a number of rows equal to the length of the sender's value,
    and a number of columns equal to the length of the receiver's value, all the elements of which are filled with
    random values uniformly distributed between 0 and 1.
  * :keyword:`AUTO_ASSIGN_MATRIX` - if the sender and receiver are of equal length, an  :keyword:`IDENTITY_MATRIX`
    is assigned;  otherwise, it a :keyword:`FULL_CONNECTIVITY_MATRIX` is assigned.
  * :keyword:`DEFAULT_MATRIX` - used if no matrix specification is provided in the constructor;  it presently
    assigns an keyword:`IDENTITY_MATRIX`.
  ..
  PsyNeuLink also provides a convenience function, :class:`random_matrix`, that can be used to generate a random matrix
  sized for a sender, receiver, with random numbers drawn from a uniform distribution within a specified range and
  with a specified offset.


``parameter_modulation_operation``

  Used to determine how the value of any projections to the :doc:`parameterState` for the ``matrix`` parameter
  influence it.  For example, this is used for a :doc:`LearningSignal` projection to apply weight changes to
  ``matrix`` during learning.  ``parameter_modulation_operation`` must be assigned a value of
  :class:`ModulationOperation`, and the operation is always applied in an element-wise (Hadamard[LINK]) fashion.
  The default operation is ``ADD``.

.. _Projection_Execution:

Execution
---------

A mapping projection uses its ``function`` and ``matrix`` parameters to transform the value of its ``sender``,
and assign this as the variable for its ``receiver``.  When it is executed, updating the ``matrix`` parameterState will
cause the value of any projections (e.g., a LearningSignal) it receives to be applied to the matrix. This will bring
into effect any changes that occurred during the previous execution (e.g., due to learning).  Because of :ref:`Lazy
Evaluation`[LINK], those changes will only be effective after the current execution (in other words, inspecting
``matrix`` will not show the effects of projections to its parameterState until the mapping projection has been
executed).

.. _Projection_Class_Reference:


Class Reference
---------------

"""

from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Functions.Function import *

class MappingError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class Mapping(Projection_Base):
    """
    Mapping(                                             \
     sender=None,                                        \
     receiver=None,                                      \
     matrix=DEFAULT_MATRIX,                              \
     param_modulation_operation=ModulationOperation.ADD, \
     params=None,                                        \
     name=None,                                          \
     prefs=None)

    Implements a projection that uses LinearMatrix function to transform value of sender, and assigns it to receiver

    COMMENT:
        Description:
            The Mapping class is a type in the Projection category of Component,
            It's function conveys (and possibly transforms) the OutputState.value of a sender
                to the InputState.value of a receiver

            IMPLEMENTATION NOTE:
                AUGMENT SO THAT SENDER CAN BE A Mechanism WITH MULTIPLE OUTPUT STATES, IN WHICH CASE:
                    RECEIVER MUST EITHER BE A MECHANISM WITH SAME NUMBER OF INPUT STATES AS SENDER HAS OUTPUTSTATES
                        (FOR WHICH SENDER OUTPUTSTATE IS MAPPED TO THE CORRESPONDING RECEIVER INPUT STATE
                            USING THE SAME MAPPING PROJECTION MATRIX, OR AN ARRAY OF THEM)
                    OR BOTH MUST BE 1D ARRAYS (I.E., SINGLE VECTOR)
                    SHOULD BE CHECKED IN OVERRIDE OF _validate_variable
                        THEN HANDLED IN _instantiate_sender and _instantiate_receiver

        Class attributes:
            + className = MAPPING
            + componentType = PROJECTION
            + paramClassDefaults (dict)
                paramClassDefaults.update({
                                   FUNCTION:LinearMatrix,
                                   FUNCTION_PARAMS: {
                                       # LinearMatrix.kwReceiver: receiver.value,
                                       LinearMatrix.MATRIX: LinearMatrix.DEFAULT_MATRIX},
                                   PROJECTION_SENDER: kwInputState, # Assigned to class ref in __init__ module
                                   PROJECTION_SENDER_VALUE: [1],
                                   })
            + classPreference (PreferenceSet): MappingPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            function (executes function specified in params[FUNCTION]
    COMMENT

    Arguments
    ---------

    sender : Optional[OutputState or Mechanism]
        Source of projection input.  If a mechanism is specified, its primary outputState will be used.  If it is not
        specified, it will be assigned in the context in which the projection is used.

    receiver: Optional[InputState or Mechanism]
        Destination of projection output.  If a mechanism is specified, its primary inputState will be used.  If it
        is not specified, it will be assigned in the context in which the projection is used.

    matrix : list, np.ndarray, np.matrix, function or keyword : default :keyword:`DEFAULT_MATRIX`
        Matrix used by ``function`` (default: LinearCombination) to transform value of ``sender``

    param_modulation_operation : ModulationOperation : default ModulationOperation.ADD
        Operation used to combine the value of any projections to the matrix's parameterState with the ``matrix``.
        Most commonly used with LearningSignal projections.

    params : Optional[Dict[param keyword, param value]]
        Dictionary that can be used to specify the parameters for the projection, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a parms dict).[LINK]
        By default, it contains an entry for the projection's default ``function`` assignment (LinearCombination);

    name : str : default Transfer-<index>
        String used for the name of the mapping projection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : Process.classPreferences]
        Preference set for the mapping projection.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    Attributes
    ----------

    monitoringMechanism : MonitoringMechanism
        source of error signal for matrix weight changes when :ref:`learning <LearningSignal> is used.




    """

    componentType = MAPPING
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({FUNCTION: LinearMatrix,
                               PROJECTION_SENDER: OUTPUT_STATE, # Assigned to class ref in __init__.py module
                               PROJECTION_SENDER_VALUE: [1],
                               })
    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 matrix=DEFAULT_MATRIX,
                 param_modulation_operation=ModulationOperation.ADD,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(
                                                 # function=function,
                                                 function_params={MATRIX: matrix},
                                                 param_modulation_operation=param_modulation_operation,
                                                 params=params)

        self.monitoringMechanism = None

        # If sender or receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        # if sender is NotImplemented or receiver is NotImplemented:
        if not sender or not receiver:
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
        If there is an functionParameterStates[LEARNING_SIGNAL], update the matrix parameterState:
                 it should set params[PARAMETER_STATE_PARAMS] = {kwLinearCombinationOperation:SUM (OR ADD??)}
                 and then call its super().execute
           - use its value to update MATRIX using CombinationOperation (see State update ??execute method??)

        Assumes that if self.monitoringMechanism is assigned *and* parameterState[MATRIX] has been instantiated
        then learningSignal exists;  this averts duck typing which otherwise would be required for the most
        frequent cases (i.e., *no* learningSignal).

        """

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
