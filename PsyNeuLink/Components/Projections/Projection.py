# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **********************************************  Projection ***********************************************************

"""
..
    Sections of this document:
      * :ref:`Projection_Overview`
      * :ref:`Projection_Creating_A_Projection`
      * :ref:`Projection_Structure`
         * :ref:`Projection_Sender`
         * :ref:`Projection_Receiver`
      * :ref:`Projection_Execution`
      * :ref:`Projection_Class_Reference`

.. _Projection_Overview:

Overview
--------

Projections allow information to be passed between mechanisms.  A projection takes an input from the output of
one mechanism (its ``sender``), and does whatever conversion is needed to transmit that information as the input
to another mechanism (its ``receiver``).  There are three types of projections that serve difference purposes:

* :doc:`Mapping`
    These take the ouptut of one :doc:`ProcessingMechanism`, convert this by convolving it with the projection's
    ``matrix`` parameter, and transmit this as input to another ProcessingMechanism.  Typically, Mapping projections
    are used to connect the mechanisms in the ``pathway`` of a :doc:`process`.
..
* :doc:`ControlSignal`
    Thess take a "control allocation" specification — usually the ouptput of a :doc:`ControlMechanism` — and transmit
    this to the parameterState of ProcessingMechanism, which uses this to modulate the value of the corresponding
    parameter of the mechanism's function.  ControlSignals projections are typically used in the context of a
    :doc:`System`.
..
* :doc:`Learning`
    These take an "error signal" — usually the output of a :doc:`MonitoringMechanism` — and transmit this to the
    parameterState of a :doc:`Mapping` projection, which uses this to modify its ``matrix`` parameter.
    ControlSignal projections are typically used in the context of a :doc:`Process` or :doc:`System` that uses learning.

COMMENT:
* Gating: takes an input signal and uses it to modulate the inputState and/or outputState of the receiver
COMMENT

.. _Projection_Creating_A_Projection:

Creating a Projection
---------------------

Projections can be created in several ways.  The simplest is to use the standard Python method of calling the
subclass for the desired type of projection.  However, projections can also be specified "in context," for example
in the ``pathway`` attribute of a process, or in a tuple with the specification of a function parameter.[LINK]
This can be done using a call to the projection subclass, or one of the following ways:

  * name of an **existing projection**:
  ..
  * name of a **projection type** (subclass);
  ..
  * **specification dictionary** -- this can contain an entry specifying the type of projection,
    and/or entries specifying the value of parameters used to instantiate it.
    These should take the following form:

      * :keyword:`PROJECTION_TYPE`: <name of a projection type>

          if this entry is absent, a default projection will be created that is appropriate for the context
          (for example, a Mapping projection for an inputState, and a ControlSignal projection for a parameterState).

      * :keyword:`PROJECTION_PARAMS`: Dict[projection argument, argument value]

          the key for each entry of the dict must be the name of a projection parameter (see :class:`Projection_Base`
          below), and the value should be the value of the parameter.  It can contain any of the standard parameters
          for instantiating a projection (see :class:`Projection_Base`) or ones specific to a particular type of
          projection (see documentation for subclass).  Note that parameter values in the specification dict will be
          used to instantiate the projection.  These can be overridden during execution by specifying
          :ref:`Mechanism_Runtime_parameters` for the projection, either when calling the * ``execute`` method for a
          :class:`mechanism` directly, or where it is specified in the ``pathway`` of a :class:`process`.

  * **automatically** -- PsyNeuLink will automatically create projections under some circumstances.  For example,
    a process automatically generates a  :class:`Mapping` projection between adjacent mechanisms in its pathway if
    none is specified; and :class:`LearningSignal`  projections are automatically generated when :ref:`Process_Learning`
    is specified for a process.  Creating a :class:`state` will also automatically generate a projection and a sender
    mechanism, if none is specified (the type of projection and its sender mechanism depend on the type of state --
    see state subclasses for details).

.. _Projection_Structure:

Structure
---------

In addition to its ``function``, a projections has two core components:

.. _Projection_Sender:

Sender
~~~~~~

This must be an :class:`OutputState`.  The projection is assigned to the sender's ``sendsToProjections`` list, and the
sender's ``value`` is used as the ``variable`` (input) for projection's ``function``.  A sender can be specified as:

  * the name of an existing inputState;
  ..
  * a string used to name a default instance of InputState (if the owner for the inputState can't be inferred
    from the context, a default mechanism will be created as the owner for the inputState, the type of which
    is determined by the projection's type — see [LINK]);
  ..
  * the name of an existing mechanism (to which a default outputState will be added);
  ..
  * a specification dictionary (see _State_Creating_A_State).

If a sender is not specified, or its specification creates a default and
 paramClassDefaults[kwProjectionSender] is used to assign a default appropriate
to the type of projection (see [LINK]).


.. _Projection_Receiver:

Receiver
~~~~~~~~

This must be an :class:`InputState` or a :class:`ParameterState`.  The projection is assigned to the receiver's
``receivesFromProjections`` list, and the output of the projection's ``function`` is transmitted to its receiver,
where it is combined with the input from any other projections to generate its ``variable``.  A receiver can be
specified as:

  * the name of an existing outputState;
  ..
  * the name of an existing mechanism or projection, for which a default state will be created and added
    (whether the object can be a mechanism or projection, and the type of state that will be created for it
    is determined by the type of projection — see subclass for details).
  ..
  * a specification dictionary (see _State_Creating_A_State).
  ..
  .. note::
     a receiver **must** be specified for a projection;  PsyNeuLink cannot create a default.  This adheres to the
     principle of :ref:"Lazy_Evaluation" which, here, means that objects can create other objects from which they
     *expect* input, but cannot *impose* the creation of "downstream" objects.

COMMENT:
    If the ``receiver`` of a projection is specified as a projection or mechanism, the type of state created and added
    to the mechanism depends on the type of projection:
        Mapping projection:
            receiver = <Mechanism>.inputState
        ControlSignal projection:
            sender = <Mechanism>.outputState
            receiver = <Mechanism>.parameterState if there is a corresponding parameter; otherwise, an error occurs
        LearningSignal projection:
            sender = <Mechanism>.outputState
            receiver = <Mapping projection>.parameterState IF AND ONLY IF there is a single one
                        that is a ParameterState;  otherwise, an exception is raised
COMMENT

.. _Projection_Execution:

Execution
---------

A projection cannot be executed directly.  It is executed when the state to which it projects — its ``receiver`` —
is updated;  that occurs when the state's owner mechanism is executed.  When a projection executes, it gets the value
of its ``sender``, assigns this as the variable for its ``function``, calls the function, and assigns the result as
the variable of its ``receiver``.  The function of a projection converts the value received from its  ``sender`` to a
form suitable as input to its ``receiver``.

.. _Projection_Class_Reference:

"""


from collections import OrderedDict

from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Globals.Registry import register_category

ProjectionRegistry = {}

kpProjectionTimeScaleLogEntry = "Projection TimeScale"

PROJECTION_SPEC_KEYWORDS = {AUTO_ASSIGN_MATRIX,
                            DEFAULT_MATRIX,
                            IDENTITY_MATRIX,
                            FULL_CONNECTIVITY_MATRIX,
                            RANDOM_CONNECTIVITY_MATRIX,
                            LEARNING_SIGNAL,
                            CONTROL_SIGNAL}

class ProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# Projection factory method:
# def projection(name=NotImplemented, params=NotImplemented, context=None):
#         """Instantiates default or specified subclass of Projection
#
#         If called w/o arguments or 1st argument=NotImplemented, instantiates default subclass (ParameterState)
#         If called with a name string:
#             - if registered in ProjectionRegistry class dictionary as name of a subclass, instantiates that class
#             - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
#         If a params dictionary is included, it is passed to the subclass
#
#         :param name:
#         :param param_defaults:
#         :return:
#         """
#
#         # Call to instantiate a particular subclass, so look up in MechanismRegistry
#         if name in ProjectionRegistry:
#             return ProjectionRegistry[name].mechanismSubclass(params)
#         # Name is not in MechanismRegistry or is not provided, so instantiate default subclass
#         else:
#             # from Components.Defaults import DefaultProjection
#             return DefaultProjection(name, params)
#

class Projection_Base(Projection):
# DOCUMENT: (lazy updating of projections re: parameter updating (e.g., matrix param of Mapping with learning).
    """
    Projection_Base(  \
    receiver,         \
    sender=None,      \
    params=None,      \
    name=None,        \
    prefs=None)


    Abstract class definition for

    .. note::
       Projections should NEVER be instantiated by a direct call to the base class.
       They should be instantiated by calling the desired subclass or the other methods for specifying a projection
       (see [LINK]).


    COMMENT:
        Description:
            Projection category of Component class (default type:  Mapping)

        Class attributes
        ----------------
            + componentCategory (str): kwProjectionFunctionCategory
            + className (str): kwProjectionFunctionCategory
            + suffix (str): " <className>"
            + registry (dict): ProjectionRegistry
            + classPreference (PreferenceSet): ProjectionPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
            + variableClassDefault (value): [0]
            + requiredParamClassDefaultTypes = {kwProjectionSender: [str, Mechanism, State]}) # Default sender type
            + paramClassDefaults (dict)
            + paramNames (dict)
            + FUNCTION (Function class or object, or method)

        Class methods
        -------------
            None

        ProjectionRegistry
        ------------------
            All Projections are registered in ProjectionRegistry, which maintains a dict for each subclass,
              a count for all instances of that type, and a dictionary of those instances
    COMMENT

    Attributes
    ----------

    variable : value
        Input to projection, received from outputState.value of sender.

    sender : State
        State (of a mechanism) from which projection receives its input.

    receiver : State
        State (of a mechanism or projection) to which projection sends its output.

    value : value
        Output of projection, transmitted to inputState.variable of receiver.

    COMMENT:
        params : Dict[param arg, parm value]
            kwProjectionSender:<Mechanism or State class or object>
                This is populated by __init__ with the default sender state for each subclass.
                It is used if sender arg is not provided.
                If it is different than the default, it overrides the sender arg even if that is provided.
            kwProjectionSenderValue:<value>
                Use to instantiate ProjectionSender (i.e., a default sender)

        paramsCurrent : Dict[param arg, parm value]
            Current value of all params for instance.

        paramInstanceDefaults : Dict[param arg, parm value]
            Defaults for instance (created and validated in Components init).

        paramNames : List[str]
            list of keys for the params in paramInstanceDefaults.

            .. _stateRegistry (Registry): registry containing a dict for the projection's parameterStates, that has
                an instance dict of the parameterStates and a count of them
                Note: registering instances of parameterStates with the projection (rather than in the StateRegistry)
                      allows the same name to be used for parameterStates belonging to different projections
                      without adding index suffixes for the name across projections
                      while still indexing multiple uses of the same base name within a projection.
    COMMENT:

    name : str : default <Projection subclass>-<index>
        Name of the projection.
        Specified in the name argument of the call to create the projection;  if not is specified,
        a default is assigned by ProjectionRegistry based on the projection's subclass
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        Preference set for the projection.
        Specified in the prefs argument of the call to create the projection;  if it is not specified, a default is
        assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    """

    color = 0

    componentCategory = kwProjectionFunctionCategory
    className = componentCategory
    suffix = " " + className

    registry = ProjectionRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    variableClassDefault = [0]

    requiredParamClassDefaultTypes = Component.requiredParamClassDefaultTypes.copy()
    requiredParamClassDefaultTypes.update({kwProjectionSender: [str, Mechanism, State]}) # Default sender type

    def __init__(self,
                 receiver,
                 sender=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Assign sender, receiver, and execute method and register mechanism with ProjectionRegistry

        This is an abstract class, and can only be called from a subclass;
           it must be called by the subclass with a context value

# DOCUMENT:  MOVE TO ABOVE, UNDER INSTANTIATION
        Initialization arguments:
            - sender (Mechanism, State or dict):
                specifies source of input to projection (default: senderDefault)
            - receiver (Mechanism, State or dict)
                 destination of projection (default: none)
            - params (dict) - dictionary of projection params:
                + FUNCTION:<method>
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
            - prefs (PreferenceSet or specification dict):
                 if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
                 dict entries must have a preference keyPath as key, and a PreferenceEntry or setting as their value
                 (see Description under PreferenceSet for details)
            - context (str): must be a reference to a subclass, or an exception will be raised

        NOTES:
        * Receiver is required, since can't instantiate a Projection without a receiving State
        * If sender and/or receiver is a Mechanism, the appropriate State is inferred as follows:
            Mapping projection:
                sender = <Mechanism>.outputState
                receiver = <Mechanism>.inputState
            ControlSignal projection:
                sender = <Mechanism>.outputState
                receiver = <Mechanism>.paramsCurrent[<param>] IF AND ONLY IF there is a single one
                            that is a ParameterState;  otherwise, an exception is raised
        * _instantiate_sender, _instantiate_receiver must be called before _instantiate_function:
            - _validate_params must be called before _instantiate_sender, as it validates kwProjectionSender
            - instantatiate_sender may alter self.variable, so it must be called before _validate_function
            - instantatiate_receiver must be called before _validate_function,
                 as the latter evaluates receiver.value to determine whether to use self.function or FUNCTION
        * If variable is incompatible with sender's output, it is set to match that and revalidated (_instantiate_sender)
        * if FUNCTION is provided but its output is incompatible with receiver value, self.function is tried
        * registers projection with ProjectionRegistry

        :param sender: (State or dict)
        :param receiver: (State or dict)
        :param param_defaults: (dict)
        :param name: (str)
        :param context: (str)
        :return: None
        """

        if not isinstance(context, Projection_Base):
            raise ProjectionError("Direct call to abstract class Projection() is not allowed; "
                                 "use projection() or one of the following subclasses: {0}".
                                 format(", ".join("{!s}".format(key) for (key) in ProjectionRegistry.keys())))

        # Register with ProjectionRegistry or create one
        register_category(entry=self,
                          base_class=Projection_Base,
                          name=name,
                          registry=ProjectionRegistry,
                          context=context)

        # # MODIFIED 9/11/16 NEW:
        # Create projection's _stateRegistry and parameterState entry
        from PsyNeuLink.Components.States.State import State_Base
        self._stateRegistry = {}
        # ParameterState
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        register_category(entry=ParameterState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

# FIX: 6/23/16 NEEDS ATTENTION *******************************************************A
#      NOTE: SENDER IS NOT YET KNOWN FOR DEFAULT controlSignal
#      WHY IS self.sender IMPLEMENTED BY sender IS NOT??

        self.sender = sender
        self.receiver = receiver

# MODIFIED 6/12/16:  VARIABLE & SENDER ASSIGNMENT MESS:
        # ADD _validate_variable, THAT CHECKS FOR SENDER?
        # WHERE DOES DEFAULT SENDER GET INSTANTIATED??
        # VARIABLE ASSIGNMENT SHOULD OCCUR AFTER THAT

# MODIFIED 6/12/16:  ADDED ASSIGNMENT HERE -- BUT SHOULD GET RID OF IT??
        # AS ASSIGNMENT SHOULD BE DONE IN _validate_variable, OR WHEREVER SENDER IS DETERMINED??
# FIX:  NEED TO KNOW HERE IF SENDER IS SPECIFIED AS A MECHANISM OR STATE
        try:
            variable = sender.value
        except:
            try:
                if self.receiver.prefs.verbosePref:
                    warnings.warn("Unable to get value of sender ({0}) for {1};  will assign default ({2})".
                                  format(sender, self.name, self.variableClassDefault))
                variable = None
            except AttributeError:
                raise ProjectionError("{} has no receiver assigned".format(self.name))

# FIX: SHOULDN'T variable_default HERE BE sender.value ??  AT LEAST FOR Mapping?, WHAT ABOUT ControlSignal??
# FIX:  ?LEAVE IT TO _validate_variable, SINCE SENDER MAY NOT YET HAVE BEEN INSTANTIATED
# MODIFIED 6/12/16:  ADDED ASSIGNMENT ABOVE
#                   (TO HANDLE INSTANTIATION OF DEFAULT ControlSignal SENDER -- BUT WHY ISN'T VALUE ESTABLISHED YET?
        # Validate variable, function and params, and assign params to paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        super(Projection_Base, self).__init__(variable_default=variable,
                                              param_defaults=params,
                                              name=self.name,
                                              prefs=prefs,
                                              context=context.__class__.__name__)

        # self.paramNames = self.paramInstanceDefaults.keys()

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Validate kwProjectionSender and/or sender arg (current self.sender), and assign one of them as self.sender

        Check:
        - that kwProjectionSender is a Mechanism or State
        - if it is different from paramClassDefaults[kwProjectionSender], use it
        - if it is the same or is invalid, check if sender arg was provided to __init__ and is valid
        - if sender arg is valid use it (if kwProjectionSender can't be used);
        - otherwise use paramClassDefaults[kwProjectionSender]
        - when done, sender is assigned to self.sender

        Note: check here only for sender's type, NOT content (e.g., length, etc.); that is done in _instantiate_sender

        :param request_set:
        :param target_set:
        :param context:
        :return:
        """

        super(Projection, self)._validate_params(request_set, target_set, context)

        try:
            sender_param = target_set[kwProjectionSender]
        except KeyError:
            # This should never happen, since kwProjectionSender is a required param
            raise ProjectionError("Program error: required param {0} missing in {1}".
                                  format(kwProjectionSender, self.name))

        # kwProjectionSender is either an instance or class of Mechanism or State:
        if (isinstance(sender_param, (Mechanism, State)) or
                (inspect.isclass(sender_param) and
                     (issubclass(sender_param, Mechanism) or issubclass(sender_param, State)))):
            # it is NOT the same as the default, use it
            if sender_param is not self.paramClassDefaults[kwProjectionSender]:
                self.sender = sender_param
            # it IS the same as the default, but sender arg was not provided, so use it (= default):
            elif self.sender is None:
                self.sender = sender_param
                if self.prefs.verbosePref:
                    warnings.warn("Neither {0} nor sender arg was provided for {1} projection to {2}; "
                                  "default ({3}) will be used".format(kwProjectionSender,
                                                                      self.name,
                                                                      self.receiver.owner.name,
                                                                      sender_param.__class__.__name__))
            # it IS the same as the default, so check if sender arg (self.sender) is valid
            elif not (isinstance(self.sender, (Mechanism, State, Process)) or
                          (inspect.isclass(self.sender) and
                               (issubclass(self.sender, Mechanism) or issubclass(self.sender, State)))):
                # sender arg (self.sender) is not valid, so use kwProjectionSender (= default)
                self.sender = sender_param
                if self.prefs.verbosePref:
                    warnings.warn("{0} was not provided for {1} projection to {2}, and sender arg ({3}) is not valid; "
                                  "default ({4}) will be used".format(kwProjectionSender,
                                                                      self.name,
                                                                      self.receiver.owner.name,
                                                                      self.sender,
                                                                      sender_param.__class__.__name__))

# FIX: IF PROJECTION, PUT HACK HERE TO ACCEPT AND FORGO ANY FURTHER PROCESSING??
            # IS the same as the default, and sender arg was provided, so use sender arg
            else:
                pass
        # kwProjectionSender is not valid, and:
        else:
            # sender arg was not provided, use paramClassDefault
            if self.sender is None:
                self.sender = self.paramClassDefaults[kwProjectionSender]
                if self.prefs.verbosePref:
                    warnings.warn("{0} ({1}) is invalid and sender arg ({2}) was not provided;"
                                  " default {3} will be used".
                                  format(kwProjectionSender, sender_param, self.sender,
                                         self.paramClassDefaults[kwProjectionSender]))
            # sender arg is also invalid, so use paramClassDefault
            elif not isinstance(self.sender, (Mechanism, State)):
                self.sender = self.paramClassDefaults[kwProjectionSender]
                if self.prefs.verbosePref:
                    warnings.warn("Both {0} ({1}) and sender arg ({2}) are both invalid; default {3} will be used".
                                  format(kwProjectionSender, sender_param, self.sender,
                                         self.paramClassDefaults[kwProjectionSender]))
            else:
                self.sender = self.paramClassDefaults[kwProjectionSender]
                if self.prefs.verbosePref:
                    warnings.warn("{0} ({1}) is invalid; sender arg ({2}) will be used".
                                  format(kwProjectionSender, sender_param, self.sender))
            if not isinstance(self.paramClassDefaults[kwProjectionSender], (Mechanism, State)):
                raise ProjectionError("Program error: {0} ({1}) and sender arg ({2}) for {3} are both absent or invalid"
                                      " and default (paramClassDefault[{4}]) is also invalid".
                                      format(kwProjectionSender,
                                             # sender_param.__name__,
                                             # self.sender.__name__,
                                             # self.paramClassDefaults[kwProjectionSender].__name__))
                                             sender_param,
                                             self.sender,
                                             self.name,
                                             self.paramClassDefaults[kwProjectionSender]))

    def _instantiate_attributes_before_function(self, context=None):

        self._instantiate_sender(context=context)

        from PsyNeuLink.Components.States.ParameterState import instantiate_parameter_states
        instantiate_parameter_states(owner=self, context=context)

    def _instantiate_sender(self, context=None):
        """Assign self.sender to outputState of sender and insure compatibility with self.variable

        Assume self.sender has been assigned in _validate_params, from either sender arg or kwProjectionSender
        Validate, set self.variable, and assign projection to sender's sendsToProjections attribute

        If self.sender is a Mechanism, re-assign it to <Mechanism>.outputState
        If self.sender is a State class reference, validate that it is a OutputState
        Assign projection to sender's sendsToProjections attribute
        If self.value / self.variable is None, set to sender.value

        Notes:
        * ControlSignal initially overrides this method to check if sender is DefaultControlMechanism;
            if so, it assigns a ControlSignal-specific inputState, outputState and ControlSignalChannel to it
        [TBI: * LearningSignal overrides this method to check if sender is kwDefaultSender;
            if so, it instantiates a default MonitoringMechanism and a projection to it from receiver's outputState]

        :param context: (str)
        :return:
        """

        from PsyNeuLink.Components.States.OutputState import OutputState

        # If sender is a class, instantiate it:
        # - assume it is Mechanism or State (as validated in _validate_params)
        # - implement default sender of the corresponding type
        if inspect.isclass(self.sender):
            if issubclass(self.sender, OutputState):
                # MODIFIED 9/12/16 NEW:
                # self.paramsCurrent['function_params']['matrix']
                # FIX: ASSIGN REFERENCE VALUE HERE IF IT IS A MAPPING PROJECTION??
                # MODIFIED 9/12/16 END
                self.sender = self.paramsCurrent[kwProjectionSender](self.paramsCurrent[kwProjectionSenderValue])
            else:
                raise ProjectionError("Sender ({0}, for {1}) must be a OutputState".
                                      format(self.sender.__class__.__name__, self.name))

        # # If sender is a Mechanism (rather than a State), get relevant outputState and assign it to self.sender
        if isinstance(self.sender, Mechanism):

            # # IMPLEMENT: HANDLE MULTIPLE SENDER -> RECEIVER MAPPINGS, EACH WITH ITS OWN MATRIX:
            # #            - kwMATRIX NEEDS TO BE A 3D np.array, EACH 3D ITEM OF WHICH IS A 2D WEIGHT MATRIX
            # #            - MAKE SURE len(self.sender.value) == len(self.receiver.inputStates.items())
            # # for i in range (len(self.sender.value)):
            # #            - CHECK EACH MATRIX AND ASSIGN??
            # # FOR NOW, ASSUME SENDER HAS ONLY ONE OUTPUT STATE, AND THAT RECEIVER HAS ONLY ONE INPUT STATE
            self.sender = self.sender.outputState

        # At this point, self.sender should be a OutputState
        if not isinstance(self.sender, OutputState):
            raise ProjectionError("Sender for Mapping projection must be a Mechanism or State")

        # Assign projection to sender's sendsToProjections list attribute
        # MODIFIED 8/4/16 OLD:  SHOULD CALL _add_projection_from
        self.sender.sendsToProjections.append(self)

        # Validate projection's variable (self.variable) against sender.outputState.value
        if iscompatible(self.variable, self.sender.value):
            # Is compatible, so assign sender.outputState.value to self.variable
            self.variable = self.sender.value

        else:
            # Not compatible, so:
            # - issue warning
            if self.prefs.verbosePref:
                warnings.warn("The variable ({0}) of {1} projection to {2} is not compatible with output ({3})"
                              " of function {4} for sender ({5}); it has been reassigned".
                      format(self.variable,
                             self.name,
                             self.receiver.owner.name,
                             self.sender.value,
                             self.sender.function.__class__.__name__,
                             self.sender.owner.name))
            # - reassign self.variable to sender.value
            self.assign_defaults(variable=self.sender.value, context=context)

    def _instantiate_attributes_after_function(self, context=None):
        self._instantiate_receiver(context=context)

    def _instantiate_receiver(self, context=None):
        """Call receiver's owner to add projection to its receivesFromProjections list

        Notes:
        * Assume that subclasses implement this method in which they:
          - test whether self.receiver is a Mechanism and, if so, replace with State appropriate for projection
          - calls this method (as super) to assign projection to the Mechanism
        * Constraint that self.value is compatible with receiver.inputState.value
            is evaluated and enforced in _instantiate_function, since that may need to be modified (see below)

        IMPLEMENTATION NOTE: since projection is added using Mechanism.add_projection(projection, state) method,
                             could add state specification as arg here, and pass through to add_projection()
                             to request a particular state

        :param context: (str)
        :return:
        """

        if isinstance(self.receiver, State):
            _add_projection_to(receiver=self.receiver.owner,
                              state=self.receiver,
                              projection_spec=self,
                              context=context)

        # This should be handled by implementation of _instantiate_receiver by projection's subclass
        elif isinstance(self.receiver, Mechanism):
            raise ProjectionError("PROGRAM ERROR: receiver for {0} was specified as a Mechanism ({1});"
                                  "this should have been handled by _instantiate_receiver for {2}".
                                  format(self.name, self.receiver.name, self.__class__.__name__))

        else:
            raise ProjectionError("Unrecognized receiver specification ({0}) for {1}".format(self.receiver, self.name))

    def add_to(self, receiver, state, context=None):
        _add_projection_to(receiver=receiver, state=state, projection_spec=self, context=context)


# from PsyNeuLink.Components.Projections.ControlSignal import is_control_signal
# from PsyNeuLink.Components.Projections.LearningSignal import is_learning_signal

def _is_projection_spec(spec):
    """Evaluate whether spec is a valid Projection specification

    Return true if spec is any of the following:
    + Projection class (or keyword string constant for one):
    + Projection object:
    + specification dict containing:
        + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection

    Otherwise, return False

    Returns: (bool)
    """
    if inspect.isclass(spec) and issubclass(spec, Projection):
        return True
    if isinstance(spec, Projection):
        return True
    if isinstance(spec, dict) and PROJECTION_TYPE in spec:
        return True
    if isinstance(spec, str) and spec in PROJECTION_SPEC_KEYWORDS:
        return True
    # MODIFIED 9/6/16 NEW:
    if isinstance(spec, tuple) and len(spec) == 2:
        # Call recursively on first item, which should be a standard projection spec
        if _is_projection_spec(spec[0]):
            # IMPLEMENTATION NOTE: keywords must be used to refer to subclass, to avoid import loop
            if _is_projection_subclass(spec[1], CONTROL_SIGNAL):
                return True
            if _is_projection_subclass(spec[1], LEARNING_SIGNAL):
                return True
    return False

def _is_projection_subclass(spec, keyword):
    """Evaluate whether spec is a valid specification of type

    keyword must specify a class registered in ProjectionRegistry

    Return true if spec ==
    + keyword
    + subclass of Projection associated with keyword (from ProjectionRegistry)
    + instance of the subclass
    + specification dict for instance of the subclass:
        keyword is a keyword for an entry in the spec dict
        keyword[spec] is a legal specification for the subclass

    Otherwise, return False
    """
    if spec is keyword:
        return True
    # Get projection subclass specified by keyword
    try:
        type = ProjectionRegistry[keyword]
    except KeyError:
        pass
    else:
        # Check if spec is either the name of the subclass or an instance of it
        if inspect.isclass(spec) and issubclass(spec, type):
            return True
        if isinstance(spec, type):
            return True
    # spec is a specification dict for an instance of the projection subclass
    if isinstance(spec, dict) and keyword in spec:
        # Recursive call to determine that the entry of specification dict is a legal spec for the projection subclass
        if _is_projection_subclass(spec[keyword], keyword):
            return True
    return False


def _add_projection_to(receiver, state, projection_spec, context=None):
    """Assign an "incoming" Projection to a receiver InputState or ParameterState of a Function object

    receiver must be an appropriate Function object (currently, a Mechanism or a Projection)
    state must be a specification of an InputState or ParameterState
    Specification of InputState can be any of the following:
            - kwInputState - assigns projection_spec to (primary) inputState
            - InputState object
            - index for Mechanism.inputStates OrderedDict
            - name of inputState (i.e., key for Mechanism.inputStates OrderedDict))
            - the keyword kwAddInputState or the name for an inputState to be added
    Specification of ParameterState must be a ParameterState object
    projection_spec can be any valid specification of a projection_spec (see State.instantiate_projections_to_state)
    IMPLEMENTATION NOTE:  ADD FULL SET OF ParameterState SPECIFICATIONS
                          CURRENTLY, ASSUMES projection_spec IS AN ALREADY INSTANTIATED PROJECTION

    Args:
        receiver (Mechanism or Projection):
        projection_spec: (Projection, dict, or str)
        state (State subclass):
        context:

    """
    from PsyNeuLink.Components.States.State import instantiate_state
    from PsyNeuLink.Components.States.State import State_Base
    from PsyNeuLink.Components.States.InputState import InputState
    from PsyNeuLink.Components.States.ParameterState import ParameterState

    if not isinstance(state, (int, str, InputState, ParameterState)):
        raise ProjectionError("State specification(s) for {0} (as receivers of {1}) contain(s) one or more items"
                             " that is not a name, reference to an inputState or parameterState object, "
                             " or an index (for inputStates)".
                             format(receiver.name, projection_spec.name))

    # state is State object, so use that
    if isinstance(state, State_Base):
        state.instantiate_projections_to_state(projections=projection_spec, context=context)
        return

    # Generic kwInputState is specified, so use (primary) inputState
    elif state is kwInputState:
        receiver.inputState.instantiate_projections_to_state(projections=projection_spec, context=context)
        return

    # input_state is index into inputStates OrderedDict, so get corresponding key and assign to input_state
    elif isinstance(state, int):
        try:
            key = list(receiver.inputStates.keys)[state]
        except IndexError:
            raise ProjectionError("Attempt to assign projection_spec ({0}) to inputState {1} of {2} "
                                 "but it has only {3} inputStates".
                                 format(projection_spec.name, state, receiver.name, len(receiver.inputStates)))
        else:
            input_state = key

    # input_state is string (possibly key retrieved above)
    #    so try as key in inputStates OrderedDict (i.e., as name of an inputState)
    if isinstance(state, str):
        try:
            receiver.inputState[state].instantiate_projections_to_state(projections=projection_spec, context=context)
        except KeyError:
            pass
        else:
            if receiver.prefs.verbosePref:
                warnings.warn("Projection_spec {0} added to {1} of {2}".
                              format(projection_spec.name, state, receiver.name))
            # return

    # input_state is either the name for a new inputState or kwAddNewInputState
    if not state is kwAddInputState:
        if receiver.prefs.verbosePref:
            reassign = input("\nAdd new inputState named {0} to {1} (as receiver for {2})? (y/n):".
                             format(input_state, receiver.name, projection_spec.name))
            while reassign != 'y' and reassign != 'n':
                reassign = input("\nAdd {0} to {1}? (y/n):".format(input_state, receiver.name))
            if reassign == 'n':
                raise ProjectionError("Unable to assign projection {0} to receiver {1}".
                                      format(projection_spec.name, receiver.name))

    input_state = instantiate_state(owner=receiver,
                                    state_type=InputState,
                                    state_name=input_state,
                                    state_spec=projection_spec.value,
                                    constraint_value=projection_spec.value,
                                    constraint_value_name='Projection_spec value for new inputState',
                                    context=context)
        #  Update inputState and inputStates
    try:
        receiver.inputStates[input_state.name] = input_state
    # No inputState(s) yet, so create them
    except AttributeError:
        receiver.inputStates = OrderedDict({input_state.name:input_state})
        receiver.inputState = list(receiver.inputStates)[0]
    input_state.instantiate_projections_to_state(projections=projection_spec, context=context)

def _add_projection_from(sender, state, projection_spec, receiver, context=None):
    """Assign an "outgoing" Projection from an OutputState of a sender Mechanism

    projection_spec can be any valid specification of a projection_spec (see State.instantiate_projections_to_state)
    state must be a specification of an outputState
    Specification of OutputState can be any of the following:
            - kwOutputState - assigns projection_spec to (primary) outputState
            - OutputState object
            - index for Mechanism.outputStates OrderedDict
            - name of outputState (i.e., key for Mechanism.outputStates OrderedDict))
            - the keyword kwAddOutputState or the name for an outputState to be added

    Args:
        sender (Mechanism):
        projection_spec: (Projection, dict, or str)
        state (OutputState, str, or value):
        context:
    """

    from PsyNeuLink.Components.States.State import instantiate_state
    from PsyNeuLink.Components.States.State import State_Base
    from PsyNeuLink.Components.States.OutputState import OutputState

    if not isinstance(state, (int, str, OutputState)):
        raise ProjectionError("State specification for {0} (as sender of {1}) must be the name, reference to "
                              "or index of an outputState of {0} )".format(sender.name, projection_spec))

    # state is State object, so use that
    if isinstance(state, State_Base):
        state.instantiate_projection_from_state(projection_spec=projection_spec, receiver=receiver, context=context)
        return

    # Generic kwOutputState is specified, so use (primary) outputState
    elif state is kwOutputState:
        sender.outputState.instantiate_projections_to_state(projections=projection_spec, context=context)
        return

    # input_state is index into outputStates OrderedDict, so get corresponding key and assign to output_state
    elif isinstance(state, int):
        try:
            key = list(sender.outputStates.keys)[state]
        except IndexError:
            raise ProjectionError("Attempt to assign projection_spec ({0}) to outputState {1} of {2} "
                                 "but it has only {3} outputStates".
                                 format(projection_spec.name, state, sender.name, len(sender.outputStates)))
        else:
            output_state = key

    # output_state is string (possibly key retrieved above)
    #    so try as key in outputStates OrderedDict (i.e., as name of an outputState)
    if isinstance(state, str):
        try:
            sender.outputState[state].instantiate_projections_to_state(projections=projection_spec, context=context)
        except KeyError:
            pass
        else:
            if sender.prefs.verbosePref:
                warnings.warn("Projection_spec {0} added to {1} of {2}".
                              format(projection_spec.name, state, sender.name))
            # return

    # input_state is either the name for a new inputState or kwAddNewInputState
    if not state is kwAddOutputState:
        if sender.prefs.verbosePref:
            reassign = input("\nAdd new outputState named {0} to {1} (as sender for {2})? (y/n):".
                             format(output_state, sender.name, projection_spec.name))
            while reassign != 'y' and reassign != 'n':
                reassign = input("\nAdd {0} to {1}? (y/n):".format(output_state, sender.name))
            if reassign == 'n':
                raise ProjectionError("Unable to assign projection {0} to sender {1}".
                                      format(projection_spec.name, sender.name))

    output_state = instantiate_state(owner=sender,
                                     state_type=OutputState,
                                     state_name=output_state,
                                     state_spec=projection_spec.value,
                                     constraint_value=projection_spec.value,
                                     constraint_value_name='Projection_spec value for new inputState',
                                     context=context)
    #  Update inputState and inputStates
    try:
        sender.outputStates[output_state.name] = output_state
    # No inputState(s) yet, so create them
    except AttributeError:
        sender.outputStates = OrderedDict({output_state.name:output_state})
        sender.outputState = list(sender.outputStates)[0]
    output_state.instantiate_projections_to_state(projections=projection_spec, context=context)