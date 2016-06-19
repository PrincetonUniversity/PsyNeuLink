#
# *********************************************  MechanismState ********************************************************
#

from Functions.ShellClasses import *
from Functions.Utility import *
from Globals.Registry import  register_category
import numpy as np

MechanismStateRegistry = {}

# class MechanismStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE
#

class MechanismStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# MechanismState factory method:
# def mechanismState(name=NotImplemented, params=NotImplemented, context=NotImplemented):
#         """Instantiates default or specified subclass of MechanismState
#
#        If called w/o arguments or 1st argument=NotImplemented, instantiates default subclass (MechanismParameterState)
#         If called with a name string:
#             - if registered in MechanismStateRegistry class dictionary as name of a subclass, instantiates that class
#             - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
#         If a params dictionary is included, it is passed to the subclass
#
#         :param name:
#         :param param_defaults:
#         :return:
#         """
#
#         # Call to instantiate a particular subclass, so look up in MechanismRegistry
#         if name in MechanismStateRegistry:
#             return MechanismStateRegistry[name].mechanismSubclass(params)
#         # Name is not in MechanismRegistry or is not provided, so instantiate default subclass
#         else:
#             # from Functions.Defaults import DefaultMechanismState
#             return DefaultMechanismState(name, params)

# DOCUMENT:  INSTANTATION CREATES AN ATTIRBUTE ON THE OWNER MECHANISM WITH THE STATE'S NAME + kwValueSuffix
#            THAT IS UPDATED BY THE STATE'S value setter METHOD (USED BY LOGGING OF MECHANISM ENTRIES)
class MechanismState_Base(MechanismState):
    """Implement abstract class for MechanismState category of class types that compute and represent mechanism states

    Description:
        Represents and updates the state of an input, output or parameter of a mechanism
            - receives inputs from projections (self.receivesFromProjections, kwMechanismStateProjections)
            - combines inputs from all projections (mapping and/or control) and uses this as variable of
                execute method to update the value attribute
        Value attribute:
             - is updated by the execute method
             - can be used as sender (input) to one or more projections
             - can be accessed by KVO
        Constraints:
            - value must be compatible with variable of execute method
            - value must be compatible with output of execute method (self.executeMethodOutputDefault)
            - execute method variable must be compatible with its output (self.executeMethodOutputDefault)
            - value must be compatible with receiver.value for all projections it receives

    Subclasses:
        Must implement:
            functionType
            ParamClassDefaults with:
                + kwExecuteMethod (or <subclass>.execute
                + kwExecuteMethodParams (optional)
                + kwProjectionType - specifies type of projection to use for instantiation of default subclass
        Standard subclasses and constraints:
            MechanismInputState - used as input state for Mechanism;  additional constraint:
                - value must be compatible with variable of owner's execute method
            MechanismOutputState - used as output state for Mechanism;  additional constraint:
                - value must be compatible with the output of the ownerMechanism's execute method
            MechanismsParameterState - used as state for Mechanism parameter;  additional constraint:
                - output of execute method must be compatible with the parameter's value

    Instantiation:
        MechanismStates should NEVER be instantiated by a direct call to the class
           (since there is no obvious default), but rather by calls to the subclass
        Subclasses can be instantiated in one of two ways:
            - call to __init__ with args to subclass with args for ownerMechanism, value and params:
            - as part of the instantiation of a Mechanism (see Mechanism.intantiate_mechanism_state for details)

    Initialization arguments:
        - value (value) - establishes type of value attribute and initializes it (default: [0])
        - ownerMechanism(Mechanism) - assigns state to mechanism (default: NotImplemented)
        - params (dict):  (if absent, default state is implemented)
            + kwExecuteMethod (method)         |  Implemented in subclasses; used in update_state()
            + kwExecuteMethodParams (dict) |
            + kwMechanismStateProjections:<projection specification or list of ones>
                if absent, no projections will be created
                projection specification can be: (see Projection for details)
                    + Projection object
                    + Projection class
                    + specification dict
                    + a list containing any or all of the above
                    if dict, must contain entries specifying a projection:
                        + kwProjectionType:<Projection class>: must be a subclass of Projection
                        + kwProjectionParams:<dict>? - must be dict of params for kwProjectionType
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
        - context (str): must be a reference to a subclass, or an exception will be raised

    MechanismStateRegistry:
        All MechanismStates are registered in MechanismStateRegistry, which maintains a dict for each subclass,
          a count for all instances of that type, and a dictionary of those instances

    Naming:
        MechanismStates can be named explicitly (using the name='<name>' argument).  If the argument is omitted,
        it will be assigned the subclass name with a hyphenated, indexed suffix ('subclass.name-n')

    Execution:


    Class attributes:
        + functionCategory = kwMechanismStateFunctionCategory
        + className = kwMechanismState
        + suffix
        + registry (dict): MechanismStateRegistry
        + classPreference (PreferenceSet): MechanismStatePreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + variableClassDefault (value): [0]
        + requiredParamClassDefaultTypes = {kwExecuteMethodParams : [dict],    # Subclass execute method params
                                           kwProjectionType: [str, Projection]})   # Default projection type
        + paramClassDefaults (dict): {kwMechanismStateProjections: []}             # Projections to MechanismStates
        + paramNames (dict)
        + ownerMechanism (Mechansim)
        + kwExecuteMethod (Function class or object, or method)

    Class methods:
        • set_value(value) -
            validates and assigns value, and updates observers
            returns None
        • update_state(context) -
            updates self.value by combining all projections and using them to compute new value
            return None

    Instance attributes:
        + ownerMechanism (Mechanism): object to which MechanismState belongs
        + value (value): current value of the MechanismState (updated by update_state method)
        + baseValue (value): value with which it was initialized (e.g., from params in call to __init__)
        + projections (list): list of projections for which MechanismState is a receiver [TBI:  change to receivesFrom]
        + sendsTo (list): list of projections for which MechanismState is a sender
        + params (dict)
        + executeMethodOutputDefault (value) - sample output of MechanismState's execute method
        + executeMethodOutputType (type) - type of output of MechanismStates's execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying MechanismStatePreferenceSet

    Instance methods:
        none
    """

    #region CLASS ATTRIBUTES

    kpState = "State"
    functionCategory = kwMechanismStateFunctionCategory
    className = kwMechanismState
    suffix = " " + className

    registry = MechanismStateRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    variableClassDefault = [0]

    requiredParamClassDefaultTypes = Function.requiredParamClassDefaultTypes.copy()
    requiredParamClassDefaultTypes.update({kwExecuteMethodParams : [dict],
                                           kwProjectionType: [str, Projection]})   # Default projection type
    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({kwMechanismStateProjections: []})
    paramNames = paramClassDefaults.keys()

    #endregion

    def __init__(self,
                 owner_mechanism,
                 value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented,
                 **kargs):
        """Initialize subclass that computes and represents the value of a particular state of a mechanism

        This is used by subclasses to implement the input, output, and parameter states of a Mechanism

        Arguments:
            - ownerMechanism (Mechanism):
                 mechanism with which state is associated (default: NotImplemented)
                 this argument is required, as can't instantiate a MechanismState without an owning Mechanism
            - value (value): value of the state:
                must be list or tuple of numbers, or a number (in which case it will be converted to a single-item list)
                must match input and output of state's update function, and any sending or receiving projections
            - params (dict):
                + if absent, implements default MechanismState determined by kwProjectionType param
                + if dict, can have the following entries:
                    + kwMechanismStateProjections:<Projection object, Projection class, dict, or list of either or both>
                        if absent, no projections will be created
                        if dict, must contain entries specifying a projection:
                            + kwProjectionType:<Projection class> - must be a subclass of Projection
                            + kwProjectionParams:<dict> - must be dict of params for kwProjectionType
            - name (str): string with name of state (default: name of ownerMechanism + suffix + instanceIndex)
            - prefs (dict): dictionary containing system preferences (default: Prefs.DEFAULTS)
            - context (str)
            - **kargs (dict): dictionary of arguments using the following keywords for each of the above kargs:
                + kwMechanismStateValue = value
                + kwMechanismStateParams = params
                + kwMechanismStateName = name
                + kwMechanismStatePrefs = prefs
                + kwMechanismStateContext = context
                NOTES:
                    * these are used for dictionary specification of a MechanismState in param declarations
                    * they take precedence over arguments specified directly in the call to __init__()

        :param ownerMechanism: (Mechanism)
        :param value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (dict)
        :return:
        """
        if kargs:
            try:
                value = kargs[kwMechanismStateValue]
            except (KeyError, NameError):
                pass
            try:
                params = kargs[kwMechanismStateParams]
            except (KeyError, NameError):
                pass
            try:
                name = kargs[kwMechanismStateName]
            except (KeyError, NameError):
                pass
            try:
                prefs = kargs[kwMechanismStatePrefs]
            except (KeyError, NameError):
                pass
            try:
                context = kargs[kwMechanismStateContext]
            except (KeyError, NameError):
                pass

        if not isinstance(context, MechanismState_Base):
            raise MechanismStateError("Direct call to abstract class MechanismState() is not allowed; "
                                      "use mechanismState() or one of the following subclasses: {0}".
                                      format(", ".join("{!s}".format(key) for (key) in MechanismStateRegistry.keys())))

        # FROM MECHANISM:
        # Note: pass name of mechanism (to override assignment of functionName in super.__init__)

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        # Not needed:  handled by subclass
        # else:
        #     self.name = name

        self.functionName = self.functionType

        register_category(self, MechanismState_Base, MechanismStateRegistry, context=context)

        #region VALIDATE ownerMechanism
        if isinstance(owner_mechanism, Mechanism):
            self.ownerMechanism = owner_mechanism
        else:
            raise MechanismStateError("ownerMechanism argument ({0}) for {1} must be a mechanism".
                                      format(owner_mechanism, self.name))
        #endregion

        self.receivesFromProjections = []
        self.sendsToProjections = []

        # VALIDATE VARIABLE, PARAMS, AND INSTANTIATE EXECUTE METHOD
        super(MechanismState_Base, self).__init__(variable_default=value,
                                                  param_defaults=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=context.__class__.__name__)
        # self.paramNames = self.paramInstanceDefaults.keys()

        self.instantiate_projections(context=context)

# FIX LOG: EITHER GET RID OF THIS NOW THAT @property HAS BEEN IMPLEMENTED, OR AT LEAST INTEGRATE WITH IT
        # add state to KVO observer dict
        self.observers = {self.kpState: []}

# FIX: WHY IS THIS COMMENTED OUT?  IS IT HANDLED BY SUBCLASSES??
    # def register_category(self):
    #     register_mechanism_state_subclass(self)

    def validate_variable(self, variable, context=NotImplemented):
        """Validate variable and assign validated values to self.variable

        Sets self.baseValue = self.value = self.variable = variable
        Insures that it is a number of list or tuple of numbers

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """

        super(MechanismState,self).validate_variable(variable, context)

        if context is NotImplemented:
            context = kwAssign + ' Base Value'
        else:
            context = context + kwAssign + ' Base Value'

        self.baseValue = self.variable


    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """validate projection specification(s)

        Call super (Function.validate_params()
        Validate following params:
            + kwMechanismStateProjections:  <entry or list of entries>; each entry must be one of the following:
                + Projection object
                + Projection class
                + specification dict, with the following entries:
                    + kwProjectionType:<Projection class> - must be a subclass of Projection
                    + kwProjectionParams:<dict> - must be dict of params for kwProjectionType
            # IMPLEMENTATION NOTE: TBI - When learning projection is implemented
            # + kwExecuteMethodParams:  <dict>, every entry of which must be one of the following:
            #     MechanismParameterState, projection, ParamValueProjection tuple or value

        :param request_set:
        :param target_set:
        :param context:
        :return:
        """

        # Check params[kwMechanismStateProjections] before calling validate_param:
        # - if projection specification is an object or class reference, needs to be wrapped in a list
        try:
            projections = target_set[kwMechanismStateProjections]
        except KeyError:
            # If no projections, ignore (none will be created)
            projections = None
        else:
            # If specification is not a list, wrap it in one:
            # - to be consistent with paramClassDefaults
            # - for consistency of treatment below
            if not isinstance(projections, list):
                projections = [projections]

        super(MechanismState, self).validate_params(request_set, target_set, context=context)

        if projections:
            # Validate projection specs in list
            from Functions.Projections import Projection
            for projection in projections:
                try:
                    issubclass(projection, Projection)
                except TypeError:
                    if (isinstance(projection, Projection) or iscompatible(projection. dict)):
                        continue
                    else:
                        if self.prefs.verbosePref:
                            print("{0} in {1} is not a projection, projection type, or specification dict; "
                                  "{2} will be used to create default {3} for {4}".
                                format(projection,
                                       self.__class__.__name__,
                                       target_set[kwProjectionType],
                                       self.ownerMechanism.name))


    def instantiate_execute_method(self, context=NotImplemented):
        """Insure that output of execute method (self.value) is compatible with its input (self.variable)

        This constraint reflects the function of MechanismState execute methods:
            they simply update the value of the MechanismState;
            accordingly, their variable, executeMethodOutputDefault and value must all be compatible

        :param context:
        :return:
        """

        super(MechanismState_Base, self).instantiate_execute_method(context=context)

        # Insure that output of execute method (self.value) is compatible with its input (self.variable)
        if not iscompatible(self.variable, self.value):
            raise MechanismStateError("Output ({0}: {1}) of execute function ({2}) for {3} {4} of {5}"
                                      " must be the same format as its input ({6}: {7})".
                                      format(type(self.value).__name__,
                                             self.value,
                                             self.execute.__self__.functionName,
                                             self.name,
                                             self.__class__.__name__,
                                             self.ownerMechanism.name,
                                             self.variable.__class__.__name__,
                                             self.variable))

    def instantiate_projections(self, context=NotImplemented):
        """Instantiate projections for a mechanismState and assign them to self.receivesFromProjections

        Check that projection spec is one or a list of any of the following:
        + Projection class (or keyword string constant for one):
            implements default projection for projection class
        + Projection object:
            checks that receiver is self
            checks that projection execute method output is compatible with self.value
        + specification dict:
            checks that projection execute method output is compatible with self.value
            implements projection
            dict must contain:
                + kwProjectionType:<Projection class> - must be a subclass of Projection
                + kwProjectionParams:<dict> - must be dict of params for kwProjectionType
        If any of the conditions above fail:
            a default projection is instantiated using self.paramsCurrent[kwProjectionType]
        Each projection in the list is added to self.receivesFromProjections
        If kwMMechanismStateProjections is absent or empty, no projections are created

        :param context: (str)
        :return mechanismState: (MechanismState)
        """

        from Functions.Projections.Projection import Projection_Base

        try:
            projection_list = self.paramsCurrent[kwMechanismStateProjections]

        except KeyError:
            # No projections specified, so none will be created
            # IMPLEMENTATION NOTE:  This is where a default projection would be implemented
            #                       if params = NotImplemented or there is no param[kwMechanismStateProjections]
            pass

        else:
            # If specification is not a list, wrap it in one for consistency of treatment below
            # (since specification can be a list, so easier to treat any as a list)
            from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
            if not isinstance(projection_list, list):
                projection_list = [projection_list]

            object_name_string = "{0}".format(self.name)
            item_prefix_string = ""
            item_suffix_string = " for {0}".format(object_name_string)
            default_string = ""
            kwDefault = "default "

            default_projection_type = self.paramsCurrent[kwProjectionType]

            # Instantiate each projection specification in the projection_list, and
            # - insure it is in self.receivesFromProjections
            # - insure the output of its execute method is compatible with self.value
            for projection_spec in projection_list:

                # If there is more than one projection specified, construct messages for use in case of failure
                if len(projection_list) > 1:
                    item_prefix_string = "Item {0} of projection list for {1}: ".\
                        format(projection_list.index(projection_spec)+1, object_name_string)
                    item_suffix_string = ""

# FIX: FROM HERE TO BOTTOM OF METHOD SHOULD ALL BE HANDLED IN __init__() FOR PROJECTION
                projection_object = None # flags whether projection object has been instantiated; doesn't store object
                projection_type = None   # stores type of projection to instantiate
                projection_params = {}

                # INSTANTIATE PROJECTION
                # If projection_spec is a Projection object:
                # - call check_projection_receiver() to check that receiver is self; if not, it:
                #     returns object with receiver reassigned to self if chosen by user
                #     else, returns new (default) kwProjectionType object with self as receiver
                # Note: projection will now be in self.receivesFromProjections list
                if isinstance(projection_spec, Projection_Base):
                    projection_object, default_class_name = \
                                                    self.check_projection_receiver(projection_spec=projection_spec,
                                                                                   messages=[item_prefix_string,
                                                                                             item_suffix_string,
                                                                                             object_name_string],
                                                                                   context=self)
                    if default_class_name:
                        # Assign name based on MechanismState's name:
                        # projection_object.name = projection_object.name.replace(default_class_name, self.name)
                        projection_object.name = self.name + '_' + projection_object.name
                        # Used for error message
                        default_string = kwDefault
# FIX:  REPLACE DEFAULT NAME (RETURNED AS DEFAULT) PROJECTION NAME WITH MechanismState'S NAME, LEAVING INDEXED SUFFIX INTACT

                # If projection_spec is a dict:
                # - get projection_type
                # - get projection_params
                # Note: this gets projection_type but does NOT not instantiate projection; so,
                #       projection is NOT yet in self.receivesFromProjections list
                elif isinstance(projection_spec, dict):
                    # Get projection type from specification dict
                    try:
                        projection_type = projection_spec[kwProjectionType]
                    except KeyError:
                        projection_type = default_projection_type
                        default_string = kwDefault
                        if self.prefs.verbosePref:
                            print("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                                  format(item_prefix_string,
                                         kwProjectionType,
                                         kwMechanismStateProjections,
                                         item_suffix_string,
                                         default_projection_type.__class__.__name__))
                    else:
                        # IMPLEMENTATION NOTE:  can add more informative reporting here about reason for failure
                        projection_type, error_str = self.parse_projection_ref(projection_spec=projection_type,
                                                                               context=self)
                        if error_str:
                            print("{0}{1} {2}; default {4} will be assigned".
                                  format(item_prefix_string,
                                         kwProjectionType,
                                         error_str,
                                         kwMechanismStateProjections,
                                         item_suffix_string,
                                         default_projection_type.__class__.__name__))

                    # Get projection params from specification dict
                    try:
                        projection_params = projection_spec[kwProjectionParams]
                    except KeyError:
                        if self.prefs.verbosePref:
                            print("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                                  format(item_prefix_string,
                                         kwProjectionParams,
                                         kwMechanismStateProjections, object_name_string,
                                         item_suffix_string,
                                         default_projection_type.__class__.__name__))

                # Check if projection_spec is class ref or keyword string constant for one
                # Note: this gets projection_type but does NOT not instantiate projection; so,
                #       projection is NOT yet in self.receivesFromProjections list
                else:
                    projection_type, err_str = self.parse_projection_ref(projection_spec=projection_spec,context=self)
                    if err_str:
                        print("{0}{1} {2}; default {4} will be assigned".
                              format(item_prefix_string,
                                     kwProjectionType,
                                     err_str,
                                     kwMechanismStateProjections,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))

                # If neither projection_object nor projection_type have been assigned, assign default type
                # Note: this gets projection_type but does NOT not instantiate projection; so,
                #       projection is NOT yet in self.receivesFromProjections list
                if not projection_object and not projection_type:
                        projection_type = default_projection_type
                        default_string = kwDefault
                        if self.prefs.verbosePref:
                            print("{0}{1} is not a Projection object or specification for one{2}; "
                                  "default {3} will be assigned".
                                  format(item_prefix_string,
                                         projection_spec.name,
                                         item_suffix_string,
                                         default_projection_type.__class__.__name__))

                # If projection_object has not been assigned, instantiate projection_type
                # Note: this automatically assigns projection to self.receivesFromProjections and
                #       to it's sender's sendsToProjections list:
                #           when a projection is instantiated, it assigns itself to:
                #               its receiver's .receivesFromProjections attribute (in Projection.instantiate_receiver)
                #               its sender's .sendsToProjections list attribute (in Projection.instantiate_sender)
                if not projection_object:
                    projection_spec = projection_type(receiver=self,
                                                      name=self.name+'_'+projection_type.className,
                                                      params=projection_params,
                                                             context=context)

                # Check that output of projection's execute method (projection_spec.value is compatible with
                #    variable of MechanismState to which it projects;  if it is not, raise exception:
                # The buck stops here; otherwise there would be an unmanageable regress of reassigning
                #    projections, requiring reassignment or modification of sender mechanismOutputState, etc.
                if not iscompatible(self.variable, projection_spec.value):
                    raise MechanismStateError("{0}Output ({1}) of execute method for {2}{3} "
                                              "is not compatible with value ({4}){5}".
                          format(item_prefix_string,
                                 projection_spec.value,
                                 default_string,
                                 projection_spec.name,
                                 self.value,
                                 item_suffix_string))
                # IMPLEMENTATION NOTE: The following is not needed, since the current MechanismState is the
                #                          the projection's receiver and, when it is instantiated,
                #                          it assigns itself to its receiver.receivesFromProjections
                # else:
                #     self.receivesFromProjections.append(projection_spec)

    def check_projection_receiver(self, projection_spec, messages=NotImplemented, context=NotImplemented):
        """Check whether Projection object belongs to MechanismState and if not return default Projection object

        Arguments:
        - projection_spec (Projection object)
        - message (list): list of three strings - prefix and suffix for error/warning message, and MechanismState name
        - context (object): ref to MechanismState object; used to identify kwProjectionType and name

        Returns: tuple (Projection object, str); second value is name of default projection, else None

        :param self:
        :param projection_spec: (Projection object)
        :param messages: (list)
        :param context: (MechanismState object)
        :return: (tuple) Projection object, str) - second value is false if default was returned
        """

        prefix = 0
        suffix = 1
        name = 2
        if messages is NotImplemented:
            messages = ["","","",context.__class__.__name__]
        message = "{0}{1} is a projection of the correct type, but its receiver is not assigned to {2}." \
                  " \nReassign (r) or use default (d)?:".\
            format(messages[prefix], projection_spec.name, messages[suffix])

        if projection_spec.receiver is not self:
            reassign = input()
            while reassign != 'r' and reassign != 'd':
                reassign = input("Reassign {0} to {1} or use default (r/d)?:".
                                 format(projection_spec.name, messages[name]))
            # User chose to reassign, so return projection object with MechanismState as its receiver
            if reassign == 'r':
                projection_spec.receiver = self
                self.receivesFromProjections.append(projection_spec)
                if self.prefs.verbosePref:
                    print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
                return (projection_spec, None)
            # User chose to assign default, so return default projection object
            elif reassign == 'd':
                print("Default {0} will be used for {1}".
                      format(projection_spec.name, messages[name]))
                return (self.paramsCurrent[kwProjectionType](receiver=self),
                        self.paramsCurrent[kwProjectionType].className)
                #     print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
            else:
                raise MechanismStateError("Program error:  reassign should be r or d")

    def parse_projection_ref(self,
                             projection_spec,
                             # messages=NotImplemented,
                             context=NotImplemented):
        """Take projection ref and return ref to corresponding type or, if invalid, to  default for context

        Arguments:
        - projection_spec (Projection subclass or str):  str must be a keyword constant for a Projection subclass
        - context (str):

        Returns tuple: (Projection subclass or None, error string)

        :param projection_spec: (Projection subclass or str)
        :param messages: (list)
        :param context: (MechanismState object)
        :return: (Projection subclass, string)
        """
        try:
            # Try projection spec as class ref
            is_projection_class = issubclass(projection_spec, Projection)
        except TypeError:
            # Try projection spec as keyword string constant
            if isinstance(projection_spec, str):
                try:
                    from Functions.Projections.Projection import ProjectionRegistry
                    projection_spec = ProjectionRegistry[projection_spec].subclass
                except KeyError:
                    # projection_spec was not a recognized key
                    return (None, "not found in ProjectionRegistry")
                # projection_spec was legitimate keyword
                else:
                    return (projection_spec, None)
            # projection_spec was neither a class reference nor a keyword
            else:
                return (None, "neither a class reference nor a keyword")
        else:
            # projection_spec was a legitimate class
            if is_projection_class:
                return (projection_spec, None)
            # projection_spec was class but not Projection
            else:
                return (None, "not a Projection subclass")

# # FIX: NO LONGER USED;  SUPERCEDED BY value setter METHOD ABOVE.  INCOROPRATE VALIDATION THERE??
#     def add_observer_for_keypath(self, object, keypath):
#         self.observers[keypath].append(object)
#
# # IMPLEMENTATION NOTE:  USE DECORATOR TO MAKE SURE THIS IS CALLED WHENEVER state.value IS ASSIGNED
#     def set_value(self, new_value):
#         """Validate value, assign it, and update any observers
#
#         Uses valueClassDefault as the template for validating new_value
#         :param new_value:
#         :return:
#         """
#
#         # Validate new_value
#         if self.prefs.paramValidationPref:
#             if not isinstance(new_value, self.variableInstanceDefault):
#                 raise MechanismStateError("Value {0} of {1} must be of type {2}".
#                                      format(new_value, self.name, self.variableInstanceDefault))
#             # Check that each element is a number
#             for element in new_value:
#                 if not isinstance(element, numbers.Number):
#                     raise MechanismStateError("Item {0} ({1}) in value of {2} is not a number".
#                                          format(new_value.index(element), element, self.name))
#
#         old_value = self.value
#
#         # Set value
#         self.value = new_value
#
#         # Update observers
#         if self.observers:
#         # if len(self.observers[self.kpState]):
#             for observer in self.observers[self.kpState]:
#                 observer.observe_value_at_keypath(self.kpState, old_value, new_value)
#
    def update(self, params=NotImplemented, time_scale=TimeScale.TRIAL, context=NotImplemented):
        """Execute function for each projection, combine them, and assign result to value

        Updates self.value by calling self.receivesFromProjections and Arithmetic function with kwArithmeticOperation

    Arguments:
    - context (str)

    :param context: (str)
    :return: None
    """

        #region FLAG FORMAT OF INPUT
        if isinstance(self.value, numbers.Number):
            # Treat as single real value
            value_is_number = True
        else:
            # Treat as vector (list or np.array)
            value_is_number = False
        #endregion

        #region AGGREGATE INPUT FROM PROJECTIONS

        #region Initialize aggregation
        from Functions.Utility import kwArithmeticInitializer
        combined_values = kwArithmeticInitializer
        #endregion

        #region Get type-specific params from kwProjectionParams
        mapping_params = merge_param_dicts(params, kwMappingParams, kwProjectionParams)
        control_signal_params = merge_param_dicts(params, kwControlSignalParams, kwProjectionParams)
        #endregion

        #region Get params for each projection, pass them to it, and get its value
        projection_value_list = []
        for projection in self.receivesFromProjections:

            # Merge with relevant projection type-specific params
            from Functions.Projections.Mapping import Mapping
            from Functions.Projections.ControlSignal import ControlSignal
            if isinstance(projection, Mapping):
                projection_params = merge_param_dicts(params, projection.name, mapping_params, )
            elif isinstance(projection, ControlSignal):
                projection_params = merge_param_dicts(params, projection.name, control_signal_params)
            if not projection_params:
                projection_params = NotImplemented

            # Update projection and get value
            projection_value = projection.update(projection_params, context=context)

            # If value is number, put in list (for aggregation below)
            # if value_is_number:
            #     projection_value = [projection_value]

            # Add projection_value to list (for aggregation below)
            projection_value_list.append(projection_value)

        #endregion
        #region Aggregate projection values

        # If there were projections:
        if projection_value_list:
            try:
                # pass only execute_method params
                execute_method_params = params[kwExecuteMethodParams]
            except (KeyError, TypeError):
                execute_method_params = NotImplemented

            # Combine projecction values
            combined_values = self.execute(variable=projection_value_list,
                                           params=execute_method_params,
                                           context=context)

            # If self.value is a number, convert combined_values back to number
            if value_is_number and combined_values:
                combined_values = combined_values[0]

        # There were no projections
        else:
            # mark combined_values as none, so that (after being assigned to self.value)
            #    it is ignored in execute method (i.e., not combined with baseValue)
            combined_values = None
        #endregion

        #region ASSIGN STATE VALUE
        context = context + kwAggregate + ' Projection Inputs'
        self.value = combined_values
        #endregion

    @property
    def ownerMechanism(self):
        return self._ownerMechanism

    @ownerMechanism.setter
    def ownerMechanism(self, assignment):
        self._ownerMechanism = assignment

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):

        self._value = assignment

        # Store value in log if specified
        # Get logPref
        if self.prefs:
            log_pref = self.prefs.logPref

        # Get context
        try:
            curr_frame = inspect.currentframe()
            prev_frame = inspect.getouterframes(curr_frame, 2)
            context = inspect.getargvalues(prev_frame[1][0]).locals['context']
        except KeyError:
            context = ""

        # If context is consistent with log_pref, record value to log
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and kwExecuting in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (kwExecuting in context and kwAssign in context))):
            self.ownerMechanism.log.entries[self.name] = LogEntry(CurrentTime(), context, assignment)
            # self.ownerMechanism.log.entries[self.name] = LogEntry(CentralClock, context, assignment)

    @property
    def baseValue(self):
        return self._baseValue

    @baseValue.setter
    def baseValue(self, item):
        self._baseValue = item

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, assignment):
        self._projections = assignment

