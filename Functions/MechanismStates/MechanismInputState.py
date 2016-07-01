#
# *******************************************  MechanismInputState *****************************************************
#

from Functions.MechanismStates.MechanismState import *
from Functions.Utility import *


# MechanismInputStatePreferenceSet = FunctionPreferenceSet(log_pref=logPrefTypeDefault,
#                                                          reportOutput_pref=reportOutputPrefTypeDefault,
#                                                          verbose_pref=verbosePrefTypeDefault,
#                                                          param_validation_pref=paramValidationTypeDefault,
#                                                          level=PreferenceLevel.TYPE,
#                                                          name='MechanismInputStateClassPreferenceSet')

# class MechanismInputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


class MechanismInputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class MechanismInputState(MechanismState_Base):
    """Implement subclass type of MechanismState that calculates and represents input of a Mechanism

    Description:
        The MechanismInputState class is a functionType in the MechanismState category of Function,
        Its kwExecuteMethod executes the projections that it receives and updates the MechanismInputState's value

    Instantiation:
        - kwMechanismInputState can be instantiated in one of two ways:
            - directly: requires explicit specification of its value and ownerMechanism
            - as part of the instantiation of a mechanism:
                - the mechanism for which it is being instantiated will automatically be used as the ownerMechanism
                - the value of the ownerMechanism's variable will be used as its value
        - self.value is set to self.variable (enforced in MechanismState_Base.validate_variable)
        - self.value must be compatible with self.ownerMechanism.variable (enforced in validate_variable)
            note: although it may receive multiple projections, the output of each must conform to self.variable,
                  as they will be combined to produce a single value that must be compatible with self.variable
        - self.executeMethod (= params[kwExecuteMethod]) must be Utility.LinearCombination (enforced in validate_params)
        - output of self.executeMethod must be compatible with self.value (enforced in validate_params)
        - if ownerMechanism is being instantiated within a configuration:
            - MechanismInputState will be assigned as the receiver of a Mapping projection from the preceding mechanism
            - if it is the first mechanism in the list, it will receive a Mapping projection from process.input

    Parameters:
        The default for kwExecuteMethod is LinearCombination using kwAritmentic.Operation.SUM:
            the output of all projections it receives are summed
# IMPLEMENTATION NOTE:  *** CONFIRM THAT THIS IS TRUE:
        kwExecuteMethod can be set to another function, so long as it has type kwLinearCombinationFunction
        The parameters of kwExecuteMethod can be set:
            - by including them at initialization (param[kwExecuteMethod] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwExecuteMethod].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)

    MechanismStateRegistry:
        All kwMechanismInputState are registered in MechanismStateRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        kwMechanismInputState can be named explicitly (using the name='<name>' argument). If this argument is omitted,
         it will be assigned "MechanismInputState" with a hyphenated, indexed suffix ('MechanismInputState-n')


    Class attributes:
        + functionType (str) = kwMechanismInputState
        + paramClassDefaults (dict)
            + kwExecuteMethod (LinearCombination, Operation.SUM)
            + kwExecuteMethodParams (dict)
            # + kwMechanismStateProjectionAggregationFunction (LinearCombination, Operation.SUM)
            # + kwMechanismStateProjectionAggregationMode (LinearCombination, Operation.SUM)
        + paramNames (dict)

    Class methods:
        instantiate_execute_method: insures that execute method is ARITHMETIC)
        update_state: gets MechanismInputStateParams and passes to super (default: LinearCombination with Operation.SUM)



    Instance attributes:
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + params (dict) - set currently in effect
        + paramNames (list) - list of keys for the params dictionary
        + ownerMechanism (Mechanism)
        + value (value)
        + projections (list)
        + params (dict)
        + name (str)
        + prefs (dict)

    Instance methods:
        none
    """

    #region CLASS ATTRIBUTES

    functionType = kwMechanismInputState

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'MechanismInputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Note: the following enforce encoding as 1D np.ndarrays (one variable/value array per state)
    variableEncodingDim = 1
    valueEncodingDim = 1

    paramClassDefaults = MechanismState_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwExecuteMethod: LinearCombination,
                               kwExecuteMethodParams: {kwOperation: LinearCombination.Operation.SUM},
                               kwProjectionType: kwMapping})

    #endregion

    def __init__(self,
                 owner_mechanism,
                 reference_value=NotImplemented,
                 value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """
IMPLEMENTATION NOTE:  *** DOCUMENTATION NEEDED (SEE CONTROL SIGNAL??)
reference_value is component of Mechanism.variable that corresponds to the current MechanismState

        :param owner_mechanism: (Mechanism)
        :param reference_value: (value)
        :param value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        :return:


        """
        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of functionName in super.__init__)
        super(MechanismInputState, self).__init__(owner_mechanism,
                                                  value=value,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)

    # # FIX:  MOVED THIS TO instantiate_execute_method, SINCE IT IS REALLY self.value THAT MATTERS, not self.variable
    # def validate_variable(self, variable, context=NotImplemented):
    #     """Insure self.value is compatible with component of ownerMechanism.variable relevant to this state
    #
    #     Validate self.value (= self.variable) against component of ownerMechanism.executeMethod's variable that
    #         corresponds to this inputState, since the latter will be called with the value of this MechanismInputState;
    #         this should have been provided as reference_value in the call to MechanismOutputState__init__()
    #
    #     Notes:
    #     # * Self.variable should have been assigned to self.value at this point (in super.validate_variable)
    #     * Self.value = self.variable is insured by constraint in instantiate_execute_method
    #     * This method is called only if the parameterValidationPref is True
    #
    #     :param variable: (anything but a dict) - variable to be validated:
    #     :param context: (str)
    #     :return none:
    #     """
    #
    #     super(MechanismInputState,self).validate_variable(variable, context)
    #
    #     # Insure that self.value is compatible with (relevant item of ) self.ownerMechanism.variable
    #     if not iscompatible(self.value, self.reference_value):
    #         raise MechanismInputStateError("Value ({0}) of inputState for {1} is not compatible with "
    #                                        "the variable ({2}) of its execute method".
    #                                        format(self.value,
    #                                               self.ownerMechanism.name,
    #                                               self.ownerMechanism.variable))

    def instantiate_execute_method(self, context=NotImplemented):
        """Insure that execute method is LinearCombination and that output is compatible with ownerMechanism.variable

        Insures that execute method:
            - is LinearCombination (to aggregate projection inputs)
            - generates an output (assigned to self.value) that is compatible with the component of
                ownerMechanism.executeMethod's variable that corresponds to this inputState,
                since the latter will be called with the value of this MechanismInputState;

        Notes:
        * Relevant component of ownerMechanism.executeMethod's variable should have been provided
            as reference_value arg in the call to MechanismInputState__init__()
        * Insures that self.value has been assigned (by call to super().validate_execute_method)
        * This method is called only if the parameterValidationPref is True

        :param context:
        :return:
        """

        super(MechanismInputState, self).instantiate_execute_method(context=context)

        # Insure that execute method is Utility.LinearCombination
        if not isinstance(self.execute.__self__, (LinearCombination, Linear)):
            raise MechanismStateError("{0} of {1} for {2} is {3}; it must be of LinearCombination or Linear type".
                                      format(kwExecuteMethod,
                                             self.name,
                                             self.ownerMechanism.name,
                                             self.execute.__self__.functionName, ))

        # Insure that self.value is compatible with (relevant item of ) self.ownerMechanism.variable
        if not iscompatible(self.value, self.reference_value):
            raise MechanismInputStateError("Value ({0}) of {1} for {2} mechanism is not compatible with "
                                           "the variable ({2}) of its execute method".
                                           format(self.value,
                                                  self.name,
                                                  self.ownerMechanism.name,
                                                  self.ownerMechanism.variable))

    def update(self, params=NotImplemented, time_scale=TimeScale.TRIAL, context=NotImplemented):
        """Process inputState params, and pass params for inputState projections to super for processing

        :param params:
        :param time_scale:
        :param context:
        :return:
        """

        try:
            # Get inputState params
            input_state_params = params[kwMechanismInputStateParams]

        except (KeyError, TypeError):
            input_state_params = NotImplemented

        # Process any inputState params here
        pass

        super(MechanismInputState, self).update(params=input_state_params,
                                                      time_scale=time_scale,
                                                      context=context)


