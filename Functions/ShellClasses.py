#
# ******************************************  ShellClasses *************************************************************
#

from Functions.Function import *

# import Functions.Process
# import Functions.Mechanisms
# import Functions.MechanismStates
# import Functions.Projections


class ShellClassError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class ShellClass(Function):
    pass

# ******************************************* SYSTEM *******************************************************************

class System(ShellClass):

    # def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
    #     raise ShellClassError("Must implement validate_params in {0}".format(self.__class__.__name__))
    def execute(self, time_scale=NotImplemented, context=NotImplemented):
        raise ShellClassError("Must implement execute in {0}".format(self.__class__.__name__))

# ****************************************** PROCESS *******************************************************************

class Process(ShellClass):

    # def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
    #     raise ShellClassError("Must implement validate_params in {0}".format(self.__class__.__name__))
    def execute(self, input=NotImplemented, time_scale=NotImplemented, params=NotImplemented, context=NotImplemented):
        raise ShellClassError("Must implement execute in {0}".format(self.__class__.__name__))
    # def set_log(self, log):
    #     raise ShellClassError("Must implement set_log in {0}".format(self.__class__.__name__))
    # def log_all_entries(self, mechanism):
    #     raise ShellClassError("Must implement log_all_entries in {0}".format(self.__class__.__name__))
    # def get_configuration(self):
    #     raise ShellClassError("Must implement get_configuration in {0}".format(self.__class__.__name__))
    # def get_mechanism_dict(self):
    #     raise ShellClassError("Must implement get_mechanism_dict in {0}".format(self.__class__.__name__))


# ******************************************* MECHANISM ****************************************************************

ParamValueProjection = namedtuple('ParamValueProjection', 'value projection')

class Mechanism(ShellClass):
    # @property
    # def inputState(self):
    #     raise ShellClassError("Must implement @property inputState method in {0}".format(self))
    # @inputState.setter
    # def inputState(self, value):
    #     raise ShellClassError("Must implement @inputState.setter method in {0}".format(self))
    # @property
    # def outputState(self):
    #     raise ShellClassError("Must implement @property outputState method in {0}".format(self))
    # @outputState.setter
    # def outputState(self, value):
    #     raise ShellClassError("Must implement @outputState.setter method in {0}".format(self))
    # def validate_variable(self, variable, context=NotImplemented):
    #     raise ShellClassError("Must implement validate_variable in {0}".format(self))
    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        raise ShellClassError("Must implement validate_params in {0}".format(self))
    # def check_mechanism_parameter_state_value(self, param_name, value):
    #     raise ShellClassError("Must implement check_mechanism_parameter_state_value in {0}".format(self))
    # def check_mechanism_state_ownership(self, param_name, mechanism_state):
    #     raise ShellClassError("Must implement check_mechanism_state_ownership in {0}".format(self))
    def execute(self, params, time_scale, context):
        raise ShellClassError("Must implement execute in {0}".format(self))
    # def update_parameter_states(self):
    #     raise ShellClassError("Must implement update_parameter_states in {0}".format(self))
    def adjust_function(self, params, context):
        raise ShellClassError("Must implement adjust_function in {0}".format(self))
    # def terminate_function(self, context):
    #     raise ShellClassError("Must implement terminate_function in {0}".format(self))
    # def get_mechanism_param_values(self):
    #     raise ShellClassError("Must implement get_mechanism_param_values in {0}".format(self))


# **************************************** MECHANISM_STATE *************************************************************

class MechanismState(ShellClass):
    @property
    def ownerMechanism(self):
        raise ShellClassError("Must implement @property ownerMechanism method in {0}".format(self.__class__.__name__))
    @ownerMechanism.setter
    def ownerMechanism(self, assignment):
        raise ShellClassError("Must implement @ownerMechanism.setter method in {0}".format(self.__class__.__name__))
    # @property
    # def value(self):
    #     raise ShellClassError("Must implement @property value method in {0}".format(self.__class__.__name__))
    # @value.setter
    # def value(self, assignment):
    #     raise ShellClassError("Must implement @value.setter method in {0}".format(self.__class__.__name__))
    @property
    def projections(self):
        raise ShellClassError("Must implement @property projections method in {0}".format(self.__class__.__name__))
    @projections.setter
    def projections(self, assignment):
        raise ShellClassError("Must implement @projections.setter method in {0}".format(self.__class__.__name__))
    def validate_variable(self, variable, context=NotImplemented):
        raise ShellClassError("Must implement validate_variable in {0}".format(self))
    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        raise ShellClassError("Must implement validate_params in {0}".format(self))
    def add_observer_for_keypath(self, object, keypath):
        raise ShellClassError("Must implement add_observer_for_keypath in {0}".format(self.__class__.__name__))
    def set_value(self, new_value):
        raise ShellClassError("Must implement set_value in {0}".format(self.__class__.__name__))
    def update(self, params=NotImplemented, context=NotImplemented):
        raise ShellClassError("Must implement update_state in {0}".format(self.__class__.__name__))

# class MechanismInputState(MechanismState):
#     def validate_variable(self, variable, context=NotImplemented):
#         raise ShellClassError("Must implement validate_variable in {0}".format(self))
#
# class MechanismOutputState(MechanismState):
#     def validate_variable(self, variable, context=NotImplemented):
#         raise ShellClassError("Must implement validate_variable in {0}".format(self))

# class MechanismParameterState(MechanismState):
#     pass

# ******************************************* PROJECTION ***************************************************************

class Projection(ShellClass):

    # def assign_states(self):
    #     raise ShellClassError("Must implement assign_states in {0}".format(self.__class__.__name__))
    def validate_states(self):
        raise ShellClassError("Must implement validate_states in {0}".format(self.__class__.__name__))
    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        raise ShellClassError("Must implement validate_params in {0}".format(self.__class__.__name__))


# ******************************************** UTILITY *****************************************************************

class Utility(ShellClass):
    def execute(self, variable, params):
        raise ShellClassError("Must implement function in {0}".format(self))
