# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************  LearningSignal **********************************************************
#

from Functions.Projections.Projection import *
from Functions.Utility import *

# Params:
kwLearningRate = "LearningRate"

class LearningSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningSignal(Projection_Base):
    """Implement projection conveying values from output of a mechanism to input of another (default: IdentityMapping)

    Description:
        The LearningSignal class is a functionType in the Projection category of Function,
        It's execute method uses the MechanismOutputState.value of a MonitoringMechanism
            to adjust the kwMatrix parameter of (in kwExecuteMethodParams) of a receiver Mapping Projection

    Instantiation:
        - LearningSignal Projections are instantiated by specifying a MonitoringMechanism sender and a Mapping receiver

    Initialization arguments:
        - sender (MonitoringMechanism) - source of projection input (default: TBI)
        - receiver: (Mapping Projection) - destination of projection output (default: TBI)
        - params (dict) - dictionary of projection params:
            + kwExecuteMethod (Utility): (default: BP)
            + kwExecuteMethodParams (dict):
                + kwLearningRate (value): (default: 1)
        - name (str) - if it is not specified, a default based on the class is assigned in register_category
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)

    Parameters:
        The default for kwExecuteMethod is BackPropagation:
        The parameters of kwExecuteMethod can be set:
            - by including them at initialization (param[kwExecuteMethod] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwExecuteMethod].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)

    ProjectionRegistry:
        All LearningSignal projections are registered in ProjectionRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        LearningSignal projections can be named explicitly (using the name argument).  If this argument is omitted,
        it will be assigned "LearningSignal" with a hyphenated, indexed suffix ('LearningSignal-n')

    Class attributes:
        + className = kwLearningSignal
        + functionType = kwProjection
        # + defaultSender (MechanismState)
        # + defaultReceiver (MechanismState)
        + paramClassDefaults (dict):
            + kwExecuteMethod (Utility): (default: BP)
            + kwExecuteMethodParams:
                + kwLearningRate (value): (default: 1)
        + paramNames (dict)
        + classPreference (PreferenceSet): LearningSignalPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

    Class methods:
        function (executes function specified in params[kwExecuteMethod]

    Instance attributes:
        + sender (MonitoringMechanism)
        + receiver (Mapping)
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramsCurrent (dict) - set currently in effect
        + variable (value) - used as input to projection's execute method
        + value (value) - output of execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying LearningSignalPreferenceSet

    Instance methods:
        none
    """

    functionType = kwLearningSignal
    className = functionType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwExecuteMethod:kwBP,
                               kwExecuteMethodParams: {
                                   kwLearningRate: 1}
                               })

    def __init__(self,
                 sender=NotImplemented,
                 receiver=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """
IMPLEMENTATION NOTE:  *** DOCUMENTATION NEEDED (SEE CONTROL SIGNAL)

        :param sender:
        :param receiver:
        :param params:
        :param name:
        :param context:
        :return:
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(sender=sender,
                         receiver=receiver,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Insure that sender is a MonitoringMechanism, the output of which is compatible with self.variable
         """

        #
        # try:
        #     param_value = params[kwMechanismInputStates]
        #
        # except KeyError:
        #     # kwMechanismInputStates not specified:
        #     # - set to None, so that it is set to default (self.variable) in instantiate_inputState
        #     # - if in VERBOSE mode, warn in instantiate_inputState, where default value is known
        #     params[kwMechanismInputStates] = None

        super().validate_params(request_set=request_set, target_set=target_set, context=NotImplemented)


    def instantiate_sender(self, context=NotImplemented):
        """Instantiate and assign default MonitoringMechanism if necessary

        :param context:
        :return:
        """


        super().instantiate_sender(context=context)



    def instantiate_receiver(self, context=NotImplemented):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a Mapping Projection, it is assigned to the kwMatrix MechanismParameterState
            for the projection;  if that does not exist, it is instantiated and assigned as the receiver

        Handle situation in which receiver is specified as a Projection (rather than MechanismParameterState)

        """

        receiver_parameter_state_name = kwMatrix

        from Functions.MechanismStates.MechanismInputState import instantiate_mechanism_state_list
        from Functions.MechanismStates.MechanismInputState import instantiate_mechanism_state
        from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
        from Functions.Projections.Mapping import Mapping

        # If receiver was specified as a Projection, it should be assigned to its kwMatrix MechanismParameterState
        if isinstance(self.receiver, Mapping):
            try:
                self.receiver = self.receiver.executeMethodParameterStates
            # receiver has no executeMethodParameterStates
            except AttributeError:
                # Get receiver.paramsCurrent[executeMethodParams][kwMatrix]
                try:
                    receiver_weight_matrix = self.receiver.paramsCurrent[kwExecuteMethodParams][kwMatrix],
                # Sanity check:  this should never occur; Mapping Projection should have kwMatrix in paramClassDefaults
                except KeyError:
                    raise LearningSignal("PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".
                                         format(self.receiver.name, kwExecuteMethodParams, kwMatrix))

                # Instantiate executeMethodParameterStates Ordered dict with MechanismParameterState for kwMatrix param
                self.receiver.executeMethodParameterStates = instantiate_mechanism_state_list(
                                                                    owner=self.receiver,
                                                                    state_list=[receiver_parameter_state_name],
                                                                    state_type=MechanismParameterState,
                                                                    state_param_identifier=kwMechanismParameterState,
                                                                    constraint_values=receiver_weight_matrix,
                                                                    constraint_values_name=kwLearningSignal,
                                                                    context=context)
            # receiver has executeMethodParameterStates but not (yet!) one for kwMatrix
            except KeyError:
                # Instantiate MechanismParameterState for kwMatrix
                self.receiver.executeMethodParameterStates[receiver_parameter_state_name] = \
                                        instantiate_mechanism_state(owner=self.receiver,
                                                                    state_type=MechanismParameterState,
                                                                    state_name=receiver_parameter_state_name,
                                                                    state_spec=kwMechanismParameterState,
                                                                    constraint_values=receiver_weight_matrix,
                                                                    constraint_values_name=kwLearningSignal,
                                                                    context=context)




        # Insure that Mapping output and receiver's variable are the same length
        try:
            receiver_len = len(self.receiver.variable)
        except TypeError:
            receiver_len = 1
        try:
            mapping_input_len = len(self.value)
        except TypeError:
            mapping_input_len = 1

        if receiver_len != mapping_input_len:
            raise ProjectionError("Length ({0}) of output for {1} projection from {2}"
                                  " must equal length ({3}) of {4} inputState".
                                  format(mapping_input_len,
                                         self.name,
                                         self.sender.name,
                                         receiver_len,
                                         self.receiver.ownerMechanism.name))

        super().instantiate_receiver(context=context)

    def update(self, params=NotImplemented, context=NotImplemented):
        """

        :return:
        """

        # IMPLEMENTATION NOTE:  ADD LearningSignal HERE IN FUTURE
        # super(Mapping, self).update(params=, context)

        return self.execute(self.sender.value, params=params, context=context)
