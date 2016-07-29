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
        super(Mapping, self).__init__(sender=sender,
                                      receiver=receiver,
                                      params=params,
                                      name=name,
                                      prefs=prefs,
                                      context=self)
        TEST = True

    def instantiate_sender(self, context=NotImplemented):
        """Parse sender (Mechanism vs. MechanismState) and insure that length of sender.value is same as self.variable

        :param context:
        :return:
        """

        # IMPLEMENTATION NOTE: RESPONSIBILITY FOR THIS REALLY SHOULD LIE IN CALL FROM Process
        # # If sender is a ProcessBufferState and this projection is for its first Mechanism, it is OK
        # from Functions.Process import ProcessInputState
        # if isinstance(self.sender, ProcessInputState):
        #     # mech_num = len(self.sender.ownerMechanism.configurationMechanismNames)
        #     mech_num = len(self.sender.ownerMechanism.mechanism_list)
        #     if mech_num > 1:
        #         raise ProjectionError("Illegal attempt to add projection from {0} to mechanism {0} in "
        #                               "configuration list; this is only allowed for first mechanism in list".
        #                               format(self.sender.name, ))

        super(Mapping, self).instantiate_sender(context=context)

# MODIFIED 7/9/16 MOVED CONTENTS TO instantiate_receiver() TO CORRECT PROBLEMS BELOW:
#     def instantiate_execute_method(self, context=NotImplemented):
#         """Check that length of receiver.variable is same as self.value
#
#         :param context:
#         :return:
#         """
# # FIX 6/12/16 ** MOVE THIS TO BELOW, SO THAT IT IS CALLED WITH SENDER AND RECEIVER LENGTHS??
#         # PASS PARAMS (WITH kwReceiver) TO INSTANTIATE_EXECUTE_METHOD??
#         super(Mapping, self).instantiate_execute_method(context=context)
#
# # FIX:        CAN'T REFERENCE self.receiver
# # FIX:              SINCE instantiate_receiver IS NOT CALLED UNTIL instantiate_attributes_after_execute_method()
# # FIX:              SO self.receiver MAY STILL BE A Mechanism, NOT INSTANTIATED, OR VALIDATED
# # FIX:        ?? MOVE TO Projection.instantiate_receiver()
#         try:
# #             # MODIFIED 7/9/16 OLD:
# #             receiver_len = len(self.receiver.value)
#             # MODIFIED 7/9/16 NEW:
#             receiver_len = len(self.receiver.variable)
#         except TypeError:
#             receiver_len = 1
#         try:
#             mapping_input_len = len(self.value)
#         except TypeError:
#             mapping_input_len = 1
#
#         if receiver_len != mapping_input_len:
#             raise ProjectionError("Length ({0}) of outputState for {1} must equal length ({2})"
#                                   " of variable for {4} projection".
#                                   format(receiver_len,
#                                          self.sender.name,
#                                          mapping_input_len,
#                                          kwMapping,
#                                          self.name))

    def instantiate_receiver(self, context=NotImplemented):
        """Handle situation in which self.receiver was specified as a Mechanism (rather than MechanismState)

        If receiver is specified as a Mechanism, it is reassigned to the (primary) inputState for that Mechanism
        If the Mechanism has more than one inputState, assignment to other inputStates must be done explicitly
            (i.e., by: instantiate_receiver(MechanismState)

        """
        # Assume that if Mechanism was specified as receiver, it should be assigned to (primary) inputState
        if isinstance(self.receiver, Mechanism):
            if (len(self.receiver.inputStates) > 1 and
                    (self.prefs.verbosePref or self.receiver.prefs.verbosePref)):
                print("{0} has more than one inputState; {1} was assigned to the first one".
                      format(self.receiver.ownerMechanism.name, self.name))
            self.receiver = self.receiver.inputState


        # MODIFIED 7/9/16 NEW [MOVED FROM instantiate_execute_method ABOVE]:
        try:
#             # MODIFIED 7/9/16 OLD:
#             receiver_len = len(self.receiver.value)
            # MODIFIED 7/9/16 NEW:
            receiver_len = len(self.receiver.variable)
        except TypeError:
            receiver_len = 1
        try:
            mapping_input_len = len(self.value)
        except TypeError:
            mapping_input_len = 1

        if receiver_len != mapping_input_len:
            # # MODIFIED 7/10/16 OLD:
            # raise ProjectionError("Length ({0}) of outputState for {1} must equal length ({2})"
            #                       " of variable for {4} projection".
            #                       format(receiver_len,
            #                              self.sender.name,
            #                              mapping_input_len,
            #                              kwMapping,
            #                              self.name))
            # MODIFIED 7/10/16 NEW:
            raise ProjectionError("Length ({0}) of output for {1} projection from {2}"
                                  " must equal length ({3}) of {4} inputState".
                                  format(mapping_input_len,
                                         self.name,
                                         self.sender.name,
                                         receiver_len,
                                         self.receiver.ownerMechanism.name))

        super(Mapping, self).instantiate_receiver(context=context)

    def update(self, params=NotImplemented, context=NotImplemented):
        """

        :return:
        """

        # IMPLEMENTATION NOTE:  ADD LearningSignal HERE IN FUTURE
        # super(Mapping, self).update(params=, context)

        return self.execute(self.sender.value, params=params, context=context)
