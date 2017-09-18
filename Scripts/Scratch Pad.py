# GLOBALS:

# MECHANISMS:
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.LCA import LCA, LCA_OUTPUT

from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVC.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Functions.Function import Logistic, Linear, LinearCombination
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms.DDM import DDM, DDM_OUTPUT, \
    DECISION_VARIABLE,RESPONSE_TIME, PROBABILITY_UPPER_THRESHOLD
from PsyNeuLink.Globals.Keywords import GAIN, THRESHOLD, SUM, PRODUCT, CONTROL, IDENTITY_MATRIX, RESULT, MEAN, VARIANCE

# COMPOSITIONS:
from PsyNeuLink.Components.Process import process

# FUNCTIONS:
from PsyNeuLink.Components.Functions.Function import CombineMeans

# STATES:
# from PsyNeuLink.Components.States.OutputState import OutputState
# from PsyNeuLink.Globals.Keywords import PARAMETER_STATE_PARAMS


# PROJECTIONS:
# from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
# from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControldProjection
# from PsyNeuLink.Components.States.ParameterState import ParameterState, PARAMETER_STATE_PARAMS
from PsyNeuLink.Components.Functions.Function import BogaczEtAl


class ScratchPadError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

# ----------------------------------------------- PsyNeuLink -----------------------------------------------------------


#region USER GUIDE
# from PsyNeuLink.Components.Process import process, Process_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism

#region SIMPLE NN EXAMPLE:

# print("SIMPLE NN EXAMPLE")
# # input_layer = TransferMechanism(size=5)
# # hidden_layer = TransferMechanism(size=2, function=Logistic)
# # output_layer = TransferMechanism(size=5, function=Logistic)
# input_layer = TransferMechanism(default_variable=[0,0,0,0,0])
# hidden_layer = TransferMechanism(default_variable=[0,0], function=Logistic)
# output_layer = TransferMechanism(default_variable=[0,0,0,0,0], function=Logistic)
# # my_process = process(pathway=[input_layer, hidden_layer, output_layer], target=[0,0,0,0,0], learning=LEARNING)
# my_process = process(pathway=[input_layer, hidden_layer, output_layer], learning=ENABLED)
#
# # my_system = system(processes=[my_process], targets=[0,0,0,0,0])
# my_system = system(processes=[my_process])
# # my_system.show_graph(show_learning=True, direction='TB')
# my_system.show_graph(show_control=True, direction='TB')
# # MappingProjection(sender=output_layer,
# #                   receiver=hidden_layer,
# #                   matrix=((.2 * np.random.rand(5, 2)) + -.1))
# # print(output_layer.execute([2,2,2,2,2]))
#
# # print(process.execute([2,2,2,2,2]))

#endregion

#region SIMPLE STROOP EXAMPLE:

# print("SIMPLE NN EXAMPLE")
# VERSION 1
# colors_input_layer = TransferMechanism(default_variable=[0,0],
#                                        function=Logistic,
#                                        name='COLORS INPUT')
# words_input_layer = TransferMechanism(default_variable=[0,0],
#                                        function=Logistic,
#                                        name='WORDS INPUT')
# output_layer = TransferMechanism(default_variable=[0,0],
#                                        function=Logistic,
#                                        name='OUTPUT')
# decision_mech = DDM(name='DECISION')
# colors_process = process(pathway=[colors_input_layer, FULL_CONNECTIVITY_MATRIX, output_layer], name='COLOR PROCESS')
# words_process = process(pathway=[words_input_layer, FULL_CONNECTIVITY_MATRIX, output_layer], name='WORD PROCESS')
# decision_process = process(pathway=[output_layer, FULL_CONNECTIVITY_MATRIX, decision_mech], name='DECISION_PROCESS')
# my_simple_Stroop = system(processes=[colors_process, words_process, decision_process])
#

# VERSION 2:
# differencing_weights = np.array([[1], [-1]])
# colors_input_layer = TransferMechanism(default_variable=[0,0], function=Logistic, name='COLORS INPUT')
# words_input_layer = TransferMechanism(default_variable=[0,0], function=Logistic, name='WORDS INPUT')
# output_layer = TransferMechanism(default_variable=[0], name='OUTPUT')
# decision_mech = DDM(name='DECISION')
# colors_process = process(pathway=[colors_input_layer, differencing_weights, output_layer],
#                          target=[0],
#                          name='COLOR PROCESS')
# words_process = process(pathway=[words_input_layer, differencing_weights, output_layer],
#                         target=[0],
#                         name='WORD PROCESS')
# decision_process = process(pathway=[output_layer, decision_mech],
#                            name='DECISION PROCESS')
# my_simple_Stroop = system(processes=[colors_process, words_process],
#                           targets=[0])
#
# my_simple_Stroop.show_graph(direction='LR')
# print(my_simple_Stroop.run(inputs=[-1, 1], targets=[-1, 1]))

#endregion

#region TEST whether function attribute assignment is used and "sticks"

# my_mech = IntegratorMechanism()
# # my_mech.function_object.rate = 2.0
# print(my_mech.execute())
# my_mech.function_object.rate = 0.9
# print(my_mech.execute())
# my_mech.function_object.rate = .75
# print(my_mech.function_object.rate)
# my_mech.function_object.rate = .2
# print(my_mech.execute())

#endregion

#region TEST Multipe Inits

# # WORKS:

# my_mech = mechanism()
# print(my_mech.name)
# my_mech = mechanism()
# print(my_mech.name)
# my_mech = mechanism()
# print(my_mech.name)
# my_mech = mechanism()
# print(my_mech.name)

# my_mech = Mechanism()

# my_mech = Mechanism_Base()

# my_process = process()
# print(my_process.name)
# my_process = process()
# print(my_process.name)
# my_process = process()
# print(my_process.name)
# my_process = process()
# print(my_process.name)

# my_process = Process()
# print(my_process.name)

# my_process = Process_Base()
# print(my_process.name)
# my_process = Process_Base()
# print(my_process.name)
# my_process = Process_Base()
# print(my_process.name)

# my_sys = system()
# print(my_sys.name)
# my_sys = system()
# print(my_sys.name)
# my_sys = system()
# print(my_sys.name)
# my_sys = system()
# print(my_sys.name)

# my_sys = System()

# my_sys = System_Base()
# print(my_sys.name)
# my_sys = System_Base()
# print(my_sys.name)
# my_sys = System_Base()
# print(my_sys.name)

#endregion

#region TEST arg vs. paramClassDefault @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# myTransfer = TransferMechanism(output_states=[*TransferMechanism.my_mean, {NAME: 'NEW_STATE'}],
#                                # input_states={NAME: 'MY INPUT'}
#                                )
# TEST_CONDITION = True

#endregion

# region TEST ReadOnlyOrderedDict

# from collections import UserDict, OrderedDict
#
# # class ReadOnlyOrderedDict(OrderedDict):
# #     def __init__(self, dict=None, **kwargs):
# #         UserDict.__init__(self, dict, **kwargs)
# #         self._ordered_keys = []
# #         for key in list(dict.keys()):
# #             self._ordered_keys.append(key)
# #         TEST = True
# #     def __setitem__(self, key, item):
# #         raise TypeError
# #     def __delitem__(self, key):
# #         raise TypeError
# #     def clear(self):
# #         raise TypeError
# #     def pop(self, key, *args):
# #         raise TypeError
# #     def popitem(self):
# #         raise TypeError
# #
# #     def update(self, dict=None):
# #         if dict is None:
# #             pass
# #         elif isinstance(dict, UserDict):
# #             self.data = dict.data
# #         elif isinstance(dict, type({})):
# #             self.data = dict
# #         else:
# #             raise TypeError
# #
# #     def okeys(self):
# #         return self._ordered_keys
# #
# #     # def __setitem__(self, key, value):
# #     #     self.data[key] = item
# #     #     self._ordered_keys.append(key)
# #
# # x = ReadOnlyOrderedDict(OrderedDict({'hello':1, 'goodbye':2}))
# # print(x.okeys())
#
#
# # Ordered UserDict
# class ReadOnlyOrderedDict(UserDict):
#     def __init__(self, dict=None, name=None, **kwargs):
#         self.name = name or self.__class__.__name__
#         UserDict.__init__(self, dict, **kwargs)
#         self._ordered_keys = []
#     def __setitem__(self, key, item):
#         raise ScratchPadError("{} is read-only".format(self.name))
#     def __delitem__(self, key):
#         raise TypeError
#     def clear(self):
#         raise TypeError
#     def pop(self, key, *args):
#         raise TypeError
#     def popitem(self):
#         raise TypeError
#     def __additem__(self, key, value):
#         self.data[key] = value
#         if not key in self._ordered_keys:
#             self._ordered_keys.append(key)
#     def keys(self):
#         return self._ordered_keys
#
# x = ReadOnlyOrderedDict()
# x.__additem__('hello',1)
# x.__additem__('hello',2)
# x.__additem__('goodbye',2)
# print(x.keys())
# for key in x.keys():
#     print(x[key])
# # x['new item']=3
#
# # # ReadOnly UserDict
# # class ReadOnlyOrderedDict(OrderedDict):
# # 	def __setitem__(self, key, item): raise TypeError
# # 	def __delitem__(self, key): raise TypeError
# # 	def clear(self): raise TypeError
# # 	def pop(self, key, *args): raise TypeError
# # 	def popitem(self): raise TypeError
# # 	def __additem__(selfself, key, item):
# #
# #
# # 	def update(self, dict=None):
# # 		if dict is None:
# # 			pass
# # 		elif isinstance(dict, UserDict):
# # 			self.data = dict.data
# # 		elif isinstance(dict, type({})):
# # 			self.data = dict
# # 		else:
# # 			raise TypeError
# #
# # x = ReadOnlyDict({'hello':1, 'goodbye':2})
# # print(list(x.keys()))
# # # x['new'] = 4
# #
# #
# # # ReadOnly UserDict
# # x = ReadOnlyDict()
# # x['hello'] = 1
# # x['goodbye'] = 2
# # print(list(x.keys()))
# #

#endregion

#region TEST AUTO_PROP @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# defaults = {'foo':5, 'bar': ['hello', 'world']}
#
# docs = {'foo': 'Foo controls the fooness, as modulated by the the bar',
#         'bar': 'Bar none, the most important property'}
#
#
# def make_property(name):
#     backing_field = '_' + name
#
#     def getter(self):
#         if hasattr(self, backing_field):
#             return getattr(self, backing_field)
#         else:
#             return defaults[name]
#
#     def setter(self, val):
#         setattr(self, backing_field, val)
#
#     # Create the property
#     prop = property(getter).setter(setter)
#
#     # Install some documentation
#     prop.__doc__ = docs[name]
#     return prop
#
#
# def autoprop(cls):
#     for k, v in defaults.items():
#         setattr(cls, k, make_property(k))
#     return cls
#
#
# @autoprop
# class Test:
#     pass
#
# if __name__ == '__main__':
#     t = Test()
#     t2 = Test()
#     print("Stored values in t", t.__dict__)
#     print("Properties on t", dir(t))
#     print("Check that default values are there by default")
#     assert t.foo == 5
#     assert t.bar == ['hello', 'world']
#     print("Assign and check the assignment holds")
#     t.foo = 20
#     assert t.foo == 20
#     print("Check that assignment on t didn't change the defaulting on t2 somehow")
#     assert t2.foo == 5
#     print("Check that changing the default changes the value on t2")
#     defaults['foo'] = 27
#     assert t2.foo == 27
#     print("But t1 keeps the value it was assigned")
#     assert t.foo == 20
#     print(""""Note that 'help(Test.foo)' and help('Test.bar') will show
#     the docs we installed are available in the help system""")
# #endregion

#region TEST Linear FUNCTION WITH MATRIX @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# print = Linear(variable=[[1,1],[2,2]])
#endregion

#region TEST 2 Mechanisms and a Projection @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
# from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
#
# my_mech_A = IntegratorMechanism()
# my_mech_B = TransferMechanism()
# proj = MappingProjection(sender=my_mech_A,
#                          receiver=my_mech_B)
# my_mech_B.execute(context=EXECUTING)
#
#endregion

#region TEST Modulation @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
# from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.ControlMechanism import ControlMechanism_Base
# from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.EVCMechanism import EVCMechanism
# from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
# from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
# from PsyNeuLink.Components.Functions.Function import *
#
# My_Transfer_Mech_A = TransferMechanism(
#                            function=Logistic(
#                                         gain=(1.0, ControlSignal(modulation=ModulationParam.ADDITIVE))))
#
#
# My_Mech_A = TransferMechanism(function=Logistic)
# My_Mech_B = TransferMechanism(function=Linear,
#                              output_states=[RESULT, MEAN])
#
# Process_A = process(pathway=[My_Mech_A])
# Process_B = process(pathway=[My_Mech_B])
# My_System = system(processes=[Process_A, Process_B])
#
# My_EVC_Mechanism = EVCMechanism(system=My_System,
#                                 monitor_for_control=[My_Mech_A.output_states[RESULT],
#                                                      My_Mech_B.output_states[MEAN]],
#                                 control_signals=[(GAIN, My_Mech_A),
#                                                  {NAME: INTERCEPT,
#                                                   MECHANISM: My_Mech_B,
#                                                   MODULATION:ModulationParam.ADDITIVE}],
#                                 name='My EVC Mechanism')
#
#
# My_Mech_A = TransferMechanism(function=Logistic)
# My_Mech_B = TransferMechanism(function=Linear,
#                              output_states=[RESULT, MEAN])
# Process_A = process(pathway=[My_Mech_A])
# Process_B = process(pathway=[My_Mech_B])
#
# My_System = system(processes=[Process_A, Process_B],
#                                 monitor_for_control=[My_Mech_A.output_states[RESULT],
#                                                      My_Mech_B.output_states[MEAN]],
#                                 control_signals=[(GAIN, My_Mech_A),
#                                                  {NAME: INTERCEPT,
#                                                   MECHANISM: My_Mech_B,
#                                                   MODULATION:ModulationParam.ADDITIVE}],
#                    name='My Test System')

#endregion

#region TEST Learning @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ComparatorMechanism \
#     import ComparatorMechanism
# from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism.LearningMechanism \
#     import LearningMechanism
# from PsyNeuLink.Components.Functions.Function import *
# from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import *
#
# my_Mech_A = TransferMechanism(size=10)
# my_Mech_B = TransferMechanism(size=10)
# my_Mech_C = TransferMechanism(size=10)
# my_mapping_AB = MappingProjection(sender=my_Mech_A, receiver=my_Mech_B)
# my_mapping_AC = MappingProjection(sender=my_Mech_A, receiver=my_Mech_C)
#
# my_comparator = ComparatorMechanism(sample=my_Mech_B, target=TARGET,
#                                     # FIX: DOESN'T WORK WITHOUT EXPLICITY SPECIFIYING input_states, BUT SHOULD
#                                     input_states=[{NAME:SAMPLE,
#                                                    VARIABLE:my_Mech_B.output_state.value,
#                                                    WEIGHT:-1
#                                                    },
#                                                   {NAME:TARGET,
#                                                    VARIABLE:my_Mech_B.output_state.value,
#                                                    # WEIGHT:1
#                                                    }]
#                                     )
# my_learning = LearningMechanism(variable=[my_Mech_A.output_state.value,
#                                           my_Mech_B.output_state.value,
#                                           my_comparator.output_state.value],
#                                 error_source=my_comparator,
#                                 function=BackPropagation(default_variable=[my_Mech_A.output_state.value,
#                                                                            my_Mech_B.output_state.value,
#                                                                            my_Mech_B.output_state.value],
#                                                          activation_derivative_fct=my_Mech_A.function_object.derivative,
#                                                          error_derivative_fct=my_Mech_A.function_object.derivative,
#                                                          error_matrix=my_mapping_AB.matrix),
#                                 learning_signals=[my_mapping_AB, my_mapping_AC])
#
# TEST = True

#endregion

# region TEST ASSIGNMENT OF PROJECTION TO PARAMETER @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Decision = DDM(function=BogaczEtAl(drift_rate=ControlProjection),
#                name='Decision')
#
# Decision.execute()


# for i, j in zip(range(5), range(5)):
#     print(i, j)
#     j = 3
#     print(j)

# # ORIGINAL:
# # transfer_mechanism_1 = TransferMechanism()
# # # transfer_mechanism_1 = TransferMechanism(noise=(0.1, ControlProjection))
# # # TM1_parameter_state = ParameterState(value=22)
# # transfer_mechanism_2 = TransferMechanism()
# # # transfer_mechanism_3 = TransferMechanism()
# # transfer_mechanism_3 = TransferMechanism(function=Linear(slope=3))
# #
# # # my_process = process(pathway=[transfer_mechanism_1,
# # #                               (transfer_mechanism_2,{PARAMETER_STATE_PARAMS:{SLOPE:(1.0,
# # #                                                                                     Modulation.OVERRIDE)}}),
# # #                               transfer_mechanism_2])
# # # my_process.run(inputs=[[[0]]])
# #
# # # mapping_1 = MappingProjection(sender=transfer_mechanism_1, receiver=transfer_mechanism_3)
# # # mapping_2 = MappingProjection(sender=transfer_mechanism_2, receiver=transfer_mechanism_3)
# # print(transfer_mechanism_3.execute(input=1.0,
# #                                    runtime_params={PARAMETER_STATE_PARAMS:{SLOPE:(2.0, Modulation.OVERRIDE)}}))
# #
#
# my_control = ControlProjection(name='MY CONTROL')
#
# transfer_mechanism_Y = TransferMechanism(function=lambda x, y: x + y,
#                                          name='MY_TRANSFER_MECH_Y'
#                                          )
#
# transfer_mechanism_Y.exeucte([2, 3])
#
# transfer_mechanism_X = TransferMechanism(function=Logistic(bias=0,
#                                                            gain=ControlProjection()),
#                                          # noise=(0.3, CONTROL_PROJECTION),
#                                          noise=ControlProjection,
#                                          # noise='MY CONTROL',
#                                          rate=(0.1, ControlProjection),
#                                          params={OUTPUT_STATES:10.0},
#                                          # params={OUTPUT_STATES:['JDC OUTPUT STATE',
#                                          #                        {NAME:TRANSFER_MEAN,
#                                          #                         CALCULATE:lambda x: np.mean(x)}]},
#                                          name='MY_TRANSFER_MECH_X'
#                                          )
#
# transfer_mechanism = TransferMechanism(function=Logistic(bias=(3, ControlProjection()),
#                                                          gain=CONTROL_PROJECTION
#                                                          ),
#                                        noise=(0.3, ControlProjection),
#                                        name='MY_TRANSFER_MECH'
#                                        )
#
#
# transfer_mechanism_1 = TransferMechanism(function=Linear(slope=(1, ControlProjection)))
# # transfer_mechanism_1 = TransferMechanism(noise=(0.1, ControlProjection))
# # TM1_parameter_state = ParameterState(value=22)
# transfer_mechanism_2 = TransferMechanism(function=Logistic(bias=(3, ControlProjection),
#                                                            gain=ControlProjection
#                                                            )
#                                          # noise=(3, ControlProjection)
#                                          )
# # transfer_mechanism_3 = TransferMechanism()
# transfer_mechanism_3 = TransferMechanism(function=Linear(slope=1))
#
# transfer_mechanism_1.execute()
# # my_process = process(pathway=[transfer_mechanism_1,
# #                               (transfer_mechanism_2,{PARAMETER_STATE_PARAMS:{SLOPE:(1.0,
# #                                                                                     Modulation.OVERRIDE)}}),
# #                               transfer_mechanism_2])
# # my_process.run(inputs=[[[0]]])
#
# # mapping_1 = MappingProjection(sender=transfer_mechanism_1, receiver=transfer_mechanism_3)
# # mapping_2 = MappingProjection(sender=transfer_mechanism_2, receiver=transfer_mechanism_3)
# transfer_mechanism_3.function_object.runtimeParamStickyAssignmentPref = False
# print(transfer_mechanism_3.execute(input=1.0,
#                                    runtime_params={PARAMETER_STATE_PARAMS:{SLOPE:(6.0, Modulation.OVERRIDE)}}))
# # print(transfer_mechanism_3.execute(input=1.0))
# print(transfer_mechanism_3.execute(input=1.0,
#                                    runtime_params={PARAMETER_STATE_PARAMS:{INTERCEPT:(100.0,
#                                                                                    Modulation.OVERRIDE),
#                                                                             # SLOPE:(6.0,
#                                                                             #        Modulation.OVERRIDE
#                                                                                       }}))
# # print(transfer_mechanism_3.run(inputs=[1.0],
# #                                num_trials=3))
#
# my_process = process(pathway=[transfer_mechanism_1,
#                                # {PARAMETER_STATE_PARAMS:{SLOPE:2}}),
#                               transfer_mechanism_3])
#
# print("My Process: \n", my_process.run(inputs=[[1.0]],
#                                        num_trials=3))
# # print("My Process: \n", my_process.execute(input=[[1.0]]))
# # print("My Process: \n", my_process.execute(input=[1.0]))
#
# # transfer_mechanism_1.assign_params(request_set={FUNCTION: Logistic(gain=10)})
#
#
#
#
#
# # transfer_process = process(pathway = [transfer_mechanism_1])
# # print(transfer_process.execute())
# print ('Done')
#
# # my_mech1 = TransferMechanism(function=Logistic)
# # my_mech2 = TransferMechanism(function=Logistic)
# # my_monitor = ComparatorMechanism()
# # my_LEARNING_PROJECTION = LearningProjection()
# # my_mapping_projection = MappingProjection(sender=my_mech1, receiver=my_mech2)
# # # my_LEARNING_PROJECTION = LearningProjection(sender=my_monitor, receiver=my_mapping_projection)
# # # my_LEARNING_PROJECTION = LearningProjection(receiver=my_mapping_projection)
# # my_LEARNING_PROJECTION._deferred_init(context="TEST")
#
# # my_DDM = DDM(function=BogaczEtAl(drift_rate=2.0,
# #                                  threshold=20.0),
# #              params={FUNCTION_PARAMS:{DRIFT_RATE:3.0,
# #                                       THRESHOLD:30.0}}
# #              )
# # # my_DDM.execute(time_scale=TimeScale.TIME_STEP)
# # my_DDM.execute()
# #
# # TEST = True
#
# # my_adaptive_integrator = IntegratorMechanism(default_variable=[0],
# #                                                      function=Integrator(
# #                                                                          # default_variable=[0,0],
# #                                                                          weighting=SIMPLE,
# #                                                                          rate=[1]
# #                                                                          )
# #                                                      )
# # print(my_adaptive_integrator.execute([1]))
# # print(my_adaptive_integrator.execute([1]))
# # print(my_adaptive_integrator.execute([1]))
# # print(my_adaptive_integrator.execute([3]))
# # print(my_adaptive_integrator.execute([3]))
# # print(my_adaptive_integrator.execute([3]))
# # print(my_adaptive_integrator.execute([3]))
#endregion

#region TEST INSTANTATION OF System() @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from Components.Mechanisms.IntegratorMechanism import IntegratorMechanism
# from Components.Function import Integrator
#
# a = IntegratorMechanism([[0],[0]], params={FUNCTION_PARAMS:{Integrator.RATE:0.1}})
#
# init = [0,0,0]
# stim = [1,1,1]
#
# old = init
# new = stim
#
# for i in range(100):
#     old = a.execute([old,new])
#     print (old)
#
# print (a.execute([,[0, 2, 0][1, 1, 1]]))
#endregion

#region TEST INSTANTATION OF System() @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# from PsyNeuLink.Components.System import System_Base
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
# from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
#
# mech = DDM()
#
# mcs = ControlProjection(receiver=mech)
#
#
# mech.execute([0])
#
# a = System_Base()
# a.execute()
#
#endregion

# #region TEST SYSTEM (test_system) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# print("TEST SYSTEM test_system")
#
# a = TransferMechanism(name='a', default_variable=[0, 0])
# b = TransferMechanism(name='b')
# c = TransferMechanism(name='c')
# d = TransferMechanism(name='d')
#
# p1 = process(pathway=[a, b, c], name='p1')
# p2 = process(pathway=[a, b, d], name='p2')
#
# s = system(
#     processes=[p1, p2],
#     name='Branch System',
#     initial_values={a: [1, 1]},
# )
#
# inputs = {a: [2, 2]}
# s.run(inputs)
# #endregion

# #region TEST MULTIPLE LEARNING SEQUENCES IN A PROCESS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# print("TEST MULTIPLE LEARNING SEQUENCES IN A PROCESS")
#
# a = TransferMechanism(name='a', default_variable=[0, 0])
# b = TransferMechanism(name='b')
# c = TransferMechanism(name='c')
# d = TransferMechanism(name='d')
#
# p1 = process(pathway=[a,
#                       # MappingProjection(matrix=(RANDOM_CONNECTIVITY_MATRIX, LEARNING),
#                       #                   name="MP-1"),
#                       b,
#                       c,
#                       # MappingProjection(matrix=(RANDOM_CONNECTIVITY_MATRIX, LEARNING_PROJECTION),
#                       #                   name="MP-2"),
#                       d],
#              # learning=LEARNING,
#              name='p1')
#
# # s = system(
# #     processes=[p1],
# #     name='Double Learning System',
# #     # initial_values={a: [1, 1]},
# # )
#
# # inputs = {a: [2, 2]}
# # s.run(inputs)
# # s.show_graph(show_learning=True)
#
# inputs = {a: [2, 2]}
# TEST = p1.execute(input=[2,2])
# # p1.run(inputs)
# TEST=True
#
# #endregion

# #region TEST ControlMechanism and ObjectiveMechanism EXAMPLES @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# print("TEST ControlMechanism and ObjectiveMechanism EXAMPLES")

# my_transfer_mech_A = TransferMechanism()
# my_DDM = DDM()
# my_transfer_mech_B = TransferMechanism(function=Logistic)
#
# my_control_mech = ControlMechanism_Base(
#                          objective_mechanism=ObjectiveMechanism(monitored_output_states=[(my_transfer_mech_A, 2, 1),
#                                                                                   my_DDM.output_states[
#                                                                                       my_DDM.RESPONSE_TIME]],
#                                                                 function=LinearCombination(operation=SUM)),
#                          control_signals=[(THRESHOLD, my_DDM),
#                                           (GAIN, my_transfer_mech_B)])


# my_control_mech = ControlMechanism_Base(objective_mechanism=[(my_transfer_mech_A, 2, 1),
#                                                     my_DDM.output_states[my_DDM.RESPONSE_TIME]],
#                                function=LinearCombination(operation=SUM),
#                                control_signals=[(THRESHOLD, my_DDM),
#                                                 (GAIN, my_transfer_mech_2)])

# my_control_mech = ControlMechanism_Base(
#                         objective_mechanism=[(my_transfer_mech_A, 2, 1),
#                                              my_DDM.output_states[my_DDM.RESPONSE_TIME]],
#                         control_signals=[(THRESHOLD, my_DDM),
#                                          (GAIN, my_transfer_mech_B)])


# my_obj_mech=ObjectiveMechanism(monitored_output_states=[(my_transfer_mech_A, 2, 1),
#                                                  my_DDM.output_states[my_DDM.RESPONSE_TIME]],
#                                function=LinearCombination(operation=PRODUCT))
#
# my_control_mech = ControlMechanism_Base(
#                         objective_mechanism=my_obj_mech,
#                         control_signals=[(THRESHOLD, my_DDM),
#                                          (GAIN, my_transfer_mech_B)])

# # Mechanisms:
# Input = TransferMechanism(name='Input')
# Decision = DDM(function=BogaczEtAl(drift_rate=(1.0, CONTROL),
#                                    threshold=(1.0, CONTROL),
#                                    noise=0.5,
#                                    starting_point=0,
#                                    t0=0.45),
#                output_states=[DECISION_VARIABLE,
#                               RESPONSE_TIME,
#                               PROBABILITY_UPPER_THRESHOLD],
#                name='Decision')
# Reward = TransferMechanism(output_states=[RESULT, MEAN, VARIANCE],
#                            name='Reward')
#
# # Processes:
# TaskExecutionProcess = process(
#     default_variable=[0],
#     pathway=[Input, IDENTITY_MATRIX, Decision],
#     name = 'TaskExecutionProcess')
# RewardProcess = process(
#     default_variable=[0],
#     pathway=[Reward],
#     name = 'RewardProcess')
#
# # System:
# mySystem = system(processes=[TaskExecutionProcess, RewardProcess],
#                   controller=EVCMechanism(objective_mechanism=ObjectiveMechanism(monitored_output_states=[
#                                                      Reward,
#                                                      Decision.output_states[Decision.PROBABILITY_UPPER_THRESHOLD],
#                                                      (Decision.output_states[Decision.RESPONSE_TIME], -1, 1)])))
#
# TEST = True
# endregion


#region TEST INPUT FORMATS

# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
# from PsyNeuLink.Components.States.InputState import InputState
#
# x = TransferMechanism([0,0,0],
#              name='x')
#
# i = InputState(owner=x, reference_value=[2,2,2], value=[1,1,1])
#
# y = TransferMechanism(default_variable=[0],
#              params={INPUT_STATES:i},
#              name='y')
#
# TEST = True
#
# print(y.run([1,2,3]))

#endregion

#region TEST INPUT FORMATS

# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
# from PsyNeuLink.Components.Process import process
# from PsyNeuLink.Components.System import system
#
#
# # UNEQUAL INPUT LENGTHS:
# inputs=[[[2,2],0],[[2,2],0]]
# # inputs=[[2,2],[0]]
# # inputs=[[[2,2],0],[[2,2],0]]
# # inputs=[[[2,2],[0]],[[2,2],[0]]]
# # inputs=[[[[2,2],[0]]],[[[2,2],[0]]]]
#
# a = TransferMechanism(name='a',default_variable=[0,0])
# b = TransferMechanism(name='b')
# c = TransferMechanism(name='c')
#
#
# print(a.execute([2,2]))
#
#
# p1 = process(pathway=[a, c], name='p1')
# p2 = process(pathway=[b, c], name='p2')
#
# s = system(processes=[p1, p2],
#            name='Convergent System')
#
# def show_trial_header():
#     print("\n############################ TRIAL {} ############################".format(CentralClock.trial))
#
# print(a.run(inputs=[[0,0],[1,1],[2,2]],
#       call_before_execution=show_trial_header))
#

# s.run(inputs=inputs,
#       call_before_trial=show_trial_header)

#endregion

#region TEST INSTANTATION OF Cyclic and Acyclic Systems @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# from PsyNeuLink.Components.System import system
# from PsyNeuLink.Components.Process import process
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
# from PsyNeuLink.Components.Process import MappingProjection
#
# a = TransferMechanism(name='a')
# b = TransferMechanism(name='b')
# c = TransferMechanism(name='c')
# d = TransferMechanism(name='d')
# e = TransferMechanism(name='e')
#
# fb1 = MappingProjection(sender=c, receiver=b, name='fb1')
# fb2 = MappingProjection(sender=d, receiver=e, name = 'fb2')
#
# p1 = process(pathway=[a, b, c, d], name='p1')
# p2 = process(pathway=[e, b, c, d], name='p2')
#
# a = system(processes=[p1, p2], name='systsem')
#
# a.show()
#
# a.execute()

# endregion

#region TEST MECHANISM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# from Components.Mechanisms.Mechanism import Mechanism, mechanism
# from Components.Mechanisms.DDM import DDM

# x = Mechanism(context=kwValidate)
# test = isinstance(x,Mechanism)
# temp = True
#
#endregion

#region TEST PROCESS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# #
# from Components.Process import *
# # from Components.Mechanisms.DDM import DDM
# from Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
#
# my_transfer = TransferMechanism()
#
# x = Process_Base(params={PATHWAY:[my_transfer]})
#
# for i in range(100):
#     x.execute([1])
#
# endregion

#region TEST LinearCombination FUNCTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Functions.Function import LinearCombination

# x = LinearCombination()
# # print (x.execute(([1, 1],[2, 2])))
#
# print (x.execute(([[1, 1],[2, 2]],
#                   [[3, 3],[4, 4]])))

# print (x.execute(([[[[1, 1],[2, 2]]]])))


#endregion

#region TEST UtilityIntegrator FUNCTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

from PsyNeuLink.Components.Functions.Function import UtilityIntegrator
print("TEST UtilityIntegrator FUNCTION")

x = UtilityIntegrator(long_term_rate=.1)
x.show_params()
print (x.execute([[1]]*10))

#endregion

#region TEST COMBINE_MEANS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# import numpy as np
# from PsyNeuLink.Globals.Utilities import is_numeric
# print("TEST CombineMeans Function")
#
#
# x = np.array([[10, 20], [10, 20]])
# # y = np.array([[10, 'a'], ['a']])
# # z = np.array([[10, 'a'], [10]])
# # print(is_numeric(x))
# # print(is_numeric(y))
# # print(is_numeric(z))
#
# z = CombineMeans(x, context='TEST')
# print (z.execute(x))
#
#endregion

#region TEST RL @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Functions.Function import *
#
# rl = Reinforcement([[0,0,0], [0,0,0], [0]])
# print(rl.execute([[0,0,0], [0, 0, 1], [7]]))
#

#endregion

#region TEST Backprop FUNCTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from Components.Function import *
#
# x = BackPropagation()
# print (x.execute(variable=[[1, 2],[0.5, 0],[5, 6]]))
#
# # y = lambda input,output: output*(np.ones_like(output)-output)
# # print (y(2, [0.25, 0.5]))
#
#
#endregion

# region TEST RecurrentTransferMechanism / LCA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# print("TEST RecurrentTransferMechanism / LCA")
#
# my_auto = LCA(
#         size=3,
#         output_states=[LCA_OUTPUT.RESULT,
#                        LCA_OUTPUT.ENTROPY,
#                        LCA_OUTPUT.ENERGY,
#                        LCA_OUTPUT.MAX_VS_AVG]
#         # inhibition
# )
#
# # my_auto = RecurrentTransferMechanism(
# #         default_variable=[0,0,0],
# #         size=3,
# #         function=Logistic,
# #         # matrix=RANDOM_CONNECTIVITY_MATRIX,
# #         matrix=np.array([[1,1,1],[1,1,1],[1,1,1]])
# #         # matrix=[[1,1,1],[1,1,1],[1,1,1]]
# # )
#
# # my_auto = TransferMechanism(default_variable=[0,0,0],
# #                             # function=Logistic
# #                             )
# #
# # my_auto_matrix = MappingProjection(sender=my_auto,
# #                                    receiver=my_auto,
# #                                    matrix=FULL_CONNECTIVITY_MATRIX)
#
# # THIS DOESN'T WORK, AS Process._instantiate_pathway() EXITS AFTER PROCESSING THE LONE MECHANISM
# #                    SO NEVER HAS A CHANCE TO SEE THE PROJECTION AND THEREBY ASSIGN IT A LearningProjection
# my_process = process(pathway=[my_auto],
#
# # THIS DOESN'T WORK, AS Process._instantiate_pathway() ONLY CHECKS PROJECTIONS AFTER ENCOUNTERING ANOTHER MECHANISM
# # my_process = process(pathway=[my_auto, my_auto_matrix],
#                      target=[0,0,0],
#                      learning=LEARNING
#                      )
#
# # my_process = process(pathway=[my_auto, FULL_CONNECTIVITY_MATRIX, my_auto],
# #                      learning=LEARNING,
# #                      target=[0,0,0])
#
# # print(my_process.execute([1,1,1]))
# # print(my_process.execute([1,1,1]))
# # print(my_process.execute([1,1,1]))
# # print(my_process.execute([1,1,1]))
# #
# input_list = {my_auto:[1,1,1]}
# target_list = {my_auto:[0,0,0]}
#
# # print(my_process.run(inputs=input_list, targets=target_list, num_trials=5))
#
# my_system = system(processes=[my_process],
#                    targets=[0,0,0])
#
# print(my_system.run(inputs=input_list,
#                     targets=target_list,
#                     num_trials=5))

#endregion

#region TEST BogaczEtAl Derivative @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Functions.Function import *
# #
# x = BogaczEtAl()
# print(x.function(params={DRIFT_RATE:1.0,
#                          THRESHOLD:1}))
# print(x.derivative())

#endregion

#region TEST SoftMax FUNCTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Functions.Function import *
# #
# x = SoftMax()
# # x = SoftMax(output=MAX_VAL)
# a = [-1, 2, 1]
# # x = SoftMax(output=SoftMax.PROB)
# y = x.function(a)
# z = x.derivative(a)
# print ("SoftMax execute return value: \n", [float(i) for i in y])
# if z.ndim == 1:
#     print ("SoftMax derivative return value: \n", [float(i) for i in z])
# else:
#     print ("SoftMax derivative return value: \n", [[float(i) for i in j] for j in z])

#endregion

#region TEST ReportOUtput Pref @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Process import *
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
# from PsyNeuLink.Components.Functions.Function import Linear
#
# my_mech = TransferMechanism(function=Linear())
#
# my_process = process(pathway=[my_mech])
#
# my_mech.reportOutputPref = False
#
# # FIX: CAN'T CHANGE reportOutputPref FOR PROCESS USE LOCAL SETTER (DEFAULT WORKS)
# my_process.reportOutputPref = False
# my_process.verbosePref = False
#
# my_process.execute()

#endregion

#region TEST Matrix Assignment to MappingProjection @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Process import *
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
# from PsyNeuLink.Components.Functions.Function import Linear
# from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
#
# my_mech = TransferMechanism(function=Linear())
# my_mech2 = TransferMechanism(function=Linear())
# my_projection = MappingProjection(sender=my_mech,
#                         receiver=my_mech2,
#                         matrix=np.ones((1,1)))
#
# my_process = process(pathway=[my_mech, my_mech2])
#
#
# my_process.execute()

#endregion

#region TEST matrix @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Input_Weights_matrix = (np.arange(2*5).reshape((2, 5)) + 1)/(2*5)
# Middle_Weights_matrix = (np.arange(5*4).reshape((5, 4)) + 1)/(5*4)
# Output_Weights_matrix = (np.arange(4*3).reshape((4, 3)) + 1)/(4*3)
#
# print ("Input Weights:\n",Input_Weights_matrix)
# print ("Middle Weights:\n",Middle_Weights_matrix)
# print ("Output Weights:\n",Output_Weights_matrix)


# a = np.array([-0.8344837,  -0.87072018,  0.10002567])
# b = (np.arange(4*3).reshape((4, 3)) + 1)/(4*3)
# c = np.dot(b, a, )
# print(c)

#endregion  ********

#region TEST Stability and Distance @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# matrix = [[0,-1],[-1,0]]
# normalize = False
# activity = [100,0]
#
#
# eng = Stability(default_variable=activity,
#              matrix=matrix,
#              normalize=normalize
#              )
#
# dist = Distance(default_variable=[activity,activity],
#                 metric=CROSS_ENTROPY,
#                 # normalize=normalize
#                 )
#
# print("Stability: ",eng.function(activity))
# print("Distance: ", dist.function(activity))

#endregion

#region TEST Matrix Assignment to MappingProjection @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# from PsyNeuLink.Components.Process import *
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
# from PsyNeuLink.Components.Functions.Function import Linear, Logistic
# from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
#
# color_naming = TransferMechanism(default_variable=[0,0],
#                         function=Linear,
#                         name="Color Naming"
#                         )
#
# word_reading = TransferMechanism(default_variable=[0,0],
#                         function=Logistic,
#                         name="Word Reading")
#
# verbal_response = TransferMechanism(default_variable=[0,0],
#                            function=Logistic)
#
# color_pathway = MappingProjection(sender=color_naming,
#                         receiver=verbal_response,
#                         matrix=IDENTITY_MATRIX,
#                         )
#
# word_pathway = MappingProjection(sender=word_reading,
#                        receiver=verbal_response,
#                         matrix=IDENTITY_MATRIX
#                        )
#
# Stroop_process = process(default_variable=[[1,2.5]],
#                          pathway=[color_naming, word_reading, verbal_response])
#
#
# Stroop_process.execute()
#
# endregion

# ----------------------------------------------- UTILITIES ------------------------------------------------------------

#region TEST typecheck: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# @tc.typecheck
# def foo2(record:(int,int,bool), rgb:tc.re("^[rgb]$")) -> tc.any(int,float) :
#     # don't expect the following to make much sense:
#     a = record[0]; b = record[1]
#     return a/b if (a/b == float(a)/b) else float(a)/b
#
# # foo2((4,10,True), "r")   # OK
# # foo2([4,10,True], "g")   # OK: list is acceptable in place of tuple
# # foo2((4,10,1), "rg")     # Wrong: 1 is not a bool, string is too long
# # # foo2(None,     "R")      # Wrong: None is no tuple, string has illegal character
#
#
# from enum import Enum
# # class Weightings(AutoNumber):
# class Weightings(Enum):
#     CONSTANT        = 'hello'
#     SIMPLE        = 'goodbye'
#     ADAPTIVE = 'you say'
#
# @tc.typecheck
# def foo3(test:tc.re('hello')):
#     a = test
#
# foo3('hello')
# # foo3('goodbye')
# # foo3(test=3)
#
# @tc.typecheck
# def foo4(test:Weightings=Weightings.SIMPLE):
#     a = test
#
# # foo4(test=Weightings.LINEAR)
# foo4(test='LINEAR')

# @tc.typecheck
# def foo5(test:tc.any(int, float)=2):
#     a = test
#
# foo5(test=1)

# options = ['Happy', 'Sad']

# @tc.typecheck
# def foo6(arg:tc.enum('Happy', 'Sad')):
#     a = arg
#
# foo6(arg='Ugh')

# @tc.typecheck
# # def foo7(arg:tc.optional(tc.any(int, float, tc.seq_of(tc.any(int, float))))):
# def foo7(arg:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))):
#     a = arg
#
# foo7(np.array([1,'a']))
#

# a = NotImplemented
# if isinstance(a, type(NotImplemented)):
#     print ("TRUE")

#endregion

#region TEST get_user_attributes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# import inspect
#
# def get_class_attributes(cls):
#     boring = dir(type('dummy', (object,), {}))
#     return [item
#             for item in inspect.getmembers(cls)
#             if item[0] not in boring]
#
# class my_class():
#     attrib1 = 0
#     attrib2 = 1
#     # def __init__(self):
#         # self.attrib1 = 0
#         # self.attrib2 = 1
#
# print(get_class_attributes(my_class))

#endregion

#region TEST Function definition in class: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class a:
#     def __init__(self):
#         a.attrib1 = True
#
#     def class_function(string):
#         return 'RETURN: ' + string
#
# print (a.class_function('hello'))
#
#endregion

#region TEST Save function args: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# def first_function(sender=NotImplemented,
#                   receiver=NotImplemented,
#                   params=NotImplemented,
#                   name=NotImplemented,
#                   prefs=NotImplemented,
#                   context=None):
#     saved_args = locals()
#     return saved_args
#
# def second_function(sender=NotImplemented,
#                   receiver=NotImplemented,
#                   params=NotImplemented,
#                   name=NotImplemented,
#                   prefs=NotImplemented,
#                   context=None):
#     saved_args = locals()
#     return saved_args
#
# a = first_function(sender='something')
# print ('a: ', a)
# a['context']='new context'
# print ('a: ', a)
# b = second_function(**a)
# print ('b: ', b)
#
#

#endregion

#region TEST Attribute assignment: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class a:
#     def __init__(self):
#         a.attrib1 = True
#
# x = a()
# print ('attrib1: ', x.attrib1)
# x.attrib2 = False
# print ('attrib2: ', x.attrib2)

#endregion

#region TEST np.array ASSIGNMENT: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# test = np.array([[0]])
# print (test)
# test[0] = np.array([5])
# print (test)

#endregion

#region TEST next: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class a:
#     pass
# b = a()
# c = a()
# l = ['hello', 1, b, 2, 'test', 3, c, 4, 'goodbye']
#
# x = [item for item in l if isinstance(item, a)]
# print (x)


# x = iter(l)
# # x = iter(['test', 'goodbye'])
# i = 0
# # while next((s for s in x if isinstance(s, int)), None):
#
# y = []
# z = next((s for s in x if isinstance(s, a)), None)
# while z:
#     y.append(z)
#     z = next((s for s in x if isinstance(s, a)), None)
#
# print (y)
# print (len(y))


# print (next((s for s in x if isinstance(s, int)), None))
# print (next((s for s in x if isinstance(s, int)), None))
# print (next((s for s in x if isinstance(s, int)), None))
# print (next((s for s in x if isinstance(s, int)), None))

#endregion

#region TEST BREAK IN FOR LOOP: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# i = 0
# for i in range(10):
#     if i == 2:
#         break
# print (i)

#endregion

#region TEST np.array DOT PRODUCT: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # output_error = np.array([3, 1])
# # weight_matrix = np.array([[1, 2], [3, 4], [5, 6]])
#
# # sender_error = 5, 13, 21
#
# # receivers = np.array([[1, 2]]).reshape(2,1)
# receivers = np.array([3,1])
# weights = np.array([[1, 2], [3, 4], [5, 6]])
# print ('receivers: \n', receivers)
# print ('weights: \n', weights)
# print ('dot product: \n', np.dot(weights, receivers))

#endregion

#region TEST PRINT W/O RETURN @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # for item in [1,2,3,4]:
# #     print(item, " ", end="")
# #
# # print("HELLO", "GOOBAH", end="")
#
#
# print("HELLO ", end="")
# print("GOOBAH", end="")
# print(" AND FINALLY")
#
#endregion

#region TEST PHASE_SPEC @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def phaseSpecFunc(freq_spec, phase_spec, phase_max):
#     for time in range(20):
#         if (time % (phase_max + 1)) == phase_spec:
#             print (time, ": FIRED")
#         else:
#             print (time, ": -----")
#
# phaseSpecFunc(freq_spec=1,
#               phase_spec=1,
#               phase_max=3)

#endregion

#region TEST CUSTOM LIST THAT GETS ITEM FROM ANOTHER LIST @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from collections import UserList
#
# # class mech_list(UserList):
# #     def __init__(self):
# #         super(mech_list, self).__init__()
# #         self.mech_tuples = mech_tuples()
# #
# #     def __getitem__(self, item):
# #         return self.mech_tuples.tuples_list[item][0]
# #
# #     def __setitem__(self, key, value):
# #         raise ("MyList is read only ")
# #
# # class mech_tuples:
# #     def __init__(self):
# #         # self.tuples_list = myList()
# #         self.tuples_list = [('mech 1', 1), ('mech 2', 2)]
#
# # x = mech_list(mech_tuples())
# # print (x[0])
#
# class mech_list(UserList):
#     def __init__(self, source_list):
#         super(mech_list, self).__init__()
#         self.mech_tuples = source_list
#
#     def __getitem__(self, item):
#         return self.mech_tuples.tuples_list[item][0]
#
#     def __setitem__(self, key, value):
#         raise ("MyList is read only ")
#
# class system:
#     def __init__(self):
#         self.tuples_list = [('mech 1', 1), ('mech 2', 2)]
#         self.mech_list = mech_list(self)
#
# x = system()
# print (x.mech_list[1])
#
#endregion

#region TEST ERROR HANDLING: NESTED EXCEPTIONS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#
# MonitoredOutputStatesOption = dict
# target_set = {
#     MONITOR_FOR_CONTROL:'state that is monitored',
#     # FUNCTION_PARAMS:{WEIGHTS:[1]}
#               }
#
# try:
#     # It IS a MonitoredOutputStatesOption specification
#     if isinstance(target_set[MONITOR_FOR_CONTROL], MonitoredOutputStatesOption):
#         # Put in a list (standard format for processing by _instantiate_monitored_output_states)
#         # target_set[MONITOR_FOR_CONTROL] = [target_set[MONITOR_FOR_CONTROL]]
#         print ("Assign monitored States")
#     # It is NOT a MonitoredOutputStatesOption specification, so assume it is a list of Mechanisms or States
#     else:
#         # for item in target_set[MONITOR_FOR_CONTROL]:
#         #     self._validate_monitored_state_in_system(item, context=context)
#         # Insure that number of weights specified in WEIGHTS functionParams equals the number of monitored states
#         print ('Validated monitored states')
#         try:
#             num_weights = len(target_set[FUNCTION_PARAMS][WEIGHTS])
#         except KeyError:
#             # raise ScratchPadError('Key error for assigning weights')
#             pass
#         else:
#             # num_monitored_states = len(target_set[MONITOR_FOR_CONTROL])
#             # if not True:
#             if True:
#                 raise ScratchPadError("Weights not equal")
# except KeyError:
#     pass

#endregion

#region TEST ERROR HANDLING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#

# myMatrix = np.matrix('1 2 3; 4 5 q')

# try:
#     myMatrix = np.matrix('1 2 3; 4 5 6')
# except (TypeError, ValueError) as e:
#     print ("Error message: {0}".format(e))

# try:
#     myMatrix = np.atleast_2d(['a', 'b'], ['c'])
# except TypeError as e:
#     print ("Array Error message: {0}".format(e))

# try:
#     myMatrix = np.matrix([[1, 2, 3], ['a', 'b', 'c']])
# except TypeError as e:
#     print ("Matrix Error message: {0}".format(e))
#
#
# print ("\nmyMatrix: \n", myMatrix)
#
#endregion

#region TEST ERROR HANDLING: INSTANTIATING A CUSTOM EXCEPTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# try:
#     raise TypeError('help')
# except:
#     print ("Exeption raised!")

#endregion

#region TEST FIND TERMINALS IN GRAPH @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#          A
#         /
#       B
#      / \
#    C    D
#   /
# E

# graph = {"B": {"A"},
#          "C": {"B"},
#          "D": {"B"},
#          "E": {"C"},
#          "A": set()}
#
# B    C
#  \  /
#   A

# graph = {
#     "A": {"B", "C"},
#     "B": set(),
#     "C": set()
# }

# receiver_mechs = set(list(graph.keys()))
#
# print ("receiver_mechs: ", receiver_mechs)
#
# sender_mechs = set()
# for receiver, sender in graph.items():
#     sender = graph[receiver]
#     sender_mechs = sender_mechs.union(sender)
#
# print ("sender_mechs: ", sender_mechs)
#
# terminal_mechs = receiver_mechs-sender_mechs
#
# print ('terminal_mechs: ', terminal_mechs )

# p2 = process(pathway=[e, c, b, d], name='p2')
# p1e = process(pathway=[a, b, c, d], name='p1e')

# graph = {"B": {"A"},
#          "C": {"B"},
#          "D": {"B"},
#          "D": {"C"},
#          "E": set(),
#          "A": set()}

# p1e: [a, b, c, d]
# p2:  [e, c, f, b, d]

# graph = {"B": {"A"},
#          "C": {"B"},
#          "D": {"B"},
#          "B": {"D"},
#          "A": set()}
#

# graph = {"B": {"A", "X"},
#                 "C": {"B", "Y"},
#                 "D": {"B"},
#                 "E": {"C"}}
#

# from toposort import toposort, toposort_flatten
#
# print("\nList of sets from toposort: ", list(toposort(graph))) # list of sets
# print("toposort_flatten (not sorted): ", toposort_flatten(graph, sort=False)) # a particular order
# print("toposort_flatten (sorted): ", toposort_flatten(graph, sort=True)) # a particular order

# from itertools import chain
# # graph ={'B': {'A', 'F'}, 'C': {'B'}, 'D': {'B'}, 'E': {'C'}}
# terminals = [k for k in graph.keys() if k not in chain(*graph.values())]
# print ("\nterminals: ", terminals)


#endregion

#region TEST TOPOSORT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# # from toposort import toposort, toposort_flatten
# # # #
# # # graph = {"C": {"B","D"},  # Note: ignores neste' sets
# # #         "C": { "A"},
# # #         "C": {"C''"},
# # #         "B": {"A'"},
# # #         "B":{"A"},
# # #         "C''":{"A'"}, # ADDED
# # #         "A":set(),
# # #         "A'":set(),
# # #         "C''":{"B''"},
# # #         "B''":{"A''"},
# # #         "A''":set(),
# # #         "D": { "B"}
# # #          }
# # #         # "D":set()}
# # #
# # #
# # #          E
# # #         /
# # #    D   C
# # #     \ / \
# # #      B   Y
# # #     / \
# # #    A   X
# # #
# # graph = {"B": {"A", "X"},
# #          "C": {"B", "Y"},
# #          "D": {"B"},
# #          "E": {"C"}}
# # #
# import re
# print()
# print( list(toposort(graph))) # list of sets
# print(toposort_flatten(graph)) # a particular order
# # print( re.sub('[\"]','',str(list(toposort(graph))))) # list of sets
# # print( re.sub('[\"]','',str(toposort_flatten(graph)))) # a particular order

#
# OUTPUT:
# [{A, A', A''}, {B'', B}, {C''}, {C}]
# [A, A', A'', B, B'', C'', C]

# #endregion

#region TEST **kwARG PASSING  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def function(arg1=1, arg2=2, **kwargs):
#     print ("arg 1: {}\narg 2: {}\nkwargs: {}\ntype of kwargs: {}".format(arg1, arg2, kwargs, type(kwargs)))
#
# function(**{'arg1':3, 'arg2':4})
#
# arg_dict = {'arg1':5, 'arg2':6, 'arg3':7}
# function(**arg_dict)

# def function(arg1=1, arg2=2):
#     print ("\targ 1: {}\n\targ 2: {}".format(arg1, arg2))
#
# print("\nArgs passed as **{'arg1':5, 'arg2':6}:")
# function(**{'arg1':5, 'arg2':6})
#
# print("\nArgs passed as *(7, 8):")
# function(*(7, 8))
#
# print("\nArgs passed as **{kwArg1:9, kwArg2:10}:")
# kwArg1 = 'arg1'
# kwArg2 = 'arg2'
# function(**{kwArg1:9, kwArg2:10})

#endregion

#region TEST @PROPERTY APPEND FOR SETTER @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class attribute_list(list):
#     def append(self, value):
#         print ('ACCESSED ATTRIBUTE APPEND')
#         super(attribute_list, self).append(value)
#
# class a:
#     def __init__(self):
#         self._attribute = attribute_list()
#         self._attribute.append('happy')
#
#     @property
#     def attribute(self):
#         return self._attribute
#
#     @attribute.setter
#     def attribute(self, value):
#         print ('ACCESSED SETTER')
#         self._attribute.append(value)
#
#     def add(self, value):
#         self.attribute = value
#
#
# x = a()
# # items = ['happy', 'sad']
# # for i in items:
# #     x.attribute = i
#
# x.attribute.append('sad')
# print(x.attribute)
#

#endregion

#region TEST HIERARCHICAL property -- WORKS! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class a:
#     def __init__(self):
#         self._attribute = None
#
#     # @property
#     # def attribute(self):
#     def getattribute(self):
#         print ('RETRIEVING PARENT attribute')
#         return self._attribute
#
#     # @attribute.setter
#     # def attribute(self, value):
#     def setattribute(self, value):
#         print ('SETTING PARENT TO: ', value)
#         self._attribute = value
#         print ('PARENT SET TO: ', self._attribute)
#
#     attribute = property(getattribute, setattribute, "I'm the attribute property")
#
# class b(a):
#     # def __init__(self):
#     #     self.attrib = 1
#     #     self._attribute = None
#     # @property
#     # def attribute(self):
#     def getattribute(self):
#         print ('RETRIEVING CHILD attribute')
#         return super(b, self).getattribute()
#
#     # @attribute.setter
#     # def attribute(self, value):
#     def setattribute(self, value):
#         # super(b, self).attribute(value)
#         # super(b,self).__set__(value)
#         super(b,self).setattribute(value)
#         print ('SET CHILD TO: ', self._attribute)
#         # self._attribute = value
#
#     attribute = property(getattribute, setattribute, "I'm the attribute property")
#
# x = a()
# x.attribute = 1
# x.attribute = 1
# print (x.attribute)
# y = b()
# y.attribute = 2
# print (y.attribute)


#endregion

#region TEST HIERARCHICAL property -- FROM BRYN @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class A(object):
#     def __init__(self):
#         self._foo = 1
#     @property
#     def foo(self):
#         return self._foo
#
#     @foo.setter
#     def foo(self, value):
#         self._foo = value
#
#
# class B(A):
#     @A.foo.setter
#     def foo(self, value):
#         A.foo.__set__(self, value * 2)
#
#
# if __name__ == '__main__':
#     a = A()
#     b = B()
#     a.foo = 5
#     b.foo = 5
#     print("a is %d, b is %d" % (a.foo, b.foo))

#endregion

#region TEST setattr for @property @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# class a:
#     # def __init__(self):
#     #     self.attrib = 1
#     @property
#     def attribute(self):
#         return self._attribute
#
#     @attribute.setter
#     def attribute(self, value):
#         print ('SETTING')
#         self._attribute = value
#
# x = a()
# # x.attribute = 1
# setattr(x, 'attribute', 2)
# print (x.attribute)
# print (x._attribute)
#endregion

#region TEST setattr @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class a:
#     def __init__(self):
#         a.foo = 3
#
# x = a()
# setattr(x, 'foo', 4)
# print (x.foo)
# print (x.__dict__)

# #endregion

#region TEST ARITHMETIC @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# x = np.array([10,10])
# y = np.array([1,2])
# q = np.array([2,3])
#
# z = LinearCombination(x,
#                       param_defaults={LinearCombination.OPERATION: LinearCombination.Operation.PRODUCT},
#                       context='TEST')
# print (z.execute([x, y, q]))

# #endregion

#region TEST LINEAR @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# x = np.array([10,10])
# y = np.array([1,2])
# q = np.array([2,3])
#
# z = Linear(x, context='TEST')
# print (z.execute([x]))
#
# #endregion

#region TEST iscompatible @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# a = 1
# b = LogEntry.OUTPUT_VALUE
#
# # if iscompatible(a,b, **{kwCompatibidlityType:Enum}):
# if iscompatible(a,b):
#     print('COMPATIBLE')
# else:
#     print('INCOMPATIBLE')
#
# #endregion

#region TEST OVER-WRITING OF LOG @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#

# class a:
#
#     attrib = 1
#
#     class LogEntry(IntEnum):
#         NONE            = 0
#         TIME_STAMP      = 1 << 0
#
#     def __init__(self):
#         self.attrib = 2
#
#
# class b(a):
#
#     class LogEntry(IntEnum):
#         OUTPUT_VALUE    = 1 << 2
#         DEFAULTS = 3
#
#     def __init__(self):
#         self.pref = self.LogEntry.DEFAULTS
#
# x = a()
# y = b()
#
# z = b.LogEntry.OUTPUT_VALUE
# print (z)

#endregion

#region TEST SHARED TUPLE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# from collections import namedtuple
#
# TestTuple = namedtuple('TestTuple', 'first second')
#
# class a:
#     def __init__(self):
#         # self.tuple_a = TestTuple('hello','world')
#         self.tuple_a = 5
#
# x = a()
#
# class b:
#     def __init__(self):
#         self.tuple_a = x.tuple_a
#
# class c:
#     def __init__(self):
#         setattr(self, 'tuple_a', x.tuple_a)
#
# y=b()
# z=c()
# x.tuple_a = 6
#
# print (y.tuple_a)
# print (z.tuple_a)
#
#
#endregion

#region TEST PREFS GETTER @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# class prefs:
#     def __init__(self):
#         self.pref_attrib = 'PREF ATTRIB'
#
# class a:
#     def __init__(self):
#         self._prefs = prefs()
#
#     @property
#     def prefs(self):
#         print ("accessed")
#         return self._prefs
#
#endregion

#region TEST: Preferences @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# # x = DDM()
# # x.prefs.show()
#
# DDM_prefs = ComponentPreferenceSet(
#                 reportOutput_pref=PreferenceEntry(True,PreferenceLevel.SYSTEM),
#                 verbose_pref=PreferenceEntry(True,PreferenceLevel.SYSTEM),
#                 kpFunctionRuntimeParams_pref=PreferenceEntry(Modulation.MULTIPLY,PreferenceLevel.TYPE)
#                 )
# DDM_prefs.show()
# # DDM.classPreferences = DDM_prefs
# #
# # DDM_prefs.show()
# # print (DDM_prefs.verbosePref)
#

#endregion

#region TEST:  GET ATTRIBUTE LIST @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# class x:
#     def __init__(self):
#         self.attrib1 = 'hello'
#         self.attrib2 = 'world'
#
# a = x()
#
# print (a.__dict__.values())
#
# for item in a.__dict__.keys():
#     if 'attrib' in item:
#         print (item)
#
# for item, value in a.__dict__.items():
#     if 'attrib' in item:
#         print (value)

#endregion

#region TEST:  PROPERTY GETTER AND SETTER @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# ************
# 
# EXAMPLE:
# 
# class ClassProperty(property):
#     def __get__(self, cls, owner):
#         return self.fget.__get__(None, owner)()
# 
# class foo(object):
#     _var=5
#     def getvar(cls):
#         return cls._var
#     getvar=classmethod(getvar)
#     def setvar(cls,value):
#         cls._var=value
#     setvar=classmethod(setvar)
#     var=ClassProperty(getvar,setvar)
# 
# assert foo.getvar() == 5
# foo.setvar(4)
# assert foo.getvar() == 4
# assert foo.var == 4
# foo.var = 3
# assert foo.var == 3
# However, the setters don't actually work:
# 
# foo.var = 4
# assert foo.var == foo._var # raises AssertionError
# foo._var is unchanged, you've simply overwritten the property with a new value.
# 
# You can also use ClassProperty as a decorator:
# 
# class Foo(object):
#     _var = 5
# 
#     @ClassProperty
#     @classmethod
#     def var(cls):
#         return cls._var
# 
#     @var.setter
#     @classmethod
#     def var(cls, value):
#         cls._var = value
# 
# assert foo.var == 5
# 
# **************
# 
# BETTER EXAMPLE:

# class foo(object):
#     _var = 5
#     class __metaclass__(type):
#     	pass
#     @classmethod
#     def getvar(cls):
#     	return cls._var
#     @classmethod
#     def setvar(cls, value):
#     	cls._var = value
# 

# class foo(object):
#     _var = 5
#     class __metaclass__(type):
#     	@property
#     	def var(cls):
#     		return cls._var
#     	@var.setter
#     	def var(cls, value):
#     		cls._var = value



# class a:
#     _cAttrib = 5
#
#     def __init__(self):
#         self._iAttrib = 2
#         pass
#
#     @property
#     def iAttrib(self):
#         return self._iAttrib
#
#     @iAttrib.setter
#     def iAttrib(self, value):
#         print('iAttrib SET')
#         self._iAttrib = value
#
#     @property
#     def cAttrib(self):
#         return self._cAttrib
#
#     @cAttrib.setter
#     def cAttrib(self, value):
#         print('cAttrib SET')
#         self._cAttrib = value

# class classProperty(property):
#     def __get__(self, cls, owner):
#         return self.fget.__get__(None, owner)()

# class a(object):
#     _c_Attrib=5
    # def get_c_Attrib(cls):
    #     return cls.__c_Attrib
    # get_c_Attrib=classmethod(get_c_Attrib)
    # def set_c_Attrib(cls,value):
    #     cls.__c_Attrib=value
    # set_c_Attrib=classmethod(set_c_Attrib)
    # _c_Attrib=ClassProperty(get_c_Attrib, set_c_Attrib)

# test = 0
# class a(object):
# 
#     _c_Attrib=5
# 
#     @classProperty
#     @classmethod
#     def c_Attrib(cls):
#         test = 1
#         return cls._c_Attrib
# 
#     @c_Attrib.setter
#     @classmethod
#     def c_Attrib(cls, value):
#         test = 1
#         print ('Did something')
#         cls._c_Attrib = value
# 

# test = 0
# class a(object):
#     _c_Attrib = 5
#     class __metaclass__(type):
#         @property
#         def c_Attrib(cls):
#             return cls._c_Attrib
#         @c_Attrib.setter
#         def c_Attrib(cls, value):
#             pass
#             # cls._c_Attrib = value
# 

# test = 0
# class a(object):
#     _c_Attrib = 5
#     class __metaclass__(type):
#         pass
#     @classmethod
#     def getc_Attrib(cls):
#         return cls._c_Attrib
#     @classmethod
#     def setc_Attrib(cls, value):
#         test = 1
#         cls._c_Attrib = value
#

# class classproperty(object):
#     def __init__(self, getter):
#         self.getter= getter
#     def __get__(self, instance, owner):
#         return self.getter(owner)
#
# class a(object):
#     _c_Attrib= 4
#     @classproperty
#     def c_Attrib(cls):
#         return cls._c_Attrib
#
# x = a()
#
# a.c_Attrib = 22
# print ('\na.c_Attrib: ',a.c_Attrib)
# print ('x.c_Attrib: ',x.c_Attrib)
# print ('a._c_Attrib: ',a._c_Attrib)
# print ('x._c_Attrib: ',x._c_Attrib)
#
# x.c_Attrib = 101
# x._c_Attrib = 99
# print ('\nx.c_Attrib: ',x.c_Attrib)
# print ('x._c_Attrib: ',x._c_Attrib)
# print ('a.c_Attrib: ',a.c_Attrib)
# print ('a._c_Attrib: ',a._c_Attrib)
#
# a.c_Attrib = 44
# a._c_Attrib = 45
# print ('\nx.c_Attrib: ',x.c_Attrib)
# print ('x._c_Attrib: ',x._c_Attrib)
# print ('a.c_Attrib: ',a.c_Attrib)
# print ('a._c_Attrib: ',a._c_Attrib)

# ------------


# class classproperty(object):
#     def __init__(self, getter):
#         self.getter= getter
#     def __get__(self, instance, owner):
#         return self.getter(owner)
#
# class a(object):
#     # classPrefs= 4
#     @classproperty
#     def c_Attrib(cls):
#         try:
#             return cls.classPrefs
#         except:
#             cls.classPrefs = 'CREATED'
#             return cls.classPrefs

# x = a()
# print (x.classPrefs)
# print (a.classPrefs)
#
# a.c_Attrib = 22
# print ('\na.c_Attrib: ',a.c_Attrib)
# print ('x.c_Attrib: ',x.c_Attrib)
# print ('a.classPrefs: ',a.classPrefs)
# print ('x.classPrefs: ',x.classPrefs)
#
# x.c_Attrib = 101
# x.classPrefs = 99
# print ('\nx.c_Attrib: ',x.c_Attrib)
# print ('x.classPrefs: ',x.classPrefs)
# print ('a.c_Attrib: ',a.c_Attrib)
# print ('a.classPrefs: ',a.classPrefs)
#
# a.c_Attrib = 44
# a.classPrefs = 45
# print ('\nx.c_Attrib: ',x.c_Attrib)
# print ('x.classPrefs: ',x.classPrefs)
# print ('a.c_Attrib: ',a.c_Attrib)
# print ('a.classPrefs: ',a.classPrefs)
#
#
#
#endregion

#region TEST:  DICTIONARY MERGE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # -
#
#
# # a = {'hello':1}
# a = {}
# # b = {'word':2}
# b = {}
# # c = {**a, **b} # AWAITING 3.5
# c = {}
# c.update(a)
# c.update(b)
# if (c):
#     print(c)
# else:
#     print('empty')
#
#
#
#endregion

# region TEST: SEQUENTIAL ERROR HANDLING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # state_params = None
# state_params = {}
# # state_params = {'Already there': 0}
# state_spec = {'hello': {'deeper dict':1}}
# key = 'goodbye'
# # key = 'hello'
#
# try:
#     state_params.update(state_spec[key])
# # state_spec[STATE_PARAMS] was not specified
# except KeyError:
#         pass
# # state_params was not specified
# except (AttributeError):
#     try:
#         state_params = state_spec[key]
#     # state_spec[STATE_PARAMS] was not specified
#     except KeyError:
#         state_params = {}
# # state_params was specified but state_spec[STATE_PARAMS] was not specified
# except TypeError:
#     pass
# #endregion

#region TEST:  ORDERED DICTIONARY ORDERING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# from collections import OrderedDict
#
# a = OrderedDict()
# a['hello'] = 1
# a['world'] = 2
#
# for x in a:
#     print ('x: ', x)
#
# print ("a: ", a)
# print (list(a.items())[0], list(a.items())[1])
#
# b = {'hello':1, 'world':2}
#
# print ("b: ", b)
#
#endregion

#region TEST:  add a parameterState to a param after an object is instantiated @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from Components.Mechanisms.DDM import DDM
# from Components.States.ParameterState import ParameterState
#
# x = DDM()
# state = x._instantiate_state(state_type=ParameterState,
#                               state_name='DDM_TEST_PARAM_STATE',
#                               state_spec=100.0,
#                               constraint_value=0.0,
#                               constraint_value_name='DDM T0 CONSTRAINT',
#                               context='EXOGENOUS SPEC')
# x.parameterStates['DDM_TEST_PARAM_STATE'] = state

# x._instantiate_state_list(state_type=ParameterState,
#                                    state_param_identifier='DDM_TEST',
#                                    constraint_value=0.0,
#                                    constraint_value_name='DDM T0 CONSTRAINT',
#                                    context='EXOGENOUS SPEC')

#endregion

#region TEST OF AutoNumber IntEnum TYPE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# #
#
# from enum import IntEnum
# class AutoNumber(IntEnum):
#     """Autonumbers IntEnum type
#
#     Adapted from AutoNumber example for Enum at https://docs.python.org/3/library/enum.html#enum.IntEnum:
#     Notes:
#     * Start of numbering changed to 0 (from 1 in example)
#     * obj based on int rather than object
#     """
#     def __new__(cls):
#         # Original example:
#         # value = len(cls.__members__) + 1
#         # obj = object.__new__(cls)
#         value = len(cls.__members__)
#         obj = int.__new__(cls)
#         obj._value_ = value
#         return obj
#
# class DDM_Output(AutoNumber):
#     DDM_DECISION_VARIABLE = ()
#     DDM_RT_MEAN = ()
#     DDM_ER_MEAN = ()
#     DDM_RT_CORRECT_MEAN = ()
#     DDM_RT_CORRECT_VARIANCE = ()
#     TOTAL_COST = ()
#     TOTAL_ALLOCATION = ()
#     NUM_OUTPUT_VALUES = ()
#
# class DDM_Output_Int(IntEnum):
#     DDM_DECISION_VARIABLE = 0
#     DDM_RT_MEAN = 1
#     DDM_ER_MEAN = 2
#     DDM_RT_CORRECT_MEAN = 3
#     DDM_RT_CORRECT_VARIANCE = 4
#     TOTAL_COST = 5
#     TOTAL_ALLOCATION = 6
#     NUM_OUTPUT_VALUES = 7
#
# x = DDM_Output.NUM_OUTPUT_VALUES
# # x = DDM_Output_Int.NUM_OUTPUT_VALUES
#
# print (x.value)

#endregion

#region TEST OF RIGHT REPLACE (rreplace) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#

# def rreplace(myStr, old, new, count):
#     return myStr[::-1].replace(old[::-1], new[::-1], count)[::-1]
#
# new_str = rreplace('hello-1', '-1', '-2', 1)
# print(new_str)

#endregion

#region TEST OF OrderedDict @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# from collections import OrderedDict
# from collections import Counter
#
# # class OrderedCounter(Counter, OrderedDict):
# #     'Counter that remembers the order elements are first encountered'
# #
# #     def __repr__(self):
# #         return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
# #
# #     def __reduce__(self):
# #         return self.__class__, (OrderedDict(self),)
#
#
# a = OrderedDict({'hello': 1,
#                  'goodbye': 2
#                  })
# a['I say'] = 'yes'
# a['You say'] = 'no'
#
# print ('dict: ', a)
#
# print (list(a.items()))
#
# for key, value in a.items():
#     print('value of {0}: '.format(key), value)
#
# print("keys.index('hello'): ", list(a.keys()).index('hello'))
# print('keys.index(1): ', list(a.values()).index(1))
# print("keys.index('I say'): ", list(a.keys()).index('I say'))
# print("keys.index('yes'): ", list(a.values()).index('yes'))
#
# print("list(values): ", list(a.values()))
# print("values[0]: ", list(a.values())[0])
# print("values[2]: ", list(a.values())[2])
#
#
#
# # for item in a if isinstance(a, list) else list(a.items()[1]:
# #     print (item)
#
# # a = [1, 2, 3]
# #
# #
# # for key, value in a.items() if isinstance(a, dict) else enumerate(a):
# #     print (value)
# #
# #
# # # for value in b:
# # for key, value in enumerate(b):
# #     print (value)
# #
# #
# # d.values().index('cat')
# # d.keys().index('animal')
# # list(d.keys()).index("animal")
# #
#
# #endregion

#region TEST OF List indexed by string @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# # a = MyClass()
# # b = MyClass()
# # # a.name = 'hello'
# # a.name.append('hello')
# # print(a.name)
# # print(b.name)
#
#
# # from collections import UserList
# #
# # class ClassListTest(UserList):
# #     """Implements dict-like list, that can be indexed by the names of the States in its entries.
# #
# #     Supports getting and setting entries in the list using string (in addition to numeric) indices.
# #     For getting an entry:
# #         the string must match the name of a State in the list; otherwise an excpetion is raised.
# #     For setting an entry:
# #         the string must match the name of the State being assigned;
# #         if there is already a State in the list the name of which matches the string, it is replaced;
# #         if there is no State in the list the name of which matches the string, the State is appended to the list.
# #
# #     IMPLEMENTATION NOTE:
# #         This class allows the states of a mechanism to be maintained in lists, while providing the convenience
# #         (to the user) of access and assignment by name (e.g., akin to a dict).
# #         Lists are used (instead of a dict or OrderedDict) since:
# #             - ordering is in many instances convenient, and in some critical (e.g., for consistent mapping from
# #                 collections of states to other variables, such as lists of their values);
# #             - they are most commonly accessed either exhaustively (e.g., in looping through them during execution),
# #                 or by index (e.g., to get the first, "primary" one), which makes the efficiencies of a dict for
# #                 accessing by key/name less critical;
# #             - the number of states in a collection for a given mechanism is likely to be small so that, even when
# #                 accessed by key/name, the inefficiencies of searching a list are likely to be inconsequential.
# #     """
# #
# #     def __init__(self, list=None, name=None, **kwargs):
# #         self.name = name or self.__class__.__name__
# #         UserList.__init__(self, list, **kwargs)
# #         # self._ordered_keys = []
# #
# #     def __getitem__(self, index):
# #         try:
# #             return self.data[index]
# #         except TypeError:
# #             index = self._get_index_for_item(index)
# #             return self.data[index]
# #
# #     def __setitem__(self, index, value):
# #         try:
# #             self.data[index] = value
# #         except TypeError:
# #             if not index is value.name:
# #                 raise ScratchPadError("Name of entry for {} ({}) must match the name of its State ({})".
# #                                       format(self.name, index, value.name))
# #             index_num = self._get_index_for_item(index)
# #             if index_num is not None:
# #                 self.data[index_num] = value
# #             else:
# #                 self.data.append(value)
# #
# #     def _get_index_for_item(self, index):
# #         if isinstance(index, str):
# #             # return self.data.index(next(obj for obj in self.data if obj.name is index))
# #             obj = next((obj for obj in self.data if obj.name is index), None)
# #             if obj is None:
# #                 return None
# #             else:
# #                 return self.data.index(obj)
# #
# #         elif isinstance(index, MyClass):
# #             return self.data.index(index)
# #         else:
# #             raise ScratchPadError("{} is not a legal index for {} (must be number, string or State".
# #                                   format(index, self.name))
# #
# #     def __delitem__(self, index):
# #         del self.data[index]
# #
# #     def clear(self):
# #         super().clear(self)
# #
# #     # def pop(self, index, *args):
# #     #     raise UtilitiesError("{} is read-only".format(self.name))
# #     # def popitem(self):
# #     #     raise UtilitiesError("{} is read-only".format(self.name))
# #
# #     def __additem__(self, index, value):
# #         if index >= len(self.data):
# #             self.data.append(value)
# #         else:
# #             self.data[index] = value
# #
# #
# #     def __contains__(self, item):
# #         if super().__contains__(item):
# #             return True
# #         else:
# #             return any(item is obj.name for obj in self.data)
# #
# #     def copy(self):
# #         return self.data.copy()
#
# class MyClass():
#     name = None
#     def __init__(self, name=None):
#         self.name = name
#         # self.name = name
#
# my_obj = MyClass(name='hello')
# my_obj_2 = MyClass(name='goodbye')
# my_obj_3 = MyClass(name='goodbye')
#
# from PsyNeuLink.Globals.Utilities import ContentAddressableList
#
# my_list = ContentAddressableList(component_type=MyClass)
# # my_list.append(my_state)
# my_list['hello'] = my_obj
# my_list['goodbye'] = my_obj_2
# print(my_list, len(my_list))
# print(my_list[0])
# print(my_list['hello'])
# print(my_list['goodbye'])
# my_list['goodbye'] = my_obj_3
# print(my_list['goodbye'])
# print('hello' in my_list)

#endregion

#region TEST parse_state_spec @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# print("TEST parse_gated_state_spec")
#
# from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanism.GatingMechanism import _parse_gating_signal_spec
# from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanism.GatingSignal import GatingSignal
# from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanism.GatingMechanism import GatingMechanism
# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
# from PsyNeuLink.Components.Functions.Function import ModulationParam
# from PsyNeuLink.Components.States.OutputState import OutputState
#
# gating_mech = GatingMechanism()
# mech_1 = DDM()
# mech_2 = DDM()
#
# single_dict = {NAME:'DECISION_VARIABLE', MECHANISM:mech_1}
# a = _parse_gating_signal_spec(owner=gating_mech, state_spec=single_dict)
# print('\nsingle_dict:', a)
#
# # THESE ARE A HACK TO ASSIGN A PRE-EXISTING GATING SIGNAL TO gating_mech WITHOUTH HAVING TO CONSTRUCT ONE
# x = DDM()
# x.name ='Default_input_state_GatingSignal'
# x.efferents = []
# gating_mech._gating_signals = [x]
# # --------------------------------------------------------------
#
# single_tuple = ('DECISION_VARIABLE', mech_1)
# b = _parse_gating_signal_spec(gating_mech, state_spec=single_tuple)
# print('\nsingle_tuple:', b)
#
# multi_states_dicts = {'MY_SIGNAL':[{NAME:'DECISION_VARIABLE',
#                                    MECHANISM:mech_1},
#                                    {NAME:'RESPONSE_TIME',
#                                    MECHANISM:mech_1}],
#                       MODULATION: ModulationParam.ADDITIVE}
# c = _parse_gating_signal_spec(gating_mech, state_spec=multi_states_dicts)
# print('\nmulti_states_dicts:', c)
#
# multi_states_tuples = {'MY_SIGNAL':[('Default_input_state',mech_1),
#                                    ('Default_input_state',mech_2)]}
# d = _parse_gating_signal_spec(gating_mech, state_spec=multi_states_tuples)
# print('\nmulti_states_tuples:', d)
#
# multi_states_combo = {'MY_SIGNAL':[{NAME:'Default_input_state',
#                                    MECHANISM:mech_1},
#                                    ('Default_input_state',mech_2)]}
# e = _parse_gating_signal_spec(gating_mech, state_spec=multi_states_combo)
# print('\nmulti_states_combo:', e)

#endregion

# region TEST parse_monitored_output_state @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import *
# from PsyNeuLink.Components.States.OutputState import OutputState
#
# print("TEST parse_monitored_output_state")
#
# def _parse_monitored_output_state(owner, monitored_output_states):
#     """Parse specifications contained in monitored_output_states list or dict,
#
#     Can take either a list or dict of specifications.
#     If it is a list, each item must be one of the following:
#         - OuptutState
#         - Mechanism
#         - string
#         - value
#         - dict
#
#     If it is a dict, each item must be an entry, the key of which must be a string that is used as a name
#         specification, and the value of which can be any of the above.
#
#     Return a list of specification dicts, one for each item of monitored_output_states
#     """
#
#
#     def parse_spec(spec):
#
#         # OutputState:
#         if isinstance(spec, OutputState):
#             name = spec.owner.name + MONITORED_OUTPUT_STATE_NAME_SUFFIX
#             value = spec.value
#             call_for_projection = True
#
#         # Mechanism:
#         elif isinstance(spec, Mechanism_Base):
#             name = spec.name + MONITORED_OUTPUT_STATE_NAME_SUFFIX
#             value = spec.output_state.value
#             call_for_projection = True
#
#         # # If spec is a MonitoredOutputStatesOption:
#         # # FIX: NOT SURE WHAT TO DO HERE YET
#         # elif isinstance(monitored_value, MonitoredOutputStatesOption):
#         #     value = ???
#         #     call_for_projection = True
#
#         # If spec is a string:
#         # - use as name of inputState
#         # - instantiate InputState with defalut value (1d array with single scalar item??)
#
#         # str:
#         elif isinstance(spec, str):
#             name = spec
#             value = DEFAULT_MONITORED_OUTPUT_STATE
#             call_for_projection = False
#
#         # value:
#         elif is_value_spec(spec):
#             name = owner.name + MONITORED_OUTPUT_STATE_NAME_SUFFIX
#             value = spec
#             call_for_projection = False
#
#         elif isinstance(spec, tuple):
#             # FIX: REPLACE CALL TO parse_spec WITH CALL TO _parse_state_spec
#             name = owner.name + MONITORED_OUTPUT_STATE_NAME_SUFFIX
#             value = spec[0]
#             call_for_projection = spec[1]
#
#         # dict:
#         elif isinstance(spec, dict):
#
#             name = None
#             for k, v in spec.items():
#                 # Key is not a spec keyword, so dict must be of the following form: STATE_NAME_ASSIGNMENT:STATE_SPEC
#                 #
#                 if not k in {NAME, VALUE, PROJECTIONS}:
#                     name = k
#                     value = v
#
#             if NAME in spec:
#                 name = spec[NAME]
#
#             call_for_projection = False
#             if PROJECTIONS in spec:
#                 call_for_projection = spec[PROJECTIONS]
#
#             if isinstance(spec[VALUE], (dict, tuple)):
#                 # FIX: REPLACE CALL TO parse_spec WITH CALL TO _parse_state_spec
#                 entry_name, value, call_for_projection = parse_spec(spec[VALUE])
#
#             else:
#                 value = spec[VALUE]
#
#         else:
#             raise ObjectiveMechanismError("Specification for {} arg of {} ({}) must be an "
#                                           "OutputState, Mechanism, value or string".
#                                           format(MONITORED_OUTPUT_STATES, owner.name, spec))
#
#         return name, value, call_for_projection
#
#     # If it is a dict, convert to list by:
#     #    - assigning the key of each entry to a NAME entry of the dict
#     #    - placing the value in a VALUE entry of the dict
#     if isinstance(monitored_output_states, dict):
#         monitored_output_states_list = []
#         for name, spec in monitored_output_states.items():
#             monitored_output_states_list.append({NAME: name, VALUE: spec})
#         monitored_output_states = monitored_output_states_list
#
#     if isinstance(monitored_output_states, list):
#
#         for i, monitored_output_state in enumerate(monitored_output_states):
#             name, value, call_for_projection = parse_spec(monitored_output_state)
#             monitored_output_states[i] = {NAME: name,
#                                    VALUE: value,
#                                    PROJECTION: call_for_projection}
#
#     else:
#         raise ObjectiveMechanismError("{} arg for {} ({} )must be a list or dict".
#                                       format(MONITORED_OUTPUT_STATES, owner.name, monitored_output_states))
#
#     return monitored_output_states
#
#
#
#     # def add_monitored_output_states(self, states_spec, context=None):
#     #     """Validate specification and then add inputState to ObjectiveFunction + MappingProjection to it from state
#     #
#     #     Use by other objects to add a state or list of states to be monitored by EVC
#     #     states_spec can be a Mechanism, OutputState or list of either or both
#     #     If item is a Mechanism, each of its outputStates will be used
#     #
#     #     Args:
#     #         states_spec (Mechanism, MechanimsOutputState or list of either or both:
#     #         context:
#     #     """
#     #     states_spec = list(states_spec)
#     #     validate_monitored_output_state(self, states_spec, context=context)
#     #     self._instantiate_monitored_output_states(states_spec, context=context)
#
# class SCRATCH_PAD():
#     name = 'SCRATCH_PAD'
#
# print(_parse_monitored_output_state(SCRATCH_PAD, {'TEST_STATE_NAME':{VALUE: (32, 'Projection')}}))

#endregion

#region PREFERENCE TESTS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#
# from Globals.Preferences.PreferenceSet import *
# from Globals.Preferences.ComponentPreferenceSet import *
#
# class a(object):
#     prefs = None
#     def __init__(self):
#         a.prefs = ComponentPreferenceSet(owner=a,
#                                         log_pref=PreferenceEntry(1,PreferenceLevel.SYSTEM),
#                                         level=PreferenceLevel.SYSTEM)
#
# class b(a):
#     prefs = None
#     def __init__(self):
#         super(b, self).__init__()
#         b.prefs = ComponentPreferenceSet(owner=b,
#                                         log_pref=PreferenceEntry(5,PreferenceLevel.CATEGORY),
#                                         level=PreferenceLevel.CATEGORY)
#
# class c(b):
#     prefs = None
#     def __init__(self):
#         super(c, self).__init__()
#         c.prefs = ComponentPreferenceSet(owner=self,
#                                         log_pref=PreferenceEntry(3,PreferenceLevel.INSTANCE),
#                                         level=PreferenceLevel.INSTANCE)
#         self.prefs = c.prefs
#
#
# x = c()
#
# x.prefs.logLevel = PreferenceLevel.CATEGORY
# y = x.prefs.logPref
# print (y)
#
# x.prefs.logLevel = PreferenceLevel.INSTANCE
# y = x.prefs.logPref
# print (x.prefs.logPref)
#
# print ("system: ", x.prefs.get_pref_setting_for_level(kpLogPref, PreferenceLevel.SYSTEM))
# print ("category: ", x.prefs.get_pref_setting_for_level(kpLogPref, PreferenceLevel.CATEGORY))
# print ("instance: ", x.prefs.get_pref_setting_for_level(kpLogPref, PreferenceLevel.INSTANCE))
#
# # # print ("system: ", b.prefs.get_pref_setting(b.prefs.logEntry, PreferenceLevel.CATEGORY))
# # # print ("system: ", x.prefs.get_pref_setting(x.prefs.logEntry, PreferenceLevel.SYSTEM))
# # # print ("category: ", x.prefs.get_pref_setting(x.prefs.logEntry, PreferenceLevel.CATEGORY))
# # # print ("instance: ", x.prefs.get_pref_setting(x.prefs.logEntry, PreferenceLevel.INSTANCE))
#

#endregion

#region ATTEMPT TO ASSIGN VARIABLE TO NAME OF ATTRIBUTE: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#
#
# y = 'x'
#
# class a:
#
#     def __init__(self, y):
#         z = getattr(self,y)
#
#     @property
#     def z(self):
#         return self._x
#
#     @z.setter
#     def z(self, value):
#         self._x = value
#
# b = a(y)
# q = getattr(a,y)
# q = 3
# print (a.q)


# from Components.States.InputState import InputState
#
# test = InputState(value=1)
# x = 1

# def func_b():
#     print('hello from b')
#
# class a:
#     def __init__(self):
#         self.func_a = func_b
#
#     def func_a(self):
#         print('hello from a')
#
# x = a()
# x.__class__.func_a(a)
# a.func_a(a)
# c = a.func_a
# c(a)
#

# # a = 'hello'
# class b:
#     pass
#
# class a(b):
#     pass
# # a = [1]
# # test = {'goodbye':2}
#
# try:
#     issubclass(a, b)
# except TypeError:
#     if isinstance(a, str):
#         try:
#             print(test[a])
#         except KeyError:
#             print("got to KeyError nested in TypeError")
#     else:
#         print("got to string else")
#
# else:
#     print("got to outer try else")

# class i():
#     attrib = 0
#     pass
#
# class a(i):
#     pass
#
# x = i()
# y = i()
#
# x.attrib = [1,1]
# y.attrib = [1,2,3]
# print ('x: ', x.attrib, 'y: ', y.attrib)


# z = 'goo'
# x = {'hello':1}
# try:
#     y.a = x[z]
# except KeyError:
#     print('key error')
# except AttributeError:
#     print('attrib error')
# else:
#     print('OK')
#
#
#                     try:
#                         self.paramClassDefaults[FUNCTION] = self.execute
#                     except KeyError:
#                         message = ("{0} missing from {1}".format(required_param, self.name))
#                         self.execute =
#                         xxx
#                     except AttributeError:
# # IMPLEMENTATION NOTE:  *** PARSE ERROR HERE:  WARN IF KEY ERROR, AND ASSIGN FUNCTION;  EXCEPT IF ATTRIBUTE ERROR
#                         raise ComponentError("Either {0} must be specified in paramClassDefaults or"
#                                             " <class.function> must be implemented for {1}".
#                                             format(required_param, self.name))
#                     else:
#                         self.requiredParamClassDefaultTypes[required_param].append(type(self.execute))
#                         if self.functionSettings & FunctionSettings.VERBOSE:
#










# class TestClass(object):
#     def __init__(self):
#         # self.prop1 = None
#         pass
#
#     @property
#     def prop1(self):
#         print ("I was here")
#         return self._prop1
#
#     @prop1.setter
#     def prop1(self, value):
#         print ("I was there")
#         self._prop1 = value
#
#
# a = TestClass()
# a._prop1 = 1
# a.prop1 = 2
# print (a.prop1)
# print (a._prop1)
#

# class C(object):
#     def __init__(self):
#         self._x = None
#
#     @property
#     def x(self):
#         """I'm the 'x' property."""
#         print ("I was here")
#         return self._x
#
#     @x.setter
#     def x(self, value):
#         print ("I was there")
#         self._x = value
#
#
# a = C()
# a.x = 2
# y = a.x
# print (y)







    # return SubTestClass()
#
#
# class TestClass(arg=NotImplemented):
#     def __init__(self):
#         print("Inited Test Class")
#
#
# class SubTestClass(TestClass):
#     def __init__(self):
#         super(SubTestClass, self).__init__()
#         print("Inited Sub Test Class")



# class x:
#       def __init__(self):
#             self.execute = self.execute
#             print("x: self.execute {0}: ",self.execute)
#
# class z(x):
#
#       def __init__(self):
#             super(z, self).__init__()
#             print("z: self.execute {0}: ",self.execute)
#
#       def function(self):
#             pass
#
# y =  z()
#

# paramDict = {'number':1, 'list':[0], 'list2':[1,2], 'number2':2}
#
# def get_params():
#       return (dict((param, value) for param, value in paramDict.items()
#                     if isinstance(value,list) ))
#
#
# print(get_params()['list2'])

# q = z()
#
# class b:
#       pass
#
# print(issubclass(z, x))

# print("x: ",x)
# print("type x: ",type(x))
# print("y: ",y)
# print("type x: ",type(y))
# print(isinstance(y, x))

# test = {'key1':1}
# try:
#       x=test["key2"]
#       x=test["key1"]
# except KeyError:
#       print("passed")
# print(x)

# ***************************************** OLD TEST SCRIPT ************************************************************

# from Components.Projections.ControlProjection import *
#
# # Initialize control_signal with some settings
# settings = ControlSignalSettings.DEFAULTS | \
#            ControlSignalSettings.DURATION_COST | \
#            ControlSignalSettings.LOG
# identity = []
# log_profile = ControlSignalLog.ALL
#
# # Set up ControlProjection
# x = ControlSignal_Base("Test Control Signal",
#                        {kwControlSignalIdentity: identity,
#                         kwControlSignalSettings: settings,
#                         kwControlSignalAllocationSamplingRange: NotImplemented,
#                        )
#
# # Can also change settings on the fly (note:  ControlProjection.OFF is just an enum defined in the ControlProjection module)
# x.set_adjustment_cost(OFF)
#
# # Display some values in control_signal (just to be sure it is set up OK)
# print("Intensity Function: ", x.functions[kwControlSignalIntensityFunction].name)
# print("Initial Intensity: ", x.intensity)
#
# # Add KVO:
# #  Utilities will observe ControlProjection.kpIntensity;
# #  the observe_value_at_keypath method in Utilities will be called each time ControlProjection.kpIntensity changes
# x.add_observer_for_keypath(Utilities,kpIntensity)
#
#
# # Assign testFunction to be a linear function, that returns the current value of an object property (intensity_cost)
# # It is called and printed out after each update of the control signal below;  note that it returns the updated value
# # Note: the function (whether a method or a lambda function) must be in a list so it is not called before being passed
# testFunction_getVersion = Function.Linear([x.get_intensity_cost])
# testFunction_lambdaVersion = Function.Linear([lambda: x.intensity_cost])
# label = x.get_intensity_cost
#
# print("\nINITIAL {0}".format(x.duration_cost))
#
# #Print out test of function with object property assigned as its default variable argument
# getVersion = testFunction_getVersion.function()
# print("{0}: {1}\n".format(label, getVersion))
# lambdaVersion = testFunction_lambdaVersion.function()
# print("{0}: {1}\n".format(label, lambdaVersion))
#
# # Initial allocation value
# z = 3
#
# Utilities.CentralClock.time_step = 0
# x.update_control_signal(z)
# getVersion = testFunction_getVersion.function()
# print("{0}: {1}\n".format(label, getVersion))
# lambdaVersion = testFunction_lambdaVersion.function()
# print("{0}: {1}\n".format(label, lambdaVersion))
#
# #Update control signal with new allocation value
# Utilities.CentralClock.time_step = 1
# x.update_control_signal(z+1)
# getVersion = testFunction_getVersion.function()
# print("{0}: {1}\n".format(label, getVersion))
# lambdaVersion = testFunction_lambdaVersion.function()
# print("{0}: {1}\n".format(label, lambdaVersion))
#
#
# #Update control signal with new allocation value
# Utilities.CentralClock.time_step = 2
# x.update_control_signal(z-2)
# getVersion = testFunction_getVersion.function()
# print("{0}: {1}\n".format(label, getVersion))
# lambdaVersion = testFunction_lambdaVersion.function()
# print("{0}: {1}\n".format(label, lambdaVersion))
#
# #Show all entries in log
# print("\n")
# x.log.print_all_entries()
#
# # q = lambda: x.intensity
# # print(q())
# print((lambda: x.intensity)())
# x.intensity = 99
# print((lambda: x.intensity)())
#


# # test DDM call from Matlab
# print("importing matlab...")
# import matlab.engine
# eng1=matlab.engine.start_matlab('-nojvm')
# print("matlab imported")
#
#
# drift = 0.1
# bias = 0.5
# thresh = 3.0
# noise = 0.5
# T0 = 200
#
#
# t = eng1.ddmSim(drift,bias,thresh,noise,T0,1,nargout=5)
#
# # run matlab function and print output
# # t=eng1.gcd(100.0, 80.0, nargout=3)
# print(t)
#
# print("AFTER MATLAB")
# #end

# exit()
