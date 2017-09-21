import numpy as np

from PsyNeuLink import TransferMechanism, SoftMax, PROB, process, \
    CentralClock, system, LinearCombination, LearningProjection
from PsyNeuLink.Globals.Keywords import NAME, SAMPLE, VARIABLE, TARGET, \
    RECEIVER, VALUE
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .PredictionErrorMechanism import PredictionErrorMechanism


def test_prediction_error_mechanism():
    pe_mech = PredictionErrorMechanism(
            sample={
                NAME: SAMPLE,
                VALUE:  0
            },
            target={
                NAME: TARGET,
                VALUE: 100
            },
            name="Prediction Error")
    val = pe_mech.execute()
    print(val)

# def test_prediction_mechanism():
#     sample = TransferMechanism(
#         default_variable=[0, 0, 0],
#         size=3,
#         name='Sample'
#     )

#     target = TransferMechanism(
#         default_variable=[0, 1, 0],
#         name='Target'
#     )

#     action_selection = TransferMechanism(
#         default_variable=[0, 0, 0],
#         function=SoftMax(output=PROB, gain=1.0),
#         name='Action Selection'
#     )

#     pred_error_mech = PredictionErrorMechanism(
#         sample=sample.output_state,
#         target=target.output_state,
#         input_states=[
#             {
#                 NAME: SAMPLE,
#                 VARIABLE: sample.output_values[0]
#             },
#             {
#                 NAME: TARGET,
#                 VARIABLE: target.output_values[0]
#             }],
#         name="Prediction Error"
#     )

#     p = process(
#         default_variable=[0, 0, 0],
#         size=3,
#         pathway=[sample, pred_error_mech, action_selection],
#         target=0
#     )

#     reward_values = [10, 10, 10]

#     action_selection.standard_output_states.value = [0, 0, 1]

#     reward = lambda: [
#         reward_values[int(np.nonzero(action_selection.output_state.value[0]))]]

#     def print_header():
#         print("\n\n**** TRIAL: {}".format(CentralClock.trial))

#     def show_weights():
#         print("Reward prediction weights: \n",
#               pred_error_mech.output_state.value)
#         print(
#             "\nAction Selected: {}; predicted reward: {}".format(
#                 np.nonzero(action_selection.output_state.value),
#                 action_selection.output_state.value[
#                     np.nonzero(action_selection.output_state.value)]))

#     input_list = {sample: [[1, 1, 1]]}

#     s = system(
#         processes=[p],
#         # targets=[0]
#     )

#     results = s.run(num_trials=10,
#                     inputs=input_list,
#                     targets=reward,
#                     call_before_trial=print_header(),
#                     call_after_trial=show_weights())

#     results_list = []

#     for elem in s.results:
#         for nested_elem in elem:
#             nested_elem = nested_elem.tolist()
#             try:
#                 iter(nested_elem)
#             except TypeError:
#                 nested_elem = [nested_elem]
#             results_list.extend(nested_elem)
