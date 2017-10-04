import numpy as np

from PsyNeuLink import TransferMechanism, process
from PsyNeuLink.Components.Functions.Function import TDDeltaFunction
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Globals.Keywords import NAME, SAMPLE, VARIABLE, TARGET, VALUE, \
    STATE_TYPE
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .PredictionErrorMechanism import PredictionErrorMechanism


# Very simple test demonstrating PredictionErrorMechanism
# Oscillates between 25 and 0 because the sample and target values are constant


def test_prediction_error_mechanism():
    sample = TransferMechanism(size=1, name=SAMPLE)
    target = TransferMechanism(size=1, name=TARGET)

    pe_mech = PredictionErrorMechanism(
        sample=sample.output_state,
        target=target,
        input_states=[{
            NAME: SAMPLE,
            STATE_TYPE: InputState,
            VALUE: sample.execute(),
            VARIABLE: sample.output_values[0]
        },
            {
                NAME: TARGET,
                STATE_TYPE: InputState,
                VARIABLE: [20]
        }
        ],
        function=TDDeltaFunction(reward=20),
        name="Prediction Error")
    trial = 1
    prediction_errors = []
    for _ in range(5):
        print("TRIAL {} ------------------------------------".format(trial))
        val = pe_mech.execute()
        prediction_errors.append(val[0][0])
        print("Trial {} prediction error = {}".format(trial, val))
        trial += 1


# def test_mini_montague_model():
#     # sample = TransferMechanism(name=SAMPLE)
#     pe_inputs = {"sample": [[0], [1], [0], [0], [0]]}
#     pe_targets = [[0], [0], [0], [0], [1]]
#     pe_mech = PredictionErrorMechanism(input_states=[{
#         NAME: SAMPLE,
#         STATE_TYPE: InputState,
#         VARIABLE: None
#     },
#         {
#         NAME: TARGET,
#         STATE_TYPE: InputState,
#         VARIABLE: None
#     }],
#         name="Prediction Error")

#     pe_process = process(pathway=[pe_mech])

#     pe_process.run(pe_inputs, targets=pe_targets)


# Script representing model training in Montague et. al (1996)
def test_montague_model():
    sample = TransferMechanism(size=1, name=SAMPLE)
    target = TransferMechanism(size=1, name=TARGET)

    sample_vals = np.zeros((60, 1))
    target_vals = np.zeros((60, 1))

    # set element 40 to 1 to represent light presentation at t = 41
    sample_vals[40] = 1

    # set element 53 to 1 to represent reward at t = 54
    target_vals[53] = 1

    pe_mech = PredictionErrorMechanism(
        sample=sample.output_state,
        target=target,
        input_states=[{
            NAME: SAMPLE,
            STATE_TYPE: InputState,
            VARIABLE: sample.output_values[0]
        },
            {
            NAME: TARGET,
            STATE_TYPE: InputState,
            VARIABLE: target.output_values[0]
        }],
        name="Prediction Error")

    # pe_process = process(pathway=[sample, pe_mech])
    # results = pe_process.run(sample_vals, targets=target_vals, num_trials=60)
    prediction_errors = []
    for i in range(60):
        print("TIMESTEP {} -------------------------------".format(i))
        val = pe_mech.execute(input=[[sample_vals[i]], target_vals[i]])
        prediction_errors.append(val[0][0])
        # take these prediction errors and feed them to a learning projection
        # that feeds back into the pe_mech with the weights
    print("Num prediction errors: {}".format(len(prediction_errors)))
    print(prediction_errors)
