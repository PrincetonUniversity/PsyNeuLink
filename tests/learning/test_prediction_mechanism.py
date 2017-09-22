import matplotlib.pyplot as plt

import numpy as np

from PsyNeuLink import TransferMechanism, SoftMax, PROB, process, \
    CentralClock, system, LinearCombination, LearningProjection
from PsyNeuLink.Components.Functions.Function import TDDeltaFunction, \
    AdaptiveIntegrator
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms\
    .IntegratorMechanism import \
    IntegratorMechanism
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Globals.Keywords import NAME, SAMPLE, VARIABLE, TARGET, \
    RECEIVER, VALUE, STATE_TYPE, INPUT_STATE
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .PredictionErrorMechanism import PredictionErrorMechanism

# Very simple test demonstrating PredictionErrorMechanism
# Oscillates between 25 and 0 because the sample and target values are constant
def test_prediction_error_mechanism():
    sample = TransferMechanism(
        default_variable=[5],
        size=1,
        name=SAMPLE
    )

    target = TransferMechanism(
        default_variable=[20],
        size=1,
        name=TARGET
    )
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
                    VALUE: 20,
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

