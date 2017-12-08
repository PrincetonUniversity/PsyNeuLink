import matplotlib.pyplot as plt
import numpy as np

from psyneulink import PredictionErrorDeltaFunction


def test_prediction_error_delta_first_run():
    learning_rate = 0.3

    stimulus_onset = 41
    sample = np.zeros(60)
    sample[stimulus_onset:] = 1

    reward_onset = 54
    target = np.zeros(60)
    target[reward_onset] = 1

    delta_function = PredictionErrorDeltaFunction()

    weights = np.zeros(60)
    for t in range(60):
        new_sample = sample * weights
        print("sample = {}".format(new_sample))
        delta_vals = delta_function.function(variable=[new_sample, target])

        for i in range(1, 60):
            weights[i - 1] = weights[i - 1] + learning_rate * sample[i - 1] * delta_vals[i]
        print("Timestep {}".format(t))
        print(delta_vals)
        print("weights = {}".format(weights))
        plt.plot(delta_vals)
        plt.show()
        plt.gcf().clear()

# TODO: add asserts