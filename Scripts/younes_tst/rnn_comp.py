import numpy as np
import psyneulink as pnl


def linear_combination_two_inputs():
    mech = pnl.ProcessingMechanism(
        name="HIDDEN",
        input_shapes=[1, 1],
        function=pnl.LinearCombination(weights=[1.0, 0.0]),
    )

    print("defaults.variable:", mech.defaults.variable)
    print("defaults.value:", mech.defaults.value)
    print("output_port defaults.value:", mech.output_port.defaults.value)

    result = mech.execute([[2.0], [5.0]])
    print("execute([[2.0], [5.0]]) ->", result)

    assert np.allclose(result, [[2.0]]), result


if __name__ == "__main__":
    linear_combination_two_inputs()