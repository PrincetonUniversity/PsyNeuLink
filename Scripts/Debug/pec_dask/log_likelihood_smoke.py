import numpy as np
import pandas as pd

import psyneulink as pnl


def run_ddm_with_params(threshold, trial_inputs):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=0.3,
            noise=1.0,
            threshold=threshold,
            non_decision_time=0.15,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    comp.run(inputs={decision: trial_inputs})

    data = pd.DataFrame(
        np.squeeze(np.array(comp.results)),
        columns=["decision", "response_time"],
    )
    data["decision"] = data["decision"].astype("category")
    return comp, data


def make_pec(comp, data, parameters, depends_on=None):
    decision = comp.nodes[0]
    pec = pnl.ParameterEstimationComposition(
        name="pec_log_likelihood_smoke",
        nodes=[comp],
        parameters=parameters,
        depends_on=depends_on,
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution",
            max_iterations=1,
        ),
        num_estimates=20,
        initial_seed=42,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec


def scalar_smoke():
    trial_inputs = np.ones((12, 1))
    comp, data = run_ddm_with_params(0.6, trial_inputs)
    decision = comp.nodes[0]
    pec = make_pec(
        comp,
        data,
        {
            ("rate", decision): np.linspace(-0.5, 0.5, 10),
            ("threshold", decision): np.linspace(0.5, 1.0, 10),
            ("non_decision_time", decision): np.linspace(0.0, 1.0, 10),
        },
    )

    ll, sim_data = pec.log_likelihood(
        0.3,
        0.6,
        0.15,
        inputs={comp: trial_inputs},
        return_sim_data=True,
    )
    print("scalar log_likelihood:", ll)
    print("scalar sim_data shape:", sim_data.shape)


def conditional_smoke():
    trial_inputs = np.ones((8, 1))
    comp, data_a = run_ddm_with_params(0.7, trial_inputs)
    _, data_b = run_ddm_with_params(0.3, trial_inputs)

    data_a["condition"] = "a"
    data_b["condition"] = "b"
    data = pd.concat([data_a, data_b], ignore_index=True)
    data["decision"] = data["decision"].astype("category")
    data["condition"] = data["condition"].astype("category")

    decision = comp.nodes[0]
    pec = make_pec(
        comp,
        data,
        {("threshold", decision): np.linspace(0.1, 1.0, 10)},
        depends_on={("threshold", decision): "condition"},
    )

    ll, sim_data = pec.log_likelihood(
        0.7,
        0.3,
        inputs={comp: np.vstack([trial_inputs, trial_inputs])},
        return_sim_data=True,
    )
    print("conditional parameter names:", pec.controller.function.fit_param_names)
    print("conditional log_likelihood:", ll)
    print("conditional sim_data shape:", sim_data.shape)


if __name__ == "__main__":
    scalar_smoke()
    conditional_smoke()
