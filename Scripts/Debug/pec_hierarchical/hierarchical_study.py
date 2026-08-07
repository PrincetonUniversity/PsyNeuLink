"""Hierarchical fitting of a group of drift-diffusion participants.

Simulates a group whose parameters vary around a population mean, fits them jointly, and reports
how well the individual and group parameters were recovered.

Run in one process::

    python hierarchical_study.py

Across a single-node cluster::

    python hierarchical_study.py --distributed --n-workers 4

Across several nodes, using the SLURM launcher::

    srun -n <workers+2> python -m psyneulink.dask_run hierarchical_study.py --distributed
"""

import argparse
import time

import numpy as np
import pandas as pd

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction
from psyneulink.core.compositions.hierarchical.transforms import BoundedTransform

# Ranges searched for each parameter. These are also the support of the group model, so they are
# shared by every participant.
FIT_RANGES = {"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)}
FIT_PARAMS = tuple(FIT_RANGES)

# Population the participants are drawn from, in the unconstrained space the group model uses:
# centred on the middle of each range, with this variance between participants.
GROUP_MEAN_Z = np.zeros(len(FIT_PARAMS))
GROUP_VARIANCE_Z = np.full(len(FIT_PARAMS), 0.36)

NON_DECISION_TIME = 0.15
TIME_STEP_SIZE = 0.01
NUM_ESTIMATES = 300


def build_model(rate=0.0, threshold=0.9, seed=None):
    """A two-alternative drift-diffusion model."""
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=rate,
            noise=1.0,
            threshold=threshold,
            non_decision_time=NON_DECISION_TIME,
            time_step_size=TIME_STEP_SIZE,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    if seed is not None:
        decision.function.parameters.seed.set(int(seed) % (2 ** 32))
    return pnl.Composition(pathways=decision), decision


def trial_inputs(n_trials):
    return np.ones((n_trials, 1))


def simulate_participant(theta, n_trials, seed):
    """Generate one participant's trials at known parameters."""
    comp, _ = build_model(rate=float(theta[0]), threshold=float(theta[1]), seed=seed)
    comp.run(inputs={comp.nodes[0]: trial_inputs(n_trials)}, context=f"simulate-{seed}")
    data = pd.DataFrame(np.squeeze(np.array(comp.results)),
                        columns=["decision", "response_time"])
    data["decision"] = data["decision"].astype("category")
    return data


def make_group_data(n_participants, n_trials, seed=0):
    """Draw participants from the population and simulate each one.

    Returns the stacked table to fit and the parameters it was generated from.
    """
    transform = BoundedTransform(
        lower=[FIT_RANGES[p][0] for p in FIT_PARAMS],
        upper=[FIT_RANGES[p][1] for p in FIT_PARAMS],
    )
    rng = np.random.default_rng(seed)
    z_true = rng.normal(GROUP_MEAN_Z, np.sqrt(GROUP_VARIANCE_Z),
                        size=(n_participants, len(FIT_PARAMS)))
    theta_true = np.vstack([transform.to_natural(z) for z in z_true])

    frames = []
    for s in range(n_participants):
        frame = simulate_participant(theta_true[s], n_trials, seed=1000 + s)
        frame.insert(0, "subject", f"S{s:02d}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), theta_true, transform


def participant_pec(data, subject_index=None):
    """Build one participant's model from their trials.

    Defined at module level so that it can be sent to a worker process.  The seed varies with the
    participant: a shared seed would give every participant the same stream of simulation noise,
    which would be absorbed into the group variance rather than averaging out.
    """
    comp, decision = build_model()
    pec = pnl.ParameterEstimationComposition(
        name="participant",
        nodes=[comp],
        parameters={
            (name, decision): np.linspace(*FIT_RANGES[name], 1000) for name in FIT_PARAMS
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=NUM_ESTIMATES,
        initial_seed=100 + (subject_index or 0),
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, {comp: trial_inputs(len(data))}


def report(results, theta_true):
    """Compare what was recovered against what the data were generated from."""
    recovered = results.subject_parameters.to_numpy()

    print(f"\n{results!r}")
    print("\nobjective by iteration:",
          np.round(results.em_history["objective"].to_numpy(), 2).tolist())

    print("\ngroup estimate:")
    print(results.group_parameters.round(4).to_string())
    print(f"\n  mean of the parameters used to generate the data: "
          f"{theta_true.mean(axis=0).round(4)}")
    print(f"  mean of the recovered participant estimates:      "
          f"{recovered.mean(axis=0).round(4)}")

    print("\nper-participant recovery:")
    error = recovered - theta_true
    for k, name in enumerate(FIT_PARAMS):
        rmse = float(np.sqrt((error[:, k] ** 2).mean()))
        corr = float(np.corrcoef(recovered[:, k], theta_true[:, k])[0, 1])
        print(f"  {name:<12} RMSE {rmse:.4f}   correlation with truth {corr:.3f}")

    # The group variance accounts for both the spread of the estimates and how uncertain each one
    # is. The two agree once the fit has converged, and differ by one update before that.
    spread = results.z_hat.var(axis=0) + results.posterior_variance.mean(axis=0)
    print(f"\n  spread of estimates + mean uncertainty: {spread.round(4)}")
    print(f"  estimated group variance:               {results.sigma.round(4)}"
          f"{'' if results.converged else '   (fit did not converge; these agree once it has)'}")

    failures = int((~results.subject_converged).sum())
    print(f"\n  participant fits that converged: "
          f"{len(results.subject_labels) - failures}/{len(results.subject_labels)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-participants", type=int, default=6)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--distributed", action="store_true",
                        help="fit participants across a Dask cluster")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="size of the cluster created for --distributed")
    args = parser.parse_args()

    data, theta_true, _ = make_group_data(args.n_participants, args.n_trials)
    print(f"{len(data)} trials from {data['subject'].nunique()} participants")

    distributed_options = {"pec_factory": participant_pec}
    if args.n_workers is not None:
        distributed_options["n_workers"] = args.n_workers

    # The model given here declares which parameters are fitted and which outputs are compared
    # against the data. It is not simulated; each participant's model comes from the factory.
    comp, decision = build_model()
    pec = pnl.ParameterEstimationComposition(
        name="group",
        nodes=[comp],
        parameters={
            (name, decision): np.linspace(*FIT_RANGES[name], 1000) for name in FIT_PARAMS
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        fit_method="hierarchical",
        hierarchical_options={
            "subject_id": "subject",
            "max_iterations": args.max_iterations,
            "tol": 1e-3,
            "estep_options": {"xatol": 1e-2, "fatol": 5e-2, "maxiter": 120},
        },
        distributed=args.distributed,
        distributed_options=distributed_options,
    )

    started = time.time()
    results = pec.run()
    print(f"fitted in {time.time() - started:.1f}s")

    report(results, theta_true)


if __name__ == "__main__":
    main()
