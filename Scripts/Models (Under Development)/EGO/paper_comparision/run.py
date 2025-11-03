import run_original as run
import utils

def main():
    # Set up the parameters for the experiment.
    params = utils.Map(
        n_participants=1,
        state_d=11,  # dimensionality of the state input
        context_d=11,  # dimensionality of the learned context representations
        output_d=11,  # dimensionality of the output layer
        episodic_lr=1,  # learning rate for the episodic pathway
        persistance=-0.8,  # bias towards memory retention in the recurrent context module
        temperature=.1,  # temperature for EM retrieval (lower is more argmax-like)
        n_optimization_steps=10,  # number of optimization steps to take for each state
        seed=0,  # random seed for reproducibility
        memory_init=.01,
        memory_len=1000,
        softmax_threshold=1e-3
    )
    # Set up the parameters for the experiment.
    params.n_participants = 1
    params.paradigms = ['blocked', 'interleaved']
    params.sim_thresh = 0.7  # filtering criterion that will be useful later.
    df, _, context_reps, _ = run.run_experiment(params)
    fig = utils.plot_results(df, 'Deterministic CSW')
    fig.show()


if __name__ == '__main__':
    main()
