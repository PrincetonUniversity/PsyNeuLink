Draft review of [PR #3642 — Add Neural Likelihood Estimation to PEC](https://github.com/PrincetonUniversity/PsyNeuLink/pull/3642).

Reviewed head `6dc48225505f80e1c262c65869d5f824a68d3ecf` against `devel` at `5cf4270d7006320ae82e155e8d45d2a68b8f9bdd`. Recommendation: **request changes**. This review is local; nothing has been posted.

1. **[P1] Use the supported builder arguments for continuous-only outcomes.**

   Location: [`neurallikelihoodfunctions.py:287`](https://github.com/PrincetonUniversity/PsyNeuLink/blob/6dc48225505f80e1c262c65869d5f824a68d3ecf/psyneulink/core/components/functions/nonstateful/neurallikelihoodfunctions.py#L283-L287).

   With the supported `sbi==0.27.0`, `likelihood_nn()` returns a builder whose signature is `build_fn(batch_theta, batch_x)`. Passing `batch_y=cond` therefore raises `TypeError: ... got an unexpected keyword argument 'batch_y'`. Every training run with no categorical outcomes fails after generating its simulations, including a DDM fit using response time alone. Loading a continuous-only artifact would reach the same call. Pass the conditioning data as `batch_theta=cond`, and cover continuous-only training and save/load in a regression test.

   Reproduced through `pnl.train_neural_likelihood(..., outcome_names=('response_time',), categorical=[False])` on an actual DDM PEC.

2. **[P2] Canonicalize InputPort keys before ordering trial features.**

   Location: [`neurallikelihoodfunctions.py:307–309`](https://github.com/PrincetonUniversity/PsyNeuLink/blob/6dc48225505f80e1c262c65869d5f824a68d3ecf/psyneulink/core/components/functions/nonstateful/neurallikelihoodfunctions.py#L307-L309).

   PEC accepts inputs keyed by `InputPort`, but `position` contains only mechanism/composition nodes. All port keys consequently receive the same `-1` sort key, preserving dictionary insertion order. Reversing an otherwise identical dictionary swaps the network's feature columns. When both inputs varied in training, metadata validation accepts the swapped columns and the estimator silently scores the wrong conditions. Resolve port keys to their model node and port order before extracting features, consistently with PEC's input normalization.

   Reproduced by training a DDM with two varying port inputs and then scoring the same data and parameters with the input dictionary reversed: the neural scores were `-10.601860046386719` and `-10.251859664916992`. Separately confirmed that PEC simulation results were identical under this dictionary reordering.

3. **[P2] Detect constant inputs without testing floating-point standard deviation against zero.**

   Location: [`neurallikelihoodfunctions.py:378–380`](https://github.com/PrincetonUniversity/PsyNeuLink/blob/6dc48225505f80e1c262c65869d5f824a68d3ecf/psyneulink/core/components/functions/nonstateful/neurallikelihoodfunctions.py#L378-L380).

   Identical floating-point values can produce a small nonzero standard deviation: `np.full((10, 1), 0.3).std(axis=0)` is approximately `5.55e-17`. This marks a constant input as a varying trial feature and omits it from `constant_inputs`. A later fit can then change that input without triggering the promised constant-input check, evaluating an estimator that never learned that dependency. Compare each column's values with its first value, or use another constancy check that is insensitive to rounding in the mean.

   Reproduced by training with input `0.3` on every trial: metadata recorded that column as varying, and scoring trials with input `0.6` succeeded instead of rejecting them.

4. **[P2] Flatten heterogeneous model inputs before converting them to a numeric array.**

   Location: [`neurallikelihoodfunctions.py:310–311`](https://github.com/PrincetonUniversity/PsyNeuLink/blob/6dc48225505f80e1c262c65869d5f824a68d3ecf/psyneulink/core/components/functions/nonstateful/neurallikelihoodfunctions.py#L310-L311).

   The supported `{model: trial_inputs}` format is ragged when input nodes have different vector lengths. Converting the entire value with `np.asarray(value, dtype=float)` raises `ValueError` before the per-trial flattening can happen. Consequently, inputs that already work with PEC cannot be used to train or score a neural likelihood. Flatten the individual node/port inputs within each trial and then stack the equally sized trial rows.

   Reproduced with a DDM driven by a scalar input node and a two-element input node: ordinary `pec.log_likelihood(..., return_sim_data=True)` returned simulations of shape `(10, 5, 2)`, while `train_neural_likelihood()` with the same PEC and inputs failed at line 310. The existing `test_pec_run_input_formats[pec_good]` also demonstrates this supported input format.

5. **[P2] Apply the training seed to network initialization.**

   Location: [`neurallikelihoodfunctions.py:427–431`](https://github.com/PrincetonUniversity/PsyNeuLink/blob/6dc48225505f80e1c262c65869d5f824a68d3ecf/psyneulink/core/components/functions/nonstateful/neurallikelihoodfunctions.py#L427-L431).

   The local `torch.Generator` controls the held-out split and minibatch ordering, but `_build_estimator()` initializes weights using PyTorch's global RNG. The public `seed` argument therefore does not seed training as documented: repeated calls with the same seed and identical simulated data produce different estimators and likelihoods. Seed network construction too, preferably within a scoped RNG context so training does not alter the caller's random stream.

   Reproduced with two calls to `train_neural_likelihood(..., seed=17)`: the saved simulation and conditioning probes matched exactly, but network weights and held-out losses differed.

Validation: **206 existing tests passed** on the exact PR head, using Python 3.11.12, `sbi 0.27.0`, and CPU PyTorch 2.14.1. The neural and hierarchical suites passed 176 tests; the Dask suite passed 30. `git diff --check` also passed. I did not run the entire repository suite.

The test commands, run from `/tmp/pnl-pr-3642-review` with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, were:

```bash
/tmp/pnl-pr3642-venv/bin/python -m pytest tests/composition/pec/test_neural_likelihood.py tests/composition/pec/test_pec_hierarchical.py -o addopts='' -q --tb=short
/tmp/pnl-pr3642-venv/bin/python -m pytest tests/composition/pec/test_pec_dask.py -o addopts='' -q --tb=short
```

The five supplemental reproductions are available locally in [pnl-pr3642-review-repro.py](/tmp/pnl-pr3642-review-repro.py), with output in [pnl-pr3642-review-repro.log](/tmp/pnl-pr3642-review-repro.log). Run the script with the same interpreter and thread settings above. It catches the expected exceptions so all five cases can execute. Existing test output is in [pnl-pr3642-tests.log](/tmp/pnl-pr3642-tests.log) and [pnl-pr3642-dask-tests.log](/tmp/pnl-pr3642-dask-tests.log).
