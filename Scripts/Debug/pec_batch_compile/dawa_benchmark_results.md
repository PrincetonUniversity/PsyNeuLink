Measured on 2026-09-23 under WSL2 with an NVIDIA RTX 2080 Ti (11 GB) and an
Intel Core i7-9700K using all eight CPU threads. The fit scripts specify 10,000
estimates. The data contain 760 retained trials per subject after filtering
missing previous-congruency values.

| Retained trials | Estimates | LLVM, 8 CPU threads | Triton, 2080 Ti | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1,000 | 1.738 s | 0.068 s | 25.6× |
| 64 | 10,000 | 15.587 s | 0.072 s | 215.6× |
| 760 | 1,000 | 19.329 s | 0.363 s | 53.2× |
| 760 | 10,000 | 178.651 s | 0.423 s | 422.6× |

These are simulation timings for one parameter candidate and one subject.
Numbers are medians of three warmed calls, except the full 760-trial, 10,000-estimate
LLVM row, which is one measured warmed call after an initial call. Both calls
actually simulated the full workload; no timing is extrapolated. Raw durations,
first-call times, software versions, input-file hash, and output summaries are
in `dawa_benchmark_results.json`.

Each backend/case ran in a fresh process, and timed cases ran serially. Setup and
JIT compilation are excluded from warmed timings. The timed region includes
input preparation, simulation, and returning decision/RT arrays to host memory.
LLVM uses PEC's actual threaded `grid_evaluate` path, not a Python loop over
estimates. Triton uses the ordinary batched compiler on the same PEC-wrapped
model, with PEC's fitting controls represented as runtime parameters. Native
LLVM produces float64 outputs; Triton produces float32 outputs. Density
estimation and optimization are excluded: LLVM's objective callback returns a
constant, while both paths return all simulated outcomes.

The inputs are subject 1 from `dawa_lca_model/flanker_data_part1.csv`, in original
order, filtered exactly as in the supplied fit script. The 64-trial cases use
the beginning of that sequence. Parameters are threshold 0.3, nondecision time
0.2, bias -0.45, control gain 10, LC mode 0.9, LC slope 1, and LC intercept 5.
These seven parameters are constant over the benchmark sequence. This measures
one representative candidate, not an optimizer run or the full multi-subject
conditional fit. Runtime will vary with candidate parameters, especially the
response threshold.

Both paths used the `recurrent` scheduler adjustment described in
`dawa_batched_README.md`. That schedule is now the default in the shared model
builder and driver; it fixes the former stall for multi-pass responses. The
recorded timings and source hash describe the benchmark before this default
change, with the same recurrent execution behavior. All GPU runs used strict
truncation checking with a 2,000-pass cap;
none truncated. Each backend reproduces its seeded repeated samples exactly.
GPU and LLVM random streams differ, so their stochastic samples are compared
as distributions rather than matched draws.

For 760 trials and 10,000 estimates, mean RT is 0.973180 s on Triton
and 0.973245 s on LLVM. Their difference is
0.064 ms; the combined Monte Carlo standard
error, treating each simulated trial sequence as an independent estimate, is
0.120 ms. The choice-1 fractions are 0.498731
and 0.498695, respectively.

There is a numerical validation caveat: with response noise disabled, LLVM
returns 0.97 s for the first retained real-data trial, while Python and Triton
return 0.96 s. Triton's first-trial LCA/LC states match Python within 8e-8;
the next three trial RTs and all four choices agree with LLVM. The one-step
LLVM discrepancy also occurs outside PEC and remains unresolved. These timing
results do not establish exact backend equivalence.

To reproduce the full-subject GPU case:

```bash
.venv/bin/python Scripts/Debug/pec_batch_compile/dawa_llvm_benchmark.py \
  --backend triton --trials 760 --estimates 10000 --repeats 3 \
  --output /tmp/dawa_triton_760_10000.json
```

Use `--backend llvm --repeats 1` for the measured full-subject LLVM case, and
`--trials 64` or `--estimates 1000` for the smaller cases. `--deterministic`
disables response noise for numerical checks. The reusable benchmark writes
progress and a JSON report after every completed run.
