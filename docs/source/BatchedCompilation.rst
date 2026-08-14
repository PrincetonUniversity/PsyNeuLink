Experimental Batched Compilation
================================

The batched compiler is an experimental execution path for evaluating many
parameter sets, subjects, trials, and stochastic estimates of a
`Composition` on a GPU.  It is primarily intended for simulations used by a
`ParameterEstimationComposition`.  It is not an `ExecutionMode`: ordinary
``Composition.run`` execution and batched simulation have different input and
result shapes.

Semantic contract
-----------------

Python PsyNeuLink execution is the semantic authority.  An explicitly
requested backend through the direct batched compiler API never falls back to
Python or LLVM.  A supported
configuration must preserve the behavior of every represented component,
port, projection, parameter, scheduler condition, state transition, and
output.  A configuration is rejected when that promise cannot be made; a
recognized component class alone is not sufficient evidence of support.

Capability diagnosis is staged:

``model_supported``
  The requested PsyNeuLink graph and its effective parameter configuration can
  be represented by the semantic IR.

``codegen_ready``
  The selected backend can emit the lowered operations.

``backend_available``
  The required packages and device are available in the current process.

``can_execute``
  All applicable stages above succeeded.

IR layers
---------

``BatchedGraphIR`` is the backend-neutral semantic graph.  Component, port,
projection, parameter, state, and RNG identities belong here.  It records
exact routing plus data-only declarations for scheduler regions, predicates,
predicate dependencies, ordered consideration sets, termination, finished
values, and retained-state reset policies.  Effective implicit scheduler
predicates are snapshotted as well as supported explicit predicates; the
declarations retain neither live PsyNeuLink components nor ``Condition``
objects.  An ``executable`` flag distinguishes a graph whose current operations
preserve all of those semantics from a graph retained only for inspection and
subsequent IR work.  Exactly validated control paths absorbed by existing
operations carry explicit metadata, but generic control/modulation operations
and a complete absorption ledger remain incremental work.  Unrepresented
variants fail closed.  Display names are diagnostic labels, not identity.

Composition lowering assigns nonnegative numeric IDs in deterministic
dependency order and carries them through ``KernelIR`` operations.  Input and
output specs bind exact live ports, projection specs bind both endpoint ports,
and each output owns an explicit flattened result slice.  The compiled plan
keeps a sidecar from those IDs to the live PsyNeuLink objects; generated code
does not use sanitized display names as identity.  These IDs are currently
lowering-local.  A serializable structural fingerprint for reconstructed
distributed models remains future work.

Receiver InputPorts are lowered independently.  Each port gathers only the
projections addressed to its numeric port ID, applies its validated ``SUM`` or
``PRODUCT`` ``LinearCombination``, and is then concatenated in PsyNeuLink
InputPort order for elementwise mechanism execution.  Identity OutputPorts are
split back into explicit slices, so routing and requested output order do not
depend on a node's primary port.  Derived OutputPort functions and external
multi-port inputs remain fail-closed; the latter require a port-keyed public
input-buffer ABI rather than the current node-keyed adapter.  InputPorts that
use default/internal values also remain fail-closed until that ABI can
represent constants explicitly.  Exact duplicate live node names or duplicate
OutputPort names on one owner are rejected temporarily because a few internal
graph lookups are still name-keyed; ordinary sanitized-name collisions and
PsyNeuLink's automatically suffixed names are already distinguished by IDs.

``KernelIR`` is the backend-neutral executable program.  It currently makes
projection, port combination, function, state, trial-loop, and output-store
operations explicit and carries the graph's scheduler, consideration-set,
termination, finished-value, and reset declarations forward.  For the exact
stateless boundary described below, a backend-neutral host planner consumes
only those typed declarations and produces a finite one-trial
``BatchedScheduleTraceSpec``.  Kernel lowering expands the trace into an
executable ``ForPasses`` containing typed ``ExecuteConsiderationSet`` regions,
followed by one output-store epilogue.  Each trial lane executes that same trace
independently, including trial-local predicate and usable-call-count resets.

Pass-wise programs outside that finite stateless boundary remain
declaration-only.  Their ``ForPasses`` attributes describe the required region
but do not authorize sequential execution or define conditional behavior; the
corresponding ``KernelIR`` is non-executable and the Triton emitter refuses it
before generating source.  Accepted one-pass static graphs retain their flat
executable operation sequence.  Parameter, subject, trial, and estimate lane
layout and fusion are lowering and optimization choices; they must not
determine model semantics.  After semantic declarations have been snapshotted,
neither the host trace planner nor a backend emitter interprets the live
Composition, scheduler, components, or ``Condition`` objects.

Current scheduler and control boundary
--------------------------------------

The semantic declarations are intentionally broader than the executable
subset.  The finite precomputed path does not enable stateful pass schedules,
``WhenFinished``-driven pass loops, generic control, or CSI; those remain
fail-closed:

.. list-table::
   :header-rows: 1
   :widths: 18 39 43

   * - Area
     - Represented in IR
     - Current execution boundary
   * - Predicates
     - Exact ``Always``, ``AtTrialStart``, ``AtPass``, and ``WhenFinished``
       predicates are typed.  ``AtPass`` requires one nonnegative integer at
       ``ENVIRONMENT_STATE_UPDATE``.  Effective implicit ``Always``,
       ``EveryNCalls(..., 1)``, and ``AllEveryNCalls(..., 1)`` predicates are
       also snapshotted.
     - One-pass cases execute when they are equivalent to the existing static
       order: ``Always``, first-pass predicates, and ``WhenFinished`` on a
       producer in an earlier consideration set.  Delayed ``AtPass(n)`` also
       executes when the complete graph qualifies for the stateless
       precomputed-trace boundary.  Within such a trace, ``Always`` may execute
       on each pass, ``AtTrialStart`` and ``AtPass`` use absolute pass indices,
       and the implicit one-call predicates consume trial-local usable counts.
       There is no generic or dynamic ``WhenFinished`` executor; only its
       already validated one-pass static equivalent remains admitted.
       Explicit call-count predicates, unsupported condition types or
       subclasses, malformed arguments, structural scheduler conditions, and
       other time scales remain fail-closed.
   * - Consideration sets
     - The scheduler's ordered consideration queue is stored with numeric set
       and component IDs.  Each set declares the PsyNeuLink frozen-input
       contract, and predicate dependencies use those identities rather than
       display-name order.
     - The host planner evaluates the typed sets in order for the stateless
       precomputed subset.  It selects every member of a set from one
       beginning-of-set predicate snapshot, consumes and publishes usable call
       counts between sets, and checks termination between sets.  Empty visits
       are omitted from the trace while absolute pass and consideration-set
       indices are retained.  Dynamic or lane-varying consideration-set
       evaluation is not implemented.
   * - Pass regions
     - Trial and nested pass regions, finished-value identities, and the
       predicate data needed by ``ForPasses`` are explicit.
     - A delayed-``AtPass`` graph gets executable ``ForPasses`` only when it is
       a stateless trial-lane graph with no retained state, RNG stream, reset,
       finished-value dependency, or scheduler component lacking a lowered
       body.  Data projections must run sender-before-receiver in the current
       trial; a receiver that would read a previous trial's held sender value
       is rejected.  Projection edges within one consideration set are also
       rejected until current/next value banks can preserve frozen inputs.
       Nontermination, invalid dependency order, malformed declarations, and
       component- or weighted-operation expansion beyond the compiler budgets
       fail closed.  Other required pass regions remain declaration-only with
       ``KernelIR.executable == False``.
   * - Termination
     - The exact default trial ``AllHaveRun()`` predicate is expanded to every
       lowered scheduler component ID, and the environment-sequence
       ``Never()`` predicate is typed independently from node predicates.
     - All configurations admitted through the current Composition compiler,
       including precomputed traces, require that exact default contract:
       ``AllHaveRun`` must use ``ENVIRONMENT_STATE_UPDATE`` with no explicit
       component operands, and sequence termination must be ``Never``.
       Subsets, custom predicates, nondefault internal time scales, malformed
       maps, and name-equivalent impostors are structured rejections rather
       than silently changing trace length.
   * - Resets
     - Retained graph state has typed reset declarations referencing numeric
       state IDs; the schema distinguishes exact ``Never`` and
       ``AtTrialStart`` policies.  DDM trial-local storage is still private to
       its mechanism operation rather than represented by these reset records.
     - The exact LCA subset currently executes only with ``Never``.  An ordinary
       DDM requires exact ``AtTrialStart`` and resets its trial-local state.  A
       validated fires-once integrating TransferMechanism may fold its
       ``AtTrialStart`` reset into one stateless affine step.  Other reset
       semantics fail closed.
   * - Control
     - Control component identity and its scheduler predicate can be retained
       in a declaration-only graph even when execution is rejected.  Exactly
       absorbed controls record source, target, parameter, and ``OVERRIDE``
       modulation metadata; generic effective-parameter operations are not in
       ``KernelIR`` yet.
     - Executable control is limited to validated absorbed paths, including the
       narrow scalar identity cue to LCA termination-threshold ``OVERRIDE`` and
       the existing affine DDM threshold-collapse chain.  Generic control and
       co-evolving LCA termination control remain fail-closed.
   * - Stepwise DDM
     - A DDM with ``execute_until_finished=False`` is structurally admitted
       only as the typed ``Always`` persistent-stepper / ``WhenFinished``
       terminator pattern, so its required dependencies and finished value can
       be declared.  An orphan stepwise DDM is rejected.
     - Ordinary ``execute_until_finished=True`` DDM execution uses its existing
       bounded inner loop.  The co-evolving stepwise form produces
       declaration-only ``ForPasses`` and is not executable yet.
   * - CSI
     - General constituent pieces already represented include exact port and
       rectangular-matrix routing, the registered nested-logistic drift UDF,
       the narrow deterministic width-two LCA subset, and typed declarations of
       the full scheduler topology.  CSI is not assigned a special model type or
       recognizer.
     - The full surrogate remains unsupported because its co-evolving
       ``Always``/``WhenFinished`` pass execution and controlled LCA finished
       transition are not executable.  Compilation fails explicitly; CSI
       likelihood and fitting support therefore remain a later acceptance
       gate.

Persistent state can be initialized either from typed constants or by applying
a registered elementwise function to an initializer with the lane's effective
parameters.  The latter reuses the same decorated implementation as ordinary
execution; it is needed for recurrent sender state whose PsyNeuLink initial
value is the mechanism function applied to its integrator initializer.

The current LCA implementation is an exact but deliberately narrow semantic
subset: width two, canonical self-excitation/competition matrix, deterministic
zero noise, zero integrator initializer and offset, ``Never`` reset, no clip,
and finite Logistic and recurrent parameters within the fp32 range that are
scalar or exactly uniform broadcasts.  ``time_step_size`` must be strictly
positive.  It supports a
nonnegative ``TimeScale.TRIAL`` execution-count threshold either directly or
through a narrow scalar identity cue -> ``OVERRIDE`` control chain.  Static
thresholds are discretized on the host (ceiling, with at least one execution);
runtime cues must be exact nonnegative integers no larger than ``2**24`` so
fp32 conversion cannot change the step count.  The effective static execution
count has the same bound.  Each node's ``max_executions_before_finished`` must
be a positive integer and is enforced independently.

The identity cue path is absorbed during lowering, so its Linear ``slope``,
``intercept``, ``scale``, and ``offset`` bindings are validated-default-only:
parameter rows may not change them until KernelIR represents and executes the
cue transform itself.  An explicit ``AtPass(0)`` on the cue at the default
``ENVIRONMENT_STATE_UPDATE`` time scale is accepted as the same static-origin
timing.  Other explicit conditions or time scales on the cue, controller, or
controlled LCA are rejected rather than ignored.

Recurrent activation is initialized by applying the registered Logistic
implementation to zero for each parameter lane and persists between trials.
The absorbed cue/control path is a validated instance of general identity
routing, not recognition of stability-flexibility or CSI.  Nonzero noise,
custom state, other termination measures, and generic co-evolving
``Always``/``WhenFinished`` scheduling and control remain structured
rejections rather than approximate execution.

Extension API
-------------

``@batched_op`` and ``@batched_node_op`` are the first-class extension APIs.
They keep Triton implementations next to their declarations and bind function
arguments to PsyNeuLink parameters by signature.  Compilation validates that
every behavior-affecting parameter is bound, restricted to validated values,
or intentionally irrelevant.  Registrations are resolved exactly and frozen
into a plan, so later registry mutations cannot change an existing plan.

Testing contract
----------------

The existing LLVM-enabled tests provide reusable semantic cases and expected
values, but Triton is not added to the global ``comp_mode`` fixture.  A
batched adapter builds independent Python and batched Compositions, expands
the additional batch axes, and compares every lane to Python through exact
component and OutputPort bindings.  LLVM is a secondary oracle for selected
complex deterministic cases.

The executable precomputed-schedule corpus covers delayed senders with
implicit receivers, an ``Always`` sender with a delayed receiver, and delayed
multi-origin fan-in with implicit ``SUM`` combination.  Each case spans
multiple trials, asserts the exact typed trace, and compares both Triton
interpreter and compiled-GPU results directly with a freshly constructed
Python Composition.

Device buffers and current Triton kernels use fp32. Discrete outcomes and
execution counts compare exactly. The exact LCA semantic cases compare
deterministic numeric outputs with ``rtol=1e-5`` and
``atol=1e-6``. This is an fp32 numerical contract, not bitwise preservation of
host fp64 arithmetic for ill-conditioned compositions; the shared corpus will
adopt the same default as existing looser cases are calibrated.
Every Triton run validates its outcome tensor before outcome-buffer host
conversion or device-side likelihood evaluation; NaN or infinite values raise
``BatchedNumericalError`` instead of being returned or scored.

Unsupported variants assert a structured diagnostic and compile rejection;
they are not skipped.  Triton's interpreter and compiled GPU modes run in
separate processes.  Interpreter coverage is useful for semantic iteration,
but a feature is not considered GPU-supported until a compiled-GPU parity
case passes.

The general operating-system and Python-version CI matrix does not install the
Triton extra and selects ``not triton``.  A dedicated Linux job installs the
``dev,triton`` extras, sets ``TRITON_INTERPRET=1`` before Python starts, selects
``triton_interpreter``, and requires that backend so a missing dependency or
incorrect mode fails rather than skips the job.

The repository does not currently declare a CUDA-capable runner label, so CI
does not claim compiled-GPU coverage.  Once managed GPU runner infrastructure
is available, its separate Linux job must leave ``TRITON_INTERPRET`` unset,
install the ``dev,triton`` extras, select ``triton_gpu``, and pass
``--require-batched-backend=triton_gpu``.  The runner must expose a usable CUDA
device; using a generic hosted ``ubuntu-latest`` runner is not a substitute.

CSI is a composition-level acceptance case, not a model kind or compiler
recognizer.  It can become supported only through general implementations of its
functions, mechanisms, ports, scheduler conditions, control projections,
state, and reset behavior.
