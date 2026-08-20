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
subsequent IR work.  Every lowered Port also has a typed owner, kind, width,
and numeric identity.  A controlled, scheduler-visible effective parameter is
represented by a typed ``BatchedModulationSpec`` containing stable source,
controller, ControlSignal, target ParameterPort, and effective-value IDs.  Its
monitor MappingProjection and ControlProjection are retained in an explicit
absorbed-projection ledger rather than disappearing during lowering.  A
separate effective-parameter declaration records the base value, initial held
ControlProjection value, lane-persistent storage, update after controller
execution, and sampling when the target ParameterPort updates.  The controller
declaration records the decorator-resolved function implementation and exact
argument-to-parameter IDs.  Compatibility name metadata is not the semantic
authority for that edge.  The compiler executes this representation for two
exact dynamic subsets: a scalar-controlled LCA followed by one stateless
``WhenFinished`` consumer, and the CSI research-model co-evolving graph
described below.  Parameters that can change a discrete pass count are frozen
or runtime-validated so host fp64 semantics and device fp32 semantics select
the same count.  Generic modulation, multiple controlled targets, and ledger
coverage for other existing folds remain incremental work.  Unrepresented
variants fail closed.  Display names are diagnostic labels, not identity.

Composition lowering assigns nonnegative numeric IDs in deterministic
dependency order and carries them through ``KernelIR`` operations.  It first
runs PsyNeuLink's ordinary structural analysis so deferred ControlProjections,
CIM routing, and scheduler dependencies match the graph Python execution will
use; lowering the same Composition before and after a Python run produces the
same declarations.  Input and
output specs bind exact live ports, projection specs bind both endpoint ports,
and each node anchors the ordered IDs of its live InputPorts, OutputPorts, and
named ParameterPorts.  Each output owns an explicit flattened result slice.  The compiled plan
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
termination, finished-value, effective-parameter, modulation, absorbed-edge,
Port, and reset declarations forward.  A GraphIR modulation must survive
exactly into KernelIR and form a bijection with one held effective value, its
two absorbed projections, and one matching dynamic finished value.  KernelIR
validates every endpoint's owner, Port kind, width, and registered controller
signature against the frozen plan snapshot.  Executable controlled programs
use explicit ``InitializeEffectiveParameter`` and ``ApplyModulation``
operations.  Their held values live at lane scope across trials, and their
dynamic regions publish only declared results to later operations.  KernelIR
has two authenticated lane-local region forms: ``lane_local_counted`` for the
single controlled-finished chain and ``lane_local_coevolving`` for the exact
CSI research topology.  A partial, forged, or otherwise unsupported
control program remains declaration-only and source emission is forbidden.

For the exact stateless and fixed-count stateful boundaries described below, a
backend-neutral host planner consumes only those typed declarations and
produces a finite one-trial ``BatchedScheduleTraceSpec``.  Kernel lowering
expands the trace into an executable ``ForPasses`` containing typed
``ExecuteConsiderationSet`` regions, followed by one output-store epilogue.
Stateless trial lanes execute that trace independently.  The fixed-count
stateful path instead executes trials serially within each subject/estimate
lane so its supported LCA state persists between trials.  Predicate and usable
call counts remain trial-local in both paths.

Pass-wise programs outside the precomputed, fixed-count, and two authenticated
dynamic boundaries remain declaration-only.  Their ``ForPasses`` attributes
describe the required region but do not authorize sequential execution or
define conditional behavior; the corresponding ``KernelIR`` is
non-executable and the Triton emitter refuses it before generating source.
Accepted one-pass static graphs retain their flat executable operation
sequence.  Parameter, subject, trial, and estimate lane layout and fusion are
lowering and optimization choices; they must not determine model semantics.
After semantic declarations have been snapshotted, neither the host trace
planner nor a backend emitter interprets the live Composition, scheduler,
components, or ``Condition`` objects.

Current scheduler and control boundary
--------------------------------------

The semantic declarations are intentionally broader than the executable
subset.  Executable pass-wise programs currently form four tiers:

* stateless schedules that can be expanded to a finite precomputed trace;
* one stateful LCA with a compile-time fixed count followed by one stateless
  ``WhenFinished`` consumer;
* the same producer/follower shape with one scalar ``OVERRIDE`` control path
  supplying a lane-varying execution count; and
* one exactly authenticated CSI research-model co-evolving topology.

The fixed-count and scalar-controlled producer use an exact, unmodified
``Never`` or ``AtTrialStart`` reset.  ``AtTrialStart`` re-evaluates every
declared state initializer before pass zero, including function-based
initializers using the current parameter lane; ``Never`` preserves state
across trials.  This is generic typed scheduler, finished-value, state, reset,
modulation, and ``KernelIR`` machinery.  LCA and CSI are not assigned model
types: they are the first complete graphs whose registered operations and
scheduler contracts satisfy these exact boundaries.  Additional controlled
targets, finished producers, scheduler shapes, and co-evolving graphs remain
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
       precomputed-trace boundary or when it is the authenticated CSI Task
       Input onset.  Within a precomputed trace, ``Always`` may execute
       on each pass, ``AtTrialStart`` and ``AtPass`` use absolute pass indices,
       and the implicit one-call predicates consume trial-local usable counts.
       A fixed ``WhenFinished`` execution count also runs for the exact
       stateful producer/follower boundary above.  Lane-varying
       ``WhenFinished`` executes only for the exact scalar-controlled
       producer/follower chain and for CSI's ordered LCA, drift, DDM, and gate
       regions.  Explicit call-count predicates, unsupported condition types
       or subclasses, malformed arguments, structural scheduler conditions,
       additional dynamic dependencies, and other time scales remain
       fail-closed.
   * - Consideration sets
     - The scheduler's ordered consideration queue is stored with numeric set
       and component IDs.  Each set declares the PsyNeuLink frozen-input
       contract, and predicate dependencies use those identities rather than
       display-name order.
     - The host planner evaluates the typed sets in order for the supported
       precomputed subsets.  It selects every member of a set from one
       beginning-of-set predicate snapshot, consumes and publishes usable call
       counts between sets, and checks termination between sets.  Empty visits
       are omitted from the trace while absolute pass and consideration-set
       indices are retained.  The authenticated dynamic regions evaluate
       their exact consideration-set order per lane.  In CSI, the absorbed
       threshold controller is in an earlier set than the LCA, so it observes
       the LCA finished transition on the following pass; the later drift and
       DDM sets observe that transition on the same pass.  Other dynamic or
       lane-varying consideration-set evaluation is not implemented.
   * - Pass regions
     - Trial and nested pass regions, finished-value identities, and the
       predicate data needed by ``ForPasses`` are explicit.
     - Outside the exact CSI region, a delayed-``AtPass`` graph gets executable
       ``ForPasses`` only when it is a stateless trial-lane graph with no
       retained state, RNG stream, reset, finished-value dependency, or
       scheduler component lacking a lowered body.  The fixed-count
       LCA/Transfer boundary also gets an executable
       pass region, with a typed one-step state update for each planned LCA
       execution.  Data projections must run sender-before-receiver in the
       current trial; a receiver that would read a previous trial's held sender
       value is rejected.  Projection edges within one consideration set are
       also rejected until current/next value banks can preserve frozen
       inputs.  ``lane_local_counted`` executes a controlled LCA once per pass
       until each lane reaches its held count.  ``lane_local_coevolving``
       executes CSI's LCA on every active outer pass, applies the Task Input's
       fixed integral onset independently, enables drift and DDM at the lane's
       affine controlled-LCA transition, and stops at the lane-local DDM limit.
       Both use a typed integer scheduling clock and the declared
       ``MAX_STEPS`` diagnostic contract.  Nontermination, invalid dependency
       order, malformed declarations, and component- or weighted-operation
       expansion beyond the compiler budgets fail closed.  Other required
       pass regions remain declaration-only with
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
     - The fixed-count and controlled LCA/follower subsets execute exact
       ``Never`` and exact ``AtTrialStart``.  ``AtTrialStart`` lowers to an
       unconditional ``ResetState`` prefix before pass zero and restores all
       of the owner's retained states from their declared initializers;
       ``Never`` emits no per-trial reset and therefore persists.  In the CSI
       region, LCA state persists across trials, DDM value/step/finished state
       resets at trial entry, and the absorbed threshold override remains held
       across trials.  The latter is updated even after a one-step DDM trial,
       matching the scheduler's final threshold-control visit.  KernelIR
       requires one canonical reset declaration per retained-state owner and
       never erases a declared reset effect.  Mutated built-in conditions fail
       closed.  Isolated/static LCA execution with ``AtTrialStart`` remains
       unsupported.  An ordinary DDM requires exact ``AtTrialStart`` and
       resets its private trial-local state.  A validated fires-once
       integrating TransferMechanism may fold its ``AtTrialStart`` reset into
       one stateless affine step.  Other reset semantics fail closed.
   * - Control
     - The scalar controlled-finished boundary has a typed ``OVERRIDE`` edge
       with exact numeric endpoint-port identities, two explicit absorbed
       projection records, a target ParameterPort, and a distinct held
       effective-parameter ID.  The held value records its initial allocation,
       cross-trial persistence, controller-update event, and ParameterPort
       sampling event.  The LCA finished declaration references that effective
       value with explicit ceiling, lower-bound, and integer-range semantics.
       GraphIR and KernelIR validate the full declaration bijection and reject
       forged or erased routes, parameter bindings, and effects.
     - ``InitializeEffectiveParameter`` allocates lane-persistent held storage,
       and ``ApplyModulation`` updates it after the registered controller.
       ``Lane-persistent`` means the value survives trials within one
       parameter/subject/estimate lane and each simulation invocation begins
       from the declared initial modulation value.  The executable dynamic
       subsets require exact ordered source, controller, and target sets and
       an integer-stable scalar count.  PsyNeuLink freezes values at set entry,
       so same-set control is rejected.  The separately validated
       run-to-completion LCA fold and affine DDM threshold-collapse fold also
       remain supported.  When invoked by PEC, lowering may remove only its
       authenticated external fitting controls, reconstruct the original
       scheduler without their ordering edges, and bind candidate values
       directly to runtime parameter lanes.  Intrinsic controls cannot be
       removed this way.  Generic control, more than one scheduler-visible
       intrinsic modulation, and arbitrary co-evolving control topologies fail
       closed.
   * - Stepwise DDM
     - A DDM with ``execute_until_finished=False`` is structurally admitted
       only as the typed ``Always`` persistent-stepper / ``WhenFinished``
       terminator pattern, so its required dependencies and finished value can
       be declared.  An orphan stepwise DDM is rejected.
     - Ordinary ``execute_until_finished=True`` DDM execution uses its existing
       bounded inner loop.  The CSI DDM executes one typed step per active
       co-evolving pass, with a lane-local execution index for collapse and RNG
       addressing, a private trial-state tuple, and a bounded truncation flag.
       Other co-evolving stepwise forms remain declaration-only.
   * - CSI
     - General constituent pieces already represented include exact port and
       rectangular-matrix routing, the registered nested-logistic drift UDF,
       the narrow numeric-noise width-two LCA subset, the DDM threshold-control
       fold, one declared DDM RNG stream, two finished values, and the full
       six-set scheduler topology.  CSI is not assigned a public model type or
       selected by display-name dispatch; admission authenticates the complete
       typed graph.
     - The research surrogate executes through one
       ``lane_local_coevolving`` region when the complete 11-node/six-set graph,
       resets, functions, matrices, controls, Ports, and scheduler conditions
       match the authenticated boundary.  Its cue count is
       ``csi_switch * cue + csi_repeat``: runtime slope/switch and
       intercept/repeat values and the resulting count must be finite,
       nonnegative exact integers no larger than ``2**24``; Linear scale and
       offset remain fixed at one and zero.  ITI is a fixed finite nonnegative
       integer no larger than ``2**24`` and must agree across Task Input
       ``AtPass``, the controller, and region metadata.  Numeric scalar or
       uniform-broadcast LCA noise is runtime mutable.  Finite nonnegative DDM
       noise may be nonzero but is fixed when the plan is compiled and uses a
       lane-local Philox stream.
       Runtime lanes also cover LCA gain, DDM non-decision time, and folded
       threshold/collapse.  The reduced three-parameter and full historical
       five-parameter PEC objectives compile and score by ignoring only
       authenticated external PEC controls.  Distribution/function-valued LCA
       noise, runtime-mutable DDM noise, changed topology, and other
       co-evolving graphs fail closed.

Persistent state can be initialized either from typed constants or by applying
a registered elementwise function to an initializer with the lane's effective
parameters.  The latter reuses the same decorated implementation as ordinary
execution; it is needed for recurrent sender state whose PsyNeuLink initial
value is the mechanism function applied to its integrator initializer.

The current LCA implementation is an exact but deliberately narrow semantic
subset: width two, a canonical self-excitation/competition matrix, zero
integrator initializer and offset, authenticated ``Never``/``AtTrialStart``
reset behavior, no clip, and finite Logistic and recurrent parameters within
the fp32 range that are scalar or exactly uniform broadcasts.
``time_step_size`` must be strictly positive.  LCA noise may be any finite
numeric fp32 scalar or exactly uniform broadcast, including negative values,
and is a runtime parameter.  It follows PsyNeuLink's deterministic numeric
semantics: the same ``noise * sqrt(time_step_size)`` term is added on every
integration step, so no RNG stream is declared.  The mechanism and integrator
numeric noise values must agree when the graph is compiled.  Callable,
distribution-valued, nonfinite, and non-broadcast noise fail closed.  The LCA
supports a
nonnegative ``TimeScale.TRIAL`` execution-count threshold either directly or
through a narrow scalar cue -> ``OVERRIDE`` control chain.  Static thresholds
are discretized on the host (ceiling, with at least one execution).  Dynamic
cue values and controller transforms must satisfy the admitted integer-stable
contract and remain no larger than ``2**24`` so fp32 conversion cannot change
the pass count.  The effective static execution count has the same bound.
Each node's ``max_executions_before_finished`` must be a positive integer and
is enforced independently.

There are three executable cue/control lowerings.  The atomic
run-to-completion fold absorbs an exact identity chain and validates its
Linear ``slope``, ``intercept``, ``scale``, and ``offset`` bindings at their
defaults.  The scheduler-visible path retains a
``BatchedModulationSpec``, emits the registered cue/controller computation,
updates a held effective count, and runs ``lane_local_counted``; its accepted
Linear transforms are frozen and integer-preserving.  The exact CSI
``lane_local_coevolving`` form keeps the cue source's ``slope`` and
``intercept`` runtime mutable, fixes source ``scale=1`` and ``offset=0``, and
authenticates the controller as an integer-preserving Linear whose intercept
equals fixed ITI.  Source coefficients and every resulting count are finite,
nonnegative exact integers no larger than ``2**24``.  Explicit conditions and
time scales are accepted only where they match the relevant authenticated
scheduler shape; other forms are rejected rather than ignored.

Recurrent activation is initialized by the registered Logistic implementation
for each parameter lane.  A never-reset numeric-noise LCA reproduces
PsyNeuLink's construction-time ``Logistic(noise * sqrt(time_step_size))``
sender value before its first real update; reset policies use their declared
initializer semantics.  CSI's LCA state persists between trials.
The absorbed and typed cue/control paths are validated instances of general
routing and modulation records, not recognition of stability-flexibility or
CSI.  Custom state, other termination measures, multiple dynamic targets, and
otherwise general co-evolving
``Always``/``WhenFinished`` scheduling remain structured rejections rather
than approximate execution.

CSI research-model execution
----------------------------

The first executable co-evolving graph is the CSI surrogate.  Its
authenticated ``BatchedGraphIR`` contains exactly eleven lowered nodes and six
ordered consideration sets: four trial origins; the absorbed threshold and CSI
controllers; the persistent LCA; the registered drift UDF; the DDM; and the two
result gates.  One finished value is the controlled LCA execution count and the
other is the DDM's dynamic finished flag.  The graph is admitted only when
every node implementation, parameter binding, Port, projection matrix, reset,
condition, control route, state declaration, and output slice matches that
complete topology.  Similar component classes or names are not sufficient.

Kernel lowering allocates persistent LCA state, the held LCA count, and the
held DDM-threshold override before ``ForTrials``.  Each trial computes its
origins and affine CSI control, then enters one ``lane_local_coevolving``
``ForPasses`` region.  The LCA steps on every active outer pass.  For an affine
cue count, the DDM begins at pass
``max(1, ceil(iti + csi_switch * cue + csi_repeat)) - 1``; Task Input remains
independently gated until ``AtPass(iti)``.  This preserves the Python scheduler
edge case in which ITI 10 and zero CSI starts the DDM on pass 9, one pass before
Task Input fires.  The DDM uses a lane-local integer execution step for
threshold collapse and RNG addressing; only DDM executions count against
``MAX_STEPS``.  DDM trial state resets each trial, while LCA state and the last
threshold override persist.  The decision and response gates consume the
region's declared outputs after DDM completion.

The runtime parameter surface includes LCA gain and numeric noise, cue
switch/slope and repeat/intercept, DDM non-decision time, folded starting
threshold, and folded threshold collapse.  Count coefficients and results obey
the exact-integer fp32 bound described above; starting threshold is finite and
nonnegative, and collapse is finite and nonpositive.  DDM noise may be any
finite nonnegative value but is fixed when the plan is compiled.  A nonzero
value uses one declared Philox stream per lane, with seeded replay, optional
common random numbers across parameter rows, and offsets independent of ITI
and safety caps.  Numeric LCA noise remains deterministic and runtime mutable;
distribution/function-valued LCA noise is unsupported.

The node-qualified runtime names used by the unsuffixed CSI builder are:

.. code-block:: python

   {
       "Task Activations [C1, C2].gain": 10.0,
       "Task Activations [C1, C2].noise": 0.05,
       "Cue Stimulus Interval.slope": 10.0,       # csi_switch
       "Cue Stimulus Interval.intercept": 0.0,    # csi_repeat
       "DDM.non_decision_time": 0.2,
       "Threshold Mechanism.intercept": 0.12,
       "Threshold Mechanism.offset-integrator_function": -0.001,
   }

PsyNeuLink's ``-N`` rebuild suffixes are accepted through generic
node-parameter aliases.  ``DDM.noise`` is not runtime mutable on this boundary.

``PECOptimizationFunction(batched_backend=...)`` compiles this graph once and
passes optimizer candidates as parameter rows.  PEC's injected fitting
``ControlMechanism`` nodes are host-side parameter carriers, not part of the
device model: lowering may ignore only controls authenticated as external PEC
parameter controls and reconstructs the original six-set scheduler without
their edges.  Intrinsic CSI controls remain required.  Both the reduced
three-parameter objective (gain, switch, non-decision time) and the full
historical five-parameter objective (adding Threshold Mechanism intercept and
integrator offset/collapse) compile and score.  The recovery script uses the
five-parameter surface.

Acceptance tests compare fresh Python with Triton interpreter and compiled CUDA
for zero and delayed ITI, repeat/switch affine transforms, mixed cue trials,
boundary transitions, persistent threshold state, numeric LCA-noise parameter
lanes, and both PEC surfaces.  Stochastic DDM tests cover deterministic replay,
common-random-number layout, and cap/onset-independent draws in both backends.
The region publishes the ordinary bounded-loop truncation diagnostic.
Runtime-mutable DDM noise, callable/distribution-valued LCA noise, a missing
registered ``Drift Rate Value`` UDF, altered 11-node/six-set topology, and other
co-evolving graphs are rejected rather than approximated.

Implementation map
------------------

Read the implementation in semantic order:

* ``psyneulink/core/batched/compiler.py``: public diagnosis, compilation, plan
  execution, and no-fallback behavior.

* ``psyneulink/core/batched/ir.py`` and
  ``psyneulink/core/batched/graph.py``: graph-level declarations and extraction
  from a live Composition.  Start with graph lowering, then the exact
  executable-boundary predicates; CSI admission is
  ``_dynamic_controlled_coevolving_graph_eligible``.

* ``psyneulink/core/batched/kernel_ir.py``: backend-neutral operations,
  complete validation, dynamic-region contracts, and GraphIR-to-KernelIR
  lowering.  Search for ``lane_local_counted`` and
  ``lane_local_coevolving`` to follow the two executable dynamic paths.

* ``psyneulink/core/batched/backend/triton/emit/``: Triton source generation.
  ``ops.py`` emits individual operations and the two lane-local regions;
  ``emitter.py`` owns values, state, loops, and source assembly.

* ``psyneulink/core/batched/components/``: registered mechanism semantics.
  ``lca.py`` contains the width-two finite-numeric-noise LCA step, and
  ``ddm.py`` contains ordinary and co-evolving DDM updates.

* ``psyneulink/core/batched/prep.py`` and
  ``psyneulink/core/batched/backend/triton/api.py``: public input/parameter
  normalization, subject/trial lane validation, device launch, result shaping,
  and diagnostics.

* ``tests/composition/pec/test_batched_controlled_finished_acceptance.py`` and
  ``tests/composition/pec/test_batched_csi_coevolving_acceptance.py``: the most
  direct executable specifications for controlled lane-local counts and the
  complete CSI boundary.  Numeric LCA noise is covered in
  ``test_batched_lca_numeric_noise.py``.  Read the structural assertions before
  the Python, interpreter, and GPU parity cases.

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

The fixed-count ``WhenFinished`` matrix exercises persistent LCA state,
renamed components with reverse insertion order, one execution, a host-fp64
threshold just above one that resolves to two executions, and a fractional
threshold that resolves to three.  Every case checks its typed graph and
``KernelIR`` trace, then compares fresh Python, Triton-interpreter, and
compiled-GPU results.  A companion reset matrix compares exact
``AtTrialStart`` and ``Never`` over three trials, including renamed/reordered
components, a newline-bearing diagnostic label, and nondefault Logistic
parameters.  It also verifies that two parameter rows and two estimate lanes
re-evaluate the ``AtTrialStart``
function initializer independently.  Isolated/static ``AtTrialStart``, an
LCA-to-DDM dependency outside the authenticated CSI graph, unsupported
lane-varying control shapes, multiple controlled producers, and custom
termination remain structured compile rejections.

The controlled-finished matrix covers lane counts one through four, exact
``AtTrialStart`` and ``Never`` state lifetime, multi-subject lanes, strict
``MAX_STEPS`` truncation, structured-IR authenticity mutations, and fresh
Python parity in interpreter and GPU modes.  CSI acceptance separately checks
the complete graph and ``lane_local_coevolving`` region, the canonical
two-trial decision/response oracle, zero-boundary threshold collapse,
cross-trial threshold persistence, one-step cleanup, delayed ITI, affine
repeat/switch counts, runtime gain/switch/repeat/non-decision-time/threshold/
collapse lanes, reduced three-parameter and full five-parameter PEC objectives,
stochastic DDM replay/common-random-number behavior, and compiled CUDA output.
The numeric-LCA-noise matrix separately checks positive, negative, near-zero,
and uniform-broadcast values, runtime noise lanes, initialization semantics,
standalone LCA parity, and CSI parity on interpreter and GPU.

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
recognizer.  The exact research boundary is supported because its functions,
mechanisms, ports, scheduler conditions, control projections, state, and reset
behavior are represented and authenticated end to end.  A new CSI variant is
supported only when those same semantic checks cover it; similarity to the
accepted graph is not sufficient.
