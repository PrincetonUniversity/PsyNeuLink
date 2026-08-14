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
exact routing plus the scheduler, reset, control, and absorption semantics of
the currently supported subset.  Unrepresented variants fail closed.  Generic
scheduler predicates, reset events, control edges, and a complete absorption
ledger remain incremental IR work.  Display names are diagnostic labels, not
identity.

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
operations explicit.  Generic scheduler predicates, conditional pass regions,
and finished/control values are not explicit yet; models that need them are
rejected.  Parameter/subject/trial/estimate lane layout and fusion are lowering
and optimization choices; they must not determine model semantics.  A backend
emitter translates ``KernelIR`` and does not infer new behavior from the live
Composition.

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
