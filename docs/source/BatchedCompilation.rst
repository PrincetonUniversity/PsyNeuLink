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
requested batched backend never falls back to Python or LLVM.  A supported
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
exact routing, scheduler and reset semantics, control/modulation edges, and
every graph element that lowering intentionally absorbs.  Display names are
diagnostic labels, not identity.

``KernelIR`` is the backend-neutral executable program.  It makes projection,
port combination, function, state, condition, loop, and output-store
operations explicit.  Parameter/subject/trial/estimate lane layout and fusion
are lowering and optimization choices; they must not determine model
semantics.  A backend emitter translates ``KernelIR`` and does not infer new
behavior from the live Composition.

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
recognizer.  It is supported only through general implementations of its
functions, mechanisms, ports, scheduler conditions, control projections,
state, and reset behavior.
