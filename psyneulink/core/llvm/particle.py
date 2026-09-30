"""Resumable, independent LLVM simulation lanes for sequential inference."""

import concurrent.futures
import ctypes

import numpy as np

from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals import get_num_threads
from .execution import CompExecution
from .scheduler import ConditionGenerator


def _random_state_paths(component, dtype, context, prefix=()):
    """Find RNG storage from the same component tree that defines LLVM state."""
    ids = component.llvm_state_ids
    if dtype.names is None or len(dtype.names) != len(ids):
        raise ValueError(f"LLVM state layout does not match {component.name!r}.")
    for field, attribute in zip(dtype.names, ids):
        path = (*prefix, field)
        field_dtype = dtype.fields[field][0]
        if attribute == "random_state":
            yield path
            continue
        if attribute == "nodes":
            children = component._all_nodes
        elif attribute == "projections":
            children = component._inner_projections
        elif attribute == "_parameter_ports":
            children = component._parameter_ports
        elif attribute in {"input_ports", "output_ports"}:
            children = getattr(component, attribute)
        else:
            parameter = getattr(component.parameters, attribute, None)
            value = None if parameter is None else parameter._get(context)
            if hasattr(value, "llvm_state_ids"):
                yield from _random_state_paths(value, field_dtype, context, path)
            continue
        children = tuple(children)
        if len(field_dtype.names or ()) != len(children):
            raise ValueError(
                f"Nested LLVM state layout does not match {component.name}.{attribute}."
            )
        for name, child in zip(field_dtype.names or (), children):
            yield from _random_state_paths(
                child, field_dtype.fields[name][0], context, (*path, name)
            )


def _field(array, path):
    for name in path:
        array = array[name]
    return array


class ParticleExecution(CompExecution):
    """Advance a population by one trial, retaining full simulation state.

    Use as a context manager. The composition must be its controller's agent
    representation, with a fixed search grid and ``num_trials_per_estimate=1``.
    Parameters and the grid are frozen at construction. Each lane owns state,
    recurrent output data and the entire nested scheduler tree. Closing the
    session releases its worker pool; it never writes simulated values back
    to the live Python composition.
    """

    def __init__(self, composition, context, num_particles):
        super().__init__(composition, context)
        if (
            isinstance(num_particles, bool)
            or not isinstance(num_particles, (int, np.integer))
            or num_particles < 1
        ):
            raise ValueError("num_particles must be a positive integer.")
        ocm = composition.controller
        if (
            ocm.agent_rep is not composition
            or ocm.parameters.num_trials_per_estimate._get(context) != 1
        ):
            raise ValueError(
                "Particle execution requires the controller's agent representation and one trial per estimate."
            )
        self.num_particles = int(num_particles)
        self._particle_binary = pnlvm.LLVMBinaryFunction.from_obj(
            ocm,
            tags=frozenset(
                {"evaluate", "alloc_range", "evaluate_type_all_results", "particle"}
            ),
            ctype_ptr_args=(5,),
            dynamic_size_args=(1, 4, 6, 8),
        )
        self.params = self._get_compilation_param(
            "_particle_params", "_get_param_initializer", 0
        )
        base_state = self._get_compilation_param(
            "_particle_state", "_get_state_initializer", 1
        )
        base_data = self._get_compilation_param(
            "_particle_data", "_get_data_initializer", 6
        )
        self.states = np.repeat(np.atleast_1d(base_state), self.num_particles, axis=0)
        self.data = np.repeat(np.atleast_1d(base_data), self.num_particles, axis=0)
        binary = self._particle_binary
        initial = binary.byref_arg_types[8](
            *ConditionGenerator(None, composition).get_condition_initializer()
        )
        self.conditions = np.repeat(
            np.frombuffer(initial, dtype=binary.np_arg_dtypes[8], count=1),
            self.num_particles,
            axis=0,
        )
        self._rng_paths = tuple(
            _random_state_paths(composition, self.states.dtype, context)
        )
        self._outputs = binary.np_buffer_for_arg(
            4, extra_dimensions=(self.num_particles, 1)
        )
        self._jobs = min(get_num_threads(), self.num_particles)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._jobs)
        self._closed = False

    @property
    def _bin_func(self):
        return self._particle_binary

    def advance(self, inputs):
        """Return a copy of the predictive outcomes, shaped [particle, output]."""
        if self._closed:
            raise RuntimeError("Particle execution session is closed.")
        binary = self._particle_binary
        ct_inputs = self._get_run_input_struct(inputs, 1, arg=5)
        input_arg = ctypes.cast(ct_inputs, binary.c_func.argtypes[5])
        outputs = self._outputs.reshape(-1, *binary.np_arg_dtypes[4].shape)
        self._outputs[...] = np.nan
        per_job = (self.num_particles + self._jobs - 1) // self._jobs
        futures = [
            self._executor.submit(
                binary,
                self.params,
                self.states,
                start,
                min(start + per_job, self.num_particles),
                outputs,
                input_arg,
                self.data,
                np.asarray(1, dtype=np.uint32),
                self.conditions,
            )
            for start in range(0, self.num_particles, per_job)
        ]
        # Wait for every writer before propagating an exception or resampling.
        concurrent.futures.wait(futures)
        for future in futures:
            future.result()
        dtype = self._outputs.dtype
        while dtype.names is not None or dtype.subdtype is not None:
            dtype = (
                dtype.fields[dtype.names[0]][0] if dtype.names else dtype.subdtype[0]
            )
        outcomes = self._outputs.view(dtype).reshape(self.num_particles, -1).copy()
        if not np.isfinite(outcomes).all():
            raise ValueError(
                "Particle simulation returned nonfinite outcomes or ended before the requested trial."
            )
        return outcomes

    def resample(self, ancestors):
        """Gather complete histories; keep each destination's advanced RNGs."""
        ancestors = np.asarray(ancestors)
        if (
            ancestors.shape != (self.num_particles,)
            or not np.issubdtype(ancestors.dtype, np.integer)
            or np.any(ancestors < 0)
            or np.any(ancestors >= self.num_particles)
        ):
            raise ValueError(
                "ancestors must contain one valid integer index per particle."
            )
        states = self.states[ancestors].copy()
        for path in self._rng_paths:
            np.copyto(_field(states, path), _field(self.states, path))
        self.states = states
        self.data = self.data[ancestors].copy()
        self.conditions = self.conditions[ancestors].copy()

    def close(self):
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
