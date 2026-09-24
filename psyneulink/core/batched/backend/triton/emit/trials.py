"""Independent progression through ordered trials in a dynamic schedule.

Each lane retains its estimate/subject identity and advances its own trial
index after completion. The existing typed consideration-set executor still
owns all mechanism execution, publications, controller effects and RNG clocks.
"""

from psyneulink.core.batched.kernel_ir import KernelDynamicScheduleProgram


def validate_trial_schedule(value):
    if not isinstance(value, str) or value not in ("synchronized", "independent"):
        raise ValueError("Triton trial_schedule must be 'synchronized' or 'independent'.")
    return value


class IndependentTrialEmitMixin:
    def _emit_trial_initializer(self, variable, expression):
        """Initialize all lanes, or reset only lanes entering their next trial."""
        if self.trial_reset_mask is not None:
            expression = f"tl.where({self.trial_reset_mask}, {expression}, {variable})"
        self.builder.line(f"{variable} = {expression}")

    def _emit_independent_trial_loop(self, body):
        regions = [i for i, op in enumerate(body) if op.kind == "ForPasses"]
        if len(regions) != 1:
            raise ValueError("Independent trials require one typed dynamic ForPasses region.")
        index = regions[0]
        region = body[index]
        program = region.attrs.get("program")
        prefix, suffix = body[:index], body[index + 1:]
        if (type(program) is not KernelDynamicScheduleProgram
                or region.attrs.get("trace_kind") != "lane_local_dynamic"
                or any(op.kind != "ResetState" for op in prefix)
                or any(op.kind not in ("StoreOutput", "StoreFlag") for op in suffix)):
            raise ValueError("Independent trials require a dynamic schedule with trial resets and output stores.")

        self.builder.line("sequence_mask = mask")
        self.builder.line("trial_idx = tl.zeros((BLOCK,), tl.int32)")
        self._emit_params(trial_varying_only=True)
        self._emit_stateful_random_base()
        self._emit_ops(prefix)
        self._emit_trial_start_inspection()
        self.dynamic_single_execution_component_ids = self._dynamic_single_execution_component_ids(program)
        self.dynamic_fuel_bounded_component_ids = self._dynamic_fuel_bounded_component_ids(program)
        carry_vars = self._emit_dynamic_carry_initializers(program)
        slot_vars = self._emit_dynamic_scheduler_initializers(program)
        has_run_bits = self._emit_dynamic_has_run_initializers(program)
        self._emit_dynamic_normal_cache_initializers(program)
        self.dynamic_program = program
        self.dynamic_slot_vars = slot_vars
        self.builder.line("dynamic_done = tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("dynamic_round = tl.zeros((BLOCK,), tl.int32)")
        with self.builder.block("while tl.max(mask.to(tl.int32), 0) > 0"):
            for consideration_set in program.consideration_sets:
                self._emit_dynamic_consideration_set(
                    program, consideration_set, slot_vars, carry_vars, has_run_bits, "dynamic_done")
                self._emit_dynamic_termination(program, has_run_bits, "dynamic_done")
            self.builder.line("dynamic_round += 1")
            self.builder.line(f"trial_complete = mask & ((dynamic_done != 0) | (dynamic_round >= {program.schedule_fuel}))")
            with self.builder.block("if tl.max(trial_complete.to(tl.int32), 0) > 0"):
                # Only completed lanes publish outcomes/diagnostics. Trials
                # with exhausted fuel follow the ordinary truncation contract.
                self.builder.line("mask = trial_complete")
                self._emit_dynamic_exit_diagnostics(program, carry_vars, slot_vars)
                self._emit_trial_output_begin()
                self._emit_ops(suffix)
                self._emit_trial_end_inspection()
                self.builder.line("trial_idx += trial_complete.to(tl.int32)")
                self.builder.line("trial_restart = trial_complete & (trial_idx < num_trials)")
                with self.builder.block("if tl.max(trial_restart.to(tl.int32), 0) > 0"):
                    self.builder.line("mask = trial_restart")
                    self.trial_reset_mask = "trial_restart"
                    self._emit_params(trial_varying_only=True)
                    self._emit_stateful_random_base()
                    self._emit_ops(prefix)
                    self._emit_trial_start_inspection()
                    self._emit_dynamic_carry_initializers(program)
                    self._emit_dynamic_scheduler_initializers(program)
                    self._emit_dynamic_has_run_initializers(program)
                    self._emit_dynamic_normal_cache_initializers(program)
                    self._emit_trial_initializer("dynamic_done", "0")
                    self._emit_trial_initializer("dynamic_round", "0")
                    self.trial_reset_mask = None
                self.builder.line("mask = sequence_mask & (trial_idx < num_trials)")
        # Final-state publication uses the original lane population, including
        # estimates that completed while other estimates were still running.
        self.builder.line("mask = sequence_mask")
        self.dynamic_active_mask = "mask"
        self.dynamic_execution_index = None
        self.dynamic_program = None
        self.dynamic_slot_vars = None
        self.dynamic_single_execution_component_ids = frozenset()
        self.dynamic_fuel_bounded_component_ids = frozenset()
        self.dynamic_normal_cache_vars = {}
        self.builder.line()
