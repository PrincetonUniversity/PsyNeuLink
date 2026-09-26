"""Profile DAWA's passive nondecision-time readout using exact compiled counts."""

import numpy as np

from psyneulink.core.batched.dependency import analyze_axis_dependencies
from psyneulink.core.batched.shifted_histogram import ShiftedHistogramScorer


class NDTProfile:
    def __init__(self, plan, names, bounds, data):
        self.index = next(i for i, name in enumerate(names) if name.startswith("RT_GATE.intercept["))
        self.name = names[self.index]
        self.dynamic_names = [name for name in names if name != self.name]
        graph = plan.ir.graph
        gate = next(n for n in graph.nodes if n.name == "RT_GATE")
        response = next(n for n in graph.nodes if n.name == "Response Units\n[Left, Right]")
        incoming = [p for p in graph.projections if p.receiver_component_id == gate.component_id]
        outgoing = [e for e in analyze_axis_dependencies(graph, plan.ir.params).edges
                    if e.producer_component_id == gate.component_id and e.consumer_component_id != gate.component_id
                    and e.kind != "schedule_termination_control"]
        # AllHaveRun observes whether this stateless gate executed, not its RT
        # value. Changing the intercept cannot change that execution condition.
        value_termination = any(gate.component_id in t.dependency_component_ids and t.condition_type != "AllHaveRun"
                                for t in graph.termination)
        fixed = plan.fixed_parameters
        expected = {"RT_GATE.slope": 1., "RT_GATE.scale": 1., "RT_GATE.offset": 0.}
        if (outgoing or value_termination or gate.function_type != "Linear" or len(incoming) != 1
                or incoming[0].sender_component_id != response.component_id
                or incoming[0].sender_port != "DECISION_TIME" or not np.array_equal(incoming[0].matrix, [[1.]])
                or any(fixed.get(k) != v for k, v in expected.items())):
            raise ValueError("NDT profiling requires an unmodulated, passive additive RT readout")
        if any(m.target_component_id == gate.component_id for m in graph.modulations):
            raise ValueError("NDT profiling cannot remove a modulated readout")
        dt = fixed[response.params['time_step_size']]
        self.support = np.arange(plan.ir.max_steps + 1, dtype=np.float32) * np.float32(dt)
        lo, hi, step = bounds[self.name]
        grid = np.linspace(lo, hi, round((hi - lo) / step) + 1)
        self.scorer = ShiftedHistogramScorer(self.support, grid, data, [0], bins=100,
                                             bin_range=[(0., 3.)], smoothing_sigma=.5, categorical_cardinalities=[2])
        self.values = self.scorer.shifts

    def expand(self, candidates, values=0.):
        rows = np.asarray(candidates, dtype=float)
        result = np.empty((len(rows), rows.shape[1] + 1))
        result[:, :self.index] = rows[:, :self.index]
        result[:, self.index] = values
        result[:, self.index + 1:] = rows[:, self.index:]
        return result

    def describe(self):
        return {"parameter": self.name, "grid_values": self.scorer.grid_size,
                "distinct_bin_maps": len(self.values), "representatives": self.values.tolist(),
                "support_values": len(self.support), "support_kind": "exact FP32 LCA count times fixed dt",
                "optimizer_parameters": self.dynamic_names,
                "note": "Profile after pooling blocks, before taking the population ranking; final selection profiles again on fresh draws."}
