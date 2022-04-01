# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Scheduler **************************************************************
import copy
import typing

import graph_scheduler
import pint

from psyneulink import _unit_registry
from psyneulink.core.globals.context import Context, handle_external_context
from psyneulink.core.globals.json import JSONDumpable
from psyneulink.core.globals.utilities import parse_valid_identifier
from psyneulink.core.scheduling.condition import _create_as_pnl_condition

__all__ = [
    'Scheduler', 'SchedulingMode'
]


SchedulingMode = graph_scheduler.scheduler.SchedulingMode


class Scheduler(graph_scheduler.Scheduler, JSONDumpable):
    def __init__(
        self,
        composition=None,
        graph=None,
        conditions=None,
        termination_conds=None,
        default_execution_id=None,
        mode: SchedulingMode = SchedulingMode.STANDARD,
        default_absolute_time_unit: typing.Union[str, pint.Quantity] = 1 * _unit_registry.ms,
        **kwargs
    ):
        """
        :param composition: (Composition) - the Composition this scheduler is scheduling for
        """

        if composition is not None:
            # dependency dict
            graph = composition.graph_processing.prune_feedback_edges()[0]
            if default_execution_id is None:
                default_execution_id = composition.default_execution_id

        # TODO: consider integrating something like this into graph-scheduler?
        self._user_specified_conds = copy.copy(conditions)

        super().__init__(
            graph=graph,
            conditions=conditions,
            termination_conds=termination_conds,
            default_execution_id=default_execution_id,
            mode=mode,
            default_absolute_time_unit=default_absolute_time_unit,
            **kwargs,
        )

        def replace_term_conds(term_conds):
            return {
                ts: _create_as_pnl_condition(cond) for ts, cond in term_conds.items()
            }

        self.default_termination_conds = replace_term_conds(self.default_termination_conds)
        self.termination_conds = replace_term_conds(self.termination_conds)

    def add_condition(self, owner, condition):
        super().add_condition(owner, _create_as_pnl_condition(condition))

    def add_condition_set(self, conditions):
        try:
            conditions = conditions.conditions
        except AttributeError:
            pass

        super().add_condition_set({
            node: _create_as_pnl_condition(cond)
            for node, cond in conditions.items()
        })

    @handle_external_context(fallback_default=True)
    def run(
        self,
        termination_conds=None,
        context=None,
        base_context=Context(execution_id=None),
        skip_trial_time_increment=False,
    ):
        yield from super().run(
            termination_conds=termination_conds,
            context=context,
            execution_id=context.execution_id,
            base_execution_id=base_context.execution_id,
            skip_environment_state_update_time_increment=skip_trial_time_increment,
        )

    def as_mdf_model(self):
        import modeci_mdf.mdf as mdf

        return mdf.ConditionSet(
            node_specific={
                parse_valid_identifier(n.name): self.conditions[n].as_mdf_model() for n in self.nodes if n in self.conditions
            },
            termination={
                str.lower(k.name): v.as_mdf_model() for k, v in self.termination_conds.items()
            },
        )

    @handle_external_context()
    def get_clock(self, context):
        return super().get_clock(context.execution_id)


_doc_subs = {
    None: [
        (
            '(When creating a Scheduler explicitly, the set of nodes)',
            'A Scheduler can be created explicitly using its constructor.  However, more commonly it is created automatically for a `Composition <Composition>` when it is created.\\1',
        ),

        (
            r'(\n\* a \*graph specification dictionary\* in the \*\*graph\*\* argument)',
            "\n* a `Composition` in the **composition** argument - if a Composition is specified, the Scheduler is created using the nodes and edges in the Composition's `graph <Composition.graph_processing>`, with any `feedback Projections <Composition_Feedback>` pruned as needed to produce an acyclic graph. If there is a cycle comprised of all non-feedback projections, this cycle is reduced to a single `consideration set <consideration_set>`\n\\1",
        ),

        ('origin nodes', '`ORIGIN` nodes',),

        (
            r'Examples\n--------.*\n\.\. _Scheduler_Class_Reference',
            """
Examples
--------

Please see `Condition` for a list of all supported Conditions and their behavior.

* Basic phasing in a linear process::

    >>> import psyneulink as pnl

    >>> A = pnl.TransferMechanism(name='A')
    >>> B = pnl.TransferMechanism(name='B')
    >>> C = pnl.TransferMechanism(name='C')

    >>> comp = pnl.Composition()

    >>> pway = comp.add_linear_processing_pathway([A, B, C])
    >>> pway.pathway
    [(TransferMechanism A), (MappingProjection MappingProjection from A[RESULT] to B[InputPort-0]), (TransferMechanism B), (MappingProjection MappingProjection from B[RESULT] to C[InputPort-0]), (TransferMechanism C)]

    >>> # implicit condition of Always for A
    >>> comp.scheduler.add_condition(B, pnl.EveryNCalls(A, 2))
    >>> comp.scheduler.add_condition(C, pnl.EveryNCalls(B, 3))

    >>> # implicit AllHaveRun Termination condition
    >>> execution_sequence = list(comp.scheduler.run())
    >>> execution_sequence
    [{(TransferMechanism A)}, {(TransferMechanism A)}, {(TransferMechanism B)}, {(TransferMechanism A)}, {(TransferMechanism A)}, {(TransferMechanism B)}, {(TransferMechanism A)}, {(TransferMechanism A)}, {(TransferMechanism B)}, {(TransferMechanism C)}]

* Alternate basic phasing in a linear process::

    >>> comp = pnl.Composition()
    >>> pway = comp.add_linear_processing_pathway([A, B])
    >>> pway.pathway
    [(TransferMechanism A), (MappingProjection MappingProjection from A[RESULT] to B[InputPort-0]), (TransferMechanism B)]

    >>> comp.scheduler.add_condition(
    ...     A,
    ...     pnl.Any(
    ...         pnl.AtPass(0),
    ...         pnl.EveryNCalls(B, 2)
    ...     )
    ... )

    >>> comp.scheduler.add_condition(
    ...     B,
    ...     pnl.Any(
    ...         pnl.EveryNCalls(A, 1),
    ...         pnl.EveryNCalls(B, 1)
    ...     )
    ... )
    >>> termination_conds = {
    ...     pnl.TimeScale.TRIAL: pnl.AfterNCalls(B, 4, time_scale=pnl.TimeScale.TRIAL)
    ... }
    >>> execution_sequence = list(comp.scheduler.run(termination_conds=termination_conds))
    >>> execution_sequence # doctest: +SKIP
    [{(TransferMechanism A)}, {(TransferMechanism B)}, {(TransferMechanism B)}, {(TransferMechanism A)}, {(TransferMechanism B)}, {(TransferMechanism B)}]

* Basic phasing in two processes::

    >>> comp = pnl.Composition()
    >>> pway = comp.add_linear_processing_pathway([A, C])
    >>> pway.pathway
    [(TransferMechanism A), (MappingProjection MappingProjection from A[RESULT] to C[InputPort-0]), (TransferMechanism C)]

    >>> pway = comp.add_linear_processing_pathway([B, C])
    >>> pway.pathway
    [(TransferMechanism B), (MappingProjection MappingProjection from B[RESULT] to C[InputPort-0]), (TransferMechanism C)]

    >>> comp.scheduler.add_condition(A, pnl.EveryNPasses(1))
    >>> comp.scheduler.add_condition(B, pnl.EveryNCalls(A, 2))
    >>> comp.scheduler.add_condition(
    ...     C,
    ...     pnl.Any(
    ...         pnl.AfterNCalls(A, 3),
    ...         pnl.AfterNCalls(B, 3)
    ...     )
    ... )
    >>> termination_conds = {
    ...     pnl.TimeScale.TRIAL: pnl.AfterNCalls(C, 4, time_scale=pnl.TimeScale.TRIAL)
    ... }
    >>> execution_sequence = list(comp.scheduler.run(termination_conds=termination_conds))
    >>> execution_sequence  # doctest: +SKIP
    [{(TransferMechanism A)}, {(TransferMechanism A), (TransferMechanism B)}, {(TransferMechanism A)}, {(TransferMechanism C)}, {(TransferMechanism A), (TransferMechanism B)}, {(TransferMechanism C)}, {(TransferMechanism A)}, {(TransferMechanism C)}, {(TransferMechanism A), (TransferMechanism B)}, {(TransferMechanism C)}]

.. _Scheduler_Class_Reference
"""
        ),

        (
            r'linear graph with two nodes.*In standard mode',
            """linear composition with two nodes::

    >>> import psyneulink as pnl

    >>> A = pnl.TransferMechanism()
    >>> B = pnl.TransferMechanism()

    >>> comp = pnl.Composition()
    >>> pway = comp.add_linear_processing_pathway([A, B])

    >>> comp.scheduler.add_condition(A, pnl.TimeInterval(start=10))
    >>> comp.scheduler.add_condition(B, pnl.TimeInterval(start=10))

In standard mode"""
        ),

        (
            r'(earlier or later \(e\.g\., when the Condition\nfor a particular node or set of nodes is met\)\.)',
            '\\1 When the Scheduler terminates a `TRIAL <TimeScale.TRIAL>`, the `Composition <Composition>` begins processing the next input specified in the call to its `run <Composition.run>` method. Thus, a `TRIAL <TimeScale.TRIAL>` is defined as the scope of processing associated with a given input to the Composition.'
        ),

        (
            '(is when all of its constituent environment state updates have terminated.)',
            """\\1 These defaults may be overriden when running a Composition, by passing a dictionary mapping `TimeScales <TimeScale>` to `Conditions <Condition>` in the **termination_processing** argument of a call to `Composition.run` (to terminate the execution of processing)::

    Composition.run(
        ...,
        termination_processing={TimeScale.TRIAL: WhenFinished(ddm)}
        )
"""
        ),

        (
            r'(however, using the `indices in the original consideration queue<.*?\.Scheduler\.consideration_queue_indices>`\.)',
            '\\1 `Composition`\\ s will execute nodes in this order, however independent usages of the scheduler may not. The user should be aware of this and set up defaults and inputs to nodes accordingly.'
        ),
    ],
    'Scheduler': [
        (
            r'(Arguments\n    ---------\n)',
            """\\1
    composition : Composition
        specifies the `Components <Component>` to be ordered for execution, and any dependencies among them,
        based on the `Composition <Composition>`\\'s `graph <Composition.graph_processing>`.
"""
        )
    ]
}
