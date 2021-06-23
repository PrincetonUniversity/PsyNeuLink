# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Scheduler **************************************************************

"""

Overview
--------

A Scheduler is used to generate the order in which the `Components <Component>` of a `Composition <Composition>` are
executed. By default, a Scheduler executes Components in an order determined by the pattern of `Projections
<Projection>` among the `Mechanisms <Mechanism>` in the `Composition <Composition>`, with each Mechanism executed once
per `PASS` through the Composition. For example, in a `Composition` in which a Mechanism *A* projects to a Mechanism
*B* that projects to a Mechanism *C*, *A* will execute first followed by *B*, and then *C* in each `PASS` through the
Composition. However, a Scheduler can be used to implement more complex patterns of execution, by specifying
`Conditions <Condition>` that determine when and how many times individual Components execute, and whether and how
this depends on the execution of other Components. Any executable Component in a Composition can be assigned a
Condition, and Conditions can be combined in arbitrary ways to generate any pattern of execution of the Components
in a Composition that is logically possible.

.. note::
   In general, `Mechanisms <Mechanism>` are the Components of a Composition that are most commonly associated with
   Conditions, and assigned to a Scheduler for execution.  However, in some circumstances, `Projections <Projection>`
   can also be assigned for execution (e.g., during `learning <Process_Execution_Learning>` to insure that
   `MappingProjections <MappingProjection>` are updated in the proper order).

.. _Scheduler_Creation:

Creating a Scheduler
--------------------

A Scheduler can be created explicitly using its constructor.  However, more commonly it is created automatically
for a `Composition <Composition>` when it is created.  When creating a Scheduler explicitly, the set of `Components
<Component>` to be executed and their order must be specified in the Scheduler's constructor using one the following:

* a `Composition` in the **composition** argument - if a Composition is specified,
  the Scheduler is created using the nodes and edges in the Composition's `graph <Composition.graph_processing>`.

* a *graph specification dictionary* in the **graph** argument -
  each entry of the dictionary must be a Component of a Composition, and the value of each entry must be a set of
  zero or more Components that project directly to the key.  The graph must be acyclic; an error is generated if any
  cycles (e.g., recurrent dependencies) are detected.  The Scheduler computes a `toposort` from the graph that is
  used as the default order of executions, subject to any `Condition`s that have been specified
  (see `below <Scheduler_Algorithm>`).

If both a Composition and a graph are specified, the Composition takes precedence, and the graph is ignored.

Conditions can be added to a Scheduler when it is created by specifying a `ConditionSet` (a set of
`Conditions <Condition>`) in the **conditions** argument of its constructor.  Individual Conditions and/or
ConditionSets can also be added after the  Scheduler has been created, using its `add_condition` and
`add_condition_set` methods, respectively.

.. _Scheduler_Algorithm:

Algorithm
---------

.. _consideration_set:

When a Scheduler is created, it constructs a `consideration_queue`:  a list of `consideration_sets <consideration_set>`
that defines the order in which `Components <Component>` are eligible to be executed.  This is based on the pattern of
`Projections <Projection>` among them specified in the `Composition`, or on the dependencies specified in the graph
specification dictionary, whichever was provided in the Scheduler's constructor.  Each `consideration_set
<consideration_set>` is a set of Components that are eligible to execute at the same time/`TIME_STEP` (i.e.,
that appear at the same "depth" in a sequence of dependencies, and among which there are no dependencies).  The first
`consideration_set <consideration_set>` consists of only `ORIGIN` Mechanisms. The second consists of all Components
that receive `Projections <Projection>` from the Mechanisms in the first `consideration_set <consideration_set>`.
The third consists of  Components that receive Projections from Components in the first two `consideration_sets
<consideration_set>`, and so forth.  When the Scheduler is run, it uses the `consideration_queue` to determine which
Components are eligible to execute in each `TIME_STEP` of a `PASS`, and then evaluates the `Condition <Condition>`
associated with each Component in the current `consideration_set <consideration_set>` to determine which should
actually be assigned for execution.

Pseudocode::

    consideration_queue <- list(toposort(graph))

    reset TimeScale.TRIAL counters
    while TimeScale.TRIAL termination conditions are not satisfied:
        reset TimeScale.PASS counters
        cur_index <- 0

        while TimeScale.TRIAL termination conditions are not satisfied
              and cur_index < len(consideration_queue):

            cur_consideration_set <- consideration_queue[cur_index]
            do:
                cur_consideration_set_has_changed <- False
                for cur_node in cur_consideration_set:
                    if  cur_node not in cur_time_step
                        and cur_node`s Condition is satisfied:

                        cur_consideration_set_has_changed <- True
                        add cur_node to cur_time_step
                        increment execution and time counters
            while cur_consideration_set_has_changed

            if cur_time_step is not empty:
                yield cur_time_step

            increment cur_index
            increment time counters

.. _Scheduler_Execution:

Execution
---------

.. note::
    This section covers normal scheduler execution
    (`Scheduler.mode = SchedulingMode.STANDARD`). See
    `Scheduler_Exact_Time` below for a description of
    `exact time mode <SchedulingMode.EXACT_TIME>`.

When a Scheduler is run, it provides a set of Components that should be run next, based on their dependencies in the
Composition or graph specification dictionary, and any `Conditions <Condition>`, specified in the Scheduler's
constructor. For each call to the `run <Scheduler.run>` method, the Scheduler sequentially evaluates its
`consideration_sets <consideration_set>` in their order in the `consideration_queue`.  For each set, it  determines
which Components in the set are allowed to execute, based on whether their associated `Condition <Condition>` has
been met. Any Component that does not have a `Condition` explicitly specified is assigned a Condition that causes it
to be executed whenever it is `under consideration <Scheduler_Algorithm>` and all its structural parents have been
executed at least once since the Component's last execution. All of the Components within a `consideration_set
<consideration_set>` that are allowed to execute comprise a `TIME_STEP` of execution. These Components are considered
as executing simultaneously.

.. note::
    The ordering of the Components specified within a `TIME_STEP` is arbitrary (and is irrelevant, as there are no
    graph dependencies among Components within the same `consideration_set <consideration_set>`). However,
    the execution of a Component within a `time_step` may trigger the execution of another Component within its
    `consideration_set <consideration_set>`, as in the example below::

            C
          ↗ ↖
         A   B

        scheduler.add_condition(B, pnl.EveryNCalls(A, 2))
        scheduler.add_condition(C, pnl.EveryNCalls(B, 1))

        time steps: [{A}, {A, B}, {C}, ...]

    Since there are no graph dependencies between `A` and `B`, they may execute in the same `TIME_STEP`. Morever,
    `A` and `B` are in the same `consideration_set <consideration_set>`. Since `B` is specified to run every two
    times `A` runs, `A`'s second execution in the second `TIME_STEP` allows `B` to run within that `TIME_STEP`,
    rather than waiting for the next `PASS`.

For each `TIME_STEP`, the Scheduler evaluates  whether any specified
`termination Conditions <Scheduler_Termination_Conditions>` have been met, and terminates if so.  Otherwise,
it returns the set of Components that should be executed in the current `TIME_STEP`. Each subsequent call to the
`run <Scheduler.run>` method returns the set of Components in the following `TIME_STEP`.

Processing of all of the `consideration_sets <consideration_set>` in the `consideration_queue` constitutes a `PASS` of
execution, over which every Component in the Composition has been considered for execution. Subsequent calls to the
`run <Scheduler.run>` method cycle back through the `consideration_queue`, evaluating the `consideration_sets
<consideration_set>` in the same order as previously. Different subsets of Components within the same `consideration_set
<consideration_set>` may be assigned to execute on each `PASS`, since different Conditions may be satisfied.

The Scheduler continues to make `PASS`es through the `consideration_queue` until a `termination Condition
<Scheduler_Termination_Conditions>` is satisfied. If no termination Conditions are specified, by default the Scheduler
terminates a `TRIAL <TimeScale.TRIAL>` when every Component has been specified for execution at least once
(corresponding to the `AllHaveRun` Condition).  However, other termination Conditions can be specified,
that may cause the Scheduler to terminate a `TRIAL <TimeScale.TRIAL>` earlier  or later (e.g., when the  Condition
for a particular Component or set of Components is met).  When the Scheduler terminates a `TRIAL <TimeScale.TRIAL>`,
the `Composition <Composition>` begins processing the next input specified in the call to its `run <Composition.run>`
method. Thus, a `TRIAL <TimeScale.TRIAL>` is defined as the scope of processing associated with a given input to the
Composition.

.. _Scheduler_Termination_Conditions:

*Termination Conditions*
~~~~~~~~~~~~~~~~~~~~~~~~

Termination conditions are `Conditions <Condition>` that specify when the open-ended units of time - `TRIAL
<TimeScale.TRIAL>` and `RUN` - have ended.  By default, the termination condition for a `TRIAL <TimeScale.TRIAL>` is
`AllHaveRun`, which is satisfied when all Components have run at least once within the trial, and the termination
condition for a `RUN` is when all of its constituent trials have terminated. These defaults may be overriden when
running a Composition, by passing a dictionary mapping `TimeScales <TimeScale>` to `Conditions <Condition>` in the
**termination_processing** argument of a call to `Composition.run` (to terminate the execution of processing)::

    Composition.run(
        ...,
        termination_processing={TimeScale.TRIAL: WhenFinished(ddm)}
        )


.. _Scheduler_Absolute_Time:

Absolute Time
-------------

The scheduler supports scheduling of models of real-time systems in
modes, both of which involve mapping real-time values to psyneulink
`Time`. The default mode is is most compatible with standard PsyNeuLink
scheduling, but can cause some unexpected behavior in certain cases
because it is inexact. The consideration queue remains intact, but as a
result, actions specified by fixed times of absolute-time-based
conditions (`start <TimeInterval.start>` and `end <TimeInterval.end>` of
`TimeInterval`, and `t` of `TimeTermination`) may not occur at exactly
the time specified. The simplest example of this situation involves a
linear composition with two nodes::

    >>> import psyneulink as pnl

    >>> A = pnl.TransferMechanism()
    >>> B = pnl.TransferMechanism()

    >>> comp = pnl.Composition()
    >>> pway = comp.add_linear_processing_pathway([A, B])

    >>> comp.scheduler.add_condition(A, pnl.TimeInterval(start=10))
    >>> comp.scheduler.add_condition(B, pnl.TimeInterval(start=10))

In standard mode, **A** and **B** are in different consideration sets,
and so can never execute at the same time. At most one of **A** and
**B** will start exactly at t=10ms, with the other starting at its next
consideration after. There are many of these examples, and while it may
be solveable in some cases, it is not a simple problem. So,
`Exact Time Mode <Scheduler_Exact_Time>` exists as an alternative
option for these cases, though it comes with its own drawbacks.

.. note::
    Due to issues with floating-point precision, absolute time values in
    conditions and `Time` are limited to 8 decimal points. If more
    precision is needed, use
    `fractions <https://docs.python.org/3/library/fractions.html>`_,
    where possible, or smaller units (e.g. microseconds instead of
    milliseconds).


.. _Scheduler_Exact_Time:

Exact Time Mode
~~~~~~~~~~~~~~~

When `Scheduler.mode` is `SchedulingMode.EXACT_TIME`, the scheduler is
capable of handling examples like the one
`above <Scheduler_Absolute_Time>`. In this mode, all nodes in the
scheduler's graph become members of the same consideration set, and may
be executed at the same time for every time step, subject to the
conditions specified. As a result, the guarantees in
`standard scheduling <Scheduler_Execution>` may not apply - that is,
that all parent nodes get a chance to execute before their children, and
that there exist no data dependencies (Projections) between nodes in the
same execution set. In exact time mode, all nodes will be in one
[unordered] execution set. An ordering may be inferred by the original
graph, however, using the `indices in the original consideration queue
<Scheduler.consideration_queue_indices>`_. `Composition`\\ s will
execute nodes in this order, however independent usages of the scheduler
may not.  The user should be aware of this and set up defaults and
inputs to nodes accordingly. Additionally, non-absolute conditions like
`EveryNCalls` may behave unexpectedly in some cases.

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

Class Reference
===============

"""

import typing

import graph_scheduler
import pint

from psyneulink import _unit_registry
from psyneulink.core.globals.context import Context, handle_external_context
from psyneulink.core.globals.json import JSONDumpable
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

    @property
    def _dict_summary(self):
        return {
            'conditions': {
                'termination': {
                    str.lower(k.name): v._dict_summary for k, v in self.termination_conds.items()
                },
                'node_specific': {
                    n.name: self.conditions[n]._dict_summary for n in self.nodes if n in self.conditions
                }
            }
        }

    @handle_external_context()
    def get_clock(self, context):
        return super().get_clock(context.execution_id)
