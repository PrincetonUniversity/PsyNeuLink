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
per `PASS` through the Composition. For example, in a `Composition` in which a Mechanism A projects to a Mechanism B
that projects to a Mechanism C, A will execute first followed by B, and then C in each `PASS` through the Composition.
However, a Scheduler can be used to implement more complex patterns of execution, by specifying `Conditions <Condition>`
that determine when and how many times individual Components execute, and whether and how this depends on the execution
of other Components. Any executable Component in a Composition can be assigned a Condition, and Conditions can be
combined in arbitrary ways to generate any pattern of execution of the Components in a Composition that is logically
possible.

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

COMMENT:
   JDC: WE MAY WANT TO CHANGE THE NAME OF THE ARGUMENT TO 'COMPOSITION` ONCE THAT IS IMPLEMENTED, TO BE FULLY GENERAL
COMMENT

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
<Scheduler_Termination_Conditions>` is satisfied. If no termination Conditions are specified, the Scheduler terminates
a `TRIAL <TimeScale.TRIAL>` when every Component has been specified for execution at least once (corresponding to the
`AllHaveRun` Condition).  However, other termination Conditions can be specified, that may cause the Scheduler to
terminate a `TRIAL <TimeScale.TRIAL>` earlier  or later (e.g., when the  Condition for a particular Component or set of
Components is met).  When the Scheduler terminates a `TRIAL <TimeScale.TRIAL>`, the `Composition <Composition>` begins
processing the next input specified in the call to its `run <Composition.run>` method.  Thus, a `TRIAL
<TimeScale.TRIAL>` is defined as the scope of processing associated with a given input to the Composition.

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

import copy
import datetime
import logging

from toposort import toposort

from psyneulink.core.globals.context import Context, handle_external_context
from psyneulink.core.globals.json import JSONDumpable
from psyneulink.core.scheduling.condition import All, AllHaveRun, Always, Condition, ConditionSet, EveryNCalls, Never
from psyneulink.core.scheduling.time import Clock, TimeScale

__all__ = [
    'Scheduler', 'SchedulerError',
]

logger = logging.getLogger(__name__)


class SchedulerError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class Scheduler(JSONDumpable):
    """Generates an order of execution for `Components <Component>` in a `Composition <Composition>` or graph
    specification dictionary, possibly determined by a set of `Conditions <Condition>`.

    Arguments
    ---------

    composition : Composition
        specifies the `Components <Component>` to be ordered for execution, and any dependencies among them,
        based on the `Composition <Composition>`\\'s `graph <Composition.graph_processing>`.

    conditions  : ConditionSet
        set of `Conditions <Condition>` that specify when individual `Components` <Component>` in **composition**
        execute and any dependencies among them.

    graph : Dict[Component: set(Component)]
        a graph specification dictionary - each entry of the dictionary must be a `Component`,
        and the value of each entry must be a set of zero or more Components that project directly to the key.

    Attributes
    ----------

    conditions : ConditionSet
        the set of Conditions the Scheduler uses when running

    execution_list : list
        the full history of time steps the Scheduler has produced

    consideration_queue : list
        a list form of the Scheduler's toposort ordering of its nodes

    termination_conds : Dict[TimeScale: Condition]
        a mapping from `TimeScales <TimeScale>` to `Conditions <Condition>` that, when met, terminate the execution
        of the specified `TimeScale`.

    times : Dict[TimeScale: Dict[TimeScale: int]]
        a structure counting the number of occurrences of a certain `TimeScale` within the scope of another `TimeScale`.
        For example, `times[TimeScale.RUN][TimeScale.PASS]` is the number of `PASS`es that have occurred in the
        current `RUN` that the Scheduler is scheduling at the time it is accessed

    clock : `Clock`
        a `Clock` object that stores the current time in this Scheduler

    """
    def __init__(
        self,
        composition=None,
        graph=None,
        conditions=None,
        termination_conds={
            TimeScale.RUN: Never(),
            TimeScale.TRIAL: AllHaveRun(),
        },
        default_execution_id=None,
        **kwargs
    ):
        """
        :param self:
        :param composition: (Composition) - the Composition this scheduler is scheduling for
        :param conditions: (ConditionSet) - a :keyword:`ConditionSet` to be scheduled
        """
        self.conditions = ConditionSet(conditions)

        # stores the in order list of self.run's yielded outputs
        self.consideration_queue = []
        self.default_termination_conds = Scheduler._parse_termination_conditions(termination_conds)
        self._termination_conds = termination_conds.copy()

        self.cycle_nodes = set()

        if composition is not None:
            self.nodes = [vert.component for vert in composition.graph_processing.vertices]
            self._init_consideration_queue_from_graph(composition.graph_processing)
            if default_execution_id is None:
                default_execution_id = composition.default_execution_id
        elif graph is not None:
            try:
                self.nodes = [vert.component for vert in graph.vertices]
                self._init_consideration_queue_from_graph(graph)
            except AttributeError:
                self.consideration_queue = list(toposort(graph))
                self.nodes = []
                for consideration_set in self.consideration_queue:
                    for node in consideration_set:
                        self.nodes.append(node)
        else:
            raise SchedulerError('Must instantiate a Scheduler with either a Composition (kwarg composition) '
                                 'or a graph dependency dict (kwarg graph)')

        self.default_execution_id = default_execution_id
        self.execution_list = {self.default_execution_id: []}
        self.clocks = {self.default_execution_id: Clock()}
        self.counts_total = {}
        self.counts_useable = {}
        self._init_counts(execution_id=self.default_execution_id)
        self.date_creation = datetime.datetime.now()
        self.date_last_run_end = None

    # the consideration queue is the ordered list of sets of nodes in the graph, by the
    # order in which they should be checked to ensure that all parents have a chance to run before their children
    def _init_consideration_queue_from_graph(self, graph):
        self.dependency_dict, self.removed_dependencies, self.structural_dependencies = graph.prune_feedback_edges()
        self.consideration_queue = list(toposort(self.dependency_dict))

    def _init_counts(self, execution_id=None, base_execution_id=None):
        """
            Attributes
            ----------

                execution_id
                    the execution_id to initialize counts for
                    default : self.default_execution_id

                base_execution_id
                    if specified, the counts for execution_id will be copied from the counts of base_execution_id
                    default : None
        """
        # all counts are divided by execution_id, which provides a context for the scheduler's execution, so that
        # it can be reused in multiple contexts

        # stores total the number of occurrences of a node through the time scale
        # i.e. the number of times node has ran/been queued to run in a trial
        if execution_id not in self.counts_total:
            self.counts_total[execution_id] = {}

            if base_execution_id is not None:
                if base_execution_id not in self.counts_total:
                    raise SchedulerError('execution_id {0} not in {1}.counts_total'.format(base_execution_id, self))

                self.counts_total[execution_id] = {
                    ts: {n: self.counts_total[base_execution_id][ts][n] for n in self.nodes} for ts in TimeScale
                }
            else:
                self.counts_total[execution_id] = {
                    ts: {n: 0 for n in self.nodes} for ts in TimeScale
                }

        # counts_useable is a dictionary intended to store the number of available "instances" of a certain node that
        # are available to expend in order to satisfy conditions such as "run B every two times A runs"
        # specifically, counts_useable[a][b] = n indicates that there are n uses of a that are available for b to expend
        # so, in the previous example B would check to see if counts_useable[A][B] >= 2, in which case B can run
        # then, counts_useable[a][b] would be reset to 0, even if it was greater than 2
        if execution_id not in self.counts_useable:
            self.counts_useable[execution_id] = {}

            if base_execution_id is not None:
                if base_execution_id not in self.counts_useable:
                    raise SchedulerError('execution_id {0} not in {1}.counts_useable'.format(base_execution_id, self))

                self.counts_useable[execution_id] = {
                    node: {n: self.counts_useable[base_execution_id][node][n] for n in self.nodes} for node in self.nodes
                }
            else:
                self.counts_useable[execution_id] = {
                    node: {n: 0 for n in self.nodes} for node in self.nodes
                }

        if execution_id not in self.execution_list:
            if base_execution_id is not None:
                if base_execution_id not in self.execution_list:
                    raise SchedulerError('execution_id {0} not in {1}.execution_list'.format(base_execution_id, self))

                self.execution_list[execution_id] = list(self.execution_list[base_execution_id])
            else:
                self.execution_list[execution_id] = []

        self._init_clock(execution_id, base_execution_id)

    def _init_clock(self, execution_id=None, base_execution_id=None):
        # instantiate new Clock for this execution_id if necessary
        # currently does not work with base_execution_id
        if execution_id not in self.clocks:
            if base_execution_id is not None:
                if base_execution_id not in self.clocks:
                    raise SchedulerError('execution_id {0} not in {1}.clocks'.format(base_execution_id, self))

                self.clocks[execution_id] = copy.deepcopy(self.clocks[base_execution_id])
            else:
                self.clocks[execution_id] = Clock()

    def _reset_counts_total(self, time_scale, execution_id=None):
        for ts in TimeScale:
            # only reset the values underneath the current scope
            # this works because the enum is set so that higher granularities of time have lower values
            if ts.value <= time_scale.value:
                for c in self.counts_total[execution_id][ts]:
                    self.counts_total[execution_id][ts][c] = 0

    def _reset_counts_useable(self, execution_id=None):
        self.counts_useable[execution_id] = {
            node: {n: 0 for n in self.nodes} for node in self.nodes
        }

    def update_termination_conditions(self, termination_conds):
        termination_conds = Scheduler._parse_termination_conditions(termination_conds)
        new_conds = self.termination_conds.copy()
        new_conds.update(termination_conds)

        return new_conds

    @staticmethod
    def _parse_termination_conditions(termination_conds):
        try:
            return {k: termination_conds[k] for k in termination_conds if isinstance(k, TimeScale) and isinstance(termination_conds[k], Condition)}
        except TypeError:
            raise TypeError('termination_conditions must be a dictionary of the form {TimeScale: Condition, ...}')

    ################################################################################
    # Wrapper methods
    #   to allow the user to ignore the ConditionSet internals
    ################################################################################
    def __contains__(self, item):
        return self.conditions.__contains__(item)

    def add_condition(self, owner, condition):
        """
        Adds a `Condition` to the Scheduler. If **owner** already has a Condition, it is overwritten
        with the new one. If you want to add multiple conditions to a single owner, use the
        `composite Conditions <Conditions_Composite>` to accurately specify the desired behavior.

        Arguments
        ---------

        owner : Component
            specifies the Component with which the **condition** should be associated. **condition**
            will govern the execution behavior of **owner**

        condition : Condition
            specifies the Condition, associated with the **owner** to be added to the ConditionSet.
        """
        self.conditions.add_condition(owner, condition)

    def add_condition_set(self, conditions):
        """
        Adds a set of `Conditions <Condition>` (in the form of a dict or another ConditionSet) to the Scheduler.
        Any Condition added here will overwrite an existing Condition for a given owner.
        If you want to add multiple conditions to a single owner, add a single `Composite Condition <Conditions_Composite>`
        to accurately specify the desired behavior.

        Arguments
        ---------

        conditions : dict[`Component <Component>`: `Condition`], `ConditionSet`
            specifies collection of Conditions to be added to this ConditionSet,

            if a dict is provided:
                each entry should map an owner `Component` (the `Component` whose execution behavior will be
                governed) to a `Condition <Condition>`

        """
        self.conditions.add_condition_set(conditions)

    ################################################################################
    # Validation methods
    #   to provide the user with info if they do something odd
    ################################################################################
    def _validate_run_state(self):
        self._validate_conditions()

    def _validate_conditions(self):
        unspecified_nodes = []
        for node in self.nodes:
            if node not in self.conditions:
                # determine parent nodes
                node_index = 0
                for i in range(len(self.consideration_queue)):
                    if node in self.consideration_queue[i]:
                        node_index = i
                        break

                if node_index > 0:
                    dependencies = list(self.consideration_queue[i - 1])
                    if len(dependencies) == 1:
                        cond = EveryNCalls(dependencies[0], 1)
                    elif len(dependencies) > 1:
                        cond = All(*[EveryNCalls(x, 1) for x in dependencies])
                    else:
                        raise SchedulerError(f'{self}: Empty consideration set in consideration_queue[{i - 1}]')
                else:
                    cond = Always()

                self.conditions.add_condition(node, cond)
                unspecified_nodes.append(node)
        if len(unspecified_nodes) > 0:
            logger.info(
                'These nodes have no Conditions specified, and will be scheduled with conditions: {0}'.format(
                    {node: self.conditions[node] for node in unspecified_nodes}
                )
            )

    ################################################################################
    # Run methods
    ################################################################################
    @handle_external_context(fallback_default=True)
    def run(self, termination_conds=None, context=None, base_context=Context(execution_id=None), skip_trial_time_increment=False):
        """
        run is a python generator, that when iterated over provides the next `TIME_STEP` of
        executions at each iteration

        :param termination_conds: (dict) - a mapping from `TimeScale`\\s to `Condition`\\s that when met
               terminate the execution of the specified `TimeScale`
        """
        self._validate_run_state()
        if termination_conds is None:
            termination_conds = self.termination_conds
        else:
            termination_conds = self.update_termination_conditions(Scheduler._parse_termination_conditions(termination_conds))

        self._init_counts(context.execution_id, base_context.execution_id)
        self._reset_counts_useable(context.execution_id)
        self._reset_counts_total(TimeScale.TRIAL, context.execution_id)

        while (
            not termination_conds[TimeScale.TRIAL].is_satisfied(scheduler=self, context=context)
            and not termination_conds[TimeScale.RUN].is_satisfied(scheduler=self, context=context)
        ):
            self._reset_counts_total(TimeScale.PASS, context.execution_id)

            execution_list_has_changed = False
            cur_index_consideration_queue = 0

            while (
                cur_index_consideration_queue < len(self.consideration_queue)
                and not termination_conds[TimeScale.TRIAL].is_satisfied(scheduler=self, context=context)
                and not termination_conds[TimeScale.RUN].is_satisfied(scheduler=self, context=context)
            ):
                # all nodes to be added during this time step
                cur_time_step_exec = set()
                # the current "layer/group" of nodes that MIGHT be added during this time step
                cur_consideration_set = self.consideration_queue[cur_index_consideration_queue]
                try:
                    iter(cur_consideration_set)
                except TypeError as e:
                    raise SchedulerError('cur_consideration_set is not iterable, did you ensure that this Scheduler was instantiated with an actual toposort output for param toposort_ordering? err: {0}'.format(e))

                # do-while, on cur_consideration_set_has_changed
                # we check whether each node in the current consideration set is allowed to run,
                # and nodes can cause cascading adds within this set
                while True:
                    cur_consideration_set_has_changed = False
                    for current_node in cur_consideration_set:
                        # only add each node once during a single time step, this also serves
                        # to prevent infinitely cascading adds
                        if current_node not in cur_time_step_exec:
                            if self.conditions.conditions[current_node].is_satisfied(scheduler=self, context=context):
                                cur_time_step_exec.add(current_node)
                                execution_list_has_changed = True
                                cur_consideration_set_has_changed = True

                                for ts in TimeScale:
                                    self.counts_total[context.execution_id][ts][current_node] += 1
                                # current_node's node is added to the execution queue, so we now need to
                                # reset all of the counts useable by current_node's node to 0
                                for n in self.counts_useable[context.execution_id]:
                                    self.counts_useable[context.execution_id][n][current_node] = 0
                                # and increment all of the counts of current_node's node useable by other
                                # nodes by 1
                                for n in self.counts_useable[context.execution_id]:
                                    self.counts_useable[context.execution_id][current_node][n] += 1
                    # do-while condition
                    if not cur_consideration_set_has_changed:
                        break

                # add a new time step at each step in a pass, if the time step would not be empty
                if len(cur_time_step_exec) >= 1:
                    self.execution_list[context.execution_id].append(cur_time_step_exec)
                    yield self.execution_list[context.execution_id][-1]

                    self.get_clock(context)._increment_time(TimeScale.TIME_STEP)

                cur_index_consideration_queue += 1

            # if an entire pass occurs with nothing running, add an empty time step
            if not execution_list_has_changed:
                self.execution_list[context.execution_id].append(set())
                yield self.execution_list[context.execution_id][-1]

                self.get_clock(context)._increment_time(TimeScale.TIME_STEP)

            self.get_clock(context)._increment_time(TimeScale.PASS)

        if not skip_trial_time_increment:
            self.get_clock(context)._increment_time(TimeScale.TRIAL)

        if termination_conds[TimeScale.RUN].is_satisfied(scheduler=self, context=context):
            self.date_last_run_end = datetime.datetime.now()

        return self.execution_list[context.execution_id]

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

    @property
    def clock(self):
        return self.clocks[self.default_execution_id]

    def get_clock(self, context):
        try:
            return self.clocks[context.default_execution_id]
        except AttributeError:
            try:
                return self.clocks[context.execution_id]
            except AttributeError:
                return self.clocks[context]
        except KeyError:
            raise

    @property
    def termination_conds(self):
        return self._termination_conds

    @termination_conds.setter
    def termination_conds(self, termination_conds):
        if termination_conds is None:
            self._termination_conds = self.default_termination_conds.copy()
        else:
            self._termination_conds.update(termination_conds)
