# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Scheduler ***************************************************************

"""

Overview
--------

A scheduler is an object that can be used to generate an order in which components are to be run, using a set of
arbitrary `Condition`<Condition>s (`ConditionSet`<ConditionSet>). These Conditions can be specified relative
to other components, for example, to generate patterns in which some components run at higher frequency than
others, or in which components wait for others to reach some stopping condition. However, the scheduler is
designed to be compatible with any Conditions.

Creating a Scheduler
--------------------

A scheduler is created either explicitly, or implicitly during the creation of a System. When creating a scheduler
explicitly, you must either provide a System, or a set of components (i.e. vertices) and a topological ordering of
these components that represents the graph structure of these components, produced by the toposort module. This
graph structure must be directed and acyclic. When providing a System, the scheduler is created using the System's
executionList as components and its executionGraph as the topological ordering. Additionally, a ConditionSet may be
assigned at creation; otherwise, a blank set is initialized. Conditions may be added directly to the scheduler's
ConditionSet later using the wrapper methods add_condition and add_condition_set.
If both a System and a set of nodes are specified, the scheduler will initialize based on the System.

Running a Scheduler
-------------------

Using the run method, which is a generator, the scheduler can specify the ordering in which its nodes should be run
to meet the conditions specified in the condition set. When calling the run method, if any nodes do not have run
conditions specified, by default they will be assigned the condition Always, which allows them to run at any time
they are under consideration. If termination conditions are not specified, by default the scheduler will terminate
using the AllHaveRun condition, which is true when all of the nodes in the scheduler have been told to run
at least once. Each generation from the run method will consist of a set of all nodes that can run in the current
time step, and this set itself represents the time step.


Algorithm
---------

A scheduler first constructs a consideration queue of its nodes using the topological ordering. This consideration
queue consists of a list of sets of nodes grouped based on their graph dependencies. The first set consists of
source nodes only. The second set consists of all nodes that have incoming edges only from nodes in the first set.
The third set consists of all other nodes that have incoming edges only from nodes in the first two sets, and so on.

When running, the scheduler maintains internal state about the number of times its nodes were set to be executed, and
the total number of steps in each TimeScale that have happened. This information is used by Conditions. The scheduler
checks each set in order in the consideration queue, and determines which nodes in this set are allowed to run based on
whether their associated conditions were met; all nodes within a consideration set that are allowed to run comprise a
`time step`<TimeScale.TIME_STEP>. These nodes are considered to be run simultaneously, and so the running of a node
within a time step may trigger the running of another node within its consideration set. The ordering of these nodes
is irrelevant, as it is necessarily the case that no parent and child nodes are within the same consideration set. A
key feature is that all parents have the chance to run (even if they do not actually run) before their children.

At the beginning of each time step, the scheduler checks whether the specified termination conditions have been met,
and terminates if so.

Examples
--------

Please see `Condition`<Condition> documentation for a list of all included Conditions and their behavior.

*Basic phasing in a linear process:*
    A = TransferMechanism(function = Linear(), name = 'A')
    B = TransferMechanism(function = Linear(), name = 'B')
    C = TransferMechanism(function = Linear(), name = 'C')

    p = process(
        pathway=[A, B, C],
        name = 'p',
    )
    s = system(
        processes=[p],
        name='s',
    )
    sched = Scheduler(system=s)

    #impicit condition of Always for A

    sched.add_condition(B, EveryNCalls(A, 2))
    sched.add_condition(C, EveryNCalls(B, 3))

    # implicit AllHaveRun condition
    output = list(sched.run())

    # output will produce
    # [A, A, B, A, A, B, A, A, B, C]

*Alternate basic phasing in a linear process:*
    A = TransferMechanism(function = Linear(), name = 'A')
    B = TransferMechanism(function = Linear(), name = 'B')

    p = process(
        pathway=[A, B],
        name = 'p',
    )
    s = system(
        processes=[p],
        name='s',
    )
    sched = Scheduler(system=s)

    sched.add_condition(A, Any(AtPass(0), EveryNCalls(B, 2)))
    sched.add_condition(B, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))

    termination_conds = {ts: None for ts in TimeScale}
    termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 4, time_scale=TimeScale.TRIAL)
    output = list(sched.run(termination_conds=termination_conds))

    # output will produce
    # [A, B, B, A, B, B]

*Basic phasing in two processes:*
    A = TransferMechanism(function = Linear(), name = 'A')
    B = TransferMechanism(function = Linear(), name = 'B')
    C = TransferMechanism(function = Linear(), name = 'C')

    p = process(
        pathway=[A, C],
        name = 'p',
    )
    q = process(
        pathway=[B, C],
        name = 'q',
    )
    s = system(
        processes=[p, q],
        name='s',
    )
    sched = Scheduler(system=s)

    sched.add_condition(A, EveryNPasses(1))
    sched.add_condition(B, EveryNCalls(A, 2))
    sched.add_condition(C, Any(AfterNCalls(A, 3), AfterNCalls(B, 3)))

    termination_conds = {ts: None for ts in TimeScale}
    termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)
    output = list(sched.run(termination_conds=termination_conds))

    # output will produce
    # [A, set([A,B]), A, C, set([A,B]), C, A, C, set([A,B]), C]

"""

import logging

from toposort import toposort

from PsyNeuLink.Globals.TimeScale import TimeScale
from PsyNeuLink.scheduling.condition import AllHaveRun, Always, ConditionSet

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)

class Scheduler(object):
    """
        An object that can be used to generate an order in which components are to be run, using a set of
        arbitrary `Condition`<Condition>s (`ConditionSet`<ConditionSet>).

        Attributes:
            condition_set (ConditionSet): the set of Conditions this scheduler will use when running
            execution_list (list): the full history of time steps this scheduler has produced
            consideration_queue (list): a list form of the scheduler's toposort ordering of its nodes
            termination_conds (dict): a mapping from :keyword:`TimeScale`s to :keyword:`Condition`s that when met terminate the execution of the specified :keyword:`TimeScale`
    """
    def __init__(self, system=None, condition_set=None, nodes=None, toposort_ordering=None):
        '''
        :param self:
        :param condition_set: (ConditionSet) - a :keyword:`ConditionSet` to be scheduled
        '''
        self.condition_set = condition_set if condition_set is not None else ConditionSet(scheduler=self)
        # stores the in order list of self.run's yielded outputs
        self.execution_list = []
        self.consideration_queue = []
        self.termination_conds = None

        if system is not None:
            self.nodes = [m[0] for m in system.executionList]
            self._init_consideration_queue_from_system(system)
        elif nodes is not None:
            self.nodes = nodes
            if toposort_ordering is None:
                raise SchedulerError('Instantiating Scheduler by list of nodes requires a toposort ordering (kwarg toposort_ordering)')
            self.consideration_queue = list(toposort_ordering)
        else:
            raise SchedulerError('Must instantiate a Scheduler with either a System (kwarg system), or a list of Mechanisms (kwarg nodes) and and a toposort ordering over them (kwarg toposort_ordering)')

        self._init_counts()

    # the consideration queue is the ordered list of sets of nodes in the composition graph, by the
    # order in which they should be checked to ensure that all parents have a chance to run before their children
    def _init_consideration_queue_from_system(self, system):
        dependencies = []
        for dependency_set in list(toposort(system.executionGraph)):
            new_set = set()
            for d in dependency_set:
                new_set.add(d.mechanism)
            dependencies.append(new_set)
        self.consideration_queue = dependencies
        logger.debug('Consideration queue: {0}'.format(self.consideration_queue))

    def _init_counts(self):
        # self.times[p][q] stores the number of TimeScale q ticks that have happened in the current TimeScale p
        self.times = {ts: {ts: 0 for ts in TimeScale} for ts in TimeScale}
        # stores total the number of occurrences of a node through the time scale
        # i.e. the number of times node has ran/been queued to run in a trial
        self.counts_total = {ts: None for ts in TimeScale}
        # counts_useable is a dictionary intended to store the number of available "instances" of a certain node that
        # are available to expend in order to satisfy conditions such as "run B every two times A runs"
        # specifically, counts_useable[a][b] = n indicates that there are n uses of a that are available for b to expend
        # so, in the previous example B would check to see if counts_useable[A][B] is 2, in which case B can run
        self.counts_useable = {node: {n: 0 for n in self.nodes} for node in self.nodes}

        for ts in TimeScale:
            self.counts_total[ts] = {n: 0 for n in self.nodes}

    def _reset_count(self, count, time_scale):
        for c in count[time_scale]:
            count[time_scale][c] = 0

    def _increment_time(self, time_scale):
        for ts in TimeScale:
            self.times[ts][time_scale] += 1

    def _reset_time(self, time_scale):
        for ts in TimeScale:
            self.times[time_scale][ts] = 0

    ################################################################################
    # Wrapper methods
    #   to allow the user to ignore the ConditionSet internals
    ################################################################################
    def __contains__(self, item):
        return self.condition_set.__contains__(item)

    def add_condition(self, owner, condition):
        '''
        :param: self:
        :param owner: the :keyword:`Component` that is dependent on the :param conditions:
        :param conditions: a :keyword:`Condition` (including All or Any)
        '''
        self.condition_set.add_condition(owner, condition)

    def add_condition_set(self, conditions):
        '''
        :param: self:
        :param conditions: a :keyword:`dict` mapping :keyword:`Component`s to :keyword:`Condition`s, can be added later with :keyword:`add_condition`
        '''
        self.condition_set.add_condition_set(conditions)

    ################################################################################
    # Validation methods
    #   to provide the user with info if they do something odd
    ################################################################################
    def _validate_run_state(self):
        self._validate_condition_set()
        self._validate_termination()

    def _validate_condition_set(self):
        unspecified_nodes = []
        for node in self.nodes:
            if node not in self.condition_set:
                self.condition_set.add_condition(node, Always())
                unspecified_nodes.append(node)
        if len(unspecified_nodes) > 0:
            logger.warning('These nodes have no Conditions specified, and will be scheduled with condition Always: {0}'.format(unspecified_nodes))

    def _validate_termination(self):
        if self.termination_conds is None:
            logger.warning('A termination Condition dict (termination_conds[<time_step>]: Condition) was not specified, and so the termination conditions for all TimeScale will be set to AllHaveRun()')
            self.termination_conds = {ts: AllHaveRun() for ts in TimeScale}
        for tc in self.termination_conds:
            if self.termination_conds[tc] is None:
                if tc in [TimeScale.TRIAL]:
                    raise SchedulerError('Must specify a {0} termination Condition (termination_conds[{0}]'.format(tc))
            else:
                if self.termination_conds[tc].scheduler is None:
                    logger.debug('Setting scheduler of {0} to self ({1})'.format(self.termination_conds[tc], self))
                    self.termination_conds[tc].scheduler = self
    ################################################################################
    # Run methods
    ################################################################################
    def run(self, termination_conds=None):
        '''
        :param self:
        :param termination_conds: (dict) - a mapping from :keyword:`TimeScale`s to :keyword:`Condition`s that when met terminate the execution of the specified :keyword:`TimeScale`
        '''
        self.termination_conds = termination_conds
        self._validate_run_state()

        logger.info('termination_conds: {0}, self.termination_conds: {1}'.format(termination_conds, self.termination_conds))

        def has_reached_termination(self, time_scale=None):
            term = True
            if time_scale is None:
                for ts in self.termination_conds:
                    term = term and self.termination_conds[ts].is_satisfied()
            else:
                term = term and self.termination_conds[time_scale].is_satisfied()

            return term

        execution_list = []
        self.counts_useable = {node: {n: 0 for n in self.nodes} for node in self.nodes}
        self._reset_count(self.counts_total, TimeScale.TRIAL)
        self._reset_time(TimeScale.TRIAL)

        while not self.termination_conds[TimeScale.TRIAL].is_satisfied():
            self._reset_count(self.counts_total, TimeScale.PASS)
            self._reset_time(TimeScale.PASS)

            execution_list_has_changed = False
            cur_index_consideration_queue = 0

            while (
                cur_index_consideration_queue < len(self.consideration_queue)
                and not self.termination_conds[TimeScale.TRIAL].is_satisfied()
            ):
                cur_time_step_exec = set()
                cur_consideration_set = self.consideration_queue[cur_index_consideration_queue]
                try:
                    iter(cur_consideration_set)
                except TypeError as e:
                    raise SchedulerError('cur_consideration_set is not iterable, did you ensure that this Scheduler was instantiated with an actual toposort output for param toposort_ordering? err: {0}'.format(e))
                logger.debug('trial, num passes in trial {0}, consideration_queue {1}'.format(self.times[TimeScale.TRIAL][TimeScale.PASS], ' '.join([str(x) for x in cur_consideration_set])))

                # do-while, on cur_consideration_set_has_changed
                while True:
                    cur_consideration_set_has_changed = False
                    for current_node in cur_consideration_set:
                        logger.debug('cur time_step exec: {0}'.format(cur_time_step_exec))
                        for n in self.counts_useable:
                            logger.debug('Counts of {0} useable by'.format(n))
                            for n2 in self.counts_useable[n]:
                                logger.debug('\t{0}: {1}'.format(n2, self.counts_useable[n][n2]))

                        if self.condition_set.conditions[current_node].is_satisfied():
                            if current_node not in cur_time_step_exec:
                                logger.debug('adding {0} to execution list'.format(current_node))
                                logger.debug('cur time_step exec pre add: {0}'.format(cur_time_step_exec))
                                cur_time_step_exec.add(current_node)
                                logger.debug('cur time_step exec post add: {0}'.format(cur_time_step_exec))
                                execution_list_has_changed = True
                                cur_consideration_set_has_changed = True

                                for ts in TimeScale:
                                    self.counts_total[ts][current_node] += 1
                                # current_node's node is added to the execution queue, so we now need to
                                # reset all of the counts useable by current_node's node to 0
                                for n in self.counts_useable:
                                    self.counts_useable[n][current_node] = 0
                                # and increment all of the counts of current_node's node useable by other
                                # nodes by 1
                                for n in self.counts_useable:
                                    self.counts_useable[current_node][n] += 1
                    # do-while condition
                    if not cur_consideration_set_has_changed:
                        break

                if len(cur_time_step_exec) >= 1:
                    self.execution_list.append(cur_time_step_exec)
                    yield self.execution_list[-1]

                    self._increment_time(TimeScale.TIME_STEP)

                cur_index_consideration_queue += 1

            if not execution_list_has_changed:
                self.execution_list.append(set())
                yield self.execution_list[-1]

                self._increment_time(TimeScale.TIME_STEP)

            # can execute the execution_list here
            logger.info(self.execution_list)
            logger.debug('Execution list: [{0}]'.format(' '.join([str(x) for x in self.execution_list])))
            self._increment_time(TimeScale.PASS)

        self._increment_time(TimeScale.TRIAL)

        return self.execution_list
