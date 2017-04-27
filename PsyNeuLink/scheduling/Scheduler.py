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
    def __init__(self, composition=None, system=None, condition_set=None, nodes=None, toposort_ordering=None):
        '''
        :param self:
        :param composition: (Composition) - the Composition this scheduler is scheduling for
        :param condition_set: (ConditionSet) - a :keyword:`ConditionSet` to be scheduled
        '''
        self.condition_set = condition_set if condition_set is not None else ConditionSet(scheduler=self)
        # stores the in order list of self.run's yielded outputs
        self.execution_list = []
        self.consideration_queue = []
        self.termination_conds = None

        if composition is not None:
            self.nodes = [vert.mechanism for vert in composition.graph.vertices]
            self._init_consideration_queue_from_composition(composition)
        elif system is not None:
            self.nodes = [m[0] for m in system.executionListProcessing]
            self._init_consideration_queue_from_system(system)
        elif nodes is not None:
            self.nodes = nodes
            if toposort_ordering is None:
                raise SchedulerError('Instantiating Scheduler by list of nodes requires a toposort ordering (kwarg toposort_ordering)')
            self.consideration_queue = list(toposort_ordering)
        else:
            raise SchedulerError('Must instantiate a Scheduler with either a Composition (kwarg composition), or a list of Mechanisms (kwarg nodes) and and a toposort ordering over them (kwarg toposort_ordering)')

        self._init_counts()

    # the consideration queue is the ordered list of sets of nodes in the composition graph, by the
    # order in which they should be checked to ensure that all parents have a chance to run before their children
    def _init_consideration_queue_from_composition(self, composition):
        dependencies = {}
        for vert in composition.graph.vertices:
            dependencies[vert.mechanism] = set()
            for parent in composition.graph.get_parents(vert.mechanism):
                dependencies[vert.mechanism].add(parent)

        self.consideration_queue = list(toposort(dependencies))
        logger.debug('Consideration queue: {0}'.format(self.consideration_queue))

    def _init_consideration_queue_from_system(self, system):
        dependencies = []
        for dependency_set in list(toposort(system.executionGraphProcessing)):
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
                                    self.times[ts][TimeScale.TIME_STEP] += 1
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
