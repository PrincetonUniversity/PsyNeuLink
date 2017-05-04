import logging

from PsyNeuLink.Globals.TimeScale import TimeScale

logger = logging.getLogger(__name__)

class ConditionError(Exception):
     def __init__(self, error_value):
        self.error_value = error_value

     def __str__(self):
        return repr(self.error_value)

class ConditionSet(object):
    def __init__(self, scheduler=None, conditions=None):
        """
        :param self:
        :param scheduler: a :keyword:`Scheduler` that these conditions are associated with, which maintains any state necessary for these conditions
        :param conditions: a :keyword:`dict` mapping :keyword:`Component`s to :keyword:`iterable`s of :keyword:`Condition`s, can be added later with :keyword:`add_condition`
        """
        # even though conditions may be added in arbitrary iterables, they are stored internally as dicts of sets
        self.conditions = conditions if conditions is not None else {}
        self.scheduler = scheduler

    def __contains__(self, item):
        return item in self.conditions

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        logger.debug('ConditionSet ({0}) setting scheduler to {1}'.format(type(self).__name__, value))
        self._scheduler = value

        for owner, cond in self.conditions.items():
            cond.scheduler = value

    def add_condition(self, owner, condition):
        """
        :param: self:
        :param owner: the :keyword:`Component` that is dependent on the :param conditions:
        :param conditions: a :keyword:`Condition` (including All or Any)
        """
        logger.debug('add_condition: Setting scheduler of {0}, (owner {2}) to self.scheduler ({1})'.format(condition, self.scheduler, owner))
        condition.owner = owner
        condition.scheduler = self.scheduler
        self.conditions[owner] = condition

    def add_condition_set(self, conditions):
        """
        :param: self:
        :param conditions: a :keyword:`dict` mapping :keyword:`Component`s to :keyword:`Condition`s, can be added later with :keyword:`add_condition`
        """
        for owner in conditions:
            conditions[owner].owner = owner
            conditions[owner].scheduler = self.scheduler
            self.conditions[owner] = conditions[owner]

# ********************************************* Condition ***************************************************************

"""

Overview
--------
`Condition`<Condition>s represent any conditions that can be satisfied. Each Condition is associated with an owner (a
`Component`<Component> that the Condition "belongs" to, relevant for relative conditions such as `EveryNCalls`), and
a `Scheduler`<Scheduler>, which maintains most of the data required to test satisfaction. These properties can usually
be determined automatically based on the context in which Conditions are created.

Creating new Conditions
-----------------------
Each Condition must
    - be a subclass of `Condition`<Condition>
    - pass `dependencies` as the first argument to the __init__ function of Condition
    - pass `func` as the second argument to the __init__ function of Condition

In determining whether a Condition is satisfied, `func` is called with `dependencies` as parameter (and optionally,
additional named and unnamed arguments).

Hint:
    If you do not want to use the dependencies parameter, and instead want to use only args or kwargs, you may
    pass a dummy variable for dependencies. See `AfterNCallsCombined`<AfterNCallsCombined> for reference:

    class AfterNCallsCombined(Condition):
        def __init__(self, *dependencies, n=None, time_scale=TimeScale.TRIAL):
            def func(_none, *dependencies, n=None):
                if self.scheduler is None:
                    raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
                if n is None:
                    raise ConditionError('{0}: keyword argument n is None'.format(type(self).__name__))
                count_sum = 0
                for d in dependencies:
                    count_sum += self.scheduler.counts_total[time_scale][d]
                return count_sum >= n
            super().__init__(None, func, *dependencies, n=n)

"""

class Condition(object):
    def __init__(self, dependencies, func, *args, **kwargs):
        """
        :param self:
        :param dependencies: one or more PNL objects over which func is evaluated to determine satisfaction of the :keyword:`Condition`
            user must ensure that dependencies are suitable as func parameters
        :param func: parameters over which func is evaluated to determine satisfaction of the :keyword:`Condition`
        :param args: additional formal arguments passed to func
        :param kwargs: additional keyword arguments passed to func
        """
        self.dependencies = dependencies
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._scheduler = None
        self._owner = None
        #logger.debug('{1} dependencies: {0}'.format(dependencies, type(self).__name__))

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        logger.debug('Condition ({0}) setting scheduler to {1}'.format(type(self).__name__, value))
        self._scheduler = value

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        logger.debug('Condition ({0}) setting owner to {1}'.format(type(self).__name__, value))
        self._owner = value

    def is_satisfied(self):
        logger.debug('Condition ({0}) using scheduler {1}'.format(type(self).__name__, self.scheduler))
        has_args = len(self.args) > 0
        has_kwargs = len(self.kwargs) > 0

        if has_args and has_kwargs:
            return self.func(self.dependencies, *self.args, **self.kwargs)
        if has_args:
            return self.func(self.dependencies, *self.args)
        if has_kwargs:
            return self.func(self.dependencies, **self.kwargs)
        return self.func(self.dependencies)

######################################################################
# Included Conditions
######################################################################

# TODO: create this class to subclass All and Any from
#class CompositeCondition(Condition):
    #def

class All(Condition):
    """
    All

    Parameters:
        - args (argtuple): one or more :keyword:`Condition`s

    Satisfied when:
        - All args are satisfied

    Notes:
        To initialize with a list (for example), conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list], unpack the list to supply its members as args

        composite_condition = All(*conditions)
    """
    def __init__(self, *args):
        """
        :param self:
        :param args: one or more :keyword:`Condition`s, all of which must be satisfied to satisfy this composite condition
        """
        super().__init__(args, self.satis)

    @Condition.scheduler.setter
    def scheduler(self, value):
        for cond in self.dependencies:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.dependencies:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, conds):
        for cond in conds:
            if not cond.is_satisfied():
                return False
        return True

class Any(Condition):
    """
    Any

    Parameters:
        - args: one or more :keyword:`Condition`s

    Satisfied when:
        - All args are satisfied

    Notes:
        To initialize with a list (for example), conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list], unpack the list to supply its members as args

        composite_condition = Any(*conditions)
    """
    def __init__(self, *args):
        """
        :param self:
        :param args: one or more :keyword:`Condition`s, any of which must be satisfied to satisfy this composite condition
        """
        super().__init__(args, self.satis)

    @Condition.scheduler.setter
    def scheduler(self, value):
        logger.debug('Any setter args: {0}'.format(self.dependencies))
        for cond in self.dependencies:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.dependencies:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, conds):
        for cond in conds:
            if cond.is_satisfied():
                return True
        return False

class Not(Condition):
    """
    Not

    Parameters:
        - condition (Condition): a :keyword:`Condition`

    Satisfied when:
        - condition is not satisfied

    Notes:

    """
    def __init__(self, condition):
        super().__init__(condition, lambda c: not c.is_satisfied())

    @Condition.scheduler.setter
    def scheduler(self, value):
        self.dependencies.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        self.dependencies.owner = value

class Always(Condition):
    """
    Always

    Parameters:

    Satisfied when:
        - always satisfied

    Notes:

    """
    def __init__(self):
        super().__init__(True, lambda x: x)

class Never(Condition):
    """
    Never

    Parameters:

    Satisfied when:
        - never satisfied

    Notes:

    """
    def __init__(self):
        super().__init__(False, lambda x: x)

class AtPass(Condition):
    """
    AtPass

    Parameters:
        - n (int): the pass at which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - within the scope of time_scale, exactly n passes have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Pass is pass 0, the second Pass is pass 1, etc.). So,
        AtPass(1) is satisfied when one pass (pass 0) has already occurred.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.PASS] == n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.format(type(self).__name__, e, time_scale))
        super().__init__(n, func)

class AfterPass(Condition):
    """
    AfterPass

    Parameters:
        - n (int): the pass after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - within the scope of time_scale, at least n+1 passes have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Pass is pass 0, the second Pass is pass 1, etc.). So,
        AfterPass(1) is satisfied after pass 1 has occurred, at pass 2, pass 3, pass 4, etc.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] >= n+1
        super().__init__(n, func, time_scale)

class AfterNCalls(Condition):
    """
    AfterNCalls

    Parameters:
        - dependency (Component):
        - n (int): the number of executions of dependency after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependency. Defaults to TimeScale.TRIAL

    Satisfied when:
        - dependency has been executed at least n+1 times within the scope of time_scale

    Notes:

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls >= n
        super().__init__(dependency, func, n)

class AfterNCallsCombined(Condition):
    """
    AfterNCallsCombined

    Parameters:
        - *dependencies (Components): variable length
        - n (int): the number of executions of all dependencies after which this condition will be satisfied. Defaults to None
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependencies. Defaults to TimeScale.TRIAL

    Satisfied when:
        - Among all dependencies, there have been at least n+1 executions within the scope of time_scale

    Notes:

    """
    def __init__(self, *dependencies, n=None, time_scale=TimeScale.TRIAL):
        logger.debug('{0} args: deps {1}, n {2}, ts {3}'.format(type(self).__name__, dependencies, n, time_scale))
        def func(_none, *dependencies, n=None):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            if n is None:
                raise ConditionError('{0}: keyword argument n is None'.format(type(self).__name__))
            count_sum = 0
            for d in dependencies:
                count_sum += self.scheduler.counts_total[time_scale][d]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(d, self.scheduler.counts_total[time_scale][d], time_scale.name))
            return count_sum >= n
        super().__init__(None, func, *dependencies, n=n)

class AfterNTrials(Condition):
    """
    AfterNTrials

    Parameters:
        - n (int): the number of TimeScale.TRIALs after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting trials. Defaults to TimeScale.RUN

    Satisfied when:
        - the count of TimeScale.TRIALs within time_scale is at least n

    Notes:

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.TRIAL] >= n
        super().__init__(n, func, time_scale)

class AllHaveRun(Condition):
    """
    AllHaveRun

    Parameters:
        - *dependencies (Components): variable length
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependencies. Defaults to TimeScale.TRIAL

    Satisfied when:
        - All dependencies have been executed at least 1 time within the scope of time_scale

    Notes:

    """
    def __init__(self, *dependencies, time_scale=TimeScale.TRIAL):
        def func(_none, *dependencies):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                if self.scheduler.counts_total[time_scale][d] < 1:
                    return False
            return True
        super().__init__(None, func, *dependencies)

class AtNCalls(Condition):
    """
    AtNCalls

    Parameters:
        - dependency (Component):
        - n (int): the number of executions of dependency at which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependency. Defaults to TimeScale.TRIAL

    Satisfied when:
        - dependency has been executed exactly n times within the scope of time_scale

    Notes:

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls == n
        super().__init__(dependency, func, n)

class BeforePass(Condition):
    """
    BeforePass

    Parameters:
        - n (int): the pass after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - within the scope of time_scale, at most n-1 passes have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Pass is pass 0, the second Pass is pass 1, etc.). So,
        BeforePass(2) is satisfied at pass 0 and pass 1

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] < n
        super().__init__(n, func, time_scale)

class EveryNPasses(Condition):
    """
    EveryNPasses

    Parameters:
        - n (int): the frequency of passes with which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - the number of passes that has occurred within time_scale is evenly divisible by n

    Notes:
        All EveryNPasses conditions will be satisfied at pass 0

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] % n == 0
        super().__init__(n, func, time_scale)

class EveryNCalls(Condition):
    """
    EveryNCalls

    Parameters:
        - dependency (Component):
        - n (int): the frequency of executions of dependency with which this condition will be satisfied

    Satisfied when:
        - since the last time this conditon's owner was called, the number of calls of dependency is at least n

    Notes:
        Whenever a Component is run, the Scheduler's count of each dependency that is "useable" by the Component is
        reset to 0

    """
    def __init__(self, dependency, n):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_useable[dependency][self.owner]
            logger.debug('{0} has reached {1} num_calls'.format(dependency, num_calls))
            return num_calls >= n
        super().__init__(dependency, func, n)

class JustRan(Condition):
    """
    JustRan

    Parameters:
        - dependency (Component):

    Satisfied when:
        - dependency has been run (or told to run) in the previous TimeScale.TIME_STEP

    Notes:
        This condition can transcend divisions between TimeScales. That is, if A runs in the final time step in a trial,
        JustRan(A) will be satisfied at the beginning of the next trial.

    """
    def __init__(self, dependency):
        def func(dependency):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            logger.debug('checking if {0} in previous execution step set'.format(dependency))
            try:
                return dependency in self.scheduler.execution_list[-1]
            except TypeError:
                return dependency == self.scheduler.execution_list[-1]
        super().__init__(dependency, func)

class WhenFinished(Condition):
    """
    WhenFinished

    Parameters:
        - dependency (Component):

    Satisfied when:
        - dependency has "finished" (i.e. its is_finished attribute is True)

    Notes:
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, dependency):
        def func(dependency):
            try:
                return dependency.is_finished
            except AttributeError as e:
                raise ConditionError('WhenFinished: Unsupported dependency type: {0}; ({1})'.format(type(dependency), e))

        super().__init__(dependency, func)

class WhenFinishedAny(Condition):
    """
    WhenFinishedAny

    Parameters:
        - *dependencies (Components): variable length

    Satisfied when:
        - any of the dependencies have "finished" (i.e. its is_finished attribute is True)

    Notes:
        This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to Any(WhenFinished(A), WhenFinished(B), WhenFinished(C))
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, *dependencies):
        def func(_none, *dependencies):
            for d in dependencies:
                try:
                    if d.is_finished:
                        return True
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAny: Unsupported dependency type: {0}; ({1})'.format(type(dependency), e))
            return False

        super().__init__(None, *dependencies, func)

class WhenFinishedAll(Condition):
    """
    WhenFinishedAll

    Parameters:
        - *dependencies (Components): variable length

    Satisfied when:
        - all of the dependencies have "finished" (i.e. its is_finished attribute is True)

    Notes:
        This is a convenience class; WhenFinishedAll(A, B, C) is equivalent to All(WhenFinished(A), WhenFinished(B), WhenFinished(C))
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, *dependencies):
        def func(_none, *dependencies):
            for d in dependencies:
                try:
                    if not d.is_finished:
                        return False
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAll: Unsupported dependency type: {0}; ({1})'.format(type(dependency), e))
            return True

        super().__init__(None, func, *dependencies)
