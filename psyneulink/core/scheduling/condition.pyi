import enum
from typing import Callable, ClassVar, Dict, Hashable, Iterable, Set, Union

import _abc
import graph_scheduler.time
import pint
from _typeshed import Incomplete

__all__ = ['Operation', 'ConditionError', 'ConditionSet', 'ConditionBase', 'Condition', 'AbsoluteCondition', 'While', 'When', 'WhileNot', 'Always', 'Never', 'CompositeCondition', 'All', 'Any', 'And', 'Or', 'Not', 'NWhen', 'TimeInterval', 'TimeTermination', 'BeforeTimeStep', 'AtTimeStep', 'AfterTimeStep', 'AfterNTimeSteps', 'BeforePass', 'AtPass', 'AfterPass', 'AfterNPasses', 'EveryNPasses', 'BeforeTrial', 'AtTrial', 'AfterTrial', 'AfterNTrials', 'AtRun', 'AfterRun', 'AfterNRuns', 'BeforeNCalls', 'AtNCalls', 'AfterCall', 'AfterNCalls', 'AfterNCallsCombined', 'EveryNCalls', 'JustRan', 'AllHaveRun', 'WhenFinished', 'WhenFinishedAny', 'WhenFinishedAll', 'AtTrialStart', 'AtTrialNStart', 'AtRunStart', 'AtRunNStart', 'Threshold', 'GraphStructureCondition', 'CustomGraphStructureCondition', 'BeforeNodes', 'BeforeNode', 'WithNode', 'AfterNodes', 'AfterNode', 'AddEdgeTo', 'RemoveEdgeFrom']


SubjectOperation = Union['Operation', str, Dict[Hashable, Union['Operation', str]]]
ConditionSetDict = Dict[Hashable, Union['ConditionBase', Iterable['ConditionBase']]]
GraphDependencyDict = Dict[Hashable, Set[Hashable]]


class Operation(enum.Enum):

    """
    Used in conjunction with `GraphStructureCondition` to indicate how a
    set of source nodes (**S** below) should be combined with a set of
    comparison nodes (**C** below) to produce a result set. Many
    Operations correspond to standard set operations.

    Each enum item can be called with a source set and comparison set as
    arguments to produce the result set.

    Attributes:
        KEEP: Returns **S**

        REPLACE: Returns **C**

        DISCARD: Returns the empty set

        INTERSECTION: Returns the set of items that are in both **S**
            and **C**

        UNION: Returns the set of items in either **S** or **C**

        MERGE: Returns the set of items in either **S** or **C**

        DIFFERENCE: Returns the set of items in **S** but not **C**

        INVERSE_DIFFERENCE: Returns the set of items in **C** but not
            **S**

        SYMMETRIC_DIFFERENCE: Returns the set of items that are in one
            of **S** or **C** but not both

    """
    _member_names_: ClassVar[list] = ...
    _member_map_: ClassVar[dict] = ...
    _member_type_: ClassVar[type[object]] = ...
    _value2member_map_: ClassVar[dict] = ...
    KEEP: ClassVar[Operation] = ...
    REPLACE: ClassVar[Operation] = ...
    DISCARD: ClassVar[Operation] = ...
    INTERSECTION: ClassVar[Operation] = ...
    UNION: ClassVar[Operation] = ...
    MERGE: ClassVar[Operation] = ...
    DIFFERENCE: ClassVar[Operation] = ...
    INVERSE_DIFFERENCE: ClassVar[Operation] = ...
    SYMMETRIC_DIFFERENCE: ClassVar[Operation] = ...
    def __call__(self, source_neighbors: set[Hashable], comparison_neighbors: set[Hashable]) -> set[Hashable]:
        """
        Returns the set resulting from applying an `Operation` on
        **source_neighbors** and **comparison_neighbors**

        Args:
            source_neighbors (Set[Hashable])
            comparison_neighbors (Set[Hashable])

        Returns:
            Set[Hashable]
        """
    @classmethod
    def __init__(cls, value) -> None: ...

class ConditionError(Exception):
    __init__: ClassVar[wrapper_descriptor] = ...

class ConditionSet:

    """Used in conjunction with a `Scheduler <graph_scheduler.scheduler.Scheduler>` to store the `Conditions <Condition>` associated with a node.

    Arguments
    ---------

    *condition_sets
        each item is a dict or ConditionSet mapping nodes to one or more
        conditions to be added via `ConditionSet.add_condition_set`

    conditions
        a dict or ConditionSet mapping nodes to one or more conditions
        to be added via `ConditionSet.add_condition_set`. Maintained for
        backwards compatibility with versions 1.x

    Attributes
    ----------

    conditions : Dict[Hashable: Union[ConditionBase, Iterable[ConditionBase]]]
        the key of each entry is a node, and its value is a condition
        associated
        with that node.  Conditions can be added to the
        ConditionSet using the ConditionSet's `add_condition` method.

    conditions_basic : Dict[Hashable: `Condition <graph_scheduler.condition.Condition>`]
        a dict mapping nodes to their single `basic Conditions
        <graph_scheduler.condition.Condition>`

    conditions_structural : Dict[Hashable: List[`GraphStructureCondition`]]
        a dict mapping nodes to their `graph structure Conditions
        <GraphStructureCondition>`

    structural_condition_order : List[`GraphStructureCondition`]
        a list storing all `GraphStructureCondition` s in this
        ConditionSet in the order in which they were added (and will be
        applied to a `Scheduler`)

    """
    def __init__(self, *condition_sets: ConditionSetDict, conditions: ConditionSetDict = ...) -> None: ...
    def __contains__(self, item) -> bool: ...
    def __iter__(self): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def add_condition(self, owner: Hashable, condition: ConditionBase):
        """
        Adds a `basic <graph_scheduler.condition.Condition>` or `graph
        structure <GraphStructureCondition>` Condition to the
        ConditionSet.

        If **condition** is basic, it will overwrite the current basic
        Condition for **owner**, if present. If you want to add multiple
        basic Conditions to a single owner, instead add a single
        `Composite Condition <Conditions_Composite>` to accurately
        specify the desired behavior.

        Arguments
        ---------

        owner : node
            specifies the node with which the **condition** should be associated. **condition**
            will govern the execution behavior of **owner**

        condition : ConditionBase
            specifies the condition associated with **owner** to be
            added to the ConditionSet.
        """
    def remove_condition(self, owner_or_condition: Hashable | ConditionBase) -> ConditionBase | None:
        """
        Removes the condition specified as or owned by
        **owner_or_condition**.

        Args:
            owner_or_condition (Union[Hashable, `ConditionBase`]):
                Either a condition or the owner of a condition

        Returns:
            The condition removed, or None if no condition removed

        Raises:
            ConditionError:
                - when **owner_or_condition** is an owner and it owns
                  multiple conditions
                - when **owner_or_condition** is a condition and its
                  owner is None
        """
    def add_condition_set(self, conditions: ConditionSet | ConditionSetDict):
        """
        Adds a set of `basic <graph_scheduler.condition.Condition>` or
        `graph structure <GraphStructureCondition>` Conditions (in the
        form of a dict or another ConditionSet) to the ConditionSet.

        Any basic Condition added here will overwrite the current basic
        Condition for a given owner, if present. If you want to add
        multiple basic Conditions to a single owner, instead add a
        single `Composite Condition <Conditions_Composite>` to
        accurately specify the desired behavior.

        Arguments
        ---------

        conditions
            specifies collection of Conditions to be added to this ConditionSet,

            if a dict is provided:
                each entry should map an owner node (the node whose
                execution behavior will be governed) to a `Condition
                <graph_scheduler.condition.Condition>` or
                `GraphStructureCondition`, or an iterable of them.

        """
    @property
    def conditions(self): ...

class ConditionBase:

    """
    Abstract base class for `basic conditions
    <graph_scheduler.condition.Condition>` and `graph structure
    conditions <GraphStructureCondition>`

    Attributes:
        owner (Hashable):
            the node with which the Condition is associated, and the
            execution of which it determines.

    """
    owner: Incomplete
    def __init__(self, _owner: Hashable = ..., **kwargs) -> None: ...

class Condition(ConditionBase):

    """
    Used in conjunction with a :class:`Scheduler` to specify the condition under which a node should be
    allowed to execute.

    Arguments
    ---------

    func : callable
        specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

    args : *args
        specifies formal arguments to pass to `func` when the Condition is evaluated.

    kwargs : **kwargs
        specifies keyword arguments to pass to `func` when the Condition is evaluated.
    """
    def __init__(self, func, *args, **kwargs) -> None: ...
    def is_satisfied(self, *args, execution_id: Incomplete | None = ..., **kwargs):
        """
        the function called to determine satisfaction of this Condition.

        Arguments
        ---------
        args : *args
            specifies additional formal arguments to pass to `func` when the Condition is evaluated.
            these are appended to the **args** specified at instantiation of this Condition

        kwargs : **kwargs
            specifies additional keyword arguments to pass to `func` when the Condition is evaluated.
            these are added to the **kwargs** specified at instantiation of this Condition

        Returns
        -------
            True - if the Condition is satisfied
            False - if the Condition is not satisfied
        """
    @property
    def absolute_intervals(self): ...
    @property
    def absolute_fixed_points(self): ...
    @property
    def is_absolute(self): ...

class AbsoluteCondition(Condition):
    def __init__(self, func, *args, **kwargs) -> None: ...
    @property
    def is_absolute(self): ...

class _DependencyValidation: ...
class While(ConditionBase):

    """
    Used in conjunction with a :class:`Scheduler` to specify the condition under which a node should be
    allowed to execute.

    Arguments
    ---------

    func : callable
        specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

    args : *args
        specifies formal arguments to pass to `func` when the Condition is evaluated.

    kwargs : **kwargs
        specifies keyword arguments to pass to `func` when the Condition is evaluated.
    """
    def __init__(self, func, *args, **kwargs) -> None: ...
    def is_satisfied(self, *args, execution_id: Incomplete | None = ..., **kwargs):
        """
        the function called to determine satisfaction of this Condition.

        Arguments
        ---------
        args : *args
            specifies additional formal arguments to pass to `func` when the Condition is evaluated.
            these are appended to the **args** specified at instantiation of this Condition

        kwargs : **kwargs
            specifies additional keyword arguments to pass to `func` when the Condition is evaluated.
            these are added to the **kwargs** specified at instantiation of this Condition

        Returns
        -------
            True - if the Condition is satisfied
            False - if the Condition is not satisfied
        """
    @property
    def absolute_intervals(self): ...
    @property
    def absolute_fixed_points(self): ...
    @property
    def is_absolute(self): ...

class When(ConditionBase):

    """
    Used in conjunction with a :class:`Scheduler` to specify the condition under which a node should be
    allowed to execute.

    Arguments
    ---------

    func : callable
        specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

    args : *args
        specifies formal arguments to pass to `func` when the Condition is evaluated.

    kwargs : **kwargs
        specifies keyword arguments to pass to `func` when the Condition is evaluated.
    """
    def __init__(self, func, *args, **kwargs) -> None: ...
    def is_satisfied(self, *args, execution_id: Incomplete | None = ..., **kwargs):
        """
        the function called to determine satisfaction of this Condition.

        Arguments
        ---------
        args : *args
            specifies additional formal arguments to pass to `func` when the Condition is evaluated.
            these are appended to the **args** specified at instantiation of this Condition

        kwargs : **kwargs
            specifies additional keyword arguments to pass to `func` when the Condition is evaluated.
            these are added to the **kwargs** specified at instantiation of this Condition

        Returns
        -------
            True - if the Condition is satisfied
            False - if the Condition is not satisfied
        """
    @property
    def absolute_intervals(self): ...
    @property
    def absolute_fixed_points(self): ...
    @property
    def is_absolute(self): ...

class WhileNot(Condition):

    """
    WhileNot

    Parameters:

        func : callable
            specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

        args : *args
            specifies formal arguments to pass to `func` when the Condition is evaluated.

        kwargs : **kwargs
            specifies keyword arguments to pass to `func` when the Condition is evaluated.

    Satisfied when:

        - **func** is False

    """
    def __init__(self, func, *args, **kwargs) -> None: ...

class Always(Condition):

    """Always

    Parameters:

        none

    Satisfied when:

        - always satisfied.

    """
    def __init__(self) -> None: ...

class Never(Condition):

    """Never

    Parameters:

        none

    Satisfied when:

        - never satisfied.
    """
    def __init__(self) -> None: ...

class CompositeCondition(Condition):
    owner: Incomplete
    def __init__(self, func, *args, **kwargs) -> None: ...
    @property
    def absolute_intervals(self): ...
    @property
    def absolute_fixed_points(self): ...
    @property
    def is_absolute(self): ...

class All(CompositeCondition):

    """All

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - all of the Conditions in args are satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(node, 5) for node in node_list]

          unpack the list to supply its members as args::

           composite_condition = All(*conditions)

    """
    def __init__(self, *args, **dependencies) -> None: ...
    def satis(self, *conds, **kwargs): ...

class Any(CompositeCondition):

    """Any

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - one or more of the Conditions in **args** is satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(node, 5) for node in node_list]

          unpack the list to supply its members as args::

           composite_condition = Any(*conditions)

    """
    def __init__(self, *args, **dependencies) -> None: ...
    def satis(self, *conds, **kwargs): ...

class And(CompositeCondition):

    """All

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - all of the Conditions in args are satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(node, 5) for node in node_list]

          unpack the list to supply its members as args::

           composite_condition = All(*conditions)

    """
    def __init__(self, *args, **dependencies) -> None: ...
    def satis(self, *conds, **kwargs): ...

class Or(CompositeCondition):

    """Any

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - one or more of the Conditions in **args** is satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(node, 5) for node in node_list]

          unpack the list to supply its members as args::

           composite_condition = Any(*conditions)

    """
    def __init__(self, *args, **dependencies) -> None: ...
    def satis(self, *conds, **kwargs): ...

class Not(Condition):

    """Not

    Parameters:

        condition(Condition): a `Condition`

    Satisfied when:

        - **condition** is not satisfied.

    """
    owner: Incomplete
    def __init__(self, condition) -> None: ...

class NWhen(Condition):

    """NWhen

    Parameters:

        condition(Condition): a `Condition`

        n(int): the maximum number of times this condition will be satisfied

    Satisfied when:

        - the first **n** times **condition** is satisfied upon evaluation

    """
    owner: Incomplete
    def __init__(self, condition, n: int = ...) -> None: ...
    def satis(self, condition, n, *args, scheduler: Incomplete | None = ..., execution_id: Incomplete | None = ..., **kwargs): ...

class TimeInterval(AbsoluteCondition):

    """TimeInterval

    Attributes:

        repeat
            the interval between *unit*s where this condition can be
            satisfied

        start
            the time at/after which this condition can be
            satisfied

        end
            the time at/fter which this condition can be
            satisfied

        unit
            the `pint.Unit` to use for scalar values of *repeat*,
            *start*, and *end*

        start_inclusive
            if True, *start* allows satisfaction exactly at the time
            corresponding to *start*. if False, satisfaction can occur
            only after *start*

        end_inclusive
            if True, *end* allows satisfaction exactly until the time
            corresponding to *end*. if False, satisfaction can occur
            only before *end*


    Satisfied when:

        Every *repeat* units of time at/after *start* and before/through
        *end*

    Notes:

        Using a `TimeInterval` as a
        `termination Condition <Scheduler_Termination_Conditions>` may
        result in unexpected behavior. The user may be inclined to
        create **TimeInterval(end=x)** to terminate at time **x**, but
        this will do the opposite and be true only and always until time
        **x**, terminating at any time before **x**. If in doubt, use
        `TimeTermination` instead.

        If the scheduler is not set to `exact_time_mode = True`,
        *start_inclusive* and *end_inclusive* may not behave as
        expected. See `Scheduler_Exact_Time` for more info.
    """
    def __init__(self, repeat: int | str | pint.Quantity = ..., start: int | str | pint.Quantity = ..., end: int | str | pint.Quantity = ..., unit: str | pint.Unit = ..., start_inclusive: bool = ..., end_inclusive: bool = ...) -> None: ...
    @property
    def absolute_intervals(self): ...
    @property
    def absolute_fixed_points(self): ...

class TimeTermination(AbsoluteCondition):

    """TimeTermination

    Attributes:

        t
            the time at/after which this condition is satisfied

        unit
            the `pint.Unit` to use for scalar values of *t*, *start*,
            and *end*

        start_inclusive
            if True, the condition is satisfied exactly at the time
            corresponding to *t*. if False, satisfaction can occur
            only after *t*

    Satisfied when:

        At/After time *t*
    """
    def __init__(self, t: int | str | pint.Quantity, inclusive: bool = ..., unit: str | pint.Unit = ...) -> None: ...
    @property
    def absolute_fixed_points(self): ...

class BeforeTimeStep(Condition):

    """BeforeTimeStep

    Parameters:

        n(int): the 'TIME_STEP' before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TIME_STEP`\\ s (default: TimeScale.TRIAL)

    Satisfied when:

        - at most n-1 `TIME_STEP`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TIME_STEP` is 0, the second `TIME_STEP` is 1, etc.);
          so, `BeforeTimeStep(2)` is satisfied at `TIME_STEP` 0 and `TIME_STEP` 1.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AtTimeStep(Condition):

    """AtTimeStep

    Parameters:

        n(int): the `TIME_STEP` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TIME_STEP`\\ s (default: TimeScale.TRIAL)

    Satisfied when:

        - exactly n `TIME_STEP`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first 'TIME_STEP' is pass 0, the second 'TIME_STEP' is 1, etc.);
          so, `AtTimeStep(1)` is satisfied when a single `TIME_STEP` (`TIME_STEP` 0) has occurred, and `AtTimeStep(2)` is satisfied
          when two `TIME_STEP`\\ s have occurred (`TIME_STEP` 0 and `TIME_STEP` 1), etc..

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterTimeStep(Condition):

    """AfterTimeStep

    Parameters:

        n(int): the `TIME_STEP` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TIME_STEP`\\ s (default: TimeScale.TRIAL)

    Satisfied when:

        - at least n+1 `TIME_STEP`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScals are zero-indexed (that is, the first `TIME_STEP` is 0, the second `TIME_STEP` is 1, etc.); so,
          `AfterTimeStep(1)` is satisfied after `TIME_STEP` 1 has occurred and thereafter (i.e., in `TIME_STEP`\\ s 2, 3, 4, etc.).

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterNTimeSteps(Condition):

    """AfterNTimeSteps

    Parameters:

        n(int): the number of `TIME_STEP`\\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TIME_STEP`\\ s (default: TimeScale.TRIAL)


    Satisfied when:

        - at least n `TIME_STEP`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class BeforePass(Condition):

    """BeforePass

    Parameters:

        n(int): the 'PASS' before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - at most n-1 `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `PASS` is 0, the second `PASS` is 1, etc.);
          so, `BeforePass(2)` is satisfied at `PASS` 0 and `PASS` 1.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AtPass(Condition):

    """AtPass

    Parameters:

        n(int): the `PASS` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - exactly n `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first 'PASS' is pass 0, the second 'PASS' is 1, etc.);
          so, `AtPass(1)` is satisfied when a single `PASS` (`PASS` 0) has occurred, and `AtPass(2)` is satisfied
          when two `PASS`\\ es have occurred (`PASS` 0 and `PASS` 1), etc..

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterPass(Condition):

    """AfterPass

    Parameters:

        n(int): the `PASS` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - at least n+1 `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `PASS` is 0, the second `PASS` is 1, etc.); so,
          `AfterPass(1)` is satisfied after `PASS` 1 has occurred and thereafter (i.e., in `PASS`\\ es 2, 3, 4, etc.).

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterNPasses(Condition):

    """AfterNPasses

    Parameters:

        n(int): the number of `PASS`\\ es after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.TRIAL)


    Satisfied when:

        - at least n `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class EveryNPasses(Condition):

    """EveryNPasses

    Parameters:

        n(int): the frequency of passes with which this condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - `PASS` 0

        - the specified number of `PASS`\\ es that has occurred within a unit of time (at the `TimeScale` specified by
          **time_scale**) is evenly divisible by n.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class BeforeTrial(Condition):

    """BeforeTrial

    Parameters:

        n(int): the `TRIAL <TimeScale.TRIAL>` before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL <TimeScale.TRIAL>`\\ s
        (default: TimeScale.RUN)

    Satisfied when:

        - at most n-1 `TRIAL <TimeScale.TRIAL>`\\ s have occurred within one unit of time at the `TimeScale`
          specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL <TimeScale.TRIAL>` is 0, the second
          `TRIAL <TimeScale.TRIAL>` is 1, etc.); so, `BeforeTrial(2)` is satisfied at `TRIAL <TimeScale.TRIAL>` 0
          and `TRIAL <TimeScale.TRIAL>` 1.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AtTrial(Condition):

    """AtTrial

    Parameters:

        n(int): the `TRIAL <TimeScale.TRIAL>` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL <TimeScale.TRIAL>`\\ s
        (default: TimeScale.RUN)

    Satisfied when:

        - exactly n `TRIAL <TimeScale.TRIAL>`\\ s have occurred within one unit of time at the `TimeScale`
          specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL <TimeScale.TRIAL>` is 0,
          the second `TRIAL <TimeScale.TRIAL>` is 1, etc.); so, `AtTrial(1)` is satisfied when one
          `TRIAL <TimeScale.TRIAL>` (`TRIAL <TimeScale.TRIAL>` 0) has already occurred.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterTrial(Condition):

    """AfterTrial

    Parameters:

        n(int): the `TRIAL <TimeScale.TRIAL>` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL <TimeScale.TRIAL>`\\ s.
        (default: TimeScale.RUN)

    Satisfied when:

        - at least n+1 `TRIAL <TimeScale.TRIAL>`\\ s have occurred within one unit of time at the `TimeScale`
          specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL <TimeScale.TRIAL>` is 0, the second
        `TRIAL <TimeScale.TRIAL>` is 1, etc.); so,  `AfterPass(1)` is satisfied after `TRIAL <TimeScale.TRIAL>` 1
        has occurred and thereafter (i.e., in `TRIAL <TimeScale.TRIAL>`\\ s 2, 3, 4, etc.).

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterNTrials(Condition):

    """AfterNTrials

    Parameters:

        n(int): the number of `TRIAL <TimeScale.TRIAL>`\\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL <TimeScale.TRIAL>`\\ s
        (default: TimeScale.RUN)

    Satisfied when:

        - at least n `TRIAL <TimeScale.TRIAL>`\\ s have occured within one unit of time at the `TimeScale`
          specified by **time_scale**.

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AtRun(Condition):

    """AtRun

    Parameters:

        n(int): the `RUN` at which the Condition is satisfied

    Satisfied when:

        - exactly n `RUN`\\ s have occurred.

    Notes:
        - `RUN`\\ s are managed by the environment         using the Scheduler (e.g.         `end_environment_sequence <Scheduler.end_environment_sequence>`        ) and are not automatically updated by this package.

    """
    def __init__(self, n) -> None: ...

class AfterRun(Condition):

    """AfterRun

    Parameters:

        n(int): the `RUN` after which the Condition is satisfied

    Satisfied when:

        - at least n+1 `RUN`\\ s have occurred.

    Notes:
        - `RUN`\\ s are managed by the environment         using the Scheduler (e.g.         `end_environment_sequence <Scheduler.end_environment_sequence>`        ) and are not automatically updated by this package.

    """
    def __init__(self, n) -> None: ...

class AfterNRuns(Condition):

    """AfterNRuns

    Parameters:

        n(int): the number of `RUN`\\ s after which the Condition is satisfied

    Satisfied when:

        - at least n `RUN`\\ s have occured.

    Notes:
        - `RUN`\\ s are managed by the environment         using the Scheduler (e.g.         `end_environment_sequence <Scheduler.end_environment_sequence>`        ) and are not automatically updated by this package.

    """
    def __init__(self, n) -> None: ...

class BeforeNCalls(_DependencyValidation, Condition):

    """BeforeNCalls

    Parameters:

        dependency (Hashable):  the node on which the Condition depends

        n(int): the number of executions of **dependency** before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the node specified in **dependency** has executed at most n-1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AtNCalls(_DependencyValidation, Condition):

    """AtNCalls

    Parameters:

        dependency (Hashable):  the node on which the Condition depends

        n(int): the number of executions of **dependency** at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the node specified in **dependency** has executed exactly n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterCall(_DependencyValidation, Condition):

    """AfterCall

    Parameters:

        dependency (Hashable):  the node on which the Condition depends

        n(int): the number of executions of **dependency** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the node specified in **dependency** has executed at least n+1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterNCalls(_DependencyValidation, Condition):

    """AfterNCalls

    Parameters:

        dependency (Hashable):  the node on which the Condition depends

        n(int): the number of executions of **dependency** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the node specified in **dependency** has executed at least n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AfterNCallsCombined(_DependencyValidation, Condition):

    """AfterNCallsCombined

    Parameters:

        *dependencies (Hashable):  one or more nodes on which the Condition depends

        n(int): the number of combined executions of all nodes specified in **dependencies** after which the
        Condition is satisfied (default: None)

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.TRIAL)


    Satisfied when:

        - there have been at least n+1 executions among all of the nodes specified in **dependencies**
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, n: Incomplete | None = ..., time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class EveryNCalls(_DependencyValidation, Condition):

    '''EveryNCalls

    Parameters:

        dependency (Hashable):  the node on which the Condition depends

        n(int): the frequency of executions of **dependency** at which the Condition is satisfied


    Satisfied when:

        - the node specified in **dependency** has executed at least n times since the last time the
          Condition\'s owner executed.


    Notes:

        - scheduler\'s count of each other node that is "useable" by the node is reset to 0 when the
          node runs

    '''
    def __init__(self, dependency, n) -> None: ...

class JustRan(_DependencyValidation, Condition):

    """JustRan

    Parameters:

        dependency (Hashable):  the node on which the Condition depends

    Satisfied when:

        - the node specified in **dependency** executed in the previous `TIME_STEP`.

    Notes:

        - This Condition can transcend divisions between `TimeScales <TimeScale>`.
          For example, if A runs in the final `TIME_STEP` of an `TRIAL <TimeScale.TRIAL>`,
          JustRan(A) is satisfied at the beginning of the next `TRIAL <TimeScale.TRIAL>`.

    """
    def __init__(self, dependency) -> None: ...

class AllHaveRun(_DependencyValidation, Condition):

    """AllHaveRun

    Parameters:

        *dependencies (Hashable):  an iterable of nodes on which the Condition depends

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - all of the nodes specified in **dependencies** have executed at least once
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class WhenFinished(_DependencyValidation, Condition):

    """WhenFinished

    Parameters:

        dependency (Hashable):  the node on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of the node specified in **dependencies** returns `True`.

    Notes:

        - This is a dynamic Condition: Each node is responsible for managing its finished status on its
          own, which can occur independently of the execution of other nodes.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

        - The is_finished method is called with `execution_id` as its           sole positional argument

    """
    def __init__(self, dependency) -> None: ...

class WhenFinishedAny(_DependencyValidation, Condition):

    """WhenFinishedAny

    Parameters:

        *dependencies (Hashable):  zero or more nodes on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of any nodes specified in **dependencies** returns `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          Any(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no nodes are specified, the condition will default to checking all of scheduler's nodes.

        - This is a dynamic Condition: Each node is responsible for managing its finished status on its
          own, which can occur independently of the execution of other nodes.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

        - The is_finished method is called with `execution_id` as its           sole positional argument

    """
    def __init__(self, *dependencies) -> None: ...

class WhenFinishedAll(_DependencyValidation, Condition):

    """WhenFinishedAll

    Parameters:

        *dependencies (Hashable):  zero or more nodes on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of all nodes specified in **dependencies** return `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          All(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no nodes are specified, the condition will default to checking all of scheduler's nodes.

        - This is a dynamic Condition: Each node is responsible for managing its finished status on its
          own, which can occur independently of the execution of other nodes.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

        - The is_finished method is called with `execution_id` as its           sole positional argument

    """
    def __init__(self, *dependencies) -> None: ...

class AtTrialStart(AtPass):

    """AtTrialStart

    Satisfied when:

        - at the beginning of an `TRIAL <TimeScale.TRIAL>`

    Notes:

        - identical to `AtPass(0) <AtPass>`
    """
    def __init__(self) -> None: ...

class AtTrialNStart(All):

    """AtTrialNStart

    Parameters:

        n(int): the `TRIAL <TimeScale.TRIAL>` on which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL <TimeScale.TRIAL>`\\ s
        (default: TimeScale.RUN)

    Satisfied when:

        - on `PASS` 0 of the specified `TRIAL <TimeScale.TRIAL>` counted using 'TimeScale`

    Notes:

        - identical to All(AtPass(0), AtTrial(n, time_scale))

    """
    def __init__(self, n, time_scale: graph_scheduler.time.TimeScale = ...) -> None: ...

class AtRunStart(AtTrial):

    """AtRunStart

    Satisfied when:

        - at the beginning of an `RUN`

    Notes:

        - identical to `AtTrial(0) <AtTrial>`
    """
    def __init__(self) -> None: ...

class AtRunNStart(All):

    """AtRunNStart

    Parameters:

        n(int): the `RUN` on which the Condition is satisfied

    Satisfied when:

        - on `TRIAL <TimeScale.TRIAL>` 0 of the specified `RUN` counted using 'TimeScale`

    Notes:

        - identical to `All(AtTrial(0), AtRun(n))`

    """
    def __init__(self, n) -> None: ...

class Threshold(_DependencyValidation, Condition):

    """Threshold

    Attributes:

        dependency
            the node on which the Condition depends

        parameter
            the name of the parameter of **dependency** whose value is
            to be compared to **threshold**

        threshold
            the fixed value compared to the value of the **parameter**

        comparator
            the string comparison operator determining the direction or
            type of comparison of the value of the **parameter**
            relative to **threshold**

        indices
            if specified, a series of indices that reach the desired
            number given an iterable value for **parameter**

        atol
            absolute tolerance for the comparison

        rtol
            relative tolerance (to **threshold**) for the comparison

        custom_parameter_getter
            if specified, a function that returns the value of
            **parameter** for **dependency**; to support class
            structures other than <**dependency**>.<**parameter**>
            without subclassing

        custom_parameter_validator
            if specified, a function that throws an exception if there
            is no **parameter** for **dependency**; to support class
            structures other than <**dependency**>.<**parameter**>
            without subclassing

    Satisfied when:

        The comparison between the value of the **parameter** and
        **threshold** using **comparator** is true. If **comparator** is
        an equality (==, !=), the comparison will be considered equal
        within tolerances **atol** and **rtol**.

    Notes:

        The comparison must be done with scalars. If the value of
        **parameter** contains more than one item, **indices** must be
        specified.
    """
    def __init__(self, dependency, parameter, threshold, comparator, indices: Incomplete | None = ..., atol: int = ..., rtol: int = ..., custom_parameter_getter: Incomplete | None = ..., custom_parameter_validator: Incomplete | None = ...) -> None: ...
    def get_parameter_value(self, execution_id: Incomplete | None = ...): ...
    def validate_parameter(self, dependency, parameter, custom_parameter_validator: Incomplete | None = ...): ...

class GraphStructureCondition(ConditionBase):

    """
    Abstract base class for `graph structure conditions
    <Condition_Graph_Structure_Intro>`

    Subclasses must implement:
        `_process`
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def modify_graph(self, graph: GraphDependencyDict) -> GraphDependencyDict:
        """
        Modifies **graph** based on the transformation specified by this
        condition

        Args:
            graph: a graph dependency dictionary

        Raises:
            ConditionError

        Returns:
            A copy of **graph** with modifications applied
        """
    def __init__(self, _owner: Hashable = ..., **kwargs) -> None: ...

class CustomGraphStructureCondition(GraphStructureCondition):

    """
    Applies a user-defined function to a graph

    Args:
        process_graph_function (Callable): a function taking an optional
            'self' argument (as the first argument, if present), and a
            graph dependency dictionary
        kwargs (**kwargs): optional arguments to be stored as attributes
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, process_graph_function: Callable, **kwargs) -> None: ...

class _GSCUsingNodes(GraphStructureCondition):

    """
    Attributes:
        nodes: the subject nodes
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *nodes: Hashable, **kwargs) -> None: ...

class _GSCSingleNode(_GSCUsingNodes):

    """
    Attributes:
        node: the subject node
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, node: Hashable, **kwargs) -> None: ...
    @property
    def node(self): ...

class _GSCWithOperations(_GSCUsingNodes):

    """
    Args:
        owner_senders: `Operation` that determines how the original
            senders of `owner <ConditionBase.owner>` (the Operation
            source) combine with the union of all original senders of
            all subject `nodes <_GSCUsingNodes.nodes>` (the
            Operation comparison) to produce the new set of senders of
            `owner <ConditionBase.owner>` after `modify_graph`
        owner_receivers: `Operation` that determines how the
            original receivers of `owner <ConditionBase.owner>` (the
            Operation source) combine with the union of all original
            receivers of all subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation comparison)
            to produce the new set of receivers of `owner
            <ConditionBase.owner>` after `modify_graph`
        subject_senders: `Operation` that determines how the
            original senders for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original senders of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of senders for
            the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        subject_receivers: `Operation` that determines how the
            original receivers for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original receivers of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of receivers
            for the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        reconnect_non_subject_receivers: If True, `modify_graph`
            will create an edge from all prior senders of `owner` to
            all receivers of `owner` that are not in `nodes`, if
            there is no longer a path from that sender to that
            receiver.
            Defaults to True.
        remove_new_self_referential_edges: If True, `modify_graph`
            will remove any newly-created edges from a node to
            itself.
            Defaults to True.
        prune_cycles: If True, `modify_graph` will attempt to prune
            any newly-created cycles, preferring to remove edges
            adjacent to `owner` that affect the placement of `owner`
            more than any subject `node <nodes>`.
            Defaults to True.
        ignore_conflicts: If True, when any two operations give
            different results for the new senders and receivers of a
            node in `modify_graph`, an error will not be raised.
            Defaults to False.

    Attributes:
        nodes: the subject nodes
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *nodes: Hashable, owner_senders: Operation | str = ..., owner_receivers: Operation | str = ..., subject_senders: SubjectOperation = ..., subject_receivers: SubjectOperation = ..., reconnect_non_subject_receivers: bool = ..., remove_new_self_referential_edges: bool = ..., prune_cycles: bool = ..., ignore_conflicts: bool = ..., **kwargs) -> None: ...

class _GSCReposition(_GSCUsingNodes):
    _already_valid_message: ClassVar[str] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *nodes: Hashable, **kwargs) -> None: ...

class BeforeNodes(_GSCReposition, _GSCWithOperations):

    """
    Adds a dependency from the owner to each of the specified nodes and
    optionally modifies the senders and receivers of all affected nodes

    Args:
        owner_senders: `Operation` that determines how the original
            senders of `owner <ConditionBase.owner>` (the Operation
            source) combine with the union of all original senders of
            all subject `nodes <_GSCUsingNodes.nodes>` (the
            Operation comparison) to produce the new set of senders of
            `owner <ConditionBase.owner>` after `modify_graph`
        owner_receivers: `Operation` that determines how the
            original receivers of `owner <ConditionBase.owner>` (the
            Operation source) combine with the union of all original
            receivers of all subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation comparison)
            to produce the new set of receivers of `owner
            <ConditionBase.owner>` after `modify_graph`
        subject_senders: `Operation` that determines how the
            original senders for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original senders of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of senders for
            the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        subject_receivers: `Operation` that determines how the
            original receivers for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original receivers of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of receivers
            for the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        reconnect_non_subject_receivers: If True, `modify_graph`
            will create an edge from all prior senders of `owner` to
            all receivers of `owner` that are not in `nodes`, if
            there is no longer a path from that sender to that
            receiver.
            Defaults to True.
        remove_new_self_referential_edges: If True, `modify_graph`
            will remove any newly-created edges from a node to
            itself.
            Defaults to True.
        prune_cycles: If True, `modify_graph` will attempt to prune
            any newly-created cycles, preferring to remove edges
            adjacent to `owner` that affect the placement of `owner`
            more than any subject `node <nodes>`.
            Defaults to True.
        ignore_conflicts: If True, when any two operations give
            different results for the new senders and receivers of a
            node in `modify_graph`, an error will not be raised.
            Defaults to False.

    Attributes:
        nodes: the subject nodes
    """
    _already_valid_message: ClassVar[str] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *nodes, owner_senders: Operation | str = ..., owner_receivers: Operation | str = ..., subject_senders: SubjectOperation = ..., subject_receivers: SubjectOperation = ..., reconnect_non_subject_receivers: bool = ..., remove_new_self_referential_edges: bool = ..., prune_cycles: bool = ..., ignore_conflicts: bool = ...) -> None: ...

class BeforeNode(BeforeNodes, _GSCSingleNode):

    """
    Adds a dependency from the owner to the specified node and
    optionally modifies the senders and receivers of both

    Args:
        owner_senders: `Operation` that determines how the original
            senders of `owner <ConditionBase.owner>` (the Operation
            source) combine with the union of all original senders of
            all subject `nodes <_GSCUsingNodes.nodes>` (the
            Operation comparison) to produce the new set of senders of
            `owner <ConditionBase.owner>` after `modify_graph`
        owner_receivers: `Operation` that determines how the
            original receivers of `owner <ConditionBase.owner>` (the
            Operation source) combine with the union of all original
            receivers of all subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation comparison)
            to produce the new set of receivers of `owner
            <ConditionBase.owner>` after `modify_graph`
        subject_senders: `Operation` that determines how the
            original senders for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original senders of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of senders for
            the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        subject_receivers: `Operation` that determines how the
            original receivers for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original receivers of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of receivers
            for the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        reconnect_non_subject_receivers: If True, `modify_graph`
            will create an edge from all prior senders of `owner` to
            all receivers of `owner` that are not in `nodes`, if
            there is no longer a path from that sender to that
            receiver.
            Defaults to True.
        remove_new_self_referential_edges: If True, `modify_graph`
            will remove any newly-created edges from a node to
            itself.
            Defaults to True.
        prune_cycles: If True, `modify_graph` will attempt to prune
            any newly-created cycles, preferring to remove edges
            adjacent to `owner` that affect the placement of `owner`
            more than any subject `node <nodes>`.
            Defaults to True.
        ignore_conflicts: If True, when any two operations give
            different results for the new senders and receivers of a
            node in `modify_graph`, an error will not be raised.
            Defaults to False.

    Attributes:
        nodes: the subject nodes

    Attributes:
        node: the subject node
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *nodes, owner_senders: Operation | str = ..., owner_receivers: Operation | str = ..., subject_senders: SubjectOperation = ..., subject_receivers: SubjectOperation = ..., reconnect_non_subject_receivers: bool = ..., remove_new_self_referential_edges: bool = ..., prune_cycles: bool = ..., ignore_conflicts: bool = ...) -> None: ...

class WithNode(_GSCReposition, _GSCWithOperations, _GSCSingleNode):

    """
    Adds a dependency from each of the senders of both the owner and the
    specified node to both the owner and the specified node, and
    optionally modifies the receivers of both

    Args:
        owner_senders: `Operation` that determines how the original
            senders of `owner <ConditionBase.owner>` (the Operation
            source) combine with the union of all original senders of
            all subject `nodes <_GSCUsingNodes.nodes>` (the
            Operation comparison) to produce the new set of senders of
            `owner <ConditionBase.owner>` after `modify_graph`
        owner_receivers: `Operation` that determines how the
            original receivers of `owner <ConditionBase.owner>` (the
            Operation source) combine with the union of all original
            receivers of all subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation comparison)
            to produce the new set of receivers of `owner
            <ConditionBase.owner>` after `modify_graph`
        subject_senders: `Operation` that determines how the
            original senders for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original senders of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of senders for
            the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        subject_receivers: `Operation` that determines how the
            original receivers for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original receivers of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of receivers
            for the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        reconnect_non_subject_receivers: If True, `modify_graph`
            will create an edge from all prior senders of `owner` to
            all receivers of `owner` that are not in `nodes`, if
            there is no longer a path from that sender to that
            receiver.
            Defaults to True.
        remove_new_self_referential_edges: If True, `modify_graph`
            will remove any newly-created edges from a node to
            itself.
            Defaults to True.
        prune_cycles: If True, `modify_graph` will attempt to prune
            any newly-created cycles, preferring to remove edges
            adjacent to `owner` that affect the placement of `owner`
            more than any subject `node <nodes>`.
            Defaults to True.
        ignore_conflicts: If True, when any two operations give
            different results for the new senders and receivers of a
            node in `modify_graph`, an error will not be raised.
            Defaults to False.

    Attributes:
        nodes: the subject nodes

    Attributes:
        node: the subject node
    """
    _already_valid_message: ClassVar[str] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, node, owner_receivers: Operation | str = ..., subject_receivers: SubjectOperation = ..., reconnect_non_subject_receivers: bool = ..., remove_new_self_referential_edges: bool = ..., prune_cycles: bool = ..., ignore_conflicts: bool = ...) -> None: ...

class AfterNodes(_GSCReposition, _GSCWithOperations):

    """
    Adds a dependency from each of the specified nodes to the owner
    and optionally modifies the senders and receivers of all
    affected nodes

    Args:
        owner_senders: `Operation` that determines how the original
            senders of `owner <ConditionBase.owner>` (the Operation
            source) combine with the union of all original senders of
            all subject `nodes <_GSCUsingNodes.nodes>` (the
            Operation comparison) to produce the new set of senders of
            `owner <ConditionBase.owner>` after `modify_graph`
        owner_receivers: `Operation` that determines how the
            original receivers of `owner <ConditionBase.owner>` (the
            Operation source) combine with the union of all original
            receivers of all subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation comparison)
            to produce the new set of receivers of `owner
            <ConditionBase.owner>` after `modify_graph`
        subject_senders: `Operation` that determines how the
            original senders for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original senders of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of senders for
            the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        subject_receivers: `Operation` that determines how the
            original receivers for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original receivers of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of receivers
            for the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        reconnect_non_subject_receivers: If True, `modify_graph`
            will create an edge from all prior senders of `owner` to
            all receivers of `owner` that are not in `nodes`, if
            there is no longer a path from that sender to that
            receiver.
            Defaults to True.
        remove_new_self_referential_edges: If True, `modify_graph`
            will remove any newly-created edges from a node to
            itself.
            Defaults to True.
        prune_cycles: If True, `modify_graph` will attempt to prune
            any newly-created cycles, preferring to remove edges
            adjacent to `owner` that affect the placement of `owner`
            more than any subject `node <nodes>`.
            Defaults to True.
        ignore_conflicts: If True, when any two operations give
            different results for the new senders and receivers of a
            node in `modify_graph`, an error will not be raised.
            Defaults to False.

    Attributes:
        nodes: the subject nodes
    """
    _already_valid_message: ClassVar[str] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *nodes, owner_senders: Operation | str = ..., owner_receivers: Operation | str = ..., subject_senders: SubjectOperation = ..., subject_receivers: SubjectOperation = ..., reconnect_non_subject_receivers: bool = ..., remove_new_self_referential_edges: bool = ..., prune_cycles: bool = ..., ignore_conflicts: bool = ...) -> None: ...

class AfterNode(AfterNodes, _GSCSingleNode):

    """
    Adds a dependency from the specified node to the owner and
    optionally modifies the senders and receivers of both

    Args:
        owner_senders: `Operation` that determines how the original
            senders of `owner <ConditionBase.owner>` (the Operation
            source) combine with the union of all original senders of
            all subject `nodes <_GSCUsingNodes.nodes>` (the
            Operation comparison) to produce the new set of senders of
            `owner <ConditionBase.owner>` after `modify_graph`
        owner_receivers: `Operation` that determines how the
            original receivers of `owner <ConditionBase.owner>` (the
            Operation source) combine with the union of all original
            receivers of all subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation comparison)
            to produce the new set of receivers of `owner
            <ConditionBase.owner>` after `modify_graph`
        subject_senders: `Operation` that determines how the
            original senders for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original senders of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of senders for
            the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        subject_receivers: `Operation` that determines how the
            original receivers for each of the subject `nodes
            <_GSCUsingNodes.nodes>` (the Operation source) combine with
            the original receivers of `owner <ConditionBase.owner>` (the
            Operation comparison) to produce the new set of receivers
            for the subject `nodes <_GSCUsingNodes.nodes>` after
            `modify_graph`. Operations are applied individually to each
            subject node, and this argument may also be specified as a
            dictionary mapping nodes to separate operations.
        reconnect_non_subject_receivers: If True, `modify_graph`
            will create an edge from all prior senders of `owner` to
            all receivers of `owner` that are not in `nodes`, if
            there is no longer a path from that sender to that
            receiver.
            Defaults to True.
        remove_new_self_referential_edges: If True, `modify_graph`
            will remove any newly-created edges from a node to
            itself.
            Defaults to True.
        prune_cycles: If True, `modify_graph` will attempt to prune
            any newly-created cycles, preferring to remove edges
            adjacent to `owner` that affect the placement of `owner`
            more than any subject `node <nodes>`.
            Defaults to True.
        ignore_conflicts: If True, when any two operations give
            different results for the new senders and receivers of a
            node in `modify_graph`, an error will not be raised.
            Defaults to False.

    Attributes:
        nodes: the subject nodes

    Attributes:
        node: the subject node
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *nodes, owner_senders: Operation | str = ..., owner_receivers: Operation | str = ..., subject_senders: SubjectOperation = ..., subject_receivers: SubjectOperation = ..., reconnect_non_subject_receivers: bool = ..., remove_new_self_referential_edges: bool = ..., prune_cycles: bool = ..., ignore_conflicts: bool = ...) -> None: ...

class AddEdgeTo(_GSCSingleNode):

    """
    Adds an edge from `AddEdgeTo.owner <ConditionBase.owner>` to
    `AddEdgeTo.node`

    Attributes:
        node: the subject node
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, node: Hashable) -> None: ...

class RemoveEdgeFrom(_GSCSingleNode):

    """
    Removes an edge from `RemoveEdgeFrom.node` to `RemoveEdgeFrom.owner
    <ConditionBase.owner>`

    Attributes:
        node: the subject node
    """
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, node: Hashable) -> None: ...
