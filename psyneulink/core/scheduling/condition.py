# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Condition **************************************************************

"""

.. _Condition_Overview

Overview
--------

`Conditions <Condition>` are used to specify when `Components <Component>` are allowed to execute.  Conditions
can be used to specify a variety of required conditions for execution, including the state of the Component
itself (e.g., how many times it has already executed, or the value of one of its attributes), the state of the
Composition (e.g., how many `TIME_STEP` s have occurred in the current `TRIAL <TimeScale.TRIAL>`), or the state of other
Components in a Composition (e.g., whether or how many times they have executed). PsyNeuLink provides a number of
`pre-specified Conditions <Condition_Pre_Specified>` that can be parametrized (e.g., how many times a Component should
be executed). `Custom conditions <Condition_Custom>` can also be created, by assigning a function to a Condition that
can reference any Component or its attributes in PsyNeuLink, thus providing considerable flexibility for scheduling.

.. note::
    Any Component that is part of a collection `specified to a Scheduler for execution <Scheduler_Creation>` can be
    associated with a Condition.  Most commonly, these are `Mechanisms <Mechanism>`.  However, in some circumstances
    `Projections <Projection>` can be included in the specification to a Scheduler (e.g., for
    `learning <Process_Learning>`) in which case these can also be assigned Conditions.



.. _Condition_Creation:

Creating Conditions
-------------------

.. _Condition_Pre_Specified:

*Pre-specified Conditions*
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Pre-specified Conditions <Condition_Pre-Specified_List>` can be instantiated and added to a `Scheduler` at any time,
and take effect immediately for the execution of that Scheduler. Most pre-specified Conditions have one or more
arguments that must be specified to achieve the desired behavior. Many Conditions are also associated with an
`owner <Condition.owner>` attribute (a `Component` to which the Condition belongs). `Scheduler`\\ s maintain the data
used to test for satisfaction of Condition, independent in different `execution context`\\ s. The Scheduler is generally
responsible for ensuring that Conditions have access to the necessary data.
When pre-specified Conditions are instantiated within a call to the `add` method of a `Scheduler` or `ConditionSet`,
the Condition's `owner <Condition.owner>` is determined through
context and assigned automatically, as in the following example::

    my_scheduler.add_condition(A, EveryNPasses(1))
    my_scheduler.add_condition(B, EveryNCalls(A, 2))
    my_scheduler.add_condition(C, EveryNCalls(B, 2))

Here, `EveryNCalls(A, 2)` for example, is assigned the `owner` `B`.

.. _Condition_Custom:

*Custom Conditions*
~~~~~~~~~~~~~~~~~~~

Custom Conditions can be created by calling the constructor for the base class (`Condition()`) or one of the
`generic classes <Conditions_Generic>`,  and assigning a function to the **func** argument and any arguments it
requires to the **args** and/or **kwargs** arguments (for formal or keyword arguments, respectively). The function
is called with **args** and **kwargs** by the `Scheduler` on each `PASS` through its `consideration_queue`, and the result is
used to determine whether the associated Component is allowed to execute on that `PASS`. Custom Conditions allow
arbitrary schedules to be created, in which the execution of each Component can depend on one or more attributes of
any other Components in the Composition.

.. _Condition_Recurrent_Example:

For example, the following script fragment creates a custom Condition in which `mech_A` is scheduled to wait to
execute until a `RecurrentTransferMechanism` `mech_B` has "converged" (that is, settled to the point that none of
its elements has changed in value more than a specified amount since the previous `TIME_STEP`)::

    def converge(mech, thresh):
        for val in mech.delta:
            if abs(val) >= thresh:
                return False
        return True
    epsilon = 0.01
    my_scheduler.add_condition(mech_A, NWhen(Condition(converge, mech_B, epsilon), 1))

In the example, a function `converge` is defined that references the `delta <TransferMechanism.delta>` attribute of
a `TransferMechanism` (which reports the change in its `value <Mechanism_Base.value>`). The function is assigned to
the standard `Condition()` with `mech_A` and `epsilon` as its arguments, and `composite Condition <Conditions_Composite>`
`NWhen` (which is satisfied the first N times after its condition becomes true),  The Condition is assigned to `mech_B`,
thus scheduling it to execute one time when all of the elements of `mech_A` have changed by less than `epsilon`.

.. _Condition_Structure:

Structure
---------

The `Scheduler` associates every Component with a Condition.  If a Component has not been explicitly assigned a
Condition, it is assigned a Condition that causes it to be executed whenever it is `under consideration <Scheduler_Algorithm>`
and all its structural parents have been executed at least once since the Component's last execution.
Condition subclasses (`listed below <Condition_Pre-Specified_List>`)
provide a standard set of Conditions that can be implemented simply by specifying their parameter(s). There are
six types:

  * `Generic <Conditions_Generic>` - satisfied when a `user-specified function and set of arguments <Condition_Custom>`
    evaluates to `True`;
  * `Static <Conditions_Static>` - satisfied either always or never;
  * `Composite <Conditions_Composite>` - satisfied based on one or more other Conditions;
  * `Time-based <Conditions_Time_Based>` - satisfied based on the current count of units of time at a specified
    `TimeScale`;
  * `Component-based <Conditions_Component_Based>` - based on the execution or state of other Components.
  * `Convenience <Conditions_Convenience>` - based on other Conditions, condensed for convenience

.. _Condition_Pre-Specified_List:

*List of Pre-specified Conditions*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    The optional `TimeScale` argument in many `Conditions <Condition>` specifies the unit of time over which the
    Condition operates;  the default value is `TRIAL <TimeScale.TRIAL>` for all Conditions except those with "Trial"
    in their name, for which it is `RUN`.


.. _Conditions_Generic:

**Generic Conditions** (used to construct `custom Conditions <Condition_Custom>`):

    * `While` (func, *args, **kwargs)
      satisfied whenever the specified function (or callable) called with args and/or kwargs evaluates to `True`.
      Equivalent to `Condition(func, *args, **kwargs)`

    * `WhileNot` (func, *args, **kwargs)
      satisfied whenever the specified function (or callable) called with args and/or kwargs evaluates to `False`.
      Equivalent to `Not(Condition(func, *args, **kwargs))`

.. _Conditions_Static:

**Static Conditions** (independent of other Conditions, Components or time):

    * `Always`
      always satisfied.

    * `Never`
      never satisfied.


.. _Conditions_Composite:

**Composite Conditions** (based on one or more other Conditions):

    * `All` (*Conditions)
      satisfied whenever all of the specified Conditions are satisfied.

    * `Any` (*Conditions)
      satisfied whenever any of the specified Conditions are satisfied.

    * `Not` (Condition)
      satisfied whenever the specified Condition is not satisfied.

    * `NWhen` (Condition, int)
      satisfied the first specified number of times the specified Condition is satisfied.


.. _Conditions_Time_Based:

**Time-Based Conditions** (based on the count of units of time at a specified `TimeScale`):


    * `BeforeTimeStep` (int[, TimeScale])
      satisfied any time before the specified `TIME_STEP` occurs.

    * `AtTimeStep` (int[, TimeScale])
      satisfied only during the specified `TIME_STEP`.

    * `AfterTimeStep` (int[, TimeScale])
      satisfied any time after the specified `TIME_STEP` has occurred.

    * `AfterNTimeSteps` (int[, TimeScale])
      satisfied when or any time after the specified number of `TIME_STEP`\\ s has occurred.

    * `BeforePass` (int[, TimeScale])
      satisfied any time before the specified `PASS` occurs.

    * `AtPass` (int[, TimeScale])
      satisfied only during the specified `PASS`.

    * `AfterPass` (int[, TimeScale])
      satisfied any time after the specified `PASS` has occurred.

    * `AfterNPasses` (int[, TimeScale])
      satisfied when or any time after the specified number of `PASS`\\ es has occurred.

    * `EveryNPasses` (int[, TimeScale])
      satisfied every time the specified number of `PASS`\\ es occurs.

    * `BeforeTrial` (int[, TimeScale])
      satisfied any time before the specified `TRIAL <TimeScale.TRIAL>` occurs.

    * `AtTrial` (int[, TimeScale])
      satisfied any time during the specified `TRIAL <TimeScale.TRIAL>`.

    * `AfterTrial` (int[, TimeScale])
      satisfied any time after the specified `TRIAL <TimeScale.TRIAL>` occurs.

    * `AfterNTrials` (int[, TimeScale])
      satisfied any time after the specified number of `TRIAL <TimeScale.TRIAL>`\\ s has occurred.

    * `AtRun` (int)
      satisfied any time during the specified `RUN`.

    * `AfterRun` (int)
      satisfied any time after the specified `RUN` occurs.

    * `AfterNRuns` (int)
      satisfied any time after the specified number of `RUN`\\ s has occurred.

.. _Conditions_Component_Based:

**Component-Based Conditions** (based on the execution or state of other Components):


    * `BeforeNCalls` (Component, int[, TimeScale])
      satisfied any time before the specified Component has executed the specified number of times.

    * `AtNCalls` (Component, int[, TimeScale])
      satisfied when the specified Component has executed the specified number of times.

    * `AfterCall` (Component, int[, TimeScale])
      satisfied any time after the Component has executed the specified number of times.

    * `AfterNCalls` (Component, int[, TimeScale])
      satisfied when or any time after the Component has executed the specified number of times.

    * `AfterNCallsCombined` (*Components, int[, TimeScale])
      satisfied when or any time after the specified Components have executed the specified number
      of times among themselves, in total.

    * `EveryNCalls` (Component, int[, TimeScale])
      satisfied when the specified Component has executed the specified number of times since the
      last time `owner` has run.

    * `JustRan` (Component)
      satisfied if the specified Component was assigned to run in the previous `TIME_STEP`.

    * `AllHaveRun` (*Components)
      satisfied when all of the specified Components have executed at least once.

    * `WhenFinished` (Component)
      satisfied when the specified Component has set its `is_finished` attribute to `True`.

    * `WhenFinishedAny` (*Components)
      satisfied when any of the specified Components has set their `is_finished` attribute to `True`.

    * `WhenFinishedAll` (*Components)
      satisfied when all of the specified Components have set their `is_finished` attributes to `True`.

.. _Conditions_Convenience:

**Convenience Conditions** (based on other Conditions, condensed for convenience)


    * `AtTrialStart`
      satisfied at the beginning of a `TRIAL <TimeScale.TRIAL>` (`AtPass(0) <AtPass>`)

    * `AtTrialNStart`
      satisfied on `PASS` 0 of the specified `TRIAL <TimeScale.TRIAL>` counted using 'TimeScale`

    * `AtRunStart`
      satisfied at the beginning of a `RUN`

    * `AtRunNStart`
      satisfied on `TRIAL <TimeScale.TRIAL>` 0 of the specified `RUN` counted using 'TimeScale`


.. Condition_Execution:

Execution
---------

When the `Scheduler` `runs <Schedule_Execution>`, it makes a sequential `PASS` through its `consideration_queue`,
evaluating each `consideration_set <consideration_set>` in the queue to determine which Components should be assigned
to execute. It evaluates the Components in each set by calling the `is_satisfied` method of the Condition associated
with each of those Components.  If it returns `True`, then the Component is assigned to the execution set for the
`TIME_STEP` of execution generated by that `PASS`.  Otherwise, the Component is not executed.

.. _Condition_Class_Reference:

Class Reference
---------------

"""

import collections
import dill
import inspect
import logging
import warnings

from psyneulink.core.globals.json import JSONDumpable
from psyneulink.core.globals.keywords import MODEL_SPEC_ID_TYPE
from psyneulink.core.globals.parameters import parse_context
from psyneulink.core.globals.utilities import call_with_pruned_args
from psyneulink.core.scheduling.time import TimeScale

__all__ = [
    'AfterCall', 'AfterNCalls', 'AfterNCallsCombined', 'AfterNPasses', 'AfterNTimeSteps', 'AfterNTrials', 'AfterPass',
    'AtRun', 'AfterRun', 'AfterNRuns', 'AfterTimeStep', 'AfterTrial', 'All', 'AllHaveRun', 'Always', 'Any',
    'AtNCalls','AtPass', 'AtRunStart', 'AtRunNStart', 'AtTimeStep', 'AtTrial',
    'AtTrialStart', 'AtTrialNStart', 'BeforeNCalls', 'BeforePass', 'BeforeTimeStep', 'BeforeTrial',
    'Condition','ConditionError', 'ConditionSet', 'EveryNCalls', 'EveryNPasses',
    'JustRan', 'Never', 'Not', 'NWhen', 'WhenFinished', 'WhenFinishedAll', 'WhenFinishedAny', 'While', 'WhileNot'
]

logger = logging.getLogger(__name__)


class ConditionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ConditionSet(object):
    """Used in conjunction with a `Scheduler` to store the `Conditions <Condition>` associated with a `Component`.

    Arguments
    ---------

    conditions : Dict[`Component <Component>`: `Condition`]
        specifies an iterable collection of `Components <Component>` and the `Conditions <Condition>` associated
        with each.

    Attributes
    ----------

    conditions : Dict[`Component <Component>`: `Condition`]
        the key of each entry is a `Component <Component>`, and its value is the `Condition <Condition>` associated
        with that Component.  Conditions can be added to the
        ConditionSet using the ConditionSet's `add_condition` method.

    """
    def __init__(self, conditions=None):
        self.conditions = {}

        if conditions is not None:
            self.add_condition_set(conditions)

    def __contains__(self, item):
        return item in self.conditions

    def __repr__(self):
        condition_str = '\n\t'.join([f'{owner}: {condition}' for owner, condition in self.conditions.items()])
        return '{0}({1}{2}{3})'.format(
            self.__class__.__name__,
            '\n\t' if len(condition_str) > 0 else '',
            condition_str,
            '\n' if len(condition_str) > 0 else ''
        )

    def __iter__(self):
        return iter(self.conditions)

    def __getitem__(self, key):
        return self.conditions[key]

    def __setitem__(self, key, value):
        self.conditions[key] = value

    def add_condition(self, owner, condition):
        """
        Adds a `Condition` to the ConditionSet. If **owner** already has a Condition, it is overwritten
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
        condition.owner = owner
        self.conditions[owner] = condition

    def add_condition_set(self, conditions):
        """
        Adds a set of `Conditions <Condition>` (in the form of a dict or another ConditionSet) to the ConditionSet.
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
        for owner in conditions:
            self.add_condition(owner, conditions[owner])


class Condition(JSONDumpable):
    """
    Used in conjunction with a `Scheduler` to specify the condition under which a `Component` should be
    allowed to execute.

    Arguments
    ---------

    func : callable
        specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

    args : *args
        specifies formal arguments to pass to `func` when the Condition is evaluated.

    kwargs : **kwargs
        specifies keyword arguments to pass to `func` when the Condition is evaluated.

    Attributes
    ----------

    owner (Component):
        the `Component` with which the Condition is associated, and the execution of which it determines.

    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._owner = None

    def __str__(self):
        return '{0}({1}{2})'.format(
            self.__class__.__name__,
            ', '.join([str(arg) for arg in self.args]) if len(self.args) > 0 else '',
            ', {0}'.format(self.kwargs) if len(self.kwargs) > 0 else ''
        )

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        logger.debug('Condition ({0}) setting owner to {1}'.format(type(self).__name__, value))
        self._owner = value

    def is_satisfied(self, *args, context=None, execution_id=None, **kwargs):
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
        if execution_id is None:
            try:
                execution_id = parse_context(context).execution_id
            except AttributeError:
                pass

        # update so that kwargs can override self.kwargs
        kwargs_to_pass = self.kwargs.copy()
        kwargs_to_pass.update(kwargs)

        return call_with_pruned_args(
            self.func,
            *self.args,
            *args,
            context=context,
            execution_id=execution_id,
            **kwargs_to_pass
        )

    @property
    def _dict_summary(self):
        from psyneulink.core.components.component import Component

        if type(self) is Condition:
            try:
                func_val = inspect.getsource(self.func)
            except OSError:
                func_val = dill.dumps(self.func)
        else:
            func_val = None

        args_list = []
        for a in self.args:
            if isinstance(a, Component):
                a = a.name
            elif isinstance(a, Condition):
                a = a._dict_summary
            args_list.append(a)

        return {
            MODEL_SPEC_ID_TYPE: self.__class__.__name__,
            'function': func_val,
            'args': args_list,
            'kwargs': self.kwargs,
        }


class _DependencyValidation:
    @Condition.owner.setter
    def owner(self, value):
        # "dependency" or "dependencies" is always the first positional argument
        if not isinstance(self.args[0], collections.abc.Iterable):
            dependencies = [self.args[0]]
        else:
            dependencies = self.args[0]

        if value in dependencies:
            warnings.warn(
                f'{self} is dependent on {value}, but you are assigning {value} as its owner.'
                ' This may result in infinite loops or unknown behavior.',
                stacklevel=5
            )

        self._owner = value


#########################################################################################################
# Included Conditions
#########################################################################################################

######################################################################
# Generic Conditions
#   - convenience wrappers
######################################################################


While = Condition


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
    def __init__(self, func, *args, **kwargs):
        def inner_func(*args, **kwargs):
            return not call_with_pruned_args(func, *args, **kwargs)
        super().__init__(inner_func, *args, **kwargs)

######################################################################
# Static Conditions
#   - independent of components and time
######################################################################


class Always(Condition):
    """Always

    Parameters:

        none

    Satisfied when:

        - always satisfied.

    """
    def __init__(self):
        super().__init__(lambda: True)


class Never(Condition):
    """Never

    Parameters:

        none

    Satisfied when:

        - never satisfied.
    """
    def __init__(self):
        super().__init__(lambda: False)

######################################################################
# Composite Conditions
#   - based on other Conditions
######################################################################

# TODO: create this class to subclass All and Any from
# class CompositeCondition(Condition):
    # def


class All(Condition):
    """All

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - all of the Conditions in args are satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list]

          unpack the list to supply its members as args::

           composite_condition = All(*conditions)

    """
    def __init__(self, *args):
        super().__init__(self.satis, *args)

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.args:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, *conds, **kwargs):
        for cond in conds:
            if not cond.is_satisfied(**kwargs):
                return False
        return True


class Any(Condition):
    """Any

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - one or more of the Conditions in **args** is satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list]

          unpack the list to supply its members as args::

           composite_condition = All(*conditions)

    """
    def __init__(self, *args):
        super().__init__(self.satis, *args)

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.args:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, *conds, **kwargs):
        for cond in conds:
            if cond.is_satisfied(**kwargs):
                return True
        return False


class Not(Condition):
    """Not

    Parameters:

        condition(Condition): a `Condition`

    Satisfied when:

        - **condition** is not satisfied.

    """
    def __init__(self, condition):
        self.condition = condition

        def inner_func(condition, *args, **kwargs):
            return not condition.is_satisfied(*args, **kwargs)
        super().__init__(inner_func, condition)

    @Condition.owner.setter
    def owner(self, value):
        self.condition.owner = value


class NWhen(Condition):
    """NWhen

    Parameters:

        condition(Condition): a `Condition`

        n(int): the maximum number of times this condition will be satisfied

    Satisfied when:

        - the first **n** times **condition** is satisfied upon evaluation

    """
    def __init__(self, condition, n=1):
        self.satisfactions = {}
        self.condition = condition

        super().__init__(self.satis, condition, n)

    @Condition.owner.setter
    def owner(self, value):
        self.condition.owner = value

    def satis(self, condition, n, *args, scheduler=None, execution_id=None, **kwargs):
        if execution_id is None:
            if scheduler is not None:
                execution_id = scheduler.default_execution_id
        # if no execution_id or scheduler is provided technically this will still work
        # indexed on None, but that's a bit weird honestly

        if execution_id not in self.satisfactions:
            self.satisfactions[execution_id] = 0

        if self.satisfactions[execution_id] < n:
            if call_with_pruned_args(condition.is_satisfied, *args, scheduler=scheduler, execution_id=execution_id, **kwargs):
                self.satisfactions[execution_id] += 1
                return True
        return False


######################################################################
# Time-based Conditions
#   - satisfied based only on TimeScales
######################################################################

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
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TIME_STEP, time_scale) < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


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
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TIME_STEP, time_scale) == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


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
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TIME_STEP, time_scale) > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AfterNTimeSteps(Condition):
    """AfterNTimeSteps

    Parameters:

        n(int): the number of `TIME_STEP`\\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TIME_STEP`\\ s (default: TimeScale.TRIAL)


    Satisfied when:

        - at least n `TIME_STEP`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TIME_STEP, time_scale) >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


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
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


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
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


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
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AfterNPasses(Condition):
    """AfterNPasses

    Parameters:

        n(int): the number of `PASS`\\ es after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.TRIAL)


    Satisfied when:

        - at least n `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


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
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) % n == 0
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


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
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TRIAL, time_scale) < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


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
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TRIAL, time_scale) == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


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
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TRIAL, time_scale) > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterNTrials(Condition):
    """AfterNTrials

    Parameters:

        n(int): the number of `TRIAL <TimeScale.TRIAL>`\\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL <TimeScale.TRIAL>`\\ s
        (default: TimeScale.RUN)

    Satisfied when:

        - at least n `TRIAL <TimeScale.TRIAL>`\\ s have occured  within one unit of time at the `TimeScale`
          specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.TRIAL, time_scale) >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AtRun(Condition):
    """AtRun

    Parameters:

        n(int): the `RUN` at which the Condition is satisfied

    Satisfied when:

        - exactly n `RUN`\\ s have occurred.

    """
    def __init__(self, n):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).time.run == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterRun(Condition):
    """AfterRun

    Parameters:

        n(int): the `RUN` after which the Condition is satisfied

    Satisfied when:

        - at least n+1 `RUN`\\ s have occurred.

    """
    def __init__(self, n):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).time.run > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterNRuns(Condition):
    """AfterNTrials

    Parameters:

        n(int): the number of `RUN`\\ s after which the Condition is satisfied

    Satisfied when:

        - at least n `RUN`\\ s have occured.

    """

    def __init__(self, n):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).time.run >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)



######################################################################
# Component-based Conditions
#   - satisfied based on executions or state of Components
######################################################################


class BeforeNCalls(_DependencyValidation, Condition):
    """BeforeNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at most n-1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)

# NOTE:
# The behavior of AtNCalls is not desired (i.e. depending on the order mechanisms are checked, B running AtNCalls(A, x))
# may run on both the xth and x+1st call of A; if A and B are not parent-child
# A fix could invalidate key assumptions and affect many other conditions
# Since this condition is unlikely to be used, it's best to leave it for now


class AtNCalls(_DependencyValidation, Condition):
    """AtNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed exactly n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class AfterCall(_DependencyValidation, Condition):
    """AfterCall

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at least n+1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class AfterNCalls(_DependencyValidation, Condition):
    """AfterNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at least n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class AfterNCallsCombined(_DependencyValidation, Condition):
    """AfterNCallsCombined

    Parameters:

        *components(Components):  one or more Components on which the Condition depends

        n(int): the number of combined executions of all Components specified in **components** after which the
        Condition is satisfied (default: None)

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component**
        (default: TimeScale.TRIAL)


    Satisfied when:

        - there have been at least n+1 executions among all of the Components specified in **components**
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, n=None, time_scale=TimeScale.TRIAL):
        logger.debug('{0} args: deps {1}, n {2}, ts {3}'.format(type(self).__name__, dependencies, n, time_scale))

        def func(*dependencies, n=None, scheduler=None, execution_id=None):
            if n is None:
                raise ConditionError(f'{type(self).__name__}: required keyword argument n is None.')
            count_sum = 0
            for d in dependencies:
                try:
                    count_sum += scheduler.counts_total[execution_id][time_scale][d]
                    logger.debug('{0} has reached {1} num_calls in {2}'.
                                 format(d, scheduler.counts_total[execution_id][time_scale][d], time_scale.name))
                except AttributeError as e:
                    raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

            return count_sum >= n
        super().__init__(func, *dependencies, n=n)


class EveryNCalls(_DependencyValidation, Condition):
    """EveryNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the frequency of executions of **component** at which the Condition is satisfied


    Satisfied when:

        - the Component specified in **component** has executed at least n times since the last time the
          Condition's owner executed.

        COMMENT:
            JDC: IS THE FOLLOWING TRUE OF ALL OF THE ABOVE AS WELL??
            K: No, EveryNCalls is tricky in how it needs to be implemented, because it's in a sense
                tracking the relative frequency of calls between two objects. So the idea is that the scheduler
                tracks how many executions of a component are "useable" by other components for EveryNCalls conditions.
                So, suppose you had something like add_condition(B, All(AfterNCalls(A, 10), EveryNCalls(A, 2))). You
                would want the AAB pattern to start happening after A has run 10 times. Useable counts allows B to see
                whether A has run enough times for it to run, and then B spends its "useable executions" of A. Then,
                A must run two more times for B to run again. If you didn't reset the counts of A useable by B
                to 0 (question below) when B runs, then in the
                above case B would continue to run every pass for the next 4 passes, because it would see an additional
                8 executions of A it could spend to execute.
            JDC: IS THIS A FORM OF MODULO?  IF SO, WOULD IT BE EASIER TO EXPLAIN IN THAT FORM?
        COMMENT

    Notes:

        - scheduler's count of each other Component that is "useable" by the Component is reset to 0 when the
          Component runs

    """
    def __init__(self, dependency, n):
        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_useable[execution_id][dependency][self.owner]
                logger.debug('{0} has reached {1} num_calls'.format(dependency, num_calls))
                return num_calls >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class JustRan(_DependencyValidation, Condition):
    """JustRan

    Parameters:

        component(Component):  the Component on which the Condition depends

    Satisfied when:

        - the Component specified in **component** executed in the previous `TIME_STEP`.

    Notes:

        - This Condition can transcend divisions between `TimeScales <TimeScale>`.
          For example, if A runs in the final `TIME_STEP` of a `TRIAL <TimeScale.TRIAL>`,
          JustRan(A) is satisfied at the beginning of the next `TRIAL <TimeScale.TRIAL>`.

    """
    def __init__(self, dependency):
        def func(dependency, scheduler=None, execution_id=None):
            logger.debug(f'checking if {dependency} in previous execution step set')
            try:
                return dependency in scheduler.execution_list[execution_id][-1]
            except TypeError:
                return dependency == scheduler.execution_list[execution_id][-1]
        super().__init__(func, dependency)


class AllHaveRun(_DependencyValidation, Condition):
    """AllHaveRun

    Parameters:

        *components(Components):  an iterable of Components on which the Condition depends

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component**
        (default: TimeScale.TRIAL)

    Satisfied when:

        - all of the Components specified in **components** have executed at least once
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, time_scale=TimeScale.TRIAL):
        def func(*dependencies, scheduler=None, execution_id=None):
            if len(dependencies) == 0:
                dependencies = scheduler.nodes
            for d in dependencies:
                try:
                    if scheduler.counts_total[execution_id][time_scale][d] < 1:
                        return False
                except AttributeError as e:
                    raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')
                except KeyError as e:
                    raise ConditionError(
                        f'{type(self).__name__}: execution_id ({scheduler}) must both be specified, and '
                        f'execution_id must be in scheduler.counts_total (scheduler: {execution_id}): {e}.')
            return True
        super().__init__(func, *dependencies)


class WhenFinished(_DependencyValidation, Condition):
    """WhenFinished

    Parameters:

        component(Component):  the Component on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of the Component specified in **components** returns `True`.

    Notes:

        - This is a dynamic Condition: Each Component is responsible for managing its finished status on its
          own, which can occur independently of the execution of other Components.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

    """
    def __init__(self, dependency):
        def func(dependency, context=None):
            try:
                return dependency.is_finished(context)
            except AttributeError as e:
                raise ConditionError(f'WhenFinished: Unsupported dependency type: {type(dependency)}; ({e}).')

        super().__init__(func, dependency)


class WhenFinishedAny(_DependencyValidation, Condition):
    """WhenFinishedAny

    Parameters:

        *components(Components):  zero or more Components on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of any Components specified in **components** returns `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          Any(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no components are specified, the condition will default to checking all of scheduler's Components.

        - This is a dynamic Condition: Each Component is responsible for managing its finished status on its
          own, which can occur independently of the execution of other Components.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

    """
    def __init__(self, *dependencies):
        def func(*dependencies, scheduler=None, context=None):
            if len(dependencies) == 0:
                dependencies = scheduler.nodes
            for d in dependencies:
                try:
                    if d.is_finished(context):
                        return True
                except AttributeError as e:
                    raise ConditionError(f'WhenFinishedAny: Unsupported dependency type: {type(d)}; ({e}).')
            return False

        super().__init__(func, *dependencies)


class WhenFinishedAll(_DependencyValidation, Condition):
    """WhenFinishedAll

    Parameters:

        *components(Components):  zero or more Components on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of all Components specified in **components** return `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          All(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no components are specified, the condition will default to checking all of scheduler's Components.

        - This is a dynamic Condition: Each Component is responsible for managing its finished status on its
          own, which can occur independently of the execution of other Components.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

    """
    def __init__(self, *dependencies):
        def func(*dependencies, scheduler=None, execution_id=None):
            if len(dependencies) == 0:
                dependencies = scheduler.nodes
            for d in dependencies:
                try:
                    if not d.is_finished(execution_id):
                        return False
                except AttributeError as e:
                    raise ConditionError(f'WhenFinishedAll: Unsupported dependency type: {type(d)}; ({e})')
            return True

        super().__init__(func, *dependencies)


######################################################################
# Convenience Conditions
######################################################################


class AtTrialStart(AtPass):
    """AtTrialStart

    Satisfied when:

        - at the beginning of a `TRIAL <TimeScale.TRIAL>`

    Notes:

        - identical to `AtPass(0) <AtPass>`
    """
    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return '{0}()'.format(self.__class__.__name__)


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
    def __init__(self, n, time_scale=TimeScale.RUN):
        return super.__init__(AtPass(0), AtTrial(n, time_scale))


class AtRunStart(AtTrial):
    """AtRunStart

    Satisfied when:

        - at the beginning of a `RUN`

    Notes:

        - identical to `AtTrial(0) <AtTrial>`
    """
    def __init__(self):
        super().__init__(0, time_scale=TimeScale.RUN)

    def __str__(self):
        return '{0}()'.format(self.__class__.__name__)


class AtRunNStart(All):
    """AtRunNStart

    Parameters:

        n(int): the `RUN` on which the Condition is satisfied

    Satisfied when:

        - on `TRIAL <TimeScale.TRIAL>` 0 of the specified `RUN` counted using 'TimeScale`

    Notes:

        - identical to `All(AtTrial(0), AtRun(n))`

    """
    def __init__(self, n):
        return super.__init__(AtTrial(0), AtRun(n))
