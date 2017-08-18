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
Composition (e.g., how many `TIME_STEP` s have occurred in the current `TRIAL`), or the state of other
Components in a Composition (e.g., whether or how many times they have executed). PsyNeuLink provides a number of
`pre-specified Conditions <Condition_Pre_Specified>` that can be parametrized (e.g., how many times a Component should
be executed). `Custom conditions <Condition_Custom>` can also be created, by assigning a function to a Condition that
can reference any Component or its attributes in PsyNeuLink, thus providing considerable flexibility for scheduling.

.. note::
    Any Component that is part of a collection `specified to a Scheduler for execution <Scheduler_Creation>` can be
    associated with a Condition.  Most commonly, these are `Mechanisms <Mechanism>`.  However, in some circumstances
    `Projections <Projection>` can be included in the specification to a Scheduler (e.g., for
    `learning <Process_Learning_Sequence>`) in which case these can also be assigned Conditions.



.. _Condition_Creation:

Creating Conditions
-------------------

.. _Condition_Pre_Specified:

Pre-specified Conditions
~~~~~~~~~~~~~~~~~~~~~~~~

`Pre-specified Conditions <Condition_Pre-Specified_List>` can be instantiated and added to a `Scheduler` at any time,
and take effect immediately for the execution of that Scheduler. Most pre-specified Conditions have one or more
arguments that must be specified to achieve the desired behavior. Many Conditions are also associated with an
`owner <Condition.owner>` attribute (a `Component` to which the Condition belongs), and a
`scheduler <Condition.scheduler>` attribute (that maintains data used to test for satisfaction of the Condition).
When pre-specified Conditions are instantiated within a call to the `add` method of a `Scheduler` or `ConditionSet`,
the Condition's `owner <Condition.owner>` and `scheduler <Condition.scheduler>` attributes are determined through
context and assigned automatically, as in the following example::

    my_scheduler.add_condition(A, EveryNPasses(1))
    my_scheduler.add_condition(B, EveryNCalls(A, 2))
    my_scheduler.add_condition(C, EveryNCalls(B, 2))

Here, `EveryNCalls(A, 2)` for example, is assigned the `owner` `B`, and the scheduler `my_scheduler`.

.. _Condition_Custom:

Custom Conditions
~~~~~~~~~~~~~~~~~

COMMENT:
    K: Thinking about it I kind of like making basic wrappers While and Until, where While is exactly the same as
        base Condition, but perhaps more friendly sounding? It evals to the output of the function exactly
        Until would just be the inversion of the function. Thoughts?
    JDC: THIS SOUNDS GOOD.
    JDC: PS - MIGHT WANT TO ADD "When", WHICH IS WHAT I THINK WE WANT FOR THE converge EXAMPLE;
                        my_scheduler.add_condition(A, Until(converge, B, epsilon))
                    CAUSES A TO EXECUTE UNTIL THE CONDITION ON B BECOMES TRUE, WHICH IS INDEED THE INVERSE OF WHILE,
                    (WHICH WOULD EXECUTE UNTIL B BECOMES FALSE);, BUT NOT WHAT WE WANT FOR CONVERGE
                    COULD USE WHILE:
                        my_scheduler.add_condition(A, While(converge, B, epsilon)))
                    WHICH WOULD WAIT UNTIL B CONVERGED, BUT SEEMS IT WOULD THEN CONTINUE TO EXECUTE AS LONG AS
                    B REMAINED "CONVERGED";
                        my_scheduler.add_condition(A, When(converge, B, epsilon)))
                    SUGGESTS (AT LEAST TO ME) THAT IT WILL HAPPEN WHEN B CONVERGES -- I.E., A WILL EXECUTE THEN
                    BUT NOT AGAIN;  MAYBE THAT CAUSES OTHER PROBLEMS (E.G., HOW WOULD THE SCHEDULER KNOW IF
                    B HAS RESET;  IS THIS SIMILAR TO THE ISSUE OF "EVERY" THAT REQUIRES "usable countes"?)
                    SEEMS LIKE WE SHOULD DISCUSS (AT LEAST SO I CAN UNDERSTAND BETTER)
COMMENT

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
a `TransferMechanism` (which reports the change in its `value <TransferMechanism.value>`). The function is assigned to
the standard `Condition()` with `mech_A` and `epsilon` as its arguments, and `composite Condition <Conditions_Composite>`
`NWhen` (which is satisfied the first N times after its condition becomes true),  The Condition is assigned to `mech_B`,
thus scheduling it to execute one time when all of the elements of `mech_A` have changed by less than `epsilon`.

.. _Condition_Structure:

Structure
---------

The `Scheduler` associates every Component with a Condition.  If a Component has not been explicitly assigned a
Condition, it is assigned the Condition `Always` that causes it to be executed whenever it is
`under consideration <Scheduler_Algorithm>`.  Condition subclasses (`listed below <Condition_Pre-Specified_List>`)
provide a standard set of Conditions that can be implemented simply by specifying their parameter(s). There are
five types:

  * `Generic <Conditions_Generic>` - satisfied when a `user-specified function and set of arguments <Condition_Custom>`
    evaluates to `True`;
  * `Static <Conditions_Static>` - satisfied either always or never;
  * `Composite <Conditions_Composite>` - satisfied based on one or more other Conditions;
  * `Time-based <Conditions_Time_Based>` - satisfied based on the current count of units of time at a specified
    `TimeScale`;
  * `Component-based <Conditions_Component_Based>` - based on the execution or state of other Components.

.. _Condition_Pre-Specified_List:

List of Pre-specified Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    The optional `TimeScale` argument in many `Conditions <Condition>` specifies the unit of time over which the
    Condition operates;  the default value is `TRIAL` for all Conditions except those with "Trial" in their name,
    for which it is `RUN`.

COMMENT:
    JDC: ADDED THESE PROVISIONAL ON IMPLEMENTING THE SUGGESTION ABOVE
    K: the condition will have to keep an internal counter, which increments every time it is satisfied, and
        fails to satisfy after N satisfactions
        Additionally, there are two ways it must be implemented, NWhen(Condition, int) would work, but to use
        the func/args/kwargs right within the NWhen construction you would need to specify n as a keyword arg
        NWhen(func, args, n=None, kwargs), due to python arguments. This would differ from every other condition
        where n can be specified without the explicit n=
COMMENT

COMMENT:
    K: I don't think we need to comment on how Always causes execution in its description,
    because it's mentioned right above
    JDC: I SEE WHAT YOU MEAN, BUT I'M INCLINED TOWARD CONSISTENCY AND COMPLENESS, EVEN AT THE EXPENSE OF OCCASIONAL
         REDUNDANCY;  IT WILL ALSO BE A BIT MORE SEPARATE IF WE INCLUDE THE "GENERIC" CATEGORY I'VE ADDED ABOVE
    K: I think mainly I just prefer to avoid referencing execution in individual conditions, instead using "satisfied"
COMMENT

.. _Conditions_Generic:

**Generic Conditions** (used to construct `custom Conditions <Condition_Custom>`):

    * `While`\\ (func, *args, **kwargs)
      \
      satisfied whenever the specified function (or callable) called with args and/or kwargs evaluates to `True`. \
      Equivalent to `Condition(func, *args, **kwargs)`

    * `WhileNot`\\ (func, *args, **kwargs)
      \
      satisfied whenever the specified function (or callable) called with args and/or kwargs evaluates to `False`. \
      Equivalent to `Not(Condition(func, *args, **kwargs))`

.. _Conditions_Static:

**Static Conditions** (independent of other Conditions, Components or time):

    * `Always`
      \
      always satisfied.

    * `Never`
      \
      never satisfied.


.. _Conditions_Composite:

**Composite Conditions** (based on one or more other Conditions):

    * `All`\\ (*Conditions)
      \
      satisfied whenever all of the specified Conditions are satisfied.

    * `Any`\\ (*Conditions)
      \
      satisfied whenever any of the specified Conditions are satisfied.

    * `Not`\\ (Condition)
      \
      satisfied whenever the specified Condition is not satisfied.

    * `NWhen`\\ (Condition, int)
      \
      satisfied the first specified number of times the specified Condition is satisfied.


.. _Conditions_Time_Based:

**Time-Based Conditions** (based on the count of units of time at a specified `TimeScale`):


    * `BeforePass`\\ (int[, TimeScale])
      \
      satisfied any time before the specified `PASS` occurs.

    * `AtPass`\\ (int[, TimeScale])
      \
      satisfied only during the specified `PASS`.

    * `AfterPass`\\ (int[, TimeScale])
      \
      satisfied any time after the specified `PASS` has occurred.

    * `AfterNPasses`\\ (int[, TimeScale])
      \
      satisfied when or any time after the specified number of `PASS`\\es has occurred.

    * `EveryNPasses`\\ (int[, TimeScale])
      \
      satisfied every time the specified number of `PASS`\\ es occurs.

    * `BeforeTrial`\\ (int[, TimeScale])
      \
      satisfied any time before the specified `TRIAL` occurs.

    * `AtTrial`\\ (int[, TimeScale])
      \
      satisfied any time during the specified `TRIAL`.

    * `AfterTrial`\\ (int[, TimeScale])
      \
      satisfied any time after the specified `TRIAL` occurs.

    * `AfterNTrials`\\ (int[, TimeScale])
      \
      satisfied any time after the specified number of `TRIAL`\\s has occurred.


.. _Conditions_Component_Based:

**Component-Based Conditions** (based on the execution or state of other Components):


    * `BeforeNCalls`\\ (Component, int[, TimeScale])
      \
      satisfied any time before the specified Component has executed the specified number of times.

    * `AtNCalls`\\ (Component, int[, TimeScale])
      \
      satisfied when the specified Component has executed the specified number of times.

    * `AfterCall`\\ (Component, int[, TimeScale])
      \
      satisfied any time after the Component has executed the specified number of times.

    * `AfterNCalls`\\ (Component, int[, TimeScale])
      \
      satisfied when or any time after the Component has executed the specified number of times.

    * `AfterNCallsCombined`\\ (*Components, int[, TimeScale])
      \
      satisfied when or any time after the specified Components have executed the specified number \
      of times among themselves, in total.

    * `EveryNCalls`\\ (Component, int[, TimeScale])
      \
      satisfied when the specified Component has executed the specified number of times since the \
      last time `owner` has run.

    * `JustRan`\\ (Component)
      \
      satisfied if the specified Component was assigned to run in the previous `TIME_STEP`.

    * `AllHaveRun`\\ (*Components)
      \
      satisfied when all of the specified Components have executed at least once.

    * `WhenFinished`\\ (Component)
      \
      satisfied when the specified Component has set its `is_finished` attribute to `True`.

    * `WhenFinishedAny`\\ (*Components)
      \
      satisfied when any of the specified Components has set their `is_finished` attribute to `True`.

    * `WhenFinishedAll`\\ (*Components)
      \
      satisfied when all of the specified Components have set their `is_finished` attributes to `True`.


.. Condition_Execution:

Execution
---------

When the `Scheduler` `runs <Schedule_Execution>`, it makes a sequential `PASS` through its `consideration_queue`,
evaluating each `consideration_set` in the queue to determine which Components should be assigned to execute.
It evaluates the Components in each set by calling the `is_satisfied` method of the Condition associated with each
of those Components.  If it returns `True`, then the Component is assigned to the execution set for the `TIME_STEP`
of execution generated by that `PASS`.  Otherwise, the Component is not executed.

.. _Condition_Class_Reference:

Class Reference
---------------

"""

import logging

from PsyNeuLink.Scheduling.TimeScale import TimeScale

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

    scheduler : Scheduler
        specifies the `Scheduler` used to evaluate and maintain a record of the information required to
        evaluate the `Conditions <Condition>`

    conditions : dict{`Component`: `Condition`}
        specifies an iterable collection of `Components <Component>` and the `Conditions <Condition>` associated
        with each.

    Attributes
    ----------

    scheduler : Scheduler
        specifies the `Scheduler` used to evaluate and maintain a record of the information required to
        evaluate the `Conditions <Condition>`

    conditions : dict{`Component`: `Condition`}
        the key of each entry is a `Component`, and its value is the `Condition <Condition>` associated
        with that Component.  Conditions can be added to the
        ConditionSet using the ConditionSet's `add_condition` method.

    """
    def __init__(self, scheduler=None, conditions=None):
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
        """Add a `Condition` to the ConditionSet.

        Arguments
        ---------

        owner : Component
            specifies the Component with which the **condition** should be associated.

        condition : Condition
            specifies the Condition, associated with the **owner** to be added to the ConditionSet.


        """
        logger.debug('add_condition: Setting scheduler of {0}, (owner {2}) to self.scheduler ({1})'.
                     format(condition, self.scheduler, owner))
        condition.owner = owner
        condition.scheduler = self.scheduler
        self.conditions[owner] = condition

    def add_condition_set(self, conditions):
        """Add a collection of `Conditions <Condition>` to the ConditionSet.

        Arguments
        ---------

        conditions : dict{`Component`: `Condition`}
            specifies an iterable collection of Conditions to be added to the ConditionSet, in the form of a dict
            each entry of which maps a `Component` (the key) to a `Condition <Condition>` (the value).

        """
        for owner in conditions:
            conditions[owner].owner = owner
            conditions[owner].scheduler = self.scheduler
            self.conditions[owner] = conditions[owner]


class Condition(object):
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

    scheduler : Scheduler
        the `Scheduler` with which the Condition is associated;  the Scheduler's state is used to evaluate whether
        the Condition`s specifications are satisfied.

    owner (Component):
        the `Component` with which the Condition is associated, and the execution of which it determines.

    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._scheduler = None
        self._owner = None

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
        '''
        the function called to determine satisfaction of this Condition.

        Returns
        -------
            True - if the Condition is satisfied
            False - if the Condition is not satisfied
        '''
        logger.debug('Condition ({0}) using scheduler {1}'.format(type(self).__name__, self.scheduler))
        has_args = len(self.args) > 0
        has_kwargs = len(self.kwargs) > 0

        if has_args and has_kwargs:
            return self.func(*self.args, **self.kwargs)
        if has_args:
            return self.func(*self.args)
        if has_kwargs:
            return self.func(**self.kwargs)
        return self.func()

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
        super().__init__(lambda *args, **kwargs: not func(*args, **kwargs), *args, **kwargs)

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

    @Condition.scheduler.setter
    def scheduler(self, value):
        for cond in self.args:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.args:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, *conds):
        for cond in conds:
            if not cond.is_satisfied():
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

    @Condition.scheduler.setter
    def scheduler(self, value):
        logger.debug('Any setter args: {0}'.format(self.args))
        for cond in self.args:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.args:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, *conds):
        for cond in conds:
            if cond.is_satisfied():
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
        super().__init__(lambda c: not c.is_satisfied(), condition)

    @Condition.scheduler.setter
    def scheduler(self, value):
        self.args[0].scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        self.args[0].owner = value


class NWhen(Condition):
    """NWhen

    Parameters:

        condition(Condition): a `Condition`

        n(int): the maximum number of times this condition will be satisfied

    Satisfied when:

        - the first **n** times **condition** is satisfied upon evaluation

    """
    def __init__(self, condition, n=1):
        self.satisfactions = 0

        super().__init__(self.satis, condition, n)

    @Condition.scheduler.setter
    def scheduler(self, value):
        self.args[0].scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        self.args[0].owner = value

    def satis(self, condition, n):
        if self.satisfactions < n:
            if condition.is_satisfied():
                self.satisfactions += 1
                return True
        return False


######################################################################
# Time-based Conditions
#   - satisfied based only on TimeScales
######################################################################


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
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] < n
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
          so, `AtPass(1)` is satisfied when a single `PASS` (`PASS` 0) has occurred, and `AtPass(2) is satisfied
          when two `PASS`\\ es have occurred (`PASS` 0 and `PASS` 1), etc..

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.PASS] == n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
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
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] > n
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
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] >= n
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
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] % n == 0
        super().__init__(func, n, time_scale)


class BeforeTrial(Condition):
    """BeforeTrial

    Parameters:

        n(int): the `TRIAL` before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\\ s (default: TimeScale.RUN)

    Satisfied when:

        - at most n-1 `TRIAL`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL` is 0, the second `TRIAL` is 1, etc.);
          so, `BeforeTrial(2)` is satisfied at `TRIAL` 0 and `TRIAL` 1.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] < n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
        super().__init__(func, n)


class AtTrial(Condition):
    """AtTrial

    Parameters:

        n(int): the `TRIAL` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\\ s (default: TimeScale.RUN)

    Satisfied when:

        - exactly n `TRIAL`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL` is 0, the second `TRIAL` is 1, etc.);
          so, `AtTrial(1)` is satisfied when one `TRIAL` (`TRIAL` 0) has already occurred.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] == n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
        super().__init__(func, n)


class AfterTrial(Condition):
    """AfterTrial

    Parameters:

        n(int): the `TRIAL` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\\ s. (default: TimeScale.RUN)

    Satisfied when:

        - at least n+1 `TRIAL`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL` is 0, the second `TRIAL` is 1, etc.);
          so,  `AfterPass(1)` is satisfied after `TRIAL` 1 has occurred and thereafter (i.e., in `TRIAL`\\ s 2, 3, 4,
          etc.).

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] > n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
        super().__init__(func, n)


class AfterNTrials(Condition):
    """AfterNTrials

    Parameters:

        n(int): the number of `TRIAL`\\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\\ s (default: TimeScale.RUN)

    Satisfied when:

        - at least n `TRIAL`\\ s have occured  within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.TRIAL] >= n
        super().__init__(func, n, time_scale)

######################################################################
# Component-based Conditions
#   - satisfied based on executions or state of Components
######################################################################


class BeforeNCalls(Condition):
    """BeforeNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at most n-1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls < n
        super().__init__(func, dependency, n)

# NOTE:
# The behavior of AtNCalls is not desired (i.e. depending on the order mechanisms are checked, B running AtNCalls(A, x))
# may run on both the xth and x+1st call of A; if A and B are not parent-child
# A fix could invalidate key assumptions and affect many other conditions
# Since this condition is unlikely to be used, it's best to leave it for now


class AtNCalls(Condition):
    """AtNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed exactly n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls == n
        super().__init__(func, dependency, n)


class AfterCall(Condition):
    """AfterCall

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at least n+1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls > n
        super().__init__(func, dependency, n)


class AfterNCalls(Condition):
    """AfterNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at least n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls >= n
        super().__init__(func, dependency, n)


class AfterNCallsCombined(Condition):
    """AfterNCallsCombined

    Parameters:

        *components(Components):  one or more Components on which the Condition depends

        n(int): the number of combined executions of all Components specified in **components** after which the \
        Condition is satisfied (default: None)

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)


    Satisfied when:

        - there have been at least n+1 executions among all of the Components specified in **components**
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, n=None, time_scale=TimeScale.TRIAL):
        logger.debug('{0} args: deps {1}, n {2}, ts {3}'.format(type(self).__name__, dependencies, n, time_scale))

        def func(*dependencies, n=None):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            if n is None:
                raise ConditionError('{0}: required keyword argument n is None'.format(type(self).__name__))
            count_sum = 0
            for d in dependencies:
                count_sum += self.scheduler.counts_total[time_scale][d]
                logger.debug('{0} has reached {1} num_calls in {2}'.
                             format(d, self.scheduler.counts_total[time_scale][d], time_scale.name))
            return count_sum >= n
        super().__init__(func, *dependencies, n=n)


class EveryNCalls(Condition):
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

        - Scheduler's count of each other Component that is "useable" by the Component is reset to 0.

    """
    def __init__(self, dependency, n):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_useable[dependency][self.owner]
            logger.debug('{0} has reached {1} num_calls'.format(dependency, num_calls))
            return num_calls >= n
        super().__init__(func, dependency, n)


class JustRan(Condition):
    """JustRan

    Parameters:

        component(Component):  the Component on which the Condition depends

    Satisfied when:

        - the Component specified in **component** executed in the previous `TIME_STEP`.

    Notes:

        - This Condition can transcend divisions between `TimeScales <TimeScale>`.
          For example, if A runs in the final `TIME_STEP` of a `TRIAL`,
          JustRan(A) is satisfied at the beginning of the next `TRIAL`.

    """
    def __init__(self, dependency):
        def func(dependency):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            logger.debug('checking if {0} in previous execution step set'.format(dependency))
            try:
                return dependency in self.scheduler.execution_list[-1]
            except TypeError:
                return dependency == self.scheduler.execution_list[-1]
        super().__init__(func, dependency)


class AllHaveRun(Condition):
    """AllHaveRun

    Parameters:

        *components(Components):  an iterable of Components on which the Condition depends

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - all of the Components specified in **components** have executed at least once
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, time_scale=TimeScale.TRIAL):
        def func(*dependencies):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                if self.scheduler.counts_total[time_scale][d] < 1:
                    return False
            return True
        super().__init__(func, *dependencies)


class WhenFinished(Condition):
    """WhenFinished

    Parameters:

        component(Component):  the Component on which the Condition depends

    Satisfied when:

        - the Component specified in **component** has set its `is_finished` attribute to `True`.

    Notes:

        - This is a dynamic Condition: Each Component is responsible for assigning its `is_finished` attribute on it
          own, which can occur independently of the execution of other Components.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

    """
    def __init__(self, dependency):
        def func(dependency):
            try:
                return dependency.is_finished
            except AttributeError as e:
                raise ConditionError('WhenFinished: Unsupported dependency type: {0}; ({1})'.
                                     format(type(dependency), e))

        super().__init__(func, dependency)


class WhenFinishedAny(Condition):
    """WhenFinishedAny

    Parameters:

        *components(Components):  zero or more Components on which the Condition depends

    Satisfied when:

        - any of the Components specified in **components** have set their `is_finished` attribute to `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          Any(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no components are specified, the condition will default to checking all of its scheduler's Components.

        - This is a dynamic Condition: Each Component is responsible for assigning its `is_finished` attribute on it
          own, which can occur independently of the execution of other Components.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

    """
    def __init__(self, *dependencies):
        def func(*dependencies):
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                try:
                    if d.is_finished:
                        return True
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAny: Unsupported dependency type: {0}; ({1})'.format(type(d), e))
            return False

        super().__init__(func, *dependencies)


class WhenFinishedAll(Condition):
    """WhenFinishedAll

    Parameters:

        *components(Components):  zero or more Components on which the Condition depends

    Satisfied when:

        - all of the Components specified in **components** have set their `is_finished` attributes to `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          All(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no components are specified, the condition will default to checking all of its scheduler's Components.

        - This is a dynamic Condition: Each Component is responsible for assigning its `is_finished` attribute on it
          own, which can occur independently of the execution of other Components.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

    """
    def __init__(self, *dependencies):
        def func(*dependencies):
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                try:
                    if not d.is_finished:
                        return False
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAll: Unsupported dependency type: {0}; ({1})'.format(type(d), e))
            return True

        super().__init__(func, *dependencies)
