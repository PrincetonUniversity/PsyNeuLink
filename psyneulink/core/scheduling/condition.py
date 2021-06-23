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

**Time-Based Conditions** (based on the count of units of time at a
specified `TimeScale` or `Time <Scheduler_Absolute_Time>`):

    * `TimeInterval` ([`pint.Quantity`, `pint.Quantity`, `pint.Quantity`])
      satisfied every time an optional amount of absolute time has
      passed in between an optional specified range

    * `TimeTermination` (`pint.Quantity`)
      satisfied after the given absolute time

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

import functools
import inspect

import dill
import graph_scheduler

from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.json import JSONDumpable
from psyneulink.core.globals.keywords import MODEL_SPEC_ID_TYPE
from psyneulink.core.globals.parameters import parse_context

__all__ = graph_scheduler.condition.__all__


def _create_as_pnl_condition(condition):
    import psyneulink as pnl

    try:
        pnl_class = getattr(pnl.core.scheduling.condition, type(condition).__name__)
    except (AttributeError, TypeError):
        return condition

    # already a pnl Condition
    if isinstance(condition, Condition):
        return condition

    if not issubclass(pnl_class, graph_scheduler.Condition):
        return None

    new_args = [_create_as_pnl_condition(a) or a for a in condition.args]
    new_kwargs = {k: _create_as_pnl_condition(v) or v for k, v in condition.kwargs.items()}
    sig = inspect.signature(pnl_class)

    if 'func' in sig.parameters or 'function' in sig.parameters:
        # Condition takes a function as an argument
        res = pnl_class(condition.func, *new_args, **new_kwargs)
    else:
        res = pnl_class(*new_args, **new_kwargs)

    res.owner = condition.owner
    return res


class Condition(graph_scheduler.Condition, JSONDumpable):
    @handle_external_context()
    def is_satisfied(self, *args, context=None, execution_id=None, **kwargs):
        if execution_id is None:
            try:
                execution_id = parse_context(context).execution_id
            except AttributeError:
                pass

        return super().is_satisfied(
            *args,
            context=context,
            execution_id=execution_id,
            **kwargs
        )

    @property
    def _dict_summary(self):
        from psyneulink.core.components.component import Component

        if type(self) is graph_scheduler.Condition:
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
            elif isinstance(a, graph_scheduler.Condition):
                a = a._dict_summary
            args_list.append(a)

        return {
            MODEL_SPEC_ID_TYPE: self.__class__.__name__,
            'function': func_val,
            'args': args_list,
            'kwargs': self.kwargs,
        }


# below produces psyneulink versions of each Condition class so that
# they are compatible with the extra changes made in Condition above
# (the scheduler does not handle Context objects or mdf/json export)
cond_dependencies = {}
pnl_conditions_module = locals()  # inserting into locals defines the classes

for cond_name in graph_scheduler.condition.__all__:
    sched_module_cond_obj = getattr(graph_scheduler.condition, cond_name)
    cond_dependencies[cond_name] = set(sched_module_cond_obj.__mro__[1:])

# iterate in order such that superclass types are before subclass types
for cond_name in sorted(
    graph_scheduler.condition.__all__,
    key=functools.cmp_to_key(lambda a, b: -1 if b in cond_dependencies[a] else 1)
):
    # don't substitute Condition because it is explicitly defined above
    if cond_name == 'Condition':
        continue

    sched_module_cond_obj = getattr(graph_scheduler.condition, cond_name)
    if (
        inspect.isclass(sched_module_cond_obj)
        and issubclass(sched_module_cond_obj, graph_scheduler.Condition)
    ):
        new_mro = []
        for cls_ in sched_module_cond_obj.__mro__:
            if cls_ is not graph_scheduler.Condition:
                try:
                    new_mro.append(pnl_conditions_module[cls_.__name__])

                except KeyError:
                    new_mro.append(cls_)
            else:
                new_mro.extend(Condition.__mro__[:-1])
        pnl_conditions_module[cond_name] = type(cond_name, tuple(new_mro), {})
    elif isinstance(sched_module_cond_obj, type):
        pnl_conditions_module[cond_name] = sched_module_cond_obj

    pnl_conditions_module[cond_name].__doc__ = sched_module_cond_obj.__doc__
