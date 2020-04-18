# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Time ***************************************************************

"""

.. _Time_Overview:

Overview
--------

:doc:`Scheduler`\\ s maintain `Clock` objects to track time. The current time in \
relation to a :doc:`Scheduler` is stored in :class:`Scheduler.clock.time <Time>` \
or :class:`Scheduler.clock.simple_time <SimpleTime>`

"""

import copy
import enum
import functools
import types

__all__ = [
    'Clock', 'TimeScale', 'Time', 'SimpleTime', 'TimeHistoryTree', 'TimeScaleError'
]


class TimeScaleError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# Time scale modes
@functools.total_ordering
class TimeScale(enum.Enum):
    """Represents divisions of time used by the `Scheduler`, `Conditions <Condition>`, and the **time_scale**
    argument of a Composition's `run <Composition.run>` method.

    The values of TimeScale are defined as follows (in order of increasingly coarse granularity):

    Attributes
    ----------

    TIME_STEP
        the nuclear unit of time, corresponding to the execution of all `Mechanism <Mechanism>`\\ s allowed to execute
        from a single `consideration_set <consideration_set>` of a `Scheduler`, and which are considered to have
        executed simultaneously.

    PASS
        a full iteration through all of the `consideration_sets <consideration_set>` in a `Scheduler's <Scheduler>`
        `consideration_queue`, consisting of one or more `TIME_STEPs <TIME_STEP>`, over which every `Component
        <Component>` `specified to a Scheduler <Scheduler_Creation>` is considered for execution at least once.

    TRIAL
        an open-ended unit of time consisting of all actions that occurs within the scope of a single input to a
        `Composition <Composition>`.

    RUN
        the scope of a call to the `run <Composition.run>` method of a `Composition <Composition>`,
        consisting of one or more `TRIALs <TimeScale.TRIAL>`.

    LIFE
        the scope of time since the creation of an object.
    """
    TIME_STEP = 0
    PASS = 1
    TRIAL = 2
    RUN = 3
    LIFE = 4

    # ordering based on enum.OrderedEnum example
    # https://docs.python.org/3/library/enum.html#orderedenum
    # https://stackoverflow.com/a/39269589/3131666
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    @classmethod
    def get_parent(cls, time_scale):
        """
        Returns
        -------
            the TimeScale one level wider in scope than time_scale : :class:`TimeScale`
        """
        return cls(time_scale.value + 1)

    @classmethod
    def get_child(cls, time_scale):
        """
        Returns
        -------
            the TimeScale one level smaller in scope than time_scale : :class:`TimeScale`
        """
        return cls(time_scale.value - 1)


class Clock:
    """
    Stores a history of :class:`TimeScale`\\ s that have occurred, and keep track of a \
    current `Time`. Used in relation to a :doc:`Scheduler`

    Attributes
    ----------
        history : `TimeHistoryTree`
            a root `TimeHistoryTree` associated with this Clock
    """
    def __init__(self):
        self.history = TimeHistoryTree()
        self._simple_time = SimpleTime()

    def __repr__(self):
        return 'Clock({0})'.format(self.time.__repr__())

    def _increment_time(self, time_scale):
        """
        Calls `self.history.increment_time <TimeHistoryTree.increment_time>`
        """
        self.history.increment_time(time_scale)

    def get_total_times_relative(self, query_time_scale, base_time_scale, base_index=None):
        """
        Convenience simplified wrapper for `TimeHistoryTree.get_total_times_relative`

        Arguments
        ---------
            query_time_scale : :class:`TimeScale`
                the unit of time whose number of ticks to be returned

            base_time_scale : :class:`TimeScale`
                the unit of time over which the number of **query_time_scale** ticks
                should be returned

            base_index : int
                the **base_index**\\ th **base_time_scale** over which the number of
                **query_time_scale** ticks should be returned
        Returns
        -------
            the number of query_time_scale s that have occurred during the scope \
            of the base_index th base_time_scale : int
        """
        if base_index is None:
            base_index = self.get_time_by_time_scale(base_time_scale)

        return self.history.get_total_times_relative(
            query_time_scale,
            {base_time_scale: base_index}
        )

    def get_time_by_time_scale(self, time_scale):
        """
        Arguments
        ---------
            time_scale : :class:`TimeScale`

        Returns
        -------
            the current value of the time unit corresponding to time_scale \
            for this Clock : int
        """
        return self.time._get_by_time_scale(time_scale)

    @property
    def time(self):
        """
        the current time : `Time`
        """
        return self.history.current_time

    @property
    def simple_time(self):
        """
        the current time in simple format : `SimpleTime`
        """
        self._simple_time.run = self.time.run
        self._simple_time.trial = self.time.trial
        self._simple_time.time_step = self.time.time_step
        return self._simple_time

    @property
    def previous_time(self):
        """
        the time that has occurred last : `Time`
        """
        return self.history.previous_time


class Time(types.SimpleNamespace):
    """
    Represents an instance of time, having values for each :class:`TimeScale`

    Attributes
    ----------
        life : int : 0
            the `TimeScale.LIFE` value

        run : int : 0
            the `TimeScale.RUN` value

        trial : int : 0
            the `TimeScale.TRIAL` value

        pass_ : int : 0
            the `TimeScale.PASS` value

        time_step : int : 0
            the `TimeScale.TIME_STEP` value

    """
    _time_scale_attr_map = {
        TimeScale.TIME_STEP: 'time_step',
        TimeScale.PASS: 'pass_',
        TimeScale.TRIAL: 'trial',
        TimeScale.RUN: 'run',
        TimeScale.LIFE: 'life'
    }

    def __init__(self, time_step=0, pass_=0, trial=0, run=0, life=0):
        super().__init__(time_step=time_step, pass_=pass_, trial=trial, run=run, life=life)

    def _get_by_time_scale(self, time_scale):
        """
        Arguments
        ---------
            time_scale : :class:`TimeScale`

        Returns
        -------
            this Time's value of a TimeScale by the TimeScale enum, rather \
            than by attribute : int
        """
        return getattr(self, self._time_scale_attr_map[time_scale])

    def _set_by_time_scale(self, time_scale, value):
        """
        Arguments
        ---------
            time_scale : :class:`TimeScale`

        Sets this Time's value of a **time_scale** by the TimeScale enum,
        rather than by attribute
        """
        setattr(self, self._time_scale_attr_map[time_scale], value)

    def _increment_by_time_scale(self, time_scale):
        """
        Increments the value of **time_scale** in this Time by one
        """
        self._set_by_time_scale(time_scale, self._get_by_time_scale(time_scale) + 1)
        self._reset_by_time_scale(time_scale)

    def _reset_by_time_scale(self, time_scale):
        """
        Resets all the times for the time scale scope up to **time_scale**
        e.g. _reset_by_time_scale(TimeScale.TRIAL) will set the values for
        TimeScale.PASS and TimeScale.TIME_STEP to 0
        """
        for relative_time_scale in TimeScale:
            # this works because the enum is set so that higher granularities of time have lower values
            if relative_time_scale >= time_scale:
                continue

            self._set_by_time_scale(relative_time_scale, 0)


class SimpleTime(types.SimpleNamespace):
    """
    A subset class of `Time`, used to provide simple access to only
    `run <Time.run>`, `trial <Time.trial>`, and `time_step <Time.time_step>`
    """
    def __init__(self, run=0, trial=0, time_step=0):
        super().__init__(run=0, trial=0, time_step=0)

    # override __repr__ because this class is used only for cosmetic simplicity
    # based on a Time object
    def __repr__(self):
        return 'Time(run: {0}, trial: {1}, time_step: {2})'.format(self.run, self.trial, self.time_step)


class TimeHistoryTree:
    """
    A tree object that stores a history of time that has occurred at various
    :class:`TimeScale`\\ s, typically used in conjunction with a `Clock`

    Attributes
    ----------
        time_scale : :class:`TimeScale` : `TimeScale.LIFE`
            the TimeScale unit this tree/node represents

        child_time_scale : :class:`TimeScale` : `TimeScale.RUN`
            the TimeScale unit for this tree's children

        children : list[`TimeHistoryTree`]
            an ordered list of this tree's children

        max_depth : :class:`TimeScale` : `TimeScale.TRIAL`
            the finest grain TimeScale that should be created as a subtree
            Setting this value lower allows for more precise measurements
            (by default, you cannot query the number of
            `TimeScale.TIME_STEP`\\ s in a certain `TimeScale.PASS`), but
            this may use a large amount of memory in large simulations

        index : int
            the index this tree has in its parent's children list

        parent : `TimeHistoryTree` : None
            the parent node of this tree, if it exists. \
            None represents no parent (i.e. root node)

        previous_time : `Time`
            a `Time` object that represents the last time that has occurred in the tree

        current_time : `Time`
            a `Time` object that represents the current time in the tree

        total_times : dict{:class:`TimeScale`: int}
            stores the total number of units of :class:`TimeScale`\\ s that have \
            occurred over this tree's scope. Only contains entries for \
            :class:`TimeScale`\\ s of finer grain than **time_scale**

    Arguments
    ---------
        enable_current_time : bool : True
            sets this tree to maintain a `Time` object. If this tree is not
            a root (i.e. **time_scale** is `TimeScale.LIFE`)
    """
    def __init__(
        self,
        time_scale=TimeScale.LIFE,
        max_depth=TimeScale.TRIAL,
        index=0,
        parent=None,
        enable_current_time=True
    ):
        if enable_current_time:
            self.current_time = Time()
            self.previous_time = None
        self.index = index
        self.time_scale = time_scale
        self.max_depth = max_depth
        self.parent = parent

        self.child_time_scale = TimeScale.get_child(time_scale)

        if self.child_time_scale >= max_depth:
            self.children = [
                TimeHistoryTree(
                    self.child_time_scale,
                    max_depth=max_depth,
                    index=0,
                    parent=self,
                    enable_current_time=False
                )
            ]
        else:
            self.children = []

        self.total_times = {ts: 0 for ts in TimeScale if ts < self.time_scale}

    def increment_time(self, time_scale):
        """
        Increases this tree's **current_time** by one **time_scale**

        Arguments
        ---------
            time_scale : :class:`TimeScale`
                the unit of time to increment
        """
        if self.child_time_scale >= self.max_depth:
            if time_scale == self.child_time_scale:
                self.children.append(
                    TimeHistoryTree(
                        self.child_time_scale,
                        max_depth=self.max_depth,
                        index=len(self.children),
                        parent=self,
                        enable_current_time=False
                    )
                )
            else:
                self.children[-1].increment_time(time_scale)
        self.total_times[time_scale] += 1
        try:
            self.previous_time = copy.copy(self.current_time)
            self.current_time._increment_by_time_scale(time_scale)
        except AttributeError:
            # not all of these objects have time tracking
            pass

    def get_total_times_relative(
        self,
        query_time_scale,
        base_indices=None
    ):
        """
        Arguments
        ---------
            query_time_scale : :class:`TimeScale`
                the :class:`TimeScale` of units to be returned

            base_indices : dict{:class:`TimeScale`: int}
                a dictionary specifying what scope of time query_time_scale \
                is over. e.g.

                    base_indices = {TimeScale.RUN: 1, TimeScale.TRIAL: 5}

                gives the number of **query_time_scale**\\ s that have occurred \
                in the 5th `TRIAL <TimeScale.TRIAL>` of the 1st `RUN`. If an entry for a :class:`TimeScale` \
                is not specified but is coarser than **query_time_scale**, the latest \
                value for that entry will be used

        Returns
        -------
            the number of units of query_time_scale that have occurred within \
            the scope of time specified by base_indices : int
        """
        if query_time_scale >= self.time_scale:
            raise TimeScaleError(
                'query_time_scale (given: {0}) must be of finer grain than {1}.time_scale ({2})'.format(
                    query_time_scale, self, self.time_scale
                )
            )

        try:
            self.current_time
        except AttributeError:
            raise TimeScaleError(
                'get_total_times_relative should only be called on a TimeHistoryTree with enable_current_time set to True'
            )

        default_base_indices = {
            TimeScale.LIFE: 0,
            TimeScale.RUN: None,
            TimeScale.TRIAL: None,
            TimeScale.PASS: None,
        }

        # overwrite defaults with dictionary passed in argument
        if base_indices is None:
            base_indices = default_base_indices
        else:
            default_base_indices.update(base_indices)
            base_indices = default_base_indices

        base_time_scale = TimeScale.LIFE
        # base_time_scale is set as the finest grain TimeScale that is specified,
        # but more coarse than query_time_scale
        # this will be where the query to attribute times will be made
        for ts in sorted(base_indices, reverse=True):
            if base_indices[ts] is not None and ts > query_time_scale:
                base_time_scale = ts

        if base_time_scale > self.time_scale:
            raise TimeScaleError(
                'base TimeScale set by base_indices ({0}) must be at least as fine as this TimeHistoryTree\'s time_scale ({1})'.format(
                    base_time_scale,
                    self.time_scale
                )
            )

        # get the root node, which will (and should) have TimeScale.LIFE
        node = self
        while node.parent is not None:
            node = node.parent

        try:
            # for all non-specified (i.e. set to None) TimeScales coarser than base_time_scale,
            # assign them to their latest time values as default
            while node.time_scale > base_time_scale:
                if base_indices[node.child_time_scale] is None:
                    base_indices[node.child_time_scale] = len(node.children) - 1
                node = node.children[base_indices[node.child_time_scale]]

            # attempt to retrieve the correct time count given the base_indices dictionary
            node = self
            while node.time_scale != base_time_scale:
                node = node.children[base_indices[node.child_time_scale]]
            return node.total_times[query_time_scale]
        except IndexError:
            raise TimeScaleError(
                'TimeHistoryTree {0}: {1} {2} does not exist in {3} {4}'.format(
                    self,
                    node.child_time_scale,
                    base_indices[node.child_time_scale],
                    node.time_scale,
                    node.index
                )
            )
