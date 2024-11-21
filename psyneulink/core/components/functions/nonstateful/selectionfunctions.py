#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************  SELECTION FUNCTIONS *****************************************************
"""

* `OneHot`

COMMENT:
* TBI Threshold
* `KWTA`
COMMENT

Functions that selects a subset of elements to maintain or transform, while nulling the others.

"""

__all__ = ['SelectionFunction', 'OneHot', 'max_vs_avg', 'max_vs_next']

import warnings

import numpy as np
from beartype import beartype

from psyneulink._typing import Optional, Literal

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import DefaultsFlexibility
from psyneulink.core.components.functions.function import (
    DEFAULT_SEED, Function, Function_Base, FunctionError,
    _random_state_getter, _seed_setter,
)
from psyneulink.core.globals.keywords import \
    (ALL, ARG_MAX, ARG_MAX_ABS, ARG_MAX_ABS_INDICATOR, ARG_MAX_INDICATOR,
     ARG_MIN, ARG_MIN_ABS, ARG_MIN_ABS_INDICATOR, ARG_MIN_INDICATOR,
     DETERMINISTIC, FIRST, LAST,
     MAX, MAX_ABS_INDICATOR, MAX_ABS_VAL, MAX_INDICATOR, MAX_VAL,
     MIN, MIN_ABS_INDICATOR, MIN_ABS_VAL, MIN_INDICATOR, MIN_VAL,
     MODE, ONE_HOT_FUNCTION, PREFERENCE_SET_NAME, PROB, PROB_INDICATOR,
     RANDOM, SELECTION_FUNCTION_TYPE)

from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import \
    REPORT_OUTPUT_PREF, PreferenceEntry, PreferenceLevel, ValidPrefSet

mode_options = [DETERMINISTIC, PROB, PROB_INDICATOR,
                ARG_MAX, ARG_MAX_ABS, ARG_MAX_INDICATOR, ARG_MAX_ABS_INDICATOR,
                ARG_MIN,  ARG_MIN_ABS, ARG_MIN_INDICATOR, ARG_MIN_ABS_INDICATOR,
                MAX_VAL, MAX_ABS_VAL, MAX_INDICATOR, MAX_ABS_INDICATOR,
                MIN_VAL, MIN_ABS_VAL, MIN_INDICATOR, MIN_ABS_INDICATOR]
tie_options = [ALL, FIRST, LAST, RANDOM]

# FIX: IMPLEMENT AS Functions
def max_vs_next(x):
    x_part = np.partition(x, -2)
    max_val = x_part[-1]
    next = x_part[-2]
    return max_val - next


def max_vs_avg(x):
    x_part = np.partition(x, -2)
    max_val = x_part[-1]
    others = x_part[:-1]
    return max_val - np.mean(others)


class SelectionFunction(Function_Base):
    """Functions that selects a particular value to maintain or transform, while nulling the others.
    """
    componentType = SELECTION_FUNCTION_TYPE


class OneHot(SelectionFunction):
    """
    OneHot(                \
         default_variable, \
         mode=MAX_VAL,     \
         direction=MAX,    \
         abs_val=FALSE     \
         indicator=FALSE,  \
         tie=ALL,          \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    Return an array with one non-zero value.

    .. _OneHot:

    `function <Selection.function>` returns an array the same length as the first item in `variable <OneHot.variable>`,
    with all of its values zeroed except one (unless there is a tie, which is handled as specified by **tie**); the
    following options can be used in any combination:

        * **mode**: determines how the non-zero value(s) in the array is (are) selected

            * *STANDARD*: value (or 1) for the element(s) with the maximum or minimum value(s) in the array,
              as specified by the options below; all other elements are zeroed;  this is the default.

            * *PROB*: value of probabilistically chosen element based on probabilities passed in second item
              of variable; if there is a tie, a single element is chosen probabilistically.

            * *PROB_INDICATOR*: same as *PROB* but chosen item is assigned a value of 1;
              if there is a tie, a single element is chosen probabilistically.

         * **direction**: *MAX* (default) or *MIN*
           determines whether the maximum or minimum value(s) in the array are selected.

         * **abs_val**: *False* (default) or *True*
           determines whether the absolute values of the elements in the array are used to
           select the maximum or minimum value(s).

         * **indicator**:  *False* (default) or *True*
           determines whether the selected values(s) is (are) replace with a value of 1.

         * **tie**: *ALL* (default), *FIRST*, *LAST* or *RANDOM*
           determines how a tie is handled when there is more than one element with the maximum or minimum value;

           *ALL*: selects all elements in the tie;

           *FIRST*: selects the value of the element with the lowest index;

           *LAST*: selects the value of the element with the lowest index;

           *RANDOM*: randomly selects one of the tied elements;

        The following convenience keywords can be used to specify particular combinations of options for the **mode**
        argument together with the **tie** argument (these are included mainly for backward compatibility):

            * *ARG_MAX*: signed value of a single element with the maximum signed value,
              or the one with lowest index if there is a tie.

            * *ARG_MAX_ABS*: absolute value of a single element with the maximum absolute value,
              or the one with lowest index if there is a tie.

            * *ARG_MAX_INDICATOR*: 1 in place of single element with maximum signed value,
              or the one with lowest index if there is a tie.

            * *ARG_MAX_ABS_INDICATOR*: 1 in place of single element with maximum absolute value,
              or the one with lowest index if there is a tie.

            * *MAX_VAL*: signed value of the element with the maximum signed value;
              if there is a tie, which elements are returned is determined by `tie_index <OneHot.tie_index>`.

            * *MAX_ABS_VAL*: absolute value of the element with the maximum absolute value;
              if there is a tie, which elements are returned is determined by `tie_index <OneHot.tie_index>`.

            * *MAX_INDICATOR*: 1 in place of the element with the maximum signed value;
              if there is a tie, which elements are returned is determined by `tie_index <OneHot.tie_index>`.

            * *MAX_ABS_INDICATOR*: 1 in place of the element(s) with the maximum absolute value;
              if there is a tie, which elements are returned is determined by `tie_index <OneHot.tie_index>`.

            * *ARG_MIN*: signed value of a single element with the minium signed value,
              or the one with lowest index if there is a tie.

            * *ARG_MIN_ABS*: absolute value of a single element with the minium absolute value,
              or the one with lowest index if there is a tie.

            * *ARG_MIN_INDICATOR*: 1 in place of single element with minimum signed value,
              or the one with lowest index if there is a tie.

            * *MIN_VAL*: signed value of the element with the minimum signed value,
              or all elements with the minimum value if there is a tie.

            * *MIN_ABS_VAL*: absolute value of element with the minimum absolute value;
              if there is a tie, which elements are returned is determined by `tie_index <OneHot.tie_index>`.

            * *MIN_INDICATOR*: 1 in place of the element with the minimum signed value;
              if there is a tie, which elements are returned is determined by `tie_index <OneHot.tie_index>`.

            * *MIN_ABS_INDICATOR*: 1 in place of the element with the minimum absolute value;
              if there is a tie, which elements are returned is determined by `tie_index <OneHot.tie_index>`.

    Arguments
    ---------

    variable : 2d np.array : default class_defaults.variable
        First (possibly only) item specifies a template for the array to be transformed;  if `mode <OneHot.mode>` is
        *PROB* then a 2nd item must be included that is a probability distribution with same length as 1st item.

    mode : DETERMINISTiC, PROB, PROB_INDICATOR,
    ARG_MAX, ARG_MAX_ABS, ARG_MAX_INDICATOR, ARG_MAX_ABS_INDICATOR,
    ARG_MIN, ARG_MIN_ABS, ARG_MIN_INDICATOR, ARG_MIN_ABS_INDICATOR,
    MAX_VAL, MAX_ABS_VAL, MAX_INDICATOR, MAX_ABS_INDICATOR,
    MIN_VAL, MIN_ABS_VAL, MIN_INDICATOR,  MIN_ABS_INDICATOR,
    : default ARG_MAX
        specifies how non-zero value(s) in the array returned by `function <OneHot.function>` are determined
        (see `above <OneHot>` for details).

    direction : MAX or MIN : default MAX
       specifies whether the maximum or minimum value(s) in the array are selected.
       (see `above <OneHot>` for details).

    abs_val : bool : default False
       specifies whether the absolute values of the elements in the array are used to
       select the maximum or minimum value(s).
       (see `above <OneHot>` for details).

    indicator :  bool : default False
       specifies whether the selected values(s) is (are) replace with a value of 1.
       (see `above <OneHot>` for details).

    tie : ALL, FIRST, LAST, RANDOM : default ALL
       specifies how a tie is handled when there is more than one element with the maximum or minimum value;
       (see `above <OneHot>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    bounds : None

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        1st item contains value to be transformed;  if `mode <OneHot.mode>` is *PROB*, 2nd item is a probability
        distribution, each element of which specifies the probability for selecting the corresponding element of the
        1st item.

    mode : DETERMINISTIC, PROB, PROB_INDICATOR,
    ARG_MAX, ARG_MAX_ABS, ARG_MAX_INDICATOR, ARG_MAX_ABS_INDICATOR,
    ARG_MIN, ARG_MIN_ABS, ARG_MIN_INDICATOR, ARG_MIN_ABS_INDICATOR,
    MAX_VAL, MAX_ABS_VAL, MAX_INDICATOR, MAX_ABS_INDICATOR,
    MIN_VAL, MIN_ABS_VAL, MIN_INDICATOR,  MIN_ABS_INDICATOR,
    : default ARG_MAX
        specifies how non-zero value(s) in the array returned by `function <OneHot.function>` are determined
        (see `above <OneHot>` for details).

    direction : MAX or MIN
       determines whether the maximum or minimum value(s) in the array are selected.
       (see `above <OneHot>` for details).

    abs_val : bool
       determines whether the absolute values of the elements in the array are used to
       select the maximum or minimum value(s).
       (see `above <OneHot>` for details).

    indicator :  bool
       determines whether the selected values(s) is (are) replace with a value of 1.
       (see `above <OneHot>` for details).

    tie : ALL, FIRST, LAST, RANDOM
       determines how a tie is handled when there is more than one element with the maximum or minimum value;
       (see `above <OneHot>` for details).

    random_state : numpy.RandomState
        private pseudorandom number generator

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).
    """

    componentName = ONE_HOT_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'OneHotClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(SelectionFunction.Parameters):
        """
            Attributes
            ----------

                mode
                    see `mode <OneHot.mode>`

                    :default value: `DETERMINISTIC`
                    :type: ``str``

                direction
                    see `direction <OneHot.direction>`

                    :default value: `MAX`
                    :type: ``str``

                abs_val
                    see `abs_val <OneHot.abs_val>`

                    :default value: `False`
                    :type: ``bool``

                indicator
                    see `indicator <OneHot.indicator>`

                    :default value: `False`
                    :type: ``bool``

                tie
                    see `tie <OneHot.tie>`

                    :default value: `ALL`
                    :type: ``str``

                random_state
                    see `random_state <OneHot.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``
        """
        mode = Parameter(DETERMINISTIC, stateful=False)
        direction = Parameter(MAX, stateful=False)
        abs_val = Parameter(False, stateful=False)
        indicator = Parameter(False, stateful=False)
        tie = Parameter(ALL, stateful=False)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_default=True, setter=_seed_setter)

        def _validate_mode(self, mode):
            if mode not in mode_options:
                # returns error message
                return 'not one of {0}'.format(mode_options)

        def _validate_ties(self, tie_index):
            if tie_index not in tie_options:
                # returns error message
                return 'not one of {0}'.format(tie_options)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 mode: Optional[Literal[
                     DETERMINISTIC, PROB, PROB_INDICATOR,
                     ARG_MAX, ARG_MAX_ABS, ARG_MAX_INDICATOR, ARG_MAX_ABS_INDICATOR,
                     ARG_MIN, ARG_MIN_ABS, ARG_MIN_INDICATOR, ARG_MIN_ABS_INDICATOR,
                     MAX_VAL, MAX_ABS_VAL, MAX_INDICATOR, MAX_ABS_INDICATOR,
                     MIN_VAL, MIN_ABS_VAL, MIN_INDICATOR, MIN_ABS_INDICATOR]] = None,
                 direction: Optional[Literal[MAX, MIN]] = None,
                 abs_val: Optional[bool] = None,
                 indicator: Optional[bool] = None,
                 tie: Optional[Literal[ALL, FIRST, LAST, RANDOM]]= None,
                 seed=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        reset_variable_shape_flexibility = False
        if mode in {PROB, PROB_INDICATOR} and default_variable is None:
            default_variable = [[0], [0]]
            reset_variable_shape_flexibility = True

        super().__init__(
            default_variable=default_variable,
            mode=mode,
            direction=direction,
            abs_val=abs_val,
            indicator=indicator,
            tie=tie,
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        if reset_variable_shape_flexibility:
            self._variable_shape_flexibility = DefaultsFlexibility.FLEXIBLE

    def _validate_params(self, request_set, target_set=None, context=None):

        if request_set[MODE] in {PROB, PROB_INDICATOR}:
            if not self.defaults.variable.ndim == 2:
                raise FunctionError("If {} for {} {} is set to {}, variable must be 2d array".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB))
            values = self.defaults.variable[0]
            prob_dist = self.defaults.variable[1]
            if len(values)!=len(prob_dist):
                raise FunctionError("If {} for {} {} is set to {}, the two items of its variable must be of equal "
                                    "length (len item 1 = {}; len item 2 = {}".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB,
                                           len(values), len(prob_dist)))
            if not all((elem>=0 and elem<=1) for elem in prob_dist)==1:
                raise FunctionError("If {} for {} {} is set to {}, the 2nd item of its variable ({}) must be an "
                                    "array of elements each of which is in the (0,1) interval".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB, prob_dist))
            if self.is_initializing:
                return
            # FIX 8/20/23: WHY DOES SUM COME UP WITH FLOATING POINT ERRORS?
            # if not np.sum(prob_dist)==1:
            if not np.allclose(np.sum(prob_dist), 1):
                raise FunctionError("If {} for {} {} is set to {}, the 2nd item of its variable ({}) must be an "
                                    "array of probabilities that sum to 1".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB, prob_dist))

        elif request_set[MODE] != DETERMINISTIC:
            # Ensure that mode is not specified counter to other options (except tie)
            if any([self.parameters.direction._user_specified,
                   self.parameters.abs_val._user_specified,
                   self.parameters.indicator._user_specified]):
                raise FunctionError(f"If {MODE} for {self.__class__.__name__} {Function.__name__} is not "
                                    f"set to 'DETERMINIST', then the 'direction', 'abs_val', and 'indicator' args "
                                    f"cannot be specified.")

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        if self.mode in {PROB, PROB_INDICATOR}:

            sum_ptr = builder.alloca(ctx.float_ty)
            builder.store(sum_ptr.type.pointee(-0.0), sum_ptr)

            rand_state_ptr = ctx.get_random_state_ptr(builder, self, state, params)
            rng_f = ctx.get_uniform_dist_function_by_state(rand_state_ptr)
            random_draw_ptr = builder.alloca(rng_f.args[-1].type.pointee)
            builder.call(rng_f, [rand_state_ptr, random_draw_ptr])
            random_draw = builder.load(random_draw_ptr)

            prob_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(1)])
            arg_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])

            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "search") as (b1, idx):

                current_ptr = b1.gep(arg_in, [ctx.int32_ty(0), idx])
                current = b1.load(current_ptr)

                # Update prefix sum
                current_prob_ptr = b1.gep(prob_in, [ctx.int32_ty(0), idx])
                sum_old = b1.load(sum_ptr)
                sum_new = b1.fadd(sum_old, b1.load(current_prob_ptr))
                b1.store(sum_new, sum_ptr)

                old_below = b1.fcmp_ordered("<=", sum_old, random_draw)
                new_above = b1.fcmp_ordered("<", random_draw, sum_new)
                cond = b1.and_(new_above, old_below)

                if self.mode == PROB:
                    val = current
                else:
                    val = current.type(1.0)

                write_val = b1.select(cond, val, val.type(0.0))
                cur_res_ptr = b1.gep(arg_out, [ctx.int32_ty(0), idx])
                builder.store(write_val, cur_res_ptr)

            return builder

        elif self.mode == DETERMINISTIC:
            direction = self.direction
            tie = self.tie
            abs_val_ptr = ctx.get_param_or_state_ptr(builder, self, self.parameters.abs_val, param_struct_ptr=params)
            indicator_ptr = ctx.get_param_or_state_ptr(builder, self, self.parameters.indicator, param_struct_ptr=params)

            abs_val = builder.load(abs_val_ptr)
            is_abs_val = builder.fcmp_unordered("!=", abs_val, abs_val.type(0))

            indicator = builder.load(indicator_ptr)
            is_indicator = builder.fcmp_unordered("!=", indicator, indicator.type(0))

        else:
            direction, abs_val, indicator, tie = self._parse_mode(self.mode)
            is_abs_val = ctx.bool_ty(abs_val)
            is_indicator = ctx.bool_ty(indicator)

        num_extremes_ptr = builder.alloca(ctx.int32_ty)
        builder.store(num_extremes_ptr.type.pointee(0), num_extremes_ptr)

        extreme_val_ptr = builder.alloca(ctx.float_ty)
        builder.store(extreme_val_ptr.type.pointee(float("NaN")), extreme_val_ptr)

        fabs_f = ctx.get_builtin("fabs", [extreme_val_ptr.type.pointee])

        with pnlvm.helpers.recursive_iterate_arrays(ctx, builder, arg_in, loop_id="count_extremes") as (loop_builder, current_ptr):

            current = loop_builder.load(current_ptr)
            current_abs = loop_builder.call(fabs_f, [current])
            current = builder.select(is_abs_val, current_abs, current)

            old_extreme = loop_builder.load(extreme_val_ptr)
            cmp_op = ">" if direction == MAX else "<"
            is_new_extreme = loop_builder.fcmp_unordered(cmp_op, current, old_extreme)

            with loop_builder.if_then(is_new_extreme):
                loop_builder.store(current, extreme_val_ptr)
                loop_builder.store(num_extremes_ptr.type.pointee(1), num_extremes_ptr)

            is_old_extreme = loop_builder.fcmp_ordered("==", current, old_extreme)
            with loop_builder.if_then(is_old_extreme):
                extreme_count = loop_builder.load(num_extremes_ptr)
                extreme_count = loop_builder.add(extreme_count, extreme_count.type(1))
                loop_builder.store(extreme_count, num_extremes_ptr)


        if tie == FIRST:
            extreme_start = num_extremes_ptr.type.pointee(0)
            extreme_stop = num_extremes_ptr.type.pointee(1)

        elif tie == LAST:
            extreme_stop = builder.load(num_extremes_ptr)
            extreme_start = builder.sub(extreme_stop, extreme_stop.type(1))

        elif tie == ALL:
            extreme_start = num_extremes_ptr.type.pointee(0)
            extreme_stop = builder.load(num_extremes_ptr)

        elif tie == RANDOM:
            rand_state_ptr = ctx.get_random_state_ptr(builder, self, state, params)
            rand_f = ctx.get_rand_int_function_by_state(rand_state_ptr)
            random_draw_ptr = builder.alloca(rand_f.args[-1].type.pointee)
            num_extremes = builder.load(num_extremes_ptr)

            builder.call(rand_f, [rand_state_ptr, ctx.int32_ty(0), num_extremes, random_draw_ptr])

            extreme_start = builder.load(random_draw_ptr)
            extreme_start = builder.trunc(extreme_start, ctx.int32_ty)
            extreme_stop = builder.add(extreme_start, extreme_start.type(1))

        else:
            assert False, "Unknown tie resolution: {}".format(tie)


        extreme_val = builder.load(extreme_val_ptr)
        extreme_write_val = builder.select(is_indicator, extreme_val.type(1), extreme_val)
        next_extreme_ptr = builder.alloca(num_extremes_ptr.type.pointee)
        builder.store(next_extreme_ptr.type.pointee(0), next_extreme_ptr)

        pnlvm.helpers.printf(ctx,
                             builder,
                             "{} replacing extreme values of %e from <%u,%u) out of %u\n".format(self.name),
                             extreme_val,
                             extreme_start,
                             extreme_stop,
                             builder.load(num_extremes_ptr),
                             tags={"one_hot"})

        with pnlvm.helpers.recursive_iterate_arrays(ctx, builder, arg_in, arg_out, loop_id="mark_extremes") as (loop_builder, current_ptr, out_ptr):
            current = loop_builder.load(current_ptr)
            current_abs = loop_builder.call(fabs_f, [current])
            current = builder.select(is_abs_val, current_abs, current)

            is_extreme = loop_builder.fcmp_ordered("==", current, extreme_val)
            current_extreme_idx = loop_builder.load(next_extreme_ptr)

            with loop_builder.if_then(is_extreme):
                next_extreme_idx = loop_builder.add(current_extreme_idx, current_extreme_idx.type(1))
                loop_builder.store(next_extreme_idx, next_extreme_ptr)

            is_after_start = loop_builder.icmp_unsigned(">=", current_extreme_idx, extreme_start)
            is_before_stop = loop_builder.icmp_unsigned("<", current_extreme_idx, extreme_stop)

            should_write_extreme = loop_builder.and_(is_extreme, is_after_start)
            should_write_extreme = loop_builder.and_(should_write_extreme, is_before_stop)

            write_value = loop_builder.select(should_write_extreme, extreme_write_val, extreme_write_val.type(0))
            loop_builder.store(write_value, out_ptr)

        return builder

    def _parse_mode(self, mode):
        """Convert mode spec to corresponding options.
        Here for convenience, but mostly for backward compatibility with old mode spec.
        """

        direction = None
        abs_val = None
        indicator = None
        tie = None

        if mode == ARG_MAX:
            direction = MAX
            abs_val = False
            indicator = False
            tie = FIRST

        elif mode == ARG_MAX_ABS:
            direction = MAX
            abs_val = True
            indicator = False
            tie = FIRST

        elif mode == ARG_MAX_INDICATOR:
            direction = MAX
            abs_val = False
            indicator = True
            tie = FIRST

        elif mode == ARG_MAX_ABS_INDICATOR:
            direction = MAX
            abs_val = True
            indicator = True
            tie = FIRST

        elif mode == MAX_VAL:
            direction = MAX
            abs_val = False
            indicator = False
            tie = ALL

        elif mode == MAX_ABS_VAL:
            direction = MAX
            abs_val = True
            indicator = False
            tie = ALL

        elif mode == MAX_INDICATOR:
            direction = MAX
            abs_val = False
            indicator = True
            tie = ALL

        elif mode == MAX_ABS_INDICATOR:
            direction = MAX
            abs_val = True
            indicator = True
            tie = ALL

        elif mode == ARG_MIN:
            direction = MIN
            abs_val = False
            indicator = False
            tie = FIRST

        elif mode == ARG_MIN_ABS:
            direction = MIN
            abs_val = True
            indicator = False
            tie = FIRST

        elif mode == ARG_MIN_INDICATOR:
            direction = MIN
            abs_val = False
            indicator = True
            tie = FIRST

        elif mode == ARG_MIN_ABS_INDICATOR:
            direction = MIN
            abs_val = True
            indicator = True
            tie = FIRST

        elif mode == MIN_VAL:
            direction = MIN
            abs_val = False
            indicator = False
            tie = ALL

        elif mode == MIN_ABS_VAL:
            direction = MIN
            abs_val = True
            indicator = False
            tie = ALL

        elif mode == MIN_INDICATOR:
            direction = MIN
            abs_val = False
            indicator = True
            tie = ALL

        elif mode == MIN_ABS_INDICATOR:
            direction = MIN
            abs_val = True
            indicator = True
            tie = ALL

        else:
            assert False, f"Unknown mode: {mode}"

        return direction, abs_val, indicator, tie

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : 2d np.array : default class_defaults.variable
           1st item is an array to be transformed;  if `mode <OneHot.mode>` is *PROB*, 2nd item must be an array of
           probabilities (i.e., elements between 0 and 1) of equal length as the 1st item.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        array with selected elements having non-zero values and all others having zeroes : np.array
            specified by `mode <OneHot.mode>`, `direction <OneHot.direction>`, `abs_val <OneHot.abs_val>`,
            `indicator <OneHot.indicator>`, and `tie <OneHot.tie>`.

        """

        mode = self.parameters.mode.get(context)
        direction = self.parameters.direction.get(context)
        abs_val = self.parameters.abs_val.get(context)
        indicator = self.parameters.indicator.get(context)
        tie = self.parameters.tie.get(context)

        if mode in {PROB, PROB_INDICATOR}:
            # 1st item of variable should be data, and 2nd a probability distribution for choosing
            if np.array(variable).ndim != 2:
                raise FunctionError(f"If {MODE} for {self.__class__.__name__} {Function.__name__} is set to "
                                    f"'PROB' or 'PROB_INDICATOR', variable must be a 2d array with the first item "
                                    f"being the data and the second being a probability distribution.")
            v = variable[0]
            prob_dist = variable[1]
            # if not prob_dist.any() and INITIALIZING in context:
            if not prob_dist.any():
                return self.convert_output_type(v)
            cum_sum = np.cumsum(prob_dist)
            random_state = self._get_current_parameter_value("random_state", context)
            random_value = random_state.uniform()
            chosen_item = next(element for element in cum_sum if element > random_value)
            chosen_in_cum_sum = np.where(cum_sum == chosen_item, 1, 0)
            if mode == PROB:
                result = v * chosen_in_cum_sum
            else:
                result = np.ones_like(v) * chosen_in_cum_sum

            # chosen_item = np.random.choice(v, 1, p=prob_dist)
            # one_hot_indicator = np.where(v == chosen_item, 1, 0)
            # return v * one_hot_indicator
            return result

        elif mode != DETERMINISTIC:
            direction, abs_val, indicator, tie = self._parse_mode(mode)

        array = variable

        if abs_val is True:
            array = np.absolute(variable)

        if direction == MAX:
            extreme_val = np.max(array)
            if extreme_val == -np.inf:
                warnings.warn(f"Array passed to {self.name} of {self.owner.name} is all -inf.")

        elif direction == MIN:
            extreme_val = np.min(array)
            if extreme_val == np.inf:
                warnings.warn(f"Array passed to {self.name} of {self.owner.name} is all inf.")

        else:
            assert False, f"Unknown direction: '{direction}'."

        extreme_indices = np.where(array == extreme_val)

        num_indices = len(extreme_indices[0])
        assert all(len(idx) == num_indices for idx in extreme_indices)

        if tie == FIRST:
            selected_idx = 0

        elif tie == LAST:
            selected_idx = -1

        elif tie == RANDOM:
            random_state = self._get_current_parameter_value("random_state", context)
            selected_idx = random_state.randint(num_indices)

        elif tie == ALL:
            selected_idx = slice(num_indices)

        else:
            assert False, f"PROGRAM ERROR: Unrecognized value for 'tie' in OneHot function: '{tie}'."


        set_indices = tuple(index[selected_idx] for index in extreme_indices)

        result = np.zeros_like(variable)
        result[set_indices] = 1 if indicator else extreme_val

        return self.convert_output_type(result)
