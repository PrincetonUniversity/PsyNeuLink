# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Condition **************************************************************

import collections
import copy
import functools
import inspect
import numbers
import warnings

import dill
import graph_scheduler
import numpy as np

from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.mdf import MDFSerializable
from psyneulink.core.globals.keywords import MODEL_SPEC_ID_TYPE, comparison_operators
from psyneulink.core.globals.parameters import parse_context
from psyneulink.core.globals.utilities import parse_valid_identifier

__all__ = copy.copy(graph_scheduler.condition.__all__)
__all__.extend(['Threshold'])


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


class Condition(graph_scheduler.Condition, MDFSerializable):
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

    def as_mdf_model(self):
        import modeci_mdf.mdf as mdf
        from psyneulink.core.components.component import Component

        def _parse_condition_arg(arg):
            if isinstance(arg, Component):
                return parse_valid_identifier(arg.name)
            elif isinstance(arg, graph_scheduler.Condition):
                return arg.as_mdf_model()
            elif arg is None or isinstance(arg, numbers.Number):
                return arg
            else:
                try:
                    iter(arg)
                except TypeError:
                    return str(arg)
                else:
                    return arg

        if type(self) in {graph_scheduler.Condition, Condition}:
            try:
                func_val = inspect.getsource(self.func)
            except OSError:
                func_val = dill.dumps(self.func)
            func_dict = {'function': func_val}
        else:
            func_dict = {}

        extra_args = {MODEL_SPEC_ID_TYPE: getattr(graph_scheduler.condition, type(self).__name__).__name__}

        sig = inspect.signature(self.__init__)

        for name, param in sig.parameters.items():
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                args_list = []
                for a in self.args:
                    if isinstance(a, Component):
                        a = parse_valid_identifier(a.name)
                    elif isinstance(a, graph_scheduler.Condition):
                        a = a.as_mdf_model()
                    args_list.append(a)
                extra_args[name] = args_list

        for i, (name, param) in enumerate(filter(
            lambda item: item[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and item[0] not in self.kwargs,
            sig.parameters.items()
        )):
            try:
                extra_args[name] = self.args[i]
            except IndexError:
                # was specified with keyword not as positional arg
                extra_args[name] = param.default

        return mdf.Condition(
            **func_dict,
            **{
                k: _parse_condition_arg(v)
                for k, v in (*self.kwargs.items(), *extra_args.items())
            }
        )


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


_doc_subs = {
    None: [
        (
            r'def converge\(node, thresh\):',
            'def converge(node, thresh, context):'
        ),
        (
            r'node\.delta',
            'node.parameters.value.get_delta(context)'
        ),
        (
            r'the ``delta`` attribute of\na node \(which reports the change in its ``value``\)',
            "the change in a node's `value`"
        )
    ]
}


class Threshold(graph_scheduler.condition._DependencyValidation, Condition):
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

    def __init__(self, dependency, parameter, threshold, comparator, indices=None, atol=0, rtol=0):
        if comparator not in comparison_operators:
            raise graph_scheduler.ConditionError(
                f'Operator must be one of {list(comparison_operators.keys())}'
            )

        if parameter not in dependency.parameters:
            raise graph_scheduler.ConditionError(
                f'{dependency} has no {parameter} parameter'
            )

        if (atol != 0 or rtol != 0) and comparator in {'<', '<=', '>', '>='}:
            warnings.warn('Tolerances for inequality comparators are ignored')

        if (
            indices is not None
            and not isinstance(indices, graph_scheduler.TimeScale)
            and not isinstance(indices, collections.abc.Iterable)
        ):
            indices = [indices]

        def func(threshold, comparator, indices, atol, rtol, execution_id):
            param_value = getattr(self.dependency.parameters, self.parameter).get(execution_id)

            if isinstance(indices, graph_scheduler.TimeScale):
                param_value = param_value._get_by_time_scale(indices)
            elif indices is not None:
                for i in indices:
                    param_value = param_value[i]

            param_value = float(param_value)

            if comparator == '==':
                return np.isclose(param_value, threshold, atol=atol, rtol=rtol)
            elif comparator == '!=':
                return not np.isclose(param_value, threshold, atol=atol, rtol=rtol)
            else:
                return comparison_operators[comparator](param_value, threshold)

        super().__init__(
            func,
            dependency=dependency,
            parameter=parameter,
            threshold=threshold,
            comparator=comparator,
            indices=indices,
            atol=atol,
            rtol=rtol,
        )

    def as_mdf_model(self):
        m = super().as_mdf_model()

        if self.parameter == 'value':
            m.kwargs['parameter'] = f'{self.dependency.name}_OutputPort_0'

        return m
