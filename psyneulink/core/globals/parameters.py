"""

.. _Parameter_Attributes:

PsyNeuLink `parameters <Parameter>` are objects that represent the user-modifiable parameters of a `Component`.
`Parameter`\\ s have names, default values, and other attributes that define how they are used in Compositions.
`Parameter` \\s also maintain and provide access to the data used in actual computations - `default values
<Parameter_Defaults>`, `current values <Parameter_Statefulness>`, `previous values <Parameter.history>`, and
`logged values <Log>`.


.. _Parameter_Defaults:

Defaults
========

The Defaults class is used to represent the default values for a `Component's parameters <Component_Parameters>`.
Parameters have two types of defaults: *instance* defaults and *class* defaults. Class defaults belong to a PsyNeuLink
class, and suggest valid types and shapes of Parameter values. Instance defaults belong to an instance of a PsyNeuLink
class, and are used to validate compatibility between this instance and other PsyNeuLink objects. For example, given a
`TransferMechanism` *t*:

    - instance defaults are accessible by ``t.defaults`` (e.g., for the `noise <TransferMechanism.noise>` parameter,
      ``t.defaults.noise.defaults.noise``)

    - class defaults are accessible by ``t.class_defaults`` or ``TransferMechanism.defaults`` (e.g.,
      ``t.class_defaults.noise`` or `TransferMechanism.defaults.noise`)

.. note::
    ``t.defaults.noise`` is shorthand for ``t.parameters.noise.default_value``, and they both refer to the default
    ``noise`` value for *t*

Default values are sometimes also used when the parameters value has not been specified; for example, a Component's
``defaults.variable`` is used as the input to a `Mechanism` if its `execute <Mechanism_Base.execute>` method is called
without any input specified, and similarly it is used for the `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of
a `Composition` which are not specified in the **inputs** argument of its `run <Composition.run>` method.

.. _Parameter_Statefulness:

Statefulness of Parameters
==========================

Parameters can have different values in different `execution contexts <Composition_Execution_Context>` in order to
ensure correctness of and allow access to `simulation <OptimizationControlMechanism_Execution>` calculations. As a
result, to inspect and use the values of a parameter, in general you need to know the execution context in which you
are interested. Much of the time, this execution context is likely to be a Composition:::

        >>> import psyneulink as pnl
        >>> c = pnl.Composition()
        >>> d = pnl.Composition()
        >>> t = pnl.TransferMechanism()
        >>> c.add_node(t)
        >>> d.add_node(t)

        >>> c.run({t: 5})
        [[array([5.])]]
        >>> print(t.value)
        [[5.]]

        >>> d.run({t: 10})
        [[array([10.])]]
        >>> print(t.value)
        [[10.]]

        >>> print(t.parameters.value.get(c))
        [[5.]]
        >>> print(t.parameters.value.get(d))
        [[10.]]


The TransferMechanism in the above snippet has a different `value <Component.value>` for each Composition it is run in.
This holds true for all of its `stateful Parameters <Component_Stateful_Parameters>`, so they can behave differently in
different execution contexts and can be modified by modulated `ModulatorySignal_Modulation`.

.. _Parameter_Dot_Notation:

.. note::
    The "dot notation" version - ``t.value`` - refers to the most recent execution context in which *t* was executed. In
    many cases, you can use this to get or set using the execution context you'd expect. However, in complex situations,
    or  if there is any doubt, it is best to explicitly specify the execution context using the parameter's `set
    <Parameter.set>` method (for a more complete descritpion of the differences between dot notation and the `set
    <Parameter.set>` method, see `BasicsAndPrimer_Parameters`.


.. technical_note::

    Developers must keep in mind state when writing new Components for PsyNeuLink. Any parameters or values that may
    change during a `run <Run_Overview>` must become stateful Parameters, or they are at risk of computational
    errors like those encountered in parallel programming.


Creating Parameters
^^^^^^^^^^^^^^^^^^^

To create new Parameters, reference this example of a new class *B*

::

    class B(A):
        class Parameters(A.Parameters):
            p = 1.0
            q = Parameter()

        def __init__(p=None, q=1.0):
            super(p=p, q=q)


- create an inner class Parameters on the Component, inheriting from the parent Component's Parameters class
- an instance of *B*.Parameters will be assigned to the parameters attribute of the class *B* and all instances of *B*
- each attribute on *B*.Parameters becomes a parameter (instance of the Parameter class)
    - as with *p*, specifying only a value uses default values for the attributes of the Parameter
    - as with *q*, specifying an explicit instance of the Parameter class allows you to modify the
      `Parameter attributes <Parameter_Attributes_Table>`
- default values for the parameters can be specified in the Parameters class body, or in the
  arguments for *B*.__init__. If both are specified and the values differ, an exception will be raised
- if you want assignments to parameter *p* to be validated, add a method _validate_p(value), \
  that returns None if value is a valid assignment, or an error string if value is not a valid assignment
    - NOTE: a validation method for *p* may reference other parameters \
        only if they are listed in *p*'s \
        `dependencies <Parameter.dependencies>`
- if you want all values set to *p* to be parsed beforehand, add a method _parse_p(value) that returns the parsed value
    - for example, convert to a numpy array or float

        ::

            def _parse_p(value):
                return np.asarray(value)

    - NOTE: parsers may not reference other parameters

- setters and getters (used for more advanced behavior than parsing) should both return the final value to return (getter) or set (setter)

    For example, `costs <ControlMechanism.costs>` of `ControlMechanism <ControlMechanism>` has a special
    getter method, which computes the cost on-the-fly:

        ::

            def _modulatory_mechanism_costs_getter(owning_component=None, context=None):
                try:
                    return [c.compute_costs(c.parameters.variable._get(context), context=context) for c in owning_component.control_signals]
                except TypeError:
                    return None

    and `matrix <RecurrentTransferMechanism.matrix>` of `RecurrentTransferMechanism` has a special setter method,
    which updates its `auto <RecurrentTransferMechanism.auto>` and `hetero <RecurrentTransferMechanism.hetero>` parameter values accordingly

        ::

            def _recurrent_transfer_mechanism_matrix_setter(value, owning_component=None, context=None):
                try:
                    value = get_matrix(value, owning_component.input_shapes[0], owning_component.input_shapes[0])
                except AttributeError:
                    pass

                if value is not None:
                    temp_matrix = value.copy()
                    owning_component.parameters.auto._set(np.diag(temp_matrix).copy(), context)
                    np.fill_diagonal(temp_matrix, 0)
                    owning_component.parameters.hetero._set(temp_matrix, context)

                return value

.. note::
    The specification of Parameters is intended to mirror the PNL class hierarchy. So, it is only necessary for each new class to declare
    Parameters that are new, or whose specification has changed from their parent's. Parameters not present in a given class can be inherited
    from parents, but will be overridden if necessary, without affecting the parents.


.. _Parameter_Special_Classes:

.. technical_note::
    Special Parameter Classes
    -------------------------
        `FunctionParameter` and `SharedParameter` are used to provide
        simpler access to some parameters of auxiliary components. They
        can be passed into the constructor of the owner, and then
        automatically passed when constructing the auxiliary component.
        The `values <Parameter.values>` of `SharedParameter`\\ s are
        shared via getter and  with those of their target `Parameter`.

        `SharedParameter`\\ s should only be used when there is a
        guarantee that their target will exist, given a specific
        Component. For example, it is acceptable that
        `TransferMechanism.integration_rate` is a `FunctionParameter`
        for the `rate` parameter of its
        `integrator_function<TransferMechanism.integrator_function>`,
        because all `TransferMechanism`\\ s have an integrator
        function, and all integrator functions have a `rate` parameter.
        It is also acceptable that
        `ControlSignal.intensity_cost_function` is a `FunctionParameter`
        corresponding to its function's
        `intensity_cost_fct <TransferWithCosts>` parameter, because a
        ControlSignal's function is always a `TransferWithCosts` and is
        not user-specifiable.

.. technical_note::
    _user_specified
    ---------------
        Currently, default argument values for __init__ methods of
        Components are all None to allow distinguishing between a
        default value a user actually passed in during construction and
        a default automatically assigned. Consider

            1. `pnl.TransferMechanism(noise=[[0, 0]])`
            2. `pnl.TransferMechanism(default_variable=[[0]], noise=[[0, 0]])`

        In case 1, the default variable is not specified by the user,
        and is `flexible <DefaultsFlexibility.FLEXIBLE>` as a result.
        Since noise indicates a shape, `[[0, 0]]` is implicitly assigned
        as the Mechanism's default variable. This is intended as a
        convenience specification. In case 2, the variable and noise are
        incompatible, and an error is raised.

Using Parameters
^^^^^^^^^^^^^^^^

Methods that are called during runtime in general must take *context* as an argument and must pass this *context* along to other
PNL methods. The most likely place this will come up would be for the *function* method on a PNL `Function` class, or *_execute* method on other
`Components`. Any getting and setting of stateful parameter values must use this *context*, and using standard attributes to store data
must be avoided at risk of causing computation errors. You may use standard attributes only when their values will never change during a
`Run <TimeScale.RUN>`.

You should avoid using `dot notation <Parameter_Dot_Notation>` in internal code, as it is ambiguous and can potentially break statefulness.

.. _Parameter_Attributes_Table:

`Parameter` **attributes**:

.. table:: **`Parameter` attributes**

+------------------+---------------+--------------------------------------------+-----------------------------------------+
|  Attribute Name  | Default value |                Description                 |                Dev notes                |
|                  |               |                                            |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|  default_value   |     None      |the default value of the Parameter          |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|       name       |     None      |the name of the Parameter                   |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|     stateful     |     True      |whether the parameter has different values  |                                         |
|                  |               |based on execution context                  |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|    modulable     |     False     |if True, the parameter can be modulated     |Currently this does not determine what   |
|                  |               |(if it belongs to a Mechanism or Projection |gets a ParameterPort, but in the future  |
|                  |               | it is assigned a `ParameterPort`)          |it should                                |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|    read_only     |     False     |whether the user should be able to set the  |Can be manually set, but will trigger a  |
|                  |               |value or not (e.g. variable and value are   |warning unless override=True             |
|                  |               |just for informational purposes).           |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|     aliases      |     None      |other names by which the parameter goes     |specify as a list of strings             |
|                  |               |(e.g. allocation is the same as variable for|                                         |
|                  |               |ControlSignal).                             |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|       user       |     True      |whether the parameter is something the user |                                         |
|                  |               |will care about (e.g. NOT context)          |                                         |
|                  |               |                                            |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|      values      |     None      |stores the parameter's values under         |                                         |
|                  |               |different execution contexts                |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|      getter      |     None      |hook that allows overriding the retrieval of|kwargs self, owning_component, and       |
|                  |               |values based on a supplied method           |context will be passed in if your        |
|                  |               |(e.g. _output_port_variable_getter)         |method uses them. modulated=True will be |
|                  |               |                                            |passed in when modulated values are      |
|                  |               |                                            |requested.                               |
|                  |               |                                            |self: the Parameter calling the setter   |
|                  |               |                                            |owning_component: the Component to which |
|                  |               |                                            |    the Parameter belongs                |
|                  |               |                                            |context: the context the setter is called|
|                  |               |                                            |    with                                 |
|                  |               |                                            |Getters must return the resulting value  |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|      setter      |     None      |hook that allows overriding the setting of  |should take a positional argument; kwargs|
|                  |               |values based on a supplied method (e.g.     |self, owning_component, and context      |
|                  |               |_recurrent_transfer_mechanism_matrix_setter)|will be passed in if your method uses    |
|                  |               |                                            |them. self - the Parameter calling the   |
|                  |               |                                            |setter; owning_component - the Component |
|                  |               |                                            |to which the Parameter belongs;          |
|                  |               |                                            |context - the context the                |
|                  |               |                                            |setter is called with; should return the |
|                  |               |                                            |value to be set                          |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|     loggable     |     True      |whether the parameter can be logged         |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|       log        |     None      |stores the log of the parameter if          |                                         |
|                  |               |applicable                                  |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|  log_condition   |     `OFF`     |the `LogCondition` for which the parameter  |                                         |
|                  |               |should be logged                            |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|     history      |     None      |stores the history of the parameter         |                                         |
|                  |               |(previous values)                           |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
|history_max_length|       1       |the maximum length of the stored history    |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+
| fallback_default |     False     |if False, the Parameter will return None if |                                         |
|                  |               |a requested value is not present for a given|                                         |
|                  |               |execution context; if True, the Parameter's |                                         |
|                  |               |default_value will be returned instead      |                                         |
+------------------+---------------+--------------------------------------------+-----------------------------------------+



Class Reference
===============

"""

import collections
import copy
import functools
import inspect
import itertools
import logging
import types
import typing
import weakref

import toposort

from psyneulink.core.globals.context import Context, ContextError, ContextFlags, _get_time, handle_external_context
from psyneulink.core.globals.context import time as time_object
from psyneulink.core.globals.keywords import SHARED_COMPONENT_TYPES
from psyneulink.core.globals.log import LogCondition, LogEntry, LogError
from psyneulink.core.globals.utilities import (
    call_with_pruned_args,
    convert_all_elements_to_np_array,
    create_union_set,
    get_alias_property_getter,
    get_alias_property_setter,
    get_deepcopy_with_shared,
    get_function_sig_default_value,
    is_numeric,
    safe_equals,
    try_extract_0d_array_item,
    unproxy_weakproxy,
    update_array_in_place,
)
from psyneulink.core.rpc.graph_pb2 import Entry, ndArray

__all__ = [
    'Defaults', 'get_validator_by_function', 'Parameter', 'ParameterAlias', 'ParameterError',
    'ParametersBase', 'parse_context', 'FunctionParameter', 'SharedParameter'
]

logger = logging.getLogger(__name__)


class ParameterError(Exception):
    pass


def _get_prefixed_method(obj, prefix, name, sep=''):
    try:
        return getattr(obj, f'{prefix}{sep}{name}')
    except AttributeError:
        return None


def get_validator_by_function(function):
    """
        Arguments
        ---------
            function
                a function that takes exactly one positional argument and returns `True` if that argument
                is a valid assignment, or `False` if that argument is not a valid assignment

        :return: A validation method for use with Parameters classes that rejects any assignment for which **function** returns False
        :rtype: types.FunctionType
    """
    def validator(self, value):
        if function(value):
            return None
        else:
            return '{0} returned False'.format(function.__name__)

    return validator


def parse_context(context):
    """
        Arguments
        ---------
            context
                An execution context (context, Composition)

        :return: the context associated with **context**
    """
    try:
        return context.default_execution_id
    except AttributeError:
        return context


def copy_parameter_value(value, shared_types=None, memo=None):
    """
        Returns a copy of **value** used as the value or spec of a
        Parameter, with exceptions.

        For example, we assume that if we have a Component in an
        iterable, it is meant to be a pointer rather than something
        used in computation requiring it to be a "real" instance
        (like `Component.function`)

        e.g. in spec attribute or Parameter `Mechanism.input_ports`
    """
    if memo is None:
        memo = {}

    if SHARED_COMPONENT_TYPES not in memo:
        from psyneulink.core.components.component import Component, ComponentsMeta
        if shared_types is None:
            shared_types = (Component, ComponentsMeta)
        else:
            shared_types = tuple(shared_types)
        memo[SHARED_COMPONENT_TYPES] = shared_types

    # trying to deepcopy a bound method of a Component will deepcopy the
    # Component, but we treat these situations like references.
    # ex: GridSearch.search_function = GridSearch._traverse_grid
    method_owner = getattr(value, '__self__', None)
    if method_owner:
        memo[id(method_owner)] = method_owner

    try:
        return copy.deepcopy(value, memo)
    except TypeError as e:
        if 'pickle' in str(e):
            return value
        else:
            raise


def get_init_signature_default_value(obj, parameter):
    """
        Returns:
            the default value of the **parameter** argument of
            the __init__ method of **obj** if it exists, or inspect._empty
    """
    # only use the signature if it's on the owner class, not a parent
    if '__init__' in obj.__dict__:
        return get_function_sig_default_value(obj.__init__, parameter)
    else:
        return inspect._empty


def check_user_specified(func):
    @functools.wraps(func)
    def check_user_specified_wrapper(self, *args, **kwargs):
        if 'params' in kwargs and kwargs['params'] is not None:
            orig_kwargs = copy.copy(kwargs)
            kwargs = {**kwargs, **kwargs['params']}
            del kwargs['params']
        else:
            orig_kwargs = kwargs

        # find the corresponding constructor in chained wrappers
        constructor = func
        while '__init__' not in constructor.__qualname__:
            constructor = constructor.__wrapped__

        try:
            self._user_specified_args
        except AttributeError:
            self._prev_constructor = constructor if '__init__' in type(self).__dict__ else None
            self._user_specified_args = copy.copy(kwargs)
        else:
            # add args determined in constructor to user_specifed.
            # since some args are set by the values of other
            # user_specified args in a constructor, we label these as
            # user_specified also (ex. LCAMechanism hetero/competition)
            for k, v in kwargs.items():
                # we only know changes in passed parameter values after
                # calling the next __init__ in the hierarchy, so can
                # only check _prev_constructor
                if k not in self._user_specified_args and self._prev_constructor is not None:
                    prev_constructor_default = get_function_sig_default_value(
                        self._prev_constructor, k
                    )
                    if (
                        # arg value passed through constructor is
                        # different than default arg in signature
                        (
                            type(prev_constructor_default) != type(v)
                            or not safe_equals(prev_constructor_default, v)
                        )
                        # arg value is different than the value given
                        # from the previous constructor in the class
                        # hierarchy
                        and (
                            k not in self._prev_kwargs
                            or (
                                type(self._prev_kwargs[k]) != type(v)
                                or not safe_equals(self._prev_kwargs[k], v)
                            )
                        )
                    ):
                        # NOTE: this is a good place to identify
                        # potentially unnecessary/inconsistent default
                        # parameter settings in body of constructors
                        self._user_specified_args[k] = v

            self._prev_constructor = constructor

        self._prev_kwargs = kwargs
        return func(self, *args, **orig_kwargs)

    return check_user_specified_wrapper


def is_array_like(obj: typing.Any) -> bool:
    """
    Returns:
        bool: True if **obj** is a numpy-array-like object. False
        otherwise
    """
    return hasattr(obj, 'dtype')


# used in Parameter._set_value. Parameter names where a change in
# shape/type should cause deletion of corresponding compiled structs
# even if the values are not synced
addl_unsynced_parameter_names = {'value'}


class ParametersTemplate:
    _deepcopy_shared_keys = ['_parent', '_params', '_owner_ref', '_children']
    _values_default_excluded_attrs = {'user': False}

    def __init__(self, owner, parent=None):
        # using weakref to allow garbage collection of unused objects of this type
        self._owner = owner
        self._parent = parent
        if isinstance(self._parent, ParametersTemplate):
            self._parent._children.add(self)

        # create list of params currently existing
        self._params = set()
        try:
            parent_keys = list(self._parent._params)
        except AttributeError:
            parent_keys = dir(type(self))
        source_keys = dir(self) + parent_keys
        for k in source_keys:
            if self._is_parameter(k):
                self._params.add(k)

        self._children = weakref.WeakSet()

    def __repr__(self):
        return '{0} :\n{1}'.format(super().__repr__(), str(self))

    def __str__(self):
        return self.show()

    def __deepcopy__(self, memo):
        newone = get_deepcopy_with_shared(self._deepcopy_shared_keys)(self, memo)

        for name, param in self.values(show_all=True).items():
            if isinstance(param, ParameterAlias):
                source_name = param.source.name
                getattr(newone, name).source = getattr(newone, source_name)

        memo[id(self)] = newone
        return newone

    def __contains__(self, item):
        if item in self._params:
            return True

        for p in self._params:
            try:
                p_attr = getattr(self, p)
            except AttributeError:
                return False
            else:
                if item is p_attr:
                    return True

        return False

    def __iter__(self):
        return iter([getattr(self, k) for k in self.values(show_all=True).keys()])

    def _is_parameter(self, param_name):
        if param_name[0] == '_':
            return False
        elif param_name in self._params:
            return True
        else:
            try:
                return not isinstance(getattr(self, param_name), (types.MethodType, types.BuiltinMethodType))
            except AttributeError:
                return True

    def _register_parameter(self, param_name):
        self._params.add(param_name)
        self._nonexistent_attr_cache.discard(param_name)

        for child in self._children:
            child._register_parameter(param_name)

    def _invalidate_nonexistent_attr_cache(self, attr):
        self._nonexistent_attr_cache.discard(attr)

        for child in self._children:
            child._invalidate_nonexistent_attr_cache(attr)

    def values(self, show_all=False):
        """
            Arguments
            ---------
                show_all : False
                    if `True`, includes non-`user<Parameter.user` parameters

            :return: a dictionary with {parameter name: parameter value} key-value pairs for each Parameter
        """
        result = {}
        for k in self._params:
            val = getattr(self, k)

            if show_all:
                result[k] = val
            else:
                # exclude any values that have an attribute/value pair listed in ParametersTemplate._values_default_excluded_attrs
                for excluded_key, excluded_val in self._values_default_excluded_attrs.items():
                    try:
                        if getattr(val, excluded_key) == excluded_val:
                            break
                    except AttributeError:
                        pass
                else:
                    result[k] = val

        return result

    def show(self, show_all=False):
        vals = self.values(show_all=show_all)
        return '(\n\t{0}\n)'.format('\n\t'.join(sorted(['{0} = {1},'.format(k, vals[k]) for k in vals])))

    def names(self, show_all=False):
        return sorted([p for p in self.values(show_all)])

    @property
    def _owner(self):
        return unproxy_weakproxy(self._owner_ref)

    @_owner.setter
    def _owner(self, value):
        try:
            self._owner_ref = weakref.proxy(value)
        except TypeError:
            self._owner_ref = value

    def _dependency_order_key(self, names=False):
        """
        Args:
            names (bool, optional): Whether sorting key is based on
            Parameter names or Parameter objects. Defaults to False.

        Returns:
            types.FunctionType: a function that may be passed in as sort
            key so that any Parameter is placed before its dependencies
        """
        parameter_function_ordering = list(toposort.toposort({
            p.name: p.dependencies for p in self if p.dependencies is not None
        }))
        parameter_function_ordering = list(
            itertools.chain.from_iterable(parameter_function_ordering)
        )

        if names:
            def ordering(p):
                try:
                    return parameter_function_ordering.index(p)
                except ValueError:
                    return -1
        else:
            def ordering(p):
                try:
                    return parameter_function_ordering.index(p.name)
                except ValueError:
                    return -1

        return ordering

    @property
    def _in_dependency_order(self):
        """
            Returns:
                list[Parameter] - a list of Parameters such that any
                Parameter is placed before all of its
                `dependencies <Parameter.dependencies>`
        """
        return sorted(self, key=self._dependency_order_key())


class Defaults(ParametersTemplate):
    """
        A class to simplify display and management of default values associated with the `Parameter`\\ s
        in a :class:`Parameters` class.

        With an instance of the Defaults class, *defaults*, *defaults.<param_name>* may be used to
        get or set the default value of the associated :class:`Parameters` object

        Attributes
        ----------
            owner
                the :class:`Parameters` object associated with this object
    """
    def __init__(self, owner, **kwargs):
        super().__init__(owner)

        try:
            vals = sorted(self.values(show_all=True).items())
            for k, v in vals:
                try:
                    setattr(self, k, kwargs[k])
                except KeyError:
                    pass
        except AttributeError:
            # this may occur if this ends up being assigned to a "base" parameters object
            # in this case it's not necessary to support kwargs assignment
            pass

    def __getattr__(self, attr):
        return getattr(self._owner.parameters, attr).default_value

    def __setattr__(self, attr, value):
        if (attr[:1] != '_'):
            param = getattr(self._owner.parameters, attr)
            param.default_value = value
        else:
            super().__setattr__(attr, value)

    def values(self, show_all=False):
        """
            Arguments
            ---------
                show_all : False
                    if `True`, includes non-`user<Parameter.user>` parameters

            :return: a dictionary with {parameter name: parameter value} key-value pairs corresponding to `owner`
        """
        return {k: v.default_value for (k, v) in self._owner.parameters.values(show_all=show_all).items()}


class ParameterBase(types.SimpleNamespace):
    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name

    def __eq__(self, other):
        return object.__eq__(self, other)

    def __hash__(self):
        return object.__hash__(self)


class Parameter(ParameterBase):
    """
    COMMENT:
        KDM 11/30/18: using nonstandard formatting below to ensure developer notes is below type in html
    COMMENT

    Attributes
    ----------
        default_value
            the default value of the Parameter.

            :default: None

        name
            the name of the Parameter.

            :default: None

        stateful
            whether the parameter has different values based on execution context.

            :default: True

        modulable
            if True, the parameter can be modulated; if the Parameter belongs to a `Mechanism <Mechanism>` or
            `Projection <Projection>`, it is assigned a `ParameterPort`.

            :default: False

            :Developer Notes: Currently this does not determine what gets a ParameterPort, but in the future it should

        modulation_combination_function
            specifies the function used in Port._get_combined_mod_val() to combine values for the parameter if
            it receives more than one ModulatoryProjections;  must be either the keyword *MULTIPLICATIVE*,
            *PRODUCT*, *ADDITIVE*, *SUM*, or a function that accepts an n dimensional array and retursn an n-1
            dimensional array.  If it is None, the an attempt is made to determine it from the an alias for the
            Parameter's name (i.e., if that is MULTIPLICATIVE_PARAM or ADDITIVE_PARAM);  otherwise the default
            behavior is determined by Port._get_combined_mod_val().

            :default: None

        read_only
            whether the user should be able to set the value or not
            (e.g. variable and value are just for informational purposes).

            :default: False

            :Developer Notes: Can be manually set, but will trigger a warning unless override=True

        function_arg
            TBD

            :default: False

        pnl_internal
            whether the parameter is an idiosyncrasy of PsyNeuLink or it is more intrinsic to the conceptual operation
            of the Component on which it resides

            :default: False

        aliases
            other names by which the parameter goes (e.g. allocation is the same as variable for ControlSignal).

            :type: list
            :default: None

            :Developer Notes: specify as a list of strings

        user
            whether the parameter is something the user will care about (e.g. NOT context).

            :default: True

        values
            stores the parameter's values under different execution contexts.

            :type: dict{execution_id: value}
            :default: None

        getter
            hook that allows overriding the retrieval of values based on a supplied method
            (e.g. _output_port_variable_getter).

            :type: types.FunctionType
            :default: None

            :Developer Notes: kwargs self, owning_component, and context will be passed in if your method uses them. self - the Parameter calling the setter; owning_component - the Component to which the Parameter belongs; context - the context the setter is called with; should return the value

        setter
            hook that allows overriding the setting of values based on a supplied method
            (e.g.  _recurrent_transfer_mechanism_matrix_setter).

            :type: types.FunctionType
            :default: None

            :Developer Notes: should take a positional argument; kwargs self, owning_component, and context will be passed in if your method uses them. self - the Parameter calling the setter; owning_component - the Component to which the Parameter belongs; context - the context the setter is called with; should return the value to be set

        loggable
            whether the parameter can be logged.

            :default: True

        log
            stores the log of the parameter if applicable.

            :type: dict{execution_id: deque([LogEntry])}
            :default: None

        log_condition
            the LogCondition for which the parameter should be logged.

            :type: `LogCondition`
            :default: `OFF <LogCondition.OFF>`

        delivery_condition
            the LogCondition for which the parameter shoud be delivered.

            :type: `LogCondition`
            :default: `OFF <LogCondition.OFF>`

        history
            stores the history of the parameter (previous values). Also see `get_previous`.

            :type: dict{execution_id: deque([LogEntry])}
            :default: None

        history_max_length
            the maximum length of the stored history.

            :default: 1

        history_min_length
            the minimum length of the stored history. generally this does not need to be
            overridden, but is used to indicate if parameter history is necessary to computation.

            :default: 0

        fallback_default
            if False, the Parameter will return None if a requested value is not present for a given execution context;
            if True, the Parameter's default_value will be returned instead.

            :default: False

        retain_old_simulation_data
            if False, the Parameter signals to other PNL objects that any values generated during simulations may be
            deleted after they are no longer needed for computation; if True, the values should be saved for later
            inspection.

            :default: False

        constructor_argument
            if not None, this indicates the argument in the owning Component's
            constructor that this Parameter corresponds to.

            :default: None

        valid_types
            if not None, this contains a tuple of `type`\\ s that are acceptable
            for values of this Parameter

            :default: None

        reference
            if False, the Parameter is not used in computation for its
            owning Component. Instead, it is just meant to store a value
            that may be used to initialize other Components

            :default: False

            :Developer Notes: Parameters with Function values marked as
            reference will not be automatically instantiated in
            _instantiate_parameter_classes or validated for variable
            shape

        dependencies
            if not None, this contains a set of Parameter
            names corresponding to Parameters whose values must be
            instantiated before that of this Parameter

            :default: None

        initializer
            the name of another Parameter that serves as this
            Parameter's `initializer <StatefulFunction.initializers>`

            :default: None

        port
            stores a reference to the ParameterPort that modulates this
            Parameter, if applicable

            :default: None

        specify_none
            if True, a user-specified value of None for this Parameter
            will set the _user_specified flag to True

            :default: False

    """
    # The values of these attributes will never be inherited from parent Parameters
    # KDM 7/12/18: consider inheriting ONLY default_value?
    _uninherited_attrs = {'name', 'values', 'history', 'log'}

    # for user convenience - these attributes will be hidden from the repr
    # display if the function is True based on the value of the attribute
    _hidden_if_unset_attrs = {
        'aliases', 'getter', 'setter', 'constructor_argument', 'spec',
        'modulation_combination_function', 'valid_types', 'initializer'
    }
    _hidden_if_false_attrs = {'read_only', 'modulable', 'fallback_default', 'retain_old_simulation_data'}
    _hidden_when = {
        **{k: lambda self, val: val is None for k in _hidden_if_unset_attrs},
        **{k: lambda self, val: val is False for k in _hidden_if_false_attrs},
        **{k: lambda self, val: self.loggable is False or self.log_condition is LogCondition.OFF for k in ['log', 'log_condition']},
        **{k: lambda self, val: self.modulable is False for k in ['modulation_combination_function']},
    }

    # for user convenience - these "properties" (see note below in _set_history_max_length)
    # will be included as "param attrs" - the attributes of a Parameter that may be of interest to/settable by users
    # To add an additional property-like param attribute, add its name here, and a _set_<param_name> method
    # (see _set_history_max_length)
    _additional_param_attr_properties = {
        'default_value',
        'history_max_length',
        'log_condition',
        'spec',
    }

    def __init__(
        self,
        default_value=None,
        name=None,
        stateful=True,
        modulable=False,
        structural=False,
        modulation_combination_function=None,
        read_only=False,
        function_arg=True,
        pnl_internal=False,
        aliases=None,
        user=True,
        values=None,
        getter=None,
        setter=None,
        loggable=True,
        log=None,
        log_condition=LogCondition.OFF,
        delivery_condition=LogCondition.OFF,
        history=None,
        history_max_length=1,
        history_min_length=0,
        fallback_default=False,
        retain_old_simulation_data=False,
        constructor_argument=None,
        spec=None,
        parse_spec=False,
        valid_types=None,
        reference=False,
        dependencies=None,
        initializer=None,
        port=None,  # if modulated, set to the ParameterPort
        mdf_name=None,
        specify_none=False,
        _owner=None,
        _inherited=False,
        # this stores a reference to the Parameter object that is the
        # closest non-inherited parent. This parent is where the
        # attributes will be taken from
        _inherited_source=None,
        _user_specified=False,
        _scalar_converted=False,
        _tracking_compiled_struct=False,
        **kwargs
    ):
        if isinstance(aliases, str):
            aliases = [aliases]

        if values is None:
            values = {}

        if history is None:
            history = {}

        if loggable and log is None:
            log = {}

        if valid_types is not None:
            if isinstance(valid_types, (list, tuple)):
                valid_types = tuple(valid_types)
            else:
                valid_types = (valid_types, )

        if dependencies is not None:
            dependencies = create_union_set(dependencies)

        super().__init__(
            default_value=default_value,
            name=name,
            stateful=stateful,
            modulable=modulable,
            structural=structural,
            modulation_combination_function=modulation_combination_function,
            read_only=read_only,
            function_arg=function_arg,
            pnl_internal=pnl_internal,
            aliases=aliases,
            user=user,
            values=values,
            getter=getter,
            setter=setter,
            loggable=loggable,
            log=log,
            log_condition=log_condition,
            delivery_condition=delivery_condition,
            history=history,
            history_max_length=history_max_length,
            history_min_length=history_min_length,
            fallback_default=fallback_default,
            retain_old_simulation_data=retain_old_simulation_data,
            constructor_argument=constructor_argument,
            spec=spec,
            parse_spec=parse_spec,
            valid_types=valid_types,
            reference=reference,
            dependencies=dependencies,
            initializer=initializer,
            port=port,
            mdf_name=mdf_name,
            specify_none=specify_none,
            _inherited=_inherited,
            _inherited_source=_inherited_source,
            _user_specified=_user_specified,
            _temp_uninherited=set(),
            _scalar_converted=_scalar_converted,
            _tracking_compiled_struct=_tracking_compiled_struct,
            **kwargs
        )

        self._owner = _owner
        self._param_attrs = [k for k in self.__dict__ if k[0] != '_'] \
            + [k for k in self.__class__.__dict__ if k in self._additional_param_attr_properties]

        self._is_invalid_source = False
        self._inherited_attrs_cache = {}
        self.__inherited = False
        self._inherited = _inherited

    def __repr__(self):
        return '{0} :\n{1}'.format(super(types.SimpleNamespace, self).__repr__(), str(self))

    def __str__(self):
        # modified from types.SimpleNamespace to exclude _-prefixed attrs
        try:
            items = (
                "{}={!r}".format(k, getattr(self, k)) for k in sorted(self._param_attrs)
                if k not in self._hidden_when or not self._hidden_when[k](self, getattr(self, k))
            )

            return "{}(\n\t\t{}\n\t)".format(type(self).__name__, "\n\t\t".join(items))
        except AttributeError:
            return super().__str__()

    def __deepcopy__(self, memo):
        result = type(self)(
            **{
                k: copy_parameter_value(getattr(self, k), memo=memo)
                for k in self._param_attrs
            },
            _owner=self._owner,
            _inherited=self._inherited,
            _user_specified=self._user_specified,
            _scalar_converted=self._scalar_converted,
        )

        # make sure default values are always deepcopied
        if (
            not self._inherited
            and id(self.default_value) in memo
            and memo[id(self.default_value)] is self.default_value
        ):
            del memo[id(self.default_value)]
            result._set_default_value(
                copy_parameter_value(self.default_value, memo), directly=True
            )

        memo[id(self)] = result

        return result

    def __getattr__(self, attr):
        # runs when the object doesn't have an attr attribute itself
        # attempt to get from its parent, which is also a Parameter

        # this is only called when self._inherited is True. We know
        # there must be a source if attr exists at all. So, find it this
        # time and only recompute lazily when the current source becomes
        # inherited itself, or a closer parent becomes uninherited. Both
        # will be indicated by the following conditional
        try:
            inherited_source = self._inherited_source()
        except TypeError:
            inherited_source = None

        if (
            inherited_source is None
            # this condition indicates the cache was invalidated
            # since it was set
            or inherited_source._is_invalid_source
        ):
            next_parent = self._parent
            while next_parent is not None:
                if not next_parent._is_invalid_source:
                    self._inherit_from(next_parent)
                    inherited_source = next_parent
                    break
                next_parent = next_parent._parent

        if inherited_source is None:
            # will fail, use default behavior
            return self.__getattribute__(attr)
        else:
            return inherited_source.__getattribute__(attr)

    def __setattr__(self, attr, value):
        if attr in self._additional_param_attr_properties:
            self._temp_uninherited.add(attr)
            self._inherited = False

            try:
                getattr(self, '_set_{0}'.format(attr))(value)
            except AttributeError:
                super().__setattr__(attr, value)

            self._temp_uninherited.remove(attr)
        else:
            super().__setattr__(attr, value)

    def reset(self):
        """
            Resets *default_value* to the value specified in its `Parameters` class declaration, or
            inherits from parent `Parameters` classes if it is not explicitly specified.
        """
        # check for default in Parameters class
        cls_param_value = inspect._empty
        if self._owner._param_is_specified_in_class(self.name):
            try:
                cls_param_value = self._owner.__class__.__dict__[self.name]
            except KeyError:
                pass
            else:
                try:
                    cls_param_value = cls_param_value.default_value
                except AttributeError:
                    pass

        # check for default in __init__ signature
        value = self._owner._reconcile_value_with_init_default(self.name, cls_param_value)
        if value is not inspect._empty:
            self.default_value = value
            return

        # no default specified, must be inherited or invalid
        if self._parent is not None:
            self._inherited = True
            return
        else:
            raise ParameterError(
                'Parameter {0} cannot be reset, as it does not have a default specification '
                'or a parent. This may occur if it was added dynamically rather than in an'
                'explict Parameters inner class on a Component'
            )

    def _register_alias(self, name):
        if self.aliases is None:
            self.aliases = [name]
        elif name not in self.aliases:
            self.aliases.append(name)

    @property
    def _inherited(self):
        return self.__inherited

    @_inherited.setter
    def _inherited(self, value):
        if value is not self._inherited:
            # invalid if set to inherited
            self._is_invalid_source = value
            self.__inherited = value

            if value:
                self._cache_inherited_attrs()
            else:
                # This is a rare operation, so we can just immediately
                # trickle down sources without performance issues.
                children = [*self._owner._children]
                while len(children) > 0:
                    next_child = children.pop()
                    next_child = getattr(next_child, self.name)

                    if next_child._inherited:
                        next_child._inherit_from(self)
                        children.extend(next_child._owner._children)

                self._restore_inherited_attrs()

    def _inherit_from(self, parent):
        self._inherited_source = weakref.ref(parent)

    def _cache_inherited_attrs(self, exclusions=None):
        if exclusions is None:
            exclusions = set()

        exclusions = self._uninherited_attrs.union(self._temp_uninherited).union(exclusions)

        for attr in self._param_attrs:
            if attr not in exclusions:
                self._inherited_attrs_cache[attr] = getattr(self, attr)
                delattr(self, attr)

    def _restore_inherited_attrs(self, exclusions=None):
        if exclusions is None:
            exclusions = set()

        exclusions = self._uninherited_attrs.union(self._temp_uninherited).union(exclusions)

        for attr in self._param_attrs:
            if (
                attr not in exclusions
                and getattr(self, attr) is getattr(self._parent, attr)
            ):
                super().__setattr__(attr, self._inherited_attrs_cache[attr])

    @property
    def _parent(self):
        try:
            return getattr(self._owner._parent, self.name)
        except AttributeError:
            return None

    def _validate(self, value):
        return self._owner._validate(self.name, value)

    def _parse(self, value, check_scalar=False):
        if is_numeric(value):
            orig_value = value
            value = convert_all_elements_to_np_array(value)
            if check_scalar:
                self._scalar_converted = orig_value is not value and value.ndim == 0

        return self._owner._parse(self.name, value)

    @property
    def _default_getter_kwargs(self):
        # self._owner: the Parameters object it belongs to
        # self._owner._owner: the Component the Parameters object belongs to
        # self._owner._owner.owner: that Component's owner if it exists
        kwargs = {
            'self': self,
            'owning_component': self._owner._owner
        }
        try:
            kwargs['owner'] = self._owner._owner.owner
        except AttributeError:
            pass

        return kwargs

    @property
    def _default_setter_kwargs(self):
        return self._default_getter_kwargs

    @handle_external_context()
    def get(self, context=None, **kwargs):
        """
            Gets the value of this `Parameter` in the context of **context**
            If no context is specified, attributes on the associated `Component` will be used

            Arguments
            ---------

                context : Context, execution_id, Composition
                    the context for which the value is stored; if a Composition, uses **context**.default_execution_id
                kwargs
                    any additional arguments to be passed to this `Parameter`'s `getter` if it exists
        """
        base_val = self._get(context, **kwargs)
        if self._scalar_converted:
            base_val = try_extract_0d_array_item(base_val)
        if is_array_like(base_val):
            base_val = copy_parameter_value(base_val)
        return base_val

    def _get(self, context=None, **kwargs):
        if not self.stateful:
            execution_id = None
        else:
            try:
                execution_id = context.execution_id
            except AttributeError as e:
                raise ParameterError(
                    '_get must pass in a Context object as the context '
                    'argument. To get parameter values using only an '
                    'execution id, use get.'
                ) from e

        if self.getter is not None:
            kwargs = {**self._default_getter_kwargs, **kwargs}
            value = call_with_pruned_args(self.getter, context=context, **kwargs)
            if self.stateful:
                self._set_value(value, execution_id=execution_id, context=context)
            return value
        else:
            try:
                return self.values[execution_id]
            except KeyError:
                logger.info('Parameter \'{0}\' has no value for execution_id {1}'.format(self.name, execution_id))
                if self.fallback_default:
                    return self.default_value
                else:
                    return None

    @handle_external_context()
    def get_previous(
        self,
        context=None,
        index: int = 1,
        range_start: int = None,
        range_end: int = None,
    ):
        """
            Gets the value set before the current value of this
            `Parameter` in the context of **context**. To return a range
            of values, use `range_start` and `range_end`. Range takes
            precedence over `index`. All history values can be accessed
            directly as a list in `Parameter.history`.

            Args:
                context : Context, execution_id, Composition
                    the context for which the value is stored; if a
                    Composition, uses **context**.default_execution_id

                index
                    how far back to look into the history. An index of
                    1 means the value this Parameter had just before
                    its current value. An index of 2 means the value
                    before that

                range_start
                    Inclusive. The index of the oldest history value to
                    return. If ``None``, the range begins at the oldest
                    value stored in history

                range_end
                    Inclusive. The index of the newest history value to
                    return. If ``None``, the range ends at the newest
                    value stored in history (does not include current
                    value in `Parameter.values`)

            Returns:
                the stored value or list of values in Parameter history

        """
        def parse_index(x, arg_name=None):
            try:
                if x < 0:
                    raise ValueError(f'{arg_name} cannot be negative')
                return -x
            except TypeError:
                return x

        # inverted because the values represent "___ from the end of history"
        index = parse_index(index, arg_name='index')
        range_start = parse_index(range_start, arg_name='range_start')

        # override index with ranges
        if range_start == range_end and range_start is not None:
            index = range_start
            range_start = range_end = None

        # fix 0 to "-0" / None
        if range_end == 0:
            range_end = None
        elif range_end is not None:
            # range_end + 1 for inclusive range
            range_end = range_end + 1

        if range_start is not None or range_end is not None:
            try:
                return list(self.history[context.execution_id])[range_start:range_end]
            except (KeyError, IndexError):
                return None
        else:
            try:
                return self.history[context.execution_id][index]
            except (KeyError, IndexError):
                return None

    @handle_external_context()
    def get_delta(self, context=None):
        """
            Gets the difference between the current value and previous value of `Parameter` in the context of **context**

            Arguments
            ---------

                context : Context, execution_id, Composition
                    the context for which the value is stored; if a Composition, uses **context**.default_execution_id
        """
        try:
            return self.get(context) - self.get_previous(context)
        except TypeError as e:
            raise TypeError(
                "Parameter '{0}' value mismatch between current ({1}) and previous ({2}) values".format(
                    self.name,
                    self.get(context),
                    self.get_previous(context)
                )
            ) from e

    @handle_external_context()
    def set(self, value, context=None, override=False, skip_history=False, skip_log=False, **kwargs):
        """
            Sets the value of this `Parameter` in the context of **context**
            If no context is specified, attributes on the associated `Component` will be used

            Arguments
            ---------

                context : Context, execution_id, Composition
                    the context for which the value is stored; if a Composition, uses **context**.default_execution_id
                override : False
                    if True, ignores a warning when attempting to set a *read-only* Parameter
                skip_history : False
                    if True, does not modify the Parameter's *history*
                skip_log : False
                    if True, does not modify the Parameter's *log*
                kwargs
                    any additional arguments to be passed to this `Parameter`'s `setter` if it exists
        """
        from psyneulink.core.components.component import Component

        if not override and self.read_only:
            raise ParameterError('Parameter \'{0}\' is read-only. Set at your own risk. Pass override=True to force set.'.format(self.name))

        value = self._parse(value, check_scalar=True)
        value = self._set(value, context, skip_history, skip_log, **kwargs)

        try:
            if isinstance(value.__self__, Component):
                value = value.__self__
        except AttributeError:
            pass

        if isinstance(value, Component):
            owner = self._owner._owner
            if value not in owner._parameter_components:
                if owner.initialization_status == ContextFlags.INITIALIZED:
                    value._initialize_from_context(context)
                    owner._parameter_components.add(value)

                    try:
                        value._update_default_variable(owner._get_parsed_variable(self, context=context), context)
                    except TypeError as e:
                        if (
                            f'unsupported for {value.__class__.__name__}' not in str(e)
                            and f'unsupported for {owner.__class__.__name__}' not in str(e)
                        ):
                            raise

        return value

    def _set(
        self,
        value,
        context,
        skip_history=False,
        skip_log=False,
        skip_delivery=False,
        compilation_sync=False,
        **kwargs,
    ):
        if not self.stateful:
            execution_id = None
        else:
            try:
                execution_id = context.execution_id
            except AttributeError as e:
                raise ParameterError(
                    '_set must pass in a Context object as the context '
                    'argument. To set parameter values using only an '
                    'execution id, use set.'
                ) from e

        if self.setter is not None:
            kwargs = {
                **self._default_setter_kwargs,
                **kwargs,
                'compilation_sync':compilation_sync,
            }
            value = call_with_pruned_args(self.setter, value, context=context, **kwargs)

        self._set_value(
            value,
            execution_id=execution_id,
            context=context,
            skip_history=skip_history,
            skip_log=skip_log,
            skip_delivery=skip_delivery,
            compilation_sync=compilation_sync,
        )
        return value

    def _set_value(
        self,
        value,
        execution_id=None,
        context=None,
        skip_history=False,
        skip_log=False,
        skip_delivery=False,
        compilation_sync=False,
    ):
        value_is_array_like = is_array_like(value)
        # store history
        if not skip_history:
            if execution_id in self.values:
                value_for_history = self.values[execution_id]
                if value_is_array_like:
                    value_for_history = copy_parameter_value(value_for_history)

                try:
                    self.history[execution_id].append(value_for_history)
                except KeyError:
                    self.history[execution_id] = collections.deque(
                        [value_for_history],
                        maxlen=self.history_max_length,
                    )

        if self.loggable:
            value_for_log = value
            if value_is_array_like:
                value_for_log = copy_parameter_value(value)
            # log value
            if not skip_log:
                self._log_value(value_for_log, context)
            # Deliver value to external application
            if not skip_delivery:
                self._deliver_value(value_for_log, context)

        value_updated = False
        if not compilation_sync:
            try:
                update_array_in_place(self.values[execution_id], value)
            except (KeyError, TypeError, ValueError):
                # no self.values for execution_id
                # failure during attempted update
                pass
            except RuntimeError:
                # torch tensor
                pass
            else:
                value_updated = True

        if not value_updated:
            self.values[execution_id] = value

            if compilation_sync:
                self._tracking_compiled_struct = True
            elif (
                value_is_array_like
                and (
                    self._tracking_compiled_struct
                    or self.name in addl_unsynced_parameter_names
                )
            ):
                # recompilation is needed for arrays that could not be
                # updated in place
                try:
                    owner_comps = self._owner._owner.compositions
                except AttributeError:
                    pass
                else:
                    for comp in owner_comps:
                        comp._delete_compilation_data(context, self)
                self._tracking_compiled_struct = False

    @handle_external_context()
    def delete(self, context=None):
        try:
            del self.values[context.execution_id]
        except KeyError:
            pass

        try:
            del self.history[context.execution_id]
        except KeyError:
            pass

        self.clear_log(context.execution_id)

    def _log_value(self, value, context=None):
        # manual logging
        if context is not None and context.source is ContextFlags.COMMAND_LINE:
            try:
                time = _get_time(self._owner._owner, context)
            except (AttributeError, ContextError):
                time = time_object(None, None, None, None)

            # this branch only ran previously when context was ContextFlags.COMMAND_LINE
            context_str = ContextFlags._get_context_string(ContextFlags.COMMAND_LINE)
            log_condition_satisfied = True

        # standard logging
        else:
            if self.log_condition is None or self.log_condition is LogCondition.OFF:
                return

            if context is None:
                context = self._owner._owner.most_recent_context

            time = _get_time(self._owner._owner, context)
            context_str = ContextFlags._get_context_string(context.flags)
            log_condition_satisfied = self.log_condition & context.flags

        if (
            not log_condition_satisfied
            and self.log_condition & LogCondition.INITIALIZATION
            and self._owner._owner.initialization_status is ContextFlags.INITIALIZING
        ):
            log_condition_satisfied = True

        if log_condition_satisfied:
            if not self.stateful:
                execution_id = None
            else:
                execution_id = context.execution_id

            if execution_id not in self.log:
                self.log[execution_id] = collections.deque([])

            self.log[execution_id].append(
                LogEntry(time, context_str, value)
            )

    def _deliver_value(self, value, context=None):
        # if a context is attached and a pipeline is attached to the context
        if context and context.rpc_pipeline:
            # manual delivery
            if context.source is ContextFlags.COMMAND_LINE:
                try:
                    time = _get_time(self._owner._owner, context)
                except (AttributeError, ContextError):
                    time = time_object(None, None, None, None)
                delivery_condition_satisfied = True

            # standard logging
            else:
                if self.delivery_condition is None or self.delivery_condition is LogCondition.OFF:
                    return

                time = _get_time(self._owner._owner, context)
                delivery_condition_satisfied = self.delivery_condition & context.flags

            if (
                not delivery_condition_satisfied
                and self.delivery_condition & LogCondition.INITIALIZATION
                and self._owner._owner.initialization_status is ContextFlags.INITIALIZING
            ):
                delivery_condition_satisfied = True

            if delivery_condition_satisfied:
                if not self.stateful:
                    execution_id = None
                else:
                    execution_id = context.execution_id
                # ADD TO PIPELINE HERE
                context.rpc_pipeline.put(
                    Entry(
                        componentName=self._get_root_owner().name,
                        parameterName=self._get_root_parameter().name,
                        time=f'{time.run}:{time.trial}:{time.pass_}:{time.time_step}',
                        context=execution_id,
                        value=ndArray(
                            shape=list(value.shape),
                            data=list(value.flatten())
                        )
                    )
                )

    def _get_root_owner(self):
        owner = self
        while True:
            if hasattr(owner, '_owner'):
                owner = owner._owner
            else:
                return owner

    def _get_root_parameter(self):
        root = self._get_root_owner()
        return self._owner._owner if not self._owner._owner == root else self

    def clear_log(self, contexts=NotImplemented):
        """
            Clears the log of this Parameter for every context in **contexts**
        """
        if self.log is None:
            return

        if contexts is NotImplemented:
            self.log.clear()
            return

        if not isinstance(contexts, list):
            contexts = [contexts]

        contexts = [parse_context(c) for c in contexts]
        execution_ids = [
            c.execution_id if hasattr(c, 'execution_id') else c
            for c in contexts
        ]

        try:
            for eid in execution_ids:
                self.log.pop(eid, None)
        except TypeError:
            self.log.pop(execution_ids, None)

    def clear_history(
        self,
        contexts: typing.Union[Context, typing.List[Context]] = NotImplemented
    ):
        """
            Clears the history of this Parameter for every context in
            `contexts`

            Args:
                contexts
        """
        if not isinstance(contexts, list):
            contexts = [contexts]

        contexts = [parse_context(c) for c in contexts]
        execution_ids = [
            c.execution_id if hasattr(c, 'execution_id') else c
            for c in contexts
        ]

        for eid in execution_ids:
            try:
                self.history[eid].clear()
            except KeyError:
                pass

    def _initialize_from_context(self, context=None, base_context=Context(execution_id=None), override=True):
        try:
            try:
                cur_val = self.values[context.execution_id]
            except KeyError:
                cur_val = None

            if cur_val is None or override:
                try:
                    new_val = self.values[base_context.execution_id]
                except KeyError:
                    return

                try:
                    new_history = self.history[base_context.execution_id]
                except KeyError:
                    new_history = NotImplemented

                new_val = copy_parameter_value(new_val)
                self.values[context.execution_id] = new_val

                if new_history is None:
                    raise ParameterError('history should always be a collections.deque if it exists')
                elif new_history is not NotImplemented:
                    # shallow copy is OK because history should not change
                    self.history[context.execution_id] = copy.copy(new_history)

        except ParameterError as e:
            raise ParameterError('Error when attempting to initialize from {0}: {1}'.format(base_context.execution_id, e))

    # KDM 7/30/18: the below is weird like this in order to use this like a property, but also include it
    # in the interface for user simplicity: that is, inheritable (by this Parameter's children or from its parent),
    # visible in a Parameter's repr, and easily settable by the user
    def _set_default_value(self, value, directly=False, check_scalar=False):
        """
        Set default_value

        Args:
            value: new default_value
            directly (bool, optional): if False, passes **value**
                through parse and validation steps. Defaults to False.
            check_scalar (bool, optional): if True, sets
                _scalar_converted attribute as appropriate for
                **value**. Defaults to False.
        """
        if not directly:
            value = self._parse(value, check_scalar=check_scalar)
            self._validate(value)

        super().__setattr__('default_value', value)

    def _set_history_max_length(self, value):
        if value < self.history_min_length:
            raise ParameterError(f'Parameter {self._owner._owner}.{self.name} requires history of length at least {self.history_min_length}.')
        super().__setattr__('history_max_length', value)
        for execution_id in self.history:
            self.history[execution_id] = collections.deque(self.history[execution_id], maxlen=value)

    def _set_log_condition(self, value):
        if not isinstance(value, LogCondition):
            if value is True:
                value = LogCondition.ALL_ASSIGNMENTS
            else:
                try:
                    value = LogCondition.from_string(value)
                except (AttributeError, LogError):
                    try:
                        value = LogCondition(value)
                    except ValueError:
                        # if this fails, value can't be interpreted as a LogCondition
                        raise

        super().__setattr__('log_condition', value)

    def _set_spec(self, value):
        if self.parse_spec:
            value = self._parse(value)
        super().__setattr__('spec', value)

    @property
    def source(self):
        return self

    @property
    def final_source(self):
        return self


class _ParameterAliasMeta(type):
    # these will not be taken from the source
    _unshared_attrs = ['name', 'aliases']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in Parameter().__dict__:
            if k not in self._unshared_attrs:
                setattr(
                    self,
                    k,
                    property(
                        fget=get_alias_property_getter(k, attr='source'),
                        fset=get_alias_property_setter(k, attr='source')
                    )
                )


# TODO: may not completely work with history/history_max_length
class ParameterAlias(ParameterBase, metaclass=_ParameterAliasMeta):
    """
        A counterpart to `Parameter` that represents a pseudo-Parameter alias that
        refers to another `Parameter`, but has a different name
    """
    def __init__(self, source=None, name=None):
        super().__init__(name=name)

        self.source = source

        try:
            source._register_alias(name)
        except AttributeError:
            pass

    def __getattr__(self, attr):
        return getattr(self.source, attr)

    # must override deepcopy despite it being essentially shallow
    # because otherwise it will default to Parameter.__deepcopy__ and
    # return an instance of Parameter
    def __deepcopy__(self, memo):
        result = ParameterAlias(source=self._source, name=self.name)
        memo[id(self)] = result

        return result

    @property
    def source(self):
        return unproxy_weakproxy(self._source)

    @source.setter
    def source(self, value):
        try:
            self._source = weakref.proxy(value)
        except TypeError:
            self._source = value


class SharedParameter(Parameter):
    """
        A Parameter that is not a "true" Parameter of a Component but a
        reference to a Parameter on one of the Component's attributes or
        other Parameters. `Values <Parameter.values>` are shared via
        getter and setter. Mainly used for more user-friendly access to
        certain Parameters, as a sort of cross-object alias.

        .. technical_note::
            See `above <Parameter_Special_Classes>` for when it is
            appropriate to use a SharedParameter

        Attributes:

            shared_parameter_name
                the name of the target Parameter on the owning
                Component's `attribute_name` Parameter or attribute

                :type: str
                :default: `Parameter.name`

            attribute_name
                the name of the owning Component's Parameter or
                attribute on which `shared_parameter_name` is the target
                Parameter of this object

                :type: str
                :default: 'function'

            primary
                whether the default value specified in the
                SharedParameter should take precedence over the default
                value specified in its target

                :type: bool
                :default: False

            getter
                :type: types.FunctionType
                :default: a function that returns the value of the \
                *shared_parameter_name* parameter of the \
                *attribute_name* Parameter/attribute of this \
                Parameter's owning Component

            setter
                :type: types.FunctionType
                :default: a function that sets the value of the \
                *shared_parameter_name* parameter of the \
                *attribute_name* Parameter/attribute of this \
                Parameter's owning Component and returns the set value
    """
    _additional_param_attr_properties = Parameter._additional_param_attr_properties.union({'name'})
    _uninherited_attrs = Parameter._uninherited_attrs.union({'attribute_name', 'shared_parameter_name'})
    # attributes that should not be inherited from source attr
    _unsourced_attrs = {'default_value', 'primary', 'getter', 'setter', 'aliases'}

    def __init__(
        self,
        default_value=None,
        attribute_name=None,
        shared_parameter_name=None,
        primary=False,
        getter=None,
        setter=None,
        **kwargs
    ):

        super().__init__(
            default_value=default_value,
            getter=getter,
            setter=setter,
            attribute_name=attribute_name,
            shared_parameter_name=shared_parameter_name,
            primary=primary,
            _source_exists=False,
            **kwargs
        )

        if getter is None:
            def getter(self, context=None):
                try:
                    return self.source._get(context)
                except (AttributeError, TypeError, IndexError):
                    return None

            self.getter = getter

        if setter is None:
            def setter(value, self, context=None):
                try:
                    return self.source._set(value, context)
                except AttributeError:
                    return None

            self.setter = setter

    def __getattr__(self, attr):
        try:
            if attr in self._unsourced_attrs:
                raise AttributeError
            return getattr(self.source, attr)
        except AttributeError:
            return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        if self._source_exists and attr in self._sourced_attrs:
            setattr(self.source, attr, value)
        else:
            super().__setattr__(attr, value)

    def _cache_inherited_attrs(self):
        super()._cache_inherited_attrs(
            exclusions=self._sourced_attrs
        )

    def _restore_inherited_attrs(self):
        super()._restore_inherited_attrs(
            exclusions=self._sourced_attrs
        )

    def _set_name(self, name):
        if self.shared_parameter_name is None:
            self.shared_parameter_name = name

        super(Parameter, self).__setattr__('name', name)

    @handle_external_context()
    def get_previous(
        self,
        context=None,
        index: int = 1,
        range_start: int = None,
        range_end: int = None,
    ):
        return self.source.get_previous(context, index, range_start, range_end)

    @handle_external_context()
    def get_delta(self, context=None):
        return self.source.get_delta(context)

    @property
    def source(self):
        try:
            obj = getattr(self._owner._owner.parameters, self.attribute_name)
            if obj.stateful:
                raise ParameterError(
                    f'Parameter {type(obj._owner._owner).__name__}.{self.attribute_name}'
                    f' is the target object of {type(self).__name__}'
                    f' {type(self._owner._owner).__name__}.{self.name} and'
                    f' cannot be stateful.'
                )
            obj = obj.values[None]
        except AttributeError:
            try:
                obj = getattr(self._owner._owner, self.attribute_name)
            except AttributeError:
                return None
        except KeyError:
            # KeyError means there is no stored value for this
            # parameter, which should only occur when the source is
            # desired for a descriptive parameter attribute value (e.g.
            # stateful or loggable) and when either self._owner._owner
            # is a type or is in the process of instantiating a
            # Parameter for an instance of a Component
            obj = getattr(self._owner._owner.defaults, self.attribute_name)

        try:
            obj = getattr(obj.parameters, self.shared_parameter_name)
            if not self._source_exists:
                for p in self._param_attrs:
                    if p not in self._uninherited_attrs and p not in self._unsourced_attrs:
                        try:
                            delattr(self, p)
                        except AttributeError:
                            pass
                self._source_exists = True
            return obj
        except AttributeError:
            return None

    @property
    def final_source(self):
        base_param = self
        while isinstance(base_param, SharedParameter):
            base_param = base_param.source

        return base_param

    @property
    def _sourced_attrs(self):
        return set([a for a in self._param_attrs if a not in self._unsourced_attrs])


class FunctionParameter(SharedParameter):
    """
        A special (and most common) case `SharedParameter` that
        references a Parameter on one of the Component's functions.

        Attributes:

            function_parameter_name
                the name of the target Parameter on the owning
                Component's `function_name` Parameter

                :type: str
                :default: `Parameter.name`

            function_name
                the name of the owning Component's Parameter on which
                `function_parameter_name` is the target Parameter of
                this object

                :type: str
                :default: 'function'
    """
    _uninherited_attrs = SharedParameter._uninherited_attrs.union({'function_name', 'function_parameter_name'})

    def __init__(
        self,
        default_value=None,
        function_parameter_name=None,
        function_name='function',
        primary=True,
        **kwargs
    ):
        super().__init__(
            default_value=default_value,
            function_name=function_name,
            function_parameter_name=function_parameter_name,
            primary=primary,
            **kwargs
        )

    @property
    def attribute_name(self):
        return self.function_name

    @attribute_name.setter
    def attribute_name(self, value):
        self.function_name = value

    @property
    def shared_parameter_name(self):
        return self.function_parameter_name

    @shared_parameter_name.setter
    def shared_parameter_name(self, value):
        self.function_parameter_name = value


# KDM 6/29/18: consider assuming that ALL parameters are stateful
#   and that anything that you would want to set as not stateful
#   are actually just "settings" or "preferences", like former prefs,
#   PROJECTION_TYPE, PROJECTION_SENDER
# classifications:
#   stateful = False : Preference
#   read_only = True : "something", computationally relevant but just for information
#   user = False     : not something the user cares about but uses same infrastructure
#
# only current candidate for separation seems to be on stateful
# for now, leave everything together. separate later if necessary
class ParametersBase(ParametersTemplate):
    """
        Base class for inner `Parameters` classes on Components (see `Component.Parameters` for example)
    """
    _parsing_method_prefix = '_parse_'
    _validation_method_prefix = '_validate_'

    def __init__(self, owner, parent=None):
        self._initializing = True
        self._nonexistent_attr_cache = set()

        super().__init__(owner=owner, parent=parent)

        aliases_to_create = {}
        for param_name in copy.copy(self._params):
            try:
                param_value = getattr(self, param_name)
            except AttributeError:
                param_value = NotImplemented

            constructor_default = get_init_signature_default_value(self._owner, param_name)

            if (
                (
                    constructor_default is not None
                    and constructor_default is not inspect._empty
                )
                or self._param_is_specified_in_class(param_name)
            ):
                # KDM 6/25/18: NOTE: this may need special handling if you're creating a ParameterAlias directly
                # in a class's Parameters class
                setattr(self, param_name, param_value)
            else:
                parent_param = getattr(self._parent, param_name)
                if isinstance(parent_param, ParameterAlias):
                    # store aliases we need to create here and then create them later, because
                    # the param that the alias is going to refer to may not have been created yet
                    # (the alias then may refer to the parent Parameter instead of the Parameter associated with this
                    # Parameters class)
                    aliases_to_create[param_name] = parent_param.source.name
                else:
                    new_param = copy.deepcopy(parent_param)
                    new_param._owner = self
                    new_param._inherited = True

                    setattr(self, param_name, new_param)

        for alias_name, source_name in aliases_to_create.items():
            # getattr here and not above, because the alias may be
            # iterated over before the source
            setattr(self, alias_name, ParameterAlias(name=alias_name, source=getattr(self, source_name)))

        values = self.values(show_all=True)
        for param, value in values.items():
            self._validate(param, value.default_value)

            if value.dependencies is not None:
                value.dependencies = {getattr(d, 'name', d) for d in value.dependencies}
                assert all(dep in values for dep in value.dependencies), (
                    f'invalid Parameter name among {param} dependencies: {value.dependencies}'
                )

        self._initializing = False

    def _throw_attr_error(self, attr):
        try:
            param_owner = self._owner
            if isinstance(param_owner, type):
                owner_string = f' of {param_owner}'
            else:
                owner_string = f' of {param_owner.name}'

            if hasattr(param_owner, 'owner') and param_owner.owner:
                owner_string += f' for {param_owner.owner.name}'
                if hasattr(param_owner.owner, 'owner') and param_owner.owner.owner:
                    owner_string += f' of {param_owner.owner.owner.name}'
        except AttributeError:
            owner_string = ''

        raise AttributeError(
            f"No attribute '{attr}' exists in the parameter hierarchy{owner_string}."
        ) from None

    def __getattr__(self, attr):
        if (
            attr in self._params
            or attr in self._nonexistent_attr_cache
            # attr can't be in __dict__ or __getattr__ would not be called
            or (
                self._parent is not None
                and attr in self._parent._nonexistent_attr_cache
            )
        ):
            self._nonexistent_attr_cache.add(attr)
            self._throw_attr_error(attr)

        try:
            return getattr(self._parent, attr)
        except AttributeError:
            self._nonexistent_attr_cache.add(attr)
            self._throw_attr_error(attr)

    def __setattr__(self, attr, value):
        # handles parsing: Parameter or ParameterAlias housekeeping if assigned, or creation of a Parameter
        # if just a value is assigned
        if not self._is_parameter(attr):
            super().__setattr__(attr, value)
        else:
            if isinstance(value, Parameter):
                is_new_parameter = False

                if value._owner is None:
                    value._owner = self
                    is_new_parameter = True
                elif value._owner is not self and self._initializing:
                    # case where no Parameters class defined on subclass
                    # but default value overridden in __init__
                    value = copy.deepcopy(value)
                    value._owner = self

                if value.name is None:
                    value.name = attr

                if self._initializing and not value._inherited:
                    reconciled_value = self._reconcile_value_with_init_default(attr, value.default_value)
                    value._set_default_value(reconciled_value, check_scalar=is_new_parameter)

                super().__setattr__(attr, value)

                if value.aliases is not None:
                    conflicts = []
                    for alias in value.aliases:
                        # there is a conflict if a non-ParameterAlias exists
                        # with the same name as the planned alias
                        if alias in self:
                            try:
                                alias_param = getattr(self, alias)
                            except AttributeError:
                                continue

                            if not isinstance(alias_param, ParameterAlias):
                                conflicts.append(alias)

                        super().__setattr__(alias, ParameterAlias(source=getattr(self, attr), name=alias))
                        self._register_parameter(alias)

                    if len(conflicts) == 1:
                        raise ParameterError(
                            f'Attempting to create an alias for the {value.name}'
                            f' Parameter on {self._owner.__name__} that would'
                            f' override the {conflicts[0]} Parameter. Instead,'
                            f' create a {conflicts[0]} Parameter with alias {value.name}.'
                        )
                    elif len(conflicts) > 1:
                        raise ParameterError(
                            f'Attempting to create aliases for the {value.name}'
                            f' Parameter on {self._owner.__name__} that would'
                            f' override other Parameters: {sorted(conflicts)}'
                        )

            elif isinstance(value, ParameterAlias):
                if value.name is None:
                    value.name = attr
                if isinstance(value.source, str):
                    try:
                        value.source = getattr(self, value.source)
                        value.source._register_alias(attr)
                    except AttributeError:
                        # developer error
                        raise ParameterError(
                            '{0}: Attempted to create an alias named {1} to {2} but attr {2} does not exist'.format(
                                self, attr, value.source
                            )
                        )
                super().__setattr__(attr, value)
            else:
                try:
                    current_value = getattr(self, attr)
                except AttributeError:
                    current_value = None

                if self._initializing:
                    value = self._reconcile_value_with_init_default(attr, value)

                # assign value to default_value
                if isinstance(current_value, (Parameter, ParameterAlias)):
                    # construct a copy because the original may be used as a base for reset()
                    new_param = copy.deepcopy(current_value)
                    # set _inherited before default_value because it will
                    # restore from cache
                    new_param._inherited = False

                    # the old/replaced Parameter should be discarded
                    current_value._is_invalid_source = True

                else:
                    new_param = Parameter(name=attr, _owner=self)

                super().__setattr__(attr, new_param)
                new_param._set_default_value(value)

            self._validate(attr, getattr(self, attr).default_value)
            self._register_parameter(attr)

        if (
            (
                attr[0] != '_'
                or attr.startswith(self._parsing_method_prefix)
                or attr.startswith(self._validation_method_prefix)
            )
            and not self._initializing
        ):
            # below does happen during deepcopy, but that should only
            # happen on instances, which won't have _children
            self._invalidate_nonexistent_attr_cache(attr)

    def _reconcile_value_with_init_default(self, attr, value):
        constructor_default = get_init_signature_default_value(self._owner, attr)
        if constructor_default is not None and constructor_default is not inspect._empty:
            if (
                value is None
                or not self._param_is_specified_in_class(attr)
                or (
                    type(constructor_default) == type(value)
                    and safe_equals(constructor_default, value)
                )
            ):
                # TODO: consider placing a developer-focused warning here?
                return constructor_default
            else:
                assert False, (
                    'PROGRAM ERROR: '
                    f'Conflicting default parameter values assigned for Parameter {attr} of {self._owner} in:'
                    f'\n\t{self._owner}.Parameters: {value}'
                    f'\n\t{self._owner}.__init__: {constructor_default}'
                    f'\nRemove one of these assignments. Prefer removing the default_value of {attr} in {self._owner}.Parameters'
                )

        return value

    def _param_is_specified_in_class(self, param_name):
        return (
            param_name in self.__class__.__dict__
            and (
                param_name not in self._parent.__class__.__dict__
                or self._parent.__class__.__dict__[param_name] is not self.__class__.__dict__[param_name]
            )
        )

    def _get_parse_method(self, parameter):
        """
        Returns:
            the parsing method for the **parameter** or for any Parameter
            attribute (ex: 'modulable') if it exists, or None if it does
            not
        """
        return _get_prefixed_method(self, self._parsing_method_prefix, parameter)

    def _get_validate_method(self, parameter):
        """
        Returns:
            the validation method for the **parameter** or for any
            Parameter attribute (ex: 'modulable') if it exists, or None
            if it does not
        """
        return _get_prefixed_method(self, self._validation_method_prefix, parameter)

    def _validate(self, attr, value):
        err_msg = None

        valid_types = getattr(self, attr).valid_types
        if valid_types is not None:
            if not isinstance(value, valid_types):
                err_msg = '{0} is an invalid type. Valid types are: {1}'.format(
                    type(value),
                    valid_types
                )

        validation_method = self._get_validate_method(attr)
        if validation_method is not None:
            err_msg = validation_method(value)
            # specifically check for False because None indicates a valid assignment
            if err_msg is False:
                err_msg = '{0} returned False'.format(validation_method)

        if err_msg is not None:
            raise ParameterError(
                "Value ({0}) assigned to parameter '{1}' of {2}.parameters is not valid: {3}".format(
                    value,
                    attr,
                    self._owner,
                    err_msg
                )
            )

    def _parse(self, attr, value):
        parse_method = self._get_parse_method(attr)
        if parse_method is not None:
            value = parse_method(value)
        return value
