Contributors Guide
==================

* `Introduction`
* `File_Structure`
* `Environment_Setup`
* `Contribution_Checklist`
* `Components_Overview`
* `Compositions_Overview`
* `Scheduler`
* `Testing`
* `Documentation`

.. _Introduction:

Introduction
------------

Thank you for your interest in contributing to PsyNeuLink! This page is written and maintained by contributors to
PsyNeuLink. It provides helpful information for new contributors that complements the user documentation.

.. _File_Structure:

File Structure
--------------

In the PsyNeuLink repo, there are many files. The following folders and files are the most relevant:

- *docs*:  directory that contains the documentation files, including this Contributors Guide

  * *source*: directory that contains the Sphinx files used to generate the HTML documentation
  * *build*: directory that contains the generated HTML documentation, which is generated using Sphinx commands

- *Scripts*:  directory that contains sample PsyNeuLink scripts. Not all of these scripts are actively maintained, and
  some may be outdated

- *tests*: directory that contains test code used by pytests, and is actively maintained

- *CONVENTIONS.md*: file that describes coding conventions that contributors must follow, such as documentation style
  and variable naming

- *psyneulink*: directory that contains the source code for PsyNeuLink

  * *core*: directory that contains the core objects of psyneulink
  * *library*: directory that contains user-contributed extensions to psyneulink and other non-core objects

.. _Environment_Setup:

Environment Setup and Installaion
---------------------------------

PsyNeuLink currently supports Python 3.6+, and we aim to support all future releases of Python.
First install Python and pip on your machine, if not installed already.
We suggest `anaconda <https://www.anaconda.com/>`_ or `pyenv <https://github.com/pyenv/pyenv>`_.
Next, clone the PsyNeuLink git repository.
Finally, navigate to the PsyNeuLink folder and install development dependencies::

    pip install -e .[dev]

If necessary, use `pip3` instead of `pip`.

PsyNeuLink uses `pytest <https://docs.pytest.org/en/latest/index.html>`_ to run its tests.
To build documentation, we use `Sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_.
To contribute, make a branch off of the ``devel`` branch.
Make a pull request to ``devel`` once your changes are complete.
``devel`` is periodically merged into the ``master`` branch, which is the branch most users use and is installed with
pip install.

.. _Contribution_Checklist:

Contribution Checklist
----------------------

This is the general workflow for contributing to PsyNeuLink:

* Using git, create a branch off of the ``devel`` branch.
* Make your changes to the code. Ideally, notify the PsyNeuLink team in advance of what you intend to do, so that
  they can provide you with relevant tips in advance.

  * While writing code on your branch, be sure to keep pulling from `devel` from time to time! Since PsyNeuLink is
    actively being developed, substantial changes may have been made to the code base on ``devel`` while you were
    working on your branch;  getting too far behind these may make it difficult for you to merge your branch when you
    are ready.
  * Be sure to write documentation for your new classes or functions, in the style of other PsyNeuLink classes.

* Once you've completed the changes and/or additions on your branch, add tests that check that these
  works as expected. This helps ensure that other developers don't accidentally break your code when making their own
  changes!
* Once your changes are complete and working, run the `pytest <https://docs.pytest.org/en/latest/index.html>`_ tests
  and make sure all tests pass. If you encounter unexpected test failures, please notify the PsyNeuLink team.
* Once all tests pass, submit a pull request to the PsyNeuLink devel branch! The PsyNeuLink team will then review your
  changes and accept the pull request if they sastify the requirements described above.

.. _Components_Overview:

Components Overview
-------------------

Most PsyNeuLink objects are `Components <Component>`. All `Functions <Function>`, `Mechanisms <Mechanism>`,
`Projections <Projection>`, and `Ports <Port>` are subclasses of Component. These subclasses use and override many
functions from the Component class, so they are initialized and executed in similar ways.

The subclasses of Component should override Component's functions to implement their own functionality.
However, function overrides must call the overridden function using `super()`, while passing the same arguments.
For example, to instantiate a Projection's receiver after instantiating its function,
the Projection_Base class overrides the `_instantiate_attributes_after_function` as follows::

    class Projection_Base(Projection):
        def _instantiate_attributes_after_function(self, context=None):
            self._instantiate_receiver(context=context)
            super()._instantiate_attributes_after_function(context=context)

If you wish to modify the behavior of a Component in PsyNeuLink, it is unlikely you will need to create an entirely
new Component (e.g., Mechanism, Projection, or Port) to do so.  Usually this can be accomplished by assigning it a
custom function, either by assigning it an instance of a `UserDefinedFunction` (in the case of simple computations),
or by creating a new subclass of `Function` (for more complex computations).  A new subclass of `Component` should be
created only if it requires a significant deviation from the usual execution pattern.  If this is the case, be sure to
file an issue in the repo outlining this need and your plan for addressing, so that members of the team can advise
you if there is an easier way of meeting your need.

Parameters
^^^^^^^^^^

Any parameters necessary for your Component should be created as `Parameter`\ s rather than simple python attributes. This ensures their values will be threadsafe and correct in all of PsyNeuLink, and they will be exposed as key parameters of your Component. See the `developer documentation for Parameters <Parameter_Developers>` for more information.

Context
^^^^^^^

You must be aware of `Context`, or your Component is likely to crash or produce incorrect results. A `Context` object stores information about the current state of execution and must be passed through most PsyNeuLink methods and functions. `Parameter` values must always be set and retrieved using a `Context` object (see `here <Parameter_Use>` for more information)

Contexts are typically generated within `Composition.run`. When using non-default contexts outside of Compositions, `_initialize_from_context` must be called manually. The below code will fail, because `m` has no parameter values for 'some custom context'.
::

    m = pnl.ProcessingMechanism()
    m.execute(1, context='some custom context')

To fix this, 'some custom context' must be initialized beforehand
::

    m._initialize_from_context(context=Context(execution_id='some custom context'))


.. _Component_Initialization:

Initialization
^^^^^^^^^^^^^^

Constructors should include explicit arguments for each of the new Parameters the class introduces or those that need preprocessing in the constructor. Any others may be passed through the `__init__` hierarchy through `**kwargs`. Additional parameter defaults for a Component's function may be passed in a dictionary in the `function_params` argument. Default/initial values for all these parameters should be set in the `Parameters` class, instead of the python standard default argument value, which should be set to `None`. This is to ensure that the `_user_specified <Parameter._user_specified>` attribute is set correctly, which is used to indicate whether the value for a Parameter was explicitly given by the user or a default was assigned.

Broadly, the sequence of events for Component initialization are as follows:

#. Call `__init__` methods in hierarchic order
#. Set Parameter default values based on input and `class defaults <Component.class_defaults>` (`_initialize_parameters`)
#. Set default `variable` based on input (`default_variable` and other Parameters) and class defaults (`_handle_default_variable`)
#. Call `_instantiate_attributes_before_function` hook
#. Construct, copy, or assign function (`_instantiate_function`)
#. Execute once to produce a default `value` (`_instantiate_value`)
#. Call `_instantiate_attributes_after_function` hook


Execution
^^^^^^^^^

Components (excluding Compositions) run the following steps during `execution <Component_Execution>`.

#. Call `_parse_function_variable` on the input `variable`
#. Call `function <Component.function>` on the result of 1.

Mechanisms add a few extra steps:

#. If no variable is passed in, call `_update_input_ports` and use the values of the `input_ports` as `variable`
#. Call `_update_parameter_ports`
#. Call `_parse_function_variable` on the input `variable`
#. Call `function <Component.function>` on the result of 3.
#. Call `_update_output_ports`
#. If `execute_until_finished` is `True`, repeat steps 1-5 until one of the following:

   a. `is_finished <Component.is_finished>` returns `True`
   b. `num_executions_before_finished` is greater than or equal to `max_executions_before_finished`

.. _Compositions_Overview:

Compositions Overview
---------------------

Execution
^^^^^^^^^

Composition execution is handled by `run <Composition.run>`, `execute <Composition.execute>` as a helper to `run`, and `evaluate <Composition.evaluate>` for simulations.

**Extensive summary of function calls here?**

.. _Scheduler:

Scheduler
---------

`Scheduler` extension is most likely to be done by adding `Condition`\ s. `Condition`\ s that require no stored state can be created ad-hoc, using just an instance of `Condition <psyneulink.core.scheduling.condition.Condition>`, `While`, or `WhileNot`. If your Condition requires stored state, then to implement a subclass you should create a function that returns `True` if the condition is satisfied, and `False` otherwise, and assign it to the `func <Condition.func>` attribute. Any `args` and `**kwargs` passed in to `Condition.__init__ <psyneulink.core.scheduling.condition.Condition>` will be given, unchanged, to each call of `func <Condition.func>`, along with an `execution_id`.

.. note::

    Your stored state must be independent for each ``context``/``execution_id``

.. _Testing:

Testing
-------

PsyNeuLink uses pytest and a test suite in the ``tests`` directory. When contributing, you should include tests with your submission. You may find it helpful to create tests for your contribution before writing it, to help you achieve your desired behavior. Code and documentation style is enforced by the python modules ``pytest-pycodestyle`` and ``pytest-pydocstyle``.

To run all the tests that must pass for your contribution to be accepted, simply run ``pytest`` in the `PsyNeuLink` directory.

.. _Documentation:

Documentation
-------------

Documentation is done through the Sphinx library. Documentation for the `master` and `devel` branches can be found `here <https://princetonuniversity.github.io/PsyNeuLink/>`_ and `here <https://princetonuniversity.github.io/PsyNeuLink/branch/devel/index.html>`_, respectively. When learning about PsyNeuLink, generating the Sphinx documentation is unnecessary because the online documentation exists.

To understand Sphinx syntax, start `here <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ .

However, when editing documentation, you should generate Sphinx documentation in order to preview your changes before publishing to `devel`. To generate Sphinx documentation from your local branch, run `make html` in Terminal, while in the `docs` folder. The resulting HTML should be in your `docs/build` folder. (Do not commit these built HTML files to Github. They are simply for testing/preview purposes.)

Example
-------

Here, we will create a custom Function, ``RandomIntegrator`` that uses stored state and randomness.

1. Inherit from a relevant PsyNeuLink Component. Use `IntegratorFunction` so that we have access to the `previous_value` and `rate` Parameters.
::

    class RandomIntegrator(IntegratorFunction):

2. Create a nested `Parameters` class with values we will need.
::

        class Parameters(IntegratorFunction.Parameters):

            random_state = Parameter(None, pnl_internal=True)
            previous_value_2 = Parameter(np.array([1000]), pnl_internal=True)

`random_state` is used to generate random numbers statefully and independently. `previous_value_2` will be used in our function, and has its default value set arbitrarily to 10 to distinguish it from `previous_value`, which is created on `IntegratorFunction.Parameters` and so does not need to be overridden here. We set the attribute `pnl_internal` to ``True`` on each of these Parameters for use with the `JSON/OpenNeuro collaboration <json>`, indicating that they are not going to be relevant to modeling platforms other than PsyNeuLink.

3. Create an `__init__` method.
::

        def __init__(
            self,
            seed=None,
            previous_value_2=None,
            **kwargs
        ):
            if seed is None:
                seed = get_global_seed()

            super().__init__(
                previous_value_2=previous_value_2,
                random_state=np.random.RandomState([seed]),
                **kwargs
            )

Note that the default value for ``previous_value_2`` is ``None``, `see above <Component_Initialization>`. Any other Parameters will be handled through `**kwargs`.

4. Write a `_function` method (`function <Function.function>` is implemented as a generic wrapper around other Function classes' `_function` methods.)
::

        def _function(
            self,
            variable=None,  # the main input
            context=None,
            params=None,    # future use, runtime_params
        ):
            rate = self.get_current_function_param('rate', context)
            if self.parameters.random_state._get(context).choice([1, 2]) == 1:
                new_value = self.parameters.previous_value._get(context) + rate * variable
                self.parameters.previous_value._set(new_value, context)
            else:
                new_value = self.parameters.previous_value_2._get(context) + rate * variable
                self.parameters.previous_value_2._set(new_value, context)

            return self.convert_output_type(new_value)

`RandomIntegrator` chooses one of its previous values, adds the product of `rate` and `variable` to it, returns the result, and stores that result back into the appropriate previous value.

We use `get_current_function_param` instead of a basic `_get` for rate, because it is a `modulable Parameter <Parameter.modulable>`, meaning it has an associated `ParameterPort` on its owning Mechanism (if it exists). This ensures that the modulated value for rate is returned, if applicable (otherwise, the base value is used, which is equivalent to `_get`. `previous_value` and `previous_value_2` are not modulable, so we can simply use `_get` directly.

We run `convert_output_type` before returning as a general pattern on Functions with simple output. See `Function_Output_Type_Conversion`.

Below is the full class, ready to be included in PsyNeuLink.

::

    import numpy as np
    from psyneulink import IntegratorFunction, Parameter
    from psyneulink.core.globals.utilities import get_global_seed


    class RandomIntegrator(IntegratorFunction):

        class Parameters(IntegratorFunction.Parameters):

            random_state = Parameter(None, pnl_internal=True)
            previous_value_2 = Parameter(np.array([1000]), pnl_internal=True)

        def __init__(
            self,
            seed=None,
            previous_value_2=None,
            **kwargs
        ):
            if seed is None:
                seed = get_global_seed()

            super().__init__(
                previous_value_2=previous_value_2,
                random_state=np.random.RandomState([seed]),
                **kwargs
            )

        def _function(
            self,
            variable=None,  # the main input
            context=None,
            params=None,    # future use, runtime_params
        ):
            rate = self.get_current_function_param('rate', context)
            if self.parameters.random_state._get(context).choice([1, 2]) == 1:
                new_value = self.parameters.previous_value._get(context) + rate * variable
                self.parameters.previous_value._set(new_value, context)
            else:
                new_value = self.parameters.previous_value_2._get(context) + rate * variable
                self.parameters.previous_value_2._set(new_value, context)

            return self.convert_output_type(new_value)
