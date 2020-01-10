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

Thank you for your interest in contributing to PsyNeuLink! This page is written and maintained by contributors to PsyNeuLink. It compiles helpful info for new contributors, which may not be covered in the user documentation.

.. _File_Structure:

File Structure
--------------

In the PsyNeuLink repo, there are many files. The following folders and files are the most relevant:

- *docs* contains the documentation files, including the Contributors Guide

    * *source* contains the Sphinx files used to generate the HTML documentation
    * *build* contains the generated HTML documentation, which is generated using Sphinx commands

..
- *Scripts* contains sample PsyNeuLink scripts. These scripts are not actively maintained and may be outdated
..
- *tests* contains the test code, which is actively maintained
..
- *CONVENTIONS.md* describes coding conventions that contributors must follow, such as documentation style and variable naming
..
- *psyneulink* contains the source code of PsyNeuLink

Within *psyneulink*, the *library* folder represents non-core objects, while the other folders hold the core of PsyNeuLink. Thus, custom Mechanisms, Projections, etc. belong in *library*.

.. _Environment_Setup:

Environment Setup
-----------------

PsyNeuLink is coded in Python 3, and should work in Python 3.6. First install Python and pip on your machine, if not installed already. Clone the PsyNeuLink git repository. To install the required packages, navigate to the PsyNeuLink folder and run `pip install -r requirements.txt` and `pip install -r dev_requirements.txt`. If necessary, use `pip3` instead of `pip`.

PsyNeuLink uses `pytest <https://docs.pytest.org/en/latest/index.html>` to run its tests. To build documentation, we use Sphinx. **insert Sphinx setup instructions here**

To contribute, make a branch off of the `devel` branch. Make a pull request to `devel` once your changes are complete. `devel` is periodically merged into the `master` branch, which is the branch most users use.

.. _Contribution_Checklist:

Contribution Checklist
----------------------

This is the general workflow for contributing to PsyNeuLink:

* Using git, create a branch off of the `devel` branch.
* Make your changes to the code. Ideally, notify the PsyNeuLink team in advance of what you intend to do, so that they can provide you with relevant tips in advance.

    * While writing code on your branch, be sure to keep pulling from `devel` from time to time! Since PsyNeuLink is being developed rapidly, substantial changes are still being made to the code.

* Once you've added your changes, add tests that check that your feature or bugfix functions as expected. This helps ensure that other developers don't accidentally break your code when making their own changes!
* Once your changes are complete and working, run the Pytest tests and make sure all tests pass. If you encounter unexpected test failures, please notify the PsyNeuLink team.
* Once tests pass, submit a pull request to the PsyNeuLink devel branch! The PsyNeuLink team will then review your changes.

.. _Components_Overview:

Components Overview
-------------------

Most PsyNeuLink objects are `Components <Component>`. All `Functions <Function>`, `Mechanisms <Mechanism>`, `Projections <Projection>`, and `Ports <Port>` are subclasses of Component. These subclasses use and override many functions from the Component class, so they are initialized and executed in similar ways.

The subclasses of Component should override Component's functions to implement their own functionality. However, function overrides must call the overridden function using `super()`, while passing the same arguments. For example, to instantiate a Projection's receiver after instantiating its function, the Projection_Base class overrides the `_instantiate_attributes_after_function` as follows::

    class Projection_Base(Projection):
	def _instantiate_attributes_after_function(self, context=None):
            self._instantiate_receiver(context=context)
            super()._instantiate_attributes_after_function(context=context)

`context` is a string argument passed among PsyNeuLink functions to provide outside information about when a function is being called. Usually, if you modify `context`, you should append to it rather than overwriting it.

.. _Compositions_Overview:

Compositions Overview
---------------------

.. _Scheduler:

Scheduler
---------

.. _Testing:

Testing
-------

.. _Documentation:

Documentation
-------------

Documentation is done through the Sphinx library. Documentation for the `master` and `devel` branches can be found `here <https://princetonuniversity.github.io/PsyNeuLink/>`_ and `here <https://princetonuniversity.github.io/PsyNeuLink/branch/devel/index.html>`_, respectively. When learning about PsyNeuLink, generating the Sphinx documentation is unnecessary because the online documentation exists.

To understand Sphinx syntax, start `here <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ .

However, when editing documentation, you should generate Sphinx documentation in order to preview your changes before publishing to `devel`. To generate Sphinx documentation from your local branch, run `make html` in Terminal, while in the `docs` folder. The resulting HTML should be in your `docs/build` folder. (Do not commit these built HTML files to Github. They are simply for testing/preview purposes.)