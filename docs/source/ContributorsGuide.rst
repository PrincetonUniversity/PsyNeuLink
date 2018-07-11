Contributors Guide
==================

* `Introduction`
* `File_Structure`
* `Environment_Setup`
* `Components_Overview`
    * `Mechanism_Implementation`
* `Compositions_Overview`
* `Scheduler`
* `Documentation`

.. _Introduction:

Introduction
------------

Thank you for your interest in contributing to PsyNeuLink! This page is written and maintained by contributors to PsyNeuLink, and it compiles helpful info for new contributors, which may not be covered in the user documentation. As you gain experience in coding PsyNeuLink, we encourage you to add your own info to this guide!

.. _File_Structure:

File Structure
--------------

In the PsyNeuLink git repository, there are many folders and files. The following folders and files are the most relevant:

- *docs* contains the documentation files, including the Contributors Guide
..
    * *source* contains the Sphinx files used to generate the HTML documentation
    * *build* contains the generated HTML documentation, which is generated using the Sphinx `html` command
..
- *Scripts* contains sample PsyNeuLink scripts, though these scripts are not actively maintained and may be outdated
..
- *tests* contains the test code, which is actively maintained
..
- *CONVENTIONS.md* describes coding conventions that contributors must follow, such as documentation style and variable naming
..
- *psyneulink* contains the source code of PsyNeuLink

Within *psyneulink*, the *library* folder represents non-core objects, while the other folders hold the core of PsyNeuLink. Thus, custom Mechanisms, Projections, etc., belong in *library*.

.. _Environment_Setup:

Environment Setup
-----------------

PsyNeuLink is coded in Python 3, and should work in Python 3.6. First install Python and pip on your machine, if not there already. Clone the PsyNeuLink git repository. To install the required packages, navigate to the PsyNeuLink folder and run `pip install -r requirements.txt` and `pip install -r dev_requirements.txt`, or use pip3 if needed.

PsyNeuLink uses `pytest <https://docs.pytest.org/en/latest/index.html>` to run its tests. To build documentation, we use Sphinx. *insert Sphinx setup instructions here*

To contribute, create a branch off of the `devel` branch, and make a pull request to `devel` once your changes are complete.

.. _Components_Overview:

Components Overview
-------------------

Most PsyNeuLink objects are `Components <Component>`. All `Functions <Function>`, `Mechanisms <Mechanism>`, `Projections <Projection>`, and `States <State>` are subclasses of Component. These subclasses use and override many functions from the Component class, so they are initialized and executed in similar ways.

The subclasses of Component should override Component's functions to implement their own functionality. However, function overrides must call the overridden function using `super()`, while passing the same arguments. For example, to instantiate a Projection's receiver after instantiating its function, the Projection_Base class overrides the `_instantiate_attributes_after_function` as follows::

    class Projection_Base(Projection):
	def _instantiate_attributes_after_function(self, context=None):
            self._instantiate_receiver(context=context)
            super()._instantiate_attributes_after_function(context=context)

`context` is a string argument passed among PsyNeuLink functions to provide outside information about when a function is being called. Usually, within a component, if you modify `context`, you should append to it rather than overwriting it.

.. _Compositions_Overview:

Compositions Overview
---------------------

.. _Mechanism_Implementation:

Mechanism Implementation
------------------------

.. _Scheduler:

Scheduler
---------

.. _Documentation:

Documentation
-------------