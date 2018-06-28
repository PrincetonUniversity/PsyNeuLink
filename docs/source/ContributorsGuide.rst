Contributors Guide
==================

* `Introduction`
* `Folders_Overview`
* `Environment_Setup`
* `Components_Overview`
    * `Mechanism_Implementation`
* `Compositions_Overview`
* `Scheduler`
* `Documentation`
* `Publishing`

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
- *CONVENTIONS.md* describes coding conventions that contributors much follow, such as documentation style and variable naming
..
- *psyneulink* contains the source code of PsyNeuLink

Within *psyneulink*, the *library* folder represents non-core objects, while the other folders represent the various core parts of PsyNeuLink. Thus, if you add a custom Mechanism, Projection, etc., add it within the *library* folder.

.. _Environment_Setup:

Environment Setup
-----------------

PsyNeuLink is coded in Python 3, and should work in Python 3.6. First install Python and pip on your machine, if not there already. Clone the PsyNeuLink git repository. To install the required packages, navigate to the PsyNeuLink folder and run `pip install -r requirements.txt` and `pip install -r dev_requirements.txt`, or use pip3 if needed.

PsyNeuLink uses `pytest <https://docs.pytest.org/en/latest/index.html>` to run its tests. To build documentation, we use Sphinx. *insert Sphinx instructions here*

To contribute, create a branch off of the `devel` branch, and make a pull request to `devel` once your changes are complete.

.. _Components_Overview:

Components Overview
-------------------

As described in the documentation, all PsyNeuLink objects are either `Components <Component>` or `Compositions <Composition>`. All `Functions <Function>`, `Mechanisms <Mechanism>`, `Projections <Projection>`, and `States <State>` are subclasses of the Component parent class. The various subclasses of Component share many common functions from the Component class, meaning they are initialized and executed in similar ways.

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

.. _Publishing:

Publishing
----------