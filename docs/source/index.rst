.. PsyNeuLink documentation master file, created by
   sphinx-quickstart on Wed Oct 19 11:51:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:tocdepth: 5

.. |logo| image:: _static/PsyNeuLink_logo_no_text.svg
    :width: 20%
    :target: http://psyneulink.org

Welcome to PsyNeuLink |logo|
============================

* `Purpose`
* `What PsyNeuLink IS <What_PsyNeuLink_IS>`
* `What PsyNeuLink is NOT <What_PsyNeuLink_is_NOT>`
* `Overview`
* `Installation`
* `Tutorial`
* `Help and Issues <Help_and_Issues>`
* `Contributors`
* `Indices_and_Tables`


.. _Purpose:

Purpose
-------

PsyNeuLink is an open-source, software environment written in Python, and designed for the needs of
neuroscientists, psychologists, computational psychiatrists and others interested in learning about and building
models of the relationship between brain function, mental processes and behavior.

PsyNeuLink can be used as a "block modeling environment", in which to construct, simulate, document, and exchange
computational models of neural mechanisms and/or psychological processes at the subsystem and system levels.
A block modeling environment allows components to be constructed that implement various, possibly disparate
functions, and then link them together into a system to examine how they interact.  In PsyNeuLink, components are
used to implement the function of brain subsystems and/or psychological processes, the interaction of which can then
be simulated at the system level.

The purpose of PsyNeuLink is to make it as easy as possible to create new and/or import existing models, and
integrate them to simluate system-level interactions.  It provides a suite of core components for
implementing models of various forms of processing, learning, and control, and its Library includes examples that
combine these components to implement published models.  As an open source project, its suite of components is meant
to be enhanced and extended, and its library is meant to provide an expanding repository of models, written in a
concise, executable, and easy to interpret form, that can be shared, compared, and extended by the scientific community.


.. _What_PsyNeuLink_IS:

What PsyNeuLink **IS**
----------------------

It is:

 - *open source*, freeing users of the costs or restrictions associated with proprietary software.

 ..

 - *computationally general* -- it can be used to implement, seamlessly integrate, and simulate interactions among
   disparate components that vary in their granularity of representation and function (from individual neurons or
   neural populations to functional subsystems and abstract cognitive functions) and at any time scale of execution.

 ..

 - *integrative* -- it provides a standard and accessible environment for model comparison, sharing, and documentation;

 ..

 - *extensible* -- it has an interface (API) that allows it to be used with other powerful tools for implementing
   individual components, such as:

    * Neuron and Nengo (biophysically realistic models of neuronal function)
    * Emergent (broad class of neurally-plausible connectionist models);
    * Pytorch and TensorFlow (ODE's, deep learning);
    * ACT-R (symbolic, production system models).

.. note::
   PsyNeuLink is alpha software, that is still being actively developed.  Although it is useable, and most of the
   documented functionality is available, some features may not yet be fully implemented and/or subject to
   modification.  Please report any bugs and/or suggestions for development to psyneulinkhelp@princeton.edu.

.. _What_PsyNeuLink_is_NOT:

What PsyNeuLink is **NOT**
--------------------------

The longterm goal of PsyNeuLink is to provide an environment that integrates comptutational modeling of brain function
and behavior at all levels of analysis.  While it is designed to be fully general, and can in principle be used to
implement models at any level, it is still under development, and current efficiency considerations make it more
suitable for some of forms of modeling than others. In its present form, it is well suited to the creation of
simple to moderately complex models, and for the integration of disparate models into a single environment, while in
it is presently less well suited to efforts involving massively large computations, such as:

 - extensive model fitting
 - large scale simulations
 - highly detailed biophysical models of neurons or neuronal populations

Other packages currently better suited to such applications are:
`Emergent <https://grey.colorado.edu/emergent/index.php/Main_Page>`_ for biologically-inspired neural network models
`Pytorch <https://pytorch.org>`_ and `TensorFlow <https://www.tensorflow.org>`_ (for deep learning models);
`HDDM <http://ski.clps.brown.edu/hddm_docs/>`_ (for Drift Diffusion Models);
`ACT-R <http://act-r.psy.cmu.edu>`_ (for production system models);
`Genesis <http://www.genesis-sim.org>`_,
`Neuron <https://www.neuron.yale.edu/neuron/>`_,
and `Nengo <http://www.nengo.ca>`_  (for biophysically-realistic models of neuronal function).

These packages are good for elaborate and detailed models of a particular form. In contrast, the focus in designing
PsyNeuLink has been to make it as flexible and easy to use as possible, with the ability to integrate components
constructed in other packages (including some of the ones listed above) into a single environment.  These are
characteristics that are often (at least in the initial stages of development) in tension with efficiency (think:
interpreted vs. compiled).

That said, priorities for ongoing development of PsyNeuLink are:
    i) acceleration, using just-in-time compilation methods and parallelization;
    ii) enhancement of the API to facilitate wrapping modules from other packages for integration into the PsyNeuLink
        environment (examples currently exist for Pytorch and Emergent);
    iii) integration of tools for parameter estimation, model comparison and data fitting; and
    iv) a graphic interface for the construction of models and realtime display of their execution.

.. _Overview:

Environment Overview
--------------------

PsyNeuLink is written in Python, and conforms to the syntax, coding standards and modular organization shared by
most Python packages.  `BasicsAndPrimer` provides an orientation to PsyNeuLink's Components, some examples of what
PsyNeuLink models look like, and some of its capabilities. `QuickReference` provides an overview of how PsyNeuLink is
organized and some of its basic principles of operation.  The `Tutorial <Tutorial>` provides an interactive guide to the
construction of models using PsyNeuLink.  `Core` contains the fundamental objects used to build PsyNeuLink models, and
`Library` contains extensions, including speciality components, implemented compositions, and published models.

.. _Installation:

Installation
------------

PsyNeuLink is compatible with python versions >= 3.5, and is available through `PyPI <https://pypi.python.org/pypi/PsyNeuLink>`__:

::

    pip install psyneulink

All prerequisite packages will be automatically added to your environment.

If you downloaded the source code, navigate to the cloned directory in a terminal,
switch to your preferred python3 environment, then run

::

    pip install .

Dependencies that are automatically installed (except those noted as optional) include:

   * numpy
   * matplotlib
   * toposort
   * typecheck-decorator (version 1.2)
   * pillow
   * llvmlite
   * mpi4py (optional)
   * autograd (optional)

Lists of required packages for PsyNeuLink, developing PsyNeuLink, and running the PsyNeuLink tutorial are also
stored in pip-style `requirements.txt`, `dev_requirements.txt`, and `tutorial_requirements.txt` in the source code.


PsyNeuLink is an open source project maintined on `GitHub <https://github.com>`_. The repo can be cloned
from `here <https://github.com/PrincetonUniversity/PsyNeuLink>`_.

If you have trouble installing the package, or run into other problems, please contact psyneulinkhelp@princeton.edu.


.. _Tutorial:


Tutorial
--------

:download:`Download PsyNeuLink Tutorial.ipynb <../../tutorial/PsyNeuLink Tutorial.ipynb>`

PsyNeuLink includes a :download:`tutorial <../../tutorial/PsyNeuLink Tutorial.ipynb>`, that provides examples of how to create basic Components
in PsyNeuLink, and combine them into Processes and a System.  The examples include construction of a simple
decision making process using a Drift Diffusion Model, a neural network model of the Stroop effect, and a
backpropagation network for learning the XOR problem.

The tutorial can be run in a browser by clicking the badge below, or `this link <https://mybinder.org/v2/gh/PrincetonUniversity/PsyNeuLink/master>`__.

.. image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/PrincetonUniversity/PsyNeuLink/master

To run the tutorial locally, you must run python 3.5 and install additional packages:

::

    pip install psyneulink[tutorial]

or if you downloaded the source:

::

    pip install .[tutorial]


To access the tutorial, make sure you fulfill the requirements
mentioned above, download the :download:`tutorial notebook <../../tutorial/PsyNeuLink Tutorial.ipynb>`,
then run the terminal command

::

    jupyter notebook


Once the notebook opens in your browser, navigate to the location where you saved the tutorial notebook, and
click on "PsyNeuLink Tutorial.ipynb".


.. _Help_and_Issues:

Help and Issue Reporting
------------------------

Help is available at psyneulinkhelp@princeton.edu.

Issues can be reported at https://github.com/PrincetonUniversity/PsyNeuLink/issues.


.. _Contributors:

Contributors
------------

*(in alphabetical order)*

* **Allie Burton**, Princeton Neuroscience Institute, Princeton University
* **Laura Bustamante**, Princeton Neuroscience Institute, Princeton University
* **Jonathan D. Cohen**, Princeton Neuroscience Institute, Princeton University
* **Samyak Gupta**, Department of Computer Science, Rutgers University
* **Abigail Hoskin**, Department of Psychology, Princeton University
* **Peter Johnson**, Princeton Neuroscience Institute, Princeton University
* **Justin Junge**, Department of Psychology, Princeton University
* **Qihong Lu**, Department of Psychology, Princeton University
* **Kristen Manning**, Princeton Neuroscience Institute, Princeton University
* **Katherine Mantel**, Princeton Neuroscience Institute, Princeton University
* **Lena Rosendahl**, Department of Mechanical and Aerospace Engineering, Princeton University
* **Dillon Smith**, Princeton Neuroscience Institute, Princeton University
* **Markus Spitzer**, Princeton Neuroscience Institute, Princeton University
* **David Turner**, Princeton Neuroscience Institute, Princeton University
* **Jan Vesely**, Department of Computer Science, Rutgers University
* **Changyan Wang**, Princeton Neuroscience Institute, Princeton University
* **Nate Wilson**, Princeton Neuroscience Institute, Princeton University

With substantial and greatly appreciated assistance from:

* **Abhishek Bhattacharjee**, Department of Computer Science, Rutgers University
* **Mihai Capota**, Intel Labs, Intel Corporation
* **Bryn Keller**, Intel Labs, Intel Corporation
* **Susan Liu**, Princeton Neuroscience Institute, Princeton University
* **Garrett McGrath**, Princeton Neuroscience Institute, Princeton University
* **Sebastian Musslick**, Princeton Neuroscience Institute, Princeton University
* **Amitai Shenhav**, Cognitive, Linguistic, & Psychological Sciences, Brown University
* **Michael Shvartsman**, Princeton Neuroscience Institute, Princeton University
* **Ben Singer**, Princeton Neuroscience Institute, Princeton University
* **Ted Willke**, Intel Labs, Intel Corporation


Table of Contents
-----------------

.. toctree::
   :titlesonly:

   self
   BasicsAndPrimer
   QuickReference
   Core
   Library
   ContributorsGuide

.. _Indices_and_Tables:

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
