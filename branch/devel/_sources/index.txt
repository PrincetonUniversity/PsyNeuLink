.. PsyNeuLink documentation master file, created by
   sphinx-quickstart on Wed Oct 19 11:51:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Intro
=====

* `Purpose`
* `What PsyNeuLink is NOT <What_PsyNeuLink_is_NOT>`
* `Overview`
* `Installation`
* `Tutorial`
* `Contributors`
* `Indices_and_Tables`


.. _Purpose:

Purpose
-------

PsyNeuLink is a "block modeling system" for cognitive neuroscience.  Block modeling systems allow components to be
constructed that implement various, possibly disparate functions, and then link them together into a system to
examine how they interact.  In PsyNeuLink, components are used to implement the function of brain subsystems and/or
psychological processes, the interaction of which can then be simulated at the system level.

PsyNeuLink is open source, and meant to be extended. Its goal is to provide an environment for implementing models
of mind/brain function that are modular, customizable, extensible, disseminable, easily reproducible and clearly
documented.  It does this in a manner that:

 - is *computationally general* -- that is, that can implement any desired mechanism or process;
 ..
 - adheres as closely as possible to the insights and design principles that have been learned in computer science
   (e.g., function-based, object-oriented programming);
 ..
 - expresses (as few as possible) *commitments* that reflect general principles of how the mind/brain is organized
   and operates, without committing to any particular detailed model or theory;
 ..
 - expresses these commitments in a form that is powerful, easy to use, and familiar to cognitive neuroscientists;
 ..
 - allows models to be simply and flexibly implemented, using a minimum of coding, and that provides seamless
   integration of, and interaction among disparate components that can vary in their:
     - granularity of representation and function
     - time-scale of operation
 ..
 - provides a standard and accessible environment for model comparison, sharing, and documentation;
 ..
 - has an interface (API) that allows it to be used with other powerful tools for implementing individual components,
   such as:

   * MatLab (general purpose mathematical tools);
   * TensorFlow, Teano (ODE's, deep learning);
   * Emergent (broad class of neurally-plausible connectionist models);
   * ACT-R (symbolic, production system models).

The goal is to encourage users to think about information processing in a "mind/brain-like" way, while imposing as few
constraints as possible on what it is possible to implement or ask the model to do.

.. _What_PsyNeuLink_is_NOT:

What PsyNeuLink is **NOT**
--------------------------

PsyNeuLink is well suited to the creation of simple to moderately complex models, and to the integration of
disparate existing models into a single, integrated system in which interactions among them can be examined.
While it is fully general, and can be used to implement virtually any kind of model, it is less well suited to other
kinds of efforts, that involve massively large computations and/or specialized functions and data types that it
currently does not support, such as:

 - extensive model fitting
 - large scale simulations
 - biophysically-realistic models of individual neurons

Other packages that are better suited to such applications are:
`Emergent <https://grey.colorado.edu/emergent/index.php/Main_Page>`_ and
`TensorFlow <https://www.tensorflow.org>`_ (for neural network models);
`HDDM <http://ski.clps.brown.edu/hddm_docs/>`_ (for Drift Diffusion Models);
`ACT-R <http://act-r.psy.cmu.edu>`_ (for production system models);
`Genesis <http://www.genesis-sim.org>`_,
`Neuron <https://www.neuron.yale.edu/neuron/>`_,
and `Nengo <http://www.nengo.ca>`_  (for biophysically-realistic models of neuronal function).
These packages are good for elaborate and detailed models of a particular form.
In contrast, the focus in designing and implementing PsyNeuLink has been to make it as flexible and easy to use as
possible, with the ability to integrate components constructed in other packages (including some of the ones listed
above) into a single functioning system.  These are characteristics that are often (at least in the initial
stages of development) in tension with efficiency (think:  interpreted vs. compiled).  One of the goals for future
development is to make PsyNeuLink more computationally efficient.  At present, however, it is best suited to
developing simpler models, or taking complex or highly detailed models that have been developed --
or subjected to extensive parameter fitting -- in other frameworks, and re-expressing them in a form that is amenable
to integration, documentation, and dissemination.

.. _Overview:

Overview
--------

PsyNeuLink is written in Python, and conforms to the syntax and coding standards for the language.
`BasicsAndSampler` provides an orientation to PsyNeuLinks Components, some examples of what PsyNeuLink models
look like, and some of its capabilities. `QuickReference` provides an overview of how PsyNeuLink is organized and
some of its basic principles of operation.  The `Tutorial <Tutorial>` provides an interactive guide to the
construction of models using PsyNeuLink.

.. _Installation:

Installation
------------

Currently, PsyNeuLink is in an alpha state and is not available through pypi/pip.
Instead, you can clone the github repo (https://github.com/PrincetonUniversity/PsyNeuLink).
Clone the master branch.
Download the package with the green "Clone or download" button on the right side of the page and "Download ZIP."

Alternatively, if you are familiar with git, the directory can be cloned as usual through the terminal. Note: The
repo is currently private, so if the link leads to a dead page, reach out to one of the developers to get access.

PsyNeuLink is compatible with any version of python 3, but the tutorial (see below) requires a 3.5 installation with
the latest versions of IPython, jupyter, and matplotlib installed.

To install the package, navigate to the cloned directory in a terminal,
switch to your preferred python3 environment, then run the command __"pip install ."__
(make sure to include the period and to use the appropriate pip/pip3 command for python 3.5).
All prerequisite packages will be automatically added to your environment.

If you have trouble installing the package, or run into other problems, please contact psyneulinkhelp@princeton.edu.


.. _Tutorial:

Tutorial
--------

The downloaded package includes a tutorial, that provides examples of how to create basic Components
in PsyNeuLink, and combine them into Processes and a System.  The examples include construction of a simple
decision making process using a Drift Diffusion Model, a neural network model of the Stroop effect, and a
backpropagation network for learning the XOR problem.

The tutorial can be run using the terminal command ``jupyter notebook`` within the root directory of the PsyNeuLink
package. Once the jupyter notebook opens, within the list of files click on "PsyNeuLink Tutorial .ipynb".  This will
open the tutorial, that will provide any additional information needed to get started.


.. _Contributors:

Contributors
------------

* **Allie Burton**, Princeton Neuroscience Institute, Princeton University
* **Jonathan D. Cohen**, Princeton Neuroscience Institute, Princeton University
* **Peter Johnson**, Princeton Neuroscience Institute, Princeton University
* **Kristen Manning**, Princeton Neuroscience Institute, Princeton University
* **Kevin Mantel**, Princeton Neuroscience Institute, Princeton University
* **Ted Willke**, Intel Labs, Intel Corporation
* **Changyan Wang**, Princeton Neuroscience Institute, Princeton University
* **Nate Wilson**, Princeton Neuroscience Institute, Princeton University

With substantial and greatly appreciated assistance from:

* **Abhishek Bhattacharjee**, Department of Computer Science, Rutgers University
* **Mihai Capota**, Intel Labs, Intel Corporation
* **Bryn Keller**, Intel Labs, Intel Corporation
* **Garrett McGrath**, Princeton Neuroscience Institute, Princeton University
* **Sebastian Musslick**, Princeton Neuroscience Institute, Princeton University
* **Amitai Shenhav**, Cognitive, Linguistic, & Psychological Sciences, Brown University
* **Michael Shvartsman**, Princeton Neuroscience Institute, Princeton University
* **Ben Singer**, Princeton Neuroscience Institute, Princeton University
* **Jan Vesely**, Department of Computer Science, Rutgers University


.. toctree::
   :titlesonly:
   :maxdepth: 1

   self
   BasicsAndSampler
   QuickReference
   Components <Component>
   Compositions <Composition>
   Scheduling

.. _Indices_and_Tables:

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
