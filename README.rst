.. image:: https://badge.fury.io/py/psyneulink.svg
    :target: https://badge.fury.io/py/psyneulink
.. image:: https://travis-ci.org/PrincetonUniversity/PsyNeuLink.svg?branch=master
    :target: https://travis-ci.org/PrincetonUniversity/PsyNeuLink
.. image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/PrincetonUniversity/PsyNeuLink/master

PsyNeuLink
==========

PsyNeuLink is an open-source, software environment written in Python, and designed for the needs of
neuroscientists, psychologists, computational psychiatrists and others interested in learning about and building
models of the relationship between brain function, mental processes and behavior.

Documentation is available at https://princetonuniversity.github.io/PsyNeuLink/

PsyNeuLink is alpha software, that is still being actively developed. Although it is useable, and most of the
documented functionality is available, some features may not yet be fully implemented and/or subject to
modification.

If you have trouble installing PsyNeuLink, run into any bugs, or have suggestions
for development, please contact psyneulinkhelp@princeton.edu.

Purpose
-------

PsyNeuLink is intended to serve as a "block modeling environment", in which to construct, simulate, document, and
exchange computational models of neural mechanisms and/or psychological processes at the subsystem and system levels.
A block modeling environment allows components to be constructed that implement various, possibly disparate
functions, and then link them together into a system to examine how they interact.  In PsyNeuLink, components are
used to implement the function of brain subsystems and/or psychological processes, the interaction of which can then
be simulated at the system level.

The purpose of PsyNeuLink is to make it as easy as possible to create new and/or import existing models, and
integrate them to simluate system-level interactions.  It provides a suite of core components for
implementing models of various forms of processing, learning, and control, and its Library includes examples that
combine these components to implement published models.  As an open source project, its suite of components is meant
to be enhanced and extended, and its library is meant to provide an expanding repository of models, written in a
concise, executable, and easy to interpret form, that can be shared, and compared by the scientific
community.


PsyNeuLink is:

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

    * Neuron (biophysically realistic models of neuronal function)
    * TensorFlow (ODE's, deep learning);
    * Emergent (broad class of neurally-plausible connectionist models);
    * ACT-R (symbolic, production system models).

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
stages of development) in tension with efficiency (think:  interpreted vs. compiled).  Two priorities for continued
development are the acceleration of PsyNeuLink, using just-in-time compilation methods, parallelization and adaptation
to FPGA hardware; and the implementation of a graphic interface for the construction of models and realtime display
of their execution.

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

Lists of required packages for PsyNeuLink, developing PsyNeuLink, and running the PsyNeuLink tutorial are also
stored in pip-style `requirements.txt`, `dev_requirements.txt`, and `tutorial_requirements.txt` in the source code.

If you have trouble installing the package, or run into other problems, please contact psyneulinkhelp@princeton.edu.

Tutorial
--------

PsyNeuLink includes a `tutorial <https://princetonuniversity.github.io/PsyNeuLink/#tutorial>`__, that provides examples of how to create basic Components
in PsyNeuLink, and combine them into Processes and a System.  The examples include construction of a simple
decision making process using a Drift Diffusion Model, a neural network model of the Stroop effect, and a
backpropagation network for learning the XOR problem.

The tutorial can be run in a browser by clicking the badge below or `this link <https://mybinder.org/v2/gh/PrincetonUniversity/PsyNeuLink/master>`__.

.. image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/PrincetonUniversity/PsyNeuLink/master

To run the tutorial locally, you must run python 3.5 and install additional packages:

::

    pip install psyneulink[tutorial]

or if you downloaded the source:

::

    pip install .[tutorial]


To access the tutorial, make sure your environment fulfills the requirements
mentioned above, download the `tutorial notebook <https://github.com/PrincetonUniversity/PsyNeuLink/raw/master/PsyNeuLink%20Tutorial.ipynb>`__, (you may need to right click the link and select "Save Link As...") then run the terminal command

::

    jupyter notebook


Once the notebook opens in your browser, navigate to the location where you saved the tutorial notebook, and
click on "PsyNeuLink Tutorial.ipynb".


Contributors
------------

* **Allie Burton**, Princeton Neuroscience Institute, Princeton University
* **Jonathan D. Cohen**, Princeton Neuroscience Institute, Princeton University
* **Peter Johnson**, Princeton Neuroscience Institute, Princeton University
* **Justin Junge**, Department of Psychology, Princeton University
* **Kristen Manning**, Princeton Neuroscience Institute, Princeton University
* **Katherine Mantel**, Princeton Neuroscience Institute, Princeton University
* **Markus Spitzer**, Princeton Neuroscience Institute, Princeton University
* **Jan Vesely**, Department of Computer Science, Rutgers University
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
* **Ted Willke**, Intel Labs, Intel Corporation

License
-------

::

    Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
         http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and limitations under the License.
