.. image:: https://badge.fury.io/py/psyneulink.svg
    :target: https://badge.fury.io/py/psyneulink

PsyNeuLink
==========

PsyNeuLink is a "block modeling system" for cognitive neuroscience.

Documentation is available at https://princetonuniversity.github.io/PsyNeuLink/

PsyNeuLink is alpha software, that is still being actively developed. Although it is useable, and most of the
documented functionality is available, some features may not yet be fully implemented and/or subject to
modification.

If you have trouble installing PsyNeuLink, run into any bugs, or have suggestions
for development, please contact psyneulinkhelp@princeton.edu.

Purpose
-------

PsyNeuLink is a "block modeling system" for cognitive neuroscience.  Block modeling systems allow components to be
constructed that implement various, possibly disparate functions, and then link them together into a system to
examine how they interact.  In PsyNeuLink, components are used to implement the function of brain subsystems and/or
psychological processes, the interaction of which can then be simulated at the system level.

PsyNeuLink is open source, and meant to be extended. Its goal is to provide an environment for implementing models
of mind/brain function that are modular, customizable, extensible, disseminable, easily reproducible and clearly
documented.  It does this in a manner that:

 - is *computationally general* -- that is, that can implement any desired mechanism or process;

 - adheres as closely as possible to the insights and design principles that have been learned in computer science
   (e.g., function-based, object-oriented programming);

 - expresses (as few as possible) *commitments* that reflect general principles of how the mind/brain is organized
   and operates, without committing to any particular detailed model or theory;

 - expresses these commitments in a form that is powerful, easy to use, and familiar to cognitive neuroscientists;

 - allows models to be simply and flexibly implemented, using a minimum of coding, and that provides seamless
   integration of, and interaction among disparate components that can vary in their:

   - granularity of representation and function
      time-scale of operation

 - provides a standard and accessible environment for model comparison, sharing, and documentation;

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

.. _Tutorial:

Tutorial
--------

PsyNeuLink includes a `tutorial <https://princetonuniversity.github.io/PsyNeuLink/#tutorial>`__, that provides examples of how to create basic Components
in PsyNeuLink, and combine them into Processes and a System.  The examples include construction of a simple
decision making process using a Drift Diffusion Model, a neural network model of the Stroop effect, and a
backpropagation network for learning the XOR problem.

The tutorial currently requires additional packages; to install the required tutorial packages, you may use PyPI:

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
* **Kristen Manning**, Princeton Neuroscience Institute, Princeton University
* **K. Mantel**, Princeton Neuroscience Institute, Princeton University
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

License
-------

::

    Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
         http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and limitations under the License.
