Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.


# PsyNeuLink

    PsyNeuLink is a block modeling system for cognitive neuroscience.
    It is open source, and meant to be extended

### Contributors

    Jonathan D. Cohen, Princeton Neuroscience Institute, Princeton University
    Peter Johnson, Princeton Neuroscience Institute, Princeton University
    Bryn Keller, Intel Labs, Intel Corporation
    Sebastian Musslick, Princeton Neuroscience Institute, Princeton University
    Aida Piccato, Princeton University
    Amitai Shenhav, Cognitive, Linguistic, & Psychological Sciences, Brown University
    Michael Shvartsman, Princeton Neuroscience Institute, Princeton University
    Ted Willke, Intel Labs, Intel Corporation
    Nate Wilson, Princeton Neuroscience Institute, Princeton University

## Installation

    Right now, PsyNeuLink is in an alpha state and is not available through pypi/pip. Instead, you can clone the github repo [here](https://github.com/PrincetonUniversity/PsyNeuLink). For now, clone the devel branch rather than the master branch. You can switch branches by pressing the "branch: master" dropdown menu on the left side of the page. Then download the package with the green "Clone or download" button on the right side of the page and "Download ZIP."

    Alternatively, if you are familiar with git, the directory can be cloned as usual through the terminal.
    Note: The repo is currently private, so if the link leads to a dead page, reach out to one of the developers to get acccess.

    PsyNeuLink is compatible with any version of python 3, but this tutorial requires a 3.5 installation with the latest versions of IPython, jupyter, and matplotlib installed.

    To install the package, navigate to the cloned directory in a terminal, switch to your preferred python3 environment, then run the command __"pip install ."__ (make sure to include the period and to use the appropriate pip/pip3 command for python 3.5). All prerequisite packages will be automatically added to your enviroment.

## Purpose

    To provide an environment for implementing models of mind/brain function
    that is modular, customizable and extensible.  It does this in a manner that:
     - is computationally general
     - adheres as closely as possible to the insights and design principles that have been learned in CS
         (e.g., function-based, object-oriented, programming)
     - expresses (the smallest number of) "commitments" that reflect general principles of how
         the brain/mind is organized and function, without committing to any particular model or theory
     - expresses these commitments in a form that is powerful, easy to use, and familiar to cognitive neuroscientitsts
     - allows models to be simply and flexibly implemented, using a minimum of coding that provides 
         seamless interaction among disparate components that can vary in their:
         - time-scale of operation
         - granularity of representation and function
    - encourages users to think about processing in a "mind/brain-like" way,
         while imposing as few constraints as possible on what they can implement or ask their model to do
    - provides a standard environment for model comparison, sharing, and integration  

## Functional Architecture

     - System 
         Set of (potentially interacting) processes, that can be managed by a “budget” of control

         - Process 
             Component that takes an input, processes it through an ordered list of mechanisms (and projections)
             and generates an output
    
             - Mechanism 
                 Component that converts an input state representation into an output state representation
                 Parameters determine its operation, under the influence of projections
                 
                 + ProcessingMechanism
                     Component that takes a
                 
                 + ControlMechanism
                 
                 + MonitoringMechanism
    
             - Projection 
                 Component that takes a source of in, possibly transforms it, and uses it to
                 determine the operation of a mechanism;  three primary types:
    
                 + Mapping
                     Takes the output of sender mechanism, possibly transforms it,
                         and provides it as the input to a receiver mechanism
    
                 + ControlSignal
                     Takes an allocation (scalar), possibly transforms it,
                     and uses it to modulate the parameter(s) of a mechanism's function
    
                 + Learning
                      Takes an error signal (scalar or vector), possibly transforms it,
                      and uses it to modulate the parameter of a projection function
                     
                 [+ GatingSignal — Not yet implemented
                     Takes a source, possibly transforms it, and uses it to
                     modulate the input or output state of a mechanism
