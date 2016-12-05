# PsyNeuLink

PsyNeuLink is a block modeling system for cognitive neuroscience.
It is open source, and meant to be extended
Documentation is available at https://princetonuniversity.github.io/PsyNeuLink/
The tutorial is available by [???] 

### Contributors

    Jonathan D. Cohen, Princeton Neuroscience Institute, Princeton University
    Peter Johnson, Princeton Neuroscience Institute, Princeton University
    Bryn Keller, Intel Labs, Intel Corporation
    Sebastian Musslick, Princeton Neuroscience Institute, Princeton University
    Amitai Shenhav, Cognitive, Linguistic, & Psychological Sciences, Brown University
    Michael Shvartsman, Princeton Neuroscience Institute, Princeton University
    Ted Willke, Intel Labs, Intel Corporation
    Nate Wilson, Princeton Neuroscience Institute, Princeton University 

### License

    Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
         http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and limitations under the License.

## Purpose

To provide an environment for implementing models of mind/brain function that is modular, customizable and extensible.  
It does this in a manner that:

- is computationally general
- adheres as closely as possible to the insights and design principles that have been learned in CS
    (e.g., function-based, object-oriented, programming)
- expresses (the smallest number of) "commitments" that reflect general principles of how
    the brain/mind is organized and function, without committing to any particular model or theory
- expresses these commitments in a form that is powerful, easy to use, and familiar to cognitive neuroscientitsts
- allows models to be simply and flexibly implemented, using a minimum of coding that provides 
     seamless interaction among disparate components that can vary in their:
     + time-scale of operation
     + granularity of representation and function
- encourages users to think about processing in a "mind/brain-like" way,
     while imposing as few constraints as possible on what they can implement or ask their model to do
- provides a standard environment for model comparison, sharing, and integration  

## Functional Architecture

- System:
     Set of (potentially interacting) processes, that can be managed by a “budget” of control and trained.

 - Process: 
     Takes an input, processes it through an ordered list of mechanisms and projections, and generates an output.

     - Mechanism: 
         Converts an input state representation into an output state representation.
         Parameters determine its operation, under the influence of projections.
         
         + ProcessingMechanism:
              Used to tranform representations.              
         
         + ControlMechanism
              Used to evaluate the consequences of transformations carried out by ProcessingMechanisms
              and modulate the parameters of those mechanisms to maximize or minimize some objective function.
         
         + MonitoringMechanism
              Used to evaluate the consequences of transformations carried out by a set of ProcessingMechanisms
              and modulate the parameters of Mapping projections between to maximize or minimize some objective function.

     - Projection: 
         Takes the output of a mechanism, possibly transforms it, and uses it to
         determine the operation of another mechanism;  three primary types:

         + MappingProjection:
             Takes the output of a mechanism and provides it as the input to another mechanism.

         + ControlProjection:
             Takes an allocation (scalar) (usually the output of a ControlMechanism) 
             and uses it to modulate the parameter(s) of a mechanism.

         + LearningProjection:
             Takes an error signal (scalar or vector) (usually the output of a Monitoring Mechanism) 
             and uses it to modulate the matrix parameter of a MappingProjection.
             
         [+ GatingProjection — Not yet implemented
             Takes a gating signal source and uses it to modulate the input or output state of a mechanism.
