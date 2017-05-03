# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  ModulatoryProjection *****************************************************

"""
.. _ModulatoryProjection_Overview:

Overview
--------

ModulatoryProjections allow information to be passed between mechanisms.  A projection takes its input from the
`outputState <OutputState>` of one mechanism (its `sender <Projection.sender>`), and does whatever conversion is
needed to transmit that information to the `inputState <InputState>` of another mechanism (its
`receiver <Projection.receiver>`).  There are three types of projections that serve difference purposes:

* `LearningProjection`
    These take an `error_signal <LearningProjection.LearningProjection.error_signal>`, usually the output of a
    `MonitoringMechanism <MonitoringMechanism>`, and transmit this to the `parameterState <ParameterState>` of a
    `MappingProjection` which uses this to modify its `matrix <MappingProjection.MappingProjection.matrix>`
    parameter. LearningProjections are used when learning has been specified for a `process <Process_Learning>`
    or `system <System_Execution_Learning>`.
..
* `GatingProjection`
    These take an `allocation <ControlProjection.ControlProjection.allocation>` specification, usually the ouptput
    of a `ControlMechanism <ControlMechanism>`, and transmit this to the `parameterState <ParameterState>` of
    a ProcessingMechanism which uses this to modulate a parameter of the mechanism or its function.
    ControlProjections are typically used in the context of a `System`.
..
* `ControlProjection`
    These take an `allocation <ControlProjection.ControlProjection.allocation>` specification, usually the ouptput
    of a `ControlMechanism <ControlMechanism>`, and transmit this to the `parameterState <ParameterState>` of
    a ProcessingMechanism which uses this to modulate a parameter of the mechanism or its function.
    ControlProjections are typically used in the context of a `System`.
..

COMMENT:
* Gating: takes an input signal and uses it to modulate the inputState and/or outputState of the receiver
COMMENT

.. _Projection_Creation:

Creating a ModulatoryProjection
-------------------------------

A ModulatoryProjection can be created on its own, by calling the constructor for the desired type of projection.  More
commonly, however, projections are either specified `in context <Projection_In_Context_Specification>`, or
are `created automatically <Projection_Automatic_Creation>`, as described below.



.. _ModulatoryProjection_Structure:

Structure
---------

.. _ModulatoryProjection_Execution:

Execution
---------

.. _ModulatoryProjection_Class_Reference:

"""

from PsyNeuLink.Components.Projections.Projection import Projection_Base

class ModulatoryProjection_Base(Projection_Base):
    def __init__(self,
                 receiver,
                 sender=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):

        super().__init__(receiver=receiver,
                         sender=sender,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)
