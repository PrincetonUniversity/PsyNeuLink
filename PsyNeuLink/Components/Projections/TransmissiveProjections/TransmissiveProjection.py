# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  TransmissiveProjection *****************************************************

"""
.. _TransmissiveProjection_Overview:

Overview
--------

ModulatoryProjections allow information to be passed between mechanisms.  A projection takes its input from the
`outputState <OutputState>` of one mechanism (its `sender <Projection.sender>`), and does whatever conversion is
needed to transmit that information to the `inputState <InputState>` of another mechanism (its
`receiver <Projection.receiver>`).  There are three types of projections that serve difference purposes:

* `MappingProjection`
    XXXX.
..


.. _Projection_Creation:

Creating a TransmissiveProjection
---------------------------------

A TransmissiveProjection can be created on its own, by calling the constructor for the desired type of projection.  More
commonly, however, projections are either specified `in context <Projection_In_Context_Specification>`, or
are `created automatically <Projection_Automatic_Creation>`, as described below.



.. _TransmissiveProjection_Structure:

Structure
---------

.. _TransmissiveProjection_Execution:

Execution
---------

.. _TransmissiveProjection_Class_Reference:

"""

from PsyNeuLink.Components.Projections.Projection import Projection_Base

class TransmissiveProjection_Base(Projection_Base):
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
