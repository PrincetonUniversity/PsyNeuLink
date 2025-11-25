# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  TargetProjection ****************************************************

"""

Contents
--------
  * `TargetProjection_Overview`
  * `TargetProjection_Creation`
      COMMENT: DELETE IF NOT NEEDED
      - `TargetProjection_Deferred_Initialization`
      COMMENT
  * `TargetProjection_Structure`
  * `TargetProjection_Execution`
  * `TargetProjection_Class_Reference`


.. _TargetProjection_Overview:

Overview
--------

A TargetProjection is used for `learning <Composition_learning>`, to specify where the target pattern of activity
used to train the output of a `ProcessingMechanism <ProcessingMechanism>` comes from. How it is used depends on the
`configuration <Composition_Learning_Configurations>` used for learning; see `Composition_Learning_Components`
for the standard learning configuration in PsyNeuLink, and `AutodiffComposition_PyTorch` for configuration when
using `PyTorch <https://pytorch.org>` for learning in an `AutodiffComposition`.


.. _TargetProjection_Creation:

Creating a TargetProjection
-----------------------------

A TargetProjection can be created in any of the ways that can be used to create a `Projection <Projection_Creation>`
(see `Projection_Sender` and `Projection_Receiver` for specifying its `sender <TargetProjection.sender>` and
`receiver <TargetProjection.receiver>` attributes, respectively).

TargetProjections are also assigned automatically in the following circumstances:

  * by a `Composition` when a learning pathway is specified (see `Composition_Learning_Methods` for details)
    and the standard learning configuration is used (see `Composition_Learning_Components` for details),
    with the automatically created `TARGET_MECHANISM <TARGET_MECHANISM>` as its `sender <TargetProjection.sender>`
    and the automatically created  `OBJECTIVE_MECHANISM <OBJECTIVE_MECHANISM>` as its `receiver
    <TargetProjection.receiver>`;

  COMMENT:  TBI
  * by a `AutodiffComposition` when `AutodiffComposition_PyTorch` is used for learning and no TargetProjections
    are specified, with the automatically created `TARGET_MECHANISM <TARGET_MECHANISM>` as its `sender
    <TargetProjection.sender>` and the automatically created  `OBJECTIVE_MECHANISM <OBJECTIVE_MECHANISM>` as its
    `receiver <TargetProjection.receiver>`;
  COMMENT

COMMENT:  DELETE IF NOT NEEDED
.. _TargetProjection_Deferred_Initialization:

*Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~
FILL OUT IF NEEDED
COMMENT

.. _TargetProjection_Structure:

Structure
---------

The `function <Projection_Base.function>` for a TargetProjection is a `Linear`, that simply passes its `variable
<Projection_Base.variable>` unmodified to its `value <Projection_Base.value>`. An error is generated if any attempt
is made to modify any of its attributes or `Parameters <Parameter>`, and they are left unchanged.


.. _TargetProjection_Execution:

Execution
---------

When a TargetProjection is executed, it simply passes the value it receives from its `sender <TargetProjection.sender>`,
which is a `TARGET MECHANISM <TARGET_MECHANISM>` in a `learning pathway <Composition_Learning_Methods>`, directly to its
`receiver <TargetProjection.receiver>`, which is an `OBJECTIVE MECHANISM <OBJECTIVE_MECHANISM>` in the same learning
pathway.

.. _TargetProjection_Class_Reference:

Class Reference
---------------

"""
import copy

import numpy as np
from typing import Union

from psyneulink._typing import Optional

from psyneulink.core.components.projections.projection import Projection_Base, ProjectionError, projection_keywords
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.keywords import \
    (AUTO_ASSIGN_MATRIX, DEFAULT, DEFAULT_MATRIX, FULL_CONNECTIVITY_MATRIX, HOLLOW_MATRIX,
     IDENTITY_MATRIX, INPUT_PORT, MAPPING_PROJECTION, MATRIX, OUTPUT_PORT, TARGET_PROJECTION, VALUE)
from psyneulink.core.globals.log import ContextFlags
from psyneulink.core.globals.parameters import FunctionParameter, Parameter, check_user_specified, copy_parameter_value
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = ['TargetProjection', 'TargetProjectionError']

projection_keywords.update({TARGET_PROJECTION})


class TargetProjectionError(ProjectionError):
    pass


class TargetProjection(Projection_Base):
    """
    TargetProjection(  \
        sender=None,   \
        receiver=None, \

    Subclass of `Projection` that transmits the `value <OutputPort.value>` of the `OutputPort` of a
    `TARGET_MECHANISM <TARGET_MECHANISM>` to the `variable <InputPort.variable>` of the `InputPort` of an
    `OBJECTIVE_MECHANISM <OBJECTIVE_MECHANISM>` in a `learning pathway <Composition_Learning_Methods>`.
    It is used to specify the target pattern of activity for `learning <Composition_learning>`.
    See `Projection <Projection_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    sender : OutputPort or Mechanism : default None
        specifies the source of the Projection's input. If a `Mechanism <Mechanism>` is specified, its
        `primary OutputPort <OutputPort_Primary>` is used. If it is not specified, it is assigned in
        the context in which the TargetProjection is used, or its initialization will be `deferred
        <Projection_Deferred_Initialization>`.

    receiver: InputPort or Mechanism : default None
        specifies the destination of the Projection's output.  If a `Mechanism <Mechanism>` is specified, its
        `primary InputPort <InputPort_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used, or its initialization will be `deferred
        <TargetProjection_Deferred_Initialization>`.


    Attributes
    ----------

    sender : OutputPort
        the `OutputPort` of the `Mechanism <Mechanism>` that is the source of the Projection's input.

    receiver: InputPort
        the `InputPort` of the `Mechanism <Mechanism>` that is the destination of the Projection's output.

    name : str
        the name of the TargetProjection. If the specified name is the name of an existing TargetProjection,
        it is appended with an indexed suffix, incremented for each TargetProjection with the same base name (see
        `Registry_Naming`). If the name is not specified in the **name** argument of its constructor, a default name is
        assigned using the following format:
        'TargetProjection from <sender Mechanism>[<OutputPort>] to <receiver Mechanism>[InputPort]'
        (for example, ``'TargetProjection from my_mech_1[OutputPort-0] to my_mech2[InputPort-0]'``).
        If either the `sender <TargetProjection.sender>` or `receiver <TargetProjection.receiver>` has not yet been
        assigned (the TargetProjection is in `deferred initialization <TargetProjection_Deferred_Initialization>`),
        then the parenthesized name of class is used in place of the unassigned attribute
        (for example, if the `sender <TargetProjection.sender>` has not yet been specified:
        ``'TargetProjection from (OutputPort-0) to my_mech2[InputPort-0]'``).

    """

    componentType = TARGET_PROJECTION
    className = componentType
    suffix = " " + className
    classPreferenceLevel = PreferenceLevel.TYPE

    class sockets:
        sender=[OUTPUT_PORT]
        receiver=[INPUT_PORT]

    projection_sender = OutputPort

    @check_user_specified
    def __init__(self,
                 sender=None,
                 receiver=None,
                 params=None,
                 name=None,
                 prefs: Optional[ValidPrefSet] = None,
                 context=None,
                 **kwargs):

        # If sender or receiver has not been assigned, defer init to Port.instantiate_projection_to_state()
        if sender is None or receiver is None:
            self.initialization_status = ContextFlags.DEFERRED_INIT

        # Validate sender (as variable) and params
        super().__init__(sender=sender,
                         receiver=receiver,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)


    @property
    def logPref(self):
        return self.prefs.logPref
