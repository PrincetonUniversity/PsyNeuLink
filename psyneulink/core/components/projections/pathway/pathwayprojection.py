# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  PathwayProjection *****************************************************

"""

Contents
--------

  * `PathwayProjection_Overview`
  * `PathwayProjection_Creation`
  * `PathwayProjection_Structure`
      - `PathwayProjection_Sender`
      - `PathwayProjection_Receiver`
  * `PathwayProjection_Execution`
  * `PathwayProjection_Class_Reference`


.. _PathwayProjection_Overview:

Overview
--------

PathwayProjections allow information to be passed between mechanisms.  A PathwayProjection takes its input from the
`OutputPort` of one Mechanism (its `sender <Projection_Base.sender>`), and does whatever conversion is needed to
transmit that information to the `InputPort` of another Mechanism (its `receiver <Projection_Base.receiver>`).  The
primary type of PathwayProjection is a `MappingProjection`.

.. _PathwayProjection_Creation:

Creating a PathwayProjection
---------------------------------

A PathwayProjection can is created in the same ways as a `Projection <Projection_Creation>`.

.. _PathwayProjection_Structure:

Structure
---------

A PathwayProjection has the same structure as a `Projection <Projection_Structure>`.


.. _PathwayProjection_Execution:

Execution
---------

A PathwayProjection executes in the same was as a `Projection <Projection_Execution>`.

.. _PathwayProjection_Class_Reference:

Class Reference
---------------

See `Projection <Projection_Class_Reference>`.

"""

from psyneulink.core.components.projections.projection import Projection_Base, ProjectionRegistry
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import NAME, PATHWAY_PROJECTION, RECEIVER, SENDER
from psyneulink.core.globals.registry import remove_instance_from_registry

__all__ = []

class PathwayProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class PathwayProjection_Base(Projection_Base):
    """Subclass of `Projection <Projection>` that projects from an `OutputPort` to an `InputPort`
    See `Projection <Projection_Class_Reference>` and subclasses for arguments and attributes.

    .. note::
       PathwayProjection is an abstract class and should *never* be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <PathwayProjection_Subtypes>`.

    """

    componentCategory = PATHWAY_PROJECTION

    def _assign_default_projection_name(self, port=None, sender_name=None, receiver_name=None):

        from psyneulink.core.components.mechanisms.mechanism import Mechanism


        name_template = "{}[{}]"
        projection_name_template = "{} from {} to {}"

        if self.initialization_status == ContextFlags.DEFERRED_INIT:
            if self._init_args[SENDER]:
                sender = self._init_args[SENDER]
                if isinstance(sender, type):
                    sender_name = f"({sender.__name__})"
                elif isinstance(sender.owner, Mechanism):
                    sender_name = name_template.format(sender.owner.name, sender_name)
            if self._init_args[RECEIVER]:
                receiver = self._init_args[RECEIVER]
                if isinstance(receiver.owner, Mechanism):
                    receiver_name = name_template.format(receiver.owner.name, receiver_name)
            projection_name = projection_name_template.format(self.className, sender_name, receiver_name)
            self._init_args[NAME] = self._init_args[NAME] or projection_name
            self.name = self._init_args[NAME]

        # If the name is not a default name, leave intact
        elif not self.className + '-' in self.name:
            return self.name

        elif self.initialization_status == ContextFlags.INITIALIZED:
            if self.sender.owner:
                sender_name = name_template.format(self.sender.owner.name, self.sender.name)
            if self.receiver.owner:
                receiver_name = name_template.format(self.receiver.owner.name, self.receiver.name)
            self.name = projection_name_template.format(self.className, sender_name, receiver_name)

        else:
            raise PathwayProjectionError(
                "PROGRAM ERROR: {} has unrecognized initialization_status ({})".format(
                    self, self.initialization_status
                )
            )
