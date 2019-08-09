# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  PathwayProjection *****************************************************

"""
.. _PathwayProjection_Overview:

Overview
--------

PathwayProjections allow information to be passed between mechanisms.  A PathwayProjection takes its input from the
`OutputState` of one Mechanism (its `sender <Projection_Base.sender>`), and does whatever conversion is needed to
transmit that information to the `InputState` of another Mechanism (its `receiver <Projection_Base.receiver>`).  The
primary type of PathwayProjection is a `MappingProjection`.

.. _Projection_Creation:

Creating a PathwayProjection
---------------------------------

A PathwayProjection can be created on its own, by calling the constructor for the desired type of projection.  More
commonly, however, projections are either specified in the context of creating a State to or from which they project
 (see `Projection_Specification`, and `State_Projections` in State), or are `created automatically
 <Projection_Automatic_Creation>`, as described below.



.. _PathwayProjection_Structure:

Structure
---------

.. _PathwayProjection_Execution:

Execution
---------

.. _PathwayProjection_Class_Reference:

"""

from psyneulink.core.components.projections.projection import Projection_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import CONTEXT, INITIALIZATION_STATUS, NAME, PATHWAY_PROJECTION, RECEIVER, SENDER
from psyneulink.core.globals.keywords import NAME, PATHWAY_PROJECTION, RECEIVER, SENDER

__all__ = []

class PathwayProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class PathwayProjection_Base(Projection_Base):
    """Subclass of `Projection <Projection>` that projects from an `OutputState` to an `InputState`

    .. note::
       PathwayProjection is an abstract class and should NEVER be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <PathwayProjection_Subtypes>`.

    """

    componentCategory = PATHWAY_PROJECTION

    def _assign_default_projection_name(self, state=None, sender_name=None, receiver_name=None):

        from psyneulink.core.components.mechanisms.mechanism import Mechanism


        name_template = "{}[{}]"
        projection_name_template = "{} from {} to {}"

        if self.initialization_status == ContextFlags.DEFERRED_INIT:
            if self.init_args[SENDER]:
                sender = self.init_args[SENDER]
                if isinstance(sender, type):
                    sender_name = f"({sender.__name__})"
                elif isinstance(sender.owner, Mechanism):
                    sender_name = name_template.format(sender.owner.name, sender_name)
            if self.init_args[RECEIVER]:
                receiver = self.init_args[RECEIVER]
                if isinstance(receiver.owner, Mechanism):
                    receiver_name = name_template.format(receiver.owner.name, receiver_name)
            projection_name = projection_name_template.format(self.className, sender_name, receiver_name)
            self.init_args[NAME] = self.init_args[NAME] or projection_name
            self.name = self.init_args[NAME]

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

    def _delete_projection(projection):
        """Delete Projection and its entry in receiver and sender lists"""
        del projection.sender.efferents[projection.sender.efferents.index(projection)]
        del projection.receiver.path_afferents[projection.receiver.path_afferents.index(projection)]
        del projection