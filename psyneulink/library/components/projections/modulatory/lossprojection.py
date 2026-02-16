# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningProjection **********************************************************

"""

Contents
--------

  * `LearningProjection_Overview`
  * `LearningProjection_Class_Reference`

.. _LearningProjection_Overview:

Overview
--------

A LossProjection is a placemarker used to represent the passing of loss from a `LossMechanism` to the node that
serves as its `sample <LossMechanism.sample>`.  It has no functionality beyond that, and is used to generate a graphic
representation of this relationship with the `show_graph <ShowGraph.show_graph>` method of an `AutodiffComposition` is
called with its **show_pytorch** argument set to `True`.

.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

from beartype import beartype

from psyneulink._typing import Optional

from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.projection import ProjectionError, projection_keywords
from psyneulink.library.components.mechanisms.processing.objective.lossmechanism import LossMechanism
from psyneulink.core.globals.keywords import DEFAULT, INPUT_PORT, LOSS_PROJECTION, OUTPUT_PORT
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.parameters import Parameter

__all__ = [
    'LossProjection',
]

class LossProjectionError(ProjectionError):
    pass


class LossProjection(ModulatoryProjection_Base):
    """
    LearningProjection(                \
                 sender=None,          \
                 receiver=None,        \
                 name=None,            \
                 prefs=None)

    Subclass of `ModulatoryProjection <ModulatoryProjection>` that represents the passing the loss from
    `LossMechanism` to the node that serves as its `sample <LossMechanism.sample>`.  It is used to generate a
    graphic representation of this relationship with the `show_graph <ShowGraph.show_graph>` method of an
    `AutodiffComposition` is called with its **show_pytorch** argument set to `True`.

    Arguments
    ---------

    sender : LossMechanism
        specifies the `LossMechanism` that computes the `loss <LossMechanism.loss>` used to train the `learning
        pathway <Composition_learning_Pathway>` terminating in the sample <LossMechanism.sample>` Mechanism.

    receiver : ProcessingMechanism
        specifies the `sample <LossMechanism.sample>` Mechanism for which the `loss <LossMechanism.loss>` is
        computed with respect to the `LossMechanism`\'s `target <LossMechanism.target>`.

    Attributes
    ----------

    sender : LossMechanism
        the `LossMechanism` that computes the `loss <LossMechanism.loss>` used to train the `learning pathway
        <Composition_learning_Pathway>` terminating in the sample <LossMechanism.sample>` Mechanism.

    receiver : ProcessingMechanism
        the `sample <LossMechanism.sample>` Mechanism for which the `loss <LossMechanism.loss>` is computed with
        respect to the `LossMechanism`\'s `target <LossMechanism.target>`.
    """

    componentType = LOSS_PROJECTION
    className = componentType
    suffix = " " + className
    classPreferenceLevel = PreferenceLevel.TYPE
    projection_sender = LossMechanism

    class sockets:
        sender=[OUTPUT_PORT]
        receiver=[INPUT_PORT]

    class Parameters(ModulatoryProjection_Base.Parameters):
        """
            Attributes
            ----------

                exponent
                    see `exponent <Projection_Base.exponent>`

                    :default value: None
                    :type:

                function
                    see `function <Projection_Base.function>`

                    :default value: `MatrixTransform`
                    :type: `Function`

                weight
                    see `weight <Projection_Base.weight>`

                    :default value: None
                    :type:
        """
        has_initializers = Parameter(None, pnl_internal=True, fallback_value=DEFAULT)


    @beartype
    def __init__(self,
                 sender: LossMechanism = None,
                 receiver: OutputPort = None,
                 name=None,
                 prefs: Optional[ValidPrefSet] = None):

        name = f'LOSS PROJECTION for {receiver.name}'
        self.sender = sender
        self.receiver = receiver
        self.learnable = False
        self.matrix = None
