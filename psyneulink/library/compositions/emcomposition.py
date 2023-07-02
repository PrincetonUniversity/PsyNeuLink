# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition *************************************************

"""

Contents
--------

  * `EMComposition_Overview`
  * `EMComposition_Creation`
  * `EMComposition_Execution`
  * `EMComposition_Examples`
  * `EMComposition_Class_Reference`


.. _EMComposition_Overview:

Overview
--------

Implements a differentiable version of an `EpisodicMemoryMechanism` as a `Composition`, that can serve as a form
of episodic, or external memory in an `AutodiffComposition` capable of learning. It implements all of the functions
of a `ContentAddressableMemory` `Function` used by an `EpisodicMemoryMechanism`, and takes all of the same arguments.

.. _EMComposition_Creation:

Creation
--------

.. _EMComposition_Execution:

Execution
---------

.. _EMComposition_Examples:

Examples
--------

.. _EMComposition_Class_Reference:

Class Reference
---------------
"""
import numpy as np

from psyneulink.core.compositions.composition import Composition, CompositionError
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.keywords import EM_COMPOSITION

__all__ = [
    'EMComposition'
]

class EMCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EMComposition(Composition):
    """
    Subclass of `Composition` that implements the functions of an `EpisodicMemoryMechanism` in a differentiable form.

    All of the arguments of the `ContentAddressableMemory` `Function` can be specified in the constructor for the
    EMComposition.  In addition, the following arguments can be specified:

    TODO:
    - DECAY WEIGHTS BY:
      ? 1-SOFTMAX / N (WHERE N = NUMBER OF ITEMS IN MEMORY)
      or
         1/N (where N=number of items in memory, and thus gets smaller as N gets
         larger) on each storage (to some asymptotic minimum value), and store the new memory to the unit with the
         smallest weights (randomly selected among “ties" [i.e., within epsilon of each other]), I think we have a
         mechanism that can adaptively use its limited capacity as sensibly as possible, by re-cycling the units
         that have the least used memories.
    - TEST ADPATIVE TEMPERATURE (SEE BELOW)
    - ADD ALL ARGS FOR CONTENTADDRESSABLEMEMORY FUNCTION TO INIT, AND MAKE THEM Parameters

    Arguments
    ---------

    learning_rate : float : default 0.001
        the learning rate passed to the optimizer if none is specified in the learn method of the EMComposition.

    disable_learning : bool: default False
        specifies whether the EMComposition should disable learning when run in `learning mode
        <Composition.learn>`.

    memory_capacity : int : default 1000
        specifies the number of items that can be stored in the EMComposition's memory.


    Attributes
    ----------


    """

    componentCategory = EM_COMPOSITION
    class Parameters(Composition.Parameters):
        learning_rate = Parameter(.001, fallback_default=True)
        losses = Parameter([])
        pytorch_representation = None
    @check_user_specified
    def __init__(self,
                 memory_shape=(2,1),
                 field_weights=(1,0),
                 learning_rate=None,
                 disable_learning=False,
                 memory_capacity=1000,
                 name="EM_composition"):

        self.memory_shape = memory_shape
        self.field_weights = field_weights
        self.learning_rate = learning_rate # FIX: MAKE THIS A PARAMETER
        self.memory_capacity = memory_capacity # FIX: MAKE THIS A READ-ONLY PARAMETER
        self.disable_learning = disable_learning


    def _create_components(self):
        match_layer = self._create_match_layer()
        weighting_layer = self._create_weighting_layer()
        retrieval_layer = self._create_retieval_layer()
        return [match_layer, weighting_layer, retrieval_layer]

    def _create_match_layer(self):
        """Create layer that computes the similarity between the input and each item in memory."""
        # numpy function that computes the similarity between the input and each item in memory
        # all entries in field_weights are the same:
        if len(self.field_weights) == 1 or

            self.field_weights = np.ones_like(self.memory_shape)
        input = np.zeros_like(self.memory_shape)


    def _create_weighting_layer(self):
        """Create layer that computes the weighting of each item in memory."""
        pass

    def _create_retieval_layer(self):
        """Create layer that retrieves an item in memory based on match_layer."""
        pass

    def softmax_temperature(self, values, epsilon=1e-8):
        """Compute the softmax temperature based on length of vector and number of (near) zero values."""
        n = len(values)
        num_zero = np.count_nonzero(values < epsilon)
        num_non_zero = n - num_zero
        gain = 1 + np.exp(1/num_non_zero) * (num_zero/n)
        return gain

    def execute(self, **kwargs):
        """Set input to weights of Projection to match_layer."""
        super().execute(**kwargs)
