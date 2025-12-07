# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# *********************************************  LossMechanism ***************************************************

"""

Contents
--------

  * `LossMechanism_Overview`
  * `LossMechanism_Creation`
  * `LossMechanism_Structure`
  * `LossMechanism_Execution`
  * `LossMechanism_Example`
  * `LossMechanism_Class_Reference`


.. _LossMechanism_Overview:

Overview
--------

A LossMechanism is a subclass of `ComparatorMechanism` that receives two inputs (a sample and a target), compares
them using its `function <LossMechanism.function>`, and places the calculated discrepancy between the two in its
*OUTCOME* `OutputPort <LossMechanism.output_port>`.  It is used to comput the loss (error) signal for learning in
in the forward method of an `AutodiffComposition`.

.. _LossMechanism_Creation:

Creating a LossMechanism
------------------------------

LossMechanisms are created automatically when specified in the constructor of an `AutodiffComposition`.

It is important to recognize that the value of the *SAMPLE* and *TARGET* InputPorts must have the same length and
type, so that they can be compared using the LossMechanism's `function <LossMechanism.function>`. They use format
of the OutputPorts (or those of the Mechanisms) specified in the **sample** and **target**  arguments, respectively,
and the `MappingProjection` to each preserves those dimensions. Therefore, the OutputPorts (or Mechanisms) specified
in the **sample** and **target** arguments must have values of the same length and type. If the **input_ports**
argument is used, then *both* the *SAMPLE* and *TARGET* InputPorts must be specified. Any of the formats for
`specifying InputPorts <InputPort_Specification>` can be used in the argument, however the number and use of these
must conform to the format of the `variable <Function.variable>` for `function <LossMechanism.function>` of
the `LossMechanism` (see `LossMechanism_Structure`). If values are assigned for the InputPorts, they must be of equal
length and type. Their types must also be compatible with the value of the OutputPorts specified in the **sample**
and **target** arguments and, again,of the same length.

.. _LossMechanism_Structure:

Structure
---------

A LossMechanism has, by default, two `input_ports <LossMechanism.input_ports>`, each of which receives a
`MappingProjection` from a corresponding OutputPort specified in the **sample** and **target** arguments of its
constructor. The InputPorts are listed in the Mechanism's `input_ports <LossMechanism.input_ports>` attribute
and named, respectively, *SAMPLE* and *TARGET*. The OutputPorts from which they receive their projections (specified
in the **sample** and **target** arguments) are listed in the Mechanism's `sample <LossMechanism.sample>` and
`target <LossMechanism.target>` attributes as well as in its `monitor <LossMechanism.monitor>` attribute.
The LossMechanism's `function <LossMechanism.function>` compares the value of the sample and target InputPorts
to  compute the loss. By default, it uses the `loss_spec <AutodiffComposition.loss_spec>` attribute of the
`AutodiffComposition` in which it is contained. However, other `LOSS` specifications or torch functions can be used.
The latter must be able to take the number of arrays with the same format as its inputs, and must `detach
<https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_ the tensor(s) used as the target values in
computing the loss. The result is assigned as the value of the Loss Mechanism's *OUTCOME* (`primary
<OutputPort_Primary>`) OutputPort, and used in the `AutodiffComposition` to compute the loss for learning.

.. _LossMechanism_Execution:

Execution
---------

When a LossMechanism is executed, it updates its input_ports with the values of the OutputPorts (or Mechanisms)
specified in its **sample** and **target** arguments, and then uses its `function <LossMechanism.function>` to compute
the loss, `detaching <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_ the target value(s). The
result is assigned to the `value <Mechanism_Base.value>` of its *OUTCOME* `output_port <LossMechanism.output_port>`,
and as the first item of the Mechanism's `output_values <LossMechanism.output_values>` attribute.

.. _LossMechanism_Class_Reference:

Class Reference
---------------

"""

import torch

from collections.abc import Iterable

import numpy as np
from beartype import beartype

from psyneulink._typing import Optional, Union

from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base, MechanismError
from psyneulink.core.components.functions.nonstateful.objectivefunctions import LossFunction
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.keywords import (
    Loss, LOSS_MECHANISM, NAME, PREFERENCE_SET_NAME, PROJECTION, SAMPLE, TARGET)
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.utilities import NumericCollections
from psyneulink.core.globals.context import handle_external_context

__all__ = ['LossMechanism', 'LossMechanismError']

class LossMechanismError(MechanismError):
    pass


class LossMechanism(ComparatorMechanism):
    """
    LossMechanism(                      \
        sample,                         \
        target,                         \
        input_ports=[SAMPLE,TARGET],    \
        loss=None,                      \
        output_ports=OUTCOME)

    Subclass of `ComparatorMechanism` that computes the loss (error) between a `sample <LossMechanism.sample>`
    and a `target <LossMechanism.target>` used for training in an `AutodiffComposition`.

    Arguments
    ---------

    sample : OutputPort, Mechanism, value, or string
        specifies the value to for which to compute the `loss <LossMechanism.loss>` with respect to the
        `target <LossMechanism.target>`.

    target :  OutputPort, Mechanism, value, or string
        specifies the value with respect to which the `loss <LossMechanism.loss>` is computed for the
        `sample <LossMechanism.sample>`.

    input_ports :  List[InputPort, value, str or dict] or Dict[] : default [SAMPLE, TARGET]
        specifies the names and/or formats to use for the values of the `sample <LossMechanism.sample>`
        and `target <LossMechanism.target>` InputPorts; by default they are named *SAMPLE* and *TARGET*,
        and their formats match the value of the OutputPorts specified in the **sample** and **target**
        arguments, respectively (see `LossMechanism_Structure` for additional details).

    loss :  Loss or PyTorch loss function : default torch.nn.MSELoss(reduction='mean')
        specifies the `function <Loss.function>` used to compute the loss for
        `sample <LossMechanism.sample>` with respect to the `target <LossMechanism.target>`.

    function : function or method
        specifies a function used  to compute the loss for the `sample <LearningMechanism.sample>` with
        respect to the `target <LearningMechanism.target>`. It can be any function that takes two arrays
        as input arguments (the sample and target values) and returns a scalar, including a `torch.nn loss
        function <https://pytorch.org/docs/stable/nn.html#loss-functions>`_. It must also `detach
        <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_ the tensor used as the target
        values in computing the loss.  If both **loss** and **function** are specified, an error is raisedIf
        neither is specified, the default is determined by the `loss_spec <AutodiffComposition.loss_spec>`
        attribute of the `AutodiffComposition` in which the LossMechanism is contained.


    Attributes
    ----------

    sample : OutputPort
        determines the value for which the `loss <LossMechanism.loss>` is computed with respect to the
        `target <LossMechanism.target>`.

    target : OutputPort
        determines the value with respect to which the `loss <LossMechanism.loss>` is computed for the
        `sample <LossMechanism.sample>`.

    input_ports : ContentAddressableList[InputPort, InputPort]
        contains the two InputPorts named, by default, *SAMPLE* and *TARGET*, each of which receives a
        `MappingProjection` from the OutputPorts referenced by the `sample <LossMechanism.sample>` and
        `target <LossMechanims.target>` attributes (see `LossMechanism_Structure` for additional details).

    loss :  Loss or PyTorch loss function : default torch.nn.MSELoss(reduction='mean')
        specifies the `function <Loss.function>` used to compute the loss for `sample <LossMechanism.sample>`
        with respect to the `target <LossMechanism.target>`.

    function : function or method
        used to compute the loss for the `sample <LearningMechanism.sample>` with respect to the
        `target <LearningMechanism.target>`; determined either by the **loss** or **function*** argument
        to the constructor.

    output_port : OutputPort
        contains the `primary <OutputPort_Primary>` OutputPort of the LossMechanism; the default is
        its *OUTCOME* OutputPort, the value of which is equal to the `value <LossMechanism.value>`
        attribute of the LossMechanism.

    output_ports : ContentAddressableList[OutputPort]
        contains, by default, only the *OUTCOME* (primary) OutputPort of the LossMechanism.

    output_values : 2d np.array
        contains one item that is the value of the *OUTCOME* OutputPort.


    """
    componentType = LOSS_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'LossCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class Parameters(ComparatorMechanism.Parameters):
        """
            Attributes
            ----------

                loss
                    see `loss <LossMechanism.loss>`

                    :default value: `Loss.MSE`
                    :type: `Loss`
        """
        loss = Parameter(Loss.MSE, stateful=False, loggable=False)

        def _validate_loss(self, function):

            def is_loss_spec_or_torch_loss(function):
                # Check for Loss spec
                from psyneulink.core.globals.keywords import Loss
                import torch.nn

                if isinstance(function, Loss):
                    return True
                # Check for torch.nn loss function instance
                if isinstance(function, torch.nn.modules.loss._Loss):
                    return True
                # Check for torch.nn loss class
                if isinstance(function, type) and issubclass(function, torch.nn.modules.loss._Loss):
                    return True
                return False

            if function and not is_loss_spec_or_torch_loss(function):
                return f"must be a torch.nn loss function."
            return None

    @handle_external_context()
    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 sample: Optional[Union[OutputPort, Mechanism_Base, dict, NumericCollections, str]] = None,
                 target: Optional[Union[OutputPort, Mechanism_Base, dict, NumericCollections, str]] = None,
                 # TEACHER_TARGET BREADCRUMB: INSTANTIATE TYPE CHECKING BELOW:
                 # function: Optional[Union[torch.nn]] = None,
                 function = None,
                 loss = None,
                 output_ports:Optional[Union[str, Iterable]] = None,
                 params=None,
                 name=None,
                 prefs: Optional[ValidPrefSet] = None,
                 context=None,
                 **kwargs
                 ):

        if function:
            if loss is not None:
                raise LossMechanismError(f"LossMechanism '{self.name}': 'function' and 'loss' arguments "
                                         f"are mutually exclusive; only one can be specified.")
        else:
            function = LossFunction(loss=loss)

        super().__init__(default_variable=default_variable,
                         sample=sample,
                         target=target,
                         function=function,
                         output_ports=output_ports,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

        self.parameters.sample._set(sample, context)
        self.parameters.target._set(target, context)


