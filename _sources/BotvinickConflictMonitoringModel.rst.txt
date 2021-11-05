
Conflict Monitoring and Cognitive Control (Botvinick et al., 2001)
==================================================================
`"Conflict Monitoring and Cognitive Control" <https://www.ncbi.nlm.nih.gov/pubmed/11488380>`_

Overview
--------
This implements a model concerning the role of the `anterior cingulate cortex (ACC)
<https://en.wikipedia.org/wiki/Anterior_cingulate_cortex>`_ in situations calling for their
involvement. The demand for control is evaluated by monitoring for conflicts in information processing.
The model is supported by data concerning the anterior cingulate cortex, a brain area involved in cognitive control,
which also appears to respond to the occurrence of conflict.
This script implements the system and plot from Figure 1 of the conflict monitoring and cognitive control paper.


.. _conflict_PNL_Fig:

.. figure:: _static/conflict_PNL.svg
   :figwidth: 45 %
   :align: left
   :alt: Botvinick et al., 2001 plot produced by PsyNeuLink

.. _conflict_monitoring_MATLAB_Fig:

.. figure:: _static/conflict_monitoring_MATLAB.svg
   :figwidth: 45 %
   :align: right
   :alt: Botvinick et al., 2001 plot produced by MATLAB


The Model
---------
The model is built as the bidirectional Stroop model by Cohen & Huston (1994): stimulus information in two hidden
layers (color, word & neutral), and task information (color naming or word reading) are connected with each other in a
bidirectional way. The response layer receives inputs from both hidden layers. A graph of the model is shown below.

.. _conflict_monitoring_Graph:

.. figure:: _static/conflict_monitoring.svg
   :figwidth: 75 %
   :align: center
   :alt: Botvinick System Graph

Network System
~~~~~~~~~~~~~~
**COLOR INPUT LAYER**:  a `TransferMechanism` with **size** = 3 (one unit for the input of one color, respectively
here blue & green), and assigned a `Linear` function with **slope** = 1.0 and **intercept** = 0.0.

**WORD INPUT LAYER**:  a `TransferMechanism` with **size** = 3 (one unit for the input of one word, respectively,
here blue & green), and assigned a `Linear` function with **slope** = 1.0 and **intercept** = 0.0.

**TASK INPUT LAYER**:  a `TransferMechanism` with **size** = 2 (one unit specified with a task
value of one, the other element set to zero), and assigned a `Linear` function with **slope** = 1.0 and **intercept** = 0.0.

**COLOR HIDDEN LAYER**: a `RecurrentTransferMechanism` with **size** = 3 (one element for each of the two colors, one
element for the neutral color and assigned a `Logistic` function with **gain** = 4.0 and **bias** = 1.0.
The **integrator_mode** = `True` and **smoothing_factor** = 0.01. Both units receive mutually inhibitory weights
(**hetero** = -2).

**WORD HIDDEN LAYER**: a `RecurrentTransferMechanism` with **size** = 3 (one element for each of the two colors, one
element for the neutral color and assigned a `Logistic` function with **gain** = 4.0 and **bias** = 1.0.
The **integrator_mode** = `True` and **smoothing_factor** = 0.01. Both units receive mutually inhibitory weights
(**hetero** = -2).

**TASK DEMAND LAYER**: a `RecurrentTransferMechanism` with **size** = 2 (one element for each of the two tasks, and
assigned a `Logistic` function with **gain** = 1.0 and **bias** = 0.0. The **integrator_mode** = `True`
and **smoothing_factor** = 0.01. Both units receive mutually inhibitory weights (**hetero** = -2).

**RESPONSE LAYER**: a `RecurrentTransferMechanism` with **size** = 2 (one element for each of the two responses, and
assigned a `Logistic` function with **gain** = 1.0 and **bias** = 0.0. The **integrator_mode** = `True`
and **smoothing_factor** = 0.01. Both units receive mutually inhibitory weights (**hetero** = -2).

**PROJECTIONS**:  The weights of the  network are implemented as `MappingProjections <MappingProjection>`.
The `matrix <MappingProjection.matrix>` parameter from the *COLOR INPUT_LAYER*, the *WORD INPUT_LAYER*, and the
*BIAS INPUT_LAYER* to the *COLOR HIDDEN LAYER* and *WORD HIDDEN LAYER* are all set with a numpy array with a value of
1.0 for the diagonal elements and all off-diagonal elements are set to 0.
The color hidden layer projects to the *TASK LAYER* with a numpy array with a value of 4.0 on the the first column, and
0.0 on the second column, and receive inputs from the *TASK LAYER* with a numpy array with a value of 4.0 on the first row
and a value of 0.0 in the second row.
The word hidden layer projects to the *TASK LAYER* with a numpy array with a value of 4.0 on the the second column, and
0.0 on the first column, and receive inputs from the *TASK LAYER* with a numpy array with a value of 4.0 on the second row
and a value of 0.0 in the first row.
The *RESPONSE LAYER* receives projections from two layers:
the *COLOR HIDDEN LAYER* with a numpy array with a value of 1.5 on the diagonal elements and 0.0 on the off-diagonal
elements.
The *WORD HIDDEN LAYER* with a numpy array with a value of 2.5 on the diagonal elements and 0.0 on the off-diagonal
elements.

Execution
---------
All units are set to zero at the beginning of the simulation. Each simulation run starts with a settling
period of 500 time steps. Then the stimulus is presented for 1000 time steps and is presented by setting the input
units to 1.0 for a given trial. Conflict is computed on the `output_port` of the *RESPONSE LAYER*. The figure plots
conflict over one trial for each of the three conditions.
The `log` function is used to record the output values of *RESPONSE LAYER*. These values are used to produce
the plot of the Figure.

Please note:
------------
Note that this script implements a slightly different Figure than in the original Figure in the paper.
However, this implementation is identical with a plot we created with an old MATLAB code which was used for the
conflict monitoring simulations.

Script: :download:`Botvinick_conflict_monitoring_model.py <../../psyneulink/library/models/Botvinick_conflict_monitoring_model.py>`
