Proactive Control & Task Control: A Stroop Model (Kalanthroff et al., 2018)
================================================================
`"Task Conflict and Proactive Control: A Computational Theory of the Stroop Task" <https://www.ncbi.nlm.nih.gov/m/pubmed/25257710/>`_

Overview
--------
Reaction times are usually faster on congruent trials than neutral trials. This finding
has been refered to as the facilitation effect. Recenty Goldfarb and Henik (2007) showed that when proactive control
(Braver, 2012) is low, subjects responded faster on neutral trials vs congruent trials, which has been refered to as
the reverse facilitation effect.

In addition, brain imaging studies showed higher activation in the anterior cingulate cortex (ACC), an area
associated with conflict, on congruent trials than neutral trials.

Here, the authors hypothesize that task conflict on congruent trials, but not neutral trials can account for the
reverse facilitation effect, under low proactive control.

Conflict is measured on the task demand layer and not on the response layer as in previous models
(e.g. see Botvinick et al., 2001). Task conflict directly inhibits the response layer, preventing a fast response when
task conflict exists. The activation of each unit in every layers of the neural network model is plotted for low
proactive activation. The PsyNeuLink model is identical to the plot of Figure 6b generated with MATLAB (see both Figures
below). On the left the PsyNeuLink Figure is plotted. On the right the MATLAB figure is plotted.

.. _PCTC_0_025_PNL_Fig:

.. figure:: _static/PCTC_0_025_PNL.svg
   :figwidth: 45 %
   :align: left
   :alt: Kalanthroff et al., 2018 plot produced by PsyNeuLink

.. _PCTC_0_025_MATLAB_FIGURE_Fig:

.. figure:: _static/PCTC_0_025_MATLAB_FIGURE.svg
   :figwidth: 45 %
   :align: right
   :alt: Kalanthroff et al., 2018 plot produced by MATLAB


The Model
---------
The model is built as the bidirectional Stroop model by Cohen & Huston (1994): stimulus information in two hidden
layers (color & word), and task information (color naming or word reading) are connected with each other with
bidirectional weights. The response layer receives input weights from both hidden layers, but does not project back to
these the two hidden layers. This is different to the previous Cohen & Huston (1994) model).
Conflict is computed on the OutPort of the task layer, and projected from the task layer to the response layer.
If task conflict exists, prevents the response layer from a fast response.
The conflict projection is the main difference from the Kalanthroff et al., (2018) model to the Cohen & Huston (1994)
model. A graph of the model is shown below.

.. _Kalanthroff2016_System_Graph:

.. figure:: _static/PCTC_Graph.svg
   :figwidth: 75 %
   :align: center
   :alt: Kalanthroff System Graph

Network System
~~~~~~~~~~~~~~
**COLOR INPUT LAYER**:  a `TransferMechanism` with **size**\ =2 (one unit for the input of one color, respectively
here blue & green), and assigned a `Linear` function with **slope**\ =1.0 and **intercept**\ =0.0.

**WORD INPUT LAYER**:  a `TransferMechanism` with **size**\ =2 (one unit for the input of one word, respectively,
here blue & green), and assigned a `Linear` function with **slope**\ =1.0 and **intercept**\ =0.0.

**BIAS INPUT LAYER**:  a `TransferMechanism` with **size**\ =2 (one unit for the bias of one of the hidden layers,
which is the same in this model), and assigned a `Linear` function with **slope**\ =1.0 and **intercept**\ =0.0.

**PROACTIVE CONTROL INPUT LAYER**:  a `TransferMechanism` with **size**\ =2 (one unit specified with a proactive control
value, the other one set to zero), and assigned a `Linear` function with **slope**\ =1.0 and **intercept**\ =0.0.

**COLOR HIDDEN LAYER**: a `RecurrentTransferMechanism` with **size**\ =2 (one element for each of the two colors, and
assigned a `Logistic` function with **gain**\ =4.0 and **bias**\ =1.0. The **integrator_mode**\ =\ `True`
and **smoothing_factor**\ =0.03. Both units receive mutually inhibitory weights (**hetero**\ =-2). A python function that
sets the output of the `Logistic` function to 0 when it receives 0 as an input is specified on the `output_ports`. It
simply subtracts 0.018 from the output of a logistic function and if this leads to a value below 0,
outputs a 0 as a minimum value.

**WORD HIDDEN LAYER**: a `RecurrentTransferMechanism` with **size**\ =2 (one element for each of the two words, and
assigned a `Logistic` function with **gain**\ =4.0 and **bias**\ =1.0. The **integrator_mode**\ =\ `True`
and **smoothing_factor**\ =0.03. Both units receive mutually inhibitory weights (**hetero**\ =-2). A python function that
sets the output of the `Logistic` function to 0 when it receives 0 as an input is specified on the `output_ports`. It
simply subtracts 0.018 from the output of a logistic function and if this leads to a value below 0,
outputs a 0 as a minimum value.

**TASK DEMAND LAYER**: a `RecurrentTransferMechanism` with **size**\ =2 (one element for each of the two tasks, and
assigned a `Logistic` function with **gain**\ =4.0 and **bias**\ =1.0. The **integrator_mode**\ =\ `True`
and **smoothing_factor**\ =0.03. Both units receive mutually inhibitory weights (**hetero**\ =-2). A python function that
sets the output of the `Logistic` function to 0 when it receives 0 as an input is specified on the `output_ports`. It
simply subtracts 0.018 from the output of a logistic function and if this leads to a value below 0,
outputs a 0 as a minimum value. A second OutputPort is specified that computes the conflict between the two task
units.

**WORD HIDDEN LAYER**: a `RecurrentTransferMechanism` with **size**\ =2 (one element for each of the two responses, and
assigned a `Logistic` function with **gain**\ =4.0 and **bias**\ =1.0. The **integrator_mode**\ =\ `True`
and **smoothing_factor**\ =0.03. Both units receive mutually inhibitory weights (**hetero**\ =-2). A python function that
sets the output of the `Logistic` function to 0 when it receives 0 as an input is specified on the `output_ports`. It
simply subtracts 0.018 from the output of a logistic function and if this leads to a value below 0,
outputs a 0 as a minimum value.

**PROJECTIONS**:  The weights of the  network are implemented as `MappingProjections <MappingProjection>`.
The `matrix <MappingProjection.matrix>` parameter from the *COLOR INPUT_LAYER*, the *WORD INPUT_LAYER*, and the
*BIAS INPUT_LAYER* to the *COLOR HIDDEN LAYER* and *WORD HIDDEN LAYER* are all set with a numpy array with a value of
1.0 for the diagonal elements and a value of 0.0 for the off-diagonal elements.
The two hidden layers both project to the *TASK LAYER* with a numpy array with a value of 2.0 on the diagonal elements
and 0.0 on the off-diagonals, and receive inputs from the *TASK LAYER* with a numpy array with a value of 1.0 for the
diagonal elements and a value of 0.0 for the off-diagonal elements. The *RESPONSE LAYER* receives projections from
three  different layers:
The *COLOR HIDDEN LAYER* with a numpy array with a value of 1.5 on the diagonal elements and 0.0 on the off-diagonal
elements.
The *WORD HIDDEN LAYER* with a numpy array with a value of 2.5 on the diagonal elements and 0.0 on the off-diagonal
elements.
The *TASK LAYER* with a numpy array with a value of length 2 with both elements set to -1. The second OutputPort
of the *TASK LAYER* is specified as the sender and the *RESPONSE LAYER* is specified as the receiver.

Execution
---------
All units are set to zero at the beginning of the simulation. Each simulation run starts with a settling
period of 500 time steps. Then the stimulus is presented for the remaining duration of the trial.
During the settling period, the proactive control unit sends input to the color-task-demand unit.
The activations of all units are updated on each time step until one of the response units reaches the threshold,
which is set to 0.7. The `log` function is used to record the output values of the two hidden layers, the task layer,
and the response layer. These values are used to produce the plot of the Figure.

Script: :download:`Download Kalanthroff_PCTC_2018.py <../../psyneulink/library/models/Kalanthroff_PCTC_2018.py>`
