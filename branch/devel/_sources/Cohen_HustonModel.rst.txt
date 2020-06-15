Stroop GRAIN model (Cohen & Huston, 1994)
================================================================
`"Progress in the use of interactive models for understanding attention and performance. In C. Umilta & M. Moscovitch(Eds.)" <https://books.google.com/books?hl=de&lr=&id=cOAmbT3ORLcC&oi=fnd&pg=PA453&dq=cohen+%26+huston+1994&ots=nguFNK-b7W&sig=DdKsbgaUPawQbckBjMU-52ODt1M#v=onepage&q&f=false>`_

Overview
--------

The model aims to capute top-down effects of selective attention and the bottom-up effects of attentional capture
for the Stroop task.
The model first aimed to simulate the standard Stroop effect, which had been modeled before in a feed forward model
(Cohen, Dunbar, McClalland 1990). In a second step, the model aimed to capute the findings by Glaser & Glaser (1982).
Glaser & Glaser (1982) vaired the stimulus onset asynchrony (SOA) by presenting the irrelevant stimulus of the Stroop task
in different intervals before, at the same time, or after the presentation of the relevant stimulus. They found that the
irrelevant stimulus caused interference (measured in increased reaction times) only when the irrelevant stimulus was
presented in the range from 100ms before and 100ms after the presentation of the relevant stimulus. The effect was
observed for congruent, neutral, and incongruent trials.

Here we implement the GRAIN model which captures the Stroop effect (left), and plot the figure the GRAIN model produces
for different SOA (right). The Figure below shows the simulated reaction times.
Note that we used different regression coefficients to produce these plots.

.. _GRAIN_STROOP_EFFECT_Fig:

.. figure:: _static/GRAIN_STROOP_EFFECT.svg
   :target: _static/GRAIN_STROOP_EFFECT.svg
   :figwidth: 45 %
   :align: left
   :alt: Cohen&Huston plot produced by PsyNeuLink

.. _Horserace_Fig:

.. figure:: _static/Horserace.svg
   :target: _static/Horserace.svg
   :figwidth: 45 %
   :align: right
   :alt: Cohen&Huston SOA produced by PsyNeuLink



The Model
---------

Two hidden layers (for colors and words each), a task layer, and a response layer comprise the system of the model,
all of which are implemented as a `RecurrentTransferMechanism` and mutual inhibition weights. A response is made when one
of the two units in the response layer hit a specified threshold.
The hidden layers receive their inputs by two input layers. The layers are connected with predetermined weights (note
that in the previous Stroop model by Cohen, Dunbar, McClalland (1990) the weights were learned).
Below the Graph of the model is shown.

.. _GRAIN_STROOP_GRAPH_Fig:

.. figure:: _static/GRAIN_STROOP_GRAPH.svg
   :target: _static/GRAIN_STROOP_GRAPH.svg
   :figwidth: 75 %
   :align: center
   :alt: Cohen&Huston System Graph

Composition
~~~~~~~~~~~

**COLOR INPUT LAYER**:  a `TransferMechanism` with **size** = 3 (one element for the input to each color in the
*HIDDEN COLOR LAYER*, respectively), and assigned a `Linear` function with **slope** = 1.0 and **intercept** = 0.0.

**WORD INPUT LAYER**:  a `TransferMechanism` with **size** = 3 (one element for the input to each word in the
*HIDDEN WORD LAYER*, respectively), and assigned a `Linear` function with **slope** = 1.0 and **intercept** = 0.0.

**TASK INPUT LAYER**:  a `TransferMechanism` with **size** = 2 (one element for the input to each task in the
*TASK LAYER*, respectively), and assigned a `Linear` function with **slope** = 1.0 and **intercept** = 0.0.

**HIDDEN COLOR LAYER**: a `RecurrentTransferMechanism` Mechanism of **size** = 3 (one element each for the color units),
and assigned a `Logistic` Function with a bias = 4.0 and intercept = 0.0.  Each element is connected to every other
element by mutually inhibitory connections with a weight specified by **hetero** = -2.0.  An integrator mechanism is
specified by setting the **integrator_mode** = `True` and **smoothing_factor** = 0.1.

**HIDDEN WORD LAYER**: a `RecurrentTransferMechanism` specified as the *HIDDEN COLOR LAYER* with **size** = 3,
a `Logistic` Function with a **bias** = 4.0 and **intercept** = 0.0, mutually inhibitory connections with a weight specified by
**hetero** = -2.0, **integrator_mode** = `True` and **smoothing_factor** = 0.1..

**RESPONSE LAYER**: a `RecurrentTransferMechanism` specified as the *HIDDEN COLOR LAYER* with the only difference of
changing the bias to 0 in the `Logistic` Function, and the size of 2.

**TASK LAYER**: a `RecurrentTransferMechanism` specified as the *RESPONSE LAYER*.

**PROJECTIONS**:  The weights of the  network are implemented as `MappingProjections <MappingProjection>`.
The projections for colors are as follows:
The `matrix <MappingProjection.matrix>` parameter for the one from the *INPUT COLOR LAYER* to the *HIDDEN COLOR LAYER*
uses a numpy array with a value of 1.0 for the diagonal elements and a value of 0.0 for the off-diagonal elements;
the one from the *HIDDEN COLOR LAYER* to the *RESPONSE LAYER* uses a numpy array with 1.5 for the diagonal elements and
0 for the off-diagonal elements; the one from the *HIDDEN COLOR LAYER* to the *TASK LAYER* uses a numpy array with 4.0
for the diagonal elements and 0 for the off-diagonal elements.
The projections for words are as follows:
The `matrix <MappingProjection.matrix>` parameter for the one from the *INPUT WORD LAYER* to the *HIDDEN WORD LAYER*
uses a numpy array with a value of 1.0 for the diagonal elements and a value of 0.0 for the off-diagonal elements;
the one from the *HIDDEN WORD LAYER* to the *RESPONSE LAYER* uses a numpy array with 2.5 for the diagonal elements and
0 for the off-diagonal elements; the one from the *HIDDEN WORD LAYER* to the *TASK LAYER* uses a numpy array with 4.0
for the diagonal elements and 0 for the off-diagonal elements.
During initialization the *RESPONSE LAYER* is silenced by setting the `matrix <MappingProjection.matrix>` from
*HIDDEN WORD LAYER* and *HIDDEN COLOR LAYER* to all 0. The *TASK LAYER* receives input from the *TASK INPUT LAYER*
during initialization.


.. _Cohen&Huston_Execution:

Execution
---------
The stimulus presentation sequence is split into 2 periods of execution, a settling period and stimulus presentation
period. During the settling period, the *TASK LAYER* receives an input of 1 for the unit which represent the given task
from the  *TASK INPUT LAYER*. The other unit receives an input of 0. This allows the system to settle into a ready state
for the given task.
After the settling period the `matrix <MappingProjection.matrix>` is changed from all 0 to 1.5 on the diaginals and 0 on
the off-diagonals for the *HIDDEN COLOR LAYER* to the *RESPONSE LAYER* and to 2.5 on the diagonals and 0 on the
off-diagonals for the *HIDDEN WORD LAYER* to the *RESPONSE LAYER*.

During the stimulus presentation period, the *TASK LAYER* keeps receiving input from *TASK INPUT LAYER*. In addition,
the *HIDDEN COLOR LAYER* and the *HIDDEN WORD LAYER* receive inputs from the *COLOR
INPUT LAYER* and *WORD INPUT LAYER*,
respectively.
The system is executed until one of the two response units hit a threshold. This is specified in the
`termination_processing` parameter in the `run` command.


The `log` function is used to record the output values of the *RESPONSE LAYER*. The cyles until a threshold is reached
is used for the regression we used here to transfer cycles to milliseconds. We used the regression `cycles*5 + 115` to
produce the plot below.

Execution horse race Figure
---------------------------
To reproduce the horse race figure, we run the system for different stimulus onset asynchronies (SOA).
Depending on a positive or negative SOA the system is executed in different ways. For negative SOA, the system is run
for the settling period first. Then, the system is run with the same input as in the settling period for different amount
of times steps, depending on the SOA. For these two runs, the weights from the *RESPONSE LAYER* to the
*HIDDEN COLOR LAYER* and the *HIDDEN WORD LAYER* are set to 0, since a response is silenced.
Now these weights get tunred on with their values as mentioned above.
The system is run a thrid time for the period with the irrelevant stimulus presented to the *WORD HIDDEN LAYER*,
until the time for the relevant stimulus to be presented.
For a forth and final time the system is run with both stimuli presented until one of the two units in the
*RESPONSE LAYER* hits threshold.
For the positive SOA, the system is run with the initial input to the *TASK LAYER* for the settling period, and
a second time for the 100 cycles, since these amount of cylces represent the time for negative SOA. Then, the weights
are turned on again and the system is run either for a certain amount of trials, of until the threshold is reached.
stimulus is turned on.

PLEASE NOTE:
-----------
Note that this implementation is slightly different than what was originally reported. The integration rate was set to
0.1 instead of 0.01. Noise was turned of to better understand the core processes, and not having to deal with several
runs, averaging these runs and plotting standard errors for these averages (which depend on the noise and amount of
runs). Finally,  different SOA and a different linear regression formula was used to reproduce the figures.
Regardless of these modifications, we were able to reproduce the core Figures with the same weights of the
original model.

Script: :download:`Download Cohen_Huston1994.py <../../psyneulink/library/models/Cohen_Huston1994.py>`

Script: :download:`Download Cohen_Huston1994_horse_race.py <../../psyneulink/library/models/Cohen_Huston1994_horse_race.py>`
