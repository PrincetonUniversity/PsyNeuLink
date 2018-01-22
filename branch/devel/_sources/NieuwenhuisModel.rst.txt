Nieuwenhuis et al. (2005)
=========================
`"The Role of the Locus Coeruleus in Mediating the Attentional Blink: A Neurocomputational Theory" <https://research.vu.nl/ws/files/2063874/Nieuwenhuis%20Journal%20of%20Experimental%20Psychology%20-%20General%20134(3)-2005%20u.pdf>`_

Overview
--------

This model implements a hypothesis concerning the role of the `locus coeruleus (LC)
<http://www.scholarpedia.org/article/Locus_coeruleus>`_ in mediating the `attentional blink
<http://www.scholarpedia.org/article/Attentional_blink>`_. The attentional blink refers to the temporary impairment
in perceiving the second of two targets presented in close temporal proximity.

In the `attentional blink paradigm <http://www.scholarpedia.org/article/Attentional_blink>`_, each trial consists of a
rapidly presented sequence of stimuli, one of which is marked as a target (e.g., by being a different color, or by being
a digit rather than a letter).  The participant is asked to observe the sequence, then identify the target (T1) as well
as the stimulus that immediately follows it (T2).  A consistent finding is that accuracy with which both T1 and T2 are
identified depends on the lag between them. In particular, accuracy is high when T2 is presented *immediately* after T1,
drops when T2 is presented between 200 and 300ms after the onset of T1, and is restored at delays longer than 400ms.

The model described by Nieuwenhuis et al. (2005) suggests that these findings can be explained by phasic activation
of LC in response to T1, and the concomitant effects of norepinephrine (NE) release on neural gain, followed by
refractoriness of the LC/NE system after a phasice response.  The model demonstrates that accuracy in identifying T2
depends on:
   * whether T1 was accurately identified;
   * the lag between T1 and T2;
   * the `mode of the LC <LCControlMechanism_Modes_Of_Operation>`
     (phasic or tonic -- see `<https://www.ncbi.nlm.nih.gov/pubmed/8027789>`_).

The Figure below shows the behavior of the model for a single execution of a trial with a lag of 200ms (without noise),
corresponding to the conditions reported in Figure 3 of Nieuwenhuis et al. (2005; averaged over 1000 executions with
noise).


.. _Nieuwenhuis2005_PsyNeuLink_Fig:

.. figure:: _static/Nieuwenhuis2005_psyneulink.svg
   :figwidth: 45 %
   :align: left
   :alt: Nieuwenhuis et al. 2005 plot produced by PsyNeuLink

.. _Nieuwenhuis2005_MATLAB_Fig:

.. figure:: _static/Nieuwenhuis2005_MATLAB.svg
   :figwidth: 45 %
   :align: left
   :alt: Nieuwenhuis et al. 2005 plot produced by MATLAB

The Model
---------

The model is comprised of two subsystems: a behavioral network, in which stimulus information feeds forward from an
input layer, via a decision layer, to a response layer;  and an LC subystem that regulates the gain of the units in
the decision and response layers.  Each of the layers in the behavioral network is implemented as a pathway of
`TransferMechanism <TransferMechanism>` and `LCA` Mechanisms, and the LC subystem uses an `LCControlMechanism` and
associated `ObjectiveMechanism`, as shown in the figure below:

.. _Nieuwenhuis2005_System_Graph:

.. figure:: _static/Nieuwenhuis_SystemGraph.svg
   :figwidth: 45 %
   :align: left
   :alt: Nieuwenhuis System Graph

Behavioral Network
~~~~~~~~~~~~~~~~~~

**INPUT LAYER**:  a `TransferMechanism` with size=3, and uses a `Linear` function with slope=1.0
and intercept=0.0.

**DECISION LAYER**: an `LCA` Mechanism of size=3, with a `Logistic` Function with a slope=1.0 and intercept=0.0, each element of which has a self-excitatory weight
with a strength specified by the `self_excitation <LCA.self_excitation>` parameter set to 2.5; a leak specified by the `leak
<LCA.leak>` parameter set to -1.0;  and every element of which is connected to every other element by mutually inhibitory weights
with a strength specified by the `competition <LCA.competition>` parameter set to 1.0.  An ordinary differential equation
describes the change in state over time, implemented in the LCA mechanism by setting its `integrator_mode <LCA
.integrator_mode>` to `True` and setting a `time_step_size <LCA.time_step_size>` to 0.02.
The output values of the `LCA` Mechanism can be evaluated with the `log` function.

**RESPONSE LAYER**: an `LCA` Mechanism of size=2, implemented as in the DECISION LAYER, with a `Logistic` Function with
a slope=1.0 and intercept=0.0; the `self_excitation <LCA.self_excitation>` parameter set to 2.0, the leak
`leak<LCA.leak>` set to -1.0, and mutually inhibitory weights specified by `competition <LCA.competition>` set to 0.
The output values of the `LCA` Mechanism can be evaluated with the `log` function.


**CONNECTIONS**:  The weights of the behavioral network are implemented as `MappingProjections <MappingProjection>`.
  The one from the *INPUT_LAYER* to the *DECISION_LAYER* uses a numpy array with a value of 1.5 for the diagonal to and
  a value of 0.33 for the off-diagonal as its `matrix <MappingProjection.matrix>` parameter;  and the one from
  the *DECISION_LAYER* to the *RESPONSE LAYER* uses a numpy array of 3.5 for the diagonal and 0 for the off-diagonal as
  its `matrix <MappingProjection.matrix>` parameter.

LC Subsystem
~~~~~~~~~~~~

The LC is implemented as a `LCControlMechanism` and its associated `ObjectiveMechanism`.  An LCControlMechanism
uses a `FitzHugh–Nagumo` integrator function to simulate the population-level activity of the LC (see `Gilzenrat
Model`), parameterized as described in the paper. The ObjectiveMechanism is specified in the `objective_mechanism
<LCControlMechanism.objective_mechanism>` argument of the LCControlMechanism constructor with a `Linear <Linear>`
function of slope=1 and intercept=0, and its `monitored_output_states <LCControlMechanism.monitored_output_states>`
argument specified as the *DECISION LAYER* with a matrix specifying the connection from each of its elements to the
`ObjectiveMechanism`. The weight from the distractor element of the *DECISION LAYER* is set to 0.0 since, as in the
original model, the distractor stimulus is assumed not to elicit an LC response. The two weights from the target 1 and
target 2 elements are set to 0.3.
Note the linear projection from the *DECISION LAYER* to the `ObjectiveMechanism` leads to an input state of length one
to the `ObjectiveMechanism`. The `ObjectiveMechanism` used does not represent a specific mechanism listed in
the paper, but is required in PsyNeuLink when a `LCControlMechanism` is implemented.
The parameters `G <LCControlMechanism.G>` = 0.5 and `k <LCControlMechanism.k>` = 1.5 are set inside the `LCControlMechanism`.
The LCControlMechanism sends `ControlProjections <ControlProjection>` to the *DECISION LAYER* and *RESPONSE LAYER*, that
regulate the `gain <Logistic.gain>` parameter of their `Logistic` Functions.

The `LCControlMechanism` outputs the values `u <LCControlMechanism.u>`, `v <LCControlMechanism.v>` and
`gain <LCControlMechanism.gain>`. The constructor can output these values using the `log` function.


Execution
---------

The `run` function executes the model, with a list of stimulus inputs specified and the number of executions specified.

Script: :download:`Download Nieuwenhuis2005Model.py <../../Scripts/Models/Nieuwenhuis2005Model.py>`

