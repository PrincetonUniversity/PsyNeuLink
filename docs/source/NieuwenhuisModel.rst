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

XXX SYSTEM GRAPH FIGURE HERE XXXX

Behavioral Network
~~~~~~~~~~~~~~~~~~

**INPUT LAYER**:  a `TransferMechanism` with size=3, and uses a `Linear` function with slope=1.0
and intercept=0.0.

**DECISION LAYER**: an `LCA` Mechanism, with a `Logistic` Function, each element of which has a self-excitatory weight
with a strength specified by the `self_excitation <LCA.self_excitation>` parameter; a leak specified by the `leak
<LCA.leak>` parameter;  and every element of which is connected to every other element by mutually inhibitory weights
with a strength specified by the `competition <LCA.competition>` parameter.  An ordinary differential equation
describes the change in state over time, implemented in the LCA mechanism by setting its `integrator_mode <LCA
.integrator_mode>` to `True` and setting a `time_step_size <LCA.time_step_size>`. XXX VALUE OF TIME_STEP_SIZE XXX

**RESPONSE LAYER**: an `LCA` mechanism and the parameters specified as in the paper. XXXX WHAT ARE THEY
(INCLUDING FUNCITON XXXX

.. note::
   [XXX IS THIS TRUE JUST OF THE RESPONSE LAYER??? IF SO, NO NEED FOR THIS NOTE, JUST REPORT ITS VALUE AS 0 IN THE
   PARAMETERS ABOVE] In Nieuwenhuis et al. (2005), the weight of the mutual inhibition among response units is not
   explicitly described;  setting it to 0.0 reproduces the results reported.]

**CONNECTIONS**:  The weights of the behavioral network are implemented as `MappingProjections <MappingProjection>`.
  The one from the *INPUT_LAYER* to the *DECISION_LAYER* uses a a numpy array of XXX VALUE??? as its `matrix
  <MappingProjection.matrix>` parameter;  and the one from the *DECISION_LAYER* to the *RESPONSE LAYER* uses a numpy
  array of XXX VALUE??? as its `matrix <MappingProjection.matrix>` parameter.

LC Subsystem
~~~~~~~~~~~~

The LC is implemented as a `LCControlMechanism` and its associated `ObjectiveMechanism`.  An LCControlMechanism
uses a `FitzHugh–Nagumo` integrator Function to simulate the population-level activity of the LC (see `Gilzenrat
Model`), parameterized as described in the paper. The ObjectiveMechanism is specified in the `objective_mechanism
<LCControlMechanism.objective_mechanism>` argument of the LCControlMechanism constructor with a `Linear <Linear>`
function, and its `monitored_output_states <LCControlMechanism.monitored_output_states>` argument specified as the
*DECISION LAYER* with a matrix specifying the connection from each of its units to the ObjectiveMechanism's single
element. XXX THIS NEEDS TO BE BETTER EXPLAINED XXXX. Note that the weight from the distractor unit is set to 0.0
since, as in the original model, the distractor stimulus is assumed not to elicit an LC response. The parameters
`G <LCMechanism.G>` and `k <LCMechanism.k>` set inside the `LCControlMechanism`.
The LCControlMechanism sends `ControlProjections <ControlProjection>` to the *DECISION LAYER* and *RESPONSE LAYER*,
that regulate the `gain <Logistic.gain>` parameter of their `Logistic` Functions.

Execution
---------

XXX DESCRIPTION OF HOW TO RUN IT, AND WHAT IT GENERATES AS ITS OUTPUT.



Script: :download:`Download Nieuwenhuis2005Model.py <../../Scripts/Models/Nieuwenhuis2005Model.py>`

