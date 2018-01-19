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
a digit rather than a letter).  The participant is asked to observe the sequence, then do their best to identify the
target (T1) as well as the stimulus that immediately follows it (T2).  A consistent finding is that the  accuracy of
identifying both T1 and T2 correctly depends on the lag between them. In particular, accuracy is high when T2 is
presented *immediately* after the first target (T1), but drops when T2 is presented between 200 and 300ms after the
onset of T1, and then is restored at delays longer than 400ms.

The model described by Nieuwenhuis et al. (2005) suggests that these findings can be explained by phasic activation
of LC in response to T1, and the concomitant effects of norepinephrine (NE) release on neural gain, followed by
refractoriness of the LC/NE response.  The model demonstrates that accuracy in identifying T2 depends on:
* whether T1 was accurately identified;
* the lag between T1 and T2;
* the `mode of the LC <LCControlMechanism_Modes_Of_Operation>` (`phasic or tonic <https://www.ncbi.nlm.nih.gov/pubmed/8027789>`_).

The Figure below show the behavior of the model for a single execution of with a lag of 200ms, corresponding to the
conditions reported in Figure 3 of Nieuwenhuis et al. (2005) (note that their Figure reports the average over 1000
executions).


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

The model consists of two subsystems: a behavioral network, in which stimulus information feeds forward from an input
layer, via a decision layer, to a response layer;  and an `LCControlMechanism` that receives a `MappingProjection`
from the decision layer, and sends `ControlProjections <ControlProjection>` to the decision and response layers that
regulate the gain of their `LogisticFunction`.


Creating Nieuwenhuis et al. (2005)
----------------------------------

After setting global variables, weights and initial values the behavioral network is created with 3 layers,
i.e. INPUT LAYER, DECISION LAYER, and RESPONSE LAYER. The INPUT LAYER is constructed with a TransferMechansim of size 3,
and a Linear function with the default slope set to 1.0 and the intercept set to 0.0.

The DECISION LAYER is implemented with a LCA mechanism where each element is connected to every other element with
mutually inhibitory weights and self-excitation weights, defined in PsyNeuLink as `competition <LCA.competition>`, `self_excitation <LCA.self_excitation>`, and
`leak <LCA.leak>`. `leak <LCA.leak>` defines the sign of the off-diagonals, here the mutually inhibitory weights. The ordinary differential
equation that describes the change in state with respect to time is implemented in the LCA mechanism with the
`integrator_mode <LCA.integrator_mode>` set to True and setting the `time_step_size <LCA.time_step_size>`.

The final step is to implement the RESPONSE LAYER:
The RESPONSE LAYER is implemented as the DECISION LAYER with a `LCA` mechanism and the parameters specified as in the
paper. (WATCH OUT !!! In the paper the weight "Mutual inhibition among response units" is not defined, but needs to be
set to 0.0 in order to reproduce the paper)

The weights of the behavioral network are created with two numpy arrays.

The LC is implemented with a `LCControlMechanism`. The `LCControlMechanism` has a FitzHugh–Nagumo system implemented.
All parameters from this system are specified as in the paper. Additionally, the `LCControlMechanism` can only monitor
output states that come from an `ObjectiveMechanism`. Thus, a `ObjectiveMechanism` is created in the
`objective_mechanism <LCControlMechanism.objective_mechanism>` parameter with a `Linear <Linear>` function set to it's
default values and the `monitored_output_states <LCControlMechanism.monitored_output_states>`
parameter set to decision_layer with the weights projecting from T1, T2 and the distraction element to the
`ObjectiveMechanism`. Note that the weights from the distraction unit are set to 0.0 since the paper did not implement
weights from the distraction unit to the LC. The parameters G and k are set inside the `LCControlMechanism`.
This LCControlMechanism projects a gain control signal to the DECISION LAYER and the RESPONSE LAYER.


Script: :download:`Download Nieuwenhuis2005Model.py <../../Scripts/Models/Nieuwenhuis2005Model.py>`

