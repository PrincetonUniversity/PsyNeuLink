Nieuwenhuis et al. (2005)
=========================
`"The Role of the Locus Coeruleus in Mediating the Attentional Blink: A Neurocomputational Theory" <https://research.vu.nl/ws/files/2063874/Nieuwenhuis%20Journal%20of%20Experimental%20Psychology%20-%20General%20134(3)-2005%20u.pdf>`_

Overview
--------


This model seeks to investigate the role of the Locus Coeruleus in mediating the attentional blink. The attentional
blink refers to the temporary impairment in perceiving the 2nd of 2 targets presented in close temporal proximity.

During the attentional blink paradigm, on each trial a list of letters is presented to subjects, colored in black on a
grey background. Additionally, two numbers are presented during each trial and the task is to correctly identify which
two digits between 2-9 were presented. A vast amount of studies showed that the probability of correctly identifying
both digits accurately depends on the lag between the two target stimuli.
More precisely, if T2 is presented right after T1 it is possible to identify both  between 200 ms and 300 ms after the onset
of the first target (T1) accuracy decreases. However, presenting the second target stimulus T2 right after the first
target stimulus T1, subjects performance is as accurate as with lags longer than 400ms between T1 and T2.

The model by Nieuwenhuis et al. (2005) shows that the findings on the attentional blink paradigm can be explained by
the mechanics of the Locus Ceruleus. This model aims to bridge findings from behavioral psychology and findings from
neurophysiology with a neurocomputational theory.

With this model it is possible to simulate that subjects behavior on identifying the second target stimuli T2 accurately
depends on:

* whether T1 was accurately identified
*  the lag between T1 and T2
* the mode of the LC

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

This example illustrates Figure 3 from the Nieuwenhuis et al. (2005) paper. A time difference between the onset of
T1 and T2 of 100 ms is framed as lag 1. In this example we simulate a time difference of 200 ms, i.e. lag 2 and one
execution only. Note that in the Nieuwenhuis et al. (2005) paper the Figure shows the average activation over 1000
executions.

The model consists of two networks. A behavioral network, feeding forward information from the input layer,
to the decision layer, to the response layer, and a LC control mechanism, projecting gain to both, the behavioral layer
and the response layer.


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