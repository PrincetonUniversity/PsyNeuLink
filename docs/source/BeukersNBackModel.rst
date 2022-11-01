
N-Back Model (Beukers et al., 2022)
==================================================================
`"When Working Memory is Just Working, Not Memory" <https://psyarxiv.com/jtw5p>`_

Overview
--------
This implements a model of the `N-back task <https://en.wikipedia.org/wiki/N-back#Neurobiology_of_n-back_task>`_
described in `Beukers et al. (2022) <https://psyarxiv.com/jtw5p>`_.  The model uses a simple implementation of episodic
memory (EM, as a form of content-retrieval memory) to store previous stimuli along with the temporal context in which
they occured, and a feedforward neural network (FFN)to evaluate whether the current stimulus is a match to the n'th
preceding stimulus (nback-level)retrieved from episodic memory.  The temporal context is provided by a randomly
drifting high dimensional vector that maintains a constant norm (i.e., drifts on a sphere).  The FFN is
trained, given an n-back level of *n*, to identify when the current stimulus matches one stored in EM
with a temporal context vector that differs by an amount corresponding to *n* time steps of drift.  During n-back
performance, the model encodes the current stimulus and temporal context, retrieves an item from EM that matches the
current stimulus, weighted by the similarity of its temporal context vector (i.e., most recent), and then uses the
FFN to evaluate whether it is an n-back match.  The model responds "match" if the FFN detects a match; otherwise, it
either responds "non-match" or, with a fixed probability (hazard rate), it uses the current stimulus and temporal
context to retrieve another sample from EM and repeat the evaluation.

This model is an example of proposed interactions between working memory (e.g., in neocortex) and episodic memory
e.g., in hippocampus and/or cerebellum) in the performance of tasks demanding of sequential processing and control,
and along the lines of models emerging machine learning that augment the use of recurrent neural networks (e.g., long
short-term memory mechanisms; LSTMs) for active memory and control with an external memory capable of rapid storage
and content-based retrieval, such as the Neural Turing Machine (NTN;
`Graves et al., 2016 <https://arxiv.org/abs/1410.5401>`_), Episodic Planning Networks (EPN;
`Ritter et al., 2020 <https://arxiv.org/abs/2006.03662>`_), and Emergent Symbols through Binding Networks (ESBN;
`Webb et al., 2021 <https://arxiv.org/abs/2012.14601>`_).

The script respectively, to construct, train and run the model:

* construct_model(args):
  takes as arguments parameters used to construct the model; for convenience, defaults are defined toward the top
  of the script (see "Construction parameters").
..
* train_network(args)
  takes as arguments the feedforward neural network Composition (FFN_COMPOSITION) and number of epochs to train.
  Note: learning_rate is set at construction (which can be specified using LEARNING_RATE under "Training parameters").
..
* run_model()
  takes as arguments the drift rate in the temporal context vector to be applied on each trial,
  and the number of trials to execute, as well as reporting and animation specifications
  (see "Execution parameters").

The default parameters are ones that have been fit to empirical data concerning human performance
(taken from `Kane et al., 2007 <https://psycnet.apa.org/record/2007-06096-010?doi=1>`_).


The Model
---------

The models is composed of two `Compositions <Composition>`: an outer one that contains the full model (nback_model),
and an `AutodiffComposition` (ffn), nested within nback_model (see red box in Figure), that implements the
feedforward neural network (ffn).

nback_model
~~~~~~~~~~~

This contains three input Mechanisms (

Both of these are constructed in the construct_model function.
The ffn Composition is trained use

.. _nback_Fig:

.. figure:: _static/N-Back_Model_movie.gif
   :align: left
   :alt: N-Back Model Animation


Training
--------


Execution
---------


Script: :download:`N-back.py <../../Scripts/Models (Under Development)/Beukers_N-Back_2022.py>`
.. Script: :download:`N-back.py <../../psyneulink/library/models/Beukers -Back.py>`
