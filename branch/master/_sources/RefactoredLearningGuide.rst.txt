Migrating Learning Compositions from before PsyNeuLink 0.7.4
************************************************************

As of PsyNeuLink release 0.7.5, the API for learning has been changed.
At a high level, the major goals of this change were:

#. To streamline the process of learning in PsyNeuLink by moving the API closer to those of other python libraries
#. To merge the APIs of standard learning and autodiff learning into one cohesive API, while minimizing impact to the user
#. To make features such as minibatching, epochs, and early stopping universally available to all methods of learning
#. To open up new avenues of development with major changes and cleanup of PsyNeuLink code

As part of these changes, there were a few small changes to the API for learning in PsyNeuLink.
This guide aims to demonstrate these changes by way of example, and should serve as a point of reference for those looking to convert 
Pre 0.7.5 PsyNeuLink code to the new API.


Learning using Standard PsyNeuLink Compositions
-----------------------------------------------
Learning using standard PsyNeuLink is largely unchanged. The two changes that impact this are:

#. The removal of the `enable_learning` parameter of compositions, and
#. The addition of the `Composition.learn` method to swap to learning mode

To demonstrate this by example, consider the two following code blocks which both implement a simple three-layer network that learns XOR

>>>     input = TransferMechanism(name='Input', default_variable=np.zeros(2))
...     hidden = TransferMechanism(name='Hidden', default_variable=np.zeros(10), function=Logistic())
...     output = TransferMechanism(name='Output', default_variable=np.zeros(1), function=Logistic())
...     input_weights = MappingProjection(name='Input Weights', matrix=np.random.rand(2,10))
...     output_weights = MappingProjection(name='Output Weights', matrix=np.random.rand(10,1))
...     xor_comp = Composition('XOR Composition')
...     learning_components = xor_comp.add_backpropagation_learning_pathway(
...     pathway=[input, input_weights, hidden, output_weights, output])
...     target = learning_components[TARGET_MECHANISM]
...     # Create inputs:            Trial 1  Trial 2  Trial 3  Trial 4
...     xor_inputs = {'stimuli':[[0, 0],  [0, 1],  [1, 0],  [1, 1]],
...     'targets':[  [0],     [1],     [1],     [0] ]}
...     xor_comp.enable_learning = True              
...     xor_comp.run(inputs={input:xor_inputs['stimuli'],
...     target:xor_inputs['targets']},
...     num_trials=1,
...     animate={'show_learning':True})

And this code block is AFTER the changes:

>>>     input = TransferMechanism(name='Input', default_variable=np.zeros(2))
...     hidden = TransferMechanism(name='Hidden', default_variable=np.zeros(10), function=Logistic())
...     output = TransferMechanism(name='Output', default_variable=np.zeros(1), function=Logistic())
...     input_weights = MappingProjection(name='Input Weights', matrix=np.random.rand(2,10))
...     output_weights = MappingProjection(name='Output Weights', matrix=np.random.rand(10,1))
...     xor_comp = Composition('XOR Composition')
...     learning_components = xor_comp.add_backpropagation_learning_pathway(
...                           pathway=[input, input_weights, hidden, output_weights, output])
...     target = learning_components[TARGET_MECHANISM]
...     
...     # Create inputs:            Trial 1  Trial 2  Trial 3  Trial 4
...     xor_inputs = {'stimuli':[[0, 0],  [0, 1],  [1, 0],  [1, 1]],
...                   'targets':[  [0],     [1],     [1],     [0] ]}
...     
...     xor_comp.learn(inputs={input:xor_inputs['stimuli'],
...                          target:xor_inputs['targets']},
...                  num_trials=1,
...                  animate={'show_learning':True})


Notice that both code blocks are largely the same, with the exception of the removal of `xor_comp.enable_learning = True` and the replacement of `xor_comp.run` with `xor_comp.learn`


The changes to the learning api also add new ways to pass input for learning, for added convenience. Namely, one can now pass targets directly to output nodes if they should desire to do so. This is demonstrated in the following block of code:


>>>     input = TransferMechanism(name='Input', default_variable=np.zeros(2))
...     hidden = TransferMechanism(name='Hidden', default_variable=np.zeros(10), function=Logistic())
...     output = TransferMechanism(name='Output', default_variable=np.zeros(1), function=Logistic())
...     input_weights = MappingProjection(name='Input Weights', matrix=np.random.rand(2,10))
...     output_weights = MappingProjection(name='Output Weights', matrix=np.random.rand(10,1))
...     xor_comp = Composition('XOR Composition')
...     xor_comp.add_backpropagation_learning_pathway(
...             pathway=[input, input_weights, hidden, output_weights, output])
...     
...     # Create inputs:            Trial 1  Trial 2  Trial 3  Trial 4
...     xor_inputs = {'stimuli':[[0, 0],  [0, 1],  [1, 0],  [1, 1]],
...                   'targets':[  [0],     [1],     [1],     [0] ]}
...     
...     xor_comp.learn(inputs={input:xor_inputs['stimuli']}, 
...                    targets={output:xor_inputs['targets']},
...                    num_trials=1,
...                    animate={'show_learning':True})

Notice that we no longer have to extract the target node from the `add_backpropagation_learning_pathway method`, and can instead pass the targets as output nodes mapped to values, in a new parameter called `targets` in the `learn` method.



Learning using AutodiffCompositions
-----------------------------------------
AutodiffCompositions are also update in this change. In addition to the API changes mentioned for learning in compositions, several parameters of AutodiffCompositions have been moved from
its constructor to be runtime parameters of its `learn` method.

This is demonstrated in the following codeblocks:

This is the OLD code:

>>>     my_mech_1 = pnl.TransferMechanism(function=pnl.Linear, size = 3)
...     my_mech_2 = pnl.TransferMechanism(function=pnl.Linear, size = 2)
...     my_projection = pnl.MappingProjection(matrix=np.random.randn(3,2),
...                         sender=my_mech_1,
...                         receiver=my_mech_2)
...     # Create AutodiffComposition
...     my_autodiff = pnl.AutodiffComposition(patience=10, min_delta=.0001)
...     my_autodiff.add_node(my_mech_1)
...     my_autodiff.add_node(my_mech_2)
...     my_autodiff.add_projection(sender=my_mech_1, projection=my_projection, receiver=my_mech_2)
...     # Specify inputs and targets
...     my_inputs = {my_mech_1: [[1, 2, 3]]}
...     my_targets = {my_mech_2: [[4, 5]]}
...     input_dict = {"inputs": my_inputs, "targets": my_targets, "epochs": 2}
...     # Run Composition with learning enabled
...     my_autodiff.learning_enabled=True # this is not strictly necessary, as learning_enabled is True by default
...     my_autodiff.run(inputs = input_dict)
...     # Run Composition with learning disabled
...     my_autodiff.learning_enabled=False
...     my_autodiff.run(inputs = input_dict)


And this is equivalent code AFTER the changes:

>>>     my_mech_1 = pnl.TransferMechanism(function=pnl.Linear, size = 3)
...     my_mech_2 = pnl.TransferMechanism(function=pnl.Linear, size = 2)
...     my_projection = pnl.MappingProjection(matrix=np.random.randn(3,2),
...                         sender=my_mech_1,
...                         receiver=my_mech_2)
...     # Create AutodiffComposition
...     my_autodiff = pnl.AutodiffComposition()
...     my_autodiff.add_node(my_mech_1)
...     my_autodiff.add_node(my_mech_2)
...     my_autodiff.add_projection(sender=my_mech_1, projection=my_projection, receiver=my_mech_2)
...     # Specify inputs and targets
...     my_inputs = {my_mech_1: [[1, 2, 3]]}
...     my_targets = {my_mech_2: [[4, 5]]}
...     input_dict = {"inputs": my_inputs, "targets": my_targets, "epochs": 2}
...     
...     # Run Composition with learning enabled
...     my_autodiff.learn(inputs = input_dict, patience=10, min_delta=.0001)
...     
...     # Run Composition with learning disabled
...     my_autodiff.run(inputs = input_dict['inputs'])

Notice that the `patience` and `epochs` parameters have been moved to the call to `my_autodiff.learn`.
In addition, notice that the inputs to `autodiffcomposition.run` can no longer be the same dictionary as those passed to `autodiffcomposition.learn`.


