# Change Set

## Time Vector

In the original implementation, the time vector is initialized and updated within the memory itself. Here, we
generate the time vector externally and pass it to the memory update.

Also the time vector update, and retrieval function has been changed. In the original version, the time vector
is initialized as small vector:

```python
time_code = torch.zeros((params['time_d'],),dtype=torch.float)+.01
```

and updated by adding a small noise value:

```python
time_code += torch.randn_like(time_code)*time_noise
```

this is equiavalent to a positive drift rate of `time_noise/2` plus a noise with standard deviation of 
`sqrt(time_noise/2)` (uniform).

However, later the time_vector is normalized both in memory and as query. Since the vector is always positive
and always growing, this means the larger the time value, the closer proceding vectors will be to each other after
normalization (which is unintended).

Instead, we use DriftOnASphere that implements a Brownian motion on a unit sphere. This means the time vector is
always normalized (length 1) but the difference between two proceeding time vectors it independent of their absoulte
value (as intended).

We use PsyNeuLink's `DriftOnASphere` function to implement this.

## Incorrect normalizing in the Context and Time weights in the Estimated Reward

In the original implementation there was a error in the normalization of the context retrieval
and time retrieval weight. The original implementation did this:

```python
context_weight /=  (context_weight + time_weight)
time_weight /=  (context_weight + time_weight)
```

however, this incorrectly normalizes the time_weight with the already updated context_weight. Instead,
we use the following:

```python
total = context_weight + time_weight
context_weight /= total
time_weight /= total
```

***

For the model we evaluate against, we have two more changes:

- Instead of initializing the memories as an empty list, we are initializing the memories with small values and
overriding them (to reflect PsyNeuLInk behaviour)
- This makes it necessary to use a softmax with threshold for retrieval to threshold low-probability/low-similarity
memories before softmaxing and make the softmax less dependent on the amount of low-similarity memories
