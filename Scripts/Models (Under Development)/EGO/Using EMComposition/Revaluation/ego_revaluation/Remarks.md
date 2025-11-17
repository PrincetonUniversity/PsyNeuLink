# For EM in PNL

It would be a good addition, if cosine similarity could be used instead of dot product
to match. This makes the field weights more comparable.

(This is not the same as normalizing memories since the memories can still
have various lengths)

Even better:
Implement context as drift on a sphere with target mode (this ensures that context is on a sphere
as is state, and time)

Also:
If we ever want to query with reward, we have to make it one hot (0, 10 are bad since the
dot product is always higher for 10 and cosine similarity is just equal for both)

Also: This actually makes a big difference!
NOTE: If we use cosine similarity, we should also let the context "walk on a sphere instead" (memory
retrieval should match memory storage)


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
