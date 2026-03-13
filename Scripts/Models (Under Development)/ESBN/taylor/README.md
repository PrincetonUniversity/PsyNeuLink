# Emergent Symbols through Binding in External Memory

Code for the paper [Emergent Symbols through Binding in External Memory](https://arxiv.org/abs/2012.14601).

The `./scripts` directory contains scripts to reproduce the primary results presented in the paper. There is one script per task. Each script will train and evaluate 10 networks on all generalization regimes for a particular task. Each script requires two arguments:
1. Architecture name (according to the names of the files in the `./models` directory, e.g. `ESBN`, `LSTM`, etc.)
2. Device index (default is `0` on machines with only one device)

For example, to reproduce the results for the ESBN architecture on the same/different discrimination task, run the following command:
```
./scripts/same_diff.sh ESBN 0
```
These scripts use the default values for the simulations reported in the paper (temporal context normalization, convolutional encoder, learning rate, number of training epochs). To reproduce some of the other results in the paper (such as for the models that needed to be trained for longer, or the experiments involving alternative encoder architectures), some of these values will need to be changed by modifying the scripts accordingly.

To reproduce the analysis of the learned representations presented in the appendix (Figure 9), run the following command:
```
python3 ./train_and_extract_reps.py
```
Then navigate to the `./extracted_reps` directory and run:
```
python3 ./learned_rep_analysis.py
```

## Prerequisites

- Python 3
- [NumPy](https://numpy.org/)
- [colorlog](https://github.com/borntyping/python-colorlog)
- [PIL](https://pillow.readthedocs.io/en/stable/)
- [PyTorch](https://pytorch.org/)

For analysis of learned representations:

- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## Authorship

All code was written by [Taylor Webb](https://github.com/taylorwwebb). 
