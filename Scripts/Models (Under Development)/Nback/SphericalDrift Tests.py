import numpy as np
from psyneulink import *

NUM_TRIALS = 48

stims = np.array([x[0] for x in em.memory])
contexts = np.array([x[1] for x in em.memory])
cos = Distance(metric=COSINE)
dist = Distance(metric=EUCLIDEAN)
diffs = [np.sum([contexts[i+1] - contexts[1]]) for i in range(NUM_TRIALS)]
diffs_1 = [np.sum([contexts[i+1] - contexts[i]]) for i in range(NUM_TRIALS)]
diffs_2 = [np.sum([contexts[i+2] - contexts[i]]) for i in range(NUM_TRIALS-1)]
dots = [[contexts[i+1] @ contexts[1]] for i in range(NUM_TRIALS)]
dot_diffs_1 = [[contexts[i+1] @ contexts[i]] for i in range(NUM_TRIALS)]
dot_diffs_2 = [[contexts[i+2] @ contexts[i]] for i in range(NUM_TRIALS-1)]
angle = [cos([contexts[i+1], contexts[1]]) for i in range(NUM_TRIALS)]
angle_1 = [cos([contexts[i+1], contexts[i]]) for i in range(NUM_TRIALS)]
angle_2 = [cos([contexts[i+2], contexts[i]]) for i in range(NUM_TRIALS-1)]
euclidean = [dist([contexts[i+1], contexts[1]]) for i in range(NUM_TRIALS)]
euclidean_1 = [dist([contexts[i+1], contexts[i]]) for i in range(NUM_TRIALS)]
euclidean_2 = [dist([contexts[i+2], contexts[i]]) for i in range(NUM_TRIALS-1)]
print("STIMS:", stims, "\n")
print("DIFFS:", diffs, "\n")
print("DIFFS 1:", diffs_1, "\n")
print("DIFFS 2:", diffs_2, "\n")
print("DOT PRODUCTS:", dots, "\n")
print("DOT DIFFS 1:", dot_diffs_1, "\n")
print("DOT DIFFS 2:", dot_diffs_2, "\n")
print("ANGLE: ", angle, "\n")
print("ANGLE_1: ", angle_1, "\n")
print("ANGLE_2: ", angle_2, "\n")
print("EUCILDEAN: ", euclidean, "\n")
print("EUCILDEAN 1: ", euclidean_1, "\n")
print("EUCILDEAN 2: ", euclidean_2, "\n")
