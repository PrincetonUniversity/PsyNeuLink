import numpy as np
import psyneulink as pnl
import multitasking_nn_model

# Setting up defaults
DEFAULT_TRAIN_ITERATIONS = 500
DEFAULT_NUM_REPLICATIONS = 100

DEFAULT_NUM_DIMENSIONS = 3
DEFAULT_NUM_FEATURES_PER_DIMENSION = 4


def load_or_create_patterns(num_dimensions=DEFAULT_NUM_DIMENSIONS,
                            num_features=DEFAULT_NUM_FEATURES_PER_DIMENSION):
    # TODO: get this code from Sebastian
    pass


def main():
    model = multitasking_nn_model.MultitaskingModel(DEFAULT_NUM_DIMENSIONS, DEFAULT_NUM_FEATURES_PER_DIMENSION)
    model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL) #show_processes=pnl.ALL)


if __name__ == '__main__':
    main()
