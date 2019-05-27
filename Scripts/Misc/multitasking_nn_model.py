import numpy as np
import psyneulink as pnl
import itertools

# Setting up default network parameters
import psyneulink.core.components.functions.transferfunctions

DEFAULT_HIDDEN_PATH_SIZE = 1
DEFAULT_OUTPUT_PATH_SIZE = 1
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_DECAY = 0
DEFAULT_BIAS = -2
DEFAULT_WEIGHT_INIT_SCALE = 2e-2
DEFAULT_HIDDEN_LAYER_SIZE = 200

# Runtime/training parameters
DEFAULT_STOPPING_THRESHOLD = 1e-4


class MultitaskingModel:
    def __init__(self, num_dimensions, num_features, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 bias=DEFAULT_BIAS,
                 weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 decay=DEFAULT_DECAY,
                 hidden_path_size=DEFAULT_HIDDEN_PATH_SIZE,
                 output_path_size=DEFAULT_OUTPUT_PATH_SIZE):

        self.num_dimensions = num_dimensions
        self.num_features = num_features
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.bias = bias
        self.weight_init_scale = weight_init_scale
        self.decay = decay
        self.hidden_path_size = hidden_path_size
        self.output_path_size = output_path_size

        # implement equivalents of setData, configure, and constructor
        self.num_tasks = self.num_dimensions ** 2

        # Here we would initialize the layer - instead initializing the PNL model:
        self.task_layer = pnl.TransferMechanism(size=self.num_tasks,
                                                name='task_input')
        self.hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                  name='hidden',
                                                  function=psyneulink.core.components.functions.transferfunctions
                                                  .Logistic)
        self.hidden_bias = pnl.TransferMechanism(default_variable=np.ones((self.hidden_layer_size,)),
                                                 name='hidden bias')
        self.input_layers = self._generate_io_layers('input')
        self.output_layers = self._generate_io_layers('output')
        self._generate_output_bias_layers()
        self._generate_processes()
        self._generate_system()

    def _generate_io_layers(self, name):
        return [pnl.TransferMechanism(size=self.num_features,
                                      name='{n}-{i}'.format(n=name, i=i))
                for i in range(self.num_dimensions)]

    def _generate_output_bias_layers(self):
        self.output_biases = [
            pnl.TransferMechanism(default_variable=np.ones((self.num_features,)),
                                  name='output-bias-{i}'.format(i=i))
            for i in range(self.num_dimensions)]

    def _generate_processes(self):
        self.input_output_processes = []
        for (input_index, output_index) in itertools.product(range(self.num_dimensions),
                                                             range(self.num_dimensions)):
            # proc = pnl.Process(pathway=[self.input_layers[input_index],
            #                          (pnl.random_matrix(self.num_features, self.hidden_layer_size, 2,
            #                                            -1) * self.weight_init_scale, pnl.LEARNING),
            #                          self.hidden_layer,
            #                          (pnl.random_matrix(self.hidden_layer_size, self.num_features, 2,
            #                                            -1) * self.weight_init_scale, pnl.LEARNING),
            #                          self.output_layers[output_index]],
            #                 name='input-{i}-output-{o}-proc'.format(i=input_index,
            #                                                         o=output_index))
            #                 #learning=pnl.LEARNING))
            #
            # proc = pnl.Process(pathway=[self.input_layers[input_index],
            #                          pnl.MappingProjection(matrix=(pnl.random_matrix(self.num_features, self.hidden_layer_size, 2,
            #                                             -1) * self.weight_init_scale, pnl.LEARNING_PROJECTION)),
            #                          self.hidden_layer,
            #                          pnl.MappingProjection(matrix=(pnl.random_matrix(self.hidden_layer_size, self.num_features, 2,
            #                                             -1) * self.weight_init_scale, pnl.LEARNING_PROJECTION)),
            #                          self.output_layers[output_index]],
            #                 name='input-{i}-output-{o}-proc'.format(i=input_index,
            #                                                         o=output_index),
            #                 learning=pnl.LEARNING)

            input_to_hidden = pnl.MappingProjection(name='input-{i}-to-hidden'.format(i=input_index),
                                                    sender=self.input_layers[input_index],
                                                    receiver=self.hidden_layer,
                                                    matrix=pnl.random_matrix(self.num_features, self.hidden_layer_size,
                                                                             2, -1) * self.weight_init_scale)

            hidden_to_output = pnl.MappingProjection(name='hidden-to-output-{o}'.format(o=output_index),
                                                     sender=self.hidden_layer,
                                                     receiver=self.output_layers[output_index],
                                                     matrix=pnl.random_matrix(self.hidden_layer_size, self.num_features,
                                                                              2, -1) * self.weight_init_scale)

            proc = pnl.Process(pathway=[self.input_layers[input_index],
                                        input_to_hidden,
                                        self.hidden_layer,
                                        hidden_to_output,
                                        self.output_layers[output_index]],
                               name='input-{i}-output-{o}-proc'.format(i=input_index,
                                                                       o=output_index),
                               learning=pnl.ENABLED)

            self.input_output_processes.append(proc)

        self.task_hidden_processes = []
        self.task_output_processes = []
        self.output_bias_processes = []
        for output_index in range(self.num_dimensions):
            self.task_hidden_processes.append(
                pnl.Process(pathway=[self.task_layer,
                                     # pnl.random_matrix(self.num_tasks, self.hidden_layer_size, 2,
                                     #                   -1) * self.weight_init_scale,
                                     self.hidden_layer,
                                     # pnl.random_matrix(self.hidden_layer_size, self.num_features, 2,
                                     #                   -1) * self.weight_init_scale,
                                     self.output_layers[output_index]],
                            name='task-hidden-proc-{o}'.format(o=output_index),
                            learning=pnl.LEARNING))

            self.task_output_processes.append(
                pnl.Process(pathway=[self.task_layer,
                                     # pnl.random_matrix(self.num_tasks, self.num_features, 2,
                                     #                   -1) * self.weight_init_scale,
                                     self.output_layers[output_index]],
                            name='task-output-proc-{o}'.format(o=output_index),
                            learning=pnl.LEARNING))

            self.output_bias_processes.append(
                pnl.Process(pathway=[self.output_biases[output_index],
                                     self.output_layers[output_index]],
                            name='output-bias-proc-{o}'.format(o=output_index)))

        self.hidden_bias_process = pnl.Process(pathway=[self.hidden_bias,
                                                        self.hidden_layer],
                                               name='hidden-bias-proc')

    def _generate_system(self):
        self.system = pnl.System(
            processes=self.input_output_processes + self.task_hidden_processes + \
                      self.task_output_processes + self.output_bias_processes + [self.hidden_bias_process],
            learning_rate=self.learning_rate

        )

model = MultitaskingModel(3, 4)
model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL) #show_processes=pnl.ALL)
