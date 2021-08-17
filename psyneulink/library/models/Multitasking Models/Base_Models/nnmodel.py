import numpy as np
import psyneulink as pnl

class NNModel():
    """Implements the neural network model. To use the model, construct NNModel
    with the init parameters, use set_data to set the training data, and then
    use configure to configure the Composition. Call train to train the network
    and run to run the newtwork on a set of inputs.
    
    Attributes:
    n_hidden: number of hidden layers
    coeff: learning rate
    bias_weight: weight from bias units to hidden and output units
    init_scale: maximum absolute magnitude of initial weights
    thresh: mean-squared error stopping criterion for learning
    decay: weight penalization parameter
    hidden_path_size: size of one path (group of units) in hidden layer
    output_path_size: size of one path (group of units) in output layer
    silence: silence output

    input_set: stimulus input data for training
    task_set: task input data for training
    train_data: target data for training

    n_input: number of stimulus input units
    n_task: number of task input units
    n_hidden: number of hidden units
    n_output: number of output units
    hidden_bias: hidden bias weight
    output_bias: output bias weight

    input_hidden_weights: stimulus input layer to hidden layer weights
    task_hidden_weights: task input layer to hidden layer weights
    task_output_weights: task input layer to output layer weights
    hidden_output_weights: hidden layer to output layer weights

    stimulus_input_layer: Mechanism for the stimulus input layer
    task_input_layer: Mechanism for the task input layer
    hidden_layer: Mechanism for the hidden layer
    output_layer: Mechanism for the output layer
    comp: the composition representing the model
    """

    def __init__(self, n_hidden, learning_rate=0.3, bias_weight=-1, init_scale=1,
        thresh=0.01, decay=0.02, hidden_path_size=1, output_path_size=1):
        """Initialize the model with basic parameters."""
        self.n_hidden = n_hidden
        self.coeff = learning_rate
        self.bias_weight = bias_weight
        self.init_scale = init_scale
        self.thresh = thresh
        self.decay = decay
        self.hidden_path_size = hidden_path_size
        self.output_path_size = output_path_size
        self.silence = False


    def set_data(self, input_data, task_data, train_data):
        """Set the model's data sets for training.

        input_data: an array of stimulus input data
        task_data: an array of task input data
        train_data: the training set"""
        self.input_set = input_data
        self.task_set = task_data
        self.train_set = train_data


    def configure(self):
        """Configure the number of units in each layer, initialize weight 
        matrices, and create the Composition.
        """
        # Get size of each layer from dimensions of input/task/output sets
        self.n_input = np.size(self.input_set, 1)
        self.n_task = np.size(self.task_set, 1)
        self.n_output = np.size(self.train_set, 1)
        self.hidden_bias = self.bias_weight
        self.output_bias = self.bias_weight

        # Initialize weight matrices
        self.input_hidden_weights = -1 + 2 * np.random.rand(
            self.n_input, self.n_hidden) * self.init_scale
        self.task_hidden_weights = -1 + 2 * np.random.rand(
            self.n_task, self.n_hidden) * self.init_scale
        self.task_output_weights = -1 + 2 * np.random.rand(
            self.n_task, self.n_output) * self.init_scale
        self.hidden_output_weights = -1 + 2 * np.random.rand(
            self.n_hidden, self.n_output) * self.init_scale

        # Set up mechanisms
        self.stimulus_input_layer = pnl.TransferMechanism(
            name='Stimulus Input Layer',
            default_variable=np.zeros(self.n_input)
        )
        self.task_input_layer = pnl.TransferMechanism(
            name='Task Input Layer',
            default_variable=np.zeros(self.n_task)
        )
        self.hidden_layer = pnl.TransferMechanism(
            name='Hidden Layer',
            default_variable=np.zeros(self.n_hidden),
            function=pnl.Logistic(bias=self.hidden_bias)
        )
        self.output_layer = pnl.TransferMechanism(
            name='Output Layer',
            default_variable=np.zeros(self.n_output),
            function=pnl.Logistic(bias=self.output_bias)
        )
        # Set up projections
        input_to_hidden = pnl.MappingProjection(
            name='Input to Hidden',
            matrix=self.input_hidden_weights
        )
        task_to_hidden = pnl.MappingProjection(
            name='Task to Hidden',
            matrix=self.task_hidden_weights
        )
        task_to_output = pnl.MappingProjection(
            name='Task to Output',
            matrix=self.task_output_weights
        )
        hidden_to_output = pnl.MappingProjection(
            name='Hidden to Output',
            matrix=self.hidden_output_weights
        )
        # Set up composition and pathways
        self.comp = pnl.AutodiffComposition(
            name='NN Model'
        )
        self.comp.add_backpropagation_learning_pathway(pathway=[
            self.stimulus_input_layer, 
            input_to_hidden, 
            self.hidden_layer,
            hidden_to_output,
            self.output_layer],
            learning_rate=self.coeff)
        self.comp.add_backpropagation_learning_pathway(pathway=[
            self.task_input_layer, 
            task_to_hidden, 
            self.hidden_layer,
            hidden_to_output,
            self.output_layer],
            learning_rate=self.coeff)
        self.comp.add_backpropagation_learning_pathway(pathway=[
            self.task_input_layer, 
            task_to_output,
            self.output_layer],
            learning_rate=self.coeff)


    def train(self, iterations):
        """Train the model.
    
        iterations: the number of training iterations"""
        self.comp.learn(
            inputs={self.stimulus_input_layer: self.input_set, 
                    self.task_input_layer: self.task_set},
            targets={self.output_layer: self.train_set},
            epochs=iterations,
            randomize_minibatches=True)


    def run(self, input_data, task_data):
        """ Run the model on the given set of input data.
        
        input_data: the stimulus input data
        task_data: the task input data"""
        return self.comp.run(
            inputs={self.stimulus_input_layer: input_data, 
                    self.task_input_layer: task_data},
            report_output=pnl.ReportOutput.OFF)