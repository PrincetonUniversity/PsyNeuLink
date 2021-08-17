import sys
import sim1_configure

iterations_train = 1000 # maximum number of training iterations

def main(graph_input, hidden_arg, silence):
    # Configure network
    task_net = sim1_configure.config(graph_input, hidden_arg, silence)
    # Train
    task_net.train(iterations_train)
    # Analysis and validation
    task_net.run(task_net.input_set, task_net.task_set)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])