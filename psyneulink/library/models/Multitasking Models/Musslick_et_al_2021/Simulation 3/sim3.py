import sys
import sim3_configure

iterations_train = 5000 # maximum number of training iterations

def main():
    # Configure network
    task_net = sim3_configure.config()
    # Train
    task_net.train(iterations_train)
    # Analysis and validation
    task_net.run(task_net.input_set, task_net.task_set)


if __name__ == '__main__':
    main()