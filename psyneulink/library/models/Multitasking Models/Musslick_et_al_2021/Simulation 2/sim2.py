import sim2_configure

iterations_train = 50 # maximum number of training iterations

def main():
    # Configure network
    task_net = sim2_configure.config()
    # Train
    task_net.train(iterations_train)
    # Analysis and validation
    task_net.run(task_net.input_set, task_net.task_set)


if __name__ == '__main__':
    main()