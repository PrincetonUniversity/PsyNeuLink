import sim6_configure

iterations_train = 5000 # maximum number of training iterations
replications = 20

def main():
    for _ in range(replications):
        # Configure networks
        tasknet_shared0, tasknet_shared1, tasknet_shared2, tasknet_shared3 = \
            sim6_configure.config()
        # Pretrain
        tasknet_shared1.train(iterations_train)
        tasknet_shared2.train(iterations_train)
        tasknet_shared3.train(iterations_train)
        # Train networks on final training pattern
        input_tested = tasknet_shared0.input_tested
        tasks_tested = tasknet_shared0.tasks_tested
        train_tested = tasknet_shared0.train_tested
        tasknet_shared0.set_data(input_tested, tasks_tested, train_tested)
        tasknet_shared1.set_data(input_tested, tasks_tested, train_tested)
        tasknet_shared2.set_data(input_tested, tasks_tested, train_tested)
        tasknet_shared3.set_data(input_tested, tasks_tested, train_tested)
        tasknet_shared0.train(iterations_train)
        tasknet_shared1.train(iterations_train)
        tasknet_shared2.train(iterations_train)
        tasknet_shared3.train(iterations_train)


if __name__ == '__main__':
    main()