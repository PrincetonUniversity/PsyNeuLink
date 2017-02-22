def first_n_calls(n, time_scale = 'trial'):
    # Currently checks all dependencies and returns False at *first* False
    def check(dependencies):
        for var in dependencies:
            # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
            num_calls = {"trial": var.component.calls_current_trial,
                         "run": var.component.calls_current_run -1 ,
                         "life": var.component.calls_since_initialization - 1}
            if not (num_calls[time_scale] <= n):
                return False
        return True
    return check


def every_n_calls(n, time_scale = 'trial'):
    # Currently checks all dependencies and returns False at *first* False
    def check(dependencies):
        for var in dependencies:
            # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
            num_calls = {"trial": var.component.calls_current_trial,
                         "run": var.component.calls_current_run - 1,
                         "life": var.component.calls_since_initialization - 1}
            if not (num_calls[time_scale]%n == 0):
                return False
        return True
    return check

