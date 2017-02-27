
def every_n_calls(n, time_scale = 'trial'):
    """
    Condition to be applied to a constraint
    Enforces condition that all constraint dependencies must run n times before each time the constraint owner runs

    Parameters
    ----------
    n -- number of times dependencies run before owner runs
    time_scale -- time_scale on which to count (trial, run or life)

    Returns
    -------
    Boolean (True if all dependencies satisfy the conditions, False if *any* do not)

    """
    def check(dependencies):
        for var in dependencies:
            # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
            num_calls = {"trial": var.component.calls_current_trial,
                         "run": var.component.calls_current_run - 1,
                         "life": var.component.calls_since_initialization - 1}
            if not (num_calls[time_scale]%n == 0 and num_calls[time_scale] != 0):
                return False
        return True
    return check

def first_n_calls_AND(n, time_scale = 'trial'):
    """
    Condition function to be applied to a constraint
    Enforces condition that dependencies must run n times on the given time scale before the owner's first run
    Owner then continues to run every time step

    Parameters
    ----------
    n -- number of times that dependencies must run before owner can run
    time_scale -- time_scale on which to count (trial, run or life)

    Returns
    -------
    Boolean (True if all dependencies satisfy the conditions, False if *any* do not)

    """
    def check(dependencies):
        for var in dependencies:
            # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
            num_calls = {"trial": var.component.calls_current_trial,
                         "run": var.component.calls_current_run -1 ,
                         "life": var.component.calls_since_initialization - 1}
            if (num_calls[time_scale] < n):
                return False
        return True
    return check


def first_n_calls_OR(n, time_scale = 'trial'):
    """
    Condition function to be applied to a constraint
    Enforces condition that dependencies must run n times on the given time scale before the owner's first run
    Owner then continues to run every time step

    Parameters
    ----------
    n -- number of times that dependencies must run before owner can run
    time_scale -- time_scale on which to count (trial, run or life)

    Returns
    -------
    Boolean (True if all dependencies satisfy the conditions, False if *any* do not)

    """
    def check(dependencies):
        for var in dependencies:
            # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
            num_calls = {"trial": var.component.calls_current_trial,
                         "run": var.component.calls_current_run -1 ,
                         "life": var.component.calls_since_initialization - 1}
            if (num_calls[time_scale] >= n):
                return True
        return False
    return check

def over_threshold_OR(threshold, time_scale = 'trial'):
    """
    Condition function to be applied to a constraint
    Enforces condition that dependencies must run n times on the given time scale before the owner's first run
    Owner then continues to run every time step

    Parameters
    ----------
    n -- number of times that dependencies must run before owner can run
    time_scale -- time_scale on which to count (trial, run or life)

    Returns
    -------
    Boolean (True if all dependencies satisfy the conditions, False if *any* do not)

    """
    def check(dependencies, time_scale = 'trial'):
        for var in dependencies:
            print(var.component.value)
            if abs(float(var.component.value)) >= threshold:
            # num_calls = {"trial": var.component.calls_current_trial,
            #              "run": var.component.calls_current_run -1 ,
            #              "life": var.component.calls_since_initialization - 1}
                return True
        return False
    return check

def terminal_AND(dependencies):
    def check(dependencies):
        for var in dependencies:
            if var.component.calls_current_trial == 0:
                return False
        return True
    return check

def terminal_OR(dependencies):
    def check(dependencies):
        for var in dependencies:
            if var.component.calls_current_trial > 0:
                return True
        return False
    return check