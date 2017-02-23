
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
            if not (num_calls[time_scale]%n == 0):
                return False
        return True
    return check

def first_n_calls(n, time_scale = 'trial'):
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

