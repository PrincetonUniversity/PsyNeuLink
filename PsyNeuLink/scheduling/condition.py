
def every_n_calls(n, time_scale = 'trial'):
    """
    Condition to be applied to a constraint
    Enforces condition that constraint dependency must run n times before each time the constraint owner runs

    Parameters
    ----------
    n -- number of times dependency runs before owner runs
    time_scale -- time_scale on which to count (trial, run or life)

    Returns
    -------
    Boolean (True if condition is met)

    """

    def check(dependencies):

        #every_n should only depend on one mechanism
        var = dependencies[0]
        # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
        num_calls = {"trial": var.component.calls_current_trial,
                    "run": var.component.calls_current_run - 1,
                    "life": var.component.calls_since_initialization - 1}
        if num_calls[time_scale] % n != 0 or num_calls[time_scale] == 0:
                return False
        return True

    return check

# NOTE: Any mechanisms 'downstream' of the owner of this condition will get stuck because the owner will stop
# running after n times (run Test 6 to see what this looks like)
def first_n_calls(n, time_scale = 'trial'):
    """
    Condition function to be applied to a constraint
    Enforces condition that owner runs the first n times its dependency runs

    Parameters
    ----------
    n -- number of times that dependencies must run before owner can run
    time_scale -- time_scale on which to count (trial, run or life)

    Returns
    -------
    Boolean (True if dependency has run <= n times)

    """
    def check(dependencies):
        # This condition should only depend on one mechanism
        var = dependencies[0]
        # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
        num_calls = {"trial": var.component.calls_current_trial,
                     "run": var.component.calls_current_run - 1,
                     "life": var.component.calls_since_initialization - 1}
        if num_calls[time_scale] > n:
            return False
        return True
    return check

# To be implemented:
# def when_done(time_scale = 'trial', op = "AND"):
#     """
#     Condition function to be applied to a constraint
#     Enforces condition that dependencies must reach a threshold value before the owner's first run
#     Owner then continues to run according to its other conditions
#
#     Parameters
#     ----------
#     threshold - value that dependencies must reach
#     op -- "AND": condition must be true of all dependencies; "OR": condition must be true of at least one dependency
#
#
#     Returns
#     -------
#     Boolean (True if the number of dependencies required by op satisfy the threshold)
#
#     """
#
#     if op == "AND":
#         def check(dependencies, time_scale = 'trial'):
#             for var in dependencies:
#                 if not var.component.is_finished:
#                     return False
#             return True
#
#     elif op == "OR":
#         def check(dependencies, time_scale = 'trial'):
#             for var in dependencies:
#                 if var.component.is_finished:
#                     return True
#             return False
#
#     return check

# NOTE: this is a scheduler-level condition and when it returns True, the trial ends
def terminal(op = "AND"):
    """
    Condition function to be applied to a Scheduler-level constraint
    Enforces condition that trial must end after all terminal mechanisms have run or one terminal mechanisms has run,
    depending on whether op is set to "AND" or "OR", respectively

    Parameters
    -------
    op is set to "AND" if all terminal mechanisms must run or "OR" if only one terminal mechanism must run

    Returns
    -------
    Boolean (True if the number of terminal mechanisms required by op have run)

    """

    if op == "AND":

        def check(dependencies):
            for var in dependencies:
                if var.component.calls_current_trial == 0:
                    return False
            return True

    elif op == "OR":

        def check(dependencies):
            for var in dependencies:
                if var.component.calls_current_trial > 0:
                    return True
            return False

    return check


# NOTE: this is a scheduler-level condition, though a similar one could be added for mechanisms
# This mechanism always depends on the clock, and when it returns True, the trial ends
# Maybe we should have a naming convention for easily spotting scheduler constraints?
def num_time_steps(num):
    """
    Condition function to be applied to a Scheduler-level constraint
    Enforces condition that trial must end when clock has run num times

    Parameters
    -------
    num is the number of times the clock will run before the trial ends

    Returns
    -------
    Boolean (Returns True if clock has run num times)

    """
    def check(Clock):
        if Clock[0].component.calls_current_trial < num:
            return False
        return True
    return check


# NOTE: with the scheduler checking only after dependency each runs, this condition works as follows:
# Wait until dependency has run n times, then owner will run and continues to run *each time dependency runs*
# The OR case does not make much sense because owner will run * each time *each* dependency runs*, regardless of
# whether it was the dependency that triggered this condition to be true
def after_n_calls(n, time_scale = 'trial', op = "AND"):
    """
    Condition function to be applied to a constraint
    Enforces condition that dependencies must run n times on the given time scale before the owner's first run

    Parameters
    ----------
    n -- number of times that dependencies must run before owner can run
    time_scale -- time_scale on which to count (trial, run or life)
    op -- "AND": condition must be true of all dependencies; "OR": condition must be true of at least one dependency


    Returns
    -------
    Boolean (True if condition is met for the number of dependencies required by op)

    """

    if op == "AND":
        def check(dependencies):
            for var in dependencies:
                # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
                num_calls = {"trial": var.component.calls_current_trial,
                             "run": var.component.calls_current_run - 1,
                             "life": var.component.calls_since_initialization - 1}
                if num_calls[time_scale] < n:
                    return False
            return True

    elif op == "OR":
        def check(dependencies):
            for var in dependencies:
                # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
                num_calls = {"trial": var.component.calls_current_trial,
                             "run": var.component.calls_current_run - 1,
                             "life": var.component.calls_since_initialization - 1}
                if (num_calls[time_scale] >= n):
                    return True
            return False
    return check

def if_finished(time_scale = 'trial', op = 'AND'):
    """
    Condition function to be applied to a constraint
    Enforces condition that dependencies must have "is_finished" set to True before the owner's first run

    Parameters
    ----------
    time_scale -- time_scale on which to count (trial, run or life)
    op -- "AND": condition must be true of all dependencies; "OR": condition must be true of at least one dependency


    Returns
    -------
    Boolean (True if condition is met for the number of dependencies required by op)

    """
    if op == "AND":
        def check(dependencies):
            for var in dependencies:
                # # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
                # num_calls = {"trial": var.component.calls_current_trial,
                #              "run": var.component.calls_current_run - 1,
                #              "life": var.component.calls_since_initialization - 1}
                if var.component.is_finished is False:
                    return False
            return True

    elif op == "OR":
        def check(dependencies):
            for var in dependencies:
                # # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
                # num_calls = {"trial": var.component.calls_current_trial,
                #              "run": var.component.calls_current_run - 1,
                #              "life": var.component.calls_since_initialization - 1}
                if var.component.is_finished:
                    return True
            return False
    return check

