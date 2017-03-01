
# every_n_calls only works with one mechanism
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


#
# def every_n_calls(n, time_scale = 'trial'):
#     """
#     Condition to be applied to a constraint
#     Enforces condition that all constraint dependencies must run n times before each time the constraint owner runs
#
#     Parameters
#     ----------
#     n -- number of times dependencies run before owner runs
#     time_scale -- time_scale on which to count (trial, run or life)
#
#     Returns
#     -------
#     Boolean (True if all dependencies satisfy the conditions, False if *any* do not)
#
#     """
#     def check(dependencies):
#         for var in dependencies:
#             # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
#             num_calls = {"trial": var.component.calls_current_trial,
#                          "run": var.component.calls_current_run - 1,
#                          "life": var.component.calls_since_initialization - 1}
#             if not (num_calls[time_scale]%n == 0 and num_calls[time_scale] != 0):
#                 return False
#         return True
#     return check
#

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
    Boolean (True if dependency has run <= n times

    """
    def check(dependencies):
        # This condition should only depend on one mechanism
        var = dependencies[0]
        # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
        num_calls = {"trial": var.component.calls_current_trial,
                     "run": var.component.calls_current_run - 1,
                     "life": var.component.calls_since_initialization - 1}
        if num_calls[time_scale] >= n:
            return False
        return True
    return check

#
# def first_n_calls_AND(n, time_scale = 'trial'):
#     """
#     Condition function to be applied to a constraint
#     Enforces condition that dependencies must run n times on the given time scale before the owner's first run
#
#     Parameters
#     ----------
#     n -- number of times that dependencies must run before owner can run
#     time_scale -- time_scale on which to count (trial, run or life)
#
#     Returns
#     -------
#     Boolean (True if all dependencies satisfy the conditions, False if *any* do not)
#
#     """
#     def check(dependencies):
#         for var in dependencies:
#             # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
#             num_calls = {"trial": var.component.calls_current_trial,
#                          "run": var.component.calls_current_run -1 ,
#                          "life": var.component.calls_since_initialization - 1}
#             if (num_calls[time_scale] < n):
#                 return False
#         return True
#     return check
#
#
# def first_n_calls_OR(n, time_scale = 'trial'):
#     """
#     Condition function to be applied to a constraint
#     Enforces condition that dependencies must run n times on the given time scale before the owner's first run
#     Owner then continues to run every time step
#
#     Parameters
#     ----------
#     n -- number of times that dependencies must run before owner can run
#     time_scale -- time_scale on which to count (trial, run or life)
#
#     Returns
#     -------
#     Boolean (True if all dependencies satisfy the conditions, False if *any* do not)
#
#     """
#     def check(dependencies):
#         for var in dependencies:
#             # calls_current_run and calls_since_initialization currently have an offset of 1 due to initialization run
#             num_calls = {"trial": var.component.calls_current_trial,
#                          "run": var.component.calls_current_run -1 ,
#                          "life": var.component.calls_since_initialization - 1}
#             if (num_calls[time_scale] >= n):
#                 return True
#         return False
#     return check

def over_threshold(threshold,  time_scale = 'trial', op = "AND"):
    """
    Condition function to be applied to a constraint
    Enforces condition that dependencies must reach a threshold value before the owner's first run
    Owner then continues to run according to its other conditions

    Parameters
    ----------
    threshold - value that dependencies must reach
    op -- "AND": condition must be true of all dependencies; "OR": condition must be true of at least one dependency


    Returns
    -------
    Boolean (True if the number of dependencies required by op satisfy the threshold)

    """

    if op == "AND":
        def check(dependencies, time_scale = 'trial'):
            for var in dependencies:
                if abs(float(var.component.value)) < threshold:
                    return False
            return True

    elif op == "OR":
        def check(dependencies, time_scale = 'trial'):
            for var in dependencies:
                if abs(float(var.component.value)) >= threshold:
                    return True
            return False

    return check

# def over_threshold_OR(threshold, time_scale = 'trial'):
#     """
#     Condition function to be applied to a constraint
#     Enforces condition that dependencies must reach a threshold value before the owner's first run
#     Owner then continues to run according to its other conditions
#
#     Parameters
#     ----------
#     threshold - value that dependencies must reach
#
#     Returns
#     -------
#     Boolean (True if any dependencies satisfy the threshold)
#
#     """
#     def check(dependencies, time_scale = 'trial'):
#         for var in dependencies:
#             print(var.component.value)
#             if abs(float(var.component.value)) >= threshold:
#                 return True
#         return False
#     return check
#
# def over_threshold_AND(threshold, time_scale = 'trial'):
#     """
#     Condition function to be applied to a constraint
#     Enforces condition that dependencies must reach a threshold value before the owner's first run
#     Owner then continues to run according to its other conditions
#
#     Parameters
#     ----------
#     threshold - value that dependencies must reach
#
#     Returns
#     -------
#     Boolean (True if all dependencies satisfy the threshold)
#
#     """
#     def check(dependencies, time_scale = 'trial'):
#         for var in dependencies:
#             if abs(float(var.component.value)) < threshold:
#                 return False
#         return True
#     return check
#

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


# def terminal_AND():
#     """
#     Condition function to be applied to a Scheduler-level constraint
#     Enforces condition that trial must end after all terminal mechanisms have run
#
#     Returns
#     -------
#     Boolean (True if all terminal mechanisms have run)
#
#     """
#     def check(dependencies):
#         for var in dependencies:
#             if var.component.calls_current_trial == 0:
#                 return False
#         return True
#     return check
#
# def terminal_OR():
#     """
#     Condition function to be applied to a Scheduler-level constraint
#     Enforces condition that trial must end if any of terminal mechanisms has run
#
#     Returns
#     -------
#     Boolean (True if at least one terminal mechanism has run)
#
#     """
#     def check(dependencies):
#         for var in dependencies:
#             if var.component.calls_current_trial > 0:
#                 return True
#         return False
#     return check

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