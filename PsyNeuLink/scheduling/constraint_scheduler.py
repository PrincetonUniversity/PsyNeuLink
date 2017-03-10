from PsyNeuLink.scheduling.condition import first_n_calls, every_n_calls, terminal, num_time_steps, after_n_calls


class Constraint(object):
    ########
    # Helper class for scheduling
    # Contains an owner (the ScheduleVariable which owns this constraint), dependencies (list of all ScheduleVariables
    # the owner depends on with respect to *this* constraint), and the condition (of this constraint)
    # Contains a method 'is_satisfied' which checks dependencies against the condition and returns a boolean
    ########
    def __init__(self, owner, dependencies, condition, time_scales = None):
        self.owner = owner # ScheduleVariable that falls under this constraint
        if isinstance(dependencies, ScheduleVariable):
            self.dependencies = (dependencies,)
        else:
            self.dependencies = dependencies # Tuple of ScheduleVariables on which this constraint depends

        self.condition = condition # Condition to be evaluated

    def is_satisfied(self):
        return self.condition(self.dependencies) #Checks dependencies against condition; returns True if ALL satisfied

class ScheduleVariable(object):
    ########
    # Helper class for Scheduler
    # Creates a ScheduleVariable which contains a component (typically a mechanism) to be scheduled
    # Initialization ---
    # - self.own_constraint_sets, self.unfilled_constraint_sets, self.filled_constraint_sets, self.dependent_constraint_sets
    # are first initialized as empty lists, then immediately adjusted based on own_constraint_sets and dependent_constraints
    # - add_own_constraint_set appends contents of own_constraint_sets to self.own_constraint_sets and self.unfilled_constraint_sets
    # - add_dependent_constraint_set appends contents of dependent_constraints to self.dependent_constraints
    # Updates ---
    # - evaluate_constraint_set appends to filled_constraint_sets and removes from unfilled_constraint_sets if constraint is
    # satisfied. If component is terminal and constraint is satisfied, self.ran is set to True
    # - new_time_step resets unfilled_constraint_sets and filled_constraint_sets
    # - new_trial calls new_trial() method on component to reset mechanism for a new trial
    ########
    def __init__(self, component, own_constraint_sets = [], dependent_constraint_sets = [], priority = None):
        self.component = component
        # Possible simplification - set default own_constraint_sets = [] etc to avoid 'is not None' logic
        self.own_constraint_sets = []
        self.unfilled_constraint_sets = []
        self.filled_constraint_sets = []
        # own_constraint_sets is a list of constraint sets, which are lists of constraint objects
        for con_set in own_constraint_sets:
            for con in con_set:
                self.add_own_constraint_set(con)
        self.dependent_constraint_sets = []
        for con_set in dependent_constraint_sets:
            for con in con_set:
                self.add_dependent_constraint_set(con)
        self.priority = priority

    def add_own_constraint_set(self, constraint):
        self.own_constraint_sets.append(constraint)
        self.unfilled_constraint_sets.append(constraint)

    def add_dependent_constraint_set(self, constraint):
        self.dependent_constraint_sets.append(constraint)

    def evaluate_constraint_set(self, constraint_set):
        ######
        # Takes in a constraint set and checks whether it's been satisfied
        ######
        for constraint in constraint_set:
            result = constraint.is_satisfied()
            if result:
                self.filled_constraint_sets.append(constraint_set)
                self.unfilled_constraint_sets.remove(constraint_set)
                return result
        return result

    def new_time_step(self):
        self.component.new_time_step()
        for con in self.filled_constraint_sets:
            self.unfilled_constraint_sets.append(con)
            self.filled_constraint_sets.remove(con)

    def new_trial(self):
        self.component.new_trial()

class Scheduler(object):
    ########
    # Constructor for Scheduler
    # Initializes empty dictionary & empty list for ScheduleVariables, empty list for constraints
    # then populates each with values passed into var_dict and constraints parameters
    # run_time_step and run_trial carries out scheduling logic by working through each ScheduleVariable,
    # its constraints, and its dependent mechanisms, beginning with the clock
    ########
    def __init__(self, var_dict = None, clock = None, constraints = None):
        self.var_dict = {}
        self.constraints = []
        self.var_list = []
        self.trial_terminated = False
        if var_dict is not None:
            self.add_vars(var_dict)
        if clock is not None:
            self.clock = self.set_clock(clock)
        else:
            self.clock = None
        if constraints is not None:
            self.add_constraints(constraints)
        self.current_step = 0


    def add_vars(self, var_list):
        #######
        # Takes in var_list, list of tuples of the form (component, priority), where component=mechanism, priority=int
        # Passes each component to ScheduleVariable, which constructs a ScheduleVariable object
        # Assembles var_dict which contains components as keys and their ScheduleVariables as values
        #######
        for var in var_list:
            self.var_dict[var[0]] = ScheduleVariable(var[0])
            self.var_dict[var[0]].priority = var[1]
            self.var_list.append(self.var_dict[var[0]])

    def add_constraints(self, constraint_sets):
        #######
        # Takes in a list of lists of tuples of constraint parameters
        # Passes constraint parameters to Constraint to assemble Constraint objects
        # Adds sets of Constraint objects to their owner's and dependencies' ScheduleVariables
        #######
        for con_set in constraint_sets:

            # create an empty list for storing the constraint objects that make up each constraint set
            constraint_objects_in_set = []

            for con in con_set:

                # Turn con into a Constraint object
                # Get owner[0], dependencies[1] and condition[2] out of each con
                if isinstance(con[0], Scheduler):   # special case where the constraint is on the entire Scheduler
                    owner = con[0]
                else:
                    owner = self.var_dict[con[0]]   # typical case where the constraint is on a ScheduleVariable

                if con[1] in self.var_dict:
                    dependencies = (self.var_dict[con[1]],)
                else:
                    dependencies = tuple((self.var_dict[mech] for mech in con[1]))

                con = Constraint(owner, dependencies, condition = con[2])

                constraint_objects_in_set.append(con)

            for con in constraint_objects_in_set:
                # Add constraint as a dependent constraint on dependencies
                for var in con.dependencies:
                    var.add_dependent_constraint_set(constraint_objects_in_set)

                if isinstance(owner, ScheduleVariable):
                    con.owner.add_own_constraint_set(constraint_objects_in_set)

            # Add constraint to the owner's ScheduleVariable
            if isinstance(owner, ScheduleVariable):
                owner.unfilled_constraint_sets.append(constraint_objects_in_set)
            # Add constraint to Scheduler and to list of constraints in this set
            self.constraints.append(constraint_objects_in_set)

    def set_clock(self, clock):
        #######
        # create a ScheduleVariable for clock and give it priority of zero
        #######
        self.clock = ScheduleVariable(clock)
        self.var_dict[clock] = self.clock
        self.var_dict[clock].priority = 0
        self.var_list.append(self.var_dict[clock])

    def run_time_step(self):
        #######
        # Resets all mechanisms in the Scheduler for this time_step
        # Initializes a firing queue, then continuously executes mechanisms and updates queue according to any
        # constraints that were satisfied by the previous execution
        #######
        def update_dependent_vars(variable):
            #######
            # Takes in the ScheduleVariable of the mechanism that *just* ran
            # Loops through all of the constraints that depend on this mechanism
            # Returns a list ('change_list') of all of the ScheduleVariables (mechanisms) that own a constraint which
            # was satisfied by this mechanism's run
            #######

            change_list = []

            for con_set in variable.dependent_constraint_sets:
                for con in con_set:
                    if isinstance(con.owner, Scheduler):        # special case where the constraint is on the Scheduler
                        if con.is_satisfied():                  # If the constraint is satisfied, end trial
                            self.trial_terminated = True

                    elif con_set in con.owner.unfilled_constraint_sets: # typical case where the constraint is on a ScheduleVariable
                        if con.owner.evaluate_constraint_set(con_set):  # If the constraint set is satisfied, pass owner to change list
                            change_list.append(con.owner)
                    change_list.sort(key=lambda x:x.priority)   # sort change list according to priority
            return change_list

        def update_firing_queue(firing_queue, change_list):
            ######
            # Takes in the current firing queue & list of schedule variables that own a recently satisfied constraint
            # Any ScheduleVariable with no remaining constraints is added to the firing queue
            ######

            for var in change_list:
                if len(var.filled_constraint_sets) == len(var.own_constraint_sets):
                    firing_queue.append(var)
            return firing_queue

        # reset all mechanisms for this time step
        for var in self.var_list:
            var.new_time_step()
        # initialize firing queue by adding clock
        firing_queue = [self.clock]
        for var in firing_queue:
            var.component.execute()
            print(var.component.name)
            change_list = update_dependent_vars(var)
            firing_queue = update_firing_queue(firing_queue, change_list)

    def run_trial(self):
        ######
        # Resets all mechanisms, then calls self.run_time_step() until the terminal mechanism runs
        ######

        # reset each mechanism for the trial
        for var in self.var_list:
            var.new_trial()

        # run time steps until terminal mechanism is run
        self.trial_terminated = False
        while(not self.trial_terminated):
            self.run_time_step()
            print('----------------')


def main():
    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
    from PsyNeuLink.Components.Functions.Function import Linear
    A = TransferMechanism(function = Linear(intercept=3.0), name = 'A')
    B = TransferMechanism(function = Linear(intercept=2.0), name = 'B')
    C = TransferMechanism(function = Linear(), name = 'C')
    Clock = TransferMechanism(function = Linear(), name = 'Clock')
    T = TransferMechanism(function = Linear(), name = 'Terminal')
    sched = Scheduler()
    sched.set_clock(Clock)
    sched.add_vars([(A, 1), (B, 2), (C, 3), (T, 0)])

    test_constraints_dict = {
                            # every_n
                             "Test 1": [[(A, (Clock,), every_n_calls(1))],
                               [(B, (A,), every_n_calls(2))],
                               [(C, (B,), every_n_calls(3))],
                               [(T, (C,), every_n_calls(4))],
                               [(sched, (T,), terminal())]],

                            "Test 1b": [[(A, (Clock,), every_n_calls(3))],
                                       [(B, (A,), every_n_calls(1)),(B, (Clock,), every_n_calls(2))],
                                       [(C, (B,), every_n_calls(3))],
                                       [(T, (C,), every_n_calls(4))],
                                       [(sched, (T,), terminal())]],

        # # after_n where C begins after 2 runs of B; C is terminal
                            #  "Test 2": [(A, (Clock,), every_n_calls(1)),
                            #             (B, (A,), every_n_calls(2)),
                            #             (C, (B,), after_n_calls(2)),
                            #             (sched, (C,), terminal())],
                            #
                            # # after_n where C begins after 2 runs of B; runs for 10 time steps
                            # "Test 3": [(A, (Clock,), every_n_calls(1)),
                            #            (B, (A,), every_n_calls(2)),
                            #            (C, (B,), after_n_calls(2)),
                            #            (sched, (Clock,), num_time_steps(10))],
                            #
                            # # after_n where C begins after 3 runs of B OR A; runs for 10 time steps
                            #  "Test 4": [(A, (Clock,), every_n_calls(1)),
                            #             (B, (A,), every_n_calls(2)),
                            #             (C, (B,A), after_n_calls(3, op = "OR")),
                            #             (sched, (Clock,), num_time_steps(10))],
                            #
                            # # after_n where C begins after 2 runs of B AND A; runs for 10 time steps
                            # "Test 5": [(A, (Clock,), every_n_calls(1)),
                            #             (B, (A,), every_n_calls(2)),
                            #             (C, (B,A), after_n_calls(3)),
                            #             (sched, (Clock,), num_time_steps(10))],
                            #
                            # # first n where A depends on the clock
                            # "Test 6": [(A, (Clock,), first_n_calls(5)),
                            #            (B, (A,), after_n_calls(5)),
                            #            (C, (B,), after_n_calls(1)),
                            #            (sched, (Clock,), num_time_steps(10))],
                            #
                            # # terminal where trial ends when A OR B runs
                            # "Test 7": [(A, (Clock,), every_n_calls(1)),
                            #            (B, (A,), every_n_calls(2)),
                            #            (sched, (A,B), terminal(op="OR"))],
                            #
                            # # terminal where trial ends when A AND B have run
                            # "Test 8": [(A, (Clock,), every_n_calls(1)),
                            #            (B, (A,), every_n_calls(2)),
                            #            (sched, (A, B), terminal())],

                              }

    test = "Test 1b"
    sched.add_constraints(test_constraints_dict[test])

    for var in sched.var_list:
        var.component.new_trial()

    sched.run_trial()
    print("--- BEGINNING TRIAL 2 ---")
    sched.run_trial()

    print('=================================')

if __name__ == '__main__':
    main()

