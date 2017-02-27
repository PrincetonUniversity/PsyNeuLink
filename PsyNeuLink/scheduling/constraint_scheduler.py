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
    # - self.own_constraints, self.unfilled_constraints, self.filled_constraints, self.dependent_constraints
    # are first initialized as empty lists, then immediately adjusted based on own_constraints and dependent_constraints
    # - add_own_constraint appends contents of own_constraints to self.own_constraints and self.unfilled_constraints
    # - add_dependent_constraint appends contents of dependent_constraints to self.dependent_constraints
    # Updates ---
    # - evaluate_constraint appends to filled_constraints and removes from unfilled_constraints if constraint is
    # satisfied. If component is terminal and constraint is satisfied, self.ran is set to True
    # - new_time_step resets unfilled_constraints and filled_constraints
    # - new_trial calls new_trial() method on component to reset mechanism for a new trial
    ########
    def __init__(self, component, own_constraints = [], dependent_constraints = [], priority = None):
        self.component = component
        # Possible simplification - set default own_constraints = [] etc to avoid 'is not None' logic
        self.own_constraints = []
        self.unfilled_constraints = []
        self.filled_constraints = []
        for con in own_constraints:
            self.add_own_constraint(con)
        self.dependent_constraints = []
        for con in dependent_constraints:
            self.add_dependent_constraint(con)
        self.priority = priority

    def add_own_constraint(self, constraint):
        self.own_constraints.append(constraint)
        self.unfilled_constraints.append(constraint)

    def add_dependent_constraint(self, constraint):
        self.dependent_constraints.append(constraint)

    def evaluate_constraint(self, constraint):
        ######
        # Takes in a constraint and checks whether it's been satisfied
        #
        result = constraint.is_satisfied()
        if result:
            self.filled_constraints.append(constraint)
            self.unfilled_constraints.remove(constraint)
            # if self.component.name == 'Terminal':
            #     self.ran = True
        return result

    def new_time_step(self):
        self.component.new_time_step()
        for con in self.filled_constraints:
            self.unfilled_constraints.append(con)
            self.filled_constraints.remove(con)

    def new_trial(self):
        self.component.new_trial()

class Scheduler(object):
    ########
    # Constructor for Scheduler
    # Initializes empty dictionary & empty list for ScheduleVariables, empty list for constraints
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
        # if terminal is not None:
        #     self.terminal = self.set_terminal(terminal)
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

    def add_constraints(self, constraints):
        #######
        # Takes in a list of tuples of constraint parameters
        # Passes constraint parameters to Constraint to assemble Constraint objects
        # Adds each Constraint object to its owner's and dependencies' ScheduleVariables
        #######
        for con in constraints:
            # Turn con into a Constraint object
            # Get owner[0], dependencies[1] and condition[2] out of each con
            if isinstance(con[0], Scheduler):
                owner = con[0]
            else:
                owner = self.var_dict[con[0]]
            if con[1] in self.var_dict:
                dependencies = (self.var_dict[con[1]],)
            else:
                dependencies = tuple((self.var_dict[mech] for mech in con[1]))
            con = Constraint(owner, dependencies, condition = con[2])
            if isinstance(owner, ScheduleVariable):
                con.owner.add_own_constraint(con)
            # Add constraint to Scheduler?, owner, and dependencies
            self.constraints.append(con)
            for var in con.dependencies:
                var.add_dependent_constraint(con)

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
            for con in variable.dependent_constraints:
                if isinstance(con.owner, Scheduler):
                    if con.is_satisfied():
                        self.trial_terminated = True
                elif con in con.owner.unfilled_constraints:
                    if con.owner.evaluate_constraint(con):
                        change_list.append(con.owner)
                change_list.sort(key=lambda x:x.priority) # sort according to priority
            return change_list

        def update_firing_queue(firing_queue, change_list):
            ######
            # Takes in the current firing queue & list of schedule variables that own a recently satisfied constraint
            # Any ScheduleVariable with no remaining constraints is added to the firing queue
            ######

            for var in change_list:
                if var.unfilled_constraints == []:
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
    from PsyNeuLink.Components.Component import Component
    from PsyNeuLink.scheduling.condition import first_n_calls_AND, every_n_calls, first_n_calls_OR, over_threshold_OR, terminal_AND, terminal_OR, num_time_steps
    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
    from PsyNeuLink.Components.Functions.Function import Linear
    A = TransferMechanism(function = Linear(slope=3, intercept=3), name = 'A')
    B = TransferMechanism(function = Linear(), name = 'B')
    C = TransferMechanism(function = Linear(), name = 'C')
    Clock = TransferMechanism(function = Linear(), name = 'Clock')
    T = TransferMechanism(function = Linear(), name = 'Terminal')
    sched = Scheduler()
    sched.set_clock(Clock)
    sched.add_vars([(A, 1), (B, 2), (C, 3), (T, 0)])
    sched.add_constraints([(A, (Clock,), every_n_calls(1)),
                           (B, (A,), every_n_calls(2)),
                           (C, (B,), every_n_calls(2)),
                           (T, (C,), every_n_calls(2)),
                           (sched, (Clock,), num_time_steps(2))])
    for var in sched.var_list:
        var.component.new_trial()


    sched.run_trial()
    print("--- BEGINNING TRIAL 2 ---")
    sched.run_trial()
    A.execute()
    A.execute()

    # for mech in sched.generate_trial():
    #     mech.execute()
    #     print(mech.name)
    print('=================================')
    # for mech in sched.generate_trial():
    #     mech.execute()
    #     print(mech.name)


    # for i in range(12):
    #     for result in sched.generate_time_step():
    #         mech.execute()

    #         print(mech.name)
    #     print('-----------------')

if __name__ == '__main__':
    main()

