class Constraint(object):
    ########
    # Helper class for scheduling
    # Contains an owner (the ScheduleVariable which owns this constraint), dependencies (list of all ScheduleVariables the
    # owner depends on with respect to *this* constraint), and the condition (of this constraint)
    # Contains a method 'is_satisfied' which checks dependencies against the condition and returns a boolean
    ########
    def __init__(self, owner, dependencies, condition, time_scales = None):
        self.owner = owner # ScheduleVariable that falls under this constraint
        if isinstance(dependencies, ScheduleVariable):
            self.dependencies = (dependencies,)
        else:
            self.dependencies = dependencies # Tuple of ScheduleVariables on which this constraint depends

        # currently, the following 3 lines are not used; time_scale is input explicitly, e.g. every_n_calls(1, "run")
        # and a default is set within each condition (usually 'trial')
        # self.time_scales = time_scales # Time Scales over which the constraint queries each dependency
        # if self.time_scales is None: # Defaults to 'trial'
        #     self.time_scales = ('trial',)*len(self.dependencies)

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
    # - If the component is a Terminal, self.once is initialized as False
    # Updates ---
    # - evaluate_constraint appends to filled_constraints and removes from unfilled_constraints if constraint is
    # satisfied. If component is terminal and constraint is satisfied, self.once is set to True
    # - new_time_step resets unfilled_constraints and filled_constraints
    # - new_trial calls new_trial() method on component to reset mechanism for a new trial
    ########
    def __init__(self, component, own_constraints = None, dependent_constraints = None, priority = None):
        self.component = component
        # Possible simplification - set default own_constraints = [] etc to avoid 'is not None' logic
        self.own_constraints = []
        self.unfilled_constraints = []
        self.filled_constraints = []
        if self.component.name == 'Terminal':
            self.once = False
        if own_constraints is not None:
            for con in own_constraints:
                self.add_own_constraint(con)
        self.dependent_constraints = []
        if dependent_constraints is not None:
            for con in dependent_constraints:
                self.add_dependent_constraint(con)
        self.priority = priority

    def add_own_constraint(self, constraint):
        self.own_constraints.append(constraint)
        self.unfilled_constraints.append(constraint)

    def add_dependent_constraint(self, constraint):
        self.dependent_constraints.append(constraint)

    def evaluate_constraint(self, constraint):
        result = constraint.is_satisfied()
        if result:
            self.filled_constraints.append(constraint)
            self.unfilled_constraints.remove(constraint)
            if self.component.name == 'Terminal':
                self.once = True
        return result

    def new_time_step(self):
        for con in self.filled_constraints:
            self.unfilled_constraints.append(con)
            self.filled_constraints.remove(con)

    def new_trial(self):
        self.component.new_trial()

# TerminalScheduleVariable is not currently used

# def TerminalScheduleVariable(ScheduleVariable):
#     def __init__(self, own_constraints = None, dependent_constraints = None, priority = None):
#         self.own_constraints = []
#         self.unfilled_constraints = []
#         self.filled_constraints = []
#         if own_constraints is not None:
#             for con in own_constraints:
#                 self.add_own_constraint(con)
#         self.dependent_constraints = []
#         if dependent_constraints is not None:
#             for con in dependent_constraints:
#                 self.add_dependent_constraint(con)
#         self.priority = priority

class Scheduler(object):
    ########
    # Constructor for Schedule
    #
    #
    ########

    def __init__(self, var_dict = None, terminal = None, clock = None, constraints = None):
        self.var_dict = {}
        self.constraints = []
        self.var_list = []
        if var_dict is not None:
            self.add_vars(var_dict)
        if clock is not None:
            self.clock = self.set_clock(clock)
        else:
            self.clock = None
        if terminal is not None:
            self.terminal = self.set_terminal(terminal)
        if constraints is not None:
            self.add_constraints(constraints)
        self.current_step = 0

    # Takes in var_list, a list of tuples of the form (component, priority)
    # Where component is the component (mechanism) object and priority is an int
    # Passes each component to ScheduleVariable, which constructs a ScheduleVariable object
    # Assembles var_dict which contains components as keys and their ScheduleVariables as values
    def add_vars(self, var_list):
        for var in var_list:
            self.var_dict[var[0]] = ScheduleVariable(var[0])
            self.var_dict[var[0]].priority = var[1]
            self.var_list.append(self.var_dict[var[0]])

    def add_constraints(self, constraints):
        for con in constraints:
            # Turn con into a Constraint object
            owner = self.var_dict[con[0]]
            if con[1] in self.var_dict:
                dependencies = (self.var_dict[con[1]],)
            else:
                dependencies = tuple((self.var_dict[mech] for mech in con[1]))
            con = Constraint(owner, dependencies, condition = con[2])

            # Add constraint to Scheduler?, owner, and dependencies
            self.constraints.append(con)
            con.owner.add_own_constraint(con)
            for var in con.dependencies:
                var.add_dependent_constraint(con)

    def set_clock(self, clock):
        self.clock = ScheduleVariable(clock)
        self.var_dict[clock] = self.clock
        self.var_dict[clock].priority = 0
        self.var_list.append(self.var_dict[clock])

    def set_terminal(self, terminal):
        self.terminal = ScheduleVariable(terminal)
        self.var_dict[terminal] = self.terminal
        self.var_dict[terminal].priority = 0
        self.var_list.append(self.var_dict[terminal])

    def run_time_step(self):
        def update_dependent_vars(variable):
            change_list = []
            for con in variable.dependent_constraints:
                if con in con.owner.unfilled_constraints:
                    if con.owner.evaluate_constraint(con):
                        change_list.append(con.owner)
            return change_list
        ## TODO: implement priority queue
        def update_firing_queue(firing_queue, change_list):
            for var in change_list:
                if var.unfilled_constraints == []:
                    firing_queue.append(var)
            return firing_queue

        for var in self.var_list:
            var.new_time_step()
        firing_queue = [self.clock]
        for var in firing_queue:
            var.component.execute()
            print(var.component.name)
            change_list = update_dependent_vars(var)
            firing_queue = update_firing_queue(firing_queue, change_list)

    # def generate_trial(self):
    #     for var in self.var_list:
    #         var.new_trial()
    #     trial_terminated = False
    #     while(not trial_terminated):
    #         for var in self.generate_time_step():
    #             if var is self.terminal:
    #                 trial_terminated = True
    #                 break
    #             else:
    #                 yield var.component

    def run_trial(self):
        for var in self.var_list:
            var.new_trial()
            var.once = False
        trial_terminated = False
        while(not trial_terminated):
            self.run_time_step()
            if self.terminal.once == True:
                trial_terminated = True
                break
            print('----------------')



def main():
    from PsyNeuLink.Components.Component import Component
    from PsyNeuLink.scheduling.condition import first_n_calls, every_n_calls
    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
    from PsyNeuLink.Components.Functions.Function import Linear
    A = TransferMechanism(function = Linear(), name = 'A')
    B = TransferMechanism(function = Linear(), name = 'B')
    C = TransferMechanism(function = Linear(), name = 'C')
    Clock = TransferMechanism(function = Linear(), name = 'Clock')
    Terminal = TransferMechanism(function = Linear(), name = 'Terminal')
    sched = Scheduler()
    sched.set_clock(Clock)
    sched.set_terminal(Terminal)
    sched.add_vars([(A, 1), (B, 2), (C, 3)])
    sched.add_constraints([
        # (A, (Clock,), every_n_calls(1)),
                           (A, (Clock,), first_n_calls(2)),
                           (B, (A,), every_n_calls(2)),
                           (C, (A,), every_n_calls(3)),
                           (Terminal, (C,), every_n_calls(2))])
    for var in sched.var_list:
        var.component.new_trial()
    # for i in range(10):
    #     sched.run_time_step()


    sched.run_trial()
    print("--- BEGINNING TRIAL 2 ---")
    sched.run_trial()


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

