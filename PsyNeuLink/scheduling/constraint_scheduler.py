from PsyNeuLink.Components.Component import Component
from PsyNeuLink.scheduling.condition import first_n_calls, every_n_calls
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Functions.Function import Linear

class Constraint(object):
    def __init__(self, owner, dependencies, condition, time_scales = None):
        self.owner = owner
        if isinstance(dependencies, ScheduleVariable):
            self.dependencies = (dependencies,)
        else:
            self.dependencies = dependencies
        self.time_scales = time_scales
        if self.time_scales is None:
            self.time_scales = ('trial',)*len(self.dependencies)
        self.condition = condition

    def is_satisfied(self):
        def resolve_time(var, scale):
            if scale == 'trial':
                return var.component.calls_current_trial
            elif scale == 'run':
                return var.component.calls_current_run
            elif scale == 'life':
                return var.component.calls_since_initialization
            else:
                pass
                #throw ValueError()
        if self.condition(*tuple((resolve_time(var, scale) for var, scale in zip(self.dependencies, self.time_scales)))):
            return True
        else:
            return False

class ScheduleVariable(object):
    def __init__(self, component, own_constraints = None, dependent_constraints = None, priority = None):
        self.component = component
        self.own_constraints = []
        self.unfilled_constraints = []
        self.filled_constraints = []
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
        return result

    def new_time_step(self):
        for con in self.filled_constraints:
            self.unfilled_constraints.append(con)
            self.filled_constraints.remove(con)

def TerminalScheduleVariable(ScheduleVariable):
    def __init__(self, own_constraints = None, dependent_constraints = None, priority = None):
        self.own_constraints = []
        self.unfilled_constraints = []
        self.filled_constraints = []
        if own_constraints is not None:
            for con in own_constraints:
                self.add_own_constraint(con)
        self.dependent_constraints = []
        if dependent_constraints is not None:
            for con in dependent_constraints:
                self.add_dependent_constraint(con)
        self.priority = priority

class Scheduler(object):

    def __init__(self, var_dict = None, terminal = None, clock = None, constraints = None):
        self.var_dict = {}
        self.constraints = []
        self.var_list = []
        self.clock = clock
        self.terminal = terminal
        if var_dict is not None:
            self.add_vars(var_dict)
        if constraints is not None:
            self.add_constraints(constraints)
        self.current_step = 0

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

    def generate_time_step(self):
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
            yield var
            change_list = update_dependent_vars(var)
            firing_queue = update_firing_queue(firing_queue, change_list)

    def generate_trial(self):
        trial_terminated = False
        while(not trial_terminated):
            for var in self.generate_time_step():
                if var is self.terminal:
                    trial_terminated = True
                    break
                else:
                    yield var.component
            print('----------------')



def main():
    A = TransferMechanism(function = Linear(), name = 'A')
    B = TransferMechanism(function = Linear(), name = 'B')
    C = TransferMechanism(function = Linear(), name = 'C')
    Clock = TransferMechanism(function = Linear(), name = 'Clock')
    Terminal = TransferMechanism(function=Linear(), name = 'Terminal')
    sched = Scheduler()
    sched.set_clock(Clock)
    sched.set_terminal(Terminal)
    sched.add_vars([(A, 1), (B, 2), (C, 3)])
    sched.add_constraints([(A, (Clock,), first_n_calls(12)),
                           (B, (A,), every_n_calls(2)),
                           (C, (B,), every_n_calls(2)),
                           (Terminal, (C,), every_n_calls(2))])
    for var in sched.var_list:
        var.component.new_trial()
    for mech in sched.generate_trial():
        mech.execute()
        print(mech.name)

    # for i in range(12):
    #     for result in sched.generate_time_step():
    #         mech.execute()

    #         print(mech.name)
    #     print('-----------------')



if __name__ == '__main__':
    main()