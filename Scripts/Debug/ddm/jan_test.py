import psyneulink as pnl
ocm_mode = "Python"

ddm = pnl.DDM(function=pnl.DriftDiffusionIntegrator(threshold=10,
                                                    time_step_size=1,
                                                    non_decision_time=0.6))
print(ddm.parameter_ports)

obj = pnl.ObjectiveMechanism(monitor=ddm.output_ports[pnl.RESPONSE_TIME])
comp = pnl.Composition(retain_old_simulation_data=True,
                       controller_mode=pnl.BEFORE)
comp.add_node(ddm, required_roles=pnl.NodeRole.INPUT)
comp.add_node(obj)

comp.add_controller(
    pnl.OptimizationControlMechanism(
        agent_rep=comp,
        state_features=[ddm.input_port],
        objective_mechanism=obj,
        control_signals=pnl.ControlSignal(
            modulates=(pnl.NON_DECISION_TIME, ddm),
            modulation=pnl.OVERRIDE,
            allocation_samples=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
    )
)
comp.controller.function.save_values = True
comp.controller.comp_execution_mode = ocm_mode

comp.run(inputs={ddm: [2]},
         num_trials=1)

print("SAVED VALUES:", comp.controller.function.saved_values)