import psyneulink as pnl
from psyneulink.core.globals.keywords import TERSE, CAPTURE, FULL


class TestProgress():
    # FIX: NEED TO DEAL WITH INDETERMINACY OF a OR b FIRST IN TIME_STEP

    def test_simple_output_and_progress(self):
        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')
        c = pnl.TransferMechanism(name='c')
        comp = pnl.Composition(pathways=[a,b,c], name='COMP')

        a.reportOutputPref=True
        b.reportOutputPref=False
        c.reportOutputPref=True

        comp.run(show_output=TERSE, show_progress=[False, CAPTURE])
        actual_output = comp.run_output
        expected_output = '\'\\nCOMP TRIAL 0 ====================\\nTime Step 0 ---------\\na executed\\nTime Step 1 ---------\\nb executed\\nTime Step 2 ---------\\nc executed\\n\''
        assert repr(actual_output) == expected_output

        comp.run(show_output=TERSE, show_progress=CAPTURE)
        actual_output = comp.run_output
        expected_output = '\'\\nCOMP TRIAL 0 ====================\\nTime Step 0 ---------\\na executed\\nTime Step 1 ---------\\nb executed\\nTime Step 2 ---------\\nc executed\\nCOMP: Executed 1 of 1 trials\''
        assert repr(actual_output) == expected_output

        comp.run(show_output=True, show_progress=[False, CAPTURE])
        actual_output = comp.run_output
        expected_output = '\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\n╭───── a ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nTime Step 1 ---------\nTime Step 2 ---------\n╭───── c ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\n'
        assert actual_output == expected_output

        comp.run(show_output=True, show_progress=CAPTURE)
        actual_output = comp.run_output
        expected_output = '\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\n╭───── a ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nTime Step 1 ---------\nTime Step 2 ---------\n╭───── c ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nCOMP: Executed 1 of 1 trials'
        assert actual_output == expected_output

        comp.run(show_output=FULL, show_progress=[False, CAPTURE])
        actual_output = comp.run_output
        expected_output = '\n┏━━  COMP: Trial 0  ━━┓\n┃                     ┃\n┃ input: [[0.0]]      ┃\n┃                     ┃\n┃ ┌─  Time Step 0 ──┐ ┃\n┃ │ ╭───── a ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 1 ──┐ ┃\n┃ │ ╭───── b ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 2 ──┐ ┃\n┃ │ ╭───── c ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ result: [[0.0]]     ┃\n┃                     ┃\n┗━━━━━━━━━━━━━━━━━━━━━┛\n\n'
        assert actual_output == expected_output

        comp.run(show_output=FULL, show_progress=CAPTURE)
        actual_output = comp.run_output
        expected_output = '\n┏━━  COMP: Trial 0  ━━┓\n┃                     ┃\n┃ input: [[0.0]]      ┃\n┃                     ┃\n┃ ┌─  Time Step 0 ──┐ ┃\n┃ │ ╭───── a ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 1 ──┐ ┃\n┃ │ ╭───── b ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 2 ──┐ ┃\n┃ │ ╭───── c ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ result: [[0.0]]     ┃\n┃                     ┃\n┗━━━━━━━━━━━━━━━━━━━━━┛\n\nCOMP: Executed 1 of 1 trials'
        assert actual_output == expected_output

        # Run these tests after ones calling run() above to avoid having to reset trial counter,
        # which increments after calls to execute()
        comp.execute(show_output=TERSE, show_progress=[False, CAPTURE])
        actual_output = comp.run_output
        expected_output = '\'\\nCOMP TRIAL 0 ====================\\nTime Step 0 ---------\\na executed\\nTime Step 1 ---------\\nb executed\\nTime Step 2 ---------\\nc executed\\n\''
        assert repr(actual_output) == expected_output

        comp.execute(show_output=TERSE, show_progress=[CAPTURE])
        actual_output = comp.run_output
        expected_output = '\'\\nCOMP TRIAL 1 ====================\\nTime Step 0 ---------\\na executed\\nTime Step 1 ---------\\nb executed\\nTime Step 2 ---------\\nc executed\\n[red]Executing COMP...\''
        assert repr(actual_output) == expected_output

        comp.execute(show_output=True, show_progress=[False, CAPTURE])
        actual_output = comp.run_output
        expected_output = '\nCOMP TRIAL 2 ====================\nTime Step 0 ---------\n╭───── a ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nTime Step 1 ---------\nTime Step 2 ---------\n╭───── c ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\n'
        assert actual_output == expected_output

        comp.execute(show_output=True, show_progress=[CAPTURE])
        actual_output = comp.run_output
        expected_output = '\nCOMP TRIAL 3 ====================\nTime Step 0 ---------\n╭───── a ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nTime Step 1 ---------\nTime Step 2 ---------\n╭───── c ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\n[red]Executing COMP...'
        assert actual_output == expected_output

        comp.execute(show_output=FULL, show_progress=[False, CAPTURE])
        actual_output = comp.run_output
        expected_output = '\n┏━━  COMP: Trial 4  ━━┓\n┃                     ┃\n┃ input: [[0.0]]      ┃\n┃                     ┃\n┃ ┌─  Time Step 0 ──┐ ┃\n┃ │ ╭───── a ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 1 ──┐ ┃\n┃ │ ╭───── b ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 2 ──┐ ┃\n┃ │ ╭───── c ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ result: [[0.0]]     ┃\n┃                     ┃\n┗━━━━━━━━━━━━━━━━━━━━━┛\n\n'
        assert actual_output == expected_output

        comp.execute(show_output=FULL, show_progress=[CAPTURE])
        actual_output = comp.run_output
        expected_output = '\n┏━━  COMP: Trial 5  ━━┓\n┃                     ┃\n┃ input: [[0.0]]      ┃\n┃                     ┃\n┃ ┌─  Time Step 0 ──┐ ┃\n┃ │ ╭───── a ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 1 ──┐ ┃\n┃ │ ╭───── b ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ ┌─  Time Step 2 ──┐ ┃\n┃ │ ╭───── c ─────╮ │ ┃\n┃ │ │ input: 0.0  │ │ ┃\n┃ │ │ output: 0.0 │ │ ┃\n┃ │ ╰─────────────╯ │ ┃\n┃ └─────────────────┘ ┃\n┃                     ┃\n┃ result: [[0.0]]     ┃\n┃                     ┃\n┗━━━━━━━━━━━━━━━━━━━━━┛\n\n[red]Executing COMP...'
        assert actual_output == expected_output

    # def test_two_mechs_in_a_time_step(self):
    #     a = TransferMechanism(name='a')
    #     b = TransferMechanism(name='b')
    #     c = TransferMechanism(name='c')
    #     comp = Composition(pathways=[[a,b],[b,a], [a,c]], name='COMP')
    #
    #     a.reportOutputPref=True
    #     b.reportOutputPref=False
    #     c.reportOutputPref=True
    #
    #     comp.run(show_output=TERSE, show_progress=[False, CAPTURE])
    #     actual_output = comp.run_output
    #     expected_output = '\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\na executed\nb executed\nTime Step 1 ---------\nc executed\n\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\na executed\nb executed\nTime Step 1 ---------\nc executed\nCOMP TRIAL 1 ====================\nTime Step 0 ---------\na executed\nb executed\nTime Step 1 ---------\nc executed\n'
    #     assert actual_output == expected_output
    #
    #     comp.run(show_output=TERSE, show_progress=CAPTURE)
    #     actual_output = comp.run_output
    #     expected_output = '\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\nb executed\na executed\nTime Step 1 ---------\nc executed\nCOMP: Executed 1 of 2 trials\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\nb executed\na executed\nTime Step 1 ---------\nc executed\nCOMP TRIAL 1 ====================\nTime Step 0 ---------\nb executed\na executed\nTime Step 1 ---------\nc executed\nCOMP: Executed 2 of 2 trials'
    #     assert actual_output == expected_output
    #
    #     comp.run(show_output=True, show_progress=CAPTURE)
    #     actual_output = comp.run_output
    #     expected_output = '\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\n╭───── a ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nTime Step 1 ---------\n╭───── c ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nCOMP: Executed 1 of 2 trials\nCOMP TRIAL 0 ====================\nTime Step 0 ---------\n╭───── a ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nTime Step 1 ---------\n╭───── c ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nCOMP TRIAL 1 ====================\nTime Step 0 ---------\n╭───── a ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nTime Step 1 ---------\n╭───── c ─────╮\n│ input: 0.0  │\n│ output: 0.0 │\n╰─────────────╯\nCOMP: Executed 2 of 2 trials'
    #     assert actual_output == expected_output
    #
    #     comp.run(num_trials=2, show_output=FULL, show_progress=CAPTURE)
    #     actual_output = comp.run_output
    #     expected_output = '\n┏━━━  COMP: Trial 0  ━━━┓\n┃                       ┃\n┃ input: [[0.0], [0.0]] ┃\n┃                       ┃\n┃ ┌─  Time Step 0 ──┐   ┃\n┃ │ ╭───── b ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ │ ╭───── a ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ └─────────────────┘   ┃\n┃                       ┃\n┃ ┌─  Time Step 1 ──┐   ┃\n┃ │ ╭───── c ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ └─────────────────┘   ┃\n┃                       ┃\n┃ result: [[0.0]]       ┃\n┃                       ┃\n┗━━━━━━━━━━━━━━━━━━━━━━━┛\n\nCOMP: Executed 1 of 2 trials\n┏━━━  COMP: Trial 0  ━━━┓\n┃                       ┃\n┃ input: [[0.0], [0.0]] ┃\n┃                       ┃\n┃ ┌─  Time Step 0 ──┐   ┃\n┃ │ ╭───── b ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ │ ╭───── a ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ └─────────────────┘   ┃\n┃                       ┃\n┃ ┌─  Time Step 1 ──┐   ┃\n┃ │ ╭───── c ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ └─────────────────┘   ┃\n┃                       ┃\n┃ result: [[0.0]]       ┃\n┃                       ┃\n┗━━━━━━━━━━━━━━━━━━━━━━━┛\n\n┏━━━  COMP: Trial 1  ━━━┓\n┃                       ┃\n┃ input: [[0.0], [0.0]] ┃\n┃                       ┃\n┃ ┌─  Time Step 0 ──┐   ┃\n┃ │ ╭───── b ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ │ ╭───── a ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ └─────────────────┘   ┃\n┃                       ┃\n┃ ┌─  Time Step 1 ──┐   ┃\n┃ │ ╭───── c ─────╮ │   ┃\n┃ │ │ input: 0.0  │ │   ┃\n┃ │ │ output: 0.0 │ │   ┃\n┃ │ ╰─────────────╯ │   ┃\n┃ └─────────────────┘   ┃\n┃                       ┃\n┃ result: [[0.0]]       ┃\n┃                       ┃\n┗━━━━━━━━━━━━━━━━━━━━━━━┛\n\nCOMP: Executed 2 of 2 trials'
    #     assert actual_output == expected_output

    def test_nested_comps_output(self):

        with_inner_controller = True
        with_outer_controller = True

        # instantiate mechanisms and inner comp
        ia = pnl.TransferMechanism(name='ia')
        ib = pnl.TransferMechanism(name='ib')
        icomp = pnl.Composition(name='icomp', controller_mode=pnl.BEFORE)

        # set up structure of inner comp
        icomp.add_node(ia, required_roles=pnl.NodeRole.INPUT)
        icomp.add_node(ib, required_roles=pnl.NodeRole.OUTPUT)
        icomp.add_projection(pnl.MappingProjection(), sender=ia, receiver=ib)

        # add controller to inner comp
        if with_inner_controller:
            icomp.add_controller(
                    pnl.OptimizationControlMechanism(
                            agent_rep=icomp,
                            features=[ia.input_port],
                            name="iController",
                            objective_mechanism=pnl.ObjectiveMechanism(
                                    monitor=ib.output_port,
                                    function=pnl.SimpleIntegrator,
                                    name="iController Objective Mechanism"
                            ),
                            function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                            control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                               variable=1.0,
                                                               intensity_cost_function=pnl.Linear(slope=0.0),
                                                               allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                                 stop=10.0,
                                                                                                 num=4))])
            )

        # instantiate outer comp
        ocomp = pnl.Composition(name='ocomp', controller_mode=pnl.BEFORE)

        # setup structure of outer comp
        ocomp.add_node(icomp)

        ocomp._analyze_graph()

        # add controller to outer comp
        if with_outer_controller:
            ocomp.add_controller(
                    pnl.OptimizationControlMechanism(
                            agent_rep=ocomp,
                            # features=[ia.input_port],
                            features=[ocomp.input_CIM.output_ports[0]],
                            name="oController",
                            objective_mechanism=pnl.ObjectiveMechanism(
                                    monitor=ib.output_port,
                                    function=pnl.SimpleIntegrator,
                                    name="oController Objective Mechanism"
                            ),
                            function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                            control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                               variable=1.0,
                                                               intensity_cost_function=pnl.Linear(slope=0.0),
                                                               allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                                 stop=10.0,
                                                                                                 num=3))])
            )

        inputs_dict = {
            icomp:
                {
                    ia: [[-2], [1]]
                }
        }

        def inputs_generator_function():
            for i in range(2):
                yield {
                    icomp:
                        {
                            ia: inputs_dict[icomp][ia][i]
                        }
                }

        inputs_generator_instance = inputs_generator_function()

        # ocomp.run(inputs=inputs_generator_function)
        # ocomp.run(inputs=inputs_generator_instance, show_progress=['simulations'])
        # ocomp.run(inputs=inputs_dict, show_progress=True)
        # ocomp.run(inputs=inputs_dict, show_progress=['simulations'])
        ocomp.run(inputs={icomp:-2}, show_output=FULL, show_progress=['simulations',CAPTURE])
        print(ocomp.run_output)
        # ocomp.execute(inputs={icomp:-2}, context='0')
