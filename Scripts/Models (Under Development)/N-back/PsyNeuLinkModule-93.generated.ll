; ModuleID = "PsyNeuLinkModule-93"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_TransferMechanism_CURRENT_STIMULUS__101"({{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* noalias nonnull %".1", {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* noalias nonnull %".2", [1 x [1 x [8 x double]]]* noalias nonnull %".3", {[8 x double]}* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %"ptr_state_is_finished_flag_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 3 , !dbg !7
  %"ptr_state_is_finished_flag_CURRENT STIMULUS_hist0" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 3, i32 0 , !dbg !7
  %"ptr_state_num_executions_before_finished_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 4 , !dbg !7
  %"ptr_state_num_executions_before_finished_CURRENT STIMULUS_hist0" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 4, i32 0 , !dbg !7
  %"ptr_param_max_executions_before_finished_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1", i32 0, i32 5 , !dbg !7
  %".6" = load double, double* %"ptr_state_is_finished_flag_CURRENT STIMULUS_hist0", !dbg !7
  %".7" = fcmp oeq double %".6", 0x3ff0000000000000 , !dbg !7
  br i1 %".7", label %"entry.if", label %"entry.endif", !dbg !7
entry.if:
  store double              0x0, double* %"ptr_state_num_executions_before_finished_CURRENT STIMULUS_hist0", !dbg !7
  br label %"entry.endif", !dbg !7
entry.endif:
  br label %"entry.endif_loop", !dbg !7
entry.endif_loop:
  %".12" = call i1 @"_TransferMechanism_CURRENT_STIMULUS__101_internal_102"({{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1", {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", [1 x [1 x [8 x double]]]* %".3", {[8 x double]}* %".4", {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1"), !dbg !7
  %".13" = load double, double* %"ptr_state_num_executions_before_finished_CURRENT STIMULUS_hist0", !dbg !7
  %".14" = fadd double %".13", 0x3ff0000000000000, !dbg !7
  store double %".14", double* %"ptr_state_num_executions_before_finished_CURRENT STIMULUS_hist0", !dbg !7
  %".16" = load double, double* %"ptr_param_max_executions_before_finished_CURRENT STIMULUS", !dbg !7
  %".17" = fcmp oge double %".14", %".16" , !dbg !7
  %"ptr_param_execute_until_finished_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1", i32 0, i32 4 , !dbg !7
  %".18" = load double, double* %"ptr_param_execute_until_finished_CURRENT STIMULUS", !dbg !7
  %".19" = fcmp oeq double %".18",              0x0 , !dbg !7
  %".20" = or i1 %".12", %".17", !dbg !7
  %".21" = or i1 %".20", %".19", !dbg !7
  %"ptr_param_integrator_mode_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1", i32 0, i32 3 , !dbg !7
  %".22" = load double, double* %"ptr_param_integrator_mode_CURRENT STIMULUS", !dbg !7
  %".23" = fcmp oeq double %".22",              0x0 , !dbg !7
  %".24" = or i1 %".21", %".23", !dbg !7
  br i1 %".24", label %"entry.endif_loop.if", label %"entry.endif_loop.endif", !dbg !7
entry.endif_end:
  ret void, !dbg !7
entry.endif_loop.if:
  %".26" = uitofp i1 %".20" to double , !dbg !7
  store double %".26", double* %"ptr_state_is_finished_flag_CURRENT STIMULUS_hist0", !dbg !7
  br label %"entry.endif_end", !dbg !7
entry.endif_loop.endif:
  br label %"entry.endif_loop", !dbg !7
}

define i1 @"_TransferMechanism_CURRENT_STIMULUS__101_internal_102"({{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* nonnull %".1", {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* nonnull %".2", [1 x [1 x [8 x double]]]* nonnull %".3", {[8 x double]}* nonnull %".4", {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* nonnull %".5") argmemonly!dbg !9
{
entry:
  %"input_ports_out" = alloca [1 x [8 x double]], !dbg !10
  %"ptr_param_input_ports_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".5", i32 0, i32 8 , !dbg !10
  %"ptr_state_input_ports_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 7 , !dbg !10
  %".7" = getelementptr [1 x [1 x [8 x double]]], [1 x [1 x [8 x double]]]* %".3", i32 0, i32 0 , !dbg !10
  %".8" = getelementptr [1 x [8 x double]], [1 x [8 x double]]* %"input_ports_out", i32 0, i32 0 , !dbg !10
  %".9" = getelementptr {{{double, {}, double, {}}, {}, {}}}, {{{double, {}, double, {}}, {}, {}}}* %"ptr_param_input_ports_CURRENT STIMULUS", i32 0, i32 0 , !dbg !10
  %".10" = getelementptr {{{}}}, {{{}}}* %"ptr_state_input_ports_CURRENT STIMULUS", i32 0, i32 0 , !dbg !10
  call void @"_InputPort_InputPort_0__103"({{double, {}, double, {}}, {}, {}}* %".9", {{}}* %".10", [1 x [8 x double]]* %".7", [8 x double]* %".8"), !dbg !10
  %"ptr_state_value_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_CURRENT STIMULUS.1" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_CURRENT STIMULUS_hist1" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2, i32 1 , !dbg !10
  %"ptr_state_value_CURRENT STIMULUS.2" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_CURRENT STIMULUS_hist0" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2, i32 0 , !dbg !10
  %".12" = load [1 x [8 x double]], [1 x [8 x double]]* %"ptr_state_value_CURRENT STIMULUS_hist0", !dbg !10
  store [1 x [8 x double]] %".12", [1 x [8 x double]]* %"ptr_state_value_CURRENT STIMULUS_hist1", !dbg !10
  %"ptr_state_value_CURRENT STIMULUS.3" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_CURRENT STIMULUS_hist0.1" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2, i32 0 , !dbg !10
  %"ptr_state_function_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 8 , !dbg !10
  %"ptr_param_function_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".5", i32 0, i32 10 , !dbg !10
  %"modulated_parameters" = alloca {[1 x double], [1 x double], [1 x double]}, !dbg !10
  %"ptr_param__parameter_ports_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".5", i32 0, i32 0 , !dbg !10
  %"ptr_state__parameter_ports_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 0 , !dbg !10
  %"ptr_param_bias_ReLU Function-0" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %"ptr_param_function_CURRENT STIMULUS", i32 0, i32 0 , !dbg !10
  %"ptr_param_bias_ReLU Function-0.1" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %"modulated_parameters", i32 0, i32 0 , !dbg !10
  %".14" = getelementptr {{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}* %"ptr_param__parameter_ports_CURRENT STIMULUS", i32 0, i32 0 , !dbg !10
  %".15" = getelementptr {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}* %"ptr_state__parameter_ports_CURRENT STIMULUS", i32 0, i32 0 , !dbg !10
  call void @"_ParameterPort_bias__105"({{double, double}}* %".14", {{}}* %".15", [1 x double]* %"ptr_param_bias_ReLU Function-0", [1 x double]* %"ptr_param_bias_ReLU Function-0.1"), !dbg !10
  %"ptr_param_leak_ReLU Function-0" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %"ptr_param_function_CURRENT STIMULUS", i32 0, i32 1 , !dbg !10
  %"ptr_param_leak_ReLU Function-0.1" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %"modulated_parameters", i32 0, i32 1 , !dbg !10
  %".17" = getelementptr {{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}* %"ptr_param__parameter_ports_CURRENT STIMULUS", i32 0, i32 2 , !dbg !10
  %".18" = getelementptr {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}* %"ptr_state__parameter_ports_CURRENT STIMULUS", i32 0, i32 2 , !dbg !10
  call void @"_ParameterPort_leak__107"({{double, double}}* %".17", {{}}* %".18", [1 x double]* %"ptr_param_leak_ReLU Function-0", [1 x double]* %"ptr_param_leak_ReLU Function-0.1"), !dbg !10
  %"ptr_param_gain_ReLU Function-0" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %"ptr_param_function_CURRENT STIMULUS", i32 0, i32 2 , !dbg !10
  %"ptr_param_gain_ReLU Function-0.1" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %"modulated_parameters", i32 0, i32 2 , !dbg !10
  %".20" = getelementptr {{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}* %"ptr_param__parameter_ports_CURRENT STIMULUS", i32 0, i32 1 , !dbg !10
  %".21" = getelementptr {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}* %"ptr_state__parameter_ports_CURRENT STIMULUS", i32 0, i32 1 , !dbg !10
  call void @"_ParameterPort_gain__109"({{double, double}}* %".20", {{}}* %".21", [1 x double]* %"ptr_param_gain_ReLU Function-0", [1 x double]* %"ptr_param_gain_ReLU Function-0.1"), !dbg !10
  call void @"_ReLU_ReLU_Function_0__111"({[1 x double], [1 x double], [1 x double]}* %"modulated_parameters", {}* %"ptr_state_function_CURRENT STIMULUS", [1 x [8 x double]]* %"input_ports_out", [1 x [8 x double]]* %"ptr_state_value_CURRENT STIMULUS_hist0.1"), !dbg !10
  %"ptr_param_clip_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1", i32 0, i32 2 , !dbg !10
  %"ptr_state_num_executions_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 9 , !dbg !10
  %"ptr_state_num_executions_CURRENT STIMULUS_hist0" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 9, i32 0 , !dbg !10
  %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_CURRENT STIMULUS_hist0", i32 0, i32 0 , !dbg !10
  %".24" = load i32, i32* %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr", !dbg !10
  %".25" = add i32 %".24", 1, !dbg !10
  store i32 %".25", i32* %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr", !dbg !10
  %"num_executions_TimeScale.PASS_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_CURRENT STIMULUS_hist0", i32 0, i32 1 , !dbg !10
  %".27" = load i32, i32* %"num_executions_TimeScale.PASS_ptr", !dbg !10
  %".28" = add i32 %".27", 1, !dbg !10
  store i32 %".28", i32* %"num_executions_TimeScale.PASS_ptr", !dbg !10
  %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_CURRENT STIMULUS_hist0", i32 0, i32 2 , !dbg !10
  %".30" = load i32, i32* %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr", !dbg !10
  %".31" = add i32 %".30", 1, !dbg !10
  store i32 %".31", i32* %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr", !dbg !10
  %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_CURRENT STIMULUS_hist0", i32 0, i32 3 , !dbg !10
  %".33" = load i32, i32* %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr", !dbg !10
  %".34" = add i32 %".33", 1, !dbg !10
  store i32 %".34", i32* %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr", !dbg !10
  %"num_executions_TimeScale.LIFE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_CURRENT STIMULUS_hist0", i32 0, i32 4 , !dbg !10
  %".36" = load i32, i32* %"num_executions_TimeScale.LIFE_ptr", !dbg !10
  %".37" = add i32 %".36", 1, !dbg !10
  store i32 %".37", i32* %"num_executions_TimeScale.LIFE_ptr", !dbg !10
  %"ptr_param_output_ports_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".5", i32 0, i32 6 , !dbg !10
  %"ptr_state_output_ports_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 5 , !dbg !10
  %".39" = getelementptr [1 x [8 x double]], [1 x [8 x double]]* %"ptr_state_value_CURRENT STIMULUS_hist0.1", i32 0, i32 0 , !dbg !10
  %".40" = getelementptr {[8 x double]}, {[8 x double]}* %".4", i32 0, i32 0 , !dbg !10
  %".41" = getelementptr {{{double, double}}}, {{{double, double}}}* %"ptr_param_output_ports_CURRENT STIMULUS", i32 0, i32 0 , !dbg !10
  %".42" = getelementptr {{{}}}, {{{}}}* %"ptr_state_output_ports_CURRENT STIMULUS", i32 0, i32 0 , !dbg !10
  call void @"_OutputPort_RESULT__112"({{double, double}}* %".41", {{}}* %".42", [8 x double]* %".39", [8 x double]* %".40"), !dbg !10
  %".44" = call i1 @"_TransferMechanism_CURRENT_STIMULUS__is_finished_114"({{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".5", {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", [1 x [1 x [8 x double]]]* %".3", {[8 x double]}* %".4"), !dbg !10
  ret i1 %".44", !dbg !10
}

declare void @"_InputPort_InputPort_0__103"({{double, {}, double, {}}, {}, {}}* %".1", {{}}* %".2", [1 x [8 x double]]* %".3", [8 x double]* %".4") 

declare void @"_ParameterPort_bias__105"({{double, double}}* %".1", {{}}* %".2", [1 x double]* %".3", [1 x double]* %".4") 

declare void @"_ParameterPort_leak__107"({{double, double}}* %".1", {{}}* %".2", [1 x double]* %".3", [1 x double]* %".4") 

declare void @"_ParameterPort_gain__109"({{double, double}}* %".1", {{}}* %".2", [1 x double]* %".3", [1 x double]* %".4") 

declare void @"_ReLU_ReLU_Function_0__111"({[1 x double], [1 x double], [1 x double]}* %".1", {}* %".2", [1 x [8 x double]]* %".3", [1 x [8 x double]]* %".4") 

declare void @"_OutputPort_RESULT__112"({{double, double}}* %".1", {{}}* %".2", [8 x double]* %".3", [8 x double]* %".4") 

declare i1 @"_TransferMechanism_CURRENT_STIMULUS__is_finished_114"({{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1", {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", [1 x [1 x [8 x double]]]* %".3", {[8 x double]}* %".4") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5, !8 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/mechanisms/processing", filename: "transfermechanism.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_TransferMechanism_CURRENT_STIMULUS__101", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!9 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_TransferMechanism_CURRENT_STIMULUS__101_internal_102", type: !4, unit: !8)
!10 = !DILocation(column: 0, line: 0, scope: !9)