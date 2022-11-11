; ModuleID = "PsyNeuLinkModule-119"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_ProcessingMechanism_DECISION_LAYER__129"({{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* noalias nonnull %".1", {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* noalias nonnull %".2", [1 x [1 x [2 x double]]]* noalias nonnull %".3", {[2 x double]}* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %"ptr_state_is_finished_flag_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 4 , !dbg !7
  %"ptr_state_is_finished_flag_DECISION LAYER_hist0" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 4, i32 0 , !dbg !7
  %"ptr_state_num_executions_before_finished_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 6 , !dbg !7
  %"ptr_state_num_executions_before_finished_DECISION LAYER_hist0" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 6, i32 0 , !dbg !7
  %"ptr_param_max_executions_before_finished_DECISION LAYER" = getelementptr {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}, {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".1", i32 0, i32 5 , !dbg !7
  %".6" = load double, double* %"ptr_state_is_finished_flag_DECISION LAYER_hist0", !dbg !7
  %".7" = fcmp oeq double %".6", 0x3ff0000000000000 , !dbg !7
  br i1 %".7", label %"entry.if", label %"entry.endif", !dbg !7
entry.if:
  store double              0x0, double* %"ptr_state_num_executions_before_finished_DECISION LAYER_hist0", !dbg !7
  br label %"entry.endif", !dbg !7
entry.endif:
  br label %"entry.endif_loop", !dbg !7
entry.endif_loop:
  %".12" = call i1 @"_ProcessingMechanism_DECISION_LAYER__129_internal_130"({{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".1", {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", [1 x [1 x [2 x double]]]* %".3", {[2 x double]}* %".4", {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".1"), !dbg !7
  %".13" = load double, double* %"ptr_state_num_executions_before_finished_DECISION LAYER_hist0", !dbg !7
  %".14" = fadd double %".13", 0x3ff0000000000000, !dbg !7
  store double %".14", double* %"ptr_state_num_executions_before_finished_DECISION LAYER_hist0", !dbg !7
  %".16" = load double, double* %"ptr_param_max_executions_before_finished_DECISION LAYER", !dbg !7
  %".17" = fcmp oge double %".14", %".16" , !dbg !7
  %"ptr_param_execute_until_finished_DECISION LAYER" = getelementptr {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}, {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".1", i32 0, i32 4 , !dbg !7
  %".18" = load double, double* %"ptr_param_execute_until_finished_DECISION LAYER", !dbg !7
  %".19" = fcmp oeq double %".18",              0x0 , !dbg !7
  %".20" = or i1 %".12", %".17", !dbg !7
  %".21" = or i1 %".20", %".19", !dbg !7
  br i1 %".21", label %"entry.endif_loop.if", label %"entry.endif_loop.endif", !dbg !7
entry.endif_end:
  ret void, !dbg !7
entry.endif_loop.if:
  %".23" = uitofp i1 %".20" to double , !dbg !7
  store double %".23", double* %"ptr_state_is_finished_flag_DECISION LAYER_hist0", !dbg !7
  br label %"entry.endif_end", !dbg !7
entry.endif_loop.endif:
  br label %"entry.endif_loop", !dbg !7
}

define i1 @"_ProcessingMechanism_DECISION_LAYER__129_internal_130"({{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* nonnull %".1", {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* nonnull %".2", [1 x [1 x [2 x double]]]* nonnull %".3", {[2 x double]}* nonnull %".4", {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* nonnull %".5") argmemonly!dbg !9
{
entry:
  %"input_ports_out" = alloca [1 x [2 x double]], !dbg !10
  %"ptr_param_input_ports_DECISION LAYER" = getelementptr {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}, {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".5", i32 0, i32 1 , !dbg !10
  %"ptr_state_input_ports_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 1 , !dbg !10
  %".7" = getelementptr [1 x [1 x [2 x double]]], [1 x [1 x [2 x double]]]* %".3", i32 0, i32 0 , !dbg !10
  %".8" = getelementptr [1 x [2 x double]], [1 x [2 x double]]* %"input_ports_out", i32 0, i32 0 , !dbg !10
  %".9" = getelementptr {{{double, {}, double, {}}, {}, {}}}, {{{double, {}, double, {}}, {}, {}}}* %"ptr_param_input_ports_DECISION LAYER", i32 0, i32 0 , !dbg !10
  %".10" = getelementptr {{{}}}, {{{}}}* %"ptr_state_input_ports_DECISION LAYER", i32 0, i32 0 , !dbg !10
  call void @"_InputPort_InputPort_0__131"({{double, {}, double, {}}, {}, {}}* %".9", {{}}* %".10", [1 x [2 x double]]* %".7", [2 x double]* %".8"), !dbg !10
  %"ptr_state_value_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_DECISION LAYER.1" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_DECISION LAYER_hist0" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 2, i32 0 , !dbg !10
  %"ptr_param_function_DECISION LAYER" = getelementptr {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}, {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".5", i32 0, i32 3 , !dbg !10
  %"modulated_parameters" = alloca {double, {double}, [1 x double]}, !dbg !10
  %".12" = bitcast {double, {double}, [1 x double]}* %"ptr_param_function_DECISION LAYER" to i8* , !dbg !10
  %".13" = bitcast {double, {double}, [1 x double]}* %"modulated_parameters" to i8* , !dbg !10
  %".14" = call i64 @"llvm.objectsize.i64"(i8* %".13", i1 1, i1 0, i1 0), !dbg !10
  %".15" = add i64 %".14", 3, !dbg !10
  %".16" = udiv i64 %".15", 4, !dbg !10
  %".17" = bitcast {double, {double}, [1 x double]}* %"ptr_param_function_DECISION LAYER" to i32* , !dbg !10
  %".18" = bitcast {double, {double}, [1 x double]}* %"modulated_parameters" to i32* , !dbg !10
  %"memcopy_loop_index_var_loc" = alloca i64, !dbg !10
  store i64 0, i64* %"memcopy_loop_index_var_loc", !dbg !10
  br label %"memcopy_loop-cond-bb", !dbg !10
memcopy_loop-cond-bb:
  %"memcopy_loop_cond_index_var" = load i64, i64* %"memcopy_loop_index_var_loc", !dbg !10
  %"memcopy_loop_loop_cond" = icmp slt i64 %"memcopy_loop_cond_index_var", %".16" , !dbg !10
  br i1 %"memcopy_loop_loop_cond", label %"memcopy_loop-cond-bb.if", label %"memcopy_loop-cond-bb.endif", !dbg !10, !prof !11
memcopy_loop-cond-bb.if:
  %"memcopy_loop_loop_index_var" = load i64, i64* %"memcopy_loop_index_var_loc", !dbg !10
  %".22" = getelementptr i32, i32* %".17", i64 %"memcopy_loop_loop_index_var" , !dbg !10
  %".23" = getelementptr i32, i32* %".18", i64 %"memcopy_loop_loop_index_var" , !dbg !10
  %".24" = load i32, i32* %".22", !dbg !10
  store i32 %".24", i32* %".23", !dbg !10
  %"memcopy_loop_index_var_inc" = add i64 %"memcopy_loop_loop_index_var", 1, !dbg !10
  store i64 %"memcopy_loop_index_var_inc", i64* %"memcopy_loop_index_var_loc", !dbg !10
  br label %"memcopy_loop-cond-bb", !dbg !10
memcopy_loop-cond-bb.endif:
  %"ptr_param__parameter_ports_DECISION LAYER" = getelementptr {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}, {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".5", i32 0, i32 0 , !dbg !10
  %"ptr_state__parameter_ports_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 0 , !dbg !10
  %"ptr_param_gain_SoftMax Function-1" = getelementptr {double, {double}, [1 x double]}, {double, {double}, [1 x double]}* %"ptr_param_function_DECISION LAYER", i32 0, i32 2 , !dbg !10
  %"ptr_param_gain_SoftMax Function-1.1" = getelementptr {double, {double}, [1 x double]}, {double, {double}, [1 x double]}* %"modulated_parameters", i32 0, i32 2 , !dbg !10
  %".28" = getelementptr {{{double, double}}, {{double, double}}}, {{{double, double}}, {{double, double}}}* %"ptr_param__parameter_ports_DECISION LAYER", i32 0, i32 0 , !dbg !10
  %".29" = getelementptr {{{}}, {{}}}, {{{}}, {{}}}* %"ptr_state__parameter_ports_DECISION LAYER", i32 0, i32 0 , !dbg !10
  call void @"_ParameterPort_gain__133"({{double, double}}* %".28", {{}}* %".29", [1 x double]* %"ptr_param_gain_SoftMax Function-1", [1 x double]* %"ptr_param_gain_SoftMax Function-1.1"), !dbg !10
  %"ptr_state_function_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 3 , !dbg !10
  call void @"_SoftMax_SoftMax_Function_1__135"({double, {double}, [1 x double]}* %"modulated_parameters", {{[1 x {[624 x i32], i32, i32, double, i32}]}}* %"ptr_state_function_DECISION LAYER", [1 x [2 x double]]* %"input_ports_out", [1 x [2 x double]]* %"ptr_state_value_DECISION LAYER_hist0"), !dbg !10
  %"ptr_state_num_executions_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 5 , !dbg !10
  %"ptr_state_num_executions_DECISION LAYER_hist0" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 5, i32 0 , !dbg !10
  %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_DECISION LAYER_hist0", i32 0, i32 0 , !dbg !10
  %".32" = load i32, i32* %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr", !dbg !10
  %".33" = add i32 %".32", 1, !dbg !10
  store i32 %".33", i32* %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr", !dbg !10
  %"num_executions_TimeScale.PASS_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_DECISION LAYER_hist0", i32 0, i32 1 , !dbg !10
  %".35" = load i32, i32* %"num_executions_TimeScale.PASS_ptr", !dbg !10
  %".36" = add i32 %".35", 1, !dbg !10
  store i32 %".36", i32* %"num_executions_TimeScale.PASS_ptr", !dbg !10
  %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_DECISION LAYER_hist0", i32 0, i32 2 , !dbg !10
  %".38" = load i32, i32* %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr", !dbg !10
  %".39" = add i32 %".38", 1, !dbg !10
  store i32 %".39", i32* %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr", !dbg !10
  %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_DECISION LAYER_hist0", i32 0, i32 3 , !dbg !10
  %".41" = load i32, i32* %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr", !dbg !10
  %".42" = add i32 %".41", 1, !dbg !10
  store i32 %".42", i32* %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr", !dbg !10
  %"num_executions_TimeScale.LIFE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_DECISION LAYER_hist0", i32 0, i32 4 , !dbg !10
  %".44" = load i32, i32* %"num_executions_TimeScale.LIFE_ptr", !dbg !10
  %".45" = add i32 %".44", 1, !dbg !10
  store i32 %".45", i32* %"num_executions_TimeScale.LIFE_ptr", !dbg !10
  %"ptr_param_output_ports_DECISION LAYER" = getelementptr {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}, {{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".5", i32 0, i32 6 , !dbg !10
  %"ptr_state_output_ports_DECISION LAYER" = getelementptr {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 7 , !dbg !10
  %".47" = getelementptr [1 x [2 x double]], [1 x [2 x double]]* %"ptr_state_value_DECISION LAYER_hist0", i32 0, i32 0 , !dbg !10
  %".48" = getelementptr {[2 x double]}, {[2 x double]}* %".4", i32 0, i32 0 , !dbg !10
  %".49" = getelementptr {{{double, double}}}, {{{double, double}}}* %"ptr_param_output_ports_DECISION LAYER", i32 0, i32 0 , !dbg !10
  %".50" = getelementptr {{{}}}, {{{}}}* %"ptr_state_output_ports_DECISION LAYER", i32 0, i32 0 , !dbg !10
  call void @"_OutputPort_OutputPort_0__137"({{double, double}}* %".49", {{}}* %".50", [2 x double]* %".47", [2 x double]* %".48"), !dbg !10
  %".52" = call i1 @"_ProcessingMechanism_DECISION_LAYER__is_finished_139"({{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".5", {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", [1 x [1 x [2 x double]]]* %".3", {[2 x double]}* %".4"), !dbg !10
  ret i1 %".52", !dbg !10
}

declare void @"_InputPort_InputPort_0__131"({{double, {}, double, {}}, {}, {}}* %".1", {{}}* %".2", [1 x [2 x double]]* %".3", [2 x double]* %".4") 

declare i64 @"llvm.objectsize.i64"(i8* %".1", i1 %".2", i1 %".3", i1 %".4") 

declare void @"_ParameterPort_gain__133"({{double, double}}* %".1", {{}}* %".2", [1 x double]* %".3", [1 x double]* %".4") 

declare void @"_SoftMax_SoftMax_Function_1__135"({double, {double}, [1 x double]}* %".1", {{[1 x {[624 x i32], i32, i32, double, i32}]}}* %".2", [1 x [2 x double]]* %".3", [1 x [2 x double]]* %".4") 

declare void @"_OutputPort_OutputPort_0__137"({{double, double}}* %".1", {{}}* %".2", [2 x double]* %".3", [2 x double]* %".4") 

declare i1 @"_ProcessingMechanism_DECISION_LAYER__is_finished_139"({{{{double, double}}, {{double, double}}}, {{{double, {}, double, {}}, {}, {}}}, double, {double, {double}, [1 x double]}, double, double, {{{double, double}}}}* %".1", {{{{}}, {{}}}, {{{}}}, [1 x [1 x [2 x double]]], {{[1 x {[624 x i32], i32, i32, double, i32}]}}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", [1 x [1 x [2 x double]]]* %".3", {[2 x double]}* %".4") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5, !8 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/mechanisms/processing", filename: "processingmechanism.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_ProcessingMechanism_DECISION_LAYER__129", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!9 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_ProcessingMechanism_DECISION_LAYER__129_internal_130", type: !4, unit: !8)
!10 = !DILocation(column: 0, line: 0, scope: !9)
!11 = !{ !"branch_weights", i32 99, i32 1 }