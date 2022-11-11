; ModuleID = "PsyNeuLinkModule-133"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Output_CIM__learning_143"({{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* noalias nonnull %".1", {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* noalias nonnull %".2", [1 x [1 x [2 x double]]]* noalias nonnull %".3", {[2 x double]}* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %"ptr_state_is_finished_flag_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 4 , !dbg !7
  %"ptr_state_is_finished_flag_WORKING MEMORY (fnn) Output_CIM_hist0" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 4, i32 0 , !dbg !7
  %"ptr_state_num_executions_before_finished_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 6 , !dbg !7
  %"ptr_state_num_executions_before_finished_WORKING MEMORY (fnn) Output_CIM_hist0" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 6, i32 0 , !dbg !7
  %"ptr_param_max_executions_before_finished_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}, {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".1", i32 0, i32 5 , !dbg !7
  %".6" = load double, double* %"ptr_state_is_finished_flag_WORKING MEMORY (fnn) Output_CIM_hist0", !dbg !7
  %".7" = fcmp oeq double %".6", 0x3ff0000000000000 , !dbg !7
  br i1 %".7", label %"entry.if", label %"entry.endif", !dbg !7
entry.if:
  store double              0x0, double* %"ptr_state_num_executions_before_finished_WORKING MEMORY (fnn) Output_CIM_hist0", !dbg !7
  br label %"entry.endif", !dbg !7
entry.endif:
  br label %"entry.endif_loop", !dbg !7
entry.endif_loop:
  %".12" = call i1 @"_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Output_CIM__learning_143_internal_144"({{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".1", {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", [1 x [1 x [2 x double]]]* %".3", {[2 x double]}* %".4", {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".1"), !dbg !7
  %".13" = load double, double* %"ptr_state_num_executions_before_finished_WORKING MEMORY (fnn) Output_CIM_hist0", !dbg !7
  %".14" = fadd double %".13", 0x3ff0000000000000, !dbg !7
  store double %".14", double* %"ptr_state_num_executions_before_finished_WORKING MEMORY (fnn) Output_CIM_hist0", !dbg !7
  %".16" = load double, double* %"ptr_param_max_executions_before_finished_WORKING MEMORY (fnn) Output_CIM", !dbg !7
  %".17" = fcmp oge double %".14", %".16" , !dbg !7
  %"ptr_param_execute_until_finished_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}, {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".1", i32 0, i32 4 , !dbg !7
  %".18" = load double, double* %"ptr_param_execute_until_finished_WORKING MEMORY (fnn) Output_CIM", !dbg !7
  %".19" = fcmp oeq double %".18",              0x0 , !dbg !7
  %".20" = or i1 %".12", %".17", !dbg !7
  %".21" = or i1 %".20", %".19", !dbg !7
  br i1 %".21", label %"entry.endif_loop.if", label %"entry.endif_loop.endif", !dbg !7
entry.endif_end:
  ret void, !dbg !7
entry.endif_loop.if:
  %".23" = uitofp i1 %".20" to double , !dbg !7
  store double %".23", double* %"ptr_state_is_finished_flag_WORKING MEMORY (fnn) Output_CIM_hist0", !dbg !7
  br label %"entry.endif_end", !dbg !7
entry.endif_loop.endif:
  br label %"entry.endif_loop", !dbg !7
}

define i1 @"_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Output_CIM__learning_143_internal_144"({{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* nonnull %".1", {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* nonnull %".2", [1 x [1 x [2 x double]]]* nonnull %".3", {[2 x double]}* nonnull %".4", {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* nonnull %".5") argmemonly!dbg !9
{
entry:
  %"input_ports_out" = alloca [1 x [2 x double]], !dbg !10
  %"ptr_param_input_ports_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}, {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".5", i32 0, i32 1 , !dbg !10
  %"ptr_state_input_ports_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 1 , !dbg !10
  %".7" = getelementptr [1 x [1 x [2 x double]]], [1 x [1 x [2 x double]]]* %".3", i32 0, i32 0 , !dbg !10
  %".8" = getelementptr [1 x [2 x double]], [1 x [2 x double]]* %"input_ports_out", i32 0, i32 0 , !dbg !10
  %".9" = getelementptr {{{double, {}, double, {}}, {}, {}}}, {{{double, {}, double, {}}, {}, {}}}* %"ptr_param_input_ports_WORKING MEMORY (fnn) Output_CIM", i32 0, i32 0 , !dbg !10
  %".10" = getelementptr {{{}}}, {{{}}}* %"ptr_state_input_ports_WORKING MEMORY (fnn) Output_CIM", i32 0, i32 0 , !dbg !10
  call void @"_InputPort_OUTPUT_CIM_DECISION_LAYER_OutputPort_0__145"({{double, {}, double, {}}, {}, {}}* %".9", {{}}* %".10", [1 x [2 x double]]* %".7", [2 x double]* %".8"), !dbg !10
  %"ptr_state_value_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_WORKING MEMORY (fnn) Output_CIM.1" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 2 , !dbg !10
  %"ptr_state_value_WORKING MEMORY (fnn) Output_CIM_hist0" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 2, i32 0 , !dbg !10
  %"ptr_param_function_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}, {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".5", i32 0, i32 3 , !dbg !10
  %"ptr_state_function_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 3 , !dbg !10
  call void @"_Identity_Identity_Function_2__learning_147"({}* %"ptr_param_function_WORKING MEMORY (fnn) Output_CIM", {}* %"ptr_state_function_WORKING MEMORY (fnn) Output_CIM", [1 x [2 x double]]* %"input_ports_out", [1 x [2 x double]]* %"ptr_state_value_WORKING MEMORY (fnn) Output_CIM_hist0"), !dbg !10
  %"ptr_state_num_executions_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 5 , !dbg !10
  %"ptr_state_num_executions_WORKING MEMORY (fnn) Output_CIM_hist0" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 5, i32 0 , !dbg !10
  %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_WORKING MEMORY (fnn) Output_CIM_hist0", i32 0, i32 0 , !dbg !10
  %".13" = load i32, i32* %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr", !dbg !10
  %".14" = add i32 %".13", 1, !dbg !10
  store i32 %".14", i32* %"num_executions_TimeScale.CONSIDERATION_SET_EXECUTION_ptr", !dbg !10
  %"num_executions_TimeScale.PASS_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_WORKING MEMORY (fnn) Output_CIM_hist0", i32 0, i32 1 , !dbg !10
  %".16" = load i32, i32* %"num_executions_TimeScale.PASS_ptr", !dbg !10
  %".17" = add i32 %".16", 1, !dbg !10
  store i32 %".17", i32* %"num_executions_TimeScale.PASS_ptr", !dbg !10
  %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_WORKING MEMORY (fnn) Output_CIM_hist0", i32 0, i32 2 , !dbg !10
  %".19" = load i32, i32* %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr", !dbg !10
  %".20" = add i32 %".19", 1, !dbg !10
  store i32 %".20", i32* %"num_executions_TimeScale.ENVIRONMENT_STATE_UPDATE_ptr", !dbg !10
  %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_WORKING MEMORY (fnn) Output_CIM_hist0", i32 0, i32 3 , !dbg !10
  %".22" = load i32, i32* %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr", !dbg !10
  %".23" = add i32 %".22", 1, !dbg !10
  store i32 %".23", i32* %"num_executions_TimeScale.ENVIRONMENT_SEQUENCE_ptr", !dbg !10
  %"num_executions_TimeScale.LIFE_ptr" = getelementptr [5 x i32], [5 x i32]* %"ptr_state_num_executions_WORKING MEMORY (fnn) Output_CIM_hist0", i32 0, i32 4 , !dbg !10
  %".25" = load i32, i32* %"num_executions_TimeScale.LIFE_ptr", !dbg !10
  %".26" = add i32 %".25", 1, !dbg !10
  store i32 %".26", i32* %"num_executions_TimeScale.LIFE_ptr", !dbg !10
  %"ptr_param_output_ports_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}, {{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".5", i32 0, i32 6 , !dbg !10
  %"ptr_state_output_ports_WORKING MEMORY (fnn) Output_CIM" = getelementptr {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}, {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", i32 0, i32 7 , !dbg !10
  %".28" = getelementptr [1 x [2 x double]], [1 x [2 x double]]* %"ptr_state_value_WORKING MEMORY (fnn) Output_CIM_hist0", i32 0, i32 0 , !dbg !10
  %".29" = getelementptr {[2 x double]}, {[2 x double]}* %".4", i32 0, i32 0 , !dbg !10
  %".30" = getelementptr {{{}}}, {{{}}}* %"ptr_param_output_ports_WORKING MEMORY (fnn) Output_CIM", i32 0, i32 0 , !dbg !10
  %".31" = getelementptr {{{}}}, {{{}}}* %"ptr_state_output_ports_WORKING MEMORY (fnn) Output_CIM", i32 0, i32 0 , !dbg !10
  call void @"_OutputPort_OUTPUT_CIM_DECISION_LAYER_OutputPort_0__148"({{}}* %".30", {{}}* %".31", [2 x double]* %".28", [2 x double]* %".29"), !dbg !10
  %".33" = call i1 @"_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Output_CIM__is_finished_learning_150"({{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".5", {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", [1 x [1 x [2 x double]]]* %".3", {[2 x double]}* %".4"), !dbg !10
  ret i1 %".33", !dbg !10
}

declare void @"_InputPort_OUTPUT_CIM_DECISION_LAYER_OutputPort_0__145"({{double, {}, double, {}}, {}, {}}* %".1", {{}}* %".2", [1 x [2 x double]]* %".3", [2 x double]* %".4") 

declare void @"_Identity_Identity_Function_2__learning_147"({}* %".1", {}* %".2", [1 x [2 x double]]* %".3", [1 x [2 x double]]* %".4") 

declare void @"_OutputPort_OUTPUT_CIM_DECISION_LAYER_OutputPort_0__148"({{}}* %".1", {{}}* %".2", [2 x double]* %".3", [2 x double]* %".4") 

declare i1 @"_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Output_CIM__is_finished_learning_150"({{}, {{{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}}}* %".1", {{}, {{{}}}, [1 x [1 x [2 x double]]], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}}}* %".2", [1 x [1 x [2 x double]]]* %".3", {[2 x double]}* %".4") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5, !8 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/mechanisms/processing", filename: "compositioninterfacemechanism.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Output_CIM__learning_143", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!9 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Output_CIM__learning_143_internal_144", type: !4, unit: !8)
!10 = !DILocation(column: 0, line: 0, scope: !9)