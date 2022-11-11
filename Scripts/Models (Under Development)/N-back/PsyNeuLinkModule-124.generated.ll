; ModuleID = "PsyNeuLinkModule-124"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_SoftMax_SoftMax_Function_1__135"({double, {double}, [1 x double]}* noalias nonnull %".1", {{[1 x {[624 x i32], i32, i32, double, i32}]}}* noalias nonnull %".2", [1 x [2 x double]]* noalias nonnull %".3", [1 x [2 x double]]* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %".6" = getelementptr [1 x [2 x double]], [1 x [2 x double]]* %".3", i32 0, i32 0 , !dbg !7
  %".7" = getelementptr [1 x [2 x double]], [1 x [2 x double]]* %".4", i32 0, i32 0 , !dbg !7
  %".8" = alloca double, !dbg !7
  store double              0x0, double* %".8", !dbg !7
  %"ptr_param_gain_SoftMax Function-1" = getelementptr {double, {double}, [1 x double]}, {double, {double}, [1 x double]}* %".1", i32 0, i32 2 , !dbg !7
  %".10" = load [1 x double], [1 x double]* %"ptr_param_gain_SoftMax Function-1", !dbg !7
  %".11" = extractvalue [1 x double] %".10", 0 , !dbg !7
  %"exp_sum_max_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"exp_sum_max_index_var_loc", !dbg !7
  br label %"exp_sum_max-cond-bb", !dbg !7
exp_sum_max-cond-bb:
  %"exp_sum_max_cond_index_var" = load i32, i32* %"exp_sum_max_index_var_loc", !dbg !7
  %"exp_sum_max_loop_cond" = icmp slt i32 %"exp_sum_max_cond_index_var", 2 , !dbg !7
  br i1 %"exp_sum_max_loop_cond", label %"exp_sum_max-cond-bb.if", label %"exp_sum_max-cond-bb.endif", !dbg !7, !prof !8
exp_sum_max-cond-bb.if:
  %"exp_sum_max_loop_index_var" = load i32, i32* %"exp_sum_max_index_var_loc", !dbg !7
  %".15" = getelementptr [2 x double], [2 x double]* %".6", i32 0, i32 %"exp_sum_max_loop_index_var" , !dbg !7
  %".16" = load double, double* %".15", !dbg !7
  %".17" = fmul double %".16", %".11", !dbg !7
  %".18" = call double @"__pnl_builtin_exp"(double %".17"), !dbg !7
  %".19" = load double, double* %".8", !dbg !7
  %".20" = fadd double %".19", %".18", !dbg !7
  store double %".20", double* %".8", !dbg !7
  %"exp_sum_max_index_var_inc" = add i32 %"exp_sum_max_loop_index_var", 1, !dbg !7
  store i32 %"exp_sum_max_index_var_inc", i32* %"exp_sum_max_index_var_loc", !dbg !7
  br label %"exp_sum_max-cond-bb", !dbg !7
exp_sum_max-cond-bb.endif:
  %".24" = load double, double* %".8", !dbg !7
  %"ptr_param_one_hot_function_SoftMax Function-1" = getelementptr {double, {double}, [1 x double]}, {double, {double}, [1 x double]}* %".1", i32 0, i32 1 , !dbg !7
  %"ptr_state_one_hot_function_SoftMax Function-1" = getelementptr {{[1 x {[624 x i32], i32, i32, double, i32}]}}, {{[1 x {[624 x i32], i32, i32, double, i32}]}}* %".2", i32 0, i32 0 , !dbg !7
  %".25" = alloca [2 x double], !dbg !7
  %"exp_div_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"exp_div_index_var_loc", !dbg !7
  br label %"exp_div-cond-bb", !dbg !7
exp_div-cond-bb:
  %"exp_div_cond_index_var" = load i32, i32* %"exp_div_index_var_loc", !dbg !7
  %"exp_div_loop_cond" = icmp slt i32 %"exp_div_cond_index_var", 2 , !dbg !7
  br i1 %"exp_div_loop_cond", label %"exp_div-cond-bb.if", label %"exp_div-cond-bb.endif", !dbg !7, !prof !8
exp_div-cond-bb.if:
  %"exp_div_loop_index_var" = load i32, i32* %"exp_div_index_var_loc", !dbg !7
  %".29" = getelementptr [2 x double], [2 x double]* %".25", i32 0, i32 %"exp_div_loop_index_var" , !dbg !7
  %".30" = getelementptr [2 x double], [2 x double]* %".6", i32 0, i32 %"exp_div_loop_index_var" , !dbg !7
  %".31" = load double, double* %".30", !dbg !7
  %".32" = fmul double %".31", %".11", !dbg !7
  %".33" = call double @"__pnl_builtin_exp"(double %".32"), !dbg !7
  %".34" = fdiv double %".33", %".24", !dbg !7
  store double %".34", double* %".29", !dbg !7
  %"exp_div_index_var_inc" = add i32 %"exp_div_loop_index_var", 1, !dbg !7
  store i32 %"exp_div_index_var_inc", i32* %"exp_div_index_var_loc", !dbg !7
  br label %"exp_div-cond-bb", !dbg !7
exp_div-cond-bb.endif:
  call void @"_OneHot_OneHot_Function_10__136"({double}* %"ptr_param_one_hot_function_SoftMax Function-1", {[1 x {[624 x i32], i32, i32, double, i32}]}* %"ptr_state_one_hot_function_SoftMax Function-1", [2 x double]* %".25", [2 x double]* %".7"), !dbg !7
  ret void, !dbg !7
}

declare double @"__pnl_builtin_exp"(double %".1") 

declare void @"_OneHot_OneHot_Function_10__136"({double}* %".1", {[1 x {[624 x i32], i32, i32, double, i32}]}* %".2", [2 x double]* %".3", [2 x double]* %".4") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/functions/nonstateful", filename: "transferfunctions.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_SoftMax_SoftMax_Function_1__135", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = !{ !"branch_weights", i32 99, i32 1 }