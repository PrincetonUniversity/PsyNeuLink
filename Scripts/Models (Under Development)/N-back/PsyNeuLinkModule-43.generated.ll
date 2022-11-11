; ModuleID = "PsyNeuLinkModule-43"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_LinearCombination_LinearCombination_Function_10__48"({double, {}, double, {}}* noalias nonnull %".1", {}* noalias nonnull %".2", [1 x [25 x double]]* noalias nonnull %".3", [25 x double]* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %"linear_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"linear_index_var_loc", !dbg !7
  br label %"linear-cond-bb", !dbg !7
linear-cond-bb:
  %"linear_cond_index_var" = load i32, i32* %"linear_index_var_loc", !dbg !7
  %"linear_loop_cond" = icmp slt i32 %"linear_cond_index_var", 25 , !dbg !7
  br i1 %"linear_loop_cond", label %"linear-cond-bb.if", label %"linear-cond-bb.endif", !dbg !7, !prof !8
linear-cond-bb.if:
  %"linear_loop_index_var" = load i32, i32* %"linear_index_var_loc", !dbg !7
  %"ptr_param_scale_LinearCombination Function-10" = getelementptr {double, {}, double, {}}, {double, {}, double, {}}* %".1", i32 0, i32 2 , !dbg !7
  %".9" = load double, double* %"ptr_param_scale_LinearCombination Function-10", !dbg !7
  %"ptr_param_offset_LinearCombination Function-10" = getelementptr {double, {}, double, {}}, {double, {}, double, {}}* %".1", i32 0, i32 0 , !dbg !7
  %".10" = load double, double* %"ptr_param_offset_LinearCombination Function-10", !dbg !7
  %"combined_result" = alloca double, !dbg !7
  store double 0x8000000000000000, double* %"combined_result", !dbg !7
  %"combine_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"combine_index_var_loc", !dbg !7
  br label %"combine-cond-bb", !dbg !7
linear-cond-bb.endif:
  ret void, !dbg !7
combine-cond-bb:
  %"combine_cond_index_var" = load i32, i32* %"combine_index_var_loc", !dbg !7
  %"combine_loop_cond" = icmp slt i32 %"combine_cond_index_var", 1 , !dbg !7
  br i1 %"combine_loop_cond", label %"combine-cond-bb.if", label %"combine-cond-bb.endif", !dbg !7, !prof !8
combine-cond-bb.if:
  %"combine_loop_index_var" = load i32, i32* %"combine_index_var_loc", !dbg !7
  %".15" = getelementptr [1 x [25 x double]], [1 x [25 x double]]* %".3", i32 0, i32 %"combine_loop_index_var", i32 %"linear_loop_index_var" , !dbg !7
  %".16" = load double, double* %".15", !dbg !7
  %"ptr_param_exponents_LinearCombination Function-10" = getelementptr {double, {}, double, {}}, {double, {}, double, {}}* %".1", i32 0, i32 3 , !dbg !7
  %"ptr_param_weights_LinearCombination Function-10" = getelementptr {double, {}, double, {}}, {double, {}, double, {}}* %".1", i32 0, i32 1 , !dbg !7
  %".17" = fmul double %".16", 0x3ff0000000000000, !dbg !7
  %".18" = load double, double* %"combined_result", !dbg !7
  %".19" = fadd double %".18", %".17", !dbg !7
  store double %".19", double* %"combined_result", !dbg !7
  %"combine_index_var_inc" = add i32 %"combine_loop_index_var", 1, !dbg !7
  store i32 %"combine_index_var_inc", i32* %"combine_index_var_loc", !dbg !7
  br label %"combine-cond-bb", !dbg !7
combine-cond-bb.endif:
  %".23" = load double, double* %"combined_result", !dbg !7
  %".24" = fmul double %".23", %".9", !dbg !7
  %".25" = fadd double %".24", %".10", !dbg !7
  %".26" = getelementptr [25 x double], [25 x double]* %".4", i32 0, i32 %"linear_loop_index_var" , !dbg !7
  store double %".25", double* %".26", !dbg !7
  %"linear_index_var_inc" = add i32 %"linear_loop_index_var", 1, !dbg !7
  store i32 %"linear_index_var_inc", i32* %"linear_index_var_loc", !dbg !7
  br label %"linear-cond-bb", !dbg !7
}

declare double @"__pnl_builtin_pow"(double %".1", double %".2") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/functions/nonstateful", filename: "combinationfunctions.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_LinearCombination_LinearCombination_Function_10__48", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = !{ !"branch_weights", i32 99, i32 1 }