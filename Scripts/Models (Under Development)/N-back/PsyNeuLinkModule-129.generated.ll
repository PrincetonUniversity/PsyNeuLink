; ModuleID = "PsyNeuLinkModule-129"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define double @"LEARNING_MSE_CALL_140"(double* nonnull %".1", i32 %".2", double* nonnull %".3") argmemonly!dbg !6
{
entry:
  %".5" = alloca double, !dbg !7
  store double 0x8000000000000000, double* %".5", !dbg !7
  %"mse_sum_loop_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"mse_sum_loop_index_var_loc", !dbg !7
  br label %"mse_sum_loop-cond-bb", !dbg !7
mse_sum_loop-cond-bb:
  %"mse_sum_loop_cond_index_var" = load i32, i32* %"mse_sum_loop_index_var_loc", !dbg !7
  %"mse_sum_loop_loop_cond" = icmp slt i32 %"mse_sum_loop_cond_index_var", %".2" , !dbg !7
  br i1 %"mse_sum_loop_loop_cond", label %"mse_sum_loop-cond-bb.if", label %"mse_sum_loop-cond-bb.endif", !dbg !7, !prof !8
mse_sum_loop-cond-bb.if:
  %"mse_sum_loop_loop_index_var" = load i32, i32* %"mse_sum_loop_index_var_loc", !dbg !7
  %".10" = getelementptr double, double* %".1", i32 %"mse_sum_loop_loop_index_var" , !dbg !7
  %".11" = getelementptr double, double* %".3", i32 %"mse_sum_loop_loop_index_var" , !dbg !7
  %".12" = load double, double* %".10", !dbg !7
  %".13" = load double, double* %".11", !dbg !7
  %".14" = fsub double %".12", %".13", !dbg !7
  %".15" = fmul double %".14", %".14", !dbg !7
  %".16" = load double, double* %".5", !dbg !7
  %".17" = fadd double %".16", %".15", !dbg !7
  store double %".17", double* %".5", !dbg !7
  %"mse_sum_loop_index_var_inc" = add i32 %"mse_sum_loop_loop_index_var", 1, !dbg !7
  store i32 %"mse_sum_loop_index_var_inc", i32* %"mse_sum_loop_index_var_loc", !dbg !7
  br label %"mse_sum_loop-cond-bb", !dbg !7
mse_sum_loop-cond-bb.endif:
  %".21" = load double, double* %".5", !dbg !7
  %".22" = uitofp i32 %".2" to double , !dbg !7
  %".23" = fdiv double %".21", %".22", !dbg !7
  store double %".23", double* %".5", !dbg !7
  %".25" = load double, double* %".5", !dbg !7
  ret double %".25", !dbg !7
}

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/library/compositions", filename: "compiledloss.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "LEARNING_MSE_CALL_140", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = !{ !"branch_weights", i32 99, i32 1 }