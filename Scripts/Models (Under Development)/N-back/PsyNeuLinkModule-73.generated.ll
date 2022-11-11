; ModuleID = "PsyNeuLinkModule-73"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_Linear_Linear_Function_41__80"({double, double}* noalias nonnull %".1", {}* noalias nonnull %".2", [1 x double]* noalias nonnull %".3", [1 x double]* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %"transfer_loop_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"transfer_loop_index_var_loc", !dbg !7
  br label %"transfer_loop-cond-bb", !dbg !7
transfer_loop-cond-bb:
  %"transfer_loop_cond_index_var" = load i32, i32* %"transfer_loop_index_var_loc", !dbg !7
  %"transfer_loop_loop_cond" = icmp slt i32 %"transfer_loop_cond_index_var", 1 , !dbg !7
  br i1 %"transfer_loop_loop_cond", label %"transfer_loop-cond-bb.if", label %"transfer_loop-cond-bb.endif", !dbg !7, !prof !8
transfer_loop-cond-bb.if:
  %"transfer_loop_loop_index_var" = load i32, i32* %"transfer_loop_index_var_loc", !dbg !7
  %".9" = getelementptr [1 x double], [1 x double]* %".3", i32 0, i32 %"transfer_loop_loop_index_var" , !dbg !7
  %".10" = getelementptr [1 x double], [1 x double]* %".4", i32 0, i32 %"transfer_loop_loop_index_var" , !dbg !7
  %"ptr_param_slope_Linear Function-41" = getelementptr {double, double}, {double, double}* %".1", i32 0, i32 0 , !dbg !7
  %"ptr_param_intercept_Linear Function-41" = getelementptr {double, double}, {double, double}* %".1", i32 0, i32 1 , !dbg !7
  %".11" = load double, double* %"ptr_param_slope_Linear Function-41", !dbg !7
  %".12" = load double, double* %"ptr_param_intercept_Linear Function-41", !dbg !7
  %".13" = load double, double* %".9", !dbg !7
  %".14" = fmul double %".13", %".11", !dbg !7
  %".15" = fadd double %".14", %".12", !dbg !7
  store double %".15", double* %".10", !dbg !7
  %"transfer_loop_index_var_inc" = add i32 %"transfer_loop_loop_index_var", 1, !dbg !7
  store i32 %"transfer_loop_index_var_inc", i32* %"transfer_loop_index_var_loc", !dbg !7
  br label %"transfer_loop-cond-bb", !dbg !7
transfer_loop-cond-bb.endif:
  ret void, !dbg !7
}

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/functions/nonstateful", filename: "transferfunctions.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_Linear_Linear_Function_41__80", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = !{ !"branch_weights", i32 99, i32 1 }