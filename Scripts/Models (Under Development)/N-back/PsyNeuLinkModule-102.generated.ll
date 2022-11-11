; ModuleID = "PsyNeuLinkModule-102"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_ReLU_ReLU_Function_0__111"({[1 x double], [1 x double], [1 x double]}* noalias nonnull %".1", {}* noalias nonnull %".2", [1 x [8 x double]]* noalias nonnull %".3", [1 x [8 x double]]* noalias nonnull %".4") argmemonly!dbg !6
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
  %".9" = getelementptr [1 x [8 x double]], [1 x [8 x double]]* %".3", i32 0, i32 %"transfer_loop_loop_index_var" , !dbg !7
  %".10" = getelementptr [1 x [8 x double]], [1 x [8 x double]]* %".4", i32 0, i32 %"transfer_loop_loop_index_var" , !dbg !7
  %"nested_transfer_loop_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"nested_transfer_loop_index_var_loc", !dbg !7
  br label %"nested_transfer_loop-cond-bb", !dbg !7
transfer_loop-cond-bb.endif:
  ret void, !dbg !7
nested_transfer_loop-cond-bb:
  %"nested_transfer_loop_cond_index_var" = load i32, i32* %"nested_transfer_loop_index_var_loc", !dbg !7
  %"nested_transfer_loop_loop_cond" = icmp slt i32 %"nested_transfer_loop_cond_index_var", 8 , !dbg !7
  br i1 %"nested_transfer_loop_loop_cond", label %"nested_transfer_loop-cond-bb.if", label %"nested_transfer_loop-cond-bb.endif", !dbg !7, !prof !8
nested_transfer_loop-cond-bb.if:
  %"nested_transfer_loop_loop_index_var" = load i32, i32* %"nested_transfer_loop_index_var_loc", !dbg !7
  %".14" = getelementptr [8 x double], [8 x double]* %".9", i32 0, i32 %"nested_transfer_loop_loop_index_var" , !dbg !7
  %".15" = getelementptr [8 x double], [8 x double]* %".10", i32 0, i32 %"nested_transfer_loop_loop_index_var" , !dbg !7
  %"ptr_param_gain_ReLU Function-0" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %".1", i32 0, i32 2 , !dbg !7
  %"ptr_param_bias_ReLU Function-0" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %".1", i32 0, i32 0 , !dbg !7
  %"ptr_param_leak_ReLU Function-0" = getelementptr {[1 x double], [1 x double], [1 x double]}, {[1 x double], [1 x double], [1 x double]}* %".1", i32 0, i32 1 , !dbg !7
  %".16" = load [1 x double], [1 x double]* %"ptr_param_gain_ReLU Function-0", !dbg !7
  %".17" = extractvalue [1 x double] %".16", 0 , !dbg !7
  %".18" = load [1 x double], [1 x double]* %"ptr_param_bias_ReLU Function-0", !dbg !7
  %".19" = extractvalue [1 x double] %".18", 0 , !dbg !7
  %".20" = load [1 x double], [1 x double]* %"ptr_param_leak_ReLU Function-0", !dbg !7
  %".21" = extractvalue [1 x double] %".20", 0 , !dbg !7
  %".22" = load double, double* %".14", !dbg !7
  %".23" = fsub double %".22", %".19", !dbg !7
  %".24" = fmul double %".23", %".17", !dbg !7
  %".25" = fmul double %".24", %".21", !dbg !7
  %".26" = call double @"llvm.maxnum.f64"(double %".24", double %".25"), !dbg !7
  store double %".26", double* %".15", !dbg !7
  %"nested_transfer_loop_index_var_inc" = add i32 %"nested_transfer_loop_loop_index_var", 1, !dbg !7
  store i32 %"nested_transfer_loop_index_var_inc", i32* %"nested_transfer_loop_index_var_loc", !dbg !7
  br label %"nested_transfer_loop-cond-bb", !dbg !7
nested_transfer_loop-cond-bb.endif:
  %"transfer_loop_index_var_inc" = add i32 %"transfer_loop_loop_index_var", 1, !dbg !7
  store i32 %"transfer_loop_index_var_inc", i32* %"transfer_loop_index_var_loc", !dbg !7
  br label %"transfer_loop-cond-bb", !dbg !7
}

declare double @"llvm.maxnum.f64"(double %".1", double %".2") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/functions/nonstateful", filename: "transferfunctions.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_ReLU_ReLU_Function_0__111", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = !{ !"branch_weights", i32 99, i32 1 }