; ModuleID = "PsyNeuLinkModule-125"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_OneHot_OneHot_Function_10__136"({double}* noalias nonnull %".1", {[1 x {[624 x i32], i32, i32, double, i32}]}* noalias nonnull %".2", [2 x double]* noalias nonnull %".3", [2 x double]* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %".6" = alloca i32, !dbg !7
  store i32 0, i32* %".6", !dbg !7
  %"search_index_var_loc" = alloca i32, !dbg !7
  store i32 0, i32* %"search_index_var_loc", !dbg !7
  br label %"search-cond-bb", !dbg !7
search-cond-bb:
  %"search_cond_index_var" = load i32, i32* %"search_index_var_loc", !dbg !7
  %"search_loop_cond" = icmp slt i32 %"search_cond_index_var", 2 , !dbg !7
  br i1 %"search_loop_cond", label %"search-cond-bb.if", label %"search-cond-bb.endif", !dbg !7, !prof !8
search-cond-bb.if:
  %"search_loop_index_var" = load i32, i32* %"search_index_var_loc", !dbg !7
  %".11" = load i32, i32* %".6", !dbg !7
  %".12" = getelementptr [2 x double], [2 x double]* %".3", i32 0, i32 %".11" , !dbg !7
  %".13" = getelementptr [2 x double], [2 x double]* %".3", i32 0, i32 %"search_loop_index_var" , !dbg !7
  %".14" = load double, double* %".12", !dbg !7
  %".15" = load double, double* %".13", !dbg !7
  %".16" = getelementptr [2 x double], [2 x double]* %".4", i32 0, i32 %".11" , !dbg !7
  %".17" = getelementptr [2 x double], [2 x double]* %".4", i32 0, i32 %"search_loop_index_var" , !dbg !7
  store double              0x0, double* %".17", !dbg !7
  %".19" = fcmp oge double %".15", %".14" , !dbg !7
  br i1 %".19", label %"search-cond-bb.if.if", label %"search-cond-bb.if.endif", !dbg !7
search-cond-bb.endif:
  ret void, !dbg !7
search-cond-bb.if.if:
  store double              0x0, double* %".16", !dbg !7
  store double 0x3ff0000000000000, double* %".17", !dbg !7
  store i32 %"search_loop_index_var", i32* %".6", !dbg !7
  br label %"search-cond-bb.if.endif", !dbg !7
search-cond-bb.if.endif:
  %"search_index_var_inc" = add i32 %"search_loop_index_var", 1, !dbg !7
  store i32 %"search_index_var_inc", i32* %"search_index_var_loc", !dbg !7
  br label %"search-cond-bb", !dbg !7
}

declare double @"llvm.fabs.f64"(double %".1") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/functions/nonstateful", filename: "selectionfunctions.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_OneHot_OneHot_Function_10__136", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = !{ !"branch_weights", i32 99, i32 1 }