; ModuleID = "PsyNeuLinkModule-36"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_OutputPort_OutputPort_0__37"({{double, double}}* noalias nonnull %".1", {{}}* noalias nonnull %".2", [1 x double]* noalias nonnull %".3", [1 x double]* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %"ptr_param_function_OutputPort-0" = getelementptr {{double, double}}, {{double, double}}* %".1", i32 0, i32 0 , !dbg !7
  %"ptr_state_function_OutputPort-0" = getelementptr {{}}, {{}}* %".2", i32 0, i32 0 , !dbg !7
  call void @"_Linear_Linear_Function_62__38"({double, double}* %"ptr_param_function_OutputPort-0", {}* %"ptr_state_function_OutputPort-0", [1 x double]* %".3", [1 x double]* %".4"), !dbg !7
  ret void, !dbg !7
}

declare void @"_Linear_Linear_Function_62__38"({double, double}* %".1", {}* %".2", [1 x double]* %".3", [1 x double]* %".4") 

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/ports", filename: "outputport.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_OutputPort_OutputPort_0__37", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)