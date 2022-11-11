; ModuleID = "PsyNeuLinkModule-23"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"_Identity_Identity_Function_5__23"({}* noalias nonnull %".1", {}* noalias nonnull %".2", [2 x double]* noalias nonnull %".3", [2 x double]* noalias nonnull %".4") argmemonly!dbg !6
{
entry:
  %".6" = load [2 x double], [2 x double]* %".3", !dbg !7
  store [2 x double] %".6", [2 x double]* %".4", !dbg !7
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
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_Identity_Identity_Function_5__23", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)