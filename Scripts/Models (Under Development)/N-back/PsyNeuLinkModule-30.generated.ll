; ModuleID = "PsyNeuLinkModule-30"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define i1 @"_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Input_CIM__is_finished_learning_30"({{}, {{{double, {}, double, {}}, {}, {}}, {{double, {}, double, {}}, {}, {}}, {{double, {}, double, {}}, {}, {}}, {{double, {}, double, {}}, {}, {}}, {{double, {}, double, {}}, {}, {}}, {{double, {}, double, {}}, {}, {}}}, double, {}, double, double, {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}}* nonnull %".1", {{}, {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, [1 x {[25 x double], [8 x double], [2 x double], [8 x double], [25 x double], [2 x double]}], {}, [1 x double], [1 x [5 x i32]], [1 x double], {{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}}* nonnull %".2", {[2 x [25 x double]], [2 x [8 x double]], [2 x [2 x double]], [2 x [8 x double]], [2 x [25 x double]], [1 x [2 x double]]}* nonnull %".3", {[25 x double], [8 x double], [2 x double], [8 x double], [25 x double], [2 x double]}* nonnull %".4") argmemonly!dbg !6
{
entry:
  ret i1 1, !dbg !7
}

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/mechanisms/processing", filename: "compositioninterfacemechanism.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_CompositionInterfaceMechanism_WORKING_MEMORY__fnn__Input_CIM__is_finished_learning_30", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)