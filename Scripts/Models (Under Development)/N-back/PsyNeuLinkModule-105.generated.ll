; ModuleID = "PsyNeuLinkModule-105"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define i1 @"_TransferMechanism_CURRENT_STIMULUS__is_finished_114"({{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* nonnull %".1", {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* nonnull %".2", [1 x [1 x [8 x double]]]* nonnull %".3", {[8 x double]}* nonnull %".4") argmemonly!dbg !6
{
entry:
  %"ptr_state_value_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2 , !dbg !7
  %"ptr_state_value_CURRENT STIMULUS_hist0" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 2, i32 0 , !dbg !7
  %"ptr_param_termination_threshold_CURRENT STIMULUS" = getelementptr {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}, {{{{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}, {{double, double}}}, {[1 x double], [1 x double], [1 x double], [1 x [8 x double]]}, {}, double, double, double, {{{double, double}}}, {double}, {{{double, {}, double, {}}, {}, {}}}, double, {[1 x double], [1 x double], [1 x double]}, {}}* %".1", i32 0, i32 11 , !dbg !7
  %"ptr_state_is_finished_flag_CURRENT STIMULUS" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 3 , !dbg !7
  %"ptr_state_is_finished_flag_CURRENT STIMULUS_hist0" = getelementptr {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}, {{{{}}, {{}}, {{}}, {{}}, {{}}, {{}}}, {[1 x [1 x [8 x double]]]}, [2 x [1 x [8 x double]]], [1 x double], [1 x double], {{{}}}, {}, {{{}}}, {}, [1 x [5 x i32]]}* %".2", i32 0, i32 3, i32 0 , !dbg !7
  %".6" = load double, double* %"ptr_state_is_finished_flag_CURRENT STIMULUS_hist0", !dbg !7
  %".7" = fcmp one double %".6",              0x0 , !dbg !7
  ret i1 %".7", !dbg !7
}

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "/Users/jdc/PyCharmProjects/PsyNeuLink/psyneulink/core/components/mechanisms/processing", filename: "transfermechanism.py")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "_TransferMechanism_CURRENT_STIMULUS__is_finished_114", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)