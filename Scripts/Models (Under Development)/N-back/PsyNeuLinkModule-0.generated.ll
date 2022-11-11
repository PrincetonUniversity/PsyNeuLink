; ModuleID = "PsyNeuLinkModule-0"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare double @"__pnl_builtin_cos"(double %".1") 

declare double @"__pnl_builtin_sin"(double %".1") 

declare double @"__pnl_builtin_exp"(double %".1") 

declare double @"__pnl_builtin_log"(double %".1") 

declare double @"__pnl_builtin_pow"(double %".1", double %".2") 

define double @"__pnl_builtin_csch"(double %".1") argmemonly!dbg !6
{
entry:
  %".3" = call double @"__pnl_builtin_exp"(double %".1"), !dbg !7
  %".4" = fsub double 0x8000000000000000, %".1", !dbg !7
  %".5" = call double @"__pnl_builtin_exp"(double %".4"), !dbg !7
  %".6" = fsub double %".3", %".5", !dbg !7
  %".7" = fdiv double 0x4000000000000000, %".6", !dbg !7
  ret double %".7", !dbg !7
}

define double @"__pnl_builtin_coth"(double %".1") argmemonly!dbg !9
{
entry:
  %".3" = fmul double 0x4000000000000000, %".1", !dbg !10
  %".4" = call double @"__pnl_builtin_exp"(double %".3"), !dbg !10
  %".5" = fsub double %".4", 0x3ff0000000000000, !dbg !10
  %".6" = fdiv double 0x4000000000000000, %".5", !dbg !10
  %".7" = fadd double 0x3ff0000000000000, %".6", !dbg !10
  ret double %".7", !dbg !10
}

define double @"__pnl_builtin_tanh"(double %".1") argmemonly!dbg !12
{
entry:
  %".3" = fmul double 0x4000000000000000, %".1", !dbg !13
  %".4" = call double @"__pnl_builtin_exp"(double %".3"), !dbg !13
  %".5" = fadd double %".4", 0x3ff0000000000000, !dbg !13
  %".6" = fdiv double 0x4000000000000000, %".5", !dbg !13
  %".7" = fsub double 0x3ff0000000000000, %".6", !dbg !13
  ret double %".7", !dbg !13
}

define i1 @"__pnl_builtin_is_close_double"(double %".1", double %".2", double %".3", double %".4") argmemonly!dbg !15
{
entry:
  %"is_close_diff" = fsub double %".1", %".2", !dbg !16
  %"is_close_abs" = call double @"llvm.fabs.f64"(double %"is_close_diff"), !dbg !16
  %"abs_val2" = call double @"llvm.fabs.f64"(double %".2"), !dbg !16
  %"is_close_rtol" = fmul double %".3", %"abs_val2", !dbg !16
  %"is_close_atol" = fadd double %"is_close_rtol", %".4", !dbg !16
  %"is_close_cmp" = fcmp ole double %"is_close_abs", %"is_close_atol" , !dbg !16
  ret i1 %"is_close_cmp", !dbg !16
}

declare double @"llvm.fabs.f64"(double %".1") 

define void @"__pnl_builtin_mt_rand_init_scalar"({[624 x i32], i32, i32, double, i32}* noalias nonnull %".1", i32 %".2") argmemonly!dbg !18
{
entry:
  %".4" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 0 , !dbg !19
  %".5" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 0 , !dbg !19
  %".6" = and i32 %".2", 4294967295, !dbg !19
  store i32 %".6", i32* %".5", !dbg !19
  %".8" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 2 , !dbg !19
  store i32 0, i32* %".8", !dbg !19
  %".10" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 3 , !dbg !19
  store double              0x0, double* %".10", !dbg !19
  %"init_seed_index_var_loc" = alloca i32, !dbg !19
  store i32 1, i32* %"init_seed_index_var_loc", !dbg !19
  br label %"init_seed-cond-bb", !dbg !19
init_seed-cond-bb:
  %"init_seed_cond_index_var" = load i32, i32* %"init_seed_index_var_loc", !dbg !19
  %"init_seed_loop_cond" = icmp slt i32 %"init_seed_cond_index_var", 624 , !dbg !19
  br i1 %"init_seed_loop_cond", label %"init_seed-cond-bb.if", label %"init_seed-cond-bb.endif", !dbg !19, !prof !20
init_seed-cond-bb.if:
  %"init_seed_loop_index_var" = load i32, i32* %"init_seed_index_var_loc", !dbg !19
  %".15" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %"init_seed_loop_index_var" , !dbg !19
  %".16" = sub i32 %"init_seed_loop_index_var", 1, !dbg !19
  %".17" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %".16" , !dbg !19
  %".18" = load i32, i32* %".17", !dbg !19
  %".19" = lshr i32 %".18", 30, !dbg !19
  %".20" = xor i32 %".18", %".19", !dbg !19
  %".21" = mul i32 %".20", 1812433253, !dbg !19
  %".22" = add i32 %".21", %"init_seed_loop_index_var", !dbg !19
  %".23" = and i32 %".22", 4294967295, !dbg !19
  store i32 %".23", i32* %".15", !dbg !19
  %"init_seed_index_var_inc" = add i32 %"init_seed_loop_index_var", 1, !dbg !19
  store i32 %"init_seed_index_var_inc", i32* %"init_seed_index_var_loc", !dbg !19
  br label %"init_seed-cond-bb", !dbg !19
init_seed-cond-bb.endif:
  %".27" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 1 , !dbg !19
  store i32 624, i32* %".27", !dbg !19
  %".29" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 4 , !dbg !19
  store i32 %".2", i32* %".29", !dbg !19
  ret void, !dbg !19
}

define void @"__pnl_builtin_mt_rand_init"({[624 x i32], i32, i32, double, i32}* noalias nonnull %".1", i32 %".2") argmemonly!dbg !22
{
entry:
  call void @"__pnl_builtin_mt_rand_init_scalar"({[624 x i32], i32, i32, double, i32}* %".1", i32 19650218), !dbg !23
  %"key_array" = alloca [1 x i32], !dbg !23
  %".5" = getelementptr [1 x i32], [1 x i32]* %"key_array", i32 0, i32 0 , !dbg !23
  store i32 %".2", i32* %".5", !dbg !23
  %"pi_slot" = alloca i32, !dbg !23
  store i32 1, i32* %"pi_slot", !dbg !23
  %"pj_slot" = alloca i32, !dbg !23
  store i32 0, i32* %"pj_slot", !dbg !23
  %".9" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 0 , !dbg !23
  %".10" = getelementptr [624 x i32], [624 x i32]* %".9", i32 0, i32 0 , !dbg !23
  %"add_key_index_var_loc" = alloca i32, !dbg !23
  store i32 0, i32* %"add_key_index_var_loc", !dbg !23
  br label %"add_key-cond-bb", !dbg !23
add_key-cond-bb:
  %"add_key_cond_index_var" = load i32, i32* %"add_key_index_var_loc", !dbg !23
  %"add_key_loop_cond" = icmp slt i32 %"add_key_cond_index_var", 624 , !dbg !23
  br i1 %"add_key_loop_cond", label %"add_key-cond-bb.if", label %"add_key-cond-bb.endif", !dbg !23, !prof !20
add_key-cond-bb.if:
  %"add_key_loop_index_var" = load i32, i32* %"add_key_index_var_loc", !dbg !23
  %".14" = load i32, i32* %"pi_slot", !dbg !23
  %".15" = sub i32 %".14", 1, !dbg !23
  %".16" = getelementptr [624 x i32], [624 x i32]* %".9", i32 0, i32 %".14" , !dbg !23
  %".17" = getelementptr [624 x i32], [624 x i32]* %".9", i32 0, i32 %".15" , !dbg !23
  %".18" = load i32, i32* %"pj_slot", !dbg !23
  %".19" = getelementptr [1 x i32], [1 x i32]* %"key_array", i32 0, i32 %".18" , !dbg !23
  %".20" = add i32 %".18", 1, !dbg !23
  %".21" = icmp uge i32 %".20", 1 , !dbg !23
  %".22" = select i1 %".21", i32 0, i32 %".20" , !dbg !23
  store i32 %".22", i32* %"pj_slot", !dbg !23
  %".24" = load i32, i32* %".17", !dbg !23
  %".25" = lshr i32 %".24", 30, !dbg !23
  %".26" = xor i32 %".24", %".25", !dbg !23
  %".27" = mul i32 %".26", 1664525, !dbg !23
  %".28" = load i32, i32* %".16", !dbg !23
  %".29" = xor i32 %".28", %".27", !dbg !23
  %".30" = load i32, i32* %".19", !dbg !23
  %".31" = add i32 %".29", %".30", !dbg !23
  %".32" = add i32 %".31", %".18", !dbg !23
  %".33" = and i32 %".32", 4294967295, !dbg !23
  store i32 %".33", i32* %".16", !dbg !23
  %".35" = add i32 %".14", 1, !dbg !23
  store i32 %".35", i32* %"pi_slot", !dbg !23
  %".37" = icmp uge i32 %".35", 624 , !dbg !23
  br i1 %".37", label %"add_key-cond-bb.if.if", label %"add_key-cond-bb.if.endif", !dbg !23, !prof !24
add_key-cond-bb.endif:
  %"second_shuffle_index_var_loc" = alloca i32, !dbg !23
  store i32 0, i32* %"second_shuffle_index_var_loc", !dbg !23
  br label %"second_shuffle-cond-bb", !dbg !23
add_key-cond-bb.if.if:
  store i32 1, i32* %"pi_slot", !dbg !23
  store i32 %".33", i32* %".10", !dbg !23
  br label %"add_key-cond-bb.if.endif", !dbg !23
add_key-cond-bb.if.endif:
  %"add_key_index_var_inc" = add i32 %"add_key_loop_index_var", 1, !dbg !23
  store i32 %"add_key_index_var_inc", i32* %"add_key_index_var_loc", !dbg !23
  br label %"add_key-cond-bb", !dbg !23
second_shuffle-cond-bb:
  %"second_shuffle_cond_index_var" = load i32, i32* %"second_shuffle_index_var_loc", !dbg !23
  %"second_shuffle_loop_cond" = icmp slt i32 %"second_shuffle_cond_index_var", 623 , !dbg !23
  br i1 %"second_shuffle_loop_cond", label %"second_shuffle-cond-bb.if", label %"second_shuffle-cond-bb.endif", !dbg !23, !prof !20
second_shuffle-cond-bb.if:
  %"second_shuffle_loop_index_var" = load i32, i32* %"second_shuffle_index_var_loc", !dbg !23
  %".47" = load i32, i32* %"pi_slot", !dbg !23
  %".48" = sub i32 %".47", 1, !dbg !23
  %".49" = getelementptr [624 x i32], [624 x i32]* %".9", i32 0, i32 %".47" , !dbg !23
  %".50" = getelementptr [624 x i32], [624 x i32]* %".9", i32 0, i32 %".48" , !dbg !23
  %".51" = load i32, i32* %".50", !dbg !23
  %".52" = lshr i32 %".51", 30, !dbg !23
  %".53" = xor i32 %".51", %".52", !dbg !23
  %".54" = mul i32 %".53", 1566083941, !dbg !23
  %".55" = load i32, i32* %".49", !dbg !23
  %".56" = xor i32 %".55", %".54", !dbg !23
  %".57" = sub i32 %".56", %".47", !dbg !23
  %".58" = and i32 %".57", 4294967295, !dbg !23
  store i32 %".58", i32* %".49", !dbg !23
  %".60" = add i32 %".47", 1, !dbg !23
  store i32 %".60", i32* %"pi_slot", !dbg !23
  %".62" = icmp uge i32 %".60", 624 , !dbg !23
  br i1 %".62", label %"second_shuffle-cond-bb.if.if", label %"second_shuffle-cond-bb.if.endif", !dbg !23, !prof !24
second_shuffle-cond-bb.endif:
  store i32 2147483648, i32* %".10", !dbg !23
  %".70" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 4 , !dbg !23
  store i32 %".2", i32* %".70", !dbg !23
  ret void, !dbg !23
second_shuffle-cond-bb.if.if:
  store i32 1, i32* %"pi_slot", !dbg !23
  store i32 %".58", i32* %".10", !dbg !23
  br label %"second_shuffle-cond-bb.if.endif", !dbg !23
second_shuffle-cond-bb.if.endif:
  %"second_shuffle_index_var_inc" = add i32 %"second_shuffle_loop_index_var", 1, !dbg !23
  store i32 %"second_shuffle_index_var_inc", i32* %"second_shuffle_index_var_loc", !dbg !23
  br label %"second_shuffle-cond-bb", !dbg !23
}

define void @"__pnl_builtin_mt_rand_int32"({[624 x i32], i32, i32, double, i32}* noalias nonnull %".1", i64* noalias nonnull %".2") argmemonly!dbg !26
{
entry:
  %".4" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 0 , !dbg !27
  %".5" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 1 , !dbg !27
  %".6" = load i32, i32* %".5", !dbg !27
  %".7" = icmp sge i32 %".6", 624 , !dbg !27
  br i1 %".7", label %"entry.if", label %"entry.endif", !dbg !27, !prof !24
entry.if:
  %".9" = extractvalue [2 x i32] [i32 0, i32 2567483615], 0 , !dbg !27
  %".10" = extractvalue [2 x i32] [i32 0, i32 2567483615], 1 , !dbg !27
  %"first_half_index_var_loc" = alloca i32, !dbg !27
  store i32 0, i32* %"first_half_index_var_loc", !dbg !27
  br label %"first_half-cond-bb", !dbg !27
entry.endif:
  %".61" = load i32, i32* %".5", !dbg !27
  %".62" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %".61" , !dbg !27
  %".63" = add i32 %".61", 1, !dbg !27
  store i32 %".63", i32* %".5", !dbg !27
  %".65" = load i32, i32* %".62", !dbg !27
  %".66" = lshr i32 %".65", 11, !dbg !27
  %".67" = xor i32 %".65", %".66", !dbg !27
  %".68" = shl i32 %".67", 7, !dbg !27
  %".69" = and i32 %".68", 2636928640, !dbg !27
  %".70" = xor i32 %".67", %".69", !dbg !27
  %".71" = shl i32 %".70", 15, !dbg !27
  %".72" = and i32 %".71", 4022730752, !dbg !27
  %".73" = xor i32 %".70", %".72", !dbg !27
  %".74" = lshr i32 %".73", 18, !dbg !27
  %".75" = xor i32 %".73", %".74", !dbg !27
  %".76" = zext i32 %".75" to i64 , !dbg !27
  store i64 %".76", i64* %".2", !dbg !27
  ret void, !dbg !27
first_half-cond-bb:
  %"first_half_cond_index_var" = load i32, i32* %"first_half_index_var_loc", !dbg !27
  %"first_half_loop_cond" = icmp slt i32 %"first_half_cond_index_var", 227 , !dbg !27
  br i1 %"first_half_loop_cond", label %"first_half-cond-bb.if", label %"first_half-cond-bb.endif", !dbg !27, !prof !20
first_half-cond-bb.if:
  %"first_half_loop_index_var" = load i32, i32* %"first_half_index_var_loc", !dbg !27
  %".14" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %"first_half_loop_index_var" , !dbg !27
  %".15" = add i32 %"first_half_loop_index_var", 1, !dbg !27
  %".16" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %".15" , !dbg !27
  %".17" = load i32, i32* %".14", !dbg !27
  %".18" = and i32 %".17", 2147483648, !dbg !27
  %".19" = load i32, i32* %".16", !dbg !27
  %".20" = and i32 %".19", 2147483647, !dbg !27
  %".21" = or i32 %".18", %".20", !dbg !27
  %".22" = and i32 %".21", 1, !dbg !27
  %".23" = trunc i32 %".22" to i1 , !dbg !27
  %".24" = select i1 %".23", i32 %".10", i32 %".9" , !dbg !27
  %".25" = lshr i32 %".21", 1, !dbg !27
  %".26" = add i32 %"first_half_loop_index_var", 397, !dbg !27
  %".27" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %".26" , !dbg !27
  %".28" = load i32, i32* %".27", !dbg !27
  %".29" = xor i32 %".28", %".25", !dbg !27
  %".30" = xor i32 %".29", %".24", !dbg !27
  store i32 %".30", i32* %".14", !dbg !27
  %"first_half_index_var_inc" = add i32 %"first_half_loop_index_var", 1, !dbg !27
  store i32 %"first_half_index_var_inc", i32* %"first_half_index_var_loc", !dbg !27
  br label %"first_half-cond-bb", !dbg !27
first_half-cond-bb.endif:
  %"second_half_index_var_loc" = alloca i32, !dbg !27
  store i32 227, i32* %"second_half_index_var_loc", !dbg !27
  br label %"second_half-cond-bb", !dbg !27
second_half-cond-bb:
  %"second_half_cond_index_var" = load i32, i32* %"second_half_index_var_loc", !dbg !27
  %"second_half_loop_cond" = icmp slt i32 %"second_half_cond_index_var", 624 , !dbg !27
  br i1 %"second_half_loop_cond", label %"second_half-cond-bb.if", label %"second_half-cond-bb.endif", !dbg !27, !prof !20
second_half-cond-bb.if:
  %"second_half_loop_index_var" = load i32, i32* %"second_half_index_var_loc", !dbg !27
  %".37" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %"second_half_loop_index_var" , !dbg !27
  %".38" = icmp eq i32 %"second_half_loop_index_var", 623 , !dbg !27
  %".39" = add i32 %"second_half_loop_index_var", 1, !dbg !27
  %".40" = select i1 %".38", i32 0, i32 %".39" , !dbg !27
  %".41" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %".40" , !dbg !27
  %".42" = load i32, i32* %".37", !dbg !27
  %".43" = and i32 %".42", 2147483648, !dbg !27
  %".44" = load i32, i32* %".41", !dbg !27
  %".45" = and i32 %".44", 2147483647, !dbg !27
  %".46" = or i32 %".43", %".45", !dbg !27
  %".47" = and i32 %".46", 1, !dbg !27
  %".48" = trunc i32 %".47" to i1 , !dbg !27
  %".49" = select i1 %".48", i32 %".10", i32 %".9" , !dbg !27
  %".50" = lshr i32 %".46", 1, !dbg !27
  %".51" = add i32 %"second_half_loop_index_var", -227, !dbg !27
  %".52" = getelementptr [624 x i32], [624 x i32]* %".4", i32 0, i32 %".51" , !dbg !27
  %".53" = load i32, i32* %".52", !dbg !27
  %".54" = xor i32 %".53", %".50", !dbg !27
  %".55" = xor i32 %".54", %".49", !dbg !27
  store i32 %".55", i32* %".37", !dbg !27
  %"second_half_index_var_inc" = add i32 %"second_half_loop_index_var", 1, !dbg !27
  store i32 %"second_half_index_var_inc", i32* %"second_half_index_var_loc", !dbg !27
  br label %"second_half-cond-bb", !dbg !27
second_half-cond-bb.endif:
  store i32 0, i32* %".5", !dbg !27
  br label %"entry.endif", !dbg !27
}

define void @"__pnl_builtin_mt_rand_double"({[624 x i32], i32, i32, double, i32}* noalias nonnull %".1", double* noalias nonnull %".2") argmemonly!dbg !29
{
entry:
  %"al_gen_int" = alloca i64, !dbg !30
  call void @"__pnl_builtin_mt_rand_int32"({[624 x i32], i32, i32, double, i32}* %".1", i64* %"al_gen_int"), !dbg !30
  %"bl_gen_int" = alloca i64, !dbg !30
  call void @"__pnl_builtin_mt_rand_int32"({[624 x i32], i32, i32, double, i32}* %".1", i64* %"bl_gen_int"), !dbg !30
  %".6" = load i64, i64* %"al_gen_int", !dbg !30
  %".7" = load i64, i64* %"bl_gen_int", !dbg !30
  %".8" = lshr i64 %".6", 5, !dbg !30
  %".9" = lshr i64 %".7", 6, !dbg !30
  %".10" = uitofp i64 %".8" to double , !dbg !30
  %".11" = uitofp i64 %".9" to double , !dbg !30
  %".12" = fmul double %".10", 0x4190000000000000, !dbg !30
  %".13" = fadd double %".12", %".11", !dbg !30
  %".14" = fdiv double %".13", 0x4340000000000000, !dbg !30
  %".15" = fcmp oge double %".14",              0x0 , !dbg !30
  call void @"llvm.assume"(i1 %".15"), !dbg !30
  %".17" = fcmp olt double %".14", 0x3ff0000000000000 , !dbg !30
  call void @"llvm.assume"(i1 %".17"), !dbg !30
  store double %".14", double* %".2", !dbg !30
  ret void, !dbg !30
}

declare void @"llvm.assume"(i1 %".1") 

define void @"__pnl_builtin_mt_rand_normal"({[624 x i32], i32, i32, double, i32}* noalias nonnull %".1", double* noalias nonnull %".2") argmemonly!dbg !32
{
entry:
  %".4" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 3 , !dbg !33
  %".5" = getelementptr {[624 x i32], i32, i32, double, i32}, {[624 x i32], i32, i32, double, i32}* %".1", i32 0, i32 2 , !dbg !33
  %".6" = load i32, i32* %".5", !dbg !33
  %".7" = icmp eq i32 %".6", 1 , !dbg !33
  br i1 %".7", label %"entry.if", label %"entry.endif", !dbg !33, !prof !24
entry.if:
  %".9" = load double, double* %".4", !dbg !33
  store double %".9", double* %".2", !dbg !33
  store double              0x0, double* %".4", !dbg !33
  store i32 0, i32* %".5", !dbg !33
  ret void, !dbg !33
entry.endif:
  br label %"gen_loop_gauss", !dbg !33
gen_loop_gauss:
  %"mt_rand_normal_tmp" = alloca double, !dbg !33
  call void @"__pnl_builtin_mt_rand_double"({[624 x i32], i32, i32, double, i32}* %".1", double* %"mt_rand_normal_tmp"), !dbg !33
  %".16" = load double, double* %"mt_rand_normal_tmp", !dbg !33
  %".17" = fmul double %".16", 0x4000000000000000, !dbg !33
  %".18" = fsub double %".17", 0x3ff0000000000000, !dbg !33
  call void @"__pnl_builtin_mt_rand_double"({[624 x i32], i32, i32, double, i32}* %".1", double* %"mt_rand_normal_tmp"), !dbg !33
  %".20" = load double, double* %"mt_rand_normal_tmp", !dbg !33
  %".21" = fmul double %".20", 0x4000000000000000, !dbg !33
  %".22" = fsub double %".21", 0x3ff0000000000000, !dbg !33
  %".23" = fmul double %".18", %".18", !dbg !33
  %".24" = fmul double %".22", %".22", !dbg !33
  %".25" = fadd double %".23", %".24", !dbg !33
  %".26" = fcmp uge double %".25", 0x3ff0000000000000 , !dbg !33
  %".27" = fcmp ueq double %".25",              0x0 , !dbg !33
  %".28" = or i1 %".26", %".27", !dbg !33
  br i1 %".28", label %"gen_loop_gauss", label %"gen_gauss_out", !dbg !33, !prof !24
gen_gauss_out:
  %".30" = call double @"__pnl_builtin_log"(double %".25"), !dbg !33
  %".31" = fmul double %".30", 0xc000000000000000, !dbg !33
  %".32" = fdiv double %".31", %".25", !dbg !33
  %".33" = call double @"llvm.sqrt.f64"(double %".32"), !dbg !33
  %".34" = fmul double %".33", %".22", !dbg !33
  store double %".34", double* %".2", !dbg !33
  %".36" = fmul double %".33", %".18", !dbg !33
  store double %".36", double* %".4", !dbg !33
  store i32 1, i32* %".5", !dbg !33
  ret void, !dbg !33
}

declare double @"llvm.sqrt.f64"(double %".1") 

define void @"__pnl_builtin_philox_rand_init"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* noalias nonnull %".1", i64 %".2") argmemonly!dbg !35
{
entry:
  store {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64} zeroinitializer, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", !dbg !36
  %".5" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 4 , !dbg !36
  store i16 4, i16* %".5", !dbg !36
  %".7" = trunc i64 %".2" to i32 , !dbg !36
  %".8" = lshr i64 %".2", 32, !dbg !36
  %".9" = trunc i64 %".8" to i32 , !dbg !36
  %".10" = insertvalue [4 x i32] zeroinitializer, i32 %".7", 0 , !dbg !36
  %".11" = insertvalue [4 x i32] %".10", i32 %".9", 1 , !dbg !36
  %".12" = extractvalue [4 x i32] %".11", 0 , !dbg !36
  %".13" = xor i32 %".12", 1135663077, !dbg !36
  %".14" = mul i32 1135663077, 2468251765, !dbg !36
  %".15" = mul i32 %".13", %".14", !dbg !36
  %".16" = lshr i32 %".15", 16, !dbg !36
  %".17" = xor i32 %".15", %".16", !dbg !36
  %".18" = insertvalue [4 x i32] zeroinitializer, i32 %".17", 0 , !dbg !36
  %".19" = extractvalue [4 x i32] %".11", 1 , !dbg !36
  %".20" = xor i32 %".19", %".14", !dbg !36
  %".21" = mul i32 %".14", 2468251765, !dbg !36
  %".22" = mul i32 %".20", %".21", !dbg !36
  %".23" = lshr i32 %".22", 16, !dbg !36
  %".24" = xor i32 %".22", %".23", !dbg !36
  %".25" = insertvalue [4 x i32] %".18", i32 %".24", 1 , !dbg !36
  %".26" = extractvalue [4 x i32] %".11", 2 , !dbg !36
  %".27" = xor i32 %".26", %".21", !dbg !36
  %".28" = mul i32 %".21", 2468251765, !dbg !36
  %".29" = mul i32 %".27", %".28", !dbg !36
  %".30" = lshr i32 %".29", 16, !dbg !36
  %".31" = xor i32 %".29", %".30", !dbg !36
  %".32" = insertvalue [4 x i32] %".25", i32 %".31", 2 , !dbg !36
  %".33" = extractvalue [4 x i32] %".11", 3 , !dbg !36
  %".34" = xor i32 %".33", %".28", !dbg !36
  %".35" = mul i32 %".28", 2468251765, !dbg !36
  %".36" = mul i32 %".34", %".35", !dbg !36
  %".37" = lshr i32 %".36", 16, !dbg !36
  %".38" = xor i32 %".36", %".37", !dbg !36
  %".39" = insertvalue [4 x i32] %".32", i32 %".38", 3 , !dbg !36
  %".40" = extractvalue [4 x i32] %".39", 0 , !dbg !36
  %".41" = extractvalue [4 x i32] %".39", 1 , !dbg !36
  %".42" = xor i32 %".40", %".35", !dbg !36
  %".43" = mul i32 %".35", 2468251765, !dbg !36
  %".44" = mul i32 %".42", %".43", !dbg !36
  %".45" = lshr i32 %".44", 16, !dbg !36
  %".46" = xor i32 %".44", %".45", !dbg !36
  %".47" = mul i32 %".41", 3389127133, !dbg !36
  %".48" = mul i32 %".46", 1232336661, !dbg !36
  %".49" = sub i32 %".47", %".48", !dbg !36
  %".50" = lshr i32 %".49", 16, !dbg !36
  %".51" = xor i32 %".49", %".50", !dbg !36
  %".52" = insertvalue [4 x i32] %".39", i32 %".51", 1 , !dbg !36
  %".53" = extractvalue [4 x i32] %".52", 0 , !dbg !36
  %".54" = extractvalue [4 x i32] %".52", 2 , !dbg !36
  %".55" = xor i32 %".53", %".43", !dbg !36
  %".56" = mul i32 %".43", 2468251765, !dbg !36
  %".57" = mul i32 %".55", %".56", !dbg !36
  %".58" = lshr i32 %".57", 16, !dbg !36
  %".59" = xor i32 %".57", %".58", !dbg !36
  %".60" = mul i32 %".54", 3389127133, !dbg !36
  %".61" = mul i32 %".59", 1232336661, !dbg !36
  %".62" = sub i32 %".60", %".61", !dbg !36
  %".63" = lshr i32 %".62", 16, !dbg !36
  %".64" = xor i32 %".62", %".63", !dbg !36
  %".65" = insertvalue [4 x i32] %".52", i32 %".64", 2 , !dbg !36
  %".66" = extractvalue [4 x i32] %".65", 0 , !dbg !36
  %".67" = extractvalue [4 x i32] %".65", 3 , !dbg !36
  %".68" = xor i32 %".66", %".56", !dbg !36
  %".69" = mul i32 %".56", 2468251765, !dbg !36
  %".70" = mul i32 %".68", %".69", !dbg !36
  %".71" = lshr i32 %".70", 16, !dbg !36
  %".72" = xor i32 %".70", %".71", !dbg !36
  %".73" = mul i32 %".67", 3389127133, !dbg !36
  %".74" = mul i32 %".72", 1232336661, !dbg !36
  %".75" = sub i32 %".73", %".74", !dbg !36
  %".76" = lshr i32 %".75", 16, !dbg !36
  %".77" = xor i32 %".75", %".76", !dbg !36
  %".78" = insertvalue [4 x i32] %".65", i32 %".77", 3 , !dbg !36
  %".79" = extractvalue [4 x i32] %".78", 1 , !dbg !36
  %".80" = extractvalue [4 x i32] %".78", 0 , !dbg !36
  %".81" = xor i32 %".79", %".69", !dbg !36
  %".82" = mul i32 %".69", 2468251765, !dbg !36
  %".83" = mul i32 %".81", %".82", !dbg !36
  %".84" = lshr i32 %".83", 16, !dbg !36
  %".85" = xor i32 %".83", %".84", !dbg !36
  %".86" = mul i32 %".80", 3389127133, !dbg !36
  %".87" = mul i32 %".85", 1232336661, !dbg !36
  %".88" = sub i32 %".86", %".87", !dbg !36
  %".89" = lshr i32 %".88", 16, !dbg !36
  %".90" = xor i32 %".88", %".89", !dbg !36
  %".91" = insertvalue [4 x i32] %".78", i32 %".90", 0 , !dbg !36
  %".92" = extractvalue [4 x i32] %".91", 1 , !dbg !36
  %".93" = extractvalue [4 x i32] %".91", 2 , !dbg !36
  %".94" = xor i32 %".92", %".82", !dbg !36
  %".95" = mul i32 %".82", 2468251765, !dbg !36
  %".96" = mul i32 %".94", %".95", !dbg !36
  %".97" = lshr i32 %".96", 16, !dbg !36
  %".98" = xor i32 %".96", %".97", !dbg !36
  %".99" = mul i32 %".93", 3389127133, !dbg !36
  %".100" = mul i32 %".98", 1232336661, !dbg !36
  %".101" = sub i32 %".99", %".100", !dbg !36
  %".102" = lshr i32 %".101", 16, !dbg !36
  %".103" = xor i32 %".101", %".102", !dbg !36
  %".104" = insertvalue [4 x i32] %".91", i32 %".103", 2 , !dbg !36
  %".105" = extractvalue [4 x i32] %".104", 1 , !dbg !36
  %".106" = extractvalue [4 x i32] %".104", 3 , !dbg !36
  %".107" = xor i32 %".105", %".95", !dbg !36
  %".108" = mul i32 %".95", 2468251765, !dbg !36
  %".109" = mul i32 %".107", %".108", !dbg !36
  %".110" = lshr i32 %".109", 16, !dbg !36
  %".111" = xor i32 %".109", %".110", !dbg !36
  %".112" = mul i32 %".106", 3389127133, !dbg !36
  %".113" = mul i32 %".111", 1232336661, !dbg !36
  %".114" = sub i32 %".112", %".113", !dbg !36
  %".115" = lshr i32 %".114", 16, !dbg !36
  %".116" = xor i32 %".114", %".115", !dbg !36
  %".117" = insertvalue [4 x i32] %".104", i32 %".116", 3 , !dbg !36
  %".118" = extractvalue [4 x i32] %".117", 2 , !dbg !36
  %".119" = extractvalue [4 x i32] %".117", 0 , !dbg !36
  %".120" = xor i32 %".118", %".108", !dbg !36
  %".121" = mul i32 %".108", 2468251765, !dbg !36
  %".122" = mul i32 %".120", %".121", !dbg !36
  %".123" = lshr i32 %".122", 16, !dbg !36
  %".124" = xor i32 %".122", %".123", !dbg !36
  %".125" = mul i32 %".119", 3389127133, !dbg !36
  %".126" = mul i32 %".124", 1232336661, !dbg !36
  %".127" = sub i32 %".125", %".126", !dbg !36
  %".128" = lshr i32 %".127", 16, !dbg !36
  %".129" = xor i32 %".127", %".128", !dbg !36
  %".130" = insertvalue [4 x i32] %".117", i32 %".129", 0 , !dbg !36
  %".131" = extractvalue [4 x i32] %".130", 2 , !dbg !36
  %".132" = extractvalue [4 x i32] %".130", 1 , !dbg !36
  %".133" = xor i32 %".131", %".121", !dbg !36
  %".134" = mul i32 %".121", 2468251765, !dbg !36
  %".135" = mul i32 %".133", %".134", !dbg !36
  %".136" = lshr i32 %".135", 16, !dbg !36
  %".137" = xor i32 %".135", %".136", !dbg !36
  %".138" = mul i32 %".132", 3389127133, !dbg !36
  %".139" = mul i32 %".137", 1232336661, !dbg !36
  %".140" = sub i32 %".138", %".139", !dbg !36
  %".141" = lshr i32 %".140", 16, !dbg !36
  %".142" = xor i32 %".140", %".141", !dbg !36
  %".143" = insertvalue [4 x i32] %".130", i32 %".142", 1 , !dbg !36
  %".144" = extractvalue [4 x i32] %".143", 2 , !dbg !36
  %".145" = extractvalue [4 x i32] %".143", 3 , !dbg !36
  %".146" = xor i32 %".144", %".134", !dbg !36
  %".147" = mul i32 %".134", 2468251765, !dbg !36
  %".148" = mul i32 %".146", %".147", !dbg !36
  %".149" = lshr i32 %".148", 16, !dbg !36
  %".150" = xor i32 %".148", %".149", !dbg !36
  %".151" = mul i32 %".145", 3389127133, !dbg !36
  %".152" = mul i32 %".150", 1232336661, !dbg !36
  %".153" = sub i32 %".151", %".152", !dbg !36
  %".154" = lshr i32 %".153", 16, !dbg !36
  %".155" = xor i32 %".153", %".154", !dbg !36
  %".156" = insertvalue [4 x i32] %".143", i32 %".155", 3 , !dbg !36
  %".157" = extractvalue [4 x i32] %".156", 3 , !dbg !36
  %".158" = extractvalue [4 x i32] %".156", 0 , !dbg !36
  %".159" = xor i32 %".157", %".147", !dbg !36
  %".160" = mul i32 %".147", 2468251765, !dbg !36
  %".161" = mul i32 %".159", %".160", !dbg !36
  %".162" = lshr i32 %".161", 16, !dbg !36
  %".163" = xor i32 %".161", %".162", !dbg !36
  %".164" = mul i32 %".158", 3389127133, !dbg !36
  %".165" = mul i32 %".163", 1232336661, !dbg !36
  %".166" = sub i32 %".164", %".165", !dbg !36
  %".167" = lshr i32 %".166", 16, !dbg !36
  %".168" = xor i32 %".166", %".167", !dbg !36
  %".169" = insertvalue [4 x i32] %".156", i32 %".168", 0 , !dbg !36
  %".170" = extractvalue [4 x i32] %".169", 3 , !dbg !36
  %".171" = extractvalue [4 x i32] %".169", 1 , !dbg !36
  %".172" = xor i32 %".170", %".160", !dbg !36
  %".173" = mul i32 %".160", 2468251765, !dbg !36
  %".174" = mul i32 %".172", %".173", !dbg !36
  %".175" = lshr i32 %".174", 16, !dbg !36
  %".176" = xor i32 %".174", %".175", !dbg !36
  %".177" = mul i32 %".171", 3389127133, !dbg !36
  %".178" = mul i32 %".176", 1232336661, !dbg !36
  %".179" = sub i32 %".177", %".178", !dbg !36
  %".180" = lshr i32 %".179", 16, !dbg !36
  %".181" = xor i32 %".179", %".180", !dbg !36
  %".182" = insertvalue [4 x i32] %".169", i32 %".181", 1 , !dbg !36
  %".183" = extractvalue [4 x i32] %".182", 3 , !dbg !36
  %".184" = extractvalue [4 x i32] %".182", 2 , !dbg !36
  %".185" = xor i32 %".183", %".173", !dbg !36
  %".186" = mul i32 %".173", 2468251765, !dbg !36
  %".187" = mul i32 %".185", %".186", !dbg !36
  %".188" = lshr i32 %".187", 16, !dbg !36
  %".189" = xor i32 %".187", %".188", !dbg !36
  %".190" = mul i32 %".184", 3389127133, !dbg !36
  %".191" = mul i32 %".189", 1232336661, !dbg !36
  %".192" = sub i32 %".190", %".191", !dbg !36
  %".193" = lshr i32 %".192", 16, !dbg !36
  %".194" = xor i32 %".192", %".193", !dbg !36
  %".195" = insertvalue [4 x i32] %".182", i32 %".194", 2 , !dbg !36
  %".196" = extractvalue [4 x i32] %".195", 0 , !dbg !36
  %".197" = xor i32 %".196", 2337405405, !dbg !36
  %".198" = mul i32 2337405405, 1492356589, !dbg !36
  %".199" = mul i32 %".197", %".198", !dbg !36
  %".200" = lshr i32 %".199", 16, !dbg !36
  %".201" = xor i32 %".199", %".200", !dbg !36
  %".202" = insertvalue [4 x i32] zeroinitializer, i32 %".201", 0 , !dbg !36
  %".203" = extractvalue [4 x i32] %".195", 1 , !dbg !36
  %".204" = xor i32 %".203", %".198", !dbg !36
  %".205" = mul i32 %".198", 1492356589, !dbg !36
  %".206" = mul i32 %".204", %".205", !dbg !36
  %".207" = lshr i32 %".206", 16, !dbg !36
  %".208" = xor i32 %".206", %".207", !dbg !36
  %".209" = insertvalue [4 x i32] %".202", i32 %".208", 1 , !dbg !36
  %".210" = extractvalue [4 x i32] %".195", 2 , !dbg !36
  %".211" = xor i32 %".210", %".205", !dbg !36
  %".212" = mul i32 %".205", 1492356589, !dbg !36
  %".213" = mul i32 %".211", %".212", !dbg !36
  %".214" = lshr i32 %".213", 16, !dbg !36
  %".215" = xor i32 %".213", %".214", !dbg !36
  %".216" = insertvalue [4 x i32] %".209", i32 %".215", 2 , !dbg !36
  %".217" = extractvalue [4 x i32] %".195", 3 , !dbg !36
  %".218" = xor i32 %".217", %".212", !dbg !36
  %".219" = mul i32 %".212", 1492356589, !dbg !36
  %".220" = mul i32 %".218", %".219", !dbg !36
  %".221" = lshr i32 %".220", 16, !dbg !36
  %".222" = xor i32 %".220", %".221", !dbg !36
  %".223" = insertvalue [4 x i32] %".216", i32 %".222", 3 , !dbg !36
  %".224" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 1 , !dbg !36
  %".225" = getelementptr [2 x i64], [2 x i64]* %".224", i32 0, i32 0 , !dbg !36
  %".226" = extractvalue [4 x i32] %".223", 0 , !dbg !36
  %".227" = zext i32 %".226" to i64 , !dbg !36
  %".228" = extractvalue [4 x i32] %".223", 1 , !dbg !36
  %".229" = zext i32 %".228" to i64 , !dbg !36
  %".230" = shl i64 %".229", 32, !dbg !36
  %".231" = or i64 %".227", %".230", !dbg !36
  store i64 %".231", i64* %".225", !dbg !36
  %".233" = getelementptr [2 x i64], [2 x i64]* %".224", i32 0, i32 1 , !dbg !36
  %".234" = extractvalue [4 x i32] %".223", 2 , !dbg !36
  %".235" = zext i32 %".234" to i64 , !dbg !36
  %".236" = extractvalue [4 x i32] %".223", 3 , !dbg !36
  %".237" = zext i32 %".236" to i64 , !dbg !36
  %".238" = shl i64 %".237", 32, !dbg !36
  %".239" = or i64 %".235", %".238", !dbg !36
  store i64 %".239", i64* %".233", !dbg !36
  %".241" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 6 , !dbg !36
  store i64 %".2", i64* %".241", !dbg !36
  ret void, !dbg !36
}

define void @"__pnl_builtin_philox_rand_int64"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* noalias nonnull %".1", i64* noalias nonnull %".2") argmemonly!dbg !38
{
entry:
  %".4" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 0 , !dbg !39
  %".5" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 1 , !dbg !39
  %".6" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 2 , !dbg !39
  %".7" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 4 , !dbg !39
  %".8" = load i16, i16* %".7", !dbg !39
  %".9" = icmp ult i16 %".8", 4 , !dbg !39
  br i1 %".9", label %"entry.if", label %"entry.endif", !dbg !39, !prof !20
entry.if:
  %".11" = getelementptr [4 x i64], [4 x i64]* %".6", i32 0, i16 %".8" , !dbg !39
  %".12" = load i64, i64* %".11", !dbg !39
  store i64 %".12", i64* %".2", !dbg !39
  %".14" = add i16 %".8", 1, !dbg !39
  store i16 %".14", i16* %".7", !dbg !39
  ret void, !dbg !39
entry.endif:
  %".17" = getelementptr [4 x i64], [4 x i64]* %".4", i32 0, i32 0 , !dbg !39
  %".18" = load i64, i64* %".17", !dbg !39
  %".19" = add i64 %".18", 1, !dbg !39
  br i1 true, label %"entry.endif.if", label %"entry.endif.endif", !dbg !39
entry.endif.if:
  store i64 %".19", i64* %".17", !dbg !39
  br label %"entry.endif.endif", !dbg !39
entry.endif.endif:
  %".23" = icmp eq i64 %".19", 0 , !dbg !39
  %".24" = and i1 true, %".23", !dbg !39
  %".25" = getelementptr [4 x i64], [4 x i64]* %".4", i32 0, i32 1 , !dbg !39
  %".26" = load i64, i64* %".25", !dbg !39
  %".27" = add i64 %".26", 1, !dbg !39
  br i1 %".24", label %"entry.endif.endif.if", label %"entry.endif.endif.endif", !dbg !39
entry.endif.endif.if:
  store i64 %".27", i64* %".25", !dbg !39
  br label %"entry.endif.endif.endif", !dbg !39
entry.endif.endif.endif:
  %".31" = icmp eq i64 %".27", 0 , !dbg !39
  %".32" = and i1 %".24", %".31", !dbg !39
  %".33" = getelementptr [4 x i64], [4 x i64]* %".4", i32 0, i32 2 , !dbg !39
  %".34" = load i64, i64* %".33", !dbg !39
  %".35" = add i64 %".34", 1, !dbg !39
  br i1 %".32", label %"entry.endif.endif.endif.if", label %"entry.endif.endif.endif.endif", !dbg !39
entry.endif.endif.endif.if:
  store i64 %".35", i64* %".33", !dbg !39
  br label %"entry.endif.endif.endif.endif", !dbg !39
entry.endif.endif.endif.endif:
  %".39" = icmp eq i64 %".35", 0 , !dbg !39
  %".40" = and i1 %".32", %".39", !dbg !39
  %".41" = getelementptr [4 x i64], [4 x i64]* %".4", i32 0, i32 3 , !dbg !39
  %".42" = load i64, i64* %".41", !dbg !39
  %".43" = add i64 %".42", 1, !dbg !39
  br i1 %".40", label %"entry.endif.endif.endif.endif.if", label %"entry.endif.endif.endif.endif.endif", !dbg !39
entry.endif.endif.endif.endif.if:
  store i64 %".43", i64* %".41", !dbg !39
  br label %"entry.endif.endif.endif.endif.endif", !dbg !39
entry.endif.endif.endif.endif.endif:
  %".47" = icmp eq i64 %".43", 0 , !dbg !39
  %".48" = and i1 %".40", %".47", !dbg !39
  %".49" = load [4 x i64], [4 x i64]* %".4", !dbg !39
  %".50" = load [2 x i64], [2 x i64]* %".5", !dbg !39
  %".51" = extractvalue [2 x i64] %".50", 0 , !dbg !39
  %".52" = extractvalue [2 x i64] %".50", 1 , !dbg !39
  %".53" = extractvalue [4 x i64] %".49", 0 , !dbg !39
  %".54" = extractvalue [4 x i64] %".49", 1 , !dbg !39
  %".55" = extractvalue [4 x i64] %".49", 2 , !dbg !39
  %".56" = extractvalue [4 x i64] %".49", 3 , !dbg !39
  %".57" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".58" = zext i64 %".53" to i128 , !dbg !39
  %".59" = mul i128 %".57", %".58", !dbg !39
  %".60" = trunc i128 %".59" to i64 , !dbg !39
  %".61" = lshr i128 %".59", 64, !dbg !39
  %".62" = trunc i128 %".61" to i64 , !dbg !39
  %".63" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".64" = zext i64 %".55" to i128 , !dbg !39
  %".65" = mul i128 %".63", %".64", !dbg !39
  %".66" = trunc i128 %".65" to i64 , !dbg !39
  %".67" = lshr i128 %".65", 64, !dbg !39
  %".68" = trunc i128 %".67" to i64 , !dbg !39
  %".69" = xor i64 %".68", %".54", !dbg !39
  %".70" = xor i64 %".69", %".51", !dbg !39
  %".71" = xor i64 %".62", %".56", !dbg !39
  %".72" = xor i64 %".71", %".52", !dbg !39
  %".73" = insertvalue [4 x i64] %".49", i64 %".70", 0 , !dbg !39
  %".74" = insertvalue [4 x i64] %".73", i64 %".66", 1 , !dbg !39
  %".75" = insertvalue [4 x i64] %".74", i64 %".72", 2 , !dbg !39
  %".76" = insertvalue [4 x i64] %".75", i64 %".60", 3 , !dbg !39
  %".77" = add i64 %".51", 11400714819323198485, !dbg !39
  %".78" = add i64 %".52", 13503953896175478587, !dbg !39
  %".79" = insertvalue [2 x i64] %".50", i64 %".77", 0 , !dbg !39
  %".80" = insertvalue [2 x i64] %".79", i64 %".78", 1 , !dbg !39
  %".81" = extractvalue [2 x i64] %".80", 0 , !dbg !39
  %".82" = extractvalue [2 x i64] %".80", 1 , !dbg !39
  %".83" = extractvalue [4 x i64] %".76", 0 , !dbg !39
  %".84" = extractvalue [4 x i64] %".76", 1 , !dbg !39
  %".85" = extractvalue [4 x i64] %".76", 2 , !dbg !39
  %".86" = extractvalue [4 x i64] %".76", 3 , !dbg !39
  %".87" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".88" = zext i64 %".83" to i128 , !dbg !39
  %".89" = mul i128 %".87", %".88", !dbg !39
  %".90" = trunc i128 %".89" to i64 , !dbg !39
  %".91" = lshr i128 %".89", 64, !dbg !39
  %".92" = trunc i128 %".91" to i64 , !dbg !39
  %".93" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".94" = zext i64 %".85" to i128 , !dbg !39
  %".95" = mul i128 %".93", %".94", !dbg !39
  %".96" = trunc i128 %".95" to i64 , !dbg !39
  %".97" = lshr i128 %".95", 64, !dbg !39
  %".98" = trunc i128 %".97" to i64 , !dbg !39
  %".99" = xor i64 %".98", %".84", !dbg !39
  %".100" = xor i64 %".99", %".81", !dbg !39
  %".101" = xor i64 %".92", %".86", !dbg !39
  %".102" = xor i64 %".101", %".82", !dbg !39
  %".103" = insertvalue [4 x i64] %".76", i64 %".100", 0 , !dbg !39
  %".104" = insertvalue [4 x i64] %".103", i64 %".96", 1 , !dbg !39
  %".105" = insertvalue [4 x i64] %".104", i64 %".102", 2 , !dbg !39
  %".106" = insertvalue [4 x i64] %".105", i64 %".90", 3 , !dbg !39
  %".107" = add i64 %".81", 11400714819323198485, !dbg !39
  %".108" = add i64 %".82", 13503953896175478587, !dbg !39
  %".109" = insertvalue [2 x i64] %".80", i64 %".107", 0 , !dbg !39
  %".110" = insertvalue [2 x i64] %".109", i64 %".108", 1 , !dbg !39
  %".111" = extractvalue [2 x i64] %".110", 0 , !dbg !39
  %".112" = extractvalue [2 x i64] %".110", 1 , !dbg !39
  %".113" = extractvalue [4 x i64] %".106", 0 , !dbg !39
  %".114" = extractvalue [4 x i64] %".106", 1 , !dbg !39
  %".115" = extractvalue [4 x i64] %".106", 2 , !dbg !39
  %".116" = extractvalue [4 x i64] %".106", 3 , !dbg !39
  %".117" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".118" = zext i64 %".113" to i128 , !dbg !39
  %".119" = mul i128 %".117", %".118", !dbg !39
  %".120" = trunc i128 %".119" to i64 , !dbg !39
  %".121" = lshr i128 %".119", 64, !dbg !39
  %".122" = trunc i128 %".121" to i64 , !dbg !39
  %".123" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".124" = zext i64 %".115" to i128 , !dbg !39
  %".125" = mul i128 %".123", %".124", !dbg !39
  %".126" = trunc i128 %".125" to i64 , !dbg !39
  %".127" = lshr i128 %".125", 64, !dbg !39
  %".128" = trunc i128 %".127" to i64 , !dbg !39
  %".129" = xor i64 %".128", %".114", !dbg !39
  %".130" = xor i64 %".129", %".111", !dbg !39
  %".131" = xor i64 %".122", %".116", !dbg !39
  %".132" = xor i64 %".131", %".112", !dbg !39
  %".133" = insertvalue [4 x i64] %".106", i64 %".130", 0 , !dbg !39
  %".134" = insertvalue [4 x i64] %".133", i64 %".126", 1 , !dbg !39
  %".135" = insertvalue [4 x i64] %".134", i64 %".132", 2 , !dbg !39
  %".136" = insertvalue [4 x i64] %".135", i64 %".120", 3 , !dbg !39
  %".137" = add i64 %".111", 11400714819323198485, !dbg !39
  %".138" = add i64 %".112", 13503953896175478587, !dbg !39
  %".139" = insertvalue [2 x i64] %".110", i64 %".137", 0 , !dbg !39
  %".140" = insertvalue [2 x i64] %".139", i64 %".138", 1 , !dbg !39
  %".141" = extractvalue [2 x i64] %".140", 0 , !dbg !39
  %".142" = extractvalue [2 x i64] %".140", 1 , !dbg !39
  %".143" = extractvalue [4 x i64] %".136", 0 , !dbg !39
  %".144" = extractvalue [4 x i64] %".136", 1 , !dbg !39
  %".145" = extractvalue [4 x i64] %".136", 2 , !dbg !39
  %".146" = extractvalue [4 x i64] %".136", 3 , !dbg !39
  %".147" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".148" = zext i64 %".143" to i128 , !dbg !39
  %".149" = mul i128 %".147", %".148", !dbg !39
  %".150" = trunc i128 %".149" to i64 , !dbg !39
  %".151" = lshr i128 %".149", 64, !dbg !39
  %".152" = trunc i128 %".151" to i64 , !dbg !39
  %".153" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".154" = zext i64 %".145" to i128 , !dbg !39
  %".155" = mul i128 %".153", %".154", !dbg !39
  %".156" = trunc i128 %".155" to i64 , !dbg !39
  %".157" = lshr i128 %".155", 64, !dbg !39
  %".158" = trunc i128 %".157" to i64 , !dbg !39
  %".159" = xor i64 %".158", %".144", !dbg !39
  %".160" = xor i64 %".159", %".141", !dbg !39
  %".161" = xor i64 %".152", %".146", !dbg !39
  %".162" = xor i64 %".161", %".142", !dbg !39
  %".163" = insertvalue [4 x i64] %".136", i64 %".160", 0 , !dbg !39
  %".164" = insertvalue [4 x i64] %".163", i64 %".156", 1 , !dbg !39
  %".165" = insertvalue [4 x i64] %".164", i64 %".162", 2 , !dbg !39
  %".166" = insertvalue [4 x i64] %".165", i64 %".150", 3 , !dbg !39
  %".167" = add i64 %".141", 11400714819323198485, !dbg !39
  %".168" = add i64 %".142", 13503953896175478587, !dbg !39
  %".169" = insertvalue [2 x i64] %".140", i64 %".167", 0 , !dbg !39
  %".170" = insertvalue [2 x i64] %".169", i64 %".168", 1 , !dbg !39
  %".171" = extractvalue [2 x i64] %".170", 0 , !dbg !39
  %".172" = extractvalue [2 x i64] %".170", 1 , !dbg !39
  %".173" = extractvalue [4 x i64] %".166", 0 , !dbg !39
  %".174" = extractvalue [4 x i64] %".166", 1 , !dbg !39
  %".175" = extractvalue [4 x i64] %".166", 2 , !dbg !39
  %".176" = extractvalue [4 x i64] %".166", 3 , !dbg !39
  %".177" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".178" = zext i64 %".173" to i128 , !dbg !39
  %".179" = mul i128 %".177", %".178", !dbg !39
  %".180" = trunc i128 %".179" to i64 , !dbg !39
  %".181" = lshr i128 %".179", 64, !dbg !39
  %".182" = trunc i128 %".181" to i64 , !dbg !39
  %".183" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".184" = zext i64 %".175" to i128 , !dbg !39
  %".185" = mul i128 %".183", %".184", !dbg !39
  %".186" = trunc i128 %".185" to i64 , !dbg !39
  %".187" = lshr i128 %".185", 64, !dbg !39
  %".188" = trunc i128 %".187" to i64 , !dbg !39
  %".189" = xor i64 %".188", %".174", !dbg !39
  %".190" = xor i64 %".189", %".171", !dbg !39
  %".191" = xor i64 %".182", %".176", !dbg !39
  %".192" = xor i64 %".191", %".172", !dbg !39
  %".193" = insertvalue [4 x i64] %".166", i64 %".190", 0 , !dbg !39
  %".194" = insertvalue [4 x i64] %".193", i64 %".186", 1 , !dbg !39
  %".195" = insertvalue [4 x i64] %".194", i64 %".192", 2 , !dbg !39
  %".196" = insertvalue [4 x i64] %".195", i64 %".180", 3 , !dbg !39
  %".197" = add i64 %".171", 11400714819323198485, !dbg !39
  %".198" = add i64 %".172", 13503953896175478587, !dbg !39
  %".199" = insertvalue [2 x i64] %".170", i64 %".197", 0 , !dbg !39
  %".200" = insertvalue [2 x i64] %".199", i64 %".198", 1 , !dbg !39
  %".201" = extractvalue [2 x i64] %".200", 0 , !dbg !39
  %".202" = extractvalue [2 x i64] %".200", 1 , !dbg !39
  %".203" = extractvalue [4 x i64] %".196", 0 , !dbg !39
  %".204" = extractvalue [4 x i64] %".196", 1 , !dbg !39
  %".205" = extractvalue [4 x i64] %".196", 2 , !dbg !39
  %".206" = extractvalue [4 x i64] %".196", 3 , !dbg !39
  %".207" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".208" = zext i64 %".203" to i128 , !dbg !39
  %".209" = mul i128 %".207", %".208", !dbg !39
  %".210" = trunc i128 %".209" to i64 , !dbg !39
  %".211" = lshr i128 %".209", 64, !dbg !39
  %".212" = trunc i128 %".211" to i64 , !dbg !39
  %".213" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".214" = zext i64 %".205" to i128 , !dbg !39
  %".215" = mul i128 %".213", %".214", !dbg !39
  %".216" = trunc i128 %".215" to i64 , !dbg !39
  %".217" = lshr i128 %".215", 64, !dbg !39
  %".218" = trunc i128 %".217" to i64 , !dbg !39
  %".219" = xor i64 %".218", %".204", !dbg !39
  %".220" = xor i64 %".219", %".201", !dbg !39
  %".221" = xor i64 %".212", %".206", !dbg !39
  %".222" = xor i64 %".221", %".202", !dbg !39
  %".223" = insertvalue [4 x i64] %".196", i64 %".220", 0 , !dbg !39
  %".224" = insertvalue [4 x i64] %".223", i64 %".216", 1 , !dbg !39
  %".225" = insertvalue [4 x i64] %".224", i64 %".222", 2 , !dbg !39
  %".226" = insertvalue [4 x i64] %".225", i64 %".210", 3 , !dbg !39
  %".227" = add i64 %".201", 11400714819323198485, !dbg !39
  %".228" = add i64 %".202", 13503953896175478587, !dbg !39
  %".229" = insertvalue [2 x i64] %".200", i64 %".227", 0 , !dbg !39
  %".230" = insertvalue [2 x i64] %".229", i64 %".228", 1 , !dbg !39
  %".231" = extractvalue [2 x i64] %".230", 0 , !dbg !39
  %".232" = extractvalue [2 x i64] %".230", 1 , !dbg !39
  %".233" = extractvalue [4 x i64] %".226", 0 , !dbg !39
  %".234" = extractvalue [4 x i64] %".226", 1 , !dbg !39
  %".235" = extractvalue [4 x i64] %".226", 2 , !dbg !39
  %".236" = extractvalue [4 x i64] %".226", 3 , !dbg !39
  %".237" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".238" = zext i64 %".233" to i128 , !dbg !39
  %".239" = mul i128 %".237", %".238", !dbg !39
  %".240" = trunc i128 %".239" to i64 , !dbg !39
  %".241" = lshr i128 %".239", 64, !dbg !39
  %".242" = trunc i128 %".241" to i64 , !dbg !39
  %".243" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".244" = zext i64 %".235" to i128 , !dbg !39
  %".245" = mul i128 %".243", %".244", !dbg !39
  %".246" = trunc i128 %".245" to i64 , !dbg !39
  %".247" = lshr i128 %".245", 64, !dbg !39
  %".248" = trunc i128 %".247" to i64 , !dbg !39
  %".249" = xor i64 %".248", %".234", !dbg !39
  %".250" = xor i64 %".249", %".231", !dbg !39
  %".251" = xor i64 %".242", %".236", !dbg !39
  %".252" = xor i64 %".251", %".232", !dbg !39
  %".253" = insertvalue [4 x i64] %".226", i64 %".250", 0 , !dbg !39
  %".254" = insertvalue [4 x i64] %".253", i64 %".246", 1 , !dbg !39
  %".255" = insertvalue [4 x i64] %".254", i64 %".252", 2 , !dbg !39
  %".256" = insertvalue [4 x i64] %".255", i64 %".240", 3 , !dbg !39
  %".257" = add i64 %".231", 11400714819323198485, !dbg !39
  %".258" = add i64 %".232", 13503953896175478587, !dbg !39
  %".259" = insertvalue [2 x i64] %".230", i64 %".257", 0 , !dbg !39
  %".260" = insertvalue [2 x i64] %".259", i64 %".258", 1 , !dbg !39
  %".261" = extractvalue [2 x i64] %".260", 0 , !dbg !39
  %".262" = extractvalue [2 x i64] %".260", 1 , !dbg !39
  %".263" = extractvalue [4 x i64] %".256", 0 , !dbg !39
  %".264" = extractvalue [4 x i64] %".256", 1 , !dbg !39
  %".265" = extractvalue [4 x i64] %".256", 2 , !dbg !39
  %".266" = extractvalue [4 x i64] %".256", 3 , !dbg !39
  %".267" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".268" = zext i64 %".263" to i128 , !dbg !39
  %".269" = mul i128 %".267", %".268", !dbg !39
  %".270" = trunc i128 %".269" to i64 , !dbg !39
  %".271" = lshr i128 %".269", 64, !dbg !39
  %".272" = trunc i128 %".271" to i64 , !dbg !39
  %".273" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".274" = zext i64 %".265" to i128 , !dbg !39
  %".275" = mul i128 %".273", %".274", !dbg !39
  %".276" = trunc i128 %".275" to i64 , !dbg !39
  %".277" = lshr i128 %".275", 64, !dbg !39
  %".278" = trunc i128 %".277" to i64 , !dbg !39
  %".279" = xor i64 %".278", %".264", !dbg !39
  %".280" = xor i64 %".279", %".261", !dbg !39
  %".281" = xor i64 %".272", %".266", !dbg !39
  %".282" = xor i64 %".281", %".262", !dbg !39
  %".283" = insertvalue [4 x i64] %".256", i64 %".280", 0 , !dbg !39
  %".284" = insertvalue [4 x i64] %".283", i64 %".276", 1 , !dbg !39
  %".285" = insertvalue [4 x i64] %".284", i64 %".282", 2 , !dbg !39
  %".286" = insertvalue [4 x i64] %".285", i64 %".270", 3 , !dbg !39
  %".287" = add i64 %".261", 11400714819323198485, !dbg !39
  %".288" = add i64 %".262", 13503953896175478587, !dbg !39
  %".289" = insertvalue [2 x i64] %".260", i64 %".287", 0 , !dbg !39
  %".290" = insertvalue [2 x i64] %".289", i64 %".288", 1 , !dbg !39
  %".291" = extractvalue [2 x i64] %".290", 0 , !dbg !39
  %".292" = extractvalue [2 x i64] %".290", 1 , !dbg !39
  %".293" = extractvalue [4 x i64] %".286", 0 , !dbg !39
  %".294" = extractvalue [4 x i64] %".286", 1 , !dbg !39
  %".295" = extractvalue [4 x i64] %".286", 2 , !dbg !39
  %".296" = extractvalue [4 x i64] %".286", 3 , !dbg !39
  %".297" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".298" = zext i64 %".293" to i128 , !dbg !39
  %".299" = mul i128 %".297", %".298", !dbg !39
  %".300" = trunc i128 %".299" to i64 , !dbg !39
  %".301" = lshr i128 %".299", 64, !dbg !39
  %".302" = trunc i128 %".301" to i64 , !dbg !39
  %".303" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".304" = zext i64 %".295" to i128 , !dbg !39
  %".305" = mul i128 %".303", %".304", !dbg !39
  %".306" = trunc i128 %".305" to i64 , !dbg !39
  %".307" = lshr i128 %".305", 64, !dbg !39
  %".308" = trunc i128 %".307" to i64 , !dbg !39
  %".309" = xor i64 %".308", %".294", !dbg !39
  %".310" = xor i64 %".309", %".291", !dbg !39
  %".311" = xor i64 %".302", %".296", !dbg !39
  %".312" = xor i64 %".311", %".292", !dbg !39
  %".313" = insertvalue [4 x i64] %".286", i64 %".310", 0 , !dbg !39
  %".314" = insertvalue [4 x i64] %".313", i64 %".306", 1 , !dbg !39
  %".315" = insertvalue [4 x i64] %".314", i64 %".312", 2 , !dbg !39
  %".316" = insertvalue [4 x i64] %".315", i64 %".300", 3 , !dbg !39
  %".317" = add i64 %".291", 11400714819323198485, !dbg !39
  %".318" = add i64 %".292", 13503953896175478587, !dbg !39
  %".319" = insertvalue [2 x i64] %".290", i64 %".317", 0 , !dbg !39
  %".320" = insertvalue [2 x i64] %".319", i64 %".318", 1 , !dbg !39
  %".321" = extractvalue [2 x i64] %".320", 0 , !dbg !39
  %".322" = extractvalue [2 x i64] %".320", 1 , !dbg !39
  %".323" = extractvalue [4 x i64] %".316", 0 , !dbg !39
  %".324" = extractvalue [4 x i64] %".316", 1 , !dbg !39
  %".325" = extractvalue [4 x i64] %".316", 2 , !dbg !39
  %".326" = extractvalue [4 x i64] %".316", 3 , !dbg !39
  %".327" = zext i64 15197193596820024467 to i128 , !dbg !39
  %".328" = zext i64 %".323" to i128 , !dbg !39
  %".329" = mul i128 %".327", %".328", !dbg !39
  %".330" = trunc i128 %".329" to i64 , !dbg !39
  %".331" = lshr i128 %".329", 64, !dbg !39
  %".332" = trunc i128 %".331" to i64 , !dbg !39
  %".333" = zext i64 14581110107779764567 to i128 , !dbg !39
  %".334" = zext i64 %".325" to i128 , !dbg !39
  %".335" = mul i128 %".333", %".334", !dbg !39
  %".336" = trunc i128 %".335" to i64 , !dbg !39
  %".337" = lshr i128 %".335", 64, !dbg !39
  %".338" = trunc i128 %".337" to i64 , !dbg !39
  %".339" = xor i64 %".338", %".324", !dbg !39
  %".340" = xor i64 %".339", %".321", !dbg !39
  %".341" = xor i64 %".332", %".326", !dbg !39
  %".342" = xor i64 %".341", %".322", !dbg !39
  %".343" = insertvalue [4 x i64] %".316", i64 %".340", 0 , !dbg !39
  %".344" = insertvalue [4 x i64] %".343", i64 %".336", 1 , !dbg !39
  %".345" = insertvalue [4 x i64] %".344", i64 %".342", 2 , !dbg !39
  %".346" = insertvalue [4 x i64] %".345", i64 %".330", 3 , !dbg !39
  %".347" = add i64 %".321", 11400714819323198485, !dbg !39
  %".348" = add i64 %".322", 13503953896175478587, !dbg !39
  %".349" = insertvalue [2 x i64] %".320", i64 %".347", 0 , !dbg !39
  %".350" = insertvalue [2 x i64] %".349", i64 %".348", 1 , !dbg !39
  store [4 x i64] %".346", [4 x i64]* %".6", !dbg !39
  store i16 1, i16* %".7", !dbg !39
  %".353" = extractvalue [4 x i64] %".346", 0 , !dbg !39
  store i64 %".353", i64* %".2", !dbg !39
  ret void, !dbg !39
}

define void @"__pnl_builtin_philox_rand_double"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* noalias nonnull %".1", double* noalias nonnull %".2") argmemonly!dbg !41
{
entry:
  %"rand_int64" = alloca i64, !dbg !42
  call void @"__pnl_builtin_philox_rand_int64"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i64* %"rand_int64"), !dbg !42
  %".5" = load i64, i64* %"rand_int64", !dbg !42
  %".6" = lshr i64 %".5", 11, !dbg !42
  %".7" = uitofp i64 %".6" to double , !dbg !42
  %".8" = fmul double %".7", 0x3ca0000000000000, !dbg !42
  store double %".8", double* %".2", !dbg !42
  ret void, !dbg !42
}

define void @"__pnl_builtin_philox_rand_normal"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* noalias nonnull %".1", double* noalias nonnull %".2") argmemonly!dbg !44
{
entry:
  %"tmp_fp" = alloca double, !dbg !45
  %"tmp_int" = alloca i64, !dbg !45
  br label %"gen_loop_ziggurat", !dbg !45
gen_loop_ziggurat:
  call void @"__pnl_builtin_philox_rand_int64"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i64* %"tmp_int"), !dbg !45
  %".6" = load i64, i64* %"tmp_int", !dbg !45
  %".7" = and i64 %".6", 255, !dbg !45
  %".8" = lshr i64 %".6", 8, !dbg !45
  %".9" = and i64 %".8", 1, !dbg !45
  %".10" = lshr i64 %".8", 1, !dbg !45
  %".11" = and i64 %".10", 4503599627370495, !dbg !45
  %".12" = uitofp i64 %".11" to double , !dbg !45
  %".13" = getelementptr [256 x double], [256 x double]* @"__pnl_builtin_wi_double", i64 0, i64 %".7" , !dbg !45
  %".14" = load double, double* %".13", !dbg !45
  %".15" = fmul double %".12", %".14", !dbg !45
  %".16" = fsub double 0x8000000000000000, %".15", !dbg !45
  %".17" = trunc i64 %".9" to i1 , !dbg !45
  %".18" = select i1 %".17", double %".16", double %".15" , !dbg !45
  %".19" = getelementptr [256 x i64], [256 x i64]* @"__pnl_builtin_ki_i64", i64 0, i64 %".7" , !dbg !45
  %".20" = load i64, i64* %".19", !dbg !45
  %".21" = icmp ult i64 %".11", %".20" , !dbg !45
  br i1 %".21", label %"gen_loop_ziggurat.if", label %"gen_loop_ziggurat.endif", !dbg !45, !prof !20
gen_loop_ziggurat.if:
  store double %".18", double* %".2", !dbg !45
  ret void, !dbg !45
gen_loop_ziggurat.endif:
  %".25" = icmp eq i64 0, %".7" , !dbg !45
  br i1 %".25", label %"gen_loop_ziggurat.endif.if", label %"gen_loop_ziggurat.endif.endif", !dbg !45
gen_loop_ziggurat.endif.if:
  call void @"__pnl_builtin_philox_rand_double"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", double* %"tmp_fp"), !dbg !45
  %".28" = load double, double* %"tmp_fp", !dbg !45
  %".29" = fsub double 0x8000000000000000, %".28", !dbg !45
  %".30" = fadd double %".29", 0x3ff0000000000000, !dbg !45
  %".31" = call double @"__pnl_builtin_log"(double %".30"), !dbg !45
  %".32" = fmul double 0xbfd183aa6c20e8c1, %".31", !dbg !45
  call void @"__pnl_builtin_philox_rand_double"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", double* %"tmp_fp"), !dbg !45
  %".34" = load double, double* %"tmp_fp", !dbg !45
  %".35" = fsub double 0x8000000000000000, %".34", !dbg !45
  %".36" = fadd double %".35", 0x3ff0000000000000, !dbg !45
  %".37" = call double @"__pnl_builtin_log"(double %".36"), !dbg !45
  %".38" = fsub double 0x8000000000000000, %".37", !dbg !45
  %".39" = fadd double %".38", %".38", !dbg !45
  %".40" = fmul double %".32", %".32", !dbg !45
  %".41" = fcmp ogt double %".39", %".40" , !dbg !45
  br i1 %".41", label %"gen_loop_ziggurat.endif.if.if", label %"gen_loop_ziggurat.endif.if.endif", !dbg !45
gen_loop_ziggurat.endif.endif:
  %".51" = getelementptr [256 x double], [256 x double]* @"__pnl_builtin_fi_double", i64 0, i64 %".7" , !dbg !45
  %".52" = load double, double* %".51", !dbg !45
  %".53" = sub i64 %".7", 1, !dbg !45
  %".54" = getelementptr [256 x double], [256 x double]* @"__pnl_builtin_fi_double", i64 0, i64 %".53" , !dbg !45
  %".55" = load double, double* %".54", !dbg !45
  %".56" = fmul double %".18", %".18", !dbg !45
  %".57" = fmul double %".56", 0xbfe0000000000000, !dbg !45
  %".58" = call double @"__pnl_builtin_exp"(double %".57"), !dbg !45
  call void @"__pnl_builtin_philox_rand_double"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", double* %"tmp_fp"), !dbg !45
  %".60" = load double, double* %"tmp_fp", !dbg !45
  %".61" = fsub double %".55", %".52", !dbg !45
  %".62" = fmul double %".61", %".60", !dbg !45
  %".63" = fadd double %".62", %".52", !dbg !45
  %".64" = fcmp olt double %".63", %".58" , !dbg !45
  br i1 %".64", label %"gen_loop_ziggurat.endif.endif.if", label %"gen_loop_ziggurat.endif.endif.endif", !dbg !45
gen_loop_ziggurat.endif.if.if:
  %".43" = fadd double 0x400d3bb48209ad33, %".32", !dbg !45
  %".44" = fsub double 0x8000000000000000, %".43", !dbg !45
  %".45" = lshr i64 %".11", 8, !dbg !45
  %".46" = trunc i64 %".45" to i1 , !dbg !45
  %".47" = select i1 %".46", double %".44", double %".43" , !dbg !45
  store double %".47", double* %".2", !dbg !45
  ret void, !dbg !45
gen_loop_ziggurat.endif.if.endif:
  br label %"gen_loop_ziggurat.endif.if", !dbg !45
gen_loop_ziggurat.endif.endif.if:
  store double %".18", double* %".2", !dbg !45
  ret void, !dbg !45
gen_loop_ziggurat.endif.endif.endif:
  br label %"gen_loop_ziggurat", !dbg !45
}

@"__pnl_builtin_wi_double" = internal constant [256 x double] [double 0x3ccf493b7815d979, double 0x3c8b8d0be3fdf6c6, double 0x3c9250af3c2c5bb4, double 0x3c957cb938443b61, double 0x3c9801fce82fa70c, double 0x3c9a230c2e4cd0bc, double 0x3c9c004d2f3861f7, double 0x3c9dac2f5a747274, double 0x3c9f32482d4cd5c3, double 0x3ca04d32278ebbad, double 0x3ca0f5053b025d43, double 0x3ca192a697413677, double 0x3ca227a28f7a1af5, double 0x3ca2b52e3863d880, double 0x3ca33c3fc05791f5, double 0x3ca3bd9ec1a2b12f, double 0x3ca439ef8dff9b55, double 0x3ca4b1bb363dfea7, double 0x3ca52575621ad374, double 0x3ca59580a707ce96, double 0x3ca60231cfd97eea, double 0x3ca66bd261a37c3d, double 0x3ca6d2a292000570, double 0x3ca736dad346f8a6, double 0x3ca798ad10b32a77, double 0x3ca7f845ad46f543, double 0x3ca855cc53430a77, double 0x3ca8b1649e7b769a, double 0x3ca90b2ea94ecf98, double 0x3ca96347822c1eea, double 0x3ca9b9c98e38c546, double 0x3caa0eccdca4a72c, double 0x3caa62676d77cd59, double 0x3caab4ad6e101630, double 0x3cab05b16d136c9c, double 0x3cab558487427a29, double 0x3caba4368e529f3a, double 0x3cabf1d62abf8232, double 0x3cac3e70f9594ef3, double 0x3cac8a13a5323b61, double 0x3cacd4c9fe72268b, double 0x3cad1e9f0e80b748, double 0x3cad679d29e41f10, double 0x3cadafce0023b8c3, double 0x3cadf73aa9f17653, double 0x3cae3debb5d2edfe, double 0x3cae83e9337a6f00, double 0x3caec93abdf982ce, double 0x3caf0de784f06226, double 0x3caf51f654d8f688, double 0x3caf956d9e87d7ae, double 0x3cafd8537dfa2eac, double 0x3cb00d56e04234ec, double 0x3cb02e40f5398f9a, double 0x3cb04eea9e16a5fc, double 0x3cb06f565b72a010, double 0x3cb08f869071f40b, double 0x3cb0af7d84bc6113, double 0x3cb0cf3d664bcc7f, double 0x3cb0eec84b16086b, double 0x3cb10e20329515ee, double 0x3cb12d4707310fbe, double 0x3cb14c3e9f8e9141, double 0x3cb16b08bfc4201e, double 0x3cb189a71a78da34, double 0x3cb1a81b51ee6d88, double 0x3cb1c666f8f82acb, double 0x3cb1e48b93e0d42e, double 0x3cb2028a9940a09f, double 0x3cb2206572c4c6e9, double 0x3cb23e1d7de9c31f, double 0x3cb25bb40ca96bfb, double 0x3cb2792a661dd37f, double 0x3cb29681c719d71b, double 0x3cb2b3bb62b82eda, double 0x3cb2d0d862e1b853, double 0x3cb2edd9e8cba98e, double 0x3cb30ac10d6e48d7, double 0x3cb3278ee1f4b930, double 0x3cb3444470265ea1, double 0x3cb360e2baca52d5, double 0x3cb37d6abe05586a, double 0x3cb399dd6fb2b264, double 0x3cb3b63bbfb83d03, double 0x3cb3d28698561de0, double 0x3cb3eebede725a83, double 0x3cb40ae571e09e74, double 0x3cb426fb2da6745d, double 0x3cb44300e83c30a4, double 0x3cb45ef773cac75d, double 0x3cb47adf9e66c336, double 0x3cb496ba32488f2f, double 0x3cb4b287f602415d, double 0x3cb4ce49acb311dc, double 0x3cb4ea001638a605, double 0x3cb505abef5e5562, double 0x3cb5214df20a8b5a, double 0x3cb53ce6d56a664f, double 0x3cb558774e1bb2c8, double 0x3cb574000e555f78, double 0x3cb58f81c60e8514, double 0x3cb5aafd23241b59, double 0x3cb5c672d17d733d, double 0x3cb5e1e37b2f8cd3, double 0x3cb5fd4fc89f5e38, double 0x3cb618b860a31fc3, double 0x3cb6341de8a2b0a2, double 0x3cb64f8104b7260b, double 0x3cb66ae257c99672, double 0x3cb6864283b13137, double 0x3cb6a1a22950b2b1, double 0x3cb6bd01e8b343bb, double 0x3cb6d8626128d352, double 0x3cb6f3c43161f854, double 0x3cb70f27f78b68eb, double 0x3cb72a8e516914c6, double 0x3cb745f7dc70eedc, double 0x3cb7616535e5731f, double 0x3cb77cd6faeff449, double 0x3cb7984dc8babd93, double 0x3cb7b3ca3c8b1409, double 0x3cb7cf4cf3db22fb, double 0x3cb7ead68c73dee7, double 0x3cb80667a486ea1f, double 0x3cb82200dac88676, double 0x3cb83da2ce899f15, double 0x3cb8594e1fd1f5bd, double 0x3cb875036f7a7ec5, double 0x3cb890c35f47f72d, double 0x3cb8ac8e9205c043, double 0x3cb8c865aba10c9c, double 0x3cb8e44951446a27, double 0x3cb9003a2973b58f, double 0x3cb91c38dc288347, double 0x3cb9384612ef0afc, double 0x3cb954627903a28a, double 0x3cb9708ebb70d5ee, double 0x3cb98ccb892e2a31, double 0x3cb9a919933f99bf, double 0x3cb9c5798cd5d92c, double 0x3cb9e1ec2b6f7411, double 0x3cb9fe7226fad24a, double 0x3cba1b0c39f93692, double 0x3cba37bb21a2c85b, double 0x3cba547f9e0bbb88, double 0x3cba715a724aa9a4, double 0x3cba8e4c64a0313d, double 0x3cbaab563e9ff108, double 0x3cbac878cd5af5ce, double 0x3cbae5b4e18bb336, double 0x3cbb030b4fc3a11a, double 0x3cbb207cf09a985b, double 0x3cbb3e0aa0e00c00, double 0x3cbb5bb541ce3d03, double 0x3cbb797db93f8927, double 0x3cbb9764f1e5f73c, double 0x3cbbb56bdb85256e, double 0x3cbbd3936b2ec0a2, double 0x3cbbf1dc9b81ae83, double 0x3cbc10486cec16a0, double 0x3cbc2ed7e5f07a2d, double 0x3cbc4d8c136e0d1c, double 0x3cbc6c6608ec8705, double 0x3cbc8b66e0eba617, double 0x3cbcaa8fbd36a2ab, double 0x3cbcc9e1c73bd690, double 0x3cbce95e3068e037, double 0x3cbd0906328b8f6e, double 0x3cbd28db1037ef20, double 0x3cbd48de1533c647, double 0x3cbd691096e7f123, double 0x3cbd8973f4d7fba5, double 0x3cbdaa0999206e70, double 0x3cbdcad2f8fc490e, double 0x3cbdebd195522e37, double 0x3cbe0d06fb49d21c, double 0x3cbe2e74c4ea46f6, double 0x3cbe501c99c1d188, double 0x3cbe72002f97fe25, double 0x3cbe94214b2abf0a, double 0x3cbeb681c0f76f08, double 0x3cbed9237610a73a, double 0x3cbefc086101eca9, double 0x3cbf1f328ac25321, double 0x3cbf42a40fb74d6d, double 0x3cbf665f20c90168, double 0x3cbf8a6604899782, double 0x3cbfaebb187122bf, double 0x3cbfd360d22fe785, double 0x3cbff859c118f60b, double 0x3cc00ed447d3a075, double 0x3cc021a8028fc947, double 0x3cc034a983a902ab, double 0x3cc047da4e3ef5c7, double 0x3cc05b3bf6adb37e, double 0x3cc06ed023a72668, double 0x3cc082988f632e17, double 0x3cc0969708e8a254, double 0x3cc0aacd7571c0c4, double 0x3cc0bf3dd1eed448, double 0x3cc0d3ea34aa3d30, double 0x3cc0e8d4cf116593, double 0x3cc0fdffefa69fb6, double 0x3cc1136e04207041, double 0x3cc129219bbb5d35, double 0x3cc13f1d69c4096d, double 0x3cc1556448602e3b, double 0x3cc16bf93b9deef3, double 0x3cc182df74d21261, double 0x3cc19a1a564eebac, double 0x3cc1b1ad777f2f8e, double 0x3cc1c99ca971a694, double 0x3cc1e1ebfbe4ae39, double 0x3cc1fa9fc2e2d901, double 0x3cc213bc9d04cc81, double 0x3cc22d477a6fd3ee, double 0x3cc24745a4ac9c24, double 0x3cc261bcc77658e0, double 0x3cc27cb2faa8592e, double 0x3cc2982ecd770e78, double 0x3cc2b437532a0a52, double 0x3cc2d0d43196db97, double 0x3cc2ee0db1a978f5, double 0x3cc30becd256aeee, double 0x3cc32a7b5e68a4a3, double 0x3cc349c405ae12a3, double 0x3cc369d27a33a840, double 0x3cc38ab39256410a, double 0x3cc3ac7570ae88fa, double 0x3cc3cf27b31704a6, double 0x3cc3f2dbaa60f475, double 0x3cc417a49cb9e5da, double 0x3cc43d9815545e94, double 0x3cc464ce44a73a15, double 0x3cc48d62759c43bc, double 0x3cc4b7739d6b5a27, double 0x3cc4e3250dcd8902, double 0x3cc5109f53e9ac41, double 0x3cc54011523a7e42, double 0x3cc571b1a94ae41b, double 0x3cc5a5c08b718dd9, double 0x3cc5dc8a243ad0fe, double 0x3cc61669cf861e4c, double 0x3cc653ce7b006aea, double 0x3cc69540be9fe5c3, double 0x3cc6db6b8d09e232, double 0x3cc72728f05f7a34, double 0x3cc7799556090673, double 0x3cc7d42df4d6ce8c, double 0x3cc839030529f234, double 0x3cc8ab0fbfaa7c14, double 0x3cc92ee0946f4496, double 0x3cc9cbee014057ab, double 0x3cca8fdc7894775a, double 0x3ccb981f3878fdb1, double 0x3ccd3bb48209ad33]
@"__pnl_builtin_ki_i64" = internal constant [256 x i64] [i64 4208095142473578, i64 0, i64 3387314423973544, i64 3838760076542274, i64 4030768804392682, i64 4136731738896254, i64 4203757248105145, i64 4249917568205994, i64 4283617341590296, i64 4309289223136604, i64 4329489775174550, i64 4345795907393188, i64 4359232558744730, i64 4370494503737299, i64 4380069246215646, i64 4388308869042394, i64 4395473957549321, i64 4401761481783924, i64 4407323076021240, i64 4412277362218204, i64 4416718463613199, i64 4420722014516422, i64 4424349484777079, i64 4427651345409294, i64 4430669422005229, i64 4433438668975191, i64 4435988524278344, i64 4438343955930065, i64 4440526279077425, i64 4442553800234660, i64 4444442329865861, i64 4446205593658138, i64 4447855565093316, i64 4449402736340121, i64 4450856340408624, i64 4452224534496486, i64 4453514552210512, i64 4454732830656798, i64 4455885117109368, i64 4456976558985043, i64 4458011780094444, i64 4458994945550386, i64 4459929817254120, i64 4460819801517196, i64 4461667990089170, i64 4462477195632268, i64 4463249982500384, i64 4463988693531856, i64 4464695473445501, i64 4465372289331869, i64 4466020948651920, i64 4466643115089764, i64 4467240322552142, i64 4467813987562542, i64 4468365420260672, i64 4468895834186994, i64 4469406355006040, i64 4469898028300364, i64 4470371826548633, i64 4470828655385770, i64 4471269359229841, i64 4471694726349190, i64 4472105493433674, i64 4472502349725738, i64 4472885940759935, i64 4473256871753524, i64 4473615710685532, i64 4473962991097124, i64 4474299214642296, i64 4474624853414418, i64 4474940352071305, i64 4475246129778808, i64 4475542581990776, i64 4475830082081194, i64 4476108982842610, i64 4476379617863426, i64 4476642302795321, i64 4476897336520866, i64 4477145002230339, i64 4477385568415884, i64 4477619289790266, i64 4477846408136804, i64 4478067153096380, i64 4478281742896886, i64 4478490385029917, i64 4478693276879082, i64 4478890606303906, i64 4479082552182886, i64 4479269284918997, i64 4479450966910588, i64 4479627752990372, i64 4479799790834988, i64 4479967221347354, i64 4480130179013872, i64 4480288792238368, i64 4480443183654460, i64 4480593470417939, i64 4480739764480586, i64 4480882172846772, i64 4481020797814010, i64 4481155737198612, i64 4481287084547452, i64 4481414929336784, i64 4481539357158974, i64 4481660449897960, i64 4481778285894165, i64 4481892940099539, i64 4482004484223382, i64 4482112986869492, i64 4482218513665204, i64 4482321127382802, i64 4482420888053758, i64 4482517853076245, i64 4482612077316275, i64 4482703613202871, i64 4482792510817576, i64 4482878817978627, i64 4482962580320076, i64 4483043841366126, i64 4483122642600925, i64 4483199023534056, i64 4483273021761922, i64 4483344673025224, i64 4483414011262724, i64 4483481068661428, i64 4483545875703378, i64 4483608461209170, i64 4483668852378323, i64 4483727074826624, i64 4483783152620564, i64 4483837108308932, i64 4483888962951686, i64 4483938736146144, i64 4483986446050596, i64 4484032109405372, i64 4484075741551420, i64 4484117356446452, i64 4484156966678662, i64 4484194583478081, i64 4484230216725550, i64 4484263874959345, i64 4484295565379450, i64 4484325293849474, i64 4484353064896186, i64 4484378881706674, i64 4484402746123075, i64 4484424658634833, i64 4484444618368474, i64 4484462623074794, i64 4484478669113436, i64 4484492751434740, i64 4484504863558830, i64 4484514997551788, i64 4484523143998833, i64 4484529291974394, i64 4484533429008906, i64 4484535541052219, i64 4484535612433424, i64 4484533625816926, i64 4484529562154580, i64 4484523400633636, i64 4484515118620291, i64 4484504691598554, i64 4484492093104164, i64 4484477294653230, i64 4484460265665252, i64 4484440973380154, i64 4484419382768918, i64 4484395456437370, i64 4484369154522621, i64 4484340434581640, i64 4484309251471359, i64 4484275557219678, i64 4484239300886654, i64 4484200428415112, i64 4484158882469814, i64 4484114602264271, i64 4484067523374160, i64 4484017577536216, i64 4483964692431365, i64 4483908791450714, i64 4483849793442887, i64 4483787612441036, i64 4483722157367660, i64 4483653331715198, i64 4483581033200083, i64 4483505153387764, i64 4483425577285833, i64 4483342182902157, i64 4483254840764470, i64 4483163413397547, i64 4483067754753536, i64 4482967709590562, i64 4482863112794072, i64 4482753788634692, i64 4482639549955636, i64 4482520197281720, i64 4482395517841076, i64 4482265284489409, i64 4482129254525304, i64 4481987168383486, i64 4481838748191074, i64 4481683696169781, i64 4481521692864464, i64 4481352395175570, i64 4481175434169564, i64 4480990412637506, i64 4480796902367134, i64 4480594441088331, i64 4480382529045225, i64 4480160625140311, i64 4479928142586662, i64 4479684443993061, i64 4479428835793398, i64 4479160561915451, i64 4478878796564388, i64 4478582635972392, i64 4478271088936406, i64 4477943065929958, i64 4477597366530538, i64 4477232664848704, i64 4476847492576192, i64 4476440219183781, i64 4476009028690434, i64 4475551892286424, i64 4475066535915646, i64 4474550401693506, i64 4474000601739904, i64 4473413862618200, i64 4472786458058295, i64 4472114126959004, i64 4471391972746494, i64 4470614338917719, i64 4469774653883156, i64 4468865235838896, i64 4467877045039530, i64 4466799366045354, i64 4465619395558397, i64 4464321701199635, i64 4462887501169282, i64 4461293691124341, i64 4459511507635972, i64 4457504658253067, i64 4455226650325010, i64 4452616884242348, i64 4449594783440798, i64 4446050695647666, i64 4441831266659618, i64 4436714892174061, i64 4430368316897338, i64 4422264825074740, i64 4411517007702132, i64 4396496531309976, i64 4373832704204284, i64 4335125104963628, i64 4251099761679434]
@"__pnl_builtin_fi_double" = internal constant [256 x double] [double 0x3ff0000000000000, double 0x3fef446ac979f087, double 0x3feeb7545b6ca915, double 0x3fee3f11e027f077, double 0x3fedd36fa704de95, double 0x3fed70920657bcf2, double 0x3fed144978a119dc, double 0x3fecbd33a8a72deb, double 0x3fec6a5ecea9787f, double 0x3fec1b1cd9eebaea, double 0x3febceeb4ee1dc82, double 0x3feb85653a8ff552, double 0x3feb3e3a8234dd10, double 0x3feaf92a3f6ce8a2, double 0x3feab5fef17a2504, double 0x3fea748bd550c9e1, double 0x3fea34aafdf5af0f, double 0x3fe9f63bee651fd8, double 0x3fe9b9228d240681, double 0x3fe97d4657617ac1, double 0x3fe94291c21b7a47, double 0x3fe908f1bd31714f, double 0x3fe8d0554fe60aa8, double 0x3fe898ad48badf02, double 0x3fe861ebfc37bcac, double 0x3fe82c050f56cf6e, double 0x3fe7f6ed4b20e2cb, double 0x3fe7c29a779c6858, double 0x3fe78f033ca0b0d5, double 0x3fe75c1f0770d856, double 0x3fe729e5f43f6d12, double 0x3fe6f850baea7aee, double 0x3fe6c7589e635a89, double 0x3fe696f75e513b2a, double 0x3fe667272a92e323, double 0x3fe637e298550c18, double 0x3fe6092498802665, double 0x3fe5dae86f4aff6a, double 0x3fe5ad29acc85c89, double 0x3fe57fe4264c8d8f, double 0x3fe55313f08d9e46, double 0x3fe526b55a656cd5, double 0x3fe4fac4e820b667, double 0x3fe4cf3f4f494ec0, double 0x3fe4a42172dc5278, double 0x3fe479685fdf5012, double 0x3fe44f114a493679, double 0x3fe425198a355fe3, double 0x3fe3fb7e99585b82, double 0x3fe3d23e10af31a3, double 0x3fe3a955a662cd0e, double 0x3fe380c32bda00d5, double 0x3fe358848bf550e9, double 0x3fe33097c9703a35, double 0x3fe308fafd6438ef, double 0x3fe2e1ac55ea3bee, double 0x3fe2baaa14d7954a, double 0x3fe293f28e93cd15, double 0x3fe26d84290504ed, double 0x3fe2475d5a90db84, double 0x3fe2217ca92ff7f2, double 0x3fe1fbe0a9929620, double 0x3fe1d687fe549969, double 0x3fe1b171573fd111, double 0x3fe18c9b709b3c50, double 0x3fe16805128639da, double 0x3fe143ad105ea99c, double 0x3fe11f9248311f38, double 0x3fe0fbb3a2325913, double 0x3fe0d810104142a0, double 0x3fe0b4a68d70d9ae, double 0x3fe091761d995d81, double 0x3fe06e7dccf03c36, double 0x3fe04bbcafa63f2e, double 0x3fe02931e18b822a, double 0x3fe006dc85b8cac4, double 0x3fdfc9778c7bbda1, double 0x3fdf859da7a900ca, double 0x3fdf4229cb2f7af3, double 0x3fdeff1a717e8f95, double 0x3fdebc6e20bd1f54, double 0x3fde7a236a4ec3c5, double 0x3fde3838ea5f9b85, double 0x3fddf6ad47763a09, double 0x3fddb57f320b56b1, double 0x3fdd74ad6426de33, double 0x3fdd3436a1021080, double 0x3fdcf419b4ae5b6d, double 0x3fdcb45573c0a848, double 0x3fdc74e8bb00d7c7, double 0x3fdc35d26f1d2cb8, double 0x3fdbf7117c616a17, double 0x3fdbb8a4d6716d91, double 0x3fdb7a8b7807131b, double 0x3fdb3cc462b331ca, double 0x3fdaff4e9ea18552, double 0x3fdac2293a5f5a9e, double 0x3fda85534aa4d880, double 0x3fda48cbea20c04d, double 0x3fda0c923946843e, double 0x3fd9d0a55e1e93df, double 0x3fd995048418c0c6, double 0x3fd959aedbe09f93, double 0x3fd91ea39b33cb17, double 0x3fd8e3e1fcb9f115, double 0x3fd8a9693fde9188, double 0x3fd86f38a8ac5ab6, double 0x3fd8354f7faa0dd9, double 0x3fd7fbad11b8d911, double 0x3fd7c250aff414b0, double 0x3fd78939af9252eb, double 0x3fd7506769c7b1ed, double 0x3fd717d93ba9614c, double 0x3fd6df8e86124caa, double 0x3fd6a786ad88de21, double 0x3fd66fc11a25cbe2, double 0x3fd6383d377be515, double 0x3fd600fa7480d2c8, double 0x3fd5c9f84376c244, double 0x3fd5933619d6eebe, double 0x3fd55cb3703d0100, double 0x3fd5266fc2533bed, double 0x3fd4f06a8ebf6d92, double 0x3fd4baa357109ca2, double 0x3fd485199fad6ad4, double 0x3fd44fccefc324fe, double 0x3fd41abcd1357a19, double 0x3fd3e5e8d08ed2db, double 0x3fd3b1507cf143ae, double 0x3fd37cf368081379, double 0x3fd348d125f9d19e, double 0x3fd314e94d5af62f, double 0x3fd2e13b77210766, double 0x3fd2adc73e963fdd, double 0x3fd27a8c414db11e, double 0x3fd2478a1f17de89, double 0x3fd214c079f7cc9e, double 0x3fd1e22ef6188116, double 0x3fd1afd539c2f050, double 0x3fd17db2ed5454e8, double 0x3fd14bc7bb34ee67, double 0x3fd11a134fcf2423, double 0x3fd0e895598709c4, double 0x3fd0b74d88b242da, double 0x3fd0863b8f904336, double 0x3fd0555f2242e9d9, double 0x3fd024b7f6c7747e, double 0x3fcfe88b89df93c5, double 0x3fcf88108cb83235, double 0x3fcf27fe6ce998d2, double 0x3fcec854a4c99c44, double 0x3fce6912b2283cdd, double 0x3fce0a3816457184, double 0x3fcdabc455c7900a, double 0x3fcd4db6f8b2514f, double 0x3fccf00f8a5e6fcc, double 0x3fcc92cd9971df53, double 0x3fcc35f0b7d89d47, double 0x3fcbd9787abe18a1, double 0x3fcb7d647a8731aa, double 0x3fcb21b452ccd13a, double 0x3fcac667a2571807, double 0x3fca6b7e0b19267e, double 0x3fca10f7322d7e3d, double 0x3fc9b6d2bfd2fe5a, double 0x3fc95d105f6a7c27, double 0x3fc903afbf74fa69, double 0x3fc8aab09192815b, double 0x3fc852128a819a38, double 0x3fc7f9d5621f7175, double 0x3fc7a1f8d368a323, double 0x3fc74a7c9c7ab5a6, double 0x3fc6f3607e964716, double 0x3fc69ca43e21f25c, double 0x3fc64647a2adf19c, double 0x3fc5f04a76f883f9, double 0x3fc59aac88f31d6c, double 0x3fc5456da9c86835, double 0x3fc4f08dade31fc1, double 0x3fc49c0c6cf5ce2d, double 0x3fc447e9c20375d5, double 0x3fc3f4258b6931ae, double 0x3fc3a0bfaae8d7ee, double 0x3fc34db805b4ab88, double 0x3fc2fb0e847c2a65, double 0x3fc2a8c3137a071a, double 0x3fc256d5a2835eb7, double 0x3fc2054625183c34, double 0x3fc1b41492757d42, double 0x3fc16340e5a82d63, double 0x3fc112cb1da26eb9, double 0x3fc0c2b33d5209ba, double 0x3fc072f94bb8bf85, double 0x3fc0239d54067d2a, double 0x3fbfa93ecb6b222c, double 0x3fbf0bff29520e1c, double 0x3fbe6f7bf29aa54b, double 0x3fbdd3b56176e88f, double 0x3fbd38abb9bd91e5, double 0x3fbc9e5f493b740a, double 0x3fbc04d0680b1015, double 0x3fbb6bff78f2e233, double 0x3fbad3ece9caf633, double 0x3fba3c9933ea6286, double 0x3fb9a604dc9d5b19, double 0x3fb9103075a4a0ab, double 0x3fb87b1c9dbf2852, double 0x3fb7e6ca013eefd6, double 0x3fb753395aaa1176, double 0x3fb6c06b73694a4c, double 0x3fb62e6124854d18, double 0x3fb59d1b577466a4, double 0x3fb50c9b06fa2bae, double 0x3fb47ce1401b2213, double 0x3fb3edef23269a86, double 0x3fb35fc5e4d93e70, double 0x3fb2d266cf9b3111, double 0x3fb245d344dd0d91, double 0x3fb1ba0cbe97897d, double 0x3fb12f14d0f2179d, double 0x3fb0a4ed2c159625, double 0x3fb01b979e30e497, double 0x3faf262c2b6c6e35, double 0x3fae16d547b25181, double 0x3fad092efeadf162, double 0x3fabfd3e0f282a2c, double 0x3faaf30790385f70, double 0x3fa9ea90f9295563, double 0x3fa8e3e02a68b5ab, double 0x3fa7defb77af271e, double 0x3fa6dbe9b398d064, double 0x3fa5dab23cf2add4, double 0x3fa4db5d0e11275d, double 0x3fa3ddf2ce98eecb, double 0x3fa2e27ce83df497, double 0x3fa1e9059f1f6abc, double 0x3fa0f1982e968011, double 0x3f9ff881d718a5c4, double 0x3f9e121adb828c75, double 0x3f9c301983cd091a, double 0x3f9a529f4e22ebf8, double 0x3f9879d1b600c10a, double 0x3f96a5daf40bbf82, double 0x3f94d6eaf2fbb064, double 0x3f930d388dab5e13, double 0x3f91490334603012, double 0x3f8f152a4f72dd49, double 0x3f8ba48d274f8fac, double 0x3f8841040d8da478, double 0x3f84eb96421acfe0, double 0x3f81a59229952f92, double 0x3f7ce160f8ec6837, double 0x3f769ea8d90cb85d, double 0x3f708a1f03b0b1fd, double 0x3f655f9f43c1b067, double 0x3f54a605b6b9f70f]
define void @"__pnl_builtin_philox_rand_int32"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* noalias nonnull %".1", i32* noalias nonnull %".2") argmemonly!dbg !47
{
entry:
  %".4" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 3 , !dbg !48
  %".5" = getelementptr {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}, {[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32 0, i32 5 , !dbg !48
  %".6" = load i1, i1* %".5", !dbg !48
  br i1 %".6", label %"entry.if", label %"entry.endif", !dbg !48
entry.if:
  %".8" = load i32, i32* %".4", !dbg !48
  store i32 %".8", i32* %".2", !dbg !48
  store i1 false, i1* %".5", !dbg !48
  ret void, !dbg !48
entry.endif:
  %"rand_i64" = alloca i64, !dbg !48
  call void @"__pnl_builtin_philox_rand_int64"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i64* %"rand_i64"), !dbg !48
  %".13" = load i64, i64* %"rand_i64", !dbg !48
  %".14" = trunc i64 %".13" to i32 , !dbg !48
  store i32 %".14", i32* %".2", !dbg !48
  %".16" = lshr i64 %".13", 32, !dbg !48
  %".17" = trunc i64 %".16" to i32 , !dbg !48
  store i32 %".17", i32* %".4", !dbg !48
  store i1 true, i1* %".5", !dbg !48
  ret void, !dbg !48
}

define void @"__pnl_builtin_philox_rand_float"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* noalias nonnull %".1", float* noalias nonnull %".2") argmemonly!dbg !50
{
entry:
  %"rand_int32" = alloca i32, !dbg !51
  call void @"__pnl_builtin_philox_rand_int32"({[4 x i64], [2 x i64], [4 x i64], i32, i16, i1, i64}* %".1", i32* %"rand_int32"), !dbg !51
  %".5" = load i32, i32* %"rand_int32", !dbg !51
  %".6" = lshr i32 %".5", 9, !dbg !51
  %".7" = uitofp i32 %".6" to float , !dbg !51
  %".8" = fmul float %".7", 0x3e80000000000000, !dbg !51
  store float %".8", float* %".2", !dbg !51
  ret void, !dbg !51
}

define void @"__pnl_builtin_vxm"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", i32 %".4", double* noalias nonnull %".5") argmemonly!dbg !53
{
entry:
  %"zero_index_var_loc" = alloca i32, !dbg !54
  store i32 0, i32* %"zero_index_var_loc", !dbg !54
  br label %"zero-cond-bb", !dbg !54
zero-cond-bb:
  %"zero_cond_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !54
  %"zero_loop_cond" = icmp slt i32 %"zero_cond_index_var", %".4" , !dbg !54
  br i1 %"zero_loop_cond", label %"zero-cond-bb.if", label %"zero-cond-bb.endif", !dbg !54, !prof !20
zero-cond-bb.if:
  %"zero_loop_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !54
  %".10" = getelementptr double, double* %".5", i32 %"zero_loop_index_var" , !dbg !54
  store double              0x0, double* %".10", !dbg !54
  %"zero_index_var_inc" = add i32 %"zero_loop_index_var", 1, !dbg !54
  store i32 %"zero_index_var_inc", i32* %"zero_index_var_loc", !dbg !54
  br label %"zero-cond-bb", !dbg !54
zero-cond-bb.endif:
  %"vxm_outer_index_var_loc" = alloca i32, !dbg !54
  store i32 0, i32* %"vxm_outer_index_var_loc", !dbg !54
  br label %"vxm_outer-cond-bb", !dbg !54
vxm_outer-cond-bb:
  %"vxm_outer_cond_index_var" = load i32, i32* %"vxm_outer_index_var_loc", !dbg !54
  %"vxm_outer_loop_cond" = icmp slt i32 %"vxm_outer_cond_index_var", %".3" , !dbg !54
  br i1 %"vxm_outer_loop_cond", label %"vxm_outer-cond-bb.if", label %"vxm_outer-cond-bb.endif", !dbg !54, !prof !20
vxm_outer-cond-bb.if:
  %"vxm_outer_loop_index_var" = load i32, i32* %"vxm_outer_index_var_loc", !dbg !54
  %"vxm_inner_index_var_loc" = alloca i32, !dbg !54
  store i32 0, i32* %"vxm_inner_index_var_loc", !dbg !54
  br label %"vxm_inner-cond-bb", !dbg !54
vxm_outer-cond-bb.endif:
  ret void, !dbg !54
vxm_inner-cond-bb:
  %"vxm_inner_cond_index_var" = load i32, i32* %"vxm_inner_index_var_loc", !dbg !54
  %"vxm_inner_loop_cond" = icmp slt i32 %"vxm_inner_cond_index_var", %".4" , !dbg !54
  br i1 %"vxm_inner_loop_cond", label %"vxm_inner-cond-bb.if", label %"vxm_inner-cond-bb.endif", !dbg !54, !prof !20
vxm_inner-cond-bb.if:
  %"vxm_inner_loop_index_var" = load i32, i32* %"vxm_inner_index_var_loc", !dbg !54
  %".20" = getelementptr double, double* %".1", i32 %"vxm_outer_loop_index_var" , !dbg !54
  %".21" = mul i32 %"vxm_outer_loop_index_var", %".4", !dbg !54
  %".22" = add i32 %".21", %"vxm_inner_loop_index_var", !dbg !54
  %".23" = getelementptr double, double* %".2", i32 %".22" , !dbg !54
  %".24" = getelementptr double, double* %".5", i32 %"vxm_inner_loop_index_var" , !dbg !54
  %".25" = load double, double* %".20", !dbg !54
  %".26" = load double, double* %".23", !dbg !54
  %".27" = load double, double* %".24", !dbg !54
  %".28" = fmul double %".25", %".26", !dbg !54
  %".29" = fadd double %".28", %".27", !dbg !54
  store double %".29", double* %".24", !dbg !54
  %"vxm_inner_index_var_inc" = add i32 %"vxm_inner_loop_index_var", 1, !dbg !54
  store i32 %"vxm_inner_index_var_inc", i32* %"vxm_inner_index_var_loc", !dbg !54
  br label %"vxm_inner-cond-bb", !dbg !54
vxm_inner-cond-bb.endif:
  %"vxm_outer_index_var_inc" = add i32 %"vxm_outer_loop_index_var", 1, !dbg !54
  store i32 %"vxm_outer_index_var_inc", i32* %"vxm_outer_index_var_loc", !dbg !54
  br label %"vxm_outer-cond-bb", !dbg !54
}

define void @"__pnl_builtin_vxm_transposed"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", i32 %".4", double* noalias nonnull %".5") argmemonly!dbg !56
{
entry:
  %"zero_index_var_loc" = alloca i32, !dbg !57
  store i32 0, i32* %"zero_index_var_loc", !dbg !57
  br label %"zero-cond-bb", !dbg !57
zero-cond-bb:
  %"zero_cond_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !57
  %"zero_loop_cond" = icmp slt i32 %"zero_cond_index_var", %".3" , !dbg !57
  br i1 %"zero_loop_cond", label %"zero-cond-bb.if", label %"zero-cond-bb.endif", !dbg !57, !prof !20
zero-cond-bb.if:
  %"zero_loop_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !57
  %".10" = getelementptr double, double* %".5", i32 %"zero_loop_index_var" , !dbg !57
  store double              0x0, double* %".10", !dbg !57
  %"zero_index_var_inc" = add i32 %"zero_loop_index_var", 1, !dbg !57
  store i32 %"zero_index_var_inc", i32* %"zero_index_var_loc", !dbg !57
  br label %"zero-cond-bb", !dbg !57
zero-cond-bb.endif:
  %"trans_vxm_outer_index_var_loc" = alloca i32, !dbg !57
  store i32 0, i32* %"trans_vxm_outer_index_var_loc", !dbg !57
  br label %"trans_vxm_outer-cond-bb", !dbg !57
trans_vxm_outer-cond-bb:
  %"trans_vxm_outer_cond_index_var" = load i32, i32* %"trans_vxm_outer_index_var_loc", !dbg !57
  %"trans_vxm_outer_loop_cond" = icmp slt i32 %"trans_vxm_outer_cond_index_var", %".3" , !dbg !57
  br i1 %"trans_vxm_outer_loop_cond", label %"trans_vxm_outer-cond-bb.if", label %"trans_vxm_outer-cond-bb.endif", !dbg !57, !prof !20
trans_vxm_outer-cond-bb.if:
  %"trans_vxm_outer_loop_index_var" = load i32, i32* %"trans_vxm_outer_index_var_loc", !dbg !57
  %"trans_vxm_inner_index_var_loc" = alloca i32, !dbg !57
  store i32 0, i32* %"trans_vxm_inner_index_var_loc", !dbg !57
  br label %"trans_vxm_inner-cond-bb", !dbg !57
trans_vxm_outer-cond-bb.endif:
  ret void, !dbg !57
trans_vxm_inner-cond-bb:
  %"trans_vxm_inner_cond_index_var" = load i32, i32* %"trans_vxm_inner_index_var_loc", !dbg !57
  %"trans_vxm_inner_loop_cond" = icmp slt i32 %"trans_vxm_inner_cond_index_var", %".4" , !dbg !57
  br i1 %"trans_vxm_inner_loop_cond", label %"trans_vxm_inner-cond-bb.if", label %"trans_vxm_inner-cond-bb.endif", !dbg !57, !prof !20
trans_vxm_inner-cond-bb.if:
  %"trans_vxm_inner_loop_index_var" = load i32, i32* %"trans_vxm_inner_index_var_loc", !dbg !57
  %".20" = getelementptr double, double* %".1", i32 %"trans_vxm_inner_loop_index_var" , !dbg !57
  %".21" = mul i32 %"trans_vxm_outer_loop_index_var", %".4", !dbg !57
  %".22" = add i32 %".21", %"trans_vxm_inner_loop_index_var", !dbg !57
  %".23" = getelementptr double, double* %".2", i32 %".22" , !dbg !57
  %".24" = getelementptr double, double* %".5", i32 %"trans_vxm_outer_loop_index_var" , !dbg !57
  %".25" = load double, double* %".20", !dbg !57
  %".26" = load double, double* %".23", !dbg !57
  %".27" = load double, double* %".24", !dbg !57
  %".28" = fmul double %".25", %".26", !dbg !57
  %".29" = fadd double %".28", %".27", !dbg !57
  store double %".29", double* %".24", !dbg !57
  %"trans_vxm_inner_index_var_inc" = add i32 %"trans_vxm_inner_loop_index_var", 1, !dbg !57
  store i32 %"trans_vxm_inner_index_var_inc", i32* %"trans_vxm_inner_index_var_loc", !dbg !57
  br label %"trans_vxm_inner-cond-bb", !dbg !57
trans_vxm_inner-cond-bb.endif:
  %"trans_vxm_outer_index_var_inc" = add i32 %"trans_vxm_outer_loop_index_var", 1, !dbg !57
  store i32 %"trans_vxm_outer_index_var_inc", i32* %"trans_vxm_outer_index_var_loc", !dbg !57
  br label %"trans_vxm_outer-cond-bb", !dbg !57
}

define void @"__pnl_builtin_vec_add"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", double* noalias nonnull %".4") argmemonly!dbg !59
{
entry:
  %"addition_index_var_loc" = alloca i32, !dbg !60
  store i32 0, i32* %"addition_index_var_loc", !dbg !60
  br label %"addition-cond-bb", !dbg !60
addition-cond-bb:
  %"addition_cond_index_var" = load i32, i32* %"addition_index_var_loc", !dbg !60
  %"addition_loop_cond" = icmp slt i32 %"addition_cond_index_var", %".3" , !dbg !60
  br i1 %"addition_loop_cond", label %"addition-cond-bb.if", label %"addition-cond-bb.endif", !dbg !60, !prof !20
addition-cond-bb.if:
  %"addition_loop_index_var" = load i32, i32* %"addition_index_var_loc", !dbg !60
  %".9" = getelementptr double, double* %".1", i32 %"addition_loop_index_var" , !dbg !60
  %".10" = getelementptr double, double* %".2", i32 %"addition_loop_index_var" , !dbg !60
  %".11" = getelementptr double, double* %".4", i32 %"addition_loop_index_var" , !dbg !60
  %".12" = load double, double* %".9", !dbg !60
  %".13" = load double, double* %".10", !dbg !60
  %".14" = fadd double %".12", %".13", !dbg !60
  store double %".14", double* %".11", !dbg !60
  %"addition_index_var_inc" = add i32 %"addition_loop_index_var", 1, !dbg !60
  store i32 %"addition_index_var_inc", i32* %"addition_index_var_loc", !dbg !60
  br label %"addition-cond-bb", !dbg !60
addition-cond-bb.endif:
  ret void, !dbg !60
}

define void @"__pnl_builtin_vec_sum"(double* noalias nonnull %".1", i32 %".2", double* noalias nonnull %".3") argmemonly!dbg !62
{
entry:
  store double              0x0, double* %".3", !dbg !63
  %"sum_index_var_loc" = alloca i32, !dbg !63
  store i32 0, i32* %"sum_index_var_loc", !dbg !63
  br label %"sum-cond-bb", !dbg !63
sum-cond-bb:
  %"sum_cond_index_var" = load i32, i32* %"sum_index_var_loc", !dbg !63
  %"sum_loop_cond" = icmp slt i32 %"sum_cond_index_var", %".2" , !dbg !63
  br i1 %"sum_loop_cond", label %"sum-cond-bb.if", label %"sum-cond-bb.endif", !dbg !63, !prof !20
sum-cond-bb.if:
  %"sum_loop_index_var" = load i32, i32* %"sum_index_var_loc", !dbg !63
  %".9" = getelementptr double, double* %".1", i32 %"sum_loop_index_var" , !dbg !63
  %".10" = load double, double* %".9", !dbg !63
  %".11" = load double, double* %".3", !dbg !63
  %".12" = fadd double %".10", %".11", !dbg !63
  store double %".12", double* %".3", !dbg !63
  %"sum_index_var_inc" = add i32 %"sum_loop_index_var", 1, !dbg !63
  store i32 %"sum_index_var_inc", i32* %"sum_index_var_loc", !dbg !63
  br label %"sum-cond-bb", !dbg !63
sum-cond-bb.endif:
  ret void, !dbg !63
}

define void @"__pnl_builtin_mat_add"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", i32 %".4", double* noalias nonnull %".5") argmemonly!dbg !65
{
entry:
  %"zero_index_var_loc" = alloca i32, !dbg !66
  store i32 0, i32* %"zero_index_var_loc", !dbg !66
  br label %"zero-cond-bb", !dbg !66
zero-cond-bb:
  %"zero_cond_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !66
  %"zero_loop_cond" = icmp slt i32 %"zero_cond_index_var", %".3" , !dbg !66
  br i1 %"zero_loop_cond", label %"zero-cond-bb.if", label %"zero-cond-bb.endif", !dbg !66, !prof !20
zero-cond-bb.if:
  %"zero_loop_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !66
  %"zero_inner_index_var_loc" = alloca i32, !dbg !66
  store i32 0, i32* %"zero_inner_index_var_loc", !dbg !66
  br label %"zero_inner-cond-bb", !dbg !66
zero-cond-bb.endif:
  ret void, !dbg !66
zero_inner-cond-bb:
  %"zero_inner_cond_index_var" = load i32, i32* %"zero_inner_index_var_loc", !dbg !66
  %"zero_inner_loop_cond" = icmp slt i32 %"zero_inner_cond_index_var", %".4" , !dbg !66
  br i1 %"zero_inner_loop_cond", label %"zero_inner-cond-bb.if", label %"zero_inner-cond-bb.endif", !dbg !66, !prof !20
zero_inner-cond-bb.if:
  %"zero_inner_loop_index_var" = load i32, i32* %"zero_inner_index_var_loc", !dbg !66
  %".13" = mul i32 %"zero_loop_index_var", %".4", !dbg !66
  %".14" = add i32 %".13", %"zero_inner_loop_index_var", !dbg !66
  %".15" = getelementptr double, double* %".1", i32 %".14" , !dbg !66
  %".16" = getelementptr double, double* %".2", i32 %".14" , !dbg !66
  %".17" = getelementptr double, double* %".5", i32 %".14" , !dbg !66
  %".18" = load double, double* %".15", !dbg !66
  %".19" = load double, double* %".16", !dbg !66
  %".20" = fadd double %".18", %".19", !dbg !66
  store double %".20", double* %".17", !dbg !66
  %"zero_inner_index_var_inc" = add i32 %"zero_inner_loop_index_var", 1, !dbg !66
  store i32 %"zero_inner_index_var_inc", i32* %"zero_inner_index_var_loc", !dbg !66
  br label %"zero_inner-cond-bb", !dbg !66
zero_inner-cond-bb.endif:
  %"zero_index_var_inc" = add i32 %"zero_loop_index_var", 1, !dbg !66
  store i32 %"zero_index_var_inc", i32* %"zero_index_var_loc", !dbg !66
  br label %"zero-cond-bb", !dbg !66
}

define void @"__pnl_builtin_vec_sub"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", double* noalias nonnull %".4") argmemonly!dbg !68
{
entry:
  %"subtraction_index_var_loc" = alloca i32, !dbg !69
  store i32 0, i32* %"subtraction_index_var_loc", !dbg !69
  br label %"subtraction-cond-bb", !dbg !69
subtraction-cond-bb:
  %"subtraction_cond_index_var" = load i32, i32* %"subtraction_index_var_loc", !dbg !69
  %"subtraction_loop_cond" = icmp slt i32 %"subtraction_cond_index_var", %".3" , !dbg !69
  br i1 %"subtraction_loop_cond", label %"subtraction-cond-bb.if", label %"subtraction-cond-bb.endif", !dbg !69, !prof !20
subtraction-cond-bb.if:
  %"subtraction_loop_index_var" = load i32, i32* %"subtraction_index_var_loc", !dbg !69
  %".9" = getelementptr double, double* %".1", i32 %"subtraction_loop_index_var" , !dbg !69
  %".10" = getelementptr double, double* %".2", i32 %"subtraction_loop_index_var" , !dbg !69
  %".11" = getelementptr double, double* %".4", i32 %"subtraction_loop_index_var" , !dbg !69
  %".12" = load double, double* %".9", !dbg !69
  %".13" = load double, double* %".10", !dbg !69
  %".14" = fsub double %".12", %".13", !dbg !69
  store double %".14", double* %".11", !dbg !69
  %"subtraction_index_var_inc" = add i32 %"subtraction_loop_index_var", 1, !dbg !69
  store i32 %"subtraction_index_var_inc", i32* %"subtraction_index_var_loc", !dbg !69
  br label %"subtraction-cond-bb", !dbg !69
subtraction-cond-bb.endif:
  ret void, !dbg !69
}

define void @"__pnl_builtin_mat_sub"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", i32 %".4", double* noalias nonnull %".5") argmemonly!dbg !71
{
entry:
  %"mat_sub_outer_index_var_loc" = alloca i32, !dbg !72
  store i32 0, i32* %"mat_sub_outer_index_var_loc", !dbg !72
  br label %"mat_sub_outer-cond-bb", !dbg !72
mat_sub_outer-cond-bb:
  %"mat_sub_outer_cond_index_var" = load i32, i32* %"mat_sub_outer_index_var_loc", !dbg !72
  %"mat_sub_outer_loop_cond" = icmp slt i32 %"mat_sub_outer_cond_index_var", %".3" , !dbg !72
  br i1 %"mat_sub_outer_loop_cond", label %"mat_sub_outer-cond-bb.if", label %"mat_sub_outer-cond-bb.endif", !dbg !72, !prof !20
mat_sub_outer-cond-bb.if:
  %"mat_sub_outer_loop_index_var" = load i32, i32* %"mat_sub_outer_index_var_loc", !dbg !72
  %"mat_sub_inner_index_var_loc" = alloca i32, !dbg !72
  store i32 0, i32* %"mat_sub_inner_index_var_loc", !dbg !72
  br label %"mat_sub_inner-cond-bb", !dbg !72
mat_sub_outer-cond-bb.endif:
  ret void, !dbg !72
mat_sub_inner-cond-bb:
  %"mat_sub_inner_cond_index_var" = load i32, i32* %"mat_sub_inner_index_var_loc", !dbg !72
  %"mat_sub_inner_loop_cond" = icmp slt i32 %"mat_sub_inner_cond_index_var", %".4" , !dbg !72
  br i1 %"mat_sub_inner_loop_cond", label %"mat_sub_inner-cond-bb.if", label %"mat_sub_inner-cond-bb.endif", !dbg !72, !prof !20
mat_sub_inner-cond-bb.if:
  %"mat_sub_inner_loop_index_var" = load i32, i32* %"mat_sub_inner_index_var_loc", !dbg !72
  %".13" = mul i32 %"mat_sub_outer_loop_index_var", %".4", !dbg !72
  %".14" = add i32 %".13", %"mat_sub_inner_loop_index_var", !dbg !72
  %".15" = getelementptr double, double* %".1", i32 %".14" , !dbg !72
  %".16" = getelementptr double, double* %".2", i32 %".14" , !dbg !72
  %".17" = getelementptr double, double* %".5", i32 %".14" , !dbg !72
  %".18" = load double, double* %".15", !dbg !72
  %".19" = load double, double* %".16", !dbg !72
  %".20" = fsub double %".18", %".19", !dbg !72
  store double %".20", double* %".17", !dbg !72
  %"mat_sub_inner_index_var_inc" = add i32 %"mat_sub_inner_loop_index_var", 1, !dbg !72
  store i32 %"mat_sub_inner_index_var_inc", i32* %"mat_sub_inner_index_var_loc", !dbg !72
  br label %"mat_sub_inner-cond-bb", !dbg !72
mat_sub_inner-cond-bb.endif:
  %"mat_sub_outer_index_var_inc" = add i32 %"mat_sub_outer_loop_index_var", 1, !dbg !72
  store i32 %"mat_sub_outer_index_var_inc", i32* %"mat_sub_outer_index_var_loc", !dbg !72
  br label %"mat_sub_outer-cond-bb", !dbg !72
}

define void @"__pnl_builtin_vec_hadamard"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", double* noalias nonnull %".4") argmemonly!dbg !74
{
entry:
  %"mult_index_var_loc" = alloca i32, !dbg !75
  store i32 0, i32* %"mult_index_var_loc", !dbg !75
  br label %"mult-cond-bb", !dbg !75
mult-cond-bb:
  %"mult_cond_index_var" = load i32, i32* %"mult_index_var_loc", !dbg !75
  %"mult_loop_cond" = icmp slt i32 %"mult_cond_index_var", %".3" , !dbg !75
  br i1 %"mult_loop_cond", label %"mult-cond-bb.if", label %"mult-cond-bb.endif", !dbg !75, !prof !20
mult-cond-bb.if:
  %"mult_loop_index_var" = load i32, i32* %"mult_index_var_loc", !dbg !75
  %".9" = getelementptr double, double* %".1", i32 %"mult_loop_index_var" , !dbg !75
  %".10" = getelementptr double, double* %".2", i32 %"mult_loop_index_var" , !dbg !75
  %".11" = getelementptr double, double* %".4", i32 %"mult_loop_index_var" , !dbg !75
  %".12" = load double, double* %".9", !dbg !75
  %".13" = load double, double* %".10", !dbg !75
  %".14" = fmul double %".12", %".13", !dbg !75
  store double %".14", double* %".11", !dbg !75
  %"mult_index_var_inc" = add i32 %"mult_loop_index_var", 1, !dbg !75
  store i32 %"mult_index_var_inc", i32* %"mult_index_var_loc", !dbg !75
  br label %"mult-cond-bb", !dbg !75
mult-cond-bb.endif:
  ret void, !dbg !75
}

define void @"__pnl_builtin_mat_hadamard"(double* noalias nonnull %".1", double* noalias nonnull %".2", i32 %".3", i32 %".4", double* noalias nonnull %".5") argmemonly!dbg !77
{
entry:
  %"mat_hadamard_outer_index_var_loc" = alloca i32, !dbg !78
  store i32 0, i32* %"mat_hadamard_outer_index_var_loc", !dbg !78
  br label %"mat_hadamard_outer-cond-bb", !dbg !78
mat_hadamard_outer-cond-bb:
  %"mat_hadamard_outer_cond_index_var" = load i32, i32* %"mat_hadamard_outer_index_var_loc", !dbg !78
  %"mat_hadamard_outer_loop_cond" = icmp slt i32 %"mat_hadamard_outer_cond_index_var", %".3" , !dbg !78
  br i1 %"mat_hadamard_outer_loop_cond", label %"mat_hadamard_outer-cond-bb.if", label %"mat_hadamard_outer-cond-bb.endif", !dbg !78, !prof !20
mat_hadamard_outer-cond-bb.if:
  %"mat_hadamard_outer_loop_index_var" = load i32, i32* %"mat_hadamard_outer_index_var_loc", !dbg !78
  %"mat_hadamard_inner_index_var_loc" = alloca i32, !dbg !78
  store i32 0, i32* %"mat_hadamard_inner_index_var_loc", !dbg !78
  br label %"mat_hadamard_inner-cond-bb", !dbg !78
mat_hadamard_outer-cond-bb.endif:
  ret void, !dbg !78
mat_hadamard_inner-cond-bb:
  %"mat_hadamard_inner_cond_index_var" = load i32, i32* %"mat_hadamard_inner_index_var_loc", !dbg !78
  %"mat_hadamard_inner_loop_cond" = icmp slt i32 %"mat_hadamard_inner_cond_index_var", %".4" , !dbg !78
  br i1 %"mat_hadamard_inner_loop_cond", label %"mat_hadamard_inner-cond-bb.if", label %"mat_hadamard_inner-cond-bb.endif", !dbg !78, !prof !20
mat_hadamard_inner-cond-bb.if:
  %"mat_hadamard_inner_loop_index_var" = load i32, i32* %"mat_hadamard_inner_index_var_loc", !dbg !78
  %".13" = mul i32 %"mat_hadamard_outer_loop_index_var", %".4", !dbg !78
  %".14" = add i32 %".13", %"mat_hadamard_inner_loop_index_var", !dbg !78
  %".15" = getelementptr double, double* %".1", i32 %".14" , !dbg !78
  %".16" = getelementptr double, double* %".2", i32 %".14" , !dbg !78
  %".17" = getelementptr double, double* %".5", i32 %".14" , !dbg !78
  %".18" = load double, double* %".15", !dbg !78
  %".19" = load double, double* %".16", !dbg !78
  %".20" = fmul double %".18", %".19", !dbg !78
  store double %".20", double* %".17", !dbg !78
  %"mat_hadamard_inner_index_var_inc" = add i32 %"mat_hadamard_inner_loop_index_var", 1, !dbg !78
  store i32 %"mat_hadamard_inner_index_var_inc", i32* %"mat_hadamard_inner_index_var_loc", !dbg !78
  br label %"mat_hadamard_inner-cond-bb", !dbg !78
mat_hadamard_inner-cond-bb.endif:
  %"mat_hadamard_outer_index_var_inc" = add i32 %"mat_hadamard_outer_loop_index_var", 1, !dbg !78
  store i32 %"mat_hadamard_outer_index_var_inc", i32* %"mat_hadamard_outer_index_var_loc", !dbg !78
  br label %"mat_hadamard_outer-cond-bb", !dbg !78
}

define void @"__pnl_builtin_vec_scalar_mult"(double* noalias nonnull %".1", double %".2", i32 %".3", double* noalias nonnull %".4") argmemonly!dbg !80
{
entry:
  %"scalar_mult_loop_index_var_loc" = alloca i32, !dbg !81
  store i32 0, i32* %"scalar_mult_loop_index_var_loc", !dbg !81
  br label %"scalar_mult_loop-cond-bb", !dbg !81
scalar_mult_loop-cond-bb:
  %"scalar_mult_loop_cond_index_var" = load i32, i32* %"scalar_mult_loop_index_var_loc", !dbg !81
  %"scalar_mult_loop_loop_cond" = icmp slt i32 %"scalar_mult_loop_cond_index_var", %".3" , !dbg !81
  br i1 %"scalar_mult_loop_loop_cond", label %"scalar_mult_loop-cond-bb.if", label %"scalar_mult_loop-cond-bb.endif", !dbg !81, !prof !20
scalar_mult_loop-cond-bb.if:
  %"scalar_mult_loop_loop_index_var" = load i32, i32* %"scalar_mult_loop_index_var_loc", !dbg !81
  %".9" = getelementptr double, double* %".1", i32 %"scalar_mult_loop_loop_index_var" , !dbg !81
  %".10" = getelementptr double, double* %".4", i32 %"scalar_mult_loop_loop_index_var" , !dbg !81
  %".11" = load double, double* %".9", !dbg !81
  %".12" = fmul double %".11", %".2", !dbg !81
  store double %".12", double* %".10", !dbg !81
  %"scalar_mult_loop_index_var_inc" = add i32 %"scalar_mult_loop_loop_index_var", 1, !dbg !81
  store i32 %"scalar_mult_loop_index_var_inc", i32* %"scalar_mult_loop_index_var_loc", !dbg !81
  br label %"scalar_mult_loop-cond-bb", !dbg !81
scalar_mult_loop-cond-bb.endif:
  ret void, !dbg !81
}

define void @"__pnl_builtin_mat_scalar_mult"(double* noalias nonnull %".1", double %".2", i32 %".3", i32 %".4", double* noalias nonnull %".5") argmemonly!dbg !83
{
entry:
  %"zero_index_var_loc" = alloca i32, !dbg !84
  store i32 0, i32* %"zero_index_var_loc", !dbg !84
  br label %"zero-cond-bb", !dbg !84
zero-cond-bb:
  %"zero_cond_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !84
  %"zero_loop_cond" = icmp slt i32 %"zero_cond_index_var", %".3" , !dbg !84
  br i1 %"zero_loop_cond", label %"zero-cond-bb.if", label %"zero-cond-bb.endif", !dbg !84, !prof !20
zero-cond-bb.if:
  %"zero_loop_index_var" = load i32, i32* %"zero_index_var_loc", !dbg !84
  %"zero_inner_index_var_loc" = alloca i32, !dbg !84
  store i32 0, i32* %"zero_inner_index_var_loc", !dbg !84
  br label %"zero_inner-cond-bb", !dbg !84
zero-cond-bb.endif:
  ret void, !dbg !84
zero_inner-cond-bb:
  %"zero_inner_cond_index_var" = load i32, i32* %"zero_inner_index_var_loc", !dbg !84
  %"zero_inner_loop_cond" = icmp slt i32 %"zero_inner_cond_index_var", %".4" , !dbg !84
  br i1 %"zero_inner_loop_cond", label %"zero_inner-cond-bb.if", label %"zero_inner-cond-bb.endif", !dbg !84, !prof !20
zero_inner-cond-bb.if:
  %"zero_inner_loop_index_var" = load i32, i32* %"zero_inner_index_var_loc", !dbg !84
  %".13" = mul i32 %"zero_loop_index_var", %".4", !dbg !84
  %".14" = add i32 %".13", %"zero_inner_loop_index_var", !dbg !84
  %".15" = getelementptr double, double* %".1", i32 %".14" , !dbg !84
  %".16" = getelementptr double, double* %".5", i32 %".14" , !dbg !84
  %".17" = load double, double* %".15", !dbg !84
  %".18" = fmul double %".2", %".17", !dbg !84
  store double %".18", double* %".16", !dbg !84
  %"zero_inner_index_var_inc" = add i32 %"zero_inner_loop_index_var", 1, !dbg !84
  store i32 %"zero_inner_index_var_inc", i32* %"zero_inner_index_var_loc", !dbg !84
  br label %"zero_inner-cond-bb", !dbg !84
zero_inner-cond-bb.endif:
  %"zero_index_var_inc" = add i32 %"zero_loop_index_var", 1, !dbg !84
  store i32 %"zero_index_var_inc", i32* %"zero_index_var_loc", !dbg !84
  br label %"zero-cond-bb", !dbg !84
}

define void @"__pnl_builtin_mat_scalar_add"(double* noalias nonnull %".1", double %".2", i32 %".3", i32 %".4", double* noalias nonnull %".5") argmemonly!dbg !86
{
entry:
  %"mat_scalar_add_outer_index_var_loc" = alloca i32, !dbg !87
  store i32 0, i32* %"mat_scalar_add_outer_index_var_loc", !dbg !87
  br label %"mat_scalar_add_outer-cond-bb", !dbg !87
mat_scalar_add_outer-cond-bb:
  %"mat_scalar_add_outer_cond_index_var" = load i32, i32* %"mat_scalar_add_outer_index_var_loc", !dbg !87
  %"mat_scalar_add_outer_loop_cond" = icmp slt i32 %"mat_scalar_add_outer_cond_index_var", %".3" , !dbg !87
  br i1 %"mat_scalar_add_outer_loop_cond", label %"mat_scalar_add_outer-cond-bb.if", label %"mat_scalar_add_outer-cond-bb.endif", !dbg !87, !prof !20
mat_scalar_add_outer-cond-bb.if:
  %"mat_scalar_add_outer_loop_index_var" = load i32, i32* %"mat_scalar_add_outer_index_var_loc", !dbg !87
  %"mat_scalar_add_inner_index_var_loc" = alloca i32, !dbg !87
  store i32 0, i32* %"mat_scalar_add_inner_index_var_loc", !dbg !87
  br label %"mat_scalar_add_inner-cond-bb", !dbg !87
mat_scalar_add_outer-cond-bb.endif:
  ret void, !dbg !87
mat_scalar_add_inner-cond-bb:
  %"mat_scalar_add_inner_cond_index_var" = load i32, i32* %"mat_scalar_add_inner_index_var_loc", !dbg !87
  %"mat_scalar_add_inner_loop_cond" = icmp slt i32 %"mat_scalar_add_inner_cond_index_var", %".4" , !dbg !87
  br i1 %"mat_scalar_add_inner_loop_cond", label %"mat_scalar_add_inner-cond-bb.if", label %"mat_scalar_add_inner-cond-bb.endif", !dbg !87, !prof !20
mat_scalar_add_inner-cond-bb.if:
  %"mat_scalar_add_inner_loop_index_var" = load i32, i32* %"mat_scalar_add_inner_index_var_loc", !dbg !87
  %".13" = mul i32 %"mat_scalar_add_outer_loop_index_var", %".4", !dbg !87
  %".14" = add i32 %".13", %"mat_scalar_add_inner_loop_index_var", !dbg !87
  %".15" = getelementptr double, double* %".1", i32 %".14" , !dbg !87
  %".16" = getelementptr double, double* %".5", i32 %".14" , !dbg !87
  %".17" = load double, double* %".15", !dbg !87
  %".18" = fadd double %".2", %".17", !dbg !87
  store double %".18", double* %".16", !dbg !87
  %"mat_scalar_add_inner_index_var_inc" = add i32 %"mat_scalar_add_inner_loop_index_var", 1, !dbg !87
  store i32 %"mat_scalar_add_inner_index_var_inc", i32* %"mat_scalar_add_inner_index_var_loc", !dbg !87
  br label %"mat_scalar_add_inner-cond-bb", !dbg !87
mat_scalar_add_inner-cond-bb.endif:
  %"mat_scalar_add_outer_index_var_inc" = add i32 %"mat_scalar_add_outer_loop_index_var", 1, !dbg !87
  store i32 %"mat_scalar_add_outer_index_var_inc", i32* %"mat_scalar_add_outer_index_var_loc", !dbg !87
  br label %"mat_scalar_add_outer-cond-bb", !dbg !87
}

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{ !5, !8, !11, !14, !17, !21, !25, !28, !31, !34, !37, !40, !43, !46, !49, !52, !55, !58, !61, !64, !67, !70, !73, !76, !79, !82, !85 }
!0 = !{ i32 2, !"Dwarf Version", i32 4 }
!1 = !{ i32 2, !"Debug Info Version", i32 3 }
!2 = !DIFile(directory: "", filename: "<pnl_builtin>")
!3 = !{ null }
!4 = !DISubroutineType(types: !3)
!5 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!6 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_csch", type: !4, unit: !5)
!7 = !DILocation(column: 0, line: 0, scope: !6)
!8 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!9 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_coth", type: !4, unit: !8)
!10 = !DILocation(column: 0, line: 0, scope: !9)
!11 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!12 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_tanh", type: !4, unit: !11)
!13 = !DILocation(column: 0, line: 0, scope: !12)
!14 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!15 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_is_close_double", type: !4, unit: !14)
!16 = !DILocation(column: 0, line: 0, scope: !15)
!17 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!18 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mt_rand_init_scalar", type: !4, unit: !17)
!19 = !DILocation(column: 0, line: 0, scope: !18)
!20 = !{ !"branch_weights", i32 99, i32 1 }
!21 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!22 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mt_rand_init", type: !4, unit: !21)
!23 = !DILocation(column: 0, line: 0, scope: !22)
!24 = !{ !"branch_weights", i32 1, i32 99 }
!25 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!26 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mt_rand_int32", type: !4, unit: !25)
!27 = !DILocation(column: 0, line: 0, scope: !26)
!28 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!29 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mt_rand_double", type: !4, unit: !28)
!30 = !DILocation(column: 0, line: 0, scope: !29)
!31 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!32 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mt_rand_normal", type: !4, unit: !31)
!33 = !DILocation(column: 0, line: 0, scope: !32)
!34 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!35 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_philox_rand_init", type: !4, unit: !34)
!36 = !DILocation(column: 0, line: 0, scope: !35)
!37 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!38 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_philox_rand_int64", type: !4, unit: !37)
!39 = !DILocation(column: 0, line: 0, scope: !38)
!40 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!41 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_philox_rand_double", type: !4, unit: !40)
!42 = !DILocation(column: 0, line: 0, scope: !41)
!43 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!44 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_philox_rand_normal", type: !4, unit: !43)
!45 = !DILocation(column: 0, line: 0, scope: !44)
!46 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!47 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_philox_rand_int32", type: !4, unit: !46)
!48 = !DILocation(column: 0, line: 0, scope: !47)
!49 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!50 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_philox_rand_float", type: !4, unit: !49)
!51 = !DILocation(column: 0, line: 0, scope: !50)
!52 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!53 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_vxm", type: !4, unit: !52)
!54 = !DILocation(column: 0, line: 0, scope: !53)
!55 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!56 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_vxm_transposed", type: !4, unit: !55)
!57 = !DILocation(column: 0, line: 0, scope: !56)
!58 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!59 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_vec_add", type: !4, unit: !58)
!60 = !DILocation(column: 0, line: 0, scope: !59)
!61 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!62 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_vec_sum", type: !4, unit: !61)
!63 = !DILocation(column: 0, line: 0, scope: !62)
!64 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!65 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mat_add", type: !4, unit: !64)
!66 = !DILocation(column: 0, line: 0, scope: !65)
!67 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!68 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_vec_sub", type: !4, unit: !67)
!69 = !DILocation(column: 0, line: 0, scope: !68)
!70 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!71 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mat_sub", type: !4, unit: !70)
!72 = !DILocation(column: 0, line: 0, scope: !71)
!73 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!74 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_vec_hadamard", type: !4, unit: !73)
!75 = !DILocation(column: 0, line: 0, scope: !74)
!76 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!77 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mat_hadamard", type: !4, unit: !76)
!78 = !DILocation(column: 0, line: 0, scope: !77)
!79 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!80 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_vec_scalar_mult", type: !4, unit: !79)
!81 = !DILocation(column: 0, line: 0, scope: !80)
!82 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!83 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mat_scalar_mult", type: !4, unit: !82)
!84 = !DILocation(column: 0, line: 0, scope: !83)
!85 = distinct !DICompileUnit(file: !2, isOptimized: false, language: DW_LANG_Python, producer: "PsyNeuLink", runtimeVersion: 0)
!86 = distinct !DISubprogram(file: !2, isLocal: false, line: 0, name: "__pnl_builtin_mat_scalar_add", type: !4, unit: !85)
!87 = !DILocation(column: 0, line: 0, scope: !86)