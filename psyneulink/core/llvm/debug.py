
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Debug environment **************************************************************
"""
Interface to PNL_LLVM_DEBUG environment variable.

The currently recognized values are:
Features:
 * "cuda-check" -- print the result of initializing pycuda

Increased debug output:
 * "compile" -- prints information messages when modules are compiled
 * "stat" -- prints code generation and compilation statistics
 * "time_stat" -- print compilation and code generation times
 * "comp_node_debug" -- print intermediate results after execution composition node wrapper.
 * "printf_tags" -- Enable print statements in compiled code with the specified tags

Compilation modifiers:
 * "const_data" -- hardcode initial output values into generated code,
                instead of loading them from the data argument
 * "const_input" -- hardcode input values for composition runs
 * "const_params" -- hardcode base parameter values into generated code,
                  instead of loading them from the param argument
 * "const_state" -- hardcode base context values into generate code,
                 instead of laoding them from the context argument
 * "opt" -- Set compiler optimization level (0,1,2,3)
 * "unaligned_copy" -- Do not assume structures are 4B aligned

CUDA options:
 * "cuda_max_regs"  -- Set maximum allowed GPU arch registers.
                       Equivalent to the CUDA JIT compiler option of the same name.
 * "cuda_no_shared" -- Do not use on-chip 'shared' memory in generated code.

Compiled code dump:
 * "dump-llvm-gen" -- Dumps LLVM IR generated by us into a file (named after the dumped module).
                      IR is dumped both after module generation and linking into global module.
 * "dump-llvm-opt" -- Dump LLVM IR after running through the optimization passes.
 * "dump-asm"      -- Dump machine specific asm, fed to the JIT engine (currently CPU ISA, or PTX).
"""

import os
from psyneulink._typing import Any, Dict

debug_env: Dict[str, Any] = dict()


def _update() -> None:
    """Update debug_env variable with the latest state of PNL_LLVM_DEBUG env var."""
    global debug_env
    debug_env.clear()
    debug_env.update({x.partition('=')[0:3:2] for x in str(os.environ.get("PNL_LLVM_DEBUG")).split(';')})


_update()
