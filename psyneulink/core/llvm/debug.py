
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
 * "cuda" -- enable execution on CUDA devices if available

Increased debug output:
 * "compile" -- prints information messages when modules are compiled
 * "stat" -- prints code generation and compilation statistics
 * "cuda_data" -- print data upload/download statistic (to GPU VRAM)
 * "comp_node_debug" -- print intermediate results after execution composition node wrapper.
 * "print_values" -- Enabled printfs in llvm code (from ctx printf helper)

Compilation modifiers:
 * "alloca_data" -- use alloca'd storage for composition data (exposes data flow)
 * "debug_info" -- emit line debugging information when generating LLVM IR
 * "const_data" -- hardcode initial output values into generated code,
                instead of loading them from the data argument
 * "const_input" -- hardcode input values for composition runs
 * "const_params" -- hardcode base parameter values into generated code,
                  instead of loading them from the param argument
 * "const_state" -- hardcode base context values into generate code,
                 instead of laoding them from the context argument
 * "no_ref_pass" -- Don't pass arguments to llvm functions by reference

Compiled code dump:
 * "llvm" -- dumps LLVM IR into a file (named after the dumped module).
           Code is dumped both after module generation and linking into global module.
 * "opt" -- dump LLVM IR after running through the optimization passes
 * "isa" -- dump machine specific ISA
"""

import os
from typing import Any, Dict

debug_env: Dict[str, Any] = dict()


def _update() -> None:
    """Update debug_env variable with the latest state of PNL_LLVM_DEBUG env var."""
    global debug_env
    debug_env.clear()
    debug_env.update({x.partition('=')[0:3:2] for x in str(os.environ.get("PNL_LLVM_DEBUG")).split(';')})


_update()
