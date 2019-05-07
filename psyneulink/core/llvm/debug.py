
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Debug environment **************************************************************
# This file provides preprocessed interface to PNL_LLVM_DEBUG environment variable
# The currently recognized values are:
# "compile" -- prints information messages when modules are compiled
# "stat" -- prints code generation and compilation statistics at the end
# "debug_info" -- emit line debugging information when generating LLVM IR
# "const_params" -- hardcode base parameter values into generated code,
#                   instead of loading them from param_struct
# "alloca_data" -- use alloca'd storage for composition data (exposes data flow)
# "comp_node_debug" -- print intermediate results after execution composition node wrapper.
# "llvm" -- dumps LLVM IR into a file (named after the dumped module).
#            Code is dumped both after module generation and linking into global module.
# "opt" -- dump LLVM IR after running through the optimization passes
# "isa" -- dump machine specific ISA
# "cuda" -- enable execution on CUDA devices if available
# "cuda_data" -- print data upload/download statistic (to GPU VRAM)

import os

debug_env = str(os.environ.get("PNL_LLVM_DEBUG")).split(',')
