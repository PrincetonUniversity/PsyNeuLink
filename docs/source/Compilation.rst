Compilation
===========

PsyNeulink includes a runtime compiler to improve performance of executed models.
This section describes the overview of the compiler design and its use.
The performance improvements varies, but it has been observed to be between one and three orders of magnitude depending on the model.


Overview
--------

The PsyNeuLink runtime compiler works in several steps when invoked via `run` or `execute`:
*Compilation*:
 #. The model is initialized. This is step is identical to non-compiled execution.
 #. Data structures (input/output/parameters) are flattened and converted to LLVM IR form.
 #. LLVM IR code is generated to match to semantics of individual components and the used scheduling rules.
 #. Host CPU compatible binary code is generated
 #. The resulting function is saved as `ctypes` function and the parameter types are converted to `ctypes` binary structures.

*Execution*:
 #. parameter structures are populated with the data from `Composition` based on the provided `execution_id`. These structures are preserved between invocations so executions with the same `execution_id` will reuse the same binary structures.
 #. `ctype` function from step 5. is executed
 #. Results are extracted from the binary structures and converted to Python format.


Use
---

Compiled form of a model can be invoked by passing one of the following values to the `bin_execute` parameter of `Composition.run`, or `Composition.exec`:

  * `False` or `Python`: Normal python execution
  * `LLVM`: Compile and execute individual nodes. The scheduling loop still runs in Python. If any of the nodes fails to compile, an error is raised. *NOTE:* Schedules that require access to node data will not work correctly.
  * `LLVMExec`: Execution of `Composition.exec` is replaced by a compiled equivalent. If the `Composition` fails to compile, an error is raised.
  * `LLVMRun`: Execution of `Composition.run` is replaced by a compiled equivalent. If the `Composition` fails to compiler, an error is raised.
  * `True`: This option attempts all three above mentioned granularities, and gracefully falls back to lower granularity. Warnings are raised in place of errors. This is the recommended way to invoke compiled execution as the final fallback is the Python baseline.

Note that data other than `Composition.run` outputs are not synchronized between Python and compiled execution.
 
 It is possible to invoke compiled version of `FUnction` s and `Mechanism` s. This functionality is provided for testing purposes only, because of the lack of data synchronization it is not recommended for general use.
