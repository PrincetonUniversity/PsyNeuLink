Core
====

* `Component`
   - `Mechanism`

      - `ProcessingMechanism`

        - `TransferMechanism`

        - `IntegratorMechanism`

        - `ObjectiveMechanism`


      - `ModulatoryMechanism`

        - `ControlMechanism`

        - `LearningMechanism`

   - `Projection`

      - `PathwayProjection`

        - `MappingProjection`

        - `MaskedMappingProjection`

        - `AutoAssociativeProjection`

      - `ModulatoryProjection`

        - `LearningProjection`

        - `ControlProjection`

        - `GatingProjection`

   - `Port`

      - `InputPort`

      - `ParameterPort`

      - `OutputPort`

      - `ModulatorySignal`

        - `LearningSignal`

        - `ControlSignal`

        - `GatingSignal`

   - `Function`

      - `NonStatefulFunctions`

            - `DistributionFunctions`

            - `LearningFunctions`

            - `ObjectiveFunctions`

            - `OptimizationFunctions`

            - `SelectionFunctions`

            - `TransferFunctions`

            - `TimerFunctions`

            - `TransformFunctions`

      - `StatefulFunctions`

            - `IntegratorFunctions`

            - `MemoryFunctions`

      - `UserDefinedFunction`


* `Composition`
   - `AutodiffComposition`
   - `CompositionFunctionApproximator`
   - `ParameterEstimationComposition`

* `Services`
   - `Registry`
   - `Preferences`
   - `Visualization`
   - `Scheduling`
   - `Compilation`
   - `Report`
   - `Log`
   - `mdf`
   - :py:module:`Graph`
