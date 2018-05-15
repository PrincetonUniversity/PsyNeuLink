from psyneulink.compositions.composition import Composition, MechanismRole
from psyneulink.components.mechanisms.mechanism import Mechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.projections.projection import Projection
from psyneulink.globals.keywords import SOFT_CLAMP


__all__ = [
    'PathwayComposition', 'PathwayCompositionError'
]

class PathwayCompositionError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class PathwayComposition(Composition):
    '''

            Arguments
            ----------

            Attributes
            ----------

            Returns
            ----------
    '''

    def __init__(self):
        super(PathwayComposition, self).__init__()

    def add_linear_processing_pathway(self, pathway):
        # First, verify that the pathway begins with a mechanism
        if isinstance(pathway[0], Mechanism):
            self.add_mechanism(pathway[0])
        else:
            # 'MappingProjection has no attribute _name' error is thrown when pathway[0] is passed to the error msg
            raise PathwayCompositionError("The first item in a linear processing pathway must be a "
                                   "mechanism.")
        # Then, add all of the remaining mechanisms in the pathway
        for c in range(1, len(pathway)):
            # if the current item is a mechanism, add it
            if isinstance(pathway[c], Mechanism):
                self.add_mechanism(pathway[c])

        # Then, loop through and validate that the mechanism-projection relationships make sense
        # and add MappingProjections where needed
        for c in range(1, len(pathway)):
            if isinstance(pathway[c], Mechanism):
                if isinstance(pathway[c - 1], Mechanism):
                    # if the previous item was also a mechanism, add a mapping projection between them
                    self.add_projection(
                        pathway[c - 1],
                        MappingProjection(
                            sender=pathway[c - 1],
                            receiver=pathway[c]
                        ),
                        pathway[c]
                    )
            # if the current item is a projection
            elif isinstance(pathway[c], Projection):
                if c == len(pathway) - 1:
                    raise PathwayCompositionError("{} is the last item in the pathway. A projection cannot be the last item in"
                                           " a linear processing pathway.".format(pathway[c]))
                # confirm that it is between two mechanisms, then add the projection
                if isinstance(pathway[c - 1], Mechanism) and isinstance(pathway[c + 1], Mechanism):
                    self.add_projection(pathway[c - 1], pathway[c], pathway[c + 1])
                else:
                    raise PathwayCompositionError(
                        "{} is not between two mechanisms. A projection in a linear processing pathway must be preceded"
                        " by a mechanism and followed by a mechanism".format(pathway[c]))
            else:
                raise PathwayCompositionError("{} is not a projection or mechanism. A linear processing pathway must be made "
                                       "up of projections and mechanisms.".format(pathway[c]))

    def execute(
        self,
        inputs,
        scheduler_processing=None,
        scheduler_learning=None,
        termination_processing=None,
        termination_learning=None,
        call_before_time_step=None,
        call_before_pass=None,
        call_after_time_step=None,
        call_after_pass=None,
        execution_id=None,
        clamp_input=SOFT_CLAMP,
        targets=None,
        runtime_params=None,
    ):

        if isinstance(inputs, list):
            inputs = {self.get_mechanisms_by_role(MechanismRole.ORIGIN).pop(): inputs}

        output = super(PathwayComposition, self).execute(
            inputs,
            scheduler_processing,
            scheduler_learning,
            termination_processing,
            termination_learning,
            call_before_time_step,
            call_before_pass,
            call_after_time_step,
            call_after_pass,
            execution_id,
            clamp_input,
            targets,
            runtime_params
        )
        return output
