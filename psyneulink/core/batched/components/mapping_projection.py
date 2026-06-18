"""Batched op for dense `MappingProjection`."""

from psyneulink.core.batched.backend.triton.api import pnl_triton_op
from psyneulink.core.batched.specs import DenseProjectionSpec, register_batched_op
from psyneulink.core.components.projections.pathway.mappingprojection import (
    MappingProjection,
)


@pnl_triton_op
def _pnl_triton_projection_term(x, coefficient):
    return x * coefficient


def _mapping_triton_emit(ctx, projection_spec, sender_values, output_vars):
    helper_name = ctx.helper_name(_pnl_triton_projection_term)
    for col_idx, output_var in enumerate(output_vars):
        terms = []
        for row_idx, sender_value in enumerate(sender_values):
            coefficient = float(projection_spec.matrix[row_idx, col_idx])
            if coefficient:
                terms.append(
                    f"{helper_name}({sender_value}, {ctx.float_literal(coefficient)})"
                )
        ctx.line(f"{output_var} = {' + '.join(terms) if terms else ctx.zero_vector()}")
    return output_vars


register_batched_op(
    DenseProjectionSpec(
        projection_class=MappingProjection,
        triton_emit=_mapping_triton_emit,
    )
)
