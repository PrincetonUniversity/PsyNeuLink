import numpy as np
import psyneulink as pnl


# -------------------------------------------------------------------------
# Construct a minimal two-field EMComposition2:
#   - KEY is a query/key field
#   - VALUE is a value field
# -------------------------------------------------------------------------

em = pnl.EMComposition2(
    name="EM2",
    memory_template=[
        [[1.0, 0.0], [10.0, 0.0]],
        [[0.0, 1.0], [0.0, 10.0]],
        [[0.1, 0.1], [0.1, 0.1]],
    ],
    memory_capacity=3,
    field_names=["KEY", "VALUE"],
    field_weights=[1.0, None],
    normalize_memories=False,
    memory_decay_rate=0.0,
    storage_prob=1.0,
    enable_learning=False,
    softmax_threshold=None,
)


def get_node(*candidate_names):
    """Return the first node found among candidate names."""
    for name in candidate_names:
        try:
            return em.nodes[name]
        except KeyError:
            pass
    raise KeyError(f"Could not find any of these nodes: {candidate_names}")


# -------------------------------------------------------------------------
# Get nodes by name.
#
# Depending on the current local spelling in EMComposition2, field memory
# nodes may be named either "[FIELD_MEMORY]" or "[FIELD MEMORY]".
# -------------------------------------------------------------------------

key_query = get_node("KEY [QUERY]")
value_value = get_node("VALUE [VALUE]")

key_field_memory = get_node("KEY [FIELD_MEMORY]", "KEY [FIELD MEMORY]")
value_field_memory = get_node("VALUE [FIELD_MEMORY]", "VALUE [FIELD MEMORY]")

retrieve = get_node("RETRIEVE")

key_retrieved = get_node("KEY [RETRIEVED]")
value_retrieved = get_node("VALUE [RETRIEVED]")


# -------------------------------------------------------------------------
# Enforce desired execution order:
#
# TimeStep 0:  ["KEY [QUERY]", "VALUE [VALUE]"]
# TimeStep 1:  ["KEY [FIELD_MEMORY]", "VALUE [FIELD_MEMORY]"]
# TimeStep 2:  "RETRIEVE"
# TimeStep 3:  ["KEY [FIELD_MEMORY]", "VALUE [FIELD_MEMORY]"]
# TimeStep 4:  ["KEY [RETRIEVED]", "VALUE [RETRIEVED]"]
# -------------------------------------------------------------------------

# The input nodes are origins; they can run at the start of the trial.
em.scheduler.add_condition(key_query, pnl.Always())
em.scheduler.add_condition(value_value, pnl.Always())

# Field-memory mechanisms must run once after inputs, then again after RETRIEVE.
em.scheduler.add_condition(
    key_field_memory,
    pnl.Any(
        pnl.All(
            pnl.AfterNCalls(key_query, 1),
            pnl.AfterNCalls(value_value, 1),
            pnl.BeforeNCalls(retrieve, 1),
        ),
        pnl.All(
            pnl.AfterNCalls(retrieve, 1),
            pnl.BeforeNCalls(key_retrieved, 1),
        ),
    ),
)

em.scheduler.add_condition(
    value_field_memory,
    pnl.Any(
        pnl.All(
            pnl.AfterNCalls(key_query, 1),
            pnl.AfterNCalls(value_value, 1),
            pnl.BeforeNCalls(retrieve, 1),
        ),
        pnl.All(
            pnl.AfterNCalls(retrieve, 1),
            pnl.BeforeNCalls(value_retrieved, 1),
        ),
    ),
)

# RETRIEVE runs only after both field-memory mechanisms have run once.
em.scheduler.add_condition(
    retrieve,
    pnl.All(
        pnl.AfterNCalls(key_field_memory, 1),
        pnl.AfterNCalls(value_field_memory, 1),
        pnl.BeforeNCalls(key_field_memory, 2),
        pnl.BeforeNCalls(value_field_memory, 2),
    ),
)

# Retrieved nodes run only after both field-memory mechanisms have run twice.
em.scheduler.add_condition(
    key_retrieved,
    pnl.All(
        pnl.AfterNCalls(key_field_memory, 2),
        pnl.AfterNCalls(value_field_memory, 2),
    ),
)

em.scheduler.add_condition(
    value_retrieved,
    pnl.All(
        pnl.AfterNCalls(key_field_memory, 2),
        pnl.AfterNCalls(value_field_memory, 2),
    ),
)

# End the trial after both retrieved nodes have executed once.
em.scheduler.termination_conds[pnl.TimeScale.TRIAL] = pnl.All(
    pnl.AfterNCalls(key_retrieved, 1),
    pnl.AfterNCalls(value_retrieved, 1),
)


# -------------------------------------------------------------------------
# Example execution
# -------------------------------------------------------------------------

result = em.run(
    inputs={
        key_query: [[1.0, 0.0]],
        value_value: [[10.0, 0.0]],
    },
    execution_mode=pnl.ExecutionMode.Python,
)

print("Result:")
print(result)

print("\nFinal memory:")
print(em.memory)