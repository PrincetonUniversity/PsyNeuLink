import psyneulink as pnl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# it kept giving noisy warnings 
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# CONFIGURATION & DIMENSIONS (values from the paper) ---
KEY_DIMENSION = 64                   # Dimension of the abstract variable (key, k_w)
VALUE_DIMENSION = 64                 # Dimension of the concrete entity embedding (value, z_t)
HIDDEN_DIMENSION = 512               # Size of the GRU's internal state (h)
GRU_INPUT_SIZE = KEY_DIMENSION + 1   # Input size to the GRU: Retrieved Key (64) + Confidence Signal (1)
OUTPUT_DIMENSION = 4                 # Set to 4 to match the generic multi-class output layer (e.g., Identity Rules)

# ======================================================================
# --- MODEL DEFINITIONS (ESBN-GRU COMPONENTS) ---
# ======================================================================

# --- 1. ABSTRACT CONTROLLER (using GRU) ---
gru_controller = pnl.GRUComposition(
    name='GRU Controller (ESBN fs)',
    input_size=GRU_INPUT_SIZE,       # Takes the retrieved key (k_r + c_k) as input
    hidden_size=HIDDEN_DIMENSION,    # Size of the sequential memory
    bias=True
)
gru_hidden_node = gru_controller.hidden_layer_node
gru_sender_port = gru_hidden_node.output_ports[0] # Source port for all controller readouts

# this was a fix I added; IsSet a small, random initial value for the GRU's hidden state (h_t=0)
# to prevent immediate NaN propagation due to zero inputs.
GRU_INITIAL_STATE = np.random.uniform(low=-0.01, high=0.01, size=HIDDEN_DIMENSION)
try: gru_hidden_node.parameters.initial_value.set(GRU_INITIAL_STATE)
except Exception: pass

# --- 2. Controller Output Readouts ---
y_hat_mech = pnl.TransferMechanism(
    name='Prediction (y_hat)',
    function=pnl.SoftMax(mask_threshold=1e-6), # Final classification output (SoftMax for multi-class tasks)
    default_variable=np.zeros(OUTPUT_DIMENSION)
)

kw_mech = pnl.TransferMechanism(
    name='Key to Write (kw)',
    function=pnl.ReLU(), # ESBN uses ReLU for k_w 
    default_variable=np.zeros(KEY_DIMENSION) # Output is the abstract variable to be bound
)

g_mech = pnl.TransferMechanism(
    name='Gate (g)',
    function=pnl.Logistic(gain=0.5, bias=0.0), # Sigmoid gate (g_t) used to modulate k_r 
    default_variable=np.zeros(1)
)

# Probe to extract the internal hidden state (h_t) for external training loop
hidden_probe = pnl.TransferMechanism(
    name='Hidden Probe', function=pnl.Identity(), default_variable=np.zeros(HIDDEN_DIMENSION))


# --- 3. CONCRETE STREAM & EXTERNAL MEMORY (M_k and M_v) ---
z_embedding_mech = pnl.TransferMechanism(
    name='Image Embedding (Zt_Input)', # Feeds the concrete entity (z_t) to the system
    default_variable=np.zeros(VALUE_DIMENSION)
)

em_memory = pnl.EMComposition_Proj(
    name='External Memory (Mk & Mv)',
    # Memory Structure: [Field 0: Zt_Value (64), Field 1: Kw_Key (64)]
    memory_template=[np.zeros(VALUE_DIMENSION), np.zeros(KEY_DIMENSION)], 
    # fix i added: Initialize memory with small values to avoid zero division warnings
    memory_fill=(0.001, 0.01), 
    # Field Weights: Field 0 (Zt) is the KEY/Query field (1.0). Field 1 (Kw) is the passive VALUE field
    field_weights=[1.0, None],  
    field_names=['Zt_Query_Field', 'Kw_Stored_Field'],
    softmax_choice=pnl.WEIGHTED_AVG # needed for differentiability 
)

# Reference Memory Nodes for Wiring
em_z_input_node = em_memory.input_nodes[0]   # Receiver for the CONCRETE query (Zt)
em_kw_input_node = em_memory.input_nodes[1]  # Receiver for the ABSTRACT variable (Kw)

# Find the retrieved key node (output from Field 1: Kw_Stored_Field)
retrieved_kw_node = next(
    (n for n in em_memory.nodes if n.name in ('KEY [RETRIEVED]', 'RETRIEVED_Kw_Stored_Field')),
    None
)
if retrieved_kw_node is None:
    # Fallback: any retrieved node that is not value
    candidates = [n for n in em_memory.nodes if 'RETRIEVED' in n.name]
    retrieved_kw_node = candidates[-1] if candidates else None
if retrieved_kw_node is None:
    raise RuntimeError(f"Could not locate retrieved key node. Available: {[n.name for n in em_memory.nodes]}")


# ======================================================================
# --- MODEL COMPOSITION AND WIRING (to make sure Binding & Indirection work) ---
# ======================================================================

esbn_gru_comp = pnl.AutodiffComposition(name='ESBN-GRU Complete Model')
esbn_gru_comp.add_nodes([gru_controller, y_hat_mech, kw_mech, g_mech, hidden_probe, em_memory, z_embedding_mech])
esbn_gru_comp.add_node(z_embedding_mech, required_roles=pnl.NodeRole.INPUT)
esbn_gru_comp.add_node(y_hat_mech, required_roles=pnl.NodeRole.OUTPUT)


# Projection 1: GRU Hidden State -> Prediction (y_hat)
esbn_gru_comp.add_projection(
    sender=gru_sender_port, receiver=y_hat_mech,
    projection=pnl.MappingProjection(matrix=np.random.normal(scale=0.001, size=(HIDDEN_DIMENSION, OUTPUT_DIMENSION)))
)

# Projection 2: GRU Hidden State -> Key to Write (kw)
# a fix to an error: having a low weight scale (0.001) prevents immediate overflow in k_w 
esbn_gru_comp.add_projection(
    sender=gru_sender_port, receiver=kw_mech,
    projection=pnl.MappingProjection(
        matrix=np.random.normal(scale=0.001, size=(HIDDEN_DIMENSION, KEY_DIMENSION)) 
    )
)

# P3: GRU Hidden State -> Gate (g)
esbn_gru_comp.add_projection(
    sender=gru_sender_port, receiver=g_mech,
    projection=pnl.MappingProjection(matrix=np.random.normal(scale=0.01, size=(HIDDEN_DIMENSION, 1)))
)

# P4: GRU Hidden State -> Hidden Probe (for external PyTorch training access to see if it works)
esbn_gru_comp.add_projection(
    sender=gru_sender_port, receiver=hidden_probe,
    projection=pnl.MappingProjection(matrix=pnl.AUTO_ASSIGN_MATRIX)
)


# --- BINDING ---
# P5: Image Embedding (Zt_Input) -> Memory Zt_Query_Field
esbn_gru_comp.add_projection(sender=z_embedding_mech, receiver=em_z_input_node)

# P6: Key to Write (kw_mech) -> Memory Kw_Stored_Field (Binds abstract key to image value)
esbn_gru_comp.add_projection(sender=kw_mech, receiver=em_kw_input_node)


# ---INDIRECTION  ---
# P7: Retrieved Key (k_r + c_k) -> GRU Input (Closes the recurrent loop)
esbn_gru_comp.add_projection(sender=retrieved_kw_node, receiver=gru_controller.input_node)


# ======================================================================
# --- Where there is a TRAINING FAILURE ---
# ======================================================================
SEQ_LEN = 4
N_CLASSES = OUTPUT_DIMENSION
N_TRAIN = 24
N_EPOCHS = 5
LR = 1e-3
rng = np.random.default_rng(7)

# --- Generate dummy sequential training data ---
train_inputs = []
train_targets_seq = []
for _ in range(N_TRAIN):
    seq = [np.random.normal(scale=0.1, size=VALUE_DIMENSION) for _ in range(SEQ_LEN)]
    target_vec = np.zeros(OUTPUT_DIMENSION); target_vec[0] = 1.0 # Arbitrary target
    train_inputs.append([np.asarray(s, dtype=np.float32) for s in seq])
    train_targets_seq.append([np.zeros(OUTPUT_DIMENSION, dtype=np.float32) for _ in range(SEQ_LEN-1)] + [target_vec.astype(np.float32)])

print("\n--- ATTEMPTING PYNL NATIVE LEARN (EXPECTED FAILURE DEMO) ---")
y_hat_mech.function = pnl.Identity() 

esbn_gru_comp.show_graph(show_pytorch=True)

# This call will FAIL due to nested Compositions.
try:
    esbn_gru_comp.learn(
        inputs={z_embedding_mech: train_inputs},
        targets={y_hat_mech: train_targets_seq},
        epochs=N_EPOCHS,
        learning_rate=LR,
        # execution_mode=pnl.ExecutionMode.Python, # The problematic execution mode
        execution_mode=pnl.ExecutionMode.PyTorch, # The problematic execution mode
        target_time_steps=[SEQ_LEN - 1]
    )
except Exception as e:
    print(f"\n=== ACTUAL ERROR FROM PSYNEULINK ===")
    print(f"Exception Type: {type(e).__name__}")
    print(f"Exception Message: {e}")
    print(f"\nFull Traceback:")
    import traceback
    traceback.print_exc()

# # ======================================================================
# # --- MANUAL TRAINING LOOP (RESERVOIR COMPUTING) ---
# # ======================================================================
# print("\n--- MANUAL PYTORCH TRAINING LOOP (Simulating Learning) ---")
#
# # Define external readout layer for training
# linear = nn.Linear(HIDDEN_DIMENSION, OUTPUT_DIMENSION)
# optimizer = torch.optim.Adam(linear.parameters(), lr=LR)
#
# for epoch in range(N_EPOCHS):
#     total_loss = 0.0
#     correct = 0
#     for seq, tgts in zip(train_inputs, train_targets_seq):
#         # 1. Forward pass (PsyNeuLink)
#         esbn_gru_comp.run(inputs={z_embedding_mech: seq}, execution_mode=pnl.ExecutionMode.Python)
#         # 2. Extract Hidden State (Reservoir)
#         h_raw = np.array(hidden_probe.parameters.value.get(esbn_gru_comp), dtype=np.float32)
#         h_vec = np.nan_to_num(h_raw, nan=0.0, posinf=1.0, neginf=-1.0)
#         # 3. PyTorch Backward Pass (Readout)
#         x = torch.from_numpy(h_vec.astype(np.float32))
#         logits = linear(x)
#         label = int(np.argmax(tgts[-1]))
#         loss = F.cross_entropy(logits.view(1, -1), torch.tensor([label], dtype=torch.long))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss.item())
#         pred = int(torch.argmax(logits).item())
#         correct += int(pred == label)
#     print(f"Epoch {epoch+1}/{N_EPOCHS}: loss={total_loss/len(train_inputs):.4f}, acc={correct/len(train_inputs):.2f}")