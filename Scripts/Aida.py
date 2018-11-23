
# coding: utf-8

# In[1]:


import psyneulink as pnl
import numpy as np

# In[2]:


# ECin = pnl.KWTA(size=8, function=pnl.Linear)
# DG = pnl.KWTA(size=400, function=pnl.Linear)
# CA3 = pnl.KWTA(size=80, function=pnl.Linear)
# CA1 = pnl.KWTA(size=100, function=pnl.Linear)
# ECout = pnl.KWTA(size=8, function=pnl.Linear)
import psyneulink.core.components.functions.transferfunctions

ECin = pnl.TransferMechanism(size=8, function=psyneulink.core.components.functions.transferfunctions.Linear(), name='ECin')
DG = pnl.TransferMechanism(size=400, function=psyneulink.core.components.functions.transferfunctions.Logistic(), name='DG')
CA3 = pnl.TransferMechanism(size=80, function=psyneulink.core.components.functions.transferfunctions.Logistic(), name='CA3')
CA1 = pnl.TransferMechanism(size=100, function=psyneulink.core.components.functions.transferfunctions.Linear(), name='CA1')
ECout = pnl.TransferMechanism(size=8, function=psyneulink.core.components.functions.transferfunctions.Logistic(), name='ECout')


# In[3]:


def make_mask(in_features, out_features, connectivity):
    mask = np.zeros((in_features, out_features))
    rand = np.random.random(mask.shape)
    idxs = np.where(rand < connectivity)
    mask[idxs[0], idxs[1]] = 1
    return mask

def make_mat(in_features, out_features, lo, high, mask):
    w = np.random.uniform(lo ,high, size=(in_features, out_features))
    w = mask * w
    return w


# In[4]:


ECin_s, ECout_s, DG_s, CA3_s, CA1_s = 8, 8, 400, 80, 100


# In[5]:


mask_ECin_DG = make_mask(ECin_s, DG_s, 0.25)
mask_DG_CA3 = make_mask(DG_s, CA3_s, 0.05)
mask_ECin_CA3 = make_mask(ECin_s, CA3_s, 0.25)

mat_ECin_DG = make_mat(ECin_s, DG_s, 0.25, 0.75, mask_ECin_DG)
mat_DG_CA3 = make_mat(DG_s, CA3_s, 0.89, 0.91, mask_DG_CA3)
mat_ECin_CA3 = make_mat(ECin_s, CA3_s, 0.25, 0.75, mask_ECin_CA3)

mat_CA3_CA1 = make_mat(CA3_s, CA1_s, 0.25, 0.75, np.ones((CA3_s, CA1_s)))
mat_CA1_ECout = make_mat(CA1_s, ECout_s, 0.25, 0.75, np.ones((CA1_s, ECout_s)))
mat_ECin_CA1 = make_mat(ECin_s, CA1_s, 0.25, 0.75, np.ones((ECin_s, CA1_s)))


# In[6]:


ECin_to_DG=pnl.MappingProjection(matrix=mat_ECin_DG)
DG_to_CA3=pnl.MappingProjection(matrix=mat_DG_CA3)
ECin_to_CA3=pnl.MappingProjection(matrix=mat_ECin_CA3)
CA3_to_CA1=pnl.MappingProjection(matrix=mat_CA3_CA1)
CA1_to_ECout=pnl.MappingProjection(sender=CA1, receiver=ECout, matrix=mat_CA1_ECout)
ECin_to_CA1=pnl.MappingProjection(sender=ECin, receiver=CA1, matrix=mat_ECin_CA1)


# In[7]:


proc_ECin_DG = pnl.Process(pathway=[ECin, ECin_to_DG, DG], learning=pnl.ENABLED, learning_rate=0.2)
proc_ECin_CA3 = pnl.Process(pathway=[ECin, ECin_to_CA3, CA3], learning=pnl.ENABLED, learning_rate=0.2)
proc_DG_CA3 = pnl.Process(pathway=[DG, DG_to_CA3, CA3], learning=pnl.ENABLED, learning_rate=0)
proc_CA3_CA1 = pnl.Process(pathway=[CA3, CA3_to_CA1, CA1], learning=pnl.ENABLED, learning_rate=0.05)
proc_CA1_ECout = pnl.Process(pathway=[CA1, ECout], learning=pnl.ENABLED, learning_rate=0.02)
proc_ECin_CA1 = pnl.Process(pathway=[ECin, CA1], learning_rate=0.02)


# In[8]:


TSP = pnl.System(processes=[proc_ECin_DG, proc_ECin_CA3, proc_DG_CA3, proc_CA3_CA1, proc_CA1_ECout])
# MSP = pnl.System(processes=[proc_ECin_CA1, proc_CA1_ECout])


# In[9]:


TSP.show_graph()
assert True

# In[10]:


## Method for making input
def statistical():
    chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    sequence = ''
    letters = range(8)
    starters = range(0, 8, 2)
    enders = range(1, 8, 2)

    ## minus phase
    idx = np.random.randint(len(starters))
    s = starters[idx]
    e = enders[idx]
    minus_input, minus_target = np.zeros((8)), np.zeros((8))
    minus_input[s] = 1.0
    minus_target[e] = 1.0
    minus_target[s] = 0.9
    sequence += chars[s]
    sequence += chars[e]

    ## plus phase
    plus_input, plus_target = minus_target, np.zeros((8))
    plus_target[s] = 1
    plus_target[e] = 1

    return (minus_input, minus_target, plus_input, plus_target)


# In[ ]:


epochs = 100
for epoch in range(epochs):
    minus_x, minus_y, plus_x, plus_y = statistical()
    TSP.run(inputs={ECin:minus_x}, targets={ECout:minus_y})
    ## Running the above line of code causes weights to get too large
