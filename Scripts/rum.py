
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import psyneulink as pnl


# In[15]:
import psyneulink.core.components.functions.transferfunctions

nouns=['oak','pine','rose','daisy','canary','robin','salmon','sunfish']
relations=['is','has','can']
is_list=['living','living thing','plant','animal','tree','flower','bird','fish','big','green','red','yellow']
has_list=['roots','leaves','bark','branches','skin','feathers','wings','gills','scales']
can_list=['grow','move','swim','fly','breathe','breathe underwater','breathe air','walk','photosynthesize']
descriptors=[nouns,is_list,has_list,can_list]

truth_nouns=np.identity(len(nouns))

truth_is=np.zeros((len(nouns),len(is_list)))

truth_is[0,:]=[1,1,1,0,1,0,0,0,1,0,0,0]
truth_is[1,:]=[1,1,1,0,1,0,0,0,1,0,0,0]
truth_is[2,:]=[1,1,1,0,0,1,0,0,0,0,0,0]
truth_is[3,:]=[1,1,1,0,0,1,0,0,0,0,0,0]
truth_is[4,:]=[1,1,0,1,0,0,1,0,0,0,0,1]
truth_is[5,:]=[1,1,0,1,0,0,1,0,0,0,0,1]
truth_is[6,:]= [1,1,0,1,0,0,0,1,1,0,1,0]
truth_is[7,:]= [1,1,0,1,0,0,0,1,1,0,0,0]

truth_has=np.zeros((len(nouns),len(has_list)))

truth_has[0,:]= [1,1,1,1,0,0,0,0,0]
truth_has[1,:]= [1,1,1,1,0,0,0,0,0]
truth_has[2,:]= [1,1,0,0,0,0,0,0,0]
truth_has[3,:]= [1,1,0,0,0,0,0,0,0]
truth_has[4,:]= [0,0,0,0,1,1,1,0,0]
truth_has[5,:]= [0,0,0,0,1,1,1,0,0]
truth_has[6,:]= [0,0,0,0,0,0,0,1,1]
truth_has[7,:]= [0,0,0,0,0,0,0,1,1]

truth_can=np.zeros((len(nouns),len(can_list)))

truth_can[0,:]= [1,0,0,0,0,0,0,0,1]
truth_can[1,:]= [1,0,0,0,0,0,0,0,1]
truth_can[2,:]= [1,0,0,0,0,0,0,0,1]
truth_can[3,:]= [1,0,0,0,0,0,0,0,1]
truth_can[4,:]= [1,1,0,1,1,0,1,1,0]
truth_can[5,:]= [1,1,0,1,1,0,1,1,0]
truth_can[6,:]= [1,1,1,0,1,1,0,0,0]
truth_can[7,:]= [1,1,1,0,1,1,0,0,0]

truths=[[truth_nouns],[truth_is],[truth_has],[truth_can]]

#dict_is={'oak':truth_is[0,:],'pine':truth_is[1,:],'rose':truth_is[2,:],'daisy':truth_is[3,:],'canary':truth_is[4,:],'robin':truth_is[5,:],'salmon':truth_is[6,:],'sunfish':truth_is[7,:]}



# In[16]:


def gen_input_vals (nouns,relations):
    X_1=np.identity(len(nouns))
    X_2=np.identity(len(relations))
    return(X_1,X_2)


# In[17]:


nouns_onehot,rels_onehot=gen_input_vals(nouns,relations)

r_1=np.shape(nouns_onehot)[0]
c_1=np.shape(nouns_onehot)[1]
r_2=np.shape(rels_onehot)[0]
c_2=np.shape(rels_onehot)[1]


# In[18]:


#gotta figure out how to make this PNL friendly (breathe deep, my dude. One thing at a time.)
#later, we want to be able to change our bias, but for now, we're going to stick with a hard-coded one.
def step(variable,params,context):
    if np.sum(variable)<.5:
        out=0
    else:
        out=1
    return(out)


# In[19]:


Step=pnl.UserDefinedFunction(custom_function=step,
                            default_variable=np.zeros(4))


# In[20]:


#we're on the part where we generalize this and apply it as the function for all the bins...
#we'd like to generalize this for size so we can consistently just call the one UDF, but specifying size, to remove
#redundancies and general clutter. lol

step_mech=pnl.ProcessingMechanism(function=pnl.UserDefinedFunction(custom_function=step, default_variable=np.zeros(4)),
                                size=4,
                               name='step_mech')


# In[21]:


nouns_in = pnl.TransferMechanism(name="nouns_input",default_variable=np.zeros(r_1))

rels_in = pnl.TransferMechanism(name="rels_input",default_variable=np.zeros(r_2))

h1 = pnl.TransferMechanism(name="hidden_nouns",
                           size=8,
                           function=psyneulink.core.components.functions.transferfunctions.Logistic)

h2 = pnl.TransferMechanism(name="hidden_mixed",
                           size=15,
                           function=psyneulink.core.components.functions.transferfunctions.Logistic)

out_sig_I = pnl.TransferMechanism(name="sig_outs_I",
                                  size=len(nouns),
                                  function=psyneulink.core.components.functions.transferfunctions.Logistic)

out_sig_is = pnl.TransferMechanism(name="sig_outs_is",
                                   size=len(is_list),
                                   function=psyneulink.core.components.functions.transferfunctions.Logistic)


out_sig_has = pnl.TransferMechanism(name="sig_outs_has",
                                    size=len(has_list),
                                    function=psyneulink.core.components.functions.transferfunctions.Logistic)


out_sig_can = pnl.TransferMechanism(name="sig_outs_can",
                                    size=len(can_list),
                                    function=psyneulink.core.components.functions.transferfunctions.Logistic)

#biases
bh1 = pnl.TransferMechanism(name="bias_hidden_nouns",
                           default_variable=np.zeros(8))

bh2 = pnl.TransferMechanism(name="bias_hidden_mixed",
                           default_variable=np.zeros(15))

bosI = pnl.TransferMechanism(name="bias_osI",
                           default_variable=np.zeros(len(nouns)))

bosi = pnl.TransferMechanism(name="bias_osi",
                            default_variable=np.zeros(len(is_list)))

bosh = pnl.TransferMechanism(name="bias_osh",
                           default_variable=np.zeros(len(has_list)))

bosc = pnl.TransferMechanism(name="bias_osc",
                           default_variable=np.zeros(len(can_list)))

#later, we'll change the out_bin_x functions to a UDF that does a step function.
# out_bin_I = pnl.TransferMechanism(name="binary_outs_I",
#                                  size=len(nouns),
#                                  function=pnl.Linear)
#
# out_bin_is = pnl.TransferMechanism(name="binary_outs_is",
#                                  size=len(is_list),
#                                  function=pnl.Linear)
#
# out_bin_has = pnl.TransferMechanism(name="binary_outs_has",
#                                  size=len(has_list),
#                                  function=pnl.Linear)
#
# out_bin_can = pnl.TransferMechanism(name="binary_outs_can",
#                                  size=len(can_list),
#                                  function=pnl.Linear)

#we'll need to add in biases too. That will come later.


# In[22]:


#I want to put in a mapping projection that just ensures all our weight matrices between sigs and bins is I.

mapII=pnl.MappingProjection(matrix=np.eye(len(nouns)),
                          name="mapII"
                          )

mapIi=pnl.MappingProjection(matrix=np.eye(len(is_list)),
                          name="mapIi"
                          )

mapIh=pnl.MappingProjection(matrix=np.eye(len(has_list)),
                          name="mapIh"
                          )

mapIc=pnl.MappingProjection(matrix=np.eye(len(can_list)),
                          name="mapIc"
                          )


# In[23]:


#This is where we build the processes.
p11=pnl.Process(pathway=[nouns_in,h1,h2],
               learning=pnl.LEARNING)

p12=pnl.Process(pathway=[rels_in,h2],
               learning=pnl.LEARNING)

p21=pnl.Process(pathway=[h2,out_sig_I],
               learning=pnl.LEARNING)

p22=pnl.Process(pathway=[h2,out_sig_is],
               learning=pnl.LEARNING)

p23=pnl.Process(pathway=[h2,out_sig_has],
               learning=pnl.LEARNING)

p24=pnl.Process(pathway=[h2,out_sig_can],
               learning=pnl.LEARNING)


# In[24]:


#These are the processes that transform sigs to bins
#
# p31=pnl.Process(pathway=[out_sig_I,
#                          mapII,
#                          out_bin_I],
#                learning=pnl.LEARNING
#                )
#
# p32=pnl.Process(pathway=[out_sig_is,
#                          mapIi,
#                          out_bin_is],
#                learning=pnl.LEARNING
#                )
#
# p33=pnl.Process(pathway=[out_sig_has,
#                          mapIh,
#                          out_bin_has],
#                learning=pnl.LEARNING
#                )
#
# p34=pnl.Process(pathway=[out_sig_can,
#                          mapIc,
#                          out_bin_can],
#                learning=pnl.LEARNING
#                )


# In[25]:


#Bias processes go here

bp1=pnl.Process(pathway=[bh1,h1],
                learning=pnl.LEARNING
               )

bp2=pnl.Process(pathway=[bh2,h2],
               learning=pnl.LEARNING
               )

bposI=pnl.Process(pathway=[bosI,out_sig_I],
               learning=pnl.LEARNING
                 )

bposi=pnl.Process(pathway=[bosi,out_sig_is],
                learning=pnl.LEARNING
                 )

bposh=pnl.Process(pathway=[bosh,out_sig_has],
                 learning=pnl.LEARNING
                 )

bposc=pnl.Process(pathway=[bosc,out_sig_can],
                 learning=pnl.LEARNING
                 )


# In[117]:


#This is where we put them all into a system

rumel_sys=pnl.System(processes=[p11,
                                bp1,
                                p12,
                                bp2,
                                p21,
                                bposI,
                                p22,
                                bposi,
                                p23,
                                bposh,
                                p24,
                                bposc,
                                # p31,
                                # p32,
                                # p33,
                                # p34
                                ])

rumel_sys.show_graph(show_learning=True)

# In[26]:


#This is where we build multiple systems that separate the learning components from the non-learning components.
#This only compiles when the one above it does not.
#This might be a good bug to report...

#Additionally, for some reason this system is a clone of the one above, regardless of whether or not we include
#the one below. If the p3x processes are defined at all, they are automatically included in the system.
#What is going on here?
# rumel_sys2a=pnl.System(processes=[p11,
#                                 bp1,
#                                 p12,
#                                 bp2,
#                                 p21,
#                                 bposI,
#                                 p22,
#                                 bposi,
#                                 p23,
#                                 bposh,
#                                 p24,
#                                 bposc])


# # In[147]:
#
#
# rumel_sys2b=pnl.System(processes=[
#                                 p31,
#                                 p32,
#                                 p33,
#                                 p34])
#
#
# # In[27]:
#
#
# rumel_sys2a.show_graph(output_fmt='jupyter')
#
#
# # In[97]:
#
#
# #so far, so hoopy. What we want is to not enable learning on our binaries. Just on our sigs.
#
#
# # In[100]:
#
#
# for noun in range(len(nouns)):
#     for rel_out in range (3):
#         rumel_sys.run(inputs={nouns_in: nouns_onehot[noun],
#                               rels_in: rels_onehot[rel_out],
#                               bh1: np.zeros(8),
#                               bh2: np.zeros(15),
#                               bosI: np.zeros(len(nouns)),
#                               bosi: np.zeros(len(is_list)),
#                               bosh: np.zeros(len(has_list)),
#                               bosc: np.zeros(len(can_list)),
#                              },
#                       targets={out_bin_I: nouns_onehot[noun],
#                                out_bin_is: truth_is[noun],
#                                out_bin_has: truth_has[noun],
#                                out_bin_can: truth_can[noun]
#                               }
#                      )
# #What we can do here, is build our inputs into a nested for loop
#
#
# # In[103]:
#
#
for noun in range(len(nouns)):
    for rel_out in range (3):
        rumel_sys.run(inputs={nouns_in: nouns_onehot [noun],
                              rels_in: rels_onehot [rel_out],
                              bh1: np.zeros(8),
                              bh2: np.zeros(15),
                              bosI: np.zeros(len(nouns)),
                              bosi: np.zeros(len(is_list)),
                              bosh: np.zeros(len(has_list)),
                              bosc: np.zeros(len(can_list)),
                             },
                      targets={out_sig_I: nouns_onehot [noun],
                               out_sig_is: truth_is [noun],
                               out_sig_has: truth_has [noun],
                               out_sig_can: truth_can [noun]
                              }
                     )
# #What we can do here, is build our inputs into a nested for loop
#
#
# # So far, what I have left to do includes:
# #
# # getting a threshold function up and running.
# #     See "step", defined below. All we need to do is make it PNL friendly. :D
# #
# #     # This is done
# #
# # also want to make sure that the weight matrices from sigs to bins is I and cannot learn
# #
# #     # Setting them to I is done. But, if we turn off learning on them, we can't run the system at all, because nothing that can learn projects to a target mechanism. It doesn't matter where we set the targets. If we set targets at sigs, it says, sigs don't project to target mechanism (target mechs are getting attached to bins). If we set targets for bins, it says targets don't project to target mechanims (target mechs are attached to bins, but bins can't learn).
# #
# #     # I think I know a work-around on this that doesn't require we make a whole new system. We use the same setup that we used for the duck-rabbit model, where we map the sig outputs to labels, which are 1 or 0, using the previously defined step function, and get the MSE out of that.
# #
# #     # I think it's okay for us to still try to set up multiple systems with overlapping mechanisms...
# #         # Information on our capacity to do this should be available in "compositions" but github pages is down right now. :/
# #
# # figure out how to turn on learning for some mechs and not for others without losing previously learned weights, either by avoiding reinitializing the system or by saving previously learned weights. :)
# #     this might be something to talk to Jon about...
# #
# # In order to do this, we will definitely* need to figure out how to put them into different systems and run the whole thing together?
# #
# # Actually seeing how it performs?
# #
# # Is that it?
# #
# # I need to get my github shit working, too, so I can work in devel and the other branches. :| Still, good progress for today, I think. :)
