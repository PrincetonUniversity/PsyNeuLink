#Creating a net in PNL:

import psyneulink as pnl
import numpy as np

import psyneulink.core.components.functions.transferfunctions

input_layer=pnl.TransferMechanism(size=2,
    name="input layer")

#### THIS IS THE PART THAT NEEDS TO BE TAKEN OUT OF THE STUDENT VERSION! ####
#bias_mech_h=pnl.TransferMechanism(
#    function=pnl.Logistic(gain=.3),
#    size=2,
#    name="bias h")

#bias_mech_out=pnl.TransferMechanism(
#    function=pnl.Logistic(gain=.3),
#    size=1,
#    name="bias out")

hidden_layer=pnl.TransferMechanism(size=2,
                                   function=psyneulink.core.components.functions.transferfunctions.Logistic,
                                   name="hidden layer"
                                   )

proj=pnl.MappingProjection(matrix=(np.random.rand(2,2)),name="proj 1")
proj_h=pnl.MappingProjection(matrix=(np.random.rand(2,1)),name="proj 2")

#p_b=pnl.MappingProjection(matrix=.1*(np.random.rand(2,2)),name="proj bias 1")
#p_b_o=pnl.MappingProjection(matrix=.1*(np.random.rand(1,1)),name="proj bias 2")

output_layer=pnl.TransferMechanism(size=1, function=psyneulink.core.components.functions.transferfunctions.Logistic, name="output layer")

#bias_h=pnl.Process(pathway=[bias_mech_h,p_b,hidden_layer],learning=pnl.ENABLED)
#bias_out=pnl.Process(pathway=[bias_mech_out,p_b_o,output_layer],learning=pnl.ENABLED)

net3l=pnl.Process(pathway=[input_layer,proj,hidden_layer,proj_h,output_layer],learning=pnl.ENABLED)

sys3l=pnl.System(processes=[
    #bias_h,
    #bias_out,
    net3l],learning_rate=8)
#### AFTER THIS PART IS FINE #####

sys3l.show_graph(output_fmt = 'jupyter')

trials=4000
X=np.array([[1,1],[1,0],[0,1],[0,0]])
#X=[[1,1,1],[1,0,1],[0,1,1],[0,0,1]]
b_h_ins=[[1,1],[1,1],[1,1],[1,1]]
b_o_ins=[[1],[1],[1],[1]]
AND_labels_pnl=[[1],[0],[0],[0]]
OR_labels_pnl= [[1],[1],[1],[0]]
XOR_labels_pnl=[[0],[1],[1],[0]]

def print_out():
    print('.')

labels=XOR_labels_pnl
J=sys3l.run(inputs={input_layer: X,
                    #bias_mech_h:b_h_ins,
                    #bias_mech_out:b_o_ins
                   },
            call_after_trial=print_out,
            targets={output_layer: labels},
            num_trials=trials)

rat = int(len(J) / 4)
J=np.reshape(J,(rat,4))
x=np.arange(0,rat,1)
# plt.plot(x,J[:,0],x,J[:,1],x,J[:,2],x,J[:,3])
# plt.title('Outputs of 3 layer PNL network as a function of trials')
# plt.show()

MSE=[]
for i in range(np.shape(J)[0]):
    diff = J[i, :] - np.array(labels).T[:]
    MSE = np.append(MSE, (1 / 4) * (np.sum(diff**2)))
x=np.arange(0,np.shape(J)[0],1)
# plt.plot(x,MSE)
# plt.title('MSE of 3 layer net in PNL as a function of trials')
# plt.show

print('The last MSE was: ',MSE[-1])
