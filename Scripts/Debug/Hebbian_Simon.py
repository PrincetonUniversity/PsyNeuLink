import psyneulink as pnl
import numpy as np
import matplotlib.pyplot as plt
#sample Hebb

FeatureNames=['small','medium','large','red','yellow','blue','circle','rectangle','triangle']

# create a variable that corresponds to the size of our feature space
sizeF = len(FeatureNames)
small_red_circle = [1,0,0,1,0,0,1,0,0]
src = small_red_circle


Hebb_comp = pnl.Composition()

Hebb_mech=pnl.RecurrentTransferMechanism(
    size=sizeF,
    function=pnl.Linear,
    #integrator_mode = True,
    #integration_rate = 0.5,
    enable_learning = True,
    learning_rate = .1,
    name='Hebb_mech',
    #matrix=pnl.AutoAssociativeProjection,
    auto=0,
    hetero=0
    )

Hebb_comp.add_node(Hebb_mech)

Hebb_comp.execution_id = 1

# Use print_info to show numerical values and vis_info to show graphs of the changing values

def print_info():
    print('\nWeight matrix:\n', Hebb_mech.matrix.base, '\nActivity: ', Hebb_mech.value)


def vis_info():
  ub=1#np.amax(np.amax(np.abs(Hebb_mech.matrix.modulated)))
  lb=-ub
  plt.figure()
  plt.imshow(Hebb_mech.matrix.modulated,cmap='RdBu_r',vmin=lb,vmax=ub)
  plt.title('PNL Hebbian Matrix')
  plt.xticks(np.arange(0,sizeF),FeatureNames,rotation=35)
  plt.yticks(np.arange(0,sizeF),FeatureNames,rotation=35)

  plt.colorbar()
  plt.show()

  plt.figure()
  plt.stem(Hebb_mech.value[0])
  plt.title('Activation from Stimulus with PNL Hebbian Matrix')
  plt.xticks(np.arange(0,sizeF),FeatureNames,rotation=35)
  plt.xlabel('Feature')
  plt.ylabel('Activation')
  plt.show()

inputs_dict = {Hebb_mech:np.array(src)}
out=Hebb_comp.learn(num_trials=5,
      # call_after_trial=vis_info,
      inputs=inputs_dict)

print_info()
