# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:47:38 2017

@author: lauraab
"""

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import *

#helpful commands! .show()   .excute()
color_word_combine = TransferMechanism(name='color_word_combine', function = Linear()) #default_variable = [1], #JON HOW TO MAKE THIS A SUBTRACTION?
decision_DDM = DDM(function=BogaczEtAl(drift_rate = (1.0), threshold = (1.0, ControlProjection(function = Linear)), starting_point = 0.0)) #if this DDM is using output of color_word_combine do I need to specify drift rate here also? do I need to make reference to the variable
                    
#color process 
#color input       ## ?Mapping Projection    ORIGIN
color_input_linear_transfer_mechanism = TransferMechanism(default_variable = [1], function = Linear(), name = "Color_Input") #1 means red
#color transfer mechanism, EVC regulates bias   ##TransferMechanism
color_logistic_transfer_mechanism = TransferMechanism(function=Logistic(gain = 1, bias = ControlProjection(function = Linear)), name = "Color_Transfer") #JON SHOULD I CHANGE GAIN OF COLOR RELATIVE TO WORD PROCESSES?  #range 0 to 1? 
color_process = process(pathway=[color_input_linear_transfer_mechanism, color_logistic_transfer_mechanism, color_word_combine])

#word process 
#word input        ## ?Mapping Projection    ORIGIN 
word_input_linear_transfer_mechanism = TransferMechanism(default_variable = [1], function=Linear(), name = "Word_Input") #1 means red, -1 means green   #DID I NEED AN OUTPUT PROJECTION HERE?
word_logistic_transfer_mechanism = TransferMechanism(function = Logistic(gain = 1, bias = ControlProjection(function = Linear)), name = "Word_Transfer") #?bias = control_signal_bias   #range 0 to 1? 
word_process = process(pathway=[word_input_linear_transfer_mechanism, word_logistic_transfer_mechanism, color_word_combine, decision_DDM]) #, color_word_combine
#word_process = process(pathway=[word_input_linear_transfer_mechanism, word_logistic_transfer_mechanism, color_word_combine]) #, color_word_combine

#added to try to get control? 
Reward = TransferMechanism(name='Reward')
RewardProcess = process(default_variable=[1], pathway=[(Reward, 1)], name = 'RewardProcess')
#mySystem = system(processes = [color_process, word_process])
mySystem = system(processes = [color_process, word_process, RewardProcess], controller = EVCMechanism, enable_controller = True, monitor_for_control=[Reward, (DDM_DECISION_VARIABLE,1,1), (DDM_RESPONSE_TIME, -1, 1)], name='Stroop Model') #output, difference between color / word unit activation  #output_values attribute?
mySystem.execute(input=[[1],[-1],[1]])

mySystem.controller.show()
mySystem.show()
# mySystem.show_graph()

#color_process.run(inputs={color_input_linear_transfer_mechanism:[1]})
# mySystem.execute(input=[[.1],[-0.1]])[0]


#output_values for color, separate for word process, and then combine?
# decision_DDM = DDM(function=BogaczEtAl(drift_rate= difference_output*, threshold=(1, ControlProjection)) #or could use params = {DRIFT_RATE:(0.2, ControlProjection), STARTING_POINT:-0.5}
#DDM, drift rate = output, EVC regulates threshold 
# DDM_PROBABILITY_UPPER_THRESHOLD      # what does this mean?: if time_scale is TimeScale.TIME_STEP, this is `None;


#EVC, objective function = error / RT : DDM_DECISION_VARIABLEexp1, DDM_RESPONSE_TIMEexp-1
#   ControlSignal
#   ControlProjection

#EVC.EVCMechanism.EVCMechanism(prediction_mechanism_type = IntegratorMechanism, prediction_mechanism_params = *, monitor_for_control = (color['bias'],word['bias'],DDM['threshold']))**  (objectiv function)
# function=ControlSignalGridSearch, value_function = error / RT *** or value_function, outcome_function = ***, cost_function = ***, combine_outcome_and_cost_function = ***
#  * want it to look only 1 step back
#  ** does monitor_for_control care only about what you need value for, or is this setting the mapping projections for EVC? 
#  *** this part of documentation was confusing as to what is default / non-default and what I need to specify 
#  *** unclear if arguments such as monitor_for_control should be specified in an EVC mechanism's parameters or in a system's parameters.. (see "Examples" on EVC page)


#EVC comments on documentation
#how do I import what is required for EVC? in general in documentation how do I know which pieces of PsyNeuLink to import? #copy from other scripts, if not recognized 
#not clear to me syntax: monitor_for_control (List[OutputState or Tuple[OutputState, list or 1d np.array, list or 1d np.array]] : default MonitoredOutputStatesOptions.PRIMARY_OUTPUT_STATES)
# class EVC.EVCMechanism.ControlSignalCosts; "An enumeration" what does this mean? a number?
#when EVC appears in a yellow box but is also part of an example command it isn't clear if there should or shouldn't be a space in that command, noticed this in code
#for example this line looks like: my_linear_ transfer _mechanism =  Transfer Mechanism (function=Linear), then when I copy and paste its my_linear_transfer_mechanism = TransferMechanism(function=Linear)






###### NOTES BEFORE 2/13/17
#color process 
#color input       ## ?Mapping Projection    ORIGIN
    #?e.g. color_input_linear_transfer_mechanism = TransferMechanism(function=Linear)
#color transfer mechanism, EVC regulates bias   ##TransferMechanism
    #e.g. color_logistic_transfer_mechanism = TransferMechanism(function=Logistic(gain=1.0, bias=-4) #?bias = control_signal_bias
    #range 0 to 1? 
    #output_values for color, separate for word process, and then combine?

#word process 
#word input        ## ?Mapping Projection    ORIGIN 
#word transfer mechanism, EVC regulates bias    ##TransferMechanism
#output, difference between color / word unit activation  #output_values attribute?
 
#DDM, drift rate = output, EVC regulates threshold 
# decision_DDM = DDM(function=BogaczEtAl(drift_rate= difference_output*, threshold=(1, ControlProjection)) #or could use params = {DRIFT_RATE:(0.2, ControlProjection), STARTING_POINT:-0.5}
# DDM_PROBABILITY_UPPER_THRESHOLD      # what does this mean?: if time_scale is TimeScale.TIME_STEP, this is `None;


#EVC, objective function = error / RT : DDM_DECISION_VARIABLEexp1, DDM_RESPONSE_TIMEexp-1
#   ControlSignal
#   ControlProjection

#EVC.EVCMechanism.EVCMechanism(prediction_mechanism_type = IntegratorMechanism, prediction_mechanism_params = *, monitor_for_control = (color['bias'],word['bias'],DDM['threshold']))**  (objectiv function)
# function=ControlSignalGridSearch, value_function = error / RT *** or value_function, outcome_function = ***, cost_function = ***, combine_outcome_and_cost_function = ***
#  * want it to look only 1 step back
#  ** does monitor_for_control care only about what you need value for, or is this setting the mapping projections for EVC? 
#  *** this part of documentation was confusing as to what is default / non-default and what I need to specify 
#  *** unclear if arguments such as monitor_for_control should be specified in an EVC mechanism's parameters or in a system's parameters.. (see "Examples" on EVC page)


#EVC comments on documentation
#how do I import what is required for EVC? in general in documentation how do I know which pieces of PsyNeuLink to import? #copy from other scripts, if not recognized 
#not clear to me syntax: monitor_for_control (List[OutputState or Tuple[OutputState, list or 1d np.array, list or 1d np.array]] : default MonitoredOutputStatesOptions.PRIMARY_OUTPUT_STATES)
# class EVC.EVCMechanism.ControlSignalCosts; "An enumeration" what does this mean? a number?
#when EVC appears in a yellow box but is also part of an example command it isn't clear if there should or shouldn't be a space in that command, noticed this in code
#for example this line looks like: my_linear_ transfer _mechanism =  Transfer Mechanism (function=Linear), then when I copy and paste its my_linear_transfer_mechanism = TransferMechanism(function=Linear)

