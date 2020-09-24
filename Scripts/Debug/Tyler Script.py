#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import pickle
from psyneulink import *
warnings.filterwarnings('ignore')


# In[63]:


def build_rl_model_dynamic():
    #RL Model - Dynamic
    word_w_pos=4
    word_w_neg=-4
    pic_w_pos=4
    pic_w_neg=-4
    word_bias=-2
    size_out_pos=1.9
    size_out_neg=1.9

    word_input = ProcessingMechanism(name='WORD INPUT'   ,size=4)
    pic_input  = ProcessingMechanism(name='PICTURE INPUT',size=4)

    word_rep = ProcessingMechanism(name='WORD REP', size=4, function=Logistic(bias=word_bias))
    pic_rep  = ProcessingMechanism(name='PIC REP',  size=4, function=Logistic(bias=-1))

    wwp = word_w_pos
    wwn = word_w_neg
    pwp = pic_w_pos
    pwn = pic_w_neg
    word_input_to_word_rep_wts = np.array([  [  wwp,  wwn,  wwn,  wwn],
                                             [  wwn,  wwp,  wwn,  wwn],
                                             [  wwn,  wwn,  wwp,  wwn],
                                             [  wwn,  wwn,  wwn,  wwp]]).T
    pic_input_to_pic_rep_wts =   np.array([  [  pwp,  pwn,  pwn,  pwn],
                                             [  pwn,  pwp,  pwn,  pwn],
                                             [  pwn,  pwn,  pwp,  pwn],
                                             [  pwn,  pwn,  pwn,  pwp]]).T
    #RL Model - Dynamic

    word_rep_to_hidden_cat_wts = np.array([  [ 1, 1,-1,-1],
                                             [-1,-1, 1, 1]]).T*2
    pic_rep_to_hidden_cat_wts = np.array( [  [ 1, 1,-1,-1],
                                             [-1,-1, 1, 1]]).T*2


    word_rep_to_hidden_sz_wts = np.array( [  [ 1.67,-1.67,    0,    0],
                                             [-1.67, 1.67,    0,    0],
                                             [    0,    0, 1.67,-1.67],
                                             [    0,    0,-1.67, 1.67]]).T
    pic_rep_to_hidden_sz_wts = np.array(  [  [ 1.67,-1.67,   0,   0],
                                             [-1.67, 1.67,   0,   0],
                                             [   0,   0, 1.67,-1.67],
                                             [   0,   0,-1.67, 1.67]]).T

    cat_hidden = ProcessingMechanism(name='CATEGORY HIDDEN', size=2, function=Logistic(bias=-4))
    sz_hidden  = ProcessingMechanism(name='SIZE HIDDEN', size=4, function=Logistic(bias=-4))

    wc_pathway = [word_input, word_input_to_word_rep_wts, word_rep, word_rep_to_hidden_cat_wts, cat_hidden]
    ws_pathway = [word_input, word_input_to_word_rep_wts, word_rep, word_rep_to_hidden_sz_wts , sz_hidden ]
    pc_pathway = [pic_input , pic_input_to_pic_rep_wts  , pic_rep , pic_rep_to_hidden_cat_wts , cat_hidden]
    ps_pathway = [pic_input , pic_input_to_pic_rep_wts  , pic_rep , pic_rep_to_hidden_sz_wts  , sz_hidden ]

    cat_hidden_to_verbal_output_wts = np.array([[ 1,-1],
                                                [-1, 1]]).T*2
    sop, son = size_out_pos, size_out_neg
    sz_hidden_to_manual_output_wts = np.array( [[ sop,-sop, sop,-sop],
                                                [-son, son,-son, son]]).T
    #RL Model - Dynamic
    verbal_output = ProcessingMechanism(name='VERBAL OUTPUT',size=2, function=Logistic(bias=-4))
    manual_output = ProcessingMechanism(name='MANUAL OUTPUT',size=2, function=Logistic(bias=-2))

    #verbal_decision = DDM(input_format=ARRAY,function=DriftDiffusionAnalytical())

    cn_pathway = [cat_hidden, cat_hidden_to_verbal_output_wts, verbal_output]
    sb_pathway = [sz_hidden, sz_hidden_to_manual_output_wts, manual_output]

    task_input = ProcessingMechanism(name='TASK INPUT',size=6)
    #wcn, p, s(both), s(an), s(in), b
    #RL Model - Dynamic
    wb = -word_bias
    task_input_to_word_rep_wts   = np.array(   [[  wb,   0,   0,   0,   0,   0],
                                                [  wb,   0,   0,   0,   0,   0],
                                                [  wb,   0,   0,   0,   0,   0],
                                                [  wb,   0,   0,   0,   0,   0]]).T
    task_input_to_pic_rep_wts    = np.array(   [[   0,   1,   0,   0,   0,   0],
                                                [   0,   1,   0,   0,   0,   0],
                                                [   0,   1,   0,   0,   0,   0],
                                                [   0,   1,   0,   0,   0,   0]]).T
    task_input_to_hidden_cat_wts = np.array(   [[   4,   0,   0,   0,   0,   0],
                                                [   4,   0,   0,   0,   0,   0]]).T
    task_input_to_hidden_sz_wts = np.array(    [[   0,   0,   1,   2,   0,   0],
                                                [   0,   0,   1,   2,   0,   0],
                                                [   0,   0,   1,   0,   2,   0],
                                                [   0,   0,   1,   0,   2,   0]]).T
    task_input_to_verbal_output_wts = np.array([[   4,   0,   0,   0,   0,   0],
                                                [   4,   0,   0,   0,   0,   0]]).T
    task_input_to_manual_output_wts = np.array([[   0,   0,   0,   0,   0,   4],
                                                [   0,   0,   0,   0,   0,   4]]).T
    taskw_pathway = [task_input, task_input_to_word_rep_wts     , word_rep]
    taskp_pathway = [task_input, task_input_to_pic_rep_wts      , pic_rep]
    taskc_pathway = [task_input, task_input_to_hidden_cat_wts   , cat_hidden]
    tasks_pathway = [task_input, task_input_to_hidden_sz_wts    , sz_hidden]
    taskn_pathway = [task_input, task_input_to_verbal_output_wts, verbal_output]
    taskb_pathway = [task_input, task_input_to_manual_output_wts, manual_output]
    #RL Model - Dynamic
    model = Composition(name='RL Model')
    model.add_linear_processing_pathway(wc_pathway)
    model.add_linear_processing_pathway(ws_pathway)
    model.add_linear_processing_pathway(pc_pathway)
    model.add_linear_processing_pathway(ps_pathway)
    model.add_linear_processing_pathway(cn_pathway)
    model.add_linear_processing_pathway(sb_pathway)
    model.add_linear_processing_pathway(taskw_pathway)
    model.add_linear_processing_pathway(taskp_pathway)
    model.add_linear_processing_pathway(taskc_pathway)
    model.add_linear_processing_pathway(tasks_pathway)
    model.add_linear_processing_pathway(taskn_pathway)
    model.add_linear_processing_pathway(taskb_pathway)
    #RL Model - Dynamic
    verbal_ddm_wts, manual_ddm_wts = np.array([[1,0]]).T, np.array([[1,0]]).T
    verbal_output_shifted = ProcessingMechanism(function=Linear)
    manual_output_shifted = ProcessingMechanism(function=Linear)
    verbal_ddm = DDM(name='VERBAL DDM',
                     function=DriftDiffusionIntegrator(noise=1.0,
                                                       starting_point=0,
                                                       threshold=0.5,
                                                       rate=2.6,
                                                       time_step_size=0.05
                                                      ),
                     #reset_stateful_function_when=AtTrialStart()
                    )
    manual_ddm = DDM(name='MANUAL DDM',
                     function=DriftDiffusionIntegrator(noise=1.0,
                                                       starting_point=0,
                                                       threshold=0.5,
                                                       rate=2.6,
                                                       time_step_size=0.05
                                                      ),
                     #reset_stateful_function_when=AtTrialStart()
                    )
    def reward_function(x,t0=0.48):
        # verbal dv, verbal rt, manual dv, manual rt
        try:
            return (x[0,0]>0).astype(float)*(x[0,2]>0).astype(float)/(np.max([x[0,1],x[0,3]])+t0)
        except:
            return float(x[0][0,0]>0)*float(x[0][0,2]>0)/(np.max([x[0][0,1],x[0][0,3]])+t0)
    verbal_dv = ProcessingMechanism(size=1,name='VERBAL DV')
    verbal_rt = ProcessingMechanism(size=1,name='VERBAL RT')
    manual_dv = ProcessingMechanism(size=1,name='MANUAL DV')
    manual_rt = ProcessingMechanism(size=1,name='MANUAL RT')
    reward_module = ProcessingMechanism(size=4,name='reward_module',function=reward_function)

    model.add_linear_processing_pathway([verbal_output,verbal_ddm_wts,verbal_ddm])
    model.add_linear_processing_pathway([manual_output,manual_ddm_wts,manual_ddm])
    model.add_nodes([verbal_dv,verbal_rt,manual_dv,manual_rt,reward_module])
    model.add_projection(projection=np.array([[1]]),sender=verbal_ddm.output_ports[DECISION_VARIABLE],
                         receiver=verbal_dv,name='VERBAL DDM DV PROJ')
    model.add_projection(projection=np.array([[1]]),sender=verbal_ddm.output_ports[RESPONSE_TIME],
                         receiver=verbal_rt,name='VERBAL DDM RT PROJ')
    model.add_projection(projection=np.array([[1]]),sender=manual_ddm.output_ports[DECISION_VARIABLE],
                         receiver=manual_dv,name='MANUAL DDM DV PROJ')
    model.add_projection(projection=np.array([[1]]),sender=manual_ddm.output_ports[RESPONSE_TIME],
                         receiver=manual_rt,name='MANUAL DDM RT PROJ')
    model.add_projection(projection=np.array([[1,0,0,0]]),sender=verbal_dv,
                         receiver=reward_module)
    model.add_projection(projection=np.array([[0,1,0,0]]),sender=verbal_rt,
                         receiver=reward_module)
    model.add_projection(projection=np.array([[0,0,1,0]]),sender=manual_dv,
                         receiver=reward_module)
    model.add_projection(projection=np.array([[0,0,0,1]]),sender=manual_rt,
                         receiver=reward_module)

    model.scheduler.add_condition(reward_module,All(WhenFinished(verbal_ddm),WhenFinished(manual_ddm)))

    #RL Model - Dynamic
    verbal_dv.log.set_log_conditions(VALUE)
    manual_dv.log.set_log_conditions(VALUE)
    global t
    t = 0
    def callback():
        global t
        print('---------------')
        print(f'Timestep {t}.')
        #Custom execution ID, otherwise the is_finished method always returns False
        print(f'Verbal DDM Decision Variable: {verbal_dv.output_values[0].flatten()[0]:5.2f} is_finished: {verbal_ddm.is_finished(context=Context(execution_id="debug"))}')
        print(f'Manual DDM Decision Variable: {manual_dv.output_values[0].flatten()[0]:5.2f} is_finished: {manual_ddm.is_finished(context=Context(execution_id="debug"))}')
        t += 1
    return model, task_input, word_input, pic_input, callback


# In[64]:


model, task_input, word_input, pic_input, callback = build_rl_model_dynamic()
model.show_graph(show_learning=True,show_controller=True)


# In[65]:


stims = {task_input:[[0,1,1,0,0,1] ],
         word_input:[[0,0,1,0] ],
         pic_input: [[1,0,0,0] ]}

model.reset()
#Add custom execution id for the context so the "is_finished" method can be printed in the callback
_ = model.run(stims,call_after_time_step=callback,context=Context(execution_id='debug'))
model.results


# In[66]:


model.log.print_entries(display=[TIME,VALUE])
model.log.clear_entries()

