function metalevel_model=trainEVOCModel(metalevel_model)

drift_rate_automatic=3;
drift_rate_controlled=1;
bias=0.5;
congruency=[true,false];

features=zeros(metalevel_model.nr_basic_features,1);

reward_rate=0.1;

nr_trials=10000;

for t=1:nr_trials
    
    if rand()<0.5
        congruent=true;
    else
        congruent=false;
    end
    
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end

    control_signals=metalevel_model.selectControlSignal(features);

    drift_rates=(drift_rate_controlled+control_signals(1))+...
        (drift_rate_automatic-control_signals(1))*direction_automatic;
    
    DDM_parameters.z=0.5+control_signals(2); %threshold
    DDM_parameters.c=1; %noise
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
    
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(control_signals)+...
        metalevel_model.reconfigurationCost(control_signals);
    
    metalevel_model=metalevel_model.learn(control_signals,features,U,cost);
    
    
end


end