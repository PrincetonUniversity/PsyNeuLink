
import EVC.*
import DDM.*

drift_rates=-1:0.1:1; %drift rate 
bias=[0.5];  %no bias

[p_error,allDTs,allRRs,allFinalRTs,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true)

probs(1,:) = (1-p_error);
probs(2,:) = p_error;
RTs = [allFinalRTs; allFinalRTs];       % use only mean RT for now

%% Stroop
clear
c=0:0.1:1; %intensity of the control signal that upregulates the controlled pathway (e.g., color naming)

drift_rate_automatic=1;
drift_rate_controlled=0.5;
bias=0.5;
congruency=[true,false];

for condition=1:length(congruency)
    congruent=congruency(condition);
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end
    
    drift_rates=c*drift_rate_controlled+(1-c)*drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,allFinalRTs,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true)
    
    error_rate(:,condition) = p_error;
    RTs(:,condition) = allFinalRTs;       % use only mean RT for now
    
end

figure()
plot(c,error_rate,'LineWidth',2),set(gca,'FontSize',18)
xlabel('Control Signal Intensity','FontSize',18)
ylabel('Error Rate in %','FontSize',18)
legend('Congruent Trials','Incongruent Trials')
title('Simulated Performance in Stroop Task','FontSize',20)

figure()
plot(c,RTs,'LineWidth',2),set(gca,'FontSize',18)
xlabel('Control Signal Intensity','FontSize',18)
ylabel('Reaction Time','FontSize',18)
legend('Congruent Trials','Incongruent Trials')
title('Simulated Performance in Stroop Task','FontSize',20)

%%
nr_features=1;
nr_control_signals=1;
metalevel_model=ModelOfEVOC(nr_features,nr_control_signals)

features=ones(nr_features,1); control_signals=ones(nr_control_signals,1);
metalevel_model.predictPerformance(features,control_signals)

VOC_samples=metalevel_model.sampleVOC(features,control_signals,1)

c=metalevel_model.selectControlSignal(features)


%% Compare the control signal intensity learned by the model to the optimal
%all trials are incongruent
clear,close all
nr_features=1;
nr_control_signals=1;

drift_rate_automatic=1;
drift_rate_controlled=1;
bias=0.5;
congruency=[true,false];

congruent=false;

if congruent
    direction_automatic=1;
else
    direction_automatic=-1;
end

metalevel_model=ModelOfEVOC(nr_features,nr_control_signals)

features=ones(nr_features,1); control_signals=ones(nr_control_signals,1);
metalevel_model.predictPerformance(features,control_signals)


signal_intensities=linspace(0,5,100);
reward_rate=0.1;
for i=1:length(signal_intensities)
    c=signal_intensities(i)
    drift_rates=c*drift_rate_controlled+drift_rate_automatic*direction_automatic;
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
        
    EU=(1-p_error)*1+p_error*(-1);
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    VOC(i)=EU-cost;
end

figure()
plot(signal_intensities,VOC)
[val,pos]=max(VOC)
c_star=signal_intensities(pos)

nr_trials=100;

for t=1:nr_trials
    congruent=false;
    
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end

    c=metalevel_model.selectControlSignal(features);
    drift_rates=c*drift_rate_controlled+...
        drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
    
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    metalevel_model=metalevel_model.learn(c,features,U,cost);
    
    chosen_signal(t)=c;
    VOC_model(t)=U-cost;
    
end

figure()
subplot(1,2,1), plot(chosen_signal),
xlabel('Trial','FontSize',18),ylabel('Chosen Control Signal','FontSize',18)

subplot(1,2,2), plot(VOC_model),
xlabel('Trial','FontSize',18),ylabel('VOC','FontSize',18)

%% Congruent trials only
clear,close all
nr_features=1;
nr_control_signals=1;

drift_rate_automatic=1;
drift_rate_controlled=1;
bias=0.5;
congruency=[true,false];

congruent=true;

if congruent
    direction_automatic=1;
else
    direction_automatic=-1;
end

metalevel_model=ModelOfEVOC(nr_features,nr_control_signals)

features=ones(nr_features,1); control_signals=ones(nr_control_signals,1);
metalevel_model.predictPerformance(features,control_signals)


signal_intensities=linspace(0,5,100);
reward_rate=0.1;
for i=1:length(signal_intensities)
    c=signal_intensities(i)
    drift_rates=c*drift_rate_controlled+drift_rate_automatic*direction_automatic;
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
        
    EU=(1-p_error)*1+p_error*(-1);
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    VOC(i)=EU-cost;
end

figure()
plot(signal_intensities,VOC)
[val,pos]=max(VOC)
c_star=signal_intensities(pos)

nr_trials=200;

for t=1:nr_trials
    
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end

    c=metalevel_model.selectControlSignal(features);
    drift_rates=c*drift_rate_controlled+...
        drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
    
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    metalevel_model=metalevel_model.learn(c,features,U,cost);
    
    chosen_signal(t)=c;
    VOC_model(t)=U-cost;
    
end

figure()
subplot(1,2,1), plot(chosen_signal),
xlabel('Trial','FontSize',18),ylabel('Chosen Control Signal','FontSize',18)

subplot(1,2,2), plot(VOC_model),
xlabel('Trial','FontSize',18),ylabel('VOC','FontSize',18)


%% 50% of the trials are congruent, a feature predicts congruency
clear,close all
nr_features=1;
nr_control_signals=1;

drift_rate_automatic=1;
drift_rate_controlled=1;
bias=0.5;

metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);

reward_rate=0.1;

nr_trials=100;

for t=1:nr_trials
    
    if rand()<0.5
        congruent=true;
        features=[1];
    else
        congruent=false;
        features=[-1];
    end
    
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end

    c=metalevel_model.selectControlSignal(features);
    drift_rates=c*drift_rate_controlled+...
        drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
    
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    metalevel_model=metalevel_model.learn(c,features,U,cost);
    
    chosen_signal(t)=c;
    VOC_model(t)=U-cost;
    trial_was_congruent(t)=congruent;
    
end

figure()
subplot(1,2,1),
plot(smooth(chosen_signal(trial_was_congruent),1)), hold on, 
plot(smooth(chosen_signal(~trial_was_congruent),1))
xlabel('Trial','FontSize',18),ylabel('Chosen Control Signal','FontSize',18)
legend('Congruent','Incongruent')
subplot(1,2,2), plot(VOC_model(trial_was_congruent)), hold on,
 plot(VOC_model(~trial_was_congruent))
xlabel('Trial','FontSize',18),ylabel('VOC','FontSize',18)
legend('Congruent','Incongruent')

mean(chosen_signal(trial_was_congruent))
mean(chosen_signal(~trial_was_congruent))


%% 50% of the trials are congruent, congruency is unpredictable

clear,close all
nr_features=1;
nr_control_signals=1;

drift_rate_automatic=1;
drift_rate_controlled=1;
bias=0.5;
congruency=[true,false];


metalevel_model=ModelOfEVOC(nr_features,nr_control_signals)

features=ones(nr_features,1); control_signals=ones(nr_control_signals,1);
metalevel_model.predictPerformance(features,control_signals)


signal_intensities=linspace(0,5,100);
reward_rate=0.1;
congruencies=[true,false];
for i=1:length(signal_intensities)
    
    for cond=1:2
        
        congruent=congruencies(cond);
        
        if congruent
            direction_automatic=1;
        else
            direction_automatic=-1;
        end
        
        c=signal_intensities(i);
        drift_rates=c*drift_rate_controlled+drift_rate_automatic*direction_automatic;
        
        [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
            EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
        
        EU=(1-p_error)*1+p_error*(-1);
        cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
            metalevel_model.reconfigurationCost(c);
        
        VOC(i,cond)=EU-cost;
    end
end

[val,pos]=max(mean(VOC,2))

figure()
plot(signal_intensities,VOC)
[val,pos]=max(VOC)
c_star=signal_intensities(pos)

clear,close all
nr_features=1;
nr_control_signals=1;

drift_rate_automatic=1;
drift_rate_controlled=1;
bias=0.5;

metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);

reward_rate=0.1;

nr_trials=100;

for t=1:nr_trials
    
    if rand()<0.5
        congruent=true;
        features=[1];
    else
        congruent=false;
        features=[1];
    end
    
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end

    c=metalevel_model.selectControlSignal(features);
    drift_rates=c*drift_rate_controlled+...
        drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
    
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    metalevel_model=metalevel_model.learn(c,features,U,cost);
    
    chosen_signal(t)=c;
    VOC_model(t)=U-cost;
    trial_was_congruent(t)=congruent;
    
end

figure()
subplot(1,2,1),
plot(smooth(chosen_signal(trial_was_congruent),1)), hold on, 
plot(smooth(chosen_signal(~trial_was_congruent),1))
xlabel('Trial','FontSize',18),ylabel('Chosen Control Signal','FontSize',18)
legend('Congruent','Incongruent')
subplot(1,2,2), plot(VOC_model(trial_was_congruent)), hold on,
 plot(VOC_model(~trial_was_congruent))
xlabel('Trial','FontSize',18),ylabel('VOC','FontSize',18)
legend('Congruent','Incongruent')

mean(chosen_signal(trial_was_congruent))
mean(chosen_signal(~trial_was_congruent))


%% Pilot Simulation 5: Congruency is partially predicted by features

clear,close all
nr_features=1;
nr_control_signals=1;

drift_rate_automatic=1;
drift_rate_controlled=0.5;
bias=0.5;

metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);

reward_rate=0.1;

nr_trials=200;
nr_simulations=20;

for sim=1:nr_simulations
    metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);

for t=1:nr_trials
    
    if rand()<0.5
        features=[1];
        training_trial_type(t)=1;
        if rand()<0.25
            congruent=true;
        else
            congruent=false;
        end
    else
        training_trial_type(t)=2;
        features=[-1];
        if rand()<0.75
            congruent=true;
        else
            congruent=false;
        end
    end
    
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end

    tic()
    c=metalevel_model.selectControlSignal(features);
    toc()
    
    drift_rates=c*drift_rate_controlled+...
        (1-c)*drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
    
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    metalevel_model=metalevel_model.learn(c,features,U,cost);
    
    chosen_signal(t)=c;
    VOC_model(t)=U-cost;
    trial_was_congruent(t)=congruent;
    
end

    mean_signal_type_A(sim)=mean(chosen_signal(training_trial_type==1));
    mean_signal_type_B(sim)=mean(chosen_signal(training_trial_type==2));
end

mean(mean_signal_type_A-mean_signal_type_B)
sem(mean_signal_type_A-mean_signal_type_B)

mean(mean_signal_type_A),sem(mean_signal_type_A)
mean(mean_signal_type_B),sem(mean_signal_type_B)

figure()
subplot(1,2,1),
plot(chosen_signal(training_trial_type==1)), hold on, 
plot(chosen_signal(training_trial_type==2))
xlabel('Trial','FontSize',18),ylabel('Chosen Control Signal','FontSize',18)
legend('Congruent','Incongruent')
subplot(1,2,2), plot(VOC_model(training_trial_type==1)), hold on,
 plot(VOC_model(training_trial_type==2))
xlabel('Trial','FontSize',18),ylabel('VOC','FontSize',18)
legend('Congruent','Incongruent')

%% Pilot Simulation 6: Transfer

clear,close all
nr_features=1;
nr_control_signals=1;

drift_rate_automatic=1;
drift_rate_controlled=1;
bias=0.5;

metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);

reward_rate=0.1;

nr_trials=200;
nr_simulations=10;

%trial types:
trial_types=[0,0,0,1; 0 0 1 0; 0 1 0 0; 1 0 0 0];
high_congruency=[false,false,true,true];
p_congruency=[0.75;0.75;0.25;0.25];
nr_trial_types=size(trial_types,1);
nr_features=size(trial_types,2);

nr_training_trials=100;
nr_test_trials=100;

for sim=1:nr_simulations
    metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);

%1. Training
for t=1:nr_training_trials
    
    training_trial_type(t)=randi(nr_trial_types);
    features=trial_types(:,training_trial_type(t));
    
    if rand()<p_congruency(training_trial_type(t))
        congruent=true;
        direction_automatic=1;
    else
        congruent=false;
        direction_automatic=-1;
    end

    tic()
    c=metalevel_model.selectControlSignal(features);
    toc()
    
    drift_rates=c*drift_rate_controlled+...
        drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
    
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    metalevel_model=metalevel_model.learn(c,features,U,cost);
    
    chosen_signal_training_trials(t)=c;
    VOC_model(t)=U-cost;
    training_trial_was_congruent(t)=congruent;
    
end

%2. Transfer phase
for t=1:nr_test_trials
    test_trial_type(t)=randi(nr_trial_types);
    features=trial_types(:,test_trial_type(t));
    
    if rand()<0.5
        congruent=true;
        direction_automatic=1;
    else
        congruent=false;
        direction_automatic=-1;
    end
    test_trial_was_congruent(t)=congruent;

    tic()
    c=metalevel_model.selectControlSignal(features);
    toc()
    
    drift_rates=c*drift_rate_controlled+...
        drift_rate_automatic*direction_automatic;
    
    
    [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
        EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,true);
        
    error=rand()<p_error;
    U=1-2*error;
    cost=reward_rate*E_RT+metalevel_model.implementationCost(c)+...
        metalevel_model.reconfigurationCost(c);
    
    reaction_time(t)=E_RT;
    error_rate(t)=p_error;
    
    metalevel_model=metalevel_model.learn(c,features,U,cost);
    
    chosen_signal(t)=c;
    VOC_model(t)=U-cost;
    
    %{
    if error_rate(t)>0.5
        throw(MException('bla','bla'))
    end
    %}

end


mean_RT_incongruent_trials(sim,1)=...
    mean(reaction_time(and(~test_trial_was_congruent,test_trial_type<=2)));
mean_RT_incongruent_trials(sim,2)=...
    mean(reaction_time(and(~test_trial_was_congruent,test_trial_type>2)));

mean_RT_congruent_trials(sim,1)=...
    mean(reaction_time(and(test_trial_was_congruent,test_trial_type<=2)));
mean_RT_congruent_trials(sim,2)=...
    mean(reaction_time(and(test_trial_was_congruent,test_trial_type>2)));

end

figure()
errorbar(1000*[mean(mean_RT_congruent_trials);mean(mean_RT_incongruent_trials)],...
    [1000*sem(mean_RT_congruent_trials);1000*sem(mean_RT_incongruent_trials)])
set(gca,'XTick',[1,2],'XTickLabel',{'Congruent','Incongruent'})
xlabel('Trial Type','FontSize',24)
ylabel('Reaction Time (ms)','FontSize',24)
set(gca,'FontSize',20)
legend('Mostly Congruent Items','Mostly Incongruent Items')




