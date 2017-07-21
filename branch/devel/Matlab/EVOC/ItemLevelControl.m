%% Simulation of List-Level Control
%Bugg, Jacoby, & Toth (2008). "Multiple Levels of Control".

clear,close all,clc

%Experimental Design according to Table 1
nr_trials=168;
%feature 1: BLUE   -  80% congruent
%feature 2: YELLOW -  80% congruent
%feature 3: GREEN  -  20% congruent
%feature 4: WHITE  -  20% congruent
%feature 5: FONT   -  depends on condition

%multi-feature condition:
%FONT: +1 (80% congruent) and  -1 (20% congruent)

%single-feature condition:
%FONT: +1 (50% congruent) and -1 (50% congruent)

trial_types=[+1,-1,-1,-1,+1; -1,+1,-1,-1,+1;-1,-1,+1,-1,-1;-1,-1,-1,+1,-1;...
             +1,-1,-1,-1,-1; -1,+1,-1,-1,-1;-1,-1,+1,-1,+1;-1,-1,-1,+1,+1];
nr_features=size(trial_types,2);

nr_control_signals=2;

p_congruent(:,1)=[0.8;0.8;0.2;0.2;0.8;0.8;0.2;0.2];
p_congruent(:,2)=[0.8;0.8;0.2;0.2;0.8;0.8;0.2;0.2];

trial_frequencies(:,1)=[58;58;58;58;0;0;0;0];     % dual-feature condition
trial_frequencies(:,2)=[29;29;29;29;29;29;29;29]; % single-feature condition

nr_conditions=2;
nr_trials=232;

nr_simulations=100;
next_sim=1;
reward_rate=0.5;
sigma_epsilon=0.5;
automatic_bias=0.25;
%% Determine the optimal control signal and the associated predictions

reward_rate=1;
sigma_epsilon=0.25;
automatic_bias=0.4;
threshold=0.2;
control_cost.a=1/4;
control_cost.b=-5;

metalevel_model=ModelOfEVOC(1,2);
metalevel_model.implementation_cost.a=control_cost.a;
metalevel_model.implementation_cost.b=control_cost.b;
metalevel_model.reconfiguration_cost.a=control_cost.a;
metalevel_model.reconfiguration_cost.b=control_cost.b;

signal_intensities=linspace(0.01,10,200);

%VOC = E[U | s,c] - cost(s,c)
congruency=[true,false];
nr_conditions=numel(congruency);
for condition=1:nr_conditions
    congruent=congruency(condition);
    
    if congruent
        direction_automatic=1;
    else
        direction_automatic=-1;
    end
        
    DDM_parameters.c=sigma_epsilon; %noise

    t=1;
    %for c1=1:length(signal_intensities)
    %    for c2=1:length(signal_intensities)
            for c3=1:length(signal_intensities)
            control_signals(:,c3)=signal_intensities(c3);%[signal_intensities(c1);signal_intensities(c2)];
            
            drift_rates=1;%+control_signals(1,c3); %drift rate            
            DDM_parameters.z=threshold;%+control_signals(2,c3); %threshold
            DDM_parameters.RVScale=1;
            bias=0.5+(direction_automatic*automatic_bias)/(1+control_signals(1,c3));
            
            [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
                EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
            
            p_error=max(0,min(1,p_error));
            
            cost=reward_rate*E_RT+metalevel_model.implementationCost(control_signals(:,t))+...
                metalevel_model.reconfigurationCost(control_signals(:,t));
                           
            EU=dot([p_error;1-p_error],[-1;1]);
            VOC_model(c3,condition)=EU-cost;
            
            t=t+1;
        %end
    end    
end

VOC_mostly_congruent=0.75*VOC_model(:,1)+0.25*VOC_model(:,2);
[val,ind]=max(VOC_mostly_congruent);
c_star_mc=control_signals(:,ind)

VOC_mostly_incongruent=0.25*VOC_model(:,1)+0.75*VOC_model(:,2);
[val,ind]=max(VOC_mostly_incongruent);
c_star_mic=control_signals(:,ind)

%I. predictions for Mostly Congruent Items
drift_rates=1;%+c_star_mc(1); %drift rate
DDM_parameters.z=threshold;%+c_star_mc(2); %threshold
%I.a) congruent trials
bias=0.5+1*automatic_bias/(1+c_star_mc);
[p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
    EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
error_rate(1,1)=max(0,min(1,p_error));
RT(1,1)=E_RT;
%I.b) incongruent trials
bias=0.5-1*automatic_bias/(1+c_star_mc);
[p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
    EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
error_rate(2,1)=max(0,min(1,p_error));
RT(2,1)=E_RT;

%II. mostly congruent items
drift_rates=1;%+c_star_mc(1); %drift rate
DDM_parameters.z=threshold;%+c_star_mc(2); %threshold

%II.a) congruent trials
bias=0.5+1*automatic_bias/(1+c_star_mic);
[p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
    EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
error_rate(1,2)=max(0,min(1,p_error));
RT(1,2)=E_RT;
%II.b) incongruent trials
bias=0.5-1*automatic_bias/(1+c_star_mic);
[p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
    EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
error_rate(2,2)=max(0,min(1,p_error));
RT(2,2)=E_RT;

figure(1)
subplot(1,2,1),plot(RT),legend('mostly congruent','mostly incongruent'),ylabel('Reaction Time','FontSize',18)
subplot(1,2,2),plot(error_rate),legend('mostly congruent','mostly incongruent'),ylabel('Error Rate','FontSize',18)


%figure()
%imagesc(VOC),colorbar()


%% Simulate Learning
%metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);
%specify some prior knowledge
%trained_metalevel_model=trainEVOCModel(metalevel_model);

clear,close all,clc


%Experimental Design according to Table 1
nr_trials=168;
%feature 1: BLUE   -  80% congruent
%feature 2: YELLOW -  80% congruent
%feature 3: GREEN  -  20% congruent
%feature 4: WHITE  -  20% congruent
%feature 5: FONT   -  depends on condition

%multi-feature condition:
%FONT: +1 (80% congruent) and  -1 (20% congruent)

%single-feature condition:
%FONT: +1 (50% congruent) and -1 (50% congruent)

trial_types=[+1,-1,-1,-1,+1; -1,+1,-1,-1,+1;-1,-1,+1,-1,-1;-1,-1,-1,+1,-1;...
             +1,-1,-1,-1,-1; -1,+1,-1,-1,-1;-1,-1,+1,-1,+1;-1,-1,-1,+1,+1];
nr_features=size(trial_types,2);

nr_control_signals=1;

p_congruent(:,1)=[0.8;0.8;0.2;0.2;0.8;0.8;0.2;0.2];
p_congruent(:,2)=[0.8;0.8;0.2;0.2;0.8;0.8;0.2;0.2];

trial_frequencies(:,1)=[58;58;58;58;0;0;0;0];     % dual-feature condition
trial_frequencies(:,2)=[29;29;29;29;29;29;29;29]; % single-feature condition

nr_conditions=2;
nr_trials=232;

nr_simulations=50;
next_sim=1;

reward_rate=25;
sigma_epsilon=0.65;
automatic_bias=0.45;
drift_rates=4;
threshold=0.75;
control_cost.a=1/2;
control_cost.b=-5;


DDM_parameters.c=sigma_epsilon; %noise
DDM_parameters.z=threshold;%+control_signals(2,c3); %threshold

type_of_trial=zeros(nr_trials,nr_conditions,nr_simulations);
for condition=1:nr_conditions
    
    for sim=next_sim:nr_simulations
        
        type_of_trial(:,condition,sim)=shuffle([...
            1*ones(trial_frequencies(1,condition),1);...
            2*ones(trial_frequencies(2,condition),1);...
            3*ones(trial_frequencies(3,condition),1);...
            4*ones(trial_frequencies(4,condition),1);...
            5*ones(trial_frequencies(5,condition),1);...
            6*ones(trial_frequencies(6,condition),1);...
            7*ones(trial_frequencies(7,condition),1);...
            8*ones(trial_frequencies(8,condition),1)...
        ]);
        
        clear metalevel_model
        metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);
        metalevel_model.implementation_cost.a=control_cost.a;
        metalevel_model.implementation_cost.b=control_cost.b;
        metalevel_model.reconfiguration_cost.a=control_cost.a;
        metalevel_model.reconfiguration_cost.b=control_cost.b;
        %metalevel_model=trained_metalevel_model;
        

        for t=1:nr_trials
            trial_type=type_of_trial(t,condition,sim);
            
            features=trial_types(trial_type,:)';
            
            if rand()<p_congruent(trial_type,condition)
                trial_congruent(t)=true;
                direction_automatic=1;
            else
                trial_congruent(t)=false;
                direction_automatic=-1;
            end
            
            %tic()
            control_signals=metalevel_model.selectControlSignal(features,1);
            %toc()
            chosen_signal(:,t)=control_signals;
            
            bias=0.5+(direction_automatic*automatic_bias)/(1+control_signals(1));

                       
            [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
                EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
            
            error=rand()<p_error;
            U=1-2*error;
            cost=reward_rate*E_RT+metalevel_model.implementationCost(control_signals)+...
                metalevel_model.reconfigurationCost(control_signals);
            
            RT(t)=E_RT;
            error_rate_by_trial(t)=min(1,max(0,p_error));
            
            metalevel_model=metalevel_model.learn(control_signals,features,U,cost);
                        
            VOC_model(t)=U-cost;
                        
        end
        
        congruent_avg_control_signals_mostly_congruent(:,sim,condition)=mean(chosen_signal(:,and(p_congruent(type_of_trial(:,condition,sim))>0.5,trial_congruent')),2);
        congruent_avg_control_signals_mostly_incongruent(:,sim,condition)=mean(chosen_signal(:,and(p_congruent(type_of_trial(:,condition,sim))<0.5, trial_congruent')),2);        
        congruent_error_rate_mostly_congruent(sim,condition)=mean(error_rate_by_trial(and(p_congruent(type_of_trial(:,condition,sim))>0.5, trial_congruent')));
        congruent_reaction_time_mostly_congruent(sim,condition)=mean(RT(and(p_congruent(type_of_trial(:,condition,sim))>0.5, trial_congruent')));
        congruent_error_rate_mostly_incongruent(sim,condition)=mean(error_rate_by_trial(and(p_congruent(type_of_trial(:,condition,sim))<0.5,trial_congruent')));
        congruent_reaction_time_mostly_incongruent(sim,condition)=mean(RT(and(p_congruent(type_of_trial(:,condition,sim))<0.5,trial_congruent')));
        
        incongruent_avg_control_signals_mostly_congruent(:,sim,condition)=mean(chosen_signal(:,and(p_congruent(type_of_trial(:,condition,sim))>0.5,~trial_congruent')),2);
        incongruent_avg_control_signals_mostly_incongruent(:,sim,condition)=mean(chosen_signal(:,and(p_congruent(type_of_trial(:,condition,sim))<0.5, ~trial_congruent')),2);        
        incongruent_error_rate_mostly_congruent(sim,condition)=mean(error_rate_by_trial(and(p_congruent(type_of_trial(:,condition,sim))>0.5, ~trial_congruent')));
        incongruent_reaction_time_mostly_congruent(sim,condition)=mean(RT(and(p_congruent(type_of_trial(:,condition,sim))>0.5, ~trial_congruent')));
        incongruent_error_rate_mostly_incongruent(sim,condition)=mean(error_rate_by_trial(and(p_congruent(type_of_trial(:,condition,sim))<0.5, ~trial_congruent')));
        incongruent_reaction_time_mostly_incongruent(sim,condition)=mean(RT(and(p_congruent(type_of_trial(:,condition,sim))<0.5,~trial_congruent')));

    end


%rows: trial type (congruent,incongruent), columns: list type (mostly incongruent,mostly congruent)
avg_error_rate=[mean(congruent_error_rate_mostly_congruent(:,condition)),mean(congruent_error_rate_mostly_incongruent(:,condition));...
                mean(incongruent_error_rate_mostly_congruent(:,condition)),mean(incongruent_error_rate_mostly_incongruent(:,condition))]; 
avg_RT=[median(congruent_reaction_time_mostly_congruent(:,condition)),median(congruent_reaction_time_mostly_incongruent(:,condition));...
    	median(incongruent_reaction_time_mostly_congruent(:,condition)),median(incongruent_reaction_time_mostly_incongruent(:,condition)) ];
avg_control_signal1=[mean(congruent_avg_control_signals_mostly_congruent(1,:,condition),2),mean(congruent_avg_control_signals_mostly_incongruent(1,:,condition),2);
                    mean(incongruent_avg_control_signals_mostly_congruent(1,:,condition)),mean(incongruent_avg_control_signals_mostly_incongruent(1,:,condition))]; 
%avg_control_signal2=[mean(congruent_avg_control_signals_mostly_congruent(2,:,condition),2),mean(congruent_avg_control_signals_mostly_incongruent(2,:,condition),2);
%                    mean(incongruent_avg_control_signals_mostly_congruent(2,:,condition)),mean(incongruent_avg_control_signals_mostly_incongruent(2,:,condition))]; 
                
                
sem_error_rate=[sem(congruent_error_rate_mostly_congruent(:,condition)),sem(congruent_error_rate_mostly_incongruent(:,condition));...
                sem(incongruent_error_rate_mostly_congruent(:,condition)),sem(incongruent_error_rate_mostly_incongruent(:,condition))]; 

sem_RT=[sem(congruent_reaction_time_mostly_congruent(:,condition)),sem(congruent_reaction_time_mostly_incongruent(:,condition));...
    	sem(incongruent_reaction_time_mostly_congruent(:,condition)),sem(incongruent_reaction_time_mostly_incongruent(:,condition)) ]

%sem_control_signal=[sem(avg_control_signals_mostly_congruent);sem(avg_control_signals_mostly_incongruent)]; 

figure(1)
subplot(1,2,condition)
barwitherr(sem_error_rate,avg_error_rate),ylabel('Error Rate','FontSize',20)
set(gca,'XTick',1:2,'XTickLabel',{'Congruent','Incongruent'},'FontSize',18)
legend('Mostly Congruent','Mostly Incongruent')
if condition==1
    title('Two Cues','FontSize',18)
else
    title('Single Cue','FontSize',18)
end

figure(2)
subplot(1,2,condition)
errorbar([1,1;2 2],avg_RT,sem_RT,sem_RT),ylabel('Reaction Time','FontSize',20)
set(gca,'XTick',1:2,'XTickLabel',{'Congruent','Incongruent'},'FontSize',18)
legend('Mostly Congruent','Mostly Incongruent')
if condition==1
    title('Two Cues','FontSize',18)
else
    title('Single Cue','FontSize',18)
end

figure(3),
subplot(1,2,condition)
plot([1,1;2 2],1+avg_control_signal1),ylabel('Drift Rate','FontSize',20)
set(gca,'XTick',1:2,'XTickLabel',{'Congruent','Incongruent'},'FontSize',18)
legend('Mostly Congruent','Mostly Incongruent')
if condition==1
    title('Two Cues','FontSize',18)
else
    title('Single Cue','FontSize',18)
end

%{
figure(4),
subplot(1,2,condition)
plot([1,1;2 2],0.5+avg_control_signal2),ylabel('Threshold','FontSize',20)
set(gca,'XTick',1:2,'XTickLabel',{'Congruent','Incongruent'},'FontSize',18)
legend('Mostly Congruent','Mostly Incongruent')
if condition==1
    title('Two Cues','FontSize',18)
else
    title('Single Cue','FontSize',18)
end
%}
end