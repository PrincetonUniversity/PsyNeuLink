%% Simulation of List-Level Control
%Bugg, Jacoby, & Toth (2008). "Multiple Levels of Control".

clear,close all,clc

%Experimental Design according to Table 1
nr_trials=168;
%                   condition 1        condition 2
%feature 1: RED  -  50% congruent,    50% congruent 
%feature 2: BLUE -  50% congruent,    50% congruent 
%feature 3: GREEN - 25% congruent,    75% congruent
%feature 4: WHITE - 25% congruent,    75% congruent

trial_types=[+1,0,0,0; 0,+1,0,0;0,0,+1,0;0,0,0,+1];
nr_features=size(trial_types,2);

nr_control_signals=1;

p_congruent(:,1)=[0.5;0.5;0.25;0.25];
p_congruent(:,2)=[0.5;0.5;0.75;0.75];

trial_frequencies(:,1)=[48;48;96;96]; % mostly incongruent list
trial_frequencies(:,2)=[48;48;96;96]; % mostly congruent list

nr_conditions=2;
nr_trials=sum(trial_frequencies);

reward_rate=25;
sigma_epsilon=0.65;
automatic_bias=0.45;
drift_rates=4;
threshold=0.75;
control_cost.a=1/2;
control_cost.b=-5;


DDM_parameters.c=sigma_epsilon; %noise
DDM_parameters.z=threshold;%+control_signals(2,c3); %threshold

nr_simulations=10;
next_sim=1;

%metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);
%specify some prior knowledge
%trained_metalevel_model=trainEVOCModel(metalevel_model);

type_of_trial=zeros(nr_trials(1),nr_conditions,nr_simulations);
for condition=1:nr_conditions
    
    for sim=next_sim:nr_simulations
        
        type_of_trial(:,condition,sim)=shuffle([...
            1*ones(trial_frequencies(1,condition),1);...
            2*ones(trial_frequencies(2,condition),1);...
            3*ones(trial_frequencies(3,condition),1);...
            4*ones(trial_frequencies(4,condition),1)...
            ]);
        
        clear metalevel_model
        metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);
        metalevel_model.implementation_cost.a=control_cost.a;
        metalevel_model.implementation_cost.b=control_cost.b;
        metalevel_model.reconfiguration_cost.a=control_cost.a;
        metalevel_model.reconfiguration_cost.b=control_cost.b;

        %metalevel_model=trained_metalevel_model;
        

        for t=1:nr_trials(condition)
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
            error_rate_by_trial(t)=p_error;
            
            metalevel_model=metalevel_model.learn(control_signals,features,U,cost);
                        
            VOC_model(t)=U-cost;
                        
        end
        
        avg_control_signals(:,sim,condition)=mean(chosen_signal,2);
        error_rate_congruent(sim,condition)=mean(error_rate_by_trial(trial_congruent));
        reaction_time_congruent(sim,condition)=mean(RT(trial_congruent));
        error_rate_incongruent(sim,condition)=mean(error_rate_by_trial(~trial_congruent));
        reaction_time_incongruent(sim,condition)=mean(RT(~trial_congruent));
    end
end

%rows: trial type (congruent,incongruent), columns: list type (mostly incongruent,mostly congruent)
avg_error_rate=[mean(error_rate_congruent);mean(error_rate_incongruent)]; 
avg_RT=[mean(reaction_time_congruent); mean(reaction_time_incongruent)];

sem_error_rate=[sem(error_rate_congruent);sem(error_rate_incongruent)]; 
sem_RT=[sem(reaction_time_congruent); sem(reaction_time_incongruent)];


figure(1)
barwitherr(sem_error_rate,avg_error_rate),ylabel('Error Rate','FontSize',20)
set(gca,'XTick',1:2,'XTickLabel',{'Congruent','Incongruent'},'FontSize',18)
legend('Mostly Incongruent List','Mostly Congruent List')

figure(2)
errorbar([1,1;2,2],1000*avg_RT,1000*sem_RT),ylabel('Reaction Time (ms)','FontSize',20)
set(gca,'XTick',1:2,'XTickLabel',{'Congruent','Incongruent'},'FontSize',18)
legend('Mostly Incongruent List','Mostly Congruent List')
