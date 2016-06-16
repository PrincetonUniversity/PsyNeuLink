%% Simulation of Transfer Effects
%Bugg, Jacoby, & Chanani (2011).

clear,close all,clc

%Design of Experiment 2 according to Table 1
nr_trials=20+2*208;
%feature 1: BIRD picture-  75% congruent
%feature 2: CAT  picture-  75% congruent
%feature 3: DOG  picture-  25% congruent
%feature 4: FISH picture-  25% congruent
%feature 5: BIRD word-  56% congruent
%feature 6: CAT  word-  56% congruent
%feature 7: DOG  word-  38% congruent
%feature 8: FISH word-  38% congruent

trial_types=zeros(16*4,8+16+12);
t=1;
for f1=1:4
    for f2=5:8
        for f3=1:4            
            trial_types(t,[f1,f2])=[1,1];
            image_ind=(f1-1)*4+f3;
            trial_types(t,8+image_ind)=1;
            t=t+1;
        end
    end
end

transfer_trial_types=zeros(16*3,8+16+12);
t=1;
for f1=1:4
    for f2=5:8
        for f3=1:3            
            transfer_trial_types(t,[f1,f2])=[1,1];
            image_ind=(f1-1)*3+f3;
            transfer_trial_types(t,24+image_ind)=1;
            t=t+1;
        end
    end
end

nr_trial_types=size(trial_types,1);

trial_frequencies=[reshape(repmat([36;4;4;4]'/4,[4,1]),[16,1]);...
                   reshape(repmat([4;36;4;4]'/4,[4,1]),[16,1]);...
                   reshape(repmat([12; 12;12;12]'/4,[4,1]),[16,1]);...
                   reshape(repmat([12; 12;12;12]'/4,[4,1]),[16,1])];
               
               
training_trial_probabilities=trial_frequencies/sum(trial_frequencies);

test_trial_frequencies=218*0.25*repmat(reshape(repmat([0.5;0.5/3;0.5/3;0.5/3]',[3,1]),[12,1]),[4,1])/3;

test_trial_probabilities=test_trial_frequencies/sum(test_trial_frequencies);

nr_features=size(trial_types,2);

nr_control_signals=1;

nr_conditions=1;
nr_trials=sum(trial_frequencies(:));


nr_simulations=10;
next_sim=1;

reward_rate=100;
sigma_epsilon=0.65;
automatic_bias=0.45;
drift_rates=3;
threshold=0.75;
control_cost.a=1/2;
control_cost.b=-5;

DDM_parameters.c=sigma_epsilon; %noise
DDM_parameters.z=threshold;%+control_signals(2,c3); %threshold

nr_training_trials=416;%sum(trial_frequencies);
nr_test_trials=216;

%temp=zeros(nr_training_trials,1);
%{
temp=[];
for t=1:size(trial_types,1)
    temp=[temp; t*ones(trial_frequencies(t),1)];
end
temp2=[];
for t=1:size(transfer_trial_types,1)
    temp2=[temp2; t*ones(test_trial_frequencies(t),1)];
end
%}

for sim=1:nr_simulations
    
    
    %type_of_trial(:,sim)=shuffle(temp);
    
    clear metalevel_model
    metalevel_model=ModelOfEVOC(nr_features,nr_control_signals);
    metalevel_model.implementation_cost.a=control_cost.a;
    metalevel_model.implementation_cost.b=control_cost.b;
    metalevel_model.reconfiguration_cost.a=control_cost.a;
    metalevel_model.reconfiguration_cost.b=control_cost.b;

    
    %1. Training
    for t=1:nr_training_trials
        
        type_of_trial(t)=sampleDiscreteDistributions(training_trial_probabilities',1);
        features=trial_types(type_of_trial(t),:)';
        
        image_ind=find(features(1:4)==1);
        word_ind=find(features(5:8)==1);
        if image_ind==word_ind
            congruent=true;
            direction_automatic=1;
        else
            congruent=false;
            direction_automatic=-1;
        end
        
        control_signals=metalevel_model.selectControlSignal(features);
        
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
        training_trial_was_congruent(t)=congruent;
        mostly_congruent_stimulus(t)=or(features(1)==1,features(2)==1);
        
    end
    
    mean(training_trial_was_congruent(mostly_congruent_stimulus))
    mean(training_trial_was_congruent(~mostly_congruent_stimulus))
    
    %2. Transfer phase    
    for t=1:nr_test_trials
        test_trial_type(t)=sampleDiscreteDistributions(test_trial_probabilities',1);
        features=trial_types(test_trial_type(t),:)';
        
        image_ind=find(features(1:4)==1);
        word_ind=find(features(5:8)==1);
        if image_ind==word_ind
            congruent=true;
            direction_automatic=1;
        else
            congruent=false;
            direction_automatic=-1;
        end
        
        test_trial_was_congruent(t)=congruent;
        
        control_signals=metalevel_model.selectControlSignal(features);
        
        bias=0.5+(direction_automatic*automatic_bias)/(1+control_signals(1));
        
        
        [p_error,allDTs,allRRs,E_RT,NDDMs,scaledX0,allFinalRTs_sepCDFs] =...
            EVC.DDM.AS_ddmSimFRG_Mat(drift_rates,bias,DDM_parameters);
        
        error=rand()<p_error;
        U=1-2*error;
        cost=reward_rate*E_RT+metalevel_model.implementationCost(control_signals)+...
            metalevel_model.reconfigurationCost(control_signals);
        
        reaction_time(t)=E_RT;
        error_rate(t)=min(1,max(0,p_error));
        
        metalevel_model=metalevel_model.learn(control_signals,features,U,cost);
        
        VOC_model(t)=U-cost;
        test_trial_was_congruent(t)=congruent;
        
        mostly_congruent(t)=or(features(1)==1,features(2)==1);
        mostly_incongruent(t)=or(features(3)==1,features(4)==1);
        
        chosen_signal(t)=control_signals;
    end   
    
    avg_control_signal(sim,1)=mean(chosen_signal(mostly_congruent));
    avg_control_signal(sim,2)=mean(chosen_signal(mostly_incongruent));
    
    mean_error_rate_congruent(sim,1)=mean(error_rate_by_trial(and(test_trial_was_congruent,mostly_congruent)));
    mean_error_rate_congruent(sim,2)=mean(error_rate_by_trial(and(test_trial_was_congruent,mostly_incongruent)));
    mean_error_rate_incongruent(sim,1)=mean(error_rate_by_trial(and(~test_trial_was_congruent,mostly_congruent)));
    mean_error_rate_incongruent(sim,2)=mean(error_rate_by_trial(and(~test_trial_was_congruent,mostly_incongruent)));
    
    
    mean_RT_incongruent_trials(sim,1)=...
        mean(reaction_time(and(~test_trial_was_congruent,mostly_congruent)));
    mean_RT_incongruent_trials(sim,2)=...
        mean(reaction_time(and(~test_trial_was_congruent,mostly_incongruent)));    
    mean_RT_congruent_trials(sim,1)=...
        mean(reaction_time(and(test_trial_was_congruent,mostly_congruent)));
    mean_RT_congruent_trials(sim,2)=...
        mean(reaction_time(and(test_trial_was_congruent,mostly_incongruent)));

end

mean(avg_control_signal)

fig1=figure(1)
errorbar([nanmean(mean_error_rate_congruent);nanmean(mean_error_rate_incongruent)],...
    [sem(mean_error_rate_congruent);sem(mean_error_rate_incongruent)])
set(gca,'XTick',[1,2],'XTickLabel',{'Congruent','Incongruent'})
xlabel('Trial Type','FontSize',24)
ylabel('Error Rate','FontSize',24)
set(gca,'FontSize',20)
legend('Mostly Congruent Items','Mostly Incongruent Items')


fig2=figure(2)
errorbar([1,1;2,2],1000*[nanmean(mean_RT_congruent_trials);nanmean(mean_RT_incongruent_trials)],...
    1000*[sem(mean_RT_congruent_trials);sem(mean_RT_incongruent_trials)]),ylabel('Reaction Time (ms)','FontSize',20)
set(gca,'XTick',1:2,'XTickLabel',{'Congruent','Incongruent'},'FontSize',18)
legend('Mostly Congruent Items','Mostly Incongruent Items','Location','NorthWest')
saveas(fig2,'TransferSimulation.png')
saveas(fig2,'TransferSimulation.fig')

save TransferSimulation