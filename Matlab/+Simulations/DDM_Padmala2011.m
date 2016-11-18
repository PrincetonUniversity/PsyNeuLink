classdef DDM_Padmala2011 < Simulations.DDMSim
    
    % description of class
    % runs a simulation with systematic reward manipulations of the current task

    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDM_Padmala2011()

            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 1;
            this.plotSum = true;
            
            this.defaultCostFnc.params{1} = 2.0;                % 2.0 % 4
            this.defaultCostFnc.params{2} = -1;                 % -1 %-1
            
            this.defaultDDMParams.c = 0.69;                     % 0.67 % 0.62
            this.defaultDDMParams.thresh = 0.76;                % 0.77 % 0.78
            this.defaultDDMParams.t0 = 0.45;                    % 0.45 % 0.2
            
            %% task environment parameters: task environment
            
            this.nTrials = 2*3*50; % (2 reward conditions, 3 congruency conditions)
            
            %% task environment parameters: trial
            this.rewardVal = 10;
            extraReward = 6.5;
            
            % create reward trial
            this.trials(1).ID = 1;                                                          % trial identification number (for task set)
            this.trials(1).typeID = 1;                                                      % trial type (defines task context)
            this.trials(1).cueID = 1;                                                       % cued information about trial identity
            this.trials(1).descr = 'Rew';                                                 % trial description
            this.trials(1).conditions    = [1 0];                                           % set of trial conditions (for logging)
            this.trials(1).outcomeValues = [this.rewardVal+extraReward 0];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(1).stimSalience  = [1 0.14];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(1).stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            1 1];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(1).params = [];                                                   % DDM specific trial parameters

            % create no-reward trial
            this.trials(2).ID = 2;                                                          % trial identification number (for task set)
            this.trials(2).typeID = 2;                                                        % trial type (defines task context)
            this.trials(2).cueID = 2;                                                       % cued information about trial identity
            this.trials(2).descr = 'NoRew';                                                 % trial description
            this.trials(2).conditions    = [0 0];                                           % set of trial conditions (for logging)
            this.trials(2).outcomeValues = [this.rewardVal 0];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(2).stimSalience  = [1 0.14];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(2).stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            1 1];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(2).params = [];                                                   % DDM specific trial parameters
                                                 % DDM specific trial parameters
            %% log parameters
            this.descr = 'Runs a conflict task simulation with reward manipulation.\n Each trial contains either a neutral, congruent or incongruent stimulus. \n There are two reward values (non-rewarded, rewarded) that are cued for each trial. \n The model is set up to always expect a neutral trial.';
            this.writeLogFile = 1; 
            this.logFileName = 'DDM_Padmala2011'; 
            
            this.logAddVars{3} = '[this.EVCM.Log.Trials.conditions]''';
            this.logAddVarNames{3} = 'condition';
        end
        
        function getResults(this)
            
            this.results.RT.rewNtr = zeros(1, this.nSubj);
            this.results.RT.rewCon = zeros(1, this.nSubj);
            this.results.RT.rewInc = zeros(1, this.nSubj);
            this.results.RT.norewNtr = zeros(1, this.nSubj);
            this.results.RT.norewCon = zeros(1, this.nSubj);
            this.results.RT.norewInc = zeros(1, this.nSubj);
            
            this.results.ER.rewNtr = zeros(1, this.nSubj);
            this.results.ER.rewCon = zeros(1, this.nSubj);
            this.results.ER.rewInc = zeros(1, this.nSubj);
            this.results.ER.norewNtr = zeros(1, this.nSubj);
            this.results.ER.norewCon = zeros(1, this.nSubj);
            this.results.ER.norewInc = zeros(1, this.nSubj);
            
            % loop through all subjects
            for subj = 1:this.nSubj
                
                % get log structure for current subject
                subjLog = this.subjData(subj).Log;
                
                % extract relevant trial conditions fpr current subject
                conditions = reshape([subjLog.TrialsOrg.conditions],[3 this.nTrials])';
                reward = conditions(:,1);
                congruency = conditions(:,2);
                
                % extract RTs and ERs from log data for current subject
                RT = this.subjData(subj).Log.RTs(:,2)';
                ER = this.subjData(subj).Log.ERs(:,2)';
                
                % get results
                this.results.RT.rewNtr(subj) = mean(RT(reward == 1 & congruency == 0))*1000;
                this.results.RT.rewCon(subj) = mean(RT(reward == 1 & congruency == 1))*1000;
                this.results.RT.rewInc(subj) = mean(RT(reward == 1 & congruency == 2))*1000;
                this.results.RT.norewNtr(subj) = mean(RT(reward == 0 & congruency == 0))*1000;
                this.results.RT.norewCon(subj) = mean(RT(reward == 0 & congruency == 1))*1000;
                this.results.RT.norewInc(subj) = mean(RT(reward == 0 & congruency == 2))*1000;
                
                this.results.ER.rewNtr(subj) = mean(1-ER(reward == 1 & congruency == 0))*100;
                this.results.ER.rewCon(subj) = mean(1-ER(reward == 1 & congruency == 1))*100;
                this.results.ER.rewInc(subj) = mean(1-ER(reward == 1 & congruency == 2))*100;
                this.results.ER.norewNtr(subj) = mean(1-ER(reward == 0 & congruency == 0))*100;
                this.results.ER.norewCon(subj) = mean(1-ER(reward == 0 & congruency == 1))*100;
                this.results.ER.norewInc(subj) = mean(1-ER(reward == 0 & congruency == 2))*100;
                
            end
        end
        
        function dispResults(this)
           disp('++++++++++ DDM_Padmala2011 ++++++++++');
        end
        
        function plotSummary(this) 
            
            f1 = figure(2);
            set(f1, 'Position', [600 0 800 300])
            this.plotPadmalaResults();
        end
        
        function plotPadmalaResults(this)
            % plot RT interaction 
            subplot(1,2,1);
            y = [mean(this.results.RT.norewCon), mean(this.results.RT.norewNtr), mean(this.results.RT.norewInc); ...
                 mean(this.results.RT.rewCon), mean(this.results.RT.rewNtr), mean(this.results.RT.rewInc)];
            h = bar(y, 'BarWidth', 1);
            ylim([525 650]);
            hold on;
            % no error bars for now
%             semBar = [std(this.results.RT.norewCon)/sqrt(length(this.results.RT.norewCon)), std(this.results.RT.norewNtr)/sqrt(length(this.results.RT.norewNtr)), std(this.results.RT.norewInc)/sqrt(length(this.results.RT.norewInc)); ...
%                       std(this.results.RT.rewCon)/sqrt(length(this.results.RT.rewCon)), std(this.results.RT.rewNtr)/sqrt(length(this.results.RT.rewNtr)), std(this.results.RT.rewInc)/sqrt(length(this.results.RT.rewInc))];
            hold off;
            set(h(1),'FaceColor',[0,114,179]/255);
            set(h(2),'FaceColor',[103,200,64]/255);
            set(h(3),'FaceColor',[246,102,1]/255);
            set(gca,'XTickLabel',{'No reward', 'Reward'},'fontsize',this.plotParams.axisFontSize);
            ylabel('Reaction time (ms)','fontWeight', 'bold', 'fontSize',16);
            l = cell(1,3);
            l{1}='congruent'; l{2}='neutral'; l{3}='incongruent';   
            legend(h,l);
            
            % plot ET interaction 
            subplot(1,2,2);
            y = [mean(this.results.ER.norewCon), mean(this.results.ER.norewNtr), mean(this.results.ER.norewInc); ...
                 mean(this.results.ER.rewCon), mean(this.results.ER.rewNtr), mean(this.results.ER.rewInc)];
            h = bar(y, 'BarWidth', 1);
            ylim([85 100]);
            hold on;
            % no error bars for now
%             semBar = [std(this.results.ER.norewCon)/sqrt(length(this.results.ER.norewCon)), std(this.results.ER.norewNtr)/sqrt(length(this.results.ER.norewNtr)), std(this.results.ER.norewInc)/sqrt(length(this.results.ER.norewInc)); ...
%                       std(this.results.ER.rewCon)/sqrt(length(this.results.ER.rewCon)), std(this.results.ER.rewNtr)/sqrt(length(this.results.ER.rewNtr)), std(this.results.ER.rewInc)/sqrt(length(this.results.ER.rewInc))];
            hold off;
            set(h(1),'FaceColor',[0,114,179]/255);
            set(h(2),'FaceColor',[103,200,64]/255);
            set(h(3),'FaceColor',[246,102,1]/255);
            set(gca,'XTickLabel',{'No reward', 'Reward'},'fontsize',this.plotParams.axisFontSize);
            ylabel('Accuracy (%)','fontWeight', 'bold', 'fontSize',16);   
            %legend(h,l);
            set(gcf,'color','white');
        end
               
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            
            % define S-R-Mappings for neutral, congruent & incongruent
            % trials
            SRMap_neutral       = [1 0; ...
                                   1 1];
                         
            SRMap_congruent     = [1 0; ...
                                   2 0];
                               
            SRMap_incongruent   = [1 0; ...
                                   0 2];
                               
            rew_trial_props = [0.5 0.5];    % proportion reward/non-reward trials
            shuffle = 1;                    % shuffle sequence
            
            % create sequence
            
            trials(1) = EVC.Trial(this.trials(1));
            trials(2) = EVC.Trial(this.trials(2));
            
            this.taskEnv = EVC.TaskEnv(trials, this.nTrials, rew_trial_props, shuffle);
            conditions = reshape([this.taskEnv.Sequence.conditions], 2, this.nTrials);
            
            % create congruency conditions 
            congrCond = repmat(1:3, 1, this.nTrials/(2*3)); 
            
            conditions(2,conditions(1,:) == 1) = congrCond(randperm(length(congrCond)));
            conditions(2,conditions(1,:) == 0) = congrCond(randperm(length(congrCond)));
            
            for trialIdx = 1:this.nTrials
                switch conditions(2,trialIdx)
                    case 1 % neutral
                        this.taskEnv.Sequence(trialIdx).stimRespMap = SRMap_neutral;
                        this.taskEnv.Sequence(trialIdx).conditions(2) = 0;
                    case 2 % congruent
                        this.taskEnv.Sequence(trialIdx).stimRespMap = SRMap_congruent;
                        this.taskEnv.Sequence(trialIdx).conditions(2) = 1;
                    case 3 % incongruent
                        this.taskEnv.Sequence(trialIdx).stimRespMap = SRMap_incongruent;
                        this.taskEnv.Sequence(trialIdx).conditions(2) = 2;
                end
                if(trialIdx > 1)
                    this.taskEnv.Sequence(trialIdx).conditions(3) = this.taskEnv.Sequence(trialIdx-1).conditions(2);
                else
                    this.taskEnv.Sequence(trialIdx).conditions(3) = 0; 
                end
            end
            
            
        end
        
    end
    
end

