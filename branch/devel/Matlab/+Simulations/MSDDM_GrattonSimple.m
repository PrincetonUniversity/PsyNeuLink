classdef MSDDM_GrattonSimple < Simulations.MSDDMSim
    
    % description of class
    % simulates sequential control adjustments in response to conflict (see
    % Gratton 1992)
    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = MSDDM_GrattonSimple()

            % call parent constructor
            this = this@Simulations.MSDDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 2;
            this.plotSum = true;
            
            this.learningFnc(1).params{1} = 2; 
            this.reconfCostFnc.params{1} = 0;
            this.reconfCostFnc.params{2} = -4;
            
            %% MSDDM processes
            
            this.defaultAutomaticProcess.duration.params{1} = 0.2195;
            this.defaultDDMParams.thresh = 0.6103; 
            this.defaultAutomaticFnc.params{1} = [-2; 2];
            
            %% control signals
            
            this.defaultControlMappingFnc.params{2} = 2.8394;
            this.defaultCostFnc.params{1} = 1.0362; % 4
            this.defaultCostFnc.params{2} = -0.4805; % -2
            
            %% task environment parameters: task environment
            
            this.nTrials = 200;
            
            %% log parameters
            
            this.writeLogFile = 1; 
            this.logFileName = 'MSDDM_GrattonSimple'; 
            
            this.logAddVars{3} = '[this.EVCM.Log.Trials.conditions]''';
            this.logAddVarNames{3} = 'condition';
        end
        
        function getResults(this)
            this.results.RT.conINC = zeros(1, this.nSubj);
            this.results.RT.conCON = zeros(1, this.nSubj);
            this.results.RT.incCON = zeros(1, this.nSubj);
            this.results.RT.incINC = zeros(1, this.nSubj);
            this.results.RT.adaptEffect = zeros(1, this.nSubj);
            this.results.RT.congrEffect = zeros(1, this.nSubj);
            
            % loop through all subjects
            for subj = 1:this.nSubj
                
                 % get log structure for current subject
                 subjLog = this.subjData(subj).Log;
                
                 % get conditions
                 conditions = reshape([subjLog.TrialsOrg.conditions],[2 this.nTrials])';
                 prevInc = conditions(:,2);
                 currInc = conditions(:,1);
                 
                 % extract RT's
                 RT = this.subjData(subj).Log.RTs(:,2)';
                 
                 % get results
                 this.results.RT.conINC(subj) = mean(RT(prevInc == 0 & currInc == 1))*1000;
                 this.results.RT.conCON(subj) = mean(RT(prevInc == 0 & currInc == 0))*1000;
                 this.results.RT.incCON(subj) = mean(RT(prevInc == 1 & currInc == 0))*1000;
                 this.results.RT.incINC(subj) = mean(RT(prevInc == 1 & currInc == 1))*1000;
                 this.results.RT.congrEffect(subj) = mean(RT(currInc == 1))*1000 - mean(RT(currInc == 0))*1000;
                 this.results.RT.adaptEffect(subj) = (this.results.RT.conINC(subj) - this.results.RT.conCON(subj)) - (this.results.RT.incINC(subj) - this.results.RT.incCON(subj));
                 
                 
                 
            end
        end
        
        function dispResults(this)
              disp('++++++++++ MSDDM GrattonSimple ++++++++++');
              disp(strcat('congruency Effect: ',num2str(mean(this.results.RT.congrEffect)),'ms'));
              disp(strcat('adaptation Effect: ',num2str(mean(this.results.RT.adaptEffect)),'ms'));
        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:this.nTrials;
            f1 = figure(1);
            set(f1, 'Position', [0 0 600 650])
            subplot(5,1,1);
            this.plotER(exampleSubj, 'expected', sampleTrials);
            subplot(5,1,2);
            this.plotER(exampleSubj, 'actual', sampleTrials);
            subplot(5,1,3);
            this.plotRT(exampleSubj, 'actual', sampleTrials);
            subplot(5,1,4);
            %this.plotEVC(exampleSubj, 'expected', sampleTrials);
            this.plotDifficulty(exampleSubj, sampleTrials);
            subplot(5,1,5);
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            ylim([0 0.7]);
            
            figure(2);
            y = [mean(this.results.RT.conCON), mean(this.results.RT.conINC), mean(this.results.RT.incCON), mean(this.results.RT.incINC)];
            h = bar(y, 'BarWidth', 1,'FaceColor',[0.7 0.7 0.7]);
            ylim([0, max(y)*1.1]);
            set(gca,'XTickLabel',{'c-C', 'c-I','i-C','i-I'},'fontsize',this.plotParams.axisFontSize);
            ylabel('Reaction time (ms)','fontWeight', 'bold', 'fontSize',16);
            
               
        end
        
        function initOptimizationTaskEnv(this)
            
            this.nTrials = 3;
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
            this.taskEnv.Sequence(1).stimSalience = [0.5 0.5];
            this.taskEnv.Sequence(1).conditions = [0 0];
            this.taskEnv.Sequence(2).stimSalience = [0.2 0.8];
            this.taskEnv.Sequence(2).conditions = [1 0];
            this.taskEnv.Sequence(3).stimSalience = [0.2 0.8];
            this.taskEnv.Sequence(3).conditions = [1 1];
            
        end
        
        function criterion = getOptimizationCriterion(this)
            
           criterion = -(this.EVCM.Log.RTs(2,2) - this.EVCM.Log.RTs(3,2));
           if(this.EVCM.Log.ERs(2,2) >= 0.5) 
              criterion = 1e16; 
           end
        end
        
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
            
            % build sequence
            for trialIdx = 1:this.nTrials
                number = rand;
                if(number <= 0.5)
                    % congruent
                    this.taskEnv.Sequence(trialIdx).stimSalience = [0.5 0.5];
                    this.taskEnv.Sequence(trialIdx).conditions(1) = 0;
                else
                    % incongruent
                    this.taskEnv.Sequence(trialIdx).stimSalience = [0.2 0.8];%[0.497 0.503];
                    this.taskEnv.Sequence(trialIdx).conditions(1) = 1;
                end
                if(trialIdx > 1)
                    this.taskEnv.Sequence(trialIdx).conditions(2) = this.taskEnv.Sequence(trialIdx-1).conditions(1);
                else
                    this.taskEnv.Sequence(trialIdx).conditions(2) = 0; 
                end
            end
            
        end
        
        
    end
    
end

