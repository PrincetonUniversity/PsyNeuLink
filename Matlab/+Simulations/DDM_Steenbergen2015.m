classdef DDM_Steenbergen2015 < Simulations.DDMSim
    
    % description of class
    % simulates sequential control adjustments in response to conflict (see
    % Gratton 1992)
    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDM_Steenbergen2015()

            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 2;
            this.plotSum = true;
            
            this.learningFnc(1).params{1} = 2; 
            this.reconfCostFnc.params{1} = 10;
            this.reconfCostFnc.params{2} = -4;
            %% task environment parameters: task environment
            
            this.nTrials = 200;
            
            %% task environment parameters: trial
            this.rewardVal = 10;
            extraReward = 6.5;
            
            %% log parameters
            
            this.writeLogFile = 1; 
            this.logFileName = 'DDM_GrattonSimple'; 
            
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
                
%                  % get conditions
%                  conditions = reshape([subjLog.TrialsOrg.conditions],[2 this.nTrials])';
%                  prevInc = conditions(:,2);
%                  currInc = conditions(:,1);
                 
                 % extract RT's
                 RT = this.subjData(subj).Log.RTs(:,2)';
                 
                 % get results
                 this.results.RT.conINC1(subj) = RT(11)*1000;
                 this.results.RT.conCON1(subj) = RT(22)*1000;
                 this.results.RT.incCON1(subj) = RT(21)*1000;
                 this.results.RT.incINC1(subj) = RT(12)*1000;
                 this.results.RT.adaptEffect1(subj) = (this.results.RT.conINC1(subj) - this.results.RT.conCON1(subj)) - (this.results.RT.incINC1(subj) - this.results.RT.incCON1(subj));
                 this.results.RT.conINC2(subj) = RT(41)*1000;
                 this.results.RT.conCON2(subj) = RT(52)*1000;
                 this.results.RT.incCON2(subj) = RT(51)*1000;
                 this.results.RT.incINC2(subj) = RT(42)*1000;
                 this.results.RT.adaptEffect2(subj) = (this.results.RT.conINC2(subj) - this.results.RT.conCON2(subj)) - (this.results.RT.incINC2(subj) - this.results.RT.incCON2(subj));
                 
                 
                 
            end
        end
        
        function dispResults(this)
              disp('++++++++++ DDMDifficultySanity ++++++++++');
              disp(strcat('congruency Effect: ',num2str(mean(this.results.RT.congrEffect)),'ms'));
              disp(strcat('adaptation Effect: ',num2str(mean(this.results.RT.adaptEffect)),'ms'));
        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:this.nTrials;
            f1 = figure(1);
            set(f1, 'Position', [0 0 600 500])
            subplot(4,1,1);
            this.plotER(exampleSubj, 'actual', sampleTrials);
            subplot(4,1,2);
            this.plotRT(exampleSubj, 'actual', sampleTrials);
            subplot(4,1,3);
            %this.plotEVC(exampleSubj, 'expected', sampleTrials);
            this.plotDifficulty(exampleSubj, sampleTrials);
            subplot(4,1,4);
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            ylim([0 0.7]);
            
%             figure(2);
%             y = [mean(this.results.RT.conCON), mean(this.results.RT.conINC), mean(this.results.RT.incCON), mean(this.results.RT.incINC)];
%             h = bar(y, 'BarWidth', 1,'FaceColor',[0.7 0.7 0.7]);
%             ylim([0, max(y)*1.1]);
%             set(gca,'XTickLabel',{'c-C', 'c-I','i-C','i-I'},'fontsize',this.plotParams.axisFontSize);
%             ylabel('Reaction time (ms)','fontWeight', 'bold', 'fontSize',16);
%             
           figure(2);
           
           y_I_1 = [mean(this.results.RT.conINC1) mean(this.results.RT.incINC1)];
           y_C_1 = [mean(this.results.RT.conCON1) mean(this.results.RT.incCON1)];
           y_I_2 = [mean(this.results.RT.conINC2) mean(this.results.RT.incINC2)];
           y_C_2 = [mean(this.results.RT.conCON2) mean(this.results.RT.incCON2)];
           x = [1 2];
           yLimit = [min([y_I_1 y_C_1 y_I_2 y_C_2])*0.9 max([y_I_1 y_C_1 y_I_2 y_C_2])*1.1];
           
           subplot(1,2,1);
           plot(x,y_I_1,'-k'); hold on;
           plot(x,y_C_1,'-k'); 
           ylim(yLimit);
           text(1.5,yLimit(1)*1.1,num2str(mean(this.results.RT.adaptEffect1)));
           hold off;
           
           subplot(1,2,2);
           plot(x,y_I_2,'-k'); hold on;
           plot(x,y_C_2,'-k'); 
           ylim(yLimit);
           text(1.5,yLimit(1)*1.1,num2str(mean(this.results.RT.adaptEffect2)));
           hold off;
        end
        
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            
            % build sequence
            conds = [ 1 1 1 1 1 1 1 1 1 1;
                      2 2 2 2 2 2 2 2 2 2;
                      1 1 1 1 1 1 1 1 1 1;
                      3 3 3 3 3 3 3 3 3 3;
                      4 4 4 4 4 4 4 4 4 4;
                      3 3 3 3 3 3 3 3 3 3];
            conds = conds';
            this.nTrials = size(conds,1)*size(conds,2);
                      
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
            
            for trialIdx = 1:this.nTrials
                
                if(conds(trialIdx) == 1)
                    % congruent 1
                    this.taskEnv.Sequence(trialIdx).stimSalience = [0.5 0.5];
                    this.taskEnv.Sequence(trialIdx).conditions(1) = 0;
                end
                if(conds(trialIdx) == 2)
                    % incongruent 1
                    this.taskEnv.Sequence(trialIdx).stimSalience = [0.48 0.52];
                    this.taskEnv.Sequence(trialIdx).conditions(1) = 1;
                end
                if(conds(trialIdx) == 3)
                    % congruent 2
                    this.taskEnv.Sequence(trialIdx).stimSalience = [0.5 0.5];
                    this.taskEnv.Sequence(trialIdx).conditions(1) = 3;
                end
                if(conds(trialIdx) == 4)
                    % incongruent 2
                    this.taskEnv.Sequence(trialIdx).stimSalience = [this.stimSalience 1-this.stimSalience];
                    this.taskEnv.Sequence(trialIdx).conditions(1) = 4;
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

