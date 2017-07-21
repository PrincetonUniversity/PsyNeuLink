classdef DDM_Laming < Simulations.DDMSim
    
    % description of class
    % simulates trial-by-trial adjustments to errors (see Laming, 1968)

    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDM_Laming()

            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 5;
            this.plotSum = true;
            
            this.RTscale = 0;
            this.binaryErrors = 1;
            this.defaultAutomaticFnc.params{1} = [0.05;1.0];
            this.learningFnc(1).params{1} = 0.3; 
            this.defaultDDMParams.bias = 0.1;
            
            this.reconfCostFnc.params{1} = 0;
            this.reconfCostFnc.params{2} = 0;
            %% control signals
            
            this.defaultCostFnc.params{1} = 4;
            this.defaultCostFnc.params{2} = -2;
            this.DDMProcesses(1) = this.defaultGlobalCtrlProcess;
            this.defaultGlobalCtrlFnc.params{3}.params{2} = 0.5;
            this.defaultGlobalCtrlFnc.params{3}.params{1} = 0.5; % intercept
            %% task environment parameters: task environment
            
            this.nTrials = 500;
            
            %% general
            
            this.writeLogFile = 1; 
            this.logFileName = 'DDM_Laming'; 
            
            this.logAddVars{3} = '[this.EVCM.Log.Trials.conditions]''';
            this.logAddVarNames{3} = 'condition';
        end
        
        function getResults(this)
            
            postTrials = 5;
            
            RTdata = nan(this.nSubj, 2+postTrials);
            ERdata = nan(this.nSubj, 2+postTrials);

            for subj = 1:this.nSubj

                logData = this.subjData(subj).Log;
                ERs = logData.ERs(:,2);
                RTs = logData.RTs(:,2);
                nTrials = length(ERs);

                post1ERs = ERs(2:end-4);
                post2ERs = ERs(3:end-3);
                post3ERs = ERs(4:end-2);
                post4ERs = ERs(5:end-1);
                post5ERs = ERs(6:end);
                ERs = ERs(1:end-5);

                post1RTs = RTs(2:end-4);
                post2RTs = RTs(3:end-3);
                post3RTs = RTs(4:end-2);
                post4RTs = RTs(5:end-1);
                post5RTs = RTs(6:end);    
                RTs = RTs(1:end-5);

                % mean RT & ER
                RTdata(subj, 1) = mean(RTs);
                ERdata(subj, 1) = sum(ERs)/nTrials;
                % error RT
                conditionIdx = find(ERs == 1);
                RTdata(subj, 2) = mean(RTs(conditionIdx));
                ERdata(subj, 2) = sum(ERs(conditionIdx)/length(conditionIdx));   % sanity check: has to be 1
                % post 1 error RT & ER
                RTdata(subj, 3) = mean(post1RTs(conditionIdx));
                ERdata(subj, 3) = sum(post1ERs(conditionIdx))/length(conditionIdx);
                % post 2 error RT & ER
                RTdata(subj, 4) = mean(post2RTs(conditionIdx));
                ERdata(subj, 4) = sum(post2ERs(conditionIdx))/length(conditionIdx);
                % post 3 error RT & ER
                RTdata(subj, 5) = mean(post3RTs(conditionIdx));
                ERdata(subj, 5) = sum(post3ERs(conditionIdx))/length(conditionIdx);
                % post 4 error RT & ER
                RTdata(subj, 6) = mean(post4RTs(conditionIdx));
                ERdata(subj, 6) = sum(post4ERs(conditionIdx))/length(conditionIdx);
                % post 5 error RT & ER
                RTdata(subj, 7) = mean(post5RTs(conditionIdx));
                ERdata(subj, 7) = sum(post5ERs(conditionIdx))/length(conditionIdx);

            end
            
            RTdata = mean(RTdata,1);
            ERdata = mean(ERdata,1);

            RTdata(2:end) = RTdata(2:end) - RTdata(1);

            this.results.RTdata = RTdata*1000; % convert to ms
            this.results.ERdata = ERdata;
        end
        
        function dispResults(this)
              disp('++++++++++ DDM_Laming ++++++++++');
        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:this.nTrials;
            f1 = figure(1);
            set(f1, 'Position', [0 0 600 900])
            subplot(5,1,1);
            this.plotER(exampleSubj, 'expected', sampleTrials);
            subplot(5,1,2);
            this.plotER(exampleSubj, 'actual', sampleTrials);
            subplot(5,1,3);
            %this.plotEVC(exampleSubj, 'expected', sampleTrials);
            %this.plotDifficulty(exampleSubj, sampleTrials);
            this.plotRT(exampleSubj, 'actual', sampleTrials);
            subplot(5,1,4);
            this.plotDifficulty(exampleSubj, sampleTrials);
            subplot(5,1,5);
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            ylim([0 0.7]);
               
        end
        
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
            
            statePerturbProb = 0;                                              % Likelihood on a given trial of stimulus saliency changing
            statePerturbRange = [-0.4:0.05:0.4];                                 % Range of possible per-trial bias changes
            statePerturbRange(statePerturbRange==0) = [];
            stateLimit = [0.5 1]; 

            % build sequence
            for trialIdx = 2:this.nTrials
                number = rand;
                if(number < statePerturbProb)
                    perturb = randsample(1:length(statePerturbRange),1);
                    newSalience = this.taskEnv.Sequence(trialIdx-1).stimSalience(1)+statePerturbRange(perturb);
                    newSalience = min(newSalience,stateLimit(2));
                    newSalience = max(newSalience,stateLimit(1));
                    this.taskEnv.Sequence(trialIdx).stimSalience(1) = newSalience;
                
                    if(statePerturbRange(perturb) > 0)
                        this.taskEnv.Sequence(trialIdx).conditions = 1;
                    else
                        this.taskEnv.Sequence(trialIdx).conditions = 1; %-1
                    end
                else 
                    this.taskEnv.Sequence(trialIdx).stimSalience(1) = this.taskEnv.Sequence(trialIdx-1).stimSalience(1);
                    this.taskEnv.Sequence(trialIdx).conditions = 0;
                end
            end
            
        end
        
        function initOptimizationTaskEnv(this)
            
            
        end
    end
    
end

