classdef MSDDM_Laming < Simulations.MSDDMSim
    
    % description of class
    % simulates trial-by-trial adjustments to errors (see Laming, 1968)

    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = MSDDM_Laming()

            % call parent constructor
            this = this@Simulations.MSDDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 5;
            this.plotSum = true;
            
            this.rewardFnc.params{2} = 0;       % RT scale  in reward function
            this.binaryErrors = 1;
            this.learningFnc(1).params{1} = 0.3; 
            
            this.reconfCostFnc.params{1} = 0;
            this.reconfCostFnc.params{2} = 0;
            
            %% default DDM parameters
            this.defaultDDMParams.drift = 0.3; 

            %% control signals
            
            this.defaultCostFnc.params{1} = 4; % 4
            this.defaultCostFnc.params{2} = -2; % -2
            this.MSDDMProcesses(1) = this.defaultGlobalCtrlProcess;
            this.defaultGlobalCtrlFnc.params{3}.params{2} = 1;
            this.defaultGlobalCtrlFnc.params{3}.params{1} = 0.1; % intercept
            %% task environment parameters: task environment
            
            this.nTrials = 20;
            
            this.defaultTrial.stimSalience = [1 2];
            %% general
            
            this.writeLogFile = 1; 
            this.logFileName = 'MSDDM_Laming'; 
            
            this.logAddVars{3} = '[this.EVCM.Log.Trials.conditions]''';
            this.logAddVarNames{3} = 'condition';
        end
        
        function getResults(this)
            
        end
        
        function dispResults(this)
              disp('++++++++++ MSDDM_Laming ++++++++++');
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
        
    end
    
end

