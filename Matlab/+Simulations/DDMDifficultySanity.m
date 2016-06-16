classdef DDMDifficultySanity < Simulations.DDMSim
    
    % description of class
    % runs a simulation with difficulty manipulations of the current task

    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDMDifficultySanity()

            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 2;
            this.plotSum = true;
            
            %% task environment parameters: task environment
            
            this.nTrials = 400;
        end
        
        function getResults(this)
            
        end
        
        function dispResults(this)
              disp('++++++++++ DDMDifficultySanity ++++++++++');
        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:this.nTrials;
            f1 = figure(1);
            set(f1, 'Position', [0 0 600 500])
            subplot(3,1,1);
            this.plotER(exampleSubj, 'actual', sampleTrials);
            subplot(3,1,2);
            %this.plotEVC(exampleSubj, 'expected', sampleTrials);
            this.plotDifficulty(exampleSubj, sampleTrials);
            subplot(3,1,3);
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            ylim([0 0.7]);
               
        end
        
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
            
            % trial rewards & conditions
            saliency = 0.1:0.1:1;
            conditions = 1:length(saliency);
            
            % check if there are enough trials for first block
            nBlockTrials = this.nTrials/(2*length(saliency));
            if(mod(nBlockTrials,1) ~= 0)
               warning('DDMDifficultySanity: cannot evenly distribute reward blocks for given number of trials'); 
            end
            
            % decreasing & increasing difficulty blocks
            blockTrials = repmat([sort(conditions,'descend') sort(conditions,'ascend')],this.nTrials/(2*length(saliency)),1);
            blockTrials = blockTrials(1:end);
            
            % build sequence
            for trialIdx = 1:this.nTrials
                this.taskEnv.Sequence(trialIdx).stimSalience = ones(1,length(this.taskEnv.Sequence(trialIdx).stimSalience));
                this.taskEnv.Sequence(trialIdx).stimSalience(1) = saliency(blockTrials(trialIdx));
                this.taskEnv.Sequence(trialIdx).conditions = blockTrials(trialIdx);
            end
            
        end
        
    end
    
end

