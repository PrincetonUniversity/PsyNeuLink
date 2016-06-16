classdef DDM_Manohar2015 < Simulations.DDMSim
    
    % description of class
    % simulates sequential control adjustments in response to conflict (see
    % Gratton 1992)
    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDM_Manohar2015()
            import EVC.*;
            
            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 1;
            this.plotSum = false;
            
            this.learningFnc(1).params{1} = 1; 
            this.attractorFnc.params{1} = 0;
            this.attractorFnc.params{2} = 0;
            
            %% control signal specification
            this.ctrlSignals(2) = CtrlSignal(this.defaultCtrlSignal);
            this.ctrlSignals(2).CtrlSigStimMap  = [0 1]; 
            
            this.defaultGlobalCtrlProcess.input.params{1} = this.ctrlSignals(2);
            this.DDMProcesses(end+1) = this.defaultGlobalCtrlProcess;
            
            %% task environment parameters: task environment
            
            this.defaultTrial.stimSalience = [1 10];
            this.nTrials = 1;
            
            %% log parameters
            
            this.writeLogFile = 1; 
            this.logFileName = 'DDM_GrattonSimple'; 
            
            this.logAddVars{3} = '[this.EVCM.Log.Trials.conditions]''';
            this.logAddVarNames{3} = 'condition';
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
               
        end
        
    end
    
    methods (Access = public)
        
        function initTaskEnv(this)
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
            
        end
        
    end
    
end

