classdef DDM_VTS_ConflictAvoidApproach < Simulations.DDMSim
    
    % description of class
    % simulates a voluntary task switching situation in which the 
    % difficulty of the initially easier is increased continously in order
    % to investigate the tention between conflict adjustment and avoidance
    
    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDM_VTS_ConflictAvoidApproach()
            import EVC.*;
            import EVC.DDM.*;
            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 1;
            this.plotSum = true;
            
            temp.reconfCostFnc.params{1} = 10; %4
            temp.reconfCostFnc.params{2} = -5; %-1
            temp.reconfCostFnc.type = EVCFnc.EXP;
            this.reconfCostFnc = EVCFnc(temp.reconfCostFnc.type, temp.reconfCostFnc.params);
            
            %% control parameters: default control signal
            
            temp.costFnc1.params{1} = 3;                                                  % parameter for exponential cost function (see EVC.CCost.m for implementation details)
            temp.costFnc1.params{2} = -1;                                                 % parameter for exponential cost function (see EVC.CCost.m for implementation details)
            temp.costFnc1.type = EVCFnc.EXP;                                              % use exponential cost fnc
            temp.costFunction1 = EVCFnc(temp.costFnc1.type, temp.costFnc1.params);
            
            this.ctrlSignals(1) = this.defaultCtrlSignal;
            this.ctrlSignals(1).CtrlSigStimMap  = [1 0]; 
            this.ctrlSignals(1).CCostFnc = temp.costFunction1;
            this.ctrlSignals(2) = CtrlSignal(this.defaultCtrlSignal);
            this.ctrlSignals(2).CtrlSigStimMap  = [0 1]; 
            this.ctrlSignals(2).CCostFnc = temp.costFunction1;
            
            % map all control signals to a specific DDM parameter
            temp.taskAControlFnc.type = DDMFnc.INTENSITY2DDM;
            temp.taskAControlFnc.params{1} = this.ctrlSignals(1);
            temp.taskAControlFnc.params{2} = this.defaultControlProxy;
            temp.taskAControlFnc.params{3} = this.defaultControlMappingFnc;
            temp.ControlFncA = DDMFnc(temp.taskAControlFnc.type, ...
                                            temp.taskAControlFnc.params);
                                        
            temp.taskBControlFnc.type = DDMFnc.INTENSITY2DDM;
            temp.taskBControlFnc.params{1} = this.ctrlSignals(2);
            temp.taskBControlFnc.params{2} = this.defaultControlProxy;
            temp.taskBControlFnc.params{3} = this.defaultControlMappingFnc;
            temp.ControlFncB = DDMFnc(temp.taskBControlFnc.type, ...
                                            temp.taskBControlFnc.params);                            
            
            % define each DDM process                            
            temp.ControlProcessA = DDMProc(DDMProc.CONTROL, ...                 
                                                    DDMProc.DRIFT, ...
                                                    temp.ControlFncA);   
                                                
            temp.ControlProcessB = DDMProc(DDMProc.CONTROL, ...                 
                                                    DDMProc.DRIFT, ...
                                                    temp.ControlFncB);                                    
                                        
            
            % put all DDM processes together
            this.DDMProcesses(1) = temp.ControlProcessA;
            this.DDMProcesses(end+1) = temp.ControlProcessB;
            
            
            %% task environment parameters: trial
            
            % create an incongruent trial for Task A
            this.trials(1).ID = 1;                                                          % trial identification number (for task set)
            this.trials(1).typeID = 1;                                                      % trial type (defines task context)
            this.trials(1).cueID = 1;                                                       % cued information about trial identity
            this.trials(1).descr = 'AB_inc';                                                 % trial description
            this.trials(1).conditions    = [0];                                           % set of trial conditions (for logging)
            this.trials(1).outcomeValues = [this.rewardVal this.rewardVal];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(1).stimSalience  = [1 1];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(1).stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            0 1];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(1).params = [];                                                   % DDM specific trial parameters

            %% task environment parameters: task environment
            
            this.nTrials = 40;
            
            %% log parameters
            this.writeLogFile = 1; 
            this.logFileName = 'VTS_AvoidApproachV2'; 
            
            this.logAddVars{3} = 'reshape([this.EVCM.Log.Trials.stimSalience],2,this.nTrials)''';
            this.logAddVarNames{3} = 'stimSalience';
            
        end
        
        function getResults(this)
            
        end
        
        function dispResults(this)

        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:(this.nTrials);
            f1 = figure(1);
            set(f1, 'Position', [0 0 600 600])
            subplot(2,1,1);
            this.plotRT(exampleSubj, 'actual', sampleTrials);
            subplot(2,1,2);
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            
        end
        
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            import EVC.*;
            
            trial = Trial(this.trials(1));
            
            % build & randomize task sequence                                           
            this.taskEnv = TaskEnv(trial, this.nTrials);  
            
            for i = 1:length(this.taskEnv.Sequence)
                this.taskEnv.Sequence(i).stimSalience = [1 0+0.05*i];
            end
        end
        
        function initCtrlSignals(this)
        end
        
    end
    
end

