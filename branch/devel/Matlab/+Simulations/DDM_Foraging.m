classdef DDM_Foraging < Simulations.DDMSim
    
    % description of class
    % simulates sequential control adjustments in response to conflict (see
    % Gratton 1992)
    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDM_Foraging()

            import EVC.*;
            import EVC.DDM.*;
            
            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 1;                             % number of subjects
            this.binaryErrors = 1;                      % use binary decisions (either harvest or switch)
            
            this.plotSum = true;                        % plot summary statistics
            
            %% general simulation parameters: reward function
            this.rewardFnc.type = EVCFnc.FORAGEVAL;

            %% general simulation parameters: learning functions
            
            this.learningFnc = LearningFnc.empty(1,0);
            temp.EVCDummy = EVCModel(0,0);
            
            temp.learningFnc(1).params{1} = 1;                                           % learning rate for other state parameters (may not be necessary for reward updates)
            temp.learningFnc(1).type = LearningFnc.FORAGING_REWARDS;                     % learning function type
            temp.learningFnc(1).EVCM = temp.EVCDummy;                                    % EVCModel dummy 
            this.learningFnc(1) = LearningFnc(temp.learningFnc(1).type, temp.learningFnc(1).params, temp.learningFnc(1).EVCM);
            this.learningFnc(1).input{1} = @() this.learningFnc(1).EVCModel.getOutcomeProb(1);           % dynamic input value for learning function
            
            %% control parameters: default control signal
            
            % reconfiguration costs
            temp.reconfCostFnc.params{1} = 0;
            temp.reconfCostFnc.params{2} = -5;
            temp.reconfCostFnc.type = EVCFnc.EXP;
            
            % control costs
            this.defaultCostFnc.params{1} = 1.9;        % 1.8
            this.defaultCostFnc.params{2} = -1;
            this.defaultCostFnc.type = EVCFnc.EXP;
            
            % create two control signals
            % signalA ... harvest
            % signalB ... switch port
            
            this.defaultCtrlSignal.IntensityRange = [0:0.01:1]; 
            this.ctrlSignals(1) = this.defaultCtrlSignal;
            this.ctrlSignals(1).CtrlSigStimMap  = [1 0]; 
            this.ctrlSignals(2) = CtrlSignal(this.defaultCtrlSignal);
            this.ctrlSignals(2).CtrlSigStimMap  = [0 1]; 
            
            % map all control signals to a specific DDM parameter
            temp.taskAControlFnc.type = DDMFnc.INTENSITY2DDM;
            temp.taskAControlFnc.params{1} = this.ctrlSignals(1);
            temp.taskAControlFnc.params{2} = this.defaultControlProxy;
            temp.taskAControlFnc.params{3} = this.defaultControlMappingFnc;
            temp.ControlFncA = DDMFnc(temp.taskAControlFnc.type, ...
                                            temp.taskAControlFnc.params);
                                        
%             temp.taskBControlFnc.type = DDMFnc.INTENSITY2DDM;
%             temp.taskBControlFnc.params{1} = this.ctrlSignals(2);
%             temp.taskBControlFnc.params{2} = this.defaultControlProxy;
%             temp.taskBControlFnc.params{3} = this.defaultControlMappingFnc;
%             temp.ControlFncB = DDMFnc(temp.taskBControlFnc.type, ...
%                                             temp.taskBControlFnc.params);                            
            
            % define each DDM process                            
            temp.ControlProcessA = DDMProc(DDMProc.CONTROL, ...                 
                                                    DDMProc.DRIFT, ...
                                                    temp.ControlFncA);   
                                                
%             temp.ControlProcessB = DDMProc(DDMProc.CONTROL, ...                 
%                                                     DDMProc.DRIFT, ...
%                                                     temp.ControlFncB);                                    
                                        
            
            % put all DDM processes together
            this.DDMProcesses(1) = temp.ControlProcessA;
            %this.DDMProcesses(end+1) = temp.ControlProcessB;
            
            %% task environment parameters: task environment
            
            this.nTrials = 100;                          % number of trials
            
            % set initial reward values for harvesting vs. switching
            harvestReward = 90;
            switchReward = 0;
            this.defaultTrial.outcomeValues = [harvestReward switchReward];
            
            %% log parameters
            
            this.writeLogFile = 1;
            this.logFileName = 'DDM_Foraging'; 
            
            this.logAddVars{3} = '[this.EVCM.Log.Trials.outcomeValues]''';
            this.logAddVarNames{3} = 'reward';
        end
        
        function getResults(this)

        end
        
        function dispResults(this)
              disp('++++++++++ Foraging ++++++++++');

        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:this.nTrials;
            f1 = figure(1);
            set(f1, 'Position', [0 0 600 500])
            subplot(4,1,1);
            this.plotOutcomeProb(exampleSubj, 'actual', 1, sampleTrials);
            subplot(4,1,2);
            this.plotRT(exampleSubj, 'actual', sampleTrials);
            subplot(4,1,3);
            %this.plotEVC(exampleSubj, 'expected', sampleTrials);
            this.plotReward(exampleSubj, sampleTrials);
            subplot(4,1,4);
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            ylim([0 0.7]);
            
               
        end
        
    end

    
end

