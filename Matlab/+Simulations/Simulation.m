classdef Simulation < handle
    
    % description of class
    
    
    % global parameters
    properties
               
        
        %% model parameters
        EVCM                                    % EVCModel: instance of EVC model
        ctrlSignals                             % CtrlSignal[]: all implemented control signals
        taskEnv                                 % TaskEnv: Task environment
        
        % general EVC simulation parameters
        nSubj                                   % EVCFnc: number of subjects
        RTconst                                 % double: offset for impact of RT on reward rate (set to 1, if RTscale = 0)
        RTscale                                 % double: impact of RT on reward rate (EVC): RR = Reward / (RTconst + RTscale * RT)
        binaryErrors                            % flag: produce binary errors?
        reconfCostFnc                           % EVCFnc: control signal reconfiguration cost function
        %taskSetInertia                         % double: degree to which old control signal persists on the next trial
        rewardFnc                               % EVCFnc: specifies reward rate function (how to calculate expected value)
        learningFnc                             % LearningFnc[]: holds learning functions
       
        % control parameters
        defaultCostFnc                          % EVC.CCost: default cost function
        defaultCtrlSignal                       % EVC.CtrlSignal: default control signal
        
        % task environment parameters
        defaultTrial                            % Trial: default trial
        trials                                  % Trial[]: holds all trial types
        rewardVal                               % EVCFnc: default value for a given rewarding outcome
        nonRewardVal                            % EVCFnc: default value for a given non-rewarding outcome
        stimSalience                            % EVCFnc: default (relative) salience for a given stimulus
        
        nTrials                                 % int: default number of trials
 
        %% simulation run parameters 
        
        currSubj                                % int: current subject
        
        % plot settings
        plotTrl                                 % flag: plot current trial?
        plotSum                                 % flag: plot summary of simulation?
        plotParams                              % struct: contains plot paramameters
        showProgress                            % flag: plot simulation progress bar?
        wrapper                                 % need to call wrapping functions?
        
        % log parameters
        descr                                   % String: Simulation description
        printResults                            % flag: calculate and print simulation results?
        printBOLD                                % flag: display BOLD results
        writeLogFile                            % flag: write log file?
        overrideLogFile                         % flag: if log file already exists, override it?
        paramPath                               % String: file directory where parameter files shall be saved
        logFilePath                             % String: file directory where log files shall be saved
        logFileName                             % String: file name of log file
        logVarNames                             % String[]: log variables to write (EVC.Log.*)
        logAddVars                              % String[]: additional log parameters
        logAddVarNames                          % String[]: names for additional log parameters
        subjData                                % contains log data of the whole batch
        results                                 % holds all calculated results
        
    end
    
    methods
        
        function this = Simulation()

            import EVC.*;

            %% general simulation parameters
            
            this.nSubj = 1;
            this.RTconst = 1;
            this.RTscale = 1;
            this.binaryErrors = 0;
            this.wrapper = 0;
            
            %% general simulation parameters: reward function
            this.RTconst = 1;
            this.RTscale = 1;
            
            % specify reward function parameters
            temp.rewardFnc.params{1} = this.RTconst;
            temp.rewardFnc.params{2} = this.RTscale;
            temp.rewardFnc.type = EVCFnc.REWRATE;
            this.rewardFnc = EVCFnc(temp.rewardFnc.type, temp.rewardFnc.params);
            
            %% general simulation parameters: reconfiguration cost function
             
%             temp.attractorFnc.params{1} = 0;
%             temp.attractorFnc.params{2} = 0;
%             temp.attractorFnc.type = EVCFnc.EXP;
%             this.attractorFnc = EVCFnc(temp.attractorFnc.type, temp.attractorFnc.params);
             
            temp.reconfCostFnc.params{1} = 10;
            temp.reconfCostFnc.params{2} = -5;
            temp.reconfCostFnc.type = EVCFnc.EXP;
            this.reconfCostFnc = EVCFnc(temp.reconfCostFnc.type, temp.reconfCostFnc.params);
            
%            this.taskSetInertia = 0;
            %% general simulation parameters: learning functions
            
            this.learningFnc = LearningFnc.empty(2,0);
            temp.EVCDummy = EVCModel(0,0);
            
            temp.learningFnc(1).params{1} = 0.5;                                          % learning rate for hidden expected state parameter (e.g. stimulus saliency), learning bases on performance
            temp.learningFnc(1).type = LearningFnc.SIMPLE_SALIENCY_RL;                  % learning function type
            temp.learningFnc(1).EVCM = temp.EVCDummy;                                   % EVCModel dummy 
            this.learningFnc(1) = LearningFnc(temp.learningFnc(1).type, temp.learningFnc(1).params, temp.learningFnc(1).EVCM);
            this.learningFnc(1).input{1} = @() this.learningFnc(1).EVCModel.getOutcomeProb(1);           % dynamic input value (actual state) for learning function
            this.learningFnc(1).input{2} = @() this.learningFnc(1).EVCModel.getOutcomeProb(0);           % dynamic input value (expected state) for learning function
            
            temp.learningFnc(2).params{1} = 1;                                          % learning rate for other state parameters (learning by exposure)
            temp.learningFnc(2).type = LearningFnc.SIMPLE_STATE_RL;                     % learning function type
            temp.learningFnc(2).EVCM = temp.EVCDummy;                                   % EVCModel dummy 
            this.learningFnc(2) = LearningFnc(temp.learningFnc(2).type, temp.learningFnc(2).params, temp.learningFnc(2).EVCM);
            this.learningFnc(2).input{1} = @() this.learningFnc(2).EVCModel.getOutcomeProb(1);           % dynamic input value for learning function
            
            %% control parameters: default cost function

            temp.costFnc.params{1} = 4;                                                  % parameter for exponential cost function (see EVC.CCost.m for implementation details)
            temp.costFnc.params{2} = -1;                                                 % parameter for exponential cost function (see EVC.CCost.m for implementation details)
            temp.costFnc.type = EVCFnc.EXP;                                              % use exponential cost fnc

            % corresponding linear cost fnc
%             temp.costFnc.params{1} = 0;                                                 % parameter for exponential cost function (see EVC.CCost.m for implementation details)
%             temp.costFnc.params{2} = 6;                                                 % parameter for exponential cost function (see EVC.CCost.m for implementation details)
%             temp.costFnc.type = EVCFnc.LINEAR;                                              % use exponential cost fnc
            
            this.defaultCostFnc = EVCFnc(temp.costFnc.type, temp.costFnc.params);

            %% control parameters: default control signal
            this.ctrlSignals = CtrlSignal.empty(1,0);
            
            % drift signal (task A)
            temp.ctrlSig.startIntensity = 0;                                            % Start intensity for ctrl signal
            temp.ctrlSig.ctrl2StimMap  = [1 0];                                         % 1st control identity (1st row) attends to 1st stimulus (col 1)
            temp.ctrlSig.lambda = 0;                                                    % outcome discounting factor
            temp.ctrlSig.intensityRange = [0:0.01:1];                                   % possible control intensity range

            this.defaultCtrlSignal = CtrlSignal(temp.ctrlSig.startIntensity, this.defaultCostFnc, temp.ctrlSig.lambda,  temp.ctrlSig.ctrl2StimMap, temp.ctrlSig.intensityRange);
            
            %% task environment parameters: trial

            this.rewardVal = 10;
            this.nonRewardVal = 0;
            this.stimSalience = 1;
            
            temp.trial.ID = 1;                                                          % trial identification number (for task set)
            temp.trial.typeID = 1;                                                      % trial type (defines task context)
            temp.trial.cueID = 1;                                                       % cued information about trial identity
            temp.trial.descr = 'default';                                               % trial description
            temp.trial.conditions    = [1];                                           % set of trial conditions (for logging)
            temp.trial.outcomeValues = [this.rewardVal this.nonRewardVal];              % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            temp.trial.stimSalience  = [this.stimSalience this.stimSalience];           % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            temp.trial.stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                        0 1];                                           % responding to stimulus 3 tends to produce second outcome by 100%
            temp.trial.params = [];                                                     % DDM specific trial parameters
            
            this.defaultTrial = Trial(temp.trial);
            
            %% task environment parameters: task environment
            
            this.nTrials = 100;                                                              


            %% plot settings

            this.plotTrl = 0;                             
            this.plotSum = 0; 
            printBOLD = 0;
            this.showProgress = 0;  
            
            this.plotParams.axisFontSize = 12;
            this.plotParams.titleFontSize = 14;
            this.plotParams.lineWidth = 2;
            this.plotParams.intensityLineWidth = 1;
            this.plotParams.expectedColor = [0 0 0.9];
            this.plotParams.actualColor = [0 0.9 0];
            this.plotParams.defaultColor = [0 0 0];
            this.plotParams.defaultBarColor = [0.7 0.7 0.7];
            this.plotParams.signalColors = [1 0.5 0; % orange
                                            0.2 0 1; % blue
                                            0 0.5 0;% green
                                            0 0 0;% black
                                            0.5 0.5 0.5];% green
            this.plotParams.signalTypes = {'-', '-', '-', '-'};                            
            this.plotParams.conditionColors = [1 0.9 0.8; % orange
                                0.9 0.8 1; % blue
                                0.8 0.9 0.8;% green
                                0.8 0.8 0.8;% black
                                0.9 0.9 0.9];% green
            

            %% log parameters
            this.descr = 'This is a most general simulation possible. \n For details see Simulations/Simulation.m';
            this.paramPath = 'params/';
            this.logFilePath = 'logfiles/';
            
            this.printResults = 0;
            this.writeLogFile = 1;                                                    
            this.overrideLogFile = 1;                                                 
            this.logFileName = 'EVCSim';                               

            this.logVarNames{1} = 'id'; 
            this.logVarNames{2} = 'LogIdx';                                           
            this.logVarNames{3} = 'EVCs(:,1)';
            this.logVarNames{4} = 'EVCs(:,2)';
            this.logVarNames{5} = 'ERs(:,1)';
            this.logVarNames{6} = 'ERs(:,2)';
            this.logVarNames{7} = 'RTs(:,1)';
            this.logVarNames{8} = 'RTs(:,2)';
            this.logVarNames{9} = 'ExpectedStateParam';
            this.logVarNames{10} = 'ActualStateParam';
            this.logVarNames{11} = 'ControlParamType';
            this.logVarNames{12} = 'ControlParamVal';
            this.logVarNames{13} = 'CtrlIntensities';
            this.logVarNames{14} = 'ACC_BOLD';

            this.logAddVars{1}     = 'transpose([this.EVCM.Log.Trials.typeID])';
            this.logAddVarNames{1} =  'TrialType';

            this.logAddVars{2}     = '{this.EVCM.Log.Trials.descr}.''';
            this.logAddVarNames{2} =  'TrialDescr';
            
            subjData(1).Log = [];
            %% cleanup
            
            clear temp;
        end
        
        function run(this)
            
            % set up wrapper specifications if necessary
            if(this.wrapper)
               this.plotSum = 0;
               this.printResults = 0;
               this.plotTrl = 0;
            end
            
            % run simulation
            
            this.runBatch();
            
            % post analysis
            
            this.getResults();
            
            if(this.printResults)
               this.dispResults(); 
            end
            
            if(this.plotSum)
               this.plotSummary(); 
            end
            
            if(this.printBOLD)
                this.plotBOLD();
            end
            % save params
            save(strcat(this.paramPath,'currParams.mat'), 'this');
    
        end
        
        
        function log(this)
            
            % add additional log parameters
            if(~isempty(this.logAddVars))
                for i = 1:length(this.logAddVars)
                % add parameter to log structure
                eval(strcat('this.EVCM.Log.',this.logAddVarNames{i},'=',this.logAddVars{i},';'));
                end
            end
            
            fileName = strcat(this.logFilePath, this.logFileName, '_', num2str(this.EVCM.id), '.txt');
            this.EVCM.writeLogFile(fileName, this.overrideLogFile, this.logVarNames);
        end
        
    end
    
    methods (Access = public)
       
        function runBatch(this)
            
            % set up global log parameters
            if(~isempty(this.logAddVars))
                for i = 1:length(this.logAddVars)
                % add parameter name to names list
                this.logVarNames{end+1} = this.logAddVarNames{i};
                end
            end
            
            for subj = 1:this.nSubj
                this.currSubj = subj;
                
                this.initCtrlSignals();
                if(~this.wrapper)
                    this.initTaskEnv();
                else
                    this.initOptimizationTaskEnv();
                end
                this.initEVCModel();
                this.setEVCModel();
                
                this.runSubj();
                
                this.subjData(subj).Log = this.EVCM.Log;
                
                if(this.writeLogFile)
                    this.log(); 
                end
                % disp progress
                disp(strcat('subj: ', num2str(this.currSubj), '/', num2str(this.nSubj)));
            end
            
        end
        
        function initSubj(this, subj)
                this.currSubj = subj;
                
                this.initCtrlSignals();
                this.initTaskEnv();
                this.initEVCModel();
                this.setEVCModel(); 
        end
        
        function runSubj(this)
            
            for TrialIdx = 1:length(this.EVCM.State.TaskEnv.Sequence)
                
                % perform all EVC calculations and simulate trial performance
                this.EVCM.executeTrial();      

                % plot results
                if(this.plotTrl)
                    this.plotTrial();
                end
                
                % move on
                this.EVCM.nextTrial();
                
            end
        end
        
        function [model] = initDebugging(this)
                this.currSubj = 1;
                this.initCtrlSignals();
                this.initTaskEnv();
                this.initEVCModel();
                this.setEVCModel();
                model = this.EVCM;
        end
    end
    
    methods (Access = protected)
        
        function initCtrlSignals(this)
            %this.ctrlSignals(1) = this.defaultCtrlSignal;
        end
        
        function initTaskEnv(this)
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
        end
        
        function initEVCModel(this)
            this.EVCM = EVC.EVCModel(this.ctrlSignals, this.taskEnv);
        end
        
        function setEVCModel(this)
            % set main simulation parameters
            this.EVCM.id = this.currSubj;
            this.EVCM.binaryErrors = this.binaryErrors;
            this.EVCM.setReconfigurationCost(this.reconfCostFnc);
            this.EVCM.setLearning(this.learningFnc);
            this.EVCM.setRewardFnc(this.rewardFnc);
        end
    end
    
    methods (Access = public)
        %% plot functions
        
        function plotEVC(this, varargin)
            
           subj2Plot = 1:this.nSubj;      % plot all subjects by default
           EVC2Plot = 1:2;                % plot both (expected & actual) EVC's by default
           trials2Plot = 1:this.nTrials;  % plot all trials by default
           
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: EVC type to plot (expected/actual)
              if(length(varargin) >= 2)
                 if(strcmp(varargin{2},'expected'))
                     EVC2Plot = 1;
                 end
                 if(strcmp(varargin{2},'actual'))
                     EVC2Plot = 2;
                 end
              end
              
              % 3rd argument: trials to plot
              if(length(varargin) >= 3)
                 trials2Plot = varargin{3};
              end
              
           end
           
           for i = 1:length(subj2Plot)
               for j = 1:length(EVC2Plot)
                color = [0 0 0];
                if(j == 1)
                   color = this.plotParams.expectedColor;
                end
                if(j == 2)
                   color = this.plotParams.actualColor; 
                end
                EVC = this.subjData(subj2Plot(i)).Log.EVCs(trials2Plot,EVC2Plot(j));
                plot(trials2Plot,EVC, ...
                    'Color', color, ...
                    'LineWidth',this.plotParams.lineWidth);
                hold on;
               end
           end
           hold off;
           
           % plot settings
           range = max(EVC) - min(EVC);
           ylim([min(EVC)-0.05*range, max(EVC)+0.05*range]);
           xlim([min(trials2Plot) max(trials2Plot)]);
           xlabel('Trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('EVC','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Expected Value of Control','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
           
        end
        
        function plotCtrlIntensity(this, varargin)
           subj2Plot = 1;      % plot first subject by default
           trials2Plot = 1:this.nTrials;  % plot all trials by default
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: trials to plot
              if(length(varargin) >= 2)
                 trials2Plot = varargin{2};
              end
           end
           ctrlIntensities = this.subjData(subj2Plot).Log.CtrlIntensities';
           nSignals = size(ctrlIntensities,1);
           sumCtrlIntensities = sum(ctrlIntensities,1);
           sumColor = this.plotParams.defaultColor;
           %colors = hsv(size(ctrlIntensities,1));
           colors = this.plotParams.signalColors;
           %colors = [0 0.5 0];
           plot(trials2Plot,sumCtrlIntensities(trials2Plot), ...
                    'Color', sumColor, ...
                    'LineWidth',this.plotParams.lineWidth);
           hold on;
           
           for i = 1:length(trials2Plot)
              baseline = 0;
              for j = 1:nSignals
                 plot([trials2Plot(i) trials2Plot(i)], [baseline baseline+ctrlIntensities(j,trials2Plot(i))], ...
                     this.plotParams.signalTypes{j}, ...
                    'Color', colors(j,:), ... % j
                    'LineWidth',this.plotParams.intensityLineWidth); 
                baseline = baseline + ctrlIntensities(j,trials2Plot(i));
              end
           end
           
           plot(trials2Plot,sumCtrlIntensities(trials2Plot), ...
                    'Color', sumColor, ...
                    'LineWidth',this.plotParams.lineWidth);
                
           hold off;
           
           % plot settings
           range = max(sumCtrlIntensities)-min(sumCtrlIntensities);
           if(min(sumCtrlIntensities) ~= max(sumCtrlIntensities))
               ylim([min(sumCtrlIntensities)-0.05*range, max(sumCtrlIntensities)+0.05*range]);
           end
           xlim([min(trials2Plot) max(trials2Plot)]);
           xlabel('trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('Intensity','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Control Signal Adjustments','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
        end
        
        function plotRT(this, varargin)
           subj2Plot = 1:this.nSubj;      % plot all subjects by default
           RT2Plot = 1:2;                % plot both (expected & actual) EVC's by default
           trials2Plot = 1:this.nTrials;  % plot all trials by default
           
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: EVC type to plot (expected/actual)
              if(length(varargin) >= 2)
                 if(strcmp(varargin{2},'expected'))
                     RT2Plot = 1;
                 end
                 if(strcmp(varargin{2},'actual'))
                     RT2Plot = 2;
                 end
              end
              
              % 3rd argument: trials to plot
              if(length(varargin) >= 3)
                 trials2Plot = varargin{3};
              end
              
           end
           
           for i = 1:length(subj2Plot)
               for j = 1:length(RT2Plot)
                color = [0 0 0];
                if(j == 1)
                   color = this.plotParams.expectedColor;
                end
                if(j == 2)
                   color = this.plotParams.actualColor; 
                end
                RT = this.subjData(subj2Plot(i)).Log.RTs(trials2Plot,RT2Plot(j));
                plot(trials2Plot,RT, ...
                    'Color', color, ...
                    'LineWidth',this.plotParams.lineWidth);
                hold on;
               end
           end
           hold off;
           
           % plot settings
           range = max(RT) - min(RT);
           if(min(RT) ~= max(RT))
               ylim([min(RT)-0.05*range, max(RT)+0.05*range]);
           end
           xlim([min(trials2Plot) max(trials2Plot)]);
           xlabel('Trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('RT (ms)','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Reaction Time','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
        end
        
        function plotER(this, varargin)
           subj2Plot = 1:this.nSubj;      % plot all subjects by default
           ER2Plot = 1:2;                % plot both (expected & actual) EVC's by default
           trials2Plot = 1:this.nTrials;  % plot all trials by default
           
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: EVC type to plot (expected/actual)
              if(length(varargin) >= 2)
                 if(strcmp(varargin{2},'expected'))
                     ER2Plot = 1;
                 end
                 if(strcmp(varargin{2},'actual'))
                     ER2Plot = 2;
                 end
              end
              
              % 3rd argument: trials to plot
              if(length(varargin) >= 3)
                 trials2Plot = varargin{3};
              end
              
           end
           
           for i = 1:length(subj2Plot)
               for j = 1:length(ER2Plot)
                color = [0 0 0];
                if(j == 1)
                   color = this.plotParams.expectedColor;
                end
                if(j == 2)
                   color = this.plotParams.actualColor; 
                end
                ER = this.subjData(subj2Plot(i)).Log.ERs(trials2Plot,ER2Plot(j));
                plot(trials2Plot,ER, ...
                    'Color', color, ...
                    'LineWidth',this.plotParams.lineWidth);
                hold on;
               end
           end
           hold off;
           
           % plot settings
           range = max(ER) - min(ER);
           if(min(ER) ~= max(ER))
               ylim([min(ER)-0.05*range, max(ER)+0.05*range]);
           end
           xlim([min(trials2Plot) max(trials2Plot)]);
           xlabel('Trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('ER (%)','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Error Rate','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
        end
        
        function plotOutcomeProb(this, varargin)
           subj2Plot = 1:this.nSubj;      % plot all subjects by default
           state2Plot = 1:2;                % plot both (expected & actual) EVC's by default
           Outcome2Plot = 1;             % plot first outcome by default
           trials2Plot = 1:this.nTrials;  % plot all trials by default
           
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: EVC type to plot (expected/actual)
              if(length(varargin) >= 2)
                 if(strcmp(varargin{2},'expected'))
                     state2Plot = 1;
                 end
                 if(strcmp(varargin{2},'actual'))
                     state2Plot = 2;
                 end
              end
              
              % 3rd argument: trials to plot
              if(length(varargin) >= 3)
                 Outcome2Plot = varargin{3};
              end
              
              % 3rd argument: trials to plot
              if(length(varargin) >= 4)
                 trials2Plot = varargin{4};
              end
              
              
           end
           
           for i = 1:length(subj2Plot)
               for j = 1:length(state2Plot)
                color = [0 0 0];
                if(j == 1)
                   color = this.plotParams.expectedColor;
                end
                if(j == 2)
                   color = this.plotParams.actualColor; 
                end
                if(state2Plot == 1)
                    ER = this.subjData(subj2Plot(i)).Log.expectedProbs(trials2Plot,Outcome2Plot);
                else
                    ER = this.subjData(subj2Plot(i)).Log.actualProbs(trials2Plot,Outcome2Plot);
                end
                plot(trials2Plot,ER, ...
                    'Color', color, ...
                    'LineWidth',this.plotParams.lineWidth);
                hold on;
               end
           end
           hold off;
           
           % plot settings
           range = max(ER) - min(ER);
           if(min(ER) ~= max(ER))
               ylim([min(ER)-0.05*range, max(ER)+0.05*range]);
           end
           xlim([min(trials2Plot) max(trials2Plot)]);
           xlabel('Trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('prob (%)','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Outcome Probability','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
        end
        
        function plotReward(this, varargin)
            subj2Plot = 1;
            trials2Plot = 1:this.nTrials;  % plot all trials by default
           
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: trials to plot
              if(length(varargin) >= 2)
                 trials2Plot = varargin{2};
              end
           end
           
           for i = 1:length(subj2Plot)
                color = this.plotParams.defaultColor;
                outcomeValues = vec2mat([this.subjData(1).Log.Trials(trials2Plot).outcomeValues],2);
                reward = outcomeValues(:,1)';
                plot(trials2Plot,reward, ...
                    'Color', color, ...
                    'LineWidth',this.plotParams.lineWidth);
                hold on;
           end
           hold off;
           
           % plot settings
           range = max(reward)-min(reward);
           yLimit = [min(reward)-0.05*range, max(reward)+0.05*range];
           if(yLimit(1) ~= yLimit(2))
                ylim([min(reward)-0.05*range, max(reward)+0.05*range]);
           end
           xlabel('trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('Reward','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Reward Manipulation','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
           
        end 
        
        function plotDifficulty(this, varargin)
            subj2Plot = 1;
            trials2Plot = 1:this.nTrials;  % plot all trials by default
           
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: trials to plot
              if(length(varargin) >= 2)
                 trials2Plot = varargin{2};
              end
           end
           
           for i = 1:length(subj2Plot)
                color = this.plotParams.defaultColor;
                stimSaliency = vec2mat([this.subjData(1).Log.Trials(trials2Plot).stimSalience],2);
                difficulty = stimSaliency(:,1)';
                plot(trials2Plot,difficulty, ...
                    'Color', color, ...
                    'LineWidth',this.plotParams.lineWidth);
                hold on;
           end
           hold off;
           
           % plot settings
           range = max(difficulty)-min(difficulty);
           if(min(difficulty) ~= max(difficulty))
            ylim([min(difficulty)-0.05*range, max(difficulty)+0.05*range]);
           end
           xlim([min(trials2Plot) max(trials2Plot)]);
           xlabel('Trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('Target Saliency','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Stimulus Saliency Manipulation (Difficulty)','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
           
        end
  
        function plotCondition(this, varargin)
            subj2Plot = 1;
            conditionIdx = 1;
            trials2Plot = 1:this.nTrials;  % plot all trials by default
           
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: trials to plot
              if(length(varargin) >= 2)
                 trials2Plot = varargin{2};
              end
              
              % 3rd argument: conditions to plot
              if(length(varargin) >= 3)
                 conditionIdx = varargin{3};
              end
           end
           
           for i = 1:length(subj2Plot)
                color = this.plotParams.defaultColor;
                subjLog = this.subjData(i).Log;
                
                for j = 1:length(conditionIdx)
                    
                    conditions =  reshape([subjLog.TrialsOrg.conditions],[length(subjLog.TrialsOrg(1).conditions) this.nTrials])';
                    plotCondition = conditions(:,j);
                    plot(trials2Plot,plotCondition, ...
                        'Color', color, ...
                        'LineWidth',this.plotParams.lineWidth);
                    hold on;
                end
           end
           hold off;
           
           % plot settings
           range = max(plotCondition)-min(plotCondition);
           ylim([min(plotCondition)-0.05*range, max(plotCondition)+0.05*range]);
           xlabel('trials','FontSize',this.plotParams.axisFontSize); % x-axis label
           ylabel('Condition','FontSize',this.plotParams.axisFontSize); % y-axis label
           title('Manipulation','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
           
        end 
        
        function plotConditionsBackground(this, condition)

           taskConditions = reshape([this.EVCM.Log.TrialsOrg.conditions],[size(this.EVCM.Log.TrialsOrg(1).conditions,2) this.nTrials])';
           task = taskConditions(:,condition);
           
           % align task numbers such that the minimum task number can be
           % used to index the first element of an array
           task = task + (1-min(task));
           
           % find starting and ending points of task blocks
           taskBlock(1) = task(1);
           taskBlockStart(1) = 1;
           taskBlockEnd = [];
           for i = 2:length(task)
              if(task(i) ~= task(i-1))
                  taskBlockEnd = [taskBlockEnd (i-1)];
                  taskBlockStart = [taskBlockStart i];
                  taskBlock = [taskBlock task(i)];
              end 
           end
           taskBlockEnd = [taskBlockEnd length(task)];
           
           % plot different tasks as different backgrounds
           yRange = [-100 100];
           plot(1:this.nTrials, linspace(yRange(1),yRange(2),this.nTrials),'w');
           hold on;
           for i = 1:length(taskBlock)
           p = patch([taskBlockStart(i) taskBlockStart(i) taskBlockEnd(i) taskBlockEnd(i)], ...
                     [yRange(1) yRange(2) yRange(2) yRange(1)], ...
                     this.plotParams.signalColors(taskBlock(i),:));
           set(p,'FaceAlpha',0.2); 
           end

        end
        
        function plotBOLD(this, varargin)
            subj2Plot = 1;
            conditionIdx = [];
            trials2Plot = 1:this.nTrials;  % plot all trials by default
            trialLength = 0;
            BOLDIdx = 1:8;
            condNames = {};
            
           if(~isempty(varargin))
              % 1st argument: subjects to plot
              subj2Plot =  varargin{1};
              
              % 2nd argument: trials to plot
              if(length(varargin) >= 2)
                 trials2Plot = varargin{2};
              end
              
              % 3rd argument: conditions to plot
              if(length(varargin) >= 3)
                 BOLDIdx = varargin{3};
              end
              
              % 4th argument: conditions to plot
              if(length(varargin) >= 4)
                 conditionIdx = varargin{4};
              end
              
              % 5th argument: condition names
              if(length(varargin) >= 5)
                 condNames = varargin{5};
              end
              
              % 6th argument: trial length for convolution
              % if 0, no convolution will be performed
              if(length(varargin) >= 6)
                 trialLength = varargin{6};
              end
           end
           
           for i = 1:length(subj2Plot)
                color = this.plotParams.defaultColor;
                conditionColors = this.plotParams.conditionColors;
                subjLog = this.subjData(i).Log;
                ACC_regressor = zscore(subjLog.ACC_BOLD(trials2Plot,:))';
                conditions =  reshape([subjLog.TrialsOrg.conditions],[length(subjLog.TrialsOrg(1).conditions) this.nTrials])';
                if(~isempty(conditionIdx))
                    cond = conditions(:,conditionIdx(1)); % for now, display just one condition
                end
                
                % convolute regressor w/ Gamma function
                if(trialLength > 0)
                    
                end
                
                figure();
                NPlots = length(BOLDIdx);
                
                for j = 1:length(BOLDIdx)
                    
                   subplot(NPlots,1,j);
                   plot(trials2Plot, zeros(1,length(trials2Plot))); hold on;
                   
                   ylimit = [min(ACC_regressor(BOLDIdx(j),:)), max(ACC_regressor(BOLDIdx(j),:))];
                   % draw conditions
                   if(~isempty(conditionIdx))
                       for k = 1:length(trials2Plot)
                           condColor = conditionColors(cond(trials2Plot(k))+min(cond)+1,:);
                           rectangle('Position',[trials2Plot(k) ylimit(1) 1 ylimit(2)-ylimit(1)],'FaceColor',condColor);
                       end
                   end
                   
                   plot(trials2Plot,ACC_regressor(BOLDIdx(j),:), ...
                        'Color', color, ...
                        'LineWidth',this.plotParams.lineWidth);
                   hold off;
                    
                   ylim(ylimit);
                   
                   if(j == length(BOLDIdx))
                       xlabel('trials','FontSize',this.plotParams.axisFontSize); % x-axis label
                   end
                   
                   if(j == 1)
                      title('BOLD response','FontSize',this.plotParams.titleFontSize,'FontWeight','bold'); 
                      
                      if(~isempty(conditionIdx))
                          allConds = unique(cond);
                          for k = 1:length(allConds)
                            currCond = allConds(k)+min(allConds)+1;
                            if(~isempty(condNames))
                               condLabel = condNames(k);
                            else
                                condLabel = strcat('cond',num2str(currCond));
                            end
                            annotation('textbox',[0.2*k 0.01 0.2 0.03], 'String',condLabel,'Color',color,'BackgroundColor',conditionColors(currCond,:),'FontWeight','bold');
                          end
                      end
                   end
                   
                   switch BOLDIdx(j)
                       case 1
                            ylabel('abs(\DeltaEVCM)','FontSize',this.plotParams.axisFontSize); % y-axis label
                       case 2
                            ylabel('\DeltaEVCM','FontSize',this.plotParams.axisFontSize); % y-axis label
                       case 3
                            ylabel('EVC(u)','FontSize',this.plotParams.axisFontSize); % y-axis label
                       case 4
                            ylabel('sum(u)','FontSize',this.plotParams.axisFontSize); % y-axis label
                       case 5
                            ylabel('Costs(u)','FontSize',this.plotParams.axisFontSize); % y-axis label
                       case 6
                            ylabel('Costs(\Deltau)','FontSize',this.plotParams.axisFontSize); % y-axis label
                       case 7
                            ylabel('abs(\Deltau)','FontSize',this.plotParams.axisFontSize); % y-axis label
                       case 8
                            ylabel('\Deltau','FontSize',this.plotParams.axisFontSize); % y-axis label
                   end
                end
                
                if(~isempty(conditionIdx))
                    figure();
                    allConds = unique(cond);
                    
                    for j = 1:length(BOLDIdx)
                        nRows = ceil(sqrt(length(BOLDIdx)));
                        nCols = ceil(sqrt(length(BOLDIdx)));

                        subplot(nRows,nCols,j);
                        
                        barData = zeros(1,length(allConds));
                        errData = zeros(1,length(allConds));
                        for k = 1:length(allConds)
                           currCond = allConds(k);
                           relevantTrialData = ACC_regressor(BOLDIdx(j),cond(trials2Plot) == currCond);
                           barData(k) = mean(relevantTrialData);
                           errData(k) = std(relevantTrialData)/sqrt(length(relevantTrialData));
                        end
                        
                        b = bar(allConds+1,barData); hold on;
                        errorbar(allConds+1,barData,errData);
                        set(b(:),'FaceColor',[0.5 0.5 0.5]);
                        hold off;
                        
                        if(~isempty(condNames))
                            set(gca,'XTickLabel',condNames)
                        end
                        
                        %ylimit = [min(ACC_regressor(BOLDIdx(j),:)), max(ACC_regressor(BOLDIdx(j),:))];
                        ylimit = [-1 1]*6;
                        ylim(ylimit);
                        
                        ylabel('BOLD','FontSize',this.plotParams.axisFontSize); % y-axis label

                        switch BOLDIdx(j)
                           case 1
                                title('abs(\DeltaEVCM)','FontSize',this.plotParams.axisFontSize); % y-axis label
                           case 2
                                title('\DeltaEVCM','FontSize',this.plotParams.axisFontSize); % y-axis label
                           case 3
                                title('EVC(u)','FontSize',this.plotParams.axisFontSize); % y-axis label
                           case 4
                                title('sum(u)','FontSize',this.plotParams.axisFontSize); % y-axis label
                           case 5
                                title('Costs(u)','FontSize',this.plotParams.axisFontSize); % y-axis label
                           case 6
                                title('Costs(\Deltau)','FontSize',this.plotParams.axisFontSize); % y-axis label
                           case 7
                                title('abs(\Deltau)','FontSize',this.plotParams.axisFontSize); % y-axis label
                           case 8
                                title('\Deltau','FontSize',this.plotParams.axisFontSize); % y-axis label
                       end
                    end
                end
                
           end
           
        end
        
        function initOptimizationTaskEnv(this)
            this.initTaskEnv();
        end
        

    end
    
    % virtual functions need to be implemented by subclasses
    methods (Abstract)
        
        getResults(this)
        
        dispResults(this)
        
        plotTrial(this)
        
        plotSummary(this)
        
    end
    
end