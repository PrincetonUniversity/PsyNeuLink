classdef MSDDMTaskSwitchingBlocked < Simulations.MSDDMSim
    
    % description of class
    % simulates a blocked task switching paradigm (see Rogers & Monsell, 1995)
    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = MSDDMTaskSwitchingBlocked()
            import EVC.*;
            import EVC.MSDDM.*;
            % call parent constructor
            this = this@Simulations.MSDDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 10;
            this.plotSum = true;
            
            temp.reconfCostFnc.params{1} = 0; % see if we can recover switch effects from MSDDM dynamics
            temp.reconfCostFnc.params{2} = 0;
            temp.reconfCostFnc.type = EVCFnc.EXP;
            this.reconfCostFnc = EVCFnc(temp.reconfCostFnc.type, temp.reconfCostFnc.params);
            
            %% control parameters: default control signal
            this.ctrlSignals(1) = this.defaultCtrlSignal;
            this.ctrlSignals(1).CtrlSigStimMap  = [1 0]; 
            this.ctrlSignals(1).IntensityRange  = 0:0.02:1; 
            this.ctrlSignals(2) = CtrlSignal(this.defaultCtrlSignal);
            this.ctrlSignals(2).CtrlSigStimMap  = [0 1]; 
            this.ctrlSignals(2).IntensityRange  = 0:0.02:1; 
            
            % map all control signals to a specific DDM parameter
            temp.taskAControlFnc.type = MSDDMFnc.INTENSITY2DDM;
            temp.taskAControlFnc.params{1} = this.ctrlSignals(1);
            temp.taskAControlFnc.params{2} = this.defaultControlProxy;
            temp.taskAControlFnc.params{3} = this.defaultControlMappingFnc;
            temp.ControlFncA = MSDDMFnc(temp.taskAControlFnc.type, ...
                                            temp.taskAControlFnc.params);
                                        
            temp.taskBControlFnc.type = MSDDMFnc.INTENSITY2DDM;
            temp.taskBControlFnc.params{1} = this.ctrlSignals(2);
            temp.taskBControlFnc.params{2} = this.defaultControlProxy;
            temp.taskBControlFnc.params{3} = this.defaultControlMappingFnc;
            temp.ControlFncB = MSDDMFnc(temp.taskBControlFnc.type, ...
                                            temp.taskBControlFnc.params); 
                                        
            temp.taskInertiaFnc.type = MSDDMFnc.PREV_INTENSITY2DDM;
            temp.taskInertiaFnc.params{1} = this.defaultControlProxy;
            temp.taskInertiaFnc.params{2} = this.defaultControlMappingFnc;
            temp.inertiaFnc = MSDDMFnc(temp.taskInertiaFnc.type, ...
                                            temp.taskInertiaFnc.params); 
                                        
            % define each DDM process   
            
            
            temp.ControlProcessA = MSDDMProc(MSDDMProc.CONTROL, ...                 
                                                    MSDDMProc.DRIFT, ...
                                                    temp.ControlFncA, ...
                                                    this.defaultControlStage, ...
                                                    this.defaultControlDuration);   
                                                
            temp.ControlProcessB = MSDDMProc(MSDDMProc.CONTROL, ...                 
                                                    MSDDMProc.DRIFT, ...
                                                    temp.ControlFncB, ...
                                                    this.defaultControlStage, ...
                                                    this.defaultControlDuration);                                    
                                        
            
            temp.inertiaProcess = MSDDMProc(MSDDMProc.DEFAULT, ...        % default automatic DDM process for actual & expected state
                                                    MSDDMProc.DRIFT, ...
                                                    temp.inertiaFnc, ...
                                                    this.defaultAutomaticStage, ...
                                                    this.defaultAutomaticDuration);
                                                
            % put all DDM processes together
            this.MSDDMProcesses(1) = temp.ControlProcessA;
            this.MSDDMProcesses(end+1) = temp.ControlProcessB;
            this.MSDDMProcesses(end+1) = temp.inertiaProcess;
            
            %% task environment parameters: trial
            this.rewardVal = 20;
            % create an incongruent trial for Task A
            this.trials(1).ID = 1;                                                          % trial identification number (for task set)
            this.trials(1).typeID = 1;                                                      % trial type (defines task context)
            this.trials(1).cueID = 1;                                                       % cued information about trial identity
            this.trials(1).descr = 'A_inc';                                                 % trial description
            this.trials(1).conditions    = [0 1];                                           % set of trial conditions (for logging)
            this.trials(1).outcomeValues = [this.rewardVal 0];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(1).stimSalience  = [1 1];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(1).stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            0 1];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(1).params = [];                                                   % DDM specific trial parameters

            this.trials(2).ID = 2;                                                          % trial identification number (for task set)
            this.trials(2).typeID = 2;                                                        % trial type (defines task context)
            this.trials(2).cueID = 1;                                                       % cued information about trial identity
            this.trials(2).descr = 'A_con';                                                 % trial description
            this.trials(2).conditions    = [0 0];                                           % set of trial conditions (for logging)
            this.trials(2).outcomeValues = [this.rewardVal 0];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(2).stimSalience  = [1 1];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(2).stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            1 0];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(2).params = [];                                                   % DDM specific trial parameters

            this.trials(3).ID = 3;                                                          % trial identification number (for task set)
            this.trials(3).typeID = 3;                                                        % trial type (defines task context)
            this.trials(3).cueID = 2;                                                       % cued information about trial identity
            this.trials(3).descr = 'B_inc';                                                 % trial description
            this.trials(3).conditions    = [1 1];                                           % set of trial conditions (for logging)
            this.trials(3).outcomeValues = [0 this.rewardVal];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(3).stimSalience  = [1 1];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(3).stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            0 1];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(3).params = [];                                                   % DDM specific trial parameters

            this.trials(4).ID = 4;                                                          % trial identification number (for task set)
            this.trials(4).typeID = 4;                                                        % trial type (defines task context)
            this.trials(4).cueID = 2;                                                       % cued information about trial identity
            this.trials(4).descr = 'B_con';                                                 % trial description
            this.trials(4).conditions    = [1 0];                                           % set of trial conditions (for logging)
            this.trials(4).outcomeValues = [0 this.rewardVal];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(4).stimSalience  = [1 1];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(4).stimRespMap   = [0 1;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            0 1];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(4).params = [];                                                   % DDM specific trial parameters

            
            %% task environment parameters: task environment
            
            this.nTrials = 20;
            
            %% log parameters
            this.writeLogFile = 1; 
            this.logFileName = 'DDM_TS_WithinBetweenCorr_C2'; 
            
            this.logAddVars{3} = 'reshape([this.EVCM.Log.Trials.conditions],3,this.nTrials)''';
            this.logAddVarNames{3} = 'condition';
        end
        
        function getResults(this)
            
            
            for subj = 1:this.nSubj
                
                subjLog = this.subjData(subj).Log;
                conditions = reshape([subjLog.TrialsOrg.conditions],[3 this.nTrials])';
                task = conditions(:,1);
                congruency = conditions(:,2);
                transition = conditions(:,3);
                
                % retrieve intensities and delta-reward from log data
                RT = this.subjData(subj).Log.RTs(:,2)';
                ER = this.subjData(subj).Log.ERs(:,2)';
                %ctrlIntensity = this.subjData(subj).Log.CtrlIntensities(:,1)';

                switch_RT = RT(transition == 1);
                rep_RT = RT(transition == 0);
                switch_ER = ER(transition == 1);
                rep_ER = ER(transition == 0);
                
                con_RT = RT(congruency == 0);
                inc_RT = RT(congruency == 1);
                con_ER = ER(congruency == 0);
                inc_ER = ER(congruency == 1);
                
                taskA_RT = RT(task == 0);
                taskB_RT = RT(task == 1);
                taskA_ER = ER(task == 0);
                taskB_ER = ER(task == 1);
                
                % extract relevant test vectors
                this.results.switch_RT(subj) = mean(switch_RT);
                this.results.rep_RT(subj) = mean(rep_RT);
                this.results.switch_ER(subj) = mean(switch_ER);
                this.results.rep_ER(subj) = mean(rep_ER);
                this.results.con_RT(subj) = mean(con_RT);
                this.results.inc_RT(subj) = mean(inc_RT);
                this.results.con_ER(subj) = mean(con_ER);
                this.results.inc_ER(subj) = mean(inc_ER);
                this.results.taskA_RT(subj) = mean(taskA_RT);
                this.results.taskB_RT(subj) = mean(taskB_RT);
                this.results.taskA_ER(subj) = mean(taskA_ER);
                this.results.taskB_ER(subj) = mean(taskB_ER);
            end
            
            % switch costs tests
            [this.results.h_switchCostsRT this.results.p_switchCostsRT] = ttest(this.results.switch_RT-this.results.rep_RT,0);
            this.results.switch_RT_mean = mean(this.results.switch_RT);
            this.results.rep_RT_mean = mean(this.results.rep_RT);
            
            [this.results.h_switchCostsER this.results.p_switchCostsER] = ttest(this.results.switch_ER-this.results.rep_ER,0);
            this.results.switch_ER_mean = mean(this.results.switch_ER);
            this.results.rep_ER_mean = mean(this.results.rep_ER);
            
            % incongruency tests
            [this.results.h_incCostsRT this.results.p_incCostsRT] = ttest(this.results.inc_RT-this.results.con_RT,0);
            this.results.con_RT_mean = mean(this.results.con_RT);
            this.results.inc_RT_mean = mean(this.results.inc_RT);
            
            [this.results.h_incCostsER this.results.p_incCostsER] = ttest(this.results.inc_ER-this.results.con_ER,0);
            this.results.con_ER_mean = mean(this.results.con_ER);
            this.results.inc_ER_mean = mean(this.results.inc_ER);
            
            % task tests
            [this.results.h_taskCostsRT this.results.p_taskCostsRT] = ttest(this.results.taskA_RT-this.results.taskB_RT,0);
            this.results.taskA_RT_mean = mean(this.results.taskA_RT);
            this.results.taskB_RT_mean = mean(this.results.taskB_RT);
            
            [this.results.h_taskCostsER this.results.p_taskCostsER] = ttest(this.results.taskA_ER-this.results.taskB_ER,0);
            this.results.taskA_ER_mean = mean(this.results.taskA_ER);
            this.results.taskB_ER_mean = mean(this.results.taskB_ER);
        end
        
        function dispResults(this)
            disp('++++++++++ DDMTaskSwitchingBlocked ++++++++++');
        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:(this.nTrials/1);
            f1 = figure(1);
            set(f1, 'Position', [0 0 1200 500])
%             subplot(3,1,1);
%             this.plotEVC(exampleSubj, 'expected', sampleTrials);
%             subplot(3,1,2);
%             this.plotRT(exampleSubj, 'actual', sampleTrials);
%             subplot(3,1,3);
            this.plotTrialConditions();
            %this.plotCtrlIntensity(exampleSubj, sampleTrials);
            this.plotRT(exampleSubj, 'actual', sampleTrials);
            
            f2 = figure(2);
            set(f2, 'Position', [600 0 650 200])
            this.plotRTResults();
            
            f3 = figure(3);
            set(f3, 'Position', [600 200 650 200])
            this.plotERResults();
        end
        
        function plotTrialConditions(this)

           taskConditions = reshape([this.EVCM.Log.TrialsOrg.conditions],[3 this.nTrials])';
           task = taskConditions(:,1);
           
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
        
        function plotRTResults(this)
            subplot(1,3,1);
            bar(1,[this.results.switch_RT_mean],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semBar1 = std(this.results.switch_RT)/sqrt(length(this.results.switch_RT));
            semBar2 = std(this.results.rep_RT)/sqrt(length(this.results.rep_RT));
            errorbar(1,[this.results.switch_RT_mean],semBar1,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.rep_RT_mean],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.rep_RT_mean],semBar2,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.switch_RT_mean, this.results.rep_RT_mean];
            range = max(max(abs(bardata)));
            ylim([0 range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'Switch', 'Rep'},'fontsize',this.plotParams.axisFontSize);
            xlabel('Transition');
            ylabel('RT (ms)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('Switch Costs','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            subplot(1,3,2);
            bar(1,[this.results.inc_RT_mean],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semBar1 = std(this.results.inc_RT)/sqrt(length(this.results.inc_RT));
            semBar2 = std(this.results.con_RT)/sqrt(length(this.results.con_RT));
            errorbar(1,[this.results.inc_RT_mean],semBar1,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.con_RT_mean],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.con_RT_mean],semBar2,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.inc_RT_mean, this.results.con_RT_mean];
            range = max(max(abs(bardata)));
            ylim([0 range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'Inc', 'Con'},'fontsize',this.plotParams.axisFontSize);
            xlabel('Congruency');
            ylabel('RT (ms)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('Incongruency Costs','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            subplot(1,3,3);
            bar(1,[this.results.taskA_RT_mean],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semBar1 = std(this.results.taskA_RT)/sqrt(length(this.results.taskA_RT));
            semBar2 = std(this.results.taskB_RT)/sqrt(length(this.results.taskB_RT));
            errorbar(1,[this.results.taskA_RT_mean],semBar1,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.taskB_RT_mean],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.taskB_RT_mean],semBar2,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.taskA_RT_mean, this.results.taskB_RT_mean];
            range = max(max(abs(bardata)));
            ylim([0 range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'A', 'B'},'fontsize',this.plotParams.axisFontSize);
            xlabel('Task');
            ylabel('RT (ms)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('Task Differences','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            
        end
        
        function plotERResults(this)
            %%
            subplot(1,3,1);
            bar(1,[this.results.switch_ER_mean],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semBar1 = std(this.results.switch_ER)/sqrt(length(this.results.switch_ER));
            semBar2 = std(this.results.rep_ER)/sqrt(length(this.results.rep_ER));
            errorbar(1,[this.results.switch_ER_mean],semBar1,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.rep_ER_mean],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.rep_ER_mean],semBar2,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.switch_ER_mean, this.results.rep_ER_mean];
            range = max(max(abs(bardata)));
            ylim([0 range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'Switch', 'Rep'},'fontsize',this.plotParams.axisFontSize);
            xlabel('Transition');
            ylabel('ER (%)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('Switch Costs','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            subplot(1,3,2);
            bar(1,[this.results.inc_ER_mean],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semBar1 = std(this.results.inc_ER)/sqrt(length(this.results.inc_ER));
            semBar2 = std(this.results.con_ER)/sqrt(length(this.results.con_ER));
            errorbar(1,[this.results.inc_ER_mean],semBar1,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.con_ER_mean],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.con_ER_mean],semBar2,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.inc_ER_mean, this.results.con_ER_mean];
            range = max(max(abs(bardata)));
            ylim([0 range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'Inc', 'Con'},'fontsize',this.plotParams.axisFontSize);
            xlabel('Congruency');
            ylabel('ER (%)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('Incongruency Costs','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            subplot(1,3,3);
            bar(1,[this.results.taskA_ER_mean],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semBar1 = std(this.results.taskA_ER)/sqrt(length(this.results.taskA_ER));
            semBar2 = std(this.results.taskB_ER)/sqrt(length(this.results.taskB_ER));
            errorbar(1,[this.results.taskA_ER_mean],semBar1,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.taskB_ER_mean],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.taskB_ER_mean],semBar2,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.taskA_ER_mean, this.results.taskB_ER_mean];
            range = max(max(abs(bardata)));
            ylim([0 range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'A', 'B'},'fontsize',this.plotParams.axisFontSize);
            xlabel('Task');
            ylabel('ER (%)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('Task Differences','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            
        end
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            import EVC.*;
            
            % build & randomize task sequence  
            trialProps = [2 1 2 1];      % defines relative proportion of each trial  
            blockSize = 5;
            this.taskEnv = TaskEnv.blockedTaskSwitchDesign(this.trials, this.nTrials, blockSize, trialProps);          
        end
        
        function initCtrlSignals(this)
        end
        
    end
    
end

