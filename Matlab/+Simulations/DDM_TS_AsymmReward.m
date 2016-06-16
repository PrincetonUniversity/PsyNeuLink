classdef DDM_TS_AsymmReward < Simulations.DDMSim
    
    % description of class
    % simulates a blocked task switching paradigm with one task being more 
    % rewarded than the other (see Umemoto & Holroyd, 2014)

    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDM_TS_AsymmReward()
            import EVC.*;
            import EVC.DDM.*;
            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 10;
            this.plotSum = true;
            
            this.defaultCostFnc.params{1} = 4;%4.2;
            this.defaultCostFnc.params{2} = -3;%-3;
            
            temp.reconfCostFnc.params{1} = 5.7;%5.7;%6.4; %5.7; %
            temp.reconfCostFnc.params{2} = -4;%-4;
            temp.reconfCostFnc.type = EVCFnc.EXP;
            this.reconfCostFnc = EVCFnc(temp.reconfCostFnc.type, temp.reconfCostFnc.params);
            
            %this.taskSetInertia = 0.15;
            
            this.learningFnc(1).params{1} = 1;
            this.defaultDDMParams.c = 0.699; %0.663;
            this.defaultDDMParams.thresh = 1.06;
            this.defaultAutomaticFnc.params{1} = [-0.1;0.1];
            
            %% control parameters: default control signal
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
            this.rewardVal = 10;
            rewardB = 8;
            
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
            this.trials(2).cueID = 2;                                                       % cued information about trial identity
            this.trials(2).descr = 'B_inc';                                                 % trial description
            this.trials(2).conditions    = [1 1];                                           % set of trial conditions (for logging)
            this.trials(2).outcomeValues = [0 rewardB];                                           % reward for correct outcome = 3; no reward/punishment for incorrect outcome
            this.trials(2).stimSalience  = [1 1];                                       % relative salience between stimulus 1 and stimulus 2 defines level of incongruency; here stim 2 is more dominant (like in stroop)
            this.trials(2).stimRespMap   = [1 0;                                            % responding to stimulus 1 tends to produce first outcome by 100%
                                            0 1];                                          % responding to stimulus 3 tends to produce second outcome by 100%
            this.trials(2).params = [];                                                   % DDM specific trial parameters
                                                 % DDM specific trial parameters

            
            %% task environment parameters: task environment
            
            this.nTrials = 120;
            
            %% log parameters
            this.writeLogFile = 1; 
            this.logFileName = 'DDM_TS_AsymmReward'; 
            
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
                
                taskA_switch_RT = RT(transition == 1 & task == 0);
                taskA_rep_RT = RT(transition == 0 & task == 0);
                taskB_switch_RT = RT(transition == 1 & task == 1);
                taskB_rep_RT = RT(transition == 0 & task == 1);
                taskA_switch_ER = ER(transition == 1 & task == 0);
                taskA_rep_ER = ER(transition == 0 & task == 0);
                taskB_switch_ER = ER(transition == 1 & task == 1);
                taskB_rep_ER = ER(transition == 0 & task == 1);
                
                % extract relevant test vectors
                this.results.RT.switch(subj) = mean(switch_RT);
                this.results.RT.rep(subj) = mean(rep_RT);
                this.results.ER.switch(subj) = mean(switch_ER);
                this.results.ER.rep(subj) = mean(rep_ER);
                this.results.RT.con(subj) = mean(con_RT);
                this.results.RT.inc(subj) = mean(inc_RT);
                this.results.ER.con(subj) = mean(con_ER);
                this.results.ER.inc(subj) = mean(inc_ER);
                this.results.RT.taskA(subj) = mean(taskA_RT);
                this.results.RT.taskB(subj) = mean(taskB_RT);
                this.results.ER.taskA(subj) = mean(taskA_ER);
                this.results.ER.taskB(subj) = mean(taskB_ER);
                this.results.RT.taskA_switch(subj) = mean(taskA_switch_RT)*1000;
                this.results.RT.taskA_rep(subj) = mean(taskA_rep_RT)*1000;
                this.results.RT.taskB_switch(subj) = mean(taskB_switch_RT)*1000;
                this.results.RT.taskB_rep(subj) = mean(taskB_rep_RT)*1000;
                this.results.ER.taskA_switch(subj) = mean(taskA_switch_ER);
                this.results.ER.taskA_rep(subj) = mean(taskA_rep_ER);
                this.results.ER.taskB_switch(subj) = mean(taskB_switch_ER);
                this.results.ER.taskB_rep(subj) = mean(taskB_rep_ER);
                
            end
            
            % switch costs tests
            [this.results.h_switchCostsRT this.results.p_switchCostsRT] = ttest(this.results.RT.switch-this.results.RT.rep,0);
            this.results.switch_RT_mean = mean(this.results.RT.switch);
            this.results.rep_RT_mean = mean(this.results.RT.rep);
            
            [this.results.h_switchCostsER this.results.p_switchCostsER] = ttest(this.results.ER.switch-this.results.ER.rep,0);
            this.results.switch_ER_mean = mean(this.results.ER.switch);
            this.results.rep_ER_mean = mean(this.results.ER.rep);
            
            % incongruency tests
            [this.results.h_incCostsRT this.results.p_incCostsRT] = ttest(this.results.RT.inc-this.results.RT.con,0);
            this.results.con_RT_mean = mean(this.results.RT.con);
            this.results.inc_RT_mean = mean(this.results.RT.inc);
            
            [this.results.h_incCostsER this.results.p_incCostsER] = ttest(this.results.ER.inc-this.results.ER.con,0);
            this.results.con_ER_mean = mean(this.results.ER.con);
            this.results.inc_ER_mean = mean(this.results.ER.inc);
            
            % task tests
            [this.results.h_taskCostsRT this.results.p_taskCostsRT] = ttest(this.results.RT.taskA-this.results.RT.taskB,0);
            this.results.taskA_RT_mean = mean(this.results.RT.taskA);
            this.results.taskB_RT_mean = mean(this.results.RT.taskB);
            
            [this.results.h_taskCostsER this.results.p_taskCostsER] = ttest(this.results.ER.taskA-this.results.ER.taskB,0);
            this.results.taskA_ER_mean = mean(this.results.ER.taskA);
            this.results.taskB_ER_mean = mean(this.results.ER.taskB);
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
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            %this.plotRT(exampleSubj, 'actual', sampleTrials);
            
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
            trialProps = [1 1]; % defines relative proportion of each trial
            trials(1) = this.trials(1);
            trials(2) = this.trials(2);
            this.taskEnv = TaskEnv.blockedTaskSwitchDesign(trials, this.nTrials, 20, trialProps);        
        end
        
        function initCtrlSignals(this)
        end
        
    end
    
end

