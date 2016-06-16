classdef DDMRewardSanity < Simulations.DDMSim
    
    % description of class
    % runs a simulation with systematic reward manipulations of the current task

    
    % global parameters
    properties
        
    end
    
    methods
        
        function this = DDMRewardSanity()

            % call parent constructor
            this = this@Simulations.DDMSim();
            
            %% general simulation parameters
            
            this.nSubj = 10;
            this.plotSum = true;
            
            %% task environment parameters: task environment
            
            this.nTrials = 400;
            
            %% log parameters
            
            this.writeLogFile = 1; 
            this.logFileName = 'DDM_RewardSanity'; 
            
            this.logAddVars{3} = '[this.EVCM.Log.Trials.conditions]''';
            this.logAddVarNames{3} = 'condition';
        end
        
        function getResults(this)
            % loop through all subjects
            
            for subj = 1:this.nSubj
                % retrieve intensities and delta-reward from log data
                rewDiff = zeros(1,length(this.subjData(subj).Log.Trials)-1);
                ctrlDiff = zeros(1,length(this.subjData(subj).Log.Trials)-1);
                RTDiff = zeros(1,length(this.subjData(subj).Log.Trials)-1);
                ERDiff = zeros(1,length(this.subjData(subj).Log.Trials)-1);
                ctrlIntensity = this.subjData(subj).Log.CtrlIntensities(:,1)';
                RT = this.subjData(subj).Log.RTs(:,2)';
                ER = this.subjData(subj).Log.ERs(:,2)';
                for i = 2:length(this.subjData(subj).Log.Trials)-1
                    rewDiff(i-1) = this.subjData(subj).Log.Trials(i).conditions(1) - this.subjData(subj).Log.Trials(i-1).conditions(1);
                    ctrlDiff(i-1) = ctrlIntensity(i+1) - ctrlIntensity(i);
                    RTDiff(i-1) = RT(i+1) - RT(i);
                    ERDiff(i-1) = ER(i+1) - ER(i);
                end

                % extract relevant test vectors
                blockRewDiff = rewDiff(1:floor(this.nTrials/2)-1);
                blockIntensity = ctrlDiff(1:floor(this.nTrials/2)-1);
                blockRT = RTDiff(1:floor(this.nTrials/2)-1);
                blockER = ERDiff(1:floor(this.nTrials/2)-1);
                
                randomRewDiff = rewDiff((this.nTrials/2):end);
                randomIntensity = ctrlDiff((this.nTrials/2):end);
                randomRT = RTDiff((this.nTrials/2):end);
                randomER = ERDiff((this.nTrials/2):end);

                this.results.blockIncrRew_Ctrl(subj) = mean(blockIntensity(blockRewDiff > 0));
                this.results.blockDecrRew_Ctrl(subj) = mean(blockIntensity(blockRewDiff < 0));
                this.results.blockIncrRew_RT(subj) = mean(blockRT(blockRewDiff > 0));
                this.results.blockDecrRew_RT(subj) = mean(blockRT(blockRewDiff < 0));
                this.results.blockIncrRew_ER(subj) = mean(blockER(blockRewDiff > 0));
                this.results.blockDecrRew_ER(subj) = mean(blockER(blockRewDiff < 0));
                
                this.results.randomIncrRew_Ctrl(subj) = mean(randomIntensity(randomRewDiff > 0));
                this.results.randomDecrRew_Ctrl(subj) = mean(randomIntensity(randomRewDiff < 0));
                this.results.randomIncrRew_RT(subj) = mean(randomRT(randomRewDiff > 0));
                this.results.randomDecrRew_RT(subj) = mean(randomRT(randomRewDiff < 0));
                this.results.randomIncrRew_ER(subj) = mean(randomER(randomRewDiff > 0));
                this.results.randomDecrRew_ER(subj) = mean(randomER(randomRewDiff < 0));
                
                this.results.fullIncrRew_Ctrl(subj) = mean(ctrlDiff(rewDiff > 0));
                this.results.fullDecrRew_Ctrl(subj) = mean(ctrlDiff(rewDiff < 0));
                this.results.fullIncrRew_RT(subj) = mean(RTDiff(rewDiff > 0));
                this.results.fullDecrRew_RT(subj) = mean(RTDiff(rewDiff < 0));
                this.results.fullIncrRew_ER(subj) = mean(ERDiff(rewDiff > 0));
                this.results.fullDecrRew_ER(subj) = mean(ERDiff(rewDiff < 0));
            end
            
            % reward increase tests
            
            [this.results.hBlock_Ctrl_Incr this.results.pBlock_Ctrl_Incr] = ttest(this.results.blockIncrRew_Ctrl,0);
            this.results.resultBlock_Ctrl_Incr = mean(this.results.blockIncrRew_Ctrl);        
            [this.results.hBlock_RT_Incr this.results.pBlock_RT_Incr] = ttest(this.results.blockIncrRew_RT,0);
            this.results.resultBlock_RT_Incr = mean(this.results.blockIncrRew_RT);
            [this.results.hBlock_ER_Incr this.results.pBlock_ER_Incr] = ttest(this.results.blockIncrRew_ER,0);
            this.results.resultBlock_ER_Incr = mean(this.results.blockIncrRew_ER);
            
            [this.results.hRandom_Ctrl_Incr this.results.pRandom_Ctrl_Incr] = ttest(this.results.randomIncrRew_Ctrl,0);
            this.results.resultRandom_Ctrl_Incr = mean(this.results.randomIncrRew_Ctrl);
            [this.results.hRandom_RT_Incr this.results.pRandom_RT_Incr] = ttest(this.results.randomIncrRew_RT,0);
            this.results.resultRandom_RT_Incr = mean(this.results.randomIncrRew_RT);
            [this.results.hRandom_ER_Incr this.results.pRandom_ER_Incr] = ttest(this.results.randomIncrRew_ER,0);
            this.results.resultRandom_ER_Incr = mean(this.results.randomIncrRew_ER);
            
            [this.results.hFull_Ctrl_Incr this.results.pFull_Ctrl_Incr] = ttest(this.results.fullIncrRew_Ctrl,0);
            this.results.resultFull_Ctrl_Incr = mean(this.results.fullIncrRew_Ctrl);
            [this.results.hFull_RT_Incr this.results.pFull_RT_Incr] = ttest(this.results.fullIncrRew_RT,0);
            this.results.resultFull_RT_Incr = mean(this.results.fullIncrRew_RT);
            [this.results.hFull_ER_Incr this.results.pFull_ER_Incr] = ttest(this.results.fullIncrRew_ER,0);
            this.results.resultFull_ER_Incr = mean(this.results.fullIncrRew_ER);
            
            % reward decrease tests
            
            [this.results.hBlock_Ctrl_Decr this.results.pBlock_Ctrl_Decr] = ttest(this.results.blockDecrRew_Ctrl,0);
            this.results.resultBlock_Ctrl_Decr = mean(this.results.blockDecrRew_Ctrl);        
            [this.results.hBlock_RT_Decr this.results.pBlock_RT_Decr] = ttest(this.results.blockDecrRew_RT,0);
            this.results.resultBlock_RT_Decr = mean(this.results.blockDecrRew_RT);
            [this.results.hBlock_ER_Decr this.results.pBlock_ER_Decr] = ttest(this.results.blockDecrRew_ER,0);
            this.results.resultBlock_ER_Decr = mean(this.results.blockDecrRew_ER);
            
            [this.results.hRandom_Ctrl_Decr this.results.pRandom_Ctrl_Decr] = ttest(this.results.randomDecrRew_Ctrl,0);
            this.results.resultRandom_Ctrl_Decr = mean(this.results.randomDecrRew_Ctrl);
            [this.results.hRandom_RT_Decr this.results.pRandom_RT_Decr] = ttest(this.results.randomDecrRew_RT,0);
            this.results.resultRandom_RT_Decr = mean(this.results.randomDecrRew_RT);
            [this.results.hRandom_ER_Decr this.results.pRandom_ER_Decr] = ttest(this.results.randomDecrRew_ER,0);
            this.results.resultRandom_ER_Decr = mean(this.results.randomDecrRew_ER);
            
            [this.results.hFull_Ctrl_Decr this.results.pFull_Ctrl_Decr] = ttest(this.results.fullDecrRew_Ctrl,0);
            this.results.resultFull_Ctrl_Decr = mean(this.results.fullDecrRew_Ctrl);
            [this.results.hFull_RT_Decr this.results.pFull_RT_Decr] = ttest(this.results.fullDecrRew_RT,0);
            this.results.resultFull_RT_Decr = mean(this.results.fullDecrRew_RT);
            [this.results.hFull_ER_Decr this.results.pFull_ER_Decr] = ttest(this.results.fullDecrRew_ER,0);
            this.results.resultFull_ER_Decr = mean(this.results.fullDecrRew_ER);
            
        end
        
        function dispResults(this)
            disp('++++++++++ DDMRewardSanity ++++++++++');
                 
            disp('Control intensity ~ reward increase');     
            if(this.results.resultBlock_Ctrl_Incr > 0 && this.results.pBlock_Ctrl_Incr < 0.05) 
                this.results.checkBlock_Ctrl_Incr = '[OK]';
            else
                this.results.checkBlock_Ctrl_Incr = '[##]'; 
            end
            
            if(this.results.resultRandom_Ctrl_Incr > 0 && this.results.pRandom_Ctrl_Incr < 0.05) 
                this.results.checkRandom_Ctrl_Incr = '[OK]';
            else
                this.results.checkRandom_Ctrl_Incr = '[##]'; 
            end
            
            if(this.results.resultFull_Ctrl_Incr > 0 && this.results.pFull_Ctrl_Incr < 0.05) 
                this.results.checkFull_Ctrl_Incr = '[OK]';
            else
                this.results.checkFull_Ctrl_Incr = '[##]'; 
            end
            disp(strcat(this.results.checkBlock_Ctrl_Incr,' blocked Reward: mControlDiff = ', num2str(this.results.resultBlock_Ctrl_Incr), ', h = ', num2str(this.results.hBlock_Ctrl_Incr),', p = ', num2str(this.results.pBlock_Ctrl_Incr)));
            disp(strcat(this.results.checkRandom_Ctrl_Incr,' random Reward: mControlDiff = ',  num2str(this.results.resultRandom_Ctrl_Incr), ', h = ', num2str(this.results.hRandom_Ctrl_Incr),', p = ', num2str(this.results.pRandom_Ctrl_Incr)));
            disp(strcat(this.results.checkFull_Ctrl_Incr,' blocked+random Reward: mControlDiff = ',  num2str(this.results.resultFull_Ctrl_Incr), ', h = ', num2str(this.results.hFull_Ctrl_Incr),', p = ', num2str(this.results.pFull_Ctrl_Incr)));
            
            disp('Control intensity ~ reward decrease');     
            if(this.results.resultBlock_Ctrl_Decr < 0 && this.results.pBlock_Ctrl_Decr < 0.05) 
                this.results.checkBlock_Ctrl_Decr = '[OK]';
            else
                this.results.checkBlock_Ctrl_Decr = '[##]'; 
            end
            
            if(this.results.resultRandom_Ctrl_Decr < 0 && this.results.pRandom_Ctrl_Decr < 0.05) 
                this.results.checkRandom_Ctrl_Decr = '[OK]';
            else
                this.results.checkRandom_Ctrl_Decr = '[##]'; 
            end
            
            if(this.results.resultFull_Ctrl_Decr < 0 && this.results.pFull_Ctrl_Decr < 0.05) 
                this.results.checkFull_Ctrl_Decr = '[OK]';
            else
                this.results.checkFull_Ctrl_Decr = '[##]'; 
            end
            disp(strcat(this.results.checkBlock_Ctrl_Decr,' blocked Reward: mControlDiff = ', num2str(this.results.resultBlock_Ctrl_Decr), ', h = ', num2str(this.results.hBlock_Ctrl_Decr),', p = ', num2str(this.results.pBlock_Ctrl_Decr)));
            disp(strcat(this.results.checkRandom_Ctrl_Decr,' random Reward: mControlDiff = ',  num2str(this.results.resultRandom_Ctrl_Decr), ', h = ', num2str(this.results.hRandom_Ctrl_Decr),', p = ', num2str(this.results.pRandom_Ctrl_Decr)));
            disp(strcat(this.results.checkFull_Ctrl_Decr,' blocked+random Reward: mControlDiff = ',  num2str(this.results.resultFull_Ctrl_Decr), ', h = ', num2str(this.results.hFull_Ctrl_Decr),', p = ', num2str(this.results.pFull_Ctrl_Decr)));
            
            disp(' ');
            disp('RT ~ reward increase');     
            if(this.results.resultBlock_RT_Incr < 0 && this.results.pBlock_RT_Incr < 0.05) 
                this.results.checkBlock_RT_Incr = '[OK]';
            else
                this.results.checkBlock_RT_Incr = '[##]'; 
            end
            
            if(this.results.resultRandom_RT_Incr < 0 && this.results.pRandom_RT_Incr < 0.05) 
                this.results.checkRandom_RT_Incr = '[OK]';
            else
                this.results.checkRandom_RT_Incr = '[##]'; 
            end
            
            if(this.results.resultFull_RT_Incr < 0 && this.results.pFull_RT_Incr < 0.05) 
                this.results.checkFull_RT_Incr = '[OK]';
            else
                this.results.checkFull_RT_Incr = '[##]'; 
            end
            disp(strcat(this.results.checkBlock_RT_Incr,' blocked Reward: mRTDiff = ', num2str(this.results.resultBlock_RT_Incr), ', h = ', num2str(this.results.hBlock_RT_Incr),', p = ', num2str(this.results.pBlock_RT_Incr)));
            disp(strcat(this.results.checkRandom_RT_Incr,' random Reward: mRTDiff = ',  num2str(this.results.resultRandom_RT_Incr), ', h = ', num2str(this.results.hRandom_RT_Incr),', p = ', num2str(this.results.pRandom_RT_Incr)));
            disp(strcat(this.results.checkFull_RT_Incr,' blocked+random Reward: mRTDiff = ',  num2str(this.results.resultFull_RT_Incr), ', h = ', num2str(this.results.hFull_RT_Incr),', p = ', num2str(this.results.pFull_RT_Incr)));
            
            disp('RT ~ reward decrease');     
            if(this.results.resultBlock_RT_Decr > 0 && this.results.pBlock_RT_Decr < 0.05) 
                this.results.checkBlock_RT_Decr = '[OK]';
            else
                this.results.checkBlock_RT_Decr = '[##]'; 
            end
            
            if(this.results.resultRandom_RT_Decr > 0 && this.results.pRandom_RT_Decr < 0.05) 
                this.results.checkRandom_RT_Decr = '[OK]';
            else
                this.results.checkRandom_RT_Decr = '[##]'; 
            end
            
            if(this.results.resultFull_RT_Decr > 0 && this.results.pFull_RT_Decr < 0.05) 
                this.results.checkFull_RT_Decr = '[OK]';
            else
                this.results.checkFull_RT_Decr = '[##]'; 
            end
            disp(strcat(this.results.checkBlock_RT_Decr,' blocked Reward: mRTDiff = ', num2str(this.results.resultBlock_RT_Decr), ', h = ', num2str(this.results.hBlock_RT_Decr),', p = ', num2str(this.results.pBlock_RT_Decr)));
            disp(strcat(this.results.checkRandom_RT_Decr,' random Reward: mRTDiff = ',  num2str(this.results.resultRandom_RT_Decr), ', h = ', num2str(this.results.hRandom_RT_Decr),', p = ', num2str(this.results.pRandom_RT_Decr)));
            disp(strcat(this.results.checkFull_RT_Decr,' blocked+random Reward: mRTDiff = ',  num2str(this.results.resultFull_RT_Decr), ', h = ', num2str(this.results.hFull_RT_Decr),', p = ', num2str(this.results.pFull_RT_Decr)));
            
            
            disp(' ');
            disp('ER ~ reward increase');     
            if(this.results.resultBlock_ER_Incr < 0 && this.results.pBlock_ER_Incr < 0.05) 
                this.results.checkBlock_ER_Incr = '[OK]';
            else
                this.results.checkBlock_ER_Incr = '[##]'; 
            end
            
            if(this.results.resultRandom_ER_Incr < 0 && this.results.pRandom_ER_Incr < 0.05) 
                this.results.checkRandom_ER_Incr = '[OK]';
            else
                this.results.checkRandom_ER_Incr = '[##]'; 
            end
            
            if(this.results.resultFull_ER_Incr < 0 && this.results.pFull_ER_Incr < 0.05) 
                this.results.checkFull_ER_Incr = '[OK]';
            else
                this.results.checkFull_ER_Incr = '[##]'; 
            end
            disp(strcat(this.results.checkBlock_ER_Incr,' blocked Reward: mERDiff = ', num2str(this.results.resultBlock_ER_Incr), ', h = ', num2str(this.results.hBlock_ER_Incr),', p = ', num2str(this.results.pBlock_ER_Incr)));
            disp(strcat(this.results.checkRandom_ER_Incr,' random Reward: mERDiff = ',  num2str(this.results.resultRandom_ER_Incr), ', h = ', num2str(this.results.hRandom_ER_Incr),', p = ', num2str(this.results.pRandom_ER_Incr)));
            disp(strcat(this.results.checkFull_ER_Incr,' blocked+random Reward: mERDiff = ',  num2str(this.results.resultFull_ER_Incr), ', h = ', num2str(this.results.hFull_ER_Incr),', p = ', num2str(this.results.pFull_ER_Incr)));
            
            disp('ER ~ reward decrease');     
            if(this.results.resultBlock_ER_Decr > 0 && this.results.pBlock_ER_Decr < 0.05) 
                this.results.checkBlock_ER_Decr = '[OK]';
            else
                this.results.checkBlock_ER_Decr = '[##]'; 
            end
            
            if(this.results.resultRandom_ER_Decr > 0 && this.results.pRandom_ER_Decr < 0.05) 
                this.results.checkRandom_ER_Decr = '[OK]';
            else
                this.results.checkRandom_ER_Decr = '[##]'; 
            end
            
            if(this.results.resultFull_ER_Decr > 0 && this.results.pFull_ER_Decr < 0.05) 
                this.results.checkFull_ER_Decr = '[OK]';
            else
                this.results.checkFull_ER_Decr = '[##]'; 
            end
            disp(strcat(this.results.checkBlock_ER_Decr,' blocked Reward: mERDiff = ', num2str(this.results.resultBlock_ER_Decr), ', h = ', num2str(this.results.hBlock_ER_Decr),', p = ', num2str(this.results.pBlock_ER_Decr)));
            disp(strcat(this.results.checkRandom_ER_Decr,' random Reward: mERDiff = ',  num2str(this.results.resultRandom_ER_Decr), ', h = ', num2str(this.results.hRandom_ER_Decr),', p = ', num2str(this.results.pRandom_ER_Decr)));
            disp(strcat(this.results.checkFull_ER_Decr,' blocked+random Reward: mERDiff = ',  num2str(this.results.resultFull_ER_Decr), ', h = ', num2str(this.results.hFull_ER_Decr),', p = ', num2str(this.results.pFull_ER_Decr)));
            
        end
        
        function plotSummary(this) 
            
            exampleSubj = 1;
            sampleTrials = 1:this.nTrials;%(this.nTrials/2);
            f1 = figure(1);
            set(f1, 'Position', [0 0 600 500])
            subplot(3,1,1);
            this.plotEVC(exampleSubj, 'expected', sampleTrials);
            subplot(3,1,2);
            this.plotReward(exampleSubj, sampleTrials);
            subplot(3,1,3);
            this.plotCtrlIntensity(exampleSubj, sampleTrials);
            
            f2 = figure(2);
            set(f2, 'Position', [600 0 800 300])
            %set(f2,'defaulttextinterpreter','latex');
            this.plotRewardResults();
        end
               
        function plotRewardResults(this)
            subplot(1,3,1);
            bar(1,[this.results.resultFull_Ctrl_Incr],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semIncr = std(this.results.fullIncrRew_Ctrl)/sqrt(length(this.results.fullIncrRew_Ctrl));
            semDecr = std(this.results.fullDecrRew_Ctrl)/sqrt(length(this.results.fullDecrRew_Ctrl));
            errorbar(1,[this.results.resultFull_Ctrl_Incr],semIncr,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.resultFull_Ctrl_Decr],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.resultFull_Ctrl_Decr],semDecr,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.resultFull_Ctrl_Incr, this.results.resultFull_Ctrl_Decr];
            range = max(max(abs(bardata)));
            ylim([-range-0.2*range range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'Incr', 'Decr'},'fontsize',this.plotParams.axisFontSize);
            xlabel('\Delta Reward (trial n)');
            ylabel('\Delta Intensity','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('Intensity Adjustment (trial n+1)','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            subplot(1,3,2);
            bar(1,[this.results.resultFull_RT_Incr],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semIncr = std(this.results.fullIncrRew_RT)/sqrt(length(this.results.fullIncrRew_RT));
            semDecr = std(this.results.fullDecrRew_RT)/sqrt(length(this.results.fullDecrRew_RT));
            errorbar(1,[this.results.resultFull_RT_Incr],semIncr,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.resultFull_RT_Decr],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.resultFull_RT_Decr],semDecr,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.resultFull_RT_Incr, this.results.resultFull_RT_Decr];
            range = max(max(abs(bardata)));
            ylim([-range-0.2*range range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'Incr', 'Decr'},'fontsize',this.plotParams.axisFontSize);
            xlabel('\Delta Reward (trial n)');
            ylabel('\Delta RT (ms)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('RT Adjustment (trial n+1)','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
            
            subplot(1,3,3);
            bar(1,[this.results.resultFull_ER_Incr],'FaceColor', this.plotParams.defaultBarColor);
            hold on;
            semIncr = std(this.results.fullIncrRew_ER)/sqrt(length(this.results.fullIncrRew_ER));
            semDecr = std(this.results.fullDecrRew_ER)/sqrt(length(this.results.fullDecrRew_ER));
            errorbar(1,[this.results.resultFull_ER_Incr],semIncr,'Color',this.plotParams.defaultColor);
            bar(2,[this.results.resultFull_ER_Decr],'FaceColor', this.plotParams.defaultBarColor);
            errorbar(2,[this.results.resultFull_ER_Decr],semDecr,'Color',this.plotParams.defaultColor);
            xlim([0 3]);
            bardata = [this.results.resultFull_ER_Incr, this.results.resultFull_ER_Decr];
            range = max(max(abs(bardata)));
            ylim([-range-0.2*range range+0.2*range]);
            set(gca,'Xtick',1:2)
            set(gca,'XTickLabel',{'Incr', 'Decr'},'fontsize',this.plotParams.axisFontSize);
            xlabel('\Delta Reward (trial n)');
            ylabel('\Delta ER (%)','FontSize',this.plotParams.axisFontSize); % y-axis label
            title('ER Adjustment (trial n+1)','FontSize',this.plotParams.titleFontSize-1,'FontWeight','bold'); % y-axis label
             
        end
    end
    
    methods (Access = protected)
        
        function initTaskEnv(this)
            this.taskEnv = EVC.TaskEnv(this.defaultTrial, this.nTrials);
            
            % trial rewards & conditions
            rewards = 10:10:100;
            conditions = 1:length(rewards);
            
            % check if there are enough trials for first block
            nBlockTrials = this.nTrials/2/(2*length(rewards));
            if(mod(nBlockTrials,1) ~= 0)
               warning('DDMRewardSanity: cannot evenly distribute reward blocks for given number of trials'); 
            end
            
            % define 1st half of experiment: increasing & decreasing reward blocks
            blockTrials = repmat([conditions sort(conditions,'descend')],this.nTrials/2/(2*length(rewards)),1);
            blockTrials = blockTrials(1:end);
            
            % build sequence
            for trialIdx = 1:this.nTrials
                this.taskEnv.Sequence(trialIdx).outcomeValues = zeros(1,length(this.taskEnv.Sequence(trialIdx).outcomeValues));
               if trialIdx <= floor(this.nTrials/2)
                   this.taskEnv.Sequence(trialIdx).outcomeValues(1) = rewards(blockTrials(trialIdx));
                   this.taskEnv.Sequence(trialIdx).conditions = blockTrials(trialIdx);
               else  
                   sample = randsample(1:length(rewards),1);
                   this.taskEnv.Sequence(trialIdx).outcomeValues(1) = rewards(sample);
                   this.taskEnv.Sequence(trialIdx).conditions = conditions(sample);
               end
            end
            
        end
        
    end
    
end

