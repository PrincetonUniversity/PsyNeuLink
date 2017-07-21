classdef EVCPlot < handle
    
   properties

   end
   
   methods (Static)
       
       function Progress(fig, TaskEnv)
           
           % get current trial & number of trials
           currentTrial = TaskEnv.CurrTrial;
           nTrials = length(TaskEnv.Sequence);
           
           % begin plot
           
           if(~exist('fig', 'var'))
              fig = 100;
           end
           
           figure(fig);
           plot(1:nTrials,repmat(10,1,nTrials),'w',1:nTrials,repmat(0,1,nTrials),'w')
           rectangle('Position',[0,0,currentTrial,10], 'FaceColor','r')
           text(nTrials/2,5,num2str(currentTrial,'%.2f'),'FontSize',18);
           xlabel('Trial');
           
       end
      
       function CtrlIRange(varargin)
          
          fig = varargin{1};
          CtrlIs = varargin{2};
          EVCs = varargin{3};
          Rewards = varargin{4};
          CCosts = varargin{5};
          
          if(length(varargin) >= 6)
               maxsub = 4;
          else
               maxsub = 3;
          end
           
          if(~exist('fig', 'var'))
             fig = 1;
          end
           
          figure(fig);
          %figure('name','Control Intensity Range');
          %%%%%%%%%%%%%%%%%%%%
          % show EVC (~= utility)
          subplot(maxsub,1,1)
          imagesc(max(EVCs));
          hold on;
          text(1-0.1,1,num2str(max(EVCs),'%.2f'),'FontSize',18);
          set(gca,'TickDir','out');
          set(gca,'Box','off');
          set(gca,'XTick',1:3)
          set(gca,'YTick',1)
          ylabel('EVC');
          set(gca,'FontSize',16);
          set(gca,'XTickLabel',{'Task 1'}); %set(gca,'XTickLabel',{'Task 1','Task 2','Task 3'}) % old one used for multiple ctrl tasks/trial types

          %%%%%%%%%%%%%%%%%%%%
          % show Control Intensity (~= cost)
          subplot(maxsub,1,2)
          imagesc(CtrlIs(find(EVCs==max(EVCs))));
          text(1,1,num2str(CtrlIs(find(EVCs==max(EVCs))),'%.2f'),'FontSize',18);
          set(gca,'TickDir','out');
          set(gca,'Box','off');
          set(gca,'XTick',1:3)
          set(gca,'YTick',1)
          ylabel('Control Intensity');
          set(gca,'FontSize',16);
          set(gca,'XTickLabel',{'Task 1'}); %set(gca,'XTickLabel',{'Task 1','Task 2','Task 3'}) % old one used for multiple ctrl tasks/trial types

          %%%%%%%%%%%%%%%%%%%%
          % show costs, rewards, and EVC
          subplot(maxsub,1,3*1-(1-1))
          plot(CtrlIs,abs(CCosts),'r',...
               CtrlIs,Rewards,'g',...               
               CtrlIs,EVCs,'b','LineWidth',1.0)
          xlabel('Control Intensity (red is cost, green is reward, blue is EVC)');
          set(gca,'TickDir','out');
          set(gca,'Box','off');
          % ylim( [-.25 1.25] ); % should keep this the same over loop iterations
          
          % show progress
           if(length(varargin) >= 6)
               
               TaskEnv = varargin{6};
               
               % get current trial & number of trials
               currentTrial = TaskEnv.CurrTrialIdx;
               nTrials = length(TaskEnv.Sequence);
               
               subplot(maxsub,1,4)
               plot(1:nTrials,repmat(10,1,nTrials),'w',1:nTrials,repmat(0,1,nTrials),'w')
               rectangle('Position',[0,0,currentTrial,10], 'FaceColor','r')
               text(nTrials/2,5,num2str(currentTrial,'%.2f'),'FontSize',18);
               xlabel('Trial Progress');
               
           end
          
          hold off;
       end
       
       function TrialSumLamingDDM(fig, Log)
           
           % collect information
           EVCs = Log.EVCs(:,1);          % use expected
           ER = Log.ERs(:,2);             % use actual        

           CtrlIs = Log.CtrlIntensities;
           CtrlSigs = Log.SignalIdxs;
           NumCtrlSigs = length(unique(CtrlSigs));
           
           expectedSeq = Log.ExpectedStateParam(:);
           actualSeq = Log.ActualStateParam(:);
           ctrlSeq = Log.ControlParam(:);
           
           % begin plot
           if(~exist('fig', 'var'))
              fig = 2;
           end
           
           % plot EVC curve
           figure(fig);
           subplot(4,1,1);
           plot( EVCs', 'b' );
           title( 'EVC''s' );

           % plot biases
           subplot(4,1,2);
           plot( CtrlIs', 'b'  );  
           plot( actualSeq', 'r' );
           hold on; plot( expectedSeq', 'g' );
           hold off;
           title( 'actual state param (red) and expected state param (green) ');

           % plot error rate (for actual states)
           subplot(4,1,3);
           plot( ER', 'r' );
           title( 'error rates (red)' );

%            % plot threshold rates (for expected states)
%            subplot(5,1,4);
%            plot( ctrlSeq', 'b' );
%            title( 'control param (blue)' ); 
           
           % plot ctrl intensity
           subplot(4,1,4);
           plot( CtrlIs', 'black'); 
           if(NumCtrlSigs > 1)
               hold on;
               colors = EVC.EVCPlot.getNColors(NumCtrlSigs);
               a = 1;
               for i = 2:length(CtrlIs)
                   if(CtrlSigs(i) ~= CtrlSigs(i-1))
                       plot(a:(i-1),CtrlIs(a:(i-1))', 'Color', colors(CtrlSigs(i-1),:));
                       a = i;
                   end
               end
               plot( a:i, CtrlIs(a:i)', 'Color', colors(CtrlSigs(i),:));
               hold off;
               title('control intensity (different colors for different control signals)');
           else
               title('control intensity');    
           end
       end
       
       function TrialSumGrattonDDM(fig, Log)
           
           % collect information
           EVCs = Log.EVCs(:,1);          % use expected
           ER = Log.ERs(:,2);             % use actual
           RT = Log.RTs(:,2);             % use actual
           
           CtrlIs = Log.CtrlIntensities;
           
           expectedSeq = Log.ExpectedStateParam(:);
           actualSeq = Log.ActualStateParam(:);
           
           % begin plot
           if(~exist('fig', 'var'))
              fig = 2;
           end
           
           % plot EVC curve
           figure(fig);
           subplot(5,1,1);
           plot( EVCs', 'b' );
           title( 'EVC''s' );

           % plot biases
           subplot(5,1,2);
           plot( CtrlIs', 'b'  );  
           plot( actualSeq', 'r' );
           hold on; plot( expectedSeq', 'g' );
           hold off;
           title( 'actual state param (red) and expected state param (green) ');

           % plot error rate (for actual states)
           subplot(5,1,3);
           plot( ER', 'r' );
           title( 'error rates (red)' );

           % plot threshold rates (for expected states)
           subplot(5,1,4);
           plot( RT', 'b' );
           title( 'RT (blue)' ); 
           
           % plot ctrl intensity
           subplot(5,1,5);
           plot( CtrlIs', 'b');
           title('control intensity (blue)');
       end
       
       function LamingDDM(fig, Log)
           
           % collect information
           expectedER = Log.ERs(:,1);
           actualER = Log.ERs(:,2);
           actualRT = Log.RTs(:,2);
           actualSeq = Log.ActualStateParam(:);
           
           % get boolean bias perturbation array
           biasPerturbSeq = zeros(length(actualSeq), 1);
           for i = 2:length(actualSeq)
               if(actualSeq(i) ~= actualSeq(i-1))
                  biasPerturbSeq(i) = 1; 
               end
           end

           % calculate Laming effect
           currPerturbRT = biasPerturbSeq(2:(length(biasPerturbSeq)));                          % -> is there a bias perturbation on the next trial? (if so, the next trial is not interesting to record post-error RTs)
           prevERdiff = actualER(1:(length(actualER)-1))-expectedER(1:(length(expectedER)-1));  % difference between actual and expected error rate across all trials except last one (no post-error slowing for last trial)
           prevRT = actualRT(1:(length(actualRT)-1));                                           % RT at previous trial (at potential error trial)
           currRT = actualRT(2:length(actualRT));                                               % RT at current trial (after potential error trial)
           diffRT = currRT - prevRT;                                                            % post-error slowing
           prevERdiff = prevERdiff(find(currPerturbRT == 0));                                   % consider valid errors trials only for trials in which actual bias hasn't changed
           diffRT = diffRT(find(currPerturbRT == 0));                                           % consider valid trials for post-error rt differences only for trials in which NO bias perturbation happened (otherwise post RT effect would be due to bias mismatch, not EVC adjustment)

           % begin plot

           if(~exist('fig', 'var'))
              fig = 3;
           end
           figure(fig);
           
           % scatter plot, 
           % x-axis: represents the error (< 0 no error in
           % previous trial, > 0 error in previous trial)
           % y-axis: post-error slowing as RT difference between error
           % trial and post-error trial
           
           subplot(2,1,1);
           scatter(prevERdiff, diffRT);

           ylim([min(-0.1,min(diffRT)-0.1) max(0.1,max(diffRT)+0.1)]);
           xlim([min(-max(prevERdiff), min(prevERdiff)) max(max(prevERdiff), -min(prevERdiff))])
           hold on;
           y=get(gca,'ylim');
           plot([0 0],y,'-');
    
           xlabel('error rate diff (actual ER - expected ER)');
           ylabel('current RT - previous RT');
           title('Laming effect');

           % mean split plot,
           % define no-error trials as trials in which the actual error is
           % smaller than the expected error
           % define error trials as trials in which the actual error is
           % larger than the expected error
           % plot post-trial RT's for both no-error and error conditions
           
           subplot(2,1,2);
           precCorr_RT   = diffRT(find(prevERdiff < 0));
           precInc_RT  = diffRT(find(prevERdiff > 0));
           
           if(isempty(precCorr_RT))
              warning('No data for non-error trials. Mean of zero assumed.'); 
              precCorr_RT = 0;
           end
           if(isempty(precInc_RT))
              warning('No data for error trials. Mean of zero assumed.'); 
              precInc_RT = 0;
           end
           
           bardata = [mean(precCorr_RT) mean(precInc_RT)];
           h = bar([mean(precCorr_RT) mean(precInc_RT)], 0.2);
           ylim([min(-0.01,min(bardata)*1.3) max(0.01,max(bardata)*1.3)]);
           hChildren = get(h, 'Children');
           hColors = [[1.0, 0.0, 0.0]; [0.0, 0.0, 1.0]];
           index = [1 2];
           set(hChildren, 'CData', index);
           colormap(hColors);
           set(gca,'xticklabel',{'after correct' 'after incorrect'})
           ylabel('mean RT slowing (current - previous)');
           title('');
       end
       
       function GrattonDDM(fig, Log)
           
            % get RT
            actualER = Log.ERs(:,2);    % use actual
            actualRT = Log.RTs(:,2);    % use actual

            % get trial type
            for i = 1:(length(Log.Trials))
                if(strcmp(Log.Trials(i).type,'incongruent'))
                    trialType(i) = 1;
                else
                    trialType(i) = 0;
                end
            end

            % calc RT conditions
            postRT = actualRT(2:length(actualRT));
            postER = actualER(2:length(actualER));
            postType = trialType(2:length(trialType));
            precType = trialType(1:(length(trialType)-1));

            precCongRT = postRT(find(precType == 0));
            precIncRT = postRT(find(precType == 1));
            
            incIncRT = postRT(find(precType == 1 & postType == 1));
            incConRT = postRT(find(precType == 1 & postType == 0));
            conIncRT = postRT(find(precType == 0 & postType == 1));
            conConRT = postRT(find(precType == 0 & postType == 0));
            
            incIncER = postER(find(precType == 1 & postType == 1));
            incConER = postER(find(precType == 1 & postType == 0));
            conIncER = postER(find(precType == 0 & postType == 1));
            conConER = postER(find(precType == 0 & postType == 0));

            % begin plot

            if(~exist('fig', 'var'))
               fig = 3;
            end           
            figure(fig);
            
            % basic Gratton
            subplot(3,1,1);
            
            bardata = [mean(precCongRT) mean(precIncRT)];
            h = bar([mean(precCongRT) mean(precIncRT)], 0.2);
            ylim([max(0,min(bardata)*0.9) max(0.01,min(bardata) + (max(bardata)-min(bardata))*1.5)]);
            hChildren = get(h, 'Children');
            hColors = [[1.0, 0.0, 0.0]; [0.0, 0.0, 1.0]];
            index = [1 2];
            set(hChildren, 'CData', index);
            colormap(hColors);
            set(gca,'xticklabel',{'after congruent' 'after incongruent'})
            ylabel('mean RT');
            title('');
            
            % full Gratton RT
            subplot(3,1,2)
            plotdata = [mean(incIncRT) mean(incConRT) mean(conIncRT) mean(conConRT)];
            
            plot([1 2],[mean(conIncRT) mean(incIncRT)], [1 2], [mean(conConRT) mean(incConRT)], '--');
            xlim([0.5 2.5]);
            ylim([min(plotdata)-0.002 max(plotdata)+0.002]);
            
            text(1,mean(conIncRT),'C-I','FontSize',14);
            text(2,mean(incIncRT),'I-I','FontSize',14);
            text(1,mean(conConRT),'C-C','FontSize',14);
            text(2,mean(incConRT),'I-C','FontSize',14);
            
            ylabel('mean RT');
            title( 'Gratton Reaction Time' ); 
            
            % full Gratton ER
            subplot(3,1,3)
            plotdata = [mean(incIncER) mean(incConER) mean(conIncER) mean(conConER)];
            
            plot([1 2],[mean(conIncER) mean(incIncER)], [1 2], [mean(conConER) mean(incConER)], '--');
            xlim([0.5 2.5]);
            ylim([min(plotdata)-0.02 max(plotdata)+0.02]);
            
            text(1,mean(conIncER)+0.01,'C-I','FontSize',14);
            text(2,mean(incIncER)+0.01,'I-I','FontSize',14);
            text(1,mean(conConER)-0.01,'C-C','FontSize',14);
            text(2,mean(incConER)-0.01,'I-C','FontSize',14);
            
            ylabel('P(error)');
            title( 'Gratton Error Rate' ); 
              
       end
       
       % OOP V5
       function StateVsEVC(EVCModel, params, filename)
          
          steps = abs((params.statePerturbRange(1)-params.statePerturbRange(length(params.statePerturbRange)))/(length(params.statePerturbRange)-1));
          stateIvals = params.stateLimit(1):steps:params.stateLimit(2);
          NIvals = length(stateIvals);
          
          maxEVC = 0;
          stateVal = 0;
          
          % for each possible state
          for stateIdx = 1:NIvals
              
              % calculate stimuli properties
              newSalience(1) = stateIvals(stateIdx);
              newSalience(2) = 1 - newSalience(1);
              EVCModel.State.TaskEnv.currentTrial().StimSalience = [newSalience(1) newSalience(2)];
              
              stateVal(stateIdx,1) = newSalience(1);
              
              EVCModel.setActualState();
              EVCModel.useActualState = 1;
              [optIntensities optEVC] = EVCModel.getOptSignals();
              
              signals(stateIdx,:) = transpose(optIntensities);
              maxEVC(stateIdx) = optEVC;
              
              stateVal(stateIdx,2) = EVCModel.getStateParam(1);
          
          end
          
          disp('min/max intensity for optimal EVC:')
          disp([sum(signals(1,:)) sum(signals(NIvals,:))]);
          disp(sum(signals(1,:))-sum(signals(NIvals,:)));
          
          % filename = 'logfiles/stateVsEVC.txt'
          
          if(~isempty(filename))
            % open file
            datafilepointer = fopen(filename,'wt'); % open ASCII file for writing

            % write header
            dataPoints = length(maxEVC);
            numSignals = size(signals,2);
            
            fprintf(datafilepointer,'%s ', 'targetSalience');
            fprintf(datafilepointer,'%s ', 'stateVal');
            for i = 1:numSignals
                fprintf(datafilepointer,'%s ', strcat('Signal_', int2str(i)));
            end
            fprintf(datafilepointer,'%s ', 'maxEVC');
            fprintf(datafilepointer,'\n');
            
            % write data
            
            for i = 1:dataPoints
                fprintf(datafilepointer,'%s ',  num2str(stateVal(i,1)));
                fprintf(datafilepointer,'%s ',  num2str(stateVal(i,2)));
                for j = 1:numSignals
                    fprintf(datafilepointer,'%s ',  num2str(signals(i,j)));
                end
                fprintf(datafilepointer,'%s ',  num2str(maxEVC(i)));
                fprintf(datafilepointer,'\n');
            end
            
            % close file
            fclose(datafilepointer);
            
          end
          
       end
       
       function IntensityVsEVC(EVCModel, params, filename)
           
           % use medium difficulty
           newSalience(1) = params.stateLimit(1)+(params.stateLimit(2)-params.stateLimit(1))/2;
           newSalience(2) = 1 - newSalience(1);
           EVCModel.State.TaskEnv.currentTrial().StimSalience = [newSalience(1) newSalience(2)];
           EVCModel.setActualState();
           EVCModel.useActualState = 1;
              
           [~, ~, signalMap, EVCMap] = EVCModel.getOptSignals();
          
           if(~isempty(filename))
            % open file
            datafilepointer = fopen(filename,'wt'); % open ASCII file for writing

            % write header
            [numSignals dataPoints] = size(signalMap);
            
            for i = 1:numSignals
                fprintf(datafilepointer,'%s ', strcat('Intensity_', int2str(i)));
            end
            fprintf(datafilepointer,'%s ', 'EVC');
            fprintf(datafilepointer,'\n');
            
            % write data
            
            for i = 1:dataPoints
                for j = 1:numSignals
                    fprintf(datafilepointer,'%s ',  num2str(signalMap(j,i)));
                end
                fprintf(datafilepointer,'%s ',  num2str(EVCMap(i)));
                fprintf(datafilepointer,'\n');
            end
            
            % close file
            fclose(datafilepointer);
            
          end
       end
       
       %% various
       
       function colors = getNColors(N)
          hueSteps = 1/N;
          for i = 0:(N-1)
             colors(i+1,:) = hsv2rgb([(0+i*hueSteps) 1.0 1.0]);
          end
       end
       
   end
end