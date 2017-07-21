function [aRT, aER, aRT_plus, aRT_minus, aCDF_T, aCDF_Y, aCDF_Y_plus, aCDF_Y_minus, ...
    simMeanRT, simMeanER,simMeanRT_plus,simMeanRT_minus, simCDF_T, simCDF_Y, simCDF_Y_plus, simCDF_Y_minus] = ...
    MSDDM_wrapper(a,s,varthresh,deadlines,thresh,x0,x0dist,runSimulations,doPlots)
% Input:
% a = vector of drift rates at each stage
% s = vector of diffusion rates at each stage
% deadlines = vector of times when stages start. Firt entry should be 0.
% thresh = vector of thresholds to test (IMPORTANT: these are not thresholds for different *stages* - currently assuming uniform threshold across stages)
% x0 = support of initial condition. Equals the initial condition in the deterministic case
% x0dist = density of x0. Equals 1 in the deterministic case
% runSimulations = binary for whether to perform monte carlo simulations in addition to generating analytic solution
% doPlots = binary for whether to generate some relevant plots
% Output:
% aRT, aER = vectors of analytic expected decision time and error rate, one for each threshold being tested [using multi_stage_ddm_metrics]
% aCDF_T, aCDF_Y = cell array of analytic CDFs (..._T = RT range, ..._Y = cumul prob), one for each threshold being tested [using multistage_ddm_fpt_dist]
% simMeanRT, simMeanER, simCDF_T, simCDF_Y = same as above, but using MC simulations to generate each value estimate

import EVC.MSDDM.*;

if ~(exist('a','var') && ~isempty(a))
    % Vector of drift rates (i.e., a0, a1, a2....)
    a=[0.1 0.2 0.05 0.3];
end

if ~(exist('s','var') && ~isempty(s))
    % Vector of diffusion rates (noise coefficients)
    s=[1 1.5 1.25 2];
end

if ~(exist('varthresh','var') && ~isempty(varthresh))
    % Vector of thresholds (IF varying by stage) - set to NaN if using uniform thresh
   varthresh =  nan;
end

if ~(exist('deadlines','var') && ~isempty(deadlines))
    % Times (secs) at which  each stage starts (i.e., t0, t1, t2....)
    deadlines=[0 1 2 3];
end

if ~(exist('thresh','var') && ~isempty(thresh))
    % Thresholds to test (can enter vector or individual point)
    thresh=2.0;
    % thresh=0.5:0.25:4;
end

if ~(exist('x0','var') && ~isempty(x0))
    % Support of initial condition and its density (i.e., can enter distribution or individual point)
    x0 = 0;      x0dist = 1;
end

if ~isnan(varthresh)
    % This just sets up a vector length 1 (actual value is unimportant)
    thresh = 1.0;
end

if ~(exist('runSimulations','var') && ~isempty(runSimulations))
    % Run MC simulations?
    runSimulations = 0;
end
% No of Monte Carlo runs
% realizations=10000;
realizations=1000;
% Discretization step for simulating DDM
step=0.005;
% If not running simulations, this arbitrarily defines upper limit of CDF
% (otherwise relies on max of simulated RT):
tfinal_fixed = 20.0;


if ~(exist('doPlots','var') && ~isempty(doPlots))
    % Generate plots?
    doPlots = 1;
end


% Initializing variables
RT=nan(1,realizations);
ER=nan(1,realizations);
simMeanRT=nan(size(thresh));
simMeanER=nan(size(thresh));
simMeanRT_plus=nan(size(thresh));
simMeanRT_minus=nan(size(thresh));
aRT=nan(size(thresh));
aER=nan(size(thresh));
aRT_plus=nan(size(thresh));
aRT_minus=nan(size(thresh));

if runSimulations
    % Resetting random number generator
    randSeed=sum(100*clock);
    try
        RandStream.setDefaultStream(RandStream('mt19937ar','Seed',randSeed)); %reset the random number generator
    catch
        RandStream.setGlobalStream(RandStream('mt19937ar','Seed',randSeed));
    end
end


%% Generate analytic and/or simulated results for the multistage DDM
for jj=1:length(thresh)
    
    if isnan(varthresh)
        % Using uniform threshold determined by thresh run
        threshold=thresh(jj)*ones(size(a));
    else
        % Using stage-varying threshold (IGNORING values in thresh)
        threshold=varthresh;
    end
    
    RT=nan(1,realizations);
    ER=nan(1,realizations);
    
    if runSimulations

        % Simulate the multistage DDM (could be done more efficiently!)
        for N=1:realizations
            
            t=0;
            x(1)=x0;
            l=1;
            stop=0;
            
            while stop==0
                
                stage=find(deadlines<=t,1,'last');
                try
                    x(l+1)= x(l) +a(stage)*step + s(stage)*randn*sqrt(step);
                catch me
                   keyboard; 
                end
                t=t+step;
                l=l+1;
                
                if (x(l)>=threshold(stage) || x(l)<=-threshold(stage))
                    stop=1;
                    RT(N)=t;
                    ER(N)=(x(l)<=-threshold(stage));
                end
            end
        end
        
        % Mean decision time and error rate from Monte Carlo simulation
        simMeanRT(jj)=mean(RT);
        simMeanER(jj)=mean(ER);
        
        % Simulated CDF for all choices
        [hFreq{jj},hTimes{jj}] = hist(RT,1000);
        tfinal(jj)= hTimes{jj}(end);
        simCDF_T{jj} = hTimes{jj};  simCDF_Y{jj} = cumsum(hFreq{jj})/realizations;

        % Simulated mean DT and CDFs for pos/neg threshold crossings separately
        simRTplus=RT(ER==0);
        simRTminus=RT(ER==1);        
        simMeanRT_plus(jj)=mean(simRTplus);
        simMeanRT_minus(jj)=mean(simRTminus);
        [hFreq_plus{jj},hTimes_plus{jj}] = hist(simRTplus,1000);
        tfinal_plus(jj)= hTimes_plus{jj}(end);
        simCDF_T_plus{jj} = hTimes_plus{jj};  simCDF_Y_plus{jj} = cumsum(hFreq_plus{jj})/realizations;
        [hFreq_minus{jj},hTimes_minus{jj}] = hist(simRTminus,1000);
        tfinal_minus(jj)= hTimes_minus{jj}(end);
        simCDF_T_minus{jj} = hTimes_minus{jj};  simCDF_Y_minus{jj} = cumsum(hFreq_minus{jj})/realizations;        
    else   
        % Setting all of these to empty in case not running simulations
        simMeanRT(jj)=nan; simMeanER(jj)=nan;
        simMeanRT_plus(jj)=nan; simMeanRT_minus(jj)=nan;
        simCDF_T{jj} = [];  simCDF_Y{jj} = [];
        simCDF_T_plus{jj} = [];  simCDF_Y_plus{jj} = [];
        simCDF_T_minus{jj} = [];  simCDF_Y_minus{jj} = [];
        tfinal(jj) = tfinal_fixed; tfinal_plus(jj) = tfinal_fixed; tfinal_minus(jj) = tfinal_fixed;
    end
    
    % Analytic decision time and error rate
    [aRT(jj) aER(jj) aRT_plus(jj) aRT_minus(jj)]= multi_stage_ddm_metrics(a ,s, deadlines, threshold, x0, x0dist);

    % Analytic CDF (T = RT range, Y = cumul prob)
    [aCDF_T{jj} aCDF_Y{jj} aCDF_Y_plus{jj} aCDF_Y_minus{jj}]=multistage_ddm_fpt_dist(a,s,threshold,x0,x0dist,deadlines,tfinal(jj));
end



%% Plotting

if doPlots
  
    % CDF plot - LEFT: all trials, RIGHT: conditional on threshold-crossing
        % (if multiple uniform thresholds, only plotting the CDFs for the final threshold)
        figure
        subplot(1,2,1);
        if runSimulations
            plot(simCDF_T{jj}, simCDF_Y{jj},'k','linewidth',2);
        end
        hold on
        plot(aCDF_T{jj},aCDF_Y{jj}, 'r--','linewidth',2)
        xlabel('Decision Time'); ylabel('CDF')
        if runSimulations
            legend('Simulation', 'Analytic','location','best')
        else
            legend('Analytic','location','best')
        end
        xlim([0 max(aCDF_T{jj})]);
        ylim([0 1]);
        set(gca,'TickDir','out');
        set(gca,'Box','off');
        
        subplot(1,2,2);
        if runSimulations
            plot(simCDF_T_plus{jj}, simCDF_Y_plus{jj},'g','linewidth',2);
            hold on
            plot(simCDF_T_minus{jj}, simCDF_Y_minus{jj},'r','linewidth',2);
        end
        hold on
        plot(aCDF_T{jj},aCDF_Y_plus{jj}, 'g--','linewidth',2)
        plot(aCDF_T{jj},aCDF_Y_minus{jj}, 'r--','linewidth',2)
        xlabel('Decision Time'); ylabel('CDF')
        xlim([0 max(aCDF_T{jj})]);
        if runSimulations
            legend('Pos-Simulation','Neg-Simulation', 'Pos-Analytic', 'Neg-Analytic','location','best')
        else
            legend('Pos-Analytic','Neg-Analytic','location','best')
        end
        set(gca,'TickDir','out');
        set(gca,'Box','off');
        
        
        % Expected ER/DT (across uniform thresholds) - LEFT: expected DT (all), MID: expected ER (all), RIGHT: expected DT (conditional)
        figure
        subplot(1,3,1);
        hold on
        if runSimulations
            plot(thresh, simMeanRT,'k','linewidth',3)
        end
        plot(thresh, aRT,'r','linewidth',3)
        axis([.4 length(thresh)+0.6 min(union(aRT,simMeanRT))-0.1 max(union(aRT,simMeanRT))+0.1]);
        if length(thresh)>1
            xlim([min(thresh) max(thresh)]);
        else
            xlim([thresh-1 thresh+1]);
        end
        if runSimulations
            legend('Simulation', 'Analytic','location','best')
        else
            legend('Analytic','location','best')
        end
        if runSimulations
            scatter(thresh, simMeanRT,45,'k','filled')
        end
        scatter(thresh, aRT,45,'r','linewidth',2)
        xlabel('Threshold');
        ylabel('Expected Decision Time')
        set(gca,'TickDir','out');
        set(gca,'Box','off');
        
        subplot(1,3,2);
        hold on
        if runSimulations
            plot(thresh, simMeanER,'k','linewidth',3)
        end
        plot(thresh, aER,'r','linewidth',3)
        axis([.4 length(thresh)+0.6 min(union(aER,simMeanER))-0.05 max(union(aER,simMeanER))+0.05]);
        if length(thresh)>1
            xlim([min(thresh) max(thresh)]);
        else
            xlim([thresh-1 thresh+1]);
        end
        if runSimulations
            legend('Simulation', 'Analytic','location','best')
        else
            legend('Analytic','location','best')
        end
        if runSimulations
            scatter(thresh, simMeanER,45,'k','filled')
        end
        scatter(thresh, aER,45,'r','linewidth',2)
        xlabel('Threshold');
        ylabel('Error Rate')
        set(gca,'TickDir','out');
        set(gca,'Box','off');
        
        subplot(1,3,3);
        hold on
        if runSimulations
            plot(thresh, simMeanRT_plus,'g','linewidth',3)
            plot(thresh, simMeanRT_minus,'r','linewidth',3)
        end
        plot(thresh, aRT_plus,'g--','linewidth',3)
        plot(thresh, aRT_minus,'r--','linewidth',3)
        axis([.4 length(thresh)+0.6 min([aRT_plus,aRT_minus,simMeanRT_plus,simMeanRT_minus])-0.1 max([aRT_plus,aRT_minus,simMeanRT_plus,simMeanRT_minus])+0.1]);
        if length(thresh)>1
            xlim([min(thresh) max(thresh)]);
        else
            xlim([thresh-1 thresh+1]);
        end
        if runSimulations
            legend('Pos-Simulation','Neg-Simulation', 'Pos-Analytic','Neg-Analytic','location','best')
        else
            legend('Pos-Analytic','Neg-Analytic','location','best')
        end
        scatter(thresh, aRT_plus,45,'g','linewidth',2)
        scatter(thresh, aRT_minus,45,'r','linewidth',2)
        if runSimulations
            scatter(thresh, simMeanRT_plus,45,'g','filled')
            scatter(thresh, simMeanRT_minus,45,'r','filled')
        end
        xlabel('Threshold');
        ylabel('Expected Decision Time')
        set(gca,'TickDir','out');
        set(gca,'Box','off');

end
