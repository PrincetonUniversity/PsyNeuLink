function [allERs,allDTs,allRRs,allFinalRTs,NDDMs,scaledX0,allFinalRTs_sepCDFs] = AS_ddmSimFRG(RVs,P,plotFigs,ddmPars,flipChoiceProb,closeFigs,allVpairs,actualChoices,sepCDFs,numNDDMsims)

import EVC.DDM.*;

% Created by A.Shenhav (with help from S.Feng) on 3/20/13
% Last modified 6/19/13 (A.S.)
% ALL INPUTS OPTIONAL:
    % RVs = array of relative values (i.e., value difference between options under selection)
    % P [default 0.5] = initial condition (encoded as choice probability, with 0.5 = 50/50, 0 = 100% lower threshold, 1 = 100% upper threshold)
    % plotFigs [default 0] = generate plots described at bottom?
    % ddmPars = structure containing parameters of DDM (see below) other than initial conditions that were set in advance 
    % flipChoiceProb [default 0] = invert error rate?
    % closeFigs [default 0] = close currently open figs
    % allVpairs = for entering a series of value pairs to determine a series of drift rates (based on value difference) while retaining *overall* value as well
    % actualChoices = binary choices from actual expt (if generating simulated RTs conditioned on correct vs. error choice)
    % sepCDFs [default 0] = produce estimated cumul. dens. fxn for correct vs. error trials?
    % numNDDMsims [default 0] = how many simulations to run of "neural DDM" (a DDM implementation from Hare et al., 2011 that is almost identical to an LCA, meant for comparison to their models)
% OUTPUTS:
    % allERs = avg error rates corresponding to each drift rate tested (e.g., based on relative option values)
    % allDTs = avg decision times (i.e., first passage times; not including non-decision components) corresponding to each drift rate tested
    % allRRs = avg reward rates corresponding to each drift rate tested
    % allFinalRTs = avg total RT (including non-decision components) corresponding to each drift rate tested
    % NDDMs = output of simulated NDDM (e.g., total estimated activity) for each drift rate tested
    % scaledX0 = initial scaled/transformed relative to threshold (so that 0 is unbiased and +/-1 represent initial conditions at upper/lower threshold)
    % allFinalRTs_sepCDFs = pairs of final RT values corresponding to mean lower/upper threshold crossing times (i.e., error/correct)

NDDMs = NaN;
    
if ~(exist('closeFigs','var'))
    closeFigs = 0; 
end

if ~exist('ddmPars','var') || isempty(ddmPars)
    ddmPars = [];
end

if closeFigs
    close all;
end

% Relative values being tested (i.e drift rates):
if ~(exist('RVs','var'))
    RVs = -0.3:0.01:.3;
end

if ~(exist('P','var'))
    P = 0.01; 
end

% Stimulus bias: P = 1: 100 percent top (positive) response, P=0: negative, P = 0.5: unbiased
Q = 1-P;

if ~exist('plotFigs','var')
    plotFigs = 0; 
end

if ~exist('flipChoiceProb','var') || isempty(flipChoiceProb)
    flipChoiceProb = 0; 
end

if ~exist('allVpairs','var') || isempty(allVpairs)
    allVpairs = nan(size(RVs));
    usingVpairs = 0;  % IGNORING RVs and calculating these based on Vpairs?
else
    usingVpairs = 1;  % IGNORING RVs and calculating these based on Vpairs? 
end

if ~exist('actualChoices','var') || isempty(actualChoices)
    actualChoices = nan(1,length(RVs)); 
end
if ~exist('sepCDFs','var') || isempty(sepCDFs)
    sepCDFs = 0; 
end
if ~exist('numNDDMsims','var') || isempty(numNDDMsims)
    numNDDMsims = 0; 
%     numNDDMsims = 1000;
end



if ~(exist('ddmPars','var') && isfield(ddmPars,'D'))
    D = 0; % ITI
else
    D = ddmPars.D;    
end
if ~(exist('ddmPars','var') && isfield(ddmPars,'Dp'))
    Dp = 0; % penalty delay
else
    Dp = ddmPars.Dp;    
end
if ~(exist('ddmPars','var') && isfield(ddmPars,'T0'))
    T0 = 0.45; % T_nd
else
    T0 = ddmPars.T0;    
end

% The following 3 parameters allow NON-decision time to be modulated by overall value 
% (e.g., if response vigor is influenced directly by value, a la Niv et al., 2006/2007)
if ~(exist('ddmPars','var') && isfield(ddmPars,'T0_VmaxScale'))
    T0_VmaxScale = 0; % Scaling based on max option value
else
    T0_VmaxScale = ddmPars.T0_VmaxScale;    
end
if ~(exist('ddmPars','var') && isfield(ddmPars,'V1scale'))
    V1scale = 1.0; % Scaling based on option value 1
else
    V1scale = ddmPars.V1scale;    
end
if ~(exist('ddmPars','var') && isfield(ddmPars,'V2scale'))
    V2scale = 1.0; % Scaling based on option value 2
else
    V2scale = ddmPars.V2scale;    
end


if ~(exist('ddmPars','var') && isfield(ddmPars,'z'))
    z = 0.75; % [+/-] threshold
else
    z = ddmPars.z;    
end
if ~(exist('ddmPars','var') && isfield(ddmPars,'c'))
    c = 0.5; % noise coefficient
else
    c = ddmPars.c;    
end


% These define how the relative values are transformed into drift rate:
if ~(exist('ddmPars','var') && isfield(ddmPars,'RVoffset'))
    RVoffset = 0; % Shift RVs (e.g. to adjust for miscalculated relative cost of one side vs. another)
else
    RVoffset = ddmPars.RVoffset;    
end

if ~(exist('ddmPars','var') && isfield(ddmPars,'RVscale'))
    RVscale = 1; % Scale RV by this to arrive at drift rate (A) 
else
    RVscale = ddmPars.RVscale;    
end


% Allows trial-by-trial modulation of T0 (e.g. based on overall value, but defaults to 0 modulation):
T0_adjustment = zeros(1,length(RVs));


Afix = 1;  % fixed drift rate for initial condition set

% % % if numNDDMsims>0
% % %     NDDMinh = 0.2;
% % %     NDDMnoise = c; % 0 +- 0.035
% % %     NDDMscaleA = RVscale; % 0.009 +- 0.005
% % %     
% % %     dT = 0.20;
% % %     T = 10.0;
% % %     
% % %     NDDMs.NDDMrts = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMmiss = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMchoices = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMcumTotalActivity = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMcumEnergy = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMavgTotalActivity = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMavgEnergy = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMmaxTotalActivity = nan(length(RVs),numNDDMsims);
% % %     NDDMs.NDDMmaxEnergy = nan(length(RVs),numNDDMsims);
% % %     
% % % end



% Looping over all relative values (can convert to vector math):
for rvi = 1:length(RVs)
    
    %%%%% THIS NEEDS TO BE ADJ SO THAT RVoffset is in original value units!! (currently in drift units instead!)
    if ~usingVpairs
        % Drift rate (keep track of sign), shifted and scaled appropriately
% %         A = (RVs(rvi)+RVoffset)*RVscale;
        A = RVs(rvi)*RVscale + RVoffset;
    else
% %         A = ((allVpairs(rvi,2)*V2scale - allVpairs(rvi,1)*V1scale)+RVoffset) * RVscale;
        A = (((allVpairs(rvi,2)*V2scale - allVpairs(rvi,1)*V1scale)) * RVscale)+RVoffset;
        T0_adjustment(rvi) = max(allVpairs(rvi,2)*V2scale,allVpairs(rvi,1)*V1scale)*T0_VmaxScale;
    end
    
    isneg = A<0;
    Ap = abs(A);
    
    if isneg
        myP = Q;
    else
        myP = P;
    end
    
    
    % Initial condition:
    y0 = ((c^2)/(2*Afix)) * log(myP/(1-myP));  % fixed-drift
    if abs(y0)>z
        y0 = sign(y0)*z;
    end
    
    % Normalized initial condition:
    x0 = y0/Ap;
    % Normalized drift rate (a~):
    AA = (Ap/c)^2;
    % Normalized threshold (z~):
    ZZ = z/Ap;
    % Total time for non-decision components:
    Dtotal = D + T0 + Dp;
    
    % Error rate
    ER = 1/(1 + exp(2*ZZ*AA)) -...
        (1 - exp(-2*x0*AA)) /...
        (exp(2*ZZ*AA) - exp(-2*ZZ*AA));
    
    % Decision time
    DT = ZZ*tanh(ZZ*AA) + ...
        ((2*ZZ*(1-exp(-2*x0*AA))) / ...
        (exp(2*ZZ*AA) - exp(-2*ZZ*AA)) - x0);
    
    % Recompute ER for negative drift rates
    if isneg, ER = 1-ER;    end
    
    
    % FAIL-SAFE:
    DT = max(DT,0);
    
    
    % Net error rate
    NER = 1 / (1 + exp(2*ZZ*AA));
    % Net decision time
    NDT =  ZZ*tanh(ZZ*AA) + ...
        ((1-2*myP)/(2*AA)) * log(myP/(1-myP));

    if sepCDFs  
        tmpDT = DT;
        tmpER = ER;
        
%         T = linspace(0,10,1200);
        T = linspace(0,10,50); % **********
        x0_forpdf = (P-0.5)*2*z;% **********

        % fixing T0 to something arbitrarily small and adding later:
        tmpT0 = 0.01;  % Just needs to be >0        
        if ~isnan(actualChoices(rvi))
            [p0,DT,ER] = ddmpdf(T,actualChoices(rvi),A,tmpT0,x0_forpdf,z,c);
            DT = DT - tmpT0;
        end
        [p0_0,DT_1,ER_0] = ddmpdf(T,0,A,tmpT0,x0_forpdf,z,c); % lower threshold% ********** - USE ACTUAL T0! - ER = P(hitting lower)
        DT_1 = DT_1 - tmpT0;
        [p0_1,DT_0,ER_1] = ddmpdf(T,1,A,tmpT0,x0_forpdf,z,c);  % upper threshold% **********- ER = P(hitting upper)
        DT_0 = DT_0 - tmpT0;

        allDTs_sepCDFs(rvi,1:2) = [DT_0,DT_1];

        ER = tmpER;
        NER = nan;
        NDT = nan;
    end
        
    % Reward rate
    RR_traditional = (1-ER) / ...
        (DT + D + T0 + T0_adjustment(rvi) + Dp*ER);
    RR = abs(A) / ...
        (DT + D + T0 + T0_adjustment(rvi) + Dp*ER);
    
        
    if A~=0 %AA~=0
        allDTs(rvi) = DT;
        allERs(rvi) = ER;
        allRRs(rvi) = RR_traditional;
        allNDTs(rvi) = NDT;
        allNERs(rvi) = NER;
    else % avoiding 0 for now
        allDTs(rvi) = nan;
        allERs(rvi) = nan;
        allRRs(rvi) = nan;
        allNDTs(rvi) = nan;
        allNERs(rvi) = nan;
    end
    
    
    
end


if flipChoiceProb
    allERs = 1-allERs;
end




allFinalRTs = allDTs + T0 + T0_adjustment;

if sepCDFs
    allFinalRTs_sepCDFs = allDTs_sepCDFs + T0 + [T0_adjustment,T0_adjustment];
    scaledX0 = x0_forpdf;
else
    allFinalRTs_sepCDFs = nan;
    scaledX0 = sign(A)*x0/ZZ;
end







%% JUNKYARD:

% % if plotFigs
% %     
% %     % dt by relative value:
% %     figure
% %     hold on;
% %     scatter(RVs,allDTs);
% %     xlabel('Relative Values');
% %     ylabel('Decision Time');
% %     axis([RVs(1) RVs(end) nanmin(allDTs)*0.95 nanmax(allDTs)*1.05]);
% %     set(gca,'FontSize',14)
% %     saveas(gcf,['Sim_DT_v_RV_Bias',num2str(P),'_Shift_',num2str(mean(RVs),'%.2f'),'_Flipped_',num2str(flipChoiceProb),'.pdf'],'pdf')
% %     
% %     % er by relative value:
% %     figure
% %     hold on;
% %     scatter(RVs,allERs);
% %     xlabel('Relative Values');
% %     ylabel('Error Rate');
% %     axis([RVs(1) RVs(end) 0 1]);
% %     set(gca,'FontSize',14)
% %     saveas(gcf,['Sim_ER_v_RV_Bias',num2str(P),'_Shift_',num2str(mean(RVs),'%.2f'),'_Flipped_',num2str(flipChoiceProb),'.pdf'],'pdf')
% % 
% %     figure
% %     hold on;
% %     scatter(RVs,allRRs);
% %     axis([RVs(1) RVs(end) nanmin(allRRs)*0.95 nanmax(allRRs)*1.05]);
% %     xlabel('Relative Values');
% %     ylabel('Reward Rate');
% %     set(gca,'FontSize',14)
% %     saveas(gcf,['Sim_RR_v_RV_Bias',num2str(P),'_Shift_',num2str(mean(RVs),'%.2f'),'_Flipped_',num2str(flipChoiceProb),'.pdf'],'pdf')
% % 
% % end

% % if numNDDMsims>0        
        
% %         for NDDMiter = 1:numNDDMsims
% %             a{1} = nan(1,T*dT);
% %             a{2} = nan(1,T*dT);
% %             
% %             % Set up in parallel:
% %             t = 0;
% %             tind = 1;
% %             a{1}(tind) = max(0,(P-0.5)*2*z);
% %             a{2}(tind) = max(0,-1*(P-0.5)*2*z);
% %             while t<T
% %                 t = t+dT;
% %                 tind = tind+1;
% %                 
% %                 a{1}(tind) = max(0,a{1}(tind-1) - NDDMinh*a{2}(tind-1) + A*NDDMscaleA + randn(1,1)*NDDMnoise);
% %                 a{2}(tind) = max(0,a{2}(tind-1) - NDDMinh*a{1}(tind-1) - A*NDDMscaleA + randn(1,1)*NDDMnoise);
% %                 
% %                 if a{1}(tind)>=z || a{2}(tind)>=z
% %                     NDDMs.NDDMmiss(rvi,NDDMiter) = 0;
% %                     
% %                     NDDMs.NDDMrts(rvi,NDDMiter) = t + T0 + T0_adjustment(rvi);
% %                     NDDMs.NDDMchoices(rvi,NDDMiter) = a{1}(tind)>a{2}(tind);  % A1 chosen?
% %                     NDDMs.NDDMcumTotalActivity(rvi,NDDMiter) = nansum(a{1}) + nansum(a{2});
% %                     NDDMs.NDDMcumEnergy(rvi,NDDMiter) = nansum(NDDMinh*prod([a{1},a{2}]));
% %                     NDDMs.NDDMavgTotalActivity(rvi,NDDMiter) = NDDMs.NDDMcumTotalActivity(rvi,NDDMiter)/tind;
% %                     NDDMs.NDDMavgEnergy(rvi,NDDMiter) = NDDMs.NDDMcumEnergy(rvi,NDDMiter)/tind;
% %                     NDDMs.NDDMmaxTotalActivity(rvi,NDDMiter) = nanmax(a{1} + a{2});
% %                     NDDMs.NDDMmaxEnergy(rvi,NDDMiter) = nanmax(NDDMinh*prod([a{1},a{2}]));
% %                     
% %                     break;
% %                 end
% %             end
% %             if isnan(NDDMs.NDDMmiss(rvi,NDDMiter))
% %                 NDDMs.NDDMmiss(rvi,NDDMiter) = 1;
% %             end
% %         end
% %         
% %         NDDMs.sum.meanMiss(rvi) = nanmean(NDDMs.NDDMmiss(rvi,:));
% %         NDDMs.sum.meanRT(rvi) = nanmean(NDDMs.NDDMrts(rvi,:));
% %         NDDMs.sum.choiceProb(rvi) = nanmean(NDDMs.NDDMchoices(rvi,:));
% %         NDDMs.sum.meanTotalActivity(rvi) = nanmean(NDDMs.NDDMcumTotalActivity(rvi,:));
% %         NDDMs.sum.meanTotalEnergy(rvi) = nanmean(NDDMs.NDDMcumEnergy(rvi,:));
% %         NDDMs.sum.meanAvgActivity(rvi) = nanmean(NDDMs.NDDMavgTotalActivity(rvi,:));
% %         NDDMs.sum.meanAvgEnergy(rvi) = nanmean(NDDMs.NDDMavgEnergy(rvi,:));
% %     elseif rvi==1
% %         NDDMs = nan;
% %     end

    %             Arev = (-1*RVs(rvi)-RVoffset)*RVscale;
%             Arev = ((allVpairs(rvi,1)*V1scale - allVpairs(rvi,2)*V2scale)-RVoffset) * RVscale;

%         ApRev = abs(Arev);


            %    y0 = ((c^2)/(2*A)) * log(P/(1-P));  % drift dependent

        %        disp('Rounded')
        %         AArev = (ApRev/c)^2;


%                 a{1}(tind) = max(0,a{1}(tind-1) - NDDMinh*dT*a{2}(tind-1) + A*NDDMscaleA*dT + randn(1,1)*NDDMnoise*sqrt(dT));
%                 a{2}(tind) = max(0,a{2}(tind-1) - NDDMinh*dT*a{1}(tind-1) - A*NDDMscaleA*dT + randn(1,1)*NDDMnoise*sqrt(dT));

        %         if round(tmpER*1000) ~= round(ER*1000)
%             keyboard;
%         end
            

        %         [p1,RT1,myP1] = ddmpdf(T,1,A,0,y0,z,c); % upper boundary
        
% % %         dtx = T(2)-T(1);
% % %         P0 = sum(trapz(p0)*dtx);
        %         P1 = sum(trapz(p1)*dtx);
        
        %         [p0,RT0,myP0] = ddmpdf(T,0,A,T0,x0,z,sigma); % lower boundary


    
    % % % ndt by relative value:
    % % figure
    % % scatter(RVs,allNDTs);
    % % xlabel('relative values');
    % % ylabel('net decision time');
    % %
    % % % ner by relative value:
    % % figure
    % % scatter(RVs,allNERs);
    % % xlabel('relative values');
    % % ylabel('net error rate');
    % %
    % rr by relative value:

