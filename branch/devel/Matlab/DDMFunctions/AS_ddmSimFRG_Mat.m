function [allERs,allDTs,allRRs,allFinalRTs,NDDMs,scaledX0,allFinalRTs_sepCDFs] = AS_ddmSimFRG_Mat(RVs,P,ddmPars,flipChoiceProb,actualChoices,sepCDFs,numNDDMsims)

import EVC.DDM.*;

% Created by A.Shenhav (with help from S.Feng) on 3/20/13
% Last modified 6/19/13 (A.S.)
% ALL INPUTS OPTIONAL:
    % RVs = array of relative values (i.e., value difference between options under selection)
    % P [default 0.5] = initial condition (encoded as choice probability, with 0.5 = 50/50, 0 = 100% lower threshold, 1 = 100% upper threshold)
    % ddmPars = structure containing parameters of DDM (see below) other than initial conditions that were set in advance 
    % flipChoiceProb [default 0] = invert error rate?
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

%% argument check    
    
NDDMs = NaN;

if ~exist('ddmPars','var') || isempty(ddmPars)
    ddmPars = [];
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


if ~exist('flipChoiceProb','var') || isempty(flipChoiceProb)
    flipChoiceProb = 0; 
end

if ~exist('actualChoices','var') || isempty(actualChoices)
    actualChoices = nan(size(RVs)); 
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
if ~(exist('ddmPars','var') && isfield(ddmPars,'t0'))
    t0 = 0.45; % T_nd
else
    t0 = ddmPars.t0;
end

% The following 3 parameters allow NON-decision time to be modulated by overall value 
% (e.g., if response vigor is influenced directly by value, a la Niv et al., 2006/2007)
if ~(exist('ddmPars','var') && isfield(ddmPars,'t0_VmaxScale'))
    t0_VmaxScale = 0; % Scaling based on max option value
else
    t0_VmaxScale = ddmPars.t0_VmaxScale;
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

% Allows trial-by-trial modulation of t0 (e.g. based on overall value, but defaults to 0 modulation):
% t0_adjustment = zeros(1,length(RVs));
t0_adjustment = zeros(size(RVs));


%% DDM calculation

Afix = 1;  % fixed drift rate for initial condition set

% former loop start: for rvi = 1:length(RVs)
    
%%%%% THIS NEEDS TO BE ADJ SO THAT RVoffset is in original value units!! (currently in drift units instead!)

A = RVs.*RVscale + RVoffset;

isneg = A<0;
Ap = abs(A);

% Bounding Ap to avoid Ap==0
Ap = max(1e-5,Ap);     
 
 
% ADDING UPPER/LOWER BOUNDS        
P = min(1-1e-12,max(1e-12,P));

myP = repmat(P,size(A));
myP(isneg) = Q;

% Initial condition:
y0 = ((c.^2)./(2*Afix)) .* log(myP./(1-myP));  % fixed-drift

% optimize computation
if(numel(z) == 1)
    y0(abs(y0)>z) = sign(y0(abs(y0)>z))*z;
else
    if(numel(y0) == 1)
        y0 = repmat(y0, size(z));
    end
    if(size(z) == size(y0))
       y0(abs(y0)>z) = sign(y0(abs(y0)>z)).*z(abs(y0)>z);
    else
       error('Dimensions of z and y0 don''t match.'); 
    end
end

% Normalized initial condition:
x0 = y0./Ap;
% Normalized drift rate (a~):
AA = (Ap./c).^2;
% Normalized threshold (z~):
ZZ = z./Ap;
% Total time for non-decision components:
Dtotal = D + t0 + Dp;

% Error rate        
ER = 1./(1 + min(1e12,exp(2*ZZ.*AA))) -...
        (1 - max(1e-12,exp(-2*x0.*AA))) ./...            
        (min(1e12,exp(2*ZZ.*AA)) - max(1e-12,exp(-2*ZZ.*AA)));       
% FAIL-SAFE (12/27/13)  
ER((isnan(ER) | isinf(ER)) & (AA<1e-6)) = 1-P;  %%% CHECK WHETHER THIS SHOULD BE 0.5 or P 


% Decision time
      DT = ZZ.*tanh(ZZ.*AA) + ...
    ((2*ZZ.*(1-max(1e-12,exp(-2*x0.*AA)))) ./ ...
    (min(1e12,exp(2*ZZ.*AA)) - max(1e-12,exp(-2*ZZ.*AA))) - x0);        
% FAIL-SAFE (12/27/13)
DT((isnan(DT) | isinf(DT)) & (AA<1e-6)) = 1e12;


% % Error rate
% ER = 1./(1 + exp(2*ZZ.*AA)) -...
%     (1 - exp(-2*x0.*AA)) ./...
%     (exp(2*ZZ.*AA) - exp(-2*ZZ.*AA));
% 
% % Decision time
% DT = ZZ.*tanh(ZZ.*AA) + ...
%     ((2*ZZ.*(1-exp(-2*x0.*AA))) ./ ...
%     (exp(2*ZZ.*AA) - exp(-2*ZZ.*AA)) - x0);


% Recompute ER for negative drift rates
ER(isneg) = 1-ER(isneg);


% FAIL-SAFE:
DT(DT<0) = 0;


% Net error rate
NER = 1 ./ (1 + exp(2*ZZ.*AA));
% Net decision time
NDT =  ZZ.*tanh(ZZ.*AA) + ...
    ((1-2*myP)./(2*AA)) .* log(myP./(1-myP));

if sepCDFs 
    warning('sepCDFs for multidimensional arguments not implemented yet.');
%     tmpDT = DT;
%     tmpER = ER;
% 
% %         T = linspace(0,10,1200);
%     T = linspace(0,10,50); % **********
%     x0_forpdf = (P-0.5)*2*z;% **********
% 
%     % fixing t0 to something arbitrarily small and adding later:
%     tmpt0 = 0.01;  % Just needs to be >0
%     if ~isnan(actualChoices(rvi))
%         [p0,DT,ER] = ddmpdf(T,actualChoices(rvi),A,tmpt0,x0_forpdf,z,c);
%         DT = DT - tmpt0;
%     end
%     [p0_0,DT_1,ER_0] = ddmpdf(T,0,A,tmpt0,x0_forpdf,z,c); % lower threshold% ********** - USE ACTUAL t0! - ER = P(hitting lower)
%     DT_1 = DT_1 - tmpt0;
%     [p0_1,DT_0,ER_1] = ddmpdf(T,1,A,tmpt0,x0_forpdf,z,c);  % upper threshold% **********- ER = P(hitting upper)
%     DT_0 = DT_0 - tmpt0;
% 
%     allDTs_sepCDFs(rvi,1:2) = [DT_0,DT_1];
% 
%     ER = tmpER;
%     NER = nan;
%     NDT = nan;
end

% Reward rate
RR_traditional = (1-ER) ./ ...
    (DT + D + t0 + t0_adjustment + Dp.*ER);
RR = abs(A) ./ ...
    (DT + D + t0 + t0_adjustment + Dp.*ER);

allDTs = DT;
allERs = ER;
allRRs = RR_traditional;
allNDTs = NDT;
allNERs = NER;

allDTs(A==0) = nan;
allERs(A==0) = nan;
allRRs(A==0) = nan;
allNDTs(A==0) = nan;
allNERs(A==0) = nan;

if flipChoiceProb
    allERs = 1-allERs;
end




allFinalRTs = allDTs + t0 + t0_adjustment;

% in case we only have a t0 multidimensional array
if(numel(allERs) == 1 && numel(allFinalRTs) > 1)
   allERs = repmat(allERs, size(allFinalRTs));
end

if sepCDFs
    warning('sepCDFs for multidimensional arguments not implemented yet.');
%   allFinalRTs_sepCDFs = allDTs_sepCDFs + t0 + [t0_adjustment,t0_adjustment];
%   scaledX0 = x0_forpdf;
else
    allFinalRTs_sepCDFs = nan;
    scaledX0 = sign(A).*x0./ZZ;
end







%% JUNKYARD:

