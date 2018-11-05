function [meanERs,meanRTs,meanDTs,condRTs,condVarRTs, condSkewRTs, allRRs,scaledX0] = ddmSimFRG(RVs,P,ddmPars,sepCDFs, flipChoiceProb,actualChoices,numNDDMsims)

import EVC.DDM.*;

% Created by A.Shenhav (with help from S.Feng) on 3/20/13
% Extended by S. Musslick (6-1-15) to compute parallelize computation for multiple drift rates and thresholds
% Last modified 6/19/13 (A.S.)
% ALL INPUTS OPTIONAL:
    % RVs = array of relative values (i.e., value difference between options under selection)
    % P [default 0.5] = initial condition (encoded as choice probability, with 0.5 = 50/50, 0 = 100% lower threshold, 1 = 100% upper threshold)
    % ddmPars = structure containing parameters of DDM (see below) other than initial conditions that were set in advance 
    % flipChoiceProb [default 0] = invert error rate?
    % allVpairs = for entering a series of value pairs to determine a series of drift rates (based on value difference) while retaining *overall* value as well
    % actualChoices = binary choices from actual expt (if generating simulated RTs conditioned on correct vs. error choice)
    % sepCDFs [default 0] = produce estimated cumul. dens. fxn for correct vs. error trials?
    % numNDDMsims [default 0] = how many simulations to run of "neural DDM" (a DDM implementation from Hare et al., 2011 that is almost identical to an LCAMechanism, meant for comparison to their models)
% OUTPUTS:
    % allERs = avg error rates corresponding to each drift rate tested (e.g., based on relative option values)
    % allDTs = avg decision times (i.e., first passage times; not including non-decision components) corresponding to each drift rate tested
    % allRRs = avg reward rates corresponding to each drift rate tested
    % allFinalRTs = avg total RT (including non-decision components) corresponding to each drift rate tested
    % NDDMs = output of simulated NDDM (e.g., total estimated activity) for each drift rate tested
    % scaledX0 = initial scaled/transformed relative to threshold (so that 0 is unbiased and +/-1 represent initial conditions at upper/lower threshold)
    % allFinalRTs_sepCDFs = pairs of final RT values corresponding to mean lower/upper threshold crossing times (i.e., error/correct)

%% argument check    

% check for vectorized parameters
numParams = max([length(RVs) length(P) length(ddmPars.T0) length(ddmPars.c) length(ddmPars.z)]);
if(numParams > 0 && length(RVs) == 1)
    RVs = repmat(RVs, 1, numParams);
elseif length(RVs) > 1 && length(RVs) ~= numParams;
    error('Drift rate vector doesn''t match parameter vector with most elements.');
end
if(numParams > 0 && length(P) == 1)
    P = repmat(P, 1, numParams);
elseif length(P) > 1 && length(P) ~= numParams;
    error('Starting point vector doesn''t match parameter vector with most elements.');
end
if(numParams > 0 && length(ddmPars.T0) == 1)
    ddmPars.T0 = repmat(ddmPars.T0, 1, numParams);
elseif length(ddmPars.T0) > 1 && length(ddmPars.T0) ~= numParams;
    error('T0 vector doesn''t match parameter vector with most elements.');
end
if(numParams > 0 && length(ddmPars.c) == 1)
    ddmPars.c = repmat(ddmPars.c, 1, numParams);
elseif length(ddmPars.c) > 1 && length(ddmPars.c) ~= numParams;
    error('Noise vector doesn''t match parameter vector with most elements.');
end
if(numParams > 0 && length(ddmPars.z) == 1)
    ddmPars.z = repmat(ddmPars.z, 1, numParams);
elseif length(ddmPars.z) > 1 && length(ddmPars.z) ~= numParams;
    error('Threshold vector doesn''t match parameter vector with most elements.');
end

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
% T0_adjustment = zeros(1,length(RVs));
% T0_adjustment = zeros(size(RVs));


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

myP = P;
myP(isneg) = Q(isneg);

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
Dtotal = D + T0 + Dp;

% Error rate        
ER = 1./(1 + min(1e12,exp(2*ZZ.*AA))) -...
        (1 - max(1e-12,exp(-2*x0.*AA))) ./...            
        (min(1e12,exp(2*ZZ.*AA)) - max(1e-12,exp(-2*ZZ.*AA)));       
% FAIL-SAFE (12/27/13)  
ER((isnan(ER) | isinf(ER)) & (AA<1e-6)) = 1-P((isnan(ER) | isinf(ER)) & (AA<1e-6));  %%% CHECK WHETHER THIS SHOULD BE 0.5 or P 


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

% calculate conditional RTs
if sepCDFs 

    x0_forpdf = (P-0.5)*2.*z; % ********** transform starting point to be centered at 0
    
    % drift a ...A
    % diffusion s ...c
    % threshold z ...z
    % starting point x0 ... x0_forpdf
    % non-decision time ... ddmPars.T0

    % fixing T0 to something arbitrarily small and adding later:
    [m_RTplus, m_RTminus, v_RTplus, v_RTminus, t_RTplus, t_RTminus] = ddm_metrics_cond_Mat(A,c,z,x0_forpdf);

    allDTs_sepCDFs = [m_RTplus; m_RTminus]; % mean decision time 
    condVarRTs = [v_RTplus; v_RTminus]; % variance of decision time
    condSkewRTs = [t_RTplus; t_RTminus]./(condVarRTs.^(1.5)); % divide third central moment by v_RT^1.5 to get skewness

    NER = nan;
    NDT = nan;
else
    condVarRTs = NaN;
    allDTs_sepCDFs = NaN;
    condSkewRTs = NaN;
end

% Reward rate
RR_traditional = (1-ER) ./ ...
    (DT + D + T0 + Dp.*ER); % + T0_adjustment
RR = abs(A) ./ ...
    (DT + D + T0 + Dp.*ER); % + T0_adjustment

meanDTs = DT;
meanERs = ER;
allRRs = RR_traditional;
allNDTs = NDT;
allNERs = NER;

meanDTs(A==0) = nan;
meanERs(A==0) = nan;
allRRs(A==0) = nan;
allNDTs(A==0) = nan;
allNERs(A==0) = nan;

if flipChoiceProb
    meanERs = 1-meanERs;
end




meanRTs = meanDTs + T0; % + T0_adjustment;

% in case we only have a T0 multidimensional array
if(numel(meanERs) == 1 && numel(meanRTs) > 1)
   meanERs = repmat(meanERs, size(meanRTs));
end

if sepCDFs
    condRTs = allDTs_sepCDFs + repmat(T0, 2, 1); % + [T0_adjustment,T0_adjustment];
    scaledX0 = x0_forpdf;
else
    condRTs = nan;
    scaledX0 = sign(A).*x0./ZZ;
end







%% JUNKYARD:

