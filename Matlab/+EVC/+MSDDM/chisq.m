% Compute Chi sq for one set of (correct or incorrect) RTs
% QUANTILE WEIGHTS HERE
function [val, df,q] = chisq(rtData,tArray,ddmcdf,nTotalTrials)
nTrials = length(rtData);

% qBins = [.1 .2 .2 .2 .2 .1];
% qBins = [.1 .1 .1 .1 .1 .1];
qBins = [.05 .1*ones(1,9) .05];
cpv = cumsum(qBins);
q = quantile(rtData,cpv);
q(end) = inf; % Matlab is lame
qcData = nTrials*qBins;

% checking work...
% qcData = histc(rtData,[-inf q]);
% qcData(end) = [];

%% Expected # of trials in each quantile
[cPs] = interp1(tArray,ddmcdf,q);
cPs(end) = ddmcdf(end);
qcExp = diff([0 cPs]);
qcExp = qcExp * nTotalTrials/sum(qcExp) * ddmcdf(end);
qcExp = max(qcExp, 1e-10*ones(1,length(qBins)));

df = (qcData-qcExp)./sqrt(qcExp);
val = norm(df).^2;

