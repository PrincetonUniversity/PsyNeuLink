clear, close all
%% Simulation parameters

%% Run msddm code, pick a set of fitted values for display
a = [.2 .6];
s = [1 1];        
th = [1.5 2.0];
x0dist = 1;        
dl = [0 1.5];
tFinal = 13;
x0 = -.1;
%T0 = .1;
disp('Forming rt cdfs')
[tArray,y,yPlus,yMinus] = multistage_ddm_fpt_dist(...
    a,s,th,x0,x0dist,dl,tFinal);
disp('Done')
%plot(tArray,y)

% Remove duplicated values from yPlus and yMinus
[yP,IP] = unique(yPlus/yPlus(end));
% IP = setdiff(IP,find(diff(yP/yP(end))==0)+1);
% yP = yPlus(IP);
[yM,IM] = unique(yMinus/yMinus(end));
% IM = setdiff(IM,find(diff(yM/yM(end))==0)-1);
% yM = yMinus(IM);
%% Draw n samples
nSamp = 10000;
rnd('default')
disp(['Drawing ' num2str(nSamp) ' samples'])
rt = nan(nSamp,1);
resp = rt;
for k = 1:nSamp
    if rand<yPlus(end)
        rt(k) = interp1(yPlus(IP)/yPlus(end),tArray(IP),rand);
        resp(k) = 1;
    else
        rt(k) = interp1(yM/yM(end),tArray(IM),rand,'nearest');
        resp(k) = -1;
    end
end

hist(rt(resp>0),50)

%% Fit like nobody's bizz
%% Genetic algorithm options
lb = [-.2 -.2 1 1 1 -1  0]; %d1 d2 dl z1 z2 x0 T0
ub = [3.0 3.0 2 3 3  1  0];
nVars = length(lb);
opts = gaoptimset('UseParallel','always','Display','diagnose',...
                  'Generations',100*nVars,...
                  'PopulationSize',10*nVars);

% impose abs(x0) \leq z1
conA = [0 0 0 -1 0 1 0;
        0 0 0 -1 0 -1 0];
conB = [0;
        0];


nTrials = length(rt);
% Solve the sucker
[recoveredParameters,finalChiSq, ...
 exitFlags] =...
    ga(@(x) obj2a1d2z1xT0(x,rt,resp), length(lb), conA,conB,...
       [],[], lb,ub,[],opts);

recoveredParameters

save('fit_sim.mat','recoveredParameters','nTrials','finalChiSq', ...
     'exitFlags','rt','resp')
