function test_msfit
close all

a = .09;
z = .215;

realX = [a z];

s = .3;
x0 = 0;
x0dist = 1;
dt = .005;
dl = [0];
nSims = 5e3;

[rt,er,rtP,rtM,tFinal] = sim_msddm(nSims,a,s,dt,z,x0,x0dist,dl)
rtResp = -1*(er-1); % responses (>0 for top boundary)

%Let's go with fminsearch first...
opts = optimset('fminsearch');
opts = optimset(opts,'Display','iter');
opts = optimset(opts,'TolFun',1e-6);
opts = optimset(opts,'TolX',1e-8);
opts = optimset(opts,'MaxFunEvals',1000*5);
opts = optimset(opts,'MaxIter',500*5);

xInit(1) = .17; % Drift
xInit(2) = .08; % z

           
% Assuming knowledge of deadlines
inferredX = fminsearch(@(x) myobj1(x,rt,rtResp,tFinal),xInit,opts);
realX
inferredX

function val = myobj1(x,rt,rtResp,tFinal)
a = x(1);
z = x(2);

x0 = 0;
dl = [0];
s = .3;

val = chisq_rt2002(rt,rtResp,a,s,z,x0,1,dl,tFinal);
