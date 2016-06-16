function test_msfit_2stage
close all

a = [.04 .08];
z = [.215 .215];

realX = [a(1) a(2) z(1)];

s = [.3 .3];
x0 = 0;
dl = [0 1];

x0dist = 1;
dt = .005;
nSims = 5e3;

[rt,er,rtP,rtM,tFinal] = sim_msddm(nSims,a,s,dt,z,x0,x0dist,dl)
rtResp = -1*(er-1); % responses (>0 for top boundary)

%Let's go with fminsearch first...
opts = optimset('fminsearch');
opts = optimset(opts,'Display','iter');
opts = optimset(opts,'TolFun',1e-10);
opts = optimset(opts,'TolX',1e-10);
opts = optimset(opts,'MaxFunEvals',1000*5);
opts = optimset(opts,'MaxIter',500*5);

%xInit = realX;
xInit(1) = .1; % Drift 1
xInit(2) = .1; % Drift 2
xInit(3) = .08; % z

           
% Assuming knowledge of deadlines
inferredX = fminsearch(@(x) myobj1(x,rt,rtResp,tFinal),xInit,opts);
realX
inferredX

function val = myobj1(x,rt,rtResp,tFinal)
a = [x(1) x(2)];
z = [x(3) x(3)];

s = [.3 .3];
x0 = 0;
dl = [0 1];

val = chisq_rt2002(rt,rtResp,a,s,z,x0,1,dl,tFinal);
