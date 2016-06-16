% 6 params: 2 drifts, 1 deadline, 2 thresholds, x0, T0

function val = obj2a1d2z1xT0(x,rt,rtResp)
a = [x(1) x(2)];
dl = [0 x(3)];
z = [x(4) x(5)];
x0 = x(6); 
T0 = x(7);
% Fixed (experimentally set) values
s = [1 1];
tFinal = max(rt) + 3;
val = rt2002(rt,rtResp,a,s,z,x0,1,dl,tFinal,T0);
