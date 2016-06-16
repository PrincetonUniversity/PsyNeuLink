% rtResp:  >0 for correct, <=0 for incorrect
function [val,df] = rt2002(rtData, rtResp, a,s,th,x0,x0dist, ...
                                  dl,tFinal,T0)
if nargin < 10
    T0 = 0;
end
[tArray,~,yPlus,yMinus] = multistage_ddm_fpt_dist(...
    a,s,th,x0,x0dist,dl,tFinal);
dt = tArray(3)-tArray(2);
nShift = round(T0/dt);
tmpt = (tArray(end)+(1:nShift)*dt);
tArray = [tArray tmpt];
yPlus = [zeros(1,nShift) yPlus];
yMinus = [zeros(1,nShift) yMinus];

% Chi-sq from Ratcliffe Tuerlinckx 2002.  
nTotalTrials = length(rtData);
[val1,df1,q1] = chisq(rtData(rtResp>0),tArray,yPlus,nTotalTrials);
[val2,df2,q2] = chisq(rtData(rtResp<=0),tArray,yMinus,nTotalTrials);
val = val1+val2;

q1 = [0 q1(1:end-1)];
q2 = [0 q2(1:end-1)];

df = [df1'; df2'];

% figure(1)
% plot(q1,df1,'r-*')
% hold on
% plot(q2,df2,'b-*')
% hold off
% drawnow

