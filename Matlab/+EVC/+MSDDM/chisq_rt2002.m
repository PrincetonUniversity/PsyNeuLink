% rtResp:  >0 for correct, <=0 for incorrect
function val = chisq_rt2002(rtData, rtResp, a,s,th,x0,x0dist, ...
                                  dl,tFinal)

[tArray,~,yPlus,yMinus] = multistage_ddm_fpt_dist(...
    a,s,th,x0,x0dist,dl,tFinal);

% From vanila Chi-sq from Ratcliffe Tuerlinckx 2002.  
nTotalTrials = length(rtData);
val1 = chisq(rtData(rtResp>0),tArray,yPlus,nTotalTrials);
val2 = chisq(rtData(rtResp<=0),tArray,yMinus,nTotalTrials);
val = val1+val2;





