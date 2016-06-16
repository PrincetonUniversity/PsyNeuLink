clc;
clear all;
        
import Simulations.*;


LB = [0 -4 0.5 0.3 0.05]; % params: cost param1, cost param2, default threshold, ddm efficiency, automatic step duration
UB = [8 0  2.5 4.0 0.3];

bestRes = [];
bestVal = 1e16;

for (i = 1:10)
    
init = rand(1,length(LB)).*(UB-LB)+LB; % random initialization within the bounds

% finding the minimum of the EVC simulation
[res,val] = fmincon(@(x) Simulations.MSDDM_GrattonSimple_wrapper(x(1),x(2),x(3),x(4),x(5)), ...
            init,[],[],[],[],LB,UB,[],...
            optimset('maxfunevals',5000,'maxiter',100,'GradObj','off','DerivativeCheck',...
            'off','LargeScale','off','Algorithm','active-set','Hessian','off'));
        
bestVal = min(bestVal, val);
if(bestVal == val)
    bestRes = res;
    disp('**************************got better');
end

end

disp(bestVal);  
disp(bestRes);
        
        %%
        clc;
        clear all;
        
        import Simulations.*;
        MSDDM_GrattonSimple_wrapper(4, -2, 1, 0.5, 0.3)