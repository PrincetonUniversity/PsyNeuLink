function [mean_RT, mean_ER, mean_RT_plus, mean_RT_minus]=multi_stage_ddm_metrics(a ,s, deadlines, thresholds, x0, x0dist)

% Input:
% a = vector of drift rates at each stage
% s = vector of diffusion rates at each stage
% deadlines = vector of times when stages start. First entry should be 0.
% thresholds = vector of thresholds at each stage.
% x0 = support of initial condition. Equals the initial condition in
% the deterministic case
% x0dist = density of x0. Equals 1 in the deterministic case

% Output
% mean_RT = mean decision time
% mean_ER = error rate
% mean_RT_plus = mean decision time conditioned on correct decision
% mean_RT_minus= mean decision time conditioned on erroneous decision 

% Initialization
stages=length(a);

ER=zeros(1,stages);

RT=zeros(1,stages);

RTplus=zeros(1,stages);

RTminus=zeros(1,stages);

p_inst_react_plus=zeros(1,stages);

p_inst_plus=zeros(1,stages);

p_inst_minus=zeros(1,stages);

p_inst_react_minus=zeros(1,stages);


% Probability of no decision at the start of the current stage
weight_left=1;


% Probabilty of decision in the given stage

weight=zeros(1,stages);


for stage=1:stages-1
    
    
    % Computation of the Lebesgue measure for the initial condition density
    
    if length(x0)>1
        step=x0(2)-x0(1);
    else
        step=1;
    end
    
    % Mean and MGF for initial condition
    
    k= 2*a(stage)/s(stage)^2;
    
    x0_mean=sum(x0.*x0dist*step);
    
    x0_mgf=sum (exp(min(100,-k*x0)).*x0dist*step);
    
    x0_sec=sum(x0.^2.*x0dist*step);
    
    % Threshold and deadline for the current stage
    
    z= thresholds(stage);
    
    deadline=deadlines(stage+1)-deadlines(stage);
    
    % Computation of the density, mean, MGF, and no decision probability at deadline.
    % The computed density is assigned as the density of initial condition for next
    % stage.
    
    x0_curr=x0;
    x0dist_curr=x0dist;
    
    step_curr=step;
    
    [x0, x0dist, pnd, mean_dead, mgf_dead, sec_dead] = EVC.MSDDM.dist_deadline(a(stage),s(stage),x0, x0dist,deadline,z, -k);
    
    
    if length(x0)>1
        step=x0(2)-x0(1);
    else
        step=1;
    end
    
    
    pnd=min(0.9999,pnd);

if isempty(stage)
stage = 1;    
end
    
    % Computation of the error rate and decision times conditioned on
    % decision in current stage.
    
    if a(stage)~=0
        
        ER(stage) =  ((x0_mgf - mgf_dead*pnd)/(1-pnd)-exp(min(100,-k*z)))./(exp(min(100,k*z))-exp(min(100,-k*z)));
        
        RT(stage) =  deadlines(stage)+((1-2*ER(stage)).*z ...
            - (x0_mean - (mean_dead -a(stage)*deadline)*pnd)/(1-pnd))/a(stage);
        

        % Computation of Reaction time conditioned on a decision
        
        
        % error rate if there was no deadline for the current stage
        
        no_deadline_error = (x0_mgf -exp(min(100,-k*z)))./(exp(min(100,k*z))- exp(min(100,-k*z)));
        
        % error rate if there was no deadline for current stage and the
        % initial distribution of evidence was the distribution at deadline
        
        no_deadline_error_next = (sum (exp(min(100,-k*x0)).*x0dist*step)-exp(min(100,-k*z)))./(exp(min(100,k*z))- exp(min(100,-k*z)));
        
        % computation of conditional decision time
        
        RT1= deadlines(stage)*(1-no_deadline_error)...
            +sum((exp(trun(k*(z-x0_curr)/2))/a(stage).*(2*z*sinh(trun(k*(z+x0_curr)/2))*cosh(trun(k*z))/(sinh(trun(k*z)))^2 - (z+x0_curr).*cosh(trun(k*(z+x0_curr)/2))/sinh(trun(k*z)))).*x0dist_curr)*step_curr;
        
        RT2= deadlines(stage+1)*(1-no_deadline_error_next)...
            +sum((exp(trun(k*(z-x0)/2))/a(stage).*(2*z*sinh(trun(k*(z+x0)/2))*cosh(trun(k*z))/(sinh(trun(k*z)))^2 - (z+x0).*cosh(trun(k*(z+x0)/2))/sinh(trun(k*z)))).*x0dist)*step;
        
        % expected decision time conditioned on the correct decision and
        % conditioned on decision time being smaller than deadline
        
        RTplus(stage)= (RT1 - RT2*pnd)/(1-pnd);
        
        
        % Similar computation for expected decision time conditioned on
        % erroneous decision
        RT1= deadlines(stage)*no_deadline_error...
            +sum((exp(trun(-k*(z+x0_curr)/2))/a(stage).*(2*z*sinh(trun(k*(z-x0_curr)/2))*cosh(trun(k*z))/(sinh(trun(k*z)))^2 - (z-x0_curr).*cosh(trun(k*(z-x0_curr)/2))/sinh(trun(k*z)))).*x0dist_curr)*step_curr;
        
        RT2= deadlines(stage+1)*no_deadline_error_next...
            +sum((exp(trun(-k*(z+x0)/2))/a(stage).*(2*z*sinh(trun(k*(z-x0)/2))*cosh(trun(k*z))/(sinh(trun(k*z)))^2 - (z-x0).*cosh(trun(k*(z-x0)/2))/sinh(trun(k*z)))).*x0dist)*step;
        
        RTminus(stage)= (RT1 - RT2*pnd)/(1-pnd);
        
    else
        
        % same computation for the zero drift case
        
        ER(stage) =  0.5*(1-(x0_mean -mean_dead*pnd)/(z*(1-pnd)));
        
        RT(stage) = deadlines(stage)+(z^2*(1-pnd)-x0_sec+(sec_dead-s(stage)^2*deadline)*pnd)/(s(stage)^2*(1-pnd));
        
        no_deadline_error = (z-x0_mean)./(2*z);
        
        no_deadline_error_next = (z-sum(x0.*x0dist*step))./(2*z);
        
        RT1= deadlines(stage)*(1-no_deadline_error)...
            +sum((4*z^2/3/s(stage)^2 -(z+x0_curr).^2/3/s(stage)^2).*x0dist_curr)*step_curr.*(1-no_deadline_error);
        
        RT2= deadlines(stage+1)*(1-no_deadline_error_next)...
            +sum((4*z^2/3/s(stage)^2 -(z+x0).^2/3/s(stage)^2).*x0dist)*step*(1-no_deadline_error_next);
        
        RTplus(stage)= (RT1 - RT2*pnd)/(1-pnd);
        
        
        RT1= deadlines(stage)*(1-no_deadline_error)...
            +sum((4*z^2/3/s(stage)^2 -(z-x0_curr).^2/3/s(stage)^2).*x0dist_curr)*step_curr.*no_deadline_error;
        
        RT2= deadlines(stage+1)*(1-no_deadline_error_next)...
            +sum((4*z^2/3/s(stage)^2 -(z-x0).^2/3/s(stage)^2).*x0dist)*step*no_deadline_error_next;
        
        RTminus(stage)= (RT1 - RT2*pnd)/(1-pnd);
        
        
    end
    
    % Accounting for the change of threshold at boundary
    
    % if threshold decreases
    if thresholds(stage+1)< thresholds(stage)
        % probability of instantaneous correct decision
        p_inst_react_plus(stage+1)= sum((x0 > thresholds(stage+1)).*x0dist)*step;
        % probability of instantaneous errorneous decision
        p_inst_react_minus(stage+1)= sum((x0 < -thresholds(stage+1)).*x0dist)*step;
        % remove the part of support outside new boundaries
        x0dist(x0 > thresholds(stage+1))=[];
        x0dist(x0 < -thresholds(stage+1))=[];
        x0(x0 > thresholds(stage+1))=[];
        x0(x0 < -thresholds(stage+1))=[];
        % normalize the distribution
        x0dist=x0dist/(sum(x0dist)*step);
    else
        % if the threshold increases
        % increase the support of x0dist
        x0dist=[x0dist(1:end-1) 0*(x0(end):x0(2)-x0(1): thresholds(stage+1))];
        x0dist=[0*(-thresholds(stage+1):x0(2)-x0(1): x0(1)) x0dist(2:end)];
        x0=[x0(1:end-1) x0(end):x0(2)-x0(1): thresholds(stage+1)];
        x0=[-thresholds(stage+1):x0(2)-x0(1): x0(1) x0(2:end)];
    end

    
    % Probability of decision in current stage
    
    weight(stage)=weight_left*(1-pnd);
    
    % Probability of no decision by the end of the current stage
    
    weight_left=weight_left*pnd;
    
    p_inst_plus(stage+1)=weight_left*p_inst_react_plus(stage+1);
    p_inst_minus(stage+1)=weight_left*p_inst_react_minus(stage+1);
    
    weight_left=weight_left - p_inst_plus(stage+1)- p_inst_minus(stage+1);
end

% The last stage

if length(x0)>1
    step=x0(2)-x0(1);
else
    step=1;
end

% Probability of entering the last stage

weight(stages)=weight_left;

% Mean and MGF of the evidence at start of last stage

k= 2*a(stages)/s(stages)^2;

x0_mean=sum(x0.*x0dist*step);

x0_mgf=sum (exp(min(100,-k*x0)).*x0dist*step);

x0_sec=sum(x0.^2.*x0dist*step);

z= thresholds(stages);


% Computation of error rate and decision time conditioned on decision in
% the last stage

if a(stages)~=0
    
    ER(stages)=(x0_mgf -exp(min(100,-k*z)))./(exp(min(100,k*z))- exp(min(100,-k*z)));
    
    RT(stages)=  deadlines(stages)+ ((1-2*ER(stages)).*z- x0_mean)/a(stages);
    
    RTplus(stages)= deadlines(stages)*(1-ER(stages))...
        +sum((exp(trun(k*(z-x0)/2))/a(stages).*(2*z*sinh(trun(k*(z+x0)/2))*cosh(trun(k*z))/(sinh(trun(k*z)))^2 - (z+x0).*cosh(trun(k*(z+x0)/2))/sinh(trun(k*z)))).*x0dist)*step;
    
    RTminus(stages) = deadlines(stages)*ER(stages)...
        +sum((exp(trun(-k*(z+x0)/2))/a(stages).*(2*z*sinh(trun(k*(z-x0)/2))*cosh(trun(k*z))/(sinh(trun(k*z)))^2 - (z-x0).*cosh(trun(k*(z-x0)/2))/sinh(trun(k*z)))).*x0dist)*step;
    
else
    
    ER(stages)= 0.5*(1-x0_mean/z);
    
    RT(stages)= deadlines(stages)+ (z^2 - x0_sec)/(s(stages)^2);
    
    RTplus(stages) = deadlines(stages)*(1-ER(stages))+ sum((4*z^2/3/s(stage)^2 - (z+x0).^2/3/s(stage)^2).*x0dist*step)*(1-ER(stages));
    
    RTminus(stages) = deadlines(stages)*ER(stages)+sum((4*z^2/3/s(stage)^2 - (z- x0).^2/3/s(stage)^2).*x0dist*step)*ER(stages);
    
end


% Weighted sum of decision times and error rates in each stage

% probability of instantaneous decisions at deadlines

p_inst_react=p_inst_minus+ p_inst_plus;

% aggregate decisions in multiple stages and at the boundaries

mean_RT=sum(weight.*RT) + deadlines*p_inst_react';

mean_ER=sum(weight.*ER) + sum(p_inst_minus);

mean_RT_plus=(sum(weight.*RTplus)+deadlines*p_inst_plus')/(1-mean_ER);

mean_RT_minus=(sum(weight.*RTminus)+deadlines*p_inst_minus')/mean_ER;




function y =trun(x)

y = max(-100,min(100,x));