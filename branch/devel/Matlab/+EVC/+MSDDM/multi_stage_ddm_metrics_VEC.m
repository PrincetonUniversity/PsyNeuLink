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
stages=size(a, 2);
simulations = size(a, 1);

ER=zeros(simulations,stages);

RT=zeros(simulations,stages);

RTplus=zeros(simulations,stages);

RTminus=zeros(simulations,stages);

p_inst_react_plus=zeros(simulations,stages);

p_inst_plus=zeros(simulations,stages);

p_inst_minus=zeros(simulations,stages);

p_inst_react_minus=zeros(simulations,stages);


% Probability of no decision at the start of the current stage
weight_left=1;


% Probabilty of decision in the given stage

weight=zeros(simulations,stages);


for stage=1:stages-1
    
    
    % Computation of the Lebesgue measure for the initial condition density
    
    if size(x0,2)>1
        step=x0(:,2)-x0(:,1);
    else
        step=1;
    end
    
    % Mean and MGF for initial condition
    
    k= 2*a(:,stage)./s(:,stage).^2;
    
    x0_mean=sum(x0.*x0dist*step, 2);
    
    x0_mgf=sum (exp(min(100, -k.*x0)).*x0dist*step,2);
    
    x0_sec=sum(x0.^2.*x0dist*step, 2);
    
    % Threshold and deadline for the current stage
    
    z= thresholds(:,stage);
    
    deadline=deadlines(:,stage+1)-deadlines(:,stage);
    
    % Computation of the density, mean, MGF, and no decision probability at deadline.
    % The computed density is assigned as the density of initial condition for next
    % stage.
    
    x0_curr=x0;
    x0dist_curr=x0dist;
    
    step_curr=step;
    
    % CHECK HERE
    [x0, x0dist, pnd, mean_dead, mgf_dead, sec_dead] = EVC.MSDDM.dist_deadline_VEC(a(:,stage),s(:,stage),x0, x0dist,deadline,z, -k);
    
    
    if size(x0,2)>1
        step=x0(:,2)-x0(:,1);
    else
        step=1;
    end
    
    
    pnd=min(0.9999,pnd);

if isempty(stage)
stage = 1;    
end
    
    % Computation of the error rate and decision times conditioned on
    % decision in current stage.
    
    nzeroIdx = find(a(:,stage) ~= 0);
    zeroIdx = find(a(:,stage) == 0);
    
    if(~isempty(nzeroIdx))
        ER(nzeroIdx, stage) =  ((x0_mgf(nzeroIdx,:) - mgf_dead(nzeroIdx,:).*pnd(nzeroIdx,:))./(1-pnd(nzeroIdx,:)) ...
                                        -exp(min(100,-k(nzeroIdx,:).*z(nzeroIdx,:))))./(exp(min(100,k(nzeroIdx,:).*z(nzeroIdx,:)))-exp(min(100,-k(nzeroIdx,:).*z(nzeroIdx,:))));
        
        RT(nzeroIdx,stage) =  deadlines(nzeroIdx,stage)+((1-2*ER(nzeroIdx,stage)).*z(nzeroIdx,:) ...
            - (x0_mean(nzeroIdx,:) - (mean_dead(nzeroIdx,:) -a(nzeroIdx, stage).*deadline(nzeroIdx,:)).*pnd)./(1-pnd(nzeroIdx,:)))./a(nzeroIdx, stage);
        

        % Computation of Reaction time conditioned on a decision
        
        
        % error rate if there was no deadline for the current stage
        
        no_deadline_error(nzeroIdx,:) = (x0_mgf(nzeroIdx,:) -exp(min(100,-k(nzeroIdx,:).*z(nzeroIdx,:))))./(exp(min(100,k(nzeroIdx,:).*z(nzeroIdx,:)))- exp(min(100,-k(nzeroIdx,:).*z(nzeroIdx,:))));
        
        % error rate if there was no deadline for current stage and the
        % initial distribution of evidence was the distribution at deadline
        
        no_deadline_error_next(nzeroIdx,:) = (sum (exp(min(100,repmat(-k(nzeroIdx,:),1,size(x0,2)).*x0(nzeroIdx,:))).*x0dist(nzeroIdx,:).*repmat(step(nzeroIdx,:),1,size(x0dist,2)),2) ...
                                                -exp(min(100,-k(nzeroIdx,:).*z(nzeroIdx,:))))./(exp(min(100,k(nzeroIdx,:).*z(nzeroIdx,:)))- exp(min(100,-k(nzeroIdx,:).*z(nzeroIdx,:))));
        
        % computation of conditional decision time
        
        RT1= deadlines(nzeroIdx,stage).*(1-no_deadline_error(nzeroIdx,:))...
            +sum((exp(trun(k(nzeroIdx,:).*(z(nzeroIdx,:)-x0_curr(nzeroIdx,:))/2))./a(nzeroIdx,stage) ...
            .*(2.*z(nzeroIdx,:).*sinh(trun(k(nzeroIdx,:).*(z(nzeroIdx,:)+x0_curr(nzeroIdx,:))/2)).*cosh(trun(k(nzeroIdx,:).*z(nzeroIdx,:)))./(sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:)))).^2 ...
            - (z(nzeroIdx,:)+x0_curr(nzeroIdx,:)).*cosh(trun(k(nzeroIdx,:).*(z(nzeroIdx,:)+x0_curr(nzeroIdx,:))/2))./sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:))))).*x0dist_curr(nzeroIdx,:),2).*step_curr;
        
        RT2= deadlines(nzeroIdx,stage+1).*(1-no_deadline_error_next(nzeroIdx,:))...
            +sum((exp(trun(repmat(k(nzeroIdx,:),1,size(x0,2)).*(repmat(z(nzeroIdx,:),1,size(x0,2))-x0(nzeroIdx,:))/2)) ...
            ./repmat(a(nzeroIdx, stage),1,size(x0,2)) ...
            .*(2*repmat(z(nzeroIdx,:),1,size(x0,2)).*sinh(trun(repmat(k(nzeroIdx,:),1,size(x0,2)).*(repmat(z(nzeroIdx,:),1,size(x0,2))+x0(nzeroIdx,:))/2)) ...
            .*repmat(cosh(trun(k(nzeroIdx,:).*z(nzeroIdx,:)))./(sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:)))).^2,1,size(x0,2)) ...
            - (repmat(z(nzeroIdx,:),1,size(x0,2))+x0(nzeroIdx,:)).*cosh(trun(repmat(k(nzeroIdx,:),1,size(x0,2)) ...
            .*(repmat(z(nzeroIdx,:),1,size(x0,2))+x0(nzeroIdx,:))/2))./repmat(sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:))),1,size(x0,2)))).*x0dist,2) ...
            .*step(nzeroIdx,:);
        
        % expected decision time conditioned on the correct decision and
        % conditioned on decision time being smaller than deadline
        
        RTplus(nzeroIdx,stage)= (RT1(nzeroIdx,:) - RT2(nzeroIdx,:).*pnd(nzeroIdx,:))./(1-pnd(nzeroIdx,:));
        
        
        % Similar computation for expected decision time conditioned on
        % erroneous decision
        RT1= deadlines(nzeroIdx,stage).*no_deadline_error(nzeroIdx,:)...
            +sum((exp(trun(-k(nzeroIdx,:).*(z(nzeroIdx,:)+x0_curr(nzeroIdx,:))/2))./a(nzeroIdx,stage).*(2*z(nzeroIdx,:).*sinh(trun(k(nzeroIdx,:).*(z(nzeroIdx,:)-x0_curr(nzeroIdx,:))/2)).*cosh(trun(k(nzeroIdx,:).*z(nzeroIdx,:)))./(sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:)))).^2  ...
            - (z(nzeroIdx,:)-x0_curr(nzeroIdx,:)).*cosh(trun(k(nzeroIdx,:).*(z(nzeroIdx,:)-x0_curr)/2))./sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:))))).*x0dist_curr(nzeroIdx,:),2)*step_curr;
        
        RT2= deadlines(nzeroIdx,stage+1).*no_deadline_error_next(nzeroIdx,:)...
            +sum((exp(trun(-repmat(k(nzeroIdx,:),1,size(x0,2)).*(repmat(z(nzeroIdx,:),1,size(x0,2))+x0)/2))./repmat(a(nzeroIdx,stage),1,size(x0,2)) ...
        .*(2*repmat(z(nzeroIdx,:),1,size(x0,2)).*sinh(trun(repmat(k(nzeroIdx,:),1,size(x0,2)) ...
        .*(repmat(z(nzeroIdx,:),1,size(x0,2))-x0)/2)).*repmat(cosh(trun(k(nzeroIdx,:).*z(nzeroIdx,:))) ...
        ./(sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:)))).^2,1,size(x0,2)) ...
        - (repmat(z(nzeroIdx,:),1,size(x0,2))-x0).*cosh(trun(repmat(k(nzeroIdx,:),1,size(x0,2)) ...
        .*(repmat(z(nzeroIdx,:),1,size(x0,2))-x0)/2))./repmat(sinh(trun(k(nzeroIdx,:).*z(nzeroIdx,:))),1,size(x0,2)))).*x0dist,2).*step;
        
        RTminus(nzeroIdx,stage)= (RT1(nzeroIdx,:) - RT2(nzeroIdx,:).*pnd(nzeroIdx,:))./(1-pnd(nzeroIdx,:));
        
    end
    
    if(~isempty(zeroIdx))
        
        % same computation for the zero drift case
        
        ER(zeroIdx,stage) =  0.5*(1-(x0_mean(zeroIdx,:)-mean_dead(zeroIdx,:).*pnd(zeroIdx,:))/(z(zeroIdx,:).*(1-pnd(zeroIdx,:))));
        
        RT(zeroIdx,stage) = deadlines(zeroIdx,stage)+(z(zeroIdx,:).^2.*(1-pnd(zeroIdx,:))-x0_sec(zeroIdx,:)+(sec_dead(zeroIdx,:)-s(zeroIdx,stage).^2.*deadline(zeroIdx,:)).*pnd(zeroIdx,:))./(s(zeroIdx,stage).^2.*(1-pnd(zeroIdx,:)));
        
        no_deadline_error(zeroIdx,:) = (z(zeroIdx,:)-x0_mean(zeroIdx,:))./(2*z(zeroIdx,:));
        
        no_deadline_error_next(zeroIdx,:) = (z(zeroIdx,:)-sum(x0(zeroIdx,:).*x0dist(zeroIdx,:).*repmat(step(zeroIdx,:),1,size(x0,2)),2))./(2.*z(zeroIdx,:));
        
        RT1(zeroIdx,:)= deadlines(zeroIdx,stage).*(1-no_deadline_error(zeroIdx,:))...
            +sum((4*z(zeroIdx,:).^2/3./s(zeroIdx,stage).^2 -(z(zeroIdx,:)+x0_curr(zeroIdx,:)).^2/3./s(zeroIdx,stage).^2).*x0dist_curr(zeroIdx,:),2)*step_curr.*(1-no_deadline_error(zeroIdx,:));
        
        RT2(zeroIdx,:)= deadlines(zeroIdx,stage+1).*(1-no_deadline_error_next(zeroIdx,:))...
            +sum((repmat(4*z(zeroIdx,:).^2/3./s(zeroIdx,stage).^2,1,size(x0,2)) ...
            -(repmat(z(zeroIdx,:),1,size(x0,2))+x0(zeroIdx,:)).^2/3./repmat(s(zeroIdx,stage),1,size(x0,2)).^2).*x0dist(zeroIdx,:),2).*step(zeroIdx,:).*(1-no_deadline_error_next(zeroIdx,:));
        
        RTplus(zeroIdx,stage)= (RT1(zeroIdx,:) - RT2(zeroIdx,:).*pnd(zeroIdx,:))./(1-pnd(zeroIdx,:));
        
        
        RT1(zeroIdx,:)= deadlines(zeroIdx,stage).*(1-no_deadline_error)...
            +sum((4*z(zeroIdx,:).^2/3./s(zeroIdx,stage).^2 -(z(zeroIdx,:)-x0_curr(zeroIdx,:)).^2/3./s(zeroIdx,stage).^2).*x0dist_curr(zeroIdx,:),2)*step_curr.*no_deadline_error(zeroIdx,:);
        
        RT2(zeroIdx,:)= deadlines(stage+1).*(1-no_deadline_error_next(zeroIdx,:))...
            +sum((repmat(4*z(zeroIdx,:).^2/3./s(zeroIdx,stage).^2,1,size(x0,2)) ...
            -(repmat(z(zeroIdx,:),1,size(x0,2))-x0(zeroIdx,:)).^2/3./repmat(s(zeroIdx,stage),1,size(x0,2)).^2) ...
            .*x0dist(zeroIdx,:), 2).*step(zeroIdx,:).*no_deadline_error_next(zeroIdx,:);
        
        RTminus(zeroIdx,stage)= (RT1(zeroIdx,:) - RT2(zeroIdx,:).*pnd(zeroIdx,:))./(1-pnd(zeroIdx,:));
        
    end

    
    % Accounting for the change of threshold at boundary
    
    % if threshold decreases
    decrIdx = find(thresholds(:,stage+1) < thresholds(:,stage));
    incrIdx = find(thresholds(:,stage+1) >= thresholds(:,stage));
    
    
    if (~isempty(decrIdx))
        
        thresholds_vec = repmat(thresholds(decrIdx,stage+1), 1, size(x0,2));
        
        % probability of instantaneous correct decision
        p_inst_react_plus(decrIdx,stage+1)= sum((x0(decrIdx,:) > thresholds_vec).*x0dist,2).*step;
        % probability of instantaneous errorneous decision
        p_inst_react_minus(decrIdx,stage+1)= sum((x0 < -thresholds_vec).*x0dist,2).*step;
        % remove the part of support outside new boundaries
        x0dist(x0(decrIdx,:) > thresholds_vec)=[];
        x0dist(decrIdx, x0(decrIdx,:) < -thresholds_vec)=[];
        
        x0(x0 > thresholds(stage+1))=[];
        x0(x0 < -thresholds(stage+1))=[];
        % normalize the distribution
        x0dist=x0dist./repmat((sum(x0dist,2).*step),1,size(x0dist,2));
    end
    if (~isempty(incrIdx))
        % if the threshold increases
        % increase the support of x0dist
        x0dist(incrIdx,:)=[x0dist(incrIdx,1:end-1) 0*(x0(incrIdx,end):x0(incrIdx,2)-x0(incrIdx,1): thresholds(incrIdx,stage+1))];
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