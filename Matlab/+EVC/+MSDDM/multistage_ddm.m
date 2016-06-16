clc
clear all
close all

% Vector of drift rates

a=[0.1 0.2 0.1 0.3];

%a=[-0.62602 -1.8781];


% Vector of diffusion rates

s=[1 1.5 1.25 2];

%s=[0.015575  0.022027];

% Discretization step for simulating DDM
step=0.001;


% Thresholds

%thresh=0.5:0.25:4;

thresh=(0.5:0.5:2);
thresh = [1 1 1 1];
%thresh=0.83976;

% Support of initial condition and its density

%x0=-0.2; x0dist=1;

x0=0; x0dist=1;

% Times at which  each stage starts

deadlines=[0 2 4 6];

%deadlines=[0 0.080117];


% No of Monte Carlo runs

realizations=1000;

% Initialization

mean_RT=zeros(size(thresh));
mean_ER=zeros(size(thresh));
mean_RT_plus=zeros(size(thresh));
mean_RT_minus=zeros(size(thresh));

aRT=zeros(size(thresh));
aER=zeros(size(thresh));
aRT_plus=zeros(size(thresh));
aRT_minus=zeros(size(thresh));

thresh=thresh'*[1 1.5 1.25 1.5];


tsim = 0;
tcomp = 0;

for jj=1:length(thresh)
    
    threshold=thresh(jj,:);
    
    RT=zeros(1,realizations);
    ER=zeros(1,realizations);
    
    % Simulate the multistage DDM
    tic
    for N=1:realizations

        t=0;
        
        x(1)=x0;
        
        l=1;
        
        stop=0;
        
        
        while stop==0
            
            stage=find(deadlines<=t,1,'last');
            
            x(l+1)= x(l) +a(stage)*step + s(stage)*randn*sqrt(step);
            t=t+step;
            l=l+1;
            
            if (x(l)>=threshold(stage) || x(l)<=-threshold(stage))
                stop=1;
                RT(N)=t;
                ER(N)=(x(l)<=-threshold(stage));
            end
            
            
        end
    end
    tt = toc;
    tsim = tsim +tt;

    % Mean decision time and error rate from Monte Carlo simulation
    mean_RT(jj)=mean(RT);
    
    mean_ER(jj)=mean(ER);
    
    RTplus=RT(ER==1);
    RTminus=RT(ER==0);
    
    mean_RT_plus(jj)=mean(RTplus);
    
    mean_RT_minus(jj)=mean(RTminus);
        
    % Analytical decision time and error rate
    tic
    [aRT(jj), aER(jj), aRT_plus(jj), aRT_minus(jj)]= multi_stage_ddm_metrics(a ,s, deadlines, threshold, x0, x0dist);
    tt = toc;
    tcomp = tcomp + tt;
end


tsim
tcomp



figure

plot(thresh(:,1), mean_RT,'k','linewidth',3)


xlabel('Threshold'); ylabel('Expected Decision Time')

hold on

plot(thresh(:,1), aRT,'r--','linewidth',3)

legend('Simulation', 'Analytic','location','best')


% hgsave('./figs/expected-decision-time.fig')
%
% hgsave('./figs/expected-decision-time.eps')



figure

plot(thresh(:,1), mean_ER,'k','linewidth',3)

hold on

plot(thresh(:,1), aER,'r--','linewidth',3)

xlabel('Threshold'); ylabel('Error Rate')

legend('Simulation', 'Analytic','location','best')



% hgsave('./figs/error-rate.fig')
%
% hgsave('./figs/error-rate.eps')



figure

plot(thresh(:,1), mean_RT_plus,'k','linewidth',3)


xlabel('Threshold'); ylabel('Expected (Correct) Decision Time')

hold on

plot(thresh(:,1), aRT_plus,'r--','linewidth',3)

legend('Simulation', 'Analytic','location','best')




figure

plot(thresh(:,1), mean_RT_minus,'k','linewidth',3)


xlabel('Threshold'); ylabel('Expected (Incorrect) Decision Time')

hold on

plot(thresh(:,1), aRT_minus,'r--','linewidth',3)

legend('Simulation', 'Analytic','location','best')

