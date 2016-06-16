clc
clear all
close all

% Vector of drift rates

a=0.15*ones(1,20);
%[0.1 0.2 0.05 0.3];


% Vector of diffusion rates

s=1.0*ones(1,20);
%[1 1.5 1.25 2];



% Discretization step for simulating DDM
step=0.005;


% Thresholds

threshold=linspace(3.0,0.01,20);
%2*[1 1.5 0.5 1.25];


% Support of initial condition and its density

x0=0; x0dist=1;

% Times at which  each stage starts

deadlines=linspace(0,5,20);
%[0 1 2 3];

% No of Monte Carlo runs

realizations=10000;


% Initialization

RT=zeros(1,realizations);
ER=zeros(1,realizations);

% Simulate the multistage DDM

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

RTplus=RT(ER==0);

RTminus=RT(ER==1);



[N,T] = hist(RT,100);

[Np, Tp]=hist(RTplus,100);

[Nm, Tm]=hist(RTminus,100);

tfinal= T(end);


[T_anal,Y_anal,Y_plus_anal,Y_minus_anal]=multistage_ddm_fpt_dist(a,s,threshold,x0,x0dist,deadlines,tfinal);



plot(T, cumsum(N)/realizations,'k','linewidth',2);

hold on

plot(T_anal,Y_anal, 'r--','linewidth',2)


xlabel('Decision Time'); ylabel('CDF')


legend('Simulation', 'Analytic','location','best')

figure

plot(Tp, cumsum(Np)/realizations,'k','linewidth',2);

hold on

plot(T_anal,Y_plus_anal, 'r--','linewidth',2)


xlabel('Correct Decision Time'); ylabel('CDF')


legend('Simulation', 'Analytic','location','best')


figure

plot(Tm, cumsum(Nm)/realizations,'k','linewidth',2);

hold on

plot(T_anal,Y_minus_anal, 'r--','linewidth',2)


xlabel('Erroneous Decision Time'); ylabel('CDF')


legend('Simulation', 'Analytic','location','best')

