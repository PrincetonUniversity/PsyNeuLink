clear, clc, close all

a = [.15 .15];
s = [.3 .3];
step = .005;
threshold = [.7 .7];
x0 = -.6;
x0dist = 1;
realizations = 1e2;
deadlines = [0 5];

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



chisq(RTplus,T_anal,Y_plus_anal,realizations)
chisq(RTminus,T_anal,Y_minus_anal,realizations)

ER = -1*(ER-1);

chisq_rt2002(RT,ER,a,s,threshold,x0,x0dist,deadlines,tfinal)