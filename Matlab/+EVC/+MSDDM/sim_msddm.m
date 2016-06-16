function [RT,ER,RTplus,RTminus,tFinal] = sim_msddm(nSims,a,s,step,th,x0,x0dist,dl);
threshold = th;
realizations = nSims;
deadlines = dl;

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
tFinal = T(end);