function [x,prob,pnd,mean_dead,mgf_dead,sec_dead] = dist_deadline(a,s,x0, x0dist,deadline,z, theta) 


%Input: 
% a = drift rate 
% s = diffusion rate 
% x0 = support of initial condition (discretized) 
% x0dist = density of x0 at points in x0 
% deadline = change point 
% z= threshold 
% theta = moment generating function parameter


%Output: 
% x = Support set of distribution at deadline
% prob = density of X(deadline) conditioned on decision time greater than deadline 
% pnd= probability of no decision until deadline 
% mean_dead= mean value of X(deadline) 
% mgf_dead= moment generating function of X(deadline)
% sec_dead = second moment of X(deadline)


% We assume x0 has a continuous distribution (point mass is also fine)


% ensure x0 is a row vector

if length(x0(:,1))>1
    x0=x0';
end


if length(x0dist(:,1))>1
    x0dist=x0dist';
end



% No of discretization points in output distribution

N_out= 100;

% Support set for the density 

x= linspace(-z,z,N_out);
%-z:step_out:z;

% Discretization parameter for the output support

if length(x)>1
    step_out= x(2)-x(1);
else
    step_out=1;
end


% Terms in the series solution to the density

N=-5:5;


% Lebegue measure for the discretized input density 

if length(x0)>1
    step_in=x0(2)-x0(1);
else
    step_in=1;
end


% Initialization

prob=zeros(1,length(x));
X0=ones(length(N),1)*x0;
X0dist=ones(length(N),1)*x0dist;
N=N'*ones(size(x0));


% The following expression computes the density at deadline. The standard
% expression is for a given initial condition. Here expectation over the
% distribution of initial condition is taken.

for kk=1:length(x)
    
    y=x(kk);
    
    prob(kk) = sum(step_in*sum((exp(-(y-X0+4*N*z).^2/2/s^2/deadline)...
        -exp(-(2*z-y-X0+4*N*z).^2/2/s^2/deadline))/sqrt(2*pi*deadline*s^2).*exp(min(100,(-a^2*deadline+2*a*(y-X0))/2/s^2)).*X0dist));
    
end


%Probability of no decision until deadline computed by marginalization of
%above probability density

pnd=sum(prob)*step_out;


% Conditional density

prob=prob/pnd;

pnd=min(1,pnd);


% Mean and MGF at deadline

mean_dead=sum(x.*prob*step_out);

mgf_dead=sum(exp(min(100,theta*x)).*prob*step_out);

sec_dead=sum(x.^2.*prob*step_out);





