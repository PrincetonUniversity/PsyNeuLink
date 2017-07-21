function [x,prob,pnd,mean_dead,mgf_dead,sec_dead] = dist_deadline_VEC(a,s,x0, x0dist,deadline,z, theta) 


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

x = zeros(size(z,1), N_out);
for i = 1:size(z,1)
    x(i,:) = linspace(-z(i),z(i),N_out);
end
%-z:step_out:z;

% Discretization parameter for the output support

if size(x,2)>1
    step_out= x(:,2)-x(:,1);
else
    step_out=ones(size(x,1),1);
end


% Terms in the series solution to the density

N=-5:5;


% Lebegue measure for the discretized input density 

if size(x0,1)>1
    step_in=x0(2,:)-x0(1,:);
else
    step_in=ones(1,size(x0,2));
end


% Initialization

prob=zeros(size(x));
X0=ones(length(N),1)*x0;
X0dist=ones(length(N),1)*x0dist;
N=N'*ones(size(x0));


% The following expression computes the density at deadline. The standard
% expression is for a given initial condition. Here expectation over the
% distribution of initial condition is taken.

z_vec = repmat(z', size(X0,1),1);
a_vec = repmat(a', size(X0,1),1);
s_vec = repmat(s', size(X0,1),1);
deadline_vec = repmat(deadline', size(X0,1),1);

for kk=1:size(x,2)
    
    y=x(:,kk);
    y_vec = repmat(y', size(X0,1),1);
    
    %prob(:,kk) = sum(step_in.*sum((exp(-(y_vec-X0+4*N.*z_vec).^2/2./s_vec.^2./deadline_vec)...
    %    -exp(-(2*z_vec-y_vec-X0+4*N.*z_vec).^2/2./s_vec.^2./deadline_vec))./sqrt(2*pi*deadline_vec.*s_vec.^2).*repmat(exp(min([repmat(100, 1,size(X0,2));(-a_vec.^2.*deadline_vec+2*a_vec.*(y_vec-X0))/2./s_vec.^2])), size(X0dist,1),1).*X0dist),1);
    prob(:,kk) = sum(step_in.*sum((exp(-(y_vec-X0+4*N.*z_vec).^2/2./s_vec.^2./deadline_vec)...
        -exp(-(2*z_vec-y_vec-X0+4*N.*z_vec).^2/2./s_vec.^2./deadline_vec))./sqrt(2*pi*deadline_vec.*s_vec.^2).*exp(min(100,(-a_vec.^2.*deadline_vec+2*a_vec.*(y_vec-X0))/2./s_vec.^2)).*X0dist),1);
end


%Probability of no decision until deadline computed by marginalization of
%above probability density

pnd=sum(prob,2).*step_out;


% Conditional density

prob=prob./repmat(pnd, 1, size(prob,2));

pnd=min([ones(size(pnd,1), 1),pnd],[],2);


% Mean and MGF at deadline
step_out_vec = repmat(step_out,1,size(prob,2));

mean_dead=sum(x.*prob.*step_out_vec,2);

mgf_dead=sum(exp(min(100, repmat(theta,1,size(x,2)).*x)).*prob.*step_out_vec,2);

sec_dead=sum(x.^2.*prob.*step_out_vec,2);





