function [dpdf,meanRT,phit] = ddmpdf(T,x,A,T0,x0,z,sigma)

import EVC.DDM.*;

if x<0 || x>1
   error; 
end

s = sigma;
nT = length(T);
dpdf = zeros(1,nT);
errtol = 1e-4;
for k = 1:nT
    t = T(k)-T0;
    if t < 0
        dpdf(k) = 0;
    else
        dpdf(k) = wfpt(t, x, A/s, 2*z/s, (x0+z)/s, errtol);
    end
end



flipped = 0;
if A<0
    x0 = -x0;
    A = -A;
    flipped = 1;
end

zz = z/A;
aa = (A/s)^2;
xx = x0/A;

ER = 1 / (1 + exp(2*zz*aa)) - ...
     (1 - exp(-2*xx*aa))/(exp(2*zz*aa)-exp(-2*zz*aa));

if x == 0
    phit = ER;
else
    phit = 1-ER;
end

if flipped
    phit = 1 - phit;
end

% Compute expected value, careful to normalize
dt = T(2)-T(1);
I = sum(trapz(dpdf)*dt);
cpdf = (1/I) .* dpdf;
meanRT = sum(trapz(T.*cpdf)*dt);