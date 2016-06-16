clc
clear all;
import EVC.MSDDM.*;

%% simple1

a = [-0.4, 1, 0.5];
s = [0.5 0.6 0.5];
deadlines = [0 0.5 1];
thresholds = [2 1 2];
x0 = 0;
x0dist = 1;

%% dist_deadline inputs
theta = -k;
a = a(:,stage);
s = s(:,stage);


%% simple 2

a = [0.5, 1, 0.5];
s = [0.5 0.5 0.5];
deadlines = [0 0.7 1];
thresholds = [1.5 1 2];
x0 = 0;
x0dist = 1;

%% vectorized

a = [-0.4, 1 0.5; ...
      0.5, 1 0.5];
  
s = [0.5 0.6 0.5; ...
     0.5 0.5 0.5];
 
deadlines = [0 0.5 1; ...
             0 0.7 1];
         
thresholds = [2 1 2; ...
             1.5 1 2];

x0 = [0; ...
      0];
x0dist = [1; ...
          1];

%% call simple
import EVC.MSDDM.*;
disp('---');
[mean_RT, mean_ER, mean_RT_plus, mean_RT_minus]= ...
    multi_stage_ddm_metrics(a ,s, deadlines, thresholds, x0, x0dist);
mean_RT
mean_ER